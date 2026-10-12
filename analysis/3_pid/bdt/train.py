#!/usr/bin/env python3
"""LightGBM hadron identification (pi-, p, K0_L) from shower-shape variables.

What changed with respect to the first version of this script
-------------------------------------------------------------
* **SMOTE leaked into the hyper-parameter search.** The training set was
  oversampled *before* ``GridSearchCV``, so each validation fold contained
  synthetic points interpolated from events in the corresponding training folds,
  and the cross-validation score was optimistic. Resampling now happens inside
  an :class:`imblearn.pipeline.Pipeline`, hence inside every fold.
* **The time smearing was degenerate.** ``smear_column_gaussian`` re-seeded a
  generator from the same value on each call, so ``tMin``, ``tMax`` and
  ``tSpread`` received the *identical* noise vector instead of independent noise.
  It now goes through :mod:`sdhcal.smearing`, which draws one generator per
  column and refuses to smear a column twice.
* **The run log became unreadable.** Adding ``val_size`` to the configuration
  shifted every later column by one, so the hyper-parameters of the model used
  downstream (``lgbm_model_9.joblib``) could no longer be recovered. Logging is
  now done by :class:`sdhcal.runlog.RunLogger`, which widens the header, records
  the feature list and the smearing configuration, and writes a JSON record per
  run.
* **The probability figure assumed exactly three classes** in a fixed order and
  mislabelled its panels; it is now generated for every pair of classes present.
* **Hard-coded ``/gridgroup`` paths** are resolved through :mod:`sdhcal.paths`.
* **A single split gave a single number.** ``--seeds n`` repeats the experiment
  and reports mean +- standard deviation.
* The saved artifact now carries the feature list and the label mapping next to
  the model, so inference can check that it is being fed the right columns in the
  right order instead of trusting the call site.

Examples
--------
::

    python train.py
    python train.py --sigma-time 0.1 --seeds 5
    python train.py --no-timing --grid-search
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sdhcal import artifacts, paths, provenance, rootio, runlog, splits
from sdhcal import config as sdconfig
from sdhcal import metrics as sdmetrics  # noqa: F401  (kept for energy-binned studies)
from sdhcal.pid_bdt import (
    CLASS_NAMES,  # noqa: F401  historical import surface
    Config,
    build_classifier,  # noqa: F401  historical import surface
    evaluate,
    search_hyperparameters,  # noqa: F401  historical import surface
    set_global_seeds,
    train,
)
from sdhcal.smearing import smear_time_columns

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)8s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("pid_lgbm")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_dataset(cfg: Config) -> pd.DataFrame:
    """Read every particle file, label the events and smear the timing once."""
    needed = list(dict.fromkeys([*cfg.feature_cols, cfg.pdg_col]))
    # eventNumber is not a feature; it is read only to identify the held-out
    # events, and it is dropped again below so it cannot reach the model.
    read = [*needed, "eventNumber"] if cfg.exclude_holdout else needed
    frames = []
    for particle in cfg.particles:
        path = paths.dataset("params", particle)
        frames.append(rootio.load_trees(path, cfg.tree_name, read))
    df = rootio.clean(pd.concat(frames, ignore_index=True), columns=needed)

    if cfg.exclude_holdout:
        holdout = rootio.load_trees(paths.dataset("holdout"), cfg.tree_name, [cfg.pdg_col, "eventNumber"])
        df, dropped = splits.exclude_keys(df, splits.event_keys(holdout))
        logger.info("Excluded %d held-out events; %d remain for training", dropped, len(df))
        if not dropped:
            raise RuntimeError(
                "--exclude-holdout removed no event: the held-out file shares no event key "
                "with the parameter files, so one of the two was not built from the other."
            )
        df = df.drop(columns="eventNumber")

    label_of = {pdg: index for index, pdg in enumerate(cfg.target_pdg)}
    df = df[df[cfg.pdg_col].isin(label_of)].copy()
    if df.empty:
        raise RuntimeError(f"no event with PDG among {cfg.target_pdg}")
    df["label"] = df[cfg.pdg_col].map(label_of).astype(int)

    counts = np.bincount(df["label"], minlength=len(cfg.particles))
    logger.info(
        "Class balance: %s",
        ", ".join(f"{p}={c}" for p, c in zip(cfg.particles, counts, strict=True)),
    )
    return smear_time_columns(df, cfg.smearing())


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_training_curve(evals: dict, out: Path) -> None:
    metric = next(iter(next(iter(evals.values())).keys()))
    train_curve = evals.get("train", {}).get(metric, [])
    valid_curve = evals.get("valid", {}).get(metric, [])
    if not train_curve or not valid_curve:
        logger.warning("No training curve available")
        return
    plt.figure(figsize=(6, 4))
    plt.plot(train_curve, label="train")
    plt.plot(valid_curve, "--", label="validation")
    plt.xlabel("Boosting iteration")
    plt.ylabel(metric.replace("_", " "))
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, labels: Sequence[str], out: Path) -> None:
    """Row-normalised confusion matrix, annotated with percentage and raw count."""
    totals = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm.astype(float), np.maximum(totals, 1)) * 100.0

    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 2.4, 1.4 * len(labels) + 2.0))
    image = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{pct[i, j]:.1f} %\n({cm[i, j]:d})",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if pct[i, j] > 55 else "black",
            )
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("PID confusion matrix (rows normalised)")
    fig.colorbar(image, ax=ax, label="fraction of true class [%]")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_probability_histograms(proba: np.ndarray, y_true: np.ndarray, cfg: Config, out: Path) -> None:
    """One panel per class: the score of that class, for every true class.

    The previous version hard-coded three panels for specific class pairs and
    mislabelled two of them; this generalises to any number of classes.
    """
    n_classes = len(cfg.particles)
    fig, axes = plt.subplots(1, n_classes, figsize=(4.6 * n_classes, 3.8), squeeze=False)
    bins = np.linspace(0.0, 1.0, 51)
    for index, axis in enumerate(axes[0]):
        for true_label, name in enumerate(cfg.labels):
            axis.hist(
                proba[y_true == true_label, index],
                bins=bins,
                histtype="step",
                linewidth=1.5,
                density=True,
                label=f"true {name}",
            )
        axis.set_xlabel(f"P(predicted = {cfg.labels[index]})")
        axis.set_ylabel("normalised events")
        axis.set_yscale("log")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)
    fig.suptitle("Classifier scores by true class")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_feature_importance(model, features: Sequence[str], out: Path) -> pd.DataFrame:
    importance = pd.DataFrame(
        {"feature": list(features), "gain": model.booster_.feature_importance(importance_type="gain")}
    ).sort_values("gain", ascending=False)
    plt.figure(figsize=(7, max(3.0, 0.32 * len(features))))
    plt.barh(importance["feature"], importance["gain"])
    plt.gca().invert_yaxis()
    plt.xlabel("Total gain")
    plt.title("LightGBM feature importance")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return importance


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
def run_single_seed(
    cfg: Config,
    seed: int,
    log: runlog.RunLogger,
    resolved_config: dict | None = None,
) -> dict[str, float]:
    set_global_seeds(seed)
    seeded = replace(cfg, random_state=seed)

    df = load_dataset(seeded)
    x = df[list(seeded.feature_cols)].astype(np.float32)
    y = df["label"].to_numpy(dtype=int)

    split = splits.split_indices(
        len(df), test_size=seeded.test_size, val_size=seeded.val_size, seed=seed, stratify=y
    )
    logger.info("Split sizes: %s", split.sizes())

    model, best = train(x.iloc[split.train], y[split.train], x.iloc[split.val], y[split.val], seeded, seed)
    results, cm, proba, y_pred = evaluate(model, x.iloc[split.test], y[split.test], seeded)
    results["seed"] = seed
    if best:
        results["best_params"] = ";".join(f"{k}={v}" for k, v in best.items())

    tag = f"{log.run_id}_seed{seed}"
    model_path = log.path("models", f"lgbm_pid_{tag}.joblib")
    artifact_config = (
        sdconfig.update_resolved_values(resolved_config, random_state=seed)
        if resolved_config is not None
        else seeded.loggable()
    )
    artifacts.save_bundle(
        model_path,
        {
            "model": model,
            "features": list(seeded.feature_cols),
            "particles": list(seeded.particles),
            "label_to_pdg": dict(enumerate(seeded.target_pdg)),
            "smearing": seeded.smearing().as_dict(),
        },
        kind="pid",
        model_family="lightgbm",
        preprocessing={"smearing": seeded.smearing().as_dict()},
        training_counts={
            "train_before_resampling": int(len(split.train)),
            "validation": int(len(split.val)),
            "test": int(len(split.test)),
        },
        configuration=artifact_config,
        provenance=provenance.environment_fingerprint(),
    )

    plot_training_curve(model.evals_result_, log.path("plots", f"training_curve_{tag}.png"))
    plot_confusion_matrix(cm, seeded.labels, log.path("plots", f"confusion_matrix_{tag}.png"))
    plot_probability_histograms(proba, y[split.test], seeded, log.path("plots", f"prob_hists_{tag}.png"))
    importance = plot_feature_importance(
        model, list(seeded.feature_cols), log.path("plots", f"feature_importance_{tag}.png")
    )
    importance.to_csv(log.path("tables", f"feature_importance_{tag}.csv"), index=False)

    # Test-set features and scores, for the SHAP and per-energy studies.
    arrays = log.path("arrays", f"test_and_scores_{tag}.npz")
    np.savez_compressed(
        arrays,
        x_test=x.iloc[split.test].to_numpy(),
        y_test=y[split.test],
        y_pred=y_pred,
        proba=proba,
        test_index=split.test,
        features=np.array(list(seeded.feature_cols)),
        confusion_matrix=cm,
    )
    log.log_artifacts(**{f"model_seed{seed}": model_path, f"arrays_seed{seed}": arrays})
    return results


def run(cfg: Config, n_seeds: int, resolved_config: dict | None = None) -> dict[str, float]:
    out_dir = paths.output_dir() / "pid_lgbm"
    log = runlog.RunLogger(out_dir, name="pid")
    log.start(cfg.loggable(), features=list(cfg.feature_cols))
    if resolved_config is not None:
        resolved_path = sdconfig.write_resolved_config(
            log.path("runs", f"pid_{log.run_id}_resolved_config.json"),
            resolved_config,
        )
        log.log_artifacts(resolved_config=resolved_path)
    logger.info(
        "=== PID run %d | %d seed(s) | timing in features: %s ===", log.run_id, n_seeds, cfg.uses_timing()
    )

    per_seed = [run_single_seed(cfg, cfg.random_state + i, log, resolved_config) for i in range(n_seeds)]
    pd.DataFrame(per_seed).to_csv(log.path("tables", f"per_seed_{log.run_id}.csv"), index=False)

    summary: dict[str, float] = {"n_seeds": n_seeds}
    numeric = [k for k, v in per_seed[0].items() if isinstance(v, (int, float))]
    for key in numeric:
        values = np.array([r[key] for r in per_seed], dtype=float)
        summary[key] = float(values.mean())
        summary[f"{key}_spread"] = float(values.std(ddof=1)) if n_seeds > 1 else float("nan")

    log.log_metrics(**summary)
    log.log_comment(
        f"accuracy={summary['accuracy']:.4f}+-{summary['accuracy_spread']:.4f} "
        f"balanced={summary['balanced_accuracy']:.4f} smear={cfg.smear_sigma_ns} "
        f"features={len(cfg.feature_cols)}"
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--particles",
        nargs="+",
        default=list(paths.active_particles()),
        help="particles to separate; the order fixes the integer labels",
    )
    p.add_argument("--features", nargs="+", default=None, help="override the feature list")
    p.add_argument(
        "--exclude-holdout",
        action="store_true",
        help="drop the held-out events before splitting, so the coupling study evaluates "
        "on events this model never saw",
    )
    p.add_argument(
        "--sigma-time",
        type=float,
        default=0.1,
        metavar="NS",
        help="width of the Gaussian time smearing in ns (default 0.1)",
    )
    p.add_argument(
        "--no-timing",
        action="store_true",
        help="drop every time variable from the features, for the timing-vs-no-timing comparison",
    )
    p.add_argument("--no-smote", action="store_true", help="use LightGBM class weights instead of SMOTE")
    p.add_argument("--seeds", type=int, default=1, metavar="N", help="number of independent splits")
    p.add_argument(
        "--grid-search", action="store_true", help="search hyper-parameters (resampling inside folds)"
    )
    p.add_argument("--config", type=Path, default=None, help="optional versioned JSON configuration")
    return p


def resolve_configuration(
    argv: Sequence[str] | None = None,
) -> tuple[Config, int, dict]:
    parser = build_parser()
    args, explicit = sdconfig.parse_cli(parser, argv)
    defaults = asdict(Config())
    defaults["seeds"] = 1

    document = None
    file_values = {}
    if args.config is not None:
        document = sdconfig.load_config(args.config)
        file_values = sdconfig.workflow_values(document, "pid_bdt", params_set=paths.params_set())

    cli_values = {}
    mapping = {
        "particles": "particles",
        "features": "feature_cols",
        "exclude_holdout": "exclude_holdout",
        "sigma_time": "smear_sigma_ns",
        "grid_search": "grid_search",
        "seeds": "seeds",
    }
    for cli_name, config_name in mapping.items():
        if cli_name in explicit:
            cli_values[config_name] = explicit[cli_name]
    if explicit.get("no_smote"):
        cli_values.update(use_smote=False, class_weight="balanced")

    effective = sdconfig.merge_config(
        defaults,
        file_values,
        cli_values,
        nullable_types={"class_weight": str, "smear_sigma_ns": (int, float)},
    )
    config_keys = set(asdict(Config()))
    cfg = Config(**{key: effective[key] for key in config_keys})
    if explicit.get("no_timing"):
        kept = [column for column in cfg.feature_cols if column not in set(cfg.smear_columns)]
        cfg = replace(cfg, feature_cols=kept, smear_sigma_ns=None)
        logger.info("Timing dropped: %d features remain", len(kept))
    n_seeds = effective["seeds"]
    resolved_values = {**asdict(cfg), "seeds": n_seeds}
    resolved = sdconfig.build_resolved_config(
        "pid_bdt",
        resolved_values,
        config_path=args.config,
        cli_overrides=[key for key in explicit if key != "config"],
        profile=document.get("profile") if document else None,
        dataset=document.get("dataset") if document else None,
    )
    return cfg, n_seeds, resolved


def main(argv: Sequence[str] | None = None) -> int:
    try:
        cfg, n_seeds, resolved = resolve_configuration(argv)
    except sdconfig.ConfigError as error:
        raise SystemExit(f"configuration error: {error}") from error
    if n_seeds < 1:
        raise SystemExit("--seeds must be at least 1")

    unknown = [particle for particle in cfg.particles if particle not in paths.active_particles()]
    if unknown:
        raise SystemExit(
            f"unknown particle(s): {', '.join(unknown)}; expected among {paths.active_particles()}"
        )

    try:
        summary = run(cfg, n_seeds, resolved)
    except (paths.DatasetNotFound, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 1

    logger.info(
        "Final: accuracy=%.4f +- %.4f | balanced=%.4f",
        summary["accuracy"],
        summary["accuracy_spread"],
        summary["balanced_accuracy"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

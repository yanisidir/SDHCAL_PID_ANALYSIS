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

    python LGBM_classifier_PID.py
    python LGBM_classifier_PID.py --sigma-time 0.1 --seeds 5
    python LGBM_classifier_PID.py --no-timing --grid-search
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold


from sdhcal import metrics as sdmetrics  # noqa: F401  (kept for energy-binned studies)
from sdhcal import paths, rootio, runlog, splits
from sdhcal.smearing import TimeSmearing, smear_time_columns

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)8s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("pid_lgbm")

#: Display names of the classes, in label order.
CLASS_NAMES = {"pi": r"$\pi^-$", "proton": "p", "kaon": r"$K^0_L$"}


@dataclass
class Config:
    """Everything that defines a run. Logged verbatim, feature list included."""

    tree_name: str = "paramsTree"
    pdg_col: str = "particlePDG"

    #: Particles to separate; the order fixes the integer labels (0, 1, 2, ...).
    particles: Sequence[str] = field(default_factory=lambda: list(paths.PARTICLES))

    feature_cols: Sequence[str] = field(
        default_factory=lambda: [
            "Thr1", "Thr2", "Thr3", "Begin", "Radius", "Density", "NClusters", "ratioThr23",
            "Zbary", "Zrms", "PctHitsFirst10", "PlanesWithClusmore2", "AvgClustSize",
            "MaxClustSize", "lambda1", "lambda2", "tMin", "tMax", "tMean", "tSpread",
            "Nmax", "z0_fit", "Xmax", "lambda", "nTrackSegments", "eccentricity3D",
        ]
    )

    # Split: both fractions are of the whole sample.
    test_size: float = 0.20
    val_size: float = 0.10
    random_state: int = 42

    # Class imbalance: SMOTE on the training split, or LightGBM class weights.
    use_smote: bool = True
    class_weight: Optional[str] = None

    # Time smearing; None means ideal timing.
    smear_sigma_ns: Optional[float] = 0.1
    smear_columns: Sequence[str] = ("tMin", "tMax", "tMean", "tSpread")

    # LightGBM.
    learning_rate: float = 0.01
    n_estimators: int = 1000
    num_leaves: int = 63
    max_depth: int = -1
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    min_child_samples: int = 30
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 50

    grid_search: bool = False
    param_grid: Dict[str, List] = field(
        default_factory=lambda: {
            "clf__num_leaves": [31, 63, 127],
            "clf__max_depth": [-1, 10, 20],
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__n_estimators": [500, 1000],
        }
    )

    @property
    def target_pdg(self) -> List[int]:
        return [paths.PARTICLE_PDG[p] for p in self.particles]

    @property
    def labels(self) -> List[str]:
        return [CLASS_NAMES.get(p, p) for p in self.particles]

    def smearing(self) -> TimeSmearing:
        return TimeSmearing(
            sigma_ns=self.smear_sigma_ns, columns=tuple(self.smear_columns), seed=self.random_state
        )

    def uses_timing(self) -> bool:
        return any(c.startswith("t") and c in self.feature_cols for c in self.smear_columns)

    def loggable(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "particles": ";".join(self.particles),
            "tree_name": self.tree_name,
            "test_size": self.test_size,
            "val_size": self.val_size,
            "random_state": self.random_state,
            "use_smote": int(self.use_smote),
            "class_weight": str(self.class_weight),
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "early_stopping_rounds": self.early_stopping_rounds,
            "grid_search": int(self.grid_search),
            "timing_in_features": int(self.uses_timing()),
        }
        out.update(self.smearing().as_dict())
        return out


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_dataset(cfg: Config) -> pd.DataFrame:
    """Read every particle file, label the events and smear the timing once."""
    needed = list(dict.fromkeys([*cfg.feature_cols, cfg.pdg_col]))
    frames = []
    for particle in cfg.particles:
        path = paths.dataset("params", particle)
        frames.append(rootio.load_trees(path, cfg.tree_name, needed))
    df = rootio.clean(pd.concat(frames, ignore_index=True), columns=needed)

    label_of = {pdg: index for index, pdg in enumerate(cfg.target_pdg)}
    df = df[df[cfg.pdg_col].isin(label_of)].copy()
    if df.empty:
        raise RuntimeError(f"no event with PDG among {cfg.target_pdg}")
    df["label"] = df[cfg.pdg_col].map(label_of).astype(int)

    counts = np.bincount(df["label"], minlength=len(cfg.particles))
    logger.info(
        "Class balance: %s",
        ", ".join(f"{p}={c}" for p, c in zip(cfg.particles, counts)),
    )
    return smear_time_columns(df, cfg.smearing())


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_classifier(cfg: Config, seed: int) -> lgb.LGBMClassifier:
    n_classes = len(cfg.particles)
    # num_class must be absent for a binary objective, not passed as None.
    multiclass = {"objective": "multiclass", "num_class": n_classes} if n_classes > 2 else {"objective": "binary"}
    return lgb.LGBMClassifier(
        **multiclass,
        learning_rate=cfg.learning_rate,
        n_estimators=cfg.n_estimators,
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        class_weight=cfg.class_weight,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def search_hyperparameters(x_train: pd.DataFrame, y_train: np.ndarray, cfg: Config, seed: int) -> Dict:
    """Grid-search with the resampling *inside* each cross-validation fold.

    Oversampling before the search is a leak: SMOTE interpolates between
    neighbours, so a synthetic point built from a training-fold event can land in
    the validation fold and be predicted almost perfectly.
    """
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline

    steps = []
    if cfg.use_smote:
        steps.append(("smote", SMOTE(random_state=seed)))
    steps.append(("clf", build_classifier(cfg, seed)))

    grid = GridSearchCV(
        Pipeline(steps),
        cfg.param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(x_train, y_train)
    best = {k.removeprefix("clf__"): v for k, v in grid.best_params_.items()}
    logger.info("Best parameters: %s (CV f1_macro=%.4f)", best, grid.best_score_)
    return best


def train(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    cfg: Config,
    seed: int,
):
    """Resample the training split only, then fit with early stopping on validation."""
    best = search_hyperparameters(x_train, y_train, cfg, seed) if cfg.grid_search else {}

    if cfg.use_smote:
        from imblearn.over_sampling import SMOTE

        x_fit, y_fit = SMOTE(random_state=seed).fit_resample(x_train, y_train)
        x_fit = pd.DataFrame(x_fit, columns=x_train.columns)
        logger.info("Training split after SMOTE: %s", np.bincount(y_fit).tolist())
    else:
        x_fit, y_fit = x_train, y_train

    model = build_classifier(cfg, seed)
    if best:
        model.set_params(**best)
    model.fit(
        x_fit,
        y_fit,
        eval_set=[(x_fit, y_fit), (x_val, y_val)],
        eval_names=["train", "valid"],
        eval_metric="multi_logloss" if len(cfg.particles) > 2 else "binary_logloss",
        callbacks=[
            lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )
    return model, best


def evaluate(model, x_test: pd.DataFrame, y_test: np.ndarray, cfg: Config):
    """Accuracy, balanced accuracy, AUC, log-loss and the confusion matrix.

    Balanced accuracy is reported alongside accuracy because the test split keeps
    the natural class proportions: a plain accuracy can be carried by the
    majority class.
    """
    n_classes = len(cfg.particles)
    proba = model.predict_proba(x_test)
    y_pred = np.argmax(proba, axis=1)

    results = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "log_loss": float(log_loss(y_test, proba, labels=np.arange(n_classes))),
        "n_test": int(y_test.size),
        "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
    }
    if n_classes > 2:
        results["auc_ovr_macro"] = float(roc_auc_score(y_test, proba, multi_class="ovr", average="macro"))
    else:
        results["auc"] = float(roc_auc_score(y_test, proba[:, 1]))

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(n_classes))
    for index, particle in enumerate(cfg.particles):
        true_count = cm[index].sum()
        results[f"efficiency_{particle}"] = float(cm[index, index] / true_count) if true_count else float("nan")
        predicted_count = cm[:, index].sum()
        results[f"purity_{particle}"] = (
            float(cm[index, index] / predicted_count) if predicted_count else float("nan")
        )

    logger.info(
        "Test: accuracy=%.4f  balanced=%.4f  logloss=%.4f%s",
        results["accuracy"],
        results["balanced_accuracy"],
        results["log_loss"],
        f"  AUC={results['auc_ovr_macro']:.4f}" if n_classes > 2 else f"  AUC={results['auc']:.4f}",
    )
    logger.info(
        "Classification report:\n%s",
        classification_report(y_test, y_pred, target_names=list(cfg.particles), digits=4, zero_division=0),
    )
    return results, cm, proba, y_pred


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_training_curve(evals: Dict, out: Path) -> None:
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
def run_single_seed(cfg: Config, seed: int, log: runlog.RunLogger) -> Dict[str, float]:
    set_global_seeds(seed)
    seeded = replace(cfg, random_state=seed)

    df = load_dataset(seeded)
    x = df[list(seeded.feature_cols)].astype(np.float32)
    y = df["label"].to_numpy(dtype=int)

    split = splits.split_indices(
        len(df), test_size=seeded.test_size, val_size=seeded.val_size, seed=seed, stratify=y
    )
    logger.info("Split sizes: %s", split.sizes())

    model, best = train(
        x.iloc[split.train], y[split.train], x.iloc[split.val], y[split.val], seeded, seed
    )
    results, cm, proba, y_pred = evaluate(model, x.iloc[split.test], y[split.test], seeded)
    results["seed"] = seed
    if best:
        results["best_params"] = ";".join(f"{k}={v}" for k, v in best.items())

    tag = f"{log.run_id}_seed{seed}"
    model_path = log.path("models", f"lgbm_pid_{tag}.joblib")
    joblib.dump(
        {
            "model": model,
            "features": list(seeded.feature_cols),
            "particles": list(seeded.particles),
            "label_to_pdg": dict(enumerate(seeded.target_pdg)),
            "smearing": seeded.smearing().as_dict(),
        },
        model_path,
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


def run(cfg: Config, n_seeds: int) -> Dict[str, float]:
    out_dir = paths.output_dir() / "pid_lgbm"
    log = runlog.RunLogger(out_dir, name="pid")
    log.start(cfg.loggable(), features=list(cfg.feature_cols))
    logger.info("=== PID run %d | %d seed(s) | timing in features: %s ===",
                log.run_id, n_seeds, cfg.uses_timing())

    per_seed = [run_single_seed(cfg, cfg.random_state + i, log) for i in range(n_seeds)]
    pd.DataFrame(per_seed).to_csv(log.path("tables", f"per_seed_{log.run_id}.csv"), index=False)

    summary: Dict[str, float] = {"n_seeds": n_seeds}
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
        "--particles", nargs="+", default=list(paths.PARTICLES),
        help="particles to separate; the order fixes the integer labels",
    )
    p.add_argument("--features", nargs="+", default=None, help="override the feature list")
    p.add_argument(
        "--sigma-time", type=float, default=0.1, metavar="NS",
        help="width of the Gaussian time smearing in ns (default 0.1)",
    )
    p.add_argument(
        "--no-timing", action="store_true",
        help="drop every time variable from the features, for the timing-vs-no-timing comparison",
    )
    p.add_argument("--no-smote", action="store_true", help="use LightGBM class weights instead of SMOTE")
    p.add_argument("--seeds", type=int, default=1, metavar="N", help="number of independent splits")
    p.add_argument("--grid-search", action="store_true", help="search hyper-parameters (resampling inside folds)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seeds < 1:
        raise SystemExit("--seeds must be at least 1")

    unknown = [p for p in args.particles if p not in paths.PARTICLES]
    if unknown:
        raise SystemExit(f"unknown particle(s): {', '.join(unknown)}; expected among {paths.PARTICLES}")

    cfg = Config(
        particles=args.particles,
        smear_sigma_ns=args.sigma_time,
        use_smote=not args.no_smote,
        class_weight="balanced" if args.no_smote else None,
        grid_search=args.grid_search,
    )
    if args.features:
        cfg = replace(cfg, feature_cols=list(args.features))
    if args.no_timing:
        kept = [c for c in cfg.feature_cols if c not in set(cfg.smear_columns)]
        cfg = replace(cfg, feature_cols=kept, smear_sigma_ns=None)
        logger.info("Timing dropped: %d features remain", len(kept))

    try:
        summary = run(cfg, args.seeds)
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

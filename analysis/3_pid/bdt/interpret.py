#!/usr/bin/env python3
"""Correlation matrix and permutation importances of the PID features.

Replaces ``legacy_importance.py``, which could not run: it
filtered on a ``primaryID`` branch that exists in no parameter file, listed two
features (``Zbary_thr2``, ``Zbary_thr3``) that exist in neither sample, named the
kaon lot unconditionally, and loaded one hard-coded legacy artifact. It also split
the data itself to build a "test" set, with no guarantee that the model it loaded
had not been trained on those very events - which is exactly what a permutation
importance must avoid.

This script instead:

* takes the feature list, the species and the time smearing **from the model
  artifact**, so the figures describe the model that was actually trained rather
  than a list maintained in parallel;
* measures the importances on the evaluation sample, which for the two-class
  study is an independent production, so no event informing the figure was ever
  seen in training;
* writes PNG, which is what the report includes, next to the CSV of the numbers.

Permutation importance is the drop in balanced accuracy when one column is
shuffled. It is reported with the spread over repeats, because a feature whose
drop is smaller than that spread is not measurably useful.

Usage::

    export SDHCAL_PARAMS_SET=2class
    python analysis/3_pid/bdt/interpret.py
    python analysis/3_pid/bdt/interpret.py --classifier path/to.joblib
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  the backend must be selected first
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

from sdhcal import artifacts, paths, rootio, runlog  # noqa: E402
from sdhcal.smearing import TimeSmearing, smear_time_columns  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)8s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("pid_interpretation")


def default_classifier() -> Path:
    """The published classifier, by the convention the coupling stage uses."""
    return paths.model_dir() / "pid_bdt.joblib"


def load_bundle(path: Path) -> dict:
    try:
        bundle = artifacts.load_bundle(path, kind="pid")
    except artifacts.ArtifactError as error:
        raise SystemExit(f"{path} is not a PID bundle: {error}") from error
    missing = [k for k in ("model", "features", "particles") if k not in bundle]
    if missing:
        raise SystemExit(
            f"{path} is not a PID bundle: missing {', '.join(missing)}. "
            "It must come from train.py, which records the feature "
            "list beside the model."
        )
    logger.info(
        "Classifier: %s | %d features | particles %s",
        path.name,
        len(bundle["features"]),
        ", ".join(bundle["particles"]),
    )
    return bundle


def load_evaluation_sample(bundle: dict, features: Sequence[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Read the evaluation sample, smeared exactly as the model was trained."""
    path = paths.dataset("holdout")
    label_to_pdg = {int(k): int(v) for k, v in (bundle.get("label_to_pdg") or {}).items()}
    if not label_to_pdg:
        label_to_pdg = dict(enumerate(paths.PARTICLE_PDG[p] for p in bundle["particles"]))
    pdg_to_label = {pdg: label for label, pdg in label_to_pdg.items()}

    frame = rootio.load_and_clean(path, "paramsTree", [*features, "particlePDG"])
    frame = frame[frame["particlePDG"].isin(pdg_to_label)].reset_index(drop=True)
    if frame.empty:
        raise SystemExit(f"{path.name} holds no event with PDG among {sorted(pdg_to_label)}")

    recorded = bundle.get("smearing") or {}
    sigma = recorded.get("smear_sigma_ns", 0.1)
    sigma = None if sigma in (None, "none") else float(sigma)
    columns = recorded.get("smear_columns")
    smearing = TimeSmearing(
        sigma_ns=sigma,
        columns=tuple(columns.split(";")) if isinstance(columns, str) else TimeSmearing().columns,
        seed=int(recorded.get("smear_seed", 42)),
    )
    frame = smear_time_columns(frame, smearing)

    labels = frame["particlePDG"].map(pdg_to_label).to_numpy(dtype=int)
    logger.info("Evaluation sample: %d events (%s)", len(frame), path.name)
    return frame, labels


def plot_correlation(frame: pd.DataFrame, features: Sequence[str], out: Path) -> pd.DataFrame:
    """Correlation heatmap, annotated only while the labels stay readable."""
    corr = frame[features].corr()
    size = max(8.0, 0.52 * len(features))
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    image = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(features)), features, rotation=90, fontsize=8)
    ax.set_yticks(range(len(features)), features, fontsize=8)
    if len(features) <= 30:
        for i in range(len(features)):
            for j in range(len(features)):
                value = corr.iat[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if abs(value) > 0.6 else "black",
                )
    fig.colorbar(image, ax=ax, shrink=0.8, label="Pearson correlation")
    ax.set_title("Feature correlation matrix")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    logger.info("wrote %s", out)
    return corr


def report_redundancy(corr: pd.DataFrame, threshold: float = 0.98) -> list[tuple[str, str, float]]:
    """Name the near-duplicate pairs, because they make the next plot misleading.

    Permutation importance shuffles one column at a time. When two columns carry
    the same information, shuffling either leaves the other intact, so both score
    near zero however useful the information is. A pair listed here is a reason to
    distrust a low bar, not evidence that the variable does not matter.
    """
    pairs = [
        (a, b, float(corr.loc[a, b]))
        for i, a in enumerate(corr.columns)
        for b in corr.columns[i + 1 :]
        if abs(corr.loc[a, b]) >= threshold
    ]
    for a, b, value in pairs:
        logger.warning(
            "%s and %s are near-duplicates (r = %+.4f): permutation importance understates both",
            a,
            b,
            value,
        )
    return pairs


def plot_permutation(result, features: Sequence[str], baseline: float, out: Path) -> pd.DataFrame:
    """Horizontal bars of the accuracy drop, with the spread over repeats."""
    table = pd.DataFrame(
        {
            "feature": list(features),
            "importance": result.importances_mean,
            "std": result.importances_std,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)

    ordered = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(5.0, 0.32 * len(features))))
    ax.barh(ordered["feature"], ordered["importance"], xerr=ordered["std"], color="tab:blue", capsize=2)
    ax.axvline(0.0, color="0.3", linewidth=0.8)
    ax.set_xlabel("Drop in balanced accuracy when the column is shuffled")
    ax.set_title(f"Permutation importances (baseline {baseline:.4f})")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    logger.info("wrote %s", out)
    return table


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--classifier",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"PID bundle to interpret (default {default_classifier()})",
    )
    parser.add_argument("--n-repeats", type=int, default=10, help="permutation repeats per feature")
    parser.add_argument(
        "--max-events",
        type=int,
        default=60_000,
        metavar="N",
        help="subsample the evaluation set before permuting, which costs "
        "n_features * n_repeats predictions (0 keeps every event)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    path = args.classifier or default_classifier()
    if not path.is_file():
        raise SystemExit(
            f"classifier not found: {path}\n"
            "Train one with make pid-2class, then publish the chosen run as "
            f"{default_classifier()}, or pass --classifier."
        )

    bundle = load_bundle(path)
    features = list(bundle["features"])
    frame, labels = load_evaluation_sample(bundle, features)

    out_dir = paths.output_dir() / "pid_interpretation"
    log = runlog.RunLogger(out_dir, name="pid_interpretation")
    log.start(
        {
            "classifier": str(path),
            "n_features": len(features),
            "particles": ";".join(bundle["particles"]),
            "n_events_available": int(len(frame)),
            "n_repeats": args.n_repeats,
            "max_events": args.max_events,
            "seed": args.seed,
        },
        features=features,
    )

    corr = plot_correlation(frame, features, log.path("plots", f"correlation_matrix_{log.run_id}.png"))
    corr.to_csv(log.path("tables", f"correlation_matrix_{log.run_id}.csv"))
    redundant = report_redundancy(corr)

    if args.max_events and len(frame) > args.max_events:
        rng = np.random.default_rng(args.seed)
        pick = rng.choice(len(frame), size=args.max_events, replace=False)
        frame, labels = frame.iloc[pick].reset_index(drop=True), labels[pick]
        logger.info("Permuting on a subsample of %d events", len(frame))

    x = frame[features]
    model = bundle["model"]
    baseline = balanced_accuracy_score(labels, model.predict(x))
    logger.info("Baseline balanced accuracy on the evaluation sample: %.4f", baseline)

    result = permutation_importance(
        model,
        x,
        labels,
        scoring="balanced_accuracy",
        n_repeats=args.n_repeats,
        random_state=args.seed,
        n_jobs=-1,
    )
    table = plot_permutation(
        result, features, baseline, log.path("plots", f"permutation_importances_{log.run_id}.png")
    )
    table.to_csv(log.path("tables", f"permutation_importances_{log.run_id}.csv"), index=False)

    significant = table[table["importance"] > table["std"]]
    logger.info(
        "%d of %d features have a drop larger than their own spread; top five: %s",
        len(significant),
        len(table),
        ", ".join(f"{r.feature} ({r.importance:+.4f})" for r in significant.head(5).itertuples()),
    )
    log.log_metrics(
        baseline_balanced_accuracy=baseline,
        n_significant=len(significant),
        n_redundant_pairs=len(redundant),
        redundant_pairs=";".join(f"{a}~{b}" for a, b, _ in redundant),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

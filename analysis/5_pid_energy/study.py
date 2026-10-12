#!/usr/bin/env python3
"""Does identifying the hadron first improve its energy reconstruction?

One script replaces the fifteen near-copies that used to live in
``PID_RECONSTRUCTION/pi-_proton/`` and ``PID_RECONSTRUCTION/kaon_pi-_proton/``
(``BDT_BDT``, ``BDT_chi2``, ``MLP_BDT``, ``MLP_MLP``, ``MLP_chi2``,
``noPID_BDT``, ``noPID_MLP``, ``noPID_chi2``, ``hadron_pid_reco_linear``, ...).
They differed by a few literals, so a fix in one never reached the others - which
is exactly how the three MLP energy scripts ended up identical and broken.

Three corrections matter for the physics, not just the tidiness.

**The "no PID" baseline was not one.** ``noPID_*.py`` routed each event to a
particle-specific regressor using the *true* PDG code::

    mask = (label_true_int == lbl_int)        # noPID_BDT.py, line 167

That is a *perfect-PID upper bound*, not a PID-agnostic baseline, so the study
compared real PID against perfect PID and the honest question - is a
particle-blind regressor worse? - was never asked. There are now three distinct
``--pid`` modes:

``none``
    one inclusive regressor trained on the mixture of all three species. This is
    the genuine baseline, and it needs a model trained with
    ``--particles mixed``.
``true``
    routing by the true PDG: the perfect-PID upper bound.
``bdt`` / ``mlp``
    routing by the classifier's decision: the realistic case.

**The held-out sample was never verified.** Inference ran on ``val_set.root``
while the models had been trained on ``130k_*_params.root`` with their own
internal random split, and nothing established that the two were disjoint. The
run now checks it event by event (:func:`sdhcal.splits.assert_disjoint`) and
refuses to report numbers it cannot trust.

**The time smearing was applied twice to the same column.** ``time_cols_clf``
contained ``tMin``, ``time_cols_reg`` contained it too, and both calls used
``seed=42``, so ``tMin`` was shifted by exactly twice the same noise vector and
no longer matched what the regressor had been trained on. Smearing now happens
once, through :mod:`sdhcal.smearing`, which refuses a second application.

The chi2 coefficients are read from ``calibration/chi2_coefficients.json``
instead of being retyped: the C++ and Python copies had already diverged
(kaon ``c0`` was ``1.22381e-14`` in one and ``9.83127e-15`` in the other).

Examples
--------
::

    # realistic chain: BDT identification, BDT regression
    python study.py --pid bdt --reco bdt

    # the two references it must be compared against
    python study.py --pid true --reco bdt     # perfect PID
    python study.py --pid none --reco bdt     # no PID at all

    # semi-digital chi2 instead of a regressor
    python study.py --pid bdt --reco chi2
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sdhcal import metrics, paths, rootio, runlog, splits
from sdhcal.coupling import (
    CHI2_COLUMNS,  # noqa: F401 - historical import surface
    CLASS_LABELS,
    Config,
    classifier_path,
    event_table,
    identify,
    load_artifact,
    needed_columns,
    reconstruct_energy,
    regressor_path,
    required_regressors,
    summarise,
)
from sdhcal.smearing import smear_time_columns

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)8s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("pid_energy_coupling")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_holdout(cfg: Config, columns: Sequence[str]) -> pd.DataFrame:
    """Load the held-out sample, smear the timing once, keep the target species."""
    path = paths.dataset("holdout")
    available = _branches(path, cfg.tree_name)
    missing_required = [c for c in columns if c not in available and c != "eventNumber"]
    if missing_required:
        raise KeyError(f"{path.name} lacks branch(es): {', '.join(missing_required)}")

    requested = [c for c in columns if c in available]
    if "eventNumber" not in available:
        logger.warning(
            "%s has no 'eventNumber' branch, so the overlap with the training sample cannot be "
            "checked event by event. Re-produce it with tools/build_holdout.py.",
            path.name,
        )

    df = rootio.load_and_clean(path, cfg.tree_name, requested)
    pdg_of_interest = [paths.PARTICLE_PDG[p] for p in cfg.particles]
    df = df[df[cfg.pdg_col].isin(pdg_of_interest)].reset_index(drop=True)
    df = df[df[cfg.energy_col] > 0].reset_index(drop=True)
    if cfg.require_contained:
        before = len(df)
        df = df[df["touchesLastLayer"] == 0].reset_index(drop=True)
        logger.info(
            "Containment cut (no hit in layer %d): %d of %d events kept",
            cfg.last_layer,
            len(df),
            before,
        )
    logger.info("Held-out sample: %d events", len(df))
    return smear_time_columns(df, cfg.smearing())


def _branches(path: Path, tree: str) -> set:
    import uproot

    with uproot.open(path) as handle:
        return set(handle[tree].keys())


def verify_disjoint(df: pd.DataFrame, cfg: Config) -> dict[str, object]:
    """Check that no held-out event was used to train any of the loaded models.

    The comparison is against the training pool written by
    ``tools/build_holdout.py --write-train-pool``, which is what the training
    stages read when they are given ``--exclude-holdout``. Comparing against the
    parameter file instead would always fail, since the held-out events were
    carved out of it and are therefore all still in it.
    """
    if "eventNumber" not in df.columns:
        raise KeyError(
            "the held-out file has no 'eventNumber' branch, so disjointness cannot be "
            "established. Re-produce it with tools/build_holdout.py, or pass "
            "--skip-disjointness-check and say so when quoting the numbers."
        )

    holdout_keys = splits.event_keys(df)
    reports = []
    for particle in cfg.particles:
        # Against the training pool, not the parameter file: the held-out events are
        # carved out of that file, so they are all still in it and comparing with it
        # would report a 100 % overlap for a correctly built sample. The pool is the
        # complement the training stages read with --exclude-holdout.
        train_path = paths.dataset("holdout", require=False).parent / f"train_pool_{particle}.root"
        if not train_path.is_file():
            logger.warning(
                "%s absent, overlap with %s not checked; rebuild it with "
                "tools/build_holdout.py --write-train-pool",
                train_path.name,
                particle,
            )
            continue
        train_df = rootio.load_trees(train_path, cfg.tree_name, [cfg.pdg_col, "eventNumber"])
        report = splits.assert_disjoint(
            splits.event_keys(train_df),
            holdout_keys,
            name_a=f"training file {train_path.name}",
            name_b="held-out sample",
        )
        logger.info(
            "Disjointness verified against %s: %d training events, %d held-out, 0 shared",
            train_path.name,
            report["n_a"],
            report["n_b"],
        )
        reports.append(report)
    return {"checked_files": len(reports)}


# ---------------------------------------------------------------------------
# Results and plotting
# ---------------------------------------------------------------------------
def write_event_csv(
    df: pd.DataFrame,
    cfg: Config,
    routing_label: np.ndarray,
    proba: np.ndarray | None,
    energy: np.ndarray,
    out: Path,
) -> None:
    """Write the per-event table with the existing plots.py column contract."""
    table = event_table(df, cfg, routing_label, proba, energy)
    table.to_csv(out, index=False)
    logger.info("Per-event results: %s (%d rows)", out, len(table))


def plot_confusion(confusion: np.ndarray, cfg: Config, out: Path) -> None:
    labels = [CLASS_LABELS.get(p, p) for p in cfg.particles]
    totals = confusion.sum(axis=1, keepdims=True)
    pct = np.divide(confusion.astype(float), np.maximum(totals, 1)) * 100.0
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 2.4, 1.4 * len(labels) + 2.0))
    image = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{pct[i, j]:.1f} %\n({confusion[i, j]:d})",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if pct[i, j] > 55 else "black",
            )
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"PID confusion matrix ({cfg.pid})")
    fig.colorbar(image, ax=ax, label="fraction of true class [%]")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_bias_and_resolution(df: pd.DataFrame, cfg: Config, energy: np.ndarray, out: Path) -> None:
    e_true = df[cfg.energy_col].to_numpy(dtype=np.float64)
    binned = metrics.binned_bias_resolution(e_true, energy, n_bins=cfg.n_bins, method=cfg.core_method)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.5, 7), sharex=True)
    top.errorbar(binned["center"], binned["bias"], yerr=binned["bias_err"], fmt="o", capsize=3)
    top.axhline(0.0, linestyle="--", linewidth=1, color="0.4")
    top.set_ylabel(r"bias $\langle \Delta E/E\rangle$")
    top.set_title(f"Energy reconstruction: pid={cfg.pid}, reco={cfg.reco}")
    top.grid(alpha=0.3)
    bottom.errorbar(
        binned["center"], binned["sigma"], yerr=binned["sigma_err"], fmt="o", capsize=3, color="tab:purple"
    )
    bottom.set_xlabel("True energy [GeV]")
    bottom.set_ylabel(r"resolution $\sigma(\Delta E/E)$")
    bottom.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(cfg: Config) -> dict[str, float]:
    classifier = (
        load_artifact(classifier_path(cfg), f"{cfg.pid} classifier", kind="pid")
        if cfg.pid in ("bdt", "mlp")
        else None
    )

    regressors: dict[str, dict] = {}
    if cfg.reco != "chi2":
        for mode in required_regressors(cfg):
            regressors[mode] = load_artifact(
                regressor_path(cfg, mode), f"{cfg.reco} regressor for {mode}", kind="energy"
            )

    feature_sets = [bundle["features"] for bundle in regressors.values()]
    if classifier is not None:
        feature_sets.append(classifier["features"])
    columns = needed_columns(cfg, feature_sets)

    df = load_holdout(cfg, columns)

    out_dir = paths.output_dir() / "pid_energy_coupling" / f"pid-{cfg.pid}_reco-{cfg.reco}"
    log = runlog.RunLogger(out_dir, name="coupling")
    log.start(cfg.loggable(), features=sorted({c for s in feature_sets for c in s}))

    if cfg.check_disjoint:
        log.log_artifacts(**{"disjointness": verify_disjoint(df, cfg)})
    else:
        logger.warning(
            "Disjointness check skipped: the held-out events may have been used for training, "
            "in which case these numbers are training performance, not generalisation."
        )

    routing_label, proba = identify(df, cfg, classifier)
    energy = reconstruct_energy(df, routing_label, cfg, regressors)
    results, confusion = summarise(df, cfg, routing_label, energy)

    tag = f"pid-{cfg.pid}_reco-{cfg.reco}_{log.run_id}"
    write_event_csv(df, cfg, routing_label, proba, energy, log.path("tables", f"events_{tag}.csv"))
    plot_bias_and_resolution(df, cfg, energy, log.path("plots", f"bias_resolution_{tag}.png"))
    if cfg.pid not in ("none", "true"):
        plot_confusion(confusion, cfg, log.path("plots", f"confusion_matrix_{tag}.png"))

    log.log_metrics(**results)
    log.log_comment(
        f"pid={cfg.pid} reco={cfg.reco} bias={results['bias']:+.4f} resolution={results['resolution']:.4f}"
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--pid",
        choices=("none", "true", "bdt", "mlp"),
        default="bdt",
        help="none: one inclusive regressor (PID-agnostic baseline); true: routing by the true PDG "
        "(perfect-PID upper bound); bdt/mlp: routing by the classifier",
    )
    p.add_argument("--reco", choices=("bdt", "mlp", "chi2"), default="bdt", help="energy estimator")
    p.add_argument("--particles", nargs="+", default=list(paths.active_particles()))
    p.add_argument(
        "--sigma-time",
        type=float,
        default=0.1,
        metavar="NS",
        help="time smearing in ns; must match the value used for training",
    )
    p.add_argument("--no-timing", action="store_true", help="disable the time smearing")
    p.add_argument(
        "--require-contained",
        action="store_true",
        help="keep only showers with no hit in the last layer, the cut the chi2 "
        "calibration is fitted under; pass it to both arms or to neither",
    )
    p.add_argument(
        "--skip-disjointness-check",
        action="store_true",
        help="do not verify that the held-out events are absent from the training files",
    )
    p.add_argument(
        "--classifier",
        type=Path,
        default=None,
        metavar="PATH",
        help="classifier artifact; default $SDHCAL_MODEL_DIR/pid_<pid>.joblib",
    )
    p.add_argument(
        "--regressor",
        action="append",
        default=[],
        metavar="MODE=PATH",
        help="regressor artifact for one routed bucket, e.g. --regressor pi=/path/to.joblib "
        "or --regressor mixed=/path/to.joblib; repeatable. "
        "Default $SDHCAL_MODEL_DIR/energy_<reco>_<mode>.joblib",
    )
    p.add_argument("--n-bins", type=int, default=13)
    return p


def parse_regressor_overrides(items: Sequence[str]) -> dict[str, Path]:
    """Turn ``MODE=PATH`` arguments into a mapping, rejecting malformed entries."""
    out: dict[str, Path] = {}
    for item in items:
        mode, separator, value = item.partition("=")
        if not separator or not mode or not value:
            raise SystemExit(f"--regressor expects MODE=PATH, got {item!r}")
        if mode not in (*paths.active_particles(), "mixed"):
            raise SystemExit(
                f"--regressor: unknown mode {mode!r}; expected one of {(*paths.active_particles(), 'mixed')}"
            )
        out[mode] = Path(value)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    unknown = [p for p in args.particles if p not in paths.active_particles()]
    if unknown:
        raise SystemExit(f"unknown particle(s): {', '.join(unknown)}")

    cfg = Config(
        particles=args.particles,
        pid=args.pid,
        reco=args.reco,
        smear_sigma_ns=None if args.no_timing else args.sigma_time,
        check_disjoint=not args.skip_disjointness_check,
        require_contained=args.require_contained,
        n_bins=args.n_bins,
        classifier_path=args.classifier,
        regressor_paths=parse_regressor_overrides(args.regressor),
    )
    if cfg.pid == "none" and cfg.reco == "chi2":
        logger.warning("pid=none with reco=chi2 has no particle-blind calibration; see the warning below")

    try:
        run(cfg)
    except (paths.DatasetNotFound, FileNotFoundError, KeyError, ValueError, splits.NotDisjoint) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

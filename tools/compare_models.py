#!/usr/bin/env python3
"""Compare several models on the same events, with the figures of merit quoted in HEP.

Reads the prediction tables written by ``tools/predict.py``, restricts every model
to the events all of them predicted, and writes the numbers the ROOT macro draws.
Computation and drawing are separate so the numbers can be tested, re-used and
re-plotted without re-running anything.

What is written, under ``$SDHCAL_OUT_DIR/benchmark/<kind>/tables``:

===================  ==================================================
``summary.csv``      one row per model: the headline figures
``roc_<model>.csv``  PID: the full ROC, for the rejection-vs-efficiency plot
``eff_<model>.csv``  PID: efficiency per true-energy point
``curve_<model>.csv``  energy: response and resolution per true-energy point
===================  ==================================================

For PID the headline is background rejection at a fixed signal efficiency rather
than accuracy, because accuracy moves with the pion/proton mixture of the sample
- an artefact of how the lots were generated - while a rejection quoted at a
stated efficiency does not.

For energy it is the stochastic and constant terms of ``sigma/E = a/sqrt(E) (+) b``.
A single resolution figure hides which of the two a model improved, and only the
constant term is something an algorithm can act on: the stochastic term is set by
the sampling of the detector.

Usage::

    export SDHCAL_PARAMS_SET=2class
    python tools/compare_models.py --kind energy --models bdt_mixed chi2
    python tools/compare_models.py --kind pid --models bdt mlp
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sdhcal import benchmark, paths

logging.basicConfig(level=logging.INFO, format="%(levelname)8s | %(message)s")
logger = logging.getLogger("compare_models")


def predictions_dir(kind: str) -> Path:
    return paths.output_dir() / "benchmark" / kind


def tables_dir(kind: str) -> Path:
    return predictions_dir(kind) / "tables"


def load_models(kind: str, names: list[str]) -> dict[str, pd.DataFrame]:
    tables = {}
    for name in names:
        path = predictions_dir(kind) / f"{name}.csv"
        if not path.is_file():
            raise SystemExit(f"no predictions for {name!r}: {path}\nProduce them with tools/predict.py.")
        tables[name] = pd.read_csv(path)
        logger.info("%-14s %7d events%s", name, len(tables[name]), _budget_note(kind, name))
    return tables


def _budget(kind: str, name: str) -> int | None:
    """Events the model was trained on, when the prediction step recorded it."""
    sidecar = predictions_dir(kind) / f"{name}.json"
    if not sidecar.is_file():
        return None
    try:
        return json.loads(sidecar.read_text(encoding="utf-8")).get("n_train")
    except json.JSONDecodeError:
        return None


def _budget_note(kind: str, name: str) -> str:
    n = _budget(kind, name)
    return f", trained on {n}" if n else ""


def _thin(frame: pd.DataFrame, column: str, n: int = 1500) -> pd.DataFrame:
    """Keep ``n`` rows spread evenly along ``column``.

    A ROC has one row per distinct score - a hundred thousand of them here - which
    no plot can use and no reader should have to parse. Thinning on the quantiles
    of the efficiency keeps the shape everywhere, including the steep low-efficiency
    end where a uniform stride would lose it. The endpoints are always kept.
    """
    if len(frame) <= n:
        return frame
    targets = np.linspace(frame[column].iloc[0], frame[column].iloc[-1], n)
    index = np.unique(np.searchsorted(frame[column].to_numpy(), targets))
    index = np.unique(np.r_[0, index[index < len(frame)], len(frame) - 1])
    return frame.iloc[index].reset_index(drop=True)


def compare_pid(tables: dict[str, pd.DataFrame], positive_pdg: int, efficiencies: list[float]) -> pd.DataFrame:
    out = tables_dir("pid")
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for name, frame in tables.items():
        summary = benchmark.pid_summary(frame, positive_pdg=positive_pdg, efficiencies=efficiencies)
        labels = (frame["particlePDG"].to_numpy() == positive_pdg).astype(int)
        curve = benchmark.roc_curve(labels, frame["score"].to_numpy())
        roc = pd.DataFrame(
            {
                "efficiency_signal": curve["efficiency_signal"],
                "efficiency_background": curve["efficiency_background"],
                "threshold": curve["threshold"],
            }
        )
        _thin(roc, "efficiency_signal").to_csv(out / f"roc_{name}.csv", index=False)

        # Efficiency per energy at the 50 % working point, the one a routing
        # decision would actually run at.
        reference = summary["working_points"][f"eff{int(round(efficiencies[0] * 100))}"]
        benchmark.efficiency_vs_energy(
            frame, positive_pdg=positive_pdg, threshold=reference["threshold"]
        ).to_csv(out / f"eff_{name}.csv", index=False)

        row = {
            "model": name,
            "n_events": len(frame),
            "auc": summary["auc"],
            "accuracy_at_0p5": summary["accuracy_at_0p5"],
        }
        for key, point in summary["working_points"].items():
            row[f"rejection_{key}"] = point["rejection"]
            row[f"effbkg_{key}"] = point["efficiency_background"]
            row[f"effbkg_err_{key}"] = point["efficiency_background_err"]
            row[f"threshold_{key}"] = point["threshold"]
        row["n_train"] = _budget("pid", name)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("auc", ascending=False, ignore_index=True)


def compare_energy(tables: dict[str, pd.DataFrame], method: str) -> pd.DataFrame:
    out = tables_dir("energy")
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for name, frame in tables.items():
        summary = benchmark.energy_summary(frame, method=method)
        summary.curve.to_csv(out / f"curve_{name}.csv", index=False)
        rows.append({"model": name, "n_train": _budget("energy", name), **summary.as_row()})

    return pd.DataFrame(rows).sort_values("constant_term", ignore_index=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kind", required=True, choices=("pid", "energy"))
    parser.add_argument("--models", nargs="+", required=True, help="names given to tools/predict.py")
    parser.add_argument(
        "--positive-pdg",
        type=int,
        default=paths.PARTICLE_PDG["pi"],
        help="PID only: the class the score refers to; must match tools/predict.py",
    )
    parser.add_argument(
        "--efficiencies",
        nargs="+",
        type=float,
        default=[0.5, 0.8, 0.9],
        help="PID only: signal efficiencies at which to quote the rejection",
    )
    parser.add_argument(
        "--core-method",
        choices=("truncated_rms", "iterative_gauss", "full_rms"),
        default="truncated_rms",
        help="energy only: estimator of the Gaussian core",
    )
    args = parser.parse_args(argv)

    tables = load_models(args.kind, args.models)
    aligned = benchmark.align(tables, args.kind)
    kept = len(next(iter(aligned.values())))
    dropped = {n: len(t) - kept for n, t in tables.items()}
    if any(dropped.values()):
        logger.info(
            "compared on the %d events every model predicted (dropped: %s)",
            kept,
            ", ".join(f"{n} -{d}" for n, d in dropped.items() if d),
        )

    if args.kind == "pid":
        summary = compare_pid(aligned, args.positive_pdg, args.efficiencies)
    else:
        summary = compare_energy(aligned, args.core_method)

    path = tables_dir(args.kind) / "summary.csv"
    summary.to_csv(path, index=False)
    logger.info("wrote %s", path)
    print()
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

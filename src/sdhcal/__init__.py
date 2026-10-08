"""Shared building blocks for the SDHCAL PID and energy-reconstruction studies.

This package exists so that the analysis scripts stop duplicating (and silently
diverging on) the four things that must be identical everywhere:

* :mod:`sdhcal.paths`     - where the data lives (no hard-coded ``/gridgroup`` paths);
* :mod:`sdhcal.smearing`  - the detector time smearing, applied once, with independent noise;
* :mod:`sdhcal.metrics`   - bias and resolution, defined unambiguously;
* :mod:`sdhcal.runlog`    - run bookkeeping that survives a schema change;
* :mod:`sdhcal.calib`     - the chi2 calibration coefficients, stored once and read by
  both the C++ and the Python side;
* :mod:`sdhcal.splits`    - event-level splits and the disjointness checks that go with them.
"""

__all__ = ["paths", "smearing", "metrics", "runlog", "calib", "splits", "rootio"]

__version__ = "1.0.0"

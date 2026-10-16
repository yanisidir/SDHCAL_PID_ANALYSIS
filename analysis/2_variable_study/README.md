# Stage 2 - Shower-variable comparison

Overlays a shower variable for pi-, p and K0 to see which ones separate the
species at all. This is what justifies the feature list used in stage 3.

```bash
make variables                       # every macro
root -l -b -q density.C       # one of them
```

**Reads** the merged parameter files under `$SDHCAL_DATA_DIR`, resolved through
[`../../cpp/include/SdhcalPaths.h`](../../cpp/include/SdhcalPaths.h).

**Writes** a PDF per variable, next to the macro.

| Macro | Variable |
|---|---|
| `begin.C` | shower start layer |
| `density.C` | local hit density |
| `radius.C` | transverse radius |
| `thr3.C` | fraction of third-threshold hits |
| `em_fraction.C` | electromagnetic-like fraction |

`emFraction` is not computed by `ShowerAnalyzer`: it is a simulation truth
branch carried through the production, so the overlay it produces here is a
truth-level comparison, not a reconstructed observable. The same branch is used
by [`../../tools/correction_mechanism.py`](../../tools/correction_mechanism.py)
to test whether the correction the energy network learns tracks the
electromagnetic fraction of the shower -- it does not.

The committed PDFs (`Begin.pdf`, `Density.pdf`, `Radius.pdf`, `Thr3.pdf`,
`emFraction.pdf`) are the outputs of these macros on the internship samples.

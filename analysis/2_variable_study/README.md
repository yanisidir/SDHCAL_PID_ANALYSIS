# Stage 2 - Shower-variable comparison

Overlays a shower variable for pi-, p and K0 to see which ones separate the
species at all. This is what justifies the feature list used in stage 3.

```bash
make variables                       # every macro
root -l -b -q compareDensity.C       # one of them
```

**Reads** the merged parameter files under `$SDHCAL_DATA_DIR`, resolved through
[`../../cpp/include/SdhcalPaths.h`](../../cpp/include/SdhcalPaths.h).

**Writes** a PDF per variable, next to the macro.

| Macro | Variable |
|---|---|
| `compareBegin.C` | shower start layer |
| `compareDensity.C` | local hit density |
| `compareRadius.C` | transverse radius |
| `compareThr3.C` | fraction of third-threshold hits |
| `compareEmFraction.C` | electromagnetic-like fraction |

The committed PDFs (`Begin.pdf`, `Density.pdf`, `Radius.pdf`, `Thr3.pdf`,
`emFraction.pdf`) are the outputs of these macros on the internship samples.

# Stage 1 - Shower features

Turns digitized SDHCAL hits into one row of shower variables per event. Every
later stage reads the output of this one.

```bash
make features                                       # every particle
./extract <input.root> tree <output.root>      # one file
IN=... OUT=... ./parallel.sh                     # many files, in parallel
```

**Reads** `$SDHCAL_DATA_DIR/digitized/130k_<particle>_E1to130_digitized.root`,
tree `tree`, branches `thr`, `K`, `x`, `y`, `z`, `I`, `J`, `time`,
`particle1PDG` and, when present, `eventNumber`.

**Writes** `$SDHCAL_DATA_DIR/params/130k_<particle>_E1to130_params.root`, tree
`paramsTree`.

## Files

- `extract.cpp`: event-loop driver. Takes input/tree/output on the command
  line, or processes every particle under `$SDHCAL_DATA_DIR`.
- `extract_parallel.cpp`: variant that appends the parameter branches to an
  existing tree, for use with GNU parallel.
- `parallel.sh`: driver for the parallel variant; directories come from
  `IN`, `OUT` and `SDHCAL_DATA_DIR`.

The observable computation itself lives in [`../../cpp/`](../../cpp/) -
`cpp/include/ShowerAnalyzer.h` and `cpp/src/ShowerAnalyzer.cpp` - because the
ROOT macros of the later stages use it too. That directory's README documents
every observable the class computes.

## Two branches that exist for the integrity of the analysis

| Branch | Why |
|---|---|
| `eventNumber` | Lets a held-out sample be proved disjoint from the training events. Without it, the stage-5 coupling study cannot tell generalisation from training performance. |
| `touchesLastLayer` | Flags showers reaching the last active layer. The chi2 reconstruction rejects them; the flag lets the ML chain apply the identical cut, so the two are compared on the same events. |

`Begin` carries the sentinel value 48 when no shower start is found - one past
the last layer, deliberately outside the physical range. It is a "not found"
marker, not a depth.

## Example Observables

The analyzer computes variables used downstream for PID and energy reconstruction, including:

- threshold counts and ratios;
- shower start and longitudinal barycentre;
- transverse radius and density;
- clustering summaries;
- timing summaries;
- fitted longitudinal-profile quantities;
- topology-related variables such as track-segment counts.

## Outputs

A ROOT file holding a `paramsTree` with one entry per event: the shower
observables, `particlePDG`, `eventNumber` and `touchesLastLayer`.

Compiled objects and binaries are build products, not source: `extract`
and `extract_parallel` are git-ignored, and `make clean` removes them.

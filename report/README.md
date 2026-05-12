# M2 Internship Report

This directory contains the LaTeX source and compiled PDF for the M2 internship report associated with this repository.

## Topic

The report documents an SDHCAL particle-identification and energy-reconstruction study. It covers the detector context, semi-digital calorimetry, hadronic shower observables, PID models, energy reconstruction methods, and studies coupling PID decisions to reconstructed energy.

## Internship Context

This work was produced as part of an M2 internship on SDHCAL analysis. The repository contains the research code and selected results, while this report provides the longer scientific narrative, methodology, and interpretation.

## Structure

- `main.tex`: main LaTeX entry point.
- `chapters/`: report chapters.
- `Annexes/`: technical appendices.
- `Fig/`: scientific figures used in the report.
- `img/`: logos and institutional images.
- `references.bib`: bibliography database.
- `main.pdf`: compiled report kept as a lightweight portfolio artifact.

## Compilation

From this directory:

```sh
make
```

The Makefile uses `latexmk` and writes temporary build products to `build/`. The final PDF is copied to `main.pdf`.

To remove generated build files:

```sh
make clean
```

To remove generated build files and the compiled PDF:

```sh
make clean-all
```

## Notes

The LaTeX source was copied without changing the scientific content. Generated temporary files are ignored by Git, and the existing compiled `main.pdf` is kept because it is small enough for the repository.

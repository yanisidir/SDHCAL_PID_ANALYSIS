# SDHCAL PID ANALYSIS

Particle identification (π⁻/K⁰/proton) and energy reconstruction in an SDHCAL (Semi-Digital Hadronic CALorimeter) from simulated datasets (1–130 GeV).
The project includes: shower parameter extraction (C++/ROOT), PID models (BDT/MLP/GNN), energy reconstruction (BDT/MLP and χ²/TMinuit methods), and associated figures.

---

## Key points

* **Data**: simulated ROOT files (raw, digitized, parameters, splits, validation sets) referenced under `data/` (not versioned).
* **Shower parameters**: computed in `ShowerAnalyzer/` and `data/scripts/` (C++/ROOT + Python utilities).
* **PID**: models **BDT**, **MLP**, **GNN** under `PID/` (artifacts, curves, confusion matrices).
* **Energy reconstruction**: variants **BDT/MLP** (Python) and **χ²/TMinuit** (C++/ROOT) under `Energy_reconstruction_ml/` and `energy_reconstruction_Tminuit/`.
* **Comparisons**: ROOT scripts to compare density/radius/EM-fraction/Thr3 under `compare_parameters/`.

---

## Table of contents

* [Prerequisites](#prerequisites)
* [Data](#data)

  * [Data pipeline (from SDHCALSim → PID/Energy)](#data-pipeline-from-sdhcalsim--pidenergy)
* [Quick installation](#quick-installation)
* [Repository structure](#repository-structure)
* [Typical pipelines](#typical-pipelines)

  * [1) Preparation & parameters](#1-preparation--parameters)
  * [2) PID (BDT/MLP/GNN)](#2-pid-bdtmlpgnn)
  * [3) Energy reconstruction](#3-energy-reconstruction)
  * [4) PID → Energy (coupling)](#4-pid--energy-coupling)
* [Quick start](#quick-start)
* [Results & figures](#results--figures)
* [Reproducibility tips](#reproducibility-tips)
* [Contributing](#contributing)
* [Contact](#contact)

---

## Prerequisites

### System tools

* **C++11** (compilation tested with C++11)
* **ROOT** (with `root-config` in PATH)
* **Python 3.9+** recommended
* (Optional) **conda** for Python environment

### Python (typical)

* numpy, pandas, scikit-learn, joblib
* lightgbm (for LGBM)
* matplotlib
* (GNN) PyTorch + PyTorch Geometric (if using `PID/GNN/`)

> Trained models/scalers (`.joblib` and `.pt/.pth` files) can be regenerated; they are not required if you retrain.

---

## Data

The datasets used for the analysis are **not included** (too large).
They come from **[SDHCALSim](https://github.com/ggarillot/SDHCALSim)** to **simulate the SDHCAL prototype**, **[digitizerTuning](https://github.com/ggarillot/digitizerTuning)** to **digitize**, and **[SDHCALMarlinProcessor](https://github.com/ggarillot/SDHCALMarlinProcessor)** to **convert LCIO→ROOT**.

### Get/link the data

1. Clone the upstream repo:

   ```bash
   git clone https://github.com/ggarillot/SDHCALSim
   export SDHCALSIM_DIR=$(pwd)/SDHCALSim
   ```
2. Choose a local location for your outputs (not versioned):

   ```bash
   export SDHCAL_DATA_DIR=/path/to/SDHCAL_data
   mkdir -p "$SDHCAL_DATA_DIR"
   ```
3. (Option A) **Environment variable**: our scripts will read `SDHCAL_DATA_DIR`.
   (Option B) **Symbolic link** at the repo root:

   ```bash
   ln -s "$SDHCAL_DATA_DIR" data
   ```

### Data pipeline (from SDHCALSim → PID/Energy)

```
SDHCALSim (slcio) ─► Digitization (slcio) ─► LcioToRoot (root) ─► ShowerAnalyzer (params.root)
   example.py            digitOnLocal.py      LcioToRootProcessor.py    computeParams*.cpp
```

> Upstream requirements:
>
> * [iLCSoft](https://github.com/iLCSoft/iLCInstall) (Marlin/LCIO),
> * [SDHCALMarlinProcessor](https://github.com/ggarillot/SDHCALMarlinProcessor) (depends on [CaloSoftWare](https://github.com/SDHCAL/CaloSoftWare)),
> * [SDHCALSim](https://github.com/ggarillot/SDHCALSim)
> * [digitizerTuning](https://github.com/ggarillot/digitizerTuning)
>   On sites with **CVMFS**, source the environment (e.g.):
>
> ```bash
> source /cvmfs/ilc.desy.de/sw/x86_64_gcc82_centos7/v02-02-01/init_ilcsoft.sh
> ```

Then follow the **four steps**:

1. **Simulation (LCIO .slcio)**
2. **Digitization (slcio → slcio)**
3. **Conversion LCIO → ROOT**
4. **Parameter extraction (params.root)** with `ShowerAnalyzer/computeParams.cpp`

---

## Quick installation

```bash
# 1) Clone this repo
git clone <REPO_URL> SDHCAL_PID_ANALYSIS
cd SDHCAL_PID_ANALYSIS

# 2) (Optional) Create the environment
conda create -n sdhcal python=3.10 -y
conda activate sdhcal

# 3) Install usual dependencies
pip install numpy pandas scikit-learn joblib lightgbm matplotlib
# (GNN) adapt to your platform:
# pip install torch torch_geometric
```

C++/ROOT example compilation:

```bash
g++ -std=c++11 ShowerAnalyzer/computeParams.cpp $(root-config --cflags --libs) -o computeParams
```

---

## Repository structure

* `data/` — **Datasets** and artifacts (not versioned)

  * `raw/`, `digitized/` — raw/digitized ROOT
  * `params/`, `merged_primaryEnergy/` — ROOT **parameters** (features) by particle/energy
  * `split*/`, `validation_set_*.root`, `val_set_*.root` — splits & validation sets
  * `scripts/` — utilities (merge, repair, visualization, split, shuffle, etc.)
  * `data_1k/` — toy dataset + scripts `root_to_csv.py`, `clean_csv.py`, `analyse_csv.py`
* `ShowerAnalyzer/` — **Parameter extraction** (computeParams, parallel version, logs)
* `PID/` — **Particle identification**

  * `BDT/`, `MLP/`, `GNN/` — training/inference, artifacts (`models/`, `plots/`, CSVs)
* `Energy_reconstruction_ml/` — **Energy reconstruction** (BDT/MLP)
* `energy_reconstruction_Tminuit/` — **χ²/TMinuit** (ROOT/C++), by species + global plots
* `compare_parameters/` — Variable comparisons (macros `.C` + figures)
* `PID_RECONSTRUCTION/` — **Coupled studies** PID → energy reconstruction (figures, CSVs)
* `tools/` — utilities (RANSAC tracks, shower visualization)

---

## Typical pipelines

### 1) Preparation & parameters

* From digitized ROOT → extract **params** with `ShowerAnalyzer`.
* Merge/clean with `data/scripts/` (merge\_primary\_energy, repair\_params, rootspliter).
* Parallel version available: `computeParams_parallel.cpp`.

### 2) PID (BDT/MLP/GNN)

* **BDT**: `PID/BDT/` (LightGBM, plots, confusion matrices)
* **MLP**: `PID/MLP/` (classification, results/plots)
* **GNN**: `PID/GNN/` (PyTorch Geometric, trained `.pt` models, plots)

### 3) Energy reconstruction

* **ML (BDT/MLP)**: `Energy_reconstruction_ml/`
* **χ² / TMinuit (ROOT/C++)**: `energy_reconstruction_Tminuit/`

### 4) PID → Energy (coupling)

* `PID_RECONSTRUCTION/`: scenarios **with** and **without** PID (per species or global)

---

## Quick start

```bash
# 0) Make sure you have a params ROOT ready:
#    $SDHCAL_DATA_DIR/params/output_params.root
export SDHCAL_DATA_DIR=/path/to/SDHCAL_data

# 1) PID (example: BDT 3 classes)
python PID/BDT/LGBM_classifier_PID.py \
  --input "$SDHCAL_DATA_DIR/params/output_params.root" \
  --out   PID/BDT/processed_data

# 2) Energy reconstruction (example: LGBM)
python Energy_reconstruction_ml/BDT/hadron_energy_reco_lgbm.py \
  --input "$SDHCAL_DATA_DIR/params/output_params.root" \
  --out   Energy_reconstruction_ml/BDT/results_all_energy_reco
```

---

## Results & figures

Typical outputs:

* `PID/*/plots/` — PID results
* `Energy_reconstruction_ml/*/plots/` and `*/results_*/*/plots/` — ML energy reconstruction
* `energy_reconstruction_Tminuit/plots/` — χ²/TMinuit
* `PID_RECONSTRUCTION/*/plots/` — Coupling PID ↔ Ereco
* `compare_parameters/plots/` — Shower variable comparisons

---

## Reproducibility tips

* **Seeds**: set numpy/sklearn/torch for reproducibility.
* **Splits**: keep `data/split*/` for fair comparisons.
* **Normalization**: save/reload scalers (`scaler_*.joblib`).
* **Versioning**: log ROOT/compiler/package versions in `Energy_reconstruction_ml/*/parameters/run_parameters*.csv`.
* **.gitignore**: raw data/models/results are ignored; provide **samples** under `data/samples/` for demo runs.

---

## Contributing

1. Fork → branch `feat/...` or `fix/...`
2. Follow **C++11** and Python **PEP8**
3. Add a **Reproducibility note** (seed, splits, versions) in the PR
4. Export figures as **.pdf** and **.png** (place in the relevant `plots/` folder)

---

## Contact

* **Author**: IDIR Mohamed Yanis (yanis.idr@outlook.fr)

---

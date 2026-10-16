# Entry points of the SDHCAL PID and energy-reconstruction analysis.
#
#   make help      list the targets
#   make venv      create .venv, install the dependencies and the sdhcal package
#   make check     compile every source, run the tests, check the C++
#
# The analysis chain, in order:
#
#   make features  1. shower parameters from the digitized hits   (C++/ROOT)
#   make variables 2. shower-variable comparison between species  (ROOT macros)
#   make pid       3. hadron identification                       (LightGBM)
#   make energy    4. energy reconstruction                       (LightGBM + chi2)
#   make coupling  5. does identifying first help?                (the three --pid modes)
#
# The pion/proton study runs off the sharded lots instead, and needs no C++ stage
# because they already carry the shower variables:
#
#   make study-2class                                             (the whole chain)
#
# Data locations come from the environment; see src/sdhcal/paths.py.
#   export SDHCAL_DATA_DIR=/path/to/root/files

PYTHON      ?= python3
VENV        ?= .venv
VENV_PYTHON := $(VENV)/bin/python
ROOT_CONFIG ?= root-config
ROOT        ?= root
REQUIRE_ROOT ?= 0

# Works in a fresh checkout, before `pip install -e .`.
export PYTHONPATH := $(CURDIR)/src:$(PYTHONPATH)

CXX       ?= g++
CXXFLAGS  ?= -std=c++17 -O2 -Wall -Wextra
ROOT_CFLAGS := $(shell "$(ROOT_CONFIG)" --cflags 2>/dev/null)
ROOT_LIBS   := $(shell "$(ROOT_CONFIG)" --libs 2>/dev/null)
CPP_INCLUDE := -I cpp/include

LINT_SOURCES := src tools tests \
                analysis/3_pid/bdt analysis/3_pid/mlp \
                analysis/4_energy/ml_bdt analysis/4_energy/ml_mlp \
                analysis/5_pid_energy
SHOWER_SOURCES := cpp/src/ShowerAnalyzer.cpp \
                  cpp/src/ShowerFeatures.cpp \
                  cpp/src/RootShowerIO.cpp
CPP_SOURCES := $(SHOWER_SOURCES) \
               analysis/1_shower_features/extract.cpp \
               analysis/1_shower_features/extract_parallel.cpp \
               tools/count_tracks_ransac.cpp
ROOT_MACROS := analysis/4_energy/chi2_tminuit/fit.C

# Stage scripts
PID_SCRIPT      := analysis/3_pid/bdt/train.py
ENERGY_SCRIPT   := analysis/4_energy/ml_bdt/train.py
COUPLING_SCRIPT := analysis/5_pid_energy/study.py
FEATURES_BIN    := analysis/1_shower_features/extract
ENRICH_BIN      := analysis/1_shower_features/extract_parallel
CPP_TEST_BIN    := /tmp/sdhcal_shower_features_test

# Arguments you will want to override, e.g. `make pid PID_ARGS="--no-timing"`
PID_ARGS      ?= --seeds 5
ENERGY_ARGS   ?= --particles all --seeds 5
COUPLING_ARGS ?=
PARTICLES     ?= pi kaon proton

.DEFAULT_GOAL := help
.PHONY: help venv check-python test check compile cpp cpp-test lint shower-analyzer \
        features variables pid energy energy-ml energy-chi2 coupling \
        calibration holdout clean clean-pyc \
        params-2class pid-2class interpret-2class energy-2class chi2-2class \
        coupling-2class study-2class predictions-2class benchmark-2class

help:
	@echo "Setup and checks:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Setup and verification
# ---------------------------------------------------------------------------
venv: ## create .venv and install the dependencies and the sdhcal package
	"$(PYTHON)" -m venv "$(VENV)"
	"$(VENV_PYTHON)" -m pip install --upgrade pip
	"$(VENV_PYTHON)" -m pip install -r requirements.txt
	"$(VENV_PYTHON)" -m pip install -e .
	@echo "Activate with: source \"$(VENV)/bin/activate\""

check-python:
	@command -v "$(PYTHON)" >/dev/null 2>&1 || { \
	  echo "Python interpreter not found: $(PYTHON)" >&2; \
	  echo "override it with: make <target> PYTHON=/path/to/python" >&2; \
	  exit 1; \
	}

test: check-python ## run the test suite
	"$(PYTHON)" -m pytest tests/ -q

compile: check-python ## byte-compile every Python file, catching syntax errors
	@if find . -name '*.py' -not -path './.git/*' -not -path './$(VENV)/*' \
	  -not -path './analysis/archive/*' -exec "$(PYTHON)" -m py_compile {} +; then \
	  failed=0; else failed=1; fi; \
	find . -name '__pycache__' -not -path './.git/*' -not -path './analysis/archive/*' -exec rm -rf {} + 2>/dev/null; \
	if [ $$failed -eq 0 ]; then echo "all Python sources compile"; else exit 1; fi

cpp: ## syntax-check the C++ sources and the ROOT macros
	@if [ -z "$(ROOT_CFLAGS)" ]; then \
	  echo "root-config not found: C++ check unavailable"; \
	  if [ "$(REQUIRE_ROOT)" = 1 ]; then exit 1; else exit 0; fi; \
	fi; \
	for f in $(CPP_SOURCES); do \
	  echo "  $$f"; \
	  "$(CXX)" $(CXXFLAGS) -fsyntax-only $(ROOT_CFLAGS) $(CPP_INCLUDE) "$$f" || exit 1; \
	done; \
	macro_check=$$(mktemp /tmp/sdhcal_macro_check.XXXXXX.cpp) || exit 1; \
	trap 'rm -f "$$macro_check"' EXIT; \
	for m in $(ROOT_MACROS); do \
	  echo "  $$m"; \
	  printf '#include "%s/%s"\nint main(){return 0;}\n' "$(CURDIR)" "$$m" > "$$macro_check"; \
	  "$(CXX)" -std=c++17 -fsyntax-only $(ROOT_CFLAGS) "$$macro_check" || exit 1; \
	done; \
	echo "all C++ sources compile"

lint: check-python ## run ruff over maintained Python code (see docs/reproducibility.md)
	"$(PYTHON)" -m ruff check $(LINT_SOURCES)
	"$(PYTHON)" -m ruff format --check src tests

check: compile test cpp cpp-test ## everything that can be verified without the data

# ---------------------------------------------------------------------------
# 1. Shower features
# ---------------------------------------------------------------------------
$(FEATURES_BIN): $(SHOWER_SOURCES) analysis/1_shower_features/extract.cpp \
                 cpp/include/ShowerAnalyzer.h cpp/include/ShowerFeatures.h cpp/include/RootShowerIO.h
	@if [ -z "$(ROOT_CFLAGS)" ]; then echo "root-config not found"; exit 1; fi
	"$(CXX)" $(CXXFLAGS) $(ROOT_CFLAGS) $(CPP_INCLUDE) \
	  analysis/1_shower_features/extract.cpp $(SHOWER_SOURCES) \
	  -o "$@" $(ROOT_LIBS)
	@echo "built $@"

$(ENRICH_BIN): $(SHOWER_SOURCES) analysis/1_shower_features/extract_parallel.cpp \
               cpp/include/ShowerAnalyzer.h cpp/include/ShowerFeatures.h cpp/include/RootShowerIO.h
	@if [ -z "$(ROOT_CFLAGS)" ]; then echo "root-config not found"; exit 1; fi
	"$(CXX)" $(CXXFLAGS) $(ROOT_CFLAGS) $(CPP_INCLUDE) \
	  analysis/1_shower_features/extract_parallel.cpp $(SHOWER_SOURCES) \
	  -o "$@" $(ROOT_LIBS)
	@echo "built $@"

shower-analyzer: $(FEATURES_BIN) $(ENRICH_BIN) ## build both historical extraction applications

cpp-test: ## run the deterministic ShowerAnalyzer characterization test
	@if [ -z "$(ROOT_CFLAGS)" ]; then \
	  echo "root-config not found: C++ unit tests BLOCKED"; \
	  if [ "$(REQUIRE_ROOT)" = 1 ]; then exit 1; else exit 0; fi; \
	fi; \
	"$(CXX)" $(CXXFLAGS) $(ROOT_CFLAGS) $(CPP_INCLUDE) \
	  tests/cpp/test_shower_features.cpp cpp/src/ShowerFeatures.cpp cpp/src/ShowerAnalyzer.cpp \
	  -o "$(CPP_TEST_BIN)" $(ROOT_LIBS) && \
	"$(CPP_TEST_BIN)"

features: $(FEATURES_BIN) ## 1. extract the shower parameters of every particle
	./$(FEATURES_BIN)

# ---------------------------------------------------------------------------
# 2. Shower-variable comparison
# ---------------------------------------------------------------------------
variables: ## 2. compare the shower variables between species
	@cd analysis/2_variable_study && for macro in begin.C density.C em_fraction.C radius.C thr3.C; do \
	  echo "  $$macro"; \
	  "$(ROOT)" -l -b -q "$$macro" || exit 1; \
	done

# ---------------------------------------------------------------------------
# 3. Particle identification
# ---------------------------------------------------------------------------
pid: ## 3. train and evaluate the hadron classifier
	"$(PYTHON)" $(PID_SCRIPT) $(PID_ARGS)

# ---------------------------------------------------------------------------
# 4. Energy reconstruction
# ---------------------------------------------------------------------------
energy: energy-ml energy-chi2 ## 4. both energy reconstructions

energy-ml: ## 4a. LightGBM regressors, per particle and inclusive
	"$(PYTHON)" $(ENERGY_SCRIPT) $(ENERGY_ARGS)
	"$(PYTHON)" $(ENERGY_SCRIPT) --particles mixed

energy-chi2: ## 4b. semi-digital chi2 calibration, then publish the coefficients
	@for p in $(PARTICLES); do \
	  echo "  chi2 fit: $$p"; \
	  "$(ROOT)" -l -b -q "analysis/4_energy/chi2_tminuit/fit.C(\"$$p\")" || exit 1; \
	done
	$(MAKE) calibration

calibration: ## merge the chi2 fit fragments into the shared calibration file
	"$(PYTHON)" tools/update_calibration.py

# ---------------------------------------------------------------------------
# 5. PID-energy coupling
# ---------------------------------------------------------------------------
coupling: ## 5. the three cases that belong in the same table
	"$(PYTHON)" $(COUPLING_SCRIPT) --pid none --reco bdt $(COUPLING_ARGS)
	"$(PYTHON)" $(COUPLING_SCRIPT) --pid true --reco bdt $(COUPLING_ARGS)
	"$(PYTHON)" $(COUPLING_SCRIPT) --pid bdt  --reco bdt $(COUPLING_ARGS)

# ---------------------------------------------------------------------------
# The two-class study: pion/proton, from the sharded lots
# ---------------------------------------------------------------------------
# The lots carry no kaon, so this chain separates two species. They already hold
# the shower variables, so `features` does not run; and they carry the true
# energy, so nothing has to be merged.
#
# The two lots are independent productions with fixed roles: 200k (continuous
# 1-130 GeV) trains and calibrates, 100k discret (20 fixed energies) evaluates.
# The evaluation sample is therefore NOT carved out of the training one - there is
# no holdout step here, and --exclude-holdout must not be passed.
#
# Everything below sets two variables, and both are load-bearing - see
# src/sdhcal/paths.py.
#
# SDHCAL_PARAMS_SET=2class redirects the stages at the consolidated files.
#
# SDHCAL_SHARDS_DIR says which shard lot those files are built from. Since the
# reorganisation it defaults to the CORRECTED lot, so the trap it used to guard
# against is gone at the source; the pin stays because it also documents, in the
# recipe itself, which lot a run read. The superseded lot produced BEFORE the
# ShowerAnalyzer corrections (annex 1 of the report) is kept beside it as
# `params-superseded`. Reading it raises nothing - it yields the inverted
# eccentricity and the unbounded Gaisser-Hillas fit, and a model trained on it
# reports entirely plausible numbers. It happened once, to the MLP, and was
# caught only because its AUC came out at 0.599 instead of 0.677.

SHARDS_2CLASS ?= $(CURDIR)/data/lots/pi-proton/params
TWO_CLASS     := SDHCAL_PARAMS_SET=2class SDHCAL_SHARDS_DIR="$(SHARDS_2CLASS)"
PID_ARGS_2    ?= --seeds 5
ENERGY_ARGS_2 ?= --seeds 5

.PHONY: check-shards-2class check-params-2class hits-2class report-figures mlp-2class
check-shards-2class:
	@test -d "$(SHARDS_2CLASS)" || { \
	  echo "missing shard lot: $(SHARDS_2CLASS)"; \
	  echo "set SHARDS_2CLASS=<dir> if the corrected lot lives elsewhere"; \
	  exit 1; }

check-params-2class: check-shards-2class
	$(TWO_CLASS) "$(PYTHON)" tools/build_2class_params.py --check-manifest

params-2class: check-shards-2class ## 2class-1. consolidate both lots: 200k -> training, discret -> evaluation
	$(TWO_CLASS) "$(PYTHON)" tools/build_2class_params.py
	$(TWO_CLASS) "$(PYTHON)" tools/build_2class_params.py --check

pid-2class: check-params-2class ## 2class-2. pion/proton classifier, trained on the 200k lot
	$(TWO_CLASS) "$(PYTHON)" $(PID_SCRIPT) $(PID_ARGS_2)

# The MLP takes one seed at a time, so the sweep is a loop rather than --seeds.
# The seeds match the BDT's convention (random_state + i) so the two spreads
# measure the same thing; a single run gives a number with no width, and the
# comparison in chapter 6 turns on differences smaller than that width.
MLP_SEEDS ?= 42 43 44 45 46

mlp-2class: check-params-2class ## 2class-2a. the same classifier as an MLP, one run per seed
	@for seed in $(MLP_SEEDS); do \
	  $(TWO_CLASS) "$(PYTHON)" analysis/3_pid/mlp/train.py --seed $$seed || exit 1; \
	done

energy-2class: check-params-2class ## 2class-3. regressors under the same containment cut as the chi2
	$(TWO_CLASS) "$(PYTHON)" $(ENERGY_SCRIPT) --require-contained $(ENERGY_ARGS_2)
	$(TWO_CLASS) "$(PYTHON)" $(ENERGY_SCRIPT) --particles mixed --require-contained
# The control for "the gain comes from the information, not from the target":
# the same regressor and the same features, only the target changes.
	$(TWO_CLASS) "$(PYTHON)" $(ENERGY_SCRIPT) --particles mixed --require-contained --ratio-to-chi2

interpret-2class: check-params-2class ## 2class-2b. correlation matrix and permutation importances
	$(TWO_CLASS) "$(PYTHON)" analysis/3_pid/bdt/interpret.py

chi2-2class: check-params-2class ## 2class-4. fit the calibration on the 200k lot, evaluate on the discret one
	$(TWO_CLASS) "$(PYTHON)" tools/fit_chi2_2class.py

# The containment cut goes to every arm or to none: the chi2 calibration is fitted
# under it, so evaluating it on the full sample would take it outside the domain
# it was fitted on. Train the regressors with --require-contained to match.
# Six scenarios, not five: --pid none --reco bdt is the inclusive regressor, the
# only one that answers whether the species label is worth anything at all to a
# model trained on the mixture. Leaving it out is what made the first reading of
# the coupling - that PID helps - rest on a comparison against a pion
# calibration applied to protons, which is a different claim.
coupling-2class: check-params-2class ## 2class-5. the six scenarios that make the report's figures
	$(TWO_CLASS) "$(PYTHON)" $(COUPLING_SCRIPT) --pid none --reco bdt  --require-contained $(COUPLING_ARGS)
	$(TWO_CLASS) "$(PYTHON)" $(COUPLING_SCRIPT) --pid bdt  --reco bdt  --require-contained $(COUPLING_ARGS)
	$(TWO_CLASS) "$(PYTHON)" $(COUPLING_SCRIPT) --pid true --reco bdt  --require-contained $(COUPLING_ARGS)
	$(TWO_CLASS) "$(PYTHON)" $(COUPLING_SCRIPT) --pid bdt  --reco chi2 --require-contained $(COUPLING_ARGS)
	$(TWO_CLASS) "$(PYTHON)" $(COUPLING_SCRIPT) --pid true --reco chi2 --require-contained $(COUPLING_ARGS)
	$(TWO_CLASS) "$(PYTHON)" $(COUPLING_SCRIPT) --pid none --reco chi2 --require-contained $(COUPLING_ARGS)

# Stops before the coupling on purpose: that stage reads the artifacts published
# under $SDHCAL_MODEL_DIR, and choosing which run and which seed feeds the
# comparison is a decision, not a default. Publish them, then run coupling-2class.
study-2class: params-2class pid-2class mlp-2class energy-2class chi2-2class ## the pion/proton chain up to the models
	@echo
	@echo "Trained. To run the coupling, publish the chosen artifacts, e.g.:"
	@echo "  cp output/pid_lgbm/models/lgbm_pid_<run>_seed42.joblib             models/pid_bdt.joblib"
	@echo "  cp output/energy_reco_lgbm/pi/models/lgbm_regressor_pi_<run>_seed42.joblib         models/energy_bdt_pi.joblib"
	@echo "  cp output/energy_reco_lgbm/proton/models/lgbm_regressor_proton_<run>_seed42.joblib models/energy_bdt_proton.joblib"
	@echo "  cp output/energy_reco_lgbm/mixed/models/lgbm_regressor_mixed_<run>_seed42.joblib   models/energy_bdt_mixed.joblib"
	@echo "then: make coupling-2class"

# ---------------------------------------------------------------------------
# Model benchmark
# ---------------------------------------------------------------------------
# Every model answers the same question about the same events. tools/predict.py
# writes one row per event, tools/compare_models.py turns those into the figures
# of merit, and the ROOT macro draws them. A new model joins the comparison as
# soon as it can write that table.

# These are the models the report compares, which is why cnn, gnn and gnn_ratio
# appear here: a default that quietly left them out would reproduce a different
# table from the published one. They reach the comparison through
# scripts/20_pid_hits.sh and scripts/30_energy_gnn.sh - the networks write their
# own predictions during training, so predict.py cannot produce them. Override
# either variable to compare a subset.
BENCH_PID    ?= mlp bdt cnn gnn
BENCH_ENERGY ?= bdt_ratio bdt_mixed gnn_ratio chi2

# The hit-level stages live in scripts/ rather than here: they are long enough
# to be run on their own, and they pin an environment of their own.
hits-2class: ## benchmark-0. the hit-level networks (hours; see scripts/)
	scripts/20_pid_hits.sh
	scripts/30_energy_gnn.sh

predictions-2class: check-params-2class ## benchmark-1. apply every saved model to the evaluation lot
	$(TWO_CLASS) "$(PYTHON)" tools/predict.py --kind pid --model models/pid_bdt.joblib --name bdt --require-contained
	$(TWO_CLASS) "$(PYTHON)" tools/predict.py --kind pid --model models/pid_mlp.joblib --name mlp --require-contained
	@for m in pi proton mixed ratio; do \
	  $(TWO_CLASS) "$(PYTHON)" tools/predict.py --kind energy \
	    --model models/energy_bdt_$$m.joblib --name bdt_$$m --require-contained || exit 1; \
	done
	$(TWO_CLASS) "$(PYTHON)" tools/predict.py --kind energy --chi2 --name chi2 --require-contained

benchmark-2class: ## benchmark-2. the comparison tables, the ROOT plots, the report figures
	$(TWO_CLASS) "$(PYTHON)" tools/compare_models.py --kind pid    --models $(BENCH_PID)
	$(TWO_CLASS) "$(PYTHON)" tools/compare_models.py --kind energy --models $(BENCH_ENERGY)
	$(TWO_CLASS) "$(PYTHON)" tools/coupling_curves.py
	$(TWO_CLASS) "$(PYTHON)" tools/timing_ablation.py
	"$(ROOT)" -l -b -q 'analysis/6_benchmark/plots.C("pid")'
	"$(ROOT)" -l -b -q 'analysis/6_benchmark/plots.C("energy")'
	"$(ROOT)" -l -b -q 'analysis/6_benchmark/plots.C("coupling")'
	"$(ROOT)" -l -b -q 'analysis/6_benchmark/plots.C("ablation")'
	"$(ROOT)" -l -b -q 'analysis/6_benchmark/plots.C("scan")'
	"$(ROOT)" -l -b -q 'analysis/6_benchmark/interpret_plots.C'
	$(MAKE) report-figures

# The renaming is what the LaTeX expects; it was a manual cp until it was not.
report-figures: ## benchmark-3. copy the benchmark plots under the names the report uses
	@cp output/benchmark/pid/plots/pid_rejection.png            report/Fig/bench_pid_rejection.png
	@cp output/benchmark/pid/plots/pid_efficiency_vs_energy.png report/Fig/bench_pid_efficiency.png
	@cp output/benchmark/energy/plots/energy_resolution.png     report/Fig/bench_energy_resolution.png
	@cp output/benchmark/energy/plots/energy_linearity.png      report/Fig/bench_energy_linearity.png
	@cp output/benchmark/coupling/plots/species_linearity_pid-none_reco-chi2.png       report/Fig/bench_species_linearity_chi2.png
	@cp output/benchmark/coupling/plots/species_resolution_pid-none_reco-chi2.png      report/Fig/bench_species_resolution_chi2.png
	@cp output/benchmark/coupling/plots/species_linearity_pid-none_reco-bdt.png        report/Fig/bench_species_linearity_bdt.png
	@cp output/benchmark/coupling/plots/species_resolution_pid-none_reco-bdt.png       report/Fig/bench_species_resolution_bdt.png
	@cp output/benchmark/coupling/plots/species_linearity_pid-bdt_reco-chi2.png        report/Fig/bench_species_linearity_chi2_pid.png
	@cp output/benchmark/coupling/plots/species_resolution_pid-bdt_reco-chi2.png       report/Fig/bench_species_resolution_chi2_pid.png
	@cp output/benchmark/coupling/plots/coupling_linearity_bdt_all.png                 report/Fig/bench_coupling_linearity_bdt.png
	@cp output/benchmark/coupling/plots/coupling_resolution_bdt_all.png                report/Fig/bench_coupling_resolution_bdt.png
	@cp output/benchmark/coupling/plots/coupling_linearity_chi2_all.png                report/Fig/bench_coupling_linearity_chi2.png
	@cp output/benchmark/coupling/plots/coupling_resolution_chi2_all.png               report/Fig/bench_coupling_resolution_chi2.png
	@cp output/benchmark/coupling/plots/coupling_linearity_chi2_p.png                  report/Fig/bench_coupling_linearity_chi2_proton.png
	@cp output/benchmark/interpret/plots/confusion_matrix.png                          report/Fig/bench_confusion_matrix.png
	@cp output/benchmark/interpret/plots/correlation_matrix.png                        report/Fig/bench_correlation_matrix.png
	@cp output/benchmark/ablation/plots/ablation_difference.png report/Fig/bench_ablation_difference.png
	@cp output/benchmark/ablation/plots/ablation_resolution.png  report/Fig/bench_ablation_resolution.png
	@cp output/benchmark/scan/plots/timing_scan_a.png report/Fig/bench_timing_scan_a.png
	@cp output/benchmark/scan/plots/timing_scan_b.png report/Fig/bench_timing_scan_b.png
	@cp output/benchmark/scan/plots/timing_scan_band.png report/Fig/bench_timing_scan_band.png
	@echo "report/Fig updated"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
holdout: ## build the held-out evaluation sample and verify it is disjoint
	"$(PYTHON)" tools/build_holdout.py --write-train-pool
	"$(PYTHON)" tools/build_holdout.py --check

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------
clean-pyc: ## remove __pycache__ directories
	find . -name '__pycache__' -not -path './.git/*' -not -path './analysis/archive/*' -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -not -path './.git/*' -not -path './analysis/archive/*' -delete

clean: clean-pyc ## remove build products
	rm -f $(FEATURES_BIN) $(ENRICH_BIN) $(CPP_TEST_BIN)
	rm -rf .pytest_cache .ruff_cache

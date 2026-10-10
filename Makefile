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

# Works in a fresh checkout, before `pip install -e .`.
export PYTHONPATH := $(CURDIR)/src:$(PYTHONPATH)

CXX       ?= g++
CXXFLAGS  ?= -std=c++17 -O2 -Wall -Wextra
ROOT_CFLAGS := $(shell $(ROOT_CONFIG) --cflags 2>/dev/null)
ROOT_LIBS   := $(shell $(ROOT_CONFIG) --libs 2>/dev/null)
CPP_INCLUDE := -I cpp/include

PY_SOURCES  := $(shell find . -name '*.py' -not -path './.git/*' -not -path './$(VENV)/*' -not -path './analysis/archive/*')
CPP_SOURCES := cpp/src/ShowerAnalyzer.cpp \
               analysis/1_shower_features/extract.cpp \
               analysis/1_shower_features/extract_parallel.cpp \
               tools/count_tracks_ransac.cpp
ROOT_MACROS := analysis/4_energy/chi2_tminuit/fit.C

# Stage scripts
PID_SCRIPT      := analysis/3_pid/bdt/train.py
ENERGY_SCRIPT   := analysis/4_energy/ml_bdt/train.py
COUPLING_SCRIPT := analysis/5_pid_energy/study.py
FEATURES_BIN    := analysis/1_shower_features/extract

# Arguments you will want to override, e.g. `make pid PID_ARGS="--no-timing"`
PID_ARGS      ?= --seeds 5
ENERGY_ARGS   ?= --particles all --seeds 5
COUPLING_ARGS ?=
PARTICLES     ?= pi kaon proton

.DEFAULT_GOAL := help
.PHONY: help venv test check compile cpp lint \
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
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt
	$(VENV_PYTHON) -m pip install -e .
	@echo "Activate with: source $(VENV)/bin/activate"

test: ## run the test suite
	$(PYTHON) -m pytest tests/ -q

compile: ## byte-compile every Python file, catching syntax errors
	@failed=0; for f in $(PY_SOURCES); do \
	  $(PYTHON) -m py_compile "$$f" 2>/dev/null || { echo "SYNTAX ERROR: $$f"; failed=1; }; \
	done; \
	find . -name '__pycache__' -not -path './.git/*' -not -path './analysis/archive/*' -exec rm -rf {} + 2>/dev/null; \
	if [ $$failed -eq 0 ]; then echo "all Python sources compile"; else exit 1; fi

cpp: ## syntax-check the C++ sources and the ROOT macros
	@if [ -z "$(ROOT_CFLAGS)" ]; then \
	  echo "root-config not found: skipping the C++ check"; exit 0; \
	fi; \
	for f in $(CPP_SOURCES); do \
	  echo "  $$f"; \
	  $(CXX) $(CXXFLAGS) -fsyntax-only $(ROOT_CFLAGS) $(CPP_INCLUDE) "$$f" || exit 1; \
	done; \
	for m in $(ROOT_MACROS); do \
	  echo "  $$m"; \
	  printf '#include "%s/%s"\nint main(){return 0;}\n' "$(CURDIR)" "$$m" > /tmp/sdhcal_macro_check.cpp; \
	  $(CXX) -std=c++17 -fsyntax-only $(ROOT_CFLAGS) /tmp/sdhcal_macro_check.cpp || exit 1; \
	done; \
	rm -f /tmp/sdhcal_macro_check.cpp; \
	echo "all C++ sources compile"

lint: ## run ruff over the package, the tools and the tests
	ruff check src tools tests
	ruff format --check src tests

check: compile test cpp ## everything that can be verified without the data

# ---------------------------------------------------------------------------
# 1. Shower features
# ---------------------------------------------------------------------------
$(FEATURES_BIN): cpp/src/ShowerAnalyzer.cpp analysis/1_shower_features/extract.cpp cpp/include/ShowerAnalyzer.h
	@if [ -z "$(ROOT_CFLAGS)" ]; then echo "root-config not found"; exit 1; fi
	$(CXX) $(CXXFLAGS) $(ROOT_CFLAGS) $(CPP_INCLUDE) \
	  analysis/1_shower_features/extract.cpp cpp/src/ShowerAnalyzer.cpp \
	  -o $@ $(ROOT_LIBS)
	@echo "built $@"

features: $(FEATURES_BIN) ## 1. extract the shower parameters of every particle
	./$(FEATURES_BIN)

# ---------------------------------------------------------------------------
# 2. Shower-variable comparison
# ---------------------------------------------------------------------------
variables: ## 2. compare the shower variables between species
	@cd analysis/2_variable_study && for macro in begin.C density.C em_fraction.C radius.C thr3.C; do \
	  echo "  $$macro"; \
	  $(ROOT) -l -b -q "$$macro" || exit 1; \
	done

# ---------------------------------------------------------------------------
# 3. Particle identification
# ---------------------------------------------------------------------------
pid: ## 3. train and evaluate the hadron classifier
	$(PYTHON) $(PID_SCRIPT) $(PID_ARGS)

# ---------------------------------------------------------------------------
# 4. Energy reconstruction
# ---------------------------------------------------------------------------
energy: energy-ml energy-chi2 ## 4. both energy reconstructions

energy-ml: ## 4a. LightGBM regressors, per particle and inclusive
	$(PYTHON) $(ENERGY_SCRIPT) $(ENERGY_ARGS)
	$(PYTHON) $(ENERGY_SCRIPT) --particles mixed

energy-chi2: ## 4b. semi-digital chi2 calibration, then publish the coefficients
	@for p in $(PARTICLES); do \
	  echo "  chi2 fit: $$p"; \
	  $(ROOT) -l -b -q "analysis/4_energy/chi2_tminuit/fit.C(\"$$p\")" || exit 1; \
	done
	$(MAKE) calibration

calibration: ## merge the chi2 fit fragments into the shared calibration file
	$(PYTHON) tools/update_calibration.py

# ---------------------------------------------------------------------------
# 5. PID-energy coupling
# ---------------------------------------------------------------------------
coupling: ## 5. the three cases that belong in the same table
	$(PYTHON) $(COUPLING_SCRIPT) --pid none --reco bdt $(COUPLING_ARGS)
	$(PYTHON) $(COUPLING_SCRIPT) --pid true --reco bdt $(COUPLING_ARGS)
	$(PYTHON) $(COUPLING_SCRIPT) --pid bdt  --reco bdt $(COUPLING_ARGS)

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
# Everything below sets SDHCAL_PARAMS_SET=2class, which is what redirects the
# stages at the consolidated files - see src/sdhcal/paths.py.

TWO_CLASS     := SDHCAL_PARAMS_SET=2class
PID_ARGS_2    ?= --seeds 5
ENERGY_ARGS_2 ?= --seeds 5

params-2class: ## 2class-1. consolidate both lots: 200k -> training, discret -> evaluation
	$(PYTHON) tools/build_2class_params.py
	$(PYTHON) tools/build_2class_params.py --check

pid-2class: ## 2class-2. pion/proton classifier, trained on the 200k lot
	$(TWO_CLASS) $(PYTHON) $(PID_SCRIPT) $(PID_ARGS_2)

energy-2class: ## 2class-3. regressors under the same containment cut as the chi2
	$(TWO_CLASS) $(PYTHON) $(ENERGY_SCRIPT) --require-contained $(ENERGY_ARGS_2)
	$(TWO_CLASS) $(PYTHON) $(ENERGY_SCRIPT) --particles mixed --require-contained

interpret-2class: ## 2class-2b. correlation matrix and permutation importances
	$(TWO_CLASS) $(PYTHON) analysis/3_pid/bdt/interpret.py

chi2-2class: ## 2class-4. fit the calibration on the 200k lot, evaluate on the discret one
	$(TWO_CLASS) $(PYTHON) tools/fit_chi2_2class.py

# The containment cut goes to every arm or to none: the chi2 calibration is fitted
# under it, so evaluating it on the full sample would take it outside the domain
# it was fitted on. Train the regressors with --require-contained to match.
coupling-2class: ## 2class-5. the five scenarios that make the report's figures
	$(TWO_CLASS) $(PYTHON) $(COUPLING_SCRIPT) --pid bdt  --reco bdt  --require-contained $(COUPLING_ARGS)
	$(TWO_CLASS) $(PYTHON) $(COUPLING_SCRIPT) --pid true --reco bdt  --require-contained $(COUPLING_ARGS)
	$(TWO_CLASS) $(PYTHON) $(COUPLING_SCRIPT) --pid bdt  --reco chi2 --require-contained $(COUPLING_ARGS)
	$(TWO_CLASS) $(PYTHON) $(COUPLING_SCRIPT) --pid true --reco chi2 --require-contained $(COUPLING_ARGS)
	$(TWO_CLASS) $(PYTHON) $(COUPLING_SCRIPT) --pid none --reco chi2 --require-contained $(COUPLING_ARGS)

# Stops before the coupling on purpose: that stage reads the artifacts published
# under $SDHCAL_MODEL_DIR, and choosing which run and which seed feeds the
# comparison is a decision, not a default. Publish them, then run coupling-2class.
study-2class: params-2class pid-2class energy-2class chi2-2class ## the pion/proton chain up to the models
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

BENCH_PID    ?= bdt mlp
BENCH_ENERGY ?= bdt_mixed bdt_pi bdt_proton chi2

predictions-2class: ## benchmark-1. apply every trained model to the evaluation lot
	$(TWO_CLASS) $(PYTHON) tools/predict.py --kind pid --model models/pid_bdt.joblib --name bdt --require-contained
	$(TWO_CLASS) $(PYTHON) tools/predict.py --kind pid --model models/pid_mlp.joblib --name mlp --require-contained
	@for m in pi proton mixed; do \
	  $(TWO_CLASS) $(PYTHON) tools/predict.py --kind energy \
	    --model models/energy_bdt_$$m.joblib --name bdt_$$m --require-contained || exit 1; \
	done
	$(TWO_CLASS) $(PYTHON) tools/predict.py --kind energy --chi2 --name chi2 --require-contained

benchmark-2class: ## benchmark-2. the comparison tables and the ROOT plots
	$(TWO_CLASS) $(PYTHON) tools/compare_models.py --kind pid    --models $(BENCH_PID)
	$(TWO_CLASS) $(PYTHON) tools/compare_models.py --kind energy --models $(BENCH_ENERGY)
	$(ROOT) -l -b -q 'analysis/6_benchmark/plots.C("pid")'
	$(ROOT) -l -b -q 'analysis/6_benchmark/plots.C("energy")'

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
holdout: ## build the held-out evaluation sample and verify it is disjoint
	$(PYTHON) tools/build_holdout.py --write-train-pool
	$(PYTHON) tools/build_holdout.py --check

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------
clean-pyc: ## remove __pycache__ directories
	find . -name '__pycache__' -not -path './.git/*' -not -path './analysis/archive/*' -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -not -path './.git/*' -not -path './analysis/archive/*' -delete

clean: clean-pyc ## remove build products
	rm -f $(FEATURES_BIN) analysis/1_shower_features/extract_parallel
	rm -rf .pytest_cache .ruff_cache

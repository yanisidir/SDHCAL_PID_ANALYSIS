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

PY_SOURCES  := $(shell find . -name '*.py' -not -path './.git/*' -not -path './$(VENV)/*')
CPP_SOURCES := cpp/src/ShowerAnalyzer.cpp \
               analysis/1_shower_features/computeParams.cpp \
               analysis/1_shower_features/computeParams_parallel.cpp \
               tools/count_tracks_ransac.cpp
ROOT_MACROS := analysis/4_energy/chi2_tminuit/EnergyReco.C

# Stage scripts
PID_SCRIPT      := analysis/3_pid/bdt/LGBM_classifier_PID.py
ENERGY_SCRIPT   := analysis/4_energy/ml_bdt/hadron_energy_reco_lgbm.py
COUPLING_SCRIPT := analysis/5_pid_energy/pid_energy_coupling.py
FEATURES_BIN    := analysis/1_shower_features/computeParams

# Arguments you will want to override, e.g. `make pid PID_ARGS="--no-timing"`
PID_ARGS      ?= --seeds 5
ENERGY_ARGS   ?= --particles all --seeds 5
COUPLING_ARGS ?=
PARTICLES     ?= pi kaon proton

.DEFAULT_GOAL := help
.PHONY: help venv test check compile cpp lint \
        features variables pid energy energy-ml energy-chi2 coupling \
        calibration holdout clean clean-pyc

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
	find . -name '__pycache__' -not -path './.git/*' -exec rm -rf {} + 2>/dev/null; \
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
$(FEATURES_BIN): cpp/src/ShowerAnalyzer.cpp analysis/1_shower_features/computeParams.cpp cpp/include/ShowerAnalyzer.h
	@if [ -z "$(ROOT_CFLAGS)" ]; then echo "root-config not found"; exit 1; fi
	$(CXX) $(CXXFLAGS) $(ROOT_CFLAGS) $(CPP_INCLUDE) \
	  analysis/1_shower_features/computeParams.cpp cpp/src/ShowerAnalyzer.cpp \
	  -o $@ $(ROOT_LIBS)
	@echo "built $@"

features: $(FEATURES_BIN) ## 1. extract the shower parameters of every particle
	./$(FEATURES_BIN)

# ---------------------------------------------------------------------------
# 2. Shower-variable comparison
# ---------------------------------------------------------------------------
variables: ## 2. compare the shower variables between species
	@cd analysis/2_variable_study && for macro in compare*.C; do \
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
	  $(ROOT) -l -b -q "analysis/4_energy/chi2_tminuit/EnergyReco.C(\"$$p\")" || exit 1; \
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
# Data
# ---------------------------------------------------------------------------
holdout: ## build the held-out evaluation sample and verify it is disjoint
	$(PYTHON) tools/build_holdout.py --write-train-pool
	$(PYTHON) tools/build_holdout.py --check

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------
clean-pyc: ## remove __pycache__ directories
	find . -name '__pycache__' -not -path './.git/*' -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -not -path './.git/*' -delete

clean: clean-pyc ## remove build products
	rm -f $(FEATURES_BIN) analysis/1_shower_features/computeParams_parallel
	rm -rf .pytest_cache .ruff_cache

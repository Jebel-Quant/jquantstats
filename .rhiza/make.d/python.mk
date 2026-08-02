## .rhiza/make.d/python.mk - the Python language layer (bundle: python-core)
#
# Everything in rhiza that only makes sense because the project is written in
# Python. `core` provides the make framework and uv/uvx as a tool runner; this
# file turns that into a Python project: the virtualenv, the `install` that syncs
# it, dependency and licence analysis of the declared dependencies, and the `all`
# aggregate naming the gates.
#
# A sibling language layer (rust.mk, from a rust-core bundle) ships the same
# *target names* — install, all — with different recipes. That contract is the
# reason book.mk, test.mk and the CI workflows can call `make install` without
# knowing the language. Only one language layer is ever synced into a repo.

# Declare phony targets (they don't produce files)
.PHONY: all deptry install license rhiza-test

# The project virtualenv, and the interpreter that fills it. PYTHON_VERSION is
# declared in rhiza.mk (core needs a Python to run its own tooling on); here
# `.python-version` — which this bundle ships — makes it the project's version too.
VENV ?= .venv
UV_SYNC_ARGS ?= --all-extras --all-groups

export UV_VENV_CLEAR := 1

# Configurable list of licenses that fail the compliance scan (semicolon-separated)
LICENSE_FAIL_ON ?= GPL;LGPL;AGPL

##@ Python
install: pre-install install-uv ## install
	# Create the virtual environment only if it doesn't exist
	@if [ ! -d "${VENV}" ]; then \
	  ${UV_BIN} venv $(if $(PYTHON_VERSION),--python $(PYTHON_VERSION)) ${VENV} || { printf "${RED}[ERROR] Failed to create virtual environment${RESET}\n"; exit 1; }; \
	else \
	  printf "${BLUE}[INFO] Using existing virtual environment at ${VENV}, skipping creation${RESET}\n"; \
	fi

	# Install the dependencies from pyproject.toml (if it exists).
	# --inexact leaves packages uv did not manage in place instead of pruning them each
	# run, so repeated 'make' targets don't churn the environment. Per-target tooling
	# (pytest, interrogate, mutmut, ...) is provisioned on the fly via `uv run --with`
	# in the individual targets, so there is no separate dependency-install step here.
	@if [ -f "pyproject.toml" ]; then \
	  if [ -f "uv.lock" ]; then \
	    if ! ${UV_BIN} lock --check >/dev/null 2>&1; then \
	      printf "${YELLOW}[WARN] uv.lock is out of sync with pyproject.toml${RESET}\n"; \
	      printf "${YELLOW}       Run 'uv sync' to update your lock file and environment${RESET}\n"; \
	      printf "${YELLOW}       Or run 'uv lock' to update only the lock file${RESET}\n"; \
	      exit 1; \
	    fi; \
	    printf "${BLUE}[INFO] Installing dependencies from lock file${RESET}\n"; \
	    ${UV_BIN} sync $(UV_SYNC_ARGS) --inexact --frozen || { printf "${RED}[ERROR] Failed to install dependencies${RESET}\n"; exit 1; }; \
	  else \
	    printf "${YELLOW}[WARN] uv.lock not found. Generating lock file and installing dependencies...${RESET}\n"; \
	    ${UV_BIN} sync $(UV_SYNC_ARGS) --inexact || { printf "${RED}[ERROR] Failed to install dependencies${RESET}\n"; exit 1; }; \
	  fi; \
	else \
	  printf "${YELLOW}[WARN] No pyproject.toml found, skipping install${RESET}\n"; \
	fi

	# Install pre-commit hooks (skip when core.hooksPath is set, e.g. by an
	# external hook manager — pre-commit refuses to install in that case)
	@if [ -f ".pre-commit-config.yaml" ]; then \
	  if [ -n "$$(git config --get core.hooksPath 2>/dev/null)" ]; then \
	    printf "${BLUE}[INFO] Skipping pre-commit hook install: core.hooksPath is set${RESET}\n"; \
	  else \
	    printf "${BLUE}[INFO] Installing pre-commit hooks...${RESET}\n"; \
	    ${UVX_BIN} -p ${PYTHON_VERSION} pre-commit install || { printf "${YELLOW}[WARN] Failed to install pre-commit hooks${RESET}\n"; }; \
	  fi; \
	fi

	@$(MAKE) post-install
	
	# Display success message with activation instructions
	@printf "\n${GREEN}[SUCCESS] Installation complete!${RESET}\n\n"
	@printf "${BLUE}To activate the virtual environment, run:${RESET}\n"
	@printf "${YELLOW}  source ${VENV}/bin/activate${RESET}\n\n"

all: fmt deptry test docs-coverage security license typecheck rhiza-test ## run all CI targets locally

# deptry scans one or more folders for dependency issues. Each feature bundle
# contributes the folders it owns to DEPTRY_FOLDERS (and any per-folder ignores
# to DEPTRY_IGNORE), so this target never needs to know which bundles are
# present. The language layer itself contributes SOURCE_FOLDER when it exists; see e.g.
# marimo.mk for a bundle that appends its own folder. Rhiza's own test folder
# (.rhiza/tests) is deliberately excluded: its tooling is provisioned on the fly
# via `uv run --with` in the individual targets, not declared in the project's
# pyproject, so deptry (which validates against pyproject) would only emit noise
# for it.
DEPTRY_FOLDERS ?=
DEPTRY_IGNORE ?=
ifneq ($(wildcard $(SOURCE_FOLDER)),)
DEPTRY_FOLDERS += $(SOURCE_FOLDER)
endif

deptry: install-uv ## Run deptry over the folders contributed by each bundle
	@if [ -n "$(strip $(DEPTRY_FOLDERS))" ]; then \
		printf "${BLUE}[INFO] Running deptry on:${RESET} $(strip $(DEPTRY_FOLDERS))\n"; \
		$(UVX_BIN) -p ${PYTHON_VERSION} deptry $(strip $(DEPTRY_FOLDERS) $(DEPTRY_IGNORE)); \
	else \
		printf "${YELLOW}[WARN] no deptry folders found, skipping.${RESET}\n"; \
	fi

license: install ## run license compliance scan (fail on GPL, LGPL, AGPL)
	@printf "${BLUE}[INFO] Running license compliance scan...${RESET}\n"
	@${UV_BIN} run --with pip-licenses pip-licenses --fail-on="${LICENSE_FAIL_ON}"

rhiza-test: install ## run rhiza's own tests (if any)
	@if [ -d ".rhiza/tests" ]; then \
		${UV_BIN} run --with pytest --with pytest-timeout --with python-dotenv --with packaging pytest .rhiza/tests; \
	else \
		printf "${YELLOW}[WARN] No .rhiza/tests directory found, skipping rhiza-tests${RESET}\n"; \
	fi

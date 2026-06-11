## .rhiza/make.d/custom-env.mk - Custom Environment Configuration
# This file example shows how to set variables for the project.

# Custom variables for this repository
PROJECT_NAME_EXTRA := Rhiza Platform
LOG_LEVEL ?= INFO

# Coverage gate: the suite is at 100 % — keep it there (template default is 90).
# Lives here (not in the root Makefile or .rhiza/.env) because the rhiza API
# tests copy those files into their sandbox and assert the template default.
# Uses ?= so a root-Makefile or CLI override still wins, as the template
# customization contract requires.
COVERAGE_FAIL_UNDER ?= 100

# Overriding core variables (be careful)
# VENV := .venv_custom

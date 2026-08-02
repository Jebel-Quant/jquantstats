## Makefile (repo-owned)
# Keep this file small. It can be edited without breaking template sync.

LOGO_FILE=.rhiza/assets/rhiza-logo.svg

# Override template default: include mkdocstrings plugin for API docs
MKDOCS_EXTRA_PACKAGES = --with 'mkdocstrings[python]'

# Always include the Rhiza API (template-managed)
include .rhiza/rhiza.mk

# Architectural import contracts (import-linter): the analytics subpackages
# (_stats, _plots, _reports, _utils) annotate against the structural Protocols in
# _protocol.py and must never import the concrete Data / Portfolio at runtime.
# Contracts live in pyproject.toml under [tool.importlinter]. Hooked into `test`
# (a double-colon rule) so a layering inversion fails CI, not just review.
.PHONY: arch
arch: install ## enforce layer boundaries with import-linter (accessors ⇏ Data/Portfolio)
	@printf "${BLUE}[INFO] Checking architectural import contracts (import-linter)${RESET}\n"
	@${UV_BIN} run --group lint lint-imports

test:: arch

# Optional: developer-local extensions (not committed)
-include local.mk

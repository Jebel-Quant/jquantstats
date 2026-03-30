## Makefile (repo-owned)
# Keep this file small. It can be edited without breaking template sync.

DOCFORMAT=google
DEFAULT_AI_MODEL=claude-sonnet-4.5
LOGO_FILE=.rhiza/assets/rhiza-logo.svg
GH_AW_ENGINE ?= copilot  # Default AI engine for gh-aw workflows (copilot, claude, or codex)

# Always include the Rhiza API (template-managed)
include .rhiza/rhiza.mk

# Optional: developer-local extensions (not committed)
-include local.mk

# Wire typecheck into make validate
post-validate::
	@$(MAKE) typecheck

## pdoc: generate standalone API reference into _book/api
.PHONY: pdoc
pdoc: install ## generate pdoc API reference
	@printf "${BLUE}[INFO] Generating pdoc API reference...${RESET}\n"
	@${UVX_BIN} pdoc src/jquantstats -o $(BOOK_OUTPUT)/api
	@printf "${GREEN}[OK] pdoc output → $(BOOK_OUTPUT)/api${RESET}\n"

## hook into book to run pdoc after mkdocs builds _book/
book:: ## include pdoc API reference in the book
	@$(MAKE) pdoc

## Custom targets

##@ Quality

.PHONY: test-kaleido
test-kaleido: install ## run kaleido static image export tests (requires jquantstats[plot])
	@printf "${BLUE}[INFO] Running kaleido static image export tests...${RESET}\n"
	@${UV_BIN} run pytest -m kaleido -v tests/

.PHONY: changelog
changelog: ## generate/update CHANGELOG.md from git history using git-cliff
	@printf "${BLUE}[INFO] Generating CHANGELOG.md with git-cliff...${RESET}\n"
	@${UVX_BIN} git-cliff --config .github/cliff.toml --output CHANGELOG.md
	@printf "${GREEN}[OK] CHANGELOG.md updated.${RESET}\n"

# Override the rhiza template's security target to ignore CVE-2026-4539
# (ReDoS in pygments AdlLexer). Fix requires pygments>=3.3, which is not
# yet published on PyPI. Remove this override once pygments 3.3 is available.
.PHONY: security
security: install ## run security scans (pip-audit and bandit)
	@printf "${BLUE}[INFO] Running pip-audit for dependency vulnerabilities...${RESET}\n"
	@${UVX_BIN} pip-audit --ignore-vuln CVE-2026-4539
	@printf "${BLUE}[INFO] Running bandit security scan...${RESET}\n"
	@${UVX_BIN} bandit -r ${SOURCE_FOLDER} -ll -q -c pyproject.toml

.PHONY: semgrep
semgrep: install ## run Semgrep static analysis (numpy rules)
	@printf "${BLUE}[INFO] Running Semgrep (numpy rules)...${RESET}\n"
	@if [ -d ${SOURCE_FOLDER} ]; then \
		${UVX_BIN} semgrep --config .github/semgrep.yml ${SOURCE_FOLDER}; \
	else \
		printf "${YELLOW}[WARN] SOURCE_FOLDER '${SOURCE_FOLDER}' not found, skipping semgrep.${RESET}\n"; \
	fi

.PHONY: license
license: install ## run license compliance scan (fail on GPL, LGPL, AGPL) and generate LICENSES.md
	@printf "${BLUE}[INFO] Running license compliance scan...${RESET}\n"
	@${UV_BIN} run --with pip-licenses pip-licenses --fail-on="GPL;LGPL;AGPL"
	@printf "${BLUE}[INFO] Generating LICENSES.md dependency report...${RESET}\n"
	@${UV_BIN} run --with pip-licenses pip-licenses --format=markdown --output-file=LICENSES.md
	@printf "${GREEN}[OK] LICENSES.md generated.${RESET}\n"

.PHONY: adr
adr: install ## Create a new Architecture Decision Record (ADR) using AI assistance
	@echo "Creating a new ADR..."
	@echo "This will trigger the adr-create workflow."
	@echo ""
	@read -p "Enter ADR title (e.g., 'Use PostgreSQL for data storage'): " title; \
	echo ""; \
	read -p "Enter brief context (optional, press Enter to skip): " context; \
	echo ""; \
	if [ -z "$$title" ]; then \
		echo "Error: Title is required"; \
		exit 1; \
	fi; \
	if [ -z "$$context" ]; then \
		gh workflow run adr-create.md -f title="$$title"; \
	else \
		gh workflow run adr-create.md -f title="$$title" -f context="$$context"; \
	fi; \
	echo ""; \
	echo "✅ ADR creation workflow triggered!"; \
	echo ""; \
	echo "The workflow will:"; \
	echo "  1. Generate the next ADR number"; \
	echo "  2. Create a comprehensive ADR document"; \
	echo "  3. Update the ADR index"; \
	echo "  4. Open a pull request for review"; \
	echo ""; \
	echo "Check workflow status: gh run list --workflow=adr-create.md"; \
	echo "View latest run: gh run view"


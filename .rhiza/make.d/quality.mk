## .rhiza/make.d/quality.mk - Quality and Formatting
# The language-neutral gates: pre-commit, the TODO sweep, and semgrep. Everything
# that needs to know how the project declares its dependencies — `deptry`, the
# licence-compliance scan — and the `all` aggregate that names the per-language
# gates live in the language layer (python.mk, from the python-core bundle).

# Declare phony targets (they don't produce files)
.PHONY: fmt todos semgrep

##@ Quality and Formatting
fmt: install-uv ## check the pre-commit hooks and the linting
	@${UVX_BIN} -p ${PYTHON_VERSION} pre-commit run --all-files

todos: ## search and report all TODO/FIXME/HACK comments in the codebase
	@printf "${BLUE}[INFO] Searching for TODO, FIXME, and HACK comments...${RESET}\n"
	@printf "${BOLD}Found the following items:${RESET}\n\n"
	@find . -type f \( -name "*.py" -o -name "*.mk" -o -name "*.sh" -o -name "*.md" -o -name "*.yml" -o -name "*.yaml" \) \
		-not -path "./.venv/*" \
		-not -path "./.git/*" \
		-not -path "./node_modules/*" \
		-not -path "./.tox/*" \
		-not -path "./build/*" \
		-not -path "./dist/*" \
		-print0 | xargs -0 grep -nHE "(TODO|FIXME|HACK):" 2>/dev/null | \
		grep -v "make todos" | \
		awk -F: '{ printf "${YELLOW}%s${RESET}:${GREEN}%s${RESET}: %s\n", $$1, $$2, substr($$0, index($$0,$$3)) }' || \
		printf "${GREEN}[SUCCESS] No TODO/FIXME/HACK comments found!${RESET}\n"
	@printf "\n${BLUE}[INFO] Search complete.${RESET}\n"

semgrep: install ## run Semgrep static analysis
	@printf "${BLUE}[INFO] Running Semgrep...${RESET}\n"
	@if [ -d ${SOURCE_FOLDER} ]; then \
		${UVX_BIN} semgrep --config .rhiza/semgrep.yml ${SOURCE_FOLDER}; \
	else \
		printf "${YELLOW}[WARN] SOURCE_FOLDER '${SOURCE_FOLDER}' not found, skipping semgrep.${RESET}\n"; \
	fi

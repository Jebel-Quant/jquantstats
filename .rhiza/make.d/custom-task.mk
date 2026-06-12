## .rhiza/make.d/custom-task.mk - Custom Repository Tasks
# This file example shows how to add new targets.

.PHONY: hello-rhiza mutation-gate

##@ Custom Tasks
hello-rhiza: ## a custom greeting task
	@printf "${GREEN}[INFO] Hello from the customised Rhiza project!${RESET}\n"

# Gated mutation testing: runs mutmut per module with a focused test runner
# and fails when survivors exceed the triaged ceilings in
# tests/mutation/baseline.json. See bin/mutation_gate.py for details.
mutation-gate: install ## run mutation tests gated against the triaged baseline
	@printf "${BLUE}[INFO] Running gated mutation tests...${RESET}\n"
	@${UV_BIN} run python bin/mutation_gate.py

# Adding logic to existing hooks
post-install:: ## run custom logic after core install
	@printf "${BLUE}[INFO] Running custom post-install steps...${RESET}\n"

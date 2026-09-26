# AI Coding Harness — evaluator entry points.
# The API key is read ONLY from the AI_API_KEY environment variable; it is never written here.
# Recipe lines are indented with TABs.

VENV := .venv
PY   := $(VENV)/bin/python

# First interpreter on PATH that is Python 3.10+ (litellm supports 3.10-3.14).
PYTHON ?= $(shell for p in python3.12 python3.11 python3.10 python3.13 python3.14 python3; do \
	command -v $$p >/dev/null 2>&1 && $$p -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,15) else 1)' 2>/dev/null && { echo $$p; break; }; \
	done)

# Optional inputs for non-interactive runs: make run REPO=<path|url> ISSUE_FILE=<file>
RUN_ARGS :=
ifneq ($(strip $(REPO)),)
RUN_ARGS += --repo "$(REPO)"
endif
ifneq ($(strip $(ISSUE_FILE)),)
RUN_ARGS += --issue-file "$(ISSUE_FILE)"
endif

.PHONY: setup run test clean ping demo eval check-key

setup:
	@test -n "$(PYTHON)" || { echo "ERROR: Python 3.10-3.14 is required (python3 on PATH is too old or missing)." >&2; exit 1; }
	@echo "Using $(PYTHON) ($$($(PYTHON) --version))"
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip -q
	$(PY) -m pip install -r requirements.txt -q
	@$(PY) -m harness --self-check
	@echo "Setup complete. Export AI_API_KEY, then run: make run"

check-key:
	@test -n "$$AI_API_KEY" || { echo 'ERROR: AI_API_KEY is not set. Run: export AI_API_KEY="<key>"' >&2; exit 1; }
	@test -x $(PY) || { echo "ERROR: environment not installed. Run: make setup" >&2; exit 1; }

run: check-key
	@$(PY) -m harness $(RUN_ARGS)

test:
	@test -x $(PY) || { echo "ERROR: environment not installed. Run: make setup" >&2; exit 1; }
	$(PY) -m pytest

ping: check-key
	@$(PY) -m harness --ping

demo: check-key
	@$(PY) -m harness --demo

eval: check-key
	@$(PY) scripts/eval.py $(EVAL_ARGS)

clean:
	rm -rf $(VENV) runs .pytest_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete

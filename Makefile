# AI Coding Harness — evaluator entry points.
# The API key is read from the environment (AI_API_KEY, or the provider's own variable such as GEMINI_API_KEY);
# it is never written here. Choose the model with MODEL=<preset|provider/model>, e.g. make run MODEL=gemini.
# Recipe lines are indented with TABs.

VENV := .venv
PY   := $(VENV)/bin/python

# First interpreter on PATH that is Python 3.10+ (litellm supports 3.10-3.14).
PYTHON ?= $(shell for p in python3.12 python3.11 python3.10 python3.13 python3.14 python3; do \
	command -v $$p >/dev/null 2>&1 && $$p -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,15) else 1)' 2>/dev/null && { echo $$p; break; }; \
	done)

# Model choice for run, ping, demo and eval: a preset from config.yaml (groq, gemini, openai, qwen, ollama, ...)
# or any LiteLLM "<provider>/<model>". Empty = model.name in config.yaml (auto = detect from the key).
MODEL_ARGS :=
ifneq ($(strip $(MODEL)),)
MODEL_ARGS += --model "$(MODEL)"
endif

# Optional inputs for non-interactive runs: make run REPO=<path|url> ISSUE_FILE=<file>
RUN_ARGS :=
ifneq ($(strip $(REPO)),)
RUN_ARGS += --repo "$(REPO)"
endif
ifneq ($(strip $(ISSUE_FILE)),)
RUN_ARGS += --issue-file "$(ISSUE_FILE)"
endif

.PHONY: setup run test clean ping demo eval check-env

setup:
	@test -n "$(PYTHON)" || { echo "ERROR: Python 3.10-3.14 is required (python3 on PATH is too old or missing)." >&2; exit 1; }
	@echo "Using $(PYTHON) ($$($(PYTHON) --version))"
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip -q
	$(PY) -m pip install -r requirements.txt -q
	@$(PY) -m harness --self-check
	@echo 'Setup complete. Export your key (export AI_API_KEY="<key>"), then run: make ping, make run'

# The key itself is checked by the harness, which knows each provider's variable and that local models need none.
check-env:
	@test -x $(PY) || { echo "ERROR: environment not installed. Run: make setup" >&2; exit 1; }

run: check-env
	@$(PY) -m harness $(MODEL_ARGS) $(RUN_ARGS)

test:
	@test -x $(PY) || { echo "ERROR: environment not installed. Run: make setup" >&2; exit 1; }
	$(PY) -m pytest

ping: check-env
	@$(PY) -m harness --ping $(MODEL_ARGS)

demo: check-env
	@$(PY) -m harness --demo $(MODEL_ARGS)

eval: check-env
	@$(PY) scripts/eval.py $(MODEL_ARGS) $(EVAL_ARGS)

clean:
	rm -rf $(VENV) runs .pytest_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete

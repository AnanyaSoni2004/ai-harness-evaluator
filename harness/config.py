"""Load config.yaml into Config dataclasses."""
from __future__ import annotations

import os
import re
import tempfile
import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from harness.types import HarnessError

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
TOOL_MODES = ("auto", "native", "text")


@dataclass
class ModelConfig:
    """Which model to call and how."""

    name: str = "openai/gpt-4o-mini"
    api_base: str | None = None
    temperature: float = 0.0
    seed: int | None = 42
    max_output_tokens: int = 4096
    context_window: int = 128000
    tool_mode: str = "auto"
    request_timeout_s: float = 120
    max_retries: int = 5


@dataclass
class BudgetsConfig:
    """Hard limits for a whole run."""

    max_total_tokens: int = 600000
    max_llm_calls: int = 120
    max_wall_clock_s: float = 1500


@dataclass
class PhasesConfig:
    """Per-phase step budgets and optional phases."""

    localize_max_steps: int = 15
    reproduce_max_steps: int = 10
    fix_max_steps: int = 25
    max_fix_attempts: int = 3
    enable_rescue: bool = True
    enable_review: bool = True


@dataclass
class ContextConfig:
    """Context-window management."""

    working_budget_tokens: int = 48000
    max_tool_output_chars: int = 8000
    keep_recent_messages: int = 8


@dataclass
class TestsConfig:
    """Test detection and timeouts."""

    __test__ = False  # not a pytest test class despite the name

    command: str | None = None
    baseline_timeout_s: float = 600
    targeted_timeout_s: float = 180
    run_full_suite_after_fix: bool = True


@dataclass
class SafetyConfig:
    """Guards on what the agent may do."""

    allow_edit_existing_tests: bool = False
    command_timeout_s: float = 120


@dataclass
class OutputConfig:
    """Where run artefacts and cloned repos go."""

    runs_dir: str = "runs"
    workspaces_dir: str | None = None


@dataclass
class Config:
    """Complete harness configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    budgets: BudgetsConfig = field(default_factory=BudgetsConfig)
    phases: PhasesConfig = field(default_factory=PhasesConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    tests: TestsConfig = field(default_factory=TestsConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def api_key(self) -> str:
        """Return AI_API_KEY from the environment, or raise HarnessError if it is unset."""
        key = os.environ.get("AI_API_KEY", "").strip()
        if not key:
            raise HarnessError('AI_API_KEY is not set. Run: export AI_API_KEY="<key>"')
        return key


def _is_secret_key(name: str) -> bool:
    """True for key names that look like credentials (plural '*_tokens' budget keys are allowed)."""
    lowered = name.lower()
    if any(s in lowered for s in ("api_key", "apikey", "api-key", "secret", "password")):
        return True
    return "token" in re.split(r"[_\-.]", lowered)


def _check_no_secrets(data: Any) -> None:
    """Raise HarnessError if any nested key name looks like a secret."""
    if isinstance(data, dict):
        for key, value in data.items():
            if _is_secret_key(str(key)):
                raise HarnessError(
                    "Secrets must not be stored in config.yaml; use the AI_API_KEY environment variable"
                )
            _check_no_secrets(value)
    elif isinstance(data, list):
        for item in data:
            _check_no_secrets(item)


def _fill(obj: Any, values: Any, section: str) -> None:
    """Copy known keys from a YAML mapping onto a dataclass instance; warn about unknown keys."""
    if values is None:
        return
    if not isinstance(values, dict):
        warnings.warn(f"config: section '{section}' should be a mapping; ignored")
        return
    known = {f.name for f in fields(obj)}
    for key, value in values.items():
        if key in known:
            setattr(obj, key, value)
        else:
            warnings.warn(f"config: unknown key '{section}.{key}' ignored")


def _apply_env_overrides(cfg: Config) -> None:
    """Development overrides: HARNESS_MODEL, HARNESS_API_BASE, HARNESS_TOOL_MODE."""
    if os.environ.get("HARNESS_MODEL"):
        cfg.model.name = os.environ["HARNESS_MODEL"]
    if os.environ.get("HARNESS_API_BASE"):
        cfg.model.api_base = os.environ["HARNESS_API_BASE"]
    if os.environ.get("HARNESS_TOOL_MODE"):
        cfg.model.tool_mode = os.environ["HARNESS_TOOL_MODE"]


def load_config(path: str | None = None) -> Config:
    """Load config.yaml (default: the project root's), apply env overrides, and return a Config."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    data: Any = {}
    if cfg_path.exists():
        try:
            data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            raise HarnessError(f"Could not parse {cfg_path}: {e}") from e
    elif path:
        raise HarnessError(f"Config file not found: {cfg_path}")
    if not isinstance(data, dict):
        raise HarnessError(f"{cfg_path} must contain a mapping at the top level")

    _check_no_secrets(data)

    cfg = Config()
    sections = {f.name for f in fields(cfg)}
    for section, values in data.items():
        if section in sections:
            _fill(getattr(cfg, section), values, section)
        else:
            warnings.warn(f"config: unknown section '{section}' ignored")

    _apply_env_overrides(cfg)

    if cfg.model.tool_mode not in TOOL_MODES:
        warnings.warn(f"config: model.tool_mode '{cfg.model.tool_mode}' is invalid; using 'auto'")
        cfg.model.tool_mode = "auto"
    if not cfg.output.workspaces_dir:
        cfg.output.workspaces_dir = str(Path(tempfile.gettempdir()) / "ai-harness-workspaces")
    return cfg

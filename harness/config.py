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

from harness import providers
from harness.types import HarnessError

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
TOOL_MODES = ("auto", "native", "text")


@dataclass
class ModelConfig:
    """Which model to call and how."""

    name: str = "auto"  # a preset name, a LiteLLM "<provider>/<model>" string, or "auto" (detect from the key)
    api_base: str | None = None
    temperature: float | None = 0.0  # None = provider default (Gemini 3 and OpenAI reasoning models want that)
    seed: int | None = 42
    max_output_tokens: int = 4096
    context_window: int | None = None  # None = from LiteLLM's model table (DEFAULT_CONTEXT_WINDOW if unknown)
    tokens_per_minute: int | None = None  # provider input-token rate limit; None = unknown (learned on error)
    reasoning_effort: str | None = None  # low | medium | high for reasoning models; None = provider default
    tool_mode: str = "auto"
    force_text_mode_for: list[str] = field(default_factory=list)
    max_consecutive_parse_failures: int = 3
    request_timeout_s: float = 120
    max_retries: int = 5
    # Set by load_config: the preset that was applied and what chose the model (shown by --ping).
    preset: str | None = None
    source: str = "config.yaml"


@dataclass
class BudgetsConfig:
    """Hard limits for a whole run."""

    max_total_tokens: int = 120000   # hard cap per issue
    soft_total_tokens: int = 45000   # beyond this, no new fix attempts
    max_llm_calls: int = 45
    max_wall_clock_s: float = 900


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
    baseline_timeout_s: float = 300
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


SPECTRUM_FORMULAS = ("ochiai", "tarantula")


@dataclass
class SpectrumConfig:
    """Execution-based fault localization (the Tracer)."""

    enabled: bool = True
    formula: str = "ochiai"
    timeout_s: float = 120
    repro_timeout_s: float = 60
    min_passing_runs: int = 3
    max_passing_tests: int = 60
    top_lines: int = 10
    top_functions: int = 5
    max_evidence_chars: int = 5000


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
    spectrum: SpectrumConfig = field(default_factory=SpectrumConfig)
    presets: dict[str, dict] = field(default_factory=dict)

    def key_and_source(self) -> tuple[str | None, str]:
        """(key, variable it came from): AI_API_KEY, else the provider's own variable (GEMINI_API_KEY, ...).
        A local model (Ollama, localhost endpoint) needs no key: (None, "none"). Raises HarnessError otherwise."""
        if self.model.name == "auto":
            key = providers.env("AI_API_KEY")
            if key:
                raise HarnessError("Could not tell the provider from AI_API_KEY's format. Choose the model: "
                                   f"make run MODEL=<{'|'.join(self.presets) or 'provider/model'}>")
            raise HarnessError('No API key found. Run: export AI_API_KEY="<key>" '
                               "(and optionally choose the model: make run MODEL=gemini)")
        provider = providers.provider_of(self.model.name)
        native = [(v, providers.env(v)) for v in providers.PROVIDER_KEY_ENV.get(provider, ()) if providers.env(v)]
        key = providers.env("AI_API_KEY")
        mismatch = providers.key_mismatch(key, provider) if key else None
        if key and not mismatch:
            return key, "AI_API_KEY"
        if native:
            return native[0][1], native[0][0]
        if key and self.model.api_base:  # a gateway or custom endpoint may accept any provider's key
            return key, "AI_API_KEY"
        if key:  # clearly another provider's key for this provider's own endpoint: never send it there
            raise HarnessError(f"{mismatch} (the key was not sent)")
        if providers.is_local(provider, self.model.api_base):
            return None, "none"
        names = " or ".join(("AI_API_KEY",) + providers.PROVIDER_KEY_ENV.get(provider, ()))
        raise HarnessError(f'No API key for {self.model.name}: set {names}. Run: export AI_API_KEY="<key>"')

    def api_key(self) -> str | None:
        """The key for the configured model (see key_and_source), or None for a local model."""
        return self.key_and_source()[0]


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


# Settings that describe one particular model; they do not carry over when another model is selected.
MODEL_SPECIFIC = ("api_base", "context_window", "tokens_per_minute", "reasoning_effort", "force_text_mode_for")
DEFAULT_CONTEXT_WINDOW = 32768


def _fill_presets(cfg: Config, values: Any) -> None:
    """Read the presets section: {name: {model settings}}."""
    if not isinstance(values, dict):
        warnings.warn("config: section 'presets' should be a mapping; ignored")
        return
    for name, preset in values.items():
        if isinstance(preset, dict) and preset.get("name"):
            cfg.presets[str(name)] = preset
        else:
            warnings.warn(f"config: preset '{name}' needs a mapping with a 'name'; ignored")


def _select_model(cfg: Config, override: str | None) -> None:
    """Resolve the model: --model, then HARNESS_MODEL, then model.name. A preset name expands to its settings;
    "auto" picks the preset matching AI_API_KEY's format or, without AI_API_KEY, the provider variable that is set.
    """
    configured = cfg.model.name
    name, source = (override, "--model") if override else (
        (providers.env("HARNESS_MODEL"), "HARNESS_MODEL") if providers.env("HARNESS_MODEL")
        else (configured, "config.yaml"))
    name = (name or "auto").strip()
    if name == "auto":
        key = providers.env("AI_API_KEY")
        detected = providers.preset_for_key(key) if key else None
        if detected in cfg.presets:
            name, source = detected, "auto: AI_API_KEY format"
        elif not key and (found := providers.preset_from_env()) and found[0] in cfg.presets:
            name, source = found[0], f"auto: {found[1]} is set"
        else:
            cfg.model.name, cfg.model.source = "auto", source  # key_and_source() explains what to do
            return
    if name != configured:
        defaults = ModelConfig()
        for key in MODEL_SPECIFIC:
            setattr(cfg.model, key, getattr(defaults, key))
    if name in cfg.presets:
        _fill(cfg.model, cfg.presets[name], f"presets.{name}")
        cfg.model.preset = name
    else:
        cfg.model.name = name
    cfg.model.source = source


def _apply_env_overrides(cfg: Config) -> None:
    """Development overrides: HARNESS_API_BASE, HARNESS_TOOL_MODE, HARNESS_SPECTRUM (HARNESS_MODEL: _select_model)."""
    if os.environ.get("HARNESS_API_BASE"):
        cfg.model.api_base = os.environ["HARNESS_API_BASE"]
    if os.environ.get("HARNESS_TOOL_MODE"):
        cfg.model.tool_mode = os.environ["HARNESS_TOOL_MODE"]
    if os.environ.get("HARNESS_SPECTRUM", "").strip().lower() in ("0", "false", "no", "off"):
        cfg.spectrum.enabled = False


CHARS_PER_TOKEN = 3.5


def derive_context_budgets(cfg: Config) -> None:
    """Cap context budgets by the model's window (55% for history, ~6% per tool output) and, when set, by
    model.tokens_per_minute: one prompt above that limit is always rejected, so history stays under 70% of
    it, one tool output under 20% and the spectrum evidence under 10%.
    """
    if not cfg.model.context_window:
        known = providers.context_window_for(cfg.model.name) if cfg.model.name != "auto" else None
        cfg.model.context_window = known or DEFAULT_CONTEXT_WINDOW
    window = int(cfg.model.context_window)
    cfg.context.working_budget_tokens = min(int(cfg.context.working_budget_tokens), int(window * 0.55))
    cfg.context.max_tool_output_chars = min(int(cfg.context.max_tool_output_chars),
                                            int(window * 0.06 * CHARS_PER_TOKEN))
    cfg.spectrum.max_evidence_chars = min(int(cfg.spectrum.max_evidence_chars),
                                          int(window * 0.04 * CHARS_PER_TOKEN))
    tpm = int(cfg.model.tokens_per_minute or 0)
    if tpm > 0:
        cfg.context.working_budget_tokens = min(cfg.context.working_budget_tokens, int(tpm * 0.7))
        cfg.context.max_tool_output_chars = min(cfg.context.max_tool_output_chars,
                                                int(tpm * 0.2 * CHARS_PER_TOKEN))
        cfg.spectrum.max_evidence_chars = min(cfg.spectrum.max_evidence_chars, int(tpm * 0.1 * CHARS_PER_TOKEN))


def load_config(path: str | None = None, model: str | None = None) -> Config:
    """Load config.yaml (default: the project root's), select the model (`model` overrides model.name and
    HARNESS_MODEL), apply env overrides, and return a Config."""
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
        if section == "presets":
            _fill_presets(cfg, values)
        elif section in sections:
            _fill(getattr(cfg, section), values, section)
        else:
            warnings.warn(f"config: unknown section '{section}' ignored")

    _select_model(cfg, model)
    _apply_env_overrides(cfg)

    if cfg.model.tool_mode not in TOOL_MODES:
        warnings.warn(f"config: model.tool_mode '{cfg.model.tool_mode}' is invalid; using 'auto'")
        cfg.model.tool_mode = "auto"
    if str(cfg.spectrum.formula).lower() not in SPECTRUM_FORMULAS:
        warnings.warn(f"config: spectrum.formula '{cfg.spectrum.formula}' is unknown; using 'ochiai'")
        cfg.spectrum.formula = "ochiai"
    cfg.spectrum.formula = str(cfg.spectrum.formula).lower()
    forced = cfg.model.force_text_mode_for
    if forced is None:
        cfg.model.force_text_mode_for = []
    elif isinstance(forced, str):
        cfg.model.force_text_mode_for = [forced]
    elif not isinstance(forced, list):
        warnings.warn("config: model.force_text_mode_for must be a list of strings; ignored")
        cfg.model.force_text_mode_for = []
    if not cfg.output.workspaces_dir:
        cfg.output.workspaces_dir = str(Path(tempfile.gettempdir()) / "ai-harness-workspaces")
    derive_context_budgets(cfg)
    return cfg

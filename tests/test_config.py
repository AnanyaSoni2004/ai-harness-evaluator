"""Tests for config.yaml loading."""
import tempfile
from pathlib import Path

import pytest

from harness.config import Config, load_config
from harness.types import HarnessError

ENV_VARS = ("HARNESS_MODEL", "HARNESS_API_BASE", "HARNESS_TOOL_MODE")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _write(tmp_path: Path, text: str) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_project_config_loads_with_defaults() -> None:
    cfg = load_config()
    assert cfg.model.temperature == 0.0
    assert cfg.model.tool_mode == "auto"
    assert cfg.budgets.max_total_tokens == 600000
    assert cfg.phases.max_fix_attempts == 3
    assert cfg.tests.command is None
    expected = Path(tempfile.gettempdir()) / "ai-harness-workspaces"
    assert cfg.output.workspaces_dir == str(expected)


def test_partial_yaml_keeps_defaults(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, "model:\n  name: deepseek/deepseek-chat\nphases:\n  fix_max_steps: 5\n"))
    assert cfg.model.name == "deepseek/deepseek-chat"
    assert cfg.model.seed == 42
    assert cfg.phases.fix_max_steps == 5
    assert cfg.context.working_budget_tokens == 48000


def test_empty_yaml_gives_defaults(tmp_path: Path) -> None:
    assert load_config(_write(tmp_path, "")) == load_config(_write(tmp_path, "{}"))


def test_unknown_keys_warn_not_fail(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="unknown"):
        cfg = load_config(_write(tmp_path, "model:\n  flavour: spicy\nextras:\n  a: 1\n"))
    assert isinstance(cfg, Config)


def test_env_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_MODEL", "dashscope/qwen-max")
    monkeypatch.setenv("HARNESS_API_BASE", "http://localhost:8000/v1")
    monkeypatch.setenv("HARNESS_TOOL_MODE", "text")
    cfg = load_config(_write(tmp_path, "model:\n  name: openai/x\n"))
    assert cfg.model.name == "dashscope/qwen-max"
    assert cfg.model.api_base == "http://localhost:8000/v1"
    assert cfg.model.tool_mode == "text"


def test_invalid_tool_mode_falls_back(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="tool_mode"):
        cfg = load_config(_write(tmp_path, "model:\n  tool_mode: telepathy\n"))
    assert cfg.model.tool_mode == "auto"


@pytest.mark.parametrize("key", ["api_key", "OPENAI_API_KEY", "token", "auth_token", "client_secret", "password"])
def test_secret_keys_rejected(tmp_path: Path, key: str) -> None:
    with pytest.raises(HarnessError, match="Secrets must not be stored"):
        load_config(_write(tmp_path, f"model:\n  {key}: abc\n"))


def test_token_budget_keys_are_not_secrets(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, "budgets:\n  max_total_tokens: 10\n"))
    assert cfg.budgets.max_total_tokens == 10


def test_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_API_KEY", "   ")
    with pytest.raises(HarnessError, match="AI_API_KEY is not set"):
        load_config().api_key()
    monkeypatch.delenv("AI_API_KEY")
    with pytest.raises(HarnessError):
        load_config().api_key()


def test_api_key_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_API_KEY", " fake-key-for-test ")
    assert load_config().api_key() == "fake-key-for-test"


def test_missing_explicit_file(tmp_path: Path) -> None:
    with pytest.raises(HarnessError, match="not found"):
        load_config(str(tmp_path / "nope.yaml"))


def test_bad_yaml(tmp_path: Path) -> None:
    with pytest.raises(HarnessError, match="Could not parse"):
        load_config(_write(tmp_path, "model: [unclosed\n"))

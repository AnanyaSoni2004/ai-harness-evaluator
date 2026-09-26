"""Tests for run_process, sanitised_env, and truncate."""
import os
import sys
import time
from pathlib import Path

import pytest

from harness import shell
from harness.shell import run_process, sanitised_env, truncate


def test_echo(tmp_path: Path) -> None:
    res = run_process("echo hello; echo oops >&2", tmp_path, 10)
    assert res.exit_code == 0 and not res.timed_out
    assert "hello" in res.output and "oops" in res.output  # stderr merged


def test_exit_code_and_cwd(tmp_path: Path) -> None:
    (tmp_path / "marker.txt").write_text("x")
    res = run_process("ls; exit 3", tmp_path, 10)
    assert res.exit_code == 3 and "marker.txt" in res.output


def test_timeout_kills_quickly(tmp_path: Path) -> None:
    start = time.monotonic()
    res = run_process("echo started; sleep 5", tmp_path, 1)
    assert res.timed_out and res.exit_code is None
    assert time.monotonic() - start < 3
    assert "started" in res.output  # partial output kept


def test_timeout_kills_grandchildren(tmp_path: Path) -> None:
    start = time.monotonic()
    res = run_process("sleep 30 & sleep 30 & wait", tmp_path, 1)
    assert res.timed_out and time.monotonic() - start < 3


def test_stdin_is_closed(tmp_path: Path) -> None:
    res = run_process("cat; echo done", tmp_path, 5)
    assert not res.timed_out and "done" in res.output


def test_api_key_absent_in_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_API_KEY", "fake-key-should-not-leak")
    monkeypatch.setenv("VIRTUAL_ENV", "/some/venv")
    res = run_process("env", tmp_path, 10)
    assert "fake-key-should-not-leak" not in res.output
    assert "AI_API_KEY" not in res.output and "VIRTUAL_ENV=" not in res.output
    assert "CI=1" in res.output and "GIT_TERMINAL_PROMPT=0" in res.output


def test_env_extra_applied(tmp_path: Path) -> None:
    res = run_process('echo "$FOO"', tmp_path, 10, env_extra={"FOO": "bar"})
    assert res.output.strip() == "bar"


def test_harness_venv_removed_from_path(monkeypatch: pytest.MonkeyPatch) -> None:
    venv_bin = str(shell._PROJECT_VENV_BIN)
    monkeypatch.setenv("PATH", os.pathsep.join([venv_bin, "/usr/bin", "/bin"]))
    path = sanitised_env()["PATH"].split(os.pathsep)
    assert venv_bin not in path and "/usr/bin" in path
    if sys.prefix != sys.base_prefix:
        assert str(Path(sys.prefix) / "bin") not in path


def test_missing_cwd_does_not_raise(tmp_path: Path) -> None:
    res = run_process("echo hi", tmp_path / "missing", 5)
    assert res.exit_code is None and "Failed to start" in res.output


def test_non_utf8_output(tmp_path: Path) -> None:
    res = run_process("printf '\\377\\376ok'", tmp_path, 5)
    assert res.output.endswith("ok")


def test_truncate_keeps_head_and_tail() -> None:
    text = "HEAD" + "x" * 10000 + "TAIL"
    out = truncate(text, 1000, head_chars=100)
    assert out.startswith("HEAD") and out.endswith("TAIL")
    assert "chars truncated" in out
    assert len(out) < 1100
    assert truncate("short", 100) == "short"


def test_truncate_head_larger_than_budget() -> None:
    out = truncate("a" * 50 + "b" * 50, 20, head_chars=1500)
    assert out.startswith("a" * 10) and out.endswith("b" * 10)

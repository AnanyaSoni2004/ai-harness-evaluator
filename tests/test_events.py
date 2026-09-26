"""Tests for redaction, the trajectory logger, and NullUI."""
import json
import threading
from pathlib import Path

import pytest

from harness.events import NullUI, Trajectory, redact
from harness.types import Usage

FAKE_KEY = "sk-test-FAKEKEY-0123456789"


def test_redact_replaces_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_API_KEY", FAKE_KEY)
    assert redact(f"auth={FAKE_KEY}; again {FAKE_KEY}") == "auth=***; again ***"
    assert redact("nothing secret") == "nothing secret"


def test_redact_ignores_short_or_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_API_KEY", "abc")
    assert redact("abc def") == "abc def"
    monkeypatch.delenv("AI_API_KEY")
    assert redact("abc def") == "abc def"


def test_trajectory_lines_parse_and_are_redacted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_API_KEY", FAKE_KEY)
    path = tmp_path / "run" / "trajectory.jsonl"
    traj = Trajectory(path)
    traj.log("llm_call", phase="fix", tokens=12, echo=f"key was {FAKE_KEY}")
    traj.log("tool", path=Path("a/b.py"), usage=Usage(1, 2))
    text = path.read_text(encoding="utf-8")
    assert FAKE_KEY not in text
    records = [json.loads(line) for line in text.splitlines()]
    assert [r["event"] for r in records] == ["llm_call", "tool"]
    assert records[0]["echo"] == "key was ***"
    assert records[0]["tokens"] == 12 and isinstance(records[0]["ts"], float)
    assert records[1]["path"] == "a/b.py"


def test_trajectory_none_is_noop(tmp_path: Path) -> None:
    Trajectory(None).log("anything", x=1)
    assert list(tmp_path.iterdir()) == []


def test_trajectory_thread_safe(tmp_path: Path) -> None:
    traj = Trajectory(tmp_path / "t.jsonl")
    threads = [threading.Thread(target=lambda i=i: [traj.log("e", i=i, j=j) for j in range(50)]) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lines = (tmp_path / "t.jsonl").read_text().splitlines()
    assert len(lines) == 200 and all(json.loads(line)["event"] == "e" for line in lines)


def test_null_ui_is_silent(capsys: pytest.CaptureFixture) -> None:
    ui = NullUI()
    ui.phase("LOCALIZE", "finding")
    ui.info("i")
    ui.warn("w")
    ui.tool_call("view_file", "a.py:1-10")
    ui.tool_result(True, "10 lines")
    ui.llm_call("fix", Usage(1, 1))
    with ui.thinking("fix"):
        pass
    assert capsys.readouterr().out == ""

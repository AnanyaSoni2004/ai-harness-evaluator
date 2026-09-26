"""Tests for the Rich terminal UI."""
import contextlib
import io

from rich.console import Console

from harness.types import IssueSpec, Metrics, RunState, Usage
from harness.ui import RichUI


def make_ui(quiet: bool = False, terminal: bool = False) -> tuple[RichUI, io.StringIO]:
    buf = io.StringIO()
    console = Console(file=buf, width=120, force_terminal=terminal, color_system=None, highlight=False)
    return RichUI(quiet=quiet, console=console), buf


def sample_state(status: str = "verified") -> RunState:
    state = RunState(run_id="r", repo="/tmp/repo", issue=IssueSpec(raw_text="x"))
    state.status = status
    state.repro = {"reproduced": True, "command": "python3 @scratch/repro.py"}
    return state


def test_progress_lines() -> None:
    ui, buf = make_ui()
    ui.phase("LOCALIZE", "finding the root cause")
    ui.tool_call("view_file", "view_file src/x.py:1-120")
    ui.tool_result(True, "120 lines")
    ui.tool_result(False, "old_str not found")
    ui.warn("careful")
    ui.info("note")
    ui.llm_call("fix", Usage(100, 20))
    out = buf.getvalue()
    assert "LOCALIZE" in out and "finding the root cause" in out
    assert "  → view_file src/x.py:1-120" in out
    assert "    ✓ 120 lines" in out and "    ✗ old_str not found" in out
    assert "⚠ careful" in out and "fix: 120 tokens" in out


def test_markup_in_dynamic_text_is_escaped() -> None:
    ui, buf = make_ui()
    ui.tool_call("search_code", "search_code '[bold]x[/bold]' in [red]")
    assert "'[bold]x[/bold]' in [red]" in buf.getvalue()


def test_quiet_mode_prints_only_summary_and_errors() -> None:
    ui, buf = make_ui(quiet=True)
    ui.banner("m", "native", "0.1.0")
    ui.phase("FIX")
    ui.tool_call("x", "x")
    ui.show_diff("--- a/x\n+++ b/x\n")
    assert buf.getvalue() == ""
    ui.error("boom")
    ui.show_summary(sample_state(), Metrics(), {"report": "/runs/r/report.md", "has_diff": True})
    out = buf.getvalue()
    assert "Error: boom" in out and "VERIFIED FIX" in out and "/runs/r/report.md" in out


def test_banner_diff_verification_summary() -> None:
    ui, buf = make_ui()
    ui.banner("deepseek/deepseek-chat", "text", "0.1.0")
    ui.show_diff("--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n")
    ui.show_diff("")
    attempt = {"attempt": 2, "verification": {"repro_before_exit": 1, "repro_after_exit": 0, "new_failures": []},
               "review": {"verdict": "approve"}}
    ui.show_verification(attempt, sample_state())
    metrics = Metrics(started_at=0.0, ended_at=12.0)
    metrics.add_llm("fix", Usage(1000, 50))
    ui.show_summary(sample_state("no_fix"), metrics, {"report": "r.md", "has_diff": False})
    out = buf.getvalue()
    assert "AI Coding Harness" in out and "deepseek/deepseek-chat" in out and "text" in out
    assert "+b" in out and "(no changes)" in out
    assert "Verification (attempt 2)" in out and "exit 1" in out and "exit 0" in out and "none ✅" in out
    assert "NO FIX" in out and "Tokens: 1050 (1000 prompt + 50 completion)" in out and "Time: 12.0s" in out


def test_thinking_is_silent_outside_a_terminal() -> None:
    ui, _ = make_ui(terminal=False)
    assert isinstance(ui.thinking("fix"), contextlib.nullcontext)
    ui_tty, _ = make_ui(terminal=True)
    with ui_tty.thinking("fix"):
        pass


def test_show_spectrum() -> None:
    ui, buf = make_ui()
    ui.show_spectrum({"ok": True, "functions": [
        {"name": "Inventory.remove", "path": "toolkit/inventory.py", "start": 16, "end": 21, "score": 0.7071}]})
    ui.show_spectrum({"ok": False, "reason": "no reproduction"})
    out = buf.getvalue()
    assert "Suspicious code (Tracer)" in out and "Inventory.remove" in out and "toolkit/inventory.py:16-21" in out
    assert "0.71" in out and "Tracer skipped: no reproduction" in out
    quiet, qbuf = make_ui(quiet=True)
    quiet.show_spectrum({"ok": False, "reason": "x"})
    assert qbuf.getvalue() == ""

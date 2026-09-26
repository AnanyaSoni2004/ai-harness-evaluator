"""Rich terminal UI (the only module importing Rich)."""
from __future__ import annotations

import contextlib
from typing import Any, ContextManager

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table

from harness import report
from harness.events import redact


class RichUI:
    """Terminal UI with the NullUI interface plus banner / verification / diff / summary views."""

    def __init__(self, quiet: bool = False, console: Console | None = None) -> None:
        """quiet=True prints only the final summary (used by scripts/eval.py)."""
        self.quiet = quiet
        self.console = console or Console(highlight=False)

    def _print(self, text: str, style: str = "") -> None:
        if not self.quiet:
            self.console.print(escape(redact(text)), style=style or None)

    # ------------------------------------------------------------------ NullUI interface
    def phase(self, name: str, detail: str = "") -> None:
        if not self.quiet:
            title = f"{escape(name)}" + (f" [dim]── {escape(detail)}[/dim]" if detail else "")
            self.console.print(Rule(title, align="left", style="cyan"))

    def info(self, msg: str) -> None:
        self._print(f"  {msg}", "dim")

    def warn(self, msg: str) -> None:
        self._print(f"  ⚠ {msg}", "yellow")

    def tool_call(self, name: str, preview: str) -> None:
        self._print(f"  → {preview or name}")

    def tool_result(self, ok: bool, summary: str) -> None:
        self._print(f"    {'✓' if ok else '✗'} {summary}", "green" if ok else "red")

    def llm_call(self, phase: str, usage: Any) -> None:
        tokens = getattr(usage, "prompt_tokens", 0) + getattr(usage, "completion_tokens", 0)
        self._print(f"    · {phase}: {tokens} tokens", "dim")

    def thinking(self, label: str) -> ContextManager[Any]:
        if self.quiet or not self.console.is_terminal:
            return contextlib.nullcontext()
        return self.console.status(f"{escape(label)}: waiting for model…", spinner="dots")

    # ------------------------------------------------------------------ extra views
    def ask(self, question: str) -> str:
        """Read one line from the user."""
        return self.console.input(f"[bold]{escape(question)}[/bold] ")

    def error(self, msg: str) -> None:
        """Errors are always shown, even in quiet mode."""
        self.console.print(f"[bold red]Error:[/bold red] {escape(redact(msg))}")

    def banner(self, model: str, tool_mode: str, version: str) -> None:
        if not self.quiet:
            body = f"Model: [bold]{escape(model)}[/bold]   Tool mode: {escape(tool_mode)}   Version: {escape(version)}"
            self.console.print(Panel(body, title="AI Coding Harness", border_style="cyan"))

    def show_verification(self, attempt: dict | None, state: Any = None) -> None:
        if self.quiet or attempt is None or state is None:
            return
        table = Table(title=f"Verification (attempt {attempt.get('attempt')})", show_lines=False)
        for column in ("Check", "Before", "After"):
            table.add_column(column)
        for check, before, after in report.evidence_rows(state, attempt):
            table.add_row(escape(check), escape(before), escape(after))
        self.console.print(table)

    def show_spectrum(self, spectrum: dict) -> None:
        """Top 5 suspicious functions after TRACE, or why the Tracer was skipped."""
        if self.quiet:
            return
        if not spectrum.get("ok"):
            self._print(f"  Tracer skipped: {spectrum.get('reason', 'not run')}", "dim")
            return
        table = Table(title="Suspicious code (Tracer)")
        for column in ("#", "Function", "Location", "Score"):
            table.add_column(column)
        for i, f in enumerate((spectrum.get("functions") or [])[:5], 1):
            table.add_row(str(i), escape(str(f["name"])), escape(f"{f['path']}:{f['start']}-{f['end']}"),
                          f"{f['score']:.2f}")
        self.console.print(table)

    def show_diff(self, diff: str) -> None:
        if not self.quiet:
            if diff.strip():
                self.console.print(Syntax(redact(diff), "diff", theme="ansi_dark", word_wrap=True))
            else:
                self.console.print("[dim](no changes)[/dim]")

    def show_summary(self, state: Any, metrics: Any, paths: dict) -> None:
        """Always printed, also in quiet mode."""
        total = metrics.to_dict()["total"]
        status = report.status_line(state, bool(paths.get("has_diff")))
        lines = [f"[bold]{escape(status)}[/bold]",
                 f"Tokens: {total['total_tokens']} ({total['prompt_tokens']} prompt + "
                 f"{total['completion_tokens']} completion)   LLM calls: {total['llm_calls']}   "
                 f"Tool calls: {total['tool_calls']}   Time: {total['seconds']:.1f}s"]
        if getattr(state, "stop_reason", ""):
            lines.append(f"Stopped early: {escape(state.stop_reason)}")
        if paths.get("report"):
            lines.append(f"Report: {escape(str(paths['report']))}")
        style = "green" if state.status == "verified" else "yellow" if state.status != "error" else "red"
        self.console.print(Panel("\n".join(lines), title="Result", border_style=style))

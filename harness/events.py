"""Trajectory JSONL logger, redact(), and the NullUI interface."""
from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, ContextManager

MIN_SECRET_LEN = 8


def redact(text: str) -> str:
    """Replace the current AI_API_KEY value (if at least 8 chars) with '***'."""
    key = os.environ.get("AI_API_KEY", "").strip()
    if len(key) >= MIN_SECRET_LEN and key in text:
        return text.replace(key, "***")
    return text


class Trajectory:
    """Append-only JSONL event log for one run. path=None makes every call a no-op."""

    def __init__(self, path: Path | None) -> None:
        """Create the logger; parent directories are created on first write."""
        self.path = Path(path) if path is not None else None
        self._lock = threading.Lock()

    def log(self, event: str, **data: Any) -> None:
        """Append one redacted JSON line {"ts", "event", **data}."""
        if self.path is None:
            return
        record = {"ts": round(time.time(), 3), "event": event, **data}
        line = redact(json.dumps(record, default=str, ensure_ascii=False))
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


class NullUI:
    """UI interface with no output. The Rich UI in harness/ui.py implements the same methods."""

    def phase(self, name: str, detail: str = "") -> None:
        """A new phase started."""

    def info(self, msg: str) -> None:
        """An informational message."""

    def warn(self, msg: str) -> None:
        """A warning message."""

    def tool_call(self, name: str, preview: str) -> None:
        """The agent is calling a tool."""

    def tool_result(self, ok: bool, summary: str) -> None:
        """A tool returned."""

    def llm_call(self, phase: str, usage: Any) -> None:
        """An LLM call completed with the given usage."""

    def thinking(self, label: str) -> ContextManager[Any]:
        """Context manager shown while waiting for the model."""
        return contextlib.nullcontext()

    def error(self, msg: str) -> None:
        """An error message."""

    def banner(self, model: str, tool_mode: str, version: str) -> None:
        """Start-up banner."""

    def show_verification(self, attempt: Any, state: Any = None) -> None:
        """Verification table for the final attempt."""

    def show_diff(self, diff: str) -> None:
        """The final patch."""

    def show_summary(self, state: Any, metrics: Any, paths: dict) -> None:
        """Final result summary."""

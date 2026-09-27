"""Exec tool: run_command."""
from __future__ import annotations

import os
import re
import shlex
import shutil
from typing import Any

from harness.shell import run_process, sanitised_env, truncate
from harness.tools.read_tools import as_int, never_raises
from harness.types import ToolResult
from harness.workspace import Workspace

MAX_COMMAND_TIMEOUT_S = 600
DEFAULT_COMMAND_TIMEOUT_S = 120
DEFAULT_MAX_OUTPUT_CHARS = 8000
EDIT_HINT = "Use str_replace to edit files so changes are tracked."

# (pattern, message) — checked in order; the first match blocks the command.
BLOCKED: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bsudo\b"), "sudo is not allowed."),
    (re.compile(r"rm\s+-[a-z]*r[a-z]*f?\s+(/|~|\*)"), "Recursive deletion of /, ~ or * is not allowed."),
    (re.compile(r"\bmkfs\b"), "Formatting file systems is not allowed."),
    (re.compile(r"\bdd\s+if="), "dd is not allowed."),
    (re.compile(r"\b(shutdown|reboot)\b"), "Shutting down or rebooting is not allowed."),
    (re.compile(r":\(\)\s*\{"), "Fork bombs are not allowed."),
    (re.compile(r"git\s+(push|reset\s+--hard|clean|commit|checkout\s+--)"),
     "This git command is not allowed: the harness manages repository state."),
    (re.compile(r"(curl|wget)[^|]*\|\s*(ba)?sh"), "Piping downloads into a shell is not allowed."),
    (re.compile(r"\bsed\s+-i\b"), EDIT_HINT),
    (re.compile(r"\bperl\s+-pi\b"), EDIT_HINT),
]


def blocked_reason(command: str) -> str | None:
    """The reason a command is blocked, or None if it may run."""
    for pattern, message in BLOCKED:
        if pattern.search(command):
            return message
    return None


def python_shims(ws: Workspace) -> str | None:
    """Directory with `python` and `python3` shims for the target repo's interpreter (None if none found).

    Models often call `python`, which does not exist on many systems (macOS has only python3); each miss
    costs a wasted model call. The shims also make both names use the interpreter the tests use.
    """
    base_path = sanitised_env()["PATH"]
    target = getattr(ws, "python_exe", None) or "python3"
    resolved = target if os.path.isabs(target) else shutil.which(target, path=base_path)
    if not resolved:
        return None
    shim_dir = ws.scratch_dir / ".bin"
    shim_dir.mkdir(parents=True, exist_ok=True)
    script = f'#!/bin/sh\nexec {shlex.quote(resolved)} "$@"\n'
    for name in ("python", "python3"):
        shim = shim_dir / name
        if not shim.exists() or shim.read_text() != script:
            shim.write_text(script)
            shim.chmod(0o755)
    return str(shim_dir)


def _cfg_value(cfg: Any, section: str, name: str, default: Any) -> Any:
    """cfg.<section>.<name>, or default when cfg (or the key) is missing."""
    return getattr(getattr(cfg, section, None), name, default) if cfg is not None else default


@never_raises
def run_command(ws: Workspace, cfg: Any, command: str = "", timeout_s: Any = None) -> ToolResult:
    """Run a shell command in the repository root (non-interactive, time-limited, output truncated)."""
    command = str(command or "").strip()
    if not command:
        return ToolResult(False, "command is required.")
    reason = blocked_reason(command)
    if reason:
        return ToolResult(False, f"Blocked: {reason}", {"blocked": True})

    scratch = str(ws.scratch_dir) + "/"
    expanded = command.replace("@scratch/", scratch)

    default_timeout = _cfg_value(cfg, "safety", "command_timeout_s", DEFAULT_COMMAND_TIMEOUT_S)
    requested = as_int(timeout_s, "timeout_s", None)
    timeout = min(requested if requested and requested > 0 else default_timeout, MAX_COMMAND_TIMEOUT_S)

    existing = os.environ.get("PYTHONPATH", "")
    pythonpath = ws.pythonpath() + (os.pathsep + existing if existing else "")
    env = {"PYTHONPATH": pythonpath}
    shims = python_shims(ws)
    if shims:
        env["PATH"] = shims + os.pathsep + sanitised_env()["PATH"]
    res = run_process(expanded, ws.repo_root, timeout, env_extra=env)

    output = res.output.replace(scratch, "@scratch/")
    max_chars = _cfg_value(cfg, "context", "max_tool_output_chars", DEFAULT_MAX_OUTPUT_CHARS)
    status = f"[TIMEOUT after {timeout:g}s]" if res.timed_out else f"[exit {res.exit_code}]"
    text = f"$ {command}\n{status} ({res.duration_s:.1f}s)\n{truncate(output, int(max_chars))}".rstrip()
    ok = res.exit_code == 0 and not res.timed_out
    return ToolResult(ok, text, {"exit_code": res.exit_code, "timed_out": res.timed_out,
                                 "duration_s": round(res.duration_s, 3)})

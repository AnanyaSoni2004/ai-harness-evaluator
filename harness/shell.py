"""run_process(), sanitised environment, and output truncation."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from harness.types import ProcessResult

_PROJECT_VENV_BIN = Path(__file__).resolve().parents[1] / ".venv" / "bin"
_KILL_GRACE_S = 5.0


def _harness_bin_dirs() -> set[str]:
    """Resolved bin directories of this harness's own virtualenv (removed from the child PATH)."""
    dirs = {str(_PROJECT_VENV_BIN.resolve())}
    if sys.prefix != sys.base_prefix:
        dirs.add(str((Path(sys.prefix) / "bin").resolve()))
    return dirs


def sanitised_env(extra: dict | None = None) -> dict[str, str]:
    """Environment for target-repo processes: no API key, no harness venv, non-interactive."""
    env = dict(os.environ)
    for name in ("AI_API_KEY", "VIRTUAL_ENV", "PYTHONHOME"):
        env.pop(name, None)
    blocked = _harness_bin_dirs()
    parts = [p for p in env.get("PATH", "").split(os.pathsep) if p]
    env["PATH"] = os.pathsep.join(p for p in parts if str(Path(p).resolve()) not in blocked)
    env.update({
        "CI": "1",
        "PAGER": "cat",
        "GIT_PAGER": "cat",
        "TERM": "dumb",
        "GIT_TERMINAL_PROMPT": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    })
    if extra:
        env.update({str(k): str(v) for k, v in extra.items()})
    return env


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGKILL the whole process group started for proc (falls back to killing proc alone)."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, AttributeError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


def _decode(data: bytes | str | None) -> str:
    """Decode subprocess output leniently."""
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    return data.decode("utf-8", errors="replace")


def run_process(cmd: str, cwd: Path, timeout_s: float, env_extra: dict | None = None) -> ProcessResult:
    """Run a shell command with a timeout, no stdin, merged output, and process-group kill on timeout."""
    start = time.monotonic()
    try:
        proc = subprocess.Popen(
            cmd, shell=True, cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, start_new_session=True, env=sanitised_env(env_extra),
        )
    except OSError as e:
        return ProcessResult(exit_code=None, output=f"Failed to start command: {type(e).__name__}: {e}",
                             timed_out=False, duration_s=time.monotonic() - start)
    try:
        out, _ = proc.communicate(timeout=timeout_s)
        return ProcessResult(exit_code=proc.returncode, output=_decode(out), timed_out=False,
                             duration_s=time.monotonic() - start)
    except subprocess.TimeoutExpired as first:
        _kill_group(proc)
        try:
            out, _ = proc.communicate(timeout=_KILL_GRACE_S)
        except subprocess.TimeoutExpired:
            # A detached grandchild still holds the pipe: give up on the rest of the output.
            out = first.output
            if proc.stdout:
                proc.stdout.close()
            proc.wait(timeout=_KILL_GRACE_S)
        return ProcessResult(exit_code=None, output=_decode(out), timed_out=True,
                             duration_s=time.monotonic() - start)


def truncate(text: str, max_chars: int, head_chars: int = 1500) -> str:
    """Keep the head and the (more important) tail of long text, marking what was cut."""
    if len(text) <= max_chars:
        return text
    head = min(head_chars, max_chars // 2)
    tail = max_chars - head
    cut = len(text) - head - tail
    return f"{text[:head]}\n...[{cut} chars truncated]...\n{text[len(text) - tail:]}"

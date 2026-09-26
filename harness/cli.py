"""Command-line interface: argument parsing, input collection, and self-check."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import pkgutil
import re
import shlex
import shutil
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import harness

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_REPO = PROJECT_ROOT / "fixtures" / "sample_repo"
DEMO_ISSUE = PROJECT_ROOT / "fixtures" / "issues" / "01_slugify.md"
GITHUB_ISSUE_URL = re.compile(r"^https?://github\.com/([^/\s]+)/([^/\s]+)/issues/(\d+)/?$")
PASTE_PROMPT = "Paste the issue (title + description). Finish with a line containing only END, or press Ctrl-D:"


class InputError(Exception):
    """Bad or missing user input (clear message, exit 1)."""


def _all_module_names() -> list[str]:
    """Return every importable harness.* module name, excluding the __main__ entry point."""
    names = ["harness"]
    for info in pkgutil.walk_packages(harness.__path__, prefix="harness."):
        if info.name.split(".")[-1] != "__main__":
            names.append(info.name)
    return sorted(names)


def self_check(config_path: str | None = None) -> int:
    """Import every module and load the config (no key, no network); report optional tools."""
    from harness.config import load_config
    from harness.types import HarnessError

    for name in _all_module_names():
        importlib.import_module(name)
    try:
        cfg = load_config(config_path)
    except HarnessError as e:
        print(f"Self-check FAILED: {e}")
        return 1
    print(f"Model (config.yaml): {cfg.model.name}")
    for tool in ("git", "rg", "node"):
        print(f"  {tool:5} {'found' if shutil.which(tool) else 'not found (optional)'}")
    print("Self-check OK")
    return 0


def make_llm(cfg: Any, ui: Any) -> Any:
    """The real model client (tests replace this with FakeLLM)."""
    from harness.events import Trajectory
    from harness.llm import LLMClient
    from harness.types import Metrics

    return LLMClient(cfg, Metrics(), Trajectory(None), ui)


def ping(cfg: Any, verbose: bool = False) -> int:
    """Probe tool mode and send one tiny request to the configured model."""
    from harness.events import Trajectory
    from harness.llm import LLMClient
    from harness.types import HarnessError, Metrics

    try:
        cfg.api_key()
        metrics = Metrics()
        client = LLMClient(cfg, metrics, Trajectory(None))
        mode = client.tool_mode
        resp = client.complete([{"role": "user", "content": "Reply with exactly: PONG"}], None, "ping")
    except HarnessError as e:
        print(f"Ping failed: {e}")
        return 1
    print(f"Model:     {cfg.model.name}")
    if verbose:
        info = client.probe_info
        print(f"Endpoint:  {cfg.model.api_base or '(provider default)'}")
        print(f"Probe:     {info.get('mode', mode)} ({info.get('reason', 'configured tool_mode, no probe')})")
        print(f"Probe raw: {info.get('raw', '')[:200]!r}")
    print(f"Tool mode: {mode}")
    print(f"Reply:     {resp.text.strip()!r}")
    print(f"Tokens:    {metrics.prompt_tokens} prompt + {metrics.completion_tokens} completion "
          f"({metrics.llm_calls} calls incl. probe)")
    return 0


# ---------------------------------------------------------------------- inputs
def is_git_url(text: str) -> bool:
    """http(s)://, git@ or *.git inputs are cloned; anything else is a local path."""
    return text.startswith(("http://", "https://", "git@")) or text.endswith(".git")


def clone_repo(url: str, cfg: Any) -> Path:
    """git clone --depth 50 <url> into <workspaces_dir>/<name>-<timestamp>."""
    from harness.shell import run_process

    base = Path(cfg.output.workspaces_dir)
    base.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"\.git$", "", url.rstrip("/").split("/")[-1].split(":")[-1]) or "repo"
    dest = base / f"{name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    res = run_process(f"git clone --depth 50 {shlex.quote(url)} {shlex.quote(str(dest))}", base, 600)
    if res.exit_code != 0 or not dest.is_dir():
        raise InputError(f"git clone failed for {url}:\n{res.output.strip()[-800:]}")
    return dest


def resolve_repo(value: str, cfg: Any) -> Path:
    """A local directory, or a git URL cloned into the workspaces dir."""
    value = value.strip()
    if is_git_url(value):
        return clone_repo(value, cfg)
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise InputError(f"Repository directory not found: {path}")
    return path


def get_repo(args: argparse.Namespace, cfg: Any, ui: Any, interactive: bool) -> Path:
    """--repo, then $REPO, then an interactive prompt."""
    value = args.repo or os.environ.get("REPO", "")
    while True:
        if value.strip():
            try:
                return resolve_repo(value, cfg)
            except InputError as e:
                if not interactive:
                    raise
                ui.error(str(e))
        elif not interactive:
            raise InputError("No repository given. Use --repo <path|git URL> or REPO=<path> make run.")
        value = ui.ask("Path or git URL of the target repository:")


def fetch_github_issue(text: str) -> str | None:
    """Title + body of a GitHub issue URL via the public API, or None on any failure."""
    m = GITHUB_ISSUE_URL.match(text.strip())
    if not m:
        return None
    api = f"https://api.github.com/repos/{m.group(1)}/{m.group(2)}/issues/{m.group(3)}"
    try:
        req = urllib.request.Request(api, headers={"Accept": "application/vnd.github+json",
                                                   "User-Agent": "ai-coding-harness"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return f"{data.get('title', '')}\n\n{data.get('body') or ''}".strip() or None
    except Exception:  # noqa: BLE001 - any network/API problem means "ask for the text instead"
        return None


def paste_issue(ui: Any) -> str:
    """Read lines until a line containing only END, or EOF (Ctrl-D)."""
    console = getattr(ui, "console", None)
    (console.print if console else print)(PASTE_PROMPT)
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def get_issue(args: argparse.Namespace, ui: Any, interactive: bool, first: bool = True) -> str:
    """--issue-file, then --issue, then piped stdin, then an interactive paste. Resolves GitHub URLs."""
    text = ""
    if first and args.issue_file:
        path = Path(args.issue_file).expanduser()
        if not path.is_file():
            raise InputError(f"Issue file not found: {path}")
        text = path.read_text(encoding="utf-8", errors="replace")
    elif first and args.issue:
        text = args.issue
    elif first and not sys.stdin.isatty():
        text = sys.stdin.read()
    while True:
        if text.strip() and GITHUB_ISSUE_URL.match(text.strip()):
            fetched = fetch_github_issue(text)
            if fetched:
                return fetched
            if not interactive:
                raise InputError("Could not fetch that GitHub issue; pass the issue text with --issue-file.")
            ui.error("Could not fetch that GitHub issue; please paste its text instead.")
            text = ""
        if text.strip():
            return text.strip()
        if not interactive:
            raise InputError("No issue text given. Use --issue-file <file>, --issue <text>, or pipe it on stdin.")
        text = paste_issue(ui)


def prepare_demo(cfg: Any) -> tuple[Path, str]:
    """Copy the bundled sample repo to the workspaces dir and load the first sample issue."""
    if not DEMO_REPO.is_dir() or not DEMO_ISSUE.is_file():
        raise InputError("Demo fixtures not found (fixtures/sample_repo and fixtures/issues/01_slugify.md).")
    dest = Path(cfg.output.workspaces_dir) / f"demo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    shutil.copytree(DEMO_REPO, dest)
    return dest, DEMO_ISSUE.read_text(encoding="utf-8")


# ---------------------------------------------------------------------- running
def exit_code_for(status: str, strict: bool) -> int:
    """0 after any completed run; with --strict-exit: 0 verified, 2 unverified/no fix, 1 error."""
    if not strict:
        return 0
    return {"verified": 0, "error": 1}.get(status, 2)


def run_once(cfg: Any, llm: Any, ui: Any, repo: Path, issue: str, strict: bool) -> int:
    """Solve one issue and show the result. Ctrl-C returns 130 after the partial report is written."""
    from harness.orchestrator import Orchestrator
    from harness.report import final_attempt

    orch = Orchestrator(cfg, llm, ui)
    try:
        state, run_dir = orch.solve(repo, issue)
    except KeyboardInterrupt:
        run_dir = getattr(orch, "run_dir", None)
        ui.error(f"Interrupted — partial report at {run_dir}" if run_dir else "Interrupted")
        return 130
    diff = orch.ws.diff()
    attempt, _ = final_attempt(state, diff)
    ui.show_verification(attempt, state)
    ui.show_diff(diff)
    ui.show_summary(state, orch.metrics, {"report": run_dir / "report.md", "has_diff": bool(diff)})
    return exit_code_for(state.status, strict)


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the harness CLI."""
    p = argparse.ArgumentParser(prog="harness", description="Autonomous coding-agent harness.")
    p.add_argument("--repo", help="target repository: local path or git URL (or env REPO)")
    p.add_argument("--issue", help="issue text")
    p.add_argument("--issue-file", help="file containing the issue text (or a GitHub issue URL)")
    p.add_argument("--config", help="path to config.yaml (default: the project's config.yaml)")
    p.add_argument("--model", help="override model.name from config.yaml")
    p.add_argument("--non-interactive", action="store_true", help="never prompt; fail if inputs are missing")
    p.add_argument("--quiet", action="store_true", help="print only the final summary")
    p.add_argument("--strict-exit", action="store_true", help="exit 0 verified, 2 unverified/no fix, 1 error")
    p.add_argument("--self-check", action="store_true", help="import all modules, load config, and exit")
    p.add_argument("--ping", action="store_true", help="send one tiny request to the configured model")
    p.add_argument("--verbose", action="store_true", help="with --ping: show endpoint and probe details")
    p.add_argument("--demo", action="store_true", help="solve a bundled sample issue on a copy of the sample repo")
    p.add_argument("--version", action="store_true", help="print the version and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    from harness.config import load_config
    from harness.types import HarnessError

    args = build_parser().parse_args(argv)
    if args.version:
        print(harness.__version__)
        return 0
    if args.self_check:
        return self_check(args.config)
    try:
        cfg = load_config(args.config)
    except HarnessError as e:
        print(f"Error: {e}")
        return 1
    if args.model:
        cfg.model.name = args.model
    if args.ping:
        return ping(cfg, verbose=args.verbose)
    try:
        cfg.api_key()
    except HarnessError as e:
        print(f"Error: {e}")
        return 1

    from harness.ui import RichUI

    ui = RichUI(quiet=args.quiet)
    interactive = not args.non_interactive and not args.demo and sys.stdin.isatty()
    try:
        llm = make_llm(cfg, ui)
        ui.banner(cfg.model.name, llm.tool_mode, harness.__version__)
        if args.demo:
            repo, issue = prepare_demo(cfg)
        else:
            repo = get_repo(args, cfg, ui, interactive)
            issue = get_issue(args, ui, interactive)
        while True:
            code = run_once(cfg, llm, ui, repo, issue, args.strict_exit)
            if code == 130 or not interactive:
                return code
            if ui.ask("Solve another issue in the same repository? [y/N]").strip().lower() not in ("y", "yes"):
                return code
            issue = get_issue(args, ui, interactive, first=False)
    except KeyboardInterrupt:
        ui.error("Interrupted")
        return 130
    except (InputError, HarnessError) as e:
        ui.error(str(e))
        return 1

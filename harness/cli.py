"""Command-line interface: argument parsing, input collection, and self-check."""
from __future__ import annotations

import argparse
import importlib
import pkgutil

import harness


def _all_module_names() -> list[str]:
    """Return every importable harness.* module name, excluding the __main__ entry point."""
    names = ["harness"]
    for info in pkgutil.walk_packages(harness.__path__, prefix="harness."):
        if info.name.split(".")[-1] != "__main__":
            names.append(info.name)
    return sorted(names)


def self_check() -> int:
    """Import every harness module and report success. Needs no network and no API key."""
    for name in _all_module_names():
        importlib.import_module(name)
    print("Self-check OK")
    return 0


def ping(verbose: bool = False) -> int:
    """Probe tool mode and send one tiny request to the configured model."""
    from harness.config import load_config
    from harness.events import Trajectory
    from harness.llm import LLMClient
    from harness.types import HarnessError, Metrics

    try:
        cfg = load_config()
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


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the harness CLI."""
    parser = argparse.ArgumentParser(prog="harness", description="Autonomous coding-agent harness.")
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    parser.add_argument("--self-check", action="store_true", help="import all modules and exit")
    parser.add_argument("--ping", action="store_true", help="send one tiny request to the configured model")
    parser.add_argument("--verbose", action="store_true", help="with --ping: show endpoint and probe details")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    args = build_parser().parse_args(argv)
    if args.version:
        print(harness.__version__)
        return 0
    if args.self_check:
        return self_check()
    if args.ping:
        return ping(verbose=args.verbose)
    build_parser().print_help()
    return 0

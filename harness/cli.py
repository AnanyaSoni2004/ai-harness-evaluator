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


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the harness CLI."""
    parser = argparse.ArgumentParser(prog="harness", description="Autonomous coding-agent harness.")
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    parser.add_argument("--self-check", action="store_true", help="import all modules and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    args = build_parser().parse_args(argv)
    if args.version:
        print(harness.__version__)
        return 0
    if args.self_check:
        return self_check()
    build_parser().print_help()
    return 0

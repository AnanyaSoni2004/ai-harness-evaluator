"""Smoke tests: the package imports and the self-check passes."""
import harness
from harness.cli import main


def test_version_is_string() -> None:
    assert isinstance(harness.__version__, str)


def test_self_check() -> None:
    assert main(["--self-check"]) == 0

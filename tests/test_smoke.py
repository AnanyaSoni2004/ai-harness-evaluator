"""Smoke tests: the package imports and the self-check passes."""
import harness
from harness.cli import main


def test_version_is_string() -> None:
    assert isinstance(harness.__version__, str)


def test_self_check() -> None:
    assert main(["--self-check"]) == 0


def test_python_dash_m_entry_point() -> None:
    import subprocess
    import sys

    res = subprocess.run([sys.executable, "-m", "harness", "--version"], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0 and res.stdout.strip() == harness.__version__

"""Tests for the standalone tracer (harness/_trace_runner.py), run as a subprocess like in production."""
import json
import shlex
import sys
from pathlib import Path

import pytest

from harness.shell import run_process

RUNNER = Path(__file__).resolve().parents[1] / "harness" / "_trace_runner.py"
PY = sys.executable


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "mod.py").write_text(
        "LIMIT = 10\n\n\ndef add(a, b):\n    total = a + b\n    return total\n\n\n"
        "def clamp(x):\n    if x > LIMIT:\n        return LIMIT\n    return x\n")
    (root / "tests" / "test_mod.py").write_text(
        "from pkg.mod import add, clamp\n\n\ndef test_add():\n    assert add(1, 2) == 3\n\n\n"
        "def test_clamp_fails():\n    assert clamp(50) == 50\n")
    return root


def trace(root: Path, out: Path, *mode_args: str, extra: str = "", prefix: str = "") -> tuple:
    cmd = (f"{shlex.quote(PY)} {prefix}{shlex.quote(str(RUNNER))} --root {shlex.quote(str(root))} "
           f"--out {shlex.quote(str(out))} {extra} -- " + " ".join(shlex.quote(a) for a in mode_args))
    res = run_process(cmd, root, 60, env_extra={"PYTHONPATH": str(root)})
    return res, json.loads(out.read_text())


def test_script_mode_records_function_lines(repo: Path, tmp_path: Path) -> None:
    script = tmp_path / "scratch" / "repro.py"
    script.parent.mkdir()
    script.write_text("import json, sys\nfrom pkg.mod import add\njson.dumps({'a': 1})\n"
                      "print(add(2, 3))\nsys.exit(1)\n")
    res, data = trace(repo, tmp_path / "out.json", "script", str(script))
    assert res.exit_code == 1 and data["exit_code"] == 1 and data["error"] is None
    assert "5" in res.output
    run = data["runs"][0]
    assert run["id"] == "repro" and run["outcome"] == "failed"
    assert set(run["lines"]) == {"pkg/__init__.py", "pkg/mod.py"} or set(run["lines"]) == {"pkg/mod.py"}
    assert {5, 6} <= set(run["lines"]["pkg/mod.py"])  # add() body
    assert not {10, 11, 12} & set(run["lines"]["pkg/mod.py"])  # clamp() never ran


def test_stdlib_never_appears(repo: Path, tmp_path: Path) -> None:
    script = tmp_path / "repro.py"
    script.write_text("import json, textwrap\nfrom pkg.mod import add\ntextwrap.dedent(json.dumps([add(1, 1)]))\n")
    _, data = trace(repo, tmp_path / "out.json", "script", str(script))
    paths = list(data["runs"][0]["lines"])
    assert paths and all(not p.startswith("/") and ".." not in p and p.startswith("pkg/") for p in paths)


def test_pytest_mode_per_test_spectra(repo: Path, tmp_path: Path) -> None:
    res, data = trace(repo, tmp_path / "out.json", "pytest", "tests/test_mod.py")
    assert data["exit_code"] == 1 and data["error"] is None
    runs = {r["id"]: r for r in data["runs"]}
    assert runs["tests/test_mod.py::test_add"]["outcome"] == "passed"
    assert runs["tests/test_mod.py::test_clamp_fails"]["outcome"] == "failed"
    add_lines = set(runs["tests/test_mod.py::test_add"]["lines"]["pkg/mod.py"])
    clamp_lines = set(runs["tests/test_mod.py::test_clamp_fails"]["lines"]["pkg/mod.py"])
    assert {5, 6} <= add_lines and not {10, 11} & add_lines
    assert {10, 11} <= clamp_lines and not {5, 6} & clamp_lines
    partial = [json.loads(l) for l in (tmp_path / "out.json.partial.jsonl").read_text().splitlines()]
    assert [p["id"] for p in partial] == list(runs)  # incremental results for timeouts


def test_exclude_drops_test_files(repo: Path, tmp_path: Path) -> None:
    _, with_tests = trace(repo, tmp_path / "a.json", "pytest", "tests/test_mod.py")
    _, without = trace(repo, tmp_path / "b.json", "pytest", "tests/test_mod.py", extra="--exclude tests")
    assert any("tests/test_mod.py" in r["lines"] for r in with_tests["runs"])
    assert all("tests/test_mod.py" not in r["lines"] for r in without["runs"])
    assert all("pkg/mod.py" in r["lines"] for r in without["runs"])


def test_raising_script_still_writes_json(repo: Path, tmp_path: Path) -> None:
    script = tmp_path / "boom.py"
    script.write_text("from pkg.mod import clamp\nclamp(99)\nraise RuntimeError('boom')\n")
    res, data = trace(repo, tmp_path / "out.json", "script", str(script))
    assert res.exit_code == 1 and data["exit_code"] == 1 and data["exception"] == "RuntimeError"
    assert "RuntimeError: boom" in res.output
    assert {10, 11} <= set(data["runs"][0]["lines"]["pkg/mod.py"])


def test_existing_tracer_guard(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text(
        "import runpy, sys\nsys.settrace(lambda *a: None)\n"
        f"sys.argv = [{str(RUNNER)!r}, '--root', {str(repo)!r}, '--out', {str(out)!r}, '--', 'script', 'x.py']\n"
        f"runpy.run_path({str(RUNNER)!r}, run_name='__main__')\n")
    res = run_process(f"{shlex.quote(PY)} {shlex.quote(str(wrapper))}", repo, 60)
    assert res.exit_code == 3
    assert json.loads(out.read_text())["error"] == "another tracer is already active"


def test_runner_dir_does_not_shadow_target_modules(repo: Path, tmp_path: Path) -> None:
    (repo / "config.py").write_text("NAME = 'target config'\n")
    (repo / "testing.py").write_text("NAME = 'target testing'\n")
    script = tmp_path / "repro.py"
    script.write_text("import os, sys, types, config, testing\nassert types.SimpleNamespace\n"
                      "print(config.NAME, '|', testing.NAME)\n"
                      "print('HARNESS_ON_PATH', any(os.path.abspath(p or os.curdir) == {!r} for p in sys.path))\n"
                      .format(str(RUNNER.parent)))
    res, data = trace(repo, tmp_path / "out.json", "script", str(script))
    assert data["exit_code"] == 0, res.output
    assert "target config | target testing" in res.output
    assert "HARNESS_ON_PATH False" in res.output


def test_bad_usage(repo: Path, tmp_path: Path) -> None:
    res, data = trace(repo, tmp_path / "out.json", "teleport")
    assert res.exit_code == 3 and "unknown mode" in data["error"]

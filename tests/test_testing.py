"""Tests for TestRunner detection, parsing, comparison and real runs."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness.testing import TestRunner, compare, parse_generic, parse_pytest, parse_unittest
from harness.tools.registry import build_registry
from harness.types import TestRun, ToolCall
from harness.workspace import Workspace

PY = sys.executable

PASS_OUT = "....                                                                     [100%]\n4 passed in 0.03s\n"
FAIL_OUT = """..F.
=================================== FAILURES ===================================
___________________________________ test_b ____________________________________
    def test_b():
>       assert 1 == 2
E       assert 1 == 2
tests/test_a.py:5: AssertionError
=========================== short test summary info ============================
FAILED tests/test_a.py::test_b - assert 1 == 2
FAILED tests/test_a.py::test_c[1-2] - ValueError: boom
1 error, 2 failed, 3 passed in 0.05s
"""
COLLECT_OUT = """
==================================== ERRORS ====================================
_______________________ ERROR collecting tests/test_x.py _______________________
ImportError while importing test module '/r/tests/test_x.py'.
E   ModuleNotFoundError: No module named 'requests'
=========================== short test summary info ============================
ERROR tests/test_x.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.10s
"""
NO_TESTS_OUT = "\nno tests ran in 0.01s\n"


def run_of(failing=(), exit_code=1, failed=None, **kw) -> TestRun:
    base = dict(command="pytest", exit_code=exit_code, passed=3, failed=len(failing) if failed is None else failed,
                errors=0, failing_ids=list(failing), duration_s=0.1, timed_out=False, env_problem=False,
                output_tail="")
    base.update(kw)
    return TestRun(**base)


# ---------------------------------------------------------------- parsing
def test_parse_pytest_pass() -> None:
    assert parse_pytest(PASS_OUT, 0) == {"passed": 4, "failed": 0, "errors": 0, "failing_ids": [], "env_problem": False}


def test_parse_pytest_fail() -> None:
    p = parse_pytest(FAIL_OUT, 1)
    assert (p["passed"], p["failed"], p["errors"]) == (3, 2, 1)
    assert p["failing_ids"] == ["tests/test_a.py::test_b", "tests/test_a.py::test_c[1-2]"]
    assert not p["env_problem"]


def test_parse_pytest_collection_error() -> None:
    p = parse_pytest(COLLECT_OUT, 2)
    assert p["errors"] == 1 and p["failing_ids"] == ["tests/test_x.py"] and p["env_problem"]
    assert parse_pytest("/usr/bin/python3: No module named pytest\n", 1)["env_problem"]


def test_parse_pytest_no_tests_is_not_failure() -> None:
    p = parse_pytest(NO_TESTS_OUT, 5)
    assert (p["passed"], p["failed"], p["errors"]) == (0, 0, 0)


def test_parse_pytest_without_summary_counts_lines() -> None:
    p = parse_pytest("FAILED tests/t.py::a - x\nFAILED tests/t.py::b - y\n", None)
    assert p["failed"] == 2 and len(p["failing_ids"]) == 2


def test_parse_pytest_ignores_counts_in_test_output() -> None:
    out = "print: 99 passed things\n" + PASS_OUT
    assert parse_pytest(out, 0)["passed"] == 4


def test_parse_unittest() -> None:
    out = ("F.E\n======================================================================\n"
           "FAIL: test_add (test_calc.CalcTest.test_add)\n----\nERROR: test_div (test_calc.CalcTest)\n----\n"
           "Ran 3 tests in 0.001s\n\nFAILED (failures=1, errors=1)\n")
    p = parse_unittest(out, 1)
    assert (p["passed"], p["failed"], p["errors"]) == (1, 1, 1)
    assert p["failing_ids"] == ["test_calc.CalcTest.test_add", "test_calc.CalcTest.test_div"]
    assert parse_unittest("...\nRan 3 tests in 0.0s\n\nOK\n", 0)["passed"] == 3


def test_parse_generic() -> None:
    assert parse_generic("", 0)["passed"] == 1 and parse_generic("", 2)["failed"] == 1


# ---------------------------------------------------------------- compare
def test_compare_with_ids() -> None:
    before = run_of(["t::old", "t::flaky"])
    after = run_of(["t::old", "t::new"])
    c = compare(before, after)
    assert c == {"new_failures": ["t::new"], "fixed": ["t::flaky"], "still_failing": ["t::old"], "regression": True}
    assert not compare(before, run_of(["t::old"]))["regression"]


def test_compare_pre_existing_failure_is_not_new() -> None:
    before = run_of(["tests/test_paging.py::test_first_page"])
    assert compare(before, run_of(["tests/test_paging.py::test_first_page"]))["new_failures"] == []


def test_compare_without_ids_uses_exit_code() -> None:
    assert compare(run_of(exit_code=0, failed=0), run_of(exit_code=1, failed=1))["regression"]
    assert not compare(run_of(exit_code=1, failed=1), run_of(exit_code=1, failed=1))["regression"]
    assert not compare(run_of(exit_code=0, failed=0), run_of(exit_code=0, failed=0))["regression"]


def test_compare_timeout_env_and_no_baseline() -> None:
    assert "(test run timed out)" in compare(run_of(exit_code=0, failed=0),
                                             run_of(exit_code=None, failed=0, timed_out=True))["new_failures"]
    assert compare(run_of(exit_code=0, failed=0), run_of(["tests/x.py"], env_problem=True))["regression"]
    assert compare(None, run_of(["t::a"]))["new_failures"] == ["t::a"]
    assert not compare(None, run_of(exit_code=0, failed=0))["regression"]


# ---------------------------------------------------------------- detection
def make_ws(tmp_path: Path, files: dict) -> Workspace:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    for rel, content in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(content)
    return Workspace(repo, tmp_path / "scratch")


@pytest.mark.parametrize("files, kind, cmd_part", [
    ({"pytest.ini": "[pytest]\n"}, "pytest", "-m pytest -q -rfE -p no:cacheprovider"),
    ({"conftest.py": ""}, "pytest", "-m pytest"),
    ({"pyproject.toml": "[tool.pytest.ini_options]\n"}, "pytest", "-m pytest"),
    ({"setup.cfg": "[tool:pytest]\n"}, "pytest", "-m pytest"),
    ({"tox.ini": "[tox]\n"}, "pytest", "-m pytest"),
    ({"tests/unit/test_a.py": ""}, "pytest", "-m pytest"),
    ({"package.json": json.dumps({"scripts": {"test": "jest"}})}, "npm", "npm test --silent --"),
    ({"go.mod": "module x\n"}, "go", "go test ./..."),
    ({"Cargo.toml": "[package]\n"}, "cargo", "cargo test"),
    ({"pom.xml": "<project/>"}, "maven", "mvn -q test"),
    ({"build.gradle": "", "gradlew": ""}, "gradle", "./gradlew test"),
    ({"build.gradle.kts": ""}, "gradle", "gradle test"),
    ({"Makefile": "all:\n\techo\ntest:\n\techo t\n"}, "make", "make test"),
    ({"test_things.py": "import unittest\n"}, "unittest", "-m unittest discover -q"),
    ({"README.md": "hi"}, "unknown", ""),
    ({"package.json": json.dumps({"scripts": {"test": "echo \"Error: no test specified\" && exit 1"}})}, "unknown", ""),
    ({"Makefile": "build:\n\techo\n"}, "unknown", ""),
])
def test_detect(tmp_path: Path, files: dict, kind: str, cmd_part: str) -> None:
    info = TestRunner(make_ws(tmp_path, files), None, python_exe=PY).detect()
    assert info["kind"] == kind and cmd_part in info["base_cmd"]
    assert info["supports_targets"] == (kind == "pytest")


def test_detect_custom_and_repo_venv(tmp_path: Path) -> None:
    ws = make_ws(tmp_path, {"pytest.ini": "", ".venv/bin/python": ""})
    assert TestRunner(ws, None).detect()["base_cmd"].startswith(str(ws.repo_root / ".venv" / "bin" / "python"))
    cfg = SimpleNamespace(tests=SimpleNamespace(command="python3 -m pytest -x"))
    info = TestRunner(ws, cfg).detect()
    assert info == {"kind": "custom", "base_cmd": "python3 -m pytest -x", "supports_targets": True}


# ---------------------------------------------------------------- real runs
@pytest.fixture()
def tiny(tmp_path: Path) -> Workspace:
    return make_ws(tmp_path, {
        "pytest.ini": "[pytest]\ntestpaths = tests\n",
        "calc.py": "def add(a, b):\n    return a - b\n\ndef mul(a, b):\n    return a * b\n",
        "tests/test_calc.py": "from calc import add, mul\n\ndef test_add():\n    assert add(1, 2) == 3\n\n"
                              "def test_mul():\n    assert mul(2, 3) == 6\n",
        "tests/test_other.py": "def test_ok():\n    assert True\n",
    })


def test_real_run_and_fix(tiny: Workspace) -> None:
    runner = TestRunner(tiny, None, python_exe=PY)
    before = runner.run()
    assert (before.passed, before.failed) == (2, 1) and before.exit_code == 1
    assert before.failing_ids == ["tests/test_calc.py::test_add"] and not before.env_problem
    tiny.write_text("calc.py", "def add(a, b):\n    return a + b\n\ndef mul(a, b):\n    return a * b\n")
    after = runner.run()
    assert after.exit_code == 0 and after.passed == 3
    assert compare(before, after) == {"new_failures": [], "fixed": ["tests/test_calc.py::test_add"],
                                      "still_failing": [], "regression": False}
    targeted = runner.run(["tests/test_other.py"])
    assert targeted.passed == 1 and targeted.command.endswith("tests/test_other.py")


def test_real_run_timeout(tiny: Workspace) -> None:
    tiny.write_text("tests/test_slow.py", "import time\n\ndef test_slow():\n    time.sleep(30)\n")
    run = TestRunner(tiny, None, python_exe=PY).run(["tests/test_slow.py"], timeout_s=2)
    assert run.timed_out and run.exit_code is None


def test_unknown_kind_run(tmp_path: Path) -> None:
    run = TestRunner(make_ws(tmp_path, {"README.md": ""}), None, python_exe=PY).run()
    assert run.command == "" and "No test command detected" in run.output_tail


def test_run_tests_tool_and_registry(tiny: Workspace) -> None:
    runner = TestRunner(tiny, None, python_exe=PY)
    res = runner.run_tests_tool("tests/test_calc.py::test_add")
    assert not res.ok
    assert res.output.splitlines()[1].startswith("0 passed, 1 failed, 0 errors (exit 1")
    assert "Failing: tests/test_calc.py::test_add" in res.output
    assert runner.run_tests_tool("tests/test_other.py").ok
    assert "Could not parse targets" in runner.run_tests_tool("'unclosed").output
    cfg = SimpleNamespace(context=SimpleNamespace(max_tool_output_chars=8000),
                          safety=SimpleNamespace(command_timeout_s=30, allow_edit_existing_tests=False))
    reg = build_registry(tiny, cfg, test_runner=runner)
    assert "run_tests" in reg.names()
    assert "2 passed, 1 failed, 0 errors" in reg.dispatch(ToolCall("c", "run_tests", {})).output  # full suite

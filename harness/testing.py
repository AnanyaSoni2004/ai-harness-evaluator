"""TestRunner: detect, run, parse and compare test results; run_tests tool."""
from __future__ import annotations

import json
import re
import shlex
import sys
from typing import Any

from harness.shell import run_process
from harness.types import TestRun, ToolResult

OUTPUT_TAIL_CHARS = 4000
FAILURE_TAIL_LINES = 25
PROBE_TIMEOUT_S = 30.0
PYTEST_FLAGS = "-m pytest -q -rfE -p no:cacheprovider"

_COUNT = {name: re.compile(rf"(\d+) {name}") for name in ("passed", "failed")}
_ERRORS = re.compile(r"(\d+) errors?\b")
_SUMMARY = re.compile(r"\b\d+ (passed|failed|errors?|skipped|deselected|xfailed|xpassed)\b.*\bin [\d.]+s")
_FAILING = re.compile(r"^(FAILED|ERROR) (\S+)", re.MULTILINE)
_PROGRESS = re.compile(r"^([.FEsxX]+)\s*(?:\[\s*\d+%\])?\s*$", re.MULTILINE)  # "....F.s  [ 80%]"
_UNITTEST_RAN = re.compile(r"^Ran (\d+) tests? in", re.MULTILINE)
_UNITTEST_FAIL = re.compile(r"^(FAIL|ERROR): (\w+) \(([\w.]+)\)", re.MULTILINE)
_UNITTEST_KV = re.compile(r"(failures|errors|skipped)=(\d+)")


def _cfg(cfg: Any, section: str, name: str, default: Any) -> Any:
    """cfg.<section>.<name> with a default when cfg or the key is missing."""
    value = getattr(getattr(cfg, section, None), name, None) if cfg is not None else None
    return default if value is None else value


def _dedupe(items: list[str]) -> list[str]:
    """Remove duplicates, keeping the first occurrence."""
    return list(dict.fromkeys(items))


def parse_pytest(output: str, exit_code: int | None) -> dict:
    """Counts, failing test IDs and env_problem from pytest output (exit 5 = no tests, not a failure)."""
    summary = next((line for line in reversed(output.splitlines()) if _SUMMARY.search(line)), "")
    failing = _dedupe([m.group(2) for m in _FAILING.finditer(output)])
    passed = int(m.group(1)) if (m := _COUNT["passed"].search(summary)) else 0
    failed = int(m.group(1)) if (m := _COUNT["failed"].search(summary)) else 0
    errors = int(m.group(1)) if (m := _ERRORS.search(summary)) else 0
    if not summary:  # -qq (the repo's addopts adds a second -q), a crash or a timeout: count what we saw
        progress = "".join(m.group(1) for m in _PROGRESS.finditer(output))
        passed = progress.count(".")
        failed = sum(1 for m in _FAILING.finditer(output) if m.group(1) == "FAILED") or progress.count("F")
        errors = sum(1 for m in _FAILING.finditer(output) if m.group(1) == "ERROR") or progress.count("E")
    collection = "ERROR collecting" in output or "during collection" in output
    env_problem = ("No module named pytest" in output
                   or (collection and bool(re.search(r"\b(ModuleNotFoundError|ImportError)\b", output))))
    if exit_code == 5:
        failed, errors = 0, 0
    return {"passed": passed, "failed": failed, "errors": errors, "failing_ids": failing, "env_problem": env_problem}


def parse_unittest(output: str, exit_code: int | None) -> dict:
    """Counts and failing IDs from `python -m unittest` output."""
    ran = int(m.group(1)) if (m := _UNITTEST_RAN.search(output)) else 0
    tail = output[output.rfind("Ran "):] if "Ran " in output else output
    kv = {k: int(v) for k, v in _UNITTEST_KV.findall(tail)}
    failed, errors, skipped = kv.get("failures", 0), kv.get("errors", 0), kv.get("skipped", 0)
    ids = []
    for m in _UNITTEST_FAIL.finditer(output):
        name, where = m.group(2), m.group(3)
        ids.append(where if where.endswith("." + name) else f"{where}.{name}")
    if not ran and exit_code not in (0, None) and not ids:
        failed = 1
    env_problem = "ModuleNotFoundError" in output and not ran
    return {"passed": max(ran - failed - errors - skipped, 0), "failed": failed, "errors": errors,
            "failing_ids": _dedupe(ids), "env_problem": env_problem}


def parse_generic(output: str, exit_code: int | None) -> dict:
    """Frameworks we do not parse: the exit code is the only signal."""
    ok = exit_code == 0
    return {"passed": 1 if ok else 0, "failed": 0 if ok else 1, "errors": 0, "failing_ids": [], "env_problem": False}


def compare(before: TestRun | None, after: TestRun) -> dict:
    """Compare two runs: new failures, fixed tests, still failing, and whether it is a regression."""
    after_ids = list(after.failing_ids)
    before_ids = list(before.failing_ids) if before else []
    new = [t for t in after_ids if t not in before_ids]
    fixed = [t for t in before_ids if t not in after_ids]
    still = [t for t in after_ids if t in before_ids]
    after_failed = after.failed + after.errors > 0 or after.exit_code not in (0, 5)
    if before is not None and not after_ids and after_failed and before.exit_code in (0, 5):
        new.append(f"(test suite failed: exit {after.exit_code}; no per-test IDs)")
    if before is not None and after.timed_out and not before.timed_out:
        new.append("(test run timed out)")
    if before is not None and after.env_problem and not before.env_problem:
        new.append("(test environment/import problem)")
    new = _dedupe(new)
    return {"new_failures": new, "fixed": fixed, "still_failing": still, "regression": bool(new)}


class TestRunner:
    """Detects the target repo's test command, runs it, and parses the results."""

    __test__ = False  # not a pytest test class despite the name

    def __init__(self, ws: Any, cfg: Any = None, python_exe: str | None = None) -> None:
        """python_exe overrides the interpreter used for pytest/unittest (tests pass sys.executable)."""
        self.ws = ws
        self.cfg = cfg
        self.python_exe = python_exe
        self._detected: dict | None = None
        self._python_cache: str | None = None
        self.known_failures: set[str] = set()  # test IDs failing before any change (set by the orchestrator)

    # ------------------------------------------------------------------ detection
    def python(self) -> str:
        """The interpreter used for Python test runs (cached); the Tracer reuses it."""
        if self._python_cache is None:
            self._python_cache = self._python()
        return self._python_cache

    def _python(self) -> str:
        """Interpreter for Python test runs: repo venv, then python3 with pytest, then sys.executable."""
        if self.python_exe:
            return self.python_exe
        root = self.ws.repo_root
        for venv in (".venv", "venv"):
            candidate = root / venv / "bin" / "python"
            if candidate.exists():
                return str(candidate)
        probe = run_process("python3 -m pytest --version", root, PROBE_TIMEOUT_S)
        if probe.exit_code == 0:
            return "python3"
        return sys.executable

    def _has_pytest_markers(self) -> bool:
        """Any of the spec's pytest markers in the repo root."""
        root = self.ws.repo_root

        def contains(name: str, needle: str) -> bool:
            path = root / name
            return path.is_file() and needle in path.read_text(encoding="utf-8", errors="replace")

        if (root / "pytest.ini").is_file() or (root / "conftest.py").is_file() or (root / "tox.ini").is_file():
            return True
        if contains("pyproject.toml", "[tool.pytest") or contains("setup.cfg", "[tool:pytest]"):
            return True
        return any(any((root / d).glob("**/test_*.py")) for d in ("tests", "test") if (root / d).is_dir())

    def detect(self) -> dict:
        """{'kind', 'base_cmd', 'supports_targets'} for this repo (cached)."""
        if self._detected is None:
            self._detected = self._detect()
        return self._detected

    def _detect(self) -> dict:
        root = self.ws.repo_root
        custom = _cfg(self.cfg, "tests", "command", None)
        if custom:
            return {"kind": "custom", "base_cmd": custom, "supports_targets": "pytest" in custom}
        if self._has_pytest_markers():
            return {"kind": "pytest", "base_cmd": f"{shlex.quote(self.python())} {PYTEST_FLAGS}",
                    "supports_targets": True}
        package = root / "package.json"
        if package.is_file():
            try:
                script = str(json.loads(package.read_text(encoding="utf-8")).get("scripts", {}).get("test", ""))
            except (ValueError, AttributeError):
                script = ""
            if script and "no test specified" not in script:
                return {"kind": "npm", "base_cmd": "npm test --silent --", "supports_targets": False}
        simple = [("go.mod", "go", "go test ./..."), ("Cargo.toml", "cargo", "cargo test"),
                  ("pom.xml", "maven", "mvn -q test")]
        for marker, kind, cmd in simple:
            if (root / marker).is_file():
                return {"kind": kind, "base_cmd": cmd, "supports_targets": False}
        if (root / "build.gradle").is_file() or (root / "build.gradle.kts").is_file():
            cmd = "./gradlew test" if (root / "gradlew").is_file() else "gradle test"
            return {"kind": "gradle", "base_cmd": cmd, "supports_targets": False}
        makefile = root / "Makefile"
        if makefile.is_file() and re.search(r"^test\s*:", makefile.read_text(encoding="utf-8", errors="replace"),
                                            re.MULTILINE):
            return {"kind": "make", "base_cmd": "make test", "supports_targets": False}
        if any(p.name.startswith("test") and p.suffix == ".py" for p in self.ws.iter_source_files()):
            return {"kind": "unittest", "base_cmd": f"{shlex.quote(self.python())} -m unittest discover -q",
                    "supports_targets": False}
        return {"kind": "unknown", "base_cmd": "", "supports_targets": False}

    # ------------------------------------------------------------------ running
    def run(self, targets: list[str] | None = None, timeout_s: float | None = None) -> TestRun:
        """Run the suite (or the given targets where supported) and parse the result."""
        info = self.detect()
        if info["kind"] == "unknown":
            return TestRun(command="", exit_code=None, passed=0, failed=0, errors=0, failing_ids=[],
                           duration_s=0.0, timed_out=False, env_problem=False,
                           output_tail="No test command detected for this repository.")
        cmd = info["base_cmd"]
        if targets and info["supports_targets"]:
            cmd += " " + " ".join(shlex.quote(t) for t in targets)
        if timeout_s is None:
            key = "targeted_timeout_s" if targets else "baseline_timeout_s"
            timeout_s = float(_cfg(self.cfg, "tests", key, 180 if targets else 600))
        res = run_process(cmd, self.ws.repo_root, timeout_s, env_extra={"PYTHONPATH": self.ws.pythonpath()})
        uses_pytest = info["kind"] == "pytest" or (info["kind"] == "custom" and "pytest" in cmd)
        if uses_pytest:
            parsed = parse_pytest(res.output, res.exit_code)
        elif info["kind"] == "unittest":
            parsed = parse_unittest(res.output, res.exit_code)
        else:
            parsed = parse_generic(res.output, res.exit_code)
        tail = res.output[-OUTPUT_TAIL_CHARS:]
        return TestRun(command=cmd, exit_code=res.exit_code, duration_s=round(res.duration_s, 3),
                       timed_out=res.timed_out, output_tail=tail, **parsed)

    def run_tests_tool(self, targets: str | list = "") -> ToolResult:
        """Tool: run the tests (all, or space-separated files/test IDs) and summarise the result."""
        try:
            if isinstance(targets, (list, tuple)):  # models often send a JSON list despite the string schema
                target_list = [str(t).strip() for t in targets if str(t).strip()]
            else:
                target_list = shlex.split(str(targets or ""))
        except ValueError as e:
            return ToolResult(False, f"Could not parse targets: {e}")
        try:
            run = self.run(target_list or None)
        except Exception as e:  # noqa: BLE001 - tools must never raise
            return ToolResult(False, f"run_tests failed: {type(e).__name__}: {e}")
        if not run.command:
            return ToolResult(False, run.output_tail)
        info = self.detect()
        status = "TIMEOUT" if run.timed_out else f"exit {run.exit_code}"
        lines = [f"$ {run.command}",
                 f"{run.passed} passed, {run.failed} failed, {run.errors} errors ({status}, {run.duration_s:.1f}s)"]
        if target_list and not info["supports_targets"]:
            lines.append(f"(targets ignored: {info['kind']} cannot select tests here; ran the full suite)")
        if run.env_problem:
            lines.append("WARNING: the test environment looks broken (import/collection error), not necessarily "
                         "your change.")
        if run.failing_ids:
            labelled = [f"{t} (already failing before your change; ignore unless related to this issue)"
                        if t in self.known_failures else t for t in run.failing_ids[:20]]
            more = f" (+{len(run.failing_ids) - 20} more)" if len(run.failing_ids) > 20 else ""
            lines.append("Failing: " + ", ".join(labelled) + more)
            new = [t for t in run.failing_ids if t not in self.known_failures]
            if self.known_failures and not new:
                lines.append("No NEW failures: every failing test was already failing before your change.")
        ok = run.exit_code in (0, 5) and not run.timed_out
        if not ok:  # only what the agent needs: the failing IDs above and the end of the output
            tail = [line for line in run.output_tail.splitlines() if line.strip()][-FAILURE_TAIL_LINES:]
            lines.append("\n".join(tail))
        return ToolResult(ok, "\n".join(lines).rstrip(),
                          {"passed": run.passed, "failed": run.failed, "failing_ids": run.failing_ids})

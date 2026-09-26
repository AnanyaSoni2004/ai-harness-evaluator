"""Tests for the Tracer's scoring, function mapping and aggregation (pure functions)."""
from pathlib import Path

import pytest

from harness.spectrum import aggregate_functions, attach_functions, enclosing_functions, score_lines
from harness.workspace import Workspace


def run(outcome: str, lines: dict) -> dict:
    return {"id": outcome, "outcome": outcome, "lines": lines}


RUNS = [run("failed", {"a.py": [1, 2, 3]}), run("passed", {"a.py": [1, 2]}), run("passed", {"a.py": [1]})]


def by_line(scored: list) -> dict:
    return {d["line"]: d for d in scored}


# ---------------------------------------------------------------- formulas
def test_ochiai_values() -> None:
    s = by_line(score_lines(RUNS, "ochiai"))
    assert s[3]["score"] == pytest.approx(1.0)                        # ef=1, ep=0
    assert s[2]["score"] == pytest.approx(1 / 2 ** 0.5, abs=1e-4)     # ef=1, ep=1
    assert s[1]["score"] == pytest.approx(1 / 3 ** 0.5, abs=1e-4)     # ef=1, ep=2
    assert (s[1]["ef"], s[1]["ep"]) == (1, 2)


def test_tarantula_values() -> None:
    s = by_line(score_lines(RUNS, "tarantula"))
    assert s[3]["score"] == pytest.approx(1.0)
    assert s[2]["score"] == pytest.approx(1 / 1.5, abs=1e-4)          # (1/1) / (1/1 + 1/2)
    assert s[1]["score"] == pytest.approx(0.5)                        # (1/1) / (1/1 + 2/2)


def test_failing_only_line_ranks_first_and_everywhere_line_last() -> None:
    ranked = score_lines(RUNS)
    assert [d["line"] for d in ranked] == [3, 2, 1]


def test_no_failing_or_no_passing_runs_gives_nothing() -> None:
    assert score_lines([run("passed", {"a.py": [1]})]) == []
    assert score_lines([run("failed", {"a.py": [1]})]) == []
    assert score_lines([run("failed", {"a.py": [1]}), run("skipped", {"a.py": [1]})]) == []  # skipped ignored


def test_lines_only_in_passing_runs_are_never_ranked() -> None:
    scored = score_lines([run("failed", {"a.py": [1]}), run("passed", {"a.py": [1, 9]})])
    assert [d["line"] for d in scored] == [1]


# ---------------------------------------------------------------- function mapping
SOURCE = '''import os

LIMIT = 3


class Stock:
    def remove(self, n):
        current = 1
        return current - n

    class Inner:
        def deep(self):
            return 1


async def fetch():
    def helper():
        return 2
    return helper()


def outer():
    x = 1
    return x
'''


def test_enclosing_functions_nested_and_async() -> None:
    names = {name: (start, end) for name, start, end in enclosing_functions("m.py", SOURCE)}
    assert names["Stock.remove"] == (7, 9)
    assert names["Stock.Inner.deep"] == (12, 13)
    assert names["fetch"] == (16, 19) and names["fetch.helper"] == (17, 18)
    assert names["outer"] == (22, 24)
    assert enclosing_functions("bad.py", "def broken(:\n") == []


@pytest.fixture()
def ws(tmp_path: Path) -> Workspace:
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "m.py").write_text(SOURCE)
    (repo / "tests" / "test_m.py").write_text("def test_x():\n    assert True\n")
    return Workspace(repo, tmp_path / "scratch")


def item(path: str, line: int, score: float = 1.0, ef: int = 1, ep: int = 0) -> dict:
    return {"path": path, "line": line, "score": score, "ef": ef, "ep": ep}


def test_attach_innermost_and_drop_module_level_and_tests(ws: Workspace) -> None:
    lines = [item("m.py", 8), item("m.py", 18), item("m.py", 13), item("m.py", 3), item("m.py", 7),
             item("tests/test_m.py", 2), item("@scratch/repro.py", 1), item("missing.py", 1)]
    attached = {(d["line"]): d["function"] for d in attach_functions(lines, ws)}
    assert attached == {8: "Stock.remove", 18: "fetch.helper", 13: "Stock.Inner.deep"}  # 3 = module, 7 = def line


def test_aggregate_functions(ws: Workspace) -> None:
    lines = attach_functions([item("m.py", 8, 0.5, ep=3), item("m.py", 9, 0.9, ep=1), item("m.py", 23, 0.7, ep=2)], ws)
    funcs = aggregate_functions(lines, top_k=5)
    assert [(f["name"], f["score"], f["top_line"]) for f in funcs] == [("Stock.remove", 0.9, 9), ("outer", 0.7, 23)]
    assert funcs[0]["start"] == 7 and funcs[0]["end"] == 9 and funcs[0]["ep"] == 1
    assert len(aggregate_functions(lines, top_k=1)) == 1


# ---------------------------------------------------------------- ties
def test_ties_are_deterministic_and_hints_break_them(ws: Workspace) -> None:
    tied = [item("m.py", 24), item("m.py", 8)]
    plain = attach_functions(tied, ws)
    assert [d["line"] for d in plain] == [8, 24]  # same score/ef/ep -> path, then line
    assert [d["line"] for d in attach_functions(list(reversed(tied)), ws)] == [8, 24]  # input order irrelevant
    hinted = attach_functions(tied, ws, hints={"symbols": ["outer"], "files": []})
    assert [d["line"] for d in hinted] == [24, 8]  # a hint symbol lifts one of two equal lines
    by_suffix = attach_functions(tied, ws, hints={"symbols": ["Stock.remove"]})
    assert by_suffix[0]["function"] == "Stock.remove"


def test_file_hint_and_score_beat_hints() -> None:
    runs = [run("failed", {"b.py": [1], "a.py": [1]}), run("passed", {"z.py": [1]})]
    assert [d["path"] for d in score_lines(runs, hints={"files": ["b.py"]})] == ["b.py", "a.py"]
    stronger = [run("failed", {"a.py": [1], "b.py": [1]}), run("passed", {"b.py": [1]})]
    assert score_lines(stronger, hints={"files": ["b.py"]})[0]["path"] == "a.py"  # score first


# ---------------------------------------------------------------- T4: collection on the sample repo
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

from harness.config import load_config  # noqa: E402
from harness.spectrum import SpectrumAnalyzer, parse_repro_command  # noqa: E402
from harness.testing import TestRunner  # noqa: E402
from harness.types import IssueSpec, RunState, TestRun  # noqa: E402

PY = sys.executable
SAMPLE = Path(__file__).resolve().parents[1] / "fixtures" / "sample_repo"
BUG4_REPRO = ("from toolkit.inventory import Inventory\nimport sys\ninv = Inventory()\ninv.add('widget', 2)\n"
              "try:\n    inv.remove('widget', 5)\nexcept ValueError:\n    sys.exit(0)\n"
              "print('BUG PRESENT', inv.count('widget'))\nsys.exit(1)\n")
FIRST_PAGE = "tests/test_paging.py::test_first_page"


def test_parse_repro_command(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    assert parse_repro_command("python3 @scratch/repro.py", scratch) == ("script", [f"{scratch}/repro.py"])
    assert parse_repro_command(f'"{PY}" @scratch/repro.py --fast; echo "exit=$?"', scratch) == (
        "script", [f"{scratch}/repro.py", "--fast"])
    assert parse_repro_command("python -m pytest tests/x.py::t -x", scratch) == ("pytest", ["tests/x.py::t", "-x"])
    assert parse_repro_command("pytest tests/x.py", scratch) == ("pytest", ["tests/x.py"])
    assert parse_repro_command("npm test", scratch) is None
    assert parse_repro_command("cd sub && python3 r.py", scratch) is None
    assert parse_repro_command("python3 -c 'print(1)'", scratch) is None


def make_analyzer(tmp_path: Path, repro_src: str, **spectrum) -> tuple:
    repo = tmp_path / "repo"
    shutil.copytree(SAMPLE, repo)
    ws = Workspace(repo, tmp_path / "scratch")
    ws.write_text("@scratch/repro.py", repro_src)
    cfg = load_config()
    for key, value in spectrum.items():
        setattr(cfg.spectrum, key, value)
    state = RunState(run_id="r", repo=str(repo), issue=IssueSpec(raw_text="x"))
    state.repro = {"reproduced": True, "command": f'"{PY}" @scratch/repro.py'}
    state.localization = {"files": ["toolkit/inventory.py"], "symbols": ["Inventory.remove"]}
    state.targeted_tests = ["tests/test_inventory.py"]
    state.baseline_full = TestRun("pytest", 1, 15, 1, 0, [FIRST_PAGE], 0.1, False, False, "")
    return SpectrumAnalyzer(ws, cfg, TestRunner(ws, cfg, python_exe=PY)), state, ws


def test_bug4_ranks_inventory_remove_first(tmp_path: Path) -> None:
    analyzer, state, _ = make_analyzer(tmp_path, BUG4_REPRO)
    result = analyzer.analyze(state)
    assert result.ok, result.reason
    assert result.functions[0]["name"] == "Inventory.remove"
    assert result.functions[0]["path"] == "toolkit/inventory.py"
    assert result.failing_runs == 1 and result.passing_runs == 3
    assert all(not l["path"].startswith("tests/") for l in result.lines)


def test_pre_existing_failure_never_a_failing_run(tmp_path: Path) -> None:
    analyzer, state, _ = make_analyzer(tmp_path, BUG4_REPRO)
    state.targeted_tests = ["tests/test_paging.py", "tests/test_inventory.py"]
    result = analyzer.analyze(state)
    assert result.ok and result.failing_runs == 1  # only the repro
    assert result.passing_runs == 5  # 3 inventory + 2 paging; test_first_page excluded entirely


def test_repro_that_exits_zero(tmp_path: Path) -> None:
    analyzer, state, _ = make_analyzer(tmp_path, "print('fine')\n")
    result = analyzer.analyze(state)
    assert not result.ok and result.reason == "repro did not fail under tracer"


def test_timeout_is_bounded(tmp_path: Path) -> None:
    analyzer, state, _ = make_analyzer(tmp_path, "import time\ntime.sleep(60)\n", timeout_s=3, repro_timeout_s=2)
    start = time.monotonic()
    result = analyzer.analyze(state)
    assert time.monotonic() - start < 3 + 5
    assert not result.ok and "timed out" in result.reason


def test_zero_passing_runs_is_not_a_ranking_of_ties(tmp_path: Path) -> None:
    analyzer, state, ws = make_analyzer(tmp_path, BUG4_REPRO)
    shutil.rmtree(ws.repo_root / "tests")  # nothing to contrast against, even after the top-up
    state.targeted_tests = []
    result = analyzer.analyze(state)
    assert not result.ok and result.reason == "only 0 passing runs; no contrast"


def test_top_up_when_targeted_tests_are_too_few(tmp_path: Path) -> None:
    analyzer, state, _ = make_analyzer(tmp_path, BUG4_REPRO)
    state.targeted_tests = []
    result = analyzer.analyze(state)
    assert result.ok and result.passing_runs >= 3


@pytest.mark.parametrize("change, reason", [
    (lambda s, ws, a: setattr(a.scfg, "enabled", False), "disabled"),
    (lambda s, ws, a: s.repro.update(reproduced=False), "no reproduction"),
    (lambda s, ws, a: s.repro.update(command="npm test"), "unsupported repro command"),
    (lambda s, ws, a: ws.write_text("toolkit/stats.py", "X = 1\n"), "repo already modified"),
])
def test_skip_reasons(tmp_path: Path, change, reason) -> None:
    analyzer, state, ws = make_analyzer(tmp_path, BUG4_REPRO)
    change(state, ws, analyzer)
    assert analyzer.analyze(state).reason == reason


def test_crash_becomes_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    analyzer, state, _ = make_analyzer(tmp_path, BUG4_REPRO)
    monkeypatch.setattr(analyzer.runner, "detect", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    result = analyzer.analyze(state)
    assert not result.ok and result.reason == "crashed: RuntimeError: boom" and result.seconds >= 0


# ---------------------------------------------------------------- T5: evidence block and diff helpers
from harness.spectrum import changed_lines, format_suspicious, touched_ranks  # noqa: E402

INV = SAMPLE / "toolkit" / "inventory.py"


def bug4_result(tmp_path: Path):
    analyzer, state, ws = make_analyzer(tmp_path, BUG4_REPRO)
    return analyzer.analyze(state), ws


def test_format_suspicious_block(tmp_path: Path) -> None:
    result, ws = bug4_result(tmp_path)
    text = format_suspicious(result, ws, top_lines=10, max_chars=5000)
    lines = text.splitlines()
    assert lines[0] == "EXECUTION EVIDENCE (spectrum-based fault localization, Ochiai; 1 failing run, 3 passing runs):"
    assert lines[2] == "#1 toolkit/inventory.py  Inventory.remove (L16-L21)  score 0.71"
    marked = [l for l in lines if l.startswith("   >> ")]
    assert marked and all("[score " in l and "fail 1/1, pass" in l for l in marked)
    assert "   >>  18 |         if qty <= 0:        [score 0.71, fail 1/1, pass 1/3]" in lines
    assert "       19 |             raise ValueError(\"quantity must be positive\")" in lines  # context
    assert "   >>  21 |         self._stock[item] = current - qty        [score 0.71, fail 1/1, pass 1/3]" in lines
    assert text.endswith("Confirm with view_file before editing.")
    assert "#2 " in text and "{" not in text.replace("{}", "")


def test_format_suspicious_merges_adjacent_lines_and_is_empty_when_not_ok(tmp_path: Path) -> None:
    result, ws = bug4_result(tmp_path)
    text = format_suspicious(result, ws)
    remove_block = text.split("#1 ", 1)[1].split("\n#2 ", 1)[0]
    numbers = [int(l.split("|")[0].split()[-1]) for l in remove_block.splitlines()[1:] if "|" in l]
    assert numbers == sorted(set(numbers)) and numbers == list(range(numbers[0], numbers[-1] + 1))
    assert format_suspicious({"ok": False, "reason": "no reproduction"}, ws) == ""
    assert format_suspicious({}, ws) == ""


def test_format_suspicious_low_confidence_and_truncation(tmp_path: Path) -> None:
    result, ws = bug4_result(tmp_path)
    result.low_confidence = True
    text = format_suspicious(result.to_dict(), ws)  # dict form (state.spectrum) accepted too
    assert text.endswith("narrows the search only slightly — rely on your own analysis.")
    short = format_suspicious(result, ws, max_chars=600)
    assert len(short) < 700 and "chars truncated" in short and short.endswith("own analysis.")


DIFF = """--- a/toolkit/inventory.py
+++ b/toolkit/inventory.py
@@ -16,6 +16,8 @@
     def remove(self, item, qty=1):
         \"\"\"Remove `qty` units of `item`.\"\"\"
         if qty <= 0:
             raise ValueError("quantity must be positive")
         current = self._stock.get(item, 0)
+        if qty > current:
+            raise ValueError("insufficient stock")
         self._stock[item] = current - qty
 
     def count(self, item):
--- /dev/null
+++ b/tests/test_new.py
@@ -0,0 +1,2 @@
+def test_x():
+    assert True
"""


def test_changed_lines_and_touched_ranks() -> None:
    touched = changed_lines(DIFF)
    assert touched["toolkit/inventory.py"] == {20}  # insertion after original line 20, context ignored
    assert touched["tests/test_new.py"] == {1}
    functions = [{"path": "toolkit/inventory.py", "name": "Inventory.count", "start": 23, "end": 25},
                 {"path": "toolkit/inventory.py", "name": "Inventory.remove", "start": 16, "end": 21}]
    assert touched_ranks(DIFF, functions) == [2]  # context lines 22-23 do not count as touching count()
    assert touched_ranks("", functions) == []

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

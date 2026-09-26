"""Tests for str_replace, create_file and check_syntax."""
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness.tools.edit_tools import check_syntax, create_file, str_replace
from harness.workspace import Workspace

HAS_NODE = shutil.which("node") is not None
SRC = "def add(a, b):\n    return a - b\n\n\ndef sub(a, b):\n    return a - b\n"
TEST = "from pkg.calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"


def cfg(allow_tests: bool = False) -> SimpleNamespace:
    return SimpleNamespace(safety=SimpleNamespace(allow_edit_existing_tests=allow_tests))


@pytest.fixture()
def ws(tmp_path: Path) -> Workspace:
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "pkg" / "calc.py").write_text(SRC)
    (repo / "tests" / "test_calc.py").write_text(TEST)
    w = Workspace(repo, tmp_path / "scratch")
    w.write_scope = "repo"
    return w


# ---------------------------------------------------------------- str_replace
def test_unique_replace(ws: Workspace) -> None:
    res = str_replace(ws, cfg(), path="pkg/calc.py", old_str="def add(a, b):\n    return a - b",
                      new_str="def add(a, b):\n    return a + b")
    assert res.ok and res.output.startswith("Edit applied to pkg/calc.py.")
    assert "    2 |     return a + b" in res.output
    assert (ws.repo_root / "pkg" / "calc.py").read_text().startswith("def add(a, b):\n    return a + b\n")
    assert ws.edited_files() == ["pkg/calc.py"]


def test_zero_match_shows_similar_regions(ws: Workspace) -> None:
    res = str_replace(ws, cfg(), path="pkg/calc.py", old_str="def add(a, b):\n  return a - b", new_str="x")
    assert not res.ok and "old_str was not found" in res.output
    assert "similar region (lines 1-2" in res.output and "    1 | def add(a, b):" in res.output
    assert "including indentation" in res.output
    assert res.output.count("--- similar region") == 2  # the blank-line region (<30% similar) is dropped
    assert "lines 5-6" in res.output


def test_zero_match_always_shows_best_region(ws: Workspace) -> None:
    res = str_replace(ws, cfg(), path="pkg/calc.py", old_str="completely unrelated text", new_str="x")
    assert res.output.count("--- similar region") == 1


def test_zero_match_line_number_prefix_hint(ws: Workspace) -> None:
    res = str_replace(ws, cfg(), path="pkg/calc.py", old_str="    2 |     return a - b", new_str="x")
    assert "line-number prefixes" in res.output and "remove them" in res.output


def test_multiple_matches_lists_lines(ws: Workspace) -> None:
    res = str_replace(ws, cfg(), path="pkg/calc.py", old_str="    return a - b", new_str="    return a + b")
    assert not res.ok and "matches 2 times" in res.output and "(lines 2, 6)" in res.output
    assert "more surrounding lines" in res.output


def test_basic_argument_errors(ws: Workspace) -> None:
    assert "use create_file" in str_replace(ws, cfg(), path="pkg/nope.py", old_str="a", new_str="b").output
    assert "must not be empty" in str_replace(ws, cfg(), path="pkg/calc.py", old_str="", new_str="b").output
    assert "identical" in str_replace(ws, cfg(), path="pkg/calc.py", old_str="add", new_str="add").output
    assert not str_replace(ws, cfg(), path="../x.py", old_str="a", new_str="b").ok


def test_syntax_guard_refuses_broken_edit(ws: Workspace) -> None:
    res = str_replace(ws, cfg(), path="pkg/calc.py", old_str="def add(a, b):", new_str="def add(a, b)")
    assert not res.ok and "Edit NOT applied" in res.output and "(line 1)" in res.output
    assert "    1 | def add(a, b)" in res.output
    assert (ws.repo_root / "pkg" / "calc.py").read_text() == SRC
    assert ws.edited_files() == []


def test_syntax_guard_allows_edit_of_already_broken_file(ws: Workspace) -> None:
    (ws.repo_root / "pkg" / "broken.py").write_text("def f(:\n    pass\n")
    res = str_replace(ws, cfg(), path="pkg/broken.py", old_str="    pass", new_str="    return 1")
    assert res.ok


def test_test_protection(ws: Workspace) -> None:
    blocked = str_replace(ws, cfg(), path="tests/test_calc.py", old_str="== 3", new_str="== -1")
    assert not blocked.ok and "Modifying existing test code is not allowed" in blocked.output
    addition = "    assert add(1, 2) == 3\n\n\ndef test_add_zero():\n    assert add(0, 0) == 0"
    added = str_replace(ws, cfg(), path="tests/test_calc.py", old_str="    assert add(1, 2) == 3", new_str=addition)
    assert added.ok
    assert str_replace(ws, cfg(allow_tests=True), path="tests/test_calc.py", old_str="== 0", new_str="== 0 + 0").ok


def test_new_test_file_can_be_edited(ws: Workspace) -> None:
    assert create_file(ws, cfg(), path="tests/test_new.py", content="def test_x():\n    assert 1\n").ok
    assert str_replace(ws, cfg(), path="tests/test_new.py", old_str="assert 1", new_str="assert 2").ok


def test_scope_enforcement(ws: Workspace) -> None:
    ws.write_text("@scratch/test_repro.py", "print('BUG PRESENT')\n")
    ws.write_scope = "none"
    assert "not allowed in this phase" in str_replace(ws, cfg(), path="pkg/calc.py", old_str="add", new_str="plus").output
    assert not create_file(ws, cfg(), path="@scratch/x.py", content="").ok
    ws.write_scope = "scratch"
    res = str_replace(ws, cfg(), path="pkg/calc.py", old_str="def add", new_str="def plus")
    assert not res.ok and "only write under @scratch/" in res.output
    assert not create_file(ws, cfg(), path="pkg/new.py", content="").ok
    scratch_edit = str_replace(ws, cfg(), path="@scratch/test_repro.py", old_str="PRESENT", new_str="FIXED")
    assert scratch_edit.ok  # scratch test-named files are not "existing tests"
    assert ws.edited_files() == []


def test_crlf_file(ws: Workspace) -> None:
    path = ws.repo_root / "pkg" / "win.py"
    path.write_bytes(b"x = 1\r\ny = 2\r\n")
    res = str_replace(ws, cfg(), path="pkg/win.py", old_str="x = 1\ny = 2", new_str="x = 1\ny = 3")
    assert res.ok
    assert path.read_bytes() == b"x = 1\r\ny = 3\r\n"


# ---------------------------------------------------------------- create_file
def test_create_file(ws: Workspace) -> None:
    res = create_file(ws, cfg(), path="pkg/util/helpers.py", content="A = 1\nB = 2\n")
    assert res.ok and res.output == "Created pkg/util/helpers.py (2 lines)."
    assert "exists; use str_replace" in create_file(ws, cfg(), path="pkg/calc.py", content="").output
    assert "directory" in create_file(ws, cfg(), path="pkg", content="").output


def test_create_file_syntax_guard(ws: Workspace) -> None:
    res = create_file(ws, cfg(), path="pkg/bad.py", content="def f(:\n")
    assert not res.ok and "File NOT created" in res.output
    assert not (ws.repo_root / "pkg" / "bad.py").exists()
    assert not create_file(ws, cfg(), path="data.json", content='{"a": 1,}').ok


def test_create_scratch_overwrite(ws: Workspace) -> None:
    assert create_file(ws, cfg(), path="@scratch/repro.py", content="print(1)\n").ok
    assert create_file(ws, cfg(), path="@scratch/repro.py", content="print(2)\n").ok
    assert (ws.scratch_dir / "repro.py").read_text() == "print(2)\n"
    assert ws.diff() == ""


# ---------------------------------------------------------------- check_syntax
def test_check_syntax_by_type() -> None:
    assert check_syntax(Path("a.py"), "x = 1\n") == (True, "")
    ok, msg = check_syntax(Path("a.py"), "x = (1,\ny = 2\n")
    assert not ok and "SyntaxError" in msg and "line" in msg
    assert check_syntax(Path("a.json"), '{"a": [1, 2]}')[0]
    ok, msg = check_syntax(Path("a.json"), '{\n  "a": 1,\n}')
    assert not ok and "JSON error" in msg and "(line 3)" in msg
    assert check_syntax(Path("a.yaml"), "a: 1\n---\nb: 2\n")[0]  # multi-document YAML is valid
    ok, msg = check_syntax(Path("a.yml"), "a: [1, 2\nb: 3\n")
    assert not ok and "YAML error" in msg
    assert check_syntax(Path("README.md"), "def f(:")[0]


@pytest.mark.skipif(not HAS_NODE, reason="node not installed")
def test_check_syntax_javascript() -> None:
    assert check_syntax(Path("a.js"), "const a = 1;\nfunction f() { return a; }\n")[0]
    assert check_syntax(Path("a.js"), 'import x from "y";\nexport const a = x;\n')[0]  # ES module syntax
    assert check_syntax(Path("a.cjs"), "module.exports = { a: 1 };\n")[0]
    ok, msg = check_syntax(Path("a.js"), "const a = 1;\nfunction f( {\n")
    assert not ok and msg.startswith("SyntaxError") and "SyntaxError: SyntaxError" not in msg


def test_check_syntax_js_without_node(monkeypatch: pytest.MonkeyPatch) -> None:
    from harness.tools import edit_tools
    monkeypatch.setattr(edit_tools.shutil, "which", lambda name: None)
    assert check_syntax(Path("a.js"), "function f( {")[0]

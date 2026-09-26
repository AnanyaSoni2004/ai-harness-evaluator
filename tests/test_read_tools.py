"""Tests for the read-only tools (with and without ripgrep)."""
import shutil
import stat
from pathlib import Path

import pytest

from harness.tools import read_tools
from harness.tools.read_tools import find_files, list_dir, search_code, view_file
from harness.workspace import Workspace

REAL_RG = shutil.which("rg")


@pytest.fixture()
def ws(tmp_path: Path) -> Workspace:
    repo = tmp_path / "repo"
    files = {
        "pkg/__init__.py": "",
        "pkg/mod.py": "def parse(x):\n    return int(x)  # parse it\n",
        "pkg/util/helpers.py": "PARSE_LIMIT = 10\n",
        "pkg/util/deep/deeper/deepest/leaf.py": "x = 1\n",
        "tests/test_mod.py": "from pkg.mod import parse\n\ndef test_parse():\n    assert parse('1') == 1\n",
        "config.py": "DEBUG = False\n",
        "node_modules/lib/parse.js": "function parse() {}\n",
        ".venv/lib/site.py": "parse = 1\n",
        "big.py": "".join(f"line {i}\n" for i in range(1, 601)),
    }
    for rel, content in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (repo / "logo.png").write_bytes(b"\x89PNG\0parse")
    return Workspace(repo, tmp_path / "scratch")


@pytest.fixture()
def no_rg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(read_tools.shutil, "which", lambda name: None)


# ---------------------------------------------------------------- list_dir
def test_list_dir_tree(ws: Workspace) -> None:
    res = list_dir(ws, None)
    assert res.ok
    lines = res.output.splitlines()
    assert lines[0] == "./"
    assert "  pkg/" in lines and "    mod.py" in lines and "    util/" in lines
    assert "      helpers.py" not in lines  # depth 2 by default
    assert not any("node_modules" in l or ".venv" in l for l in lines)
    assert lines.index("  pkg/") < lines.index("  big.py")  # directories first


def test_list_dir_depth_clamped_and_subdir(ws: Workspace) -> None:
    res = list_dir(ws, None, path="pkg", depth=99)
    assert res.output.splitlines()[0] == "pkg/"
    assert "leaf.py" not in res.output  # depth capped at 4
    assert "deepest/" in res.output
    assert list_dir(ws, None, path="pkg", depth="1").output.count("\n") == 3


def test_list_dir_errors(ws: Workspace) -> None:
    assert not list_dir(ws, None, path="nope").ok
    assert "view_file" in list_dir(ws, None, path="config.py").output
    assert not list_dir(ws, None, path="../..").ok
    assert "integer" in list_dir(ws, None, depth="deep").output


def test_list_dir_entry_cap(ws: Workspace) -> None:
    many = ws.repo_root / "many"
    many.mkdir()
    for i in range(320):
        (many / f"f{i:03}.txt").write_text("x")
    res = list_dir(ws, None, path="many")
    assert res.data["truncated"] and "stopped at 300 entries" in res.output
    assert res.output.count(".txt") == 300


# ---------------------------------------------------------------- find_files
def test_find_files_by_name_and_path(ws: Workspace) -> None:
    assert find_files(ws, None, pattern="helpers.py").output == "pkg/util/helpers.py"
    out = find_files(ws, None, pattern="pkg/*.py").output.splitlines()
    assert "pkg/mod.py" in out and "pkg/__init__.py" in out
    assert "config.py" in find_files(ws, None, pattern="**/*config*.py").output.splitlines()
    assert "node_modules" not in find_files(ws, None, pattern="*.js").output


def test_find_files_empty_and_missing_pattern(ws: Workspace) -> None:
    res = find_files(ws, None, pattern="*.rs")
    assert res.ok and "No files match" in res.output
    assert not find_files(ws, None).ok


def test_find_files_cap(ws: Workspace) -> None:
    for i in range(120):
        (ws.repo_root / f"gen_{i:03}.txt").write_text("x")
    res = find_files(ws, None, pattern="gen_*.txt")
    assert res.output.count("gen_") == 100 and "20 more files" in res.output


# ---------------------------------------------------------------- search_code (python fallback)
def test_search_python_literal(ws: Workspace, no_rg: None) -> None:
    res = search_code(ws, None, query="parse")
    lines = res.output.splitlines()
    assert "pkg/mod.py:" in lines and ">>    1 | def parse(x):" in lines
    assert "tests/test_mod.py:" in lines and ">>    1 | from pkg.mod import parse" in lines
    assert "      2 |     return int(x)  # parse it" not in lines  # line 2 is also a match (marked >>)
    assert ">>    2 |     return int(x)  # parse it" in lines
    assert not any("node_modules" in l or ".venv" in l or "logo.png" in l for l in lines)
    assert "PARSE_LIMIT" not in res.output  # case-sensitive literal search


def test_search_python_regex_glob_and_path(ws: Workspace, no_rg: None) -> None:
    assert search_code(ws, None, query=r"def \w+\(", regex=True).output.startswith("pkg/mod.py:\n>>    1 | def parse")
    res = search_code(ws, None, query="parse", file_glob="test_*.py")
    assert [l for l in res.output.splitlines() if l.endswith(":") and " | " not in l] == ["tests/test_mod.py:"]
    res = search_code(ws, None, query="parse", path="pkg")
    assert all(l.startswith("pkg/") for l in res.output.splitlines() if l.endswith(":") and " | " not in l)
    assert not search_code(ws, None, query="(", regex=True).ok


def test_search_python_no_match_and_cap(ws: Workspace, no_rg: None) -> None:
    res = search_code(ws, None, query="zzz_nothing")
    assert res.ok and res.output == "No matches for 'zzz_nothing'. Try a shorter or different term."
    res = search_code(ws, None, query="line ", path="big.py")
    assert res.output.count(">> ") == 25 and "575 more matches; narrow the query" in res.output
    assert f"   {26:>4} | line 26" in res.output and f"   {29:>4} |" not in res.output  # 3 context lines
    assert res.output.count("big.py:") == 1  # overlapping windows merged into one block
    assert not search_code(ws, None, query="").ok


def test_search_long_line_truncated(ws: Workspace, no_rg: None) -> None:
    (ws.repo_root / "long.py").write_text("needle = '" + "x" * 500 + "'\n")
    out = search_code(ws, None, query="needle").output
    assert out.splitlines()[1].endswith("...") and len(out.splitlines()[1]) < 220


# ---------------------------------------------------------------- search_code (ripgrep)
def test_search_uses_rg_when_available(ws: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = tmp_path / "bin" / "rg"
    fake.parent.mkdir()
    fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$(dirname "$0")/args.txt"\n'
                    'eval last=\\${$#}\necho "$last/pkg/mod.py:2:    return int(x)  # parse it"\n')
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(read_tools.shutil, "which", lambda name: str(fake))
    res = search_code(ws, None, query="return int", file_glob="*.py")
    assert res.output == f"pkg/mod.py:\n   {1:>4} | def parse(x):\n>> {2:>4} |     return int(x)  # parse it"
    args = (fake.parent / "args.txt").read_text().splitlines()
    assert args[:3] == ["--line-number", "--no-heading", "--color"]
    assert "-F" in args and "!node_modules/" in args and "*.py" in args
    assert args[-2:] == ["return int", str(ws.repo_root)]


@pytest.mark.skipif(REAL_RG is None, reason="ripgrep not installed")
def test_search_real_rg(ws: Workspace) -> None:
    res = search_code(ws, None, query="parse")
    assert ">>    1 | def parse(x):" in res.output.splitlines()
    assert "node_modules" not in res.output and ".venv" not in res.output
    assert search_code(ws, None, query="zzz_nothing").output.startswith("No matches")


@pytest.mark.skipif(REAL_RG is None, reason="ripgrep not installed")
def test_rg_and_python_backends_agree(ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    (ws.repo_root / ".github").mkdir()
    (ws.repo_root / ".github" / "ci.yml").write_text("run: line 1\n")
    queries = [("parse", {}), ("line ", {}), (r"def \w+", {"regex": True}), ("parse", {"file_glob": "*.py"}),
               ("parse", {"path": "pkg"}), ("zzz_nothing", {})]
    rg_out = [search_code(ws, None, query=q, **kw).output for q, kw in queries]
    monkeypatch.setattr(read_tools.shutil, "which", lambda name: None)
    py_out = [search_code(ws, None, query=q, **kw).output for q, kw in queries]
    assert rg_out == py_out


# ---------------------------------------------------------------- view_file
def test_view_file_basic(ws: Workspace) -> None:
    res = view_file(ws, None, path="pkg/mod.py")
    assert res.output.splitlines() == [
        "File: pkg/mod.py (2 lines) — showing 1-2",
        "    1 | def parse(x):",
        "    2 |     return int(x)  # parse it",
    ]


def test_view_file_paging(ws: Workspace) -> None:
    res = view_file(ws, None, path="big.py")  # long file, no start_line: outline + first 40 lines
    assert res.output.splitlines()[0] == "File: big.py (600 lines) — outline, then lines 1-40"
    assert "   40 | line 40" in res.output and "   41 | line 41" not in res.output
    assert "start_line (and end_line)" in res.output and res.data["outline"]
    res = view_file(ws, None, path="big.py", start_line=1)  # default window: 80 lines
    assert "showing 1-80" in res.output and res.output.endswith("[520 more lines — call view_file with start_line=81]")
    res = view_file(ws, None, path="big.py", start_line=100, end_line=9999)  # hard cap: 200 lines
    assert "showing 100-299" in res.output
    res = view_file(ws, None, path="big.py", start_line="590", end_line=9999)
    assert "showing 590-600" in res.output and "more lines" not in res.output
    res = view_file(ws, None, path="big.py", start_line=10, end_line=12)
    assert res.output.count(" | ") == 3 and "[588 more lines" in res.output


def test_view_file_errors(ws: Workspace) -> None:
    assert "past the end" in view_file(ws, None, path="pkg/mod.py", start_line=50).output
    assert "must be >=" in view_file(ws, None, path="big.py", start_line=10, end_line=5).output
    assert "directory" in view_file(ws, None, path="pkg").output
    assert "not found" in view_file(ws, None, path="nope.py").output
    assert "binary" in view_file(ws, None, path="logo.png").output
    assert not view_file(ws, None, path="../../etc/passwd").ok
    assert not view_file(ws, None).ok
    assert view_file(ws, None, path="pkg/__init__.py").output.endswith("(empty file)")


def test_view_file_scratch(ws: Workspace) -> None:
    ws.write_text("@scratch/repro.py", "print('hi')\n")
    assert "File: @scratch/repro.py (1 lines)" in view_file(ws, None, path="@scratch/repro.py").output

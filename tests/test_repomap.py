"""Tests for symbol outlines, build_repo_map and the repo_map tool."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness.repomap import build_repo_map, outline_file, repo_map
from harness.tools.registry import build_registry
from harness.types import ToolCall
from harness.workspace import Workspace

PY = '''import os


class Inventory:
    """Stock."""

    def add(self, item, qty):
        pass

    async def sync(self):
        pass

    class Meta:
        def describe(self):
            pass


def helper():
    def inner():
        pass


async def fetch():
    pass
'''


def test_python_outline() -> None:
    assert outline_file("inv.py", PY) == [
        "class Inventory (L4)",
        "  def add (L7)",
        "  async def sync (L10)",
        "  class Meta (L13)",
        "    def describe (L14)",
        "def helper (L18)",
        "async def fetch (L23)",
    ]


def test_python_syntax_error_falls_back_to_regex() -> None:
    text = "class A:\n    def ok(self):\n        pass\n\ndef broken(:\n    pass\n"
    assert outline_file("x.py", text) == ["class A (L1)", "  def ok (L2)", "def broken (L5)"]


@pytest.mark.parametrize("name, text, expected", [
    ("a.ts", "export function slugify(s: string): string {\n}\nexport default class Store {\n  async load(id) {\n"
             "    if (x) {\n  }\n}\nconst add = (a, b) => a + b;\nexport interface Opts {}\n",
     ["function slugify (L1)", "class Store (L3)", "  load() (L4)", "const add (L8)", "type Opts (L9)"]),
    ("main.go", "package x\n\ntype Server struct {\n}\n\nfunc (s *Server) Start() error {\n}\n\nfunc main() {\n}\n",
     ["type Server struct (L3)", "func (s *Server) Start (L6)", "func main (L9)"]),
    ("A.java", "public class Account {\n    public void deposit(int x) {\n    }\n    private static int fee() {\n"
               "    }\n}\n", ["class Account (L1)", "  deposit() (L2)", "  fee() (L4)"]),
    ("a.kt", "data class User(val n: String)\nfun main() {\n}\nclass Repo {\n    suspend fun load(id: Int) {}\n}\n",
     ["class User (L1)", "fun main (L2)", "class Repo (L4)", "  fun load (L5)"]),
    ("lib.rs", "pub struct Stack {\n}\nimpl Stack {\n    pub fn push(&mut self) {}\n}\nimpl Display for Stack {\n}\n"
               "trait Shape {}\nasync fn run() {}\n",
     ["struct Stack (L1)", "impl Stack (L3)", "  fn push (L4)", "impl Display for Stack (L6)", "trait Shape (L8)",
      "fn run (L9)"]),
    ("a.rb", "module Shop\n  class Cart\n    def total\n    end\n    def self.build\n    end\n  end\nend\n",
     ["module Shop (L1)", "  class Cart (L2)", "  def total (L3)", "  def self.build (L5)"]),
    ("README.md", "# class Foo\n", []),
])
def test_regex_outlines(name: str, text: str, expected: list) -> None:
    assert outline_file(name, text) == expected


@pytest.fixture()
def ws(tmp_path: Path) -> Workspace:
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "inv.py").write_text(PY)
    (repo / "pkg" / "util.py").write_text("def a():\n    pass\n")
    (repo / "pkg" / "empty.py").write_text("X = 1\n")
    (repo / "notes.txt").write_text("def not_code():\n")
    return Workspace(repo, tmp_path / "scratch")


def test_build_repo_map_order_and_format(ws: Workspace) -> None:
    text = build_repo_map(ws, ["pkg/util.py", "pkg/inv.py"])
    assert text.startswith("pkg/util.py:\n  def a (L1)\npkg/inv.py:\n  class Inventory (L4)\n    def add (L7)")
    everything = build_repo_map(ws)
    assert "pkg/empty.py:" in everything and "notes.txt" not in everything


def test_build_repo_map_truncates(ws: Workspace) -> None:
    text = build_repo_map(ws, ["pkg/util.py", "pkg/inv.py", "pkg/empty.py"], max_chars=40)
    assert text.startswith("pkg/util.py:")
    assert "repo map truncated; 2 more file(s)" in text


def test_repo_map_tool(ws: Workspace) -> None:
    cfg = SimpleNamespace(context=SimpleNamespace(max_tool_output_chars=8000))
    res = repo_map(ws, cfg, path="pkg/inv.py")
    assert res.ok and res.output.splitlines()[:2] == ["pkg/inv.py:", "  class Inventory (L4)"]
    assert "(no classes or functions found)" in repo_map(ws, cfg, path="pkg/empty.py").output
    assert "pkg/util.py:" in repo_map(ws, cfg, path="pkg").output
    assert not repo_map(ws, cfg, path="nope").ok
    assert not repo_map(ws, cfg, path="../..").ok


def test_registry_now_includes_repo_map(ws: Workspace) -> None:
    cfg = SimpleNamespace(context=SimpleNamespace(max_tool_output_chars=8000),
                          safety=SimpleNamespace(command_timeout_s=30, allow_edit_existing_tests=False))
    reg = build_registry(ws, cfg)
    assert "repo_map" in reg.names()
    assert "class Inventory" in reg.dispatch(ToolCall("c", "repo_map", {"path": "pkg"})).output

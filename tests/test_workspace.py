"""Tests for Workspace: path safety, scratch mapping, edit history, diff, revert, snapshots."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from harness.workspace import Workspace


@pytest.fixture()
def ws(tmp_path: Path) -> Workspace:
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "mod.py").write_text("def f():\n    return 1\n")
    (repo / ".git").mkdir()
    (repo / ".git" / "config").write_text("[core]\n")
    return Workspace(repo, tmp_path / "scratch")


# ---------------------------------------------------------------- paths
def test_pythonpath_adds_src_only_for_src_layout(ws: Workspace) -> None:
    root = str(ws.repo_root)
    assert ws.pythonpath() == root
    (ws.repo_root / "src" / "strkit").mkdir(parents=True)
    (ws.repo_root / "src" / "strkit" / "__init__.py").write_text("")
    assert ws.pythonpath().split(os.pathsep) == [root, str(ws.repo_root / "src")]
    (ws.repo_root / "src" / "__init__.py").write_text("")  # src is itself a package: imported as `src.x`
    assert ws.pythonpath() == root


@pytest.mark.parametrize("bad", ["../outside.txt", "pkg/../../outside.txt", "/etc/passwd", ".git/config",
                                 "pkg/../.git/config", "@scratch/../repo/pkg/mod.py"])
def test_escape_attempts_rejected(ws: Workspace, bad: str) -> None:
    with pytest.raises(ValueError):
        ws.resolve(bad)


def test_symlink_escape_rejected(ws: Workspace, tmp_path: Path) -> None:
    (tmp_path / "secret.txt").write_text("s")
    (ws.repo_root / "link.txt").symlink_to(tmp_path / "secret.txt")
    with pytest.raises(ValueError):
        ws.resolve("link.txt")


def test_resolve_inside_repo(ws: Workspace) -> None:
    assert ws.resolve("pkg/mod.py") == ws.repo_root / "pkg" / "mod.py"
    assert ws.resolve("./pkg/../pkg/mod.py") == ws.repo_root / "pkg" / "mod.py"
    assert ws.resolve(str(ws.repo_root / "pkg" / "mod.py")) == ws.repo_root / "pkg" / "mod.py"
    assert ws.resolve(".") == ws.repo_root and ws.resolve("") == ws.repo_root


def test_scratch_mapping(ws: Workspace) -> None:
    target = ws.resolve("@scratch/repro.py")
    assert target == ws.scratch_dir / "repro.py"
    assert ws.is_scratch("@scratch/repro.py") and ws.is_scratch(target)
    assert not ws.is_scratch("pkg/mod.py")
    assert ws.rel(target) == "@scratch/repro.py"
    assert ws.rel("pkg/mod.py") == "pkg/mod.py"
    assert ws.resolve(str(target)) == target  # absolute scratch paths (seen in command output) are allowed


@pytest.mark.parametrize("path, expected", [
    ("tests/test_x.py", True), ("pkg/test/helpers.py", True), ("test_mod.py", True), ("pkg/mod_test.py", True),
    ("cmd/main_test.go", True), ("src/a.test.ts", True), ("src/a.test.jsx", True), ("src/a.spec.js", True),
    ("web/__tests__/a.js", True), ("spec/models/user_spec.rb", True),
    ("pkg/mod.py", False), ("pkg/testing.py", False), ("contest.py", False), ("src/latest.ts", False),
])
def test_is_test_file(path: str, expected: bool) -> None:
    assert Workspace.is_test_file(path) is expected


# ---------------------------------------------------------------- read / write / history
def test_read_rejects_binary_and_missing(ws: Workspace) -> None:
    (ws.repo_root / "blob.bin").write_bytes(b"abc\0def")
    with pytest.raises(ValueError, match="binary"):
        ws.read_text("blob.bin")
    with pytest.raises(ValueError, match="not found"):
        ws.read_text("nope.py")
    assert ws.read_text("pkg/mod.py").startswith("def f")


def test_edit_diff_revert(ws: Workspace) -> None:
    ws.write_text("pkg/mod.py", "def f():\n    return 2\n")
    assert ws.edited_files() == ["pkg/mod.py"]
    diff = ws.diff()
    assert "--- a/pkg/mod.py" in diff and "+++ b/pkg/mod.py" in diff
    assert "-    return 1" in diff and "+    return 2" in diff
    ws.revert_all()
    assert (ws.repo_root / "pkg" / "mod.py").read_text() == "def f():\n    return 1\n"
    assert ws.diff() == "" and ws.edited_files() == []


def test_original_kept_across_multiple_edits(ws: Workspace) -> None:
    ws.write_text("pkg/mod.py", "v2\n")
    ws.write_text("pkg/mod.py", "def f():\n    return 1\n")  # edited back to the original
    assert ws.edited_files() == [] and ws.diff() == ""


def test_created_file_removed_by_revert(ws: Workspace) -> None:
    assert not ws.existed_at_start("pkg/new/helper.py")
    ws.write_text("pkg/new/helper.py", "X = 1\n")
    assert not ws.existed_at_start("pkg/new/helper.py")
    assert ws.existed_at_start("pkg/mod.py")
    diff = ws.diff()
    assert "--- /dev/null" in diff and "+++ b/pkg/new/helper.py" in diff
    ws.revert_all()
    assert not (ws.repo_root / "pkg" / "new" / "helper.py").exists()


def test_crlf_preserved(ws: Workspace) -> None:
    path = ws.repo_root / "win.py"
    path.write_bytes(b"a = 1\r\nb = 2\r\n")
    ws.write_text("win.py", "a = 1\nb = 3\n")
    assert path.read_bytes() == b"a = 1\r\nb = 3\r\n"
    ws.write_text("win.py", "a = 1\r\nb = 4\r\n")  # already CRLF: no doubled \r
    assert path.read_bytes() == b"a = 1\r\nb = 4\r\n"
    ws.revert_all()
    assert path.read_bytes() == b"a = 1\r\nb = 2\r\n"


def test_scratch_writes_not_in_history_or_diff(ws: Workspace) -> None:
    ws.write_text("@scratch/repro.py", "print('x')\n")
    assert (ws.scratch_dir / "repro.py").exists()
    assert ws.diff() == "" and ws.edited_files() == []
    ws.revert_all()
    assert (ws.scratch_dir / "repro.py").exists()


def test_snapshot_restore_round_trip(ws: Workspace) -> None:
    ws.write_text("pkg/mod.py", "attempt 1\n")
    snap = ws.snapshot()
    ws.write_text("pkg/mod.py", "attempt 2\n")
    ws.write_text("pkg/extra.py", "created after snapshot\n")
    ws.restore(snap)
    assert (ws.repo_root / "pkg" / "mod.py").read_text() == "attempt 1\n"
    assert not (ws.repo_root / "pkg" / "extra.py").exists()
    assert ws.edited_files() == ["pkg/mod.py"]
    assert "+attempt 1" in ws.diff()


def test_diff_applies_with_git(ws: Workspace, tmp_path: Path) -> None:
    if not shutil.which("git"):
        pytest.skip("git not available")
    pristine = tmp_path / "pristine"
    shutil.copytree(ws.repo_root, pristine, ignore=shutil.ignore_patterns(".git"))
    (ws.repo_root / "nonl.txt").write_text("no newline")
    (pristine / "nonl.txt").write_text("no newline")
    ws.write_text("pkg/mod.py", "def f():\n    return 42\n")
    ws.write_text("nonl.txt", "still no newline")
    ws.write_text("added.py", "A = 1\n")
    (tmp_path / "patch.diff").write_text(ws.diff())
    res = subprocess.run(["git", "apply", "--check", str(tmp_path / "patch.diff")], cwd=pristine,
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


# ---------------------------------------------------------------- walking
def test_iter_source_files_skips_ignored(ws: Workspace) -> None:
    root = ws.repo_root
    for d in ("node_modules/lib", ".venv/bin", "__pycache__", "build", "pkg.egg-info"):
        (root / d).mkdir(parents=True, exist_ok=True)
        (root / d / "x.py").write_text("x = 1\n")
    (root / "big.txt").write_text("a" * 1_000_001)
    (root / "img.png").write_bytes(b"\x89PNG\0\0")
    (root / "README.md").write_text("# hi\n")
    files = [ws.rel(p) for p in ws.iter_source_files()]
    assert files == ["README.md", "pkg/mod.py"]


def test_restore_snapshot_after_revert_all(ws: Workspace) -> None:
    ws.write_text("pkg/mod.py", "attempt 1\n")
    ws.write_text("pkg/new.py", "created in attempt 1\n")
    snap = ws.snapshot()
    ws.revert_all()  # e.g. before a clean-slate rescue attempt
    assert ws.diff() == ""
    ws.restore(snap)  # the best attempt was the one before the revert
    assert (ws.repo_root / "pkg" / "mod.py").read_text() == "attempt 1\n"
    assert (ws.repo_root / "pkg" / "new.py").read_text() == "created in attempt 1\n"
    assert ws.edited_files() == ["pkg/mod.py", "pkg/new.py"] and "+attempt 1" in ws.diff()
    ws.revert_all()
    assert (ws.repo_root / "pkg" / "mod.py").read_text() == "def f():\n    return 1\n"
    assert not (ws.repo_root / "pkg" / "new.py").exists()

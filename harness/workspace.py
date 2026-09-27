"""Workspace: safe paths, @scratch/, edit history, diff, revert, snapshots."""
from __future__ import annotations

import difflib
import os
import re
from pathlib import Path
from typing import Any, Iterator

SCRATCH_PREFIX = "@scratch"
IGNORED_DIRS = frozenset({
    ".git", "node_modules", ".venv", "venv", "env", "__pycache__", "dist", "build", ".tox",
    ".mypy_cache", ".pytest_cache", ".idea", ".vscode", "site-packages", "target", ".next", "coverage",
})
MAX_SOURCE_BYTES = 1_000_000
BINARY_SNIFF_BYTES = 8192

_TEST_FILE = re.compile(
    r"(^|/)(tests?|__tests__|spec)/"
    r"|(^|/)test_[^/]*\.py$"
    r"|_test\.py$"
    r"|_test\.go$"
    r"|\.test\.(js|ts|jsx|tsx)$"
    r"|\.spec\.(js|ts)$"
)


def _is_binary(path: Path) -> bool:
    """True if the first 8 KB of the file contain a NUL byte."""
    try:
        with path.open("rb") as fh:
            return b"\0" in fh.read(BINARY_SNIFF_BYTES)
    except OSError:
        return False


def _read_raw(path: Path) -> str | None:
    """File content without newline translation, or None if the file does not exist."""
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        return fh.read()


def _write_raw(path: Path, content: str) -> None:
    """Write content exactly as given (no newline translation), creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(content)


class Workspace:
    """The target repository plus a scratch directory, with every repo edit recorded for diff/revert."""

    def __init__(self, repo_root: Path, scratch_dir: Path, cfg: Any = None) -> None:
        """repo_root is the target repo; scratch_dir holds throwaway files and must live outside it."""
        self.repo_root = Path(repo_root).resolve()
        self.scratch_dir = Path(scratch_dir).resolve()
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.write_scope = "none"  # "none" | "scratch" | "repo", set per phase by the orchestrator
        self.python_exe: str | None = None  # the target repo's interpreter (set by the orchestrator)
        self._originals: dict[str, str | None] = {}  # repo-relative path -> content before first edit

    # ------------------------------------------------------------------ paths
    def pythonpath(self) -> str:
        """PYTHONPATH for the repo's processes: the root, plus src/ for a src layout (src/<pkg>/__init__.py),
        so an uninstalled src-layout package still imports in tests and reproduction scripts."""
        paths = [str(self.repo_root)]
        src = self.repo_root / "src"
        if src.is_dir() and not (src / "__init__.py").exists() and any(src.glob("*/__init__.py")):
            paths.append(str(src))
        return os.pathsep.join(paths)

    def resolve(self, path: str | Path) -> Path:
        """Map a model-supplied path to an absolute path; raise ValueError if it escapes the workspace."""
        text = str(path).strip()
        if text == SCRATCH_PREFIX or text.startswith(SCRATCH_PREFIX + "/"):
            rest = text[len(SCRATCH_PREFIX):].lstrip("/")
            target = (self.scratch_dir / rest).resolve()
            if target != self.scratch_dir and self.scratch_dir not in target.parents:
                raise ValueError(f"Path '{text}' escapes @scratch/.")
            return target
        raw = Path(text or ".")
        target = (raw if raw.is_absolute() else self.repo_root / raw).resolve()
        if target == self.scratch_dir or self.scratch_dir in target.parents:
            return target
        if target != self.repo_root and self.repo_root not in target.parents:
            raise ValueError(f"Path '{text}' is outside the repository. Use a path relative to the repository root.")
        if ".git" in target.relative_to(self.repo_root).parts:
            raise ValueError(f"Path '{text}' is inside .git/, which is off limits.")
        return target

    def _abs(self, path: str | Path) -> Path:
        """Resolve strings through resolve(); absolute Paths are only normalised."""
        if isinstance(path, Path) and path.is_absolute():
            return path.resolve()
        return self.resolve(path)

    def is_scratch(self, path: str | Path) -> bool:
        """True if the path lies inside the scratch directory."""
        try:
            target = self._abs(path)
        except ValueError:
            return False
        return target == self.scratch_dir or self.scratch_dir in target.parents

    def rel(self, path: str | Path) -> str:
        """Display path: repo-relative POSIX path, or '@scratch/<name>' for scratch files."""
        target = self._abs(path)
        if self.is_scratch(target):
            rest = target.relative_to(self.scratch_dir).as_posix()
            return SCRATCH_PREFIX + "/" + ("" if rest == "." else rest)
        rel = target.relative_to(self.repo_root).as_posix()
        return rel

    @staticmethod
    def is_test_file(rel_path: str) -> bool:
        """True for paths that look like test code (tests/ dirs, test_*.py, *_test.go, *.spec.ts, ...)."""
        return bool(_TEST_FILE.search(str(rel_path).replace("\\", "/")))

    # ------------------------------------------------------------------ read / write
    def read_text(self, path: str | Path) -> str:
        """Read a text file (UTF-8, errors replaced). Raises ValueError for binary or missing files."""
        target = self._abs(path)
        if not target.is_file():
            raise ValueError(f"File not found: {self.rel(target)}")
        if _is_binary(target):
            raise ValueError(f"{self.rel(target)} is a binary file.")
        return target.read_text(encoding="utf-8", errors="replace")

    def write_text(self, path: str | Path, content: str) -> None:
        """Write a file, recording the original of repo files and keeping their newline style."""
        target = self._abs(path)
        current = _read_raw(target)
        if not self.is_scratch(target):
            key = self.rel(target)
            if key not in self._originals:
                self._originals[key] = current
        if current is not None and "\r\n" in current:
            content = content.replace("\r\n", "\n").replace("\n", "\r\n")
        _write_raw(target, content)

    # ------------------------------------------------------------------ history
    def existed_at_start(self, path: str | Path) -> bool:
        """True if the file existed before the harness touched it."""
        target = self._abs(path)
        key = self.rel(target)
        if key in self._originals:
            return self._originals[key] is not None
        return target.is_file()

    def edited_files(self) -> list[str]:
        """Repo-relative paths whose current content differs from their original."""
        return sorted(k for k, orig in self._originals.items() if _read_raw(self.repo_root / k) != orig)

    def revert_all(self) -> None:
        """Restore every edited repo file and delete files the harness created."""
        for key, original in self._originals.items():
            target = self.repo_root / key
            if original is None:
                if target.is_file():
                    target.unlink()
            else:
                _write_raw(target, original)
        self._originals.clear()

    def snapshot(self) -> dict[str, str | None]:
        """Current content of every touched repo file (None if it does not exist now)."""
        return {key: _read_raw(self.repo_root / key) for key in self._originals}

    def restore(self, snapshot: dict[str, str | None]) -> None:
        """Return touched files to a snapshot; files first touched after it go back to their original.

        Snapshot files the history no longer knows (e.g. after revert_all) are re-recorded first, so a
        snapshot taken before a revert can still be restored and later diffed/reverted.
        """
        for key in snapshot:
            if key not in self._originals:
                self._originals[key] = _read_raw(self.repo_root / key)
        for key, original in list(self._originals.items()):
            wanted = snapshot.get(key, original)
            target = self.repo_root / key
            if wanted is None:
                if target.is_file():
                    target.unlink()
            else:
                _write_raw(target, wanted)

    def diff(self) -> str:
        """Unified diff (a/ b/ prefixes, /dev/null for created/deleted files) of all repo edits."""
        chunks: list[str] = []
        for key in sorted(self._originals):
            before = self._originals[key]
            after = _read_raw(self.repo_root / key)
            if before == after:
                continue
            lines = difflib.unified_diff(
                (before or "").splitlines(keepends=True), (after or "").splitlines(keepends=True),
                fromfile="/dev/null" if before is None else f"a/{key}",
                tofile="/dev/null" if after is None else f"b/{key}",
            )
            for line in lines:
                chunks.append(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n")
        return "".join(chunks)

    # ------------------------------------------------------------------ walking
    def iter_source_files(self) -> Iterator[Path]:
        """Yield text files in the repo (sorted), skipping vendor/build dirs, files over 1 MB and binaries."""
        for dirpath, dirnames, filenames in os.walk(self.repo_root):
            dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS and not d.endswith(".egg-info"))
            for name in sorted(filenames):
                path = Path(dirpath) / name
                try:
                    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_SOURCE_BYTES:
                        continue
                except OSError:
                    continue
                if not _is_binary(path):
                    yield path

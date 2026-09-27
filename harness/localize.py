"""Issue term extraction, file ranking, and related-test discovery."""
from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any

CODE_EXTS = ("py", "js", "ts", "tsx", "jsx", "go", "rs", "java", "kt", "rb", "php", "cs", "cpp", "c", "h")
MAX_RANK_FILES = 5000

_PATH = re.compile(r"[\w./-]+\.(?:" + "|".join(CODE_EXTS) + r")\b")
_FRAME = re.compile(r'File "(.+?)", line (\d+)')
_BACKTICK = re.compile(r"`([^`\n]+)`")
_ERROR = re.compile(r"\b\w+(?:Error|Exception)\b")
_CAMEL = re.compile(r"\b(?:[A-Z][a-z0-9]+[A-Z]\w*|[a-z]+[A-Z]\w*)\b")
_SNAKE = re.compile(r"\b[A-Za-z0-9]+_[A-Za-z0-9_]+\b")
_DOTTED = re.compile(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b")
_QUOTED = re.compile(r"\"([^\"\n]{3,80})\"|'([^'\n]{3,80})'")
_IDENT = re.compile(r"[A-Za-z_][\w.]*")
_WORD = re.compile(r"\b[a-z][a-z0-9]{3,}\b")

STOP_WORDS = frozenset("""
a an and are as at be been but by can could did do does doing done for from get gets got had has have
how i if in into is it its just like may me might more most my no not now of on one only or our out
should so some such than that the their them then there these they this those to too up us use used
using very was way we were what when where which while who why will with would you your
also after again all any because before being both each few further here him his her hers let lot
many much must nor off once other own same she shall still thing things try want wants work works
bug bugs issue issues error errors exception problem expected actual behavior behaviour result results
return returns returned value values call calls called function functions method methods class classes
file files line lines code test tests case cases example instead however currently correctly correct
wrong incorrect please thanks thank steps reproduce version python none null true false self import
from def return print output input string number list dict object type run running make makes made
""".split())


def _add(terms: dict[str, float], order: dict[str, int], term: str, weight: float) -> None:
    """Record a term, keeping the max weight and the first-seen position."""
    term = term.strip().strip(".,:;()[]{}")
    if len(term) < 3 or term.lower() in STOP_WORDS:
        return
    if term not in order:
        order[term] = len(order)
    terms[term] = max(weight, terms.get(term, 0.0))


def extract_terms(issue_text: str) -> list[tuple[str, float]]:
    """Weighted search terms from issue text: paths/frames 5, backticks/errors 3, identifiers 2, quotes 1."""
    text = issue_text or ""
    terms: dict[str, float] = {}
    order: dict[str, int] = {}
    for m in _FRAME.finditer(text):
        _add(terms, order, m.group(1), 5)
    for m in _PATH.finditer(text):
        _add(terms, order, m.group(0), 5)
    for m in _BACKTICK.finditer(text):
        span = m.group(1).strip()
        if _IDENT.fullmatch(span):
            _add(terms, order, span, 3)
        for ident in _IDENT.findall(span):
            _add(terms, order, ident, 3)
            for part in ident.split("."):  # `Inventory.remove` also yields remove
                _add(terms, order, part, 3)
    for m in _ERROR.finditer(text):
        _add(terms, order, m.group(0), 3)
    for pattern in (_CAMEL, _SNAKE, _DOTTED):
        for m in pattern.finditer(text):
            _add(terms, order, m.group(0), 2)
    for m in _QUOTED.finditer(text):
        _add(terms, order, m.group(1) or m.group(2), 1)
    # Plain words also count, weakly, so "slugify is broken" still finds slugify.
    for m in _WORD.finditer(text):
        _add(terms, order, m.group(0), 0.5)
    return sorted(terms.items(), key=lambda kv: (-kv[1], order[kv[0]]))


def _is_code(rel: str) -> bool:
    """True for files with a recognised source-code extension."""
    return rel.rsplit(".", 1)[-1].lower() in CODE_EXTS if "." in rel else False


def _path_match(term: str, rel: str) -> bool:
    """Case-insensitive path match; path-like terms also match by suffix (absolute traceback paths)."""
    t, r = term.lower().replace("\\", "/"), rel.lower()
    if t in r:
        return True
    return "/" in t and (t.endswith("/" + r) or t.endswith(r)) and len(r) > 3


def rank_files(ws: Any, terms: list[tuple[str, float]], top_k: int = 10) -> list[dict]:
    """Score source files by term hits in their path and content; test files count half."""
    if not terms:
        return []
    scored: list[dict] = []
    for i, path in enumerate(ws.iter_source_files()):
        if i >= MAX_RANK_FILES:
            break
        rel = ws.rel(path)
        if not _is_code(rel):
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        score = 0.0
        hits: list[str] = []
        for term, weight in terms:
            s = 0.0
            if _path_match(term, rel):
                s += 3 * weight
            count = content.count(term)
            if count:
                s += weight * min(count, 5) * 0.5
            if s:
                score += s
                hits.append(term)
        if score <= 0:
            continue
        is_test = ws.is_test_file(rel)
        if is_test:
            score *= 0.5
        scored.append({"path": rel, "score": round(score, 2), "terms": hits[:8], "is_test": is_test})
    scored.sort(key=lambda d: (-d["score"], d["path"]))
    return scored[:top_k]


def _import_patterns(stem: str, package: str | None) -> list[re.Pattern]:
    """Regexes that detect an import of a module stem (Python and JS/TS styles)."""
    s = re.escape(stem)
    pats = [
        rf"^\s*import\s+([\w.]+\.)?{s}\b",
        rf"^\s*from\s+([\w.]+\.)?{s}\s+import\b",
        rf"^\s*from\s+\.*[\w.]*\s+import\s+[^\n]*\b{s}\b",
        rf"""(?:require\(|from\s+)['"][^'"]*\b{s}(?:\.\w+)?['"]""",
    ]
    if package:
        p = re.escape(package)
        pats.append(rf"^\s*from\s+([\w.]+\.)?{p}\.{s}\b")
    return [re.compile(p, re.MULTILINE) for p in pats]


def related_tests(ws: Any, source_paths: list[str], limit: int = 10) -> list[str]:
    """Test files named after, or importing, the given source modules (name matches first)."""
    stems: list[tuple[str, str | None]] = []
    for src in source_paths:
        p = PurePosixPath(src)
        stem, package = p.stem, (p.parent.name or None)
        if stem == "__init__":
            if not package:
                continue
            stem, package = package, (p.parent.parent.name or None)
        if len(stem) >= 2:
            stems.append((stem, package))
    if not stems:
        return []
    sources = set(source_paths)
    by_name: list[str] = []
    by_import: list[str] = []
    patterns = {stem: _import_patterns(stem, pkg) for stem, pkg in stems}
    for path in ws.iter_source_files():
        rel = ws.rel(path)
        if rel in sources or not ws.is_test_file(rel) or not _is_code(rel):
            continue
        name = PurePosixPath(rel).name
        if any(stem in name for stem, _ in stems):
            by_name.append(rel)
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if any(p.search(content) for pats in patterns.values() for p in pats):
            by_import.append(rel)
    return (by_name + by_import)[:limit]


_TRACE_FRAME = re.compile(r'File "(.+?)", line (\d+)(?:, in ([^\s,]+))?')


def localize_from_traceback(ws: Any, issue_text: str) -> dict | None:
    """Localization straight from a traceback that names repo source files (no LLM), or None.

    Frames are matched to repo files by the longest existing path suffix (so /home/u/proj/pkg/m.py finds
    pkg/m.py); test files are ignored. The innermost repo frame is where the failure surfaced.
    """
    frames: list[tuple[str, int, str]] = []
    for path, line, func in _TRACE_FRAME.findall(issue_text or ""):
        parts = [p for p in path.replace("\\", "/").split("/") if p]
        for i in range(len(parts)):
            rel = "/".join(parts[i:])
            if (ws.repo_root / rel).is_file() and _is_code(rel) and not ws.is_test_file(rel):
                frames.append((rel, int(line), func or ""))
                break
    if not frames:
        return None
    rel, line, func = frames[-1]
    files = list(dict.fromkeys(f for f, _, _ in reversed(frames)))[:3]
    symbols = list(dict.fromkeys(fn for _, _, fn in reversed(frames) if fn and not fn.startswith("<")))[:3]
    where = f"{rel}:{line}" + (f" in {func}" if func and not func.startswith("<") else "")
    return {"files": files, "symbols": symbols, "confidence": "medium", "finished": True, "source": "traceback",
            "root_cause": f"The traceback in the issue shows the failure surfacing at {where}."}

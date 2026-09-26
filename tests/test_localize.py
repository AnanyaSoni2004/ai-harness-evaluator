"""Tests for issue term extraction, file ranking and related-test discovery."""
from pathlib import Path

import pytest

from harness.localize import extract_terms, rank_files, related_tests
from harness.workspace import Workspace

ISSUE = '''Inventory goes negative

Calling `Inventory.remove` with more than the stock does not raise. Expected "insufficient stock".
See max_retries and parseDuration too.

Traceback (most recent call last):
  File "/home/dev/shop/toolkit/inventory.py", line 22, in remove
    self._stock[item] = current - qty
ValueError: bad
'''


def test_extract_terms_weights() -> None:
    terms = dict(extract_terms(ISSUE))
    assert terms["/home/dev/shop/toolkit/inventory.py"] == 5
    assert terms["Inventory.remove"] == 3 and terms["remove"] == 3
    assert terms["ValueError"] == 3
    assert terms["max_retries"] == 2 and terms["parseDuration"] == 2
    assert terms["insufficient stock"] == 1
    assert terms["negative"] == 0.5  # plain word, weak
    assert "the" not in terms and "does" not in terms and "is" not in terms  # stop-words / too short
    ordered = [t for t, _ in extract_terms(ISSUE)]
    assert ordered[0] == "/home/dev/shop/toolkit/inventory.py"  # highest weight first


def test_extract_terms_empty() -> None:
    assert extract_terms("") == []


@pytest.fixture()
def ws(tmp_path: Path) -> Workspace:
    repo = tmp_path / "repo"
    files = {
        "toolkit/__init__.py": "",
        "toolkit/inventory.py": "class Inventory:\n    def remove(self, item, qty):\n        current = 0\n",
        "toolkit/stock_report.py": "# mentions Inventory and remove and stock a lot\n" * 3,
        "toolkit/text.py": "def slugify(s):\n    return s\n",
        "tests/test_inventory.py": "from toolkit.inventory import Inventory\n\ndef test_remove():\n    pass\n",
        "tests/test_misc.py": "import toolkit.text\n",
        "tests/test_pkg.py": "from toolkit import inventory\n",
        "tests/test_other.py": "from toolkit.text import slugify\n",
        "tests/js/cart.test.js": "const inv = require('../../src/inventory');\n",
        "README.md": "Inventory remove stock negative Inventory remove Inventory remove\n",
    }
    for rel, content in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return Workspace(repo, tmp_path / "scratch")


def test_traceback_file_ranks_first(ws: Workspace) -> None:
    ranked = rank_files(ws, extract_terms(ISSUE))
    assert ranked[0]["path"] == "toolkit/inventory.py"
    assert "/home/dev/shop/toolkit/inventory.py" in ranked[0]["terms"]
    paths = [r["path"] for r in ranked]
    assert "README.md" not in paths  # only code files are ranked
    test_entry = next(r for r in ranked if r["path"] == "tests/test_inventory.py")
    assert test_entry["is_test"]


def test_test_files_count_half(ws: Workspace) -> None:
    ranked = {r["path"]: r for r in rank_files(ws, [("slugify", 2.0)])}
    assert ranked["toolkit/text.py"]["score"] == 1.0  # 2 (weight) * 1 (hit) * 0.5
    assert ranked["tests/test_other.py"]["score"] == 0.5  # same hit count, halved for a test file
    path_hit = {r["path"]: r for r in rank_files(ws, [("Inventory", 2.0)])}
    assert path_hit["tests/test_inventory.py"]["score"] == 3.5  # (3*2 path + 2*1*0.5 content) / 2


def test_plain_word_finds_function(ws: Workspace) -> None:
    ranked = rank_files(ws, extract_terms("slugify does not collapse separators"))
    assert ranked[0]["path"] == "toolkit/text.py"


def test_rank_top_k_and_no_terms(ws: Workspace) -> None:
    assert len(rank_files(ws, extract_terms(ISSUE), top_k=2)) == 2
    assert rank_files(ws, []) == []


def test_related_tests(ws: Workspace) -> None:
    found = related_tests(ws, ["toolkit/inventory.py"])
    assert found[0] == "tests/test_inventory.py"  # name match first
    assert set(found) == {"tests/test_inventory.py", "tests/test_pkg.py", "tests/js/cart.test.js"}
    text_tests = related_tests(ws, ["toolkit/text.py"])
    assert set(text_tests) == {"tests/test_misc.py", "tests/test_other.py"}
    assert related_tests(ws, ["toolkit/__init__.py"])  # package-level import resolves via the package name
    assert related_tests(ws, ["toolkit/inventory.py"], limit=1) == ["tests/test_inventory.py"]
    assert related_tests(ws, []) == []

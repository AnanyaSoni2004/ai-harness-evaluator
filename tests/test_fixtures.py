"""Guards for the bundled practice repo: baseline, hidden tests, and the claims made in the issues."""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from harness.testing import parse_pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SAMPLE = FIXTURES / "sample_repo"

# Reference fixes (file, old, new): used only here, to prove each hidden test is satisfiable.
REFERENCE_FIXES = {
    1: ("toolkit/text.py", 'return re.sub(r"[^a-z0-9]", "-", s)', 'return re.sub(r"[^a-z0-9]+", "-", s).strip("-")'),
    2: ("toolkit/paging.py", "start = page * per_page", "start = (page - 1) * per_page"),
    3: ("toolkit/durations.py",
        '    match = re.fullmatch(r"(\\d+)([smhd])", text)\n    if not match:\n'
        '        raise ValueError(f"invalid duration: {text!r}")\n'
        "    return int(match.group(1)) * UNITS[match.group(2)]\n",
        '    if not re.fullmatch(r"(?:\\d+[smhd])+", text):\n'
        '        raise ValueError(f"invalid duration: {text!r}")\n'
        '    return sum(int(n) * UNITS[u] for n, u in re.findall(r"(\\d+)([smhd])", text))\n'),
    4: ("toolkit/inventory.py", "        current = self._stock.get(item, 0)\n",
        '        current = self._stock.get(item, 0)\n        if qty > current:\n'
        '            raise ValueError("insufficient stock")\n'),
    5: ("toolkit/stats.py", "    return ordered[len(ordered) // 2]\n",
        "    mid = len(ordered) // 2\n    if len(ordered) % 2:\n        return ordered[mid]\n"
        "    return (ordered[mid - 1] + ordered[mid]) / 2\n"),
}


def pytest_in(repo: Path, *targets: str) -> dict:
    res = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rfE", "-p", "no:cacheprovider", *targets],
                         cwd=repo, capture_output=True, text=True, timeout=120)
    return parse_pytest(res.stdout + res.stderr, res.returncode)


@pytest.fixture()
def copy(tmp_path: Path) -> Path:
    dest = tmp_path / "sample_repo"
    shutil.copytree(SAMPLE, dest)
    return dest


def test_layout() -> None:
    assert (SAMPLE / "pytest.ini").read_text() == "[pytest]\ntestpaths = tests\n"
    assert not (SAMPLE / ".git").exists()
    assert sorted(p.name for p in (FIXTURES / "issues").glob("*.md")) == [
        "01_slugify.md", "02_paginate.md", "03_durations.md", "04_inventory.md", "05_median.md"]
    assert sorted(p.name for p in (FIXTURES / "hidden_tests").glob("*.py")) == [
        f"test_issue_0{i}.py" for i in range(1, 6)]
    for issue in (FIXTURES / "issues").glob("*.md"):
        text = issue.read_text()
        assert text.startswith("# ") and "**Expected**" in text and "**Actual**" in text
    assert "Traceback (most recent call last):" in (FIXTURES / "issues" / "03_durations.md").read_text()


def test_baseline_has_exactly_one_failure(copy: Path) -> None:
    result = pytest_in(copy)
    assert result["failing_ids"] == ["tests/test_paging.py::test_first_page"]
    assert result["failed"] == 1 and result["passed"] == 15


@pytest.mark.parametrize("n", range(1, 6))
def test_hidden_test_fails_before_and_passes_after_reference_fix(copy: Path, n: int) -> None:
    hidden = f"tests/test_issue_0{n}.py"
    shutil.copy(FIXTURES / "hidden_tests" / f"test_issue_0{n}.py", copy / hidden)
    before = pytest_in(copy, hidden)
    assert before["failed"] >= 1, f"hidden test {n} should fail on the buggy repo"

    rel, old, new = REFERENCE_FIXES[n]
    source = (copy / rel).read_text()
    assert source.count(old) == 1, f"reference fix {n} no longer matches {rel}"
    (copy / rel).write_text(source.replace(old, new))
    assert pytest_in(copy, hidden)["failed"] == 0, f"hidden test {n} must be satisfiable"

    (copy / hidden).unlink()
    visible = pytest_in(copy)
    expected = [] if n == 2 else ["tests/test_paging.py::test_first_page"]
    assert visible["failing_ids"] == expected  # the fix breaks nothing else


def test_issue_claims_match_the_buggy_code(copy: Path) -> None:
    code = ("from toolkit.text import slugify\nfrom toolkit.paging import paginate\n"
            "from toolkit.inventory import Inventory\nfrom toolkit.stats import median\n"
            "from toolkit.durations import parse_duration\n"
            "print(repr(slugify('Hello  World!')), repr(slugify('  Release notes: v2.0  ')))\n"
            "print(paginate(list(range(1, 11)), 1, 3), paginate(list(range(1, 11)), 4, 3))\n"
            "inv = Inventory(); inv.add('widget', 2); inv.remove('widget', 5); print(inv.count('widget'))\n"
            "print(median([1, 2, 3, 4]), median([10, 20]))\n"
            "try:\n    parse_duration('1h30m')\nexcept ValueError as e:\n    print(e)\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=copy, capture_output=True, text=True).stdout.splitlines()
    assert out == ["'hello--world-' '--release-notes--v2-0--'", "[4, 5, 6] []", "-3", "3 20",
                   "invalid duration: '1h30m'"]
    issue3 = (FIXTURES / "issues" / "03_durations.md").read_text()
    line = int(re.search(r'toolkit/durations\.py", line (\d+)', issue3).group(1))
    assert "raise ValueError" in (copy / "toolkit" / "durations.py").read_text().splitlines()[line - 1]

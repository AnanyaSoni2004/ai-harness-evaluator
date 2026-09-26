"""Hidden test for issue 01 (slugify)."""
from toolkit.text import slugify


def test_collapses_separators_and_strips_ends():
    assert slugify("Hello  World!") == "hello-world"
    assert slugify("  Release notes: v2.0  ") == "release-notes-v2-0"


def test_edge_already_hyphenated_input():
    assert slugify("--Already--Slugged--") == "already-slugged"
    assert slugify("hello world") == "hello-world"

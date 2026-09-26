"""Hidden test for issue 02 (paginate)."""
from toolkit.paging import paginate

ITEMS = list(range(1, 11))


def test_pages_are_one_based():
    assert paginate(ITEMS, 1, 3) == [1, 2, 3]
    assert paginate(ITEMS, 2, 3) == [4, 5, 6]


def test_edge_partial_last_page_and_beyond():
    assert paginate(ITEMS, 4, 3) == [10]
    assert paginate(ITEMS, 5, 3) == []

from toolkit.paging import paginate


def test_first_page():
    assert paginate([1, 2, 3, 4, 5], 1, 2) == [1, 2]


def test_empty_list():
    assert paginate([], 1, 10) == []


def test_page_past_the_end():
    assert paginate([1, 2, 3], 3, 2) == []

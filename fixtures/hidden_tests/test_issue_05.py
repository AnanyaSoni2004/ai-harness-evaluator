"""Hidden test for issue 05 (median)."""
from toolkit.stats import median


def test_even_length_averages_middle_values():
    assert median([1, 2, 3, 4]) == 2.5
    assert median([10, 20]) == 15.0


def test_edge_unsorted_input_and_odd_length():
    assert median([4, 1, 3, 2]) == 2.5
    assert median([5, 1, 3]) == 3

import pytest

from toolkit.stats import mean, median


def test_mean():
    assert mean([1, 2, 3, 4]) == 2.5


def test_median_odd_length():
    assert median([3, 1, 2]) == 2


def test_empty_raises():
    with pytest.raises(ValueError):
        median([])

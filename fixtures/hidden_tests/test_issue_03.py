"""Hidden test for issue 03 (parse_duration)."""
import pytest

from toolkit.durations import parse_duration


def test_combined_units():
    assert parse_duration("1h30m") == 5400
    assert parse_duration("2d3h4m5s") == 2 * 86400 + 3 * 3600 + 4 * 60 + 5
    assert parse_duration("90s") == 90


def test_edge_invalid_still_raises():
    for bad in ("1h30", "h1", "", "1x"):
        with pytest.raises(ValueError):
            parse_duration(bad)

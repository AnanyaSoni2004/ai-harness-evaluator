import pytest

from toolkit.durations import parse_duration


def test_seconds():
    assert parse_duration("45s") == 45


def test_minutes():
    assert parse_duration("15m") == 900


def test_hours_and_days():
    assert parse_duration("2h") == 7200
    assert parse_duration("1d") == 86400


def test_invalid():
    with pytest.raises(ValueError):
        parse_duration("soon")

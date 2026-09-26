import pytest

from toolkit.inventory import Inventory


def test_add_and_count():
    inv = Inventory()
    inv.add("apple", 3)
    assert inv.count("apple") == 3
    assert inv.count("pear") == 0


def test_remove_within_stock():
    inv = Inventory()
    inv.add("apple", 5)
    inv.remove("apple", 2)
    assert inv.count("apple") == 3


def test_add_rejects_non_positive():
    with pytest.raises(ValueError):
        Inventory().add("apple", 0)

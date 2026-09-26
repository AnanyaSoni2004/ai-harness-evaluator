"""Hidden test for issue 04 (Inventory.remove)."""
import pytest

from toolkit.inventory import Inventory


def test_remove_more_than_stock_raises_and_keeps_stock():
    inv = Inventory()
    inv.add("widget", 2)
    with pytest.raises(ValueError, match="insufficient stock"):
        inv.remove("widget", 5)
    assert inv.count("widget") == 2


def test_edge_remove_everything_and_unknown_item():
    inv = Inventory()
    inv.add("widget", 2)
    inv.remove("widget", 2)
    assert inv.count("widget") == 0
    with pytest.raises(ValueError, match="insufficient stock"):
        inv.remove("gadget", 1)

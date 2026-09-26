"""A minimal in-memory stock keeper."""


class Inventory:
    """Tracks how many units of each item are in stock."""

    def __init__(self):
        self._stock = {}

    def add(self, item, qty=1):
        """Add `qty` units of `item`."""
        if qty <= 0:
            raise ValueError("quantity must be positive")
        self._stock[item] = self._stock.get(item, 0) + qty

    def remove(self, item, qty=1):
        """Remove `qty` units of `item`."""
        if qty <= 0:
            raise ValueError("quantity must be positive")
        current = self._stock.get(item, 0)
        self._stock[item] = current - qty

    def count(self, item):
        """Units of `item` currently in stock."""
        return self._stock.get(item, 0)

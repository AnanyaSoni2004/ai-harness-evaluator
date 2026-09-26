"""Pagination helpers."""


def paginate(items, page, per_page):
    """Return the items on `page` (1-based), `per_page` items per page."""
    start = page * per_page
    return items[start:start + per_page]

# paginate() skips the first page of results

**Description**

Our list endpoint uses `toolkit.paging.paginate(items, page, per_page)` with 1-based page numbers,
as documented. Page 1 never shows the first items: it shows what should be on page 2, and the
last page comes back empty.

**Steps to reproduce**

```python
from toolkit.paging import paginate
items = list(range(1, 11))   # 1..10
print(paginate(items, 1, 3))
print(paginate(items, 4, 3))
```

**Expected**

```
[1, 2, 3]
[10]
```

**Actual**

```
[4, 5, 6]
[]
```

The existing test `tests/test_paging.py::test_first_page` also fails on main.

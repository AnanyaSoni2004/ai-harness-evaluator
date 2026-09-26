# Inventory.remove lets stock go negative

**Description**

`Inventory.remove` happily removes more units than we have. The warehouse dashboard then shows
negative stock, and the next `add` "fixes" the number in a misleading way.

**Steps to reproduce**

```python
from toolkit.inventory import Inventory
inv = Inventory()
inv.add("widget", 2)
inv.remove("widget", 5)
print(inv.count("widget"))
```

**Expected**

`remove` should refuse the operation and raise `ValueError("insufficient stock")`, leaving the stock
unchanged (still 2).

**Actual**

No error; `count("widget")` returns `-3`.

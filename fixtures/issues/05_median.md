# median() is wrong for lists with an even number of values

**Description**

The latency report shows medians that are consistently a bit too high. It looks like
`toolkit.stats.median` just picks one of the two middle values when the list has an even length.

**Steps to reproduce**

```python
from toolkit.stats import median
print(median([1, 2, 3, 4]))
print(median([10, 20]))
```

**Expected**

`2.5` and `15.0`: for an even number of values the median is the average of the two middle values.

**Actual**

`3` and `20`

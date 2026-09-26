# parse_duration crashes on combined units like "1h30m"

**Description**

Our nightly scheduler reads job timeouts from YAML. Single units work (`90s`, `45m`, `2h`), but as soon
as someone writes a combined value the whole scheduler fails to start.

**Steps to reproduce**

Set `timeout: 1h30m` in `jobs.yaml` and start the scheduler:

```
Traceback (most recent call last):
  File "/home/maria/scheduler/jobs/nightly.py", line 8, in <module>
    print(load_timeout({"timeout": "1h30m"}))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/maria/scheduler/jobs/nightly.py", line 5, in load_timeout
    return parse_duration(config["timeout"])
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/maria/scheduler/toolkit/durations.py", line 12, in parse_duration
    raise ValueError(f"invalid duration: {text!r}")
ValueError: invalid duration: '1h30m'
```

**Expected**

`parse_duration("1h30m")` returns `5400`, and any combination of `d`, `h`, `m` and `s` works,
e.g. `"2d3h"` or `"1m30s"`. Genuinely invalid input should still raise `ValueError`.

**Actual**

`ValueError: invalid duration: '1h30m'`

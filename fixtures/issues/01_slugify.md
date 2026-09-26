# slugify() produces double hyphens and a trailing hyphen

**Description**

We use `slugify` to build article URLs. Titles that contain punctuation or more than one space
between words produce ugly slugs with repeated hyphens and a dangling hyphen at the end, which
our CMS then rejects.

**Steps to reproduce**

```python
from toolkit.text import slugify
print(slugify("Hello  World!"))
```

**Expected**

`hello-world`: runs of separators should collapse into a single hyphen, and the slug should
never start or end with a hyphen.

**Actual**

`hello--world-`

Another example: `slugify("  Release notes: v2.0  ")` gives `--release-notes--v2-0--`.

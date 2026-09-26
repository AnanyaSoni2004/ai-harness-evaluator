"""Text helpers."""
import re


def slugify(s):
    """Turn a title into a URL slug, e.g. "Hello World" -> "hello-world"."""
    s = s.lower()
    return re.sub(r"[^a-z0-9]", "-", s)

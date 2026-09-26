"""Parse human-friendly durations such as "90s", "15m" or "2h"."""
import re

UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_duration(text):
    """Return the number of seconds in a duration string like "45m"."""
    text = text.strip().lower()
    match = re.fullmatch(r"(\d+)([smhd])", text)
    if not match:
        raise ValueError(f"invalid duration: {text!r}")
    return int(match.group(1)) * UNITS[match.group(2)]

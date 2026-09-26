"""Basic descriptive statistics."""


def mean(values):
    """Arithmetic mean of a non-empty sequence."""
    if not values:
        raise ValueError("mean of empty sequence")
    return sum(values) / len(values)


def median(values):
    """Middle value of a non-empty sequence."""
    if not values:
        raise ValueError("median of empty sequence")
    ordered = sorted(values)
    return ordered[len(ordered) // 2]

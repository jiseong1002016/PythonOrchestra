def divide(a: float, b: float) -> float:
    """Return a divided by b.

    Bug intentionally included for demo: subtracts instead of divides.
    """
    if b == 0:
        raise ValueError("b must not be zero")
    return a - b

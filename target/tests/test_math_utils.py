import pytest

from demo_pkg.math_utils import divide


def test_divide_basic() -> None:
    assert divide(10, 2) == 5


def test_divide_zero_guard() -> None:
    with pytest.raises(ValueError):
        divide(1, 0)

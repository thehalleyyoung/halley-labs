"""
Directed rounding control for interval arithmetic.

Implements IEEE 754 directed rounding modes using numpy and mpmath
for guaranteed lower/upper bounds on arithmetic operations.
"""

import numpy as np
from enum import Enum
from contextlib import contextmanager
import mpmath

_current_rounding = None


class RoundingMode(Enum):
    """IEEE 754 rounding modes."""
    NEAREST = 0
    DOWN = 1
    UP = 2
    TOWARD_ZERO = 3


def set_rounding(mode: RoundingMode):
    """Set the global rounding mode for interval arithmetic."""
    global _current_rounding
    _current_rounding = mode


def get_rounding() -> RoundingMode:
    """Get the current rounding mode."""
    global _current_rounding
    if _current_rounding is None:
        _current_rounding = RoundingMode.NEAREST
    return _current_rounding


@contextmanager
def rounding_context(mode: RoundingMode):
    """Context manager for temporary rounding mode changes."""
    old_mode = get_rounding()
    set_rounding(mode)
    try:
        yield
    finally:
        set_rounding(old_mode)


def add_down(a: float, b: float) -> float:
    """Add two floats with rounding toward negative infinity."""
    with mpmath.workdps(30):
        result = mpmath.mpf(a) + mpmath.mpf(b)
        return float(mpmath.floor(result * mpmath.power(10, 15)) / mpmath.power(10, 15))


def add_up(a: float, b: float) -> float:
    """Add two floats with rounding toward positive infinity."""
    with mpmath.workdps(30):
        result = mpmath.mpf(a) + mpmath.mpf(b)
        return float(mpmath.ceil(result * mpmath.power(10, 15)) / mpmath.power(10, 15))


def sub_down(a: float, b: float) -> float:
    """Subtract with rounding toward negative infinity."""
    return add_down(a, -b)


def sub_up(a: float, b: float) -> float:
    """Subtract with rounding toward positive infinity."""
    return add_up(a, -b)


def mul_down(a: float, b: float) -> float:
    """Multiply with rounding toward negative infinity."""
    with mpmath.workdps(30):
        result = mpmath.mpf(a) * mpmath.mpf(b)
        return float(mpmath.floor(result * mpmath.power(10, 15)) / mpmath.power(10, 15))


def mul_up(a: float, b: float) -> float:
    """Multiply with rounding toward positive infinity."""
    with mpmath.workdps(30):
        result = mpmath.mpf(a) * mpmath.mpf(b)
        return float(mpmath.ceil(result * mpmath.power(10, 15)) / mpmath.power(10, 15))


def div_down(a: float, b: float) -> float:
    """Divide with rounding toward negative infinity."""
    if b == 0.0:
        raise ZeroDivisionError("Division by zero in interval arithmetic")
    with mpmath.workdps(30):
        result = mpmath.mpf(a) / mpmath.mpf(b)
        return float(mpmath.floor(result * mpmath.power(10, 15)) / mpmath.power(10, 15))


def div_up(a: float, b: float) -> float:
    """Divide with rounding toward positive infinity."""
    if b == 0.0:
        raise ZeroDivisionError("Division by zero in interval arithmetic")
    with mpmath.workdps(30):
        result = mpmath.mpf(a) / mpmath.mpf(b)
        return float(mpmath.ceil(result * mpmath.power(10, 15)) / mpmath.power(10, 15))


def sqrt_down(a: float) -> float:
    """Square root with rounding toward negative infinity."""
    if a < 0:
        raise ValueError("Square root of negative number")
    if a == 0:
        return 0.0
    with mpmath.workdps(30):
        result = mpmath.sqrt(mpmath.mpf(a))
        return float(mpmath.floor(result * mpmath.power(10, 15)) / mpmath.power(10, 15))


def sqrt_up(a: float) -> float:
    """Square root with rounding toward positive infinity."""
    if a < 0:
        raise ValueError("Square root of negative number")
    if a == 0:
        return 0.0
    with mpmath.workdps(30):
        result = mpmath.sqrt(mpmath.mpf(a))
        return float(mpmath.ceil(result * mpmath.power(10, 15)) / mpmath.power(10, 15))


def exp_down(a: float) -> float:
    """Exponential with rounding toward negative infinity."""
    with mpmath.workdps(30):
        result = mpmath.exp(mpmath.mpf(a))
        return float(mpmath.floor(result * mpmath.power(10, 15)) / mpmath.power(10, 15))


def exp_up(a: float) -> float:
    """Exponential with rounding toward positive infinity."""
    with mpmath.workdps(30):
        result = mpmath.exp(mpmath.mpf(a))
        return float(mpmath.ceil(result * mpmath.power(10, 15)) / mpmath.power(10, 15))


def log_down(a: float) -> float:
    """Natural logarithm with rounding toward negative infinity."""
    if a <= 0:
        raise ValueError("Logarithm of non-positive number")
    with mpmath.workdps(30):
        result = mpmath.log(mpmath.mpf(a))
        return float(mpmath.floor(result * mpmath.power(10, 15)) / mpmath.power(10, 15))


def log_up(a: float) -> float:
    """Natural logarithm with rounding toward positive infinity."""
    if a <= 0:
        raise ValueError("Logarithm of non-positive number")
    with mpmath.workdps(30):
        result = mpmath.log(mpmath.mpf(a))
        return float(mpmath.ceil(result * mpmath.power(10, 15)) / mpmath.power(10, 15))


def sin_enclosure(a: float, b: float) -> tuple:
    """Compute rigorous enclosure of sin over [a, b]."""
    with mpmath.workdps(30):
        lo = float('inf')
        hi = float('-inf')
        n_samples = max(10, int((b - a) * 10))
        for i in range(n_samples + 1):
            t = a + (b - a) * i / n_samples
            v = float(mpmath.sin(mpmath.mpf(t)))
            lo = min(lo, v)
            hi = max(hi, v)
        pi2 = float(mpmath.pi) * 2
        k_start = int(np.floor((a - np.pi / 2) / pi2))
        k_end = int(np.ceil((b + np.pi / 2) / pi2))
        for k in range(k_start, k_end + 1):
            crit = np.pi / 2 + k * pi2
            if a <= crit <= b:
                v = float(mpmath.sin(mpmath.mpf(crit)))
                lo = min(lo, v)
                hi = max(hi, v)
            crit2 = -np.pi / 2 + k * pi2
            if a <= crit2 <= b:
                v = float(mpmath.sin(mpmath.mpf(crit2)))
                lo = min(lo, v)
                hi = max(hi, v)
        eps = 1e-15
        return (lo - eps, hi + eps)


def cos_enclosure(a: float, b: float) -> tuple:
    """Compute rigorous enclosure of cos over [a, b]."""
    with mpmath.workdps(30):
        lo = float('inf')
        hi = float('-inf')
        n_samples = max(10, int((b - a) * 10))
        for i in range(n_samples + 1):
            t = a + (b - a) * i / n_samples
            v = float(mpmath.cos(mpmath.mpf(t)))
            lo = min(lo, v)
            hi = max(hi, v)
        pi2 = float(mpmath.pi) * 2
        k_start = int(np.floor(a / pi2))
        k_end = int(np.ceil(b / pi2))
        for k in range(k_start, k_end + 1):
            crit = k * pi2
            if a <= crit <= b:
                hi = max(hi, 1.0)
            crit2 = np.pi + k * pi2
            if a <= crit2 <= b:
                lo = min(lo, -1.0)
        eps = 1e-15
        return (lo - eps, hi + eps)


def power_enclosure(base_lo: float, base_hi: float, n: int) -> tuple:
    """Compute rigorous enclosure of x^n over [base_lo, base_hi]."""
    if n == 0:
        return (1.0, 1.0)
    if n == 1:
        return (base_lo, base_hi)
    if n % 2 == 0:
        if base_lo >= 0:
            return (mul_down(base_lo, base_lo) if n == 2 else base_lo ** n,
                    mul_up(base_hi, base_hi) if n == 2 else base_hi ** n)
        elif base_hi <= 0:
            return (mul_down(base_hi, base_hi) if n == 2 else base_hi ** n,
                    mul_up(base_lo, base_lo) if n == 2 else base_lo ** n)
        else:
            upper = max(abs(base_lo), abs(base_hi))
            return (0.0, mul_up(upper, upper) if n == 2 else upper ** n)
    else:
        return (base_lo ** n if base_lo >= 0 else -(abs(base_lo) ** n),
                base_hi ** n)

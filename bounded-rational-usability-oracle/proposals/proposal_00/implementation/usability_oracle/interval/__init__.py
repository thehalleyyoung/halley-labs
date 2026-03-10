"""Interval arithmetic for parameter uncertainty propagation.

This package provides rigorous interval arithmetic primitives used by
the bounded-rational usability oracle to propagate parameter uncertainty
through cognitive model computations.

References
----------
Moore, R. E., Kearfott, R. B., & Cloud, M. J. (2009).
    *Introduction to Interval Analysis*. SIAM.
"""

from __future__ import annotations

from usability_oracle.interval.interval import Interval
from usability_oracle.interval.arithmetic import IntervalArithmetic

__all__ = ["Interval", "IntervalArithmetic"]

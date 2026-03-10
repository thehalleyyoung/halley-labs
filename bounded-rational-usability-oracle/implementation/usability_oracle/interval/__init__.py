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
from usability_oracle.interval.affine import (
    AffineForm,
    add as affine_add,
    multiply as affine_multiply,
    divide as affine_divide,
    power as affine_power,
    exp as affine_exp,
    log as affine_log,
    to_interval as affine_to_interval,
    from_interval as affine_from_interval,
)
from usability_oracle.interval.ieee_interval import (
    Decoration,
    IEEEInterval,
    ieee_add,
    ieee_sub,
    ieee_mul,
    ieee_div,
    sqrt_interval,
    pow_interval,
    intersection as ieee_intersection,
    hull as ieee_hull,
)
from usability_oracle.interval.dependency import (
    DependencyTracker,
    OpType,
    prune_domains,
    hull_consistency,
    box_consistency,
)
from usability_oracle.interval.contractors import (
    Contractor,
    ContractionResult,
    ContractionStatus,
    ForwardBackwardContractor,
    NewtonContractor,
    KrawczykContractor,
    BisectionStrategy,
    BisectionMethod,
    ContractorQueue,
    pave,
    PavingResult,
)

__all__ = [
    "Interval",
    "IntervalArithmetic",
    "AffineForm",
    "affine_add",
    "affine_multiply",
    "affine_divide",
    "affine_power",
    "affine_exp",
    "affine_log",
    "affine_to_interval",
    "affine_from_interval",
    "Decoration",
    "IEEEInterval",
    "ieee_add",
    "ieee_sub",
    "ieee_mul",
    "ieee_div",
    "sqrt_interval",
    "pow_interval",
    "ieee_intersection",
    "ieee_hull",
    "DependencyTracker",
    "OpType",
    "prune_domains",
    "hull_consistency",
    "box_consistency",
    "Contractor",
    "ContractionResult",
    "ContractionStatus",
    "ForwardBackwardContractor",
    "NewtonContractor",
    "KrawczykContractor",
    "BisectionStrategy",
    "BisectionMethod",
    "ContractorQueue",
    "pave",
    "PavingResult",
]

"""
usability_oracle.comparison — Paired usability comparison module.

Constructs a union MDP from two UI versions, computes bounded-rational
policies at matched rationality parameters, and tests for statistically
significant cost regressions with formal error bounds.

Re-exports
----------
- :class:`ComparisonResult`, :class:`RegressionReport`, :class:`ComparisonContext`
- :class:`BottleneckChange`, :class:`ChangeDirection`
- :class:`AlignmentResult`, :class:`StateMapping`, :class:`Partition`
- :class:`PairedComparator`
- :class:`UnionMDPBuilder`
- :class:`RegressionTester`, :class:`HypothesisResult`
- :class:`ErrorBoundComputer`
- :class:`ParameterFreeComparator`
- :class:`RegressionReporter`
"""

from __future__ import annotations

from usability_oracle.comparison.models import (
    AlignmentResult,
    BottleneckChange,
    ChangeDirection,
    ComparisonContext,
    ComparisonResult,
    Partition,
    PartitionBlock,
    RegressionReport,
    StateMapping,
)
from usability_oracle.comparison.paired import PairedComparator
from usability_oracle.comparison.union_mdp import UnionMDPBuilder
from usability_oracle.comparison.hypothesis import HypothesisResult, RegressionTester
from usability_oracle.comparison.error_bounds import ErrorBoundComputer
from usability_oracle.comparison.parameter_free import ParameterFreeComparator
from usability_oracle.comparison.reporter import RegressionReporter

__all__ = [
    # models
    "AlignmentResult",
    "StateMapping",
    "Partition",
    "PartitionBlock",
    "BottleneckChange",
    "ChangeDirection",
    "ComparisonResult",
    "RegressionReport",
    "ComparisonContext",
    # paired comparator
    "PairedComparator",
    # union MDP
    "UnionMDPBuilder",
    # hypothesis testing
    "RegressionTester",
    "HypothesisResult",
    # error bounds
    "ErrorBoundComputer",
    # parameter-free comparison
    "ParameterFreeComparator",
    # reporter
    "RegressionReporter",
]

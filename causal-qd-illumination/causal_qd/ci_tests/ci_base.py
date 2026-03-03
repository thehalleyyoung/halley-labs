"""Base classes for conditional independence tests."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import FrozenSet

from causal_qd.types import DataMatrix, NodeSet, PValue


@dataclass(frozen=True)
class CITestResult:
    """Result of a conditional independence test.

    Attributes
    ----------
    statistic:
        Test statistic value.
    p_value:
        p-value of the test.
    is_independent:
        Whether the null hypothesis of conditional independence is
        *not* rejected at the chosen significance level.
    conditioning_set:
        The set of variables conditioned on.
    """

    statistic: float
    p_value: PValue
    is_independent: bool
    conditioning_set: FrozenSet[int]


class CITest(ABC):
    """Abstract base class for conditional independence tests.

    Subclasses implement a specific statistical test for the null
    hypothesis ``X ⊥ Y | S`` (X is conditionally independent of Y
    given the conditioning set S).
    """

    @abstractmethod
    def test(
        self,
        x: int,
        y: int,
        conditioning_set: FrozenSet[int],
        data: DataMatrix,
        alpha: float = 0.05,
    ) -> CITestResult:
        """Test conditional independence of variables *x* and *y* given *conditioning_set*.

        Parameters
        ----------
        x:
            Column index of the first variable.
        y:
            Column index of the second variable.
        conditioning_set:
            Frozenset of column indices to condition on.
        data:
            Observed data matrix (N × p).
        alpha:
            Significance level for the test.

        Returns
        -------
        CITestResult
            The test result including statistic, p-value, and decision.
        """

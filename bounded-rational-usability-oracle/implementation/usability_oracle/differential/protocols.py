"""
usability_oracle.differential.protocols — Differential privacy protocols.

Structural interfaces for applying differential-privacy mechanisms to
usability data, tracking privacy budgets via composition theorems, and
aggregating usability metrics under (ε, δ)-DP guarantees.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from usability_oracle.differential.types import (
        CompositionResult,
        CompositionTheorem,
        NoiseConfig,
        PrivacyBudget,
        PrivacyGuarantee,
        PrivacyMechanismSpec,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PrivacyMechanism — apply a noise mechanism to a query
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class PrivacyMechanism(Protocol):
    """Apply a differential-privacy noise mechanism to a query result.

    Implementations calibrate and inject noise to achieve (ε, δ)-DP.
    """

    def calibrate(
        self,
        sensitivity: float,
        budget: PrivacyBudget,
    ) -> NoiseConfig:
        """Calibrate noise parameters for the given sensitivity and budget.

        Parameters
        ----------
        sensitivity : float
            Global sensitivity Δf of the query function.
        budget : PrivacyBudget
            Target privacy budget.

        Returns
        -------
        NoiseConfig
            Calibrated noise configuration.
        """
        ...

    def apply(
        self,
        value: float,
        noise_config: NoiseConfig,
        *,
        seed: Optional[int] = None,
    ) -> float:
        """Apply noise to a single scalar value.

        Parameters
        ----------
        value : float
            True query answer.
        noise_config : NoiseConfig
            Calibrated noise parameters.
        seed : Optional[int]
            RNG seed for reproducibility.

        Returns
        -------
        float
            Noised query answer.
        """
        ...

    def apply_vector(
        self,
        values: Sequence[float],
        noise_config: NoiseConfig,
        *,
        seed: Optional[int] = None,
    ) -> Sequence[float]:
        """Apply noise to a vector of values.

        Parameters
        ----------
        values : Sequence[float]
            True query answers.
        noise_config : NoiseConfig
            Calibrated noise parameters.
        seed : Optional[int]
            RNG seed.

        Returns
        -------
        Sequence[float]
            Noised query answers.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# PrivacyAccountant — track and compose privacy budgets
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class PrivacyAccountant(Protocol):
    """Track cumulative privacy spending across multiple mechanism invocations.

    Supports basic, advanced, Rényi, and moments-accountant composition.
    """

    def compose(
        self,
        budgets: Sequence[PrivacyBudget],
        *,
        theorem: CompositionTheorem,
        delta_target: Optional[float] = None,
    ) -> CompositionResult:
        """Compose multiple privacy budgets.

        Parameters
        ----------
        budgets : Sequence[PrivacyBudget]
            Individual mechanism budgets.
        theorem : CompositionTheorem
            Which composition theorem to apply.
        delta_target : Optional[float]
            Target δ for advanced composition (required for some theorems).

        Returns
        -------
        CompositionResult
            Composed privacy guarantee.
        """
        ...

    def remaining_budget(
        self,
        total_budget: PrivacyBudget,
        spent: Sequence[PrivacyBudget],
    ) -> PrivacyBudget:
        """Compute remaining privacy budget.

        Parameters
        ----------
        total_budget : PrivacyBudget
            Allocated total budget.
        spent : Sequence[PrivacyBudget]
            Budgets consumed so far.

        Returns
        -------
        PrivacyBudget
            Remaining (ε, δ) budget.
        """
        ...

    def can_afford(
        self,
        remaining: PrivacyBudget,
        cost: PrivacyBudget,
    ) -> bool:
        """Check if a mechanism invocation is within the remaining budget.

        Parameters
        ----------
        remaining : PrivacyBudget
            Remaining budget.
        cost : PrivacyBudget
            Cost of the proposed mechanism.

        Returns
        -------
        bool
            True if the mechanism can be afforded.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# PrivateAggregator — aggregate usability metrics under DP guarantees
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class PrivateAggregator(Protocol):
    """Aggregate usability metrics from multiple users under differential privacy.

    Provides DP-protected estimates of mean task time, success rate,
    error rate, and other usability metrics.
    """

    def private_mean(
        self,
        values: Sequence[float],
        budget: PrivacyBudget,
        *,
        clipping_bound: Optional[float] = None,
    ) -> tuple[float, PrivacyGuarantee]:
        """Compute differentially-private mean.

        Parameters
        ----------
        values : Sequence[float]
            Per-user values.
        budget : PrivacyBudget
            Privacy budget for this query.
        clipping_bound : Optional[float]
            Per-record clipping bound (auto-estimated if None).

        Returns
        -------
        tuple[float, PrivacyGuarantee]
            (noised mean, privacy guarantee).
        """
        ...

    def private_histogram(
        self,
        values: Sequence[float],
        bins: Sequence[float],
        budget: PrivacyBudget,
    ) -> tuple[Sequence[int], PrivacyGuarantee]:
        """Compute differentially-private histogram.

        Parameters
        ----------
        values : Sequence[float]
            Per-user values.
        bins : Sequence[float]
            Bin edges.
        budget : PrivacyBudget
            Privacy budget.

        Returns
        -------
        tuple[Sequence[int], PrivacyGuarantee]
            (noised bin counts, privacy guarantee).
        """
        ...

    def private_quantile(
        self,
        values: Sequence[float],
        quantile: float,
        budget: PrivacyBudget,
    ) -> tuple[float, PrivacyGuarantee]:
        """Compute differentially-private quantile.

        Parameters
        ----------
        values : Sequence[float]
            Per-user values.
        quantile : float
            Desired quantile in [0, 1].
        budget : PrivacyBudget
            Privacy budget.

        Returns
        -------
        tuple[float, PrivacyGuarantee]
            (noised quantile value, privacy guarantee).
        """
        ...


__all__ = [
    "PrivacyAccountant",
    "PrivacyMechanism",
    "PrivateAggregator",
]

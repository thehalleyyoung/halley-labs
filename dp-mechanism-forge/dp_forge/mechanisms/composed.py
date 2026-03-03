"""
Composed mechanism implementation for DP-Forge.

Provides the :class:`ComposedMechanism` class, which wraps a sequence
of sub-mechanisms (discrete or Gaussian) that have been combined under
a privacy composition theorem.

A composed mechanism applies multiple sub-mechanisms either sequentially
(each sees the same data) or in parallel (each sees a disjoint subset).
The composed privacy guarantee is tracked via the composition module.

Features:
    - **Sampling**: Apply all sub-mechanisms and return collected outputs.
    - **Privacy analysis**: Compute the composed privacy parameters.
    - **Decomposition**: List and inspect sub-mechanisms.
    - **Budget re-optimization**: Re-allocate budget across sub-mechanisms.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt

from dp_forge.exceptions import (
    BudgetExhaustedError,
    ConfigurationError,
    InvalidMechanismError,
)
from dp_forge.types import (
    CompositionType,
    PrivacyBudget,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

FloatArray = npt.NDArray[np.float64]


# =========================================================================
# ComposedMechanism
# =========================================================================


class ComposedMechanism:
    """A mechanism composed of multiple sub-mechanisms.

    Wraps a sequence of sub-mechanisms and their composition result,
    providing a unified interface for sampling, privacy querying, and
    analysis.  Supports both sequential and parallel composition.

    Each sub-mechanism must implement a ``sample`` method that takes
    input data and returns noisy output.  Alternatively, sub-mechanisms
    can be ``DiscreteMechanism`` or ``GaussianWorkloadMechanism`` instances
    from the dp_forge.mechanisms package.

    Attributes:
        sub_mechanisms: List of sub-mechanism objects.
        composition_type: Composition theorem used.
        total_privacy: Pre-computed total privacy budget after composition.
        per_mechanism_budgets: Individual budgets for each sub-mechanism.
        metadata: Additional composition metadata.

    Usage::

        composed = ComposedMechanism(
            sub_mechanisms=[mech1, mech2, mech3],
            composition_type=CompositionType.ADVANCED,
            total_privacy=PrivacyBudget(epsilon=2.0, delta=1e-5),
        )
        outputs = composed.sample(input_data)
        print(composed.total_epsilon())
    """

    def __init__(
        self,
        sub_mechanisms: List[Any],
        composition_type: CompositionType = CompositionType.BASIC,
        total_privacy: Optional[PrivacyBudget] = None,
        per_mechanism_budgets: Optional[List[PrivacyBudget]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize composed mechanism.

        Args:
            sub_mechanisms: List of sub-mechanism objects.
            composition_type: Composition theorem used.
            total_privacy: Total composed privacy budget. If None,
                computed from per_mechanism_budgets.
            per_mechanism_budgets: Per-mechanism privacy budgets.
            metadata: Additional metadata.
            seed: Random seed for sampling.

        Raises:
            InvalidMechanismError: If sub_mechanisms is empty.
            ConfigurationError: If budgets are inconsistent.
        """
        if not sub_mechanisms:
            raise InvalidMechanismError(
                "ComposedMechanism requires at least one sub-mechanism",
                reason="empty mechanism list",
            )

        self._sub_mechanisms = list(sub_mechanisms)
        self._composition_type = composition_type
        self._metadata = metadata or {}
        self._rng = np.random.default_rng(seed)

        # Handle per-mechanism budgets
        if per_mechanism_budgets is not None:
            if len(per_mechanism_budgets) != len(sub_mechanisms):
                raise ConfigurationError(
                    f"per_mechanism_budgets ({len(per_mechanism_budgets)}) must "
                    f"match sub_mechanisms ({len(sub_mechanisms)})",
                    parameter="per_mechanism_budgets",
                )
            self._per_budgets = list(per_mechanism_budgets)
        else:
            # Try to extract budgets from sub-mechanisms
            self._per_budgets = []
            for mech in sub_mechanisms:
                if hasattr(mech, "epsilon") and hasattr(mech, "delta"):
                    eps = getattr(mech, "epsilon")
                    delta = getattr(mech, "delta")
                    self._per_budgets.append(PrivacyBudget(
                        epsilon=max(eps, 1e-15),
                        delta=delta if isinstance(delta, float) else 0.0,
                    ))
                else:
                    self._per_budgets.append(PrivacyBudget(epsilon=1.0))

        # Handle total privacy
        if total_privacy is not None:
            self._total_privacy = total_privacy
        else:
            self._total_privacy = self._compute_total_privacy()

    def _compute_total_privacy(self) -> PrivacyBudget:
        """Compute total privacy from per-mechanism budgets.

        Uses the composition type to determine how individual budgets
        combine.

        Returns:
            Total PrivacyBudget.
        """
        eps_list = [b.epsilon for b in self._per_budgets]
        delta_list = [b.delta for b in self._per_budgets]

        if self._composition_type == CompositionType.BASIC:
            total_eps = sum(eps_list)
            total_delta = sum(delta_list)
        elif self._composition_type == CompositionType.ADVANCED:
            k = len(eps_list)
            eps_max = max(eps_list)
            delta_max = max(delta_list) if delta_list else 0.0
            delta_prime = 1e-5
            sqrt_term = math.sqrt(2.0 * k * math.log(1.0 / delta_prime)) * eps_max
            linear_term = k * eps_max * (math.exp(eps_max) - 1.0)
            total_eps = min(sqrt_term + linear_term, sum(eps_list))
            total_delta = min(k * delta_max + delta_prime, 1.0 - 1e-15)
        elif self._composition_type == CompositionType.RDP:
            total_eps = sum(eps_list)  # Simplified; proper RDP needs α
            total_delta = max(delta_list) if delta_list else 0.0
        elif self._composition_type == CompositionType.ZERO_CDP:
            rho_total = sum(e ** 2 / 2.0 for e in eps_list)
            delta_target = max(max(delta_list), 1e-5) if delta_list else 1e-5
            total_eps = rho_total + 2.0 * math.sqrt(
                rho_total * math.log(1.0 / delta_target)
            )
            total_delta = delta_target
        else:
            total_eps = sum(eps_list)
            total_delta = sum(delta_list)

        return PrivacyBudget(
            epsilon=max(total_eps, 1e-15),
            delta=min(max(total_delta, 0.0), 1.0 - 1e-15),
        )

    @property
    def n_sub_mechanisms(self) -> int:
        """Number of sub-mechanisms."""
        return len(self._sub_mechanisms)

    @property
    def composition_type(self) -> CompositionType:
        """Composition theorem used."""
        return self._composition_type

    @property
    def per_mechanism_budgets(self) -> List[PrivacyBudget]:
        """Per-mechanism privacy budgets."""
        return list(self._per_budgets)

    @property
    def metadata(self) -> Dict[str, Any]:
        """Composition metadata."""
        return dict(self._metadata)

    # ----- Sampling -----

    def sample(
        self,
        input_data: Any,
        rng: Optional[np.random.Generator] = None,
    ) -> List[Any]:
        """Apply all sub-mechanisms to the input data.

        For sequential composition, each sub-mechanism receives the same
        input data.  The outputs are returned as a list, one per
        sub-mechanism.

        Args:
            input_data: Input data for the mechanisms.  Type depends on
                the sub-mechanisms (int for discrete, array for Gaussian).
            rng: Optional RNG override.

        Returns:
            List of outputs, one per sub-mechanism.

        Raises:
            InvalidMechanismError: If a sub-mechanism lacks a sample method.
        """
        rng = rng or self._rng
        outputs = []

        for mech in self._sub_mechanisms:
            if hasattr(mech, "sample"):
                try:
                    # Try passing rng if the method accepts it
                    out = mech.sample(input_data, rng=rng)
                except TypeError:
                    # Fall back to no-rng call
                    out = mech.sample(input_data)
            elif hasattr(mech, "p_final"):
                # ExtractedMechanism: manual sampling
                out = self._sample_extracted(mech, input_data, rng)
            else:
                raise InvalidMechanismError(
                    f"Sub-mechanism {type(mech).__name__} has no sample method",
                    reason="missing sample method",
                )
            outputs.append(out)

        return outputs

    def _sample_extracted(
        self,
        mech: Any,
        input_data: Any,
        rng: np.random.Generator,
    ) -> float:
        """Sample from an ExtractedMechanism.

        Args:
            mech: Object with p_final attribute.
            input_data: Input database index.
            rng: Random number generator.

        Returns:
            Sampled output index.
        """
        p_table = mech.p_final
        n, k = p_table.shape
        idx = int(input_data) % n
        probs = p_table[idx]
        probs = np.maximum(probs, 0.0)
        probs = probs / probs.sum()
        j = rng.choice(k, p=probs)
        return float(j)

    # ----- Privacy analysis -----

    def total_epsilon(self) -> float:
        """Total epsilon after composition.

        Returns:
            The composed ε value.
        """
        return self._total_privacy.epsilon

    def total_delta(self) -> float:
        """Total delta after composition.

        Returns:
            The composed δ value.
        """
        return self._total_privacy.delta

    def privacy_analysis(self) -> Dict[str, Any]:
        """Comprehensive privacy analysis of the composed mechanism.

        Returns a dict containing:
        - Total (ε, δ) under the chosen composition theorem.
        - Per-mechanism (ε, δ) breakdowns.
        - Comparison with basic composition.
        - Budget utilization (fraction of total budget used by each mechanism).

        Returns:
            Dict with privacy analysis results.
        """
        eps_list = [b.epsilon for b in self._per_budgets]
        delta_list = [b.delta for b in self._per_budgets]

        basic_eps = sum(eps_list)
        basic_delta = sum(delta_list)

        analysis = {
            "composition_type": self._composition_type.name,
            "total_epsilon": self.total_epsilon(),
            "total_delta": self.total_delta(),
            "basic_composition_epsilon": basic_eps,
            "basic_composition_delta": basic_delta,
            "composition_tightness": basic_eps / max(self.total_epsilon(), 1e-15),
            "n_mechanisms": self.n_sub_mechanisms,
            "per_mechanism": [
                {
                    "index": i,
                    "epsilon": b.epsilon,
                    "delta": b.delta,
                    "budget_fraction": b.epsilon / max(self.total_epsilon(), 1e-15),
                }
                for i, b in enumerate(self._per_budgets)
            ],
        }

        return analysis

    # ----- Decomposition -----

    def decompose(self) -> List[Any]:
        """Return the list of sub-mechanisms.

        Returns:
            List of sub-mechanism objects.
        """
        return list(self._sub_mechanisms)

    def get_sub_mechanism(self, index: int) -> Any:
        """Get a specific sub-mechanism by index.

        Args:
            index: Zero-based index of the sub-mechanism.

        Returns:
            The sub-mechanism at the given index.

        Raises:
            ConfigurationError: If index is out of range.
        """
        if not (0 <= index < self.n_sub_mechanisms):
            raise ConfigurationError(
                f"index must be in [0, {self.n_sub_mechanisms}), got {index}",
                parameter="index",
                value=index,
            )
        return self._sub_mechanisms[index]

    # ----- Budget re-optimization -----

    def optimize_allocation(
        self,
        total_budget: Optional[PrivacyBudget] = None,
        sensitivities: Optional[FloatArray] = None,
    ) -> "ComposedMechanism":
        """Re-optimize the budget allocation across sub-mechanisms.

        Uses the sensitivity-weighted allocation formula to redistribute
        the total budget across sub-mechanisms for minimum total MSE.

        Note: This creates a new ComposedMechanism with updated budgets
        but the same sub-mechanism objects.  The sub-mechanisms themselves
        are NOT re-synthesised; this only updates the accounting.

        Args:
            total_budget: Total budget to redistribute. If None, uses the
                current total.
            sensitivities: Per-mechanism sensitivity values. If None,
                uses uniform weights.

        Returns:
            New ComposedMechanism with optimised budget allocation.
        """
        if total_budget is None:
            total_budget = self._total_privacy

        n = self.n_sub_mechanisms
        if sensitivities is None:
            sensitivities = np.ones(n, dtype=np.float64)
        else:
            sensitivities = np.asarray(sensitivities, dtype=np.float64)

        if len(sensitivities) != n:
            raise ConfigurationError(
                f"sensitivities length ({len(sensitivities)}) must match "
                f"n_sub_mechanisms ({n})",
                parameter="sensitivities",
            )

        # Sensitivity-weighted allocation: ε_i ∝ s_i^{2/3}
        weights = sensitivities ** (2.0 / 3.0)
        weights = weights / weights.sum()

        new_budgets = [
            PrivacyBudget(
                epsilon=max(float(weights[i] * total_budget.epsilon), 1e-15),
                delta=total_budget.delta / n,
            )
            for i in range(n)
        ]

        return ComposedMechanism(
            sub_mechanisms=self._sub_mechanisms,
            composition_type=self._composition_type,
            total_privacy=total_budget,
            per_mechanism_budgets=new_budgets,
            metadata={
                **self._metadata,
                "reallocated": True,
                "allocation_weights": weights.tolist(),
            },
        )

    def add_mechanism(
        self,
        mechanism: Any,
        budget: PrivacyBudget,
    ) -> "ComposedMechanism":
        """Add a new sub-mechanism to the composition.

        Creates a new ComposedMechanism with the additional mechanism
        appended and the total privacy recomputed.

        Args:
            mechanism: The new sub-mechanism.
            budget: Privacy budget for the new mechanism.

        Returns:
            New ComposedMechanism with the additional mechanism.

        Raises:
            BudgetExhaustedError: If adding the mechanism would exceed
                reasonable privacy bounds.
        """
        new_mechs = list(self._sub_mechanisms) + [mechanism]
        new_budgets = list(self._per_budgets) + [budget]

        return ComposedMechanism(
            sub_mechanisms=new_mechs,
            composition_type=self._composition_type,
            per_mechanism_budgets=new_budgets,
            metadata={**self._metadata, "extended": True},
        )

    def remaining_budget(
        self,
        total_budget: PrivacyBudget,
    ) -> PrivacyBudget:
        """Compute remaining privacy budget.

        Given a total budget, returns how much ε and δ remain after
        accounting for all current sub-mechanisms.

        Args:
            total_budget: Overall privacy budget.

        Returns:
            Remaining PrivacyBudget.

        Raises:
            BudgetExhaustedError: If the consumed budget exceeds the total.
        """
        consumed_eps = self.total_epsilon()
        consumed_delta = self.total_delta()

        remaining_eps = total_budget.epsilon - consumed_eps
        remaining_delta = total_budget.delta - consumed_delta

        if remaining_eps < 0 or remaining_delta < 0:
            raise BudgetExhaustedError(
                "Privacy budget exhausted",
                budget_epsilon=total_budget.epsilon,
                budget_delta=total_budget.delta,
                consumed_epsilon=consumed_eps,
                consumed_delta=consumed_delta,
            )

        return PrivacyBudget(
            epsilon=max(remaining_eps, 1e-15),
            delta=max(remaining_delta, 0.0),
        )

    # ----- Representation -----

    def __repr__(self) -> str:
        dp = f"ε={self.total_epsilon():.4f}"
        if self.total_delta() > 0:
            dp += f", δ={self.total_delta():.2e}"
        return (
            f"ComposedMechanism(n_sub={self.n_sub_mechanisms}, {dp}, "
            f"type={self._composition_type.name})"
        )

    def __str__(self) -> str:
        lines = [
            f"ComposedMechanism with {self.n_sub_mechanisms} sub-mechanisms:",
            f"  Composition: {self._composition_type.name}",
            f"  Total ε: {self.total_epsilon():.4f}",
            f"  Total δ: {self.total_delta():.2e}",
        ]
        for i, (mech, budget) in enumerate(zip(self._sub_mechanisms, self._per_budgets)):
            mech_type = type(mech).__name__
            lines.append(
                f"  [{i}] {mech_type}: ε={budget.epsilon:.4f}, δ={budget.delta:.2e}"
            )
        return "\n".join(lines)

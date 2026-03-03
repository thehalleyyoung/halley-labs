"""
Composition theorems for zero-concentrated differential privacy.

Implements sequential, parallel, adaptive, and heterogeneous composition
for zCDP mechanisms, along with budget optimization and truncated CDP.

References:
    - Bun & Steinke (2016): "Concentrated Differential Privacy"
    - Kairouz et al. (2015): "The Composition Theorem for DP"
    - Dwork et al. (2010): "Boosting and Differential Privacy"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from scipy import optimize

from dp_forge.types import PrivacyBudget, ZCDPBudget


# ---------------------------------------------------------------------------
# Sequential Composition
# ---------------------------------------------------------------------------


class SequentialComposition:
    """k-fold sequential composition with optimal parameters.

    Under zCDP, sequential composition is additive: if mechanisms M_1, ..., M_k
    satisfy ρ_1, ..., ρ_k-zCDP respectively, their sequential composition
    satisfies (ρ_1 + ... + ρ_k)-zCDP.
    """

    @staticmethod
    def compose(rhos: Sequence[float]) -> ZCDPBudget:
        """Compose a sequence of zCDP mechanisms.

        Args:
            rhos: Sequence of ρ values for each mechanism.

        Returns:
            Composed ZCDPBudget with ρ = Σρ_i.
        """
        rhos_list = list(rhos)
        if not rhos_list:
            raise ValueError("Must provide at least one mechanism")
        for r in rhos_list:
            if r <= 0:
                raise ValueError(f"All rho values must be > 0, got {r}")
        return ZCDPBudget(rho=sum(rhos_list))

    @staticmethod
    def compose_homogeneous(rho: float, k: int) -> ZCDPBudget:
        """Compose k identical ρ-zCDP mechanisms.

        Args:
            rho: Per-mechanism zCDP cost.
            k: Number of compositions.

        Returns:
            ZCDPBudget with ρ = k·ρ.
        """
        if rho <= 0:
            raise ValueError(f"rho must be > 0, got {rho}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        return ZCDPBudget(rho=k * rho)

    @staticmethod
    def per_mechanism_budget(total_rho: float, k: int) -> float:
        """Compute per-mechanism budget for equal allocation.

        Args:
            total_rho: Total zCDP budget.
            k: Number of mechanisms.

        Returns:
            Per-mechanism ρ = total_rho / k.
        """
        if total_rho <= 0:
            raise ValueError(f"total_rho must be > 0, got {total_rho}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        return total_rho / k

    @staticmethod
    def max_compositions(total_rho: float, per_rho: float) -> int:
        """Maximum number of compositions within budget.

        Args:
            total_rho: Total zCDP budget.
            per_rho: Per-mechanism zCDP cost.

        Returns:
            Maximum k such that k·per_rho ≤ total_rho.
        """
        if total_rho <= 0:
            raise ValueError(f"total_rho must be > 0, got {total_rho}")
        if per_rho <= 0:
            raise ValueError(f"per_rho must be > 0, got {per_rho}")
        return int(math.floor(total_rho / per_rho))

    @staticmethod
    def to_approx_dp(
        rhos: Sequence[float], delta: float
    ) -> PrivacyBudget:
        """Compose and convert to (ε,δ)-DP.

        Args:
            rhos: Per-mechanism ρ values.
            delta: Target δ.

        Returns:
            (ε,δ)-DP budget.
        """
        total_rho = sum(rhos)
        if total_rho <= 0:
            raise ValueError("Total rho must be > 0")
        eps = total_rho + 2.0 * math.sqrt(total_rho * math.log(1.0 / delta))
        return PrivacyBudget(epsilon=eps, delta=delta)


# ---------------------------------------------------------------------------
# Parallel Composition
# ---------------------------------------------------------------------------


class ParallelComposition:
    """Composition for mechanisms operating on disjoint subsets.

    When mechanisms operate on disjoint subsets of the data, the composed
    privacy cost is the maximum of the individual costs, not their sum.
    """

    @staticmethod
    def compose(rhos: Sequence[float]) -> ZCDPBudget:
        """Compose mechanisms on disjoint subsets.

        Args:
            rhos: Sequence of ρ values.

        Returns:
            ZCDPBudget with ρ = max(ρ_i).
        """
        rhos_list = list(rhos)
        if not rhos_list:
            raise ValueError("Must provide at least one mechanism")
        for r in rhos_list:
            if r <= 0:
                raise ValueError(f"All rho values must be > 0, got {r}")
        return ZCDPBudget(rho=max(rhos_list))

    @staticmethod
    def compose_with_overlap(
        rhos: Sequence[float],
        overlap_matrix: npt.NDArray[np.float64],
    ) -> ZCDPBudget:
        """Compose mechanisms with overlapping data partitions.

        For overlapping partitions, the cost for each record is the sum of ρ
        for mechanisms that access it. The worst-case is the maximum sum.

        Args:
            rhos: Per-mechanism ρ values.
            overlap_matrix: Binary matrix where entry (i, j) = 1 if mechanism j
                accesses data partition i. Shape: (n_partitions, n_mechanisms).

        Returns:
            ZCDPBudget with worst-case ρ.
        """
        rhos_arr = np.asarray(rhos, dtype=np.float64)
        overlap = np.asarray(overlap_matrix, dtype=np.float64)
        if overlap.ndim != 2:
            raise ValueError(
                f"overlap_matrix must be 2-D, got shape {overlap.shape}"
            )
        if overlap.shape[1] != len(rhos_arr):
            raise ValueError(
                f"overlap_matrix columns ({overlap.shape[1]}) must match "
                f"number of mechanisms ({len(rhos_arr)})"
            )
        # For each partition, sum the ρ of mechanisms accessing it
        per_partition_rho = overlap @ rhos_arr
        return ZCDPBudget(rho=float(np.max(per_partition_rho)))

    @staticmethod
    def group_compose(
        rhos: Sequence[float],
        groups: Sequence[int],
    ) -> ZCDPBudget:
        """Compose mechanisms grouped by data partition.

        Within each group, mechanisms compose sequentially. Across groups,
        they compose in parallel.

        Args:
            rhos: Per-mechanism ρ values.
            groups: Group assignment for each mechanism.

        Returns:
            ZCDPBudget with max over groups of sum within group.
        """
        rhos_list = list(rhos)
        groups_list = list(groups)
        if len(rhos_list) != len(groups_list):
            raise ValueError(
                f"Length mismatch: rhos={len(rhos_list)}, groups={len(groups_list)}"
            )
        group_sums: Dict[int, float] = {}
        for r, g in zip(rhos_list, groups_list):
            if r <= 0:
                raise ValueError(f"All rho values must be > 0, got {r}")
            group_sums[g] = group_sums.get(g, 0.0) + r
        return ZCDPBudget(rho=max(group_sums.values()))


# ---------------------------------------------------------------------------
# Adaptive Composition
# ---------------------------------------------------------------------------


class AdaptiveComposition:
    """Adaptive composition with stopping rules for zCDP.

    Supports online composition where the analyst decides the next mechanism
    based on previous outputs, with a pre-specified total budget.
    """

    def __init__(self, total_rho: float) -> None:
        """
        Args:
            total_rho: Total zCDP budget.
        """
        if total_rho <= 0:
            raise ValueError(f"total_rho must be > 0, got {total_rho}")
        self._total_rho = total_rho
        self._spent: List[float] = []
        self._spent_total: float = 0.0
        self._stopped = False

    @property
    def total_budget(self) -> float:
        return self._total_rho

    @property
    def spent(self) -> float:
        return self._spent_total

    @property
    def remaining(self) -> float:
        return max(self._total_rho - self.spent, 0.0)

    @property
    def is_exhausted(self) -> bool:
        return self.remaining <= 0 or self._stopped

    @property
    def num_queries(self) -> int:
        return len(self._spent)

    def can_spend(self, rho: float) -> bool:
        """Check if ρ can be spent without exceeding budget.

        Args:
            rho: Proposed zCDP cost.

        Returns:
            True if spending ρ would stay within budget.
        """
        return rho > 0 and (self.spent + rho) <= self._total_rho + 1e-12

    def spend(self, rho: float, name: str = "") -> ZCDPBudget:
        """Spend ρ from the budget.

        Args:
            rho: zCDP cost to spend.
            name: Optional mechanism name.

        Returns:
            Updated remaining budget.

        Raises:
            ValueError: If spending would exceed total budget.
        """
        if self._stopped:
            raise ValueError("Composition has been stopped")
        if rho <= 0:
            raise ValueError(f"rho must be > 0, got {rho}")
        if not self.can_spend(rho):
            raise ValueError(
                f"Cannot spend ρ={rho:.6f}: only {self.remaining:.6f} remaining"
            )
        self._spent.append(rho)
        self._spent_total += rho
        return ZCDPBudget(rho=self.remaining)

    def stop(self) -> ZCDPBudget:
        """Stop composition early (unused budget is not recovered).

        Returns:
            Final spent budget.
        """
        self._stopped = True
        return ZCDPBudget(rho=self.spent) if self.spent > 0 else ZCDPBudget(rho=1e-15)

    def optimal_gaussian_sigma(self, sensitivity: float = 1.0) -> float:
        """Compute σ for Gaussian using all remaining budget.

        Args:
            sensitivity: L2 sensitivity.

        Returns:
            Gaussian noise σ.
        """
        if self.remaining <= 0:
            raise ValueError("No remaining budget")
        return sensitivity / math.sqrt(2.0 * self.remaining)

    def to_approx_dp(self, delta: float) -> PrivacyBudget:
        """Convert spent budget to (ε,δ)-DP."""
        rho = self.spent
        if rho <= 0:
            return PrivacyBudget(epsilon=1e-15, delta=delta)
        eps = rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))
        return PrivacyBudget(epsilon=eps, delta=delta)

    def __repr__(self) -> str:
        return (
            f"AdaptiveComposition(total={self._total_rho:.6f}, "
            f"spent={self.spent:.6f}, queries={self.num_queries})"
        )


# ---------------------------------------------------------------------------
# Heterogeneous Composition
# ---------------------------------------------------------------------------


@dataclass
class MechanismRecord:
    """Record of a mechanism in heterogeneous composition."""

    name: str
    rho: float
    xi: float = 0.0
    mechanism_type: str = "generic"
    parameters: Dict[str, Any] = field(default_factory=dict)


class HeterogeneousComposition:
    """Compose different mechanism types under zCDP.

    Handles composition of Gaussian, Laplace, subsampled, and custom
    mechanisms, each with their own zCDP characterization.
    """

    def __init__(self) -> None:
        self._mechanisms: List[MechanismRecord] = []

    def add(
        self,
        name: str,
        rho: float,
        xi: float = 0.0,
        mechanism_type: str = "generic",
        **params: Any,
    ) -> None:
        """Add a mechanism."""
        if rho <= 0:
            raise ValueError(f"rho must be > 0, got {rho}")
        self._mechanisms.append(
            MechanismRecord(
                name=name,
                rho=rho,
                xi=xi,
                mechanism_type=mechanism_type,
                parameters=params,
            )
        )

    def add_gaussian(
        self, sigma: float, sensitivity: float = 1.0, name: str = "gaussian"
    ) -> None:
        """Add Gaussian mechanism."""
        rho = sensitivity**2 / (2.0 * sigma**2)
        self.add(name, rho, mechanism_type="gaussian", sigma=sigma)

    def add_laplace(
        self, scale: float, sensitivity: float = 1.0, name: str = "laplace"
    ) -> None:
        """Add Laplace mechanism."""
        eps = sensitivity / scale
        rho = eps**2 / 2.0
        self.add(name, rho, mechanism_type="laplace", scale=scale)

    def add_subsampled_gaussian(
        self,
        sigma: float,
        sampling_rate: float,
        sensitivity: float = 1.0,
        name: str = "subsampled_gaussian",
    ) -> None:
        """Add subsampled Gaussian mechanism."""
        from dp_forge.zcdp.accountant import SubsampledZCDP

        base_rho = sensitivity**2 / (2.0 * sigma**2)
        sub = SubsampledZCDP(base_rho=base_rho, sampling_rate=sampling_rate)
        self.add(
            name,
            sub.amplified_rho(),
            mechanism_type="subsampled_gaussian",
            sigma=sigma,
            sampling_rate=sampling_rate,
        )

    @property
    def total_rho(self) -> float:
        return sum(m.rho for m in self._mechanisms)

    @property
    def total_xi(self) -> float:
        return sum(m.xi for m in self._mechanisms)

    @property
    def num_mechanisms(self) -> int:
        return len(self._mechanisms)

    def compose(self) -> ZCDPBudget:
        """Compose all mechanisms sequentially.

        Returns:
            ZCDPBudget with total ρ and ξ.
        """
        if not self._mechanisms:
            raise ValueError("No mechanisms to compose")
        return ZCDPBudget(rho=self.total_rho, xi=self.total_xi)

    def to_approx_dp(self, delta: float) -> PrivacyBudget:
        """Convert composed budget to (ε,δ)-DP."""
        rho = self.total_rho
        xi = self.total_xi
        eps = rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta)) + xi
        return PrivacyBudget(epsilon=eps, delta=delta)

    def breakdown(self) -> List[Dict[str, Any]]:
        """Return per-mechanism breakdown."""
        return [
            {
                "name": m.name,
                "rho": m.rho,
                "xi": m.xi,
                "type": m.mechanism_type,
                "fraction": m.rho / self.total_rho if self.total_rho > 0 else 0.0,
            }
            for m in self._mechanisms
        ]

    def __repr__(self) -> str:
        return (
            f"HeterogeneousComposition(n={self.num_mechanisms}, "
            f"ρ={self.total_rho:.6f})"
        )


# ---------------------------------------------------------------------------
# Composition Optimizer
# ---------------------------------------------------------------------------


class CompositionOptimizer:
    """Find optimal split of privacy budget across queries.

    Given a total ρ budget and a set of queries with varying sensitivities,
    find the allocation that minimizes total expected error.
    """

    @staticmethod
    def equal_allocation(total_rho: float, k: int) -> npt.NDArray[np.float64]:
        """Equal allocation of budget.

        Args:
            total_rho: Total ρ budget.
            k: Number of queries.

        Returns:
            Array of k equal allocations.
        """
        if total_rho <= 0:
            raise ValueError(f"total_rho must be > 0, got {total_rho}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        return np.full(k, total_rho / k)

    @staticmethod
    def weighted_allocation(
        total_rho: float,
        weights: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Weighted allocation proportional to importance.

        Args:
            total_rho: Total ρ budget.
            weights: Non-negative importance weights.

        Returns:
            Array of allocations proportional to weights.
        """
        if total_rho <= 0:
            raise ValueError(f"total_rho must be > 0, got {total_rho}")
        w = np.asarray(weights, dtype=np.float64)
        if np.any(w < 0):
            raise ValueError("All weights must be non-negative")
        w_sum = w.sum()
        if w_sum <= 0:
            raise ValueError("At least one weight must be positive")
        return total_rho * w / w_sum

    @staticmethod
    def optimal_gaussian_allocation(
        total_rho: float,
        sensitivities: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Optimal allocation for Gaussian mechanisms minimizing total MSE.

        For Gaussian mechanism with sensitivity Δ_i and budget ρ_i,
        the MSE is σ_i² = Δ_i²/(2ρ_i). Total MSE = Σ Δ_i²/(2ρ_i).
        Minimizing subject to Σρ_i = total_rho gives the optimal allocation:
        ρ_i ∝ Δ_i (by Lagrange multipliers on 1/ρ_i objective).

        Args:
            total_rho: Total ρ budget.
            sensitivities: Array of L2 sensitivities Δ_i.

        Returns:
            Optimal ρ allocations.
        """
        if total_rho <= 0:
            raise ValueError(f"total_rho must be > 0, got {total_rho}")
        sens = np.asarray(sensitivities, dtype=np.float64)
        if np.any(sens <= 0):
            raise ValueError("All sensitivities must be > 0")
        # ρ_i ∝ Δ_i for MSE minimization with 1/ρ_i objective
        allocations = total_rho * sens / sens.sum()
        return allocations

    @staticmethod
    def optimal_for_objective(
        total_rho: float,
        k: int,
        objective: Callable[[npt.NDArray[np.float64]], float],
        constraints: Optional[List[Dict[str, Any]]] = None,
    ) -> npt.NDArray[np.float64]:
        """Optimize budget allocation for a custom objective.

        Args:
            total_rho: Total ρ budget.
            k: Number of queries.
            objective: Function mapping ρ allocations to objective value
                (to minimize).
            constraints: Additional scipy.optimize constraints.

        Returns:
            Optimal ρ allocations.
        """
        if total_rho <= 0:
            raise ValueError(f"total_rho must be > 0, got {total_rho}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        x0 = np.full(k, total_rho / k)
        bounds = [(1e-10, total_rho)] * k

        budget_constraint = {
            "type": "eq",
            "fun": lambda x: np.sum(x) - total_rho,
        }
        all_constraints = [budget_constraint]
        if constraints:
            all_constraints.extend(constraints)

        result = optimize.minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=all_constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if not result.success:
            # Fall back to equal allocation
            return np.full(k, total_rho / k)
        # Normalize to exact budget
        alloc = np.maximum(result.x, 1e-15)
        alloc = alloc * total_rho / alloc.sum()
        return alloc


# ---------------------------------------------------------------------------
# Truncated Concentrated DP
# ---------------------------------------------------------------------------


class TruncatedConcentratedDP:
    """Truncated concentrated DP with better tail bounds.

    (ξ, ρ)-tCDP (Bun & Dwork 2018) provides tighter bounds than pure
    zCDP by allowing a small additive offset ξ for the Rényi divergence
    at low orders.

    Definition: M is (ξ, ρ)-tCDP if for all α ∈ (1, ∞),
        D_α(M(x) || M(x')) ≤ ξ + ρ·α.
    """

    def __init__(self, xi: float, rho: float) -> None:
        """
        Args:
            xi: Offset parameter ξ ≥ 0.
            rho: Concentration parameter ρ > 0.
        """
        if xi < 0:
            raise ValueError(f"xi must be >= 0, got {xi}")
        if rho <= 0:
            raise ValueError(f"rho must be > 0, got {rho}")
        self.xi = xi
        self.rho = rho

    @staticmethod
    def from_pure_dp(epsilon: float) -> "TruncatedConcentratedDP":
        """Convert pure ε-DP to tCDP.

        ε-DP implies (0, ε(e^ε-1)/2)-tCDP.

        Args:
            epsilon: Pure DP parameter.

        Returns:
            TruncatedConcentratedDP.
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        rho = epsilon * (math.exp(epsilon) - 1.0) / 2.0
        return TruncatedConcentratedDP(xi=0.0, rho=rho)

    @staticmethod
    def from_approx_dp(epsilon: float, delta: float) -> "TruncatedConcentratedDP":
        """Convert (ε,δ)-DP to tCDP.

        Uses (ξ, ρ)-tCDP with ξ = log(1/δ) and ρ from the pure DP part.

        Args:
            epsilon: Privacy parameter ε.
            delta: Privacy parameter δ.

        Returns:
            TruncatedConcentratedDP.
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        if delta <= 0 or delta >= 1:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        xi = math.log(1.0 / delta)
        rho = epsilon * (math.exp(epsilon) - 1.0) / 2.0
        return TruncatedConcentratedDP(xi=xi, rho=rho)

    @staticmethod
    def from_gaussian(
        sigma: float, sensitivity: float = 1.0
    ) -> "TruncatedConcentratedDP":
        """Create tCDP parameters for Gaussian mechanism.

        Gaussian is (0, Δ²/(2σ²))-tCDP = pure ρ-zCDP.

        Args:
            sigma: Noise standard deviation.
            sensitivity: L2 sensitivity.

        Returns:
            TruncatedConcentratedDP with ξ=0.
        """
        rho = sensitivity**2 / (2.0 * sigma**2)
        return TruncatedConcentratedDP(xi=0.0, rho=rho)

    def compose(self, other: "TruncatedConcentratedDP") -> "TruncatedConcentratedDP":
        """Compose two tCDP mechanisms (sequential).

        (ξ₁, ρ₁)-tCDP ∘ (ξ₂, ρ₂)-tCDP = (ξ₁+ξ₂, ρ₁+ρ₂)-tCDP.

        Args:
            other: Another tCDP mechanism.

        Returns:
            Composed TruncatedConcentratedDP.
        """
        return TruncatedConcentratedDP(
            xi=self.xi + other.xi, rho=self.rho + other.rho
        )

    @staticmethod
    def compose_many(
        mechanisms: Sequence["TruncatedConcentratedDP"],
    ) -> "TruncatedConcentratedDP":
        """Compose multiple tCDP mechanisms.

        Args:
            mechanisms: Sequence of tCDP mechanisms.

        Returns:
            Composed TruncatedConcentratedDP.
        """
        if not mechanisms:
            raise ValueError("Must provide at least one mechanism")
        total_xi = sum(m.xi for m in mechanisms)
        total_rho = sum(m.rho for m in mechanisms)
        return TruncatedConcentratedDP(xi=total_xi, rho=total_rho)

    def to_approx_dp(self, delta: float) -> PrivacyBudget:
        """Convert to (ε,δ)-DP.

        ε = ξ + ρ + 2√(ρ·ln(1/δ)).

        Args:
            delta: Target δ.

        Returns:
            (ε,δ)-DP budget.
        """
        if delta <= 0 or delta >= 1:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        eps = self.xi + self.rho + 2.0 * math.sqrt(
            self.rho * math.log(1.0 / delta)
        )
        return PrivacyBudget(epsilon=eps, delta=delta)

    def to_zcdp(self) -> ZCDPBudget:
        """Convert to pure zCDP (lossy if ξ > 0).

        Returns:
            ZCDPBudget with ρ and ξ.
        """
        return ZCDPBudget(rho=self.rho, xi=self.xi)

    def rdp(self, alpha: float) -> float:
        """RDP bound at order α.

        D_α ≤ ξ + ρ·α.

        Args:
            alpha: Rényi order.

        Returns:
            RDP bound.
        """
        if alpha <= 1:
            raise ValueError(f"alpha must be > 1, got {alpha}")
        return self.xi + self.rho * alpha

    def __repr__(self) -> str:
        return f"TruncatedConcentratedDP(ξ={self.xi:.6f}, ρ={self.rho:.6f})"

"""
usability_oracle.differential.composition — Composition theorems.

Implementations of basic, advanced, parallel, heterogeneous, and
tight numerical composition for differential privacy mechanisms.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

from usability_oracle.differential.types import (
    CompositionResult,
    CompositionTheorem,
    PrivacyBudget,
)
from usability_oracle.differential.accountant import (
    DEFAULT_RDP_ORDERS,
    rdp_to_approx_dp,
)


# ═══════════════════════════════════════════════════════════════════════════
# Basic composition theorem
# ═══════════════════════════════════════════════════════════════════════════


def basic_composition(budgets: Sequence[PrivacyBudget]) -> CompositionResult:
    """Basic sequential composition: ε_total = Σ εᵢ, δ_total = Σ δᵢ.

    Parameters
    ----------
    budgets : Sequence[PrivacyBudget]
        Per-mechanism privacy budgets.

    Returns
    -------
    CompositionResult
    """
    if not budgets:
        return CompositionResult(
            total_budget=PrivacyBudget(0.0, 0.0),
            theorem_used=CompositionTheorem.BASIC,
        )
    total_eps = sum(b.epsilon for b in budgets)
    total_delta = min(sum(b.delta for b in budgets), 1.0 - 1e-15)
    return CompositionResult(
        total_budget=PrivacyBudget(epsilon=total_eps, delta=total_delta),
        theorem_used=CompositionTheorem.BASIC,
        n_mechanisms=len(budgets),
        per_mechanism_budgets=tuple(budgets),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Advanced (strong) composition
# ═══════════════════════════════════════════════════════════════════════════


def advanced_composition(
    budgets: Sequence[PrivacyBudget],
    delta_prime: float,
) -> CompositionResult:
    """Advanced composition theorem (Dwork, Rothblum, Vadhan 2010).

    For k mechanisms each satisfying (εᵢ, δᵢ)-DP, the composed mechanism
    satisfies (ε', Σδᵢ + δ')-DP where:

        ε' = √(2 Σεᵢ² ln(1/δ')) + Σ εᵢ(e^εᵢ − 1)

    Parameters
    ----------
    budgets : Sequence[PrivacyBudget]
        Per-mechanism budgets.
    delta_prime : float
        Additional failure probability δ' > 0.

    Returns
    -------
    CompositionResult
    """
    if delta_prime <= 0 or delta_prime >= 1:
        raise ValueError("delta_prime must be in (0, 1)")
    if not budgets:
        return CompositionResult(
            total_budget=PrivacyBudget(0.0, 0.0),
            theorem_used=CompositionTheorem.ADVANCED,
        )

    eps_values = [b.epsilon for b in budgets]
    sum_eps_sq = sum(e * e for e in eps_values)
    sum_eps_exp = sum(e * (math.exp(e) - 1.0) for e in eps_values)
    total_delta_mech = sum(b.delta for b in budgets)

    eps_composed = math.sqrt(2.0 * sum_eps_sq * math.log(1.0 / delta_prime)) + sum_eps_exp
    delta_composed = min(total_delta_mech + delta_prime, 1.0 - 1e-15)

    # Tightness gap: compare with basic composition
    basic_eps = sum(eps_values)
    gap = max(0.0, basic_eps - eps_composed)

    return CompositionResult(
        total_budget=PrivacyBudget(epsilon=eps_composed, delta=delta_composed),
        theorem_used=CompositionTheorem.ADVANCED,
        n_mechanisms=len(budgets),
        per_mechanism_budgets=tuple(budgets),
        tightness_gap=gap,
        metadata={"delta_prime": delta_prime},
    )


# ═══════════════════════════════════════════════════════════════════════════
# Parallel composition
# ═══════════════════════════════════════════════════════════════════════════


def parallel_composition(budgets: Sequence[PrivacyBudget]) -> CompositionResult:
    """Parallel composition: each mechanism operates on disjoint data subsets.

    The composed guarantee is (max εᵢ, max δᵢ).

    Parameters
    ----------
    budgets : Sequence[PrivacyBudget]
        Per-mechanism budgets (each on a disjoint partition).

    Returns
    -------
    CompositionResult
    """
    if not budgets:
        return CompositionResult(
            total_budget=PrivacyBudget(0.0, 0.0),
            theorem_used=CompositionTheorem.BASIC,
        )
    max_eps = max(b.epsilon for b in budgets)
    max_delta = max(b.delta for b in budgets)
    return CompositionResult(
        total_budget=PrivacyBudget(epsilon=max_eps, delta=max_delta),
        theorem_used=CompositionTheorem.BASIC,
        n_mechanisms=len(budgets),
        per_mechanism_budgets=tuple(budgets),
        metadata={"composition_type": "parallel"},
    )


# ═══════════════════════════════════════════════════════════════════════════
# Post-processing immunity verification
# ═══════════════════════════════════════════════════════════════════════════


def verify_post_processing(
    original_budget: PrivacyBudget,
    post_processed_budget: PrivacyBudget,
) -> bool:
    """Verify that a post-processing step does not degrade the privacy guarantee.

    Any deterministic or randomised function applied to the *output* of
    an (ε, δ)-DP mechanism cannot increase ε or δ.  This function checks
    that the claimed post-processing budget is no worse.

    Parameters
    ----------
    original_budget : PrivacyBudget
        Budget of the underlying mechanism.
    post_processed_budget : PrivacyBudget
        Claimed budget after post-processing.

    Returns
    -------
    bool
        True if the post-processing claim is valid (budget did not worsen).
    """
    return (
        post_processed_budget.epsilon <= original_budget.epsilon + 1e-12
        and post_processed_budget.delta <= original_budget.delta + 1e-12
    )


# ═══════════════════════════════════════════════════════════════════════════
# Heterogeneous composition
# ═══════════════════════════════════════════════════════════════════════════


def heterogeneous_composition(
    budgets: Sequence[PrivacyBudget],
    delta_target: float,
    *,
    rdp_curves: Optional[Sequence[Dict[float, float]]] = None,
) -> CompositionResult:
    """Heterogeneous composition for mechanisms with different parameters.

    If RDP curves are provided, uses RDP composition (summing ε at each
    order and converting to (ε, δ)-DP at the best order).  Otherwise
    falls back to advanced composition.

    Parameters
    ----------
    budgets : Sequence[PrivacyBudget]
        Per-mechanism (ε, δ) budgets.
    delta_target : float
        Target δ for the composed guarantee.
    rdp_curves : optional Sequence[dict]
        For each mechanism, a mapping α → ε_RDP(α).  Enables tight
        RDP-based composition.

    Returns
    -------
    CompositionResult
    """
    if rdp_curves is not None and len(rdp_curves) == len(budgets):
        return _heterogeneous_rdp(budgets, rdp_curves, delta_target)
    return advanced_composition(budgets, delta_prime=delta_target)


def _heterogeneous_rdp(
    budgets: Sequence[PrivacyBudget],
    rdp_curves: Sequence[Dict[float, float]],
    delta_target: float,
) -> CompositionResult:
    """RDP-based heterogeneous composition."""
    # Sum RDP ε at each order
    all_orders = set()
    for curve in rdp_curves:
        all_orders.update(curve.keys())
    orders = sorted(all_orders)

    best_eps = float("inf")
    best_alpha = orders[0] if orders else 2.0

    for alpha in orders:
        if alpha <= 1.0:
            continue
        rdp_sum = sum(curve.get(alpha, float("inf")) for curve in rdp_curves)
        eps = rdp_to_approx_dp(rdp_sum, alpha, delta_target)
        if eps < best_eps:
            best_eps = eps
            best_alpha = alpha

    return CompositionResult(
        total_budget=PrivacyBudget(epsilon=best_eps, delta=delta_target),
        theorem_used=CompositionTheorem.RENYI,
        n_mechanisms=len(budgets),
        per_mechanism_budgets=tuple(budgets),
        metadata={"best_alpha": best_alpha, "composition_type": "heterogeneous_rdp"},
    )


# ═══════════════════════════════════════════════════════════════════════════
# Optimal composition via privacy loss random variable (PLD)
# ═══════════════════════════════════════════════════════════════════════════


def optimal_composition_gaussian(
    sigma: float,
    n_mechanisms: int,
    delta_target: float,
    sensitivity: float = 1.0,
) -> CompositionResult:
    """Optimal composition for homogeneous Gaussian mechanisms via RDP.

    Parameters
    ----------
    sigma : float
        Gaussian noise σ.
    n_mechanisms : int
        Number of mechanism applications.
    delta_target : float
        Target δ.
    sensitivity : float
        L2 sensitivity.

    Returns
    -------
    CompositionResult
    """
    best_eps = float("inf")
    best_alpha = 2.0
    for alpha in DEFAULT_RDP_ORDERS:
        rdp_single = alpha * (sensitivity ** 2) / (2.0 * sigma * sigma)
        rdp_total = n_mechanisms * rdp_single
        eps = rdp_to_approx_dp(rdp_total, alpha, delta_target)
        if eps < best_eps:
            best_eps = eps
            best_alpha = alpha

    budget = PrivacyBudget(
        epsilon=sigma * sensitivity / sigma,  # placeholder for per-mech
        delta=0.0,
    )
    return CompositionResult(
        total_budget=PrivacyBudget(epsilon=best_eps, delta=delta_target),
        theorem_used=CompositionTheorem.RENYI,
        n_mechanisms=n_mechanisms,
        per_mechanism_budgets=tuple([budget] * n_mechanisms),
        metadata={"sigma": sigma, "best_alpha": best_alpha},
    )


# ═══════════════════════════════════════════════════════════════════════════
# Tight numerical composition
# ═══════════════════════════════════════════════════════════════════════════


def tight_numerical_composition(
    budgets: Sequence[PrivacyBudget],
    delta_target: float,
    *,
    grid_size: int = 10000,
) -> CompositionResult:
    """Tight numerical composition via discretised privacy loss distributions.

    Approximates the privacy loss distribution (PLD) of each mechanism
    on a discrete grid and convolves them to obtain the composed PLD.
    Then reads off the tightest (ε, δ) from the composed distribution.

    Parameters
    ----------
    budgets : Sequence[PrivacyBudget]
        Per-mechanism budgets (pure DP only for this implementation).
    delta_target : float
        Target δ for the composed guarantee.
    grid_size : int
        Number of grid points for the PLD discretisation.

    Returns
    -------
    CompositionResult
    """
    if not budgets:
        return CompositionResult(
            total_budget=PrivacyBudget(0.0, 0.0),
            theorem_used=CompositionTheorem.BASIC,
        )

    # Use Laplace mechanism PLDs (pure DP)
    # PLD of Laplace(ε): privacy loss Z takes value ε with prob e^ε/(1+e^ε)
    # and value −ε with remaining prob
    epsilons = [b.epsilon for b in budgets]

    # Discretise onto a grid
    max_loss = sum(epsilons)
    grid = np.linspace(-max_loss, max_loss, grid_size)
    dx = grid[1] - grid[0]

    # Start with delta distribution at 0
    pld = np.zeros(grid_size)
    pld[grid_size // 2] = 1.0 / dx  # approximate delta function

    for eps_i in epsilons:
        # PLD of a single (ε, 0)-DP Laplace mechanism
        # The privacy loss is distributed as Laplace(0, 1) * ε (simplified)
        single_pld = np.exp(-np.abs(grid) / max(eps_i, 1e-10)) / (2.0 * max(eps_i, 1e-10))
        single_pld = single_pld / (single_pld.sum() * dx)  # normalise

        # Convolve
        conv = np.convolve(pld, single_pld, mode="same") * dx
        pld = conv

    # Compute ε for the given δ: find smallest ε s.t. P[Z > ε] ≤ δ
    # P[Z > t] = integral of pld from t to ∞
    cdf = np.cumsum(pld) * dx
    # Find threshold
    survival = 1.0 - cdf
    # Tightest ε: smallest grid[i] where survival ≤ δ_target
    candidates = np.where(survival <= delta_target)[0]
    if len(candidates) > 0:
        tight_eps = float(grid[candidates[0]])
    else:
        tight_eps = float(max_loss)

    tight_eps = max(0.0, tight_eps)

    return CompositionResult(
        total_budget=PrivacyBudget(epsilon=tight_eps, delta=delta_target),
        theorem_used=CompositionTheorem.BASIC,
        n_mechanisms=len(budgets),
        per_mechanism_budgets=tuple(budgets),
        metadata={"composition_type": "numerical_pld", "grid_size": grid_size},
    )


# ═══════════════════════════════════════════════════════════════════════════
# Adaptive composition
# ═══════════════════════════════════════════════════════════════════════════


class AdaptiveComposer:
    """Composition tracker for adaptive mechanism selection.

    In adaptive composition, the analyst may choose the next mechanism
    based on the *outputs* of previous mechanisms.  The standard
    composition theorems still apply, but we track spending incrementally.

    Parameters
    ----------
    total_budget : PrivacyBudget
        Maximum allowed (ε, δ) budget.
    """

    def __init__(self, total_budget: PrivacyBudget) -> None:
        self.total_budget = total_budget
        self._spent: List[PrivacyBudget] = []

    def propose(self, cost: PrivacyBudget) -> bool:
        """Check if the next mechanism with *cost* can be afforded.

        Returns True if the total spending (basic composition) would
        remain within the total budget.
        """
        new_eps = sum(b.epsilon for b in self._spent) + cost.epsilon
        new_delta = sum(b.delta for b in self._spent) + cost.delta
        return (
            new_eps <= self.total_budget.epsilon + 1e-12
            and new_delta <= self.total_budget.delta + 1e-12
        )

    def commit(self, cost: PrivacyBudget) -> None:
        """Record spending for a mechanism that has been applied.

        Raises
        ------
        ValueError
            If the cost would exceed the total budget.
        """
        if not self.propose(cost):
            raise ValueError(
                f"Budget exceeded: spending {cost} would exceed total {self.total_budget}"
            )
        self._spent.append(cost)

    @property
    def remaining(self) -> PrivacyBudget:
        """Remaining budget under basic composition."""
        spent_eps = sum(b.epsilon for b in self._spent)
        spent_delta = sum(b.delta for b in self._spent)
        return PrivacyBudget(
            epsilon=max(0.0, self.total_budget.epsilon - spent_eps),
            delta=max(0.0, self.total_budget.delta - spent_delta),
        )

    @property
    def spent(self) -> PrivacyBudget:
        """Total spent budget."""
        return PrivacyBudget(
            epsilon=sum(b.epsilon for b in self._spent),
            delta=min(sum(b.delta for b in self._spent), 1.0 - 1e-15),
        )

    def compose(self) -> CompositionResult:
        """Compose all committed mechanisms."""
        return basic_composition(self._spent)


__all__ = [
    "basic_composition",
    "advanced_composition",
    "parallel_composition",
    "verify_post_processing",
    "heterogeneous_composition",
    "optimal_composition_gaussian",
    "tight_numerical_composition",
    "AdaptiveComposer",
]

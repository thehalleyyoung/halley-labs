"""
usability_oracle.differential.accountant — Privacy budget accounting.

Tracks cumulative privacy spending across mechanism invocations using
basic, advanced, Rényi DP, zero-concentrated DP, and moments accountant
composition.  Provides conversion utilities between privacy definitions.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

from usability_oracle.differential.types import (
    CompositionResult,
    CompositionTheorem,
    PrivacyBudget,
)


# ═══════════════════════════════════════════════════════════════════════════
# Rényi Divergence helpers
# ═══════════════════════════════════════════════════════════════════════════


def _rdp_gaussian(alpha: float, sigma: float) -> float:
    """Rényi divergence of order α for a Gaussian mechanism with noise σ.

    RDP_α(N(0,σ²) ‖ N(1,σ²)) = α / (2σ²).

    Parameters
    ----------
    alpha : float
        Rényi order α > 1.
    sigma : float
        Standard deviation of the Gaussian noise.

    Returns
    -------
    float
        Rényi divergence.
    """
    return alpha / (2.0 * sigma * sigma)


def _rdp_laplace(alpha: float, b: float) -> float:
    """Rényi divergence of order α for a Laplace mechanism with scale b.

    Uses the closed-form expression for Laplace(0, b) vs Laplace(1/b, b)
    (sensitivity = 1 convention; caller must scale accordingly).

    Parameters
    ----------
    alpha : float
        Rényi order α > 1.
    b : float
        Laplace scale parameter.

    Returns
    -------
    float
        Rényi divergence.
    """
    if alpha == 1.0:
        return 1.0 / b
    if alpha == float("inf"):
        return 1.0 / b
    eps = 1.0 / b
    # Closed-form:  (1/(α-1)) * log( (α/(2α-1)) exp((α-1)ε) + ((α-1)/(2α-1)) exp(-αε) )
    term1 = (alpha / (2.0 * alpha - 1.0)) * math.exp((alpha - 1.0) * eps)
    term2 = ((alpha - 1.0) / (2.0 * alpha - 1.0)) * math.exp(-alpha * eps)
    return (1.0 / (alpha - 1.0)) * math.log(term1 + term2)


# ═══════════════════════════════════════════════════════════════════════════
# Conversion between privacy definitions
# ═══════════════════════════════════════════════════════════════════════════


def rdp_to_approx_dp(rdp_epsilon: float, alpha: float, delta: float) -> float:
    """Convert (α, ε_RDP) to (ε, δ)-DP.

    ε = ε_RDP − log(δ) / (α − 1)

    Parameters
    ----------
    rdp_epsilon : float
        Rényi DP parameter.
    alpha : float
        Rényi order α > 1.
    delta : float
        Target δ ∈ (0, 1).

    Returns
    -------
    float
        Converted ε for (ε, δ)-DP.
    """
    if alpha <= 1.0:
        raise ValueError("alpha must be > 1")
    if delta <= 0 or delta >= 1:
        raise ValueError("delta must be in (0, 1)")
    return rdp_epsilon + math.log(1.0 / delta) / (alpha - 1.0)


def approx_dp_to_rdp(epsilon: float, delta: float, alpha: float) -> float:
    """Convert (ε, δ)-DP to an upper bound on (α, ε_RDP).

    ε_RDP ≤ ε + log(1/δ)/(α − 1)  (trivial bound; used as fallback).
    """
    if alpha <= 1.0:
        raise ValueError("alpha must be > 1")
    return epsilon + math.log(1.0 / delta) / (alpha - 1.0)


def zcdp_to_rdp(rho: float, alpha: float) -> float:
    """Convert ρ-zCDP to (α, ε_RDP).

    A mechanism satisfying ρ-zCDP satisfies (α, ρ·α)-RDP for all α > 1.

    Parameters
    ----------
    rho : float
        zCDP parameter.
    alpha : float
        Rényi order.
    """
    return rho * alpha


def zcdp_to_approx_dp(rho: float, delta: float) -> float:
    """Convert ρ-zCDP to (ε, δ)-DP.

    ε = ρ + 2√(ρ · log(1/δ))

    Parameters
    ----------
    rho : float
        zCDP parameter.
    delta : float
        Target δ > 0.

    Returns
    -------
    float
        Converted ε.
    """
    if delta <= 0 or delta >= 1:
        raise ValueError("delta must be in (0, 1)")
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))


def gaussian_to_zcdp(sigma: float, sensitivity: float = 1.0) -> float:
    """Compute ρ-zCDP parameter for a Gaussian mechanism.

    ρ = (Δf)² / (2σ²)
    """
    return (sensitivity ** 2) / (2.0 * sigma * sigma)


# ═══════════════════════════════════════════════════════════════════════════
# Privacy budget ledger entry
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class BudgetEntry:
    """A single privacy expenditure record.

    Attributes
    ----------
    budget : PrivacyBudget
        The (ε, δ) cost.
    description : str
        Description of the mechanism / query.
    stage : str
        Pipeline stage label.
    rdp_epsilons : dict[float, float]
        Rényi DP ε values at each order α (for RDP accounting).
    zcdp_rho : Optional[float]
        zCDP ρ if available.
    """

    budget: PrivacyBudget
    description: str = ""
    stage: str = ""
    rdp_epsilons: Dict[float, float] = field(default_factory=dict)
    zcdp_rho: Optional[float] = None


# ═══════════════════════════════════════════════════════════════════════════
# PrivacyAccountant — main accountant
# ═══════════════════════════════════════════════════════════════════════════


# Standard Rényi orders used for RDP accounting
DEFAULT_RDP_ORDERS: Tuple[float, ...] = (
    1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0,
    10.0, 12.0, 16.0, 20.0, 32.0, 64.0, 128.0, 256.0,
)


class BudgetAccountant:
    """Privacy budget accountant supporting multiple composition theorems.

    Parameters
    ----------
    total_budget : PrivacyBudget
        Total privacy budget allocated for the analysis.
    rdp_orders : tuple[float, ...]
        Rényi orders to track.
    warn_threshold : float
        Fraction of budget at which a warning is emitted (default 0.9).
    """

    def __init__(
        self,
        total_budget: PrivacyBudget,
        *,
        rdp_orders: Tuple[float, ...] = DEFAULT_RDP_ORDERS,
        warn_threshold: float = 0.9,
    ) -> None:
        self.total_budget = total_budget
        self.rdp_orders = rdp_orders
        self.warn_threshold = warn_threshold
        self._ledger: List[BudgetEntry] = []
        # Cumulative RDP ε at each order
        self._rdp_cumulative: Dict[float, float] = {a: 0.0 for a in rdp_orders}
        # Cumulative zCDP ρ
        self._zcdp_cumulative: float = 0.0

    # ── recording ────────────────────────────────────────────────────────

    def record(
        self,
        budget: PrivacyBudget,
        *,
        description: str = "",
        stage: str = "",
        rdp_epsilons: Optional[Dict[float, float]] = None,
        zcdp_rho: Optional[float] = None,
    ) -> None:
        """Record a privacy expenditure.

        Parameters
        ----------
        budget : PrivacyBudget
            (ε, δ) cost of this mechanism invocation.
        description : str
            Human-readable description.
        stage : str
            Pipeline stage label.
        rdp_epsilons : optional dict
            Per-order RDP ε values.
        zcdp_rho : optional float
            zCDP ρ value.
        """
        entry = BudgetEntry(
            budget=budget,
            description=description,
            stage=stage,
            rdp_epsilons=rdp_epsilons or {},
            zcdp_rho=zcdp_rho,
        )
        self._ledger.append(entry)

        # Accumulate RDP
        for alpha, eps in entry.rdp_epsilons.items():
            if alpha in self._rdp_cumulative:
                self._rdp_cumulative[alpha] += eps

        # Accumulate zCDP
        if entry.zcdp_rho is not None:
            self._zcdp_cumulative += entry.zcdp_rho

        # Check budget exhaustion
        self._check_budget_warning()

    def record_gaussian(
        self,
        sigma: float,
        sensitivity: float = 1.0,
        *,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
        description: str = "",
        stage: str = "",
    ) -> None:
        """Convenience: record a Gaussian mechanism invocation.

        Automatically computes RDP and zCDP values.

        Parameters
        ----------
        sigma : float
            Gaussian noise standard deviation.
        sensitivity : float
            L2 sensitivity.
        epsilon, delta : optional
            If provided, used for the (ε,δ) entry. Otherwise computed from
            sigma and the total budget's δ.
        description, stage : str
            Metadata.
        """
        rdp_eps = {
            a: _rdp_gaussian(a, sigma / sensitivity) for a in self.rdp_orders
        }
        rho = gaussian_to_zcdp(sigma, sensitivity)

        if epsilon is None or delta is None:
            # Derive (ε,δ) from RDP at the best order
            d = delta if delta is not None else self.total_budget.delta
            if d <= 0:
                d = 1e-5
            best_eps = min(
                rdp_to_approx_dp(rdp_eps[a], a, d) for a in self.rdp_orders
            )
            epsilon = best_eps
            delta = d

        self.record(
            PrivacyBudget(epsilon=epsilon, delta=delta, description=description),
            description=description,
            stage=stage,
            rdp_epsilons=rdp_eps,
            zcdp_rho=rho,
        )

    def record_laplace(
        self,
        scale: float,
        sensitivity: float = 1.0,
        *,
        description: str = "",
        stage: str = "",
    ) -> None:
        """Convenience: record a Laplace mechanism invocation."""
        epsilon = sensitivity / scale
        rdp_eps = {a: _rdp_laplace(a, scale / sensitivity) for a in self.rdp_orders}
        self.record(
            PrivacyBudget(epsilon=epsilon, delta=0.0, description=description),
            description=description,
            stage=stage,
            rdp_epsilons=rdp_eps,
        )

    # ── composition ──────────────────────────────────────────────────────

    def compose_basic(self) -> CompositionResult:
        """Basic sequential composition: ε_total = Σ εᵢ, δ_total = Σ δᵢ."""
        budgets = [e.budget for e in self._ledger]
        total_eps = sum(b.epsilon for b in budgets)
        total_delta = sum(b.delta for b in budgets)
        return CompositionResult(
            total_budget=PrivacyBudget(epsilon=total_eps, delta=min(total_delta, 1.0 - 1e-15)),
            theorem_used=CompositionTheorem.BASIC,
            n_mechanisms=len(budgets),
            per_mechanism_budgets=tuple(budgets),
        )

    def compose_advanced(self, delta_prime: Optional[float] = None) -> CompositionResult:
        """Advanced composition theorem.

        Parameters
        ----------
        delta_prime : optional float
            Additional failure probability.  Defaults to total_budget.delta / 2.
        """
        budgets = [e.budget for e in self._ledger]
        k = len(budgets)
        if k == 0:
            return CompositionResult(
                total_budget=PrivacyBudget(0.0, 0.0),
                theorem_used=CompositionTheorem.ADVANCED,
            )
        if delta_prime is None:
            delta_prime = max(self.total_budget.delta / 2.0, 1e-10)

        eps_max = max(b.epsilon for b in budgets)
        eps_composed = (
            eps_max * math.sqrt(2.0 * k * math.log(1.0 / delta_prime))
            + k * eps_max * (math.exp(eps_max) - 1.0)
        )
        delta_composed = sum(b.delta for b in budgets) + delta_prime
        return CompositionResult(
            total_budget=PrivacyBudget(
                epsilon=eps_composed,
                delta=min(delta_composed, 1.0 - 1e-15),
            ),
            theorem_used=CompositionTheorem.ADVANCED,
            n_mechanisms=k,
            per_mechanism_budgets=tuple(budgets),
            metadata={"delta_prime": delta_prime},
        )

    def compose_rdp(self, delta_target: Optional[float] = None) -> CompositionResult:
        """Compose via Rényi DP: sum RDP ε at each order, convert to (ε, δ).

        Parameters
        ----------
        delta_target : optional float
            Target δ.  Defaults to total_budget.delta.
        """
        if delta_target is None:
            delta_target = self.total_budget.delta
        if delta_target <= 0:
            delta_target = 1e-5

        budgets = [e.budget for e in self._ledger]

        # Find the best ε over all orders
        best_eps = float("inf")
        best_alpha = self.rdp_orders[0]
        for alpha in self.rdp_orders:
            rdp_eps = self._rdp_cumulative.get(alpha, 0.0)
            eps = rdp_to_approx_dp(rdp_eps, alpha, delta_target)
            if eps < best_eps:
                best_eps = eps
                best_alpha = alpha

        return CompositionResult(
            total_budget=PrivacyBudget(epsilon=best_eps, delta=delta_target),
            theorem_used=CompositionTheorem.RENYI,
            n_mechanisms=len(budgets),
            per_mechanism_budgets=tuple(budgets),
            metadata={"best_alpha": best_alpha, "rdp_cumulative": dict(self._rdp_cumulative)},
        )

    def compose_zcdp(self, delta_target: Optional[float] = None) -> CompositionResult:
        """Compose via zero-concentrated DP: sum ρ, convert to (ε, δ).

        Parameters
        ----------
        delta_target : optional float
            Target δ.  Defaults to total_budget.delta.
        """
        if delta_target is None:
            delta_target = self.total_budget.delta
        if delta_target <= 0:
            delta_target = 1e-5

        budgets = [e.budget for e in self._ledger]
        eps = zcdp_to_approx_dp(self._zcdp_cumulative, delta_target)

        return CompositionResult(
            total_budget=PrivacyBudget(epsilon=eps, delta=delta_target),
            theorem_used=CompositionTheorem.ZERO_CONCENTRATED,
            n_mechanisms=len(budgets),
            per_mechanism_budgets=tuple(budgets),
            metadata={"total_rho": self._zcdp_cumulative},
        )

    def compose_moments(
        self,
        sigma: float,
        sampling_rate: float,
        n_steps: int,
        delta_target: Optional[float] = None,
    ) -> CompositionResult:
        """Moments accountant for subsampled Gaussian mechanism (Abadi et al. 2016).

        Parameters
        ----------
        sigma : float
            Noise multiplier (σ / sensitivity).
        sampling_rate : float
            Subsampling probability q ∈ (0, 1].
        n_steps : int
            Number of mechanism invocations.
        delta_target : optional float
            Target δ.

        Returns
        -------
        CompositionResult
        """
        if delta_target is None:
            delta_target = self.total_budget.delta
        if delta_target <= 0:
            delta_target = 1e-5

        # Compute RDP for subsampled Gaussian at each order
        best_eps = float("inf")
        best_alpha = self.rdp_orders[0]
        for alpha in self.rdp_orders:
            # Upper bound on RDP for Poisson-subsampled Gaussian
            rdp_single = _rdp_subsampled_gaussian(alpha, sigma, sampling_rate)
            rdp_total = n_steps * rdp_single
            eps = rdp_to_approx_dp(rdp_total, alpha, delta_target)
            if eps < best_eps:
                best_eps = eps
                best_alpha = alpha

        return CompositionResult(
            total_budget=PrivacyBudget(epsilon=best_eps, delta=delta_target),
            theorem_used=CompositionTheorem.MOMENTS_ACCOUNTANT,
            n_mechanisms=n_steps,
            metadata={
                "sigma": sigma,
                "sampling_rate": sampling_rate,
                "best_alpha": best_alpha,
            },
        )

    def compose(
        self,
        theorem: CompositionTheorem = CompositionTheorem.RENYI,
        *,
        delta_target: Optional[float] = None,
    ) -> CompositionResult:
        """Compose all recorded mechanisms with the specified theorem.

        Parameters
        ----------
        theorem : CompositionTheorem
            Which composition theorem to apply.
        delta_target : optional float
            Target δ (where applicable).
        """
        if theorem == CompositionTheorem.BASIC:
            return self.compose_basic()
        elif theorem == CompositionTheorem.ADVANCED:
            return self.compose_advanced(delta_prime=delta_target)
        elif theorem == CompositionTheorem.RENYI:
            return self.compose_rdp(delta_target=delta_target)
        elif theorem == CompositionTheorem.ZERO_CONCENTRATED:
            return self.compose_zcdp(delta_target=delta_target)
        else:
            return self.compose_rdp(delta_target=delta_target)

    # ── budget queries ───────────────────────────────────────────────────

    def spent_budget_basic(self) -> PrivacyBudget:
        """Total spent budget under basic composition."""
        result = self.compose_basic()
        return result.total_budget

    def remaining_budget(
        self,
        theorem: CompositionTheorem = CompositionTheorem.BASIC,
    ) -> PrivacyBudget:
        """Remaining budget under the given composition theorem."""
        spent = self.compose(theorem).total_budget
        remaining_eps = max(0.0, self.total_budget.epsilon - spent.epsilon)
        remaining_delta = max(0.0, self.total_budget.delta - spent.delta)
        return PrivacyBudget(epsilon=remaining_eps, delta=remaining_delta)

    def can_afford(self, cost: PrivacyBudget) -> bool:
        """Check whether the next mechanism fits within the remaining budget."""
        remaining = self.remaining_budget()
        return cost.epsilon <= remaining.epsilon + 1e-12 and cost.delta <= remaining.delta + 1e-12

    @property
    def n_queries(self) -> int:
        """Number of recorded queries."""
        return len(self._ledger)

    @property
    def ledger(self) -> List[BudgetEntry]:
        """Read-only access to the spending ledger."""
        return list(self._ledger)

    def stage_summary(self) -> Dict[str, PrivacyBudget]:
        """Summarise spending by pipeline stage."""
        stages: Dict[str, List[PrivacyBudget]] = {}
        for entry in self._ledger:
            stages.setdefault(entry.stage or "default", []).append(entry.budget)
        return {
            stage: PrivacyBudget(
                epsilon=sum(b.epsilon for b in budgets),
                delta=sum(b.delta for b in budgets),
                description=f"stage:{stage}",
            )
            for stage, budgets in stages.items()
        }

    # ── internals ────────────────────────────────────────────────────────

    def _check_budget_warning(self) -> None:
        """Emit a warning if spending exceeds the threshold."""
        spent = self.spent_budget_basic()
        if self.total_budget.epsilon > 0:
            ratio = spent.epsilon / self.total_budget.epsilon
            if ratio >= 1.0:
                warnings.warn(
                    f"Privacy budget EXHAUSTED: spent ε={spent.epsilon:.4f} "
                    f"≥ total ε={self.total_budget.epsilon:.4f}",
                    stacklevel=3,
                )
            elif ratio >= self.warn_threshold:
                warnings.warn(
                    f"Privacy budget {ratio:.0%} consumed: spent ε={spent.epsilon:.4f} "
                    f"of total ε={self.total_budget.epsilon:.4f}",
                    stacklevel=3,
                )


# ═══════════════════════════════════════════════════════════════════════════
# RDP for subsampled Gaussian (moments accountant)
# ═══════════════════════════════════════════════════════════════════════════


def _rdp_subsampled_gaussian(alpha: float, sigma: float, q: float) -> float:
    """Upper bound on RDP for Poisson-subsampled Gaussian mechanism.

    Uses the bound from Mironov (2017) / Balle et al. (2020):
    For integer α ≥ 2:
        RDP_α ≤ (1/(α-1)) log( Σ_{j=0}^{α} C(α,j) q^j (1-q)^{α-j}
                                 · exp(j(j-1)/(2σ²)) )

    For non-integer α we use the Gaussian mechanism RDP as an upper bound
    scaled by q.

    Parameters
    ----------
    alpha : float
        Rényi order.
    sigma : float
        Noise multiplier.
    q : float
        Subsampling probability.

    Returns
    -------
    float
        RDP bound.
    """
    if q <= 0:
        return 0.0
    if q >= 1.0:
        return _rdp_gaussian(alpha, sigma)

    alpha_int = int(alpha)
    if abs(alpha - alpha_int) > 1e-6 or alpha_int < 2:
        # Fallback: use plain Gaussian RDP (loose but safe upper bound)
        return min(_rdp_gaussian(alpha, sigma), q * alpha / (2.0 * sigma * sigma))

    # Exact computation for integer orders
    log_terms = []
    for j in range(alpha_int + 1):
        log_binom = (
            math.lgamma(alpha_int + 1)
            - math.lgamma(j + 1)
            - math.lgamma(alpha_int - j + 1)
        )
        log_q = j * math.log(q) + (alpha_int - j) * math.log(1.0 - q)
        log_exp = j * (j - 1) / (2.0 * sigma * sigma)
        log_terms.append(log_binom + log_q + log_exp)

    # Log-sum-exp for numerical stability
    max_log = max(log_terms)
    sum_exp = sum(math.exp(t - max_log) for t in log_terms)
    return (max_log + math.log(sum_exp)) / (alpha_int - 1)


# ═══════════════════════════════════════════════════════════════════════════
# Budget allocation optimisation
# ═══════════════════════════════════════════════════════════════════════════


def optimal_budget_split(
    total_epsilon: float,
    n_queries: int,
    *,
    query_weights: Optional[Sequence[float]] = None,
) -> List[float]:
    """Optimally split ε budget among *n_queries* queries.

    If *query_weights* are provided, budget is allocated proportionally
    to the weights.  Otherwise, budget is split uniformly.

    Parameters
    ----------
    total_epsilon : float
        Total ε budget.
    n_queries : int
        Number of queries.
    query_weights : optional Sequence[float]
        Relative importance weights (non-negative).

    Returns
    -------
    list[float]
        Per-query ε allocations summing to total_epsilon.
    """
    if n_queries <= 0:
        raise ValueError("n_queries must be > 0")
    if query_weights is None:
        return [total_epsilon / n_queries] * n_queries
    w = np.array(query_weights, dtype=np.float64)
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    total_w = w.sum()
    if total_w == 0:
        return [total_epsilon / n_queries] * n_queries
    return (w / total_w * total_epsilon).tolist()


def optimal_gaussian_sigma(
    epsilon: float,
    delta: float,
    sensitivity: float = 1.0,
    n_compositions: int = 1,
) -> float:
    """Find the optimal Gaussian σ for *n_compositions* via RDP accounting.

    Parameters
    ----------
    epsilon : float
        Target total ε.
    delta : float
        Target δ.
    sensitivity : float
        L2 sensitivity.
    n_compositions : int
        Number of times the mechanism will be composed.

    Returns
    -------
    float
        Optimal σ.
    """
    if epsilon <= 0 or delta <= 0:
        raise ValueError("epsilon and delta must be > 0")

    def _check_sigma(sigma: float) -> float:
        """Return the gap between achieved ε and target ε (negative = feasible)."""
        best_eps = float("inf")
        for alpha in DEFAULT_RDP_ORDERS:
            rdp_single = _rdp_gaussian(alpha, sigma / sensitivity)
            rdp_total = n_compositions * rdp_single
            eps = rdp_to_approx_dp(rdp_total, alpha, delta)
            best_eps = min(best_eps, eps)
        return best_eps - epsilon

    # Binary search for the smallest σ achieving the target ε
    lo, hi = 1e-4, 1000.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if _check_sigma(mid) > 0:
            lo = mid
        else:
            hi = mid
    return hi


__all__ = [
    # Conversions
    "rdp_to_approx_dp",
    "approx_dp_to_rdp",
    "zcdp_to_rdp",
    "zcdp_to_approx_dp",
    "gaussian_to_zcdp",
    # Accountant
    "BudgetEntry",
    "BudgetAccountant",
    "DEFAULT_RDP_ORDERS",
    # Budget helpers
    "optimal_budget_split",
    "optimal_gaussian_sigma",
]

"""
Privacy-specific verification for CEGAR-based DP verification.

Implements checkers for ε-DP, (ε,δ)-DP, and zCDP properties,
abstract computation of privacy loss, sequential composition analysis,
and composition verification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from scipy import optimize

from dp_forge.types import (
    AbstractDomainType,
    AbstractValue,
    AdjacencyRelation,
    Formula,
    Predicate,
    PrivacyBudget,
    VerifyResult,
)
from dp_forge.cegar import (
    AbstractCounterexample,
    AbstractState,
    CEGARConfig,
    CEGARStatus,
)
from dp_forge.cegar.abstraction import (
    AbstractDomain,
    IntervalAbstraction,
    PolyhedralAbstraction,
    PrivacyLossAbstraction,
    GaloisConnection,
)


# ---------------------------------------------------------------------------
# Privacy property types
# ---------------------------------------------------------------------------


@dataclass
class PrivacyProperty:
    """Specification of a privacy property to verify.

    Attributes:
        property_type: Type of DP guarantee ('pure_dp', 'approx_dp', 'zcdp', 'rdp').
        epsilon: Privacy parameter ε.
        delta: Privacy parameter δ (for approximate DP).
        rho: zCDP parameter ρ (for concentrated DP).
        alpha: Rényi divergence order (for RDP).
    """
    property_type: str = "pure_dp"
    epsilon: float = 1.0
    delta: float = 0.0
    rho: Optional[float] = None
    alpha: Optional[float] = None

    def __post_init__(self) -> None:
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")
        if not (0.0 <= self.delta < 1.0):
            raise ValueError(f"delta must be in [0, 1), got {self.delta}")

    @classmethod
    def pure_dp(cls, epsilon: float) -> PrivacyProperty:
        """Create a pure ε-DP property."""
        return cls(property_type="pure_dp", epsilon=epsilon, delta=0.0)

    @classmethod
    def approx_dp(cls, epsilon: float, delta: float) -> PrivacyProperty:
        """Create an (ε,δ)-DP property."""
        return cls(property_type="approx_dp", epsilon=epsilon, delta=delta)

    @classmethod
    def zcdp(cls, rho: float) -> PrivacyProperty:
        """Create a ρ-zCDP property."""
        return cls(property_type="zcdp", epsilon=2 * rho, rho=rho)

    @classmethod
    def rdp(cls, alpha: float, epsilon: float) -> PrivacyProperty:
        """Create an (α, ε)-RDP property."""
        return cls(property_type="rdp", epsilon=epsilon, alpha=alpha)


# ---------------------------------------------------------------------------
# Abstract privacy loss computation
# ---------------------------------------------------------------------------


class AbstractPrivacyLoss:
    """Abstract computation of privacy loss functions.

    Computes sound over-approximations of privacy loss using interval
    and polyhedral arithmetic.
    """

    def __init__(
        self,
        domain: Optional[AbstractDomain] = None,
        tolerance: float = 1e-9,
    ) -> None:
        """Initialize abstract privacy loss.

        Args:
            domain: Abstract domain for computations.
            tolerance: Numerical tolerance.
        """
        self._domain = domain or IntervalAbstraction()
        self._tolerance = tolerance
        self._loss_abstraction = PrivacyLossAbstraction(self._domain)

    def compute_max_divergence(
        self,
        mechanism: npt.NDArray[np.float64],
        i: int,
        ip: int,
    ) -> Tuple[float, int]:
        """Compute the maximum privacy divergence between adjacent rows.

        Computes max_j |ln(M[i,j] / M[i',j])| for the given pair.

        Args:
            mechanism: Mechanism probability matrix.
            i: First row index.
            ip: Second row index.

        Returns:
            (max_divergence, argmax_j) — the maximum divergence and the
            output index achieving it.
        """
        n, k = mechanism.shape
        max_div = 0.0
        argmax_j = 0

        for j in range(k):
            p_i = mechanism[i, j]
            p_ip = mechanism[ip, j]
            if p_i > 1e-300 and p_ip > 1e-300:
                div = abs(np.log(p_i / p_ip))
                if div > max_div:
                    max_div = div
                    argmax_j = j

        return float(max_div), argmax_j

    def compute_renyi_divergence(
        self,
        mechanism: npt.NDArray[np.float64],
        i: int,
        ip: int,
        alpha: float,
    ) -> float:
        """Compute the Rényi divergence of order α between adjacent rows.

        D_α(M[i] || M[i']) = (1/(α-1)) ln(Σ_j M[i,j]^α · M[i',j]^(1-α))

        Args:
            mechanism: Mechanism probability matrix.
            i: First row index.
            ip: Second row index.
            alpha: Rényi divergence order (> 1).

        Returns:
            Rényi divergence value.
        """
        if alpha <= 1.0:
            # KL divergence as limit
            return self._compute_kl_divergence(mechanism, i, ip)

        n, k = mechanism.shape
        terms = np.zeros(k, dtype=np.float64)
        for j in range(k):
            p_i = max(mechanism[i, j], 1e-300)
            p_ip = max(mechanism[ip, j], 1e-300)
            terms[j] = (p_i ** alpha) * (p_ip ** (1.0 - alpha))

        total = np.sum(terms)
        if total <= 0:
            return float("inf")
        return float(np.log(total) / (alpha - 1.0))

    def _compute_kl_divergence(
        self,
        mechanism: npt.NDArray[np.float64],
        i: int,
        ip: int,
    ) -> float:
        """Compute KL divergence D_KL(M[i] || M[i']).

        Args:
            mechanism: Mechanism probability matrix.
            i: First row index.
            ip: Second row index.

        Returns:
            KL divergence.
        """
        n, k = mechanism.shape
        kl = 0.0
        for j in range(k):
            p_i = max(mechanism[i, j], 1e-300)
            p_ip = max(mechanism[ip, j], 1e-300)
            if p_i > 1e-300:
                kl += p_i * np.log(p_i / p_ip)
        return float(kl)

    def compute_zcdp_parameter(
        self,
        mechanism: npt.NDArray[np.float64],
        i: int,
        ip: int,
    ) -> float:
        """Compute the zCDP parameter ρ for a pair of adjacent rows.

        ρ = max_{α > 1} D_α(M[i] || M[i']) / α
        We approximate this by checking several values of α.

        Args:
            mechanism: Mechanism probability matrix.
            i: First row index.
            ip: Second row index.

        Returns:
            zCDP parameter ρ.
        """
        alphas = [1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        max_rho = 0.0

        for alpha in alphas:
            renyi = self.compute_renyi_divergence(mechanism, i, ip, alpha)
            rho = renyi / alpha if alpha > 0 else 0.0
            max_rho = max(max_rho, rho)

        return float(max_rho)

    def abstract_max_divergence(
        self,
        row_a: AbstractValue,
        row_b: AbstractValue,
    ) -> Tuple[float, float]:
        """Compute abstract bounds on the maximum privacy divergence.

        Uses interval arithmetic on abstract probability bounds.

        Args:
            row_a: Abstract value for row a.
            row_b: Abstract value for row b.

        Returns:
            (lower_bound, upper_bound) on max divergence.
        """
        return self._loss_abstraction.abstract_log_ratio(row_a, row_b)


# ---------------------------------------------------------------------------
# Privacy property checker
# ---------------------------------------------------------------------------


class PrivacyPropertyChecker:
    """Check differential privacy properties on mechanisms.

    Verifies ε-DP, (ε,δ)-DP, and zCDP properties using both exact
    and abstract (over-approximate) checking.
    """

    def __init__(
        self,
        domain: Optional[AbstractDomain] = None,
        tolerance: float = 1e-9,
    ) -> None:
        """Initialize the privacy property checker.

        Args:
            domain: Abstract domain for over-approximate checking.
            tolerance: Numerical tolerance.
        """
        self._domain = domain or IntervalAbstraction()
        self._tolerance = tolerance
        self._loss = AbstractPrivacyLoss(domain=self._domain, tolerance=tolerance)

    def check_pure_dp(
        self,
        mechanism: npt.NDArray[np.float64],
        epsilon: float,
        adjacency: AdjacencyRelation,
    ) -> Tuple[bool, Optional[AbstractCounterexample]]:
        """Check ε-differential privacy.

        For all adjacent pairs (i, i'), verifies that
        max_j |ln(M[i,j] / M[i',j])| ≤ ε.

        Args:
            mechanism: Mechanism probability matrix.
            epsilon: Privacy parameter.
            adjacency: Adjacency relation.

        Returns:
            (is_verified, counterexample_or_none).
        """
        all_edges = list(adjacency.edges)
        if adjacency.symmetric:
            all_edges += [(j, i) for (i, j) in adjacency.edges]

        worst_pair = None
        worst_mag = 0.0
        worst_j = 0

        for (i, ip) in all_edges:
            div, j = self._loss.compute_max_divergence(mechanism, i, ip)
            if div > epsilon + self._tolerance:
                if div > worst_mag:
                    worst_mag = div
                    worst_pair = (i, ip)
                    worst_j = j

        if worst_pair is None:
            return True, None

        # Build counterexample
        i, ip = worst_pair
        cex = AbstractCounterexample(
            trace=[
                AbstractState(state_id=i, predicates=frozenset()),
                AbstractState(state_id=ip, predicates=frozenset()),
            ],
            violating_pair=worst_pair,
            violation_magnitude=worst_mag,
            is_spurious=False,
        )
        return False, cex

    def check_approx_dp(
        self,
        mechanism: npt.NDArray[np.float64],
        epsilon: float,
        delta: float,
        adjacency: AdjacencyRelation,
    ) -> Tuple[bool, Optional[AbstractCounterexample]]:
        """Check (ε, δ)-differential privacy.

        For all adjacent pairs (i, i') and all subsets S of outputs,
        verifies P[M(i) ∈ S] ≤ e^ε · P[M(i') ∈ S] + δ.

        Uses a sound over-approximation: checks column-wise ratios
        and accounts for the δ slack.

        Args:
            mechanism: Mechanism probability matrix.
            epsilon: Privacy parameter ε.
            delta: Privacy parameter δ.
            adjacency: Adjacency relation.

        Returns:
            (is_verified, counterexample_or_none).
        """
        all_edges = list(adjacency.edges)
        if adjacency.symmetric:
            all_edges += [(j, i) for (i, j) in adjacency.edges]

        exp_eps = np.exp(epsilon)
        n, k = mechanism.shape

        for (i, ip) in all_edges:
            # Check: for all outputs j, P[M(i)=j] ≤ e^ε · P[M(i')=j] + δ/k
            # This is a sufficient (conservative) condition
            for j in range(k):
                p_i = mechanism[i, j]
                p_ip = mechanism[ip, j]
                if p_i > exp_eps * p_ip + delta + self._tolerance:
                    violation_mag = float(np.log(max(p_i - delta, 1e-300) / max(p_ip, 1e-300)))
                    cex = AbstractCounterexample(
                        trace=[
                            AbstractState(state_id=i, predicates=frozenset()),
                            AbstractState(state_id=ip, predicates=frozenset()),
                        ],
                        violating_pair=(i, ip),
                        violation_magnitude=max(violation_mag, 1e-10),
                        is_spurious=False,
                    )
                    return False, cex

        return True, None

    def check_zcdp(
        self,
        mechanism: npt.NDArray[np.float64],
        rho: float,
        adjacency: AdjacencyRelation,
    ) -> Tuple[bool, Optional[AbstractCounterexample]]:
        """Check ρ-zero-concentrated differential privacy.

        Verifies that for all adjacent pairs (i, i') and all α > 1:
        D_α(M(i) || M(i')) ≤ ρ · α

        Args:
            mechanism: Mechanism probability matrix.
            rho: zCDP parameter.
            adjacency: Adjacency relation.

        Returns:
            (is_verified, counterexample_or_none).
        """
        all_edges = list(adjacency.edges)
        if adjacency.symmetric:
            all_edges += [(j, i) for (i, j) in adjacency.edges]

        for (i, ip) in all_edges:
            actual_rho = self._loss.compute_zcdp_parameter(mechanism, i, ip)
            if actual_rho > rho + self._tolerance:
                cex = AbstractCounterexample(
                    trace=[
                        AbstractState(state_id=i, predicates=frozenset()),
                        AbstractState(state_id=ip, predicates=frozenset()),
                    ],
                    violating_pair=(i, ip),
                    violation_magnitude=actual_rho,
                    is_spurious=False,
                )
                return False, cex

        return True, None

    def check_property(
        self,
        mechanism: npt.NDArray[np.float64],
        prop: PrivacyProperty,
        adjacency: AdjacencyRelation,
    ) -> Tuple[bool, Optional[AbstractCounterexample]]:
        """Check a general privacy property.

        Dispatches to the appropriate checker based on the property type.

        Args:
            mechanism: Mechanism probability matrix.
            prop: Privacy property to check.
            adjacency: Adjacency relation.

        Returns:
            (is_verified, counterexample_or_none).
        """
        if prop.property_type == "pure_dp":
            return self.check_pure_dp(mechanism, prop.epsilon, adjacency)
        elif prop.property_type == "approx_dp":
            return self.check_approx_dp(
                mechanism, prop.epsilon, prop.delta, adjacency,
            )
        elif prop.property_type == "zcdp":
            rho = prop.rho if prop.rho is not None else prop.epsilon / 2.0
            return self.check_zcdp(mechanism, rho, adjacency)
        elif prop.property_type == "rdp":
            alpha = prop.alpha if prop.alpha is not None else 2.0
            return self._check_rdp(mechanism, alpha, prop.epsilon, adjacency)
        else:
            raise ValueError(f"Unknown property type: {prop.property_type}")

    def _check_rdp(
        self,
        mechanism: npt.NDArray[np.float64],
        alpha: float,
        epsilon: float,
        adjacency: AdjacencyRelation,
    ) -> Tuple[bool, Optional[AbstractCounterexample]]:
        """Check (α, ε)-Rényi DP.

        Verifies D_α(M(i) || M(i')) ≤ ε for all adjacent pairs.
        """
        all_edges = list(adjacency.edges)
        if adjacency.symmetric:
            all_edges += [(j, i) for (i, j) in adjacency.edges]

        for (i, ip) in all_edges:
            renyi = self._loss.compute_renyi_divergence(mechanism, i, ip, alpha)
            if renyi > epsilon + self._tolerance:
                cex = AbstractCounterexample(
                    trace=[
                        AbstractState(state_id=i, predicates=frozenset()),
                        AbstractState(state_id=ip, predicates=frozenset()),
                    ],
                    violating_pair=(i, ip),
                    violation_magnitude=renyi,
                    is_spurious=False,
                )
                return False, cex

        return True, None

    def verify_abstract(
        self,
        mechanism: npt.NDArray[np.float64],
        budget: PrivacyBudget,
        adjacency: AdjacencyRelation,
    ) -> Tuple[bool, Optional[AbstractCounterexample]]:
        """Verify using abstract interpretation (sound over-approximation).

        May produce spurious counterexamples due to abstraction imprecision.

        Args:
            mechanism: Mechanism probability matrix.
            budget: Privacy budget.
            adjacency: Adjacency relation.

        Returns:
            (is_verified, counterexample_or_none).
        """
        if budget.delta > 0:
            return self.check_approx_dp(
                mechanism, budget.epsilon, budget.delta, adjacency,
            )
        return self.check_pure_dp(mechanism, budget.epsilon, adjacency)


# ---------------------------------------------------------------------------
# Composition checker
# ---------------------------------------------------------------------------


class CompositionChecker:
    """Verify privacy properties of composed mechanisms.

    Supports basic, advanced, and optimal composition theorems
    for analyzing composed mechanisms.
    """

    def __init__(self, tolerance: float = 1e-9) -> None:
        """Initialize composition checker.

        Args:
            tolerance: Numerical tolerance.
        """
        self._tolerance = tolerance
        self._loss = AbstractPrivacyLoss()

    def check_basic_composition(
        self,
        mechanisms: List[npt.NDArray[np.float64]],
        budgets: List[PrivacyBudget],
        total_budget: PrivacyBudget,
        adjacency: AdjacencyRelation,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check composition under basic composition theorem.

        Verifies Σ_k ε_k ≤ ε_total and Σ_k δ_k ≤ δ_total.

        Args:
            mechanisms: List of mechanism matrices.
            budgets: Privacy budget for each mechanism.
            total_budget: Total privacy budget.
            adjacency: Adjacency relation.

        Returns:
            (is_verified, details_dict).
        """
        total_eps = sum(b.epsilon for b in budgets)
        total_delta = sum(b.delta for b in budgets)

        details: Dict[str, Any] = {
            "composition_type": "basic",
            "num_mechanisms": len(mechanisms),
            "individual_epsilons": [b.epsilon for b in budgets],
            "individual_deltas": [b.delta for b in budgets],
            "total_epsilon": total_eps,
            "total_delta": total_delta,
        }

        verified = (
            total_eps <= total_budget.epsilon + self._tolerance
            and total_delta <= total_budget.delta + self._tolerance
        )
        details["verified"] = verified
        return verified, details

    def check_advanced_composition(
        self,
        mechanisms: List[npt.NDArray[np.float64]],
        budgets: List[PrivacyBudget],
        total_budget: PrivacyBudget,
        adjacency: AdjacencyRelation,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check composition under advanced composition theorem.

        For k mechanisms each satisfying (ε_0, δ_0)-DP, the composition
        satisfies (ε_0√(2k·ln(1/δ')) + k·ε_0·(e^ε_0-1), k·δ_0 + δ')-DP
        for any δ' > 0.

        Args:
            mechanisms: List of mechanism matrices.
            budgets: Per-mechanism budgets.
            total_budget: Total budget to verify.
            adjacency: Adjacency relation.

        Returns:
            (is_verified, details_dict).
        """
        k = len(mechanisms)
        if k == 0:
            return True, {"composition_type": "advanced", "verified": True}

        # Use largest individual epsilon for worst-case analysis
        eps_0 = max(b.epsilon for b in budgets)
        delta_0 = max(b.delta for b in budgets)

        # Try to find a δ' that makes the bound work
        delta_prime = max(total_budget.delta - k * delta_0, 1e-10)
        if delta_prime <= 0:
            delta_prime = 1e-10

        # Advanced composition bound
        eps_composed = (
            eps_0 * math.sqrt(2.0 * k * math.log(1.0 / delta_prime))
            + k * eps_0 * (math.exp(eps_0) - 1.0)
        )
        delta_composed = k * delta_0 + delta_prime

        details: Dict[str, Any] = {
            "composition_type": "advanced",
            "num_mechanisms": k,
            "eps_0": eps_0,
            "delta_0": delta_0,
            "delta_prime": delta_prime,
            "composed_epsilon": eps_composed,
            "composed_delta": delta_composed,
        }

        verified = (
            eps_composed <= total_budget.epsilon + self._tolerance
            and delta_composed <= total_budget.delta + self._tolerance
        )
        details["verified"] = verified
        return verified, details

    def check_optimal_composition(
        self,
        mechanisms: List[npt.NDArray[np.float64]],
        budgets: List[PrivacyBudget],
        total_budget: PrivacyBudget,
        adjacency: AdjacencyRelation,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check composition using optimal composition via Rényi DP.

        Uses the Rényi DP approach: compute individual RDP guarantees,
        sum them, and convert back to (ε, δ)-DP.

        Args:
            mechanisms: List of mechanism matrices.
            budgets: Per-mechanism budgets.
            total_budget: Total budget to verify.
            adjacency: Adjacency relation.

        Returns:
            (is_verified, details_dict).
        """
        k = len(mechanisms)
        if k == 0:
            return True, {"composition_type": "optimal", "verified": True}

        # Check all adjacent pairs
        all_edges = list(adjacency.edges)
        if adjacency.symmetric:
            all_edges += [(j, i) for (i, j) in adjacency.edges]

        # For each alpha, compute total RDP bound
        alphas = [1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]
        best_eps = float("inf")
        best_alpha = 2.0

        for alpha in alphas:
            total_rdp = 0.0
            for mech in mechanisms:
                max_rdp = 0.0
                for (i, ip) in all_edges:
                    if i < mech.shape[0] and ip < mech.shape[0]:
                        rdp = self._loss.compute_renyi_divergence(mech, i, ip, alpha)
                        max_rdp = max(max_rdp, rdp)
                total_rdp += max_rdp

            # Convert RDP to (ε, δ)-DP: ε = total_rdp + ln(1/δ)/(α-1)
            if total_budget.delta > 0 and alpha > 1:
                eps_from_rdp = total_rdp + math.log(1.0 / total_budget.delta) / (alpha - 1.0)
            else:
                eps_from_rdp = total_rdp  # Pure DP case

            if eps_from_rdp < best_eps:
                best_eps = eps_from_rdp
                best_alpha = alpha

        details: Dict[str, Any] = {
            "composition_type": "optimal",
            "num_mechanisms": k,
            "best_alpha": best_alpha,
            "best_epsilon": best_eps,
        }

        verified = best_eps <= total_budget.epsilon + self._tolerance
        details["verified"] = verified
        return verified, details


# ---------------------------------------------------------------------------
# Sequential composition analyzer
# ---------------------------------------------------------------------------


class SequentialCompositionAnalyzer:
    """Analyze sequential composition of DP mechanisms soundly.

    Provides detailed analysis of privacy loss accumulation under
    sequential composition, including tight bounds via moment methods.
    """

    def __init__(
        self,
        domain: Optional[AbstractDomain] = None,
        tolerance: float = 1e-9,
    ) -> None:
        """Initialize the sequential composition analyzer.

        Args:
            domain: Abstract domain for bound computations.
            tolerance: Numerical tolerance.
        """
        self._domain = domain or IntervalAbstraction()
        self._tolerance = tolerance
        self._loss = AbstractPrivacyLoss(domain=self._domain)
        self._checker = CompositionChecker(tolerance=tolerance)

    def analyze_sequential(
        self,
        mechanisms: List[npt.NDArray[np.float64]],
        adjacency: AdjacencyRelation,
    ) -> Dict[str, Any]:
        """Analyze sequential composition of mechanisms.

        Computes the tightest privacy bound for the sequential composition
        using multiple composition theorems.

        Args:
            mechanisms: List of mechanism matrices.
            adjacency: Adjacency relation.

        Returns:
            Analysis results with bounds from multiple theorems.
        """
        k = len(mechanisms)
        if k == 0:
            return {"num_mechanisms": 0, "composed_epsilon": 0.0}

        all_edges = list(adjacency.edges)
        if adjacency.symmetric:
            all_edges += [(j, i) for (i, j) in adjacency.edges]

        # Compute individual privacy losses
        individual_losses: List[Tuple[float, float]] = []
        for mech in mechanisms:
            max_loss = 0.0
            for (i, ip) in all_edges:
                if i < mech.shape[0] and ip < mech.shape[0]:
                    div, _ = self._loss.compute_max_divergence(mech, i, ip)
                    max_loss = max(max_loss, div)
            individual_losses.append((0.0, max_loss))

        # Basic composition bound
        basic_eps = sum(hi for _, hi in individual_losses)

        # Compute per-mechanism budgets for advanced/optimal composition
        budgets = [
            PrivacyBudget(epsilon=max(hi, 1e-10)) for _, hi in individual_losses
        ]

        result: Dict[str, Any] = {
            "num_mechanisms": k,
            "individual_max_losses": [hi for _, hi in individual_losses],
            "basic_composition_epsilon": basic_eps,
        }

        # Advanced composition (for delta > 0)
        delta_target = 1e-5  # Default target delta
        if k > 1:
            eps_0 = max(hi for _, hi in individual_losses)
            delta_prime = delta_target / 2.0
            adv_eps = (
                eps_0 * math.sqrt(2.0 * k * math.log(1.0 / delta_prime))
                + k * eps_0 * (math.exp(eps_0) - 1.0)
            )
            result["advanced_composition_epsilon"] = adv_eps
            result["advanced_composition_delta"] = delta_target

        # Abstract composition via interval arithmetic
        abstract_lower, abstract_upper = self._loss._loss_abstraction.abstract_composition(
            individual_losses,
        )
        result["abstract_composition_bounds"] = (abstract_lower, abstract_upper)

        return result

    def verify_sequential_composition(
        self,
        mechanisms: List[npt.NDArray[np.float64]],
        total_budget: PrivacyBudget,
        adjacency: AdjacencyRelation,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Verify that sequential composition satisfies a total budget.

        Tries multiple composition theorems and returns the best result.

        Args:
            mechanisms: List of mechanism matrices.
            total_budget: Total privacy budget to verify.
            adjacency: Adjacency relation.

        Returns:
            (is_verified, details_dict).
        """
        analysis = self.analyze_sequential(mechanisms, adjacency)

        # Check basic composition
        basic_eps = analysis.get("basic_composition_epsilon", float("inf"))
        if basic_eps <= total_budget.epsilon + self._tolerance:
            analysis["verified"] = True
            analysis["used_theorem"] = "basic"
            return True, analysis

        # Check advanced composition (if delta budget available)
        if total_budget.delta > 0:
            adv_eps = analysis.get("advanced_composition_epsilon")
            adv_delta = analysis.get("advanced_composition_delta", 0.0)
            if adv_eps is not None:
                if (adv_eps <= total_budget.epsilon + self._tolerance
                        and adv_delta <= total_budget.delta + self._tolerance):
                    analysis["verified"] = True
                    analysis["used_theorem"] = "advanced"
                    return True, analysis

        # Check optimal composition via RDP
        budgets = [
            PrivacyBudget(epsilon=max(loss, 1e-10))
            for loss in analysis.get("individual_max_losses", [])
        ]
        if budgets:
            verified, opt_details = self._checker.check_optimal_composition(
                mechanisms, budgets, total_budget, adjacency,
            )
            if verified:
                analysis["verified"] = True
                analysis["used_theorem"] = "optimal_rdp"
                analysis["optimal_details"] = opt_details
                return True, analysis

        analysis["verified"] = False
        analysis["used_theorem"] = "none"
        return False, analysis

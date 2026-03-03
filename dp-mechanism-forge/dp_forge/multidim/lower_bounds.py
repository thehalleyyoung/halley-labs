"""
Information-theoretic lower bounds for multi-dimensional DP mechanisms.

Implements Fano–Assouad lower bounds and minimax error bounds for
d-dimensional queries under differential privacy constraints.  These
bounds certify how close a synthesised mechanism is to the information-
theoretic optimum.

Theorems Implemented:
    - **Fano–Assouad (T17)**: Lower bound on estimation error for any
      ε-DP mechanism answering d-dimensional queries, based on mutual
      information constraints under DP.
    - **Minimax bound**: Worst-case error lower bound over all mechanisms
      in a given family.
    - **Gap analysis**: Ratio of achieved error to lower bound, measuring
      optimality of the synthesised mechanism.

Classes:
    LowerBoundResult  — structured output of lower bound computation
    LowerBoundComputer — main computation engine
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from dp_forge.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class LowerBoundResult:
    """Result of a lower bound computation.

    Attributes:
        bound_value: The computed lower bound on expected error.
        bound_type: Name of the theorem/method used.
        parameters: Parameters used in the computation.
        achieved_error: Achieved error of the mechanism (if provided).
        gap_ratio: achieved_error / bound_value (optimality gap).
        is_tight: Whether the gap ratio is close to 1.
        metadata: Additional computation details.
    """

    bound_value: float
    bound_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    achieved_error: Optional[float] = None
    gap_ratio: Optional[float] = None
    is_tight: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.bound_value < 0:
            raise ValueError(
                f"bound_value must be >= 0, got {self.bound_value}"
            )
        if self.achieved_error is not None and self.achieved_error < 0:
            raise ValueError(
                f"achieved_error must be >= 0, got {self.achieved_error}"
            )
        if self.achieved_error is not None and self.bound_value > 0:
            self.gap_ratio = self.achieved_error / self.bound_value
            self.is_tight = self.gap_ratio < 2.0

    def __repr__(self) -> str:
        gap = f", gap={self.gap_ratio:.2f}" if self.gap_ratio is not None else ""
        return (
            f"LowerBoundResult(bound={self.bound_value:.6f}, "
            f"type={self.bound_type!r}{gap})"
        )


class LowerBoundComputer:
    """Computes information-theoretic lower bounds for DP mechanism design.

    Provides Fano–Assouad bounds, minimax bounds, and optimality gap
    analysis for multi-dimensional query mechanisms.

    Args:
        loss_type: Loss function type ("L1", "L2", "Linf").
    """

    def __init__(self, loss_type: str = "L2") -> None:
        if loss_type not in ("L1", "L2", "Linf"):
            raise ConfigurationError(
                f"loss_type must be L1, L2, or Linf, got {loss_type!r}",
                parameter="loss_type",
                value=loss_type,
            )
        self._loss_type = loss_type

    def fano_assouad(
        self,
        d: int,
        epsilon: float,
        delta: float = 0.0,
        sensitivity: float = 1.0,
        domain_size: int = 2,
        n_databases: Optional[int] = None,
    ) -> LowerBoundResult:
        """Compute Fano–Assouad lower bound for d-dimensional queries.

        Based on Theorem T17: for any ε-DP mechanism answering d scalar
        queries each with sensitivity Δ over a domain of size n, the
        minimax L2 error is at least:

            Ω(d · Δ² / ε²)  for pure DP (δ = 0)
            Ω(d · Δ² / (ε² + ε))  for approximate DP

        The proof constructs a packing of hypotheses in the data domain
        and bounds mutual information under DP constraints.

        Args:
            d: Number of dimensions (queries).
            epsilon: Privacy parameter ε.
            delta: Privacy parameter δ.
            sensitivity: Per-coordinate sensitivity Δ.
            domain_size: Size of each coordinate's input domain.
            n_databases: Total number of databases (for refined bounds).

        Returns:
            LowerBoundResult with the Fano–Assouad bound.
        """
        if d < 1:
            raise ConfigurationError(
                f"d must be >= 1, got {d}", parameter="d", value=d
            )
        if epsilon <= 0:
            raise ConfigurationError(
                f"epsilon must be > 0, got {epsilon}",
                parameter="epsilon", value=epsilon,
            )
        if domain_size < 2:
            raise ConfigurationError(
                f"domain_size must be >= 2, got {domain_size}",
                parameter="domain_size", value=domain_size,
            )
        # Mutual information bound under ε-DP
        # I(X; Y) ≤ ε(e^ε - 1) per query (Duchi et al.)
        mi_per_query = epsilon * (math.exp(epsilon) - 1.0)
        if delta > 0:
            # Additional δ term via data processing inequality
            mi_per_query += delta * math.log(domain_size)
        # Fano's inequality: for M hypotheses with pairwise distance r,
        # error ≥ r/2 · (1 - (I + log2) / log M)
        n_hypotheses = domain_size ** d if n_databases is None else n_databases
        log_M = d * math.log(domain_size) if n_databases is None else math.log(max(n_databases, 2))
        total_mi = d * mi_per_query
        fano_factor = max(0.0, 1.0 - (total_mi + math.log(2)) / max(log_M, 1e-30))
        if self._loss_type == "L2":
            # Packing radius: Δ/(2√d) for L2
            packing_radius_sq = (sensitivity ** 2) / 4.0
            bound = d * packing_radius_sq * fano_factor
        elif self._loss_type == "L1":
            packing_radius = sensitivity / 2.0
            bound = d * packing_radius * fano_factor
        else:  # Linf
            packing_radius = sensitivity / 2.0
            bound = packing_radius * fano_factor
        # Assouad refinement: use binary hypercube packing
        assouad_bound = self._assouad_refinement(
            d, epsilon, delta, sensitivity, domain_size
        )
        bound = max(bound, assouad_bound)
        return LowerBoundResult(
            bound_value=bound,
            bound_type="fano_assouad",
            parameters={
                "d": d,
                "epsilon": epsilon,
                "delta": delta,
                "sensitivity": sensitivity,
                "domain_size": domain_size,
                "loss_type": self._loss_type,
            },
            metadata={
                "mi_per_query": mi_per_query,
                "fano_factor": fano_factor,
                "assouad_bound": assouad_bound,
            },
        )

    def _assouad_refinement(
        self,
        d: int,
        epsilon: float,
        delta: float,
        sensitivity: float,
        domain_size: int,
    ) -> float:
        """Assouad's lemma refinement for the lower bound.

        Uses a 2^d binary hypercube packing where each vertex differs
        in exactly one coordinate from its neighbours.  Under ε-DP,
        the testing error for each coordinate is bounded by the
        TV distance between adjacent mechanisms.

        Returns:
            Assouad lower bound value.
        """
        # TV distance bound: TV(M(x), M(x')) ≤ (e^ε - 1) / (e^ε + 1)
        tv_bound = (math.exp(epsilon) - 1.0) / (math.exp(epsilon) + 1.0)
        if delta > 0:
            tv_bound = min(tv_bound + delta, 1.0)
        # Per-coordinate testing error: (1 - tv_bound) / 2
        testing_error = (1.0 - tv_bound) / 2.0
        if self._loss_type == "L2":
            # Each coordinate contributes Δ² · testing_error / 4
            per_coord = (sensitivity ** 2 / 4.0) * (1.0 - tv_bound)
            return d * per_coord
        elif self._loss_type == "L1":
            per_coord = (sensitivity / 2.0) * (1.0 - tv_bound)
            return d * per_coord
        else:  # Linf
            per_coord = (sensitivity / 2.0) * (1.0 - tv_bound)
            return per_coord

    def minimax_bound(
        self,
        d: int,
        epsilon: float,
        delta: float = 0.0,
        sensitivity: float = 1.0,
        n_databases: int = 2,
        k: int = 100,
    ) -> LowerBoundResult:
        """Compute minimax error lower bound for DP mechanism synthesis.

        The minimax risk for answering d counting queries under ε-DP
        is known to be Θ(d/ε²) for L2 loss (Hardt & Talwar 2010).

        For finite output grids of size k, an additional discretisation
        lower bound applies: error ≥ 1/(12k²) per coordinate.

        Args:
            d: Number of dimensions.
            epsilon: Privacy parameter.
            delta: Approximate DP parameter.
            sensitivity: Query sensitivity.
            n_databases: Number of databases.
            k: Output grid size per coordinate.

        Returns:
            LowerBoundResult with the minimax bound.
        """
        if d < 1 or epsilon <= 0:
            raise ConfigurationError(
                "d must be >= 1 and epsilon must be > 0",
                parameter="d,epsilon",
            )
        # Privacy lower bound
        if self._loss_type == "L2":
            if delta == 0:
                # Pure DP: Ω(dΔ²/ε²)
                privacy_bound = d * sensitivity ** 2 / (epsilon ** 2)
            else:
                # Approximate DP: Ω(dΔ²·min(1/ε², 1/(ε√(log(1/δ)))))
                log_term = math.log(1.0 / max(delta, 1e-30))
                bound1 = d * sensitivity ** 2 / (epsilon ** 2)
                bound2 = d * sensitivity ** 2 / (epsilon * math.sqrt(log_term))
                privacy_bound = min(bound1, bound2)
        elif self._loss_type == "L1":
            privacy_bound = d * sensitivity / epsilon
        else:  # Linf
            privacy_bound = sensitivity / epsilon
        # Discretisation lower bound
        discretisation_bound = d / (12.0 * k ** 2) if self._loss_type == "L2" else 0.0
        bound = max(privacy_bound, discretisation_bound)
        # Constant factor (from tight analysis)
        # Known tight constant for Laplace: 2/ε² per coordinate
        tight_factor = 0.5 if self._loss_type == "L2" else 1.0
        bound *= tight_factor
        return LowerBoundResult(
            bound_value=bound,
            bound_type="minimax",
            parameters={
                "d": d,
                "epsilon": epsilon,
                "delta": delta,
                "sensitivity": sensitivity,
                "k": k,
                "loss_type": self._loss_type,
            },
            metadata={
                "privacy_bound": privacy_bound,
                "discretisation_bound": discretisation_bound,
            },
        )

    def gap_analysis(
        self,
        achieved_error: float,
        d: int,
        epsilon: float,
        delta: float = 0.0,
        sensitivity: float = 1.0,
        domain_size: int = 2,
        k: int = 100,
    ) -> LowerBoundResult:
        """Compute the optimality gap between achieved and lower bound.

        Runs both Fano–Assouad and minimax bounds and takes the
        tighter of the two.

        Args:
            achieved_error: Error of the synthesised mechanism.
            d: Number of dimensions.
            epsilon: Privacy parameter.
            delta: Approximate DP parameter.
            sensitivity: Query sensitivity.
            domain_size: Input domain size per coordinate.
            k: Output grid size.

        Returns:
            LowerBoundResult with gap_ratio = achieved/bound.
        """
        fano = self.fano_assouad(
            d=d, epsilon=epsilon, delta=delta,
            sensitivity=sensitivity, domain_size=domain_size,
        )
        minimax = self.minimax_bound(
            d=d, epsilon=epsilon, delta=delta,
            sensitivity=sensitivity, k=k,
        )
        best_bound = max(fano.bound_value, minimax.bound_value)
        bound_type = "fano_assouad" if fano.bound_value >= minimax.bound_value else "minimax"
        return LowerBoundResult(
            bound_value=best_bound,
            bound_type=f"gap_analysis({bound_type})",
            achieved_error=achieved_error,
            parameters={
                "d": d,
                "epsilon": epsilon,
                "delta": delta,
                "sensitivity": sensitivity,
                "loss_type": self._loss_type,
            },
            metadata={
                "fano_bound": fano.bound_value,
                "minimax_bound": minimax.bound_value,
            },
        )

    def per_coordinate_bounds(
        self,
        epsilons: Sequence[float],
        sensitivities: Sequence[float],
        domain_sizes: Optional[Sequence[int]] = None,
    ) -> List[LowerBoundResult]:
        """Compute per-coordinate minimax bounds.

        Useful for evaluating whether the budget allocation across
        coordinates is well-balanced.

        Args:
            epsilons: Per-coordinate ε values.
            sensitivities: Per-coordinate sensitivities.
            domain_sizes: Per-coordinate domain sizes.

        Returns:
            List of LowerBoundResult, one per coordinate.
        """
        d = len(epsilons)
        if len(sensitivities) != d:
            raise ValueError("epsilons and sensitivities must have same length")
        if domain_sizes is None:
            domain_sizes = [2] * d
        results = []
        for i in range(d):
            result = self.minimax_bound(
                d=1,
                epsilon=epsilons[i],
                sensitivity=sensitivities[i],
            )
            result.parameters["coordinate"] = i
            results.append(result)
        return results

    def composition_overhead(
        self,
        d: int,
        epsilon: float,
        composition_type: str = "basic",
    ) -> float:
        """Estimate the overhead factor from composition.

        Composition introduces slack: the composed mechanism's error
        exceeds what would be achievable with a joint mechanism. This
        method estimates the overhead ratio.

        Args:
            d: Number of coordinates.
            epsilon: Total privacy budget.
            composition_type: "basic", "advanced", or "rdp".

        Returns:
            Estimated overhead factor ≥ 1.
        """
        if composition_type == "basic":
            # Basic: each coordinate gets ε/d, error ∝ d²/ε²
            # Joint mechanism: error ∝ d/ε²
            # Overhead: d
            return float(d)
        elif composition_type == "advanced":
            # Advanced: each gets ~ε/√(d), error ∝ d²/ε²
            # Overhead: √d approximately
            return math.sqrt(d)
        elif composition_type == "rdp":
            # RDP: overhead is ~√d for small ε
            return math.sqrt(d)
        return float(d)

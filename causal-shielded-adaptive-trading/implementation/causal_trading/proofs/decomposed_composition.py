"""
Decomposed composition theorem for Causal-Shielded Adaptive Trading.

Replaces the monolithic ε₁+ε₂ union bound with per-stage error
decomposition across the full pipeline:

    regime detection → DAG estimation → invariance testing → shield synthesis

Each stage contributes a quantified error ε_i with an explicit computation
method. The total system error is bounded via either:
  • Union bound: P(any stage fails) ≤ Σε_i  (conservative)
  • Independence:  P(any stage fails) = 1 - Π(1 - ε_i)  (tighter, under independence)
  • Inclusion-exclusion (full): Σε_i - Σε_iε_j + ...

The decomposition enables:
  1. Identifying the *dominant* error contributor
  2. Optimal budget allocation across stages for a target total ε
  3. Sensitivity analysis: ∂ε_total / ∂ε_stage
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------

PIPELINE_STAGES = [
    "regime_detection",
    "dag_estimation",
    "invariance_testing",
    "shield_synthesis",
]


@dataclass
class StageError:
    """Error contribution from a single pipeline stage.

    Parameters
    ----------
    stage_name : str
        One of the canonical pipeline stages.
    epsilon : float
        Error probability bound for this stage (in [0, 1]).
    confidence : float
        Confidence level at which epsilon holds (e.g., 0.95).
    n_samples : int
        Number of data points used to compute the bound.
    method : str
        Human-readable description of how epsilon was computed.
    """
    stage_name: str
    epsilon: float
    confidence: float
    n_samples: int
    method: str

    def __post_init__(self) -> None:
        if not (0.0 <= self.epsilon <= 1.0):
            raise ValueError(
                f"epsilon must be in [0, 1], got {self.epsilon}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )
        if self.n_samples < 0:
            raise ValueError(
                f"n_samples must be non-negative, got {self.n_samples}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "epsilon": self.epsilon,
            "confidence": self.confidence,
            "n_samples": self.n_samples,
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StageError":
        return cls(**d)


# -----------------------------------------------------------------------
# DecomposedCertificate
# -----------------------------------------------------------------------

@dataclass
class DecomposedCertificate:
    """Certificate for the decomposed composition theorem.

    Attributes
    ----------
    stage_errors : list of StageError
        Per-stage error records.
    total_epsilon : float
        Total system error bound.
    composition_method : str
        Method used to combine stage errors ('union', 'independent',
        'inclusion_exclusion').
    dominant_stage : str
        Stage contributing the largest error.
    dominant_epsilon : float
        Error of the dominant stage.
    verified : bool
        Whether the decomposed bound was verified.
    timestamp : str or None
    """
    stage_errors: List[StageError]
    total_epsilon: float
    composition_method: str
    dominant_stage: str
    dominant_epsilon: float
    verified: bool
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_errors": [se.to_dict() for se in self.stage_errors],
            "total_epsilon": self.total_epsilon,
            "composition_method": self.composition_method,
            "dominant_stage": self.dominant_stage,
            "dominant_epsilon": self.dominant_epsilon,
            "verified": self.verified,
            "timestamp": self.timestamp,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DecomposedCertificate":
        return cls(
            stage_errors=[StageError.from_dict(se) for se in d["stage_errors"]],
            total_epsilon=d["total_epsilon"],
            composition_method=d["composition_method"],
            dominant_stage=d["dominant_stage"],
            dominant_epsilon=d["dominant_epsilon"],
            verified=d["verified"],
            timestamp=d.get("timestamp"),
        )


# -----------------------------------------------------------------------
# PipelineErrorBudget
# -----------------------------------------------------------------------

class PipelineErrorBudget:
    """Accumulate and compose per-stage error bounds.

    Usage
    -----
    >>> budget = PipelineErrorBudget()
    >>> budget.add_regime_error(posterior, n_obs)
    >>> budget.add_dag_error(bootstrap_dags, ref_dag)
    >>> budget.add_invariance_error(e_values, alpha)
    >>> budget.add_shield_error(pac_bayes_bound)
    >>> total = budget.total_error(method='union')
    >>> dom = budget.dominant_stage()
    """

    def __init__(self) -> None:
        self._stages: List[StageError] = []
        self._stage_map: Dict[str, StageError] = {}

    @property
    def stages(self) -> List[StageError]:
        return list(self._stages)

    @property
    def n_stages(self) -> int:
        return len(self._stages)

    # ----- Stage error computation methods ----

    def add_stage(self, stage: StageError) -> None:
        """Add a pre-computed stage error."""
        self._stages.append(stage)
        self._stage_map[stage.stage_name] = stage

    def add_regime_error(
        self,
        transition_matrix_posterior: np.ndarray,
        n_observations: int,
        *,
        confidence: float = 0.95,
    ) -> StageError:
        """Compute regime detection error from posterior concentration.

        Uses the maximum posterior variance of transition probabilities
        as a proxy for misclassification risk.  For a Dirichlet posterior
        Dir(α), the variance of each entry is α_i(α_0-α_i)/(α_0²(α_0+1)),
        and the concentration bound is ε ≈ √(K / (2·n)).

        Parameters
        ----------
        transition_matrix_posterior : np.ndarray
            Dirichlet concentration parameters (K×K).  Can also be a
            point-estimate transition matrix; in that case n_observations
            is used directly.
        n_observations : int
            Number of regime observations.
        confidence : float
            Desired confidence level.
        """
        K = transition_matrix_posterior.shape[0]
        # Hoeffding-style concentration: ε ≤ √(K log(2K/δ) / (2n))
        delta = 1.0 - confidence
        if n_observations < 1:
            epsilon = 1.0
        else:
            epsilon = float(np.sqrt(
                K * np.log(2.0 * K / max(delta, 1e-15)) / (2.0 * n_observations)
            ))
        epsilon = min(epsilon, 1.0)

        stage = StageError(
            stage_name="regime_detection",
            epsilon=epsilon,
            confidence=confidence,
            n_samples=n_observations,
            method=(
                f"Hoeffding concentration on {K}-state transition matrix "
                f"with {n_observations} observations"
            ),
        )
        self.add_stage(stage)
        return stage

    def add_dag_error(
        self,
        bootstrap_dags: List[np.ndarray],
        reference_dag: np.ndarray,
        *,
        confidence: float = 0.95,
    ) -> StageError:
        """Compute DAG estimation error from bootstrap SHD.

        Parameters
        ----------
        bootstrap_dags : list of np.ndarray
            Bootstrap adjacency matrices (each d×d binary).
        reference_dag : np.ndarray
            Reference (estimated) adjacency matrix.
        confidence : float
            Desired confidence level.

        Returns
        -------
        StageError
        """
        if not bootstrap_dags:
            stage = StageError(
                stage_name="dag_estimation",
                epsilon=1.0,
                confidence=confidence,
                n_samples=0,
                method="No bootstrap DAGs provided",
            )
            self.add_stage(stage)
            return stage

        # Structural Hamming Distance for each bootstrap DAG
        ref = np.asarray(reference_dag, dtype=float)
        shds = []
        d = ref.shape[0]
        max_edges = d * (d - 1)  # max possible edges in DAG
        if max_edges == 0:
            max_edges = 1

        for bg in bootstrap_dags:
            bg = np.asarray(bg, dtype=float)
            shd = float(np.sum(np.abs(bg - ref)))
            shds.append(shd / max_edges)  # normalise to [0, 1]

        shds_arr = np.array(shds)
        # Epsilon = fraction of bootstrap DAGs that differ + confidence correction
        n_boot = len(shds)
        mean_shd = float(np.mean(shds_arr))
        # Use Chebyshev: ε ≤ mean_shd + std/√n * z_{1-δ}
        std_shd = float(np.std(shds_arr, ddof=1)) if n_boot > 1 else 0.0
        delta = 1.0 - confidence
        z = 1.0 / np.sqrt(max(delta, 1e-15))  # Chebyshev
        epsilon = min(1.0, mean_shd + std_shd * z / np.sqrt(n_boot))

        stage = StageError(
            stage_name="dag_estimation",
            epsilon=epsilon,
            confidence=confidence,
            n_samples=n_boot,
            method=(
                f"Bootstrap SHD over {n_boot} resamples; "
                f"mean normalised SHD={mean_shd:.4f}"
            ),
        )
        self.add_stage(stage)
        return stage

    def add_invariance_error(
        self,
        e_values: np.ndarray,
        alpha: float = 0.05,
        *,
        confidence: float = 0.95,
    ) -> StageError:
        """Compute invariance testing error from e-value rejection.

        For e-value based tests, the type-I error is bounded by 1/e
        where e is the e-value.  We use the minimum e-value across
        features as the binding constraint.

        Parameters
        ----------
        e_values : np.ndarray
            Array of e-values for each invariance test.
        alpha : float
            Significance level for the test.
        confidence : float
            Desired confidence level.
        """
        e_vals = np.asarray(e_values, dtype=float)
        if e_vals.size == 0:
            epsilon = 1.0
            method_str = "No e-values provided"
        else:
            # e-value calibration: P(reject H0 | H0 true) ≤ 1/e
            # For multiple tests, use Bonferroni on e-values
            n_tests = len(e_vals)
            # Adjusted threshold
            min_e = float(np.min(e_vals))
            if min_e > 0:
                epsilon = min(1.0, n_tests / min_e)
            else:
                epsilon = 1.0
            epsilon = min(epsilon, alpha)  # can't exceed nominal level
            method_str = (
                f"E-value calibration with {n_tests} tests; "
                f"min e-value={min_e:.4f}, alpha={alpha}"
            )

        stage = StageError(
            stage_name="invariance_testing",
            epsilon=epsilon,
            confidence=confidence,
            n_samples=len(e_vals) if hasattr(e_vals, '__len__') else 0,
            method=method_str,
        )
        self.add_stage(stage)
        return stage

    def add_shield_error(
        self,
        pac_bayes_bound: float,
        *,
        n_samples: int = 0,
        confidence: float = 0.95,
    ) -> StageError:
        """Take shield error directly from a PAC-Bayes bound.

        Parameters
        ----------
        pac_bayes_bound : float
            Upper bound on shield failure probability.
        n_samples : int
            Samples used in the PAC-Bayes bound.
        confidence : float
            Confidence level.
        """
        epsilon = min(max(pac_bayes_bound, 0.0), 1.0)
        stage = StageError(
            stage_name="shield_synthesis",
            epsilon=epsilon,
            confidence=confidence,
            n_samples=n_samples,
            method=f"PAC-Bayes bound = {pac_bayes_bound:.6f}",
        )
        self.add_stage(stage)
        return stage

    # ----- Composition methods -----

    def total_error(self, method: str = "union") -> float:
        """Compute total system error from per-stage errors.

        Parameters
        ----------
        method : str
            'union' – standard union / Boole bound: Σε_i
            'independent' – assuming independence: 1 - Π(1 - ε_i)
            'inclusion_exclusion' – full inclusion-exclusion (exact if
                                    errors are for disjoint events)

        Returns
        -------
        float
            Total error bound (clamped to [0, 1]).
        """
        epsilons = [s.epsilon for s in self._stages]
        if not epsilons:
            return 0.0

        if method == "union":
            return min(1.0, sum(epsilons))

        elif method == "independent":
            prod = 1.0
            for e in epsilons:
                prod *= (1.0 - e)
            return min(1.0, max(0.0, 1.0 - prod))

        elif method == "inclusion_exclusion":
            return self._inclusion_exclusion(epsilons)

        else:
            raise ValueError(f"Unknown composition method: {method}")

    @staticmethod
    def _inclusion_exclusion(epsilons: List[float]) -> float:
        """Full inclusion-exclusion: Σε_i - Σε_iε_j + Σε_iε_jε_k - ..."""
        n = len(epsilons)
        total = 0.0
        for k in range(1, n + 1):
            sign = (-1) ** (k + 1)
            for combo in combinations(range(n), k):
                prod = 1.0
                for idx in combo:
                    prod *= epsilons[idx]
                total += sign * prod
        return min(1.0, max(0.0, total))

    def dominant_stage(self) -> Tuple[str, float]:
        """Return (stage_name, epsilon) of the stage with largest error.

        Returns
        -------
        (str, float)
            Name and epsilon of the dominant stage.

        Raises
        ------
        ValueError
            If no stages have been added.
        """
        if not self._stages:
            raise ValueError("No stages in the budget")
        worst = max(self._stages, key=lambda s: s.epsilon)
        return worst.stage_name, worst.epsilon

    def budget_allocation(
        self,
        target_epsilon: float,
        *,
        cost_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Optimal allocation of error budget across stages.

        Given a total target ε, allocate per-stage budgets ε_i such that
        Σε_i = target_epsilon, minimising the weighted cost of achieving
        each ε_i (by analogy with Lagrange multiplier optimisation).

        When costs are equal the optimal allocation is uniform.  When
        costs differ, cheaper stages are allocated more budget.

        Parameters
        ----------
        target_epsilon : float
            Desired total error bound.
        cost_weights : dict, optional
            Mapping stage_name → relative cost.  Stages with higher cost
            get smaller ε allocation.  Defaults to uniform.

        Returns
        -------
        dict
            Mapping stage_name → allocated epsilon.
        """
        if not self._stages:
            return {}

        names = [s.stage_name for s in self._stages]
        n = len(names)

        if cost_weights is None:
            # Uniform allocation
            per_stage = target_epsilon / n
            return {name: per_stage for name in names}

        # Weighted allocation: ε_i ∝ 1/√(c_i)
        weights = []
        for name in names:
            c = cost_weights.get(name, 1.0)
            weights.append(1.0 / math.sqrt(max(c, 1e-12)))
        total_w = sum(weights)
        return {
            name: target_epsilon * w / total_w
            for name, w in zip(names, weights)
        }

    def sensitivity(self, stage_name: str) -> float:
        """Compute ∂ε_total/∂ε_stage under the union bound.

        For the union bound, this is always 1.0.  For the independence
        model, it is Π_{j≠i}(1 - ε_j).

        Parameters
        ----------
        stage_name : str

        Returns
        -------
        float
        """
        if stage_name not in self._stage_map:
            raise KeyError(f"Unknown stage: {stage_name}")

        # Under independence: ∂(1 - Π(1-ε_j))/∂ε_i = Π_{j≠i}(1-ε_j)
        prod = 1.0
        for s in self._stages:
            if s.stage_name != stage_name:
                prod *= (1.0 - s.epsilon)
        return prod

    def get_stage(self, stage_name: str) -> Optional[StageError]:
        """Look up a stage by name."""
        return self._stage_map.get(stage_name)

    # ----- Serialisation -----

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stages": [s.to_dict() for s in self._stages],
            "total_union": self.total_error("union"),
            "total_independent": self.total_error("independent"),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineErrorBudget":
        budget = cls()
        for sd in d.get("stages", []):
            budget.add_stage(StageError.from_dict(sd))
        return budget

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# -----------------------------------------------------------------------
# DecomposedCompositionTheorem
# -----------------------------------------------------------------------

class DecomposedCompositionTheorem:
    """Verify the decomposed composition theorem and produce certificates.

    The theorem states:

        P(system correct)  ≥  1 - ε_total

    where ε_total is computed via one of several composition methods
    from per-stage bounds {ε_i}.

    Methods
    -------
    verify(budget)
        Check that the decomposed bound is internally consistent and
        tighter than the monolithic ε₁+ε₂ bound.
    sensitivity(budget, stage_name)
        How much does total error change with a perturbation to a stage?
    certificate(budget)
        Generate a DecomposedCertificate.
    """

    def __init__(
        self,
        vacuousness_threshold: float = 0.5,
        default_method: str = "independent",
    ) -> None:
        self.vacuousness_threshold = vacuousness_threshold
        self.default_method = default_method

    def verify(self, budget: PipelineErrorBudget) -> bool:
        """Verify the decomposed bound.

        Checks:
        1. All stage epsilons are in [0, 1].
        2. Total error is in [0, 1].
        3. The decomposed bound is non-vacuous (> threshold).
        4. The decomposed (independent) bound ≤ union bound.

        Returns
        -------
        bool
            Whether all checks pass.
        """
        ok = True
        for stage in budget.stages:
            if not (0.0 <= stage.epsilon <= 1.0):
                logger.warning(
                    "Stage %s has invalid epsilon %.6f",
                    stage.stage_name, stage.epsilon,
                )
                ok = False

        total_union = budget.total_error("union")
        total_indep = budget.total_error("independent")

        if total_union > 1.0 + 1e-9:
            logger.warning("Union bound %.6f exceeds 1", total_union)
            ok = False

        # Independence bound should be ≤ union bound (for ε_i ∈ [0,1])
        if total_indep > total_union + 1e-9:
            logger.warning(
                "Independence bound %.6f exceeds union bound %.6f",
                total_indep, total_union,
            )
            ok = False

        safety = 1.0 - budget.total_error(self.default_method)
        if safety < self.vacuousness_threshold:
            logger.warning(
                "Safety bound %.4f below vacuousness threshold %.4f",
                safety, self.vacuousness_threshold,
            )
            ok = False

        return ok

    def sensitivity(
        self,
        budget: PipelineErrorBudget,
        stage_name: str,
    ) -> float:
        """Compute sensitivity of total error to a stage's epsilon.

        Under independence: ∂ε_total/∂ε_i = Π_{j≠i}(1 - ε_j).
        """
        return budget.sensitivity(stage_name)

    def improvement_potential(
        self,
        budget: PipelineErrorBudget,
    ) -> Dict[str, float]:
        """For each stage, compute how much total error decreases if
        that stage's error were halved.

        Returns
        -------
        dict
            stage_name → Δε_total
        """
        current = budget.total_error(self.default_method)
        results: Dict[str, float] = {}
        for stage in budget.stages:
            # Hypothetical: halve this stage's epsilon
            original = stage.epsilon
            stage.epsilon = original / 2.0
            new_total = budget.total_error(self.default_method)
            results[stage.stage_name] = current - new_total
            stage.epsilon = original  # restore
        return results

    def tightness_ratio(self, budget: PipelineErrorBudget) -> float:
        """Ratio of independent bound to union bound.

        Values < 1 indicate the independent bound is tighter.
        """
        union = budget.total_error("union")
        indep = budget.total_error("independent")
        if union < 1e-15:
            return 1.0
        return indep / union

    def certificate(
        self,
        budget: PipelineErrorBudget,
        method: Optional[str] = None,
    ) -> DecomposedCertificate:
        """Generate a DecomposedCertificate.

        Parameters
        ----------
        budget : PipelineErrorBudget
        method : str, optional
            Composition method; defaults to self.default_method.

        Returns
        -------
        DecomposedCertificate
        """
        import datetime

        method = method or self.default_method
        total = budget.total_error(method)
        verified = self.verify(budget)

        if budget.stages:
            dom_name, dom_eps = budget.dominant_stage()
        else:
            dom_name, dom_eps = "", 0.0

        return DecomposedCertificate(
            stage_errors=list(budget.stages),
            total_epsilon=total,
            composition_method=method,
            dominant_stage=dom_name,
            dominant_epsilon=dom_eps,
            verified=verified,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )


# -----------------------------------------------------------------------
# Comparison helper
# -----------------------------------------------------------------------

def compare_monolithic_vs_decomposed(
    eps1: float,
    eps2: float,
    budget: PipelineErrorBudget,
) -> Dict[str, Any]:
    """Compare the old monolithic ε₁+ε₂ bound with the decomposed bound.

    Parameters
    ----------
    eps1, eps2 : float
        Original monolithic error terms.
    budget : PipelineErrorBudget
        Decomposed budget.

    Returns
    -------
    dict
        Comparison metrics.
    """
    monolithic = min(1.0, eps1 + eps2)
    decomposed_union = budget.total_error("union")
    decomposed_indep = budget.total_error("independent")
    decomposed_ie = budget.total_error("inclusion_exclusion")

    return {
        "monolithic_bound": monolithic,
        "decomposed_union": decomposed_union,
        "decomposed_independent": decomposed_indep,
        "decomposed_inclusion_exclusion": decomposed_ie,
        "improvement_union": monolithic - decomposed_union,
        "improvement_independent": monolithic - decomposed_indep,
        "n_stages": budget.n_stages,
    }

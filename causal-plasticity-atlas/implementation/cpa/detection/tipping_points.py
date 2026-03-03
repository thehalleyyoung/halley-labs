"""
Tipping-Point Detection via PELT (ALG4).

Detects tipping points in mechanism divergence sequences across ordered
causal contexts, using the PELT algorithm with permutation validation
and mechanism-level attribution.

Classes
-------
PELTDetector       — Full ALG4 pipeline.
SegmentAnalyzer    — Between-tipping-point segment characterization.
TippingPointReport — Human-readable reports.

Theory reference: ALG4 in the CPA specification.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from cpa.detection.changepoint import (
    ChangepointResult,
    CostFunction,
    L2Cost,
    PELTSolver,
    Segment,
    _pelt_core,
    _validate_signal,
    compute_penalty,
    compute_segment_statistics,
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class TippingPoint:
    """A single detected tipping point."""

    location: int            # Index in the context sequence
    p_value: float           # Permutation p-value
    fdr_adjusted_p: Optional[float] = None
    effect_size: float = 0.0  # Cohen's d
    effect_ci: Optional[tuple[float, float]] = None
    change_type: str = "unknown"  # "structural", "parametric", "both"
    attributed_mechanisms: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def is_significant(self) -> bool:
        """Check if p-value is significant (using FDR if available)."""
        p = self.fdr_adjusted_p if self.fdr_adjusted_p is not None else self.p_value
        return p < 0.05


@dataclass
class TippingPointResult:
    """Complete result from tipping-point detection."""

    tipping_points: list[TippingPoint]
    divergence_sequence: NDArray
    segments: list[Segment]
    n_contexts: int
    penalty: float
    method: str
    raw_changepoint_result: Optional[ChangepointResult] = None
    metadata: dict = field(default_factory=dict)

    @property
    def n_tipping_points(self) -> int:
        return len(self.tipping_points)

    @property
    def significant_tipping_points(self) -> list[TippingPoint]:
        """Return only significant tipping points."""
        return [tp for tp in self.tipping_points if tp.is_significant]

    def segment_boundaries(self) -> list[tuple[int, int]]:
        """Return (start, end) for each segment."""
        return [(s.start, s.end) for s in self.segments]


@dataclass
class SegmentDescription:
    """Characterization of a segment between tipping points."""

    start: int
    end: int
    n_contexts: int
    mean_divergence: float
    std_divergence: float
    homogeneity: float  # 1 - CV (coefficient of variation)
    dominant_structure: Optional[str] = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Divergence computation utilities
# ---------------------------------------------------------------------------

def _compute_pairwise_divergence_matrix(
    adjacencies: list[NDArray],
    datasets: list[NDArray],
    target_idx: Optional[int] = None,
    structural_weight: float = 0.5,
    parametric_weight: float = 0.5,
) -> NDArray:
    """Compute pairwise mechanism divergence matrix D[a][b].

    D[a][b] = w_s * structural_divergence(a, b) + w_p * parametric_divergence(a, b)

    structural_divergence = normalized Hamming distance of parent sets
    parametric_divergence = sqrt(JSD) between regression models

    Parameters
    ----------
    adjacencies : K adjacency matrices
    datasets : K data arrays
    target_idx : if given, compute for single variable; else average over all
    structural_weight : weight for structural divergence (default 0.5)
    parametric_weight : weight for parametric divergence (default 0.5)

    Returns
    -------
    (K, K) divergence matrix
    """
    K = len(adjacencies)
    n_vars = adjacencies[0].shape[0]
    D = np.zeros((K, K), dtype=np.float64)

    if target_idx is not None:
        variables = [target_idx]
    else:
        variables = list(range(n_vars))

    for a in range(K):
        for b in range(a + 1, K):
            struct_div = 0.0
            param_div = 0.0
            n_valid = 0

            for var in variables:
                # Structural divergence: normalized Hamming distance
                parents_a = set(
                    j for j in range(n_vars) if adjacencies[a][j, var] != 0
                )
                parents_b = set(
                    j for j in range(n_vars) if adjacencies[b][j, var] != 0
                )
                union_size = len(parents_a | parents_b)
                if union_size > 0:
                    hamming = len(parents_a.symmetric_difference(parents_b))
                    struct_div += hamming / union_size
                # else both empty → structural divergence = 0

                # Parametric divergence
                parents_a_list = sorted(parents_a)
                parents_b_list = sorted(parents_b)

                y_a = datasets[a][:, var]
                y_b = datasets[b][:, var]

                if parents_a_list == parents_b_list:
                    # Same structure: compare regression parameters
                    if len(parents_a_list) > 0:
                        X_a = datasets[a][:, parents_a_list]
                        X_b = datasets[b][:, parents_b_list]
                        coef_a, int_a, var_a = _fit_ols(X_a, y_a)
                        coef_b, int_b, var_b = _fit_ols(X_b, y_b)
                        jsd = _jsd_gaussian_regression(
                            coef_a, int_a, var_a,
                            coef_b, int_b, var_b,
                        )
                    else:
                        int_a = float(np.mean(y_a))
                        var_a = float(np.var(y_a, ddof=1)) if len(y_a) > 1 else 1e-10
                        int_b = float(np.mean(y_b))
                        var_b = float(np.var(y_b, ddof=1)) if len(y_b) > 1 else 1e-10
                        jsd = _jsd_gaussian(int_a, var_a, int_b, var_b)
                    param_div += math.sqrt(max(jsd, 0.0))
                else:
                    # Different structures: use marginal comparison
                    int_a = float(np.mean(y_a))
                    var_a = float(np.var(y_a, ddof=1)) if len(y_a) > 1 else 1e-10
                    int_b = float(np.mean(y_b))
                    var_b = float(np.var(y_b, ddof=1)) if len(y_b) > 1 else 1e-10
                    jsd = _jsd_gaussian(int_a, var_a, int_b, var_b)
                    param_div += math.sqrt(max(jsd, 0.0))

                n_valid += 1

            if n_valid > 0:
                struct_div /= n_valid
                param_div /= n_valid

            D[a, b] = structural_weight * struct_div + parametric_weight * param_div
            D[b, a] = D[a, b]

    return D


def _compute_consecutive_divergence_sequence(
    divergence_matrix: NDArray,
) -> NDArray:
    """Extract consecutive divergence sequence from pairwise matrix.

    d[k] = D[k][k+1] for k = 0, ..., K-2

    Parameters
    ----------
    divergence_matrix : (K, K) pairwise divergence matrix

    Returns
    -------
    (K-1,) consecutive divergence sequence
    """
    K = divergence_matrix.shape[0]
    if K < 2:
        return np.array([0.0])
    return np.array([divergence_matrix[k, k + 1] for k in range(K - 1)])


# ---------------------------------------------------------------------------
# Helper: OLS and JSD (local copies to avoid circular imports)
# ---------------------------------------------------------------------------

def _fit_ols(X: NDArray, y: NDArray) -> tuple[NDArray, float, float]:
    """Fit OLS regression y ~ X + intercept."""
    n = len(y)
    if X.ndim == 1:
        X = X[:, np.newaxis]
    p = X.shape[1]
    X_aug = np.column_stack([np.ones(n), X])
    try:
        beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(X_aug) @ y
    resid = y - X_aug @ beta
    df = max(n - p - 1, 1)
    res_var = float(np.sum(resid ** 2) / df)
    return beta[1:], float(beta[0]), max(res_var, 1e-15)


def _jsd_gaussian(mu1: float, var1: float, mu2: float, var2: float) -> float:
    """JSD between two univariate Gaussians."""
    if var1 < 1e-15 or var2 < 1e-15:
        return 0.0
    kl_12 = 0.5 * (math.log(var2 / var1) + var1 / var2 + (mu1 - mu2) ** 2 / var2 - 1)
    kl_21 = 0.5 * (math.log(var1 / var2) + var2 / var1 + (mu2 - mu1) ** 2 / var1 - 1)
    return max(0.5 * (kl_12 + kl_21), 0.0)


def _jsd_gaussian_regression(
    coefs_a: NDArray, int_a: float, var_a: float,
    coefs_b: NDArray, int_b: float, var_b: float,
) -> float:
    """JSD between two conditional Gaussians with same parent set."""
    if var_a < 1e-15 or var_b < 1e-15:
        return 0.0
    mean_diff = int_a - int_b
    coef_diff = np.asarray(coefs_a) - np.asarray(coefs_b)
    effective_sq = mean_diff ** 2 + float(np.sum(coef_diff ** 2))
    kl_ab = 0.5 * (math.log(var_b / var_a) + var_a / var_b + effective_sq / var_b - 1)
    kl_ba = 0.5 * (math.log(var_a / var_b) + var_b / var_a + effective_sq / var_a - 1)
    return max(0.5 * (kl_ab + kl_ba), 0.0)


# ---------------------------------------------------------------------------
# Cost function for mechanism divergence
# ---------------------------------------------------------------------------

class DivergenceCost(CostFunction):
    """Cost function using within-segment divergence variance.

    Cost = sum of (divergence - segment_mean)^2 + variance_penalty * segment_length
    """

    def __init__(self, signal: NDArray, variance_penalty: float = 0.0):
        self._variance_penalty = variance_penalty
        super().__init__(signal)

    def _precompute(self) -> None:
        self._cumsum = np.zeros((self._n + 1, self._d), dtype=np.float64)
        self._cumsum_sq = np.zeros((self._n + 1, self._d), dtype=np.float64)
        np.cumsum(self._signal, axis=0, out=self._cumsum[1:])
        np.cumsum(self._signal ** 2, axis=0, out=self._cumsum_sq[1:])

    def cost(self, start: int, end: int) -> float:
        if end <= start:
            return 0.0
        n = end - start
        seg_sum = self._cumsum[end] - self._cumsum[start]
        seg_sum_sq = self._cumsum_sq[end] - self._cumsum_sq[start]
        c = np.sum(seg_sum_sq - seg_sum ** 2 / n)
        # Add variance penalty
        c += self._variance_penalty * n
        return max(float(c), 0.0)


# ---------------------------------------------------------------------------
# PELTDetector — Full ALG4 Pipeline
# ---------------------------------------------------------------------------

class PELTDetector:
    """Tipping-Point Detection via PELT (ALG4).

    Detects tipping points in mechanism divergence sequences across
    ordered causal contexts.

    Steps:
        1. Mechanism divergence sequence computation
        2. PELT dynamic programming
        3. Backtracking for optimal segmentation
        4. Permutation validation
        5. Mechanism-level attribution
        6. Effect size computation

    Parameters
    ----------
    penalty_factor : float
        BIC-type penalty multiplier C (default 1.0).
        Final penalty = C * log(K).
    min_segment_length : int
        Minimum segment length (default 2).
    structural_weight : float
        Weight for structural divergence (default 0.5).
    parametric_weight : float
        Weight for parametric divergence (default 0.5).
    variance_penalty : float
        Within-segment variance penalty (default 0.0).
    n_permutations : int
        Permutation samples for validation (default 999).
    significance_level : float
        Significance level for tipping point validation (default 0.05).
    random_state : int or None
        Random seed.

    Examples
    --------
    >>> detector = PELTDetector(penalty_factor=1.5)
    >>> result = detector.detect(
    ...     adjacencies=[adj1, adj2, adj3, adj4, adj5],
    ...     datasets=[data1, data2, data3, data4, data5],
    ... )
    >>> for tp in result.significant_tipping_points:
    ...     print(f"Tipping point at context {tp.location}, p={tp.p_value:.3f}")
    """

    def __init__(
        self,
        penalty_factor: float = 1.0,
        min_segment_length: int = 2,
        structural_weight: float = 0.5,
        parametric_weight: float = 0.5,
        variance_penalty: float = 0.0,
        n_permutations: int = 999,
        significance_level: float = 0.05,
        random_state: Optional[int] = None,
    ):
        if structural_weight + parametric_weight <= 0:
            raise ValueError("Weights must sum to a positive value.")
        if min_segment_length < 1:
            raise ValueError("min_segment_length must be >= 1.")

        self.penalty_factor = penalty_factor
        self.min_segment_length = max(min_segment_length, 2)
        self.structural_weight = structural_weight
        self.parametric_weight = parametric_weight
        self.variance_penalty = variance_penalty
        self.n_permutations = n_permutations
        self.significance_level = significance_level
        self.random_state = random_state

    def detect(
        self,
        adjacencies: list[NDArray],
        datasets: list[NDArray],
        context_order: Optional[list[int]] = None,
        target_idx: Optional[int] = None,
    ) -> TippingPointResult:
        """Detect tipping points in the mechanism divergence sequence.

        Parameters
        ----------
        adjacencies : list of K adjacency matrices (must be ordered)
        datasets : list of K data arrays (same order)
        context_order : optional explicit ordering (default: 0, 1, ..., K-1)
        target_idx : if given, detect for a single variable; else global

        Returns
        -------
        TippingPointResult
        """
        K = len(adjacencies)
        self._validate_inputs(adjacencies, datasets, K)

        if context_order is not None:
            if sorted(context_order) != list(range(K)):
                raise ValueError("context_order must be a permutation of 0..K-1.")
            adjacencies = [adjacencies[i] for i in context_order]
            datasets = [datasets[i] for i in context_order]

        # Step 1: Mechanism Divergence Sequence
        div_matrix = _compute_pairwise_divergence_matrix(
            adjacencies, datasets, target_idx,
            self.structural_weight, self.parametric_weight,
        )
        div_seq = _compute_consecutive_divergence_sequence(div_matrix)

        if len(div_seq) < 2 * self.min_segment_length:
            return TippingPointResult(
                tipping_points=[],
                divergence_sequence=div_seq,
                segments=[Segment(start=0, end=len(div_seq), cost=0.0)],
                n_contexts=K,
                penalty=0.0,
                method="PELT",
                metadata={"warning": "Too few contexts for changepoint detection"},
            )

        # Step 2: PELT Dynamic Programming
        signal = div_seq.reshape(-1, 1)
        penalty = self.penalty_factor * math.log(max(K, 2))
        cost_fn = DivergenceCost(signal, self.variance_penalty)
        changepoints = _pelt_core(cost_fn, len(div_seq), penalty, self.min_segment_length)

        # Step 3: Backtracking — build segments
        boundaries = [0] + sorted(changepoints) + [len(div_seq)]
        segments = []
        for i in range(len(boundaries) - 1):
            s, e = boundaries[i], boundaries[i + 1]
            sc = cost_fn.cost(s, e)
            seg_data = div_seq[s:e]
            segments.append(Segment(
                start=s,
                end=e,
                cost=sc,
                mean=np.mean(seg_data) if e > s else 0.0,
                variance=float(np.var(seg_data)) if e - s > 1 else 0.0,
                count=e - s,
            ))

        # Step 4: Permutation Validation
        tipping_points = self._permutation_validation(
            div_seq, changepoints, K
        )

        # Step 5: Mechanism-Level Attribution
        self._attribute_mechanisms(
            tipping_points, adjacencies, datasets, div_matrix, target_idx
        )

        # Step 6: Effect Sizes
        self._compute_effect_sizes(tipping_points, div_seq)

        # Build raw result for reference
        raw_result = ChangepointResult(
            changepoints=changepoints,
            segments=segments,
            cost=sum(s.cost for s in segments),
            penalty=penalty,
            n_changepoints=len(changepoints),
            method="PELT",
        )

        return TippingPointResult(
            tipping_points=tipping_points,
            divergence_sequence=div_seq,
            segments=segments,
            n_contexts=K,
            penalty=penalty,
            method="PELT",
            raw_changepoint_result=raw_result,
            metadata={
                "structural_weight": self.structural_weight,
                "parametric_weight": self.parametric_weight,
                "penalty_factor": self.penalty_factor,
                "n_permutations": self.n_permutations,
            },
        )

    def detect_from_divergence(
        self,
        divergence_sequence: NDArray,
    ) -> TippingPointResult:
        """Detect tipping points from a precomputed divergence sequence.

        Parameters
        ----------
        divergence_sequence : (K-1,) array of consecutive divergences

        Returns
        -------
        TippingPointResult
        """
        div_seq = np.asarray(divergence_sequence, dtype=np.float64).ravel()
        n = len(div_seq)

        if n < 2 * self.min_segment_length:
            return TippingPointResult(
                tipping_points=[],
                divergence_sequence=div_seq,
                segments=[Segment(start=0, end=n, cost=0.0)],
                n_contexts=n + 1,
                penalty=0.0,
                method="PELT",
            )

        signal = div_seq.reshape(-1, 1)
        penalty = self.penalty_factor * math.log(max(n + 1, 2))
        cost_fn = DivergenceCost(signal, self.variance_penalty)
        changepoints = _pelt_core(cost_fn, n, penalty, self.min_segment_length)

        boundaries = [0] + sorted(changepoints) + [n]
        segments = []
        for i in range(len(boundaries) - 1):
            s, e = boundaries[i], boundaries[i + 1]
            sc = cost_fn.cost(s, e)
            seg_data = div_seq[s:e]
            segments.append(Segment(
                start=s,
                end=e,
                cost=sc,
                mean=np.mean(seg_data) if e > s else 0.0,
                variance=float(np.var(seg_data)) if e - s > 1 else 0.0,
                count=e - s,
            ))

        tipping_points = self._permutation_validation(
            div_seq, changepoints, n + 1
        )
        self._compute_effect_sizes(tipping_points, div_seq)

        return TippingPointResult(
            tipping_points=tipping_points,
            divergence_sequence=div_seq,
            segments=segments,
            n_contexts=n + 1,
            penalty=penalty,
            method="PELT",
        )

    # ---- Step 4: Permutation Validation ----

    def _permutation_validation(
        self,
        div_seq: NDArray,
        changepoints: list[int],
        K: int,
    ) -> list[TippingPoint]:
        """Validate changepoints via permutation testing.

        For each changepoint, tests whether the divergence change at
        that location is significant under the null of no ordering.

        Uses 999 permutation null samples per changepoint with
        Benjamini-Hochberg FDR correction.
        """
        if not changepoints:
            return []

        rng = np.random.default_rng(self.random_state)
        n = len(div_seq)
        tipping_points = []
        raw_p_values = []

        for cp in sorted(changepoints):
            # Observed statistic: absolute difference in segment means
            left = div_seq[:cp] if cp > 0 else np.array([0.0])
            right = div_seq[cp:] if cp < n else np.array([0.0])
            observed_stat = abs(np.mean(left) - np.mean(right)) if len(left) > 0 and len(right) > 0 else 0.0

            # Permutation null
            null_stats = np.zeros(self.n_permutations, dtype=np.float64)
            for b in range(self.n_permutations):
                perm = rng.permutation(n)
                perm_seq = div_seq[perm]
                perm_left = perm_seq[:cp]
                perm_right = perm_seq[cp:]
                if len(perm_left) > 0 and len(perm_right) > 0:
                    null_stats[b] = abs(np.mean(perm_left) - np.mean(perm_right))

            p_value = float((np.sum(null_stats >= observed_stat) + 1) / (self.n_permutations + 1))
            raw_p_values.append(p_value)

            tp = TippingPoint(
                location=cp,
                p_value=p_value,
                metadata={
                    "observed_statistic": observed_stat,
                    "null_mean": float(np.mean(null_stats)),
                    "null_std": float(np.std(null_stats)),
                },
            )
            tipping_points.append(tp)

        # Benjamini-Hochberg FDR correction
        if len(raw_p_values) > 1:
            adjusted = self._benjamini_hochberg(np.array(raw_p_values))
            for i, tp in enumerate(tipping_points):
                tp.fdr_adjusted_p = float(adjusted[i])
        elif len(raw_p_values) == 1:
            tipping_points[0].fdr_adjusted_p = tipping_points[0].p_value

        return tipping_points

    # ---- Step 5: Mechanism-Level Attribution ----

    def _attribute_mechanisms(
        self,
        tipping_points: list[TippingPoint],
        adjacencies: list[NDArray],
        datasets: list[NDArray],
        div_matrix: NDArray,
        target_idx: Optional[int],
    ) -> None:
        """Identify which mechanisms change most at each tipping point.

        For each tipping point at location cp:
          - Compare adjacencies[cp] vs adjacencies[cp+1]
          - Identify structural vs parametric changes
          - Compute per-mechanism effect sizes
        """
        K = len(adjacencies)
        n_vars = adjacencies[0].shape[0]

        variables = [target_idx] if target_idx is not None else list(range(n_vars))

        for tp in tipping_points:
            cp = tp.location
            if cp < 0 or cp >= K - 1:
                continue

            # Compare contexts cp and cp+1
            adj_before = adjacencies[cp]
            adj_after = adjacencies[cp + 1]
            data_before = datasets[cp]
            data_after = datasets[cp + 1]

            structural_changes = 0
            parametric_changes = 0
            mechanism_details = []

            for var in variables:
                parents_before = set(
                    j for j in range(n_vars) if adj_before[j, var] != 0
                )
                parents_after = set(
                    j for j in range(n_vars) if adj_after[j, var] != 0
                )

                struct_change = parents_before != parents_after
                if struct_change:
                    structural_changes += 1

                # Parametric comparison
                y_before = data_before[:, var]
                y_after = data_after[:, var]
                int_before = float(np.mean(y_before))
                var_before = max(float(np.var(y_before, ddof=1)) if len(y_before) > 1 else 1e-10, 1e-15)
                int_after = float(np.mean(y_after))
                var_after = max(float(np.var(y_after, ddof=1)) if len(y_after) > 1 else 1e-10, 1e-15)
                jsd = _jsd_gaussian(int_before, var_before, int_after, var_after)
                sqrt_jsd = math.sqrt(max(jsd, 0.0))

                if sqrt_jsd > 0.1:
                    parametric_changes += 1

                # Cohen's d for this mechanism
                pooled_std = math.sqrt((var_before + var_after) / 2)
                cohens_d = abs(int_before - int_after) / pooled_std if pooled_std > 1e-10 else 0.0

                mechanism_details.append({
                    "variable": var,
                    "structural_change": struct_change,
                    "parents_before": sorted(parents_before),
                    "parents_after": sorted(parents_after),
                    "sqrt_jsd": sqrt_jsd,
                    "cohens_d": cohens_d,
                })

            tp.attributed_mechanisms = mechanism_details

            # Classify change type
            if structural_changes > 0 and parametric_changes > 0:
                tp.change_type = "both"
            elif structural_changes > 0:
                tp.change_type = "structural"
            elif parametric_changes > 0:
                tp.change_type = "parametric"
            else:
                tp.change_type = "minor"

    # ---- Step 6: Effect Sizes ----

    def _compute_effect_sizes(
        self,
        tipping_points: list[TippingPoint],
        div_seq: NDArray,
    ) -> None:
        """Compute Cohen's d and CIs for each tipping point.

        Cohen's d = (mean_after - mean_before) / pooled_sd
        CIs via noncentral t approximation.
        """
        n = len(div_seq)

        for tp in tipping_points:
            cp = tp.location
            left = div_seq[:cp] if cp > 0 else np.array([0.0])
            right = div_seq[cp:] if cp < n else np.array([0.0])

            n_left = len(left)
            n_right = len(right)

            if n_left < 2 or n_right < 2:
                tp.effect_size = 0.0
                tp.effect_ci = (0.0, 0.0)
                continue

            mean_left = np.mean(left)
            mean_right = np.mean(right)
            var_left = np.var(left, ddof=1)
            var_right = np.var(right, ddof=1)

            pooled_var = (
                (n_left - 1) * var_left + (n_right - 1) * var_right
            ) / (n_left + n_right - 2)
            pooled_sd = math.sqrt(max(pooled_var, 1e-15))

            d = (mean_right - mean_left) / pooled_sd
            tp.effect_size = d

            # CI for Cohen's d (approximate)
            se_d = math.sqrt(
                (n_left + n_right) / (n_left * n_right)
                + d ** 2 / (2 * (n_left + n_right))
            )
            tp.effect_ci = (d - 1.96 * se_d, d + 1.96 * se_d)

    # ---- Validation ----

    def _validate_inputs(
        self,
        adjacencies: list[NDArray],
        datasets: list[NDArray],
        K: int,
    ) -> None:
        """Validate inputs for tipping-point detection."""
        if K < 3:
            raise ValueError(
                f"At least 3 ordered contexts required for tipping-point detection, got {K}."
            )
        if len(adjacencies) != len(datasets):
            raise ValueError("Number of adjacencies must match number of datasets.")

        n_vars = adjacencies[0].shape[0]
        for k in range(K):
            if adjacencies[k].shape != (n_vars, n_vars):
                raise ValueError(f"Adjacency {k} has wrong shape.")
            if datasets[k].shape[1] != n_vars:
                raise ValueError(f"Dataset {k} has wrong number of columns.")

    # ---- Utility ----

    @staticmethod
    def _benjamini_hochberg(p_values: NDArray) -> NDArray:
        """Benjamini-Hochberg FDR correction."""
        m = len(p_values)
        order = np.argsort(p_values)
        adjusted = np.zeros(m)
        cummin = 1.0
        for rank in range(m - 1, -1, -1):
            idx = order[rank]
            adjusted_val = p_values[idx] * m / (rank + 1)
            cummin = min(cummin, adjusted_val)
            adjusted[idx] = min(cummin, 1.0)
        return adjusted


# ---------------------------------------------------------------------------
# SegmentAnalyzer
# ---------------------------------------------------------------------------

class SegmentAnalyzer:
    """Characterize segments between tipping points.

    Provides within-segment homogeneity measures, segment comparisons,
    and structural characterization.

    Parameters
    ----------
    significance_level : float
        Significance level for homogeneity tests (default 0.05).
    """

    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level

    def characterize_segments(
        self,
        result: TippingPointResult,
        adjacencies: Optional[list[NDArray]] = None,
    ) -> list[SegmentDescription]:
        """Characterize all segments in a tipping-point result.

        Parameters
        ----------
        result : TippingPointResult
        adjacencies : optional adjacency matrices for structural characterization

        Returns
        -------
        list of SegmentDescription
        """
        descriptions = []
        div_seq = result.divergence_sequence

        for seg in result.segments:
            seg_data = div_seq[seg.start:seg.end]
            n = len(seg_data)

            if n == 0:
                descriptions.append(SegmentDescription(
                    start=seg.start,
                    end=seg.end,
                    n_contexts=0,
                    mean_divergence=0.0,
                    std_divergence=0.0,
                    homogeneity=1.0,
                ))
                continue

            mean_div = float(np.mean(seg_data))
            std_div = float(np.std(seg_data, ddof=1)) if n > 1 else 0.0

            # Homogeneity: 1 - CV (coefficient of variation)
            if abs(mean_div) > 1e-10:
                cv = std_div / abs(mean_div)
                homogeneity = max(0.0, 1.0 - cv)
            else:
                homogeneity = 1.0

            desc = SegmentDescription(
                start=seg.start,
                end=seg.end,
                n_contexts=n + 1,  # n divergences = n+1 contexts
                mean_divergence=mean_div,
                std_divergence=std_div,
                homogeneity=homogeneity,
            )

            # Structural characterization
            if adjacencies is not None:
                desc.dominant_structure = self._identify_dominant_structure(
                    adjacencies, seg.start, seg.end
                )

            descriptions.append(desc)

        return descriptions

    def within_segment_homogeneity(
        self,
        divergence_sequence: NDArray,
        segment: Segment,
    ) -> dict:
        """Compute detailed homogeneity measures for a segment.

        Returns dict with CV, range, IQR, trend statistics.
        """
        seg_data = divergence_sequence[segment.start:segment.end]
        n = len(seg_data)

        if n < 2:
            return {
                "cv": 0.0,
                "range": 0.0,
                "iqr": 0.0,
                "trend_slope": 0.0,
                "trend_r2": 0.0,
                "homogeneity_score": 1.0,
                "n_points": n,
            }

        mean_val = np.mean(seg_data)
        std_val = np.std(seg_data, ddof=1)
        cv = std_val / abs(mean_val) if abs(mean_val) > 1e-10 else 0.0

        # Trend detection via linear regression
        x = np.arange(n, dtype=np.float64)
        if np.std(x) > 0:
            slope = np.cov(x, seg_data)[0, 1] / np.var(x)
            y_pred = slope * x + (mean_val - slope * np.mean(x))
            ss_res = np.sum((seg_data - y_pred) ** 2)
            ss_tot = np.sum((seg_data - mean_val) ** 2)
            r2 = 1 - ss_res / max(ss_tot, 1e-15)
        else:
            slope = 0.0
            r2 = 0.0

        q25 = np.percentile(seg_data, 25)
        q75 = np.percentile(seg_data, 75)

        return {
            "cv": float(cv),
            "range": float(np.max(seg_data) - np.min(seg_data)),
            "iqr": float(q75 - q25),
            "trend_slope": float(slope),
            "trend_r2": float(max(r2, 0.0)),
            "homogeneity_score": max(0.0, 1.0 - cv),
            "n_points": n,
        }

    def compare_segments(
        self,
        divergence_sequence: NDArray,
        segment_a: Segment,
        segment_b: Segment,
    ) -> dict:
        """Compare two segments statistically.

        Returns dict with mean difference, effect size, Mann-Whitney U stat.
        """
        data_a = divergence_sequence[segment_a.start:segment_a.end]
        data_b = divergence_sequence[segment_b.start:segment_b.end]

        if len(data_a) < 2 or len(data_b) < 2:
            return {
                "mean_diff": 0.0,
                "cohens_d": 0.0,
                "mann_whitney_u": 0.0,
                "p_value": 1.0,
            }

        mean_a = np.mean(data_a)
        mean_b = np.mean(data_b)
        var_a = np.var(data_a, ddof=1)
        var_b = np.var(data_b, ddof=1)

        pooled_var = (
            (len(data_a) - 1) * var_a + (len(data_b) - 1) * var_b
        ) / (len(data_a) + len(data_b) - 2)
        pooled_sd = math.sqrt(max(pooled_var, 1e-15))

        d = (mean_b - mean_a) / pooled_sd if pooled_sd > 1e-10 else 0.0

        # Mann-Whitney U test (simplified)
        u_stat = 0.0
        for a in data_a:
            for b in data_b:
                if b > a:
                    u_stat += 1
                elif b == a:
                    u_stat += 0.5
        n1, n2 = len(data_a), len(data_b)
        expected_u = n1 * n2 / 2
        var_u = n1 * n2 * (n1 + n2 + 1) / 12
        z = (u_stat - expected_u) / math.sqrt(max(var_u, 1e-15))
        # Two-sided p-value from normal approximation
        from scipy.stats import norm
        p_value = 2 * (1 - norm.cdf(abs(z)))

        return {
            "mean_diff": float(mean_b - mean_a),
            "cohens_d": float(d),
            "mann_whitney_u": float(u_stat),
            "z_score": float(z),
            "p_value": float(p_value),
        }

    def _identify_dominant_structure(
        self,
        adjacencies: list[NDArray],
        start: int,
        end: int,
    ) -> str:
        """Identify the dominant DAG structure within a segment."""
        # Count edges across segment contexts
        n_vars = adjacencies[0].shape[0]
        edge_counts = np.zeros((n_vars, n_vars), dtype=np.float64)
        n_contexts = 0

        for k in range(start, min(end + 1, len(adjacencies))):
            edge_counts += (adjacencies[k] != 0).astype(np.float64)
            n_contexts += 1

        if n_contexts == 0:
            return "empty"

        # Count total and stable edges
        total_edges = np.sum(edge_counts > 0)
        stable_edges = np.sum(edge_counts == n_contexts)

        if total_edges == 0:
            return "empty_graph"
        stability_ratio = stable_edges / total_edges
        if stability_ratio > 0.8:
            return "stable"
        elif stability_ratio > 0.5:
            return "moderately_stable"
        else:
            return "variable"


# ---------------------------------------------------------------------------
# TippingPointReport
# ---------------------------------------------------------------------------

class TippingPointReport:
    """Generate human-readable tipping-point reports.

    Parameters
    ----------
    max_mechanisms : int
        Max mechanisms to show per tipping point (default 10).
    """

    def __init__(self, max_mechanisms: int = 10):
        self.max_mechanisms = max_mechanisms

    def generate(
        self,
        result: TippingPointResult,
        title: str = "Tipping-Point Detection Report",
        segment_descriptions: Optional[list[SegmentDescription]] = None,
    ) -> str:
        """Generate a full tipping-point report.

        Parameters
        ----------
        result : TippingPointResult
        title : report title
        segment_descriptions : optional segment characterizations

        Returns
        -------
        str : formatted report text
        """
        lines = []
        lines.append("=" * 72)
        lines.append(title.center(72))
        lines.append("=" * 72)
        lines.append("")

        # Overview
        lines.extend(self._overview_section(result))
        lines.append("")

        # Divergence sequence
        lines.extend(self._divergence_section(result))
        lines.append("")

        # Tipping points
        lines.extend(self._tipping_point_section(result))
        lines.append("")

        # Segments
        lines.extend(self._segment_section(result, segment_descriptions))
        lines.append("")

        # Attribution
        lines.extend(self._attribution_section(result))
        lines.append("")

        lines.append("=" * 72)
        lines.append("End of Report".center(72))
        lines.append("=" * 72)

        return "\n".join(lines)

    def visualization_data(
        self,
        result: TippingPointResult,
    ) -> dict:
        """Return data suitable for visualization.

        Returns dict with arrays for divergence sequence plots.
        """
        data = {
            "divergence_sequence": result.divergence_sequence.tolist(),
            "context_indices": list(range(len(result.divergence_sequence))),
            "tipping_point_locations": [tp.location for tp in result.tipping_points],
            "tipping_point_p_values": [tp.p_value for tp in result.tipping_points],
            "tipping_point_effects": [tp.effect_size for tp in result.tipping_points],
            "tipping_point_types": [tp.change_type for tp in result.tipping_points],
            "segment_boundaries": result.segment_boundaries(),
            "segment_means": [
                float(np.mean(result.divergence_sequence[s.start:s.end]))
                if s.end > s.start else 0.0
                for s in result.segments
            ],
        }

        # Significance markers
        data["significant_tipping_points"] = [
            tp.location for tp in result.significant_tipping_points
        ]

        return data

    def _overview_section(self, result: TippingPointResult) -> list[str]:
        lines = ["OVERVIEW", "-" * 40]
        lines.append(f"Number of contexts:       {result.n_contexts}")
        lines.append(f"Divergence sequence len:  {len(result.divergence_sequence)}")
        lines.append(f"Detected tipping points:  {result.n_tipping_points}")
        lines.append(f"Significant (FDR < 0.05): {len(result.significant_tipping_points)}")
        lines.append(f"Number of segments:       {len(result.segments)}")
        lines.append(f"Penalty used:             {result.penalty:.4f}")
        lines.append(f"Method:                   {result.method}")
        return lines

    def _divergence_section(self, result: TippingPointResult) -> list[str]:
        lines = ["DIVERGENCE SEQUENCE", "-" * 40]
        div_seq = result.divergence_sequence
        lines.append(f"  Mean:   {np.mean(div_seq):.4f}")
        lines.append(f"  Std:    {np.std(div_seq):.4f}")
        lines.append(f"  Min:    {np.min(div_seq):.4f}")
        lines.append(f"  Max:    {np.max(div_seq):.4f}")
        lines.append(f"  Range:  {np.max(div_seq) - np.min(div_seq):.4f}")

        # ASCII spark line
        n = len(div_seq)
        if n > 0:
            bars = "▁▂▃▄▅▆▇█"
            min_v = np.min(div_seq)
            max_v = np.max(div_seq)
            if max_v > min_v:
                norm = (div_seq - min_v) / (max_v - min_v)
                spark = "".join(bars[min(int(v * 7), 7)] for v in norm)
            else:
                spark = bars[4] * n
            lines.append(f"  Sequence: {spark}")

        return lines

    def _tipping_point_section(self, result: TippingPointResult) -> list[str]:
        lines = ["TIPPING POINTS", "-" * 40]
        if not result.tipping_points:
            lines.append("  No tipping points detected.")
            return lines

        for i, tp in enumerate(result.tipping_points):
            sig = "***" if tp.is_significant else ""
            fdr_str = (
                f"FDR-p={tp.fdr_adjusted_p:.4f}" if tp.fdr_adjusted_p is not None
                else f"p={tp.p_value:.4f}"
            )
            lines.append(
                f"  [{i+1}] Location: {tp.location}, {fdr_str}, "
                f"d={tp.effect_size:+.3f}, type={tp.change_type} {sig}"
            )
            if tp.effect_ci:
                lines.append(
                    f"       Effect CI: [{tp.effect_ci[0]:.3f}, {tp.effect_ci[1]:.3f}]"
                )

        return lines

    def _segment_section(
        self,
        result: TippingPointResult,
        descriptions: Optional[list[SegmentDescription]],
    ) -> list[str]:
        lines = ["SEGMENTS", "-" * 40]

        for i, seg in enumerate(result.segments):
            seg_data = result.divergence_sequence[seg.start:seg.end]
            mean_v = float(np.mean(seg_data)) if len(seg_data) > 0 else 0.0
            std_v = float(np.std(seg_data)) if len(seg_data) > 1 else 0.0

            lines.append(
                f"  Segment {i+1}: [{seg.start}, {seg.end}), "
                f"len={seg.length}, mean={mean_v:.4f}, std={std_v:.4f}"
            )

            if descriptions and i < len(descriptions):
                desc = descriptions[i]
                lines.append(
                    f"    Homogeneity: {desc.homogeneity:.3f}"
                    + (f", Structure: {desc.dominant_structure}" if desc.dominant_structure else "")
                )

        return lines

    def _attribution_section(self, result: TippingPointResult) -> list[str]:
        lines = ["MECHANISM ATTRIBUTION", "-" * 40]
        if not result.tipping_points:
            lines.append("  No attributions available.")
            return lines

        for i, tp in enumerate(result.tipping_points):
            lines.append(f"  Tipping Point [{i+1}] (loc={tp.location}, type={tp.change_type}):")
            if not tp.attributed_mechanisms:
                lines.append("    No mechanism details available.")
                continue

            # Sort by effect size
            sorted_mechs = sorted(
                tp.attributed_mechanisms,
                key=lambda m: m.get("cohens_d", 0),
                reverse=True,
            )

            for j, mech in enumerate(sorted_mechs[:self.max_mechanisms]):
                var_idx = mech["variable"]
                struct = "STRUCTURAL" if mech.get("structural_change") else "parametric"
                d = mech.get("cohens_d", 0)
                jsd = mech.get("sqrt_jsd", 0)
                lines.append(
                    f"    Var {var_idx}: {struct}, √JSD={jsd:.4f}, d={d:.3f}"
                )

            if len(sorted_mechs) > self.max_mechanisms:
                lines.append(
                    f"    ... and {len(sorted_mechs) - self.max_mechanisms} more mechanisms"
                )

        return lines

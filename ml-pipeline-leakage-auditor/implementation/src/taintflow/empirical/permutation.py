"""taintflow.empirical.permutation – Permutation tests for leakage detection.

Provides non-parametric hypothesis tests that assess whether the observed
mutual information between train features and test labels is significantly
larger than expected under the null hypothesis of independence (no leakage).

Key classes:
* :class:`PermutationLeakageTest` – basic permutation test.
* :class:`ConditionalPermutationTest` – leakage conditioned on pipeline stage.
* :class:`FeaturePermutationTest` – per-feature independence tests.
* :class:`StagePermutationTest` – per-stage leakage tests.
* :class:`EffectSizeEstimator` – Cohen's *d* for leakage magnitude.

Multiple-testing corrections (Bonferroni, Benjamini–Hochberg) are built in.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from taintflow.empirical.ksg import KSGEstimator, MutualInformationResult


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PermutationResult:
    """Result of a single permutation test.

    Attributes
    ----------
    p_value:
        Proportion of permuted statistics ≥ observed (upper-tail).
    test_statistic:
        Observed MI estimate.
    null_distribution:
        All permuted MI values (sorted).
    n_permutations:
        Number of permutations performed.
    feature_name:
        Optional label for the feature tested.
    stage_name:
        Optional label for the pipeline stage.
    """

    p_value: float
    test_statistic: float
    null_distribution: Tuple[float, ...] = ()
    n_permutations: int = 0
    feature_name: str = ""
    stage_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "p_value": self.p_value,
            "test_statistic": self.test_statistic,
            "null_distribution": list(self.null_distribution),
            "n_permutations": self.n_permutations,
            "feature_name": self.feature_name,
            "stage_name": self.stage_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PermutationResult:
        return cls(
            p_value=float(data["p_value"]),
            test_statistic=float(data["test_statistic"]),
            null_distribution=tuple(data.get("null_distribution", ())),
            n_permutations=int(data.get("n_permutations", 0)),
            feature_name=str(data.get("feature_name", "")),
            stage_name=str(data.get("stage_name", "")),
        )

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not 0.0 <= self.p_value <= 1.0:
            errors.append("p_value must be in [0, 1]")
        if self.n_permutations < 0:
            errors.append("n_permutations must be non-negative")
        return errors


@dataclass(frozen=True)
class CorrectedResult:
    """Collection of permutation results with multiple-testing correction.

    Attributes
    ----------
    results:
        Per-test results.
    corrected_p_values:
        Adjusted p-values (same order as *results*).
    rejected:
        Boolean per test indicating rejection at *alpha*.
    correction_method:
        ``"bonferroni"`` or ``"benjamini_hochberg"``.
    alpha:
        Family-wise / FDR significance level.
    """

    results: Tuple[PermutationResult, ...] = ()
    corrected_p_values: Tuple[float, ...] = ()
    rejected: Tuple[bool, ...] = ()
    correction_method: str = "bonferroni"
    alpha: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "corrected_p_values": list(self.corrected_p_values),
            "rejected": list(self.rejected),
            "correction_method": self.correction_method,
            "alpha": self.alpha,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CorrectedResult:
        return cls(
            results=tuple(
                PermutationResult.from_dict(r) for r in data.get("results", [])
            ),
            corrected_p_values=tuple(data.get("corrected_p_values", ())),
            rejected=tuple(data.get("rejected", ())),
            correction_method=str(data.get("correction_method", "bonferroni")),
            alpha=float(data.get("alpha", 0.05)),
        )

    def validate(self) -> List[str]:
        errors: List[str] = []
        if len(self.corrected_p_values) != len(self.results):
            errors.append(
                "corrected_p_values length must match results length"
            )
        if self.correction_method not in ("bonferroni", "benjamini_hochberg"):
            errors.append("Unknown correction_method")
        return errors


# ---------------------------------------------------------------------------
# Multiple-testing correction helpers
# ---------------------------------------------------------------------------

def _bonferroni(p_values: List[float], alpha: float) -> Tuple[List[float], List[bool]]:
    """Bonferroni correction: p_adj = min(p * m, 1)."""
    m = len(p_values)
    adjusted = [min(p * m, 1.0) for p in p_values]
    rejected = [p <= alpha for p in adjusted]
    return adjusted, rejected


def _benjamini_hochberg(
    p_values: List[float], alpha: float
) -> Tuple[List[float], List[bool]]:
    """Benjamini–Hochberg procedure controlling FDR at level *alpha*."""
    m = len(p_values)
    if m == 0:
        return [], []

    indexed = sorted(enumerate(p_values), key=lambda t: t[1])
    adjusted = [0.0] * m

    prev = 1.0
    for rank_rev, (orig_idx, p) in enumerate(reversed(indexed)):
        rank = m - rank_rev  # 1-based rank from bottom
        adj = min(p * m / rank, prev)
        adj = min(adj, 1.0)
        adjusted[orig_idx] = adj
        prev = adj

    rejected = [p <= alpha for p in adjusted]
    return adjusted, rejected


def _apply_correction(
    results: List[PermutationResult],
    method: str,
    alpha: float,
) -> CorrectedResult:
    """Apply a multiple-testing correction to a list of results."""
    p_values = [r.p_value for r in results]
    if method == "bonferroni":
        adj, rej = _bonferroni(p_values, alpha)
    elif method == "benjamini_hochberg":
        adj, rej = _benjamini_hochberg(p_values, alpha)
    else:
        raise ValueError(f"Unknown correction method: {method}")

    return CorrectedResult(
        results=tuple(results),
        corrected_p_values=tuple(adj),
        rejected=tuple(rej),
        correction_method=method,
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Core permutation engine
# ---------------------------------------------------------------------------

def _compute_p_value(observed: float, null: List[float]) -> float:
    """Upper-tail p-value: (# null >= observed + 1) / (B + 1).

    The "+1" accounts for the observed statistic itself (exact test).
    """
    b = len(null)
    count = sum(1 for v in null if v >= observed)
    return (count + 1) / (b + 1)


def _permuted_mi(
    x: List[List[float]],
    y: List[List[float]],
    estimator: KSGEstimator,
    rng: random.Random,
) -> float:
    """Compute MI on data with *y* rows randomly permuted."""
    n = len(y)
    perm = list(range(n))
    rng.shuffle(perm)
    y_perm = [y[p] for p in perm]
    return estimator.estimate(x, y_perm).estimate


# ---------------------------------------------------------------------------
# PermutationLeakageTest
# ---------------------------------------------------------------------------

class PermutationLeakageTest:
    """Permutation test for mutual-information–based leakage detection.

    Null hypothesis: train features X and test labels Y are independent
    (no leakage).  The test statistic is the KSG mutual-information
    estimate; the reference distribution is generated by randomly permuting
    the label vector.

    Parameters
    ----------
    n_permutations:
        Number of permutations for the null distribution.
    k:
        k-NN parameter forwarded to :class:`KSGEstimator`.
    alpha:
        Significance level.
    correction:
        Multiple-testing correction method when testing several features
        (``"bonferroni"`` or ``"benjamini_hochberg"``).
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_permutations: int = 1000,
        k: int = 3,
        alpha: float = 0.05,
        correction: str = "bonferroni",
        seed: Optional[int] = None,
    ) -> None:
        self.n_permutations = n_permutations
        self.k = k
        self.alpha = alpha
        self.correction = correction
        self._rng = random.Random(seed)
        self._estimator = KSGEstimator(k=k, seed=seed)

    def test(
        self,
        x: List[List[float]],
        y: List[List[float]],
    ) -> PermutationResult:
        """Run a two-sample permutation test on (X, Y).

        Returns
        -------
        PermutationResult
        """
        observed = self._estimator.estimate(x, y).estimate

        null_dist: List[float] = []
        for _ in range(self.n_permutations):
            null_dist.append(_permuted_mi(x, y, self._estimator, self._rng))

        p = _compute_p_value(observed, null_dist)
        null_dist.sort()

        return PermutationResult(
            p_value=p,
            test_statistic=observed,
            null_distribution=tuple(null_dist),
            n_permutations=self.n_permutations,
        )


# ---------------------------------------------------------------------------
# Conditional permutation test (conditioned on pipeline stage)
# ---------------------------------------------------------------------------

class ConditionalPermutationTest:
    """Test I(X; Y | Stage) by permuting within stage strata.

    For each pipeline stage, the test permutes only the rows belonging to
    that stage, preserving the stage structure in the null distribution.

    Parameters
    ----------
    n_permutations:
        Number of permutations.
    k:
        k-NN parameter.
    alpha:
        Significance level.
    seed:
        Random seed.
    """

    def __init__(
        self,
        n_permutations: int = 1000,
        k: int = 3,
        alpha: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        self.n_permutations = n_permutations
        self.k = k
        self.alpha = alpha
        self._rng = random.Random(seed)
        self._estimator = KSGEstimator(k=k, seed=seed)

    def test(
        self,
        x: List[List[float]],
        y: List[List[float]],
        stage_labels: List[str],
    ) -> PermutationResult:
        """Run a conditional permutation test.

        Parameters
        ----------
        x, y:
            Feature and target arrays.
        stage_labels:
            Per-sample pipeline-stage identifier.
        """
        observed = self._estimator.estimate(x, y).estimate

        # Build stage → index mapping
        stage_map: Dict[str, List[int]] = {}
        for i, s in enumerate(stage_labels):
            stage_map.setdefault(s, []).append(i)

        null_dist: List[float] = []
        for _ in range(self.n_permutations):
            y_perm = list(y)
            for indices in stage_map.values():
                perm = indices[:]
                self._rng.shuffle(perm)
                for orig, shuffled in zip(indices, perm):
                    y_perm[orig] = y[shuffled]
            null_dist.append(self._estimator.estimate(x, y_perm).estimate)

        null_dist.sort()
        p = _compute_p_value(observed, null_dist)

        return PermutationResult(
            p_value=p,
            test_statistic=observed,
            null_distribution=tuple(null_dist),
            n_permutations=self.n_permutations,
        )


# ---------------------------------------------------------------------------
# Feature permutation test (one test per feature)
# ---------------------------------------------------------------------------

class FeaturePermutationTest:
    """Test each feature independently for leakage.

    Applies a separate permutation test to every feature column and returns
    corrected p-values.

    Parameters
    ----------
    n_permutations:
        Permutations per feature.
    k:
        k-NN parameter.
    alpha:
        Significance level.
    correction:
        ``"bonferroni"`` or ``"benjamini_hochberg"``.
    seed:
        Random seed.
    """

    def __init__(
        self,
        n_permutations: int = 1000,
        k: int = 3,
        alpha: float = 0.05,
        correction: str = "bonferroni",
        seed: Optional[int] = None,
    ) -> None:
        self.n_permutations = n_permutations
        self.k = k
        self.alpha = alpha
        self.correction = correction
        self._rng = random.Random(seed)
        self._estimator = KSGEstimator(k=k, seed=seed)

    def test(
        self,
        features: List[List[List[float]]],
        target: List[List[float]],
        feature_names: Optional[List[str]] = None,
    ) -> CorrectedResult:
        """Test each feature array against *target*.

        Parameters
        ----------
        features:
            List of N × d_i feature arrays.
        target:
            N × d_y target array.
        feature_names:
            Optional human-readable labels.
        """
        n_features = len(features)
        names = feature_names or [f"feature_{i}" for i in range(n_features)]
        results: List[PermutationResult] = []

        for fi, feat in enumerate(features):
            observed = self._estimator.estimate(feat, target).estimate
            null_dist: List[float] = []
            for _ in range(self.n_permutations):
                null_dist.append(
                    _permuted_mi(feat, target, self._estimator, self._rng)
                )
            null_dist.sort()
            p = _compute_p_value(observed, null_dist)
            results.append(
                PermutationResult(
                    p_value=p,
                    test_statistic=observed,
                    null_distribution=tuple(null_dist),
                    n_permutations=self.n_permutations,
                    feature_name=names[fi],
                )
            )

        return _apply_correction(results, self.correction, self.alpha)


# ---------------------------------------------------------------------------
# Stage permutation test (one test per pipeline stage)
# ---------------------------------------------------------------------------

class StagePermutationTest:
    """Test each pipeline stage independently for leakage.

    Rows are grouped by *stage_labels* and MI is estimated on each group.

    Parameters
    ----------
    n_permutations:
        Permutations per stage.
    k:
        k-NN parameter.
    alpha:
        Significance level.
    correction:
        ``"bonferroni"`` or ``"benjamini_hochberg"``.
    seed:
        Random seed.
    """

    def __init__(
        self,
        n_permutations: int = 1000,
        k: int = 3,
        alpha: float = 0.05,
        correction: str = "bonferroni",
        seed: Optional[int] = None,
    ) -> None:
        self.n_permutations = n_permutations
        self.k = k
        self.alpha = alpha
        self.correction = correction
        self._rng = random.Random(seed)
        self._estimator = KSGEstimator(k=k, seed=seed)

    def test(
        self,
        x: List[List[float]],
        y: List[List[float]],
        stage_labels: List[str],
    ) -> CorrectedResult:
        """Run per-stage permutation tests.

        Parameters
        ----------
        x, y:
            Full data arrays.
        stage_labels:
            Per-sample stage identifiers.
        """
        stage_map: Dict[str, List[int]] = {}
        for i, s in enumerate(stage_labels):
            stage_map.setdefault(s, []).append(i)

        results: List[PermutationResult] = []
        for stage_name, indices in stage_map.items():
            if len(indices) < self.k + 2:
                results.append(
                    PermutationResult(
                        p_value=1.0,
                        test_statistic=0.0,
                        n_permutations=0,
                        stage_name=stage_name,
                    )
                )
                continue

            xs = [x[i] for i in indices]
            ys = [y[i] for i in indices]

            observed = self._estimator.estimate(xs, ys).estimate
            null_dist: List[float] = []
            for _ in range(self.n_permutations):
                null_dist.append(
                    _permuted_mi(xs, ys, self._estimator, self._rng)
                )
            null_dist.sort()
            p = _compute_p_value(observed, null_dist)
            results.append(
                PermutationResult(
                    p_value=p,
                    test_statistic=observed,
                    null_distribution=tuple(null_dist),
                    n_permutations=self.n_permutations,
                    stage_name=stage_name,
                )
            )

        return _apply_correction(results, self.correction, self.alpha)


# ---------------------------------------------------------------------------
# Power analysis
# ---------------------------------------------------------------------------

def estimate_required_permutations(
    desired_p: float = 0.05,
    power: float = 0.80,
    effect_size: float = 0.3,
    n_samples: int = 100,
    k: int = 3,
    seed: Optional[int] = None,
) -> int:
    """Estimate the number of permutations needed to achieve *power*.

    Uses the analytic formula for the power of a permutation test as a
    function of the number of permutations *B*, the significance level
    *desired_p*, and the expected effect (expressed as the MI shift in
    standard-deviation units of the null distribution).

    For a permutation test the minimum detectable p-value is 1/(B+1), so we
    need at least B ≥ 1/desired_p − 1.  For adequate power (P[reject | H1])
    we need the test-statistic to lie in the rejection region with
    probability ≥ *power*.  Under the normal approximation to the null
    distribution the required *B* is:

        B = ceil( (z_alpha + z_beta)^2 / (effect_size^2 * desired_p) )

    clamped to the floor of 1/desired_p − 1.

    Parameters
    ----------
    desired_p:
        Significance threshold.
    power:
        Target statistical power (1 − β).
    effect_size:
        Expected MI in units of the null's standard deviation.
    n_samples:
        Number of data samples.
    k:
        k-NN parameter (informational – affects null variance estimate).
    seed:
        Unused – present for API consistency.

    Returns
    -------
    int
        Recommended number of permutations.
    """
    if desired_p <= 0 or desired_p >= 1:
        raise ValueError("desired_p must be in (0, 1)")
    if power <= 0 or power >= 1:
        raise ValueError("power must be in (0, 1)")

    # Normal quantile via Beasley–Springer–Moro rational approximation
    z_alpha = _normal_quantile(1 - desired_p)
    z_beta = _normal_quantile(power)

    b_min = math.ceil(1.0 / desired_p) - 1
    b_power = math.ceil((z_alpha + z_beta) ** 2 / max(effect_size ** 2 * desired_p, 1e-12))
    return max(b_min, b_power, 100)


def _normal_quantile(p: float) -> float:
    """Inverse standard-normal CDF (Beasley–Springer–Moro algorithm).

    Accurate to ~1e-9 for p in (1e-8, 1 − 1e-8).
    """
    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0, 1)")

    # Rational approximation coefficients (Abramowitz & Stegun 26.2.23)
    a = [
        -3.969683028665376e01,
         2.209460984245205e02,
        -2.759285104469687e02,
         1.383577518672690e02,
        -3.066479806614716e01,
         2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
         1.615858368580409e02,
        -1.556989798598866e02,
         6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
         4.374664141464968e00,
         2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


# ---------------------------------------------------------------------------
# Effect size estimator
# ---------------------------------------------------------------------------

class EffectSizeEstimator:
    """Estimate the effect size of observed leakage.

    Computes Cohen's *d* : the observed MI minus the null mean, divided by
    the null standard deviation.  Also provides a "leakage bits" conversion
    (MI in nats → bits via division by ln 2).

    Parameters
    ----------
    n_permutations:
        Number of permutations used to build the null distribution.
    k:
        k-NN parameter.
    seed:
        Random seed.
    """

    def __init__(
        self,
        n_permutations: int = 1000,
        k: int = 3,
        seed: Optional[int] = None,
    ) -> None:
        self.n_permutations = n_permutations
        self.k = k
        self._rng = random.Random(seed)
        self._estimator = KSGEstimator(k=k, seed=seed)

    def cohens_d(
        self,
        x: List[List[float]],
        y: List[List[float]],
    ) -> Dict[str, float]:
        """Compute Cohen's *d* for the MI leakage signal.

        Returns
        -------
        dict
            Keys: ``"cohens_d"``, ``"observed_mi"``, ``"null_mean"``,
            ``"null_std"``, ``"leakage_bits"``.
        """
        observed = self._estimator.estimate(x, y).estimate

        null_vals: List[float] = []
        for _ in range(self.n_permutations):
            null_vals.append(
                _permuted_mi(x, y, self._estimator, self._rng)
            )

        null_mean = sum(null_vals) / len(null_vals)
        null_var = sum((v - null_mean) ** 2 for v in null_vals) / max(len(null_vals) - 1, 1)
        null_std = math.sqrt(null_var)

        d = (observed - null_mean) / null_std if null_std > 0 else 0.0

        return {
            "cohens_d": d,
            "observed_mi": observed,
            "null_mean": null_mean,
            "null_std": null_std,
            "leakage_bits": observed / math.log(2) if observed > 0 else 0.0,
        }

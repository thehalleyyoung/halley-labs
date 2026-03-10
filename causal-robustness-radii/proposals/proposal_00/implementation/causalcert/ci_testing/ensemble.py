"""
Cauchy combination test for ensemble CI testing (ALG 6).

Aggregates p-values from multiple heterogeneous CI tests using the Cauchy
combination (Liu & Xie, 2020), which is valid under arbitrary dependence
between the constituent tests.

Key features:
- Exact Cauchy combination formula: T = sum(w_i * tan((0.5 - p_i) * pi))
- p-value from the standard Cauchy CDF
- Weighted and unweighted variants
- Degenerate case handling (p=0, p=1)
- Adaptive weight selection based on sample size and variable types
- Integration with CI test caching
"""

from __future__ import annotations

import warnings
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from causalcert.ci_testing.base import (
    BaseCITest,
    _validate_inputs,
)
from causalcert.types import CITestMethod, CITestResult, NodeId, NodeSet

_EPS = 1e-15
_P_FLOOR = 1e-15  # Minimum p-value to avoid tan(pi/2)
_P_CEIL = 1.0 - 1e-15  # Maximum p-value to avoid tan(-pi/2)


# ---------------------------------------------------------------------------
# Cauchy combination core
# ---------------------------------------------------------------------------


def cauchy_combine_pvalues(
    p_values: Sequence[float],
    weights: Sequence[float],
) -> float:
    """Combine p-values via the Cauchy combination (Liu & Xie 2020).

    The Cauchy combination statistic is:

        T = sum_{i=1}^{K} w_i * tan((0.5 - p_i) * pi)

    Under the null hypothesis, T follows a standard Cauchy distribution
    (when weights sum to 1 and individual tests are valid), regardless of
    the dependence structure among the p-values.

    The combined p-value is:

        p_combined = 1/2 - arctan(T) / pi

    Parameters
    ----------
    p_values : Sequence[float]
        Individual p-values from K tests.
    weights : Sequence[float]
        Non-negative weights that must sum to 1.

    Returns
    -------
    float
        Combined p-value in ``(0, 1)``.

    References
    ----------
    Liu, Y. & Xie, J. (2020). Cauchy combination test: a powerful test with
    analytic p-value calculation under arbitrary dependency structures.
    *Journal of the American Statistical Association*, 115(529), 393–402.
    """
    p_arr = np.asarray(p_values, dtype=np.float64)
    w_arr = np.asarray(weights, dtype=np.float64)

    if len(p_arr) == 0:
        return 1.0

    if len(p_arr) != len(w_arr):
        raise ValueError("p_values and weights must have the same length.")

    # Validate weights
    if np.any(w_arr < 0):
        raise ValueError("Weights must be non-negative.")
    w_sum = np.sum(w_arr)
    if w_sum < _EPS:
        raise ValueError("Weights must not all be zero.")
    # Normalise
    w_arr = w_arr / w_sum

    # Clamp p-values to avoid infinities in tan
    p_arr = np.clip(p_arr, _P_FLOOR, _P_CEIL)

    # Cauchy combination: T = sum(w_i * tan((0.5 - p_i) * pi))
    T = np.sum(w_arr * np.tan((0.5 - p_arr) * np.pi))

    # Combined p-value from Cauchy CDF
    # P(Cauchy > T) = 1/2 - arctan(T) / pi
    p_combined = 0.5 - np.arctan(T) / np.pi

    return float(np.clip(p_combined, _P_FLOOR, _P_CEIL))


def cauchy_combine_unweighted(p_values: Sequence[float]) -> float:
    """Cauchy combination with equal weights.

    Parameters
    ----------
    p_values : Sequence[float]
        Individual p-values.

    Returns
    -------
    float
        Combined p-value.
    """
    n = len(p_values)
    if n == 0:
        return 1.0
    weights = [1.0 / n] * n
    return cauchy_combine_pvalues(p_values, weights)


# ---------------------------------------------------------------------------
# Adaptive weight selection
# ---------------------------------------------------------------------------


def _adaptive_weights(
    n_samples: int,
    n_conditioning: int,
    methods: Sequence[CITestMethod],
) -> list[float]:
    """Select weights adaptively based on sample size and test properties.

    Heuristic assignment:
    - Small samples (n < 100): upweight partial correlation, downweight KCI/CRT
    - Medium samples (100 ≤ n < 1000): roughly equal
    - Large samples (n ≥ 1000): upweight KCI and CRT
    - Large conditioning sets: upweight CRT, downweight partial correlation

    Parameters
    ----------
    n_samples : int
        Sample size.
    n_conditioning : int
        Size of the conditioning set.
    methods : Sequence[CITestMethod]
        CI test methods being combined.

    Returns
    -------
    list[float]
        Normalised weights.
    """
    K = len(methods)
    if K == 0:
        return []

    weights = np.ones(K, dtype=np.float64)

    for i, m in enumerate(methods):
        if m == CITestMethod.PARTIAL_CORRELATION:
            # Partial correlation: good for small n, linear, low-dimensional
            if n_samples < 100:
                weights[i] = 2.0
            elif n_conditioning > max(n_samples * 0.3, 10):
                weights[i] = 0.5  # High-dimensional conditioning is bad
            else:
                weights[i] = 1.0

        elif m == CITestMethod.KERNEL:
            # KCI: good for nonlinear, needs moderate n
            if n_samples < 50:
                weights[i] = 0.3
            elif n_samples >= 500:
                weights[i] = 2.0
            else:
                weights[i] = 1.0

        elif m == CITestMethod.RANK:
            # Rank: robust, moderate power
            weights[i] = 1.0

        elif m == CITestMethod.CRT:
            # CRT: good with large conditioning, needs many samples
            if n_samples < 100:
                weights[i] = 0.5
            elif n_conditioning > 5:
                weights[i] = 1.5
            else:
                weights[i] = 1.0

        elif m == CITestMethod.ENSEMBLE:
            # Nested ensemble (unusual but handle gracefully)
            weights[i] = 1.0

    # Normalise
    total = np.sum(weights)
    if total < _EPS:
        return [1.0 / K] * K
    return (weights / total).tolist()


# ---------------------------------------------------------------------------
# Ensemble configuration
# ---------------------------------------------------------------------------


class EnsembleConfig:
    """Configuration for which tests to include in the Cauchy ensemble.

    Parameters
    ----------
    include_partial_corr : bool
        Include Fisher-z partial correlation test.
    include_kernel : bool
        Include KCI test.
    include_rank : bool
        Include rank-based CI test.
    include_crt : bool
        Include conditional randomization test.
    adaptive_weights : bool
        Use adaptive weight selection based on sample size.
    """

    def __init__(
        self,
        include_partial_corr: bool = True,
        include_kernel: bool = True,
        include_rank: bool = True,
        include_crt: bool = False,
        adaptive_weights: bool = True,
    ) -> None:
        self.include_partial_corr = include_partial_corr
        self.include_kernel = include_kernel
        self.include_rank = include_rank
        self.include_crt = include_crt
        self.adaptive_weights = adaptive_weights

    def active_methods(self) -> list[CITestMethod]:
        """Return the list of active CI test methods."""
        methods = []
        if self.include_partial_corr:
            methods.append(CITestMethod.PARTIAL_CORRELATION)
        if self.include_kernel:
            methods.append(CITestMethod.KERNEL)
        if self.include_rank:
            methods.append(CITestMethod.RANK)
        if self.include_crt:
            methods.append(CITestMethod.CRT)
        return methods


# ---------------------------------------------------------------------------
# Main CauchyCombinationTest class
# ---------------------------------------------------------------------------


class CauchyCombinationTest(BaseCITest):
    """Cauchy combination aggregation of multiple CI tests (ALG 6).

    Given a collection of base CI testers, runs each on the same triple
    and combines their p-values via the Cauchy combination, which requires
    no independence assumption between tests.

    Parameters
    ----------
    base_tests : Sequence[BaseCITest]
        Constituent CI tests to aggregate.
    weights : Sequence[float] | None
        Non-negative weights for each test.  ``None`` ⇒ equal weights.
    alpha : float
        Significance level.
    adaptive : bool
        If ``True`` and *weights* is ``None``, select weights adaptively
        based on sample size and conditioning-set dimension.
    cache : dict | None
        Optional shared cache for CI test results.
    seed : int
        Random seed.
    """

    method = CITestMethod.ENSEMBLE

    def __init__(
        self,
        base_tests: Sequence[BaseCITest],
        weights: Sequence[float] | None = None,
        alpha: float = 0.05,
        adaptive: bool = True,
        cache: dict[tuple, CITestResult] | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(alpha=alpha, seed=seed)
        self.base_tests = list(base_tests)
        self.adaptive = adaptive
        self._cache = cache or {}

        if len(self.base_tests) == 0:
            raise ValueError("At least one base test is required.")

        if weights is None:
            self._fixed_weights: list[float] | None = None
        else:
            if len(weights) != len(self.base_tests):
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match "
                    f"number of base tests ({len(self.base_tests)})."
                )
            total = sum(weights)
            if total < _EPS:
                raise ValueError("Weights must not all be zero.")
            self._fixed_weights = [w / total for w in weights]

    @property
    def weights(self) -> list[float]:
        """Current weights (fixed or default equal)."""
        if self._fixed_weights is not None:
            return self._fixed_weights
        K = len(self.base_tests)
        return [1.0 / K] * K

    def _get_weights(self, n: int, k: int) -> list[float]:
        """Get weights, optionally adapting to sample size.

        Parameters
        ----------
        n : int
            Sample size.
        k : int
            Conditioning-set size.

        Returns
        -------
        list[float]
            Normalised weights.
        """
        if self._fixed_weights is not None:
            return self._fixed_weights
        if self.adaptive:
            methods = [t.method for t in self.base_tests]
            return _adaptive_weights(n, k, methods)
        K = len(self.base_tests)
        return [1.0 / K] * K

    def test(
        self,
        x: NodeId,
        y: NodeId,
        conditioning_set: NodeSet,
        data: pd.DataFrame,
    ) -> CITestResult:
        """Run all base tests and combine via Cauchy combination.

        Parameters
        ----------
        x, y : NodeId
            Variables to test.
        conditioning_set : NodeSet
            Conditioning variables.
        data : pd.DataFrame
            Observational data.

        Returns
        -------
        CITestResult
            Combined result with the Cauchy-combination p-value.
        """
        _validate_inputs(data, x, y, conditioning_set)

        n = len(data)
        k = len(conditioning_set)

        # Run each base test (using cache if available)
        sub_results: list[CITestResult] = []
        for bt in self.base_tests:
            cache_key = (x, y, conditioning_set, bt.method)
            cached = self._cache.get(cache_key)
            if cached is not None:
                sub_results.append(cached)
            else:
                try:
                    result = bt.test(x, y, conditioning_set, data)
                except Exception as exc:
                    warnings.warn(
                        f"Base test {bt.method.value} failed: {exc}. "
                        "Assigning p=1.0 (conservative).",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    result = CITestResult(
                        x=x, y=y,
                        conditioning_set=conditioning_set,
                        statistic=0.0, p_value=1.0,
                        method=bt.method, reject=False, alpha=bt.alpha,
                    )
                self._cache[cache_key] = result
                sub_results.append(result)

        # Extract p-values
        p_values = [r.p_value for r in sub_results]
        w = self._get_weights(n, k)

        # Combine
        combined_p = self.cauchy_combine(p_values, w)

        # Compute the Cauchy statistic for reporting
        p_arr = np.clip(p_values, _P_FLOOR, _P_CEIL)
        w_arr = np.asarray(w)
        T = float(np.sum(w_arr * np.tan((0.5 - p_arr) * np.pi)))

        return self._make_result(x, y, conditioning_set, T, combined_p)

    @staticmethod
    def cauchy_combine(
        p_values: Sequence[float],
        weights: Sequence[float],
    ) -> float:
        """Combine p-values via the Cauchy combination.

        Parameters
        ----------
        p_values : Sequence[float]
            Individual p-values.
        weights : Sequence[float]
            Non-negative weights (must sum to 1).

        Returns
        -------
        float
            Combined p-value.
        """
        return cauchy_combine_pvalues(p_values, weights)

    def test_batch(
        self,
        triples: list[tuple[NodeId, NodeId, NodeSet]],
        data: pd.DataFrame,
    ) -> list[CITestResult]:
        """Batch test with shared caching across triples.

        Parameters
        ----------
        triples : list[tuple[NodeId, NodeId, NodeSet]]
            CI test triples.
        data : pd.DataFrame
            Observational data.

        Returns
        -------
        list[CITestResult]
        """
        return [self.test(x, y, s, data) for x, y, s in triples]

    @property
    def cache_size(self) -> int:
        """Number of cached sub-test results."""
        return len(self._cache)

    def clear_cache(self) -> None:
        """Clear the sub-test cache."""
        self._cache.clear()

    def __repr__(self) -> str:
        tests = ", ".join(t.method.value for t in self.base_tests)
        return (
            f"CauchyCombinationTest(tests=[{tests}], "
            f"adaptive={self.adaptive}, alpha={self.alpha})"
        )

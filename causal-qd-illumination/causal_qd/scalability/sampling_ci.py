"""Sub-sampling-based conditional independence test for large datasets.

Provides:
  - SamplingCI: run CI tests on random sub-samples and aggregate
  - Monte Carlo CI test approximation
  - Importance-sampling based CI for rare conditioning events
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, FrozenSet, List, Optional

import numpy as np
from scipy import stats as sp_stats

from causal_qd.types import DataMatrix, PValue

if TYPE_CHECKING:
    from causal_qd.ci_tests.ci_base import CITest, CITestResult

logger = logging.getLogger(__name__)


class SamplingCI:
    """Run CI tests on random sub-samples and aggregate p-values.

    Reduces the cost of CI testing on very large datasets by
    evaluating the base test on several smaller random sub-samples
    and combining p-values using Fisher's method or the median.

    Parameters
    ----------
    base_test :
        A CI test exposing ``test(x, y, conditioning_set, data, alpha)``.
    sample_fraction :
        Fraction of the data to use per sub-sample (in ``(0, 1]``).
    n_repeats :
        Number of sub-sample evaluations to aggregate.
    aggregation :
        Method to aggregate p-values: ``"median"`` or ``"fisher"``.
    """

    def __init__(
        self,
        base_test: CITest,
        sample_fraction: float = 0.5,
        n_repeats: int = 5,
        aggregation: str = "median",
    ) -> None:
        if not 0.0 < sample_fraction <= 1.0:
            raise ValueError("sample_fraction must be in (0, 1]")
        self._base = base_test
        self._frac = sample_fraction
        self._n_repeats = max(1, n_repeats)
        self._aggregation = aggregation
        self._n_tests: int = 0

    def test(
        self,
        x: int,
        y: int,
        conditioning_set: FrozenSet[int],
        data: DataMatrix,
        alpha: float = 0.05,
        rng: Optional[np.random.Generator] = None,
    ) -> PValue:
        """Aggregated CI test via sub-sampling.

        Parameters
        ----------
        x, y :
            Indices of the two variables.
        conditioning_set :
            Set of conditioning variable indices.
        data :
            ``N × p`` data matrix.
        alpha :
            Significance level.
        rng :
            Optional random generator.

        Returns
        -------
        PValue
            Aggregated p-value.
        """
        if rng is None:
            rng = np.random.default_rng()

        n = data.shape[0]
        sample_size = max(2 + len(conditioning_set) + 2, int(n * self._frac))
        sample_size = min(sample_size, n)

        pvalues: List[float] = []
        for _ in range(self._n_repeats):
            indices = rng.choice(n, size=sample_size, replace=False)
            result = self._base.test(x, y, conditioning_set, data[indices], alpha)
            pvalues.append(result.p_value)
            self._n_tests += 1

        return self._aggregate(pvalues)

    def _aggregate(self, pvalues: List[float]) -> PValue:
        """Aggregate a list of p-values."""
        if self._aggregation == "fisher":
            return self._fisher_combine(pvalues)
        return float(np.median(pvalues))

    @staticmethod
    def _fisher_combine(pvalues: List[float]) -> PValue:
        """Combine p-values using Fisher's method.

        The test statistic is ``-2 * sum(log(p_i))`` which follows
        a chi-squared distribution with ``2k`` degrees of freedom.
        """
        # Clamp p-values to avoid log(0)
        clamped = [max(p, 1e-300) for p in pvalues]
        chi2_stat = -2.0 * sum(math.log(p) for p in clamped)
        dof = 2 * len(clamped)
        combined_p = float(1.0 - sp_stats.chi2.cdf(chi2_stat, dof))
        return combined_p

    @property
    def n_tests(self) -> int:
        """Total number of base CI tests performed."""
        return self._n_tests


class MonteCarloCITest:
    """Monte Carlo approximation of CI statistics.

    For very large conditioning sets, exact computation of partial
    correlations can be expensive.  This class approximates the
    CI test statistic using random projections of the conditioning
    set.

    Parameters
    ----------
    n_projections :
        Number of random projections to average over.
    projection_dim :
        Dimensionality of each random projection of the conditioning set.
    """

    def __init__(
        self,
        n_projections: int = 10,
        projection_dim: int = 3,
    ) -> None:
        self._n_projections = max(1, n_projections)
        self._projection_dim = max(1, projection_dim)

    def test(
        self,
        x: int,
        y: int,
        conditioning_set: FrozenSet[int],
        data: DataMatrix,
        alpha: float = 0.05,
        rng: Optional[np.random.Generator] = None,
    ) -> PValue:
        """Approximate CI test using random projections of conditioning set.

        Parameters
        ----------
        x, y :
            Variable indices.
        conditioning_set :
            Conditioning set indices.
        data :
            Data matrix.
        alpha :
            Significance level.
        rng :
            Random generator.

        Returns
        -------
        PValue
        """
        if rng is None:
            rng = np.random.default_rng()

        n_samples = data.shape[0]
        cond_list = sorted(conditioning_set)

        if len(cond_list) <= self._projection_dim:
            # Small enough for exact computation
            return self._exact_partial_corr_test(x, y, cond_list, data)

        # Monte Carlo: project conditioning set to lower dimensions
        pvalues = []
        for _ in range(self._n_projections):
            # Random subset of conditioning variables
            subset_size = min(self._projection_dim, len(cond_list))
            subset = list(rng.choice(cond_list, size=subset_size, replace=False))
            p = self._exact_partial_corr_test(x, y, subset, data)
            pvalues.append(p)

        return float(np.median(pvalues))

    @staticmethod
    def _exact_partial_corr_test(
        x: int, y: int, cond: List[int], data: DataMatrix,
    ) -> PValue:
        """Exact partial correlation CI test."""
        n = data.shape[0]
        s_size = len(cond)
        dof = n - s_size - 2

        if dof < 1:
            return 1.0

        indices = sorted({x, y} | set(cond))
        sub = data[:, indices]
        cov = np.cov(sub, rowvar=False)
        cov += 1e-10 * np.eye(cov.shape[0])

        try:
            precision = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            return 1.0

        idx_map = {v: i for i, v in enumerate(indices)}
        ix, iy = idx_map[x], idx_map[y]
        denom = math.sqrt(abs(precision[ix, ix] * precision[iy, iy]))
        r = float(-precision[ix, iy] / denom) if denom > 1e-15 else 0.0

        r = max(-1 + 1e-12, min(1 - 1e-12, r))
        z = 0.5 * math.log((1 + r) / (1 - r)) * math.sqrt(dof)
        return float(2.0 * sp_stats.norm.sf(abs(z)))


class ImportanceSamplingCI:
    """Importance-sampling based CI test for rare conditioning events.

    For conditioning variables with rare values, standard sub-sampling
    may miss important subgroups.  This class uses importance weights
    to ensure adequate representation.

    Parameters
    ----------
    n_samples :
        Number of importance-weighted samples.
    """

    def __init__(self, n_samples: int = 500) -> None:
        self._n_samples = max(10, n_samples)

    def test(
        self,
        x: int,
        y: int,
        conditioning_set: FrozenSet[int],
        data: DataMatrix,
        alpha: float = 0.05,
        rng: Optional[np.random.Generator] = None,
    ) -> PValue:
        """CI test with importance sampling.

        Parameters
        ----------
        x, y :
            Variable indices.
        conditioning_set :
            Conditioning set indices.
        data :
            Data matrix.
        alpha :
            Significance level.
        rng :
            Random generator.

        Returns
        -------
        PValue
        """
        if rng is None:
            rng = np.random.default_rng()

        n = data.shape[0]
        cond_list = sorted(conditioning_set)

        if not cond_list:
            # Unconditional test: just use correlation
            corr = np.corrcoef(data[:, x], data[:, y])[0, 1]
            r = max(-1 + 1e-12, min(1 - 1e-12, corr))
            dof = n - 2
            if dof < 1:
                return 1.0
            t_stat = r * math.sqrt(dof / (1 - r ** 2))
            return float(2.0 * sp_stats.t.sf(abs(t_stat), df=dof))

        # Compute importance weights based on conditioning variable density
        cond_data = data[:, cond_list]
        # KDE-like density estimation via distance to mean
        mean_cond = cond_data.mean(axis=0)
        dists = np.linalg.norm(cond_data - mean_cond, axis=1)
        # Inverse density weighting: give more weight to rare values
        weights = 1.0 / (dists + 1e-8)
        weights /= weights.sum()

        # Weighted sub-sample
        sample_size = min(self._n_samples, n)
        indices = rng.choice(n, size=sample_size, replace=True, p=weights)
        sampled = data[indices]

        # Run standard partial correlation test on importance-sampled data
        return MonteCarloCITest._exact_partial_corr_test(x, y, cond_list, sampled)

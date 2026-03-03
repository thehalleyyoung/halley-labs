"""Adaptive conditional independence test selection.

Automatically selects the most appropriate CI test based on data
characteristics such as variable types, sample size, and estimated
nonlinearity.

Also provides :class:`CITestSuite` for running multiple CI tests and
combining their results via Fisher's method or Bonferroni correction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Protocol for CI tests
# ---------------------------------------------------------------------------

class CITest(Protocol):
    """Protocol for conditional independence tests."""

    def test(
        self,
        data: NDArray[np.float64],
        x: int,
        y: int,
        conditioning_set: Set[int] | None = None,
    ) -> Tuple[float, float]: ...


# ---------------------------------------------------------------------------
# Data characteristics
# ---------------------------------------------------------------------------

@dataclass
class DataCharacteristics:
    """Summary of data properties relevant for test selection.

    Attributes
    ----------
    is_continuous : bool
        All inspected variables are continuous.
    is_discrete : bool
        All inspected variables are discrete / categorical.
    is_mixed : bool
        Some variables are continuous and some are discrete.
    n_samples : int
        Number of observations.
    n_variables : int
        Number of variables (columns).
    estimated_nonlinearity : float
        Score in [0, 1] estimating degree of nonlinear dependence.
    is_gaussian : bool
        Whether inspected variables pass a normality test.
    conditioning_set_size : int
        Size of the conditioning set for this particular test invocation.
    """

    is_continuous: bool = True
    is_discrete: bool = False
    is_mixed: bool = False
    n_samples: int = 0
    n_variables: int = 0
    estimated_nonlinearity: float = 0.0
    is_gaussian: bool = True
    conditioning_set_size: int = 0


# ---------------------------------------------------------------------------
# Data inspection helpers
# ---------------------------------------------------------------------------

def _is_discrete(
    data: NDArray,
    col: int,
    threshold: int = 20,
) -> bool:
    """Check whether column *col* looks discrete.

    A column is considered discrete if the number of unique values is
    at most *threshold* or if all values are integers.
    """
    vals = data[:, col]
    n_unique = len(np.unique(vals))
    if n_unique <= threshold:
        return True
    # Check if all values are (close to) integers
    if np.issubdtype(vals.dtype, np.integer):
        return True
    if np.allclose(vals, np.round(vals)):
        return n_unique <= max(threshold, int(0.05 * len(vals)))
    return False


def _is_gaussian(
    data: NDArray,
    col: int,
    alpha: float = 0.05,
    max_samples: int = 5000,
) -> bool:
    """Test column *col* for normality using the Shapiro-Wilk test.

    For very large samples (>5000) we subsample to keep the test fast.
    """
    vals = data[:, col].astype(np.float64)
    n = len(vals)
    if n < 8:
        return True  # can't reliably reject

    if n > max_samples:
        rng = np.random.default_rng(seed=0)
        vals = rng.choice(vals, size=max_samples, replace=False)

    _, p = sp_stats.shapiro(vals)
    return p >= alpha


def _has_nonlinear_dependence(
    data: NDArray,
    x: int,
    y: int,
    max_samples: int = 2000,
) -> float:
    """Estimate degree of nonlinear dependence between columns x and y.

    Computes the difference between the Spearman rank correlation and the
    Pearson correlation.  A large gap suggests nonlinear (monotone)
    dependence, while a further check via the Hoeffding-style statistic
    captures non-monotone patterns.

    Returns a score in [0, 1].
    """
    xv = data[:, x].astype(np.float64)
    yv = data[:, y].astype(np.float64)

    if len(xv) > max_samples:
        rng = np.random.default_rng(seed=1)
        idx = rng.choice(len(xv), size=max_samples, replace=False)
        xv, yv = xv[idx], yv[idx]

    if np.std(xv) < 1e-10 or np.std(yv) < 1e-10:
        return 0.0

    pearson_r = np.abs(np.corrcoef(xv, yv)[0, 1])
    spearman_r = np.abs(sp_stats.spearmanr(xv, yv).statistic)

    # Fit a simple quadratic and compare R^2 vs linear R^2
    linear_r2 = pearson_r ** 2

    # Polynomial fit (degree 2)
    try:
        coeffs = np.polyfit(xv, yv, deg=2)
        y_pred = np.polyval(coeffs, xv)
        ss_res = np.sum((yv - y_pred) ** 2)
        ss_tot = np.sum((yv - yv.mean()) ** 2)
        if ss_tot > 1e-15:
            quad_r2 = 1.0 - ss_res / ss_tot
        else:
            quad_r2 = 0.0
    except (np.linalg.LinAlgError, ValueError):
        quad_r2 = linear_r2

    # Nonlinearity score: combine rank–Pearson gap and quadratic improvement
    gap = max(spearman_r - pearson_r, 0.0)
    quad_gain = max(quad_r2 - linear_r2, 0.0)
    score = min(gap + quad_gain, 1.0)
    return float(score)


def _sample_size_adequate(
    n: int,
    k: int,
    test_type: str,
) -> bool:
    """Check whether the sample size *n* is adequate for the test.

    Parameters
    ----------
    n : int
        Sample size.
    k : int
        Conditioning-set size.
    test_type : str
        ``"fisher_z"``, ``"chi_squared"``, ``"kci"``, or ``"cmi"``.
    """
    if test_type == "fisher_z":
        return n - k - 3 > 0
    if test_type in ("chi_squared", "g_test"):
        # Rule of thumb: at least 5 expected per cell; rough proxy
        return n >= 10 * (k + 2)
    if test_type == "kci":
        return n >= 20
    if test_type == "cmi":
        return n >= max(50, 10 * (k + 2))
    return n > k + 5


# ---------------------------------------------------------------------------
# Adaptive CI test
# ---------------------------------------------------------------------------

class AdaptiveCITest:
    """Adaptive CI test that delegates to the best-suited backend.

    When no tests are provided, the class lazily instantiates the
    built-in tests (Fisher-z, Chi-squared, KCI, CMI) on first use.

    Parameters
    ----------
    alpha : float
        Significance level.
    tests : dict[str, CITest] | None
        Named CI test instances to choose from.
    auto_select : bool
        If ``True`` (default), automatically select the best test.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        tests: Optional[Dict[str, CITest]] = None,
        auto_select: bool = True,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        self.alpha = alpha
        self.tests: Dict[str, CITest] = tests if tests is not None else {}
        self.auto_select = auto_select
        self._defaults_loaded = len(self.tests) > 0

    # -- lazy default tests -------------------------------------------------

    def _ensure_defaults(self) -> None:
        """Lazily create default CI test instances."""
        if self._defaults_loaded:
            return
        from cpa.ci_tests.fisher_z import FisherZTest
        from cpa.ci_tests.kernel_ci import KernelCITest
        from cpa.ci_tests.discrete_ci import ChiSquaredTest
        from cpa.ci_tests.conditional_mutual_info import CMITest

        self.tests.setdefault("fisher_z", FisherZTest(alpha=self.alpha))
        self.tests.setdefault("chi_squared", ChiSquaredTest(alpha=self.alpha))
        self.tests.setdefault("kci", KernelCITest(alpha=self.alpha, n_bootstrap=200))
        self.tests.setdefault("cmi", CMITest(alpha=self.alpha, n_permutations=200))
        self._defaults_loaded = True

    # -- public interface ---------------------------------------------------

    def test(
        self,
        data: NDArray[np.float64],
        x: int,
        y: int,
        conditioning_set: Set[int] | None = None,
    ) -> Tuple[float, float]:
        """Adaptively select and run a CI test.

        Returns (statistic, p_value).
        """
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("data must be 2-D")
        self._ensure_defaults()

        chosen = self.select_test(data, x, y, conditioning_set)
        return chosen.test(data, x, y, conditioning_set)

    def select_test(
        self,
        data: NDArray[np.float64],
        x: int,
        y: int,
        conditioning_set: Set[int] | None = None,
    ) -> CITest:
        """Choose the most appropriate CI test for the given variables.

        Selection logic:
        1. If both variables are discrete → Chi-squared.
        2. If data is continuous & Gaussian with no strong nonlinearity
           → Fisher-z.
        3. If data is continuous but nonlinear or non-Gaussian → KCI
           (small n) or CMI (large n).
        4. Fall back to CMI for mixed data.
        """
        self._ensure_defaults()
        z = conditioning_set if conditioning_set is not None else set()
        n = data.shape[0]
        k = len(z)

        # Check discreteness of x and y
        x_discrete = _is_discrete(data, x)
        y_discrete = _is_discrete(data, y)

        if x_discrete and y_discrete:
            if "chi_squared" in self.tests:
                return self.tests["chi_squared"]

        if not x_discrete and not y_discrete:
            # Both continuous
            x_gauss = _is_gaussian(data, x)
            y_gauss = _is_gaussian(data, y)
            nonlin = _has_nonlinear_dependence(data, x, y)

            if x_gauss and y_gauss and nonlin < 0.15:
                if _sample_size_adequate(n, k, "fisher_z"):
                    return self.tests.get("fisher_z", self.tests["cmi"])

            # Non-Gaussian or nonlinear
            if n <= 500 and "kci" in self.tests:
                return self.tests["kci"]
            if "cmi" in self.tests:
                return self.tests["cmi"]

        # Mixed or fallback
        if "cmi" in self.tests:
            return self.tests["cmi"]

        # Last resort: return first available test
        return next(iter(self.tests.values()))

    def characterize_data(
        self,
        data: NDArray[np.float64],
    ) -> DataCharacteristics:
        """Analyse *data* to determine its characteristics.

        Inspects all columns for discreteness, normality, and pairwise
        nonlinearity.
        """
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("data must be 2-D")
        n, p = data.shape

        discrete_flags = [_is_discrete(data, j) for j in range(p)]
        all_discrete = all(discrete_flags)
        all_continuous = not any(discrete_flags)
        is_mixed = not all_discrete and not all_continuous

        # Normality of continuous columns
        gauss_flags = []
        for j in range(p):
            if not discrete_flags[j]:
                gauss_flags.append(_is_gaussian(data, j))
        is_gaussian = all(gauss_flags) if gauss_flags else False

        # Average nonlinearity across a sample of pairs
        nonlin_scores: List[float] = []
        cont_cols = [j for j in range(p) if not discrete_flags[j]]
        max_pairs = min(10, len(cont_cols) * (len(cont_cols) - 1) // 2)
        pair_count = 0
        for i in range(len(cont_cols)):
            for j in range(i + 1, len(cont_cols)):
                nonlin_scores.append(
                    _has_nonlinear_dependence(data, cont_cols[i], cont_cols[j])
                )
                pair_count += 1
                if pair_count >= max_pairs:
                    break
            if pair_count >= max_pairs:
                break
        avg_nonlin = float(np.mean(nonlin_scores)) if nonlin_scores else 0.0

        return DataCharacteristics(
            is_continuous=all_continuous,
            is_discrete=all_discrete,
            is_mixed=is_mixed,
            n_samples=n,
            n_variables=p,
            estimated_nonlinearity=avg_nonlin,
            is_gaussian=is_gaussian,
        )


# ---------------------------------------------------------------------------
# CI test suite (consensus testing)
# ---------------------------------------------------------------------------

class CITestSuite:
    """Run multiple CI tests and aggregate their results.

    Parameters
    ----------
    tests : dict[str, CITest]
        Named CI test instances.
    alpha : float
        Overall significance level.
    """

    def __init__(
        self,
        tests: Dict[str, CITest],
        alpha: float = 0.05,
    ) -> None:
        if not tests:
            raise ValueError("At least one test is required")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        self.tests = dict(tests)
        self.alpha = alpha

    def consensus_test(
        self,
        data: NDArray[np.float64],
        x: int,
        y: int,
        conditioning_set: Set[int] | None = None,
        method: str = "fisher",
    ) -> Tuple[float, bool, Dict[str, Tuple[float, float]]]:
        """Run all tests and aggregate p-values.

        Parameters
        ----------
        data : ndarray of shape (n, p)
        x, y : int
        conditioning_set : set of int or None
        method : str
            ``"fisher"`` (Fisher's combined p-value) or ``"bonferroni"``
            (Bonferroni correction).

        Returns
        -------
        combined_p : float
            Aggregated p-value.
        independent : bool
            Whether the combined p-value indicates independence.
        individual : dict
            Mapping from test name to (statistic, p_value).
        """
        data = np.asarray(data, dtype=np.float64)
        individual: Dict[str, Tuple[float, float]] = {}

        for name, test in self.tests.items():
            try:
                stat, pval = test.test(data, x, y, conditioning_set)
                individual[name] = (stat, pval)
            except Exception:
                # Skip tests that fail for this data
                continue

        if not individual:
            return (1.0, True, individual)

        p_values = [pv for _, pv in individual.values()]

        if method == "fisher":
            combined_p = self._fisher_combined_p(p_values)
        elif method == "bonferroni":
            combined_p = self._bonferroni_correction(p_values)
        else:
            raise ValueError(f"Unknown method: {method!r}")

        independent = combined_p >= self.alpha
        return (combined_p, independent, individual)

    @staticmethod
    def _fisher_combined_p(p_values: List[float]) -> float:
        """Fisher's method for combining independent p-values.

        The test statistic is  -2 ∑ ln(p_i),  which follows a χ²
        distribution with 2k degrees of freedom under the null.

        Parameters
        ----------
        p_values : list of float

        Returns
        -------
        float
            Combined p-value.
        """
        if not p_values:
            return 1.0
        k = len(p_values)
        # Clamp p-values away from 0 to avoid log(0)
        clamped = [max(p, 1e-300) for p in p_values]
        chi2_stat = -2.0 * sum(np.log(p) for p in clamped)
        combined_p = float(sp_stats.chi2.sf(chi2_stat, df=2 * k))
        return combined_p

    @staticmethod
    def _bonferroni_correction(p_values: List[float]) -> float:
        """Bonferroni correction: multiply smallest p-value by k.

        Parameters
        ----------
        p_values : list of float

        Returns
        -------
        float
            Bonferroni-corrected p-value (clamped to [0, 1]).
        """
        if not p_values:
            return 1.0
        k = len(p_values)
        return min(min(p_values) * k, 1.0)

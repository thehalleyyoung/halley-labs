"""
Adaptive CI test selection and sequential testing.

Provides a meta-learning approach that selects the most appropriate CI
test based on data characteristics, aggregates results with confidence
weights, and supports sequential testing with alpha-spending for early
stopping.

Key features:
- Feature extraction: sample size, dimensionality, nonlinearity measures
- Decision-rule test selection calibrated on synthetic benchmarks
- Confidence-weighted aggregation of multiple test results
- O'Brien–Fleming style alpha-spending for sequential testing
- Calibration assessment on held-out data

References
----------
Shah, R. D. & Peters, J. (2020). The hardness of conditional
    independence testing and the generalised covariance measure.
    *Annals of Statistics*, 48(3), 1514–1538.

O'Brien, P. C. & Fleming, T. R. (1979). A multiple testing procedure
    for clinical trials. *Biometrics*, 35(3), 549–556.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from causalcert.ci_testing.base import (
    BaseCITest,
    CITestConfig,
    _extract_columns,
    _insufficient_sample_result,
    _validate_inputs,
)
from causalcert.types import CITestMethod, CITestResult, NodeId, NodeSet

_EPS = 1e-10


# ---------------------------------------------------------------------------
# Data characteristic features
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DataCharacteristics:
    """Extracted features describing the data for meta-learning.

    Attributes
    ----------
    n_samples : int
        Sample size.
    dim_x : int
        Dimensionality of X.
    dim_y : int
        Dimensionality of Y.
    dim_z : int
        Dimensionality of the conditioning set.
    nonlinearity_xy : float
        Nonlinearity measure between X and Y (0 = linear, 1 = highly
        nonlinear).
    skewness_x : float
        Skewness of X.
    kurtosis_x : float
        Excess kurtosis of X.
    skewness_y : float
        Skewness of Y.
    kurtosis_y : float
        Excess kurtosis of Y.
    n_discrete_x : bool
        Whether X appears discrete.
    n_discrete_y : bool
        Whether Y appears discrete.
    rank_deficiency : float
        Ratio of effective rank to nominal dimension of Z.
    """

    n_samples: int = 0
    dim_x: int = 1
    dim_y: int = 1
    dim_z: int = 0
    nonlinearity_xy: float = 0.0
    skewness_x: float = 0.0
    kurtosis_x: float = 0.0
    skewness_y: float = 0.0
    kurtosis_y: float = 0.0
    n_discrete_x: bool = False
    n_discrete_y: bool = False
    rank_deficiency: float = 1.0


def _extract_characteristics(
    x_col: np.ndarray,
    y_col: np.ndarray,
    z_cols: np.ndarray | None,
) -> DataCharacteristics:
    """Extract data characteristics for meta-learning.

    Parameters
    ----------
    x_col : np.ndarray
        X values ``(n,)``.
    y_col : np.ndarray
        Y values ``(n,)``.
    z_cols : np.ndarray | None
        Conditioning variables ``(n, k)`` or ``None``.

    Returns
    -------
    DataCharacteristics
    """
    n = len(x_col)

    # Nonlinearity: compare linear R² with Spearman rank correlation
    try:
        pearson_r = np.corrcoef(x_col, y_col)[0, 1]
        spearman_r, _ = stats.spearmanr(x_col, y_col)
        pearson_r2 = pearson_r ** 2
        spearman_r2 = spearman_r ** 2
        nonlinearity = max(spearman_r2 - pearson_r2, 0.0)
    except Exception:
        nonlinearity = 0.0

    # Discreteness check
    x_disc = len(np.unique(x_col)) <= min(10, n * 0.05)
    y_disc = len(np.unique(y_col)) <= min(10, n * 0.05)

    # Conditioning set rank
    dim_z = 0
    rank_def = 1.0
    if z_cols is not None:
        dim_z = z_cols.shape[1]
        if dim_z > 0:
            try:
                s = np.linalg.svd(z_cols, compute_uv=False)
                effective_rank = np.sum(s > s[0] * 1e-6)
                rank_def = effective_rank / max(dim_z, 1)
            except Exception:
                rank_def = 1.0

    return DataCharacteristics(
        n_samples=n,
        dim_x=1,
        dim_y=1,
        dim_z=dim_z,
        nonlinearity_xy=nonlinearity,
        skewness_x=float(stats.skew(x_col)),
        kurtosis_x=float(stats.kurtosis(x_col)),
        skewness_y=float(stats.skew(y_col)),
        kurtosis_y=float(stats.kurtosis(y_col)),
        n_discrete_x=x_disc,
        n_discrete_y=y_disc,
        rank_deficiency=rank_def,
    )


# ---------------------------------------------------------------------------
# Test selection via decision rules
# ---------------------------------------------------------------------------


def _score_method(
    method: CITestMethod,
    chars: DataCharacteristics,
) -> float:
    """Score a CI test method based on data characteristics.

    Higher score ⇒ better fit for the data.  Decision rules are
    calibrated on synthetic benchmarks covering linear/nonlinear,
    small/large n, and low/high conditioning-set dimension.

    Parameters
    ----------
    method : CITestMethod
        CI test method to score.
    chars : DataCharacteristics
        Data characteristics.

    Returns
    -------
    float
        Suitability score (higher is better).
    """
    n = chars.n_samples
    k = chars.dim_z
    nl = chars.nonlinearity_xy

    if method == CITestMethod.PARTIAL_CORRELATION:
        # Good for linear, small n, low k
        score = 5.0
        if nl > 0.1:
            score -= 3.0 * nl
        if n < 50:
            score += 2.0
        if k > max(n * 0.3, 10):
            score -= 2.0
        if chars.n_discrete_x or chars.n_discrete_y:
            score -= 1.0
        return score

    elif method == CITestMethod.KERNEL:
        # Good for nonlinear, medium-large n
        score = 4.0
        score += 3.0 * nl
        if n < 50:
            score -= 3.0
        elif n >= 500:
            score += 1.5
        if k > 10:
            score -= 1.0
        return score

    elif method == CITestMethod.HSIC:
        # Similar to kernel but with gamma approximation
        score = 4.0
        score += 2.5 * nl
        if n < 30:
            score -= 3.0
        elif n >= 200:
            score += 1.0
        return score

    elif method == CITestMethod.MUTUAL_INFO:
        # Good for nonlinear, mixed variable types
        score = 3.5
        score += 2.0 * nl
        if chars.n_discrete_x or chars.n_discrete_y:
            score += 2.0
        if n < 100:
            score -= 1.5
        if k > 5:
            score -= 0.5 * (k - 5)
        return score

    elif method == CITestMethod.CLASSIFIER:
        # Good for complex nonlinear, large n, mixed types
        score = 3.0
        score += 2.5 * nl
        if n < 100:
            score -= 4.0
        elif n >= 500:
            score += 2.0
        if chars.n_discrete_x or chars.n_discrete_y:
            score += 1.0
        return score

    elif method == CITestMethod.RANK:
        # Robust, moderate power, works everywhere
        score = 3.5
        if abs(chars.skewness_x) > 2 or abs(chars.skewness_y) > 2:
            score += 1.0  # heavy tails
        if abs(chars.kurtosis_x) > 5 or abs(chars.kurtosis_y) > 5:
            score += 1.0
        return score

    elif method == CITestMethod.CRT:
        # Good with large conditioning sets
        score = 3.0
        if n < 100:
            score -= 2.0
        if k > 5:
            score += 1.5
        return score

    return 1.0  # Default for unknown methods


def _select_tests(
    chars: DataCharacteristics,
    available: Sequence[BaseCITest],
    max_tests: int = 3,
) -> list[tuple[BaseCITest, float]]:
    """Select the best tests and their confidence weights.

    Parameters
    ----------
    chars : DataCharacteristics
        Data characteristics.
    available : Sequence[BaseCITest]
        Available CI test instances.
    max_tests : int
        Maximum number of tests to select.

    Returns
    -------
    list[tuple[BaseCITest, float]]
        Selected ``(test, weight)`` pairs sorted by score descending.
    """
    scored = []
    for test in available:
        s = _score_method(test.method, chars)
        scored.append((test, s))

    scored.sort(key=lambda t: t[1], reverse=True)
    top = scored[:max_tests]

    # Softmax weights
    scores = np.array([s for _, s in top], dtype=np.float64)
    scores -= scores.max()  # numerical stability
    exp_scores = np.exp(scores)
    weights = exp_scores / exp_scores.sum()

    return [(t, float(w)) for (t, _), w in zip(top, weights)]


# ---------------------------------------------------------------------------
# Alpha-spending for sequential testing
# ---------------------------------------------------------------------------


def _obrien_fleming_boundary(
    alpha: float,
    n_looks: int,
    look: int,
) -> float:
    """O'Brien–Fleming alpha-spending boundary.

    The O'Brien–Fleming boundary is very conservative at early looks
    and spends most of the alpha at the final look.

    alpha*(look) = 2 * (1 - Phi(z_{alpha/2} / sqrt(look / n_looks)))

    Parameters
    ----------
    alpha : float
        Overall significance level.
    n_looks : int
        Total planned number of looks (tests).
    look : int
        Current look index (1-based).

    Returns
    -------
    float
        Alpha to spend at this look.
    """
    if look < 1 or look > n_looks:
        return 0.0
    z_alpha = stats.norm.ppf(1 - alpha / 2.0)
    t = look / n_looks
    boundary = 2.0 * stats.norm.sf(z_alpha / math.sqrt(t))
    return float(np.clip(boundary, 0.0, alpha))


def _pocock_boundary(
    alpha: float,
    n_looks: int,
    look: int,
) -> float:
    """Pocock alpha-spending boundary (constant boundary).

    Spends alpha equally across all looks (adjusted for multiplicity).

    Parameters
    ----------
    alpha : float
        Overall significance level.
    n_looks : int
        Total number of looks.
    look : int
        Current look (1-based).

    Returns
    -------
    float
        Alpha at this look.
    """
    if look < 1 or look > n_looks:
        return 0.0
    # Bonferroni-like constant boundary
    return alpha / n_looks


# ---------------------------------------------------------------------------
# Confidence-weighted aggregation
# ---------------------------------------------------------------------------


def _confidence_weighted_combine(
    p_values: Sequence[float],
    weights: Sequence[float],
) -> float:
    """Combine p-values using confidence-weighted aggregation.

    Uses a weighted generalisation of the Cauchy combination that gives
    more weight to tests with higher confidence (suitability) scores.

    T = sum(w_i * tan((0.5 - p_i) * pi))
    p_combined = 0.5 - arctan(T) / pi

    Parameters
    ----------
    p_values : Sequence[float]
        Individual p-values.
    weights : Sequence[float]
        Confidence weights (must sum to 1).

    Returns
    -------
    float
        Combined p-value.
    """
    p_arr = np.asarray(p_values, dtype=np.float64)
    w_arr = np.asarray(weights, dtype=np.float64)

    if len(p_arr) == 0:
        return 1.0

    w_sum = w_arr.sum()
    if w_sum < _EPS:
        w_arr = np.ones_like(w_arr) / len(w_arr)
    else:
        w_arr = w_arr / w_sum

    p_arr = np.clip(p_arr, 1e-15, 1 - 1e-15)
    T = float(np.sum(w_arr * np.tan((0.5 - p_arr) * np.pi)))
    p_combined = 0.5 - np.arctan(T) / np.pi
    return float(np.clip(p_combined, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Calibration check
# ---------------------------------------------------------------------------


def _calibration_check(
    test: BaseCITest,
    data: pd.DataFrame,
    x: NodeId,
    y: NodeId,
    conditioning_set: NodeSet,
    *,
    n_bootstrap: int = 50,
    holdout_fraction: float = 0.3,
    seed: int = 42,
) -> float:
    """Check test calibration on held-out bootstrap samples.

    Under the null, p-values should be approximately uniform.
    We compute the KS statistic against Uniform(0, 1).

    Parameters
    ----------
    test : BaseCITest
        CI test to calibrate.
    data : pd.DataFrame
        Full dataset.
    x, y : NodeId
        Variables to test.
    conditioning_set : NodeSet
        Conditioning set.
    n_bootstrap : int
        Number of bootstrap rounds.
    holdout_fraction : float
        Fraction of data held out per round.
    seed : int
        Random seed.

    Returns
    -------
    float
        KS statistic (lower ⇒ better calibrated).
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    holdout_n = max(int(n * holdout_fraction), 10)

    p_values: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=holdout_n, replace=True)
        sub = data.iloc[idx].reset_index(drop=True)
        try:
            result = test.test(x, y, conditioning_set, sub)
            p_values.append(result.p_value)
        except Exception:
            continue

    if len(p_values) < 5:
        return 1.0  # Cannot assess calibration

    ks_stat, _ = stats.kstest(p_values, "uniform")
    return float(ks_stat)


# ---------------------------------------------------------------------------
# AdaptiveEnsemble class
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AdaptiveConfig:
    """Configuration for the adaptive ensemble.

    Attributes
    ----------
    max_tests : int
        Maximum number of tests to run.
    sequential : bool
        Enable sequential testing with alpha-spending.
    spending_function : str
        ``"obrien_fleming"`` or ``"pocock"``.
    calibration_check : bool
        Run calibration check on held-out data.
    calibration_bootstrap : int
        Number of bootstrap samples for calibration.
    """

    max_tests: int = 3
    sequential: bool = True
    spending_function: str = "obrien_fleming"
    calibration_check: bool = False
    calibration_bootstrap: int = 50


class AdaptiveEnsemble(BaseCITest):
    """Adaptive CI test with meta-learning selection.

    Automatically selects the most suitable CI tests based on data
    characteristics, runs them with optional sequential early stopping,
    and combines their results via confidence-weighted aggregation.

    Parameters
    ----------
    base_tests : Sequence[BaseCITest]
        Pool of available CI tests.
    alpha : float
        Significance level.
    seed : int
        Random seed.
    config : CITestConfig | None
        Base CI test configuration.
    adaptive_config : AdaptiveConfig | None
        Adaptive-specific configuration.
    """

    method = CITestMethod.ADAPTIVE

    def __init__(
        self,
        base_tests: Sequence[BaseCITest],
        alpha: float = 0.05,
        seed: int = 42,
        config: CITestConfig | None = None,
        adaptive_config: AdaptiveConfig | None = None,
    ) -> None:
        super().__init__(alpha=alpha, seed=seed, config=config)
        if not base_tests:
            raise ValueError("At least one base test is required.")
        self.base_tests = list(base_tests)
        self.adaptive_config = adaptive_config or AdaptiveConfig()
        self._last_selection: list[tuple[CITestMethod, float]] = []
        self._last_sub_results: list[CITestResult] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_alpha_boundary(self, look: int, n_looks: int) -> float:
        """Get the alpha boundary for the current look.

        Parameters
        ----------
        look : int
            Current look (1-based).
        n_looks : int
            Total looks.

        Returns
        -------
        float
            Alpha boundary.
        """
        cfg = self.adaptive_config
        if cfg.spending_function == "pocock":
            return _pocock_boundary(self.alpha, n_looks, look)
        return _obrien_fleming_boundary(self.alpha, n_looks, look)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def test(
        self,
        x: NodeId,
        y: NodeId,
        conditioning_set: NodeSet,
        data: pd.DataFrame,
    ) -> CITestResult:
        """Test X ⊥ Y | Z with adaptive test selection.

        Parameters
        ----------
        x : NodeId
            First variable.
        y : NodeId
            Second variable.
        conditioning_set : NodeSet
            Conditioning variables.
        data : pd.DataFrame
            Observational data.

        Returns
        -------
        CITestResult
        """
        _validate_inputs(data, x, y, conditioning_set)
        x_col, y_col, z_cols = _extract_columns(data, x, y, conditioning_set)

        n = len(x_col)
        if n < self.config.min_samples:
            return _insufficient_sample_result(
                x, y, conditioning_set, self.method, self.alpha,
            )

        cfg = self.adaptive_config

        # Extract characteristics and select tests
        chars = _extract_characteristics(x_col, y_col, z_cols)
        selected = _select_tests(chars, self.base_tests, cfg.max_tests)
        self._last_selection = [
            (t.method, w) for t, w in selected
        ]

        # Run tests (sequentially with alpha-spending, or all at once)
        sub_results: list[CITestResult] = []
        sub_weights: list[float] = []
        n_looks = len(selected)

        for look_idx, (test, weight) in enumerate(selected, start=1):
            try:
                result = test.test(x, y, conditioning_set, data)
            except Exception as exc:
                warnings.warn(
                    f"Test {test.method.value} failed: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                result = CITestResult(
                    x=x, y=y,
                    conditioning_set=conditioning_set,
                    statistic=0.0, p_value=1.0,
                    method=test.method, reject=False,
                    alpha=self.alpha,
                )
            sub_results.append(result)
            sub_weights.append(weight)

            # Sequential early stopping
            if cfg.sequential and look_idx < n_looks:
                boundary = self._get_alpha_boundary(look_idx, n_looks)
                if result.p_value < boundary:
                    # Early rejection
                    break
                # Early acceptance: if p-value is very large
                if result.p_value > 1 - boundary:
                    break

        self._last_sub_results = sub_results

        # Combine p-values
        p_values = [r.p_value for r in sub_results]
        combined_p = _confidence_weighted_combine(p_values, sub_weights)

        # Calibration adjustment (optional)
        if cfg.calibration_check and len(selected) > 0:
            best_test = selected[0][0]
            ks = _calibration_check(
                best_test, data, x, y, conditioning_set,
                n_bootstrap=cfg.calibration_bootstrap,
                seed=self.seed,
            )
            # If poorly calibrated (KS > 0.3), inflate the p-value
            if ks > 0.3:
                combined_p = min(combined_p * (1 + ks), 1.0)

        # Aggregate statistic
        stat = float(np.mean([r.statistic for r in sub_results]))

        return self._make_result(x, y, conditioning_set, stat, combined_p)

    @property
    def last_selection(self) -> list[tuple[CITestMethod, float]]:
        """Return the test selection from the last run.

        Returns
        -------
        list[tuple[CITestMethod, float]]
            ``(method, weight)`` pairs.
        """
        return self._last_selection

    @property
    def last_sub_results(self) -> list[CITestResult]:
        """Return sub-test results from the last run.

        Returns
        -------
        list[CITestResult]
        """
        return self._last_sub_results

    def __repr__(self) -> str:  # noqa: D105
        tests = ", ".join(t.method.value for t in self.base_tests)
        cfg = self.adaptive_config
        return (
            f"AdaptiveEnsemble(tests=[{tests}], "
            f"sequential={cfg.sequential}, alpha={self.alpha})"
        )

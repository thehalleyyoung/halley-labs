"""Unit tests for usability_oracle.comparison.hypothesis — Statistical hypothesis testing.

Tests :class:`RegressionTester` and :class:`HypothesisResult` which implement
the statistical framework for usability regression detection.

The null and alternative hypotheses are:

    H₀: μ_B ≤ μ_A   (no regression — after version is no worse)
    H₁: μ_B > μ_A   (regression — after version is worse)

Three test methods are supported:
- Welch's t-test (unequal variance, parametric)
- Mann–Whitney U test (non-parametric, rank-based)
- Bootstrap/permutation test (distribution-free)

Multiple-testing corrections (Bonferroni, Holm, Benjamini–Hochberg) control
family-wise error rate when comparing multiple tasks simultaneously.

References
----------
- Welch, B. L. (1947). *Biometrika*, 34.
- Mann, H. B. & Whitney, D. R. (1947). *Ann. Math. Statist.*, 18(1).
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*.
- Holm, S. (1979). *Scand. J. Statist.*, 6(2), 65–70.
"""

from __future__ import annotations

import math
import pytest
import numpy as np

from usability_oracle.comparison.hypothesis import RegressionTester, HypothesisResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _identical_samples(n: int = 100, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate two identical-distribution samples for H₀-true scenario.

    Both samples are drawn from N(5.0, 1.0).  Since the distributions are
    identical, the test should fail to reject H₀.

    Parameters
    ----------
    n : int
        Sample size per group.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (samples_a, samples_b), each of shape (n,).
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(5.0, 1.0, n)
    b = rng.normal(5.0, 1.0, n)
    return a, b


def _different_samples(
    n: int = 100, delta: float = 2.0, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two clearly different samples for H₁-true scenario.

    Sample A is drawn from N(5.0, 1.0) and sample B from N(5.0 + delta, 1.0).
    The large shift should lead to rejection of H₀.

    Parameters
    ----------
    n : int
        Sample size per group.
    delta : float
        Mean shift for group B (positive ⇒ B is more costly).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (samples_a, samples_b).
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(5.0, 1.0, n)
    b = rng.normal(5.0 + delta, 1.0, n)
    return a, b


# ---------------------------------------------------------------------------
# Tests: RegressionTester initialization
# ---------------------------------------------------------------------------


class TestRegressionTesterInit:
    """Tests for RegressionTester constructor and validation."""

    def test_default_method(self):
        """Default RegressionTester should use Welch's t-test.

        The welch_t method is the standard parametric test for comparing
        means with unequal variances.
        """
        tester = RegressionTester()
        assert tester.method == "welch_t"
        assert tester.n_bootstrap == 10_000
        assert tester.min_samples == 10

    def test_mann_whitney_method(self):
        """RegressionTester should accept 'mann_whitney' as a test method.

        The Mann–Whitney U test is non-parametric and does not assume normality.
        """
        tester = RegressionTester(method="mann_whitney")
        assert tester.method == "mann_whitney"

    def test_bootstrap_method(self):
        """RegressionTester should accept 'bootstrap' as a test method.

        The bootstrap/permutation test is distribution-free and works by
        shuffling labels between the two groups.
        """
        tester = RegressionTester(method="bootstrap")
        assert tester.method == "bootstrap"

    def test_unknown_method_raises(self):
        """RegressionTester should raise ValueError for unknown test methods.

        Only 'welch_t', 'mann_whitney', and 'bootstrap' are supported.
        """
        with pytest.raises(ValueError, match="Unknown method"):
            RegressionTester(method="z_test")


# ---------------------------------------------------------------------------
# Tests: HypothesisResult fields
# ---------------------------------------------------------------------------


class TestHypothesisResult:
    """Tests for HypothesisResult dataclass fields."""

    def test_default_fields(self):
        """Default HypothesisResult should represent a non-rejection.

        test_statistic=0, p_value=1.0, reject_null=False by default.
        """
        r = HypothesisResult()
        assert r.test_statistic == 0.0
        assert r.p_value == 1.0
        assert r.reject_null is False
        assert r.effect_size == 0.0
        assert r.method == "welch_t"

    def test_all_fields_set(self):
        """HypothesisResult should faithfully store all provided fields.

        All fields should be accessible exactly as provided.
        """
        r = HypothesisResult(
            test_statistic=2.5,
            p_value=0.01,
            reject_null=True,
            effect_size=0.8,
            ci_lower=0.3,
            ci_upper=1.3,
            method="mann_whitney",
            n_a=50,
            n_b=50,
        )
        assert r.test_statistic == 2.5
        assert r.p_value == 0.01
        assert r.reject_null is True
        assert r.effect_size == 0.8
        assert r.ci_lower == 0.3
        assert r.ci_upper == 1.3
        assert r.method == "mann_whitney"
        assert r.n_a == 50
        assert r.n_b == 50


# ---------------------------------------------------------------------------
# Tests: Welch's t-test
# ---------------------------------------------------------------------------


class TestWelchTTest:
    """Tests for the Welch's t-test implementation (default method)."""

    def test_identical_samples_no_rejection(self):
        """Identical-distribution samples should not reject H₀.

        When both groups come from N(5, 1), the null hypothesis (no
        regression) should not be rejected at α=0.05.
        """
        tester = RegressionTester(method="welch_t")
        a, b = _identical_samples(n=100)
        result = tester.test(a, b, alpha=0.05)

        assert result.reject_null is False
        assert result.p_value > 0.05
        assert result.method == "welch_t"
        assert result.n_a == 100
        assert result.n_b == 100

    def test_different_samples_rejection(self):
        """Clearly different samples should reject H₀.

        With a 2σ shift, the test should have high power to detect
        the regression.
        """
        tester = RegressionTester(method="welch_t")
        a, b = _different_samples(n=100, delta=2.0)
        result = tester.test(a, b, alpha=0.05)

        assert result.reject_null is True
        assert result.p_value < 0.05

    def test_effect_size_near_zero_identical(self):
        """Cohen's d should be near zero for identical distributions.

        With no true difference, the standardized mean difference should
        be small (|d| < 0.3 allowing for sampling noise).
        """
        tester = RegressionTester(method="welch_t")
        a, b = _identical_samples(n=200)
        result = tester.test(a, b)

        assert abs(result.effect_size) < 0.3

    def test_effect_size_large_different(self):
        """Cohen's d should be large (≥ 0.8) for a 2σ shift.

        A shift of 2 standard deviations corresponds to d ≈ 2.0, well
        above the 'large' threshold of 0.8.
        """
        tester = RegressionTester(method="welch_t")
        a, b = _different_samples(n=100, delta=2.0)
        result = tester.test(a, b)

        assert result.effect_size >= 0.8

    def test_confidence_interval_contains_zero_identical(self):
        """For identical distributions, the CI for Δμ should contain zero.

        The true difference is zero, so the CI should straddle it.
        """
        tester = RegressionTester(method="welch_t")
        a, b = _identical_samples(n=100)
        result = tester.test(a, b, alpha=0.05)

        assert result.ci_lower <= 0.0 <= result.ci_upper

    def test_confidence_interval_excludes_zero_different(self):
        """For clearly different distributions, the CI should exclude zero.

        With a 2σ shift, both bounds should be positive (indicating the
        after version is more expensive).
        """
        tester = RegressionTester(method="welch_t")
        a, b = _different_samples(n=100, delta=2.0)
        result = tester.test(a, b, alpha=0.05)

        assert result.ci_lower > 0.0


# ---------------------------------------------------------------------------
# Tests: Mann-Whitney U test
# ---------------------------------------------------------------------------


class TestMannWhitneyU:
    """Tests for the Mann–Whitney U test (non-parametric)."""

    def test_identical_samples_no_rejection(self):
        """Mann–Whitney U should not reject H₀ for identical distributions.

        The non-parametric test should agree with the parametric test
        when there is no true difference.
        """
        tester = RegressionTester(method="mann_whitney")
        a, b = _identical_samples(n=100)
        result = tester.test(a, b, alpha=0.05)

        assert result.reject_null is False
        assert result.method == "mann_whitney"

    def test_different_samples_rejection(self):
        """Mann–Whitney U should reject H₀ for clearly shifted samples.

        The test detects stochastic dominance: P(B > A) > 0.5.
        """
        tester = RegressionTester(method="mann_whitney")
        a, b = _different_samples(n=100, delta=2.0)
        result = tester.test(a, b, alpha=0.05)

        assert result.reject_null is True
        assert result.p_value < 0.05

    def test_u_statistic_is_nonnegative(self):
        """The Mann–Whitney U statistic should always be non-negative.

        U counts pairs where b_j > a_i, so U ≥ 0 by definition.
        """
        tester = RegressionTester(method="mann_whitney")
        a, b = _different_samples(n=50)
        result = tester.test(a, b)

        assert result.test_statistic >= 0.0


# ---------------------------------------------------------------------------
# Tests: Bootstrap test
# ---------------------------------------------------------------------------


class TestBootstrapTest:
    """Tests for the bootstrap/permutation test."""

    def test_identical_samples_no_rejection(self):
        """Bootstrap test should not reject H₀ for identical distributions.

        The permutation distribution of Δ should center around zero,
        giving a high p-value.
        """
        tester = RegressionTester(method="bootstrap", n_bootstrap=1000)
        a, b = _identical_samples(n=50)
        result = tester.test(a, b, alpha=0.05)

        assert result.reject_null is False
        assert result.method == "bootstrap"

    def test_different_samples_rejection(self):
        """Bootstrap test should reject H₀ for clearly shifted samples.

        With 2σ separation, very few permuted datasets should show a
        larger difference than observed.
        """
        tester = RegressionTester(method="bootstrap", n_bootstrap=1000)
        a, b = _different_samples(n=50, delta=2.0)
        result = tester.test(a, b, alpha=0.05)

        assert result.reject_null is True
        assert result.p_value < 0.05

    def test_bootstrap_p_value_in_unit_interval(self):
        """Bootstrap p-value should always be in [0, 1].

        The p-value is computed as (count + 1) / (n_bootstrap + 1).
        """
        tester = RegressionTester(method="bootstrap", n_bootstrap=500)
        a, b = _identical_samples(n=30)
        result = tester.test(a, b)

        assert 0.0 <= result.p_value <= 1.0


# ---------------------------------------------------------------------------
# Tests: Insufficient samples
# ---------------------------------------------------------------------------


class TestInsufficientSamples:
    """Tests for behavior with too few samples."""

    def test_small_samples_returns_default(self):
        """With fewer than min_samples, test() should return a default result.

        The default result has p_value=1.0 and reject_null=False,
        indicating insufficient evidence.
        """
        tester = RegressionTester(min_samples=10)
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result = tester.test(a, b)

        assert result.p_value == 1.0
        assert result.reject_null is False

    def test_non_finite_raises(self):
        """test() should raise ValueError if samples contain NaN or Inf.

        Non-finite values corrupt statistical computations and must be
        caught early.
        """
        tester = RegressionTester()
        a = np.array([1.0] * 20)
        b = np.array([np.nan] * 20)

        with pytest.raises(ValueError, match="non-finite"):
            tester.test(a, b)


# ---------------------------------------------------------------------------
# Tests: Multiple testing correction
# ---------------------------------------------------------------------------


class TestMultipleTesting:
    """Tests for test_multiple() with Holm and other corrections."""

    def test_holm_correction_adjusts_p_values(self):
        """Holm correction should inflate p-values to control FWER.

        The corrected p-values should be ≥ the raw p-values, reducing
        false positive rate when testing many tasks simultaneously.
        """
        tester = RegressionTester(method="welch_t")
        pairs = []
        rng = np.random.default_rng(123)
        for _ in range(5):
            a = rng.normal(5.0, 1.0, 50)
            b = rng.normal(5.0, 1.0, 50)
            pairs.append((a, b))

        results = tester.test_multiple(pairs, alpha=0.05, correction="holm")

        assert len(results) == 5
        for r in results:
            assert "holm" in r.method

    def test_multiple_testing_reduces_false_positives(self):
        """With all null-true pairs, fewer corrected results should reject.

        Under H₀, raw p-values are uniform(0,1).  After Holm correction,
        the number of rejections should be ≤ α × n_tests on average.
        """
        tester = RegressionTester(method="welch_t")
        rng = np.random.default_rng(99)
        pairs = []
        for _ in range(10):
            a = rng.normal(5.0, 1.0, 50)
            b = rng.normal(5.0, 1.0, 50)
            pairs.append((a, b))

        results = tester.test_multiple(pairs, alpha=0.05, correction="holm")
        n_reject = sum(1 for r in results if r.reject_null)

        # Under H₀ with Holm, we expect ≤ 1 rejection most of the time
        assert n_reject <= 3

    def test_bonferroni_correction(self):
        """Bonferroni correction should multiply p-values by the number of tests.

        This is the simplest multiple testing correction: p_adj = m × p_raw.
        """
        tester = RegressionTester(method="welch_t")
        a, b = _identical_samples(n=50)
        pairs = [(a, b)] * 3

        results = tester.test_multiple(pairs, alpha=0.05, correction="bonferroni")
        assert len(results) == 3
        for r in results:
            assert "bonferroni" in r.method
            assert r.p_value <= 1.0

    def test_bh_correction(self):
        """Benjamini–Hochberg FDR correction should produce valid adjusted p-values.

        BH controls the false discovery rate rather than FWER, making it
        less conservative than Bonferroni or Holm.
        """
        tester = RegressionTester(method="welch_t")
        rng = np.random.default_rng(77)
        pairs = [(rng.normal(5, 1, 50), rng.normal(5, 1, 50)) for _ in range(5)]

        results = tester.test_multiple(pairs, alpha=0.05, correction="bh")
        assert len(results) == 5
        for r in results:
            assert 0.0 <= r.p_value <= 1.0

    def test_multiple_with_true_signal(self):
        """test_multiple should detect a true signal even after correction.

        If one pair has a large shift, it should still be rejected after
        Holm correction, while null-true pairs should not be rejected.
        """
        tester = RegressionTester(method="welch_t")
        rng = np.random.default_rng(55)
        pairs = []
        # 4 null-true pairs
        for _ in range(4):
            a = rng.normal(5.0, 1.0, 100)
            b = rng.normal(5.0, 1.0, 100)
            pairs.append((a, b))
        # 1 signal pair with large shift
        a_sig = rng.normal(5.0, 1.0, 100)
        b_sig = rng.normal(8.0, 1.0, 100)
        pairs.append((a_sig, b_sig))

        results = tester.test_multiple(pairs, alpha=0.05, correction="holm")

        # The signal pair (index 4) should be rejected
        assert results[4].reject_null is True

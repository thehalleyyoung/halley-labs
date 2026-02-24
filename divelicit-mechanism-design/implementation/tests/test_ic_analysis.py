"""Tests for IC violation analysis."""

import numpy as np
import pytest

from src.ic_analysis import (
    analyze_ic_violations,
    multiple_testing_correction,
    enhanced_bootstrap_ci,
    VCGConditionAnalysis,
)


@pytest.fixture
def small_instance():
    rng = np.random.RandomState(42)
    n, d = 10, 8
    embs = rng.randn(n, d)
    quals = rng.uniform(0.3, 0.9, n)
    return embs, quals


class TestICViolationAnalysis:
    """Test the IC violation analysis with VCG condition identification."""

    def test_analysis_completes(self, small_instance):
        embs, quals = small_instance
        result = analyze_ic_violations(
            embs, quals, k=3,
            n_random_trials=20, n_adversarial_trials=20, seed=42,
        )
        assert isinstance(result, VCGConditionAnalysis)
        assert result.total_tests > 0

    def test_c1_satisfied(self, small_instance):
        """Quasi-linearity should always be satisfied."""
        embs, quals = small_instance
        result = analyze_ic_violations(
            embs, quals, k=3,
            n_random_trials=20, n_adversarial_trials=20, seed=42,
        )
        assert result.c1_quasi_linearity is True
        assert result.c1_max_error < 1e-6

    def test_violation_types_sum(self, small_instance):
        embs, quals = small_instance
        result = analyze_ic_violations(
            embs, quals, k=3,
            n_random_trials=30, n_adversarial_trials=30, seed=42,
        )
        assert (result.type_a_count + result.type_b_count + result.type_c_count
                == result.total_violations)

    def test_ci_valid(self, small_instance):
        embs, quals = small_instance
        result = analyze_ic_violations(
            embs, quals, k=3,
            n_random_trials=30, n_adversarial_trials=30, seed=42,
        )
        lo, hi = result.empirical_ci
        assert 0 <= lo <= hi <= 1

    def test_theoretical_bound_positive(self, small_instance):
        embs, quals = small_instance
        result = analyze_ic_violations(
            embs, quals, k=3,
            n_random_trials=20, n_adversarial_trials=20, seed=42,
        )
        assert result.theoretical_epsilon_ic >= 0

    def test_explanation_nonempty(self, small_instance):
        embs, quals = small_instance
        result = analyze_ic_violations(
            embs, quals, k=3,
            n_random_trials=10, n_adversarial_trials=10, seed=42,
        )
        assert len(result.explanation) > 100
        assert "VCG Condition" in result.explanation


class TestMultipleTestingCorrection:
    """Test Bonferroni and BH corrections."""

    def test_bonferroni_basic(self):
        p_values = [0.01, 0.04, 0.03, 0.06]
        result = multiple_testing_correction(p_values, method="bonferroni", alpha=0.05)
        assert result["n_tests"] == 4
        # Bonferroni: p * m, so 0.01 * 4 = 0.04 < 0.05 => significant
        assert result["corrected_p_values"][0] == pytest.approx(0.04, abs=1e-10)
        assert result["significant"][0] is True

    def test_bonferroni_conservative(self):
        p_values = [0.02, 0.03]
        result = multiple_testing_correction(p_values, method="bonferroni", alpha=0.05)
        # 0.02 * 2 = 0.04 < 0.05 => significant
        # 0.03 * 2 = 0.06 > 0.05 => not significant
        assert result["significant"][0] is True
        assert result["significant"][1] is False

    def test_bh_basic(self):
        p_values = [0.01, 0.03, 0.04, 0.06]
        result = multiple_testing_correction(p_values, method="bh", alpha=0.05)
        assert result["n_tests"] == 4
        assert result["n_significant"] >= 1

    def test_bh_less_conservative_than_bonferroni(self):
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
        bonf = multiple_testing_correction(p_values, method="bonferroni", alpha=0.05)
        bh = multiple_testing_correction(p_values, method="bh", alpha=0.05)
        assert bh["n_significant"] >= bonf["n_significant"]

    def test_empty_p_values(self):
        result = multiple_testing_correction([], method="bonferroni")
        assert result["corrected_p_values"] == []

    def test_single_p_value(self):
        result = multiple_testing_correction([0.03], method="bonferroni", alpha=0.05)
        assert result["corrected_p_values"] == [0.03]
        assert result["significant"] == [True]

    def test_all_significant(self):
        result = multiple_testing_correction([0.001, 0.002], method="bonferroni", alpha=0.05)
        assert all(result["significant"])

    def test_none_significant(self):
        result = multiple_testing_correction([0.5, 0.8], method="bonferroni", alpha=0.05)
        assert not any(result["significant"])


class TestEnhancedBootstrapCI:
    """Test enhanced bootstrap CI with multiple seeds."""

    def test_basic_ci(self):
        rng = np.random.RandomState(42)
        values = rng.normal(5.0, 1.0, 100)
        result = enhanced_bootstrap_ci(values, n_seeds=5, n_bootstrap=500)
        assert result["ci_lower"] < 5.0 < result["ci_upper"]

    def test_many_seeds_more_stable(self):
        rng = np.random.RandomState(42)
        values = rng.normal(5.0, 1.0, 50)
        r5 = enhanced_bootstrap_ci(values, n_seeds=5, n_bootstrap=500)
        r20 = enhanced_bootstrap_ci(values, n_seeds=20, n_bootstrap=500)
        # More seeds should give smaller endpoint variance
        assert r20["ci_stability_lower_std"] <= r5["ci_stability_lower_std"] + 0.1

    def test_per_seed_cis_count(self):
        values = np.random.randn(30)
        result = enhanced_bootstrap_ci(values, n_seeds=10, n_bootstrap=200)
        assert len(result["per_seed_cis"]) == 10

    def test_total_samples_correct(self):
        values = np.random.randn(30)
        result = enhanced_bootstrap_ci(values, n_seeds=8, n_bootstrap=300)
        assert result["total_bootstrap_samples"] == 8 * 300

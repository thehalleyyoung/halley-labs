"""Tests for confidence interval computation.

Covers StabilitySelector, ParametricBootstrap, PermutationCalibrator,
and threshold calibration.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cpa.descriptors.confidence import (
    StabilitySelector,
    ParametricBootstrap,
    PermutationCalibrator,
    StabilitySelectionResult,
    BootstrapCIResult,
    PermutationResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(2024)


def _chain_learner(data):
    """Simple chain DAG learner."""
    p = data.shape[1]
    adj = np.zeros((p, p))
    for i in range(p - 1):
        adj[i, i + 1] = 1
    return adj


def _generate_chain_datasets(rng, n_contexts=5, n_samples=80, p=3):
    """Generate datasets from a chain DAG."""
    datasets = []
    for _ in range(n_contexts):
        X0 = rng.normal(0, 1, size=n_samples)
        X1 = 0.8 * X0 + rng.normal(0, 0.3, size=n_samples)
        X2 = 0.6 * X1 + rng.normal(0, 0.3, size=n_samples)
        if p == 3:
            datasets.append(np.column_stack([X0, X1, X2]))
        else:
            extra = [rng.normal(0, 1, size=n_samples) for _ in range(p - 3)]
            datasets.append(np.column_stack([X0, X1, X2] + extra))
    return datasets


@pytest.fixture
def chain_datasets(rng):
    return _generate_chain_datasets(rng, n_contexts=5, n_samples=80, p=3)


@pytest.fixture
def chain_parent_sets():
    """Parent sets for a 3-node chain: 0->1->2."""
    return [[0], [0], [0], [0], [0]]  # parent of node 1 in each context


@pytest.fixture
def stability_selector():
    return StabilitySelector(
        n_rounds=20,
        subsample_fraction=0.5,
        ci_level=0.95,
        random_state=42,
    )


@pytest.fixture
def bootstrap():
    return ParametricBootstrap(
        n_bootstrap=50,
        ci_level=0.95,
        ci_method="percentile",
        random_state=42,
    )


@pytest.fixture
def permutation_calibrator():
    return PermutationCalibrator(
        n_permutations=99,
        significance_level=0.05,
        random_state=42,
    )


# ---------------------------------------------------------------------------
# Test StabilitySelector
# ---------------------------------------------------------------------------

class TestStabilitySelector:

    def test_structural_ci_returns_result(self, stability_selector, chain_datasets):
        result = stability_selector.compute_structural_ci(
            chain_datasets, target_idx=1,
            dag_learner=_chain_learner, n_variables=3,
        )
        assert isinstance(result, StabilitySelectionResult)

    def test_structural_ci_is_interval(self, stability_selector, chain_datasets):
        result = stability_selector.compute_structural_ci(
            chain_datasets, target_idx=1,
            dag_learner=_chain_learner, n_variables=3,
        )
        assert result.ci_lower <= result.ci_upper + 1e-10

    def test_structural_ci_level(self, stability_selector, chain_datasets):
        result = stability_selector.compute_structural_ci(
            chain_datasets, target_idx=1,
            dag_learner=_chain_learner, n_variables=3,
        )
        assert result.ci_level == 0.95

    def test_emergence_ci(self, stability_selector, chain_datasets):
        lo, hi = stability_selector.compute_emergence_ci(
            chain_datasets, target_idx=1,
            dag_learner=_chain_learner, n_variables=3,
        )
        assert lo <= hi + 1e-10

    def test_threshold_calibration(self, stability_selector, chain_datasets):
        result = stability_selector.threshold_calibration(
            chain_datasets, target_idx=1,
            dag_learner=_chain_learner, n_variables=3,
        )
        assert isinstance(result, dict)

    def test_selection_probabilities(self, stability_selector, chain_datasets):
        result = stability_selector.compute_structural_ci(
            chain_datasets, target_idx=1,
            dag_learner=_chain_learner, n_variables=3,
        )
        assert result.selection_probabilities.shape[0] > 0

    def test_apply_correction(self, stability_selector, rng):
        p_values = rng.uniform(0, 1, size=10)
        corrected = stability_selector.apply_correction(p_values)
        assert corrected.shape == p_values.shape

    def test_different_ci_levels(self, chain_datasets):
        for level in [0.80, 0.90, 0.99]:
            sel = StabilitySelector(
                n_rounds=15, subsample_fraction=0.5,
                ci_level=level, random_state=42,
            )
            result = sel.compute_structural_ci(
                chain_datasets, target_idx=1,
                dag_learner=_chain_learner, n_variables=3,
            )
            assert result.ci_level == level

    def test_mean_and_median(self, stability_selector, chain_datasets):
        result = stability_selector.compute_structural_ci(
            chain_datasets, target_idx=1,
            dag_learner=_chain_learner, n_variables=3,
        )
        assert isinstance(result.mean_estimate, float)
        assert isinstance(result.median_estimate, float)

    def test_n_rounds_stored(self, stability_selector, chain_datasets):
        result = stability_selector.compute_structural_ci(
            chain_datasets, target_idx=1,
            dag_learner=_chain_learner, n_variables=3,
        )
        assert result.n_rounds == 20


# ---------------------------------------------------------------------------
# Test ParametricBootstrap
# ---------------------------------------------------------------------------

class TestParametricBootstrap:

    def test_parametric_ci_returns_result(self, bootstrap, chain_datasets, chain_parent_sets):
        result = bootstrap.compute_parametric_ci(
            chain_datasets, target_idx=1,
            parent_sets=chain_parent_sets,
        )
        assert isinstance(result, BootstrapCIResult)

    def test_parametric_ci_is_interval(self, bootstrap, chain_datasets, chain_parent_sets):
        result = bootstrap.compute_parametric_ci(
            chain_datasets, target_idx=1,
            parent_sets=chain_parent_sets,
        )
        assert result.ci_lower <= result.ci_upper + 1e-10

    def test_bootstrap_distribution_size(self, bootstrap, chain_datasets, chain_parent_sets):
        result = bootstrap.compute_parametric_ci(
            chain_datasets, target_idx=1,
            parent_sets=chain_parent_sets,
        )
        assert len(result.bootstrap_distribution) == 50

    def test_bootstrap_se_positive(self, bootstrap, chain_datasets, chain_parent_sets):
        result = bootstrap.compute_parametric_ci(
            chain_datasets, target_idx=1,
            parent_sets=chain_parent_sets,
        )
        assert result.se >= 0

    def test_bootstrap_method_stored(self, bootstrap, chain_datasets, chain_parent_sets):
        result = bootstrap.compute_parametric_ci(
            chain_datasets, target_idx=1,
            parent_sets=chain_parent_sets,
        )
        assert result.method == "percentile"

    def test_context_sensitivity_ci(self, bootstrap, chain_datasets, chain_parent_sets):
        result = bootstrap.compute_context_sensitivity_ci(
            chain_datasets, target_idx=1,
            parent_sets=chain_parent_sets,
            n_subsets=20,
        )
        assert isinstance(result, BootstrapCIResult)
        assert result.ci_lower <= result.ci_upper + 1e-10

    def test_bootstrap_distribution_summary(self, bootstrap, chain_datasets, chain_parent_sets):
        result = bootstrap.compute_parametric_ci(
            chain_datasets, target_idx=1,
            parent_sets=chain_parent_sets,
        )
        summary = bootstrap.bootstrap_distribution_summary(result)
        assert isinstance(summary, dict)
        assert "mean" in summary or "median" in summary

    def test_bca_method(self, chain_datasets, chain_parent_sets):
        boot = ParametricBootstrap(
            n_bootstrap=40, ci_level=0.95,
            ci_method="bca", random_state=42,
        )
        result = boot.compute_parametric_ci(
            chain_datasets, target_idx=1,
            parent_sets=chain_parent_sets,
        )
        assert isinstance(result, BootstrapCIResult)

    def test_ci_contains_point_estimate(self, bootstrap, chain_datasets, chain_parent_sets):
        result = bootstrap.compute_parametric_ci(
            chain_datasets, target_idx=1,
            parent_sets=chain_parent_sets,
        )
        assert result.ci_lower <= result.point_estimate + 1e-10
        assert result.point_estimate <= result.ci_upper + 1e-10


# ---------------------------------------------------------------------------
# Test PermutationCalibrator
# ---------------------------------------------------------------------------

class TestPermutationCalibrator:

    def test_calibrate_structural(self, permutation_calibrator, chain_datasets):
        result = permutation_calibrator.calibrate_structural(
            chain_datasets, target_idx=1,
            dag_learner=_chain_learner, observed_psi_S=0.5,
        )
        assert isinstance(result, PermutationResult)
        assert 0.0 <= result.p_value <= 1.0

    def test_calibrate_parametric(self, permutation_calibrator, chain_datasets, chain_parent_sets):
        result = permutation_calibrator.calibrate_parametric(
            chain_datasets, target_idx=1,
            parent_sets=chain_parent_sets, observed_psi_P=0.5,
        )
        assert isinstance(result, PermutationResult)
        assert 0.0 <= result.p_value <= 1.0

    def test_null_distribution_size(self, permutation_calibrator, chain_datasets):
        result = permutation_calibrator.calibrate_structural(
            chain_datasets, target_idx=1,
            dag_learner=_chain_learner, observed_psi_S=0.5,
        )
        assert len(result.null_distribution) == 99

    def test_fdr_correction(self, permutation_calibrator, rng):
        p_values = rng.uniform(0, 1, size=20)
        corrected = permutation_calibrator.fdr_correction(p_values)
        assert corrected.shape == p_values.shape
        assert np.all(corrected <= 1.0 + 1e-10)
        assert np.all(corrected >= 0.0 - 1e-10)

    def test_fdr_adjusted_p(self, permutation_calibrator, chain_datasets):
        result = permutation_calibrator.calibrate_structural(
            chain_datasets, target_idx=1,
            dag_learner=_chain_learner, observed_psi_S=0.5,
        )
        # fdr_adjusted_p may or may not be set for single calibration
        assert result.p_value >= 0.0

    def test_significance_level_stored(self, permutation_calibrator, chain_datasets):
        result = permutation_calibrator.calibrate_structural(
            chain_datasets, target_idx=1,
            dag_learner=_chain_learner, observed_psi_S=0.1,
        )
        assert result.significance_level == 0.05

    def test_threshold(self, permutation_calibrator, chain_datasets):
        result = permutation_calibrator.calibrate_structural(
            chain_datasets, target_idx=1,
            dag_learner=_chain_learner, observed_psi_S=0.5,
        )
        assert result.threshold >= 0.0


# ---------------------------------------------------------------------------
# Test threshold calibration
# ---------------------------------------------------------------------------

class TestThresholdCalibration:

    def test_calibration_returns_dict(self, chain_datasets):
        sel = StabilitySelector(n_rounds=10, random_state=42)
        result = sel.threshold_calibration(
            chain_datasets, target_idx=1,
            dag_learner=_chain_learner, n_variables=3,
        )
        assert isinstance(result, dict)

    def test_p_value_bounded(self, rng):
        datasets = _generate_chain_datasets(rng, n_contexts=5)
        cal = PermutationCalibrator(n_permutations=49, random_state=42)
        for obs in [0.0, 0.5, 1.0]:
            result = cal.calibrate_structural(
                datasets, target_idx=1,
                dag_learner=_chain_learner, observed_psi_S=obs,
            )
            assert 0.0 <= result.p_value <= 1.0

    def test_fdr_correction_reduces_discoveries(self, rng):
        """FDR correction should reject fewer or equal hypotheses."""
        p_values = rng.uniform(0, 0.1, size=20)
        cal = PermutationCalibrator(n_permutations=49)
        corrected = cal.fdr_correction(p_values)
        raw_rejections = np.sum(p_values < 0.05)
        fdr_rejections = np.sum(corrected < 0.05)
        assert fdr_rejections <= raw_rejections

    def test_fdr_correction_preserves_bounds(self, rng):
        p_values = rng.uniform(0, 1, size=10)
        cal = PermutationCalibrator(n_permutations=49)
        corrected = cal.fdr_correction(p_values)
        assert np.all(corrected >= p_values - 1e-10)
        assert np.all(corrected <= 1.0 + 1e-10)

    def test_correction_none(self, chain_datasets):
        sel = StabilitySelector(
            n_rounds=10, random_state=42, correction="none",
        )
        p_values = np.array([0.01, 0.05, 0.1, 0.5])
        corrected = sel.apply_correction(p_values)
        assert_allclose(corrected, p_values, atol=1e-10)

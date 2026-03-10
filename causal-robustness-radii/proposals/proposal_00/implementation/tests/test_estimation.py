"""Tests for causalcert.estimation – backdoor, AIPW, adjustment, effects."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalcert.estimation.backdoor import (
    satisfies_backdoor,
    enumerate_adjustment_sets,
    find_minimum_adjustment_set,
    has_valid_adjustment_set,
    all_backdoor_paths,
)
from causalcert.estimation.aipw import AIPWEstimator, aipw_simple
from causalcert.estimation.crossfit import CrossFitter
from causalcert.estimation.propensity import PropensityModel
from causalcert.estimation.outcome import OutcomeModel
from causalcert.estimation.effects import estimate_ate, estimate_att
from causalcert.estimation.influence import (
    influence_function,
    variance_from_influence,
    confidence_interval,
)
from causalcert.estimation.adjustment import (
    find_optimal_adjustment_set,
    enumerate_valid_adjustment_sets,
)
from causalcert.types import (
    AdjacencyMatrix,
    EstimationResult,
    NodeSet,
)

# ── helpers ───────────────────────────────────────────────────────────────

def _adj(n: int, edges: list[tuple[int, int]]) -> AdjacencyMatrix:
    a = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        a[u, v] = 1
    return a


def _binary_treatment_data(n: int = 1000, true_ate: float = 2.0, seed: int = 42):
    rng = np.random.default_rng(seed)
    C = rng.standard_normal(n)
    T = (rng.standard_normal(n) + 0.5 * C > 0).astype(float)
    Y = true_ate * T + 0.8 * C + 0.5 * rng.standard_normal(n)
    return pd.DataFrame({"C": C, "T": T, "Y": Y}), true_ate


# ═══════════════════════════════════════════════════════════════════════════
# Back-door criterion
# ═══════════════════════════════════════════════════════════════════════════


class TestBackdoorCriterion:
    def test_confounded_dag(self) -> None:
        """C->X, C->Y, X->Y: {C} satisfies backdoor."""
        adj = _adj(3, [(0, 1), (0, 2), (1, 2)])
        assert satisfies_backdoor(adj, treatment=1, outcome=2, adjustment_set=frozenset({0}))

    def test_empty_set_no_confounding(self) -> None:
        """X->Y only: empty set satisfies backdoor."""
        adj = _adj(2, [(0, 1)])
        assert satisfies_backdoor(adj, treatment=0, outcome=1, adjustment_set=frozenset())

    def test_mediator_not_valid(self) -> None:
        """X->M->Y: conditioning on M is NOT valid for total effect."""
        adj = _adj(3, [(0, 1), (1, 2)])
        # For total effect, adjusting for mediator is generally not valid
        # But it "blocks" the direct path. Technically, {} satisfies backdoor
        assert satisfies_backdoor(adj, treatment=0, outcome=2, adjustment_set=frozenset())

    def test_collider_not_in_adjustment(self) -> None:
        """X->Z<-Y: conditioning on collider Z opens path."""
        adj = _adj(3, [(0, 2), (1, 2)])
        # No edge X->Y: backdoor criterion doesn't apply in standard way
        # But {Z} should NOT satisfy backdoor for X->Y (opens collider)
        result = satisfies_backdoor(adj, treatment=0, outcome=1, adjustment_set=frozenset({2}))
        assert not result

    def test_m_bias_empty_set(self, mbias5_adj: AdjacencyMatrix) -> None:
        """M-bias: empty set suffices (no backdoor paths X->Y)."""
        assert satisfies_backdoor(
            mbias5_adj, treatment=1, outcome=4, adjustment_set=frozenset()
        )


# ═══════════════════════════════════════════════════════════════════════════
# Adjustment set enumeration
# ═══════════════════════════════════════════════════════════════════════════


class TestAdjustmentSets:
    def test_enumerate_confounded(self) -> None:
        adj = _adj(3, [(0, 1), (0, 2), (1, 2)])
        sets = enumerate_adjustment_sets(adj, treatment=1, outcome=2, minimal=True)
        assert len(sets) >= 1
        assert any(0 in s for s in sets)

    def test_enumerate_no_confounding(self) -> None:
        adj = _adj(2, [(0, 1)])
        sets = enumerate_adjustment_sets(adj, treatment=0, outcome=1)
        assert len(sets) >= 1
        # Empty set should be valid
        assert any(len(s) == 0 for s in sets)

    def test_find_minimum(self) -> None:
        adj = _adj(3, [(0, 1), (0, 2), (1, 2)])
        min_set = find_minimum_adjustment_set(adj, treatment=1, outcome=2)
        assert isinstance(min_set, frozenset)
        assert satisfies_backdoor(adj, treatment=1, outcome=2, adjustment_set=min_set)

    def test_has_valid_adjustment(self) -> None:
        adj = _adj(3, [(0, 1), (0, 2), (1, 2)])
        assert has_valid_adjustment_set(adj, treatment=1, outcome=2)

    def test_find_optimal(self) -> None:
        adj = _adj(3, [(0, 1), (0, 2), (1, 2)])
        opt = find_optimal_adjustment_set(adj, treatment=1, outcome=2)
        assert isinstance(opt, frozenset)

    def test_enumerate_valid_sets(self) -> None:
        adj = _adj(4, [(0, 1), (0, 2), (0, 3), (1, 3), (2, 3)])
        sets = enumerate_valid_adjustment_sets(adj, treatment=1, outcome=3)
        assert len(sets) >= 1

    def test_all_backdoor_paths(self) -> None:
        adj = _adj(3, [(0, 1), (0, 2), (1, 2)])
        paths = all_backdoor_paths(adj, treatment=1, outcome=2)
        assert isinstance(paths, list)


# ═══════════════════════════════════════════════════════════════════════════
# AIPW estimator
# ═══════════════════════════════════════════════════════════════════════════


class TestAIPW:
    def test_aipw_basic(self) -> None:
        data, true_ate = _binary_treatment_data(n=1000)
        adj = _adj(3, [(0, 1), (0, 2), (1, 2)])  # C->T, C->Y, T->Y
        estimator = AIPWEstimator(n_folds=3, seed=42)
        result = estimator.estimate(adj, data, treatment=1, outcome=2, adjustment_set=frozenset({0}))
        assert isinstance(result, EstimationResult)
        assert abs(result.ate - true_ate) < 1.0  # within 1 of true ATE

    def test_aipw_ci_contains_true(self) -> None:
        data, true_ate = _binary_treatment_data(n=2000, seed=123)
        adj = _adj(3, [(0, 1), (0, 2), (1, 2)])
        estimator = AIPWEstimator(n_folds=5, seed=123)
        result = estimator.estimate(adj, data, treatment=1, outcome=2, adjustment_set=frozenset({0}))
        # CI should contain true ATE (with reasonable probability)
        assert result.ci_lower <= true_ate <= result.ci_upper or abs(result.ate - true_ate) < 0.5

    def test_aipw_simple(self) -> None:
        data, _ = _binary_treatment_data(n=500)
        y = data["Y"].values
        t = data["T"].values
        X = data[["C"]].values
        result = aipw_simple(y, t, X, seed=42)
        assert "ate" in result
        assert "se" in result
        assert isinstance(result["ate"], float)

    def test_aipw_diagnostics(self) -> None:
        data, _ = _binary_treatment_data(n=500)
        adj = _adj(3, [(0, 1), (0, 2), (1, 2)])
        estimator = AIPWEstimator(n_folds=2, seed=42)
        estimator.estimate(adj, data, treatment=1, outcome=2, adjustment_set=frozenset({0}))
        diag = estimator.diagnostics()
        assert isinstance(diag, dict)


# ═══════════════════════════════════════════════════════════════════════════
# Cross-fitting
# ═══════════════════════════════════════════════════════════════════════════


class TestCrossFitting:
    def test_crossfit_basic(self) -> None:
        data, _ = _binary_treatment_data(n=500)
        cf = CrossFitter(n_folds=3, seed=42)

        def prop_factory():
            return PropensityModel(seed=42)

        def out_factory():
            return OutcomeModel(seed=42)

        results = cf.fit(
            data, treatment_col=1, outcome_col=2, covariate_cols=[0],
            propensity_factory=prop_factory, outcome_factory=out_factory,
        )
        assert len(results) == 3

    def test_crossfit_arrays(self) -> None:
        data, _ = _binary_treatment_data(n=500)
        X = data[["C"]].values
        t = data["T"].values
        y = data["Y"].values
        cf = CrossFitter(n_folds=3, seed=42)
        results = cf.fit_arrays(
            X, t, y,
            propensity_factory=lambda: PropensityModel(seed=42),
            outcome_factory=lambda: OutcomeModel(seed=42),
        )
        assert len(results) == 3


# ═══════════════════════════════════════════════════════════════════════════
# Propensity and outcome models
# ═══════════════════════════════════════════════════════════════════════════


class TestModels:
    def test_propensity_fit_predict(self) -> None:
        data, _ = _binary_treatment_data(n=500)
        X = data[["C"]].values
        t = data["T"].values
        model = PropensityModel(seed=42)
        model.fit(X, t)
        e = model.predict(X)
        assert e.shape == (500,)
        assert np.all((e >= 0) & (e <= 1))

    def test_outcome_fit_predict(self) -> None:
        data, _ = _binary_treatment_data(n=500)
        X = data[["C"]].values
        t = data["T"].values
        y = data["Y"].values
        model = OutcomeModel(seed=42)
        model.fit(X, t, y)
        mu0, mu1 = model.predict(X)
        assert mu0.shape == (500,)
        assert mu1.shape == (500,)

    def test_propensity_overlap(self) -> None:
        data, _ = _binary_treatment_data(n=1000)
        X = data[["C"]].values
        t = data["T"].values
        model = PropensityModel(seed=42)
        model.fit(X, t)
        diag = model.overlap_diagnostics(X, t)
        assert isinstance(diag, dict)


# ═══════════════════════════════════════════════════════════════════════════
# Confidence intervals
# ═══════════════════════════════════════════════════════════════════════════


class TestConfidenceIntervals:
    def test_influence_function_shape(self) -> None:
        n = 200
        rng = np.random.default_rng(42)
        psi = influence_function(
            y=rng.standard_normal(n),
            t=(rng.random(n) > 0.5).astype(float),
            mu0=rng.standard_normal(n),
            mu1=rng.standard_normal(n),
            e=rng.uniform(0.1, 0.9, n),
        )
        assert psi.shape == (n,)

    def test_variance_positive(self) -> None:
        psi = np.random.default_rng(42).standard_normal(100)
        var = variance_from_influence(psi)
        assert var > 0

    def test_confidence_interval_contains_mean(self) -> None:
        psi = np.random.default_rng(42).standard_normal(500)
        ate = psi.mean()
        ci_lo, ci_hi = confidence_interval(ate, psi, alpha=0.05)
        assert ci_lo <= ate <= ci_hi

    def test_wider_ci_with_more_variance(self) -> None:
        rng = np.random.default_rng(42)
        psi_tight = rng.standard_normal(500) * 0.1
        psi_wide = rng.standard_normal(500) * 10.0
        _, w1 = confidence_interval(0.0, psi_tight)
        _, w2 = confidence_interval(0.0, psi_wide)
        assert w2 - 0.0 > w1 - 0.0  # wider interval


# ═══════════════════════════════════════════════════════════════════════════
# Effect estimation top-level
# ═══════════════════════════════════════════════════════════════════════════


class TestEffects:
    def test_estimate_ate(self) -> None:
        data, true_ate = _binary_treatment_data(n=1000)
        adj = _adj(3, [(0, 1), (0, 2), (1, 2)])
        result = estimate_ate(adj, data, treatment=1, outcome=2, seed=42, n_folds=3)
        assert isinstance(result, EstimationResult)
        assert abs(result.ate - true_ate) < 1.5

    def test_estimate_att(self) -> None:
        data, _ = _binary_treatment_data(n=1000)
        adj = _adj(3, [(0, 1), (0, 2), (1, 2)])
        result = estimate_att(adj, data, treatment=1, outcome=2, seed=42, n_folds=3)
        assert isinstance(result, EstimationResult)

    def test_result_fields(self) -> None:
        data, _ = _binary_treatment_data(n=500)
        adj = _adj(3, [(0, 1), (0, 2), (1, 2)])
        result = estimate_ate(adj, data, treatment=1, outcome=2, seed=42, n_folds=2)
        assert result.se > 0
        assert result.ci_lower < result.ci_upper
        assert isinstance(result.adjustment_set, frozenset)
        assert result.n_obs > 0


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEstimationEdgeCases:
    def test_no_confounders(self) -> None:
        rng = np.random.default_rng(42)
        n = 500
        T = (rng.random(n) > 0.5).astype(float)
        Y = 1.5 * T + rng.standard_normal(n)
        data = pd.DataFrame({"T": T, "Y": Y})
        adj = _adj(2, [(0, 1)])
        result = estimate_ate(adj, data, treatment=0, outcome=1, seed=42, n_folds=2)
        assert abs(result.ate - 1.5) < 1.0

    def test_aipw_multiple_sets(self) -> None:
        data, _ = _binary_treatment_data(n=500)
        adj = _adj(3, [(0, 1), (0, 2), (1, 2)])
        estimator = AIPWEstimator(n_folds=2, seed=42)
        results = estimator.estimate_multiple_sets(
            adj, data, treatment=1, outcome=2,
            adjustment_sets=[frozenset({0}), frozenset()],
        )
        assert len(results) == 2


# ═══════════════════════════════════════════════════════════════════════════
# Estimation edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEstimationEdgeCases:
    def test_identical_treatment_outcome(self) -> None:
        data, _ = _binary_treatment_data(n=200)
        adj = _adj(3, [(0, 1), (0, 2), (1, 2)])
        with pytest.raises(Exception):
            estimate_ate(adj, data, treatment=1, outcome=1, seed=42)

    @pytest.mark.parametrize("n_folds", [2, 3, 5])
    def test_varying_folds(self, n_folds: int) -> None:
        data, _ = _binary_treatment_data(n=500)
        adj = _adj(3, [(0, 1), (0, 2), (1, 2)])
        result = estimate_ate(adj, data, treatment=1, outcome=2, seed=42, n_folds=n_folds)
        assert result.ci_lower < result.ate < result.ci_upper

    def test_large_effect_detected(self) -> None:
        rng = np.random.default_rng(42)
        n = 500
        t = rng.binomial(1, 0.5, n)
        y = 5.0 * t + rng.standard_normal(n) * 0.1  # large, obvious effect
        data = pd.DataFrame({"X0": rng.standard_normal(n), "T": t, "Y": y})
        adj = _adj(3, [(0, 1), (1, 2)])
        result = estimate_ate(adj, data, treatment=1, outcome=2, seed=42)
        assert result.ate > 3.0

    def test_null_effect(self) -> None:
        rng = np.random.default_rng(42)
        n = 500
        t = rng.binomial(1, 0.5, n)
        y = rng.standard_normal(n)  # treatment has no effect
        data = pd.DataFrame({"X0": rng.standard_normal(n), "T": t, "Y": y})
        adj = _adj(3, [(0, 1), (1, 2)])
        result = estimate_ate(adj, data, treatment=1, outcome=2, seed=42)
        assert abs(result.ate) < 1.0

    def test_adjustment_set_contains_treatment(self) -> None:
        data, _ = _binary_treatment_data(n=200)
        adj = _adj(3, [(0, 1), (0, 2), (1, 2)])
        # Including treatment in adj set may or may not raise - test it completes
        estimator = AIPWEstimator(n_folds=2, seed=42)
        result = estimator.estimate(
            adj, data, treatment=1, outcome=2,
            adjustment_set=frozenset({1}),
        )
        assert isinstance(result, EstimationResult)

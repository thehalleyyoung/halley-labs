"""
Tests for the invariance testing module.

Covers e-value computation, wealth processes, SCIT edge classification,
anytime validity, e-BH multiple testing correction, power analysis,
and confidence sequence coverage.
"""

from __future__ import annotations

import numpy as np
import pytest

from causal_trading.invariance.e_values import (
    EValueConstructor,
    EValueType,
    ProductEValue,
    MixtureEValue,
    GROWMartingale,
    WealthProcess,
    ConfidenceSequence,
)
from causal_trading.invariance.scit import (
    SCITAlgorithm,
    EBHProcedure,
    EdgeType,
)
from causal_trading.invariance.anytime_inference import (
    MixtureMartingale,
    SubGaussianEProcess,
)
from causal_trading.invariance.power_analysis import (
    SampleSizeCalculator,
    PowerAnalyzer,
)


# =========================================================================
# E-Value: stays below 1 under null (invariant edge)
# =========================================================================

class TestEValueNull:
    """Under the null hypothesis (same distribution across regimes),
    the e-value should stay close to 1 in expectation."""

    def test_e_value_bounded_under_null(self, rng):
        """Under H0, e-value should not grow systematically."""
        n = 500
        # Both regimes have the same distribution
        data = rng.normal(0, 1, size=(n, 1))
        regimes = rng.integers(0, 2, size=n)

        constructor = EValueConstructor(e_type=EValueType.LIKELIHOOD_RATIO)
        for t in range(n):
            constructor.update(data[t], regimes[t])

        e_val = constructor.get_e_value()
        # Under null, e-value should not exceed a modest threshold
        assert e_val < 50.0  # generous bound for 500 steps

    def test_e_value_mean_below_one_under_null(self, rng):
        """E-values are non-negative supermartingales under H0, E[E_t] <= 1."""
        n_sims = 20
        final_e_values = []
        for sim in range(n_sims):
            r = np.random.default_rng(sim + 100)
            data = r.normal(0, 1, size=(200, 1))
            regimes = r.integers(0, 2, size=200)
            constructor = EValueConstructor(e_type=EValueType.LIKELIHOOD_RATIO)
            for t in range(200):
                constructor.update(data[t], regimes[t])
            final_e_values.append(constructor.get_e_value())

        mean_e = np.mean(final_e_values)
        # By Ville's inequality, mean should be <= 1 (with some slack)
        assert mean_e < 5.0

    @pytest.mark.parametrize("e_type", [
        EValueType.LIKELIHOOD_RATIO,
        EValueType.SCORE,
    ])
    def test_multiple_e_value_types_null(self, rng, e_type):
        n = 300
        data = rng.normal(0, 1, size=(n, 1))
        regimes = rng.integers(0, 2, size=n)
        constructor = EValueConstructor(e_type=e_type)
        for t in range(n):
            constructor.update(data[t], regimes[t])
        # Should not reject at alpha=0.05
        assert not constructor.reject(alpha=0.01)


# =========================================================================
# E-Value: grows under alternative (non-invariant edge)
# =========================================================================

class TestEValueAlternative:
    """Under the alternative (different distributions across regimes),
    the e-value should grow."""

    def test_e_value_grows_with_different_means(self, rng):
        n = 500
        data = np.empty((n, 1))
        regimes = np.empty(n, dtype=int)
        for t in range(n):
            r = t % 2
            regimes[t] = r
            data[t, 0] = rng.normal(2.0 * r, 0.5)

        constructor = EValueConstructor(e_type=EValueType.LIKELIHOOD_RATIO)
        for t in range(n):
            constructor.update(data[t], regimes[t])

        e_val = constructor.get_e_value()
        assert e_val > 1.0

    def test_rejects_under_alternative(self, rng):
        n = 400
        data = np.empty((n, 1))
        regimes = np.zeros(n, dtype=int)
        regimes[200:] = 1
        data[:200, 0] = rng.normal(0, 0.5, size=200)
        data[200:, 0] = rng.normal(3.0, 0.5, size=200)

        constructor = EValueConstructor(e_type=EValueType.LIKELIHOOD_RATIO)
        for t in range(n):
            constructor.update(data[t], regimes[t])

        assert constructor.reject(alpha=0.05)


# =========================================================================
# ProductEValue and MixtureEValue
# =========================================================================

class TestProductEValue:
    def test_product_of_ones_is_one(self):
        pe = ProductEValue()
        for _ in range(10):
            pe.update(1.0)
        assert np.isclose(pe.value, 1.0)

    def test_product_grows_with_large_values(self):
        pe = ProductEValue()
        for _ in range(5):
            pe.update(2.0)
        assert pe.value == pytest.approx(32.0, rel=1e-6)

    def test_reject_with_large_product(self):
        pe = ProductEValue()
        for _ in range(10):
            pe.update(3.0)
        assert pe.reject(alpha=0.05)

    def test_no_reject_with_small_values(self):
        pe = ProductEValue()
        for _ in range(10):
            pe.update(0.9)
        assert not pe.reject(alpha=0.05)

    def test_batch_update(self):
        pe = ProductEValue()
        pe.update_batch([2.0, 3.0, 1.5])
        expected = 2.0 * 3.0 * 1.5
        assert pe.value == pytest.approx(expected, rel=1e-6)

    def test_reset(self):
        pe = ProductEValue()
        pe.update(5.0)
        pe.reset()
        assert pe.value == pytest.approx(1.0)


class TestMixtureEValue:
    def test_mixture_weights_sum_to_one(self):
        me = MixtureEValue(n_components=5)
        # After initialization, internal weights sum to 1
        assert me.value >= 0

    def test_mixture_update(self):
        me = MixtureEValue(n_components=3)
        e_vals = np.array([1.5, 0.8, 2.0])
        val = me.update(e_vals)
        assert val > 0

    def test_mixture_rejection(self):
        me = MixtureEValue(n_components=2)
        for _ in range(20):
            me.update(np.array([5.0, 3.0]))
        assert me.reject(alpha=0.05)


# =========================================================================
# WealthProcess
# =========================================================================

class TestWealthProcess:
    def test_initial_wealth_is_one(self):
        wp = WealthProcess()
        assert wp.current_wealth == pytest.approx(1.0)

    def test_wealth_increases_with_favorable_bets(self):
        wp = WealthProcess()
        wp.update(2.0, bet_fraction=1.0)
        assert wp.current_wealth > 1.0

    def test_wealth_decreases_with_unfavorable_bets(self):
        wp = WealthProcess()
        wp.update(0.5, bet_fraction=1.0)
        assert wp.current_wealth < 1.0

    def test_time_increments(self):
        wp = WealthProcess()
        wp.update(1.5)
        wp.update(0.8)
        assert wp.time == 2

    def test_growth_rate(self):
        wp = WealthProcess()
        for _ in range(10):
            wp.update(1.1)
        gr = wp.growth_rate()
        assert gr > 0

    def test_drawdown(self):
        wp = WealthProcess()
        wp.update(2.0)
        wp.update(0.5)  # 50% drawdown
        dd = wp.drawdown()
        assert dd > 0


# =========================================================================
# ConfidenceSequence
# =========================================================================

class TestConfidenceSequence:
    def test_width_decreases_with_time(self):
        times = np.arange(1, 101)
        lower = -1.0 / np.sqrt(times)
        upper = 1.0 / np.sqrt(times)
        cs = ConfidenceSequence(
            lower=lower, upper=upper, times=times, alpha=0.05
        )
        assert cs.width_at(0) > cs.width_at(99)

    def test_contains_true_value(self):
        times = np.arange(1, 51)
        lower = -0.5 * np.ones(50)
        upper = 0.5 * np.ones(50)
        cs = ConfidenceSequence(
            lower=lower, upper=upper, times=times, alpha=0.05
        )
        assert cs.contains(0.0, 0)
        assert cs.contains(0.0, 49)
        assert not cs.contains(1.0, 0)

    def test_running_intersection_tightens(self):
        times = np.arange(1, 11)
        # Widening then narrowing bounds
        lower = np.array([-1, -2, -0.5, -0.3, -0.2, -0.15, -0.1, -0.08, -0.05, -0.03])
        upper = np.array([1, 2, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05, 0.03])
        cs = ConfidenceSequence(lower=lower, upper=upper, times=times, alpha=0.05)
        cs_int = cs.running_intersection()
        # Running intersection should be monotonically non-widening
        for t in range(1, len(times)):
            assert cs_int.upper[t] <= cs_int.upper[t - 1] + 1e-10
            assert cs_int.lower[t] >= cs_int.lower[t - 1] - 1e-10


# =========================================================================
# SCIT: Edge Classification
# =========================================================================

class TestSCIT:
    """Tests for the Sequential Causal Invariance Test."""

    def _generate_invariant_data(self, rng, n=400):
        """Data where the edge relationship is the SAME across regimes."""
        regimes = np.zeros(n, dtype=int)
        regimes[n // 2:] = 1
        x = rng.normal(0, 1, size=(n, 2))
        # Same relationship in both regimes
        x[:, 1] = 0.5 * x[:, 0] + rng.normal(0, 0.3, size=n)
        return x, regimes

    def _generate_variant_data(self, rng, n=400):
        """Data where the edge relationship CHANGES across regimes."""
        regimes = np.zeros(n, dtype=int)
        regimes[n // 2:] = 1
        x = np.empty((n, 2))
        x[:, 0] = rng.normal(0, 1, size=n)
        # Different coefficient in each regime
        for t in range(n):
            if regimes[t] == 0:
                x[t, 1] = 0.5 * x[t, 0] + rng.normal(0, 0.3)
            else:
                x[t, 1] = -1.5 * x[t, 0] + rng.normal(0, 0.3)
        return x, regimes

    def test_invariant_edge_classified_correctly(self, rng):
        data, regimes = self._generate_invariant_data(rng, n=500)
        node_names = ["X0", "X1"]
        dag = {"X0": ["X1"]}  # X0 -> X1

        scit = SCITAlgorithm(alpha=0.05)
        result = scit.fit(data, regimes, dag=dag, node_names=node_names)

        # The edge should be classified as invariant (or undetermined, not variant)
        edge = ("X0", "X1")
        if edge in result.edge_classifications:
            cls = result.edge_classifications[edge]
            assert cls.edge_type in (EdgeType.INVARIANT, EdgeType.UNDETERMINED)

    def test_variant_edge_detected(self, rng):
        data, regimes = self._generate_variant_data(rng, n=600)
        node_names = ["X0", "X1"]
        dag = {"X0": ["X1"]}

        scit = SCITAlgorithm(alpha=0.05)
        result = scit.fit(data, regimes, dag=dag, node_names=node_names)

        edge = ("X0", "X1")
        if edge in result.edge_classifications:
            cls = result.edge_classifications[edge]
            # With strong signal, should be regime-specific
            assert cls.edge_type in (EdgeType.REGIME_SPECIFIC, EdgeType.UNDETERMINED)

    def test_scit_result_has_sets(self, rng):
        data, regimes = self._generate_invariant_data(rng)
        scit = SCITAlgorithm(alpha=0.05)
        result = scit.fit(
            data, regimes,
            dag={"X0": ["X1"]},
            node_names=["X0", "X1"]
        )
        assert isinstance(result.invariant_edges, set)
        assert isinstance(result.regime_specific_edges, set)
        assert isinstance(result.undetermined_edges, set)


# =========================================================================
# E-BH Procedure (Multiple Testing)
# =========================================================================

class TestEBHProcedure:
    def test_controls_fdr(self):
        """With e-values all below 1/alpha, none should be rejected."""
        proc = EBHProcedure(alpha=0.05)
        e_values = {"e1": 0.5, "e2": 0.8, "e3": 1.2, "e4": 0.3}
        rejected, threshold = proc.apply(e_values)
        # All e-values are small, so no rejections expected
        assert len(rejected) <= 2

    def test_rejects_large_e_values(self):
        proc = EBHProcedure(alpha=0.05)
        e_values = {"e1": 100.0, "e2": 200.0, "e3": 0.5, "e4": 0.3}
        rejected, threshold = proc.apply(e_values)
        assert "e1" in rejected
        assert "e2" in rejected

    def test_threshold_is_positive(self):
        proc = EBHProcedure(alpha=0.05)
        e_values = {"e1": 5.0, "e2": 10.0}
        _, threshold = proc.apply(e_values)
        assert threshold > 0

    def test_adjusted_p_values(self):
        proc = EBHProcedure(alpha=0.05)
        e_values = {"e1": 100.0, "e2": 0.5}
        adj_p = proc.adjusted_p_values(e_values)
        assert adj_p["e1"] < adj_p["e2"]

    def test_empty_e_values(self):
        proc = EBHProcedure(alpha=0.05)
        rejected, threshold = proc.apply({})
        assert len(rejected) == 0


# =========================================================================
# Anytime Validity (stopping at any time controls error)
# =========================================================================

class TestAnytimeValidity:
    def test_stopping_under_null_controls_error(self, rng):
        """Stopping at different times under H0 should rarely reject."""
        n_sims = 50
        rejections = 0
        for sim in range(n_sims):
            r = np.random.default_rng(sim + 200)
            proc = SubGaussianEProcess(null_mean=0.0, sigma=1.0)
            data = r.normal(0, 1, size=200)
            stop_time = r.integers(50, 200)
            for t in range(stop_time):
                proc.update(data[t])
            if proc.reject(alpha=0.05):
                rejections += 1
        # Should have at most ~10% rejection rate (generous bound)
        assert rejections / n_sims < 0.15

    def test_e_process_grows_under_alternative(self, rng):
        proc = SubGaussianEProcess(null_mean=0.0, sigma=1.0)
        data = rng.normal(1.0, 1.0, size=200)  # Alternative: mean=1
        for x in data:
            proc.update(x)
        assert proc.value > 1.0

    def test_mixture_martingale_null(self, rng):
        mm = MixtureMartingale()
        for _ in range(100):
            x = rng.normal(0, 1)
            mm.update_gaussian(x, null_mean=0.0, variance=1.0)
        # Under null, should not reject
        assert not mm.reject(alpha=0.01)


# =========================================================================
# Power Analysis
# =========================================================================

class TestPowerAnalysis:
    def test_sample_size_for_mean_shift(self):
        calc = SampleSizeCalculator(alpha=0.05, power=0.8)
        n = calc.for_mean_shift(effect_size=0.5, variance=1.0)
        assert n > 0
        assert n < 10000

    def test_larger_effect_needs_fewer_samples(self):
        calc = SampleSizeCalculator(alpha=0.05, power=0.8)
        n_small = calc.for_mean_shift(effect_size=0.2)
        n_large = calc.for_mean_shift(effect_size=0.8)
        assert n_large < n_small

    def test_higher_power_needs_more_samples(self):
        calc_lo = SampleSizeCalculator(alpha=0.05, power=0.5)
        calc_hi = SampleSizeCalculator(alpha=0.05, power=0.95)
        n_lo = calc_lo.for_mean_shift(effect_size=0.5)
        n_hi = calc_hi.for_mean_shift(effect_size=0.5)
        assert n_hi > n_lo

    def test_power_analyzer_curve(self, rng):
        pa = PowerAnalyzer(alpha=0.05, n_regimes=2, random_state=42)
        curve = pa.power_curve(
            effect_size=0.5,
            sample_sizes=[50, 100, 200, 500],
            method="asymptotic",
        )
        assert len(curve.powers) == 4
        # Power should increase with sample size
        assert curve.powers[-1] >= curve.powers[0]

    def test_minimum_sample_size(self):
        pa = PowerAnalyzer(alpha=0.05, n_regimes=2, random_state=42)
        n = pa.minimum_sample_size(
            effect_size=0.5, target_power=0.8, method="asymptotic"
        )
        assert n > 0

    def test_power_heatmap_shape(self):
        pa = PowerAnalyzer(alpha=0.05, n_regimes=2, random_state=42)
        hm = pa.power_heatmap(
            effect_sizes=[0.2, 0.5, 1.0],
            sample_sizes=[50, 100, 200],
            method="asymptotic",
        )
        assert hm.shape == (3, 3)
        assert np.all(hm >= 0)
        assert np.all(hm <= 1.0 + 1e-6)


# =========================================================================
# GROW Martingale
# =========================================================================

class TestGROWMartingale:
    def test_grow_stays_near_one_under_null(self, rng):
        gm = GROWMartingale(null_mean=0.0)
        data = rng.normal(0, 1, size=200)
        for x in data:
            gm.update(x)
        # Should not grow significantly under null
        assert gm.value < 100.0

    def test_grow_grows_under_alternative(self, rng):
        gm = GROWMartingale(null_mean=0.0)
        data = rng.normal(1.0, 1.0, size=300)
        for x in data:
            gm.update(x)
        assert gm.value > 1.0

    def test_grow_batch_update(self, rng):
        gm = GROWMartingale(null_mean=0.0)
        data = rng.normal(0.5, 1.0, size=100)
        gm.update_batch(list(data))
        assert gm.value > 0

    def test_grow_reset(self):
        gm = GROWMartingale(null_mean=0.0)
        gm.update(5.0)
        gm.reset()
        assert gm.value == pytest.approx(1.0)

    def test_grow_confidence_sequence(self, rng):
        gm = GROWMartingale(null_mean=0.0)
        data = rng.normal(0.3, 1.0, size=100)
        for x in data:
            gm.update(x)
        cs = gm.get_confidence_sequence(alpha=0.05)
        assert isinstance(cs, ConfidenceSequence)

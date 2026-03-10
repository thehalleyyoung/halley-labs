"""
Unit tests for usability_oracle.algebra.context — Context modulation Δ.

Tests the ContextModulator and CognitiveContext classes which model how
cognitive state factors (fatigue, working-memory load, practice, stress,
age) modulate the cost of UI task steps.

Key modulation models under test:
- Fatigue: Weber-Fechner logarithmic degradation
- Working-memory load: Cowan capacity overload
- Practice: Power Law of Practice (Newell & Rosenbloom 1981)
- Stress: Yerkes-Dodson inverted-U arousal model
- Age: Generalised slowing (Salthouse 1996)
"""

import math
import pytest

from usability_oracle.algebra.models import CostElement
from usability_oracle.algebra.context import ContextModulator, CognitiveContext


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def modulator():
    """Return a fresh ContextModulator instance."""
    return ContextModulator()


@pytest.fixture
def base_element():
    """A baseline cost element representing a typical UI interaction."""
    return CostElement(mu=2.0, sigma_sq=0.5, kappa=0.1, lambda_=0.05)


@pytest.fixture
def unit_element():
    """A unit cost element for testing multiplicative factors."""
    return CostElement(mu=1.0, sigma_sq=0.0, kappa=0.0, lambda_=0.0)


# ---------------------------------------------------------------------------
# CognitiveContext construction and validation
# ---------------------------------------------------------------------------

class TestCognitiveContext:
    """Tests for CognitiveContext dataclass construction and validation."""

    def test_default_values(self):
        """All fields default to None except custom which is empty dict."""
        ctx = CognitiveContext()
        assert ctx.elapsed_time is None
        assert ctx.working_memory_load is None
        assert ctx.repetitions is None
        assert ctx.stress_level is None
        assert ctx.age_percentile is None
        assert ctx.custom == {}

    def test_full_construction(self):
        """All fields can be set at construction time."""
        ctx = CognitiveContext(
            elapsed_time=3600.0,
            working_memory_load=5.0,
            repetitions=10,
            stress_level=0.3,
            age_percentile=0.7,
            custom={"screen_brightness": 0.8},
        )
        assert ctx.elapsed_time == 3600.0
        assert ctx.working_memory_load == 5.0
        assert ctx.repetitions == 10
        assert math.isclose(ctx.stress_level, 0.3)
        assert math.isclose(ctx.age_percentile, 0.7)
        assert ctx.custom["screen_brightness"] == 0.8

    def test_stress_clamped_to_zero_one(self):
        """Stress level is clamped to [0, 1]."""
        ctx_low = CognitiveContext(stress_level=-0.5)
        assert ctx_low.stress_level >= 0.0
        ctx_high = CognitiveContext(stress_level=1.5)
        assert ctx_high.stress_level <= 1.0

    def test_age_percentile_clamped(self):
        """Age percentile is clamped to [0, 1]."""
        ctx_low = CognitiveContext(age_percentile=-0.2)
        assert ctx_low.age_percentile >= 0.0
        ctx_high = CognitiveContext(age_percentile=1.3)
        assert ctx_high.age_percentile <= 1.0

    def test_working_memory_load_nonneg(self):
        """Working memory load is clamped to ≥ 0."""
        ctx = CognitiveContext(working_memory_load=-1.0)
        assert ctx.working_memory_load >= 0.0

    def test_repetitions_nonneg(self):
        """Repetitions is clamped to ≥ 0."""
        ctx = CognitiveContext(repetitions=-5)
        assert ctx.repetitions >= 0

    def test_custom_dict_independent(self):
        """Custom dicts are independent across instances."""
        ctx1 = CognitiveContext(custom={"a": 1})
        ctx2 = CognitiveContext()
        assert "a" not in ctx2.custom


# ---------------------------------------------------------------------------
# Neutral context (no modulation)
# ---------------------------------------------------------------------------

class TestNeutralContext:
    """Tests that a fully-None context produces no modulation."""

    def test_neutral_context_no_change(self, modulator, base_element):
        """Modulating with all-None context returns the original element."""
        ctx = CognitiveContext()
        result = modulator.modulate(base_element, ctx)
        assert math.isclose(result.mu, base_element.mu, rel_tol=1e-10)
        assert math.isclose(result.sigma_sq, base_element.sigma_sq, rel_tol=1e-10)
        assert math.isclose(result.kappa, base_element.kappa, rel_tol=1e-10)
        assert math.isclose(result.lambda_, base_element.lambda_, rel_tol=1e-10)

    def test_modulate_returns_cost_element(self, modulator, base_element):
        """modulate() always returns a CostElement."""
        ctx = CognitiveContext(elapsed_time=1000.0)
        result = modulator.modulate(base_element, ctx)
        assert isinstance(result, CostElement)


# ---------------------------------------------------------------------------
# Fatigue modulation
# ---------------------------------------------------------------------------

class TestFatigueModulation:
    """Tests for _fatigue_modulation — Weber-Fechner degradation."""

    def test_no_fatigue_at_zero_time(self, modulator, base_element):
        """No fatigue effect when elapsed_time = 0."""
        ctx = CognitiveContext(elapsed_time=0.0)
        result = modulator.modulate(base_element, ctx)
        assert math.isclose(result.mu, base_element.mu, rel_tol=1e-10)

    def test_fatigue_increases_mu(self, modulator, base_element):
        """Fatigue increases cost (μ) over time."""
        ctx = CognitiveContext(elapsed_time=7200.0)  # 2 hours
        result = modulator.modulate(base_element, ctx)
        assert result.mu > base_element.mu

    def test_fatigue_increases_with_time(self, modulator, base_element):
        """More elapsed time → more fatigue → higher cost."""
        r1 = modulator.modulate(base_element, CognitiveContext(elapsed_time=1800.0))
        r2 = modulator.modulate(base_element, CognitiveContext(elapsed_time=7200.0))
        assert r2.mu > r1.mu

    def test_fatigue_increases_variance(self, modulator, base_element):
        """Fatigue also increases variance (more unpredictable)."""
        ctx = CognitiveContext(elapsed_time=3600.0)
        result = modulator.modulate(base_element, ctx)
        assert result.sigma_sq > base_element.sigma_sq

    def test_fatigue_rate_constant(self, modulator):
        """FATIGUE_RATE is the expected default value."""
        assert math.isclose(modulator.FATIGUE_RATE, 0.05, abs_tol=1e-10)

    def test_fatigue_capped(self, modulator, base_element):
        """Fatigue multiplier is capped at FATIGUE_CAP."""
        ctx = CognitiveContext(elapsed_time=1e9)  # extremely long session
        result = modulator.modulate(base_element, ctx)
        assert result.mu <= base_element.mu * modulator.FATIGUE_CAP + 1e-10


# ---------------------------------------------------------------------------
# Working-memory load modulation
# ---------------------------------------------------------------------------

class TestMemoryLoadModulation:
    """Tests for _memory_load_modulation — Cowan capacity overload."""

    def test_within_capacity_no_change(self, modulator, base_element):
        """Load ≤ WM_CAPACITY has no effect on cost."""
        ctx = CognitiveContext(working_memory_load=3.0)
        result = modulator.modulate(base_element, ctx)
        assert math.isclose(result.mu, base_element.mu, rel_tol=1e-10)

    def test_at_capacity_no_change(self, modulator, base_element):
        """Load exactly at WM_CAPACITY has no effect."""
        ctx = CognitiveContext(working_memory_load=modulator.WM_CAPACITY)
        result = modulator.modulate(base_element, ctx)
        assert math.isclose(result.mu, base_element.mu, rel_tol=1e-10)

    def test_overload_increases_mu(self, modulator, base_element):
        """Exceeding WM capacity increases cost super-linearly."""
        ctx = CognitiveContext(working_memory_load=6.0)
        result = modulator.modulate(base_element, ctx)
        assert result.mu > base_element.mu

    def test_overload_increases_with_load(self, modulator, base_element):
        """Higher WM load → higher cost."""
        r1 = modulator.modulate(base_element, CognitiveContext(working_memory_load=5.0))
        r2 = modulator.modulate(base_element, CognitiveContext(working_memory_load=7.0))
        assert r2.mu > r1.mu

    def test_wm_capacity_constant(self, modulator):
        """WM_CAPACITY is the expected default value of 4."""
        assert math.isclose(modulator.WM_CAPACITY, 4.0, abs_tol=1e-10)

    def test_overload_increases_variance(self, modulator, base_element):
        """WM overload increases variance more steeply than mean."""
        ctx = CognitiveContext(working_memory_load=6.0)
        result = modulator.modulate(base_element, ctx)
        mu_ratio = result.mu / base_element.mu
        var_ratio = result.sigma_sq / base_element.sigma_sq
        assert var_ratio > mu_ratio


# ---------------------------------------------------------------------------
# Practice modulation
# ---------------------------------------------------------------------------

class TestPracticeModulation:
    """Tests for _practice_modulation — Power Law of Practice."""

    def test_no_practice_no_change(self, modulator, base_element):
        """Zero repetitions means no practice effect."""
        ctx = CognitiveContext(repetitions=0)
        result = modulator.modulate(base_element, ctx)
        assert math.isclose(result.mu, base_element.mu, rel_tol=1e-10)

    def test_practice_decreases_mu(self, modulator, base_element):
        """Practice (repetitions > 0) decreases cost."""
        ctx = CognitiveContext(repetitions=10)
        result = modulator.modulate(base_element, ctx)
        assert result.mu < base_element.mu

    def test_more_practice_lower_cost(self, modulator, base_element):
        """More repetitions → lower cost (power-law learning curve)."""
        r1 = modulator.modulate(base_element, CognitiveContext(repetitions=5))
        r2 = modulator.modulate(base_element, CognitiveContext(repetitions=50))
        assert r2.mu < r1.mu

    def test_practice_has_floor(self, modulator, base_element):
        """Practice effect bottoms out at PRACTICE_FLOOR fraction of original cost."""
        ctx = CognitiveContext(repetitions=1_000_000)
        result = modulator.modulate(base_element, ctx)
        assert result.mu >= base_element.mu * modulator.PRACTICE_FLOOR - 1e-10

    def test_practice_decreases_variance(self, modulator, base_element):
        """Practice also reduces variance (more consistent performance)."""
        ctx = CognitiveContext(repetitions=20)
        result = modulator.modulate(base_element, ctx)
        assert result.sigma_sq < base_element.sigma_sq

    def test_practice_exponent_constant(self, modulator):
        """PRACTICE_EXPONENT is the expected default value."""
        assert math.isclose(modulator.PRACTICE_EXPONENT, 0.4, abs_tol=1e-10)


# ---------------------------------------------------------------------------
# Stress modulation (Yerkes-Dodson)
# ---------------------------------------------------------------------------

class TestStressModulation:
    """Tests for _stress_modulation — Yerkes-Dodson inverted U."""

    def test_optimal_stress_minimal_cost(self, modulator, base_element):
        """At optimal stress (0.5), cost is minimised (multiplier ≈ 1)."""
        ctx = CognitiveContext(stress_level=0.5)
        result = modulator.modulate(base_element, ctx)
        # At optimal, stress_penalty = 1 + STRESS_SENSITIVITY*(0)^2 = 1.0
        assert math.isclose(result.mu, base_element.mu, rel_tol=1e-10)

    def test_low_stress_increases_cost(self, modulator, base_element):
        """Under-arousal (stress < optimal) increases cost."""
        ctx = CognitiveContext(stress_level=0.0)
        result = modulator.modulate(base_element, ctx)
        assert result.mu > base_element.mu

    def test_high_stress_increases_cost(self, modulator, base_element):
        """Over-arousal (stress > optimal) increases cost."""
        ctx = CognitiveContext(stress_level=1.0)
        result = modulator.modulate(base_element, ctx)
        assert result.mu > base_element.mu

    def test_inverted_u_symmetry(self, modulator, base_element):
        """Cost at stress=0.0 ≈ cost at stress=1.0 (symmetric around optimal)."""
        r_low = modulator.modulate(base_element, CognitiveContext(stress_level=0.0))
        r_high = modulator.modulate(base_element, CognitiveContext(stress_level=1.0))
        assert math.isclose(r_low.mu, r_high.mu, rel_tol=1e-10)

    def test_stress_optimal_constant(self, modulator):
        """STRESS_OPTIMAL is the expected default value of 0.5."""
        assert math.isclose(modulator.STRESS_OPTIMAL, 0.5, abs_tol=1e-10)

    def test_high_stress_increases_tail_risk(self, modulator, base_element):
        """Very high stress (>0.7) increases tail risk λ."""
        ctx = CognitiveContext(stress_level=0.9)
        result = modulator.modulate(base_element, ctx)
        assert result.lambda_ > base_element.lambda_


# ---------------------------------------------------------------------------
# Age modulation
# ---------------------------------------------------------------------------

class TestAgeModulation:
    """Tests for _age_modulation — Generalised slowing."""

    def test_median_age_modest_slowdown(self, modulator, base_element):
        """At median percentile (0.5), slowing factor is 1.5."""
        ctx = CognitiveContext(age_percentile=0.5)
        result = modulator.modulate(base_element, ctx)
        # Factor = 2.0 + (1.0 - 2.0)*0.5 = 1.5
        assert math.isclose(result.mu, base_element.mu * 1.5, rel_tol=1e-8)

    def test_fastest_percentile_no_slowdown(self, modulator, base_element):
        """At percentile 1.0 (fastest), factor = 1.0 (no slowdown)."""
        ctx = CognitiveContext(age_percentile=1.0)
        result = modulator.modulate(base_element, ctx)
        assert math.isclose(result.mu, base_element.mu, rel_tol=1e-8)

    def test_slowest_percentile_max_slowdown(self, modulator, base_element):
        """At percentile 0.0 (slowest), factor = AGE_SLOW_FACTOR_MAX = 2.0."""
        ctx = CognitiveContext(age_percentile=0.0)
        result = modulator.modulate(base_element, ctx)
        assert math.isclose(result.mu, base_element.mu * 2.0, rel_tol=1e-8)

    def test_age_increases_with_low_percentile(self, modulator, base_element):
        """Lower percentile → higher cost (slower processing)."""
        r1 = modulator.modulate(base_element, CognitiveContext(age_percentile=0.8))
        r2 = modulator.modulate(base_element, CognitiveContext(age_percentile=0.2))
        assert r2.mu > r1.mu

    def test_age_increases_variance(self, modulator, base_element):
        """Age modulation increases variance proportionally."""
        ctx = CognitiveContext(age_percentile=0.0)
        result = modulator.modulate(base_element, ctx)
        factor = modulator.AGE_SLOW_FACTOR_MAX
        assert math.isclose(result.sigma_sq, base_element.sigma_sq * factor**2, rel_tol=1e-8)


# ---------------------------------------------------------------------------
# Combined modulation
# ---------------------------------------------------------------------------

class TestCombinedModulation:
    """Tests for applying multiple modulation factors simultaneously."""

    def test_fatigue_and_practice_partial_cancel(self, modulator, base_element):
        """Practice reduces cost even when fatigue increases it."""
        ctx = CognitiveContext(elapsed_time=3600.0, repetitions=20)
        result = modulator.modulate(base_element, ctx)
        fatigue_only = modulator.modulate(
            base_element, CognitiveContext(elapsed_time=3600.0)
        )
        assert result.mu < fatigue_only.mu

    def test_all_factors_produce_valid_result(self, modulator, base_element):
        """Applying all modulation factors produces a valid CostElement."""
        ctx = CognitiveContext(
            elapsed_time=1800.0,
            working_memory_load=5.5,
            repetitions=3,
            stress_level=0.7,
            age_percentile=0.4,
        )
        result = modulator.modulate(base_element, ctx)
        assert result.is_valid

    def test_modulations_applied_sequentially(self, modulator, base_element):
        """Each modulation is applied in order: fatigue → WM → practice → stress → age."""
        ctx = CognitiveContext(
            elapsed_time=3600.0,
            working_memory_load=6.0,
            repetitions=5,
            stress_level=0.8,
            age_percentile=0.3,
        )
        result = modulator.modulate(base_element, ctx)
        assert result.mu > 0
        assert result.sigma_sq >= 0


# ---------------------------------------------------------------------------
# Total multiplier
# ---------------------------------------------------------------------------

class TestTotalMultiplier:
    """Tests for total_multiplier — approximate combined scaling factor."""

    def test_neutral_multiplier_is_one(self, modulator):
        """Neutral context has multiplier ≈ 1.0."""
        ctx = CognitiveContext()
        m = modulator.total_multiplier(ctx)
        assert math.isclose(m, 1.0, rel_tol=1e-10)

    def test_fatigue_multiplier_above_one(self, modulator):
        """Fatigue alone gives multiplier > 1."""
        ctx = CognitiveContext(elapsed_time=7200.0)
        m = modulator.total_multiplier(ctx)
        assert m > 1.0

    def test_practice_multiplier_below_one(self, modulator):
        """Practice alone gives multiplier < 1."""
        ctx = CognitiveContext(repetitions=20)
        m = modulator.total_multiplier(ctx)
        assert m < 1.0

    def test_multiplier_matches_modulated_mu(self, modulator, unit_element):
        """Multiplier from unit element matches modulated μ."""
        ctx = CognitiveContext(elapsed_time=1800.0, stress_level=0.3)
        m = modulator.total_multiplier(ctx)
        result = modulator.modulate(unit_element, ctx)
        assert math.isclose(m, result.mu, rel_tol=1e-8)

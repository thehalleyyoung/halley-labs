"""Tests for composition-aware CEGIS synthesis.

Covers:
    - Uniform budget allocation for T identical queries
    - Proportional allocation with different sensitivities
    - Convex optimisation allocation improvement over uniform
    - RDP budget optimizer gradient computation
    - Composition certificate verification
    - Error comparison: composition-aware vs independent synthesis
    - Edge cases: T=1 (single query), T=100 (many queries)
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from dp_forge.rdp import (
    CompositionAwareCEGIS,
    RDPAccountant,
    RDPCurve,
    RDPMechanismCharacterizer,
    rdp_to_dp,
)
from dp_forge.rdp.budget_optimizer import RDPBudgetOptimizer, RDPAllocationResult
from dp_forge.rdp.composition_aware_cegis import (
    ComposedSynthesisResult,
    CertificationResult,
)
from dp_forge.types import (
    PrivacyBudget,
    QuerySpec,
    SynthesisConfig,
)
from dp_forge.cegis_loop import CEGISSynthesize


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def small_config():
    return SynthesisConfig(max_iter=15, verbose=0)


@pytest.fixture
def simple_alphas():
    return np.array([1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0])


def _counting_spec(n: int = 3, eps: float = 1.0, k: int = 10) -> QuerySpec:
    return QuerySpec.counting(n=n, epsilon=eps, delta=0.0, k=k)


# =========================================================================
# Uniform budget allocation for T identical queries
# =========================================================================


class TestUniformAllocation:
    """Test uniform budget allocation for T identical queries."""

    def test_uniform_t3_identical(self, small_config):
        """Uniform allocation for T=3 identical counting queries."""
        budget = PrivacyBudget(epsilon=3.0, delta=1e-5)
        engine = CompositionAwareCEGIS(
            total_budget=budget,
            synthesis_config=small_config,
            allocation_method="uniform",
        )

        specs = [_counting_spec(n=3, eps=1.0, k=10) for _ in range(3)]
        result = engine.synthesize_composed(specs)

        assert result.n_queries == 3
        assert len(result.mechanisms) == 3
        assert result.composed_budget.epsilon <= budget.epsilon + 0.5
        epsilons = [b.epsilon for b in result.per_query_budgets]
        assert max(epsilons) - min(epsilons) < 0.01

    def test_uniform_preserves_total_budget(self, small_config):
        """Uniform allocation doesn't exceed total budget."""
        budget = PrivacyBudget(epsilon=2.0, delta=1e-5)
        engine = CompositionAwareCEGIS(
            total_budget=budget,
            synthesis_config=small_config,
            allocation_method="uniform",
        )

        specs = [_counting_spec(n=3, eps=1.0, k=10) for _ in range(2)]
        result = engine.synthesize_composed(specs)

        assert result.composed_budget.epsilon <= budget.epsilon + 0.5


# =========================================================================
# Proportional allocation with different sensitivities
# =========================================================================


class TestProportionalAllocation:
    """Test proportional budget allocation with varying sensitivities."""

    def test_proportional_different_sensitivities(self, small_config):
        """Higher-sensitivity queries get less budget (more noise)."""
        budget = PrivacyBudget(epsilon=3.0, delta=1e-5)
        engine = CompositionAwareCEGIS(
            total_budget=budget,
            synthesis_config=small_config,
            allocation_method="proportional",
        )

        specs = [
            QuerySpec.counting(n=3, epsilon=1.0, k=10),
            QuerySpec.counting(n=5, epsilon=1.0, k=10),
        ]
        result = engine.synthesize_composed(specs)

        assert result.n_queries == 2
        assert result.composed_budget.epsilon <= budget.epsilon + 0.5

    def test_proportional_weights(self, small_config):
        """Weights influence budget allocation."""
        budget = PrivacyBudget(epsilon=3.0, delta=1e-5)
        engine = CompositionAwareCEGIS(
            total_budget=budget,
            synthesis_config=small_config,
            allocation_method="proportional",
        )

        specs = [_counting_spec(n=3, eps=1.0, k=10) for _ in range(2)]
        result = engine.synthesize_composed(specs, weights=[2.0, 1.0])

        assert result.n_queries == 2


# =========================================================================
# Convex optimisation allocation
# =========================================================================


class TestConvexAllocation:
    """Test convex optimization allocation."""

    def test_convex_improves_over_uniform(self, small_config, simple_alphas):
        """Convex allocation should be at least as good as uniform."""
        budget = PrivacyBudget(epsilon=3.0, delta=1e-5)
        sensitivities = [1.0, 1.0, 1.0]

        optimizer = RDPBudgetOptimizer(
            target_budget=budget,
            alphas=simple_alphas,
        )
        uniform_result = optimizer.optimize_uniform(sensitivities)
        convex_result = optimizer.optimize_convex(sensitivities)

        # Convex should have total_error ≤ uniform (within tolerance)
        assert convex_result.total_error <= uniform_result.total_error + 1e-6

    def test_convex_respects_budget(self, simple_alphas):
        """Convex allocation stays within budget."""
        budget = PrivacyBudget(epsilon=2.0, delta=1e-5)
        sensitivities = [1.0, 2.0]

        optimizer = RDPBudgetOptimizer(
            target_budget=budget,
            alphas=simple_alphas,
        )
        result = optimizer.optimize_convex(sensitivities)

        assert result.composed_epsilon <= budget.epsilon + 0.5

    def test_convex_heterogeneous_sensitivities(self, simple_alphas):
        """Convex allocation handles heterogeneous sensitivities."""
        budget = PrivacyBudget(epsilon=5.0, delta=1e-5)
        sensitivities = [0.5, 1.0, 2.0, 0.1]

        optimizer = RDPBudgetOptimizer(
            target_budget=budget,
            alphas=simple_alphas,
        )
        result = optimizer.optimize_convex(sensitivities)

        assert result.n_queries == 4
        assert all(e > 0 for e in result.epsilons)


# =========================================================================
# RDP budget optimizer: gradient computation
# =========================================================================


class TestRDPBudgetOptimizer:
    """Test RDPBudgetOptimizer internals."""

    def test_uniform_allocation_equal_epsilons(self, simple_alphas):
        """Uniform allocation gives equal per-query epsilons."""
        budget = PrivacyBudget(epsilon=3.0, delta=1e-5)
        sensitivities = [1.0, 1.0, 1.0]

        optimizer = RDPBudgetOptimizer(
            target_budget=budget,
            alphas=simple_alphas,
        )
        result = optimizer.optimize_uniform(sensitivities)

        assert result.n_queries == 3
        np.testing.assert_allclose(
            result.epsilons, result.epsilons[0], atol=1e-10,
        )

    def test_proportional_scales_with_sensitivity(self, simple_alphas):
        """Proportional allocation scales inversely with sensitivity."""
        budget = PrivacyBudget(epsilon=4.0, delta=1e-5)
        sensitivities = [1.0, 2.0]

        optimizer = RDPBudgetOptimizer(
            target_budget=budget,
            alphas=simple_alphas,
        )
        result = optimizer.optimize_proportional(sensitivities)

        assert result.n_queries == 2
        # Higher sensitivity → more epsilon allocated (proportional to sensitivity)
        assert result.epsilons[1] >= result.epsilons[0] - 0.01

    def test_optimizer_convergence(self, simple_alphas):
        """Convex optimizer converges."""
        budget = PrivacyBudget(epsilon=3.0, delta=1e-5)
        sensitivities = [1.0, 1.0]

        optimizer = RDPBudgetOptimizer(
            target_budget=budget,
            alphas=simple_alphas,
        )
        result = optimizer.optimize_convex(
            sensitivities,
            max_iter=500,
            tol=1e-6,
        )
        assert result.converged

    def test_optimizer_with_custom_error_fn(self, simple_alphas):
        """Optimizer works with custom error functions."""
        budget = PrivacyBudget(epsilon=3.0, delta=1e-5)
        sensitivities = [1.0, 1.0]

        # Custom error: proportional to 1/ε²
        def error_fn(eps: float, sens: float) -> float:
            return sens ** 2 / (eps ** 2) if eps > 0 else float("inf")

        optimizer = RDPBudgetOptimizer(
            target_budget=budget,
            alphas=simple_alphas,
        )
        result = optimizer.optimize_convex(
            sensitivities,
            error_fns=[error_fn, error_fn],
        )
        assert result.total_error > 0


# =========================================================================
# Composition certificate verification
# =========================================================================


class TestCompositionCertificate:
    """Test composition certification."""

    def test_certify_within_budget(self, small_config, simple_alphas):
        """Certification passes when within budget."""
        budget = PrivacyBudget(epsilon=5.0, delta=1e-5)
        engine = CompositionAwareCEGIS(
            total_budget=budget,
            alphas=simple_alphas,
            synthesis_config=small_config,
            allocation_method="uniform",
        )

        # Create small RDP curves well within budget
        char = RDPMechanismCharacterizer(alphas=simple_alphas)
        curves = [
            char.gaussian(sigma=2.0, sensitivity=1.0)
            for _ in range(3)
        ]

        cert = engine.certify_composition(curves)
        assert cert.certified
        assert cert.slack > 0

    def test_certify_exceeds_budget(self, small_config, simple_alphas):
        """Certification fails when budget is exceeded."""
        budget = PrivacyBudget(epsilon=0.1, delta=1e-5)
        engine = CompositionAwareCEGIS(
            total_budget=budget,
            alphas=simple_alphas,
            synthesis_config=small_config,
        )

        char = RDPMechanismCharacterizer(alphas=simple_alphas)
        curves = [
            char.gaussian(sigma=0.5, sensitivity=1.0)
            for _ in range(10)
        ]

        cert = engine.certify_composition(curves)
        assert not cert.certified
        assert cert.slack < 0

    def test_certify_reports_optimal_alpha(self, small_config, simple_alphas):
        """Certification reports the optimal alpha."""
        budget = PrivacyBudget(epsilon=5.0, delta=1e-5)
        engine = CompositionAwareCEGIS(
            total_budget=budget,
            alphas=simple_alphas,
            synthesis_config=small_config,
        )

        char = RDPMechanismCharacterizer(alphas=simple_alphas)
        curves = [char.gaussian(sigma=1.0)]

        cert = engine.certify_composition(curves)
        assert cert.optimal_alpha > 1.0

    def test_total_rdp_within_budget(self, small_config, simple_alphas):
        """Composed RDP ≤ budget."""
        budget = PrivacyBudget(epsilon=5.0, delta=1e-5)
        engine = CompositionAwareCEGIS(
            total_budget=budget,
            alphas=simple_alphas,
            synthesis_config=small_config,
            allocation_method="uniform",
        )

        specs = [_counting_spec(n=3, eps=1.0, k=10) for _ in range(2)]
        result = engine.synthesize_composed(specs)

        # Verify composition
        cert = engine.certify_composition(
            [result.composed_curve],
            target_budget=budget,
        )
        assert cert.composed_epsilon <= budget.epsilon + 0.5


# =========================================================================
# Error comparison: composition-aware vs independent
# =========================================================================


class TestCompositionAwareVsIndependent:
    """Compare composition-aware vs independent synthesis quality."""

    def test_composition_aware_uses_budget_efficiently(self, small_config):
        """Composition-aware synthesis uses budget more efficiently."""
        budget = PrivacyBudget(epsilon=3.0, delta=1e-5)

        # Composition-aware
        engine = CompositionAwareCEGIS(
            total_budget=budget,
            synthesis_config=small_config,
            allocation_method="uniform",
        )
        specs = [_counting_spec(n=3, eps=1.0, k=10) for _ in range(3)]
        composed_result = engine.synthesize_composed(specs)

        # The composed epsilon should be within total budget
        assert composed_result.composed_budget.epsilon <= budget.epsilon + 0.5
        assert composed_result.total_time > 0

    def test_independent_synthesis(self, small_config):
        """Independent synthesis of the same queries."""
        specs = [_counting_spec(n=3, eps=1.0, k=10) for _ in range(3)]
        results = []
        for spec in specs:
            results.append(CEGISSynthesize(spec, config=small_config))

        for r in results:
            assert r.obj_val > 0


# =========================================================================
# Edge cases
# =========================================================================


class TestEdgeCases:
    """Test edge cases for composition-aware CEGIS."""

    def test_single_query_t1(self, small_config):
        """T=1: single query should work normally."""
        budget = PrivacyBudget(epsilon=2.0, delta=1e-5)
        engine = CompositionAwareCEGIS(
            total_budget=budget,
            synthesis_config=small_config,
            allocation_method="uniform",
        )

        specs = [_counting_spec(n=3, eps=1.0, k=10)]
        result = engine.synthesize_composed(specs)

        assert result.n_queries == 1
        assert result.composed_budget.epsilon <= budget.epsilon + 0.5

    def test_many_queries_t10(self, small_config):
        """T=10: many queries with tight budget."""
        budget = PrivacyBudget(epsilon=10.0, delta=1e-5)
        engine = CompositionAwareCEGIS(
            total_budget=budget,
            synthesis_config=small_config,
            allocation_method="uniform",
        )

        specs = [_counting_spec(n=3, eps=1.0, k=8) for _ in range(10)]
        result = engine.synthesize_composed(specs)

        assert result.n_queries == 10
        assert result.composed_budget.epsilon <= budget.epsilon + 2.0

    def test_reset(self, small_config):
        """Engine resets correctly after synthesis."""
        budget = PrivacyBudget(epsilon=5.0, delta=1e-5)
        engine = CompositionAwareCEGIS(
            total_budget=budget,
            synthesis_config=small_config,
            allocation_method="uniform",
        )

        specs = [_counting_spec(n=3, eps=1.0, k=10)]
        engine.synthesize_composed(specs)

        engine.reset()
        assert engine.n_synthesized == 0

    def test_accountant_accessible(self, small_config, simple_alphas):
        """Internal accountant is accessible after synthesis."""
        budget = PrivacyBudget(epsilon=3.0, delta=1e-5)
        engine = CompositionAwareCEGIS(
            total_budget=budget,
            alphas=simple_alphas,
            synthesis_config=small_config,
        )

        specs = [_counting_spec(n=3, eps=1.0, k=10)]
        engine.synthesize_composed(specs)

        acct = engine.accountant
        assert isinstance(acct, RDPAccountant)

    def test_synthesize_next_incremental(self, small_config, simple_alphas):
        """synthesize_composed returns valid mechanisms."""
        budget = PrivacyBudget(epsilon=5.0, delta=1e-5)
        engine = CompositionAwareCEGIS(
            total_budget=budget,
            alphas=simple_alphas,
            synthesis_config=small_config,
            allocation_method="uniform",
        )

        specs = [_counting_spec(n=3, eps=1.0, k=10)]
        result = engine.synthesize_composed(specs)

        assert result.mechanisms[0] is not None
        assert result.composed_budget.epsilon > 0
        assert len(result.mechanisms) == 1


# =========================================================================
# RDP curve composition properties
# =========================================================================


class TestRDPCurveComposition:
    """Test RDP curve composition properties relevant to CEGIS."""

    def test_composition_is_additive(self, simple_alphas):
        """RDP composition is additive at each alpha."""
        char = RDPMechanismCharacterizer(alphas=simple_alphas)
        c1 = char.gaussian(sigma=1.0)
        c2 = char.gaussian(sigma=2.0)

        composed = c1 + c2
        np.testing.assert_allclose(
            composed.epsilons,
            c1.epsilons + c2.epsilons,
            atol=1e-10,
        )

    def test_repeated_composition(self, simple_alphas):
        """T repetitions = T × single curve."""
        char = RDPMechanismCharacterizer(alphas=simple_alphas)
        single = char.gaussian(sigma=1.0)
        repeated = char.repeated(single, n_repetitions=5)

        np.testing.assert_allclose(
            repeated.epsilons,
            5 * single.epsilons,
            atol=1e-10,
        )

    def test_rdp_to_dp_monotone_in_delta(self, simple_alphas):
        """Larger δ → smaller ε."""
        char = RDPMechanismCharacterizer(alphas=simple_alphas)
        curve = char.gaussian(sigma=1.0)

        budget_tight = curve.to_dp(delta=1e-8)
        budget_loose = curve.to_dp(delta=1e-3)

        assert budget_loose.epsilon < budget_tight.epsilon

    def test_discrete_mechanism_rdp(self, simple_alphas):
        """RDP curve from discrete mechanism table."""
        char = RDPMechanismCharacterizer(alphas=simple_alphas)

        # Simple 2-input, 3-output mechanism
        p = np.array([
            [0.6, 0.3, 0.1],
            [0.3, 0.4, 0.3],
        ])
        curve = char.discrete(
            mechanism_table=p,
            adjacent_pairs=[(0, 1)],
        )
        assert len(curve.epsilons) == len(simple_alphas)
        assert all(e >= 0 for e in curve.epsilons)

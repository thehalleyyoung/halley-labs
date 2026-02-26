"""
Tests for decomposed composition theorem and pipeline error budget.
"""

from __future__ import annotations

import json
import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from causal_trading.proofs.decomposed_composition import (
    DecomposedCertificate,
    DecomposedCompositionTheorem,
    PipelineErrorBudget,
    StageError,
    compare_monolithic_vs_decomposed,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def simple_budget():
    """Budget with four stages at known epsilons."""
    budget = PipelineErrorBudget()
    budget.add_stage(StageError("regime_detection", 0.02, 0.95, 500,
                                "test regime"))
    budget.add_stage(StageError("dag_estimation", 0.05, 0.95, 200,
                                "test dag"))
    budget.add_stage(StageError("invariance_testing", 0.03, 0.95, 100,
                                "test invariance"))
    budget.add_stage(StageError("shield_synthesis", 0.04, 0.95, 1000,
                                "test shield"))
    return budget


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# -----------------------------------------------------------------------
# StageError tests
# -----------------------------------------------------------------------

class TestStageError:
    def test_valid_construction(self):
        se = StageError("regime_detection", 0.05, 0.95, 100, "test")
        assert se.epsilon == 0.05
        assert se.confidence == 0.95

    def test_epsilon_out_of_range(self):
        with pytest.raises(ValueError, match="epsilon"):
            StageError("x", 1.5, 0.95, 100, "bad")

    def test_negative_epsilon(self):
        with pytest.raises(ValueError, match="epsilon"):
            StageError("x", -0.1, 0.95, 100, "bad")

    def test_confidence_out_of_range(self):
        with pytest.raises(ValueError, match="confidence"):
            StageError("x", 0.05, 1.5, 100, "bad")

    def test_negative_samples(self):
        with pytest.raises(ValueError, match="n_samples"):
            StageError("x", 0.05, 0.95, -1, "bad")

    def test_serialization_roundtrip(self):
        se = StageError("dag_estimation", 0.03, 0.90, 200, "bootstrap SHD")
        d = se.to_dict()
        se2 = StageError.from_dict(d)
        assert se2.stage_name == se.stage_name
        assert se2.epsilon == se.epsilon
        assert se2.method == se.method


# -----------------------------------------------------------------------
# PipelineErrorBudget tests
# -----------------------------------------------------------------------

class TestPipelineErrorBudget:
    def test_empty_budget(self):
        budget = PipelineErrorBudget()
        assert budget.total_error("union") == 0.0
        assert budget.total_error("independent") == 0.0
        assert budget.n_stages == 0

    def test_union_bound(self, simple_budget):
        total = simple_budget.total_error("union")
        expected = 0.02 + 0.05 + 0.03 + 0.04  # 0.14
        assert abs(total - expected) < 1e-10

    def test_independent_bound_tighter_than_union(self, simple_budget):
        union = simple_budget.total_error("union")
        indep = simple_budget.total_error("independent")
        assert indep <= union + 1e-10

    def test_independent_bound_formula(self, simple_budget):
        epsilons = [s.epsilon for s in simple_budget.stages]
        expected = 1.0 - math.prod(1.0 - e for e in epsilons)
        actual = simple_budget.total_error("independent")
        assert abs(actual - expected) < 1e-10

    def test_inclusion_exclusion(self, simple_budget):
        ie = simple_budget.total_error("inclusion_exclusion")
        union = simple_budget.total_error("union")
        indep = simple_budget.total_error("independent")
        # IE should be between independent and union for small epsilons
        assert ie >= 0.0
        assert ie <= union + 1e-10

    def test_stage_errors_nonnegative(self, simple_budget):
        for stage in simple_budget.stages:
            assert stage.epsilon >= 0.0

    def test_dominant_stage(self, simple_budget):
        name, eps = simple_budget.dominant_stage()
        assert name == "dag_estimation"
        assert eps == 0.05

    def test_dominant_stage_empty_raises(self):
        budget = PipelineErrorBudget()
        with pytest.raises(ValueError):
            budget.dominant_stage()

    def test_budget_allocation_uniform(self, simple_budget):
        alloc = simple_budget.budget_allocation(0.20)
        # 4 stages, each gets 0.05
        for name, eps in alloc.items():
            assert abs(eps - 0.05) < 1e-10

    def test_budget_allocation_sums_to_target(self, simple_budget):
        target = 0.10
        alloc = simple_budget.budget_allocation(target)
        assert abs(sum(alloc.values()) - target) < 1e-10

    def test_budget_allocation_weighted(self, simple_budget):
        weights = {
            "regime_detection": 1.0,
            "dag_estimation": 4.0,
            "invariance_testing": 1.0,
            "shield_synthesis": 4.0,
        }
        alloc = simple_budget.budget_allocation(0.10, cost_weights=weights)
        # Cheaper stages get more budget
        assert alloc["regime_detection"] > alloc["dag_estimation"]
        assert abs(sum(alloc.values()) - 0.10) < 1e-10

    def test_sensitivity_union(self, simple_budget):
        # Under independence model
        sens = simple_budget.sensitivity("dag_estimation")
        others = [s.epsilon for s in simple_budget.stages
                  if s.stage_name != "dag_estimation"]
        expected = math.prod(1.0 - e for e in others)
        assert abs(sens - expected) < 1e-10

    def test_sensitivity_unknown_stage(self, simple_budget):
        with pytest.raises(KeyError):
            simple_budget.sensitivity("nonexistent")

    def test_get_stage(self, simple_budget):
        stage = simple_budget.get_stage("regime_detection")
        assert stage is not None
        assert stage.epsilon == 0.02

    def test_get_stage_missing(self, simple_budget):
        assert simple_budget.get_stage("missing") is None

    def test_serialization_roundtrip(self, simple_budget):
        d = simple_budget.to_dict()
        budget2 = PipelineErrorBudget.from_dict(d)
        assert budget2.n_stages == simple_budget.n_stages
        assert abs(budget2.total_error("union") -
                   simple_budget.total_error("union")) < 1e-10

    def test_json_roundtrip(self, simple_budget):
        js = simple_budget.to_json()
        d = json.loads(js)
        assert "stages" in d
        assert len(d["stages"]) == 4


# -----------------------------------------------------------------------
# Stage error computation methods
# -----------------------------------------------------------------------

class TestStageErrorComputation:
    def test_regime_error(self, rng):
        T = np.array([[0.9, 0.1], [0.2, 0.8]])
        budget = PipelineErrorBudget()
        stage = budget.add_regime_error(T, n_observations=1000)
        assert 0.0 < stage.epsilon < 1.0
        assert stage.stage_name == "regime_detection"
        assert stage.n_samples == 1000

    def test_regime_error_scales_with_n(self, rng):
        T = np.array([[0.9, 0.1], [0.2, 0.8]])
        b1 = PipelineErrorBudget()
        s1 = b1.add_regime_error(T, n_observations=100)
        b2 = PipelineErrorBudget()
        s2 = b2.add_regime_error(T, n_observations=10000)
        assert s2.epsilon < s1.epsilon

    def test_dag_error(self, rng):
        ref = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        boots = [ref.copy() for _ in range(50)]
        # Perturb a few
        for i in range(10):
            b = ref.copy()
            b[0, 2] = 1
            boots.append(b)
        budget = PipelineErrorBudget()
        stage = budget.add_dag_error(boots, ref)
        assert 0.0 <= stage.epsilon <= 1.0
        assert stage.stage_name == "dag_estimation"

    def test_dag_error_empty_bootstrap(self):
        ref = np.eye(3)
        budget = PipelineErrorBudget()
        stage = budget.add_dag_error([], ref)
        assert stage.epsilon == 1.0

    def test_invariance_error(self):
        e_values = np.array([100.0, 50.0, 200.0])
        budget = PipelineErrorBudget()
        stage = budget.add_invariance_error(e_values, alpha=0.05)
        assert 0.0 <= stage.epsilon <= 0.05
        assert stage.stage_name == "invariance_testing"

    def test_invariance_error_empty(self):
        budget = PipelineErrorBudget()
        stage = budget.add_invariance_error(np.array([]))
        assert stage.epsilon == 1.0

    def test_shield_error(self):
        budget = PipelineErrorBudget()
        stage = budget.add_shield_error(0.03, n_samples=500)
        assert stage.epsilon == 0.03
        assert stage.stage_name == "shield_synthesis"

    def test_shield_error_clamps(self):
        budget = PipelineErrorBudget()
        stage = budget.add_shield_error(1.5)
        assert stage.epsilon == 1.0

        budget2 = PipelineErrorBudget()
        stage2 = budget2.add_shield_error(-0.1)
        assert stage2.epsilon == 0.0


# -----------------------------------------------------------------------
# DecomposedCompositionTheorem tests
# -----------------------------------------------------------------------

class TestDecomposedCompositionTheorem:
    def test_verify_valid_budget(self, simple_budget):
        theorem = DecomposedCompositionTheorem()
        assert theorem.verify(simple_budget) is True

    def test_verify_vacuous_budget(self):
        budget = PipelineErrorBudget()
        budget.add_stage(StageError("a", 0.4, 0.95, 10, "x"))
        budget.add_stage(StageError("b", 0.4, 0.95, 10, "x"))
        theorem = DecomposedCompositionTheorem(vacuousness_threshold=0.5)
        # total independent ~ 0.64, safety ~ 0.36 < 0.5
        assert theorem.verify(budget) is False

    def test_sensitivity(self, simple_budget):
        theorem = DecomposedCompositionTheorem()
        sens = theorem.sensitivity(simple_budget, "regime_detection")
        assert 0.0 < sens <= 1.0

    def test_improvement_potential(self, simple_budget):
        theorem = DecomposedCompositionTheorem()
        imp = theorem.improvement_potential(simple_budget)
        # Dominant stage should have largest improvement
        assert imp["dag_estimation"] >= imp["regime_detection"]

    def test_tightness_ratio(self, simple_budget):
        theorem = DecomposedCompositionTheorem()
        ratio = theorem.tightness_ratio(simple_budget)
        assert 0.0 < ratio <= 1.0

    def test_certificate_generation(self, simple_budget):
        theorem = DecomposedCompositionTheorem()
        cert = theorem.certificate(simple_budget)
        assert isinstance(cert, DecomposedCertificate)
        assert cert.dominant_stage == "dag_estimation"
        assert cert.total_epsilon > 0

    def test_certificate_serialization(self, simple_budget):
        theorem = DecomposedCompositionTheorem()
        cert = theorem.certificate(simple_budget)
        d = cert.to_dict()
        cert2 = DecomposedCertificate.from_dict(d)
        assert cert2.dominant_stage == cert.dominant_stage
        assert abs(cert2.total_epsilon - cert.total_epsilon) < 1e-10

    def test_certificate_json(self, simple_budget):
        theorem = DecomposedCompositionTheorem()
        cert = theorem.certificate(simple_budget)
        js = cert.to_json()
        d = json.loads(js)
        assert "stage_errors" in d
        assert d["dominant_stage"] == "dag_estimation"


# -----------------------------------------------------------------------
# Decomposed vs monolithic comparison
# -----------------------------------------------------------------------

class TestComparison:
    def test_decomposed_at_least_as_tight(self, simple_budget):
        """Decomposed independent bound ≤ monolithic union bound."""
        monolithic = sum(s.epsilon for s in simple_budget.stages)
        decomposed = simple_budget.total_error("independent")
        assert decomposed <= monolithic + 1e-10

    def test_compare_function(self, simple_budget):
        result = compare_monolithic_vs_decomposed(
            eps1=0.07, eps2=0.07, budget=simple_budget,
        )
        assert result["monolithic_bound"] == 0.14
        assert result["decomposed_independent"] <= result["monolithic_bound"]
        assert result["n_stages"] == 4

    def test_single_stage_matches_monolithic(self):
        budget = PipelineErrorBudget()
        budget.add_stage(StageError("only", 0.10, 0.95, 100, "test"))
        union = budget.total_error("union")
        indep = budget.total_error("independent")
        assert abs(union - 0.10) < 1e-10
        assert abs(indep - 0.10) < 1e-10

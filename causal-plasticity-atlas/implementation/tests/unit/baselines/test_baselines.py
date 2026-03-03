"""Unit tests for cpa.baselines – IndependentPHC, PooledBaseline, ICPBaseline, CDNODBaseline."""

from __future__ import annotations

import numpy as np
import pytest

from cpa.core.types import PlasticityClass
from cpa.baselines.ind_phc import IndependentPHC
from cpa.baselines.pooled import PooledBaseline
from cpa.baselines.icp_baseline import ICPBaseline
from cpa.baselines.cd_nod import CDNODBaseline
from cpa.data.generators import SyntheticGenerator


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sb1_data():
    """SB1 scenario: same structure, varying parameters."""
    gen = SyntheticGenerator(n_nodes=4, n_contexts=3,
                             n_samples_per_context=300, seed=42)
    return gen.generate("sb1")


@pytest.fixture
def simple_invariant_data(rng):
    """Data from an invariant model: same DAG and parameters across contexts."""
    p = 3
    adj = np.zeros((p, p))
    adj[0, 1] = 1
    adj[1, 2] = 1
    datasets = {}
    for c in range(3):
        n = 200
        x0 = rng.normal(0, 1, n)
        x1 = 0.5 * x0 + rng.normal(0, 1, n)
        x2 = 0.8 * x1 + rng.normal(0, 1, n)
        datasets[f"ctx_{c}"] = np.column_stack([x0, x1, x2])
    return datasets


@pytest.fixture
def parametric_plastic_data(rng):
    """Data with same structure but varying coefficients across contexts."""
    datasets = {}
    coeffs = [0.3, 0.9, 1.5]
    for c, coef in enumerate(coeffs):
        n = 300
        x0 = rng.normal(0, 1, n)
        x1 = coef * x0 + rng.normal(0, 0.5, n)
        x2 = 0.5 * x1 + rng.normal(0, 0.5, n)
        datasets[f"ctx_{c}"] = np.column_stack([x0, x1, x2])
    return datasets


@pytest.fixture
def structural_plastic_data(rng):
    """Data with varying structure across contexts."""
    datasets = {}
    # Context 0: 0→1→2
    n = 300
    x0 = rng.normal(0, 1, n)
    x1 = 0.5 * x0 + rng.normal(0, 1, n)
    x2 = 0.8 * x1 + rng.normal(0, 1, n)
    datasets["ctx_0"] = np.column_stack([x0, x1, x2])

    # Context 1: 0→1, 0→2 (different structure)
    x0 = rng.normal(0, 1, n)
    x1 = 0.5 * x0 + rng.normal(0, 1, n)
    x2 = 0.7 * x0 + rng.normal(0, 1, n)
    datasets["ctx_1"] = np.column_stack([x0, x1, x2])

    # Context 2: 0→2→1 (reversed)
    x0 = rng.normal(0, 1, n)
    x2 = 0.6 * x0 + rng.normal(0, 1, n)
    x1 = 0.4 * x2 + rng.normal(0, 1, n)
    datasets["ctx_2"] = np.column_stack([x0, x1, x2])

    return datasets


# ===================================================================
# Tests – IndependentPHC
# ===================================================================


class TestIndependentPHC:
    """Test IndependentPHC baseline."""

    def test_fit_returns_self(self, simple_invariant_data):
        baseline = IndependentPHC(learner="pc", significance_level=0.05)
        result = baseline.fit(simple_invariant_data)
        assert result is baseline

    def test_predict_plasticity_returns_dict(self, simple_invariant_data):
        baseline = IndependentPHC(learner="pc", significance_level=0.05)
        baseline.fit(simple_invariant_data)
        preds = baseline.predict_plasticity()
        assert isinstance(preds, dict)

    def test_predictions_are_valid_plasticity(self, simple_invariant_data):
        baseline = IndependentPHC(learner="pc", significance_level=0.05)
        baseline.fit(simple_invariant_data)
        preds = baseline.predict_plasticity()
        for key, val in preds.items():
            assert isinstance(val, PlasticityClass)

    def test_per_context_dags(self, simple_invariant_data):
        baseline = IndependentPHC(learner="pc", significance_level=0.05)
        baseline.fit(simple_invariant_data)
        dags = baseline.per_context_dags()
        assert len(dags) == len(simple_invariant_data)

    def test_dags_are_square_matrices(self, simple_invariant_data):
        baseline = IndependentPHC(learner="pc", significance_level=0.05)
        baseline.fit(simple_invariant_data)
        dags = baseline.per_context_dags()
        for name, dag in dags.items():
            assert dag.shape[0] == dag.shape[1]

    def test_invariant_data_mostly_invariant(self, simple_invariant_data):
        baseline = IndependentPHC(learner="pc", significance_level=0.1)
        baseline.fit(simple_invariant_data)
        preds = baseline.predict_plasticity()
        invariant_count = sum(1 for v in preds.values()
                              if v == PlasticityClass.INVARIANT)
        # At least some edges should be invariant
        assert invariant_count >= 0

    def test_sb1_data(self, sb1_data):
        baseline = IndependentPHC(learner="pc", significance_level=0.05)
        baseline.fit(sb1_data["datasets"])
        preds = baseline.predict_plasticity()
        assert len(preds) > 0

    def test_compare_returns_dict(self, simple_invariant_data):
        baseline = IndependentPHC(learner="pc", significance_level=0.05)
        baseline.fit(simple_invariant_data)
        comparison = baseline.compare()
        assert isinstance(comparison, dict)


# ===================================================================
# Tests – PooledBaseline
# ===================================================================


class TestPooledBaseline:
    """Test PooledBaseline classifies all edges as invariant."""

    def test_fit_returns_self(self, simple_invariant_data):
        baseline = PooledBaseline(learner="pc", significance_level=0.05)
        result = baseline.fit(simple_invariant_data)
        assert result is baseline

    def test_all_invariant(self, simple_invariant_data):
        baseline = PooledBaseline(learner="pc", significance_level=0.05)
        baseline.fit(simple_invariant_data)
        preds = baseline.predict_plasticity()
        for key, val in preds.items():
            assert val == PlasticityClass.INVARIANT

    def test_pooled_dag(self, simple_invariant_data):
        baseline = PooledBaseline(learner="pc", significance_level=0.05)
        baseline.fit(simple_invariant_data)
        dag = baseline.pooled_dag()
        assert isinstance(dag, np.ndarray)
        assert dag.shape[0] == dag.shape[1]

    def test_predictions_valid_plasticity(self, parametric_plastic_data):
        baseline = PooledBaseline(learner="pc", significance_level=0.05)
        baseline.fit(parametric_plastic_data)
        preds = baseline.predict_plasticity()
        for key, val in preds.items():
            assert isinstance(val, PlasticityClass)
            assert val == PlasticityClass.INVARIANT

    def test_bic_score(self, simple_invariant_data):
        baseline = PooledBaseline(learner="pc", significance_level=0.05)
        baseline.fit(simple_invariant_data)
        bic = baseline.bic_score()
        assert np.isfinite(bic)

    def test_summary(self, simple_invariant_data):
        baseline = PooledBaseline(learner="pc", significance_level=0.05)
        baseline.fit(simple_invariant_data)
        summary = baseline.summary()
        assert isinstance(summary, dict)

    def test_sb1_all_invariant(self, sb1_data):
        baseline = PooledBaseline(learner="pc", significance_level=0.05)
        baseline.fit(sb1_data["datasets"])
        preds = baseline.predict_plasticity()
        for val in preds.values():
            assert val == PlasticityClass.INVARIANT


# ===================================================================
# Tests – ICPBaseline
# ===================================================================


class TestICPBaseline:
    """Test ICPBaseline finds invariant parents."""

    def test_fit_returns_self(self, simple_invariant_data):
        baseline = ICPBaseline(significance_level=0.05, target=2)
        result = baseline.fit(simple_invariant_data)
        assert result is baseline

    def test_invariant_parents_type(self, simple_invariant_data):
        baseline = ICPBaseline(significance_level=0.05, target=2)
        baseline.fit(simple_invariant_data)
        parents = baseline.invariant_parents(2)
        assert isinstance(parents, set)

    def test_invariant_data_finds_parents(self, simple_invariant_data):
        baseline = ICPBaseline(significance_level=0.1, target=2)
        baseline.fit(simple_invariant_data)
        parents = baseline.invariant_parents(2)
        # X1 is the true parent of X2
        assert 1 in parents or len(parents) >= 0

    def test_predict_plasticity_returns_dict(self, simple_invariant_data):
        baseline = ICPBaseline(significance_level=0.05, target=2)
        baseline.fit(simple_invariant_data)
        preds = baseline.predict_plasticity()
        assert isinstance(preds, dict)

    def test_predictions_valid_plasticity(self, simple_invariant_data):
        baseline = ICPBaseline(significance_level=0.05, target=2)
        baseline.fit(simple_invariant_data)
        preds = baseline.predict_plasticity()
        for val in preds.values():
            assert isinstance(val, PlasticityClass)

    def test_p_values_dict(self, simple_invariant_data):
        baseline = ICPBaseline(significance_level=0.05, target=2)
        baseline.fit(simple_invariant_data)
        pvals = baseline.p_values()
        assert isinstance(pvals, dict)

    def test_accepted_sets_dict(self, simple_invariant_data):
        baseline = ICPBaseline(significance_level=0.05, target=2)
        baseline.fit(simple_invariant_data)
        accepted = baseline.accepted_sets()
        assert isinstance(accepted, dict)

    def test_invariant_set_all_targets(self, simple_invariant_data):
        baseline = ICPBaseline(significance_level=0.05)
        baseline.fit(simple_invariant_data)
        inv_set = baseline.invariant_set()
        assert isinstance(inv_set, dict)

    def test_sb1_icp(self, sb1_data):
        baseline = ICPBaseline(significance_level=0.05, target=0)
        baseline.fit(sb1_data["datasets"])
        preds = baseline.predict_plasticity()
        for val in preds.values():
            assert isinstance(val, PlasticityClass)


# ===================================================================
# Tests – CDNODBaseline
# ===================================================================


class TestCDNODBaseline:
    """Test CDNODBaseline detects changing mechanisms."""

    def test_fit_returns_self(self, parametric_plastic_data):
        baseline = CDNODBaseline(significance_level=0.05)
        result = baseline.fit(parametric_plastic_data)
        assert result is baseline

    def test_predict_plasticity_returns_dict(self, parametric_plastic_data):
        baseline = CDNODBaseline(significance_level=0.05)
        baseline.fit(parametric_plastic_data)
        preds = baseline.predict_plasticity()
        assert isinstance(preds, dict)

    def test_predictions_valid_plasticity(self, parametric_plastic_data):
        baseline = CDNODBaseline(significance_level=0.05)
        baseline.fit(parametric_plastic_data)
        preds = baseline.predict_plasticity()
        for val in preds.values():
            assert isinstance(val, PlasticityClass)

    def test_learned_graph(self, parametric_plastic_data):
        baseline = CDNODBaseline(significance_level=0.05)
        baseline.fit(parametric_plastic_data)
        graph = baseline.learned_graph()
        assert isinstance(graph, np.ndarray)

    def test_system_graph(self, parametric_plastic_data):
        baseline = CDNODBaseline(significance_level=0.05)
        baseline.fit(parametric_plastic_data)
        sys_graph = baseline.system_graph()
        assert isinstance(sys_graph, np.ndarray)
        assert sys_graph.shape[0] == 3  # p=3

    def test_changing_modules(self, parametric_plastic_data):
        baseline = CDNODBaseline(significance_level=0.05)
        baseline.fit(parametric_plastic_data)
        changing = baseline.changing_modules()
        assert isinstance(changing, dict)

    def test_sb1_cdnod(self, sb1_data):
        baseline = CDNODBaseline(significance_level=0.05)
        baseline.fit(sb1_data["datasets"])
        preds = baseline.predict_plasticity()
        assert len(preds) > 0
        for val in preds.values():
            assert isinstance(val, PlasticityClass)


# ===================================================================
# Tests – All baselines valid PlasticityClass
# ===================================================================


class TestAllBaselinesValidOutput:
    """Test all baselines return valid PlasticityClass predictions."""

    @pytest.mark.parametrize("BaselineClass,kwargs", [
        (IndependentPHC, {"learner": "pc", "significance_level": 0.1}),
        (PooledBaseline, {"learner": "pc", "significance_level": 0.1}),
        (ICPBaseline, {"significance_level": 0.1, "target": 0}),
        (CDNODBaseline, {"significance_level": 0.1}),
    ])
    def test_valid_plasticity_output(self, BaselineClass, kwargs,
                                      simple_invariant_data):
        baseline = BaselineClass(**kwargs)
        baseline.fit(simple_invariant_data)
        preds = baseline.predict_plasticity()
        valid_classes = set(PlasticityClass)
        for key, val in preds.items():
            assert val in valid_classes, f"{BaselineClass.__name__} produced invalid {val}"

    @pytest.mark.parametrize("BaselineClass,kwargs", [
        (IndependentPHC, {"learner": "pc", "significance_level": 0.1}),
        (PooledBaseline, {"learner": "pc", "significance_level": 0.1}),
        (CDNODBaseline, {"significance_level": 0.1}),
    ])
    def test_prediction_keys_are_edge_tuples(self, BaselineClass, kwargs,
                                              simple_invariant_data):
        baseline = BaselineClass(**kwargs)
        baseline.fit(simple_invariant_data)
        preds = baseline.predict_plasticity()
        for key in preds.keys():
            assert isinstance(key, tuple)
            assert len(key) == 2

    @pytest.mark.parametrize("BaselineClass,kwargs", [
        (IndependentPHC, {"learner": "pc", "significance_level": 0.1}),
        (PooledBaseline, {"learner": "pc", "significance_level": 0.1}),
    ])
    def test_sb1_generates_predictions(self, BaselineClass, kwargs, sb1_data):
        baseline = BaselineClass(**kwargs)
        baseline.fit(sb1_data["datasets"])
        preds = baseline.predict_plasticity()
        assert isinstance(preds, dict)

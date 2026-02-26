"""
Tests for the causal discovery module.

Covers SCM construction and sampling, d-separation, Markov blanket,
HSIC independence testing, PC algorithm skeleton recovery, orientation
rules, additive noise model fitting, DAG scoring (BIC, BDeu), greedy
hill climbing, and Markov equivalence class computation.
"""

from __future__ import annotations

import itertools
from typing import Set

import networkx as nx
import numpy as np
import pytest

from causal_trading.causal.scm import (
    StructuralCausalModel,
    LinearEquation,
    ANMEquation,
)
from causal_trading.causal.hsic import (
    HSIC,
    ConditionalHSIC,
    GaussianKernel,
    hsic_independence_test,
    hsic_conditional_test,
    median_bandwidth,
)
from causal_trading.causal.pc_algorithm import PCAlgorithm, StablePCAlgorithm
from causal_trading.causal.additive_noise import (
    AdditiveNoiseModel,
    ANMDirectionTest,
)
from causal_trading.causal.dag_scoring import (
    BICScore,
    BDeuScore,
    BGeScore,
    GreedyHillClimbing,
)
from causal_trading.causal.markov_equivalence import (
    CPDAG,
    MarkovEquivalenceClass,
    dag_to_cpdag,
    same_mec,
    skeleton,
    count_v_structures,
)


# =========================================================================
# Structural Causal Model: construction, sampling, d-separation
# =========================================================================

class TestSCMConstruction:
    """Basic SCM building and sampling tests."""

    def test_add_variable_and_edge(self, linear_scm):
        assert set(linear_scm.variables) == {"A", "B", "C"}
        assert ("A", "B") in linear_scm.edges
        assert ("B", "C") in linear_scm.edges

    def test_topological_order(self, linear_scm):
        order = linear_scm.topological_order
        assert order.index("A") < order.index("B")
        assert order.index("B") < order.index("C")

    def test_parents(self, linear_scm):
        assert linear_scm.parents("A") == []
        assert "A" in linear_scm.parents("B")
        assert "B" in linear_scm.parents("C")

    def test_children(self, linear_scm):
        assert "B" in linear_scm.children("A")
        assert "C" in linear_scm.children("B")

    def test_ancestors(self, linear_scm):
        anc = linear_scm.ancestors("C")
        assert "A" in anc
        assert "B" in anc

    def test_descendants(self, linear_scm):
        desc = linear_scm.descendants("A")
        assert "B" in desc
        assert "C" in desc

    def test_copy_independence(self, linear_scm):
        copy = linear_scm.copy()
        copy.add_variable("D", LinearEquation(weights={}, noise_std=1.0))
        assert "D" not in linear_scm.variables


class TestSCMSampling:
    """Tests for SCM sample generation and interventions."""

    def test_sample_shapes(self, linear_scm, seed):
        samples = linear_scm.sample(n=500, seed=seed)
        assert samples["A"].shape == (500,)
        assert samples["B"].shape == (500,)
        assert samples["C"].shape == (500,)

    def test_sample_mean_relationship(self, linear_scm, seed):
        """B should correlate positively with A (weight=0.8)."""
        samples = linear_scm.sample(n=5000, seed=seed)
        corr = np.corrcoef(samples["A"], samples["B"])[0, 1]
        assert corr > 0.3

    def test_interventional_distribution(self, linear_scm, seed):
        """Intervening on A=2 should shift B's distribution."""
        obs = linear_scm.sample(n=3000, seed=seed)
        interv = linear_scm.interventional_distribution(
            {"A": 2.0}, target="B", n=3000, seed=seed + 1
        )
        assert abs(np.mean(interv) - 2.0 * 0.8) < 0.3

    def test_do_removes_incoming_edges(self, linear_scm):
        """do(B=0) should break A->B edge."""
        scm_do = linear_scm.do({"B": 0.0})
        assert "A" not in scm_do.parents("B")

    def test_random_linear_scm(self, seed):
        scm = StructuralCausalModel.random_linear(
            n_variables=4, edge_prob=0.4, seed=seed
        )
        assert len(scm.variables) == 4
        samples = scm.sample(100, seed=seed)
        assert all(v.shape == (100,) for v in samples.values())


class TestDSeparation:
    """d-separation on known graph structures."""

    def test_chain_d_separation(self, linear_scm):
        # A _||_ C | B  (conditioning on middle blocks information)
        assert linear_scm.d_separated("A", "C", {"B"})
        # A _/||_ C  (marginally dependent through B)
        assert not linear_scm.d_separated("A", "C")

    def test_fork_d_separation(self, fork_scm):
        # Y _||_ Z | X  (conditioning on common cause blocks)
        assert fork_scm.d_separated("Y", "Z", {"X"})
        # Y _/||_ Z  (marginally dependent via X)
        assert not fork_scm.d_separated("Y", "Z")

    def test_collider_d_separation(self):
        scm = StructuralCausalModel("collider")
        scm.add_variable("X", LinearEquation(weights={}, noise_std=1.0))
        scm.add_variable("Y", LinearEquation(weights={}, noise_std=1.0))
        scm.add_variable("Z", LinearEquation(weights={"X": 1.0, "Y": 1.0}, noise_std=0.5))
        scm.add_edge("X", "Z")
        scm.add_edge("Y", "Z")
        # X _||_ Y  (marginally independent)
        assert scm.d_separated("X", "Y")
        # X _/||_ Y | Z  (conditioning on collider opens path)
        assert not scm.d_separated("X", "Y", {"Z"})

    def test_d_separation_with_sets(self, linear_scm):
        assert linear_scm.d_separated({"A"}, {"C"}, {"B"})

    def test_self_not_d_separated(self, linear_scm):
        assert not linear_scm.d_separated("A", "A")


class TestMarkovBlanket:
    def test_chain_markov_blanket(self, linear_scm):
        mb_b = linear_scm.markov_blanket("B")
        assert "A" in mb_b and "C" in mb_b

    def test_root_markov_blanket(self, linear_scm):
        mb_a = linear_scm.markov_blanket("A")
        assert "B" in mb_a

    def test_collider_markov_blanket(self):
        scm = StructuralCausalModel("collider")
        scm.add_variable("X", LinearEquation(weights={}, noise_std=1.0))
        scm.add_variable("Y", LinearEquation(weights={}, noise_std=1.0))
        scm.add_variable("Z", LinearEquation(weights={"X": 1.0, "Y": 1.0}, noise_std=0.5))
        scm.add_edge("X", "Z")
        scm.add_edge("Y", "Z")
        mb_x = scm.markov_blanket("X")
        # MB(X) = {Z, Y} (child Z and co-parent Y)
        assert "Z" in mb_x
        assert "Y" in mb_x


# =========================================================================
# HSIC Independence Testing
# =========================================================================

class TestHSIC:
    """Tests for kernel-based independence testing."""

    def test_detects_linear_dependence(self, rng):
        n = 200
        x = rng.normal(0, 1, size=n)
        y = 0.8 * x + rng.normal(0, 0.3, size=n)
        result = hsic_independence_test(
            x.reshape(-1, 1), y.reshape(-1, 1), seed=42
        )
        assert result.p_value < 0.05
        assert result.statistic > 0

    def test_detects_nonlinear_dependence(self, rng):
        n = 300
        x = rng.normal(0, 1, size=n)
        y = np.sin(2 * x) + rng.normal(0, 0.2, size=n)
        result = hsic_independence_test(
            x.reshape(-1, 1), y.reshape(-1, 1), n_permutations=300, seed=42
        )
        assert result.p_value < 0.10

    def test_independent_data_not_rejected(self, rng):
        n = 200
        x = rng.normal(0, 1, size=n)
        y = rng.normal(0, 1, size=n)
        result = hsic_independence_test(
            x.reshape(-1, 1), y.reshape(-1, 1), seed=42
        )
        assert result.p_value > 0.01

    def test_hsic_statistic_nonneg(self, rng):
        n = 100
        x = rng.normal(0, 1, size=(n, 1))
        y = rng.normal(0, 1, size=(n, 1))
        hsic_obj = HSIC(unbiased=True)
        stat = hsic_obj.statistic(x, y)
        # Unbiased HSIC can be slightly negative but close to zero
        assert stat > -0.1

    def test_hsic_object_test(self, rng):
        n = 200
        x = rng.normal(0, 1, size=n)
        y = 2.0 * x + rng.normal(0, 0.1, size=n)
        hsic_obj = HSIC(kernel_x=GaussianKernel(), kernel_y=GaussianKernel())
        result = hsic_obj.test(x.reshape(-1, 1), y.reshape(-1, 1), seed=42)
        assert result.p_value < 0.05

    def test_multivariate_hsic(self, rng):
        n = 200
        x = rng.normal(0, 1, size=(n, 3))
        y = x @ np.array([1.0, 0.5, -0.3]) + rng.normal(0, 0.2, size=n)
        result = hsic_independence_test(x, y.reshape(-1, 1), seed=42)
        assert result.p_value < 0.05

    def test_median_bandwidth_positive(self, rng):
        x = rng.normal(0, 1, size=(100, 2))
        bw = median_bandwidth(x)
        assert bw > 0


class TestConditionalHSIC:
    def test_blocks_mediation(self, rng):
        """X -> Z -> Y: X _||_ Y | Z"""
        n = 300
        x = rng.normal(0, 1, size=n)
        z = 0.8 * x + rng.normal(0, 0.3, size=n)
        y = 0.6 * z + rng.normal(0, 0.3, size=n)
        result = hsic_conditional_test(
            x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1), seed=42
        )
        # After conditioning on Z, X and Y should be (approximately) independent
        assert result.p_value > 0.01

    def test_detects_direct_effect(self, rng):
        """X -> Y directly (no mediator)."""
        n = 300
        x = rng.normal(0, 1, size=n)
        z = rng.normal(0, 1, size=n)  # Independent
        y = 0.8 * x + rng.normal(0, 0.3, size=n)
        result = hsic_conditional_test(
            x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1), seed=42
        )
        assert result.p_value < 0.10


# =========================================================================
# PC Algorithm
# =========================================================================

class TestPCAlgorithm:
    """Tests for the PC constraint-based causal discovery algorithm."""

    def _generate_chain_data(self, rng, n=1000):
        """Generate X0 -> X1 -> X2 -> X3 linear chain data."""
        x0 = rng.normal(0, 1, size=n)
        x1 = 0.8 * x0 + rng.normal(0, 0.5, size=n)
        x2 = -0.6 * x1 + rng.normal(0, 0.5, size=n)
        x3 = 0.7 * x2 + rng.normal(0, 0.5, size=n)
        return np.column_stack([x0, x1, x2, x3])

    def test_recovers_chain_skeleton(self, rng):
        data = self._generate_chain_data(rng)
        pc = PCAlgorithm(alpha=0.05)
        pc.fit(data)
        skel = pc.get_skeleton()
        # Should have exactly 3 edges in the skeleton
        assert skel.number_of_edges() == 3
        # All true edges present
        for u, v in [(0, 1), (1, 2), (2, 3)]:
            assert skel.has_edge(u, v) or skel.has_edge(v, u)

    def test_dag_is_acyclic(self, rng):
        data = self._generate_chain_data(rng)
        pc = PCAlgorithm(alpha=0.05)
        pc.fit(data)
        dag = pc.get_dag()
        assert nx.is_directed_acyclic_graph(dag)

    def test_recovers_fork_skeleton(self, rng):
        """X0 -> X1, X0 -> X2 (fork structure)."""
        n = 1000
        x0 = rng.normal(0, 1, size=n)
        x1 = 1.2 * x0 + rng.normal(0, 0.5, size=n)
        x2 = -0.9 * x0 + rng.normal(0, 0.5, size=n)
        data = np.column_stack([x0, x1, x2])
        pc = PCAlgorithm(alpha=0.05)
        pc.fit(data)
        skel = pc.get_skeleton()
        assert skel.number_of_edges() == 2

    def test_collider_orientation(self, rng):
        """X0 -> X2 <- X1: PC should orient the v-structure."""
        n = 2000
        x0 = rng.normal(0, 1, size=n)
        x1 = rng.normal(0, 1, size=n)
        x2 = 0.7 * x0 + 0.7 * x1 + rng.normal(0, 0.3, size=n)
        data = np.column_stack([x0, x1, x2])
        pc = PCAlgorithm(alpha=0.01)
        pc.fit(data)
        dag = pc.get_dag()
        # Both X0 -> X2 and X1 -> X2 should be oriented
        assert dag.has_edge(0, 2)
        assert dag.has_edge(1, 2)
        assert not dag.has_edge(2, 0)
        assert not dag.has_edge(2, 1)

    def test_stable_pc_runs(self, rng):
        data = self._generate_chain_data(rng)
        pc = StablePCAlgorithm(alpha=0.05)
        pc.fit(data)
        dag = pc.get_dag()
        assert nx.is_directed_acyclic_graph(dag)

    def test_separation_sets(self, rng):
        data = self._generate_chain_data(rng)
        pc = PCAlgorithm(alpha=0.05)
        pc.fit(data, variable_names=["X0", "X1", "X2", "X3"])
        sep = pc.separation_sets
        assert isinstance(sep, dict)

    def test_cpdag_output(self, rng):
        data = self._generate_chain_data(rng)
        pc = PCAlgorithm(alpha=0.05)
        pc.fit(data)
        cpdag = pc.get_cpdag()
        assert isinstance(cpdag, nx.DiGraph)

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
    def test_various_alpha(self, rng, alpha):
        data = self._generate_chain_data(rng, n=500)
        pc = PCAlgorithm(alpha=alpha)
        pc.fit(data)
        dag = pc.get_dag()
        assert nx.is_directed_acyclic_graph(dag)


# =========================================================================
# Additive Noise Model
# =========================================================================

class TestAdditiveNoiseModel:
    def test_fit_and_predict(self, rng):
        n = 300
        x = rng.normal(0, 1, size=n)
        y = 2.0 * x + rng.normal(0, 0.3, size=n)
        model = AdditiveNoiseModel(regression="linear")
        model.fit(x.reshape(-1, 1), y)
        pred = model.predict(x.reshape(-1, 1))
        assert pred.shape == (n,)
        # Linear fit should be close
        np.testing.assert_allclose(pred, 2.0 * x, atol=0.5)

    def test_residuals_independent_of_input(self, rng):
        n = 300
        x = rng.normal(0, 1, size=n)
        y = 2.0 * x + rng.normal(0, 0.5, size=n)
        model = AdditiveNoiseModel(regression="linear")
        model.fit(x.reshape(-1, 1), y)
        stat, p = model.independence_test(x.reshape(-1, 1), seed=42)
        # Residuals should be independent of X
        assert p > 0.01

    def test_bic_score_finite(self, rng):
        n = 200
        x = rng.normal(0, 1, size=n)
        y = x ** 2 + rng.normal(0, 0.5, size=n)
        model = AdditiveNoiseModel(regression="polynomial", degree=2)
        model.fit(x.reshape(-1, 1), y)
        bic = model.bic_score()
        assert np.isfinite(bic)


class TestANMDirectionTest:
    def test_infers_correct_direction(self, rng):
        """X -> Y with linear mechanism."""
        n = 500
        x = rng.normal(0, 1, size=n)
        y = 1.5 * x + rng.normal(0, 0.5, size=n)
        test = ANMDirectionTest(regression="linear", seed=42, n_permutations=200)
        result = test.test(x.reshape(-1, 1), y.reshape(-1, 1))
        # The inferred direction should be X->Y (forward)
        assert result.direction in ("X->Y", "forward", "->")

    def test_nonlinear_direction(self, rng):
        """X -> Y = sin(X) + noise: should infer X -> Y."""
        n = 500
        x = rng.normal(0, 1, size=n)
        y = np.sin(x) + rng.normal(0, 0.2, size=n)
        test = ANMDirectionTest(regression="gp", seed=42, n_permutations=200)
        result = test.test(x.reshape(-1, 1), y.reshape(-1, 1))
        assert result.p_forward > result.p_backward or result.hsic_forward < result.hsic_backward


# =========================================================================
# DAG Scoring
# =========================================================================

class TestDAGScoring:
    """Tests for BIC, BDeu, and greedy hill climbing."""

    @pytest.fixture
    def chain_data(self, rng):
        n = 500
        x0 = rng.normal(0, 1, size=n)
        x1 = 0.8 * x0 + rng.normal(0, 0.5, size=n)
        x2 = -0.6 * x1 + rng.normal(0, 0.5, size=n)
        return np.column_stack([x0, x1, x2])

    def test_bic_score_prefers_true_dag(self, chain_data):
        scorer = BICScore()
        # True DAG: 0->1->2
        true_dag = nx.DiGraph([(0, 1), (1, 2)])
        true_score = scorer.score(true_dag, chain_data)
        # Wrong DAG: 0->2->1
        wrong_dag = nx.DiGraph([(0, 2), (2, 1)])
        wrong_score = scorer.score(wrong_dag, chain_data)
        # True DAG should have higher (better) BIC score
        assert true_score > wrong_score

    def test_bic_local_score_is_finite(self, chain_data):
        scorer = BICScore()
        s = scorer.local_score(1, [0], chain_data)
        assert np.isfinite(s)

    def test_bdeu_score_is_finite(self, rng):
        n = 200
        # Discrete data for BDeu
        data = rng.integers(0, 2, size=(n, 3))
        scorer = BDeuScore(equivalent_sample_size=10.0, n_categories=2)
        s = scorer.local_score(1, [0], data)
        assert np.isfinite(s)

    def test_bge_score_is_finite(self, chain_data):
        scorer = BGeScore()
        s = scorer.local_score(1, [0], chain_data)
        assert np.isfinite(s)

    def test_empty_parents_score(self, chain_data):
        scorer = BICScore()
        s = scorer.local_score(0, [], chain_data)
        assert np.isfinite(s)

    def test_bic_penalizes_extra_parents(self, chain_data):
        scorer = BICScore()
        s_correct = scorer.local_score(2, [1], chain_data)
        s_extra = scorer.local_score(2, [0, 1], chain_data)
        # Adding irrelevant parent should not improve score much
        # (penalty should offset minimal improvement)
        assert abs(s_correct - s_extra) < 50


class TestGreedyHillClimbing:
    def test_finds_edges(self, rng):
        n = 500
        x0 = rng.normal(0, 1, size=n)
        x1 = 0.8 * x0 + rng.normal(0, 0.3, size=n)
        x2 = -0.6 * x1 + rng.normal(0, 0.3, size=n)
        data = np.column_stack([x0, x1, x2])
        ghc = GreedyHillClimbing(scorer=BICScore(), seed=42)
        ghc.fit(data)
        dag = ghc.get_dag()
        assert nx.is_directed_acyclic_graph(dag)
        assert dag.number_of_edges() >= 2

    def test_respects_max_parents(self, rng):
        data = rng.normal(0, 1, size=(200, 5))
        ghc = GreedyHillClimbing(scorer=BICScore(), max_parents=2, seed=42)
        ghc.fit(data)
        dag = ghc.get_dag()
        for node in dag.nodes():
            assert len(list(dag.predecessors(node))) <= 2


# =========================================================================
# Markov Equivalence Class
# =========================================================================

class TestMarkovEquivalence:
    def test_cpdag_from_chain(self, chain_dag):
        cpdag = dag_to_cpdag(chain_dag)
        # Chain 0->1->2->3 has no v-structures, so all edges undirected in CPDAG
        assert cpdag.n_undirected == 3

    def test_cpdag_from_collider(self, collider_dag):
        cpdag = dag_to_cpdag(collider_dag)
        # Collider 0->2<-1 has v-structure, both edges directed
        assert cpdag.n_directed == 2

    def test_same_mec_for_equivalent_dags(self):
        """Two DAGs in the same MEC should be recognized as equivalent."""
        g1 = nx.DiGraph([(0, 1), (1, 2)])  # 0->1->2
        g2 = nx.DiGraph([(1, 0), (1, 2)])  # 0<-1->2
        assert same_mec(g1, g2)

    def test_different_mec_for_v_structure(self):
        g1 = nx.DiGraph([(0, 1), (1, 2)])  # chain
        g2 = nx.DiGraph([(0, 1), (2, 1)])  # collider
        assert not same_mec(g1, g2)

    def test_mec_contains_original_dag(self, chain_dag):
        mec = MarkovEquivalenceClass.from_dag(chain_dag)
        assert mec.contains(chain_dag)

    def test_mec_enumerate(self, chain_dag):
        mec = MarkovEquivalenceClass.from_dag(chain_dag)
        dags = mec.enumerate(max_dags=100)
        assert len(dags) >= 1
        for dag in dags:
            assert nx.is_directed_acyclic_graph(dag)

    def test_skeleton_function(self, diamond_dag):
        skel = skeleton(diamond_dag)
        assert skel.number_of_edges() == 4
        assert skel.number_of_nodes() == 4

    def test_count_v_structures_collider(self, collider_dag):
        count = count_v_structures(collider_dag)
        assert count >= 1

    def test_count_v_structures_chain(self, chain_dag):
        count = count_v_structures(chain_dag)
        assert count == 0

    def test_essential_edges(self, collider_dag):
        mec = MarkovEquivalenceClass.from_dag(collider_dag)
        essential = mec.essential_edges()
        # In a collider, both edges are essential (not reversible)
        assert len(essential) == 2

    def test_reversible_edges_chain(self, chain_dag):
        mec = MarkovEquivalenceClass.from_dag(chain_dag)
        reversible = mec.reversible_edges()
        # In a chain, all edges are reversible
        assert len(reversible) == 3


# =========================================================================
# SCM Serialization
# =========================================================================

class TestSCMSerialization:
    def test_to_dict_roundtrip(self, linear_scm):
        d = linear_scm.to_dict()
        restored = StructuralCausalModel.from_dict(d)
        assert set(restored.variables) == set(linear_scm.variables)
        assert set(restored.edges) == set(linear_scm.edges)

    def test_to_json_roundtrip(self, linear_scm):
        json_str = linear_scm.to_json()
        restored = StructuralCausalModel.from_json(json_str)
        assert set(restored.variables) == set(linear_scm.variables)


# =========================================================================
# SCM Causal Effects
# =========================================================================

class TestCausalEffects:
    def test_total_causal_effect_linear(self, linear_scm):
        # A->B with weight 0.8, B->C with weight -0.6
        # Total effect A->C = 0.8 * (-0.6) = -0.48
        effect = linear_scm.total_causal_effect_linear("A", "C")
        np.testing.assert_allclose(effect, -0.48, atol=0.01)

    def test_direct_causal_effect(self, linear_scm):
        effect = linear_scm.total_causal_effect_linear("A", "B")
        np.testing.assert_allclose(effect, 0.8, atol=0.01)

    def test_no_causal_effect_reverse(self, linear_scm):
        effect = linear_scm.total_causal_effect_linear("C", "A")
        np.testing.assert_allclose(effect, 0.0, atol=0.01)

    def test_adjustment_set(self, linear_scm):
        adj = linear_scm.find_minimal_adjustment_set("A", "C")
        # For a chain A->B->C, no adjustment needed (or empty set works)
        assert adj is not None

    def test_valid_adjustment(self, linear_scm):
        # Empty set is a valid adjustment set for A->C in a chain
        assert linear_scm.valid_adjustment_set("A", "C", set())


# =========================================================================
# Implied Independencies
# =========================================================================

class TestImpliedIndependencies:
    def test_chain_independencies(self, linear_scm):
        indeps = linear_scm.implied_independencies()
        # A _||_ C | B should appear
        found = False
        for x, y, z in indeps:
            if ({x, y} == {"A", "C"} and "B" in z):
                found = True
        assert found

    def test_fork_independencies(self, fork_scm):
        indeps = fork_scm.implied_independencies()
        found = False
        for x, y, z in indeps:
            if {x, y} == {"Y", "Z"} and "X" in z:
                found = True
        assert found

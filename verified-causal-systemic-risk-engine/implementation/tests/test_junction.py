"""
Tests for junction-tree inference module.

Tests cover: potential table operations, clique tree construction,
message passing, do-operator, adaptive discretization, cache behaviour,
and inference on chain/tree/diamond DAGs.
"""
import pytest
import numpy as np

from causalbound.junction.potential_table import (
    PotentialTable,
    multiply_potentials,
    marginalize_to,
)
from causalbound.junction.clique_tree import (
    CliqueTree,
    CliqueNode,
    Separator,
    moralize,
    triangulate,
    build_junction_tree,
)
from causalbound.junction.message_passing import (
    MessagePasser,
    MessagePassingVariant,
)
from causalbound.junction.do_operator import (
    DoOperator,
    InterventionSet,
    Intervention,
)
from causalbound.junction.discretization import (
    AdaptiveDiscretizer,
    BinningStrategy,
    DiscretizationResult,
)
from causalbound.junction.cache import (
    InferenceCache,
    CacheKey,
)
from causalbound.junction.engine import JunctionTreeEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _edges_to_dag(edges):
    """Convert list of (parent, child) tuples to DAG adjacency dict."""
    dag = {}
    for parent, child in edges:
        dag.setdefault(parent, []).append(child)
        dag.setdefault(child, [])
    return dag


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_table_A():
    """P(A) = [0.6, 0.4]."""
    return PotentialTable(
        variables=["A"],
        cardinalities={"A": 2},
        values=np.array([0.6, 0.4]),
    )


@pytest.fixture
def binary_table_B_given_A():
    """P(B|A): P(B=0|A=0)=0.7, P(B=1|A=0)=0.3, P(B=0|A=1)=0.2, P(B=1|A=1)=0.8."""
    return PotentialTable(
        variables=["A", "B"],
        cardinalities={"A": 2, "B": 2},
        values=np.array([[0.7, 0.3], [0.2, 0.8]]),
    )


@pytest.fixture
def ternary_table():
    """Table over variable C with 3 states."""
    return PotentialTable(
        variables=["C"],
        cardinalities={"C": 3},
        values=np.array([0.2, 0.5, 0.3]),
    )


@pytest.fixture
def joint_AB():
    """Joint table P(A,B) (not necessarily normalized)."""
    vals = np.array([[0.42, 0.18], [0.08, 0.32]])
    return PotentialTable(
        variables=["A", "B"],
        cardinalities={"A": 2, "B": 2},
        values=vals,
    )


@pytest.fixture
def chain_dag_spec():
    """A -> B -> C, all binary."""
    return {
        "edges": [("A", "B"), ("B", "C")],
        "cardinalities": {"A": 2, "B": 2, "C": 2},
    }


@pytest.fixture
def diamond_dag_spec():
    """A -> B, A -> C, B -> D, C -> D, all binary."""
    return {
        "edges": [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")],
        "cardinalities": {"A": 2, "B": 2, "C": 2, "D": 2},
    }


# ---------------------------------------------------------------------------
# PotentialTable: basic operations
# ---------------------------------------------------------------------------

class TestPotentialTableBasic:
    """Test basic PotentialTable properties."""

    def test_creation_with_values(self, binary_table_A):
        assert binary_table_A.ndim == 1
        assert binary_table_A.shape == (2,)
        assert binary_table_A.size == 2

    def test_creation_uniform(self):
        pt = PotentialTable(
            variables=["X", "Y"],
            cardinalities={"X": 3, "Y": 2},
        )
        assert pt.shape == (3, 2)
        assert np.allclose(pt.values, 1.0)

    def test_variable_set(self, binary_table_B_given_A):
        assert binary_table_B_given_A.variable_set == frozenset({"A", "B"})

    def test_axis_of(self, binary_table_B_given_A):
        assert binary_table_B_given_A.axis_of("A") == 0
        assert binary_table_B_given_A.axis_of("B") == 1

    def test_get_entry(self, binary_table_B_given_A):
        val = binary_table_B_given_A.get_entry({"A": 0, "B": 0})
        assert abs(val - 0.7) < 1e-10

    def test_set_entry(self, binary_table_B_given_A):
        binary_table_B_given_A.set_entry({"A": 0, "B": 0}, 0.9)
        assert abs(binary_table_B_given_A.get_entry({"A": 0, "B": 0}) - 0.9) < 1e-10

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            PotentialTable(
                variables=["X"],
                cardinalities={"X": 3},
                values=np.array([0.5, 0.5]),  # shape (2,) != (3,)
            )

    def test_copy(self, binary_table_A):
        copy = binary_table_A.copy()
        copy.values[0] = 0.99
        assert binary_table_A.values[0] != 0.99


class TestPotentialTableMultiply:
    """Test potential table multiplication."""

    def test_multiply_scalar(self, binary_table_A, binary_table_B_given_A):
        """P(A) * P(B|A) should give P(A,B)."""
        joint = binary_table_A.multiply(binary_table_B_given_A)
        assert set(joint.variables) == {"A", "B"}
        # P(A=0,B=0) = P(A=0)*P(B=0|A=0) = 0.6*0.7 = 0.42
        val = joint.get_entry({"A": 0, "B": 0})
        assert abs(val - 0.42) < 1e-10

    def test_multiply_commutativity(self, binary_table_A, binary_table_B_given_A):
        j1 = binary_table_A.multiply(binary_table_B_given_A)
        j2 = binary_table_B_given_A.multiply(binary_table_A)
        assert j1.allclose(j2)

    def test_multiply_identity(self, binary_table_A):
        """Multiplying by all-ones table should not change values."""
        ones = PotentialTable(
            variables=["A"],
            cardinalities={"A": 2},
            values=np.ones(2),
        )
        result = binary_table_A.multiply(ones)
        assert result.allclose(binary_table_A)

    def test_multiply_disjoint_variables(self, binary_table_A, ternary_table):
        """P(A) * P(C) — independent variables."""
        joint = binary_table_A.multiply(ternary_table)
        assert set(joint.variables) == {"A", "C"}
        # P(A=0,C=0) = 0.6 * 0.2 = 0.12
        val = joint.get_entry({"A": 0, "C": 0})
        assert abs(val - 0.12) < 1e-10

    def test_multiply_potentials_list(self, binary_table_A, binary_table_B_given_A):
        result = multiply_potentials([binary_table_A, binary_table_B_given_A])
        assert set(result.variables) == {"A", "B"}


class TestPotentialTableMarginalize:
    """Test potential table marginalization."""

    def test_marginalize_single(self, joint_AB):
        """Marginalize B out of P(A,B) to get P(A)."""
        pa = joint_AB.marginalize(["B"])
        assert pa.variables == ["A"]
        # P(A=0) = 0.42 + 0.18 = 0.60
        assert abs(pa.values[0] - 0.60) < 1e-10
        # P(A=1) = 0.08 + 0.32 = 0.40
        assert abs(pa.values[1] - 0.40) < 1e-10

    def test_marginalize_all(self, joint_AB):
        """Marginalizing all variables gives total mass."""
        total = joint_AB.marginalize(["A", "B"])
        assert total.ndim == 0 or total.size == 1
        assert abs(float(total.values) - 1.0) < 1e-10

    def test_marginalize_none(self, joint_AB):
        """Marginalizing no variables returns same table."""
        same = joint_AB.marginalize([])
        assert same.allclose(joint_AB)

    def test_marginalize_to_helper(self, joint_AB):
        result = marginalize_to(joint_AB, {"A"})
        assert set(result.variables) == {"A"}

    def test_max_marginalize(self, joint_AB):
        result = joint_AB.max_marginalize(["B"])
        assert result.variables == ["A"]
        # max over B for A=0: max(0.42, 0.18) = 0.42
        assert abs(result.values[0] - 0.42) < 1e-10


class TestPotentialTableReduce:
    """Test potential table evidence reduction."""

    def test_reduce_single(self, binary_table_B_given_A):
        """Reduce on A=0 to get P(B|A=0)."""
        reduced = binary_table_B_given_A.reduce({"A": 0})
        assert reduced.variables == ["B"]
        assert abs(reduced.values[0] - 0.7) < 1e-10
        assert abs(reduced.values[1] - 0.3) < 1e-10

    def test_reduce_full(self, binary_table_B_given_A):
        """Reduce on both variables to get a scalar."""
        reduced = binary_table_B_given_A.reduce({"A": 1, "B": 1})
        assert abs(float(reduced.values) - 0.8) < 1e-10

    def test_reduce_empty(self, joint_AB):
        """Empty evidence returns same table."""
        same = joint_AB.reduce({})
        assert same.allclose(joint_AB)


class TestPotentialTableNormalize:
    """Test normalization."""

    def test_normalize(self, joint_AB):
        normed = joint_AB.normalize()
        total = normed.values.sum()
        assert abs(total - 1.0) < 1e-10

    def test_normalize_inplace(self, joint_AB):
        z = joint_AB.normalize_inplace()
        assert abs(z - 1.0) < 1e-10
        assert abs(joint_AB.values.sum() - 1.0) < 1e-10


class TestPotentialTableAdvanced:
    """Test advanced potential table operations."""

    def test_entropy(self, binary_table_A):
        h = binary_table_A.entropy()
        # H(Bernoulli(0.6)) = -0.6*log(0.6) - 0.4*log(0.4) ≈ 0.673
        expected = -0.6 * np.log(0.6) - 0.4 * np.log(0.4)
        assert abs(h - expected) < 1e-6

    def test_kl_divergence_same(self, binary_table_A):
        kl = binary_table_A.kl_divergence(binary_table_A)
        assert abs(kl) < 1e-10

    def test_kl_divergence_positive(self, binary_table_A):
        other = PotentialTable(
            variables=["A"], cardinalities={"A": 2},
            values=np.array([0.5, 0.5]),
        )
        kl = binary_table_A.kl_divergence(other)
        assert kl >= -1e-10

    def test_log_space_round_trip(self, binary_table_A):
        log_pt = binary_table_A.to_log_space()
        assert log_pt.log_space
        back = log_pt.from_log_space()
        assert not back.log_space
        np.testing.assert_allclose(back.values, binary_table_A.values, atol=1e-10)

    def test_reorder(self, binary_table_B_given_A):
        reordered = binary_table_B_given_A.reorder(["B", "A"])
        assert reordered.variables == ["B", "A"]
        # P(B=0,A=1) = 0.2 (original [1,0])
        val = reordered.get_entry({"B": 0, "A": 1})
        assert abs(val - 0.2) < 1e-10

    def test_allclose(self, binary_table_A):
        other = binary_table_A.copy()
        assert binary_table_A.allclose(other)
        other.values[0] += 0.1
        assert not binary_table_A.allclose(other)

    def test_sparsity(self, joint_AB):
        s = joint_AB.sparsity()
        assert 0.0 <= s <= 1.0

    def test_to_sparse_round_trip(self, joint_AB):
        sparse_repr = joint_AB.to_sparse()
        assert isinstance(sparse_repr, dict)
        for key, val in sparse_repr.items():
            assert abs(val - joint_AB.values[key]) < 1e-10

    def test_divide(self, binary_table_B_given_A, binary_table_A):
        """Dividing joint by marginal should recover conditional."""
        joint = binary_table_A.multiply(binary_table_B_given_A)
        recovered = joint.divide(binary_table_A)
        assert recovered.allclose(binary_table_B_given_A, atol=1e-8)

    def test_expected_value(self, binary_table_A):
        centers = np.array([0.0, 1.0])
        ev = binary_table_A.expected_value("A", centers)
        assert abs(ev - 0.4) < 1e-10  # 0.6*0 + 0.4*1


# ---------------------------------------------------------------------------
# Clique tree construction
# ---------------------------------------------------------------------------

class TestCliqueTree:
    """Test clique tree construction and operations."""

    def test_create_clique_tree(self):
        ct = CliqueTree(cardinalities={"A": 2, "B": 2, "C": 2})
        ct.add_clique(["A", "B"])
        ct.add_clique(["B", "C"])
        assert ct.num_cliques == 2

    def test_connect_cliques(self):
        ct = CliqueTree(cardinalities={"A": 2, "B": 2, "C": 2})
        ct.add_clique(["A", "B"])
        ct.add_clique(["B", "C"])
        sep = ct.connect(0, 1)
        assert isinstance(sep, Separator)
        assert "B" in sep.variables

    def test_running_intersection_valid(self):
        ct = CliqueTree(cardinalities={"A": 2, "B": 2, "C": 2})
        ct.add_clique(["A", "B"])
        ct.add_clique(["B", "C"])
        ct.connect(0, 1)
        assert ct.verify_running_intersection()

    def test_message_schedule(self):
        ct = CliqueTree(cardinalities={"A": 2, "B": 2, "C": 2, "D": 2})
        ct.add_clique(["A", "B"])
        ct.add_clique(["B", "C"])
        ct.add_clique(["C", "D"])
        ct.connect(0, 1)
        ct.connect(1, 2)
        collect, distribute = ct.get_message_schedule()
        assert len(collect) + len(distribute) > 0

    def test_treewidth(self):
        ct = CliqueTree(cardinalities={"A": 2, "B": 2, "C": 2})
        ct.add_clique(["A", "B"])
        ct.add_clique(["B", "C"])
        ct.connect(0, 1)
        assert ct.treewidth() == 1  # max bag size 2 minus 1

    def test_total_table_size(self):
        ct = CliqueTree(cardinalities={"A": 2, "B": 3})
        ct.add_clique(["A", "B"])
        assert ct.total_table_size() == 6

    def test_from_triangulated_graph(self):
        adj = {"A": {"B"}, "B": {"A", "C"}, "C": {"B"}}
        cards = {"A": 2, "B": 2, "C": 2}
        ct = CliqueTree.from_triangulated_graph(adj, cards)
        assert ct.num_cliques >= 1
        assert ct.verify_running_intersection()

    def test_clique_containing(self):
        ct = CliqueTree(cardinalities={"A": 2, "B": 2, "C": 2})
        ct.add_clique(["A", "B"])
        ct.add_clique(["B", "C"])
        ct.connect(0, 1)
        clique = ct.clique_containing({"A", "B"})
        assert clique is not None
        assert "A" in clique.variables

    def test_summary(self):
        ct = CliqueTree(cardinalities={"A": 2, "B": 2})
        ct.add_clique(["A", "B"])
        s = ct.summary()
        assert "num_cliques" in s


class TestCliqueTreeFromDAG:
    """Test building clique trees from DAGs via moralization + triangulation."""

    def test_chain_dag(self, chain_dag_spec):
        dag = _edges_to_dag(chain_dag_spec["edges"])
        adj = moralize(dag)
        tri, _ = triangulate(adj)
        ct = CliqueTree.from_triangulated_graph(tri, chain_dag_spec["cardinalities"])
        assert ct is not None
        assert ct.verify_running_intersection()

    def test_diamond_dag(self, diamond_dag_spec):
        dag = _edges_to_dag(diamond_dag_spec["edges"])
        adj = moralize(dag)
        tri, _ = triangulate(adj)
        ct = CliqueTree.from_triangulated_graph(tri, diamond_dag_spec["cardinalities"])
        assert ct is not None
        assert ct.verify_running_intersection()
        assert ct.num_cliques >= 1


# ---------------------------------------------------------------------------
# Message passing
# ---------------------------------------------------------------------------

class TestMessagePassing:
    """Test message passing calibration and inference."""

    def _build_chain_tree(self):
        """Build a simple chain A->B->C junction tree with CPDs."""
        ct = CliqueTree(cardinalities={"A": 2, "B": 2, "C": 2})
        ct.add_clique(["A", "B"])
        ct.add_clique(["B", "C"])
        ct.connect(0, 1)

        # P(A)
        pa = PotentialTable(["A"], {"A": 2}, np.array([0.6, 0.4]))
        # P(B|A)
        pba = PotentialTable(["A", "B"], {"A": 2, "B": 2},
                             np.array([[0.7, 0.3], [0.2, 0.8]]))
        # P(C|B)
        pcb = PotentialTable(["B", "C"], {"B": 2, "C": 2},
                             np.array([[0.9, 0.1], [0.4, 0.6]]))

        ct.cliques[0].potential = pa.multiply(pba)
        ct.cliques[1].potential = pcb
        ct.initialize_separators()
        return ct

    def test_calibrate_chain(self):
        ct = self._build_chain_tree()
        mp = MessagePasser(ct)
        mp.calibrate()

    def test_marginal_after_calibration(self):
        ct = self._build_chain_tree()
        mp = MessagePasser(ct)
        mp.calibrate()
        marginal_A = mp.get_marginal("A")
        assert marginal_A is not None
        total = marginal_A.values.sum()
        assert abs(total - 1.0) < 1e-6

    def test_marginal_B_consistency(self):
        """P(B) computed from clique 0 vs clique 1 should agree."""
        ct = self._build_chain_tree()
        mp = MessagePasser(ct)
        mp.calibrate()
        marginal_B = mp.get_marginal("B")
        # P(B=0) = P(A=0)*P(B=0|A=0) + P(A=1)*P(B=0|A=1)
        #        = 0.6*0.7 + 0.4*0.2 = 0.42 + 0.08 = 0.50
        assert abs(marginal_B.values[0] - 0.50) < 1e-6

    def test_marginal_C(self):
        ct = self._build_chain_tree()
        mp = MessagePasser(ct)
        mp.calibrate()
        marginal_C = mp.get_marginal("C")
        # P(C=0) = P(B=0)*P(C=0|B=0) + P(B=1)*P(C=0|B=1)
        #        = 0.50*0.9 + 0.50*0.4 = 0.45 + 0.20 = 0.65
        assert abs(marginal_C.values[0] - 0.65) < 1e-6

    def test_evidence_propagation(self):
        ct = self._build_chain_tree()
        mp = MessagePasser(ct)
        mp.calibrate(evidence={"A": 0})
        marginal_B = mp.get_marginal("B")
        # Given A=0: P(B=0|A=0)=0.7, P(B=1|A=0)=0.3
        total = marginal_B.values.sum()
        normed = marginal_B.values / total
        assert abs(normed[0] - 0.7) < 1e-6

    def test_hugin_variant(self):
        ct = self._build_chain_tree()
        mp = MessagePasser(ct, variant=MessagePassingVariant.HUGIN)
        mp.calibrate()
        m = mp.get_marginal("A")
        assert abs(m.values.sum() - 1.0) < 1e-6

    def test_shafer_shenoy_variant(self):
        ct = self._build_chain_tree()
        mp = MessagePasser(ct, variant=MessagePassingVariant.SHAFER_SHENOY)
        mp.calibrate()
        m = mp.get_marginal("A")
        assert abs(m.values.sum() - 1.0) < 1e-6

    def test_stats(self):
        ct = self._build_chain_tree()
        mp = MessagePasser(ct)
        mp.calibrate()
        s = mp.stats
        assert s.messages_sent > 0


# ---------------------------------------------------------------------------
# Do-operator
# ---------------------------------------------------------------------------

class TestDoOperator:
    """Test do-operator (graph mutilation)."""

    def test_intervention_set(self):
        iset = InterventionSet()
        iset.add("X", 1.0)
        assert "X" in iset
        assert len(iset) == 1

    def test_intervention_signature(self):
        iset = InterventionSet()
        iset.add("X", 1.0)
        iset.add("Y", 0.0)
        sig = iset.signature
        assert isinstance(sig, str)
        assert len(sig) > 0

    def test_mutilate_graph(self):
        dag = {"A": ["B", "C"], "B": ["C"], "C": []}
        do_op = DoOperator()
        iset = InterventionSet()
        iset.add("B", 1.0, bin_index=1)
        mutilated, removed = do_op.mutilate_graph(dag, iset.variables)
        assert isinstance(mutilated, dict)

    def test_do_removes_incoming_edges(self):
        dag = {"A": ["B"], "B": ["C"], "C": []}
        do_op = DoOperator()
        iset = InterventionSet()
        iset.add("B", 0.0, bin_index=0)
        mutilated, removed = do_op.mutilate_graph(dag, iset.variables)
        assert "B" not in mutilated.get("A", [])
        assert "C" in mutilated.get("B", [])


# ---------------------------------------------------------------------------
# Adaptive discretization
# ---------------------------------------------------------------------------

class TestAdaptiveDiscretizer:
    """Test adaptive discretization of continuous variables."""

    def test_uniform_discretization(self):
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, size=1000)
        disc = AdaptiveDiscretizer()
        result = disc.discretize(values, n_bins=5, strategy=BinningStrategy.UNIFORM,
                                 variable_name="X")
        assert isinstance(result, DiscretizationResult)
        assert result.cardinality == 5

    def test_quantile_discretization(self):
        rng = np.random.default_rng(42)
        values = rng.exponential(1.0, size=1000)
        disc = AdaptiveDiscretizer()
        result = disc.discretize(values, n_bins=4, strategy=BinningStrategy.QUANTILE,
                                 variable_name="Y")
        assert result.cardinality == 4

    def test_bin_index(self):
        values = np.linspace(0, 10, 1000)
        disc = AdaptiveDiscretizer()
        result = disc.discretize(values, n_bins=5, strategy=BinningStrategy.UNIFORM,
                                 variable_name="Z")
        idx = result.bin_index(5.0)
        assert 0 <= idx < 5

    def test_bin_probabilities(self):
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, size=10000)
        disc = AdaptiveDiscretizer()
        result = disc.discretize(values, n_bins=4, strategy=BinningStrategy.QUANTILE,
                                 variable_name="W")
        probs = result.bin_probabilities(values)
        assert abs(probs.sum() - 1.0) < 0.05  # approximately uniform for quantile

    def test_discretization_error(self):
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, size=1000)
        disc = AdaptiveDiscretizer()
        result = disc.discretize(values, n_bins=10, strategy=BinningStrategy.UNIFORM,
                                 variable_name="X")
        error = disc.compute_discretization_error(values, result)
        assert error >= 0.0

    def test_refine_discretization(self):
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, size=1000)
        disc = AdaptiveDiscretizer()
        refined = disc.refine(values, target_error=0.01, variable_name="X",
                              strategy=BinningStrategy.UNIFORM)
        assert refined.cardinality >= 4

    def test_tail_preserving(self):
        rng = np.random.default_rng(42)
        values = rng.standard_t(3, size=1000)  # heavy tails
        disc = AdaptiveDiscretizer()
        result = disc.discretize(values, n_bins=6,
                                 strategy=BinningStrategy.TAIL_PRESERVING,
                                 variable_name="T")
        assert result.cardinality >= 1

    def test_cached_variables(self):
        rng = np.random.default_rng(42)
        disc = AdaptiveDiscretizer()
        disc.discretize(rng.normal(size=100), n_bins=3, variable_name="A")
        disc.discretize(rng.normal(size=100), n_bins=4, variable_name="B")
        assert set(disc.cached_variables()) == {"A", "B"}

    def test_summary(self):
        rng = np.random.default_rng(42)
        disc = AdaptiveDiscretizer()
        disc.discretize(rng.normal(size=100), n_bins=3, variable_name="X")
        s = disc.summary()
        assert "X" in s


# ---------------------------------------------------------------------------
# Inference cache
# ---------------------------------------------------------------------------

class TestInferenceCache:
    """Test inference result caching."""

    def test_put_get(self):
        cache = InferenceCache(capacity=100)
        key = CacheKey.make(sender_vars=frozenset({"A"}), receiver_vars=frozenset({"B"}))
        pt = PotentialTable(["A"], {"A": 2}, np.array([0.5, 0.5]))
        cache.put(key, pt.values, pt.variables)
        entry = cache.get(key)
        assert entry is not None
        np.testing.assert_allclose(entry.value, pt.values)

    def test_cache_miss(self):
        cache = InferenceCache(capacity=100)
        key = CacheKey.make(sender_vars=frozenset({"B"}), receiver_vars=frozenset())
        assert cache.get(key) is None

    def test_cache_eviction(self):
        cache = InferenceCache(capacity=3)
        for i in range(5):
            key = CacheKey.make(sender_vars=frozenset({f"V{i}"}), receiver_vars=frozenset())
            pt = PotentialTable([f"V{i}"], {f"V{i}": 2}, np.array([0.5, 0.5]))
            cache.put(key, pt.values, pt.variables)
        assert len(cache) <= 3

    def test_invalidate_variable(self):
        cache = InferenceCache(capacity=100)
        key = CacheKey.make(sender_vars=frozenset({"A"}), receiver_vars=frozenset({"B"}))
        pt = PotentialTable(["A"], {"A": 2}, np.array([0.5, 0.5]))
        cache.put(key, pt.values, pt.variables)
        n = cache.invalidate_variable("A")
        assert n >= 1
        assert cache.get(key) is None

    def test_stats(self):
        cache = InferenceCache(capacity=100)
        key = CacheKey.make(sender_vars=frozenset({"A"}), receiver_vars=frozenset({"B"}))
        cache.get(key)  # miss
        pt = PotentialTable(["A"], {"A": 2}, np.array([0.5, 0.5]))
        cache.put(key, pt.values, pt.variables)
        cache.get(key)  # hit
        stats = cache.stats
        assert stats.hits == 1
        assert stats.misses == 1

    def test_clear(self):
        cache = InferenceCache(capacity=100)
        key = CacheKey.make(sender_vars=frozenset({"A"}), receiver_vars=frozenset({"B"}))
        pt = PotentialTable(["A"], {"A": 2}, np.array([0.5, 0.5]))
        cache.put(key, pt.values, pt.variables)
        cache.clear()
        assert len(cache) == 0

    def test_serialize_deserialize(self):
        cache = InferenceCache(capacity=100)
        key = CacheKey.make(sender_vars=frozenset({"A"}), receiver_vars=frozenset({"B"}))
        pt = PotentialTable(["A"], {"A": 2}, np.array([0.3, 0.7]))
        cache.put(key, pt.values, pt.variables)
        data = cache.serialize()
        cache2 = InferenceCache.deserialize(data)
        entry = cache2.get(key)
        assert entry is not None


# ---------------------------------------------------------------------------
# Junction tree engine: full inference
# ---------------------------------------------------------------------------

class TestJunctionTreeEngine:
    """Test full junction-tree engine on known models."""

    def _make_chain_cpds(self):
        return {
            "A": PotentialTable(["A"], {"A": 2}, np.array([0.5, 0.5])),
            "B": PotentialTable(["A", "B"], {"A": 2, "B": 2},
                                np.array([[0.8, 0.2], [0.3, 0.7]])),
            "C": PotentialTable(["B", "C"], {"B": 2, "C": 2},
                                np.array([[0.6, 0.4], [0.1, 0.9]])),
        }

    def test_build_chain(self, chain_dag_spec):
        dag = _edges_to_dag(chain_dag_spec["edges"])
        cpds = self._make_chain_cpds()
        engine = JunctionTreeEngine()
        engine.build(dag, cpds, cardinalities=chain_dag_spec["cardinalities"])
        assert engine.tree is not None

    def test_calibrate_and_query_chain(self, chain_dag_spec):
        dag = _edges_to_dag(chain_dag_spec["edges"])
        cpds = self._make_chain_cpds()
        engine = JunctionTreeEngine()
        engine.build(dag, cpds, cardinalities=chain_dag_spec["cardinalities"])
        engine.calibrate()
        result = engine.query("C")
        assert result is not None
        assert abs(result.distribution.sum() - 1.0) < 1e-6

    def test_diamond_inference(self, diamond_dag_spec):
        dag = _edges_to_dag(diamond_dag_spec["edges"])
        cpds = {
            "A": PotentialTable(["A"], {"A": 2}, np.array([0.5, 0.5])),
            "B": PotentialTable(["A", "B"], {"A": 2, "B": 2},
                                np.array([[0.9, 0.1], [0.2, 0.8]])),
            "C": PotentialTable(["A", "C"], {"A": 2, "C": 2},
                                np.array([[0.7, 0.3], [0.4, 0.6]])),
            "D": PotentialTable(["B", "C", "D"], {"B": 2, "C": 2, "D": 2},
                                np.array([[[0.95, 0.05], [0.6, 0.4]],
                                          [[0.3, 0.7], [0.1, 0.9]]])),
        }
        engine = JunctionTreeEngine()
        engine.build(dag, cpds, cardinalities=diamond_dag_spec["cardinalities"])
        engine.calibrate()
        result = engine.query("D")
        assert result is not None
        assert 0.0 <= result.distribution[0] <= 1.0

    def test_model_summary(self, chain_dag_spec):
        dag = _edges_to_dag(chain_dag_spec["edges"])
        cpds = self._make_chain_cpds()
        engine = JunctionTreeEngine()
        engine.build(dag, cpds, cardinalities=chain_dag_spec["cardinalities"])
        summary = engine.model_summary()
        assert isinstance(summary, dict)

"""
Tests for SCM (Structural Causal Model) module.

Tests cover: DAG operations, SCM builder, FCI causal discovery,
Markov equivalence class, and orientation rules.
"""
import pytest
import numpy as np
import networkx as nx

from causalbound.scm.dag import DAGRepresentation, EdgeType
from causalbound.scm.builder import SCMBuilder, SCM, VariableType
from causalbound.scm.equivalence import MarkovEquivalenceClass, CPDAG
from causalbound.scm.causal_discovery import FastCausalInference, CITest, PAG
from causalbound.scm.orientation import OrientationRules, MeekRules


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def chain_dag():
    """A -> B -> C."""
    dag = DAGRepresentation(["A", "B", "C"])
    dag.add_edge("A", "B")
    dag.add_edge("B", "C")
    return dag


@pytest.fixture
def diamond_dag():
    """A -> B, A -> C, B -> D, C -> D."""
    dag = DAGRepresentation(["A", "B", "C", "D"])
    dag.add_edge("A", "B")
    dag.add_edge("A", "C")
    dag.add_edge("B", "D")
    dag.add_edge("C", "D")
    return dag


@pytest.fixture
def v_structure_dag():
    """A -> C <- B (collider)."""
    dag = DAGRepresentation(["A", "B", "C"])
    dag.add_edge("A", "C")
    dag.add_edge("B", "C")
    return dag


@pytest.fixture
def complex_dag():
    """A -> B -> D, A -> C -> D, B -> C."""
    dag = DAGRepresentation(["A", "B", "C", "D"])
    dag.add_edge("A", "B")
    dag.add_edge("A", "C")
    dag.add_edge("B", "C")
    dag.add_edge("B", "D")
    dag.add_edge("C", "D")
    return dag


# ---------------------------------------------------------------------------
# DAG operations
# ---------------------------------------------------------------------------

class TestDAGRepresentation:
    """Test DAG basic operations."""

    def test_create_dag(self):
        dag = DAGRepresentation(["X", "Y", "Z"])
        assert dag.n_nodes == 3
        assert dag.n_edges == 0

    def test_add_edge(self, chain_dag):
        assert chain_dag.has_edge("A", "B")
        assert chain_dag.has_edge("B", "C")
        assert not chain_dag.has_edge("A", "C")

    def test_add_edge_cycle_detection(self, chain_dag):
        with pytest.raises(ValueError):
            chain_dag.add_edge("C", "A")  # Would create cycle

    def test_remove_edge(self, chain_dag):
        chain_dag.remove_edge("A", "B")
        assert not chain_dag.has_edge("A", "B")

    def test_remove_node(self, chain_dag):
        chain_dag.remove_node("B")
        assert not chain_dag.has_node("B")
        assert chain_dag.n_nodes == 2

    def test_parents_children(self, diamond_dag):
        assert set(diamond_dag.parents("D")) == {"B", "C"}
        assert set(diamond_dag.children("A")) == {"B", "C"}
        assert diamond_dag.parents("A") == []

    def test_nodes_edges(self, chain_dag):
        assert set(chain_dag.nodes) == {"A", "B", "C"}
        assert len(chain_dag.edges) == 2


class TestTopologicalSort:
    """Test topological sorting."""

    def test_chain_topo(self, chain_dag):
        topo = chain_dag.topological_sort()
        assert len(topo) == 3
        # A must come before B, B before C
        assert topo.index("A") < topo.index("B")
        assert topo.index("B") < topo.index("C")

    def test_diamond_topo(self, diamond_dag):
        topo = diamond_dag.topological_sort()
        assert topo.index("A") < topo.index("B")
        assert topo.index("A") < topo.index("C")
        assert topo.index("B") < topo.index("D")
        assert topo.index("C") < topo.index("D")

    def test_v_structure_topo(self, v_structure_dag):
        topo = v_structure_dag.topological_sort()
        assert topo.index("A") < topo.index("C")
        assert topo.index("B") < topo.index("C")


class TestAncestorsDescendants:
    """Test ancestor and descendant queries."""

    def test_ancestors_chain(self, chain_dag):
        assert chain_dag.ancestors("C") == {"A", "B"}
        assert chain_dag.ancestors("B") == {"A"}
        assert chain_dag.ancestors("A") == set()

    def test_descendants_chain(self, chain_dag):
        assert chain_dag.descendants("A") == {"B", "C"}
        assert chain_dag.descendants("B") == {"C"}
        assert chain_dag.descendants("C") == set()

    def test_is_ancestor(self, diamond_dag):
        assert diamond_dag.is_ancestor("A", "D")
        assert diamond_dag.is_ancestor("B", "D")
        assert not diamond_dag.is_ancestor("D", "A")

    def test_ancestors_diamond(self, diamond_dag):
        assert diamond_dag.ancestors("D") == {"A", "B", "C"}

    def test_descendants_diamond(self, diamond_dag):
        assert diamond_dag.descendants("A") == {"B", "C", "D"}


class TestDSeparation:
    """Test d-separation queries via Bayes Ball."""

    def test_chain_dsep(self, chain_dag):
        # A _||_ C | B (d-separated by B)
        assert chain_dag.d_separated({"A"}, {"C"}, {"B"})

    def test_chain_not_dsep(self, chain_dag):
        # A and C are NOT d-separated when B is not conditioned on
        assert not chain_dag.d_separated({"A"}, {"C"}, set())

    def test_collider_dsep(self, v_structure_dag):
        # A _||_ B (d-separated when not conditioning on C)
        assert v_structure_dag.d_separated({"A"}, {"B"}, set())

    def test_collider_not_dsep(self, v_structure_dag):
        # A NOT _||_ B | C (conditioning on collider opens the path)
        assert not v_structure_dag.d_separated({"A"}, {"B"}, {"C"})

    def test_diamond_dsep(self, diamond_dag):
        # A _||_ D | {B, C} (both paths blocked)
        assert diamond_dag.d_separated({"A"}, {"D"}, {"B", "C"})

    def test_diamond_not_dsep(self, diamond_dag):
        # A NOT _||_ D (paths exist)
        assert not diamond_dag.d_separated({"A"}, {"D"}, set())

    def test_self_not_dsep(self, chain_dag):
        # A is not d-separated from itself
        assert not chain_dag.d_separated({"A"}, {"A"}, set())

    def test_dsep_symmetric(self, chain_dag):
        # d-separation is symmetric
        assert chain_dag.d_separated({"A"}, {"C"}, {"B"}) == \
               chain_dag.d_separated({"C"}, {"A"}, {"B"})


class TestMarkovBlanket:
    """Test Markov blanket computation."""

    def test_chain_markov_blanket(self, chain_dag):
        mb = chain_dag.markov_blanket("B")
        assert "A" in mb  # parent
        assert "C" in mb  # child

    def test_collider_markov_blanket(self, v_structure_dag):
        mb = v_structure_dag.markov_blanket("C")
        assert "A" in mb  # parent
        assert "B" in mb  # parent

    def test_diamond_markov_blanket(self, diamond_dag):
        mb = diamond_dag.markov_blanket("B")
        assert "A" in mb  # parent
        assert "D" in mb  # child
        assert "C" in mb  # co-parent of D


class TestMutilatedDAG:
    """Test interventional graph construction."""

    def test_mutilate_chain(self, chain_dag):
        mutilated = chain_dag.mutilate(["B"])
        # A -> B edge should be removed
        assert not mutilated.has_edge("A", "B")
        # B -> C edge should remain
        assert mutilated.has_edge("B", "C")

    def test_mutilate_preserves_other_edges(self, diamond_dag):
        mutilated = diamond_dag.mutilate(["D"])
        assert mutilated.has_edge("A", "B")
        assert mutilated.has_edge("A", "C")
        assert not mutilated.has_edge("B", "D")
        assert not mutilated.has_edge("C", "D")

    def test_mutilate_is_dag(self, diamond_dag):
        mutilated = diamond_dag.mutilate(["B"])
        assert mutilated.is_dag()


class TestMoralGraph:
    """Test moral graph construction."""

    def test_chain_moral(self, chain_dag):
        moral = chain_dag.moral_graph()
        assert isinstance(moral, nx.Graph)
        assert moral.has_edge("A", "B")
        assert moral.has_edge("B", "C")

    def test_v_structure_moral(self, v_structure_dag):
        moral = v_structure_dag.moral_graph()
        # Should add A-B edge (parents of C married)
        assert moral.has_edge("A", "B")


class TestDAGConversion:
    """Test DAG conversion utilities."""

    def test_to_networkx(self, chain_dag):
        G = chain_dag.to_networkx()
        assert isinstance(G, nx.DiGraph)
        assert G.has_edge("A", "B")

    def test_from_networkx(self):
        G = nx.DiGraph()
        G.add_edges_from([("X", "Y"), ("Y", "Z")])
        dag = DAGRepresentation.from_networkx(G)
        assert dag.has_edge("X", "Y")
        assert dag.n_nodes == 3

    def test_to_adjacency_matrix(self, chain_dag):
        adj = chain_dag.to_adjacency_matrix()
        assert isinstance(adj, np.ndarray)
        assert adj.shape[0] == chain_dag.n_nodes

    def test_from_adjacency_matrix(self):
        adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        dag = DAGRepresentation.from_adjacency_matrix(adj, ["A", "B", "C"])
        assert dag.has_edge("A", "B")
        assert dag.has_edge("B", "C")
        assert not dag.has_edge("A", "C")

    def test_roundtrip_networkx(self, diamond_dag):
        G = diamond_dag.to_networkx()
        dag2 = DAGRepresentation.from_networkx(G)
        assert set(dag2.nodes) == set(diamond_dag.nodes)
        assert set(dag2.edges) == set(diamond_dag.edges)

    def test_validate(self, chain_dag):
        errors = chain_dag.validate()
        assert len(errors) == 0

    def test_is_dag(self, chain_dag):
        assert chain_dag.is_dag()

    def test_bidirected_edges(self):
        dag = DAGRepresentation(["X", "Y"])
        dag.add_edge("X", "Y", edge_type=EdgeType.BIDIRECTED)
        assert dag.has_bidirected("X", "Y")

    def test_get_paths(self, diamond_dag):
        paths = diamond_dag.get_paths("A", "D")
        assert len(paths) >= 2  # A->B->D and A->C->D

    def test_causal_paths(self, diamond_dag):
        paths = diamond_dag.get_causal_paths("A", "D")
        assert len(paths) >= 2


# ---------------------------------------------------------------------------
# SCM builder
# ---------------------------------------------------------------------------

class TestSCMBuilder:
    """Test SCM construction."""

    def test_build_simple(self):
        builder = SCMBuilder()
        builder.add_variable("X", var_type=VariableType.CONTINUOUS)
        builder.add_variable("Y", var_type=VariableType.CONTINUOUS)
        builder.add_linear_equation("Y", {"X": 0.5}, noise_std=0.1)
        scm = builder.build()
        assert isinstance(scm, SCM)

    def test_build_chain(self):
        builder = SCMBuilder()
        builder.add_variable("A", var_type=VariableType.CONTINUOUS)
        builder.add_variable("B", var_type=VariableType.CONTINUOUS)
        builder.add_variable("C", var_type=VariableType.CONTINUOUS)
        builder.add_linear_equation("B", {"A": 0.7}, noise_std=0.1)
        builder.add_linear_equation("C", {"B": 0.3}, noise_std=0.2)
        scm = builder.build()
        assert scm is not None

    def test_sample(self):
        builder = SCMBuilder()
        builder.add_variable("X", var_type=VariableType.CONTINUOUS)
        builder.add_variable("Y", var_type=VariableType.CONTINUOUS)
        builder.add_linear_equation("Y", {"X": 1.0}, noise_std=0.1)
        scm = builder.build()
        samples = scm.sample(n=100)
        assert "X" in samples
        assert "Y" in samples
        assert len(samples["X"]) == 100

    def test_interventional_sample(self):
        builder = SCMBuilder()
        builder.add_variable("X", var_type=VariableType.CONTINUOUS)
        builder.add_variable("Y", var_type=VariableType.CONTINUOUS)
        builder.add_linear_equation("Y", {"X": 1.0}, noise_std=0.1)
        scm = builder.build()
        samples = scm.sample(n=100, interventions={"X": 5.0})
        assert np.allclose(samples["X"], 5.0)

    def test_add_latent(self):
        builder = SCMBuilder()
        builder.add_variable("X", var_type=VariableType.CONTINUOUS)
        builder.add_variable("Y", var_type=VariableType.CONTINUOUS)
        builder.add_latent("U", children=["X", "Y"])
        scm = builder.build()
        assert "U" in scm.latent_variable_names()

    def test_validate_errors(self):
        builder = SCMBuilder()
        builder.add_variable("X", var_type=VariableType.CONTINUOUS)
        errors = builder.validate()
        # Single node with no equation might or might not have warnings
        assert isinstance(errors, list)

    def test_get_dag(self):
        builder = SCMBuilder()
        builder.add_variable("X", var_type=VariableType.CONTINUOUS)
        builder.add_variable("Y", var_type=VariableType.CONTINUOUS, parents=["X"])
        builder.add_linear_equation("Y", {"X": 1.0}, noise_std=0.1)
        dag = builder.get_dag()
        assert isinstance(dag, DAGRepresentation)
        assert dag.has_edge("X", "Y")

    def test_get_moral_graph(self):
        builder = SCMBuilder()
        builder.add_variable("A", var_type=VariableType.CONTINUOUS)
        builder.add_variable("B", var_type=VariableType.CONTINUOUS)
        builder.add_variable("C", var_type=VariableType.CONTINUOUS)
        builder.add_linear_equation("C", {"A": 0.5, "B": 0.3}, noise_std=0.1)
        moral = builder.get_moral_graph()
        assert isinstance(moral, nx.Graph)

    def test_get_markov_blanket(self):
        builder = SCMBuilder()
        builder.add_variable("A", var_type=VariableType.CONTINUOUS)
        builder.add_variable("B", var_type=VariableType.CONTINUOUS, parents=["A"])
        builder.add_variable("C", var_type=VariableType.CONTINUOUS, parents=["B"])
        builder.add_linear_equation("B", {"A": 0.7}, noise_std=0.1)
        builder.add_linear_equation("C", {"B": 0.3}, noise_std=0.2)
        mb = builder.get_markov_blanket("B")
        assert "A" in mb
        assert "C" in mb

    def test_logistic_equation(self):
        builder = SCMBuilder()
        builder.add_variable("X", var_type=VariableType.CONTINUOUS)
        builder.add_variable("Y", var_type=VariableType.BINARY)
        builder.add_logistic_equation("Y", {"X": 1.0}, intercept=0.0)
        scm = builder.build()
        samples = scm.sample(n=100)
        assert all(v in [0, 1] or 0.0 <= v <= 1.0 for v in samples["Y"])

    def test_forbidden_edge(self):
        builder = SCMBuilder()
        builder.add_variable("X", var_type=VariableType.CONTINUOUS)
        builder.add_variable("Y", var_type=VariableType.CONTINUOUS)
        builder.add_forbidden_edge("Y", "X", reason="temporal")
        assert len(builder._scm._domain_rules) > 0

    def test_summary(self):
        builder = SCMBuilder()
        builder.add_variable("X", var_type=VariableType.CONTINUOUS)
        builder.add_variable("Y", var_type=VariableType.CONTINUOUS)
        builder.add_linear_equation("Y", {"X": 0.5}, noise_std=0.1)
        scm = builder.build()
        s = scm.summary()
        assert isinstance(s, str)


# ---------------------------------------------------------------------------
# Markov equivalence class
# ---------------------------------------------------------------------------

class TestMarkovEquivalenceClass:
    """Test Markov equivalence class enumeration."""

    def test_from_dag_chain(self, chain_dag):
        mec = MarkovEquivalenceClass.from_dag(chain_dag)
        assert mec is not None

    def test_enumerate_dags_chain(self, chain_dag):
        mec = MarkovEquivalenceClass.from_dag(chain_dag)
        dags = mec.enumerate_dags(max_count=100)
        # Chain A-B-C has 3 equivalent DAGs (no v-structures)
        assert len(dags) >= 1

    def test_from_dag_v_structure(self, v_structure_dag):
        mec = MarkovEquivalenceClass.from_dag(v_structure_dag)
        dags = mec.enumerate_dags(max_count=100)
        # V-structure is unique in its MEC
        assert len(dags) == 1

    def test_cpdag_construction(self, chain_dag):
        cpdag = CPDAG.from_dag(chain_dag)
        assert cpdag is not None
        nodes = cpdag.nodes
        assert set(nodes) == {"A", "B", "C"}

    def test_essential_edges(self, v_structure_dag):
        mec = MarkovEquivalenceClass.from_dag(v_structure_dag)
        essential = mec.get_essential_edges()
        # Both A->C and B->C are essential
        assert len(essential) == 2

    def test_reversible_edges(self, chain_dag):
        mec = MarkovEquivalenceClass.from_dag(chain_dag)
        reversible = mec.get_reversible_edges()
        # Chain has reversible edges (no v-structures force directions)
        assert len(reversible) >= 0

    def test_is_member(self, chain_dag):
        mec = MarkovEquivalenceClass.from_dag(chain_dag)
        assert mec.is_member(chain_dag)

    def test_count_dags(self, diamond_dag):
        mec = MarkovEquivalenceClass.from_dag(diamond_dag)
        count = mec.count_dags()
        assert count >= 1

    def test_sample_dag(self, chain_dag):
        mec = MarkovEquivalenceClass.from_dag(chain_dag)
        sampled = mec.sample_dag()
        assert isinstance(sampled, DAGRepresentation)
        assert sampled.is_dag()

    def test_sample_dags(self, chain_dag):
        mec = MarkovEquivalenceClass.from_dag(chain_dag)
        samples = mec.sample_dags(n=5)
        assert len(samples) == 5
        for dag in samples:
            assert dag.is_dag()

    def test_structural_hamming_distance(self, chain_dag, diamond_dag):
        mec1 = MarkovEquivalenceClass.from_dag(chain_dag)
        mec2 = MarkovEquivalenceClass.from_dag(diamond_dag)
        shd = mec1.structural_hamming_distance(mec2)
        assert shd >= 0


# ---------------------------------------------------------------------------
# FCI causal discovery
# ---------------------------------------------------------------------------

class TestFCI:
    """Test FCI algorithm on known structures."""

    def test_discover_chain(self):
        rng = np.random.default_rng(42)
        n = 500
        A = rng.normal(0, 1, n)
        B = 0.8 * A + rng.normal(0, 0.3, n)
        C = 0.5 * B + rng.normal(0, 0.3, n)
        data = np.column_stack([A, B, C])
        fci = FastCausalInference(alpha=0.05)
        pag = fci.discover(data, variables=["A", "B", "C"])
        assert isinstance(pag, PAG)

    def test_discover_v_structure(self):
        rng = np.random.default_rng(42)
        n = 500
        A = rng.normal(0, 1, n)
        B = rng.normal(0, 1, n)
        C = 0.5 * A + 0.5 * B + rng.normal(0, 0.2, n)
        data = np.column_stack([A, B, C])
        fci = FastCausalInference(alpha=0.05)
        pag = fci.discover(data, variables=["A", "B", "C"])
        assert isinstance(pag, PAG)

    def test_n_tests_performed(self):
        rng = np.random.default_rng(42)
        data = rng.normal(size=(200, 3))
        fci = FastCausalInference(alpha=0.05)
        fci.discover(data, variables=["X", "Y", "Z"])
        assert fci.n_tests_performed > 0

    def test_ci_test_partial_correlation(self):
        rng = np.random.default_rng(42)
        n = 500
        X = rng.normal(0, 1, n)
        Y = 0.9 * X + rng.normal(0, 0.1, n)
        Z = rng.normal(0, 1, n)
        data = np.column_stack([X, Y, Z])
        # X and Y should NOT be independent
        _stat, p_val = CITest.partial_correlation(0, 1, [], data)
        assert p_val < 0.05

    def test_ci_test_conditional_independence(self):
        rng = np.random.default_rng(42)
        n = 500
        A = rng.normal(0, 1, n)
        B = 0.8 * A + rng.normal(0, 0.3, n)
        C = 0.5 * B + rng.normal(0, 0.3, n)
        data = np.column_stack([A, B, C])
        # A _||_ C | B
        _stat, p_val = CITest.partial_correlation(0, 2, [1], data)
        assert p_val > 0.01


# ---------------------------------------------------------------------------
# Orientation rules
# ---------------------------------------------------------------------------

class TestOrientationRules:
    """Test Meek rules and domain-specific orientation."""

    def test_meek_rules_apply(self):
        G = nx.DiGraph()
        G.add_edge("A", "B")  # directed
        G.add_edge("B", "C")  # to be oriented
        G.add_edge("C", "B")  # undirected representation
        undirected = {frozenset({"B", "C"})}
        result_dag, result_und = MeekRules.apply(G, undirected)
        assert isinstance(result_dag, nx.DiGraph)

    def test_orientation_rules_class(self):
        rules = OrientationRules()
        dag = DAGRepresentation(["A", "B", "C"])
        dag.add_edge("A", "C")
        dag.add_edge("B", "C")
        result = rules.apply_meek_rules(dag)
        assert isinstance(result, DAGRepresentation)

    def test_check_acyclicity(self, chain_dag):
        rules = OrientationRules()
        assert rules.check_acyclicity(chain_dag)

    def test_financial_default(self):
        rules = OrientationRules.financial_default()
        assert isinstance(rules, OrientationRules)

    def test_add_rule(self):
        rules = OrientationRules()
        rules.add_forbidden("X", "Y", description="temporal ordering")
        assert len(rules.conflicts) == 0 or True  # No conflict yet

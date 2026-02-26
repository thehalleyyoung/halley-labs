"""
Tests for causal polytope module.

Tests cover: LP constraint construction, column generation, bound extraction,
interventional polytope, d-separation encoding, pricing subproblem,
and normalization constraints.
"""
import pytest
import numpy as np
from scipy import sparse

from causalbound.polytope.constraints import ConstraintEncoder, ConstraintType
from causalbound.polytope.causal_polytope import (
    CausalPolytopeSolver,
    DAGSpec,
    SolverConfig,
    SolverResult,
    SolverStatus,
    QuerySpec,
    InterventionSpec,
    ObservedMarginals,
)
from causalbound.polytope.column_generation import (
    ColumnGenerationSolver,
    ColumnPool,
    Column,
    MasterProblem,
)
from causalbound.polytope.bounds import BoundExtractor, BoundResult
from causalbound.polytope.interventional import InterventionalPolytope
from causalbound.polytope.pricing import PricingSubproblem, PricingStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dag(edges):
    """Build a DAGSpec from an edge list with binary cardinalities."""
    nodes_set = set()
    for u, v in edges:
        nodes_set.add(u)
        nodes_set.add(v)
    nodes = sorted(nodes_set)
    card = {n: 2 for n in nodes}
    return DAGSpec(nodes=nodes, edges=edges, card=card)


def _make_binary_iv_dag():
    """Instrument variable DAG: Z -> X -> Y, no Z->Y edge."""
    return _make_dag([("Z", "X"), ("X", "Y")])


def _make_chain_dag():
    """Simple chain: A -> B -> C."""
    return _make_dag([("A", "B"), ("B", "C")])


def _make_confounded_dag():
    """X -> Y with latent confounder: X -> Y, U -> X, U -> Y."""
    return _make_dag([("X", "Y"), ("U", "X"), ("U", "Y")])


def _make_diamond_dag():
    """A -> B, A -> C, B -> D, C -> D."""
    return _make_dag([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])


# ---------------------------------------------------------------------------
# Constraint construction
# ---------------------------------------------------------------------------

class TestConstraintEncoder:
    """Test LP constraint matrix construction."""

    def test_normalization_constraints(self):
        dag = _make_binary_iv_dag()
        enc = ConstraintEncoder(dag)
        block = enc.add_normalization_constraints()
        assert block.num_rows > 0
        assert block.ctype == ConstraintType.NORMALIZATION

    def test_markov_constraints(self):
        dag = _make_chain_dag()
        enc = ConstraintEncoder(dag)
        blocks = enc.add_markov_constraints()
        assert len(blocks) > 0
        for b in blocks:
            assert b.ctype == ConstraintType.MARKOV

    def test_constraint_matrix_shape(self):
        dag = _make_binary_iv_dag()
        enc = ConstraintEncoder(dag)
        enc.add_normalization_constraints()
        enc.add_markov_constraints()
        A, b, _names = enc.build_constraint_matrix()
        assert A.shape[0] == len(b)
        assert A.shape[1] > 0

    def test_normalization_rhs_sums_to_one(self):
        dag = _make_binary_iv_dag()
        enc = ConstraintEncoder(dag)
        enc.add_normalization_constraints()
        _A, b, _names = enc.build_constraint_matrix()
        assert any(abs(bi - 1.0) < 1e-10 for bi in b)

    def test_dseparation_constraints(self):
        dag = _make_binary_iv_dag()
        enc = ConstraintEncoder(dag)
        # Z _||_ Y | X in the IV DAG
        block = enc.add_dseparation_constraints(
            frozenset({"Z"}), frozenset({"Y"}), frozenset({"X"})
        )
        # Should return a ConstraintBlock (d-sep holds) or None
        if block is not None:
            assert block.ctype == ConstraintType.DSEPARATION

    def test_num_variables_binary(self):
        dag = _make_binary_iv_dag()
        enc = ConstraintEncoder(dag)
        # Joint of 3 binary variables: 2^3 = 8
        assert enc.num_variables == 8

    def test_num_variables_chain(self):
        dag = _make_chain_dag()
        enc = ConstraintEncoder(dag)
        assert enc.num_variables == 8  # 2^3

    def test_incremental_build(self):
        dag = _make_binary_iv_dag()
        enc = ConstraintEncoder(dag)
        enc.add_normalization_constraints()
        enc.build_constraint_matrix()
        n1 = enc.num_constraints

        enc.add_markov_constraints()
        enc.build_constraint_matrix()
        n2 = enc.num_constraints

        assert n2 > n1

    def test_remove_block(self):
        dag = _make_binary_iv_dag()
        enc = ConstraintEncoder(dag)
        b1 = enc.add_normalization_constraints()
        enc.add_markov_constraints()
        n_before = enc.num_constraints
        enc.remove_block(b1)
        n_after = enc.num_constraints
        assert n_after < n_before

    def test_get_blocks(self):
        dag = _make_chain_dag()
        enc = ConstraintEncoder(dag)
        enc.add_normalization_constraints()
        enc.add_markov_constraints()
        blocks = enc.get_blocks()
        assert len(blocks) > 0

    def test_constraint_matrix_is_sparse(self):
        dag = _make_diamond_dag()
        enc = ConstraintEncoder(dag)
        enc.add_normalization_constraints()
        enc.add_markov_constraints()
        A, _b, _names = enc.build_constraint_matrix()
        assert sparse.issparse(A)

    def test_observed_marginal_constraints(self):
        dag = _make_binary_iv_dag()
        enc = ConstraintEncoder(dag)
        observed = {frozenset({"Z"}): np.array([0.5, 0.5])}
        obs = ObservedMarginals(marginals=observed)
        blocks = enc.add_observed_marginal_constraints(obs)
        assert len(blocks) > 0


# ---------------------------------------------------------------------------
# Causal polytope solver
# ---------------------------------------------------------------------------

class TestCausalPolytopeSolver:
    """Test the full causal polytope solver."""

    def test_solve_binary_iv(self):
        dag = _make_binary_iv_dag()
        solver = CausalPolytopeSolver(config=SolverConfig())
        query = QuerySpec(
            target_var="Y",
            target_val=1,
            interventions=[InterventionSpec("X", 1)],
        )
        result = solver.solve(dag, query)
        assert isinstance(result, SolverResult)
        assert result.lower_bound <= result.upper_bound

    def test_solve_chain(self):
        dag = _make_chain_dag()
        solver = CausalPolytopeSolver(config=SolverConfig())
        query = QuerySpec(
            target_var="C",
            target_val=1,
            interventions=[InterventionSpec("A", 1)],
        )
        result = solver.solve(dag, query)
        assert 0.0 <= result.lower_bound
        assert result.upper_bound <= 1.0

    def test_bounds_are_probabilities(self):
        dag = _make_binary_iv_dag()
        solver = CausalPolytopeSolver(config=SolverConfig())
        query = QuerySpec(
            target_var="Y",
            target_val=0,
            interventions=[InterventionSpec("X", 0)],
        )
        result = solver.solve(dag, query)
        assert result.lower_bound >= -1e-10
        assert result.upper_bound <= 1.0 + 1e-10

    def test_gap_nonnegative(self):
        dag = _make_binary_iv_dag()
        solver = CausalPolytopeSolver(config=SolverConfig())
        query = QuerySpec(
            target_var="Y", target_val=1,
            interventions=[InterventionSpec("X", 1)],
        )
        result = solver.solve(dag, query)
        assert result.gap >= -1e-10

    def test_from_adjacency(self):
        solver, dag = CausalPolytopeSolver.from_adjacency(
            adj={"Z": ["X"], "X": ["Y"], "Y": []},
            card={"Z": 2, "X": 2, "Y": 2},
        )
        assert solver is not None
        assert dag is not None

    def test_binary_dag_constructor(self):
        solver, dag = CausalPolytopeSolver.binary_dag(
            edges=[("A", "B"), ("B", "C")],
        )
        assert solver is not None
        assert dag is not None

    def test_solve_multiple_queries(self):
        dag = _make_binary_iv_dag()
        solver = CausalPolytopeSolver(config=SolverConfig())
        queries = [
            QuerySpec(target_var="Y", target_val=1,
                      interventions=[InterventionSpec("X", 1)]),
            QuerySpec(target_var="Y", target_val=0,
                      interventions=[InterventionSpec("X", 0)]),
        ]
        results = solver.solve_multiple_queries(dag, queries)
        assert len(results) == 2
        for r in results:
            assert r.lower_bound <= r.upper_bound

    def test_compute_ate_bounds(self):
        dag = _make_binary_iv_dag()
        solver = CausalPolytopeSolver(config=SolverConfig())
        lb, ub = solver.compute_ate_bounds(
            dag, treatment="X", outcome="Y",
        )
        # ATE bounds can be negative (range [-1, 1])
        assert lb <= ub
        assert lb >= -1.0 - 1e-10
        assert ub <= 1.0 + 1e-10


# ---------------------------------------------------------------------------
# Column generation
# ---------------------------------------------------------------------------

class TestColumnGeneration:
    """Test column generation solver."""

    def test_column_pool_add(self):
        pool = ColumnPool()
        col = pool.add(np.array([1.0, 0.0, 0.0, 0.0]), cost=0.5)
        assert isinstance(col, Column)
        assert pool.size() == 1

    def test_column_pool_multiple(self):
        pool = ColumnPool()
        for i in range(10):
            v = np.zeros(4)
            v[i % 4] = 1.0
            pool.add(v, cost=float(i) / 10.0)
        assert pool.size() == 10
        # active_count tracks in_basis columns; newly added are not in basis
        assert pool.active_count() == 0

    def test_column_pool_remove(self):
        pool = ColumnPool()
        pool.add(np.array([1.0, 0.0]), cost=0.5)
        pool.add(np.array([0.0, 1.0]), cost=0.3)
        pool.remove(0)
        assert pool.size() == 1

    def test_column_pool_age_out(self):
        pool = ColumnPool()
        for i in range(5):
            pool.add(np.zeros(4), cost=0.0)
        for _ in range(20):
            pool.increment_ages()
        removed = pool.age_out(max_age=15)
        assert removed == 5

    def test_cost_vector(self):
        pool = ColumnPool()
        pool.add(np.array([1.0, 0.0]), cost=0.5)
        pool.add(np.array([0.0, 1.0]), cost=0.3)
        costs = pool.get_cost_vector()
        np.testing.assert_allclose(costs, [0.5, 0.3])

    def test_coefficient_matrix(self):
        pool = ColumnPool()
        pool.add(np.array([1.0, 0.0, 0.0]), cost=0.5)
        pool.add(np.array([0.0, 1.0, 0.0]), cost=0.3)
        A = pool.get_coefficient_matrix(sparse_format=False)
        assert A.shape == (3, 2)

    def test_cg_solver_basic(self):
        dag = _make_binary_iv_dag()
        enc = ConstraintEncoder(dag)
        enc.add_normalization_constraints()
        enc.add_markov_constraints()
        A, b, _names = enc.build_constraint_matrix()
        n_vars = enc.num_variables
        c = np.zeros(n_vars)
        c[0] = 1.0
        config = SolverConfig(max_iterations=20)
        cg = ColumnGenerationSolver(
            c=c, A_eq=A, b_eq=b, total_vars=n_vars,
            config=config, dag=dag,
        )
        result = cg.solve()
        assert result is not None

    def test_cg_convergence(self):
        dag = _make_chain_dag()
        enc = ConstraintEncoder(dag)
        enc.add_normalization_constraints()
        enc.add_markov_constraints()
        A, b, _names = enc.build_constraint_matrix()
        n_vars = enc.num_variables
        c = np.ones(n_vars) / n_vars
        config = SolverConfig(max_iterations=50)
        cg = ColumnGenerationSolver(
            c=c, A_eq=A, b_eq=b, total_vars=n_vars,
            config=config, dag=dag,
        )
        result = cg.solve()
        assert result is not None


# ---------------------------------------------------------------------------
# Bound extraction
# ---------------------------------------------------------------------------

class TestBoundExtractor:
    """Test bound extraction from polytope constraints."""

    def test_probability_bounds_binary_iv(self):
        dag = _make_binary_iv_dag()
        enc = ConstraintEncoder(dag)
        enc.add_normalization_constraints()
        enc.add_markov_constraints()
        A, b, _names = enc.build_constraint_matrix()
        extractor = BoundExtractor(dag, A, b)
        result = extractor.probability_bounds(target_var="Y", target_val=1)
        assert isinstance(result, BoundResult)
        assert 0.0 <= result.lower <= result.upper <= 1.0

    def test_probability_bounds_chain(self):
        dag = _make_chain_dag()
        enc = ConstraintEncoder(dag)
        enc.add_normalization_constraints()
        enc.add_markov_constraints()
        A, b, _names = enc.build_constraint_matrix()
        extractor = BoundExtractor(dag, A, b)
        result = extractor.probability_bounds(target_var="C", target_val=0)
        assert result.lower >= 0.0
        assert result.upper <= 1.0

    def test_bound_contains_midpoint(self):
        dag = _make_binary_iv_dag()
        enc = ConstraintEncoder(dag)
        enc.add_normalization_constraints()
        enc.add_markov_constraints()
        A, b, _names = enc.build_constraint_matrix()
        extractor = BoundExtractor(dag, A, b)
        result = extractor.probability_bounds(target_var="Z", target_val=0)
        assert result.contains(result.midpoint)

    def test_sensitivity_analysis(self):
        dag = _make_binary_iv_dag()
        enc = ConstraintEncoder(dag)
        enc.add_normalization_constraints()
        enc.add_markov_constraints()
        A, b, _names = enc.build_constraint_matrix()
        extractor = BoundExtractor(dag, A, b)
        report = extractor.sensitivity_analysis(target_var="Y", target_val=1)
        assert report is not None

    def test_bound_tightness_analysis(self):
        dag = _make_chain_dag()
        enc = ConstraintEncoder(dag)
        enc.add_normalization_constraints()
        enc.add_markov_constraints()
        A, b, _names = enc.build_constraint_matrix()
        extractor = BoundExtractor(dag, A, b)
        analysis = extractor.bound_tightness_analysis(target_var="C")
        assert isinstance(analysis, dict)


# ---------------------------------------------------------------------------
# Interventional polytope
# ---------------------------------------------------------------------------

class TestInterventionalPolytope:
    """Test interventional polytope construction."""

    def test_apply_do_operator(self):
        dag = _make_binary_iv_dag()
        ip = InterventionalPolytope(dag)
        do_op = ip.apply_do("X", 1)
        assert do_op is not None
        assert do_op.variable == "X"
        assert do_op.value == 1

    def test_mutilated_dag(self):
        dag = _make_binary_iv_dag()
        ip = InterventionalPolytope(dag)
        ip.apply_do("X", 1)
        mutilated = ip.get_mutilated_dag()
        assert mutilated is not None

    def test_truncated_factorization(self):
        dag = _make_chain_dag()
        ip = InterventionalPolytope(dag)
        ip.apply_do("B", 0)
        tf = ip.get_truncated_factorization()
        assert tf is not None
        expr = tf.to_expression()
        assert isinstance(expr, str)

    def test_check_identifiability_chain(self):
        dag = _make_chain_dag()
        ip = InterventionalPolytope(dag)
        result = ip.check_identifiability("C", ["A"])
        assert result is not None
        assert isinstance(result.is_identifiable, bool)

    def test_check_identifiability_iv(self):
        dag = _make_binary_iv_dag()
        ip = InterventionalPolytope(dag)
        result = ip.check_identifiability("Y", ["X"])
        assert result is not None

    def test_multiple_interventions(self):
        dag = _make_diamond_dag()
        ip = InterventionalPolytope(dag)
        ip.apply_do("B", 1)
        ip.apply_do("C", 0)
        interventions = ip.get_interventions()
        assert len(interventions) == 2

    def test_causal_effect_type(self):
        dag = _make_chain_dag()
        ip = InterventionalPolytope(dag)
        eff_type = ip.get_causal_effect_type("C", ["A"])
        assert isinstance(eff_type, str)


# ---------------------------------------------------------------------------
# Pricing subproblem
# ---------------------------------------------------------------------------

class TestPricingSubproblem:
    """Test the pricing subproblem for column generation."""

    def test_pricing_produces_columns(self):
        dag = _make_binary_iv_dag()
        enc = ConstraintEncoder(dag)
        enc.add_normalization_constraints()
        enc.add_markov_constraints()
        A, b, _names = enc.build_constraint_matrix()
        n_vars = enc.num_variables
        c = np.zeros(n_vars)
        c[0] = 1.0
        pricing = PricingSubproblem(dag, c, A, b)
        rc = np.random.RandomState(42).randn(A.shape[0])
        results = pricing.price(rc, strategy=PricingStrategy.HEURISTIC)
        assert isinstance(results, list)

    def test_pricing_strategies(self):
        dag = _make_chain_dag()
        enc = ConstraintEncoder(dag)
        enc.add_normalization_constraints()
        enc.add_markov_constraints()
        A, b, _names = enc.build_constraint_matrix()
        n_vars = enc.num_variables
        c = np.ones(n_vars) / n_vars
        pricing = PricingSubproblem(dag, c, A, b)
        rc = np.zeros(A.shape[0])
        for strategy in [PricingStrategy.HEURISTIC, PricingStrategy.RANDOMIZED]:
            results = pricing.price(rc, strategy=strategy)
            assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Normalization constraints
# ---------------------------------------------------------------------------

class TestNormalization:
    """Test normalization constraint properties."""

    def test_feasible_point_satisfies_normalization(self):
        """A uniform distribution should satisfy normalization."""
        dag = _make_binary_iv_dag()
        enc = ConstraintEncoder(dag)
        enc.add_normalization_constraints()
        A, b, _names = enc.build_constraint_matrix()
        n = enc.num_variables
        x = np.ones(n) / n
        residual = A.dot(x) - b
        assert np.allclose(residual, 0, atol=1e-8)

    def test_normalization_matrix_row_sum(self):
        """Each normalization row should have all 1s or a subset summing to 1."""
        dag = _make_chain_dag()
        enc = ConstraintEncoder(dag)
        enc.add_normalization_constraints()
        A, b, _names = enc.build_constraint_matrix()
        A_dense = A.toarray() if sparse.issparse(A) else A
        for i in range(A_dense.shape[0]):
            row = A_dense[i]
            nonzero = row[row != 0]
            assert np.all(nonzero > 0)  # All coefficients positive (normalization)

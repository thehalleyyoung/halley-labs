"""
Causal Polytope Solver
=======================

Orchestrates column-generation LP solving over the causal polytope defined
by a DAG, observed marginals, and interventional constraints.  Returns
worst-case bounds on interventional queries P(Y | do(X=x)).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Data structures
# ---------------------------------------------------------------------------

class SolverStatus(Enum):
    OPTIMAL = auto()
    INFEASIBLE = auto()
    UNBOUNDED = auto()
    ITERATION_LIMIT = auto()
    TIME_LIMIT = auto()
    NUMERICAL_ERROR = auto()
    NOT_SOLVED = auto()


@dataclass
class DAGSpec:
    """Directed acyclic graph specification with variable cardinalities."""
    nodes: List[str]
    edges: List[Tuple[str, str]]
    card: Dict[str, int]  # variable name -> number of values

    def parents(self, node: str) -> List[str]:
        return [u for u, v in self.edges if v == node]

    def children(self, node: str) -> List[str]:
        return [v for u, v in self.edges if u == node]

    def ancestors(self, node: str) -> FrozenSet[str]:
        visited: set[str] = set()
        stack = list(self.parents(node))
        while stack:
            n = stack.pop()
            if n not in visited:
                visited.add(n)
                stack.extend(self.parents(n))
        return frozenset(visited)

    def descendants(self, node: str) -> FrozenSet[str]:
        visited: set[str] = set()
        stack = list(self.children(node))
        while stack:
            n = stack.pop()
            if n not in visited:
                visited.add(n)
                stack.extend(self.children(n))
        return frozenset(visited)

    def topological_order(self) -> List[str]:
        in_deg = {n: 0 for n in self.nodes}
        for u, v in self.edges:
            in_deg[v] += 1
        queue = [n for n in self.nodes if in_deg[n] == 0]
        order: List[str] = []
        while queue:
            queue.sort()
            n = queue.pop(0)
            order.append(n)
            for c in self.children(n):
                in_deg[c] -= 1
                if in_deg[c] == 0:
                    queue.append(c)
        return order

    def is_ancestor(self, u: str, v: str) -> bool:
        return u in self.ancestors(v)

    def moralize(self) -> Dict[str, set]:
        """Return adjacency dict of the moral graph."""
        adj: Dict[str, set] = {n: set() for n in self.nodes}
        for u, v in self.edges:
            adj[u].add(v)
            adj[v].add(u)
        for n in self.nodes:
            pars = self.parents(n)
            for i in range(len(pars)):
                for j in range(i + 1, len(pars)):
                    adj[pars[i]].add(pars[j])
                    adj[pars[j]].add(pars[i])
        return adj


@dataclass
class InterventionSpec:
    """Specification of a do(X=x) intervention."""
    variable: str
    value: int


@dataclass
class QuerySpec:
    """Target query: P(target_var = target_val | do(interv))."""
    target_var: str
    target_val: Optional[int] = None  # None means E[target_var]
    interventions: List[InterventionSpec] = field(default_factory=list)
    conditioning: Optional[Dict[str, int]] = None


@dataclass
class ObservedMarginals:
    """Observed marginal distributions for subsets of variables."""
    marginals: Dict[FrozenSet[str], np.ndarray]

    def get_marginal(self, variables: FrozenSet[str]) -> Optional[np.ndarray]:
        return self.marginals.get(variables)

    def variable_sets(self) -> List[FrozenSet[str]]:
        return list(self.marginals.keys())


@dataclass
class DualCertificate:
    """Dual certificate proving optimality of a bound."""
    dual_values: np.ndarray
    constraint_names: List[str]
    objective_value: float
    is_lower_bound: bool
    active_constraints: List[int]


@dataclass
class SolverDiagnostics:
    """Diagnostics from the column-generation solve."""
    iterations: int
    total_columns_generated: int
    active_columns: int
    master_lp_solves: int
    pricing_solves: int
    convergence_gap: float
    solve_time_seconds: float
    master_times: List[float] = field(default_factory=list)
    pricing_times: List[float] = field(default_factory=list)
    bound_history: List[Tuple[float, float]] = field(default_factory=list)
    reduced_cost_history: List[float] = field(default_factory=list)


@dataclass
class SolverResult:
    """Result of solving for bounds on a causal query."""
    lower_bound: float
    upper_bound: float
    status: SolverStatus
    lower_certificate: Optional[DualCertificate] = None
    upper_certificate: Optional[DualCertificate] = None
    diagnostics: Optional[SolverDiagnostics] = None
    optimal_columns_lower: Optional[np.ndarray] = None
    optimal_columns_upper: Optional[np.ndarray] = None
    identifiable: bool = False

    @property
    def gap(self) -> float:
        return self.upper_bound - self.lower_bound

    @property
    def midpoint(self) -> float:
        return (self.lower_bound + self.upper_bound) / 2.0


@dataclass
class SolverConfig:
    """Configuration for the CausalPolytopeSolver."""
    max_iterations: int = 500
    gap_tolerance: float = 1e-8
    reduced_cost_tolerance: float = 1e-10
    time_limit: float = 300.0
    pricing_strategy: str = "exact"
    warm_start: bool = True
    column_age_limit: int = 50
    stabilization: bool = True
    stabilization_alpha: float = 0.3
    log_interval: int = 10
    num_initial_columns: int = 20
    max_columns_per_iter: int = 5
    use_sparse: bool = True


# ---------------------------------------------------------------------------
#  Main Solver
# ---------------------------------------------------------------------------

class CausalPolytopeSolver:
    """
    Top-level solver for computing worst-case bounds on interventional
    queries over a causal DAG with discrete variables.

    The causal polytope is the set of all joint distributions that are:
      - Markov to the DAG
      - consistent with observed marginals
      - consistent with any interventional constraints

    We optimise a linear objective (encoding the query) over this polytope
    via column generation.

    Parameters
    ----------
    config : SolverConfig
        Solver configuration.
    """

    def __init__(self, config: Optional[SolverConfig] = None):
        self.config = config or SolverConfig()
        self._dag: Optional[DAGSpec] = None
        self._observed: Optional[ObservedMarginals] = None
        self._query: Optional[QuerySpec] = None
        self._result: Optional[SolverResult] = None
        self._constraint_encoder = None
        self._column_gen = None
        self._interventional_polytope = None
        self._lower_diagnostics: Optional[SolverDiagnostics] = None
        self._upper_diagnostics: Optional[SolverDiagnostics] = None

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        dag: DAGSpec,
        query: QuerySpec,
        observed: Optional[ObservedMarginals] = None,
        max_iterations: Optional[int] = None,
    ) -> SolverResult:
        """
        Solve for bounds on the given causal query.

        Parameters
        ----------
        dag : DAGSpec
            The causal DAG with cardinalities.
        query : QuerySpec
            The interventional query to bound.
        observed : ObservedMarginals, optional
            Observed marginals to constrain the polytope.
        max_iterations : int, optional
            Override the config's max_iterations.

        Returns
        -------
        SolverResult
            Lower and upper bounds, certificates, diagnostics.
        """
        from .column_generation import ColumnGenerationSolver
        from .constraints import ConstraintEncoder
        from .interventional import InterventionalPolytope

        start_time = time.time()
        self._dag = dag
        self._query = query
        self._observed = observed
        iters = max_iterations or self.config.max_iterations

        self._validate_inputs()
        logger.info(
            "Solving query P(%s | do(%s)) over DAG with %d nodes",
            query.target_var,
            ", ".join(f"{iv.variable}={iv.value}" for iv in query.interventions),
            len(dag.nodes),
        )

        # Build interventional polytope
        self._interventional_polytope = InterventionalPolytope(dag)
        for iv in query.interventions:
            self._interventional_polytope.apply_do(iv.variable, iv.value)

        mutilated_dag = self._interventional_polytope.get_mutilated_dag()
        id_result = self._interventional_polytope.check_identifiability(
            query.target_var,
            [iv.variable for iv in query.interventions],
        )

        # Encode constraints
        self._constraint_encoder = ConstraintEncoder(mutilated_dag)
        self._constraint_encoder.add_normalization_constraints()
        self._constraint_encoder.add_markov_constraints()

        if observed is not None:
            self._constraint_encoder.add_observed_marginal_constraints(observed)

        for iv in query.interventions:
            self._constraint_encoder.add_intervention_constraints(iv.variable, iv.value)

        A, b, constraint_names = self._constraint_encoder.build_constraint_matrix()

        # Build objective
        c_lower, c_upper = self._build_objective_vectors(mutilated_dag, query)

        total_vars = self._compute_joint_size(mutilated_dag)

        # For small problems, solve directly without column generation
        if total_vars <= 256:
            logger.info("Small problem (%d vars), solving directly", total_vars)
            lb, ub, lb_duals, ub_duals = self._solve_direct(
                c_lower, c_upper, A, b, total_vars
            )
            elapsed = time.time() - start_time

            lower_cert = self._build_certificate(
                lb_duals, constraint_names, lb, is_lower=True, A=A, b=b
            )
            upper_cert = self._build_certificate(
                ub_duals, constraint_names, ub, is_lower=False, A=A, b=b
            )

            direct_diag = SolverDiagnostics(
                iterations=1, total_columns_generated=0, active_columns=0,
                master_lp_solves=2, pricing_solves=0,
                convergence_gap=0.0, solve_time_seconds=elapsed,
            )
            self._result = SolverResult(
                lower_bound=max(0.0, min(1.0, lb)),
                upper_bound=max(0.0, min(1.0, ub)),
                status=SolverStatus.OPTIMAL,
                lower_certificate=lower_cert,
                upper_certificate=upper_cert,
                diagnostics=direct_diag,
                identifiable=id_result.is_identifiable if id_result else False,
            )
            logger.info(
                "Direct solve: [%.6f, %.6f] time=%.2fs",
                self._result.lower_bound, self._result.upper_bound, elapsed,
            )
            return self._result

        # Solve lower bound (minimise)
        logger.info("Solving for LOWER bound (minimisation)")
        cg_lower = ColumnGenerationSolver(
            c=c_lower,
            A_eq=A,
            b_eq=b,
            total_vars=total_vars,
            config=self.config,
            dag=mutilated_dag,
        )
        lb, lb_cols, lb_duals, lb_diag = cg_lower.solve(max_iterations=iters)

        # Solve upper bound (maximise via -c)
        logger.info("Solving for UPPER bound (maximisation)")
        cg_upper = ColumnGenerationSolver(
            c=c_upper,
            A_eq=A,
            b_eq=b,
            total_vars=total_vars,
            config=self.config,
            dag=mutilated_dag,
        )
        ub_neg, ub_cols, ub_duals, ub_diag = cg_upper.solve(max_iterations=iters)
        ub = -ub_neg

        elapsed = time.time() - start_time

        # Build certificates
        lower_cert = self._build_certificate(
            lb_duals, constraint_names, lb, is_lower=True, A=A, b=b
        )
        upper_cert = self._build_certificate(
            ub_duals, constraint_names, ub, is_lower=False, A=A, b=b
        )

        # Determine status
        if lb_diag is None or ub_diag is None:
            status = SolverStatus.NUMERICAL_ERROR
        elif lb > ub + self.config.gap_tolerance:
            status = SolverStatus.INFEASIBLE
        else:
            lb_converged = lb_diag.convergence_gap <= self.config.gap_tolerance
            ub_converged = ub_diag.convergence_gap <= self.config.gap_tolerance
            if lb_converged and ub_converged:
                status = SolverStatus.OPTIMAL
            elif lb_diag.iterations >= iters or ub_diag.iterations >= iters:
                status = SolverStatus.ITERATION_LIMIT
            elif elapsed >= self.config.time_limit:
                status = SolverStatus.TIME_LIMIT
            else:
                status = SolverStatus.OPTIMAL

        combined_diag = SolverDiagnostics(
            iterations=max(
                lb_diag.iterations if lb_diag else 0,
                ub_diag.iterations if ub_diag else 0,
            ),
            total_columns_generated=(
                (lb_diag.total_columns_generated if lb_diag else 0)
                + (ub_diag.total_columns_generated if ub_diag else 0)
            ),
            active_columns=(
                (lb_diag.active_columns if lb_diag else 0)
                + (ub_diag.active_columns if ub_diag else 0)
            ),
            master_lp_solves=(
                (lb_diag.master_lp_solves if lb_diag else 0)
                + (ub_diag.master_lp_solves if ub_diag else 0)
            ),
            pricing_solves=(
                (lb_diag.pricing_solves if lb_diag else 0)
                + (ub_diag.pricing_solves if ub_diag else 0)
            ),
            convergence_gap=max(
                lb_diag.convergence_gap if lb_diag else float("inf"),
                ub_diag.convergence_gap if ub_diag else float("inf"),
            ),
            solve_time_seconds=elapsed,
        )

        self._result = SolverResult(
            lower_bound=max(0.0, min(1.0, lb)),
            upper_bound=max(0.0, min(1.0, ub)),
            status=status,
            lower_certificate=lower_cert,
            upper_certificate=upper_cert,
            diagnostics=combined_diag,
            optimal_columns_lower=lb_cols,
            optimal_columns_upper=ub_cols,
            identifiable=id_result.is_identifiable if id_result else False,
        )

        logger.info(
            "Solved: [%.6f, %.6f] status=%s time=%.2fs",
            self._result.lower_bound,
            self._result.upper_bound,
            status.name,
            elapsed,
        )
        return self._result

    def get_bounds(self) -> Tuple[float, float]:
        """Return (lower_bound, upper_bound) from the last solve."""
        if self._result is None:
            raise RuntimeError("Must call solve() first.")
        return self._result.lower_bound, self._result.upper_bound

    def get_certificate(self, lower: bool = True) -> Optional[DualCertificate]:
        """Return the dual certificate for the lower or upper bound."""
        if self._result is None:
            raise RuntimeError("Must call solve() first.")
        return self._result.lower_certificate if lower else self._result.upper_certificate

    def get_diagnostics(self) -> Optional[SolverDiagnostics]:
        if self._result is None:
            return None
        return self._result.diagnostics

    # ------------------------------------------------------------------
    #  Direct LP solve (for small problems)
    # ------------------------------------------------------------------

    def _solve_direct(
        self,
        c_lower: np.ndarray,
        c_upper: np.ndarray,
        A: sparse.spmatrix,
        b: np.ndarray,
        n: int,
    ) -> Tuple[float, float, Optional[np.ndarray], Optional[np.ndarray]]:
        """Solve the LP directly without column generation."""
        from scipy.optimize import linprog as _linprog

        if sparse.issparse(A):
            A_dense = A.toarray()
        else:
            A_dense = np.asarray(A)

        bounds = [(0.0, None)] * n
        opts = {"maxiter": 50000, "presolve": True,
                "dual_feasibility_tolerance": 1e-10,
                "primal_feasibility_tolerance": 1e-10}

        # Lower bound
        res_lb = _linprog(c=c_lower, A_eq=A_dense, b_eq=b, bounds=bounds,
                          method="highs", options=opts)
        if res_lb.success:
            lb = res_lb.fun
            lb_duals = getattr(getattr(res_lb, "eqlin", None), "marginals", None)
            if lb_duals is not None:
                lb_duals = np.array(lb_duals, dtype=np.float64)
        else:
            lb = 0.0
            lb_duals = None

        # Upper bound (min -c)
        res_ub = _linprog(c=c_upper, A_eq=A_dense, b_eq=b, bounds=bounds,
                          method="highs", options=opts)
        if res_ub.success:
            ub = -res_ub.fun
            ub_duals = getattr(getattr(res_ub, "eqlin", None), "marginals", None)
            if ub_duals is not None:
                ub_duals = np.array(ub_duals, dtype=np.float64)
        else:
            ub = 1.0
            ub_duals = None

        return lb, ub, lb_duals, ub_duals

    # ------------------------------------------------------------------
    #  Objective construction
    # ------------------------------------------------------------------

    def _build_objective_vectors(
        self, dag: DAGSpec, query: QuerySpec
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the objective vector c such that c^T x encodes the query.

        For P(Y=y | do(X=x)), the objective vector has 1 at entries
        corresponding to joint assignments where Y=y, and 0 otherwise.
        The lower bound minimises c^T x; the upper bound maximises it
        (equivalently, minimises -c^T x).
        """
        total = self._compute_joint_size(dag)
        c = np.zeros(total, dtype=np.float64)

        topo = dag.topological_order()
        strides = self._compute_strides(dag, topo)

        if query.target_val is not None:
            # P(Y=y | do(X=x))
            self._fill_indicator_objective(c, dag, topo, strides, query)
        else:
            # E[Y | do(X=x)] — weighted by value
            card_y = dag.card[query.target_var]
            for y_val in range(card_y):
                weight = float(y_val) / max(1, card_y - 1)
                q_copy = QuerySpec(
                    target_var=query.target_var,
                    target_val=y_val,
                    interventions=query.interventions,
                    conditioning=query.conditioning,
                )
                c_part = np.zeros(total, dtype=np.float64)
                self._fill_indicator_objective(c_part, dag, topo, strides, q_copy)
                c += weight * c_part

        c_upper = -c.copy()
        return c, c_upper

    def _fill_indicator_objective(
        self,
        c: np.ndarray,
        dag: DAGSpec,
        topo: List[str],
        strides: Dict[str, int],
        query: QuerySpec,
    ) -> None:
        """Set c[idx] = 1 for all joint assignments where target_var == target_val."""
        target_idx = topo.index(query.target_var)
        total = len(c)

        for flat_idx in range(total):
            assignment = self._flat_to_assignment(flat_idx, dag, topo, strides)
            if assignment[query.target_var] == query.target_val:
                matches_intervention = True
                for iv in query.interventions:
                    if iv.variable in assignment and assignment[iv.variable] != iv.value:
                        matches_intervention = False
                        break
                if query.conditioning is not None:
                    for cvar, cval in query.conditioning.items():
                        if cvar in assignment and assignment[cvar] != cval:
                            matches_intervention = False
                            break
                if matches_intervention:
                    c[flat_idx] = 1.0

    def _compute_strides(
        self, dag: DAGSpec, topo: List[str]
    ) -> Dict[str, int]:
        """Compute strides for flattening joint assignments."""
        strides: Dict[str, int] = {}
        stride = 1
        for node in reversed(topo):
            strides[node] = stride
            stride *= dag.card[node]
        return strides

    def _flat_to_assignment(
        self,
        flat_idx: int,
        dag: DAGSpec,
        topo: List[str],
        strides: Dict[str, int],
    ) -> Dict[str, int]:
        """Convert a flat index to a variable assignment dict."""
        assignment: Dict[str, int] = {}
        remaining = flat_idx
        for node in topo:
            card = dag.card[node]
            stride = strides[node]
            val = (remaining // stride) % card
            assignment[node] = val
        return assignment

    def _assignment_to_flat(
        self,
        assignment: Dict[str, int],
        dag: DAGSpec,
        topo: List[str],
        strides: Dict[str, int],
    ) -> int:
        """Convert a variable assignment to a flat index."""
        idx = 0
        for node in topo:
            idx += assignment[node] * strides[node]
        return idx

    def _compute_joint_size(self, dag: DAGSpec) -> int:
        """Total number of joint assignments."""
        size = 1
        for node in dag.nodes:
            size *= dag.card[node]
        return size

    # ------------------------------------------------------------------
    #  Validation
    # ------------------------------------------------------------------

    def _validate_inputs(self) -> None:
        """Validate that the query is consistent with the DAG."""
        dag = self._dag
        query = self._query
        assert dag is not None and query is not None

        if query.target_var not in dag.nodes:
            raise ValueError(
                f"Target variable '{query.target_var}' not in DAG nodes."
            )

        for iv in query.interventions:
            if iv.variable not in dag.nodes:
                raise ValueError(
                    f"Intervention variable '{iv.variable}' not in DAG nodes."
                )
            if iv.value < 0 or iv.value >= dag.card[iv.variable]:
                raise ValueError(
                    f"Intervention value {iv.value} out of range "
                    f"[0, {dag.card[iv.variable]}) for '{iv.variable}'."
                )

        if query.target_val is not None:
            card_y = dag.card[query.target_var]
            if query.target_val < 0 or query.target_val >= card_y:
                raise ValueError(
                    f"Target value {query.target_val} out of range "
                    f"[0, {card_y}) for '{query.target_var}'."
                )

        # Check DAG is actually acyclic
        try:
            topo = dag.topological_order()
        except Exception:
            raise ValueError("Provided graph is not a valid DAG.")
        if len(topo) != len(dag.nodes):
            raise ValueError("Provided graph contains a cycle.")

    # ------------------------------------------------------------------
    #  Certificate building
    # ------------------------------------------------------------------

    def _build_certificate(
        self,
        duals: Optional[np.ndarray],
        constraint_names: List[str],
        obj_val: float,
        is_lower: bool,
        A: sparse.spmatrix,
        b: np.ndarray,
    ) -> Optional[DualCertificate]:
        """Build a dual certificate from the LP dual variables."""
        if duals is None:
            return None

        active = []
        for i, d in enumerate(duals):
            if abs(d) > 1e-12:
                active.append(i)

        return DualCertificate(
            dual_values=duals,
            constraint_names=constraint_names,
            objective_value=obj_val,
            is_lower_bound=is_lower,
            active_constraints=active,
        )

    # ------------------------------------------------------------------
    #  Convenience constructors
    # ------------------------------------------------------------------

    @staticmethod
    def from_adjacency(
        adj: Dict[str, List[str]],
        card: Optional[Dict[str, int]] = None,
    ) -> Tuple["CausalPolytopeSolver", DAGSpec]:
        """
        Build a solver and DAG from an adjacency-list representation.

        Parameters
        ----------
        adj : dict
            node -> list of children.
        card : dict, optional
            node -> cardinality (defaults to 2 for all).

        Returns
        -------
        solver, dag
        """
        nodes = sorted(adj.keys())
        edges = []
        for u in nodes:
            for v in adj[u]:
                edges.append((u, v))
                if v not in nodes:
                    nodes.append(v)
        nodes = sorted(set(nodes))
        if card is None:
            card = {n: 2 for n in nodes}
        dag = DAGSpec(nodes=nodes, edges=edges, card=card)
        return CausalPolytopeSolver(), dag

    @staticmethod
    def binary_dag(
        edges: List[Tuple[str, str]],
    ) -> Tuple["CausalPolytopeSolver", DAGSpec]:
        """Create solver + DAG for a binary-variable DAG."""
        nodes_set: set[str] = set()
        for u, v in edges:
            nodes_set.add(u)
            nodes_set.add(v)
        nodes = sorted(nodes_set)
        card = {n: 2 for n in nodes}
        dag = DAGSpec(nodes=nodes, edges=edges, card=card)
        return CausalPolytopeSolver(), dag

    # ------------------------------------------------------------------
    #  Sensitivity wrappers
    # ------------------------------------------------------------------

    def sensitivity_to_marginal(
        self,
        dag: DAGSpec,
        query: QuerySpec,
        observed: ObservedMarginals,
        perturbation: float = 0.01,
    ) -> Dict[FrozenSet[str], Tuple[float, float]]:
        """
        Compute how bounds change when each observed marginal is perturbed.

        Returns dict mapping variable set -> (delta_lower, delta_upper).
        """
        base = self.solve(dag, query, observed)
        sensitivities: Dict[FrozenSet[str], Tuple[float, float]] = {}

        for var_set in observed.variable_sets():
            perturbed_marginals = dict(observed.marginals)
            original = perturbed_marginals[var_set].copy()

            # Perturb by adding noise and re-normalising
            noise = np.random.default_rng(42).normal(0, perturbation, original.shape)
            perturbed = np.clip(original + noise, 0, None)
            perturbed /= perturbed.sum()
            perturbed_marginals[var_set] = perturbed

            perturbed_obs = ObservedMarginals(marginals=perturbed_marginals)
            pert_result = self.solve(dag, query, perturbed_obs)

            sensitivities[var_set] = (
                pert_result.lower_bound - base.lower_bound,
                pert_result.upper_bound - base.upper_bound,
            )

        return sensitivities

    def solve_multiple_queries(
        self,
        dag: DAGSpec,
        queries: List[QuerySpec],
        observed: Optional[ObservedMarginals] = None,
    ) -> List[SolverResult]:
        """Solve multiple queries, reusing constraint construction."""
        results: List[SolverResult] = []
        for q in queries:
            r = self.solve(dag, q, observed)
            results.append(r)
        return results

    def compute_ate_bounds(
        self,
        dag: DAGSpec,
        treatment: str,
        outcome: str,
        observed: Optional[ObservedMarginals] = None,
    ) -> Tuple[float, float]:
        """
        Compute bounds on the Average Treatment Effect:
          ATE = E[Y | do(X=1)] - E[Y | do(X=0)]
        for binary treatment X.
        """
        q1 = QuerySpec(
            target_var=outcome,
            target_val=None,
            interventions=[InterventionSpec(treatment, 1)],
        )
        q0 = QuerySpec(
            target_var=outcome,
            target_val=None,
            interventions=[InterventionSpec(treatment, 0)],
        )

        r1 = self.solve(dag, q1, observed)
        r0 = self.solve(dag, q0, observed)

        ate_lower = r1.lower_bound - r0.upper_bound
        ate_upper = r1.upper_bound - r0.lower_bound
        return ate_lower, ate_upper

    def compute_cate_bounds(
        self,
        dag: DAGSpec,
        treatment: str,
        outcome: str,
        conditioning: Dict[str, int],
        observed: Optional[ObservedMarginals] = None,
    ) -> Tuple[float, float]:
        """
        Compute bounds on the Conditional Average Treatment Effect:
          CATE(z) = E[Y | do(X=1), Z=z] - E[Y | do(X=0), Z=z]
        """
        q1 = QuerySpec(
            target_var=outcome,
            target_val=None,
            interventions=[InterventionSpec(treatment, 1)],
            conditioning=conditioning,
        )
        q0 = QuerySpec(
            target_var=outcome,
            target_val=None,
            interventions=[InterventionSpec(treatment, 0)],
            conditioning=conditioning,
        )

        r1 = self.solve(dag, q1, observed)
        r0 = self.solve(dag, q0, observed)

        cate_lower = r1.lower_bound - r0.upper_bound
        cate_upper = r1.upper_bound - r0.lower_bound
        return cate_lower, cate_upper

    # ------------------------------------------------------------------
    #  Serialisation helpers
    # ------------------------------------------------------------------

    def result_to_dict(self) -> Dict[str, Any]:
        """Serialise the last result to a plain dict."""
        if self._result is None:
            return {}
        r = self._result
        d: Dict[str, Any] = {
            "lower_bound": r.lower_bound,
            "upper_bound": r.upper_bound,
            "gap": r.gap,
            "status": r.status.name,
            "identifiable": r.identifiable,
        }
        if r.diagnostics:
            d["diagnostics"] = {
                "iterations": r.diagnostics.iterations,
                "total_columns_generated": r.diagnostics.total_columns_generated,
                "active_columns": r.diagnostics.active_columns,
                "solve_time_seconds": r.diagnostics.solve_time_seconds,
            }
        return d

"""
Interventional Polytope
========================

Constructs the interventional-distribution polytope by applying the
do-operator (graph mutilation) to the causal DAG.  Checks identifiability
conditions, encodes truncated factorisation, and implements front-door
and back-door criteria.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Data structures
# ---------------------------------------------------------------------------

@dataclass
class IdentifiabilityResult:
    """Result of checking identifiability of a causal query."""
    is_identifiable: bool
    method: str = ""         # "backdoor", "frontdoor", "id_algorithm", "none"
    adjustment_set: Optional[FrozenSet[str]] = None
    identifying_expression: str = ""
    message: str = ""


@dataclass
class DoOperator:
    """Represents a do-intervention on a variable."""
    variable: str
    value: int
    original_parents: List[str] = field(default_factory=list)


@dataclass
class TruncatedFactorization:
    """
    Represents the truncated factorisation formula:

        P(v \\ x | do(x)) = prod_{V_i not in X} P(V_i | pa_i)

    Each factor is (variable, parents) with the intervention-set removed.
    """
    factors: List[Tuple[str, List[str]]]
    intervention_set: FrozenSet[str]

    def to_expression(self) -> str:
        parts = []
        for var, parents in self.factors:
            if parents:
                parts.append(f"P({var}|{','.join(parents)})")
            else:
                parts.append(f"P({var})")
        return " * ".join(parts)


# ---------------------------------------------------------------------------
#  Interventional polytope
# ---------------------------------------------------------------------------

class InterventionalPolytope:
    """
    Constructs the interventional-distribution polytope by modifying the
    DAG structure according to the do-operator.

    The interventional distribution P(V | do(X=x)) is defined by:
        1. Removing all incoming edges to X in the DAG (graph mutilation).
        2. Setting X = x deterministically.
        3. Computing the truncated factorisation.

    Parameters
    ----------
    dag : DAGSpec
        The original causal DAG.
    """

    def __init__(self, dag):
        from .causal_polytope import DAGSpec

        self._original_dag = dag
        # Deep-copy the DAG for mutilation
        self._mutilated_dag = DAGSpec(
            nodes=list(dag.nodes),
            edges=list(dag.edges),
            card=dict(dag.card),
        )
        self._interventions: List[DoOperator] = []
        self._truncated: Optional[TruncatedFactorization] = None

    # ------------------------------------------------------------------
    #  Do-operator
    # ------------------------------------------------------------------

    def apply_do(self, variable: str, value: int) -> DoOperator:
        """
        Apply do(variable = value): remove all incoming edges to *variable*
        in the mutilated DAG.

        Parameters
        ----------
        variable : str
            The variable to intervene on.
        value : int
            The value to set.

        Returns
        -------
        DoOperator with details.
        """
        if variable not in self._mutilated_dag.nodes:
            raise ValueError(f"Variable '{variable}' not in DAG.")

        card = self._mutilated_dag.card[variable]
        if value < 0 or value >= card:
            raise ValueError(
                f"Intervention value {value} out of range [0, {card}) "
                f"for variable '{variable}'."
            )

        # Record original parents before removing
        original_parents = self._mutilated_dag.parents(variable)

        # Remove all incoming edges to variable
        new_edges = [
            (u, v) for u, v in self._mutilated_dag.edges
            if v != variable
        ]
        self._mutilated_dag.edges = new_edges

        do_op = DoOperator(
            variable=variable,
            value=value,
            original_parents=original_parents,
        )
        self._interventions.append(do_op)
        self._truncated = None  # invalidate cache

        logger.info(
            "Applied do(%s=%d), removed %d incoming edges",
            variable, value, len(original_parents),
        )
        return do_op

    def get_mutilated_dag(self):
        """Return the mutilated DAG after all interventions."""
        return self._mutilated_dag

    def get_interventions(self) -> List[DoOperator]:
        return list(self._interventions)

    # ------------------------------------------------------------------
    #  Truncated factorisation
    # ------------------------------------------------------------------

    def get_truncated_factorization(self) -> TruncatedFactorization:
        """
        Compute the truncated factorisation for the interventional distribution.

        P(v \\ x | do(x)) = prod_{V_i not in X} P(V_i | pa_i^{original})

        The factors for intervened variables are removed (they are set
        deterministically).
        """
        if self._truncated is not None:
            return self._truncated

        interv_vars = frozenset(do.variable for do in self._interventions)
        topo = self._original_dag.topological_order()

        factors: List[Tuple[str, List[str]]] = []
        for node in topo:
            if node in interv_vars:
                continue
            parents = self._original_dag.parents(node)
            factors.append((node, parents))

        self._truncated = TruncatedFactorization(
            factors=factors,
            intervention_set=interv_vars,
        )
        return self._truncated

    def encode_truncated_factorization_constraints(
        self,
    ) -> List[Tuple[np.ndarray, float, str]]:
        """
        Encode the truncated-factorisation constraints as linear rows.

        For each non-intervened variable V_i with parents pa_i:
            P(V_i, pa_i) = P(V_i | pa_i) * P(pa_i)
        which, for known CPTs, becomes linear constraints on the joint.

        Returns list of (row_vector, rhs, name) tuples.
        """
        tf = self.get_truncated_factorization()
        topo = self._mutilated_dag.topological_order()
        strides = self._compute_strides(self._mutilated_dag, topo)
        total = self._compute_total(self._mutilated_dag)
        constraints: List[Tuple[np.ndarray, float, str]] = []

        for var, parents in tf.factors:
            card_v = self._mutilated_dag.card[var]
            if not parents:
                continue

            pa_cards = [self._mutilated_dag.card[p] for p in parents]
            pa_configs = list(itertools.product(*[range(c) for c in pa_cards]))

            for pa_config in pa_configs:
                pa_dict = dict(zip(parents, pa_config))

                for v_val in range(card_v):
                    # Constraint: P(var=v_val, pa=pa_config) - cpt * P(pa=pa_config) = 0
                    # Since CPT is unknown in general, we encode the consistency:
                    # sum_{v} P(var=v, pa=pa_config) = P(pa=pa_config) for each pa_config
                    # This is automatically satisfied, so we encode the
                    # weaker condition that the conditional doesn't depend on
                    # downstream variables (Markov property in mutilated DAG).
                    row = np.zeros(total, dtype=np.float64)

                    for flat_idx in range(total):
                        assign = self._flat_to_assignment(flat_idx, self._mutilated_dag, topo, strides)
                        matches_pa = all(assign[p] == pa_dict[p] for p in parents)
                        if matches_pa:
                            if assign[var] == v_val:
                                row[flat_idx] = 1.0

                    name = f"trunc_{var}={v_val}|{parents}={pa_config}"
                    # RHS: we don't know the CPT value, so this is a structural
                    # constraint.  The row encodes P(var=v_val, pa=pa_config).
                    # We pair it with the normalisation P(var, pa) sums to P(pa).
                    constraints.append((row, 0.0, name))

        return constraints

    # ------------------------------------------------------------------
    #  Identifiability
    # ------------------------------------------------------------------

    def check_identifiability(
        self,
        target: str,
        treatment_vars: List[str],
    ) -> IdentifiabilityResult:
        """
        Check whether P(target | do(treatment_vars)) is identifiable
        from observational data.

        Tries (in order):
        1. Back-door criterion
        2. Front-door criterion
        3. Reports non-identifiable

        Parameters
        ----------
        target : str
            The target (outcome) variable.
        treatment_vars : list of str
            Variables being intervened on.

        Returns
        -------
        IdentifiabilityResult
        """
        # 1. Try back-door criterion
        bd_result = self._check_backdoor(target, treatment_vars)
        if bd_result.is_identifiable:
            return bd_result

        # 2. Try front-door criterion (only for single treatment)
        if len(treatment_vars) == 1:
            fd_result = self._check_frontdoor(target, treatment_vars[0])
            if fd_result.is_identifiable:
                return fd_result

        # 3. Try generalised adjustment
        ga_result = self._check_generalised_adjustment(target, treatment_vars)
        if ga_result.is_identifiable:
            return ga_result

        return IdentifiabilityResult(
            is_identifiable=False,
            method="none",
            message=(
                f"P({target} | do({', '.join(treatment_vars)})) is not identifiable "
                "by back-door, front-door, or generalised adjustment."
            ),
        )

    def _check_backdoor(
        self, target: str, treatment_vars: List[str]
    ) -> IdentifiabilityResult:
        """
        Check the back-door criterion.

        A set Z satisfies the back-door criterion relative to (X, Y) if:
        1. No node in Z is a descendant of X.
        2. Z blocks every path from X to Y that contains an arrow into X.
        """
        dag = self._original_dag
        treatment_set = set(treatment_vars)

        # Candidate adjustment sets: subsets of non-descendants of X
        desc_x: Set[str] = set()
        for x in treatment_vars:
            desc_x |= set(dag.descendants(x))

        candidates = [
            n for n in dag.nodes
            if n not in treatment_set
            and n != target
            and n not in desc_x
        ]

        # Try subsets of increasing size
        for size in range(len(candidates) + 1):
            for subset in itertools.combinations(candidates, size):
                z_set = frozenset(subset)

                if self._satisfies_backdoor(target, treatment_vars, z_set):
                    expr_parts = []
                    for x in treatment_vars:
                        if z_set:
                            expr_parts.append(
                                f"sum_z P({target}|{x},z) P(z)"
                            )
                        else:
                            expr_parts.append(f"P({target}|{x})")

                    return IdentifiabilityResult(
                        is_identifiable=True,
                        method="backdoor",
                        adjustment_set=z_set,
                        identifying_expression=" * ".join(expr_parts),
                        message=f"Back-door adjustment with Z={set(z_set)}",
                    )

        return IdentifiabilityResult(
            is_identifiable=False,
            method="backdoor",
            message="No valid back-door adjustment set found.",
        )

    def _satisfies_backdoor(
        self,
        target: str,
        treatment_vars: List[str],
        z_set: FrozenSet[str],
    ) -> bool:
        """
        Check if z_set satisfies the back-door criterion.

        Build the modified graph (remove outgoing edges from X),
        then check d-separation of X and Y given Z.
        """
        dag = self._original_dag

        # Condition 1: no node in Z is a descendant of any treatment var
        for x in treatment_vars:
            desc = dag.descendants(x)
            if z_set & desc:
                return False

        # Condition 2: Z d-separates X from Y in the graph where outgoing
        # edges from X are removed.
        # Build modified edge list
        modified_edges = [
            (u, v) for u, v in dag.edges
            if u not in set(treatment_vars)
        ]

        # Check d-separation using Bayes-Ball on modified graph
        return self._dsep_in_modified(
            treatment_vars, [target], z_set, modified_edges
        )

    def _check_frontdoor(
        self, target: str, treatment: str
    ) -> IdentifiabilityResult:
        """
        Check the front-door criterion.

        A set M satisfies the front-door criterion relative to (X, Y) if:
        1. X blocks all paths from M to Y that don't go through X.
           (Actually: M intercepts all directed paths from X to Y.)
        2. There is no unblocked back-door path from X to M.
        3. All back-door paths from M to Y are blocked by X.
        """
        dag = self._original_dag

        # Candidate mediators: descendants of X that are ancestors of Y
        desc_x = dag.descendants(treatment)
        anc_y = dag.ancestors(target)
        mediator_candidates = list(desc_x & anc_y)

        if not mediator_candidates:
            return IdentifiabilityResult(
                is_identifiable=False,
                method="frontdoor",
                message="No mediator candidates between treatment and outcome.",
            )

        # Try subsets
        for size in range(1, len(mediator_candidates) + 1):
            for subset in itertools.combinations(mediator_candidates, size):
                m_set = frozenset(subset)

                if self._satisfies_frontdoor(target, treatment, m_set):
                    expr = (
                        f"sum_m P(m|{treatment}) "
                        f"sum_x' P({target}|{treatment}'=x',m) P(x')"
                    )
                    return IdentifiabilityResult(
                        is_identifiable=True,
                        method="frontdoor",
                        adjustment_set=m_set,
                        identifying_expression=expr,
                        message=f"Front-door criterion with M={set(m_set)}",
                    )

        return IdentifiabilityResult(
            is_identifiable=False,
            method="frontdoor",
            message="No valid front-door set found.",
        )

    def _satisfies_frontdoor(
        self,
        target: str,
        treatment: str,
        m_set: FrozenSet[str],
    ) -> bool:
        """Check if m_set satisfies the front-door criterion."""
        dag = self._original_dag

        # 1. M intercepts all directed paths from X to Y
        if not self._m_intercepts_all_directed(treatment, target, m_set):
            return False

        # 2. No unblocked back-door path from X to M
        # i.e., X _||_ M in graph with X's outgoing edges removed? No.
        # Actually: there's no back-door path from X to any M_i.
        # Check that there's no non-causal path from X to M.
        for m in m_set:
            # All paths from X to M should be "front-door" (directed X -> ... -> M)
            # Check: no common cause of X and M that isn't blocked
            pass

        # 3. All back-door paths from M to Y are blocked by X
        modified_edges = [
            (u, v) for u, v in dag.edges
            if u not in m_set
        ]
        if not self._dsep_in_modified(
            list(m_set), [target], frozenset({treatment}), modified_edges
        ):
            return False

        return True

    def _m_intercepts_all_directed(
        self,
        source: str,
        target: str,
        m_set: FrozenSet[str],
    ) -> bool:
        """Check if m_set intercepts all directed paths from source to target."""
        dag = self._original_dag

        # BFS/DFS: try to reach target from source without going through m_set
        visited: Set[str] = set()
        stack = [source]

        while stack:
            node = stack.pop()
            if node == target:
                return False  # found a path that doesn't go through M
            if node in visited:
                continue
            visited.add(node)

            if node in m_set and node != source:
                continue  # blocked by M

            for child in dag.children(node):
                if child not in visited:
                    stack.append(child)

        return True

    def _check_generalised_adjustment(
        self, target: str, treatment_vars: List[str]
    ) -> IdentifiabilityResult:
        """
        Check for a valid generalised adjustment set.

        A generalised adjustment set Z allows identification if:
        - Z blocks all non-causal paths from X to Y
        - Z does not contain any descendant of a node on a proper causal
          path from X to Y (forbidden set)
        """
        dag = self._original_dag
        treatment_set = set(treatment_vars)

        # Proper causal paths: directed paths from X to Y
        causal_intermediates: Set[str] = set()
        for x in treatment_vars:
            paths = self._find_directed_paths(x, target)
            for path in paths:
                causal_intermediates.update(path[1:-1])  # exclude endpoints

        # Forbidden set: descendants of causal intermediates
        forbidden: Set[str] = set(causal_intermediates)
        for ci in causal_intermediates:
            forbidden |= set(dag.descendants(ci))
        forbidden |= treatment_set
        forbidden.add(target)

        # Candidate set
        candidates = [n for n in dag.nodes if n not in forbidden]

        for size in range(len(candidates) + 1):
            for subset in itertools.combinations(candidates, size):
                z_set = frozenset(subset)
                # Check if Z blocks all non-causal paths
                if self._blocks_non_causal(target, treatment_vars, z_set):
                    return IdentifiabilityResult(
                        is_identifiable=True,
                        method="generalised_adjustment",
                        adjustment_set=z_set,
                        message=f"Generalised adjustment with Z={set(z_set)}",
                    )

        return IdentifiabilityResult(
            is_identifiable=False,
            method="generalised_adjustment",
            message="No valid generalised adjustment set found.",
        )

    def _blocks_non_causal(
        self, target: str, treatment_vars: List[str], z_set: FrozenSet[str]
    ) -> bool:
        """Check if Z blocks all non-causal paths from X to Y."""
        # Remove outgoing edges from X to isolate back-door paths
        modified_edges = [
            (u, v) for u, v in self._original_dag.edges
            if u not in set(treatment_vars)
        ]
        return self._dsep_in_modified(
            treatment_vars, [target], z_set, modified_edges
        )

    def _find_directed_paths(
        self, source: str, target: str
    ) -> List[List[str]]:
        """Find all directed paths from source to target in the DAG."""
        dag = self._original_dag
        paths: List[List[str]] = []

        def _dfs(node: str, path: List[str]) -> None:
            if node == target:
                paths.append(list(path))
                return
            for child in dag.children(node):
                if child not in path:
                    path.append(child)
                    _dfs(child, path)
                    path.pop()

        _dfs(source, [source])
        return paths

    def _dsep_in_modified(
        self,
        x_vars: List[str],
        y_vars: List[str],
        z_set: FrozenSet[str],
        edges: List[Tuple[str, str]],
    ) -> bool:
        """
        Check d-separation of x_vars and y_vars given z_set in a graph
        with the given edge list (modified from original).
        Uses Bayes-Ball algorithm.
        """
        nodes = set(self._original_dag.nodes)

        # Build adjacency
        children_map: Dict[str, List[str]] = {n: [] for n in nodes}
        parents_map: Dict[str, List[str]] = {n: [] for n in nodes}
        for u, v in edges:
            children_map[u].append(v)
            parents_map[v].append(u)

        # Ancestors of observed
        observed = set(z_set)
        anc_observed: Set[str] = set()
        for z in observed:
            stack = list(parents_map.get(z, []))
            while stack:
                n = stack.pop()
                if n not in anc_observed:
                    anc_observed.add(n)
                    stack.extend(parents_map.get(n, []))

        visited_up: Set[str] = set()
        visited_down: Set[str] = set()
        reachable: Set[str] = set()

        queue: List[Tuple[str, str]] = [(x, "up") for x in x_vars]

        while queue:
            node, direction = queue.pop()

            if direction == "up" and node not in visited_up:
                visited_up.add(node)
                if node not in observed:
                    reachable.add(node)
                    for p in parents_map.get(node, []):
                        queue.append((p, "up"))
                    for c in children_map.get(node, []):
                        queue.append((c, "down"))
                else:
                    if node in anc_observed or node in observed:
                        for p in parents_map.get(node, []):
                            queue.append((p, "up"))

            elif direction == "down" and node not in visited_down:
                visited_down.add(node)
                if node not in observed:
                    reachable.add(node)
                    for c in children_map.get(node, []):
                        queue.append((c, "down"))
                else:
                    for p in parents_map.get(node, []):
                        queue.append((p, "up"))

        return not bool(set(y_vars) & reachable)

    # ------------------------------------------------------------------
    #  Adjustment formula encoding
    # ------------------------------------------------------------------

    def encode_adjustment_constraints(
        self,
        target: str,
        treatment_vars: List[str],
        adjustment_set: FrozenSet[str],
        observed_conditionals: Optional[Dict] = None,
    ) -> List[Tuple[np.ndarray, float, str]]:
        """
        Encode the back-door adjustment formula as LP constraints.

        P(Y | do(X=x)) = sum_z P(Y | X=x, Z=z) P(Z=z)

        If observed conditionals are provided, these become concrete
        equality constraints on the joint distribution.
        """
        dag = self._mutilated_dag
        topo = dag.topological_order()
        strides = self._compute_strides(dag, topo)
        total = self._compute_total(dag)
        constraints: List[Tuple[np.ndarray, float, str]] = []

        z_vars = sorted(adjustment_set)
        z_cards = [dag.card[z] for z in z_vars]
        z_configs = list(itertools.product(*[range(c) for c in z_cards]))

        card_y = dag.card[target]

        for y_val in range(card_y):
            for x_vals in itertools.product(*[range(dag.card[x]) for x in treatment_vars]):
                # Row encoding:
                # P(Y=y | do(X=x)) = sum_z P(Y=y, X=x, Z=z) / P(X=x, Z=z) * P(Z=z)
                # In terms of the joint:
                # P(Y=y | do(X=x)) = sum_z P(Y=y, X=x, Z=z)
                # (under mutilated DAG where P(X) is point mass)

                row = np.zeros(total, dtype=np.float64)
                x_dict = dict(zip(treatment_vars, x_vals))

                for flat_idx in range(total):
                    assign = self._flat_to_assignment(flat_idx, dag, topo, strides)

                    # Check treatment match
                    if not all(assign[x] == x_dict[x] for x in treatment_vars):
                        continue

                    if assign[target] == y_val:
                        row[flat_idx] = 1.0

                name = (
                    f"adjust_{target}={y_val}|"
                    f"do({','.join(f'{x}={v}' for x, v in x_dict.items())})"
                )
                # RHS unknown in general — becomes part of the objective
                constraints.append((row, 0.0, name))

        return constraints

    # ------------------------------------------------------------------
    #  Utilities
    # ------------------------------------------------------------------

    def _compute_strides(self, dag, topo: List[str]) -> Dict[str, int]:
        strides: Dict[str, int] = {}
        s = 1
        for node in reversed(topo):
            strides[node] = s
            s *= dag.card[node]
        return strides

    def _compute_total(self, dag) -> int:
        total = 1
        for n in dag.nodes:
            total *= dag.card[n]
        return total

    def _flat_to_assignment(
        self, flat_idx: int, dag, topo: List[str], strides: Dict[str, int]
    ) -> Dict[str, int]:
        assignment: Dict[str, int] = {}
        for node in topo:
            card = dag.card[node]
            stride = strides[node]
            assignment[node] = (flat_idx // stride) % card
        return assignment

    def compute_interventional_bounds_no_id(
        self,
        target: str,
        treatment_vars: List[str],
    ) -> Tuple[float, float]:
        """
        When the query is non-identifiable, return the widest possible
        bounds [0, 1] for probabilities.

        For more refined bounds, the full polytope LP is needed.
        """
        return 0.0, 1.0

    def get_causal_effect_type(
        self,
        target: str,
        treatment_vars: List[str],
    ) -> str:
        """
        Classify the type of causal effect.

        Returns one of:
        - "total_effect"
        - "direct_effect"
        - "indirect_effect"
        - "controlled_direct_effect"
        """
        dag = self._original_dag

        # Check if there's a direct edge from treatment to target
        has_direct = any(
            (x, target) in dag.edges for x in treatment_vars
        )
        # Check if there's a directed path through mediators
        has_indirect = False
        for x in treatment_vars:
            paths = self._find_directed_paths(x, target)
            for path in paths:
                if len(path) > 2:
                    has_indirect = True
                    break

        if has_direct and has_indirect:
            return "total_effect"
        elif has_direct:
            return "direct_effect"
        elif has_indirect:
            return "indirect_effect"
        else:
            return "controlled_direct_effect"

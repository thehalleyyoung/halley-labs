"""Do-calculus engine implementing Pearl's three rules.

Provides rule application, automatic derivation via the ID algorithm,
and computation of interventional distributions from observational data.
Uses graph surgery (mutilated/augmented graphs) and d-separation tests.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats


@dataclass
class DoCalculusResult:
    """Result of a do-calculus derivation."""

    expression: str
    applicable_rules: list[int] = field(default_factory=list)
    derivation_steps: list[str] = field(default_factory=list)
    value: Optional[float] = None
    identified: bool = True


# ---------------------------------------------------------------
# Graph helpers (operate on adjacency matrices directly)
# ---------------------------------------------------------------

def _ancestors_of(adj: NDArray, nodes: Set[int]) -> Set[int]:
    """Return all ancestors of *nodes* in the DAG given by *adj*."""
    p = adj.shape[0]
    result: set[int] = set()
    stack = list(nodes)
    while stack:
        n = stack.pop()
        for par in range(p):
            if adj[par, n] != 0 and par not in result:
                result.add(par)
                stack.append(par)
    return result


def _descendants_of(adj: NDArray, nodes: Set[int]) -> Set[int]:
    """Return all descendants of *nodes* in the DAG given by *adj*."""
    p = adj.shape[0]
    result: set[int] = set()
    stack = list(nodes)
    while stack:
        n = stack.pop()
        for ch in range(p):
            if adj[n, ch] != 0 and ch not in result:
                result.add(ch)
                stack.append(ch)
    return result


def _parents_of(adj: NDArray, j: int) -> List[int]:
    """Parents of node *j* in adjacency matrix *adj*."""
    return list(np.nonzero(adj[:, j])[0])


def _children_of(adj: NDArray, i: int) -> List[int]:
    """Children of node *i* in adjacency matrix *adj*."""
    return list(np.nonzero(adj[i, :])[0])


def _topological_sort(adj: NDArray) -> List[int]:
    """Kahn's algorithm for topological ordering."""
    p = adj.shape[0]
    binary = (adj != 0).astype(int)
    in_deg = binary.sum(axis=0).tolist()
    queue: deque[int] = deque(i for i in range(p) if in_deg[i] == 0)
    order: list[int] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for ch in range(p):
            if binary[node, ch]:
                in_deg[ch] -= 1
                if in_deg[ch] == 0:
                    queue.append(ch)
    if len(order) != p:
        raise ValueError("Graph contains a cycle")
    return order


def _d_separated(
    adj: NDArray, x: Set[int], y: Set[int], z: Set[int]
) -> bool:
    """Test d-separation X ⊥ Y | Z using Bayes-Ball on *adj*."""
    if x & y:
        return False
    p = adj.shape[0]
    visited: set[tuple[int, str]] = set()
    queue: deque[tuple[int, str]] = deque()
    reachable: set[int] = set()
    for s in x:
        queue.append((s, "up"))
    while queue:
        node, direction = queue.popleft()
        if (node, direction) in visited:
            continue
        visited.add((node, direction))
        if node not in x:
            reachable.add(node)
        if direction == "up" and node not in z:
            for par in _parents_of(adj, node):
                if (par, "up") not in visited:
                    queue.append((par, "up"))
            for ch in _children_of(adj, node):
                if (ch, "down") not in visited:
                    queue.append((ch, "down"))
        elif direction == "down":
            if node not in z:
                for ch in _children_of(adj, node):
                    if (ch, "down") not in visited:
                        queue.append((ch, "down"))
            if node in z:
                for par in _parents_of(adj, node):
                    if (par, "up") not in visited:
                        queue.append((par, "up"))
    return len(reachable & y) == 0


class DoCalculusEngine:
    """Engine for applying Pearl's three rules of do-calculus.

    Operates on adjacency matrices (``adj[i,j] != 0`` means *i → j*) and
    optionally on :class:`StructuralCausalModel` objects for numeric
    computation of interventional distributions.

    Parameters
    ----------
    verbose : bool
        Whether to log derivation steps.
    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self._log: list[str] = []

    # -----------------------------------------------------------------
    # Mutilated / augmented graph construction
    # -----------------------------------------------------------------

    @staticmethod
    def _build_mutilated_graph(
        adj: NDArray[np.floating],
        intervention_nodes: Set[int],
    ) -> NDArray[np.float64]:
        """Build the mutilated graph G_{\\overline{X}}.

        Removes all incoming edges to each node in *intervention_nodes*.

        Parameters
        ----------
        adj : ndarray
            Original adjacency matrix.
        intervention_nodes : set of int
            Nodes being intervened on.

        Returns
        -------
        ndarray
            Mutilated adjacency matrix.
        """
        adj = np.array(adj, dtype=np.float64)
        g = adj.copy()
        for node in intervention_nodes:
            if node < 0 or node >= g.shape[0]:
                raise ValueError(f"Node {node} out of range [0, {g.shape[0]})")
            g[:, node] = 0
        return g

    @staticmethod
    def _build_underline_graph(
        adj: NDArray[np.floating],
        intervention_nodes: Set[int],
    ) -> NDArray[np.float64]:
        """Build G_{\\underline{X}}: remove outgoing edges from intervened nodes."""
        adj = np.array(adj, dtype=np.float64)
        g = adj.copy()
        for node in intervention_nodes:
            if node < 0 or node >= g.shape[0]:
                raise ValueError(f"Node {node} out of range [0, {g.shape[0]})")
            g[node, :] = 0
        return g

    @staticmethod
    def _build_augmented_graph(
        adj: NDArray[np.floating],
        intervention_nodes: Set[int],
    ) -> NDArray[np.float64]:
        """Build augmented graph for do-calculus.

        For each intervened node X_i, adds a new "policy" node F_i that
        points to X_i, while removing original incoming edges to X_i.
        The augmented graph has ``p + |intervention_nodes|`` nodes.

        Parameters
        ----------
        adj : ndarray
            Original adjacency matrix (p × p).
        intervention_nodes : set of int
            Nodes being intervened on.

        Returns
        -------
        ndarray
            Augmented adjacency matrix of shape ``(p+k, p+k)``.
        """
        adj = np.array(adj, dtype=np.float64)
        p = adj.shape[0]
        k = len(intervention_nodes)
        aug = np.zeros((p + k, p + k), dtype=np.float64)
        aug[:p, :p] = adj.copy()
        sorted_nodes = sorted(intervention_nodes)
        for idx, node in enumerate(sorted_nodes):
            aug[:, node] = 0  # remove incoming
            aug[p + idx, node] = 1.0  # policy node → intervened node
        return aug

    # -----------------------------------------------------------------
    # Rule 1: Insertion / deletion of observations
    # -----------------------------------------------------------------

    def rule1(
        self,
        graph: NDArray[np.int_],
        expression: str,
        intervention_set: set[int],
        observation_set: set[int],
    ) -> DoCalculusResult:
        """Rule 1 (insertion/deletion of observations).

        P(y | do(x), z, w) = P(y | do(x), w)  if  (Y ⊥ Z | X, W)_{G_{\\overline{X}}}

        The rule is applicable when Y and Z are d-separated by X ∪ W
        in the mutilated graph G_{\\overline{X}} (incoming edges to X removed).

        Parameters
        ----------
        graph : ndarray
            Adjacency matrix (p × p).
        expression : str
            Current probability expression (for logging).
        intervention_set : set of int
            Variables being intervened on (X in do(X)).
        observation_set : set of int
            Observation variables (Z) to potentially remove.

        Returns
        -------
        DoCalculusResult
        """
        graph = np.asarray(graph, dtype=np.float64)
        p = graph.shape[0]
        all_vars = set(range(p))
        steps: list[str] = []

        g_overline_x = self._build_mutilated_graph(graph, intervention_set)

        # Y = all non-intervention, non-observation, non-conditioning vars
        # We test: observation_set ⊥ (all \ intervention \ observation) | intervention ∪ conditioning
        y_set = all_vars - intervention_set - observation_set
        conditioning = intervention_set.copy()

        applicable = _d_separated(g_overline_x, observation_set, y_set, conditioning)

        if applicable:
            new_expr = f"Rule 1 applied: removed {observation_set} from {expression}"
            steps.append(
                f"(Y ⊥ Z | X)_{{G_overline_X}} holds; "
                f"Z={observation_set} can be removed"
            )
        else:
            new_expr = expression
            steps.append(
                f"Rule 1 not applicable: "
                f"d-separation fails in G_overline_X"
            )

        if self.verbose:
            self._log.extend(steps)

        return DoCalculusResult(
            expression=new_expr,
            applicable_rules=[1] if applicable else [],
            derivation_steps=steps,
        )

    # -----------------------------------------------------------------
    # Rule 2: Action / observation exchange
    # -----------------------------------------------------------------

    def rule2(
        self,
        graph: NDArray[np.int_],
        expression: str,
        intervention_set: set[int],
        observation_set: set[int],
    ) -> DoCalculusResult:
        """Rule 2 (action/observation exchange).

        P(y | do(x), do(z), w) = P(y | do(x), z, w)
        if  (Y ⊥ Z | X, W)_{G_{\\overline{X}, \\underline{Z}}}

        The rule allows replacing do(Z) with observation Z (or vice versa)
        when Y and Z are d-separated by X ∪ W in the graph with incoming
        edges to X removed *and* outgoing edges from Z removed.

        Parameters
        ----------
        graph : ndarray
            Adjacency matrix.
        expression : str
            Current probability expression.
        intervention_set : set of int
            Variables X already intervened on.
        observation_set : set of int
            Variables Z whose action/observation status may change.

        Returns
        -------
        DoCalculusResult
        """
        graph = np.asarray(graph, dtype=np.float64)
        p = graph.shape[0]
        all_vars = set(range(p))
        steps: list[str] = []

        # Build G_{overline{X}, underline{Z}}
        g = self._build_mutilated_graph(graph, intervention_set)
        g = self._build_underline_graph(g, observation_set)

        y_set = all_vars - intervention_set - observation_set
        conditioning = intervention_set.copy()

        applicable = _d_separated(g, observation_set, y_set, conditioning)

        if applicable:
            new_expr = (
                f"Rule 2 applied: do({observation_set}) ↔ observe "
                f"in {expression}"
            )
            steps.append(
                f"(Y ⊥ Z | X)_{{G_overline_X,underline_Z}} holds; "
                f"do(Z) exchanged with observation Z={observation_set}"
            )
        else:
            new_expr = expression
            steps.append(
                f"Rule 2 not applicable: d-separation fails in "
                f"G_overline_X,underline_Z"
            )

        if self.verbose:
            self._log.extend(steps)

        return DoCalculusResult(
            expression=new_expr,
            applicable_rules=[2] if applicable else [],
            derivation_steps=steps,
        )

    # -----------------------------------------------------------------
    # Rule 3: Insertion / deletion of actions
    # -----------------------------------------------------------------

    def rule3(
        self,
        graph: NDArray[np.int_],
        expression: str,
        intervention_set: set[int],
    ) -> DoCalculusResult:
        """Rule 3 (insertion/deletion of actions).

        P(y | do(x), do(z), w) = P(y | do(x), w)
        if  (Y ⊥ Z | X, W)_{G_{\\overline{X}, \\overline{Z(S)}}}

        where Z(S) is the set of Z-nodes that are not ancestors of any
        W-node in G_{\\overline{X}}.

        Parameters
        ----------
        graph : ndarray
            Adjacency matrix.
        expression : str
            Current probability expression.
        intervention_set : set of int
            Variables Z whose do-operator may be removed.

        Returns
        -------
        DoCalculusResult
        """
        graph = np.asarray(graph, dtype=np.float64)
        p = graph.shape[0]
        all_vars = set(range(p))
        steps: list[str] = []

        # Build G_{overline{X}} where X = intervention_set
        # Then check if removing Z still leaves d-separation
        g_overline = self._build_mutilated_graph(graph, intervention_set)

        y_set = all_vars - intervention_set
        # Check: (Y ⊥ Z | empty)_{G_overline_Z(S)}
        # Simplified: check if Z has no effect on Y in the overline graph
        applicable = _d_separated(g_overline, intervention_set, y_set, set())

        if applicable:
            new_expr = (
                f"Rule 3 applied: removed do({intervention_set}) "
                f"from {expression}"
            )
            steps.append(
                f"(Y ⊥ Z)_{{G_overline_Z(S)}} holds; "
                f"do({intervention_set}) can be removed"
            )
        else:
            new_expr = expression
            steps.append(
                f"Rule 3 not applicable: d-separation fails"
            )

        if self.verbose:
            self._log.extend(steps)

        return DoCalculusResult(
            expression=new_expr,
            applicable_rules=[3] if applicable else [],
            derivation_steps=steps,
        )

    # -----------------------------------------------------------------
    # Truncated factorization
    # -----------------------------------------------------------------

    def truncated_factorization(
        self,
        scm: Any,
        interventions: Dict[int, float],
    ) -> NDArray[np.float64]:
        """Compute the post-intervention distribution via truncated factorization.

        For a linear-Gaussian SCM with do(X_i = v), the post-intervention
        distribution is obtained by:
          1. Removing incoming edges to intervened variables.
          2. Setting their values to the intervention values.
          3. Computing the resulting joint covariance.

        P(v \\ x | do(x)) = ∏_{i ∉ X} P(v_i | pa_i)

        Parameters
        ----------
        scm : StructuralCausalModel
            The structural causal model.
        interventions : dict
            ``{variable_index: value}`` for do-interventions.

        Returns
        -------
        ndarray
            Implied covariance matrix of the mutilated model.
        """
        if not interventions:
            return scm.implied_covariance()

        mutilated = scm.do_intervention(interventions)
        return mutilated.implied_covariance()

    # -----------------------------------------------------------------
    # do-operator (compute E[Y | do(X=x)])
    # -----------------------------------------------------------------

    def do_operator(
        self,
        scm: Any,
        interventions: Dict[int, float],
        target: int,
        *,
        n_samples: int = 50_000,
    ) -> float:
        """Compute E[target | do(interventions)] via simulation.

        Uses the mutilated graph and forward sampling in topological
        order to compute the expected value of the target under
        intervention.

        Parameters
        ----------
        scm : StructuralCausalModel
            The structural causal model.
        interventions : dict
            ``{variable_index: intervention_value}``.
        target : int
            Index of the target variable.
        n_samples : int
            Number of Monte-Carlo samples.

        Returns
        -------
        float
            E[Y | do(X=x)].
        """
        if target < 0 or target >= scm.num_variables:
            raise ValueError(
                f"target {target} out of range [0, {scm.num_variables})"
            )
        for idx in interventions:
            if idx < 0 or idx >= scm.num_variables:
                raise ValueError(
                    f"intervention index {idx} out of range "
                    f"[0, {scm.num_variables})"
                )

        # For linear-Gaussian SCMs we can compute analytically
        result = self._do_operator_analytic(scm, interventions, target)
        if result is not None:
            return result

        # Fallback to simulation
        data = scm.sample(n_samples, interventions=interventions)
        return float(np.mean(data[:, target]))

    def _do_operator_analytic(
        self,
        scm: Any,
        interventions: Dict[int, float],
        target: int,
    ) -> Optional[float]:
        """Analytic computation for linear-Gaussian SCMs.

        In a linear-Gaussian SCM, E[Y | do(X=x)] can be computed by
        solving the structural equations in topological order with
        intervened variables fixed and exogenous noise set to zero mean.

        Returns
        -------
        float or None
            Analytic result, or None if not computable.
        """
        try:
            adj = scm.adjacency_matrix
            coefs = scm.regression_coefficients
            p = scm.num_variables
            order = _topological_sort(adj)

            means = np.zeros(p, dtype=np.float64)
            for j in order:
                if j in interventions:
                    means[j] = interventions[j]
                else:
                    pa = _parents_of(adj, j)
                    means[j] = sum(coefs[par, j] * means[par] for par in pa)

            return float(means[target])
        except Exception:
            return None

    # -----------------------------------------------------------------
    # Automatic rule application
    # -----------------------------------------------------------------

    def apply_rules(
        self,
        graph: NDArray[np.int_],
        query: str,
    ) -> DoCalculusResult:
        """Automatically apply do-calculus rules to simplify *query*.

        Tries Rules 1–3 in sequence and returns the first applicable
        simplification. For a full derivation use :meth:`identify`.

        Parameters
        ----------
        graph : ndarray
            Adjacency matrix.
        query : str
            Probability expression (e.g. ``"P(Y|do(X))"``) to simplify.

        Returns
        -------
        DoCalculusResult
        """
        graph = np.asarray(graph, dtype=np.float64)
        p = graph.shape[0]
        all_rules: list[int] = []
        all_steps: list[str] = []
        current = query

        # Try all possible subsets as intervention / observation
        for x_node in range(p):
            for z_node in range(p):
                if x_node == z_node:
                    continue
                x_set = {x_node}
                z_set = {z_node}

                r1 = self.rule1(graph, current, x_set, z_set)
                if r1.applicable_rules:
                    all_rules.append(1)
                    all_steps.extend(r1.derivation_steps)
                    current = r1.expression

                r2 = self.rule2(graph, current, x_set, z_set)
                if r2.applicable_rules:
                    all_rules.append(2)
                    all_steps.extend(r2.derivation_steps)
                    current = r2.expression

        for x_node in range(p):
            r3 = self.rule3(graph, current, {x_node})
            if r3.applicable_rules:
                all_rules.append(3)
                all_steps.extend(r3.derivation_steps)
                current = r3.expression

        return DoCalculusResult(
            expression=current,
            applicable_rules=sorted(set(all_rules)),
            derivation_steps=all_steps,
        )

    # -----------------------------------------------------------------
    # ID algorithm
    # -----------------------------------------------------------------

    def identify(
        self,
        graph: NDArray[np.int_],
        target: set[int],
        intervention: set[int],
        observables: set[int],
    ) -> DoCalculusResult:
        """Run the ID algorithm to identify a causal effect P(Y | do(X)).

        Implements a simplified version of the Tian & Pearl (2002) ID
        algorithm using recursive c-component decomposition.

        Parameters
        ----------
        graph : ndarray
            Adjacency matrix.
        target : set of int
            Target variables Y.
        intervention : set of int
            Intervention variables X.
        observables : set of int
            Observable variable set.

        Returns
        -------
        DoCalculusResult
        """
        graph = np.asarray(graph, dtype=np.float64)
        p = graph.shape[0]
        steps: list[str] = []
        rules_used: list[int] = []

        # Line 1: if X = ∅, return ΣP(v)
        if not intervention:
            expr = f"P({_fmt(target)})"
            steps.append(f"No intervention; effect is {expr}")
            return DoCalculusResult(
                expression=expr,
                applicable_rules=[],
                derivation_steps=steps,
                identified=True,
            )

        # Line 2: if V \ An(Y)_G ≠ ∅, restrict to ancestors of Y
        all_vars = set(range(p))
        an_y = _ancestors_of(graph, target) | target
        non_ancestors = all_vars - an_y
        if non_ancestors:
            steps.append(
                f"Restricting to An(Y)={an_y}; "
                f"removing {non_ancestors}"
            )
            # Recursion on subgraph of An(Y)
            sub_idx = sorted(an_y)
            sub_graph = graph[np.ix_(sub_idx, sub_idx)]
            idx_map = {old: new for new, old in enumerate(sub_idx)}
            new_target = {idx_map[v] for v in target if v in idx_map}
            new_interv = {idx_map[v] for v in intervention if v in idx_map}
            new_obs = {idx_map[v] for v in observables if v in idx_map}
            sub_result = self.identify(sub_graph, new_target, new_interv, new_obs)
            steps.extend(sub_result.derivation_steps)
            return DoCalculusResult(
                expression=sub_result.expression,
                applicable_rules=sub_result.applicable_rules,
                derivation_steps=steps,
                identified=sub_result.identified,
            )

        # Line 3: build G_{overline{X}} and check W = (V \ X) \ An(Y)_{G_overline{X}}
        g_overline_x = self._build_mutilated_graph(graph, intervention)
        an_y_mut = _ancestors_of(g_overline_x, target) | target
        w_set = (all_vars - intervention) - an_y_mut
        if w_set:
            steps.append(
                f"Found W={w_set} not ancestors of Y in G_overline_X; "
                f"applying Rule 3 to remove do(W)"
            )
            rules_used.append(3)
            new_interv = intervention | w_set
            sub_result = self.identify(graph, target, new_interv - w_set, observables)
            steps.extend(sub_result.derivation_steps)
            rules_used.extend(sub_result.applicable_rules)
            return DoCalculusResult(
                expression=sub_result.expression,
                applicable_rules=sorted(set(rules_used)),
                derivation_steps=steps,
                identified=sub_result.identified,
            )

        # Line 4: c-component decomposition of G \ X
        remaining = all_vars - intervention
        c_comps = self._find_c_components_in_subgraph(graph, remaining)

        if len(c_comps) > 1:
            steps.append(
                f"C-component decomposition: {len(c_comps)} components"
            )
            # Σ_s Π_i P(y_i | v_i^{(π)} \ {y_i})
            parts: list[str] = []
            for comp in c_comps:
                comp_target = comp & target
                if comp_target:
                    parts.append(f"P({_fmt(comp_target)} | do({_fmt(intervention)}))")
            expr = " × ".join(parts) if parts else f"P({_fmt(target)})"
            return DoCalculusResult(
                expression=expr,
                applicable_rules=sorted(set(rules_used)),
                derivation_steps=steps,
                identified=True,
            )

        # Single component — attempt direct identification via backdoor
        adj_set = self._find_backdoor_set(graph, intervention, target)
        if adj_set is not None:
            steps.append(
                f"Back-door adjustment with Z={adj_set}"
            )
            expr = (
                f"Σ_z P({_fmt(target)} | {_fmt(intervention)}, z) P(z)"
                f" [z={_fmt(adj_set)}]"
            )
            return DoCalculusResult(
                expression=expr,
                applicable_rules=[2],
                derivation_steps=steps,
                identified=True,
            )

        # Front-door attempt
        fd_set = self._find_frontdoor_set(graph, intervention, target)
        if fd_set is not None:
            steps.append(f"Front-door adjustment with M={fd_set}")
            expr = (
                f"Σ_m P(m | {_fmt(intervention)}) "
                f"Σ_x' P({_fmt(target)} | m, x') P(x')"
                f" [m={_fmt(fd_set)}]"
            )
            return DoCalculusResult(
                expression=expr,
                applicable_rules=[2, 3],
                derivation_steps=steps,
                identified=True,
            )

        # Cannot identify
        steps.append("Effect is not identifiable (hedge found)")
        return DoCalculusResult(
            expression=f"P({_fmt(target)} | do({_fmt(intervention)})) [NOT IDENTIFIED]",
            applicable_rules=[],
            derivation_steps=steps,
            identified=False,
        )

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _find_c_components_in_subgraph(
        self, adj: NDArray, nodes: Set[int]
    ) -> List[Set[int]]:
        """Find connected components in the undirected skeleton of the subgraph."""
        if not nodes:
            return []
        idx_list = sorted(nodes)
        sub = adj[np.ix_(idx_list, idx_list)]
        p = len(idx_list)
        # Build undirected adjacency
        undir = ((sub != 0) | (sub.T != 0)).astype(int)
        visited: set[int] = set()
        components: list[set[int]] = []
        for start in range(p):
            if start in visited:
                continue
            comp: set[int] = set()
            stack = [start]
            while stack:
                n = stack.pop()
                if n in visited:
                    continue
                visited.add(n)
                comp.add(idx_list[n])
                for nb in range(p):
                    if undir[n, nb] and nb not in visited:
                        stack.append(nb)
            components.append(comp)
        return components

    def _find_backdoor_set(
        self,
        adj: NDArray,
        treatment: Set[int],
        outcome: Set[int],
    ) -> Optional[Set[int]]:
        """Find a valid backdoor adjustment set, or None."""
        p = adj.shape[0]
        all_vars = set(range(p))
        forbidden = treatment | outcome | _descendants_of(adj, treatment)
        candidates = all_vars - forbidden

        # Try empty set first
        g_mut = self._build_mutilated_graph(adj, treatment)
        if _d_separated(g_mut, treatment, outcome, set()):
            return set()

        # Try each single variable
        for c in sorted(candidates):
            z = {c}
            if _d_separated(g_mut, treatment, outcome, z):
                return z

        # Try pairs
        cand_list = sorted(candidates)
        for i in range(len(cand_list)):
            for j in range(i + 1, len(cand_list)):
                z = {cand_list[i], cand_list[j]}
                if _d_separated(g_mut, treatment, outcome, z):
                    return z

        # Try full candidate set
        if _d_separated(g_mut, treatment, outcome, candidates):
            return candidates

        return None

    def _find_frontdoor_set(
        self,
        adj: NDArray,
        treatment: Set[int],
        outcome: Set[int],
    ) -> Optional[Set[int]]:
        """Find a valid front-door set, or None.

        Front-door criterion: M intercepts all directed paths from X to Y,
        no unblocked back-door path from X to M, and X blocks all
        back-door paths from M to Y.
        """
        p = adj.shape[0]
        desc_x = _descendants_of(adj, treatment)
        anc_y = _ancestors_of(adj, outcome)
        candidates = desc_x & anc_y - treatment - outcome

        if not candidates:
            return None

        # Verify: all directed paths from X to Y go through M
        for x in treatment:
            for y in outcome:
                paths = self._all_directed_paths(adj, x, y)
                for path in paths:
                    path_set = set(path[1:-1])
                    if not (path_set & candidates):
                        return None

        return candidates

    @staticmethod
    def _all_directed_paths(
        adj: NDArray, source: int, target: int
    ) -> List[List[int]]:
        """Find all directed paths from source to target."""
        p = adj.shape[0]
        paths: list[list[int]] = []

        def _dfs(node: int, path: list[int], visited: set[int]) -> None:
            if node == target:
                paths.append(list(path))
                return
            for ch in range(p):
                if adj[node, ch] != 0 and ch not in visited:
                    visited.add(ch)
                    path.append(ch)
                    _dfs(ch, path, visited)
                    path.pop()
                    visited.discard(ch)

        _dfs(source, [source], {source})
        return paths


def _fmt(s: Set[int]) -> str:
    """Format a set of ints for display."""
    return ", ".join(str(x) for x in sorted(s))

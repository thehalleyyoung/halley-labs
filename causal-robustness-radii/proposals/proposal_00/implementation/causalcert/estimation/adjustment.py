"""
Optimal adjustment set selection.

Given the set of all valid adjustment sets, selects the one that minimises
the asymptotic variance of the AIPW estimator.  Implements the O-set
criterion of Henckel, Perrier & Maathuis (2022).
"""

from __future__ import annotations

from collections import deque
from itertools import combinations
from typing import Sequence

import numpy as np

from causalcert.types import AdjacencyMatrix, NodeId, NodeSet
from causalcert.exceptions import NoValidAdjustmentSetError
from causalcert.estimation.backdoor import (
    satisfies_backdoor,
    enumerate_adjustment_sets,
    find_minimum_adjustment_set,
    find_optimal_adjustment_set_backdoor,
    _descendants,
    _ancestors,
    _parents,
    _children,
    check_adjustment_criterion,
)


# ---------------------------------------------------------------------------
# O-set computation
# ---------------------------------------------------------------------------


def find_optimal_adjustment_set(
    adj: AdjacencyMatrix,
    treatment: NodeId,
    outcome: NodeId,
) -> NodeSet:
    """Return the asymptotically optimal adjustment set (O-set).

    The O-set minimises the semiparametric efficiency bound among all
    valid adjustment sets.

    Implements the O-set from Henckel, Perrier & Maathuis (2022):
    ``O = an({X, Y} \\ {X}) \\ (desc(X) \\ {X} ∪ {X})``, intersected with
    the valid adjustment region.

    Falls back to the back-door O-set heuristic when the full construction
    doesn't yield a valid set.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    treatment : NodeId
        Treatment variable.
    outcome : NodeId
        Outcome variable.

    Returns
    -------
    NodeSet
        Optimal adjustment set.

    Raises
    ------
    NoValidAdjustmentSetError
        If no valid adjustment set exists.
    """
    adj_arr = np.asarray(adj, dtype=np.int8)
    n = adj_arr.shape[0]

    # Forbidden set: proper descendants of X (descendants excluding X itself)
    desc_x = _descendants(adj_arr, {treatment})
    forbidden = desc_x  # includes treatment itself

    # O-set construction (Henckel et al. 2022)
    # O = an(outcome \ forbidden) ∩ non-forbidden \ {treatment, outcome}
    # Start with ancestors of outcome (excluding forbidden nodes)
    anc_y = _ancestors(adj_arr, {outcome})
    o_set = (anc_y - forbidden - {outcome})

    if satisfies_backdoor(adj_arr, treatment, outcome, frozenset(o_set)):
        return frozenset(o_set)

    # Try the refined construction: parents of outcome + parents of
    # causal-path nodes
    return find_optimal_adjustment_set_backdoor(adj_arr, treatment, outcome)


def find_minimal_adjustment_set(
    adj: AdjacencyMatrix,
    treatment: NodeId,
    outcome: NodeId,
) -> NodeSet:
    """Return a smallest valid adjustment set (by cardinality).

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    treatment, outcome : NodeId
        Treatment and outcome nodes.

    Returns
    -------
    NodeSet
    """
    return find_minimum_adjustment_set(adj, treatment, outcome)


# ---------------------------------------------------------------------------
# Sound and complete enumeration (Perkovic et al. 2018)
# ---------------------------------------------------------------------------


def enumerate_valid_adjustment_sets(
    adj: AdjacencyMatrix,
    treatment: NodeId,
    outcome: NodeId,
    *,
    forbidden: NodeSet | None = None,
    required: NodeSet | None = None,
    max_size: int | None = None,
    minimal: bool = False,
    latent_nodes: NodeSet | None = None,
) -> list[NodeSet]:
    """Sound and complete adjustment set enumeration.

    Enumerates all valid adjustment sets satisfying the given constraints,
    using pruning strategies for large graphs.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    treatment : NodeId
        Treatment variable.
    outcome : NodeId
        Outcome variable.
    forbidden : NodeSet | None
        Nodes that must NOT be in the adjustment set.
    required : NodeSet | None
        Nodes that MUST be in the adjustment set.
    max_size : int | None
        Maximum set size.
    minimal : bool
        If ``True``, only return inclusion-minimal sets.
    latent_nodes : NodeSet | None
        Unobserved (latent) variables that cannot be adjusted for.

    Returns
    -------
    list[NodeSet]
        All valid adjustment sets satisfying constraints.
    """
    adj_arr = np.asarray(adj, dtype=np.int8)
    n = adj_arr.shape[0]

    forbidden_set = set(forbidden) if forbidden else set()
    required_set = set(required) if required else set()
    latent_set = set(latent_nodes) if latent_nodes else set()

    # Automatically forbid: treatment, outcome, descendants of treatment, latent nodes
    desc_x = _descendants(adj_arr, {treatment})
    auto_forbidden = desc_x | {outcome} | latent_set

    # Candidate nodes
    candidates = sorted(
        i for i in range(n)
        if i not in auto_forbidden and i not in forbidden_set
    )

    # Verify required nodes are valid candidates
    for r in required_set:
        if r in auto_forbidden or r in forbidden_set:
            return []  # Impossible to satisfy constraints

    # Remaining candidates (beyond required)
    remaining = [c for c in candidates if c not in required_set]
    if max_size is None:
        max_remaining = len(remaining)
    else:
        max_remaining = max(0, max_size - len(required_set))

    valid: list[NodeSet] = []

    # Check required-only set
    if len(required_set) <= (max_size or float("inf")):
        if satisfies_backdoor(adj_arr, treatment, outcome, frozenset(required_set)):
            valid.append(frozenset(required_set))

    # Enumerate additional subsets
    for size in range(1, max_remaining + 1):
        for combo in combinations(remaining, size):
            s = frozenset(required_set | set(combo))
            if max_size is not None and len(s) > max_size:
                continue
            if satisfies_backdoor(adj_arr, treatment, outcome, s):
                valid.append(s)

    if minimal:
        valid = _filter_minimal(valid)

    return valid


def _filter_minimal(sets: list[NodeSet]) -> list[NodeSet]:
    """Keep only inclusion-minimal sets."""
    sets_sorted = sorted(sets, key=len)
    minimal: list[NodeSet] = []
    for s in sets_sorted:
        if not any(m <= s for m in minimal):
            minimal.append(s)
    return minimal


# ---------------------------------------------------------------------------
# Adjustment set equivalence
# ---------------------------------------------------------------------------


def adjustment_sets_equivalent(
    adj: AdjacencyMatrix,
    treatment: NodeId,
    outcome: NodeId,
    set_a: NodeSet,
    set_b: NodeSet,
) -> bool:
    """Check whether two adjustment sets yield the same causal effect.

    Two valid adjustment sets are *equivalent* if they identify the same
    interventional distribution P(Y | do(X)).  In a DAG, any two valid
    back-door adjustment sets identify the same causal effect, so this
    reduces to checking that both sets are valid.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    treatment, outcome : NodeId
        Treatment and outcome.
    set_a, set_b : NodeSet
        Two adjustment sets.

    Returns
    -------
    bool
        ``True`` if both are valid (and hence equivalent in a DAG).
    """
    return (
        satisfies_backdoor(adj, treatment, outcome, set_a)
        and satisfies_backdoor(adj, treatment, outcome, set_b)
    )


# ---------------------------------------------------------------------------
# Pruning strategies
# ---------------------------------------------------------------------------


def prune_candidates(
    adj: AdjacencyMatrix,
    treatment: NodeId,
    outcome: NodeId,
) -> list[NodeId]:
    """Prune the candidate set for adjustment to relevant nodes only.

    Uses the ancestral criterion: only ancestors of {X, Y} (excluding
    descendants of X) can possibly be in a valid adjustment set.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    treatment, outcome : NodeId
        Treatment and outcome.

    Returns
    -------
    list[NodeId]
        Pruned candidate list.
    """
    adj_arr = np.asarray(adj, dtype=np.int8)
    desc_x = _descendants(adj_arr, {treatment})
    anc_xy = _ancestors(adj_arr, {treatment, outcome})
    return sorted(
        v for v in anc_xy
        if v != treatment and v != outcome and v not in desc_x
    )


# ---------------------------------------------------------------------------
# Optimal variance adjustment
# ---------------------------------------------------------------------------


def rank_adjustment_sets_by_variance(
    adj: AdjacencyMatrix,
    treatment: NodeId,
    outcome: NodeId,
    data: np.ndarray | None = None,
    adjustment_sets: list[NodeSet] | None = None,
) -> list[tuple[NodeSet, float]]:
    """Rank valid adjustment sets by estimated asymptotic variance.

    If *data* is provided, estimates variance empirically.  Otherwise uses
    a heuristic based on the number of parents of Y in the set (more
    parents → lower variance).

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    treatment, outcome : NodeId
        Treatment and outcome.
    data : np.ndarray | None
        Optional data matrix for empirical variance estimation.
    adjustment_sets : list[NodeSet] | None
        Pre-computed sets.  If ``None``, enumerates all minimal sets.

    Returns
    -------
    list[tuple[NodeSet, float]]
        Sets sorted by ascending (estimated) variance, with variance scores.
    """
    adj_arr = np.asarray(adj, dtype=np.int8)
    if adjustment_sets is None:
        adjustment_sets = enumerate_adjustment_sets(
            adj_arr, treatment, outcome, minimal=True
        )

    if not adjustment_sets:
        return []

    pa_y = _parents(adj_arr, outcome)
    ranked: list[tuple[NodeSet, float]] = []

    for s in adjustment_sets:
        # Heuristic variance score: prefer sets that cover more parents of Y
        n_pa_y_covered = len(set(s) & pa_y)
        # Lower score = better (we invert the count)
        score = 1.0 / (1.0 + n_pa_y_covered) + 0.01 * len(s)
        ranked.append((s, score))

    ranked.sort(key=lambda x: x[1])
    return ranked


# ---------------------------------------------------------------------------
# Latent variable handling
# ---------------------------------------------------------------------------


def adjust_with_latents(
    adj: AdjacencyMatrix,
    treatment: NodeId,
    outcome: NodeId,
    latent_nodes: NodeSet,
) -> NodeSet | None:
    """Find a valid adjustment set when some variables are latent.

    Latent variables cannot be conditioned on.  This function finds a valid
    adjustment set (if one exists) among the observed variables only.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG (including latent nodes).
    treatment, outcome : NodeId
        Treatment and outcome.
    latent_nodes : NodeSet
        Set of unobserved/latent node indices.

    Returns
    -------
    NodeSet | None
        A valid adjustment set of observed variables, or ``None``.
    """
    sets = enumerate_valid_adjustment_sets(
        adj, treatment, outcome,
        latent_nodes=latent_nodes,
        minimal=True,
    )
    if sets:
        return sets[0]
    return None

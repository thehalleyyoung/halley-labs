"""
Systematic DAG perturbation for robustness analysis.

Enumerates, samples, and classifies structural edits to a causal DAG.
Perturbations are guaranteed to preserve acyclicity and can be filtered
by their impact on causal identification and effect estimation.

Classes
-------
- :class:`PerturbedDAG` — A DAG together with the edits that produced it.
- :class:`PerturbationImpact` — Classification of a perturbation's effect.
- :class:`PerturbationGenerator` — Main perturbation generation engine.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

from causalcert.types import (
    AdjacencyMatrix,
    EdgeTuple,
    EditType,
    NodeId,
    NodeSet,
    StructuralEdit,
)


# ============================================================================
# Supporting types
# ============================================================================


class ImpactCategory(Enum):
    """How a perturbation affects causal conclusions."""

    NO_CHANGE = "no_change"
    """The perturbation does not change the identification status or ATE sign."""

    SIGN_CHANGE = "sign_change"
    """The ATE sign flips under the perturbation."""

    MAGNITUDE_CHANGE = "magnitude_change"
    """The ATE magnitude changes significantly (>50 %) but sign is preserved."""

    IDENTIFICATION_CHANGE = "identification_change"
    """The treatment effect is no longer identifiable (or becomes identifiable)."""

    STRUCTURE_ONLY = "structure_only"
    """DAG structure changes but no effect on the T→Y query."""


@dataclass(frozen=True, slots=True)
class PerturbationImpact:
    """Classification of a single perturbation's causal impact.

    Attributes
    ----------
    category : ImpactCategory
        High-level impact class.
    ate_before : float | None
        ATE under the original DAG (if computed).
    ate_after : float | None
        ATE under the perturbed DAG (if computed).
    identified_before : bool
        Whether the effect was identifiable before.
    identified_after : bool
        Whether the effect is identifiable after.
    """

    category: ImpactCategory
    ate_before: float | None = None
    ate_after: float | None = None
    identified_before: bool = True
    identified_after: bool = True


@dataclass(frozen=True, slots=True)
class PerturbedDAG:
    """A DAG produced by applying a sequence of structural edits.

    Attributes
    ----------
    adjacency : AdjacencyMatrix
        The perturbed adjacency matrix.
    edits : tuple[StructuralEdit, ...]
        Edit sequence that produced this DAG.
    edit_distance : int
        Number of edits (= ``len(edits)``).
    impact : PerturbationImpact | None
        Impact classification (populated lazily).
    """

    adjacency: AdjacencyMatrix
    edits: tuple[StructuralEdit, ...] = ()
    edit_distance: int = 0
    impact: PerturbationImpact | None = None


# ============================================================================
# Internal helpers
# ============================================================================


def _is_dag(adj: NDArray) -> bool:
    """Return ``True`` if *adj* is a DAG (topological sort succeeds)."""
    n = adj.shape[0]
    in_deg = adj.sum(axis=0).astype(int).copy()
    queue = deque(int(i) for i in range(n) if in_deg[i] == 0)
    count = 0
    while queue:
        v = queue.popleft()
        count += 1
        for c in range(n):
            if adj[v, c]:
                in_deg[c] -= 1
                if in_deg[c] == 0:
                    queue.append(c)
    return count == n


def _has_directed_path(adj: NDArray, u: int, v: int) -> bool:
    """Check reachability from *u* to *v*."""
    if u == v:
        return True
    visited: set[int] = set()
    queue = deque([u])
    while queue:
        node = queue.popleft()
        for c in np.nonzero(adj[node])[0]:
            c = int(c)
            if c == v:
                return True
            if c not in visited:
                visited.add(c)
                queue.append(c)
    return False


def _apply_edit(adj: NDArray, edit: StructuralEdit) -> NDArray | None:
    """Apply *edit* to a copy of *adj*.  Returns ``None`` if the edit is invalid."""
    new = adj.copy()
    u, v = edit.source, edit.target
    if edit.edit_type == EditType.ADD:
        if new[u, v]:
            return None  # edge already exists
        new[u, v] = 1
    elif edit.edit_type == EditType.DELETE:
        if not new[u, v]:
            return None  # edge doesn't exist
        new[u, v] = 0
    elif edit.edit_type == EditType.REVERSE:
        if not new[u, v]:
            return None
        new[u, v] = 0
        new[v, u] = 1
    else:
        return None
    if not _is_dag(new):
        return None
    return new


def _enumerate_single_edits(adj: NDArray) -> list[StructuralEdit]:
    """Enumerate all valid single-edit perturbations."""
    n = adj.shape[0]
    edits: list[StructuralEdit] = []
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            if adj[u, v]:
                # Can delete
                edits.append(StructuralEdit(EditType.DELETE, u, v))
                # Can reverse (if acyclic)
                edits.append(StructuralEdit(EditType.REVERSE, u, v))
            else:
                # Can add (if acyclic)
                edits.append(StructuralEdit(EditType.ADD, u, v))
    return edits


def _valid_single_edits(adj: NDArray) -> list[tuple[StructuralEdit, NDArray]]:
    """Return single edits that preserve acyclicity, with resulting DAGs."""
    results: list[tuple[StructuralEdit, NDArray]] = []
    for edit in _enumerate_single_edits(adj):
        new = _apply_edit(adj, edit)
        if new is not None:
            results.append((edit, new))
    return results


# ============================================================================
# PerturbationGenerator
# ============================================================================


class PerturbationGenerator:
    """Engine for systematic DAG perturbation and neighbourhood enumeration.

    Parameters
    ----------
    preserve_acyclicity : bool
        If ``True`` (default), all generated perturbations are acyclic.
    max_k : int
        Maximum edit distance for neighbourhood enumeration.
    seed : int
        Random seed for stochastic sampling.
    """

    def __init__(
        self,
        preserve_acyclicity: bool = True,
        max_k: int = 3,
        seed: int = 42,
    ) -> None:
        self.preserve_acyclicity = preserve_acyclicity
        self.max_k = max_k
        self._rng = np.random.default_rng(seed)

    # -- Single-edit perturbations -------------------------------------------

    def single_edit_neighbourhood(
        self, adj: AdjacencyMatrix
    ) -> list[PerturbedDAG]:
        """Return all valid single-edit perturbations.

        Parameters
        ----------
        adj : AdjacencyMatrix
            Original DAG adjacency matrix.

        Returns
        -------
        list[PerturbedDAG]
        """
        adj = np.asarray(adj, dtype=np.int8)
        results: list[PerturbedDAG] = []
        for edit, new_adj in _valid_single_edits(adj):
            results.append(PerturbedDAG(
                adjacency=new_adj,
                edits=(edit,),
                edit_distance=1,
            ))
        return results

    # -- k-edit neighbourhood ------------------------------------------------

    def k_edit_neighbourhood(
        self,
        adj: AdjacencyMatrix,
        k: int,
    ) -> list[PerturbedDAG]:
        """Enumerate all perturbations up to *k* edits.

        Uses breadth-first enumeration with de-duplication.

        Parameters
        ----------
        adj : AdjacencyMatrix
            Original DAG.
        k : int
            Maximum edit distance (≤ ``self.max_k``).

        Returns
        -------
        list[PerturbedDAG]
        """
        adj = np.asarray(adj, dtype=np.int8)
        k = min(k, self.max_k)

        # BFS over edit sequences, keyed by adjacency bytes for dedup
        seen: set[bytes] = {adj.tobytes()}
        frontier: list[tuple[NDArray, tuple[StructuralEdit, ...]]] = [
            (adj, ())
        ]
        all_results: list[PerturbedDAG] = []

        for depth in range(1, k + 1):
            next_frontier: list[tuple[NDArray, tuple[StructuralEdit, ...]]] = []
            for current_adj, current_edits in frontier:
                for edit, new_adj in _valid_single_edits(current_adj):
                    key = new_adj.tobytes()
                    if key not in seen:
                        seen.add(key)
                        new_edits = current_edits + (edit,)
                        pd_item = PerturbedDAG(
                            adjacency=new_adj,
                            edits=new_edits,
                            edit_distance=depth,
                        )
                        all_results.append(pd_item)
                        next_frontier.append((new_adj, new_edits))
            frontier = next_frontier

        return all_results

    # -- Random perturbation sampling ----------------------------------------

    def random_perturbations(
        self,
        adj: AdjacencyMatrix,
        n_samples: int,
        k: int = 1,
    ) -> list[PerturbedDAG]:
        """Sample random perturbations of distance exactly *k*.

        Useful when the full k-neighbourhood is too large to enumerate.

        Parameters
        ----------
        adj : AdjacencyMatrix
            Original DAG.
        n_samples : int
            Number of perturbations to sample.
        k : int
            Edit distance.

        Returns
        -------
        list[PerturbedDAG]
        """
        adj = np.asarray(adj, dtype=np.int8)
        results: list[PerturbedDAG] = []
        seen: set[bytes] = {adj.tobytes()}
        attempts = 0
        max_attempts = n_samples * 50

        while len(results) < n_samples and attempts < max_attempts:
            attempts += 1
            current = adj.copy()
            edits: list[StructuralEdit] = []
            success = True

            for _ in range(k):
                candidates = _valid_single_edits(current)
                if not candidates:
                    success = False
                    break
                idx = self._rng.integers(len(candidates))
                edit, new_adj = candidates[idx]
                edits.append(edit)
                current = new_adj

            if not success:
                continue

            key = current.tobytes()
            if key in seen:
                continue
            seen.add(key)
            results.append(PerturbedDAG(
                adjacency=current,
                edits=tuple(edits),
                edit_distance=k,
            ))

        return results

    # -- Constraint-preserving perturbations ---------------------------------

    def constrained_perturbations(
        self,
        adj: AdjacencyMatrix,
        *,
        protected_edges: Sequence[EdgeTuple] = (),
        forbidden_edges: Sequence[EdgeTuple] = (),
        k: int = 1,
    ) -> list[PerturbedDAG]:
        """Generate perturbations respecting edge constraints.

        Parameters
        ----------
        adj : AdjacencyMatrix
            Original DAG.
        protected_edges : Sequence[EdgeTuple]
            Edges that must not be deleted or reversed.
        forbidden_edges : Sequence[EdgeTuple]
            Edges that must not be added.
        k : int
            Maximum edit distance.

        Returns
        -------
        list[PerturbedDAG]
        """
        adj = np.asarray(adj, dtype=np.int8)
        protected = set(protected_edges)
        forbidden = set(forbidden_edges)

        def is_allowed(edit: StructuralEdit) -> bool:
            e = (edit.source, edit.target)
            if edit.edit_type == EditType.DELETE and e in protected:
                return False
            if edit.edit_type == EditType.REVERSE and e in protected:
                return False
            if edit.edit_type == EditType.ADD and e in forbidden:
                return False
            return True

        # Filter single edits
        base_edits = [
            (e, a) for e, a in _valid_single_edits(adj) if is_allowed(e)
        ]

        if k == 1:
            return [
                PerturbedDAG(adjacency=a, edits=(e,), edit_distance=1)
                for e, a in base_edits
            ]

        # Multi-step: BFS with constraint filtering
        seen: set[bytes] = {adj.tobytes()}
        frontier = [(adj, ())]
        all_results: list[PerturbedDAG] = []

        for depth in range(1, k + 1):
            next_frontier = []
            for cur, cur_edits in frontier:
                candidates = [
                    (e, a) for e, a in _valid_single_edits(cur)
                    if is_allowed(e)
                ]
                for edit, new_adj in candidates:
                    key = new_adj.tobytes()
                    if key not in seen:
                        seen.add(key)
                        new_edits = cur_edits + (edit,)
                        all_results.append(PerturbedDAG(
                            adjacency=new_adj,
                            edits=new_edits,
                            edit_distance=depth,
                        ))
                        next_frontier.append((new_adj, new_edits))
            frontier = next_frontier

        return all_results

    # -- Impact classification -----------------------------------------------

    def classify_impact(
        self,
        original_adj: AdjacencyMatrix,
        perturbed: PerturbedDAG,
        treatment: int,
        outcome: int,
        *,
        weights: NDArray[np.float64] | None = None,
    ) -> PerturbationImpact:
        """Classify the causal impact of a perturbation.

        Parameters
        ----------
        original_adj : AdjacencyMatrix
            Original DAG.
        perturbed : PerturbedDAG
            The perturbation to classify.
        treatment, outcome : int
            Treatment and outcome node indices.
        weights : NDArray | None
            Weight matrix for ATE computation (linear SEM only).

        Returns
        -------
        PerturbationImpact
        """
        orig = np.asarray(original_adj, dtype=np.int8)
        pert = np.asarray(perturbed.adjacency, dtype=np.int8)

        id_before = self._is_identifiable(orig, treatment, outcome)
        id_after = self._is_identifiable(pert, treatment, outcome)

        if id_before != id_after:
            return PerturbationImpact(
                category=ImpactCategory.IDENTIFICATION_CHANGE,
                identified_before=id_before,
                identified_after=id_after,
            )

        if weights is not None:
            ate_before = self._compute_linear_ate(orig, weights, treatment, outcome)
            ate_after = self._compute_linear_ate(pert, weights, treatment, outcome)

            if ate_before * ate_after < 0:
                return PerturbationImpact(
                    category=ImpactCategory.SIGN_CHANGE,
                    ate_before=ate_before,
                    ate_after=ate_after,
                    identified_before=id_before,
                    identified_after=id_after,
                )

            if abs(ate_before) > 1e-10:
                rel_change = abs(ate_after - ate_before) / abs(ate_before)
                if rel_change > 0.5:
                    return PerturbationImpact(
                        category=ImpactCategory.MAGNITUDE_CHANGE,
                        ate_before=ate_before,
                        ate_after=ate_after,
                        identified_before=id_before,
                        identified_after=id_after,
                    )

            return PerturbationImpact(
                category=ImpactCategory.NO_CHANGE,
                ate_before=ate_before,
                ate_after=ate_after,
                identified_before=id_before,
                identified_after=id_after,
            )

        # No weights — structure-only check
        if np.array_equal(
            self._relevant_subgraph(orig, treatment, outcome),
            self._relevant_subgraph(pert, treatment, outcome),
        ):
            return PerturbationImpact(category=ImpactCategory.NO_CHANGE)

        return PerturbationImpact(category=ImpactCategory.STRUCTURE_ONLY)

    @staticmethod
    def _is_identifiable(adj: NDArray, treatment: int, outcome: int) -> bool:
        """Check if the back-door criterion can be satisfied."""
        n = adj.shape[0]
        # Simple check: find if there exists any valid adjustment set
        # (parents of treatment that block all back-door paths)
        pa_t = set(int(p) for p in np.nonzero(adj[:, treatment])[0])
        # Check if conditioning on pa(T) blocks all non-causal paths
        # For simplicity, pa(T) is always a valid adjustment set if
        # no descendant of T is in pa(T)
        desc_t: set[int] = set()
        queue = deque([treatment])
        while queue:
            v = queue.popleft()
            for c in np.nonzero(adj[v])[0]:
                c = int(c)
                if c not in desc_t:
                    desc_t.add(c)
                    queue.append(c)
        return not bool(pa_t & desc_t)

    @staticmethod
    def _compute_linear_ate(
        adj: NDArray, weights: NDArray, treatment: int, outcome: int
    ) -> float:
        """Total causal effect in a linear SEM."""
        n = adj.shape[0]
        W = np.zeros((n, n), dtype=np.float64)
        for u in range(n):
            for v in range(n):
                if adj[u, v]:
                    W[u, v] = weights[u, v]
        M = np.eye(n) - W.T
        try:
            inv_M = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            return 0.0
        return float(inv_M[treatment, outcome])

    @staticmethod
    def _relevant_subgraph(
        adj: NDArray, treatment: int, outcome: int
    ) -> NDArray:
        """Extract the ancestral subgraph of {treatment, outcome}."""
        n = adj.shape[0]
        anc: set[int] = {treatment, outcome}
        queue = deque([treatment, outcome])
        while queue:
            v = queue.popleft()
            for p in np.nonzero(adj[:, v])[0]:
                p = int(p)
                if p not in anc:
                    anc.add(p)
                    queue.append(p)
        idx = sorted(anc)
        arr = np.array(idx, dtype=int)
        return adj[np.ix_(arr, arr)]

    # -- Frontier generation -------------------------------------------------

    def generate_perturbation_frontier(
        self,
        adj: AdjacencyMatrix,
        k_max: int,
        treatment: int | None = None,
        outcome: int | None = None,
        *,
        weights: NDArray[np.float64] | None = None,
        max_per_level: int = 500,
    ) -> list[PerturbedDAG]:
        """Generate the perturbation frontier up to distance *k_max*.

        Enumerates (or samples) perturbations at each distance and
        optionally classifies their causal impact.

        Parameters
        ----------
        adj : AdjacencyMatrix
            Original DAG.
        k_max : int
            Maximum edit distance.
        treatment, outcome : int | None
            If given, perturbations are classified by impact.
        weights : NDArray | None
            Weight matrix for linear ATE computation.
        max_per_level : int
            Cap per distance level (random sampling if exceeded).

        Returns
        -------
        list[PerturbedDAG]
            Perturbations sorted by edit distance.
        """
        adj = np.asarray(adj, dtype=np.int8)
        k_max = min(k_max, self.max_k)
        all_perturbed: list[PerturbedDAG] = []

        seen: set[bytes] = {adj.tobytes()}
        frontier: list[tuple[NDArray, tuple[StructuralEdit, ...]]] = [
            (adj, ())
        ]

        for depth in range(1, k_max + 1):
            next_frontier: list[tuple[NDArray, tuple[StructuralEdit, ...]]] = []
            for current_adj, current_edits in frontier:
                for edit, new_adj in _valid_single_edits(current_adj):
                    key = new_adj.tobytes()
                    if key not in seen:
                        seen.add(key)
                        new_edits = current_edits + (edit,)
                        next_frontier.append((new_adj, new_edits))

            # Cap if too many
            if len(next_frontier) > max_per_level:
                indices = self._rng.choice(
                    len(next_frontier), max_per_level, replace=False
                )
                next_frontier = [next_frontier[i] for i in indices]

            for new_adj, new_edits in next_frontier:
                pd_item = PerturbedDAG(
                    adjacency=new_adj,
                    edits=new_edits,
                    edit_distance=depth,
                )
                if treatment is not None and outcome is not None:
                    impact = self.classify_impact(
                        adj, pd_item, treatment, outcome, weights=weights
                    )
                    pd_item = PerturbedDAG(
                        adjacency=new_adj,
                        edits=new_edits,
                        edit_distance=depth,
                        impact=impact,
                    )
                all_perturbed.append(pd_item)

            frontier = next_frontier

        return all_perturbed

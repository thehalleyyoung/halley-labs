"""
Do-calculus graph mutilation and interventional query support.

Implements Pearl's do-operator for interventional queries:
  - Graph mutilation (remove incoming edges to intervened variables)
  - CPD modification (replace with delta distributions)
  - Junction-tree reconstruction after mutilation
  - Truncated-factorization computation
  - Caching of mutilated structures for repeated interventional queries
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from .potential_table import PotentialTable
from .clique_tree import CliqueTree, build_junction_tree


# ------------------------------------------------------------------ #
#  Intervention specification
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class Intervention:
    """Specification of a single-variable hard intervention do(X = x)."""

    variable: str
    value: float
    bin_index: Optional[int] = None

    def __repr__(self) -> str:
        return f"do({self.variable}={self.value})"


@dataclass
class InterventionSet:
    """A collection of simultaneous interventions."""

    interventions: Dict[str, Intervention] = field(default_factory=dict)

    def add(self, variable: str, value: float, bin_index: Optional[int] = None) -> None:
        self.interventions[variable] = Intervention(variable, value, bin_index)

    @property
    def variables(self) -> Set[str]:
        return set(self.interventions.keys())

    @property
    def signature(self) -> str:
        items = sorted(
            (v, iv.value) for v, iv in self.interventions.items()
        )
        raw = json.dumps(items, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def __len__(self) -> int:
        return len(self.interventions)

    def __contains__(self, variable: str) -> bool:
        return variable in self.interventions

    def __repr__(self) -> str:
        parts = [str(iv) for iv in self.interventions.values()]
        return f"InterventionSet({', '.join(parts)})"


# ------------------------------------------------------------------ #
#  Mutilated DAG
# ------------------------------------------------------------------ #

@dataclass
class MutilatedDAG:
    """Result of applying do-calculus mutilation to a DAG."""

    original_dag: Dict[str, List[str]]
    mutilated_dag: Dict[str, List[str]]
    intervened_variables: Set[str]
    removed_edges: List[Tuple[str, str]]  # (parent, child) removed
    modified_cpds: Dict[str, PotentialTable]

    @property
    def signature(self) -> str:
        items = sorted(self.intervened_variables)
        return hashlib.sha256(str(items).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ #
#  Do operator
# ------------------------------------------------------------------ #

class DoOperator:
    """Implements do-calculus graph mutilation and interventional inference.

    Parameters
    ----------
    cache_mutilations : bool
        Whether to cache mutilated DAGs and junction trees for reuse.
    """

    def __init__(self, cache_mutilations: bool = True) -> None:
        self.cache_mutilations = cache_mutilations
        self._dag_cache: Dict[str, MutilatedDAG] = {}
        self._tree_cache: Dict[str, CliqueTree] = {}

    # ------------------------------------------------------------------ #
    #  Core API
    # ------------------------------------------------------------------ #

    def apply(
        self,
        dag: Dict[str, List[str]],
        cpds: Dict[str, PotentialTable],
        cardinalities: Dict[str, int],
        intervention: InterventionSet,
    ) -> Tuple[CliqueTree, Dict[str, PotentialTable]]:
        """Apply an interventional set and return a ready-to-calibrate
        junction tree with modified CPDs.

        Parameters
        ----------
        dag : original DAG adjacency list (parent → children).
        cpds : original CPDs keyed by child variable.
        cardinalities : variable → discrete cardinality.
        intervention : the do-operator specification.

        Returns
        -------
        (junction_tree, modified_cpds) ready for message passing.
        """
        sig = intervention.signature
        if self.cache_mutilations and sig in self._dag_cache:
            mutilated = self._dag_cache[sig]
            new_cpds = dict(mutilated.modified_cpds)
        else:
            mutilated_dag, removed = self.mutilate_graph(
                dag, intervention.variables
            )
            new_cpds = self.modify_cpds(cpds, cardinalities, intervention)
            mutilated = MutilatedDAG(
                original_dag=dag,
                mutilated_dag=mutilated_dag,
                intervened_variables=intervention.variables,
                removed_edges=removed,
                modified_cpds=new_cpds,
            )
            if self.cache_mutilations:
                self._dag_cache[sig] = mutilated

        # Build junction tree from mutilated DAG
        if self.cache_mutilations and sig in self._tree_cache:
            tree = self._rebuild_tree(self._tree_cache[sig], cardinalities, new_cpds)
        else:
            tree = build_junction_tree(
                mutilated.mutilated_dag, cardinalities
            )
            if self.cache_mutilations:
                self._tree_cache[sig] = tree

        tree.assign_cpds(new_cpds)
        tree.initialize_potentials()
        tree.initialize_separators()
        return tree, new_cpds

    def mutilate_graph(
        self,
        dag: Dict[str, List[str]],
        variables: Set[str],
    ) -> Tuple[Dict[str, List[str]], List[Tuple[str, str]]]:
        """Remove all incoming edges to the intervened variables.

        Parameters
        ----------
        dag : parent → children adjacency list.
        variables : set of intervened variable names.

        Returns
        -------
        (mutilated_dag, list_of_removed_edges)
        """
        mutilated: Dict[str, List[str]] = {}
        removed: List[Tuple[str, str]] = []

        # Deep-copy the DAG
        all_nodes: Set[str] = set(dag.keys())
        for children in dag.values():
            all_nodes.update(children)

        for node in all_nodes:
            mutilated.setdefault(node, [])

        for parent, children in dag.items():
            for child in children:
                if child in variables:
                    removed.append((parent, child))
                else:
                    mutilated.setdefault(parent, []).append(child)

        return mutilated, removed

    def modify_cpds(
        self,
        cpds: Dict[str, PotentialTable],
        cardinalities: Dict[str, int],
        intervention: InterventionSet,
    ) -> Dict[str, PotentialTable]:
        """Replace CPDs for intervened variables with delta distributions.

        For do(X = x), the CPD of X becomes:
            P(X = x) = 1, P(X = k) = 0 for k ≠ x

        Parameters
        ----------
        cpds : original CPDs.
        cardinalities : variable → cardinality.
        intervention : the intervention set.

        Returns
        -------
        Modified CPD dict (original CPDs are not mutated).
        """
        new_cpds: Dict[str, PotentialTable] = {}
        for var, cpd in cpds.items():
            if var in intervention.interventions:
                iv = intervention.interventions[var]
                new_cpds[var] = self._make_delta_cpd(
                    var, cardinalities[var], iv
                )
            else:
                # Check if any parent of this variable was intervened
                # If so, we need to reduce the CPD scope
                new_cpds[var] = self._reduce_cpd_scope(
                    cpd, var, intervention, cardinalities
                )

        return new_cpds

    def get_truncated_factorization(
        self,
        cpds: Dict[str, PotentialTable],
        intervention: InterventionSet,
        cardinalities: Dict[str, int],
    ) -> PotentialTable:
        """Compute the truncated factorization for an interventional
        distribution.

        P(v₁, ..., vₙ | do(X=x)) = ∏_{i : Vᵢ ∉ X} P(vᵢ | pa(vᵢ))

        This multiplies all CPDs except those of intervened variables
        (whose CPDs are replaced with deltas).
        """
        modified = self.modify_cpds(cpds, cardinalities, intervention)
        tables: List[PotentialTable] = []
        for var, cpd in modified.items():
            tables.append(cpd)

        if not tables:
            return PotentialTable([], {})

        result = tables[0]
        for t in tables[1:]:
            result = result.multiply(t)
        return result

    # ------------------------------------------------------------------ #
    #  Adjustment formula helpers
    # ------------------------------------------------------------------ #

    def compute_adjustment(
        self,
        dag: Dict[str, List[str]],
        treatment: str,
        outcome: str,
    ) -> Optional[Set[str]]:
        """Find a valid adjustment set for estimating the causal effect
        of *treatment* on *outcome* using the back-door criterion.

        Returns the adjustment set or None if no valid set exists.
        """
        # Find parents of treatment (potential confounders)
        parents_of_treatment: Set[str] = set()
        for parent, children in dag.items():
            if treatment in children:
                parents_of_treatment.add(parent)

        # Simple back-door: parents of treatment that are not descendants
        # of treatment
        descendants = self._get_descendants(dag, treatment)
        adjustment = parents_of_treatment - descendants - {treatment, outcome}

        # Verify: the adjustment set must block all back-door paths
        if self._blocks_backdoor_paths(dag, treatment, outcome, adjustment):
            return adjustment
        return None

    def _get_descendants(
        self, dag: Dict[str, List[str]], node: str
    ) -> Set[str]:
        """Get all descendants of a node via BFS."""
        descendants: Set[str] = set()
        queue = list(dag.get(node, []))
        while queue:
            current = queue.pop(0)
            if current not in descendants:
                descendants.add(current)
                queue.extend(dag.get(current, []))
        return descendants

    def _get_ancestors(
        self, dag: Dict[str, List[str]], node: str
    ) -> Set[str]:
        """Get all ancestors of a node."""
        # Build reverse adjacency
        parents: Dict[str, Set[str]] = {}
        for parent, children in dag.items():
            for child in children:
                parents.setdefault(child, set()).add(parent)

        ancestors: Set[str] = set()
        queue = list(parents.get(node, set()))
        while queue:
            current = queue.pop(0)
            if current not in ancestors:
                ancestors.add(current)
                queue.extend(parents.get(current, set()))
        return ancestors

    def _blocks_backdoor_paths(
        self,
        dag: Dict[str, List[str]],
        treatment: str,
        outcome: str,
        adjustment: Set[str],
    ) -> bool:
        """Check whether *adjustment* blocks all back-door paths from
        *treatment* to *outcome*.

        Uses a simplified d-separation check.
        """
        # Build parent map
        parents: Dict[str, Set[str]] = {}
        for parent, children in dag.items():
            for child in children:
                parents.setdefault(child, set()).add(parent)

        # Ancestors of the adjustment set ∪ {treatment, outcome}
        relevant = set(adjustment) | {treatment, outcome}
        for v in list(relevant):
            relevant |= self._get_ancestors(dag, v)

        # BFS on the moral graph excluding treatment→children edges
        # and conditioned on the adjustment set
        blocked = set(adjustment)

        # Check reachability from treatment's parents to outcome
        # in the ancestral graph conditioned on adjustment
        moral_graph: Dict[str, Set[str]] = {v: set() for v in relevant}
        for parent_node, children in dag.items():
            if parent_node not in relevant:
                continue
            for child in children:
                if child not in relevant:
                    continue
                if parent_node == treatment:
                    continue  # back-door: exclude treatment→child edges
                moral_graph[parent_node].add(child)
                moral_graph[child].add(parent_node)

        # Marry co-parents
        for child, pset in parents.items():
            if child not in relevant:
                continue
            plist = [p for p in pset if p in relevant]
            for i in range(len(plist)):
                for j in range(i + 1, len(plist)):
                    moral_graph[plist[i]].add(plist[j])
                    moral_graph[plist[j]].add(plist[i])

        # Check if treatment is reachable from outcome without going
        # through blocked nodes
        visited: Set[str] = set()
        queue = [treatment]
        while queue:
            current = queue.pop(0)
            if current == outcome:
                return False  # path found → not blocked
            if current in visited:
                continue
            visited.add(current)
            if current in blocked and current != treatment:
                continue
            for nb in moral_graph.get(current, set()):
                if nb not in visited:
                    queue.append(nb)
        return True  # no path found → blocked

    # ------------------------------------------------------------------ #
    #  Private helpers
    # ------------------------------------------------------------------ #

    def _make_delta_cpd(
        self, variable: str, cardinality: int, intervention: Intervention
    ) -> PotentialTable:
        """Create a delta-distribution CPD for an intervened variable.

        The CPD has a single variable (no parents after mutilation).
        """
        values = np.zeros(cardinality, dtype=np.float64)
        if intervention.bin_index is not None:
            idx = min(max(intervention.bin_index, 0), cardinality - 1)
            values[idx] = 1.0
        else:
            # Map continuous value to nearest bin (assume uniform grid)
            idx = min(
                int(intervention.value * cardinality),
                cardinality - 1,
            )
            idx = max(idx, 0)
            values[idx] = 1.0

        return PotentialTable(
            [variable],
            {variable: cardinality},
            values,
        )

    def _reduce_cpd_scope(
        self,
        cpd: PotentialTable,
        variable: str,
        intervention: InterventionSet,
        cardinalities: Dict[str, int],
    ) -> PotentialTable:
        """For a non-intervened variable whose parents include intervened
        variables, reduce the CPD by conditioning on the intervention values.
        """
        evidence: Dict[str, int] = {}
        for parent_var in cpd.variables:
            if parent_var == variable:
                continue
            if parent_var in intervention.interventions:
                iv = intervention.interventions[parent_var]
                if iv.bin_index is not None:
                    evidence[parent_var] = iv.bin_index
                else:
                    card = cardinalities.get(parent_var, 2)
                    evidence[parent_var] = min(
                        max(int(iv.value * card), 0), card - 1
                    )

        if evidence:
            return cpd.reduce(evidence)
        return cpd.copy()

    def _rebuild_tree(
        self,
        cached_tree: CliqueTree,
        cardinalities: Dict[str, int],
        cpds: Dict[str, PotentialTable],
    ) -> CliqueTree:
        """Rebuild a junction tree from a cached structure template."""
        tree = CliqueTree(cardinalities)
        for clique in cached_tree.cliques:
            tree.add_clique(clique.variables)
        for idx_a, nbrs in cached_tree._adjacency.items():
            for idx_b in nbrs:
                if idx_a < idx_b:
                    tree.connect(idx_a, idx_b)
        return tree

    # ------------------------------------------------------------------ #
    #  Cache management
    # ------------------------------------------------------------------ #

    def clear_cache(self) -> None:
        """Clear all cached mutilated structures."""
        self._dag_cache.clear()
        self._tree_cache.clear()

    def cached_interventions(self) -> List[str]:
        """Return signatures of cached interventions."""
        return list(self._dag_cache.keys())

    def __repr__(self) -> str:
        return (
            f"DoOperator(cached={len(self._dag_cache)}, "
            f"cache_enabled={self.cache_mutilations})"
        )

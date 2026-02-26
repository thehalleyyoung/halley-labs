"""
Symmetry detection and reduction for explicit-state exploration.

Identifies permutation symmetries in TLA+ CONSTANT sets and computes
canonical state representatives, enabling symmetry-reduced state spaces.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
)

from .graph import TransitionGraph, StateNode, TransitionEdge


# ---------------------------------------------------------------------------
# Permutation representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Permutation:
    """A permutation of elements from a symmetric set, stored as a mapping."""

    mapping: Tuple[Tuple[str, str], ...]

    @classmethod
    def identity(cls, elements: Iterable[str]) -> "Permutation":
        return cls(mapping=tuple((e, e) for e in sorted(elements)))

    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> "Permutation":
        return cls(mapping=tuple(sorted(d.items())))

    def as_dict(self) -> Dict[str, str]:
        return dict(self.mapping)

    def apply(self, element: str) -> str:
        d = self.as_dict()
        return d.get(element, element)

    def compose(self, other: "Permutation") -> "Permutation":
        """Return self ∘ other (apply other first, then self)."""
        d_self = self.as_dict()
        d_other = other.as_dict()
        all_keys = set(d_self.keys()) | set(d_other.keys())
        result = {}
        for k in all_keys:
            intermediate = d_other.get(k, k)
            result[k] = d_self.get(intermediate, intermediate)
        return Permutation.from_dict(result)

    def inverse(self) -> "Permutation":
        d = self.as_dict()
        return Permutation.from_dict({v: k for k, v in d.items()})

    def order(self) -> int:
        """Return the order (smallest n s.t. self^n = identity)."""
        current = self
        identity = Permutation.identity(dict(self.mapping).keys())
        n = 1
        while current != identity:
            current = current.compose(self)
            n += 1
            if n > 1000:
                break
        return n

    def is_identity(self) -> bool:
        return all(k == v for k, v in self.mapping)

    def cycle_notation(self) -> List[List[str]]:
        d = self.as_dict()
        visited: Set[str] = set()
        cycles: List[List[str]] = []
        for start in sorted(d.keys()):
            if start in visited:
                continue
            cycle = []
            current = start
            while current not in visited:
                visited.add(current)
                cycle.append(current)
                current = d.get(current, current)
            if len(cycle) > 1:
                cycles.append(cycle)
        return cycles

    def __repr__(self) -> str:
        cycles = self.cycle_notation()
        if not cycles:
            return "id"
        parts = ["(" + " ".join(c) + ")" for c in cycles]
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Permutation group
# ---------------------------------------------------------------------------

class PermutationGroup:
    """
    A permutation group represented by a set of generators.

    The full group is computed lazily via closure.
    """

    def __init__(self, generators: List[Permutation], elements: Set[str]) -> None:
        self._generators = generators
        self._elements = elements
        self._group: Optional[Set[Permutation]] = None

    @property
    def generators(self) -> List[Permutation]:
        return list(self._generators)

    @property
    def elements(self) -> Set[str]:
        return set(self._elements)

    def group_elements(self) -> Set[Permutation]:
        if self._group is not None:
            return self._group
        self._group = self._compute_closure()
        return self._group

    def _compute_closure(self) -> Set[Permutation]:
        """Compute the full group by iterated closure of generators."""
        group: Set[Permutation] = {Permutation.identity(self._elements)}
        for g in self._generators:
            group.add(g)

        changed = True
        while changed:
            changed = False
            new_elements: Set[Permutation] = set()
            for a in group:
                for b in group:
                    composed = a.compose(b)
                    if composed not in group:
                        new_elements.add(composed)
                        changed = True
                inv = a.inverse()
                if inv not in group:
                    new_elements.add(inv)
                    changed = True
            group |= new_elements
            if len(group) > 10000:
                break
        return group

    def order(self) -> int:
        return len(self.group_elements())

    @classmethod
    def symmetric_group(cls, elements: Set[str]) -> "PermutationGroup":
        """Full symmetric group on *elements*."""
        elems = sorted(elements)
        generators: List[Permutation] = []
        if len(elems) >= 2:
            swap = {elems[0]: elems[1], elems[1]: elems[0]}
            for e in elems[2:]:
                swap[e] = e
            generators.append(Permutation.from_dict(swap))
        if len(elems) >= 3:
            rotation = {}
            for i in range(len(elems)):
                rotation[elems[i]] = elems[(i + 1) % len(elems)]
            generators.append(Permutation.from_dict(rotation))
        return cls(generators, elements)

    @classmethod
    def cyclic_group(cls, elements: Set[str]) -> "PermutationGroup":
        elems = sorted(elements)
        if len(elems) <= 1:
            return cls([], elements)
        rotation = {}
        for i in range(len(elems)):
            rotation[elems[i]] = elems[(i + 1) % len(elems)]
        return cls([Permutation.from_dict(rotation)], elements)


# ---------------------------------------------------------------------------
# Orbit
# ---------------------------------------------------------------------------

@dataclass
class Orbit:
    """An equivalence class of states under a symmetry group."""

    representative: StateNode
    members: Set[str] = field(default_factory=set)

    @property
    def size(self) -> int:
        return len(self.members)

    def __contains__(self, state_hash: str) -> bool:
        return state_hash in self.members

    def __repr__(self) -> str:
        return f"Orbit(rep={self.representative.state_hash[:8]}, size={self.size})"


# ---------------------------------------------------------------------------
# Symmetry detector
# ---------------------------------------------------------------------------

class SymmetryDetector:
    """
    Detects and exploits permutation symmetries in TLA+ state spaces.

    Parameters
    ----------
    symmetric_sets : dict mapping set names to their elements
        e.g. {"Proc": {"p1", "p2", "p3"}, "Val": {"v1", "v2"}}
    variable_types : dict mapping variable names to their type info
        Used to know which variables reference symmetric elements.
    """

    def __init__(
        self,
        symmetric_sets: Dict[str, Set[str]],
        variable_types: Optional[Dict[str, str]] = None,
    ) -> None:
        self._symmetric_sets = {k: set(v) for k, v in symmetric_sets.items()}
        self._variable_types = variable_types or {}
        self._groups: Dict[str, PermutationGroup] = {}
        self._orbit_cache: Dict[str, str] = {}  # state_hash -> canonical_hash

        for name, elems in self._symmetric_sets.items():
            self._groups[name] = PermutationGroup.symmetric_group(elems)

    @property
    def symmetric_set_names(self) -> List[str]:
        return list(self._symmetric_sets.keys())

    def get_group(self, set_name: str) -> Optional[PermutationGroup]:
        return self._groups.get(set_name)

    # -- State permutation -------------------------------------------------

    def apply_permutation_to_state(
        self, state: Dict[str, Any], perm: Permutation
    ) -> Dict[str, Any]:
        """Apply a permutation to all symmetric elements in a state."""
        return self._permute_value(state, perm)

    def _permute_value(self, value: Any, perm: Permutation) -> Any:
        if isinstance(value, str):
            return perm.apply(value)
        if isinstance(value, dict):
            new_dict = {}
            for k, v in value.items():
                new_key = perm.apply(k) if isinstance(k, str) else k
                new_dict[new_key] = self._permute_value(v, perm)
            return new_dict
        if isinstance(value, (list, tuple)):
            result = [self._permute_value(item, perm) for item in value]
            return type(value)(result)
        if isinstance(value, set):
            return {self._permute_value(item, perm) for item in value}
        if isinstance(value, frozenset):
            return frozenset(self._permute_value(item, perm) for item in value)
        return value

    # -- Canonical form ----------------------------------------------------

    def canonical_form(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute the canonical (lexicographically smallest) representative
        of the orbit of *state* under the combined symmetry group.
        """
        all_perms = self._all_combined_permutations()
        candidates = []
        for perm in all_perms:
            permuted = self.apply_permutation_to_state(state, perm)
            serialized = json.dumps(permuted, sort_keys=True, default=str)
            candidates.append((serialized, permuted))
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def canonical_hash(self, state: Dict[str, Any]) -> str:
        canon = self.canonical_form(state)
        serialized = json.dumps(canon, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def _all_combined_permutations(self) -> List[Permutation]:
        """Enumerate all permutations from the product of all symmetry groups."""
        group_elements_list: List[List[Permutation]] = []
        for name in sorted(self._symmetric_sets.keys()):
            group = self._groups[name]
            group_elements_list.append(list(group.group_elements()))

        if not group_elements_list:
            return [Permutation(mapping=())]

        combined: List[Permutation] = []
        for combo in itertools.product(*group_elements_list):
            merged = Permutation(mapping=())
            for perm in combo:
                merged = merged.compose(perm)
            combined.append(merged)
        return combined

    # -- Orbit computation -------------------------------------------------

    def compute_orbit(
        self, state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Compute the full orbit of *state*."""
        orbit_set: Set[str] = set()
        orbit_states: List[Dict[str, Any]] = []
        for perm in self._all_combined_permutations():
            permuted = self.apply_permutation_to_state(state, perm)
            key = json.dumps(permuted, sort_keys=True, default=str)
            if key not in orbit_set:
                orbit_set.add(key)
                orbit_states.append(permuted)
        return orbit_states

    def compute_orbits_from_graph(
        self, graph: TransitionGraph
    ) -> List[Orbit]:
        """Partition all states in the graph into orbits."""
        assigned: Set[str] = set()
        orbits: List[Orbit] = []

        for node in graph.all_states():
            if node.state_hash in assigned:
                continue
            orbit_states = self.compute_orbit(node.full_state)
            members: Set[str] = set()
            for os in orbit_states:
                oh = hashlib.sha256(
                    json.dumps(os, sort_keys=True, default=str).encode()
                ).hexdigest()
                if graph.has_state(oh):
                    members.add(oh)
                    assigned.add(oh)

            if not members:
                members.add(node.state_hash)
                assigned.add(node.state_hash)

            canon = self.canonical_form(node.full_state)
            canon_hash = hashlib.sha256(
                json.dumps(canon, sort_keys=True, default=str).encode()
            ).hexdigest()
            rep_node = graph.get_state(canon_hash)
            if rep_node is None:
                rep_node = node

            orbits.append(Orbit(representative=rep_node, members=members))
        return orbits

    def select_representative(
        self, orbit: Orbit, graph: TransitionGraph
    ) -> StateNode:
        return orbit.representative

    # -- Symmetry-reduced exploration --------------------------------------

    def reduce_graph(self, graph: TransitionGraph) -> TransitionGraph:
        """
        Build a symmetry-reduced graph: one representative per orbit
        with canonical transitions.
        """
        orbits = self.compute_orbits_from_graph(graph)
        hash_to_rep: Dict[str, str] = {}
        for orbit in orbits:
            rep_hash = orbit.representative.state_hash
            for member in orbit.members:
                hash_to_rep[member] = rep_hash

        reduced = TransitionGraph()
        for orbit in orbits:
            rep = orbit.representative
            is_init = any(
                m in graph.initial_states for m in orbit.members
            )
            reduced.add_state(rep, is_initial=is_init)

        for edge in graph.get_all_edges():
            src_rep = hash_to_rep.get(edge.source_hash, edge.source_hash)
            tgt_rep = hash_to_rep.get(edge.target_hash, edge.target_hash)
            if reduced.has_state(src_rep) and reduced.has_state(tgt_rep):
                new_edge = TransitionEdge(
                    action_label=edge.action_label,
                    source_hash=src_rep,
                    target_hash=tgt_rep,
                    is_stuttering=(src_rep == tgt_rep),
                )
                reduced.add_transition(new_edge)
        return reduced

    def is_symmetric_pair(
        self, state_a: Dict[str, Any], state_b: Dict[str, Any]
    ) -> bool:
        """Check whether two states are in the same orbit."""
        return self.canonical_hash(state_a) == self.canonical_hash(state_b)

    # -- Automorphism detection on graph -----------------------------------

    def detect_graph_automorphisms(
        self, graph: TransitionGraph, max_automorphisms: int = 100
    ) -> List[Dict[str, str]]:
        """
        Heuristic automorphism detection on the labelled transition graph.
        Returns a list of node-level mappings that preserve adjacency.
        """
        nodes = sorted(graph.all_state_hashes())
        if len(nodes) > 50:
            return []

        degree_classes: Dict[Tuple[int, int], List[str]] = defaultdict(list)
        for h in nodes:
            key = (graph.in_degree(h), graph.out_degree(h))
            degree_classes[key].append(h)

        automorphisms: List[Dict[str, str]] = []
        class_lists = [sorted(v) for v in degree_classes.values()]
        perm_iters = [itertools.permutations(cl) for cl in class_lists]

        for combo in itertools.product(*perm_iters):
            mapping: Dict[str, str] = {}
            for original_list, perm in zip(class_lists, combo):
                for orig, img in zip(original_list, perm):
                    mapping[orig] = img

            if self._is_graph_automorphism(graph, mapping):
                automorphisms.append(mapping)
                if len(automorphisms) >= max_automorphisms:
                    return automorphisms
        return automorphisms

    def _is_graph_automorphism(
        self, graph: TransitionGraph, mapping: Dict[str, str]
    ) -> bool:
        for edge in graph.get_all_edges():
            mapped_src = mapping.get(edge.source_hash, edge.source_hash)
            mapped_tgt = mapping.get(edge.target_hash, edge.target_hash)
            mapped_edges = graph.get_edges(mapped_src, mapped_tgt)
            if not any(e.action_label == edge.action_label for e in mapped_edges):
                return False
        return True

    # -- Statistics --------------------------------------------------------

    def reduction_ratio(self, graph: TransitionGraph) -> float:
        """Ratio of reduced graph size to original graph size."""
        orbits = self.compute_orbits_from_graph(graph)
        if graph.num_states == 0:
            return 1.0
        return len(orbits) / graph.num_states

    def summary(self, graph: TransitionGraph) -> Dict[str, Any]:
        orbits = self.compute_orbits_from_graph(graph)
        orbit_sizes = [o.size for o in orbits]
        return {
            "num_symmetric_sets": len(self._symmetric_sets),
            "symmetric_set_sizes": {
                k: len(v) for k, v in self._symmetric_sets.items()
            },
            "num_orbits": len(orbits),
            "avg_orbit_size": (
                sum(orbit_sizes) / len(orbit_sizes) if orbit_sizes else 0
            ),
            "max_orbit_size": max(orbit_sizes, default=0),
            "reduction_ratio": self.reduction_ratio(graph),
        }

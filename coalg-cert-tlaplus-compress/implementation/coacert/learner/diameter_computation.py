"""
Exact diameter computation for hypothesis coalgebras.

Provides exact diameter computation via all-pairs BFS, incremental
diameter maintenance as the hypothesis grows, and distinguishing-depth
computation for partitions.

The diameter of a hypothesis H is the length of the longest shortest path
between any two reachable states.  This is the key parameter in the W-method
bound: conformance testing to depth k >= diam(H) + (m - n + 1) is complete.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Deque,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
)

from .equivalence_oracle import HypothesisInterface

logger = logging.getLogger(__name__)


@dataclass
class DiameterCertificate:
    """Certificate proving the computed diameter of a hypothesis.

    Contains the diameter value, the witness pair of states that achieves
    it, and the all-pairs distance matrix summary.
    """

    diameter: int = 0
    witness_source: str = ""
    witness_target: str = ""
    witness_path_length: int = 0
    state_count: int = 0
    reachable_pairs: int = 0
    total_pairs: int = 0
    computation_method: str = "all_pairs_bfs"
    eccentricities: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diameter": self.diameter,
            "witness_source": self.witness_source,
            "witness_target": self.witness_target,
            "witness_path_length": self.witness_path_length,
            "state_count": self.state_count,
            "reachable_pairs": self.reachable_pairs,
            "total_pairs": self.total_pairs,
            "computation_method": self.computation_method,
            "eccentricities": self.eccentricities,
        }

    def summary(self) -> str:
        return (
            f"DiameterCertificate(d={self.diameter}, "
            f"witness={self.witness_source}→{self.witness_target}, "
            f"states={self.state_count})"
        )


class ExactDiameterComputer:
    """Compute the exact diameter of a hypothesis coalgebra via all-pairs BFS.

    The diameter is max over all reachable pairs (s, t) of the shortest
    path length from s to t.  Complexity: O(n · (n + m)) where n = |states|
    and m = |transitions|.

    Parameters
    ----------
    hypothesis : HypothesisInterface
        The hypothesis coalgebra to compute the diameter of.
    """

    def __init__(self, hypothesis: Any) -> None:
        self._hypothesis = hypothesis
        self._states: List[str] = []
        self._actions: List[str] = []
        self._distances: Dict[str, Dict[str, int]] = {}
        self._diameter: Optional[int] = None
        self._certificate: Optional[DiameterCertificate] = None

    def compute(self) -> DiameterCertificate:
        """Compute the exact diameter and return a certificate."""
        self._states = sorted(
            self._hypothesis.states()
            if callable(getattr(self._hypothesis, "states", None))
            else []
        )
        self._actions = sorted(
            self._hypothesis.actions()
            if callable(getattr(self._hypothesis, "actions", None))
            else []
        )

        if not self._states:
            self._certificate = DiameterCertificate(
                diameter=0,
                computation_method="all_pairs_bfs",
            )
            return self._certificate

        # All-pairs BFS
        diameter = 0
        witness_src = ""
        witness_tgt = ""
        eccentricities: Dict[str, int] = {}
        reachable_pairs = 0

        for source in self._states:
            dist = self._bfs_from(source)
            self._distances[source] = dist
            reachable_pairs += len(dist) - 1  # exclude self

            if dist:
                max_dist = max(dist.values())
                eccentricities[source] = max_dist
                if max_dist > diameter:
                    diameter = max_dist
                    witness_src = source
                    # Find the farthest state
                    for s, d in dist.items():
                        if d == max_dist:
                            witness_tgt = s
                            break

        n = len(self._states)
        self._diameter = diameter
        self._certificate = DiameterCertificate(
            diameter=diameter,
            witness_source=witness_src,
            witness_target=witness_tgt,
            witness_path_length=diameter,
            state_count=n,
            reachable_pairs=reachable_pairs,
            total_pairs=n * (n - 1) if n > 1 else 0,
            computation_method="all_pairs_bfs",
            eccentricities=eccentricities,
        )

        logger.info(
            "Computed diameter %d for %d-state hypothesis", diameter, n
        )
        return self._certificate

    def _bfs_from(self, source: str) -> Dict[str, int]:
        """BFS from a single source, returning distance map."""
        dist: Dict[str, int] = {source: 0}
        queue: Deque[str] = deque([source])

        while queue:
            s = queue.popleft()
            for act in self._actions:
                t = (
                    self._hypothesis.transition(s, act)
                    if callable(
                        getattr(self._hypothesis, "transition", None)
                    )
                    else None
                )
                if t is not None and t not in dist:
                    dist[t] = dist[s] + 1
                    queue.append(t)

        return dist

    def distance(self, s: str, t: str) -> Optional[int]:
        """Return the shortest path distance from s to t, or None."""
        if not self._distances:
            self.compute()
        src_dist = self._distances.get(s)
        if src_dist is None:
            return None
        return src_dist.get(t)

    @property
    def diameter(self) -> int:
        """Return the computed diameter (compute if necessary)."""
        if self._diameter is None:
            self.compute()
        return self._diameter  # type: ignore[return-value]

    def compute_with_certificate(self) -> Tuple[int, DiameterCertificate]:
        """Compute diameter and return both the value and its certificate.

        Convenience method that returns a (diameter, certificate) tuple.
        """
        cert = self.compute()
        return cert.diameter, cert

    @property
    def certificate(self) -> Optional[DiameterCertificate]:
        return self._certificate


class IncrementalDiameter:
    """Maintain diameter incrementally as the hypothesis grows.

    Instead of recomputing all-pairs BFS from scratch when a new state
    is added, this class tracks the current diameter and only performs
    BFS from/to the new state.

    This is sound: the diameter can only increase or stay the same when
    states are added (monotonicity of diameter under state addition to
    a connected automaton).
    """

    def __init__(self) -> None:
        self._current_diameter: int = 0
        self._state_count: int = 0
        self._distances: Dict[str, Dict[str, int]] = {}
        self._known_states: Set[str] = set()

    def update(self, hypothesis: Any) -> int:
        """Update the diameter given the current hypothesis.

        If no new states have been added, returns the cached diameter.
        If new states exist, runs BFS from/to them and updates.

        Returns the current diameter.
        """
        states = set(
            hypothesis.states()
            if callable(getattr(hypothesis, "states", None))
            else []
        )
        actions = sorted(
            hypothesis.actions()
            if callable(getattr(hypothesis, "actions", None))
            else []
        )

        new_states = states - self._known_states
        if not new_states:
            return self._current_diameter

        # For new states, run BFS from them
        for source in new_states:
            dist = self._bfs_from(hypothesis, source, actions)
            self._distances[source] = dist
            if dist:
                max_d = max(dist.values())
                self._current_diameter = max(self._current_diameter, max_d)

        # Also run BFS from existing states to see if new states are
        # reachable at greater distance. Full recompute if many new states.
        if len(new_states) > len(self._known_states) // 2 + 1:
            # Substantial change: full recompute
            computer = ExactDiameterComputer(hypothesis)
            cert = computer.compute()
            self._current_diameter = cert.diameter
            self._distances = computer._distances
        else:
            # Incremental: recompute only from existing states
            for source in self._known_states:
                dist = self._bfs_from(hypothesis, source, actions)
                self._distances[source] = dist
                if dist:
                    max_d = max(dist.values())
                    self._current_diameter = max(self._current_diameter, max_d)

        self._known_states = states
        self._state_count = len(states)
        return self._current_diameter

    def _bfs_from(
        self, hypothesis: Any, source: str, actions: List[str]
    ) -> Dict[str, int]:
        """BFS from source."""
        dist: Dict[str, int] = {source: 0}
        queue: Deque[str] = deque([source])
        while queue:
            s = queue.popleft()
            for act in actions:
                t = (
                    hypothesis.transition(s, act)
                    if callable(
                        getattr(hypothesis, "transition", None)
                    )
                    else None
                )
                if t is not None and t not in dist:
                    dist[t] = dist[s] + 1
                    queue.append(t)
        return dist

    @property
    def current_diameter(self) -> int:
        return self._current_diameter

    @property
    def state_count(self) -> int:
        return self._state_count


def compute_distinguishing_depth(
    hypothesis: Any,
    partition: Optional[List[FrozenSet[str]]] = None,
) -> int:
    """Compute the distinguishing depth of a hypothesis partition.

    The distinguishing depth is the maximum depth needed to distinguish
    any two classes in the partition.  For a minimal hypothesis, this
    equals the diameter.

    Parameters
    ----------
    hypothesis : HypothesisInterface
        The hypothesis coalgebra.
    partition : list of frozenset, optional
        The partition of states into equivalence classes.
        If None, each state is its own class.

    Returns
    -------
    int
        The maximum depth needed to distinguish any two classes.
    """
    states = sorted(
        hypothesis.states()
        if callable(getattr(hypothesis, "states", None))
        else []
    )
    actions = sorted(
        hypothesis.actions()
        if callable(getattr(hypothesis, "actions", None))
        else []
    )

    if not states:
        return 0

    # Build partition if not given
    if partition is None:
        partition = [frozenset([s]) for s in states]

    # For each pair of classes, find the depth at which they are distinguished
    max_depth = 0
    representatives = [next(iter(cls)) for cls in partition if cls]

    for i, rep1 in enumerate(representatives):
        for rep2 in representatives[i + 1:]:
            depth = _find_distinguishing_depth(
                hypothesis, rep1, rep2, actions, len(states)
            )
            if depth is not None:
                max_depth = max(max_depth, depth)

    return max_depth


def _find_distinguishing_depth(
    hypothesis: Any,
    s1: str,
    s2: str,
    actions: List[str],
    max_search: int,
) -> Optional[int]:
    """Find the minimum depth at which s1 and s2 are distinguished."""
    queue: Deque[Tuple[str, str, int]] = deque([(s1, s2, 0)])
    visited: Set[Tuple[str, str]] = set()

    while queue:
        q1, q2, depth = queue.popleft()
        if (q1, q2) in visited:
            continue
        visited.add((q1, q2))

        obs1 = (
            hypothesis.observation_at(q1)
            if callable(getattr(hypothesis, "observation_at", None))
            else None
        )
        obs2 = (
            hypothesis.observation_at(q2)
            if callable(getattr(hypothesis, "observation_at", None))
            else None
        )
        if obs1 != obs2:
            return depth

        if depth >= max_search:
            continue

        for act in actions:
            t1 = (
                hypothesis.transition(q1, act)
                if callable(getattr(hypothesis, "transition", None))
                else None
            )
            t2 = (
                hypothesis.transition(q2, act)
                if callable(getattr(hypothesis, "transition", None))
                else None
            )
            if t1 is not None and t2 is not None and (t1, t2) not in visited:
                queue.append((t1, t2, depth + 1))

    return None

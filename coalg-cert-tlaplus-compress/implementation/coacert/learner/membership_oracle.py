"""
Membership oracle for L*-style coalgebraic learning.

Given an access sequence (list of actions), the membership oracle executes
the sequence on the concrete transition system and returns the F-behaviour
at the reached state(s).  Because the system may be nondeterministic, the
oracle tracks *all* states reachable via the given action sequence and
returns the combined observation.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from .observation_table import (
    AccessSequence,
    Observation,
    Suffix,
    _make_observation,
    _merge_observations,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MembershipResult:
    """Result of a single membership query."""

    sequence: Tuple[str, ...]
    observation: Optional[Observation]
    reached_states: FrozenSet[str]
    timed_out: bool = False
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.observation is not None and not self.timed_out


# ---------------------------------------------------------------------------
# Query statistics
# ---------------------------------------------------------------------------

@dataclass
class OracleStats:
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    batch_queries: int = 0
    total_steps_executed: int = 0
    timeouts: int = 0
    errors: int = 0
    total_time_seconds: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def summary(self) -> str:
        return (
            f"MQ stats: {self.total_queries} queries "
            f"({self.cache_hits} cached, {self.cache_misses} executed), "
            f"{self.total_steps_executed} steps, "
            f"{self.timeouts} timeouts, {self.total_time_seconds:.2f}s"
        )


# ---------------------------------------------------------------------------
# Concrete system interface (protocol)
# ---------------------------------------------------------------------------

class ConcreteSystemInterface:
    """Abstract interface for querying a concrete transition system.

    Subclass this and implement the methods, or pass callables to
    ``MembershipOracle`` directly.
    """

    def initial_states(self) -> Set[str]:
        raise NotImplementedError

    def successors(self, state: str, action: str) -> Set[str]:
        raise NotImplementedError

    def get_propositions(self, state: str) -> FrozenSet[str]:
        raise NotImplementedError

    def get_successor_map(self, state: str) -> Dict[str, FrozenSet[str]]:
        raise NotImplementedError

    def available_actions(self, state: str) -> Set[str]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Membership oracle
# ---------------------------------------------------------------------------

class MembershipOracle:
    """Execute membership queries on a concrete transition system.

    Parameters
    ----------
    system : ConcreteSystemInterface or None
        The concrete system to query.  If None, callback functions must
        be provided.
    initial_states_fn : callable, optional
        Returns the set of initial states.
    successors_fn : callable, optional
        Given (state, action), returns successor states.
    propositions_fn : callable, optional
        Given a state, returns its atomic propositions.
    successor_map_fn : callable, optional
        Given a state, returns ``{action: frozenset_of_targets}``.
    timeout : float
        Maximum seconds per query before declaring a timeout.
    cache_enabled : bool
        Whether to cache query results.
    """

    def __init__(
        self,
        system: Optional[ConcreteSystemInterface] = None,
        *,
        initial_states_fn: Optional[Callable[[], Set[str]]] = None,
        successors_fn: Optional[Callable[[str, str], Set[str]]] = None,
        propositions_fn: Optional[Callable[[str], FrozenSet[str]]] = None,
        successor_map_fn: Optional[
            Callable[[str], Dict[str, FrozenSet[str]]]
        ] = None,
        timeout: float = 30.0,
        cache_enabled: bool = True,
    ) -> None:
        self._system = system
        self._initial_states_fn = initial_states_fn
        self._successors_fn = successors_fn
        self._propositions_fn = propositions_fn
        self._successor_map_fn = successor_map_fn
        self._timeout = timeout
        self._cache_enabled = cache_enabled

        self._cache: Dict[Tuple[str, ...], MembershipResult] = {}
        self._stats = OracleStats()

    # -- system access helpers ----------------------------------------------

    def _initial_states(self) -> Set[str]:
        if self._initial_states_fn is not None:
            return self._initial_states_fn()
        if self._system is not None:
            return self._system.initial_states()
        raise RuntimeError("No system or initial_states_fn provided")

    def _successors(self, state: str, action: str) -> Set[str]:
        if self._successors_fn is not None:
            return self._successors_fn(state, action)
        if self._system is not None:
            return self._system.successors(state, action)
        raise RuntimeError("No system or successors_fn provided")

    def _propositions(self, state: str) -> FrozenSet[str]:
        if self._propositions_fn is not None:
            return self._propositions_fn(state)
        if self._system is not None:
            return self._system.get_propositions(state)
        raise RuntimeError("No system or propositions_fn provided")

    def _successor_map(self, state: str) -> Dict[str, FrozenSet[str]]:
        if self._successor_map_fn is not None:
            return self._successor_map_fn(state)
        if self._system is not None:
            return self._system.get_successor_map(state)
        raise RuntimeError("No system or successor_map_fn provided")

    # -- core query execution -----------------------------------------------

    def _execute_sequence(
        self, sequence: Tuple[str, ...]
    ) -> Tuple[Set[str], bool]:
        """Walk the sequence from initial states, tracking all reachable
        states.  Returns ``(reached, timed_out)``."""
        t0 = time.monotonic()
        current_states = set(self._initial_states())
        steps = 0

        for action in sequence:
            if time.monotonic() - t0 > self._timeout:
                return current_states, True
            next_states: Set[str] = set()
            for s in current_states:
                next_states |= self._successors(s, action)
            if not next_states:
                return set(), False
            current_states = next_states
            steps += 1

        self._stats.total_steps_executed += steps
        return current_states, False

    def _observe_states(
        self, states: Set[str]
    ) -> Observation:
        """Collect the F-behaviour at a set of states."""
        obs: Observation = frozenset()
        for s in states:
            props = self._propositions(s)
            succ_map = self._successor_map(s)
            atom = _make_observation(props, succ_map)
            obs = _merge_observations(obs, atom)
        return obs

    # -- public query interface ---------------------------------------------

    def query(
        self,
        access_sequence: AccessSequence,
        suffix: Suffix = (),
    ) -> MembershipResult:
        """Execute a membership query for ``access_sequence · suffix``.

        Returns the F-behaviour at the states reached after executing the
        full concatenated sequence.
        """
        full_sequence = access_sequence + suffix
        self._stats.total_queries += 1

        # Cache lookup
        if self._cache_enabled and full_sequence in self._cache:
            self._stats.cache_hits += 1
            return self._cache[full_sequence]

        self._stats.cache_misses += 1
        t0 = time.monotonic()

        try:
            reached, timed_out = self._execute_sequence(full_sequence)
            if timed_out:
                self._stats.timeouts += 1
                result = MembershipResult(
                    sequence=full_sequence,
                    observation=None,
                    reached_states=frozenset(reached),
                    timed_out=True,
                )
            elif not reached:
                result = MembershipResult(
                    sequence=full_sequence,
                    observation=frozenset(),
                    reached_states=frozenset(),
                )
            else:
                obs = self._observe_states(reached)
                result = MembershipResult(
                    sequence=full_sequence,
                    observation=obs,
                    reached_states=frozenset(reached),
                )
        except Exception as exc:
            self._stats.errors += 1
            result = MembershipResult(
                sequence=full_sequence,
                observation=None,
                reached_states=frozenset(),
                error=str(exc),
            )

        elapsed = time.monotonic() - t0
        self._stats.total_time_seconds += elapsed

        if self._cache_enabled and result.success:
            self._cache[full_sequence] = result

        return result

    def query_observation(
        self,
        access_sequence: AccessSequence,
        suffix: Suffix = (),
    ) -> Optional[Observation]:
        """Convenience: return only the observation (or None on failure)."""
        return self.query(access_sequence, suffix).observation

    # -- batch queries ------------------------------------------------------

    def batch_query(
        self,
        queries: Sequence[Tuple[AccessSequence, Suffix]],
    ) -> List[MembershipResult]:
        """Execute a batch of membership queries.

        Queries are grouped by shared prefix to amortise re-execution.
        """
        self._stats.batch_queries += 1

        # Sort by full sequence to help cache adjacency
        sorted_queries = sorted(queries, key=lambda q: q[0] + q[1])
        results: List[MembershipResult] = [
            MembershipResult((), None, frozenset())
        ] * len(queries)

        index_map = {
            (q[0], q[1]): i for i, q in enumerate(queries)
        }

        for access_seq, suffix in sorted_queries:
            result = self.query(access_seq, suffix)
            idx = index_map.get((access_seq, suffix))
            if idx is not None:
                results[idx] = result

        return results

    def fill_table_cells(
        self,
        table: Any,  # ObservationTable — avoid circular import at type level
    ) -> int:
        """Fill all unfilled cells in the observation table. Returns count."""
        unfilled = table.unfilled_cells()
        if not unfilled:
            return 0

        results = self.batch_query(unfilled)
        filled_count = 0
        for (seq, suffix), result in zip(unfilled, results):
            if result.success and result.observation is not None:
                table.set_cell(seq, suffix, result.observation)
                filled_count += 1
        return filled_count

    # -- cache management ---------------------------------------------------

    def clear_cache(self) -> int:
        """Clear the cache.  Returns number of entries removed."""
        n = len(self._cache)
        self._cache.clear()
        return n

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def invalidate_prefix(self, prefix: Tuple[str, ...]) -> int:
        """Remove cache entries whose sequence starts with *prefix*."""
        to_remove = [
            k for k in self._cache if k[: len(prefix)] == prefix
        ]
        for k in to_remove:
            del self._cache[k]
        return len(to_remove)

    # -- statistics ---------------------------------------------------------

    @property
    def stats(self) -> OracleStats:
        return self._stats

    def reset_stats(self) -> None:
        self._stats = OracleStats()

    # -- semantic engine integration ----------------------------------------

    @classmethod
    def from_transition_graph(
        cls,
        graph: Any,
        timeout: float = 30.0,
        cache_enabled: bool = True,
    ) -> "MembershipOracle":
        """Create an oracle backed by a ``TransitionGraph``."""

        def initial_fn() -> Set[str]:
            return set(graph.initial_states)

        def succ_fn(state: str, action: str) -> Set[str]:
            return graph.get_successors(state, action)

        def props_fn(state: str) -> FrozenSet[str]:
            return frozenset(graph.get_labels(state))

        def succ_map_fn(state: str) -> Dict[str, FrozenSet[str]]:
            result: Dict[str, FrozenSet[str]] = {}
            transitions = graph.transitions.get(state, {})
            for act, targets in transitions.items():
                result[act] = frozenset(targets)
            return result

        return cls(
            initial_states_fn=initial_fn,
            successors_fn=succ_fn,
            propositions_fn=props_fn,
            successor_map_fn=succ_map_fn,
            timeout=timeout,
            cache_enabled=cache_enabled,
        )

    def __repr__(self) -> str:
        return (
            f"MembershipOracle(queries={self._stats.total_queries}, "
            f"cache={self.cache_size})"
        )

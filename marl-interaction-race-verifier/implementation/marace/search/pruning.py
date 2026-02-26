"""AI-guided pruning using abstract interpretation for MARACE.

Provides a hierarchy of pruning strategies that reduce the schedule
exploration space during multi-agent race verification.  Each pruner
implements a different heuristic—safety-margin analysis over zonotopes,
happens-before consistency checking, symmetry reduction, and dominance
elimination—and they can be composed via :class:`CompositePruner`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from marace.abstract.zonotope import Zonotope
from marace.hb.hb_graph import HBGraph, HBRelation

logger = logging.getLogger(__name__)

__all__ = [
    "PruningStatistics",
    "AbstractPruner",
    "SafetyMarginPruner",
    "HBConsistencyPruner",
    "SymmetryPruner",
    "DominancePruner",
    "CompositePruner",
]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class PruningStatistics:
    """Aggregate bookkeeping for pruning decisions.

    Tracks how many nodes were visited and pruned, broken down by the
    pruner method that triggered each prune.
    """

    total_nodes_visited: int = 0
    nodes_pruned: int = 0
    pruned_by_method: Dict[str, int] = field(default_factory=dict)

    # -- derived metrics -----------------------------------------------------

    @property
    def pruning_ratio(self) -> float:
        """Fraction of visited nodes that were pruned (0.0–1.0)."""
        if self.total_nodes_visited == 0:
            return 0.0
        return self.nodes_pruned / self.total_nodes_visited

    # -- combinators ----------------------------------------------------------

    def merge(self, other: PruningStatistics) -> PruningStatistics:
        """Return a *new* ``PruningStatistics`` that is the sum of *self*
        and *other*.

        Parameters
        ----------
        other:
            Statistics object to merge with this one.

        Returns
        -------
        PruningStatistics
            A fresh instance whose counters are the element-wise sum of
            both operands.
        """
        merged_by_method: Dict[str, int] = dict(self.pruned_by_method)
        for method, count in other.pruned_by_method.items():
            merged_by_method[method] = merged_by_method.get(method, 0) + count
        return PruningStatistics(
            total_nodes_visited=self.total_nodes_visited + other.total_nodes_visited,
            nodes_pruned=self.nodes_pruned + other.nodes_pruned,
            pruned_by_method=merged_by_method,
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"PruningStatistics(visited={self.total_nodes_visited}, "
            f"pruned={self.nodes_pruned}, ratio={self.pruning_ratio:.3f}, "
            f"by_method={self.pruned_by_method})"
        )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class AbstractPruner(ABC):
    """Base class for every pruning strategy.

    Subclasses must implement :meth:`can_prune` and the :attr:`name`
    property.  The convenience wrapper :meth:`try_prune` handles
    statistics bookkeeping so that callers need not do it manually.
    """

    def __init__(self) -> None:
        self._stats = PruningStatistics()

    # -- abstract interface ---------------------------------------------------

    @abstractmethod
    def can_prune(self, schedule: List[str], state: np.ndarray) -> bool:
        """Decide whether the search branch rooted at *schedule* with
        abstract *state* can be safely pruned.

        Parameters
        ----------
        schedule:
            Ordered list of event identifiers explored so far.
        state:
            Concrete or representative state vector at this point in
            the exploration.

        Returns
        -------
        bool
            ``True`` if this branch can be pruned without losing
            reachable violations.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for this pruner."""

    # -- statistics -----------------------------------------------------------

    @property
    def stats(self) -> PruningStatistics:
        """Read-only view of the running statistics."""
        return self._stats

    # -- convenience wrapper --------------------------------------------------

    def try_prune(self, schedule: List[str], state: np.ndarray) -> bool:
        """Attempt to prune and update statistics accordingly.

        This method increments :pyattr:`stats.total_nodes_visited` on
        every call and :pyattr:`stats.nodes_pruned` when the branch is
        pruned.

        Parameters
        ----------
        schedule:
            Ordered list of event identifiers explored so far.
        state:
            Concrete or representative state vector.

        Returns
        -------
        bool
            ``True`` if the branch was pruned.
        """
        self._stats.total_nodes_visited += 1
        pruned = self.can_prune(schedule, state)
        if pruned:
            self._stats.nodes_pruned += 1
            self._stats.pruned_by_method[self.name] = (
                self._stats.pruned_by_method.get(self.name, 0) + 1
            )
            logger.debug(
                "Pruner '%s' pruned schedule of length %d",
                self.name,
                len(schedule),
            )
        return pruned


# ---------------------------------------------------------------------------
# Concrete pruners
# ---------------------------------------------------------------------------


class SafetyMarginPruner(AbstractPruner):
    """Prune branches whose safety margin exceeds a threshold.

    The *safety margin* is defined as the minimum value of the linear
    functional ``safety_direction^T x`` over the zonotope that
    over-approximates reachable states.  When this minimum is above
    ``margin_threshold`` the entire zonotope is on the safe side and
    no violation can be reached from this branch.

    Parameters
    ----------
    margin_threshold:
        Scalar threshold.  A branch is pruned when the computed margin
        strictly exceeds this value.
    safety_direction:
        Direction vector used to evaluate the margin.  Its
        dimensionality must match the state space.
    state_zonotope:
        Optional initial zonotope.  May be set (or replaced) later via
        :pyattr:`state_zonotope`.
    """

    def __init__(
        self,
        margin_threshold: float,
        safety_direction: np.ndarray,
        state_zonotope: Optional[Zonotope] = None,
    ) -> None:
        super().__init__()
        if margin_threshold < 0.0:
            raise ValueError(
                f"margin_threshold must be non-negative, got {margin_threshold}"
            )
        self.margin_threshold: float = margin_threshold
        self.safety_direction: np.ndarray = np.asarray(
            safety_direction, dtype=np.float64
        )
        self.state_zonotope: Optional[Zonotope] = state_zonotope

    @property
    def name(self) -> str:  # noqa: D401
        return "SafetyMarginPruner"

    def can_prune(self, schedule: List[str], state: np.ndarray) -> bool:
        """Return ``True`` when the minimum safety margin over the
        zonotope exceeds the threshold.

        If no zonotope has been configured the branch cannot be pruned
        and the method returns ``False``.
        """
        if self.state_zonotope is None:
            return False

        # The margin is the *minimum* of the safety functional over the
        # zonotope.  ``minimize`` returns (min_value, witness_point).
        try:
            min_value, _ = self.state_zonotope.minimize(self.safety_direction)
        except Exception:
            logger.warning(
                "SafetyMarginPruner: zonotope.minimize raised; skipping prune",
                exc_info=True,
            )
            return False

        return float(min_value) > self.margin_threshold


class HBConsistencyPruner(AbstractPruner):
    """Prune schedules that violate happens-before constraints.

    A schedule ``[e0, e1, …, ek]`` is inconsistent if some event
    ``ei`` appears *before* ``ej`` in the list but the happens-before
    graph requires ``ej → ei`` (i.e. ``ej`` must happen before ``ei``).

    Parameters
    ----------
    hb_graph:
        The happens-before graph encoding causal ordering constraints
        among events.
    """

    def __init__(self, hb_graph: HBGraph) -> None:
        super().__init__()
        self.hb_graph: HBGraph = hb_graph

    @property
    def name(self) -> str:  # noqa: D401
        return "HBConsistencyPruner"

    def can_prune(self, schedule: List[str], state: np.ndarray) -> bool:
        """Return ``True`` if *schedule* violates any happens-before
        edge in the graph.

        The check runs in O(n²) worst-case where *n* = ``len(schedule)``
        but exits early on the first violation found.
        """
        if not schedule:
            return False

        # Build a position index: event -> first position in schedule.
        position: Dict[str, int] = {}
        for idx, event in enumerate(schedule):
            if event not in position:
                position[event] = idx

        # For every pair (earlier_in_schedule, later_in_schedule) verify
        # that the HB graph does not require the opposite ordering.
        for i in range(len(schedule)):
            for j in range(i + 1, len(schedule)):
                e_i = schedule[i]
                e_j = schedule[j]
                relation = self.hb_graph.query_hb(e_i, e_j)
                if relation is HBRelation.AFTER:
                    # The graph says e_j must happen *before* e_i, but
                    # in the schedule e_i precedes e_j → inconsistent.
                    logger.debug(
                        "HBConsistencyPruner: %s must precede %s", e_j, e_i
                    )
                    return True
        return False


class SymmetryPruner(AbstractPruner):
    """Prune symmetric schedule permutations.

    Two schedules are *symmetric* if they differ only in the ordering of
    concurrent (causally unrelated) events.  The pruner keeps a set of
    canonical forms and prunes any schedule whose canonical form has
    already been explored.

    The canonical form is obtained by sorting concurrent events by their
    agent identifier (the part of the event id before the first ``'.'``
    or ``'_'`` separator, falling back to lexicographic order on the
    full event id).

    Parameters
    ----------
    hb_graph:
        Optional happens-before graph.  When provided, only concurrent
        events are re-ordered during canonicalisation.  When ``None``
        *all* events are simply sorted lexicographically (a coarser but
        still sound over-approximation).
    """

    def __init__(self, hb_graph: Optional[HBGraph] = None) -> None:
        super().__init__()
        self.hb_graph: Optional[HBGraph] = hb_graph
        self.seen_canonical: Set[Tuple[str, ...]] = set()

    @property
    def name(self) -> str:  # noqa: D401
        return "SymmetryPruner"

    # -- canonicalisation -----------------------------------------------------

    @staticmethod
    def _agent_id(event: Any) -> str:
        """Extract the agent identifier from an event.

        Accepts either a string event id (splits on ``'.'`` / ``'_'``)
        or any object with an ``agent_id`` attribute (e.g.
        :class:`~marace.search.mcts.ScheduleAction`).
        """
        if hasattr(event, "agent_id"):
            return str(event.agent_id)
        event_str = str(event)
        for sep in (".", "_"):
            if sep in event_str:
                return event_str.split(sep, 1)[0]
        return event_str

    @staticmethod
    def _event_key(event: Any) -> str:
        """Return a stable string key for *event* suitable for sorting."""
        if hasattr(event, "agent_id"):
            return f"{event.agent_id}:{getattr(event, 'timing_offset', 0.0)}"
        return str(event)

    def _canonicalize(self, schedule: List[Any]) -> Tuple[str, ...]:
        """Return the canonical form of *schedule*.

        Concurrent events (events for which the HB graph reports
        ``CONCURRENT``) are sorted by :meth:`_agent_id` and then by
        the full event key as a tiebreaker.  Events that are causally
        ordered retain their relative positions.
        """
        if not schedule:
            return ()

        keys = [self._event_key(e) for e in schedule]

        if self.hb_graph is None or self.hb_graph.num_events == 0:
            # Without an HB graph fall back to a simple lexicographic
            # sort of the whole schedule.
            return tuple(sorted(keys))

        # Partition the schedule into maximal runs of mutually
        # concurrent events.  A new run starts whenever the next event
        # is causally ordered w.r.t. the first event of the current
        # run.
        runs: List[List[int]] = []
        current_run: List[int] = [0]

        for idx in range(1, len(schedule)):
            first_key = keys[current_run[0]]
            cur_key = keys[idx]
            # Only use HB check when both events exist in the graph
            if (first_key in self.hb_graph and cur_key in self.hb_graph
                    and self.hb_graph.are_concurrent(first_key, cur_key)):
                current_run.append(idx)
            else:
                runs.append(current_run)
                current_run = [idx]
        runs.append(current_run)

        # Sort each run internally by (agent_id, event_key).
        canonical: List[str] = []
        for run in runs:
            sorted_indices = sorted(
                run,
                key=lambda i: (self._agent_id(schedule[i]), keys[i]),
            )
            canonical.extend(keys[i] for i in sorted_indices)

        return tuple(canonical)

    # -- pruning logic --------------------------------------------------------

    def can_prune(self, schedule: List[str], state: np.ndarray) -> bool:
        """Return ``True`` if an equivalent (symmetric) schedule has
        already been explored."""
        canonical = self._canonicalize(schedule)
        if canonical in self.seen_canonical:
            return True
        self.seen_canonical.add(canonical)
        return False


class DominancePruner(AbstractPruner):
    """Prune schedules that are *dominated* by previously seen ones.

    A schedule is dominated if, for every safety direction considered,
    its margin is strictly worse (lower) than the best margin recorded
    so far by some other schedule.  Dominated schedules can never
    produce a tighter safety violation than those already discovered.

    The pruner maintains a Pareto front of non-dominated margin vectors
    and prunes any new schedule whose margin vector is component-wise
    strictly worse than at least one entry in the front.

    Parameters
    ----------
    safety_directions:
        Collection of direction vectors along which margins are
        evaluated.  Each direction produces one component of the
        margin vector.
    state_zonotope:
        Optional zonotope for margin computation.  May be set later.
    """

    def __init__(
        self,
        safety_directions: Optional[List[np.ndarray]] = None,
        state_zonotope: Optional[Zonotope] = None,
    ) -> None:
        super().__init__()
        self.safety_directions: List[np.ndarray] = [
            np.asarray(d, dtype=np.float64) for d in (safety_directions or [])
        ]
        self.state_zonotope: Optional[Zonotope] = state_zonotope
        # Pareto front of margin vectors (each entry is a list of floats,
        # one per safety direction).
        self.existing_margins: List[List[float]] = []

    @property
    def name(self) -> str:  # noqa: D401
        return "DominancePruner"

    # -- helpers --------------------------------------------------------------

    def _compute_margins(self) -> Optional[List[float]]:
        """Compute the margin vector from the current zonotope.

        Returns ``None`` when the zonotope or directions are unavailable.
        """
        if self.state_zonotope is None or not self.safety_directions:
            return None

        margins: List[float] = []
        for direction in self.safety_directions:
            try:
                min_val, _ = self.state_zonotope.minimize(direction)
                margins.append(float(min_val))
            except Exception:
                logger.warning(
                    "DominancePruner: minimize raised for direction; "
                    "returning None",
                    exc_info=True,
                )
                return None
        return margins

    @staticmethod
    def _is_dominated(
        candidate: List[float], existing: List[List[float]]
    ) -> bool:
        """Return ``True`` if *candidate* is strictly dominated by any
        entry in *existing*.

        A margin vector *a* dominates *b* iff ``a[i] < b[i]`` for every
        component *i* (lower margin = closer to violation = more
        informative schedule).
        """
        for other in existing:
            if all(c > o for c, o in zip(candidate, other)):
                return True
        return False

    # -- pruning logic --------------------------------------------------------

    def can_prune(self, schedule: List[str], state: np.ndarray) -> bool:
        """Return ``True`` when the current schedule is dominated.

        If margin computation fails (e.g. no zonotope configured) the
        branch is conservatively *not* pruned.
        """
        margins = self._compute_margins()
        if margins is None:
            return False

        if self._is_dominated(margins, self.existing_margins):
            return True

        # Not dominated – add to the Pareto front and optionally clean
        # up entries that the new vector now dominates.
        self.existing_margins = [
            m
            for m in self.existing_margins
            if not all(m_i > c_i for m_i, c_i in zip(m, margins))
        ]
        self.existing_margins.append(margins)
        return False


class CompositePruner(AbstractPruner):
    """Chain several pruners and prune if *any* one of them fires.

    The pruners are tried in the order they were supplied.  Evaluation
    short-circuits on the first pruner that returns ``True``.

    Parameters
    ----------
    pruners:
        Ordered sequence of pruners to consult.
    """

    def __init__(self, pruners: Optional[List[AbstractPruner]] = None) -> None:
        super().__init__()
        self.pruners: List[AbstractPruner] = list(pruners or [])

    @property
    def name(self) -> str:  # noqa: D401
        return "CompositePruner"

    # -- composite stats ------------------------------------------------------

    @property
    def combined_stats(self) -> PruningStatistics:
        """Merge statistics from all child pruners into a single report.

        The returned object also incorporates the composite pruner's own
        visit counter so that ``combined_stats.total_nodes_visited``
        reflects the total number of calls to :meth:`try_prune` on the
        composite itself.
        """
        merged = PruningStatistics(
            total_nodes_visited=self._stats.total_nodes_visited,
            nodes_pruned=self._stats.nodes_pruned,
            pruned_by_method=dict(self._stats.pruned_by_method),
        )
        for pruner in self.pruners:
            child = pruner.stats
            for method, count in child.pruned_by_method.items():
                merged.pruned_by_method[method] = (
                    merged.pruned_by_method.get(method, 0) + count
                )
        return merged

    # -- pruning logic --------------------------------------------------------

    def can_prune(self, schedule: List[str], state: np.ndarray) -> bool:
        """Try each child pruner in order; return ``True`` on the first
        hit.

        Each child's :meth:`try_prune` is called (not :meth:`can_prune`)
        so that per-child statistics are kept up to date.
        """
        for pruner in self.pruners:
            if pruner.try_prune(schedule, state):
                logger.debug(
                    "CompositePruner: pruned by child '%s'", pruner.name
                )
                return True
        return False

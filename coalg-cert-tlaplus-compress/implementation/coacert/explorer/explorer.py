"""
State-space exploration engine for explicit-state model checking.

Supports BFS, DFS, and on-the-fly exploration with depth limits,
memory bounds, duplicate detection, and progress reporting.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
)

from .graph import TransitionGraph, StateNode, TransitionEdge

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols for pluggable semantic engine
# ---------------------------------------------------------------------------

class SemanticEngine(Protocol):
    """Interface that a TLA+ semantic evaluator must satisfy."""

    def compute_initial_states(self) -> List[Dict[str, Any]]:
        """Return all initial states of the specification."""
        ...

    def compute_next_states(
        self, state: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Return (action_label, successor_state) pairs."""
        ...

    def evaluate_invariant(
        self, state: Dict[str, Any], invariant: str
    ) -> bool:
        """Return True iff *invariant* holds in *state*."""
        ...

    def compute_atomic_propositions(
        self, state: Dict[str, Any]
    ) -> FrozenSet[str]:
        """Return the set of atomic propositions true in *state*."""
        ...


# ---------------------------------------------------------------------------
# Exploration mode
# ---------------------------------------------------------------------------

class ExplorationMode(Enum):
    BFS = auto()
    DFS = auto()
    ON_THE_FLY = auto()


# ---------------------------------------------------------------------------
# Exploration statistics
# ---------------------------------------------------------------------------

@dataclass
class ExplorationStats:
    mode: str = ""
    states_explored: int = 0
    states_seen: int = 0
    transitions_found: int = 0
    max_depth_reached: int = 0
    depth_limit: Optional[int] = None
    states_per_second: float = 0.0
    elapsed_seconds: float = 0.0
    peak_queue_size: int = 0
    states_evicted: int = 0
    duplicate_states: int = 0
    deadlock_states: int = 0
    invariant_violations: int = 0
    completed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"Exploration({self.mode}): "
            f"{self.states_explored} states, "
            f"{self.transitions_found} transitions, "
            f"depth {self.max_depth_reached}, "
            f"{self.states_per_second:.1f} states/s, "
            f"{self.elapsed_seconds:.2f}s"
        )


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

@dataclass
class ProgressReport:
    states_explored: int
    queue_size: int
    current_depth: int
    states_per_second: float
    elapsed_seconds: float
    memory_usage_mb: float = 0.0


ProgressCallback = Callable[[ProgressReport], None]


def _default_progress(report: ProgressReport) -> None:
    logger.info(
        "Progress: %d states, depth %d, %.1f states/s, queue %d",
        report.states_explored,
        report.current_depth,
        report.states_per_second,
        report.queue_size,
    )


# ---------------------------------------------------------------------------
# State hashing
# ---------------------------------------------------------------------------

def canonical_state_hash(state: Dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 hash of a TLA+ state."""
    canonical = json.dumps(state, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Checkpoint for resumable exploration
# ---------------------------------------------------------------------------

@dataclass
class ExplorationCheckpoint:
    """Snapshot of exploration state for pause/resume."""

    visited_hashes: Set[str] = field(default_factory=set)
    frontier: List[Tuple[str, int]] = field(default_factory=list)  # (hash, depth)
    stats: Optional[ExplorationStats] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "visited_hashes": sorted(self.visited_hashes),
            "frontier": self.frontier,
            "stats": self.stats.to_dict() if self.stats else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExplorationCheckpoint":
        cp = cls()
        cp.visited_hashes = set(d.get("visited_hashes", []))
        cp.frontier = [tuple(x) for x in d.get("frontier", [])]
        if d.get("stats"):
            cp.stats = ExplorationStats(**d["stats"])
        return cp

    def serialize(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def deserialize(cls, s: str) -> "ExplorationCheckpoint":
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# On-the-fly processor
# ---------------------------------------------------------------------------

class OnTheFlyProcessor(Protocol):
    """Callback interface for on-the-fly exploration."""

    def process_state(self, node: StateNode, depth: int) -> bool:
        """Process a newly discovered state.  Return False to abort."""
        ...

    def process_transition(self, edge: TransitionEdge, depth: int) -> bool:
        """Process a newly discovered transition.  Return False to abort."""
        ...


class NullProcessor:
    """Default processor that accepts everything."""

    def process_state(self, node: StateNode, depth: int) -> bool:
        return True

    def process_transition(self, edge: TransitionEdge, depth: int) -> bool:
        return True


class InvariantCheckProcessor:
    """Processor that checks invariants on-the-fly."""

    def __init__(
        self,
        engine: SemanticEngine,
        invariants: List[str],
    ) -> None:
        self._engine = engine
        self._invariants = invariants
        self.violations: List[Tuple[str, StateNode]] = []

    def process_state(self, node: StateNode, depth: int) -> bool:
        for inv in self._invariants:
            if not self._engine.evaluate_invariant(node.full_state, inv):
                self.violations.append((inv, node))
                return False
        return True

    def process_transition(self, edge: TransitionEdge, depth: int) -> bool:
        return True


# ---------------------------------------------------------------------------
# Explicit-state explorer
# ---------------------------------------------------------------------------

class ExplicitStateExplorer:
    """
    Core state-space exploration engine.

    Parameters
    ----------
    engine : SemanticEngine
        Provides initial states, next-state relation, etc.
    graph : TransitionGraph, optional
        Graph to populate; a new one is created if not given.
    depth_limit : int, optional
        Maximum exploration depth (None = unlimited).
    memory_limit_mb : float, optional
        Approximate memory cap; evicts oldest states when exceeded.
    progress_interval : int
        Report progress every N states.
    progress_callback : ProgressCallback, optional
        Receives periodic progress reports.
    """

    def __init__(
        self,
        engine: SemanticEngine,
        graph: Optional[TransitionGraph] = None,
        *,
        depth_limit: Optional[int] = None,
        memory_limit_mb: Optional[float] = None,
        progress_interval: int = 1000,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        self._engine = engine
        self._graph = graph or TransitionGraph()
        self._depth_limit = depth_limit
        self._memory_limit_mb = memory_limit_mb
        self._progress_interval = max(1, progress_interval)
        self._progress_cb = progress_callback or _default_progress

        self._visited: Set[str] = set()
        self._state_depths: Dict[str, int] = {}
        self._stats = ExplorationStats()
        self._aborted = False
        self._checkpoint: Optional[ExplorationCheckpoint] = None

    # -- Properties --------------------------------------------------------

    @property
    def graph(self) -> TransitionGraph:
        return self._graph

    @property
    def stats(self) -> ExplorationStats:
        return self._stats

    @property
    def visited_count(self) -> int:
        return len(self._visited)

    # -- State management --------------------------------------------------

    def _make_node(self, state: Dict[str, Any]) -> StateNode:
        h = canonical_state_hash(state)
        props = self._engine.compute_atomic_propositions(state)
        return StateNode(state_hash=h, full_state=state, atomic_propositions=props)

    def _register_state(self, node: StateNode, depth: int, *, is_initial: bool = False) -> bool:
        """Register a state; returns True if new."""
        h = node.state_hash
        if h in self._visited:
            self._stats.duplicate_states += 1
            return False
        self._visited.add(h)
        self._state_depths[h] = depth
        self._graph.add_state(node, is_initial=is_initial)
        self._stats.states_seen += 1
        if depth > self._stats.max_depth_reached:
            self._stats.max_depth_reached = depth
        return True

    def _should_evict(self) -> bool:
        if self._memory_limit_mb is None:
            return False
        usage = sys.getsizeof(self._visited) / (1024 * 1024)
        return usage > self._memory_limit_mb

    def _evict_states(self, frontier_hashes: Set[str]) -> int:
        """Evict oldest explored states not in the frontier."""
        evicted = 0
        to_remove = []
        for h in list(self._visited):
            if h in frontier_hashes:
                continue
            to_remove.append(h)
            if len(to_remove) >= max(1, len(self._visited) // 10):
                break
        for h in to_remove:
            self._visited.discard(h)
            self._state_depths.pop(h, None)
            evicted += 1
        self._stats.states_evicted += evicted
        return evicted

    # -- Progress ----------------------------------------------------------

    def _maybe_report(self, queue_size: int, depth: int, t0: float) -> None:
        if self._stats.states_explored % self._progress_interval != 0:
            return
        elapsed = time.time() - t0
        sps = self._stats.states_explored / elapsed if elapsed > 0 else 0.0
        self._progress_cb(
            ProgressReport(
                states_explored=self._stats.states_explored,
                queue_size=queue_size,
                current_depth=depth,
                states_per_second=sps,
                elapsed_seconds=elapsed,
            )
        )

    # -- BFS ---------------------------------------------------------------

    def explore_bfs(
        self,
        *,
        depth_limit: Optional[int] = None,
        processor: Optional[OnTheFlyProcessor] = None,
        initial_states: Optional[List[Dict[str, Any]]] = None,
    ) -> ExplorationStats:
        """Breadth-first exploration."""
        limit = depth_limit or self._depth_limit
        proc = processor or NullProcessor()
        self._stats = ExplorationStats(mode="BFS", depth_limit=limit)
        t0 = time.time()

        seeds = initial_states or self._engine.compute_initial_states()
        queue: Deque[Tuple[str, int]] = deque()

        for state in seeds:
            node = self._make_node(state)
            if self._register_state(node, 0, is_initial=True):
                if not proc.process_state(node, 0):
                    self._stats.invariant_violations += 1
                    self._finalize(t0)
                    return self._stats
                queue.append((node.state_hash, 0))

        while queue:
            current_hash, depth = queue.popleft()
            current_node = self._graph.get_state(current_hash)
            if current_node is None:
                continue

            if limit is not None and depth >= limit:
                continue

            self._stats.states_explored += 1
            queue_size = len(queue)
            if queue_size > self._stats.peak_queue_size:
                self._stats.peak_queue_size = queue_size
            self._maybe_report(queue_size, depth, t0)

            successors = self._engine.compute_next_states(current_node.full_state)

            if not successors:
                self._stats.deadlock_states += 1

            for action, succ_state in successors:
                succ_node = self._make_node(succ_state)
                is_new = self._register_state(succ_node, depth + 1)

                is_stutter = (succ_node.state_hash == current_hash)
                edge = TransitionEdge(
                    action_label=action,
                    source_hash=current_hash,
                    target_hash=succ_node.state_hash,
                    is_stuttering=is_stutter,
                )
                self._graph.add_transition(edge)
                self._stats.transitions_found += 1

                if not proc.process_transition(edge, depth + 1):
                    self._finalize(t0)
                    return self._stats

                if is_new:
                    if not proc.process_state(succ_node, depth + 1):
                        self._stats.invariant_violations += 1
                        self._finalize(t0)
                        return self._stats
                    queue.append((succ_node.state_hash, depth + 1))

            if self._should_evict():
                frontier = {h for h, _ in queue}
                self._evict_states(frontier)

        self._stats.completed = True
        self._finalize(t0)
        return self._stats

    # -- DFS ---------------------------------------------------------------

    def explore_dfs(
        self,
        *,
        depth_limit: Optional[int] = None,
        processor: Optional[OnTheFlyProcessor] = None,
        initial_states: Optional[List[Dict[str, Any]]] = None,
    ) -> ExplorationStats:
        """Depth-first exploration."""
        limit = depth_limit or self._depth_limit
        proc = processor or NullProcessor()
        self._stats = ExplorationStats(mode="DFS", depth_limit=limit)
        t0 = time.time()

        seeds = initial_states or self._engine.compute_initial_states()
        stack: List[Tuple[str, int]] = []

        for state in seeds:
            node = self._make_node(state)
            if self._register_state(node, 0, is_initial=True):
                if not proc.process_state(node, 0):
                    self._stats.invariant_violations += 1
                    self._finalize(t0)
                    return self._stats
                stack.append((node.state_hash, 0))

        while stack:
            current_hash, depth = stack.pop()
            current_node = self._graph.get_state(current_hash)
            if current_node is None:
                continue

            if limit is not None and depth >= limit:
                continue

            self._stats.states_explored += 1
            stack_size = len(stack)
            if stack_size > self._stats.peak_queue_size:
                self._stats.peak_queue_size = stack_size
            self._maybe_report(stack_size, depth, t0)

            successors = self._engine.compute_next_states(current_node.full_state)

            if not successors:
                self._stats.deadlock_states += 1

            for action, succ_state in successors:
                succ_node = self._make_node(succ_state)
                is_new = self._register_state(succ_node, depth + 1)

                is_stutter = (succ_node.state_hash == current_hash)
                edge = TransitionEdge(
                    action_label=action,
                    source_hash=current_hash,
                    target_hash=succ_node.state_hash,
                    is_stuttering=is_stutter,
                )
                self._graph.add_transition(edge)
                self._stats.transitions_found += 1

                if not proc.process_transition(edge, depth + 1):
                    self._finalize(t0)
                    return self._stats

                if is_new:
                    if not proc.process_state(succ_node, depth + 1):
                        self._stats.invariant_violations += 1
                        self._finalize(t0)
                        return self._stats
                    stack.append((succ_node.state_hash, depth + 1))

            if self._should_evict():
                frontier = {h for h, _ in stack}
                self._evict_states(frontier)

        self._stats.completed = True
        self._finalize(t0)
        return self._stats

    # -- On-the-fly --------------------------------------------------------

    def explore_on_the_fly(
        self,
        processor: OnTheFlyProcessor,
        *,
        mode: ExplorationMode = ExplorationMode.BFS,
        depth_limit: Optional[int] = None,
        initial_states: Optional[List[Dict[str, Any]]] = None,
    ) -> ExplorationStats:
        """Explore and process simultaneously with early termination."""
        if mode == ExplorationMode.DFS:
            return self.explore_dfs(
                depth_limit=depth_limit,
                processor=processor,
                initial_states=initial_states,
            )
        return self.explore_bfs(
            depth_limit=depth_limit,
            processor=processor,
            initial_states=initial_states,
        )

    # -- Multi-initial-state -----------------------------------------------

    def explore_from_states(
        self,
        states: List[Dict[str, Any]],
        *,
        mode: ExplorationMode = ExplorationMode.BFS,
        depth_limit: Optional[int] = None,
    ) -> ExplorationStats:
        """Explore from arbitrary (multiple) initial states."""
        if mode == ExplorationMode.DFS:
            return self.explore_dfs(depth_limit=depth_limit, initial_states=states)
        return self.explore_bfs(depth_limit=depth_limit, initial_states=states)

    # -- Checkpointing -----------------------------------------------------

    def save_checkpoint(self) -> ExplorationCheckpoint:
        cp = ExplorationCheckpoint(
            visited_hashes=set(self._visited),
            frontier=[],
            stats=self._stats,
        )
        self._checkpoint = cp
        return cp

    def resume_from_checkpoint(
        self,
        checkpoint: ExplorationCheckpoint,
        *,
        mode: ExplorationMode = ExplorationMode.BFS,
        depth_limit: Optional[int] = None,
        processor: Optional[OnTheFlyProcessor] = None,
    ) -> ExplorationStats:
        """Resume exploration from a saved checkpoint."""
        self._visited = set(checkpoint.visited_hashes)
        if checkpoint.stats:
            self._stats = checkpoint.stats
        limit = depth_limit or self._depth_limit
        proc = processor or NullProcessor()
        t0 = time.time()

        frontier_items = checkpoint.frontier
        if not frontier_items:
            frontier_items = self._compute_frontier()

        if mode == ExplorationMode.DFS:
            container: Any = list(frontier_items)
            pop_method = "pop"
        else:
            container = deque(frontier_items)
            pop_method = "popleft"

        while container:
            current_hash, depth = getattr(container, pop_method)()
            current_node = self._graph.get_state(current_hash)
            if current_node is None:
                continue
            if limit is not None and depth >= limit:
                continue

            self._stats.states_explored += 1
            self._maybe_report(len(container), depth, t0)

            successors = self._engine.compute_next_states(current_node.full_state)
            for action, succ_state in successors:
                succ_node = self._make_node(succ_state)
                is_new = self._register_state(succ_node, depth + 1)

                edge = TransitionEdge(
                    action_label=action,
                    source_hash=current_hash,
                    target_hash=succ_node.state_hash,
                    is_stuttering=(succ_node.state_hash == current_hash),
                )
                self._graph.add_transition(edge)
                self._stats.transitions_found += 1

                if is_new:
                    if not proc.process_state(succ_node, depth + 1):
                        self._stats.invariant_violations += 1
                        self._finalize(t0)
                        return self._stats
                    if mode == ExplorationMode.DFS:
                        container.append((succ_node.state_hash, depth + 1))
                    else:
                        container.append((succ_node.state_hash, depth + 1))

        self._stats.completed = True
        self._finalize(t0)
        return self._stats

    def _compute_frontier(self) -> List[Tuple[str, int]]:
        """Identify unexplored boundary states."""
        frontier: List[Tuple[str, int]] = []
        for h in self._visited:
            node = self._graph.get_state(h)
            if node is None:
                continue
            for succ_h in self._graph.get_successor_hashes(h):
                if succ_h not in self._visited:
                    depth = self._state_depths.get(succ_h, 0)
                    frontier.append((succ_h, depth))
        if not frontier:
            for h in self._visited:
                depth = self._state_depths.get(h, 0)
                successors = self._engine.compute_next_states(
                    self._graph.get_state(h).full_state
                )
                for action, succ_state in successors:
                    sh = canonical_state_hash(succ_state)
                    if sh not in self._visited:
                        frontier.append((sh, depth + 1))
                        break
        return frontier

    # -- Finalization ------------------------------------------------------

    def _finalize(self, t0: float) -> None:
        elapsed = time.time() - t0
        self._stats.elapsed_seconds = elapsed
        if elapsed > 0:
            self._stats.states_per_second = self._stats.states_explored / elapsed

    # -- Utility -----------------------------------------------------------

    def reset(self) -> None:
        self._visited.clear()
        self._state_depths.clear()
        self._stats = ExplorationStats()
        self._aborted = False
        self._checkpoint = None

    def get_state_depth(self, state_hash: str) -> Optional[int]:
        return self._state_depths.get(state_hash)

    def is_explored(self, state_hash: str) -> bool:
        return state_hash in self._visited

    def get_exploration_frontier(self) -> Set[str]:
        """States in the graph with unexplored successors."""
        frontier: Set[str] = set()
        for h in self._visited:
            succs = self._graph.get_successor_hashes(h)
            if any(s not in self._visited for s in succs):
                frontier.add(h)
        return frontier

    def find_deadlocks(self) -> List[StateNode]:
        return self._graph.get_deadlock_states()

    def find_invariant_violations(
        self, invariant: str
    ) -> List[StateNode]:
        """Find all states that violate *invariant*."""
        violations = []
        for node in self._graph.all_states():
            if not self._engine.evaluate_invariant(node.full_state, invariant):
                violations.append(node)
        return violations

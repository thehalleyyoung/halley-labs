"""
Monte Carlo Tree Search over schedule space for MARACE.

Implements UCB1-based MCTS to explore the combinatorial space of
multi-agent interleavings (schedules) while respecting happens-before
constraints.  The search is *k*-bounded: only the first *max_depth*
scheduling decisions are explored, and safety margins are estimated
via random rollouts beyond that horizon.

The core loop follows the standard selection → expansion → simulation →
backpropagation cycle, with two MARACE-specific extensions:

* **HB-consistent action filtering** – only interleavings that do not
  violate happens-before constraints are considered during expansion
  and simulation.
* **Safety-margin–biased UCB** – the UCB score incorporates the
  evaluated safety margin so that the search is drawn towards schedules
  that are *close* to a race condition (low / negative margin).
"""

from __future__ import annotations

import copy
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from marace.hb.hb_graph import HBGraph, HBRelation

logger = logging.getLogger(__name__)

__all__ = [
    "ScheduleAction",
    "SearchBudget",
    "SearchResult",
    "MCTSNode",
    "MCTSTree",
    "MCTS",
]


# ======================================================================
# Data containers
# ======================================================================


@dataclass
class ScheduleAction:
    """A single scheduling decision: which agent acts next and when.

    Attributes
    ----------
    agent_id:
        Identifier of the agent selected to take the next step.
    timing_offset:
        Continuous timing offset (in seconds) applied to this action
        relative to the previous step.  Allows modelling of
        non-deterministic timing.
    event_type:
        Label for the kind of event being scheduled (default
        ``"action"``).
    """

    agent_id: str
    timing_offset: float
    event_type: str = "action"

    # Hashable so it can serve as a dictionary key in MCTSNode.children.
    def __hash__(self) -> int:
        return hash((self.agent_id, self.timing_offset, self.event_type))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScheduleAction):
            return NotImplemented
        return (
            self.agent_id == other.agent_id
            and self.timing_offset == other.timing_offset
            and self.event_type == other.event_type
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ScheduleAction(agent={self.agent_id!r}, "
            f"offset={self.timing_offset:.4f}, type={self.event_type!r})"
        )


@dataclass
class SearchBudget:
    """Resource limits that govern how long the MCTS search may run.

    The search terminates as soon as *any* budget dimension is exhausted.

    Attributes
    ----------
    iteration_count:
        Maximum number of MCTS iterations (selection–backprop cycles).
    time_limit_seconds:
        Wall-clock time limit in seconds.
    max_nodes:
        Upper bound on the total number of tree nodes created.
    """

    iteration_count: int = 10_000
    time_limit_seconds: float = 60.0
    max_nodes: int = 500_000

    def is_exhausted(
        self,
        iterations_done: int,
        time_elapsed: float,
        nodes_created: int,
    ) -> bool:
        """Return ``True`` when at least one budget dimension is exceeded.

        Parameters
        ----------
        iterations_done:
            Number of MCTS iterations completed so far.
        time_elapsed:
            Wall-clock seconds since search start.
        nodes_created:
            Total tree nodes allocated.

        Returns
        -------
        bool
        """
        return (
            iterations_done >= self.iteration_count
            or time_elapsed >= self.time_limit_seconds
            or nodes_created >= self.max_nodes
        )


@dataclass
class SearchResult:
    """Outcome produced by :meth:`MCTS.search`.

    Attributes
    ----------
    best_schedule:
        Sequence of :class:`ScheduleAction` that achieves the lowest
        observed safety margin.
    safety_margin:
        The safety margin of *best_schedule*.  A non-positive value
        indicates a potential race condition.
    replay_trace:
        Optional trace of intermediate abstract states useful for
        debugging / visualisation.
    statistics:
        Dictionary of summary statistics (iterations, nodes_explored,
        time_elapsed, pruned_count, best_margin_over_time).
    """

    best_schedule: List[ScheduleAction] = field(default_factory=list)
    safety_margin: float = float("inf")
    replay_trace: Optional[List[np.ndarray]] = None
    statistics: Dict[str, object] = field(default_factory=dict)

    # -- derived property -----------------------------------------------------

    @property
    def is_race_found(self) -> bool:
        """``True`` when the best schedule's margin is ≤ 0."""
        return self.safety_margin <= 0.0

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SearchResult(margin={self.safety_margin:.6f}, "
            f"race_found={self.is_race_found}, "
            f"schedule_len={len(self.best_schedule)}, "
            f"stats_keys={list(self.statistics.keys())})"
        )


# ======================================================================
# MCTSNode
# ======================================================================


class MCTSNode:
    """A single node in the MCTS search tree.

    Each node represents a *partial schedule*—a prefix of the full
    interleaving of agent actions—together with the abstract state
    that results from executing that prefix.

    Attributes
    ----------
    schedule:
        Actions taken to reach this node (root has an empty list).
    abstract_state:
        State vector summarising the system after the partial schedule.
    parent:
        Pointer to the parent node (``None`` for the root).
    depth:
        Depth in the tree (root = 0).
    prior:
        Prior probability / bias used during expansion (default 1.0).
    """

    __slots__ = (
        "schedule",
        "abstract_state",
        "children",
        "visit_count",
        "value_sum",
        "prior",
        "parent",
        "depth",
        "_valid_actions",
    )

    def __init__(
        self,
        schedule: List[ScheduleAction],
        abstract_state: np.ndarray,
        parent: Optional[MCTSNode] = None,
        depth: int = 0,
        prior: float = 1.0,
    ) -> None:
        self.schedule = list(schedule)
        self.abstract_state = abstract_state.copy()
        self.children: Dict[ScheduleAction, MCTSNode] = {}
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.prior = prior
        self.parent = parent
        self.depth = depth
        # Lazily populated by the tree during expansion.
        self._valid_actions: Optional[List[ScheduleAction]] = None

    # -- properties -----------------------------------------------------------

    @property
    def value(self) -> float:
        """Mean value (lower = closer to a race)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def is_terminal(self) -> bool:
        """A node is terminal when no valid actions remain.

        ``_valid_actions`` must be populated first via the tree's
        expansion logic; if it has not been set yet the node is
        *assumed* non-terminal.
        """
        if self._valid_actions is None:
            return False
        return len(self._valid_actions) == 0

    @property
    def is_fully_expanded(self) -> bool:
        """``True`` when every valid child has been instantiated."""
        if self._valid_actions is None:
            return False
        return len(self.children) >= len(self._valid_actions)

    # -- UCB ------------------------------------------------------------------

    def ucb_score(self, exploration_constant: float = 1.414) -> float:
        """Compute the UCB1 score for this node.

        Lower *value* is better (closer to a race), so we *negate* the
        exploitation term to bias selection towards low-margin children.

        Parameters
        ----------
        exploration_constant:
            Balances exploitation vs. exploration (default √2 ≈ 1.414).

        Returns
        -------
        float
            The UCB1 score.  Unvisited nodes return ``+inf`` to ensure
            they are selected first.
        """
        if self.visit_count == 0:
            return float("inf")
        if self.parent is None or self.parent.visit_count == 0:
            return -self.value
        exploitation = -self.value  # lower margin is better → negate
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visit_count) / self.visit_count
        )
        return exploitation + exploration

    # -- selection helpers ----------------------------------------------------

    def best_child(self, exploration_constant: float = 1.414) -> MCTSNode:
        """Return the child with the highest UCB score.

        Parameters
        ----------
        exploration_constant:
            Passed through to :meth:`ucb_score`.

        Returns
        -------
        MCTSNode

        Raises
        ------
        ValueError
            If the node has no children.
        """
        if not self.children:
            raise ValueError("best_child called on a childless node")
        return max(
            self.children.values(),
            key=lambda c: c.ucb_score(exploration_constant),
        )

    # -- dunder ---------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"MCTSNode(depth={self.depth}, visits={self.visit_count}, "
            f"value={self.value:.4f}, children={len(self.children)}, "
            f"schedule_len={len(self.schedule)})"
        )


# ======================================================================
# MCTSTree
# ======================================================================


class MCTSTree:
    """Container that manages the full MCTS search tree.

    Provides bookkeeping (total size, best-path extraction, statistics
    aggregation) on top of the raw :class:`MCTSNode` graph.

    Parameters
    ----------
    root_state:
        Abstract state vector for the root node.
    """

    def __init__(self, root_state: np.ndarray) -> None:
        self._root = MCTSNode(
            schedule=[],
            abstract_state=root_state,
            parent=None,
            depth=0,
        )
        self._size: int = 1

    # -- properties -----------------------------------------------------------

    @property
    def root(self) -> MCTSNode:
        """The tree root."""
        return self._root

    @property
    def size(self) -> int:
        """Total number of nodes in the tree."""
        return self._size

    # -- mutation -------------------------------------------------------------

    def register_node(self) -> None:
        """Increment the internal node counter.

        Call this every time a new :class:`MCTSNode` is appended to the
        tree so that :attr:`size` stays accurate.
        """
        self._size += 1

    # -- path extraction ------------------------------------------------------

    def get_best_path(self) -> List[ScheduleAction]:
        """Trace the most-visited path from root to a leaf.

        At each level the child with the highest visit count is chosen,
        breaking ties by lower *value* (closer to a race).

        Returns
        -------
        List[ScheduleAction]
            The schedule along the best path.
        """
        path: List[ScheduleAction] = []
        node = self._root
        while node.children:
            best = max(
                node.children.items(),
                key=lambda kv: (kv[1].visit_count, -kv[1].value),
            )
            action, child = best
            path.append(action)
            node = child
        return path

    def get_best_leaf(self) -> MCTSNode:
        """Return the leaf node at the end of :meth:`get_best_path`."""
        node = self._root
        while node.children:
            best = max(
                node.children.items(),
                key=lambda kv: (kv[1].visit_count, -kv[1].value),
            )
            node = best[1]
        return node

    # -- statistics -----------------------------------------------------------

    def get_statistics(self) -> Dict[str, object]:
        """Aggregate tree-level statistics.

        Returns
        -------
        Dict[str, object]
            Keys: ``total_nodes``, ``max_depth``, ``root_visits``,
            ``branching_factor_avg``.
        """
        max_depth = 0
        total_children = 0
        internal_nodes = 0
        queue: List[MCTSNode] = [self._root]
        while queue:
            node = queue.pop()
            if node.depth > max_depth:
                max_depth = node.depth
            if node.children:
                internal_nodes += 1
                total_children += len(node.children)
                queue.extend(node.children.values())
        branching = (
            total_children / internal_nodes if internal_nodes > 0 else 0.0
        )
        return {
            "total_nodes": self._size,
            "max_depth": max_depth,
            "root_visits": self._root.visit_count,
            "branching_factor_avg": round(branching, 3),
        }

    # -- reset ----------------------------------------------------------------

    def reset(self, root_state: Optional[np.ndarray] = None) -> None:
        """Discard the tree and create a fresh root.

        Parameters
        ----------
        root_state:
            New root abstract state.  If ``None`` the previous root
            state is reused.
        """
        state = root_state if root_state is not None else self._root.abstract_state
        self._root = MCTSNode(
            schedule=[],
            abstract_state=state,
            parent=None,
            depth=0,
        )
        self._size = 1

    # -- dunder ---------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return f"MCTSTree(size={self._size}, root_visits={self._root.visit_count})"


# ======================================================================
# MCTS
# ======================================================================


class MCTS:
    """Monte Carlo Tree Search over multi-agent schedule space.

    The search biases exploration towards schedules whose safety margin
    is low (i.e. close to a race condition) using a negated-value UCB1
    formulation.  Actions that violate happens-before constraints are
    filtered out during both expansion and simulation.

    Parameters
    ----------
    agent_ids:
        Identifiers of the agents whose actions are interleaved.
    max_depth:
        Maximum schedule length (*k*-bounded search).
    hb_graph:
        The happens-before graph used to prune inconsistent
        interleavings.
    safety_evaluator:
        Callable ``(state, schedule) -> float`` that computes the
        safety margin for a given abstract state and partial schedule.
        Non-positive return values indicate a potential violation.
    exploration_constant:
        UCB1 exploration weight (default √2).
    timing_range:
        ``(lo, hi)`` bounds for the continuous timing offset sampled
        during expansion and simulation.
    seed:
        RNG seed for reproducible searches.
    """

    def __init__(
        self,
        agent_ids: List[str],
        max_depth: int,
        hb_graph: HBGraph,
        safety_evaluator: Callable[[np.ndarray, List[ScheduleAction]], float],
        exploration_constant: float = 1.414,
        timing_range: Tuple[float, float] = (0.0, 1.0),
        seed: Optional[int] = None,
    ) -> None:
        self._agent_ids = list(agent_ids)
        self._max_depth = max_depth
        self._hb_graph = hb_graph
        self._safety_evaluator = safety_evaluator
        self._exploration_constant = exploration_constant
        self._timing_lo, self._timing_hi = timing_range
        self._rng = np.random.default_rng(seed)

        # Runtime bookkeeping (reset each search call).
        self._pruned_count: int = 0
        self._best_margin_over_time: List[Tuple[int, float]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        initial_state: np.ndarray,
        budget: SearchBudget,
    ) -> SearchResult:
        """Run the MCTS loop and return the best adversarial schedule.

        Parameters
        ----------
        initial_state:
            Abstract state vector at the root (before any scheduling
            decisions have been made).
        budget:
            Resource limits that govern termination.

        Returns
        -------
        SearchResult
            Contains the best schedule, its safety margin, and summary
            statistics.
        """
        self._pruned_count = 0
        self._best_margin_over_time = []
        tree = MCTSTree(initial_state)
        best_margin = float("inf")
        best_schedule: List[ScheduleAction] = []

        start_time = time.monotonic()
        iteration = 0

        logger.info(
            "MCTS search started (agents=%s, max_depth=%d, budget=%s)",
            self._agent_ids,
            self._max_depth,
            budget,
        )

        while not budget.is_exhausted(
            iteration,
            time.monotonic() - start_time,
            tree.size,
        ):
            # 1. Selection
            leaf = self._select(tree.root)

            # 2. Expansion
            child = self._expand(leaf, tree)

            # 3. Simulation (rollout)
            rollout_value = self._simulate(child)

            # 4. Backpropagation
            self._backpropagate(child, rollout_value)

            # Track best margin.
            if rollout_value < best_margin:
                best_margin = rollout_value
                best_schedule = list(child.schedule)
                logger.debug(
                    "New best margin %.6f at iteration %d (depth=%d)",
                    best_margin,
                    iteration,
                    child.depth,
                )
            self._best_margin_over_time.append((iteration, best_margin))

            iteration += 1

        elapsed = time.monotonic() - start_time

        # Also check the most-visited path.
        most_visited_schedule = tree.get_best_path()
        most_visited_margin = self._evaluate_safety_margin(
            initial_state, most_visited_schedule,
        )
        if most_visited_margin < best_margin:
            best_margin = most_visited_margin
            best_schedule = most_visited_schedule

        tree_stats = tree.get_statistics()
        statistics: Dict[str, object] = {
            "iterations": iteration,
            "nodes_explored": tree.size,
            "time_elapsed": round(elapsed, 4),
            "pruned_count": self._pruned_count,
            "best_margin_over_time": list(self._best_margin_over_time),
            **tree_stats,
        }

        logger.info(
            "MCTS search completed: iterations=%d, nodes=%d, "
            "best_margin=%.6f, elapsed=%.2fs",
            iteration,
            tree.size,
            best_margin,
            elapsed,
        )

        return SearchResult(
            best_schedule=best_schedule,
            safety_margin=best_margin,
            replay_trace=None,
            statistics=statistics,
        )

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Descend the tree using UCB1 until a non-fully-expanded or
        terminal node is reached.

        At each level the child with the highest UCB score is followed.
        The UCB formulation *negates* the value term so that
        low-margin (adversarial) branches are preferred.

        Parameters
        ----------
        node:
            Node at which to begin the descent (usually the root).

        Returns
        -------
        MCTSNode
            The selected leaf / expandable node.
        """
        current = node
        while current.is_fully_expanded and not current.is_terminal:
            if not current.children:
                break
            current = current.best_child(self._exploration_constant)
        return current

    # ------------------------------------------------------------------
    # Expansion
    # ------------------------------------------------------------------

    def _expand(self, node: MCTSNode, tree: MCTSTree) -> MCTSNode:
        """Add one new child to *node* by choosing an untried valid
        action, and return the new child.

        If *node* is terminal or at the depth limit, it is returned
        unchanged.

        Parameters
        ----------
        node:
            The node to expand.
        tree:
            The owning tree (for bookkeeping).

        Returns
        -------
        MCTSNode
            The newly created child, or *node* if expansion is not
            possible.
        """
        if node.depth >= self._max_depth:
            return node

        # Ensure valid actions are computed.
        if node._valid_actions is None:
            node._valid_actions = self._get_valid_actions(node)

        if node.is_terminal or node.is_fully_expanded:
            return node

        # Pick the first untried action.
        tried_actions = set(node.children.keys())
        untried = [a for a in node._valid_actions if a not in tried_actions]
        if not untried:
            return node

        action = untried[int(self._rng.integers(0, len(untried)))]

        child_state = self._apply_action(node.abstract_state, action)
        child = MCTSNode(
            schedule=node.schedule + [action],
            abstract_state=child_state,
            parent=node,
            depth=node.depth + 1,
            prior=1.0 / max(len(node._valid_actions), 1),
        )
        node.children[action] = child
        tree.register_node()
        return child

    # ------------------------------------------------------------------
    # Simulation (rollout)
    # ------------------------------------------------------------------

    def _simulate(self, node: MCTSNode) -> float:
        """Perform a random rollout from *node* to estimate the safety
        margin.

        Actions are sampled uniformly from the valid set at each step
        until either the depth limit is reached or no valid actions
        remain.  The safety margin is then evaluated on the resulting
        full schedule.

        Parameters
        ----------
        node:
            Starting point of the rollout.

        Returns
        -------
        float
            The estimated safety margin (lower = closer to a race).
        """
        schedule = list(node.schedule)
        state = node.abstract_state.copy()
        depth = node.depth

        for _ in range(self._max_depth - depth):
            actions = self._get_valid_actions_for_rollout(schedule, state)
            if not actions:
                break
            action = actions[int(self._rng.integers(0, len(actions)))]
            state = self._apply_action(state, action)
            schedule.append(action)

        return self._evaluate_safety_margin(state, schedule)

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Propagate the rollout value up from *node* to the root.

        Every ancestor's visit count is incremented and its value sum
        is updated.  Because lower margins are better (adversarial
        search), the raw *value* is propagated without negation—
        selection handles sign via :meth:`MCTSNode.ucb_score`.

        Parameters
        ----------
        node:
            Leaf node returned by :meth:`_simulate`.
        value:
            Safety margin estimated during the rollout.
        """
        current: Optional[MCTSNode] = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += value
            current = current.parent

    # ------------------------------------------------------------------
    # Valid-action generation
    # ------------------------------------------------------------------

    def _get_valid_actions(self, node: MCTSNode) -> List[ScheduleAction]:
        """Compute the set of actions available at *node* that are
        consistent with the happens-before graph.

        An action ``(agent_id, timing_offset)`` is valid when scheduling
        that agent at this point does not create a cycle in the
        happens-before relation—i.e. no event of *agent_id* that must
        come after an event not yet scheduled would be forced before it.

        Parameters
        ----------
        node:
            The tree node whose valid actions we need.

        Returns
        -------
        List[ScheduleAction]
            HB-consistent actions.  Empty list signals a terminal node.
        """
        scheduled_agents = [a.agent_id for a in node.schedule]
        actions: List[ScheduleAction] = []

        for agent_id in self._agent_ids:
            if not self._is_hb_consistent(agent_id, scheduled_agents):
                self._pruned_count += 1
                continue
            offset = float(
                self._rng.uniform(self._timing_lo, self._timing_hi)
            )
            actions.append(
                ScheduleAction(
                    agent_id=agent_id,
                    timing_offset=offset,
                    event_type="action",
                )
            )

        return actions

    def _get_valid_actions_for_rollout(
        self,
        schedule: List[ScheduleAction],
        state: np.ndarray,
    ) -> List[ScheduleAction]:
        """Lightweight action generation used during random rollouts.

        Parameters
        ----------
        schedule:
            Schedule prefix built so far during the rollout.
        state:
            Current abstract state (unused in the base implementation
            but available for subclass overrides).

        Returns
        -------
        List[ScheduleAction]
        """
        scheduled_agents = [a.agent_id for a in schedule]
        actions: List[ScheduleAction] = []
        for agent_id in self._agent_ids:
            if not self._is_hb_consistent(agent_id, scheduled_agents):
                continue
            offset = float(
                self._rng.uniform(self._timing_lo, self._timing_hi)
            )
            actions.append(
                ScheduleAction(
                    agent_id=agent_id,
                    timing_offset=offset,
                    event_type="action",
                )
            )
        return actions

    # ------------------------------------------------------------------
    # HB consistency
    # ------------------------------------------------------------------

    def _is_hb_consistent(
        self,
        agent_id: str,
        scheduled_agents: List[str],
    ) -> bool:
        """Check whether scheduling *agent_id* next is consistent with
        the happens-before graph.

        The check verifies that no event belonging to *agent_id* is
        required (by an HB edge) to come *after* an event of an agent
        that has *not yet* been scheduled.  This is a lightweight
        approximation that avoids a full topological-sort at every node.

        Parameters
        ----------
        agent_id:
            The agent we want to schedule next.
        scheduled_agents:
            Agent IDs that have already been scheduled (in order).

        Returns
        -------
        bool
            ``True`` if scheduling *agent_id* is HB-consistent.
        """
        if self._hb_graph.num_events == 0:
            return True

        agent_events: List[str] = []
        for eid in self._hb_graph._g.nodes():
            data = self._hb_graph._g.nodes[eid]
            if data.get("agent_id") == agent_id:
                agent_events.append(eid)

        if not agent_events:
            # No HB events registered for this agent → always valid.
            return True

        scheduled_set = set(scheduled_agents)
        for evt in agent_events:
            # Check predecessors: every predecessor's agent must have
            # been scheduled already.
            for pred in self._hb_graph._g.predecessors(evt):
                pred_agent = self._hb_graph._g.nodes[pred].get("agent_id")
                if pred_agent and pred_agent not in scheduled_set:
                    return False

        return True

    # ------------------------------------------------------------------
    # State transition
    # ------------------------------------------------------------------

    def _apply_action(
        self,
        state: np.ndarray,
        action: ScheduleAction,
    ) -> np.ndarray:
        """Produce the successor abstract state after *action*.

        The default implementation applies a small deterministic
        perturbation so that distinct schedule paths yield
        distinguishable states.  Subclasses should override this with
        the actual transition semantics of the system under test.

        Parameters
        ----------
        state:
            Current abstract state vector.
        action:
            The scheduling decision just taken.

        Returns
        -------
        np.ndarray
            Updated state vector (a fresh copy).
        """
        new_state = state.copy()
        agent_hash = hash(action.agent_id) % (2**16)
        perturbation = (agent_hash / (2**16)) * action.timing_offset * 0.01
        new_state = new_state + perturbation
        return new_state

    # ------------------------------------------------------------------
    # Safety evaluation
    # ------------------------------------------------------------------

    def _evaluate_safety_margin(
        self,
        state: np.ndarray,
        schedule: List[ScheduleAction],
    ) -> float:
        """Evaluate the safety margin for *schedule* applied from
        *state*.

        Delegates to the user-supplied ``safety_evaluator`` callback and
        clamps ``NaN`` / ``inf`` results to a large positive value so
        that the tree statistics remain well-defined.

        Parameters
        ----------
        state:
            Abstract state at which the schedule begins.
        schedule:
            Full or partial schedule to evaluate.

        Returns
        -------
        float
            The computed safety margin.
        """
        try:
            margin = float(self._safety_evaluator(state, schedule))
        except Exception:
            logger.warning(
                "safety_evaluator raised an exception for schedule of "
                "length %d; returning +inf",
                len(schedule),
                exc_info=True,
            )
            return float("inf")

        if math.isnan(margin):
            logger.debug("safety_evaluator returned NaN; clamping to +inf")
            return float("inf")
        return margin

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"MCTS(agents={self._agent_ids}, max_depth={self._max_depth}, "
            f"c={self._exploration_constant})"
        )

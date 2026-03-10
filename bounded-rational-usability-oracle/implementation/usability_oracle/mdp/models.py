"""
usability_oracle.mdp.models — Core MDP data structures.

Defines :class:`State`, :class:`Action`, :class:`Transition`, and :class:`MDP`
data classes for representing Markov Decision Processes constructed from UI
accessibility trees.

The MDP follows the standard tuple ⟨S, A, T, R, γ⟩ where:
  - S is a finite set of states (tree position × task progress),
  - A is a finite set of UI actions,
  - T : S × A × S → [0, 1] is the transition probability function,
  - R : S × A → ℝ is the (negative) cost / reward,
  - γ ∈ [0, 1) is the discount factor.

References
----------
- Puterman, M. L. (1994). *Markov Decision Processes*.
- Todorov, E. (2007). Linearly solvable MDPs. *NIPS*.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class State:
    """A single state in the UI-navigation MDP.

    Encodes the user's *current focus position* in the accessibility tree,
    the *task progress* achieved so far (as feature flags), and auxiliary
    metadata needed by cognitive cost models.

    Parameters
    ----------
    state_id : str
        Unique identifier (typically ``<node_id>:<progress_hash>``).
    features : dict[str, float]
        Numeric feature vector for this state (visual complexity, depth, …).
    label : str
        Human-readable label, e.g. ``"button:Submit @ step 2"``.
    is_terminal : bool
        Whether further actions are possible.
    is_goal : bool
        Whether this state satisfies the task specification.
    metadata : dict
        Arbitrary extra data (e.g. working-memory contents, scroll offset).
    """

    state_id: str
    features: dict[str, float] = field(default_factory=dict)
    label: str = ""
    is_terminal: bool = False
    is_goal: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    # Convenience accessors ------------------------------------------------

    @property
    def node_id(self) -> str:
        """The accessibility node this state is centred on."""
        return self.metadata.get("node_id", self.state_id.split(":")[0])

    @property
    def task_progress(self) -> float:
        """Fraction of task sub-goals completed."""
        return self.features.get("task_progress", 0.0)

    @property
    def working_memory_load(self) -> float:
        """Number of items held in working memory (0–7+)."""
        return self.features.get("working_memory_load", 0.0)

    def __repr__(self) -> str:
        return (
            f"State(id={self.state_id!r}, label={self.label!r}, "
            f"terminal={self.is_terminal}, goal={self.is_goal})"
        )


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Action:
    """An atomic user action in the MDP.

    Actions map to observable UI interactions: clicking a target, typing text
    into an input, pressing Tab, scrolling, reading content, or navigating
    via keyboard shortcuts.

    Parameters
    ----------
    action_id : str
        Unique identifier.
    action_type : str
        One of ``click``, ``type``, ``tab``, ``scroll``, ``navigate``,
        ``read``, ``select``, ``back``.
    target_node_id : str | None
        Accessibility node the action targets (``None`` for global actions
        like Back or scroll).
    description : str
        Human-readable description.
    preconditions : list[str]
        State predicates that must hold for this action to be available.
    """

    action_id: str
    action_type: str
    target_node_id: Optional[str] = None
    description: str = ""
    preconditions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Action type constants ------------------------------------------------

    CLICK: str = "click"
    TYPE: str = "type"
    TAB: str = "tab"
    SCROLL: str = "scroll"
    NAVIGATE: str = "navigate"
    READ: str = "read"
    SELECT: str = "select"
    BACK: str = "back"
    SWIPE: str = "swipe"
    LONG_PRESS: str = "long_press"

    _ACTION_TYPES: frozenset[str] = frozenset(
        {"click", "type", "tab", "scroll", "navigate", "read", "select",
         "back", "swipe", "long_press"}
    )

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty if valid)."""
        errors: list[str] = []
        if not self.action_id:
            errors.append("action_id must be non-empty")
        if self.action_type not in self._ACTION_TYPES:
            errors.append(
                f"Unknown action_type {self.action_type!r}; "
                f"expected one of {sorted(self._ACTION_TYPES)}"
            )
        if self.action_type in {"click", "type", "select", "read"} and not self.target_node_id:
            errors.append(f"Action type {self.action_type!r} requires a target_node_id")
        return errors

    def __repr__(self) -> str:
        return (
            f"Action(id={self.action_id!r}, type={self.action_type!r}, "
            f"target={self.target_node_id!r})"
        )


# ---------------------------------------------------------------------------
# Transition
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Transition:
    """A single probabilistic transition in the MDP.

    Represents T(target | source, action) with an associated action cost.

    Parameters
    ----------
    source : str
        Source state ID.
    action : str
        Action ID.
    target : str
        Target state ID.
    probability : float
        Transition probability in [0, 1].
    cost : float
        Non-negative cost incurred by this transition (cognitive + motor).
    """

    source: str
    action: str
    target: str
    probability: float
    cost: float = 0.0

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not 0.0 <= self.probability <= 1.0:
            errors.append(f"probability {self.probability} not in [0,1]")
        if self.cost < 0:
            errors.append(f"cost {self.cost} is negative")
        if not self.source:
            errors.append("source must be non-empty")
        if not self.action:
            errors.append("action must be non-empty")
        if not self.target:
            errors.append("target must be non-empty")
        return errors

    def __repr__(self) -> str:
        return (
            f"Transition({self.source!r} --{self.action!r}--> "
            f"{self.target!r}, p={self.probability:.3f}, c={self.cost:.3f})"
        )


# ---------------------------------------------------------------------------
# MDP
# ---------------------------------------------------------------------------

@dataclass
class MDP:
    """Finite Markov Decision Process for UI navigation.

    The transition structure is stored both as a flat list of
    :class:`Transition` objects and as a nested dict-of-dicts for O(1) lookup:

        ``transition_matrix[source][action] -> list[(target, probability, cost)]``

    Parameters
    ----------
    states : dict[str, State]
    actions : dict[str, Action]
    transitions : list[Transition]
    initial_state : str
    goal_states : set[str]
    discount : float
        Default discount factor γ ∈ (0, 1].
    """

    states: dict[str, State] = field(default_factory=dict)
    actions: dict[str, Action] = field(default_factory=dict)
    transitions: list[Transition] = field(default_factory=list)
    initial_state: str = ""
    goal_states: set[str] = field(default_factory=set)
    discount: float = 0.99

    # Sparse transition matrix: state -> action -> [(target, prob, cost)]
    transition_matrix: dict[str, dict[str, list[tuple[str, float, float]]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list)),
        repr=False,
    )

    # Pre-computed reverse adjacency
    _predecessors: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set), repr=False
    )

    def __post_init__(self) -> None:
        """Build sparse transition matrix from transition list."""
        self._rebuild_index()

    # ── Index management --------------------------------------------------

    def _rebuild_index(self) -> None:
        """(Re)build the transition matrix and predecessor map from *transitions*."""
        self.transition_matrix = defaultdict(lambda: defaultdict(list))
        self._predecessors = defaultdict(set)
        for t in self.transitions:
            self.transition_matrix[t.source][t.action].append(
                (t.target, t.probability, t.cost)
            )
            self._predecessors[t.target].add(t.source)

    def add_transition(self, transition: Transition) -> None:
        """Append a transition and update indices."""
        self.transitions.append(transition)
        self.transition_matrix[transition.source][transition.action].append(
            (transition.target, transition.probability, transition.cost)
        )
        self._predecessors[transition.target].add(transition.source)

    # ── Queries -----------------------------------------------------------

    def get_actions(self, state_id: str) -> list[str]:
        """Return action IDs available from *state_id*."""
        return list(self.transition_matrix.get(state_id, {}).keys())

    def get_transitions(
        self, state_id: str, action_id: str
    ) -> list[tuple[str, float, float]]:
        """Return ``[(target, probability, cost)]`` for *(state, action)*."""
        return self.transition_matrix.get(state_id, {}).get(action_id, [])

    def get_successors(self, state_id: str) -> set[str]:
        """Return set of states reachable in one step from *state_id*."""
        succs: set[str] = set()
        for action_outcomes in self.transition_matrix.get(state_id, {}).values():
            for target, _p, _c in action_outcomes:
                succs.add(target)
        return succs

    def get_predecessors(self, state_id: str) -> set[str]:
        """Return set of states that can reach *state_id* in one step."""
        return set(self._predecessors.get(state_id, set()))

    def is_reachable(self, state_id: str) -> bool:
        """True if *state_id* is reachable from the initial state via BFS."""
        if state_id == self.initial_state:
            return True
        visited: set[str] = set()
        queue: deque[str] = deque([self.initial_state])
        while queue:
            s = queue.popleft()
            if s == state_id:
                return True
            if s in visited:
                continue
            visited.add(s)
            queue.extend(self.get_successors(s) - visited)
        return False

    def reachable_states(self) -> set[str]:
        """BFS from initial state; return set of reachable state IDs."""
        visited: set[str] = set()
        queue: deque[str] = deque([self.initial_state])
        while queue:
            s = queue.popleft()
            if s in visited:
                continue
            visited.add(s)
            queue.extend(self.get_successors(s) - visited)
        return visited

    # ── Size properties ---------------------------------------------------

    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    @property
    def n_transitions(self) -> int:
        return len(self.transitions)

    # ── NetworkX conversion -----------------------------------------------

    def to_networkx(self) -> Any:
        """Return a ``networkx.DiGraph`` representation of the MDP.

        Nodes carry State attributes; edges carry action, probability, and
        cost attributes.  Requires ``networkx`` to be installed.
        """
        import networkx as nx  # type: ignore[import-untyped]

        G = nx.DiGraph()
        for sid, state in self.states.items():
            G.add_node(
                sid,
                label=state.label,
                is_terminal=state.is_terminal,
                is_goal=state.is_goal,
                features=dict(state.features),
            )
        for t in self.transitions:
            G.add_edge(
                t.source,
                t.target,
                action=t.action,
                probability=t.probability,
                cost=t.cost,
            )
        return G

    # ── Validation --------------------------------------------------------

    def validate(self) -> list[str]:
        """Structural validation of the MDP.  Returns list of error strings."""
        errors: list[str] = []

        # Initial state must exist
        if self.initial_state and self.initial_state not in self.states:
            errors.append(f"initial_state {self.initial_state!r} not in states")

        # Goal states must exist
        for gs in self.goal_states:
            if gs not in self.states:
                errors.append(f"goal_state {gs!r} not in states")

        # Discount bounds
        if not 0.0 < self.discount <= 1.0:
            errors.append(f"discount {self.discount} not in (0, 1]")

        # Transitions reference valid states / actions
        for t in self.transitions:
            t_errors = t.validate()
            errors.extend(t_errors)
            if t.source not in self.states:
                errors.append(f"transition source {t.source!r} not in states")
            if t.target not in self.states:
                errors.append(f"transition target {t.target!r} not in states")
            if t.action not in self.actions:
                errors.append(f"transition action {t.action!r} not in actions")

        # Probability normalisation per (state, action)
        for sid in self.transition_matrix:
            for aid, outcomes in self.transition_matrix[sid].items():
                total = sum(p for _, p, _ in outcomes)
                if abs(total - 1.0) > 1e-6:
                    errors.append(
                        f"T({sid!r}, {aid!r}) probabilities sum to {total:.6f} ≠ 1"
                    )

        return errors

    # ── Statistics ---------------------------------------------------------

    def statistics(self) -> MDPStatistics:
        """Compute summary statistics about this MDP."""
        branching: list[int] = []
        for sid in self.states:
            succs = self.get_successors(sid)
            branching.append(len(succs))
        avg_branching = float(np.mean(branching)) if branching else 0.0

        # Approximate diameter via BFS from initial state
        diameter = 0
        if self.initial_state:
            visited: dict[str, int] = {}
            queue: deque[tuple[str, int]] = deque([(self.initial_state, 0)])
            while queue:
                s, d = queue.popleft()
                if s in visited:
                    continue
                visited[s] = d
                diameter = max(diameter, d)
                for succ in self.get_successors(s):
                    if succ not in visited:
                        queue.append((succ, d + 1))

        # Simple ergodicity check: all states reachable and all states can
        # reach at least one goal state
        reachable = self.reachable_states()
        all_reachable = len(reachable) == len(self.states)

        # Check if every reachable state can reach a goal state (reverse BFS)
        reverse_reachable: set[str] = set()
        rev_queue: deque[str] = deque(self.goal_states)
        while rev_queue:
            s = rev_queue.popleft()
            if s in reverse_reachable:
                continue
            reverse_reachable.add(s)
            for pred in self.get_predecessors(s):
                if pred not in reverse_reachable:
                    rev_queue.append(pred)
        can_reach_goal = reachable <= reverse_reachable
        is_ergodic = all_reachable and can_reach_goal

        return MDPStatistics(
            n_states=self.n_states,
            n_actions=self.n_actions,
            n_transitions=self.n_transitions,
            branching_factor=avg_branching,
            diameter=diameter,
            is_ergodic=is_ergodic,
        )

    def __repr__(self) -> str:
        return (
            f"MDP(states={self.n_states}, actions={self.n_actions}, "
            f"transitions={self.n_transitions}, γ={self.discount})"
        )


# ---------------------------------------------------------------------------
# MDPStatistics
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MDPStatistics:
    """Summary statistics of an MDP.

    Attributes
    ----------
    n_states : int
    n_actions : int
    n_transitions : int
    branching_factor : float
        Average out-degree (number of successors) per state.
    diameter : int
        Longest shortest-path distance from the initial state.
    is_ergodic : bool
        True if every state is reachable *and* can reach a goal state.
    """

    n_states: int
    n_actions: int
    n_transitions: int
    branching_factor: float
    diameter: int
    is_ergodic: bool

    def __repr__(self) -> str:
        return (
            f"MDPStatistics(|S|={self.n_states}, |A|={self.n_actions}, "
            f"|T|={self.n_transitions}, bf={self.branching_factor:.2f}, "
            f"diam={self.diameter}, ergodic={self.is_ergodic})"
        )

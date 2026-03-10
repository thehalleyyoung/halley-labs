"""
usability_oracle.mdp.hierarchical — Hierarchical MDP (options framework).

Implements the options framework for semi-MDPs, enabling temporal
abstraction over multi-step UI interactions.  An *option* is a
temporally extended action consisting of:

    ω = (I, π, β)

where:
  - I ⊆ S is the *initiation set* (states where the option can start),
  - π : S → A is the option's *internal policy*,
  - β : S → [0, 1] is the *termination condition*.

UI tasks naturally decompose into hierarchical sub-tasks:
  - Navigate to section → fill form → submit
  - Open menu → select item → confirm

References
----------
- Sutton, R. S., Precup, D. & Singh, S. (1999). Between MDPs and
  semi-MDPs: A framework for temporal abstraction in RL. *AIJ*.
- Dietterich, T. G. (2000). Hierarchical reinforcement learning with
  the MAXQ value function decomposition. *JAIR*.
- Barto, A. G. & Mahadevan, S. (2003). Recent advances in hierarchical
  reinforcement learning. *Discrete Event Dynamic Systems*.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

from usability_oracle.mdp.models import Action, MDP, State, Transition

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Option
# ---------------------------------------------------------------------------


@dataclass
class Option:
    """A temporally extended action (option) in the SMDP framework.

    An option ω = (I, π, β) consists of:
    - An initiation set I: states where the option may begin.
    - An internal policy π: mapping states to primitive actions.
    - A termination condition β: probability of terminating at each state.

    Parameters
    ----------
    option_id : str
        Unique identifier.
    name : str
        Human-readable name (e.g. "navigate_to_form").
    initiation_set : set[str]
        State IDs where the option can be initiated.
    policy : dict[str, str]
        Internal policy: state_id → action_id.
    termination : dict[str, float]
        Termination probabilities: state_id → β(s) ∈ [0, 1].
    description : str
        Description of what this option accomplishes.
    """

    option_id: str
    name: str = ""
    initiation_set: set[str] = field(default_factory=set)
    policy: dict[str, str] = field(default_factory=dict)
    termination: dict[str, float] = field(default_factory=dict)
    description: str = ""

    def can_initiate(self, state_id: str) -> bool:
        """Check if this option can be initiated in *state_id*."""
        return state_id in self.initiation_set

    def get_action(self, state_id: str) -> Optional[str]:
        """Return the action prescribed by the internal policy."""
        return self.policy.get(state_id)

    def termination_prob(self, state_id: str) -> float:
        """Return β(s): probability of termination in *state_id*."""
        return self.termination.get(state_id, 0.0)

    def is_terminal(self, state_id: str) -> bool:
        """True if the option terminates with certainty at *state_id*."""
        return self.termination.get(state_id, 0.0) >= 1.0 - 1e-10

    @property
    def n_states(self) -> int:
        return len(self.initiation_set)

    def validate(self) -> list[str]:
        """Validate option consistency."""
        errors: list[str] = []
        if not self.option_id:
            errors.append("option_id must be non-empty")
        if not self.initiation_set:
            errors.append(f"Option {self.option_id!r}: empty initiation set")
        for sid in self.initiation_set:
            if sid not in self.policy:
                errors.append(
                    f"Option {self.option_id!r}: state {sid!r} in "
                    f"initiation set but not in policy"
                )
        for sid, beta in self.termination.items():
            if not 0.0 <= beta <= 1.0:
                errors.append(
                    f"Option {self.option_id!r}: β({sid!r})={beta} "
                    f"not in [0,1]"
                )
        return errors

    def __repr__(self) -> str:
        return (
            f"Option({self.option_id!r}, name={self.name!r}, "
            f"|I|={len(self.initiation_set)})"
        )


# ---------------------------------------------------------------------------
# Option execution
# ---------------------------------------------------------------------------


@dataclass
class OptionExecution:
    """Record of a single option execution (trajectory segment).

    Parameters
    ----------
    option_id : str
    start_state : str
    end_state : str
    steps : list[tuple[str, str, str]]
        [(state, action, next_state), …]
    total_cost : float
    duration : int
        Number of primitive steps.
    """

    option_id: str
    start_state: str = ""
    end_state: str = ""
    steps: list[tuple[str, str, str]] = field(default_factory=list)
    total_cost: float = 0.0
    duration: int = 0

    def __repr__(self) -> str:
        return (
            f"OptionExecution({self.option_id!r}, "
            f"{self.start_state!r}→{self.end_state!r}, "
            f"steps={self.duration})"
        )


class OptionExecutor:
    """Execute options in an MDP environment.

    Simulates option execution by following the internal policy until
    the termination condition triggers or a step limit is reached.

    Parameters
    ----------
    mdp : MDP
    rng : np.random.Generator, optional
    """

    def __init__(
        self, mdp: MDP, rng: Optional[np.random.Generator] = None
    ) -> None:
        self.mdp = mdp
        self.rng = rng or np.random.default_rng()

    def execute(
        self, option: Option, start_state: str, max_steps: int = 100
    ) -> OptionExecution:
        """Execute an option from *start_state*.

        Parameters
        ----------
        option : Option
        start_state : str
        max_steps : int

        Returns
        -------
        OptionExecution
        """
        if not option.can_initiate(start_state):
            return OptionExecution(
                option_id=option.option_id,
                start_state=start_state,
                end_state=start_state,
            )

        current = start_state
        exec_record = OptionExecution(
            option_id=option.option_id, start_state=start_state
        )

        for _ in range(max_steps):
            # Check termination
            beta = option.termination_prob(current)
            if self.rng.random() < beta:
                break

            # Get action from option policy
            action_id = option.get_action(current)
            if action_id is None:
                break

            # Sample transition
            outcomes = self.mdp.get_transitions(current, action_id)
            if not outcomes:
                break

            targets = [t for t, _, _ in outcomes]
            probs = np.array([p for _, p, _ in outcomes], dtype=np.float64)
            costs = [c for _, _, c in outcomes]

            total = probs.sum()
            if total <= 0:
                break
            probs /= total

            idx = int(self.rng.choice(len(targets), p=probs))
            next_state = targets[idx]
            cost = costs[idx]

            exec_record.steps.append((current, action_id, next_state))
            exec_record.total_cost += cost
            exec_record.duration += 1

            current = next_state

            # Check if we've left the option's scope
            state_obj = self.mdp.states.get(current)
            if state_obj and (state_obj.is_terminal or state_obj.is_goal):
                break

        exec_record.end_state = current
        return exec_record


# ---------------------------------------------------------------------------
# Option discovery
# ---------------------------------------------------------------------------


class OptionDiscovery:
    """Discover options from MDP structure and task hierarchy.

    Methods:
    1. **Bottleneck states**: options connecting bottleneck states.
    2. **Sub-goal options**: options for each task sub-goal.
    3. **Graph partitioning**: options from state-space partitions.

    Parameters
    ----------
    mdp : MDP
    """

    def __init__(self, mdp: MDP) -> None:
        self.mdp = mdp

    def from_subgoals(
        self,
        subgoal_states: list[set[str]],
        values: Optional[dict[str, float]] = None,
    ) -> list[Option]:
        """Create options from a set of sub-goal state sets.

        Each option navigates from any reachable state to a sub-goal
        state set, using the greedy policy from value iteration.

        Parameters
        ----------
        subgoal_states : list[set[str]]
            Each set defines a sub-goal (e.g. states where a form field
            is filled).
        values : dict[str, float], optional
            Value function to derive the internal policy.

        Returns
        -------
        list[Option]
        """
        options: list[Option] = []

        for i, goal_set in enumerate(subgoal_states):
            # Compute values targeting this sub-goal
            if values is None:
                v = self._compute_subgoal_values(goal_set)
            else:
                v = values

            # Extract greedy policy toward sub-goal
            policy, initiation = self._greedy_policy_to(goal_set, v)

            # Termination: deterministic at sub-goal states
            termination: dict[str, float] = {}
            for sid in initiation:
                termination[sid] = 1.0 if sid in goal_set else 0.0

            option = Option(
                option_id=f"option_subgoal_{i}",
                name=f"reach_subgoal_{i}",
                initiation_set=initiation,
                policy=policy,
                termination=termination,
                description=f"Navigate to sub-goal {i}",
            )
            options.append(option)

        return options

    def from_bottlenecks(
        self, n_options: int = 5
    ) -> list[Option]:
        """Discover options based on bottleneck (betweenness) analysis.

        Identifies high-betweenness states and creates options to
        navigate between them.

        Parameters
        ----------
        n_options : int
            Number of options to create.

        Returns
        -------
        list[Option]
        """
        bottlenecks = self._find_bottleneck_states(n_options + 1)

        options: list[Option] = []
        for i, target in enumerate(bottlenecks[:n_options]):
            goal_set = {target}
            values = self._compute_subgoal_values(goal_set)
            policy, initiation = self._greedy_policy_to(goal_set, values)

            termination = {sid: (1.0 if sid == target else 0.0) for sid in initiation}

            option = Option(
                option_id=f"option_bottleneck_{i}",
                name=f"reach_{target}",
                initiation_set=initiation,
                policy=policy,
                termination=termination,
                description=f"Navigate to bottleneck state {target}",
            )
            options.append(option)

        return options

    def _compute_subgoal_values(
        self, goal_set: set[str]
    ) -> dict[str, float]:
        """BFS-based distance values: V(s) = −distance(s, goal_set)."""
        distances: dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque()
        for g in goal_set:
            distances[g] = 0
            queue.append((g, 0))

        while queue:
            sid, d = queue.popleft()
            for pred in self.mdp.get_predecessors(sid):
                if pred not in distances:
                    distances[pred] = d + 1
                    queue.append((pred, d + 1))

        max_dist = max(distances.values()) if distances else 1
        return {sid: -d / max(max_dist, 1) for sid, d in distances.items()}

    def _greedy_policy_to(
        self,
        goal_set: set[str],
        values: dict[str, float],
    ) -> tuple[dict[str, str], set[str]]:
        """Extract greedy policy toward a goal set from values."""
        policy: dict[str, str] = {}
        initiation: set[str] = set()

        for sid in self.mdp.states:
            if self.mdp.states[sid].is_terminal:
                continue
            actions = self.mdp.get_actions(sid)
            if not actions:
                continue

            best_action = actions[0]
            best_val = -math.inf
            for aid in actions:
                q = 0.0
                for target, prob, cost in self.mdp.get_transitions(sid, aid):
                    q += prob * (values.get(target, -100.0) - cost)
                if q > best_val:
                    best_val = q
                    best_action = aid

            policy[sid] = best_action
            initiation.add(sid)

        # Add goal states with self-loop
        for g in goal_set:
            if g in self.mdp.states:
                initiation.add(g)
                actions = self.mdp.get_actions(g)
                if actions:
                    policy[g] = actions[0]

        return policy, initiation

    def _find_bottleneck_states(self, n: int) -> list[str]:
        """Find states with high betweenness centrality (approximate).

        Uses BFS-based betweenness approximation from a sample of
        source states.
        """
        states = list(self.mdp.states.keys())
        if len(states) <= n:
            return states

        betweenness: dict[str, float] = defaultdict(float)

        # Sample source states for efficiency
        rng = np.random.default_rng(42)
        sample_size = min(50, len(states))
        sources = [states[i] for i in rng.choice(len(states), sample_size, replace=False)]

        for src in sources:
            # BFS from src
            pred: dict[str, list[str]] = defaultdict(list)
            dist: dict[str, int] = {src: 0}
            sigma: dict[str, int] = defaultdict(int)
            sigma[src] = 1
            queue: deque[str] = deque([src])
            order: list[str] = []

            while queue:
                s = queue.popleft()
                order.append(s)
                for succ in self.mdp.get_successors(s):
                    if succ not in dist:
                        dist[succ] = dist[s] + 1
                        queue.append(succ)
                    if dist.get(succ, -1) == dist[s] + 1:
                        sigma[succ] += sigma[s]
                        pred[succ].append(s)

            # Backtrack to accumulate betweenness
            delta: dict[str, float] = defaultdict(float)
            for s in reversed(order):
                for p in pred[s]:
                    frac = sigma[p] / max(sigma[s], 1)
                    delta[p] += frac * (1.0 + delta[s])
                if s != src:
                    betweenness[s] += delta[s]

        # Top-n by betweenness
        sorted_states = sorted(betweenness.items(), key=lambda x: -x[1])
        return [s for s, _ in sorted_states[:n]]


# ---------------------------------------------------------------------------
# Hierarchical value iteration
# ---------------------------------------------------------------------------


class HierarchicalValueIteration:
    """Value iteration over semi-MDP with options.

    Extends standard value iteration to handle temporally extended
    actions (options) alongside primitive actions.

    The Bellman equation for options:

        V*(s) = max_{ω ∈ Ω_s} [ R(s, ω) + γ^k Σ_{s'} P(s', k | s, ω) V*(s') ]

    where k is the option duration and P(s', k | s, ω) is the
    multi-step transition probability.

    Parameters
    ----------
    mdp : MDP
    options : list[Option]
    include_primitives : bool
        Whether to also consider primitive actions.
    """

    def __init__(
        self,
        mdp: MDP,
        options: list[Option],
        include_primitives: bool = True,
    ) -> None:
        self.mdp = mdp
        self.options = options
        self.include_primitives = include_primitives
        self._executor = OptionExecutor(mdp)

        # Pre-compute option models: P(s'|s,ω) and R(s,ω)
        self._option_models: dict[str, dict[str, list[tuple[str, float, float, int]]]] = {}
        self._precompute_option_models()

    def _precompute_option_models(self, n_samples: int = 200) -> None:
        """Estimate option transition and reward models via Monte Carlo.

        For each option ω and initiation state s, estimates:
        - P(s' | s, ω): distribution over termination states
        - R(s, ω): expected cumulative cost
        - k(s, ω): expected duration
        """
        rng = np.random.default_rng(42)
        executor = OptionExecutor(self.mdp, rng)

        for option in self.options:
            self._option_models[option.option_id] = {}

            for sid in option.initiation_set:
                outcomes: dict[str, list[tuple[float, int]]] = defaultdict(list)

                for _ in range(n_samples):
                    exec_result = executor.execute(option, sid)
                    end = exec_result.end_state
                    outcomes[end].append(
                        (exec_result.total_cost, exec_result.duration)
                    )

                # Aggregate
                model: list[tuple[str, float, float, int]] = []
                for end_state, runs in outcomes.items():
                    prob = len(runs) / n_samples
                    avg_cost = sum(c for c, _ in runs) / len(runs)
                    avg_dur = int(round(sum(d for _, d in runs) / len(runs)))
                    model.append((end_state, prob, avg_cost, max(avg_dur, 1)))

                self._option_models[option.option_id][sid] = model

    def solve(
        self,
        discount: Optional[float] = None,
        epsilon: float = 1e-6,
        max_iter: int = 5_000,
    ) -> tuple[dict[str, float], dict[str, str]]:
        """Run hierarchical value iteration.

        Returns
        -------
        values : dict[str, float]
            Optimal value function.
        policy : dict[str, str]
            Policy mapping state → option_id or action_id.
        """
        gamma = discount if discount is not None else self.mdp.discount

        values: dict[str, float] = {sid: 0.0 for sid in self.mdp.states}

        for iteration in range(max_iter):
            new_values: dict[str, float] = {}

            for sid, state in self.mdp.states.items():
                if state.is_terminal or state.is_goal:
                    new_values[sid] = 0.0
                    continue

                best_val = math.inf
                found_action = False

                # Primitive actions
                if self.include_primitives:
                    for aid in self.mdp.get_actions(sid):
                        q = 0.0
                        for target, prob, cost in self.mdp.get_transitions(sid, aid):
                            q += prob * (cost + gamma * values.get(target, 0.0))
                        best_val = min(best_val, q)
                        found_action = True

                # Options
                for option in self.options:
                    if not option.can_initiate(sid):
                        continue

                    model = self._option_models.get(
                        option.option_id, {}
                    ).get(sid, [])

                    if not model:
                        continue

                    q = 0.0
                    for end_state, prob, cost, duration in model:
                        discounted = (gamma ** duration) * values.get(end_state, 0.0)
                        q += prob * (cost + discounted)

                    best_val = min(best_val, q)
                    found_action = True

                new_values[sid] = best_val if found_action else values.get(sid, 0.0)

            # Check convergence
            max_diff = max(
                abs(new_values[s] - values[s]) for s in self.mdp.states
            )
            values = new_values

            if max_diff < epsilon:
                logger.info(
                    "Hierarchical VI converged in %d iterations", iteration + 1
                )
                break

        # Extract policy
        policy = self._extract_policy(values, gamma)
        return values, policy

    def _extract_policy(
        self, values: dict[str, float], discount: float
    ) -> dict[str, str]:
        """Extract the greedy policy from values (options + primitives)."""
        policy: dict[str, str] = {}

        for sid, state in self.mdp.states.items():
            if state.is_terminal or state.is_goal:
                continue

            best_action = ""
            best_val = math.inf

            # Primitive actions
            if self.include_primitives:
                for aid in self.mdp.get_actions(sid):
                    q = 0.0
                    for target, prob, cost in self.mdp.get_transitions(sid, aid):
                        q += prob * (cost + discount * values.get(target, 0.0))
                    if q < best_val:
                        best_val = q
                        best_action = aid

            # Options
            for option in self.options:
                if not option.can_initiate(sid):
                    continue
                model = self._option_models.get(
                    option.option_id, {}
                ).get(sid, [])
                if not model:
                    continue

                q = 0.0
                for end_state, prob, cost, duration in model:
                    q += prob * (cost + (discount ** duration) * values.get(end_state, 0.0))

                if q < best_val:
                    best_val = q
                    best_action = option.option_id

            if best_action:
                policy[sid] = best_action

        return policy


# ---------------------------------------------------------------------------
# MAXQ decomposition
# ---------------------------------------------------------------------------


@dataclass
class MAXQNode:
    """A node in the MAXQ task hierarchy.

    Internal nodes represent composite tasks; leaves are primitive actions.

    Parameters
    ----------
    node_id : str
    name : str
    children : list[str]
        Child node IDs (empty for primitives).
    is_primitive : bool
    termination_predicate : set[str]
        States where this sub-task is considered complete.
    """

    node_id: str
    name: str = ""
    children: list[str] = field(default_factory=list)
    is_primitive: bool = False
    termination_predicate: set[str] = field(default_factory=set)

    @property
    def is_composite(self) -> bool:
        return not self.is_primitive

    def __repr__(self) -> str:
        kind = "primitive" if self.is_primitive else "composite"
        return f"MAXQNode({self.node_id!r}, {kind}, children={self.children})"


@dataclass
class MAXQDecomposition:
    """MAXQ value function decomposition.

    Decomposes V(s) into:
        Q(i, s, a) = V(a, s) + C(i, s, a)

    where V(a, s) is the value of completing sub-task a from s,
    and C(i, s, a) is the completion function (cost to finish task i
    after sub-task a completes).

    Parameters
    ----------
    nodes : dict[str, MAXQNode]
    root : str
        Root task node ID.
    mdp : MDP
    """

    nodes: dict[str, MAXQNode] = field(default_factory=dict)
    root: str = ""
    mdp: Optional[MDP] = None

    # V(node_id, state_id) -> value
    V: dict[str, dict[str, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    # C(parent_id, state_id, child_id) -> completion cost
    C: dict[str, dict[str, dict[str, float]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(dict))
    )

    def add_node(self, node: MAXQNode) -> None:
        """Add a task node to the hierarchy."""
        self.nodes[node.node_id] = node

    def solve(
        self,
        discount: Optional[float] = None,
        epsilon: float = 1e-6,
        max_iter: int = 200,
    ) -> None:
        """Solve the MAXQ decomposition via recursive value iteration.

        Computes V(i, s) for all nodes i and states s, bottom-up.
        """
        if self.mdp is None:
            raise RuntimeError("MDP not set for MAXQ decomposition")

        gamma = discount if discount is not None else self.mdp.discount

        # Topological sort (bottom-up)
        order = self._topological_order()

        for iteration in range(max_iter):
            max_diff = 0.0

            for node_id in order:
                node = self.nodes[node_id]

                if node.is_primitive:
                    # V(primitive, s) = expected immediate reward
                    for sid in self.mdp.states:
                        old_v = self.V[node_id].get(sid, 0.0)
                        transitions = self.mdp.get_transitions(sid, node_id)
                        new_v = 0.0
                        for _target, prob, cost in transitions:
                            new_v -= prob * cost
                        self.V[node_id][sid] = new_v
                        max_diff = max(max_diff, abs(new_v - old_v))
                else:
                    # V(composite, s) = max_child [V(child, s) + C(node, s, child)]
                    for sid in self.mdp.states:
                        if sid in node.termination_predicate:
                            self.V[node_id][sid] = 0.0
                            continue

                        old_v = self.V[node_id].get(sid, 0.0)
                        best = -math.inf

                        for child_id in node.children:
                            child_v = self.V[child_id].get(sid, 0.0)
                            c = self.C[node_id][sid].get(child_id, 0.0)
                            total = child_v + c
                            if total > best:
                                best = total

                        new_v = best if best > -math.inf else 0.0
                        self.V[node_id][sid] = new_v
                        max_diff = max(max_diff, abs(new_v - old_v))

                    # Update completion functions
                    self._update_completion(node_id, gamma)

            if max_diff < epsilon:
                logger.info("MAXQ converged in %d iterations", iteration + 1)
                break

    def _update_completion(
        self, parent_id: str, discount: float
    ) -> None:
        """Update completion function C(parent, s, child) for one parent."""
        if self.mdp is None:
            return

        parent = self.nodes[parent_id]

        for child_id in parent.children:
            child_node = self.nodes.get(child_id)
            if child_node is None:
                continue

            for sid in self.mdp.states:
                # C(parent, s, child) = expected cost-to-go after child
                # terminates, under parent's policy
                if child_node.is_primitive:
                    # After primitive action, we're in the next state
                    transitions = self.mdp.get_transitions(sid, child_id)
                    c = 0.0
                    for target, prob, cost in transitions:
                        v_parent = self.V[parent_id].get(target, 0.0)
                        c += prob * (discount * v_parent)
                    self.C[parent_id][sid][child_id] = c
                else:
                    # After composite child terminates, use parent value
                    c = 0.0
                    for term_s in child_node.termination_predicate:
                        v_parent = self.V[parent_id].get(term_s, 0.0)
                        c = max(c, discount * v_parent)
                    self.C[parent_id][sid][child_id] = c

    def _topological_order(self) -> list[str]:
        """Return nodes in bottom-up topological order."""
        visited: set[str] = set()
        order: list[str] = []

        def dfs(nid: str) -> None:
            if nid in visited:
                return
            visited.add(nid)
            node = self.nodes.get(nid)
            if node:
                for child in node.children:
                    dfs(child)
            order.append(nid)

        if self.root:
            dfs(self.root)
        else:
            for nid in self.nodes:
                dfs(nid)

        return order

    def get_policy(self) -> dict[str, str]:
        """Extract a flat policy from the MAXQ decomposition.

        For each state, follows the hierarchy to find the best
        primitive action.

        Returns
        -------
        dict[str, str]
            state_id → action_id (primitive)
        """
        if self.mdp is None:
            return {}

        policy: dict[str, str] = {}
        for sid in self.mdp.states:
            action = self._get_action(self.root, sid)
            if action:
                policy[sid] = action
        return policy

    def _get_action(self, node_id: str, state_id: str) -> Optional[str]:
        """Recursively find the best primitive action from a task node."""
        node = self.nodes.get(node_id)
        if node is None:
            return None

        if node.is_primitive:
            return node_id

        # Pick best child
        best_child = ""
        best_val = -math.inf
        for child_id in node.children:
            child_v = self.V[child_id].get(state_id, 0.0)
            c = self.C[node_id][state_id].get(child_id, 0.0)
            total = child_v + c
            if total > best_val:
                best_val = total
                best_child = child_id

        if best_child:
            return self._get_action(best_child, state_id)
        return None


# ---------------------------------------------------------------------------
# Task hierarchy to option mapping
# ---------------------------------------------------------------------------


class TaskToOptionMapper:
    """Map a task hierarchy (MAXQ) to the options framework.

    Converts each composite MAXQ node into an :class:`Option` with:
    - Initiation set: states not in the termination predicate
    - Policy: derived from the MAXQ decomposition
    - Termination: 1.0 at states in the termination predicate

    Parameters
    ----------
    mdp : MDP
    maxq : MAXQDecomposition
    """

    def __init__(self, mdp: MDP, maxq: MAXQDecomposition) -> None:
        self.mdp = mdp
        self.maxq = maxq

    def to_options(self) -> list[Option]:
        """Convert all composite MAXQ nodes to options.

        Returns
        -------
        list[Option]
        """
        options: list[Option] = []

        for nid, node in self.maxq.nodes.items():
            if node.is_primitive:
                continue

            # Initiation set: all states not at termination
            initiation = {
                sid for sid in self.mdp.states
                if sid not in node.termination_predicate
            }

            # Policy from MAXQ
            policy: dict[str, str] = {}
            for sid in initiation:
                action = self.maxq._get_action(nid, sid)
                if action:
                    policy[sid] = action

            # Termination
            termination: dict[str, float] = {}
            for sid in self.mdp.states:
                termination[sid] = 1.0 if sid in node.termination_predicate else 0.0

            option = Option(
                option_id=f"option_{nid}",
                name=node.name or nid,
                initiation_set=initiation,
                policy=policy,
                termination=termination,
                description=f"MAXQ task: {node.name or nid}",
            )
            options.append(option)

        return options

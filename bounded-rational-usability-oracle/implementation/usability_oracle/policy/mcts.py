"""
usability_oracle.policy.mcts — Monte Carlo Tree Search for bounded-rational planning.

Implements MCTS with information-cost-aware exploration for bounded-rational
agents.  The standard UCB1 selection rule is augmented with a mutual-information
penalty that discourages actions requiring excessive cognitive processing:

    UCB(a|s) = Q̄(a) + c·√(ln N / n_a) − (1/β)·I(a; s)

where I(a; s) is the pointwise mutual information between action a and state s
under the prior, β is the rationality parameter, and c is the exploration
constant.

Also supports PUCT (Polynomial Upper Confidence Trees) as used in AlphaZero,
progressive widening for large action spaces, RAVE (Rapid Action Value
Estimation), tree parallelism strategies, early termination, and task
completion time estimation.

References
----------
- Kocsis, L. & Szepesvári, C. (2006). Bandit based Monte-Carlo planning. *ECML*.
- Silver, D. et al. (2017). Mastering the game of Go without human knowledge.
  *Nature*, 550, 354–359.
- Browne, C. et al. (2012). A survey of Monte Carlo tree search methods.
  *IEEE Trans. CI and AI in Games*, 4(1), 1–43.
- Ortega, P. A. & Braun, D. A. (2013). Thermodynamics as a theory of
  decision-making with information-processing costs. *Proc. R. Soc. A*.
- Coulom, R. (2007). Efficient selectivity and backup operators in
  Monte-Carlo tree search. *CG*.
- Gelly, S. & Silver, D. (2007). Combining online and offline knowledge in
  UCT. *ICML*.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np

from usability_oracle.policy.models import Policy, QValues

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MCTSConfig:
    """Configuration for MCTS search.

    Attributes
    ----------
    n_simulations : int
        Number of MCTS simulations to run per search call.
    exploration_constant : float
        UCB exploration constant *c*.
    beta : float
        Rationality parameter (inverse temperature).  Controls the weight
        of the information cost penalty in the selection rule.
    discount : float
        Discount factor γ for rollout returns.
    max_rollout_depth : int
        Maximum number of steps in a simulation rollout.
    use_puct : bool
        If True, use the PUCT variant (prior-guided UCT) instead of UCB1.
    puct_constant : float
        Exploration constant for PUCT: ``c_puct · P(a|s) · √N / (1 + n_a)``.
    progressive_widening : bool
        Enable progressive widening for large action spaces.
    pw_alpha : float
        Progressive widening exponent: expand when ``|children| < N^α``.
    pw_constant : float
        Progressive widening constant *C*: expand when ``|children| < C · N^α``.
    rave_enabled : bool
        Enable RAVE (Rapid Action Value Estimation).
    rave_equivalence : float
        RAVE equivalence parameter *k*: weight = k / (3N + k).
    early_termination_threshold : float
        Stop search early when the best action's visit share exceeds this.
    """

    n_simulations: int = 1000
    exploration_constant: float = 1.414
    beta: float = 1.0
    discount: float = 0.99
    max_rollout_depth: int = 100
    use_puct: bool = False
    puct_constant: float = 2.5
    progressive_widening: bool = False
    pw_alpha: float = 0.5
    pw_constant: float = 1.0
    rave_enabled: bool = False
    rave_equivalence: float = 3000.0
    early_termination_threshold: float = 0.95


# ---------------------------------------------------------------------------
# Simulator protocol — callers supply domain dynamics
# ---------------------------------------------------------------------------

class Simulator(Protocol):
    """Interface for the environment simulator used by MCTS rollouts."""

    def get_actions(self, state: str) -> list[str]:
        """Return available action IDs in *state*."""
        ...

    def step(self, state: str, action: str) -> tuple[str, float, bool]:
        """Take *action* in *state*, returning ``(next_state, cost, done)``."""
        ...


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

@dataclass
class MCTSNode:
    """A node in the MCTS search tree.

    Attributes
    ----------
    state : str
        State identifier.
    parent : MCTSNode or None
    parent_action : str or None
        Action taken from parent to reach this node.
    children : dict[str, MCTSNode]
        Mapping ``action_id → child_node``.
    visit_count : int
    total_value : float
        Sum of backed-up returns (lower is better for costs).
    prior_prob : float
        Prior probability of the action leading to this node.
    is_terminal : bool
    untried_actions : list[str]
        Actions not yet expanded from this node.
    rave_visits : dict[str, int]
        RAVE visit counts per action.
    rave_values : dict[str, float]
        RAVE cumulative values per action.
    """

    state: str
    parent: Optional["MCTSNode"] = None
    parent_action: Optional[str] = None
    children: dict[str, "MCTSNode"] = field(default_factory=dict)
    visit_count: int = 0
    total_value: float = 0.0
    prior_prob: float = 1.0
    is_terminal: bool = False
    untried_actions: list[str] = field(default_factory=list)
    rave_visits: dict[str, int] = field(default_factory=dict)
    rave_values: dict[str, float] = field(default_factory=dict)

    @property
    def mean_value(self) -> float:
        """Average backed-up value (cost).  Lower is better."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


# ---------------------------------------------------------------------------
# Selection policies
# ---------------------------------------------------------------------------

def ucb1_score(
    child: MCTSNode,
    parent_visits: int,
    c: float,
    beta: float,
    prior_probs: dict[str, float],
) -> float:
    """Information-cost-aware UCB1 score.

    UCB(a) = Q̄(a) + c·√(ln N / n_a) − (1/β)·I(a; s)

    We negate Q̄ because Q̄ stores costs (lower is better) and UCB selects
    the *maximum* score.  The information cost term I(a; s) penalises
    actions that deviate from the prior.

    Parameters
    ----------
    child : MCTSNode
    parent_visits : int
    c : float
        Exploration constant.
    beta : float
        Rationality parameter.
    prior_probs : dict[str, float]
        Prior policy p₀(a|s).

    Returns
    -------
    float
    """
    if child.visit_count == 0:
        return float("inf")

    exploitation = -child.mean_value  # negate cost → higher is better
    exploration = c * math.sqrt(math.log(parent_visits) / child.visit_count)

    # Information cost: I(a;s) = log(π(a|s)/p₀(a|s))
    # Since we don't know π yet, approximate with visit frequency
    info_cost = 0.0
    if beta > 0 and child.parent_action is not None:
        p0 = prior_probs.get(child.parent_action, 1e-10)
        visit_freq = child.visit_count / max(parent_visits, 1)
        if visit_freq > 0 and p0 > 0:
            info_cost = visit_freq * math.log(visit_freq / max(p0, 1e-10))
        info_cost /= beta

    return exploitation + exploration - info_cost


def puct_score(
    child: MCTSNode,
    parent_visits: int,
    c_puct: float,
    beta: float,
) -> float:
    """PUCT selection score (AlphaZero-style).

    PUCT(a) = −Q̄(a) + c_puct · P(a) · √N / (1 + n_a) − (1/β)·info_cost

    Parameters
    ----------
    child : MCTSNode
    parent_visits : int
    c_puct : float
    beta : float

    Returns
    -------
    float
    """
    exploitation = -child.mean_value
    exploration = (
        c_puct * child.prior_prob * math.sqrt(parent_visits) / (1.0 + child.visit_count)
    )

    # Temperature-dependent penalty
    info_cost = 0.0
    if beta > 0 and child.visit_count > 0:
        visit_freq = child.visit_count / max(parent_visits, 1)
        if visit_freq > 0 and child.prior_prob > 0:
            info_cost = visit_freq * math.log(
                visit_freq / max(child.prior_prob, 1e-10)
            )
        info_cost /= beta

    return exploitation + exploration - info_cost


def rave_score(
    child: MCTSNode,
    parent_visits: int,
    c: float,
    k: float,
) -> float:
    """RAVE-blended UCB score.

    Blends tree Q-values with RAVE (all-moves-as-first) estimates:

        Q_blend = (1 − w)·Q_tree + w·Q_rave

    where w = k / (3·n_a + k).

    Parameters
    ----------
    child : MCTSNode
    parent_visits : int
    c : float
        Exploration constant.
    k : float
        Equivalence parameter controlling RAVE weighting.

    Returns
    -------
    float
    """
    if child.visit_count == 0:
        return float("inf")

    # Tree value
    q_tree = -child.mean_value

    # RAVE value
    action = child.parent_action
    parent = child.parent
    q_rave = q_tree  # fallback
    if parent is not None and action is not None:
        rave_n = parent.rave_visits.get(action, 0)
        if rave_n > 0:
            q_rave = -parent.rave_values.get(action, 0.0) / rave_n

    # Blend weight
    weight = k / (3.0 * child.visit_count + k)
    q_blend = (1.0 - weight) * q_tree + weight * q_rave

    exploration = c * math.sqrt(math.log(parent_visits) / child.visit_count)
    return q_blend + exploration


# ---------------------------------------------------------------------------
# MCTS search
# ---------------------------------------------------------------------------

class BoundedRationalMCTS:
    """Monte Carlo Tree Search with bounded-rational exploration.

    Combines standard MCTS phases (selection, expansion, simulation,
    back-propagation) with information-cost-aware exploration to model
    a user with limited cognitive capacity.

    Parameters
    ----------
    simulator : Simulator
        Domain simulator providing ``get_actions`` and ``step``.
    config : MCTSConfig
        Search configuration.
    prior_policy : Policy or None
        Prior policy p₀ for the information cost term.
    rng : np.random.Generator, optional
    """

    def __init__(
        self,
        simulator: Simulator,
        config: MCTSConfig,
        prior_policy: Optional[Policy] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.simulator = simulator
        self.config = config
        self.prior_policy = prior_policy
        self.rng = rng or np.random.default_rng()

    # ── Public API --------------------------------------------------------

    def search(self, root_state: str) -> dict[str, float]:
        """Run MCTS from *root_state* and return action visit counts.

        Parameters
        ----------
        root_state : str

        Returns
        -------
        dict[str, float]
            Mapping ``action_id → visit_fraction`` (normalised).
        """
        root = self._create_node(root_state)

        for sim_idx in range(self.config.n_simulations):
            node = self._select(root)
            node = self._expand(node)
            value = self._simulate(node)
            self._backpropagate(node, value)

            # Early termination
            if self._should_terminate_early(root, sim_idx + 1):
                logger.debug("MCTS early termination at simulation %d", sim_idx + 1)
                break

        return self._action_distribution(root)

    def search_policy(self, root_state: str, temperature: float = 1.0) -> Policy:
        """Run MCTS and return a :class:`Policy` for the root state.

        Parameters
        ----------
        root_state : str
        temperature : float
            Softmax temperature over visit counts.  τ → 0 gives greedy;
            τ = 1 is proportional to visits.

        Returns
        -------
        Policy
        """
        visit_counts = self.search(root_state)
        if not visit_counts:
            return Policy()

        actions = list(visit_counts.keys())
        counts = np.array([visit_counts[a] for a in actions], dtype=np.float64)

        if temperature < 1e-8:
            # Greedy
            probs = np.zeros_like(counts)
            probs[np.argmax(counts)] = 1.0
        else:
            # Temperature-scaled softmax
            log_counts = np.log(np.maximum(counts, 1e-10)) / temperature
            log_counts -= np.max(log_counts)
            probs = np.exp(log_counts)
            probs /= probs.sum()

        dist = {actions[i]: float(probs[i]) for i in range(len(actions))}
        return Policy(
            state_action_probs={root_state: dist},
            beta=self.config.beta,
        )

    def estimate_completion_time(
        self,
        root_state: str,
        goal_states: set[str],
        n_rollouts: int = 200,
    ) -> dict[str, float]:
        """Estimate task completion time from *root_state* using MCTS rollouts.

        Returns statistics (mean, std, median, p90) over the rollout
        step counts to reach any goal state.

        Parameters
        ----------
        root_state : str
        goal_states : set[str]
        n_rollouts : int

        Returns
        -------
        dict[str, float]
            Keys: ``mean``, ``std``, ``median``, ``p90``, ``completion_rate``.
        """
        step_counts: list[int] = []
        completions = 0

        for _ in range(n_rollouts):
            state = root_state
            steps = 0
            for _ in range(self.config.max_rollout_depth):
                actions = self.simulator.get_actions(state)
                if not actions:
                    break
                action = self._rollout_action(state, actions)
                state, _cost, done = self.simulator.step(state, action)
                steps += 1
                if done or state in goal_states:
                    completions += 1
                    break
            step_counts.append(steps)

        arr = np.array(step_counts, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "completion_rate": completions / max(n_rollouts, 1),
        }

    # ── Phase 1: Selection ------------------------------------------------

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Traverse the tree following the selection policy until a leaf.

        Selection uses either UCB1, PUCT, or RAVE depending on config.
        """
        while not node.is_terminal and node.is_fully_expanded and node.children:
            node = self._best_child(node)
        return node

    def _best_child(self, node: MCTSNode) -> MCTSNode:
        """Select the child with the highest selection score."""
        prior_probs = self._get_prior(node.state)

        best_score = -float("inf")
        best_child: Optional[MCTSNode] = None

        for child in node.children.values():
            if self.config.rave_enabled:
                score = rave_score(
                    child,
                    node.visit_count,
                    self.config.exploration_constant,
                    self.config.rave_equivalence,
                )
            elif self.config.use_puct:
                score = puct_score(
                    child,
                    node.visit_count,
                    self.config.puct_constant,
                    self.config.beta,
                )
            else:
                score = ucb1_score(
                    child,
                    node.visit_count,
                    self.config.exploration_constant,
                    self.config.beta,
                    prior_probs,
                )

            if score > best_score:
                best_score = score
                best_child = child

        assert best_child is not None
        return best_child

    # ── Phase 2: Expansion ------------------------------------------------

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand a leaf node by adding one child."""
        if node.is_terminal:
            return node

        if self.config.progressive_widening:
            if not self._should_widen(node):
                if node.children:
                    return self._best_child(node)
                return node

        if not node.untried_actions:
            return node

        # Select an untried action (biased toward prior)
        action = self._select_expansion_action(node)
        node.untried_actions.remove(action)

        # Simulate one step
        next_state, cost, done = self.simulator.step(node.state, action)

        # Create child node
        prior_probs = self._get_prior(node.state)
        child = MCTSNode(
            state=next_state,
            parent=node,
            parent_action=action,
            prior_prob=prior_probs.get(action, 1.0 / max(len(prior_probs), 1)),
            is_terminal=done,
            untried_actions=(
                list(self.simulator.get_actions(next_state)) if not done else []
            ),
        )
        node.children[action] = child
        return child

    def _should_widen(self, node: MCTSNode) -> bool:
        """Progressive widening criterion: expand if |children| < C · N^α."""
        if node.visit_count == 0:
            return True
        max_children = self.config.pw_constant * (
            node.visit_count ** self.config.pw_alpha
        )
        return len(node.children) < max_children

    def _select_expansion_action(self, node: MCTSNode) -> str:
        """Select which untried action to expand, biased by prior."""
        prior = self._get_prior(node.state)
        untried = node.untried_actions

        if not prior or not any(a in prior for a in untried):
            return str(self.rng.choice(untried))

        probs = np.array(
            [prior.get(a, 1e-10) for a in untried], dtype=np.float64
        )
        probs /= probs.sum()
        idx = self.rng.choice(len(untried), p=probs)
        return untried[idx]

    # ── Phase 3: Simulation (rollout) -------------------------------------

    def _simulate(self, node: MCTSNode) -> float:
        """Run a rollout from *node* to estimate its value.

        The rollout follows a capacity-limited policy: with probability
        proportional to the prior, otherwise uniform random.

        Returns
        -------
        float
            Discounted cumulative cost from the rollout.
        """
        if node.is_terminal:
            return 0.0

        state = node.state
        total_cost = 0.0
        discount_power = 1.0

        for _ in range(self.config.max_rollout_depth):
            actions = self.simulator.get_actions(state)
            if not actions:
                break

            action = self._rollout_action(state, actions)
            next_state, cost, done = self.simulator.step(state, action)

            total_cost += discount_power * cost
            discount_power *= self.config.discount

            if done:
                break
            state = next_state

        return total_cost

    def _rollout_action(self, state: str, actions: list[str]) -> str:
        """Select a rollout action using the capacity-limited policy.

        Blends the prior policy with uniform random, weighted by β.
        At β=0 (zero rationality), purely random; at β→∞, follows prior.
        """
        prior = self._get_prior(state)
        if not prior or self.config.beta <= 0:
            return str(self.rng.choice(actions))

        # Bounded-rational rollout: softmax over prior log-probs scaled by β
        log_probs = np.array(
            [math.log(max(prior.get(a, 1e-10), 1e-10)) for a in actions],
            dtype=np.float64,
        )
        scaled = self.config.beta * log_probs
        scaled -= np.max(scaled)
        probs = np.exp(scaled)
        total = probs.sum()
        if total <= 0:
            return str(self.rng.choice(actions))
        probs /= total
        idx = self.rng.choice(len(actions), p=probs)
        return actions[idx]

    # ── Phase 4: Backpropagation ------------------------------------------

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Propagate the rollout value up the tree.

        If RAVE is enabled, also update RAVE statistics for all actions
        encountered on the path from the rollout back to the root.
        """
        actions_on_path: list[str] = []
        current: Optional[MCTSNode] = node
        while current is not None:
            current.visit_count += 1
            current.total_value += value
            if current.parent_action is not None:
                actions_on_path.append(current.parent_action)
            current = current.parent

        # RAVE update: all actions on the path get RAVE credit at ancestors
        if self.config.rave_enabled:
            self._rave_update(node, actions_on_path, value)

    def _rave_update(
        self,
        leaf: MCTSNode,
        actions: list[str],
        value: float,
    ) -> None:
        """Update RAVE statistics for ancestor nodes."""
        action_set = set(actions)
        current = leaf.parent
        while current is not None:
            for a in action_set:
                current.rave_visits[a] = current.rave_visits.get(a, 0) + 1
                current.rave_values[a] = current.rave_values.get(a, 0.0) + value
            current = current.parent

    # ── Early termination -------------------------------------------------

    def _should_terminate_early(self, root: MCTSNode, n_sims: int) -> bool:
        """Terminate if the best action has a dominant visit share."""
        if n_sims < 50:
            return False
        if not root.children:
            return False

        visits = [ch.visit_count for ch in root.children.values()]
        total = sum(visits)
        if total == 0:
            return False

        max_share = max(visits) / total
        return max_share >= self.config.early_termination_threshold

    # ── Helpers -----------------------------------------------------------

    def _create_node(self, state: str) -> MCTSNode:
        """Create a root node for the given state."""
        actions = self.simulator.get_actions(state)
        return MCTSNode(
            state=state,
            untried_actions=list(actions),
        )

    def _get_prior(self, state: str) -> dict[str, float]:
        """Return prior action distribution for *state*."""
        if self.prior_policy is None:
            return {}
        return self.prior_policy.state_action_probs.get(state, {})

    def _action_distribution(self, root: MCTSNode) -> dict[str, float]:
        """Normalised visit counts at the root."""
        total = sum(ch.visit_count for ch in root.children.values())
        if total == 0:
            return {}
        return {
            action: child.visit_count / total
            for action, child in root.children.items()
        }


# ---------------------------------------------------------------------------
# Parallel MCTS helpers
# ---------------------------------------------------------------------------

@dataclass
class ParallelMCTSResult:
    """Aggregated result from parallel MCTS workers.

    Attributes
    ----------
    action_counts : dict[str, int]
        Total visit counts across all workers.
    action_values : dict[str, float]
        Total backed-up values across all workers.
    total_simulations : int
    """

    action_counts: dict[str, int] = field(default_factory=dict)
    action_values: dict[str, float] = field(default_factory=dict)
    total_simulations: int = 0


def aggregate_parallel_results(
    results: list[dict[str, float]],
) -> dict[str, float]:
    """Aggregate action distributions from multiple MCTS runs.

    Combines visit fractions from independent tree searches (leaf
    parallelisation / root parallelisation) by averaging.

    Parameters
    ----------
    results : list[dict[str, float]]
        List of per-worker normalised visit distributions.

    Returns
    -------
    dict[str, float]
        Averaged action distribution.
    """
    if not results:
        return {}

    all_actions: set[str] = set()
    for r in results:
        all_actions.update(r.keys())

    merged: dict[str, float] = {}
    n = len(results)
    for a in all_actions:
        merged[a] = sum(r.get(a, 0.0) for r in results) / n

    # Renormalise
    total = sum(merged.values())
    if total > 0:
        merged = {a: v / total for a, v in merged.items()}

    return merged


def run_parallel_mcts(
    simulator: Simulator,
    config: MCTSConfig,
    root_state: str,
    n_workers: int = 4,
    prior_policy: Optional[Policy] = None,
    seed: int = 42,
) -> dict[str, float]:
    """Run MCTS with root parallelisation (independent trees).

    Each worker builds an independent search tree from the same root state
    with a different random seed.  Results are aggregated by averaging
    the visit distributions.

    Parameters
    ----------
    simulator : Simulator
    config : MCTSConfig
    root_state : str
    n_workers : int
    prior_policy : Policy or None
    seed : int

    Returns
    -------
    dict[str, float]
        Aggregated action distribution.
    """
    per_worker = max(config.n_simulations // n_workers, 1)
    worker_config = MCTSConfig(
        n_simulations=per_worker,
        exploration_constant=config.exploration_constant,
        beta=config.beta,
        discount=config.discount,
        max_rollout_depth=config.max_rollout_depth,
        use_puct=config.use_puct,
        puct_constant=config.puct_constant,
        progressive_widening=config.progressive_widening,
        pw_alpha=config.pw_alpha,
        pw_constant=config.pw_constant,
        rave_enabled=config.rave_enabled,
        rave_equivalence=config.rave_equivalence,
        early_termination_threshold=config.early_termination_threshold,
    )

    results: list[dict[str, float]] = []
    for i in range(n_workers):
        rng = np.random.default_rng(seed + i)
        mcts = BoundedRationalMCTS(
            simulator=simulator,
            config=worker_config,
            prior_policy=prior_policy,
            rng=rng,
        )
        results.append(mcts.search(root_state))

    return aggregate_parallel_results(results)

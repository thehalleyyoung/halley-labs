"""
MCTS search engine for adversarial worst-case scenario discovery.

Implements the full Monte Carlo Tree Search loop (select, expand, rollout,
backpropagate) with integration to junction-tree inference, d-separation
pruning, causal UCB, PAC convergence monitoring, and progressive widening.
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import networkx as nx
except ImportError:
    nx = None  # type: ignore

from .causal_ucb import CausalUCB
from .convergence import ConvergenceMonitor
from .pruning import DSeparationPruner
from .rollout import RolloutPolicy, RolloutScheduler
from .tree_node import MCTSNode

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------

@dataclass
class SearchConfig:
    """Configuration for MCTS search."""

    n_rollouts: int = 10000
    budget_seconds: float = 300.0
    exploration_constant: float = 1.414
    maximize: bool = True
    ucb_variant: str = "ucb1"  # "ucb1" or "ucb_tuned"

    # PAC parameters
    epsilon: float = 0.05
    delta: float = 0.05

    # Progressive widening
    progressive_widening: bool = False
    pw_k_factor: float = 1.0
    pw_alpha: float = 0.5

    # Pruning
    enable_pruning: bool = True
    prune_interval: int = 100  # Re-prune every N rollouts

    # Convergence
    convergence_check_interval: int = 200
    snapshot_interval: int = 500

    # Batching
    batch_size: int = 1

    # Discretization
    values_per_variable: int = 10
    shock_range: Tuple[float, float] = (-3.0, 3.0)

    # Rollout policy
    rollout_policy: RolloutPolicy = RolloutPolicy.RANDOM
    rollout_epsilon: float = 0.1


@dataclass
class ScenarioReport:
    """Report for a discovered adversarial scenario."""

    state: Dict[str, float]
    value: float
    visit_count: int
    confidence_interval: Tuple[float, float]
    rank: int
    action_sequence: List[Tuple[str, float]]


@dataclass
class SearchResult:
    """Complete result from an MCTS search."""

    best_scenario: Optional[ScenarioReport]
    all_scenarios: List[ScenarioReport]
    converged: bool
    total_rollouts: int
    elapsed_seconds: float
    tree_stats: Dict[str, Any]
    convergence_stats: Dict[str, Any]
    pruning_stats: Dict[str, Any]
    rollout_stats: Dict[str, Any]


# -----------------------------------------------------------------------
# MCTSSearch
# -----------------------------------------------------------------------

class MCTSSearch:
    """
    Monte Carlo Tree Search engine for adversarial scenario discovery.

    Searches over the space of shock assignments to interface variables
    in a causal DAG, looking for assignments that maximize (or minimize)
    a target loss variable evaluated through junction-tree inference.

    Parameters
    ----------
    config : SearchConfig or None
        Search configuration. Uses defaults if None.
    dag : nx.DiGraph or None
        Causal DAG for d-separation pruning.
    random_seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        config: Optional[SearchConfig] = None,
        dag: Any = None,
        random_seed: Optional[int] = None,
    ) -> None:
        self.config = config if config is not None else SearchConfig()
        self._dag = dag
        self._rng = np.random.RandomState(random_seed)

        # Components
        self._ucb = CausalUCB(
            exploration_constant=self.config.exploration_constant,
            use_variance=(self.config.ucb_variant == "ucb_tuned"),
            maximize=self.config.maximize,
        )
        self._pruner = DSeparationPruner(dag=dag)
        self._convergence = ConvergenceMonitor(
            default_epsilon=self.config.epsilon,
            default_delta=self.config.delta,
        )
        self._rollout_scheduler = RolloutScheduler(
            default_policy=self.config.rollout_policy,
            epsilon=self.config.rollout_epsilon,
            random_seed=random_seed,
        )

        # Tree
        self._root: Optional[MCTSNode] = None
        self._target_variable: Optional[str] = None
        self._interface_vars: List[str] = []
        self._all_actions: List[Tuple[str, float]] = []

        # Tracking
        self._total_rollouts: int = 0
        self._best_value: float = float("-inf") if self.config.maximize else float("inf")
        self._best_state: Optional[Dict[str, float]] = None
        self._search_history: List[Dict[str, Any]] = []
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API: search
    # ------------------------------------------------------------------

    def search(
        self,
        interface_vars: List[str],
        inference_engine: Any,
        n_rollouts: Optional[int] = None,
        budget_seconds: Optional[float] = None,
        target_variable: Optional[str] = None,
        shock_values: Optional[Dict[str, List[float]]] = None,
    ) -> SearchResult:
        """
        Run MCTS search over interface variable shock assignments.

        Parameters
        ----------
        interface_vars : list of str
            Interface / separator variables to assign shocks to.
        inference_engine : object
            Junction-tree inference engine for rollout evaluation.
            Must support query(evidence, target) -> float.
        n_rollouts : int or None
            Maximum number of rollouts. Uses config default if None.
        budget_seconds : float or None
            Maximum wall-clock time. Uses config default if None.
        target_variable : str or None
            Target loss variable for inference queries.
        shock_values : dict or None
            Maps variable name to list of possible shock values.
            If None, discretizes shock_range uniformly.

        Returns
        -------
        SearchResult
            Complete search results with best scenario and diagnostics.
        """
        if n_rollouts is None:
            n_rollouts = self.config.n_rollouts
        if budget_seconds is None:
            budget_seconds = self.config.budget_seconds

        self._start_time = time.time()
        self._target_variable = target_variable
        self._interface_vars = list(interface_vars)

        # Build action space
        self._all_actions = self._build_action_space(
            interface_vars, shock_values
        )

        # Initialize rollout scheduler
        self._setup_rollout_scheduler(interface_vars, shock_values)

        # Initialize tree
        self._root = MCTSNode(available_actions=list(self._all_actions))

        # Precompute d-sep oracle if DAG is available
        if self._dag is not None and self.config.enable_pruning:
            self._pruner.precompute_dsep_oracle(self._dag)

        # Reset monitors
        self._convergence.reset()
        self._total_rollouts = 0
        self._best_value = float("-inf") if self.config.maximize else float("inf")
        self._best_state = None

        logger.info(
            "Starting MCTS search: %d interface vars, %d actions, "
            "budget=%d rollouts / %.0fs",
            len(interface_vars),
            len(self._all_actions),
            n_rollouts,
            budget_seconds,
        )

        # Main MCTS loop
        converged = False
        while self._total_rollouts < n_rollouts:
            elapsed = time.time() - self._start_time
            if elapsed >= budget_seconds:
                logger.info("Time budget exhausted after %d rollouts", self._total_rollouts)
                break

            # Run one MCTS iteration (or batch)
            if self.config.batch_size > 1:
                self._run_batch_iteration(inference_engine)
            else:
                self._run_single_iteration(inference_engine)

            # Periodic pruning
            if (
                self.config.enable_pruning
                and self._dag is not None
                and self._total_rollouts % self.config.prune_interval == 0
            ):
                self._run_pruning_pass()

            # Periodic convergence check
            if self._total_rollouts % self.config.convergence_check_interval == 0:
                if self._convergence.is_converged(
                    self.config.epsilon, self.config.delta
                ):
                    converged = True
                    logger.info(
                        "Converged after %d rollouts", self._total_rollouts
                    )
                    break

            # Periodic snapshot
            if self._total_rollouts % self.config.snapshot_interval == 0:
                self._convergence.record_snapshot(
                    self.config.epsilon, self.config.delta
                )

        elapsed = time.time() - self._start_time

        # Collect results
        best_scenario = self.get_best_scenario()
        all_scenarios = self.get_scenario_ranking(top_k=20)

        return SearchResult(
            best_scenario=best_scenario,
            all_scenarios=all_scenarios,
            converged=converged,
            total_rollouts=self._total_rollouts,
            elapsed_seconds=elapsed,
            tree_stats=self._root.subtree_statistics() if self._root else {},
            convergence_stats=self._convergence.get_convergence_stats(),
            pruning_stats=self._pruner.get_pruning_summary(),
            rollout_stats=self._rollout_scheduler.get_memoization_stats(),
        )

    # ------------------------------------------------------------------
    # MCTS iteration
    # ------------------------------------------------------------------

    def _run_single_iteration(self, inference_engine: Any) -> None:
        """Execute one iteration of the MCTS loop."""
        assert self._root is not None

        # 1. Selection
        node = self._select(self._root)

        # 2. Expansion
        if not node.is_fully_expanded() and node.stats.visit_count > 0:
            node = self._expand(node)

        # 3. Rollout
        value = self._rollout(node, inference_engine)

        # 4. Backpropagation
        self._backpropagate(node, value)

        # Update tracking
        self._total_rollouts += 1
        self._update_best(node.state, value)

        # Update convergence monitor
        arm_key = self._state_to_arm_key(node.state)
        self._convergence.update(arm_key, value)
        self._ucb.update_arm(arm_key, value)

    def _run_batch_iteration(self, inference_engine: Any) -> None:
        """Execute a batch of MCTS iterations for efficiency."""
        assert self._root is not None
        batch_size = self.config.batch_size

        nodes = []
        states = []

        # Select and expand batch_size nodes
        for _ in range(batch_size):
            node = self._select(self._root)
            if not node.is_fully_expanded() and node.stats.visit_count > 0:
                node = self._expand(node)
            nodes.append(node)
            states.append(dict(node.state))

        # Batch rollout
        remaining_vars = [
            v for v in self._interface_vars if v not in states[0]
        ] if states else []

        batch_result = self._rollout_scheduler.schedule_batch(
            states,
            inference_engine,
            self._target_variable,
            remaining_vars,
        )

        # Backpropagate all results
        for node, result in zip(nodes, batch_result.results):
            self._backpropagate(node, result.value)
            self._total_rollouts += 1
            self._update_best(node.state, result.value)

            arm_key = self._state_to_arm_key(node.state)
            self._convergence.update(arm_key, result.value)
            self._ucb.update_arm(arm_key, result.value)

    # ------------------------------------------------------------------
    # Selection phase
    # ------------------------------------------------------------------

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Select a leaf node by traversing the tree using UCB.

        Walks from the root, selecting the child with the highest UCB
        score at each step, until a leaf (unexpanded) node is reached.
        """
        current = node

        while not current.is_leaf():
            if not current.is_fully_expanded():
                return current

            # Skip fully-pruned nodes
            unpruned_children = {
                a: c for a, c in current.children.items() if not c.pruned
            }

            if not unpruned_children:
                return current

            # Select child with highest UCB score
            if self._dag is not None and self._target_variable is not None:
                # Use causal UCB
                best_child = self._causal_select_child(current)
            else:
                best_child = current.select_child(
                    self.config.exploration_constant,
                    self.config.maximize,
                    self.config.ucb_variant,
                )

            current = best_child

        return current

    def _causal_select_child(self, node: MCTSNode) -> MCTSNode:
        """Select child using causal UCB with information-gain bonus."""
        best_score = float("-inf")
        best_child = None
        parent_visits = node.stats.visit_count

        for action, child in node.children.items():
            if child.pruned:
                continue

            causal_bonus = self._ucb.compute_information_gain_bonus(
                action, self._dag, self._target_variable, node.state
            )

            score = self._ucb.compute_score(child, parent_visits, causal_bonus)

            if score > best_score:
                best_score = score
                best_child = child

        if best_child is None:
            # Fallback: standard UCB
            return node.select_child(
                self.config.exploration_constant,
                self.config.maximize,
                self.config.ucb_variant,
            )

        return best_child

    # ------------------------------------------------------------------
    # Expansion phase
    # ------------------------------------------------------------------

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Expand the node by adding a new child.

        If progressive widening is enabled, may not expand if the
        current number of children already satisfies the widening bound.
        """
        if self.config.progressive_widening:
            child = node.expand_progressive(
                self._progressive_action_generator,
                self.config.pw_k_factor,
                self.config.pw_alpha,
            )
            if child is not None:
                # Set available actions for new child
                child_actions = self._get_child_actions(child)
                child.set_available_actions(child_actions)
                return child
            # Progressive widening chose not to expand; return existing best child
            if node.children:
                return node.select_child(
                    self.config.exploration_constant,
                    self.config.maximize,
                    self.config.ucb_variant,
                )
            return node

        # Standard expansion
        unexpanded = node.unexpanded_actions()
        if not unexpanded:
            return node

        # Filter out pruned actions
        if self.config.enable_pruning and self._dag is not None and self._target_variable:
            evidence_set = set(node.state.keys())
            unexpanded = [
                a for a in unexpanded
                if not self._pruner.is_prunable(
                    a[0], self._target_variable, node.state, self._dag
                )
            ]

        if not unexpanded:
            unexpanded = node.unexpanded_actions()

        if not unexpanded:
            return node

        # Select action to expand (random among unexpanded)
        idx = self._rng.randint(len(unexpanded))
        action = unexpanded[idx]

        child_actions = self._get_child_actions_for(node, action)
        child = node.expand(action, child_actions)

        return child

    def _progressive_action_generator(
        self,
        state: Dict[str, float],
        expanded_actions: set,
    ) -> Optional[Tuple[str, float]]:
        """
        Generate a new action for progressive widening.

        Samples from the action space, avoiding already-expanded actions.
        """
        assigned_vars = set(state.keys())
        available_vars = [v for v in self._interface_vars if v not in assigned_vars]

        if not available_vars:
            return None

        # Pick a random variable
        var_idx = self._rng.randint(len(available_vars))
        var = available_vars[var_idx]

        # Sample a shock value
        lo, hi = self.config.shock_range
        value = float(self._rng.uniform(lo, hi))

        action = (var, value)

        # Avoid duplicates (check by variable name match)
        for existing in expanded_actions:
            if existing[0] == var and abs(existing[1] - value) < 1e-6:
                # Try again with different value
                value = float(self._rng.uniform(lo, hi))
                action = (var, value)

        return action

    def _get_child_actions(self, child: MCTSNode) -> List[Tuple[str, float]]:
        """Build the set of available actions for a child node."""
        assigned_vars = set(child.state.keys())
        remaining_vars = [v for v in self._interface_vars if v not in assigned_vars]

        actions = []
        for var in remaining_vars:
            for a in self._all_actions:
                if a[0] == var:
                    actions.append(a)

        return actions

    def _get_child_actions_for(
        self, parent: MCTSNode, action: Tuple[str, float]
    ) -> List[Tuple[str, float]]:
        """Build actions for a hypothetical child created by `action`."""
        assigned_vars = set(parent.state.keys()) | {action[0]}
        remaining_vars = [v for v in self._interface_vars if v not in assigned_vars]

        actions = []
        for var in remaining_vars:
            for a in self._all_actions:
                if a[0] == var:
                    actions.append(a)

        return actions

    # ------------------------------------------------------------------
    # Rollout phase
    # ------------------------------------------------------------------

    def _rollout(self, node: MCTSNode, inference_engine: Any) -> float:
        """
        Evaluate the node via rollout.

        Completes the partial assignment and evaluates using the
        inference engine through the rollout scheduler.
        """
        remaining_vars = [
            v for v in self._interface_vars if v not in node.state
        ]

        result = self._rollout_scheduler.evaluate_rollout(
            state=dict(node.state),
            inference_engine=inference_engine,
            target_variable=self._target_variable,
            remaining_variables=remaining_vars,
        )

        return result.value

    # ------------------------------------------------------------------
    # Backpropagation phase
    # ------------------------------------------------------------------

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Backpropagate the rollout value up the tree."""
        node.backpropagate(value)

    # ------------------------------------------------------------------
    # Pruning pass
    # ------------------------------------------------------------------

    def _run_pruning_pass(self) -> None:
        """Run d-separation pruning on all expanded internal nodes."""
        if self._root is None or self._dag is None or self._target_variable is None:
            return

        stack = [self._root]
        total_pruned = 0

        while stack:
            node = stack.pop()
            if node.is_leaf() or node.pruned:
                continue

            pruned = self._ucb.prune_irrelevant(
                node, self._dag, self._target_variable
            )
            total_pruned += pruned

            for child in node.children.values():
                if not child.pruned:
                    stack.append(child)

        if total_pruned > 0:
            logger.debug(
                "Pruning pass: %d branches pruned at rollout %d",
                total_pruned,
                self._total_rollouts,
            )

    # ------------------------------------------------------------------
    # Action space construction
    # ------------------------------------------------------------------

    def _build_action_space(
        self,
        interface_vars: List[str],
        shock_values: Optional[Dict[str, List[float]]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Build the discrete action space for MCTS.

        Each action is a (variable, shock_value) pair. If shock_values
        is provided, those values are used; otherwise the shock_range
        is discretized uniformly.
        """
        actions: List[Tuple[str, float]] = []

        for var in interface_vars:
            if shock_values is not None and var in shock_values:
                for val in shock_values[var]:
                    actions.append((var, val))
            else:
                lo, hi = self.config.shock_range
                n_vals = self.config.values_per_variable
                values = np.linspace(lo, hi, n_vals)
                for val in values:
                    actions.append((var, float(val)))

        return actions

    def _setup_rollout_scheduler(
        self,
        interface_vars: List[str],
        shock_values: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        """Configure the rollout scheduler with variable domains."""
        domains = {}
        discretization = {}

        for var in interface_vars:
            lo, hi = self.config.shock_range
            domains[var] = (lo, hi)

            if shock_values is not None and var in shock_values:
                discretization[var] = shock_values[var]
            else:
                discretization[var] = list(
                    np.linspace(lo, hi, self.config.values_per_variable)
                )

        self._rollout_scheduler.set_variable_domains(domains)
        self._rollout_scheduler.set_variable_discretization(discretization)
        self._rollout_scheduler.set_rollout_policy(self.config.rollout_policy)

    # ------------------------------------------------------------------
    # Best-tracking
    # ------------------------------------------------------------------

    def _update_best(
        self, state: Dict[str, float], value: float
    ) -> None:
        """Update the best scenario seen so far."""
        if self.config.maximize:
            if value > self._best_value:
                self._best_value = value
                self._best_state = dict(state)
        else:
            if value < self._best_value:
                self._best_value = value
                self._best_state = dict(state)

    def _state_to_arm_key(self, state: Dict[str, float]) -> str:
        """Convert a partial state to a hashable arm key for convergence tracking."""
        items = sorted(state.items())
        return "|".join(f"{k}={v:.6g}" for k, v in items)

    # ------------------------------------------------------------------
    # Public API: results
    # ------------------------------------------------------------------

    def get_best_scenario(self) -> Optional[ScenarioReport]:
        """
        Return the best scenario found during search.

        Returns
        -------
        ScenarioReport or None
        """
        if self._best_state is None:
            return None

        arm_key = self._state_to_arm_key(self._best_state)
        ci = self._convergence.get_confidence_interval(arm_key)

        # Find the tree node corresponding to the best state
        action_sequence = []
        if self._root is not None:
            node = self._find_node_for_state(self._root, self._best_state)
            if node is not None:
                action_sequence = node.get_action_sequence()
                visit_count = node.stats.visit_count
            else:
                visit_count = 0
        else:
            visit_count = 0

        return ScenarioReport(
            state=dict(self._best_state),
            value=self._best_value,
            visit_count=visit_count,
            confidence_interval=ci,
            rank=1,
            action_sequence=action_sequence,
        )

    def get_scenario_ranking(
        self, top_k: int = 10
    ) -> List[ScenarioReport]:
        """
        Return the top-k scenarios ranked by value.

        Parameters
        ----------
        top_k : int
            Number of top scenarios to return.

        Returns
        -------
        list of ScenarioReport
        """
        if self._root is None:
            return []

        # Collect all leaf/expanded nodes with their mean values
        candidates: List[Tuple[MCTSNode, float]] = []
        self._collect_candidates(self._root, candidates)

        # Sort by mean value
        if self.config.maximize:
            candidates.sort(key=lambda x: x[1], reverse=True)
        else:
            candidates.sort(key=lambda x: x[1])

        candidates = candidates[:top_k]

        reports = []
        for rank, (node, mean_val) in enumerate(candidates, start=1):
            arm_key = self._state_to_arm_key(node.state)
            ci = self._convergence.get_confidence_interval(arm_key)

            reports.append(
                ScenarioReport(
                    state=dict(node.state),
                    value=mean_val,
                    visit_count=node.stats.visit_count,
                    confidence_interval=ci,
                    rank=rank,
                    action_sequence=node.get_action_sequence(),
                )
            )

        return reports

    def _collect_candidates(
        self,
        node: MCTSNode,
        candidates: List[Tuple[MCTSNode, float]],
    ) -> None:
        """Recursively collect nodes with visit counts as scenario candidates."""
        if node.stats.visit_count > 0 and node.state:
            candidates.append((node, node.stats.mean_value))

        for child in node.children.values():
            if not child.pruned:
                self._collect_candidates(child, candidates)

    def _find_node_for_state(
        self, root: MCTSNode, state: Dict[str, float]
    ) -> Optional[MCTSNode]:
        """Find the tree node matching the given state."""
        if root.state == state:
            return root

        for child in root.children.values():
            result = self._find_node_for_state(child, state)
            if result is not None:
                return result

        return None

    # ------------------------------------------------------------------
    # Public API: statistics
    # ------------------------------------------------------------------

    def get_convergence_stats(self) -> Dict[str, Any]:
        """Return convergence statistics."""
        return self._convergence.get_convergence_stats()

    def get_tree_stats(self) -> Dict[str, Any]:
        """Return search tree statistics."""
        if self._root is None:
            return {}
        return self._root.subtree_statistics()

    def get_search_summary(self) -> Dict[str, Any]:
        """Return a summary of the search."""
        elapsed = time.time() - self._start_time if self._start_time > 0 else 0

        return {
            "total_rollouts": self._total_rollouts,
            "elapsed_seconds": elapsed,
            "rollouts_per_second": self._total_rollouts / elapsed if elapsed > 0 else 0,
            "best_value": self._best_value if self._best_state else None,
            "best_state": self._best_state,
            "converged": self._convergence.is_converged(),
            "n_interface_vars": len(self._interface_vars),
            "n_actions": len(self._all_actions),
            "tree_size": self._root.tree_size() if self._root else 0,
        }

    # ------------------------------------------------------------------
    # Incremental search
    # ------------------------------------------------------------------

    def continue_search(
        self,
        inference_engine: Any,
        additional_rollouts: int = 1000,
        additional_seconds: float = 60.0,
    ) -> SearchResult:
        """
        Continue search from the current tree state.

        Parameters
        ----------
        inference_engine : object
            Inference engine for rollouts.
        additional_rollouts : int
            Additional rollouts to perform.
        additional_seconds : float
            Additional time budget.

        Returns
        -------
        SearchResult
        """
        if self._root is None:
            raise RuntimeError("Cannot continue search; no search has been started.")

        start_rollouts = self._total_rollouts
        start_time = time.time()
        converged = False

        while (self._total_rollouts - start_rollouts) < additional_rollouts:
            elapsed = time.time() - start_time
            if elapsed >= additional_seconds:
                break

            self._run_single_iteration(inference_engine)

            if self._total_rollouts % self.config.convergence_check_interval == 0:
                if self._convergence.is_converged():
                    converged = True
                    break

        elapsed = time.time() - start_time

        return SearchResult(
            best_scenario=self.get_best_scenario(),
            all_scenarios=self.get_scenario_ranking(top_k=20),
            converged=converged,
            total_rollouts=self._total_rollouts,
            elapsed_seconds=elapsed,
            tree_stats=self._root.subtree_statistics(),
            convergence_stats=self._convergence.get_convergence_stats(),
            pruning_stats=self._pruner.get_pruning_summary(),
            rollout_stats=self._rollout_scheduler.get_memoization_stats(),
        )

    # ------------------------------------------------------------------
    # Interface shock enumeration
    # ------------------------------------------------------------------

    def enumerate_interface_shocks(
        self,
        interface_vars: List[str],
        dag: Any,
        target: str,
        shock_values: Optional[Dict[str, List[float]]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Enumerate shock actions over separator variables, filtering
        by d-separation relevance.

        Parameters
        ----------
        interface_vars : list of str
            Interface (separator) variables.
        dag : nx.DiGraph
            Causal DAG.
        target : str
            Target loss variable.
        shock_values : dict or None
            Specific shock values per variable.

        Returns
        -------
        list of (variable, value) tuples
            Relevant shock actions after d-sep filtering.
        """
        all_actions = self._build_action_space(interface_vars, shock_values)

        if dag is None:
            return all_actions

        # Filter by d-separation
        relevant_vars = self._pruner.get_relevant_variables(
            target, {}, dag
        )
        relevant_set = set(relevant_vars)

        filtered = [a for a in all_actions if a[0] in relevant_set]

        pruning_ratio = 1.0 - len(filtered) / len(all_actions) if all_actions else 0.0
        logger.info(
            "Interface shock enumeration: %d total, %d relevant (%.1f%% pruned)",
            len(all_actions),
            len(filtered),
            pruning_ratio * 100,
        )

        return filtered if filtered else all_actions

    # ------------------------------------------------------------------
    # PAC convergence analysis
    # ------------------------------------------------------------------

    def compute_pac_rollout_budget(
        self,
        n_interface_vars: int,
        values_per_var: Optional[int] = None,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Compute the PAC rollout budget before running search.

        Parameters
        ----------
        n_interface_vars : int
            Number of interface variables.
        values_per_var : int or None
            Discretization granularity. Uses config default if None.
        epsilon : float or None
            Accuracy parameter. Uses config default if None.
        delta : float or None
            Confidence parameter. Uses config default if None.

        Returns
        -------
        dict
            Contains 'n_arms', 'hoeffding_budget', 'bernstein_budget',
            'recommended_budget'.
        """
        if values_per_var is None:
            values_per_var = self.config.values_per_variable
        if epsilon is None:
            epsilon = self.config.epsilon
        if delta is None:
            delta = self.config.delta

        # Full action space (product of all variable-value combinations)
        n_arms = n_interface_vars * values_per_var

        # With pruning, the effective arms may be smaller
        effective_arms = n_arms
        if self._dag is not None and self._target_variable is not None:
            effective_arms = self._ucb.compute_effective_action_space_size(
                self._dag, self._target_variable, {}, values_per_var
            )

        hoeffding_result = self._ucb.compute_pac_bound(
            effective_arms, epsilon, delta, bound_type="hoeffding"
        )

        bernstein_result = self._ucb.compute_pac_bound(
            effective_arms, epsilon, delta, bound_type="bernstein"
        )

        recommended = min(
            hoeffding_result.required_rollouts,
            bernstein_result.required_rollouts,
        )

        return {
            "n_arms_full": n_arms,
            "n_arms_effective": effective_arms,
            "hoeffding_budget": hoeffding_result.required_rollouts,
            "bernstein_budget": bernstein_result.required_rollouts,
            "recommended_budget": recommended,
            "epsilon": epsilon,
            "delta": delta,
            "pruning_reduction": 1.0 - effective_arms / n_arms if n_arms > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # Tree visualization helpers
    # ------------------------------------------------------------------

    def get_tree_summary(self, max_depth: int = 3) -> Dict[str, Any]:
        """
        Return a JSON-serializable summary of the search tree.

        Parameters
        ----------
        max_depth : int
            Maximum depth to include.
        """
        if self._root is None:
            return {}
        return self._root.to_dict(max_depth=max_depth)

    def get_top_branches(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Return the top-k branches from the root by visit count.

        Returns
        -------
        list of dict
            Each dict contains the branch action, visits, mean value,
            and confidence interval.
        """
        if self._root is None:
            return []

        branches = []
        for action, child in self._root.children.items():
            arm_key = self._state_to_arm_key(child.state)
            ci = self._convergence.get_confidence_interval(arm_key)

            branches.append({
                "action": action,
                "visits": child.stats.visit_count,
                "mean_value": child.stats.mean_value,
                "ci_lower": ci[0],
                "ci_upper": ci[1],
                "pruned": child.pruned,
                "n_descendants": child.tree_size(),
            })

        branches.sort(key=lambda x: x["visits"], reverse=True)
        return branches[:top_k]

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset search state entirely."""
        self._root = None
        self._total_rollouts = 0
        self._best_value = float("-inf") if self.config.maximize else float("inf")
        self._best_state = None
        self._search_history.clear()
        self._convergence.reset()
        self._rollout_scheduler.reset()
        self._ucb.reset()
        self._pruner.clear_cache()
        self._pruner.clear_history()

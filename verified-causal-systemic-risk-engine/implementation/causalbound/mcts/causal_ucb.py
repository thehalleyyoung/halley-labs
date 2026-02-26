"""
Causal UCB acquisition function for MCTS.

Uses d-separation queries on the underlying causal DAG to prune
irrelevant arms and compute tighter PAC bounds. Implements a
causal exploration bonus based on information gain and supports
non-stationary reward distributions with confidence interval tracking.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

try:
    import networkx as nx
except ImportError:
    nx = None  # type: ignore

from .tree_node import MCTSNode, NodeStatistics


# -----------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------

@dataclass
class ArmStatistics:
    """Per-arm tracking with sliding window for non-stationarity."""

    total: NodeStatistics = field(default_factory=NodeStatistics)
    window_values: List[float] = field(default_factory=list)
    window_size: int = 200
    confidence_lower: float = float("-inf")
    confidence_upper: float = float("inf")
    pruned: bool = False
    pruned_reason: str = ""

    def update(self, value: float) -> None:
        """Record a new reward observation for this arm."""
        self.total.update(value)
        self.window_values.append(value)
        if len(self.window_values) > self.window_size:
            self.window_values.pop(0)

    @property
    def windowed_mean(self) -> float:
        """Mean value over recent observations."""
        if not self.window_values:
            return 0.0
        return float(np.mean(self.window_values))

    @property
    def windowed_variance(self) -> float:
        """Variance over recent observations."""
        if len(self.window_values) < 2:
            return float("inf")
        return float(np.var(self.window_values, ddof=1))


@dataclass
class PACResult:
    """Result of a PAC bound computation."""

    required_rollouts: int
    epsilon: float
    delta: float
    n_effective_arms: int
    bound_type: str  # "hoeffding" or "bernstein"


# -----------------------------------------------------------------------
# CausalUCB
# -----------------------------------------------------------------------

class CausalUCB:
    """
    Causal UCB acquisition function.

    Combines standard UCB1 exploration bonus with causal pruning to
    reduce the effective action space and accelerate convergence.

    Parameters
    ----------
    exploration_constant : float
        Base exploration parameter c in UCB1.
    causal_bonus_weight : float
        Weight applied to the causal information-gain bonus.
    use_variance : bool
        If True, use UCB-Tuned (variance-aware) instead of plain UCB1.
    window_size : int
        Sliding window length for non-stationary reward tracking.
    maximize : bool
        If True, search for the arm with maximum value (worst-case).
    """

    def __init__(
        self,
        exploration_constant: float = 1.414,
        causal_bonus_weight: float = 0.5,
        use_variance: bool = True,
        window_size: int = 200,
        maximize: bool = True,
    ) -> None:
        self.exploration_constant = exploration_constant
        self.causal_bonus_weight = causal_bonus_weight
        self.use_variance = use_variance
        self.window_size = window_size
        self.maximize = maximize

        self._arm_stats: Dict[Any, ArmStatistics] = defaultdict(
            lambda: ArmStatistics(window_size=self.window_size)
        )
        self._pruned_arms: Set[Any] = set()
        self._dsep_cache: Dict[FrozenSet, bool] = {}
        self._total_pulls: int = 0
        self._info_gain_estimates: Dict[Any, float] = {}

    # ------------------------------------------------------------------
    # Core UCB computation
    # ------------------------------------------------------------------

    def compute_score(
        self,
        node: MCTSNode,
        parent_visits: int,
        causal_bonus: float = 0.0,
    ) -> float:
        """
        Compute causal UCB score for a node.

        Score = exploitation + exploration + causal_bonus

        Parameters
        ----------
        node : MCTSNode
            The node to score.
        parent_visits : int
            Visit count of the parent node.
        causal_bonus : float
            Optional pre-computed causal information-gain bonus.

        Returns
        -------
        float
            Combined UCB score.
        """
        if node.pruned:
            return float("-inf")

        if node.stats.visit_count == 0:
            return float("inf")

        exploitation = node.stats.mean_value
        if not self.maximize:
            exploitation = -exploitation

        log_parent = math.log(max(1, parent_visits))
        n = node.stats.visit_count

        if self.use_variance and n >= 2:
            var = node.stats.variance
            var = min(max(var, 0.0), 1e6)
            v_bound = var + math.sqrt(2.0 * log_parent / n)
            exploration = self.exploration_constant * math.sqrt(
                (log_parent / n) * min(0.25, v_bound)
            )
        else:
            exploration = self.exploration_constant * math.sqrt(log_parent / n)

        causal_term = self.causal_bonus_weight * causal_bonus

        return exploitation + exploration + causal_term

    def compute_information_gain_bonus(
        self,
        action: Tuple[str, float],
        dag: Any,
        target: str,
        evidence: Dict[str, float],
    ) -> float:
        """
        Estimate information gain from observing `action`.

        Uses a simple proxy: the number of target-reachable variables whose
        d-separation status changes when `action` is added to the evidence.

        Parameters
        ----------
        action : tuple
            (variable, value) pair.
        dag : nx.DiGraph
            Causal DAG.
        target : str
            Target loss variable.
        evidence : dict
            Current evidence / partial assignment.

        Returns
        -------
        float
            Estimated information gain (non-negative).
        """
        if nx is None:
            return 0.0

        var_name = action[0]
        if var_name in evidence:
            return 0.0

        evidence_vars = set(evidence.keys())
        new_evidence_vars = evidence_vars | {var_name}

        active_before = self._count_active_paths(dag, var_name, target, evidence_vars)
        active_after = self._count_active_paths(dag, var_name, target, new_evidence_vars)

        gain = max(0.0, active_before - active_after)

        n_nodes = dag.number_of_nodes()
        if n_nodes > 0:
            gain /= n_nodes

        self._info_gain_estimates[action] = gain
        return gain

    def _count_active_paths(
        self,
        dag: Any,
        source: str,
        target: str,
        evidence: Set[str],
    ) -> int:
        """
        Count the number of nodes on active trails from source to target.

        Uses a simplified Bayes-ball style reachability check.
        """
        if nx is None:
            return 0

        if source not in dag or target not in dag:
            return 0

        visited: Set[Tuple[str, str]] = set()
        queue: List[Tuple[str, str]] = []

        if source not in evidence:
            queue.append((source, "up"))
            queue.append((source, "down"))

        reachable: Set[str] = set()

        while queue:
            node, direction = queue.pop(0)
            if (node, direction) in visited:
                continue
            visited.add((node, direction))

            if node == target:
                reachable.add(node)

            if direction == "up" and node not in evidence:
                for parent in dag.predecessors(node):
                    reachable.add(parent)
                    queue.append((parent, "up"))
                for child in dag.successors(node):
                    queue.append((child, "down"))

            elif direction == "down":
                if node not in evidence:
                    for child in dag.successors(node):
                        reachable.add(child)
                        queue.append((child, "down"))
                if node in evidence:
                    for parent in dag.predecessors(node):
                        reachable.add(parent)
                        queue.append((parent, "up"))

        # Count reachable that include descendants of evidence
        descendants_of_evidence: Set[str] = set()
        for ev in evidence:
            if ev in dag:
                descendants_of_evidence.update(nx.descendants(dag, ev))

        for ev in evidence:
            if ev in dag:
                if descendants_of_evidence & reachable:
                    for parent in dag.predecessors(ev):
                        if parent not in evidence:
                            reachable.add(parent)

        return len(reachable)

    # ------------------------------------------------------------------
    # D-separation pruning
    # ------------------------------------------------------------------

    def prune_irrelevant(
        self,
        node: MCTSNode,
        dag: Any,
        target: str,
    ) -> int:
        """
        Prune children of `node` whose action variable is d-separated
        from `target` given the current partial assignment (evidence).

        Parameters
        ----------
        node : MCTSNode
            The parent node whose children to prune.
        dag : nx.DiGraph
            Causal DAG.
        target : str
            The target loss variable.

        Returns
        -------
        int
            Number of children pruned.
        """
        if nx is None:
            return 0

        evidence = set(node.state.keys())
        pruned_count = 0

        for action, child in node.children.items():
            if child.pruned:
                continue

            var_name = action[0]
            if self._is_dseparated(var_name, target, evidence, dag):
                child.mark_pruned(
                    f"{var_name} d-separated from {target} given {evidence}"
                )
                self._pruned_arms.add(action)
                pruned_count += 1

        unexpanded = node.unexpanded_actions()
        new_available = []
        for action in unexpanded:
            var_name = action[0]
            if not self._is_dseparated(var_name, target, evidence, dag):
                new_available.append(action)
            else:
                self._pruned_arms.add(action)
                pruned_count += 1

        node.set_available_actions(
            new_available + [a for a in node.unexpanded_actions() if a not in new_available and a not in self._pruned_arms]
        )

        return pruned_count

    def _is_dseparated(
        self,
        source: str,
        target: str,
        evidence: Set[str],
        dag: Any,
    ) -> bool:
        """
        Check if source is d-separated from target given evidence.

        Uses Bayes-ball algorithm with memoization.
        """
        cache_key = frozenset([
            ("src", source),
            ("tgt", target),
            *[("ev", e) for e in sorted(evidence)],
        ])

        if cache_key in self._dsep_cache:
            return self._dsep_cache[cache_key]

        if source not in dag or target not in dag:
            self._dsep_cache[cache_key] = True
            return True

        if source == target:
            self._dsep_cache[cache_key] = False
            return False

        reachable = self._bayes_ball_reachable(source, evidence, dag)
        is_dsep = target not in reachable
        self._dsep_cache[cache_key] = is_dsep
        return is_dsep

    def _bayes_ball_reachable(
        self,
        source: str,
        evidence: Set[str],
        dag: Any,
    ) -> Set[str]:
        """
        Compute the set of nodes reachable from source via active trails.

        Implements the Bayes-ball algorithm (Shachter, 1998).
        """
        evidence_set = set(evidence)

        visited_up: Set[str] = set()
        visited_down: Set[str] = set()
        queue: List[Tuple[str, str]] = [(source, "up")]

        reachable: Set[str] = set()

        # Precompute descendants of evidence for collider handling
        desc_of_evidence: Set[str] = set()
        for ev in evidence_set:
            if ev in dag:
                desc_of_evidence.update(nx.descendants(dag, ev))
                desc_of_evidence.add(ev)

        while queue:
            node, direction = queue.pop(0)

            if direction == "up":
                if node in visited_up:
                    continue
                visited_up.add(node)
            else:
                if node in visited_down:
                    continue
                visited_down.add(node)

            reachable.add(node)

            if direction == "up" and node not in evidence_set:
                # Traverse to parents (continue up the chain)
                for parent in dag.predecessors(node):
                    queue.append((parent, "up"))
                # Traverse to children (go through the node)
                for child in dag.successors(node):
                    queue.append((child, "down"))

            elif direction == "down":
                if node not in evidence_set:
                    # Non-evidence: pass through to children
                    for child in dag.successors(node):
                        queue.append((child, "down"))
                # If evidence or descendant of evidence, activate collider
                if node in desc_of_evidence:
                    for parent in dag.predecessors(node):
                        queue.append((parent, "up"))

        return reachable

    # ------------------------------------------------------------------
    # Effective action space
    # ------------------------------------------------------------------

    def get_effective_arms(
        self,
        dag: Any,
        target: str,
        evidence: Dict[str, float],
    ) -> List[str]:
        """
        Return the set of variables that are *not* d-separated from
        `target` given the current evidence. These are the "effective"
        arms that need to be explored.

        Parameters
        ----------
        dag : nx.DiGraph
            Causal DAG.
        target : str
            Target loss variable.
        evidence : dict
            Current partial assignment.

        Returns
        -------
        list of str
            Variables that may affect the target.
        """
        if nx is None:
            return list(dag.nodes()) if hasattr(dag, "nodes") else []

        evidence_set = set(evidence.keys())
        effective = []

        for node in dag.nodes():
            if node == target:
                continue
            if node in evidence:
                continue
            if not self._is_dseparated(node, target, evidence_set, dag):
                effective.append(node)

        return effective

    def compute_effective_action_space_size(
        self,
        dag: Any,
        target: str,
        evidence: Dict[str, float],
        values_per_variable: int = 10,
    ) -> int:
        """
        Compute the size of the effective action space after pruning.

        Parameters
        ----------
        dag : nx.DiGraph
            Causal DAG.
        target : str
            Target loss variable.
        evidence : dict
            Current evidence.
        values_per_variable : int
            Discretization granularity per variable.

        Returns
        -------
        int
            Effective number of arms.
        """
        effective_vars = self.get_effective_arms(dag, target, evidence)
        return len(effective_vars) * values_per_variable

    # ------------------------------------------------------------------
    # PAC bound computation
    # ------------------------------------------------------------------

    def compute_pac_bound(
        self,
        n_arms: int,
        epsilon: float,
        delta: float,
        value_range: float = 1.0,
        bound_type: str = "hoeffding",
        empirical_variance: Optional[float] = None,
    ) -> PACResult:
        """
        Compute the number of rollouts required for (epsilon, delta)-PAC
        identification of the best arm.

        Parameters
        ----------
        n_arms : int
            Number of arms (effective action space size).
        epsilon : float
            Accuracy parameter: find arm within epsilon of optimal.
        delta : float
            Confidence parameter: succeed with probability >= 1 - delta.
        value_range : float
            Range of possible reward values [0, value_range].
        bound_type : str
            "hoeffding" for Hoeffding-based bound, "bernstein" for
            Bernstein-based (tighter with known variance).
        empirical_variance : float or None
            Empirical variance estimate (used for Bernstein bound).

        Returns
        -------
        PACResult
            Required number of rollouts and metadata.
        """
        if n_arms <= 0:
            return PACResult(
                required_rollouts=0,
                epsilon=epsilon,
                delta=delta,
                n_effective_arms=0,
                bound_type=bound_type,
            )

        if epsilon <= 0:
            epsilon = 1e-6

        # Union bound: need per-arm confidence delta / n_arms
        per_arm_delta = delta / max(1, n_arms)

        if bound_type == "bernstein" and empirical_variance is not None:
            # Bernstein inequality
            # P(|X - mu| >= t) <= 2 exp(-n*t^2 / (2*sigma^2 + 2*R*t/3))
            # Solving for n:
            sigma_sq = empirical_variance
            r = value_range
            log_term = math.log(2.0 / per_arm_delta)

            # Quadratic in n: n*eps^2 - (2*sigma^2 + 2*R*eps/3)*log(2/delta_a) >= 0
            # n >= (2*sigma^2 + 2*R*eps/3) * log(2/delta_a) / eps^2
            numerator = (2.0 * sigma_sq + 2.0 * r * epsilon / 3.0) * log_term
            n_per_arm = math.ceil(numerator / (epsilon * epsilon))
        else:
            # Hoeffding inequality
            # P(|X - mu| >= t) <= 2 exp(-2*n*t^2 / R^2)
            # n >= R^2 * log(2/delta_a) / (2*eps^2)
            log_term = math.log(2.0 / per_arm_delta)
            n_per_arm = math.ceil(
                (value_range ** 2) * log_term / (2.0 * epsilon * epsilon)
            )

        total_rollouts = n_per_arm * n_arms

        return PACResult(
            required_rollouts=total_rollouts,
            epsilon=epsilon,
            delta=delta,
            n_effective_arms=n_arms,
            bound_type=bound_type,
        )

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self,
        node: MCTSNode,
        available_actions: List[Tuple[str, float]],
        dag: Any = None,
        target: Optional[str] = None,
    ) -> Tuple[str, float]:
        """
        Select the next action to take from the given node.

        If a DAG and target are provided, causal pruning and info-gain
        bonuses are incorporated. Otherwise falls back to standard UCB.

        Parameters
        ----------
        node : MCTSNode
            Current node.
        available_actions : list
            Candidate (variable, value) pairs.
        dag : nx.DiGraph or None
            Causal DAG for pruning.
        target : str or None
            Target variable for pruning.

        Returns
        -------
        tuple
            Selected (variable, value) action.
        """
        if not available_actions:
            raise ValueError("No available actions to select from")

        if len(available_actions) == 1:
            return available_actions[0]

        evidence = dict(node.state)

        # Filter out d-separated actions
        if dag is not None and target is not None:
            evidence_set = set(evidence.keys())
            filtered = [
                a for a in available_actions
                if not self._is_dseparated(a[0], target, evidence_set, dag)
            ]
            if filtered:
                available_actions = filtered

        # Score each action
        parent_visits = node.stats.visit_count
        best_action = None
        best_score = float("-inf")

        for action in available_actions:
            if action in node.children:
                child = node.children[action]
                if child.pruned:
                    continue

                # Compute causal bonus
                causal_bonus = 0.0
                if dag is not None and target is not None:
                    causal_bonus = self.compute_information_gain_bonus(
                        action, dag, target, evidence
                    )

                score = self.compute_score(child, parent_visits, causal_bonus)
            else:
                # Unexpanded: give maximum priority
                score = float("inf")

                # Break ties with info gain
                if dag is not None and target is not None:
                    info_gain = self.compute_information_gain_bonus(
                        action, dag, target, evidence
                    )
                    # Use large number + info_gain to differentiate among inf scores
                    score = 1e18 + info_gain

            if score > best_score:
                best_score = score
                best_action = action

        if best_action is None:
            # Fallback: random selection
            idx = np.random.randint(len(available_actions))
            return available_actions[idx]

        return best_action

    # ------------------------------------------------------------------
    # Confidence intervals
    # ------------------------------------------------------------------

    def update_arm(self, arm: Any, value: float) -> None:
        """
        Record a new observation for the given arm.

        Parameters
        ----------
        arm : hashable
            Arm identifier.
        value : float
            Observed reward.
        """
        self._arm_stats[arm].update(value)
        self._total_pulls += 1

    def get_confidence_interval(
        self,
        arm: Any,
        delta: float = 0.05,
        value_range: float = 1.0,
        use_empirical_bernstein: bool = True,
    ) -> Tuple[float, float]:
        """
        Compute a confidence interval for the arm's mean reward.

        Parameters
        ----------
        arm : hashable
            Arm identifier.
        delta : float
            Confidence level: interval holds with probability >= 1 - delta.
        value_range : float
            Range of reward values.
        use_empirical_bernstein : bool
            If True and enough data, use the empirical Bernstein bound.

        Returns
        -------
        tuple of float
            (lower_bound, upper_bound)
        """
        stats = self._arm_stats[arm]
        n = stats.total.visit_count

        if n == 0:
            return (float("-inf"), float("inf"))

        mean = stats.total.mean_value

        if use_empirical_bernstein and n >= 10:
            # Empirical Bernstein bound
            var = stats.windowed_variance if len(stats.window_values) >= 2 else stats.total.variance
            var = min(max(var, 0.0), value_range ** 2)
            log_term = math.log(3.0 / delta)

            width = math.sqrt(2.0 * var * log_term / n) + 3.0 * value_range * log_term / n
        else:
            # Hoeffding bound
            log_term = math.log(2.0 / delta)
            width = value_range * math.sqrt(log_term / (2.0 * n))

        lower = mean - width
        upper = mean + width

        stats.confidence_lower = lower
        stats.confidence_upper = upper

        return (lower, upper)

    def get_all_confidence_intervals(
        self, delta: float = 0.05, value_range: float = 1.0
    ) -> Dict[Any, Tuple[float, float]]:
        """Compute confidence intervals for all tracked arms."""
        result = {}
        for arm in self._arm_stats:
            result[arm] = self.get_confidence_interval(arm, delta, value_range)
        return result

    # ------------------------------------------------------------------
    # Non-stationarity handling
    # ------------------------------------------------------------------

    def detect_nonstationarity(
        self,
        arm: Any,
        window_fraction: float = 0.5,
        significance: float = 2.0,
    ) -> bool:
        """
        Detect if an arm's reward distribution appears non-stationary.

        Compares the mean in the first half of the window to the second
        half and flags if the difference exceeds `significance` standard
        deviations.

        Parameters
        ----------
        arm : hashable
            Arm identifier.
        window_fraction : float
            Fraction for splitting window into halves (default 0.5).
        significance : float
            Number of standard deviations for flagging.

        Returns
        -------
        bool
            True if non-stationarity is detected.
        """
        stats = self._arm_stats[arm]
        values = stats.window_values
        n = len(values)

        if n < 20:
            return False

        split = int(n * window_fraction)
        first_half = np.array(values[:split])
        second_half = np.array(values[split:])

        mean_diff = abs(float(np.mean(first_half) - np.mean(second_half)))

        pooled_std = math.sqrt(
            (float(np.var(first_half, ddof=1)) / len(first_half))
            + (float(np.var(second_half, ddof=1)) / len(second_half))
        )

        if pooled_std < 1e-12:
            return False

        z_score = mean_diff / pooled_std
        return z_score > significance

    def get_windowed_scores(
        self,
        node: MCTSNode,
        parent_visits: int,
    ) -> float:
        """
        Compute UCB score using windowed (recent) statistics.

        If nonstationarity is detected, use the windowed mean instead
        of the full-history mean.
        """
        action = node.action
        if action is None or action not in self._arm_stats:
            return self.compute_score(node, parent_visits)

        stats = self._arm_stats[action]
        if not self.detect_nonstationarity(action):
            return self.compute_score(node, parent_visits)

        # Use windowed mean
        if not stats.window_values:
            return self.compute_score(node, parent_visits)

        windowed_mean = stats.windowed_mean
        n = len(stats.window_values)

        if n == 0:
            return float("inf")

        exploitation = windowed_mean
        if not self.maximize:
            exploitation = -exploitation

        log_parent = math.log(max(1, parent_visits))
        exploration = self.exploration_constant * math.sqrt(log_parent / n)

        return exploitation + exploration

    # ------------------------------------------------------------------
    # Statistics and diagnostics
    # ------------------------------------------------------------------

    def get_arm_summary(self) -> Dict[str, Any]:
        """Return summary statistics for all tracked arms."""
        summary = {
            "total_pulls": self._total_pulls,
            "n_arms_tracked": len(self._arm_stats),
            "n_arms_pruned": len(self._pruned_arms),
            "arms": {},
        }

        for arm, stats in self._arm_stats.items():
            arm_key = str(arm)
            summary["arms"][arm_key] = {
                "visits": stats.total.visit_count,
                "mean": stats.total.mean_value,
                "windowed_mean": stats.windowed_mean,
                "variance": stats.total.variance if stats.total.visit_count >= 2 else None,
                "pruned": stats.pruned,
                "ci_lower": stats.confidence_lower,
                "ci_upper": stats.confidence_upper,
            }

        return summary

    def clear_cache(self) -> None:
        """Clear d-separation and info-gain caches."""
        self._dsep_cache.clear()
        self._info_gain_estimates.clear()

    def reset(self) -> None:
        """Reset all arm statistics and caches."""
        self._arm_stats.clear()
        self._pruned_arms.clear()
        self._dsep_cache.clear()
        self._info_gain_estimates.clear()
        self._total_pulls = 0

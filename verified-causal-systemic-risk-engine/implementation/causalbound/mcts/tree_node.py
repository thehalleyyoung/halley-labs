"""
Tree node for MCTS search tree.

Stores visit counts, value sums, children, parent pointers, actions,
and state snapshots. Provides UCB computation, child selection, expansion,
and backpropagation with variance tracking.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class NodeStatistics:
    """Aggregate statistics for an MCTS node."""

    visit_count: int = 0
    value_sum: float = 0.0
    value_squared_sum: float = 0.0
    min_value: float = float("inf")
    max_value: float = float("-inf")

    @property
    def mean_value(self) -> float:
        """Return the mean value across all visits."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def variance(self) -> float:
        """Return the sample variance of observed values."""
        if self.visit_count < 2:
            return float("inf")
        mean = self.mean_value
        return (self.value_squared_sum / self.visit_count) - mean * mean

    @property
    def std_dev(self) -> float:
        """Return the standard deviation of observed values."""
        var = self.variance
        if var == float("inf"):
            return float("inf")
        return math.sqrt(max(0.0, var))

    def update(self, value: float) -> None:
        """Update statistics with a new observed value."""
        self.visit_count += 1
        self.value_sum += value
        self.value_squared_sum += value * value
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize statistics to a dictionary."""
        return {
            "visit_count": self.visit_count,
            "mean_value": self.mean_value,
            "variance": self.variance if self.visit_count >= 2 else None,
            "min_value": self.min_value if self.visit_count > 0 else None,
            "max_value": self.max_value if self.visit_count > 0 else None,
        }


class MCTSNode:
    """
    Tree node in a Monte Carlo Tree Search.

    Each node represents a partial shock assignment to interface variables.
    Children correspond to extending the assignment by one additional
    variable-value pair.

    Attributes
    ----------
    node_id : str
        Unique identifier for this node.
    parent : MCTSNode or None
        Parent node in the search tree.
    action : tuple or None
        (variable, value) pair that led from parent to this node.
    state : dict
        Partial assignment mapping variable names to shock values.
    children : dict
        Maps actions to child MCTSNode instances.
    stats : NodeStatistics
        Aggregate statistics for this node.
    pruned : bool
        Whether this node has been pruned by d-separation analysis.
    depth : int
        Depth of this node in the search tree (root = 0).
    available_actions : list
        Actions not yet expanded from this node.
    """

    def __init__(
        self,
        parent: Optional["MCTSNode"] = None,
        action: Optional[Tuple[str, float]] = None,
        state: Optional[Dict[str, float]] = None,
        available_actions: Optional[List[Tuple[str, float]]] = None,
    ) -> None:
        self.node_id: str = str(uuid.uuid4())[:12]
        self.parent: Optional[MCTSNode] = parent
        self.action: Optional[Tuple[str, float]] = action
        self.state: Dict[str, float] = dict(state) if state else {}
        self.children: Dict[Tuple[str, float], MCTSNode] = {}
        self.stats: NodeStatistics = NodeStatistics()
        self.pruned: bool = False
        self.depth: int = (parent.depth + 1) if parent else 0
        self._available_actions: List[Tuple[str, float]] = list(
            available_actions
        ) if available_actions else []
        self._expanded_actions: set = set()
        self._progressive_widening_count: int = 0
        self._rave_stats: Dict[Tuple[str, float], NodeStatistics] = {}
        self._amaf_enabled: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def visit_count(self) -> int:
        """Total number of visits to this node."""
        return self.stats.visit_count

    @property
    def mean_value(self) -> float:
        """Mean rollout value at this node."""
        return self.stats.mean_value

    @property
    def is_root(self) -> bool:
        """True if this node has no parent."""
        return self.parent is None

    @property
    def n_children(self) -> int:
        """Number of expanded children."""
        return len(self.children)

    # ------------------------------------------------------------------
    # UCB score computation
    # ------------------------------------------------------------------

    def get_ucb_score(
        self,
        exploration_constant: float = 1.414,
        parent_visits: Optional[int] = None,
        maximize: bool = True,
    ) -> float:
        """
        Compute UCB1 score for this node.

        UCB1 = mean_value + c * sqrt(ln(N) / n)

        where N = parent visits, n = this node visits, c = exploration constant.

        Parameters
        ----------
        exploration_constant : float
            Controls exploration-exploitation tradeoff. Default sqrt(2).
        parent_visits : int or None
            Override parent visit count (used when parent is virtual).
        maximize : bool
            If True, higher values are better (worst-case search).

        Returns
        -------
        float
            UCB1 score for this node.
        """
        if self.stats.visit_count == 0:
            return float("inf")

        if parent_visits is None:
            if self.parent is not None:
                parent_visits = self.parent.stats.visit_count
            else:
                parent_visits = self.stats.visit_count

        if parent_visits == 0:
            return float("inf")

        exploitation = self.stats.mean_value
        if not maximize:
            exploitation = -exploitation

        exploration = exploration_constant * math.sqrt(
            math.log(parent_visits) / self.stats.visit_count
        )

        return exploitation + exploration

    def get_ucb_tuned_score(
        self,
        exploration_constant: float = 1.414,
        parent_visits: Optional[int] = None,
        maximize: bool = True,
    ) -> float:
        """
        Compute UCB-Tuned score using empirical variance.

        UCB-Tuned replaces the standard exploration term with a tighter
        bound that accounts for variance in the reward distribution.

        UCB-Tuned = mean + sqrt( (ln N / n) * min(1/4, V_n) )

        where V_n = variance_estimate + sqrt(2 ln N / n).
        """
        if self.stats.visit_count == 0:
            return float("inf")

        if parent_visits is None:
            parent_visits = (
                self.parent.stats.visit_count if self.parent else self.stats.visit_count
            )

        if parent_visits == 0:
            return float("inf")

        n = self.stats.visit_count
        log_N = math.log(parent_visits)

        var = self.stats.variance if self.stats.visit_count >= 2 else 0.25
        var = min(var, 1e6)

        v_n = var + math.sqrt(2.0 * log_N / n)
        exploration_term = math.sqrt((log_N / n) * min(0.25, v_n))

        exploitation = self.stats.mean_value
        if not maximize:
            exploitation = -exploitation

        return exploitation + exploration_constant * exploration_term

    def get_progressive_bias_score(
        self,
        exploration_constant: float = 1.414,
        heuristic_value: float = 0.0,
        bias_weight: float = 1.0,
        parent_visits: Optional[int] = None,
        maximize: bool = True,
    ) -> float:
        """
        UCB with progressive bias from a heuristic.

        Adds a bias term that decreases with visit count, allowing
        domain heuristics to guide early search.
        """
        ucb = self.get_ucb_score(exploration_constant, parent_visits, maximize)

        if self.stats.visit_count == 0:
            return ucb

        bias = bias_weight * heuristic_value / (self.stats.visit_count + 1)
        return ucb + bias

    # ------------------------------------------------------------------
    # Child selection
    # ------------------------------------------------------------------

    def select_child(
        self,
        exploration_constant: float = 1.414,
        maximize: bool = True,
        ucb_variant: str = "ucb1",
    ) -> "MCTSNode":
        """
        Select the child with the highest UCB score.

        Parameters
        ----------
        exploration_constant : float
            Exploration parameter c.
        maximize : bool
            Whether to maximize (worst-case) or minimize.
        ucb_variant : str
            One of "ucb1", "ucb_tuned".

        Returns
        -------
        MCTSNode
            Child with highest UCB score.

        Raises
        ------
        ValueError
            If the node has no children.
        """
        if not self.children:
            raise ValueError("Cannot select child from node with no children")

        unpruned = {
            a: c for a, c in self.children.items() if not c.pruned
        }
        if not unpruned:
            unpruned = self.children

        best_score = float("-inf")
        best_child = None

        parent_visits = self.stats.visit_count

        for action, child in unpruned.items():
            if ucb_variant == "ucb_tuned":
                score = child.get_ucb_tuned_score(
                    exploration_constant, parent_visits, maximize
                )
            else:
                score = child.get_ucb_score(
                    exploration_constant, parent_visits, maximize
                )

            if score > best_score:
                best_score = score
                best_child = child

        assert best_child is not None
        return best_child

    def best_child(self, maximize: bool = True) -> "MCTSNode":
        """
        Return the child with the best mean value (exploitation only).

        Parameters
        ----------
        maximize : bool
            If True return child with highest mean value.

        Returns
        -------
        MCTSNode
            Best child by mean value.
        """
        if not self.children:
            raise ValueError("No children to select from")

        if maximize:
            return max(
                self.children.values(),
                key=lambda c: c.stats.mean_value,
            )
        else:
            return min(
                self.children.values(),
                key=lambda c: c.stats.mean_value,
            )

    def most_visited_child(self) -> "MCTSNode":
        """Return the child with the most visits."""
        if not self.children:
            raise ValueError("No children to select from")
        return max(
            self.children.values(),
            key=lambda c: c.stats.visit_count,
        )

    # ------------------------------------------------------------------
    # Expansion
    # ------------------------------------------------------------------

    def expand(
        self,
        action: Tuple[str, float],
        child_available_actions: Optional[List[Tuple[str, float]]] = None,
    ) -> "MCTSNode":
        """
        Expand a new child node for the given action.

        Parameters
        ----------
        action : tuple
            (variable, value) pair to add to the partial assignment.
        child_available_actions : list or None
            Actions available from the new child.

        Returns
        -------
        MCTSNode
            Newly created child node.
        """
        if action in self.children:
            return self.children[action]

        new_state = dict(self.state)
        var_name, var_value = action
        new_state[var_name] = var_value

        child = MCTSNode(
            parent=self,
            action=action,
            state=new_state,
            available_actions=child_available_actions,
        )

        self.children[action] = child
        self._expanded_actions.add(action)
        return child

    def expand_progressive(
        self,
        action_generator,
        k_factor: float = 1.0,
        alpha: float = 0.5,
    ) -> Optional["MCTSNode"]:
        """
        Progressive widening expansion.

        Only expand a new child if the number of children is less than
        k * N^alpha, where N is the visit count.

        Parameters
        ----------
        action_generator : callable
            Function that returns a new (variable, value) action to try.
        k_factor : float
            Scaling constant for progressive widening.
        alpha : float
            Exponent for progressive widening (0 < alpha < 1).

        Returns
        -------
        MCTSNode or None
            New child if expansion occurred, else None.
        """
        max_children = max(1, int(k_factor * (self.stats.visit_count ** alpha)))

        if len(self.children) >= max_children:
            return None

        action = action_generator(self.state, self._expanded_actions)
        if action is None:
            return None

        return self.expand(action)

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------

    def backpropagate(self, value: float) -> None:
        """
        Backpropagate a rollout value up the tree from this node to the root.

        Parameters
        ----------
        value : float
            Rollout evaluation result.
        """
        node: Optional[MCTSNode] = self
        while node is not None:
            node.stats.update(value)
            node = node.parent

    def backpropagate_with_discount(
        self, value: float, discount: float = 1.0
    ) -> None:
        """
        Backpropagate with a discount factor applied at each level.

        The value is multiplied by `discount` at each step toward the root,
        giving more weight to nodes closer to the leaf.
        """
        node: Optional[MCTSNode] = self
        current_value = value
        while node is not None:
            node.stats.update(current_value)
            current_value *= discount
            node = node.parent

    def backpropagate_rave(
        self, value: float, actions_taken: List[Tuple[str, float]]
    ) -> None:
        """
        Backpropagate with RAVE (Rapid Action Value Estimation).

        Updates AMAF statistics for all ancestors that could have taken
        any of the actions encountered during the rollout.
        """
        self.backpropagate(value)

        action_set = set(actions_taken)
        node: Optional[MCTSNode] = self.parent
        while node is not None:
            for a in action_set:
                if a not in node._rave_stats:
                    node._rave_stats[a] = NodeStatistics()
                node._rave_stats[a].update(value)
            node = node.parent

    # ------------------------------------------------------------------
    # Status queries
    # ------------------------------------------------------------------

    def is_leaf(self) -> bool:
        """True if this node has no children."""
        return len(self.children) == 0

    def is_fully_expanded(self) -> bool:
        """True if all available actions have been expanded."""
        if not self._available_actions:
            return True
        return all(a in self._expanded_actions for a in self._available_actions)

    def unexpanded_actions(self) -> List[Tuple[str, float]]:
        """Return list of actions not yet expanded."""
        return [
            a for a in self._available_actions if a not in self._expanded_actions
        ]

    def set_available_actions(
        self, actions: List[Tuple[str, float]]
    ) -> None:
        """Set the list of available actions from this node."""
        self._available_actions = list(actions)

    def mark_pruned(self, reason: str = "") -> None:
        """
        Mark this node and all descendants as pruned.

        Parameters
        ----------
        reason : str
            Human-readable reason for pruning (e.g., d-separation).
        """
        self.pruned = True
        self._pruned_reason = reason
        for child in self.children.values():
            child.mark_pruned(reason)

    # ------------------------------------------------------------------
    # Tree traversal and information
    # ------------------------------------------------------------------

    def get_path_from_root(self) -> List["MCTSNode"]:
        """Return the path from the root to this node."""
        path = []
        node: Optional[MCTSNode] = self
        while node is not None:
            path.append(node)
            node = node.parent
        path.reverse()
        return path

    def get_action_sequence(self) -> List[Tuple[str, float]]:
        """Return the sequence of actions from root to this node."""
        path = self.get_path_from_root()
        return [n.action for n in path if n.action is not None]

    def tree_size(self) -> int:
        """Count total number of nodes in the subtree rooted here."""
        count = 1
        for child in self.children.values():
            count += child.tree_size()
        return count

    def tree_depth(self) -> int:
        """Maximum depth of the subtree rooted at this node."""
        if not self.children:
            return 0
        return 1 + max(c.tree_depth() for c in self.children.values())

    def get_all_leaves(self) -> List["MCTSNode"]:
        """Collect all leaf nodes in the subtree."""
        if self.is_leaf():
            return [self]
        leaves = []
        for child in self.children.values():
            leaves.extend(child.get_all_leaves())
        return leaves

    def subtree_statistics(self) -> Dict[str, Any]:
        """
        Compute aggregate statistics for the subtree.

        Returns
        -------
        dict
            Contains total_nodes, total_visits, max_depth,
            n_leaves, n_pruned, mean_branching_factor.
        """
        total_nodes = 0
        total_visits = 0
        n_leaves = 0
        n_pruned = 0
        max_depth = 0
        branching_factors = []

        stack = [self]
        while stack:
            node = stack.pop()
            total_nodes += 1
            total_visits += node.stats.visit_count
            if node.pruned:
                n_pruned += 1
            if node.is_leaf():
                n_leaves += 1
                max_depth = max(max_depth, node.depth)
            else:
                branching_factors.append(len(node.children))
                stack.extend(node.children.values())

        mean_bf = (
            float(np.mean(branching_factors)) if branching_factors else 0.0
        )

        return {
            "total_nodes": total_nodes,
            "total_visits": total_visits,
            "max_depth": max_depth,
            "n_leaves": n_leaves,
            "n_pruned": n_pruned,
            "mean_branching_factor": mean_bf,
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self, max_depth: int = -1) -> Dict[str, Any]:
        """
        Serialize the node (and subtree) to a dictionary.

        Parameters
        ----------
        max_depth : int
            Maximum depth to serialize. -1 for unlimited.
        """
        result: Dict[str, Any] = {
            "node_id": self.node_id,
            "action": self.action,
            "state": dict(self.state),
            "stats": self.stats.to_dict(),
            "pruned": self.pruned,
            "depth": self.depth,
            "n_children": len(self.children),
        }

        if max_depth != 0 and self.children:
            result["children"] = {
                str(a): c.to_dict(max_depth - 1 if max_depth > 0 else -1)
                for a, c in self.children.items()
            }

        return result

    def __repr__(self) -> str:
        return (
            f"MCTSNode(id={self.node_id}, depth={self.depth}, "
            f"visits={self.stats.visit_count}, "
            f"mean={self.stats.mean_value:.4f}, "
            f"children={len(self.children)}, "
            f"pruned={self.pruned})"
        )

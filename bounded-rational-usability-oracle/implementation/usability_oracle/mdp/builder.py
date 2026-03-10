"""
usability_oracle.mdp.builder — MDP construction from accessibility trees.

Enumerates the MDP state space as the Cartesian product of
*(accessibility-tree focus position) × (task-progress bitvector)*.

Each state records which tree node has focus and how far the user has
progressed through the task specification's sub-goals.  Actions are
derived from the interactive capabilities of each node (click, type, tab,
scroll, read, navigate), and transition probabilities default to 1.0 for
deterministic UIs.

References
----------
- Chen, X. et al. (2001). Model-based usability evaluation — KLM-GOMS.
- John, B. E. & Kieras, D. E. (1996). The GOMS family of user interface
  analysis techniques. *ACM TCHI*.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from usability_oracle.mdp.models import Action, MDP, State, Transition


# ---------------------------------------------------------------------------
# Lightweight stand-in protocols so the builder can accept trees / task specs
# without hard-coupling to specific classes.
# ---------------------------------------------------------------------------

class _TreeProtocol:
    """Duck-typed interface for AccessibilityTree."""

    root: Any
    node_index: dict[str, Any]

    def get_node(self, node_id: str) -> Any: ...
    def get_interactive_nodes(self) -> list[Any]: ...
    def get_focusable_nodes(self) -> list[Any]: ...
    def get_visible_nodes(self) -> list[Any]: ...
    def path_between(self, src: str, dst: str) -> Optional[list[str]]: ...


class _TaskSpecProtocol:
    """Duck-typed interface for TaskSpec."""

    task_id: str
    sub_goals: list[Any]
    target_node_ids: list[str]
    description: str


# ---------------------------------------------------------------------------
# Builder configuration
# ---------------------------------------------------------------------------

@dataclass
class MDPBuilderConfig:
    """Parameters governing MDP construction."""

    max_states: int = 50_000
    max_task_progress_bits: int = 8
    include_read_actions: bool = True
    include_scroll_actions: bool = True
    include_back_actions: bool = True
    deterministic: bool = True
    base_step_cost: float = 1.0
    click_cost: float = 0.5
    type_cost_per_char: float = 0.3
    tab_cost: float = 0.2
    scroll_cost: float = 0.4
    read_cost: float = 0.6
    navigate_cost: float = 0.3
    swipe_cost: float = 0.6
    long_press_cost: float = 0.75
    discount: float = 0.99


# ---------------------------------------------------------------------------
# MDPBuilder
# ---------------------------------------------------------------------------

class MDPBuilder:
    """Construct an :class:`MDP` from an accessibility tree and task spec.

    Usage
    -----
    >>> builder = MDPBuilder()
    >>> mdp = builder.build(tree, task_spec)
    """

    def __init__(self, config: Optional[MDPBuilderConfig] = None) -> None:
        self.config = config or MDPBuilderConfig()

    # ── Public entry point ------------------------------------------------

    def build(self, tree: Any, task_spec: Any) -> MDP:
        """Build a complete MDP from *tree* and *task_spec*.

        Steps:
        1. Enumerate states from (node, progress) pairs.
        2. Enumerate available actions per interactive node.
        3. Build transitions linking (state, action) → state'.
        4. Identify goal states from the task specification.
        5. Prune unreachable states.

        Parameters
        ----------
        tree : AccessibilityTree
        task_spec : TaskSpec (or any object with ``sub_goals``,
            ``target_node_ids``, ``task_id`` attributes)

        Returns
        -------
        MDP
        """
        states = self._enumerate_states(tree, task_spec)
        actions = self._enumerate_actions(tree)
        actions = self._prune_action_space(actions, task_spec)
        transitions = self._build_transitions(states, actions, tree, task_spec)
        goal_states = self._identify_goal_states(states, task_spec)

        # Determine initial state: root node, zero progress
        initial_state_id = self._make_state_id(tree.root.id, frozenset())

        # If root isn't in the state set (not focusable), use the first
        # available state with zero progress as the entry point.
        if initial_state_id not in states:
            focusable = tree.get_focusable_nodes()
            if not focusable:
                focusable = tree.get_visible_nodes()
            if focusable:
                initial_state_id = self._make_state_id(
                    focusable[0].id, frozenset()
                )

        mdp = MDP(
            states=states,
            actions=actions,
            transitions=transitions,
            initial_state=initial_state_id,
            goal_states=goal_states,
            discount=self.config.discount,
        )

        mdp = self._prune_unreachable(mdp)
        return mdp

    # ── State enumeration -------------------------------------------------

    def _enumerate_states(
        self, tree: Any, task_spec: Any
    ) -> dict[str, State]:
        """Create one state per (focusable-node, task-progress-set) pair.

        Task progress is represented as a frozenset of completed sub-goal
        indices.  We cap the number of sub-goals to avoid combinatorial
        explosion.
        """
        focusable_nodes = tree.get_focusable_nodes()
        if not focusable_nodes:
            # Fallback: use all visible nodes
            focusable_nodes = tree.get_visible_nodes()
        if not focusable_nodes:
            focusable_nodes = [tree.root]

        # Flatten deep hierarchies to bound state space size
        if len(focusable_nodes) > 80:
            focusable_nodes = self._flatten_deep_hierarchy(tree, focusable_nodes)

        # Sub-goal indices
        sub_goal_ids: list[int] = list(
            range(min(len(getattr(task_spec, "sub_goals", [])),
                      self.config.max_task_progress_bits))
        )

        # Generate all subsets of sub-goal indices (power set)
        progress_sets: list[frozenset[int]] = []
        for r in range(len(sub_goal_ids) + 1):
            for combo in itertools.combinations(sub_goal_ids, r):
                progress_sets.append(frozenset(combo))
                if len(progress_sets) * len(focusable_nodes) > self.config.max_states:
                    break
            if len(progress_sets) * len(focusable_nodes) > self.config.max_states:
                break

        states: dict[str, State] = {}
        for node in focusable_nodes:
            for progress in progress_sets:
                state = self._state_from_tree_position(tree, node.id, progress, task_spec)
                states[state.state_id] = state

        return states

    def _state_from_tree_position(
        self,
        tree: Any,
        node_id: str,
        task_progress: frozenset[int],
        task_spec: Any,
    ) -> State:
        """Create a :class:`State` from a tree node and progress set."""
        state_id = self._make_state_id(node_id, task_progress)
        node = tree.get_node(node_id)

        features = self._extract_features(tree, node_id)
        n_subgoals = len(getattr(task_spec, "sub_goals", []))
        progress_frac = len(task_progress) / max(n_subgoals, 1)
        features["task_progress"] = progress_frac

        target_ids = set(getattr(task_spec, "target_node_ids", []))
        all_done = n_subgoals > 0 and len(task_progress) == n_subgoals
        is_goal = all_done and (node_id in target_ids or not target_ids)

        # Early completion: if task allows it and >=80% subgoals are done
        if (
            not is_goal
            and n_subgoals > 0
            and getattr(task_spec, "allow_early_completion", False)
            and progress_frac >= 0.8
            and (node_id in target_ids or not target_ids)
        ):
            is_goal = True

        label_name = getattr(node, "name", node_id) if node else node_id
        label = f"{label_name} [progress={len(task_progress)}/{n_subgoals}]"

        return State(
            state_id=state_id,
            features=features,
            label=label,
            is_terminal=is_goal,
            is_goal=is_goal,
            metadata={
                "node_id": node_id,
                "task_progress": sorted(task_progress),
                "working_memory": list(task_progress)[-4:] if task_progress else [],
            },
        )

    def _extract_features(self, tree: Any, node_id: str) -> dict[str, float]:
        """Compute numeric features for a tree node."""
        node = tree.get_node(node_id)
        if node is None:
            return {"depth": 0.0, "n_children": 0.0, "n_siblings": 0.0}

        # Depth in tree
        depth = float(getattr(node, "depth", 0))

        # Number of children
        children = getattr(node, "children", [])
        n_children = float(len(children))

        # Siblings
        parent_id = getattr(node, "parent_id", None)
        n_siblings = 0.0
        if parent_id:
            parent = tree.get_node(parent_id)
            if parent:
                n_siblings = float(len(getattr(parent, "children", [])) - 1)

        # Visual complexity: count visible nodes in subtree
        subtree_size = float(getattr(node, "subtree_size", lambda: 1)())

        # Interactive density: interactive children / total children
        interactive_count = sum(
            1 for c in children if getattr(c, "is_interactive", lambda: False)()
        )
        density = interactive_count / max(n_children, 1.0)

        # Bounding box area (normalised)
        bbox = getattr(node, "bounding_box", None)
        area = 0.0
        if bbox is not None:
            area = getattr(bbox, "area", 0.0)

        # Working memory load estimate: depth serves as proxy
        wm_load = min(depth, 7.0)

        return {
            "depth": depth,
            "n_children": n_children,
            "n_siblings": n_siblings,
            "subtree_size": subtree_size,
            "interaction_density": density,
            "bbox_area": area,
            "working_memory_load": wm_load,
        }

    # ── Action enumeration ------------------------------------------------

    def _enumerate_actions(self, tree: Any) -> dict[str, Action]:
        """Create actions for every interactive capability in the tree."""
        actions: dict[str, Action] = {}

        interactive_nodes = tree.get_interactive_nodes()
        for node in interactive_nodes:
            role = getattr(node, "role", "")
            nid = node.id

            # Click action for every interactive node
            aid = f"click:{nid}"
            actions[aid] = Action(
                action_id=aid,
                action_type="click",
                target_node_id=nid,
                description=f"Click {getattr(node, 'name', nid)}",
                preconditions=[f"visible:{nid}"],
            )

            # Type action for text inputs
            if role in ("textbox", "searchbox", "input", "textarea", "combobox"):
                aid = f"type:{nid}"
                actions[aid] = Action(
                    action_id=aid,
                    action_type="type",
                    target_node_id=nid,
                    description=f"Type into {getattr(node, 'name', nid)}",
                    preconditions=[f"focused:{nid}"],
                )

            # Select action for list-like widgets
            if role in ("option", "menuitem", "tab", "treeitem", "listitem"):
                aid = f"select:{nid}"
                actions[aid] = Action(
                    action_id=aid,
                    action_type="select",
                    target_node_id=nid,
                    description=f"Select {getattr(node, 'name', nid)}",
                    preconditions=[f"visible:{nid}"],
                )

        # Read actions for nodes with text content
        if self.config.include_read_actions:
            for node in tree.get_visible_nodes():
                name = getattr(node, "name", "")
                if name and len(name) > 0:
                    aid = f"read:{node.id}"
                    actions[aid] = Action(
                        action_id=aid,
                        action_type="read",
                        target_node_id=node.id,
                        description=f"Read '{name[:30]}'",
                        preconditions=[f"visible:{node.id}"],
                    )

        # Global tab action
        aid = "tab:forward"
        actions[aid] = Action(
            action_id=aid,
            action_type="tab",
            target_node_id=None,
            description="Press Tab (move focus forward)",
        )
        aid = "tab:backward"
        actions[aid] = Action(
            action_id=aid,
            action_type="tab",
            target_node_id=None,
            description="Press Shift+Tab (move focus backward)",
        )

        # Scroll actions
        if self.config.include_scroll_actions:
            for direction in ("up", "down"):
                aid = f"scroll:{direction}"
                actions[aid] = Action(
                    action_id=aid,
                    action_type="scroll",
                    target_node_id=None,
                    description=f"Scroll {direction}",
                )

        # Back / navigate action
        if self.config.include_back_actions:
            aid = "navigate:back"
            actions[aid] = Action(
                action_id=aid,
                action_type="back",
                target_node_id=None,
                description="Navigate back",
            )

        return actions

    # ── Transition construction -------------------------------------------

    def _build_transitions(
        self,
        states: dict[str, State],
        actions: dict[str, Action],
        tree: Any,
        task_spec: Any,
    ) -> list[Transition]:
        """Build the transition list linking states via actions."""
        transitions: list[Transition] = []
        focusable_ids = self._ordered_focusable_ids(tree)
        focusable_set = set(focusable_ids)

        target_node_to_subgoal: dict[str, int] = {}
        for i, sg in enumerate(getattr(task_spec, "sub_goals", [])):
            tgt = getattr(sg, "target_node_id", None)
            if tgt is not None:
                target_node_to_subgoal[tgt] = i

        for sid, state in states.items():
            node_id = state.metadata.get("node_id", "")
            progress = frozenset(state.metadata.get("task_progress", []))

            if state.is_terminal:
                continue

            available_actions = self._available_actions(state, actions, tree)

            for aid in available_actions:
                action = actions[aid]
                targets = self._compute_targets(
                    action, node_id, progress, tree, task_spec,
                    focusable_ids, target_node_to_subgoal,
                )
                for target_node, new_progress, prob in targets:
                    target_sid = self._make_state_id(target_node, new_progress)
                    if target_sid not in states:
                        continue
                    cost = self._compute_transition_cost(
                        action, state, states[target_sid], tree
                    )
                    transitions.append(Transition(
                        source=sid,
                        action=aid,
                        target=target_sid,
                        probability=prob,
                        cost=cost,
                    ))

        return transitions

    def _available_actions(
        self, state: State, actions: dict[str, Action], tree: Any
    ) -> list[str]:
        """Return action IDs applicable in *state*."""
        node_id = state.metadata.get("node_id", "")
        node = tree.get_node(node_id)
        result: list[str] = []
        for aid, action in actions.items():
            # Global actions are always available
            if action.target_node_id is None:
                result.append(aid)
                continue
            # Check basic visibility precondition
            target_node = tree.get_node(action.target_node_id)
            if target_node is None:
                continue
            if getattr(target_node, "state", None) and getattr(
                target_node.state, "hidden", False
            ):
                continue
            # Click/select: target must exist
            if action.action_type in ("click", "select", "read"):
                result.append(aid)
            # Type: must be focused on the target
            elif action.action_type == "type" and action.target_node_id == node_id:
                result.append(aid)
        return result

    def _compute_targets(
        self,
        action: Action,
        current_node: str,
        progress: frozenset[int],
        tree: Any,
        task_spec: Any,
        focusable_ids: list[str],
        target_node_to_subgoal: dict[str, int],
    ) -> list[tuple[str, frozenset[int], float]]:
        """Determine (target_node, new_progress, probability) tuples."""
        results: list[tuple[str, frozenset[int], float]] = []
        p = 1.0 if self.config.deterministic else 0.9

        if action.action_type == "click" and action.target_node_id:
            target = action.target_node_id
            new_prog = self._update_progress(target, progress, target_node_to_subgoal)
            results.append((target, new_prog, p))
            if not self.config.deterministic:
                results.append((current_node, progress, 1.0 - p))

        elif action.action_type == "type" and action.target_node_id:
            target = action.target_node_id
            new_prog = self._update_progress(target, progress, target_node_to_subgoal)
            results.append((target, new_prog, p))
            if not self.config.deterministic:
                results.append((current_node, progress, 1.0 - p))

        elif action.action_type == "select" and action.target_node_id:
            target = action.target_node_id
            new_prog = self._update_progress(target, progress, target_node_to_subgoal)
            results.append((target, new_prog, p))
            if not self.config.deterministic:
                results.append((current_node, progress, 1.0 - p))

        elif action.action_type == "tab":
            if not focusable_ids:
                results.append((current_node, progress, 1.0))
            else:
                try:
                    idx = focusable_ids.index(current_node)
                except ValueError:
                    idx = 0
                if action.action_id == "tab:forward":
                    next_idx = (idx + 1) % len(focusable_ids)
                else:
                    next_idx = (idx - 1) % len(focusable_ids)
                results.append((focusable_ids[next_idx], progress, 1.0))

        elif action.action_type == "scroll":
            # Scroll keeps focus but may reveal different nodes
            results.append((current_node, progress, 1.0))

        elif action.action_type == "read" and action.target_node_id:
            # Reading does not move focus or change progress
            results.append((current_node, progress, 1.0))

        elif action.action_type in ("back", "navigate"):
            # Navigate back to parent node
            node = tree.get_node(current_node)
            parent_id = getattr(node, "parent_id", None) if node else None
            if parent_id and parent_id in set(focusable_ids):
                results.append((parent_id, progress, 1.0))
            elif focusable_ids:
                results.append((focusable_ids[0], progress, 1.0))
            else:
                results.append((current_node, progress, 1.0))

        else:
            results.append((current_node, progress, 1.0))

        return results

    def _update_progress(
        self,
        node_id: str,
        progress: frozenset[int],
        target_node_to_subgoal: dict[str, int],
    ) -> frozenset[int]:
        """Return updated progress set if *node_id* completes a sub-goal."""
        sg_idx = target_node_to_subgoal.get(node_id)
        if sg_idx is not None:
            return progress | {sg_idx}
        return progress

    def _compute_transition_cost(
        self,
        action: Action,
        source_state: State,
        target_state: State,
        tree: Any,
    ) -> float:
        """Compute cognitive + motor cost of a transition.

        Cost components:
        - Base step cost (constant attention cost per action)
        - Action-type specific cost (Fitts' law approximation for click)
        - Working-memory transition cost (if WM load changes)
        """
        cfg = self.config
        cost = cfg.base_step_cost

        # Power law of practice: reduce cost when task progress is high (>75%)
        progress_frac = source_state.features.get("task_progress", 0.0)
        if progress_frac > 0.75:
            cost *= 0.8

        if action.action_type == "click":
            cost += cfg.click_cost
            # Fitts' law approximation: cost ∝ log₂(distance / size + 1)
            src_node = tree.get_node(source_state.metadata.get("node_id", ""))
            tgt_node = tree.get_node(target_state.metadata.get("node_id", ""))
            if src_node and tgt_node:
                src_bbox = getattr(src_node, "bounding_box", None)
                tgt_bbox = getattr(tgt_node, "bounding_box", None)
                if src_bbox is not None and tgt_bbox is not None:
                    dist = src_bbox.distance_to(tgt_bbox)
                    width = max(getattr(tgt_bbox, "width", 1.0), 1.0)
                    fitts = math.log2(dist / width + 1.0)
                    cost += fitts * 0.1

        elif action.action_type == "type":
            char_count = action.metadata.get("char_count", 5) if action.metadata else 5
            cost += cfg.type_cost_per_char * float(char_count)

        elif action.action_type == "tab":
            cost += cfg.tab_cost

        elif action.action_type == "scroll":
            cost += cfg.scroll_cost

        elif action.action_type == "swipe":
            cost += cfg.swipe_cost
            # Swipe cost scales with distance like scroll but higher precision
            src_node = tree.get_node(source_state.metadata.get("node_id", ""))
            tgt_node = tree.get_node(target_state.metadata.get("node_id", ""))
            if src_node and tgt_node:
                src_bbox = getattr(src_node, "bounding_box", None)
                tgt_bbox = getattr(tgt_node, "bounding_box", None)
                if src_bbox is not None and tgt_bbox is not None:
                    dist = src_bbox.distance_to(tgt_bbox)
                    cost += 0.15 * math.log2(dist + 1.0)

        elif action.action_type == "long_press":
            cost += cfg.long_press_cost

        elif action.action_type == "read":
            cost += cfg.read_cost

        elif action.action_type in ("navigate", "back"):
            cost += cfg.navigate_cost

        elif action.action_type == "select":
            cost += cfg.click_cost
            # Hick-Hyman approximation for selection cost
            n_choices = source_state.features.get("n_children", 1.0)
            if n_choices > 1:
                cost += 0.15 * math.log2(n_choices)

        # Working-memory cost: penalty for WM load change
        wm_src = source_state.features.get("working_memory_load", 0.0)
        wm_tgt = target_state.features.get("working_memory_load", 0.0)
        wm_delta = abs(wm_tgt - wm_src)
        cost += 0.05 * wm_delta

        return cost

    # ── Goal identification -----------------------------------------------

    def _identify_goal_states(
        self, states: dict[str, State], task_spec: Any
    ) -> set[str]:
        """Return IDs of states satisfying the task specification."""
        return {sid for sid, s in states.items() if s.is_goal}

    # ── Pruning -----------------------------------------------------------

    def _prune_unreachable(self, mdp: MDP) -> MDP:
        """Remove states not reachable from the initial state."""
        reachable = mdp.reachable_states()
        if len(reachable) == len(mdp.states):
            return mdp

        new_states = {sid: s for sid, s in mdp.states.items() if sid in reachable}
        new_transitions = [
            t for t in mdp.transitions
            if t.source in reachable and t.target in reachable
        ]
        new_goals = mdp.goal_states & reachable

        return MDP(
            states=new_states,
            actions=mdp.actions,
            transitions=new_transitions,
            initial_state=mdp.initial_state,
            goal_states=new_goals,
            discount=mdp.discount,
        )

    # ── Helpers -----------------------------------------------------------

    def _flatten_deep_hierarchy(
        self,
        tree: Any,
        focusable_nodes: list[Any],
        max_effective_depth: int = 5,
    ) -> list[Any]:
        """Collapse deep tree siblings into representative group nodes.

        For trees with deeply nested hierarchies (10+ levels, 100+
        interactive elements), enumerating every focusable node creates
        prohibitively large state spaces.  This method groups nodes that
        share a common ancestor at *max_effective_depth* and retains only
        the most interactive child per group as a representative.

        Parameters
        ----------
        tree : AccessibilityTree
        focusable_nodes : list
            Original focusable-node list from the tree.
        max_effective_depth : int
            Maximum depth before siblings are collapsed.

        Returns
        -------
        list
            Reduced list of effective focusable nodes.
        """
        shallow: list[Any] = []
        deep_groups: dict[str, list[Any]] = defaultdict(list)

        for node in focusable_nodes:
            depth = getattr(node, "depth", 0)
            if depth <= max_effective_depth:
                shallow.append(node)
            else:
                # Walk up to find the ancestor at max_effective_depth
                ancestor = node
                for _ in range(depth - max_effective_depth):
                    parent_id = getattr(ancestor, "parent_id", None)
                    if parent_id is None:
                        break
                    parent = tree.get_node(parent_id)
                    if parent is None:
                        break
                    ancestor = parent
                deep_groups[ancestor.id].append(node)

        # For each group, keep the child with the most interactive descendants
        for _ancestor_id, group in deep_groups.items():
            best = max(
                group,
                key=lambda n: len(getattr(n, "children", [])),
            )
            shallow.append(best)

        return shallow

    def _prune_action_space(
        self,
        actions: dict[str, Action],
        task_spec: Any,
        max_actions: int = 50,
    ) -> dict[str, Action]:
        """Prune the action dict when it exceeds *max_actions*.

        Pruning priority:
        1. Always keep global actions (tab, scroll, back).
        2. Always keep actions targeting task-spec nodes.
        3. Score remaining actions by proximity to task targets and
           retain the top-k to fill the budget.

        Parameters
        ----------
        actions : dict[str, Action]
        task_spec : TaskSpec
        max_actions : int

        Returns
        -------
        dict[str, Action]
        """
        if len(actions) <= max_actions:
            return actions

        target_ids = set(getattr(task_spec, "target_node_ids", []))

        kept: dict[str, Action] = {}
        remaining: list[tuple[str, Action]] = []

        for aid, action in actions.items():
            # Global actions (no target node)
            if action.target_node_id is None:
                kept[aid] = action
            # Actions targeting task-spec nodes
            elif action.target_node_id in target_ids:
                kept[aid] = action
            else:
                remaining.append((aid, action))

        budget = max_actions - len(kept)
        if budget <= 0:
            return kept

        # Score remaining by whether they share a prefix with any target id
        # (proxy for tree proximity when full path info is unavailable)
        def _proximity_score(action: Action) -> float:
            if not target_ids or action.target_node_id is None:
                return 0.0
            nid = action.target_node_id
            best = 0.0
            for tid in target_ids:
                common = len(
                    [a for a, b in zip(nid, tid) if a == b]
                )
                best = max(best, common / max(len(tid), 1))
            return best

        remaining.sort(key=lambda pair: _proximity_score(pair[1]), reverse=True)
        for aid, action in remaining[:budget]:
            kept[aid] = action

        return kept

    @staticmethod
    def _make_state_id(node_id: str, progress: frozenset[int]) -> str:
        """Deterministic state ID from node and progress set."""
        prog_str = ",".join(str(i) for i in sorted(progress))
        return f"{node_id}:{prog_str}" if prog_str else f"{node_id}:"

    def _ordered_focusable_ids(self, tree: Any) -> list[str]:
        """Return focusable node IDs in tab order (BFS by default)."""
        focusable = tree.get_focusable_nodes()
        if not focusable:
            focusable = tree.get_visible_nodes()
        # Sort by tab-index then BFS order (depth then index_in_parent)
        return sorted(
            [n.id for n in focusable],
            key=lambda nid: (
                getattr(tree.get_node(nid), "depth", 0),
                getattr(tree.get_node(nid), "index_in_parent", 0),
            ),
        )

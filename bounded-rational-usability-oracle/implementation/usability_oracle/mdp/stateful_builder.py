"""Stateful MDP builder for interactive components.

Extends :class:`MDPBuilder` to model expand/collapse transitions for
real interactive components (accordions, tabs). In the base builder,
state = (focus_node, task_progress). Here we add a third dimension:
**component state** — the expand/collapse or selected-tab configuration.

For an accordion with N sections, each section can be expanded or collapsed,
adding up to 2^N component states. For tabs with K panels, the active tab
adds K component states. We bound this by only tracking components relevant
to the task specification.

Each state transition between component configurations incurs cognitive cost:
- **Accordion expand**: visual scanning + memory update ≈ 1.5 bits
- **Accordion collapse**: memory release ≈ 0.5 bits
- **Tab switch**: context switch + visual reorientation ≈ 2.0 bits
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from usability_oracle.accessibility.models import AccessibilityNode, AccessibilityTree
from usability_oracle.mdp.models import Action, MDP, State, Transition
from usability_oracle.mdp.builder import MDPBuilder, MDPBuilderConfig


# ── Component state costs ─────────────────────────────────────────────────

COMPONENT_COSTS: Dict[str, float] = {
    "accordion_expand": 1.5,
    "accordion_collapse": 0.5,
    "tab_switch": 2.0,
    "accordion_scan": 0.8,   # Scanning accordion headers to find target
    "tab_scan": 0.6,         # Scanning tab labels
}


# ── Component descriptor ──────────────────────────────────────────────────

@dataclass
class ComponentDescriptor:
    """Describes an interactive component detected in the tree."""
    component_id: str
    component_type: str          # "accordion" or "tabs"
    section_ids: List[str] = field(default_factory=list)
    section_names: List[str] = field(default_factory=list)
    content_ids: List[str] = field(default_factory=list)
    initial_expanded: Set[int] = field(default_factory=set)


@dataclass(frozen=True)
class ComponentState:
    """Immutable component state configuration."""
    expanded_sections: FrozenSet[int] = frozenset()
    active_tab: int = 0

    def key(self) -> str:
        exp = ",".join(str(i) for i in sorted(self.expanded_sections))
        return f"exp={exp};tab={self.active_tab}"


# ── Component detector ────────────────────────────────────────────────────

def detect_components(tree: AccessibilityTree) -> List[ComponentDescriptor]:
    """Find accordion and tab components in the accessibility tree."""
    components: List[ComponentDescriptor] = []

    for node in tree.node_index.values():
        ctype = node.properties.get("component_type", "")
        dm = node.properties.get("data-module", "")
        cls = node.properties.get("class", "")

        # Accordion detection
        if ctype == "accordion" or "govuk-accordion" in dm or "govuk-accordion" in cls:
            sections = []
            section_names = []
            content_ids = []
            initial_expanded: Set[int] = set()

            for i, child in enumerate(node.children):
                child_cls = child.properties.get("class", "")
                if "accordion__section" in child_cls and "header" not in child_cls:
                    sections.append(child.id)
                    # Find section heading
                    for desc in child.get_descendants():
                        if desc.role == "heading" or "section-heading" in desc.properties.get("class", ""):
                            section_names.append(desc.name or f"Section {i+1}")
                            break
                    else:
                        section_names.append(f"Section {i+1}")
                    # Find content div
                    for desc in child.get_descendants():
                        if "section-content" in desc.properties.get("class", ""):
                            content_ids.append(desc.id)
                            break
                    else:
                        content_ids.append(child.id)
                    # Check if initially expanded
                    if child.state.expanded:
                        initial_expanded.add(len(sections) - 1)

            if sections:
                components.append(ComponentDescriptor(
                    component_id=node.id,
                    component_type="accordion",
                    section_ids=sections,
                    section_names=section_names,
                    content_ids=content_ids,
                    initial_expanded=initial_expanded,
                ))

        # Tab detection
        if ctype == "tabs" or node.role == "tablist" or "govuk-tabs" in dm:
            tab_nodes = []
            panel_ids = []
            tab_names = []

            for desc in node.get_descendants():
                if desc.role == "tab":
                    tab_nodes.append(desc.id)
                    tab_names.append(desc.name or f"Tab {len(tab_nodes)}")
                    ctrl = desc.properties.get("aria-controls", "")
                    if ctrl:
                        panel_ids.append(ctrl)

            if tab_nodes:
                components.append(ComponentDescriptor(
                    component_id=node.id,
                    component_type="tabs",
                    section_ids=tab_nodes,
                    section_names=tab_names,
                    content_ids=panel_ids,
                    initial_expanded={0},  # First tab active
                ))

    return components


# ── Stateful MDP builder ─────────────────────────────────────────────────

class StatefulMDPBuilder:
    """Build an MDP that includes component-state transitions.

    Wraps :class:`MDPBuilder` and augments the state space with
    component configurations. The resulting MDP models:

    - Accordion expand/collapse as explicit actions with cognitive costs
    - Tab switching as explicit actions with context-switch costs
    - Content visibility gated on component state
    """

    def __init__(
        self,
        config: Optional[MDPBuilderConfig] = None,
        max_component_states: int = 32,
    ) -> None:
        self.config = config or MDPBuilderConfig()
        self.base_builder = MDPBuilder(self.config)
        self.max_component_states = max_component_states

    def build(
        self,
        tree: AccessibilityTree,
        task_spec: Any,
        components: Optional[List[ComponentDescriptor]] = None,
    ) -> MDP:
        """Build an MDP with stateful component transitions.

        Parameters
        ----------
        tree : AccessibilityTree
        task_spec : duck-typed task spec with sub_goals, target_node_ids
        components : list of detected components (auto-detected if None)

        Returns
        -------
        MDP with component-state-augmented state space
        """
        if components is None:
            components = detect_components(tree)

        if not components:
            # No interactive components — fall back to base builder
            return self.base_builder.build(tree, task_spec)

        # Build base MDP
        base_mdp = self.base_builder.build(tree, task_spec)

        # Augment with component states
        return self._augment_with_components(base_mdp, tree, components, task_spec)

    def _augment_with_components(
        self,
        base_mdp: MDP,
        tree: AccessibilityTree,
        components: List[ComponentDescriptor],
        task_spec: Any,
    ) -> MDP:
        """Add component-state transitions to the base MDP."""
        # Generate bounded component state configurations
        comp_states = self._enumerate_component_states(components)

        if len(comp_states) <= 1:
            return base_mdp

        # Augment states
        new_states: Dict[str, State] = {}
        for sid, state in base_mdp.states.items():
            for cs in comp_states:
                cs_key = cs.key()
                aug_sid = f"{sid}|{cs_key}"
                new_states[aug_sid] = State(
                    state_id=aug_sid,
                    features=dict(state.features),
                    label=f"{state.label} [{cs_key}]",
                    is_terminal=state.is_terminal,
                    is_goal=state.is_goal,
                    metadata={
                        **state.metadata,
                        "component_state": cs_key,
                        "expanded_sections": sorted(cs.expanded_sections),
                        "active_tab": cs.active_tab,
                    },
                )

        # Copy base transitions for each component state
        new_transitions: List[Transition] = []
        for t in base_mdp.transitions:
            for cs in comp_states:
                cs_key = cs.key()
                new_transitions.append(Transition(
                    source=f"{t.source}|{cs_key}",
                    action=t.action,
                    target=f"{t.target}|{cs_key}",
                    probability=t.probability,
                    cost=t.cost,
                ))

        # Add component-state transition actions
        new_actions = dict(base_mdp.actions)
        comp_transitions = self._build_component_transitions(
            base_mdp, components, comp_states, new_states, new_actions
        )
        new_transitions.extend(comp_transitions)

        # Filter to valid transitions
        valid_sids = set(new_states.keys())
        new_transitions = [
            t for t in new_transitions
            if t.source in valid_sids and t.target in valid_sids
        ]

        # Set initial state
        initial_cs = comp_states[0]
        initial_sid = f"{base_mdp.initial_state}|{initial_cs.key()}"

        # Goal states: any component state is acceptable
        new_goals: Set[str] = set()
        for gs in base_mdp.goal_states:
            for cs in comp_states:
                aug = f"{gs}|{cs.key()}"
                if aug in new_states:
                    new_goals.add(aug)

        return MDP(
            states=new_states,
            actions=new_actions,
            transitions=new_transitions,
            initial_state=initial_sid,
            goal_states=new_goals,
            discount=base_mdp.discount,
        )

    def _enumerate_component_states(
        self, components: List[ComponentDescriptor]
    ) -> List[ComponentState]:
        """Generate bounded set of component state configurations."""
        states = [ComponentState()]

        for comp in components:
            if comp.component_type == "accordion":
                n = len(comp.section_ids)
                # For small accordions, enumerate all; for large, sample
                if n <= 4:
                    new_states = []
                    for r in range(n + 1):
                        for combo in itertools.combinations(range(n), r):
                            for base in states:
                                new_states.append(ComponentState(
                                    expanded_sections=base.expanded_sections | frozenset(combo),
                                    active_tab=base.active_tab,
                                ))
                                if len(new_states) > self.max_component_states:
                                    return new_states[:self.max_component_states]
                    states = new_states
                else:
                    # For large accordions: all-collapsed, each-one-expanded
                    new_states = []
                    for base in states:
                        new_states.append(base)
                        for i in range(n):
                            new_states.append(ComponentState(
                                expanded_sections=base.expanded_sections | frozenset({i}),
                                active_tab=base.active_tab,
                            ))
                    states = new_states[:self.max_component_states]

            elif comp.component_type == "tabs":
                n = len(comp.section_ids)
                new_states = []
                for base in states:
                    for tab_idx in range(n):
                        new_states.append(ComponentState(
                            expanded_sections=base.expanded_sections,
                            active_tab=tab_idx,
                        ))
                        if len(new_states) > self.max_component_states:
                            return new_states[:self.max_component_states]
                states = new_states

        # Deduplicate
        seen: Set[str] = set()
        unique = []
        for s in states:
            k = s.key()
            if k not in seen:
                seen.add(k)
                unique.append(s)
        return unique[:self.max_component_states]

    def _build_component_transitions(
        self,
        base_mdp: MDP,
        components: List[ComponentDescriptor],
        comp_states: List[ComponentState],
        all_states: Dict[str, State],
        actions: Dict[str, Action],
    ) -> List[Transition]:
        """Create transitions for accordion expand/collapse and tab switching."""
        transitions: List[Transition] = []
        cs_index = {cs.key(): cs for cs in comp_states}

        for comp in components:
            if comp.component_type == "accordion":
                transitions.extend(
                    self._accordion_transitions(
                        comp, base_mdp, comp_states, cs_index, all_states, actions
                    )
                )
            elif comp.component_type == "tabs":
                transitions.extend(
                    self._tab_transitions(
                        comp, base_mdp, comp_states, cs_index, all_states, actions
                    )
                )

        return transitions

    def _accordion_transitions(
        self,
        comp: ComponentDescriptor,
        base_mdp: MDP,
        comp_states: List[ComponentState],
        cs_index: Dict[str, ComponentState],
        all_states: Dict[str, State],
        actions: Dict[str, Action],
    ) -> List[Transition]:
        transitions = []
        n = len(comp.section_ids)

        for i in range(n):
            # Expand action
            expand_aid = f"expand:{comp.component_id}:{i}"
            actions[expand_aid] = Action(
                action_id=expand_aid,
                action_type="click",
                target_node_id=comp.section_ids[i] if i < len(comp.section_ids) else None,
                description=f"Expand '{comp.section_names[i][:30]}'",
                metadata={"component_action": "expand", "section": i},
            )

            # Collapse action
            collapse_aid = f"collapse:{comp.component_id}:{i}"
            actions[collapse_aid] = Action(
                action_id=collapse_aid,
                action_type="click",
                target_node_id=comp.section_ids[i] if i < len(comp.section_ids) else None,
                description=f"Collapse '{comp.section_names[i][:30]}'",
                metadata={"component_action": "collapse", "section": i},
            )

            # Create transitions between component states
            for cs in comp_states:
                for base_sid in base_mdp.states:
                    src_aug = f"{base_sid}|{cs.key()}"
                    if src_aug not in all_states:
                        continue

                    # Expand: add section i
                    if i not in cs.expanded_sections:
                        new_cs = ComponentState(
                            expanded_sections=cs.expanded_sections | frozenset({i}),
                            active_tab=cs.active_tab,
                        )
                        new_key = new_cs.key()
                        if new_key in cs_index:
                            tgt_aug = f"{base_sid}|{new_key}"
                            if tgt_aug in all_states:
                                transitions.append(Transition(
                                    source=src_aug,
                                    action=expand_aid,
                                    target=tgt_aug,
                                    probability=1.0,
                                    cost=COMPONENT_COSTS["accordion_expand"],
                                ))

                    # Collapse: remove section i
                    if i in cs.expanded_sections:
                        new_cs = ComponentState(
                            expanded_sections=cs.expanded_sections - frozenset({i}),
                            active_tab=cs.active_tab,
                        )
                        new_key = new_cs.key()
                        if new_key in cs_index:
                            tgt_aug = f"{base_sid}|{new_key}"
                            if tgt_aug in all_states:
                                transitions.append(Transition(
                                    source=src_aug,
                                    action=collapse_aid,
                                    target=tgt_aug,
                                    probability=1.0,
                                    cost=COMPONENT_COSTS["accordion_collapse"],
                                ))

        return transitions

    def _tab_transitions(
        self,
        comp: ComponentDescriptor,
        base_mdp: MDP,
        comp_states: List[ComponentState],
        cs_index: Dict[str, ComponentState],
        all_states: Dict[str, State],
        actions: Dict[str, Action],
    ) -> List[Transition]:
        transitions = []
        n = len(comp.section_ids)

        for target_tab in range(n):
            switch_aid = f"switch_tab:{comp.component_id}:{target_tab}"
            actions[switch_aid] = Action(
                action_id=switch_aid,
                action_type="select",
                target_node_id=comp.section_ids[target_tab] if target_tab < len(comp.section_ids) else None,
                description=f"Switch to tab '{comp.section_names[target_tab][:30]}'",
                metadata={"component_action": "tab_switch", "tab": target_tab},
            )

            for cs in comp_states:
                if cs.active_tab == target_tab:
                    continue
                for base_sid in base_mdp.states:
                    src_aug = f"{base_sid}|{cs.key()}"
                    if src_aug not in all_states:
                        continue
                    new_cs = ComponentState(
                        expanded_sections=cs.expanded_sections,
                        active_tab=target_tab,
                    )
                    new_key = new_cs.key()
                    if new_key in cs_index:
                        tgt_aug = f"{base_sid}|{new_key}"
                        if tgt_aug in all_states:
                            transitions.append(Transition(
                                source=src_aug,
                                action=switch_aid,
                                target=tgt_aug,
                                probability=1.0,
                                cost=COMPONENT_COSTS["tab_switch"],
                            ))

        return transitions

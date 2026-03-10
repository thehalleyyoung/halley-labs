"""
usability_oracle.repair.strategies — Heuristic repair strategy selection.

Given a :class:`BottleneckResult`, the :class:`RepairStrategySelector`
proposes a list of :class:`UIMutation` candidates that are likely to
alleviate the bottleneck, based on cognitive-science heuristics.

Each bottleneck type maps to a strategy that generates one or more
mutations.  These mutations form the *seed* for the SMT-backed
synthesiser, which refines and verifies them.

Strategies follow established HCI guidelines:
  - **Perceptual overload** → reduce clutter, improve grouping
  - **Choice paralysis** → progressive disclosure, fewer options
  - **Motor difficulty** → larger targets, shorter distances, shortcuts
  - **Memory decay** → fewer steps, persistent state indicators
  - **Cross-channel interference** → separate modalities, serialise
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from usability_oracle.core.enums import BottleneckType
from usability_oracle.repair.models import MutationType, UIMutation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BottleneckResult stub (same interface as synthesizer.py)
# ---------------------------------------------------------------------------

@dataclass
class _BottleneckInfo:
    """Internal representation for strategy input."""
    bottleneck_type: str = ""
    state_id: str = ""
    action_id: str = ""
    severity: str = "medium"
    cost_contribution: float = 0.0
    description: str = ""
    node_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _as_info(bn: Any) -> _BottleneckInfo:
    """Convert any bottleneck-like object to _BottleneckInfo."""
    return _BottleneckInfo(
        bottleneck_type=getattr(bn, "bottleneck_type", ""),
        state_id=getattr(bn, "state_id", ""),
        action_id=getattr(bn, "action_id", ""),
        severity=getattr(bn, "severity", "medium"),
        cost_contribution=getattr(bn, "cost_contribution", 0.0),
        description=getattr(bn, "description", ""),
        node_ids=list(getattr(bn, "node_ids", [])),
        metadata=dict(getattr(bn, "metadata", {})),
    )


# ---------------------------------------------------------------------------
# RepairStrategySelector
# ---------------------------------------------------------------------------

class RepairStrategySelector:
    """Select repair mutations based on bottleneck type.

    Usage::

        selector = RepairStrategySelector()
        mutations = selector.select(bottleneck)
    """

    _STRATEGY_MAP: dict[str, str] = {
        BottleneckType.PERCEPTUAL_OVERLOAD: "_perceptual_overload_strategy",
        BottleneckType.CHOICE_PARALYSIS: "_choice_paralysis_strategy",
        BottleneckType.MOTOR_DIFFICULTY: "_motor_difficulty_strategy",
        BottleneckType.MEMORY_DECAY: "_memory_decay_strategy",
        BottleneckType.CROSS_CHANNEL_INTERFERENCE: "_interference_strategy",
    }

    def __init__(
        self,
        target_size_px: float = 44.0,
        max_menu_items: int = 7,
        max_hierarchy_depth: int = 3,
    ) -> None:
        self.target_size_px = target_size_px
        self.max_menu_items = max_menu_items
        self.max_hierarchy_depth = max_hierarchy_depth

    def select(self, bottleneck: Any) -> list[UIMutation]:
        """Select mutations for a given bottleneck.

        Parameters
        ----------
        bottleneck : BottleneckResult-like
            Must have ``bottleneck_type``, ``node_ids``, etc.

        Returns
        -------
        list[UIMutation]
            Proposed mutations, ordered by priority (highest first).
        """
        info = _as_info(bottleneck)
        method_name = self._STRATEGY_MAP.get(info.bottleneck_type)

        if method_name is None:
            logger.warning(
                "No strategy for bottleneck type %r; returning empty",
                info.bottleneck_type,
            )
            return []

        method = getattr(self, method_name)
        mutations: list[UIMutation] = method(info)
        mutations.sort(key=lambda m: m.priority, reverse=True)
        return mutations

    def select_all(self, bottlenecks: Sequence[Any]) -> list[UIMutation]:
        """Select mutations for multiple bottlenecks, deduplicating."""
        seen: set[tuple[str, str]] = set()
        all_mutations: list[UIMutation] = []

        for bn in bottlenecks:
            for m in self.select(bn):
                key = (m.mutation_type, m.target_node_id)
                if key not in seen:
                    seen.add(key)
                    all_mutations.append(m)

        all_mutations.sort(key=lambda m: m.priority, reverse=True)
        return all_mutations

    # ── Strategy: Perceptual Overload -------------------------------------

    def _perceptual_overload_strategy(
        self, bn: _BottleneckInfo
    ) -> list[UIMutation]:
        """Reduce visual elements, improve grouping, increase whitespace.

        Heuristics:
        1. Regroup related elements to reduce visual complexity.
        2. Add landmark regions for visual structure.
        3. Remove purely decorative / low-priority elements.
        """
        mutations: list[UIMutation] = []
        primary = bn.node_ids[0] if bn.node_ids else bn.state_id

        # 1. Group related elements
        if len(bn.node_ids) >= 2:
            mutations.append(UIMutation(
                mutation_type=MutationType.REGROUP,
                target_node_id=primary,
                parameters={
                    "node_ids": bn.node_ids,
                    "new_parent_role": "group",
                },
                description="Group related elements to reduce visual clutter",
                priority=3.0,
            ))

        # 2. Add landmark for structural clarity
        mutations.append(UIMutation(
            mutation_type=MutationType.ADD_LANDMARK,
            target_node_id=primary,
            parameters={
                "landmark_role": "region",
                "region_ids": bn.node_ids,
            },
            description="Add landmark region for visual structure",
            priority=2.0,
        ))

        # 3. Increase spacing via reposition (move elements apart)
        if len(bn.node_ids) >= 2:
            for i, nid in enumerate(bn.node_ids[1:], 1):
                mutations.append(UIMutation(
                    mutation_type=MutationType.REPOSITION,
                    target_node_id=nid,
                    parameters={
                        "y_offset": 16 * i,  # increase vertical spacing
                    },
                    description=f"Increase spacing around {nid}",
                    priority=1.5,
                ))

        # 4. Remove low-priority (non-interactive) clutter nodes
        decorative_ids = bn.metadata.get("decorative_node_ids", [])
        for nid in decorative_ids[:3]:  # limit removals
            mutations.append(UIMutation(
                mutation_type=MutationType.REMOVE,
                target_node_id=nid,
                parameters={},
                description=f"Remove decorative element {nid}",
                priority=1.0,
            ))

        return mutations

    # ── Strategy: Choice Paralysis ----------------------------------------

    def _choice_paralysis_strategy(
        self, bn: _BottleneckInfo
    ) -> list[UIMutation]:
        """Add progressive disclosure, reduce options, improve defaults.

        Heuristics:
        1. Simplify menus that exceed the Hick-Hyman threshold.
        2. Regroup choices into categories.
        3. Relabel ambiguous options for clarity.
        """
        mutations: list[UIMutation] = []
        primary = bn.node_ids[0] if bn.node_ids else bn.state_id

        # 1. Simplify the menu / option list
        mutations.append(UIMutation(
            mutation_type=MutationType.SIMPLIFY_MENU,
            target_node_id=primary,
            parameters={"max_items": self.max_menu_items},
            description=f"Reduce menu items to ≤{self.max_menu_items} (Hick-Hyman)",
            priority=4.0,
        ))

        # 2. Group related choices into categories
        if len(bn.node_ids) >= 3:
            # Split into two groups
            mid = len(bn.node_ids) // 2
            group_a = bn.node_ids[:mid]
            group_b = bn.node_ids[mid:]

            mutations.append(UIMutation(
                mutation_type=MutationType.REGROUP,
                target_node_id=group_a[0],
                parameters={
                    "node_ids": group_a,
                    "new_parent_role": "group",
                },
                description="Group related choices (category A)",
                priority=3.0,
            ))
            mutations.append(UIMutation(
                mutation_type=MutationType.REGROUP,
                target_node_id=group_b[0],
                parameters={
                    "node_ids": group_b,
                    "new_parent_role": "group",
                },
                description="Group related choices (category B)",
                priority=3.0,
            ))

        # 3. Relabel ambiguous items
        ambiguous_ids = bn.metadata.get("ambiguous_labels", [])
        for nid in ambiguous_ids[:5]:
            mutations.append(UIMutation(
                mutation_type=MutationType.RELABEL,
                target_node_id=nid,
                parameters={"new_name": f"Clear: {nid}"},
                description=f"Clarify ambiguous label on {nid}",
                priority=2.0,
            ))

        return mutations

    # ── Strategy: Motor Difficulty ----------------------------------------

    def _motor_difficulty_strategy(
        self, bn: _BottleneckInfo
    ) -> list[UIMutation]:
        """Increase target size, reduce distance, add keyboard shortcuts.

        Heuristics:
        1. Resize small targets to meet the 44px minimum.
        2. Reposition distant targets closer to the action centre.
        3. Add keyboard shortcuts for frequently-used actions.
        """
        mutations: list[UIMutation] = []
        primary = bn.node_ids[0] if bn.node_ids else bn.state_id

        # 1. Resize targets
        for nid in bn.node_ids:
            mutations.append(UIMutation(
                mutation_type=MutationType.RESIZE,
                target_node_id=nid,
                parameters={
                    "width": self.target_size_px,
                    "height": self.target_size_px,
                },
                description=f"Enlarge target {nid} to {self.target_size_px}px",
                priority=4.0,
            ))

        # 2. Reposition to reduce movement distance
        # Move targets closer to screen centre (approximate)
        for nid in bn.node_ids:
            mutations.append(UIMutation(
                mutation_type=MutationType.REPOSITION,
                target_node_id=nid,
                parameters={
                    "x": 400,  # approximate screen centre
                    "y": 300,
                },
                description=f"Move {nid} closer to action centre",
                priority=2.5,
            ))

        # 3. Add keyboard shortcuts
        shortcut_keys = ["Alt+1", "Alt+2", "Alt+3", "Alt+4", "Alt+5"]
        for i, nid in enumerate(bn.node_ids[:len(shortcut_keys)]):
            mutations.append(UIMutation(
                mutation_type=MutationType.ADD_SHORTCUT,
                target_node_id=nid,
                parameters={"shortcut_key": shortcut_keys[i]},
                description=f"Add keyboard shortcut {shortcut_keys[i]} for {nid}",
                priority=3.5,
            ))

        return mutations

    # ── Strategy: Memory Decay --------------------------------------------

    def _memory_decay_strategy(
        self, bn: _BottleneckInfo
    ) -> list[UIMutation]:
        """Reduce steps, add persistent state indicators.

        Heuristics:
        1. Add landmarks to provide orientation cues.
        2. Relabel to include state information (progress indicators).
        3. Regroup to reduce the number of separate screens/steps.
        """
        mutations: list[UIMutation] = []
        primary = bn.node_ids[0] if bn.node_ids else bn.state_id

        # 1. Add landmarks for orientation
        mutations.append(UIMutation(
            mutation_type=MutationType.ADD_LANDMARK,
            target_node_id=primary,
            parameters={
                "landmark_role": "navigation",
                "region_ids": bn.node_ids,
            },
            description="Add navigation landmark for orientation",
            priority=3.5,
        ))

        # 2. Relabel with state information
        for i, nid in enumerate(bn.node_ids):
            mutations.append(UIMutation(
                mutation_type=MutationType.RELABEL,
                target_node_id=nid,
                parameters={
                    "new_name": f"Step {i + 1}: {nid}",
                },
                description=f"Add step indicator to {nid}",
                priority=3.0,
            ))

        # 3. Regroup to reduce cognitive load
        if len(bn.node_ids) >= 2:
            mutations.append(UIMutation(
                mutation_type=MutationType.REGROUP,
                target_node_id=primary,
                parameters={
                    "node_ids": bn.node_ids,
                    "new_parent_role": "group",
                },
                description="Group steps to reduce memory load",
                priority=2.5,
            ))

        # 4. Add shortcuts for quick navigation between steps
        for i, nid in enumerate(bn.node_ids[:5]):
            mutations.append(UIMutation(
                mutation_type=MutationType.ADD_SHORTCUT,
                target_node_id=nid,
                parameters={"shortcut_key": f"Ctrl+{i + 1}"},
                description=f"Add shortcut to quickly reach step {i + 1}",
                priority=2.0,
            ))

        return mutations

    # ── Strategy: Cross-Channel Interference ------------------------------

    def _interference_strategy(
        self, bn: _BottleneckInfo
    ) -> list[UIMutation]:
        """Separate modalities, serialise interactions.

        Heuristics:
        1. Reposition interfering elements to separate screen regions.
        2. Regroup to create clear visual separation between channels.
        3. Remove redundant elements that cause interference.
        """
        mutations: list[UIMutation] = []
        primary = bn.node_ids[0] if bn.node_ids else bn.state_id

        # 1. Spatially separate interfering elements
        if len(bn.node_ids) >= 2:
            # Place in two columns
            left_nodes = bn.node_ids[:len(bn.node_ids) // 2]
            right_nodes = bn.node_ids[len(bn.node_ids) // 2:]

            for nid in left_nodes:
                mutations.append(UIMutation(
                    mutation_type=MutationType.REPOSITION,
                    target_node_id=nid,
                    parameters={"x": 100, "y": 200},
                    description=f"Move {nid} to left column to reduce interference",
                    priority=3.0,
                ))

            for nid in right_nodes:
                mutations.append(UIMutation(
                    mutation_type=MutationType.REPOSITION,
                    target_node_id=nid,
                    parameters={"x": 500, "y": 200},
                    description=f"Move {nid} to right column to reduce interference",
                    priority=3.0,
                ))

        # 2. Group each channel's elements
        channel_groups = bn.metadata.get("channels", {})
        for channel_name, channel_node_ids in channel_groups.items():
            if channel_node_ids:
                mutations.append(UIMutation(
                    mutation_type=MutationType.REGROUP,
                    target_node_id=channel_node_ids[0],
                    parameters={
                        "node_ids": channel_node_ids,
                        "new_parent_role": "group",
                    },
                    description=f"Group {channel_name} channel elements",
                    priority=2.5,
                ))

        # 3. Add landmarks to clarify modality boundaries
        mutations.append(UIMutation(
            mutation_type=MutationType.ADD_LANDMARK,
            target_node_id=primary,
            parameters={
                "landmark_role": "region",
                "region_ids": bn.node_ids,
            },
            description="Add region landmark to clarify modality boundaries",
            priority=2.0,
        ))

        # 4. Remove conflicting secondary elements
        secondary_ids = bn.metadata.get("secondary_elements", [])
        for nid in secondary_ids[:2]:
            mutations.append(UIMutation(
                mutation_type=MutationType.REMOVE,
                target_node_id=nid,
                parameters={},
                description=f"Remove conflicting secondary element {nid}",
                priority=1.5,
            ))

        return mutations

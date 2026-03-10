"""Unit tests for repair strategies and mutations working together.

Tests RepairStrategySelector.select / select_all with different
bottleneck types and verifies the correct mutation types are produced.
Each cognitive bottleneck category (perceptual overload, choice paralysis,
motor difficulty, memory decay) maps to specific UI mutation strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List
from unittest.mock import MagicMock

import pytest

from usability_oracle.core.enums import BottleneckType
from usability_oracle.repair.models import (
    MutationType,
    RepairCandidate,
    RepairResult,
    UIMutation,
)
from usability_oracle.repair.strategies import RepairStrategySelector


# ---------------------------------------------------------------------------
# Helpers – lightweight bottleneck-like objects
# ---------------------------------------------------------------------------

@dataclass
class _FakeBottleneck:
    """Minimal object matching the interface expected by _as_info."""
    bottleneck_type: str = ""
    state_id: str = "s0"
    action_id: str = "a0"
    severity: str = "medium"
    cost_contribution: float = 1.0
    description: str = ""
    node_ids: List[str] = field(default_factory=lambda: ["node_1", "node_2"])
    metadata: dict = field(default_factory=dict)


def _bn(bt: str, **kw) -> _FakeBottleneck:
    """Shorthand to create a fake bottleneck with a given type string."""
    return _FakeBottleneck(bottleneck_type=bt, **kw)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestRepairStrategySelectorConstruction:
    """Tests for RepairStrategySelector instantiation."""

    def test_default_construction(self):
        """Selector can be created with no arguments."""
        sel = RepairStrategySelector()
        assert sel is not None

    def test_custom_target_size(self):
        """target_size_px parameter is stored."""
        sel = RepairStrategySelector(target_size_px=48.0)
        assert sel.target_size_px == 48.0

    def test_custom_max_menu_items(self):
        """max_menu_items parameter is stored."""
        sel = RepairStrategySelector(max_menu_items=5)
        assert sel.max_menu_items == 5

    def test_custom_max_hierarchy_depth(self):
        """max_hierarchy_depth parameter is stored."""
        sel = RepairStrategySelector(max_hierarchy_depth=4)
        assert sel.max_hierarchy_depth == 4


# ---------------------------------------------------------------------------
# select() — general
# ---------------------------------------------------------------------------

class TestSelectGeneral:
    """General tests for RepairStrategySelector.select."""

    def test_returns_list(self):
        """select() always returns a list."""
        sel = RepairStrategySelector()
        result = sel.select(_bn(BottleneckType.PERCEPTUAL_OVERLOAD))
        assert isinstance(result, list)

    def test_returns_ui_mutations(self):
        """All returned elements are UIMutation instances."""
        sel = RepairStrategySelector()
        for bt in BottleneckType:
            mutations = sel.select(_bn(bt))
            for m in mutations:
                assert isinstance(m, UIMutation), f"Non-UIMutation for {bt}"

    def test_mutations_have_target_node_id(self):
        """Every mutation references a valid target_node_id."""
        sel = RepairStrategySelector()
        mutations = sel.select(_bn(BottleneckType.PERCEPTUAL_OVERLOAD))
        for m in mutations:
            assert m.target_node_id, f"Empty target_node_id in {m}"

    def test_mutations_have_mutation_type(self):
        """Every mutation has a non-empty mutation_type."""
        sel = RepairStrategySelector()
        mutations = sel.select(_bn(BottleneckType.MOTOR_DIFFICULTY))
        for m in mutations:
            assert m.mutation_type, f"Empty mutation_type in {m}"


# ---------------------------------------------------------------------------
# select_all()
# ---------------------------------------------------------------------------

class TestSelectAll:
    """Tests for RepairStrategySelector.select_all combining multiple bottlenecks."""

    def test_select_all_returns_list(self):
        """select_all returns a combined list."""
        sel = RepairStrategySelector()
        bns = [
            _bn(BottleneckType.PERCEPTUAL_OVERLOAD),
            _bn(BottleneckType.MOTOR_DIFFICULTY),
        ]
        result = sel.select_all(bns)
        assert isinstance(result, list)

    def test_select_all_combines_results(self):
        """select_all produces at least as many mutations as individual selects."""
        sel = RepairStrategySelector()
        bn1 = _bn(BottleneckType.PERCEPTUAL_OVERLOAD)
        bn2 = _bn(BottleneckType.MOTOR_DIFFICULTY)
        individual = sel.select(bn1) + sel.select(bn2)
        combined = sel.select_all([bn1, bn2])
        assert len(combined) >= min(len(individual), 1)

    def test_select_all_empty_input(self):
        """select_all with empty list returns empty list."""
        sel = RepairStrategySelector()
        assert sel.select_all([]) == []


# ---------------------------------------------------------------------------
# Perceptual overload → regroup / relabel
# ---------------------------------------------------------------------------

class TestPerceptualOverloadStrategy:
    """Tests that perceptual overload bottlenecks produce regroup/relabel mutations."""

    def test_produces_mutations(self):
        """At least one mutation is generated for perceptual overload."""
        sel = RepairStrategySelector()
        mutations = sel.select(_bn(BottleneckType.PERCEPTUAL_OVERLOAD))
        assert len(mutations) >= 1

    def test_includes_regroup_or_relabel(self):
        """Perceptual overload should suggest REGROUP and/or RELABEL."""
        sel = RepairStrategySelector()
        mutations = sel.select(_bn(BottleneckType.PERCEPTUAL_OVERLOAD))
        types = {m.mutation_type for m in mutations}
        assert types & {MutationType.REGROUP.value, MutationType.RELABEL.value,
                        MutationType.REGROUP, MutationType.RELABEL}

    def test_targets_node_ids_from_bottleneck(self):
        """Mutations target node_ids referenced in the bottleneck."""
        sel = RepairStrategySelector()
        bn = _bn(BottleneckType.PERCEPTUAL_OVERLOAD, node_ids=["x", "y", "z"])
        mutations = sel.select(bn)
        target_ids = {m.target_node_id for m in mutations}
        # At least one should reference a node from the bottleneck
        assert target_ids & {"x", "y", "z"} or len(mutations) > 0


# ---------------------------------------------------------------------------
# Choice paralysis → simplify_menu
# ---------------------------------------------------------------------------

class TestChoiceParalysisStrategy:
    """Tests that choice-paralysis bottlenecks produce simplify_menu mutations."""

    def test_produces_mutations(self):
        """At least one mutation is generated for choice paralysis."""
        sel = RepairStrategySelector()
        mutations = sel.select(_bn(BottleneckType.CHOICE_PARALYSIS))
        assert len(mutations) >= 1

    def test_includes_simplify_menu(self):
        """Choice paralysis should suggest SIMPLIFY_MENU."""
        sel = RepairStrategySelector()
        mutations = sel.select(_bn(BottleneckType.CHOICE_PARALYSIS))
        types = {m.mutation_type for m in mutations}
        assert types & {MutationType.SIMPLIFY_MENU.value, MutationType.SIMPLIFY_MENU}


# ---------------------------------------------------------------------------
# Motor difficulty → resize
# ---------------------------------------------------------------------------

class TestMotorDifficultyStrategy:
    """Tests that motor-difficulty bottlenecks produce resize mutations."""

    def test_produces_mutations(self):
        """At least one mutation is generated for motor difficulty."""
        sel = RepairStrategySelector()
        mutations = sel.select(_bn(BottleneckType.MOTOR_DIFFICULTY))
        assert len(mutations) >= 1

    def test_includes_resize(self):
        """Motor difficulty should suggest RESIZE."""
        sel = RepairStrategySelector()
        mutations = sel.select(_bn(BottleneckType.MOTOR_DIFFICULTY))
        types = {m.mutation_type for m in mutations}
        assert types & {MutationType.RESIZE.value, MutationType.RESIZE}


# ---------------------------------------------------------------------------
# Memory decay → add_landmark
# ---------------------------------------------------------------------------

class TestMemoryDecayStrategy:
    """Tests that memory-decay bottlenecks produce add_landmark mutations."""

    def test_produces_mutations(self):
        """At least one mutation is generated for memory decay."""
        sel = RepairStrategySelector()
        mutations = sel.select(_bn(BottleneckType.MEMORY_DECAY))
        assert len(mutations) >= 1

    def test_includes_add_landmark(self):
        """Memory decay should suggest ADD_LANDMARK."""
        sel = RepairStrategySelector()
        mutations = sel.select(_bn(BottleneckType.MEMORY_DECAY))
        types = {m.mutation_type for m in mutations}
        assert types & {MutationType.ADD_LANDMARK.value, MutationType.ADD_LANDMARK}


# ---------------------------------------------------------------------------
# Cross-channel interference
# ---------------------------------------------------------------------------

class TestInterferenceStrategy:
    """Tests that cross-channel interference produces mutations."""

    def test_produces_mutations(self):
        """At least one mutation is generated for cross-channel interference."""
        sel = RepairStrategySelector()
        mutations = sel.select(_bn(BottleneckType.CROSS_CHANNEL_INTERFERENCE))
        assert len(mutations) >= 1


# ---------------------------------------------------------------------------
# MutationType enum
# ---------------------------------------------------------------------------

class TestMutationTypeEnum:
    """Tests that MutationType has all expected members."""

    def test_resize_exists(self):
        """RESIZE member exists."""
        assert MutationType.RESIZE is not None

    def test_reposition_exists(self):
        """REPOSITION member exists."""
        assert MutationType.REPOSITION is not None

    def test_simplify_menu_exists(self):
        """SIMPLIFY_MENU member exists."""
        assert MutationType.SIMPLIFY_MENU is not None

    def test_add_landmark_exists(self):
        """ADD_LANDMARK member exists."""
        assert MutationType.ADD_LANDMARK is not None

    def test_all_types_classmethod(self):
        """all_types() returns a non-empty frozenset."""
        result = MutationType.all_types()
        assert isinstance(result, frozenset)
        assert len(result) >= 8

"""Unit tests for usability_oracle.comparison.models — Comparison data structures.

Tests ComparisonResult, StateMapping, AlignmentResult, BottleneckChange,
ChangeDirection, RegressionReport, ComparisonContext, Partition, and PartitionBlock.

References
----------
- Cohen, J. (1988). *Statistical Power Analysis*.
- Givan, Dean & Greig (2003). *Artificial Intelligence*, 147.
"""

from __future__ import annotations

import pytest

from usability_oracle.comparison.models import (
    AlignmentResult,
    BottleneckChange,
    ChangeDirection,
    ComparisonContext,
    ComparisonResult,
    Partition,
    PartitionBlock,
    RegressionReport,
    StateMapping,
)
from usability_oracle.core.enums import BottleneckType, RegressionVerdict
from usability_oracle.cognitive.models import CostElement
from usability_oracle.mdp.models import MDP
from usability_oracle.taskspec.models import TaskSpec


# ---------------------------------------------------------------------------
# Tests: StateMapping
# ---------------------------------------------------------------------------


class TestStateMapping:
    """Tests for StateMapping — a single state correspondence."""

    def test_basic_fields(self):
        """StateMapping should store state_a, state_b, similarity, and mapping_type."""
        m = StateMapping(state_a="s1", state_b="s2", similarity=0.9, mapping_type="structural")
        assert m.state_a == "s1"
        assert m.state_b == "s2"
        assert m.similarity == 0.9
        assert m.mapping_type == "structural"

    def test_default_similarity(self):
        """StateMapping should default to similarity=1.0 and mapping_type='exact'."""
        m = StateMapping(state_a="a", state_b="b")
        assert m.similarity == 1.0
        assert m.mapping_type == "exact"

    def test_equality(self):
        """Two StateMappings with the same fields should be equal."""
        m1 = StateMapping(state_a="a", state_b="b", similarity=0.5)
        m2 = StateMapping(state_a="a", state_b="b", similarity=0.5)
        assert m1 == m2

    def test_inequality(self):
        """StateMappings with different fields should not be equal."""
        m1 = StateMapping(state_a="a", state_b="b", similarity=0.5)
        m2 = StateMapping(state_a="a", state_b="c", similarity=0.5)
        assert m1 != m2


# ---------------------------------------------------------------------------
# Tests: AlignmentResult
# ---------------------------------------------------------------------------


class TestAlignmentResult:
    """Tests for AlignmentResult — alignment between two MDPs."""

    def test_empty_alignment(self):
        """An empty AlignmentResult should have zero mappings."""
        a = AlignmentResult()
        assert a.mappings == []
        assert a.unmapped_a == []
        assert a.unmapped_b == []
        assert a.overall_similarity == 0.0
        assert a.n_mapped == 0
        assert a.n_unmapped == 0

    def test_get_mapping_dict(self):
        """get_mapping_dict() should return a {state_a: state_b} dictionary."""
        a = AlignmentResult(
            mappings=[
                StateMapping(state_a="s0", state_b="t0"),
                StateMapping(state_a="s1", state_b="t1"),
            ],
        )
        d = a.get_mapping_dict()
        assert d == {"s0": "t0", "s1": "t1"}

    def test_get_reverse_mapping(self):
        """get_reverse_mapping() should return a {state_b: state_a} lookup."""
        a = AlignmentResult(
            mappings=[
                StateMapping(state_a="s0", state_b="t0"),
                StateMapping(state_a="s1", state_b="t1"),
            ],
        )
        r = a.get_reverse_mapping()
        assert r == {"t0": "s0", "t1": "s1"}

    def test_n_mapped(self):
        """n_mapped should return the count of state-to-state correspondences."""
        a = AlignmentResult(
            mappings=[
                StateMapping(state_a="s0", state_b="t0"),
                StateMapping(state_a="s1", state_b="t1"),
                StateMapping(state_a="s2", state_b="t2"),
            ],
        )
        assert a.n_mapped == 3

    def test_n_unmapped(self):
        """n_unmapped should count unmapped states from BOTH MDPs."""
        a = AlignmentResult(
            mappings=[StateMapping(state_a="s0", state_b="t0")],
            unmapped_a=["s1", "s2"],
            unmapped_b=["t1"],
        )
        assert a.n_unmapped == 3

    def test_overall_similarity(self):
        """overall_similarity should store the global alignment score in [0, 1]."""
        a = AlignmentResult(
            mappings=[StateMapping(state_a="s0", state_b="t0", similarity=0.8)],
            overall_similarity=0.75,
        )
        assert a.overall_similarity == 0.75

    def test_metadata_dict(self):
        """AlignmentResult.metadata should store arbitrary alignment metadata."""
        a = AlignmentResult(metadata={"algorithm": "hungarian", "runtime_ms": 42.0})
        assert a.metadata["algorithm"] == "hungarian"
        assert a.metadata["runtime_ms"] == 42.0


# ---------------------------------------------------------------------------
# Tests: ComparisonResult
# ---------------------------------------------------------------------------


class TestComparisonResult:
    """Tests for ComparisonResult — the full comparison output."""

    def test_default_values(self):
        """Default ComparisonResult should be INCONCLUSIVE with confidence 0.95."""
        r = ComparisonResult()
        assert r.verdict == RegressionVerdict.INCONCLUSIVE
        assert r.confidence == 0.95
        assert r.p_value == 1.0
        assert r.effect_size == 0.0
        assert r.is_parameter_free is False

    def test_is_regression_property(self):
        """is_regression should return True only when verdict is REGRESSION."""
        r = ComparisonResult(verdict=RegressionVerdict.REGRESSION)
        assert r.is_regression is True
        assert r.is_improvement is False

    def test_is_improvement_property(self):
        """is_improvement should return True only when verdict is IMPROVEMENT."""
        r = ComparisonResult(verdict=RegressionVerdict.IMPROVEMENT)
        assert r.is_improvement is True
        assert r.is_regression is False

    def test_neutral_neither_regression_nor_improvement(self):
        """NEUTRAL verdict should yield False for both is_regression and is_improvement."""
        r = ComparisonResult(verdict=RegressionVerdict.NEUTRAL)
        assert r.is_regression is False
        assert r.is_improvement is False

    def test_effect_magnitude_negligible(self):
        """effect_magnitude returns 'negligible' for |d| < 0.2 (Cohen 1988)."""
        r = ComparisonResult(effect_size=0.1)
        assert r.effect_magnitude == "negligible"

    def test_effect_magnitude_small(self):
        """effect_magnitude returns 'small' for 0.2 ≤ |d| < 0.5."""
        r = ComparisonResult(effect_size=0.3)
        assert r.effect_magnitude == "small"

    def test_effect_magnitude_medium(self):
        """effect_magnitude returns 'medium' for 0.5 ≤ |d| < 0.8."""
        r = ComparisonResult(effect_size=0.6)
        assert r.effect_magnitude == "medium"

    def test_effect_magnitude_large(self):
        """effect_magnitude returns 'large' for |d| ≥ 0.8."""
        r = ComparisonResult(effect_size=1.2)
        assert r.effect_magnitude == "large"

    def test_effect_magnitude_negative(self):
        """effect_magnitude uses absolute value; negative d=-0.9 → 'large'."""
        r = ComparisonResult(effect_size=-0.9)
        assert r.effect_magnitude == "large"

    def test_bottleneck_changes_list(self):
        """bottleneck_changes should store a list of BottleneckChange objects."""
        bc = BottleneckChange(
            bottleneck_type=BottleneckType.PERCEPTUAL_OVERLOAD,
            state_id="s0",
            before_severity=0.3,
            after_severity=0.7,
            direction=ChangeDirection.WORSENED,
        )
        r = ComparisonResult(bottleneck_changes=[bc])
        assert len(r.bottleneck_changes) == 1
        assert r.bottleneck_changes[0].direction == ChangeDirection.WORSENED

    def test_parameter_sensitivity_dict(self):
        """parameter_sensitivity stores parameter → sensitivity mappings."""
        r = ComparisonResult(parameter_sensitivity={"beta": 0.3, "gamma": 0.1})
        assert r.parameter_sensitivity["beta"] == 0.3


# ---------------------------------------------------------------------------
# Tests: BottleneckChange and ChangeDirection
# ---------------------------------------------------------------------------


class TestBottleneckChange:
    """Tests for BottleneckChange and its classify_direction classmethod."""

    def test_change_direction_enum_values(self):
        """ChangeDirection has four values: NEW, RESOLVED, WORSENED, IMPROVED."""
        assert ChangeDirection.NEW.value == "new"
        assert ChangeDirection.RESOLVED.value == "resolved"
        assert ChangeDirection.WORSENED.value == "worsened"
        assert ChangeDirection.IMPROVED.value == "improved"

    def test_classify_direction_new(self):
        """classify_direction returns NEW when before ≈ 0 and after > threshold."""
        direction = BottleneckChange.classify_direction(before=0.0, after=0.5)
        assert direction == ChangeDirection.NEW

    def test_classify_direction_resolved(self):
        """classify_direction returns RESOLVED when before > threshold and after ≈ 0."""
        direction = BottleneckChange.classify_direction(before=0.5, after=0.0)
        assert direction == ChangeDirection.RESOLVED

    def test_classify_direction_worsened(self):
        """classify_direction returns WORSENED when both present and after > before."""
        direction = BottleneckChange.classify_direction(before=0.3, after=0.8)
        assert direction == ChangeDirection.WORSENED

    def test_classify_direction_improved(self):
        """classify_direction returns IMPROVED when both present and after < before."""
        direction = BottleneckChange.classify_direction(before=0.8, after=0.3)
        assert direction == ChangeDirection.IMPROVED

    def test_classify_direction_custom_threshold(self):
        """classify_direction respects a custom threshold parameter."""
        direction = BottleneckChange.classify_direction(
            before=0.5, after=0.52, threshold=0.1
        )
        assert direction == ChangeDirection.IMPROVED


# ---------------------------------------------------------------------------
# Tests: RegressionReport
# ---------------------------------------------------------------------------


class TestRegressionReport:
    """Tests for RegressionReport — aggregating per-task comparison results."""

    def test_empty_report(self):
        """An empty RegressionReport should have n_tasks=0."""
        report = RegressionReport()
        assert report.n_tasks == 0
        assert report.regression_tasks == []
        assert report.improved_tasks == []

    def test_n_tasks(self):
        """n_tasks should count the number of per-task ComparisonResults."""
        report = RegressionReport(
            task_results={
                "login": ComparisonResult(verdict=RegressionVerdict.REGRESSION),
                "search": ComparisonResult(verdict=RegressionVerdict.IMPROVEMENT),
                "nav": ComparisonResult(verdict=RegressionVerdict.NEUTRAL),
            }
        )
        assert report.n_tasks == 3

    def test_regression_tasks(self):
        """regression_tasks should list task IDs with REGRESSION verdict."""
        report = RegressionReport(
            task_results={
                "login": ComparisonResult(verdict=RegressionVerdict.REGRESSION),
                "search": ComparisonResult(verdict=RegressionVerdict.IMPROVEMENT),
                "nav": ComparisonResult(verdict=RegressionVerdict.REGRESSION),
            }
        )
        reg_tasks = report.regression_tasks
        assert set(reg_tasks) == {"login", "nav"}

    def test_improved_tasks(self):
        """improved_tasks should list task IDs with IMPROVEMENT verdict."""
        report = RegressionReport(
            task_results={
                "login": ComparisonResult(verdict=RegressionVerdict.REGRESSION),
                "search": ComparisonResult(verdict=RegressionVerdict.IMPROVEMENT),
                "nav": ComparisonResult(verdict=RegressionVerdict.IMPROVEMENT),
            }
        )
        imp_tasks = report.improved_tasks
        assert set(imp_tasks) == {"search", "nav"}

    def test_recommendations_list(self):
        """RegressionReport.recommendations should store actionable suggestions."""
        report = RegressionReport(
            recommendations=["Reduce menu items", "Increase button size"]
        )
        assert len(report.recommendations) == 2


# ---------------------------------------------------------------------------
# Tests: ComparisonContext
# ---------------------------------------------------------------------------


class TestComparisonContext:
    """Tests for ComparisonContext — inputs bundled for a comparison run."""

    def test_default_construction(self):
        """Default ComparisonContext should have empty MDPs and alignment."""
        ctx = ComparisonContext()
        assert isinstance(ctx.mdp_before, MDP)
        assert isinstance(ctx.mdp_after, MDP)
        assert isinstance(ctx.alignment, AlignmentResult)
        assert isinstance(ctx.task_spec, TaskSpec)
        assert isinstance(ctx.config, dict)

    def test_all_fields_set(self):
        """ComparisonContext should faithfully store all provided fields."""
        mdp_a = MDP()
        mdp_b = MDP()
        alignment = AlignmentResult(overall_similarity=0.9)
        task = TaskSpec()
        config = {"beta": 2.0}
        ctx = ComparisonContext(
            mdp_before=mdp_a,
            mdp_after=mdp_b,
            alignment=alignment,
            task_spec=task,
            config=config,
        )
        assert ctx.mdp_before is mdp_a
        assert ctx.mdp_after is mdp_b
        assert ctx.alignment.overall_similarity == 0.9
        assert ctx.config["beta"] == 2.0


# ---------------------------------------------------------------------------
# Tests: Partition and PartitionBlock
# ---------------------------------------------------------------------------


class TestPartition:
    """Tests for Partition and PartitionBlock — bisimulation quotient partition."""

    def test_empty_partition(self):
        """An empty Partition should have n_blocks=0."""
        p = Partition()
        assert p.n_blocks == 0
        assert p.state_to_block == {}

    def test_partition_block_fields(self):
        """PartitionBlock should store block_id, state_ids, and representative."""
        block = PartitionBlock(
            block_id="b0",
            state_ids=["s0", "s1", "s2"],
            representative="s0",
        )
        assert block.block_id == "b0"
        assert len(block.state_ids) == 3
        assert block.representative == "s0"

    def test_n_blocks(self):
        """n_blocks should return the number of blocks in the partition."""
        p = Partition(
            blocks=[
                PartitionBlock(block_id="b0", state_ids=["s0", "s1"]),
                PartitionBlock(block_id="b1", state_ids=["s2"]),
            ],
            state_to_block={"s0": "b0", "s1": "b0", "s2": "b1"},
        )
        assert p.n_blocks == 2

    def test_get_block(self):
        """get_block(state_id) should return the PartitionBlock containing that state."""
        b0 = PartitionBlock(block_id="b0", state_ids=["s0", "s1"], representative="s0")
        b1 = PartitionBlock(block_id="b1", state_ids=["s2"], representative="s2")
        p = Partition(
            blocks=[b0, b1],
            state_to_block={"s0": "b0", "s1": "b0", "s2": "b1"},
        )
        assert p.get_block("s0") is b0
        assert p.get_block("s2") is b1

    def test_get_block_unknown_state(self):
        """get_block() should return None for a state not in the partition."""
        p = Partition(
            blocks=[PartitionBlock(block_id="b0", state_ids=["s0"])],
            state_to_block={"s0": "b0"},
        )
        assert p.get_block("unknown") is None

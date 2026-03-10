"""
Unit tests for usability_oracle.core.enums.

Covers all 13 enum types: AccessibilityRole, CognitiveLaw, BottleneckType,
EditOperationType, RegressionVerdict, PipelineStage, ComparisonMode,
OutputFormat, AlignmentPass, Severity, MotorChannel, PerceptualChannel.
Tests classification properties, ordering, inverse mappings, and string
representations.
"""

from __future__ import annotations

import pytest

from usability_oracle.core.enums import (
    AccessibilityRole,
    AlignmentPass,
    BottleneckType,
    CognitiveLaw,
    ComparisonMode,
    EditOperationType,
    MotorChannel,
    OutputFormat,
    PerceptualChannel,
    PipelineStage,
    RegressionVerdict,
    Severity,
)


# ═══════════════════════════════════════════════════════════════════════════
# AccessibilityRole
# ═══════════════════════════════════════════════════════════════════════════


class TestAccessibilityRole:
    """Tests for AccessibilityRole classification properties."""

    def test_is_interactive(self) -> None:
        """BUTTON and SLIDER are interactive; HEADING is not."""
        assert AccessibilityRole.BUTTON.is_interactive is True
        assert AccessibilityRole.SLIDER.is_interactive is True
        assert AccessibilityRole.HEADING.is_interactive is False

    def test_is_landmark(self) -> None:
        """NAVIGATION and BANNER are landmarks; BUTTON is not."""
        assert AccessibilityRole.NAVIGATION.is_landmark is True
        assert AccessibilityRole.BANNER.is_landmark is True
        assert AccessibilityRole.BUTTON.is_landmark is False

    def test_is_container(self) -> None:
        """MENU and TABLE are containers; BUTTON is not."""
        assert AccessibilityRole.MENU.is_container is True
        assert AccessibilityRole.TABLE.is_container is True
        assert AccessibilityRole.BUTTON.is_container is False

    def test_is_widget(self) -> None:
        """BUTTON is widget (interactive non-container); MENU is not."""
        assert AccessibilityRole.BUTTON.is_widget is True
        assert AccessibilityRole.MENU.is_widget is False

    def test_expected_motor_channel(self) -> None:
        """SLIDER and TEXTFIELD use the 'hand' motor channel."""
        assert AccessibilityRole.SLIDER.expected_motor_channel == "hand"
        assert AccessibilityRole.TEXTFIELD.expected_motor_channel == "hand"


# ═══════════════════════════════════════════════════════════════════════════
# CognitiveLaw
# ═══════════════════════════════════════════════════════════════════════════


class TestCognitiveLaw:
    """Tests for CognitiveLaw classification and processing_stage."""

    def test_is_motor(self) -> None:
        """FITTS is motor; HICK_HYMAN is not."""
        assert CognitiveLaw.FITTS.is_motor is True
        assert CognitiveLaw.HICK_HYMAN.is_motor is False

    def test_is_cognitive(self) -> None:
        """HICK_HYMAN is cognitive; FITTS is not."""
        assert CognitiveLaw.HICK_HYMAN.is_cognitive is True
        assert CognitiveLaw.FITTS.is_cognitive is False

    def test_is_perceptual(self) -> None:
        """VISUAL_SEARCH is perceptual; FITTS is not."""
        assert CognitiveLaw.VISUAL_SEARCH.is_perceptual is True
        assert CognitiveLaw.FITTS.is_perceptual is False

    def test_processing_stage(self) -> None:
        """Each law maps to the correct MHP stage string."""
        assert CognitiveLaw.FITTS.processing_stage == "motor"
        assert CognitiveLaw.HICK_HYMAN.processing_stage == "cognitive"
        assert CognitiveLaw.VISUAL_SEARCH.processing_stage == "perceptual"

    def test_mutual_exclusion(self) -> None:
        """Each law belongs to exactly one processing stage."""
        for law in CognitiveLaw:
            flags = [law.is_motor, law.is_cognitive, law.is_perceptual]
            assert sum(flags) == 1, f"{law} has {sum(flags)} stages"


# ═══════════════════════════════════════════════════════════════════════════
# BottleneckType
# ═══════════════════════════════════════════════════════════════════════════


class TestBottleneckType:
    """Tests for BottleneckType properties."""

    def test_cognitive_law(self) -> None:
        """Each bottleneck maps to the correct cognitive law."""
        assert BottleneckType.MOTOR_DIFFICULTY.cognitive_law == CognitiveLaw.FITTS
        assert BottleneckType.CHOICE_PARALYSIS.cognitive_law == CognitiveLaw.HICK_HYMAN

    def test_severity_weight(self) -> None:
        """All severity weights lie in [0,1]; MEMORY_DECAY is highest at 0.9."""
        for bt in BottleneckType:
            assert 0.0 <= bt.severity_weight <= 1.0
        assert BottleneckType.MEMORY_DECAY.severity_weight == 0.9

    def test_suggested_action(self) -> None:
        """Every bottleneck has a non-empty suggested action string."""
        for bt in BottleneckType:
            assert len(bt.suggested_action) > 0


# ═══════════════════════════════════════════════════════════════════════════
# RegressionVerdict
# ═══════════════════════════════════════════════════════════════════════════


class TestRegressionVerdict:
    """Tests for RegressionVerdict properties."""

    def test_is_actionable(self) -> None:
        """REGRESSION and INCONCLUSIVE are actionable; others are not."""
        assert RegressionVerdict.REGRESSION.is_actionable is True
        assert RegressionVerdict.INCONCLUSIVE.is_actionable is True
        assert RegressionVerdict.IMPROVEMENT.is_actionable is False
        assert RegressionVerdict.NEUTRAL.is_actionable is False

    def test_exit_code(self) -> None:
        """Exit codes: REGRESSION=1, IMPROVEMENT/NEUTRAL=0, INCONCLUSIVE=2."""
        assert RegressionVerdict.REGRESSION.exit_code == 1
        assert RegressionVerdict.IMPROVEMENT.exit_code == 0
        assert RegressionVerdict.INCONCLUSIVE.exit_code == 2

    def test_emoji(self) -> None:
        """Every verdict has a non-empty emoji representation."""
        for v in RegressionVerdict:
            assert len(v.emoji) > 0


# ═══════════════════════════════════════════════════════════════════════════
# PipelineStage
# ═══════════════════════════════════════════════════════════════════════════


class TestPipelineStage:
    """Tests for PipelineStage order and depends_on properties."""

    def test_order(self) -> None:
        """PARSE is first (0); OUTPUT is last; orders are monotonically increasing."""
        assert PipelineStage.PARSE.order == 0
        assert PipelineStage.OUTPUT.order == len(PipelineStage) - 1
        stages = list(PipelineStage)
        for i in range(1, len(stages)):
            assert stages[i].order == stages[i - 1].order + 1

    def test_depends_on(self) -> None:
        """PARSE depends on None; every other stage depends on its predecessor."""
        assert PipelineStage.PARSE.depends_on is None
        assert PipelineStage.ALIGN.depends_on == PipelineStage.PARSE
        stages = list(PipelineStage)
        for i in range(1, len(stages)):
            assert stages[i].depends_on == stages[i - 1]


# ═══════════════════════════════════════════════════════════════════════════
# Severity
# ═══════════════════════════════════════════════════════════════════════════


class TestSeverity:
    """Tests for Severity numeric, comparisons, and sarif_level."""

    def test_numeric(self) -> None:
        """CRITICAL=4, INFO=0."""
        assert Severity.CRITICAL.numeric == 4
        assert Severity.INFO.numeric == 0

    def test_comparisons(self) -> None:
        """Rich comparisons (__ge__, __gt__, __le__, __lt__) work correctly."""
        assert Severity.CRITICAL >= Severity.HIGH
        assert Severity.HIGH > Severity.MEDIUM
        assert Severity.LOW <= Severity.MEDIUM
        assert Severity.INFO < Severity.LOW

    def test_sarif_level(self) -> None:
        """CRITICAL/HIGH -> 'error', MEDIUM -> 'warning', INFO -> 'none'."""
        assert Severity.CRITICAL.sarif_level == "error"
        assert Severity.MEDIUM.sarif_level == "warning"
        assert Severity.INFO.sarif_level == "none"


# ═══════════════════════════════════════════════════════════════════════════
# EditOperationType
# ═══════════════════════════════════════════════════════════════════════════


class TestEditOperationType:
    """Tests for EditOperationType properties."""

    def test_is_structural(self) -> None:
        """ADD/REMOVE are structural; RENAME is not."""
        assert EditOperationType.ADD.is_structural is True
        assert EditOperationType.REMOVE.is_structural is True
        assert EditOperationType.RENAME.is_structural is False

    def test_is_visual(self) -> None:
        """MOVE and RESIZE are visual operations."""
        assert EditOperationType.MOVE.is_visual is True
        assert EditOperationType.RESIZE.is_visual is True

    def test_inverse(self) -> None:
        """ADD inverts to REMOVE, REMOVE to ADD, RENAME to itself."""
        assert EditOperationType.ADD.inverse == EditOperationType.REMOVE
        assert EditOperationType.REMOVE.inverse == EditOperationType.ADD
        assert EditOperationType.RENAME.inverse == EditOperationType.RENAME


# ═══════════════════════════════════════════════════════════════════════════
# OutputFormat
# ═══════════════════════════════════════════════════════════════════════════


class TestOutputFormat:
    """Tests for OutputFormat file_extension and mime_type."""

    def test_file_extension(self) -> None:
        """Each format has the expected file extension."""
        assert OutputFormat.JSON.file_extension == ".json"
        assert OutputFormat.SARIF.file_extension == ".sarif"
        assert OutputFormat.HTML.file_extension == ".html"

    def test_mime_type(self) -> None:
        """Each format has the expected MIME type."""
        assert OutputFormat.JSON.mime_type == "application/json"
        assert OutputFormat.SARIF.mime_type == "application/sarif+json"
        assert OutputFormat.HTML.mime_type == "text/html"
        assert OutputFormat.CONSOLE.mime_type == "text/plain"


# ═══════════════════════════════════════════════════════════════════════════
# MotorChannel
# ═══════════════════════════════════════════════════════════════════════════


class TestMotorChannel:
    """Tests for MotorChannel properties."""

    def test_bandwidth_and_preparation(self) -> None:
        """HAND has 10 bits/s bandwidth and 150ms preparation time."""
        assert MotorChannel.HAND.typical_bandwidth_bits_per_sec == 10.0
        assert MotorChannel.VOICE.typical_bandwidth_bits_per_sec == 40.0
        assert MotorChannel.HAND.preparation_time_ms == 150.0

    def test_can_parallel_with(self) -> None:
        """HAND can operate in parallel with EYE and VOICE."""
        expected = frozenset({MotorChannel.EYE, MotorChannel.VOICE})
        assert MotorChannel.HAND.can_parallel_with == expected


# ═══════════════════════════════════════════════════════════════════════════
# PerceptualChannel
# ═══════════════════════════════════════════════════════════════════════════


class TestPerceptualChannel:
    """Tests for PerceptualChannel properties."""

    def test_encoding_and_capacity(self) -> None:
        """VISUAL encoding 100ms, AUDITORY 70ms; VISUAL capacity 40 bits/s."""
        assert PerceptualChannel.VISUAL.typical_encoding_time_ms == 100.0
        assert PerceptualChannel.AUDITORY.typical_encoding_time_ms == 70.0
        assert PerceptualChannel.VISUAL.channel_capacity_bits_per_sec == 40.0

    def test_paired_motor_channel(self) -> None:
        """VISUAL pairs with HAND; AUDITORY pairs with VOICE."""
        assert PerceptualChannel.VISUAL.paired_motor_channel == MotorChannel.HAND
        assert PerceptualChannel.AUDITORY.paired_motor_channel == MotorChannel.VOICE

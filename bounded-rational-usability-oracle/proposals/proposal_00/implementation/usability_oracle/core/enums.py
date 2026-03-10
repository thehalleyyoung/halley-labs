"""
usability_oracle.core.enums — All enumerations used across the oracle pipeline.

Each enum carries a human-readable *label* (via ``__str__``) and, where relevant,
helper properties that map between classification dimensions.
"""

from __future__ import annotations

from enum import Enum, auto, unique
from typing import FrozenSet, Optional


# ═══════════════════════════════════════════════════════════════════════════
# AccessibilityRole — ARIA / platform accessibility roles
# ═══════════════════════════════════════════════════════════════════════════

@unique
class AccessibilityRole(Enum):
    """Standard accessibility roles roughly aligned with WAI-ARIA 1.2."""

    BUTTON = "button"
    LINK = "link"
    TEXTFIELD = "textfield"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    COMBOBOX = "combobox"
    SLIDER = "slider"
    TAB = "tab"
    MENU = "menu"
    MENUITEM = "menuitem"
    TREE = "tree"
    TREEITEM = "treeitem"
    LIST = "list"
    LISTITEM = "listitem"
    TABLE = "table"
    ROW = "row"
    CELL = "cell"
    HEADING = "heading"
    LANDMARK = "landmark"
    REGION = "region"
    DIALOG = "dialog"
    ALERT = "alert"
    TOOLBAR = "toolbar"
    SEPARATOR = "separator"
    IMAGE = "image"
    GROUP = "group"
    DOCUMENT = "document"
    FORM = "form"
    NAVIGATION = "navigation"
    SEARCH = "search"
    BANNER = "banner"
    MAIN = "main"
    COMPLEMENTARY = "complementary"
    CONTENTINFO = "contentinfo"
    GENERIC = "generic"

    # --- classification helpers -------------------------------------------

    @property
    def is_interactive(self) -> bool:
        """Return True if this role represents a control the user interacts with."""
        return self in _INTERACTIVE_ROLES

    @property
    def is_landmark(self) -> bool:
        """Return True if this role is a landmark/region."""
        return self in _LANDMARK_ROLES

    @property
    def is_container(self) -> bool:
        """Return True if this role typically contains child widgets."""
        return self in _CONTAINER_ROLES

    @property
    def is_widget(self) -> bool:
        """Return True if the role is a concrete widget (non-structural)."""
        return self.is_interactive and not self.is_container

    @property
    def is_structural(self) -> bool:
        """True for roles that describe document structure, not interaction."""
        return not self.is_interactive

    @property
    def expected_motor_channel(self) -> str:
        """Primary motor channel for interaction ('hand', 'eye', 'voice')."""
        if self in (AccessibilityRole.SLIDER, AccessibilityRole.TEXTFIELD):
            return "hand"
        if self in (AccessibilityRole.LINK, AccessibilityRole.BUTTON):
            return "hand"
        return "hand"

    @property
    def default_tab_stop(self) -> bool:
        """Whether this role is a default tab stop in keyboard navigation."""
        return self in _TAB_STOP_ROLES

    def __str__(self) -> str:
        return self.value


_INTERACTIVE_ROLES: FrozenSet[AccessibilityRole] = frozenset({
    AccessibilityRole.BUTTON,
    AccessibilityRole.LINK,
    AccessibilityRole.TEXTFIELD,
    AccessibilityRole.CHECKBOX,
    AccessibilityRole.RADIO,
    AccessibilityRole.COMBOBOX,
    AccessibilityRole.SLIDER,
    AccessibilityRole.TAB,
    AccessibilityRole.MENUITEM,
    AccessibilityRole.TREEITEM,
    AccessibilityRole.LISTITEM,
    AccessibilityRole.CELL,
    AccessibilityRole.ROW,
})

_LANDMARK_ROLES: FrozenSet[AccessibilityRole] = frozenset({
    AccessibilityRole.LANDMARK,
    AccessibilityRole.REGION,
    AccessibilityRole.NAVIGATION,
    AccessibilityRole.SEARCH,
    AccessibilityRole.BANNER,
    AccessibilityRole.MAIN,
    AccessibilityRole.COMPLEMENTARY,
    AccessibilityRole.CONTENTINFO,
    AccessibilityRole.FORM,
})

_CONTAINER_ROLES: FrozenSet[AccessibilityRole] = frozenset({
    AccessibilityRole.MENU,
    AccessibilityRole.TREE,
    AccessibilityRole.LIST,
    AccessibilityRole.TABLE,
    AccessibilityRole.TOOLBAR,
    AccessibilityRole.GROUP,
    AccessibilityRole.DIALOG,
    AccessibilityRole.FORM,
    AccessibilityRole.DOCUMENT,
})

_TAB_STOP_ROLES: FrozenSet[AccessibilityRole] = frozenset({
    AccessibilityRole.BUTTON,
    AccessibilityRole.LINK,
    AccessibilityRole.TEXTFIELD,
    AccessibilityRole.CHECKBOX,
    AccessibilityRole.RADIO,
    AccessibilityRole.COMBOBOX,
    AccessibilityRole.SLIDER,
    AccessibilityRole.TAB,
})


# ═══════════════════════════════════════════════════════════════════════════
# CognitiveLaw — identifiers for modelled cognitive laws
# ═══════════════════════════════════════════════════════════════════════════

@unique
class CognitiveLaw(Enum):
    """Enumeration of the cognitive / motor laws encoded by the cost model.

    Each member corresponds to a well-known HCI quantitative law whose
    parameters appear in :class:`usability_oracle.core.config.CognitiveConfig`.
    """

    FITTS = "fitts"
    """Fitts' Law: movement time = a + b * log2(D/W + 1)."""

    HICK_HYMAN = "hick_hyman"
    """Hick-Hyman Law: choice RT = a + b * log2(n + 1)."""

    VISUAL_SEARCH = "visual_search"
    """Visual search: RT = intercept + slope * set_size."""

    WORKING_MEMORY_DECAY = "working_memory_decay"
    """Working memory: exponential decay P(recall) ~ exp(-lambda*t)."""

    MOTOR_EXECUTION = "motor_execution"
    """Aggregate motor execution cost (keystroke-level model)."""

    PERCEPTUAL_PROCESSING = "perceptual_processing"
    """Perceptual encoding latency (visual, auditory, haptic)."""

    @property
    def is_motor(self) -> bool:
        return self in (CognitiveLaw.FITTS, CognitiveLaw.MOTOR_EXECUTION)

    @property
    def is_cognitive(self) -> bool:
        return self in (CognitiveLaw.HICK_HYMAN, CognitiveLaw.WORKING_MEMORY_DECAY)

    @property
    def is_perceptual(self) -> bool:
        return self in (CognitiveLaw.VISUAL_SEARCH, CognitiveLaw.PERCEPTUAL_PROCESSING)

    @property
    def processing_stage(self) -> str:
        """MHP stage: 'perceptual', 'cognitive', or 'motor'."""
        if self.is_perceptual:
            return "perceptual"
        if self.is_cognitive:
            return "cognitive"
        return "motor"

    @property
    def human_readable(self) -> str:
        """Pretty name for reports."""
        _map = {
            CognitiveLaw.FITTS: "Fitts' Law (motor pointing)",
            CognitiveLaw.HICK_HYMAN: "Hick-Hyman Law (choice reaction)",
            CognitiveLaw.VISUAL_SEARCH: "Visual Search (set-size effect)",
            CognitiveLaw.WORKING_MEMORY_DECAY: "Working Memory Decay",
            CognitiveLaw.MOTOR_EXECUTION: "Motor Execution (KLM)",
            CognitiveLaw.PERCEPTUAL_PROCESSING: "Perceptual Processing",
        }
        return _map.get(self, self.value)

    def __str__(self) -> str:
        return self.value


# ═══════════════════════════════════════════════════════════════════════════
# BottleneckType — cognitive bottleneck classification
# ═══════════════════════════════════════════════════════════════════════════

@unique
class BottleneckType(Enum):
    """Categories of cognitive bottleneck the system can identify.

    Each bottleneck maps to the dominant cognitive law that explains it.
    """

    PERCEPTUAL_OVERLOAD = "perceptual_overload"
    """Too many visual items; visual search dominates completion time."""

    CHOICE_PARALYSIS = "choice_paralysis"
    """Too many interactive choices; Hick-Hyman dominates."""

    MOTOR_DIFFICULTY = "motor_difficulty"
    """Target too small or too distant; Fitts' Law dominates."""

    MEMORY_DECAY = "memory_decay"
    """Task requires recall after delay exceeding WM half-life."""

    CROSS_CHANNEL_INTERFERENCE = "cross_channel_interference"
    """Conflicting demands across motor/perceptual channels."""

    @property
    def cognitive_law(self) -> CognitiveLaw:
        """Primary cognitive law associated with this bottleneck."""
        return _BOTTLENECK_TO_LAW[self]

    @property
    def severity_weight(self) -> float:
        """Default severity weighting (0-1 scale) for ranking bottlenecks."""
        return _BOTTLENECK_SEVERITY[self]

    @property
    def suggested_action(self) -> str:
        """One-liner repair suggestion."""
        return _BOTTLENECK_SUGGESTION[self]

    @property
    def affected_channel(self) -> str:
        """Which resource channel is overloaded."""
        return _BOTTLENECK_CHANNEL[self]

    def __str__(self) -> str:
        return self.value


_BOTTLENECK_TO_LAW = {
    BottleneckType.PERCEPTUAL_OVERLOAD: CognitiveLaw.VISUAL_SEARCH,
    BottleneckType.CHOICE_PARALYSIS: CognitiveLaw.HICK_HYMAN,
    BottleneckType.MOTOR_DIFFICULTY: CognitiveLaw.FITTS,
    BottleneckType.MEMORY_DECAY: CognitiveLaw.WORKING_MEMORY_DECAY,
    BottleneckType.CROSS_CHANNEL_INTERFERENCE: CognitiveLaw.MOTOR_EXECUTION,
}

_BOTTLENECK_SEVERITY = {
    BottleneckType.PERCEPTUAL_OVERLOAD: 0.7,
    BottleneckType.CHOICE_PARALYSIS: 0.8,
    BottleneckType.MOTOR_DIFFICULTY: 0.6,
    BottleneckType.MEMORY_DECAY: 0.9,
    BottleneckType.CROSS_CHANNEL_INTERFERENCE: 0.5,
}

_BOTTLENECK_SUGGESTION = {
    BottleneckType.PERCEPTUAL_OVERLOAD: "Reduce visible item count or add visual grouping",
    BottleneckType.CHOICE_PARALYSIS: "Reduce number of choices or introduce progressive disclosure",
    BottleneckType.MOTOR_DIFFICULTY: "Increase target size or reduce movement distance",
    BottleneckType.MEMORY_DECAY: "Provide persistent cues or reduce memory-dependent steps",
    BottleneckType.CROSS_CHANNEL_INTERFERENCE: "Separate competing channel demands temporally",
}

_BOTTLENECK_CHANNEL = {
    BottleneckType.PERCEPTUAL_OVERLOAD: "visual",
    BottleneckType.CHOICE_PARALYSIS: "cognitive",
    BottleneckType.MOTOR_DIFFICULTY: "motor",
    BottleneckType.MEMORY_DECAY: "cognitive",
    BottleneckType.CROSS_CHANNEL_INTERFERENCE: "multi-channel",
}


# ═══════════════════════════════════════════════════════════════════════════
# EditOperationType — accessibility tree edit operations
# ═══════════════════════════════════════════════════════════════════════════

@unique
class EditOperationType(Enum):
    """Atomic edit operations for transforming one accessibility tree into another."""

    RENAME = "rename"
    """Change the accessible name/label of a node."""

    RETYPE = "retype"
    """Change the accessibility role of a node."""

    MOVE = "move"
    """Move a node to a new position (same parent)."""

    RESIZE = "resize"
    """Change the bounding box dimensions of a node."""

    ADD = "add"
    """Insert a new node into the tree."""

    REMOVE = "remove"
    """Delete a node from the tree."""

    REORDER = "reorder"
    """Change sibling order within the same parent."""

    REPARENT = "reparent"
    """Move a node to a different parent."""

    MODIFY_PROPERTY = "modify_property"
    """Change a non-structural property (e.g. description, value)."""

    MODIFY_STATE = "modify_state"
    """Change an accessibility state (e.g. checked, expanded, disabled)."""

    @property
    def is_structural(self) -> bool:
        """True for operations that change the tree topology."""
        return self in (
            EditOperationType.ADD,
            EditOperationType.REMOVE,
            EditOperationType.REPARENT,
            EditOperationType.REORDER,
        )

    @property
    def is_visual(self) -> bool:
        """True for operations that change visual appearance."""
        return self in (
            EditOperationType.MOVE,
            EditOperationType.RESIZE,
            EditOperationType.RENAME,
        )

    @property
    def is_semantic(self) -> bool:
        """True for operations that potentially change task semantics."""
        return self in (
            EditOperationType.RETYPE,
            EditOperationType.ADD,
            EditOperationType.REMOVE,
            EditOperationType.MODIFY_STATE,
        )

    @property
    def default_weight(self) -> float:
        """Default edit-distance weight for alignment scoring."""
        return _EDIT_WEIGHTS.get(self, 1.0)

    @property
    def inverse(self) -> Optional[EditOperationType]:
        """The inverse operation, if one exists."""
        return _EDIT_INVERSES.get(self)

    def __str__(self) -> str:
        return self.value


_EDIT_WEIGHTS = {
    EditOperationType.RENAME: 0.3,
    EditOperationType.RETYPE: 0.8,
    EditOperationType.MOVE: 0.5,
    EditOperationType.RESIZE: 0.4,
    EditOperationType.ADD: 1.0,
    EditOperationType.REMOVE: 1.0,
    EditOperationType.REORDER: 0.2,
    EditOperationType.REPARENT: 0.9,
    EditOperationType.MODIFY_PROPERTY: 0.3,
    EditOperationType.MODIFY_STATE: 0.6,
}

_EDIT_INVERSES = {
    EditOperationType.ADD: EditOperationType.REMOVE,
    EditOperationType.REMOVE: EditOperationType.ADD,
    EditOperationType.RENAME: EditOperationType.RENAME,
    EditOperationType.RETYPE: EditOperationType.RETYPE,
    EditOperationType.MOVE: EditOperationType.MOVE,
    EditOperationType.RESIZE: EditOperationType.RESIZE,
    EditOperationType.REORDER: EditOperationType.REORDER,
    EditOperationType.REPARENT: EditOperationType.REPARENT,
    EditOperationType.MODIFY_PROPERTY: EditOperationType.MODIFY_PROPERTY,
    EditOperationType.MODIFY_STATE: EditOperationType.MODIFY_STATE,
}


# ═══════════════════════════════════════════════════════════════════════════
# RegressionVerdict — outcome of a usability comparison
# ═══════════════════════════════════════════════════════════════════════════

@unique
class RegressionVerdict(Enum):
    """Verdict produced by the statistical comparator."""

    REGRESSION = "regression"
    """The new UI is significantly worse for the tested task(s)."""

    IMPROVEMENT = "improvement"
    """The new UI is significantly better."""

    NEUTRAL = "neutral"
    """No statistically significant difference detected."""

    INCONCLUSIVE = "inconclusive"
    """Insufficient data or conflicting signals."""

    @property
    def is_actionable(self) -> bool:
        """True when the verdict should block or flag a deployment."""
        return self in (RegressionVerdict.REGRESSION, RegressionVerdict.INCONCLUSIVE)

    @property
    def exit_code(self) -> int:
        """Suggested process exit code for CI integration."""
        _codes = {
            RegressionVerdict.REGRESSION: 1,
            RegressionVerdict.IMPROVEMENT: 0,
            RegressionVerdict.NEUTRAL: 0,
            RegressionVerdict.INCONCLUSIVE: 2,
        }
        return _codes[self]

    @property
    def emoji(self) -> str:
        """Emoji for terminal / Markdown output."""
        return {
            RegressionVerdict.REGRESSION: "\u274c",
            RegressionVerdict.IMPROVEMENT: "\u2705",
            RegressionVerdict.NEUTRAL: "\u2796",
            RegressionVerdict.INCONCLUSIVE: "\u2753",
        }[self]

    def __str__(self) -> str:
        return self.value


# ═══════════════════════════════════════════════════════════════════════════
# PipelineStage — stages of the end-to-end oracle pipeline
# ═══════════════════════════════════════════════════════════════════════════

@unique
class PipelineStage(Enum):
    """Ordered stages of the oracle analysis pipeline."""

    PARSE = "parse"
    """Parse raw UI into an accessibility tree."""

    ALIGN = "align"
    """Align nodes across old and new accessibility trees."""

    COST = "cost"
    """Compute cognitive costs for every transition."""

    MDP_BUILD = "mdp_build"
    """Construct the usability MDP from the tree and costs."""

    BISIMULATE = "bisimulate"
    """Reduce the MDP via bisimulation quotient."""

    POLICY = "policy"
    """Compute the bounded-rational policy pi*(beta)."""

    COMPARE = "compare"
    """Statistically compare old vs. new policies."""

    BOTTLENECK = "bottleneck"
    """Classify cognitive bottlenecks."""

    REPAIR = "repair"
    """Synthesise candidate UI repairs."""

    OUTPUT = "output"
    """Format and emit results."""

    @property
    def order(self) -> int:
        """Zero-based execution order."""
        return list(PipelineStage).index(self)

    @property
    def depends_on(self) -> Optional[PipelineStage]:
        """Immediate predecessor stage (None for PARSE)."""
        members = list(PipelineStage)
        idx = members.index(self)
        return members[idx - 1] if idx > 0 else None

    @property
    def config_section(self) -> str:
        """Name of the corresponding config section."""
        _map = {
            PipelineStage.PARSE: "parser",
            PipelineStage.ALIGN: "alignment",
            PipelineStage.COST: "cognitive",
            PipelineStage.MDP_BUILD: "mdp",
            PipelineStage.BISIMULATE: "bisimulation",
            PipelineStage.POLICY: "policy",
            PipelineStage.COMPARE: "comparison",
            PipelineStage.BOTTLENECK: "cognitive",
            PipelineStage.REPAIR: "repair",
            PipelineStage.OUTPUT: "output",
        }
        return _map[self]

    def __str__(self) -> str:
        return self.value


# ═══════════════════════════════════════════════════════════════════════════
# ComparisonMode — statistical comparison strategy
# ═══════════════════════════════════════════════════════════════════════════

@unique
class ComparisonMode(Enum):
    """Strategy for comparing usability between two UI versions."""

    PAIRED = "paired"
    """Paired comparison (same tasks, same user model)."""

    INDEPENDENT = "independent"
    """Independent samples (different user populations)."""

    PARAMETER_FREE = "parameter_free"
    """Non-parametric / distribution-free comparison."""

    @property
    def requires_pairing(self) -> bool:
        return self == ComparisonMode.PAIRED

    @property
    def test_name(self) -> str:
        """Name of the default statistical test for this mode."""
        return {
            ComparisonMode.PAIRED: "paired t-test",
            ComparisonMode.INDEPENDENT: "Welch's t-test",
            ComparisonMode.PARAMETER_FREE: "Mann-Whitney U",
        }[self]

    def __str__(self) -> str:
        return self.value


# ═══════════════════════════════════════════════════════════════════════════
# OutputFormat — report output formats
# ═══════════════════════════════════════════════════════════════════════════

@unique
class OutputFormat(Enum):
    """Supported report output formats."""

    JSON = "json"
    SARIF = "sarif"
    HTML = "html"
    CONSOLE = "console"

    @property
    def file_extension(self) -> str:
        _ext = {
            OutputFormat.JSON: ".json",
            OutputFormat.SARIF: ".sarif",
            OutputFormat.HTML: ".html",
            OutputFormat.CONSOLE: ".txt",
        }
        return _ext[self]

    @property
    def mime_type(self) -> str:
        _mime = {
            OutputFormat.JSON: "application/json",
            OutputFormat.SARIF: "application/sarif+json",
            OutputFormat.HTML: "text/html",
            OutputFormat.CONSOLE: "text/plain",
        }
        return _mime[self]

    @property
    def is_machine_readable(self) -> bool:
        return self in (OutputFormat.JSON, OutputFormat.SARIF)

    def __str__(self) -> str:
        return self.value


# ═══════════════════════════════════════════════════════════════════════════
# AlignmentPass — phases of the tree alignment algorithm
# ═══════════════════════════════════════════════════════════════════════════

@unique
class AlignmentPass(Enum):
    """Multi-pass alignment strategy."""

    EXACT = "exact"
    """First pass: exact matching by ID / role+label."""

    FUZZY = "fuzzy"
    """Second pass: fuzzy matching (similarity threshold)."""

    RESIDUAL = "residual"
    """Third pass: residual unmatched nodes (add/remove)."""

    @property
    def pass_order(self) -> int:
        return list(AlignmentPass).index(self)

    @property
    def requires_threshold(self) -> bool:
        return self == AlignmentPass.FUZZY

    def __str__(self) -> str:
        return self.value


# ═══════════════════════════════════════════════════════════════════════════
# Severity — issue severity levels
# ═══════════════════════════════════════════════════════════════════════════

@unique
class Severity(Enum):
    """Severity levels for usability findings and diagnostics."""

    CRITICAL = "critical"
    """Blocks task completion entirely."""

    HIGH = "high"
    """Significantly increases task time or error rate."""

    MEDIUM = "medium"
    """Noticeably degrades the experience."""

    LOW = "low"
    """Minor inconvenience."""

    INFO = "info"
    """Informational note, no user impact."""

    @property
    def numeric(self) -> int:
        """Numeric severity (4=critical ... 0=info)."""
        return {
            Severity.CRITICAL: 4,
            Severity.HIGH: 3,
            Severity.MEDIUM: 2,
            Severity.LOW: 1,
            Severity.INFO: 0,
        }[self]

    @property
    def sarif_level(self) -> str:
        """SARIF-compatible level string."""
        return {
            Severity.CRITICAL: "error",
            Severity.HIGH: "error",
            Severity.MEDIUM: "warning",
            Severity.LOW: "note",
            Severity.INFO: "none",
        }[self]

    @property
    def color_code(self) -> str:
        """ANSI color code for terminal output."""
        return {
            Severity.CRITICAL: "\033[91m",  # bright red
            Severity.HIGH: "\033[31m",      # red
            Severity.MEDIUM: "\033[33m",    # yellow
            Severity.LOW: "\033[36m",       # cyan
            Severity.INFO: "\033[37m",      # white
        }[self]

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.numeric >= other.numeric

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.numeric > other.numeric

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.numeric <= other.numeric

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.numeric < other.numeric

    def __str__(self) -> str:
        return self.value


# ═══════════════════════════════════════════════════════════════════════════
# MotorChannel / PerceptualChannel — multi-resource theory
# ═══════════════════════════════════════════════════════════════════════════

@unique
class MotorChannel(Enum):
    """Motor output channels (Wickens' multiple-resource theory)."""

    HAND = "hand"
    """Manual pointing / clicking / dragging / typing."""

    EYE = "eye"
    """Saccadic eye movement / gaze-based interaction."""

    VOICE = "voice"
    """Speech-based input."""

    @property
    def typical_bandwidth_bits_per_sec(self) -> float:
        """Approximate information throughput (bits/s)."""
        return {
            MotorChannel.HAND: 10.0,
            MotorChannel.EYE: 3.0,
            MotorChannel.VOICE: 40.0,
        }[self]

    @property
    def preparation_time_ms(self) -> float:
        """Typical motor preparation time (ms)."""
        return {
            MotorChannel.HAND: 150.0,
            MotorChannel.EYE: 100.0,
            MotorChannel.VOICE: 200.0,
        }[self]

    @property
    def can_parallel_with(self) -> frozenset[MotorChannel]:
        """Set of channels that can operate in parallel with this one."""
        return {
            MotorChannel.HAND: frozenset({MotorChannel.EYE, MotorChannel.VOICE}),
            MotorChannel.EYE: frozenset({MotorChannel.HAND, MotorChannel.VOICE}),
            MotorChannel.VOICE: frozenset({MotorChannel.HAND, MotorChannel.EYE}),
        }[self]

    def __str__(self) -> str:
        return self.value


@unique
class PerceptualChannel(Enum):
    """Perceptual input channels."""

    VISUAL = "visual"
    """Visual perception (foveal + peripheral)."""

    AUDITORY = "auditory"
    """Auditory perception."""

    HAPTIC = "haptic"
    """Tactile / haptic feedback."""

    @property
    def typical_encoding_time_ms(self) -> float:
        """Approximate perceptual encoding time (ms)."""
        return {
            PerceptualChannel.VISUAL: 100.0,
            PerceptualChannel.AUDITORY: 70.0,
            PerceptualChannel.HAPTIC: 120.0,
        }[self]

    @property
    def channel_capacity_bits_per_sec(self) -> float:
        """Approximate channel capacity (bits/s)."""
        return {
            PerceptualChannel.VISUAL: 40.0,
            PerceptualChannel.AUDITORY: 25.0,
            PerceptualChannel.HAPTIC: 5.0,
        }[self]

    @property
    def paired_motor_channel(self) -> MotorChannel:
        """Most natural paired motor channel."""
        return {
            PerceptualChannel.VISUAL: MotorChannel.HAND,
            PerceptualChannel.AUDITORY: MotorChannel.VOICE,
            PerceptualChannel.HAPTIC: MotorChannel.HAND,
        }[self]

    def __str__(self) -> str:
        return self.value


__all__ = [
    "AccessibilityRole",
    "BottleneckType",
    "CognitiveLaw",
    "EditOperationType",
    "RegressionVerdict",
    "PipelineStage",
    "ComparisonMode",
    "OutputFormat",
    "AlignmentPass",
    "Severity",
    "MotorChannel",
    "PerceptualChannel",
]

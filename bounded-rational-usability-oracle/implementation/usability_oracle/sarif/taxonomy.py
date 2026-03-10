"""
usability_oracle.sarif.taxonomy — SARIF taxonomies for usability.

Defines usability-specific taxonomies in SARIF format:
  - Cognitive bottleneck taxonomy (UO-xxx identifiers)
  - WCAG guideline taxonomy
  - Severity-level mapping
  - Custom property bags for cognitive cost metrics
  - Taxonomy registration / lookup
  - CWE-like identifier system for usability issues

These taxonomies can be attached to SARIF runs via
``run.taxonomies`` and referenced from results via ``result.taxa``.

Reference: https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html §3.19.3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from usability_oracle.core.enums import BottleneckType, CognitiveLaw, Severity
from usability_oracle.sarif.schema import (
    Level,
    MultiformatMessageString,
    ReportingConfiguration,
    ReportingDescriptor,
    ToolComponent,
    ToolComponentReference,
)


# ═══════════════════════════════════════════════════════════════════════════
# Severity mapping
# ═══════════════════════════════════════════════════════════════════════════

_SEVERITY_TO_LEVEL: Dict[Severity, Level] = {
    Severity.CRITICAL: Level.ERROR,
    Severity.HIGH: Level.ERROR,
    Severity.MEDIUM: Level.WARNING,
    Severity.LOW: Level.NOTE,
    Severity.INFO: Level.NONE,
}


def severity_to_sarif_level(severity: Severity) -> Level:
    """Map an internal :class:`Severity` to a SARIF :class:`Level`."""
    return _SEVERITY_TO_LEVEL.get(severity, Level.WARNING)


def sarif_level_to_severity(level: Level) -> Severity:
    """Map a SARIF :class:`Level` back to an internal :class:`Severity`."""
    _map: Dict[Level, Severity] = {
        Level.ERROR: Severity.HIGH,
        Level.WARNING: Severity.MEDIUM,
        Level.NOTE: Severity.LOW,
        Level.NONE: Severity.INFO,
    }
    return _map.get(level, Severity.MEDIUM)


# ═══════════════════════════════════════════════════════════════════════════
# Cognitive bottleneck taxonomy
# ═══════════════════════════════════════════════════════════════════════════

_BOTTLENECK_TAXONOMY_GUID = "d4e5f6a7-b8c9-0d1e-2f3a-4b5c6d7e8f90"
_BOTTLENECK_TAXONOMY_NAME = "UsabilityOracle/CognitiveBottleneck"
_BOTTLENECK_TAXONOMY_VERSION = "1.0.0"

# UO-xxx taxonomy identifiers (CWE-like).
_BOTTLENECK_TAXA: Tuple[Tuple[str, BottleneckType, str], ...] = (
    (
        "UO-001",
        BottleneckType.PERCEPTUAL_OVERLOAD,
        "Perceptual overload: too many visual items cause visual-search "
        "dominated completion time.  Governed by visual search slope × set "
        "size.",
    ),
    (
        "UO-002",
        BottleneckType.CHOICE_PARALYSIS,
        "Choice paralysis: excessive interactive choices increase reaction "
        "time per Hick-Hyman Law.  Remediate via progressive disclosure or "
        "defaults.",
    ),
    (
        "UO-003",
        BottleneckType.MOTOR_DIFFICULTY,
        "Motor difficulty: target too small or too distant, dominated by "
        "Fitts' Law.  Increase target size or reduce movement distance.",
    ),
    (
        "UO-004",
        BottleneckType.MEMORY_DECAY,
        "Memory decay: task requires recall after delay exceeding working-"
        "memory half-life.  Provide persistent cues or reduce memory-"
        "dependent steps.",
    ),
    (
        "UO-005",
        BottleneckType.CROSS_CHANNEL_INTERFERENCE,
        "Cross-channel interference: conflicting demands across motor and "
        "perceptual channels.  Separate competing demands temporally or "
        "modally.",
    ),
)


def _make_bottleneck_taxon(
    taxon_id: str, bt: BottleneckType, description: str
) -> ReportingDescriptor:
    """Create a SARIF taxon for a cognitive bottleneck type."""
    return ReportingDescriptor(
        id=taxon_id,
        name=bt.value.replace("_", "-"),
        short_description=MultiformatMessageString(
            text=bt.suggested_action,
        ),
        full_description=MultiformatMessageString(text=description),
        default_configuration=ReportingConfiguration(
            level=_SEVERITY_TO_LEVEL.get(
                {
                    0.9: Severity.CRITICAL,
                    0.8: Severity.HIGH,
                    0.7: Severity.MEDIUM,
                    0.6: Severity.LOW,
                    0.5: Severity.LOW,
                }.get(bt.severity_weight, Severity.MEDIUM),
                Level.WARNING,
            ),
        ),
        properties={
            "tags": ["usability", "cognitive-bottleneck", bt.affected_channel],
            "cognitiveLaw": bt.cognitive_law.value,
            "severityWeight": bt.severity_weight,
        },
    )


def cognitive_bottleneck_taxonomy() -> ToolComponent:
    """Build the full cognitive-bottleneck taxonomy as a :class:`ToolComponent`.

    This taxonomy can be attached to a SARIF run's ``taxonomies`` array.
    """
    taxa = tuple(
        _make_bottleneck_taxon(tid, bt, desc)
        for tid, bt, desc in _BOTTLENECK_TAXA
    )
    return ToolComponent(
        name=_BOTTLENECK_TAXONOMY_NAME,
        guid=_BOTTLENECK_TAXONOMY_GUID,
        version=_BOTTLENECK_TAXONOMY_VERSION,
        short_description=MultiformatMessageString(
            text=(
                "Cognitive bottleneck taxonomy for the Bounded-Rational "
                "Usability Oracle."
            ),
        ),
        full_description=MultiformatMessageString(
            text=(
                "Classifies usability issues by the dominant cognitive "
                "bottleneck: perceptual overload, choice paralysis, motor "
                "difficulty, memory decay, or cross-channel interference.  "
                "Each category maps to a well-known HCI quantitative law "
                "(Fitts, Hick-Hyman, visual search, working memory decay)."
            ),
        ),
        organization="usability-oracle",
        taxa=taxa,
        is_comprehensive=True,
        properties={
            "domain": "human-computer-interaction",
            "framework": "bounded-rational-usability-oracle",
        },
    )


def bottleneck_taxonomy_reference() -> ToolComponentReference:
    """Return a :class:`ToolComponentReference` to the bottleneck taxonomy."""
    return ToolComponentReference(
        name=_BOTTLENECK_TAXONOMY_NAME,
        guid=_BOTTLENECK_TAXONOMY_GUID,
    )


# ═══════════════════════════════════════════════════════════════════════════
# WCAG taxonomy
# ═══════════════════════════════════════════════════════════════════════════

_WCAG_TAXONOMY_GUID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
_WCAG_TAXONOMY_NAME = "WCAG/2.2"
_WCAG_TAXONOMY_VERSION = "2.2"

# Core WCAG principles as taxa.
_WCAG_PRINCIPLES: Tuple[Tuple[str, str, str], ...] = (
    ("P1", "Perceivable",
     "Information and UI components must be presentable to users in ways "
     "they can perceive."),
    ("P2", "Operable",
     "UI components and navigation must be operable."),
    ("P3", "Understandable",
     "Information and the operation of the UI must be understandable."),
    ("P4", "Robust",
     "Content must be robust enough to be interpreted reliably by a wide "
     "variety of user agents, including assistive technologies."),
)

# A representative subset of WCAG 2.2 success criteria as taxa.
_WCAG_CRITERIA: Tuple[Tuple[str, str, str, str], ...] = (
    ("1.1.1", "Non-text Content", "A",
     "All non-text content has a text alternative."),
    ("1.3.1", "Info and Relationships", "A",
     "Structure and relationships conveyed through presentation can be "
     "programmatically determined."),
    ("1.4.3", "Contrast (Minimum)", "AA",
     "Text and images of text have a contrast ratio of at least 4.5:1."),
    ("1.4.11", "Non-text Contrast", "AA",
     "Visual information required to identify UI components has a contrast "
     "ratio of at least 3:1."),
    ("2.1.1", "Keyboard", "A",
     "All functionality is operable through a keyboard interface."),
    ("2.4.7", "Focus Visible", "AA",
     "Any keyboard operable UI has a visible focus indicator."),
    ("2.5.5", "Target Size (Enhanced)", "AAA",
     "Target size for pointer inputs is at least 44×44 CSS pixels."),
    ("2.5.8", "Target Size (Minimum)", "AA",
     "Target size is at least 24×24 CSS pixels."),
    ("3.2.1", "On Focus", "A",
     "Receiving focus does not initiate a change of context."),
    ("3.3.2", "Labels or Instructions", "A",
     "Labels or instructions are provided when content requires user input."),
    ("4.1.2", "Name, Role, Value", "A",
     "All UI components have accessible name, role, and value."),
)


def wcag_taxonomy() -> ToolComponent:
    """Build the WCAG 2.2 taxonomy as a :class:`ToolComponent`."""
    taxa: List[ReportingDescriptor] = []

    # Principles.
    for pid, pname, pdesc in _WCAG_PRINCIPLES:
        taxa.append(
            ReportingDescriptor(
                id=f"WCAG-{pid}",
                name=pname.lower(),
                short_description=MultiformatMessageString(text=pname),
                full_description=MultiformatMessageString(text=pdesc),
                properties={"type": "principle"},
            )
        )

    # Success criteria.
    for sc_id, sc_name, sc_level, sc_desc in _WCAG_CRITERIA:
        taxa.append(
            ReportingDescriptor(
                id=f"WCAG-{sc_id}",
                name=sc_name.lower().replace(" ", "-").replace(",", ""),
                short_description=MultiformatMessageString(text=sc_name),
                full_description=MultiformatMessageString(text=sc_desc),
                help_uri=f"https://www.w3.org/WAI/WCAG22/Understanding/{sc_name.lower().replace(' ', '-').replace(',', '').replace('(', '').replace(')', '')}",
                properties={
                    "type": "successCriterion",
                    "conformanceLevel": sc_level,
                    "scId": sc_id,
                },
            )
        )

    return ToolComponent(
        name=_WCAG_TAXONOMY_NAME,
        guid=_WCAG_TAXONOMY_GUID,
        version=_WCAG_TAXONOMY_VERSION,
        short_description=MultiformatMessageString(
            text="Web Content Accessibility Guidelines (WCAG) 2.2",
        ),
        organization="W3C",
        taxa=tuple(taxa),
        is_comprehensive=False,
        properties={
            "domain": "accessibility",
            "specUrl": "https://www.w3.org/TR/WCAG22/",
        },
    )


def wcag_taxonomy_reference() -> ToolComponentReference:
    """Return a :class:`ToolComponentReference` to the WCAG taxonomy."""
    return ToolComponentReference(
        name=_WCAG_TAXONOMY_NAME,
        guid=_WCAG_TAXONOMY_GUID,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Taxonomy registry
# ═══════════════════════════════════════════════════════════════════════════

class TaxonomyRegistry:
    """Registry for SARIF taxonomies.

    Supports registration and lookup by name or GUID::

        registry = TaxonomyRegistry()
        registry.register(cognitive_bottleneck_taxonomy())
        registry.register(wcag_taxonomy())

        tax = registry.lookup("UsabilityOracle/CognitiveBottleneck")
    """

    def __init__(self) -> None:
        self._by_name: Dict[str, ToolComponent] = {}
        self._by_guid: Dict[str, ToolComponent] = {}

    def register(self, taxonomy: ToolComponent) -> None:
        """Register a taxonomy :class:`ToolComponent`."""
        if taxonomy.name:
            self._by_name[taxonomy.name] = taxonomy
        if taxonomy.guid:
            self._by_guid[taxonomy.guid] = taxonomy

    def lookup(self, name_or_guid: str) -> Optional[ToolComponent]:
        """Look up a taxonomy by name or GUID."""
        return self._by_name.get(name_or_guid) or self._by_guid.get(
            name_or_guid
        )

    def all_taxonomies(self) -> Tuple[ToolComponent, ...]:
        """Return all registered taxonomies."""
        seen: Dict[str, ToolComponent] = {}
        for t in list(self._by_name.values()) + list(self._by_guid.values()):
            key = t.guid or t.name
            if key not in seen:
                seen[key] = t
        return tuple(seen.values())

    def references(self) -> Tuple[ToolComponentReference, ...]:
        """Return :class:`ToolComponentReference` for all registered taxonomies."""
        return tuple(
            ToolComponentReference(name=t.name, guid=t.guid)
            for t in self.all_taxonomies()
        )


def default_registry() -> TaxonomyRegistry:
    """Create a registry pre-loaded with the standard usability taxonomies."""
    reg = TaxonomyRegistry()
    reg.register(cognitive_bottleneck_taxonomy())
    reg.register(wcag_taxonomy())
    return reg


# ═══════════════════════════════════════════════════════════════════════════
# Cognitive-cost property bag
# ═══════════════════════════════════════════════════════════════════════════

def cognitive_cost_properties(
    *,
    entropy: Optional[float] = None,
    mutual_info: Optional[float] = None,
    channel_capacity: Optional[float] = None,
    utilization: Optional[float] = None,
    completion_time_ms: Optional[float] = None,
    fitts_id: Optional[float] = None,
    hick_bits: Optional[float] = None,
    visual_set_size: Optional[int] = None,
    confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a SARIF properties dict with cognitive cost metrics.

    These can be attached to a result's ``properties`` bag for downstream
    consumption by viewers and dashboards.
    """
    props: Dict[str, Any] = {}
    if entropy is not None:
        props["entropy"] = entropy
    if mutual_info is not None:
        props["mutualInfo"] = mutual_info
    if channel_capacity is not None:
        props["channelCapacity"] = channel_capacity
    if utilization is not None:
        props["utilization"] = utilization
    if completion_time_ms is not None:
        props["completionTimeMs"] = completion_time_ms
    if fitts_id is not None:
        props["fittsIndexOfDifficulty"] = fitts_id
    if hick_bits is not None:
        props["hickBits"] = hick_bits
    if visual_set_size is not None:
        props["visualSetSize"] = visual_set_size
    if confidence is not None:
        props["confidence"] = confidence
    return props


# ═══════════════════════════════════════════════════════════════════════════
# UO identifier system (CWE-like)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class UsabilityIssueId:
    """A CWE-like identifier for usability issues.

    Format: ``UO-NNN`` where NNN is a zero-padded three-digit number.

    Attributes:
        number: Integer identifier (1-999).
        name: Short descriptive name.
        description: Full description.
        bottleneck_type: Associated :class:`BottleneckType` if applicable.
    """

    number: int
    name: str
    description: str = ""
    bottleneck_type: Optional[BottleneckType] = None

    @property
    def id_string(self) -> str:
        return f"UO-{self.number:03d}"

    def to_taxon(self) -> ReportingDescriptor:
        """Convert to a SARIF :class:`ReportingDescriptor` (taxon)."""
        props: Dict[str, Any] = {}
        if self.bottleneck_type is not None:
            props["bottleneckType"] = self.bottleneck_type.value
        return ReportingDescriptor(
            id=self.id_string,
            name=self.name.lower().replace(" ", "-"),
            short_description=MultiformatMessageString(text=self.name),
            full_description=MultiformatMessageString(text=self.description),
            properties=props,
        )


# Pre-defined issue identifiers.
USABILITY_ISSUES: Tuple[UsabilityIssueId, ...] = tuple(
    UsabilityIssueId(
        number=i + 1,
        name=bt.value.replace("_", " ").title(),
        description=desc,
        bottleneck_type=bt,
    )
    for i, (_, bt, desc) in enumerate(_BOTTLENECK_TAXA)
)


def lookup_usability_issue(issue_id: str) -> Optional[UsabilityIssueId]:
    """Look up a :class:`UsabilityIssueId` by its string id (e.g. "UO-001")."""
    for uid in USABILITY_ISSUES:
        if uid.id_string == issue_id:
            return uid
    return None

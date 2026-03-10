"""
usability_oracle.wcag.parser — WCAG 2.2 XML specification parser.

Parses WCAG 2.x XML documents (as published by W3C) and builds structured
:class:`WCAGGuideline`, :class:`SuccessCriterion`, and principle objects.
Also provides a built-in catalogue of WCAG 2.2 criteria for use when
no external XML file is available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from lxml import etree
except ImportError:  # pragma: no cover — lxml is a declared dependency
    etree = None  # type: ignore[assignment]

from usability_oracle.core.errors import ParseError
from usability_oracle.wcag.types import (
    ConformanceLevel,
    SuccessCriterion,
    WCAGGuideline,
    WCAGPrinciple,
)


# ═══════════════════════════════════════════════════════════════════════════
# Well-known WCAG XML namespaces
# ═══════════════════════════════════════════════════════════════════════════

_WCAG_NS = {
    "wcag": "http://www.w3.org/WAI/GL/",
    "html": "http://www.w3.org/1999/xhtml",
    "xml": "http://www.w3.org/XML/1998/namespace",
}

_LEVEL_MAP = {"A": ConformanceLevel.A, "AA": ConformanceLevel.AA, "AAA": ConformanceLevel.AAA}

_PRINCIPLE_MAP = {
    1: WCAGPrinciple.PERCEIVABLE,
    2: WCAGPrinciple.OPERABLE,
    3: WCAGPrinciple.UNDERSTANDABLE,
    4: WCAGPrinciple.ROBUST,
}

# Regex for extracting version from XML processing instructions or content
_VERSION_PATTERN = re.compile(r"WCAG\s+(2\.[012])", re.IGNORECASE)


# ═══════════════════════════════════════════════════════════════════════════
# Technique and Understanding link references
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class TechniqueRef:
    """Reference to a WCAG sufficient/advisory/failure technique."""

    technique_id: str
    title: str
    url: str
    category: str  # "sufficient", "advisory", "failure"


@dataclass(frozen=True, slots=True)
class UnderstandingLink:
    """Link to the Understanding document for a success criterion."""

    sc_id: str
    url: str
    title: str


# ═══════════════════════════════════════════════════════════════════════════
# Parsed WCAG specification
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class WCAGSpecification:
    """Complete parsed WCAG specification."""

    version: str
    principles: Dict[int, WCAGPrinciple]
    guidelines: List[WCAGGuideline]
    criteria: List[SuccessCriterion]
    techniques: Dict[str, List[TechniqueRef]]  # sc_id → techniques
    understanding_links: Dict[str, UnderstandingLink]  # sc_id → link

    @property
    def criterion_count(self) -> int:
        return len(self.criteria)

    def criteria_at_level(self, level: ConformanceLevel) -> List[SuccessCriterion]:
        return [sc for sc in self.criteria if sc.level <= level]

    def criterion_by_id(self, sc_id: str) -> Optional[SuccessCriterion]:
        for sc in self.criteria:
            if sc.sc_id == sc_id:
                return sc
        return None

    def guidelines_for_principle(self, principle: WCAGPrinciple) -> List[WCAGGuideline]:
        return [g for g in self.guidelines if g.principle == principle]


# ═══════════════════════════════════════════════════════════════════════════
# XML Parser
# ═══════════════════════════════════════════════════════════════════════════

class WCAGXMLParser:
    """Parse WCAG 2.x XML specification documents.

    Handles the XML format used by W3C for publishing WCAG guidelines.
    Supports versions 2.0, 2.1, and 2.2 with automatic version detection.

    If *lxml* is not available at runtime, falls back to the built-in
    catalogue (:meth:`load_builtin`).
    """

    def __init__(self) -> None:
        self._spec: Optional[WCAGSpecification] = None

    # -- public interface ---------------------------------------------------

    def parse_xml(self, xml_source: str | bytes) -> WCAGSpecification:
        """Parse a WCAG XML specification document.

        Parameters
        ----------
        xml_source : str | bytes
            Raw XML content (string or bytes).

        Returns
        -------
        WCAGSpecification
            Structured specification.

        Raises
        ------
        ParseError
            If the XML is malformed or does not contain recognisable WCAG structure.
        """
        if etree is None:
            raise ParseError("lxml is required for XML parsing; install with: pip install lxml")

        try:
            if isinstance(xml_source, str):
                xml_source = xml_source.encode("utf-8")
            tree = etree.fromstring(xml_source)
        except etree.XMLSyntaxError as exc:
            raise ParseError(f"Malformed XML: {exc}") from exc

        version = self._detect_version(tree, xml_source)
        principles: Dict[int, WCAGPrinciple] = {}
        guidelines: List[WCAGGuideline] = []
        all_criteria: List[SuccessCriterion] = []
        techniques: Dict[str, List[TechniqueRef]] = {}
        understanding: Dict[str, UnderstandingLink] = {}

        # Try multiple known XML structures
        guideline_elems = self._find_guidelines(tree)

        for gl_elem in guideline_elems:
            gl_id, gl_name, principle_num, criteria = self._parse_guideline(gl_elem, version)

            if principle_num in _PRINCIPLE_MAP:
                principle = _PRINCIPLE_MAP[principle_num]
                principles[principle_num] = principle
            else:
                principle = WCAGPrinciple.PERCEIVABLE  # fallback

            for sc in criteria:
                all_criteria.append(sc)
                # Extract techniques
                tech_refs = self._extract_techniques(gl_elem, sc.sc_id)
                if tech_refs:
                    techniques[sc.sc_id] = tech_refs
                # Understanding link
                understanding[sc.sc_id] = UnderstandingLink(
                    sc_id=sc.sc_id,
                    url=f"https://www.w3.org/WAI/WCAG22/Understanding/{sc.sc_id.replace('.', '')}",
                    title=f"Understanding SC {sc.sc_id}: {sc.name}",
                )

            guidelines.append(WCAGGuideline(
                guideline_id=gl_id,
                name=gl_name,
                principle=principle,
                criteria=tuple(criteria),
            ))

        if not all_criteria:
            raise ParseError("No success criteria found in the XML document.")

        self._spec = WCAGSpecification(
            version=version,
            principles=principles,
            guidelines=guidelines,
            criteria=all_criteria,
            techniques=techniques,
            understanding_links=understanding,
        )
        return self._spec

    def load_builtin(self) -> WCAGSpecification:
        """Load the built-in WCAG 2.2 catalogue without XML parsing.

        Returns
        -------
        WCAGSpecification
            Complete WCAG 2.2 specification from the bundled catalogue.
        """
        criteria = _BUILTIN_CRITERIA_22()
        guidelines_map: Dict[str, List[SuccessCriterion]] = {}
        for sc in criteria:
            guidelines_map.setdefault(sc.guideline_id, []).append(sc)

        guidelines: List[WCAGGuideline] = []
        for gl_id in sorted(guidelines_map.keys()):
            scs = guidelines_map[gl_id]
            principle = scs[0].principle
            # Derive guideline name from the first criterion's guideline_id
            gl_name = _GUIDELINE_NAMES.get(gl_id, f"Guideline {gl_id}")
            guidelines.append(WCAGGuideline(
                guideline_id=gl_id,
                name=gl_name,
                principle=principle,
                criteria=tuple(scs),
            ))

        techniques: Dict[str, List[TechniqueRef]] = {}
        understanding: Dict[str, UnderstandingLink] = {}
        for sc in criteria:
            understanding[sc.sc_id] = UnderstandingLink(
                sc_id=sc.sc_id,
                url=f"https://www.w3.org/WAI/WCAG22/Understanding/{sc.sc_id.replace('.', '')}",
                title=f"Understanding SC {sc.sc_id}: {sc.name}",
            )

        self._spec = WCAGSpecification(
            version="2.2",
            principles={p.number: p for p in WCAGPrinciple},
            guidelines=guidelines,
            criteria=criteria,
            techniques=techniques,
            understanding_links=understanding,
        )
        return self._spec

    # -- protocol conformance -----------------------------------------------

    def load_criteria(self) -> Sequence[SuccessCriterion]:
        """Load all WCAG 2.2 success criteria (WCAGParser protocol)."""
        if self._spec is None:
            self.load_builtin()
        assert self._spec is not None
        return list(self._spec.criteria)

    def load_guidelines(self) -> Sequence[WCAGGuideline]:
        """Load all WCAG 2.2 guidelines (WCAGParser protocol)."""
        if self._spec is None:
            self.load_builtin()
        assert self._spec is not None
        return list(self._spec.guidelines)

    def criteria_for_level(self, level: ConformanceLevel) -> Sequence[SuccessCriterion]:
        """Criteria at or below the given level (WCAGParser protocol)."""
        if self._spec is None:
            self.load_builtin()
        assert self._spec is not None
        return self._spec.criteria_at_level(level)

    def criterion_by_id(self, sc_id: str) -> Optional[SuccessCriterion]:
        """Look up a criterion by dotted id (WCAGParser protocol)."""
        if self._spec is None:
            self.load_builtin()
        assert self._spec is not None
        return self._spec.criterion_by_id(sc_id)

    # -- internal helpers ---------------------------------------------------

    def _detect_version(self, root: Any, raw: bytes) -> str:
        """Detect WCAG version from XML attributes, PIs, or content."""
        # Check root element attributes
        for attr in ("version", "data-version"):
            v = root.get(attr, "")
            m = _VERSION_PATTERN.search(v)
            if m:
                return m.group(1)

        # Check processing instructions
        raw_str = raw.decode("utf-8", errors="replace")
        m = _VERSION_PATTERN.search(raw_str[:2000])
        if m:
            return m.group(1)

        # Default to 2.2
        return "2.2"

    def _find_guidelines(self, root: Any) -> List[Any]:
        """Locate guideline elements in various XML schema layouts."""
        # Try namespaced search
        for ns_prefix, ns_uri in _WCAG_NS.items():
            results = root.findall(f".//{{{ns_uri}}}guideline")
            if results:
                return results

        # Try non-namespaced
        results = root.findall(".//guideline")
        if results:
            return results

        # Try by id pattern (e.g. elements with id like "1.1", "1.2")
        gl_pattern = re.compile(r"^\d+\.\d+$")
        results = [el for el in root.iter() if gl_pattern.match(el.get("id", ""))]
        if results:
            return results

        # Try section-based layout
        results = root.findall(".//section[@class='guideline']")
        if results:
            return results

        return []

    def _parse_guideline(
        self, elem: Any, version: str
    ) -> Tuple[str, str, int, List[SuccessCriterion]]:
        """Parse a single guideline element into id, name, principle number, and criteria."""
        gl_id = elem.get("id", elem.get("name", ""))
        gl_name = ""
        principle_num = 1

        # Extract guideline id and name
        id_match = re.match(r"(\d+)\.(\d+)", gl_id)
        if id_match:
            principle_num = int(id_match.group(1))
        else:
            # Try to extract from text content
            for child in elem:
                text = (child.text or "").strip()
                id_match = re.match(r"(\d+)\.(\d+)", text)
                if id_match:
                    gl_id = id_match.group(0)
                    principle_num = int(id_match.group(1))
                    break

        # Extract name from title child or name attribute
        title_elem = elem.find("title") or elem.find("name") or elem.find("head")
        if title_elem is not None and title_elem.text:
            gl_name = title_elem.text.strip()
        elif elem.get("name"):
            gl_name = elem.get("name", "")

        # Parse success criteria within this guideline
        criteria: List[SuccessCriterion] = []
        sc_elems = (
            elem.findall(".//successcriterion")
            or elem.findall(".//success-criterion")
            or elem.findall(".//sc")
        )

        principle = _PRINCIPLE_MAP.get(principle_num, WCAGPrinciple.PERCEIVABLE)

        for sc_elem in sc_elems:
            sc = self._parse_criterion(sc_elem, gl_id, principle)
            if sc:
                criteria.append(sc)

        return gl_id, gl_name, principle_num, criteria

    def _parse_criterion(
        self, elem: Any, guideline_id: str, principle: WCAGPrinciple
    ) -> Optional[SuccessCriterion]:
        """Parse a single success criterion element."""
        sc_id = elem.get("id", elem.get("name", ""))
        if not sc_id:
            # Try text content
            text = (elem.text or "").strip()
            m = re.match(r"(\d+\.\d+\.\d+)", text)
            if m:
                sc_id = m.group(1)
            else:
                return None

        # Normalise id format
        sc_id = sc_id.strip()
        if not re.match(r"\d+\.\d+\.\d+", sc_id):
            return None

        # Extract level
        level_str = elem.get("level", elem.get("conformance-level", "A")).strip().upper()
        level = _LEVEL_MAP.get(level_str, ConformanceLevel.A)

        # Extract name
        name = ""
        name_elem = elem.find("title") or elem.find("name") or elem.find("head")
        if name_elem is not None and name_elem.text:
            name = name_elem.text.strip()
        elif elem.get("name"):
            name = elem.get("name", "")

        # Extract description
        desc = ""
        desc_elem = elem.find("description") or elem.find("text") or elem.find("p")
        if desc_elem is not None:
            desc = (desc_elem.text or "").strip()

        # Construct URL
        base_url = f"https://www.w3.org/TR/WCAG22/"
        url = f"{base_url}#{''.join(name.lower().split())}" if name else base_url

        return SuccessCriterion(
            sc_id=sc_id,
            name=name,
            level=level,
            principle=principle,
            guideline_id=guideline_id,
            description=desc,
            url=url,
        )

    def _extract_techniques(self, gl_elem: Any, sc_id: str) -> List[TechniqueRef]:
        """Extract technique references for a success criterion."""
        techniques: List[TechniqueRef] = []

        for tech_elem in gl_elem.findall(".//technique"):
            tech_id = tech_elem.get("id", "")
            title_elem = tech_elem.find("title") or tech_elem.find("name")
            title = (title_elem.text if title_elem is not None and title_elem.text else "").strip()
            category = tech_elem.get("type", "sufficient")

            if tech_id:
                techniques.append(TechniqueRef(
                    technique_id=tech_id,
                    title=title,
                    url=f"https://www.w3.org/WAI/WCAG22/Techniques/{tech_id}",
                    category=category,
                ))

        return techniques


# ═══════════════════════════════════════════════════════════════════════════
# Built-in WCAG 2.2 catalogue
# ═══════════════════════════════════════════════════════════════════════════

_GUIDELINE_NAMES: Dict[str, str] = {
    "1.1": "Text Alternatives",
    "1.2": "Time-based Media",
    "1.3": "Adaptable",
    "1.4": "Distinguishable",
    "2.1": "Keyboard Accessible",
    "2.2": "Enough Time",
    "2.3": "Seizures and Physical Reactions",
    "2.4": "Navigable",
    "2.5": "Input Modalities",
    "3.1": "Readable",
    "3.2": "Predictable",
    "3.3": "Input Assistance",
    "4.1": "Compatible",
}


def _BUILTIN_CRITERIA_22() -> List[SuccessCriterion]:
    """Return the complete WCAG 2.2 success criteria catalogue."""
    P = WCAGPrinciple
    L = ConformanceLevel
    _base = "https://www.w3.org/TR/WCAG22/"

    # (sc_id, name, level, principle, guideline_id, description)
    _raw: List[Tuple[str, str, ConformanceLevel, WCAGPrinciple, str, str]] = [
        # 1 Perceivable
        ("1.1.1", "Non-text Content", L.A, P.PERCEIVABLE, "1.1",
         "All non-text content has a text alternative."),
        ("1.2.1", "Audio-only and Video-only (Prerecorded)", L.A, P.PERCEIVABLE, "1.2",
         "Alternatives provided for prerecorded audio-only and video-only media."),
        ("1.2.2", "Captions (Prerecorded)", L.A, P.PERCEIVABLE, "1.2",
         "Captions are provided for all prerecorded audio content."),
        ("1.2.3", "Audio Description or Media Alternative (Prerecorded)", L.A, P.PERCEIVABLE, "1.2",
         "An alternative or audio description is provided for prerecorded video."),
        ("1.2.4", "Captions (Live)", L.AA, P.PERCEIVABLE, "1.2",
         "Captions are provided for all live audio content."),
        ("1.2.5", "Audio Description (Prerecorded)", L.AA, P.PERCEIVABLE, "1.2",
         "Audio description is provided for all prerecorded video content."),
        ("1.2.6", "Sign Language (Prerecorded)", L.AAA, P.PERCEIVABLE, "1.2",
         "Sign language interpretation is provided for all prerecorded audio content."),
        ("1.2.7", "Extended Audio Description (Prerecorded)", L.AAA, P.PERCEIVABLE, "1.2",
         "Extended audio description is provided for all prerecorded video."),
        ("1.2.8", "Media Alternative (Prerecorded)", L.AAA, P.PERCEIVABLE, "1.2",
         "An alternative for time-based media is provided."),
        ("1.2.9", "Audio-only (Live)", L.AAA, P.PERCEIVABLE, "1.2",
         "An alternative is provided for live audio-only content."),
        ("1.3.1", "Info and Relationships", L.A, P.PERCEIVABLE, "1.3",
         "Information and relationships conveyed through presentation can be programmatically determined."),
        ("1.3.2", "Meaningful Sequence", L.A, P.PERCEIVABLE, "1.3",
         "The correct reading sequence can be programmatically determined."),
        ("1.3.3", "Sensory Characteristics", L.A, P.PERCEIVABLE, "1.3",
         "Instructions do not rely solely on sensory characteristics."),
        ("1.3.4", "Orientation", L.AA, P.PERCEIVABLE, "1.3",
         "Content does not restrict its view to a single display orientation."),
        ("1.3.5", "Identify Input Purpose", L.AA, P.PERCEIVABLE, "1.3",
         "The purpose of each input field can be programmatically determined."),
        ("1.3.6", "Identify Purpose", L.AAA, P.PERCEIVABLE, "1.3",
         "The purpose of UI components and regions can be programmatically determined."),
        ("1.4.1", "Use of Color", L.A, P.PERCEIVABLE, "1.4",
         "Color is not used as the only visual means of conveying information."),
        ("1.4.2", "Audio Control", L.A, P.PERCEIVABLE, "1.4",
         "A mechanism is available to pause or stop automatically playing audio."),
        ("1.4.3", "Contrast (Minimum)", L.AA, P.PERCEIVABLE, "1.4",
         "Text has a contrast ratio of at least 4.5:1 (3:1 for large text)."),
        ("1.4.4", "Resize Text", L.AA, P.PERCEIVABLE, "1.4",
         "Text can be resized without assistive technology up to 200%."),
        ("1.4.5", "Images of Text", L.AA, P.PERCEIVABLE, "1.4",
         "Text is used to convey information rather than images of text."),
        ("1.4.6", "Contrast (Enhanced)", L.AAA, P.PERCEIVABLE, "1.4",
         "Text has a contrast ratio of at least 7:1 (4.5:1 for large text)."),
        ("1.4.7", "Low or No Background Audio", L.AAA, P.PERCEIVABLE, "1.4",
         "Prerecorded audio-only content has minimal background sounds."),
        ("1.4.8", "Visual Presentation", L.AAA, P.PERCEIVABLE, "1.4",
         "Blocks of text meet specific visual presentation criteria."),
        ("1.4.9", "Images of Text (No Exception)", L.AAA, P.PERCEIVABLE, "1.4",
         "Images of text are only used for pure decoration or essential situations."),
        ("1.4.10", "Reflow", L.AA, P.PERCEIVABLE, "1.4",
         "Content can be presented without loss at 320 CSS px width."),
        ("1.4.11", "Non-text Contrast", L.AA, P.PERCEIVABLE, "1.4",
         "UI components and graphical objects have a contrast ratio of at least 3:1."),
        ("1.4.12", "Text Spacing", L.AA, P.PERCEIVABLE, "1.4",
         "No loss of content when adjusting text spacing properties."),
        ("1.4.13", "Content on Hover or Focus", L.AA, P.PERCEIVABLE, "1.4",
         "Additional content triggered by hover or focus is dismissible and persistent."),

        # 2 Operable
        ("2.1.1", "Keyboard", L.A, P.OPERABLE, "2.1",
         "All functionality is operable through a keyboard interface."),
        ("2.1.2", "No Keyboard Trap", L.A, P.OPERABLE, "2.1",
         "Keyboard focus can always be moved away from any component."),
        ("2.1.3", "Keyboard (No Exception)", L.AAA, P.OPERABLE, "2.1",
         "All functionality is operable through a keyboard without exception."),
        ("2.1.4", "Character Key Shortcuts", L.A, P.OPERABLE, "2.1",
         "Single character key shortcuts can be turned off, remapped, or are only active on focus."),
        ("2.2.1", "Timing Adjustable", L.A, P.OPERABLE, "2.2",
         "Time limits can be adjusted by the user."),
        ("2.2.2", "Pause, Stop, Hide", L.A, P.OPERABLE, "2.2",
         "Moving, blinking, scrolling, or auto-updating content can be paused, stopped, or hidden."),
        ("2.2.3", "No Timing", L.AAA, P.OPERABLE, "2.2",
         "Timing is not an essential part of the activity."),
        ("2.2.4", "Interruptions", L.AAA, P.OPERABLE, "2.2",
         "Interruptions can be postponed or suppressed by the user."),
        ("2.2.5", "Re-authenticating", L.AAA, P.OPERABLE, "2.2",
         "Data is saved when a re-authentication is required."),
        ("2.2.6", "Timeouts", L.AAA, P.OPERABLE, "2.2",
         "Users are warned of inactivity timeouts that cause data loss."),
        ("2.3.1", "Three Flashes or Below Threshold", L.A, P.OPERABLE, "2.3",
         "Content does not flash more than three times per second."),
        ("2.3.2", "Three Flashes", L.AAA, P.OPERABLE, "2.3",
         "Content does not contain anything that flashes more than three times per second."),
        ("2.3.3", "Animation from Interactions", L.AAA, P.OPERABLE, "2.3",
         "Motion animation triggered by interaction can be disabled."),
        ("2.4.1", "Bypass Blocks", L.A, P.OPERABLE, "2.4",
         "A mechanism is available to bypass blocks of content that are repeated."),
        ("2.4.2", "Page Titled", L.A, P.OPERABLE, "2.4",
         "Web pages have titles that describe topic or purpose."),
        ("2.4.3", "Focus Order", L.A, P.OPERABLE, "2.4",
         "Focus order preserves meaning and operability."),
        ("2.4.4", "Link Purpose (In Context)", L.A, P.OPERABLE, "2.4",
         "The purpose of each link can be determined from the link text."),
        ("2.4.5", "Multiple Ways", L.AA, P.OPERABLE, "2.4",
         "More than one way is available to locate a web page within a set."),
        ("2.4.6", "Headings and Labels", L.AA, P.OPERABLE, "2.4",
         "Headings and labels describe topic or purpose."),
        ("2.4.7", "Focus Visible", L.AA, P.OPERABLE, "2.4",
         "Keyboard focus indicator is visible."),
        ("2.4.8", "Location", L.AAA, P.OPERABLE, "2.4",
         "Information about the user's location within a set of web pages is available."),
        ("2.4.9", "Link Purpose (Link Only)", L.AAA, P.OPERABLE, "2.4",
         "Purpose of each link can be identified from the link text alone."),
        ("2.4.10", "Section Headings", L.AAA, P.OPERABLE, "2.4",
         "Section headings are used to organise content."),
        ("2.4.11", "Focus Not Obscured (Minimum)", L.AA, P.OPERABLE, "2.4",
         "When a component receives focus, it is not entirely hidden."),
        ("2.4.12", "Focus Not Obscured (Enhanced)", L.AAA, P.OPERABLE, "2.4",
         "When a component receives focus, no part of it is hidden."),
        ("2.4.13", "Focus Appearance", L.AAA, P.OPERABLE, "2.4",
         "Focus indicators meet minimum area, contrast, and change requirements."),
        ("2.5.1", "Pointer Gestures", L.A, P.OPERABLE, "2.5",
         "All multipoint or path-based gestures have single-pointer alternatives."),
        ("2.5.2", "Pointer Cancellation", L.A, P.OPERABLE, "2.5",
         "Functions triggered by single pointer can be cancelled."),
        ("2.5.3", "Label in Name", L.A, P.OPERABLE, "2.5",
         "The accessible name contains the visible label text."),
        ("2.5.4", "Motion Actuation", L.A, P.OPERABLE, "2.5",
         "Functionality triggered by device motion has UI alternatives."),
        ("2.5.5", "Target Size (Enhanced)", L.AAA, P.OPERABLE, "2.5",
         "Touch/click targets are at least 44 by 44 CSS pixels."),
        ("2.5.6", "Concurrent Input Mechanisms", L.AAA, P.OPERABLE, "2.5",
         "Content does not restrict the input modalities available."),
        ("2.5.7", "Dragging Movements", L.AA, P.OPERABLE, "2.5",
         "Dragging functionality has single-pointer alternatives."),
        ("2.5.8", "Target Size (Minimum)", L.AA, P.OPERABLE, "2.5",
         "Touch/click targets are at least 24 by 24 CSS pixels."),

        # 3 Understandable
        ("3.1.1", "Language of Page", L.A, P.UNDERSTANDABLE, "3.1",
         "The default human language of each web page can be programmatically determined."),
        ("3.1.2", "Language of Parts", L.AA, P.UNDERSTANDABLE, "3.1",
         "The language of each passage can be programmatically determined."),
        ("3.1.3", "Unusual Words", L.AAA, P.UNDERSTANDABLE, "3.1",
         "A mechanism is available for identifying specific definitions of words."),
        ("3.1.4", "Abbreviations", L.AAA, P.UNDERSTANDABLE, "3.1",
         "A mechanism for identifying the expanded form of abbreviations is available."),
        ("3.1.5", "Reading Level", L.AAA, P.UNDERSTANDABLE, "3.1",
         "Supplemental content is provided when text requires advanced reading ability."),
        ("3.1.6", "Pronunciation", L.AAA, P.UNDERSTANDABLE, "3.1",
         "A mechanism is available for identifying pronunciation of words."),
        ("3.2.1", "On Focus", L.A, P.UNDERSTANDABLE, "3.2",
         "Receiving focus does not initiate a change of context."),
        ("3.2.2", "On Input", L.A, P.UNDERSTANDABLE, "3.2",
         "Changing a UI component setting does not automatically change context."),
        ("3.2.3", "Consistent Navigation", L.AA, P.UNDERSTANDABLE, "3.2",
         "Navigational mechanisms that are repeated are consistent."),
        ("3.2.4", "Consistent Identification", L.AA, P.UNDERSTANDABLE, "3.2",
         "Components with the same functionality are identified consistently."),
        ("3.2.5", "Change on Request", L.AAA, P.UNDERSTANDABLE, "3.2",
         "Changes of context are initiated only by user request."),
        ("3.2.6", "Consistent Help", L.A, P.UNDERSTANDABLE, "3.2",
         "Help mechanisms are located in the same relative order."),
        ("3.3.1", "Error Identification", L.A, P.UNDERSTANDABLE, "3.3",
         "Input errors are automatically detected and described in text."),
        ("3.3.2", "Labels or Instructions", L.A, P.UNDERSTANDABLE, "3.3",
         "Labels or instructions are provided when content requires user input."),
        ("3.3.3", "Error Suggestion", L.AA, P.UNDERSTANDABLE, "3.3",
         "Known suggestions for correcting input errors are provided."),
        ("3.3.4", "Error Prevention (Legal, Financial, Data)", L.AA, P.UNDERSTANDABLE, "3.3",
         "Submissions are reversible, checked, or confirmed."),
        ("3.3.5", "Help", L.AAA, P.UNDERSTANDABLE, "3.3",
         "Context-sensitive help is available."),
        ("3.3.6", "Error Prevention (All)", L.AAA, P.UNDERSTANDABLE, "3.3",
         "All submissions are reversible, checked, or confirmed."),
        ("3.3.7", "Redundant Entry", L.A, P.UNDERSTANDABLE, "3.3",
         "Information previously entered is auto-populated or selectable."),
        ("3.3.8", "Accessible Authentication (Minimum)", L.AA, P.UNDERSTANDABLE, "3.3",
         "No cognitive function test for any step of authentication."),
        ("3.3.9", "Accessible Authentication (Enhanced)", L.AAA, P.UNDERSTANDABLE, "3.3",
         "No cognitive function test for any step of authentication (no exceptions)."),

        # 4 Robust
        ("4.1.1", "Parsing", L.A, P.ROBUST, "4.1",
         "In content implemented using markup languages, elements are complete and properly nested."),
        ("4.1.2", "Name, Role, Value", L.A, P.ROBUST, "4.1",
         "For all UI components, name and role can be programmatically determined."),
        ("4.1.3", "Status Messages", L.AA, P.ROBUST, "4.1",
         "Status messages can be programmatically determined without receiving focus."),
    ]

    return [
        SuccessCriterion(
            sc_id=sc_id, name=name, level=level, principle=principle,
            guideline_id=gl_id, description=desc,
            url=f"https://www.w3.org/TR/WCAG22/#{name.lower().replace(' ', '-').replace('(', '').replace(')', '')}",
        )
        for sc_id, name, level, principle, gl_id, desc in _raw
    ]


__all__ = [
    "TechniqueRef",
    "UnderstandingLink",
    "WCAGSpecification",
    "WCAGXMLParser",
]

"""Unit tests for usability_oracle.wcag.parser — WCAG 2.2 parser.

Tests built-in catalogue, XML parsing, version detection, principle/
guideline/criterion extraction, and error handling.
"""

from __future__ import annotations

import pytest

from usability_oracle.core.errors import ParseError
from usability_oracle.wcag.parser import (
    WCAGXMLParser,
    WCAGSpecification,
    TechniqueRef,
    UnderstandingLink,
)
from usability_oracle.wcag.types import (
    ConformanceLevel,
    SuccessCriterion,
    WCAGGuideline,
    WCAGPrinciple,
)


# ---------------------------------------------------------------------------
# Sample XML documents
# ---------------------------------------------------------------------------

SIMPLE_WCAG_XML = """<?xml version="1.0" encoding="UTF-8"?>
<!-- WCAG 2.2 -->
<guidelines version="WCAG 2.2">
  <guideline id="1.1">
    <title>Text Alternatives</title>
    <successcriterion id="1.1.1" level="A">
      <title>Non-text Content</title>
      <description>All non-text content has a text alternative.</description>
    </successcriterion>
  </guideline>
  <guideline id="1.4">
    <title>Distinguishable</title>
    <successcriterion id="1.4.3" level="AA">
      <title>Contrast (Minimum)</title>
      <description>Text has a contrast ratio of at least 4.5:1.</description>
    </successcriterion>
    <successcriterion id="1.4.6" level="AAA">
      <title>Contrast (Enhanced)</title>
      <description>Text has a contrast ratio of at least 7:1.</description>
    </successcriterion>
  </guideline>
</guidelines>
"""

WCAG_21_XML = """<?xml version="1.0" encoding="UTF-8"?>
<!-- WCAG 2.1 -->
<guidelines version="WCAG 2.1">
  <guideline id="2.1">
    <title>Keyboard Accessible</title>
    <successcriterion id="2.1.1" level="A">
      <title>Keyboard</title>
    </successcriterion>
  </guideline>
</guidelines>
"""

WCAG_20_XML = """<?xml version="1.0" encoding="UTF-8"?>
<!-- WCAG 2.0 -->
<guidelines version="WCAG 2.0">
  <guideline id="3.1">
    <title>Readable</title>
    <successcriterion id="3.1.1" level="A">
      <title>Language of Page</title>
    </successcriterion>
  </guideline>
</guidelines>
"""

MALFORMED_XML = """<?xml version="1.0"?>
<guidelines><unclosed>
"""

EMPTY_GUIDELINES_XML = """<?xml version="1.0" encoding="UTF-8"?>
<guidelines version="WCAG 2.2">
</guidelines>
"""


# ═══════════════════════════════════════════════════════════════════════════
# Built-in catalogue tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBuiltinCatalogue:
    """Tests for WCAGXMLParser.load_builtin() and built-in criteria."""

    def test_load_builtin_returns_specification(self) -> None:
        parser = WCAGXMLParser()
        spec = parser.load_builtin()
        assert isinstance(spec, WCAGSpecification)
        assert spec.version == "2.2"

    def test_builtin_has_all_criteria(self) -> None:
        parser = WCAGXMLParser()
        spec = parser.load_builtin()
        # WCAG 2.2 has 86 or 87 criteria (78 from 2.0, +12 from 2.1, +9 from 2.2 minus deprecated)
        assert spec.criterion_count >= 78

    def test_builtin_has_all_four_principles(self) -> None:
        parser = WCAGXMLParser()
        spec = parser.load_builtin()
        principles = {sc.principle for sc in spec.criteria}
        assert WCAGPrinciple.PERCEIVABLE in principles
        assert WCAGPrinciple.OPERABLE in principles
        assert WCAGPrinciple.UNDERSTANDABLE in principles
        assert WCAGPrinciple.ROBUST in principles

    def test_builtin_has_all_three_levels(self) -> None:
        parser = WCAGXMLParser()
        spec = parser.load_builtin()
        levels = {sc.level for sc in spec.criteria}
        assert ConformanceLevel.A in levels
        assert ConformanceLevel.AA in levels
        assert ConformanceLevel.AAA in levels

    def test_criterion_by_id_known(self) -> None:
        parser = WCAGXMLParser()
        spec = parser.load_builtin()
        sc = spec.criterion_by_id("1.1.1")
        assert sc is not None
        assert sc.name == "Non-text Content"
        assert sc.level == ConformanceLevel.A
        assert sc.principle == WCAGPrinciple.PERCEIVABLE

    def test_criterion_by_id_unknown_returns_none(self) -> None:
        parser = WCAGXMLParser()
        spec = parser.load_builtin()
        assert spec.criterion_by_id("99.99.99") is None

    def test_criteria_at_level_a(self) -> None:
        parser = WCAGXMLParser()
        spec = parser.load_builtin()
        a_criteria = spec.criteria_at_level(ConformanceLevel.A)
        assert len(a_criteria) > 0
        assert all(sc.level == ConformanceLevel.A for sc in a_criteria)

    def test_criteria_at_level_aa_includes_a(self) -> None:
        parser = WCAGXMLParser()
        spec = parser.load_builtin()
        a_criteria = spec.criteria_at_level(ConformanceLevel.A)
        aa_criteria = spec.criteria_at_level(ConformanceLevel.AA)
        assert len(aa_criteria) > len(a_criteria)

    def test_criteria_at_level_aaa_includes_all(self) -> None:
        parser = WCAGXMLParser()
        spec = parser.load_builtin()
        aaa_criteria = spec.criteria_at_level(ConformanceLevel.AAA)
        assert len(aaa_criteria) == spec.criterion_count

    def test_guidelines_for_principle(self) -> None:
        parser = WCAGXMLParser()
        spec = parser.load_builtin()
        perceivable = spec.guidelines_for_principle(WCAGPrinciple.PERCEIVABLE)
        assert len(perceivable) > 0
        assert all(g.principle == WCAGPrinciple.PERCEIVABLE for g in perceivable)

    def test_guidelines_have_criteria(self) -> None:
        parser = WCAGXMLParser()
        spec = parser.load_builtin()
        for g in spec.guidelines:
            assert g.criterion_count > 0

    def test_understanding_links_generated(self) -> None:
        parser = WCAGXMLParser()
        spec = parser.load_builtin()
        assert "1.1.1" in spec.understanding_links
        link = spec.understanding_links["1.1.1"]
        assert isinstance(link, UnderstandingLink)
        assert "1.1.1" in link.sc_id

    def test_known_criterion_411_parsing(self) -> None:
        parser = WCAGXMLParser()
        spec = parser.load_builtin()
        sc = spec.criterion_by_id("4.1.1")
        assert sc is not None
        assert sc.principle == WCAGPrinciple.ROBUST
        assert sc.level == ConformanceLevel.A

    def test_known_criterion_143_contrast(self) -> None:
        parser = WCAGXMLParser()
        spec = parser.load_builtin()
        sc = spec.criterion_by_id("1.4.3")
        assert sc is not None
        assert sc.level == ConformanceLevel.AA
        assert "contrast" in sc.name.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Protocol conformance
# ═══════════════════════════════════════════════════════════════════════════


class TestProtocolConformance:
    """Tests for WCAGParser protocol methods."""

    def test_load_criteria_returns_sequence(self) -> None:
        parser = WCAGXMLParser()
        criteria = parser.load_criteria()
        assert len(criteria) > 0
        assert all(isinstance(sc, SuccessCriterion) for sc in criteria)

    def test_load_guidelines_returns_sequence(self) -> None:
        parser = WCAGXMLParser()
        guidelines = parser.load_guidelines()
        assert len(guidelines) > 0
        assert all(isinstance(g, WCAGGuideline) for g in guidelines)

    def test_criteria_for_level(self) -> None:
        parser = WCAGXMLParser()
        aa_criteria = parser.criteria_for_level(ConformanceLevel.AA)
        assert len(aa_criteria) > 0

    def test_criterion_by_id_protocol(self) -> None:
        parser = WCAGXMLParser()
        sc = parser.criterion_by_id("2.4.1")
        assert sc is not None
        assert sc.sc_id == "2.4.1"

    def test_lazy_load_on_protocol_call(self) -> None:
        """Protocol methods should auto-load the builtin catalogue."""
        parser = WCAGXMLParser()
        assert parser._spec is None
        _ = parser.load_criteria()
        assert parser._spec is not None


# ═══════════════════════════════════════════════════════════════════════════
# XML parsing
# ═══════════════════════════════════════════════════════════════════════════


class TestXMLParsing:
    """Tests for XML parsing with sample documents."""

    @pytest.fixture(autouse=True)
    def _skip_without_lxml(self) -> None:
        try:
            from lxml import etree  # noqa: F401
        except ImportError:
            pytest.skip("lxml not available")

    def test_parse_simple_xml(self) -> None:
        parser = WCAGXMLParser()
        spec = parser.parse_xml(SIMPLE_WCAG_XML)
        assert isinstance(spec, WCAGSpecification)
        assert len(spec.criteria) >= 2

    def test_version_detection_22(self) -> None:
        parser = WCAGXMLParser()
        spec = parser.parse_xml(SIMPLE_WCAG_XML)
        assert spec.version == "2.2"

    def test_version_detection_21(self) -> None:
        parser = WCAGXMLParser()
        spec = parser.parse_xml(WCAG_21_XML)
        assert spec.version == "2.1"

    def test_version_detection_20(self) -> None:
        parser = WCAGXMLParser()
        spec = parser.parse_xml(WCAG_20_XML)
        assert spec.version == "2.0"

    def test_criterion_levels_from_xml(self) -> None:
        parser = WCAGXMLParser()
        spec = parser.parse_xml(SIMPLE_WCAG_XML)
        sc_111 = spec.criterion_by_id("1.1.1")
        sc_143 = spec.criterion_by_id("1.4.3")
        sc_146 = spec.criterion_by_id("1.4.6")
        assert sc_111 is not None and sc_111.level == ConformanceLevel.A
        assert sc_143 is not None and sc_143.level == ConformanceLevel.AA
        assert sc_146 is not None and sc_146.level == ConformanceLevel.AAA

    def test_principle_extraction(self) -> None:
        parser = WCAGXMLParser()
        spec = parser.parse_xml(SIMPLE_WCAG_XML)
        assert 1 in spec.principles

    def test_guideline_extraction(self) -> None:
        parser = WCAGXMLParser()
        spec = parser.parse_xml(SIMPLE_WCAG_XML)
        assert len(spec.guidelines) >= 2
        gl_ids = {g.guideline_id for g in spec.guidelines}
        assert "1.1" in gl_ids
        assert "1.4" in gl_ids

    def test_malformed_xml_raises_parse_error(self) -> None:
        parser = WCAGXMLParser()
        with pytest.raises(ParseError, match="Malformed XML"):
            parser.parse_xml(MALFORMED_XML)

    def test_empty_guidelines_raises_parse_error(self) -> None:
        parser = WCAGXMLParser()
        with pytest.raises(ParseError, match="No success criteria"):
            parser.parse_xml(EMPTY_GUIDELINES_XML)

    def test_parse_bytes_input(self) -> None:
        parser = WCAGXMLParser()
        spec = parser.parse_xml(SIMPLE_WCAG_XML.encode("utf-8"))
        assert spec.criterion_count >= 2


# ═══════════════════════════════════════════════════════════════════════════
# SuccessCriterion serialization
# ═══════════════════════════════════════════════════════════════════════════


class TestSuccessCriterionSerialization:
    """Tests for SuccessCriterion to_dict/from_dict round trip."""

    def test_round_trip(self) -> None:
        sc = SuccessCriterion(
            sc_id="1.4.3", name="Contrast (Minimum)",
            level=ConformanceLevel.AA,
            principle=WCAGPrinciple.PERCEIVABLE,
            guideline_id="1.4", description="Test desc",
        )
        d = sc.to_dict()
        restored = SuccessCriterion.from_dict(d)
        assert restored.sc_id == sc.sc_id
        assert restored.level == sc.level
        assert restored.principle == sc.principle

    def test_to_dict_keys(self) -> None:
        sc = SuccessCriterion(
            sc_id="2.1.1", name="Keyboard",
            level=ConformanceLevel.A,
            principle=WCAGPrinciple.OPERABLE,
            guideline_id="2.1",
        )
        d = sc.to_dict()
        assert d["sc_id"] == "2.1.1"
        assert d["level"] == "A"
        assert d["principle"] == "operable"


# ═══════════════════════════════════════════════════════════════════════════
# WCAGGuideline
# ═══════════════════════════════════════════════════════════════════════════


class TestWCAGGuideline:
    """Tests for WCAGGuideline."""

    def test_criteria_at_level(self) -> None:
        sc_a = SuccessCriterion(sc_id="1.1.1", name="A", level=ConformanceLevel.A,
                                principle=WCAGPrinciple.PERCEIVABLE, guideline_id="1.1")
        sc_aa = SuccessCriterion(sc_id="1.1.2", name="AA", level=ConformanceLevel.AA,
                                 principle=WCAGPrinciple.PERCEIVABLE, guideline_id="1.1")
        gl = WCAGGuideline(guideline_id="1.1", name="Test", principle=WCAGPrinciple.PERCEIVABLE,
                           criteria=(sc_a, sc_aa))
        assert gl.criterion_count == 2
        assert len(gl.criteria_at_level(ConformanceLevel.A)) == 1
        assert len(gl.criteria_at_level(ConformanceLevel.AA)) == 2

    def test_criterion_count(self) -> None:
        gl = WCAGGuideline(guideline_id="1.1", name="Test", principle=WCAGPrinciple.PERCEIVABLE,
                           criteria=())
        assert gl.criterion_count == 0


# ═══════════════════════════════════════════════════════════════════════════
# ConformanceLevel ordering
# ═══════════════════════════════════════════════════════════════════════════


class TestConformanceLevel:
    """Tests for ConformanceLevel comparison."""

    def test_ordering(self) -> None:
        assert ConformanceLevel.A < ConformanceLevel.AA
        assert ConformanceLevel.AA < ConformanceLevel.AAA
        assert ConformanceLevel.A <= ConformanceLevel.A

    def test_numeric(self) -> None:
        assert ConformanceLevel.A.numeric == 1
        assert ConformanceLevel.AA.numeric == 2
        assert ConformanceLevel.AAA.numeric == 3

    def test_str(self) -> None:
        assert str(ConformanceLevel.AA) == "AA"

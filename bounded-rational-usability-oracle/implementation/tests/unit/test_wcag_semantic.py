"""Unit tests for usability_oracle.wcag.semantic — semantic structure analysis.

Tests heading extraction/validation, landmark detection, list/table/form
structure validation, ARIA role validation, and the aggregate
analyse_semantics function.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from usability_oracle.wcag.semantic import (
    HeadingInfo,
    LandmarkInfo,
    SemanticAnalysisResult,
    analyse_semantics,
    extract_headings,
    extract_landmarks,
    validate_aria_roles,
    validate_form_structure,
    validate_heading_hierarchy,
    validate_landmarks,
    validate_list_structure,
    validate_table_structure,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _node(
    role: str = "",
    name: str = "",
    nid: str = "",
    properties: dict | None = None,
    children: list | None = None,
    depth: int = 0,
    parent: SimpleNamespace | None = None,
) -> SimpleNamespace:
    n = SimpleNamespace(
        id=nid,
        node_id=nid,
        role=role,
        name=name,
        properties=properties or {},
        children=children or [],
        depth=depth,
        bounding_box=None,
        state=SimpleNamespace(hidden=False, disabled=False),
        parent=parent,
    )
    return n


def _tree(root: SimpleNamespace, metadata: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(root=root, metadata=metadata or {})


# ═══════════════════════════════════════════════════════════════════════════
# extract_headings
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractHeadings:
    """Tests for heading extraction from accessibility trees."""

    def test_heading_with_aria_level(self) -> None:
        h = _node(role="heading", name="Title", nid="h1",
                  properties={"aria-level": 1})
        root = _node(children=[h])
        headings = extract_headings(_tree(root))
        assert len(headings) == 1
        assert isinstance(headings[0], HeadingInfo)
        assert headings[0].level == 1
        assert headings[0].name == "Title"

    def test_h1_through_h6_roles(self) -> None:
        nodes = [_node(role=f"h{i}", name=f"H{i}", nid=f"h{i}") for i in range(1, 7)]
        root = _node(children=nodes)
        headings = extract_headings(_tree(root))
        assert len(headings) == 6
        levels = [h.level for h in headings]
        assert levels == [1, 2, 3, 4, 5, 6]

    def test_non_heading_ignored(self) -> None:
        p = _node(role="paragraph", name="Text", nid="p1")
        root = _node(children=[p])
        headings = extract_headings(_tree(root))
        assert len(headings) == 0

    def test_nested_headings(self) -> None:
        h2 = _node(role="heading", name="Sub", nid="h2",
                   properties={"aria-level": 2})
        section = _node(role="region", name="Section", nid="s1", children=[h2])
        h1 = _node(role="heading", name="Main", nid="h1",
                   properties={"aria-level": 1})
        root = _node(children=[h1, section])
        headings = extract_headings(_tree(root))
        assert len(headings) == 2


# ═══════════════════════════════════════════════════════════════════════════
# validate_heading_hierarchy
# ═══════════════════════════════════════════════════════════════════════════


class TestValidateHeadingHierarchy:
    """Tests for heading-level gap and missing-h1 detection."""

    def test_valid_hierarchy_no_violations(self) -> None:
        h1 = _node(role="heading", name="Title", nid="h1",
                   properties={"aria-level": 1})
        h2 = _node(role="heading", name="Sub", nid="h2",
                   properties={"aria-level": 2})
        root = _node(children=[h1, h2])
        violations = validate_heading_hierarchy(_tree(root))
        assert len(violations) == 0

    def test_missing_h1_violation(self) -> None:
        h2 = _node(role="heading", name="Sub", nid="h2",
                   properties={"aria-level": 2})
        root = _node(children=[h2])
        violations = validate_heading_hierarchy(_tree(root))
        msgs = [v.message for v in violations]
        assert any("h1" in m.lower() for m in msgs)

    def test_level_skip_violation(self) -> None:
        h1 = _node(role="heading", name="Title", nid="h1",
                   properties={"aria-level": 1})
        h3 = _node(role="heading", name="Deep", nid="h3",
                   properties={"aria-level": 3})
        root = _node(children=[h1, h3])
        violations = validate_heading_hierarchy(_tree(root))
        assert any("skip" in v.message.lower() for v in violations)

    def test_empty_heading_violation(self) -> None:
        h1 = _node(role="heading", name="Title", nid="h1",
                   properties={"aria-level": 1})
        h2 = _node(role="heading", name="", nid="h2",
                   properties={"aria-level": 2})
        root = _node(children=[h1, h2])
        violations = validate_heading_hierarchy(_tree(root))
        assert any("empty" in v.message.lower() for v in violations)

    def test_no_headings_no_violation(self) -> None:
        root = _node(children=[_node(role="paragraph", name="Text", nid="p1")])
        violations = validate_heading_hierarchy(_tree(root))
        assert len(violations) == 0


# ═══════════════════════════════════════════════════════════════════════════
# extract_landmarks / validate_landmarks
# ═══════════════════════════════════════════════════════════════════════════


class TestLandmarks:
    """Tests for landmark extraction and validation."""

    def test_extract_landmarks(self) -> None:
        main = _node(role="main", name="Content", nid="main")
        nav = _node(role="navigation", name="Nav", nid="nav")
        root = _node(children=[nav, main])
        landmarks = extract_landmarks(_tree(root))
        assert len(landmarks) == 2
        roles = {lm.role for lm in landmarks}
        assert "main" in roles
        assert "navigation" in roles

    def test_landmark_info_attributes(self) -> None:
        child = _node(role="paragraph", nid="p")
        main = _node(role="main", name="Main Content", nid="main", children=[child])
        root = _node(children=[main])
        landmarks = extract_landmarks(_tree(root))
        assert len(landmarks) == 1
        lm = landmarks[0]
        assert isinstance(lm, LandmarkInfo)
        assert lm.role == "main"
        assert lm.label == "Main Content"
        assert lm.child_count == 1

    def test_no_main_landmark_violation(self) -> None:
        nav = _node(role="navigation", name="Nav", nid="nav")
        root = _node(nid="root", children=[nav])
        violations = validate_landmarks(_tree(root))
        assert any("main" in v.message.lower() for v in violations)

    def test_duplicate_nav_without_labels_violation(self) -> None:
        nav1 = _node(role="navigation", name="", nid="n1")
        nav2 = _node(role="navigation", name="", nid="n2")
        main = _node(role="main", nid="main")
        root = _node(nid="root", children=[nav1, nav2, main])
        violations = validate_landmarks(_tree(root))
        label_violations = [v for v in violations if "label" in v.message.lower()]
        assert len(label_violations) >= 1

    def test_valid_landmarks_no_violations(self) -> None:
        nav = _node(role="navigation", name="Primary Nav", nid="nav")
        main = _node(role="main", name="", nid="main")
        root = _node(nid="root", children=[nav, main])
        violations = validate_landmarks(_tree(root))
        # Should have no violation about missing main
        main_violations = [v for v in violations if "No main" in v.message]
        assert len(main_violations) == 0


# ═══════════════════════════════════════════════════════════════════════════
# validate_list_structure
# ═══════════════════════════════════════════════════════════════════════════


class TestValidateListStructure:
    """Tests for list/listitem role validation."""

    def test_valid_list_no_violations(self) -> None:
        items = [_node(role="listitem", name=f"Item {i}", nid=f"li{i}") for i in range(3)]
        lst = _node(role="list", nid="list1", children=items)
        root = _node(children=[lst])
        violations = validate_list_structure(_tree(root))
        assert len(violations) == 0

    def test_non_listitem_child_violation(self) -> None:
        bad_child = _node(role="paragraph", name="Oops", nid="p1")
        li = _node(role="listitem", name="OK", nid="li1")
        lst = _node(role="list", nid="list1", children=[li, bad_child])
        root = _node(children=[lst])
        violations = validate_list_structure(_tree(root))
        assert len(violations) >= 1
        assert any("non-listitem" in v.message.lower() for v in violations)

    def test_presentation_child_allowed(self) -> None:
        wrapper = _node(role="presentation", nid="w1",
                        children=[_node(role="listitem", name="Item", nid="li1")])
        lst = _node(role="list", nid="list1", children=[wrapper])
        root = _node(children=[lst])
        violations = validate_list_structure(_tree(root))
        assert len(violations) == 0


# ═══════════════════════════════════════════════════════════════════════════
# validate_table_structure
# ═══════════════════════════════════════════════════════════════════════════


class TestValidateTableStructure:
    """Tests for table caption, headers, and row consistency."""

    def test_table_without_caption_violation(self) -> None:
        cell = _node(role="cell", name="Data", nid="c1")
        row = _node(role="row", nid="r1", children=[cell])
        table = _node(role="table", name="", nid="t1", children=[row])
        root = _node(children=[table])
        violations = validate_table_structure(_tree(root))
        assert any("caption" in v.message.lower() or "name" in v.message.lower()
                    for v in violations)

    def test_table_without_headers_violation(self) -> None:
        cell1 = _node(role="cell", name="A", nid="c1")
        cell2 = _node(role="cell", name="B", nid="c2")
        row = _node(role="row", nid="r1", children=[cell1, cell2])
        table = _node(role="table", name="Data Table", nid="t1", children=[row])
        root = _node(children=[table])
        violations = validate_table_structure(_tree(root))
        assert any("header" in v.message.lower() for v in violations)

    def test_inconsistent_row_lengths_violation(self) -> None:
        row1 = _node(role="row", nid="r1",
                     children=[_node(role="columnheader", name="A", nid="h1"),
                               _node(role="columnheader", name="B", nid="h2")])
        row2 = _node(role="row", nid="r2",
                     children=[_node(role="cell", name="1", nid="c1")])
        table = _node(role="table", name="Table", nid="t1", children=[row1, row2])
        root = _node(children=[table])
        violations = validate_table_structure(_tree(root))
        assert any("inconsistent" in v.message.lower() for v in violations)

    def test_valid_table_minimal_violations(self) -> None:
        header = _node(role="columnheader", name="Name", nid="h1")
        row1 = _node(role="row", nid="r1", children=[header])
        cell = _node(role="cell", name="Alice", nid="c1")
        row2 = _node(role="row", nid="r2", children=[cell])
        table = _node(role="table", name="Users", nid="t1", children=[row1, row2])
        root = _node(children=[table])
        violations = validate_table_structure(_tree(root))
        # No caption or header violations expected
        caption_v = [v for v in violations if "caption" in v.message.lower()]
        header_v = [v for v in violations if "without" in v.message.lower() and "header" in v.message.lower()]
        assert len(caption_v) == 0
        assert len(header_v) == 0


# ═══════════════════════════════════════════════════════════════════════════
# validate_form_structure
# ═══════════════════════════════════════════════════════════════════════════


class TestValidateFormStructure:
    """Tests for form control labelling."""

    def test_unlabelled_textbox_violation(self) -> None:
        textbox = _node(role="textbox", name="", nid="txt1")
        root = _node(children=[textbox])
        violations = validate_form_structure(_tree(root))
        assert any("no accessible name" in v.message.lower() for v in violations)

    def test_labelled_textbox_no_violation(self) -> None:
        textbox = _node(role="textbox", name="Username", nid="txt1")
        root = _node(children=[textbox])
        violations = validate_form_structure(_tree(root))
        name_violations = [v for v in violations if v.node_id == "txt1"
                           and "no accessible name" in v.message.lower()]
        assert len(name_violations) == 0

    def test_unlabelled_checkbox_violation(self) -> None:
        cb = _node(role="checkbox", name="", nid="cb1")
        root = _node(children=[cb])
        violations = validate_form_structure(_tree(root))
        assert any(v.node_id == "cb1" for v in violations)


# ═══════════════════════════════════════════════════════════════════════════
# validate_aria_roles
# ═══════════════════════════════════════════════════════════════════════════


class TestValidateAriaRoles:
    """Tests for ARIA role usage validation (SC 4.1.2)."""

    def test_abstract_role_violation(self) -> None:
        node = _node(role="command", name="Do thing", nid="cmd1")
        root = _node(children=[node])
        violations = validate_aria_roles(_tree(root))
        assert any("abstract" in v.message.lower() for v in violations)

    def test_button_without_name_violation(self) -> None:
        btn = _node(role="button", name="", nid="btn1")
        root = _node(children=[btn])
        violations = validate_aria_roles(_tree(root))
        assert any(v.node_id == "btn1" and "no accessible name" in v.message.lower()
                    for v in violations)

    def test_valid_button_no_violation(self) -> None:
        btn = _node(role="button", name="Submit", nid="btn1")
        root = _node(children=[btn])
        violations = validate_aria_roles(_tree(root))
        btn_violations = [v for v in violations if v.node_id == "btn1"]
        assert len(btn_violations) == 0

    def test_presentation_role_ignored(self) -> None:
        node = _node(role="presentation", name="", nid="p1")
        root = _node(children=[node])
        violations = validate_aria_roles(_tree(root))
        assert all(v.node_id != "p1" for v in violations)


# ═══════════════════════════════════════════════════════════════════════════
# analyse_semantics — aggregate
# ═══════════════════════════════════════════════════════════════════════════


class TestAnalyseSemantics:
    """Tests for the aggregate semantic analysis."""

    def test_returns_result_object(self) -> None:
        root = _node(children=[_node(role="main", nid="main")])
        result = analyse_semantics(_tree(root))
        assert isinstance(result, SemanticAnalysisResult)

    def test_all_violations_aggregates(self) -> None:
        h3 = _node(role="heading", name="Deep", nid="h3",
                   properties={"aria-level": 3})
        root = _node(nid="root", children=[h3])
        result = analyse_semantics(_tree(root))
        assert result.total_violations >= 1
        assert len(result.all_violations) == result.total_violations

    def test_clean_page_low_violations(self) -> None:
        h1 = _node(role="heading", name="Title", nid="h1",
                   properties={"aria-level": 1})
        main = _node(role="main", nid="main", children=[h1])
        root = _node(nid="root", children=[main])
        result = analyse_semantics(_tree(root))
        # May still have minor issues but heading and main should be fine
        heading_v = result.heading_violations
        landmark_v = [v for v in result.landmark_violations if "No main" in v.message]
        assert len(heading_v) == 0
        assert len(landmark_v) == 0

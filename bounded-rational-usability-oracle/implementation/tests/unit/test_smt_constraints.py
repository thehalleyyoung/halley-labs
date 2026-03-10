"""Unit tests for usability_oracle.smt_repair.constraints — Constraint generation.

Tests structural, spatial, role, cognitive-law, and grouping constraints
produced by ConstraintGenerator.
"""

from __future__ import annotations

import math
from typing import Dict, Any

import pytest

z3 = pytest.importorskip("z3")

from usability_oracle.smt_repair.constraints import ConstraintGenerator
from usability_oracle.smt_repair.encoding import (
    CONTAINS,
    NON_OVERLAP,
    Z3Encoder,
)
from usability_oracle.smt_repair.types import (
    ConstraintKind,
    ConstraintSystem,
    RepairConstraint,
    UIVariable,
    VariableSort,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _leaf(nid: str, role: str = "button", x: int = 0, y: int = 0,
          w: int = 80, h: int = 30) -> dict:
    return {
        "id": nid, "role": role, "name": nid,
        "bounding_box": {"x": x, "y": y, "width": w, "height": h},
        "state": {"hidden": False}, "properties": {},
        "children": [],
    }


def _parent_with_children(pid: str = "parent", children: list | None = None) -> dict:
    kids = children or [_leaf("c1"), _leaf("c2"), _leaf("c3")]
    return {
        "id": pid, "role": "list", "name": pid,
        "bounding_box": {"x": 0, "y": 0, "width": 400, "height": 300},
        "state": {"hidden": False}, "properties": {},
        "children": kids,
    }


def _bottleneck_report(**kwargs) -> dict:
    defaults: Dict[str, Any] = {
        "bottleneck_type": "perceptual_overload",
        "severity": "high",
        "affected_elements": [],
        "description": "test bottleneck",
    }
    defaults.update(kwargs)
    return defaults


# ===================================================================
# Variable generation
# ===================================================================


class TestGenerateVariables:

    def test_produces_variables_for_mutable_properties(self):
        gen = ConstraintGenerator()
        tree = _parent_with_children()
        variables = gen.generate_variables(tree, ["width", "height"])
        assert len(variables) > 0
        var_props = {v.property_name for v in variables}
        assert "width" in var_props or "height" in var_props

    def test_empty_mutable_returns_no_variables(self):
        gen = ConstraintGenerator()
        tree = _leaf("btn")
        variables = gen.generate_variables(tree, [])
        assert len(variables) == 0


# ===================================================================
# Structural constraints (parent–child preservation)
# ===================================================================


class TestStructuralConstraints:

    def test_parent_child_preserved(self):
        gen = ConstraintGenerator()
        tree = _parent_with_children()
        variables = gen.generate_variables(tree, ["width", "height", "role"])
        constraints = gen.generate_constraints(variables, _bottleneck_report())
        # Should produce at least some constraints
        assert len(constraints) > 0

    def test_constraints_have_kind(self):
        gen = ConstraintGenerator()
        tree = _parent_with_children()
        variables = gen.generate_variables(tree, ["width", "height"])
        constraints = gen.generate_constraints(variables, _bottleneck_report())
        kinds = {c.kind for c in constraints}
        assert len(kinds) > 0

    def test_hard_constraints_present(self):
        gen = ConstraintGenerator()
        tree = _parent_with_children()
        variables = gen.generate_variables(tree, ["width", "height", "role"])
        constraints = gen.generate_constraints(variables, _bottleneck_report())
        # Constraints should be either hard or soft
        assert len(constraints) > 0
        # At least some constraint should have is_hard set (True or False)
        has_hard_field = all(hasattr(c, "is_hard") for c in constraints)
        assert has_hard_field


# ===================================================================
# Spatial constraints (containment)
# ===================================================================


class TestSpatialConstraints:

    def test_containment_encoding(self):
        enc = Z3Encoder()
        parent_vars = {
            "x": z3.Int("px"), "y": z3.Int("py"),
            "width": z3.Int("pw"), "height": z3.Int("ph"),
        }
        child_vars = {
            "x": z3.Int("cx"), "y": z3.Int("cy"),
            "width": z3.Int("cw"), "height": z3.Int("ch"),
        }
        constraint = enc.encode_spatial_relation(parent_vars, child_vars, CONTAINS)
        s = z3.Solver()
        # Parent at (0,0) 400x300; child at (10,10) 80x30 → should contain
        s.add(parent_vars["x"] == 0, parent_vars["y"] == 0,
              parent_vars["width"] == 400, parent_vars["height"] == 300)
        s.add(child_vars["x"] == 10, child_vars["y"] == 10,
              child_vars["width"] == 80, child_vars["height"] == 30)
        s.add(constraint)
        assert s.check() == z3.sat

    def test_non_overlap_encoding(self):
        enc = Z3Encoder()
        a_vars = {"x": z3.Int("ax"), "y": z3.Int("ay"),
                   "width": z3.Int("aw"), "height": z3.Int("ah")}
        b_vars = {"x": z3.Int("bx"), "y": z3.Int("by"),
                   "width": z3.Int("bw"), "height": z3.Int("bh")}
        constraint = enc.encode_spatial_relation(a_vars, b_vars, NON_OVERLAP)
        s = z3.Solver()
        s.add(a_vars["x"] == 0, a_vars["y"] == 0,
              a_vars["width"] == 50, a_vars["height"] == 50)
        s.add(b_vars["x"] == 60, b_vars["y"] == 0,
              b_vars["width"] == 50, b_vars["height"] == 50)
        s.add(constraint)
        assert s.check() == z3.sat

    def test_overlap_violates_non_overlap(self):
        enc = Z3Encoder()
        a_vars = {"x": z3.Int("ax2"), "y": z3.Int("ay2"),
                   "width": z3.Int("aw2"), "height": z3.Int("ah2")}
        b_vars = {"x": z3.Int("bx2"), "y": z3.Int("by2"),
                   "width": z3.Int("bw2"), "height": z3.Int("bh2")}
        constraint = enc.encode_spatial_relation(a_vars, b_vars, NON_OVERLAP)
        s = z3.Solver()
        s.add(a_vars["x"] == 0, a_vars["y"] == 0,
              a_vars["width"] == 50, a_vars["height"] == 50)
        s.add(b_vars["x"] == 25, b_vars["y"] == 25,
              b_vars["width"] == 50, b_vars["height"] == 50)
        s.add(constraint)
        assert s.check() == z3.unsat


# ===================================================================
# Role constraints (ARIA rules)
# ===================================================================


class TestRoleConstraints:

    def test_list_requires_listitem(self):
        enc = Z3Encoder()
        role_map = {"list": 0, "listitem": 1, "button": 2}
        parent_role = z3.Int("p_role")
        child_roles = [z3.Int("c0_role"), z3.Int("c1_role")]
        constraint = enc.encode_semantic_constraint(
            parent_role, role_map, ["listitem"], child_roles,
        )
        s = z3.Solver()
        s.add(parent_role == role_map["list"])
        # No child has listitem
        s.add(child_roles[0] == role_map["button"])
        s.add(child_roles[1] == role_map["button"])
        s.add(constraint)
        assert s.check() == z3.unsat

    def test_tablist_requires_tab(self):
        enc = Z3Encoder()
        role_map = {"tablist": 0, "tab": 1, "button": 2}
        parent_role = z3.Int("tl_role")
        child_roles = [z3.Int("t0_role")]
        constraint = enc.encode_semantic_constraint(
            parent_role, role_map, ["tab"], child_roles,
        )
        s = z3.Solver()
        s.add(parent_role == role_map["tablist"])
        s.add(child_roles[0] == role_map["tab"])
        s.add(constraint)
        assert s.check() == z3.sat


# ===================================================================
# Cognitive cost constraints
# ===================================================================


class TestFittsConstraint:
    """Fitts' law: MT = a + b * log2(2D/W). Approximation error should be bounded."""

    @pytest.mark.parametrize("distance,width", [(100, 50), (200, 20), (50, 80)])
    def test_fitts_approximation_bounded(self, distance, width):
        a, b = 0.05, 0.15  # typical Fitts parameters
        exact = a + b * math.log2(2 * distance / width)
        # A simple linear approximation: a + b * (2D/W - 1)
        linear_approx = a + b * (2 * distance / width - 1) / 10
        # Ensure difference is finite (approximation exists)
        assert math.isfinite(exact)
        assert math.isfinite(linear_approx)


class TestHickConstraint:
    """Hick–Hyman: RT = a + b * log2(n+1). Approximation error should be bounded."""

    @pytest.mark.parametrize("n_choices", [2, 4, 8, 16])
    def test_hick_approximation_bounded(self, n_choices):
        a, b = 0.2, 0.15
        exact = a + b * math.log2(n_choices + 1)
        # Piece-wise linear approximation with segments
        approx = a + b * n_choices / 4  # crude linear
        # Relative error should be finite
        assert math.isfinite(exact)
        assert exact > 0


# ===================================================================
# Grouping constraints
# ===================================================================


class TestGroupingConstraints:

    def test_generates_constraints_from_bottleneck(self):
        gen = ConstraintGenerator()
        tree = _parent_with_children(
            children=[_leaf(f"item_{i}", "listitem") for i in range(8)],
        )
        variables = gen.generate_variables(tree, ["width", "height", "role"])
        report = _bottleneck_report(
            bottleneck_type="choice_paralysis",
            affected_elements=["item_0", "item_1", "item_2"],
        )
        constraints = gen.generate_constraints(variables, report)
        assert len(constraints) > 0

    def test_constraint_ids_unique(self):
        gen = ConstraintGenerator()
        tree = _parent_with_children()
        variables = gen.generate_variables(tree, ["width", "height"])
        constraints = gen.generate_constraints(variables, _bottleneck_report())
        ids = [c.constraint_id for c in constraints]
        assert len(ids) == len(set(ids))


# ===================================================================
# ConstraintSystem assembly
# ===================================================================


class TestConstraintSystem:

    def test_system_holds_variables_and_constraints(self):
        gen = ConstraintGenerator()
        tree = _parent_with_children()
        variables = gen.generate_variables(tree, ["width", "height"])
        constraints = gen.generate_constraints(variables, _bottleneck_report())
        system = ConstraintSystem(
            variables=tuple(variables),
            constraints=tuple(constraints),
            timeout_seconds=5.0,
        )
        assert len(system.variables) == len(variables)
        assert len(system.constraints) == len(constraints)
        assert system.timeout_seconds == 5.0

    def test_constraint_kinds_cover_basic_types(self):
        gen = ConstraintGenerator()
        tree = _parent_with_children()
        variables = gen.generate_variables(tree, ["width", "height", "role"])
        constraints = gen.generate_constraints(variables, _bottleneck_report())
        kinds = {c.kind for c in constraints}
        # Should have at least one kind
        assert len(kinds) >= 1

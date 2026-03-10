"""Unit tests for usability_oracle.smt_repair.encoding — Z3 encoding of UI trees.

Tests that accessibility tree nodes are correctly encoded as Z3 variables,
domain bounds are respected, spatial relations are encoded properly, and
model decoding recovers the original structure.
"""

from __future__ import annotations

import pytest

z3 = pytest.importorskip("z3")

from usability_oracle.smt_repair.encoding import (
    ABOVE,
    CONTAINS,
    LEFT_OF,
    NON_OVERLAP,
    TreeEncoding,
    Z3Encoder,
    _COMMON_ROLES,
)
from usability_oracle.smt_repair.types import VariableSort


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _leaf_node(nid: str = "btn1", role: str = "button", name: str = "OK",
               x: int = 10, y: int = 20, w: int = 80, h: int = 30) -> dict:
    return {
        "id": nid, "role": role, "name": name,
        "bounding_box": {"x": x, "y": y, "width": w, "height": h},
        "state": {"hidden": False}, "properties": {}, "children": [],
    }


def _tree_with_children() -> dict:
    return {
        "id": "root", "role": "generic", "name": "Root",
        "bounding_box": {"x": 0, "y": 0, "width": 400, "height": 300},
        "state": {"hidden": False}, "properties": {},
        "children": [
            _leaf_node("child_a", "button", "A", 10, 10, 80, 30),
            _leaf_node("child_b", "link", "B", 100, 10, 80, 30),
        ],
    }


# ===================================================================
# TreeEncoding construction
# ===================================================================


class TestEncodeTree:
    """encode_tree should produce Z3 variables for every node."""

    def test_leaf_produces_variables(self):
        enc = Z3Encoder()
        result = enc.encode_tree(_leaf_node())
        # At minimum: x, y, width, height, hidden, role, name_len
        assert len(result.variables) >= 6
        assert len(result.node_vars) == 1

    def test_tree_encodes_all_nodes(self):
        enc = Z3Encoder()
        tree = _tree_with_children()
        result = enc.encode_tree(tree)
        assert "root" in result.node_vars
        assert "child_a" in result.node_vars
        assert "child_b" in result.node_vars

    def test_node_vars_contain_bbox_keys(self):
        enc = Z3Encoder()
        result = enc.encode_tree(_leaf_node("n1"))
        nv = result.node_vars["n1"]
        for key in ("x", "y", "width", "height"):
            assert key in nv

    def test_node_vars_contain_role(self):
        enc = Z3Encoder()
        result = enc.encode_tree(_leaf_node("n1"))
        assert "role" in result.node_vars["n1"]

    def test_empty_tree_encoding(self):
        enc = Z3Encoder()
        tree = {"id": "empty", "role": "generic", "name": "",
                "bounding_box": {}, "state": {}, "properties": {}, "children": []}
        result = enc.encode_tree(tree)
        assert "empty" in result.node_vars
        assert len(result.node_vars) == 1


class TestEncodeNode:
    """encode_node should encode a single node without recursing."""

    def test_single_node(self):
        enc = Z3Encoder()
        result = enc.encode_node(_leaf_node("solo"))
        assert len(result.node_vars) == 1
        assert "solo" in result.node_vars


# ===================================================================
# Variable bounds
# ===================================================================


class TestIntegerBounds:

    def test_default_bounds(self):
        enc = Z3Encoder()
        var, asserts = enc.encode_integer_variable("test_int")
        assert len(asserts) == 2
        s = z3.Solver()
        s.add(asserts)
        s.add(var == 5000)
        assert s.check() == z3.sat

    def test_custom_bounds_respected(self):
        enc = Z3Encoder()
        var, asserts = enc.encode_integer_variable("bounded", bounds=(10, 50))
        s = z3.Solver()
        s.add(asserts)
        s.add(var == 60)
        assert s.check() == z3.unsat

    def test_custom_bounds_lower_edge(self):
        enc = Z3Encoder()
        var, asserts = enc.encode_integer_variable("bounded", bounds=(10, 50))
        s = z3.Solver()
        s.add(asserts)
        s.add(var == 10)
        assert s.check() == z3.sat

    @pytest.mark.parametrize("val", [0, 10000])
    def test_default_edge_values(self, val):
        enc = Z3Encoder()
        var, asserts = enc.encode_integer_variable("edge")
        s = z3.Solver()
        s.add(asserts)
        s.add(var == val)
        assert s.check() == z3.sat


class TestRealBounds:

    def test_default_bounds(self):
        enc = Z3Encoder()
        var, asserts = enc.encode_real_variable("test_real")
        assert len(asserts) == 2

    def test_custom_bounds_respected(self):
        enc = Z3Encoder()
        var, asserts = enc.encode_real_variable("rbounded", bounds=(0.0, 1.0))
        s = z3.Solver()
        s.add(asserts)
        s.add(var == z3.RealVal(1.5))
        assert s.check() == z3.unsat

    def test_custom_bounds_interior(self):
        enc = Z3Encoder()
        var, asserts = enc.encode_real_variable("rbounded", bounds=(0.0, 1.0))
        s = z3.Solver()
        s.add(asserts)
        s.add(var == z3.RealVal(0.5))
        assert s.check() == z3.sat


# ===================================================================
# Enum encoding
# ===================================================================


class TestEnumVariable:

    def test_produces_value_map(self):
        enc = Z3Encoder()
        var, asserts, vmap = enc.encode_enum_variable("role", ["button", "link", "checkbox"])
        assert vmap == {"button": 0, "link": 1, "checkbox": 2}

    def test_domain_constraint_enforced(self):
        enc = Z3Encoder()
        var, asserts, vmap = enc.encode_enum_variable("role", ["a", "b"])
        s = z3.Solver()
        s.add(asserts)
        s.add(var == 5)
        assert s.check() == z3.unsat


# ===================================================================
# Spatial relation encoding
# ===================================================================


class TestSpatialRelation:

    @staticmethod
    def _make_vars(prefix: str):
        return {
            "x": z3.Int(f"{prefix}_x"),
            "y": z3.Int(f"{prefix}_y"),
            "width": z3.Int(f"{prefix}_w"),
            "height": z3.Int(f"{prefix}_h"),
        }

    def test_left_of(self):
        enc = Z3Encoder()
        a, b = self._make_vars("a"), self._make_vars("b")
        constraint = enc.encode_spatial_relation(a, b, LEFT_OF)
        s = z3.Solver()
        s.add(a["x"] == 0, a["width"] == 10, b["x"] == 20)
        s.add(constraint)
        assert s.check() == z3.sat

    def test_left_of_violated(self):
        enc = Z3Encoder()
        a, b = self._make_vars("a"), self._make_vars("b")
        constraint = enc.encode_spatial_relation(a, b, LEFT_OF)
        s = z3.Solver()
        s.add(a["x"] == 0, a["width"] == 30, b["x"] == 20)
        s.add(constraint)
        assert s.check() == z3.unsat

    def test_above(self):
        enc = Z3Encoder()
        a, b = self._make_vars("a"), self._make_vars("b")
        constraint = enc.encode_spatial_relation(a, b, ABOVE)
        s = z3.Solver()
        s.add(a["y"] == 0, a["height"] == 10, b["y"] == 10)
        s.add(constraint)
        assert s.check() == z3.sat

    def test_contains(self):
        enc = Z3Encoder()
        a, b = self._make_vars("a"), self._make_vars("b")
        constraint = enc.encode_spatial_relation(a, b, CONTAINS)
        s = z3.Solver()
        s.add(a["x"] == 0, a["y"] == 0, a["width"] == 100, a["height"] == 100)
        s.add(b["x"] == 10, b["y"] == 10, b["width"] == 20, b["height"] == 20)
        s.add(constraint)
        assert s.check() == z3.sat

    def test_unknown_relation_raises(self):
        enc = Z3Encoder()
        a, b = self._make_vars("a"), self._make_vars("b")
        with pytest.raises(ValueError, match="Unknown"):
            enc.encode_spatial_relation(a, b, "diagonal")


# ===================================================================
# Model decoding
# ===================================================================


class TestDecodeModel:

    def test_decode_recovers_structure(self):
        enc = Z3Encoder()
        node = _leaf_node("dec1", "button", "OK", 10, 20, 80, 30)
        encoding = enc.encode_tree(node)

        s = z3.Solver()
        s.add(encoding.assertions)
        assert s.check() == z3.sat

        model = s.model()
        decoded = enc.decode_model(model, encoding)
        assert "dec1" in decoded

    def test_decode_returns_dict_per_node(self):
        enc = Z3Encoder()
        tree = _tree_with_children()
        encoding = enc.encode_tree(tree)
        s = z3.Solver()
        s.add(encoding.assertions)
        assert s.check() == z3.sat
        decoded = enc.decode_model(s.model(), encoding)
        assert len(decoded) >= 3


# ===================================================================
# Semantic constraint encoding
# ===================================================================


class TestSemanticConstraint:

    def test_required_child_role(self):
        enc = Z3Encoder()
        role_map = {"list": 0, "listitem": 1, "button": 2}
        parent_var = z3.Int("parent_role")
        child_vars = [z3.Int("c0_role"), z3.Int("c1_role")]
        constraint = enc.encode_semantic_constraint(
            parent_var, role_map, ["listitem"], child_vars,
        )
        s = z3.Solver()
        s.add(parent_var == 0)
        s.add(child_vars[0] == 1)  # listitem
        s.add(child_vars[1] == 2)
        s.add(constraint)
        assert s.check() == z3.sat

    def test_missing_child_role_unsat(self):
        enc = Z3Encoder()
        role_map = {"list": 0, "listitem": 1, "button": 2}
        parent_var = z3.Int("parent_role")
        child_vars = [z3.Int("c0_role")]
        constraint = enc.encode_semantic_constraint(
            parent_var, role_map, ["listitem"], child_vars,
        )
        s = z3.Solver()
        s.add(parent_var == 0)
        s.add(child_vars[0] == 2)  # button, not listitem
        s.add(constraint)
        assert s.check() == z3.unsat

    def test_empty_required_children_trivially_true(self):
        enc = Z3Encoder()
        parent_var = z3.Int("p")
        constraint = enc.encode_semantic_constraint(parent_var, {}, [], [])
        s = z3.Solver()
        s.add(constraint)
        assert s.check() == z3.sat

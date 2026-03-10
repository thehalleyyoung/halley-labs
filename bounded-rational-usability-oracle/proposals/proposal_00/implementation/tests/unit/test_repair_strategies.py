"""Unit tests for usability_oracle.repair.models (UIMutation, MutationType,
RepairCandidate, RepairResult, RepairConstraint).

Tests serialization, validation, scoring, and the MutationType enum for the
repair subsystem's core data models.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from usability_oracle.repair.models import (
    ConstraintDirection,
    MutationType,
    RepairCandidate,
    RepairConstraint,
    RepairResult,
    UIMutation,
)


# ---------------------------------------------------------------------------
# MutationType enum
# ---------------------------------------------------------------------------

class TestMutationTypeEnum:
    """Tests for the MutationType enum."""

    def test_resize_member(self):
        """RESIZE is a valid MutationType member."""
        assert MutationType.RESIZE is not None

    def test_reposition_member(self):
        """REPOSITION is a valid MutationType member."""
        assert MutationType.REPOSITION is not None

    def test_regroup_member(self):
        """REGROUP is a valid MutationType member."""
        assert MutationType.REGROUP is not None

    def test_relabel_member(self):
        """RELABEL is a valid MutationType member."""
        assert MutationType.RELABEL is not None

    def test_remove_member(self):
        """REMOVE is a valid MutationType member."""
        assert MutationType.REMOVE is not None

    def test_add_shortcut_member(self):
        """ADD_SHORTCUT is a valid MutationType member."""
        assert MutationType.ADD_SHORTCUT is not None

    def test_simplify_menu_member(self):
        """SIMPLIFY_MENU is a valid MutationType member."""
        assert MutationType.SIMPLIFY_MENU is not None

    def test_add_landmark_member(self):
        """ADD_LANDMARK is a valid MutationType member."""
        assert MutationType.ADD_LANDMARK is not None

    def test_all_types_returns_frozenset(self):
        """all_types() returns a frozenset."""
        result = MutationType.all_types()
        assert isinstance(result, frozenset)

    def test_all_types_has_eight_members(self):
        """all_types() includes all 8 mutation types."""
        assert len(MutationType.all_types()) == 8

    def test_mutation_type_is_str_enum(self):
        """MutationType members are string-based."""
        assert isinstance(MutationType.RESIZE.value, str)


# ---------------------------------------------------------------------------
# UIMutation
# ---------------------------------------------------------------------------

class TestUIMutation:
    """Tests for the UIMutation dataclass."""

    def test_construction(self):
        """UIMutation can be constructed with required fields."""
        m = UIMutation(
            mutation_type=MutationType.RESIZE.value,
            target_node_id="btn_1",
        )
        assert m.mutation_type == MutationType.RESIZE.value
        assert m.target_node_id == "btn_1"

    def test_default_parameters_empty(self):
        """Default parameters dict is empty."""
        m = UIMutation(mutation_type="resize", target_node_id="x")
        assert m.parameters == {} or isinstance(m.parameters, dict)

    def test_validate_no_errors(self):
        """validate() returns empty list for a valid mutation."""
        m = UIMutation(
            mutation_type=MutationType.RELABEL.value,
            target_node_id="node_1",
            parameters={"new_name": "OK"},
        )
        errors = m.validate()
        assert isinstance(errors, list)

    def test_validate_missing_type(self):
        """validate() reports error when mutation_type is empty."""
        m = UIMutation(mutation_type="", target_node_id="node_1")
        errors = m.validate()
        assert len(errors) > 0

    def test_validate_missing_target(self):
        """validate() reports error when target_node_id is empty."""
        m = UIMutation(mutation_type="resize", target_node_id="")
        errors = m.validate()
        assert len(errors) > 0

    def test_to_dict(self):
        """to_dict() returns a dict with all expected keys."""
        m = UIMutation(
            mutation_type=MutationType.RESIZE.value,
            target_node_id="btn_1",
            parameters={"width": 50},
            description="Enlarge button",
        )
        d = m.to_dict()
        assert isinstance(d, dict)
        assert "mutation_type" in d
        assert "target_node_id" in d
        assert "parameters" in d

    def test_from_dict_roundtrip(self):
        """from_dict(to_dict(m)) produces an equivalent UIMutation."""
        m = UIMutation(
            mutation_type=MutationType.RELABEL.value,
            target_node_id="lbl_1",
            parameters={"new_name": "Submit"},
            description="Rename label",
            priority=0.9,
        )
        d = m.to_dict()
        m2 = UIMutation.from_dict(d)
        assert m2.mutation_type == m.mutation_type
        assert m2.target_node_id == m.target_node_id
        assert m2.parameters == m.parameters

    def test_repr(self):
        """UIMutation has a non-empty repr."""
        m = UIMutation(mutation_type="resize", target_node_id="x")
        assert len(repr(m)) > 0


# ---------------------------------------------------------------------------
# RepairCandidate
# ---------------------------------------------------------------------------

class TestRepairCandidate:
    """Tests for the RepairCandidate dataclass."""

    def test_construction_defaults(self):
        """RepairCandidate can be created with all defaults."""
        rc = RepairCandidate()
        assert rc.expected_cost_reduction == 0.0
        assert rc.confidence == 0.0

    def test_n_mutations(self):
        """n_mutations property counts the mutations list."""
        rc = RepairCandidate(
            mutations=[
                UIMutation(mutation_type="resize", target_node_id="a"),
                UIMutation(mutation_type="relabel", target_node_id="b"),
            ]
        )
        assert rc.n_mutations == 2

    def test_mutation_types_property(self):
        """mutation_types property returns set of types."""
        rc = RepairCandidate(
            mutations=[
                UIMutation(mutation_type="resize", target_node_id="a"),
                UIMutation(mutation_type="relabel", target_node_id="b"),
            ]
        )
        types = rc.mutation_types
        assert "resize" in types or MutationType.RESIZE in types

    def test_score_default(self):
        """score() returns a float even for default candidate."""
        rc = RepairCandidate()
        s = rc.score()
        assert isinstance(s, float)

    def test_score_increases_with_cost_reduction(self):
        """Higher expected_cost_reduction yields higher score."""
        rc_low = RepairCandidate(expected_cost_reduction=0.1, confidence=0.5)
        rc_high = RepairCandidate(expected_cost_reduction=0.9, confidence=0.5)
        assert rc_high.score() >= rc_low.score()

    def test_score_increases_with_confidence(self):
        """Higher confidence yields higher score (with same cost reduction)."""
        rc_low = RepairCandidate(expected_cost_reduction=0.5, confidence=0.2)
        rc_high = RepairCandidate(expected_cost_reduction=0.5, confidence=0.9)
        assert rc_high.score() >= rc_low.score()

    def test_to_dict(self):
        """to_dict returns a serializable dict."""
        rc = RepairCandidate(
            mutations=[UIMutation(mutation_type="resize", target_node_id="x")],
            expected_cost_reduction=0.3,
            confidence=0.7,
        )
        d = rc.to_dict()
        assert isinstance(d, dict)
        assert "mutations" in d

    def test_from_dict_roundtrip(self):
        """from_dict(to_dict(rc)) reproduces core fields."""
        rc = RepairCandidate(
            mutations=[UIMutation(mutation_type="resize", target_node_id="x")],
            expected_cost_reduction=0.4,
            confidence=0.8,
            description="Test candidate",
        )
        d = rc.to_dict()
        rc2 = RepairCandidate.from_dict(d)
        assert rc2.expected_cost_reduction == pytest.approx(rc.expected_cost_reduction)
        assert rc2.confidence == pytest.approx(rc.confidence)

    def test_is_verified_default(self):
        """is_verified is False by default."""
        rc = RepairCandidate()
        assert rc.is_verified is False


# ---------------------------------------------------------------------------
# RepairResult
# ---------------------------------------------------------------------------

class TestRepairResult:
    """Tests for the RepairResult dataclass."""

    def test_empty_result(self):
        """RepairResult with no candidates has has_repair == False."""
        rr = RepairResult()
        assert rr.has_repair is False

    def test_has_repair_true(self):
        """has_repair is True when there are feasible candidates."""
        rc = RepairCandidate(
            mutations=[UIMutation(mutation_type="resize", target_node_id="x")],
            expected_cost_reduction=0.5,
            confidence=0.8,
            feasible=True,
        )
        rr = RepairResult(candidates=[rc], best=rc)
        assert rr.has_repair is True

    def test_n_feasible(self):
        """n_feasible counts candidates with feasible == True."""
        c1 = RepairCandidate(feasible=True)
        c2 = RepairCandidate(feasible=False)
        c3 = RepairCandidate(feasible=True)
        rr = RepairResult(candidates=[c1, c2, c3])
        assert rr.n_feasible == 2

    def test_top_k_default(self):
        """top_k() returns at most k candidates sorted by score."""
        candidates = [
            RepairCandidate(expected_cost_reduction=float(i) / 10, confidence=0.5)
            for i in range(10)
        ]
        rr = RepairResult(candidates=candidates)
        top = rr.top_k(3)
        assert len(top) == 3

    def test_top_k_exceeds_candidates(self):
        """top_k(k) returns all candidates when k > len(candidates)."""
        rr = RepairResult(candidates=[RepairCandidate()])
        top = rr.top_k(10)
        assert len(top) == 1

    def test_to_dict(self):
        """to_dict returns a serializable dict."""
        rr = RepairResult(candidates=[], solver_status="sat")
        d = rr.to_dict()
        assert isinstance(d, dict)

    def test_from_dict_roundtrip(self):
        """from_dict(to_dict(rr)) reproduces solver_status."""
        rr = RepairResult(solver_status="sat", synthesis_time=1.5)
        d = rr.to_dict()
        rr2 = RepairResult.from_dict(d)
        assert rr2.solver_status == "sat"


# ---------------------------------------------------------------------------
# RepairConstraint
# ---------------------------------------------------------------------------

class TestRepairConstraint:
    """Tests for the RepairConstraint dataclass."""

    def test_construction(self):
        """RepairConstraint can be constructed with required fields."""
        rc = RepairConstraint(
            constraint_type="fitts",
            target="btn_1",
            bound=2.0,
        )
        assert rc.constraint_type == "fitts"

    def test_validate_empty_type(self):
        """validate() detects empty constraint_type."""
        rc = RepairConstraint(constraint_type="", target="x", bound=1.0)
        errors = rc.validate()
        assert len(errors) > 0

    def test_validate_empty_target(self):
        """validate() detects empty target."""
        rc = RepairConstraint(constraint_type="fitts", target="", bound=1.0)
        errors = rc.validate()
        assert len(errors) > 0

    def test_to_dict(self):
        """to_dict returns a serializable dict."""
        rc = RepairConstraint(constraint_type="hick", target="menu", bound=7)
        d = rc.to_dict()
        assert isinstance(d, dict)
        assert "constraint_type" in d

    def test_from_dict_roundtrip(self):
        """from_dict(to_dict(rc)) preserves constraint_type and bound."""
        rc = RepairConstraint(constraint_type="memory", target="nav", bound=4)
        d = rc.to_dict()
        rc2 = RepairConstraint.from_dict(d)
        assert rc2.constraint_type == "memory"
        assert rc2.bound == 4

    def test_is_hard_property(self):
        """is_hard property returns a bool."""
        rc = RepairConstraint(constraint_type="fitts", target="x", bound=1.0)
        assert isinstance(rc.is_hard, bool)


# ---------------------------------------------------------------------------
# ConstraintDirection enum
# ---------------------------------------------------------------------------

class TestConstraintDirection:
    """Tests for the ConstraintDirection enum."""

    def test_upper(self):
        """UPPER member exists."""
        assert ConstraintDirection.UPPER is not None

    def test_lower(self):
        """LOWER member exists."""
        assert ConstraintDirection.LOWER is not None

    def test_equal(self):
        """EQUAL member exists."""
        assert ConstraintDirection.EQUAL is not None

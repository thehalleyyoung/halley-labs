"""
Tests for arc.algebra.quality_delta
====================================

Covers: ViolationType, SeverityLevel, ConstraintType, DistributionSummary,
ConstraintStatus, QualityState, all QualityOperation subclasses,
QualityDelta lattice operations, algebraic laws, and serialization.
"""

import math

import pytest

try:
    from arc.algebra.quality_delta import (
        ConstraintAdded,
        ConstraintRemoved,
        ConstraintStatus,
        ConstraintType,
        DistributionShift,
        DistributionSummary,
        QualityDelta,
        QualityImprovement,
        QualityOperation,
        QualityState,
        QualityViolation,
        SeverityLevel,
        ViolationType,
    )

    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(not HAS_MODULE, reason="module not available")


# ============================================================================
# 1. ViolationType enum members
# ============================================================================

class TestViolationType:
    """All ViolationType members exist and have expected values."""

    EXPECTED_MEMBERS = [
        "NULL_IN_NON_NULL",
        "UNIQUENESS_VIOLATION",
        "FOREIGN_KEY_VIOLATION",
        "CHECK_VIOLATION",
        "TYPE_MISMATCH",
        "RANGE_VIOLATION",
        "PATTERN_VIOLATION",
        "CUSTOM_RULE_VIOLATION",
        "REFERENTIAL_INTEGRITY",
        "DOMAIN_VIOLATION",
        "STATISTICAL_OUTLIER",
        "COMPLETENESS_VIOLATION",
        "TIMELINESS_VIOLATION",
        "CONSISTENCY_VIOLATION",
    ]

    @pytest.mark.parametrize("name", EXPECTED_MEMBERS)
    def test_member_exists(self, name):
        member = ViolationType[name]
        assert member.name == name

    def test_member_count(self):
        assert len(ViolationType) == len(self.EXPECTED_MEMBERS)

    def test_all_values_are_strings(self):
        for member in ViolationType:
            assert isinstance(member.value, str)


# ============================================================================
# 2. SeverityLevel ordering and round-trip
# ============================================================================

class TestSeverityLevel:
    """SeverityLevel ordering, to_float, from_float round-trip."""

    def test_ordering(self):
        levels = list(SeverityLevel)
        for i in range(len(levels) - 1):
            assert levels[i].value < levels[i + 1].value

    def test_to_float_range(self):
        for level in SeverityLevel:
            f = level.to_float()
            assert 0.0 <= f <= 1.0

    def test_to_float_monotone(self):
        prev = -1.0
        for level in SeverityLevel:
            f = level.to_float()
            assert f >= prev
            prev = f

    def test_none_is_zero(self):
        assert SeverityLevel.NONE.to_float() == 0.0

    def test_fatal_is_one(self):
        assert SeverityLevel.FATAL.to_float() == 1.0

    @pytest.mark.parametrize("level", list(SeverityLevel))
    def test_round_trip(self, level):
        f = level.to_float()
        recovered = SeverityLevel.from_float(f)
        assert recovered == level

    def test_from_float_clamps_negative(self):
        result = SeverityLevel.from_float(-0.5)
        assert result == SeverityLevel.NONE

    def test_from_float_clamps_above_one(self):
        result = SeverityLevel.from_float(1.5)
        assert result == SeverityLevel.FATAL

    def test_from_float_rounds(self):
        result = SeverityLevel.from_float(0.49)
        assert result in list(SeverityLevel)

    def test_members_exist(self):
        expected = {"NONE", "INFO", "WARNING", "ERROR", "CRITICAL", "FATAL"}
        actual = {m.name for m in SeverityLevel}
        assert actual == expected


# ============================================================================
# 3. ConstraintType enum
# ============================================================================

class TestConstraintType:

    EXPECTED = [
        "NOT_NULL", "UNIQUE", "PRIMARY_KEY", "FOREIGN_KEY", "CHECK",
        "EXCLUSION", "STATISTICAL", "DISTRIBUTION", "COMPLETENESS",
        "TIMELINESS", "CUSTOM",
    ]

    @pytest.mark.parametrize("name", EXPECTED)
    def test_member_exists(self, name):
        assert ConstraintType[name].name == name

    def test_count(self):
        assert len(ConstraintType) == len(self.EXPECTED)


# ============================================================================
# 4. DistributionSummary
# ============================================================================

class TestDistributionSummary:

    def test_creation_defaults(self):
        ds = DistributionSummary()
        assert ds.mean is None
        assert ds.stddev is None
        assert ds.null_fraction == 0.0
        assert ds.distinct_count is None

    def test_creation_with_values(self):
        ds = DistributionSummary(
            mean=10.0, stddev=2.0, min_val=0.0, max_val=20.0,
            null_fraction=0.05, distinct_count=100,
        )
        assert ds.mean == 10.0
        assert ds.stddev == 2.0
        assert ds.min_val == 0.0
        assert ds.max_val == 20.0
        assert ds.null_fraction == 0.05
        assert ds.distinct_count == 100

    def test_distance_to_self_is_zero(self):
        ds = DistributionSummary(mean=5.0, stddev=1.0, null_fraction=0.1, distinct_count=50)
        assert ds.distance_to(ds) == 0.0

    def test_distance_to_different(self):
        ds1 = DistributionSummary(mean=5.0, stddev=1.0, null_fraction=0.0, distinct_count=10)
        ds2 = DistributionSummary(mean=10.0, stddev=1.0, null_fraction=0.5, distinct_count=100)
        d = ds1.distance_to(ds2)
        assert d > 0.0

    def test_distance_bounded_by_one(self):
        ds1 = DistributionSummary(mean=0.0, stddev=0.01, null_fraction=0.0, distinct_count=1)
        ds2 = DistributionSummary(mean=1000.0, stddev=0.01, null_fraction=1.0, distinct_count=10000)
        d = ds1.distance_to(ds2)
        assert d <= 1.0

    def test_distance_symmetric_approximately(self):
        ds1 = DistributionSummary(mean=3.0, stddev=1.0, null_fraction=0.1, distinct_count=20)
        ds2 = DistributionSummary(mean=7.0, stddev=1.0, null_fraction=0.3, distinct_count=40)
        assert abs(ds1.distance_to(ds2) - ds2.distance_to(ds1)) < 1e-9

    def test_distance_with_none_means(self):
        ds1 = DistributionSummary(null_fraction=0.1)
        ds2 = DistributionSummary(null_fraction=0.3)
        d = ds1.distance_to(ds2)
        assert d >= 0.0

    def test_frozen(self):
        ds = DistributionSummary(mean=1.0)
        with pytest.raises(AttributeError):
            ds.mean = 2.0  # type: ignore[misc]

    def test_histogram_buckets(self):
        ds = DistributionSummary(histogram_buckets=(0.0, 1.0, 2.0, 3.0))
        assert ds.histogram_buckets == (0.0, 1.0, 2.0, 3.0)


# ============================================================================
# 5. QualityState
# ============================================================================

class TestQualityState:

    def _make_violation(self, cid="c1", severity=SeverityLevel.ERROR):
        return QualityViolation(
            constraint_id=cid, severity=severity, affected_tuples=5,
            violation_type=ViolationType.NULL_IN_NON_NULL, columns=("col_a",),
        )

    def test_fresh_state(self):
        s = QualityState()
        assert s.overall_score == 1.0
        assert s.violation_count() == 0
        assert not s.has_violations()
        assert s.max_severity() == SeverityLevel.NONE

    def test_add_violation(self):
        s = QualityState()
        v = self._make_violation()
        s.add_violation(v)
        assert s.violation_count() == 1
        assert s.has_violations()
        assert s.overall_score < 1.0

    def test_remove_violations(self):
        s = QualityState()
        v = self._make_violation()
        s.add_violation(v)
        removed = s.remove_violations("c1")
        assert len(removed) == 1
        assert s.violation_count() == 0
        assert s.overall_score == 1.0

    def test_remove_violations_returns_empty_for_unknown(self):
        s = QualityState()
        removed = s.remove_violations("nonexistent")
        assert removed == []

    def test_set_constraint_status(self):
        s = QualityState()
        cs = ConstraintStatus(
            constraint_id="pk_id", constraint_type=ConstraintType.PRIMARY_KEY,
            is_active=True, is_satisfied=True, columns=("id",),
        )
        s.set_constraint_status("pk_id", cs)
        assert "pk_id" in s.constraint_statuses
        assert s.constraint_statuses["pk_id"].constraint_type == ConstraintType.PRIMARY_KEY

    def test_remove_constraint(self):
        s = QualityState()
        v = self._make_violation("c2")
        s.add_violation(v)
        s.set_constraint_status("c2", ConstraintStatus(
            constraint_id="c2", constraint_type=ConstraintType.CHECK,
        ))
        s.remove_constraint("c2")
        assert "c2" not in s.constraint_statuses
        assert s.violation_count() == 0

    def test_max_severity(self):
        s = QualityState()
        s.add_violation(self._make_violation("c1", SeverityLevel.WARNING))
        s.add_violation(self._make_violation("c2", SeverityLevel.CRITICAL))
        assert s.max_severity() == SeverityLevel.CRITICAL

    def test_copy_independence(self):
        s = QualityState()
        s.add_violation(self._make_violation())
        s2 = s.copy()
        s2.remove_violations("c1")
        assert s.violation_count() == 1
        assert s2.violation_count() == 0

    def test_multiple_violations_same_constraint(self):
        s = QualityState()
        s.add_violation(self._make_violation("c1", SeverityLevel.WARNING))
        s.add_violation(self._make_violation("c1", SeverityLevel.ERROR))
        assert s.violation_count() == 2

    def test_score_decreases_with_severity(self):
        s1 = QualityState()
        s1.add_violation(self._make_violation("c1", SeverityLevel.INFO))
        s2 = QualityState()
        s2.add_violation(self._make_violation("c1", SeverityLevel.FATAL))
        assert s1.overall_score >= s2.overall_score

    def test_column_distributions(self):
        s = QualityState()
        ds = DistributionSummary(mean=5.0, stddev=1.0)
        s.column_distributions["col_x"] = ds
        assert s.column_distributions["col_x"].mean == 5.0


# ============================================================================
# 6. QualityOperation subclasses
# ============================================================================

class TestQualityViolation:

    def _make(self, **kw):
        defaults = dict(
            constraint_id="nn_email", severity=SeverityLevel.ERROR,
            affected_tuples=10, violation_type=ViolationType.NULL_IN_NON_NULL,
            columns=("email",),
        )
        defaults.update(kw)
        return QualityViolation(**defaults)

    def test_creation(self):
        v = self._make()
        assert v.constraint_id == "nn_email"
        assert v.severity == SeverityLevel.ERROR

    def test_inverse_is_improvement(self):
        v = self._make()
        inv = v.inverse()
        assert isinstance(inv, QualityImprovement)
        assert inv.constraint_id == v.constraint_id
        assert inv.old_severity == v.severity
        assert inv.new_severity == SeverityLevel.NONE
        assert inv.fixed_tuples == v.affected_tuples

    def test_severity_score_in_range(self):
        v = self._make()
        s = v.severity_score()
        assert 0.0 <= s <= 1.0

    def test_severity_score_zero_tuples(self):
        v = self._make(affected_tuples=0)
        s = v.severity_score()
        # base * (0.5 + 0.5*0) = base*0.5; still positive for non-NONE severity
        assert 0.0 <= s <= 1.0

    def test_affected_constraints(self):
        v = self._make()
        assert v.affected_constraints() == {"nn_email"}

    def test_affected_columns(self):
        v = self._make()
        assert v.affected_columns() == {"email"}

    def test_apply_adds_violation(self):
        v = self._make()
        s = QualityState()
        s2 = v.apply(s)
        assert s2.violation_count() == 1
        assert s.violation_count() == 0

    def test_dominates_same_constraint_higher_severity(self):
        v1 = self._make(severity=SeverityLevel.CRITICAL, affected_tuples=100)
        v2 = self._make(severity=SeverityLevel.WARNING, affected_tuples=10)
        assert v1.dominates(v2)
        assert not v2.dominates(v1)

    def test_dominates_different_constraint(self):
        v1 = self._make(constraint_id="a", severity=SeverityLevel.FATAL)
        v2 = self._make(constraint_id="b", severity=SeverityLevel.INFO)
        assert not v1.dominates(v2)

    def test_dominates_non_violation(self):
        v = self._make()
        imp = QualityImprovement(
            constraint_id="nn_email", old_severity=SeverityLevel.ERROR,
            new_severity=SeverityLevel.NONE, fixed_tuples=10, columns=("email",),
        )
        assert not v.dominates(imp)

    def test_equality(self):
        v1 = self._make()
        v2 = self._make()
        assert v1 == v2
        assert hash(v1) == hash(v2)

    def test_to_dict(self):
        v = self._make(message="test msg")
        d = v.to_dict()
        assert d["op"] == "QUALITY_VIOLATION"
        assert d["constraint_id"] == "nn_email"
        assert d["severity"] == "ERROR"
        assert d["message"] == "test msg"


class TestQualityImprovement:

    def _make(self, **kw):
        defaults = dict(
            constraint_id="nn_email", old_severity=SeverityLevel.ERROR,
            new_severity=SeverityLevel.NONE, fixed_tuples=10, columns=("email",),
        )
        defaults.update(kw)
        return QualityImprovement(**defaults)

    def test_inverse_is_violation(self):
        imp = self._make()
        inv = imp.inverse()
        assert isinstance(inv, QualityViolation)
        assert inv.severity == imp.old_severity

    def test_severity_score_is_negative(self):
        imp = self._make()
        assert imp.severity_score() < 0

    def test_severity_score_no_change(self):
        imp = self._make(old_severity=SeverityLevel.ERROR, new_severity=SeverityLevel.ERROR)
        assert imp.severity_score() == 0.0

    def test_apply_removes_violations(self):
        v = QualityViolation(
            constraint_id="nn_email", severity=SeverityLevel.ERROR,
            affected_tuples=10, violation_type=ViolationType.NULL_IN_NON_NULL,
            columns=("email",),
        )
        state = v.apply(QualityState())
        imp = self._make()
        state2 = imp.apply(state)
        assert state2.violation_count() == 0
        assert state2.overall_score == 1.0

    def test_dominates(self):
        i1 = self._make(new_severity=SeverityLevel.NONE, fixed_tuples=100)
        i2 = self._make(new_severity=SeverityLevel.WARNING, fixed_tuples=5)
        assert i1.dominates(i2)
        assert not i2.dominates(i1)


class TestConstraintAdded:

    def _make(self, **kw):
        defaults = dict(
            constraint_id="pk_users", constraint_type=ConstraintType.PRIMARY_KEY,
            columns=("id",),
        )
        defaults.update(kw)
        return ConstraintAdded(**defaults)

    def test_inverse_is_removed(self):
        ca = self._make()
        inv = ca.inverse()
        assert isinstance(inv, ConstraintRemoved)
        assert inv.constraint_id == "pk_users"

    def test_severity_score_zero(self):
        assert self._make().severity_score() == 0.0

    def test_apply_sets_status(self):
        ca = self._make()
        s = ca.apply(QualityState())
        assert "pk_users" in s.constraint_statuses
        st = s.constraint_statuses["pk_users"]
        assert st.is_active
        assert st.is_satisfied

    def test_dominates_always_false(self):
        ca = self._make()
        assert not ca.dominates(ca)

    def test_affected_constraints(self):
        assert self._make().affected_constraints() == {"pk_users"}

    def test_affected_columns(self):
        assert self._make().affected_columns() == {"id"}


class TestConstraintRemoved:

    def _make(self, **kw):
        defaults = dict(constraint_id="pk_users", reason="testing")
        defaults.update(kw)
        return ConstraintRemoved(**defaults)

    def test_inverse_is_added(self):
        cr = self._make(_preserved_type=ConstraintType.PRIMARY_KEY,
                        _preserved_columns=("id",))
        inv = cr.inverse()
        assert isinstance(inv, ConstraintAdded)
        assert inv.constraint_type == ConstraintType.PRIMARY_KEY
        assert inv.columns == ("id",)

    def test_inverse_uses_defaults_when_no_preserved(self):
        cr = self._make()
        inv = cr.inverse()
        assert isinstance(inv, ConstraintAdded)
        assert inv.constraint_type == ConstraintType.CUSTOM

    def test_severity_score(self):
        assert self._make().severity_score() == 0.1

    def test_apply_removes_constraint(self):
        s = QualityState()
        s.set_constraint_status("pk_users", ConstraintStatus(
            constraint_id="pk_users", constraint_type=ConstraintType.PRIMARY_KEY,
        ))
        cr = self._make()
        s2 = cr.apply(s)
        assert "pk_users" not in s2.constraint_statuses

    def test_dominates_always_false(self):
        cr = self._make()
        assert not cr.dominates(cr)


class TestDistributionShift:

    def _make(self, **kw):
        defaults = dict(
            column="age",
            old_dist=DistributionSummary(mean=30.0, stddev=5.0),
            new_dist=DistributionSummary(mean=50.0, stddev=10.0),
            psi_score=0.15, ks_statistic=0.08,
        )
        defaults.update(kw)
        return DistributionShift(**defaults)

    def test_inverse_swaps_dists(self):
        ds = self._make()
        inv = ds.inverse()
        assert isinstance(inv, DistributionShift)
        assert inv.old_dist == ds.new_dist
        assert inv.new_dist == ds.old_dist

    def test_severity_score_in_range(self):
        ds = self._make()
        s = ds.severity_score()
        assert 0.0 <= s <= 1.0

    def test_severity_score_zero(self):
        ds = self._make(psi_score=0.0, ks_statistic=0.0)
        assert ds.severity_score() == 0.0

    def test_severity_score_high(self):
        ds = self._make(psi_score=0.5, ks_statistic=0.2)
        assert ds.severity_score() > 0.5

    def test_affected_constraints(self):
        ds = self._make()
        assert ds.affected_constraints() == {"dist_age"}

    def test_affected_columns(self):
        ds = self._make()
        assert ds.affected_columns() == {"age"}

    def test_apply_updates_distribution(self):
        ds = self._make()
        state = QualityState()
        s2 = ds.apply(state)
        assert s2.column_distributions["age"] == ds.new_dist

    def test_dominates_same_column_higher_severity(self):
        d1 = self._make(psi_score=0.5, ks_statistic=0.3)
        d2 = self._make(psi_score=0.1, ks_statistic=0.05)
        assert d1.dominates(d2)

    def test_dominates_different_column(self):
        d1 = self._make(column="age", psi_score=0.5)
        d2 = self._make(column="salary", psi_score=0.1)
        assert not d1.dominates(d2)


# ============================================================================
# 7. QualityDelta lattice operations
# ============================================================================

class TestQualityDeltaLattice:
    """Join (⊔) and meet (⊓) operations."""

    def _v(self, cid, severity=SeverityLevel.ERROR, tuples=5):
        return QualityViolation(
            constraint_id=cid, severity=severity, affected_tuples=tuples,
            violation_type=ViolationType.NULL_IN_NON_NULL, columns=("c",),
        )

    def test_join_with_bottom(self):
        a = QualityDelta.from_operation(self._v("c1"))
        b = QualityDelta.bottom()
        assert a.join(b) == a
        assert b.join(a) == a

    def test_join_with_top(self):
        a = QualityDelta.from_operation(self._v("c1"))
        t = QualityDelta.top()
        assert a.join(t) == t
        assert t.join(a) == t

    def test_join_keeps_more_severe(self):
        v_warn = self._v("c1", SeverityLevel.WARNING, 5)
        v_err = self._v("c1", SeverityLevel.ERROR, 10)
        a = QualityDelta.from_operation(v_warn)
        b = QualityDelta.from_operation(v_err)
        joined = a.join(b)
        violations = joined.get_violations()
        assert len(violations) == 1
        assert violations[0].severity == SeverityLevel.ERROR

    def test_join_different_constraints(self):
        a = QualityDelta.from_operation(self._v("c1"))
        b = QualityDelta.from_operation(self._v("c2"))
        joined = a.join(b)
        assert joined.operation_count() == 2

    def test_meet_with_bottom(self):
        a = QualityDelta.from_operation(self._v("c1"))
        b = QualityDelta.bottom()
        assert a.meet(b) == b
        assert b.meet(a) == b

    def test_meet_with_top(self):
        a = QualityDelta.from_operation(self._v("c1"))
        t = QualityDelta.top()
        assert a.meet(t) == a
        assert t.meet(a) == a

    def test_meet_keeps_less_severe(self):
        v_warn = self._v("c1", SeverityLevel.WARNING, 5)
        v_err = self._v("c1", SeverityLevel.ERROR, 10)
        a = QualityDelta.from_operation(v_warn)
        b = QualityDelta.from_operation(v_err)
        met = a.meet(b)
        violations = met.get_violations()
        assert len(violations) == 1
        assert violations[0].severity == SeverityLevel.WARNING

    def test_meet_disjoint_constraints(self):
        a = QualityDelta.from_operation(self._v("c1"))
        b = QualityDelta.from_operation(self._v("c2"))
        met = a.meet(b)
        assert met.is_bottom()


# ============================================================================
# 8. Lattice algebraic laws
# ============================================================================

class TestLatticeLaws:

    def _v(self, cid, severity=SeverityLevel.ERROR, tuples=5):
        return QualityViolation(
            constraint_id=cid, severity=severity, affected_tuples=tuples,
            violation_type=ViolationType.CHECK_VIOLATION, columns=("x",),
        )

    @pytest.fixture()
    def a(self):
        return QualityDelta.from_operation(self._v("c1", SeverityLevel.WARNING, 3))

    @pytest.fixture()
    def b(self):
        return QualityDelta.from_operation(self._v("c1", SeverityLevel.ERROR, 10))

    @pytest.fixture()
    def c(self):
        return QualityDelta.from_operation(self._v("c2", SeverityLevel.CRITICAL, 1))

    # --- Idempotent ---
    def test_join_idempotent(self, a):
        assert a.join(a) == a

    def test_meet_idempotent(self, a):
        assert a.meet(a) == a

    # --- Commutative ---
    def test_join_commutative(self, a, b):
        assert a.join(b) == b.join(a)

    def test_meet_commutative(self, a, b):
        assert a.meet(b) == b.meet(a)

    # --- Associative ---
    def test_join_associative(self, a, b, c):
        assert a.join(b).join(c) == a.join(b.join(c))

    def test_meet_associative(self, a, b, c):
        assert a.meet(b).meet(c) == a.meet(b.meet(c))

    # --- Absorption ---
    def test_absorption_join_meet(self, a, b):
        """a ⊔ (a ⊓ b) = a"""
        assert a.join(a.meet(b)) == a

    def test_absorption_meet_join(self, a, b):
        """a ⊓ (a ⊔ b) = a"""
        assert a.meet(a.join(b)) == a


# ============================================================================
# 9. Bottom/Top identity elements
# ============================================================================

class TestBottomTopElements:

    def _v(self, cid="c1"):
        return QualityViolation(
            constraint_id=cid, severity=SeverityLevel.ERROR, affected_tuples=5,
            violation_type=ViolationType.NULL_IN_NON_NULL, columns=("x",),
        )

    def test_bottom_is_bottom(self):
        assert QualityDelta.bottom().is_bottom()

    def test_top_is_top(self):
        assert QualityDelta.top().is_top()

    def test_bottom_not_top(self):
        assert not QualityDelta.bottom().is_top()

    def test_top_not_bottom(self):
        assert not QualityDelta.top().is_bottom()

    def test_bottom_identity_for_join(self):
        a = QualityDelta.from_operation(self._v())
        assert a.join(QualityDelta.bottom()) == a

    def test_top_identity_for_meet(self):
        a = QualityDelta.from_operation(self._v())
        assert a.meet(QualityDelta.top()) == a

    def test_bottom_annihilator_for_meet(self):
        a = QualityDelta.from_operation(self._v())
        assert a.meet(QualityDelta.bottom()) == QualityDelta.bottom()

    def test_top_annihilator_for_join(self):
        a = QualityDelta.from_operation(self._v())
        assert a.join(QualityDelta.top()) == QualityDelta.top()

    def test_bottom_severity_zero(self):
        assert QualityDelta.bottom().severity() == 0.0

    def test_top_severity_one(self):
        assert QualityDelta.top().severity() == 1.0


# ============================================================================
# 10. Severity computation
# ============================================================================

class TestSeverityComputation:

    def test_single_violation_severity(self):
        v = QualityViolation(
            constraint_id="c1", severity=SeverityLevel.ERROR,
            affected_tuples=100, violation_type=ViolationType.CHECK_VIOLATION,
            columns=("x",),
        )
        d = QualityDelta.from_operation(v)
        s = d.severity()
        assert 0.0 < s <= 1.0

    def test_multiple_violations_increase_severity(self):
        v1 = QualityViolation(
            constraint_id="c1", severity=SeverityLevel.WARNING,
            affected_tuples=10, violation_type=ViolationType.CHECK_VIOLATION,
            columns=("x",),
        )
        v2 = QualityViolation(
            constraint_id="c2", severity=SeverityLevel.WARNING,
            affected_tuples=10, violation_type=ViolationType.NULL_IN_NON_NULL,
            columns=("y",),
        )
        single = QualityDelta.from_operation(v1)
        multi = QualityDelta.from_operations([v1, v2])
        assert multi.severity() >= single.severity()

    def test_improvement_reduces_severity(self):
        v = QualityViolation(
            constraint_id="c1", severity=SeverityLevel.ERROR,
            affected_tuples=10, violation_type=ViolationType.CHECK_VIOLATION,
            columns=("x",),
        )
        imp = QualityImprovement(
            constraint_id="c2", old_severity=SeverityLevel.ERROR,
            new_severity=SeverityLevel.NONE, fixed_tuples=10, columns=("y",),
        )
        d_v = QualityDelta.from_operation(v)
        d_both = QualityDelta.from_operations([v, imp])
        assert d_both.severity() <= d_v.severity()


# ============================================================================
# 11. QualityDelta compose and inverse
# ============================================================================

class TestQualityDeltaComposeInverse:

    def _v(self, cid, severity=SeverityLevel.ERROR, tuples=5):
        return QualityViolation(
            constraint_id=cid, severity=severity, affected_tuples=tuples,
            violation_type=ViolationType.NULL_IN_NON_NULL, columns=("x",),
        )

    def test_compose_concatenates(self):
        a = QualityDelta.from_operation(self._v("c1"))
        b = QualityDelta.from_operation(self._v("c2"))
        if not hasattr(a, "compose"):
            pytest.skip("compose not implemented")
        composed = a.compose(b)
        assert composed.operation_count() >= 2

    def test_compose_with_bottom(self):
        a = QualityDelta.from_operation(self._v("c1"))
        bot = QualityDelta.bottom()
        if not hasattr(a, "compose"):
            pytest.skip("compose not implemented")
        assert a.compose(bot).operation_count() == a.operation_count()
        assert bot.compose(a).operation_count() == a.operation_count()

    def test_inverse_round_trip(self):
        v = self._v("c1")
        d = QualityDelta.from_operation(v)
        if not hasattr(d, "inverse"):
            pytest.skip("inverse not implemented")
        inv = d.inverse()
        assert inv.operation_count() == d.operation_count()
        inv_ops = inv.operations
        assert all(isinstance(op, QualityImprovement) for op in inv_ops)

    def test_inverse_of_bottom(self):
        bot = QualityDelta.bottom()
        if not hasattr(bot, "inverse"):
            pytest.skip("inverse not implemented")
        assert bot.inverse().is_bottom()


# ============================================================================
# 12. apply_to_quality_state
# ============================================================================

class TestApplyToQualityState:

    def test_empty_delta_preserves_state(self):
        s = QualityState()
        d = QualityDelta.bottom()
        s2 = d.apply_to_quality_state(s)
        assert s2.overall_score == s.overall_score
        assert s2.violation_count() == s.violation_count()

    def test_violation_degrades_state(self):
        v = QualityViolation(
            constraint_id="c1", severity=SeverityLevel.CRITICAL,
            affected_tuples=100, violation_type=ViolationType.FOREIGN_KEY_VIOLATION,
            columns=("fk_col",),
        )
        d = QualityDelta.from_operation(v)
        s = QualityState()
        s2 = d.apply_to_quality_state(s)
        assert s2.overall_score < s.overall_score
        assert s2.violation_count() == 1

    def test_improvement_after_violation(self):
        v = QualityViolation(
            constraint_id="c1", severity=SeverityLevel.ERROR,
            affected_tuples=10, violation_type=ViolationType.NULL_IN_NON_NULL,
            columns=("email",),
        )
        imp = QualityImprovement(
            constraint_id="c1", old_severity=SeverityLevel.ERROR,
            new_severity=SeverityLevel.NONE, fixed_tuples=10, columns=("email",),
        )
        d = QualityDelta.from_operations([v, imp])
        s = QualityState()
        s2 = d.apply_to_quality_state(s)
        assert s2.violation_count() == 0
        assert s2.overall_score == 1.0

    def test_constraint_added(self):
        ca = ConstraintAdded(
            constraint_id="uq_email", constraint_type=ConstraintType.UNIQUE,
            columns=("email",),
        )
        d = QualityDelta.from_operation(ca)
        s = d.apply_to_quality_state(QualityState())
        assert "uq_email" in s.constraint_statuses

    def test_constraint_removed(self):
        s = QualityState()
        s.set_constraint_status("uq_email", ConstraintStatus(
            constraint_id="uq_email", constraint_type=ConstraintType.UNIQUE,
        ))
        cr = ConstraintRemoved(constraint_id="uq_email", reason="dropped")
        d = QualityDelta.from_operation(cr)
        s2 = d.apply_to_quality_state(s)
        assert "uq_email" not in s2.constraint_statuses

    def test_distribution_shift(self):
        old = DistributionSummary(mean=10.0, stddev=2.0)
        new = DistributionSummary(mean=50.0, stddev=10.0)
        ds = DistributionShift(column="age", old_dist=old, new_dist=new,
                               psi_score=0.3, ks_statistic=0.15)
        d = QualityDelta.from_operation(ds)
        s2 = d.apply_to_quality_state(QualityState())
        assert s2.column_distributions["age"] == new

    def test_original_state_unchanged(self):
        s = QualityState()
        v = QualityViolation(
            constraint_id="c1", severity=SeverityLevel.ERROR,
            affected_tuples=5, violation_type=ViolationType.CHECK_VIOLATION,
            columns=("x",),
        )
        d = QualityDelta.from_operation(v)
        _ = d.apply_to_quality_state(s)
        assert s.violation_count() == 0


# ============================================================================
# 13. Edge cases
# ============================================================================

class TestEdgeCases:

    def test_empty_delta(self):
        d = QualityDelta.bottom()
        assert d.is_bottom()
        assert d.operation_count() == 0
        assert d.severity() == 0.0
        assert not d.has_violations()
        assert not d.has_improvements()
        assert d.violation_count() == 0

    def test_single_violation(self):
        v = QualityViolation(
            constraint_id="c1", severity=SeverityLevel.INFO,
            affected_tuples=1, violation_type=ViolationType.PATTERN_VIOLATION,
            columns=("name",),
        )
        d = QualityDelta.from_operation(v)
        assert d.operation_count() == 1
        assert d.has_violations()
        assert d.violation_count() == 1
        assert d.severity() > 0.0

    def test_cascading_violations(self):
        ops = [
            QualityViolation(
                constraint_id=f"c{i}", severity=SeverityLevel.ERROR,
                affected_tuples=10, violation_type=ViolationType.CHECK_VIOLATION,
                columns=("x",),
            )
            for i in range(5)
        ]
        d = QualityDelta.from_operations(ops)
        assert d.violation_count() == 5
        assert d.severity() > 0.0

    def test_bool_empty_is_false(self):
        assert not bool(QualityDelta.bottom())

    def test_bool_non_empty_is_true(self):
        v = QualityViolation(
            constraint_id="c1", severity=SeverityLevel.ERROR,
            affected_tuples=1, violation_type=ViolationType.CHECK_VIOLATION,
            columns=("x",),
        )
        assert bool(QualityDelta.from_operation(v))

    def test_len(self):
        v = QualityViolation(
            constraint_id="c1", severity=SeverityLevel.ERROR,
            affected_tuples=1, violation_type=ViolationType.CHECK_VIOLATION,
            columns=("x",),
        )
        d = QualityDelta.from_operations([v, v])
        assert len(d) == 2

    def test_iter(self):
        v = QualityViolation(
            constraint_id="c1", severity=SeverityLevel.ERROR,
            affected_tuples=1, violation_type=ViolationType.CHECK_VIOLATION,
            columns=("x",),
        )
        d = QualityDelta.from_operation(v)
        ops = list(d)
        assert len(ops) == 1

    def test_partial_order_le(self):
        bot = QualityDelta.bottom()
        a = QualityDelta.from_operation(QualityViolation(
            constraint_id="c1", severity=SeverityLevel.ERROR,
            affected_tuples=10, violation_type=ViolationType.CHECK_VIOLATION,
            columns=("x",),
        ))
        assert bot <= a

    def test_equality_bottom(self):
        assert QualityDelta.bottom() == QualityDelta.bottom()

    def test_equality_top(self):
        assert QualityDelta.top() == QualityDelta.top()

    def test_filter_by_constraint(self):
        v1 = QualityViolation(
            constraint_id="c1", severity=SeverityLevel.ERROR,
            affected_tuples=5, violation_type=ViolationType.CHECK_VIOLATION,
            columns=("x",),
        )
        v2 = QualityViolation(
            constraint_id="c2", severity=SeverityLevel.WARNING,
            affected_tuples=3, violation_type=ViolationType.NULL_IN_NON_NULL,
            columns=("y",),
        )
        d = QualityDelta.from_operations([v1, v2])
        filtered = d.filter_by_constraint("c1")
        assert filtered.operation_count() == 1

    def test_filter_by_column(self):
        v1 = QualityViolation(
            constraint_id="c1", severity=SeverityLevel.ERROR,
            affected_tuples=5, violation_type=ViolationType.CHECK_VIOLATION,
            columns=("x",),
        )
        v2 = QualityViolation(
            constraint_id="c2", severity=SeverityLevel.WARNING,
            affected_tuples=3, violation_type=ViolationType.NULL_IN_NON_NULL,
            columns=("y",),
        )
        d = QualityDelta.from_operations([v1, v2])
        filtered = d.filter_by_column("x")
        assert filtered.operation_count() == 1

    def test_rename_column(self):
        v = QualityViolation(
            constraint_id="c1", severity=SeverityLevel.ERROR,
            affected_tuples=5, violation_type=ViolationType.CHECK_VIOLATION,
            columns=("old_col",),
        )
        d = QualityDelta.from_operation(v)
        renamed = d.rename_column("old_col", "new_col")
        assert renamed.get_violations()[0].columns == ("new_col",)

    def test_drop_column(self):
        v1 = QualityViolation(
            constraint_id="c1", severity=SeverityLevel.ERROR,
            affected_tuples=5, violation_type=ViolationType.CHECK_VIOLATION,
            columns=("x",),
        )
        v2 = QualityViolation(
            constraint_id="c2", severity=SeverityLevel.WARNING,
            affected_tuples=3, violation_type=ViolationType.NULL_IN_NON_NULL,
            columns=("y",),
        )
        d = QualityDelta.from_operations([v1, v2])
        dropped = d.drop_column("x")
        assert dropped.operation_count() == 1


# ============================================================================
# 14. Serialization round-trip
# ============================================================================

class TestSerialization:

    def test_violation_round_trip(self):
        v = QualityViolation(
            constraint_id="c1", severity=SeverityLevel.ERROR,
            affected_tuples=10, violation_type=ViolationType.NULL_IN_NON_NULL,
            columns=("email",), message="null found",
        )
        d = QualityDelta.from_operation(v)
        data = d.to_dict()
        recovered = QualityDelta.from_dict(data)
        assert recovered.operation_count() == 1
        rv = recovered.get_violations()[0]
        assert rv.constraint_id == "c1"
        assert rv.severity == SeverityLevel.ERROR
        assert rv.violation_type == ViolationType.NULL_IN_NON_NULL
        assert rv.columns == ("email",)
        assert rv.message == "null found"

    def test_improvement_round_trip(self):
        imp = QualityImprovement(
            constraint_id="c1", old_severity=SeverityLevel.ERROR,
            new_severity=SeverityLevel.NONE, fixed_tuples=10,
            columns=("email",), message="fixed",
        )
        d = QualityDelta.from_operation(imp)
        data = d.to_dict()
        recovered = QualityDelta.from_dict(data)
        ri = recovered.get_improvements()[0]
        assert ri.constraint_id == "c1"
        assert ri.old_severity == SeverityLevel.ERROR
        assert ri.new_severity == SeverityLevel.NONE

    def test_constraint_added_round_trip(self):
        ca = ConstraintAdded(
            constraint_id="uq_email", constraint_type=ConstraintType.UNIQUE,
            predicate=None, columns=("email",),
        )
        d = QualityDelta.from_operation(ca)
        data = d.to_dict()
        recovered = QualityDelta.from_dict(data)
        rc = recovered.get_constraint_additions()[0]
        assert rc.constraint_id == "uq_email"
        assert rc.constraint_type == ConstraintType.UNIQUE

    def test_constraint_removed_round_trip(self):
        cr = ConstraintRemoved(constraint_id="uq_email", reason="dropped")
        d = QualityDelta.from_operation(cr)
        data = d.to_dict()
        recovered = QualityDelta.from_dict(data)
        rc = recovered.get_constraint_removals()[0]
        assert rc.constraint_id == "uq_email"
        assert rc.reason == "dropped"

    def test_distribution_shift_round_trip(self):
        ds = DistributionShift(
            column="age",
            old_dist=DistributionSummary(mean=30.0),
            new_dist=DistributionSummary(mean=50.0),
            psi_score=0.2, ks_statistic=0.1,
        )
        d = QualityDelta.from_operation(ds)
        data = d.to_dict()
        recovered = QualityDelta.from_dict(data)
        rd = recovered.get_distribution_shifts()[0]
        assert rd.column == "age"
        assert rd.psi_score == 0.2
        assert rd.ks_statistic == 0.1

    def test_mixed_ops_round_trip(self):
        ops = [
            QualityViolation(
                constraint_id="c1", severity=SeverityLevel.WARNING,
                affected_tuples=5, violation_type=ViolationType.CHECK_VIOLATION,
                columns=("x",),
            ),
            QualityImprovement(
                constraint_id="c2", old_severity=SeverityLevel.ERROR,
                new_severity=SeverityLevel.NONE, fixed_tuples=10,
                columns=("y",),
            ),
            ConstraintAdded(
                constraint_id="pk_id", constraint_type=ConstraintType.PRIMARY_KEY,
                columns=("id",),
            ),
            ConstraintRemoved(constraint_id="old_c", reason="migrated"),
            DistributionShift(
                column="age",
                old_dist=DistributionSummary(),
                new_dist=DistributionSummary(mean=10.0),
                psi_score=0.05, ks_statistic=0.02,
            ),
        ]
        d = QualityDelta.from_operations(ops)
        data = d.to_dict()
        recovered = QualityDelta.from_dict(data)
        assert recovered.operation_count() == 5

    def test_empty_delta_round_trip(self):
        d = QualityDelta.bottom()
        data = d.to_dict()
        recovered = QualityDelta.from_dict(data)
        assert recovered.is_bottom()

    def test_to_dict_structure(self):
        v = QualityViolation(
            constraint_id="c1", severity=SeverityLevel.ERROR,
            affected_tuples=5, violation_type=ViolationType.CHECK_VIOLATION,
            columns=("x",),
        )
        d = QualityDelta.from_operation(v)
        data = d.to_dict()
        assert "operations" in data
        assert isinstance(data["operations"], list)
        assert len(data["operations"]) == 1
        assert data["operations"][0]["op"] == "QUALITY_VIOLATION"

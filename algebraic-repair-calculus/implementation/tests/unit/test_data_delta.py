"""
Tests for arc.algebra.data_delta

Covers TypedTuple, MultiSet, DataOperations (Insert/Delete/Update),
DataDelta algebraic laws (group: associativity, identity, inverse, closure),
normalization, apply_to_data, compression, diff, and serialization.
"""

import pytest

try:
    from arc.algebra.data_delta import (
        TypedTuple,
        MultiSet,
        DataOperation,
        InsertOp,
        DeleteOp,
        UpdateOp,
        DataDelta,
    )

    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(not HAS_MODULE, reason="module not available")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_tuple():
    return TypedTuple.from_dict({"id": 1, "name": "Alice", "age": 30})


@pytest.fixture
def another_tuple():
    return TypedTuple.from_dict({"id": 2, "name": "Bob", "age": 25})


@pytest.fixture
def sample_multiset(sample_tuple, another_tuple):
    return MultiSet.from_tuples([sample_tuple, another_tuple])


@pytest.fixture
def empty_multiset():
    return MultiSet.empty()


@pytest.fixture
def duplicate_multiset(sample_tuple):
    return MultiSet.from_tuples([sample_tuple, sample_tuple, sample_tuple])


@pytest.fixture
def sample_data(sample_tuple, another_tuple):
    """A multiset representing base data."""
    return MultiSet.from_tuples([sample_tuple, another_tuple])


# ===================================================================
# Section 1: TypedTuple — creation
# ===================================================================


class TestTypedTupleCreation:
    def test_from_dict(self):
        t = TypedTuple.from_dict({"x": 1, "y": "hello"})
        assert t.get("x") == 1
        assert t.get("y") == "hello"

    def test_from_row(self):
        t = TypedTuple.from_row(["a", "b", "c"], [10, 20, 30])
        assert t.get("a") == 10
        assert t.get("b") == 20
        assert t.get("c") == 30

    def test_columns_property(self):
        t = TypedTuple.from_dict({"x": 1, "y": 2})
        cols = t.columns
        assert "x" in cols
        assert "y" in cols

    def test_values_property(self):
        t = TypedTuple.from_dict({"a": 10})
        vals = t.values
        assert vals["a"] == 10

    def test_getitem(self):
        t = TypedTuple.from_dict({"key": "value"})
        assert t["key"] == "value"

    def test_contains(self):
        t = TypedTuple.from_dict({"x": 1})
        assert "x" in t
        assert "missing" not in t

    def test_empty_dict(self):
        t = TypedTuple.from_dict({})
        assert len(t.columns) == 0


# ===================================================================
# Section 2: TypedTuple — projection, extension, drop, rename
# ===================================================================


class TestTypedTupleTransforms:
    def test_project(self, sample_tuple):
        projected = sample_tuple.project(["id", "name"])
        assert "id" in projected.columns
        assert "name" in projected.columns
        assert "age" not in projected.columns

    def test_project_single(self, sample_tuple):
        projected = sample_tuple.project(["id"])
        assert projected.get("id") == 1
        assert len(projected.columns) == 1

    def test_extend(self, sample_tuple):
        extended = sample_tuple.extend("email", "alice@example.com")
        assert extended.get("email") == "alice@example.com"
        assert extended.get("id") == 1

    def test_drop(self, sample_tuple):
        dropped = sample_tuple.drop("age")
        assert "age" not in dropped.columns
        assert "id" in dropped.columns

    def test_rename(self, sample_tuple):
        renamed = sample_tuple.rename("name", "full_name")
        assert "full_name" in renamed.columns
        assert "name" not in renamed.columns
        assert renamed.get("full_name") == "Alice"

    def test_update_value(self, sample_tuple):
        updated = sample_tuple.update_value("age", 31)
        assert updated.get("age") == 31
        assert updated.get("name") == "Alice"

    def test_coerce_column(self, sample_tuple):
        coerced = sample_tuple.coerce_column("age", str)
        assert coerced.get("age") == "30"

    def test_merge(self):
        t1 = TypedTuple.from_dict({"a": 1})
        t2 = TypedTuple.from_dict({"b": 2})
        merged = t1.merge(t2)
        assert merged.get("a") == 1
        assert merged.get("b") == 2

    def test_project_empty(self, sample_tuple):
        projected = sample_tuple.project([])
        assert len(projected.columns) == 0


# ===================================================================
# Section 3: TypedTuple — equality and hashing
# ===================================================================


class TestTypedTupleEquality:
    def test_equal_tuples(self):
        t1 = TypedTuple.from_dict({"x": 1, "y": 2})
        t2 = TypedTuple.from_dict({"x": 1, "y": 2})
        assert t1 == t2

    def test_unequal_values(self):
        t1 = TypedTuple.from_dict({"x": 1})
        t2 = TypedTuple.from_dict({"x": 2})
        assert t1 != t2

    def test_unequal_columns(self):
        t1 = TypedTuple.from_dict({"x": 1})
        t2 = TypedTuple.from_dict({"y": 1})
        assert t1 != t2

    def test_hash_equal(self):
        t1 = TypedTuple.from_dict({"a": 10, "b": 20})
        t2 = TypedTuple.from_dict({"a": 10, "b": 20})
        assert hash(t1) == hash(t2)

    def test_hash_usable_in_set(self):
        t1 = TypedTuple.from_dict({"x": 1})
        t2 = TypedTuple.from_dict({"x": 1})
        t3 = TypedTuple.from_dict({"x": 2})
        s = {t1, t2, t3}
        assert len(s) == 2

    def test_hash_usable_as_dict_key(self):
        t = TypedTuple.from_dict({"k": "v"})
        d = {t: "found"}
        assert d[t] == "found"


# ===================================================================
# Section 4: MultiSet — basic operations
# ===================================================================


class TestMultiSetBasic:
    def test_empty(self):
        ms = MultiSet.empty()
        assert ms.is_empty()
        assert ms.cardinality() == 0

    def test_from_tuples(self, sample_tuple, another_tuple):
        ms = MultiSet.from_tuples([sample_tuple, another_tuple])
        assert ms.cardinality() == 2

    def test_from_dicts(self):
        ms = MultiSet.from_dicts([{"x": 1}, {"x": 2}, {"x": 1}])
        assert ms.cardinality() == 3

    def test_from_rows(self):
        ms = MultiSet.from_rows(["a", "b"], [[1, 2], [3, 4]])
        assert ms.cardinality() == 2

    def test_add(self, empty_multiset, sample_tuple):
        empty_multiset.add(sample_tuple)
        assert empty_multiset.contains(sample_tuple)
        assert empty_multiset.cardinality() == 1

    def test_remove(self, sample_multiset, sample_tuple):
        sample_multiset.remove(sample_tuple)
        assert sample_multiset.multiplicity(sample_tuple) == 0

    def test_contains(self, sample_multiset, sample_tuple):
        assert sample_multiset.contains(sample_tuple)

    def test_multiplicity(self, duplicate_multiset, sample_tuple):
        assert duplicate_multiset.multiplicity(sample_tuple) == 3

    def test_tuples(self, sample_multiset):
        tuples = list(sample_multiset.tuples())
        assert len(tuples) == 2

    def test_unique_tuples(self, duplicate_multiset):
        unique = list(duplicate_multiset.unique_tuples())
        assert len(unique) == 1

    def test_distinct_count(self, duplicate_multiset):
        assert duplicate_multiset.distinct_count() == 1

    def test_copy(self, sample_multiset, sample_tuple):
        cp = sample_multiset.copy()
        cp.remove(sample_tuple)
        assert sample_multiset.contains(sample_tuple)

    def test_to_dicts(self, sample_multiset):
        dicts = sample_multiset.to_dicts()
        assert isinstance(dicts, list)
        assert len(dicts) == 2


# ===================================================================
# Section 5: MultiSet — set operations
# ===================================================================


class TestMultiSetOperations:
    def test_union(self, sample_tuple, another_tuple):
        ms1 = MultiSet.from_tuples([sample_tuple])
        ms2 = MultiSet.from_tuples([another_tuple])
        union = ms1.union(ms2)
        assert union.cardinality() == 2

    def test_intersection(self, sample_tuple, another_tuple):
        ms1 = MultiSet.from_tuples([sample_tuple, another_tuple])
        ms2 = MultiSet.from_tuples([sample_tuple])
        inter = ms1.intersection(ms2)
        assert inter.contains(sample_tuple)
        assert not inter.contains(another_tuple) or inter.multiplicity(another_tuple) == 0

    def test_difference(self, sample_tuple, another_tuple):
        ms1 = MultiSet.from_tuples([sample_tuple, another_tuple])
        ms2 = MultiSet.from_tuples([sample_tuple])
        diff = ms1.difference(ms2)
        assert diff.contains(another_tuple)
        assert not diff.contains(sample_tuple) or diff.multiplicity(sample_tuple) == 0

    def test_sum(self, sample_tuple):
        ms1 = MultiSet.from_tuples([sample_tuple])
        ms2 = MultiSet.from_tuples([sample_tuple])
        total = ms1.sum(ms2)
        assert total.multiplicity(sample_tuple) == 2

    def test_union_with_empty(self, sample_multiset, empty_multiset):
        result = sample_multiset.union(empty_multiset)
        assert result.cardinality() == sample_multiset.cardinality()

    def test_intersection_with_empty(self, sample_multiset, empty_multiset):
        result = sample_multiset.intersection(empty_multiset)
        assert result.is_empty()

    def test_difference_with_self(self, sample_multiset):
        diff = sample_multiset.difference(sample_multiset)
        assert diff.is_empty() or diff.cardinality() == 0


# ===================================================================
# Section 6: MultiSet — projection, filter, map, distinct
# ===================================================================


class TestMultiSetTransforms:
    def test_project(self, sample_multiset):
        projected = sample_multiset.project(["id"])
        for t in projected.tuples():
            assert "id" in t.columns
            assert "name" not in t.columns

    def test_filter(self, sample_multiset):
        filtered = sample_multiset.filter(lambda t: t.get("id") == 1)
        assert filtered.cardinality() == 1

    def test_map_tuples(self, sample_multiset):
        mapped = sample_multiset.map_tuples(lambda t: t.update_value("age", t.get("age") + 1))
        for t in mapped.tuples():
            assert t.get("age") in [31, 26]

    def test_distinct(self, duplicate_multiset):
        dist = duplicate_multiset.distinct()
        assert dist.cardinality() == 1

    def test_columns(self, sample_multiset):
        cols = sample_multiset.columns()
        assert "id" in cols
        assert "name" in cols
        assert "age" in cols

    def test_elements_property(self, sample_multiset):
        elems = sample_multiset.elements
        assert elems is not None


# ===================================================================
# Section 7: MultiSet — edge cases
# ===================================================================


class TestMultiSetEdgeCases:
    def test_empty_set_operations(self):
        e1 = MultiSet.empty()
        e2 = MultiSet.empty()
        assert e1.union(e2).is_empty()
        assert e1.intersection(e2).is_empty()
        assert e1.difference(e2).is_empty()

    def test_single_element(self):
        t = TypedTuple.from_dict({"x": 42})
        ms = MultiSet.from_tuples([t])
        assert ms.cardinality() == 1
        assert ms.multiplicity(t) == 1

    def test_high_multiplicity(self):
        t = TypedTuple.from_dict({"v": 0})
        ms = MultiSet.from_tuples([t] * 100)
        assert ms.multiplicity(t) == 100
        assert ms.cardinality() == 100
        assert ms.distinct_count() == 1

    def test_remove_nonexistent(self):
        ms = MultiSet.empty()
        t = TypedTuple.from_dict({"x": 1})
        # Removing from empty should not crash or should raise
        try:
            ms.remove(t)
        except (KeyError, ValueError):
            pass  # acceptable

    def test_from_dicts_empty(self):
        ms = MultiSet.from_dicts([])
        assert ms.is_empty()


# ===================================================================
# Section 8: InsertOp
# ===================================================================


class TestInsertOp:
    def test_create(self, sample_tuple):
        ms = MultiSet.from_tuples([sample_tuple])
        op = InsertOp(tuples=ms)
        assert op.affected_rows_count() == 1

    def test_inverse_is_delete(self, sample_tuple):
        ms = MultiSet.from_tuples([sample_tuple])
        op = InsertOp(tuples=ms)
        inv = op.inverse()
        assert isinstance(inv, DeleteOp)

    def test_apply_adds_to_data(self, sample_tuple, empty_multiset):
        ms = MultiSet.from_tuples([sample_tuple])
        op = InsertOp(tuples=ms)
        result = op.apply(empty_multiset.copy())
        assert result.contains(sample_tuple)

    def test_is_zero_empty_insert(self):
        op = InsertOp(tuples=MultiSet.empty())
        assert op.is_zero() is True

    def test_is_zero_nonempty_insert(self, sample_tuple):
        op = InsertOp(tuples=MultiSet.from_tuples([sample_tuple]))
        assert op.is_zero() is False

    def test_affected_rows_count(self, sample_tuple, another_tuple):
        ms = MultiSet.from_tuples([sample_tuple, another_tuple])
        op = InsertOp(tuples=ms)
        assert op.affected_rows_count() == 2

    def test_multiple_inserts(self, sample_tuple):
        ms = MultiSet.from_tuples([sample_tuple, sample_tuple])
        op = InsertOp(tuples=ms)
        assert op.affected_rows_count() == 2


# ===================================================================
# Section 9: DeleteOp
# ===================================================================


class TestDeleteOp:
    def test_create(self, sample_tuple):
        ms = MultiSet.from_tuples([sample_tuple])
        op = DeleteOp(tuples=ms)
        assert op.affected_rows_count() == 1

    def test_inverse_is_insert(self, sample_tuple):
        ms = MultiSet.from_tuples([sample_tuple])
        op = DeleteOp(tuples=ms)
        inv = op.inverse()
        assert isinstance(inv, InsertOp)

    def test_apply_removes_from_data(self, sample_tuple, another_tuple):
        data = MultiSet.from_tuples([sample_tuple, another_tuple])
        ms = MultiSet.from_tuples([sample_tuple])
        op = DeleteOp(tuples=ms)
        result = op.apply(data.copy())
        assert not result.contains(sample_tuple) or result.multiplicity(sample_tuple) == 0
        assert result.contains(another_tuple)

    def test_is_zero_empty_delete(self):
        op = DeleteOp(tuples=MultiSet.empty())
        assert op.is_zero() is True

    def test_affected_rows(self, sample_tuple, another_tuple):
        ms = MultiSet.from_tuples([sample_tuple, another_tuple])
        op = DeleteOp(tuples=ms)
        assert op.affected_rows_count() == 2


# ===================================================================
# Section 10: UpdateOp
# ===================================================================


class TestUpdateOp:
    def test_create(self, sample_tuple):
        new_tuple = sample_tuple.update_value("age", 31)
        old_ms = MultiSet.from_tuples([sample_tuple])
        new_ms = MultiSet.from_tuples([new_tuple])
        op = UpdateOp(old_tuples=old_ms, new_tuples=new_ms)
        assert op.affected_rows_count() >= 1

    def test_to_delete_insert(self, sample_tuple):
        new_tuple = sample_tuple.update_value("name", "Alicia")
        old_ms = MultiSet.from_tuples([sample_tuple])
        new_ms = MultiSet.from_tuples([new_tuple])
        op = UpdateOp(old_tuples=old_ms, new_tuples=new_ms)
        delete_op, insert_op = op.to_delete_insert()
        assert isinstance(delete_op, DeleteOp)
        assert isinstance(insert_op, InsertOp)

    def test_changed_columns(self, sample_tuple):
        new_tuple = sample_tuple.update_value("age", 99)
        old_ms = MultiSet.from_tuples([sample_tuple])
        new_ms = MultiSet.from_tuples([new_tuple])
        op = UpdateOp(old_tuples=old_ms, new_tuples=new_ms)
        changed = op.changed_columns()
        assert "age" in changed

    def test_inverse(self, sample_tuple):
        new_tuple = sample_tuple.update_value("age", 31)
        old_ms = MultiSet.from_tuples([sample_tuple])
        new_ms = MultiSet.from_tuples([new_tuple])
        op = UpdateOp(old_tuples=old_ms, new_tuples=new_ms)
        inv = op.inverse()
        assert isinstance(inv, UpdateOp)


# ===================================================================
# Section 11: DataDelta — zero element
# ===================================================================


class TestDataDeltaZero:
    def test_zero_is_zero(self):
        delta = DataDelta.zero()
        assert delta.is_zero() is True

    def test_zero_operation_count(self):
        delta = DataDelta.zero()
        assert delta.operation_count() == 0

    def test_zero_affected_rows(self):
        delta = DataDelta.zero()
        assert delta.affected_rows_count() == 0

    def test_zero_net_row_change(self):
        delta = DataDelta.zero()
        assert delta.net_row_change() == 0

    def test_zero_apply_preserves_data(self, sample_data):
        delta = DataDelta.zero()
        result = delta.apply_to_data(sample_data.copy())
        assert result.cardinality() == sample_data.cardinality()


# ===================================================================
# Section 12: DataDelta — creation helpers
# ===================================================================


class TestDataDeltaCreation:
    def test_from_operation(self, sample_tuple):
        ms = MultiSet.from_tuples([sample_tuple])
        op = InsertOp(tuples=ms)
        delta = DataDelta.from_operation(op)
        assert delta.operation_count() == 1

    def test_from_operations(self, sample_tuple, another_tuple):
        ms1 = MultiSet.from_tuples([sample_tuple])
        ms2 = MultiSet.from_tuples([another_tuple])
        delta = DataDelta.from_operations([InsertOp(tuples=ms1), InsertOp(tuples=ms2)])
        assert delta.operation_count() == 2

    def test_insert_helper(self, sample_tuple):
        ms = MultiSet.from_tuples([sample_tuple])
        delta = DataDelta.insert(ms)
        assert delta.operation_count() >= 1

    def test_delete_helper(self, sample_tuple):
        ms = MultiSet.from_tuples([sample_tuple])
        delta = DataDelta.delete(ms)
        assert delta.operation_count() >= 1

    def test_update_helper(self, sample_tuple):
        new_tuple = sample_tuple.update_value("age", 99)
        old_ms = MultiSet.from_tuples([sample_tuple])
        new_ms = MultiSet.from_tuples([new_tuple])
        delta = DataDelta.update(old_ms, new_ms)
        assert delta.operation_count() >= 1

    def test_from_diff(self, sample_tuple, another_tuple):
        old = MultiSet.from_tuples([sample_tuple])
        new = MultiSet.from_tuples([sample_tuple, another_tuple])
        delta = DataDelta.from_diff(old, new)
        result = delta.apply_to_data(old.copy())
        assert result.contains(another_tuple)


# ===================================================================
# Section 13: DataDelta — group laws (composition)
# ===================================================================


class TestDataDeltaComposition:
    """DataDelta with compose should satisfy group laws."""

    def _insert_delta(self, *dicts):
        tuples = [TypedTuple.from_dict(d) for d in dicts]
        return DataDelta.insert(MultiSet.from_tuples(tuples))

    def _delete_delta(self, *dicts):
        tuples = [TypedTuple.from_dict(d) for d in dicts]
        return DataDelta.delete(MultiSet.from_tuples(tuples))

    # -- Closure ------------------------------------------------------

    def test_closure(self):
        a = self._insert_delta({"x": 1})
        b = self._insert_delta({"x": 2})
        result = a.compose(b)
        assert isinstance(result, DataDelta)

    # -- Associativity: (a∘b)∘c = a∘(b∘c) ----------------------------

    def test_associativity(self):
        a = self._insert_delta({"v": 1})
        b = self._insert_delta({"v": 2})
        c = self._insert_delta({"v": 3})

        base = MultiSet.empty()
        left = a.compose(b).compose(c)
        right = a.compose(b.compose(c))

        r_left = left.apply_to_data(base.copy())
        r_right = right.apply_to_data(base.copy())
        assert r_left.cardinality() == r_right.cardinality()

    def test_associativity_mixed(self, sample_tuple, another_tuple):
        t3 = TypedTuple.from_dict({"id": 3, "name": "Charlie", "age": 35})
        a = DataDelta.insert(MultiSet.from_tuples([sample_tuple]))
        b = DataDelta.insert(MultiSet.from_tuples([another_tuple]))
        c = DataDelta.insert(MultiSet.from_tuples([t3]))

        base = MultiSet.empty()
        left = a.compose(b).compose(c)
        right = a.compose(b.compose(c))

        r_left = left.apply_to_data(base.copy())
        r_right = right.apply_to_data(base.copy())
        assert r_left.cardinality() == r_right.cardinality()

    # -- Identity: 0∘δ = δ∘0 = δ ------------------------------------

    def test_left_identity(self):
        delta = self._insert_delta({"x": 1})
        zero = DataDelta.zero()
        composed = zero.compose(delta)
        base = MultiSet.empty()
        result = composed.apply_to_data(base.copy())
        assert result.cardinality() == 1

    def test_right_identity(self):
        delta = self._insert_delta({"x": 1})
        zero = DataDelta.zero()
        composed = delta.compose(zero)
        base = MultiSet.empty()
        result = composed.apply_to_data(base.copy())
        assert result.cardinality() == 1

    # -- Inverse: δ∘δ⁻¹ = 0 -----------------------------------------

    def test_inverse_insert(self):
        delta = self._insert_delta({"x": 1})
        inv = delta.inverse()
        composed = delta.compose(inv)
        base = MultiSet.empty()
        result = composed.apply_to_data(base.copy())
        assert result.is_empty() or result.cardinality() == 0

    def test_inverse_delete(self, sample_tuple):
        base = MultiSet.from_tuples([sample_tuple])
        delta = DataDelta.delete(MultiSet.from_tuples([sample_tuple]))
        inv = delta.inverse()
        composed = delta.compose(inv)
        result = composed.apply_to_data(base.copy())
        assert result.cardinality() == base.cardinality()


# ===================================================================
# Section 14: DataDelta — inverse semantics
# ===================================================================


class TestDataDeltaInverse:
    def test_insert_inverse_is_delete(self, sample_tuple):
        ms = MultiSet.from_tuples([sample_tuple])
        delta = DataDelta.insert(ms)
        inv = delta.inverse()
        # The inverse of an insert should delete
        base = MultiSet.from_tuples([sample_tuple])
        result = inv.apply_to_data(base.copy())
        assert result.is_empty() or result.cardinality() == 0

    def test_delete_inverse_is_insert(self, sample_tuple):
        ms = MultiSet.from_tuples([sample_tuple])
        delta = DataDelta.delete(ms)
        inv = delta.inverse()
        # The inverse of a delete should insert
        base = MultiSet.empty()
        result = inv.apply_to_data(base.copy())
        assert result.contains(sample_tuple)

    def test_zero_inverse_is_zero(self):
        delta = DataDelta.zero()
        inv = delta.inverse()
        assert inv.is_zero()

    def test_double_inverse(self, sample_tuple):
        delta = DataDelta.insert(MultiSet.from_tuples([sample_tuple]))
        assert delta.inverse().inverse().operation_count() == delta.operation_count()

    def test_inverse_of_composed(self, sample_tuple, another_tuple):
        d1 = DataDelta.insert(MultiSet.from_tuples([sample_tuple]))
        d2 = DataDelta.insert(MultiSet.from_tuples([another_tuple]))
        composed = d1.compose(d2)
        inv = composed.inverse()
        base = MultiSet.empty()
        forward = composed.apply_to_data(base.copy())
        back = inv.apply_to_data(forward.copy())
        assert back.is_empty() or back.cardinality() == 0


# ===================================================================
# Section 15: DataDelta — normalization
# ===================================================================


class TestDataDeltaNormalize:
    def test_cancel_insert_delete(self, sample_tuple):
        ms = MultiSet.from_tuples([sample_tuple])
        delta = DataDelta.from_operations([InsertOp(tuples=ms), DeleteOp(tuples=ms)])
        norm = delta.normalize()
        base = MultiSet.empty()
        result = norm.apply_to_data(base.copy())
        assert result.is_empty() or result.cardinality() == 0

    def test_normalize_preserves_net_effect(self, sample_tuple, another_tuple):
        ms1 = MultiSet.from_tuples([sample_tuple])
        ms2 = MultiSet.from_tuples([another_tuple])
        ops = [InsertOp(tuples=ms1), InsertOp(tuples=ms2), DeleteOp(tuples=ms1)]
        delta = DataDelta.from_operations(ops)
        norm = delta.normalize()

        base = MultiSet.empty()
        r1 = delta.apply_to_data(base.copy())
        r2 = norm.apply_to_data(base.copy())
        assert r1.cardinality() == r2.cardinality()

    def test_normalize_empty(self):
        delta = DataDelta.zero()
        norm = delta.normalize()
        assert norm.is_zero()

    def test_normalize_idempotent(self, sample_tuple):
        ms = MultiSet.from_tuples([sample_tuple])
        delta = DataDelta.from_operations([InsertOp(tuples=ms)])
        n1 = delta.normalize()
        n2 = n1.normalize()
        base = MultiSet.empty()
        r1 = n1.apply_to_data(base.copy())
        r2 = n2.apply_to_data(base.copy())
        assert r1.cardinality() == r2.cardinality()

    def test_normalize_reduces_operations(self, sample_tuple):
        ms = MultiSet.from_tuples([sample_tuple])
        ops = [InsertOp(tuples=ms), DeleteOp(tuples=ms), InsertOp(tuples=ms), DeleteOp(tuples=ms)]
        delta = DataDelta.from_operations(ops)
        norm = delta.normalize()
        assert norm.operation_count() <= delta.operation_count()


# ===================================================================
# Section 16: apply_to_data
# ===================================================================


class TestApplyToData:
    def test_insert_adds_rows(self, sample_tuple, empty_multiset):
        ms = MultiSet.from_tuples([sample_tuple])
        delta = DataDelta.insert(ms)
        result = delta.apply_to_data(empty_multiset.copy())
        assert result.contains(sample_tuple)
        assert result.cardinality() == 1

    def test_delete_removes_rows(self, sample_tuple, another_tuple):
        data = MultiSet.from_tuples([sample_tuple, another_tuple])
        ms = MultiSet.from_tuples([sample_tuple])
        delta = DataDelta.delete(ms)
        result = delta.apply_to_data(data.copy())
        assert not result.contains(sample_tuple) or result.multiplicity(sample_tuple) == 0
        assert result.contains(another_tuple)

    def test_update_modifies_rows(self, sample_tuple):
        data = MultiSet.from_tuples([sample_tuple])
        new_tuple = sample_tuple.update_value("age", 99)
        old_ms = MultiSet.from_tuples([sample_tuple])
        new_ms = MultiSet.from_tuples([new_tuple])
        delta = DataDelta.update(old_ms, new_ms)
        result = delta.apply_to_data(data.copy())
        found = [t for t in result.tuples() if t.get("age") == 99]
        assert len(found) >= 1

    def test_zero_preserves(self, sample_data):
        delta = DataDelta.zero()
        result = delta.apply_to_data(sample_data.copy())
        assert result.cardinality() == sample_data.cardinality()

    def test_multiple_inserts(self, sample_tuple, another_tuple, empty_multiset):
        ms1 = MultiSet.from_tuples([sample_tuple])
        ms2 = MultiSet.from_tuples([another_tuple])
        delta = DataDelta.from_operations([InsertOp(tuples=ms1), InsertOp(tuples=ms2)])
        result = delta.apply_to_data(empty_multiset.copy())
        assert result.cardinality() == 2

    def test_insert_then_delete(self, sample_tuple, empty_multiset):
        ms = MultiSet.from_tuples([sample_tuple])
        delta = DataDelta.from_operations([InsertOp(tuples=ms), DeleteOp(tuples=ms)])
        result = delta.apply_to_data(empty_multiset.copy())
        assert result.is_empty() or result.cardinality() == 0


# ===================================================================
# Section 17: DataDelta — compression
# ===================================================================


class TestDataDeltaCompression:
    def test_compress_reduces_operations(self, sample_tuple):
        ms = MultiSet.from_tuples([sample_tuple])
        ops = [InsertOp(tuples=ms), DeleteOp(tuples=ms), InsertOp(tuples=ms)]
        delta = DataDelta.from_operations(ops)
        compressed = delta.compress()
        assert compressed.operation_count() <= delta.operation_count()

    def test_compress_preserves_semantics(self, sample_tuple, another_tuple):
        ms1 = MultiSet.from_tuples([sample_tuple])
        ms2 = MultiSet.from_tuples([another_tuple])
        ops = [InsertOp(tuples=ms1), InsertOp(tuples=ms2), DeleteOp(tuples=ms1)]
        delta = DataDelta.from_operations(ops)
        compressed = delta.compress()
        base = MultiSet.empty()
        r1 = delta.apply_to_data(base.copy())
        r2 = compressed.apply_to_data(base.copy())
        assert r1.cardinality() == r2.cardinality()

    def test_compress_empty(self):
        delta = DataDelta.zero()
        compressed = delta.compress()
        assert compressed.is_zero()

    def test_net_row_change_insert(self, sample_tuple):
        ms = MultiSet.from_tuples([sample_tuple])
        delta = DataDelta.insert(ms)
        assert delta.net_row_change() == 1

    def test_net_row_change_delete(self, sample_tuple):
        ms = MultiSet.from_tuples([sample_tuple])
        delta = DataDelta.delete(ms)
        assert delta.net_row_change() == -1

    def test_net_row_change_balanced(self, sample_tuple):
        ms = MultiSet.from_tuples([sample_tuple])
        delta = DataDelta.from_operations([InsertOp(tuples=ms), DeleteOp(tuples=ms)])
        assert delta.net_row_change() == 0


# ===================================================================
# Section 18: Edge cases
# ===================================================================


class TestEdgeCases:
    def test_empty_data_insert(self):
        t = TypedTuple.from_dict({"a": 1})
        ms = MultiSet.from_tuples([t])
        delta = DataDelta.insert(ms)
        result = delta.apply_to_data(MultiSet.empty())
        assert result.cardinality() == 1

    def test_single_row_delete(self):
        t = TypedTuple.from_dict({"a": 1})
        data = MultiSet.from_tuples([t])
        delta = DataDelta.delete(MultiSet.from_tuples([t]))
        result = delta.apply_to_data(data.copy())
        assert result.is_empty() or result.cardinality() == 0

    def test_large_data_insert(self):
        tuples = [TypedTuple.from_dict({"i": i}) for i in range(200)]
        ms = MultiSet.from_tuples(tuples)
        delta = DataDelta.insert(ms)
        result = delta.apply_to_data(MultiSet.empty())
        assert result.cardinality() == 200

    def test_large_data_delete(self):
        tuples = [TypedTuple.from_dict({"i": i}) for i in range(200)]
        data = MultiSet.from_tuples(tuples)
        delta = DataDelta.delete(data.copy())
        result = delta.apply_to_data(data.copy())
        assert result.is_empty() or result.cardinality() == 0

    def test_compose_many(self):
        deltas = [DataDelta.insert(MultiSet.from_tuples([TypedTuple.from_dict({"v": i})])) for i in range(10)]
        result = DataDelta.zero()
        for d in deltas:
            result = result.compose(d)
        data = result.apply_to_data(MultiSet.empty())
        assert data.cardinality() == 10

    def test_empty_tuple_multiset(self):
        t = TypedTuple.from_dict({})
        ms = MultiSet.from_tuples([t])
        assert ms.cardinality() == 1


# ===================================================================
# Section 19: diff relations (from_diff)
# ===================================================================


class TestDiffRelations:
    def test_diff_empty_to_populated(self, sample_tuple, another_tuple):
        old = MultiSet.empty()
        new = MultiSet.from_tuples([sample_tuple, another_tuple])
        delta = DataDelta.from_diff(old, new)
        result = delta.apply_to_data(old.copy())
        assert result.cardinality() == 2

    def test_diff_populated_to_empty(self, sample_tuple, another_tuple):
        old = MultiSet.from_tuples([sample_tuple, another_tuple])
        new = MultiSet.empty()
        delta = DataDelta.from_diff(old, new)
        result = delta.apply_to_data(old.copy())
        assert result.is_empty() or result.cardinality() == 0

    def test_diff_identical(self, sample_data):
        delta = DataDelta.from_diff(sample_data, sample_data.copy())
        assert delta.is_zero() or delta.net_row_change() == 0

    def test_diff_add_one(self, sample_tuple, another_tuple):
        old = MultiSet.from_tuples([sample_tuple])
        new = MultiSet.from_tuples([sample_tuple, another_tuple])
        delta = DataDelta.from_diff(old, new)
        result = delta.apply_to_data(old.copy())
        assert result.contains(another_tuple)
        assert result.cardinality() == 2

    def test_diff_remove_one(self, sample_tuple, another_tuple):
        old = MultiSet.from_tuples([sample_tuple, another_tuple])
        new = MultiSet.from_tuples([sample_tuple])
        delta = DataDelta.from_diff(old, new)
        result = delta.apply_to_data(old.copy())
        assert result.cardinality() == 1
        assert result.contains(sample_tuple)

    def test_diff_round_trip(self, sample_tuple, another_tuple):
        t3 = TypedTuple.from_dict({"id": 3, "name": "Charlie", "age": 35})
        old = MultiSet.from_tuples([sample_tuple, another_tuple])
        new = MultiSet.from_tuples([another_tuple, t3])
        delta = DataDelta.from_diff(old, new)
        result = delta.apply_to_data(old.copy())
        assert result.contains(another_tuple)
        assert result.contains(t3)
        assert not result.contains(sample_tuple) or result.multiplicity(sample_tuple) == 0

    def test_diff_with_duplicates(self, sample_tuple):
        old = MultiSet.from_tuples([sample_tuple])
        new = MultiSet.from_tuples([sample_tuple, sample_tuple])
        delta = DataDelta.from_diff(old, new)
        result = delta.apply_to_data(old.copy())
        assert result.multiplicity(sample_tuple) == 2


# ===================================================================
# Section 20: Serialization round-trip
# ===================================================================


class TestSerialization:
    def test_zero_round_trip(self):
        delta = DataDelta.zero()
        d = delta.to_dict()
        restored = DataDelta.from_dict(d)
        assert restored.is_zero()

    def test_insert_round_trip(self, sample_tuple):
        ms = MultiSet.from_tuples([sample_tuple])
        delta = DataDelta.insert(ms)
        d = delta.to_dict()
        restored = DataDelta.from_dict(d)
        assert restored.operation_count() == delta.operation_count()

    def test_delete_round_trip(self, sample_tuple):
        ms = MultiSet.from_tuples([sample_tuple])
        delta = DataDelta.delete(ms)
        d = delta.to_dict()
        restored = DataDelta.from_dict(d)
        assert restored.operation_count() == delta.operation_count()

    def test_multi_op_round_trip(self, sample_tuple, another_tuple):
        ms1 = MultiSet.from_tuples([sample_tuple])
        ms2 = MultiSet.from_tuples([another_tuple])
        delta = DataDelta.from_operations([InsertOp(tuples=ms1), DeleteOp(tuples=ms2)])
        d = delta.to_dict()
        restored = DataDelta.from_dict(d)
        assert restored.operation_count() == 2

    def test_round_trip_preserves_effect(self, sample_tuple, another_tuple):
        ms1 = MultiSet.from_tuples([sample_tuple])
        ms2 = MultiSet.from_tuples([another_tuple])
        delta = DataDelta.from_operations([InsertOp(tuples=ms1), InsertOp(tuples=ms2)])
        d = delta.to_dict()
        restored = DataDelta.from_dict(d)
        base = MultiSet.empty()
        r1 = delta.apply_to_data(base.copy())
        r2 = restored.apply_to_data(base.copy())
        assert r1.cardinality() == r2.cardinality()

    def test_to_dict_is_dict(self):
        delta = DataDelta.zero()
        d = delta.to_dict()
        assert isinstance(d, dict)

    def test_complex_round_trip(self, sample_tuple, another_tuple):
        t3 = TypedTuple.from_dict({"id": 3, "name": "Charlie", "age": 35})
        ms_insert = MultiSet.from_tuples([t3])
        ms_delete = MultiSet.from_tuples([sample_tuple])
        new_tuple = another_tuple.update_value("age", 99)
        old_ms = MultiSet.from_tuples([another_tuple])
        new_ms = MultiSet.from_tuples([new_tuple])
        ops = [
            InsertOp(tuples=ms_insert),
            DeleteOp(tuples=ms_delete),
            UpdateOp(old_tuples=old_ms, new_tuples=new_ms),
        ]
        delta = DataDelta.from_operations(ops)
        d = delta.to_dict()
        restored = DataDelta.from_dict(d)
        assert restored.operation_count() == delta.operation_count()


# ===================================================================
# Section 21: Integration — full workflows
# ===================================================================


class TestIntegration:
    def test_full_lifecycle(self, sample_tuple, another_tuple):
        """Insert rows, update one, delete another, then invert all."""
        base = MultiSet.empty()
        d1 = DataDelta.insert(MultiSet.from_tuples([sample_tuple, another_tuple]))
        after_insert = d1.apply_to_data(base.copy())
        assert after_insert.cardinality() == 2

        new_tuple = sample_tuple.update_value("age", 99)
        d2 = DataDelta.update(
            MultiSet.from_tuples([sample_tuple]),
            MultiSet.from_tuples([new_tuple]),
        )
        after_update = d2.apply_to_data(after_insert.copy())
        found = [t for t in after_update.tuples() if t.get("age") == 99]
        assert len(found) >= 1

        d3 = DataDelta.delete(MultiSet.from_tuples([another_tuple]))
        after_delete = d3.apply_to_data(after_update.copy())
        assert not after_delete.contains(another_tuple) or after_delete.multiplicity(another_tuple) == 0

        total = d1.compose(d2).compose(d3)
        inv = total.inverse()
        recovered = inv.apply_to_data(after_delete.copy())
        assert recovered.is_empty() or recovered.cardinality() == 0

    def test_diff_then_apply(self, sample_tuple, another_tuple):
        t3 = TypedTuple.from_dict({"id": 3, "name": "Charlie", "age": 35})
        old = MultiSet.from_tuples([sample_tuple, another_tuple])
        new = MultiSet.from_tuples([another_tuple, t3])
        delta = DataDelta.from_diff(old, new)
        result = delta.apply_to_data(old.copy())
        assert result.contains(t3)
        assert result.contains(another_tuple)

    def test_normalize_then_apply(self, sample_tuple, another_tuple):
        ms1 = MultiSet.from_tuples([sample_tuple])
        ms2 = MultiSet.from_tuples([another_tuple])
        ops = [InsertOp(tuples=ms1), DeleteOp(tuples=ms1), InsertOp(tuples=ms2)]
        delta = DataDelta.from_operations(ops)
        norm = delta.normalize()
        base = MultiSet.empty()
        r1 = delta.apply_to_data(base.copy())
        r2 = norm.apply_to_data(base.copy())
        assert r1.cardinality() == r2.cardinality()

    def test_serialize_inverse_compose(self, sample_tuple):
        ms = MultiSet.from_tuples([sample_tuple])
        delta = DataDelta.insert(ms)
        restored = DataDelta.from_dict(delta.to_dict())
        composed = restored.compose(restored.inverse())
        base = MultiSet.empty()
        result = composed.apply_to_data(base.copy())
        assert result.is_empty() or result.cardinality() == 0

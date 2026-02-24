"""
Tests for Tinelli-Zarba theory combination (theory_combination.py).

Verifies arrangement enumeration, individual theory combination,
and full multi-theory consistency checking for the finite-domain
device (5 elements) and phase (2 elements) theories combined with
stably-infinite broadcast/stride theories.
"""

from __future__ import annotations

import pytest

try:
    import z3

    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

import sys, os

# Ensure the project root is on sys.path for imports
_project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.smt.theory_combination import (
    _enumerate_partitions,
    _partition_to_equalities_disequalities,
    DomainKind,
    TheorySolver,
    CombinationResult,
)

if HAS_Z3:
    from src.smt.theory_combination import (
        TheoryCombination,
        TensorTheoryCombination,
    )

pytestmark = pytest.mark.skipif(not HAS_Z3, reason="z3 not installed")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Arrangement enumeration tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPartitionEnumeration:
    """Test the restricted growth string enumeration."""

    def test_empty(self):
        assert _enumerate_partitions(0, 5) == [[]]

    def test_single_element(self):
        # 1 element, any max_classes => 1 partition: [0]
        assert _enumerate_partitions(1, 5) == [[0]]

    def test_two_elements_unlimited(self):
        # 2 elements, max_classes >= 2 => 2 partitions:
        # [0,0] (all same) and [0,1] (all different)
        parts = _enumerate_partitions(2, 5)
        assert len(parts) == 2
        assert [0, 0] in parts
        assert [0, 1] in parts

    def test_two_elements_max_one_class(self):
        # 2 elements, max 1 class => only [0,0]
        parts = _enumerate_partitions(2, 1)
        assert parts == [[0, 0]]

    def test_three_elements_max_two_classes(self):
        # S(3,1) + S(3,2) = 1 + 3 = 4 partitions
        # {abc}, {a|bc}, {b|ac}, {c|ab}
        parts = _enumerate_partitions(3, 2)
        assert len(parts) == 4
        assert [0, 0, 0] in parts  # {abc}
        assert [0, 0, 1] in parts  # {ab|c}
        assert [0, 1, 0] in parts  # {ac|b}
        assert [0, 1, 1] in parts  # {a|bc}

    def test_three_elements_unlimited(self):
        # Bell number B(3) = 5 partitions
        parts = _enumerate_partitions(3, 3)
        assert len(parts) == 5

    def test_four_elements_max_two(self):
        # S(4,1) + S(4,2) = 1 + 7 = 8 partitions
        parts = _enumerate_partitions(4, 2)
        assert len(parts) == 8

    def test_device_domain_two_vars(self):
        """Two shared device vars over 5-element domain.

        With 2 vars over a 5-element domain, max_classes=5 but we
        only have 2 vars so at most 2 classes: 2 partitions.
        """
        parts = _enumerate_partitions(2, 5)
        assert len(parts) == 2

    def test_phase_domain_three_vars(self):
        """Three shared phase vars over 2-element domain.

        With 3 vars, max 2 classes: S(3,1) + S(3,2) = 4 partitions.
        """
        parts = _enumerate_partitions(3, 2)
        assert len(parts) == 4

    def test_canonical_form(self):
        """Partitions use canonical restricted growth strings."""
        parts = _enumerate_partitions(3, 3)
        for p in parts:
            # First element is always 0
            assert p[0] == 0
            # Each element is at most 1 + max of previous elements
            for i in range(1, len(p)):
                assert p[i] <= max(p[:i]) + 1


class TestPartitionToConstraints:
    """Test conversion from partition to equality/disequality pairs."""

    def test_all_equal(self):
        vars_dummy = ["a", "b", "c"]
        eq, diseq = _partition_to_equalities_disequalities(
            vars_dummy, [0, 0, 0]
        )
        assert set(eq) == {(0, 1), (0, 2), (1, 2)}
        assert diseq == []

    def test_all_distinct(self):
        vars_dummy = ["a", "b", "c"]
        eq, diseq = _partition_to_equalities_disequalities(
            vars_dummy, [0, 1, 2]
        )
        assert eq == []
        assert set(diseq) == {(0, 1), (0, 2), (1, 2)}

    def test_mixed(self):
        # [0, 0, 1] => a=b, a≠c, b≠c
        vars_dummy = ["a", "b", "c"]
        eq, diseq = _partition_to_equalities_disequalities(
            vars_dummy, [0, 0, 1]
        )
        assert (0, 1) in eq
        assert (0, 2) in diseq
        assert (1, 2) in diseq


# ═══════════════════════════════════════════════════════════════════════════
# 2. TheorySolver validation tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTheorySolver:
    def test_finite_requires_domain_size(self):
        s = z3.Solver()
        with pytest.raises(ValueError):
            TheorySolver(
                name="bad",
                solver=s,
                domain_kind=DomainKind.FINITE,
                domain_size=None,
            )

    def test_stably_infinite_no_size_needed(self):
        s = z3.Solver()
        ts = TheorySolver(
            name="ok",
            solver=s,
            domain_kind=DomainKind.STABLY_INFINITE,
        )
        assert ts.domain_size is None


# ═══════════════════════════════════════════════════════════════════════════
# 3. Basic combination tests (two solvers, same variables)
# ═══════════════════════════════════════════════════════════════════════════


class TestBasicCombination:
    """Test TheoryCombination with simple Z3 solver pairs."""

    def test_empty_combination(self):
        combo = TheoryCombination()
        result = combo.check_combination()
        assert result.is_consistent

    def test_single_sat_theory(self):
        s = z3.Solver()
        x = z3.Int("x")
        s.add(x > 0)
        combo = TheoryCombination()
        combo.add_theory(
            TheorySolver(
                name="t1",
                solver=s,
                domain_kind=DomainKind.STABLY_INFINITE,
                shared_vars=[x],
            )
        )
        result = combo.check_combination()
        assert result.is_consistent

    def test_single_unsat_theory(self):
        s = z3.Solver()
        x = z3.Int("x")
        s.add(x > 0, x < 0)
        combo = TheoryCombination()
        combo.add_theory(
            TheorySolver(
                name="t1",
                solver=s,
                domain_kind=DomainKind.STABLY_INFINITE,
                shared_vars=[x],
            )
        )
        result = combo.check_combination()
        assert not result.is_consistent

    def test_two_stably_infinite_consistent(self):
        """Two stably-infinite theories with consistent constraints."""
        s1 = z3.Solver()
        s2 = z3.Solver()
        x = z3.Int("x")
        s1.add(x > 0, x < 10)
        s2.add(x > 5, x < 20)
        combo = TheoryCombination()
        combo.add_theory(
            TheorySolver(
                name="t1",
                solver=s1,
                domain_kind=DomainKind.STABLY_INFINITE,
                shared_vars=[x],
            )
        )
        combo.add_theory(
            TheorySolver(
                name="t2",
                solver=s2,
                domain_kind=DomainKind.STABLY_INFINITE,
                shared_vars=[x],
            )
        )
        result = combo.check_combination()
        assert result.is_consistent


# ═══════════════════════════════════════════════════════════════════════════
# 4. Finite-domain combination tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFiniteDomainCombination:
    """Test Tinelli-Zarba arrangement enumeration with finite domains."""

    def test_device_two_vars_must_equal(self):
        """Two device vars forced equal in one solver, compatible in other."""
        from src.smt.device_theory import DeviceSort, DEVICE_VALS

        s1 = z3.Solver()
        s2 = z3.Solver()
        d1 = z3.Const("d1", DeviceSort)
        d2 = z3.Const("d2", DeviceSort)

        # Theory 1: both on CPU
        s1.add(d1 == DEVICE_VALS["CPU"])
        s1.add(d2 == DEVICE_VALS["CPU"])

        # Theory 2: d1 == d2 (no specific device)
        s2.add(d1 == d2)

        combo = TheoryCombination()
        combo.add_theory(
            TheorySolver(
                name="device1",
                solver=s1,
                domain_kind=DomainKind.FINITE,
                domain_size=5,
                shared_vars=[d1, d2],
            )
        )
        combo.add_theory(
            TheorySolver(
                name="device2",
                solver=s2,
                domain_kind=DomainKind.FINITE,
                domain_size=5,
                shared_vars=[d1, d2],
            )
        )
        result = combo.check_combination()
        assert result.is_consistent
        # The arrangement [0,0] (d1=d2) should work
        assert result.satisfying_arrangement is not None

    def test_device_two_vars_contradictory(self):
        """One solver forces equal, other forces different."""
        from src.smt.device_theory import DeviceSort, DEVICE_VALS

        s1 = z3.Solver()
        s2 = z3.Solver()
        d1 = z3.Const("d_a", DeviceSort)
        d2 = z3.Const("d_b", DeviceSort)

        # Theory 1: d1 on CPU, d2 on CUDA_0
        s1.add(d1 == DEVICE_VALS["CPU"])
        s1.add(d2 == DEVICE_VALS["CUDA_0"])

        # Theory 2: d1 == d2 (must be same)
        s2.add(d1 == d2)

        combo = TheoryCombination()
        combo.add_theory(
            TheorySolver(
                name="device_assign",
                solver=s1,
                domain_kind=DomainKind.FINITE,
                domain_size=5,
                shared_vars=[d1, d2],
            )
        )
        combo.add_theory(
            TheorySolver(
                name="device_eq",
                solver=s2,
                domain_kind=DomainKind.FINITE,
                domain_size=5,
                shared_vars=[d1, d2],
            )
        )
        result = combo.check_combination()
        assert not result.is_consistent

    def test_phase_consistent(self):
        """Two phase solvers that agree on arrangement."""
        s1 = z3.Solver()
        s2 = z3.Solver()
        p1 = z3.Bool("p1")
        p2 = z3.Bool("p2")

        # Theory 1: both in train phase
        s1.add(p1 == True)
        s1.add(p2 == True)

        # Theory 2: p1 == p2
        s2.add(p1 == p2)

        combo = TheoryCombination()
        combo.add_theory(
            TheorySolver(
                name="phase1",
                solver=s1,
                domain_kind=DomainKind.FINITE,
                domain_size=2,
                shared_vars=[p1, p2],
            )
        )
        combo.add_theory(
            TheorySolver(
                name="phase2",
                solver=s2,
                domain_kind=DomainKind.FINITE,
                domain_size=2,
                shared_vars=[p1, p2],
            )
        )
        result = combo.check_combination()
        assert result.is_consistent

    def test_phase_inconsistent(self):
        """Phase theory: one solver says same, other says different."""
        s1 = z3.Solver()
        s2 = z3.Solver()
        p1 = z3.Bool("p_x")
        p2 = z3.Bool("p_y")

        # Theory 1: different phases
        s1.add(p1 == True)
        s1.add(p2 == False)

        # Theory 2: must be same
        s2.add(p1 == p2)

        combo = TheoryCombination()
        combo.add_theory(
            TheorySolver(
                name="phase_assign",
                solver=s1,
                domain_kind=DomainKind.FINITE,
                domain_size=2,
                shared_vars=[p1, p2],
            )
        )
        combo.add_theory(
            TheorySolver(
                name="phase_eq",
                solver=s2,
                domain_kind=DomainKind.FINITE,
                domain_size=2,
                shared_vars=[p1, p2],
            )
        )
        result = combo.check_combination()
        assert not result.is_consistent

    def test_three_device_vars(self):
        """Three device vars with a transitive equality chain."""
        from src.smt.device_theory import DeviceSort, DEVICE_VALS

        s1 = z3.Solver()
        s2 = z3.Solver()
        a = z3.Const("dev_a", DeviceSort)
        b = z3.Const("dev_b", DeviceSort)
        c = z3.Const("dev_c", DeviceSort)

        # Theory 1: a=CPU, b=CPU, c=CPU
        s1.add(a == DEVICE_VALS["CPU"])
        s1.add(b == DEVICE_VALS["CPU"])
        s1.add(c == DEVICE_VALS["CPU"])

        # Theory 2: a==b and b==c (transitivity)
        s2.add(a == b, b == c)

        combo = TheoryCombination()
        combo.add_theory(
            TheorySolver(
                name="dev_concrete",
                solver=s1,
                domain_kind=DomainKind.FINITE,
                domain_size=5,
                shared_vars=[a, b, c],
            )
        )
        combo.add_theory(
            TheorySolver(
                name="dev_abstract",
                solver=s2,
                domain_kind=DomainKind.FINITE,
                domain_size=5,
                shared_vars=[a, b, c],
            )
        )
        result = combo.check_combination()
        assert result.is_consistent
        assert result.total_arrangements_checked >= 1


# ═══════════════════════════════════════════════════════════════════════════
# 5. TensorTheoryCombination tests (full system)
# ═══════════════════════════════════════════════════════════════════════════


class TestTensorTheoryCombination:
    """Test the specialized tensor theory combination wrapper."""

    def test_broadcast_and_device_consistent(self):
        """Broadcast theory + device theory, no shared vars between sorts."""
        from src.smt.device_theory import DeviceSort, DEVICE_VALS

        # Broadcast solver
        bc_solver = z3.Solver()
        dim_a, dim_b = z3.Ints("dim_a dim_b")
        bc_solver.add(dim_a == 3, dim_b == 1)

        # Device solver
        dev_solver = z3.Solver()
        dev_x = z3.Const("dev_x", DeviceSort)
        dev_y = z3.Const("dev_y", DeviceSort)
        dev_solver.add(dev_x == DEVICE_VALS["CPU"])
        dev_solver.add(dev_y == DEVICE_VALS["CPU"])

        combo = TensorTheoryCombination()
        combo.add_broadcast_theory(bc_solver, [dim_a, dim_b])
        combo.add_device_theory(dev_solver, [dev_x, dev_y])
        result = combo.verify_theory_combination_consistency()
        assert result.is_consistent

    def test_device_and_phase_cross_sort(self):
        """Device + phase theories, each with their own shared vars."""
        from src.smt.device_theory import DeviceSort, DEVICE_VALS

        dev_solver = z3.Solver()
        d1 = z3.Const("d1_cross", DeviceSort)
        d2 = z3.Const("d2_cross", DeviceSort)
        dev_solver.add(d1 == DEVICE_VALS["CUDA_0"])
        dev_solver.add(d2 == DEVICE_VALS["CUDA_0"])

        phase_solver = z3.Solver()
        p1 = z3.Bool("p1_cross")
        p2 = z3.Bool("p2_cross")
        phase_solver.add(p1 == True, p2 == True)

        combo = TensorTheoryCombination()
        combo.add_device_theory(dev_solver, [d1, d2])
        combo.add_phase_theory(phase_solver, [p1, p2])
        result = combo.verify_theory_combination_consistency()
        assert result.is_consistent

    def test_full_four_theory_consistent(self):
        """All four theories, all consistent."""
        from src.smt.device_theory import DeviceSort, DEVICE_VALS

        bc_solver = z3.Solver()
        dim_x = z3.Int("dim_x")
        bc_solver.add(dim_x >= 1)

        stride_solver = z3.Solver()
        stride_solver.add(dim_x >= 1)

        dev_solver = z3.Solver()
        dev = z3.Const("dev_full", DeviceSort)
        dev_solver.add(dev == DEVICE_VALS["CPU"])

        phase_solver = z3.Solver()
        ph = z3.Bool("ph_full")
        phase_solver.add(ph == True)

        combo = TensorTheoryCombination()
        combo.add_broadcast_theory(bc_solver, [dim_x])
        combo.add_stride_theory(stride_solver, [dim_x])
        combo.add_device_theory(dev_solver, [dev])
        combo.add_phase_theory(phase_solver, [ph])
        result = combo.verify_theory_combination_consistency()
        assert result.is_consistent

    def test_full_four_theory_device_conflict(self):
        """Four theories where device theory has internal conflict."""
        from src.smt.device_theory import DeviceSort, DEVICE_VALS

        bc_solver = z3.Solver()
        dim_x = z3.Int("dim_xc")
        bc_solver.add(dim_x >= 1)

        stride_solver = z3.Solver()
        stride_solver.add(dim_x >= 1)

        dev_solver = z3.Solver()
        dev = z3.Const("dev_conflict", DeviceSort)
        dev_solver.add(dev == DEVICE_VALS["CPU"])
        dev_solver.add(dev == DEVICE_VALS["CUDA_0"])  # conflict

        phase_solver = z3.Solver()
        ph = z3.Bool("ph_conf")
        phase_solver.add(ph == True)

        combo = TensorTheoryCombination()
        combo.add_broadcast_theory(bc_solver, [dim_x])
        combo.add_stride_theory(stride_solver, [dim_x])
        combo.add_device_theory(dev_solver, [dev])
        combo.add_phase_theory(phase_solver, [ph])
        result = combo.verify_theory_combination_consistency()
        assert not result.is_consistent


# ═══════════════════════════════════════════════════════════════════════════
# 6. Edge cases and arrangement counting
# ═══════════════════════════════════════════════════════════════════════════


class TestArrangementCounting:
    """Verify correctness of arrangement counts for relevant domain sizes."""

    def test_stirling_device_1var(self):
        """1 var, 5-element domain => 1 partition."""
        parts = _enumerate_partitions(1, 5)
        assert len(parts) == 1

    def test_stirling_device_2var(self):
        """2 vars, 5-element domain => B(2) = 2 partitions."""
        assert len(_enumerate_partitions(2, 5)) == 2

    def test_stirling_device_3var(self):
        """3 vars, 5-element domain => B(3) = 5 partitions."""
        assert len(_enumerate_partitions(3, 5)) == 5

    def test_stirling_device_4var(self):
        """4 vars, 5-element domain => B(4) = 15 partitions."""
        assert len(_enumerate_partitions(4, 5)) == 15

    def test_stirling_phase_1var(self):
        """1 var, 2-element domain => 1 partition."""
        assert len(_enumerate_partitions(1, 2)) == 1

    def test_stirling_phase_2var(self):
        """2 vars, 2-element domain => 2 partitions."""
        assert len(_enumerate_partitions(2, 2)) == 2

    def test_stirling_phase_3var(self):
        """3 vars, 2-element domain => 4 partitions."""
        assert len(_enumerate_partitions(3, 2)) == 4

    def test_stirling_phase_4var(self):
        """4 vars, 2-element domain => S(4,1)+S(4,2) = 1+7 = 8."""
        assert len(_enumerate_partitions(4, 2)) == 8

    def test_five_vars_five_element_domain(self):
        """5 vars, 5 elements => B(5) = 52 partitions."""
        assert len(_enumerate_partitions(5, 5)) == 52


class TestNoSharedVars:
    """When theories have no shared variables, combination is trivially OK."""

    def test_independent_theories(self):
        s1 = z3.Solver()
        s2 = z3.Solver()
        x = z3.Int("x_ind")
        y = z3.Int("y_ind")
        s1.add(x > 0)
        s2.add(y > 0)

        combo = TheoryCombination()
        combo.add_theory(
            TheorySolver(
                name="t1",
                solver=s1,
                domain_kind=DomainKind.STABLY_INFINITE,
                shared_vars=[],
            )
        )
        combo.add_theory(
            TheorySolver(
                name="t2",
                solver=s2,
                domain_kind=DomainKind.STABLY_INFINITE,
                shared_vars=[],
            )
        )
        result = combo.check_combination()
        assert result.is_consistent

    def test_independent_finite_theories(self):
        """Finite theories with no shared vars — each checked solo."""
        from src.smt.device_theory import DeviceSort, DEVICE_VALS

        s1 = z3.Solver()
        d1 = z3.Const("d_solo1", DeviceSort)
        s1.add(d1 == DEVICE_VALS["CPU"])

        s2 = z3.Solver()
        p1 = z3.Bool("p_solo1")
        s2.add(p1 == True)

        combo = TheoryCombination()
        combo.add_theory(
            TheorySolver(
                name="device",
                solver=s1,
                domain_kind=DomainKind.FINITE,
                domain_size=5,
                shared_vars=[],
            )
        )
        combo.add_theory(
            TheorySolver(
                name="phase",
                solver=s2,
                domain_kind=DomainKind.FINITE,
                domain_size=2,
                shared_vars=[],
            )
        )
        result = combo.check_combination()
        assert result.is_consistent

"""
Custom Z3 Theory Plugin for Tensor Stride Constraints.

Encodes multiplicative stride/shape relationships for tensor memory
layout verification.  For a contiguous tensor with shape
(d0, d1, ..., dn), the strides satisfy:

    stride[i] = product(shape[i+1:])

Formal Specification
--------------------

**Signature** Σ_stride = (S, F, P) where:
  - Sorts S = {Dim, Stride} (both positive integers ≥ 1)
  - Functions F:
    cstride : Shape → Stride^n      (contiguous stride computation)
    numel   : Shape → Dim            (total element count = ∏ shape[i])
  - Predicates P:
    contiguous : Shape × Stride^n    (stride matches contiguous layout)
    reshape_ok : Shape × Shape       (element count preservation)

**Axioms**:
  A1 (contiguous def):  contiguous(s, t) ⟺
                         t[n-1] = 1 ∧ ∀i < n-1. t[i] = t[i+1] × s[i+1]
  A2 (reshape):          reshape_ok(s_in, s_out) ⟺ numel(s_in) = numel(s_out)
  A3 (conv output):      H' = ⌊(H + 2p - k) / s⌋ + 1

**Decision procedure**: Contiguous stride constraints (A1) are linear
  (QF-LIA).  Reshape constraints (A2) involve multiplication of
  variables (QF-NIA); handled by Z3's nonlinear solver.
  The UserPropagator eagerly propagates stride values when shape
  dimensions are fixed.

**Soundness**: Propagated stride equalities follow directly from A1.
  Conflicts are raised only when concrete values violate A1.

This module provides:
  - A Z3 UserPropagateBase that eagerly propagates stride values
    when shape dimensions are fixed (simplex-like solver for
    multiplicative constraints over bounded positive integers).
  - Theory lemma generation for stride inconsistencies.
  - High-level ``StridePropagator`` / ``StrideTheoryPlugin`` classes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import reduce
from operator import mul
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Z3 import with graceful fallback
# ---------------------------------------------------------------------------

try:
    import z3

    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False


# ═══════════════════════════════════════════════════════════════════════════
# 1. Pure stride helpers
# ═══════════════════════════════════════════════════════════════════════════


def compute_contiguous_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Compute contiguous (row-major / C-order) strides for a shape.

    Args:
        shape: Tensor dimensions, e.g. (2, 3, 4).

    Returns:
        Strides tuple, e.g. (12, 4, 1) for shape (2, 3, 4).
    """
    if not shape:
        return ()
    n = len(shape)
    strides = [0] * n
    strides[n - 1] = 1
    for i in range(n - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


def is_contiguous(
    shape: Tuple[int, ...], strides: Tuple[int, ...]
) -> bool:
    """Check whether strides match contiguous layout for the given shape."""
    if len(shape) != len(strides):
        return False
    return strides == compute_contiguous_strides(shape)


def total_elements(shape: Tuple[int, ...]) -> int:
    """Total number of elements in a tensor of the given shape."""
    return reduce(mul, shape, 1)


def stride_divides_shape(
    shape: Tuple[int, ...], strides: Tuple[int, ...]
) -> bool:
    """Check that each stride divides the product of trailing shape dims.

    This is a necessary condition for a valid strided view.
    """
    if len(shape) != len(strides):
        return False
    n = len(shape)
    for i in range(n):
        trailing = reduce(mul, shape[i + 1 :], 1)
        if strides[i] != 0 and trailing % strides[i] != 0:
            return False
    return True


# ═══════════════════════════════════════════════════════════════════════════
# 2. Trail for backtracking
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class _StrideTrailFrame:
    """Snapshot of stride theory state at a push point."""

    fixed_vars: Dict[int, int]


# ═══════════════════════════════════════════════════════════════════════════
# 3. StridePropagator — Z3 UserPropagateBase implementation
# ═══════════════════════════════════════════════════════════════════════════

if HAS_Z3:

    class StridePropagator(z3.UserPropagateBase):
        """Z3 theory propagator for multiplicative stride constraints.

        Implements a simplex-like solver for multiplicative constraints
        over bounded positive integers.  Supports:

        - Contiguous stride constraints: stride[i] = product(shape[i+1:])
        - Reshape validity: product(old_shape) == product(new_shape)
        - Divisibility constraints for strided views
        """

        # Upper bound for shape dimensions (avoids unbounded search)
        MAX_DIM = 2**20

        def __init__(self, s: z3.Solver) -> None:
            super().__init__(s)

            self._vars: Dict[int, z3.ExprRef] = {}
            self._fixed: Dict[int, int] = {}
            self._trail: List[_StrideTrailFrame] = []

            # Constraint storage
            # contiguous stride: (shape_vars, stride_vars)
            self._contiguous: List[
                Tuple[List[z3.ExprRef], List[z3.ExprRef]]
            ] = []
            # reshape validity: (old_shape_vars, new_shape_vars)
            self._reshapes: List[
                Tuple[List[z3.ExprRef], List[z3.ExprRef]]
            ] = []
            # divisibility: (dividend_var, divisor_var) s.t. divisor | dividend
            self._divisibility: List[Tuple[z3.ExprRef, z3.ExprRef]] = []

            self.add_fixed(self._on_fixed)
            self.add_final(self._on_final)
            self.add_created(self._on_created)

        # ---------------------------------------------------------------
        # Variable registration
        # ---------------------------------------------------------------

        def _register_var(self, v: z3.ExprRef) -> None:
            vid = v.get_id()
            if vid not in self._vars:
                self._vars[vid] = v
                self.add(v)

        # ---------------------------------------------------------------
        # Backtracking
        # ---------------------------------------------------------------

        def push(self) -> None:
            self._trail.append(_StrideTrailFrame(fixed_vars=dict(self._fixed)))

        def pop(self, num_scopes: int) -> None:
            for _ in range(num_scopes):
                if self._trail:
                    frame = self._trail.pop()
                    self._fixed = frame.fixed_vars

        # ---------------------------------------------------------------
        # Callbacks
        # ---------------------------------------------------------------

        def _on_created(self, var: z3.ExprRef) -> None:
            self._vars[var.get_id()] = var

        def _on_fixed(self, var: z3.ExprRef, value: z3.ExprRef) -> None:
            """Eagerly propagate when Z3 fixes a dimension or stride variable.

            Uses a simplex-like strategy: when all but one variable in a
            multiplicative constraint are fixed, solve for the remaining one.
            """
            vid = var.get_id()
            try:
                concrete = value.as_long()
            except Exception:
                return
            self._fixed[vid] = concrete

            # Propagate contiguous stride constraints
            for shapes, strides in self._contiguous:
                self._propagate_contiguous(shapes, strides)

            # Propagate reshape constraints
            for old_s, new_s in self._reshapes:
                self._propagate_reshape(old_s, new_s)

            # Check divisibility
            for dividend, divisor in self._divisibility:
                self._propagate_divisibility(dividend, divisor)

        def _on_final(self) -> None:
            """Final consistency check for all stride constraints."""
            for shapes, strides in self._contiguous:
                self._check_contiguous_final(shapes, strides)
            for old_s, new_s in self._reshapes:
                self._check_reshape_final(old_s, new_s)
            for dividend, divisor in self._divisibility:
                self._check_divisibility_final(dividend, divisor)

        # ---------------------------------------------------------------
        # Contiguous stride propagation
        # ---------------------------------------------------------------

        def _propagate_contiguous(
            self,
            shapes: List[z3.ExprRef],
            strides: List[z3.ExprRef],
        ) -> None:
            """Propagate stride values when shape dims become known.

            For stride[n-1] = 1, stride[i] = stride[i+1] * shape[i+1].
            When shape dims are fixed right-to-left, eagerly compute
            stride values.
            """
            n = len(shapes)
            if n == 0:
                return

            # Try to propagate stride[n-1] = 1
            sn_id = strides[n - 1].get_id()
            sn_val = self._fixed.get(sn_id)
            if sn_val is None:
                self.propagate(
                    strides[n - 1] == z3.IntVal(1),
                    ids=[strides[n - 1]],
                )
            elif sn_val != 1:
                self.conflict(deps=[strides[n - 1]])
                return

            # Propagate from right to left
            for i in range(n - 2, -1, -1):
                shape_next = self._fixed.get(shapes[i + 1].get_id())
                stride_next = self._fixed.get(strides[i + 1].get_id())
                stride_cur = self._fixed.get(strides[i].get_id())

                if shape_next is not None and stride_next is not None:
                    expected = stride_next * shape_next
                    if stride_cur is None:
                        self.propagate(
                            strides[i] == z3.IntVal(expected),
                            ids=[shapes[i + 1], strides[i + 1]],
                        )
                    elif stride_cur != expected:
                        self.conflict(
                            deps=[shapes[i + 1], strides[i + 1], strides[i]]
                        )
                        return

        def _check_contiguous_final(
            self,
            shapes: List[z3.ExprRef],
            strides: List[z3.ExprRef],
        ) -> None:
            """Final check that strides match contiguous layout."""
            n = len(shapes)
            s_vals = [self._fixed.get(s.get_id()) for s in shapes]
            t_vals = [self._fixed.get(s.get_id()) for s in strides]
            if any(v is None for v in s_vals) or any(v is None for v in t_vals):
                return
            expected = compute_contiguous_strides(tuple(s_vals))  # type: ignore[arg-type]
            for i in range(n):
                if t_vals[i] != expected[i]:
                    self.conflict(deps=list(shapes) + list(strides))
                    return

        # ---------------------------------------------------------------
        # Reshape propagation
        # ---------------------------------------------------------------

        def _propagate_reshape(
            self,
            old_shape: List[z3.ExprRef],
            new_shape: List[z3.ExprRef],
        ) -> None:
            """Propagate reshape validity: products must match.

            When all dims of one shape are known and all but one of the
            other are known, solve for the missing dim (simplex-like).
            """
            old_vals = [self._fixed.get(v.get_id()) for v in old_shape]
            new_vals = [self._fixed.get(v.get_id()) for v in new_shape]

            old_known = all(v is not None for v in old_vals)
            new_known = all(v is not None for v in new_vals)

            if old_known and new_known:
                old_prod = reduce(mul, old_vals, 1)  # type: ignore[arg-type]
                new_prod = reduce(mul, new_vals, 1)  # type: ignore[arg-type]
                if old_prod != new_prod:
                    self.conflict(deps=list(old_shape) + list(new_shape))
                return

            # Simplex-like: solve for one missing dim
            if old_known:
                old_prod = reduce(mul, old_vals, 1)  # type: ignore[arg-type]
                missing_idx = [i for i, v in enumerate(new_vals) if v is None]
                if len(missing_idx) == 1:
                    idx = missing_idx[0]
                    known_prod = reduce(
                        mul,
                        [v for v in new_vals if v is not None],
                        1,
                    )
                    if known_prod != 0 and old_prod % known_prod == 0:
                        solved = old_prod // known_prod
                        deps = list(old_shape) + [
                            new_shape[i]
                            for i in range(len(new_shape))
                            if i != idx
                        ]
                        self.propagate(
                            new_shape[idx] == z3.IntVal(solved),
                            ids=deps,
                        )

            if new_known:
                new_prod = reduce(mul, new_vals, 1)  # type: ignore[arg-type]
                missing_idx = [i for i, v in enumerate(old_vals) if v is None]
                if len(missing_idx) == 1:
                    idx = missing_idx[0]
                    known_prod = reduce(
                        mul,
                        [v for v in old_vals if v is not None],
                        1,
                    )
                    if known_prod != 0 and new_prod % known_prod == 0:
                        solved = new_prod // known_prod
                        deps = list(new_shape) + [
                            old_shape[i]
                            for i in range(len(old_shape))
                            if i != idx
                        ]
                        self.propagate(
                            old_shape[idx] == z3.IntVal(solved),
                            ids=deps,
                        )

        def _check_reshape_final(
            self,
            old_shape: List[z3.ExprRef],
            new_shape: List[z3.ExprRef],
        ) -> None:
            """Final check that reshape preserves total elements."""
            old_vals = [self._fixed.get(v.get_id()) for v in old_shape]
            new_vals = [self._fixed.get(v.get_id()) for v in new_shape]
            if any(v is None for v in old_vals) or any(
                v is None for v in new_vals
            ):
                return
            old_prod = reduce(mul, old_vals, 1)  # type: ignore[arg-type]
            new_prod = reduce(mul, new_vals, 1)  # type: ignore[arg-type]
            if old_prod != new_prod:
                self.conflict(deps=list(old_shape) + list(new_shape))

        # ---------------------------------------------------------------
        # Divisibility propagation
        # ---------------------------------------------------------------

        def _propagate_divisibility(
            self,
            dividend: z3.ExprRef,
            divisor: z3.ExprRef,
        ) -> None:
            """Check divisibility when both values are fixed."""
            va = self._fixed.get(dividend.get_id())
            vb = self._fixed.get(divisor.get_id())
            if va is not None and vb is not None:
                if vb == 0 or va % vb != 0:
                    self.conflict(deps=[dividend, divisor])

        def _check_divisibility_final(
            self,
            dividend: z3.ExprRef,
            divisor: z3.ExprRef,
        ) -> None:
            """Final divisibility check."""
            va = self._fixed.get(dividend.get_id())
            vb = self._fixed.get(divisor.get_id())
            if va is not None and vb is not None:
                if vb == 0 or va % vb != 0:
                    self.conflict(deps=[dividend, divisor])

    # ═══════════════════════════════════════════════════════════════════════
    # 4. High-level constraint builders
    # ═══════════════════════════════════════════════════════════════════════

    def contiguous_strides(
        prop: StridePropagator,
        shape: List[z3.ExprRef],
        strides: List[z3.ExprRef],
    ) -> z3.ExprRef:
        """Assert contiguous (row-major) stride layout.

        Args:
            prop: The stride propagator.
            shape: Z3 Int vars for tensor shape.
            strides: Z3 Int vars for tensor strides.

        Returns:
            Z3 Bool conjunction encoding the stride relationship.
        """
        n = len(shape)
        if len(strides) != n:
            return z3.BoolVal(False)
        for v in list(shape) + list(strides):
            prop._register_var(v)
        prop._contiguous.append((list(shape), list(strides)))

        clauses = [strides[n - 1] == 1]
        for i in range(n - 2, -1, -1):
            clauses.append(strides[i] == strides[i + 1] * shape[i + 1])
        return z3.And(*clauses)

    def reshape_valid(
        prop: StridePropagator,
        old_shape: List[z3.ExprRef],
        new_shape: List[z3.ExprRef],
    ) -> z3.ExprRef:
        """Assert reshape validity: total elements are preserved.

        Args:
            prop: The stride propagator.
            old_shape: Z3 Int vars for the original shape.
            new_shape: Z3 Int vars for the target shape.

        Returns:
            Z3 Bool asserting product(old_shape) == product(new_shape).
        """
        for v in list(old_shape) + list(new_shape):
            prop._register_var(v)
        prop._reshapes.append((list(old_shape), list(new_shape)))

        # Encode as explicit product equality
        def _product(vs: List[z3.ExprRef]) -> z3.ExprRef:
            if not vs:
                return z3.IntVal(1)
            result = vs[0]
            for v in vs[1:]:
                result = result * v
            return result

        return _product(old_shape) == _product(new_shape)

    def divisibility_constraint(
        prop: StridePropagator,
        dividend: z3.ExprRef,
        divisor: z3.ExprRef,
    ) -> z3.ExprRef:
        """Assert that divisor divides dividend.

        Args:
            prop: The stride propagator.
            dividend: Z3 Int var.
            divisor: Z3 Int var.

        Returns:
            Z3 Bool: dividend % divisor == 0.
        """
        prop._register_var(dividend)
        prop._register_var(divisor)
        prop._divisibility.append((dividend, divisor))
        return dividend % divisor == 0

    # ═══════════════════════════════════════════════════════════════════════
    # 5. StrideTheoryPlugin — convenience wrapper
    # ═══════════════════════════════════════════════════════════════════════

    class StrideTheoryPlugin:
        """High-level integration wrapper for attaching the stride
        theory to any Z3 Solver.

        Usage::

            solver = z3.Solver()
            plugin = StrideTheoryPlugin(solver)
            d0, d1, d2 = z3.Ints("d0 d1 d2")
            s0, s1, s2 = z3.Ints("s0 s1 s2")
            solver.add(plugin.contiguous_strides([d0,d1,d2], [s0,s1,s2]))
            solver.add(d0 == 2, d1 == 3, d2 == 4)
            assert solver.check() == z3.sat
        """

        def __init__(self, solver: z3.Solver) -> None:
            self.solver = solver
            self.propagator = StridePropagator(solver)

        def contiguous_strides(
            self,
            shape: List[z3.ExprRef],
            strides: List[z3.ExprRef],
        ) -> z3.ExprRef:
            """Assert contiguous stride layout."""
            return contiguous_strides(self.propagator, shape, strides)

        def reshape_valid(
            self,
            old_shape: List[z3.ExprRef],
            new_shape: List[z3.ExprRef],
        ) -> z3.ExprRef:
            """Assert reshape preserves total elements."""
            return reshape_valid(self.propagator, old_shape, new_shape)

        def divisibility_constraint(
            self,
            dividend: z3.ExprRef,
            divisor: z3.ExprRef,
        ) -> z3.ExprRef:
            """Assert divisor divides dividend."""
            return divisibility_constraint(
                self.propagator, dividend, divisor
            )


# ═══════════════════════════════════════════════════════════════════════════
# 6. Self-test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not HAS_Z3:
        print("SKIP: z3 not installed")
        raise SystemExit(0)

    import sys

    passed = 0
    failed = 0

    def _assert(cond: bool, msg: str) -> None:
        global passed, failed
        if cond:
            passed += 1
            print(f"  PASS: {msg}")
        else:
            failed += 1
            print(f"  FAIL: {msg}")

    # --- Pure helpers ---
    print("=== Pure stride helpers ===")
    _assert(
        compute_contiguous_strides((2, 3, 4)) == (12, 4, 1),
        "strides(2,3,4) == (12,4,1)",
    )
    _assert(
        compute_contiguous_strides((5,)) == (1,),
        "strides(5,) == (1,)",
    )
    _assert(
        compute_contiguous_strides((2, 3)) == (3, 1),
        "strides(2,3) == (3,1)",
    )
    _assert(
        is_contiguous((2, 3, 4), (12, 4, 1)),
        "(2,3,4) with strides (12,4,1) is contiguous",
    )
    _assert(
        not is_contiguous((2, 3, 4), (12, 3, 1)),
        "(2,3,4) with strides (12,3,1) is NOT contiguous",
    )
    _assert(total_elements((2, 3, 4)) == 24, "total_elements(2,3,4) == 24")

    # --- StrideTheoryPlugin: contiguous strides ---
    print("\n=== StrideTheoryPlugin: contiguous strides ===")
    s = z3.Solver()
    plugin = StrideTheoryPlugin(s)
    d0, d1, d2 = z3.Ints("d0 d1 d2")
    st0, st1, st2 = z3.Ints("st0 st1 st2")
    s.add(plugin.contiguous_strides([d0, d1, d2], [st0, st1, st2]))
    s.add(d0 == 2, d1 == 3, d2 == 4)
    result = s.check()
    _assert(result == z3.sat, "contiguous strides for (2,3,4) SAT")
    if result == z3.sat:
        model = s.model()
        _assert(model[st0].as_long() == 12, "stride[0] == 12")
        _assert(model[st1].as_long() == 4, "stride[1] == 4")
        _assert(model[st2].as_long() == 1, "stride[2] == 1")

    # Wrong strides should be UNSAT
    print("\n=== StrideTheoryPlugin: wrong strides UNSAT ===")
    s2 = z3.Solver()
    p2 = StrideTheoryPlugin(s2)
    a0, a1, a2 = z3.Ints("a0 a1 a2")
    b0, b1, b2 = z3.Ints("b0 b1 b2")
    s2.add(p2.contiguous_strides([a0, a1, a2], [b0, b1, b2]))
    s2.add(a0 == 2, a1 == 3, a2 == 4)
    s2.add(b0 == 12, b1 == 3, b2 == 1)  # b1 should be 4, not 3
    _assert(s2.check() == z3.unsat, "wrong strides UNSAT")

    # --- StrideTheoryPlugin: reshape validity ---
    print("\n=== StrideTheoryPlugin: reshape validity ===")
    s3 = z3.Solver()
    p3 = StrideTheoryPlugin(s3)
    r0, r1 = z3.Ints("r0 r1")
    t0, t1, t2 = z3.Ints("t0 t1 t2")
    s3.add(p3.reshape_valid([r0, r1], [t0, t1, t2]))
    s3.add(r0 == 6, r1 == 4)  # 24 elements
    s3.add(t0 == 2, t1 == 3)  # t2 should be 4
    s3.add(t2 > 0)
    result3 = s3.check()
    _assert(result3 == z3.sat, "reshape (6,4) -> (2,3,?) SAT")
    if result3 == z3.sat:
        _assert(s3.model()[t2].as_long() == 4, "inferred t2 == 4")

    # Invalid reshape
    s4 = z3.Solver()
    p4 = StrideTheoryPlugin(s4)
    u0, u1 = z3.Ints("u0 u1")
    v0, v1 = z3.Ints("v0 v1")
    s4.add(p4.reshape_valid([u0, u1], [v0, v1]))
    s4.add(u0 == 3, u1 == 4)  # 12 elements
    s4.add(v0 == 5, v1 == 3)  # 15 elements — mismatch
    _assert(s4.check() == z3.unsat, "reshape (3,4) -> (5,3) UNSAT")

    # --- StrideTheoryPlugin: divisibility ---
    print("\n=== StrideTheoryPlugin: divisibility ===")
    s5 = z3.Solver()
    p5 = StrideTheoryPlugin(s5)
    dd, dv = z3.Ints("dd dv")
    s5.add(p5.divisibility_constraint(dd, dv))
    s5.add(dd == 12, dv == 4)
    _assert(s5.check() == z3.sat, "12 % 4 == 0 SAT")

    s6 = z3.Solver()
    p6 = StrideTheoryPlugin(s6)
    dd2, dv2 = z3.Ints("dd2 dv2")
    s6.add(p6.divisibility_constraint(dd2, dv2))
    s6.add(dd2 == 12, dv2 == 5)
    _assert(s6.check() == z3.unsat, "12 % 5 != 0 UNSAT")

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    print("All stride_theory tests passed!")

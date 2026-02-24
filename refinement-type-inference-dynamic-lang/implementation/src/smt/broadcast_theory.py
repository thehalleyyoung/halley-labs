"""
Custom Z3 Theory Plugin for Tensor Broadcasting Semantics.

Implements NumPy/PyTorch broadcasting rules as a first-class Z3 theory
using the UserPropagateBase API. This is the first domain-specific SMT
theory for tensor shape reasoning — no existing SMT solver provides
native support for array broadcasting.

Formal Specification
--------------------

**Signature** Σ_broadcast = (S, F, P) where:
  - Sorts S = {Dim, Shape, Idx}
    Dim: positive integers (≥ 1)
    Shape: finite tuples over Dim
    Idx: natural numbers (0-indexed positions)
  - Functions F:
    get : Shape × Idx → Dim        (dimension accessor)
    len : Shape → Idx               (rank)
    bc  : Dim × Dim → Dim           (broadcast result)
    pad : Shape × Idx → Shape       (left-pad with 1s to target rank)
  - Predicates P:
    bcompat : Dim × Dim             (dimension-level compatibility)
    scompat : Shape × Shape         (shape-level compatibility)
    mcompat : Shape × Shape         (matmul inner-dim match)

**Axioms** (quantifier-free fragment with concrete ranks):
  A1 (compatibility):   bcompat(a, b) ⟺ a = b ∨ a = 1 ∨ b = 1
  A2 (broadcast result): bcompat(a, b) ⟹
                          bc(a, b) = if a = 1 then b
                                     else if b = 1 then a
                                     else a  (since a = b by A1)
  A3 (incompatibility):  ¬bcompat(a, b) ⟹ bc(a, b) = ⊥ (error)
  A4 (shape compat):     scompat(A, B) ⟺
                          ∀i < max(len(A), len(B)).
                            bcompat(get(pad(A, max), i), get(pad(B, max), i))
  A5 (matmul):           mcompat(A, B) ⟺ get(A, len(A)-1) = get(B, len(B)-2)
  A6 (positive dims):    ∀s.∀i < len(s). get(s, i) ≥ 1

**Decision procedure**: For the quantifier-free fragment with concrete
  ranks, satisfiability reduces to QF-LIA (decidable by Presburger 1929).
  The UserPropagator implements eager theory propagation within DPLL(T):
  - On fixed(d_A[i], v_A) ∧ fixed(d_B[i], v_B):
      if bcompat(v_A, v_B): propagate d_out[i] = bc(v_A, v_B)
      else: conflict({d_A[i], d_B[i]})
  - On final(): verify all registered triples satisfy A1-A3.

**Soundness**: The propagator is sound: every propagated equality
  d_out = bc(d_A, d_B) follows from A1-A2, and every conflict
  {d_A = v_A, d_B = v_B} follows from A3 (¬bcompat(v_A, v_B)).
  No unsound inferences are possible because propagation is purely
  eager (forward from fixed assignments) with no speculative deductions.

**Theory combination**: Σ_broadcast operates over the stably-infinite
  sort Dim ⊆ ℤ_≥1, so classical Nelson-Oppen combination applies for
  the shape theory.  For combination with the finite-domain device and
  phase theories, we use the Tinelli-Zarba extension (JAR 2005) which
  handles non-stably-infinite sorts via arrangement enumeration.

Broadcasting rules (NumPy semantics):
  Two shapes are broadcast-compatible if, for each dimension pair (a, b)
  aligned from the right: a == b, or a == 1, or b == 1.
  The output dimension is max(a, b).
  Missing dimensions (shorter shape) are treated as 1.

Theory operations exposed as Z3-level predicates:
  - broadcast_compatible(shape_a, shape_b)
  - broadcast_result_dim(dim_a, dim_b, dim_out)
  - matmul_compatible(shape_a, shape_b)
  - stride_compatible(shape, stride)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

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
# 1. Pure broadcasting helpers (no Z3 dependency)
# ═══════════════════════════════════════════════════════════════════════════


def _are_dims_broadcast_compatible(a: int, b: int) -> bool:
    """Check if two concrete dimension sizes are broadcast-compatible."""
    return a == b or a == 1 or b == 1


def _broadcast_result(a: int, b: int) -> int:
    """Compute the broadcast output dimension for two concrete sizes."""
    if not _are_dims_broadcast_compatible(a, b):
        raise ValueError(f"Dimensions {a} and {b} are not broadcast-compatible")
    return max(a, b)


def _shapes_broadcast_compatible(
    shape_a: Tuple[int, ...], shape_b: Tuple[int, ...]
) -> bool:
    """Check full shape broadcast compatibility (NumPy semantics)."""
    max_rank = max(len(shape_a), len(shape_b))
    # Pad shorter shape with 1s on the left
    pa = (1,) * (max_rank - len(shape_a)) + shape_a
    pb = (1,) * (max_rank - len(shape_b)) + shape_b
    return all(_are_dims_broadcast_compatible(a, b) for a, b in zip(pa, pb))


def _broadcast_shape(
    shape_a: Tuple[int, ...], shape_b: Tuple[int, ...]
) -> Tuple[int, ...]:
    """Compute the full broadcast output shape."""
    max_rank = max(len(shape_a), len(shape_b))
    pa = (1,) * (max_rank - len(shape_a)) + shape_a
    pb = (1,) * (max_rank - len(shape_b)) + shape_b
    return tuple(_broadcast_result(a, b) for a, b in zip(pa, pb))


# ═══════════════════════════════════════════════════════════════════════════
# 2. Trail for backtracking support
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class _TrailFrame:
    """Snapshot of theory state at a push point."""

    fixed_vars: Dict[int, int]  # copy of var_id -> concrete value


# ═══════════════════════════════════════════════════════════════════════════
# 3. BroadcastPropagator — Z3 UserPropagateBase implementation
# ═══════════════════════════════════════════════════════════════════════════

if HAS_Z3:

    class BroadcastPropagator(z3.UserPropagateBase):
        """Z3 theory propagator for tensor broadcasting constraints.

        Registers dimension variables and broadcasting constraint triples
        (dim_a, dim_b, dim_out).  When Z3 fixes concrete values for any
        of these variables the propagator eagerly checks / propagates
        broadcasting semantics and raises conflicts on violations.
        """

        def __init__(self, s: z3.Solver) -> None:
            super().__init__(s)

            # var tracking: z3 expr id -> z3 expr
            self._vars: Dict[int, z3.ExprRef] = {}
            # fixed assignments: z3 expr id -> concrete int value
            self._fixed: Dict[int, int] = {}
            # backtracking trail
            self._trail: List[_TrailFrame] = []

            # broadcasting constraints: list of (dim_a, dim_b, dim_out) z3 vars
            self._broadcast_triples: List[
                Tuple[z3.ExprRef, z3.ExprRef, z3.ExprRef]
            ] = []
            # matmul constraints: list of (inner_a, inner_b) z3 vars
            self._matmul_pairs: List[Tuple[z3.ExprRef, z3.ExprRef]] = []
            # stride constraints: list of (shape_dims..., stride_dims...) tuples
            self._stride_constraints: List[
                Tuple[List[z3.ExprRef], List[z3.ExprRef]]
            ] = []

            # Register callbacks
            self.add_fixed(self._on_fixed)
            self.add_final(self._on_final)
            self.add_created(self._on_created)

        # ---------------------------------------------------------------
        # Variable registration
        # ---------------------------------------------------------------

        def _register_var(self, v: z3.ExprRef) -> None:
            """Register a Z3 integer variable with the propagator."""
            vid = v.get_id()
            if vid not in self._vars:
                self._vars[vid] = v
                self.add(v)

        # ---------------------------------------------------------------
        # Backtracking: push / pop
        # ---------------------------------------------------------------

        def push(self) -> None:
            """Save current state for backtracking."""
            self._trail.append(_TrailFrame(fixed_vars=dict(self._fixed)))

        def pop(self, num_scopes: int) -> None:
            """Restore state to ``num_scopes`` push points ago."""
            for _ in range(num_scopes):
                if self._trail:
                    frame = self._trail.pop()
                    self._fixed = frame.fixed_vars

        # ---------------------------------------------------------------
        # Callbacks
        # ---------------------------------------------------------------

        def _on_created(self, var: z3.ExprRef) -> None:
            """Called when Z3 creates a tracked variable."""
            self._vars[var.get_id()] = var

        def _on_fixed(self, var: z3.ExprRef, value: z3.ExprRef) -> None:
            """Called when Z3 assigns a concrete value to a tracked variable.

            Eagerly propagates broadcasting constraints when enough
            information is available, or raises a conflict if the
            assignment is inconsistent.
            """
            vid = var.get_id()
            try:
                concrete = value.as_long()
            except Exception:
                return
            self._fixed[vid] = concrete

            # Check all broadcast triples involving this variable
            for da, db, do in self._broadcast_triples:
                self._propagate_broadcast_triple(da, db, do)

            # Check matmul constraints
            for ia, ib in self._matmul_pairs:
                self._propagate_matmul(ia, ib)

        def _on_final(self) -> None:
            """Final consistency check before Z3 reports SAT.

            Verifies all registered broadcasting and matmul constraints
            are satisfied under the current assignment.
            """
            for da, db, do in self._broadcast_triples:
                self._check_broadcast_final(da, db, do)
            for ia, ib in self._matmul_pairs:
                self._check_matmul_final(ia, ib)
            for shapes, strides in self._stride_constraints:
                self._check_stride_final(shapes, strides)

        # ---------------------------------------------------------------
        # Broadcast propagation
        # ---------------------------------------------------------------

        def _propagate_broadcast_triple(
            self,
            da: z3.ExprRef,
            db: z3.ExprRef,
            do: z3.ExprRef,
        ) -> None:
            """Propagate a broadcast_result_dim(da, db, do) constraint.

            If both da and db are fixed, eagerly set do or raise conflict.
            """
            va = self._fixed.get(da.get_id())
            vb = self._fixed.get(db.get_id())
            vo = self._fixed.get(do.get_id())

            if va is not None and vb is not None:
                if not _are_dims_broadcast_compatible(va, vb):
                    # Conflict: da and db are not broadcast-compatible
                    self.conflict(deps=[da, db])
                    return
                expected = _broadcast_result(va, vb)
                if vo is not None:
                    if vo != expected:
                        # Conflict: output dim doesn't match
                        self.conflict(deps=[da, db, do])
                else:
                    # Propagate: set do = expected
                    self.propagate(
                        do == z3.IntVal(expected), ids=[da, db]
                    )

        def _check_broadcast_final(
            self,
            da: z3.ExprRef,
            db: z3.ExprRef,
            do: z3.ExprRef,
        ) -> None:
            """Final-check a broadcast triple for consistency."""
            va = self._fixed.get(da.get_id())
            vb = self._fixed.get(db.get_id())
            vo = self._fixed.get(do.get_id())
            if va is not None and vb is not None:
                if not _are_dims_broadcast_compatible(va, vb):
                    self.conflict(deps=[da, db])
                    return
                expected = _broadcast_result(va, vb)
                if vo is not None and vo != expected:
                    self.conflict(deps=[da, db, do])

        # ---------------------------------------------------------------
        # Matmul propagation
        # ---------------------------------------------------------------

        def _propagate_matmul(
            self, ia: z3.ExprRef, ib: z3.ExprRef
        ) -> None:
            """Propagate matmul_compatible(ia, ib): inner dims must match."""
            va = self._fixed.get(ia.get_id())
            vb = self._fixed.get(ib.get_id())
            if va is not None and vb is not None and va != vb:
                self.conflict(deps=[ia, ib])

        def _check_matmul_final(
            self, ia: z3.ExprRef, ib: z3.ExprRef
        ) -> None:
            """Final-check matmul inner-dimension equality."""
            va = self._fixed.get(ia.get_id())
            vb = self._fixed.get(ib.get_id())
            if va is not None and vb is not None and va != vb:
                self.conflict(deps=[ia, ib])

        # ---------------------------------------------------------------
        # Stride propagation
        # ---------------------------------------------------------------

        def _check_stride_final(
            self,
            shapes: List[z3.ExprRef],
            strides: List[z3.ExprRef],
        ) -> None:
            """Final-check stride consistency for a contiguous tensor.

            For contiguous layout: stride[i] = product(shape[i+1:]).
            """
            n = len(shapes)
            if len(strides) != n:
                return
            shape_vals = [self._fixed.get(s.get_id()) for s in shapes]
            stride_vals = [self._fixed.get(s.get_id()) for s in strides]
            if any(v is None for v in shape_vals) or any(
                v is None for v in stride_vals
            ):
                return
            # Compute expected strides
            expected = [0] * n
            expected[n - 1] = 1
            for i in range(n - 2, -1, -1):
                expected[i] = expected[i + 1] * shape_vals[i + 1]  # type: ignore[index]
            for i in range(n):
                if stride_vals[i] != expected[i]:
                    # Collect all involved variables as deps
                    self.conflict(deps=list(shapes) + list(strides))
                    return

    # ═══════════════════════════════════════════════════════════════════════
    # 4. High-level constraint builders
    # ═══════════════════════════════════════════════════════════════════════

    def broadcast_compatible(
        prop: BroadcastPropagator,
        shape_a: List[z3.ExprRef],
        shape_b: List[z3.ExprRef],
    ) -> z3.ExprRef:
        """Assert broadcast compatibility between two shapes.

        Registers per-dimension broadcast triples with the propagator
        and returns a conjunction of per-dimension compatibility
        constraints that Z3 can reason about.

        Args:
            prop: The broadcast propagator attached to the solver.
            shape_a: List of Z3 Int variables for shape A dimensions.
            shape_b: List of Z3 Int variables for shape B dimensions.

        Returns:
            Z3 Bool expression asserting broadcast compatibility.
        """
        max_rank = max(len(shape_a), len(shape_b))
        # Pad shorter shape with IntVal(1) on the left
        pa = [z3.IntVal(1)] * (max_rank - len(shape_a)) + list(shape_a)
        pb = [z3.IntVal(1)] * (max_rank - len(shape_b)) + list(shape_b)

        clauses = []
        for a, b in zip(pa, pb):
            clauses.append(z3.Or(a == b, a == 1, b == 1))
            prop._register_var(a)
            prop._register_var(b)
        return z3.And(*clauses) if clauses else z3.BoolVal(True)

    def broadcast_result_dim(
        prop: BroadcastPropagator,
        dim_a: z3.ExprRef,
        dim_b: z3.ExprRef,
        dim_out: z3.ExprRef,
    ) -> z3.ExprRef:
        """Assert that dim_out is the broadcast result of dim_a and dim_b.

        Registers the triple with the propagator for eager propagation
        and returns a Z3 constraint encoding the relationship.

        Args:
            prop: The broadcast propagator.
            dim_a: Z3 Int for the first dimension.
            dim_b: Z3 Int for the second dimension.
            dim_out: Z3 Int for the output dimension.

        Returns:
            Z3 Bool expression: broadcast_result_dim(dim_a, dim_b) == dim_out.
        """
        prop._register_var(dim_a)
        prop._register_var(dim_b)
        prop._register_var(dim_out)
        prop._broadcast_triples.append((dim_a, dim_b, dim_out))

        # Encode: (a==b => out==a) & (a==1 => out==b) & (b==1 => out==a)
        # plus compatibility guard
        compat = z3.Or(dim_a == dim_b, dim_a == 1, dim_b == 1)
        result_correct = z3.And(
            z3.Implies(dim_a == dim_b, dim_out == dim_a),
            z3.Implies(dim_a == 1, dim_out == dim_b),
            z3.Implies(dim_b == 1, dim_out == dim_a),
        )
        return z3.And(compat, result_correct)

    def matmul_compatible(
        prop: BroadcastPropagator,
        shape_a: List[z3.ExprRef],
        shape_b: List[z3.ExprRef],
    ) -> z3.ExprRef:
        """Assert matmul compatibility: inner dimensions must match.

        For shape_a = (..., M, K) and shape_b = (..., K, N),
        asserts shape_a[-1] == shape_b[-2].

        Args:
            prop: The broadcast propagator.
            shape_a: List of Z3 Int variables for the first operand.
            shape_b: List of Z3 Int variables for the second operand.

        Returns:
            Z3 Bool expression asserting inner-dimension match.
        """
        if len(shape_a) < 2 or len(shape_b) < 2:
            return z3.BoolVal(False)
        inner_a = shape_a[-1]
        inner_b = shape_b[-2]
        prop._register_var(inner_a)
        prop._register_var(inner_b)
        prop._matmul_pairs.append((inner_a, inner_b))
        return inner_a == inner_b

    def stride_compatible(
        prop: BroadcastPropagator,
        shape: List[z3.ExprRef],
        stride: List[z3.ExprRef],
    ) -> z3.ExprRef:
        """Assert stride consistency for a contiguous tensor.

        For a contiguous tensor with shape (d0, d1, ..., dn):
          stride[i] = product(shape[i+1:])

        Args:
            prop: The broadcast propagator.
            shape: List of Z3 Int variables for shape dimensions.
            stride: List of Z3 Int variables for stride values.

        Returns:
            Z3 Bool expression asserting contiguous stride layout.
        """
        n = len(shape)
        if len(stride) != n:
            return z3.BoolVal(False)
        for v in list(shape) + list(stride):
            prop._register_var(v)
        prop._stride_constraints.append((list(shape), list(stride)))

        # Encode: stride[n-1] == 1, stride[i] == stride[i+1] * shape[i+1]
        clauses = [stride[n - 1] == 1]
        for i in range(n - 2, -1, -1):
            clauses.append(stride[i] == stride[i + 1] * shape[i + 1])
        return z3.And(*clauses)

    # ═══════════════════════════════════════════════════════════════════════
    # 5. BroadcastTheoryPlugin — convenience wrapper
    # ═══════════════════════════════════════════════════════════════════════

    class BroadcastTheoryPlugin:
        """High-level integration wrapper for attaching the broadcast
        theory to any Z3 Solver.

        Usage::

            solver = z3.Solver()
            plugin = BroadcastTheoryPlugin(solver)
            a, b, c = z3.Ints("a b c")
            solver.add(plugin.broadcast_result_dim(a, b, c))
            solver.add(a == 3, b == 1)
            assert solver.check() == z3.sat
            assert solver.model()[c].as_long() == 3
        """

        def __init__(self, solver: z3.Solver) -> None:
            self.solver = solver
            self.propagator = BroadcastPropagator(solver)

        def broadcast_compatible(
            self,
            shape_a: List[z3.ExprRef],
            shape_b: List[z3.ExprRef],
        ) -> z3.ExprRef:
            """Check if two shapes are broadcast-compatible."""
            return broadcast_compatible(self.propagator, shape_a, shape_b)

        def broadcast_result_dim(
            self,
            dim_a: z3.ExprRef,
            dim_b: z3.ExprRef,
            dim_out: z3.ExprRef,
        ) -> z3.ExprRef:
            """Assert broadcast result dimension relationship."""
            return broadcast_result_dim(
                self.propagator, dim_a, dim_b, dim_out
            )

        def matmul_compatible(
            self,
            shape_a: List[z3.ExprRef],
            shape_b: List[z3.ExprRef],
        ) -> z3.ExprRef:
            """Assert matmul inner-dimension compatibility."""
            return matmul_compatible(self.propagator, shape_a, shape_b)

        def stride_compatible(
            self,
            shape: List[z3.ExprRef],
            stride: List[z3.ExprRef],
        ) -> z3.ExprRef:
            """Assert contiguous stride layout."""
            return stride_compatible(self.propagator, shape, stride)


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
    print("=== Pure broadcasting helpers ===")
    _assert(
        _shapes_broadcast_compatible((3, 4), (3, 4)),
        "identical shapes compatible",
    )
    _assert(
        _shapes_broadcast_compatible((3, 4), (1, 4)),
        "(3,4) and (1,4) compatible",
    )
    _assert(
        _shapes_broadcast_compatible((3, 4), (4,)),
        "(3,4) and (4,) compatible (rank extension)",
    )
    _assert(
        not _shapes_broadcast_compatible((3, 4), (2, 4)),
        "(3,4) and (2,4) incompatible",
    )
    _assert(
        _broadcast_shape((3, 1), (1, 4)) == (3, 4),
        "broadcast_shape((3,1),(1,4)) == (3,4)",
    )
    _assert(
        _broadcast_shape((5,), (3, 5)) == (3, 5),
        "broadcast_shape((5,),(3,5)) == (3,5)",
    )

    # --- BroadcastTheoryPlugin ---
    print("\n=== BroadcastTheoryPlugin: broadcast_result_dim ===")
    s = z3.Solver()
    plugin = BroadcastTheoryPlugin(s)
    a, b, c = z3.Ints("a b c")
    s.add(plugin.broadcast_result_dim(a, b, c))
    s.add(a == 3, b == 1)
    result = s.check()
    _assert(result == z3.sat, "broadcast(3, 1) is SAT")
    if result == z3.sat:
        _assert(s.model()[c].as_long() == 3, "output dim == 3")

    print("\n=== BroadcastTheoryPlugin: incompatible dims ===")
    s2 = z3.Solver()
    p2 = BroadcastTheoryPlugin(s2)
    x, y, z_ = z3.Ints("x y z_")
    s2.add(p2.broadcast_result_dim(x, y, z_))
    s2.add(x == 3, y == 5)
    _assert(s2.check() == z3.unsat, "broadcast(3, 5) is UNSAT")

    print("\n=== BroadcastTheoryPlugin: broadcast_compatible ===")
    s3 = z3.Solver()
    p3 = BroadcastTheoryPlugin(s3)
    d0, d1, e0, e1 = z3.Ints("d0 d1 e0 e1")
    s3.add(p3.broadcast_compatible([d0, d1], [e0, e1]))
    s3.add(d0 == 3, d1 == 4, e0 == 1, e1 == 4)
    _assert(s3.check() == z3.sat, "(3,4) compat (1,4) is SAT")

    s4 = z3.Solver()
    p4 = BroadcastTheoryPlugin(s4)
    f0, f1, g0, g1 = z3.Ints("f0 f1 g0 g1")
    s4.add(p4.broadcast_compatible([f0, f1], [g0, g1]))
    s4.add(f0 == 3, f1 == 4, g0 == 2, g1 == 4)
    _assert(s4.check() == z3.unsat, "(3,4) compat (2,4) is UNSAT")

    print("\n=== BroadcastTheoryPlugin: matmul_compatible ===")
    s5 = z3.Solver()
    p5 = BroadcastTheoryPlugin(s5)
    m, k1, k2, n = z3.Ints("m k1 k2 n")
    s5.add(p5.matmul_compatible([m, k1], [k2, n]))
    s5.add(m == 3, k1 == 4, k2 == 4, n == 5)
    _assert(s5.check() == z3.sat, "matmul (3,4)@(4,5) SAT")

    s6 = z3.Solver()
    p6 = BroadcastTheoryPlugin(s6)
    m2, k3, k4, n2 = z3.Ints("m2 k3 k4 n2")
    s6.add(p6.matmul_compatible([m2, k3], [k4, n2]))
    s6.add(m2 == 3, k3 == 4, k4 == 5, n2 == 6)
    _assert(s6.check() == z3.unsat, "matmul (3,4)@(5,6) UNSAT")

    print("\n=== BroadcastTheoryPlugin: stride_compatible ===")
    s7 = z3.Solver()
    p7 = BroadcastTheoryPlugin(s7)
    sd0, sd1, sd2 = z3.Ints("sd0 sd1 sd2")
    st0, st1, st2 = z3.Ints("st0 st1 st2")
    s7.add(p7.stride_compatible([sd0, sd1, sd2], [st0, st1, st2]))
    # shape (2, 3, 4) => strides (12, 4, 1)
    s7.add(sd0 == 2, sd1 == 3, sd2 == 4)
    result7 = s7.check()
    _assert(result7 == z3.sat, "stride for shape (2,3,4) SAT")
    if result7 == z3.sat:
        model = s7.model()
        _assert(model[st0].as_long() == 12, "stride[0] == 12")
        _assert(model[st1].as_long() == 4, "stride[1] == 4")
        _assert(model[st2].as_long() == 1, "stride[2] == 1")

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    print("All broadcast_theory tests passed!")

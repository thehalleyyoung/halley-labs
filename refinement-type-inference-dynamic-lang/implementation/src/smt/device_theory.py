"""
Custom Z3 Theory Plugin for Tensor Device Consistency Checking.

Implements device placement rules as a first-class Z3 theory using
the UserPropagateBase API.  Ensures tensors that interact in an
operation reside on the same device (unless an explicit device
transfer is present).

Device model:
  - Enumeration sort with values: CPU, CUDA_0, CUDA_1, CUDA_2, CUDA_3
  - same_device(a, b): a and b must be equal
  - transfer_device(in, out, target): out == target regardless of in
  - inherit_device(in, out): out == in (output inherits input device)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Z3 import with graceful fallback
# ---------------------------------------------------------------------------

try:
    import z3

    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

# ---------------------------------------------------------------------------
# Device enumeration constants
# ---------------------------------------------------------------------------

DEVICE_NAMES = ("CPU", "CUDA_0", "CUDA_1", "CUDA_2", "CUDA_3")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Trail for backtracking support
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class _DeviceTrailFrame:
    """Snapshot of device theory state at a push point."""

    fixed_vars: Dict[int, int]  # var_id -> concrete enum index


# ═══════════════════════════════════════════════════════════════════════════
# 2. DevicePropagator — Z3 UserPropagateBase implementation
# ═══════════════════════════════════════════════════════════════════════════

if HAS_Z3:

    # Lazy-init to avoid Z3 "sort already declared" on re-import
    DeviceSort: z3.SortRef = None  # type: ignore[assignment]
    DEVICE_VALS: Dict[str, z3.ExprRef] = {}

    def _ensure_device_sort() -> None:
        global DeviceSort, DEVICE_VALS
        if DeviceSort is not None:
            return
        # If module was already imported under its package name, reuse
        import sys
        _pkg = "src.smt.device_theory"
        if _pkg in sys.modules and hasattr(sys.modules[_pkg], "DeviceSort"):
            _other = sys.modules[_pkg]
            if _other.DeviceSort is not None:
                DeviceSort = _other.DeviceSort
                DEVICE_VALS.update(_other.DEVICE_VALS)
                return
        try:
            _sort, _device_consts = z3.EnumSort("Device", DEVICE_NAMES)
        except z3.z3types.Z3Exception:
            # Fallback: use uninterpreted sort + distinct constants
            _sort = z3.DeclareSort("DeviceFallback")
            _device_consts = [z3.Const(name, _sort) for name in DEVICE_NAMES]
        DeviceSort = _sort
        DEVICE_VALS.update(
            {name: val for name, val in zip(DEVICE_NAMES, _device_consts)}
        )

    _ensure_device_sort()

    class DevicePropagator(z3.UserPropagateBase):
        """Z3 theory propagator for tensor device placement constraints.

        Registers device variables and constraint pairs/triples.
        When Z3 fixes concrete values for device variables the
        propagator eagerly checks consistency and raises conflicts
        on violations.
        """

        def __init__(self, s: z3.Solver) -> None:
            super().__init__(s)

            # var tracking: z3 expr id -> z3 expr
            self._vars: Dict[int, z3.ExprRef] = {}
            # fixed assignments: z3 expr id -> concrete enum index
            self._fixed: Dict[int, int] = {}
            # backtracking trail
            self._trail: List[_DeviceTrailFrame] = []

            # Constraints
            # same_device: (dev_a, dev_b) pairs that must be equal
            self._same_device_pairs: List[
                Tuple[z3.ExprRef, z3.ExprRef]
            ] = []
            # transfer: (dev_in, dev_out, target_const) — dev_out == target
            self._transfer_triples: List[
                Tuple[z3.ExprRef, z3.ExprRef, z3.ExprRef]
            ] = []
            # inherit: (dev_in, dev_out) — dev_out == dev_in
            self._inherit_pairs: List[
                Tuple[z3.ExprRef, z3.ExprRef]
            ] = []

            # Register callbacks
            self.add_fixed(self._on_fixed)
            self.add_final(self._on_final)
            self.add_created(self._on_created)

        # ---------------------------------------------------------------
        # Variable registration
        # ---------------------------------------------------------------

        def _register_var(self, v: z3.ExprRef) -> None:
            """Register a Z3 Device variable with the propagator."""
            vid = v.get_id()
            if vid not in self._vars:
                self._vars[vid] = v
                self.add(v)

        # ---------------------------------------------------------------
        # Backtracking: push / pop
        # ---------------------------------------------------------------

        def push(self) -> None:
            """Save current state for backtracking."""
            self._trail.append(_DeviceTrailFrame(fixed_vars=dict(self._fixed)))

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

            Eagerly propagates device constraints when enough
            information is available, or raises a conflict if the
            assignment is inconsistent.
            """
            vid = var.get_id()
            self._fixed[vid] = value.get_id()

            # Check same-device constraints
            for da, db in self._same_device_pairs:
                self._propagate_same_device(da, db)

            # Check transfer constraints
            for di, do, tgt in self._transfer_triples:
                self._propagate_transfer(di, do, tgt)

            # Check inherit constraints
            for di, do in self._inherit_pairs:
                self._propagate_inherit(di, do)

        def _on_final(self) -> None:
            """Final consistency check before Z3 reports SAT."""
            for da, db in self._same_device_pairs:
                self._check_same_device_final(da, db)
            for di, do, tgt in self._transfer_triples:
                self._check_transfer_final(di, do, tgt)
            for di, do in self._inherit_pairs:
                self._check_inherit_final(di, do)

        # ---------------------------------------------------------------
        # Same-device propagation
        # ---------------------------------------------------------------

        def _propagate_same_device(
            self, da: z3.ExprRef, db: z3.ExprRef
        ) -> None:
            """If both are fixed, they must be equal."""
            va = self._fixed.get(da.get_id())
            vb = self._fixed.get(db.get_id())
            if va is not None and vb is not None and va != vb:
                self.conflict(deps=[da, db])

        def _check_same_device_final(
            self, da: z3.ExprRef, db: z3.ExprRef
        ) -> None:
            va = self._fixed.get(da.get_id())
            vb = self._fixed.get(db.get_id())
            if va is not None and vb is not None and va != vb:
                self.conflict(deps=[da, db])

        # ---------------------------------------------------------------
        # Transfer propagation
        # ---------------------------------------------------------------

        def _propagate_transfer(
            self,
            dev_in: z3.ExprRef,
            dev_out: z3.ExprRef,
            target: z3.ExprRef,
        ) -> None:
            """Output device must equal the transfer target."""
            vo = self._fixed.get(dev_out.get_id())
            vt = self._fixed.get(target.get_id())
            if vo is not None and vt is not None and vo != vt:
                self.conflict(deps=[dev_out, target])

        def _check_transfer_final(
            self,
            dev_in: z3.ExprRef,
            dev_out: z3.ExprRef,
            target: z3.ExprRef,
        ) -> None:
            vo = self._fixed.get(dev_out.get_id())
            vt = self._fixed.get(target.get_id())
            if vo is not None and vt is not None and vo != vt:
                self.conflict(deps=[dev_out, target])

        # ---------------------------------------------------------------
        # Inherit propagation
        # ---------------------------------------------------------------

        def _propagate_inherit(
            self, dev_in: z3.ExprRef, dev_out: z3.ExprRef
        ) -> None:
            """Output device must equal input device."""
            vi = self._fixed.get(dev_in.get_id())
            vo = self._fixed.get(dev_out.get_id())
            if vi is not None and vo is not None and vi != vo:
                self.conflict(deps=[dev_in, dev_out])

        def _check_inherit_final(
            self, dev_in: z3.ExprRef, dev_out: z3.ExprRef
        ) -> None:
            vi = self._fixed.get(dev_in.get_id())
            vo = self._fixed.get(dev_out.get_id())
            if vi is not None and vo is not None and vi != vo:
                self.conflict(deps=[dev_in, dev_out])

    # ═══════════════════════════════════════════════════════════════════════
    # 3. High-level constraint builders
    # ═══════════════════════════════════════════════════════════════════════

    def same_device(
        prop: DevicePropagator,
        dev_a: z3.ExprRef,
        dev_b: z3.ExprRef,
    ) -> z3.ExprRef:
        """Assert that two tensors are on the same device.

        Args:
            prop: The device propagator.
            dev_a: Z3 Device variable for the first tensor.
            dev_b: Z3 Device variable for the second tensor.

        Returns:
            Z3 Bool: dev_a == dev_b.
        """
        prop._register_var(dev_a)
        prop._register_var(dev_b)
        prop._same_device_pairs.append((dev_a, dev_b))
        return dev_a == dev_b

    def transfer_device(
        prop: DevicePropagator,
        dev_in: z3.ExprRef,
        dev_out: z3.ExprRef,
        target: z3.ExprRef,
    ) -> z3.ExprRef:
        """Assert a device transfer: output gets the target device.

        Models `.to(device)`, `.cuda()`, `.cpu()` operations.

        Args:
            prop: The device propagator.
            dev_in: Z3 Device variable for the input tensor.
            dev_out: Z3 Device variable for the output tensor.
            target: Z3 Device constant for the target device.

        Returns:
            Z3 Bool: dev_out == target.
        """
        prop._register_var(dev_in)
        prop._register_var(dev_out)
        prop._register_var(target)
        prop._transfer_triples.append((dev_in, dev_out, target))
        return dev_out == target

    def inherit_device(
        prop: DevicePropagator,
        dev_in: z3.ExprRef,
        dev_out: z3.ExprRef,
    ) -> z3.ExprRef:
        """Assert device inheritance: output inherits input's device.

        Args:
            prop: The device propagator.
            dev_in: Z3 Device variable for the input tensor.
            dev_out: Z3 Device variable for the output tensor.

        Returns:
            Z3 Bool: dev_out == dev_in.
        """
        prop._register_var(dev_in)
        prop._register_var(dev_out)
        prop._inherit_pairs.append((dev_in, dev_out))
        return dev_out == dev_in

    # ═══════════════════════════════════════════════════════════════════════
    # 4. DeviceTheoryPlugin — convenience wrapper
    # ═══════════════════════════════════════════════════════════════════════

    class DeviceTheoryPlugin:
        """High-level integration wrapper for attaching the device
        theory to any Z3 Solver.

        Usage::

            solver = z3.Solver()
            plugin = DeviceTheoryPlugin(solver)
            a = z3.Const("a", DeviceSort)
            b = z3.Const("b", DeviceSort)
            solver.add(plugin.same_device(a, b))
            solver.add(a == DEVICE_VALS["CPU"])
            assert solver.check() == z3.sat
            assert solver.model()[b] == DEVICE_VALS["CPU"]
        """

        def __init__(self, solver: z3.Solver) -> None:
            self.solver = solver
            self.propagator = DevicePropagator(solver)

        def same_device(
            self,
            dev_a: z3.ExprRef,
            dev_b: z3.ExprRef,
        ) -> z3.ExprRef:
            """Assert two tensors must be on the same device."""
            return same_device(self.propagator, dev_a, dev_b)

        def transfer_device(
            self,
            dev_in: z3.ExprRef,
            dev_out: z3.ExprRef,
            target: z3.ExprRef,
        ) -> z3.ExprRef:
            """Assert a device transfer operation."""
            return transfer_device(self.propagator, dev_in, dev_out, target)

        def inherit_device(
            self,
            dev_in: z3.ExprRef,
            dev_out: z3.ExprRef,
        ) -> z3.ExprRef:
            """Assert device inheritance from input to output."""
            return inherit_device(self.propagator, dev_in, dev_out)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Self-test
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

    # --- same_device ---
    print("=== DeviceTheoryPlugin: same_device ===")
    s = z3.Solver()
    plugin = DeviceTheoryPlugin(s)
    a = z3.Const("a", DeviceSort)
    b = z3.Const("b", DeviceSort)
    s.add(plugin.same_device(a, b))
    s.add(a == DEVICE_VALS["CPU"])
    result = s.check()
    _assert(result == z3.sat, "same_device(CPU, ?) SAT")
    if result == z3.sat:
        _assert(
            s.model()[b] == DEVICE_VALS["CPU"],
            "b inferred as CPU",
        )

    print("\n=== DeviceTheoryPlugin: same_device conflict ===")
    s2 = z3.Solver()
    p2 = DeviceTheoryPlugin(s2)
    c = z3.Const("c", DeviceSort)
    d = z3.Const("d", DeviceSort)
    s2.add(p2.same_device(c, d))
    s2.add(c == DEVICE_VALS["CPU"], d == DEVICE_VALS["CUDA_0"])
    _assert(s2.check() == z3.unsat, "same_device(CPU, CUDA_0) UNSAT")

    # --- transfer_device ---
    print("\n=== DeviceTheoryPlugin: transfer_device ===")
    s3 = z3.Solver()
    p3 = DeviceTheoryPlugin(s3)
    e = z3.Const("e", DeviceSort)
    f = z3.Const("f", DeviceSort)
    s3.add(p3.transfer_device(e, f, DEVICE_VALS["CUDA_1"]))
    s3.add(e == DEVICE_VALS["CPU"])
    result3 = s3.check()
    _assert(result3 == z3.sat, "transfer CPU -> CUDA_1 SAT")
    if result3 == z3.sat:
        _assert(
            s3.model()[f] == DEVICE_VALS["CUDA_1"],
            "output device == CUDA_1",
        )

    # Transfer with wrong output should be UNSAT
    print("\n=== DeviceTheoryPlugin: transfer conflict ===")
    s4 = z3.Solver()
    p4 = DeviceTheoryPlugin(s4)
    g = z3.Const("g", DeviceSort)
    h = z3.Const("h", DeviceSort)
    s4.add(p4.transfer_device(g, h, DEVICE_VALS["CUDA_0"]))
    s4.add(g == DEVICE_VALS["CPU"], h == DEVICE_VALS["CUDA_1"])
    _assert(s4.check() == z3.unsat, "transfer target CUDA_0 but out CUDA_1 UNSAT")

    # --- inherit_device ---
    print("\n=== DeviceTheoryPlugin: inherit_device ===")
    s5 = z3.Solver()
    p5 = DeviceTheoryPlugin(s5)
    i = z3.Const("i", DeviceSort)
    j = z3.Const("j", DeviceSort)
    s5.add(p5.inherit_device(i, j))
    s5.add(i == DEVICE_VALS["CUDA_2"])
    result5 = s5.check()
    _assert(result5 == z3.sat, "inherit CUDA_2 SAT")
    if result5 == z3.sat:
        _assert(
            s5.model()[j] == DEVICE_VALS["CUDA_2"],
            "output inherits CUDA_2",
        )

    # Inherit conflict
    print("\n=== DeviceTheoryPlugin: inherit conflict ===")
    s6 = z3.Solver()
    p6 = DeviceTheoryPlugin(s6)
    k = z3.Const("k", DeviceSort)
    l = z3.Const("l", DeviceSort)
    s6.add(p6.inherit_device(k, l))
    s6.add(k == DEVICE_VALS["CUDA_0"], l == DEVICE_VALS["CPU"])
    _assert(s6.check() == z3.unsat, "inherit CUDA_0 but out CPU UNSAT")

    # --- Chain: transfer then same_device ---
    print("\n=== DeviceTheoryPlugin: transfer chain ===")
    s7 = z3.Solver()
    p7 = DeviceTheoryPlugin(s7)
    t1 = z3.Const("t1", DeviceSort)
    t2 = z3.Const("t2", DeviceSort)
    t3 = z3.Const("t3", DeviceSort)
    # t1 on CPU, transfer to CUDA_0 giving t2, then t2 and t3 must match
    s7.add(p7.transfer_device(t1, t2, DEVICE_VALS["CUDA_0"]))
    s7.add(p7.same_device(t2, t3))
    s7.add(t1 == DEVICE_VALS["CPU"])
    result7 = s7.check()
    _assert(result7 == z3.sat, "transfer chain SAT")
    if result7 == z3.sat:
        _assert(
            s7.model()[t3] == DEVICE_VALS["CUDA_0"],
            "t3 inferred as CUDA_0 via chain",
        )

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    print("All device_theory tests passed!")

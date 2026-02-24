"""
Custom Z3 Theory Plugin for Train/Eval Phase Tracking.

Implements phase-dependent behaviour rules as a first-class Z3 theory
using the UserPropagateBase API.  Tracks whether a model is in TRAIN
or EVAL phase and propagates phase-dependent semantics:

  - Dropout: identity in EVAL (output == input), may differ in TRAIN.
  - BatchNorm: uses running statistics in EVAL, batch statistics in TRAIN.

Phase model:
  - Z3 Bool: True == TRAIN, False == EVAL
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

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
# 1. Trail for backtracking support
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class _PhaseTrailFrame:
    """Snapshot of phase theory state at a push point."""

    fixed_vars: Dict[int, bool]  # var_id -> concrete bool value


# ═══════════════════════════════════════════════════════════════════════════
# 2. PhasePropagator — Z3 UserPropagateBase implementation
# ═══════════════════════════════════════════════════════════════════════════

if HAS_Z3:

    class PhasePropagator(z3.UserPropagateBase):
        """Z3 theory propagator for train/eval phase constraints.

        Registers phase variables and phase-dependent behaviour
        constraints.  When Z3 fixes a phase variable the propagator
        eagerly checks consistency and raises conflicts on violations.
        """

        def __init__(self, s: z3.Solver) -> None:
            super().__init__(s)

            # var tracking: z3 expr id -> z3 expr
            self._vars: Dict[int, z3.ExprRef] = {}
            # fixed assignments: z3 expr id -> concrete bool
            self._fixed: Dict[int, bool] = {}
            # backtracking trail
            self._trail: List[_PhaseTrailFrame] = []

            # Constraints
            # set_phase: (phase_var, is_train_bool)
            self._phase_assignments: List[
                Tuple[z3.ExprRef, bool]
            ] = []
            # dropout: (phase_var, input_active, output_active)
            self._dropout_constraints: List[
                Tuple[z3.ExprRef, z3.ExprRef, z3.ExprRef]
            ] = []
            # batchnorm: (phase_var, uses_running_stats)
            self._batchnorm_constraints: List[
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
            """Register a Z3 Bool variable with the propagator."""
            vid = v.get_id()
            if vid not in self._vars:
                self._vars[vid] = v
                self.add(v)

        # ---------------------------------------------------------------
        # Backtracking: push / pop
        # ---------------------------------------------------------------

        def push(self) -> None:
            """Save current state for backtracking."""
            self._trail.append(_PhaseTrailFrame(fixed_vars=dict(self._fixed)))

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
            """Called when Z3 assigns a concrete value to a tracked variable."""
            vid = var.get_id()
            try:
                concrete = bool(z3.is_true(value))
            except Exception:
                return
            self._fixed[vid] = concrete

            # Check phase assignments
            for pv, is_train in self._phase_assignments:
                self._propagate_phase(pv, is_train)

            # Check dropout constraints
            for pv, inp, out in self._dropout_constraints:
                self._propagate_dropout(pv, inp, out)

            # Check batchnorm constraints
            for pv, urs in self._batchnorm_constraints:
                self._propagate_batchnorm(pv, urs)

        def _on_final(self) -> None:
            """Final consistency check before Z3 reports SAT."""
            for pv, is_train in self._phase_assignments:
                self._check_phase_final(pv, is_train)
            for pv, inp, out in self._dropout_constraints:
                self._check_dropout_final(pv, inp, out)
            for pv, urs in self._batchnorm_constraints:
                self._check_batchnorm_final(pv, urs)

        # ---------------------------------------------------------------
        # Phase assignment propagation
        # ---------------------------------------------------------------

        def _propagate_phase(
            self, phase_var: z3.ExprRef, is_train: bool
        ) -> None:
            val = self._fixed.get(phase_var.get_id())
            if val is not None and val != is_train:
                self.conflict(deps=[phase_var])

        def _check_phase_final(
            self, phase_var: z3.ExprRef, is_train: bool
        ) -> None:
            val = self._fixed.get(phase_var.get_id())
            if val is not None and val != is_train:
                self.conflict(deps=[phase_var])

        # ---------------------------------------------------------------
        # Dropout propagation
        # ---------------------------------------------------------------

        def _propagate_dropout(
            self,
            phase_var: z3.ExprRef,
            input_active: z3.ExprRef,
            output_active: z3.ExprRef,
        ) -> None:
            """In EVAL mode, output_active must equal input_active (identity).
            In TRAIN mode, no additional constraint (dropout may zero out).
            """
            phase = self._fixed.get(phase_var.get_id())
            if phase is None:
                return
            # EVAL mode (phase == False): output must equal input
            if not phase:
                vi = self._fixed.get(input_active.get_id())
                vo = self._fixed.get(output_active.get_id())
                if vi is not None and vo is not None and vi != vo:
                    self.conflict(
                        deps=[phase_var, input_active, output_active]
                    )

        def _check_dropout_final(
            self,
            phase_var: z3.ExprRef,
            input_active: z3.ExprRef,
            output_active: z3.ExprRef,
        ) -> None:
            phase = self._fixed.get(phase_var.get_id())
            if phase is None:
                return
            if not phase:
                vi = self._fixed.get(input_active.get_id())
                vo = self._fixed.get(output_active.get_id())
                if vi is not None and vo is not None and vi != vo:
                    self.conflict(
                        deps=[phase_var, input_active, output_active]
                    )

        # ---------------------------------------------------------------
        # BatchNorm propagation
        # ---------------------------------------------------------------

        def _propagate_batchnorm(
            self,
            phase_var: z3.ExprRef,
            uses_running_stats: z3.ExprRef,
        ) -> None:
            """In EVAL, uses_running_stats must be True.
            In TRAIN, uses_running_stats must be False.
            """
            phase = self._fixed.get(phase_var.get_id())
            urs = self._fixed.get(uses_running_stats.get_id())
            if phase is None or urs is None:
                return
            # EVAL => uses_running_stats==True; TRAIN => uses_running_stats==False
            expected = not phase  # eval(False)->True, train(True)->False
            if urs != expected:
                self.conflict(deps=[phase_var, uses_running_stats])

        def _check_batchnorm_final(
            self,
            phase_var: z3.ExprRef,
            uses_running_stats: z3.ExprRef,
        ) -> None:
            phase = self._fixed.get(phase_var.get_id())
            urs = self._fixed.get(uses_running_stats.get_id())
            if phase is None or urs is None:
                return
            expected = not phase
            if urs != expected:
                self.conflict(deps=[phase_var, uses_running_stats])

    # ═══════════════════════════════════════════════════════════════════════
    # 3. High-level constraint builders
    # ═══════════════════════════════════════════════════════════════════════

    def set_phase(
        prop: PhasePropagator,
        phase_var: z3.ExprRef,
        is_train: bool,
    ) -> z3.ExprRef:
        """Assert the phase of the model.

        Args:
            prop: The phase propagator.
            phase_var: Z3 Bool variable representing the phase.
            is_train: True for TRAIN, False for EVAL.

        Returns:
            Z3 Bool: phase_var == is_train.
        """
        prop._register_var(phase_var)
        prop._phase_assignments.append((phase_var, is_train))
        return phase_var == z3.BoolVal(is_train)

    def dropout_behavior(
        prop: PhasePropagator,
        phase_var: z3.ExprRef,
        input_active: z3.ExprRef,
        output_active: z3.ExprRef,
    ) -> z3.ExprRef:
        """Assert dropout phase-dependent behaviour.

        In EVAL: output_active == input_active (identity).
        In TRAIN: no constraint on output_active (may differ).

        Args:
            prop: The phase propagator.
            phase_var: Z3 Bool (True=TRAIN, False=EVAL).
            input_active: Z3 Bool for input activation state.
            output_active: Z3 Bool for output activation state.

        Returns:
            Z3 Bool encoding the phase-dependent dropout constraint.
        """
        prop._register_var(phase_var)
        prop._register_var(input_active)
        prop._register_var(output_active)
        prop._dropout_constraints.append(
            (phase_var, input_active, output_active)
        )
        # In EVAL (phase==False), output must equal input
        return z3.Implies(
            z3.Not(phase_var), output_active == input_active
        )

    def batchnorm_behavior(
        prop: PhasePropagator,
        phase_var: z3.ExprRef,
        uses_running_stats: z3.ExprRef,
    ) -> z3.ExprRef:
        """Assert batchnorm phase-dependent behaviour.

        In EVAL: uses_running_stats must be True.
        In TRAIN: uses_running_stats must be False.

        Args:
            prop: The phase propagator.
            phase_var: Z3 Bool (True=TRAIN, False=EVAL).
            uses_running_stats: Z3 Bool for batchnorm stat source.

        Returns:
            Z3 Bool encoding the phase-dependent batchnorm constraint.
        """
        prop._register_var(phase_var)
        prop._register_var(uses_running_stats)
        prop._batchnorm_constraints.append((phase_var, uses_running_stats))
        # EVAL => running stats; TRAIN => batch stats
        return z3.And(
            z3.Implies(z3.Not(phase_var), uses_running_stats),
            z3.Implies(phase_var, z3.Not(uses_running_stats)),
        )

    # ═══════════════════════════════════════════════════════════════════════
    # 4. PhaseTheoryPlugin — convenience wrapper
    # ═══════════════════════════════════════════════════════════════════════

    class PhaseTheoryPlugin:
        """High-level integration wrapper for attaching the phase
        theory to any Z3 Solver.

        Usage::

            solver = z3.Solver()
            plugin = PhaseTheoryPlugin(solver)
            phase = z3.Bool("phase")
            solver.add(plugin.set_phase(phase, is_train=False))
            urs = z3.Bool("urs")
            solver.add(plugin.batchnorm_behavior(phase, urs))
            assert solver.check() == z3.sat
            assert z3.is_true(solver.model()[urs])
        """

        def __init__(self, solver: z3.Solver) -> None:
            self.solver = solver
            self.propagator = PhasePropagator(solver)

        def set_phase(
            self,
            phase_var: z3.ExprRef,
            is_train: bool,
        ) -> z3.ExprRef:
            """Assert the model phase."""
            return set_phase(self.propagator, phase_var, is_train)

        def dropout_behavior(
            self,
            phase_var: z3.ExprRef,
            input_active: z3.ExprRef,
            output_active: z3.ExprRef,
        ) -> z3.ExprRef:
            """Assert dropout phase-dependent behaviour."""
            return dropout_behavior(
                self.propagator, phase_var, input_active, output_active
            )

        def batchnorm_behavior(
            self,
            phase_var: z3.ExprRef,
            uses_running_stats: z3.ExprRef,
        ) -> z3.ExprRef:
            """Assert batchnorm phase-dependent behaviour."""
            return batchnorm_behavior(
                self.propagator, phase_var, uses_running_stats
            )


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

    # --- set_phase ---
    print("=== PhaseTheoryPlugin: set_phase TRAIN ===")
    s = z3.Solver()
    plugin = PhaseTheoryPlugin(s)
    phase = z3.Bool("phase")
    s.add(plugin.set_phase(phase, is_train=True))
    result = s.check()
    _assert(result == z3.sat, "set_phase(TRAIN) SAT")
    if result == z3.sat:
        _assert(z3.is_true(s.model()[phase]), "phase == True (TRAIN)")

    print("\n=== PhaseTheoryPlugin: set_phase EVAL ===")
    s2 = z3.Solver()
    p2 = PhaseTheoryPlugin(s2)
    phase2 = z3.Bool("phase2")
    s2.add(p2.set_phase(phase2, is_train=False))
    result2 = s2.check()
    _assert(result2 == z3.sat, "set_phase(EVAL) SAT")
    if result2 == z3.sat:
        _assert(z3.is_false(s2.model()[phase2]), "phase == False (EVAL)")

    # --- dropout EVAL: output must equal input ---
    print("\n=== PhaseTheoryPlugin: dropout EVAL identity ===")
    s3 = z3.Solver()
    p3 = PhaseTheoryPlugin(s3)
    ph3 = z3.Bool("ph3")
    inp3 = z3.Bool("inp3")
    out3 = z3.Bool("out3")
    s3.add(p3.set_phase(ph3, is_train=False))
    s3.add(p3.dropout_behavior(ph3, inp3, out3))
    s3.add(inp3 == True)
    result3 = s3.check()
    _assert(result3 == z3.sat, "dropout EVAL with input=True SAT")
    if result3 == z3.sat:
        _assert(z3.is_true(s3.model()[out3]), "dropout EVAL: output==input")

    # dropout EVAL conflict: input != output
    print("\n=== PhaseTheoryPlugin: dropout EVAL conflict ===")
    s4 = z3.Solver()
    p4 = PhaseTheoryPlugin(s4)
    ph4 = z3.Bool("ph4")
    inp4 = z3.Bool("inp4")
    out4 = z3.Bool("out4")
    s4.add(p4.set_phase(ph4, is_train=False))
    s4.add(p4.dropout_behavior(ph4, inp4, out4))
    s4.add(inp4 == True, out4 == False)
    _assert(s4.check() == z3.unsat, "dropout EVAL input!=output UNSAT")

    # dropout TRAIN: output may differ from input
    print("\n=== PhaseTheoryPlugin: dropout TRAIN ===")
    s5 = z3.Solver()
    p5 = PhaseTheoryPlugin(s5)
    ph5 = z3.Bool("ph5")
    inp5 = z3.Bool("inp5")
    out5 = z3.Bool("out5")
    s5.add(p5.set_phase(ph5, is_train=True))
    s5.add(p5.dropout_behavior(ph5, inp5, out5))
    s5.add(inp5 == True, out5 == False)
    _assert(s5.check() == z3.sat, "dropout TRAIN input!=output SAT")

    # --- batchnorm EVAL: must use running stats ---
    print("\n=== PhaseTheoryPlugin: batchnorm EVAL ===")
    s6 = z3.Solver()
    p6 = PhaseTheoryPlugin(s6)
    ph6 = z3.Bool("ph6")
    urs6 = z3.Bool("urs6")
    s6.add(p6.set_phase(ph6, is_train=False))
    s6.add(p6.batchnorm_behavior(ph6, urs6))
    result6 = s6.check()
    _assert(result6 == z3.sat, "batchnorm EVAL SAT")
    if result6 == z3.sat:
        _assert(
            z3.is_true(s6.model()[urs6]),
            "batchnorm EVAL: uses_running_stats==True",
        )

    # batchnorm TRAIN: must NOT use running stats
    print("\n=== PhaseTheoryPlugin: batchnorm TRAIN ===")
    s7 = z3.Solver()
    p7 = PhaseTheoryPlugin(s7)
    ph7 = z3.Bool("ph7")
    urs7 = z3.Bool("urs7")
    s7.add(p7.set_phase(ph7, is_train=True))
    s7.add(p7.batchnorm_behavior(ph7, urs7))
    result7 = s7.check()
    _assert(result7 == z3.sat, "batchnorm TRAIN SAT")
    if result7 == z3.sat:
        _assert(
            z3.is_false(s7.model()[urs7]),
            "batchnorm TRAIN: uses_running_stats==False",
        )

    # batchnorm EVAL conflict: uses_running_stats==False
    print("\n=== PhaseTheoryPlugin: batchnorm EVAL conflict ===")
    s8 = z3.Solver()
    p8 = PhaseTheoryPlugin(s8)
    ph8 = z3.Bool("ph8")
    urs8 = z3.Bool("urs8")
    s8.add(p8.set_phase(ph8, is_train=False))
    s8.add(p8.batchnorm_behavior(ph8, urs8))
    s8.add(urs8 == False)
    _assert(s8.check() == z3.unsat, "batchnorm EVAL running_stats=False UNSAT")

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    print("All phase_theory tests passed!")

"""
K-Induction Engine for Tensor Shape Verification.

Implements the k-induction algorithm as a complement to IC3/PDR and BMC.
K-induction combines bounded model checking (base case) with an induction
step: if no counterexample exists up to depth k AND any k consecutive
safe states imply the (k+1)-th is also safe, then the property holds
for all depths.

Reference: Sheeran, Singh, Stalmarck (2000) "Checking Safety Properties
Using Induction and a SAT-Solver"

For tensor shape verification:
  - Base case: check shape compatibility for chains of length 1..k
  - Induction step: assume k arbitrary consecutive layers are safe,
    prove the next layer is also safe
  - If both pass at depth k, the model is SAFE for all depths

Advantages over pure BMC:
  - Can prove unbounded safety (not just absence of bugs up to depth k)
  - Simpler than IC3/PDR (no frame management, no cube blocking)

Advantages of IC3/PDR over k-induction:
  - IC3/PDR finds minimal inductive invariants
  - IC3/PDR typically converges faster (fewer Z3 queries)
  - IC3/PDR handles non-inductive properties via frame strengthening
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

from src.model_checker import (
    ComputationGraph,
    ComputationStep,
    ConstraintVerifier,
    Device,
    LayerDef,
    ModelState,
    OpKind,
    Phase,
    SafetyCertificate,
    SafetyViolation,
    VerificationResult,
    extract_computation_graph,
)
from src.tensor_shapes import TensorShape, ShapeDim

from src.ic3_pdr import ShapeTransitionSystem


class KInductionVerdict(Enum):
    SAFE = auto()
    UNSAFE = auto()
    UNKNOWN = auto()


@dataclass
class KInductionResult:
    """Result of k-induction verification.

    Attributes
    ----------
    verdict : KInductionVerdict
        SAFE if k-induction proof succeeded, UNSAFE if base case found
        counterexample, UNKNOWN if max_k reached without conclusion.
    k : int
        The depth at which the proof succeeded or a counterexample was found.
    verification_time_ms : float
        Wall-clock time in milliseconds.
    z3_queries : int
        Total number of Z3 solver queries.
    base_case_time_ms : float
        Time spent on base case checks.
    induction_step_time_ms : float
        Time spent on induction step checks.
    symbolic_dims : dict
        Symbolic dimensions used.
    invariant_description : str or None
        Human-readable description of the inductive argument.
    """
    verdict: KInductionVerdict
    k: int = 0
    verification_time_ms: float = 0.0
    z3_queries: int = 0
    base_case_time_ms: float = 0.0
    induction_step_time_ms: float = 0.0
    symbolic_dims: Dict[str, str] = field(default_factory=dict)
    invariant_description: Optional[str] = None


class KInductionSolver:
    """K-induction solver for tensor shape verification.

    Algorithm:
      for k = 1, 2, 3, ...:
        1. Base case: Check that no counterexample exists at depths 0..k-1
           (standard BMC)
        2. Induction step: Assume k consecutive steps are safe, check if
           the (k+1)-th step is also safe
        3. If both pass, property holds for all depths (return SAFE)
        4. If base case fails, return UNSAFE
        5. If induction step fails but base case passes, increment k
    """

    def __init__(
        self,
        transition_system: ShapeTransitionSystem,
        max_k: int = 50,
        solver_timeout_ms: int = 5000,
    ) -> None:
        if not HAS_Z3:
            raise RuntimeError("Z3 is required for k-induction")

        self.ts = transition_system
        self.max_k = max_k
        self.solver_timeout_ms = solver_timeout_ms
        self._z3_queries = 0
        self._base_time = 0.0
        self._ind_time = 0.0

    def solve(self) -> KInductionResult:
        """Run k-induction. Returns result with verdict.

        For acyclic computation graphs, the algorithm simplifies:
        we check whether init + all transitions + NOT(all safety) is SAT.
        If UNSAT, the model is safe for all symbolic dim values.
        If SAT, a counterexample exists.

        For k > 1, we strengthen the induction hypothesis by assuming
        the first k-1 safety constraints hold, helping prove deeper
        properties.
        """
        t0 = time.monotonic()

        init_constraints = self.ts.get_init_constraints()
        trans_constraints = self.ts.get_all_transition_constraints()
        all_safety = self.ts.get_all_safety_constraints()
        num_steps = self.ts.num_steps()

        for k in range(1, self.max_k + 1):
            # --- Base case: init + transitions entail safety for steps 0..k-1 ---
            base_t0 = time.monotonic()
            base_ok = self._check_base_case_k(
                init_constraints, trans_constraints, k, num_steps
            )
            self._base_time += (time.monotonic() - base_t0) * 1000

            if not base_ok:
                elapsed = (time.monotonic() - t0) * 1000
                return KInductionResult(
                    verdict=KInductionVerdict.UNSAFE,
                    k=k,
                    verification_time_ms=elapsed,
                    z3_queries=self._z3_queries,
                    base_case_time_ms=self._base_time,
                    induction_step_time_ms=self._ind_time,
                    symbolic_dims=self.ts.symbolic_dims,
                )

            # --- Induction step: transitions + hypothesis(0..k-1) => safety(k..n) ---
            ind_t0 = time.monotonic()
            ind_ok = self._check_induction_step_k(
                trans_constraints, k, num_steps
            )
            self._ind_time += (time.monotonic() - ind_t0) * 1000

            if ind_ok:
                elapsed = (time.monotonic() - t0) * 1000
                return KInductionResult(
                    verdict=KInductionVerdict.SAFE,
                    k=k,
                    verification_time_ms=elapsed,
                    z3_queries=self._z3_queries,
                    base_case_time_ms=self._base_time,
                    induction_step_time_ms=self._ind_time,
                    symbolic_dims=self.ts.symbolic_dims,
                    invariant_description=(
                        f"k-induction proof at k={k}: "
                        f"shape safety is {k}-inductive over "
                        f"{num_steps} computation steps"
                    ),
                )

            # If k >= num_steps, we've checked everything (base case covers all)
            if k >= num_steps:
                # Base case passed for all steps, so model is safe
                elapsed = (time.monotonic() - t0) * 1000
                return KInductionResult(
                    verdict=KInductionVerdict.SAFE,
                    k=k,
                    verification_time_ms=elapsed,
                    z3_queries=self._z3_queries,
                    base_case_time_ms=self._base_time,
                    induction_step_time_ms=self._ind_time,
                    symbolic_dims=self.ts.symbolic_dims,
                    invariant_description=(
                        f"k-induction proof at k={k}: "
                        f"base case covers all {num_steps} steps"
                    ),
                )

        elapsed = (time.monotonic() - t0) * 1000
        return KInductionResult(
            verdict=KInductionVerdict.UNKNOWN,
            k=self.max_k,
            verification_time_ms=elapsed,
            z3_queries=self._z3_queries,
            base_case_time_ms=self._base_time,
            induction_step_time_ms=self._ind_time,
            symbolic_dims=self.ts.symbolic_dims,
        )

    def _check_base_case_k(
        self,
        init_constraints: List[Any],
        trans_constraints: List[Any],
        k: int,
        num_steps: int,
    ) -> bool:
        """Check base case: init + transitions => safety for steps 0..k-1.

        Returns True if safe (no counterexample found).
        """
        solver = z3.Solver()
        solver.set("timeout", self.solver_timeout_ms)

        for c in init_constraints:
            solver.add(c)
        for c in trans_constraints:
            solver.add(c)

        # Negate safety for steps 0..min(k,num_steps)-1
        neg_safety = []
        for step_idx in range(min(k, num_steps)):
            for c in self.ts.get_safety_constraints(step_idx):
                neg_safety.append(z3.Not(c))

        if not neg_safety:
            return True

        solver.add(z3.Or(*neg_safety) if len(neg_safety) > 1 else neg_safety[0])

        self._z3_queries += 1
        result = solver.check()
        return result == z3.unsat

    def _check_induction_step_k(
        self,
        trans_constraints: List[Any],
        k: int,
        num_steps: int,
    ) -> bool:
        """Check induction step at depth k.

        Assume safety holds for steps 0..k-1 (the induction hypothesis).
        Prove safety holds for steps k..num_steps-1.

        Returns True if the induction step holds.
        """
        if k >= num_steps:
            return True  # Nothing left to prove

        solver = z3.Solver()
        solver.set("timeout", self.solver_timeout_ms)

        # Transitions define shape propagation
        for c in trans_constraints:
            solver.add(c)

        # Induction hypothesis: safety holds for steps 0..k-1
        for step_idx in range(min(k, num_steps)):
            for c in self.ts.get_safety_constraints(step_idx):
                solver.add(c)

        # Try to violate safety at steps k..num_steps-1
        neg_safety = []
        for step_idx in range(k, num_steps):
            for c in self.ts.get_safety_constraints(step_idx):
                neg_safety.append(z3.Not(c))

        if not neg_safety:
            return True

        solver.add(z3.Or(*neg_safety) if len(neg_safety) > 1 else neg_safety[0])

        self._z3_queries += 1
        result = solver.check()
        return result == z3.unsat


def k_induction_verify(
    model_source: str,
    symbolic_dims: Optional[Dict[str, str]] = None,
    input_shapes: Optional[Dict[str, tuple]] = None,
    max_k: int = 50,
    solver_timeout_ms: int = 5000,
) -> KInductionResult:
    """Run k-induction verification on a PyTorch nn.Module.

    Parameters
    ----------
    model_source : str
        Python source code containing an nn.Module subclass.
    symbolic_dims : dict, optional
        Mapping from shape position names to symbolic parameter names.
    input_shapes : dict, optional
        Mapping from forward parameter names to shape tuples.
    max_k : int
        Maximum induction depth.
    solver_timeout_ms : int
        Z3 solver timeout per query.

    Returns
    -------
    KInductionResult
        Contains verdict, k value, timing, and statistics.
    """
    t0 = time.monotonic()
    symbolic_dims = symbolic_dims or {}

    if not HAS_Z3:
        return KInductionResult(
            verdict=KInductionVerdict.UNKNOWN,
            verification_time_ms=(time.monotonic() - t0) * 1000,
            symbolic_dims=symbolic_dims,
        )

    try:
        graph = extract_computation_graph(model_source)
    except (ValueError, SyntaxError) as exc:
        logger.error("Failed to extract computation graph: %s", exc)
        return KInductionResult(
            verdict=KInductionVerdict.UNKNOWN,
            verification_time_ms=(time.monotonic() - t0) * 1000,
            symbolic_dims=symbolic_dims,
        )

    if input_shapes is None:
        input_shapes = {}
        for inp_name in graph.input_names:
            input_shapes[inp_name] = ("batch", 10)

    resolved_shapes: Dict[str, tuple] = {}
    for inp_name, shape in input_shapes.items():
        new_shape = []
        for dim_val in shape:
            if isinstance(dim_val, str) and dim_val in symbolic_dims:
                new_shape.append(symbolic_dims[dim_val])
            else:
                new_shape.append(dim_val)
        resolved_shapes[inp_name] = tuple(new_shape)

    try:
        ts = ShapeTransitionSystem(
            graph, resolved_shapes, symbolic_dims, solver_timeout_ms
        )
    except Exception as exc:
        logger.error("Failed to build transition system: %s", exc)
        return KInductionResult(
            verdict=KInductionVerdict.UNKNOWN,
            verification_time_ms=(time.monotonic() - t0) * 1000,
            symbolic_dims=symbolic_dims,
        )

    solver = KInductionSolver(ts, max_k=max_k, solver_timeout_ms=solver_timeout_ms)
    result = solver.solve()
    result.verification_time_ms = (time.monotonic() - t0) * 1000
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Comparative verification: IC3/PDR vs k-induction vs BMC
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MethodComparison:
    """Comparison of verification methods on a single model."""
    model_name: str
    ic3_safe: Optional[bool] = None
    ic3_time_ms: float = 0.0
    ic3_z3_queries: int = 0
    ic3_frames: int = 0
    k_ind_verdict: Optional[str] = None
    k_ind_time_ms: float = 0.0
    k_ind_z3_queries: int = 0
    k_ind_k: int = 0
    bmc_safe: Optional[bool] = None
    bmc_time_ms: float = 0.0
    bmc_z3_queries: int = 0
    agree: bool = True
    winner: str = ""  # fastest correct method

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "ic3": {"safe": self.ic3_safe, "time_ms": round(self.ic3_time_ms, 2),
                    "z3_queries": self.ic3_z3_queries, "frames": self.ic3_frames},
            "k_induction": {"verdict": self.k_ind_verdict, "time_ms": round(self.k_ind_time_ms, 2),
                           "z3_queries": self.k_ind_z3_queries, "k": self.k_ind_k},
            "bmc": {"safe": self.bmc_safe, "time_ms": round(self.bmc_time_ms, 2),
                    "z3_queries": self.bmc_z3_queries},
            "agree": self.agree,
            "winner": self.winner,
        }


def compare_verification_methods(
    model_source: str,
    model_name: str = "model",
    symbolic_dims: Optional[Dict[str, str]] = None,
    input_shapes: Optional[Dict[str, tuple]] = None,
    max_k: int = 50,
    solver_timeout_ms: int = 5000,
) -> MethodComparison:
    """Run IC3/PDR, k-induction, and BMC on same model, compare results.

    Returns a MethodComparison with timing, query counts, and agreement.
    """
    from src.ic3_pdr import ic3_verify
    from src.bmc_baseline import verify_model_bmc

    comp = MethodComparison(model_name=model_name)
    symbolic_dims = symbolic_dims or {}

    # IC3/PDR
    try:
        ic3_result = ic3_verify(
            model_source, symbolic_dims=symbolic_dims,
            input_shapes=input_shapes, solver_timeout_ms=solver_timeout_ms,
        )
        comp.ic3_safe = ic3_result.safe
        comp.ic3_time_ms = ic3_result.verification_time_ms
        comp.ic3_z3_queries = ic3_result.z3_queries
        comp.ic3_frames = ic3_result.frames_computed
    except Exception:
        comp.ic3_safe = None

    # K-induction
    try:
        ki_result = k_induction_verify(
            model_source, symbolic_dims=symbolic_dims,
            input_shapes=input_shapes, max_k=max_k,
            solver_timeout_ms=solver_timeout_ms,
        )
        comp.k_ind_verdict = ki_result.verdict.name
        comp.k_ind_time_ms = ki_result.verification_time_ms
        comp.k_ind_z3_queries = ki_result.z3_queries
        comp.k_ind_k = ki_result.k
    except Exception:
        comp.k_ind_verdict = None

    # BMC
    try:
        bmc_result = verify_model_bmc(
            model_source,
            input_shapes=input_shapes,
        )
        comp.bmc_safe = bmc_result.safe
        comp.bmc_time_ms = bmc_result.time_ms
        comp.bmc_z3_queries = bmc_result.z3_queries
    except Exception:
        comp.bmc_safe = None

    # Check agreement
    verdicts = []
    if comp.ic3_safe is not None:
        verdicts.append(("IC3", comp.ic3_safe))
    if comp.k_ind_verdict is not None:
        ki_safe = comp.k_ind_verdict == "SAFE"
        ki_unsafe = comp.k_ind_verdict == "UNSAFE"
        if ki_safe or ki_unsafe:
            verdicts.append(("k-ind", ki_safe))
    if comp.bmc_safe is not None:
        verdicts.append(("BMC", comp.bmc_safe))

    if len(verdicts) >= 2:
        comp.agree = all(v == verdicts[0][1] for _, v in verdicts)

    # Determine fastest correct method
    times = []
    if comp.ic3_safe is not None:
        times.append(("IC3/PDR", comp.ic3_time_ms))
    if comp.k_ind_verdict in ("SAFE", "UNSAFE"):
        times.append(("k-induction", comp.k_ind_time_ms))
    if comp.bmc_safe is not None:
        times.append(("BMC", comp.bmc_time_ms))
    if times:
        comp.winner = min(times, key=lambda x: x[1])[0]

    return comp

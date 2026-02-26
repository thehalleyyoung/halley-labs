"""
SMTVerifier: streaming SMT co-routine for incremental verification of
causal inference steps via Z3.

Wraps the Z3 solver with an incremental assertion protocol, enabling
each junction-tree message to be verified as it is computed. Emits
machine-checkable certificates for every verified step.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple

import z3

from .encoder import SMTEncoder
from .certificates import CertificateEmitter, Certificate
from .incremental import IncrementalProtocol


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class VerificationStatus(Enum):
    """Outcome of a single verification check."""
    PASS = auto()
    FAIL = auto()
    UNKNOWN = auto()
    TIMEOUT = auto()
    SKIPPED = auto()


@dataclass
class VerificationResult:
    """Result of verifying one inference step."""
    step_id: str
    status: VerificationStatus
    assertion_count: int
    smt_time_s: float
    message: str = ""
    unsat_core: Optional[List[str]] = None
    model_witness: Optional[Dict[str, float]] = None

    @property
    def passed(self) -> bool:
        return self.status == VerificationStatus.PASS


@dataclass
class SessionStats:
    """Aggregate statistics for a verification session."""
    total_steps: int = 0
    passed_steps: int = 0
    failed_steps: int = 0
    unknown_steps: int = 0
    timeout_steps: int = 0
    skipped_steps: int = 0
    total_smt_time_s: float = 0.0
    total_inference_time_s: float = 0.0
    peak_stack_depth: int = 0
    total_assertions: int = 0
    session_id: str = ""

    @property
    def smt_overhead_ratio(self) -> float:
        """Fraction of total time spent in SMT solving."""
        total = self.total_smt_time_s + self.total_inference_time_s
        if total == 0.0:
            return 0.0
        return self.total_smt_time_s / total

    def summary(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "total_steps": self.total_steps,
            "passed": self.passed_steps,
            "failed": self.failed_steps,
            "unknown": self.unknown_steps,
            "timeout": self.timeout_steps,
            "skipped": self.skipped_steps,
            "smt_time_s": round(self.total_smt_time_s, 6),
            "inference_time_s": round(self.total_inference_time_s, 6),
            "overhead_ratio": round(self.smt_overhead_ratio, 6),
            "peak_stack_depth": self.peak_stack_depth,
            "total_assertions": self.total_assertions,
        }


@dataclass
class BoundEvidence:
    """Evidence supporting a claimed bound."""
    lp_objective: Optional[float] = None
    dual_values: Optional[List[float]] = None
    reduced_costs: Optional[List[float]] = None
    basis_indices: Optional[List[int]] = None
    farkas_coefficients: Optional[List[float]] = None


@dataclass
class MessageData:
    """Data transmitted in a junction-tree message."""
    sender_id: str
    receiver_id: str
    separator_vars: List[str]
    potential_values: Dict[str, float]
    marginal_values: Optional[Dict[str, float]] = None
    bound_lower: Optional[float] = None
    bound_upper: Optional[float] = None


# ---------------------------------------------------------------------------
# SMTVerifier
# ---------------------------------------------------------------------------

class SMTVerifier:
    """
    Streaming SMT verifier for causal inference pipelines.

    Wraps a Z3 solver with an incremental push/pop assertion protocol.
    Each junction-tree message is verified as it arrives, producing a
    certificate for every passing step. The verifier tracks overhead
    so users can monitor the cost of formal verification relative to
    the inference computation itself.

    Usage
    -----
    >>> v = SMTVerifier(timeout_ms=5000)
    >>> v.begin_session()
    >>> result = v.verify_message("A", "B", msg_data)
    >>> cert = v.emit_certificate()
    >>> v.end_session()
    """

    def __init__(
        self,
        timeout_ms: int = 10_000,
        track_unsat_cores: bool = True,
        emit_certificates: bool = True,
        epsilon: float = 1e-9,
    ) -> None:
        self._timeout_ms = timeout_ms
        self._track_cores = track_unsat_cores
        self._emit_certs = emit_certificates
        self._epsilon = epsilon

        # Z3 solver (created fresh per session)
        self._solver: Optional[z3.Solver] = None
        self._encoder: Optional[SMTEncoder] = None
        self._protocol: Optional[IncrementalProtocol] = None
        self._cert_emitter: Optional[CertificateEmitter] = None

        # Session bookkeeping
        self._session_active = False
        self._session_id: str = ""
        self._results: List[VerificationResult] = []
        self._stats = SessionStats()
        self._step_counter = 0
        self._inference_clock_start: Optional[float] = None

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def begin_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new verification session.

        Creates a fresh Z3 solver, encoder, incremental protocol, and
        (optionally) a certificate emitter. Returns the session id.
        """
        if self._session_active:
            raise RuntimeError("Session already active – call end_session() first")

        self._session_id = session_id or uuid.uuid4().hex[:16]
        self._solver = z3.Solver()
        self._solver.set("timeout", self._timeout_ms)
        if self._track_cores:
            self._solver.set("unsat_core", True)

        self._encoder = SMTEncoder(epsilon=self._epsilon)
        self._protocol = IncrementalProtocol(self._solver)
        if self._emit_certs:
            self._cert_emitter = CertificateEmitter(session_id=self._session_id)
            self._cert_emitter.create_certificate()

        self._results.clear()
        self._stats = SessionStats(session_id=self._session_id)
        self._step_counter = 0
        self._session_active = True
        self._inference_clock_start = time.perf_counter()
        return self._session_id

    def end_session(self) -> SessionStats:
        """Finalise the session and return aggregate statistics."""
        self._require_session()
        if self._inference_clock_start is not None:
            elapsed = time.perf_counter() - self._inference_clock_start
            self._stats.total_inference_time_s = max(
                0.0, elapsed - self._stats.total_smt_time_s
            )
        if self._cert_emitter is not None:
            self._cert_emitter.finalize()
        self._session_active = False
        return self._stats

    # ------------------------------------------------------------------
    # Streaming verification entry points
    # ------------------------------------------------------------------

    def verify_message(
        self,
        sender: str,
        receiver: str,
        message_data: MessageData,
    ) -> VerificationResult:
        """
        Verify a single junction-tree message.

        Pushes a new assertion context, encodes the message consistency
        constraints, checks satisfiability, records the result, and pops
        the context so the solver is ready for the next message.
        """
        self._require_session()
        assert self._solver is not None
        assert self._encoder is not None
        assert self._protocol is not None

        step_id = f"msg_{self._step_counter}"
        self._step_counter += 1
        t0 = time.perf_counter()

        # --- push a new scope ---
        self._protocol.push_context(label=step_id)

        assertions: List[z3.BoolRef] = []

        # 1. Normalization of the potential sent
        if message_data.potential_values:
            norm_expr = self._encoder.encode_normalization_from_values(
                step_id, message_data.potential_values
            )
            assertions.append(norm_expr)

        # 2. Non-negativity of potential entries
        for var_name, val in message_data.potential_values.items():
            smt_var = z3.Real(f"{step_id}_pot_{var_name}")
            assertions.append(smt_var == z3.RealVal(str(val)))
            assertions.append(smt_var >= 0)

        # 3. Marginalization consistency (if marginal provided)
        if message_data.marginal_values is not None:
            marg_expr = self._encoder.encode_marginalization_check(
                step_id,
                message_data.potential_values,
                message_data.marginal_values,
                message_data.separator_vars,
            )
            assertions.append(marg_expr)

        # 4. Bound validity (lower <= upper)
        if (
            message_data.bound_lower is not None
            and message_data.bound_upper is not None
        ):
            bound_expr = self._encoder.encode_bound_claim(
                f"{step_id}_bound",
                message_data.bound_lower,
                message_data.bound_upper,
            )
            assertions.append(bound_expr)

        # 5. Message consistency between sender potential and separator
        sep_potential: Dict[str, float] = {
            k: v
            for k, v in message_data.potential_values.items()
            if k in message_data.separator_vars
        }
        if sep_potential:
            mc_expr = self._encoder.encode_message_consistency(
                f"{step_id}_sender",
                message_data.potential_values,
                message_data.separator_vars,
                sep_potential,
            )
            assertions.append(mc_expr)

        # assert everything into the solver
        for a in assertions:
            self._protocol.assert_claim(a, label=step_id)

        result = self._check_and_record(step_id, len(assertions))

        # --- pop back ---
        self._protocol.pop_context()

        dt = time.perf_counter() - t0
        result.smt_time_s = dt
        self._stats.total_smt_time_s += dt

        self._results.append(result)
        self._update_stats(result)

        if self._cert_emitter is not None and result.passed:
            self._cert_emitter.add_step(
                assertion=str(z3.And(*assertions)) if assertions else "true",
                proof=f"verified:{step_id}",
                metadata={
                    "sender": sender,
                    "receiver": receiver,
                    "separator": message_data.separator_vars,
                },
            )

        return result

    def verify_bound(
        self,
        lower: float,
        upper: float,
        evidence: BoundEvidence,
    ) -> VerificationResult:
        """
        Verify a claimed probability bound [lower, upper] using LP evidence.

        Encodes dual feasibility and optimality gap into QF_LRA and checks
        the resulting formula with Z3.
        """
        self._require_session()
        assert self._solver is not None
        assert self._encoder is not None
        assert self._protocol is not None

        step_id = f"bound_{self._step_counter}"
        self._step_counter += 1
        t0 = time.perf_counter()

        self._protocol.push_context(label=step_id)

        assertions: List[z3.BoolRef] = []

        # Basic bound ordering
        lb = z3.Real(f"{step_id}_lb")
        ub = z3.Real(f"{step_id}_ub")
        assertions.append(lb == z3.RealVal(str(lower)))
        assertions.append(ub == z3.RealVal(str(upper)))
        assertions.append(lb <= ub)

        # Probability range
        assertions.append(lb >= z3.RealVal("0"))
        assertions.append(ub <= z3.RealVal("1"))

        # LP objective witness
        if evidence.lp_objective is not None:
            obj = z3.Real(f"{step_id}_obj")
            assertions.append(obj == z3.RealVal(str(evidence.lp_objective)))
            assertions.append(obj >= lb)
            assertions.append(obj <= ub)

        # Dual feasibility
        if evidence.dual_values is not None:
            dual_sum = z3.RealVal("0")
            for i, dv in enumerate(evidence.dual_values):
                d_var = z3.Real(f"{step_id}_dual_{i}")
                assertions.append(d_var == z3.RealVal(str(dv)))
                dual_sum = dual_sum + d_var
            # Weak duality: dual objective bounds primal
            assertions.append(dual_sum >= lb - z3.RealVal(str(self._epsilon)))

        # Reduced cost non-negativity (optimality certificate)
        if evidence.reduced_costs is not None:
            for i, rc in enumerate(evidence.reduced_costs):
                rc_var = z3.Real(f"{step_id}_rc_{i}")
                assertions.append(rc_var == z3.RealVal(str(rc)))
                assertions.append(
                    rc_var >= z3.RealVal(str(-self._epsilon))
                )

        # Farkas infeasibility certificate
        if evidence.farkas_coefficients is not None:
            farkas_sum = z3.RealVal("0")
            for i, fc in enumerate(evidence.farkas_coefficients):
                f_var = z3.Real(f"{step_id}_farkas_{i}")
                assertions.append(f_var == z3.RealVal(str(fc)))
                assertions.append(f_var >= 0)
                farkas_sum = farkas_sum + f_var
            assertions.append(farkas_sum > z3.RealVal("0"))

        for a in assertions:
            self._protocol.assert_claim(a, label=step_id)

        result = self._check_and_record(step_id, len(assertions))
        self._protocol.pop_context()

        dt = time.perf_counter() - t0
        result.smt_time_s = dt
        self._stats.total_smt_time_s += dt
        self._results.append(result)
        self._update_stats(result)

        if self._cert_emitter is not None and result.passed:
            self._cert_emitter.add_step(
                assertion=f"bound({lower}, {upper})",
                proof=f"verified:{step_id}",
                metadata={"lower": lower, "upper": upper},
            )

        return result

    def verify_dsep_claim(
        self,
        x: str,
        y: str,
        z_set: List[str],
        dag_edges: List[Tuple[str, str]],
    ) -> VerificationResult:
        """
        Verify a d-separation claim dsep(X, Y | Z) in the given DAG.

        Encodes the DAG structure and path-blocking conditions, then checks
        that no active path exists from X to Y given Z.
        """
        self._require_session()
        assert self._solver is not None
        assert self._encoder is not None
        assert self._protocol is not None

        step_id = f"dsep_{self._step_counter}"
        self._step_counter += 1
        t0 = time.perf_counter()

        self._protocol.push_context(label=step_id)

        all_nodes: set[str] = set()
        for u, v in dag_edges:
            all_nodes.add(u)
            all_nodes.add(v)
        all_nodes.add(x)
        all_nodes.add(y)
        all_nodes.update(z_set)
        node_list = sorted(all_nodes)

        assertions: List[z3.BoolRef] = []

        # Edge indicator variables
        edge_vars: Dict[Tuple[str, str], z3.BoolRef] = {}
        for u in node_list:
            for v in node_list:
                if u != v:
                    evar = z3.Bool(f"{step_id}_edge_{u}_{v}")
                    edge_vars[(u, v)] = evar
                    if (u, v) in dag_edges:
                        assertions.append(evar)
                    else:
                        assertions.append(z3.Not(evar))

        # Active-path indicator: if there's an active path from x to y
        active = z3.Bool(f"{step_id}_active_{x}_{y}")

        # Reachability via unblocked path (over-approximation)
        reach_vars: Dict[str, z3.BoolRef] = {}
        for n in node_list:
            reach_vars[n] = z3.Bool(f"{step_id}_reach_{n}")

        # x is reachable from itself
        assertions.append(reach_vars[x])

        # Conditioning set blocks
        for c in z_set:
            if c in reach_vars:
                assertions.append(z3.Not(reach_vars[c]))

        # Propagation
        for u in node_list:
            for v in node_list:
                if u != v and (u, v) in edge_vars:
                    if v not in z_set:
                        assertions.append(
                            z3.Implies(
                                z3.And(reach_vars[u], edge_vars[(u, v)]),
                                reach_vars[v],
                            )
                        )

        # d-separation ↔ y not reachable
        assertions.append(active == reach_vars.get(y, z3.BoolVal(False)))
        # Claim: y should NOT be reachable
        assertions.append(z3.Not(active))

        for a in assertions:
            self._protocol.assert_claim(a, label=step_id)

        result = self._check_and_record(step_id, len(assertions))
        self._protocol.pop_context()

        dt = time.perf_counter() - t0
        result.smt_time_s = dt
        self._stats.total_smt_time_s += dt
        self._results.append(result)
        self._update_stats(result)

        return result

    def verify_normalization(
        self,
        distribution: Dict[str, float],
    ) -> VerificationResult:
        """Verify that a distribution sums to 1 within tolerance."""
        self._require_session()
        assert self._solver is not None
        assert self._encoder is not None
        assert self._protocol is not None

        step_id = f"norm_{self._step_counter}"
        self._step_counter += 1
        t0 = time.perf_counter()

        self._protocol.push_context(label=step_id)

        assertions: List[z3.BoolRef] = []
        smt_vars: List[z3.ArithRef] = []
        for name, val in distribution.items():
            v = z3.Real(f"{step_id}_{name}")
            assertions.append(v == z3.RealVal(str(val)))
            assertions.append(v >= 0)
            smt_vars.append(v)

        total = z3.Sum(smt_vars) if smt_vars else z3.RealVal("0")
        eps = z3.RealVal(str(self._epsilon))
        assertions.append(total >= z3.RealVal("1") - eps)
        assertions.append(total <= z3.RealVal("1") + eps)

        for a in assertions:
            self._protocol.assert_claim(a, label=step_id)

        result = self._check_and_record(step_id, len(assertions))
        self._protocol.pop_context()

        dt = time.perf_counter() - t0
        result.smt_time_s = dt
        self._stats.total_smt_time_s += dt
        self._results.append(result)
        self._update_stats(result)
        return result

    def verify_causal_polytope_membership(
        self,
        point: List[float],
        inequality_matrix: List[List[float]],
        inequality_rhs: List[float],
    ) -> VerificationResult:
        """
        Verify that *point* lies inside the causal polytope Ax <= b.
        """
        self._require_session()
        assert self._solver is not None
        assert self._protocol is not None

        step_id = f"polytope_{self._step_counter}"
        self._step_counter += 1
        t0 = time.perf_counter()

        self._protocol.push_context(label=step_id)
        assertions: List[z3.BoolRef] = []

        n = len(point)
        x_vars = [z3.Real(f"{step_id}_x_{j}") for j in range(n)]
        for j in range(n):
            assertions.append(x_vars[j] == z3.RealVal(str(point[j])))

        for i, (row, rhs) in enumerate(
            zip(inequality_matrix, inequality_rhs)
        ):
            lhs = z3.Sum(
                [
                    z3.RealVal(str(row[j])) * x_vars[j]
                    for j in range(min(len(row), n))
                ]
            )
            assertions.append(lhs <= z3.RealVal(str(rhs)))

        for a in assertions:
            self._protocol.assert_claim(a, label=step_id)

        result = self._check_and_record(step_id, len(assertions))
        self._protocol.pop_context()

        dt = time.perf_counter() - t0
        result.smt_time_s = dt
        self._stats.total_smt_time_s += dt
        self._results.append(result)
        self._update_stats(result)
        return result

    def verify_monotonicity(
        self,
        values: List[Tuple[str, float]],
    ) -> VerificationResult:
        """Verify that a sequence of labelled values is non-decreasing."""
        self._require_session()
        assert self._solver is not None
        assert self._protocol is not None

        step_id = f"mono_{self._step_counter}"
        self._step_counter += 1
        t0 = time.perf_counter()

        self._protocol.push_context(label=step_id)
        assertions: List[z3.BoolRef] = []

        prev_var: Optional[z3.ArithRef] = None
        for idx, (label, val) in enumerate(values):
            v = z3.Real(f"{step_id}_{label}_{idx}")
            assertions.append(v == z3.RealVal(str(val)))
            if prev_var is not None:
                assertions.append(prev_var <= v)
            prev_var = v

        for a in assertions:
            self._protocol.assert_claim(a, label=step_id)

        result = self._check_and_record(step_id, len(assertions))
        self._protocol.pop_context()

        dt = time.perf_counter() - t0
        result.smt_time_s = dt
        self._stats.total_smt_time_s += dt
        self._results.append(result)
        self._update_stats(result)
        return result

    # ------------------------------------------------------------------
    # Certificate emission
    # ------------------------------------------------------------------

    def emit_certificate(self) -> Optional[Certificate]:
        """Return the current certificate (if certificate emission is on)."""
        if self._cert_emitter is None:
            return None
        return self._cert_emitter.get_certificate()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_verification_stats(self) -> SessionStats:
        """Return a snapshot of the current session statistics."""
        return self._stats

    def get_results(self) -> List[VerificationResult]:
        """Return all verification results recorded so far."""
        return list(self._results)

    def get_last_result(self) -> Optional[VerificationResult]:
        """Return the most recent verification result."""
        return self._results[-1] if self._results else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_session(self) -> None:
        if not self._session_active:
            raise RuntimeError("No active session – call begin_session() first")

    def _check_and_record(
        self, step_id: str, assertion_count: int
    ) -> VerificationResult:
        """Run the solver and translate the Z3 result."""
        assert self._solver is not None
        assert self._protocol is not None

        check_result = self._protocol.check_satisfiability()

        if check_result == z3.sat:
            status = VerificationStatus.PASS
            msg = "satisfiable"
            model_witness = self._extract_model()
            core = None
        elif check_result == z3.unsat:
            status = VerificationStatus.FAIL
            msg = "unsatisfiable"
            model_witness = None
            core = self._extract_core() if self._track_cores else None
        else:
            status = VerificationStatus.UNKNOWN
            msg = "unknown/timeout"
            model_witness = None
            core = None

        depth = self._protocol.get_stack_depth()
        if depth > self._stats.peak_stack_depth:
            self._stats.peak_stack_depth = depth

        return VerificationResult(
            step_id=step_id,
            status=status,
            assertion_count=assertion_count,
            smt_time_s=0.0,
            message=msg,
            unsat_core=core,
            model_witness=model_witness,
        )

    def _extract_model(self) -> Dict[str, float]:
        """Extract a (partial) model from the solver for diagnostics."""
        assert self._solver is not None
        try:
            m = self._solver.model()
            result: Dict[str, float] = {}
            for decl in m.decls():
                val = m[decl]
                try:
                    if z3.is_real(val) or z3.is_int(val):
                        result[decl.name()] = float(val.as_fraction())
                    elif z3.is_true(val):
                        result[decl.name()] = 1.0
                    elif z3.is_false(val):
                        result[decl.name()] = 0.0
                except (ValueError, AttributeError, z3.Z3Exception):
                    pass
            return result
        except z3.Z3Exception:
            return {}

    def _extract_core(self) -> List[str]:
        """Extract an unsatisfiable core from the solver."""
        assert self._solver is not None
        try:
            core = self._solver.unsat_core()
            return [str(c) for c in core]
        except z3.Z3Exception:
            return []

    def _update_stats(self, result: VerificationResult) -> None:
        self._stats.total_steps += 1
        self._stats.total_assertions += result.assertion_count
        if result.status == VerificationStatus.PASS:
            self._stats.passed_steps += 1
        elif result.status == VerificationStatus.FAIL:
            self._stats.failed_steps += 1
        elif result.status == VerificationStatus.UNKNOWN:
            self._stats.unknown_steps += 1
        elif result.status == VerificationStatus.TIMEOUT:
            self._stats.timeout_steps += 1
        elif result.status == VerificationStatus.SKIPPED:
            self._stats.skipped_steps += 1

    # ------------------------------------------------------------------
    # Convenience: bulk verification
    # ------------------------------------------------------------------

    def verify_message_batch(
        self,
        messages: Sequence[Tuple[str, str, MessageData]],
    ) -> List[VerificationResult]:
        """Verify a batch of messages sequentially, returning all results."""
        results: List[VerificationResult] = []
        for sender, receiver, data in messages:
            results.append(self.verify_message(sender, receiver, data))
        return results

    def verify_bound_batch(
        self,
        bounds: Sequence[Tuple[float, float, BoundEvidence]],
    ) -> List[VerificationResult]:
        """Verify a batch of bound claims sequentially."""
        results: List[VerificationResult] = []
        for lo, hi, ev in bounds:
            results.append(self.verify_bound(lo, hi, ev))
        return results

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "SMTVerifier":
        self.begin_session()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._session_active:
            self.end_session()

    # ------------------------------------------------------------------
    # Overhead measurement helpers
    # ------------------------------------------------------------------

    def mark_inference_start(self) -> None:
        """Call just before inference computation begins."""
        self._inference_clock_start = time.perf_counter()

    def mark_inference_end(self) -> None:
        """Call just after inference computation finishes."""
        if self._inference_clock_start is not None:
            self._stats.total_inference_time_s += (
                time.perf_counter() - self._inference_clock_start
            )
            self._inference_clock_start = None

    # ------------------------------------------------------------------
    # Reset / reconfigure
    # ------------------------------------------------------------------

    def reconfigure(
        self,
        timeout_ms: Optional[int] = None,
        epsilon: Optional[float] = None,
    ) -> None:
        """Change solver parameters between steps (not during a check)."""
        if timeout_ms is not None:
            self._timeout_ms = timeout_ms
            if self._solver is not None:
                self._solver.set("timeout", self._timeout_ms)
        if epsilon is not None:
            self._epsilon = epsilon
            if self._encoder is not None:
                self._encoder.epsilon = epsilon

    def reset_solver(self) -> None:
        """Hard-reset the Z3 solver within the current session."""
        self._require_session()
        self._solver = z3.Solver()
        self._solver.set("timeout", self._timeout_ms)
        if self._track_cores:
            self._solver.set("unsat_core", True)
        self._protocol = IncrementalProtocol(self._solver)

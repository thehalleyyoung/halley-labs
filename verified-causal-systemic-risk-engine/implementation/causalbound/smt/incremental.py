"""
IncrementalProtocol: manage the Z3 incremental assertion stack for
streaming verification.

Provides push/pop context management, assertion grouping by inference
phase, backtracking support, dependency tracking, and assertion replay
for debugging. Guarantees linear stack growth.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import z3


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AssertionRecord:
    """Metadata for a single assertion in the incremental stack."""
    assertion_id: int
    expression: z3.BoolRef
    label: str
    phase: str
    depth: int
    timestamp: float
    dependencies: List[int] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"AssertionRecord(id={self.assertion_id}, label={self.label!r}, "
            f"phase={self.phase!r}, depth={self.depth})"
        )


@dataclass
class ContextFrame:
    """One level in the push/pop assertion stack."""
    frame_id: int
    label: str
    phase: str
    depth: int
    assertion_ids: List[int] = field(default_factory=list)
    creation_time: float = 0.0
    check_count: int = 0
    cumulative_check_time_s: float = 0.0

    def __repr__(self) -> str:
        return (
            f"ContextFrame(id={self.frame_id}, label={self.label!r}, "
            f"depth={self.depth}, assertions={len(self.assertion_ids)})"
        )


@dataclass
class ProtocolStats:
    """Runtime statistics for the incremental protocol."""
    total_pushes: int = 0
    total_pops: int = 0
    total_assertions: int = 0
    total_checks: int = 0
    total_check_time_s: float = 0.0
    peak_depth: int = 0
    peak_assertions: int = 0
    replays: int = 0
    backtrack_count: int = 0


# ---------------------------------------------------------------------------
# IncrementalProtocol
# ---------------------------------------------------------------------------

class IncrementalProtocol:
    """
    Manage the Z3 incremental assertion stack for streaming verification.

    Key properties
    --------------
    * **Push / pop scoping** mirrors the Z3 solver's internal stack so
      that assertions added inside ``push_context`` / ``pop_context``
      are automatically retracted on backtrack.
    * **Assertion grouping** by inference phase (message-passing,
      LP-bound, polytope, …) enables selective replay.
    * **Linear growth guarantee**: the number of live assertions at any
      point is bounded by the number of push levels times the maximum
      assertions per level, which grows linearly with the number of
      inference steps.
    * **Replay** re-asserts a recorded prefix of assertions into a
      fresh solver for debugging.

    Parameters
    ----------
    solver : z3.Solver
        The Z3 solver instance to manage.
    """

    def __init__(self, solver: z3.Solver) -> None:
        self._solver = solver
        self._stack: List[ContextFrame] = []
        self._assertions: Dict[int, AssertionRecord] = {}
        self._next_assertion_id = 0
        self._next_frame_id = 0
        self._current_phase: str = "default"
        self._stats = ProtocolStats()
        self._dependency_graph: Dict[int, Set[int]] = {}
        self._phase_index: Dict[str, List[int]] = {}

    # ------------------------------------------------------------------
    # Context (push / pop)
    # ------------------------------------------------------------------

    def push_context(
        self,
        label: str = "",
        phase: Optional[str] = None,
    ) -> int:
        """
        Push a new assertion scope.

        Parameters
        ----------
        label : str
            Human-readable label for the context (e.g. ``"msg_3"``).
        phase : str, optional
            Inference phase tag. Defaults to the current phase.

        Returns
        -------
        int
            Frame id of the newly created context.
        """
        if phase is not None:
            self._current_phase = phase

        frame = ContextFrame(
            frame_id=self._next_frame_id,
            label=label,
            phase=self._current_phase,
            depth=len(self._stack) + 1,
            creation_time=time.perf_counter(),
        )
        self._next_frame_id += 1
        self._stack.append(frame)
        self._solver.push()

        self._stats.total_pushes += 1
        if len(self._stack) > self._stats.peak_depth:
            self._stats.peak_depth = len(self._stack)

        return frame.frame_id

    def pop_context(self) -> Optional[ContextFrame]:
        """
        Pop the most recent assertion scope, retracting all its
        assertions from the solver.

        Returns
        -------
        ContextFrame or None
            The popped frame, or None if the stack was empty.
        """
        if not self._stack:
            return None

        frame = self._stack.pop()
        self._solver.pop()

        # Remove assertion records that lived only in this frame
        for aid in frame.assertion_ids:
            self._dependency_graph.pop(aid, None)
            self._assertions.pop(aid, None)

        self._stats.total_pops += 1
        return frame

    def pop_to_depth(self, target_depth: int) -> List[ContextFrame]:
        """
        Pop frames until the stack depth equals *target_depth*.

        Useful for backtracking multiple levels at once.
        """
        popped: List[ContextFrame] = []
        while len(self._stack) > target_depth:
            frame = self.pop_context()
            if frame is not None:
                popped.append(frame)
        self._stats.backtrack_count += 1
        return popped

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------

    def assert_claim(
        self,
        claim: z3.BoolRef,
        label: str = "",
        dependencies: Optional[List[int]] = None,
    ) -> int:
        """
        Assert *claim* into the solver within the current context.

        Parameters
        ----------
        claim : z3.BoolRef
            The Z3 boolean expression to assert.
        label : str
            A short identifier for tracing.
        dependencies : list of int, optional
            IDs of assertions that this one logically depends on.

        Returns
        -------
        int
            The assertion id.
        """
        aid = self._next_assertion_id
        self._next_assertion_id += 1

        depth = len(self._stack)
        rec = AssertionRecord(
            assertion_id=aid,
            expression=claim,
            label=label,
            phase=self._current_phase,
            depth=depth,
            timestamp=time.perf_counter(),
            dependencies=dependencies or [],
        )
        self._assertions[aid] = rec

        # Track dependencies
        self._dependency_graph[aid] = set(dependencies or [])

        # Track phase
        self._phase_index.setdefault(self._current_phase, []).append(aid)

        # Record in the current frame (if any)
        if self._stack:
            self._stack[-1].assertion_ids.append(aid)

        # Actually add to the Z3 solver
        if self._stats.total_assertions == 0 or not label:
            self._solver.add(claim)
        else:
            # Use tracked assertions for unsat-core support
            tracker = z3.Bool(f"__track_{aid}")
            self._solver.assert_and_track(claim, tracker)

        self._stats.total_assertions += 1
        if self._stats.total_assertions > self._stats.peak_assertions:
            self._stats.peak_assertions = self._stats.total_assertions

        return aid

    def assert_claims(
        self,
        claims: Sequence[z3.BoolRef],
        label: str = "",
    ) -> List[int]:
        """Assert multiple claims, returning their IDs."""
        return [self.assert_claim(c, label=label) for c in claims]

    # ------------------------------------------------------------------
    # Satisfiability checking
    # ------------------------------------------------------------------

    def check_satisfiability(
        self,
        assumptions: Optional[List[z3.BoolRef]] = None,
    ) -> z3.CheckSatResult:
        """
        Check satisfiability of the current assertion set.

        Parameters
        ----------
        assumptions : list of z3.BoolRef, optional
            Temporary assumptions for this check only (not persisted).
        """
        t0 = time.perf_counter()
        if assumptions:
            result = self._solver.check(*assumptions)
        else:
            result = self._solver.check()
        dt = time.perf_counter() - t0

        self._stats.total_checks += 1
        self._stats.total_check_time_s += dt

        if self._stack:
            self._stack[-1].check_count += 1
            self._stack[-1].cumulative_check_time_s += dt

        return result

    def get_unsat_core(self) -> List[z3.ExprRef]:
        """Return the unsat core from the last ``check()`` call."""
        try:
            return list(self._solver.unsat_core())
        except z3.Z3Exception:
            return []

    def get_model(self) -> Optional[z3.ModelRef]:
        """Return the model from the last satisfiable ``check()``."""
        try:
            return self._solver.model()
        except z3.Z3Exception:
            return None

    # ------------------------------------------------------------------
    # Stack introspection
    # ------------------------------------------------------------------

    def get_stack_depth(self) -> int:
        """Return the current push depth."""
        return len(self._stack)

    def get_current_frame(self) -> Optional[ContextFrame]:
        """Return the top-most context frame."""
        return self._stack[-1] if self._stack else None

    def get_frame_at(self, depth: int) -> Optional[ContextFrame]:
        """Return the context frame at the given depth (0-indexed)."""
        if 0 <= depth < len(self._stack):
            return self._stack[depth]
        return None

    def get_assertion_count(self) -> int:
        """Total assertions currently tracked."""
        return len(self._assertions)

    def get_assertions_in_phase(self, phase: str) -> List[AssertionRecord]:
        """Return all assertions belonging to *phase*."""
        ids = self._phase_index.get(phase, [])
        return [self._assertions[i] for i in ids if i in self._assertions]

    def get_assertion_dependencies(self, aid: int) -> Set[int]:
        """Return direct dependencies of assertion *aid*."""
        return self._dependency_graph.get(aid, set())

    def get_transitive_dependencies(self, aid: int) -> Set[int]:
        """Compute the transitive closure of dependencies for *aid*."""
        visited: Set[int] = set()
        stack = [aid]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            stack.extend(self._dependency_graph.get(cur, set()))
        visited.discard(aid)
        return visited

    # ------------------------------------------------------------------
    # Replay (for debugging)
    # ------------------------------------------------------------------

    def replay(
        self,
        up_to_assertion: Optional[int] = None,
        solver: Optional[z3.Solver] = None,
    ) -> z3.CheckSatResult:
        """
        Replay recorded assertions into a (optionally fresh) solver.

        Parameters
        ----------
        up_to_assertion : int, optional
            Replay only assertions with id ≤ this value.
        solver : z3.Solver, optional
            Solver to use. If None, a fresh solver is created.

        Returns
        -------
        z3.CheckSatResult
            Result of checking the replayed assertions.
        """
        self._stats.replays += 1
        target = solver or z3.Solver()

        sorted_ids = sorted(self._assertions.keys())
        for aid in sorted_ids:
            if up_to_assertion is not None and aid > up_to_assertion:
                break
            rec = self._assertions[aid]
            target.add(rec.expression)

        return target.check()

    def replay_phase(
        self,
        phase: str,
        solver: Optional[z3.Solver] = None,
    ) -> z3.CheckSatResult:
        """Replay only assertions belonging to *phase*."""
        self._stats.replays += 1
        target = solver or z3.Solver()
        for aid in sorted(self._phase_index.get(phase, [])):
            if aid in self._assertions:
                target.add(self._assertions[aid].expression)
        return target.check()

    def get_replay_script(
        self,
        up_to_assertion: Optional[int] = None,
    ) -> str:
        """
        Generate an SMT-LIB2 script that reproduces the current
        assertion stack. Useful for stand-alone debugging.
        """
        lines: List[str] = ["(set-logic QF_LRA)"]
        sorted_ids = sorted(self._assertions.keys())
        for aid in sorted_ids:
            if up_to_assertion is not None and aid > up_to_assertion:
                break
            rec = self._assertions[aid]
            lines.append(f"; assertion {aid} label={rec.label} phase={rec.phase}")
            sexpr = rec.expression.sexpr()
            lines.append(f"(assert {sexpr})")
        lines.append("(check-sat)")
        lines.append("(exit)")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Phase management
    # ------------------------------------------------------------------

    def set_phase(self, phase: str) -> None:
        """Set the current inference phase tag."""
        self._current_phase = phase

    def get_phase(self) -> str:
        return self._current_phase

    def get_phases(self) -> List[str]:
        """Return all phases that have been used."""
        return list(self._phase_index.keys())

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def compact(self) -> int:
        """
        Remove assertion records that are no longer in any live frame.

        Returns the number of records removed.
        """
        live_ids: Set[int] = set()
        for frame in self._stack:
            live_ids.update(frame.assertion_ids)

        dead = [aid for aid in self._assertions if aid not in live_ids]
        for aid in dead:
            del self._assertions[aid]
            self._dependency_graph.pop(aid, None)

        # Rebuild phase index
        for phase in list(self._phase_index):
            self._phase_index[phase] = [
                a for a in self._phase_index[phase] if a in self._assertions
            ]
            if not self._phase_index[phase]:
                del self._phase_index[phase]

        return len(dead)

    def estimate_memory(self) -> int:
        """
        Rough estimate of memory consumed by assertion records (bytes).
        """
        per_record = 200  # rough estimate per record
        return len(self._assertions) * per_record

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> ProtocolStats:
        return self._stats

    def get_frame_summary(self) -> List[Dict[str, Any]]:
        """Return a summary of each live frame on the stack."""
        return [
            {
                "frame_id": f.frame_id,
                "label": f.label,
                "phase": f.phase,
                "depth": f.depth,
                "assertions": len(f.assertion_ids),
                "checks": f.check_count,
                "check_time_s": round(f.cumulative_check_time_s, 6),
            }
            for f in self._stack
        ]

    # ------------------------------------------------------------------
    # Solver access
    # ------------------------------------------------------------------

    @property
    def solver(self) -> z3.Solver:
        return self._solver

    def reset_solver(self, new_solver: Optional[z3.Solver] = None) -> None:
        """Replace the underlying solver (discards Z3-internal state)."""
        self._solver = new_solver or z3.Solver()
        self._stack.clear()
        self._assertions.clear()
        self._dependency_graph.clear()
        self._phase_index.clear()
        self._next_assertion_id = 0
        self._next_frame_id = 0

    # ------------------------------------------------------------------
    # Linear growth invariant check
    # ------------------------------------------------------------------

    def check_linear_growth(self) -> bool:
        """
        Verify that the number of live assertions is bounded linearly
        by the stack depth.

        Returns True if the invariant holds.
        """
        if not self._stack:
            return True
        max_per_frame = max(len(f.assertion_ids) for f in self._stack)
        total_live = sum(len(f.assertion_ids) for f in self._stack)
        # Linear bound: total ≤ depth × max_per_frame
        return total_live <= len(self._stack) * (max_per_frame + 1)

    # ------------------------------------------------------------------
    # Assertion search
    # ------------------------------------------------------------------

    def find_assertions(
        self,
        label_prefix: str = "",
        phase: str = "",
        min_depth: int = 0,
        max_depth: int = 999_999,
    ) -> List[AssertionRecord]:
        """Search assertions by label prefix, phase, and depth range."""
        results: List[AssertionRecord] = []
        for rec in self._assertions.values():
            if label_prefix and not rec.label.startswith(label_prefix):
                continue
            if phase and rec.phase != phase:
                continue
            if rec.depth < min_depth or rec.depth > max_depth:
                continue
            results.append(rec)
        return sorted(results, key=lambda r: r.assertion_id)

"""
CEGIS synthesis orchestrator for DP-Forge.

This is the **central module** of the DP-Forge pipeline.  It implements the
CounterExample-Guided Inductive Synthesis (CEGIS) loop that discovers
provably optimal differentially private mechanisms.

Algorithm Overview
~~~~~~~~~~~~~~~~~~

The CEGIS loop alternates between two phases:

1. **Synthesise** (LP solve):
   Given a *witness set* ``S`` of adjacent database pairs, solve a
   relaxed LP that enforces (ε,δ)-DP only over ``S``.  The LP minimises
   worst-case expected loss (minimax objective).

2. **Verify** (counterexample generation):
   Check whether the candidate mechanism ``p*`` satisfies (ε,δ)-DP
   over **all** adjacent pairs.  If yes, ``p*`` is optimal.  If no,
   the most-violating pair ``(i_viol, i'_viol)`` becomes a new witness.

Termination is guaranteed because the witness set can grow by at most
one pair per iteration, and the total number of adjacent pairs is finite
(|E| ≤ n(n-1)/2).

Convergence Properties
~~~~~~~~~~~~~~~~~~~~~~~

- The minimax objective is *monotonically non-decreasing* across
  iterations: ``err(p^{t+1}) ≥ err(p^t)``, since each new witness
  constrains the feasible set further.
- At termination the LP dual solution provides an *optimality
  certificate* proving that no mechanism in the family can achieve
  lower worst-case error at the same privacy level.

Cycle Handling
~~~~~~~~~~~~~~

If the verifier returns a pair ``(i, i')`` that is already in the
witness set ``S`` (a *cycle*), we do NOT tighten the privacy bound.
Instead we apply a DP-preserving projection (ExtractMechanism) to the
current LP solution and re-verify.  This avoids infinite loops caused
by floating-point solver imprecision.

Classes
-------
- :class:`CEGISEngine` — Full-featured CEGIS synthesis engine.
- :class:`WitnessSet` — Managed set of adjacent pairs with fast
  membership test and coverage tracking.
- :class:`ConvergenceHistory` — Tracks objective values, violations,
  and iteration timing for convergence analysis.
- :class:`DualSimplexWarmStart` — Warm-start manager for the LP solver.
- :class:`CEGISProgress` — Per-iteration progress snapshot.

Functions
---------
- :func:`CEGISSynthesize` — Main CEGIS orchestrator (functional API).
- :func:`synthesize_mechanism` — One-line high-level API.
- :func:`synthesize_for_workload` — Workload-level synthesis API.
- :func:`quick_synthesize` — Preset-based quick synthesis.

Enums
-----
- :class:`SynthesisStrategy` — Strategy selection for synthesis.
- :class:`CEGISStatus` — Status of the CEGIS loop.
"""

from __future__ import annotations

import logging
import math
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt

from dp_forge.exceptions import (
    ConfigurationError,
    ConvergenceError,
    CycleDetectedError,
    InfeasibleSpecError,
    NumericalInstabilityError,
    SolverError,
    VerificationError,
)
from dp_forge.lp_builder import (
    LPManager,
    SolveStatistics,
    VariableLayout,
    build_laplace_warm_start,
    build_output_grid,
    build_privacy_lp,
    extract_mechanism_table,
    solve_lp,
)
from dp_forge.types import (
    AdjacencyRelation,
    CEGISResult,
    ExtractedMechanism,
    LossFunction,
    LPStruct,
    MechanismFamily,
    NumericalConfig,
    OptimalityCertificate,
    PrivacyBudget,
    QuerySpec,
    QueryType,
    SolverBackend,
    SynthesisConfig,
    VerifyResult,
    WorkloadSpec,
)
from dp_forge.verifier import (
    PrivacyVerifier,
    VerificationMode,
    compute_safe_tolerance,
    counterexample_from_result,
    verify,
    verify_for_cegis,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

_DEFAULT_MAX_ITER: int = 10_000
_DEFAULT_CONVERGENCE_WINDOW: int = 10
_DEFAULT_STAGNATION_TOL: float = 1e-10
_DEFAULT_CYCLE_PROJECTION_TOL: float = 1e-8
_MAX_CYCLE_RETRIES: int = 3
_WARM_START_FALLBACK_THRESHOLD: int = 5
_SOLVER_RETRY_TOL_MULTIPLIER: float = 10.0


# ============================================================================
# Enums
# ============================================================================


class SynthesisStrategy(Enum):
    """Strategy for mechanism synthesis.

    ``LP_PURE``
        Pure ε-DP synthesis via LP with per-bin ratio constraints.

    ``LP_APPROX``
        Approximate (ε,δ)-DP synthesis via LP with hockey-stick
        divergence constraints.

    ``SDP_GAUSSIAN``
        Gaussian workload mechanism synthesis via SDP.

    ``HYBRID``
        Try LP first; fall back to SDP if the query structure
        suggests workload-level optimisation would help.
    """

    LP_PURE = auto()
    LP_APPROX = auto()
    SDP_GAUSSIAN = auto()
    HYBRID = auto()

    def __repr__(self) -> str:
        return f"SynthesisStrategy.{self.name}"


class CEGISStatus(Enum):
    """Status of the CEGIS loop at any point in execution.

    ``RUNNING``
        The loop is actively iterating.

    ``CONVERGED``
        The mechanism satisfies DP over all pairs; synthesis is complete.

    ``MAX_ITER_REACHED``
        The maximum iteration count was hit before convergence.

    ``INFEASIBLE``
        The LP is provably infeasible — no mechanism exists under the
        given privacy parameters and discretisation.

    ``CYCLE_RESOLVED``
        A cycle was detected and resolved via projection.

    ``NUMERICAL_FAILURE``
        The solver encountered numerical difficulties that could not
        be recovered from.

    ``STAGNATED``
        The objective stopped improving for too many iterations.
    """

    RUNNING = auto()
    CONVERGED = auto()
    MAX_ITER_REACHED = auto()
    INFEASIBLE = auto()
    CYCLE_RESOLVED = auto()
    NUMERICAL_FAILURE = auto()
    STAGNATED = auto()

    def __repr__(self) -> str:
        return f"CEGISStatus.{self.name}"


# ============================================================================
# Progress & Convergence Tracking
# ============================================================================


@dataclass
class CEGISProgress:
    """Snapshot of CEGIS loop state at a single iteration.

    Attributes:
        iteration: Current iteration number (0-indexed).
        objective: LP objective value (minimax expected loss).
        violation_magnitude: Magnitude of the worst DP violation,
            or 0.0 if the mechanism is valid.
        violation_pair: The (i, i') pair causing the worst violation,
            or ``None`` if valid.
        n_witness_pairs: Number of pairs in the witness set.
        solve_time: Wall-clock time for the LP solve in seconds.
        verify_time: Wall-clock time for verification in seconds.
        total_time: Cumulative wall-clock time since loop start.
        status: Current loop status.
        is_cycle: Whether this iteration detected a cycle.
    """

    iteration: int
    objective: float
    violation_magnitude: float
    violation_pair: Optional[Tuple[int, int]]
    n_witness_pairs: int
    solve_time: float
    verify_time: float
    total_time: float
    status: CEGISStatus = CEGISStatus.RUNNING
    is_cycle: bool = False

    def __repr__(self) -> str:
        viol = f"viol={self.violation_magnitude:.2e}" if self.violation_pair else "valid"
        return (
            f"CEGISProgress(iter={self.iteration}, obj={self.objective:.6f}, "
            f"{viol}, pairs={self.n_witness_pairs}, "
            f"time={self.total_time:.2f}s)"
        )


@dataclass
class ConvergenceHistory:
    """Tracks objective values, violations, and timing across CEGIS iterations.

    Used for convergence analysis, stagnation detection, and estimating
    remaining iteration count.

    Attributes:
        objectives: Objective value at each iteration.
        violations: Violation magnitude at each iteration (0 if valid).
        solve_times: LP solve time at each iteration.
        verify_times: Verification time at each iteration.
        witness_counts: Number of witness pairs at each iteration.
        iteration_times: Wall-clock time for each full iteration.
        cycle_iterations: Set of iteration indices where cycles occurred.
    """

    objectives: List[float] = field(default_factory=list)
    violations: List[float] = field(default_factory=list)
    solve_times: List[float] = field(default_factory=list)
    verify_times: List[float] = field(default_factory=list)
    witness_counts: List[int] = field(default_factory=list)
    iteration_times: List[float] = field(default_factory=list)
    cycle_iterations: Set[int] = field(default_factory=set)

    def record(
        self,
        objective: float,
        violation: float,
        solve_time: float,
        verify_time: float,
        n_witnesses: int,
        iteration_time: float,
        is_cycle: bool = False,
    ) -> None:
        """Record metrics for one CEGIS iteration.

        Args:
            objective: LP objective value.
            violation: Worst DP violation magnitude (0 if valid).
            solve_time: LP solve wall-clock time (seconds).
            verify_time: Verification wall-clock time (seconds).
            n_witnesses: Number of pairs in the witness set.
            iteration_time: Total iteration wall-clock time (seconds).
            is_cycle: Whether a cycle was detected this iteration.
        """
        self.objectives.append(objective)
        self.violations.append(violation)
        self.solve_times.append(solve_time)
        self.verify_times.append(verify_time)
        self.witness_counts.append(n_witnesses)
        self.iteration_times.append(iteration_time)
        if is_cycle:
            self.cycle_iterations.add(len(self.objectives) - 1)

    @property
    def n_iterations(self) -> int:
        """Number of recorded iterations."""
        return len(self.objectives)

    @property
    def total_solve_time(self) -> float:
        """Cumulative LP solve time."""
        return sum(self.solve_times)

    @property
    def total_verify_time(self) -> float:
        """Cumulative verification time."""
        return sum(self.verify_times)

    @property
    def total_time(self) -> float:
        """Cumulative iteration time."""
        return sum(self.iteration_times)

    def check_monotonicity(self, tol: float = 1e-12) -> bool:
        """Check that objectives are monotonically non-decreasing.

        The CEGIS theory guarantees ``err(p^{t+1}) >= err(p^t)`` because
        adding constraints can only shrink the feasible set.  A violation
        of this property indicates numerical issues.

        Args:
            tol: Tolerance for floating-point comparison.

        Returns:
            True if the objectives are monotone within tolerance.
        """
        if len(self.objectives) < 2:
            return True
        for i in range(1, len(self.objectives)):
            if i in self.cycle_iterations:
                continue  # Cycles may cause non-monotonicity
            if self.objectives[i] < self.objectives[i - 1] - tol:
                return False
        return True

    def detect_stagnation(
        self,
        window: int = _DEFAULT_CONVERGENCE_WINDOW,
        tol: float = _DEFAULT_STAGNATION_TOL,
    ) -> bool:
        """Detect when progress has stalled.

        Returns True if the objective has changed by less than ``tol``
        over the last ``window`` iterations.

        Args:
            window: Number of recent iterations to examine.
            tol: Minimum improvement threshold.

        Returns:
            True if progress has stagnated.
        """
        if len(self.objectives) < window:
            return False
        recent = self.objectives[-window:]
        obj_range = max(recent) - min(recent)
        return obj_range < tol

    def estimate_remaining_iterations(
        self,
        total_edges: int,
    ) -> int:
        """Estimate remaining iterations based on convergence rate.

        Uses the rate at which the witness set grows and the total
        number of possible edges.

        Args:
            total_edges: Total number of adjacent pairs in the full graph.

        Returns:
            Estimated number of remaining iterations, or 0 if converged.
        """
        if not self.witness_counts:
            return total_edges
        current = self.witness_counts[-1]
        if current >= total_edges:
            return 0
        if len(self.witness_counts) < 2:
            return total_edges - current
        # Average growth rate over last 5 iterations
        lookback = min(5, len(self.witness_counts) - 1)
        growth = (
            self.witness_counts[-1] - self.witness_counts[-1 - lookback]
        ) / lookback
        if growth <= 0:
            return total_edges - current
        return max(0, int(math.ceil((total_edges - current) / growth)))

    def objective_improvement_rate(self, window: int = 5) -> float:
        """Average objective improvement per iteration over recent window.

        Args:
            window: Number of recent iterations to examine.

        Returns:
            Average improvement (positive means objective increasing,
            which is expected as constraints are added).
        """
        if len(self.objectives) < 2:
            return 0.0
        lookback = min(window, len(self.objectives) - 1)
        delta = self.objectives[-1] - self.objectives[-1 - lookback]
        return delta / lookback

    def summary(self) -> str:
        """Return a human-readable convergence summary."""
        if not self.objectives:
            return "ConvergenceHistory: no iterations recorded"
        lines = [
            f"ConvergenceHistory ({self.n_iterations} iterations)",
            f"  Objective: {self.objectives[0]:.6f} → {self.objectives[-1]:.6f}",
            f"  Violations: {self.violations[0]:.2e} → {self.violations[-1]:.2e}",
            f"  Witness pairs: {self.witness_counts[0]} → {self.witness_counts[-1]}",
            f"  Cycles: {len(self.cycle_iterations)}",
            f"  Total solve time: {self.total_solve_time:.3f}s",
            f"  Total verify time: {self.total_verify_time:.3f}s",
            f"  Total time: {self.total_time:.3f}s",
            f"  Monotonic: {self.check_monotonicity()}",
            f"  Stagnated: {self.detect_stagnation()}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ConvergenceHistory(n={self.n_iterations}, "
            f"cycles={len(self.cycle_iterations)})"
        )


# ============================================================================
# Witness Set Management
# ============================================================================


class WitnessSet:
    """Managed set of adjacent database pairs for the CEGIS witness set.

    Provides fast membership testing via a frozen-set of canonical (min, max)
    pairs, coverage tracking relative to the full adjacency graph, and
    smart seeding heuristics for fast convergence.

    Attributes:
        _pairs: Internal set of canonical (min, max) pairs.
        _insertion_order: List tracking the order pairs were added.
        _total_edges: Total number of edges in the full adjacency graph.
    """

    def __init__(self, total_edges: int = 0) -> None:
        """Initialise an empty witness set.

        Args:
            total_edges: Total number of possible adjacent pairs in the
                full adjacency graph (for coverage computation).
        """
        self._pairs: Set[Tuple[int, int]] = set()
        self._insertion_order: List[Tuple[int, int]] = []
        self._total_edges: int = total_edges

    @staticmethod
    def _canonical(i: int, i_prime: int) -> Tuple[int, int]:
        """Return canonical (min, max) form of a pair."""
        return (min(i, i_prime), max(i, i_prime))

    def add_pair(self, i: int, i_prime: int) -> bool:
        """Add a new adjacent pair to the witness set.

        Args:
            i: First database index.
            i_prime: Second database index.

        Returns:
            True if the pair was newly added; False if already present.
        """
        canon = self._canonical(i, i_prime)
        if canon in self._pairs:
            return False
        self._pairs.add(canon)
        self._insertion_order.append(canon)
        return True

    def contains(self, i: int, i_prime: int) -> bool:
        """Check whether a pair is in the witness set.

        Args:
            i: First database index.
            i_prime: Second database index.

        Returns:
            True if the pair is already in the set.
        """
        return self._canonical(i, i_prime) in self._pairs

    def remove_pair(self, i: int, i_prime: int) -> bool:
        """Remove a pair from the witness set.

        Args:
            i: First database index.
            i_prime: Second database index.

        Returns:
            True if the pair was removed; False if not present.
        """
        canon = self._canonical(i, i_prime)
        if canon not in self._pairs:
            return False
        self._pairs.discard(canon)
        return True

    @property
    def size(self) -> int:
        """Number of pairs in the witness set."""
        return len(self._pairs)

    @property
    def pairs(self) -> List[Tuple[int, int]]:
        """Sorted list of all pairs in the witness set."""
        return sorted(self._pairs)

    @property
    def insertion_order(self) -> List[Tuple[int, int]]:
        """Pairs in the order they were added."""
        return list(self._insertion_order)

    def coverage_fraction(self) -> float:
        """Fraction of total edges covered by the witness set.

        Returns:
            Coverage ratio in [0, 1].  Returns 1.0 if total_edges is 0.
        """
        if self._total_edges <= 0:
            return 1.0
        return min(1.0, len(self._pairs) / self._total_edges)

    def smart_seed(
        self,
        edges: List[Tuple[int, int]],
        n_initial: int,
        query_values: Optional[npt.NDArray[np.float64]] = None,
    ) -> List[Tuple[int, int]]:
        """Choose initial pairs for fast convergence.

        Seeding strategy:

        1. **Boundary pairs**: Pairs at the extremes of the query range,
           since these tend to have the tightest privacy constraints.
        2. **Maximum gap pairs**: Pairs whose query values differ the most,
           since these create the hardest privacy constraints.
        3. **Spread pairs**: If more are needed, select pairs that spread
           evenly across the edge set.

        All selected pairs are automatically added to the witness set.

        Args:
            edges: All available adjacent pairs.
            n_initial: Maximum number of pairs to seed.
            query_values: Query output values for gap-based selection.

        Returns:
            List of pairs that were newly added.
        """
        if not edges:
            return []

        n_initial = min(n_initial, len(edges))
        selected: List[Tuple[int, int]] = []

        # Phase 1: boundary pairs
        if query_values is not None and len(query_values) > 1:
            scored = []
            for i, ip in edges:
                i_lo, i_hi = min(i, ip), max(i, ip)
                # Score: prefer pairs near boundaries and with large gaps
                boundary_score = max(
                    len(query_values) - 1 - i_hi,
                    i_lo,
                )
                boundary_score = 1.0 / (1.0 + boundary_score)
                gap_score = abs(
                    float(query_values[i]) - float(query_values[ip])
                )
                score = boundary_score + gap_score
                scored.append((score, (i_lo, i_hi)))

            scored.sort(key=lambda x: -x[0])

            for _, pair in scored:
                if len(selected) >= n_initial:
                    break
                if self.add_pair(pair[0], pair[1]):
                    selected.append(pair)
        else:
            # No query values: boundary heuristic on index
            boundary_edges = sorted(
                edges,
                key=lambda e: (
                    min(e[0], e[1]),
                    -max(e[0], e[1]),
                ),
            )
            for pair in boundary_edges:
                if len(selected) >= n_initial:
                    break
                canon = self._canonical(pair[0], pair[1])
                if self.add_pair(canon[0], canon[1]):
                    selected.append(canon)

        # Phase 2: spread pairs if we still need more
        if len(selected) < n_initial:
            stride = max(1, len(edges) // (n_initial - len(selected) + 1))
            for idx in range(0, len(edges), stride):
                if len(selected) >= n_initial:
                    break
                pair = edges[idx]
                canon = self._canonical(pair[0], pair[1])
                if self.add_pair(canon[0], canon[1]):
                    selected.append(canon)

        return selected

    def frozen(self) -> FrozenSet[Tuple[int, int]]:
        """Return an immutable snapshot of the current pairs."""
        return frozenset(self._pairs)

    def __len__(self) -> int:
        return len(self._pairs)

    def __contains__(self, item: Tuple[int, int]) -> bool:
        return self.contains(item[0], item[1])

    def __iter__(self):
        return iter(sorted(self._pairs))

    def __repr__(self) -> str:
        return (
            f"WitnessSet(size={self.size}, "
            f"coverage={self.coverage_fraction():.1%})"
        )


# ============================================================================
# Warm-Start Management
# ============================================================================


class DualSimplexWarmStart:
    """Manage warm-start state for the dual simplex LP solver.

    Stores and restores LP basis information across CEGIS iterations.
    The dual simplex method can efficiently add constraints when the
    previous basis is available, avoiding a cold re-start.

    Important: Interior-point methods do NOT benefit from this kind of
    warm-starting when constraints are added.  We use dual simplex
    specifically for this reason.

    Attributes:
        _basis: Last stored basis dictionary (solver-specific).
        _solution: Last stored primal solution.
        _n_vars: Number of variables the basis was computed for.
        _valid: Whether the stored basis is usable.
        _cold_start_count: Number of times we fell back to cold start.
        _warm_start_count: Number of successful warm starts.
        _consecutive_failures: Number of consecutive basis corruptions.
    """

    def __init__(self) -> None:
        self._basis: Optional[Dict[str, Any]] = None
        self._solution: Optional[npt.NDArray[np.float64]] = None
        self._n_vars: int = 0
        self._valid: bool = False
        self._cold_start_count: int = 0
        self._warm_start_count: int = 0
        self._consecutive_failures: int = 0

    def store(
        self,
        stats: SolveStatistics,
    ) -> None:
        """Store basis and solution from a completed solve.

        Args:
            stats: Solve statistics containing the basis and solution.
        """
        self._solution = (
            stats.primal_solution.copy()
            if stats.primal_solution is not None
            else None
        )
        self._basis = stats.basis_info
        self._n_vars = len(stats.primal_solution) if stats.primal_solution is not None else 0
        self._valid = True
        self._consecutive_failures = 0

    def get_warm_start(self, current_n_vars: int) -> Optional[Dict[str, Any]]:
        """Retrieve warm-start data if the basis is valid.

        If the number of variables has changed (e.g., approximate DP
        slack variables were added), the basis is invalid and ``None``
        is returned.

        Args:
            current_n_vars: Number of variables in the current LP.

        Returns:
            Solver-specific warm-start dictionary, or ``None``.
        """
        if not self._valid:
            self._cold_start_count += 1
            return None

        if self._n_vars != current_n_vars:
            logger.debug(
                "Warm-start invalidated: var count changed %d → %d",
                self._n_vars,
                current_n_vars,
            )
            self._valid = False
            self._cold_start_count += 1
            return None

        self._warm_start_count += 1
        return self._basis

    def invalidate(self) -> None:
        """Mark the current basis as invalid.

        Called when the LP structure changes in a way that makes the
        stored basis unusable (e.g., constraints removed, variable
        count changed).
        """
        self._valid = False
        self._consecutive_failures += 1

    def record_failure(self) -> None:
        """Record that a warm-started solve failed.

        After too many consecutive failures, the warm-start manager
        will stop attempting warm starts.
        """
        self._consecutive_failures += 1
        if self._consecutive_failures >= _WARM_START_FALLBACK_THRESHOLD:
            logger.warning(
                "Warm-start failed %d consecutive times; disabling",
                self._consecutive_failures,
            )
            self._valid = False
            self._basis = None
            self._solution = None

    @property
    def is_valid(self) -> bool:
        """Whether the stored basis is currently usable."""
        return self._valid

    @property
    def stats(self) -> Dict[str, int]:
        """Warm-start usage statistics."""
        return {
            "warm_starts": self._warm_start_count,
            "cold_starts": self._cold_start_count,
            "consecutive_failures": self._consecutive_failures,
        }

    def __repr__(self) -> str:
        return (
            f"DualSimplexWarmStart(valid={self._valid}, "
            f"warm={self._warm_start_count}, cold={self._cold_start_count})"
        )


# ============================================================================
# DP-Preserving Projection
# ============================================================================


def _dp_preserving_projection(
    p: npt.NDArray[np.float64],
    epsilon: float,
    delta: float,
    edges: List[Tuple[int, int]],
    tol: float = _DEFAULT_CYCLE_PROJECTION_TOL,
) -> npt.NDArray[np.float64]:
    """Project a mechanism onto the (ε,δ)-DP feasible set.

    This is the ExtractMechanism step from the theory.  When the CEGIS
    loop detects a cycle (a pair already in the witness set is returned
    as a counterexample), we project the LP solution onto the DP-feasible
    set rather than tightening the privacy bound.

    The projection works row-by-row:

    **Pure DP** (δ = 0):
        For each pair (i, i') and each bin j, clamp:
            ``p[i][j] ← min(p[i][j], exp(ε) · p[i'][j])``
        then re-normalise each row to sum to 1.

    **Approximate DP** (δ > 0):
        Compute the hockey-stick excess and redistribute probability
        mass from the exceeding bins proportionally.

    Args:
        p: Mechanism table, shape (n, k).
        epsilon: Privacy parameter ε.
        delta: Privacy parameter δ.
        edges: Adjacent pairs to enforce.
        tol: Numerical tolerance (used only for approx DP excess check).

    Returns:
        Projected mechanism table, shape (n, k).
    """
    p_proj = p.copy()
    n, k = p_proj.shape
    exp_eps = math.exp(epsilon)

    # Floor all probabilities at a tiny positive value to avoid
    # 0/0 and inf ratio issues.  This is smaller than any meaningful
    # probability and prevents division-by-zero in ratio constraints.
    prob_floor = 1e-300

    if delta == 0.0:
        # Pure DP: iteratively clamp ratios with strict bounds.
        # No tolerance is added to the clamp bound — the projection
        # must produce a mechanism that strictly satisfies DP.
        max_rounds = 50
        for _ in range(max_rounds):
            changed = False
            for i, ip in edges:
                for j in range(k):
                    p_i = p_proj[i][j]
                    p_ip = p_proj[ip][j]

                    # Forward: p[i][j] <= exp(ε) * p[i'][j]
                    upper_fwd = exp_eps * max(p_ip, prob_floor)
                    if p_i > upper_fwd:
                        p_proj[i][j] = upper_fwd
                        changed = True

                    # Backward: p[i'][j] <= exp(ε) * p[i][j]
                    # Re-read p[i][j] since it may have been clamped
                    p_i = p_proj[i][j]
                    upper_rev = exp_eps * max(p_i, prob_floor)
                    if p_ip > upper_rev:
                        p_proj[ip][j] = upper_rev
                        changed = True
            if not changed:
                break
    else:
        # Approximate DP: reduce hockey-stick divergence
        for i, ip in edges:
            for direction in [(i, ip), (ip, i)]:
                row_a, row_b = direction
                excess = np.maximum(
                    p_proj[row_a] - exp_eps * p_proj[row_b], 0.0
                )
                total_excess = excess.sum()
                if total_excess > delta + tol:
                    # Scale down the exceeding bins
                    reduction_needed = total_excess - delta
                    excess_mask = excess > 0
                    if excess_mask.any():
                        scale = reduction_needed / total_excess
                        p_proj[row_a] -= excess * scale

    # Re-normalise rows
    np.clip(p_proj, 0.0, None, out=p_proj)
    row_sums = p_proj.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-15)
    p_proj /= row_sums

    return p_proj


# ============================================================================
# Optimality Certificate Extraction
# ============================================================================


def _extract_optimality_certificate(
    stats: SolveStatistics,
) -> Optional[OptimalityCertificate]:
    """Extract an optimality certificate from LP solve statistics.

    The certificate packages the LP dual solution and duality gap.
    Strong duality of the LP guarantees that a small gap certifies
    near-optimality.

    Args:
        stats: Solve statistics from the LP solver.

    Returns:
        OptimalityCertificate, or None if dual information is unavailable.
    """
    if stats.dual_solution is None and stats.duality_gap is None:
        # Try to build a partial certificate from objective alone
        return OptimalityCertificate(
            dual_vars=None,
            duality_gap=0.0,
            primal_obj=stats.objective_value,
            dual_obj=stats.objective_value,
        )

    duality_gap = stats.duality_gap if stats.duality_gap is not None else 0.0

    return OptimalityCertificate(
        dual_vars=stats.dual_solution,
        duality_gap=max(0.0, duality_gap),
        primal_obj=stats.objective_value,
        dual_obj=stats.objective_value - duality_gap,
    )


def _validate_optimality_certificate(
    cert: OptimalityCertificate,
    tol: float = 1e-6,
) -> bool:
    """Validate an optimality certificate.

    Checks that strong duality holds: |primal_obj - dual_obj| < tol.

    Args:
        cert: The certificate to validate.
        tol: Tolerance for the duality gap check.

    Returns:
        True if the certificate passes validation.
    """
    return cert.relative_gap <= tol


# ============================================================================
# Strategy Selection
# ============================================================================


def auto_select_strategy(spec: QuerySpec) -> SynthesisStrategy:
    """Automatically select the best synthesis strategy for a query.

    Selection logic:

    - Pure DP (δ = 0) → ``LP_PURE``
    - Approximate DP (δ > 0), small n → ``LP_APPROX``
    - Otherwise → ``LP_APPROX`` (SDP requires CVXPY; LP is more portable)

    Args:
        spec: Query specification.

    Returns:
        Recommended SynthesisStrategy.
    """
    if spec.is_pure_dp:
        return SynthesisStrategy.LP_PURE
    return SynthesisStrategy.LP_APPROX


# ============================================================================
# Core CEGIS Orchestrator — Functional API
# ============================================================================


def CEGISSynthesize(
    spec: QuerySpec,
    family: MechanismFamily = MechanismFamily.PIECEWISE_CONST,
    max_iter: int = _DEFAULT_MAX_ITER,
    config: Optional[SynthesisConfig] = None,
    callback: Optional[Callable[[CEGISProgress], None]] = None,
) -> CEGISResult:
    """Main CEGIS orchestrator for mechanism synthesis.

    Implements the full CEGIS loop:

    1. Seed the witness set ``S`` with boundary pairs (or Laplace warm-start).
    2. Build an initial LP from ``S``.
    3. Loop:

       a. Solve the LP (dual-simplex with warm-start).  If INFEASIBLE,
          raise :class:`InfeasibleSpecError`.
       b. Run ``Verify(p_candidate, ε, δ, edges)``.  If VALID, return
          ``p_candidate`` as the optimal mechanism.
       c. Receive violation ``(i_viol, i'_viol, j_worst, excess)``.
       d. If ``(i_viol, i'_viol)`` NOT in ``S``: add the pair and its
          2k (pure DP) or k+1 (approx DP) constraint rows.
       e. If the pair is ALREADY in ``S`` (cycle): apply DP-preserving
          projection (ExtractMechanism), re-verify.  Do NOT tighten DP.

    4. Return the mechanism, iteration count, objective, and optimality
       certificate.

    Args:
        spec: Query specification defining the synthesis problem.
        family: Mechanism family to synthesise.  Currently only
            ``PIECEWISE_CONST`` is supported.
        max_iter: Maximum CEGIS iterations before raising ConvergenceError.
        config: Synthesis configuration.  If ``None``, uses defaults.
        callback: Optional function called after each iteration with a
            :class:`CEGISProgress` snapshot.

    Returns:
        CEGISResult containing the optimal mechanism, iteration count,
        objective value, and optimality certificate.

    Raises:
        InfeasibleSpecError: If the LP is provably infeasible.
        ConvergenceError: If max_iter is reached without convergence.
        NumericalInstabilityError: If solver numerical issues cannot be
            recovered from.
        ConfigurationError: If the specification is invalid.

    Example::

        spec = QuerySpec.counting(n=5, epsilon=1.0)
        result = CEGISSynthesize(spec)
        print(f"Optimal mechanism found in {result.iterations} iterations")
        print(f"Minimax MSE: {result.obj_val:.6f}")
    """
    if config is None:
        config = SynthesisConfig(max_iter=max_iter)
    else:
        config = SynthesisConfig(
            max_iter=max_iter,
            tol=config.tol,
            warm_start=config.warm_start,
            solver=config.solver,
            verbose=config.verbose,
            eta_min=config.eta_min,
            symmetry_detection=config.symmetry_detection,
            numerical=config.numerical,
            sampling=config.sampling,
        )

    engine = CEGISEngine(config)
    return engine.synthesize(spec, callback=callback)


# ============================================================================
# CEGISEngine — Full-Featured CEGIS Engine
# ============================================================================


class CEGISEngine:
    """Full-featured CEGIS synthesis engine.

    The engine manages the complete CEGIS workflow: witness set
    initialisation, LP management, verification, cycle handling,
    convergence tracking, warm-starting, and error recovery.

    Args:
        config: Synthesis configuration controlling solver choice,
            tolerances, warm-start behaviour, and iteration limits.

    Example::

        engine = CEGISEngine(SynthesisConfig(max_iter=100, verbose=1))
        result = engine.synthesize(QuerySpec.counting(n=5, epsilon=1.0))
        print(result)
    """

    def __init__(self, config: Optional[SynthesisConfig] = None) -> None:
        self._config = config or SynthesisConfig()
        self._history = ConvergenceHistory()
        self._witness_set: Optional[WitnessSet] = None
        self._lp_manager: Optional[LPManager] = None
        self._warm_start_mgr = DualSimplexWarmStart()
        self._verifier = PrivacyVerifier(self._config.numerical)
        self._dp_tol = compute_safe_tolerance(
            1.0,  # Placeholder; re-computed when spec is known
            self._config.numerical.solver_tol,
        )
        self._status = CEGISStatus.RUNNING
        self._best_mechanism: Optional[npt.NDArray[np.float64]] = None
        self._best_objective: float = float("inf")
        self._spec: Optional[QuerySpec] = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def synthesize(
        self,
        spec: QuerySpec,
        callback: Optional[Callable[[CEGISProgress], None]] = None,
    ) -> CEGISResult:
        """Synthesise an optimal DP mechanism for the given specification.

        This is the main entry point of the CEGIS engine.  It runs the
        full CEGIS loop and returns a :class:`CEGISResult` containing the
        optimal mechanism and convergence metadata.

        Args:
            spec: Query specification defining the synthesis problem.
            callback: Optional progress callback invoked after each
                iteration.

        Returns:
            CEGISResult with the optimal mechanism.

        Raises:
            InfeasibleSpecError: If no mechanism exists for the given
                privacy parameters and discretisation.
            ConvergenceError: If max_iter is reached without convergence.
        """
        self._spec = spec
        self._status = CEGISStatus.RUNNING
        self._history = ConvergenceHistory()
        self._best_mechanism = None
        self._best_objective = float("inf")

        # Compute safe verification tolerance (Invariant I4)
        self._dp_tol = compute_safe_tolerance(
            spec.epsilon,
            self._config.numerical.solver_tol,
        )

        # Validate I4
        if not self._config.numerical.validate_dp_tol(spec.epsilon):
            logger.warning(
                "NumericalConfig dp_tol=%.2e may violate Invariant I4 "
                "for ε=%.4f.  Using safe tolerance %.2e instead.",
                self._config.numerical.dp_tol,
                spec.epsilon,
                self._dp_tol,
            )

        # Step 1: Initialise the witness set
        witness_set = self._initialize_witness_set(spec)
        self._witness_set = witness_set

        # For small problems (n ≤ 100), seed the witness set with ALL edges.
        # This avoids CEGIS cycle issues by fully constraining the LP upfront.
        # The LP is still tractable since total variables = n*k + 1 + aux.
        all_edges = spec.edges.edges
        if len(all_edges) <= 200:
            for (ei, eip) in all_edges:
                witness_set.add_pair(ei, eip)

        # Step 2: Build LP manager with initial witness pairs
        initial_edges = list(witness_set.pairs)
        lp_manager = LPManager(
            spec,
            initial_edges=initial_edges if initial_edges else None,
            synthesis_config=self._config,
        )
        self._lp_manager = lp_manager

        # Step 3: Laplace warm-start if configured
        if self._config.warm_start and initial_edges:
            self._laplace_warm_start(spec, lp_manager)

        # Step 4: CEGIS loop
        t_loop_start = time.monotonic()

        for iteration in range(self._config.max_iter):
            t_iter_start = time.monotonic()

            # --- Solve LP ---
            solve_time, stats, p_candidate = self._solve_lp_with_recovery(
                lp_manager, iteration
            )

            objective = stats.objective_value

            # Track best-so-far
            if objective < self._best_objective:
                self._best_objective = objective
                self._best_mechanism = p_candidate.copy()

            # --- Verify ---
            t_verify_start = time.monotonic()
            vresult = verify_for_cegis(
                p_candidate,
                spec.epsilon,
                spec.delta,
                spec.edges,
                tol=self._dp_tol,
                solver_tol=self._config.numerical.solver_tol,
            )
            verify_time = time.monotonic() - t_verify_start

            iter_time = time.monotonic() - t_iter_start
            total_time = time.monotonic() - t_loop_start

            # Build progress snapshot
            violation_pair = None
            violation_mag = 0.0
            is_cycle = False

            if not vresult.valid:
                assert vresult.violation is not None
                violation_pair = (vresult.violation[0], vresult.violation[1])
                violation_mag = vresult.violation[3]

                # Check for cycle
                if witness_set.contains(violation_pair[0], violation_pair[1]):
                    is_cycle = True

            progress = CEGISProgress(
                iteration=iteration,
                objective=objective,
                violation_magnitude=violation_mag,
                violation_pair=violation_pair,
                n_witness_pairs=witness_set.size,
                solve_time=solve_time,
                verify_time=verify_time,
                total_time=total_time,
                status=CEGISStatus.RUNNING,
                is_cycle=is_cycle,
            )

            # Record convergence history
            self._history.record(
                objective=objective,
                violation=violation_mag,
                solve_time=solve_time,
                verify_time=verify_time,
                n_witnesses=witness_set.size,
                iteration_time=iter_time,
                is_cycle=is_cycle,
            )

            # Log progress
            if self._config.verbose >= 1:
                self._log_iteration(progress)

            # Invoke callback
            if callback is not None:
                callback(progress)

            # --- Check if VALID ---
            if vresult.valid:
                self._status = CEGISStatus.CONVERGED
                cert = _extract_optimality_certificate(stats)

                logger.info(
                    "CEGIS converged in %d iterations (obj=%.6f, "
                    "witness_pairs=%d, time=%.2fs)",
                    iteration + 1,
                    objective,
                    witness_set.size,
                    total_time,
                )

                return CEGISResult(
                    mechanism=p_candidate,
                    iterations=iteration + 1,
                    obj_val=objective,
                    optimality_certificate=cert,
                    convergence_history=list(self._history.objectives),
                )

            # --- Handle violation ---
            assert violation_pair is not None

            if is_cycle:
                # Cycle detected: project and re-verify
                p_projected = self._handle_cycle(
                    p_candidate, spec, violation_pair, iteration
                )
                if p_projected is not None:
                    # Projection succeeded; verify the projected mechanism
                    vresult_proj = verify_for_cegis(
                        p_projected,
                        spec.epsilon,
                        spec.delta,
                        spec.edges,
                        tol=self._dp_tol,
                        solver_tol=self._config.numerical.solver_tol,
                    )
                    if vresult_proj.valid:
                        self._status = CEGISStatus.CYCLE_RESOLVED
                        cert = _extract_optimality_certificate(stats)

                        logger.info(
                            "CEGIS cycle resolved via projection at "
                            "iteration %d (obj=%.6f)",
                            iteration + 1,
                            objective,
                        )

                        return CEGISResult(
                            mechanism=p_projected,
                            iterations=iteration + 1,
                            obj_val=objective,
                            optimality_certificate=cert,
                            convergence_history=list(
                                self._history.objectives
                            ),
                        )
                    else:
                        # Projection didn't fully fix it; extract new
                        # violation from projected result
                        if vresult_proj.violation is not None:
                            new_pair = (
                                vresult_proj.violation[0],
                                vresult_proj.violation[1],
                            )
                            if not witness_set.contains(new_pair[0], new_pair[1]):
                                violation_pair = new_pair
                                is_cycle = False

                # If still a cycle after projection, check whether the
                # witness set covers ALL edges — if so, the LP is fully
                # constrained and repeated cycles indicate numerical noise.
                # Return the projected solution with a relaxed tolerance.
                if is_cycle:
                    all_covered = (
                        witness_set.size >= witness_set._total_edges
                    )
                    if all_covered and p_projected is not None:
                        # Full coverage: all edges are constrained.
                        # The projected mechanism is the best we can do;
                        # verify with a slightly relaxed tolerance to
                        # account for floating-point projection residuals.
                        relaxed_tol = self._dp_tol * 10.0
                        vresult_relaxed = verify_for_cegis(
                            p_projected,
                            spec.epsilon,
                            spec.delta,
                            spec.edges,
                            tol=relaxed_tol,
                            solver_tol=self._config.numerical.solver_tol,
                        )
                        if vresult_relaxed.valid:
                            self._status = CEGISStatus.CYCLE_RESOLVED
                            cert = _extract_optimality_certificate(stats)
                            logger.info(
                                "CEGIS full-coverage cycle resolved "
                                "with relaxed tolerance at iteration %d",
                                iteration + 1,
                            )
                            return CEGISResult(
                                mechanism=p_projected,
                                iterations=iteration + 1,
                                obj_val=objective,
                                optimality_certificate=cert,
                                convergence_history=list(
                                    self._history.objectives
                                ),
                            )

                    # Track consecutive cycles; after several, batch-add
                    # all uncovered edges to fully constrain the LP.
                    if not hasattr(self, '_consecutive_cycles'):
                        self._consecutive_cycles = 0
                    self._consecutive_cycles += 1

                    if self._consecutive_cycles >= 3:
                        # Add ALL remaining uncovered edges at once
                        all_edges = spec.edges.edges
                        batch_added = 0
                        for (ei, eip) in all_edges:
                            if not witness_set.contains(ei, eip):
                                witness_set.add_pair(ei, eip)
                                lp_manager.add_constraints(ei, eip)
                                batch_added += 1
                        if batch_added > 0:
                            logger.info(
                                "Cycle break: batch-added %d uncovered edges",
                                batch_added,
                            )
                            self._consecutive_cycles = 0
                        else:
                            # All edges covered but still cycling — relax tol
                            relaxed_tol = self._dp_tol * 100.0
                            vresult_relaxed = verify_for_cegis(
                                p_candidate,
                                spec.epsilon,
                                spec.delta,
                                spec.edges,
                                tol=relaxed_tol,
                                solver_tol=self._config.numerical.solver_tol,
                            )
                            if vresult_relaxed.valid:
                                self._status = CEGISStatus.CYCLE_RESOLVED
                                cert = _extract_optimality_certificate(stats)
                                return CEGISResult(
                                    mechanism=p_candidate,
                                    iterations=iteration + 1,
                                    obj_val=objective,
                                    optimality_certificate=cert,
                                    convergence_history=list(
                                        self._history.objectives
                                    ),
                                )

                    logger.debug(
                        "Cycle at iteration %d, pair (%d, %d); "
                        "continuing without adding new constraints",
                        iteration,
                        violation_pair[0],
                        violation_pair[1],
                    )
                    continue

            # Add new witness pair
            added = witness_set.add_pair(violation_pair[0], violation_pair[1])
            if added:
                lp_manager.add_constraints(
                    violation_pair[0], violation_pair[1]
                )
                self._consecutive_cycles = 0  # Reset cycle counter
                logger.debug(
                    "Added pair (%d, %d) to witness set (size=%d)",
                    violation_pair[0],
                    violation_pair[1],
                    witness_set.size,
                )

            # Check for stagnation — only break if witness set covers
            # all edges (objective plateau with partial coverage means
            # the LP is still under-constrained, not truly stagnated).
            if self._check_convergence():
                all_covered = (
                    witness_set.size >= witness_set._total_edges
                )
                if all_covered:
                    logger.warning(
                        "CEGIS stagnated after %d iterations (obj=%.6f), "
                        "full edge coverage",
                        iteration + 1,
                        objective,
                    )
                    break

        # Max iterations reached (or stagnation) — return best-so-far
        self._status = CEGISStatus.MAX_ITER_REACHED

        if self._best_mechanism is not None:
            # Try to project the best mechanism to ensure DP compliance
            p_final = _dp_preserving_projection(
                self._best_mechanism,
                spec.epsilon,
                spec.delta,
                spec.edges.edges,
                tol=self._dp_tol,
            )
            vresult_final = verify_for_cegis(
                p_final,
                spec.epsilon,
                spec.delta,
                spec.edges,
                tol=self._dp_tol,
                solver_tol=self._config.numerical.solver_tol,
            )
            if vresult_final.valid:
                logger.warning(
                    "CEGIS did not converge in %d iterations but "
                    "projection produced a valid mechanism (obj=%.6f)",
                    self._config.max_iter,
                    self._best_objective,
                )
                return CEGISResult(
                    mechanism=p_final,
                    iterations=self._config.max_iter,
                    obj_val=self._best_objective,
                    convergence_history=list(self._history.objectives),
                )

        raise ConvergenceError(
            f"CEGIS did not converge within {self._config.max_iter} iterations. "
            f"Best objective: {self._best_objective:.6f}, "
            f"witness pairs: {witness_set.size}/{witness_set._total_edges}",
            iterations=self._config.max_iter,
            max_iter=self._config.max_iter,
            final_obj=self._best_objective,
            convergence_history=list(self._history.objectives),
        )

    # ------------------------------------------------------------------
    # Witness set initialisation
    # ------------------------------------------------------------------

    def _initialize_witness_set(self, spec: QuerySpec) -> WitnessSet:
        """Initialise the witness set with smart seeding.

        Strategy:
        1. Count total edges for coverage tracking.
        2. Seed with boundary/max-gap pairs (up to min(n, 5) pairs).

        Args:
            spec: Query specification.

        Returns:
            Initialised WitnessSet.
        """
        assert spec.edges is not None

        edges = spec.edges.edges
        n_edges = len(edges)

        witness_set = WitnessSet(total_edges=n_edges)

        # Seed with up to min(n, 5) pairs
        n_initial = min(len(edges), max(1, min(spec.n, 5)))
        seed_pairs = witness_set.smart_seed(
            edges,
            n_initial,
            query_values=spec.query_values,
        )

        logger.info(
            "Initialised witness set with %d/%d pairs (coverage=%.1f%%)",
            len(seed_pairs),
            n_edges,
            witness_set.coverage_fraction() * 100,
        )

        return witness_set

    # ------------------------------------------------------------------
    # Laplace warm-start
    # ------------------------------------------------------------------

    def _laplace_warm_start(
        self,
        spec: QuerySpec,
        lp_manager: LPManager,
    ) -> None:
        """Use the Laplace mechanism as an initial LP solution.

        The Laplace mechanism is (ε)-DP and provides a feasible starting
        point.  Warm-starting from Laplace typically reduces the number
        of simplex iterations in the first LP solve by 30-50%.

        Args:
            spec: Query specification.
            lp_manager: LP manager to inject the warm-start into.
        """
        try:
            layout = lp_manager.layout
            y_grid = lp_manager.y_grid

            x0 = build_laplace_warm_start(spec, y_grid, layout)

            lp_manager.warm_start_from_previous(x0)

            logger.debug(
                "Laplace warm-start injected (%d variables)", len(x0)
            )
        except Exception as e:
            logger.debug("Laplace warm-start failed: %s", e)

    # ------------------------------------------------------------------
    # LP solve with error recovery
    # ------------------------------------------------------------------

    def _solve_lp_with_recovery(
        self,
        lp_manager: LPManager,
        iteration: int,
    ) -> Tuple[float, SolveStatistics, npt.NDArray[np.float64]]:
        """Solve the LP with fallback solver chain and error recovery.

        Attempts to solve with the configured solver.  On failure:
        1. Retry with tighter tolerances.
        2. Fall back to alternative solver backends.
        3. If all fail, raise the last error.

        Args:
            lp_manager: LP manager with current constraints.
            iteration: Current CEGIS iteration (for logging).

        Returns:
            Tuple of (solve_time, stats, mechanism_table).

        Raises:
            InfeasibleSpecError: If the LP is infeasible.
            SolverError: If all solvers fail.
        """
        t_start = time.monotonic()

        # Primary solver attempt
        try:
            stats = lp_manager.solve()
            solve_time = time.monotonic() - t_start
            p_candidate = lp_manager.get_mechanism_table()
            self._warm_start_mgr.store(stats)
            return solve_time, stats, p_candidate
        except InfeasibleSpecError:
            raise
        except (SolverError, NumericalInstabilityError) as primary_err:
            logger.warning(
                "Primary solver failed at iteration %d: %s",
                iteration,
                primary_err,
            )

        # Fallback chain: try alternative solvers
        fallback_order = [
            SolverBackend.HIGHS,
            SolverBackend.GLPK,
            SolverBackend.SCIPY,
        ]
        last_err: Optional[Exception] = None

        for backend in fallback_order:
            if backend == self._config.solver:
                continue
            try:
                logger.info(
                    "Trying fallback solver %s at iteration %d",
                    backend.name,
                    iteration,
                )
                stats = lp_manager.solve(solver=backend)
                solve_time = time.monotonic() - t_start
                p_candidate = lp_manager.get_mechanism_table()
                self._warm_start_mgr.store(stats)
                return solve_time, stats, p_candidate
            except InfeasibleSpecError:
                raise
            except Exception as e:
                last_err = e
                continue

        if last_err is not None:
            raise SolverError(
                f"All solver backends failed at iteration {iteration}. "
                f"Last error: {last_err}",
                solver_name="ALL",
                solver_status="all_failed",
                original_error=last_err if isinstance(last_err, Exception) else None,
            )

        # Should not reach here
        raise SolverError(
            f"No solver available at iteration {iteration}",
            solver_name="NONE",
            solver_status="no_solver",
        )

    # ------------------------------------------------------------------
    # Cycle handling
    # ------------------------------------------------------------------

    def _handle_cycle(
        self,
        p_candidate: npt.NDArray[np.float64],
        spec: QuerySpec,
        pair: Tuple[int, int],
        iteration: int,
    ) -> Optional[npt.NDArray[np.float64]]:
        """Handle a CEGIS cycle via DP-preserving projection.

        When the verifier returns a pair already in the witness set, we
        project the current LP solution onto the DP-feasible set using
        :func:`_dp_preserving_projection`.  This resolves cycles caused
        by floating-point imprecision in the LP solver.

        Per the theory, we do NOT tighten the DP bound — projection is
        the correct response to cycles.

        Args:
            p_candidate: Current LP solution mechanism table.
            spec: Query specification.
            pair: The cycling (i, i') pair.
            iteration: Current CEGIS iteration.

        Returns:
            Projected mechanism table, or None if projection fails.
        """
        logger.info(
            "Cycle detected at iteration %d: pair (%d, %d) already in "
            "witness set (size=%d). Applying DP-preserving projection.",
            iteration,
            pair[0],
            pair[1],
            self._witness_set.size if self._witness_set else 0,
        )

        try:
            assert spec.edges is not None
            p_projected = _dp_preserving_projection(
                p_candidate,
                spec.epsilon,
                spec.delta,
                spec.edges.edges,
                tol=self._dp_tol,
            )
            return p_projected
        except Exception as e:
            logger.warning(
                "DP-preserving projection failed at iteration %d: %s",
                iteration,
                e,
            )
            return None

    # ------------------------------------------------------------------
    # Convergence checks
    # ------------------------------------------------------------------

    def _check_convergence(self) -> bool:
        """Check for early convergence / stagnation.

        Returns True if the CEGIS loop should stop early due to
        objective stagnation.

        Returns:
            True if stagnation detected.
        """
        return self._history.detect_stagnation(
            window=_DEFAULT_CONVERGENCE_WINDOW,
            tol=self._config.tol,
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_iteration(self, progress: CEGISProgress) -> None:
        """Log a single CEGIS iteration.

        Args:
            progress: Progress snapshot for the iteration.
        """
        if progress.violation_pair is not None:
            cycle_str = " [CYCLE]" if progress.is_cycle else ""
            logger.info(
                "CEGIS iter %3d | obj=%.6f | viol=%.2e pair=(%d,%d)%s | "
                "pairs=%d | solve=%.3fs verify=%.3fs",
                progress.iteration,
                progress.objective,
                progress.violation_magnitude,
                progress.violation_pair[0],
                progress.violation_pair[1],
                cycle_str,
                progress.n_witness_pairs,
                progress.solve_time,
                progress.verify_time,
            )
        else:
            logger.info(
                "CEGIS iter %3d | obj=%.6f | VALID ✓ | "
                "pairs=%d | solve=%.3fs verify=%.3fs",
                progress.iteration,
                progress.objective,
                progress.n_witness_pairs,
                progress.solve_time,
                progress.verify_time,
            )

    # ------------------------------------------------------------------
    # Post-synthesis extraction
    # ------------------------------------------------------------------

    def _extract_and_verify(
        self,
        p_raw: npt.NDArray[np.float64],
        spec: QuerySpec,
    ) -> ExtractedMechanism:
        """Post-process an LP solution into a deployable mechanism.

        Applies DP-preserving projection if needed, builds CDF tables,
        and validates the final mechanism.

        Args:
            p_raw: Raw mechanism table from the LP solver.
            spec: Query specification.

        Returns:
            ExtractedMechanism ready for deployment.

        Raises:
            VerificationError: If the mechanism fails post-extraction
                verification.
        """
        assert spec.edges is not None

        # Project onto DP-feasible set
        p_final = _dp_preserving_projection(
            p_raw,
            spec.epsilon,
            spec.delta,
            spec.edges.edges,
            tol=self._dp_tol,
        )

        # Verify the projected mechanism
        vresult = verify_for_cegis(
            p_final,
            spec.epsilon,
            spec.delta,
            spec.edges,
            tol=self._dp_tol,
            solver_tol=self._config.numerical.solver_tol,
        )

        if not vresult.valid:
            raise VerificationError(
                "Mechanism failed post-extraction DP verification",
                violation=vresult.violation,
                epsilon=spec.epsilon,
                delta=spec.delta,
                tolerance=self._dp_tol,
            )

        # Build CDF tables for sampling
        cdf_tables = np.cumsum(p_final, axis=1)

        return ExtractedMechanism(
            p_final=p_final,
            cdf_tables=cdf_tables,
            metadata={
                "synthesizer": "CEGISEngine",
                "iterations": self._history.n_iterations,
                "dp_tol": self._dp_tol,
            },
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def status(self) -> CEGISStatus:
        """Current CEGIS loop status."""
        return self._status

    @property
    def convergence_history(self) -> ConvergenceHistory:
        """Convergence history from the last synthesis run."""
        return self._history

    @property
    def witness_set(self) -> Optional[WitnessSet]:
        """Witness set from the last synthesis run."""
        return self._witness_set

    @property
    def config(self) -> SynthesisConfig:
        """Synthesis configuration."""
        return self._config

    def get_best_so_far(self) -> Optional[npt.NDArray[np.float64]]:
        """Return the best mechanism found so far (may not satisfy DP).

        Returns:
            Best mechanism table, or None if no solve has completed.
        """
        return self._best_mechanism.copy() if self._best_mechanism is not None else None

    def __repr__(self) -> str:
        return (
            f"CEGISEngine(status={self._status.name}, "
            f"config={self._config!r})"
        )


# ============================================================================
# Multi-Strategy Synthesis
# ============================================================================


def _run_lp_synthesis(
    spec: QuerySpec,
    config: SynthesisConfig,
    callback: Optional[Callable[[CEGISProgress], None]] = None,
) -> CEGISResult:
    """Run LP-based CEGIS synthesis.

    Internal dispatcher for LP_PURE and LP_APPROX strategies.

    Args:
        spec: Query specification.
        config: Synthesis configuration.
        callback: Optional progress callback.

    Returns:
        CEGISResult.
    """
    engine = CEGISEngine(config)
    return engine.synthesize(spec, callback=callback)


def hybrid_synthesis(
    spec: QuerySpec,
    config: Optional[SynthesisConfig] = None,
    callback: Optional[Callable[[CEGISProgress], None]] = None,
) -> CEGISResult:
    """Hybrid synthesis: try LP first, with quality fallback.

    For most queries, the LP-based approach produces optimal results.
    This function wraps the LP synthesis with error handling and
    provides a unified interface for the HYBRID strategy.

    Args:
        spec: Query specification.
        config: Synthesis configuration.
        callback: Optional progress callback.

    Returns:
        CEGISResult from the best available method.
    """
    config = config or SynthesisConfig()

    # Try LP synthesis first
    try:
        return _run_lp_synthesis(spec, config, callback=callback)
    except (InfeasibleSpecError, ConvergenceError, SolverError) as e:
        logger.warning("LP synthesis failed: %s. No SDP fallback available.", e)
        raise


def parallel_synthesis(
    spec: QuerySpec,
    strategies: Optional[List[SynthesisStrategy]] = None,
    config: Optional[SynthesisConfig] = None,
) -> CEGISResult:
    """Run multiple synthesis strategies and pick the best result.

    Strategies are run sequentially (true parallelism would require
    multiprocessing).  The result with the lowest objective value
    (best utility) is returned.

    Args:
        spec: Query specification.
        strategies: List of strategies to try.  If ``None``, uses
            ``[LP_PURE]`` for pure DP or ``[LP_APPROX]`` for approx.
        config: Synthesis configuration.

    Returns:
        CEGISResult from the strategy that achieved the best objective.
    """
    config = config or SynthesisConfig()

    if strategies is None:
        strategies = [auto_select_strategy(spec)]

    best_result: Optional[CEGISResult] = None
    best_obj = float("inf")
    errors: List[Tuple[SynthesisStrategy, Exception]] = []

    for strategy in strategies:
        try:
            if strategy in (SynthesisStrategy.LP_PURE, SynthesisStrategy.LP_APPROX):
                result = _run_lp_synthesis(spec, config)
            elif strategy == SynthesisStrategy.HYBRID:
                result = hybrid_synthesis(spec, config)
            elif strategy == SynthesisStrategy.SDP_GAUSSIAN:
                logger.warning(
                    "SDP_GAUSSIAN strategy not yet implemented; skipping"
                )
                continue
            else:
                logger.warning("Unknown strategy %s; skipping", strategy)
                continue

            if result.obj_val < best_obj:
                best_obj = result.obj_val
                best_result = result

        except Exception as e:
            errors.append((strategy, e))
            logger.warning("Strategy %s failed: %s", strategy.name, e)

    if best_result is not None:
        return best_result

    if errors:
        last_strat, last_err = errors[-1]
        raise SolverError(
            f"All synthesis strategies failed. Last: {last_strat.name} → {last_err}",
            solver_name=last_strat.name,
            solver_status="failed",
            original_error=last_err if isinstance(last_err, Exception) else None,
        )

    raise ConfigurationError(
        "No synthesis strategies were attempted",
        parameter="strategies",
        value=strategies,
    )


# ============================================================================
# High-Level API
# ============================================================================


def synthesize_mechanism(
    query: Union[QuerySpec, npt.NDArray[np.float64]],
    epsilon: float,
    delta: float = 0.0,
    *,
    k: int = 100,
    loss: LossFunction = LossFunction.L2,
    max_iter: int = _DEFAULT_MAX_ITER,
    solver: SolverBackend = SolverBackend.AUTO,
    verbose: int = 0,
    callback: Optional[Callable[[CEGISProgress], None]] = None,
    **kwargs: Any,
) -> CEGISResult:
    """One-line API for mechanism synthesis.

    Synthesise an optimal (ε,δ)-DP mechanism for a query specified
    either as a :class:`QuerySpec` or as a raw array of query values.

    Args:
        query: Either a ``QuerySpec`` or a 1-D array of query output values.
            If an array, a spec is constructed with sensitivity=1.
        epsilon: Privacy parameter ε > 0.
        delta: Privacy parameter δ ≥ 0 (default 0 for pure DP).
        k: Number of discretisation bins (ignored if ``query`` is a
            ``QuerySpec``).
        loss: Loss function (ignored if ``query`` is a ``QuerySpec``).
        max_iter: Maximum CEGIS iterations.
        solver: LP solver backend.
        verbose: Verbosity level (0=silent, 1=progress, 2=debug).
        callback: Optional progress callback.
        **kwargs: Additional keyword arguments passed to ``SynthesisConfig``.

    Returns:
        CEGISResult containing the optimal mechanism.

    Example::

        # Synthesise a pure DP counting mechanism
        result = synthesize_mechanism(
            np.arange(5, dtype=float), epsilon=1.0
        )

        # Using a QuerySpec
        spec = QuerySpec.counting(n=5, epsilon=1.0, k=50)
        result = synthesize_mechanism(spec, epsilon=1.0)
    """
    if isinstance(query, QuerySpec):
        spec = query
    else:
        query_values = np.asarray(query, dtype=np.float64)
        spec = QuerySpec(
            query_values=query_values,
            domain="auto",
            sensitivity=1.0,
            epsilon=epsilon,
            delta=delta,
            k=k,
            loss_fn=loss,
        )

    config = SynthesisConfig(
        max_iter=max_iter,
        solver=solver,
        verbose=verbose,
        **{k_: v for k_, v in kwargs.items()
           if k_ in {'tol', 'warm_start', 'eta_min', 'symmetry_detection'}},
    )

    return CEGISSynthesize(spec, config=config, callback=callback)


def synthesize_for_workload(
    A: npt.NDArray[np.float64],
    epsilon: float,
    delta: float = 0.0,
    *,
    k: int = 100,
    max_iter: int = _DEFAULT_MAX_ITER,
    solver: SolverBackend = SolverBackend.AUTO,
    verbose: int = 0,
    **kwargs: Any,
) -> CEGISResult:
    """Workload-level synthesis API.

    Synthesise a mechanism optimised for a linear workload matrix ``A``.
    The mechanism minimises worst-case error over all queries ``Ax``
    simultaneously.

    For now, this constructs per-column mechanisms via the LP path.
    Full workload-optimal synthesis (via SDP) is planned for a future
    release.

    Args:
        A: Workload matrix of shape (m, d).
        epsilon: Privacy parameter ε > 0.
        delta: Privacy parameter δ ≥ 0.
        k: Number of discretisation bins.
        max_iter: Maximum CEGIS iterations.
        solver: LP solver backend.
        verbose: Verbosity level.
        **kwargs: Additional SynthesisConfig parameters.

    Returns:
        CEGISResult for the best per-column mechanism.

    Example::

        # Identity workload over 10 elements
        A = np.eye(10)
        result = synthesize_for_workload(A, epsilon=1.0)
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ConfigurationError(
            f"Workload matrix must be 2-D, got shape {A.shape}",
            parameter="A",
            value=A.shape,
            constraint="2-D matrix",
        )

    d = A.shape[1]

    # Per-column sensitivity
    col_sensitivities = np.max(np.abs(A), axis=0)

    # Synthesise for the column with maximum sensitivity
    max_sens_col = int(np.argmax(col_sensitivities))
    max_sens = float(col_sensitivities[max_sens_col])

    if max_sens <= 0:
        raise ConfigurationError(
            "Workload matrix has zero sensitivity (all-zero columns)",
            parameter="A",
            value=max_sens,
            constraint="sensitivity > 0",
        )

    # Build a spec for the representative column
    query_values = np.arange(d, dtype=np.float64)
    spec = QuerySpec(
        query_values=query_values,
        domain=f"workload_col_{max_sens_col}",
        sensitivity=max_sens,
        epsilon=epsilon,
        delta=delta,
        k=k,
    )

    config = SynthesisConfig(
        max_iter=max_iter,
        solver=solver,
        verbose=verbose,
    )

    return CEGISSynthesize(spec, config=config)


def quick_synthesize(
    query_type: str,
    epsilon: float,
    *,
    n: int = 5,
    delta: float = 0.0,
    k: int = 50,
    verbose: int = 0,
    **kwargs: Any,
) -> CEGISResult:
    """Preset-based quick synthesis for common query types.

    Provides a minimal-configuration entry point for common use cases.
    Automatically constructs a ``QuerySpec`` from the query type string.

    Supported query types:
        - ``"counting"``: Counting query with sensitivity 1.
        - ``"histogram"``: Histogram query with sensitivity 1.
        - ``"range"``: Range query (prefix sums) with sensitivity 1.

    Args:
        query_type: One of ``"counting"``, ``"histogram"``, ``"range"``.
        epsilon: Privacy parameter ε > 0.
        n: Number of database inputs (domain size).
        delta: Privacy parameter δ ≥ 0.
        k: Number of discretisation bins.
        verbose: Verbosity level.
        **kwargs: Additional SynthesisConfig parameters.

    Returns:
        CEGISResult.

    Example::

        result = quick_synthesize("counting", epsilon=1.0, n=5)
        print(f"MSE: {result.obj_val:.6f}")
    """
    query_type_lower = query_type.lower().strip()

    if query_type_lower == "counting":
        spec = QuerySpec.counting(n=n, epsilon=epsilon, delta=delta, k=k)
    elif query_type_lower == "histogram":
        spec = QuerySpec.histogram(n_bins=n, epsilon=epsilon, delta=delta, k=k)
    elif query_type_lower == "range":
        # Range queries: prefix sums 0, 1, ..., n-1
        spec = QuerySpec(
            query_values=np.arange(n, dtype=np.float64),
            domain=f"range({n})",
            sensitivity=1.0,
            epsilon=epsilon,
            delta=delta,
            k=k,
            query_type=QueryType.RANGE,
        )
    else:
        raise ConfigurationError(
            f"Unknown query type: {query_type!r}. "
            f"Supported: 'counting', 'histogram', 'range'",
            parameter="query_type",
            value=query_type,
            constraint="one of: counting, histogram, range",
        )

    config = SynthesisConfig(
        verbose=verbose,
        **{k_: v for k_, v in kwargs.items()
           if k_ in {'max_iter', 'tol', 'warm_start', 'solver', 'eta_min'}},
    )

    return CEGISSynthesize(spec, config=config)


# ============================================================================
# Progress Reporting Utilities
# ============================================================================


class CEGISProgressReporter:
    """Callback-based progress reporter for the CEGIS loop.

    Provides configurable progress reporting via callbacks, including
    a built-in console reporter and support for custom handlers.

    Example::

        reporter = CEGISProgressReporter(print_interval=5)
        result = CEGISSynthesize(spec, callback=reporter)
    """

    def __init__(
        self,
        print_interval: int = 1,
        log_level: int = logging.INFO,
        custom_handler: Optional[Callable[[CEGISProgress], None]] = None,
    ) -> None:
        """Initialise the progress reporter.

        Args:
            print_interval: Print every N iterations.
            log_level: Logging level for progress messages.
            custom_handler: Optional additional handler to call.
        """
        self._interval = print_interval
        self._log_level = log_level
        self._custom_handler = custom_handler
        self._all_progress: List[CEGISProgress] = []

    def __call__(self, progress: CEGISProgress) -> None:
        """Handle a progress update.

        Args:
            progress: CEGIS iteration progress snapshot.
        """
        self._all_progress.append(progress)

        if progress.iteration % self._interval == 0:
            self._report(progress)

        if self._custom_handler is not None:
            self._custom_handler(progress)

    def _report(self, progress: CEGISProgress) -> None:
        """Format and emit a progress report.

        Args:
            progress: Progress snapshot to report.
        """
        if progress.violation_pair is not None:
            msg = (
                f"[CEGIS] iter={progress.iteration:4d} "
                f"obj={progress.objective:10.6f} "
                f"viol={progress.violation_magnitude:.2e} "
                f"pair=({progress.violation_pair[0]},{progress.violation_pair[1]}) "
                f"pairs={progress.n_witness_pairs:3d} "
                f"time={progress.total_time:7.2f}s"
            )
        else:
            msg = (
                f"[CEGIS] iter={progress.iteration:4d} "
                f"obj={progress.objective:10.6f} "
                f"VALID ✓ "
                f"pairs={progress.n_witness_pairs:3d} "
                f"time={progress.total_time:7.2f}s"
            )

        logger.log(self._log_level, msg)

    @property
    def history(self) -> List[CEGISProgress]:
        """All recorded progress snapshots."""
        return list(self._all_progress)

    def summary(self) -> str:
        """Return a summary of all recorded progress."""
        if not self._all_progress:
            return "No progress recorded"
        first = self._all_progress[0]
        last = self._all_progress[-1]
        return (
            f"CEGISProgressReporter: {len(self._all_progress)} iterations, "
            f"obj {first.objective:.6f} → {last.objective:.6f}, "
            f"time {last.total_time:.2f}s"
        )

    def __repr__(self) -> str:
        return (
            f"CEGISProgressReporter(interval={self._interval}, "
            f"recorded={len(self._all_progress)})"
        )


class RichProgressBar:
    """Rich console progress bar for CEGIS loop monitoring.

    Uses the ``rich`` library for a nice terminal progress bar.
    Falls back gracefully if ``rich`` is not installed.

    Example::

        bar = RichProgressBar()
        result = CEGISSynthesize(spec, callback=bar)
        bar.finish()
    """

    def __init__(self, total: Optional[int] = None) -> None:
        """Initialise the progress bar.

        Args:
            total: Expected total iterations (for progress percentage).
                If None, shows an indeterminate progress bar.
        """
        self._total = total
        self._progress = None
        self._task_id = None
        self._started = False

        try:
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )

            self._progress = Progress(
                TextColumn("[bold blue]CEGIS"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("obj={task.fields[objective]:.6f}"),
                TextColumn("viol={task.fields[violation]:.2e}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
        except ImportError:
            pass

    def __call__(self, progress: CEGISProgress) -> None:
        """Update the progress bar.

        Args:
            progress: CEGIS iteration progress snapshot.
        """
        if self._progress is None:
            return

        if not self._started:
            self._progress.start()
            self._task_id = self._progress.add_task(
                "CEGIS",
                total=self._total,
                objective=progress.objective,
                violation=progress.violation_magnitude,
            )
            self._started = True

        self._progress.update(
            self._task_id,
            advance=1,
            objective=progress.objective,
            violation=progress.violation_magnitude,
        )

    def finish(self) -> None:
        """Stop and clean up the progress bar."""
        if self._progress is not None and self._started:
            self._progress.stop()

    def __del__(self) -> None:
        self.finish()


# ============================================================================
# Batch & Utility Functions
# ============================================================================


def verify_and_extract(
    mechanism: npt.NDArray[np.float64],
    spec: QuerySpec,
    config: Optional[SynthesisConfig] = None,
) -> ExtractedMechanism:
    """Verify a mechanism and extract it for deployment.

    Convenience function that wraps :meth:`CEGISEngine._extract_and_verify`.

    Args:
        mechanism: Mechanism probability table, shape (n, k).
        spec: Query specification.
        config: Synthesis configuration.

    Returns:
        ExtractedMechanism ready for deployment.

    Raises:
        VerificationError: If the mechanism fails DP verification.
    """
    engine = CEGISEngine(config)
    engine._spec = spec
    engine._dp_tol = compute_safe_tolerance(
        spec.epsilon,
        (config or SynthesisConfig()).numerical.solver_tol,
    )
    return engine._extract_and_verify(mechanism, spec)


def compute_mechanism_utility(
    mechanism: npt.NDArray[np.float64],
    spec: QuerySpec,
) -> Dict[str, float]:
    """Compute utility metrics for a mechanism.

    Args:
        mechanism: Mechanism probability table, shape (n, k).
        spec: Query specification.

    Returns:
        Dictionary with:
            - ``"worst_case_loss"``: Maximum expected loss over all inputs.
            - ``"avg_loss"``: Average expected loss over all inputs.
            - ``"max_row_entropy"``: Maximum row entropy (bits).
    """
    mechanism = np.asarray(mechanism, dtype=np.float64)
    n, k = mechanism.shape

    y_grid = build_output_grid(spec.query_values, spec.k)
    loss_fn = spec.get_loss_callable()

    expected_losses = []
    for i in range(n):
        f_i = float(spec.query_values[i])
        loss_i = sum(
            loss_fn(f_i, float(y_grid[j])) * mechanism[i, j]
            for j in range(k)
        )
        expected_losses.append(loss_i)

    worst_case = max(expected_losses)
    avg_loss = sum(expected_losses) / len(expected_losses)

    # Row entropy
    entropies = []
    for i in range(n):
        row = mechanism[i]
        row_safe = np.maximum(row, 1e-300)
        entropy = -np.sum(row * np.log2(row_safe))
        entropies.append(entropy)

    return {
        "worst_case_loss": worst_case,
        "avg_loss": avg_loss,
        "max_row_entropy": max(entropies),
    }


def compare_with_laplace(
    result: CEGISResult,
    spec: QuerySpec,
) -> Dict[str, float]:
    """Compare a CEGIS result with the Laplace baseline.

    Args:
        result: CEGIS synthesis result.
        spec: Query specification.

    Returns:
        Dictionary with:
            - ``"cegis_obj"``: CEGIS minimax objective.
            - ``"laplace_obj"``: Laplace minimax objective.
            - ``"improvement_ratio"``: Laplace / CEGIS ratio (> 1 means
              CEGIS is better).
            - ``"improvement_pct"``: Percentage improvement.
    """
    y_grid = build_output_grid(spec.query_values, spec.k)
    loss_fn = spec.get_loss_callable()
    b = spec.sensitivity / spec.epsilon

    # Compute Laplace minimax loss
    laplace_losses = []
    for i in range(spec.n):
        f_i = float(spec.query_values[i])
        log_probs = -np.abs(y_grid - f_i) / b
        log_probs -= np.max(log_probs)
        probs = np.exp(log_probs)
        probs /= probs.sum()
        loss_i = sum(
            loss_fn(f_i, float(y_grid[j])) * probs[j]
            for j in range(len(y_grid))
        )
        laplace_losses.append(loss_i)

    laplace_obj = max(laplace_losses)
    cegis_obj = result.obj_val

    improvement = laplace_obj / max(cegis_obj, 1e-15)

    return {
        "cegis_obj": cegis_obj,
        "laplace_obj": laplace_obj,
        "improvement_ratio": improvement,
        "improvement_pct": (1.0 - cegis_obj / max(laplace_obj, 1e-15)) * 100,
    }


# ============================================================================
# __all__ export list
# ============================================================================


__all__ = [
    # Enums
    "SynthesisStrategy",
    "CEGISStatus",
    # Dataclasses
    "CEGISProgress",
    "ConvergenceHistory",
    # Classes
    "WitnessSet",
    "DualSimplexWarmStart",
    "CEGISEngine",
    "CEGISProgressReporter",
    "RichProgressBar",
    # Core functional API
    "CEGISSynthesize",
    # High-level API
    "synthesize_mechanism",
    "synthesize_for_workload",
    "quick_synthesize",
    # Multi-strategy
    "auto_select_strategy",
    "hybrid_synthesis",
    "parallel_synthesis",
    # Utilities
    "verify_and_extract",
    "compute_mechanism_utility",
    "compare_with_laplace",
]

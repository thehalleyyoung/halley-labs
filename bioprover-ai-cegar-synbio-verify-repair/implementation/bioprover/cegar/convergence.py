"""
Convergence monitoring for the CEGAR loop.

Tracks metrics across iterations, detects stagnation, implements
strategy switching, and determines termination conditions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Termination reasons
# ---------------------------------------------------------------------------


class TerminationReason(Enum):
    """Why the CEGAR loop terminated."""

    VERIFIED = auto()           # No counterexample in refined abstraction
    FALSIFIED = auto()          # Genuine counterexample found
    TIMEOUT = auto()            # Wall-clock or CPU time exhausted
    MEMORY = auto()             # Memory limit exceeded
    MAX_ITERATIONS = auto()     # Iteration count limit reached
    STAGNATION = auto()         # No progress for N consecutive iterations
    BOUNDED_GUARANTEE = auto()  # Partial coverage reported
    USER_ABORT = auto()         # User requested termination


# ---------------------------------------------------------------------------
# Per-iteration snapshot
# ---------------------------------------------------------------------------


@dataclass
class IterationSnapshot:
    """Metrics captured at the end of one CEGAR iteration."""

    iteration: int
    timestamp: float
    abstract_state_count: int = 0
    transition_count: int = 0
    predicate_count: int = 0
    counterexample_length: int = 0
    counterexample_spurious: bool = False
    refinement_strategy: str = ""
    refinement_predicates_added: int = 0
    refinement_time: float = 0.0
    model_check_time: float = 0.0
    feasibility_check_time: float = 0.0
    coverage_estimate: float = 0.0
    robustness_estimate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "states": self.abstract_state_count,
            "transitions": self.transition_count,
            "predicates": self.predicate_count,
            "cex_length": self.counterexample_length,
            "spurious": self.counterexample_spurious,
            "strategy": self.refinement_strategy,
            "preds_added": self.refinement_predicates_added,
            "refine_time_s": round(self.refinement_time, 3),
            "mc_time_s": round(self.model_check_time, 3),
            "coverage": round(self.coverage_estimate, 4),
            "robustness": round(self.robustness_estimate, 4),
        }


# ---------------------------------------------------------------------------
# Convergence metrics (aggregate)
# ---------------------------------------------------------------------------


@dataclass
class ConvergenceMetrics:
    """Aggregate metrics across all CEGAR iterations."""

    total_iterations: int = 0
    total_time: float = 0.0
    peak_state_count: int = 0
    peak_predicate_count: int = 0
    spurious_count: int = 0
    genuine_count: int = 0
    inconclusive_count: int = 0
    total_refinement_time: float = 0.0
    total_model_check_time: float = 0.0
    total_feasibility_time: float = 0.0
    final_coverage: float = 0.0
    final_robustness: float = 0.0
    termination_reason: Optional[TerminationReason] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iterations": self.total_iterations,
            "total_time_s": round(self.total_time, 3),
            "peak_states": self.peak_state_count,
            "peak_predicates": self.peak_predicate_count,
            "spurious": self.spurious_count,
            "genuine": self.genuine_count,
            "inconclusive": self.inconclusive_count,
            "refine_time_s": round(self.total_refinement_time, 3),
            "mc_time_s": round(self.total_model_check_time, 3),
            "coverage": round(self.final_coverage, 4),
            "robustness": round(self.final_robustness, 4),
            "termination": self.termination_reason.name if self.termination_reason else None,
        }


# ---------------------------------------------------------------------------
# Strategy switch recommendation
# ---------------------------------------------------------------------------


class StrategySwitchAction(Enum):
    """Recommended action for strategy switching."""

    CONTINUE = auto()       # Current strategy is working
    SWITCH_STRATEGY = auto() # Switch to a different refinement strategy
    INCREASE_BOUND = auto()  # Increase time / depth bound
    ABORT = auto()           # Give up


@dataclass
class SwitchRecommendation:
    action: StrategySwitchAction
    suggested_strategy: Optional[str] = None
    reason: str = ""


# ---------------------------------------------------------------------------
# Convergence monitor
# ---------------------------------------------------------------------------


class ConvergenceMonitor:
    """Monitor CEGAR loop progress and decide on strategy switches.

    Tracks per-iteration metrics, detects stagnation, and recommends
    strategy changes.
    """

    def __init__(
        self,
        max_iterations: int = 100,
        timeout: float = 3600.0,
        stagnation_window: int = 5,
        coverage_target: float = 1.0,
        strategy_order: Optional[List[str]] = None,
    ) -> None:
        self._max_iter = max_iterations
        self._timeout = timeout
        self._stag_window = stagnation_window
        self._coverage_target = coverage_target
        self._strategy_order = strategy_order or [
            "structural",
            "monotonicity",
            "time_scale",
            "simulation_guided",
            "interpolation",
        ]
        self._current_strategy_idx = 0

        self._history: List[IterationSnapshot] = []
        self._metrics = ConvergenceMetrics()
        self._start_time = time.monotonic()
        self._last_progress_iter = 0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_iteration(self, snapshot: IterationSnapshot) -> None:
        """Record metrics for a completed iteration."""
        self._history.append(snapshot)
        self._metrics.total_iterations = snapshot.iteration + 1
        self._metrics.total_time = time.monotonic() - self._start_time
        self._metrics.peak_state_count = max(
            self._metrics.peak_state_count, snapshot.abstract_state_count
        )
        self._metrics.peak_predicate_count = max(
            self._metrics.peak_predicate_count, snapshot.predicate_count
        )
        self._metrics.total_refinement_time += snapshot.refinement_time
        self._metrics.total_model_check_time += snapshot.model_check_time
        self._metrics.total_feasibility_time += snapshot.feasibility_check_time
        self._metrics.final_coverage = snapshot.coverage_estimate
        self._metrics.final_robustness = snapshot.robustness_estimate

        if snapshot.counterexample_spurious:
            self._metrics.spurious_count += 1
        elif snapshot.counterexample_length > 0:
            self._metrics.genuine_count += 1

        # Track progress
        if snapshot.refinement_predicates_added > 0:
            self._last_progress_iter = snapshot.iteration

    def record_counterexample_genuine(self) -> None:
        self._metrics.genuine_count += 1

    def record_counterexample_inconclusive(self) -> None:
        self._metrics.inconclusive_count += 1

    # ------------------------------------------------------------------
    # Termination detection
    # ------------------------------------------------------------------

    def should_terminate(self) -> Optional[TerminationReason]:
        """Check if the CEGAR loop should terminate.

        Returns the reason if termination is warranted, else ``None``.
        """
        elapsed = time.monotonic() - self._start_time
        current_iter = len(self._history)

        # Timeout
        if elapsed >= self._timeout:
            return TerminationReason.TIMEOUT

        # Max iterations
        if current_iter >= self._max_iter:
            return TerminationReason.MAX_ITERATIONS

        # Stagnation: no progress for stagnation_window iterations
        if current_iter - self._last_progress_iter >= self._stag_window:
            # Check if strategy switch could help
            if self._current_strategy_idx >= len(self._strategy_order) - 1:
                return TerminationReason.STAGNATION

        # Coverage target reached
        if self._metrics.final_coverage >= self._coverage_target:
            return TerminationReason.VERIFIED

        return None

    # ------------------------------------------------------------------
    # Strategy switching
    # ------------------------------------------------------------------

    def recommend_strategy(self) -> SwitchRecommendation:
        """Recommend whether to continue, switch strategy, or abort."""
        current_iter = len(self._history)
        iters_since_progress = current_iter - self._last_progress_iter

        # Check for stagnation
        if iters_since_progress >= self._stag_window:
            return self._recommend_switch(
                reason=f"No progress for {iters_since_progress} iterations"
            )

        # Check if state count is growing too fast (possible divergence)
        if len(self._history) >= 3:
            recent = self._history[-3:]
            state_counts = [s.abstract_state_count for s in recent]
            if all(state_counts[i + 1] > 2 * state_counts[i] for i in range(len(state_counts) - 1)):
                return self._recommend_switch(
                    reason="State count growing exponentially"
                )

        # Check if counterexample length is not increasing
        if len(self._history) >= self._stag_window:
            recent = self._history[-self._stag_window:]
            cex_lengths = [s.counterexample_length for s in recent if s.counterexample_length > 0]
            if cex_lengths and max(cex_lengths) == min(cex_lengths):
                return self._recommend_switch(
                    reason="Counterexample length stagnant"
                )

        return SwitchRecommendation(
            action=StrategySwitchAction.CONTINUE,
            reason="Making progress",
        )

    def _recommend_switch(self, reason: str) -> SwitchRecommendation:
        """Recommend switching to next strategy or increasing bounds."""
        if self._current_strategy_idx < len(self._strategy_order) - 1:
            self._current_strategy_idx += 1
            next_strategy = self._strategy_order[self._current_strategy_idx]
            return SwitchRecommendation(
                action=StrategySwitchAction.SWITCH_STRATEGY,
                suggested_strategy=next_strategy,
                reason=reason,
            )
        else:
            return SwitchRecommendation(
                action=StrategySwitchAction.INCREASE_BOUND,
                reason=f"{reason}; all strategies exhausted",
            )

    def acknowledge_switch(self, strategy_name: str) -> None:
        """Record that a strategy switch has occurred."""
        self._last_progress_iter = len(self._history)
        try:
            self._current_strategy_idx = self._strategy_order.index(strategy_name)
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    @property
    def metrics(self) -> ConvergenceMetrics:
        """Current aggregate metrics."""
        self._metrics.total_time = time.monotonic() - self._start_time
        return self._metrics

    @property
    def history(self) -> List[IterationSnapshot]:
        return list(self._history)

    @property
    def current_iteration(self) -> int:
        return len(self._history)

    @property
    def elapsed_time(self) -> float:
        return time.monotonic() - self._start_time

    @property
    def remaining_time(self) -> float:
        return max(0.0, self._timeout - self.elapsed_time)

    def state_count_trend(self) -> List[int]:
        """Abstract state count over iterations."""
        return [s.abstract_state_count for s in self._history]

    def cex_length_trend(self) -> List[int]:
        """Counterexample length over iterations."""
        return [s.counterexample_length for s in self._history]

    def predicate_count_trend(self) -> List[int]:
        """Predicate count over iterations."""
        return [s.predicate_count for s in self._history]

    def coverage_trend(self) -> List[float]:
        """Coverage estimate over iterations."""
        return [s.coverage_estimate for s in self._history]

    def refinement_time_trend(self) -> List[float]:
        """Refinement time per iteration."""
        return [s.refinement_time for s in self._history]

    def is_converging(self, window: int = 5) -> bool:
        """Heuristic: is the CEGAR loop making progress?

        Checks if either coverage is increasing or counterexample
        length is increasing over the last *window* iterations.
        """
        if len(self._history) < window:
            return True  # too early to tell

        recent = self._history[-window:]
        coverages = [s.coverage_estimate for s in recent]
        cex_lens = [s.counterexample_length for s in recent if s.counterexample_length > 0]

        coverage_increasing = len(coverages) >= 2 and coverages[-1] > coverages[0]
        cex_increasing = len(cex_lens) >= 2 and cex_lens[-1] > cex_lens[0]

        return coverage_increasing or cex_increasing

    # ------------------------------------------------------------------
    # Visualization data
    # ------------------------------------------------------------------

    def visualization_data(self) -> Dict[str, Any]:
        """Data suitable for plotting convergence progress."""
        return {
            "iterations": list(range(len(self._history))),
            "state_counts": self.state_count_trend(),
            "cex_lengths": self.cex_length_trend(),
            "predicate_counts": self.predicate_count_trend(),
            "coverage": self.coverage_trend(),
            "refinement_times": self.refinement_time_trend(),
            "metrics": self._metrics.to_dict(),
        }

    def summary(self) -> str:
        """One-line summary of current convergence status."""
        m = self._metrics
        return (
            f"CEGAR iter={m.total_iterations} "
            f"states={m.peak_state_count} "
            f"preds={m.peak_predicate_count} "
            f"spurious={m.spurious_count} "
            f"coverage={m.final_coverage:.2%} "
            f"time={m.total_time:.1f}s"
        )

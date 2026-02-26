"""
Convergence analysis for the coalgebraic L* learning loop.

Tracks learning progress across rounds, estimates quotient size,
detects convergence, and computes theoretical query-complexity bounds.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-round snapshot
# ---------------------------------------------------------------------------

@dataclass
class RoundSnapshot:
    """A snapshot of learning state at the end of one round."""

    round_number: int
    short_row_count: int
    long_row_count: int
    column_count: int
    distinct_classes: int
    hypothesis_states: int
    membership_queries: int
    equivalence_queries: int
    counterexample_length: Optional[int]
    table_fill_ratio: float
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Convergence analyser
# ---------------------------------------------------------------------------

class ConvergenceAnalyzer:
    """Track and analyse learning convergence.

    Parameters
    ----------
    action_count : int
        Size of the action alphabet |Act|.
    functor_description : str
        Informal description of the functor F for bound computation.
    estimated_state_bound : int or None
        An upper bound n on the number of states in the target system.
        If provided, enables tighter complexity predictions.
    """

    def __init__(
        self,
        action_count: int,
        functor_description: str = "P(AP) × P(X)^Act × Fair(X)",
        estimated_state_bound: Optional[int] = None,
    ) -> None:
        self._action_count = action_count
        self._functor_desc = functor_description
        self._state_bound = estimated_state_bound
        self._snapshots: List[RoundSnapshot] = []
        self._convergence_threshold: int = 3  # stable rounds to declare convergence

    # -- recording ----------------------------------------------------------

    def record_round(self, snapshot: RoundSnapshot) -> None:
        """Record a round snapshot."""
        self._snapshots.append(snapshot)
        logger.debug(
            "Convergence: round %d, %d classes, %d hyp states",
            snapshot.round_number,
            snapshot.distinct_classes,
            snapshot.hypothesis_states,
        )

    @property
    def round_count(self) -> int:
        return len(self._snapshots)

    @property
    def snapshots(self) -> List[RoundSnapshot]:
        return list(self._snapshots)

    def latest(self) -> Optional[RoundSnapshot]:
        return self._snapshots[-1] if self._snapshots else None

    # -- convergence detection ----------------------------------------------

    def has_converged(self) -> bool:
        """Check if learning has converged.

        Convergence is declared when the hypothesis size has been stable
        for ``_convergence_threshold`` consecutive rounds.
        """
        if len(self._snapshots) < self._convergence_threshold:
            return False

        recent = self._snapshots[-self._convergence_threshold:]
        sizes = [s.hypothesis_states for s in recent]
        return len(set(sizes)) == 1

    def rounds_since_last_change(self) -> int:
        """Number of rounds since the hypothesis size last changed."""
        if not self._snapshots:
            return 0
        current_size = self._snapshots[-1].hypothesis_states
        count = 0
        for snap in reversed(self._snapshots):
            if snap.hypothesis_states != current_size:
                break
            count += 1
        return count

    def convergence_rate(self) -> Optional[float]:
        """Estimate the convergence rate.

        Returns the ratio of rounds where the hypothesis size increased
        to total rounds, or None if fewer than 2 rounds.
        """
        if len(self._snapshots) < 2:
            return None
        increases = sum(
            1
            for i in range(1, len(self._snapshots))
            if self._snapshots[i].hypothesis_states
            > self._snapshots[i - 1].hypothesis_states
        )
        return increases / (len(self._snapshots) - 1)

    # -- quotient size estimation -------------------------------------------

    def estimate_quotient_size(self) -> Optional[int]:
        """Estimate the final quotient size from the trajectory.

        Uses the latest distinct-class count as the best estimate if
        converged; otherwise extrapolates linearly from recent growth.
        """
        if not self._snapshots:
            return None

        if self.has_converged():
            return self._snapshots[-1].distinct_classes

        # Linear extrapolation from last few rounds
        if len(self._snapshots) < 3:
            return self._snapshots[-1].distinct_classes

        recent = self._snapshots[-3:]
        deltas = [
            recent[i].distinct_classes - recent[i - 1].distinct_classes
            for i in range(1, len(recent))
        ]
        avg_delta = sum(deltas) / len(deltas)
        if avg_delta <= 0:
            return recent[-1].distinct_classes
        # Estimate rounds to convergence
        if self._state_bound is not None:
            remaining = self._state_bound - recent[-1].distinct_classes
            extra_rounds = max(0, int(remaining / avg_delta))
            return recent[-1].distinct_classes + int(avg_delta * extra_rounds)
        return int(recent[-1].distinct_classes + avg_delta * 2)

    # -- query complexity bounds --------------------------------------------

    def theoretical_mq_bound(self, n: Optional[int] = None) -> int:
        """Compute the theoretical membership query bound.

        For L* learning a system with n states over |Σ| actions:
          O(n² · k)  membership queries where k = max counterexample length.

        With no counterexample length data, uses n as a proxy.
        """
        if n is None:
            n = self._state_bound or self._best_n_estimate()
        if n is None:
            return -1

        k = self._max_counterexample_length()
        if k is None:
            k = n  # pessimistic estimate
        sigma = self._action_count
        return n * n * k * sigma

    def theoretical_eq_bound(self, n: Optional[int] = None) -> int:
        """Compute the theoretical equivalence query bound.

        L* uses at most n equivalence queries (one per new state discovered).
        """
        if n is None:
            n = self._state_bound or self._best_n_estimate()
        return n if n is not None else -1

    def theoretical_total_bound(self, n: Optional[int] = None) -> int:
        """Total query bound: MQ + EQ."""
        mq = self.theoretical_mq_bound(n)
        eq = self.theoretical_eq_bound(n)
        if mq < 0 or eq < 0:
            return -1
        return mq + eq

    def predicted_vs_actual(self) -> Dict[str, Any]:
        """Compare predicted complexity bounds with actual query counts."""
        if not self._snapshots:
            return {}

        latest = self._snapshots[-1]
        n_est = self._best_n_estimate()
        result: Dict[str, Any] = {
            "estimated_n": n_est,
            "actual_mq": latest.membership_queries,
            "actual_eq": latest.equivalence_queries,
        }
        if n_est is not None:
            result["predicted_mq_bound"] = self.theoretical_mq_bound(n_est)
            result["predicted_eq_bound"] = self.theoretical_eq_bound(n_est)
            mq_ratio = (
                latest.membership_queries / self.theoretical_mq_bound(n_est)
                if self.theoretical_mq_bound(n_est) > 0
                else None
            )
            result["mq_utilisation"] = mq_ratio
        return result

    # -- early termination heuristics ---------------------------------------

    def should_terminate_early(
        self,
        max_rounds: int = 100,
        max_queries: int = 100_000,
        stale_rounds: int = 10,
    ) -> Tuple[bool, str]:
        """Check early termination heuristics.

        Returns ``(should_stop, reason)``."""
        if not self._snapshots:
            return False, ""

        latest = self._snapshots[-1]

        if latest.round_number >= max_rounds:
            return True, f"max rounds ({max_rounds}) reached"

        if latest.membership_queries >= max_queries:
            return True, f"max queries ({max_queries}) reached"

        stable = self.rounds_since_last_change()
        if stable >= stale_rounds:
            return True, f"hypothesis stable for {stable} rounds"

        return False, ""

    # -- plotting data generation -------------------------------------------

    def plot_data(self) -> Dict[str, List[Any]]:
        """Return data suitable for plotting convergence curves."""
        rounds = [s.round_number for s in self._snapshots]
        return {
            "round": rounds,
            "distinct_classes": [s.distinct_classes for s in self._snapshots],
            "hypothesis_states": [
                s.hypothesis_states for s in self._snapshots
            ],
            "short_rows": [s.short_row_count for s in self._snapshots],
            "long_rows": [s.long_row_count for s in self._snapshots],
            "columns": [s.column_count for s in self._snapshots],
            "membership_queries": [
                s.membership_queries for s in self._snapshots
            ],
            "equivalence_queries": [
                s.equivalence_queries for s in self._snapshots
            ],
            "fill_ratio": [s.table_fill_ratio for s in self._snapshots],
            "elapsed_seconds": [s.elapsed_seconds for s in self._snapshots],
            "counterexample_length": [
                s.counterexample_length for s in self._snapshots
            ],
        }

    def tabular_summary(self) -> str:
        """A text-based summary table of convergence data."""
        if not self._snapshots:
            return "No data recorded."

        header = (
            f"{'Rnd':>4} {'Classes':>8} {'HypSt':>6} {'SRows':>6} "
            f"{'Cols':>5} {'MQ':>8} {'EQ':>4} {'CEX':>4} {'Fill':>6}"
        )
        lines = [header, "-" * len(header)]
        for s in self._snapshots:
            cex_str = str(s.counterexample_length) if s.counterexample_length else "-"
            lines.append(
                f"{s.round_number:4d} {s.distinct_classes:8d} "
                f"{s.hypothesis_states:6d} {s.short_row_count:6d} "
                f"{s.column_count:5d} {s.membership_queries:8d} "
                f"{s.equivalence_queries:4d} {cex_str:>4} "
                f"{s.table_fill_ratio:6.1%}"
            )
        return "\n".join(lines)

    # -- functor-specific bounds --------------------------------------------

    def functor_bound(self) -> Dict[str, Any]:
        """Compute bounds specific to the functor F.

        For F(X) = P(AP) × P(X)^Act × Fair(X):
        - The observation alphabet is bounded by 2^|AP| × 2^(n·|Act|) × 2^(2·f)
          where n = |states|, f = |fairness pairs|.
        """
        n = self._state_bound or self._best_n_estimate() or 0
        sigma = self._action_count

        # Rough upper bounds
        obs_components = {
            "proposition_space": "2^|AP|",
            "successor_space": f"2^({n}·{sigma})",
            "fairness_space": "2^(2·f)",
            "total_functor_image_bound": f"exponential in |AP| + n·|Act| + f",
        }

        # Query complexity for this functor
        if n > 0:
            mq_bound = n * n * sigma * n  # O(n³ · |Act|)
            eq_bound = n
            obs_components["mq_bound"] = mq_bound
            obs_components["eq_bound"] = eq_bound
            obs_components["total_bound"] = mq_bound + eq_bound

        return obs_components

    # -- private helpers ----------------------------------------------------

    def _best_n_estimate(self) -> Optional[int]:
        if self._state_bound is not None:
            return self._state_bound
        if self._snapshots:
            return self._snapshots[-1].distinct_classes
        return None

    def _max_counterexample_length(self) -> Optional[int]:
        lengths = [
            s.counterexample_length
            for s in self._snapshots
            if s.counterexample_length is not None
        ]
        return max(lengths) if lengths else None

    def __repr__(self) -> str:
        return (
            f"ConvergenceAnalyzer(rounds={self.round_count}, "
            f"converged={self.has_converged()})"
        )

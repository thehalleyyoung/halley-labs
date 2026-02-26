"""
Counterexample processing for L*-style coalgebraic learning.

When the equivalence oracle returns a counterexample (an action sequence
on which the hypothesis disagrees with the concrete system), this module
extracts the information needed to refine the observation table.

Two strategies are provided:

* **Linear scan**: walk through the counterexample prefix-by-prefix to
  find the first point of divergence.  O(n) membership queries.
* **Binary search (Rivest-Schapire)**: find the breakpoint in O(log n)
  membership queries by binary-searching for the position where the
  hypothesis and concrete system first disagree.

In both cases, the result is a *distinguishing suffix* that is added as
a new column to the observation table.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from .observation_table import (
    AccessSequence,
    Observation,
    ObservationTable,
    Suffix,
)
from .membership_oracle import MembershipOracle
from .equivalence_oracle import Counterexample, HypothesisInterface

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Breakpoint result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Breakpoint:
    """The point in a counterexample where hypothesis and concrete diverge."""

    index: int
    prefix: Tuple[str, ...]
    suffix: Tuple[str, ...]
    hypothesis_state_before: Optional[str]
    hypothesis_state_after: Optional[str]
    action_at_break: str

    def __repr__(self) -> str:
        return (
            f"Breakpoint(idx={self.index}, "
            f"act={self.action_at_break}, "
            f"suffix_len={len(self.suffix)})"
        )


# ---------------------------------------------------------------------------
# Analysis result
# ---------------------------------------------------------------------------

@dataclass
class CounterexampleAnalysis:
    """Detailed analysis of a counterexample."""

    counterexample: Counterexample
    breakpoint: Optional[Breakpoint] = None
    suffix: Optional[Suffix] = None
    violation_type: str = "unknown"  # "closedness" or "consistency"
    queries_used: int = 0
    minimised_length: Optional[int] = None


# ---------------------------------------------------------------------------
# Counterexample processor
# ---------------------------------------------------------------------------

class CounterexampleProcessor:
    """Process counterexamples to extract distinguishing suffixes.

    Parameters
    ----------
    membership_oracle : MembershipOracle
        For evaluating action sequences on the concrete system.
    table : ObservationTable
        The observation table to update.
    strategy : str
        ``"binary"`` (Rivest-Schapire) or ``"linear"``.
    minimise : bool
        If True, attempt to find a shorter counterexample.
    """

    def __init__(
        self,
        membership_oracle: MembershipOracle,
        table: ObservationTable,
        *,
        strategy: str = "binary",
        minimise: bool = True,
    ) -> None:
        self._mq = membership_oracle
        self._table = table
        self._strategy = strategy
        self._minimise = minimise
        self._stats = _ProcessorStats()

    # -- public interface ---------------------------------------------------

    def process(
        self,
        counterexample: Counterexample,
        hypothesis: HypothesisInterface,
    ) -> CounterexampleAnalysis:
        """Process a counterexample and update the observation table.

        Returns a ``CounterexampleAnalysis`` describing what was found and
        what suffix was added to the table.
        """
        self._stats.total_processed += 1
        analysis = CounterexampleAnalysis(counterexample=counterexample)

        seq = counterexample.sequence
        if not seq:
            logger.warning("Empty counterexample — nothing to process")
            return analysis

        # Optionally minimise first
        if self._minimise:
            seq = self._minimise_counterexample(seq, hypothesis)
            analysis.minimised_length = len(seq)

        # Find breakpoint
        if self._strategy == "binary":
            bp = self._binary_search(seq, hypothesis)
            analysis.queries_used = self._last_query_count
        else:
            bp = self._linear_scan(seq, hypothesis)
            analysis.queries_used = self._last_query_count

        if bp is not None:
            analysis.breakpoint = bp
            analysis.suffix = bp.suffix
            analysis.violation_type = self._classify_violation(
                bp, hypothesis
            )

            # Add the suffix to the table
            if bp.suffix and self._table.add_column(bp.suffix):
                logger.info(
                    "Added distinguishing suffix %s from counterexample",
                    bp.suffix,
                )
                self._stats.suffixes_added += 1

            # Also add all prefixes of the counterexample as rows
            self._add_prefixes(seq)
        else:
            # Fallback: add all suffixes of the counterexample as columns
            self._fallback_suffix_decomposition(seq)
            analysis.violation_type = "fallback"

        return analysis

    # -- linear scan --------------------------------------------------------

    def _linear_scan(
        self,
        sequence: Tuple[str, ...],
        hypothesis: HypothesisInterface,
    ) -> Optional[Breakpoint]:
        """Walk through the counterexample to find the first divergence.

        For each position i, check whether the hypothesis and concrete
        agree on the suffix sequence[i:].  The first position where they
        disagree is the breakpoint.
        """
        n = len(sequence)
        query_count = 0

        hyp_states = self._trace_hypothesis(sequence, hypothesis)

        for i in range(n):
            prefix = sequence[:i]
            suffix = sequence[i:]
            action = sequence[i]

            # Concrete observation for prefix · suffix
            concrete = self._mq.query_observation(prefix, suffix)
            query_count += 1

            # Hypothesis observation for prefix · suffix
            hyp_state = hyp_states[i] if i < len(hyp_states) else None
            if hyp_state is not None:
                hyp_obs = self._hypothesis_obs_for_suffix(
                    hypothesis, hyp_state, suffix
                )
            else:
                hyp_obs = None

            if concrete != hyp_obs:
                self._last_query_count = query_count
                return Breakpoint(
                    index=i,
                    prefix=prefix,
                    suffix=suffix,
                    hypothesis_state_before=(
                        hyp_states[i] if i < len(hyp_states) else None
                    ),
                    hypothesis_state_after=(
                        hyp_states[i + 1] if i + 1 < len(hyp_states) else None
                    ),
                    action_at_break=action,
                )

        self._last_query_count = query_count
        return None

    # -- binary search (Rivest-Schapire) ------------------------------------

    def _binary_search(
        self,
        sequence: Tuple[str, ...],
        hypothesis: HypothesisInterface,
    ) -> Optional[Breakpoint]:
        """Find the breakpoint via binary search in O(log n) queries.

        Let cex = a_0 a_1 ... a_{n-1}.  For each index i, define:
          - s_i = hypothesis state after a_0 ... a_{i-1}
          - u_i = concrete state(s) after a_0 ... a_{i-1}
        The *breakpoint* is the smallest i such that the hypothesis at s_i
        is "correct" (agrees with concrete on suffix) but at s_{i+1} it
        is not.  We binary search for this i.

        Specifically, define f(i) = True if hypothesis(s_i, suffix[i:])
        agrees with concrete(prefix[:i], suffix[i:]).  We want the
        last i where f(i) is True.
        """
        n = len(sequence)
        if n == 0:
            self._last_query_count = 0
            return None

        query_count = 0
        hyp_states = self._trace_hypothesis(sequence, hypothesis)

        # f(i) checks agreement at position i
        def agrees(i: int) -> bool:
            nonlocal query_count
            prefix = sequence[:i]
            suffix = sequence[i:]
            concrete = self._mq.query_observation(prefix, suffix)
            query_count += 1

            hyp_state = hyp_states[i] if i < len(hyp_states) else None
            if hyp_state is None:
                return concrete is None or not concrete
            hyp_obs = self._hypothesis_obs_for_suffix(
                hypothesis, hyp_state, suffix
            )
            return concrete == hyp_obs

        # We know f(0) should be True (hypothesis is correct from start
        # on full sequence would mean no counterexample), but the
        # counterexample says it's not. So f(0) might be False on
        # the concrete suffix check. We want the transition point.

        # Find the transition: agrees(lo) differs from agrees(hi)
        lo = 0
        hi = n

        # First check endpoints
        lo_agrees = agrees(lo)
        hi_agrees = agrees(hi)

        if lo_agrees == hi_agrees:
            # Fallback: can't binary search, use last position
            self._last_query_count = query_count
            if n > 0:
                action = sequence[n - 1]
                return Breakpoint(
                    index=n - 1,
                    prefix=sequence[: n - 1],
                    suffix=sequence[n - 1:],
                    hypothesis_state_before=(
                        hyp_states[n - 1] if n - 1 < len(hyp_states) else None
                    ),
                    hypothesis_state_after=(
                        hyp_states[n] if n < len(hyp_states) else None
                    ),
                    action_at_break=action,
                )
            return None

        while hi - lo > 1:
            mid = (lo + hi) // 2
            if agrees(mid) == lo_agrees:
                lo = mid
            else:
                hi = mid

        # The breakpoint is between lo and hi
        break_idx = lo
        action = sequence[break_idx] if break_idx < n else sequence[-1]
        suffix = sequence[break_idx + 1:] if break_idx + 1 <= n else ()

        # The distinguishing suffix is (a_{break}, suffix)
        dist_suffix = (action,) + suffix if suffix else (action,)

        self._last_query_count = query_count
        self._stats.binary_searches += 1

        return Breakpoint(
            index=break_idx,
            prefix=sequence[:break_idx],
            suffix=dist_suffix,
            hypothesis_state_before=(
                hyp_states[break_idx]
                if break_idx < len(hyp_states)
                else None
            ),
            hypothesis_state_after=(
                hyp_states[break_idx + 1]
                if break_idx + 1 < len(hyp_states)
                else None
            ),
            action_at_break=action,
        )

    # -- counterexample minimisation ----------------------------------------

    def _minimise_counterexample(
        self,
        sequence: Tuple[str, ...],
        hypothesis: HypothesisInterface,
    ) -> Tuple[str, ...]:
        """Attempt to find a shorter counterexample.

        Uses a greedy approach: try removing each action in turn and
        check if the shortened sequence is still a counterexample.
        """
        current = list(sequence)
        improved = True

        while improved and len(current) > 1:
            improved = False
            for i in range(len(current)):
                candidate = tuple(current[:i] + current[i + 1:])
                if self._is_counterexample(candidate, hypothesis):
                    current = list(candidate)
                    improved = True
                    self._stats.minimisation_steps += 1
                    break

        result = tuple(current)
        if len(result) < len(sequence):
            logger.info(
                "Minimised counterexample: %d → %d",
                len(sequence),
                len(result),
            )
        return result

    def _is_counterexample(
        self,
        sequence: Tuple[str, ...],
        hypothesis: HypothesisInterface,
    ) -> bool:
        """Check if *sequence* is a valid counterexample."""
        hyp_state = hypothesis.state_reached(sequence)
        if hyp_state is None:
            concrete = self._mq.query_observation(sequence)
            return concrete is not None and len(concrete) > 0

        hyp_obs = hypothesis.observation_at(hyp_state)
        concrete = self._mq.query_observation(sequence)
        return concrete != hyp_obs

    # -- violation classification -------------------------------------------

    def _classify_violation(
        self,
        bp: Breakpoint,
        hypothesis: HypothesisInterface,
    ) -> str:
        """Classify whether the breakpoint indicates a closedness or
        consistency violation."""
        if bp.hypothesis_state_after is None:
            return "closedness"

        # If two short rows are equivalent in the table but their
        # extensions differ, it's a consistency violation.
        # Otherwise, it's a closedness violation (missing row class).
        prefix = bp.prefix
        if self._table.is_short(prefix):
            return "consistency"
        return "closedness"

    # -- helpers ------------------------------------------------------------

    def _trace_hypothesis(
        self,
        sequence: Tuple[str, ...],
        hypothesis: HypothesisInterface,
    ) -> List[Optional[str]]:
        """Trace the hypothesis states along *sequence*.

        Returns a list of length n+1 where entry i is the hypothesis
        state after executing sequence[:i].
        """
        states: List[Optional[str]] = []
        s: Optional[str] = hypothesis.initial_state()
        states.append(s)
        for act in sequence:
            if s is None:
                states.append(None)
            else:
                s = hypothesis.transition(s, act)
                states.append(s)
        return states

    def _hypothesis_obs_for_suffix(
        self,
        hypothesis: HypothesisInterface,
        state: str,
        suffix: Tuple[str, ...],
    ) -> Optional[Observation]:
        """Get the hypothesis observation after applying *suffix*."""
        s: Optional[str] = state
        for act in suffix:
            if s is None:
                return None
            s = hypothesis.transition(s, act)
        if s is None:
            return None
        return hypothesis.observation_at(s)

    def _add_prefixes(self, sequence: Tuple[str, ...]) -> int:
        """Add all prefixes of the counterexample as rows in the table."""
        added = 0
        for i in range(len(sequence) + 1):
            prefix = sequence[:i]
            if not self._table.has_row(prefix):
                self._table.add_short_row(prefix)
                added += 1
        return added

    def _fallback_suffix_decomposition(
        self, sequence: Tuple[str, ...]
    ) -> int:
        """Fallback: add all non-trivial suffixes as columns."""
        added = 0
        for i in range(1, len(sequence) + 1):
            suffix = sequence[i:]
            if suffix and self._table.add_column(suffix):
                added += 1
        return added

    # -- statistics ---------------------------------------------------------

    @property
    def stats(self) -> "_ProcessorStats":
        return self._stats


@dataclass
class _ProcessorStats:
    total_processed: int = 0
    suffixes_added: int = 0
    binary_searches: int = 0
    minimisation_steps: int = 0

    def summary(self) -> str:
        return (
            f"CEX stats: {self.total_processed} processed, "
            f"{self.suffixes_added} suffixes added, "
            f"{self.binary_searches} binary searches, "
            f"{self.minimisation_steps} minimisation steps"
        )

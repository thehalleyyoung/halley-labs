"""
Main L*-style coalgebraic learning algorithm.

Implements the Angluin L* loop adapted for F-coalgebras:

  1. Initialise the observation table with the initial state.
  2. Fill the table via membership queries.
  3. Check closedness: every long row has an equivalent short row.
  4. Check consistency: equivalent short rows have equivalent extensions.
  5. If not closed/consistent, add rows/columns to fix.
  6. When closed and consistent, construct a hypothesis coalgebra.
  7. Pose an equivalence query.
  8. If a counterexample is found, process it and add to the table.
  9. Repeat until no counterexample is found (or resource limits hit).

The algorithm converges to the unique minimal coalgebra that is
behaviourally equivalent to the concrete system, provided the system
has finitely many behavioural equivalence classes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
)

from .observation_table import ObservationTable, AccessSequence, Suffix, Observation
from .membership_oracle import MembershipOracle
from .equivalence_oracle import (
    Counterexample,
    EquivalenceOracle,
    HypothesisInterface,
)
from .hypothesis import HypothesisBuilder, HypothesisCoalgebra
from .counterexample import CounterexampleProcessor
from .convergence import ConvergenceAnalyzer, RoundSnapshot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Learning configuration
# ---------------------------------------------------------------------------

@dataclass
class LearnerConfig:
    """Configuration parameters for the learning algorithm."""

    max_rounds: int = 200
    conformance_depth: int = 5
    max_conformance_depth: int = 15
    random_walks: int = 300
    max_random_length: int = 25
    timeout_seconds: float = 600.0
    counterexample_strategy: str = "binary"  # or "linear"
    minimise_counterexamples: bool = True
    adaptive_depth: bool = True
    auto_depth: bool = False
    seed: Optional[int] = None
    confidence: float = 0.95
    stale_round_limit: int = 10
    max_total_queries: int = 500_000
    enable_compression: bool = False
    checkpoint_interval: int = 0  # 0 = disabled
    verbose: bool = False

    def __repr__(self) -> str:
        return (
            f"LearnerConfig(max_rounds={self.max_rounds}, "
            f"depth={self.conformance_depth}–{self.max_conformance_depth}, "
            f"timeout={self.timeout_seconds}s)"
        )


# ---------------------------------------------------------------------------
# Learning result
# ---------------------------------------------------------------------------

@dataclass
class LearningResult:
    """Result of a complete learning run."""

    success: bool
    hypothesis: Optional[HypothesisCoalgebra]
    final_table: Optional[ObservationTable]
    rounds: int
    total_membership_queries: int
    total_equivalence_queries: int
    counterexamples: List[Counterexample]
    elapsed_seconds: float
    termination_reason: str
    convergence_data: Optional[Dict[str, List[Any]]] = None
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    conformance_certificate: Optional[Any] = None

    def summary(self) -> str:
        status = "CONVERGED" if self.success else "INCOMPLETE"
        hyp_states = (
            self.hypothesis.state_count if self.hypothesis else 0
        )
        return (
            f"Learning {status}: {self.rounds} rounds, "
            f"{hyp_states} states, "
            f"{self.total_membership_queries} MQ, "
            f"{self.total_equivalence_queries} EQ, "
            f"{len(self.counterexamples)} counterexamples, "
            f"{self.elapsed_seconds:.2f}s — {self.termination_reason}"
        )


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

@dataclass
class LearnerProgress:
    """Progress information sent to callbacks."""

    round_number: int
    phase: str  # "fill", "close", "consistent", "hypothesis", "equivalence", "counterexample"
    hypothesis_states: int
    table_short_rows: int
    table_columns: int
    membership_queries: int
    equivalence_queries: int
    elapsed_seconds: float
    message: str = ""


ProgressCallback = Callable[[LearnerProgress], None]


# ---------------------------------------------------------------------------
# Main learner
# ---------------------------------------------------------------------------

class CoalgebraicLearner:
    """L*-style active learning algorithm for F-coalgebras.

    Parameters
    ----------
    membership_oracle : MembershipOracle
        Oracle for membership (suffix) queries.
    equivalence_oracle : EquivalenceOracle
        Oracle for equivalence (conformance) queries.
    actions : set of str
        The action alphabet.
    config : LearnerConfig, optional
        Learning configuration.
    progress_callback : callable, optional
        Called at each major step with a ``LearnerProgress`` object.
    """

    def __init__(
        self,
        membership_oracle: MembershipOracle,
        equivalence_oracle: EquivalenceOracle,
        actions: Set[str],
        config: Optional[LearnerConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        self._mq = membership_oracle
        self._eq = equivalence_oracle
        self._actions = set(actions)
        self._config = config or LearnerConfig()
        self._progress_cb = progress_callback

        # State
        self._table: Optional[ObservationTable] = None
        self._hypothesis: Optional[HypothesisCoalgebra] = None
        self._cex_processor: Optional[CounterexampleProcessor] = None
        self._convergence: Optional[ConvergenceAnalyzer] = None
        self._round: int = 0
        self._eq_queries: int = 0
        self._counterexamples: List[Counterexample] = []
        self._checkpoints: List[Dict[str, Any]] = []
        self._t0: float = 0.0

    # -- main learning loop -------------------------------------------------

    def learn(self) -> LearningResult:
        """Run the full L* learning loop.

        Returns a ``LearningResult`` describing the outcome.
        """
        self._t0 = time.monotonic()
        logger.info("Starting coalgebraic L* learning, config=%s", self._config)

        # Initialise
        self._initialise()
        termination_reason = "unknown"

        try:
            while self._round < self._config.max_rounds:
                self._round += 1
                logger.info("=== Round %d ===", self._round)

                # Check resource limits
                should_stop, reason = self._check_limits()
                if should_stop:
                    termination_reason = reason
                    break

                # Phase 1: fill the observation table
                self._emit_progress("fill", "Filling observation table")
                self._fill_table()

                # Phase 2: close the table
                self._emit_progress("close", "Checking closedness")
                closed = self._close_table()
                if not closed:
                    continue  # re-fill after structural changes

                # Phase 3: make the table consistent
                self._emit_progress("consistent", "Checking consistency")
                consistent = self._make_consistent()
                if not consistent:
                    continue  # re-fill after structural changes

                # Phase 4: construct hypothesis
                self._emit_progress("hypothesis", "Building hypothesis")
                self._build_hypothesis()

                if self._hypothesis is None:
                    termination_reason = "hypothesis construction failed"
                    break

                # Phase 5: equivalence query
                self._emit_progress("equivalence", "Posing equivalence query")
                counterexample = self._pose_equivalence_query()

                if counterexample is None:
                    # No counterexample — learning complete!
                    termination_reason = "equivalence confirmed"
                    self._record_round_snapshot(None)
                    break

                # Phase 6: process counterexample
                self._emit_progress(
                    "counterexample",
                    f"Processing counterexample (len={counterexample.length})",
                )
                self._process_counterexample(counterexample)
                self._record_round_snapshot(counterexample)

                # Optional compression
                if (
                    self._config.enable_compression
                    and self._round % 10 == 0
                    and self._table is not None
                ):
                    self._table = self._table.compress()

                # Checkpoint
                if (
                    self._config.checkpoint_interval > 0
                    and self._round % self._config.checkpoint_interval == 0
                ):
                    self._save_checkpoint()

            else:
                termination_reason = f"max rounds ({self._config.max_rounds}) reached"

        except KeyboardInterrupt:
            termination_reason = "interrupted"
            logger.warning("Learning interrupted by user")
        except Exception as exc:
            termination_reason = f"error: {exc}"
            logger.exception("Learning failed with exception")

        elapsed = time.monotonic() - self._t0
        success = termination_reason == "equivalence confirmed"

        # Build conformance certificate when auto_depth is enabled
        conf_cert = None
        if self._config.auto_depth and self._hypothesis is not None:
            try:
                from .conformance_gap import ConformanceCompleteCertificate
                from .diameter_computation import ExactDiameterComputer

                computer = ExactDiameterComputer(self._hypothesis)
                d_cert = computer.compute()
                n = d_cert.state_count
                m = 2 * n  # conservative default
                conf_cert = ConformanceCompleteCertificate.build(
                    hypothesis_states=n,
                    concrete_bound=m,
                    diameter=d_cert.diameter,
                    actual_depth=self._config.conformance_depth,
                    n_actions=len(self._actions),
                )
            except Exception as exc:
                logger.debug("Auto-depth certificate skipped: %s", exc)

        result = LearningResult(
            success=success,
            hypothesis=self._hypothesis,
            final_table=self._table,
            rounds=self._round,
            total_membership_queries=self._mq.stats.total_queries,
            total_equivalence_queries=self._eq_queries,
            counterexamples=list(self._counterexamples),
            elapsed_seconds=elapsed,
            termination_reason=termination_reason,
            convergence_data=(
                self._convergence.plot_data()
                if self._convergence
                else None
            ),
            checkpoints=self._checkpoints,
            conformance_certificate=conf_cert,
        )
        logger.info("Learning result: %s", result.summary())
        return result

    # -- initialisation -----------------------------------------------------

    def _initialise(self) -> None:
        """Set up the observation table and supporting infrastructure."""
        self._table = ObservationTable(self._actions)
        self._cex_processor = CounterexampleProcessor(
            self._mq,
            self._table,
            strategy=self._config.counterexample_strategy,
            minimise=self._config.minimise_counterexamples,
        )
        self._convergence = ConvergenceAnalyzer(
            action_count=len(self._actions),
        )
        self._round = 0
        self._eq_queries = 0
        self._counterexamples = []
        self._hypothesis = None

        # Fill initial cell
        initial_obs = self._mq.query_observation((), ())
        if initial_obs is not None:
            self._table.set_cell((), (), initial_obs)

        # Ensure extensions of the initial row exist
        self._table.ensure_extensions()

        logger.info(
            "Initialised table with %d actions, %s",
            len(self._actions),
            self._table,
        )

    def resume(
        self,
        checkpoint: Dict[str, Any],
    ) -> LearningResult:
        """Resume learning from a checkpoint.

        Parameters
        ----------
        checkpoint : dict
            A checkpoint dict as saved by ``_save_checkpoint``.
        """
        self._t0 = time.monotonic()

        # Restore table
        table_data = checkpoint.get("table")
        if table_data is not None:
            self._table = ObservationTable.from_dict(table_data)
        else:
            self._initialise()
            return self.learn()

        self._round = checkpoint.get("round", 0)
        self._eq_queries = checkpoint.get("eq_queries", 0)

        self._cex_processor = CounterexampleProcessor(
            self._mq,
            self._table,
            strategy=self._config.counterexample_strategy,
            minimise=self._config.minimise_counterexamples,
        )
        self._convergence = ConvergenceAnalyzer(
            action_count=len(self._actions),
        )

        logger.info("Resuming learning from round %d", self._round)
        return self.learn()

    # -- table filling ------------------------------------------------------

    def _fill_table(self) -> None:
        """Fill all unfilled cells in the observation table."""
        assert self._table is not None
        self._table.ensure_extensions()
        filled = self._mq.fill_table_cells(self._table)
        if filled > 0:
            logger.debug("Filled %d table cells", filled)

    # -- closedness ---------------------------------------------------------

    def _close_table(self) -> bool:
        """Repair closedness violations.

        Returns True if the table is now closed (possibly after repairs),
        False if repairs were made and the table needs re-filling.
        """
        assert self._table is not None
        repairs = 0
        max_repairs = len(self._actions) * 50  # safety bound

        while repairs < max_repairs:
            unclosed = self._table.find_unclosed_row()
            if unclosed is None:
                if repairs > 0:
                    logger.info("Closed table after %d promotions", repairs)
                    return False  # needs re-filling
                return True

            # Promote the unclosed long row to a short row
            self._table.promote_to_short(unclosed)
            repairs += 1

            # Fill new cells
            self._table.ensure_extensions()
            new_filled = self._mq.fill_table_cells(self._table)
            logger.debug(
                "Promoted %s to short row, filled %d new cells",
                unclosed,
                new_filled,
            )

        logger.warning("Closedness repair exceeded %d iterations", max_repairs)
        return False

    # -- consistency --------------------------------------------------------

    def _make_consistent(self) -> bool:
        """Repair consistency violations.

        Returns True if the table is now consistent (possibly after
        repairs), False if repairs were made and re-filling is needed.
        """
        assert self._table is not None
        repairs = 0
        max_repairs = 100

        while repairs < max_repairs:
            inconsistency = self._table.find_inconsistency()
            if inconsistency is None:
                if repairs > 0:
                    logger.info(
                        "Made table consistent after %d column additions",
                        repairs,
                    )
                    return False  # needs re-filling
                return True

            s1, s2, action, distinguishing_col = inconsistency
            # The new column is (action,) + distinguishing_col
            new_col: Suffix = (action,) + distinguishing_col
            added = self._table.add_column(new_col)
            repairs += 1

            if added:
                logger.debug(
                    "Consistency violation: row(%s) ≡ row(%s) but "
                    "%s-extensions differ on %s → added column %s",
                    s1, s2, action, distinguishing_col, new_col,
                )
                # Fill new column
                self._mq.fill_table_cells(self._table)
            else:
                logger.debug(
                    "Column %s already exists; continuing", new_col
                )

        logger.warning(
            "Consistency repair exceeded %d iterations", max_repairs
        )
        return False

    # -- hypothesis construction --------------------------------------------

    def _build_hypothesis(self) -> None:
        """Build a hypothesis coalgebra from the current table."""
        assert self._table is not None

        if not self._table.is_closed() or not self._table.is_consistent():
            logger.warning(
                "Table is not closed/consistent — cannot build hypothesis"
            )
            self._hypothesis = None
            return

        builder = HypothesisBuilder(self._table)
        try:
            hyp = builder.build()
        except ValueError as exc:
            logger.error("Hypothesis construction failed: %s", exc)
            self._hypothesis = None
            return

        # Validate
        issues = builder.validate(hyp)
        if issues:
            logger.warning("Hypothesis validation issues: %s", issues)

        # Minimise
        hyp = builder.minimize(hyp)

        self._hypothesis = hyp
        self._emit_progress(
            "hypothesis",
            f"Hypothesis built: {hyp.state_count} states",
        )

    # -- equivalence query --------------------------------------------------

    def _pose_equivalence_query(self) -> Optional[Counterexample]:
        """Pose an equivalence query on the current hypothesis."""
        if self._hypothesis is None:
            return None

        self._eq_queries += 1
        return self._eq.check_equivalence(self._hypothesis)

    # -- counterexample processing ------------------------------------------

    def _process_counterexample(self, cex: Counterexample) -> None:
        """Process a counterexample and update the observation table."""
        assert self._table is not None
        assert self._cex_processor is not None
        assert self._hypothesis is not None

        self._counterexamples.append(cex)
        analysis = self._cex_processor.process(cex, self._hypothesis)

        logger.info(
            "Counterexample processed: type=%s, suffix=%s, queries=%d",
            analysis.violation_type,
            analysis.suffix,
            analysis.queries_used,
        )

    # -- convergence tracking -----------------------------------------------

    def _record_round_snapshot(
        self, cex: Optional[Counterexample]
    ) -> None:
        """Record a convergence snapshot for the current round."""
        if self._convergence is None or self._table is None:
            return

        stats = self._table.stats()
        snapshot = RoundSnapshot(
            round_number=self._round,
            short_row_count=stats.short_row_count,
            long_row_count=stats.long_row_count,
            column_count=stats.column_count,
            distinct_classes=stats.distinct_signatures,
            hypothesis_states=(
                self._hypothesis.state_count if self._hypothesis else 0
            ),
            membership_queries=self._mq.stats.total_queries,
            equivalence_queries=self._eq_queries,
            counterexample_length=cex.length if cex else None,
            table_fill_ratio=stats.fill_ratio,
            elapsed_seconds=time.monotonic() - self._t0,
        )
        self._convergence.record_round(snapshot)

    # -- resource limits ----------------------------------------------------

    def _check_limits(self) -> Tuple[bool, str]:
        """Check if any resource limit has been reached."""
        elapsed = time.monotonic() - self._t0
        if elapsed > self._config.timeout_seconds:
            return True, f"timeout ({self._config.timeout_seconds}s)"

        if self._mq.stats.total_queries >= self._config.max_total_queries:
            return True, f"max queries ({self._config.max_total_queries})"

        if self._convergence is not None:
            should_stop, reason = self._convergence.should_terminate_early(
                max_rounds=self._config.max_rounds,
                max_queries=self._config.max_total_queries,
                stale_rounds=self._config.stale_round_limit,
            )
            if should_stop:
                return True, reason

        return False, ""

    # -- checkpointing ------------------------------------------------------

    def _save_checkpoint(self) -> Dict[str, Any]:
        """Save a checkpoint of the current state."""
        assert self._table is not None
        cp = {
            "round": self._round,
            "eq_queries": self._eq_queries,
            "table": self._table.to_dict(),
            "mq_stats": {
                "total": self._mq.stats.total_queries,
                "cache_hits": self._mq.stats.cache_hits,
            },
            "hypothesis_states": (
                self._hypothesis.state_count if self._hypothesis else 0
            ),
            "timestamp": time.time(),
        }
        self._checkpoints.append(cp)
        logger.info("Saved checkpoint at round %d", self._round)
        return cp

    # -- progress callback --------------------------------------------------

    def _emit_progress(self, phase: str, message: str = "") -> None:
        if self._progress_cb is None:
            return
        self._progress_cb(
            LearnerProgress(
                round_number=self._round,
                phase=phase,
                hypothesis_states=(
                    self._hypothesis.state_count if self._hypothesis else 0
                ),
                table_short_rows=(
                    len(self._table.short_rows) if self._table else 0
                ),
                table_columns=(
                    len(self._table.columns) if self._table else 0
                ),
                membership_queries=self._mq.stats.total_queries,
                equivalence_queries=self._eq_queries,
                elapsed_seconds=time.monotonic() - self._t0,
                message=message,
            )
        )

    # -- convenience constructors -------------------------------------------

    @classmethod
    def from_transition_graph(
        cls,
        graph: Any,
        *,
        config: Optional[LearnerConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> "CoalgebraicLearner":
        """Create a learner from a ``TransitionGraph``.

        Sets up the membership oracle, equivalence oracle, and learner
        from the graph.
        """
        cfg = config or LearnerConfig()
        actions = set(graph.actions)

        mq = MembershipOracle.from_transition_graph(graph)
        eq = EquivalenceOracle(
            mq,
            actions,
            initial_depth=cfg.conformance_depth,
            max_depth=cfg.max_conformance_depth,
            random_walks=cfg.random_walks,
            max_random_length=cfg.max_random_length,
            adaptive=cfg.adaptive_depth,
            timeout=cfg.timeout_seconds / 3,
            seed=cfg.seed,
            confidence=cfg.confidence,
        )

        return cls(
            membership_oracle=mq,
            equivalence_oracle=eq,
            actions=actions,
            config=cfg,
            progress_callback=progress_callback,
        )

    # -- accessors ----------------------------------------------------------

    @property
    def table(self) -> Optional[ObservationTable]:
        return self._table

    @property
    def hypothesis(self) -> Optional[HypothesisCoalgebra]:
        return self._hypothesis

    @property
    def convergence(self) -> Optional[ConvergenceAnalyzer]:
        return self._convergence

    @property
    def round_number(self) -> int:
        return self._round

    @property
    def counterexamples(self) -> List[Counterexample]:
        return list(self._counterexamples)

    def learning_summary(self) -> str:
        """Return a human-readable summary of the learning state."""
        parts = [f"Round: {self._round}"]
        if self._table:
            parts.append(f"Table: {self._table}")
        if self._hypothesis:
            parts.append(f"Hypothesis: {self._hypothesis.state_count} states")
        parts.append(f"MQ: {self._mq.stats.total_queries}")
        parts.append(f"EQ: {self._eq_queries}")
        parts.append(f"Counterexamples: {len(self._counterexamples)}")
        if self._convergence:
            parts.append(
                f"Converged: {self._convergence.has_converged()}"
            )
        return " | ".join(parts)

    def __repr__(self) -> str:
        return (
            f"CoalgebraicLearner(round={self._round}, "
            f"actions={len(self._actions)})"
        )

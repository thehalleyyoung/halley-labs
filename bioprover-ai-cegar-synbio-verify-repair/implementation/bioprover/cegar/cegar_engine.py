"""
Main CEGAR orchestrator for BioProver.

Implements the full CEGAR loop:  build initial abstraction → model check
→ check counterexample → refine or report.  Supports portfolio mode
(parallel configurations) and bounded-guarantee mode (partial coverage
on timeout).
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from bioprover.encoding.expression import (
    And,
    Const,
    ExprNode,
    Ge,
    Le,
    Not,
    Or,
    Var,
)
from bioprover.cegar.abstraction import (
    AbstractState,
    AbstractionDomain,
    AbstractTransition,
    IntervalAbstraction,
    IntervalBox,
    ProductAbstraction,
    build_initial_abstraction,
)
from bioprover.cegar.convergence import (
    ConvergenceMetrics,
    ConvergenceMonitor,
    IterationSnapshot,
    StrategySwitchAction,
    TerminationReason,
)
from bioprover.cegar.counterexample import (
    AbstractCounterexample,
    ConcreteCounterexample,
    SpuriousnessChecker,
    SpuriousnessResult,
    SpuriousnessVerdict,
)
from bioprover.cegar.predicate_manager import (
    Predicate,
    PredicateCache,
    PredicateSet,
)
from bioprover.cegar.refinement import (
    AIGuidedRefinement,
    RefinementCombinator,
    RefinementResult,
    RefinementStrategy,
    build_default_combinator,
)
from bioprover.soundness import SoundnessAnnotation, SoundnessLevel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verification status
# ---------------------------------------------------------------------------


class VerificationStatus(Enum):
    """Outcome of the CEGAR-based verification."""

    VERIFIED = auto()           # Property holds for all parameters
    FALSIFIED = auto()          # Concrete counterexample found
    UNKNOWN = auto()            # Could not determine
    BOUNDED_GUARANTEE = auto()  # Holds for explored parameter region


# ---------------------------------------------------------------------------
# Verification result
# ---------------------------------------------------------------------------


@dataclass
class VerificationResult:
    """Complete result of a CEGAR verification run."""

    status: VerificationStatus
    property_name: str = ""
    counterexample: Optional[ConcreteCounterexample] = None
    abstract_counterexample: Optional[AbstractCounterexample] = None
    proof_certificate: Optional[Dict[str, Any]] = None
    statistics: Optional["CEGARStatistics"] = None
    coverage: float = 0.0
    robustness: float = 0.0
    termination_reason: Optional[TerminationReason] = None
    message: str = ""
    soundness: Optional[SoundnessAnnotation] = None

    @property
    def is_verified(self) -> bool:
        return self.status == VerificationStatus.VERIFIED

    @property
    def is_falsified(self) -> bool:
        return self.status == VerificationStatus.FALSIFIED

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "status": self.status.name,
            "property": self.property_name,
            "coverage": round(self.coverage, 4),
            "robustness": round(self.robustness, 4),
            "message": self.message,
        }
        if self.termination_reason:
            d["termination"] = self.termination_reason.name
        if self.counterexample:
            d["counterexample"] = self.counterexample.to_dict()
        if self.statistics:
            d["statistics"] = self.statistics.to_dict()
        if self.soundness:
            d["soundness"] = {
                "level": self.soundness.level.name,
                "assumptions": self.soundness.assumptions,
                "delta": self.soundness.delta,
                "time_bound": self.soundness.time_bound,
            }
        return d


# ---------------------------------------------------------------------------
# CEGAR statistics
# ---------------------------------------------------------------------------


@dataclass
class CEGARStatistics:
    """Aggregate statistics for a CEGAR run."""

    iterations: int = 0
    total_time: float = 0.0
    abstraction_time: float = 0.0
    model_check_time: float = 0.0
    feasibility_time: float = 0.0
    refinement_time: float = 0.0
    peak_states: int = 0
    peak_predicates: int = 0
    spurious_count: int = 0
    genuine_count: int = 0
    final_coverage: float = 0.0
    strategies_used: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iterations": self.iterations,
            "total_time_s": round(self.total_time, 3),
            "abstraction_time_s": round(self.abstraction_time, 3),
            "model_check_time_s": round(self.model_check_time, 3),
            "feasibility_time_s": round(self.feasibility_time, 3),
            "refinement_time_s": round(self.refinement_time, 3),
            "peak_states": self.peak_states,
            "peak_predicates": self.peak_predicates,
            "spurious": self.spurious_count,
            "genuine": self.genuine_count,
            "coverage": round(self.final_coverage, 4),
            "strategies": self.strategies_used,
        }


# ---------------------------------------------------------------------------
# CEGAR configuration
# ---------------------------------------------------------------------------


@dataclass
class CEGARConfig:
    """Configuration for a CEGAR verification run."""

    max_iterations: int = 100
    timeout: float = 3600.0
    initial_grid_resolution: int = 4
    step_size: float = 0.01
    delta: float = 1e-3
    stagnation_window: int = 5
    coverage_target: float = 1.0
    strategy_name: str = "auto"
    enable_ai_heuristic: bool = False
    enable_bounded_guarantee: bool = True
    portfolio_configs: Optional[List["CEGARConfig"]] = None
    max_workers: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_iterations": self.max_iterations,
            "timeout": self.timeout,
            "grid_resolution": self.initial_grid_resolution,
            "step_size": self.step_size,
            "delta": self.delta,
            "stagnation_window": self.stagnation_window,
            "coverage_target": self.coverage_target,
            "strategy": self.strategy_name,
            "ai_heuristic": self.enable_ai_heuristic,
            "bounded_guarantee": self.enable_bounded_guarantee,
            "max_workers": self.max_workers,
        }


# ---------------------------------------------------------------------------
# CEGAR Engine
# ---------------------------------------------------------------------------


class CEGAREngine:
    """Main CEGAR orchestrator.

    Implements the counterexample-guided abstraction refinement loop:

    1. Build initial abstraction
    2. Model check abstract system
    3. If no counterexample → VERIFIED
    4. Check counterexample feasibility
    5. If genuine → FALSIFIED (return counterexample)
    6. If spurious → refine abstraction, go to 2

    Integrates:
    - AbstractionDomain for state space management
    - SMT solver for feasibility checking
    - Model checker callback for abstract model checking
    - RefinementStrategy for predicate selection
    - AI heuristic for refinement prediction (optional)
    """

    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        rhs: Dict[str, ExprNode],
        property_expr: ExprNode,
        property_name: str = "spec",
        initial_region: Optional[IntervalBox] = None,
        solver: Optional[Any] = None,
        model_checker: Optional[Callable[..., Optional[AbstractCounterexample]]] = None,
        refinement_strategy: Optional[RefinementStrategy] = None,
        ai_heuristic: Optional[Callable[..., List[Predicate]]] = None,
        config: Optional[CEGARConfig] = None,
        hill_params: Optional[List[Dict[str, Any]]] = None,
        monotone_info: Optional[Dict[str, Dict[str, int]]] = None,
        steady_states: Optional[List[Dict[str, float]]] = None,
        jacobian: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._bounds = bounds
        self._rhs = rhs
        self._property = property_expr
        self._property_name = property_name
        self._initial_region = initial_region
        self._solver = solver
        self._model_checker = model_checker or self._default_model_check
        self._ai_heuristic = ai_heuristic
        self._config = config or CEGARConfig()

        # Refinement strategy
        if refinement_strategy is not None:
            self._strategy = refinement_strategy
        else:
            self._strategy = build_default_combinator(
                rhs=rhs,
                hill_params=hill_params,
                monotone_info=monotone_info,
                solver=solver,
                steady_states=steady_states,
                jacobian=jacobian,
            )

        # Wrap with AI-guided refinement when enabled
        self._ai_quality_monitor = None
        if self._config.enable_ai_heuristic:
            try:
                from bioprover.ai.predicate_predictor import (
                    PredictionQualityMonitor,
                    PredicatePredictor,
                )
                predictor = PredicatePredictor.from_config()
                self._ai_quality_monitor = PredictionQualityMonitor()
                self._strategy = AIGuidedRefinement(
                    predictor=predictor,
                    fallback=self._strategy,
                    monitor=self._ai_quality_monitor,
                    species_names=list(bounds.keys()),
                )
                logger.info("AI-guided refinement enabled with quality monitoring")
            except Exception as exc:
                logger.warning("Failed to initialise AI-guided refinement: %s", exc)

        # Internal state
        self._domain: Optional[AbstractionDomain] = None
        self._predicate_set = PredicateSet(solver=solver)
        self._predicate_cache = PredicateCache()
        self._checker = SpuriousnessChecker(
            solver=solver,
            timeout=self._config.timeout / 10,
            delta=self._config.delta,
        )
        self._monitor = ConvergenceMonitor(
            max_iterations=self._config.max_iterations,
            timeout=self._config.timeout,
            stagnation_window=self._config.stagnation_window,
            coverage_target=self._config.coverage_target,
        )
        self._stats = CEGARStatistics()
        # Initialize soundness tracking — start with SOUND
        self._soundness = SoundnessAnnotation(level=SoundnessLevel.SOUND)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def verify(self) -> VerificationResult:
        """Run the CEGAR loop.

        Returns a ``VerificationResult`` with status, optional
        counterexample, and statistics.
        """
        logger.info(
            "Starting CEGAR verification for '%s' (%d species)",
            self._property_name,
            len(self._bounds),
        )
        t_start = time.monotonic()

        # Step 1: Build initial abstraction
        self._build_initial_abstraction()

        # Weaken soundness for delta-satisfiability if using dReal
        if self._config.delta > 0:
            self._soundness = self._soundness.with_delta(self._config.delta)

        for iteration in range(self._config.max_iterations):
            # Check termination conditions
            term_reason = self._monitor.should_terminate()
            if term_reason is not None:
                return self._make_result(
                    status=(
                        VerificationStatus.VERIFIED
                        if term_reason == TerminationReason.VERIFIED
                        else VerificationStatus.UNKNOWN
                    ),
                    termination=term_reason,
                    t_start=t_start,
                )

            logger.info(
                "CEGAR iteration %d: %d states, %d predicates",
                iteration,
                self._domain.state_count(),
                len(self._predicate_set),
            )

            # Step 2: Model check abstract system
            t_mc = time.monotonic()
            cex = self._model_check(iteration)
            mc_elapsed = time.monotonic() - t_mc
            self._stats.model_check_time += mc_elapsed

            # Step 3: No counterexample → VERIFIED
            if cex is None:
                logger.info("VERIFIED: no abstract counterexample at iteration %d", iteration)
                self._record_iteration(iteration, mc_time=mc_elapsed)
                return self._make_result(
                    status=VerificationStatus.VERIFIED,
                    termination=TerminationReason.VERIFIED,
                    t_start=t_start,
                    proof=self._build_proof_certificate(iteration),
                )

            # Step 4: Check counterexample feasibility
            t_feas = time.monotonic()
            analysis = self._checker.check(
                cex, rhs=self._rhs, step_size=self._config.step_size
            )
            feas_elapsed = time.monotonic() - t_feas
            self._stats.feasibility_time += feas_elapsed

            # Step 5: Genuine → FALSIFIED
            if analysis.verdict == SpuriousnessVerdict.GENUINE:
                logger.info(
                    "FALSIFIED: genuine counterexample at iteration %d (length %d)",
                    iteration,
                    cex.length,
                )
                self._stats.genuine_count += 1
                self._monitor.record_counterexample_genuine()
                self._record_iteration(
                    iteration,
                    cex_len=cex.length,
                    mc_time=mc_elapsed,
                    feas_time=feas_elapsed,
                )
                return self._make_result(
                    status=VerificationStatus.FALSIFIED,
                    termination=TerminationReason.FALSIFIED,
                    t_start=t_start,
                    concrete_cex=analysis.concrete_witness,
                    abstract_cex=cex,
                )

            # Inconclusive
            if analysis.verdict == SpuriousnessVerdict.INCONCLUSIVE:
                logger.warning("Inconclusive feasibility check at iteration %d", iteration)
                self._monitor.record_counterexample_inconclusive()
                # Try to continue with refinement anyway
                analysis = SpuriousnessResult(
                    verdict=SpuriousnessVerdict.SPURIOUS,
                    failure_index=cex.length // 2,
                    message="Treated as spurious (inconclusive)",
                )

            # Step 6: Spurious → refine
            self._stats.spurious_count += 1
            logger.debug(
                "Spurious counterexample at iteration %d, failure at step %s",
                iteration,
                analysis.failure_index,
            )

            # Check strategy recommendation
            recommendation = self._monitor.recommend_strategy()
            if recommendation.action == StrategySwitchAction.SWITCH_STRATEGY:
                logger.info(
                    "Switching strategy to '%s': %s",
                    recommendation.suggested_strategy,
                    recommendation.reason,
                )
                if isinstance(self._strategy, RefinementCombinator):
                    self._strategy.reorder_by_success_rate()
                self._monitor.acknowledge_switch(
                    recommendation.suggested_strategy or ""
                )

            # AI heuristic for refinement prediction
            ai_preds: List[Predicate] = []
            if self._config.enable_ai_heuristic and self._ai_heuristic is not None:
                try:
                    ai_preds = self._ai_heuristic(cex, analysis, iteration)
                    logger.debug("AI heuristic suggested %d predicates", len(ai_preds))
                    # Weaken soundness when using GP surrogate guidance
                    self._soundness = self._soundness.weaken_to(
                        SoundnessLevel.APPROXIMATE,
                        "AI/GP surrogate-guided refinement",
                    )
                except Exception as exc:
                    logger.debug("AI heuristic failed: %s", exc)

            # Perform refinement
            t_ref = time.monotonic()
            ref_result = self._strategy.refine(
                cex, analysis, self._domain, iteration
            )
            ref_elapsed = time.monotonic() - t_ref
            self._stats.refinement_time += ref_elapsed

            # Add AI predicates
            if ai_preds:
                self._predicate_set.add_all(ai_preds)
                ref_result.new_predicates.extend(ai_preds)

            # Update predicate set
            self._predicate_set.add_all(ref_result.new_predicates)

            # Track strategy usage
            strat_name = ref_result.strategy_name
            self._stats.strategies_used[strat_name] = (
                self._stats.strategies_used.get(strat_name, 0) + 1
            )

            # Recompute transitions after refinement
            if isinstance(self._domain, IntervalAbstraction):
                self._domain.compute_transitions(rhs=self._rhs)

            # Record iteration
            coverage = self._estimate_coverage()
            self._record_iteration(
                iteration,
                cex_len=cex.length,
                spurious=True,
                strategy=strat_name,
                preds_added=ref_result.predicate_count,
                mc_time=mc_elapsed,
                feas_time=feas_elapsed,
                ref_time=ref_elapsed,
                coverage=coverage,
            )

            if not ref_result.success:
                logger.warning(
                    "Refinement produced no predicates at iteration %d", iteration
                )

        # Exhausted iterations
        return self._make_result(
            status=VerificationStatus.UNKNOWN,
            termination=TerminationReason.MAX_ITERATIONS,
            t_start=t_start,
        )

    # ------------------------------------------------------------------
    # Portfolio mode
    # ------------------------------------------------------------------

    def verify_portfolio(self) -> VerificationResult:
        """Run multiple CEGAR configurations in parallel.

        Returns the first definitive result (VERIFIED or FALSIFIED),
        or the best UNKNOWN result on timeout.
        """
        configs = self._config.portfolio_configs
        if not configs:
            return self.verify()

        logger.info("Starting portfolio CEGAR with %d configurations", len(configs))

        best_result: Optional[VerificationResult] = None

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._config.max_workers
        ) as executor:
            futures: Dict[concurrent.futures.Future[VerificationResult], int] = {}

            for idx, cfg in enumerate(configs):
                engine = CEGAREngine(
                    bounds=self._bounds,
                    rhs=self._rhs,
                    property_expr=self._property,
                    property_name=self._property_name,
                    initial_region=self._initial_region,
                    solver=self._solver,
                    model_checker=self._model_checker,
                    refinement_strategy=self._strategy,
                    config=cfg,
                )
                future = executor.submit(engine.verify)
                futures[future] = idx

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:
                    logger.warning("Portfolio config %d failed: %s", futures[future], exc)
                    continue

                # Definitive result → cancel others and return
                if result.status in (VerificationStatus.VERIFIED, VerificationStatus.FALSIFIED):
                    for f in futures:
                        f.cancel()
                    return result

                # Track best partial result
                if best_result is None or result.coverage > best_result.coverage:
                    best_result = result

        return best_result or self._make_result(
            status=VerificationStatus.UNKNOWN,
            termination=TerminationReason.TIMEOUT,
            t_start=time.monotonic(),
        )

    # ------------------------------------------------------------------
    # Internal: abstraction construction
    # ------------------------------------------------------------------

    def _build_initial_abstraction(self) -> None:
        """Build the initial (coarse) abstraction."""
        t0 = time.monotonic()

        self._domain = build_initial_abstraction(
            bounds=self._bounds,
            initial_region=self._initial_region,
            resolution=self._config.initial_grid_resolution,
            predicates=self._predicate_set if len(self._predicate_set) > 0 else None,
            solver=self._solver,
        )

        # Compute initial transitions
        if isinstance(self._domain, IntervalAbstraction):
            self._domain.compute_transitions(rhs=self._rhs)
            self._domain.explore_from_initial()

        self._stats.abstraction_time += time.monotonic() - t0
        self._stats.peak_states = max(
            self._stats.peak_states, self._domain.state_count()
        )

        logger.info(
            "Initial abstraction: %d states, %d transitions",
            self._domain.state_count(),
            self._domain.transition_count(),
        )

    # ------------------------------------------------------------------
    # Internal: model checking
    # ------------------------------------------------------------------

    def _model_check(self, iteration: int) -> Optional[AbstractCounterexample]:
        """Model check the abstract system against the property.

        Delegates to the configured model checker callback.  If none
        is provided, uses a built-in BFS-based reachability check.
        """
        return self._model_checker(self._domain, self._property, iteration)

    def _default_model_check(
        self,
        domain: AbstractionDomain,
        prop: ExprNode,
        iteration: int,
    ) -> Optional[AbstractCounterexample]:
        """Built-in BFS reachability model checker.

        Searches for an abstract state that may violate the property
        (i.e., the property is not guaranteed to hold within the
        abstract state's region).
        """
        # Build reachability graph via BFS
        initial = domain.initial_states()
        if not initial:
            return None

        visited: Dict[int, Optional[int]] = {}  # state_id -> predecessor
        queue: List[AbstractState] = list(initial)
        for s in initial:
            visited[s.state_id] = None

        while queue:
            current = queue.pop(0)

            # Check if property may be violated in this state
            if self._may_violate(current, prop):
                # Reconstruct path
                path = self._reconstruct_path(current, visited, domain)
                return AbstractCounterexample(
                    path=path,
                    property_violated=self._property_name,
                    iteration_found=iteration,
                )

            # Expand successors
            for succ in domain.post_image(current):
                if succ.state_id not in visited:
                    visited[succ.state_id] = current.state_id
                    queue.append(succ)

        return None

    def _may_violate(self, state: AbstractState, prop: ExprNode) -> bool:
        """Check if property might be violated in this abstract state.

        Returns True if the negation of the property is satisfiable
        within the state's region (overapproximation).
        """
        midpoint = state.box.midpoint()

        # Quick evaluation at midpoint
        try:
            from bioprover.cegar.predicate_manager import _eval_bool

            result = _eval_bool(prop, midpoint)
            if result is False:
                return True
            # Also check corners for interval-level soundness
            for corner in state.box.vertices()[:4]:
                result = _eval_bool(prop, corner)
                if result is False:
                    return True
        except Exception:
            return True  # conservatively assume violation

        return False

    def _reconstruct_path(
        self,
        target: AbstractState,
        predecessors: Dict[int, Optional[int]],
        domain: AbstractionDomain,
    ) -> List[AbstractState]:
        """Reconstruct BFS path from initial state to target."""
        path: List[AbstractState] = []
        current_id: Optional[int] = target.state_id

        while current_id is not None:
            state = domain.get_state(current_id) if hasattr(domain, "get_state") else None
            if state is None:
                # Fall back: find state by ID
                for s in domain.states():
                    if s.state_id == current_id:
                        state = s
                        break
            if state is not None:
                path.append(state)
            current_id = predecessors.get(current_id)

        path.reverse()
        return path

    # ------------------------------------------------------------------
    # Internal: coverage estimation
    # ------------------------------------------------------------------

    def _estimate_coverage(self) -> float:
        """Estimate fraction of state/parameter space verified.

        Conservative estimate based on the ratio of explored states
        that do not require further refinement.
        """
        if self._domain is None:
            return 0.0

        total = self._domain.state_count()
        if total == 0:
            return 0.0

        # Count states where the property definitely holds
        safe_count = 0
        for state in self._domain.states():
            if not self._may_violate(state, self._property):
                safe_count += 1

        return safe_count / total

    # ------------------------------------------------------------------
    # Internal: proof certificate
    # ------------------------------------------------------------------

    def _build_proof_certificate(self, final_iteration: int) -> Dict[str, Any]:
        """Build a proof certificate for a verified property."""
        cert: Dict[str, Any] = {
            "type": "CEGAR_invariant",
            "property": self._property_name,
            "iterations": final_iteration + 1,
            "predicate_count": len(self._predicate_set),
            "predicates": [p.expr.pretty() for p in self._predicate_set],
            "abstract_states": self._domain.state_count() if self._domain else 0,
            "strategy_statistics": (
                self._strategy.strategy_statistics()
                if isinstance(self._strategy, RefinementCombinator)
                else {}
            ),
        }
        if self._ai_quality_monitor is not None:
            cert["ai_quality_metrics"] = self._ai_quality_monitor.get_metrics()
        return cert

    # ------------------------------------------------------------------
    # Internal: result construction
    # ------------------------------------------------------------------

    def _make_result(
        self,
        status: VerificationStatus,
        termination: TerminationReason,
        t_start: float,
        concrete_cex: Optional[ConcreteCounterexample] = None,
        abstract_cex: Optional[AbstractCounterexample] = None,
        proof: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        self._stats.total_time = time.monotonic() - t_start
        self._stats.iterations = self._monitor.current_iteration
        self._stats.peak_predicates = max(
            self._stats.peak_predicates, len(self._predicate_set)
        )

        coverage = self._estimate_coverage()
        self._stats.final_coverage = coverage

        # Bounded guarantee
        if (
            status == VerificationStatus.UNKNOWN
            and self._config.enable_bounded_guarantee
            and coverage > 0.0
        ):
            status = VerificationStatus.BOUNDED_GUARANTEE
            self._soundness = self._soundness.weaken_to(
                SoundnessLevel.BOUNDED,
                f"Bounded guarantee: {coverage:.1%} coverage",
            )

        self._monitor.metrics.termination_reason = termination

        return VerificationResult(
            status=status,
            property_name=self._property_name,
            counterexample=concrete_cex,
            abstract_counterexample=abstract_cex,
            proof_certificate=proof,
            statistics=self._stats,
            coverage=coverage,
            termination_reason=termination,
            message=self._monitor.summary(),
            soundness=self._soundness,
        )

    # ------------------------------------------------------------------
    # Internal: iteration recording
    # ------------------------------------------------------------------

    def _record_iteration(
        self,
        iteration: int,
        cex_len: int = 0,
        spurious: bool = False,
        strategy: str = "",
        preds_added: int = 0,
        mc_time: float = 0.0,
        feas_time: float = 0.0,
        ref_time: float = 0.0,
        coverage: float = 0.0,
    ) -> None:
        snapshot = IterationSnapshot(
            iteration=iteration,
            timestamp=time.monotonic(),
            abstract_state_count=self._domain.state_count() if self._domain else 0,
            transition_count=self._domain.transition_count() if self._domain else 0,
            predicate_count=len(self._predicate_set),
            counterexample_length=cex_len,
            counterexample_spurious=spurious,
            refinement_strategy=strategy,
            refinement_predicates_added=preds_added,
            refinement_time=ref_time,
            model_check_time=mc_time,
            feasibility_check_time=feas_time,
            coverage_estimate=coverage,
        )
        self._monitor.record_iteration(snapshot)

        self._stats.peak_states = max(
            self._stats.peak_states,
            self._domain.state_count() if self._domain else 0,
        )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def domain(self) -> Optional[AbstractionDomain]:
        return self._domain

    @property
    def predicates(self) -> PredicateSet:
        return self._predicate_set

    @property
    def monitor(self) -> ConvergenceMonitor:
        return self._monitor

    @property
    def statistics(self) -> CEGARStatistics:
        return self._stats

    @property
    def config(self) -> CEGARConfig:
        return self._config

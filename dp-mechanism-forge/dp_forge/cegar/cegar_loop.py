"""
Main CEGAR loop for differential privacy mechanism verification.

Implements the CEGAREngine orchestrating the abstract → verify → refine
cycle, termination checking, and comprehensive result reporting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from dp_forge.types import (
    AbstractDomainType,
    AbstractValue,
    AdjacencyRelation,
    Formula,
    Predicate,
    PrivacyBudget,
    VerifyResult,
)
from dp_forge.cegar import (
    AbstractCounterexample,
    AbstractState,
    AbstractionType,
    CEGARConfig,
    CEGARResult,
    CEGARStatus,
    RefinementResult,
    RefinementStrategy,
)
from dp_forge.cegar.abstraction import (
    AbstractDomain,
    GaloisConnection,
    IntervalAbstraction,
    PolyhedralAbstraction,
    ZonotopeAbstraction,
    PrivacyLossAbstraction,
)
from dp_forge.cegar.predicate_abstraction import (
    CartesianAbstraction,
    BooleanAbstraction,
    PredicateAbstractionManager,
    PredicateDiscovery,
    PredicateEvaluator,
)
from dp_forge.cegar.refinement import (
    CounterexampleAnalysis,
    CraigInterpolationRefiner,
    ConvergenceAccelerator,
    LazyAbstractionTree,
    RefinementEngine,
    RefinementStrategySelector,
)
from dp_forge.cegar.privacy_verifier import (
    AbstractPrivacyLoss,
    CompositionChecker,
    PrivacyProperty,
    PrivacyPropertyChecker,
    SequentialCompositionAnalyzer,
)


# ---------------------------------------------------------------------------
# Termination checker
# ---------------------------------------------------------------------------


class TerminationChecker:
    """Checks termination conditions for the CEGAR loop.

    Monitors iteration count, time budget, predicate count, and
    convergence to determine when to stop.
    """

    def __init__(self, config: CEGARConfig) -> None:
        """Initialize the termination checker.

        Args:
            config: CEGAR configuration with limits.
        """
        self._config = config
        self._start_time = time.time()
        self._iteration = 0
        self._predicate_count = 0
        self._prev_predicate_count = 0
        self._stagnation_count = 0

    def reset(self) -> None:
        """Reset the termination checker for a new run."""
        self._start_time = time.time()
        self._iteration = 0
        self._predicate_count = 0
        self._prev_predicate_count = 0
        self._stagnation_count = 0

    def update(self, iteration: int, predicate_count: int) -> None:
        """Update the checker with current loop state.

        Args:
            iteration: Current iteration number.
            predicate_count: Current number of predicates.
        """
        self._iteration = iteration
        if predicate_count == self._predicate_count:
            self._stagnation_count += 1
        else:
            self._stagnation_count = 0
        self._prev_predicate_count = self._predicate_count
        self._predicate_count = predicate_count

    @property
    def elapsed_time(self) -> float:
        """Elapsed time since start in seconds."""
        return time.time() - self._start_time

    def should_terminate(self) -> Tuple[bool, CEGARStatus]:
        """Check if the CEGAR loop should terminate.

        Returns:
            (should_stop, reason) tuple.
        """
        if self._iteration >= self._config.max_refinements:
            return True, CEGARStatus.REFINEMENT_LIMIT

        if self.elapsed_time >= self._config.timeout_seconds:
            return True, CEGARStatus.TIMEOUT

        if self._predicate_count >= self._config.predicate_limit:
            return True, CEGARStatus.REFINEMENT_LIMIT

        # Stagnation detection
        if self._stagnation_count >= 10:
            return True, CEGARStatus.UNKNOWN

        return False, CEGARStatus.UNKNOWN

    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of the current termination state.

        Returns:
            Dictionary with status information.
        """
        return {
            "iteration": self._iteration,
            "elapsed_time": self.elapsed_time,
            "predicate_count": self._predicate_count,
            "stagnation_count": self._stagnation_count,
            "time_remaining": max(0, self._config.timeout_seconds - self.elapsed_time),
        }


# ---------------------------------------------------------------------------
# Abstraction-refinement loop
# ---------------------------------------------------------------------------


class AbstractionRefinementLoop:
    """Abstract → verify → refine cycle.

    Encapsulates a single iteration of the CEGAR loop:
    1. Build/update abstract model
    2. Verify property on abstract model
    3. If violation found, check feasibility
    4. If spurious, refine abstraction
    """

    def __init__(
        self,
        config: CEGARConfig,
        domain: AbstractDomain,
        verifier: PrivacyPropertyChecker,
        refinement_engine: RefinementEngine,
    ) -> None:
        """Initialize the abstraction-refinement loop.

        Args:
            config: CEGAR configuration.
            domain: Abstract domain.
            verifier: Privacy property checker.
            refinement_engine: Refinement engine.
        """
        self._config = config
        self._domain = domain
        self._verifier = verifier
        self._engine = refinement_engine
        self._predicate_manager: Optional[PredicateAbstractionManager] = None

    def initialize(
        self,
        mechanism: npt.NDArray[np.float64],
        adjacency: AdjacencyRelation,
    ) -> None:
        """Initialize the loop with a mechanism.

        Args:
            mechanism: Mechanism probability matrix.
            adjacency: Adjacency relation.
        """
        self._predicate_manager = PredicateAbstractionManager(
            abstraction_type=(
                AbstractionType.CARTESIAN
                if self._config.abstraction_type == AbstractionType.CARTESIAN
                else AbstractionType.PREDICATE
            ),
        )
        self._predicate_manager.initialize(mechanism, adjacency)

        # Initialize refinement tree if using lazy abstraction
        if self._config.lazy_refinement:
            initial_value = self._domain.alpha(mechanism)
            self._engine.initialize_tree(initial_value)

    def step(
        self,
        mechanism: npt.NDArray[np.float64],
        budget: PrivacyBudget,
        adjacency: AdjacencyRelation,
        iteration: int,
    ) -> Tuple[Optional[bool], Optional[AbstractCounterexample], Optional[RefinementResult]]:
        """Execute one iteration of the abstraction-refinement loop.

        Args:
            mechanism: Mechanism probability matrix.
            budget: Privacy budget.
            adjacency: Adjacency relation.
            iteration: Current iteration number.

        Returns:
            (verified_or_none, counterexample_or_none, refinement_or_none)
            - verified=True: property verified
            - verified=False: genuine counterexample found
            - verified=None: refinement performed, loop should continue
        """
        # Step 1: Abstract verification
        is_verified, cex = self._verifier.verify_abstract(
            mechanism, budget, adjacency,
        )

        if is_verified:
            return True, None, None

        if cex is None:
            return True, None, None

        # Step 2: Check counterexample feasibility and refine
        is_genuine, refinement = self._engine.analyze_and_refine(
            cex, mechanism, budget,
        )

        if is_genuine:
            return False, cex, None

        # Step 3: Refinement was performed
        if refinement is not None and self._predicate_manager is not None:
            for pred in refinement.new_predicates:
                self._predicate_manager.add_predicate(pred)

        return None, cex, refinement

    @property
    def predicate_count(self) -> int:
        """Current number of predicates."""
        if self._predicate_manager is not None:
            return len(self._predicate_manager.predicates)
        return 0

    @property
    def predicates(self) -> List[Predicate]:
        """Current predicates."""
        if self._predicate_manager is not None:
            return self._predicate_manager.predicates
        return []


# ---------------------------------------------------------------------------
# CEGAR Engine
# ---------------------------------------------------------------------------


class CEGAREngine:
    """Main CEGAR loop orchestrator.

    Coordinates the full CEGAR verification process: initialization,
    abstract verification, counterexample analysis, refinement, and
    termination checking.
    """

    def __init__(self, config: Optional[CEGARConfig] = None) -> None:
        """Initialize the CEGAR engine.

        Args:
            config: CEGAR configuration. Uses defaults if not provided.
        """
        self._config = config or CEGARConfig()
        self._domain = self._create_domain()
        self._verifier = PrivacyPropertyChecker(domain=self._domain)
        self._refinement_engine = RefinementEngine(
            config=self._config, domain=self._domain,
        )
        self._loop = AbstractionRefinementLoop(
            config=self._config,
            domain=self._domain,
            verifier=self._verifier,
            refinement_engine=self._refinement_engine,
        )
        self._termination = TerminationChecker(self._config)

    def _create_domain(self) -> AbstractDomain:
        """Create the abstract domain based on configuration."""
        if self._config.abstraction_type == AbstractionType.INTERVAL:
            return IntervalAbstraction()
        elif self._config.abstraction_type == AbstractionType.OCTAGON:
            return PolyhedralAbstraction()
        else:
            return IntervalAbstraction()

    def verify(
        self,
        mechanism: npt.NDArray[np.float64],
        budget: PrivacyBudget,
        adjacency: Optional[AdjacencyRelation] = None,
    ) -> CEGARResult:
        """Run the CEGAR loop to verify DP guarantees.

        Args:
            mechanism: The n × k probability table.
            budget: Target privacy budget.
            adjacency: Adjacency relation (defaults to Hamming-1).

        Returns:
            CEGARResult with verification outcome.
        """
        mechanism = np.asarray(mechanism, dtype=np.float64)
        if mechanism.ndim == 1:
            mechanism = mechanism.reshape(1, -1)

        n, k = mechanism.shape
        if adjacency is None:
            adjacency = AdjacencyRelation.hamming_distance_1(n)

        self._termination.reset()
        self._loop.initialize(mechanism, adjacency)

        # First: try direct verification (no abstraction overhead)
        is_verified, cex = self._verifier.verify_abstract(
            mechanism, budget, adjacency,
        )

        if is_verified:
            return CEGARResult(
                status=CEGARStatus.VERIFIED,
                verified=True,
                refinement_count=0,
                predicate_count=self._loop.predicate_count,
                proof_predicates=self._loop.predicates,
                total_time=self._termination.elapsed_time,
            )

        if cex is not None:
            # Check if this is a genuine violation
            is_genuine, _ = CounterexampleAnalysis().classify_counterexample(
                cex, mechanism, budget,
            )
            if is_genuine:
                return CEGARResult(
                    status=CEGARStatus.COUNTEREXAMPLE_FOUND,
                    verified=False,
                    refinement_count=0,
                    predicate_count=0,
                    counterexample=cex,
                    total_time=self._termination.elapsed_time,
                )

        # Enter CEGAR loop
        iteration = 0
        last_cex = cex

        while True:
            iteration += 1
            self._termination.update(iteration, self._loop.predicate_count)

            should_stop, stop_reason = self._termination.should_terminate()
            if should_stop:
                return CEGARResult(
                    status=stop_reason,
                    verified=False,
                    refinement_count=iteration,
                    predicate_count=self._loop.predicate_count,
                    counterexample=last_cex,
                    total_time=self._termination.elapsed_time,
                )

            verified, cex, refinement = self._loop.step(
                mechanism, budget, adjacency, iteration,
            )

            if cex is not None:
                last_cex = cex

            if verified is True:
                return CEGARResult(
                    status=CEGARStatus.VERIFIED,
                    verified=True,
                    refinement_count=iteration,
                    predicate_count=self._loop.predicate_count,
                    proof_predicates=self._loop.predicates,
                    total_time=self._termination.elapsed_time,
                )

            if verified is False:
                return CEGARResult(
                    status=CEGARStatus.COUNTEREXAMPLE_FOUND,
                    verified=False,
                    refinement_count=iteration,
                    predicate_count=self._loop.predicate_count,
                    counterexample=cex,
                    total_time=self._termination.elapsed_time,
                )

            # verified is None — refinement happened, continue
            if self._config.verbose >= 2:
                status = self._termination.get_status_summary()
                print(
                    f"  CEGAR iter {iteration}: "
                    f"predicates={status['predicate_count']}, "
                    f"time={status['elapsed_time']:.1f}s"
                )

    def verify_with_synthesis(
        self,
        mechanism: npt.NDArray[np.float64],
        budget: PrivacyBudget,
        adjacency: Optional[AdjacencyRelation] = None,
    ) -> Tuple[CEGARResult, Optional[npt.NDArray[np.float64]]]:
        """Verify and optionally synthesize a repaired mechanism.

        If verification fails, attempts to repair the mechanism using
        the predicates and counterexample information discovered during
        refinement.

        Args:
            mechanism: The n × k probability table.
            budget: Target privacy budget.
            adjacency: Adjacency relation.

        Returns:
            (CEGARResult, repaired_mechanism_or_none).
        """
        result = self.verify(mechanism, budget, adjacency)

        if result.verified:
            return result, None

        # Attempt simple repair: clip probability ratios
        repaired = self._attempt_repair(mechanism, budget, adjacency)
        return result, repaired

    def _attempt_repair(
        self,
        mechanism: npt.NDArray[np.float64],
        budget: PrivacyBudget,
        adjacency: Optional[AdjacencyRelation],
    ) -> Optional[npt.NDArray[np.float64]]:
        """Attempt to repair a mechanism to satisfy DP.

        Uses a simple projection approach: clip probability ratios
        to satisfy the epsilon bound.

        Args:
            mechanism: Original mechanism matrix.
            budget: Target privacy budget.
            adjacency: Adjacency relation.

        Returns:
            Repaired mechanism or None if repair failed.
        """
        n, k = mechanism.shape
        if adjacency is None:
            adjacency = AdjacencyRelation.hamming_distance_1(n)

        repaired = mechanism.copy()
        exp_eps = np.exp(budget.epsilon)

        all_edges = list(adjacency.edges)
        if adjacency.symmetric:
            all_edges += [(j, i) for (i, j) in adjacency.edges]

        # Iterative projection
        for _ in range(100):
            changed = False
            for (i, ip) in all_edges:
                for j in range(k):
                    p_i = repaired[i, j]
                    p_ip = repaired[ip, j]
                    if p_ip > 1e-300:
                        if p_i > exp_eps * p_ip:
                            # Project: set p_i to geometric mean that satisfies bound
                            target = np.sqrt(p_i * exp_eps * p_ip)
                            repaired[i, j] = min(target, exp_eps * p_ip)
                            changed = True
                    if p_i > 1e-300:
                        if p_ip > exp_eps * p_i:
                            target = np.sqrt(p_ip * exp_eps * p_i)
                            repaired[ip, j] = min(target, exp_eps * p_i)
                            changed = True

            # Re-normalize rows
            for i in range(n):
                row_sum = np.sum(repaired[i])
                if row_sum > 1e-300:
                    repaired[i] /= row_sum

            if not changed:
                break

        # Verify the repair
        checker = PrivacyPropertyChecker()
        is_ok, _ = checker.verify_abstract(repaired, budget, adjacency)
        if is_ok:
            return repaired
        return None


# ---------------------------------------------------------------------------
# Public API implementations
# ---------------------------------------------------------------------------


def cegar_verify_impl(
    mechanism: npt.NDArray[np.float64],
    epsilon: float,
    delta: float = 0.0,
    *,
    config: Optional[CEGARConfig] = None,
) -> CEGARResult:
    """Implementation of the cegar_verify convenience function.

    Args:
        mechanism: The n × k probability table.
        epsilon: Privacy parameter ε.
        delta: Privacy parameter δ.
        config: Optional CEGAR configuration.

    Returns:
        CEGARResult with verification outcome.
    """
    budget = PrivacyBudget(epsilon=epsilon, delta=delta)
    engine = CEGAREngine(config=config)
    return engine.verify(mechanism, budget)


def discover_predicates_impl(
    mechanism: npt.NDArray[np.float64],
    budget: PrivacyBudget,
    adjacency: AdjacencyRelation,
    *,
    max_predicates: int = 50,
) -> List[Predicate]:
    """Implementation of the discover_predicates function.

    Uses heuristic and interpolation-based strategies to find predicates
    that are likely to lead to a proof or genuine counterexample.

    Args:
        mechanism: The n × k probability table.
        budget: Target privacy budget.
        adjacency: Adjacency relation.
        max_predicates: Maximum number of predicates to discover.

    Returns:
        List of discovered predicates.
    """
    discovery = PredicateDiscovery()
    predicates = discovery.discover_from_mechanism(
        mechanism, adjacency, budget, max_predicates=max_predicates,
    )
    return predicates

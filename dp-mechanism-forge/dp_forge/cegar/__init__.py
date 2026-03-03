"""
CEGAR-based verification for differential privacy mechanisms.

This package implements Counterexample-Guided Abstraction Refinement (CEGAR)
for verifying and synthesising DP mechanisms. The CEGAR loop alternates
between abstract verification (which may report spurious counterexamples)
and refinement (which eliminates spurious counterexamples by adding
predicates or splitting abstract states).

Architecture:
    1. **AbstractionManager** — Maintains the current predicate abstraction
       and maps concrete mechanism states to abstract states.
    2. **AbstractVerifier** — Verifies DP properties on the abstract model;
       if a violation is found, returns an abstract counterexample trace.
    3. **RefinementEngine** — Checks whether an abstract counterexample is
       feasible in the concrete model; if spurious, computes new predicates
       via Craig interpolation to refine the abstraction.
    4. **CEGARLoop** — Orchestrates the abstraction–verification–refinement
       cycle until a proof or genuine counterexample is found.
    5. **PredicateDiscovery** — Heuristic and interpolation-based strategies
       for discovering useful predicates.

Example::

    from dp_forge.cegar import CEGARLoop, CEGARConfig

    config = CEGARConfig(max_refinements=100, abstraction_type="predicate")
    loop = CEGARLoop(config=config)
    result = loop.verify(mechanism, privacy_budget)
    if result.verified:
        print(f"Verified in {result.refinement_count} refinements")
    else:
        print(f"Counterexample: {result.counterexample}")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt

from dp_forge.types import (
    AbstractDomainType,
    AbstractValue,
    AdjacencyRelation,
    Formula,
    InterpolantType,
    Predicate,
    PrivacyBudget,
    VerifyResult,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AbstractionType(Enum):
    """Type of abstraction used in the CEGAR loop."""

    PREDICATE = auto()
    INTERVAL = auto()
    OCTAGON = auto()
    CARTESIAN = auto()

    def __repr__(self) -> str:
        return f"AbstractionType.{self.name}"


class RefinementStrategy(Enum):
    """Strategy for refining the abstraction after a spurious counterexample."""

    INTERPOLATION = auto()
    WEAKEST_PRECONDITION = auto()
    STRONGEST_POSTCONDITION = auto()
    IMPACT = auto()
    LAZY = auto()

    def __repr__(self) -> str:
        return f"RefinementStrategy.{self.name}"


class CEGARStatus(Enum):
    """Status of the CEGAR verification loop."""

    VERIFIED = auto()
    COUNTEREXAMPLE_FOUND = auto()
    REFINEMENT_LIMIT = auto()
    TIMEOUT = auto()
    UNKNOWN = auto()

    def __repr__(self) -> str:
        return f"CEGARStatus.{self.name}"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CEGARConfig:
    """Configuration for the CEGAR verification loop.

    Attributes:
        max_refinements: Maximum number of abstraction refinements.
        abstraction_type: Initial abstraction type.
        refinement_strategy: Strategy for computing refinements.
        interpolant_type: Type of interpolants used for predicate discovery.
        timeout_seconds: Maximum wall-clock time in seconds.
        predicate_limit: Maximum number of predicates in the abstraction.
        lazy_refinement: Whether to use lazy abstraction (on-the-fly).
        verbose: Verbosity level (0=silent, 1=progress, 2=debug).
    """

    max_refinements: int = 100
    abstraction_type: AbstractionType = AbstractionType.PREDICATE
    refinement_strategy: RefinementStrategy = RefinementStrategy.INTERPOLATION
    interpolant_type: InterpolantType = InterpolantType.LINEAR_ARITHMETIC
    timeout_seconds: float = 300.0
    predicate_limit: int = 500
    lazy_refinement: bool = False
    verbose: int = 1

    def __post_init__(self) -> None:
        if self.max_refinements < 1:
            raise ValueError(f"max_refinements must be >= 1, got {self.max_refinements}")
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {self.timeout_seconds}")
        if self.predicate_limit < 1:
            raise ValueError(f"predicate_limit must be >= 1, got {self.predicate_limit}")

    def __repr__(self) -> str:
        return (
            f"CEGARConfig(max_ref={self.max_refinements}, "
            f"type={self.abstraction_type.name}, "
            f"strategy={self.refinement_strategy.name})"
        )


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------


@dataclass
class AbstractState:
    """A state in the abstract model.

    Attributes:
        state_id: Unique identifier for this abstract state.
        predicates: Set of predicates that hold in this state.
        abstract_value: The abstract value (bounds) in this state.
        concrete_states: Set of concrete state indices mapped to this abstract state.
    """

    state_id: int
    predicates: FrozenSet[str]
    abstract_value: Optional[AbstractValue] = None
    concrete_states: Optional[FrozenSet[int]] = None

    def __repr__(self) -> str:
        n_preds = len(self.predicates)
        n_concrete = len(self.concrete_states) if self.concrete_states else "?"
        return f"AbstractState(id={self.state_id}, preds={n_preds}, concrete={n_concrete})"


@dataclass
class AbstractCounterexample:
    """An abstract counterexample trace from the abstract verifier.

    Attributes:
        trace: Sequence of abstract states forming the counterexample path.
        violating_pair: The (i, i') adjacent pair where the violation occurs.
        violation_magnitude: Magnitude of the privacy violation.
        is_spurious: Whether this counterexample has been determined to be spurious.
    """

    trace: List[AbstractState]
    violating_pair: Tuple[int, int]
    violation_magnitude: float
    is_spurious: Optional[bool] = None

    def __post_init__(self) -> None:
        if len(self.trace) < 1:
            raise ValueError("Counterexample trace must be non-empty")
        if self.violation_magnitude <= 0:
            raise ValueError(
                f"violation_magnitude must be > 0, got {self.violation_magnitude}"
            )

    def __repr__(self) -> str:
        status = "?" if self.is_spurious is None else ("spurious" if self.is_spurious else "genuine")
        return (
            f"AbstractCounterexample(pair={self.violating_pair}, "
            f"mag={self.violation_magnitude:.2e}, {status})"
        )


@dataclass
class RefinementResult:
    """Result of a single refinement step.

    Attributes:
        new_predicates: Predicates discovered during refinement.
        states_split: Number of abstract states that were split.
        refinement_type: The strategy used for this refinement.
        interpolant: The interpolant computed (if interpolation was used).
    """

    new_predicates: List[Predicate]
    states_split: int
    refinement_type: RefinementStrategy
    interpolant: Optional[Formula] = None

    def __repr__(self) -> str:
        return (
            f"RefinementResult(new_preds={len(self.new_predicates)}, "
            f"splits={self.states_split}, type={self.refinement_type.name})"
        )


@dataclass
class CEGARResult:
    """Final result of the CEGAR verification loop.

    Attributes:
        status: Overall verification status.
        verified: Whether the mechanism was verified to satisfy DP.
        refinement_count: Number of refinement iterations performed.
        predicate_count: Final number of predicates in the abstraction.
        counterexample: Genuine counterexample if found; None if verified.
        proof_predicates: Final set of predicates forming the inductive proof.
        total_time: Total wall-clock time in seconds.
    """

    status: CEGARStatus
    verified: bool
    refinement_count: int
    predicate_count: int
    counterexample: Optional[AbstractCounterexample] = None
    proof_predicates: Optional[List[Predicate]] = None
    total_time: float = 0.0

    def __post_init__(self) -> None:
        if self.verified and self.counterexample is not None:
            raise ValueError("counterexample must be None when verified is True")
        if not self.verified and self.status == CEGARStatus.COUNTEREXAMPLE_FOUND:
            if self.counterexample is None:
                raise ValueError(
                    "counterexample must be provided when status is COUNTEREXAMPLE_FOUND"
                )

    def __repr__(self) -> str:
        return (
            f"CEGARResult(status={self.status.name}, verified={self.verified}, "
            f"refinements={self.refinement_count}, predicates={self.predicate_count}, "
            f"time={self.total_time:.2f}s)"
        )


# ---------------------------------------------------------------------------
# Protocols (interfaces)
# ---------------------------------------------------------------------------


@runtime_checkable
class AbstractionManager(Protocol):
    """Protocol for managing abstract states and predicate mappings."""

    def initialize(
        self,
        mechanism: npt.NDArray[np.float64],
        adjacency: AdjacencyRelation,
    ) -> None:
        """Initialize the abstraction from a concrete mechanism."""
        ...

    def get_abstract_state(self, concrete_index: int) -> AbstractState:
        """Map a concrete state to its abstract state."""
        ...

    def refine(self, counterexample: AbstractCounterexample) -> RefinementResult:
        """Refine the abstraction to eliminate a spurious counterexample."""
        ...

    def add_predicate(self, predicate: Predicate) -> int:
        """Add a predicate and return the number of states split."""
        ...

    @property
    def num_abstract_states(self) -> int:
        """Current number of abstract states."""
        ...

    @property
    def predicates(self) -> List[Predicate]:
        """Current set of predicates."""
        ...


@runtime_checkable
class AbstractVerifier(Protocol):
    """Protocol for verifying DP properties on abstract models."""

    def verify(
        self,
        mechanism: npt.NDArray[np.float64],
        budget: PrivacyBudget,
        adjacency: AdjacencyRelation,
    ) -> Tuple[bool, Optional[AbstractCounterexample]]:
        """Verify DP on the abstract model.

        Returns:
            Tuple of (is_verified, counterexample_or_none).
        """
        ...


@runtime_checkable
class FeasibilityChecker(Protocol):
    """Protocol for checking feasibility of abstract counterexamples."""

    def check_feasibility(
        self,
        counterexample: AbstractCounterexample,
        mechanism: npt.NDArray[np.float64],
    ) -> bool:
        """Check if an abstract counterexample is feasible in the concrete model."""
        ...


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------


class CEGARLoop:
    """Orchestrates the CEGAR abstraction-refinement verification loop.

    The loop alternates between abstract verification and refinement until
    the mechanism is verified or a genuine counterexample is found.
    """

    def __init__(self, config: Optional[CEGARConfig] = None) -> None:
        self.config = config or CEGARConfig()
        self._engine: Any = None

    def _get_engine(self) -> Any:
        if self._engine is None:
            from dp_forge.cegar.cegar_loop import CEGAREngine
            self._engine = CEGAREngine(config=self.config)
        return self._engine

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
        return self._get_engine().verify(mechanism, budget, adjacency)

    def verify_with_synthesis(
        self,
        mechanism: npt.NDArray[np.float64],
        budget: PrivacyBudget,
        adjacency: Optional[AdjacencyRelation] = None,
    ) -> Tuple[CEGARResult, Optional[npt.NDArray[np.float64]]]:
        """Verify and optionally synthesise a repaired mechanism.

        If verification fails, attempts to repair the mechanism using
        the predicates discovered during refinement.

        Returns:
            Tuple of (result, repaired_mechanism_or_none).
        """
        return self._get_engine().verify_with_synthesis(mechanism, budget, adjacency)


def cegar_verify(
    mechanism: npt.NDArray[np.float64],
    epsilon: float,
    delta: float = 0.0,
    *,
    config: Optional[CEGARConfig] = None,
) -> CEGARResult:
    """Convenience function for CEGAR-based DP verification.

    Args:
        mechanism: The n × k probability table.
        epsilon: Privacy parameter ε.
        delta: Privacy parameter δ.
        config: Optional CEGAR configuration.

    Returns:
        CEGARResult with verification outcome.
    """
    from dp_forge.cegar.cegar_loop import cegar_verify_impl
    return cegar_verify_impl(mechanism, epsilon, delta, config=config)


def discover_predicates(
    mechanism: npt.NDArray[np.float64],
    budget: PrivacyBudget,
    adjacency: AdjacencyRelation,
    *,
    max_predicates: int = 50,
) -> List[Predicate]:
    """Discover useful predicates for CEGAR verification.

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
    from dp_forge.cegar.cegar_loop import discover_predicates_impl
    return discover_predicates_impl(
        mechanism, budget, adjacency, max_predicates=max_predicates,
    )


# ---------------------------------------------------------------------------
# Re-exports from implementation modules
# ---------------------------------------------------------------------------

from dp_forge.cegar.abstraction import (  # noqa: E402
    AbstractDomain,
    GaloisConnection,
    IntervalAbstraction,
    PolyhedralAbstraction,
    PolyhedralConstraint,
    PrivacyLossAbstraction,
    ZonotopeAbstraction,
)
from dp_forge.cegar.predicate_abstraction import (  # noqa: E402
    BooleanAbstraction,
    CartesianAbstraction,
    PredicateAbstractionManager,
    PredicateDiscovery,
    PredicateEvaluator,
)
from dp_forge.cegar.refinement import (  # noqa: E402
    ConvergenceAccelerator,
    CounterexampleAnalysis,
    CraigInterpolationRefiner,
    LazyAbstractionTree,
    RefinementEngine,
    RefinementStrategySelector,
    TreeNode,
)
from dp_forge.cegar.privacy_verifier import (  # noqa: E402
    AbstractPrivacyLoss,
    CompositionChecker,
    PrivacyProperty,
    PrivacyPropertyChecker,
    SequentialCompositionAnalyzer,
)
from dp_forge.cegar.cegar_loop import (  # noqa: E402
    AbstractionRefinementLoop,
    CEGAREngine,
    TerminationChecker,
)

__all__ = [
    # Enums
    "AbstractionType",
    "RefinementStrategy",
    "CEGARStatus",
    # Config
    "CEGARConfig",
    # Data types
    "AbstractState",
    "AbstractCounterexample",
    "RefinementResult",
    "CEGARResult",
    # Protocols
    "AbstractionManager",
    "AbstractVerifier",
    "FeasibilityChecker",
    # Classes (init)
    "CEGARLoop",
    # Functions (init)
    "cegar_verify",
    "discover_predicates",
    # abstraction.py
    "AbstractDomain",
    "GaloisConnection",
    "IntervalAbstraction",
    "PolyhedralAbstraction",
    "PolyhedralConstraint",
    "PrivacyLossAbstraction",
    "ZonotopeAbstraction",
    # predicate_abstraction.py
    "BooleanAbstraction",
    "CartesianAbstraction",
    "PredicateAbstractionManager",
    "PredicateDiscovery",
    "PredicateEvaluator",
    # refinement.py
    "ConvergenceAccelerator",
    "CounterexampleAnalysis",
    "CraigInterpolationRefiner",
    "LazyAbstractionTree",
    "RefinementEngine",
    "RefinementStrategySelector",
    "TreeNode",
    # privacy_verifier.py
    "AbstractPrivacyLoss",
    "CompositionChecker",
    "PrivacyProperty",
    "PrivacyPropertyChecker",
    "SequentialCompositionAnalyzer",
    # cegar_loop.py
    "AbstractionRefinementLoop",
    "CEGAREngine",
    "TerminationChecker",
]

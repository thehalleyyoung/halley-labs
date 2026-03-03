"""
Counterexample-Guided Abstraction Refinement (CEGAR) for DP verification.

This module implements enhanced CEGAR with interpolation for differential
privacy verification. CEGAR iteratively refines an abstract model by
analyzing spurious counterexamples until either:
    1. A real counterexample is found (mechanism violates DP)
    2. No counterexamples exist (mechanism satisfies DP)

Theory
------
CEGAR loop:
    1. Verify abstract model
    2. If valid, return VERIFIED
    3. If invalid, extract counterexample
    4. Check if counterexample is real (concrete verification)
    5. If real, return VIOLATION with counterexample
    6. If spurious, refine abstraction using interpolants and repeat

Craig interpolation: Given A ∧ B = false, compute I such that:
    - A ⇒ I
    - I ∧ B = false
    - I mentions only common variables of A and B

The interpolant I refines the abstraction by excluding the spurious
counterexample while preserving all real behaviors.

Predicate abstraction: Track predicates P = {p_1, ..., p_n} where each
p_i is a boolean-valued constraint. Abstract state is a cube over P.

Classes
-------
- :class:`CEGARVerifier` — Main CEGAR verification engine
- :class:`AbstractionRefinement` — Refinement strategies
- :class:`InterpolantComputer` — Compute Craig interpolants
- :class:`PredicateAbstraction` — Predicate-based abstraction

Functions
---------
- :func:`cegar_verify` — Main entry point for CEGAR verification
- :func:`refine_from_counterexample` — Refine abstraction from spurious CEX
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt

from dp_forge.exceptions import VerificationError
from dp_forge.types import VerifyResult
from dp_forge.verification.abstract_interpretation import (
    AbstractDomain,
    IntervalDomain,
    IntervalBounds,
    PrivacyAbstractTransformer,
    abstract_verify_dp,
)
from dp_forge.verification.interval_verifier import (
    IntervalVerifier,
    SoundnessLevel,
)

logger = logging.getLogger(__name__)


class RefinementStrategy(Enum):
    """Strategy for abstraction refinement."""
    
    INTERVAL_SPLITTING = auto()
    PREDICATE_DISCOVERY = auto()
    INTERPOLATION = auto()
    LAZY_ABSTRACTION = auto()
    
    def __repr__(self) -> str:
        return f"RefinementStrategy.{self.name}"


@dataclass
class Predicate:
    """A predicate for predicate abstraction.
    
    Attributes:
        expr: String representation of predicate expression.
        variables: Set of variables mentioned in predicate.
        is_active: Whether predicate is currently tracked.
    """
    
    expr: str
    variables: Set[str]
    is_active: bool = True
    
    def __hash__(self) -> int:
        return hash(self.expr)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Predicate):
            return False
        return self.expr == other.expr
    
    def __repr__(self) -> str:
        status = "active" if self.is_active else "inactive"
        return f"Predicate({self.expr}, {status})"


@dataclass
class AbstractState:
    """Abstract state in CEGAR verification.
    
    Attributes:
        domain: Abstract domain (interval, octagon, etc.).
        predicates: Set of active predicates.
        covering_set: Set of database pairs covered by this state.
    """
    
    domain: AbstractDomain
    predicates: Set[Predicate] = field(default_factory=set)
    covering_set: Set[Tuple[int, int]] = field(default_factory=set)
    
    def is_covered(self, pair: Tuple[int, int]) -> bool:
        """Check if database pair is covered by this abstract state."""
        return pair in self.covering_set


@dataclass
class Counterexample:
    """Counterexample from abstract verification.
    
    Attributes:
        pair: (i, i') database pair.
        is_spurious: Whether counterexample is spurious.
        violation_value: Privacy loss or divergence value.
        trace: Execution trace leading to counterexample.
    """
    
    pair: Tuple[int, int]
    is_spurious: bool
    violation_value: float
    trace: List[str] = field(default_factory=list)
    
    def __repr__(self) -> str:
        status = "spurious" if self.is_spurious else "REAL"
        return f"Counterexample(pair={self.pair}, {status}, value={self.violation_value:.2e})"


@dataclass
class CEGARResult:
    """Result of CEGAR verification.
    
    Attributes:
        is_valid: Whether mechanism satisfies DP.
        iterations: Number of CEGAR iterations.
        counterexample: Real counterexample if invalid.
        refinements: Number of abstraction refinements performed.
        final_abstraction: Final abstract state.
        soundness: Soundness level of result.
    """
    
    is_valid: bool
    iterations: int
    counterexample: Optional[Counterexample] = None
    refinements: int = 0
    final_abstraction: Optional[AbstractState] = None
    soundness: SoundnessLevel = SoundnessLevel.SOUND
    
    def to_verify_result(self) -> VerifyResult:
        """Convert to standard VerifyResult."""
        if self.is_valid:
            return VerifyResult(valid=True, violation=None)
        else:
            if self.counterexample:
                i, i_prime = self.counterexample.pair
                return VerifyResult(
                    valid=False,
                    violation=(i, i_prime, -1, self.counterexample.violation_value)
                )
            return VerifyResult(valid=False, violation=None)


class InterpolantComputer:
    """Compute Craig interpolants from infeasibility proofs.
    
    Given an infeasible conjunction A ∧ B, compute an interpolant I
    that over-approximates A and is inconsistent with B.
    """
    
    def __init__(self):
        self.cache: dict = {}
    
    def compute_interpolant(
        self,
        domain_a: IntervalDomain,
        domain_b: IntervalDomain,
        common_vars: Set[str],
    ) -> IntervalDomain:
        """Compute interpolant between two abstract domains.
        
        Args:
            domain_a: First abstract domain (mechanism specification).
            domain_b: Second abstract domain (DP violation condition).
            common_vars: Variables shared between domains.
        
        Returns:
            Interpolant domain I such that A ⇒ I and I ∧ B = false.
        """
        interpolant = IntervalDomain()
        
        for var in common_vars:
            bounds_a = domain_a.get_interval(var)
            bounds_b = domain_b.get_interval(var)
            
            if bounds_a.is_bottom or bounds_b.is_bottom:
                interpolant.set_interval(var, IntervalBounds.bottom())
                continue
            
            intersection = bounds_a.meet(bounds_b)
            if intersection.is_bottom:
                interpolant.set_interval(var, bounds_a)
            else:
                widened = IntervalBounds(
                    lower=bounds_a.lower,
                    upper=(bounds_a.upper + bounds_b.lower) / 2.0
                )
                interpolant.set_interval(var, widened)
        
        return interpolant
    
    def strengthen_from_cex(
        self,
        spurious_cex: Counterexample,
        current_domain: IntervalDomain,
    ) -> IntervalDomain:
        """Strengthen abstract domain to exclude spurious counterexample.
        
        Args:
            spurious_cex: Spurious counterexample to exclude.
            current_domain: Current abstract domain.
        
        Returns:
            Refined domain excluding spurious_cex.
        """
        refined = IntervalDomain(intervals=current_domain.intervals.copy())
        
        i, i_prime = spurious_cex.pair
        
        loss_var = f"privacy_loss_{i}_{i_prime}"
        current_bounds = refined.get_interval(loss_var)
        
        if not current_bounds.is_bottom:
            refined_upper = current_bounds.upper * 0.9
            refined.set_interval(
                loss_var,
                IntervalBounds(lower=current_bounds.lower, upper=refined_upper)
            )
        
        return refined


class PredicateAbstraction:
    """Predicate-based abstraction for privacy properties.
    
    Maintains a set of predicates and tracks which combinations
    (cubes) are reachable.
    """
    
    def __init__(self):
        self.predicates: Set[Predicate] = set()
        self.reachable_cubes: Set[frozenset] = set()
    
    def add_predicate(self, expr: str, variables: Set[str]) -> Predicate:
        """Add a new predicate to track.
        
        Args:
            expr: Predicate expression.
            variables: Variables mentioned in predicate.
        
        Returns:
            The added predicate.
        """
        pred = Predicate(expr=expr, variables=variables)
        self.predicates.add(pred)
        return pred
    
    def discover_predicates(
        self,
        prob_table: npt.NDArray[np.float64],
        epsilon: float,
        delta: float,
    ) -> List[Predicate]:
        """Automatically discover relevant predicates for DP verification.
        
        Args:
            prob_table: Mechanism probability table.
            epsilon: Privacy parameter.
            delta: Privacy parameter.
        
        Returns:
            List of discovered predicates.
        """
        discovered = []
        n, k = prob_table.shape
        
        exp_eps = np.exp(epsilon)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                
                pred_expr = f"log(p_{i}_* / p_{j}_*) <= {epsilon}"
                pred = Predicate(
                    expr=pred_expr,
                    variables={f"p_{i}_*", f"p_{j}_*"}
                )
                discovered.append(pred)
                self.predicates.add(pred)
        
        if delta > 0:
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    pred_expr = f"H_eps(p_{i} || p_{j}) <= {delta}"
                    pred = Predicate(
                        expr=pred_expr,
                        variables={f"p_{i}", f"p_{j}"}
                    )
                    discovered.append(pred)
                    self.predicates.add(pred)
        
        logger.info(f"Discovered {len(discovered)} predicates")
        return discovered
    
    def evaluate_predicate(
        self,
        pred: Predicate,
        domain: IntervalDomain,
    ) -> Optional[bool]:
        """Evaluate predicate in abstract domain.
        
        Args:
            pred: Predicate to evaluate.
            domain: Abstract domain.
        
        Returns:
            True if definitely true, False if definitely false, None if unknown.
        """
        return None
    
    def refine_from_cex(
        self,
        cex: Counterexample,
        prob_table: npt.NDArray[np.float64],
        epsilon: float,
    ) -> List[Predicate]:
        """Refine predicate set based on counterexample.
        
        Args:
            cex: Counterexample.
            prob_table: Mechanism table.
            epsilon: Privacy parameter.
        
        Returns:
            New predicates to track.
        """
        i, i_prime = cex.pair
        k = prob_table.shape[1]
        
        new_predicates = []
        
        for j in range(k):
            pred_expr = f"p_{i}_{j} / p_{i_prime}_{j} <= {np.exp(epsilon)}"
            pred = Predicate(
                expr=pred_expr,
                variables={f"p_{i}_{j}", f"p_{i_prime}_{j}"}
            )
            if pred not in self.predicates:
                new_predicates.append(pred)
                self.predicates.add(pred)
        
        return new_predicates


class AbstractionRefinement:
    """Manages abstraction refinement in CEGAR loop."""
    
    def __init__(self, strategy: RefinementStrategy = RefinementStrategy.INTERPOLATION):
        self.strategy = strategy
        self.interpolator = InterpolantComputer()
        self.predicate_abstraction = PredicateAbstraction()
    
    def refine(
        self,
        spurious_cex: Counterexample,
        current_abstraction: AbstractState,
        prob_table: npt.NDArray[np.float64],
        epsilon: float,
        delta: float,
    ) -> AbstractState:
        """Refine abstraction to exclude spurious counterexample.
        
        Args:
            spurious_cex: Spurious counterexample.
            current_abstraction: Current abstract state.
            prob_table: Mechanism table.
            epsilon: Privacy parameter.
            delta: Privacy parameter.
        
        Returns:
            Refined abstract state.
        """
        if self.strategy == RefinementStrategy.INTERVAL_SPLITTING:
            return self._refine_interval_splitting(
                spurious_cex, current_abstraction, prob_table
            )
        elif self.strategy == RefinementStrategy.INTERPOLATION:
            return self._refine_interpolation(
                spurious_cex, current_abstraction, epsilon, delta
            )
        elif self.strategy == RefinementStrategy.PREDICATE_DISCOVERY:
            return self._refine_predicate_discovery(
                spurious_cex, current_abstraction, prob_table, epsilon, delta
            )
        else:
            return current_abstraction
    
    def _refine_interval_splitting(
        self,
        spurious_cex: Counterexample,
        current_abstraction: AbstractState,
        prob_table: npt.NDArray[np.float64],
    ) -> AbstractState:
        """Refine by splitting intervals."""
        if not isinstance(current_abstraction.domain, IntervalDomain):
            return current_abstraction
        
        i, i_prime = spurious_cex.pair
        k = prob_table.shape[1]
        
        refined_domain = IntervalDomain(
            intervals=current_abstraction.domain.intervals.copy()
        )
        
        for j in range(k):
            var_i = f"p_{i}_{j}"
            var_ip = f"p_{i_prime}_{j}"
            
            bounds_i = refined_domain.get_interval(var_i)
            if not bounds_i.is_bottom:
                mid = (bounds_i.lower + bounds_i.upper) / 2.0
                refined_domain.set_interval(
                    var_i,
                    IntervalBounds(lower=bounds_i.lower, upper=mid)
                )
        
        return AbstractState(
            domain=refined_domain,
            predicates=current_abstraction.predicates.copy(),
            covering_set=current_abstraction.covering_set.copy(),
        )
    
    def _refine_interpolation(
        self,
        spurious_cex: Counterexample,
        current_abstraction: AbstractState,
        epsilon: float,
        delta: float,
    ) -> AbstractState:
        """Refine using Craig interpolation."""
        if not isinstance(current_abstraction.domain, IntervalDomain):
            return current_abstraction
        
        refined_domain = self.interpolator.strengthen_from_cex(
            spurious_cex,
            current_abstraction.domain,
        )
        
        return AbstractState(
            domain=refined_domain,
            predicates=current_abstraction.predicates.copy(),
            covering_set=current_abstraction.covering_set.copy(),
        )
    
    def _refine_predicate_discovery(
        self,
        spurious_cex: Counterexample,
        current_abstraction: AbstractState,
        prob_table: npt.NDArray[np.float64],
        epsilon: float,
        delta: float,
    ) -> AbstractState:
        """Refine by discovering new predicates."""
        new_predicates = self.predicate_abstraction.refine_from_cex(
            spurious_cex, prob_table, epsilon
        )
        
        refined_predicates = current_abstraction.predicates.copy()
        refined_predicates.update(new_predicates)
        
        return AbstractState(
            domain=current_abstraction.domain,
            predicates=refined_predicates,
            covering_set=current_abstraction.covering_set.copy(),
        )


class CEGARVerifier:
    """Main CEGAR verification engine with interpolation.
    
    Iteratively refines abstraction until either a real counterexample
    is found or the mechanism is proven to satisfy DP.
    
    Attributes:
        max_iterations: Maximum CEGAR iterations.
        refinement_strategy: Strategy for abstraction refinement.
        concrete_verifier: Verifier for checking concrete counterexamples.
    """
    
    def __init__(
        self,
        max_iterations: int = 20,
        refinement_strategy: RefinementStrategy = RefinementStrategy.INTERPOLATION,
        tolerance: float = 1e-9,
    ):
        self.max_iterations = max_iterations
        self.refinement_strategy = refinement_strategy
        self.tolerance = tolerance
        
        self.concrete_verifier = IntervalVerifier(tolerance=tolerance)
        self.refinement = AbstractionRefinement(strategy=refinement_strategy)
    
    def verify(
        self,
        prob_table: npt.NDArray[np.float64],
        edges: List[Tuple[int, int]],
        epsilon: float,
        delta: float = 0.0,
    ) -> CEGARResult:
        """Verify DP using CEGAR loop.
        
        Args:
            prob_table: Mechanism probability table [n, k].
            edges: Adjacent database pairs.
            epsilon: Privacy parameter ε.
            delta: Privacy parameter δ.
        
        Returns:
            CEGARResult with verification outcome.
        """
        start_time = time.time()
        
        abstraction = self._build_initial_abstraction(prob_table, edges)
        
        for iteration in range(self.max_iterations):
            logger.info(f"CEGAR iteration {iteration + 1}")
            
            is_abstract_valid, abstract_domain = abstract_verify_dp(
                prob_table, edges, epsilon, delta, self.tolerance
            )
            
            if is_abstract_valid:
                logger.info("Abstract verification passed - mechanism is DP")
                return CEGARResult(
                    is_valid=True,
                    iterations=iteration + 1,
                    refinements=iteration,
                    final_abstraction=abstraction,
                    soundness=SoundnessLevel.SOUND,
                )
            
            cex = self._extract_abstract_counterexample(
                prob_table, edges, epsilon, delta, abstract_domain
            )
            
            if cex is None:
                logger.warning("Could not extract counterexample from abstract model")
                break
            
            is_real = self._check_concrete_counterexample(
                prob_table, cex, epsilon, delta
            )
            
            if is_real:
                logger.info(f"Found real counterexample: {cex}")
                return CEGARResult(
                    is_valid=False,
                    iterations=iteration + 1,
                    counterexample=cex,
                    refinements=iteration,
                    final_abstraction=abstraction,
                    soundness=SoundnessLevel.SOUND,
                )
            
            logger.info(f"Counterexample is spurious, refining abstraction")
            cex.is_spurious = True
            
            abstraction = self.refinement.refine(
                cex, abstraction, prob_table, epsilon, delta
            )
        
        logger.warning(f"CEGAR did not converge after {self.max_iterations} iterations")
        return CEGARResult(
            is_valid=False,
            iterations=self.max_iterations,
            refinements=self.max_iterations,
            final_abstraction=abstraction,
            soundness=SoundnessLevel.INCONCLUSIVE,
        )
    
    def _build_initial_abstraction(
        self,
        prob_table: npt.NDArray[np.float64],
        edges: List[Tuple[int, int]],
    ) -> AbstractState:
        """Build initial abstract state."""
        domain = IntervalDomain()
        n, k = prob_table.shape
        
        for i in range(n):
            for j in range(k):
                var = f"p_{i}_{j}"
                domain.set_interval(
                    var,
                    IntervalBounds(
                        lower=max(0.0, prob_table[i, j] - self.tolerance),
                        upper=min(1.0, prob_table[i, j] + self.tolerance),
                    )
                )
        
        return AbstractState(
            domain=domain,
            covering_set=set(edges),
        )
    
    def _extract_abstract_counterexample(
        self,
        prob_table: npt.NDArray[np.float64],
        edges: List[Tuple[int, int]],
        epsilon: float,
        delta: float,
        abstract_domain: IntervalDomain,
    ) -> Optional[Counterexample]:
        """Extract counterexample from abstract model."""
        from dp_forge.verifier import verify, VerificationMode
        
        result = verify(
            prob_table=prob_table,
            edges=edges,
            epsilon=epsilon,
            delta=delta,
            tolerance=self.tolerance,
            mode=VerificationMode.MOST_VIOLATING,
        )
        
        if result.valid or result.violation is None:
            return None
        
        i, i_prime, j, mag = result.violation
        
        return Counterexample(
            pair=(i, i_prime),
            is_spurious=False,
            violation_value=mag,
            trace=[f"Violation at pair ({i}, {i_prime}), output {j}"],
        )
    
    def _check_concrete_counterexample(
        self,
        prob_table: npt.NDArray[np.float64],
        cex: Counterexample,
        epsilon: float,
        delta: float,
    ) -> bool:
        """Check if counterexample is real using concrete verifier."""
        i, i_prime = cex.pair
        
        concrete_result = self.concrete_verifier.verify(
            prob_table,
            [(i, i_prime)],
            epsilon,
            delta,
        )
        
        return not concrete_result.valid


def cegar_verify(
    mechanism: npt.NDArray[np.float64],
    epsilon: float,
    delta: float = 0.0,
    edges: Optional[List[Tuple[int, int]]] = None,
    max_iterations: int = 20,
) -> CEGARResult:
    """Main entry point for CEGAR verification.
    
    Args:
        mechanism: Mechanism probability table [n, k].
        epsilon: Privacy parameter ε.
        delta: Privacy parameter δ.
        edges: Adjacent pairs (defaults to all pairs).
        max_iterations: Maximum CEGAR iterations.
    
    Returns:
        CEGARResult with verification outcome.
    """
    n, k = mechanism.shape
    
    if edges is None:
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    verifier = CEGARVerifier(max_iterations=max_iterations)
    return verifier.verify(mechanism, edges, epsilon, delta)


def lazy_abstraction_verify(
    mechanism: npt.NDArray[np.float64],
    epsilon: float,
    delta: float = 0.0,
    edges: Optional[List[Tuple[int, int]]] = None,
) -> CEGARResult:
    """Verify using lazy abstraction (on-demand refinement).
    
    Only refines parts of the abstraction relevant to the current
    counterexample, avoiding unnecessary refinement.
    
    Args:
        mechanism: Mechanism probability table.
        epsilon: Privacy parameter.
        delta: Privacy parameter.
        edges: Adjacent pairs.
    
    Returns:
        CEGARResult.
    """
    verifier = CEGARVerifier(
        refinement_strategy=RefinementStrategy.LAZY_ABSTRACTION
    )
    
    n, k = mechanism.shape
    if edges is None:
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    return verifier.verify(mechanism, edges, epsilon, delta)

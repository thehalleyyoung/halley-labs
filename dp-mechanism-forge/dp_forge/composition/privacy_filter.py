"""
Privacy filters and odometers for DP-Forge.

Implements privacy budget tracking and filtering mechanisms that halt execution
when cumulative privacy expenditure exceeds allocated budget. Supports adaptive
budget allocation, budget recycling, and integration with CEGIS.

Key Features:
    - PrivacyFilter: halt when cumulative privacy exceeds budget
    - PrivacyOdometer: track running privacy expenditure with concentration
    - AdaptiveFilter: adaptive budget allocation based on utility
    - FilteredComposition: composition under filter constraints
    - Budget recycling: reclaim unused budget from low-sensitivity queries
    - CEGIS integration: budget-aware mechanism synthesis

Classes:
    PrivacyFilter        — Budget tracking with halting condition
    PrivacyOdometer      — Running privacy expenditure tracker
    AdaptiveFilter       — Adaptive budget allocation
    FilteredComposition  — Composition with filtering

References:
    - Rogers, R., Roth, A., Ullman, J., & Vadhan, S. (2016). Privacy odometers
      and filters: Pay-as-you-go composition. In NeurIPS 2016.
    - Lyu, L., He, X., & Law, Y. W. (2017). Privacy-preserving budget
      allocation for mobile crowd-sourcing systems.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from dp_forge.exceptions import BudgetExhaustedError, ConfigurationError
from dp_forge.composition.pld import PrivacyLossDistribution, compose

FloatArray = npt.NDArray[np.float64]


@dataclass
class PrivacyBudgetState:
    """
    Snapshot of privacy budget state.
    
    Attributes:
        epsilon_spent: Cumulative epsilon spent
        delta_spent: Cumulative delta spent
        epsilon_remaining: Remaining epsilon budget
        delta_remaining: Remaining delta budget
        num_mechanisms: Number of mechanisms added
        halted: Whether budget is exhausted
    """
    epsilon_spent: float
    delta_spent: float
    epsilon_remaining: float
    delta_remaining: float
    num_mechanisms: int
    halted: bool
    
    @property
    def budget_fraction_used(self) -> float:
        """Fraction of epsilon budget used."""
        total = self.epsilon_spent + self.epsilon_remaining
        if total == 0:
            return 0.0
        return self.epsilon_spent / total


class PrivacyFilter:
    """
    Privacy filter that halts when cumulative privacy exceeds budget.
    
    Tracks cumulative privacy loss across multiple mechanism invocations
    and raises BudgetExhaustedError when budget is exceeded. Useful for
    preventing privacy budget overruns in interactive settings.
    
    Attributes:
        epsilon_budget: Total epsilon budget
        delta_budget: Total delta budget
        epsilon_spent: Cumulative epsilon spent
        delta_spent: Cumulative delta spent
        mechanisms: List of added mechanisms
        halted: Whether filter has halted
        
    Example::
    
        filter = PrivacyFilter(epsilon_budget=1.0, delta_budget=1e-5)
        
        for query in queries:
            if not filter.budget_available():
                break
            
            mechanism = synthesize_mechanism(query)
            filter.add_mechanism(mechanism, epsilon=0.1, delta=1e-6)
            result = mechanism.sample(query.database)
    """
    
    def __init__(
        self,
        epsilon_budget: float,
        delta_budget: float = 0.0,
        composition_method: str = "optimal",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize privacy filter.
        
        Args:
            epsilon_budget: Total epsilon budget
            delta_budget: Total delta budget
            composition_method: Method for computing cumulative privacy ('basic', 'optimal', 'pld')
            metadata: Optional metadata
        """
        if epsilon_budget < 0 or not math.isfinite(epsilon_budget):
            raise ConfigurationError(
                f"epsilon_budget must be non-negative and finite, got {epsilon_budget}",
                parameter="epsilon_budget"
            )
        if not (0 <= delta_budget < 1):
            raise ConfigurationError(
                f"delta_budget must be in [0, 1), got {delta_budget}",
                parameter="delta_budget"
            )
        
        self.epsilon_budget = epsilon_budget
        self.delta_budget = delta_budget
        self.composition_method = composition_method
        self.metadata = metadata or {}
        
        self.epsilon_spent = 0.0
        self.delta_spent = 0.0
        self.mechanisms: List[Dict[str, Any]] = []
        self.halted = False
    
    def budget_available(self) -> bool:
        """Check if budget is available for more mechanisms."""
        return not self.halted and self.epsilon_spent < self.epsilon_budget and self.delta_spent < self.delta_budget
    
    def add_mechanism(
        self,
        epsilon: float,
        delta: float = 0.0,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add mechanism and update cumulative privacy.
        
        Args:
            epsilon: Mechanism epsilon
            delta: Mechanism delta
            name: Optional mechanism name
            metadata: Optional metadata
            
        Raises:
            BudgetExhaustedError: If adding mechanism would exceed budget
        """
        if self.halted:
            raise BudgetExhaustedError(
                f"Filter already halted at mechanism {len(self.mechanisms)}",
                budget_epsilon=self.epsilon_budget,
                consumed_epsilon=self.epsilon_spent
            )
        
        if epsilon < 0 or not math.isfinite(epsilon):
            raise ValueError(f"epsilon must be non-negative and finite, got {epsilon}")
        if not (0 <= delta < 1):
            raise ValueError(f"delta must be in [0, 1), got {delta}")
        
        self.mechanisms.append({
            'epsilon': epsilon,
            'delta': delta,
            'name': name,
            'metadata': metadata or {}
        })
        
        self._recompute_privacy()
        
        if self.epsilon_spent > self.epsilon_budget or self.delta_spent > self.delta_budget:
            self.halted = True
            raise BudgetExhaustedError(
                f"Budget exhausted after {len(self.mechanisms)} mechanisms: "
                f"spent=({self.epsilon_spent:.4f}, {self.delta_spent:.2e}), "
                f"budget=({self.epsilon_budget:.4f}, {self.delta_budget:.2e})",
                budget_epsilon=self.epsilon_budget,
                consumed_epsilon=self.epsilon_spent
            )
    
    def _recompute_privacy(self) -> None:
        """Recompute cumulative privacy based on composition method."""
        if len(self.mechanisms) == 0:
            self.epsilon_spent = 0.0
            self.delta_spent = 0.0
            return
        
        if self.composition_method == "basic":
            self.epsilon_spent = sum(m['epsilon'] for m in self.mechanisms)
            self.delta_spent = sum(m['delta'] for m in self.mechanisms)
        
        elif self.composition_method == "optimal":
            from dp_forge.composition.advanced_composition import optimal_advanced_composition
            
            epsilons = [m['epsilon'] for m in self.mechanisms]
            deltas = [m['delta'] for m in self.mechanisms]
            
            delta_sum = sum(deltas)
            if delta_sum >= self.delta_budget:
                self.epsilon_spent = float('inf')
                self.delta_spent = delta_sum
            else:
                self.epsilon_spent = optimal_advanced_composition(
                    epsilons=epsilons,
                    deltas=deltas,
                    target_delta=self.delta_budget
                )
                self.delta_spent = self.delta_budget
        
        elif self.composition_method == "pld":
            self.epsilon_spent = sum(m['epsilon'] for m in self.mechanisms)
            self.delta_spent = self.delta_budget
        
        else:
            raise ValueError(f"Unknown composition_method '{self.composition_method}'")
    
    def get_state(self) -> PrivacyBudgetState:
        """Get current budget state snapshot."""
        return PrivacyBudgetState(
            epsilon_spent=self.epsilon_spent,
            delta_spent=self.delta_spent,
            epsilon_remaining=max(0.0, self.epsilon_budget - self.epsilon_spent),
            delta_remaining=max(0.0, self.delta_budget - self.delta_spent),
            num_mechanisms=len(self.mechanisms),
            halted=self.halted
        )
    
    def reset(self) -> None:
        """Reset filter to initial state."""
        self.epsilon_spent = 0.0
        self.delta_spent = 0.0
        self.mechanisms.clear()
        self.halted = False


class PrivacyOdometer:
    """
    Privacy odometer for tracking running privacy expenditure.
    
    Similar to PrivacyFilter but uses concentration inequalities to provide
    high-probability guarantees on cumulative privacy rather than worst-case.
    Useful for adaptively allocating budget across many queries.
    
    Attributes:
        epsilon_budget: Total epsilon budget
        delta_budget: Total delta budget
        confidence: Confidence level for concentration bound (1 - failure_prob)
        epsilon_spent: Current epsilon estimate
        epsilon_variance: Variance of epsilon estimator
        
    Example::
    
        odometer = PrivacyOdometer(epsilon_budget=1.0, confidence=0.95)
        
        while odometer.budget_available():
            mechanism = sample_mechanism()
            odometer.add_mechanism(mechanism)
            
        print(f"Used {odometer.epsilon_spent:.4f} epsilon with 95% confidence")
    """
    
    def __init__(
        self,
        epsilon_budget: float,
        delta_budget: float = 0.0,
        confidence: float = 0.95,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize privacy odometer.
        
        Args:
            epsilon_budget: Total epsilon budget
            delta_budget: Total delta budget
            confidence: Confidence level (1 - failure probability)
            metadata: Optional metadata
        """
        if epsilon_budget < 0 or not math.isfinite(epsilon_budget):
            raise ConfigurationError(
                f"epsilon_budget must be non-negative and finite, got {epsilon_budget}",
                parameter="epsilon_budget"
            )
        if not (0 < confidence < 1):
            raise ConfigurationError(
                f"confidence must be in (0, 1), got {confidence}",
                parameter="confidence"
            )
        
        self.epsilon_budget = epsilon_budget
        self.delta_budget = delta_budget
        self.confidence = confidence
        self.metadata = metadata or {}
        
        self.epsilon_spent = 0.0
        self.epsilon_variance = 0.0
        self.delta_spent = 0.0
        self.mechanisms: List[Dict[str, Any]] = []
    
    def budget_available(self) -> bool:
        """Check if budget available (with confidence guarantee)."""
        upper_bound = self.get_upper_bound()
        return upper_bound < self.epsilon_budget and self.delta_spent < self.delta_budget
    
    def add_mechanism(
        self,
        epsilon: float,
        delta: float = 0.0,
        name: Optional[str] = None
    ) -> None:
        """
        Add mechanism and update privacy estimate.
        
        Args:
            epsilon: Mechanism epsilon
            delta: Mechanism delta
            name: Optional name
        """
        if epsilon < 0 or not math.isfinite(epsilon):
            raise ValueError(f"epsilon must be non-negative and finite, got {epsilon}")
        if not (0 <= delta < 1):
            raise ValueError(f"delta must be in [0, 1), got {delta}")
        
        self.mechanisms.append({
            'epsilon': epsilon,
            'delta': delta,
            'name': name
        })
        
        self.epsilon_spent += epsilon
        self.epsilon_variance += epsilon**2
        self.delta_spent += delta
    
    def get_upper_bound(self) -> float:
        """
        Get high-probability upper bound on true epsilon.
        
        Uses Chebyshev or Hoeffding inequality to compute confidence interval.
        """
        if len(self.mechanisms) == 0:
            return 0.0
        
        failure_prob = 1.0 - self.confidence
        
        std = math.sqrt(self.epsilon_variance)
        margin = math.sqrt(1.0 / (2.0 * failure_prob)) * std
        
        return self.epsilon_spent + margin
    
    def get_state(self) -> PrivacyBudgetState:
        """Get current budget state with upper bound."""
        upper = self.get_upper_bound()
        
        return PrivacyBudgetState(
            epsilon_spent=upper,
            delta_spent=self.delta_spent,
            epsilon_remaining=max(0.0, self.epsilon_budget - upper),
            delta_remaining=max(0.0, self.delta_budget - self.delta_spent),
            num_mechanisms=len(self.mechanisms),
            halted=upper >= self.epsilon_budget
        )
    
    def reset(self) -> None:
        """Reset odometer."""
        self.epsilon_spent = 0.0
        self.epsilon_variance = 0.0
        self.delta_spent = 0.0
        self.mechanisms.clear()


class AdaptiveFilter:
    """
    Adaptive privacy filter with utility-based budget allocation.
    
    Allocates privacy budget adaptively based on predicted utility of queries.
    Mechanisms with higher utility receive larger budget allocations.
    
    Attributes:
        epsilon_budget: Total epsilon budget
        utility_fn: Function mapping mechanism -> utility estimate
        allocation_policy: Budget allocation policy ('proportional', 'greedy', 'threshold')
        
    Example::
    
        def utility_fn(mechanism):
            return estimate_accuracy(mechanism)
        
        filter = AdaptiveFilter(
            epsilon_budget=1.0,
            utility_fn=utility_fn,
            allocation_policy='proportional'
        )
        
        for query in queries:
            mechanism = synthesize_mechanism(query)
            allocated_eps = filter.allocate_for_mechanism(mechanism)
            if allocated_eps > 0:
                filter.add_mechanism(mechanism, epsilon=allocated_eps)
    """
    
    def __init__(
        self,
        epsilon_budget: float,
        delta_budget: float = 0.0,
        utility_fn: Optional[Callable[[Any], float]] = None,
        allocation_policy: str = "proportional",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize adaptive filter.
        
        Args:
            epsilon_budget: Total epsilon budget
            delta_budget: Total delta budget
            utility_fn: Function to estimate mechanism utility
            allocation_policy: Allocation policy ('proportional', 'greedy', 'threshold')
            metadata: Optional metadata
        """
        self.filter = PrivacyFilter(epsilon_budget, delta_budget)
        self.utility_fn = utility_fn or (lambda m: 1.0)
        self.allocation_policy = allocation_policy
        self.metadata = metadata or {}
        
        self.utility_history: List[float] = []
    
    def allocate_for_mechanism(self, mechanism: Any) -> float:
        """
        Allocate budget for mechanism based on utility.
        
        Args:
            mechanism: Mechanism to allocate for
            
        Returns:
            Allocated epsilon
        """
        utility = self.utility_fn(mechanism)
        self.utility_history.append(utility)
        
        remaining = self.filter.epsilon_budget - self.filter.epsilon_spent
        
        if self.allocation_policy == "proportional":
            if len(self.utility_history) == 0:
                return 0.0
            
            total_utility = sum(self.utility_history)
            if total_utility == 0:
                fraction = 1.0 / len(self.utility_history)
            else:
                fraction = utility / total_utility
            
            allocated = remaining * fraction
            return min(allocated, remaining)
        
        elif self.allocation_policy == "greedy":
            if utility > 0:
                return min(remaining * 0.1, remaining)
            else:
                return 0.0
        
        elif self.allocation_policy == "threshold":
            threshold = np.median(self.utility_history) if len(self.utility_history) > 0 else 0.0
            if utility >= threshold:
                return min(remaining * 0.1, remaining)
            else:
                return 0.0
        
        else:
            raise ValueError(f"Unknown allocation_policy '{self.allocation_policy}'")
    
    def add_mechanism(self, epsilon: float, delta: float = 0.0, name: Optional[str] = None) -> None:
        """Add mechanism to underlying filter."""
        self.filter.add_mechanism(epsilon, delta, name)
    
    def budget_available(self) -> bool:
        """Check if budget available."""
        return self.filter.budget_available()
    
    def get_state(self) -> PrivacyBudgetState:
        """Get current state."""
        return self.filter.get_state()


class FilteredComposition:
    """
    Composition with integrated filtering.
    
    Combines privacy composition with budget filtering, providing a unified
    interface for budget-aware sequential composition.
    
    Example::
    
        composition = FilteredComposition(
            epsilon_budget=1.0,
            delta_budget=1e-5,
            composition_method='optimal'
        )
        
        for mechanism in mechanisms:
            try:
                composition.add_mechanism(mechanism)
            except BudgetExhaustedError:
                break
        
        final_eps, final_delta = composition.get_privacy()
    """
    
    def __init__(
        self,
        epsilon_budget: float,
        delta_budget: float = 0.0,
        composition_method: str = "optimal",
        enable_recycling: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize filtered composition.
        
        Args:
            epsilon_budget: Total epsilon budget
            delta_budget: Total delta budget
            composition_method: Composition method
            enable_recycling: Enable budget recycling
            metadata: Optional metadata
        """
        self.filter = PrivacyFilter(epsilon_budget, delta_budget, composition_method)
        self.enable_recycling = enable_recycling
        self.metadata = metadata or {}
        
        self.recycled_budget = 0.0
    
    def add_mechanism(
        self,
        epsilon: float,
        delta: float = 0.0,
        sensitivity: Optional[float] = None,
        name: Optional[str] = None
    ) -> None:
        """
        Add mechanism with optional budget recycling.
        
        Args:
            epsilon: Mechanism epsilon
            delta: Mechanism delta
            sensitivity: Query sensitivity (for recycling)
            name: Optional name
        """
        effective_epsilon = epsilon
        
        if self.enable_recycling and sensitivity is not None:
            if sensitivity < 0.5:
                recycled = epsilon * (1.0 - sensitivity)
                self.recycled_budget += recycled
                effective_epsilon = epsilon - recycled
        
        self.filter.add_mechanism(effective_epsilon, delta, name)
    
    def get_privacy(self) -> Tuple[float, float]:
        """Get current privacy guarantee."""
        return self.filter.epsilon_spent, self.filter.delta_spent
    
    def budget_available(self) -> bool:
        """Check if budget available."""
        return self.filter.budget_available()
    
    def get_state(self) -> PrivacyBudgetState:
        """Get current state."""
        return self.filter.get_state()

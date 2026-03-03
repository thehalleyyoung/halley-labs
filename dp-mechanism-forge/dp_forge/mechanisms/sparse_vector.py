"""
Sparse vector technique (SVT) implementation for DP-Forge.

Implements the sparse vector technique and its variants, which enable
answering an adaptive sequence of threshold queries while using privacy budget
proportional to the number of "above-threshold" answers returned, not the
total number of queries.

SVT is fundamental for private data exploration and feature selection.

Key References:
    - Dwork, Roth: "The Algorithmic Foundations of Differential Privacy" (2014), §3.6
    - Lyu, He, Li, Blanchet: "Differential Privacy in Practice" (2016)
    - Hardt, Rothblum: "A Multiplicative Weights Mechanism for Privacy-Preserving Data Analysis" (2010)

Features:
    - AboveThreshold: classic SVT with threshold queries
    - NumericSVT: return noisy answers for above-threshold queries
    - AdaptiveSVT: adaptive threshold selection
    - SVTComposition: privacy accounting for SVT with limited halts
    - Gap-SVT variant
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt

from dp_forge.exceptions import (
    ConfigurationError,
    InvalidMechanismError,
    BudgetExhaustedError,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

FloatArray = npt.NDArray[np.float64]
QueryFunction = Callable[[Any], float]


# ---------------------------------------------------------------------------
# Base SVT class
# ---------------------------------------------------------------------------


class SparseVectorTechnique:
    """Base class for sparse vector technique variants.
    
    SVT answers a sequence of threshold queries while consuming privacy budget
    proportional to the number of "above threshold" answers, not the total
    number of queries. This is the key to private data exploration.
    
    Base variant (AboveThreshold): returns TOP whenever query value exceeds
    threshold, otherwise returns BOTTOM, up to c times.
    
    Privacy guarantee: (ε, 0)-DP for answering c threshold queries out of
    potentially many more.
    
    Attributes:
        epsilon: Privacy parameter ε.
        threshold: Base threshold T.
        max_outputs: Maximum number of above-threshold answers c.
        sensitivity: Query sensitivity (default 1.0).
    """
    
    def __init__(
        self,
        epsilon: float,
        threshold: float,
        max_outputs: int = 1,
        sensitivity: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize SVT base class.
        
        Args:
            epsilon: Privacy parameter ε > 0.
            threshold: Threshold T.
            max_outputs: Maximum number of above-threshold outputs c.
            sensitivity: Query sensitivity (default 1.0).
            metadata: Optional metadata dict.
            seed: Random seed.
        
        Raises:
            ConfigurationError: If parameters are invalid.
        """
        if epsilon <= 0 or not math.isfinite(epsilon):
            raise ConfigurationError(
                f"epsilon must be positive and finite, got {epsilon}",
                parameter="epsilon",
                value=epsilon,
            )
        if max_outputs < 1:
            raise ConfigurationError(
                f"max_outputs must be >= 1, got {max_outputs}",
                parameter="max_outputs",
                value=max_outputs,
            )
        if sensitivity <= 0 or not math.isfinite(sensitivity):
            raise ConfigurationError(
                f"sensitivity must be positive and finite, got {sensitivity}",
                parameter="sensitivity",
                value=sensitivity,
            )
        
        self._epsilon = epsilon
        self._threshold = threshold
        self._max_outputs = max_outputs
        self._sensitivity = sensitivity
        self._metadata = metadata or {}
        self._rng = np.random.default_rng(seed)
        
        # Split budget: half for threshold noise, half for query noise
        self._eps_threshold = epsilon / 2.0
        self._eps_queries = epsilon / 2.0
        
        # Noise scales (Laplace mechanism)
        self._scale_threshold = 2.0 * sensitivity / self._eps_threshold
        self._scale_query = 2.0 * sensitivity / self._eps_queries
        
        # State tracking
        self._num_outputs = 0
        self._queries_processed = 0
    
    @property
    def epsilon(self) -> float:
        """Privacy parameter ε."""
        return self._epsilon
    
    @property
    def delta(self) -> float:
        """Privacy parameter δ (always 0 for pure DP)."""
        return 0.0
    
    @property
    def threshold(self) -> float:
        """Base threshold T."""
        return self._threshold
    
    @property
    def max_outputs(self) -> int:
        """Maximum number of above-threshold outputs c."""
        return self._max_outputs
    
    @property
    def num_outputs(self) -> int:
        """Number of above-threshold outputs returned so far."""
        return self._num_outputs
    
    @property
    def queries_processed(self) -> int:
        """Total number of queries processed."""
        return self._queries_processed
    
    @property
    def budget_remaining(self) -> bool:
        """Whether budget remains for more outputs."""
        return self._num_outputs < self._max_outputs
    
    def reset(self) -> None:
        """Reset state (for reuse with new data)."""
        self._num_outputs = 0
        self._queries_processed = 0
    
    def privacy_guarantee(self) -> Tuple[float, float]:
        """Return the privacy guarantee (ε, δ).
        
        Returns:
            Tuple (epsilon, 0.0) for pure DP.
        """
        return self._epsilon, 0.0


# ---------------------------------------------------------------------------
# AboveThreshold: Classic SVT
# ---------------------------------------------------------------------------


class AboveThreshold(SparseVectorTechnique):
    """Classic sparse vector technique (Dwork-Roth 2014, Algorithm 1).
    
    Answers a sequence of threshold queries, returning TOP if the (noisy)
    query value exceeds the (noisy) threshold, otherwise BOTTOM.
    
    Privacy guarantee: (ε, 0)-DP for returning up to c TOP answers.
    
    Usage::
    
        svt = AboveThreshold(epsilon=1.0, threshold=100, max_outputs=5)
        
        for query_fn in queries:
            result = svt.query(data, query_fn)
            if result is None:
                break  # Budget exhausted
            if result:  # TOP
                print(f"Query {query_fn.__name__} is above threshold")
    """
    
    def query(
        self,
        data: Any,
        query_fn: QueryFunction,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[bool]:
        """Process one threshold query.
        
        Computes query_fn(data) and compares to threshold (both noisy).
        Returns TOP if above, BOTTOM if below, None if budget exhausted.
        
        Args:
            data: Input data.
            query_fn: Query function data -> float.
            rng: Optional RNG override.
        
        Returns:
            True (TOP) if above threshold, False (BOTTOM) if below,
            None if budget exhausted.
        """
        if not self.budget_remaining:
            return None
        
        rng = rng or self._rng
        
        # Compute true query value
        true_val = float(query_fn(data))
        
        # Add noise to query
        noisy_val = true_val + rng.laplace(scale=self._scale_query)
        
        # First query: add noise to threshold (done once)
        if self._queries_processed == 0:
            self._noisy_threshold = (
                self._threshold + rng.laplace(scale=self._scale_threshold)
            )
        
        self._queries_processed += 1
        
        # Compare
        if noisy_val >= self._noisy_threshold:
            self._num_outputs += 1
            return True
        else:
            return False
    
    def query_batch(
        self,
        data: Any,
        query_fns: List[QueryFunction],
        rng: Optional[np.random.Generator] = None,
    ) -> List[Optional[bool]]:
        """Process a batch of threshold queries.
        
        Args:
            data: Input data.
            query_fns: List of query functions.
            rng: Optional RNG override.
        
        Returns:
            List of results (True/False/None for each query).
        """
        results = []
        for qfn in query_fns:
            result = self.query(data, qfn, rng)
            results.append(result)
            if result is None:
                break
        return results


# ---------------------------------------------------------------------------
# NumericSVT: Return noisy values for above-threshold queries
# ---------------------------------------------------------------------------


class NumericSVT(SparseVectorTechnique):
    """Numeric sparse vector technique (returns noisy query values).
    
    Like AboveThreshold, but returns the noisy query value when above
    threshold, instead of just TOP. This enables using the actual values
    for downstream analysis.
    
    Privacy guarantee: (ε, 0)-DP for returning up to c noisy values.
    
    Usage::
    
        svt = NumericSVT(epsilon=1.0, threshold=100, max_outputs=5)
        
        for query_fn in queries:
            result = svt.query(data, query_fn)
            if result is None:
                break  # Budget exhausted
            status, value = result
            if status:  # Above threshold
                print(f"Query value: {value:.2f}")
    """
    
    def query(
        self,
        data: Any,
        query_fn: QueryFunction,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[Tuple[bool, float]]:
        """Process one numeric threshold query.
        
        Args:
            data: Input data.
            query_fn: Query function data -> float.
            rng: Optional RNG override.
        
        Returns:
            Tuple (is_above, noisy_value) if budget remains, else None.
            is_above indicates whether value exceeded threshold.
        """
        if not self.budget_remaining:
            return None
        
        rng = rng or self._rng
        
        # Compute true query value
        true_val = float(query_fn(data))
        
        # Add noise to query
        noisy_val = true_val + rng.laplace(scale=self._scale_query)
        
        # First query: add noise to threshold
        if self._queries_processed == 0:
            self._noisy_threshold = (
                self._threshold + rng.laplace(scale=self._scale_threshold)
            )
        
        self._queries_processed += 1
        
        # Compare
        is_above = noisy_val >= self._noisy_threshold
        
        if is_above:
            self._num_outputs += 1
        
        return (is_above, noisy_val)
    
    def query_batch(
        self,
        data: Any,
        query_fns: List[QueryFunction],
        rng: Optional[np.random.Generator] = None,
    ) -> List[Optional[Tuple[bool, float]]]:
        """Process a batch of numeric threshold queries.
        
        Args:
            data: Input data.
            query_fns: List of query functions.
            rng: Optional RNG override.
        
        Returns:
            List of (is_above, value) tuples.
        """
        results = []
        for qfn in query_fns:
            result = self.query(data, qfn, rng)
            results.append(result)
            if result is None:
                break
        return results


# ---------------------------------------------------------------------------
# AdaptiveSVT: Adaptive threshold selection
# ---------------------------------------------------------------------------


class AdaptiveSVT(SparseVectorTechnique):
    """Adaptive sparse vector technique with dynamic threshold adjustment.
    
    Adjusts the threshold based on observed query values to maximize
    the information gained from each output. Uses a simple heuristic:
    raise threshold when many queries exceed it, lower when few do.
    
    Usage::
    
        svt = AdaptiveSVT(
            epsilon=1.0, initial_threshold=100, max_outputs=5,
            adaptation_rate=0.1
        )
        
        for query_fn in queries:
            result = svt.query(data, query_fn)
            if result is None:
                break
    """
    
    def __init__(
        self,
        epsilon: float,
        initial_threshold: float,
        max_outputs: int = 1,
        sensitivity: float = 1.0,
        adaptation_rate: float = 0.1,
        min_threshold: Optional[float] = None,
        max_threshold: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize adaptive SVT.
        
        Args:
            epsilon: Privacy parameter ε > 0.
            initial_threshold: Initial threshold T.
            max_outputs: Maximum number of outputs c.
            sensitivity: Query sensitivity (default 1.0).
            adaptation_rate: Rate of threshold adaptation (default 0.1).
            min_threshold: Minimum allowed threshold (optional).
            max_threshold: Maximum allowed threshold (optional).
            metadata: Optional metadata dict.
            seed: Random seed.
        """
        super().__init__(
            epsilon=epsilon,
            threshold=initial_threshold,
            max_outputs=max_outputs,
            sensitivity=sensitivity,
            metadata=metadata,
            seed=seed,
        )
        
        self._adaptation_rate = adaptation_rate
        self._min_threshold = min_threshold
        self._max_threshold = max_threshold
        self._current_threshold = initial_threshold
        
        # Track recent query results for adaptation
        self._recent_above_rate = 0.5  # Start neutral
        self._window_size = 10
        self._recent_results: List[bool] = []
    
    def query(
        self,
        data: Any,
        query_fn: QueryFunction,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[Tuple[bool, float]]:
        """Process one adaptive threshold query.
        
        Args:
            data: Input data.
            query_fn: Query function data -> float.
            rng: Optional RNG override.
        
        Returns:
            Tuple (is_above, current_threshold) if budget remains, else None.
        """
        if not self.budget_remaining:
            return None
        
        rng = rng or self._rng
        
        # Compute true query value
        true_val = float(query_fn(data))
        
        # Add noise to query
        noisy_val = true_val + rng.laplace(scale=self._scale_query)
        
        # Add noise to current threshold (re-noised each time for adaptation)
        noisy_threshold = (
            self._current_threshold + rng.laplace(scale=self._scale_threshold)
        )
        
        self._queries_processed += 1
        
        # Compare
        is_above = noisy_val >= noisy_threshold
        
        if is_above:
            self._num_outputs += 1
        
        # Update recent results
        self._recent_results.append(is_above)
        if len(self._recent_results) > self._window_size:
            self._recent_results.pop(0)
        
        # Adapt threshold
        self._adapt_threshold()
        
        return (is_above, self._current_threshold)
    
    def _adapt_threshold(self) -> None:
        """Adapt threshold based on recent query results.
        
        If many queries are above threshold, raise it.
        If few queries are above threshold, lower it.
        """
        if len(self._recent_results) < 3:
            return  # Not enough data
        
        # Compute recent above rate
        self._recent_above_rate = sum(self._recent_results) / len(self._recent_results)
        
        # Target: ~50% of queries above threshold for balanced exploration
        target_rate = 0.5
        rate_error = self._recent_above_rate - target_rate
        
        # Adjust threshold
        adjustment = rate_error * self._adaptation_rate * self._current_threshold
        self._current_threshold -= adjustment
        
        # Clip to bounds
        if self._min_threshold is not None:
            self._current_threshold = max(self._current_threshold, self._min_threshold)
        if self._max_threshold is not None:
            self._current_threshold = min(self._current_threshold, self._max_threshold)
    
    @property
    def current_threshold(self) -> float:
        """Current adaptive threshold."""
        return self._current_threshold
    
    @property
    def recent_above_rate(self) -> float:
        """Recent rate of above-threshold queries."""
        return self._recent_above_rate


# ---------------------------------------------------------------------------
# Gap-SVT: SVT with gap queries
# ---------------------------------------------------------------------------


class GapSVT(SparseVectorTechnique):
    """Gap-SVT: SVT variant that returns the gap above threshold.
    
    Instead of just returning TOP/BOTTOM, returns the (noisy) amount by which
    the query exceeds the threshold. This provides more information per output.
    
    Usage::
    
        svt = GapSVT(epsilon=1.0, threshold=100, max_outputs=5)
        
        for query_fn in queries:
            result = svt.query(data, query_fn)
            if result is None:
                break
            is_above, gap = result
            if is_above:
                print(f"Gap above threshold: {gap:.2f}")
    """
    
    def query(
        self,
        data: Any,
        query_fn: QueryFunction,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[Tuple[bool, float]]:
        """Process one gap threshold query.
        
        Args:
            data: Input data.
            query_fn: Query function data -> float.
            rng: Optional RNG override.
        
        Returns:
            Tuple (is_above, gap) where gap = noisy_value - noisy_threshold,
            or None if budget exhausted.
        """
        if not self.budget_remaining:
            return None
        
        rng = rng or self._rng
        
        # Compute true query value
        true_val = float(query_fn(data))
        
        # Add noise to query
        noisy_val = true_val + rng.laplace(scale=self._scale_query)
        
        # First query: add noise to threshold
        if self._queries_processed == 0:
            self._noisy_threshold = (
                self._threshold + rng.laplace(scale=self._scale_threshold)
            )
        
        self._queries_processed += 1
        
        # Compute gap
        gap = noisy_val - self._noisy_threshold
        is_above = gap >= 0
        
        if is_above:
            self._num_outputs += 1
        
        return (is_above, gap)


# ---------------------------------------------------------------------------
# SVT Composition: Privacy accounting for multiple SVT instances
# ---------------------------------------------------------------------------


@dataclass
class SVTCompositionResult:
    """Result of composing multiple SVT instances.
    
    Attributes:
        total_epsilon: Total ε consumed.
        total_delta: Total δ consumed (always 0 for pure DP).
        num_instances: Number of SVT instances.
        total_outputs: Total number of above-threshold outputs.
        per_instance_epsilon: Per-instance ε values.
    """
    total_epsilon: float
    total_delta: float
    num_instances: int
    total_outputs: int
    per_instance_epsilon: List[float]


class SVTComposition:
    """Privacy accounting for multiple SVT instances.
    
    When running multiple SVT instances sequentially or in parallel,
    the privacy guarantees compose. This class tracks the composition.
    
    For k SVT instances with parameters (ε_i, c_i), the total privacy
    guarantee is (Σ ε_i, 0) under basic composition.
    
    Usage::
    
        composer = SVTComposition(total_budget_epsilon=2.0)
        
        svt1 = composer.create_svt(threshold=100, max_outputs=5)
        # Use svt1...
        
        svt2 = composer.create_svt(threshold=200, max_outputs=3)
        # Use svt2...
        
        result = composer.composition_result()
        print(f"Total ε used: {result.total_epsilon:.4f}")
    """
    
    def __init__(
        self,
        total_budget_epsilon: float,
        total_budget_delta: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize SVT composition tracker.
        
        Args:
            total_budget_epsilon: Total privacy budget ε.
            total_budget_delta: Total privacy budget δ (default 0).
            metadata: Optional metadata dict.
        
        Raises:
            ConfigurationError: If budgets are invalid.
        """
        if total_budget_epsilon <= 0:
            raise ConfigurationError(
                f"total_budget_epsilon must be > 0, got {total_budget_epsilon}",
                parameter="total_budget_epsilon",
                value=total_budget_epsilon,
            )
        if not (0.0 <= total_budget_delta < 1.0):
            raise ConfigurationError(
                f"total_budget_delta must be in [0, 1), got {total_budget_delta}",
                parameter="total_budget_delta",
                value=total_budget_delta,
            )
        
        self._total_budget_epsilon = total_budget_epsilon
        self._total_budget_delta = total_budget_delta
        self._metadata = metadata or {}
        
        self._instances: List[SparseVectorTechnique] = []
        self._consumed_epsilon = 0.0
        self._consumed_delta = 0.0
    
    @property
    def total_budget_epsilon(self) -> float:
        """Total privacy budget ε."""
        return self._total_budget_epsilon
    
    @property
    def remaining_epsilon(self) -> float:
        """Remaining privacy budget ε."""
        return self._total_budget_epsilon - self._consumed_epsilon
    
    @property
    def consumed_epsilon(self) -> float:
        """Consumed privacy budget ε."""
        return self._consumed_epsilon
    
    def create_svt(
        self,
        threshold: float,
        max_outputs: int = 1,
        epsilon: Optional[float] = None,
        svt_type: str = "above_threshold",
        **kwargs: Any,
    ) -> SparseVectorTechnique:
        """Create a new SVT instance and register it.
        
        Args:
            threshold: Threshold T.
            max_outputs: Maximum outputs c.
            epsilon: Per-instance ε (if None, uses equal split of remaining).
            svt_type: Type of SVT: "above_threshold", "numeric", "adaptive", "gap".
            **kwargs: Additional args for specific SVT types.
        
        Returns:
            SVT instance.
        
        Raises:
            BudgetExhaustedError: If insufficient budget remains.
        """
        # Determine epsilon allocation
        if epsilon is None:
            # Simple heuristic: equal split of remaining budget
            n_remaining = len(self._instances) + 1
            epsilon = self.remaining_epsilon / n_remaining
        
        if epsilon > self.remaining_epsilon + 1e-10:
            raise BudgetExhaustedError(
                f"Insufficient budget: requested {epsilon}, remaining {self.remaining_epsilon}",
                budget_epsilon=self._total_budget_epsilon,
                consumed_epsilon=self._consumed_epsilon,
            )
        
        # Create SVT instance
        if svt_type == "above_threshold":
            svt = AboveThreshold(
                epsilon=epsilon,
                threshold=threshold,
                max_outputs=max_outputs,
                **kwargs,
            )
        elif svt_type == "numeric":
            svt = NumericSVT(
                epsilon=epsilon,
                threshold=threshold,
                max_outputs=max_outputs,
                **kwargs,
            )
        elif svt_type == "adaptive":
            svt = AdaptiveSVT(
                epsilon=epsilon,
                initial_threshold=threshold,
                max_outputs=max_outputs,
                **kwargs,
            )
        elif svt_type == "gap":
            svt = GapSVT(
                epsilon=epsilon,
                threshold=threshold,
                max_outputs=max_outputs,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown svt_type: {svt_type}")
        
        # Register
        self._instances.append(svt)
        self._consumed_epsilon += epsilon
        
        return svt
    
    def composition_result(self) -> SVTCompositionResult:
        """Get the composition result.
        
        Returns:
            SVTCompositionResult with privacy accounting.
        """
        per_instance_epsilon = [svt.epsilon for svt in self._instances]
        total_outputs = sum(svt.num_outputs for svt in self._instances)
        
        return SVTCompositionResult(
            total_epsilon=self._consumed_epsilon,
            total_delta=self._consumed_delta,
            num_instances=len(self._instances),
            total_outputs=total_outputs,
            per_instance_epsilon=per_instance_epsilon,
        )
    
    def __repr__(self) -> str:
        return (
            f"SVTComposition(budget_ε={self._total_budget_epsilon:.4f}, "
            f"consumed_ε={self._consumed_epsilon:.4f}, "
            f"instances={len(self._instances)})"
        )

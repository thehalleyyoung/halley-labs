"""
Rollout evaluation scheduler for MCTS.

Manages rollout evaluation with memoized inference, cache-aware batch
scheduling, configurable rollout policies, and detailed statistics.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# -----------------------------------------------------------------------
# Rollout policies
# -----------------------------------------------------------------------

class RolloutPolicy(Enum):
    """Available rollout policies."""

    RANDOM = "random"
    HEURISTIC = "heuristic"
    LEARNED = "learned"
    GREEDY = "greedy"
    EPSILON_GREEDY = "epsilon_greedy"


@dataclass
class RolloutResult:
    """Result from a single rollout evaluation."""

    value: float
    state: Dict[str, float]
    cached: bool = False
    wall_time_seconds: float = 0.0
    inference_calls: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    """Result from a batch of rollout evaluations."""

    results: List[RolloutResult]
    total_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def mean_value(self) -> float:
        if not self.results:
            return 0.0
        return float(np.mean([r.value for r in self.results]))

    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


# -----------------------------------------------------------------------
# LRU cache for inference results
# -----------------------------------------------------------------------

class _LRUCache:
    """Simple LRU cache with max size."""

    def __init__(self, max_size: int = 10000) -> None:
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
        self._cache[key] = value

    def contains(self, key: str) -> bool:
        return key in self._cache

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0


# -----------------------------------------------------------------------
# RolloutScheduler
# -----------------------------------------------------------------------

class RolloutScheduler:
    """
    Manage rollout evaluation with memoized inference.

    Integrates with a junction-tree inference engine to evaluate scenarios.
    Provides caching, batching, configurable rollout policies, and
    performance statistics.

    Parameters
    ----------
    cache_size : int
        Maximum number of inference results to cache.
    default_policy : RolloutPolicy
        Default rollout policy for completing partial assignments.
    epsilon : float
        Epsilon for epsilon-greedy policy.
    random_seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        cache_size: int = 10000,
        default_policy: RolloutPolicy = RolloutPolicy.RANDOM,
        epsilon: float = 0.1,
        random_seed: Optional[int] = None,
    ) -> None:
        self._cache = _LRUCache(max_size=cache_size)
        self._policy = default_policy
        self._epsilon = epsilon
        self._rng = np.random.RandomState(random_seed)

        # Custom heuristic/learned policy functions
        self._heuristic_fn: Optional[Callable] = None
        self._learned_fn: Optional[Callable] = None

        # Statistics
        self._total_rollouts: int = 0
        self._total_time: float = 0.0
        self._rollout_times: List[float] = []
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._inference_calls: int = 0
        self._batch_count: int = 0

        # Variable metadata for random completion
        self._variable_domains: Dict[str, Tuple[float, float]] = {}
        self._variable_discretization: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_rollout_policy(self, policy: RolloutPolicy) -> None:
        """
        Set the rollout policy.

        Parameters
        ----------
        policy : RolloutPolicy
            Policy to use for completing partial assignments.
        """
        self._policy = policy

    def set_heuristic_function(
        self, fn: Callable[[Dict[str, float], List[str]], Dict[str, float]]
    ) -> None:
        """
        Register a heuristic function for the HEURISTIC rollout policy.

        The function receives (partial_assignment, remaining_variables)
        and returns a complete assignment.
        """
        self._heuristic_fn = fn

    def set_learned_function(
        self, fn: Callable[[Dict[str, float]], float]
    ) -> None:
        """
        Register a learned value function for the LEARNED rollout policy.

        The function receives a state and returns an estimated value
        without needing to run inference.
        """
        self._learned_fn = fn

    def set_variable_domains(
        self, domains: Dict[str, Tuple[float, float]]
    ) -> None:
        """
        Set the domain (min, max) for each variable.

        Used by the random rollout policy to sample completions.
        """
        self._variable_domains = dict(domains)

    def set_variable_discretization(
        self, discretization: Dict[str, List[float]]
    ) -> None:
        """
        Set discrete values for each variable.

        Used by rollout policies that enumerate over discrete values.
        """
        self._variable_discretization = dict(discretization)

    # ------------------------------------------------------------------
    # State hashing
    # ------------------------------------------------------------------

    @staticmethod
    def _state_hash(state: Dict[str, float]) -> str:
        """Compute a deterministic hash of a state dictionary."""
        items = sorted(state.items())
        raw = "|".join(f"{k}={v:.10g}" for k, v in items)
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    # ------------------------------------------------------------------
    # Single rollout evaluation
    # ------------------------------------------------------------------

    def evaluate_rollout(
        self,
        state: Dict[str, float],
        inference_engine: Any,
        target_variable: Optional[str] = None,
        remaining_variables: Optional[List[str]] = None,
    ) -> RolloutResult:
        """
        Evaluate a single rollout from the given state.

        1. Complete the partial assignment using the rollout policy.
        2. Check the cache for the completed state.
        3. If cache miss, invoke the inference engine.
        4. Return the result.

        Parameters
        ----------
        state : dict
            Partial shock assignment (variable -> value).
        inference_engine : object
            Must implement ``query(evidence, target) -> float``.
        target_variable : str or None
            Target loss variable for inference.
        remaining_variables : list of str or None
            Variables not yet assigned (for rollout completion).

        Returns
        -------
        RolloutResult
        """
        t0 = time.time()

        # Step 1: Complete the assignment
        complete_state = self._complete_assignment(state, remaining_variables)

        # Step 2: Check cache
        state_key = self._state_hash(complete_state)
        cached_value = self._cache.get(state_key)

        if cached_value is not None:
            self._cache_hits += 1
            elapsed = time.time() - t0
            self._record_rollout(elapsed)
            return RolloutResult(
                value=cached_value,
                state=complete_state,
                cached=True,
                wall_time_seconds=elapsed,
                inference_calls=0,
            )

        # Step 3: Evaluate via inference engine
        self._cache_misses += 1

        if self._policy == RolloutPolicy.LEARNED and self._learned_fn is not None:
            value = self._learned_fn(complete_state)
            n_calls = 0
        else:
            value = self._run_inference(
                complete_state, inference_engine, target_variable
            )
            n_calls = 1

        # Step 4: Cache and return
        self._cache.put(state_key, value)

        elapsed = time.time() - t0
        self._record_rollout(elapsed)

        return RolloutResult(
            value=value,
            state=complete_state,
            cached=False,
            wall_time_seconds=elapsed,
            inference_calls=n_calls,
        )

    def _run_inference(
        self,
        state: Dict[str, float],
        inference_engine: Any,
        target_variable: Optional[str] = None,
    ) -> float:
        """
        Run the inference engine on a complete state.

        Handles multiple inference engine interfaces:
        - query(evidence, target) -> float
        - evaluate(state) -> float
        - __call__(state) -> float
        """
        self._inference_calls += 1

        if hasattr(inference_engine, "query") and target_variable is not None:
            return float(inference_engine.query(state, target_variable))
        elif hasattr(inference_engine, "evaluate"):
            return float(inference_engine.evaluate(state))
        elif callable(inference_engine):
            return float(inference_engine(state))
        else:
            raise TypeError(
                f"Inference engine of type {type(inference_engine)} "
                "does not support query(), evaluate(), or __call__()"
            )

    # ------------------------------------------------------------------
    # Assignment completion
    # ------------------------------------------------------------------

    def _complete_assignment(
        self,
        partial: Dict[str, float],
        remaining_variables: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Complete a partial assignment using the current rollout policy.

        Parameters
        ----------
        partial : dict
            Current partial assignment.
        remaining_variables : list of str or None
            Variables that need values.

        Returns
        -------
        dict
            Complete assignment.
        """
        if remaining_variables is None or len(remaining_variables) == 0:
            return dict(partial)

        remaining = [v for v in remaining_variables if v not in partial]

        if not remaining:
            return dict(partial)

        complete = dict(partial)

        if self._policy == RolloutPolicy.RANDOM:
            complete.update(self._random_completion(remaining))
        elif self._policy == RolloutPolicy.HEURISTIC:
            complete.update(self._heuristic_completion(partial, remaining))
        elif self._policy == RolloutPolicy.GREEDY:
            complete.update(self._greedy_completion(remaining))
        elif self._policy == RolloutPolicy.EPSILON_GREEDY:
            complete.update(self._epsilon_greedy_completion(partial, remaining))
        elif self._policy == RolloutPolicy.LEARNED:
            # Learned policy doesn't need completion; pass partial directly
            pass
        else:
            complete.update(self._random_completion(remaining))

        return complete

    def _random_completion(
        self, remaining: List[str]
    ) -> Dict[str, float]:
        """Sample values uniformly from variable domains."""
        completion: Dict[str, float] = {}
        for var in remaining:
            if var in self._variable_discretization:
                values = self._variable_discretization[var]
                idx = self._rng.randint(len(values))
                completion[var] = values[idx]
            elif var in self._variable_domains:
                lo, hi = self._variable_domains[var]
                completion[var] = float(self._rng.uniform(lo, hi))
            else:
                completion[var] = float(self._rng.standard_normal())
        return completion

    def _heuristic_completion(
        self,
        partial: Dict[str, float],
        remaining: List[str],
    ) -> Dict[str, float]:
        """Use the registered heuristic function."""
        if self._heuristic_fn is not None:
            result = self._heuristic_fn(partial, remaining)
            return {k: v for k, v in result.items() if k in remaining}
        return self._random_completion(remaining)

    def _greedy_completion(
        self, remaining: List[str]
    ) -> Dict[str, float]:
        """
        Greedy completion: pick extreme values from variable domains.

        For worst-case search, pick the extreme value most likely to
        cause high loss.
        """
        completion: Dict[str, float] = {}
        for var in remaining:
            if var in self._variable_discretization:
                values = self._variable_discretization[var]
                # Pick the maximum value (greedy worst-case)
                completion[var] = max(values)
            elif var in self._variable_domains:
                lo, hi = self._variable_domains[var]
                completion[var] = hi  # worst-case: max value
            else:
                completion[var] = 3.0  # high shock
        return completion

    def _epsilon_greedy_completion(
        self,
        partial: Dict[str, float],
        remaining: List[str],
    ) -> Dict[str, float]:
        """
        Epsilon-greedy: with probability epsilon use random, else greedy.
        """
        if self._rng.random() < self._epsilon:
            return self._random_completion(remaining)
        else:
            return self._greedy_completion(remaining)

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    def schedule_batch(
        self,
        states: List[Dict[str, float]],
        inference_engine: Any = None,
        target_variable: Optional[str] = None,
        remaining_variables: Optional[List[str]] = None,
    ) -> BatchResult:
        """
        Schedule a batch of rollout evaluations.

        Optimizes by:
        1. Checking cache for all states first.
        2. Batching cache misses for inference.
        3. Updating cache with new results.

        Parameters
        ----------
        states : list of dict
            Partial assignments to evaluate.
        inference_engine : object or None
            Inference engine.
        target_variable : str or None
            Target variable.
        remaining_variables : list of str or None
            Variables for completion.

        Returns
        -------
        BatchResult
        """
        t0 = time.time()
        self._batch_count += 1

        results: List[RolloutResult] = []
        cache_hits = 0
        cache_misses = 0

        # Phase 1: complete all assignments and partition into cached/uncached
        completed_states = []
        state_keys = []

        for state in states:
            complete = self._complete_assignment(state, remaining_variables)
            completed_states.append(complete)
            state_keys.append(self._state_hash(complete))

        # Phase 2: resolve from cache
        uncached_indices = []
        for i, key in enumerate(state_keys):
            cached_value = self._cache.get(key)
            if cached_value is not None:
                cache_hits += 1
                results.append(
                    RolloutResult(
                        value=cached_value,
                        state=completed_states[i],
                        cached=True,
                    )
                )
            else:
                cache_misses += 1
                results.append(None)  # type: ignore[arg-type]
                uncached_indices.append(i)

        # Phase 3: batch inference for uncached
        if uncached_indices and inference_engine is not None:
            # Try batch inference if available
            if hasattr(inference_engine, "batch_query"):
                batch_states = [completed_states[i] for i in uncached_indices]
                batch_values = inference_engine.batch_query(
                    batch_states, target_variable
                )
                self._inference_calls += len(uncached_indices)

                for idx, val in zip(uncached_indices, batch_values):
                    value = float(val)
                    self._cache.put(state_keys[idx], value)
                    results[idx] = RolloutResult(
                        value=value,
                        state=completed_states[idx],
                        cached=False,
                        inference_calls=1,
                    )
            else:
                # Fall back to sequential evaluation
                for idx in uncached_indices:
                    value = self._run_inference(
                        completed_states[idx],
                        inference_engine,
                        target_variable,
                    )
                    self._cache.put(state_keys[idx], value)
                    results[idx] = RolloutResult(
                        value=value,
                        state=completed_states[idx],
                        cached=False,
                        inference_calls=1,
                    )

        total_time = time.time() - t0
        self._total_rollouts += len(states)
        self._total_time += total_time
        self._cache_hits += cache_hits
        self._cache_misses += cache_misses

        return BatchResult(
            results=[r for r in results if r is not None],
            total_time=total_time,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
        )

    # ------------------------------------------------------------------
    # Cost estimation
    # ------------------------------------------------------------------

    def estimate_rollout_cost(
        self,
        state: Dict[str, float],
        remaining_variables: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Estimate the cost (time, cache status) of evaluating a rollout.

        Parameters
        ----------
        state : dict
            Partial assignment.
        remaining_variables : list of str or None
            Variables for completion.

        Returns
        -------
        dict
            Contains 'estimated_time', 'in_cache', 'n_remaining_vars',
            'estimated_inference_cost'.
        """
        complete = self._complete_assignment(state, remaining_variables)
        key = self._state_hash(complete)
        in_cache = self._cache.contains(key)

        avg_time = (
            self._total_time / self._total_rollouts
            if self._total_rollouts > 0
            else 0.01
        )

        n_remaining = (
            len(remaining_variables) if remaining_variables else 0
        )

        # Cached lookups are ~100x faster
        estimated_time = avg_time * (0.01 if in_cache else 1.0)

        return {
            "estimated_time": estimated_time,
            "in_cache": in_cache,
            "n_remaining_vars": n_remaining,
            "estimated_inference_cost": 0.0 if in_cache else avg_time,
            "cache_key": key,
        }

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _record_rollout(self, elapsed: float) -> None:
        """Record timing for a single rollout."""
        self._total_rollouts += 1
        self._total_time += elapsed
        self._rollout_times.append(elapsed)

    def get_memoization_stats(self) -> Dict[str, Any]:
        """
        Return memoization and performance statistics.

        Returns
        -------
        dict
            Cache statistics and rollout timings.
        """
        total_queries = self._cache_hits + self._cache_misses

        time_stats: Dict[str, Any] = {}
        if self._rollout_times:
            times = np.array(self._rollout_times)
            time_stats = {
                "mean_time": float(np.mean(times)),
                "median_time": float(np.median(times)),
                "min_time": float(np.min(times)),
                "max_time": float(np.max(times)),
                "std_time": float(np.std(times)),
                "p95_time": float(np.percentile(times, 95)),
                "p99_time": float(np.percentile(times, 99)),
            }

        return {
            "total_rollouts": self._total_rollouts,
            "total_time": self._total_time,
            "cache_size": self._cache.size,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": self._cache_hits / total_queries if total_queries > 0 else 0.0,
            "inference_calls": self._inference_calls,
            "batch_count": self._batch_count,
            "rollout_timings": time_stats,
        }

    def get_throughput(self) -> float:
        """Return rollouts per second."""
        if self._total_time <= 0:
            return 0.0
        return self._total_rollouts / self._total_time

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def prefill_cache(
        self,
        states: List[Dict[str, float]],
        values: List[float],
    ) -> int:
        """
        Pre-fill the cache with known state-value pairs.

        Parameters
        ----------
        states : list of dict
            States to cache.
        values : list of float
            Corresponding values.

        Returns
        -------
        int
            Number of entries added.
        """
        count = 0
        for state, value in zip(states, values):
            key = self._state_hash(state)
            if not self._cache.contains(key):
                self._cache.put(key, value)
                count += 1
        return count

    def clear_cache(self) -> None:
        """Clear the inference cache."""
        self._cache.clear()

    def reset_stats(self) -> None:
        """Reset all statistics without clearing the cache."""
        self._total_rollouts = 0
        self._total_time = 0.0
        self._rollout_times.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._inference_calls = 0
        self._batch_count = 0

    def reset(self) -> None:
        """Reset everything including the cache."""
        self.clear_cache()
        self.reset_stats()

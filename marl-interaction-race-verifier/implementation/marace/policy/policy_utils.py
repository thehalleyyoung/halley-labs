"""
Policy utilities for MARACE.

Provides caching, normalization, simple policy implementations, and
comparison / sampling helpers used throughout the MARACE policy
verification pipeline.  All numerical operations are backed by NumPy
for efficient batched evaluation.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ======================================================================
# PolicyCache
# ======================================================================

class PolicyCache:
    """LRU cache for policy evaluations keyed by state hash.

    Avoids redundant forward passes when the same (or very similar)
    states are queried repeatedly during abstract-domain analysis.

    Parameters:
        max_size: Maximum number of entries before eviction.
    """

    def __init__(self, max_size: int = 10000) -> None:
        self._max_size = max_size
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, state: np.ndarray) -> Optional[np.ndarray]:
        """Look up a cached action for *state*, or return ``None``."""
        key = self._hash_state(state)
        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            return self._cache[key].copy()
        self._misses += 1
        return None

    def put(self, state: np.ndarray, action: np.ndarray) -> None:
        """Store *action* for the given *state*."""
        key = self._hash_state(state)
        self._cache[key] = action.copy()
        self._cache.move_to_end(key)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        """Flush the cache and reset statistics."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups served from cache."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        return len(self._cache)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _hash_state(self, state: np.ndarray) -> int:
        """Fast, deterministic hash of a state array."""
        return hash(state.tobytes())


# ======================================================================
# NormalizationWrapper
# ======================================================================

class NormalizationWrapper:
    """Handle observation / action normalization and denormalization.

    Any statistic that is ``None`` is treated as identity (no-op).
    Observations are clipped to ``[-clip_obs, clip_obs]`` after
    normalization; actions are clipped similarly.

    Parameters:
        obs_mean:  Mean of observations (or ``None`` for identity).
        obs_std:   Std-dev of observations (or ``None`` for identity).
        act_mean:  Mean of actions (or ``None`` for identity).
        act_std:   Std-dev of actions (or ``None`` for identity).
        clip_obs:  Clip normalized observations to this absolute value.
        clip_act:  Clip normalized actions to this absolute value.
    """

    def __init__(
        self,
        obs_mean: Optional[np.ndarray] = None,
        obs_std: Optional[np.ndarray] = None,
        act_mean: Optional[np.ndarray] = None,
        act_std: Optional[np.ndarray] = None,
        clip_obs: float = 10.0,
        clip_act: float = float("inf"),
    ) -> None:
        self._obs_mean = obs_mean
        self._obs_std = obs_std
        self._act_mean = act_mean
        self._act_std = act_std
        self._clip_obs = clip_obs
        self._clip_act = clip_act

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize raw observation to zero-mean / unit-variance."""
        out = obs.copy()
        if self._obs_mean is not None:
            out = out - self._obs_mean
        if self._obs_std is not None:
            out = out / np.maximum(self._obs_std, 1e-8)
        return np.clip(out, -self._clip_obs, self._clip_obs)

    def denormalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Invert observation normalization."""
        out = obs.copy()
        if self._obs_std is not None:
            out = out * self._obs_std
        if self._obs_mean is not None:
            out = out + self._obs_mean
        return out

    # ------------------------------------------------------------------
    # Action helpers
    # ------------------------------------------------------------------

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Normalize raw action."""
        out = action.copy()
        if self._act_mean is not None:
            out = out - self._act_mean
        if self._act_std is not None:
            out = out / np.maximum(self._act_std, 1e-8)
        return np.clip(out, -self._clip_act, self._clip_act)

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Invert action normalization."""
        out = action.copy()
        if self._act_std is not None:
            out = out * self._act_std
        if self._act_mean is not None:
            out = out + self._act_mean
        return out

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def identity(cls) -> NormalizationWrapper:
        """Return a no-op wrapper that leaves inputs unchanged."""
        return cls()


# ======================================================================
# DummyPolicy
# ======================================================================

class DummyPolicy:
    """Simple deterministic policy that always returns a fixed output.

    Useful as a stand-in during unit tests or when an agent's policy is
    not yet loaded.

    Parameters:
        output_dim:      Dimensionality of the action space.
        constant_output: Fixed action vector.  Defaults to zeros.
    """

    def __init__(
        self,
        output_dim: int,
        constant_output: Optional[np.ndarray] = None,
    ) -> None:
        self._output_dim = output_dim
        if constant_output is not None:
            self._output = np.asarray(constant_output, dtype=np.float64).ravel()
        else:
            self._output = np.zeros(output_dim, dtype=np.float64)

    def evaluate(self, observation: np.ndarray) -> np.ndarray:
        """Return the constant action regardless of *observation*."""
        return self._output.copy()

    def evaluate_batch(self, observations: np.ndarray) -> np.ndarray:
        """Vectorized evaluate over a batch of observations."""
        n = observations.shape[0]
        return np.tile(self._output, (n, 1))


# ======================================================================
# RandomPolicy
# ======================================================================

class RandomPolicy:
    """Uniformly random policy baseline.

    Parameters:
        output_dim: Dimensionality of the action space.
        low:        Lower bound of the uniform distribution.
        high:       Upper bound of the uniform distribution.
        seed:       Optional random seed for reproducibility.
    """

    def __init__(
        self,
        output_dim: int,
        low: float = -1.0,
        high: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self._output_dim = output_dim
        self._low = low
        self._high = high
        self._rng = np.random.default_rng(seed)

    def evaluate(self, observation: np.ndarray) -> np.ndarray:
        """Sample a random action (observation is ignored)."""
        return self._rng.uniform(self._low, self._high, size=self._output_dim)

    def evaluate_batch(self, observations: np.ndarray) -> np.ndarray:
        """Sample random actions for a batch of observations."""
        n = observations.shape[0]
        return self._rng.uniform(
            self._low, self._high, size=(n, self._output_dim)
        )


# ======================================================================
# LinearPolicy
# ======================================================================

class LinearPolicy:
    """Linear policy ``a = W @ obs + b`` for benchmarking abstract analysis.

    Parameters:
        weight: Weight matrix of shape ``(output_dim, input_dim)``.
        bias:   Optional bias vector of shape ``(output_dim,)``.
    """

    def __init__(
        self,
        weight: np.ndarray,
        bias: Optional[np.ndarray] = None,
    ) -> None:
        self._weight = np.asarray(weight, dtype=np.float64)
        self._bias = (
            np.asarray(bias, dtype=np.float64)
            if bias is not None
            else np.zeros(self._weight.shape[0], dtype=np.float64)
        )

    def evaluate(self, observation: np.ndarray) -> np.ndarray:
        """Compute ``W @ obs + b``."""
        return self._weight @ np.asarray(observation, dtype=np.float64) + self._bias

    def evaluate_batch(self, observations: np.ndarray) -> np.ndarray:
        """Vectorized evaluate: ``(W @ obs^T)^T + b``."""
        obs = np.asarray(observations, dtype=np.float64)
        return (self._weight @ obs.T).T + self._bias

    @classmethod
    def from_random(
        cls,
        input_dim: int,
        output_dim: int,
        seed: Optional[int] = None,
    ) -> LinearPolicy:
        """Create a randomly initialized linear policy."""
        rng = np.random.default_rng(seed)
        weight = rng.standard_normal((output_dim, input_dim))
        bias = rng.standard_normal(output_dim)
        return cls(weight, bias)


# ======================================================================
# PolicyComparator
# ======================================================================

class PolicyComparator:
    """Compare two policies' behaviour over a bounded state region.

    Both *policy_a* and *policy_b* must be callables accepting a single
    ``np.ndarray`` observation and returning an ``np.ndarray`` action.

    Parameters:
        policy_a: First policy callable.
        policy_b: Second policy callable.
    """

    def __init__(
        self,
        policy_a: Callable[[np.ndarray], np.ndarray],
        policy_b: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        self._policy_a = policy_a
        self._policy_b = policy_b

    def max_difference(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        n_samples: int = 1000,
    ) -> float:
        """Estimate the maximum L-inf action difference via sampling."""
        states = self._sample_uniform(lower, upper, n_samples)
        diffs = self._compute_diffs(states)
        return float(np.max(diffs))

    def mean_difference(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        n_samples: int = 1000,
    ) -> float:
        """Estimate the mean L-inf action difference via sampling."""
        states = self._sample_uniform(lower, upper, n_samples)
        diffs = self._compute_diffs(states)
        return float(np.mean(diffs))

    def agreement_ratio(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        threshold: float = 0.01,
        n_samples: int = 1000,
    ) -> float:
        """Fraction of sampled states where policies agree within *threshold*."""
        states = self._sample_uniform(lower, upper, n_samples)
        diffs = self._compute_diffs(states)
        return float(np.mean(diffs <= threshold))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_uniform(
        lower: np.ndarray, upper: np.ndarray, n: int
    ) -> np.ndarray:
        return np.random.uniform(lower, upper, size=(n, len(lower)))

    def _compute_diffs(self, states: np.ndarray) -> np.ndarray:
        """Per-sample L-inf difference between the two policies."""
        actions_a = np.array([self._policy_a(s) for s in states])
        actions_b = np.array([self._policy_b(s) for s in states])
        return np.max(np.abs(actions_a - actions_b), axis=1)


# ======================================================================
# PolicySampler
# ======================================================================

class PolicySampler:
    """Sample state-action pairs from a policy over various distributions.

    Parameters:
        policy_fn: Callable that maps a single observation to an action.
        input_dim: Dimensionality of the observation space.
    """

    def __init__(
        self,
        policy_fn: Callable[[np.ndarray], np.ndarray],
        input_dim: int,
    ) -> None:
        self._policy_fn = policy_fn
        self._input_dim = input_dim

    def sample_uniform(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        n_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample states uniformly from ``[lower, upper]`` and evaluate."""
        states = np.random.uniform(lower, upper, size=(n_samples, self._input_dim))
        actions = np.array([self._policy_fn(s) for s in states])
        return states, actions

    def sample_gaussian(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        n_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample states from a Gaussian and evaluate."""
        states = np.random.normal(
            loc=mean, scale=std, size=(n_samples, self._input_dim)
        )
        actions = np.array([self._policy_fn(s) for s in states])
        return states, actions

    def sample_grid(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        points_per_dim: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the policy on a regular grid over ``[lower, upper]``.

        Note: total samples grow as ``points_per_dim ** input_dim``, so
        this is only practical for low-dimensional inputs.
        """
        axes = [
            np.linspace(lower[i], upper[i], points_per_dim)
            for i in range(self._input_dim)
        ]
        grids = np.meshgrid(*axes, indexing="ij")
        states = np.column_stack([g.ravel() for g in grids])
        actions = np.array([self._policy_fn(s) for s in states])
        return states, actions

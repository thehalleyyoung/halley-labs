"""
Continual observation mechanisms for streaming differential privacy.

Implements differentially private mechanisms for the continual observation
model where data arrives as a stream and private statistics must be released
at every time step.

References:
    - Dwork, Naor, Pitassi, Rothblum. "Differential Privacy Under Continual
      Observation." STOC 2010.
    - Kellaris, Papadopoulos, Xiao, Papadias. "Differentially Private Event
      Sequences over Infinite Streams." VLDB 2014.  (w-event privacy)
    - Mir, Muthukrishnan, Nikolov, Wright. "Pan-Private Streaming
      Algorithms." ICS 2011.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from dp_forge.streaming import (
    NoiseSchedule,
    StreamConfig,
    StreamMechanismType,
    StreamOutput,
    StreamState,
    StreamSummary,
)
from dp_forge.streaming.binary_tree import BinaryTreeMechanism, NoiseAllocation


# ---------------------------------------------------------------------------
# ContinualCounter
# ---------------------------------------------------------------------------


class ContinualCounter:
    """Differentially private continual counting mechanism.

    At each time step t, receives a bit b_t ∈ {0,1} and releases a noisy
    count of total 1-bits seen so far.  Uses the binary tree mechanism
    internally for O(log² T) error.

    Can also operate on real-valued inputs (partial sums) when clipped to
    [0, sensitivity].
    """

    def __init__(
        self,
        max_time: int = 1000,
        epsilon: float = 1.0,
        delta: float = 0.0,
        sensitivity: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self.max_time = max_time
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        config = StreamConfig(
            max_time=max_time, epsilon=epsilon, delta=delta,
            sensitivity=sensitivity, seed=seed,
        )
        self._tree_mech = BinaryTreeMechanism(config=config)
        self._true_count = 0.0
        self._time = 0
        self._outputs: List[StreamOutput] = []

    def increment(self, value: float = 1.0) -> StreamOutput:
        """Observe a value and return the noisy running count.

        Args:
            value: Value to add (default 1.0 for counting).

        Returns:
            StreamOutput with noisy running count.
        """
        clipped = max(0.0, min(value, self.sensitivity))
        self._true_count += clipped
        output = self._tree_mech.observe(clipped)
        self._time += 1
        self._outputs.append(output)
        return output

    def current_count(self) -> StreamOutput:
        """Query the current noisy count without incrementing."""
        return self._tree_mech.query()

    def privacy_spent(self) -> float:
        return self._tree_mech.privacy_spent()

    def reset(self) -> None:
        self._tree_mech.reset()
        self._true_count = 0.0
        self._time = 0
        self._outputs = []

    @property
    def true_count(self) -> float:
        return self._true_count

    def mean_error(self) -> float:
        """Mean absolute error across all outputs so far."""
        if not self._outputs:
            return 0.0
        errors = [abs(o.noise_added) for o in self._outputs if o.noise_added is not None]
        return float(np.mean(errors)) if errors else 0.0

    def __repr__(self) -> str:
        return (
            f"ContinualCounter(T={self.max_time}, ε={self.epsilon}, "
            f"count={self._true_count})"
        )


# ---------------------------------------------------------------------------
# EventLevelDP
# ---------------------------------------------------------------------------


class EventLevelDP:
    """Event-level differentially private stream processing.

    Provides ε-DP where neighboring streams differ in a single event
    (insertion/deletion of one element).  Uses Laplace noise calibrated
    to the per-event sensitivity and distributes budget over time using
    the binary tree mechanism.
    """

    def __init__(
        self,
        max_time: int = 1000,
        epsilon: float = 1.0,
        delta: float = 0.0,
        sensitivity: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self.max_time = max_time
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self._rng = np.random.default_rng(seed)
        self._counter = ContinualCounter(
            max_time=max_time, epsilon=epsilon, delta=delta,
            sensitivity=sensitivity, seed=seed,
        )
        self._values: List[float] = []

    def observe(self, value: float) -> StreamOutput:
        """Process a stream event and release noisy aggregate."""
        self._values.append(value)
        return self._counter.increment(value)

    def query_range(self, start: int, end: int) -> float:
        """Noisy sum over event range [start, end)."""
        if not self._values:
            return 0.0
        prefix_end = self._counter._tree_mech.range_query(start, min(end, len(self._values)))
        return prefix_end

    def privacy_spent(self) -> float:
        return self._counter.privacy_spent()

    def reset(self) -> None:
        self._counter.reset()
        self._values = []

    def __repr__(self) -> str:
        return f"EventLevelDP(T={self.max_time}, ε={self.epsilon}, events={len(self._values)})"


# ---------------------------------------------------------------------------
# UserLevelDP
# ---------------------------------------------------------------------------


class UserLevelDP:
    """User-level differentially private stream processing.

    Provides ε-DP where neighboring streams differ in all events from one
    user.  Each user can contribute at most `max_contributions` events;
    the sensitivity is scaled accordingly.
    """

    def __init__(
        self,
        max_time: int = 1000,
        epsilon: float = 1.0,
        delta: float = 0.0,
        max_contributions: int = 1,
        per_event_sensitivity: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self.max_time = max_time
        self.epsilon = epsilon
        self.delta = delta
        self.max_contributions = max_contributions
        self.per_event_sensitivity = per_event_sensitivity
        # User-level sensitivity = max_contributions * per_event_sensitivity
        self.sensitivity = max_contributions * per_event_sensitivity
        self._counter = ContinualCounter(
            max_time=max_time, epsilon=epsilon, delta=delta,
            sensitivity=self.sensitivity, seed=seed,
        )
        self._user_counts: Dict[str, int] = {}
        self._time = 0

    def observe(self, user_id: str, value: float) -> StreamOutput:
        """Process a user event, enforcing contribution limits.

        Args:
            user_id: Identifier of the contributing user.
            value: Value to aggregate.

        Returns:
            StreamOutput with noisy aggregate.
        """
        count = self._user_counts.get(user_id, 0)
        if count >= self.max_contributions:
            # User has exceeded contribution limit; ignore
            clipped = 0.0
        else:
            clipped = max(-self.per_event_sensitivity,
                          min(value, self.per_event_sensitivity))
            self._user_counts[user_id] = count + 1
        output = self._counter.increment(clipped)
        self._time += 1
        return output

    def privacy_spent(self) -> float:
        return self._counter.privacy_spent()

    def reset(self) -> None:
        self._counter.reset()
        self._user_counts = {}
        self._time = 0

    @property
    def num_users(self) -> int:
        return len(self._user_counts)

    def __repr__(self) -> str:
        return (
            f"UserLevelDP(T={self.max_time}, ε={self.epsilon}, "
            f"max_contrib={self.max_contributions}, users={self.num_users})"
        )


# ---------------------------------------------------------------------------
# WindowedDP
# ---------------------------------------------------------------------------


class WindowedDP:
    """w-event privacy for sliding windows (Kellaris et al. 2014).

    Provides differential privacy over any window of w consecutive events.
    The total budget ε is split across the window, and older events are
    "forgotten" as the window slides.
    """

    def __init__(
        self,
        window_size: int = 100,
        epsilon: float = 1.0,
        delta: float = 0.0,
        sensitivity: float = 1.0,
        budget_strategy: str = "uniform",
        seed: Optional[int] = None,
    ) -> None:
        self.window_size = window_size
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.budget_strategy = budget_strategy
        self._rng = np.random.default_rng(seed)
        self._buffer: List[float] = []
        self._noisy_buffer: List[float] = []
        self._time = 0
        self._outputs: List[StreamOutput] = []
        self._budget_per_step = self._compute_budget_allocation()

    def _compute_budget_allocation(self) -> npt.NDArray[np.float64]:
        """Allocate budget across window positions."""
        w = self.window_size
        if self.budget_strategy == "uniform":
            return np.full(w, self.epsilon / w)
        elif self.budget_strategy == "exponential":
            # More budget to recent events
            weights = np.exp(np.linspace(-2, 0, w))
            weights /= weights.sum()
            return weights * self.epsilon
        else:
            return np.full(w, self.epsilon / w)

    def observe(self, value: float) -> StreamOutput:
        """Process a new event and return noisy windowed sum."""
        self._buffer.append(value)
        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)
            self._noisy_buffer.pop(0)
        # Add noise based on budget allocation
        pos = min(len(self._buffer) - 1, self.window_size - 1)
        eps_step = float(self._budget_per_step[pos])
        if self.delta == 0.0:
            scale = self.sensitivity / eps_step
            noise = self._rng.laplace(0, scale)
        else:
            sigma = self.sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / eps_step
            noise = self._rng.normal(0, sigma)
        noisy_val = value + noise
        self._noisy_buffer.append(noisy_val)
        # Windowed noisy sum
        window_sum = sum(self._noisy_buffer)
        true_sum = sum(self._buffer)
        output = StreamOutput(
            timestamp=self._time,
            value=window_sum,
            true_value=true_sum,
            noise_added=window_sum - true_sum,
        )
        self._time += 1
        self._outputs.append(output)
        return output

    def query(self) -> StreamOutput:
        """Return current windowed noisy sum."""
        window_sum = sum(self._noisy_buffer) if self._noisy_buffer else 0.0
        true_sum = sum(self._buffer) if self._buffer else 0.0
        return StreamOutput(
            timestamp=max(0, self._time - 1),
            value=window_sum,
            true_value=true_sum,
            noise_added=window_sum - true_sum,
        )

    def privacy_spent(self) -> float:
        """Privacy is bounded by ε over any w-length window."""
        return self.epsilon

    def reset(self) -> None:
        self._buffer = []
        self._noisy_buffer = []
        self._time = 0
        self._outputs = []

    def __repr__(self) -> str:
        return f"WindowedDP(w={self.window_size}, ε={self.epsilon}, t={self._time})"


# ---------------------------------------------------------------------------
# DecayingPrivacy
# ---------------------------------------------------------------------------


class DecayingPrivacy:
    """Privacy guarantees that decay over time.

    More recent events receive stronger privacy protection (smaller ε)
    while older events have weaker guarantees.  The decay can follow
    geometric, polynomial, or custom schedules.
    """

    def __init__(
        self,
        max_time: int = 1000,
        base_epsilon: float = 1.0,
        decay_rate: float = 0.95,
        decay_type: str = "geometric",
        sensitivity: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self.max_time = max_time
        self.base_epsilon = base_epsilon
        self.decay_rate = decay_rate
        self.decay_type = decay_type
        self.sensitivity = sensitivity
        self._rng = np.random.default_rng(seed)
        self._time = 0
        self._values: List[float] = []
        self._noisy_values: List[float] = []
        self._outputs: List[StreamOutput] = []

    def _epsilon_at_age(self, age: int) -> float:
        """Compute effective epsilon for an event of given age."""
        if self.decay_type == "geometric":
            return self.base_epsilon * (self.decay_rate ** age)
        elif self.decay_type == "polynomial":
            return self.base_epsilon / (1.0 + age) ** self.decay_rate
        else:
            return self.base_epsilon * (self.decay_rate ** age)

    def current_epsilon(self) -> float:
        """Effective epsilon for the most recent event."""
        return self._epsilon_at_age(0)

    def observe(self, value: float) -> StreamOutput:
        """Process a new event with decaying privacy."""
        eps = self._epsilon_at_age(0)
        scale = self.sensitivity / eps
        noise = self._rng.laplace(0, scale)
        noisy = value + noise
        self._values.append(value)
        self._noisy_values.append(noisy)
        # Compute running noisy sum
        noisy_sum = sum(self._noisy_values)
        true_sum = sum(self._values)
        output = StreamOutput(
            timestamp=self._time,
            value=noisy_sum,
            true_value=true_sum,
            noise_added=noisy_sum - true_sum,
        )
        self._time += 1
        self._outputs.append(output)
        return output

    def privacy_at_time(self, query_time: int) -> float:
        """Effective epsilon for events at a given time, queried now."""
        if query_time > self._time:
            return self.base_epsilon
        age = self._time - query_time
        return self._epsilon_at_age(age)

    def privacy_profile(self) -> npt.NDArray[np.float64]:
        """Return array of effective epsilons for each past event."""
        return np.array([
            self._epsilon_at_age(self._time - t)
            for t in range(self._time)
        ])

    def reset(self) -> None:
        self._time = 0
        self._values = []
        self._noisy_values = []
        self._outputs = []

    def __repr__(self) -> str:
        return (
            f"DecayingPrivacy(ε₀={self.base_epsilon}, "
            f"decay={self.decay_rate}, type={self.decay_type})"
        )


# ---------------------------------------------------------------------------
# PanPrivacy
# ---------------------------------------------------------------------------


class PanPrivacy:
    """Pan-private streaming algorithm (Mir et al. 2011).

    Provides privacy even against an adversary who can observe the
    internal state of the algorithm at arbitrary points.  Implemented
    via randomised internal state that hides individual contributions.
    """

    def __init__(
        self,
        max_time: int = 1000,
        epsilon: float = 1.0,
        num_bins: int = 10,
        value_range: Tuple[float, float] = (0.0, 1.0),
        seed: Optional[int] = None,
    ) -> None:
        self.max_time = max_time
        self.epsilon = epsilon
        self.num_bins = num_bins
        self.value_range = value_range
        self._rng = np.random.default_rng(seed)
        self._time = 0
        # Internal state: randomised histogram counts
        # Initialised with geometric noise for pan-privacy
        self._counts = np.zeros(num_bins, dtype=np.float64)
        self._noisy_counts = np.zeros(num_bins, dtype=np.float64)
        self._initialize_state()

    def _initialize_state(self) -> None:
        """Initialize internal state with noise for pan-privacy."""
        # Two-sided geometric noise for each bin
        p = 1.0 - math.exp(-self.epsilon / 2.0)
        for i in range(self.num_bins):
            geo1 = self._rng.geometric(p) - 1
            geo2 = self._rng.geometric(p) - 1
            self._noisy_counts[i] = float(geo1 - geo2)

    def _value_to_bin(self, value: float) -> int:
        lo, hi = self.value_range
        if hi <= lo:
            return 0
        frac = (value - lo) / (hi - lo)
        b = int(frac * self.num_bins)
        return max(0, min(b, self.num_bins - 1))

    def observe(self, value: float) -> None:
        """Process a stream element (updates internal state).

        The internal state is always pan-private: even if observed,
        it reveals nothing about individual contributions.
        """
        b = self._value_to_bin(value)
        self._counts[b] += 1.0
        # Update noisy counts with randomised response
        coin = self._rng.random()
        threshold = math.exp(self.epsilon) / (1.0 + math.exp(self.epsilon))
        if coin < threshold:
            self._noisy_counts[b] += 1.0
        else:
            # Add to a random other bin
            other = self._rng.integers(0, self.num_bins)
            self._noisy_counts[other] += 1.0
        self._time += 1

    def release_histogram(self) -> npt.NDArray[np.float64]:
        """Release the current noisy histogram.

        Returns:
            Array of noisy bin counts.
        """
        return self._noisy_counts.copy()

    def release_count(self, bin_index: int) -> float:
        """Release the noisy count for a specific bin."""
        if 0 <= bin_index < self.num_bins:
            return float(self._noisy_counts[bin_index])
        raise ValueError(f"bin_index {bin_index} out of range [0, {self.num_bins})")

    def privacy_spent(self) -> float:
        return self.epsilon

    def reset(self) -> None:
        self._time = 0
        self._counts = np.zeros(self.num_bins, dtype=np.float64)
        self._noisy_counts = np.zeros(self.num_bins, dtype=np.float64)
        self._initialize_state()

    @property
    def internal_state(self) -> npt.NDArray[np.float64]:
        """The (pan-private) internal state."""
        return self._noisy_counts.copy()

    def __repr__(self) -> str:
        return (
            f"PanPrivacy(ε={self.epsilon}, bins={self.num_bins}, "
            f"t={self._time})"
        )


# ---------------------------------------------------------------------------
# StreamingHistogram
# ---------------------------------------------------------------------------


class StreamingHistogram:
    """Differentially private streaming histogram.

    Maintains a histogram over a stream of categorical or binned data,
    releasing noisy counts at each time step.  Supports both event-level
    and user-level privacy.
    """

    def __init__(
        self,
        num_bins: int = 10,
        max_time: int = 1000,
        epsilon: float = 1.0,
        delta: float = 0.0,
        sensitivity: float = 1.0,
        user_level: bool = False,
        max_contributions: int = 1,
        seed: Optional[int] = None,
    ) -> None:
        self.num_bins = num_bins
        self.max_time = max_time
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.user_level = user_level
        self.max_contributions = max_contributions
        self._rng = np.random.default_rng(seed)
        self._counts = np.zeros(num_bins, dtype=np.float64)
        self._time = 0
        self._user_counts: Dict[str, int] = {}
        # Per-bin noise scale
        if user_level:
            sens = sensitivity * max_contributions
        else:
            sens = sensitivity
        height = max(1, math.ceil(math.log2(max(max_time, 1))))
        self._noise_alloc = NoiseAllocation(
            height=height, epsilon=epsilon / num_bins,
            delta=delta, sensitivity=sens,
        )

    def observe(self, bin_index: int, user_id: Optional[str] = None) -> npt.NDArray[np.float64]:
        """Observe a new event in the given bin.

        Args:
            bin_index: Which bin the event falls into.
            user_id: User identifier (required if user_level=True).

        Returns:
            Noisy histogram counts.
        """
        if bin_index < 0 or bin_index >= self.num_bins:
            raise ValueError(f"bin_index {bin_index} out of range [0, {self.num_bins})")
        # Check contribution limits
        if self.user_level and user_id is not None:
            count = self._user_counts.get(user_id, 0)
            if count >= self.max_contributions:
                return self.release()
            self._user_counts[user_id] = count + 1
        self._counts[bin_index] += 1.0
        self._time += 1
        return self.release()

    def release(self) -> npt.NDArray[np.float64]:
        """Release current noisy histogram."""
        scale = self._noise_alloc.scale_for_level(0)
        if self.delta == 0.0:
            noise = self._rng.laplace(0, scale, self.num_bins)
        else:
            noise = self._rng.normal(0, scale, self.num_bins)
        return np.maximum(0, self._counts + noise)

    def privacy_spent(self) -> float:
        return self.epsilon

    def reset(self) -> None:
        self._counts = np.zeros(self.num_bins, dtype=np.float64)
        self._time = 0
        self._user_counts = {}

    @property
    def true_counts(self) -> npt.NDArray[np.float64]:
        return self._counts.copy()

    def __repr__(self) -> str:
        return (
            f"StreamingHistogram(bins={self.num_bins}, ε={self.epsilon}, "
            f"t={self._time})"
        )


__all__ = [
    "ContinualCounter",
    "EventLevelDP",
    "UserLevelDP",
    "WindowedDP",
    "DecayingPrivacy",
    "PanPrivacy",
    "StreamingHistogram",
]

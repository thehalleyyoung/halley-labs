"""
Online mechanism synthesis for streaming differential privacy.

Implements adaptive and online algorithms that synthesize DP mechanisms
on-the-fly as stream data arrives, including budget allocation, regret
minimisation, privacy filters, and the sparse-vector / above-threshold
technique.

References:
    - Dwork, Roth. "The Algorithmic Foundations of Differential Privacy."
      Foundations and Trends in TCS 9(3-4), 2014.  (AboveThreshold, SVT)
    - Rogers, Roth, Ullman, Vadhan. "Privacy Odometers and Filters:
      Pay-as-you-go Composition." NeurIPS 2016.
    - Feldman, Zrnic. "Individual Privacy Accounting via a Rényi Filter."
      NeurIPS 2021.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from dp_forge.streaming import (
    NoiseSchedule,
    SparseVectorConfig,
    SparseVectorOutput,
    StreamConfig,
    StreamMechanismType,
    StreamOutput,
    StreamState,
    StreamSummary,
    ThresholdPolicy,
)
from dp_forge.types import PrivacyBudget


# ---------------------------------------------------------------------------
# StreamAccountant
# ---------------------------------------------------------------------------


class StreamAccountant:
    """Track privacy consumption over time in a streaming setting.

    Supports basic, advanced, and zero-concentrated DP composition.
    Provides both an odometer (running total) and a filter (halt when
    budget exceeded).
    """

    def __init__(
        self,
        total_epsilon: float = 1.0,
        total_delta: float = 0.0,
        composition: str = "basic",
    ) -> None:
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.composition = composition
        self._epsilon_history: List[float] = []
        self._delta_history: List[float] = []
        self._spent_epsilon = 0.0
        self._spent_delta = 0.0
        self._exhausted = False

    def add_observation(self, epsilon: float, delta: float = 0.0) -> bool:
        """Record privacy cost and return True if budget remains.

        Args:
            epsilon: ε cost of this observation.
            delta: δ cost of this observation.

        Returns:
            True if budget still available, False if exhausted.
        """
        self._epsilon_history.append(epsilon)
        self._delta_history.append(delta)
        if self.composition == "basic":
            self._spent_epsilon += epsilon
            self._spent_delta += delta
        elif self.composition == "advanced":
            k = len(self._epsilon_history)
            if k > 0 and self.total_delta > 0:
                max_eps = max(self._epsilon_history)
                self._spent_epsilon = (
                    max_eps * math.sqrt(2.0 * k * math.log(1.0 / self.total_delta))
                    + k * max_eps * (math.exp(max_eps) - 1.0)
                )
            else:
                self._spent_epsilon += epsilon
            self._spent_delta = sum(self._delta_history)
        elif self.composition == "zcdp":
            # zCDP composition: ρ_total = Σ ρ_i where ρ_i = ε_i²/2
            rho_total = sum(e ** 2 / 2.0 for e in self._epsilon_history)
            if self.total_delta > 0:
                self._spent_epsilon = (
                    rho_total + 2.0 * math.sqrt(rho_total * math.log(1.0 / self.total_delta))
                )
            else:
                self._spent_epsilon = 2.0 * rho_total
            self._spent_delta = sum(self._delta_history)
        self._exhausted = (
            self._spent_epsilon > self.total_epsilon
            or self._spent_delta > self.total_delta
        )
        return not self._exhausted

    def budget_remaining(self) -> Tuple[float, float]:
        """Return (remaining_epsilon, remaining_delta)."""
        return (
            max(0.0, self.total_epsilon - self._spent_epsilon),
            max(0.0, self.total_delta - self._spent_delta),
        )

    @property
    def is_exhausted(self) -> bool:
        return self._exhausted

    @property
    def spent_epsilon(self) -> float:
        return self._spent_epsilon

    @property
    def spent_delta(self) -> float:
        return self._spent_delta

    @property
    def num_observations(self) -> int:
        return len(self._epsilon_history)

    def privacy_curve(self) -> npt.NDArray[np.float64]:
        """Return cumulative epsilon over time."""
        if not self._epsilon_history:
            return np.array([], dtype=np.float64)
        return np.cumsum(self._epsilon_history)

    def __repr__(self) -> str:
        return (
            f"StreamAccountant(ε={self._spent_epsilon:.4f}/{self.total_epsilon}, "
            f"δ={self._spent_delta:.6f}/{self.total_delta}, "
            f"comp={self.composition})"
        )


# ---------------------------------------------------------------------------
# AdaptiveBudgetAllocator
# ---------------------------------------------------------------------------


class AdaptiveBudgetAllocator:
    """Dynamically allocate privacy budget across time steps.

    Adapts the per-step budget based on observed data characteristics,
    remaining budget, and an allocation strategy.
    """

    def __init__(
        self,
        total_epsilon: float = 1.0,
        max_time: int = 1000,
        strategy: str = "uniform",
        decay_rate: float = 0.99,
    ) -> None:
        self.total_epsilon = total_epsilon
        self.max_time = max_time
        self.strategy = strategy
        self.decay_rate = decay_rate
        self._allocations: List[float] = []
        self._remaining = total_epsilon
        self._time = 0

    def allocate(self) -> float:
        """Return the epsilon budget for the next time step.

        Returns:
            Per-step epsilon.
        """
        if self._remaining <= 0:
            return 0.0
        remaining_steps = max(1, self.max_time - self._time)
        if self.strategy == "uniform":
            eps = self.total_epsilon / self.max_time
        elif self.strategy == "geometric":
            # Geometric decay: more budget early
            eps = self._remaining * (1.0 - self.decay_rate)
            eps = max(eps, self._remaining / remaining_steps)
        elif self.strategy == "sqrt":
            # Proportional to 1/sqrt(remaining_steps)
            weight = 1.0 / math.sqrt(remaining_steps)
            total_weight = sum(
                1.0 / math.sqrt(max(1, self.max_time - s))
                for s in range(self._time, self.max_time)
            )
            eps = self._remaining * weight / max(total_weight, 1e-12)
        elif self.strategy == "doubling":
            # Doubling: each epoch doubles the budget
            epoch = int(math.log2(max(self._time + 1, 1)))
            epoch_size = 2 ** epoch
            eps = self._remaining / max(remaining_steps, 1)
            eps *= (epoch + 1)
            eps = min(eps, self._remaining)
        else:
            eps = self.total_epsilon / self.max_time
        eps = min(eps, self._remaining)
        self._remaining -= eps
        self._allocations.append(eps)
        self._time += 1
        return eps

    def peek(self) -> float:
        """Preview the next allocation without consuming budget."""
        remaining_steps = max(1, self.max_time - self._time)
        if self.strategy == "uniform":
            return self.total_epsilon / self.max_time
        elif self.strategy == "geometric":
            return self._remaining * (1.0 - self.decay_rate)
        return self._remaining / remaining_steps

    @property
    def remaining_budget(self) -> float:
        return self._remaining

    @property
    def allocation_history(self) -> List[float]:
        return list(self._allocations)

    def reset(self) -> None:
        self._allocations = []
        self._remaining = self.total_epsilon
        self._time = 0

    def __repr__(self) -> str:
        return (
            f"AdaptiveBudgetAllocator(ε={self.total_epsilon}, "
            f"remaining={self._remaining:.4f}, strategy={self.strategy})"
        )


# ---------------------------------------------------------------------------
# PrivacyFilter
# ---------------------------------------------------------------------------


class PrivacyFilter:
    """Privacy filter for budget enforcement (Rogers et al. 2016).

    A filter monitors the privacy loss in real time and halts the mechanism
    when the budget is about to be exceeded, providing a valid privacy
    guarantee even for adaptively chosen queries.
    """

    def __init__(
        self,
        total_epsilon: float = 1.0,
        total_delta: float = 0.0,
        slack: float = 0.01,
    ) -> None:
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.slack = slack
        self._spent_epsilon = 0.0
        self._spent_delta = 0.0
        self._halted = False
        self._queries_answered = 0

    def check_and_charge(self, epsilon: float, delta: float = 0.0) -> bool:
        """Check if query can be answered and charge budget.

        Args:
            epsilon: ε cost of this query.
            delta: δ cost of this query.

        Returns:
            True if query was answered, False if filter halted.
        """
        if self._halted:
            return False
        # Check if answering would exceed budget (with slack)
        if (self._spent_epsilon + epsilon > self.total_epsilon + self.slack
                or self._spent_delta + delta > self.total_delta + self.slack):
            self._halted = True
            return False
        self._spent_epsilon += epsilon
        self._spent_delta += delta
        self._queries_answered += 1
        return True

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def budget_remaining(self) -> float:
        return max(0.0, self.total_epsilon - self._spent_epsilon)

    @property
    def queries_answered(self) -> int:
        return self._queries_answered

    def reset(self) -> None:
        self._spent_epsilon = 0.0
        self._spent_delta = 0.0
        self._halted = False
        self._queries_answered = 0

    def __repr__(self) -> str:
        status = "HALTED" if self._halted else "active"
        return (
            f"PrivacyFilter({status}, ε={self._spent_epsilon:.4f}/"
            f"{self.total_epsilon}, queries={self._queries_answered})"
        )


# ---------------------------------------------------------------------------
# AboveThreshold
# ---------------------------------------------------------------------------


class AboveThreshold:
    """Differentially private above-threshold mechanism (SVT).

    Given a stream of sensitivity-1 queries q_1, q_2, ..., answers
    "is q_t(D) ≥ T?" while spending O(ε) total budget regardless of the
    number of below-threshold answers.  Halts after c above-threshold
    answers.

    This is the standard Sparse Vector Technique from Dwork & Roth (2014).
    """

    def __init__(
        self,
        threshold: float = 0.0,
        epsilon: float = 1.0,
        max_above: int = 1,
        sensitivity: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self.threshold = threshold
        self.epsilon = epsilon
        self.max_above = max_above
        self.sensitivity = sensitivity
        self._rng = np.random.default_rng(seed)
        # Split budget: ε/2 for threshold noise, ε/2 for query noise
        self._threshold_scale = 2.0 * self.sensitivity / self.epsilon
        self._query_scale = 4.0 * self.max_above * self.sensitivity / self.epsilon
        # Noisy threshold (set once)
        self._noisy_threshold = (
            self.threshold + self._rng.laplace(0, self._threshold_scale)
        )
        self._above_count = 0
        self._total_queries = 0
        self._halted = False
        self._results: List[SparseVectorOutput] = []

    def test(self, query_value: float) -> SparseVectorOutput:
        """Test whether query_value exceeds the noisy threshold.

        Args:
            query_value: True answer to the query.

        Returns:
            SparseVectorOutput with above/below decision.
        """
        if self._halted:
            output = SparseVectorOutput(
                timestamp=self._total_queries,
                above_threshold=False,
                queries_remaining=0,
                halted=True,
            )
            self._results.append(output)
            return output
        noisy_query = query_value + self._rng.laplace(0, self._query_scale)
        above = noisy_query >= self._noisy_threshold
        if above:
            self._above_count += 1
            if self._above_count >= self.max_above:
                self._halted = True
        self._total_queries += 1
        output = SparseVectorOutput(
            timestamp=self._total_queries - 1,
            above_threshold=above,
            queries_remaining=max(0, self.max_above - self._above_count),
            halted=self._halted,
        )
        self._results.append(output)
        return output

    def batch_test(self, query_values: Sequence[float]) -> List[SparseVectorOutput]:
        """Test multiple queries in sequence."""
        return [self.test(q) for q in query_values]

    def privacy_spent(self) -> float:
        return self.epsilon

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def above_count(self) -> int:
        return self._above_count

    def reset(self) -> None:
        self._noisy_threshold = (
            self.threshold + self._rng.laplace(0, self._threshold_scale)
        )
        self._above_count = 0
        self._total_queries = 0
        self._halted = False
        self._results = []

    def __repr__(self) -> str:
        status = "HALTED" if self._halted else "active"
        return (
            f"AboveThreshold({status}, T={self.threshold}, "
            f"above={self._above_count}/{self.max_above}, "
            f"queries={self._total_queries})"
        )


# ---------------------------------------------------------------------------
# ExponentialMechanism (stream setting)
# ---------------------------------------------------------------------------


class ExponentialMechanism:
    """Exponential mechanism adapted for streaming settings.

    At each time step, selects an output from a discrete set of candidates
    using the exponential mechanism with a quality score function.  Budget
    is consumed per selection.
    """

    def __init__(
        self,
        candidates: Sequence[float],
        epsilon: float = 1.0,
        sensitivity: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self.candidates = np.asarray(candidates, dtype=np.float64)
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self._rng = np.random.default_rng(seed)
        self._time = 0
        self._selections: List[float] = []

    def select(
        self,
        quality_scores: npt.NDArray[np.float64],
        per_step_epsilon: Optional[float] = None,
    ) -> float:
        """Select a candidate using the exponential mechanism.

        Args:
            quality_scores: Quality score for each candidate (higher = better).
            per_step_epsilon: Per-step budget (default: self.epsilon).

        Returns:
            The selected candidate value.
        """
        eps = per_step_epsilon or self.epsilon
        scores = np.asarray(quality_scores, dtype=np.float64)
        if len(scores) != len(self.candidates):
            raise ValueError("scores length must match candidates length")
        # Exponential mechanism weights
        weights = eps * scores / (2.0 * self.sensitivity)
        # Numerical stability: subtract max
        weights -= np.max(weights)
        probs = np.exp(weights)
        probs /= probs.sum()
        idx = self._rng.choice(len(self.candidates), p=probs)
        selected = float(self.candidates[idx])
        self._selections.append(selected)
        self._time += 1
        return selected

    def select_with_score(
        self,
        score_fn: Callable[[float], float],
        per_step_epsilon: Optional[float] = None,
    ) -> float:
        """Select using a score function applied to each candidate."""
        scores = np.array([score_fn(c) for c in self.candidates])
        return self.select(scores, per_step_epsilon)

    @property
    def selection_history(self) -> List[float]:
        return list(self._selections)

    def reset(self) -> None:
        self._time = 0
        self._selections = []

    def __repr__(self) -> str:
        return (
            f"ExponentialMechanism(|R|={len(self.candidates)}, "
            f"ε={self.epsilon}, t={self._time})"
        )


# ---------------------------------------------------------------------------
# RegretMinimizer
# ---------------------------------------------------------------------------


class RegretMinimizer:
    """Minimize regret in online mechanism selection.

    Uses multiplicative weights / Hedge algorithm to adaptively select
    among a set of DP mechanisms, minimizing cumulative regret.
    """

    def __init__(
        self,
        num_mechanisms: int = 3,
        learning_rate: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        self.num_mechanisms = num_mechanisms
        self.learning_rate = learning_rate
        self._rng = np.random.default_rng(seed)
        self._weights = np.ones(num_mechanisms, dtype=np.float64) / num_mechanisms
        self._cumulative_loss: npt.NDArray[np.float64] = np.zeros(num_mechanisms)
        self._time = 0
        self._selections: List[int] = []
        self._losses: List[npt.NDArray[np.float64]] = []

    def select(self) -> int:
        """Select a mechanism index using current weights.

        Returns:
            Index of the selected mechanism.
        """
        probs = self._weights / self._weights.sum()
        idx = int(self._rng.choice(self.num_mechanisms, p=probs))
        self._selections.append(idx)
        return idx

    def update(self, losses: npt.NDArray[np.float64]) -> None:
        """Update weights after observing losses for all mechanisms.

        Args:
            losses: Loss vector (one per mechanism). Lower is better.
        """
        losses = np.asarray(losses, dtype=np.float64)
        if len(losses) != self.num_mechanisms:
            raise ValueError("losses length must match num_mechanisms")
        self._losses.append(losses.copy())
        self._cumulative_loss += losses
        # Multiplicative weights update
        self._weights *= np.exp(-self.learning_rate * losses)
        # Normalise
        total = self._weights.sum()
        if total > 0:
            self._weights /= total
        else:
            self._weights = np.ones(self.num_mechanisms) / self.num_mechanisms
        self._time += 1

    def regret(self) -> float:
        """Compute regret: cumulative loss of selections vs best fixed mechanism."""
        if not self._losses:
            return 0.0
        total_selected = sum(
            self._losses[t][self._selections[t]]
            for t in range(len(self._selections))
            if t < len(self._losses)
        )
        best_fixed = float(np.min(self._cumulative_loss))
        return total_selected - best_fixed

    def regret_bound(self) -> float:
        """Theoretical regret bound: O(sqrt(T log K))."""
        if self._time == 0:
            return 0.0
        return math.sqrt(
            2.0 * self._time * math.log(self.num_mechanisms) / self.learning_rate
        )

    @property
    def current_weights(self) -> npt.NDArray[np.float64]:
        return self._weights.copy()

    @property
    def best_mechanism(self) -> int:
        return int(np.argmin(self._cumulative_loss))

    def reset(self) -> None:
        self._weights = np.ones(self.num_mechanisms) / self.num_mechanisms
        self._cumulative_loss = np.zeros(self.num_mechanisms)
        self._time = 0
        self._selections = []
        self._losses = []

    def __repr__(self) -> str:
        return (
            f"RegretMinimizer(K={self.num_mechanisms}, "
            f"t={self._time}, regret={self.regret():.4f})"
        )


# ---------------------------------------------------------------------------
# OnlineSynthesizer
# ---------------------------------------------------------------------------


class OnlineSynthesizer:
    """Synthesize a DP mechanism for each time step.

    Combines budget allocation, mechanism selection, and privacy accounting
    into an end-to-end online mechanism synthesis pipeline.
    """

    def __init__(
        self,
        max_time: int = 1000,
        epsilon: float = 1.0,
        delta: float = 0.0,
        sensitivity: float = 1.0,
        budget_strategy: str = "uniform",
        seed: Optional[int] = None,
    ) -> None:
        self.max_time = max_time
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self._rng = np.random.default_rng(seed)
        self._allocator = AdaptiveBudgetAllocator(
            total_epsilon=epsilon, max_time=max_time, strategy=budget_strategy,
        )
        self._accountant = StreamAccountant(
            total_epsilon=epsilon, total_delta=delta,
        )
        self._filter = PrivacyFilter(
            total_epsilon=epsilon, total_delta=delta,
        )
        self._time = 0
        self._running_sum = 0.0
        self._outputs: List[StreamOutput] = []

    def observe(self, value: float) -> StreamOutput:
        """Process a new value and return a noisy aggregate.

        Allocates budget, checks filter, adds calibrated noise, and
        tracks privacy consumption.
        """
        if self._filter.is_halted:
            # Budget exhausted; return last output
            output = StreamOutput(
                timestamp=self._time, value=self._running_sum,
                true_value=self._running_sum, noise_added=0.0,
            )
            self._time += 1
            return output
        eps_step = self._allocator.allocate()
        # Check filter
        if not self._filter.check_and_charge(eps_step, 0.0):
            output = StreamOutput(
                timestamp=self._time, value=self._running_sum,
                true_value=self._running_sum, noise_added=0.0,
            )
            self._time += 1
            return output
        self._accountant.add_observation(eps_step, 0.0)
        self._running_sum += value
        # Add noise calibrated to per-step budget
        if eps_step > 0:
            scale = self.sensitivity / eps_step
            noise = self._rng.laplace(0, scale)
        else:
            noise = 0.0
        noisy_sum = self._running_sum + noise
        output = StreamOutput(
            timestamp=self._time, value=noisy_sum,
            true_value=self._running_sum,
            noise_added=noise,
        )
        self._outputs.append(output)
        self._time += 1
        return output

    def query(self) -> StreamOutput:
        """Query current noisy aggregate without new observation."""
        if not self._outputs:
            return StreamOutput(timestamp=0, value=0.0, true_value=0.0, noise_added=0.0)
        return self._outputs[-1]

    def privacy_spent(self) -> float:
        return self._accountant.spent_epsilon

    @property
    def is_halted(self) -> bool:
        return self._filter.is_halted

    @property
    def budget_remaining(self) -> float:
        return self._allocator.remaining_budget

    def summarize(self) -> StreamSummary:
        """Return summary statistics."""
        if not self._outputs:
            return StreamSummary(
                total_time_steps=0, total_privacy_spent=self.privacy_spent(),
                mean_absolute_error=0.0, max_absolute_error=0.0, rmse=0.0,
                mechanism_type=StreamMechanismType.PRIVATE_COUNTER,
            )
        errors = [
            abs(o.noise_added) for o in self._outputs if o.noise_added is not None
        ]
        if not errors:
            errors = [0.0]
        return StreamSummary(
            total_time_steps=len(self._outputs),
            total_privacy_spent=self.privacy_spent(),
            mean_absolute_error=float(np.mean(errors)),
            max_absolute_error=float(np.max(errors)),
            rmse=float(np.sqrt(np.mean(np.array(errors) ** 2))),
            mechanism_type=StreamMechanismType.PRIVATE_COUNTER,
        )

    def reset(self) -> None:
        self._allocator.reset()
        self._accountant = StreamAccountant(
            total_epsilon=self.epsilon, total_delta=self.delta,
        )
        self._filter.reset()
        self._time = 0
        self._running_sum = 0.0
        self._outputs = []

    def __repr__(self) -> str:
        return (
            f"OnlineSynthesizer(T={self.max_time}, ε={self.epsilon}, "
            f"spent={self.privacy_spent():.4f})"
        )


__all__ = [
    "OnlineSynthesizer",
    "AdaptiveBudgetAllocator",
    "StreamAccountant",
    "RegretMinimizer",
    "ExponentialMechanism",
    "PrivacyFilter",
    "AboveThreshold",
]

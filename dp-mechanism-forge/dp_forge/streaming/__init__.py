"""
Streaming differential privacy mechanisms for continual observation.

This package implements DP mechanisms for the streaming/continual observation
setting where data arrives over time and the mechanism must release outputs
at each time step while maintaining privacy across all time steps.

Key mechanisms:
- **Binary tree mechanism**: O(log T) error using binary tree aggregation
  (Dwork et al. 2010, Chan et al. 2011).
- **Factorisation mechanism**: Optimal factorisation M = LR for streaming
  workloads (Li et al. 2015).
- **Sparse vector technique**: Private threshold testing on a stream with
  logarithmic budget consumption.
- **Private counters**: Continual counting with various noise schedules.
- **Sliding window**: DP mechanisms with bounded memory (sliding window).

Architecture:
    1. **StreamingMechanism** — Base protocol for all streaming mechanisms.
    2. **BinaryTreeMechanism** — Binary tree aggregation for prefix sums.
    3. **FactorisationMechanism** — Matrix factorisation for general workloads.
    4. **SparseVectorMechanism** — Above-threshold with privacy budget tracking.
    5. **SlidingWindowMechanism** — Bounded-memory streaming DP.
    6. **StreamAccountant** — Privacy accounting for streaming mechanisms.

Example::

    from dp_forge.streaming import BinaryTreeMechanism, StreamConfig

    config = StreamConfig(max_time=1000, epsilon=1.0)
    mech = BinaryTreeMechanism(config=config)

    for t, value in enumerate(stream):
        noisy_sum = mech.observe(value)
        print(f"t={t}: noisy prefix sum = {noisy_sum:.2f}")

    print(f"Total privacy: {mech.privacy_spent()}")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
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
    PrivacyBudget,
    PrivacyNotion,
    StreamEvent,
    StreamEventType,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StreamMechanismType(Enum):
    """Types of streaming DP mechanisms."""

    BINARY_TREE = auto()
    FACTORISATION = auto()
    SPARSE_VECTOR = auto()
    PRIVATE_COUNTER = auto()
    SLIDING_WINDOW = auto()
    MATRIX_MECHANISM = auto()

    def __repr__(self) -> str:
        return f"StreamMechanismType.{self.name}"


class NoiseSchedule(Enum):
    """Noise schedule for streaming mechanisms."""

    UNIFORM = auto()
    GEOMETRIC_DECAY = auto()
    EXPONENTIAL_DECAY = auto()
    ADAPTIVE = auto()

    def __repr__(self) -> str:
        return f"NoiseSchedule.{self.name}"


class TreeStructure(Enum):
    """Tree structure for tree-based streaming mechanisms."""

    BINARY = auto()
    B_ARY = auto()
    LOGARITHMIC = auto()
    OPTIMAL = auto()

    def __repr__(self) -> str:
        return f"TreeStructure.{self.name}"


class ThresholdPolicy(Enum):
    """Policy for sparse vector threshold comparisons."""

    ABOVE_THRESHOLD = auto()
    BELOW_THRESHOLD = auto()
    NUMERIC_SPARSE = auto()

    def __repr__(self) -> str:
        return f"ThresholdPolicy.{self.name}"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class StreamConfig:
    """Configuration for streaming DP mechanisms.

    Attributes:
        max_time: Maximum number of time steps (T).
        epsilon: Per-step or total privacy parameter ε.
        delta: Privacy parameter δ.
        privacy_notion: Which DP notion to use.
        noise_schedule: Noise schedule over time.
        tree_structure: Tree structure for tree-based mechanisms.
        branching_factor: Branching factor for B-ary trees.
        window_size: Window size for sliding window mechanisms.
        sensitivity: Per-step sensitivity.
        seed: Random seed for reproducibility.
        verbose: Verbosity level.
    """

    max_time: int = 1000
    epsilon: float = 1.0
    delta: float = 0.0
    privacy_notion: PrivacyNotion = PrivacyNotion.PURE_DP
    noise_schedule: NoiseSchedule = NoiseSchedule.UNIFORM
    tree_structure: TreeStructure = TreeStructure.BINARY
    branching_factor: int = 2
    window_size: Optional[int] = None
    sensitivity: float = 1.0
    seed: Optional[int] = None
    verbose: int = 1

    def __post_init__(self) -> None:
        if self.max_time < 1:
            raise ValueError(f"max_time must be >= 1, got {self.max_time}")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")
        if not (0.0 <= self.delta < 1.0):
            raise ValueError(f"delta must be in [0, 1), got {self.delta}")
        if self.branching_factor < 2:
            raise ValueError(f"branching_factor must be >= 2, got {self.branching_factor}")
        if self.sensitivity <= 0:
            raise ValueError(f"sensitivity must be > 0, got {self.sensitivity}")
        if self.window_size is not None and self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}")

    def __repr__(self) -> str:
        return (
            f"StreamConfig(T={self.max_time}, ε={self.epsilon}, "
            f"noise={self.noise_schedule.name})"
        )


@dataclass
class SparseVectorConfig:
    """Configuration specific to the sparse vector technique.

    Attributes:
        threshold: Threshold value for above-threshold testing.
        max_queries: Maximum number of "above threshold" answers.
        threshold_noise_scale: Scale of Laplace noise added to threshold.
        query_noise_scale: Scale of Laplace noise added to queries.
        policy: Threshold comparison policy.
    """

    threshold: float = 0.0
    max_queries: int = 1
    threshold_noise_scale: Optional[float] = None
    query_noise_scale: Optional[float] = None
    policy: ThresholdPolicy = ThresholdPolicy.ABOVE_THRESHOLD

    def __post_init__(self) -> None:
        if self.max_queries < 1:
            raise ValueError(f"max_queries must be >= 1, got {self.max_queries}")

    def __repr__(self) -> str:
        return (
            f"SparseVectorConfig(threshold={self.threshold}, "
            f"max_queries={self.max_queries}, policy={self.policy.name})"
        )


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------


@dataclass
class StreamState:
    """Current state of a streaming mechanism.

    Attributes:
        current_time: Current time step.
        running_sum: Running true sum of observed values.
        noisy_sum: Most recent noisy sum released.
        privacy_spent: Cumulative privacy budget spent.
        num_observations: Number of values observed.
        buffer: Internal buffer (mechanism-specific).
    """

    current_time: int = 0
    running_sum: float = 0.0
    noisy_sum: float = 0.0
    privacy_spent: float = 0.0
    num_observations: int = 0
    buffer: Optional[npt.NDArray[np.float64]] = None

    def __repr__(self) -> str:
        return (
            f"StreamState(t={self.current_time}, sum={self.running_sum:.2f}, "
            f"noisy={self.noisy_sum:.2f}, ε_spent={self.privacy_spent:.4f})"
        )


@dataclass
class StreamOutput:
    """Output from a single step of a streaming mechanism.

    Attributes:
        timestamp: Time step of this output.
        value: The noisy value released.
        true_value: The true value (for error measurement, not released).
        noise_added: The noise that was added.
        error: Absolute error |noisy - true|.
    """

    timestamp: int
    value: float
    true_value: Optional[float] = None
    noise_added: Optional[float] = None

    @property
    def error(self) -> Optional[float]:
        """Absolute error if true_value is known."""
        if self.true_value is None:
            return None
        return abs(self.value - self.true_value)

    def __repr__(self) -> str:
        err = f", err={self.error:.4f}" if self.error is not None else ""
        return f"StreamOutput(t={self.timestamp}, value={self.value:.4f}{err})"


@dataclass
class SparseVectorOutput:
    """Output from the sparse vector technique.

    Attributes:
        timestamp: Time step.
        above_threshold: Whether the query was above threshold.
        noisy_answer: Noisy answer (only for numeric sparse variant).
        queries_remaining: Number of "above" answers remaining.
        halted: Whether the mechanism has halted (budget exhausted).
    """

    timestamp: int
    above_threshold: bool
    noisy_answer: Optional[float] = None
    queries_remaining: int = 0
    halted: bool = False

    def __repr__(self) -> str:
        result = "⊤" if self.above_threshold else "⊥"
        status = ", HALTED" if self.halted else f", remaining={self.queries_remaining}"
        return f"SparseVectorOutput(t={self.timestamp}, {result}{status})"


@dataclass
class StreamSummary:
    """Summary statistics for a streaming mechanism run.

    Attributes:
        total_time_steps: Total number of time steps processed.
        total_privacy_spent: Total ε spent.
        mean_absolute_error: Mean absolute error across all time steps.
        max_absolute_error: Maximum absolute error.
        rmse: Root mean squared error.
        mechanism_type: Type of streaming mechanism used.
    """

    total_time_steps: int
    total_privacy_spent: float
    mean_absolute_error: float
    max_absolute_error: float
    rmse: float
    mechanism_type: StreamMechanismType

    def __repr__(self) -> str:
        return (
            f"StreamSummary(T={self.total_time_steps}, ε={self.total_privacy_spent:.4f}, "
            f"MAE={self.mean_absolute_error:.4f}, RMSE={self.rmse:.4f})"
        )


# ---------------------------------------------------------------------------
# Protocols (interfaces)
# ---------------------------------------------------------------------------


@runtime_checkable
class StreamingMechanism(Protocol):
    """Protocol for all streaming DP mechanisms."""

    def observe(self, value: float) -> StreamOutput:
        """Process a new observation and return noisy output."""
        ...

    def query(self) -> StreamOutput:
        """Query the current aggregate without a new observation."""
        ...

    def privacy_spent(self) -> float:
        """Return total privacy budget spent so far."""
        ...

    def reset(self) -> None:
        """Reset the mechanism state."""
        ...

    @property
    def state(self) -> StreamState:
        """Current mechanism state."""
        ...


# ---------------------------------------------------------------------------
# Public API classes
# ---------------------------------------------------------------------------


class BinaryTreeMechanism:
    """Binary tree aggregation for continual prefix sums.

    Achieves O(log²T) error for T time steps while maintaining ε-DP.
    Each node in the binary tree is a partial sum with independent noise.
    """

    def __init__(self, config: Optional[StreamConfig] = None) -> None:
        self.config = config or StreamConfig()
        self._state = StreamState()

    def observe(self, value: float) -> StreamOutput:
        """Process a new value and return the noisy prefix sum.

        Args:
            value: New stream value.

        Returns:
            StreamOutput with noisy prefix sum.
        """
        raise NotImplementedError("BinaryTreeMechanism.observe")

    def query(self) -> StreamOutput:
        """Query the current noisy prefix sum without new observation."""
        raise NotImplementedError("BinaryTreeMechanism.query")

    def privacy_spent(self) -> float:
        """Return total privacy budget spent."""
        raise NotImplementedError("BinaryTreeMechanism.privacy_spent")

    def reset(self) -> None:
        """Reset the mechanism."""
        self._state = StreamState()

    @property
    def state(self) -> StreamState:
        """Current mechanism state."""
        return self._state

    def summarize(self) -> StreamSummary:
        """Return summary statistics of the streaming session."""
        raise NotImplementedError("BinaryTreeMechanism.summarize")


class FactorisationMechanism:
    """Matrix factorisation mechanism for streaming workloads.

    Factors the workload matrix W = LR and adds noise in the R space
    for optimal error under the workload.
    """

    def __init__(
        self,
        workload: npt.NDArray[np.float64],
        config: Optional[StreamConfig] = None,
    ) -> None:
        self.workload = np.asarray(workload, dtype=np.float64)
        self.config = config or StreamConfig()
        self._state = StreamState()

    def observe(self, value: float) -> StreamOutput:
        """Process a new value and return the noisy workload answer."""
        raise NotImplementedError("FactorisationMechanism.observe")

    def query(self) -> StreamOutput:
        """Query the current noisy workload answer."""
        raise NotImplementedError("FactorisationMechanism.query")

    def privacy_spent(self) -> float:
        """Return total privacy budget spent."""
        raise NotImplementedError("FactorisationMechanism.privacy_spent")

    def reset(self) -> None:
        """Reset the mechanism."""
        self._state = StreamState()

    @property
    def state(self) -> StreamState:
        return self._state


class SparseVectorMechanism:
    """Sparse vector technique for private threshold queries.

    Answers "is query(stream) above threshold?" queries with logarithmic
    privacy cost in the number of queries.
    """

    def __init__(
        self,
        stream_config: Optional[StreamConfig] = None,
        svt_config: Optional[SparseVectorConfig] = None,
    ) -> None:
        self.stream_config = stream_config or StreamConfig()
        self.svt_config = svt_config or SparseVectorConfig()
        self._state = StreamState()

    def test(self, query_value: float) -> SparseVectorOutput:
        """Test whether a query value exceeds the noisy threshold.

        Args:
            query_value: The true query answer to test.

        Returns:
            SparseVectorOutput with above/below threshold decision.
        """
        raise NotImplementedError("SparseVectorMechanism.test")

    def privacy_spent(self) -> float:
        """Return total privacy budget spent."""
        raise NotImplementedError("SparseVectorMechanism.privacy_spent")

    @property
    def halted(self) -> bool:
        """Whether the mechanism has halted."""
        raise NotImplementedError("SparseVectorMechanism.halted")

    def reset(self) -> None:
        """Reset the mechanism."""
        self._state = StreamState()


class SlidingWindowMechanism:
    """Sliding window DP mechanism with bounded memory.

    Maintains privacy guarantees while only considering the most recent
    W observations.
    """

    def __init__(self, config: Optional[StreamConfig] = None) -> None:
        self.config = config or StreamConfig()
        if self.config.window_size is None:
            self.config.window_size = 100
        self._state = StreamState()

    def observe(self, value: float) -> StreamOutput:
        """Process a new value and return noisy windowed aggregate."""
        raise NotImplementedError("SlidingWindowMechanism.observe")

    def query(self) -> StreamOutput:
        """Query the current noisy windowed aggregate."""
        raise NotImplementedError("SlidingWindowMechanism.query")

    def privacy_spent(self) -> float:
        """Return total privacy budget spent."""
        raise NotImplementedError("SlidingWindowMechanism.privacy_spent")

    def reset(self) -> None:
        """Reset the mechanism."""
        self._state = StreamState()

    @property
    def state(self) -> StreamState:
        return self._state


class StreamAccountant:
    """Privacy accounting for streaming DP mechanisms.

    Tracks cumulative privacy budget across multiple streaming
    mechanism operations using advanced composition.
    """

    def __init__(
        self,
        total_budget: PrivacyBudget,
        privacy_notion: PrivacyNotion = PrivacyNotion.PURE_DP,
    ) -> None:
        self.total_budget = total_budget
        self.privacy_notion = privacy_notion
        self._spent_epsilon: float = 0.0
        self._spent_delta: float = 0.0

    def add_observation(self, epsilon: float, delta: float = 0.0) -> bool:
        """Record privacy cost of an observation.

        Args:
            epsilon: ε cost of this observation.
            delta: δ cost of this observation.

        Returns:
            True if budget is still available, False if exhausted.
        """
        raise NotImplementedError("StreamAccountant.add_observation")

    def budget_remaining(self) -> PrivacyBudget:
        """Return the remaining privacy budget."""
        raise NotImplementedError("StreamAccountant.budget_remaining")

    @property
    def is_exhausted(self) -> bool:
        """Whether the privacy budget is exhausted."""
        return (
            self._spent_epsilon >= self.total_budget.epsilon
            or self._spent_delta >= self.total_budget.delta
        )

    def __repr__(self) -> str:
        return (
            f"StreamAccountant(spent_ε={self._spent_epsilon:.4f}/"
            f"{self.total_budget.epsilon}, "
            f"spent_δ={self._spent_delta:.6f}/{self.total_budget.delta})"
        )


# ---------------------------------------------------------------------------
# Submodule imports
# ---------------------------------------------------------------------------

from dp_forge.streaming.binary_tree import (
    TreeNode,
    TreeConstruction,
    NoiseAllocation,
    RangeQuery,
    MatrixFactorizationTree,
    HybridTree,
)
# Override the stub BinaryTreeMechanism with the full implementation
from dp_forge.streaming.binary_tree import (  # noqa: F811
    BinaryTreeMechanism as BinaryTreeMechanism,
)

from dp_forge.streaming.continual import (
    ContinualCounter,
    EventLevelDP,
    UserLevelDP,
    WindowedDP,
    DecayingPrivacy,
    PanPrivacy,
    StreamingHistogram,
)

from dp_forge.streaming.factorization import (
    MatrixMechanism,
    FactorizationOptimizer,
    LowerTriangular,
    ToeplitzFactorization,
    BandedFactorization,
    OnlineFactorization,
    FactorizationError,
)

# Override the stub StreamAccountant with the full implementation
from dp_forge.streaming.online_synthesis import (  # noqa: F811
    OnlineSynthesizer,
    AdaptiveBudgetAllocator,
    StreamAccountant as StreamAccountant,
    RegretMinimizer,
    ExponentialMechanism,
    PrivacyFilter,
    AboveThreshold,
)


__all__ = [
    # Enums
    "StreamMechanismType",
    "NoiseSchedule",
    "TreeStructure",
    "ThresholdPolicy",
    # Config
    "StreamConfig",
    "SparseVectorConfig",
    # Data types
    "StreamState",
    "StreamOutput",
    "SparseVectorOutput",
    "StreamSummary",
    # Protocols
    "StreamingMechanism",
    # Classes (original stubs)
    "BinaryTreeMechanism",
    "FactorisationMechanism",
    "SparseVectorMechanism",
    "SlidingWindowMechanism",
    "StreamAccountant",
    # Binary tree
    "TreeNode",
    "TreeConstruction",
    "NoiseAllocation",
    "RangeQuery",
    "MatrixFactorizationTree",
    "HybridTree",
    # Continual observation
    "ContinualCounter",
    "EventLevelDP",
    "UserLevelDP",
    "WindowedDP",
    "DecayingPrivacy",
    "PanPrivacy",
    "StreamingHistogram",
    # Matrix factorization
    "MatrixMechanism",
    "FactorizationOptimizer",
    "LowerTriangular",
    "ToeplitzFactorization",
    "BandedFactorization",
    "OnlineFactorization",
    "FactorizationError",
    # Online synthesis
    "OnlineSynthesizer",
    "AdaptiveBudgetAllocator",
    "RegretMinimizer",
    "ExponentialMechanism",
    "PrivacyFilter",
    "AboveThreshold",
]

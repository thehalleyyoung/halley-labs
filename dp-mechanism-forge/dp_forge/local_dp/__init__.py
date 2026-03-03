"""
Local differential privacy mechanisms.

This package implements local DP (LDP) mechanisms where each user
perturbs their own data before sending it to an (untrusted) aggregator.
Unlike central DP, the server never sees raw user data.

Key mechanisms:
- **Randomized response**: Classic binary/categorical randomized response
  (Warner 1965).
- **RAPPOR**: Randomized Aggregatable Privacy-Preserving Ordinal Response
  (Erlingsson et al. 2014) using Bloom filters.
- **Optimal unary encoding (OUE)**: Optimised unary encoding for frequency
  estimation (Wang et al. 2017).
- **Local hashing**: Hash-based LDP for large domains.
- **Piecewise mechanism**: Optimal LDP for numerical data.
- **Frequency oracle**: Generic frequency estimation from LDP reports.

Architecture:
    1. **LDPEncoder** — Encodes a value into a local DP report.
    2. **LDPAggregator** — Aggregates LDP reports from multiple users
       to estimate frequencies/statistics.
    3. **FrequencyOracle** — Unified interface for frequency estimation.
    4. **MeanEstimator** — Unified interface for mean estimation.
    5. **LDPOptimizer** — Optimises LDP mechanism parameters for a given
       utility-privacy trade-off.

Example::

    from dp_forge.local_dp import RandomizedResponse, FrequencyOracle

    # Each user locally perturbs their binary value
    rr = RandomizedResponse(epsilon=2.0, domain_size=2)
    report = rr.encode(true_value=1)

    # Aggregator collects reports from all users
    oracle = FrequencyOracle(mechanism=rr, num_users=10000)
    oracle.add_reports(all_reports)
    estimated_freq = oracle.estimate_frequency(value=1)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt

from dp_forge.types import (
    LocalDPBudget,
    PrivacyBudget,
    PrivacyNotion,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class LDPMechanismType(Enum):
    """Types of local DP mechanisms."""

    RANDOMIZED_RESPONSE = auto()
    RAPPOR = auto()
    OPTIMAL_UNARY_ENCODING = auto()
    LOCAL_HASHING = auto()
    PIECEWISE = auto()
    SQUARE_WAVE = auto()
    HYBRID = auto()

    def __repr__(self) -> str:
        return f"LDPMechanismType.{self.name}"


class EncodingType(Enum):
    """Types of encoding used in LDP mechanisms."""

    UNARY = auto()
    BINARY = auto()
    BLOOM_FILTER = auto()
    HASH = auto()
    DIRECT = auto()

    def __repr__(self) -> str:
        return f"EncodingType.{self.name}"


class AggregationType(Enum):
    """Types of aggregation for LDP reports."""

    FREQUENCY = auto()
    MEAN = auto()
    HISTOGRAM = auto()
    HEAVY_HITTER = auto()
    DISTRIBUTION = auto()

    def __repr__(self) -> str:
        return f"AggregationType.{self.name}"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LDPConfig:
    """Configuration for local DP mechanisms.

    Attributes:
        mechanism_type: Type of LDP mechanism.
        epsilon: Privacy parameter ε.
        domain_size: Size of the input domain.
        encoding_type: Encoding strategy.
        num_hash_functions: Number of hash functions (for RAPPOR/hashing).
        bloom_filter_size: Size of Bloom filter (for RAPPOR).
        seed: Random seed for reproducibility.
        verbose: Verbosity level.
    """

    mechanism_type: LDPMechanismType = LDPMechanismType.RANDOMIZED_RESPONSE
    epsilon: float = 1.0
    domain_size: int = 2
    encoding_type: EncodingType = EncodingType.DIRECT
    num_hash_functions: int = 1
    bloom_filter_size: int = 64
    seed: Optional[int] = None
    verbose: int = 1

    def __post_init__(self) -> None:
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")
        if self.domain_size < 2:
            raise ValueError(f"domain_size must be >= 2, got {self.domain_size}")
        if self.num_hash_functions < 1:
            raise ValueError(f"num_hash_functions must be >= 1, got {self.num_hash_functions}")
        if self.bloom_filter_size < 1:
            raise ValueError(f"bloom_filter_size must be >= 1, got {self.bloom_filter_size}")

    def __repr__(self) -> str:
        return (
            f"LDPConfig(type={self.mechanism_type.name}, ε={self.epsilon}, "
            f"domain={self.domain_size})"
        )


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------


@dataclass
class LDPReport:
    """A local DP report from a single user.

    Attributes:
        user_id: Identifier for the reporting user.
        encoded_value: The perturbed/encoded value.
        mechanism_type: Which LDP mechanism produced this report.
        domain_size: Size of the input domain.
    """

    user_id: int
    encoded_value: Union[int, npt.NDArray[np.int8], npt.NDArray[np.float64]]
    mechanism_type: LDPMechanismType
    domain_size: int

    def __repr__(self) -> str:
        if isinstance(self.encoded_value, (int, np.integer)):
            val = str(self.encoded_value)
        else:
            val = f"array(len={len(self.encoded_value)})"
        return (
            f"LDPReport(user={self.user_id}, value={val}, "
            f"type={self.mechanism_type.name})"
        )


@dataclass
class FrequencyEstimate:
    """Estimated frequency distribution from LDP aggregation.

    Attributes:
        frequencies: Estimated frequency for each domain value.
        confidence_intervals: Optional confidence intervals per value.
        num_reports: Number of reports used in estimation.
        mechanism_type: LDP mechanism used.
    """

    frequencies: npt.NDArray[np.float64]
    confidence_intervals: Optional[npt.NDArray[np.float64]] = None
    num_reports: int = 0
    mechanism_type: Optional[LDPMechanismType] = None

    def __post_init__(self) -> None:
        self.frequencies = np.asarray(self.frequencies, dtype=np.float64)

    @property
    def domain_size(self) -> int:
        """Size of the domain."""
        return len(self.frequencies)

    @property
    def total_frequency(self) -> float:
        """Sum of estimated frequencies."""
        return float(np.sum(self.frequencies))

    def __repr__(self) -> str:
        return (
            f"FrequencyEstimate(domain={self.domain_size}, "
            f"reports={self.num_reports}, total={self.total_frequency:.4f})"
        )


@dataclass
class MeanEstimate:
    """Estimated mean from LDP aggregation.

    Attributes:
        mean: Estimated mean value.
        variance: Estimated variance of the estimator.
        confidence_interval: 95% confidence interval.
        num_reports: Number of reports used.
    """

    mean: float
    variance: float
    confidence_interval: Tuple[float, float]
    num_reports: int = 0

    def __post_init__(self) -> None:
        if self.variance < 0:
            raise ValueError(f"variance must be >= 0, got {self.variance}")

    @property
    def std_error(self) -> float:
        """Standard error of the mean estimate."""
        if self.num_reports == 0:
            return float("inf")
        return (self.variance / self.num_reports) ** 0.5

    def __repr__(self) -> str:
        lo, hi = self.confidence_interval
        return (
            f"MeanEstimate(mean={self.mean:.4f}, CI=[{lo:.4f}, {hi:.4f}], "
            f"n={self.num_reports})"
        )


@dataclass
class LDPAnalysis:
    """Analysis of an LDP mechanism's utility-privacy trade-off.

    Attributes:
        mechanism_type: LDP mechanism analysed.
        epsilon: Privacy parameter.
        domain_size: Domain size.
        expected_mse: Expected mean squared error per user.
        communication_bits: Bits of communication per report.
        computation_cost: Computational cost per report (relative).
    """

    mechanism_type: LDPMechanismType
    epsilon: float
    domain_size: int
    expected_mse: float
    communication_bits: int
    computation_cost: float = 1.0

    def __repr__(self) -> str:
        return (
            f"LDPAnalysis(type={self.mechanism_type.name}, ε={self.epsilon}, "
            f"MSE={self.expected_mse:.4f}, bits={self.communication_bits})"
        )


# ---------------------------------------------------------------------------
# Protocols (interfaces)
# ---------------------------------------------------------------------------


@runtime_checkable
class LDPEncoder(Protocol):
    """Protocol for LDP encoding (client-side perturbation)."""

    def encode(self, value: int) -> LDPReport:
        """Encode a value into a local DP report."""
        ...

    @property
    def epsilon(self) -> float:
        """Privacy parameter ε of this encoder."""
        ...

    @property
    def domain_size(self) -> int:
        """Size of the input domain."""
        ...


@runtime_checkable
class LDPDecoder(Protocol):
    """Protocol for LDP decoding (server-side aggregation)."""

    def aggregate(self, reports: Sequence[LDPReport]) -> FrequencyEstimate:
        """Aggregate reports into a frequency estimate."""
        ...

    def estimate_frequency(self, reports: Sequence[LDPReport], value: int) -> float:
        """Estimate frequency of a specific value."""
        ...


# ---------------------------------------------------------------------------
# Public API classes
# ---------------------------------------------------------------------------


class RandomizedResponse:
    """Classic randomized response mechanism for categorical data.

    Each user reports their true value with probability p = e^ε / (e^ε + d - 1)
    and a uniformly random value otherwise.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        domain_size: int = 2,
        seed: Optional[int] = None,
    ) -> None:
        self._epsilon = epsilon
        self._domain_size = domain_size
        self._rng = np.random.default_rng(seed)

    def encode(self, value: int, user_id: int = 0) -> LDPReport:
        """Encode a value using randomized response.

        Args:
            value: True value in [0, domain_size).
            user_id: Identifier for the user.

        Returns:
            LDPReport with the perturbed value.
        """
        raise NotImplementedError("RandomizedResponse.encode")

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def domain_size(self) -> int:
        return self._domain_size

    @property
    def truth_probability(self) -> float:
        """Probability of reporting the true value."""
        import math
        return math.exp(self._epsilon) / (math.exp(self._epsilon) + self._domain_size - 1)

    def analysis(self) -> LDPAnalysis:
        """Return utility analysis of this mechanism."""
        raise NotImplementedError("RandomizedResponse.analysis")


class RAPPOREncoder:
    """RAPPOR: Randomized Aggregatable Privacy-Preserving Ordinal Response.

    Uses Bloom filter encoding with two rounds of randomization:
    permanent randomized response (PRR) and instantaneous randomized
    response (IRR).
    """

    def __init__(self, config: Optional[LDPConfig] = None) -> None:
        self.config = config or LDPConfig(
            mechanism_type=LDPMechanismType.RAPPOR,
            encoding_type=EncodingType.BLOOM_FILTER,
        )

    def encode(self, value: int, user_id: int = 0) -> LDPReport:
        """Encode a value using RAPPOR.

        Args:
            value: True value.
            user_id: User identifier.

        Returns:
            LDPReport with Bloom filter-based perturbation.
        """
        raise NotImplementedError("RAPPOREncoder.encode")

    def encode_string(self, value: str, user_id: int = 0) -> LDPReport:
        """Encode a string value using RAPPOR with hashing.

        Args:
            value: String value to encode.
            user_id: User identifier.

        Returns:
            LDPReport with Bloom filter encoding.
        """
        raise NotImplementedError("RAPPOREncoder.encode_string")


class OptimalUnaryEncoder:
    """Optimal Unary Encoding (OUE) for frequency estimation.

    Achieves optimal variance among unary encoding LDP mechanisms.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        domain_size: int = 10,
        seed: Optional[int] = None,
    ) -> None:
        self._epsilon = epsilon
        self._domain_size = domain_size
        self._rng = np.random.default_rng(seed)

    def encode(self, value: int, user_id: int = 0) -> LDPReport:
        """Encode a value using optimal unary encoding.

        Args:
            value: True value in [0, domain_size).
            user_id: User identifier.

        Returns:
            LDPReport with unary encoding.
        """
        raise NotImplementedError("OptimalUnaryEncoder.encode")


class FrequencyOracle:
    """Server-side frequency estimation from LDP reports.

    Aggregates reports from multiple users using the appropriate
    debiasing formula for the mechanism type.
    """

    def __init__(
        self,
        mechanism_type: LDPMechanismType = LDPMechanismType.RANDOMIZED_RESPONSE,
        epsilon: float = 1.0,
        domain_size: int = 2,
    ) -> None:
        self.mechanism_type = mechanism_type
        self._epsilon = epsilon
        self._domain_size = domain_size
        self._reports: List[LDPReport] = []

    def add_report(self, report: LDPReport) -> None:
        """Add a single LDP report."""
        self._reports.append(report)

    def add_reports(self, reports: Sequence[LDPReport]) -> None:
        """Add multiple LDP reports."""
        self._reports.extend(reports)

    def estimate_frequencies(self) -> FrequencyEstimate:
        """Estimate frequency distribution from collected reports.

        Returns:
            FrequencyEstimate with debiased frequencies.
        """
        raise NotImplementedError("FrequencyOracle.estimate_frequencies")

    def estimate_frequency(self, value: int) -> float:
        """Estimate frequency of a specific value.

        Args:
            value: Value to estimate frequency for.

        Returns:
            Estimated frequency (fraction).
        """
        raise NotImplementedError("FrequencyOracle.estimate_frequency")

    @property
    def num_reports(self) -> int:
        """Number of collected reports."""
        return len(self._reports)

    def reset(self) -> None:
        """Clear all collected reports."""
        self._reports.clear()


class LDPMeanEstimator:
    """Mean estimation from LDP reports for numerical data.

    Supports piecewise mechanism (optimal for bounded numerical data)
    and Laplace mechanism variants.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
        mechanism_type: LDPMechanismType = LDPMechanismType.PIECEWISE,
    ) -> None:
        self._epsilon = epsilon
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self.mechanism_type = mechanism_type

    def encode(self, value: float, user_id: int = 0) -> LDPReport:
        """Encode a numerical value with LDP guarantee.

        Args:
            value: True value in [lower_bound, upper_bound].
            user_id: User identifier.

        Returns:
            LDPReport with perturbed value.
        """
        raise NotImplementedError("LDPMeanEstimator.encode")

    def aggregate(self, reports: Sequence[LDPReport]) -> MeanEstimate:
        """Aggregate reports to estimate the mean.

        Args:
            reports: Collection of LDP reports.

        Returns:
            MeanEstimate with estimated mean and confidence interval.
        """
        raise NotImplementedError("LDPMeanEstimator.aggregate")


class LDPOptimizer:
    """Optimise LDP mechanism parameters for a target utility-privacy trade-off."""

    def select_mechanism(
        self,
        epsilon: float,
        domain_size: int,
        num_users: int,
        aggregation_type: AggregationType = AggregationType.FREQUENCY,
    ) -> Tuple[LDPMechanismType, LDPConfig]:
        """Select the optimal LDP mechanism for the given parameters.

        Args:
            epsilon: Privacy parameter.
            domain_size: Input domain size.
            num_users: Expected number of users.
            aggregation_type: What statistic to estimate.

        Returns:
            Tuple of (best_mechanism_type, optimal_config).
        """
        raise NotImplementedError("LDPOptimizer.select_mechanism")

    def compare_mechanisms(
        self,
        epsilon: float,
        domain_size: int,
        num_users: int,
    ) -> List[LDPAnalysis]:
        """Compare all LDP mechanisms on utility metrics.

        Args:
            epsilon: Privacy parameter.
            domain_size: Input domain size.
            num_users: Expected number of users.

        Returns:
            List of LDPAnalysis, one per mechanism type, sorted by MSE.
        """
        raise NotImplementedError("LDPOptimizer.compare_mechanisms")


# ---------------------------------------------------------------------------
# Imports from submodules
# ---------------------------------------------------------------------------

from dp_forge.local_dp.randomized_response import (
    RandomizedResponse as _RR_impl,
    GeneralizedRR,
    OptimalRR,
    DirectEncoding,
    KaryRR,
    SubsetSelection,
    UnaryEncoding,
)
from dp_forge.local_dp.frequency_oracle import (
    FrequencyOracle as _FO_base,
    OLHEstimator,
    HadamardResponse,
    ProjectionEstimator,
    CMS,
    HeavyHitterDetector,
    FrequencyCalibrator,
)
from dp_forge.local_dp.mean_estimation import (
    DuchiMeanEstimator,
    PiecewiseMechanism,
    HybridMechanism,
    MultidimensionalMean,
    PrivateStochasticGradient,
    ClippedMean,
)
from dp_forge.local_dp.protocols import (
    RAPPOREncoder as _RAPPOR_impl,
    InstantaneousRAPPOR,
    LongitudinalRAPPOR,
    PrioritySampling,
    ShuffleModel,
    SecureAggregation,
    PrivateHistogram,
)

__all__ = [
    # Enums
    "LDPMechanismType",
    "EncodingType",
    "AggregationType",
    # Config
    "LDPConfig",
    # Data types
    "LDPReport",
    "FrequencyEstimate",
    "MeanEstimate",
    "LDPAnalysis",
    # Protocols
    "LDPEncoder",
    "LDPDecoder",
    # Stub classes (kept for backward compat)
    "RandomizedResponse",
    "RAPPOREncoder",
    "OptimalUnaryEncoder",
    "FrequencyOracle",
    "LDPMeanEstimator",
    "LDPOptimizer",
    # randomized_response module
    "GeneralizedRR",
    "OptimalRR",
    "DirectEncoding",
    "KaryRR",
    "SubsetSelection",
    "UnaryEncoding",
    # frequency_oracle module
    "OLHEstimator",
    "HadamardResponse",
    "ProjectionEstimator",
    "CMS",
    "HeavyHitterDetector",
    "FrequencyCalibrator",
    # mean_estimation module
    "DuchiMeanEstimator",
    "PiecewiseMechanism",
    "HybridMechanism",
    "MultidimensionalMean",
    "PrivateStochasticGradient",
    "ClippedMean",
    # protocols module
    "InstantaneousRAPPOR",
    "LongitudinalRAPPOR",
    "PrioritySampling",
    "ShuffleModel",
    "SecureAggregation",
    "PrivateHistogram",
]

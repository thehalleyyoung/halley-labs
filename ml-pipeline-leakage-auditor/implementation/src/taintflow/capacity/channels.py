"""
taintflow.capacity.channels – Information-theoretic channel capacity models.

Statistical operations in ML pipelines act as lossy information-theoretic
channels whose capacity can be bounded.  This module implements the core
channel abstractions: Gaussian, discrete, and binary channels, along with
composition rules (sequential / parallel) and the data-processing inequality.

Key identity:  C_gaussian = 0.5 * log2(1 + SNR)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from taintflow.core.types import OpType, Origin


# ===================================================================
#  Constants
# ===================================================================

_LOG2_E: float = math.log2(math.e)
_EPSILON: float = 1e-15
_DEFAULT_B_MAX: float = 64.0


# ===================================================================
#  Supporting dataclasses
# ===================================================================


class ChannelKind(Enum):
    """Classification of information-theoretic channels."""

    GAUSSIAN = auto()
    DISCRETE = auto()
    BINARY = auto()
    DETERMINISTIC = auto()
    ERASURE = auto()
    COMPOSED_SEQUENTIAL = auto()
    COMPOSED_PARALLEL = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class ChannelCapacityBound:
    """A capacity bound with associated tightness metadata.

    Attributes:
        bits: Upper bound on mutual information in bits.
        tightness_factor: κ – multiplicative gap between bound and true
            capacity.  κ = 1 means the bound is exact.  κ = O(log n) means
            the bound is off by at most a logarithmic factor.
        is_tight: Whether the bound is provably tight (κ = 1).
        confidence: Confidence level in the bound (1.0 = provably correct).
        channel_kind: The type of channel that produced this bound.
        description: Human-readable explanation of the bound.
    """

    bits: float
    tightness_factor: float = 1.0
    is_tight: bool = False
    confidence: float = 1.0
    channel_kind: ChannelKind = ChannelKind.UNKNOWN
    description: str = ""

    def __post_init__(self) -> None:
        if self.bits < 0.0:
            object.__setattr__(self, "bits", 0.0)
        if self.tightness_factor < 1.0:
            object.__setattr__(self, "tightness_factor", 1.0)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.bits < 0.0:
            errors.append(f"bits must be >= 0, got {self.bits}")
        if self.tightness_factor < 1.0:
            errors.append(f"tightness_factor must be >= 1, got {self.tightness_factor}")
        if not (0.0 <= self.confidence <= 1.0):
            errors.append(f"confidence must be in [0,1], got {self.confidence}")
        return errors

    def attenuated(self, factor: float) -> "ChannelCapacityBound":
        """Return a bound scaled by *factor* (e.g. for sub-channels)."""
        return ChannelCapacityBound(
            bits=self.bits * max(0.0, factor),
            tightness_factor=self.tightness_factor,
            is_tight=self.is_tight,
            confidence=self.confidence,
            channel_kind=self.channel_kind,
            description=f"attenuated({factor:.4f}): {self.description}",
        )

    def with_additive_gap(self, gap: float) -> "ChannelCapacityBound":
        """Return a bound inflated by an additive gap (loosening)."""
        return ChannelCapacityBound(
            bits=self.bits + max(0.0, gap),
            tightness_factor=self.tightness_factor,
            is_tight=False,
            confidence=self.confidence,
            channel_kind=self.channel_kind,
            description=f"+{gap:.4f} gap: {self.description}",
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "bits": self.bits,
            "tightness_factor": self.tightness_factor,
            "is_tight": self.is_tight,
            "confidence": self.confidence,
            "channel_kind": self.channel_kind.name,
        }
        if self.description:
            d["description"] = self.description
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChannelCapacityBound":
        kind = ChannelKind[data.get("channel_kind", "UNKNOWN")]
        return cls(
            bits=float(data["bits"]),
            tightness_factor=float(data.get("tightness_factor", 1.0)),
            is_tight=bool(data.get("is_tight", False)),
            confidence=float(data.get("confidence", 1.0)),
            channel_kind=kind,
            description=str(data.get("description", "")),
        )

    def __repr__(self) -> str:
        tight = "tight" if self.is_tight else f"κ={self.tightness_factor:.1f}"
        return f"CapBound({self.bits:.4f} bits, {tight})"

    @staticmethod
    def zero() -> "ChannelCapacityBound":
        """The trivial zero-capacity bound (no leakage)."""
        return ChannelCapacityBound(
            bits=0.0,
            tightness_factor=1.0,
            is_tight=True,
            confidence=1.0,
            channel_kind=ChannelKind.DETERMINISTIC,
            description="zero capacity",
        )

    @staticmethod
    def infinite(b_max: float = _DEFAULT_B_MAX) -> "ChannelCapacityBound":
        """Conservative bound: assume worst-case leakage up to b_max."""
        return ChannelCapacityBound(
            bits=b_max,
            tightness_factor=float("inf"),
            is_tight=False,
            confidence=1.0,
            channel_kind=ChannelKind.UNKNOWN,
            description=f"conservative ∞ bound (capped at {b_max})",
        )


# ===================================================================
#  Channel base class
# ===================================================================


class Channel:
    """Abstract base for information-theoretic channel models.

    Subclasses must implement :meth:`capacity` which returns the Shannon
    capacity in bits, and :meth:`capacity_bound` which returns a
    :class:`ChannelCapacityBound` with tightness metadata.
    """

    def __init__(self, kind: ChannelKind, description: str = "") -> None:
        self._kind = kind
        self._description = description

    @property
    def kind(self) -> ChannelKind:
        return self._kind

    @property
    def description(self) -> str:
        return self._description

    def capacity(self, rho: float, n: int, d: int) -> float:
        """Compute channel capacity in bits.

        Args:
            rho: Test fraction ρ ∈ (0, 1).
            n: Total sample size.
            d: Dimensionality (number of features).

        Returns:
            Capacity in bits (non-negative).
        """
        raise NotImplementedError

    def capacity_bound(self, rho: float, n: int, d: int) -> ChannelCapacityBound:
        """Compute a capacity bound with tightness metadata."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._description})"


# ===================================================================
#  Gaussian channel
# ===================================================================


class GaussianChannel(Channel):
    """Additive white Gaussian noise (AWGN) channel model.

    The capacity of a Gaussian channel with signal-to-noise ratio SNR is:

        C = 0.5 * log₂(1 + SNR)

    In the ML pipeline context, when a statistic is computed on a dataset
    of n samples with test fraction ρ, the signal power is proportional to
    ρ and the noise power is proportional to 1 − ρ, giving:

        SNR(ρ, n) = n_te / n_tr = ρ / (1 − ρ)

    scaled by a per-use factor for finite-sample effects.
    """

    def __init__(
        self,
        signal_power: Optional[float] = None,
        noise_power: Optional[float] = None,
        description: str = "AWGN channel",
    ) -> None:
        super().__init__(kind=ChannelKind.GAUSSIAN, description=description)
        self._signal_power = signal_power
        self._noise_power = noise_power

    @staticmethod
    def _snr_from_test_fraction(rho: float, n: int) -> float:
        """Compute effective SNR from test fraction and sample size.

        For an estimator computed on the full dataset of size n, the test
        partition of size n_te = ρn contributes signal while the training
        partition of size n_tr = (1−ρ)n contributes noise.

        SNR = n_te / (n − n_te) = ρn / ((1 − ρ)n) = ρ / (1 − ρ)

        For finite n the effective SNR incorporates sample-size corrections:
        SNR_eff = n_te / (n − n_te) = ρ / (1 − ρ)
        """
        if rho <= 0.0 or rho >= 1.0:
            return 0.0 if rho <= 0.0 else float("inf")
        n_te = rho * n
        n_tr = n - n_te
        if n_tr <= 0:
            return float("inf")
        return n_te / n_tr

    @staticmethod
    def gaussian_capacity_from_snr(snr: float) -> float:
        """C = 0.5 * log₂(1 + SNR)."""
        if snr <= 0.0:
            return 0.0
        if math.isinf(snr):
            return float("inf")
        return 0.5 * math.log2(1.0 + snr)

    def capacity(self, rho: float, n: int, d: int) -> float:
        """Capacity per feature for a Gaussian channel parameterized by (ρ, n, d).

        Returns total capacity across d features:
            C_total = d * 0.5 * log₂(1 + SNR(ρ, n))
        """
        if rho <= 0.0:
            return 0.0
        if self._signal_power is not None and self._noise_power is not None:
            if self._noise_power <= 0.0:
                return d * _DEFAULT_B_MAX
            snr = self._signal_power / self._noise_power
        else:
            snr = self._snr_from_test_fraction(rho, n)
        cap_per_feature = self.gaussian_capacity_from_snr(snr)
        return d * cap_per_feature

    def capacity_per_feature(self, rho: float, n: int) -> float:
        """Capacity per single feature."""
        if rho <= 0.0:
            return 0.0
        if self._signal_power is not None and self._noise_power is not None:
            if self._noise_power <= 0.0:
                return _DEFAULT_B_MAX
            snr = self._signal_power / self._noise_power
        else:
            snr = self._snr_from_test_fraction(rho, n)
        return self.gaussian_capacity_from_snr(snr)

    def capacity_bound(self, rho: float, n: int, d: int) -> ChannelCapacityBound:
        """Compute Gaussian capacity bound with tightness κ = 1 (tight)."""
        bits = self.capacity(rho, n, d)
        return ChannelCapacityBound(
            bits=bits,
            tightness_factor=1.0,
            is_tight=True,
            confidence=1.0,
            channel_kind=ChannelKind.GAUSSIAN,
            description=(
                f"Gaussian: C = d·0.5·log₂(1 + ρ/(1−ρ)) "
                f"with ρ={rho:.3f}, n={n}, d={d}"
            ),
        )

    def finite_sample_capacity(self, rho: float, n: int, d: int) -> float:
        """Capacity with finite-sample correction.

        For finite n, the capacity is reduced by a correction term
        accounting for estimation error of the channel parameters:

            C_finite ≈ C − (d / (2n)) * log₂(e)

        This is the penalty for not knowing the true channel parameters.
        """
        c_asymptotic = self.capacity(rho, n, d)
        if n <= d:
            return c_asymptotic
        correction = (d / (2.0 * n)) * _LOG2_E
        return max(0.0, c_asymptotic - correction)

    def __repr__(self) -> str:
        if self._signal_power is not None:
            return f"GaussianChannel(S={self._signal_power}, N={self._noise_power})"
        return "GaussianChannel(parametric)"


# ===================================================================
#  Discrete channel
# ===================================================================


class DiscreteChannel(Channel):
    """Channel model for discrete-valued operations.

    For a channel with input alphabet of size |X| and output alphabet of
    size |Y|, the capacity is bounded by:

        C ≤ min(log₂|X|, log₂|Y|)

    For a symmetric discrete channel with crossover probability p (i.e.
    the probability that any input symbol is mapped to a different output
    symbol), a tighter bound applies:

        C = log₂|Y| − H(p) − p·log₂(|Y| − 1)

    where H(p) = −p log₂(p) − (1−p) log₂(1−p) is binary entropy.
    """

    def __init__(
        self,
        input_alphabet_size: int,
        output_alphabet_size: int,
        crossover_prob: float = 0.0,
        description: str = "discrete channel",
    ) -> None:
        super().__init__(kind=ChannelKind.DISCRETE, description=description)
        self._input_size = max(1, input_alphabet_size)
        self._output_size = max(1, output_alphabet_size)
        self._crossover_prob = max(0.0, min(1.0, crossover_prob))

    @property
    def input_alphabet_size(self) -> int:
        return self._input_size

    @property
    def output_alphabet_size(self) -> int:
        return self._output_size

    @staticmethod
    def binary_entropy(p: float) -> float:
        """Compute H(p) = −p log₂(p) − (1−p) log₂(1−p)."""
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)

    @staticmethod
    def entropy_from_distribution(probs: Sequence[float]) -> float:
        """Compute Shannon entropy H(X) = −Σ pᵢ log₂(pᵢ) from a distribution."""
        h = 0.0
        for p in probs:
            if p > _EPSILON:
                h -= p * math.log2(p)
        return h

    def noiseless_capacity(self) -> float:
        """Capacity of a noiseless discrete channel: min(log₂|X|, log₂|Y|)."""
        return min(math.log2(self._input_size), math.log2(self._output_size))

    def symmetric_channel_capacity(self) -> float:
        """Capacity of a symmetric discrete channel.

        C = log₂|Y| − H(p) − p·log₂(|Y| − 1)

        For p = 0 (noiseless), this reduces to log₂|Y| (if |X| ≥ |Y|).
        """
        p = self._crossover_prob
        if p <= 0.0:
            return self.noiseless_capacity()
        if self._output_size <= 1:
            return 0.0
        log_y = math.log2(self._output_size)
        h_p = self.binary_entropy(p)
        if self._output_size > 1:
            cap = log_y - h_p - p * math.log2(self._output_size - 1)
        else:
            cap = 0.0
        return max(0.0, cap)

    def capacity(self, rho: float, n: int, d: int) -> float:
        """Capacity for discrete channel parameterized by (ρ, n, d).

        The test fraction ρ modulates the crossover probability: when
        the test fraction is small, less information leaks.  The effective
        crossover probability is p_eff = 1 − ρ (heuristic: higher test
        fraction → lower noise → more leakage).

        Total capacity = d * C_per_feature.
        """
        if rho <= 0.0:
            return 0.0
        if self._crossover_prob > 0.0:
            cap_per_feature = self.symmetric_channel_capacity()
        else:
            p_eff = 1.0 - rho
            saved = self._crossover_prob
            self.__dict__["_crossover_prob"] = p_eff
            cap_per_feature = self.symmetric_channel_capacity()
            self.__dict__["_crossover_prob"] = saved
        return d * cap_per_feature

    def capacity_bound(self, rho: float, n: int, d: int) -> ChannelCapacityBound:
        """Capacity bound for discrete channel."""
        bits = self.capacity(rho, n, d)
        noiseless_cap = d * self.noiseless_capacity()
        if noiseless_cap > 0:
            kappa = noiseless_cap / max(bits, _EPSILON)
        else:
            kappa = 1.0
        return ChannelCapacityBound(
            bits=bits,
            tightness_factor=max(1.0, kappa),
            is_tight=(self._crossover_prob > 0.0),
            confidence=1.0,
            channel_kind=ChannelKind.DISCRETE,
            description=(
                f"Discrete: |X|={self._input_size}, |Y|={self._output_size}, "
                f"p={self._crossover_prob:.3f}, ρ={rho:.3f}"
            ),
        )

    def __repr__(self) -> str:
        return (
            f"DiscreteChannel(|X|={self._input_size}, |Y|={self._output_size}, "
            f"p={self._crossover_prob:.3f})"
        )


# ===================================================================
#  Binary channel
# ===================================================================


class BinaryChannel(Channel):
    """Binary symmetric channel (BSC) for binary classification outputs.

    The BSC has input and output alphabets {0, 1} with crossover
    probability p:  Pr(Y ≠ X) = p.

    Capacity:  C_BSC = 1 − H(p)
    where H(p) is the binary entropy function.
    """

    def __init__(
        self,
        crossover_prob: float = 0.0,
        description: str = "binary symmetric channel",
    ) -> None:
        super().__init__(kind=ChannelKind.BINARY, description=description)
        self._crossover_prob = max(0.0, min(1.0, crossover_prob))

    @property
    def crossover_probability(self) -> float:
        return self._crossover_prob

    def capacity_fixed(self) -> float:
        """Capacity of the BSC: C = 1 − H(p)."""
        return max(0.0, 1.0 - DiscreteChannel.binary_entropy(self._crossover_prob))

    def capacity(self, rho: float, n: int, d: int) -> float:
        """BSC capacity parameterized by (ρ, n, d).

        For binary classification, each output feature is a single bit.
        The effective crossover probability depends on the test fraction:

            p_eff = 0.5 * (1 − ρ)  (no leakage when ρ=0, full BSC when ρ=1)

        Total capacity = d * (1 − H(p_eff)).
        """
        if rho <= 0.0:
            return 0.0
        if self._crossover_prob > 0.0:
            p_eff = self._crossover_prob
        else:
            p_eff = 0.5 * (1.0 - rho)
        cap_per_bit = max(0.0, 1.0 - DiscreteChannel.binary_entropy(p_eff))
        return d * cap_per_bit

    def capacity_bound(self, rho: float, n: int, d: int) -> ChannelCapacityBound:
        """BSC capacity bound – tight by definition."""
        bits = self.capacity(rho, n, d)
        return ChannelCapacityBound(
            bits=bits,
            tightness_factor=1.0,
            is_tight=True,
            confidence=1.0,
            channel_kind=ChannelKind.BINARY,
            description=(
                f"BSC: C = d·(1 − H(p)), p={self._crossover_prob:.3f}, "
                f"ρ={rho:.3f}, d={d}"
            ),
        )

    def error_exponent(self, rate: float) -> float:
        """Reliability function (random coding exponent) for the BSC.

        E_r(R) = max_{0≤s≤1} [E_0(s) − sR]
        For BSC: E_0(s) = −log₂(p^{1/(1+s)} + (1−p)^{1/(1+s)})^{1+s}

        Simplified: uses the sphere-packing bound approximation.
        """
        c = self.capacity_fixed()
        if rate >= c or rate <= 0.0:
            return 0.0
        p = self._crossover_prob
        if p <= 0.0 or p >= 0.5:
            return 0.0
        return c - rate

    def __repr__(self) -> str:
        return f"BinaryChannel(p={self._crossover_prob:.4f})"


# ===================================================================
#  Data-Processing Inequality
# ===================================================================


class DataProcessingInequality:
    """Implements the data-processing inequality for Markov chains.

    For a Markov chain X → Y → Z:
        I(X; Z) ≤ min(I(X; Y), I(Y; Z))

    This is the fundamental tool for bounding information leakage through
    composed pipeline stages.
    """

    @staticmethod
    def apply(
        i_xy: float,
        i_yz: float,
    ) -> float:
        """Apply the DPI to bound I(X; Z) given I(X; Y) and I(Y; Z).

        For X → Y → Z:  I(X; Z) ≤ min(I(X; Y), I(Y; Z)).
        """
        return min(i_xy, i_yz)

    @staticmethod
    def apply_chain(mutual_informations: Sequence[float]) -> float:
        """Apply the DPI to a chain X₁ → X₂ → ... → Xₖ.

        The mutual information between any two non-adjacent variables is
        bounded by the minimum of all intermediate mutual informations:

            I(X₁; Xₖ) ≤ min(I(X₁; X₂), I(X₂; X₃), ..., I(Xₖ₋₁; Xₖ))
        """
        if not mutual_informations:
            return 0.0
        return min(mutual_informations)

    @staticmethod
    def apply_bounds(
        bounds: Sequence[ChannelCapacityBound],
    ) -> ChannelCapacityBound:
        """Apply the DPI to a sequence of capacity bounds.

        For X₁ → X₂ → ... → Xₖ, the end-to-end capacity is bounded
        by the minimum link capacity (bottleneck).
        """
        if not bounds:
            return ChannelCapacityBound.zero()
        min_bound = min(bounds, key=lambda b: b.bits)
        max_kappa = max(b.tightness_factor for b in bounds)
        all_tight = all(b.is_tight for b in bounds)
        min_conf = min(b.confidence for b in bounds)
        descriptions = [b.description for b in bounds if b.description]
        return ChannelCapacityBound(
            bits=min_bound.bits,
            tightness_factor=max_kappa,
            is_tight=all_tight,
            confidence=min_conf,
            channel_kind=ChannelKind.COMPOSED_SEQUENTIAL,
            description=(
                f"DPI chain (min of {len(bounds)} links): "
                + "; ".join(descriptions[:3])
            ),
        )


# ===================================================================
#  Channel composition
# ===================================================================


class ChannelComposition:
    """Compose channels sequentially (series) or in parallel.

    Sequential composition (Markov chain X → Y → Z):
        Capacity bounded by data-processing inequality:
        C_seq ≤ min(C_XY, C_YZ)

    Parallel composition (independent channels):
        Total capacity is the sum:
        C_par = C₁ + C₂ + ... + Cₖ
    """

    @staticmethod
    def sequential(channels: Sequence[Channel], rho: float, n: int, d: int) -> ChannelCapacityBound:
        """Compose channels sequentially using the data-processing inequality.

        For X → Y → Z (two stages), the capacity is bounded by the
        bottleneck:  C ≤ min(C₁, C₂).
        """
        if not channels:
            return ChannelCapacityBound.zero()
        bounds = [ch.capacity_bound(rho, n, d) for ch in channels]
        return DataProcessingInequality.apply_bounds(bounds)

    @staticmethod
    def parallel(channels: Sequence[Channel], rho: float, n: int, d: int) -> ChannelCapacityBound:
        """Compose independent parallel channels (chain rule).

        For independent channels, total capacity is additive:
        C_total = Σᵢ Cᵢ

        This models parallel feature processing where each feature goes
        through its own independent channel.
        """
        if not channels:
            return ChannelCapacityBound.zero()
        bounds = [ch.capacity_bound(rho, n, d) for ch in channels]
        total_bits = sum(b.bits for b in bounds)
        max_kappa = max(b.tightness_factor for b in bounds)
        all_tight = all(b.is_tight for b in bounds)
        min_conf = min(b.confidence for b in bounds)
        return ChannelCapacityBound(
            bits=total_bits,
            tightness_factor=max_kappa,
            is_tight=all_tight,
            confidence=min_conf,
            channel_kind=ChannelKind.COMPOSED_PARALLEL,
            description=f"parallel sum of {len(channels)} channels",
        )

    @staticmethod
    def sequential_from_bounds(
        bounds: Sequence[ChannelCapacityBound],
    ) -> ChannelCapacityBound:
        """Sequential composition directly from precomputed bounds."""
        return DataProcessingInequality.apply_bounds(bounds)

    @staticmethod
    def parallel_from_bounds(
        bounds: Sequence[ChannelCapacityBound],
    ) -> ChannelCapacityBound:
        """Parallel composition directly from precomputed bounds."""
        if not bounds:
            return ChannelCapacityBound.zero()
        total_bits = sum(b.bits for b in bounds)
        max_kappa = max(b.tightness_factor for b in bounds)
        all_tight = all(b.is_tight for b in bounds)
        min_conf = min(b.confidence for b in bounds)
        descriptions = [b.description for b in bounds if b.description]
        return ChannelCapacityBound(
            bits=total_bits,
            tightness_factor=max_kappa,
            is_tight=all_tight,
            confidence=min_conf,
            channel_kind=ChannelKind.COMPOSED_PARALLEL,
            description=(
                f"parallel sum of {len(bounds)} bounds: "
                + "; ".join(descriptions[:3])
            ),
        )

    @staticmethod
    def feedback_composition(
        forward: ChannelCapacityBound,
        feedback: ChannelCapacityBound,
    ) -> ChannelCapacityBound:
        """Capacity with feedback does not increase capacity for memoryless channels.

        Cover & Thomas Theorem 7.12.1: Feedback does not increase the
        capacity of a discrete memoryless channel.

        Returns the forward channel capacity (feedback is ignored).
        """
        return ChannelCapacityBound(
            bits=forward.bits,
            tightness_factor=forward.tightness_factor,
            is_tight=forward.is_tight,
            confidence=forward.confidence,
            channel_kind=forward.channel_kind,
            description=f"feedback ignored: {forward.description}",
        )


# ===================================================================
#  Parametric capacity computations
# ===================================================================


def capacity_mean_statistic(rho: float, n: int, d: int) -> ChannelCapacityBound:
    """Capacity bound for sample mean computed on the full dataset.

    The sample mean x̄ = (1/n) Σᵢ xᵢ has capacity:

        C_mean ≤ d · 0.5 · log₂(1 + n_te/(n − n_te))

    per feature, where n_te = ρn.  This is the Gaussian channel with
    SNR = n_te / (n − n_te) = ρ / (1 − ρ).

    Tight for Gaussian data (κ = 1).
    """
    if rho <= 0.0 or n <= 0 or d <= 0:
        return ChannelCapacityBound.zero()
    n_te = rho * n
    n_tr = n - n_te
    if n_tr <= 0:
        return ChannelCapacityBound.infinite()
    snr = n_te / n_tr
    cap_per_feature = 0.5 * math.log2(1.0 + snr)
    total = d * cap_per_feature
    return ChannelCapacityBound(
        bits=total,
        tightness_factor=1.0,
        is_tight=True,
        confidence=1.0,
        channel_kind=ChannelKind.GAUSSIAN,
        description=(
            f"mean: C = d·0.5·log₂(1 + ρ/(1−ρ)), "
            f"ρ={rho:.3f}, n={n}, d={d}"
        ),
    )


def capacity_variance_statistic(rho: float, n: int, d: int) -> ChannelCapacityBound:
    """Capacity bound for sample variance computed on the full dataset.

    The sample variance s² = (1/(n−1)) Σᵢ (xᵢ − x̄)² has the same
    asymptotic capacity as the mean for Gaussian data because the
    variance is a sufficient statistic for the scale parameter:

        C_var ≤ d · 0.5 · log₂(1 + n_te/(n − n_te))

    Tight for Gaussian data with κ = 1.
    """
    return ChannelCapacityBound(
        bits=capacity_mean_statistic(rho, n, d).bits,
        tightness_factor=1.0,
        is_tight=True,
        confidence=1.0,
        channel_kind=ChannelKind.GAUSSIAN,
        description=(
            f"variance: same Gaussian capacity as mean, "
            f"ρ={rho:.3f}, n={n}, d={d}"
        ),
    )


def capacity_sum_statistic(rho: float, n: int, d: int) -> ChannelCapacityBound:
    """Capacity bound for sum statistic.

    sum(X) = n · mean(X), so the capacity is identical to the mean.
    A deterministic scaling does not change mutual information.
    """
    bound = capacity_mean_statistic(rho, n, d)
    return ChannelCapacityBound(
        bits=bound.bits,
        tightness_factor=1.0,
        is_tight=True,
        confidence=1.0,
        channel_kind=ChannelKind.GAUSSIAN,
        description=f"sum: same capacity as mean (deterministic scaling), ρ={rho:.3f}",
    )


def capacity_count_statistic(rho: float, n: int, d: int = 1) -> ChannelCapacityBound:
    """Capacity bound for count (row count) statistic.

    count(X) is a deterministic function of the dataset shape.  If the
    test/train split is known, count leaks 0 bits.  If the split is
    unknown but the total count is computed on the mixed dataset, the
    count itself reveals the dataset size but not the partition membership.

    For the leakage perspective: count leaks at most log₂(n+1) bits
    (the information needed to specify a count in {0, ..., n}), but
    typically this is harmless.  We bound it as for the mean.
    """
    if rho <= 0.0 or n <= 0:
        return ChannelCapacityBound.zero()
    n_te = rho * n
    n_tr = n - n_te
    if n_tr <= 0:
        return ChannelCapacityBound.infinite()
    snr = n_te / n_tr
    bits = 0.5 * math.log2(1.0 + snr)
    return ChannelCapacityBound(
        bits=bits,
        tightness_factor=1.0,
        is_tight=True,
        confidence=1.0,
        channel_kind=ChannelKind.GAUSSIAN,
        description=f"count: C = 0.5·log₂(1 + ρ/(1−ρ)), ρ={rho:.3f}, n={n}",
    )


def capacity_median_statistic(rho: float, n: int, d: int) -> ChannelCapacityBound:
    """Capacity bound for sample median.

    The median is a rank statistic.  Its capacity is bounded by the
    Gaussian channel capacity multiplied by a logarithmic factor:

        C_median ≤ d · O(log n) · 0.5 · log₂(1 + ρ/(1−ρ))

    The κ = O(log n) factor arises because the median depends on order
    statistics which provide more information than linear statistics.
    """
    if rho <= 0.0 or n <= 0 or d <= 0:
        return ChannelCapacityBound.zero()
    n_te = rho * n
    n_tr = n - n_te
    if n_tr <= 0:
        return ChannelCapacityBound.infinite()
    snr = n_te / n_tr
    cap_per_feature = 0.5 * math.log2(1.0 + snr)
    kappa = max(1.0, math.log2(max(2, n)))
    total = d * kappa * cap_per_feature
    return ChannelCapacityBound(
        bits=total,
        tightness_factor=kappa,
        is_tight=False,
        confidence=0.95,
        channel_kind=ChannelKind.GAUSSIAN,
        description=(
            f"median: C ≤ d·O(log n)·C_gauss, "
            f"κ={kappa:.2f}, ρ={rho:.3f}, n={n}, d={d}"
        ),
    )


def capacity_quantile_statistic(
    rho: float, n: int, d: int, q: float = 0.5
) -> ChannelCapacityBound:
    """Capacity bound for sample quantile at level q.

    Quantiles are rank statistics.  The capacity depends on the quantile
    level q and the sample size n.  The asymptotic variance of the q-th
    quantile is q(1−q) / (n · f(F⁻¹(q))²) where f is the density.

    For a general bound (density-free):
        C_quantile ≤ d · O(log n) · 0.5 · log₂(1 + ρ/(1−ρ))

    Same tightness factor as the median: κ = O(log n).
    """
    if rho <= 0.0 or n <= 0 or d <= 0:
        return ChannelCapacityBound.zero()
    n_te = rho * n
    n_tr = n - n_te
    if n_tr <= 0:
        return ChannelCapacityBound.infinite()
    snr = n_te / n_tr
    cap_per_feature = 0.5 * math.log2(1.0 + snr)
    kappa = max(1.0, math.log2(max(2, n)))
    q_factor = 1.0
    if 0.0 < q < 1.0:
        q_factor = 1.0 / max(_EPSILON, 2.0 * math.sqrt(q * (1.0 - q)))
        q_factor = min(q_factor, kappa)
    total = d * kappa * cap_per_feature
    return ChannelCapacityBound(
        bits=total,
        tightness_factor=kappa,
        is_tight=False,
        confidence=0.95,
        channel_kind=ChannelKind.GAUSSIAN,
        description=(
            f"quantile(q={q:.2f}): C ≤ d·O(log n)·C_gauss, "
            f"κ={kappa:.2f}, ρ={rho:.3f}, n={n}, d={d}"
        ),
    )


def capacity_pca_statistic(rho: float, n: int, d: int) -> ChannelCapacityBound:
    """Capacity bound for PCA / SVD computed on the full dataset.

    PCA computes the sample covariance matrix, which is a d×d matrix with
    d(d+1)/2 free parameters (symmetric).  The capacity is bounded by:

        C_pca ≤ d² · C_cov(n_te, n)

    where C_cov is the per-entry covariance capacity.  The tightness
    factor is κ = O(log d) because PCA uses only the top eigenvectors.

        C_cov = 0.5 · log₂(1 + ρ/(1−ρ))

    Total: C_pca ≤ d² · 0.5 · log₂(1 + ρ/(1−ρ)), κ = O(log d).
    """
    if rho <= 0.0 or n <= 0 or d <= 0:
        return ChannelCapacityBound.zero()
    n_te = rho * n
    n_tr = n - n_te
    if n_tr <= 0:
        return ChannelCapacityBound.infinite()
    snr = n_te / n_tr
    c_cov = 0.5 * math.log2(1.0 + snr)
    total = d * d * c_cov
    kappa = max(1.0, math.log2(max(2, d)))
    return ChannelCapacityBound(
        bits=total,
        tightness_factor=kappa,
        is_tight=False,
        confidence=0.95,
        channel_kind=ChannelKind.GAUSSIAN,
        description=(
            f"PCA/SVD: C ≤ d²·C_cov, d²={d*d}, "
            f"κ=O(log d)={kappa:.2f}, ρ={rho:.3f}, n={n}"
        ),
    )


def capacity_groupby_mean(
    rho: float,
    n: int,
    d: int,
    n_groups: int,
    group_entropy: Optional[float] = None,
) -> ChannelCapacityBound:
    """Capacity bound for GroupBy.transform('mean').

    Each group mean leaks at most H(group_key) bits per group, where
    H(group_key) is the entropy of the group key distribution.

    If group entropy is not given, we use the worst case: log₂(n_groups).

        C_groupby ≤ d · H(group_key)

    Tight with κ = O(1).
    """
    if rho <= 0.0 or n <= 0 or d <= 0 or n_groups <= 0:
        return ChannelCapacityBound.zero()
    if group_entropy is None:
        h_key = math.log2(max(1, n_groups))
    else:
        h_key = max(0.0, group_entropy)
    total = d * h_key
    return ChannelCapacityBound(
        bits=total,
        tightness_factor=1.0,
        is_tight=True,
        confidence=1.0,
        channel_kind=ChannelKind.DISCRETE,
        description=(
            f"groupby_mean: C ≤ d·H(key), H={h_key:.2f}, "
            f"n_groups={n_groups}, d={d}, ρ={rho:.3f}"
        ),
    )


def capacity_target_encoding(
    rho: float,
    n: int,
    d: int,
    n_levels: int,
    target_entropy: Optional[float] = None,
) -> ChannelCapacityBound:
    """Capacity bound for target encoding (mean target per level).

    Target encoding maps categorical levels to the mean of the target
    variable within each level.  This leaks at most H(Y|group) bits
    per level, where H(Y|group) is the conditional entropy of the target
    given the group.

    Worst case: H(Y|group) = H(Y) ≤ log₂(n_levels).

        C_target ≤ d · H(Y|group) per level

    Tight with κ = O(1).
    """
    if rho <= 0.0 or n <= 0 or d <= 0 or n_levels <= 0:
        return ChannelCapacityBound.zero()
    if target_entropy is None:
        h_y = math.log2(max(1, n_levels))
    else:
        h_y = max(0.0, target_entropy)
    total = d * h_y
    return ChannelCapacityBound(
        bits=total,
        tightness_factor=1.0,
        is_tight=True,
        confidence=1.0,
        channel_kind=ChannelKind.DISCRETE,
        description=(
            f"target_encoding: C ≤ d·H(Y|group), "
            f"H={h_y:.2f}, n_levels={n_levels}, d={d}"
        ),
    )


def capacity_covariance_matrix(rho: float, n: int, d: int) -> ChannelCapacityBound:
    """Capacity bound for the full sample covariance matrix.

    The covariance matrix has d(d+1)/2 free entries (symmetric).  Each
    entry is a bilinear function of the data, so the Gaussian channel
    bound applies to each entry independently:

        C_cov ≤ d(d+1)/2 · 0.5 · log₂(1 + ρ/(1−ρ))
    """
    if rho <= 0.0 or n <= 0 or d <= 0:
        return ChannelCapacityBound.zero()
    n_te = rho * n
    n_tr = n - n_te
    if n_tr <= 0:
        return ChannelCapacityBound.infinite()
    snr = n_te / n_tr
    c_per_entry = 0.5 * math.log2(1.0 + snr)
    n_entries = d * (d + 1) // 2
    total = n_entries * c_per_entry
    return ChannelCapacityBound(
        bits=total,
        tightness_factor=1.0,
        is_tight=True,
        confidence=1.0,
        channel_kind=ChannelKind.GAUSSIAN,
        description=(
            f"covariance: C = d(d+1)/2·C_gauss, "
            f"entries={n_entries}, ρ={rho:.3f}, n={n}, d={d}"
        ),
    )

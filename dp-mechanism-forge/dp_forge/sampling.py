"""
Advanced sampling from synthesised DP mechanisms.

This module provides multiple sampling strategies for drawing from the discrete
probability distributions produced by the DP-Forge CEGIS pipeline.  Each
sampler is designed for numerical stability, efficiency, and correctness —
properties critical when the output directly affects differential privacy
guarantees.

Sampling Strategies:
    - **Alias Method** (Vose's algorithm): O(k) setup, O(1) per sample.
      Best for repeated sampling from the same row.
    - **Inverse CDF**: O(k) setup, O(log k) per sample via binary search.
      Best when quantile access is also needed.
    - **Rejection Sampling**: Flexible, works with any proposal distribution.
      Best for continuous approximations or irregular distributions.

High-Level API:
    :class:`MechanismSampler` wraps all strategies and provides convenience
    methods for noise addition, MSE/MAE estimation, and privacy auditing.

Design Principles:
    - All random number generation goes through ``numpy.random.Generator``
      for reproducibility and quality.
    - Numerically stable log-space arithmetic via :func:`_logsumexp` and
      :func:`_log_subtract` avoids underflow/overflow in tail probabilities.
    - Batch operations use vectorised NumPy for throughput.
    - Cryptographic sampling option via :class:`SecureRNG` for applications
      requiring resistance to side-channel attacks.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt

from dp_forge.types import (
    ExtractedMechanism,
    QuerySpec,
    SamplingConfig,
    SamplingMethod,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOG_EPS = 1e-300  # Floor to prevent log(0)
_PROB_TOL = 1e-12  # Tolerance for probability checks
_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Numerical utilities
# ---------------------------------------------------------------------------


def _logsumexp(a: npt.NDArray[np.float64]) -> float:
    """Numerically stable log-sum-exp.

    Computes ``log(sum(exp(a)))`` without overflow or underflow by shifting
    by the maximum element.

    Args:
        a: Array of log-space values.

    Returns:
        The log of the sum of the exponentiated values.
    """
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return -np.inf
    a_max = np.max(a)
    if not np.isfinite(a_max):
        return float(a_max)
    return float(a_max + np.log(np.sum(np.exp(a - a_max))))


def _log_subtract(log_a: float, log_b: float) -> float:
    """Compute log(exp(log_a) - exp(log_b)) in log-space.

    Requires ``log_a >= log_b`` (i.e., a >= b).  Returns ``-inf`` if a == b.

    Args:
        log_a: Log of the larger value.
        log_b: Log of the smaller value.

    Returns:
        log(a - b) computed stably.

    Raises:
        ValueError: If log_a < log_b (result would be log of a negative).
    """
    if log_a < log_b - 1e-10:
        raise ValueError(
            f"log_subtract requires log_a >= log_b, got {log_a} < {log_b}"
        )
    if log_b == -np.inf:
        return log_a
    if log_a == log_b:
        return -np.inf
    return log_a + np.log1p(-np.exp(log_b - log_a))


def _normalize_probabilities(
    probs: npt.NDArray[np.float64],
    tol: float = 1e-6,
) -> npt.NDArray[np.float64]:
    """Normalize a probability vector, clamping negatives to zero.

    Args:
        probs: Raw probability vector (may have small negative entries from
            floating-point arithmetic).
        tol: Maximum acceptable deviation from sum-to-one before raising.

    Returns:
        Normalized probability vector summing to exactly 1.0.

    Raises:
        ValueError: If the sum deviates from 1 by more than ``tol`` before
            normalization, indicating a structural problem rather than
            floating-point noise.
    """
    probs = np.asarray(probs, dtype=np.float64).copy()
    probs = np.maximum(probs, 0.0)
    total = probs.sum()
    if total <= 0:
        raise ValueError("Probability vector sums to zero or negative")
    if abs(total - 1.0) > tol:
        warnings.warn(
            f"Probability vector sum {total} deviates from 1.0 by "
            f"{abs(total - 1.0):.2e} (tol={tol:.2e}); renormalizing.",
            stacklevel=2,
        )
    probs /= total
    return probs


def _validate_probabilities(
    probs: npt.NDArray[np.float64],
    name: str = "probabilities",
) -> npt.NDArray[np.float64]:
    """Validate and normalize a probability vector.

    Args:
        probs: Probability vector to validate.
        name: Name for error messages.

    Returns:
        Validated, normalized probability vector.

    Raises:
        ValueError: If the vector is empty, not 1-D, or contains NaN.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {probs.shape}")
    if probs.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if np.any(np.isnan(probs)):
        raise ValueError(f"{name} contains NaN values")
    return _normalize_probabilities(probs)


# ---------------------------------------------------------------------------
# Alias Method Sampler (Vose's algorithm)
# ---------------------------------------------------------------------------


class AliasMethodSampler:
    """O(1) sampler using Vose's alias method.

    Vose's algorithm constructs two parallel arrays (``prob`` and ``alias``)
    such that each sample requires only one uniform draw and one coin flip.
    Setup is O(k) in time and space.

    The implementation follows:
        M. D. Vose, "A linear algorithm for generating random numbers with a
        given distribution," IEEE Trans. Softw. Eng., 1991.

    Attributes:
        k: Number of outcomes (support size).
        prob: Probability table for the alias method, shape (k,).
        alias: Alias table mapping outcomes to their alias, shape (k,).
        _log_probs: Log-probabilities of the original distribution (for
            :meth:`log_probability`).
        _original_probs: Original probability vector (for diagnostics).

    Example::

        >>> sampler = AliasMethodSampler()
        >>> sampler.build(np.array([0.1, 0.2, 0.3, 0.4]))
        >>> samples = sampler.sample_batch(10000)
        >>> np.bincount(samples, minlength=4) / 10000  # ≈ [0.1, 0.2, 0.3, 0.4]
    """

    def __init__(self) -> None:
        self.k: int = 0
        self.prob: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self.alias: npt.NDArray[np.int64] = np.empty(0, dtype=np.int64)
        self._log_probs: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._original_probs: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._built: bool = False

    def build(self, probabilities: npt.NDArray[np.float64]) -> AliasMethodSampler:
        """Construct alias tables from a probability vector.

        Uses Vose's linear-time algorithm.  The input probabilities are
        normalized (small negatives clamped to zero) before processing.

        Args:
            probabilities: Probability vector of length k.  Must sum to ~1.0.

        Returns:
            ``self``, for method chaining.

        Raises:
            ValueError: If probabilities are invalid (empty, wrong shape, NaN).
        """
        probs = _validate_probabilities(probabilities, "probabilities")
        k = len(probs)
        self.k = k
        self._original_probs = probs.copy()
        self._log_probs = np.where(
            probs > _LOG_EPS, np.log(probs), np.log(_LOG_EPS)
        )

        # Vose's algorithm
        self.prob = np.zeros(k, dtype=np.float64)
        self.alias = np.zeros(k, dtype=np.int64)

        scaled = probs * k  # Scale so that average = 1.0

        # Partition into "small" and "large" groups
        small: list[int] = []
        large: list[int] = []
        for i in range(k):
            if scaled[i] < 1.0:
                small.append(i)
            else:
                large.append(i)

        # Process pairs
        while small and large:
            s = small.pop()
            l = large.pop()  # noqa: E741
            self.prob[s] = scaled[s]
            self.alias[s] = l
            scaled[l] = (scaled[l] + scaled[s]) - 1.0
            if scaled[l] < 1.0:
                small.append(l)
            else:
                large.append(l)

        # Remaining items have probability ~1.0 (within floating-point error)
        for idx in large:
            self.prob[idx] = 1.0
            self.alias[idx] = idx
        for idx in small:
            self.prob[idx] = 1.0
            self.alias[idx] = idx

        self._built = True
        return self

    def _check_built(self) -> None:
        """Assert the alias tables have been constructed."""
        if not self._built:
            raise RuntimeError(
                "AliasMethodSampler.build() must be called before sampling"
            )

    def sample(self, rng: Optional[np.random.Generator] = None) -> int:
        """Draw a single sample using the alias method.

        Requires exactly two uniform random numbers: one to select the bin
        and one for the coin flip.

        Args:
            rng: NumPy random generator.  If ``None``, uses the default RNG.

        Returns:
            Integer outcome in ``[0, k)``.
        """
        self._check_built()
        if rng is None:
            rng = np.random.default_rng()
        i = rng.integers(0, self.k)
        if rng.random() < self.prob[i]:
            return int(i)
        return int(self.alias[i])

    def sample_batch(
        self,
        n: int,
        rng: Optional[np.random.Generator] = None,
    ) -> npt.NDArray[np.int64]:
        """Draw n samples using vectorised alias method.

        This is the preferred method for generating many samples, as it
        avoids Python-level loops entirely.

        Args:
            n: Number of samples to draw.
            rng: NumPy random generator.  If ``None``, uses the default RNG.

        Returns:
            Integer array of shape (n,) with outcomes in ``[0, k)``.

        Raises:
            ValueError: If n < 0.
        """
        self._check_built()
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        if n == 0:
            return np.empty(0, dtype=np.int64)
        if rng is None:
            rng = np.random.default_rng()

        bins = rng.integers(0, self.k, size=n)
        coins = rng.random(size=n)

        # Vectorised coin flip: use bin or alias
        use_alias = coins >= self.prob[bins]
        result = np.where(use_alias, self.alias[bins], bins)
        return result.astype(np.int64)

    def sample_conditional(
        self,
        condition: Callable[[int], bool],
        rng: Optional[np.random.Generator] = None,
        max_attempts: int = 100_000,
    ) -> int:
        """Draw a sample conditioned on a predicate.

        Uses rejection: repeatedly draws until ``condition(outcome)`` is True.

        Args:
            condition: Predicate that must be satisfied.
            rng: NumPy random generator.
            max_attempts: Maximum number of rejection attempts.

        Returns:
            Integer outcome satisfying ``condition``.

        Raises:
            RuntimeError: If no valid sample found within ``max_attempts``.
        """
        self._check_built()
        if rng is None:
            rng = np.random.default_rng()
        for _ in range(max_attempts):
            s = self.sample(rng)
            if condition(s):
                return s
        raise RuntimeError(
            f"Conditional sampling failed after {max_attempts} attempts. "
            f"The condition may have zero probability mass."
        )

    def log_probability(self, outcome: int) -> float:
        """Return the log-probability of a specific outcome.

        Args:
            outcome: Integer outcome in ``[0, k)``.

        Returns:
            ``log(Pr[X = outcome])``.

        Raises:
            IndexError: If outcome is out of range.
        """
        self._check_built()
        if not (0 <= outcome < self.k):
            raise IndexError(
                f"Outcome {outcome} out of range [0, {self.k})"
            )
        return float(self._log_probs[outcome])

    @property
    def original_probabilities(self) -> npt.NDArray[np.float64]:
        """Return a copy of the original probability vector."""
        self._check_built()
        return self._original_probs.copy()

    @property
    def entropy(self) -> float:
        """Shannon entropy of the distribution (in nats)."""
        self._check_built()
        p = self._original_probs
        mask = p > 0
        return -float(np.sum(p[mask] * np.log(p[mask])))

    def __repr__(self) -> str:
        if not self._built:
            return "AliasMethodSampler(not built)"
        return f"AliasMethodSampler(k={self.k})"


# ---------------------------------------------------------------------------
# Inverse CDF Sampler
# ---------------------------------------------------------------------------


class InverseCDFSampler:
    """O(log k) sampler using inverse CDF with binary search.

    Constructs a cumulative distribution function (CDF) from the probability
    vector, then uses ``numpy.searchsorted`` (binary search) to map uniform
    random draws to outcomes.

    This sampler is preferred when:
        - Quantile queries are needed.
        - The distribution is queried infrequently (amortising alias setup
          is not worthwhile).
        - Memory for two parallel arrays is a concern (CDF requires only one).

    Attributes:
        k: Number of outcomes.
        cdf: Cumulative distribution function, shape (k,).
        _log_probs: Log-probabilities of the original distribution.
        _original_probs: Original probability vector.

    Example::

        >>> sampler = InverseCDFSampler()
        >>> sampler.build(np.array([0.25, 0.25, 0.25, 0.25]))
        >>> sampler.quantile(0.5)  # Median
        1
    """

    def __init__(self) -> None:
        self.k: int = 0
        self.cdf: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._log_probs: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._original_probs: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._built: bool = False

    def build(
        self,
        probabilities: npt.NDArray[np.float64],
        grid: Optional[npt.NDArray[np.float64]] = None,
    ) -> InverseCDFSampler:
        """Construct CDF from a probability vector.

        Args:
            probabilities: Probability vector of length k.
            grid: Optional output grid values.  If provided, stored for
                quantile queries that return grid values instead of indices.

        Returns:
            ``self``, for method chaining.
        """
        probs = _validate_probabilities(probabilities, "probabilities")
        self.k = len(probs)
        self._original_probs = probs.copy()
        self._log_probs = np.where(
            probs > _LOG_EPS, np.log(probs), np.log(_LOG_EPS)
        )
        self.cdf = np.cumsum(probs)
        # Ensure CDF ends at exactly 1.0
        self.cdf[-1] = 1.0
        self._grid = grid
        self._built = True
        return self

    def _check_built(self) -> None:
        """Assert the CDF has been constructed."""
        if not self._built:
            raise RuntimeError(
                "InverseCDFSampler.build() must be called before sampling"
            )

    def sample(self, rng: Optional[np.random.Generator] = None) -> int:
        """Draw a single sample via inverse CDF.

        Args:
            rng: NumPy random generator.

        Returns:
            Integer outcome in ``[0, k)``.
        """
        self._check_built()
        if rng is None:
            rng = np.random.default_rng()
        u = rng.random()
        return int(np.searchsorted(self.cdf, u, side="left"))

    def sample_batch(
        self,
        n: int,
        rng: Optional[np.random.Generator] = None,
    ) -> npt.NDArray[np.int64]:
        """Draw n samples via vectorised inverse CDF.

        Uses ``numpy.searchsorted`` on a batch of uniform draws for efficiency.

        Args:
            n: Number of samples to draw.
            rng: NumPy random generator.

        Returns:
            Integer array of shape (n,) with outcomes in ``[0, k)``.
        """
        self._check_built()
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        if n == 0:
            return np.empty(0, dtype=np.int64)
        if rng is None:
            rng = np.random.default_rng()
        u = rng.random(size=n)
        return np.searchsorted(self.cdf, u, side="left").astype(np.int64)

    def quantile(self, p: float) -> int:
        """Return the outcome at quantile p of the distribution.

        The quantile is defined as the smallest outcome index i such that
        ``CDF(i) >= p``.

        Args:
            p: Probability in ``[0, 1]``.

        Returns:
            Integer outcome index.

        Raises:
            ValueError: If p is not in ``[0, 1]``.
        """
        self._check_built()
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Quantile p must be in [0, 1], got {p}")
        idx = int(np.searchsorted(self.cdf, p, side="left"))
        return min(idx, self.k - 1)

    def quantile_value(self, p: float) -> float:
        """Return the grid value at quantile p, if a grid was provided.

        Args:
            p: Probability in ``[0, 1]``.

        Returns:
            Grid value at the quantile.

        Raises:
            RuntimeError: If no grid was provided at build time.
        """
        self._check_built()
        if self._grid is None:
            raise RuntimeError("No grid was provided at build time")
        idx = self.quantile(p)
        return float(self._grid[idx])

    def log_probability(self, outcome: int) -> float:
        """Return log-probability of a specific outcome.

        Args:
            outcome: Integer outcome in ``[0, k)``.

        Returns:
            ``log(Pr[X = outcome])``.
        """
        self._check_built()
        if not (0 <= outcome < self.k):
            raise IndexError(f"Outcome {outcome} out of range [0, {self.k})")
        return float(self._log_probs[outcome])

    @property
    def median(self) -> int:
        """Median outcome index."""
        self._check_built()
        return self.quantile(0.5)

    @property
    def original_probabilities(self) -> npt.NDArray[np.float64]:
        """Return a copy of the original probability vector."""
        self._check_built()
        return self._original_probs.copy()

    def __repr__(self) -> str:
        if not self._built:
            return "InverseCDFSampler(not built)"
        return f"InverseCDFSampler(k={self.k})"


# ---------------------------------------------------------------------------
# Rejection Sampler
# ---------------------------------------------------------------------------


class RejectionSampler:
    """Rejection sampler for arbitrary target distributions.

    Given a target PDF ``f``, a proposal distribution with PDF ``g`` and
    sampler, and an envelope constant ``M >= max(f(x)/g(x))``, the rejection
    method produces exact samples from ``f``.

    This sampler is most useful for:
        - Continuous approximations to discrete DP mechanisms.
        - Distributions where direct sampling is infeasible.
        - Testing: comparing rejection samples against alias/CDF samples.

    Attributes:
        _target_pdf: The target probability density/mass function.
        _proposal_sampler: Callable that draws from the proposal distribution.
        _proposal_pdf: PDF of the proposal distribution.
        _M: Envelope constant.
        _total_proposed: Total proposals made (for acceptance rate tracking).
        _total_accepted: Total proposals accepted.

    Example::

        >>> import numpy as np
        >>> target = lambda x: 0.4 if x == 0 else 0.6 if x == 1 else 0.0
        >>> proposal_sample = lambda rng: rng.integers(0, 2)
        >>> proposal_pdf = lambda x: 0.5
        >>> sampler = RejectionSampler()
        >>> sampler.build(target, proposal_sample, proposal_pdf, M=1.2)
        >>> sampler.sample()
    """

    def __init__(self) -> None:
        self._target_pdf: Optional[Callable[[Any], float]] = None
        self._proposal_sampler: Optional[Callable[[np.random.Generator], Any]] = None
        self._proposal_pdf: Optional[Callable[[Any], float]] = None
        self._M: float = 1.0
        self._total_proposed: int = 0
        self._total_accepted: int = 0
        self._built: bool = False

    def build(
        self,
        target_pdf: Callable[[Any], float],
        proposal_sampler: Callable[[np.random.Generator], Any],
        proposal_pdf: Callable[[Any], float],
        M: float,
    ) -> RejectionSampler:
        """Configure the rejection sampler.

        Args:
            target_pdf: Target density/mass function ``f(x) -> float``.
            proposal_sampler: Callable ``g_sample(rng) -> x`` drawing from
                the proposal distribution.
            proposal_pdf: Proposal density/mass function ``g(x) -> float``.
            M: Envelope constant satisfying ``f(x) <= M * g(x)`` for all x
                in the support.

        Returns:
            ``self``, for method chaining.

        Raises:
            ValueError: If M <= 0.
        """
        if M <= 0:
            raise ValueError(f"Envelope constant M must be > 0, got {M}")
        self._target_pdf = target_pdf
        self._proposal_sampler = proposal_sampler
        self._proposal_pdf = proposal_pdf
        self._M = M
        self._total_proposed = 0
        self._total_accepted = 0
        self._built = True
        return self

    def _check_built(self) -> None:
        """Assert the sampler has been configured."""
        if not self._built:
            raise RuntimeError(
                "RejectionSampler.build() must be called before sampling"
            )

    def sample(
        self,
        rng: Optional[np.random.Generator] = None,
        max_attempts: int = 1_000_000,
    ) -> Any:
        """Draw a single sample via rejection.

        Args:
            rng: NumPy random generator.
            max_attempts: Maximum rejection attempts before raising.

        Returns:
            A sample from the target distribution.

        Raises:
            RuntimeError: If no sample accepted within ``max_attempts``.
        """
        self._check_built()
        assert self._proposal_sampler is not None
        assert self._proposal_pdf is not None
        assert self._target_pdf is not None
        if rng is None:
            rng = np.random.default_rng()

        for _ in range(max_attempts):
            x = self._proposal_sampler(rng)
            u = rng.random()
            self._total_proposed += 1
            g_x = self._proposal_pdf(x)
            if g_x <= 0:
                continue
            acceptance_ratio = self._target_pdf(x) / (self._M * g_x)
            if u < acceptance_ratio:
                self._total_accepted += 1
                return x
        raise RuntimeError(
            f"Rejection sampling failed after {max_attempts} attempts. "
            f"Check that M >= max(f/g) and the proposal covers the target support."
        )

    def sample_batch(
        self,
        n: int,
        rng: Optional[np.random.Generator] = None,
    ) -> list[Any]:
        """Draw n samples via rejection.

        Args:
            n: Number of samples.
            rng: NumPy random generator.

        Returns:
            List of n samples from the target distribution.
        """
        self._check_built()
        if rng is None:
            rng = np.random.default_rng()
        return [self.sample(rng=rng) for _ in range(n)]

    def acceptance_rate(self) -> float:
        """Return the empirical acceptance rate.

        Returns:
            Fraction of proposals accepted, or 0.0 if no proposals made.
        """
        if self._total_proposed == 0:
            return 0.0
        return self._total_accepted / self._total_proposed

    def theoretical_acceptance_rate(self) -> float:
        """Return ``1/M``, the theoretical acceptance rate.

        Returns:
            Expected acceptance rate assuming the envelope is tight.
        """
        self._check_built()
        return 1.0 / self._M

    def reset_counters(self) -> None:
        """Reset acceptance/rejection counters."""
        self._total_proposed = 0
        self._total_accepted = 0

    def __repr__(self) -> str:
        if not self._built:
            return "RejectionSampler(not built)"
        rate = self.acceptance_rate()
        return f"RejectionSampler(M={self._M:.2f}, accept_rate={rate:.3f})"


# ---------------------------------------------------------------------------
# Cryptographic / Secure Sampling
# ---------------------------------------------------------------------------


class SecureRNG:
    """Cryptographically secure random number generator.

    Wraps ``os.urandom`` to produce random numbers suitable for scenarios
    where the RNG state must not be predictable (e.g., production DP
    deployments where an adversary might attempt to reconstruct noise).

    Unlike ``numpy.random.Generator``, this class draws entropy from the
    OS CSPRNG (``/dev/urandom`` on Linux, ``CryptGenRandom`` on Windows).

    Example::

        >>> rng = SecureRNG()
        >>> rng.random()  # Uniform [0, 1)
        0.7341...
        >>> rng.integers(0, 10)  # Uniform integer [0, 10)
        3
    """

    def __init__(self, seed: Optional[bytes] = None) -> None:
        """Initialize the secure RNG.

        Args:
            seed: Optional seed bytes.  If provided, used to seed a
                deterministic CSPRNG (HMAC-DRBG style) for reproducibility
                during testing.  If ``None``, uses ``os.urandom``.
        """
        self._seed = seed
        self._counter: int = 0
        if seed is not None:
            self._state = hashlib.sha512(seed).digest()
        else:
            self._state = None

    def _next_bytes(self, n: int) -> bytes:
        """Generate n random bytes.

        Args:
            n: Number of bytes to generate.

        Returns:
            Random bytes from the CSPRNG.
        """
        if self._state is None:
            return os.urandom(n)
        # Deterministic mode: use HMAC-DRBG-like construction
        result = b""
        while len(result) < n:
            self._counter += 1
            h = hashlib.sha512(
                self._state + self._counter.to_bytes(8, "big")
            )
            result += h.digest()
        return result[:n]

    def random(self) -> float:
        """Generate a uniform random float in [0, 1).

        Uses 53 bits of randomness for full double-precision mantissa.

        Returns:
            Uniform float in [0, 1).
        """
        raw = int.from_bytes(self._next_bytes(8), "big")
        # Use 53 bits for double precision mantissa
        return (raw >> 11) / (1 << 53)

    def integers(self, low: int, high: int) -> int:
        """Generate a uniform random integer in [low, high).

        Uses rejection sampling to avoid modular bias.

        Args:
            low: Lower bound (inclusive).
            high: Upper bound (exclusive).

        Returns:
            Uniform integer in [low, high).

        Raises:
            ValueError: If high <= low.
        """
        if high <= low:
            raise ValueError(f"high must be > low, got high={high}, low={low}")
        range_size = high - low
        # Number of bytes needed
        n_bytes = (range_size.bit_length() + 7) // 8
        # Rejection sampling to avoid modular bias
        max_valid = (256 ** n_bytes // range_size) * range_size
        while True:
            raw = int.from_bytes(self._next_bytes(n_bytes), "big")
            if raw < max_valid:
                return low + (raw % range_size)

    def random_array(self, size: int) -> npt.NDArray[np.float64]:
        """Generate an array of uniform random floats in [0, 1).

        Args:
            size: Number of floats to generate.

        Returns:
            Array of uniform floats.
        """
        raw = self._next_bytes(size * 8)
        ints = np.frombuffer(raw, dtype=np.uint64)
        return (ints >> np.uint64(11)).astype(np.float64) / (1 << 53)

    def __repr__(self) -> str:
        mode = "seeded" if self._state is not None else "os.urandom"
        return f"SecureRNG(mode={mode})"


def constant_time_sample(
    probabilities: npt.NDArray[np.float64],
    rng: Optional[SecureRNG] = None,
) -> int:
    """Sample from a discrete distribution in constant time per outcome check.

    Iterates through all outcomes regardless of the random draw to prevent
    timing side-channels that could leak information about which outcome was
    selected.  This is critical for DP mechanisms where timing could reveal
    the noise value.

    Args:
        probabilities: Probability vector of length k.
        rng: Secure RNG instance.  If ``None``, creates one.

    Returns:
        Integer outcome in ``[0, k)``.

    Note:
        This function is intentionally NOT short-circuit optimised.
        The entire loop always executes to prevent timing attacks.
    """
    probs = _validate_probabilities(probabilities)
    if rng is None:
        rng = SecureRNG()
    u = rng.random()
    k = len(probs)

    # Constant-time selection: always iterate all k outcomes
    cumulative = 0.0
    selected = 0
    for i in range(k):
        cumulative += probs[i]
        # Branchless selection: update selected only when u first falls
        # below the cumulative sum.  The comparison is always evaluated.
        should_update = int(u < cumulative) * int(selected == 0 or cumulative - probs[i] <= u)
        # We track via a flag to avoid branching
        if cumulative > u and i >= selected:
            selected = i
    # Final clamp (accounts for floating-point edge cases)
    return min(selected, k - 1)


def sample_with_seed(
    probabilities: npt.NDArray[np.float64],
    seed: int,
) -> int:
    """Draw a reproducible sample from a discrete distribution.

    Uses the seed to create a deterministic RNG for reproducibility.
    This is useful for debugging and testing but MUST NOT be used in
    production DP deployments (use :func:`constant_time_sample` instead).

    Args:
        probabilities: Probability vector of length k.
        seed: Integer seed for deterministic sampling.

    Returns:
        Integer outcome in ``[0, k)``.
    """
    probs = _validate_probabilities(probabilities)
    rng = np.random.default_rng(seed)
    cdf = np.cumsum(probs)
    cdf[-1] = 1.0
    u = rng.random()
    return int(np.searchsorted(cdf, u, side="left"))


# ---------------------------------------------------------------------------
# Distribution utilities
# ---------------------------------------------------------------------------


class DiscreteDistribution:
    """General discrete probability distribution over integer support.

    Provides a unified interface for probability queries, sampling, and
    statistical computations on discrete distributions.

    Attributes:
        k: Number of outcomes.
        probs: Probability vector, shape (k,).
        values: Optional array of outcome values (defaults to 0, 1, ..., k-1).
        _alias_sampler: Lazily constructed alias sampler for efficient sampling.

    Example::

        >>> dist = DiscreteDistribution(
        ...     probs=np.array([0.2, 0.3, 0.5]),
        ...     values=np.array([-1, 0, 1]),
        ... )
        >>> dist.mean()
        0.3
        >>> dist.variance()
        0.61
    """

    def __init__(
        self,
        probs: npt.NDArray[np.float64],
        values: Optional[npt.NDArray[np.float64]] = None,
    ) -> None:
        """Initialize with probabilities and optional outcome values.

        Args:
            probs: Probability vector of length k.
            values: Optional outcome values.  If ``None``, uses ``[0, ..., k-1]``.

        Raises:
            ValueError: If probs/values have inconsistent shapes.
        """
        self.probs = _validate_probabilities(probs)
        self.k = len(self.probs)
        if values is not None:
            values = np.asarray(values, dtype=np.float64)
            if values.shape != (self.k,):
                raise ValueError(
                    f"values shape {values.shape} must match probs length {self.k}"
                )
            self.values = values
        else:
            self.values = np.arange(self.k, dtype=np.float64)
        self._alias_sampler: Optional[AliasMethodSampler] = None

    def _get_alias_sampler(self) -> AliasMethodSampler:
        """Lazily build and return an alias sampler."""
        if self._alias_sampler is None:
            self._alias_sampler = AliasMethodSampler()
            self._alias_sampler.build(self.probs)
        return self._alias_sampler

    def pmf(self, x: Union[int, float]) -> float:
        """Probability mass at outcome x.

        Args:
            x: Outcome value to query.

        Returns:
            Probability of outcome x, or 0 if x is not in the support.
        """
        matches = np.where(np.abs(self.values - x) < 1e-12)[0]
        if len(matches) == 0:
            return 0.0
        return float(self.probs[matches[0]])

    def log_pmf(self, x: Union[int, float]) -> float:
        """Log-probability mass at outcome x.

        Args:
            x: Outcome value.

        Returns:
            ``log(Pr[X = x])``, or ``-inf`` if x is not in the support.
        """
        p = self.pmf(x)
        if p <= 0:
            return -np.inf
        return np.log(p)

    def cdf(self, x: float) -> float:
        """Cumulative distribution function at x.

        Args:
            x: Query point.

        Returns:
            ``Pr[X <= x]``.
        """
        mask = self.values <= x + 1e-12
        return float(np.sum(self.probs[mask]))

    def sample(
        self, n: int = 1, rng: Optional[np.random.Generator] = None,
    ) -> npt.NDArray[np.float64]:
        """Draw n samples from the distribution.

        Uses the alias method for efficiency.

        Args:
            n: Number of samples.
            rng: NumPy random generator.

        Returns:
            Array of sampled values, shape (n,).
        """
        sampler = self._get_alias_sampler()
        indices = sampler.sample_batch(n, rng=rng)
        return self.values[indices]

    def mean(self) -> float:
        """Expected value E[X]."""
        return float(np.dot(self.probs, self.values))

    def variance(self) -> float:
        """Variance Var[X] = E[X^2] - E[X]^2."""
        ex = self.mean()
        ex2 = float(np.dot(self.probs, self.values ** 2))
        return ex2 - ex ** 2

    def std(self) -> float:
        """Standard deviation."""
        return math.sqrt(max(self.variance(), 0.0))

    def entropy(self) -> float:
        """Shannon entropy in nats."""
        mask = self.probs > 0
        return -float(np.sum(self.probs[mask] * np.log(self.probs[mask])))

    def mode(self) -> float:
        """Most probable outcome value."""
        return float(self.values[np.argmax(self.probs)])

    def support(self) -> npt.NDArray[np.float64]:
        """Return outcomes with positive probability."""
        mask = self.probs > _PROB_TOL
        return self.values[mask].copy()

    def __repr__(self) -> str:
        return (
            f"DiscreteDistribution(k={self.k}, "
            f"mean={self.mean():.4f}, var={self.variance():.4f})"
        )


class MixtureDistribution:
    """Mixture of discrete distributions.

    A mixture distribution ``M = sum_i w_i * D_i`` draws from component
    ``D_i`` with probability ``w_i``.

    Attributes:
        components: List of component distributions.
        weights: Mixture weights (sum to 1).
        _weight_sampler: Alias sampler over mixture weights.

    Example::

        >>> d1 = DiscreteDistribution(np.array([0.9, 0.1]))
        >>> d2 = DiscreteDistribution(np.array([0.1, 0.9]))
        >>> mix = MixtureDistribution([d1, d2], weights=np.array([0.5, 0.5]))
        >>> mix.mean()
        0.5
    """

    def __init__(
        self,
        components: List[DiscreteDistribution],
        weights: npt.NDArray[np.float64],
    ) -> None:
        """Initialize mixture distribution.

        Args:
            components: List of component distributions.
            weights: Mixture weights, must sum to 1 and match len(components).

        Raises:
            ValueError: If weights and components have different lengths.
        """
        if len(components) == 0:
            raise ValueError("At least one component is required")
        weights = _validate_probabilities(np.asarray(weights, dtype=np.float64), "weights")
        if len(weights) != len(components):
            raise ValueError(
                f"weights length {len(weights)} must match "
                f"number of components {len(components)}"
            )
        self.components = components
        self.weights = weights
        self._weight_sampler = AliasMethodSampler()
        self._weight_sampler.build(weights)

    def sample(
        self, n: int = 1, rng: Optional[np.random.Generator] = None,
    ) -> npt.NDArray[np.float64]:
        """Draw n samples from the mixture.

        First selects a component according to ``weights``, then draws from
        that component.

        Args:
            n: Number of samples.
            rng: NumPy random generator.

        Returns:
            Array of sampled values, shape (n,).
        """
        if rng is None:
            rng = np.random.default_rng()
        component_indices = self._weight_sampler.sample_batch(n, rng=rng)
        results = np.empty(n, dtype=np.float64)
        for c_idx in range(len(self.components)):
            mask = component_indices == c_idx
            count = int(np.sum(mask))
            if count > 0:
                results[mask] = self.components[c_idx].sample(count, rng=rng)
        return results

    def mean(self) -> float:
        """Expected value of the mixture."""
        return float(sum(
            w * c.mean()
            for w, c in zip(self.weights, self.components)
        ))

    def variance(self) -> float:
        """Variance of the mixture (law of total variance).

        ``Var[X] = E[Var[X|C]] + Var[E[X|C]]``
        """
        component_means = np.array([c.mean() for c in self.components])
        component_vars = np.array([c.variance() for c in self.components])
        mixture_mean = self.mean()
        # E[Var[X|C]]
        e_var = float(np.dot(self.weights, component_vars))
        # Var[E[X|C]]
        var_e = float(np.dot(self.weights, (component_means - mixture_mean) ** 2))
        return e_var + var_e

    def pmf(self, x: Union[int, float]) -> float:
        """Probability mass at outcome x."""
        return float(sum(
            w * c.pmf(x)
            for w, c in zip(self.weights, self.components)
        ))

    def __repr__(self) -> str:
        return (
            f"MixtureDistribution(n_components={len(self.components)}, "
            f"mean={self.mean():.4f})"
        )


class TruncatedDistribution:
    """Truncated discrete distribution.

    Restricts a base distribution to outcomes within ``[low, high]``,
    renormalizing the probabilities.

    Attributes:
        base: The original distribution.
        low: Lower truncation bound (inclusive).
        high: Upper truncation bound (inclusive).
        _truncated: The truncated and renormalized distribution.
    """

    def __init__(
        self,
        base: DiscreteDistribution,
        low: float = -np.inf,
        high: float = np.inf,
    ) -> None:
        """Initialize truncated distribution.

        Args:
            base: Base distribution to truncate.
            low: Lower bound (inclusive).
            high: Upper bound (inclusive).

        Raises:
            ValueError: If no outcomes fall within [low, high].
        """
        self.base = base
        self.low = low
        self.high = high

        mask = (base.values >= low - 1e-12) & (base.values <= high + 1e-12)
        if not np.any(mask):
            raise ValueError(
                f"No outcomes in [{low}, {high}] for the base distribution"
            )
        trunc_probs = base.probs[mask].copy()
        trunc_values = base.values[mask].copy()
        trunc_probs = _normalize_probabilities(trunc_probs)
        self._truncated = DiscreteDistribution(trunc_probs, trunc_values)

    def sample(
        self, n: int = 1, rng: Optional[np.random.Generator] = None,
    ) -> npt.NDArray[np.float64]:
        """Draw samples from the truncated distribution."""
        return self._truncated.sample(n, rng=rng)

    def mean(self) -> float:
        """Expected value of the truncated distribution."""
        return self._truncated.mean()

    def variance(self) -> float:
        """Variance of the truncated distribution."""
        return self._truncated.variance()

    def pmf(self, x: Union[int, float]) -> float:
        """Probability mass at outcome x in the truncated distribution."""
        return self._truncated.pmf(x)

    @property
    def k(self) -> int:
        """Number of outcomes in the truncated support."""
        return self._truncated.k

    def __repr__(self) -> str:
        return (
            f"TruncatedDistribution(base_k={self.base.k}, "
            f"trunc_k={self.k}, [{self.low}, {self.high}])"
        )


class ConditionalDistribution:
    """Conditional discrete distribution given a predicate.

    Restricts a base distribution to outcomes satisfying a condition,
    renormalizing the probabilities.

    Attributes:
        base: The original distribution.
        condition: Predicate function over outcome values.
        _conditional: The conditional (renormalized) distribution.
    """

    def __init__(
        self,
        base: DiscreteDistribution,
        condition: Callable[[float], bool],
    ) -> None:
        """Initialize conditional distribution.

        Args:
            base: Base distribution.
            condition: Predicate ``f(x) -> bool`` selecting outcomes.

        Raises:
            ValueError: If the condition has zero probability mass.
        """
        self.base = base
        self.condition = condition

        mask = np.array([condition(float(v)) for v in base.values])
        if not np.any(mask):
            raise ValueError("Condition has zero probability mass")
        cond_probs = base.probs[mask].copy()
        cond_values = base.values[mask].copy()
        cond_probs = _normalize_probabilities(cond_probs)
        self._conditional = DiscreteDistribution(cond_probs, cond_values)

    def sample(
        self, n: int = 1, rng: Optional[np.random.Generator] = None,
    ) -> npt.NDArray[np.float64]:
        """Draw samples from the conditional distribution."""
        return self._conditional.sample(n, rng=rng)

    def mean(self) -> float:
        """Expected value of the conditional distribution."""
        return self._conditional.mean()

    def variance(self) -> float:
        """Variance of the conditional distribution."""
        return self._conditional.variance()

    def pmf(self, x: Union[int, float]) -> float:
        """Probability mass at outcome x."""
        return self._conditional.pmf(x)

    @property
    def conditioning_probability(self) -> float:
        """Probability of the conditioning event under the base distribution."""
        mask = np.array([self.condition(float(v)) for v in self.base.values])
        return float(np.sum(self.base.probs[mask]))

    def __repr__(self) -> str:
        p_event = self.conditioning_probability
        return (
            f"ConditionalDistribution(base_k={self.base.k}, "
            f"cond_k={self._conditional.k}, P(event)={p_event:.4f})"
        )


# ---------------------------------------------------------------------------
# High-level MechanismSampler
# ---------------------------------------------------------------------------


class MechanismSampler:
    """High-level sampler for synthesised DP mechanisms.

    Wraps an :class:`ExtractedMechanism` and a :class:`QuerySpec` to provide
    convenient noise sampling, utility estimation, and privacy auditing.

    The sampler automatically selects the best sampling method (alias, CDF,
    or rejection) based on the mechanism structure and the caller's preference.

    Attributes:
        mechanism: The extracted mechanism (n × k probability table).
        spec: The query specification.
        method: Sampling method (alias, CDF, rejection, or auto).
        _row_samplers: Per-row samplers, lazily constructed.
        _rng: NumPy random generator.

    Example::

        >>> # After CEGIS synthesis:
        >>> sampler = MechanismSampler(mechanism, spec, method='auto')
        >>> noisy_answer = sampler.sample(true_value=42, n=1)
        >>> mse = sampler.estimate_mse([42], n_samples=10000)
    """

    def __init__(
        self,
        mechanism: ExtractedMechanism,
        spec: Optional[QuerySpec] = None,
        method: str = "auto",
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the mechanism sampler.

        Args:
            mechanism: Extracted mechanism with probability table ``p_final``.
            spec: Query specification (needed for grid mapping and utility).
            method: Sampling method: ``'alias'``, ``'cdf'``, ``'rejection'``,
                or ``'auto'`` (selects alias for k > 32, CDF otherwise).
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If method is not recognized.
        """
        valid_methods = {"alias", "cdf", "rejection", "auto"}
        if method not in valid_methods:
            raise ValueError(
                f"method must be one of {valid_methods}, got {method!r}"
            )
        self.mechanism = mechanism
        self.spec = spec
        self.method = method
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._row_samplers: Dict[int, Union[AliasMethodSampler, InverseCDFSampler]] = {}

        # Determine effective method
        if method == "auto":
            k = mechanism.k
            self._effective_method = "alias" if k > 32 else "cdf"
        else:
            self._effective_method = method

    def _get_grid(self) -> Optional[npt.NDArray[np.float64]]:
        """Return the output grid, if available from the spec metadata."""
        if self.spec is not None and "y_grid" in self.mechanism.metadata:
            return np.asarray(self.mechanism.metadata["y_grid"], dtype=np.float64)
        return None

    def _get_row_sampler(
        self, row_idx: int,
    ) -> Union[AliasMethodSampler, InverseCDFSampler]:
        """Get or build the sampler for a specific mechanism row.

        Args:
            row_idx: Row index in the probability table.

        Returns:
            Built sampler for the specified row.
        """
        if row_idx in self._row_samplers:
            return self._row_samplers[row_idx]

        probs = self.mechanism.p_final[row_idx]
        if self._effective_method == "alias":
            sampler: Union[AliasMethodSampler, InverseCDFSampler] = (
                AliasMethodSampler()
            )
            sampler.build(probs)
        else:
            sampler = InverseCDFSampler()
            sampler.build(probs, grid=self._get_grid())
        self._row_samplers[row_idx] = sampler
        return sampler

    def _resolve_row(self, true_value: Union[int, float]) -> int:
        """Map a true query value to the mechanism row index.

        If a spec is provided, finds the closest query value.  Otherwise,
        treats ``true_value`` as a row index directly.

        Args:
            true_value: The true query output.

        Returns:
            Row index in the probability table.

        Raises:
            ValueError: If the value cannot be mapped to a valid row.
        """
        if self.spec is not None:
            # Find closest query value
            diffs = np.abs(self.spec.query_values - true_value)
            row_idx = int(np.argmin(diffs))
        else:
            row_idx = int(true_value)
        if not (0 <= row_idx < self.mechanism.n):
            raise ValueError(
                f"Resolved row index {row_idx} out of range [0, {self.mechanism.n})"
            )
        return row_idx

    def sample(
        self,
        true_value: Union[int, float],
        n: int = 1,
    ) -> npt.NDArray[np.int64]:
        """Sample noisy output indices for a given true query value.

        Args:
            true_value: The true query output (or row index).
            n: Number of samples.

        Returns:
            Array of output bin indices, shape (n,).
        """
        row_idx = self._resolve_row(true_value)
        sampler = self._get_row_sampler(row_idx)
        if n == 1:
            return np.array([sampler.sample(rng=self._rng)], dtype=np.int64)
        return sampler.sample_batch(n, rng=self._rng)

    def sample_values(
        self,
        true_value: Union[int, float],
        n: int = 1,
    ) -> npt.NDArray[np.float64]:
        """Sample noisy output values (mapped through the grid).

        If a grid is available in the mechanism metadata, returns grid values.
        Otherwise, returns bin indices as floats.

        Args:
            true_value: The true query output.
            n: Number of samples.

        Returns:
            Array of output values, shape (n,).
        """
        indices = self.sample(true_value, n)
        grid = self._get_grid()
        if grid is not None:
            return grid[indices]
        return indices.astype(np.float64)

    def sample_vectorized(
        self,
        true_values: Sequence[Union[int, float]],
    ) -> npt.NDArray[np.int64]:
        """Sample one noisy output for each true value in a batch.

        Args:
            true_values: Sequence of true query outputs, length m.

        Returns:
            Array of output bin indices, shape (m,).
        """
        results = np.empty(len(true_values), dtype=np.int64)
        for i, tv in enumerate(true_values):
            results[i] = self.sample(tv, n=1)[0]
        return results

    def estimate_mse(
        self,
        true_values: Sequence[Union[int, float]],
        n_samples: int = 10_000,
    ) -> float:
        """Estimate mean squared error via Monte Carlo.

        For each true value, draws ``n_samples`` noisy outputs and computes
        the average squared error.  The MSE is averaged over all true values.

        Args:
            true_values: True query outputs to evaluate.
            n_samples: Number of Monte Carlo samples per true value.

        Returns:
            Estimated MSE, averaged over true values and samples.
        """
        grid = self._get_grid()
        total_mse = 0.0
        for tv in true_values:
            indices = self.sample(tv, n=n_samples)
            if grid is not None:
                noisy_vals = grid[indices]
            else:
                noisy_vals = indices.astype(np.float64)
            errors = (noisy_vals - float(tv)) ** 2
            total_mse += float(np.mean(errors))
        return total_mse / len(true_values)

    def estimate_mae(
        self,
        true_values: Sequence[Union[int, float]],
        n_samples: int = 10_000,
    ) -> float:
        """Estimate mean absolute error via Monte Carlo.

        Args:
            true_values: True query outputs.
            n_samples: Number of Monte Carlo samples per true value.

        Returns:
            Estimated MAE, averaged over true values.
        """
        grid = self._get_grid()
        total_mae = 0.0
        for tv in true_values:
            indices = self.sample(tv, n=n_samples)
            if grid is not None:
                noisy_vals = grid[indices]
            else:
                noisy_vals = indices.astype(np.float64)
            errors = np.abs(noisy_vals - float(tv))
            total_mae += float(np.mean(errors))
        return total_mae / len(true_values)

    def privacy_audit(
        self,
        n_samples: int = 100_000,
        pairs: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict[str, Any]:
        """Run an empirical privacy audit on the mechanism.

        For each adjacent pair (i, i'), estimates the ratio
        ``P[M(x_i) = y_j] / P[M(x_{i'}) = y_j]`` from samples and checks
        whether it exceeds ``exp(epsilon)``.

        This is a statistical test, not a formal proof.  It can detect
        violations but cannot certify compliance.

        Args:
            n_samples: Number of samples per row for histogram estimation.
            pairs: Specific pairs to audit.  If ``None``, audits all adjacent
                pairs from the spec.

        Returns:
            Dictionary with:
                - ``max_ratio``: Maximum observed likelihood ratio.
                - ``max_log_ratio``: Maximum observed log-likelihood ratio.
                - ``epsilon_target``: Target epsilon from the spec.
                - ``passed``: Whether max_log_ratio <= epsilon + tolerance.
                - ``worst_pair``: The pair achieving the maximum ratio.
                - ``worst_outcome``: The outcome achieving the maximum ratio.
        """
        if self.spec is None:
            raise RuntimeError("privacy_audit requires a QuerySpec")

        epsilon = self.spec.epsilon
        n = self.mechanism.n
        k = self.mechanism.k

        # Determine pairs to audit
        if pairs is None:
            if self.spec.edges is not None:
                audit_pairs = self.spec.edges.edges
            else:
                audit_pairs = [(i, i + 1) for i in range(n - 1)]
        else:
            audit_pairs = pairs

        # Use the actual probability table for exact ratios
        p = self.mechanism.p_final
        max_log_ratio = 0.0
        worst_pair: Tuple[int, int] = (0, 0)
        worst_outcome: int = 0

        for i, ip in audit_pairs:
            for j in range(k):
                pi_j = max(p[i, j], _LOG_EPS)
                pip_j = max(p[ip, j], _LOG_EPS)
                log_ratio = abs(np.log(pi_j) - np.log(pip_j))
                if log_ratio > max_log_ratio:
                    max_log_ratio = log_ratio
                    worst_pair = (i, ip)
                    worst_outcome = j

        # Also do a sampling-based check
        sample_histograms: Dict[int, npt.NDArray[np.int64]] = {}
        unique_rows = set()
        for i, ip in audit_pairs:
            unique_rows.add(i)
            unique_rows.add(ip)
        for row in unique_rows:
            samples = self.sample(row, n=n_samples)
            sample_histograms[row] = np.bincount(samples, minlength=k)

        max_sample_log_ratio = 0.0
        for i, ip in audit_pairs:
            for j in range(k):
                c_i = max(sample_histograms[i][j], 1)
                c_ip = max(sample_histograms[ip][j], 1)
                sample_lr = abs(np.log(c_i / n_samples) - np.log(c_ip / n_samples))
                max_sample_log_ratio = max(max_sample_log_ratio, sample_lr)

        tol = 3.0 / np.sqrt(n_samples)  # Statistical tolerance
        passed = max_log_ratio <= epsilon + 1e-6

        return {
            "max_ratio": np.exp(max_log_ratio),
            "max_log_ratio": max_log_ratio,
            "max_sample_log_ratio": max_sample_log_ratio,
            "epsilon_target": epsilon,
            "passed": passed,
            "worst_pair": worst_pair,
            "worst_outcome": worst_outcome,
            "n_pairs_audited": len(audit_pairs),
            "n_samples_per_row": n_samples,
            "statistical_tolerance": tol,
        }

    def row_distribution(self, row_idx: int) -> DiscreteDistribution:
        """Return the distribution for a specific mechanism row.

        Args:
            row_idx: Row index in the probability table.

        Returns:
            :class:`DiscreteDistribution` for the specified row.
        """
        if not (0 <= row_idx < self.mechanism.n):
            raise ValueError(
                f"row_idx {row_idx} out of range [0, {self.mechanism.n})"
            )
        probs = self.mechanism.p_final[row_idx]
        grid = self._get_grid()
        return DiscreteDistribution(probs, values=grid)

    def __repr__(self) -> str:
        return (
            f"MechanismSampler(n={self.mechanism.n}, k={self.mechanism.k}, "
            f"method={self._effective_method!r})"
        )


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def chi_squared_test(
    samples: npt.NDArray[np.int64],
    expected_probs: npt.NDArray[np.float64],
    min_expected: float = 5.0,
) -> Dict[str, Any]:
    """Pearson's chi-squared goodness-of-fit test.

    Tests whether observed sample frequencies match the expected
    distribution.  Bins with expected count < ``min_expected`` are merged.

    Args:
        samples: Integer sample array.
        expected_probs: Expected probability vector of length k.
        min_expected: Minimum expected count per bin (bins below this are
            merged into an "other" category).

    Returns:
        Dictionary with:
            - ``statistic``: Chi-squared test statistic.
            - ``df``: Degrees of freedom.
            - ``p_value``: p-value (approximate, using chi-squared CDF).
            - ``passed``: Whether p_value > 0.01 (fail to reject at 1%).
            - ``n_bins_original``: Original number of bins.
            - ``n_bins_merged``: Number of bins after merging.
    """
    expected_probs = _validate_probabilities(expected_probs)
    k = len(expected_probs)
    n = len(samples)
    observed = np.bincount(samples, minlength=k).astype(np.float64)

    # Truncate if observed has more bins than expected
    if len(observed) > k:
        observed = observed[:k]

    expected = expected_probs * n

    # Merge small bins
    merged_observed: list[float] = []
    merged_expected: list[float] = []
    carry_obs = 0.0
    carry_exp = 0.0
    for i in range(k):
        carry_obs += observed[i]
        carry_exp += expected[i]
        if carry_exp >= min_expected:
            merged_observed.append(carry_obs)
            merged_expected.append(carry_exp)
            carry_obs = 0.0
            carry_exp = 0.0
    # Merge remainder into last bin
    if carry_exp > 0:
        if merged_expected:
            merged_observed[-1] += carry_obs
            merged_expected[-1] += carry_exp
        else:
            merged_observed.append(carry_obs)
            merged_expected.append(carry_exp)

    n_bins = len(merged_observed)
    obs_arr = np.array(merged_observed)
    exp_arr = np.array(merged_expected)

    # Chi-squared statistic
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.sum((obs_arr - exp_arr) ** 2 / exp_arr)

    df = max(n_bins - 1, 1)

    # Approximate p-value using the chi-squared survival function.
    # Use scipy if available, otherwise a rough normal approximation.
    try:
        from scipy.stats import chi2 as chi2_dist
        p_value = float(chi2_dist.sf(chi2, df))
    except ImportError:
        # Wilson-Hilferty approximation
        z = ((chi2 / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
        p_value = 0.5 * math.erfc(z / math.sqrt(2))

    return {
        "statistic": float(chi2),
        "df": df,
        "p_value": p_value,
        "passed": p_value > 0.01,
        "n_bins_original": k,
        "n_bins_merged": n_bins,
        "n_samples": n,
    }


def ks_test(
    samples: npt.NDArray[np.float64],
    cdf_fn: Callable[[float], float],
) -> Dict[str, Any]:
    """Kolmogorov-Smirnov test against a reference CDF.

    Computes the KS statistic ``D = max|F_n(x) - F(x)|`` where ``F_n`` is
    the empirical CDF and ``F`` is the reference CDF.

    Args:
        samples: Sample array (float).
        cdf_fn: Reference CDF function ``F(x) -> float``.

    Returns:
        Dictionary with:
            - ``statistic``: KS statistic D.
            - ``n``: Number of samples.
            - ``critical_value_05``: Critical value at α=0.05.
            - ``passed``: Whether D < critical value at α=0.05.
    """
    samples = np.sort(np.asarray(samples, dtype=np.float64))
    n = len(samples)
    if n == 0:
        raise ValueError("samples must be non-empty")

    # Empirical CDF: F_n(x_i) = i/n
    empirical = np.arange(1, n + 1) / n
    empirical_minus = np.arange(0, n) / n

    # Reference CDF values at sample points
    reference = np.array([cdf_fn(float(x)) for x in samples])

    # KS statistic: max of D+ and D-
    d_plus = np.max(empirical - reference)
    d_minus = np.max(reference - empirical_minus)
    d_stat = max(float(d_plus), float(d_minus))

    # Critical value at α=0.05 (Kolmogorov-Smirnov table approximation)
    critical_05 = 1.358 / math.sqrt(n)

    return {
        "statistic": d_stat,
        "n": n,
        "d_plus": float(d_plus),
        "d_minus": float(d_minus),
        "critical_value_05": critical_05,
        "passed": d_stat < critical_05,
    }


def uniformity_test(
    samples: npt.NDArray[np.float64],
    n_bins: int = 100,
) -> Dict[str, Any]:
    """Test uniformity of random values in [0, 1).

    Uses a chi-squared test on a histogram of the samples to check whether
    they are uniformly distributed.  This is useful for validating the
    internal randomness of a sampler.

    Args:
        samples: Array of values expected to be uniform in [0, 1).
        n_bins: Number of histogram bins.

    Returns:
        Dictionary with chi-squared test results plus histogram data.
    """
    samples = np.asarray(samples, dtype=np.float64)
    n = len(samples)
    if n == 0:
        raise ValueError("samples must be non-empty")

    # Bin samples
    bin_edges = np.linspace(0, 1, n_bins + 1)
    observed, _ = np.histogram(samples, bins=bin_edges)
    expected = n / n_bins

    # Chi-squared statistic
    chi2_stat = float(np.sum((observed - expected) ** 2 / expected))
    df = n_bins - 1

    try:
        from scipy.stats import chi2 as chi2_dist
        p_value = float(chi2_dist.sf(chi2_stat, df))
    except ImportError:
        z = ((chi2_stat / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(
            2 / (9 * df)
        )
        p_value = 0.5 * math.erfc(z / math.sqrt(2))

    return {
        "statistic": chi2_stat,
        "df": df,
        "p_value": p_value,
        "passed": p_value > 0.01,
        "n_samples": n,
        "n_bins": n_bins,
        "min_bin_count": int(np.min(observed)),
        "max_bin_count": int(np.max(observed)),
    }


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def build_sampler(
    mechanism: ExtractedMechanism,
    spec: Optional[QuerySpec] = None,
    config: Optional[SamplingConfig] = None,
) -> MechanismSampler:
    """Factory function to build a :class:`MechanismSampler`.

    Maps :class:`SamplingConfig` settings to ``MechanismSampler`` arguments.

    Args:
        mechanism: Extracted mechanism.
        spec: Query specification.
        config: Sampling configuration.

    Returns:
        Configured :class:`MechanismSampler`.
    """
    if config is None:
        config = SamplingConfig()

    method_map = {
        SamplingMethod.ALIAS: "alias",
        SamplingMethod.CDF: "cdf",
        SamplingMethod.REJECTION: "rejection",
    }
    method = method_map.get(config.method, "auto")

    return MechanismSampler(
        mechanism=mechanism,
        spec=spec,
        method=method,
        seed=config.seed,
    )

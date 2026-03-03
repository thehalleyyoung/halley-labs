"""
LDP protocols: RAPPOR, shuffle model, secure aggregation, histograms.

Implements RAPPOR (Erlingsson et al. 2014), instantaneous and longitudinal
variants, priority sampling, shuffle model amplification (Balle et al. 2019),
secure aggregation with LDP fallback, and private histogram construction.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

from dp_forge.local_dp import (
    FrequencyEstimate,
    LDPConfig,
    LDPMechanismType,
    LDPReport,
    EncodingType,
)


# ---------------------------------------------------------------------------
# RAPPOR Bloom-filter helpers
# ---------------------------------------------------------------------------


def _bloom_encode(value: int, num_hashes: int, bloom_size: int) -> npt.NDArray[np.int8]:
    """Encode a value into a Bloom filter bit array."""
    bits = np.zeros(bloom_size, dtype=np.int8)
    for h in range(num_hashes):
        digest = hashlib.sha256(f"{h}:{value}".encode()).hexdigest()
        idx = int(digest, 16) % bloom_size
        bits[idx] = 1
    return bits


def _bloom_encode_string(value: str, num_hashes: int, bloom_size: int) -> npt.NDArray[np.int8]:
    """Encode a string value into a Bloom filter bit array."""
    bits = np.zeros(bloom_size, dtype=np.int8)
    for h in range(num_hashes):
        digest = hashlib.sha256(f"{h}:{value}".encode()).hexdigest()
        idx = int(digest, 16) % bloom_size
        bits[idx] = 1
    return bits


# ---------------------------------------------------------------------------
# RAPPOREncoder – basic RAPPOR
# ---------------------------------------------------------------------------


class RAPPOREncoder:
    """RAPPOR protocol (Erlingsson et al. 2014).

    Two-phase randomization:
    1. Permanent Randomized Response (PRR): creates a memoised noisy Bloom
       filter for each user (f parameter).
    2. Instantaneous Randomized Response (IRR): applies per-report noise
       (p, q parameters).
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        bloom_size: int = 64,
        num_hashes: int = 2,
        f: float = 0.5,
        p: Optional[float] = None,
        q: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        self._epsilon = epsilon
        self._bloom_size = bloom_size
        self._num_hashes = num_hashes
        self._f = f  # PRR parameter
        self._rng = np.random.default_rng(seed)
        # IRR parameters derived from ε if not provided
        e_half = math.exp(epsilon / 2.0)
        self._p = p if p is not None else e_half / (e_half + 1.0)
        self._q = q if q is not None else 1.0 / (e_half + 1.0)
        # Memoised permanent responses per user
        self._memos: Dict[int, npt.NDArray[np.int8]] = {}

    def _permanent_rr(self, true_bits: npt.NDArray[np.int8], user_id: int) -> npt.NDArray[np.int8]:
        """Permanent randomized response (memoised per user)."""
        if user_id in self._memos:
            return self._memos[user_id]
        result = true_bits.copy()
        for i in range(self._bloom_size):
            r = self._rng.random()
            if r < self._f / 2.0:
                result[i] = 1
            elif r < self._f:
                result[i] = 0
            # else keep original
        self._memos[user_id] = result
        return result

    def _instantaneous_rr(self, bits: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
        """Instantaneous randomized response."""
        result = np.zeros(self._bloom_size, dtype=np.int8)
        for i in range(self._bloom_size):
            if bits[i] == 1:
                result[i] = 1 if self._rng.random() < self._p else 0
            else:
                result[i] = 1 if self._rng.random() < self._q else 0
        return result

    def encode(self, value: int, user_id: int = 0) -> LDPReport:
        true_bits = _bloom_encode(value, self._num_hashes, self._bloom_size)
        prr_bits = self._permanent_rr(true_bits, user_id)
        irr_bits = self._instantaneous_rr(prr_bits)
        return LDPReport(
            user_id=user_id,
            encoded_value=irr_bits,
            mechanism_type=LDPMechanismType.RAPPOR,
            domain_size=self._bloom_size,
        )

    def encode_string(self, value: str, user_id: int = 0) -> LDPReport:
        true_bits = _bloom_encode_string(value, self._num_hashes, self._bloom_size)
        prr_bits = self._permanent_rr(true_bits, user_id)
        irr_bits = self._instantaneous_rr(prr_bits)
        return LDPReport(
            user_id=user_id,
            encoded_value=irr_bits,
            mechanism_type=LDPMechanismType.RAPPOR,
            domain_size=self._bloom_size,
        )

    def aggregate(
        self, reports: Sequence[LDPReport], candidate_values: Sequence[int],
    ) -> FrequencyEstimate:
        """Aggregate RAPPOR reports given a set of candidate values.

        Returns estimated frequency for each candidate.
        """
        n = len(reports)
        num_candidates = len(candidate_values)
        if n == 0:
            return FrequencyEstimate(
                frequencies=np.zeros(num_candidates), num_reports=0,
                mechanism_type=LDPMechanismType.RAPPOR,
            )
        # Count bit frequencies across reports
        bit_sums = np.zeros(self._bloom_size, dtype=np.float64)
        for r in reports:
            bit_sums += np.asarray(r.encoded_value, dtype=np.float64)
        # Debias each bit
        q_star = (1 - self._f / 2.0) * self._q + (self._f / 2.0) * self._p
        p_star = (1 - self._f / 2.0) * self._p + (self._f / 2.0) * self._q
        debiased = (bit_sums / n - q_star) / (p_star - q_star)
        debiased = np.clip(debiased, 0.0, 1.0)
        # Estimate frequency for each candidate
        freqs = np.zeros(num_candidates, dtype=np.float64)
        for idx, v in enumerate(candidate_values):
            bloom = _bloom_encode(v, self._num_hashes, self._bloom_size)
            bit_indices = np.where(bloom == 1)[0]
            if len(bit_indices) > 0:
                freqs[idx] = float(np.mean(debiased[bit_indices]))
        freqs = np.clip(freqs, 0.0, 1.0)
        total = freqs.sum()
        if total > 0:
            freqs /= total
        return FrequencyEstimate(
            frequencies=freqs, num_reports=n,
            mechanism_type=LDPMechanismType.RAPPOR,
        )

    @property
    def epsilon(self) -> float:
        return self._epsilon


# ---------------------------------------------------------------------------
# InstantaneousRAPPOR – one-round (no memoization)
# ---------------------------------------------------------------------------


class InstantaneousRAPPOR:
    """One-round RAPPOR without permanent randomization.

    Simpler variant that applies only instantaneous randomized response
    to a Bloom filter encoding. No memoisation needed.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        bloom_size: int = 64,
        num_hashes: int = 2,
        seed: Optional[int] = None,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        self._epsilon = epsilon
        self._bloom_size = bloom_size
        self._num_hashes = num_hashes
        self._rng = np.random.default_rng(seed)
        e_half = math.exp(epsilon / 2.0)
        self._p = e_half / (e_half + 1.0)
        self._q = 1.0 / (e_half + 1.0)

    def encode(self, value: int, user_id: int = 0) -> LDPReport:
        true_bits = _bloom_encode(value, self._num_hashes, self._bloom_size)
        noisy_bits = np.zeros(self._bloom_size, dtype=np.int8)
        for i in range(self._bloom_size):
            prob = self._p if true_bits[i] == 1 else self._q
            noisy_bits[i] = 1 if self._rng.random() < prob else 0
        return LDPReport(
            user_id=user_id,
            encoded_value=noisy_bits,
            mechanism_type=LDPMechanismType.RAPPOR,
            domain_size=self._bloom_size,
        )

    def aggregate(
        self, reports: Sequence[LDPReport], candidate_values: Sequence[int],
    ) -> FrequencyEstimate:
        n = len(reports)
        num_candidates = len(candidate_values)
        if n == 0:
            return FrequencyEstimate(
                frequencies=np.zeros(num_candidates), num_reports=0,
                mechanism_type=LDPMechanismType.RAPPOR,
            )
        bit_sums = np.zeros(self._bloom_size, dtype=np.float64)
        for r in reports:
            bit_sums += np.asarray(r.encoded_value, dtype=np.float64)
        debiased = (bit_sums / n - self._q) / (self._p - self._q)
        debiased = np.clip(debiased, 0.0, 1.0)
        freqs = np.zeros(num_candidates, dtype=np.float64)
        for idx, v in enumerate(candidate_values):
            bloom = _bloom_encode(v, self._num_hashes, self._bloom_size)
            bit_indices = np.where(bloom == 1)[0]
            if len(bit_indices) > 0:
                freqs[idx] = float(np.mean(debiased[bit_indices]))
        freqs = np.clip(freqs, 0.0, 1.0)
        total = freqs.sum()
        if total > 0:
            freqs /= total
        return FrequencyEstimate(
            frequencies=freqs, num_reports=n,
            mechanism_type=LDPMechanismType.RAPPOR,
        )

    @property
    def epsilon(self) -> float:
        return self._epsilon


# ---------------------------------------------------------------------------
# LongitudinalRAPPOR – multi-round with memoisation
# ---------------------------------------------------------------------------


class LongitudinalRAPPOR:
    """Multi-round RAPPOR with memoisation for longitudinal data.

    Each user maintains a permanent memoised Bloom filter. At each round,
    a fresh IRR is applied to the memo, providing protection against
    longitudinal tracking.
    """

    def __init__(
        self,
        epsilon_one: float = 1.0,
        epsilon_inf: float = 0.5,
        bloom_size: int = 64,
        num_hashes: int = 2,
        seed: Optional[int] = None,
    ) -> None:
        self._epsilon_one = epsilon_one
        self._epsilon_inf = epsilon_inf
        self._bloom_size = bloom_size
        self._num_hashes = num_hashes
        self._rng = np.random.default_rng(seed)
        # PRR parameter f controls longitudinal privacy
        self._f = 1.0 / (1.0 + math.exp(epsilon_inf / 2.0))
        # IRR parameters for single-round privacy
        e_half = math.exp(epsilon_one / 2.0)
        self._p = e_half / (e_half + 1.0)
        self._q = 1.0 / (e_half + 1.0)
        self._memos: Dict[int, npt.NDArray[np.int8]] = {}

    def encode(self, value: int, user_id: int = 0) -> LDPReport:
        true_bits = _bloom_encode(value, self._num_hashes, self._bloom_size)
        # Permanent RR (memoised)
        if user_id not in self._memos:
            memo = true_bits.copy()
            for i in range(self._bloom_size):
                r = self._rng.random()
                if r < self._f:
                    memo[i] = 1
                elif r < 2 * self._f:
                    memo[i] = 0
            self._memos[user_id] = memo
        memo = self._memos[user_id]
        # Instantaneous RR
        noisy = np.zeros(self._bloom_size, dtype=np.int8)
        for i in range(self._bloom_size):
            prob = self._p if memo[i] == 1 else self._q
            noisy[i] = 1 if self._rng.random() < prob else 0
        return LDPReport(
            user_id=user_id,
            encoded_value=noisy,
            mechanism_type=LDPMechanismType.RAPPOR,
            domain_size=self._bloom_size,
        )

    def aggregate(
        self, reports: Sequence[LDPReport], candidate_values: Sequence[int],
    ) -> FrequencyEstimate:
        n = len(reports)
        num_candidates = len(candidate_values)
        if n == 0:
            return FrequencyEstimate(
                frequencies=np.zeros(num_candidates), num_reports=0,
                mechanism_type=LDPMechanismType.RAPPOR,
            )
        bit_sums = np.zeros(self._bloom_size, dtype=np.float64)
        for r in reports:
            bit_sums += np.asarray(r.encoded_value, dtype=np.float64)
        q_star = (1 - self._f) * self._q + self._f * self._p
        p_star = (1 - self._f) * self._p + self._f * self._q
        debiased = (bit_sums / n - q_star) / (p_star - q_star)
        debiased = np.clip(debiased, 0.0, 1.0)
        freqs = np.zeros(num_candidates, dtype=np.float64)
        for idx, v in enumerate(candidate_values):
            bloom = _bloom_encode(v, self._num_hashes, self._bloom_size)
            bit_indices = np.where(bloom == 1)[0]
            if len(bit_indices) > 0:
                freqs[idx] = float(np.mean(debiased[bit_indices]))
        freqs = np.clip(freqs, 0.0, 1.0)
        total = freqs.sum()
        if total > 0:
            freqs /= total
        return FrequencyEstimate(
            frequencies=freqs, num_reports=n,
            mechanism_type=LDPMechanismType.RAPPOR,
        )

    @property
    def epsilon_one(self) -> float:
        return self._epsilon_one

    @property
    def epsilon_inf(self) -> float:
        return self._epsilon_inf


# ---------------------------------------------------------------------------
# PrioritySampling
# ---------------------------------------------------------------------------


class PrioritySampling:
    """Priority sampling for LDP (Jia & Raskhodnikova 2019).

    Assigns random priorities to domain values and reports the highest
    priority value with perturbation. Useful for distinct-count and
    set-union estimation.
    """

    def __init__(
        self,
        epsilon: float,
        domain_size: int,
        seed: Optional[int] = None,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        self._epsilon = epsilon
        self._domain_size = domain_size
        self._rng = np.random.default_rng(seed)
        e_eps = math.exp(epsilon)
        self._p = e_eps / (e_eps + domain_size - 1)
        self._q = 1.0 / (e_eps + domain_size - 1)

    def encode(self, value_set: Sequence[int], user_id: int = 0) -> LDPReport:
        """Encode a set of values using priority sampling.

        Args:
            value_set: Set of values the user holds (subset of domain).
            user_id: User identifier.
        """
        if not value_set:
            # Report a random value with equal probabilities
            reported = int(self._rng.integers(0, self._domain_size))
        else:
            # Assign random priorities, pick highest
            priorities = self._rng.random(len(value_set))
            best = value_set[int(np.argmax(priorities))]
            # Apply RR
            if self._rng.random() < self._p:
                reported = best
            else:
                candidates = [i for i in range(self._domain_size) if i != best]
                reported = int(self._rng.choice(candidates))
        return LDPReport(
            user_id=user_id,
            encoded_value=reported,
            mechanism_type=LDPMechanismType.RANDOMIZED_RESPONSE,
            domain_size=self._domain_size,
        )

    def estimate_set_union(self, reports: Sequence[LDPReport]) -> FrequencyEstimate:
        """Estimate frequency of each value in the union of user sets."""
        n = len(reports)
        counts = np.zeros(self._domain_size, dtype=np.float64)
        for r in reports:
            counts[int(r.encoded_value)] += 1
        freqs = (counts / max(n, 1) - self._q) / (self._p - self._q)
        freqs = np.clip(freqs, 0.0, 1.0)
        return FrequencyEstimate(frequencies=freqs, num_reports=n)

    @property
    def epsilon(self) -> float:
        return self._epsilon


# ---------------------------------------------------------------------------
# ShuffleModel – privacy amplification by shuffling
# ---------------------------------------------------------------------------


class ShuffleModel:
    """Shuffle model amplification (Balle et al. 2019, Erlingsson et al. 2019).

    Adds a trusted shuffler between users and the analyser. Amplifies
    local ε to a central ε_c ≈ O(ε_local * sqrt(ln(1/δ) / n)).
    """

    def __init__(
        self,
        local_epsilon: float,
        num_users: int,
        delta: float = 1e-6,
        seed: Optional[int] = None,
    ) -> None:
        if local_epsilon <= 0:
            raise ValueError(f"local_epsilon must be > 0, got {local_epsilon}")
        if num_users < 1:
            raise ValueError(f"num_users must be >= 1, got {num_users}")
        if not (0 < delta < 1):
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        self._local_eps = local_epsilon
        self._n = num_users
        self._delta = delta
        self._rng = np.random.default_rng(seed)

    @property
    def central_epsilon(self) -> float:
        """Amplified central ε via shuffling (Balle et al. 2019).

        Uses ε_c = log(1 + (e^ε_l - 1) * (sqrt(2*ln(2/δ)/n) + 1/n * (e^ε_l - 1))).
        """
        e_el = math.exp(self._local_eps) - 1.0
        n = self._n
        term = e_el * (math.sqrt(2.0 * math.log(2.0 / self._delta) / n)
                       + e_el / n)
        return math.log(1.0 + term)

    def shuffle(self, reports: Sequence[LDPReport]) -> List[LDPReport]:
        """Randomly permute reports (simulating a trusted shuffler)."""
        shuffled = list(reports)
        self._rng.shuffle(shuffled)
        # Strip user IDs for privacy
        return [
            LDPReport(
                user_id=-1,  # anonymised
                encoded_value=r.encoded_value,
                mechanism_type=r.mechanism_type,
                domain_size=r.domain_size,
            )
            for r in shuffled
        ]

    @property
    def local_epsilon(self) -> float:
        return self._local_eps

    @property
    def delta(self) -> float:
        return self._delta

    @property
    def num_users(self) -> int:
        return self._n

    def amplification_factor(self) -> float:
        """Ratio of local to central epsilon (higher = more amplification)."""
        ce = self.central_epsilon
        if ce <= 0:
            return float("inf")
        return self._local_eps / ce


# ---------------------------------------------------------------------------
# SecureAggregation – secure aggregation with LDP fallback
# ---------------------------------------------------------------------------


class SecureAggregation:
    """Secure aggregation with LDP fallback.

    When secure aggregation is available, adds discrete Gaussian noise
    split across users. Falls back to per-user LDP when the secure
    channel is unavailable.
    """

    def __init__(
        self,
        epsilon: float,
        domain_size: int,
        num_users: int,
        secure: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        self._epsilon = epsilon
        self._domain_size = domain_size
        self._n = num_users
        self._secure = secure
        self._rng = np.random.default_rng(seed)

    def encode(self, value: int, user_id: int = 0) -> LDPReport:
        if not (0 <= value < self._domain_size):
            raise ValueError(f"value must be in [0, {self._domain_size})")
        if self._secure:
            # In secure aggregation, each user adds a share of noise
            noise_scale = math.sqrt(2.0 * math.log(1.25)) / (self._epsilon * math.sqrt(self._n))
            vec = np.zeros(self._domain_size, dtype=np.float64)
            vec[value] = 1.0
            noise = self._rng.normal(0.0, noise_scale, size=self._domain_size)
            vec += noise
            return LDPReport(
                user_id=user_id,
                encoded_value=vec,
                mechanism_type=LDPMechanismType.RANDOMIZED_RESPONSE,
                domain_size=self._domain_size,
            )
        else:
            # Fallback: per-user randomized response
            e_eps = math.exp(self._epsilon)
            p = e_eps / (e_eps + self._domain_size - 1)
            if self._rng.random() < p:
                reported = value
            else:
                candidates = [i for i in range(self._domain_size) if i != value]
                reported = int(self._rng.choice(candidates))
            return LDPReport(
                user_id=user_id,
                encoded_value=reported,
                mechanism_type=LDPMechanismType.RANDOMIZED_RESPONSE,
                domain_size=self._domain_size,
            )

    def aggregate(self, reports: Sequence[LDPReport]) -> FrequencyEstimate:
        n = len(reports)
        if n == 0:
            return FrequencyEstimate(
                frequencies=np.zeros(self._domain_size), num_reports=0,
            )
        if self._secure:
            sums = np.zeros(self._domain_size, dtype=np.float64)
            for r in reports:
                sums += np.asarray(r.encoded_value, dtype=np.float64)
            freqs = sums / n
            freqs = np.clip(freqs, 0.0, 1.0)
            total = freqs.sum()
            if total > 0:
                freqs /= total
        else:
            e_eps = math.exp(self._epsilon)
            p = e_eps / (e_eps + self._domain_size - 1)
            q = 1.0 / (e_eps + self._domain_size - 1)
            counts = np.zeros(self._domain_size, dtype=np.float64)
            for r in reports:
                counts[int(r.encoded_value)] += 1
            freqs = (counts / n - q) / (p - q)
            freqs = np.clip(freqs, 0.0, 1.0)
            total = freqs.sum()
            if total > 0:
                freqs /= total
        return FrequencyEstimate(frequencies=freqs, num_reports=n)

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def is_secure(self) -> bool:
        return self._secure


# ---------------------------------------------------------------------------
# PrivateHistogram
# ---------------------------------------------------------------------------


class PrivateHistogram:
    """Build a histogram under local differential privacy.

    Wraps an LDP frequency oracle and provides a histogram interface
    with confidence intervals and optional post-processing.
    """

    def __init__(
        self,
        epsilon: float,
        domain_size: int,
        mechanism: str = "rr",
        seed: Optional[int] = None,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        self._epsilon = epsilon
        self._domain_size = domain_size
        self._mechanism_name = mechanism
        self._rng = np.random.default_rng(seed)
        e_eps = math.exp(epsilon)
        self._p = e_eps / (e_eps + domain_size - 1)
        self._q = 1.0 / (e_eps + domain_size - 1)

    def encode(self, value: int, user_id: int = 0) -> LDPReport:
        if not (0 <= value < self._domain_size):
            raise ValueError(f"value must be in [0, {self._domain_size})")
        if self._rng.random() < self._p:
            reported = value
        else:
            candidates = [i for i in range(self._domain_size) if i != value]
            reported = int(self._rng.choice(candidates))
        return LDPReport(
            user_id=user_id,
            encoded_value=reported,
            mechanism_type=LDPMechanismType.RANDOMIZED_RESPONSE,
            domain_size=self._domain_size,
        )

    def build_histogram(
        self,
        reports: Sequence[LDPReport],
        normalise: bool = True,
    ) -> FrequencyEstimate:
        """Build a debiased histogram from LDP reports."""
        n = len(reports)
        counts = np.zeros(self._domain_size, dtype=np.float64)
        for r in reports:
            counts[int(r.encoded_value)] += 1
        freqs = (counts / max(n, 1) - self._q) / (self._p - self._q)
        freqs = np.clip(freqs, 0.0, None)
        # Confidence intervals
        var_per = self._q * (1 - self._q) / ((self._p - self._q) ** 2 * max(n, 1))
        ci_half = 1.96 * math.sqrt(var_per)
        cis = np.column_stack([freqs - ci_half, freqs + ci_half])
        if normalise:
            total = freqs.sum()
            if total > 0:
                freqs /= total
                cis /= total
        return FrequencyEstimate(
            frequencies=freqs,
            confidence_intervals=cis,
            num_reports=n,
            mechanism_type=LDPMechanismType.RANDOMIZED_RESPONSE,
        )

    def encode_and_build(
        self,
        values: Sequence[int],
        normalise: bool = True,
    ) -> FrequencyEstimate:
        """Convenience: encode all values and build histogram."""
        reports = [self.encode(int(v), uid) for uid, v in enumerate(values)]
        return self.build_histogram(reports, normalise)

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def domain_size(self) -> int:
        return self._domain_size


__all__ = [
    "RAPPOREncoder",
    "InstantaneousRAPPOR",
    "LongitudinalRAPPOR",
    "PrioritySampling",
    "ShuffleModel",
    "SecureAggregation",
    "PrivateHistogram",
]

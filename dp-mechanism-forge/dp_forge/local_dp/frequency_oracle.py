"""
Frequency estimation oracles for local differential privacy.

Implements optimal local hashing (OLH), Hadamard response, random
projection estimator, count-min sketch with LDP, heavy-hitter detection,
and frequency calibration following Bassily et al. (2017), Wang et al. (2017),
Acharya et al. (2019).
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

from dp_forge.local_dp import (
    FrequencyEstimate,
    LDPMechanismType,
    LDPReport,
)


# ---------------------------------------------------------------------------
# Abstract FrequencyOracle
# ---------------------------------------------------------------------------


class FrequencyOracle(ABC):
    """Abstract base class for LDP frequency oracles."""

    def __init__(self, epsilon: float, domain_size: int, seed: Optional[int] = None) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        if domain_size < 2:
            raise ValueError(f"domain_size must be >= 2, got {domain_size}")
        self._epsilon = epsilon
        self._domain_size = domain_size
        self._rng = np.random.default_rng(seed)

    @abstractmethod
    def encode(self, value: int, user_id: int = 0) -> LDPReport:
        """Client-side perturbation."""
        ...

    @abstractmethod
    def estimate_all(self, reports: Sequence[LDPReport]) -> FrequencyEstimate:
        """Server-side aggregation for all values."""
        ...

    def estimate_single(self, reports: Sequence[LDPReport], value: int) -> float:
        """Estimate frequency for a single value."""
        est = self.estimate_all(reports)
        return float(est.frequencies[value])

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def domain_size(self) -> int:
        return self._domain_size


# ---------------------------------------------------------------------------
# OLH – Optimal Local Hashing
# ---------------------------------------------------------------------------


class OLHEstimator(FrequencyOracle):
    """Optimal local hashing (OLH) estimator (Wang et al. 2017).

    Each user hashes their value to {0, ..., g-1} using a random hash,
    then applies randomized response on the hashed value. Optimal
    g = ceil(e^ε + 1).
    """

    def __init__(
        self,
        epsilon: float,
        domain_size: int,
        g: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(epsilon, domain_size, seed)
        self._g = g if g is not None else max(2, int(math.ceil(math.exp(epsilon) + 1)))
        e_eps = math.exp(epsilon)
        self._p = e_eps / (e_eps + self._g - 1)
        self._q = 1.0 / (e_eps + self._g - 1)

    def _hash(self, value: int, hash_seed: int) -> int:
        """Deterministic hash of (value, seed) -> [0, g)."""
        h = hashlib.sha256(f"{value}:{hash_seed}".encode()).hexdigest()
        return int(h, 16) % self._g

    def encode(self, value: int, user_id: int = 0) -> LDPReport:
        if not (0 <= value < self._domain_size):
            raise ValueError(f"value must be in [0, {self._domain_size})")
        hash_seed = int(self._rng.integers(0, 2**31))
        hashed = self._hash(value, hash_seed)
        if self._rng.random() < self._p:
            reported = hashed
        else:
            candidates = [i for i in range(self._g) if i != hashed]
            reported = int(self._rng.choice(candidates))
        encoded = np.array([hash_seed, reported], dtype=np.int64)
        return LDPReport(
            user_id=user_id,
            encoded_value=encoded,
            mechanism_type=LDPMechanismType.LOCAL_HASHING,
            domain_size=self._domain_size,
        )

    def estimate_all(self, reports: Sequence[LDPReport]) -> FrequencyEstimate:
        n = len(reports)
        if n == 0:
            return FrequencyEstimate(
                frequencies=np.zeros(self._domain_size), num_reports=0,
            )
        counts = np.zeros(self._domain_size, dtype=np.float64)
        for v in range(self._domain_size):
            c = 0
            for r in reports:
                enc = np.asarray(r.encoded_value)
                h_seed, rep = int(enc[0]), int(enc[1])
                if self._hash(v, h_seed) == rep:
                    c += 1
            counts[v] = (c / n - self._q) / (self._p - self._q)
        freqs = np.clip(counts, 0.0, 1.0)
        total = freqs.sum()
        if total > 0:
            freqs /= total
        return FrequencyEstimate(
            frequencies=freqs, num_reports=n,
            mechanism_type=LDPMechanismType.LOCAL_HASHING,
        )

    @property
    def g(self) -> int:
        return self._g


# ---------------------------------------------------------------------------
# HadamardResponse
# ---------------------------------------------------------------------------


class HadamardResponse(FrequencyOracle):
    """Hadamard random response (Acharya et al. 2019).

    Uses the Hadamard matrix to encode values, then applies randomized
    response on a single coefficient.
    """

    def __init__(
        self,
        epsilon: float,
        domain_size: int,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(epsilon, domain_size, seed)
        # Round domain to next power of 2
        self._m = 1
        while self._m < domain_size:
            self._m *= 2
        e_eps = math.exp(epsilon)
        self._p = e_eps / (e_eps + 1.0)
        self._q = 1.0 / (e_eps + 1.0)

    @staticmethod
    def _hadamard_entry(i: int, j: int) -> int:
        """Compute H[i, j] = (-1)^{popcount(i & j)}."""
        return 1 - 2 * bin(i & j).count("1") % 2

    def encode(self, value: int, user_id: int = 0) -> LDPReport:
        if not (0 <= value < self._domain_size):
            raise ValueError(f"value must be in [0, {self._domain_size})")
        # Pick a random coefficient index
        j = int(self._rng.integers(0, self._m))
        h_val = self._hadamard_entry(value, j)  # +1 or -1
        # Report sign with RR
        if self._rng.random() < self._p:
            reported_sign = h_val
        else:
            reported_sign = -h_val
        encoded = np.array([j, reported_sign], dtype=np.int64)
        return LDPReport(
            user_id=user_id,
            encoded_value=encoded,
            mechanism_type=LDPMechanismType.RANDOMIZED_RESPONSE,
            domain_size=self._domain_size,
        )

    def estimate_all(self, reports: Sequence[LDPReport]) -> FrequencyEstimate:
        n = len(reports)
        if n == 0:
            return FrequencyEstimate(
                frequencies=np.zeros(self._domain_size), num_reports=0,
            )
        # Accumulate Hadamard coefficients
        coeffs = np.zeros(self._m, dtype=np.float64)
        coeff_counts = np.zeros(self._m, dtype=np.float64)
        for r in reports:
            enc = np.asarray(r.encoded_value)
            j_idx, sign = int(enc[0]), int(enc[1])
            debiased = (sign * self._m) / (self._p - self._q)
            coeffs[j_idx] += debiased
            coeff_counts[j_idx] += 1
        # Average coefficients
        mask = coeff_counts > 0
        coeffs[mask] /= coeff_counts[mask]
        # Inverse Hadamard transform (self-inverse up to scaling)
        freqs = np.zeros(self._domain_size, dtype=np.float64)
        for v in range(self._domain_size):
            s = 0.0
            for j in range(self._m):
                if coeff_counts[j] > 0:
                    s += self._hadamard_entry(v, j) * coeffs[j]
            freqs[v] = s / self._m
        freqs = np.clip(freqs, 0.0, 1.0)
        total = freqs.sum()
        if total > 0:
            freqs /= total
        return FrequencyEstimate(
            frequencies=freqs, num_reports=n,
            mechanism_type=LDPMechanismType.RANDOMIZED_RESPONSE,
        )


# ---------------------------------------------------------------------------
# ProjectionEstimator – random projection based
# ---------------------------------------------------------------------------


class ProjectionEstimator(FrequencyOracle):
    """Random projection based frequency estimator.

    Projects the one-hot encoding onto a random direction and adds noise,
    using the Johnson-Lindenstrauss property for utility.
    """

    def __init__(
        self,
        epsilon: float,
        domain_size: int,
        projection_dim: int = 16,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(epsilon, domain_size, seed)
        self._proj_dim = projection_dim
        # Generate a fixed random projection matrix (shared by all users)
        self._proj_rng = np.random.default_rng(seed if seed is not None else 42)
        self._A = self._proj_rng.standard_normal((projection_dim, domain_size)) / math.sqrt(projection_dim)
        self._noise_scale = math.sqrt(2.0) / epsilon

    def encode(self, value: int, user_id: int = 0) -> LDPReport:
        if not (0 <= value < self._domain_size):
            raise ValueError(f"value must be in [0, {self._domain_size})")
        # Project one-hot vector
        projected = self._A[:, value].copy()
        # Add Gaussian noise for privacy
        noise = self._rng.normal(0.0, self._noise_scale, size=self._proj_dim)
        noisy = projected + noise
        return LDPReport(
            user_id=user_id,
            encoded_value=noisy.astype(np.float64),
            mechanism_type=LDPMechanismType.RANDOMIZED_RESPONSE,
            domain_size=self._domain_size,
        )

    def estimate_all(self, reports: Sequence[LDPReport]) -> FrequencyEstimate:
        n = len(reports)
        if n == 0:
            return FrequencyEstimate(
                frequencies=np.zeros(self._domain_size), num_reports=0,
            )
        # Average the projections
        avg = np.zeros(self._proj_dim, dtype=np.float64)
        for r in reports:
            avg += np.asarray(r.encoded_value, dtype=np.float64)
        avg /= n
        # Least-squares recovery: find f that minimises ||A f - avg||
        result, _, _, _ = np.linalg.lstsq(self._A, avg, rcond=None)
        freqs = np.clip(result, 0.0, 1.0)
        total = freqs.sum()
        if total > 0:
            freqs /= total
        return FrequencyEstimate(frequencies=freqs, num_reports=n)


# ---------------------------------------------------------------------------
# CMS – Count-Min Sketch with LDP
# ---------------------------------------------------------------------------


class CMS(FrequencyOracle):
    """Count-min sketch with local differential privacy (Apple, 2017).

    Each user hashes their value with a random hash function and applies
    randomized response to the hashed value.
    """

    def __init__(
        self,
        epsilon: float,
        domain_size: int,
        num_hashes: int = 4,
        sketch_width: int = 256,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(epsilon, domain_size, seed)
        self._num_hashes = num_hashes
        self._width = sketch_width
        e_eps = math.exp(epsilon)
        self._p = e_eps / (e_eps + self._width - 1)
        self._q = 1.0 / (e_eps + self._width - 1)

    def _hash(self, value: int, hash_idx: int) -> int:
        h = hashlib.sha256(f"{hash_idx}:{value}".encode()).hexdigest()
        return int(h, 16) % self._width

    def encode(self, value: int, user_id: int = 0) -> LDPReport:
        if not (0 <= value < self._domain_size):
            raise ValueError(f"value must be in [0, {self._domain_size})")
        # Pick a random hash function
        h_idx = int(self._rng.integers(0, self._num_hashes))
        hashed = self._hash(value, h_idx)
        if self._rng.random() < self._p:
            reported = hashed
        else:
            candidates = [i for i in range(self._width) if i != hashed]
            reported = int(self._rng.choice(candidates))
        encoded = np.array([h_idx, reported], dtype=np.int64)
        return LDPReport(
            user_id=user_id,
            encoded_value=encoded,
            mechanism_type=LDPMechanismType.LOCAL_HASHING,
            domain_size=self._domain_size,
        )

    def estimate_all(self, reports: Sequence[LDPReport]) -> FrequencyEstimate:
        n = len(reports)
        if n == 0:
            return FrequencyEstimate(
                frequencies=np.zeros(self._domain_size), num_reports=0,
            )
        # Build sketch table counts per hash function
        sketch = np.zeros((self._num_hashes, self._width), dtype=np.float64)
        hash_counts = np.zeros(self._num_hashes, dtype=np.float64)
        for r in reports:
            enc = np.asarray(r.encoded_value)
            h_idx, rep = int(enc[0]), int(enc[1])
            sketch[h_idx, rep] += 1
            hash_counts[h_idx] += 1
        # Debias
        for h in range(self._num_hashes):
            if hash_counts[h] > 0:
                sketch[h] = (sketch[h] / hash_counts[h] - self._q) / (self._p - self._q)
        # Estimate each value: min across hash functions
        freqs = np.zeros(self._domain_size, dtype=np.float64)
        for v in range(self._domain_size):
            estimates = []
            for h_idx in range(self._num_hashes):
                if hash_counts[h_idx] > 0:
                    bucket = self._hash(v, h_idx)
                    estimates.append(sketch[h_idx, bucket])
            freqs[v] = min(estimates) if estimates else 0.0
        freqs = np.clip(freqs, 0.0, 1.0)
        total = freqs.sum()
        if total > 0:
            freqs /= total
        return FrequencyEstimate(frequencies=freqs, num_reports=n)


# ---------------------------------------------------------------------------
# HeavyHitterDetector
# ---------------------------------------------------------------------------


class HeavyHitterDetector:
    """Find frequent items under local differential privacy.

    Uses a two-phase approach: first estimates frequencies via a frequency
    oracle, then identifies items exceeding a threshold.
    """

    def __init__(
        self,
        epsilon: float,
        domain_size: int,
        threshold: float = 0.01,
        oracle: Optional[FrequencyOracle] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._epsilon = epsilon
        self._domain_size = domain_size
        self._threshold = threshold
        self._seed = seed
        self._oracle = oracle or OLHEstimator(epsilon, domain_size, seed=seed)

    def encode(self, value: int, user_id: int = 0) -> LDPReport:
        return self._oracle.encode(value, user_id)

    def detect(self, reports: Sequence[LDPReport]) -> List[Tuple[int, float]]:
        """Detect heavy hitters from LDP reports.

        Returns:
            Sorted list of (value, estimated_frequency) above threshold.
        """
        est = self._oracle.estimate_all(reports)
        results: List[Tuple[int, float]] = []
        for v in range(self._domain_size):
            f = float(est.frequencies[v])
            if f >= self._threshold:
                results.append((v, f))
        results.sort(key=lambda x: -x[1])
        return results

    def top_k(self, reports: Sequence[LDPReport], k: int = 10) -> List[Tuple[int, float]]:
        """Return top-k most frequent items."""
        est = self._oracle.estimate_all(reports)
        indexed = [(v, float(est.frequencies[v])) for v in range(self._domain_size)]
        indexed.sort(key=lambda x: -x[1])
        return indexed[:k]

    @property
    def epsilon(self) -> float:
        return self._epsilon


# ---------------------------------------------------------------------------
# FrequencyCalibrator – post-processing of frequency estimates
# ---------------------------------------------------------------------------


class FrequencyCalibrator:
    """Calibrate and post-process frequency estimates from LDP.

    Applies normalization, projection onto the probability simplex,
    and optional smoothing.
    """

    def __init__(self, domain_size: int, smoothing: float = 0.0) -> None:
        self._domain_size = domain_size
        self._smoothing = smoothing

    def calibrate(self, estimate: FrequencyEstimate) -> FrequencyEstimate:
        """Calibrate a raw frequency estimate."""
        freqs = estimate.frequencies.copy()
        # Apply Laplace smoothing
        if self._smoothing > 0:
            freqs += self._smoothing
        # Project onto probability simplex
        freqs = self._project_simplex(freqs)
        return FrequencyEstimate(
            frequencies=freqs,
            confidence_intervals=estimate.confidence_intervals,
            num_reports=estimate.num_reports,
            mechanism_type=estimate.mechanism_type,
        )

    @staticmethod
    def _project_simplex(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Project vector onto the probability simplex (Duchi et al. 2008)."""
        n = len(v)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1.0
        rho = np.nonzero(u > cssv / np.arange(1, n + 1))[0][-1]
        theta = cssv[rho] / (rho + 1.0)
        return np.maximum(v - theta, 0.0)

    def merge_estimates(
        self, estimates: Sequence[FrequencyEstimate], weights: Optional[Sequence[float]] = None,
    ) -> FrequencyEstimate:
        """Merge multiple frequency estimates with optional weighting."""
        if not estimates:
            raise ValueError("Need at least one estimate")
        d = estimates[0].domain_size
        if weights is None:
            weights = [e.num_reports for e in estimates]
        total_w = sum(weights)
        if total_w == 0:
            total_w = 1.0
        merged = np.zeros(d, dtype=np.float64)
        total_reports = 0
        for e, w in zip(estimates, weights):
            merged += w * e.frequencies
            total_reports += e.num_reports
        merged /= total_w
        merged = self._project_simplex(merged)
        return FrequencyEstimate(
            frequencies=merged, num_reports=total_reports,
        )


__all__ = [
    "FrequencyOracle",
    "OLHEstimator",
    "HadamardResponse",
    "ProjectionEstimator",
    "CMS",
    "HeavyHitterDetector",
    "FrequencyCalibrator",
]

"""
Mean estimation under local differential privacy.

Implements Duchi et al. mean estimator, piecewise mechanism, hybrid
mechanism, multidimensional mean, private stochastic gradient, and
clipped mean following Duchi et al. (2018), Wang et al. (2019).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from dp_forge.local_dp import (
    LDPMechanismType,
    LDPReport,
    MeanEstimate,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _normalise(value: float, lo: float, hi: float) -> float:
    """Map [lo, hi] -> [-1, 1]."""
    if hi <= lo:
        return 0.0
    return 2.0 * (value - lo) / (hi - lo) - 1.0


def _denormalise(value: float, lo: float, hi: float) -> float:
    """Map [-1, 1] -> [lo, hi]."""
    return lo + (value + 1.0) / 2.0 * (hi - lo)


# ---------------------------------------------------------------------------
# DuchiMeanEstimator
# ---------------------------------------------------------------------------


class DuchiMeanEstimator:
    """Duchi et al. (2018) optimal mean estimator for 1-d bounded data.

    Maps value to [-1, 1], then reports +C or -C with probabilities
    depending on the normalised value, where C = (e^ε + 1) / (e^ε - 1).
    """

    def __init__(
        self,
        epsilon: float,
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        self._epsilon = epsilon
        self._lo = lower_bound
        self._hi = upper_bound
        self._rng = np.random.default_rng(seed)
        e_eps = math.exp(epsilon)
        self._C = (e_eps + 1.0) / (e_eps - 1.0)

    def encode(self, value: float, user_id: int = 0) -> LDPReport:
        t = _normalise(_clip(value, self._lo, self._hi), self._lo, self._hi)
        prob_positive = (t + 1.0) / 2.0  # P[report +C]
        if self._rng.random() < prob_positive:
            reported = self._C
        else:
            reported = -self._C
        return LDPReport(
            user_id=user_id,
            encoded_value=np.float64(reported),
            mechanism_type=LDPMechanismType.PIECEWISE,
            domain_size=2,
        )

    def aggregate(self, reports: Sequence[LDPReport]) -> MeanEstimate:
        n = len(reports)
        if n == 0:
            mid = (self._lo + self._hi) / 2.0
            return MeanEstimate(mean=mid, variance=0.0, confidence_interval=(mid, mid), num_reports=0)
        values = np.array([float(r.encoded_value) for r in reports])
        est_norm = float(np.mean(values))
        est = _denormalise(np.clip(est_norm, -1.0, 1.0), self._lo, self._hi)
        var = self._C ** 2 / n
        se = math.sqrt(var)
        ci = (est - 1.96 * se * (self._hi - self._lo) / 2,
              est + 1.96 * se * (self._hi - self._lo) / 2)
        return MeanEstimate(mean=est, variance=var, confidence_interval=ci, num_reports=n)

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def variance_per_report(self) -> float:
        return self._C ** 2


# ---------------------------------------------------------------------------
# PiecewiseMechanism
# ---------------------------------------------------------------------------


class PiecewiseMechanism:
    """Piecewise constant mechanism for mean estimation (Wang et al. 2019).

    Achieves lower variance than Duchi for moderate ε by using a piecewise
    density over the output range.
    """

    def __init__(
        self,
        epsilon: float,
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        self._epsilon = epsilon
        self._lo = lower_bound
        self._hi = upper_bound
        self._rng = np.random.default_rng(seed)
        e_half = math.exp(epsilon / 2.0)
        self._C = (e_half + 1.0) / (e_half - 1.0)

    def encode(self, value: float, user_id: int = 0) -> LDPReport:
        t = _normalise(_clip(value, self._lo, self._hi), self._lo, self._hi)
        C = self._C
        e_half = math.exp(self._epsilon / 2.0)
        # Left and right boundaries of the "high-probability" interval
        l_t = (C + 1.0) / 2.0 * t - (C - 1.0) / 2.0
        r_t = l_t + C - 1.0
        # Sample
        p_in = e_half / (e_half + 1.0)
        if self._rng.random() < p_in:
            x = self._rng.uniform(l_t, r_t)
        else:
            # Sample from outside [l_t, r_t] within [-C, C]
            left_len = l_t - (-C)
            right_len = C - r_t
            total_out = left_len + right_len
            if total_out <= 0:
                x = self._rng.uniform(-C, C)
            elif self._rng.random() < left_len / total_out:
                x = self._rng.uniform(-C, l_t)
            else:
                x = self._rng.uniform(r_t, C)
        return LDPReport(
            user_id=user_id,
            encoded_value=np.float64(x),
            mechanism_type=LDPMechanismType.PIECEWISE,
            domain_size=2,
        )

    def aggregate(self, reports: Sequence[LDPReport]) -> MeanEstimate:
        n = len(reports)
        if n == 0:
            mid = (self._lo + self._hi) / 2.0
            return MeanEstimate(mean=mid, variance=0.0, confidence_interval=(mid, mid), num_reports=0)
        values = np.array([float(r.encoded_value) for r in reports])
        est_norm = float(np.mean(values))
        est = _denormalise(np.clip(est_norm, -1.0, 1.0), self._lo, self._hi)
        var = self._C ** 2 / n
        se = math.sqrt(var) * (self._hi - self._lo) / 2
        ci = (est - 1.96 * se, est + 1.96 * se)
        return MeanEstimate(mean=est, variance=var, confidence_interval=ci, num_reports=n)

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def variance_per_report(self) -> float:
        return self._C ** 2


# ---------------------------------------------------------------------------
# HybridMechanism – combines Duchi and Piecewise
# ---------------------------------------------------------------------------


class HybridMechanism:
    """Hybrid of Duchi and piecewise mechanisms (Wang et al. 2019).

    Uses Duchi for large ε and piecewise for small ε to achieve
    near-optimal variance across the full range of ε.
    """

    def __init__(
        self,
        epsilon: float,
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        self._epsilon = epsilon
        self._lo = lower_bound
        self._hi = upper_bound
        self._rng = np.random.default_rng(seed)
        # Crossover: piecewise is better for ε < ~1.29
        self._duchi = DuchiMeanEstimator(epsilon, lower_bound, upper_bound, seed)
        self._piecewise = PiecewiseMechanism(epsilon, lower_bound, upper_bound, seed)
        self._use_piecewise = (epsilon <= math.log(5.0 + 2.0 * math.sqrt(6.0)))

    def encode(self, value: float, user_id: int = 0) -> LDPReport:
        if self._use_piecewise:
            return self._piecewise.encode(value, user_id)
        return self._duchi.encode(value, user_id)

    def aggregate(self, reports: Sequence[LDPReport]) -> MeanEstimate:
        if self._use_piecewise:
            return self._piecewise.aggregate(reports)
        return self._duchi.aggregate(reports)

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def active_mechanism(self) -> str:
        return "piecewise" if self._use_piecewise else "duchi"


# ---------------------------------------------------------------------------
# MultidimensionalMean
# ---------------------------------------------------------------------------


class MultidimensionalMean:
    """Mean estimation in high dimensions under LDP.

    Uses random sampling + 1-d Duchi mechanism: each user picks a random
    coordinate, scales and perturbs it (Duchi et al. 2018, Sec 5).
    """

    def __init__(
        self,
        epsilon: float,
        dimension: int,
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        if dimension < 1:
            raise ValueError(f"dimension must be >= 1, got {dimension}")
        self._epsilon = epsilon
        self._dim = dimension
        self._lo = lower_bound
        self._hi = upper_bound
        self._rng = np.random.default_rng(seed)
        self._duchi = DuchiMeanEstimator(epsilon, lower_bound, upper_bound, seed)
        e_eps = math.exp(epsilon)
        self._C = (e_eps + 1.0) / (e_eps - 1.0)

    def encode(self, value: npt.NDArray[np.float64], user_id: int = 0) -> LDPReport:
        """Encode a d-dimensional value.

        Randomly selects one coordinate, perturbs it with Duchi, and scales
        by d to obtain an unbiased estimate.
        """
        if len(value) != self._dim:
            raise ValueError(f"Expected dimension {self._dim}, got {len(value)}")
        j = int(self._rng.integers(0, self._dim))
        t = _normalise(_clip(float(value[j]), self._lo, self._hi), self._lo, self._hi)
        prob_pos = (t + 1.0) / 2.0
        if self._rng.random() < prob_pos:
            z = self._C * self._dim
        else:
            z = -self._C * self._dim
        encoded = np.array([j, z], dtype=np.float64)
        return LDPReport(
            user_id=user_id,
            encoded_value=encoded,
            mechanism_type=LDPMechanismType.PIECEWISE,
            domain_size=self._dim,
        )

    def aggregate(self, reports: Sequence[LDPReport]) -> npt.NDArray[np.float64]:
        """Aggregate reports to estimate d-dimensional mean."""
        n = len(reports)
        if n == 0:
            return np.full(self._dim, (self._lo + self._hi) / 2.0)
        sums = np.zeros(self._dim, dtype=np.float64)
        counts = np.zeros(self._dim, dtype=np.float64)
        for r in reports:
            enc = np.asarray(r.encoded_value, dtype=np.float64)
            j = int(enc[0])
            z = enc[1]
            sums[j] += z
            counts[j] += 1
        mask = counts > 0
        est_norm = np.zeros(self._dim)
        est_norm[mask] = sums[mask] / counts[mask] / self._dim
        est = np.array([_denormalise(np.clip(v, -1.0, 1.0), self._lo, self._hi)
                        for v in est_norm])
        return est

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def dimension(self) -> int:
        return self._dim


# ---------------------------------------------------------------------------
# PrivateStochasticGradient – LDP variant of private SGD
# ---------------------------------------------------------------------------


class PrivateStochasticGradient:
    """Privacy-preserving stochastic gradient descent (LDP variant).

    Each user computes a gradient on their local data, clips and perturbs
    it using the Duchi mechanism, and sends it to the server.
    """

    def __init__(
        self,
        epsilon: float,
        dimension: int,
        clip_norm: float = 1.0,
        learning_rate: float = 0.01,
        seed: Optional[int] = None,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        self._epsilon = epsilon
        self._dim = dimension
        self._clip_norm = clip_norm
        self._lr = learning_rate
        self._rng = np.random.default_rng(seed)
        e_eps = math.exp(epsilon)
        self._C = (e_eps + 1.0) / (e_eps - 1.0)

    def _clip_gradient(self, grad: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        norm = float(np.linalg.norm(grad))
        if norm > self._clip_norm:
            return grad * (self._clip_norm / norm)
        return grad.copy()

    def perturb_gradient(
        self, grad: npt.NDArray[np.float64], user_id: int = 0,
    ) -> LDPReport:
        """Perturb a gradient vector with LDP guarantees."""
        clipped = self._clip_gradient(grad)
        # Random coordinate selection
        j = int(self._rng.integers(0, self._dim))
        val = clipped[j] / self._clip_norm  # normalise to [-1, 1]
        val = np.clip(val, -1.0, 1.0)
        prob_pos = (val + 1.0) / 2.0
        if self._rng.random() < prob_pos:
            z = self._C * self._dim * self._clip_norm
        else:
            z = -self._C * self._dim * self._clip_norm
        encoded = np.array([j, z], dtype=np.float64)
        return LDPReport(
            user_id=user_id,
            encoded_value=encoded,
            mechanism_type=LDPMechanismType.PIECEWISE,
            domain_size=self._dim,
        )

    def aggregate_gradients(
        self, reports: Sequence[LDPReport],
    ) -> npt.NDArray[np.float64]:
        """Aggregate perturbed gradients into average gradient estimate."""
        n = len(reports)
        if n == 0:
            return np.zeros(self._dim)
        sums = np.zeros(self._dim, dtype=np.float64)
        counts = np.zeros(self._dim, dtype=np.float64)
        for r in reports:
            enc = np.asarray(r.encoded_value, dtype=np.float64)
            j = int(enc[0])
            z = enc[1]
            sums[j] += z
            counts[j] += 1
        mask = counts > 0
        avg = np.zeros(self._dim)
        avg[mask] = sums[mask] / counts[mask] / self._dim
        return avg

    def step(
        self, params: npt.NDArray[np.float64], reports: Sequence[LDPReport],
    ) -> npt.NDArray[np.float64]:
        """Perform one SGD step using aggregated private gradients."""
        avg_grad = self.aggregate_gradients(reports)
        return params - self._lr * avg_grad

    @property
    def epsilon(self) -> float:
        return self._epsilon


# ---------------------------------------------------------------------------
# ClippedMean – clipped mean estimation with LDP
# ---------------------------------------------------------------------------


class ClippedMean:
    """Clipped mean estimation with local differential privacy.

    Clips each user's value to [lower, upper] and then applies the
    hybrid mechanism for mean estimation.
    """

    def __init__(
        self,
        epsilon: float,
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
        adaptive_clip: bool = False,
        clip_quantile: float = 0.95,
        seed: Optional[int] = None,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        self._epsilon = epsilon
        self._lo = lower_bound
        self._hi = upper_bound
        self._adaptive = adaptive_clip
        self._quantile = clip_quantile
        self._rng = np.random.default_rng(seed)
        self._mechanism = HybridMechanism(epsilon, lower_bound, upper_bound, seed)

    def encode(self, value: float, user_id: int = 0) -> LDPReport:
        clipped = _clip(value, self._lo, self._hi)
        return self._mechanism.encode(clipped, user_id)

    def encode_batch(
        self, values: npt.NDArray[np.float64],
    ) -> List[LDPReport]:
        if self._adaptive and len(values) > 0:
            q_val = float(np.quantile(np.abs(values), self._quantile))
            self._lo = -q_val
            self._hi = q_val
            self._mechanism = HybridMechanism(
                self._epsilon, self._lo, self._hi,
            )
        return [self.encode(float(v), uid) for uid, v in enumerate(values)]

    def aggregate(self, reports: Sequence[LDPReport]) -> MeanEstimate:
        return self._mechanism.aggregate(reports)

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def bounds(self) -> Tuple[float, float]:
        return (self._lo, self._hi)


__all__ = [
    "DuchiMeanEstimator",
    "PiecewiseMechanism",
    "HybridMechanism",
    "MultidimensionalMean",
    "PrivateStochasticGradient",
    "ClippedMean",
]

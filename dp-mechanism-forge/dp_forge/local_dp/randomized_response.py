"""
Randomized response mechanisms for local differential privacy.

Implements classic and generalized randomized response, optimal response
probabilities, direct encoding, k-ary RR, subset selection, and unary
encoding (OUE/SUE) following Warner (1965), Kairouz et al. (2016),
Wang et al. (2017).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

from dp_forge.local_dp import (
    FrequencyEstimate,
    LDPAnalysis,
    LDPConfig,
    LDPMechanismType,
    LDPReport,
)


# ---------------------------------------------------------------------------
# RandomizedResponse (classic binary / d-ary)
# ---------------------------------------------------------------------------


class RandomizedResponse:
    """Classic randomized response (Warner 1965, generalised to d values).

    User reports true value with probability p = e^eps / (e^eps + d - 1)
    and a uniformly random other value otherwise.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        domain_size: int = 2,
        seed: Optional[int] = None,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        if domain_size < 2:
            raise ValueError(f"domain_size must be >= 2, got {domain_size}")
        self._epsilon = epsilon
        self._domain_size = domain_size
        self._rng = np.random.default_rng(seed)
        e_eps = math.exp(epsilon)
        self._p = e_eps / (e_eps + domain_size - 1)
        self._q = 1.0 / (e_eps + domain_size - 1)

    # -- encoding ----------------------------------------------------------

    def encode(self, value: int, user_id: int = 0) -> LDPReport:
        """Perturb *value* using randomized response."""
        if not (0 <= value < self._domain_size):
            raise ValueError(f"value must be in [0, {self._domain_size}), got {value}")
        if self._rng.random() < self._p:
            reported = value
        else:
            candidates = list(range(self._domain_size))
            candidates.remove(value)
            reported = int(self._rng.choice(candidates))
        return LDPReport(
            user_id=user_id,
            encoded_value=reported,
            mechanism_type=LDPMechanismType.RANDOMIZED_RESPONSE,
            domain_size=self._domain_size,
        )

    def encode_batch(
        self, values: npt.NDArray[np.int64], user_ids: Optional[npt.NDArray[np.int64]] = None,
    ) -> List[LDPReport]:
        """Encode a batch of values."""
        n = len(values)
        if user_ids is None:
            user_ids = np.arange(n)
        return [self.encode(int(v), uid) for v, uid in zip(values, user_ids)]

    # -- aggregation / debiasing -------------------------------------------

    def aggregate(self, reports: Sequence[LDPReport]) -> FrequencyEstimate:
        """Debias collected reports to estimate true frequency distribution."""
        n = len(reports)
        if n == 0:
            return FrequencyEstimate(
                frequencies=np.zeros(self._domain_size),
                num_reports=0,
                mechanism_type=LDPMechanismType.RANDOMIZED_RESPONSE,
            )
        counts = np.zeros(self._domain_size, dtype=np.float64)
        for r in reports:
            counts[int(r.encoded_value)] += 1
        freqs = (counts / n - self._q) / (self._p - self._q)
        freqs = np.clip(freqs, 0.0, 1.0)
        total = freqs.sum()
        if total > 0:
            freqs /= total
        var_per = self._q * (1 - self._q) / ((self._p - self._q) ** 2 * n)
        ci = 1.96 * math.sqrt(var_per)
        cis = np.column_stack([freqs - ci, freqs + ci])
        return FrequencyEstimate(
            frequencies=freqs,
            confidence_intervals=cis,
            num_reports=n,
            mechanism_type=LDPMechanismType.RANDOMIZED_RESPONSE,
        )

    def estimate_frequency(self, reports: Sequence[LDPReport], value: int) -> float:
        """Estimate frequency of a single value."""
        est = self.aggregate(reports)
        return float(est.frequencies[value])

    # -- properties --------------------------------------------------------

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def domain_size(self) -> int:
        return self._domain_size

    @property
    def truth_probability(self) -> float:
        return self._p

    @property
    def lie_probability(self) -> float:
        return self._q

    # -- analysis ----------------------------------------------------------

    def analysis(self) -> LDPAnalysis:
        d = self._domain_size
        e_eps = math.exp(self._epsilon)
        mse = (d - 1) * (d - 1 + e_eps) / (e_eps - 1) ** 2
        bits = int(math.ceil(math.log2(d)))
        return LDPAnalysis(
            mechanism_type=LDPMechanismType.RANDOMIZED_RESPONSE,
            epsilon=self._epsilon,
            domain_size=d,
            expected_mse=mse,
            communication_bits=bits,
        )


# ---------------------------------------------------------------------------
# GeneralizedRR – per-value perturbation probabilities
# ---------------------------------------------------------------------------


class GeneralizedRR:
    """Generalised randomized response for categorical data.

    Allows specifying arbitrary perturbation probabilities subject to
    the ε-LDP constraint: for all v, v', P[report v | true v] / P[report v | true v'] <= e^ε.
    """

    def __init__(
        self,
        epsilon: float,
        domain_size: int,
        p_matrix: Optional[npt.NDArray[np.float64]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._epsilon = epsilon
        self._domain_size = domain_size
        self._rng = np.random.default_rng(seed)
        if p_matrix is not None:
            self._P = np.asarray(p_matrix, dtype=np.float64)
            if self._P.shape != (domain_size, domain_size):
                raise ValueError("p_matrix shape mismatch")
        else:
            e_eps = math.exp(epsilon)
            p = e_eps / (e_eps + domain_size - 1)
            q = 1.0 / (e_eps + domain_size - 1)
            self._P = np.full((domain_size, domain_size), q)
            np.fill_diagonal(self._P, p)

    def encode(self, value: int, user_id: int = 0) -> LDPReport:
        row = self._P[value]
        reported = int(self._rng.choice(self._domain_size, p=row))
        return LDPReport(
            user_id=user_id,
            encoded_value=reported,
            mechanism_type=LDPMechanismType.RANDOMIZED_RESPONSE,
            domain_size=self._domain_size,
        )

    def aggregate(self, reports: Sequence[LDPReport]) -> FrequencyEstimate:
        n = len(reports)
        counts = np.zeros(self._domain_size, dtype=np.float64)
        for r in reports:
            counts[int(r.encoded_value)] += 1
        observed = counts / max(n, 1)
        try:
            freqs = np.linalg.solve(self._P.T, observed)
        except np.linalg.LinAlgError:
            freqs = observed
        freqs = np.clip(freqs, 0.0, None)
        total = freqs.sum()
        if total > 0:
            freqs /= total
        return FrequencyEstimate(
            frequencies=freqs, num_reports=n,
            mechanism_type=LDPMechanismType.RANDOMIZED_RESPONSE,
        )

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def domain_size(self) -> int:
        return self._domain_size

    @property
    def perturbation_matrix(self) -> npt.NDArray[np.float64]:
        return self._P.copy()


# ---------------------------------------------------------------------------
# OptimalRR – minimise variance subject to ε-LDP
# ---------------------------------------------------------------------------


class OptimalRR:
    """Optimal randomized response that minimises estimator variance.

    For frequency estimation of a single value, the optimal mechanism
    (Kairouz et al. 2016) uses p* = e^ε / (e^ε + 1) and q* = 1 / (e^ε + 1).
    """

    def __init__(
        self,
        epsilon: float,
        domain_size: int,
        target_value: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._epsilon = epsilon
        self._domain_size = domain_size
        self._target = target_value
        self._rng = np.random.default_rng(seed)
        e_eps = math.exp(epsilon)
        self._p_star = e_eps / (e_eps + 1.0)
        self._q_star = 1.0 / (e_eps + 1.0)

    def encode(self, value: int, user_id: int = 0) -> LDPReport:
        """Binary encoding: report 1 if value == target with prob p*, else q*."""
        is_target = (value == self._target) if self._target is not None else True
        prob = self._p_star if is_target else self._q_star
        reported = int(self._rng.random() < prob)
        return LDPReport(
            user_id=user_id,
            encoded_value=reported,
            mechanism_type=LDPMechanismType.RANDOMIZED_RESPONSE,
            domain_size=2,
        )

    def estimate_frequency(self, reports: Sequence[LDPReport]) -> float:
        n = len(reports)
        if n == 0:
            return 0.0
        count = sum(1 for r in reports if int(r.encoded_value) == 1)
        return (count / n - self._q_star) / (self._p_star - self._q_star)

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def variance(self) -> float:
        """Variance of the estimator per report (worst case)."""
        return self._p_star * self._q_star / (self._p_star - self._q_star) ** 2


# ---------------------------------------------------------------------------
# DirectEncoding – one-hot + perturbation for frequency estimation
# ---------------------------------------------------------------------------


class DirectEncoding:
    """Direct encoding for frequency estimation (Bassily & Smith 2015).

    Each user sends a perturbed indicator vector of length d.
    """

    def __init__(
        self,
        epsilon: float,
        domain_size: int,
        seed: Optional[int] = None,
    ) -> None:
        self._epsilon = epsilon
        self._domain_size = domain_size
        self._rng = np.random.default_rng(seed)
        e_eps = math.exp(epsilon)
        self._p = e_eps / (e_eps + 1.0)
        self._q = 1.0 / (e_eps + 1.0)

    def encode(self, value: int, user_id: int = 0) -> LDPReport:
        if not (0 <= value < self._domain_size):
            raise ValueError(f"value must be in [0, {self._domain_size})")
        vec = np.zeros(self._domain_size, dtype=np.int8)
        for j in range(self._domain_size):
            if j == value:
                vec[j] = 1 if self._rng.random() < self._p else 0
            else:
                vec[j] = 1 if self._rng.random() < self._q else 0
        return LDPReport(
            user_id=user_id,
            encoded_value=vec,
            mechanism_type=LDPMechanismType.RANDOMIZED_RESPONSE,
            domain_size=self._domain_size,
        )

    def aggregate(self, reports: Sequence[LDPReport]) -> FrequencyEstimate:
        n = len(reports)
        if n == 0:
            return FrequencyEstimate(
                frequencies=np.zeros(self._domain_size), num_reports=0,
            )
        sums = np.zeros(self._domain_size, dtype=np.float64)
        for r in reports:
            sums += np.asarray(r.encoded_value, dtype=np.float64)
        freqs = (sums / n - self._q) / (self._p - self._q)
        freqs = np.clip(freqs, 0.0, 1.0)
        total = freqs.sum()
        if total > 0:
            freqs /= total
        return FrequencyEstimate(frequencies=freqs, num_reports=n)

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def domain_size(self) -> int:
        return self._domain_size


# ---------------------------------------------------------------------------
# KaryRR – k-ary randomized response
# ---------------------------------------------------------------------------


class KaryRR:
    """k-ary randomized response (Kairouz et al. 2016).

    Maps each input value to a randomly chosen output in {0, ..., k-1}
    where k can be smaller than the domain for communication savings.
    """

    def __init__(
        self,
        epsilon: float,
        domain_size: int,
        k: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._epsilon = epsilon
        self._domain_size = domain_size
        self._k = k if k is not None else domain_size
        if self._k < 2:
            raise ValueError(f"k must be >= 2, got {self._k}")
        self._rng = np.random.default_rng(seed)
        e_eps = math.exp(epsilon)
        self._p = e_eps / (e_eps + self._k - 1)
        self._q = 1.0 / (e_eps + self._k - 1)

    def _hash_value(self, value: int) -> int:
        """Hash domain value to output space [0, k)."""
        return value % self._k

    def encode(self, value: int, user_id: int = 0) -> LDPReport:
        if not (0 <= value < self._domain_size):
            raise ValueError(f"value must be in [0, {self._domain_size})")
        hashed = self._hash_value(value)
        if self._rng.random() < self._p:
            reported = hashed
        else:
            candidates = [i for i in range(self._k) if i != hashed]
            reported = int(self._rng.choice(candidates))
        return LDPReport(
            user_id=user_id,
            encoded_value=reported,
            mechanism_type=LDPMechanismType.RANDOMIZED_RESPONSE,
            domain_size=self._k,
        )

    def aggregate(self, reports: Sequence[LDPReport]) -> FrequencyEstimate:
        n = len(reports)
        counts = np.zeros(self._k, dtype=np.float64)
        for r in reports:
            counts[int(r.encoded_value)] += 1
        freqs = (counts / max(n, 1) - self._q) / (self._p - self._q)
        freqs = np.clip(freqs, 0.0, 1.0)
        total = freqs.sum()
        if total > 0:
            freqs /= total
        return FrequencyEstimate(frequencies=freqs, num_reports=n)

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def k(self) -> int:
        return self._k


# ---------------------------------------------------------------------------
# SubsetSelection – random subset selection mechanism
# ---------------------------------------------------------------------------


class SubsetSelection:
    """Random subset selection mechanism (Ye & Barg 2018).

    Reports a random subset of size k that contains the true value with
    higher probability, achieving near-optimal variance for large domains.
    """

    def __init__(
        self,
        epsilon: float,
        domain_size: int,
        seed: Optional[int] = None,
    ) -> None:
        self._epsilon = epsilon
        self._domain_size = domain_size
        self._rng = np.random.default_rng(seed)
        e_eps = math.exp(epsilon)
        # Optimal subset size
        d = domain_size
        self._k = max(1, min(d - 1, int(round(d / (e_eps + 1.0)))))
        self._p_include = e_eps * self._k / (e_eps * self._k + d - self._k)
        self._p_exclude = self._k / (e_eps * self._k + d - self._k)

    def encode(self, value: int, user_id: int = 0) -> LDPReport:
        if not (0 <= value < self._domain_size):
            raise ValueError(f"value must be in [0, {self._domain_size})")
        d = self._domain_size
        k = self._k
        if self._rng.random() < self._p_include:
            # Build a subset of size k containing value
            others = [i for i in range(d) if i != value]
            chosen_others = self._rng.choice(others, size=k - 1, replace=False)
            subset = np.sort(np.append(chosen_others, value)).astype(np.int8)
        else:
            # Build a subset of size k NOT containing value
            others = [i for i in range(d) if i != value]
            subset = np.sort(self._rng.choice(others, size=min(k, len(others)), replace=False)).astype(np.int8)
        return LDPReport(
            user_id=user_id,
            encoded_value=subset,
            mechanism_type=LDPMechanismType.RANDOMIZED_RESPONSE,
            domain_size=self._domain_size,
        )

    def aggregate(self, reports: Sequence[LDPReport]) -> FrequencyEstimate:
        n = len(reports)
        counts = np.zeros(self._domain_size, dtype=np.float64)
        for r in reports:
            subset = np.asarray(r.encoded_value)
            for v in subset:
                counts[int(v)] += 1
        d = self._domain_size
        k = self._k
        e_eps = math.exp(self._epsilon)
        # Expected count if value has freq f: n * [f * p_in * 1 + (1-f) * (k/(d-1)) * ...] 
        # Use debiasing: f_hat = (c/n - k/d) / (p_include - k/d)
        p_in = e_eps * k / (e_eps * k + d - k)
        baseline = k / d
        freqs = (counts / max(n, 1) - baseline) / (p_in - baseline) if abs(p_in - baseline) > 1e-12 else counts / max(n, 1)
        freqs = np.clip(freqs, 0.0, 1.0)
        total = freqs.sum()
        if total > 0:
            freqs /= total
        return FrequencyEstimate(frequencies=freqs, num_reports=n)

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def subset_size(self) -> int:
        return self._k


# ---------------------------------------------------------------------------
# UnaryEncoding – OUE (optimal) and SUE (symmetric)
# ---------------------------------------------------------------------------


class UnaryEncoding:
    """Unary encoding for LDP frequency estimation (Wang et al. 2017).

    Supports two variants:
    - **OUE** (optimal): p = 1/2, q = 1/(e^ε + 1) — minimises variance.
    - **SUE** (symmetric): p = e^(ε/2) / (e^(ε/2) + 1), q = 1/(e^(ε/2) + 1).
    """

    def __init__(
        self,
        epsilon: float,
        domain_size: int,
        variant: str = "OUE",
        seed: Optional[int] = None,
    ) -> None:
        self._epsilon = epsilon
        self._domain_size = domain_size
        self._variant = variant.upper()
        self._rng = np.random.default_rng(seed)
        if self._variant == "OUE":
            self._p = 0.5
            self._q = 1.0 / (math.exp(epsilon) + 1.0)
        elif self._variant == "SUE":
            e_half = math.exp(epsilon / 2.0)
            self._p = e_half / (e_half + 1.0)
            self._q = 1.0 / (e_half + 1.0)
        else:
            raise ValueError(f"Unknown variant {variant!r}, expected 'OUE' or 'SUE'")

    def encode(self, value: int, user_id: int = 0) -> LDPReport:
        if not (0 <= value < self._domain_size):
            raise ValueError(f"value must be in [0, {self._domain_size})")
        vec = np.zeros(self._domain_size, dtype=np.int8)
        for j in range(self._domain_size):
            prob = self._p if j == value else self._q
            vec[j] = 1 if self._rng.random() < prob else 0
        return LDPReport(
            user_id=user_id,
            encoded_value=vec,
            mechanism_type=LDPMechanismType.OPTIMAL_UNARY_ENCODING,
            domain_size=self._domain_size,
        )

    def aggregate(self, reports: Sequence[LDPReport]) -> FrequencyEstimate:
        n = len(reports)
        if n == 0:
            return FrequencyEstimate(
                frequencies=np.zeros(self._domain_size), num_reports=0,
            )
        sums = np.zeros(self._domain_size, dtype=np.float64)
        for r in reports:
            sums += np.asarray(r.encoded_value, dtype=np.float64)
        freqs = (sums / n - self._q) / (self._p - self._q)
        freqs = np.clip(freqs, 0.0, 1.0)
        total = freqs.sum()
        if total > 0:
            freqs /= total
        var_per = self._p * (1 - self._p) / ((self._p - self._q) ** 2 * n)
        ci = 1.96 * math.sqrt(var_per)
        cis = np.column_stack([freqs - ci, freqs + ci])
        return FrequencyEstimate(
            frequencies=freqs, confidence_intervals=cis, num_reports=n,
            mechanism_type=LDPMechanismType.OPTIMAL_UNARY_ENCODING,
        )

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def domain_size(self) -> int:
        return self._domain_size

    @property
    def variant(self) -> str:
        return self._variant

    def analysis(self) -> LDPAnalysis:
        e_eps = math.exp(self._epsilon)
        if self._variant == "OUE":
            mse = 4.0 * e_eps / (e_eps - 1.0) ** 2
        else:
            e_half = math.exp(self._epsilon / 2.0)
            mse = e_half * (e_half + 1.0) ** 2 / (e_half - 1.0) ** 2 / 4.0
        return LDPAnalysis(
            mechanism_type=LDPMechanismType.OPTIMAL_UNARY_ENCODING,
            epsilon=self._epsilon,
            domain_size=self._domain_size,
            expected_mse=mse,
            communication_bits=self._domain_size,
        )


__all__ = [
    "RandomizedResponse",
    "GeneralizedRR",
    "OptimalRR",
    "DirectEncoding",
    "KaryRR",
    "SubsetSelection",
    "UnaryEncoding",
]

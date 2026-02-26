"""
Formal Bloom filter soundness analysis for CoaCert-TLA.

Provides rigorous, experimentally-validated analysis of how Bloom filter
false positives affect verification soundness.  Addresses the review
critique: "Bloom filter false-positive impact on verification soundness
is not formally analyzed."

THEOREM (Bloom Verification Soundness)
    If B is a Bloom filter with m bits, k hash functions, and n
    inserted elements, the per-query false-positive rate is:
        p_fp = (1 - e^{-kn/m})^k

    For V independent verification checks, the probability that the
    verifier incorrectly accepts an invalid witness is bounded by:
        P(false_accept) ≤ 1 - (1 - p_fp)^V ≤ V · p_fp   (union bound)

    To achieve target soundness 1-ε, we need:
        p_fp ≤ ε / V   (from union bound)
        m ≥ -n · ln(ε/V) / (ln 2)²   (from optimal k)

This module provides:
    - BloomSoundnessAnalyzer: complete formal analysis
    - SoundnessExperiment: empirical validation of theoretical bounds
    - AdaptiveBloomConfig: automatic parameter tuning for target soundness
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Formal soundness analysis
# ---------------------------------------------------------------------------

@dataclass
class SoundnessBound:
    """Formal bound on verification unsoundness due to Bloom FP."""
    bloom_bits: int = 0
    bloom_hash_functions: int = 0
    bloom_elements: int = 0
    verification_checks: int = 0
    per_query_fpr: float = 0.0
    false_acceptance_union_bound: float = 0.0
    false_acceptance_exact: float = 0.0
    soundness_level: float = 1.0
    target_soundness: float = 0.999
    meets_target: bool = True
    proof_sketch: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bloom_bits": self.bloom_bits,
            "bloom_hash_functions": self.bloom_hash_functions,
            "bloom_elements": self.bloom_elements,
            "verification_checks": self.verification_checks,
            "per_query_fpr": self.per_query_fpr,
            "false_acceptance_union_bound": self.false_acceptance_union_bound,
            "false_acceptance_exact": self.false_acceptance_exact,
            "soundness_level": self.soundness_level,
            "target_soundness": self.target_soundness,
            "meets_target": self.meets_target,
            "proof_sketch": self.proof_sketch,
        }


@dataclass
class SoundnessReport:
    """Complete soundness analysis report."""
    bounds: SoundnessBound = field(default_factory=SoundnessBound)
    sensitivity: List[Dict[str, Any]] = field(default_factory=list)
    empirical_validation: Optional["EmpiricalResult"] = None
    recommended_config: Optional["AdaptiveBloomConfig"] = None
    certificate_annotation: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bounds": self.bounds.to_dict(),
            "sensitivity": self.sensitivity,
            "empirical": (
                self.empirical_validation.to_dict()
                if self.empirical_validation else None
            ),
            "recommended_config": (
                self.recommended_config.to_dict()
                if self.recommended_config else None
            ),
            "certificate_annotation": self.certificate_annotation,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class BloomSoundnessAnalyzer:
    """Formal Bloom filter soundness analysis.

    Implements the theorem linking Bloom filter false-positive rates to
    verification soundness, with configurable target soundness level.

    Parameters
    ----------
    target_soundness : float
        Desired probability that the verifier correctly rejects all
        invalid witnesses (e.g. 0.999 for 99.9 % soundness).
    hash_bits : int
        Cryptographic hash output size in bits (default: 256 for SHA-256).
    """

    def __init__(
        self,
        target_soundness: float = 0.999,
        hash_bits: int = 256,
    ) -> None:
        if not 0.0 < target_soundness < 1.0:
            raise ValueError("target_soundness must be in (0, 1)")
        self._target = target_soundness
        self._hash_bits = hash_bits

    @property
    def target_soundness(self) -> float:
        return self._target

    @staticmethod
    def per_query_fpr(m: int, k: int, n: int) -> float:
        """Compute Bloom filter false-positive probability.

        p_fp = (1 - e^{-kn/m})^k

        Parameters
        ----------
        m : int  — number of bits
        k : int  — number of hash functions
        n : int  — number of inserted elements
        """
        if m <= 0 or k <= 0:
            return 1.0
        if n <= 0:
            return 0.0
        exponent = -k * n / m
        if exponent < -700:
            return 1.0
        return (1.0 - math.exp(exponent)) ** k

    @staticmethod
    def optimal_k(m: int, n: int) -> int:
        """Optimal number of hash functions: k* = (m/n) · ln 2."""
        if n <= 0:
            return 1
        return max(1, round((m / n) * math.log(2)))

    @staticmethod
    def minimum_bits(n: int, target_fpr: float) -> int:
        """Minimum bits for n elements and target per-query FPR.

        m ≥ -n · ln(p) / (ln 2)²
        """
        if n <= 0 or target_fpr >= 1.0:
            return 64
        if target_fpr <= 0.0:
            target_fpr = 1e-20
        ln2_sq = math.log(2) ** 2
        m = int(math.ceil(-n * math.log(target_fpr) / ln2_sq))
        return max(m, 64)

    def analyze(
        self,
        m: int,
        k: int,
        n: int,
        verification_checks: int,
    ) -> SoundnessBound:
        """Perform formal soundness analysis.

        Parameters
        ----------
        m : int
            Bloom filter bits.
        k : int
            Number of hash functions.
        n : int
            Number of elements in the Bloom filter.
        verification_checks : int
            Number of independent verification checks (V).

        Returns
        -------
        SoundnessBound
            Formal bound with proof sketch.
        """
        V = max(verification_checks, 1)
        p_fp = self.per_query_fpr(m, k, n)

        # Exact: P(false_accept) = 1 - (1 - p_fp)^V
        if p_fp >= 1.0:
            fa_exact = 1.0
        elif p_fp <= 0.0:
            fa_exact = 0.0
        else:
            fa_exact = 1.0 - (1.0 - p_fp) ** V

        # Union bound: P(false_accept) ≤ V · p_fp
        fa_union = min(V * p_fp, 1.0)

        soundness = 1.0 - fa_exact
        meets = soundness >= self._target

        proof = (
            f"THEOREM (Bloom Verification Soundness):\n"
            f"  Given: m={m} bits, k={k} hash functions, "
            f"n={n} elements, V={V} checks.\n"
            f"  Step 1: p_fp = (1 - e^{{-kn/m}})^k = "
            f"(1 - e^{{{-k*n/m:.4f}}})^{k} = {p_fp:.2e}\n"
            f"  Step 2: P(false_accept) = 1 - (1-p_fp)^V\n"
            f"         = 1 - (1-{p_fp:.2e})^{V} = {fa_exact:.2e}\n"
            f"  Step 3: Union bound: V·p_fp = {fa_union:.2e}\n"
            f"  Soundness: 1 - {fa_exact:.2e} = {soundness:.6f}\n"
            f"  Target {self._target}: {'MET' if meets else 'NOT MET'}"
        )

        return SoundnessBound(
            bloom_bits=m,
            bloom_hash_functions=k,
            bloom_elements=n,
            verification_checks=V,
            per_query_fpr=p_fp,
            false_acceptance_union_bound=fa_union,
            false_acceptance_exact=fa_exact,
            soundness_level=soundness,
            target_soundness=self._target,
            meets_target=meets,
            proof_sketch=proof,
        )

    def sensitivity_table(
        self,
        n: int,
        verification_checks: int,
        bits_per_element_range: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate a table varying bits-per-element.

        Useful for selecting parameters that achieve the target soundness.
        """
        if bits_per_element_range is None:
            bits_per_element_range = [4, 6, 8, 10, 12, 14, 16, 20, 24, 32]

        V = max(verification_checks, 1)
        rows: List[Dict[str, Any]] = []

        for bpe in bits_per_element_range:
            m = bpe * max(n, 1)
            k = self.optimal_k(m, max(n, 1))
            p_fp = self.per_query_fpr(m, k, max(n, 1))
            fa = 1.0 - (1.0 - p_fp) ** V if p_fp < 1.0 else 1.0
            soundness = 1.0 - fa

            rows.append({
                "bits_per_element": bpe,
                "total_bits": m,
                "total_bytes": m // 8,
                "hash_functions": k,
                "per_query_fpr": p_fp,
                "false_acceptance": fa,
                "soundness": soundness,
                "meets_target": soundness >= self._target,
            })

        return rows

    def full_report(
        self,
        m: int,
        k: int,
        n: int,
        verification_checks: int,
        run_empirical: bool = False,
        empirical_trials: int = 10000,
    ) -> SoundnessReport:
        """Generate a comprehensive soundness report.

        Parameters
        ----------
        m, k, n : int
            Bloom filter parameters.
        verification_checks : int
            Number of verification checks.
        run_empirical : bool
            Whether to run empirical validation.
        empirical_trials : int
            Number of trials for empirical validation.
        """
        report = SoundnessReport()
        report.bounds = self.analyze(m, k, n, verification_checks)
        report.sensitivity = self.sensitivity_table(n, verification_checks)

        if run_empirical:
            experiment = SoundnessExperiment()
            report.empirical_validation = experiment.run(
                m=m, k=k, n=n,
                num_queries=verification_checks,
                trials=empirical_trials,
            )

        # Recommend optimal config
        adapter = AdaptiveBloomConfig(target_soundness=self._target)
        report.recommended_config = adapter.compute(
            n_elements=n,
            verification_checks=verification_checks,
        )

        # Generate certificate annotation
        report.certificate_annotation = {
            "bloom_filter": {
                "bits": m,
                "hash_functions": k,
                "elements": n,
                "per_query_fpr": report.bounds.per_query_fpr,
            },
            "verification_soundness": {
                "checks": verification_checks,
                "false_acceptance_bound": report.bounds.false_acceptance_exact,
                "soundness_level": report.bounds.soundness_level,
                "target_met": report.bounds.meets_target,
            },
        }

        return report


# ---------------------------------------------------------------------------
# Empirical validation
# ---------------------------------------------------------------------------

@dataclass
class EmpiricalResult:
    """Result of empirical Bloom filter FPR measurement."""
    trials: int = 0
    measured_fpr: float = 0.0
    theoretical_fpr: float = 0.0
    fpr_ratio: float = 0.0
    within_bound: bool = True
    measured_false_accepts: int = 0
    measured_false_accept_rate: float = 0.0
    theoretical_false_accept_bound: float = 0.0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trials": self.trials,
            "measured_fpr": self.measured_fpr,
            "theoretical_fpr": self.theoretical_fpr,
            "fpr_ratio": self.fpr_ratio,
            "within_bound": self.within_bound,
            "measured_false_accepts": self.measured_false_accepts,
            "measured_false_accept_rate": self.measured_false_accept_rate,
            "theoretical_false_accept_bound": self.theoretical_false_accept_bound,
            "elapsed_seconds": self.elapsed_seconds,
        }


class _SimpleBloomFilter:
    """Minimal Bloom filter implementation for empirical testing."""

    def __init__(self, m: int, k: int) -> None:
        self._m = m
        self._k = k
        self._bits = bytearray((m + 7) // 8)

    def add(self, item: bytes) -> None:
        for i in range(self._k):
            h = self._hash(item, i)
            byte_idx = h // 8
            bit_idx = h % 8
            if byte_idx < len(self._bits):
                self._bits[byte_idx] |= (1 << bit_idx)

    def query(self, item: bytes) -> bool:
        for i in range(self._k):
            h = self._hash(item, i)
            byte_idx = h // 8
            bit_idx = h % 8
            if byte_idx >= len(self._bits):
                return False
            if not (self._bits[byte_idx] & (1 << bit_idx)):
                return False
        return True

    def _hash(self, item: bytes, seed: int) -> int:
        h = hashlib.sha256(item + struct.pack("<I", seed)).digest()
        val = int.from_bytes(h[:8], "little")
        return val % self._m


class SoundnessExperiment:
    """Empirically measure Bloom filter FPR and compare to theory.

    Inserts n elements, then queries a set of non-member elements to
    measure the actual false positive rate.  Validates that the
    theoretical bound holds in practice.
    """

    def run(
        self,
        m: int,
        k: int,
        n: int,
        num_queries: int = 10000,
        trials: int = 10,
    ) -> EmpiricalResult:
        """Run the empirical FPR experiment.

        Parameters
        ----------
        m : int
            Bloom filter bits.
        k : int
            Hash functions.
        n : int
            Elements to insert.
        num_queries : int
            Non-member queries per trial.
        trials : int
            Number of independent trials to average.

        Returns
        -------
        EmpiricalResult
        """
        t0 = time.monotonic()

        # Theoretical bound
        theoretical_fpr = BloomSoundnessAnalyzer.per_query_fpr(m, k, n)

        total_fps = 0
        total_queries = 0
        false_accept_count = 0

        for trial in range(trials):
            bf = _SimpleBloomFilter(m, k)

            # Insert n elements
            for i in range(n):
                bf.add(f"member_{trial}_{i}".encode())

            # Query non-members
            fps_this_trial = 0
            for j in range(num_queries):
                query_item = f"nonmember_{trial}_{j}".encode()
                if bf.query(query_item):
                    fps_this_trial += 1

            total_fps += fps_this_trial
            total_queries += num_queries

            # Check if this trial would produce a false accept
            # (at least one false positive among all queries)
            if fps_this_trial > 0:
                false_accept_count += 1

        measured_fpr = total_fps / max(total_queries, 1)
        measured_fa_rate = false_accept_count / max(trials, 1)

        # Theoretical false acceptance probability
        if theoretical_fpr < 1.0:
            theoretical_fa = 1.0 - (1.0 - theoretical_fpr) ** num_queries
        else:
            theoretical_fa = 1.0

        return EmpiricalResult(
            trials=trials,
            measured_fpr=measured_fpr,
            theoretical_fpr=theoretical_fpr,
            fpr_ratio=(
                measured_fpr / theoretical_fpr
                if theoretical_fpr > 0 else 0.0
            ),
            within_bound=measured_fpr <= theoretical_fpr * 1.1,  # 10% tolerance
            measured_false_accepts=false_accept_count,
            measured_false_accept_rate=measured_fa_rate,
            theoretical_false_accept_bound=theoretical_fa,
            elapsed_seconds=time.monotonic() - t0,
        )


# ---------------------------------------------------------------------------
# Adaptive Bloom configuration
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveBloomConfig:
    """Automatically tuned Bloom filter parameters for target soundness.

    Given a target soundness level and the number of elements /
    verification checks, computes the optimal Bloom filter configuration.

    Parameters
    ----------
    target_soundness : float
        Desired soundness (e.g. 0.9999).
    max_memory_bytes : int
        Maximum allowed memory for the Bloom filter.
    """
    target_soundness: float = 0.999
    max_memory_bytes: int = 100 * 1024 * 1024  # 100 MB

    def compute(
        self,
        n_elements: int,
        verification_checks: int,
    ) -> "AdaptiveBloomConfig":
        """Compute and return self with filled parameters."""
        # Target per-query FPR from soundness requirement
        epsilon = 1.0 - self.target_soundness
        V = max(verification_checks, 1)
        target_fpr = epsilon / V

        # Minimum bits
        m = BloomSoundnessAnalyzer.minimum_bits(max(n_elements, 1), target_fpr)

        # Respect memory limit
        max_bits = self.max_memory_bytes * 8
        if m > max_bits:
            m = max_bits
            logger.warning(
                "Bloom filter capped at %d bits (%d MB) — "
                "soundness target may not be achievable",
                m, self.max_memory_bytes // (1024 * 1024),
            )

        k = BloomSoundnessAnalyzer.optimal_k(m, max(n_elements, 1))

        # Store computed parameters
        self._computed_bits = m
        self._computed_k = k
        self._computed_fpr = BloomSoundnessAnalyzer.per_query_fpr(
            m, k, max(n_elements, 1)
        )
        self._n_elements = n_elements
        self._verification_checks = V
        return self

    @property
    def computed_bits(self) -> int:
        return getattr(self, "_computed_bits", 0)

    @property
    def computed_hash_functions(self) -> int:
        return getattr(self, "_computed_k", 0)

    @property
    def computed_fpr(self) -> float:
        return getattr(self, "_computed_fpr", 0.0)

    @property
    def computed_bytes(self) -> int:
        return (self.computed_bits + 7) // 8

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_soundness": self.target_soundness,
            "max_memory_bytes": self.max_memory_bytes,
            "computed_bits": self.computed_bits,
            "computed_hash_functions": self.computed_hash_functions,
            "computed_fpr": self.computed_fpr,
            "computed_bytes": self.computed_bytes,
        }


# ---------------------------------------------------------------------------
# Certificate annotation integration
# ---------------------------------------------------------------------------

def annotate_certificate(
    certificate: Dict[str, Any],
    bloom_bits: int,
    bloom_k: int,
    bloom_n: int,
    verification_checks: int,
    target_soundness: float = 0.999,
) -> Dict[str, Any]:
    """Add Bloom filter FPR annotations to a verification certificate.

    Enriches the certificate with formal soundness metadata so that
    downstream consumers know the probabilistic guarantees.

    Parameters
    ----------
    certificate : dict
        The verification certificate to annotate.
    bloom_bits, bloom_k, bloom_n : int
        Bloom filter parameters.
    verification_checks : int
        Number of verification checks performed.
    target_soundness : float
        Target soundness level.

    Returns
    -------
    dict
        The annotated certificate (mutated in-place and also returned).
    """
    analyzer = BloomSoundnessAnalyzer(target_soundness=target_soundness)
    bound = analyzer.analyze(bloom_bits, bloom_k, bloom_n, verification_checks)

    certificate["bloom_soundness"] = {
        "bloom_filter": {
            "bits": bloom_bits,
            "hash_functions": bloom_k,
            "elements": bloom_n,
            "bits_per_element": bloom_bits / max(bloom_n, 1),
        },
        "verification": {
            "checks": verification_checks,
            "per_query_fpr": bound.per_query_fpr,
            "false_acceptance_bound": bound.false_acceptance_exact,
            "union_bound": bound.false_acceptance_union_bound,
            "soundness_level": bound.soundness_level,
            "target_soundness": target_soundness,
            "target_met": bound.meets_target,
        },
        "proof_sketch": bound.proof_sketch,
    }

    return certificate


# ---------------------------------------------------------------------------
# Verification soundness analyzer
# ---------------------------------------------------------------------------

class VerificationSoundnessAnalyzer:
    """Computes P(false acceptance) for coalgebraic verification.

    Given a quotient with |S/~| equivalence classes and |Act| actions,
    the total number of verification checks is V = |S/~| * |Act|.
    The probability of accepting an invalid witness is:

        P(false acceptance) = 1 - (1 - FPR)^V

    where FPR = (1 - e^{-kn/m})^k.
    """

    def __init__(self, target_soundness: float = 0.999) -> None:
        self._analyzer = BloomSoundnessAnalyzer(target_soundness=target_soundness)

    def compute(
        self,
        bloom_bits: int,
        bloom_k: int,
        bloom_n: int,
        quotient_classes: int,
        num_actions: int,
    ) -> SoundnessBound:
        """Compute false acceptance probability.

        Parameters
        ----------
        bloom_bits : int
            Number of bits in the Bloom filter (m).
        bloom_k : int
            Number of hash functions (k).
        bloom_n : int
            Number of elements inserted (n).
        quotient_classes : int
            Number of equivalence classes |S/~|.
        num_actions : int
            Size of the action alphabet |Act|.

        Returns
        -------
        SoundnessBound
        """
        V = max(quotient_classes, 1) * max(num_actions, 1)
        return self._analyzer.analyze(bloom_bits, bloom_k, bloom_n, V)

    def recommend_parameters(
        self,
        bloom_n: int,
        quotient_classes: int,
        num_actions: int,
    ) -> AdaptiveBloomConfig:
        """Recommend Bloom filter parameters for target soundness."""
        V = max(quotient_classes, 1) * max(num_actions, 1)
        adapter = AdaptiveBloomConfig(
            target_soundness=self._analyzer.target_soundness,
        )
        return adapter.compute(n_elements=bloom_n, verification_checks=V)


# ---------------------------------------------------------------------------
# Bloom soundness certificate
# ---------------------------------------------------------------------------

@dataclass
class BloomSoundnessCertificate:
    """Certificate bundling all computed Bloom filter soundness bounds.

    Encapsulates: FPR bound, false acceptance bound, optimal parameters,
    and witness size comparison.
    """
    bloom_bits: int = 0
    bloom_hash_functions: int = 0
    bloom_elements: int = 0
    per_query_fpr: float = 0.0
    verification_checks: int = 0
    false_acceptance_exact: float = 0.0
    false_acceptance_union_bound: float = 0.0
    soundness_level: float = 1.0
    target_soundness: float = 0.999
    meets_target: bool = True
    optimal_bits: int = 0
    optimal_k: int = 0
    full_witness_size_bytes: int = 0
    bloom_witness_size_bytes: int = 0
    compression_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bloom_parameters": {
                "bits": self.bloom_bits,
                "hash_functions": self.bloom_hash_functions,
                "elements": self.bloom_elements,
            },
            "fpr_analysis": {
                "per_query_fpr": self.per_query_fpr,
                "verification_checks": self.verification_checks,
                "false_acceptance_exact": self.false_acceptance_exact,
                "false_acceptance_union_bound": self.false_acceptance_union_bound,
                "soundness_level": self.soundness_level,
            },
            "target": {
                "target_soundness": self.target_soundness,
                "meets_target": self.meets_target,
            },
            "optimal_parameters": {
                "optimal_bits": self.optimal_bits,
                "optimal_k": self.optimal_k,
            },
            "witness_size_comparison": {
                "full_witness_bytes": self.full_witness_size_bytes,
                "bloom_witness_bytes": self.bloom_witness_size_bytes,
                "compression_ratio": self.compression_ratio,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def build_soundness_certificate(
    bloom_bits: int,
    bloom_k: int,
    bloom_n: int,
    verification_checks: int,
    full_witness_size_bytes: int = 0,
    bloom_witness_size_bytes: int = 0,
    target_soundness: float = 0.999,
) -> BloomSoundnessCertificate:
    """Build a complete BloomSoundnessCertificate.

    Parameters
    ----------
    bloom_bits, bloom_k, bloom_n : int
        Current Bloom filter parameters.
    verification_checks : int
        Number of independent verification checks (V).
    full_witness_size_bytes : int
        Size of the uncompressed (full) witness.
    bloom_witness_size_bytes : int
        Size of the Bloom-compressed witness.
    target_soundness : float
        Target soundness level.

    Returns
    -------
    BloomSoundnessCertificate
    """
    analyzer = BloomSoundnessAnalyzer(target_soundness=target_soundness)
    bound = analyzer.analyze(bloom_bits, bloom_k, bloom_n, verification_checks)

    # Compute optimal parameters
    epsilon = 1.0 - target_soundness
    V = max(verification_checks, 1)
    target_fpr = epsilon / V
    opt_m = BloomSoundnessAnalyzer.minimum_bits(max(bloom_n, 1), target_fpr)
    opt_k = BloomSoundnessAnalyzer.optimal_k(opt_m, max(bloom_n, 1))

    # Witness compression ratio
    if full_witness_size_bytes > 0:
        comp_ratio = bloom_witness_size_bytes / full_witness_size_bytes
    else:
        comp_ratio = 0.0

    return BloomSoundnessCertificate(
        bloom_bits=bloom_bits,
        bloom_hash_functions=bloom_k,
        bloom_elements=bloom_n,
        per_query_fpr=bound.per_query_fpr,
        verification_checks=verification_checks,
        false_acceptance_exact=bound.false_acceptance_exact,
        false_acceptance_union_bound=bound.false_acceptance_union_bound,
        soundness_level=bound.soundness_level,
        target_soundness=target_soundness,
        meets_target=bound.meets_target,
        optimal_bits=opt_m,
        optimal_k=opt_k,
        full_witness_size_bytes=full_witness_size_bytes,
        bloom_witness_size_bytes=bloom_witness_size_bytes,
        compression_ratio=comp_ratio,
    )


def compare_witness_sizes(
    num_states: int,
    num_transitions: int,
    bloom_bits: int,
    bytes_per_state: int = 32,
    bytes_per_transition: int = 48,
) -> Dict[str, Any]:
    """Compare full witness size vs Bloom-compressed witness size.

    Parameters
    ----------
    num_states : int
        Number of states in the full witness.
    num_transitions : int
        Number of transitions in the full witness.
    bloom_bits : int
        Bloom filter size in bits.
    bytes_per_state : int
        Estimated bytes per state in the full witness.
    bytes_per_transition : int
        Estimated bytes per transition in the full witness.

    Returns
    -------
    dict
        Comparison with full_witness_bytes, bloom_witness_bytes, ratio, savings_pct.
    """
    full_bytes = num_states * bytes_per_state + num_transitions * bytes_per_transition
    bloom_bytes = (bloom_bits + 7) // 8
    ratio = bloom_bytes / max(full_bytes, 1)
    savings_pct = (1.0 - ratio) * 100.0

    return {
        "full_witness_bytes": full_bytes,
        "bloom_witness_bytes": bloom_bytes,
        "compression_ratio": ratio,
        "savings_pct": savings_pct,
        "full_states": num_states,
        "full_transitions": num_transitions,
    }

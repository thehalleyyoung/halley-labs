"""
Bloom filter false-positive rate analysis for CoaCert-TLA.

The witness verification system uses hash-based data structures
(Merkle trees, hash chains) that are susceptible to hash collisions.
This module provides formal analysis of:

1. False-positive probability bounds for Bloom filters
2. Impact on verification soundness
3. Optimal parameter selection
4. Collision probability for SHA-256 based Merkle trees

THEOREM (Bloom Filter False Positive Rate):
  For a Bloom filter with m bits, k hash functions, and n inserted
  elements, the false positive probability is:
    p(FP) = (1 - e^{-kn/m})^k

  For the optimal number of hash functions k* = (m/n)·ln(2):
    p*(FP) = (1/2)^{k*} = 2^{-(m/n)·ln(2)}

THEOREM (Verification Soundness under Bloom Filter FP):
  Let W be a witness with N equivalence class entries, each checked
  via a Bloom filter with FP rate p. The probability that the
  verifier incorrectly accepts an invalid witness is bounded by:
    P(false_accept) ≤ 1 - (1-p)^N ≤ N·p

  For this probability to be below a target ε, we need:
    p ≤ ε / N

THEOREM (SHA-256 Collision Resistance):
  For SHA-256 with 256-bit output, the birthday-bound collision
  probability after q queries is:
    P(collision) ≈ q² / (2 · 2^256)

  For q = 10^9 (1 billion states):
    P(collision) ≈ 10^{18} / 2^{257} ≈ 4.3 × 10^{-60}
"""

from __future__ import annotations

import json
import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bloom filter parameters
# ---------------------------------------------------------------------------

@dataclass
class BloomFilterConfig:
    """Configuration for a Bloom filter."""
    num_bits: int  # m
    num_hash_functions: int  # k
    expected_elements: int  # n
    target_fpr: float = 0.01  # target false positive rate

    @property
    def bits_per_element(self) -> float:
        return self.num_bits / max(self.expected_elements, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_bits": self.num_bits,
            "num_hash_functions": self.num_hash_functions,
            "expected_elements": self.expected_elements,
            "target_fpr": self.target_fpr,
            "bits_per_element": self.bits_per_element,
        }


def optimal_bloom_parameters(
    n: int,
    target_fpr: float = 0.01,
) -> BloomFilterConfig:
    """Compute optimal Bloom filter parameters for n elements and target FPR.

    The optimal parameters are:
      m = -n·ln(p) / (ln(2))²
      k = (m/n)·ln(2)

    Parameters
    ----------
    n : int
        Expected number of elements.
    target_fpr : float
        Target false positive rate.

    Returns
    -------
    BloomFilterConfig
        Optimal configuration.
    """
    if n <= 0:
        return BloomFilterConfig(
            num_bits=64, num_hash_functions=1,
            expected_elements=0, target_fpr=target_fpr,
        )

    # m = -n * ln(p) / (ln(2))^2
    ln2_sq = math.log(2) ** 2
    m = int(math.ceil(-n * math.log(target_fpr) / ln2_sq))
    m = max(m, 64)

    # k = (m/n) * ln(2)
    k = max(1, round((m / n) * math.log(2)))

    return BloomFilterConfig(
        num_bits=m,
        num_hash_functions=k,
        expected_elements=n,
        target_fpr=target_fpr,
    )


def false_positive_bound(m: int, k: int, n: int) -> float:
    """Compute the false positive probability bound for a Bloom filter.

    FPR = (1 - e^{-kn/m})^k

    Parameters
    ----------
    m : int
        Number of bits.
    k : int
        Number of hash functions.
    n : int
        Number of inserted elements.
    """
    if m <= 0 or k <= 0:
        return 1.0
    exponent = -k * n / m
    # Guard against numerical issues
    if exponent < -700:
        return 1.0
    return (1.0 - math.exp(exponent)) ** k


# ---------------------------------------------------------------------------
# Soundness analysis
# ---------------------------------------------------------------------------

@dataclass
class BloomFilterSoundnessResult:
    """Result of analyzing Bloom filter impact on verification soundness.

    Contains:
    - Per-component FPR analysis
    - Aggregate false acceptance probability
    - Optimal parameter recommendations
    - SHA-256 collision analysis
    """

    bloom_fpr: float = 0.0
    witness_entries: int = 0
    false_acceptance_bound: float = 0.0
    sha256_collision_prob: float = 0.0
    combined_unsoundness_bound: float = 0.0
    target_soundness: float = 0.999
    recommended_config: Optional[BloomFilterConfig] = None
    details: List[str] = field(default_factory=list)
    analysis_components: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bloom_fpr": self.bloom_fpr,
            "witness_entries": self.witness_entries,
            "false_acceptance_bound": self.false_acceptance_bound,
            "sha256_collision_prob": self.sha256_collision_prob,
            "combined_unsoundness_bound": self.combined_unsoundness_bound,
            "target_soundness": self.target_soundness,
            "recommended_config": (
                self.recommended_config.to_dict()
                if self.recommended_config else None
            ),
            "details": self.details,
            "analysis_components": self.analysis_components,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class BloomFilterAnalysis:
    """Analyze the impact of Bloom filter false positives on verification.

    This analysis addresses the review critique that "Bloom filter
    false-positive impact on verification soundness is not formally analyzed."

    The analysis proceeds in three phases:
    1. Compute the FPR for the configured Bloom filter parameters
    2. Bound the probability that the verifier falsely accepts
    3. Recommend parameters to achieve a target soundness level
    """

    def __init__(
        self,
        target_soundness: float = 0.999,
    ) -> None:
        self._target_soundness = target_soundness

    def analyze(
        self,
        bloom_config: Optional[BloomFilterConfig] = None,
        witness_entries: int = 0,
        total_states: int = 0,
        equivalence_classes: int = 0,
        hash_bits: int = 256,
    ) -> BloomFilterSoundnessResult:
        """Perform complete soundness analysis.

        Parameters
        ----------
        bloom_config : BloomFilterConfig, optional
            Bloom filter configuration. If None, defaults are computed.
        witness_entries : int
            Number of entries in the witness to be verified.
        total_states : int
            Total number of states in the system.
        equivalence_classes : int
            Number of equivalence classes in the quotient.
        hash_bits : int
            Number of bits in the cryptographic hash (default: 256 for SHA-256).
        """
        result = BloomFilterSoundnessResult(
            target_soundness=self._target_soundness,
        )

        if witness_entries == 0:
            witness_entries = max(equivalence_classes, 1)
        result.witness_entries = witness_entries

        # Phase 1: Bloom filter FPR
        if bloom_config is None:
            target_fpr = (1.0 - self._target_soundness) / max(witness_entries, 1)
            bloom_config = optimal_bloom_parameters(
                n=equivalence_classes or witness_entries,
                target_fpr=target_fpr,
            )
        result.recommended_config = bloom_config

        result.bloom_fpr = false_positive_bound(
            bloom_config.num_bits,
            bloom_config.num_hash_functions,
            bloom_config.expected_elements,
        )
        result.details.append(
            f"Bloom FPR: {result.bloom_fpr:.2e} "
            f"(m={bloom_config.num_bits}, k={bloom_config.num_hash_functions}, "
            f"n={bloom_config.expected_elements})"
        )

        # Phase 2: False acceptance probability
        # P(false_accept) ≤ 1 - (1-p)^N ≤ N·p (union bound)
        p = result.bloom_fpr
        n = result.witness_entries
        if p * n < 0.01:
            result.false_acceptance_bound = p * n  # union bound tight
        else:
            result.false_acceptance_bound = 1.0 - (1.0 - p) ** n
        result.details.append(
            f"False acceptance bound: {result.false_acceptance_bound:.2e} "
            f"(N={n} entries, p={p:.2e})"
        )

        result.analysis_components.append({
            "component": "bloom_filter",
            "fpr": result.bloom_fpr,
            "entries_checked": n,
            "false_acceptance_prob": result.false_acceptance_bound,
        })

        # Phase 3: SHA-256 collision probability
        q = max(total_states, equivalence_classes, 1)
        # P(collision) ≈ q² / (2 · 2^hash_bits)
        log2_collision = 2 * math.log2(q) - 1 - hash_bits
        if log2_collision < -1000:
            result.sha256_collision_prob = 0.0
        else:
            result.sha256_collision_prob = 2 ** log2_collision
        result.details.append(
            f"SHA-{hash_bits} collision probability: {result.sha256_collision_prob:.2e} "
            f"(q={q} hash queries)"
        )

        result.analysis_components.append({
            "component": "hash_collision",
            "hash_bits": hash_bits,
            "queries": q,
            "collision_prob": result.sha256_collision_prob,
        })

        # Combined unsoundness bound (union bound over all sources)
        result.combined_unsoundness_bound = (
            result.false_acceptance_bound + result.sha256_collision_prob
        )
        result.details.append(
            f"Combined unsoundness bound: {result.combined_unsoundness_bound:.2e}"
        )

        # Check if we meet target soundness
        soundness_achieved = 1.0 - result.combined_unsoundness_bound
        if soundness_achieved >= self._target_soundness:
            result.details.append(
                f"✓ Target soundness {self._target_soundness} achieved: "
                f"{soundness_achieved:.6f}"
            )
        else:
            result.details.append(
                f"✗ Target soundness {self._target_soundness} NOT achieved: "
                f"{soundness_achieved:.6f}"
            )
            # Recommend better parameters
            better = self._recommend_for_target(
                witness_entries, self._target_soundness
            )
            result.details.append(
                f"  Recommended: m={better.num_bits}, k={better.num_hash_functions}"
            )

        return result

    def _recommend_for_target(
        self,
        witness_entries: int,
        target: float,
    ) -> BloomFilterConfig:
        """Recommend Bloom filter parameters to achieve target soundness."""
        epsilon = 1.0 - target
        per_entry_fpr = epsilon / max(witness_entries, 1)
        return optimal_bloom_parameters(
            n=witness_entries,
            target_fpr=per_entry_fpr,
        )

    def sensitivity_analysis(
        self,
        witness_entries: int,
        n_elements: int,
        bit_ranges: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Analyze sensitivity of soundness to Bloom filter parameters.

        Returns a table showing FPR and false acceptance probability
        for different numbers of bits per element.
        """
        if bit_ranges is None:
            bit_ranges = [4, 8, 10, 12, 14, 16, 20, 24, 32]

        results = []
        for bpe in bit_ranges:
            m = bpe * n_elements
            k = max(1, round(bpe * math.log(2)))
            fpr = false_positive_bound(m, k, n_elements)
            fa_bound = min(fpr * witness_entries, 1.0)

            results.append({
                "bits_per_element": bpe,
                "total_bits": m,
                "total_bytes": m // 8,
                "hash_functions": k,
                "fpr": fpr,
                "false_acceptance_bound": fa_bound,
                "soundness": 1.0 - fa_bound,
                "meets_target": (1.0 - fa_bound) >= self._target_soundness,
            })

        return results

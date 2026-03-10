"""
usability_oracle.simulation.activation — ACT-R activation dynamics.

Implements the declarative memory activation equations from ACT-R
(Anderson, 2007; Anderson et al., 2004).  Activation determines both
the probability and latency of memory retrieval, forming the basis
for working memory and long-term memory predictions.

Core equations:
    Base-level activation: B_i = ln(Σ t_j^{-d})
    Spreading activation: S_i = Σ W_j * S_{ji}
    Total activation: A_i = B_i + S_i + PM_i + ε
    Retrieval probability: P(retrieve) = 1 / (1 + exp(-(A_i - τ) / s))
    Retrieval latency: T = F * exp(-f * A_i)

References:
    Anderson, J. R. (2007). *How Can the Human Mind Occur in the Physical
        Universe?* Oxford University Press. (Ch. 3, Declarative Memory.)
    Anderson, J. R., Bothell, D., Byrne, M. D., Douglass, S., Lebiere, C.,
        & Qin, Y. (2004). An integrated theory of the mind. *Psychological
        Review*, 111(4), 1036-1060.
    Anderson, J. R., & Lebiere, C. (1998). *The Atomic Components of
        Thought*. Erlbaum.
    Pavlik, P. I., & Anderson, J. R. (2005). Practice and forgetting
        effects on vocabulary memory. *Cognitive Science*, 29(4), 559-586.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# ACT-R default parameters
# ═══════════════════════════════════════════════════════════════════════════

# All defaults from Anderson (2007) and Anderson & Lebiere (1998)

DEFAULT_DECAY: float = 0.5
"""Base-level learning decay parameter *d* (default 0.5).

Controls the rate at which memory traces decay.  Higher values mean
faster forgetting.

Reference: Anderson (2007), p. 60.
"""

DEFAULT_NOISE_S: float = 0.25
"""Activation noise parameter *s* (default 0.25).

The scale parameter of the logistic noise distribution added to
activation values.  s ≈ π/√6 * σ of the equivalent normal.

Reference: Anderson (2007), p. 68.
"""

DEFAULT_RETRIEVAL_THRESHOLD: float = -0.5
"""Retrieval threshold τ (default -0.5).

Chunks with activation below τ are not retrievable.

Reference: Anderson (2007), p. 69.
"""

DEFAULT_LATENCY_FACTOR: float = 0.630
"""Latency factor F (default 0.63 s).

Controls the overall time scale of retrieval.

Reference: Anderson (2007), p. 71.
"""

DEFAULT_LATENCY_EXPONENT: float = 1.0
"""Latency exponent f (default 1.0).

Controls sensitivity of retrieval time to activation.

Reference: Anderson (2007), p. 71.
"""

DEFAULT_MISMATCH_PENALTY: float = 1.5
"""Mismatch penalty P (default 1.5).

Applied per slot when a chunk partially matches a retrieval cue.

Reference: Anderson (2007), p. 75.
"""

DEFAULT_MAX_SPREADING: float = 1.0
"""Maximum spreading activation W (default 1.0).

Total activation spread from the source set.

Reference: Anderson (2007), p. 72.
"""


# ═══════════════════════════════════════════════════════════════════════════
# ChunkActivation — activation computation for a single chunk
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ChunkActivation:
    """Computes and caches activation components for a memory chunk.

    Activation is the sum of:
    1. Base-level activation B_i (recency/frequency)
    2. Spreading activation S_i (context)
    3. Partial matching penalty PM_i (similarity)
    4. Noise ε (stochastic perturbation)

    A_i = B_i + S_i + PM_i + ε

    Reference: Anderson (2007), Eq. 3.1.
    """

    chunk_id: str = ""
    base_level: float = 0.0
    spreading: float = 0.0
    partial_match: float = 0.0
    noise_value: float = 0.0

    @property
    def total_activation(self) -> float:
        """A_i = B_i + S_i + PM_i + ε"""
        return self.base_level + self.spreading + self.partial_match + self.noise_value

    def above_threshold(self, threshold: float = DEFAULT_RETRIEVAL_THRESHOLD) -> bool:
        return self.total_activation >= threshold

    def to_dict(self) -> Dict[str, float]:
        return {
            "chunk_id": self.chunk_id,
            "base_level": self.base_level,
            "spreading": self.spreading,
            "partial_match": self.partial_match,
            "noise": self.noise_value,
            "total": self.total_activation,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Base-level learning
# ═══════════════════════════════════════════════════════════════════════════

def base_level_learning(
    presentations: List[float],
    current_time: float,
    decay: float = DEFAULT_DECAY,
) -> float:
    """Compute base-level activation from presentation history.

    B_i = ln(Σ_{j=1}^{n} t_j^{-d})

    where t_j is the time since the j-th presentation and d is the
    decay parameter.

    Parameters:
        presentations: List of timestamps when the chunk was accessed.
        current_time: Current simulation time.
        decay: Decay exponent d (default 0.5).

    Returns:
        Base-level activation B_i.

    Reference: Anderson (2007), Eq. 3.2.
    """
    if not presentations:
        return -float("inf")

    total = 0.0
    for t_j in presentations:
        age = current_time - t_j
        if age > 0:
            total += age ** (-decay)
        elif age == 0:
            total += 1.0  # Avoid division by zero; just-presented items

    if total <= 0:
        return -100.0  # Effectively unretrievable
    return math.log(total)


def base_level_approximation(
    n_presentations: int,
    total_lifetime: float,
    decay: float = DEFAULT_DECAY,
) -> float:
    """Approximate base-level activation using the closed-form integral.

    When individual presentation times are unavailable, approximate:
        B_i ≈ ln(n / (1-d) * L^{1-d} / L)
            = ln(n) + (1-d) * ln(L) - ln(1-d)    for d ≠ 1

    where n is the number of presentations and L is the chunk's lifetime.

    This assumes uniformly distributed presentations over the lifetime.

    Reference: Anderson & Lebiere (1998), p. 124.
    """
    if n_presentations <= 0 or total_lifetime <= 0:
        return -100.0
    if decay >= 1.0:
        return math.log(n_presentations) - decay * math.log(total_lifetime)
    return (
        math.log(n_presentations)
        + (1.0 - decay) * math.log(total_lifetime)
        - math.log(1.0 - decay)
        - math.log(total_lifetime)
    )


# ═══════════════════════════════════════════════════════════════════════════
# Spreading activation
# ═══════════════════════════════════════════════════════════════════════════

def spreading_activation(
    source_chunks: List[str],
    association_strengths: Dict[Tuple[str, str], float],
    target_chunk: str,
    total_source_activation: float = DEFAULT_MAX_SPREADING,
) -> float:
    """Compute spreading activation from source chunks to a target.

    S_i = Σ_j W_j * S_{ji}

    where W_j is the attentional weight on source chunk j (distributed
    evenly: W_j = W / n) and S_{ji} is the association strength from
    source j to target i.

    Parameters:
        source_chunks: List of chunk IDs in the current focus of attention.
        association_strengths: Dict mapping (source_id, target_id) -> S_{ji}.
        target_chunk: The chunk for which to compute spreading activation.
        total_source_activation: Total activation W to spread (default 1.0).

    Returns:
        Spreading activation S_i for the target chunk.

    Reference: Anderson (2007), Eq. 3.3.
    """
    if not source_chunks:
        return 0.0

    w_j = total_source_activation / len(source_chunks)
    total = 0.0

    for source in source_chunks:
        s_ji = association_strengths.get((source, target_chunk), 0.0)
        total += w_j * s_ji

    return total


def compute_association_strength(
    fan: int,
    max_strength: float = 2.0,
) -> float:
    """Compute the association strength S_{ji} based on fan.

    S_{ji} = S_max - ln(fan_j)

    where fan_j is the number of chunks associated with chunk j.
    Higher fan means more diluted associations.

    Reference: Anderson (2007), p. 73.
    """
    if fan <= 0:
        return max_strength
    return max_strength - math.log(fan)


# ═══════════════════════════════════════════════════════════════════════════
# Partial matching
# ═══════════════════════════════════════════════════════════════════════════

def partial_matching(
    chunk_slots: Dict[str, Any],
    probe_slots: Dict[str, Any],
    similarity_fn: Optional[Callable[[Any, Any], float]] = None,
    mismatch_penalty: float = DEFAULT_MISMATCH_PENALTY,
) -> float:
    """Compute partial matching penalty PM_i.

    PM_i = Σ_k P * Sim(d_k, c_k)

    where P is the mismatch penalty, d_k is the value in the retrieval
    request (probe) for slot k, c_k is the value in the chunk, and
    Sim is a similarity function returning values in [-1, 0] (0 = identical,
    -1 = maximally dissimilar).

    Parameters:
        chunk_slots: Slot values of the memory chunk.
        probe_slots: Slot values of the retrieval request.
        similarity_fn: Custom similarity function(a, b) -> [-1, 0].
            Default: 0 for exact match, -1 for mismatch.
        mismatch_penalty: Penalty weight P (default 1.5).

    Returns:
        Partial matching component PM_i (non-positive).

    Reference: Anderson (2007), Eq. 3.5.
    """
    if similarity_fn is None:
        def similarity_fn(a: Any, b: Any) -> float:
            return 0.0 if a == b else -1.0

    total_penalty = 0.0
    for slot_name, probe_value in probe_slots.items():
        chunk_value = chunk_slots.get(slot_name)
        if chunk_value is None:
            total_penalty += mismatch_penalty * (-1.0)
        else:
            sim = similarity_fn(probe_value, chunk_value)
            total_penalty += mismatch_penalty * sim

    return total_penalty


def string_similarity(a: str, b: str) -> float:
    """Simple character-overlap similarity for strings.

    Returns a value in [-1, 0] where 0 = identical.
    Uses Jaccard similarity on character bigrams.
    """
    if a == b:
        return 0.0
    if not a or not b:
        return -1.0

    bigrams_a = {a[i:i+2] for i in range(len(a) - 1)} if len(a) > 1 else {a}
    bigrams_b = {b[i:i+2] for i in range(len(b) - 1)} if len(b) > 1 else {b}

    intersection = len(bigrams_a & bigrams_b)
    union = len(bigrams_a | bigrams_b)

    if union == 0:
        return -1.0
    jaccard = intersection / union
    return -(1.0 - jaccard)


def numeric_similarity(a: float, b: float, max_diff: float = 10.0) -> float:
    """Numeric similarity function.

    Returns value in [-1, 0] scaled by the absolute difference.
    """
    if a == b:
        return 0.0
    diff = abs(a - b)
    return -min(diff / max_diff, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Noise
# ═══════════════════════════════════════════════════════════════════════════

def noise(
    sigma: float = DEFAULT_NOISE_S,
    rng: Optional[np.random.RandomState] = None,
) -> float:
    """Generate logistic noise perturbation.

    ACT-R uses logistic noise with scale parameter s:
        ε ~ Logistic(0, s)

    The logistic distribution is used instead of normal because it has
    heavier tails, which better captures the variability of human
    memory retrieval.

    The relationship to the standard deviation σ of a normal:
        s ≈ σ * √3 / π

    Parameters:
        sigma: Scale parameter s of the logistic distribution.
        rng: Optional NumPy random state for reproducibility.

    Returns:
        A single noise sample ε.

    Reference: Anderson (2007), p. 68.
    """
    if sigma <= 0:
        return 0.0
    if rng is not None:
        return float(rng.logistic(0, sigma))
    return float(np.random.logistic(0, sigma))


def noise_batch(
    n: int,
    sigma: float = DEFAULT_NOISE_S,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Generate a batch of logistic noise samples."""
    if sigma <= 0:
        return np.zeros(n)
    if rng is not None:
        return rng.logistic(0, sigma, size=n)
    return np.random.logistic(0, sigma, size=n)


# ═══════════════════════════════════════════════════════════════════════════
# Retrieval probability
# ═══════════════════════════════════════════════════════════════════════════

def retrieval_probability(
    activation: float,
    threshold: float = DEFAULT_RETRIEVAL_THRESHOLD,
    noise_s: float = DEFAULT_NOISE_S,
) -> float:
    """Compute the probability of successful retrieval.

    P(retrieve) = 1 / (1 + exp(-(A_i - τ) / s))

    This is the logistic (sigmoid) function of the activation-threshold
    difference, scaled by noise.

    Parameters:
        activation: Total activation A_i.
        threshold: Retrieval threshold τ.
        noise_s: Noise scale parameter s.

    Returns:
        Probability of retrieval in [0, 1].

    Reference: Anderson (2007), Eq. 3.6.
    """
    if noise_s <= 0:
        return 1.0 if activation >= threshold else 0.0
    exponent = -(activation - threshold) / noise_s
    # Numerical stability
    if exponent > 500:
        return 0.0
    if exponent < -500:
        return 1.0
    return 1.0 / (1.0 + math.exp(exponent))


def retrieval_probability_batch(
    activations: np.ndarray,
    threshold: float = DEFAULT_RETRIEVAL_THRESHOLD,
    noise_s: float = DEFAULT_NOISE_S,
) -> np.ndarray:
    """Vectorized retrieval probability for an array of activations."""
    if noise_s <= 0:
        return (activations >= threshold).astype(float)
    exponent = -(activations - threshold) / noise_s
    exponent = np.clip(exponent, -500, 500)
    return 1.0 / (1.0 + np.exp(exponent))


# ═══════════════════════════════════════════════════════════════════════════
# Retrieval latency
# ═══════════════════════════════════════════════════════════════════════════

def retrieval_time(
    activation: float,
    latency_factor: float = DEFAULT_LATENCY_FACTOR,
    latency_exponent: float = DEFAULT_LATENCY_EXPONENT,
) -> float:
    """Compute retrieval time from activation.

    T = F * exp(-f * A_i)

    Higher activation → faster retrieval.

    Parameters:
        activation: Total activation A_i.
        latency_factor: Scaling factor F (default 0.63 s).
        latency_exponent: Exponent f (default 1.0).

    Returns:
        Retrieval time in seconds.

    Reference: Anderson (2007), Eq. 3.4.
    """
    exponent = -latency_exponent * activation
    if exponent > 500:
        return latency_factor * 1e10  # Effectively infinite
    return latency_factor * math.exp(exponent)


def retrieval_time_batch(
    activations: np.ndarray,
    latency_factor: float = DEFAULT_LATENCY_FACTOR,
    latency_exponent: float = DEFAULT_LATENCY_EXPONENT,
) -> np.ndarray:
    """Vectorized retrieval time for an array of activations."""
    exponents = np.clip(-latency_exponent * activations, -500, 500)
    return latency_factor * np.exp(exponents)


# ═══════════════════════════════════════════════════════════════════════════
# Memory decay curve
# ═══════════════════════════════════════════════════════════════════════════

def decay_curve(
    n_presentations: int,
    time_points: np.ndarray,
    presentation_time: float = 0.0,
    decay: float = DEFAULT_DECAY,
    threshold: float = DEFAULT_RETRIEVAL_THRESHOLD,
    noise_s: float = DEFAULT_NOISE_S,
) -> Dict[str, np.ndarray]:
    """Compute activation and retrieval probability over time.

    Models the classic forgetting curve: after n presentations at
    presentation_time, how does activation evolve?

    Parameters:
        n_presentations: Number of times the chunk was studied.
        time_points: Array of time points at which to evaluate.
        presentation_time: When the presentations occurred.
        decay: Decay parameter d.
        threshold: Retrieval threshold τ.
        noise_s: Noise scale s.

    Returns:
        Dict with 'time', 'activation', 'retrieval_prob', 'retrieval_time'.

    Reference: Anderson (2007), Ch. 3, Forgetting.
    """
    activations = np.zeros_like(time_points, dtype=float)
    for i, t in enumerate(time_points):
        age = t - presentation_time
        if age <= 0:
            activations[i] = 0.0
        else:
            # n presentations all at presentation_time
            activations[i] = math.log(n_presentations * age ** (-decay)) if age > 0 else 0.0

    probs = retrieval_probability_batch(activations, threshold, noise_s)
    times = retrieval_time_batch(activations)

    return {
        "time": time_points,
        "activation": activations,
        "retrieval_prob": probs,
        "retrieval_time": times,
    }


def spacing_effect(
    presentation_times: List[float],
    test_time: float,
    decay: float = DEFAULT_DECAY,
) -> Dict[str, float]:
    """Demonstrate the spacing effect on memory retention.

    Compares massed practice (all presentations close together) with
    spaced practice (presentations distributed over time).

    Reference: Pavlik & Anderson (2005).
    """
    activation = base_level_learning(presentation_times, test_time, decay)
    prob = retrieval_probability(activation)
    latency = retrieval_time(activation)

    return {
        "n_presentations": len(presentation_times),
        "test_time": test_time,
        "activation": activation,
        "retrieval_prob": prob,
        "retrieval_time": latency,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Interference model
# ═══════════════════════════════════════════════════════════════════════════

def fan_effect(
    target_activation: float,
    n_associated_chunks: int,
    max_association: float = 2.0,
) -> float:
    """Model the fan effect on retrieval.

    The fan effect causes retrieval slowdown when a cue is associated
    with many chunks, because spreading activation is diluted.

    Effective activation = target_activation + S_max - ln(fan)

    Reference: Anderson (2007), p. 73.
    """
    if n_associated_chunks <= 0:
        return target_activation + max_association
    fan_penalty = max_association - math.log(n_associated_chunks)
    return target_activation + fan_penalty


def proactive_interference(
    activation: float,
    n_prior_items: int,
    interference_rate: float = 0.05,
) -> float:
    """Model proactive interference from previously learned items.

    Each prior item reduces activation by interference_rate.

    Reference: Anderson & Lebiere (1998), Ch. 7.
    """
    penalty = interference_rate * n_prior_items
    return activation - penalty


def retroactive_interference(
    activation: float,
    n_intervening_items: int,
    interference_rate: float = 0.08,
) -> float:
    """Model retroactive interference from items learned after the target.

    Reference: Anderson & Lebiere (1998), Ch. 7.
    """
    penalty = interference_rate * n_intervening_items
    return activation - penalty

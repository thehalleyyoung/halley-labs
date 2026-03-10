"""ACT-R declarative memory system for usability modelling.

Implements the core declarative memory module from the ACT-R cognitive
architecture (Anderson & Lebiere, 1998; Anderson, 2007), including
base-level activation, spreading activation, partial matching, retrieval
time prediction, and memory decay.  Designed for vectorised computation
over many chunks using NumPy.

References
----------
Anderson, J. R. (2007). *How Can the Human Mind Occur in the Physical
    Universe?* Oxford University Press.
Anderson, J. R. & Lebiere, C. (1998). *The Atomic Components of Thought*.
    Lawrence Erlbaum Associates.
Anderson, J. R., Bothell, D., Byrne, M. D., Douglass, S., Lebiere, C.,
    & Qin, Y. (2004). An integrated theory of the mind. *Psychological
    Review*, 111(4), 1036-1060.
Pavlik, P. I. & Anderson, J. R. (2005). Practice and forgetting effects
    on vocabulary memory: An activation-based model of the spacing effect.
    *Cognitive Science*, 29(4), 559-586.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A declarative memory chunk (type–slot–value triple store).

    In ACT-R, knowledge is represented as chunks with a type and a set
    of named slots holding values (Anderson & Lebiere, 1998, Ch. 2).

    Attributes
    ----------
    name : str
        Unique identifier for this chunk.
    chunk_type : str
        Category type (e.g. ``"button"``, ``"menu-item"``).
    slots : dict[str, Any]
        Mapping of slot names to their values.
    creation_time : float
        Simulation time at which this chunk was created (seconds).
    access_times : list[float]
        Sorted list of times at which this chunk has been retrieved.
    """

    name: str
    chunk_type: str
    slots: Dict[str, Any] = field(default_factory=dict)
    creation_time: float = 0.0
    access_times: List[float] = field(default_factory=list)

    def record_access(self, time: float) -> None:
        """Record a retrieval of this chunk at *time*."""
        self.access_times.append(time)

    def slot_values(self) -> List[Any]:
        """Return slot values in sorted-key order."""
        return [self.slots[k] for k in sorted(self.slots)]

    def matches(self, pattern: Dict[str, Any]) -> bool:
        """Return True if every slot in *pattern* matches this chunk exactly."""
        for slot, value in pattern.items():
            if slot not in self.slots or self.slots[slot] != value:
                return False
        return True


# ---------------------------------------------------------------------------
# Declarative memory module
# ---------------------------------------------------------------------------


class ACTRDeclarativeMemory:
    """ACT-R declarative memory with activation-based retrieval.

    The total activation of chunk *i* is:

    .. math::

        A_i = B_i + S_i + P_i + \\varepsilon

    where *B_i* is the base-level activation, *S_i* is spreading
    activation from source chunks, *P_i* is the partial-matching
    penalty, and ε is logistic noise (Anderson, 2007, Ch. 3).

    Parameters
    ----------
    decay : float
        Base-level decay parameter *d* (default 0.5).
    latency_factor : float
        Retrieval latency factor *F* (seconds, default 1.0).
    latency_exponent : float
        Retrieval latency exponent *f* (default 1.0).
    retrieval_threshold : float
        Minimum activation for successful retrieval *τ* (default 0.0).
    noise_s : float
        Logistic noise scale *s* (default 0.25). Activation noise
        has variance π²s²/3.
    mismatch_penalty : float
        Maximum mismatch penalty *MP* (default 1.5).
    max_spreading : float
        Total source activation *W* (default 1.0).
    optimized_learning : bool
        If True, use the approximate base-level formula instead of
        the full sum (Anderson, 2007, Eq. 3.4).
    """

    def __init__(
        self,
        decay: float = 0.5,
        latency_factor: float = 1.0,
        latency_exponent: float = 1.0,
        retrieval_threshold: float = 0.0,
        noise_s: float = 0.25,
        mismatch_penalty: float = 1.5,
        max_spreading: float = 1.0,
        optimized_learning: bool = False,
    ) -> None:
        self.decay = decay
        self.latency_factor = latency_factor
        self.latency_exponent = latency_exponent
        self.retrieval_threshold = retrieval_threshold
        self.noise_s = noise_s
        self.mismatch_penalty = mismatch_penalty
        self.max_spreading = max_spreading
        self.optimized_learning = optimized_learning

        self._chunks: Dict[str, Chunk] = {}
        self._similarity_cache: Dict[Tuple[Any, Any], float] = {}
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------ #
    # Chunk management
    # ------------------------------------------------------------------ #

    def add_chunk(self, chunk: Chunk) -> None:
        """Add a chunk to declarative memory."""
        self._chunks[chunk.name] = chunk

    def get_chunk(self, name: str) -> Optional[Chunk]:
        """Return a chunk by name, or ``None``."""
        return self._chunks.get(name)

    @property
    def chunks(self) -> List[Chunk]:
        """All chunks in memory."""
        return list(self._chunks.values())

    @property
    def chunk_count(self) -> int:
        """Number of chunks in memory."""
        return len(self._chunks)

    def set_similarity(self, value_a: Any, value_b: Any, sim: float) -> None:
        """Set the similarity between two slot values.

        Similarity should be in [-1, 0] where 0 means identical and
        -1 means maximally dissimilar (Anderson & Lebiere, 1998, Ch. 4).
        """
        sim = float(np.clip(sim, -1.0, 0.0))
        self._similarity_cache[(value_a, value_b)] = sim
        self._similarity_cache[(value_b, value_a)] = sim

    def get_similarity(self, value_a: Any, value_b: Any) -> float:
        """Return similarity between two values (default -1 if unknown)."""
        if value_a == value_b:
            return 0.0
        return self._similarity_cache.get((value_a, value_b), -1.0)

    # ------------------------------------------------------------------ #
    # Base-level activation
    # ------------------------------------------------------------------ #

    def base_level_activation(
        self,
        chunk: Chunk,
        current_time: float,
    ) -> float:
        """Compute base-level activation *B_i*.

        Full equation (Anderson, 2007, Eq. 3.3):

        .. math::

            B_i = \\ln\\!\\left(\\sum_{j=1}^{n} t_j^{-d}\\right)

        where *t_j* = ``current_time − access_times[j]`` and *d* is the
        decay parameter.

        The optimized-learning approximation (Eq. 3.4) is:

        .. math::

            B_i = \\ln\\!\\left(\\frac{n}{1-d}\\right) -
                  d \\cdot \\ln(L)

        where *n* is the access count and *L* is the lifetime of the
        chunk.

        Parameters
        ----------
        chunk : Chunk
            Target chunk.
        current_time : float
            Current simulation time (seconds).

        Returns
        -------
        float
            Base-level activation value.
        """
        times = chunk.access_times
        if not times:
            # No references — use creation time as single access
            elapsed = max(current_time - chunk.creation_time, 0.001)
            return -self.decay * math.log(elapsed)

        if self.optimized_learning:
            return self._base_level_optimized(chunk, current_time)

        return self._base_level_exact(chunk, current_time)

    def _base_level_exact(self, chunk: Chunk, current_time: float) -> float:
        """Full summation base-level (vectorised)."""
        times = np.asarray(chunk.access_times, dtype=np.float64)
        ages = np.maximum(current_time - times, 0.001)
        powered = np.power(ages, -self.decay)
        return float(np.log(np.sum(powered)))

    def _base_level_optimized(self, chunk: Chunk, current_time: float) -> float:
        """Optimized-learning approximation (Anderson, 2007, Eq. 3.4)."""
        n = len(chunk.access_times)
        if n == 0:
            elapsed = max(current_time - chunk.creation_time, 0.001)
            return -self.decay * math.log(elapsed)
        lifetime = max(current_time - chunk.creation_time, 0.001)
        return math.log(n / (1.0 - self.decay)) - self.decay * math.log(lifetime)

    # ------------------------------------------------------------------ #
    # Base-level activation — vectorised over all chunks
    # ------------------------------------------------------------------ #

    def base_level_all(self, current_time: float) -> NDArray[np.floating]:
        """Compute base-level activation for every chunk (vectorised).

        Returns
        -------
        numpy.ndarray
            Array of shape ``(chunk_count,)`` with base-level values.
        """
        result = np.empty(self.chunk_count, dtype=np.float64)
        for i, chunk in enumerate(self._chunks.values()):
            result[i] = self.base_level_activation(chunk, current_time)
        return result

    # ------------------------------------------------------------------ #
    # Spreading activation
    # ------------------------------------------------------------------ #

    def spreading_activation(
        self,
        chunk: Chunk,
        source_chunks: Sequence[Chunk],
    ) -> float:
        """Compute spreading activation *S_i* from source chunks.

        .. math::

            S_i = \\sum_j W_j \\cdot S_{ji}

        where *W_j* is the attentional weight of source *j* (evenly
        divided among sources so they sum to ``max_spreading``), and
        *S_ji* is the associative strength from source *j* to chunk *i*
        (Anderson, 2007, Ch. 3).

        The associative strength uses a fan-based computation:

        .. math::

            S_{ji} = S - \\ln(\\text{fan}_j)

        where *S* = ``max_spreading`` and fan_j counts the chunks
        associated with source *j*.

        Parameters
        ----------
        chunk : Chunk
            Target chunk whose spreading activation is computed.
        source_chunks : sequence of Chunk
            Chunks currently in the focus of attention (buffers).

        Returns
        -------
        float
            Total spreading activation.
        """
        if not source_chunks:
            return 0.0

        n_sources = len(source_chunks)
        w_j = self.max_spreading / n_sources

        total = 0.0
        for source in source_chunks:
            s_ji = self._associative_strength(source, chunk)
            total += w_j * s_ji
        return total

    def _associative_strength(self, source: Chunk, target: Chunk) -> float:
        """Fan-based associative strength S_ji."""
        # Count how many chunks share a slot value with the source
        fan = 1  # self-association
        source_values = set(source.slots.values())
        for c in self._chunks.values():
            if c.name == source.name:
                continue
            if source_values & set(c.slots.values()):
                fan += 1

        # Check if target shares values with source
        target_values = set(target.slots.values())
        if not (source_values & target_values):
            return 0.0

        return max(0.0, self.max_spreading - math.log(fan))

    def spreading_activation_vectorised(
        self,
        source_chunks: Sequence[Chunk],
    ) -> NDArray[np.floating]:
        """Compute spreading activation for all chunks from sources.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(chunk_count,)`` with spreading values.
        """
        result = np.zeros(self.chunk_count, dtype=np.float64)
        if not source_chunks:
            return result
        for i, chunk in enumerate(self._chunks.values()):
            result[i] = self.spreading_activation(chunk, source_chunks)
        return result

    # ------------------------------------------------------------------ #
    # Partial matching
    # ------------------------------------------------------------------ #

    def partial_match_penalty(
        self,
        chunk: Chunk,
        request: Dict[str, Any],
    ) -> float:
        """Compute partial-matching penalty *P_i*.

        .. math::

            P_i = \\sum_k M_{ki} \\cdot MP

        where *M_ki* is the similarity between the requested value for
        slot *k* and the chunk's actual value, and *MP* is the mismatch
        penalty parameter (Anderson & Lebiere, 1998, Ch. 4).

        Parameters
        ----------
        chunk : Chunk
            Candidate chunk.
        request : dict
            Slot-value pattern for the retrieval request.

        Returns
        -------
        float
            Penalty (non-positive).
        """
        penalty = 0.0
        for slot, requested_value in request.items():
            chunk_value = chunk.slots.get(slot)
            if chunk_value is None:
                penalty += -self.mismatch_penalty
            else:
                sim = self.get_similarity(requested_value, chunk_value)
                penalty += self.mismatch_penalty * sim
        return penalty

    def partial_match_vectorised(
        self,
        request: Dict[str, Any],
    ) -> NDArray[np.floating]:
        """Compute partial-match penalties for all chunks."""
        result = np.zeros(self.chunk_count, dtype=np.float64)
        for i, chunk in enumerate(self._chunks.values()):
            result[i] = self.partial_match_penalty(chunk, request)
        return result

    # ------------------------------------------------------------------ #
    # Total activation
    # ------------------------------------------------------------------ #

    def activation(
        self,
        chunk: Chunk,
        current_time: float,
        source_chunks: Optional[Sequence[Chunk]] = None,
        request: Optional[Dict[str, Any]] = None,
        add_noise: bool = True,
    ) -> float:
        """Compute total activation for a single chunk.

        .. math::

            A_i = B_i + S_i + P_i + \\varepsilon

        Parameters
        ----------
        chunk : Chunk
            Target chunk.
        current_time : float
            Current simulation time.
        source_chunks : sequence of Chunk, optional
            Source chunks for spreading activation.
        request : dict, optional
            Retrieval request for partial matching.
        add_noise : bool
            Whether to add logistic noise.

        Returns
        -------
        float
            Total activation value.
        """
        a = self.base_level_activation(chunk, current_time)

        if source_chunks:
            a += self.spreading_activation(chunk, source_chunks)

        if request is not None:
            a += self.partial_match_penalty(chunk, request)

        if add_noise and self.noise_s > 0:
            a += self._activation_noise()

        return a

    def activation_all(
        self,
        current_time: float,
        source_chunks: Optional[Sequence[Chunk]] = None,
        request: Optional[Dict[str, Any]] = None,
        add_noise: bool = True,
    ) -> NDArray[np.floating]:
        """Compute activation for every chunk (vectorised).

        Returns
        -------
        numpy.ndarray
            Array of shape ``(chunk_count,)`` with activation values.
        """
        base = self.base_level_all(current_time)
        spreading = self.spreading_activation_vectorised(
            source_chunks or []
        )
        partial = (
            self.partial_match_vectorised(request)
            if request is not None
            else np.zeros(self.chunk_count, dtype=np.float64)
        )
        total = base + spreading + partial

        if add_noise and self.noise_s > 0:
            noise = self._rng.logistic(0.0, self.noise_s, self.chunk_count)
            total += noise

        return total

    def _activation_noise(self) -> float:
        """Sample a single logistic noise value."""
        return float(self._rng.logistic(0.0, self.noise_s))

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    def retrieval_time(self, activation: float) -> float:
        """Compute retrieval latency from activation.

        .. math::

            T = F \\cdot e^{-f \\cdot A_i}

        Parameters
        ----------
        activation : float
            Total activation of the chunk being retrieved.

        Returns
        -------
        float
            Retrieval time in seconds.
        """
        return self.latency_factor * math.exp(
            -self.latency_exponent * activation
        )

    def retrieval_probability(self, activation: float) -> float:
        """Probability of successful retrieval given activation.

        Uses the logistic retrieval probability:

        .. math::

            P = \\frac{1}{1 + e^{(\\tau - A_i) / s}}

        Parameters
        ----------
        activation : float
            Total activation of the chunk.

        Returns
        -------
        float
            Retrieval probability in [0, 1].
        """
        if self.noise_s <= 0:
            return 1.0 if activation >= self.retrieval_threshold else 0.0
        exponent = (self.retrieval_threshold - activation) / self.noise_s
        exponent = float(np.clip(exponent, -500.0, 500.0))
        return 1.0 / (1.0 + math.exp(exponent))

    def retrieve(
        self,
        request: Dict[str, Any],
        current_time: float,
        source_chunks: Optional[Sequence[Chunk]] = None,
        partial: bool = False,
    ) -> Tuple[Optional[Chunk], float]:
        """Attempt to retrieve a chunk matching *request*.

        Returns the highest-activation chunk whose activation exceeds
        the retrieval threshold, along with the retrieval time.

        Parameters
        ----------
        request : dict
            Slot-value pattern to match.
        current_time : float
            Current simulation time.
        source_chunks : sequence of Chunk, optional
            Source chunks for spreading activation.
        partial : bool
            If True, use partial matching; otherwise only exact matches.

        Returns
        -------
        tuple[Chunk | None, float]
            (retrieved chunk or None on failure, retrieval time).
        """
        best_chunk: Optional[Chunk] = None
        best_activation = -float("inf")

        for chunk in self._chunks.values():
            if not partial and not chunk.matches(request):
                continue

            act = self.activation(
                chunk,
                current_time,
                source_chunks=source_chunks,
                request=request if partial else None,
                add_noise=True,
            )

            if act > best_activation:
                best_activation = act
                best_chunk = chunk

        if best_chunk is None or best_activation < self.retrieval_threshold:
            rt = self.retrieval_time(self.retrieval_threshold)
            return None, rt

        best_chunk.record_access(current_time)
        rt = self.retrieval_time(best_activation)
        return best_chunk, rt

    # ------------------------------------------------------------------ #
    # Decay modelling
    # ------------------------------------------------------------------ #

    def decay_curve(
        self,
        n_presentations: int,
        delays: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Compute activation over time for a chunk with *n* presentations.

        Assumes presentations are evenly spaced before delay onset.

        Parameters
        ----------
        n_presentations : int
            Number of times the chunk was presented/rehearsed.
        delays : numpy.ndarray
            Array of retention intervals (seconds after last presentation).

        Returns
        -------
        numpy.ndarray
            Base-level activation at each delay.
        """
        n_presentations = max(1, int(n_presentations))
        delays = np.asarray(delays, dtype=np.float64)
        delays = np.maximum(delays, 0.001)

        # Approximate: all presentations at time 0, then measure at delay
        powered = n_presentations * np.power(delays, -self.decay)
        return np.log(powered)

    def forgetting_probability(
        self,
        chunk: Chunk,
        current_time: float,
    ) -> float:
        """Probability that retrieval of *chunk* fails at *current_time*.

        Returns
        -------
        float
            Forgetting probability in [0, 1].
        """
        act = self.base_level_activation(chunk, current_time)
        return 1.0 - self.retrieval_probability(act)

    # ------------------------------------------------------------------ #
    # Interference
    # ------------------------------------------------------------------ #

    def fan_effect(self, chunk: Chunk) -> int:
        """Compute the fan of *chunk* (number of associations).

        The fan of a chunk is the number of other chunks that share at
        least one slot value with it.  Higher fan reduces retrieval speed
        (Anderson, 1974).

        Returns
        -------
        int
            Fan count (>= 1, counting the chunk itself).
        """
        target_values = set(chunk.slots.values())
        fan = 1
        for other in self._chunks.values():
            if other.name == chunk.name:
                continue
            if target_values & set(other.slots.values()):
                fan += 1
        return fan

    def interference_penalty(self, chunk: Chunk) -> float:
        """Activation penalty from fan-based interference.

        Higher fan (more associations) reduces the strength of each
        association, resulting in lower activation.

        Returns
        -------
        float
            Negative penalty value (higher fan → more negative).
        """
        fan = self.fan_effect(chunk)
        return -math.log(fan)

    def interference_all(self) -> NDArray[np.floating]:
        """Compute interference penalties for all chunks (vectorised).

        Returns
        -------
        numpy.ndarray
            Array of shape ``(chunk_count,)`` with interference values.
        """
        result = np.empty(self.chunk_count, dtype=np.float64)
        for i, chunk in enumerate(self._chunks.values()):
            result[i] = self.interference_penalty(chunk)
        return result

    # ------------------------------------------------------------------ #
    # Batch retrieval utilities
    # ------------------------------------------------------------------ #

    def retrieval_times_all(
        self,
        current_time: float,
        source_chunks: Optional[Sequence[Chunk]] = None,
    ) -> NDArray[np.floating]:
        """Compute retrieval times for every chunk (vectorised).

        Returns
        -------
        numpy.ndarray
            Array of shape ``(chunk_count,)`` with retrieval times in
            seconds.
        """
        activations = self.activation_all(
            current_time, source_chunks=source_chunks, add_noise=False
        )
        return self.latency_factor * np.exp(
            -self.latency_exponent * activations
        )

    def retrieval_probabilities_all(
        self,
        current_time: float,
        source_chunks: Optional[Sequence[Chunk]] = None,
    ) -> NDArray[np.floating]:
        """Compute retrieval probabilities for every chunk (vectorised).

        Returns
        -------
        numpy.ndarray
            Array of shape ``(chunk_count,)`` with probabilities in [0, 1].
        """
        activations = self.activation_all(
            current_time, source_chunks=source_chunks, add_noise=False
        )
        if self.noise_s <= 0:
            return (activations >= self.retrieval_threshold).astype(np.float64)
        exponent = (self.retrieval_threshold - activations) / self.noise_s
        exponent = np.clip(exponent, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(exponent))

    # ------------------------------------------------------------------ #
    # Memory merge / strengthening
    # ------------------------------------------------------------------ #

    def merge_chunk(self, chunk: Chunk, current_time: float) -> Chunk:
        """Merge *chunk* into memory, strengthening if it already exists.

        If a chunk with the same name exists, the new access time is
        appended.  Otherwise the chunk is added fresh.

        Returns
        -------
        Chunk
            The merged (existing or new) chunk.
        """
        existing = self._chunks.get(chunk.name)
        if existing is not None:
            existing.record_access(current_time)
            # Update any differing slots
            existing.slots.update(chunk.slots)
            return existing
        chunk.record_access(current_time)
        self.add_chunk(chunk)
        return chunk

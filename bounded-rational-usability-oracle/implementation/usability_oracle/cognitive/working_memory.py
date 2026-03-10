"""Working memory models for predicting cognitive load in interactive systems.

Implements computational models of human working memory grounded in the
multi-component model (Baddeley, 2000), the capacity limit of attention
(Cowan, 2001), and the classic chunking framework (Miller, 1956).  These
models predict recall probability, rehearsal costs, and interference effects
that arise when users must maintain information while interacting with a
user interface.

References
----------
Miller, G. A. (1956). The magical number seven, plus or minus two: Some
    limits on our capacity for processing information. Psychological
    Review, 63(2), 81-97.
Baddeley, A. D. (2000). The episodic buffer: A new component of working
    memory? Trends in Cognitive Sciences, 4(11), 417-423.
Cowan, N. (2001). The magical number 4 in short-term memory: A
    reconsideration of mental storage capacity. Behavioral and Brain
    Sciences, 24(1), 87-114.
Oberauer, K., & Lewandowsky, S. (2011). Modeling working memory: A
    computational implementation of the Time-Based Resource-Sharing
    model. Psychonomic Bulletin & Review, 18(1), 10-45.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from usability_oracle.interval.interval import Interval


class WorkingMemoryModel:
    """Model of human working memory capacity, decay and interference.

    Central assumptions (Cowan, 2001):
    * The focus of attention holds approximately 4 ± 1 items.
    * Unattended items decay exponentially with a half-life of ~9 s.
    * Chunks formed by long-term-memory associations reduce the
      effective load.
    * Proactive and similarity-based interference degrade recall.

    All times are in seconds unless otherwise noted.
    """

    # --- Capacity parameters (Cowan, 2001; Miller, 1956) --------------------

    DEFAULT_CAPACITY: int = 4
    """Central capacity limit: ~4 chunks (Cowan, 2001)."""

    MIN_CAPACITY: int = 3
    """Lower bound of the typical capacity range."""

    MAX_CAPACITY: int = 7
    """Upper bound (Miller's 7 ± 2 for well-chunked material)."""

    # --- Decay parameters ---------------------------------------------------

    DEFAULT_DECAY_RATE: float = 0.077
    """Exponential decay rate (λ) such that half-life ≈ 9 s:
    λ = ln(2) / 9 ≈ 0.077 (Oberauer & Lewandowsky, 2011)."""

    # --- Core predictions ---------------------------------------------------

    @staticmethod
    def predict_recall_probability(
        items: int,
        delay: float,
        capacity: int = DEFAULT_CAPACITY,
        decay_rate: float = DEFAULT_DECAY_RATE,
    ) -> float:
        """Predict the probability of correctly recalling all items.

        When the number of items is within capacity, recall probability
        depends only on temporal decay.  When items exceed capacity, an
        additional factor accounts for the proportion of items that can
        be maintained in the focus of attention (Cowan, 2001).

        Parameters
        ----------
        items : int
            Number of distinct information items to remember.
        delay : float
            Retention interval in seconds.
        capacity : int
            Working-memory slot count.
        decay_rate : float
            Exponential decay constant (1/s).

        Returns
        -------
        float
            Recall probability in [0, 1].
        """
        items = max(1, int(items))
        delay = max(0.0, float(delay))
        capacity = max(1, int(capacity))

        decay = math.exp(-decay_rate * delay)

        if items <= capacity:
            probability = decay
        else:
            probability = (capacity / items) * decay

        return float(np.clip(probability, 0.0, 1.0))

    @staticmethod
    def predict_recall_interval(
        items: int,
        delay: Interval,
        capacity: int = DEFAULT_CAPACITY,
        decay_rate: float = DEFAULT_DECAY_RATE,
    ) -> Interval:
        """Interval-valued prediction of recall probability.

        Propagates uncertainty in the retention delay through the recall
        model using monotone interval arithmetic.  Because recall is a
        monotonically *decreasing* function of delay, the lower bound
        on probability corresponds to the *upper* bound on delay and
        vice-versa.

        Parameters
        ----------
        items : int
            Number of items.
        delay : Interval
            Uncertain retention interval [low, high] in seconds.
        capacity : int
            Working-memory capacity.
        decay_rate : float
            Exponential decay rate.

        Returns
        -------
        Interval
            Interval enclosing the true recall probability.
        """
        items = max(1, int(items))
        capacity = max(1, int(capacity))

        low_delay = max(0.0, delay.low)
        high_delay = max(0.0, delay.high)

        # recall is monotone decreasing in delay, so bounds swap
        p_high = WorkingMemoryModel.predict_recall_probability(
            items, low_delay, capacity, decay_rate
        )
        p_low = WorkingMemoryModel.predict_recall_probability(
            items, high_delay, capacity, decay_rate
        )
        return Interval(p_low, p_high)

    # --- Chunking -----------------------------------------------------------

    @staticmethod
    def chunk_count(
        elements: list,
        grouping: list[list[int]],
    ) -> int:
        """Count the number of working-memory chunks.

        Each entry in *grouping* specifies the indices of elements that
        form a single chunk (e.g., through spatial proximity or semantic
        association).  Elements not included in any group are treated as
        individual chunks (Miller, 1956).

        Parameters
        ----------
        elements : list
            The raw list of display elements.
        grouping : list[list[int]]
            Groups of element indices that form chunks.

        Returns
        -------
        int
            Total number of chunks.
        """
        grouped_indices: set[int] = set()
        n_chunks = 0
        for group in grouping:
            if group:
                n_chunks += 1
                grouped_indices.update(group)

        # Every ungrouped element counts as its own chunk
        for idx in range(len(elements)):
            if idx not in grouped_indices:
                n_chunks += 1

        return max(1, n_chunks)

    # --- Rehearsal ----------------------------------------------------------

    @staticmethod
    def rehearsal_cost(
        items: int,
        time_available: float,
        articulation_rate: float = 0.300,
    ) -> float:
        """Compute the temporal cost of sub-vocal rehearsal.

        The phonological loop can refresh items via rehearsal at a rate
        of approximately one item per 300 ms (Baddeley, 2000).  If the
        available time is insufficient to rehearse all items, decay
        proceeds unchecked on the unrehearsed portion.

        Parameters
        ----------
        items : int
            Number of items requiring rehearsal.
        time_available : float
            Time (seconds) available for rehearsal.
        articulation_rate : float
            Time per item for articulatory rehearsal (seconds).

        Returns
        -------
        float
            Effective delay experienced by the unrehearsed items.  If
            all items can be rehearsed, returns 0.0.
        """
        items = max(1, int(items))
        time_available = max(0.0, float(time_available))
        time_needed = items * articulation_rate

        if time_available >= time_needed:
            # All items rehearsed successfully — no effective delay
            return 0.0

        # Fraction of items that could not be rehearsed
        unrehearsed_fraction = 1.0 - (time_available / time_needed)
        # Unrehearsed items experience the full delay since last refresh
        effective_delay = unrehearsed_fraction * time_needed
        return float(effective_delay)

    # --- Interference -------------------------------------------------------

    @staticmethod
    def interference_factor(
        similar_items: int,
        total_items: int,
    ) -> float:
        """Compute interference from visually or semantically similar items.

        High similarity among items increases confusion errors.  The
        interference factor multiplicatively inflates the effective
        memory load (Duncan & Humphreys, 1989 applied to memory).

        Parameters
        ----------
        similar_items : int
            Number of items that are similar to the target item.
        total_items : int
            Total number of items in memory.

        Returns
        -------
        float
            Multiplicative interference factor (>= 1.0).
        """
        total_items = max(1, int(total_items))
        similar_items = int(np.clip(similar_items, 0, total_items))
        similarity_ratio = similar_items / total_items
        return 1.0 + 0.5 * similarity_ratio

    # --- Load cost ----------------------------------------------------------

    @staticmethod
    def load_cost(
        current_load: int,
        capacity: int = DEFAULT_CAPACITY,
    ) -> float:
        """Cost multiplier as working-memory load approaches capacity.

        Performance degrades super-linearly as the number of maintained
        items approaches the capacity limit.  The model uses a
        resonance-like function that diverges at capacity, capturing the
        dramatic increase in errors and RT near the limit (Cowan, 2001).

        Parameters
        ----------
        current_load : int
            Number of items currently held in working memory.
        capacity : int
            Working-memory capacity.

        Returns
        -------
        float
            Multiplicative cost factor (>= 1.0).  Returns a large
            constant (10.0) when load meets or exceeds capacity.
        """
        capacity = max(1, int(capacity))
        current_load = max(0, int(current_load))

        if current_load >= capacity:
            return 10.0  # Saturated — severe performance degradation

        ratio = current_load / capacity
        denominator = 1.0 - ratio ** 2
        return 1.0 / denominator

    # --- Proactive interference ---------------------------------------------

    @staticmethod
    def proactive_interference(
        prior_items: int,
        delay: float,
        release_rate: float = 0.1,
    ) -> float:
        """Interference from items encoded in a prior task or trial.

        Proactive interference (PI) builds with the number of
        previously memorised items and decays over time as the memory
        traces become less active (Keppel & Underwood, 1962).

        Parameters
        ----------
        prior_items : int
            Number of items from prior episodes still in memory.
        delay : float
            Time since the prior items were relevant (seconds).
        release_rate : float
            Rate at which PI dissipates (1/s).

        Returns
        -------
        float
            Interference magnitude (non-negative).  Zero means no
            interference.
        """
        prior_items = max(0, int(prior_items))
        delay = max(0.0, float(delay))
        return prior_items * math.exp(-release_rate * delay)

    # --- Composite cost -----------------------------------------------------

    @staticmethod
    def total_memory_cost(
        items: int,
        delay: float,
        similar_items: int = 0,
        prior_items: int = 0,
        capacity: int = DEFAULT_CAPACITY,
        decay_rate: float = DEFAULT_DECAY_RATE,
    ) -> float:
        """Combined working-memory cost integrating all factors.

        Computes a composite cost metric (higher = harder) by combining
        recall failure probability, load cost, similarity-based
        interference and proactive interference.

        Parameters
        ----------
        items : int
            Items to maintain.
        delay : float
            Retention interval (seconds).
        similar_items : int
            Number of similar items (for interference).
        prior_items : int
            Items from prior episodes (for proactive interference).
        capacity : int
            Working-memory capacity.
        decay_rate : float
            Temporal decay rate.

        Returns
        -------
        float
            Total cost (non-negative, unitless).  Higher values indicate
            greater difficulty.
        """
        items = max(1, int(items))
        delay = max(0.0, float(delay))

        # Base recall failure probability
        recall_p = WorkingMemoryModel.predict_recall_probability(
            items, delay, capacity, decay_rate
        )
        failure_cost = 1.0 - recall_p

        # Load cost
        lc = WorkingMemoryModel.load_cost(items, capacity)

        # Interference
        intf = WorkingMemoryModel.interference_factor(similar_items, items)

        # Proactive interference
        pi = WorkingMemoryModel.proactive_interference(prior_items, delay)

        # Composite: multiplicative interaction of load and interference,
        # additive contribution of failure probability and PI.
        cost = (failure_cost * lc * intf) + pi
        return float(max(0.0, cost))

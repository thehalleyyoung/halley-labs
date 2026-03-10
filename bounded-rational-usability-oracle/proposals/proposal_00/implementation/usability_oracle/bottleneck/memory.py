"""
usability_oracle.bottleneck.memory — Memory decay detector.

Detects states where working-memory demands exceed the user's capacity,
based on Cowan's capacity limit (4 ± 1 chunks) and exponential decay:

    P(recall) = exp(−λ · t)

where λ = ln(2) / t_half (half-life ≈ 7 seconds for unrehearsted items).

The information retained after delay *t* is:

    I_retained(t) = I_original · P(recall | t)

Memory overload occurs when:
  1. The number of items in working memory exceeds 4 ± 1 (Miller/Cowan), or
  2. Cross-page memory demand requires recalling information from previous
     pages without visible cues.

References
----------
- Cowan, N. (2001). The magical number 4 in short-term memory. *BBS*.
- Miller, G. A. (1956). The magical number seven, plus or minus two. *PR*.
- Baddeley, A. (2003). Working memory: looking back and looking forward.
  *Nature Reviews Neuroscience*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from usability_oracle.bottleneck.models import BottleneckResult
from usability_oracle.core.enums import BottleneckType, CognitiveLaw, Severity
from usability_oracle.mdp.models import MDP


# ---------------------------------------------------------------------------
# MemoryDecayDetector
# ---------------------------------------------------------------------------

@dataclass
class MemoryDecayDetector:
    """Detect working-memory overload at UI states.

    Parameters
    ----------
    OVERLOAD_THRESHOLD : int
        Maximum items in working memory before overload (Cowan's 4 ± 1).
        Default 4.
    DECAY_HALF_LIFE : float
        Half-life of working memory items (seconds).  Default 7.0.
    RECALL_THRESHOLD : float
        Minimum recall probability below which information is considered lost.
        Default 0.5.
    CROSS_PAGE_THRESHOLD : int
        Maximum items that must be remembered across page transitions.
        Default 2.
    """

    OVERLOAD_THRESHOLD: int = 4
    DECAY_HALF_LIFE: float = 7.0
    RECALL_THRESHOLD: float = 0.5
    CROSS_PAGE_THRESHOLD: int = 2

    # ── Public API --------------------------------------------------------

    def detect(
        self,
        trajectory: list[Any],
        state: str,
        mdp: MDP,
    ) -> Optional[BottleneckResult]:
        """Detect memory decay bottleneck at *state*.

        Parameters
        ----------
        trajectory : list
            Recent trajectory steps (list of dicts or TrajectoryStep objects).
        state : str
            Current state identifier.
        mdp : MDP

        Returns
        -------
        BottleneckResult or None
        """
        state_obj = mdp.states.get(state)
        if state_obj is None:
            return None

        features = state_obj.features

        # Compute working memory load from state features
        wm_load = self._working_memory_load(features)

        # Compute cross-page memory demand from trajectory
        cross_page_demand = self._cross_page_memory_demand(trajectory)

        # Estimate recall probability for oldest item
        delay = self._estimate_delay(trajectory)
        recall_prob = self._recall_probability(wm_load, delay)

        # Information retained
        original_bits = wm_load * 2.5  # ~2.5 bits per chunk (conservative)
        retained_bits = self._information_retained(original_bits, delay)

        # Build evidence
        evidence: dict[str, float] = {
            "working_memory_load": float(wm_load),
            "cross_page_demand": float(cross_page_demand),
            "recall_probability": recall_prob,
            "delay_seconds": delay,
            "information_original_bits": original_bits,
            "information_retained_bits": retained_bits,
            "overload_threshold": float(self.OVERLOAD_THRESHOLD),
        }

        # Detection logic
        is_overloaded = False
        confidence = 0.0
        issues: list[str] = []

        # Check 1: Working memory capacity exceeded
        if wm_load > self.OVERLOAD_THRESHOLD:
            is_overloaded = True
            excess = wm_load - self.OVERLOAD_THRESHOLD
            confidence = max(confidence, min(1.0, 0.5 + excess * 0.15))
            issues.append(
                f"WM load ({wm_load}) exceeds capacity ({self.OVERLOAD_THRESHOLD})"
            )

        # Check 2: Cross-page memory demand
        if cross_page_demand > self.CROSS_PAGE_THRESHOLD:
            is_overloaded = True
            excess = cross_page_demand - self.CROSS_PAGE_THRESHOLD
            confidence = max(confidence, min(1.0, 0.4 + excess * 0.2))
            issues.append(
                f"Cross-page demand ({cross_page_demand}) "
                f"exceeds threshold ({self.CROSS_PAGE_THRESHOLD})"
            )

        # Check 3: Low recall probability
        if recall_prob < self.RECALL_THRESHOLD and wm_load > 2:
            is_overloaded = True
            confidence = max(confidence, min(1.0, 1.0 - recall_prob))
            issues.append(
                f"Recall probability ({recall_prob:.2f}) "
                f"below threshold ({self.RECALL_THRESHOLD})"
            )

        if not is_overloaded:
            return None

        severity = self._severity_from_load(wm_load, cross_page_demand, recall_prob)

        return BottleneckResult(
            bottleneck_type=BottleneckType.MEMORY_DECAY,
            severity=severity,
            confidence=confidence,
            affected_states=[state],
            affected_actions=mdp.get_actions(state),
            cognitive_law=CognitiveLaw.WORKING_MEMORY_DECAY,
            channel="cognitive",
            evidence=evidence,
            description=(
                f"Memory overload at state {state!r}: "
                f"WM load={wm_load}, cross-page={cross_page_demand}, "
                f"recall={recall_prob:.2f}. {'; '.join(issues)}"
            ),
            recommendation=(
                "Reduce working-memory demands by showing information in context, "
                "reducing multi-step workflows, or adding persistent visible cues."
            ),
            repair_hints=[
                "Reduce number of items user must remember simultaneously",
                "Add persistent visual cues or breadcrumbs",
                "Show previously-entered information on subsequent pages",
                "Reduce number of steps in multi-page workflows",
                "Provide summary/review before final submission",
            ],
        )

    # ── Working memory model ----------------------------------------------

    def _working_memory_load(self, state_features: dict[str, float]) -> int:
        """Estimate the number of items in working memory from state features.

        Uses ``working_memory_load`` feature if available, otherwise estimates
        from the number of form fields, choices, and visible information items.

        Parameters
        ----------
        state_features : dict[str, float]

        Returns
        -------
        int
            Estimated number of WM items (chunks).
        """
        # Direct feature
        if "working_memory_load" in state_features:
            return int(state_features["working_memory_load"])

        load = 0
        # Count items that require memorisation
        load += int(state_features.get("n_form_fields", 0))
        load += int(state_features.get("n_required_decisions", 0))
        load += int(state_features.get("n_cross_references", 0))

        # If nothing specific, estimate from general complexity
        if load == 0:
            n_elements = int(state_features.get("n_elements", 0))
            # Rough heuristic: ~20% of elements need cognitive tracking
            load = max(1, n_elements // 5)

        return load

    def _recall_probability(self, items: int, delay: float) -> float:
        """Compute probability of recalling all items after *delay* seconds.

        P(recall all) = (P(recall one))^items
        P(recall one) = exp(-λ · t)

        where λ = ln(2) / t_half.

        Parameters
        ----------
        items : int
            Number of items in working memory.
        delay : float
            Time since encoding (seconds).

        Returns
        -------
        float
            Probability ∈ [0, 1].
        """
        if items <= 0 or delay <= 0:
            return 1.0

        decay_rate = math.log(2) / self.DECAY_HALF_LIFE
        p_single = math.exp(-decay_rate * delay)
        # Probability of recalling ALL items
        p_all = p_single ** items
        return max(0.0, min(1.0, p_all))

    def _information_retained(
        self,
        original_bits: float,
        delay: float,
    ) -> float:
        """Compute information retained after *delay*.

        I_retained = I_original · exp(-λ · t)

        Parameters
        ----------
        original_bits : float
            Original information in bits.
        delay : float
            Time since encoding (seconds).

        Returns
        -------
        float
            Retained information in bits.
        """
        if delay <= 0:
            return original_bits
        decay_rate = math.log(2) / self.DECAY_HALF_LIFE
        return original_bits * math.exp(-decay_rate * delay)

    def _cross_page_memory_demand(self, trajectory: list[Any]) -> int:
        """Estimate the number of items that must be remembered across pages.

        Heuristic: count unique states visited in the last N steps where
        information from a prior state is needed at the current state.

        Parameters
        ----------
        trajectory : list
            Recent trajectory (list of step dicts/objects).

        Returns
        -------
        int
            Cross-page memory demand.
        """
        if not trajectory:
            return 0

        # Track distinct "pages" (states) and information carried forward
        seen_states: set[str] = set()
        cross_page_items = 0

        for step in trajectory:
            if isinstance(step, dict):
                state_id = step.get("state", "")
            elif hasattr(step, "state"):
                state_id = step.state
            else:
                continue

            if state_id in seen_states:
                # Returning to a previously-seen state requires recall
                cross_page_items += 1
            seen_states.add(state_id)

        # Also count the number of distinct states (each transition may
        # require remembering context from the previous page)
        n_distinct = len(seen_states)
        if n_distinct > 3:
            cross_page_items += (n_distinct - 3)

        return cross_page_items

    def _estimate_delay(self, trajectory: list[Any]) -> float:
        """Estimate the time delay since the oldest relevant memory encoding.

        Uses timestamps if available, otherwise estimates from trajectory
        length × average step time.

        Parameters
        ----------
        trajectory : list

        Returns
        -------
        float
            Estimated delay in seconds.
        """
        if not trajectory:
            return 0.0

        # Try to use timestamps
        timestamps: list[float] = []
        for step in trajectory:
            ts = None
            if isinstance(step, dict):
                ts = step.get("timestamp")
            elif hasattr(step, "timestamp"):
                ts = step.timestamp
            if ts is not None and ts > 0:
                timestamps.append(float(ts))

        if len(timestamps) >= 2:
            return timestamps[-1] - timestamps[0]

        # Fallback: estimate ~3 seconds per step
        return len(trajectory) * 3.0

    # ── Helpers -----------------------------------------------------------

    def _severity_from_load(
        self,
        wm_load: int,
        cross_page: int,
        recall_prob: float,
    ) -> Severity:
        """Map memory metrics to severity."""
        score = (
            max(0, wm_load - self.OVERLOAD_THRESHOLD) * 0.5
            + max(0, cross_page - self.CROSS_PAGE_THRESHOLD) * 0.4
            + (1.0 - recall_prob) * 2.0
        )
        if score > 3.0:
            return Severity.CRITICAL
        elif score > 2.0:
            return Severity.HIGH
        elif score > 1.0:
            return Severity.MEDIUM
        elif score > 0.3:
            return Severity.LOW
        return Severity.INFO

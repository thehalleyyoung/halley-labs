"""
usability_oracle.scheduling.deadline — Deadline modelling for user tasks.

Detects, classifies, and analyses deadlines in UI interaction tasks.
Deadlines arise from UI timeouts, animation windows, transition durations,
and explicit time limits.  This module computes urgency, deadline-miss
probability under bounded rationality, and temporal demand functions.

The free-energy integration models deadline pressure as a modified reward
function with time-discount, so that a bounded-rational agent allocates
more information processing to tasks approaching their deadline.

References
----------
* Wickens, C. D. et al. (2015). *Engineering Psychology and Human
  Performance* (4th ed.).  Pearson.
* Ortega, P. A. & Braun, D. A. (2013). Thermodynamics as a Theory of
  Decision-Making with Information-Processing Costs.  *Proc. R. Soc. A*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from usability_oracle.scheduling.types import (
    DeadlineModel,
    Schedule,
    ScheduledTask,
)


# ═══════════════════════════════════════════════════════════════════════════
# Deadline classification
# ═══════════════════════════════════════════════════════════════════════════

@unique
class DeadlineType(Enum):
    """Classification of UI deadline sources."""

    TIMEOUT = "timeout"
    """Session or component timeout (hard deadline)."""

    ANIMATION = "animation"
    """Animation window requiring synchronised input (soft deadline)."""

    TRANSITION = "transition"
    """Page or state transition with limited response window."""

    USER_EXPECTATION = "user_expectation"
    """Implicit deadline from user's internal time expectations."""

    NONE = "none"
    """No deadline applies."""


@dataclass(frozen=True, slots=True)
class DeadlineInfo:
    """Rich deadline descriptor for a detected deadline.

    Attributes
    ----------
    deadline_type : DeadlineType
        Source classification.
    is_hard : bool
        True ⟹ missing the deadline causes task failure.
    absolute_deadline_s : float
        Absolute wall-clock deadline in seconds.
    source_description : str
        Human-readable source description.
    confidence : float
        Detection confidence in [0, 1].
    """

    deadline_type: DeadlineType = DeadlineType.NONE
    is_hard: bool = False
    absolute_deadline_s: float = float("inf")
    source_description: str = ""
    confidence: float = 1.0


# ═══════════════════════════════════════════════════════════════════════════
# DeadlineDetector — identify deadlines from UI context
# ═══════════════════════════════════════════════════════════════════════════

class DeadlineDetector:
    """Detect and classify deadlines from UI context information.

    Scans timeout values, animation durations, and transition parameters
    to produce :class:`DeadlineInfo` descriptors.

    Parameters
    ----------
    default_soft_margin_s : float
        Default margin before a soft deadline starts degrading utility.
    timeout_threshold_s : float
        Durations below this threshold are classified as hard deadlines;
        above as soft deadlines.
    """

    def __init__(
        self,
        default_soft_margin_s: float = 0.5,
        timeout_threshold_s: float = 5.0,
    ) -> None:
        self._soft_margin = default_soft_margin_s
        self._timeout_threshold = timeout_threshold_s

    def detect_from_context(
        self,
        context: Mapping[str, Any],
    ) -> DeadlineInfo:
        """Detect deadline from a UI context dictionary.

        Recognised keys:

        * ``timeout_s`` — session or component timeout.
        * ``animation_duration_s`` — animation window.
        * ``transition_duration_s`` — page transition duration.
        * ``user_expected_s`` — user's internal expectation.

        Parameters
        ----------
        context : Mapping[str, Any]
            UI context dictionary.

        Returns
        -------
        DeadlineInfo
        """
        if "timeout_s" in context:
            t = float(context["timeout_s"])
            return DeadlineInfo(
                deadline_type=DeadlineType.TIMEOUT,
                is_hard=True,
                absolute_deadline_s=t,
                source_description=f"Timeout at {t:.1f}s",
                confidence=0.95,
            )

        if "animation_duration_s" in context:
            t = float(context["animation_duration_s"])
            return DeadlineInfo(
                deadline_type=DeadlineType.ANIMATION,
                is_hard=False,
                absolute_deadline_s=t + self._soft_margin,
                source_description=f"Animation window {t:.1f}s",
                confidence=0.8,
            )

        if "transition_duration_s" in context:
            t = float(context["transition_duration_s"])
            is_hard = t < self._timeout_threshold
            return DeadlineInfo(
                deadline_type=DeadlineType.TRANSITION,
                is_hard=is_hard,
                absolute_deadline_s=t,
                source_description=f"Transition window {t:.1f}s",
                confidence=0.85,
            )

        if "user_expected_s" in context:
            t = float(context["user_expected_s"])
            return DeadlineInfo(
                deadline_type=DeadlineType.USER_EXPECTATION,
                is_hard=False,
                absolute_deadline_s=t,
                source_description=f"User expectation {t:.1f}s",
                confidence=0.6,
            )

        return DeadlineInfo()

    def to_deadline_model(self, info: DeadlineInfo) -> DeadlineModel:
        """Convert :class:`DeadlineInfo` to a :class:`DeadlineModel`.

        Parameters
        ----------
        info : DeadlineInfo

        Returns
        -------
        DeadlineModel
        """
        if info.deadline_type == DeadlineType.NONE:
            return DeadlineModel()

        hard_dl = info.absolute_deadline_s if info.is_hard else None
        soft_dl = info.absolute_deadline_s if not info.is_hard else None

        return DeadlineModel(
            hard_deadline_s=hard_dl,
            soft_deadline_s=soft_dl,
            urgency_decay_rate=0.1,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Urgency computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_urgency(
    time_remaining_s: float,
    estimated_completion_s: float,
) -> float:
    """Compute urgency as the ratio of estimated completion to remaining time.

    .. math::

        U = \\frac{C_{\\text{est}}}{T_{\\text{remaining}}}

    Urgency > 1 means the task is likely to miss its deadline.

    Parameters
    ----------
    time_remaining_s : float
        Time remaining until deadline.
    estimated_completion_s : float
        Estimated time to complete the task.

    Returns
    -------
    float
        Urgency value; higher is more urgent.

    Raises
    ------
    ValueError
        If estimated_completion_s is negative.
    """
    if estimated_completion_s < 0:
        raise ValueError("estimated_completion_s must be non-negative")
    if time_remaining_s <= 0:
        return float("inf")
    return estimated_completion_s / time_remaining_s


def urgency_batch(
    time_remaining: np.ndarray,
    estimated_completion: np.ndarray,
) -> np.ndarray:
    """Vectorised urgency for arrays of tasks.

    Parameters
    ----------
    time_remaining : np.ndarray
        Shape ``(n,)`` of remaining times.
    estimated_completion : np.ndarray
        Shape ``(n,)`` of estimated completion times.

    Returns
    -------
    np.ndarray
        Shape ``(n,)`` urgency values.
    """
    tr = np.asarray(time_remaining, dtype=np.float64)
    ec = np.asarray(estimated_completion, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(tr > 0, ec / tr, np.inf)
    return u


# ═══════════════════════════════════════════════════════════════════════════
# Deadline-miss probability under bounded rationality
# ═══════════════════════════════════════════════════════════════════════════

class DeadlineAnalyzer:
    """Analyse deadline-miss probability under bounded rationality.

    Implements the :class:`~usability_oracle.scheduling.protocols.DeadlinePredictor`
    protocol.

    The model assumes task completion time follows a log-normal distribution
    whose mean grows with cognitive cost and whose variance is modulated by
    the rationality parameter β.

    Parameters
    ----------
    duration_cv : float
        Coefficient of variation for task duration (σ / μ).
    """

    def __init__(self, duration_cv: float = 0.3) -> None:
        if duration_cv <= 0:
            raise ValueError("duration_cv must be positive")
        self._cv = duration_cv

    def predict_miss_probability(
        self,
        task: ScheduledTask,
        *,
        beta: float = 1.0,
    ) -> float:
        """Predict deadline-miss probability.

        Uses a log-normal completion-time model.  Lower β increases the
        variance, modelling greater human temporal variability.

        Parameters
        ----------
        task : ScheduledTask
            Task with deadline information.
        beta : float
            Rationality parameter (≥ 0).

        Returns
        -------
        float
            Miss probability in [0, 1].
        """
        if task.deadline is None or task.deadline.hard_deadline_s is None:
            return 0.0

        mu = task.estimated_duration_s
        if mu <= 0:
            return 0.0

        deadline = task.deadline.hard_deadline_s
        # Effective CV increases as β decreases (more variable behaviour)
        effective_cv = self._cv / max(beta, 0.01)

        # Log-normal parameters
        sigma_sq = math.log(1.0 + effective_cv ** 2)
        sigma = math.sqrt(sigma_sq)
        mu_ln = math.log(mu) - sigma_sq / 2.0

        # P(T > deadline) = 1 - Φ((ln(deadline) - μ_ln) / σ)
        if deadline <= 0:
            return 1.0
        z = (math.log(deadline) - mu_ln) / sigma
        miss_prob = 1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        return float(np.clip(miss_prob, 0.0, 1.0))

    def expected_completion_time(
        self,
        task: ScheduledTask,
        *,
        beta: float = 1.0,
    ) -> float:
        """Expected completion time under bounded rationality.

        Higher β (more rational) yields completion time closer to the
        estimated duration; lower β adds overhead from sub-optimal
        information processing.

        Parameters
        ----------
        task : ScheduledTask
        beta : float
            Rationality parameter.

        Returns
        -------
        float
            Expected seconds to complete.
        """
        mu = task.estimated_duration_s
        # Bounded-rational overhead: extra time ~ 1/β
        overhead_factor = 1.0 + (1.0 / max(beta, 0.01)) * self._cv
        return mu * overhead_factor

    def compute_slack(
        self,
        schedule: Schedule,
        task_id: str,
    ) -> Optional[float]:
        """Compute slack (spare time) for *task_id* in *schedule*.

        Parameters
        ----------
        schedule : Schedule
        task_id : str

        Returns
        -------
        Optional[float]
            Slack in seconds, or None if no deadline.
        """
        for tid, start, end in schedule.assignments:
            if tid == task_id:
                # We need the original task to know the deadline.
                # Approximate: use schedule metadata if available.
                return None  # requires task context
        return None

    def compute_slack_with_tasks(
        self,
        schedule: Schedule,
        task_id: str,
        tasks: Sequence[ScheduledTask],
    ) -> Optional[float]:
        """Compute slack with task deadline information.

        Parameters
        ----------
        schedule : Schedule
        task_id : str
        tasks : Sequence[ScheduledTask]

        Returns
        -------
        Optional[float]
            Slack in seconds, or None if no deadline applies.
        """
        task_map = {t.task_id: t for t in tasks}
        task = task_map.get(task_id)
        if task is None or task.deadline is None:
            return None
        dl = task.deadline.hard_deadline_s or task.deadline.soft_deadline_s
        if dl is None:
            return None

        for tid, start, end in schedule.assignments:
            if tid == task_id:
                abs_deadline = task.arrival_time_s + dl
                return abs_deadline - end
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Temporal demand function
# ═══════════════════════════════════════════════════════════════════════════

def temporal_demand_function(
    tasks: Sequence[ScheduledTask],
    time_points: np.ndarray,
) -> np.ndarray:
    """Compute the temporal demand function over a time horizon.

    For each time point *t*, the demand is the number of tasks whose
    ``[arrival, arrival + deadline]`` window includes *t*.

    Parameters
    ----------
    tasks : Sequence[ScheduledTask]
        Tasks with deadlines.
    time_points : np.ndarray
        Shape ``(m,)`` time points to evaluate.

    Returns
    -------
    np.ndarray
        Shape ``(m,)`` demand values (integer counts cast to float).
    """
    t = np.asarray(time_points, dtype=np.float64)
    demand = np.zeros_like(t)

    for task in tasks:
        arrival = task.arrival_time_s
        if task.deadline is not None and task.deadline.hard_deadline_s is not None:
            dl = arrival + task.deadline.hard_deadline_s
        elif task.deadline is not None and task.deadline.soft_deadline_s is not None:
            dl = arrival + task.deadline.soft_deadline_s
        else:
            dl = arrival + task.estimated_duration_s * 2.0  # heuristic

        mask = (t >= arrival) & (t <= dl)
        demand += mask.astype(np.float64)

    return demand


# ═══════════════════════════════════════════════════════════════════════════
# Free-energy integration: deadline pressure
# ═══════════════════════════════════════════════════════════════════════════

class FreeEnergyDeadlinePressure:
    """Model deadline pressure as a modification to the free-energy objective.

    Under the bounded-rational (free-energy) framework, the agent
    minimises:

    .. math::

        F = \\mathbb{E}_\\pi[C] + \\frac{1}{\\beta} D_{\\text{KL}}(\\pi \\| \\pi_0)

    where *C* is the cost (negative reward).  Deadline pressure modifies
    *C* by adding a time-dependent penalty:

    .. math::

        C_{\\text{deadline}}(t) = \\lambda \\cdot \\max(0, t - d) + \\gamma \\cdot e^{-\\delta(d - t)}

    where *d* is the deadline, λ is the penalty for lateness, γ is the
    anticipatory pressure, and δ is the temporal discount rate.

    Parameters
    ----------
    lateness_penalty : float
        λ — penalty per second of lateness.
    anticipatory_pressure : float
        γ — magnitude of anticipatory urgency before deadline.
    temporal_discount : float
        δ — exponential discount rate for approach to deadline.
    """

    def __init__(
        self,
        lateness_penalty: float = 1.0,
        anticipatory_pressure: float = 0.5,
        temporal_discount: float = 0.2,
    ) -> None:
        self._lambda = lateness_penalty
        self._gamma = anticipatory_pressure
        self._delta = temporal_discount

    def deadline_cost(
        self,
        current_time: float,
        deadline: float,
    ) -> float:
        """Compute deadline-pressure cost at *current_time*.

        Parameters
        ----------
        current_time : float
        deadline : float

        Returns
        -------
        float
            Non-negative cost value.
        """
        remaining = deadline - current_time
        lateness = max(0.0, current_time - deadline)
        # Anticipatory pressure rises exponentially as deadline approaches
        anticipatory = self._gamma * math.exp(-self._delta * max(remaining, 0.0))
        return self._lambda * lateness + anticipatory

    def modified_reward(
        self,
        base_reward: float,
        current_time: float,
        deadline: float,
        beta: float = 1.0,
    ) -> float:
        """Compute modified reward incorporating deadline pressure.

        Parameters
        ----------
        base_reward : float
            Original task reward.
        current_time : float
        deadline : float
        beta : float
            Rationality parameter.

        Returns
        -------
        float
            Modified reward value.
        """
        pressure = self.deadline_cost(current_time, deadline)
        # Time discount: future rewards are worth less under bounded rationality
        remaining = max(deadline - current_time, 0.01)
        time_discount = math.exp(-remaining / max(beta, 0.01))
        return base_reward * (1.0 - time_discount) - pressure

    def pressure_profile(
        self,
        deadline: float,
        time_points: np.ndarray,
    ) -> np.ndarray:
        """Compute deadline pressure over a time series.

        Parameters
        ----------
        deadline : float
        time_points : np.ndarray
            Shape ``(m,)`` time points.

        Returns
        -------
        np.ndarray
            Shape ``(m,)`` pressure values.
        """
        t = np.asarray(time_points, dtype=np.float64)
        remaining = deadline - t
        lateness = np.maximum(0.0, t - deadline)
        anticipatory = self._gamma * np.exp(
            -self._delta * np.maximum(remaining, 0.0)
        )
        return self._lambda * lateness + anticipatory


__all__ = [
    "DeadlineAnalyzer",
    "DeadlineDetector",
    "DeadlineInfo",
    "DeadlineType",
    "FreeEnergyDeadlinePressure",
    "compute_urgency",
    "temporal_demand_function",
    "urgency_batch",
]

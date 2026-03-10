"""
usability_oracle.algebra.context — Context modulation operator Δ.

Applies cognitive-context adjustments to a :class:`CostElement`,
modelling how factors such as fatigue, working-memory load, practice,
stress, and age affect the cognitive cost of a UI task step.

Each modulation factor is backed by well-established results from
cognitive psychology:

* **Fatigue**: Weber-Fechner logarithmic degradation [Hockey 1997]
* **Working-memory load**: Cowan (2001) capacity limit, load × cost scaling
* **Practice**: Power Law of Practice [Newell & Rosenbloom 1981]:
  ``T_n = T_1 · n^{-α}``
* **Stress**: Yerkes-Dodson inverted-U arousal model
* **Age**: generalised slowing factor [Salthouse 1996]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from usability_oracle.algebra.models import CostElement


# ---------------------------------------------------------------------------
# CognitiveContext — bundles all modulation parameters
# ---------------------------------------------------------------------------


@dataclass
class CognitiveContext:
    """Parameters describing the user's current cognitive state.

    All parameters are optional; unset values (``None``) are treated as
    *no modulation* for that factor.

    Parameters
    ----------
    elapsed_time : float | None
        Wall-clock time (seconds) since the beginning of the session.
        Used for fatigue modulation.
    working_memory_load : float | None
        Number of items currently held in working memory (0–7 scale
        based on Cowan's *k* estimate).
    repetitions : int | None
        Number of prior repetitions of this task (for practice effects).
    stress_level : float | None
        Normalised stress / arousal level ∈ [0, 1].
        0 = under-aroused, 0.5 = optimal, 1 = over-aroused.
    age_percentile : float | None
        User's age expressed as a processing-speed percentile ∈ [0, 1].
        0.5 = median adult, 0.0 = slowest, 1.0 = fastest.
    custom : dict
        Additional user-defined modulation parameters.
    """

    elapsed_time: Optional[float] = None
    working_memory_load: Optional[float] = None
    repetitions: Optional[int] = None
    stress_level: Optional[float] = None
    age_percentile: Optional[float] = None
    custom: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.working_memory_load is not None:
            self.working_memory_load = max(0.0, float(self.working_memory_load))
        if self.stress_level is not None:
            self.stress_level = max(0.0, min(1.0, float(self.stress_level)))
        if self.age_percentile is not None:
            self.age_percentile = max(0.0, min(1.0, float(self.age_percentile)))
        if self.repetitions is not None:
            self.repetitions = max(0, int(self.repetitions))


# ---------------------------------------------------------------------------
# ContextModulator
# ---------------------------------------------------------------------------


class ContextModulator:
    r"""Apply context-dependent modulation (Δ operator) to cost elements.

    The Δ operator transforms a cost element in-place according to the
    user's cognitive state:

    .. math::

        Δ(c, \text{ctx}) = f_{\text{fatigue}} ∘ f_{\text{WM}} ∘
            f_{\text{practice}} ∘ f_{\text{stress}} ∘ f_{\text{age}}(c)

    Each modulation function adjusts μ, σ², κ, λ according to empirical
    scaling laws.

    Usage::

        mod = ContextModulator()
        ctx = CognitiveContext(elapsed_time=1800, working_memory_load=4)
        adjusted = mod.modulate(element, ctx)
    """

    # -- default model parameters --------------------------------------------

    # Fatigue: scale factor per log-hour
    FATIGUE_RATE: float = 0.05
    # Maximum fatigue multiplier
    FATIGUE_CAP: float = 2.0

    # Working memory: Cowan's k (typical capacity)
    WM_CAPACITY: float = 4.0
    # WM overload scaling exponent
    WM_OVERLOAD_EXPONENT: float = 1.5

    # Practice: power-law exponent (typically 0.2–0.6)
    PRACTICE_EXPONENT: float = 0.4
    # Minimum fraction of original cost after practice
    PRACTICE_FLOOR: float = 0.3

    # Stress: Yerkes-Dodson parameters
    # Optimal arousal level (minimum cost)
    STRESS_OPTIMAL: float = 0.5
    # Width of the optimal zone
    STRESS_SENSITIVITY: float = 4.0

    # Age: generalised slowing factor range
    AGE_SLOW_FACTOR_MIN: float = 1.0   # at percentile 1.0 (fastest)
    AGE_SLOW_FACTOR_MAX: float = 2.0   # at percentile 0.0 (slowest)

    # -- main entry point ----------------------------------------------------

    def modulate(self, element: CostElement, context: CognitiveContext) -> CostElement:
        """Apply all applicable context modulations.

        Parameters
        ----------
        element : CostElement
            The base cost element to modulate.
        context : CognitiveContext
            The cognitive state parameters.

        Returns
        -------
        CostElement
            The modulated cost element.
        """
        result = CostElement(
            mu=element.mu,
            sigma_sq=element.sigma_sq,
            kappa=element.kappa,
            lambda_=element.lambda_,
        )

        if context.elapsed_time is not None:
            result = self._fatigue_modulation(result, context.elapsed_time)

        if context.working_memory_load is not None:
            result = self._memory_load_modulation(result, context.working_memory_load)

        if context.repetitions is not None:
            result = self._practice_modulation(result, context.repetitions)

        if context.stress_level is not None:
            result = self._stress_modulation(result, context.stress_level)

        if context.age_percentile is not None:
            result = self._age_modulation(result, context.age_percentile)

        return result

    # -- individual modulation functions -------------------------------------

    def _fatigue_modulation(
        self, element: CostElement, elapsed_time: float
    ) -> CostElement:
        r"""Apply fatigue modulation based on time-on-task.

        Model (Weber-Fechner degradation):

        .. math::

            f_{\text{fatigue}}(μ) = μ · \min\bigl(
                1 + r · \ln(1 + t/3600),\; f_{\max}
            \bigr)

        where *r* is ``FATIGUE_RATE`` and *t* is elapsed time in seconds.

        Fatigue increases both the mean and variance of costs, and raises
        tail risk.

        Parameters
        ----------
        element : CostElement
        elapsed_time : float
            Seconds since session start.

        Returns
        -------
        CostElement
        """
        if elapsed_time <= 0:
            return element

        hours = elapsed_time / 3600.0
        # Logarithmic fatigue accumulation
        fatigue_multiplier = min(
            1.0 + self.FATIGUE_RATE * math.log1p(hours),
            self.FATIGUE_CAP,
        )

        return CostElement(
            mu=element.mu * fatigue_multiplier,
            sigma_sq=element.sigma_sq * fatigue_multiplier ** 2,
            kappa=element.kappa + 0.1 * (fatigue_multiplier - 1.0),
            lambda_=min(
                element.lambda_ + 0.05 * (fatigue_multiplier - 1.0),
                1.0,
            ),
        )

    def _memory_load_modulation(
        self, element: CostElement, working_memory_load: float
    ) -> CostElement:
        r"""Apply working-memory load modulation.

        Model (Cowan 2001):

        .. math::

            f_{\text{WM}}(μ) = μ · \begin{cases}
                1 & \text{if } k ≤ K \\
                \bigl(\frac{k}{K}\bigr)^{α} & \text{if } k > K
            \end{cases}

        where *K* is ``WM_CAPACITY`` (≈ 4) and *α* is
        ``WM_OVERLOAD_EXPONENT``.

        Exceeding WM capacity causes super-linear cost growth due to
        the need for rehearsal, chunking failures, and error recovery.

        Parameters
        ----------
        element : CostElement
        working_memory_load : float
            Current WM load (number of items).

        Returns
        -------
        CostElement
        """
        if working_memory_load <= self.WM_CAPACITY:
            # Within capacity: minimal effect
            return element

        # Overload: super-linear scaling
        overload_ratio = working_memory_load / self.WM_CAPACITY
        scale = overload_ratio ** self.WM_OVERLOAD_EXPONENT

        # Variance increases more steeply than mean (more unpredictable)
        var_scale = scale ** 1.5

        return CostElement(
            mu=element.mu * scale,
            sigma_sq=element.sigma_sq * var_scale,
            kappa=element.kappa + 0.2 * (overload_ratio - 1.0),
            lambda_=min(
                element.lambda_ + 0.1 * (overload_ratio - 1.0),
                1.0,
            ),
        )

    def _practice_modulation(
        self, element: CostElement, repetitions: int
    ) -> CostElement:
        r"""Apply practice (learning) modulation.

        Power Law of Practice [Newell & Rosenbloom 1981]:

        .. math::

            f_{\text{practice}}(μ) = μ · \max\bigl(
                n^{-α},\; f_{\min}
            \bigr)

        where *n* is the trial number (``repetitions + 1``) and *α* is
        ``PRACTICE_EXPONENT``.

        Practice reduces both mean and variance, and decreases tail risk.

        Parameters
        ----------
        element : CostElement
        repetitions : int
            Number of prior repetitions (0 = first attempt).

        Returns
        -------
        CostElement
        """
        if repetitions <= 0:
            return element

        n = repetitions + 1  # trial number (1-indexed)
        # Power law: T_n = T_1 * n^(-alpha)
        learning_factor = max(
            n ** (-self.PRACTICE_EXPONENT),
            self.PRACTICE_FLOOR,
        )

        return CostElement(
            mu=element.mu * learning_factor,
            sigma_sq=element.sigma_sq * learning_factor ** 2,
            kappa=element.kappa * learning_factor,
            lambda_=element.lambda_ * learning_factor,
        )

    def _stress_modulation(
        self, element: CostElement, stress_level: float
    ) -> CostElement:
        r"""Apply stress / arousal modulation.

        Yerkes-Dodson inverted-U model:

        .. math::

            f_{\text{stress}}(μ) = μ · \bigl(
                1 + β · (s - s^*)^2
            \bigr)

        where *s* is the stress level, *s** is the optimal arousal
        (``STRESS_OPTIMAL``), and *β* is ``STRESS_SENSITIVITY``.

        Both under-arousal (drowsiness) and over-arousal (anxiety) increase
        cognitive cost; the minimum is at moderate arousal.

        Parameters
        ----------
        element : CostElement
        stress_level : float
            Normalised stress level ∈ [0, 1].

        Returns
        -------
        CostElement
        """
        deviation = stress_level - self.STRESS_OPTIMAL
        stress_penalty = 1.0 + self.STRESS_SENSITIVITY * deviation * deviation

        # High stress especially increases tail risk and skewness
        tail_adjustment = 0.0
        if stress_level > 0.7:
            tail_adjustment = 0.15 * (stress_level - 0.7) / 0.3

        return CostElement(
            mu=element.mu * stress_penalty,
            sigma_sq=element.sigma_sq * stress_penalty ** 2,
            kappa=element.kappa + 0.3 * abs(deviation),
            lambda_=min(element.lambda_ + tail_adjustment, 1.0),
        )

    def _age_modulation(
        self, element: CostElement, age_percentile: float
    ) -> CostElement:
        r"""Apply age-related modulation.

        Generalised slowing model [Salthouse 1996]:

        .. math::

            f_{\text{age}}(μ) = μ · s(p)

        where ``s(p)`` linearly interpolates between ``AGE_SLOW_FACTOR_MAX``
        (at percentile 0) and ``AGE_SLOW_FACTOR_MIN`` (at percentile 1).

        A percentile of 0.5 represents a median-speed adult.  Lower
        percentiles correspond to older adults or users with slower
        processing speed.

        Parameters
        ----------
        element : CostElement
        age_percentile : float
            Processing-speed percentile ∈ [0, 1].

        Returns
        -------
        CostElement
        """
        # Linear interpolation: percentile 1.0 → factor 1.0, percentile 0.0 → factor 2.0
        slow_factor = (
            self.AGE_SLOW_FACTOR_MAX
            + (self.AGE_SLOW_FACTOR_MIN - self.AGE_SLOW_FACTOR_MAX) * age_percentile
        )

        return CostElement(
            mu=element.mu * slow_factor,
            sigma_sq=element.sigma_sq * slow_factor ** 2,
            kappa=element.kappa,
            lambda_=min(
                element.lambda_ * slow_factor,
                1.0,
            ),
        )

    # -- combined modulation factor ------------------------------------------

    def total_multiplier(self, context: CognitiveContext) -> float:
        """Compute the approximate total multiplicative factor on μ.

        Useful for quick estimation without full composition.

        Parameters
        ----------
        context : CognitiveContext

        Returns
        -------
        float
            Approximate multiplier ``M`` such that ``μ_adjusted ≈ M · μ_base``.
        """
        unit = CostElement(mu=1.0, sigma_sq=0.0, kappa=0.0, lambda_=0.0)
        result = self.modulate(unit, context)
        return result.mu

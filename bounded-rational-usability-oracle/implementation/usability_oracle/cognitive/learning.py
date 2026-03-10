"""Cognitive learning models for usability prediction.

Implements empirically-grounded models of skill acquisition and
forgetting, including the power law of practice, log-linear learning
curves, Fitts and Posner's three-stage model, transfer of learning,
and the Ebbinghaus forgetting curve.

References
----------
Newell, A. & Rosenbloom, P. S. (1981). Mechanisms of skill acquisition
    and the law of practice. In J. R. Anderson (Ed.), *Cognitive Skills
    and Their Acquisition* (pp. 1-55). Lawrence Erlbaum.
Anderson, J. R. (1982). Acquisition of cognitive skill. *Psychological
    Review*, 89(4), 369-406.
Fitts, P. M. & Posner, M. I. (1967). *Human Performance*. Brooks/Cole.
Ebbinghaus, H. (1885). *Über das Gedächtnis*. Duncker & Humblot.
Singley, M. K. & Anderson, J. R. (1989). *The Transfer of Cognitive
    Skill*. Harvard University Press.
Heathcote, A., Brown, S., & Mewhort, D. J. K. (2000). The power law
    repealed: The case for an exponential law of practice.
    *Psychonomic Bulletin & Review*, 7(2), 185-207.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Skill stage model (Fitts & Posner, 1967)
# ---------------------------------------------------------------------------


class SkillStage(Enum):
    """Fitts & Posner (1967) three-stage model of skill acquisition.

    Cognitive → Associative → Autonomous
    """

    COGNITIVE = "cognitive"
    ASSOCIATIVE = "associative"
    AUTONOMOUS = "autonomous"


@dataclass
class SkillProfile:
    """Parameter profile for a given skill level.

    Attributes
    ----------
    stage : SkillStage
        Current learning stage.
    practice_trials : int
        Number of practice trials completed.
    error_rate : float
        Current error probability.
    speed_multiplier : float
        Multiplier on base task time (< 1 means faster than baseline).
    cognitive_load : float
        Working-memory load fraction in [0, 1].
    variability : float
        Response-time coefficient of variation.
    """

    stage: SkillStage = SkillStage.COGNITIVE
    practice_trials: int = 0
    error_rate: float = 0.30
    speed_multiplier: float = 1.0
    cognitive_load: float = 1.0
    variability: float = 0.30


# ---------------------------------------------------------------------------
# Predefined user profiles
# ---------------------------------------------------------------------------


NOVICE_PROFILE = SkillProfile(
    stage=SkillStage.COGNITIVE,
    practice_trials=0,
    error_rate=0.30,
    speed_multiplier=3.0,
    cognitive_load=1.0,
    variability=0.40,
)

INTERMEDIATE_PROFILE = SkillProfile(
    stage=SkillStage.ASSOCIATIVE,
    practice_trials=50,
    error_rate=0.10,
    speed_multiplier=1.5,
    cognitive_load=0.50,
    variability=0.20,
)

EXPERT_PROFILE = SkillProfile(
    stage=SkillStage.AUTONOMOUS,
    practice_trials=500,
    error_rate=0.02,
    speed_multiplier=0.7,
    cognitive_load=0.10,
    variability=0.08,
)


# ---------------------------------------------------------------------------
# Learning curve models
# ---------------------------------------------------------------------------


class LearningModel:
    """Cognitive learning models for task performance prediction.

    Provides power-law and log-linear learning curves, skill stage
    classification, transfer estimation, and forgetting functions.
    """

    # ------------------------------------------------------------------ #
    # Power law of practice (Newell & Rosenbloom, 1981)
    # ------------------------------------------------------------------ #

    @staticmethod
    def power_law(
        n_trials: NDArray[np.integer] | Sequence[int],
        t1: float = 2.0,
        alpha: float = 0.4,
    ) -> NDArray[np.floating]:
        """Predict performance time across practice trials.

        .. math::

            T(n) = T_1 \\cdot n^{-\\alpha}

        Parameters
        ----------
        n_trials : array-like of int
            Trial numbers (>= 1).
        t1 : float
            Time on the first trial (seconds).
        alpha : float
            Learning-rate exponent (typically 0.2–0.6).

        Returns
        -------
        numpy.ndarray
            Predicted times for each trial.
        """
        n = np.asarray(n_trials, dtype=np.float64)
        n = np.maximum(n, 1.0)
        return t1 * np.power(n, -alpha)

    @staticmethod
    def power_law_scalar(
        n: int,
        t1: float = 2.0,
        alpha: float = 0.4,
    ) -> float:
        """Scalar version of the power law.

        Parameters
        ----------
        n : int
            Trial number (>= 1).
        t1 : float
            First-trial time.
        alpha : float
            Learning rate.

        Returns
        -------
        float
            Predicted time.
        """
        return t1 * max(1, n) ** (-alpha)

    # ------------------------------------------------------------------ #
    # Log-linear learning curve
    # ------------------------------------------------------------------ #

    @staticmethod
    def log_linear(
        n_trials: NDArray[np.integer] | Sequence[int],
        t1: float = 2.0,
        beta: float = 0.3,
    ) -> NDArray[np.floating]:
        """Log-linear (exponential) learning curve.

        .. math::

            T(n) = T_1 \\cdot e^{-\\beta (n - 1)}

        An alternative to the power law advocated by Heathcote et al.
        (2000) that can better fit individual-participant data.

        Parameters
        ----------
        n_trials : array-like of int
            Trial numbers (>= 1).
        t1 : float
            First-trial time (seconds).
        beta : float
            Exponential learning rate.

        Returns
        -------
        numpy.ndarray
            Predicted times.
        """
        n = np.asarray(n_trials, dtype=np.float64)
        n = np.maximum(n, 1.0)
        return t1 * np.exp(-beta * (n - 1.0))

    # ------------------------------------------------------------------ #
    # Skill acquisition stages
    # ------------------------------------------------------------------ #

    @staticmethod
    def classify_stage(
        practice_trials: int,
        cognitive_threshold: int = 10,
        autonomous_threshold: int = 200,
    ) -> SkillStage:
        """Classify the current skill stage from practice count.

        Uses Fitts & Posner (1967) three-stage framework with
        configurable thresholds.

        Parameters
        ----------
        practice_trials : int
            Number of practice trials completed.
        cognitive_threshold : int
            Trials below this → cognitive stage.
        autonomous_threshold : int
            Trials above this → autonomous stage.

        Returns
        -------
        SkillStage
        """
        if practice_trials < cognitive_threshold:
            return SkillStage.COGNITIVE
        elif practice_trials >= autonomous_threshold:
            return SkillStage.AUTONOMOUS
        else:
            return SkillStage.ASSOCIATIVE

    @staticmethod
    def stage_parameters(stage: SkillStage) -> Dict[str, float]:
        """Return typical performance parameters for a skill stage.

        Based on Anderson (1982) ACT* theory stage characteristics.

        Parameters
        ----------
        stage : SkillStage

        Returns
        -------
        dict[str, float]
            ``"speed_multiplier"``, ``"error_rate"``,
            ``"cognitive_load"``, ``"variability"``.
        """
        params = {
            SkillStage.COGNITIVE: {
                "speed_multiplier": 3.0,
                "error_rate": 0.30,
                "cognitive_load": 1.0,
                "variability": 0.40,
            },
            SkillStage.ASSOCIATIVE: {
                "speed_multiplier": 1.5,
                "error_rate": 0.10,
                "cognitive_load": 0.50,
                "variability": 0.20,
            },
            SkillStage.AUTONOMOUS: {
                "speed_multiplier": 0.7,
                "error_rate": 0.02,
                "cognitive_load": 0.10,
                "variability": 0.08,
            },
        }
        return params[stage]

    @staticmethod
    def build_skill_profile(practice_trials: int) -> SkillProfile:
        """Build a complete skill profile from practice count.

        Parameters
        ----------
        practice_trials : int
            Number of completed practice trials.

        Returns
        -------
        SkillProfile
        """
        stage = LearningModel.classify_stage(practice_trials)
        params = LearningModel.stage_parameters(stage)
        return SkillProfile(
            stage=stage,
            practice_trials=practice_trials,
            error_rate=params["error_rate"],
            speed_multiplier=params["speed_multiplier"],
            cognitive_load=params["cognitive_load"],
            variability=params["variability"],
        )

    # ------------------------------------------------------------------ #
    # Transfer of learning
    # ------------------------------------------------------------------ #

    @staticmethod
    def transfer_ratio(
        shared_elements: int,
        source_total: int,
        target_total: int,
    ) -> float:
        """Estimate transfer of learning between tasks.

        Uses the identical-elements theory (Singley & Anderson, 1989):
        transfer is proportional to the fraction of shared production
        rules / cognitive elements.

        .. math::

            \\text{transfer} = \\frac{|S \\cap T|}{|T|}

        Parameters
        ----------
        shared_elements : int
            Number of cognitive elements shared between tasks.
        source_total : int
            Total elements in the source task.
        target_total : int
            Total elements in the target task.

        Returns
        -------
        float
            Transfer ratio in [0, 1].
        """
        if target_total <= 0:
            return 0.0
        ratio = min(shared_elements, target_total) / target_total
        return float(np.clip(ratio, 0.0, 1.0))

    @staticmethod
    def transferred_trials(
        source_trials: int,
        transfer: float,
    ) -> float:
        """Compute effective practice trials on new task via transfer.

        Parameters
        ----------
        source_trials : int
            Practice trials on the source task.
        transfer : float
            Transfer ratio in [0, 1].

        Returns
        -------
        float
            Effective trials on the target task.
        """
        return max(0.0, source_trials * transfer)

    @staticmethod
    def predict_with_transfer(
        source_trials: int,
        target_trials: int,
        transfer: float,
        t1: float = 2.0,
        alpha: float = 0.4,
    ) -> float:
        """Predict task time accounting for transfer.

        Parameters
        ----------
        source_trials : int
            Practice on source task.
        target_trials : int
            Practice on target task.
        transfer : float
            Transfer ratio.
        t1 : float
            First-trial time.
        alpha : float
            Learning rate.

        Returns
        -------
        float
            Predicted time on target task.
        """
        effective = target_trials + LearningModel.transferred_trials(
            source_trials, transfer
        )
        return LearningModel.power_law_scalar(max(1, int(effective)), t1, alpha)

    # ------------------------------------------------------------------ #
    # Learning rate estimation
    # ------------------------------------------------------------------ #

    @staticmethod
    def estimate_learning_rate(
        task_complexity: float,
        base_rate: float = 0.4,
    ) -> float:
        """Estimate learning rate from task complexity.

        More complex tasks have slower learning rates (smaller α).

        .. math::

            \\alpha = \\frac{\\alpha_{\\text{base}}}{1 + \\log_2(C)}

        Parameters
        ----------
        task_complexity : float
            Complexity measure (>= 1.0).  Number of steps or
            information-theoretic bits.
        base_rate : float
            Learning rate for a simple (complexity=1) task.

        Returns
        -------
        float
            Estimated learning rate.
        """
        complexity = max(1.0, task_complexity)
        return base_rate / (1.0 + math.log2(complexity))

    # ------------------------------------------------------------------ #
    # Ebbinghaus forgetting curve
    # ------------------------------------------------------------------ #

    @staticmethod
    def ebbinghaus_retention(
        delays: NDArray[np.floating] | Sequence[float],
        strength: float = 1.0,
        stability: float = 1.0,
    ) -> NDArray[np.floating]:
        """Ebbinghaus forgetting curve.

        .. math::

            R(t) = e^{-t / (S \\cdot \\text{stability})}

        where *S* is the memory strength and *stability* determines
        how resistant the memory is to decay.

        Parameters
        ----------
        delays : array-like of float
            Retention intervals in seconds.
        strength : float
            Memory strength parameter (>= 0).
        stability : float
            Memory stability (>= 0.01).

        Returns
        -------
        numpy.ndarray
            Retention probabilities in [0, 1].
        """
        t = np.asarray(delays, dtype=np.float64)
        t = np.maximum(t, 0.0)
        stability = max(stability, 0.01)
        return np.exp(-t / (strength * stability))

    @staticmethod
    def ebbinghaus_scalar(
        delay: float,
        strength: float = 1.0,
        stability: float = 1.0,
    ) -> float:
        """Scalar Ebbinghaus forgetting curve.

        Parameters
        ----------
        delay : float
            Retention interval (seconds).
        strength : float
            Memory strength.
        stability : float
            Memory stability.

        Returns
        -------
        float
            Retention probability.
        """
        delay = max(delay, 0.0)
        stability = max(stability, 0.01)
        return math.exp(-delay / (strength * stability))

    @staticmethod
    def spaced_repetition_strength(
        n_reviews: int,
        base_strength: float = 1.0,
        gain_per_review: float = 0.5,
    ) -> float:
        """Memory strength after spaced repetition reviews.

        Each review multiplicatively increases strength:

        .. math::

            S(n) = S_0 \\cdot (1 + g)^n

        Parameters
        ----------
        n_reviews : int
            Number of spaced repetition reviews.
        base_strength : float
            Initial memory strength.
        gain_per_review : float
            Proportional strength gain per review.

        Returns
        -------
        float
            Updated memory strength.
        """
        n = max(0, n_reviews)
        return base_strength * (1.0 + gain_per_review) ** n

    # ------------------------------------------------------------------ #
    # Composite: learning + forgetting
    # ------------------------------------------------------------------ #

    @staticmethod
    def performance_over_time(
        practice_trials: int,
        delay_since_practice: float,
        t1: float = 2.0,
        alpha: float = 0.4,
        stability: float = 10.0,
    ) -> float:
        """Predict performance accounting for both learning and forgetting.

        Combines the power law of practice with Ebbinghaus forgetting:

        .. math::

            T(n, t) = T_1 \\cdot n^{-\\alpha} \\cdot
            \\left(1 + \\frac{1 - R(t)}{R(t)}\\right)

        Parameters
        ----------
        practice_trials : int
            Number of practice trials.
        delay_since_practice : float
            Time since last practice (seconds).
        t1 : float
            First-trial time.
        alpha : float
            Learning rate.
        stability : float
            Memory stability for forgetting.

        Returns
        -------
        float
            Predicted task time (seconds).
        """
        learned_time = LearningModel.power_law_scalar(
            practice_trials, t1, alpha
        )
        retention = LearningModel.ebbinghaus_scalar(
            delay_since_practice, strength=1.0, stability=stability
        )
        retention = max(retention, 0.01)
        # Forgetting inflates time: less retention → slower performance
        forgetting_factor = 1.0 / retention
        return learned_time * forgetting_factor

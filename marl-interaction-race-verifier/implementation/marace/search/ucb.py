"""
UCB1 variants with abstract safety margins for MARACE.

Implements several Upper Confidence Bound selection policies tailored to
multi-agent race verification.  The standard UCB1 formula is extended with
abstract-interpretation–derived safety margins (via zonotope support
functions), progressive widening for large branching factors, and a
multi-objective variant that balances race detection, coverage, and
timing diversity.

Each policy computes a *score* used by the MCTS search to select which
child node to expand next.  All variants share a configurable
:class:`ExplorationBonus` that encapsulates the exploration term.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from marace.abstract.zonotope import Zonotope

logger = logging.getLogger(__name__)

__all__ = [
    "ExplorationBonus",
    "AbstractMarginEstimate",
    "UCB1Standard",
    "UCB1Safety",
    "UCB1Progressive",
    "MultiObjectiveUCB",
]


# ======================================================================
# Exploration bonus
# ======================================================================

@dataclass
class ExplorationBonus:
    """Configurable exploration bonus computation.

    The bonus follows the UCB1 formula:

        bonus = C * sqrt(ln(N) / n_i)

    where *C* is the exploration constant, *N* the parent visit count,
    and *n_i* the child visit count.  An optional exponential decay
    can shrink the constant over time, and a minimum bonus floor is
    enforced.

    Parameters
    ----------
    exploration_constant : float
        Base exploration constant (default √2, the theoretically optimal
        value for rewards in [0, 1]).
    decay_factor : float
        If positive, the effective constant is
        ``exploration_constant * exp(-decay_factor * parent_visits)``.
        Set to 0.0 (default) to disable decay.
    min_bonus : float
        Minimum bonus value returned regardless of the formula.
    """

    exploration_constant: float = 1.414
    decay_factor: float = 0.0
    min_bonus: float = 0.0

    def compute(self, parent_visits: int, child_visits: int) -> float:
        """Return the exploration bonus for a child node.

        Parameters
        ----------
        parent_visits : int
            Total number of visits to the parent node (*N*).
        child_visits : int
            Number of visits to the child node (*n_i*).

        Returns
        -------
        float
            Non-negative exploration bonus.
        """
        if child_visits <= 0:
            return float("inf")
        if parent_visits <= 0:
            return self.min_bonus

        c = self.exploration_constant
        if self.decay_factor > 0.0:
            c *= math.exp(-self.decay_factor * parent_visits)

        bonus = c * math.sqrt(math.log(parent_visits) / child_visits)
        return max(bonus, self.min_bonus)


# ======================================================================
# Abstract safety-margin estimation
# ======================================================================

@dataclass
class AbstractMarginEstimate:
    """Compute a safety margin from zonotope analysis.

    Given a half-space safety specification of the form

        safety_direction^T x ≤ safety_threshold

    the *margin* is the distance from the zonotope to the unsafe
    half-space boundary, estimated via the zonotope support function.

    A positive margin means the zonotope is entirely inside the safe
    region; a negative margin indicates a potential violation.

    Parameters
    ----------
    safety_direction : np.ndarray
        Normal vector pointing *towards* the unsafe region (shape ``(n,)``).
    safety_threshold : float
        Scalar threshold defining the safety boundary.
    """

    safety_direction: np.ndarray
    safety_threshold: float

    def __post_init__(self) -> None:
        self.safety_direction = np.asarray(
            self.safety_direction, dtype=np.float64
        ).ravel()
        norm = np.linalg.norm(self.safety_direction)
        if norm == 0.0:
            raise ValueError("safety_direction must be non-zero")
        # Normalise so that margin values are in consistent units.
        self.safety_direction = self.safety_direction / norm
        self.safety_threshold = float(self.safety_threshold) / norm

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def estimate_margin(self, zonotope: Zonotope) -> float:
        """Estimate the safety margin using the zonotope support function.

        Returns ``threshold − h_Z(d)`` where *h_Z(d)* is the support
        function evaluated in the safety direction.  A positive value
        means the zonotope is entirely within the safe region.

        Parameters
        ----------
        zonotope : Zonotope
            Abstract state to evaluate.

        Returns
        -------
        float
            Safety margin (positive ⇒ safe, negative ⇒ potential violation).
        """
        h = zonotope.support_function(self.safety_direction)
        margin = self.safety_threshold - h
        logger.debug(
            "margin=%.4f  (support=%.4f, threshold=%.4f)",
            margin, h, self.safety_threshold,
        )
        return float(margin)

    def margin_confidence(
        self, zonotope: Zonotope
    ) -> Tuple[float, float]:
        """Compute lower and upper bounds on the safety margin.

        The *upper* bound uses the support function in the positive
        direction (worst-case penetration); the *lower* bound uses the
        support function in the negative direction (best-case clearance).

        Parameters
        ----------
        zonotope : Zonotope
            Abstract state to evaluate.

        Returns
        -------
        Tuple[float, float]
            ``(lower_margin, upper_margin)`` pair.
        """
        h_max = zonotope.support_function(self.safety_direction)
        h_min_val, _ = zonotope.minimize(self.safety_direction)

        upper_margin = self.safety_threshold - h_min_val
        lower_margin = self.safety_threshold - h_max
        return float(lower_margin), float(upper_margin)


# ======================================================================
# UCB1 – standard variant
# ======================================================================

@dataclass
class UCB1Standard:
    """Standard UCB1 selection policy.

    score = value + bonus(N, n_i)

    Parameters
    ----------
    bonus : ExplorationBonus
        Exploration bonus configuration.
    """

    bonus: ExplorationBonus = field(default_factory=ExplorationBonus)

    def score(
        self,
        value: float,
        parent_visits: int,
        child_visits: int,
    ) -> float:
        """Compute UCB1 score for a child node.

        Parameters
        ----------
        value : float
            Mean value estimate of the child.
        parent_visits : int
            Visit count of the parent.
        child_visits : int
            Visit count of the child.

        Returns
        -------
        float
            UCB1 score.
        """
        return value + self.bonus.compute(parent_visits, child_visits)


# ======================================================================
# UCB1 – safety-margin variant
# ======================================================================

@dataclass
class UCB1Safety:
    """UCB1 variant augmented with abstract safety-margin estimates.

    When a zonotope representing the reachable states at the child node
    is available, the score blends the standard value estimate with the
    safety margin:

        score = value + safety_weight * margin + bonus

    If no zonotope is provided the policy degrades gracefully to plain
    UCB1.

    Parameters
    ----------
    bonus : ExplorationBonus
        Exploration bonus configuration.
    margin_estimator : AbstractMarginEstimate
        Estimator for safety margins.
    safety_weight : float
        Relative weight of the safety margin in the score.
    """

    bonus: ExplorationBonus = field(default_factory=ExplorationBonus)
    margin_estimator: AbstractMarginEstimate = field(default=None)  # type: ignore[assignment]
    safety_weight: float = 1.0

    def score(
        self,
        value: float,
        parent_visits: int,
        child_visits: int,
        state_zonotope: Optional[Zonotope] = None,
    ) -> float:
        """Compute safety-aware UCB1 score.

        Parameters
        ----------
        value : float
            Mean value estimate of the child.
        parent_visits : int
            Visit count of the parent.
        child_visits : int
            Visit count of the child.
        state_zonotope : Zonotope, optional
            Zonotope over-approximation of the reachable states at the
            child.  When ``None`` the safety term is omitted.

        Returns
        -------
        float
            Safety-aware UCB1 score.
        """
        exploration = self.bonus.compute(parent_visits, child_visits)

        safety_term = 0.0
        if state_zonotope is not None and self.margin_estimator is not None:
            margin = self.margin_estimator.estimate_margin(state_zonotope)
            safety_term = self.safety_weight * margin
            logger.debug(
                "UCB1Safety: value=%.4f  margin=%.4f  safety_term=%.4f  "
                "exploration=%.4f",
                value, margin, safety_term, exploration,
            )

        return value + safety_term + exploration


# ======================================================================
# UCB1 – progressive widening
# ======================================================================

@dataclass
class UCB1Progressive:
    """UCB1 with progressive widening for large branching factors.

    Progressive widening limits the number of children expanded at a
    node: a new child is created only when

        num_children < widening_constant * visit_count^widening_exponent

    This prevents the search from spreading too thinly in high-branching
    scheduling spaces typical in multi-agent race verification.

    Parameters
    ----------
    bonus : ExplorationBonus
        Exploration bonus configuration.
    widening_constant : float
        Multiplicative constant in the widening criterion.
    widening_exponent : float
        Exponent in the widening criterion (typically in (0, 1)).
    """

    bonus: ExplorationBonus = field(default_factory=ExplorationBonus)
    widening_constant: float = 1.0
    widening_exponent: float = 0.5

    def should_widen(self, num_children: int, visit_count: int) -> bool:
        """Decide whether to expand a new child.

        Parameters
        ----------
        num_children : int
            Current number of children at the node.
        visit_count : int
            Total visit count at the node.

        Returns
        -------
        bool
            ``True`` if a new child should be created.
        """
        if visit_count <= 0:
            return True
        threshold = self.widening_constant * (visit_count ** self.widening_exponent)
        return num_children < threshold

    def score(
        self,
        value: float,
        parent_visits: int,
        child_visits: int,
    ) -> float:
        """Compute UCB1 score (same formula as standard UCB1).

        Progressive widening only affects *whether* new children are
        created; score computation among existing children is unchanged.

        Parameters
        ----------
        value : float
            Mean value estimate of the child.
        parent_visits : int
            Visit count of the parent.
        child_visits : int
            Visit count of the child.

        Returns
        -------
        float
            UCB1 score.
        """
        return value + self.bonus.compute(parent_visits, child_visits)


# ======================================================================
# Multi-objective UCB
# ======================================================================

@dataclass
class MultiObjectiveUCB:
    """Multi-objective UCB balancing race detection, coverage, and diversity.

    The score is a weighted combination of three objective values plus
    an exploration bonus:

        score = w_r * race_value
              + w_c * coverage_value
              + w_d * diversity_value
              + bonus(N, n_i)

    Parameters
    ----------
    race_weight : float
        Weight for the race-detection objective.
    coverage_weight : float
        Weight for the state-space coverage objective.
    diversity_weight : float
        Weight for the timing-diversity objective.
    bonus : ExplorationBonus
        Exploration bonus configuration.
    """

    race_weight: float = 1.0
    coverage_weight: float = 1.0
    diversity_weight: float = 1.0
    bonus: ExplorationBonus = field(default_factory=ExplorationBonus)

    def score(
        self,
        race_value: float,
        coverage_value: float,
        diversity_value: float,
        parent_visits: int,
        child_visits: int,
    ) -> float:
        """Compute the multi-objective UCB score.

        Parameters
        ----------
        race_value : float
            Estimated value w.r.t. race detection.
        coverage_value : float
            Estimated value w.r.t. state-space coverage.
        diversity_value : float
            Estimated value w.r.t. timing diversity.
        parent_visits : int
            Visit count of the parent node.
        child_visits : int
            Visit count of the child node.

        Returns
        -------
        float
            Multi-objective UCB score.
        """
        weighted = (
            self.race_weight * race_value
            + self.coverage_weight * coverage_value
            + self.diversity_weight * diversity_value
        )
        exploration = self.bonus.compute(parent_visits, child_visits)
        return weighted + exploration

    @staticmethod
    def compute_diversity(
        actions: List[Any],
        existing_actions: List[Any],
    ) -> float:
        """Compute timing diversity of *actions* w.r.t. *existing_actions*.

        Diversity is measured as the mean pairwise Euclidean distance
        between ``actions`` and ``existing_actions`` when they are
        numeric (convertible to numpy arrays).  If the lists are empty
        or non-numeric, the diversity defaults to 0.0.

        Parameters
        ----------
        actions : list
            Candidate actions to evaluate.
        existing_actions : list
            Actions already selected / explored.

        Returns
        -------
        float
            Non-negative diversity score.
        """
        if not actions or not existing_actions:
            return 0.0

        try:
            a = np.asarray(actions, dtype=np.float64)
            b = np.asarray(existing_actions, dtype=np.float64)
        except (ValueError, TypeError):
            return 0.0

        if a.ndim == 1:
            a = a.reshape(-1, 1)
        if b.ndim == 1:
            b = b.reshape(-1, 1)

        if a.shape[1] != b.shape[1]:
            return 0.0

        # Pairwise Euclidean distances between every (a_i, b_j) pair.
        diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]  # (|a|, |b|, d)
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))        # (|a|, |b|)
        return float(np.mean(dists))

"""
Exponential mechanism for private mechanism selection in DP-Forge.

Implements the exponential mechanism for privately selecting among
candidate mechanisms.  Given T candidate mechanisms with error scores,
selects one with probability proportional to ``exp(-ε_select · score / (2Δ))``
where Δ is the score sensitivity.

Key Components:
    - ``ExponentialSelector``: Core exponential mechanism with score
      sensitivity computation and utility guarantees.
    - ``CandidatePool``: Manages mechanism candidates from different
      grid sizes / configurations.

Mathematical Background:
    The exponential mechanism (McSherry & Talwar, 2007) selects
    outcome r with probability:

        Pr[M(x) = r] ∝ exp(ε · u(x, r) / (2 Δu))

    where u(x, r) is the utility/score function and Δu is the
    sensitivity of u (max change when one record changes).

    Utility guarantee (Theorem 3.11, Dwork & Roth):
        With probability ≥ 1 - β, the selected mechanism has score:
            score(r) ≤ OPT + (2 Δu / ε) · (ln(T) + ln(1/β))
        where T is the number of candidates and OPT is the best score.

Usage::

    from dp_forge.exponential_mechanism import ExponentialSelector, CandidatePool

    pool = CandidatePool()
    pool.add("mech_k50", score=0.05, metadata={"k": 50})
    pool.add("mech_k100", score=0.03, metadata={"k": 100})

    selector = ExponentialSelector(epsilon_select=0.5)
    chosen = selector.select(pool)
    bound = selector.utility_bound(pool, beta=0.05)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

from .exceptions import (
    BudgetExhaustedError,
    ConfigurationError,
    DPForgeError,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Candidate dataclass
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """A candidate mechanism for selection.

    Attributes:
        name: Unique identifier for the candidate.
        score: Error/loss score (lower is better).
        metadata: Arbitrary metadata (grid size, params, etc.).
        mechanism: Optional reference to the mechanism object.
    """

    name: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    mechanism: Any = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.score):
            raise ConfigurationError(
                f"Candidate score must be finite, got {self.score}",
                parameter="score",
                value=self.score,
            )

    def __repr__(self) -> str:
        return f"Candidate({self.name!r}, score={self.score:.6f})"


# ---------------------------------------------------------------------------
# CandidatePool
# ---------------------------------------------------------------------------


class CandidatePool:
    """Pool of candidate mechanisms for exponential mechanism selection.

    Manages a collection of candidate mechanisms with their error
    scores.  Supports adding, removing, and querying candidates.

    Candidates are stored sorted by score (ascending, lower is better).
    """

    def __init__(self) -> None:
        self._candidates: Dict[str, Candidate] = {}

    def add(
        self,
        name: str,
        score: float,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        mechanism: Any = None,
    ) -> Candidate:
        """Add a candidate to the pool.

        Args:
            name: Unique identifier.
            score: Error/loss score (lower is better).
            metadata: Optional metadata dict.
            mechanism: Optional mechanism object.

        Returns:
            The created Candidate.

        Raises:
            ConfigurationError: If name already exists.
        """
        if name in self._candidates:
            raise ConfigurationError(
                f"Candidate {name!r} already exists in pool",
                parameter="name",
                value=name,
            )

        cand = Candidate(
            name=name,
            score=score,
            metadata=metadata or {},
            mechanism=mechanism,
        )
        self._candidates[name] = cand
        return cand

    def remove(self, name: str) -> None:
        """Remove a candidate from the pool.

        Args:
            name: Candidate identifier.

        Raises:
            ConfigurationError: If name not in pool.
        """
        if name not in self._candidates:
            raise ConfigurationError(
                f"Candidate {name!r} not in pool",
                parameter="name",
                value=name,
            )
        del self._candidates[name]

    def get(self, name: str) -> Candidate:
        """Get a candidate by name.

        Args:
            name: Candidate identifier.

        Returns:
            The Candidate.

        Raises:
            ConfigurationError: If name not in pool.
        """
        if name not in self._candidates:
            raise ConfigurationError(
                f"Candidate {name!r} not in pool",
                parameter="name",
            )
        return self._candidates[name]

    @property
    def candidates(self) -> List[Candidate]:
        """All candidates, sorted by score (ascending)."""
        return sorted(self._candidates.values(), key=lambda c: c.score)

    @property
    def size(self) -> int:
        """Number of candidates in the pool."""
        return len(self._candidates)

    @property
    def scores(self) -> npt.NDArray[np.float64]:
        """Array of scores for all candidates (sorted ascending)."""
        return np.array([c.score for c in self.candidates], dtype=np.float64)

    @property
    def best_score(self) -> float:
        """Best (minimum) score in the pool."""
        if not self._candidates:
            return float("inf")
        return min(c.score for c in self._candidates.values())

    @property
    def best_candidate(self) -> Optional[Candidate]:
        """Candidate with the best (lowest) score."""
        if not self._candidates:
            return None
        return min(self._candidates.values(), key=lambda c: c.score)

    @property
    def score_range(self) -> float:
        """Range of scores in the pool (max - min)."""
        if len(self._candidates) < 2:
            return 0.0
        scores = [c.score for c in self._candidates.values()]
        return max(scores) - min(scores)

    def update_score(self, name: str, new_score: float) -> None:
        """Update the score of a candidate.

        Args:
            name: Candidate identifier.
            new_score: New score value.
        """
        cand = self.get(name)
        if not math.isfinite(new_score):
            raise ConfigurationError(
                f"Score must be finite, got {new_score}",
                parameter="new_score",
            )
        cand.score = new_score

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (
            f"CandidatePool(n={self.size}, "
            f"best={self.best_score:.6f})"
        )


# ---------------------------------------------------------------------------
# SelectionResult
# ---------------------------------------------------------------------------


@dataclass
class SelectionResult:
    """Result of exponential mechanism selection.

    Attributes:
        selected: The chosen candidate.
        selection_probability: Probability of selecting this candidate.
        probabilities: Selection probabilities for all candidates.
        utility_bound: Upper bound on selected score minus optimal.
        epsilon_used: Privacy budget consumed.
        score_sensitivity: Score sensitivity used.
        n_candidates: Number of candidates considered.
    """

    selected: Candidate
    selection_probability: float
    probabilities: Dict[str, float]
    utility_bound: float
    epsilon_used: float
    score_sensitivity: float
    n_candidates: int

    def __repr__(self) -> str:
        return (
            f"SelectionResult(selected={self.selected.name!r}, "
            f"score={self.selected.score:.6f}, "
            f"prob={self.selection_probability:.4f}, "
            f"bound={self.utility_bound:.6f})"
        )


# ---------------------------------------------------------------------------
# ExponentialSelector
# ---------------------------------------------------------------------------


class ExponentialSelector:
    """Exponential mechanism for private mechanism selection.

    Given T candidate mechanisms with error scores, selects one
    privately with probability proportional to
    ``exp(-ε · score / (2Δ))`` where Δ is the score sensitivity.

    Args:
        epsilon_select: Privacy budget for selection.
        score_sensitivity: Sensitivity of the score function. If None,
            will be computed from candidates when possible.
        seed: Random seed for reproducibility.
        base_measure: Base measure weights (uniform if None).

    Raises:
        ConfigurationError: If epsilon_select <= 0.
    """

    def __init__(
        self,
        epsilon_select: float,
        *,
        score_sensitivity: Optional[float] = None,
        seed: Optional[int] = None,
        base_measure: Optional[npt.NDArray[np.float64]] = None,
    ) -> None:
        if epsilon_select <= 0:
            raise ConfigurationError(
                "epsilon_select must be positive",
                parameter="epsilon_select",
                value=epsilon_select,
                constraint="epsilon_select > 0",
            )
        if score_sensitivity is not None and score_sensitivity <= 0:
            raise ConfigurationError(
                "score_sensitivity must be positive",
                parameter="score_sensitivity",
                value=score_sensitivity,
                constraint="score_sensitivity > 0",
            )

        self.epsilon_select = epsilon_select
        self._score_sensitivity = score_sensitivity
        self._rng = np.random.default_rng(seed)
        self._base_measure = base_measure

    def select(
        self,
        pool: CandidatePool,
        *,
        score_sensitivity: Optional[float] = None,
    ) -> SelectionResult:
        """Select a candidate using the exponential mechanism.

        Args:
            pool: Pool of candidates to select from.
            score_sensitivity: Override the score sensitivity.

        Returns:
            A :class:`SelectionResult` with the chosen candidate
            and diagnostic information.

        Raises:
            DPForgeError: If pool is empty.
            ConfigurationError: If score sensitivity is not available.
        """
        if pool.size == 0:
            raise DPForgeError("Cannot select from empty pool")

        candidates = pool.candidates
        n = len(candidates)
        scores = np.array([c.score for c in candidates], dtype=np.float64)

        # Determine score sensitivity
        delta_u = score_sensitivity or self._score_sensitivity
        if delta_u is None:
            delta_u = self.score_sensitivity(pool)
        if delta_u <= 0:
            delta_u = 1.0
            logger.warning("Score sensitivity is 0; using default Δu=1.0")

        # Compute unnormalized log-probabilities
        # exp(-ε · score / (2Δu)) — note: lower score = higher utility
        # So we negate: u(x,r) = -score(r), giving exp(ε · (-score) / (2Δu))
        log_weights = -self.epsilon_select * scores / (2.0 * delta_u)

        # Apply base measure
        if self._base_measure is not None:
            bm = np.asarray(self._base_measure, dtype=np.float64)
            if len(bm) == n:
                log_weights += np.log(np.maximum(bm, 1e-300))

        # Numerical stability: subtract max
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        total = np.sum(weights)
        probs = weights / total

        # Sample
        idx = self._rng.choice(n, p=probs)
        selected = candidates[idx]

        # Build probability dict
        prob_dict = {c.name: float(probs[i]) for i, c in enumerate(candidates)}

        # Utility bound
        bound = self._utility_bound(n, delta_u)

        result = SelectionResult(
            selected=selected,
            selection_probability=float(probs[idx]),
            probabilities=prob_dict,
            utility_bound=bound,
            epsilon_used=self.epsilon_select,
            score_sensitivity=delta_u,
            n_candidates=n,
        )

        logger.info(
            "Exponential mechanism selected %r (score=%.6f, prob=%.4f) "
            "from %d candidates",
            selected.name, selected.score,
            float(probs[idx]), n,
        )

        return result

    def score_sensitivity(
        self,
        pool: CandidatePool,
        *,
        method: str = "range",
    ) -> float:
        """Compute or estimate score sensitivity.

        The score sensitivity Δu measures how much the score of any
        single candidate can change when one database record changes.

        Methods:
            - ``"range"``: Uses the score range as a conservative
              upper bound. This is valid when scores are computed
              from disjoint data.
            - ``"max_diff"``: Maximum absolute score difference between
              adjacent candidates. Appropriate when candidates are
              ordered by a single parameter.
            - ``"unit"``: Returns 1.0 (for normalized scores).

        Args:
            pool: Candidate pool.
            method: Sensitivity estimation method.

        Returns:
            Estimated score sensitivity Δu.
        """
        if method == "unit":
            return 1.0

        scores = pool.scores
        if len(scores) == 0:
            return 1.0

        if method == "range":
            return float(np.max(scores) - np.min(scores)) if len(scores) > 1 else 1.0

        if method == "max_diff":
            if len(scores) < 2:
                return 1.0
            sorted_scores = np.sort(scores)
            diffs = np.abs(sorted_scores[1:] - sorted_scores[:-1])
            return float(np.max(diffs))

        raise ConfigurationError(
            f"Unknown sensitivity method {method!r}",
            parameter="method",
            value=method,
        )

    def utility_bound(
        self,
        pool: CandidatePool,
        *,
        beta: float = 0.05,
        score_sensitivity: Optional[float] = None,
    ) -> float:
        """Compute utility guarantee for the exponential mechanism.

        With probability ≥ 1 - β, the selected candidate satisfies:

            score(selected) ≤ OPT + (2Δu / ε) · (ln(T) + ln(1/β))

        Args:
            pool: Candidate pool with T candidates.
            beta: Failure probability.
            score_sensitivity: Override score sensitivity.

        Returns:
            Upper bound on score(selected) - OPT.
        """
        delta_u = score_sensitivity or self._score_sensitivity
        if delta_u is None:
            delta_u = self.score_sensitivity(pool)
        if delta_u <= 0:
            delta_u = 1.0

        return self._utility_bound(pool.size, delta_u, beta=beta)

    def _utility_bound(
        self,
        n_candidates: int,
        delta_u: float,
        beta: float = 0.05,
    ) -> float:
        """Internal utility bound computation."""
        if n_candidates <= 1:
            return 0.0
        if beta <= 0 or beta >= 1:
            beta = 0.05

        return (2.0 * delta_u / self.epsilon_select) * (
            math.log(n_candidates) + math.log(1.0 / beta)
        )

    def selection_probabilities(
        self,
        pool: CandidatePool,
        *,
        score_sensitivity: Optional[float] = None,
    ) -> Dict[str, float]:
        """Compute selection probabilities without sampling.

        Args:
            pool: Candidate pool.
            score_sensitivity: Override score sensitivity.

        Returns:
            Dict mapping candidate names to selection probabilities.
        """
        if pool.size == 0:
            return {}

        candidates = pool.candidates
        scores = np.array([c.score for c in candidates], dtype=np.float64)

        delta_u = score_sensitivity or self._score_sensitivity
        if delta_u is None:
            delta_u = self.score_sensitivity(pool)
        if delta_u <= 0:
            delta_u = 1.0

        log_weights = -self.epsilon_select * scores / (2.0 * delta_u)
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        probs = weights / np.sum(weights)

        return {c.name: float(probs[i]) for i, c in enumerate(candidates)}


# ---------------------------------------------------------------------------
# Budget allocation
# ---------------------------------------------------------------------------


@dataclass
class BudgetAllocation:
    """Privacy budget allocation between synthesis and selection.

    Attributes:
        epsilon_total: Total privacy budget.
        epsilon_synthesis: Budget for mechanism synthesis.
        epsilon_selection: Budget for exponential mechanism selection.
        delta_total: Total delta budget.
        delta_synthesis: Delta for synthesis.
        delta_selection: Delta for selection (typically 0 for exp mech).
        split_ratio: Fraction of epsilon allocated to synthesis.
    """

    epsilon_total: float
    epsilon_synthesis: float
    epsilon_selection: float
    delta_total: float = 0.0
    delta_synthesis: float = 0.0
    delta_selection: float = 0.0
    split_ratio: float = 0.9

    def __post_init__(self) -> None:
        if self.epsilon_total <= 0:
            raise ConfigurationError(
                "epsilon_total must be positive",
                parameter="epsilon_total",
                value=self.epsilon_total,
            )
        consumed = self.epsilon_synthesis + self.epsilon_selection
        if consumed > self.epsilon_total + 1e-10:
            raise BudgetExhaustedError(
                f"Budget allocation exceeds total: "
                f"{consumed:.6f} > {self.epsilon_total:.6f}",
                budget_epsilon=self.epsilon_total,
                consumed_epsilon=consumed,
            )

    def __repr__(self) -> str:
        return (
            f"BudgetAllocation(total_ε={self.epsilon_total:.4f}, "
            f"synth={self.epsilon_synthesis:.4f}, "
            f"select={self.epsilon_selection:.4f}, "
            f"ratio={self.split_ratio:.2f})"
        )


def allocate_budget(
    epsilon_total: float,
    delta_total: float = 0.0,
    *,
    split_ratio: float = 0.9,
    n_candidates: int = 1,
    target_utility_gap: Optional[float] = None,
) -> BudgetAllocation:
    """Allocate privacy budget between synthesis and selection.

    By default, allocates 90% of ε to synthesis and 10% to selection
    (via basic composition).  Can optimize the split to minimize the
    total expected error.

    Args:
        epsilon_total: Total ε budget.
        delta_total: Total δ budget.
        split_ratio: Fraction of ε for synthesis (default 0.9).
        n_candidates: Number of candidate mechanisms.
        target_utility_gap: If given, compute the minimum ε_select
            needed to achieve this utility gap bound.

    Returns:
        A :class:`BudgetAllocation` instance.

    Raises:
        ConfigurationError: If parameters are invalid.
    """
    if epsilon_total <= 0:
        raise ConfigurationError(
            "epsilon_total must be positive",
            parameter="epsilon_total",
            value=epsilon_total,
        )
    if not (0 < split_ratio < 1):
        raise ConfigurationError(
            "split_ratio must be in (0, 1)",
            parameter="split_ratio",
            value=split_ratio,
        )

    if n_candidates <= 1:
        # No selection needed
        return BudgetAllocation(
            epsilon_total=epsilon_total,
            epsilon_synthesis=epsilon_total,
            epsilon_selection=0.0,
            delta_total=delta_total,
            delta_synthesis=delta_total,
            delta_selection=0.0,
            split_ratio=1.0,
        )

    if target_utility_gap is not None and target_utility_gap > 0:
        # Compute min ε_select for target gap:
        # gap = (2 * Δu / ε_s) * ln(T/β) with Δu=1, β=0.05
        # ε_s = 2 * ln(T/0.05) / gap
        ln_term = math.log(n_candidates) + math.log(20)  # ln(T) + ln(1/0.05)
        eps_select_needed = 2.0 * ln_term / target_utility_gap
        eps_select = min(eps_select_needed, epsilon_total * (1 - split_ratio))
        eps_synth = epsilon_total - eps_select
    else:
        eps_synth = epsilon_total * split_ratio
        eps_select = epsilon_total - eps_synth

    return BudgetAllocation(
        epsilon_total=epsilon_total,
        epsilon_synthesis=eps_synth,
        epsilon_selection=eps_select,
        delta_total=delta_total,
        delta_synthesis=delta_total,
        delta_selection=0.0,
        split_ratio=eps_synth / epsilon_total,
    )

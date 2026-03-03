"""Advanced selection strategies for MAP-Elites archive sampling.

Provides multiple selection policies that determine how parent solutions are
chosen from the archive for variation.  Each strategy implements
:meth:`select` which returns a list of :class:`~causal_qd.archive.archive_base.ArchiveEntry`
objects drawn from the current archive.

Strategies range from simple uniform random to curiosity-driven, novelty-
based, quality-weighted, tournament, rank, Boltzmann (softmax), and
multi-objective Pareto selection.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from causal_qd.archive.archive_base import Archive, ArchiveEntry
from causal_qd.types import BehavioralDescriptor, CellIndex, QualityScore


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class SelectionStrategy(ABC):
    """Abstract base class for archive selection strategies.

    Subclasses implement :meth:`select` to draw *n* parent solutions from
    the archive according to a particular policy.
    """

    @abstractmethod
    def select(
        self,
        archive: Archive,
        n: int,
        rng: np.random.Generator,
    ) -> List[ArchiveEntry]:
        """Select *n* entries from *archive*.

        Parameters
        ----------
        archive : Archive
            The MAP-Elites archive to select from.
        n : int
            Number of parents to select.
        rng : numpy.random.Generator
            Random number generator for reproducibility.

        Returns
        -------
        List[ArchiveEntry]
            Selected parent entries.
        """

    def update(self, entry: ArchiveEntry, was_improvement: bool) -> None:
        """Optional hook called after each offspring evaluation.

        Parameters
        ----------
        entry : ArchiveEntry
            The offspring that was evaluated.
        was_improvement : bool
            Whether the offspring was accepted into the archive.
        """

    def reset(self) -> None:
        """Reset any internal state (e.g., between runs)."""


# ---------------------------------------------------------------------------
# UniformSelection
# ---------------------------------------------------------------------------


class UniformSelection(SelectionStrategy):
    """Select parents uniformly at random from all occupied cells.

    This is the simplest and most common strategy in vanilla MAP-Elites.
    Every occupied cell has equal probability of being chosen regardless
    of the quality of its elite.
    """

    def select(
        self,
        archive: Archive,
        n: int,
        rng: np.random.Generator,
    ) -> List[ArchiveEntry]:
        """Draw *n* elites uniformly at random with replacement.

        Parameters
        ----------
        archive : Archive
            Source archive.
        n : int
            Number of parents to draw.
        rng : numpy.random.Generator
            Random state.

        Returns
        -------
        List[ArchiveEntry]
            *n* uniformly sampled elites.
        """
        return archive.sample(n, rng)


# ---------------------------------------------------------------------------
# CuriosityDrivenSelection
# ---------------------------------------------------------------------------


@dataclass
class _CellStats:
    """Per-cell statistics for curiosity tracking."""
    visit_count: int = 0
    improvement_count: int = 0
    total_quality_gain: float = 0.0
    last_improvement_gen: int = -1
    quality_history: List[float] = field(default_factory=list)


class CuriosityDrivenSelection(SelectionStrategy):
    """Select cells with high improvement potential using UCB1.

    Maintains per-cell statistics tracking how often each cell has been
    selected and how often the resulting offspring led to an archive
    improvement.  Cells are scored with a UCB1-style formula that
    balances exploitation (high historical improvement rate) with
    exploration (rarely visited cells) and novelty (cells not improved
    recently).

    Parameters
    ----------
    exploration_constant : float
        UCB1 exploration coefficient *c*.  Higher values increase the
        tendency to explore rarely visited cells.  Default ``2.0``.
    novelty_weight : float
        Weight of the novelty bonus for cells that haven't been improved
        in a while.  Default ``0.5``.
    window_size : int
        Number of recent interactions to consider when computing
        improvement rate.  Default ``200``.
    """

    def __init__(
        self,
        exploration_constant: float = 2.0,
        novelty_weight: float = 0.5,
        window_size: int = 200,
    ) -> None:
        self._c = exploration_constant
        self._novelty_weight = novelty_weight
        self._window_size = window_size
        self._cell_stats: Dict[CellIndex, _CellStats] = defaultdict(_CellStats)
        self._total_visits: int = 0
        self._current_generation: int = 0
        self._recent_history: deque[Tuple[CellIndex, bool]] = deque(
            maxlen=window_size
        )
        self._last_selected_cells: List[CellIndex] = []

    def select(
        self,
        archive: Archive,
        n: int,
        rng: np.random.Generator,
    ) -> List[ArchiveEntry]:
        """Select *n* parents using UCB1 scores over archive cells.

        Each occupied cell is assigned a UCB1 score combining its
        improvement rate, visit frequency, and novelty bonus.  Parents
        are sampled proportionally to softmax(UCB1 scores).

        Parameters
        ----------
        archive : Archive
            Source archive.
        n : int
            Number of parents.
        rng : numpy.random.Generator
            Random state.

        Returns
        -------
        List[ArchiveEntry]
        """
        elites = archive.elites()
        if not elites:
            return []
        if len(elites) == 1:
            return elites * n

        scores = np.array(
            [self._ucb1_score(e) for e in elites], dtype=np.float64
        )

        # Softmax to probabilities
        scores -= scores.max()
        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum()

        indices = rng.choice(len(elites), size=n, replace=True, p=probs)
        self._last_selected_cells = []
        selected: List[ArchiveEntry] = []
        for idx in indices:
            entry = elites[idx]
            selected.append(entry)
            # Derive a cell index from the entry
            cell = self._entry_cell(entry)
            self._cell_stats[cell].visit_count += 1
            self._total_visits += 1
            self._last_selected_cells.append(cell)

        return selected

    def update(self, entry: ArchiveEntry, was_improvement: bool) -> None:
        """Record the outcome of an offspring generated from the last selection.

        Parameters
        ----------
        entry : ArchiveEntry
            The evaluated offspring.
        was_improvement : bool
            Whether it was accepted into the archive.
        """
        self._current_generation += 1

        if self._last_selected_cells:
            cell = self._last_selected_cells.pop(0)
            self._recent_history.append((cell, was_improvement))
            stats = self._cell_stats[cell]
            if was_improvement:
                stats.improvement_count += 1
                stats.total_quality_gain += entry.quality
                stats.last_improvement_gen = self._current_generation
            stats.quality_history.append(entry.quality)
            if len(stats.quality_history) > self._window_size:
                stats.quality_history = stats.quality_history[-self._window_size:]

    def reset(self) -> None:
        """Clear all tracked statistics."""
        self._cell_stats.clear()
        self._total_visits = 0
        self._current_generation = 0
        self._recent_history.clear()
        self._last_selected_cells.clear()

    def _ucb1_score(self, entry: ArchiveEntry) -> float:
        """Compute UCB1 score for a cell.

        UCB1(cell) = improvement_rate + c * sqrt(ln(N) / n_cell) + novelty_bonus

        where improvement_rate is the fraction of visits that resulted in
        archive improvements, N is total visits across all cells, n_cell
        is visits to this cell, and novelty_bonus rewards cells not
        improved recently.
        """
        cell = self._entry_cell(entry)
        stats = self._cell_stats[cell]

        if stats.visit_count == 0:
            return float("inf")

        # Exploitation: improvement rate
        improvement_rate = stats.improvement_count / max(stats.visit_count, 1)

        # Exploration: UCB1 term
        if self._total_visits > 0:
            exploration = self._c * math.sqrt(
                math.log(self._total_visits) / stats.visit_count
            )
        else:
            exploration = self._c

        # Novelty bonus: reward cells not improved recently
        if stats.last_improvement_gen >= 0:
            generations_since = self._current_generation - stats.last_improvement_gen
            novelty = self._novelty_weight * math.log1p(generations_since)
        else:
            novelty = self._novelty_weight * math.log1p(self._current_generation + 1)

        return improvement_rate + exploration + novelty

    @staticmethod
    def _entry_cell(entry: ArchiveEntry) -> CellIndex:
        """Extract a hashable cell index from an entry.

        Uses the descriptor rounded to 2 decimal places as a proxy when
        no explicit cell_index attribute is available.
        """
        if hasattr(entry, "cell_index") and hasattr(entry, "metadata"):
            ci = entry.metadata.get("cell_index")
            if ci is not None:
                return tuple(ci) if not isinstance(ci, tuple) else ci
        desc = np.asarray(entry.descriptor)
        return tuple(np.round(desc, 2).tolist())


# ---------------------------------------------------------------------------
# NoveltySelection
# ---------------------------------------------------------------------------


class NoveltySelection(SelectionStrategy):
    """Select parents whose offspring tend to produce novel descriptors.

    Assigns each elite a *novelty score* based on the average distance
    to its *k* nearest neighbours in descriptor space.  Parents with
    higher novelty scores are more likely to be selected.

    Parameters
    ----------
    k_neighbors : int
        Number of nearest neighbours for novelty computation.
        Default ``15``.
    novelty_archive_size : int
        Maximum number of historical descriptors stored in the novelty
        archive.  Default ``500``.
    weight_exponent : float
        Exponent applied to novelty scores before normalization.
        Higher values increase selection pressure towards novel elites.
        Default ``1.0``.
    """

    def __init__(
        self,
        k_neighbors: int = 15,
        novelty_archive_size: int = 500,
        weight_exponent: float = 1.0,
    ) -> None:
        self._k = k_neighbors
        self._max_archive = novelty_archive_size
        self._exponent = weight_exponent
        self._novelty_archive: List[npt.NDArray[np.float64]] = []

    def select(
        self,
        archive: Archive,
        n: int,
        rng: np.random.Generator,
    ) -> List[ArchiveEntry]:
        """Select *n* parents weighted by novelty score.

        Parameters
        ----------
        archive : Archive
            Source archive.
        n : int
            Number of parents.
        rng : numpy.random.Generator
            Random state.

        Returns
        -------
        List[ArchiveEntry]
        """
        elites = archive.elites()
        if not elites:
            return []
        if len(elites) == 1:
            return elites * n

        descriptors = np.array([e.descriptor for e in elites], dtype=np.float64)

        # Combine archive descriptors with novelty archive
        if self._novelty_archive:
            all_desc = np.vstack(
                [descriptors, np.array(self._novelty_archive, dtype=np.float64)]
            )
        else:
            all_desc = descriptors

        novelty_scores = self._compute_novelty_scores(descriptors, all_desc)

        # Apply exponent and normalize to probabilities
        scores = np.power(novelty_scores + 1e-10, self._exponent)
        probs = scores / scores.sum()

        indices = rng.choice(len(elites), size=n, replace=True, p=probs)
        return [elites[i] for i in indices]

    def update(self, entry: ArchiveEntry, was_improvement: bool) -> None:
        """Add the offspring descriptor to the novelty archive."""
        desc = np.asarray(entry.descriptor, dtype=np.float64)
        self._novelty_archive.append(desc.copy())
        if len(self._novelty_archive) > self._max_archive:
            self._novelty_archive.pop(0)

    def reset(self) -> None:
        """Clear the novelty archive."""
        self._novelty_archive.clear()

    def _compute_novelty_scores(
        self,
        query_descriptors: npt.NDArray[np.float64],
        reference_descriptors: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute k-nearest-neighbour novelty for each query descriptor.

        For each query descriptor, find the *k* closest descriptors in
        the reference set (excluding itself if present) and average
        the distances.

        Parameters
        ----------
        query_descriptors : ndarray, shape (m, d)
            Descriptors to score.
        reference_descriptors : ndarray, shape (M, d)
            Reference pool including archive and novelty archive.

        Returns
        -------
        ndarray, shape (m,)
            Mean k-NN distance for each query descriptor.
        """
        m = query_descriptors.shape[0]
        k = min(self._k, reference_descriptors.shape[0] - 1)
        k = max(k, 1)

        scores = np.zeros(m, dtype=np.float64)
        for i in range(m):
            dists = np.linalg.norm(
                reference_descriptors - query_descriptors[i], axis=1
            )
            # Sort and skip zero-distance (self)
            sorted_dists = np.sort(dists)
            # Take k nearest (skip first if it is 0.0 = self)
            start = 1 if sorted_dists[0] < 1e-12 else 0
            end = start + k
            if end > len(sorted_dists):
                end = len(sorted_dists)
            if end <= start:
                scores[i] = 0.0
            else:
                scores[i] = float(np.mean(sorted_dists[start:end]))

        return scores


# ---------------------------------------------------------------------------
# QualityWeightedSelection
# ---------------------------------------------------------------------------


class QualityWeightedSelection(SelectionStrategy):
    """Select parents with probability proportional to their quality.

    Higher-quality elites are more likely to be selected.  A minimum
    probability floor prevents starvation of low-quality cells.

    Parameters
    ----------
    min_prob : float
        Minimum selection probability for any elite, preventing
        complete starvation.  Default ``0.01``.
    temperature : float
        Temperature for softmax conversion of quality scores.
        Lower temperature → more greedy.  Default ``1.0``.
    """

    def __init__(
        self,
        min_prob: float = 0.01,
        temperature: float = 1.0,
    ) -> None:
        self._min_prob = min_prob
        self._temperature = max(temperature, 1e-10)

    def select(
        self,
        archive: Archive,
        n: int,
        rng: np.random.Generator,
    ) -> List[ArchiveEntry]:
        """Select *n* parents proportional to quality score.

        Parameters
        ----------
        archive, n, rng
            Standard selection parameters.

        Returns
        -------
        List[ArchiveEntry]
        """
        elites = archive.elites()
        if not elites:
            return []
        if len(elites) == 1:
            return elites * n

        qualities = np.array([e.quality for e in elites], dtype=np.float64)

        # Softmax with temperature
        logits = qualities / self._temperature
        logits -= logits.max()
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()

        # Enforce minimum probability
        probs = np.maximum(probs, self._min_prob)
        probs /= probs.sum()

        indices = rng.choice(len(elites), size=n, replace=True, p=probs)
        return [elites[i] for i in indices]


# ---------------------------------------------------------------------------
# TournamentSelection
# ---------------------------------------------------------------------------


class TournamentSelection(SelectionStrategy):
    """Tournament selection: pick k random elites, return the best.

    For each of the *n* requested parents, a tournament of *k* randomly
    chosen elites is held and the one with the highest quality wins.

    Parameters
    ----------
    tournament_size : int
        Number of elites competing in each tournament.  Default ``5``.
    """

    def __init__(self, tournament_size: int = 5) -> None:
        self._k = tournament_size

    def select(
        self,
        archive: Archive,
        n: int,
        rng: np.random.Generator,
    ) -> List[ArchiveEntry]:
        """Run *n* tournaments of size *k*, each returning the best elite.

        Parameters
        ----------
        archive, n, rng
            Standard selection parameters.

        Returns
        -------
        List[ArchiveEntry]
        """
        elites = archive.elites()
        if not elites:
            return []
        if len(elites) == 1:
            return elites * n

        result: List[ArchiveEntry] = []
        k = min(self._k, len(elites))
        for _ in range(n):
            competitors_idx = rng.choice(len(elites), size=k, replace=False)
            best_idx = max(competitors_idx, key=lambda i: elites[i].quality)
            result.append(elites[best_idx])
        return result


# ---------------------------------------------------------------------------
# RankSelection
# ---------------------------------------------------------------------------


class RankSelection(SelectionStrategy):
    """Rank-based selection: probability proportional to rank.

    Elites are sorted by quality and assigned ranks 1, 2, …, N (best
    gets highest rank).  Selection probability is proportional to rank.

    Parameters
    ----------
    selection_pressure : float
        Controls the ratio of highest to lowest selection probability.
        A value of 2.0 means the best elite is twice as likely to be
        selected as the median elite.  Default ``1.5``.
    """

    def __init__(self, selection_pressure: float = 1.5) -> None:
        self._sp = max(selection_pressure, 1.0)

    def select(
        self,
        archive: Archive,
        n: int,
        rng: np.random.Generator,
    ) -> List[ArchiveEntry]:
        """Select *n* parents using rank-based probabilities.

        Parameters
        ----------
        archive, n, rng
            Standard selection parameters.

        Returns
        -------
        List[ArchiveEntry]
        """
        elites = archive.elites()
        if not elites:
            return []
        if len(elites) == 1:
            return elites * n

        N = len(elites)
        qualities = np.array([e.quality for e in elites], dtype=np.float64)
        sorted_indices = np.argsort(qualities)  # ascending
        ranks = np.empty(N, dtype=np.float64)
        for rank_pos, orig_idx in enumerate(sorted_indices):
            ranks[orig_idx] = rank_pos + 1  # 1-based rank

        # Linear ranking: P(i) = (2 - sp) / N + 2 * rank(i) * (sp - 1) / (N * (N - 1))
        # Simplified: proportional to rank^alpha
        probs = np.power(ranks, self._sp - 1.0)
        probs /= probs.sum()

        indices = rng.choice(N, size=n, replace=True, p=probs)
        return [elites[i] for i in indices]


# ---------------------------------------------------------------------------
# BoltzmannSelection
# ---------------------------------------------------------------------------


class BoltzmannSelection(SelectionStrategy):
    """Softmax (Boltzmann) selection with configurable temperature schedule.

    Selection probability is ``exp(quality / T) / Z`` where *T* is the
    temperature and *Z* is the partition function.  Higher temperature
    gives more uniform selection; lower temperature makes selection
    more greedy.

    Parameters
    ----------
    initial_temperature : float
        Starting temperature.  Default ``1.0``.
    min_temperature : float
        Minimum temperature (floor).  Default ``0.01``.
    cooling_rate : float
        Multiplicative cooling factor applied each time :meth:`update`
        is called.  ``T ← T * cooling_rate``.  Default ``0.999``.
    """

    def __init__(
        self,
        initial_temperature: float = 1.0,
        min_temperature: float = 0.01,
        cooling_rate: float = 0.999,
    ) -> None:
        self._temp = initial_temperature
        self._min_temp = min_temperature
        self._cooling_rate = cooling_rate
        self._initial_temp = initial_temperature

    @property
    def temperature(self) -> float:
        """Current Boltzmann temperature."""
        return self._temp

    def select(
        self,
        archive: Archive,
        n: int,
        rng: np.random.Generator,
    ) -> List[ArchiveEntry]:
        """Select *n* parents using Boltzmann probabilities.

        Parameters
        ----------
        archive, n, rng
            Standard selection parameters.

        Returns
        -------
        List[ArchiveEntry]
        """
        elites = archive.elites()
        if not elites:
            return []
        if len(elites) == 1:
            return elites * n

        qualities = np.array([e.quality for e in elites], dtype=np.float64)
        logits = qualities / max(self._temp, 1e-15)
        logits -= logits.max()
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()

        indices = rng.choice(len(elites), size=n, replace=True, p=probs)
        return [elites[i] for i in indices]

    def update(self, entry: ArchiveEntry, was_improvement: bool) -> None:
        """Cool the temperature by the cooling rate."""
        self._temp = max(self._temp * self._cooling_rate, self._min_temp)

    def reset(self) -> None:
        """Reset temperature to initial value."""
        self._temp = self._initial_temp


# ---------------------------------------------------------------------------
# MultiObjectiveSelection
# ---------------------------------------------------------------------------


class MultiObjectiveSelection(SelectionStrategy):
    """Pareto-optimal selection across multiple quality dimensions.

    When solutions have multiple quality objectives (e.g., BIC score,
    sparsity, robustness) stored in ``entry.metadata``, this strategy
    preferentially selects solutions on the Pareto front.

    Parameters
    ----------
    objective_keys : List[str]
        Keys in ``entry.metadata`` to use as quality dimensions.
        If empty, uses ``entry.quality`` as the sole objective.
    pareto_fraction : float
        Fraction of selections drawn from the Pareto front.  Remaining
        selections are uniform random.  Default ``0.7``.
    """

    def __init__(
        self,
        objective_keys: Optional[List[str]] = None,
        pareto_fraction: float = 0.7,
    ) -> None:
        self._keys = objective_keys or []
        self._pareto_frac = min(max(pareto_fraction, 0.0), 1.0)

    def select(
        self,
        archive: Archive,
        n: int,
        rng: np.random.Generator,
    ) -> List[ArchiveEntry]:
        """Select *n* parents, preferring Pareto-optimal solutions.

        Parameters
        ----------
        archive, n, rng
            Standard selection parameters.

        Returns
        -------
        List[ArchiveEntry]
        """
        elites = archive.elites()
        if not elites:
            return []
        if len(elites) == 1:
            return elites * n

        objectives = self._extract_objectives(elites)
        pareto_mask = self._compute_pareto_front(objectives)
        pareto_indices = np.where(pareto_mask)[0]
        non_pareto_indices = np.where(~pareto_mask)[0]

        n_pareto = int(n * self._pareto_frac)
        n_other = n - n_pareto

        selected: List[ArchiveEntry] = []

        # Draw from Pareto front
        if len(pareto_indices) > 0 and n_pareto > 0:
            pi = rng.choice(pareto_indices, size=n_pareto, replace=True)
            selected.extend(elites[i] for i in pi)

        # Draw from remaining
        all_indices = np.arange(len(elites))
        pool = non_pareto_indices if len(non_pareto_indices) > 0 else all_indices
        if n_other > 0:
            oi = rng.choice(pool, size=n_other, replace=True)
            selected.extend(elites[i] for i in oi)

        return selected

    def _extract_objectives(
        self, elites: List[ArchiveEntry]
    ) -> npt.NDArray[np.float64]:
        """Extract objective matrix from elites.

        Parameters
        ----------
        elites : List[ArchiveEntry]
            Elites to extract objectives from.

        Returns
        -------
        ndarray, shape (m, k)
            Matrix of objective values.  Higher is better.
        """
        if not self._keys:
            return np.array(
                [[e.quality] for e in elites], dtype=np.float64
            )

        rows = []
        for e in elites:
            row = [e.quality]
            for key in self._keys:
                val = e.metadata.get(key, 0.0)
                row.append(float(val))
            rows.append(row)
        return np.array(rows, dtype=np.float64)

    @staticmethod
    def _compute_pareto_front(
        objectives: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.bool_]:
        """Identify Pareto-optimal solutions (maximization).

        A solution *i* is Pareto-optimal if no other solution *j*
        dominates it (i.e., *j* is at least as good on all objectives
        and strictly better on at least one).

        Parameters
        ----------
        objectives : ndarray, shape (m, k)
            Objective values for each solution.

        Returns
        -------
        ndarray, shape (m,)
            Boolean mask where ``True`` indicates a Pareto-optimal solution.
        """
        m = objectives.shape[0]
        is_pareto = np.ones(m, dtype=np.bool_)

        for i in range(m):
            if not is_pareto[i]:
                continue
            for j in range(m):
                if i == j or not is_pareto[j]:
                    continue
                # Check if j dominates i
                at_least_as_good = np.all(objectives[j] >= objectives[i])
                strictly_better = np.any(objectives[j] > objectives[i])
                if at_least_as_good and strictly_better:
                    is_pareto[i] = False
                    break

        return is_pareto


# ---------------------------------------------------------------------------
# EpsilonGreedySelection
# ---------------------------------------------------------------------------


class EpsilonGreedySelection(SelectionStrategy):
    """Epsilon-greedy: select best with probability 1-ε, random otherwise.

    With probability ``1 - epsilon`` the highest-quality elite is
    selected; otherwise a uniformly random elite is chosen.

    Parameters
    ----------
    epsilon : float
        Exploration rate.  Default ``0.1``.
    decay_rate : float
        Multiplicative decay applied to epsilon after each update.
        Default ``0.9995``.
    min_epsilon : float
        Floor for epsilon.  Default ``0.01``.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        decay_rate: float = 0.9995,
        min_epsilon: float = 0.01,
    ) -> None:
        self._epsilon = epsilon
        self._decay_rate = decay_rate
        self._min_epsilon = min_epsilon
        self._initial_epsilon = epsilon

    @property
    def epsilon(self) -> float:
        """Current epsilon value."""
        return self._epsilon

    def select(
        self,
        archive: Archive,
        n: int,
        rng: np.random.Generator,
    ) -> List[ArchiveEntry]:
        """Epsilon-greedy selection of *n* parents.

        Parameters
        ----------
        archive, n, rng
            Standard selection parameters.

        Returns
        -------
        List[ArchiveEntry]
        """
        elites = archive.elites()
        if not elites:
            return []
        if len(elites) == 1:
            return elites * n

        best = archive.best()
        selected: List[ArchiveEntry] = []
        for _ in range(n):
            if rng.random() < self._epsilon:
                idx = rng.integers(0, len(elites))
                selected.append(elites[idx])
            else:
                selected.append(best)
        return selected

    def update(self, entry: ArchiveEntry, was_improvement: bool) -> None:
        """Decay epsilon."""
        self._epsilon = max(
            self._epsilon * self._decay_rate, self._min_epsilon
        )

    def reset(self) -> None:
        """Reset epsilon to initial value."""
        self._epsilon = self._initial_epsilon


# ---------------------------------------------------------------------------
# Composite selection
# ---------------------------------------------------------------------------


class CompositeSelection(SelectionStrategy):
    """Combine multiple selection strategies with configurable weights.

    Each call to :meth:`select` randomly chooses which strategy to use
    based on the given weights.  This enables flexible mixing of
    exploration-focused and exploitation-focused selection.

    Parameters
    ----------
    strategies : Sequence[SelectionStrategy]
        Component strategies.
    weights : Sequence[float] | None
        Per-strategy weight.  Normalized internally.
        If ``None``, uniform weights are used.
    """

    def __init__(
        self,
        strategies: Sequence[SelectionStrategy],
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        self._strategies = list(strategies)
        if weights is None:
            self._weights = np.ones(len(strategies)) / len(strategies)
        else:
            w = np.array(weights, dtype=np.float64)
            self._weights = w / w.sum()

    def select(
        self,
        archive: Archive,
        n: int,
        rng: np.random.Generator,
    ) -> List[ArchiveEntry]:
        """Select *n* parents using a randomly chosen sub-strategy.

        For each parent, one strategy is sampled according to the
        weights and used to select a single parent.

        Parameters
        ----------
        archive, n, rng
            Standard selection parameters.

        Returns
        -------
        List[ArchiveEntry]
        """
        if not self._strategies:
            return []

        selected: List[ArchiveEntry] = []
        for _ in range(n):
            idx = rng.choice(len(self._strategies), p=self._weights)
            result = self._strategies[idx].select(archive, 1, rng)
            if result:
                selected.append(result[0])

        return selected

    def update(self, entry: ArchiveEntry, was_improvement: bool) -> None:
        """Forward update to all sub-strategies."""
        for s in self._strategies:
            s.update(entry, was_improvement)

    def reset(self) -> None:
        """Reset all sub-strategies."""
        for s in self._strategies:
            s.reset()

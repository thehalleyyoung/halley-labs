"""Curiosity signal computation for the QD search.

Combines novelty (inverse visit count) and surprise (deviation from
quality EMA) into a single curiosity score that guides parent selection
toward under-explored and surprising regions of the behavior space.

Classes
-------
CuriosityComputer
    Main curiosity signal computation engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cpa.utils.logging import get_logger

logger = get_logger("exploration.curiosity")


# ---------------------------------------------------------------------------
# Curiosity configuration
# ---------------------------------------------------------------------------


@dataclass
class CuriosityConfig:
    """Configuration for curiosity computation.

    Attributes
    ----------
    mu : float
        Weight for the surprise component (default 0.5).
    ema_alpha : float
        Exponential moving average decay for quality EMA (default 0.1).
    novelty_decay : float
        Decay factor for novelty (controls how fast novelty drops, default 1.0).
    min_curiosity : float
        Minimum curiosity value to prevent starvation (default 0.01).
    max_curiosity : float
        Maximum curiosity value for clipping (default 10.0).
    exploration_bonus : float
        Bonus curiosity for completely unvisited cells (default 2.0).
    temperature : float
        Temperature for curiosity-based selection (default 1.0).
    curiosity_decay_rate : float
        Rate at which global curiosity decays over generations (default 0.999).
    """

    mu: float = 0.5
    ema_alpha: float = 0.1
    novelty_decay: float = 1.0
    min_curiosity: float = 0.01
    max_curiosity: float = 10.0
    exploration_bonus: float = 2.0
    temperature: float = 1.0
    curiosity_decay_rate: float = 0.999


# ---------------------------------------------------------------------------
# Curiosity computer
# ---------------------------------------------------------------------------


class CuriosityComputer:
    """Compute curiosity signals for the QD search.

    Curiosity = novelty + mu * surprise

    where:
    - novelty = 1 / (1 + visit_count) + exploration_bonus (if unvisited)
    - surprise = |quality - quality_EMA|
    - quality_EMA is an exponential moving average of quality per cell

    Parameters
    ----------
    n_cells : int
        Number of CVT cells in the archive.
    config : CuriosityConfig, optional
        Curiosity computation parameters.

    Examples
    --------
    >>> cc = CuriosityComputer(n_cells=1000)
    >>> curiosity = cc.compute(cell_idx=42, quality=0.8)
    >>> batch = cc.compute_batch(cell_indices, qualities)
    """

    def __init__(
        self,
        n_cells: int,
        config: Optional[CuriosityConfig] = None,
    ) -> None:
        self.n_cells = n_cells
        self.config = config or CuriosityConfig()

        # Per-cell tracking arrays
        self._visit_counts = np.zeros(n_cells, dtype=np.int64)
        self._quality_ema = np.zeros(n_cells, dtype=np.float64)
        self._quality_ema_initialized = np.zeros(n_cells, dtype=bool)

        # Global tracking
        self._total_evaluations = 0
        self._generation = 0
        self._global_curiosity_scale = 1.0

        # History for logging/analysis
        self._curiosity_history: List[Dict[str, float]] = []
        self._novelty_history: List[float] = []
        self._surprise_history: List[float] = []

    # ----- Core computation -----

    def compute(self, cell_idx: int, quality: float) -> float:
        """Compute the curiosity signal for a single evaluation.

        Parameters
        ----------
        cell_idx : int
            CVT cell index where the genome was placed.
        quality : float
            Quality score of the genome.

        Returns
        -------
        float
            Curiosity value.
        """
        novelty = self._compute_novelty(cell_idx)
        surprise = self._compute_surprise(cell_idx, quality)

        # Update tracking
        self._update_ema(cell_idx, quality)
        self._visit_counts[cell_idx] += 1
        self._total_evaluations += 1

        # Combined curiosity
        curiosity = novelty + self.config.mu * surprise
        curiosity *= self._global_curiosity_scale
        curiosity = np.clip(
            curiosity, self.config.min_curiosity, self.config.max_curiosity
        )

        return float(curiosity)

    def compute_batch(
        self,
        cell_indices: np.ndarray,
        qualities: np.ndarray,
    ) -> np.ndarray:
        """Compute curiosity signals for a batch of evaluations.

        Parameters
        ----------
        cell_indices : np.ndarray
            Shape (N,) integer cell indices.
        qualities : np.ndarray
            Shape (N,) quality scores.

        Returns
        -------
        np.ndarray
            Shape (N,) curiosity values.
        """
        n = len(cell_indices)
        curiosities = np.zeros(n, dtype=np.float64)
        novelties = np.zeros(n, dtype=np.float64)
        surprises = np.zeros(n, dtype=np.float64)

        for i in range(n):
            idx = int(cell_indices[i])
            q = float(qualities[i])

            novelties[i] = self._compute_novelty(idx)
            surprises[i] = self._compute_surprise(idx, q)

            # Update tracking
            self._update_ema(idx, q)
            self._visit_counts[idx] += 1
            self._total_evaluations += 1

        curiosities = novelties + self.config.mu * surprises
        curiosities *= self._global_curiosity_scale
        curiosities = np.clip(
            curiosities, self.config.min_curiosity, self.config.max_curiosity
        )

        # Record history
        if len(novelties) > 0:
            self._novelty_history.append(float(np.mean(novelties)))
            self._surprise_history.append(float(np.mean(surprises)))
            self._curiosity_history.append({
                "generation": self._generation,
                "mean_curiosity": float(np.mean(curiosities)),
                "mean_novelty": float(np.mean(novelties)),
                "mean_surprise": float(np.mean(surprises)),
                "max_curiosity": float(np.max(curiosities)),
                "min_curiosity": float(np.min(curiosities)),
                "total_evaluations": self._total_evaluations,
            })

        return curiosities

    def peek_curiosity(self, cell_idx: int) -> float:
        """Peek at the current curiosity for a cell without updating.

        Useful for visualization and selection without side effects.

        Parameters
        ----------
        cell_idx : int

        Returns
        -------
        float
        """
        novelty = self._compute_novelty(cell_idx)
        # Use current EMA for surprise estimate (quality=EMA → 0 surprise)
        surprise = 0.0  # No actual quality to compare
        curiosity = novelty + self.config.mu * surprise
        curiosity *= self._global_curiosity_scale
        return float(np.clip(
            curiosity, self.config.min_curiosity, self.config.max_curiosity
        ))

    def peek_batch(self, cell_indices: np.ndarray) -> np.ndarray:
        """Peek at curiosity values for multiple cells without updating.

        Parameters
        ----------
        cell_indices : np.ndarray
            Shape (N,) cell indices.

        Returns
        -------
        np.ndarray
            Shape (N,) curiosity values.
        """
        novelties = np.array([
            self._compute_novelty(int(idx)) for idx in cell_indices
        ])
        curiosities = novelties * self._global_curiosity_scale
        return np.clip(
            curiosities, self.config.min_curiosity, self.config.max_curiosity
        )

    # ----- Internal computations -----

    def _compute_novelty(self, cell_idx: int) -> float:
        """Compute novelty for a cell based on visit count.

        Parameters
        ----------
        cell_idx : int

        Returns
        -------
        float
            Novelty score in [0, ∞).
        """
        count = self._visit_counts[cell_idx]
        base_novelty = 1.0 / (1.0 + count * self.config.novelty_decay)

        # Exploration bonus for unvisited cells
        if count == 0:
            base_novelty += self.config.exploration_bonus

        return base_novelty

    def _compute_surprise(self, cell_idx: int, quality: float) -> float:
        """Compute surprise as deviation from quality EMA.

        Parameters
        ----------
        cell_idx : int
        quality : float

        Returns
        -------
        float
            Surprise score (absolute deviation).
        """
        if not self._quality_ema_initialized[cell_idx]:
            # First visit: surprise is proportional to quality magnitude
            return abs(quality)

        ema = self._quality_ema[cell_idx]
        return abs(quality - ema)

    def _update_ema(self, cell_idx: int, quality: float) -> None:
        """Update the quality EMA for a cell.

        Parameters
        ----------
        cell_idx : int
        quality : float
        """
        alpha = self.config.ema_alpha
        if not self._quality_ema_initialized[cell_idx]:
            self._quality_ema[cell_idx] = quality
            self._quality_ema_initialized[cell_idx] = True
        else:
            self._quality_ema[cell_idx] = (
                alpha * quality + (1 - alpha) * self._quality_ema[cell_idx]
            )

    # ----- Generation management -----

    def advance_generation(self) -> None:
        """Advance to the next generation, applying curiosity decay.

        Should be called at the end of each QD search iteration.
        """
        self._generation += 1
        self._global_curiosity_scale *= self.config.curiosity_decay_rate
        self._global_curiosity_scale = max(self._global_curiosity_scale, 0.01)

    def reset_generation_counter(self) -> None:
        """Reset the generation counter and curiosity scale."""
        self._generation = 0
        self._global_curiosity_scale = 1.0

    # ----- Exploration-exploitation balance -----

    def exploration_ratio(self) -> float:
        """Estimate current exploration vs exploitation balance.

        Returns
        -------
        float
            Value in [0, 1]. High = exploring, low = exploiting.
        """
        if self._total_evaluations == 0:
            return 1.0

        # Fraction of unvisited cells
        unvisited = float(np.mean(self._visit_counts == 0))

        # Coefficient of variation of visit counts
        visited_counts = self._visit_counts[self._visit_counts > 0]
        if len(visited_counts) < 2:
            cv = 1.0
        else:
            cv = float(np.std(visited_counts) / (np.mean(visited_counts) + 1e-15))

        # High unvisited + high CV = exploring
        return 0.6 * unvisited + 0.4 * min(cv, 1.0)

    def should_increase_exploration(self, target_ratio: float = 0.3) -> bool:
        """Check if exploration should be increased.

        Parameters
        ----------
        target_ratio : float
            Target exploration ratio.

        Returns
        -------
        bool
        """
        return self.exploration_ratio() < target_ratio

    def adjust_temperature(self, target_exploration: float = 0.3) -> float:
        """Adaptively adjust the selection temperature.

        Increases temperature if exploration is too low, decreases
        if exploitation is desired.

        Parameters
        ----------
        target_exploration : float

        Returns
        -------
        float
            New temperature value.
        """
        current = self.exploration_ratio()
        if current < target_exploration:
            self.config.temperature *= 1.1
        elif current > target_exploration + 0.2:
            self.config.temperature *= 0.95

        self.config.temperature = np.clip(self.config.temperature, 0.1, 10.0)
        return self.config.temperature

    # ----- Statistics and logging -----

    def get_stats(self) -> Dict[str, Any]:
        """Get current curiosity statistics.

        Returns
        -------
        dict
        """
        visited_mask = self._visit_counts > 0
        n_visited = int(np.sum(visited_mask))

        stats: Dict[str, Any] = {
            "generation": self._generation,
            "total_evaluations": self._total_evaluations,
            "global_curiosity_scale": self._global_curiosity_scale,
            "n_visited_cells": n_visited,
            "n_unvisited_cells": self.n_cells - n_visited,
            "exploration_ratio": self.exploration_ratio(),
            "temperature": self.config.temperature,
        }

        if n_visited > 0:
            visited_counts = self._visit_counts[visited_mask]
            stats["visit_count_mean"] = float(np.mean(visited_counts))
            stats["visit_count_std"] = float(np.std(visited_counts))
            stats["visit_count_max"] = int(np.max(visited_counts))

            ema_vals = self._quality_ema[visited_mask]
            stats["quality_ema_mean"] = float(np.mean(ema_vals))
            stats["quality_ema_std"] = float(np.std(ema_vals))

        return stats

    def get_curiosity_history(self) -> List[Dict[str, float]]:
        """Return the full curiosity history.

        Returns
        -------
        list of dict
        """
        return list(self._curiosity_history)

    def get_novelty_trend(self) -> np.ndarray:
        """Return the novelty trend over generations.

        Returns
        -------
        np.ndarray
        """
        return np.array(self._novelty_history, dtype=np.float64)

    def get_surprise_trend(self) -> np.ndarray:
        """Return the surprise trend over generations.

        Returns
        -------
        np.ndarray
        """
        return np.array(self._surprise_history, dtype=np.float64)

    def most_curious_cells(self, n: int = 10) -> np.ndarray:
        """Return indices of the n most curious cells.

        Parameters
        ----------
        n : int

        Returns
        -------
        np.ndarray
            Cell indices sorted by descending curiosity.
        """
        curiosities = np.array([
            self.peek_curiosity(i) for i in range(self.n_cells)
        ])
        order = np.argsort(curiosities)[::-1]
        return order[:n]

    def least_curious_cells(self, n: int = 10) -> np.ndarray:
        """Return indices of the n least curious cells.

        Parameters
        ----------
        n : int

        Returns
        -------
        np.ndarray
        """
        curiosities = np.array([
            self.peek_curiosity(i) for i in range(self.n_cells)
        ])
        order = np.argsort(curiosities)
        return order[:n]

    # ----- Serialization -----

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the curiosity computer state.

        Returns
        -------
        dict
        """
        return {
            "n_cells": self.n_cells,
            "config": {
                "mu": self.config.mu,
                "ema_alpha": self.config.ema_alpha,
                "novelty_decay": self.config.novelty_decay,
                "min_curiosity": self.config.min_curiosity,
                "max_curiosity": self.config.max_curiosity,
                "exploration_bonus": self.config.exploration_bonus,
                "temperature": self.config.temperature,
                "curiosity_decay_rate": self.config.curiosity_decay_rate,
            },
            "visit_counts": self._visit_counts.tolist(),
            "quality_ema": self._quality_ema.tolist(),
            "quality_ema_initialized": self._quality_ema_initialized.tolist(),
            "total_evaluations": self._total_evaluations,
            "generation": self._generation,
            "global_curiosity_scale": self._global_curiosity_scale,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CuriosityComputer":
        """Deserialize from dictionary.

        Parameters
        ----------
        d : dict

        Returns
        -------
        CuriosityComputer
        """
        config_dict = d.get("config", {})
        config = CuriosityConfig(**config_dict)
        cc = cls(n_cells=d["n_cells"], config=config)
        cc._visit_counts = np.array(d["visit_counts"], dtype=np.int64)
        cc._quality_ema = np.array(d["quality_ema"], dtype=np.float64)
        cc._quality_ema_initialized = np.array(
            d["quality_ema_initialized"], dtype=bool
        )
        cc._total_evaluations = d.get("total_evaluations", 0)
        cc._generation = d.get("generation", 0)
        cc._global_curiosity_scale = d.get("global_curiosity_scale", 1.0)
        return cc

    def reset(self) -> None:
        """Reset all state to initial values."""
        self._visit_counts = np.zeros(self.n_cells, dtype=np.int64)
        self._quality_ema = np.zeros(self.n_cells, dtype=np.float64)
        self._quality_ema_initialized = np.zeros(self.n_cells, dtype=bool)
        self._total_evaluations = 0
        self._generation = 0
        self._global_curiosity_scale = 1.0
        self._curiosity_history.clear()
        self._novelty_history.clear()
        self._surprise_history.clear()

    def __repr__(self) -> str:
        return (
            f"CuriosityComputer(cells={self.n_cells}, gen={self._generation}, "
            f"evals={self._total_evaluations}, "
            f"explore={self.exploration_ratio():.2f})"
        )

"""Adaptive parameter control for MAP-Elites causal discovery.

Provides controllers that automatically tune mutation/crossover rates,
archive resolution, selection temperature, and batch size during the
evolutionary search based on performance feedback.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from causal_qd.types import QualityScore


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class ParameterController(ABC):
    """Abstract base for adaptive parameter controllers."""

    @abstractmethod
    def step(self, feedback: Dict[str, Any]) -> None:
        """Update internal state based on feedback from the last iteration.

        Parameters
        ----------
        feedback : dict
            Dictionary with performance metrics such as 'n_improvements',
            'archive_size', 'qd_score', 'mean_quality', etc.
        """

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Return the current adapted parameter values.

        Returns
        -------
        Dict[str, Any]
            Parameter name → current value.
        """

    def reset(self) -> None:
        """Reset to initial state."""


# ---------------------------------------------------------------------------
# AdaptiveRateController
# ---------------------------------------------------------------------------


@dataclass
class _OperatorStats:
    """Per-operator running statistics."""
    trials: int = 0
    successes: int = 0
    reward_sum: float = 0.0
    history: List[bool] = field(default_factory=list)


class AdaptiveRateController(ParameterController):
    """Adapt mutation and crossover rates based on operator success.

    Tracks the success rate of each operator using a sliding window
    and updates selection probabilities via a multi-armed bandit
    (UCB1 or softmax).

    Parameters
    ----------
    operator_names : List[str]
        Names of the operators being adapted.
    initial_rates : List[float] | None
        Starting probabilities for each operator.  Normalized internally.
    window_size : int
        Size of the sliding window for success tracking.  Default ``100``.
    adaptation_method : str
        ``"ucb1"`` or ``"softmax"``.  Default ``"softmax"``.
    temperature : float
        Softmax temperature (only used if method is ``"softmax"``).
        Default ``0.5``.
    exploration_constant : float
        UCB1 exploration constant (only used if method is ``"ucb1"``).
        Default ``1.5``.
    min_rate : float
        Floor probability for any operator.  Default ``0.05``.
    """

    def __init__(
        self,
        operator_names: List[str],
        initial_rates: Optional[List[float]] = None,
        window_size: int = 100,
        adaptation_method: str = "softmax",
        temperature: float = 0.5,
        exploration_constant: float = 1.5,
        min_rate: float = 0.05,
    ) -> None:
        self._names = list(operator_names)
        n = len(operator_names)

        if initial_rates is None:
            self._rates = np.ones(n, dtype=np.float64) / n
        else:
            r = np.array(initial_rates, dtype=np.float64)
            self._rates = r / r.sum()

        self._initial_rates = self._rates.copy()
        self._window = window_size
        self._method = adaptation_method
        self._temp = temperature
        self._c = exploration_constant
        self._min_rate = min_rate

        self._stats = {name: _OperatorStats() for name in operator_names}
        self._total_trials = 0
        self._history: deque[Tuple[str, bool]] = deque(maxlen=window_size)

    @property
    def rates(self) -> Dict[str, float]:
        """Current operator rates as a dict."""
        return dict(zip(self._names, self._rates.tolist()))

    def record_outcome(self, operator_name: str, success: bool) -> None:
        """Record an operator application outcome.

        Parameters
        ----------
        operator_name : str
            Which operator was applied.
        success : bool
            Whether the result was accepted into the archive.
        """
        if operator_name not in self._stats:
            return

        stats = self._stats[operator_name]
        stats.trials += 1
        if success:
            stats.successes += 1
        stats.history.append(success)
        if len(stats.history) > self._window:
            old = stats.history.pop(0)
            stats.trials -= 1
            if old:
                stats.successes -= 1

        self._total_trials += 1
        self._history.append((operator_name, success))

    def step(self, feedback: Dict[str, Any]) -> None:
        """Recompute operator rates based on accumulated statistics.

        Parameters
        ----------
        feedback : dict
            Ignored in this controller (rates update from record_outcome).
        """
        if self._total_trials == 0:
            return

        n = len(self._names)
        if self._method == "ucb1":
            self._update_ucb1()
        else:
            self._update_softmax()

        # Enforce minimum rate
        self._rates = np.maximum(self._rates, self._min_rate)
        self._rates /= self._rates.sum()

    def get_params(self) -> Dict[str, Any]:
        """Return current rates.

        Returns
        -------
        Dict[str, Any]
            ``{operator_name: probability}`` for each operator.
        """
        return {"operator_rates": self.rates}

    def reset(self) -> None:
        """Reset to initial rates."""
        self._rates = self._initial_rates.copy()
        self._stats = {name: _OperatorStats() for name in self._names}
        self._total_trials = 0
        self._history.clear()

    def _update_softmax(self) -> None:
        """Update rates using softmax over success rates."""
        n = len(self._names)
        success_rates = np.zeros(n, dtype=np.float64)
        for i, name in enumerate(self._names):
            s = self._stats[name]
            if s.trials > 0:
                success_rates[i] = s.successes / s.trials
            else:
                success_rates[i] = 0.5  # Optimistic prior

        logits = success_rates / max(self._temp, 1e-10)
        logits -= logits.max()
        exp_logits = np.exp(logits)
        self._rates = exp_logits / exp_logits.sum()

    def _update_ucb1(self) -> None:
        """Update rates using UCB1 scores, then normalize."""
        n = len(self._names)
        scores = np.zeros(n, dtype=np.float64)
        for i, name in enumerate(self._names):
            s = self._stats[name]
            if s.trials == 0:
                scores[i] = float("inf")
            else:
                exploit = s.successes / s.trials
                explore = self._c * math.sqrt(
                    math.log(self._total_trials) / s.trials
                )
                scores[i] = exploit + explore

        # Convert scores to probabilities via softmax
        finite_scores = scores.copy()
        inf_mask = np.isinf(finite_scores)
        if np.any(inf_mask):
            finite_scores[inf_mask] = 0
            self._rates = inf_mask.astype(np.float64)
            self._rates /= self._rates.sum()
        else:
            finite_scores -= finite_scores.max()
            exp_s = np.exp(finite_scores)
            self._rates = exp_s / exp_s.sum()


# ---------------------------------------------------------------------------
# ArchiveScheduler
# ---------------------------------------------------------------------------


class ArchiveScheduler(ParameterController):
    """Adjust archive resolution over time.

    Starts with a coarse grid and refines as the archive fills.
    Provides the current target resolution dimensions and triggers
    CVT centroid recomputation when resolution changes.

    Parameters
    ----------
    initial_dims : Tuple[int, ...]
        Starting grid dimensions.  Default ``(5, 5)``.
    max_dims : Tuple[int, ...]
        Maximum grid dimensions.  Default ``(50, 50)``.
    coverage_threshold : float
        Coverage level at which the grid is refined.  Default ``0.7``.
    refinement_factor : float
        Multiplicative factor for each refinement step.  Default ``1.5``.
    descriptor_dim : int
        Number of descriptor dimensions.  Default ``2``.
    """

    def __init__(
        self,
        initial_dims: Tuple[int, ...] = (5, 5),
        max_dims: Tuple[int, ...] = (50, 50),
        coverage_threshold: float = 0.7,
        refinement_factor: float = 1.5,
        descriptor_dim: int = 2,
    ) -> None:
        self._current_dims = list(initial_dims)
        self._initial_dims = list(initial_dims)
        self._max_dims = list(max_dims)
        self._threshold = coverage_threshold
        self._factor = refinement_factor
        self._desc_dim = descriptor_dim
        self._refinement_count = 0
        self._last_coverage = 0.0
        self._needs_rebuild = False

    @property
    def current_dims(self) -> Tuple[int, ...]:
        """Current archive grid dimensions."""
        return tuple(self._current_dims)

    @property
    def needs_rebuild(self) -> bool:
        """Whether the archive needs to be rebuilt at new resolution."""
        return self._needs_rebuild

    def acknowledge_rebuild(self) -> None:
        """Mark that the archive has been rebuilt."""
        self._needs_rebuild = False

    def step(self, feedback: Dict[str, Any]) -> None:
        """Check coverage and refine if threshold is exceeded.

        Parameters
        ----------
        feedback : dict
            Must contain 'coverage' (float in [0, 1]).
        """
        coverage = feedback.get("coverage", 0.0)
        self._last_coverage = coverage

        if coverage >= self._threshold:
            new_dims = []
            refined = False
            for cur, mx in zip(self._current_dims, self._max_dims):
                new = min(int(cur * self._factor), mx)
                if new > cur:
                    refined = True
                new_dims.append(new)

            if refined:
                self._current_dims = new_dims
                self._refinement_count += 1
                self._needs_rebuild = True

    def get_params(self) -> Dict[str, Any]:
        """Return current resolution and refinement state.

        Returns
        -------
        Dict[str, Any]
        """
        return {
            "archive_dims": tuple(self._current_dims),
            "refinement_count": self._refinement_count,
            "needs_rebuild": self._needs_rebuild,
            "last_coverage": self._last_coverage,
        }

    def reset(self) -> None:
        """Reset to initial resolution."""
        self._current_dims = list(self._initial_dims)
        self._refinement_count = 0
        self._last_coverage = 0.0
        self._needs_rebuild = False

    def compute_cvt_centroids(
        self,
        n_centroids: int,
        bounds_low: npt.NDArray[np.float64],
        bounds_high: npt.NDArray[np.float64],
        rng: np.random.Generator,
        n_samples: int = 10000,
        n_iterations: int = 50,
    ) -> npt.NDArray[np.float64]:
        """Compute CVT centroids using Lloyd's algorithm.

        Generates a Centroidal Voronoi Tessellation by iteratively
        assigning random samples to nearest centroids and moving
        centroids to cluster means.

        Parameters
        ----------
        n_centroids : int
            Number of centroids to compute.
        bounds_low, bounds_high : ndarray
            Per-dimension bounds of the descriptor space.
        rng : numpy.random.Generator
            Random state.
        n_samples : int
            Number of random samples per iteration.  Default ``10000``.
        n_iterations : int
            Number of Lloyd iterations.  Default ``50``.

        Returns
        -------
        ndarray, shape (n_centroids, d)
            Centroid positions.
        """
        d = len(bounds_low)

        # Initialize centroids uniformly
        centroids = rng.uniform(
            bounds_low, bounds_high, size=(n_centroids, d)
        )

        for _ in range(n_iterations):
            # Generate random samples
            samples = rng.uniform(
                bounds_low, bounds_high, size=(n_samples, d)
            )

            # Assign samples to nearest centroid
            dists = np.linalg.norm(
                samples[:, np.newaxis, :] - centroids[np.newaxis, :, :],
                axis=2,
            )
            assignments = np.argmin(dists, axis=1)

            # Move centroids to mean of assigned samples
            new_centroids = centroids.copy()
            for k in range(n_centroids):
                mask = assignments == k
                if mask.sum() > 0:
                    new_centroids[k] = samples[mask].mean(axis=0)

            # Check convergence
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift < 1e-6:
                break

        return centroids


# ---------------------------------------------------------------------------
# TemperatureScheduler
# ---------------------------------------------------------------------------


class TemperatureScheduler(ParameterController):
    """Anneal selection temperature over generations.

    Supports linear, exponential, and cosine annealing schedules.

    Parameters
    ----------
    initial_temperature : float
        Starting temperature.  Default ``2.0``.
    final_temperature : float
        End temperature.  Default ``0.1``.
    total_steps : int
        Number of steps over which to anneal.  Default ``1000``.
    schedule : str
        Annealing schedule: ``"linear"``, ``"exponential"``, or
        ``"cosine"``.  Default ``"exponential"``.
    """

    def __init__(
        self,
        initial_temperature: float = 2.0,
        final_temperature: float = 0.1,
        total_steps: int = 1000,
        schedule: str = "exponential",
    ) -> None:
        self._t0 = initial_temperature
        self._t_final = final_temperature
        self._total = total_steps
        self._schedule = schedule
        self._current_step = 0
        self._current_temp = initial_temperature

    @property
    def temperature(self) -> float:
        """Current temperature."""
        return self._current_temp

    def step(self, feedback: Dict[str, Any]) -> None:
        """Advance one step in the annealing schedule.

        Parameters
        ----------
        feedback : dict
            Unused.
        """
        self._current_step += 1
        t = min(self._current_step / max(self._total, 1), 1.0)

        if self._schedule == "linear":
            self._current_temp = self._t0 + (self._t_final - self._t0) * t
        elif self._schedule == "exponential":
            if self._t0 > 0 and self._t_final > 0:
                log_ratio = math.log(self._t_final / self._t0)
                self._current_temp = self._t0 * math.exp(log_ratio * t)
            else:
                self._current_temp = self._t0 + (self._t_final - self._t0) * t
        elif self._schedule == "cosine":
            self._current_temp = (
                self._t_final
                + 0.5 * (self._t0 - self._t_final) * (1 + math.cos(math.pi * t))
            )
        else:
            self._current_temp = self._t0 + (self._t_final - self._t0) * t

    def get_params(self) -> Dict[str, Any]:
        """Return current temperature.

        Returns
        -------
        Dict[str, Any]
        """
        return {
            "temperature": self._current_temp,
            "step": self._current_step,
            "progress": min(self._current_step / max(self._total, 1), 1.0),
        }

    def reset(self) -> None:
        """Reset to initial temperature."""
        self._current_step = 0
        self._current_temp = self._t0


# ---------------------------------------------------------------------------
# PopulationScheduler
# ---------------------------------------------------------------------------


class PopulationScheduler(ParameterController):
    """Adjust batch size over generations.

    Starts with a small batch size and increases it as the search
    progresses, allowing more thorough exploration in later generations
    when the archive is more populated.

    Parameters
    ----------
    initial_batch : int
        Starting batch size.  Default ``20``.
    max_batch : int
        Maximum batch size.  Default ``200``.
    growth_rate : float
        Multiplicative growth per step.  Default ``1.01``.
    plateau_threshold : int
        Number of non-improving steps before increasing batch.
        Default ``10``.
    """

    def __init__(
        self,
        initial_batch: int = 20,
        max_batch: int = 200,
        growth_rate: float = 1.01,
        plateau_threshold: int = 10,
    ) -> None:
        self._batch = float(initial_batch)
        self._initial = initial_batch
        self._max = max_batch
        self._growth = growth_rate
        self._plateau_thresh = plateau_threshold
        self._plateau_count = 0
        self._last_qd = 0.0

    @property
    def batch_size(self) -> int:
        """Current batch size (integer)."""
        return int(self._batch)

    def step(self, feedback: Dict[str, Any]) -> None:
        """Adjust batch size based on improvement feedback.

        Parameters
        ----------
        feedback : dict
            Should contain 'qd_score' and/or 'n_improvements'.
        """
        qd = feedback.get("qd_score", self._last_qd)
        n_imp = feedback.get("n_improvements", 0)

        if n_imp == 0:
            self._plateau_count += 1
        else:
            self._plateau_count = 0

        # Grow batch on plateau
        if self._plateau_count >= self._plateau_thresh:
            self._batch = min(self._batch * 1.5, self._max)
            self._plateau_count = 0
        else:
            # Gradual growth
            self._batch = min(self._batch * self._growth, self._max)

        self._last_qd = qd

    def get_params(self) -> Dict[str, Any]:
        """Return current batch size.

        Returns
        -------
        Dict[str, Any]
        """
        return {
            "batch_size": self.batch_size,
            "plateau_count": self._plateau_count,
        }

    def reset(self) -> None:
        """Reset to initial batch size."""
        self._batch = float(self._initial)
        self._plateau_count = 0
        self._last_qd = 0.0


# ---------------------------------------------------------------------------
# AdaptiveController: combine all controllers
# ---------------------------------------------------------------------------


class AdaptiveController:
    """Composite controller that manages multiple parameter controllers.

    Provides a single interface to step all controllers and retrieve
    all adapted parameters.

    Parameters
    ----------
    controllers : Dict[str, ParameterController]
        Named controllers.  Keys are used as namespaces in the
        combined parameter dict.
    """

    def __init__(
        self, controllers: Optional[Dict[str, ParameterController]] = None
    ) -> None:
        self._controllers = dict(controllers) if controllers else {}

    def add(self, name: str, controller: ParameterController) -> None:
        """Register a named controller.

        Parameters
        ----------
        name : str
            Namespace key.
        controller : ParameterController
            Controller instance.
        """
        self._controllers[name] = controller

    def step(self, feedback: Dict[str, Any]) -> None:
        """Step all controllers.

        Parameters
        ----------
        feedback : dict
            Performance metrics forwarded to each controller.
        """
        for ctrl in self._controllers.values():
            ctrl.step(feedback)

    def get_params(self) -> Dict[str, Any]:
        """Collect parameters from all controllers.

        Returns
        -------
        Dict[str, Any]
            Flat dict with controller name prefixes.
        """
        result: Dict[str, Any] = {}
        for name, ctrl in self._controllers.items():
            params = ctrl.get_params()
            for k, v in params.items():
                result[f"{name}.{k}"] = v
        return result

    def reset(self) -> None:
        """Reset all controllers."""
        for ctrl in self._controllers.values():
            ctrl.reset()

    def __getitem__(self, name: str) -> ParameterController:
        return self._controllers[name]

    def __contains__(self, name: str) -> bool:
        return name in self._controllers

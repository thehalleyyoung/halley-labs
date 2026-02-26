"""
Parallel ensemble simulation management.

Provides EnsembleSimulator for running multiple SSA/hybrid trajectories
in parallel with statistics accumulation, convergence detection, and
trajectory storage.
"""

from __future__ import annotations

import multiprocessing
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .ssa import (
    DirectMethod,
    NextReactionMethod,
    Reaction,
    StochasticState,
    TrajectoryRecorder,
)


def _run_single_trajectory(args: Tuple) -> Dict[str, Any]:
    """Worker function for parallel trajectory execution.

    Must be a module-level function for pickling by multiprocessing.
    """
    (
        reactions_data,
        num_species,
        initial_state,
        t_end,
        sample_times,
        seed,
        method,
    ) = args

    # Reconstruct reactions from serializable data
    reactions = []
    for rd in reactions_data:
        reactions.append(
            Reaction(
                name=rd["name"],
                reactants=rd["reactants"],
                products=rd["products"],
                rate_constant=rd["rate_constant"],
            )
        )

    if method == "next_reaction":
        sim = NextReactionMethod(reactions, num_species, seed=seed)
    else:
        sim = DirectMethod(reactions, num_species, seed=seed)

    rec = sim.simulate_trajectory(
        initial_state,
        t_end,
        record_mode="full",
    )
    trajectory = rec.interpolate_at(sample_times)
    return {
        "trajectory": trajectory,
        "event_count": rec.event_count,
    }


def _serialize_reactions(reactions: List[Reaction]) -> List[Dict]:
    """Serialize reactions to pickle-safe format."""
    data = []
    for rxn in reactions:
        data.append({
            "name": rxn.name,
            "reactants": dict(rxn.reactants),
            "products": dict(rxn.products),
            "rate_constant": rxn.rate_constant,
        })
    return data


@dataclass
class EnsembleStatistics:
    """Accumulated statistics from ensemble simulations.

    Attributes:
        times: Sample time points.
        mean: Mean trajectory [n_times x n_species].
        variance: Variance [n_times x n_species].
        std: Standard deviation [n_times x n_species].
        quantiles: Dict mapping quantile value to array [n_times x n_species].
        histograms: Per-species histograms at selected times.
        num_runs: Number of completed runs.
        total_events: Total stochastic events across all runs.
    """

    times: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    std: np.ndarray
    quantiles: Dict[float, np.ndarray]
    num_runs: int = 0
    total_events: int = 0

    @staticmethod
    def compute(
        trajectories: np.ndarray,
        times: np.ndarray,
        quantile_levels: Optional[List[float]] = None,
    ) -> "EnsembleStatistics":
        """Compute statistics from trajectory array [n_runs x n_times x n_species]."""
        if quantile_levels is None:
            quantile_levels = [0.025, 0.25, 0.5, 0.75, 0.975]

        mean = np.mean(trajectories, axis=0)
        variance = np.var(trajectories, axis=0)
        std = np.std(trajectories, axis=0)
        quantiles = {}
        for q in quantile_levels:
            quantiles[q] = np.quantile(trajectories, q, axis=0)

        return EnsembleStatistics(
            times=times,
            mean=mean,
            variance=variance,
            std=std,
            quantiles=quantiles,
            num_runs=trajectories.shape[0],
        )

    def confidence_interval(
        self, level: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute confidence interval for the mean.

        Returns (lower, upper) arrays [n_times x n_species].
        Uses normal approximation: mean +/- z * std / sqrt(n).
        """
        from scipy import stats

        z = stats.norm.ppf(1 - (1 - level) / 2)
        se = self.std / np.sqrt(max(self.num_runs, 1))
        return self.mean - z * se, self.mean + z * se

    def histogram_at_time(
        self,
        trajectories: np.ndarray,
        time_idx: int,
        species_idx: int,
        bins: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute histogram of species values at a specific time point."""
        values = trajectories[:, time_idx, species_idx]
        counts, bin_edges = np.histogram(values, bins=bins)
        return counts, bin_edges


class ConvergenceDetector:
    """Detects when ensemble statistics have converged.

    Monitors the running mean and variance and checks if their
    relative change falls below a threshold.
    """

    def __init__(
        self,
        atol: float = 1.0,
        rtol: float = 0.01,
        window: int = 20,
        min_runs: int = 50,
    ):
        self.atol = atol
        self.rtol = rtol
        self.window = window
        self.min_runs = min_runs
        self._mean_history: List[np.ndarray] = []
        self._var_history: List[np.ndarray] = []

    def update(self, running_mean: np.ndarray, running_var: np.ndarray):
        """Add a new snapshot of running statistics."""
        self._mean_history.append(running_mean.copy())
        self._var_history.append(running_var.copy())

    def is_converged(self) -> bool:
        """Check if statistics have stabilized."""
        n = len(self._mean_history)
        if n < max(self.window, self.min_runs):
            return False

        recent_means = np.array(self._mean_history[-self.window:])
        recent_vars = np.array(self._var_history[-self.window:])

        # Check if mean has stabilized
        mean_spread = np.max(recent_means, axis=0) - np.min(recent_means, axis=0)
        mean_avg = np.mean(np.abs(recent_means), axis=0)
        mean_avg_safe = np.where(mean_avg > self.atol, mean_avg, 1.0)
        mean_converged = np.all(mean_spread / mean_avg_safe < self.rtol)

        # Check if variance has stabilized
        var_spread = np.max(recent_vars, axis=0) - np.min(recent_vars, axis=0)
        var_avg = np.mean(np.abs(recent_vars), axis=0)
        var_avg_safe = np.where(var_avg > self.atol, var_avg, 1.0)
        var_converged = np.all(var_spread / var_avg_safe < self.rtol)

        return bool(mean_converged and var_converged)

    def reset(self):
        self._mean_history.clear()
        self._var_history.clear()


class TrajectoryStore:
    """Stores and retrieves ensemble trajectories.

    Supports in-memory storage with optional disk-backed mode
    for large ensembles.
    """

    def __init__(
        self,
        num_species: int,
        num_times: int,
        max_memory_runs: int = 10000,
        disk_path: Optional[str] = None,
    ):
        self.num_species = num_species
        self.num_times = num_times
        self.max_memory_runs = max_memory_runs
        self.disk_path = disk_path
        self._trajectories: List[np.ndarray] = []
        self._disk_count = 0

    @property
    def num_runs(self) -> int:
        return len(self._trajectories) + self._disk_count

    def add(self, trajectory: np.ndarray):
        """Add a trajectory [n_times x n_species]."""
        if len(self._trajectories) < self.max_memory_runs:
            self._trajectories.append(trajectory)
        elif self.disk_path is not None:
            self._flush_to_disk(trajectory)
        else:
            # Overwrite oldest
            idx = self._disk_count % self.max_memory_runs
            self._trajectories[idx] = trajectory
            self._disk_count += 1

    def _flush_to_disk(self, trajectory: np.ndarray):
        """Save trajectory to disk."""
        if self.disk_path is None:
            return
        os.makedirs(self.disk_path, exist_ok=True)
        path = os.path.join(self.disk_path, f"traj_{self._disk_count}.npy")
        np.save(path, trajectory)
        self._disk_count += 1

    def get_all(self) -> np.ndarray:
        """Return all in-memory trajectories as array [n_runs x n_times x n_species]."""
        if not self._trajectories:
            return np.empty((0, self.num_times, self.num_species))
        return np.array(self._trajectories)

    def get_trajectory(self, idx: int) -> np.ndarray:
        """Get a single trajectory by index."""
        if idx < len(self._trajectories):
            return self._trajectories[idx]
        # Try disk
        if self.disk_path is not None:
            path = os.path.join(
                self.disk_path,
                f"traj_{idx - len(self._trajectories)}.npy",
            )
            if os.path.exists(path):
                return np.load(path)
        raise IndexError(f"Trajectory {idx} not found")

    def running_statistics(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute running mean and variance from in-memory trajectories."""
        if not self._trajectories:
            return np.zeros(self.num_species), np.zeros(self.num_species)
        arr = np.array(self._trajectories)
        # Use last time point for convergence checking
        final_states = arr[:, -1, :]
        return np.mean(final_states, axis=0), np.var(final_states, axis=0)

    def clear(self):
        self._trajectories.clear()
        self._disk_count = 0


class EnsembleSimulator:
    """Parallel ensemble simulator with convergence detection.

    Runs multiple SSA/hybrid trajectories using multiprocessing and
    accumulates statistics. Can stop early when convergence is detected.

    Parameters:
        reactions: List of Reaction objects.
        num_species: Number of species.
        method: 'direct' or 'next_reaction'.
        seed: Base random seed.
        n_workers: Number of parallel workers (0 = serial).
        convergence_detector: Optional convergence detector.
    """

    def __init__(
        self,
        reactions: List[Reaction],
        num_species: int,
        method: str = "direct",
        seed: Optional[int] = None,
        n_workers: int = 0,
        convergence_detector: Optional[ConvergenceDetector] = None,
    ):
        self.reactions = reactions
        self.num_species = num_species
        self.method = method
        self.seed = seed
        self.n_workers = n_workers
        self.convergence_detector = convergence_detector
        self._reactions_data = _serialize_reactions(reactions)

    def run(
        self,
        initial_state: np.ndarray,
        t_end: float,
        num_runs: int = 100,
        sample_times: Optional[np.ndarray] = None,
        sample_interval: float = 1.0,
        batch_size: int = 10,
        store: Optional[TrajectoryStore] = None,
    ) -> EnsembleStatistics:
        """Run ensemble simulation.

        Parameters:
            initial_state: Initial copy numbers.
            t_end: End time.
            num_runs: Maximum number of trajectories.
            sample_times: Times at which to sample (overrides sample_interval).
            sample_interval: Uniform sampling interval.
            batch_size: Trajectories per batch (for convergence checking).
            store: Optional TrajectoryStore for trajectory persistence.

        Returns:
            EnsembleStatistics with accumulated results.
        """
        if sample_times is None:
            sample_times = np.arange(0, t_end + sample_interval, sample_interval)

        n_times = len(sample_times)
        if store is None:
            store = TrajectoryStore(self.num_species, n_times)

        completed = 0
        total_events = 0

        while completed < num_runs:
            batch_end = min(completed + batch_size, num_runs)
            current_batch_size = batch_end - completed

            args_list = []
            for i in range(current_batch_size):
                run_seed = (
                    (self.seed + completed + i) if self.seed is not None else None
                )
                args_list.append((
                    self._reactions_data,
                    self.num_species,
                    initial_state.copy(),
                    t_end,
                    sample_times,
                    run_seed,
                    self.method,
                ))

            # Execute batch
            if self.n_workers > 0 and current_batch_size > 1:
                results = self._run_parallel(args_list)
            else:
                results = self._run_serial(args_list)

            for result in results:
                store.add(result["trajectory"])
                total_events += result["event_count"]
                completed += 1

            # Check convergence
            if self.convergence_detector is not None and completed >= batch_size:
                running_mean, running_var = store.running_statistics()
                self.convergence_detector.update(running_mean, running_var)
                if self.convergence_detector.is_converged():
                    break

        trajectories = store.get_all()
        stats = EnsembleStatistics.compute(trajectories, sample_times)
        stats.total_events = total_events
        return stats

    def _run_serial(
        self, args_list: List[Tuple]
    ) -> List[Dict[str, Any]]:
        """Run trajectories serially."""
        return [_run_single_trajectory(args) for args in args_list]

    def _run_parallel(
        self, args_list: List[Tuple]
    ) -> List[Dict[str, Any]]:
        """Run trajectories in parallel using multiprocessing."""
        n_workers = min(self.n_workers, len(args_list))
        try:
            with multiprocessing.Pool(n_workers) as pool:
                results = pool.map(_run_single_trajectory, args_list)
            return results
        except Exception:
            # Fallback to serial on error
            return self._run_serial(args_list)

    def export_visualization_data(
        self,
        stats: EnsembleStatistics,
        species_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Export statistics in a format suitable for visualization.

        Returns dict with arrays for plotting mean +/- CI, quantiles, etc.
        """
        if species_names is None:
            species_names = [f"S{i}" for i in range(self.num_species)]

        ci_low, ci_high = stats.confidence_interval(0.95)

        export = {
            "times": stats.times.tolist(),
            "species_names": species_names,
            "num_runs": stats.num_runs,
            "total_events": stats.total_events,
            "species": {},
        }
        for i, name in enumerate(species_names):
            export["species"][name] = {
                "mean": stats.mean[:, i].tolist(),
                "std": stats.std[:, i].tolist(),
                "ci_lower": ci_low[:, i].tolist(),
                "ci_upper": ci_high[:, i].tolist(),
                "quantiles": {
                    str(q): vals[:, i].tolist()
                    for q, vals in stats.quantiles.items()
                },
            }
        return export

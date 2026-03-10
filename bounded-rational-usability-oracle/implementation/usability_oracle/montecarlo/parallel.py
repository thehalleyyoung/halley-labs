"""
usability_oracle.montecarlo.parallel — Parallel Monte Carlo execution engine.

Uses ``multiprocessing.Pool`` for CPU parallelism with proper per-worker
seed management via ``numpy.random.SeedSequence`` to ensure reproducibility
and statistical independence across workers.
"""

from __future__ import annotations

import multiprocessing
import time
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from usability_oracle.montecarlo.types import (
    ImportanceWeight,
    MCConfig,
    SampleStatistics,
    SamplingStrategy,
    TerminationReason,
    TrajectoryBundle,
    VarianceEstimate,
)


# ═══════════════════════════════════════════════════════════════════════════
# Worker task type
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _WorkerTask:
    """Serialisable task description for a worker process."""
    transition_model: Dict[str, Dict[str, Dict[str, float]]]
    cost_model: Dict[str, Dict[str, float]]
    policy: Dict[str, Dict[str, float]]
    initial_state_distribution: Dict[str, float]
    goal_states: FrozenSet[str]
    num_samples: int
    max_trajectory_length: int
    strategy: SamplingStrategy
    detect_cycles: bool
    confidence_level: float
    seed: int


def _worker_fn(task: _WorkerTask) -> Dict[str, Any]:
    """Worker function executed in a child process.

    Imports the sampler lazily to avoid pickling issues.
    """
    from usability_oracle.montecarlo.sampler import TrajectorySamplerImpl

    config = MCConfig(
        num_samples=task.num_samples,
        max_trajectory_length=task.max_trajectory_length,
        strategy=task.strategy,
        seed=task.seed,
        detect_cycles=task.detect_cycles,
        confidence_level=task.confidence_level,
    )

    sampler = TrajectorySamplerImpl(seed=task.seed)
    bundle = sampler.sample(
        task.transition_model,
        task.cost_model,
        task.policy,
        task.initial_state_distribution,
        task.goal_states,
        config,
    )

    return {
        "costs": list(bundle.costs),
        "lengths": list(bundle.lengths),
        "termination_reasons": [r.value for r in bundle.termination_reasons],
        "importance_weights": (
            [w.to_dict() for w in bundle.importance_weights]
            if bundle.importance_weights is not None
            else None
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# ParallelMCExecutor
# ═══════════════════════════════════════════════════════════════════════════

class ParallelMCExecutor:
    """Parallel Monte Carlo executor using ``multiprocessing.Pool``.

    Implements the
    :class:`~usability_oracle.montecarlo.protocols.ParallelExecutor` protocol.

    Parameters:
        n_workers: Number of worker processes.  Defaults to the number of
            CPU cores.
    """

    def __init__(self, n_workers: Optional[int] = None) -> None:
        self._n_workers = n_workers or max(multiprocessing.cpu_count(), 1)

    @property
    def num_workers(self) -> int:
        """Number of active parallel workers."""
        return self._n_workers

    def execute(
        self,
        sampler: Any,  # TrajectorySampler protocol
        transition_model: Dict[str, Dict[str, Dict[str, float]]],
        cost_model: Dict[str, Dict[str, float]],
        policy: Dict[str, Dict[str, float]],
        initial_state_distribution: Dict[str, float],
        goal_states: FrozenSet[str],
        config: MCConfig,
    ) -> TrajectoryBundle:
        """Execute parallel Monte Carlo sampling.

        Splits the sample budget across workers, generates independent
        seeds via ``SeedSequence.spawn()``, and merges partial results
        using Welford's online algorithm for numerical stability.

        Parameters:
            sampler: Trajectory sampler (used only for its type;
                fresh samplers are created in each worker).
            transition_model: T(s'|s,a).
            cost_model: C(s,a).
            policy: π(a|s).
            initial_state_distribution: P(s₀).
            goal_states: Absorbing goal states.
            config: Sampling configuration.

        Returns:
            Merged :class:`TrajectoryBundle`.
        """
        t0 = time.monotonic()

        n_workers = min(self._n_workers, config.num_samples)
        if n_workers <= 1:
            # Fall back to sequential
            from usability_oracle.montecarlo.sampler import TrajectorySamplerImpl
            seq_sampler = TrajectorySamplerImpl(seed=config.seed)
            return seq_sampler.sample(
                transition_model, cost_model, policy,
                initial_state_distribution, goal_states, config,
            )

        chunks = chunk_trajectories(config.num_samples, n_workers)
        seeds = _generate_worker_seeds(config.seed, n_workers)

        tasks = [
            _WorkerTask(
                transition_model=transition_model,
                cost_model=cost_model,
                policy=policy,
                initial_state_distribution=initial_state_distribution,
                goal_states=goal_states,
                num_samples=chunk_size,
                max_trajectory_length=config.max_trajectory_length,
                strategy=config.strategy,
                detect_cycles=config.detect_cycles,
                confidence_level=config.confidence_level,
                seed=seed,
            )
            for chunk_size, seed in zip(chunks, seeds)
        ]

        with multiprocessing.Pool(processes=n_workers) as pool:
            partial_results = pool.map(_worker_fn, tasks)

        bundle = merge_statistics(partial_results, config)
        elapsed = time.monotonic() - t0

        return TrajectoryBundle(
            num_trajectories=bundle.num_trajectories,
            costs=bundle.costs,
            lengths=bundle.lengths,
            termination_reasons=bundle.termination_reasons,
            importance_weights=bundle.importance_weights,
            statistics=bundle.statistics,
            config=bundle.config,
            wall_clock_seconds=elapsed,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Module-level functions
# ═══════════════════════════════════════════════════════════════════════════

def chunk_trajectories(n_total: int, n_workers: int) -> List[int]:
    """Partition *n_total* trajectories into *n_workers* balanced chunks.

    The first ``n_total % n_workers`` workers each receive one extra
    trajectory, ensuring load balance.

    Parameters:
        n_total: Total number of trajectories.
        n_workers: Number of workers.

    Returns:
        List of chunk sizes, one per worker.

    Example:
        >>> chunk_trajectories(10, 3)
        [4, 3, 3]
    """
    if n_workers <= 0:
        raise ValueError("n_workers must be > 0")
    if n_total <= 0:
        return [0] * n_workers

    base = n_total // n_workers
    remainder = n_total % n_workers
    return [base + (1 if i < remainder else 0) for i in range(n_workers)]


def merge_statistics(
    partial_results: Sequence[Dict[str, Any]],
    config: MCConfig,
) -> TrajectoryBundle:
    """Merge partial worker results into a single TrajectoryBundle.

    Uses Welford's online algorithm for numerically stable combination
    of means and variances from partial results.

    Parameters:
        partial_results: Sequence of worker result dicts with keys
            ``costs``, ``lengths``, ``termination_reasons``,
            and optionally ``importance_weights``.
        config: Original sampling configuration.

    Returns:
        Combined :class:`TrajectoryBundle`.
    """
    all_costs: List[float] = []
    all_lengths: List[int] = []
    all_reasons: List[TerminationReason] = []
    all_weights: Optional[List[ImportanceWeight]] = None

    has_weights = any(
        pr.get("importance_weights") is not None for pr in partial_results
    )
    if has_weights:
        all_weights = []

    for pr in partial_results:
        all_costs.extend(pr["costs"])
        all_lengths.extend(pr["lengths"])
        all_reasons.extend(
            TerminationReason(r) for r in pr["termination_reasons"]
        )
        if has_weights and all_weights is not None:
            iw_data = pr.get("importance_weights")
            if iw_data is not None:
                offset = len(all_weights)
                for w_dict in iw_data:
                    all_weights.append(ImportanceWeight(
                        sample_id=offset + w_dict["sample_id"],
                        raw_weight=w_dict["raw_weight"],
                        log_weight=w_dict["log_weight"],
                        normalised_weight=w_dict["normalised_weight"],
                    ))

    # Re-normalise importance weights across all workers
    if all_weights is not None and all_weights:
        log_ws = np.array([w.log_weight for w in all_weights], dtype=np.float64)
        max_log = np.max(log_ws)
        log_sum = max_log + np.log(np.sum(np.exp(log_ws - max_log)))
        normalised = np.exp(log_ws - log_sum)
        all_weights = [
            ImportanceWeight(
                sample_id=w.sample_id,
                raw_weight=w.raw_weight,
                log_weight=w.log_weight,
                normalised_weight=float(normalised[i]),
            )
            for i, w in enumerate(all_weights)
        ]

    n = len(all_costs)
    stats = _welford_statistics(
        all_costs, all_lengths, all_reasons,
        all_weights, config.confidence_level,
    )

    return TrajectoryBundle(
        num_trajectories=n,
        costs=tuple(all_costs),
        lengths=tuple(all_lengths),
        termination_reasons=tuple(all_reasons),
        importance_weights=tuple(all_weights) if all_weights is not None else None,
        statistics=stats,
        config=config,
        wall_clock_seconds=0.0,
    )


def adaptive_allocation(
    budget: int,
    partial_results: Dict[str, Dict[str, Any]],
) -> Dict[str, int]:
    """Allocate additional samples to high-variance states.

    Given a remaining sample budget and per-state partial results
    (each containing ``"mean"`` and ``"variance"`` keys), allocate
    budget proportional to the standard deviation of each state's
    cost estimate.

    Parameters:
        budget: Number of additional trajectories to allocate.
        partial_results: Mapping state → {"mean": float, "variance": float, "count": int}.

    Returns:
        Mapping state → number of additional samples.
    """
    if not partial_results or budget <= 0:
        return {}

    states = list(partial_results.keys())
    stds = np.array([
        max(partial_results[s].get("variance", 0.0), 0.0) ** 0.5
        for s in states
    ], dtype=np.float64)

    total_std = stds.sum()
    if total_std <= 0:
        # Uniform allocation
        per = max(budget // len(states), 1)
        return {s: per for s in states}

    # Proportional to standard deviation
    alloc = np.maximum(np.round(stds / total_std * budget).astype(int), 1)
    diff = budget - alloc.sum()
    if diff > 0:
        indices = np.argsort(-stds)
        for i in range(diff):
            alloc[indices[i % len(indices)]] += 1
    elif diff < 0:
        indices = np.argsort(stds)
        for i in range(-diff):
            idx = indices[i % len(indices)]
            if alloc[idx] > 1:
                alloc[idx] -= 1

    return {s: int(alloc[i]) for i, s in enumerate(states)}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _generate_worker_seeds(base_seed: Optional[int], n_workers: int) -> List[int]:
    """Generate independent seeds for each worker via SeedSequence."""
    ss = np.random.SeedSequence(base_seed)
    children = ss.spawn(n_workers)
    return [int(child.generate_state(1)[0]) for child in children]


def _welford_statistics(
    costs: List[float],
    lengths: List[int],
    reasons: List[TerminationReason],
    weights: Optional[List[ImportanceWeight]],
    confidence_level: float,
) -> SampleStatistics:
    """Compute statistics using Welford's online algorithm.

    Welford's algorithm maintains a running mean and M₂ accumulator:
        δ  = xᵢ - mean_{i-1}
        mean_i = mean_{i-1} + δ/i
        M₂_i = M₂_{i-1} + δ·(xᵢ - mean_i)
        variance = M₂ / (n-1)      (Bessel correction)

    This avoids catastrophic cancellation for large or skewed data.
    """
    n = len(costs)
    if n == 0:
        zero_ve = VarianceEstimate(
            sample_variance=0.0, standard_error=0.0,
            coefficient_of_variation=0.0,
            effective_sample_size=0.0, ess_ratio=0.0,
        )
        return SampleStatistics(
            mean_cost=0.0, median_cost=0.0, std_cost=0.0,
            min_cost=0.0, max_cost=0.0, percentiles={},
            mean_length=0.0, goal_reach_rate=0.0,
            variance_estimate=zero_ve,
        )

    # Welford's online algorithm
    mean = 0.0
    m2 = 0.0
    for i, x in enumerate(costs, 1):
        delta = x - mean
        mean += delta / i
        delta2 = x - mean
        m2 += delta * delta2

    var = m2 / (n - 1) if n > 1 else 0.0

    if weights is not None:
        w = np.array([iw.normalised_weight for iw in weights], dtype=np.float64)
        arr = np.array(costs, dtype=np.float64)
        mean = float(np.sum(w * arr))
        var = float(np.sum(w * (arr - mean) ** 2))
        w_sq_sum = float(np.sum(w ** 2))
        ess = 1.0 / w_sq_sum if w_sq_sum > 0 else 0.0
    else:
        ess = float(n)

    std = var ** 0.5 if var > 0 else 0.0
    se = std / (n ** 0.5) if n > 0 else 0.0
    cv = std / abs(mean) if abs(mean) > 1e-15 else 0.0
    ess_ratio = ess / n if n > 0 else 0.0

    arr = np.array(costs, dtype=np.float64)
    percentiles = {p: float(np.percentile(arr, p)) for p in [5, 25, 50, 75, 95]}
    goal_count = sum(1 for r in reasons if r == TerminationReason.GOAL_REACHED)

    ve = VarianceEstimate(
        sample_variance=var,
        standard_error=se,
        coefficient_of_variation=cv,
        effective_sample_size=ess,
        ess_ratio=ess_ratio,
    )

    return SampleStatistics(
        mean_cost=mean,
        median_cost=float(np.median(arr)),
        std_cost=std,
        min_cost=float(np.min(arr)),
        max_cost=float(np.max(arr)),
        percentiles=percentiles,
        mean_length=float(np.mean(np.array(lengths, dtype=np.float64))),
        goal_reach_rate=goal_count / n,
        variance_estimate=ve,
    )

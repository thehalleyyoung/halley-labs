"""
Monte Carlo simulation framework for causal robustness studies.

Provides :class:`MonteCarloRunner` for running repeated simulation
experiments with parallel execution, coverage estimation, power
analysis, and bias-variance decomposition.

Classes
-------
- :class:`SimStudyConfig` — Configuration for a simulation study.
- :class:`ReplicateResult` — Result of a single replicate.
- :class:`SimStudyResult` — Aggregated simulation study results.
- :class:`MonteCarloRunner` — Orchestrates the simulation.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray


# ============================================================================
# Configuration & result types
# ============================================================================


@dataclass(slots=True)
class SimStudyConfig:
    """Configuration for a Monte Carlo simulation study.

    Attributes
    ----------
    n_replicates : int
        Number of independent replications.
    sample_sizes : tuple[int, ...]
        Sample sizes to evaluate.
    estimator_fn : Callable
        Function ``(data, adj, treatment, outcome) -> float`` returning
        an ATE estimate.
    dgp_fn : Callable
        Function ``(n_samples, rng) -> (data, adj, true_ate)`` generating
        one replicate.
    true_ate : float
        Ground-truth ATE.
    alpha : float
        Significance level for confidence intervals.
    n_jobs : int
        Number of parallel workers (1 = sequential).
    seed : int
        Master random seed.
    checkpoint_dir : str | None
        Directory for checkpointing partial results.
    checkpoint_every : int
        Checkpoint after every *k* replicates.
    """

    n_replicates: int = 500
    sample_sizes: tuple[int, ...] = (100, 250, 500, 1000, 2000)
    estimator_fn: Callable[..., float] | None = None
    dgp_fn: Callable[..., tuple[NDArray, NDArray, float]] | None = None
    true_ate: float = 1.0
    alpha: float = 0.05
    n_jobs: int = 1
    seed: int = 42
    checkpoint_dir: str | None = None
    checkpoint_every: int = 50


@dataclass(frozen=True, slots=True)
class ReplicateResult:
    """Result of a single simulation replicate.

    Attributes
    ----------
    sample_size : int
        Sample size used.
    estimate : float
        Point estimate of the ATE.
    se : float
        Standard error of the estimate.
    ci_lower : float
        Lower confidence bound.
    ci_upper : float
        Upper confidence bound.
    covers_truth : bool
        Whether the CI covers the true ATE.
    true_ate : float
        True ATE.
    elapsed_s : float
        Wall-clock time in seconds.
    seed_used : int
        Seed used for this replicate.
    """

    sample_size: int
    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    covers_truth: bool
    true_ate: float
    elapsed_s: float = 0.0
    seed_used: int = 0


@dataclass(slots=True)
class SampleSizeSummary:
    """Aggregated statistics for one sample size.

    Attributes
    ----------
    sample_size : int
        Sample size.
    mean_estimate : float
        Mean ATE estimate across replicates.
    bias : float
        Bias = mean_estimate - true_ate.
    variance : float
        Variance of the estimates.
    mse : float
        Mean squared error = bias² + variance.
    rmse : float
        Root mean squared error.
    coverage : float
        Empirical coverage probability.
    coverage_se : float
        Standard error of coverage estimate.
    mean_ci_width : float
        Mean CI width.
    power : float
        Rejection rate (power) at the given alpha.
    power_se : float
        Standard error of the power estimate.
    n_replicates : int
        Number of replicates.
    """

    sample_size: int = 0
    mean_estimate: float = 0.0
    bias: float = 0.0
    variance: float = 0.0
    mse: float = 0.0
    rmse: float = 0.0
    coverage: float = 0.0
    coverage_se: float = 0.0
    mean_ci_width: float = 0.0
    power: float = 0.0
    power_se: float = 0.0
    n_replicates: int = 0


@dataclass(slots=True)
class SimStudyResult:
    """Aggregated Monte Carlo simulation study results.

    Attributes
    ----------
    config : SimStudyConfig
        Study configuration.
    summaries : list[SampleSizeSummary]
        Per-sample-size aggregations.
    all_results : list[ReplicateResult]
        All individual replicate results.
    total_time_s : float
        Total wall-clock time.
    """

    config: SimStudyConfig
    summaries: list[SampleSizeSummary] = field(default_factory=list)
    all_results: list[ReplicateResult] = field(default_factory=list)
    total_time_s: float = 0.0

    def to_dataframe(self) -> pd.DataFrame:
        """Convert summaries to a DataFrame."""
        rows = []
        for s in self.summaries:
            rows.append({
                "sample_size": s.sample_size,
                "mean_estimate": s.mean_estimate,
                "bias": s.bias,
                "variance": s.variance,
                "mse": s.mse,
                "rmse": s.rmse,
                "coverage": s.coverage,
                "coverage_se": s.coverage_se,
                "mean_ci_width": s.mean_ci_width,
                "power": s.power,
                "power_se": s.power_se,
                "n_replicates": s.n_replicates,
            })
        return pd.DataFrame(rows)

    @property
    def power_curve(self) -> tuple[tuple[int, ...], tuple[float, ...]]:
        """Return ``(sample_sizes, power_values)``."""
        sizes = tuple(s.sample_size for s in self.summaries)
        powers = tuple(s.power for s in self.summaries)
        return sizes, powers

    @property
    def coverage_curve(self) -> tuple[tuple[int, ...], tuple[float, ...]]:
        """Return ``(sample_sizes, coverage_values)``."""
        sizes = tuple(s.sample_size for s in self.summaries)
        covs = tuple(s.coverage for s in self.summaries)
        return sizes, covs


# ============================================================================
# Default estimator / DGP for self-contained usage
# ============================================================================


def _default_estimator(
    data: NDArray[np.float64],
    adj: NDArray[np.int8],
    treatment: int,
    outcome: int,
) -> float:
    """Naive difference-in-means estimator (for illustration)."""
    t = data[:, treatment]
    y = data[:, outcome]
    treated = t > np.median(t)
    if treated.sum() == 0 or (~treated).sum() == 0:
        return 0.0
    return float(y[treated].mean() - y[~treated].mean())


# ============================================================================
# Single replicate worker
# ============================================================================


def _run_single_replicate(
    rep_idx: int,
    sample_size: int,
    dgp_fn: Callable,
    estimator_fn: Callable,
    true_ate: float,
    alpha: float,
    seed: int,
) -> ReplicateResult:
    """Execute one replicate (designed for parallel dispatch)."""
    rng = np.random.default_rng(seed)
    t0 = time.monotonic()

    data, adj, ate = dgp_fn(sample_size, rng)
    true = ate if ate is not None else true_ate

    estimate = estimator_fn(data, adj, 0, 1)

    # Bootstrap SE
    n = data.shape[0]
    n_boot = 200
    boot_ests = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_ests[b] = estimator_fn(data[idx], adj, 0, 1)

    se = float(np.std(boot_ests, ddof=1))
    if se < 1e-12:
        se = 1e-6

    from scipy import stats as sp_stats
    z = sp_stats.norm.ppf(1 - alpha / 2)
    ci_lo = estimate - z * se
    ci_hi = estimate + z * se
    covers = ci_lo <= true <= ci_hi

    elapsed = time.monotonic() - t0

    return ReplicateResult(
        sample_size=sample_size,
        estimate=estimate,
        se=se,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        covers_truth=covers,
        true_ate=true,
        elapsed_s=elapsed,
        seed_used=seed,
    )


# ============================================================================
# MonteCarloRunner
# ============================================================================


class MonteCarloRunner:
    """Orchestrates Monte Carlo simulation studies.

    Parameters
    ----------
    config : SimStudyConfig
        Study configuration.
    progress_callback : Callable[[int, int], None] | None
        Called with ``(completed, total)`` after each replicate.
    """

    def __init__(
        self,
        config: SimStudyConfig,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        self.config = config
        self.progress_callback = progress_callback
        self._rng = np.random.default_rng(config.seed)
        self._checkpoint_path: Path | None = None
        if config.checkpoint_dir:
            self._checkpoint_path = Path(config.checkpoint_dir)
            self._checkpoint_path.mkdir(parents=True, exist_ok=True)

    # -- Main entry point ----------------------------------------------------

    def run(self) -> SimStudyResult:
        """Execute the full simulation study.

        Returns
        -------
        SimStudyResult
        """
        cfg = self.config
        if cfg.dgp_fn is None:
            raise ValueError("dgp_fn must be provided in SimStudyConfig")
        estimator_fn = cfg.estimator_fn or _default_estimator

        t0 = time.monotonic()
        all_results: list[ReplicateResult] = []
        # Load checkpoint if available
        all_results = self._load_checkpoint()
        completed_keys = {
            (r.sample_size, r.seed_used) for r in all_results
        }

        # Build task list
        tasks: list[tuple[int, int, int]] = []  # (rep_idx, sample_size, seed)
        seed_seq = self._rng.integers(0, 2**31, size=(
            len(cfg.sample_sizes) * cfg.n_replicates
        ))
        idx = 0
        for ns in cfg.sample_sizes:
            for rep in range(cfg.n_replicates):
                seed = int(seed_seq[idx])
                idx += 1
                if (ns, seed) not in completed_keys:
                    tasks.append((rep, ns, seed))

        total = len(cfg.sample_sizes) * cfg.n_replicates
        completed = len(all_results)

        if cfg.n_jobs == 1:
            all_results = self._run_sequential(
                tasks, cfg, estimator_fn, all_results, completed, total,
            )
        else:
            all_results = self._run_parallel(
                tasks, cfg, estimator_fn, all_results, completed, total,
            )

        elapsed = time.monotonic() - t0

        # Aggregate
        summaries = self._aggregate(all_results, cfg)

        return SimStudyResult(
            config=cfg,
            summaries=summaries,
            all_results=all_results,
            total_time_s=elapsed,
        )

    # -- Sequential execution ------------------------------------------------

    def _run_sequential(
        self,
        tasks: list[tuple[int, int, int]],
        cfg: SimStudyConfig,
        estimator_fn: Callable,
        results: list[ReplicateResult],
        completed: int,
        total: int,
    ) -> list[ReplicateResult]:
        for rep, ns, seed in tasks:
            r = _run_single_replicate(
                rep, ns, cfg.dgp_fn, estimator_fn, cfg.true_ate, cfg.alpha, seed
            )
            results.append(r)
            completed += 1
            if self.progress_callback:
                self.progress_callback(completed, total)
            if (
                cfg.checkpoint_every > 0
                and completed % cfg.checkpoint_every == 0
            ):
                self._save_checkpoint(results)
        return results

    # -- Parallel execution --------------------------------------------------

    def _run_parallel(
        self,
        tasks: list[tuple[int, int, int]],
        cfg: SimStudyConfig,
        estimator_fn: Callable,
        results: list[ReplicateResult],
        completed: int,
        total: int,
    ) -> list[ReplicateResult]:
        n_workers = cfg.n_jobs if cfg.n_jobs > 0 else os.cpu_count() or 1

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    _run_single_replicate,
                    rep, ns, cfg.dgp_fn, estimator_fn,
                    cfg.true_ate, cfg.alpha, seed,
                ): (ns, seed)
                for rep, ns, seed in tasks
            }
            for future in as_completed(futures):
                r = future.result()
                results.append(r)
                completed += 1
                if self.progress_callback:
                    self.progress_callback(completed, total)
                if (
                    cfg.checkpoint_every > 0
                    and completed % cfg.checkpoint_every == 0
                ):
                    self._save_checkpoint(results)

        return results

    # -- Aggregation ---------------------------------------------------------

    def _aggregate(
        self,
        results: list[ReplicateResult],
        cfg: SimStudyConfig,
    ) -> list[SampleSizeSummary]:
        """Compute per-sample-size summaries."""
        by_n: dict[int, list[ReplicateResult]] = {}
        for r in results:
            by_n.setdefault(r.sample_size, []).append(r)

        summaries: list[SampleSizeSummary] = []
        for ns in sorted(by_n.keys()):
            reps = by_n[ns]
            estimates = np.array([r.estimate for r in reps])
            covers = np.array([r.covers_truth for r in reps], dtype=float)
            ci_widths = np.array([r.ci_upper - r.ci_lower for r in reps])

            n_rep = len(reps)
            mean_est = float(estimates.mean())
            bias = mean_est - cfg.true_ate
            var = float(estimates.var(ddof=1)) if n_rep > 1 else 0.0
            mse = bias ** 2 + var
            rmse = np.sqrt(mse)
            cov = float(covers.mean())
            cov_se = float(np.sqrt(cov * (1 - cov) / max(n_rep, 1)))
            mean_w = float(ci_widths.mean())

            # Power: fraction of times CI excludes zero
            rejects = np.array([
                not (r.ci_lower <= 0 <= r.ci_upper) for r in reps
            ], dtype=float)
            power = float(rejects.mean())
            power_se = float(np.sqrt(power * (1 - power) / max(n_rep, 1)))

            summaries.append(SampleSizeSummary(
                sample_size=ns,
                mean_estimate=mean_est,
                bias=bias,
                variance=var,
                mse=mse,
                rmse=float(rmse),
                coverage=cov,
                coverage_se=cov_se,
                mean_ci_width=mean_w,
                power=power,
                power_se=power_se,
                n_replicates=n_rep,
            ))

        return summaries

    # -- Bias-variance decomposition -----------------------------------------

    @staticmethod
    def bias_variance_decomposition(
        results: list[ReplicateResult],
        true_ate: float,
    ) -> dict[str, float]:
        """Compute bias-variance decomposition from replicate results.

        Parameters
        ----------
        results : list[ReplicateResult]
            Replicate results (typically for a single sample size).
        true_ate : float
            Ground-truth ATE.

        Returns
        -------
        dict[str, float]
            Keys: ``"bias"``, ``"variance"``, ``"mse"``, ``"rmse"``.
        """
        ests = np.array([r.estimate for r in results])
        mean_est = ests.mean()
        bias = float(mean_est - true_ate)
        var = float(ests.var(ddof=1)) if len(ests) > 1 else 0.0
        mse = bias ** 2 + var
        return {
            "bias": bias,
            "variance": var,
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
        }

    # -- Coverage confidence bands -------------------------------------------

    @staticmethod
    def coverage_confidence_band(
        coverage: float,
        n_replicates: int,
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        """Compute a confidence interval for the coverage probability.

        Parameters
        ----------
        coverage : float
            Estimated coverage.
        n_replicates : int
            Number of replicates.
        confidence : float
            Confidence level.

        Returns
        -------
        tuple[float, float]
            (lower, upper) bounds.
        """
        from scipy import stats as sp_stats
        z = sp_stats.norm.ppf(1 - (1 - confidence) / 2)
        se = np.sqrt(coverage * (1 - coverage) / max(n_replicates, 1))
        return (
            float(max(0, coverage - z * se)),
            float(min(1, coverage + z * se)),
        )

    # -- Checkpointing -------------------------------------------------------

    def _save_checkpoint(self, results: list[ReplicateResult]) -> None:
        """Save results to a JSON checkpoint."""
        if self._checkpoint_path is None:
            return
        path = self._checkpoint_path / "mc_checkpoint.json"
        data = [
            {
                "sample_size": r.sample_size,
                "estimate": r.estimate,
                "se": r.se,
                "ci_lower": r.ci_lower,
                "ci_upper": r.ci_upper,
                "covers_truth": r.covers_truth,
                "true_ate": r.true_ate,
                "elapsed_s": r.elapsed_s,
                "seed_used": r.seed_used,
            }
            for r in results
        ]
        path.write_text(json.dumps(data, indent=2))

    def _load_checkpoint(self) -> list[ReplicateResult]:
        """Load results from a checkpoint file, if it exists."""
        if self._checkpoint_path is None:
            return []
        path = self._checkpoint_path / "mc_checkpoint.json"
        if not path.exists():
            return []
        try:
            raw = json.loads(path.read_text())
            return [
                ReplicateResult(
                    sample_size=r["sample_size"],
                    estimate=r["estimate"],
                    se=r["se"],
                    ci_lower=r["ci_lower"],
                    ci_upper=r["ci_upper"],
                    covers_truth=r["covers_truth"],
                    true_ate=r["true_ate"],
                    elapsed_s=r.get("elapsed_s", 0.0),
                    seed_used=r.get("seed_used", 0),
                )
                for r in raw
            ]
        except (json.JSONDecodeError, KeyError):
            return []


# ============================================================================
# Convenience entry point
# ============================================================================


def run_simulation_study(config: SimStudyConfig) -> SimStudyResult:
    """Run a Monte Carlo simulation study.

    Parameters
    ----------
    config : SimStudyConfig
        Full study configuration.

    Returns
    -------
    SimStudyResult
        Aggregated results.
    """
    runner = MonteCarloRunner(config)
    return runner.run()

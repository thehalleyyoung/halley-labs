"""
Bootstrap fragility analysis.

Provides bootstrap-based confidence intervals for fragility scores,
stability analysis of fragility rankings, sample-size sensitivity,
subsampling-based fragility, and permutation-based null distributions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from causalcert.types import (
    AdjacencyMatrix,
    EdgeTuple,
    FragilityChannel,
    FragilityScore,
    NodeId,
    NodeSet,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BootstrapFragilityResult:
    """Bootstrap CI for a single edge's fragility score."""
    edge: EdgeTuple
    point_estimate: float
    bootstrap_mean: float
    bootstrap_std: float
    ci_lower: float
    ci_upper: float
    n_bootstrap: int
    bootstrap_scores: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RankStabilityResult:
    """Stability of an edge's rank across bootstrap replicates."""
    edge: EdgeTuple
    mean_rank: float
    std_rank: float
    median_rank: float
    rank_ci_lower: float
    rank_ci_upper: float
    prob_top_k: float


@dataclass(frozen=True, slots=True)
class SampleSizeSensitivityResult:
    """How fragility scores change with sample size."""
    sample_sizes: tuple[int, ...]
    mean_scores: tuple[float, ...]
    std_scores: tuple[float, ...]
    convergence_rate: float


@dataclass(slots=True)
class BootstrapAnalysisReport:
    """Full bootstrap fragility analysis report."""
    edge_cis: list[BootstrapFragilityResult]
    rank_stability: list[RankStabilityResult]
    sample_sensitivity: dict[EdgeTuple, SampleSizeSensitivityResult]
    overall_stability_score: float


# ---------------------------------------------------------------------------
# Scoring function type
# ---------------------------------------------------------------------------

FragilityScoringFn = Callable[
    [AdjacencyMatrix, pd.DataFrame, NodeId, NodeId],
    list[FragilityScore],
]


# ---------------------------------------------------------------------------
# Core bootstrap fragility
# ---------------------------------------------------------------------------


def _resample_data(
    data: pd.DataFrame,
    seed: int,
    frac: float = 1.0,
) -> pd.DataFrame:
    """Resample data with replacement."""
    rng = np.random.default_rng(seed)
    n = int(len(data) * frac)
    indices = rng.choice(len(data), size=n, replace=True)
    return data.iloc[indices].reset_index(drop=True)


def _subsample_data(
    data: pd.DataFrame,
    size: int,
    seed: int,
) -> pd.DataFrame:
    """Subsample without replacement."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(data), size=min(size, len(data)), replace=False)
    return data.iloc[indices].reset_index(drop=True)


def bootstrap_fragility_scores(
    adj: AdjacencyMatrix,
    data: pd.DataFrame,
    treatment: NodeId,
    outcome: NodeId,
    scoring_fn: FragilityScoringFn,
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    seed: int = 42,
) -> list[BootstrapFragilityResult]:
    """Compute bootstrap CIs for fragility scores.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    data : pd.DataFrame
        Observational data.
    treatment, outcome : NodeId
        Treatment and outcome nodes.
    scoring_fn : FragilityScoringFn
        Function that computes fragility scores given (adj, data, treatment, outcome).
    n_bootstrap : int
        Number of bootstrap replicates.
    alpha : float
        Significance level for CIs (default 0.05 → 95% CI).
    seed : int
        Random seed.

    Returns
    -------
    list[BootstrapFragilityResult]
        Bootstrap CIs for each edge, sorted by point estimate descending.
    """
    # Point estimates
    base_scores = scoring_fn(adj, data, treatment, outcome)
    edge_to_base: dict[EdgeTuple, float] = {
        fs.edge: fs.total_score for fs in base_scores
    }
    edges = list(edge_to_base.keys())

    # Bootstrap replicates
    boot_matrix: dict[EdgeTuple, list[float]] = {e: [] for e in edges}

    for b in range(n_bootstrap):
        resampled = _resample_data(data, seed=seed + b)
        try:
            boot_scores = scoring_fn(adj, resampled, treatment, outcome)
            boot_map = {fs.edge: fs.total_score for fs in boot_scores}
            for e in edges:
                boot_matrix[e].append(boot_map.get(e, 0.0))
        except Exception:
            logger.debug("Bootstrap replicate %d failed, skipping", b)
            for e in edges:
                boot_matrix[e].append(edge_to_base[e])

    # Compute CIs
    results: list[BootstrapFragilityResult] = []
    lower_q = alpha / 2
    upper_q = 1.0 - alpha / 2

    for edge in edges:
        scores = np.array(boot_matrix[edge])
        ci_lo = float(np.percentile(scores, lower_q * 100))
        ci_hi = float(np.percentile(scores, upper_q * 100))
        results.append(BootstrapFragilityResult(
            edge=edge,
            point_estimate=edge_to_base[edge],
            bootstrap_mean=float(np.mean(scores)),
            bootstrap_std=float(np.std(scores)),
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            n_bootstrap=n_bootstrap,
            bootstrap_scores=tuple(float(s) for s in scores),
        ))

    results.sort(key=lambda x: x.point_estimate, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Rank stability
# ---------------------------------------------------------------------------


def rank_stability_analysis(
    adj: AdjacencyMatrix,
    data: pd.DataFrame,
    treatment: NodeId,
    outcome: NodeId,
    scoring_fn: FragilityScoringFn,
    n_bootstrap: int = 200,
    top_k: int = 3,
    alpha: float = 0.05,
    seed: int = 42,
) -> list[RankStabilityResult]:
    """Analyze stability of edge fragility ranking across bootstrap samples.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    data : pd.DataFrame
        Observational data.
    treatment, outcome : NodeId
        Treatment and outcome nodes.
    scoring_fn : FragilityScoringFn
        Fragility scoring function.
    n_bootstrap : int
        Number of bootstrap replicates.
    top_k : int
        Compute probability of being in top-k.
    alpha : float
        CI significance level.
    seed : int
        Random seed.

    Returns
    -------
    list[RankStabilityResult]
        One result per edge, sorted by mean rank ascending.
    """
    base_scores = scoring_fn(adj, data, treatment, outcome)
    edges = [fs.edge for fs in base_scores]

    rank_matrix: dict[EdgeTuple, list[int]] = {e: [] for e in edges}

    for b in range(n_bootstrap):
        resampled = _resample_data(data, seed=seed + b)
        try:
            boot_scores = scoring_fn(adj, resampled, treatment, outcome)
            # Sort by score descending, assign ranks
            sorted_boot = sorted(boot_scores, key=lambda x: x.total_score, reverse=True)
            for rank_idx, fs in enumerate(sorted_boot):
                if fs.edge in rank_matrix:
                    rank_matrix[fs.edge].append(rank_idx + 1)
        except Exception:
            for idx, e in enumerate(edges):
                rank_matrix[e].append(idx + 1)

    results: list[RankStabilityResult] = []
    lower_q = alpha / 2
    upper_q = 1.0 - alpha / 2

    for edge in edges:
        ranks = np.array(rank_matrix[edge], dtype=float)
        if len(ranks) == 0:
            continue
        prob_top = float(np.mean(ranks <= top_k))
        results.append(RankStabilityResult(
            edge=edge,
            mean_rank=float(np.mean(ranks)),
            std_rank=float(np.std(ranks)),
            median_rank=float(np.median(ranks)),
            rank_ci_lower=float(np.percentile(ranks, lower_q * 100)),
            rank_ci_upper=float(np.percentile(ranks, upper_q * 100)),
            prob_top_k=prob_top,
        ))

    results.sort(key=lambda x: x.mean_rank)
    return results


# ---------------------------------------------------------------------------
# Sample size sensitivity
# ---------------------------------------------------------------------------


def sample_size_sensitivity(
    adj: AdjacencyMatrix,
    data: pd.DataFrame,
    treatment: NodeId,
    outcome: NodeId,
    scoring_fn: FragilityScoringFn,
    sample_fractions: Sequence[float] = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0),
    n_repeats: int = 20,
    seed: int = 42,
) -> dict[EdgeTuple, SampleSizeSensitivityResult]:
    """Analyze how fragility scores change with sample size.

    For each fraction of the full dataset, computes mean/std of fragility
    scores across multiple subsamples.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    data : pd.DataFrame
        Full observational data.
    treatment, outcome : NodeId
        Treatment and outcome nodes.
    scoring_fn : FragilityScoringFn
        Fragility scoring function.
    sample_fractions : Sequence[float]
        Fractions of the full sample to test.
    n_repeats : int
        Number of subsampling repeats per fraction.
    seed : int
        Random seed.

    Returns
    -------
    dict[EdgeTuple, SampleSizeSensitivityResult]
    """
    n_full = len(data)
    sample_sizes_list = [max(10, int(f * n_full)) for f in sample_fractions]

    # First pass: get edge list
    base_scores = scoring_fn(adj, data, treatment, outcome)
    edges = [fs.edge for fs in base_scores]

    # Collect scores at each sample size
    edge_scores: dict[EdgeTuple, dict[int, list[float]]] = {
        e: {sz: [] for sz in sample_sizes_list} for e in edges
    }

    for sz_idx, sz in enumerate(sample_sizes_list):
        for rep in range(n_repeats):
            sub = _subsample_data(data, sz, seed=seed + sz_idx * 1000 + rep)
            try:
                scores = scoring_fn(adj, sub, treatment, outcome)
                score_map = {fs.edge: fs.total_score for fs in scores}
                for e in edges:
                    edge_scores[e][sz].append(score_map.get(e, 0.0))
            except Exception:
                for e in edges:
                    edge_scores[e][sz].append(0.0)

    results: dict[EdgeTuple, SampleSizeSensitivityResult] = {}
    for edge in edges:
        means = []
        stds = []
        for sz in sample_sizes_list:
            arr = np.array(edge_scores[edge][sz])
            means.append(float(np.mean(arr)))
            stds.append(float(np.std(arr)))

        # Estimate convergence rate: fit std ~ C / sqrt(n)
        if len(sample_sizes_list) >= 2 and any(s > 0 for s in stds):
            log_n = np.log(np.array(sample_sizes_list, dtype=float))
            log_std = np.log(np.array(stds, dtype=float) + 1e-12)
            if len(log_n) >= 2:
                slope = float(np.polyfit(log_n, log_std, 1)[0])
                conv_rate = -slope
            else:
                conv_rate = 0.0
        else:
            conv_rate = 0.0

        results[edge] = SampleSizeSensitivityResult(
            sample_sizes=tuple(sample_sizes_list),
            mean_scores=tuple(means),
            std_scores=tuple(stds),
            convergence_rate=conv_rate,
        )

    return results


# ---------------------------------------------------------------------------
# Permutation-based null distribution
# ---------------------------------------------------------------------------


def permutation_null_distribution(
    adj: AdjacencyMatrix,
    data: pd.DataFrame,
    treatment: NodeId,
    outcome: NodeId,
    scoring_fn: FragilityScoringFn,
    n_permutations: int = 200,
    seed: int = 42,
) -> dict[EdgeTuple, np.ndarray]:
    """Generate null distribution of fragility scores via permutation.

    Permutes the outcome column to break the treatment-outcome relationship,
    then computes fragility scores under the null hypothesis.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    data : pd.DataFrame
        Observational data.
    treatment, outcome : NodeId
        Treatment and outcome variable indices.
    scoring_fn : FragilityScoringFn
        Fragility scoring function.
    n_permutations : int
        Number of permutations.
    seed : int
        Random seed.

    Returns
    -------
    dict[EdgeTuple, np.ndarray]
        Mapping from edge to array of null scores.
    """
    rng = np.random.default_rng(seed)
    base_scores = scoring_fn(adj, data, treatment, outcome)
    edges = [fs.edge for fs in base_scores]

    null_scores: dict[EdgeTuple, list[float]] = {e: [] for e in edges}
    outcome_col = data.columns[outcome]

    for _ in range(n_permutations):
        perm_data = data.copy()
        perm_data[outcome_col] = rng.permutation(perm_data[outcome_col].values)
        try:
            perm_scores = scoring_fn(adj, perm_data, treatment, outcome)
            perm_map = {fs.edge: fs.total_score for fs in perm_scores}
            for e in edges:
                null_scores[e].append(perm_map.get(e, 0.0))
        except Exception:
            for e in edges:
                null_scores[e].append(0.0)

    return {e: np.array(v) for e, v in null_scores.items()}


def permutation_p_values(
    adj: AdjacencyMatrix,
    data: pd.DataFrame,
    treatment: NodeId,
    outcome: NodeId,
    scoring_fn: FragilityScoringFn,
    n_permutations: int = 200,
    seed: int = 42,
) -> dict[EdgeTuple, float]:
    """Compute permutation p-values for each edge's fragility score.

    The p-value is the fraction of null scores >= the observed score.
    """
    base_scores = scoring_fn(adj, data, treatment, outcome)
    base_map = {fs.edge: fs.total_score for fs in base_scores}

    null_dist = permutation_null_distribution(
        adj, data, treatment, outcome, scoring_fn, n_permutations, seed
    )

    p_values: dict[EdgeTuple, float] = {}
    for edge, observed in base_map.items():
        null = null_dist.get(edge, np.array([0.0]))
        p_values[edge] = float(np.mean(null >= observed))

    return p_values


# ---------------------------------------------------------------------------
# Subsampling-based fragility
# ---------------------------------------------------------------------------


def subsampling_fragility(
    adj: AdjacencyMatrix,
    data: pd.DataFrame,
    treatment: NodeId,
    outcome: NodeId,
    scoring_fn: FragilityScoringFn,
    subsample_fraction: float = 0.5,
    n_subsamples: int = 100,
    seed: int = 42,
) -> list[BootstrapFragilityResult]:
    """Fragility analysis via subsampling (without replacement).

    Similar to bootstrap but uses subsamples of size ``subsample_fraction * n``
    drawn without replacement. This can give better coverage properties
    than the bootstrap for non-smooth statistics.

    Returns
    -------
    list[BootstrapFragilityResult]
        Results in same format as bootstrap (reusing the dataclass).
    """
    base_scores = scoring_fn(adj, data, treatment, outcome)
    edge_to_base: dict[EdgeTuple, float] = {
        fs.edge: fs.total_score for fs in base_scores
    }
    edges = list(edge_to_base.keys())
    sub_size = max(10, int(subsample_fraction * len(data)))

    sub_matrix: dict[EdgeTuple, list[float]] = {e: [] for e in edges}

    for s in range(n_subsamples):
        sub = _subsample_data(data, sub_size, seed=seed + s)
        try:
            sub_scores = scoring_fn(adj, sub, treatment, outcome)
            score_map = {fs.edge: fs.total_score for fs in sub_scores}
            for e in edges:
                sub_matrix[e].append(score_map.get(e, 0.0))
        except Exception:
            for e in edges:
                sub_matrix[e].append(edge_to_base[e])

    results: list[BootstrapFragilityResult] = []
    for edge in edges:
        scores = np.array(sub_matrix[edge])
        results.append(BootstrapFragilityResult(
            edge=edge,
            point_estimate=edge_to_base[edge],
            bootstrap_mean=float(np.mean(scores)),
            bootstrap_std=float(np.std(scores)),
            ci_lower=float(np.percentile(scores, 2.5)),
            ci_upper=float(np.percentile(scores, 97.5)),
            n_bootstrap=n_subsamples,
            bootstrap_scores=tuple(float(s) for s in scores),
        ))

    results.sort(key=lambda x: x.point_estimate, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Full bootstrap analysis
# ---------------------------------------------------------------------------


def full_bootstrap_analysis(
    adj: AdjacencyMatrix,
    data: pd.DataFrame,
    treatment: NodeId,
    outcome: NodeId,
    scoring_fn: FragilityScoringFn,
    n_bootstrap: int = 200,
    top_k: int = 3,
    sample_fractions: Sequence[float] = (0.2, 0.5, 0.8, 1.0),
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapAnalysisReport:
    """Run comprehensive bootstrap fragility analysis.

    Combines bootstrap CIs, rank stability, and sample-size sensitivity
    into a single report.
    """
    edge_cis = bootstrap_fragility_scores(
        adj, data, treatment, outcome, scoring_fn,
        n_bootstrap=n_bootstrap, alpha=alpha, seed=seed,
    )
    rank_stab = rank_stability_analysis(
        adj, data, treatment, outcome, scoring_fn,
        n_bootstrap=n_bootstrap, top_k=top_k, alpha=alpha, seed=seed,
    )
    sample_sens = sample_size_sensitivity(
        adj, data, treatment, outcome, scoring_fn,
        sample_fractions=sample_fractions, n_repeats=min(20, n_bootstrap),
        seed=seed,
    )

    # Overall stability: average of (1 - normalized rank std)
    if rank_stab:
        max_rank = max(r.mean_rank for r in rank_stab)
        norm_stds = [r.std_rank / max(max_rank, 1.0) for r in rank_stab]
        overall = float(1.0 - np.mean(norm_stds))
    else:
        overall = 1.0

    return BootstrapAnalysisReport(
        edge_cis=edge_cis,
        rank_stability=rank_stab,
        sample_sensitivity=sample_sens,
        overall_stability_score=overall,
    )

"""
Evaluation metrics for CausalCert benchmarks.

Provides coverage rate, interval width, fragility AUC, and other metrics
for comparing robustness radius estimates against ground truth.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from causalcert.evaluation.dgp import DGPInstance
from causalcert.types import CITestResult, FragilityScore, RobustnessRadius


# ---------------------------------------------------------------------------
# Radius metrics
# ---------------------------------------------------------------------------


def coverage_rate(
    instances: Sequence[DGPInstance],
    estimated_radii: Sequence[RobustnessRadius],
) -> float:
    """Fraction of instances where the true radius falls within [LB, UB].

    Parameters
    ----------
    instances : Sequence[DGPInstance]
        Ground-truth instances.
    estimated_radii : Sequence[RobustnessRadius]
        Estimated radii.

    Returns
    -------
    float
        Coverage rate in [0, 1].
    """
    if not instances:
        return 0.0
    covered = sum(
        1
        for inst, est in zip(instances, estimated_radii)
        if est.lower_bound <= inst.true_radius <= est.upper_bound
    )
    return covered / len(instances)


def interval_width(
    estimated_radii: Sequence[RobustnessRadius],
) -> float:
    """Mean interval width (UB − LB) across instances.

    Parameters
    ----------
    estimated_radii : Sequence[RobustnessRadius]

    Returns
    -------
    float
    """
    if not estimated_radii:
        return 0.0
    widths = [r.upper_bound - r.lower_bound for r in estimated_radii]
    return float(np.mean(widths))


def exact_match_rate(
    instances: Sequence[DGPInstance],
    estimated_radii: Sequence[RobustnessRadius],
) -> float:
    """Fraction of instances where LB == UB == true radius.

    Parameters
    ----------
    instances : Sequence[DGPInstance]
    estimated_radii : Sequence[RobustnessRadius]

    Returns
    -------
    float
    """
    if not instances:
        return 0.0
    exact = sum(
        1
        for inst, est in zip(instances, estimated_radii)
        if est.lower_bound == est.upper_bound == inst.true_radius
    )
    return exact / len(instances)


def within_one_rate(
    instances: Sequence[DGPInstance],
    estimated_radii: Sequence[RobustnessRadius],
) -> float:
    """Fraction of instances where UB is within 1 of the true radius.

    Parameters
    ----------
    instances : Sequence[DGPInstance]
    estimated_radii : Sequence[RobustnessRadius]

    Returns
    -------
    float
    """
    if not instances:
        return 0.0
    close = sum(
        1
        for inst, est in zip(instances, estimated_radii)
        if abs(est.upper_bound - inst.true_radius) <= 1
    )
    return close / len(instances)


def mean_absolute_error(
    instances: Sequence[DGPInstance],
    estimated_radii: Sequence[RobustnessRadius],
    use_upper: bool = True,
) -> float:
    """Mean absolute error of radius estimates.

    Parameters
    ----------
    instances : Sequence[DGPInstance]
    estimated_radii : Sequence[RobustnessRadius]
    use_upper : bool
        If ``True``, compare against upper bound; otherwise midpoint.

    Returns
    -------
    float
    """
    if not instances:
        return 0.0
    errors = []
    for inst, est in zip(instances, estimated_radii):
        if use_upper:
            errors.append(abs(est.upper_bound - inst.true_radius))
        else:
            midpoint = (est.lower_bound + est.upper_bound) / 2.0
            errors.append(abs(midpoint - inst.true_radius))
    return float(np.mean(errors))


# ---------------------------------------------------------------------------
# Fragility metrics
# ---------------------------------------------------------------------------


def fragility_auc(
    scores: Sequence[FragilityScore],
    true_fragile_edges: set[tuple[int, int]],
) -> float:
    """AUC for fragility ranking — how well the scorer identifies truly fragile edges.

    Parameters
    ----------
    scores : Sequence[FragilityScore]
        Predicted fragility scores (sorted by score).
    true_fragile_edges : set[tuple[int, int]]
        Ground-truth set of fragile edges.

    Returns
    -------
    float
        AUC in [0, 1].
    """
    if not scores or not true_fragile_edges:
        return 0.0

    # Create binary labels and scores for ROC computation
    y_true = []
    y_score = []
    for fs in scores:
        y_true.append(1 if fs.edge in true_fragile_edges else 0)
        y_score.append(fs.total_score)

    y_true_arr = np.array(y_true)
    y_score_arr = np.array(y_score)

    n_pos = int(y_true_arr.sum())
    n_neg = len(y_true_arr) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5  # Degenerate case

    # Sort by decreasing score
    order = np.argsort(-y_score_arr)
    y_sorted = y_true_arr[order]

    # Compute AUC via Wilcoxon-Mann-Whitney
    tp = 0
    auc = 0.0
    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            auc += tp
    auc /= (n_pos * n_neg)

    return float(auc)


def fragility_precision_at_k(
    scores: Sequence[FragilityScore],
    true_fragile_edges: set[tuple[int, int]],
    k: int = 5,
) -> float:
    """Precision@k for fragility ranking.

    Parameters
    ----------
    scores : Sequence[FragilityScore]
    true_fragile_edges : set[tuple[int, int]]
    k : int

    Returns
    -------
    float
    """
    if not scores or k <= 0:
        return 0.0
    top_k = scores[:k]
    hits = sum(1 for fs in top_k if fs.edge in true_fragile_edges)
    return hits / min(k, len(scores))


def fragility_recall_at_k(
    scores: Sequence[FragilityScore],
    true_fragile_edges: set[tuple[int, int]],
    k: int = 10,
) -> float:
    """Recall@k for fragility ranking.

    Parameters
    ----------
    scores : Sequence[FragilityScore]
    true_fragile_edges : set[tuple[int, int]]
    k : int

    Returns
    -------
    float
    """
    if not true_fragile_edges:
        return 0.0
    top_k = scores[:k]
    hits = sum(1 for fs in top_k if fs.edge in true_fragile_edges)
    return hits / len(true_fragile_edges)


# ---------------------------------------------------------------------------
# CI test calibration
# ---------------------------------------------------------------------------


def pvalue_calibration(
    ci_results: Sequence[CITestResult],
    true_independent: set[tuple[int, int, frozenset[int]]],
) -> dict[str, float]:
    """Assess calibration of CI test p-values.

    Under the null (true independence), p-values should be uniform.
    Computes the Kolmogorov-Smirnov statistic.

    Parameters
    ----------
    ci_results : Sequence[CITestResult]
        All CI test results.
    true_independent : set[tuple[int, int, frozenset[int]]]
        Triples that are truly independent.

    Returns
    -------
    dict[str, float]
        ``ks_statistic``, ``ks_pvalue``, ``mean_p``, ``std_p``.
    """
    null_pvals = []
    for r in ci_results:
        key = (r.x, r.y, r.conditioning_set)
        if key in true_independent:
            null_pvals.append(r.p_value)

    if len(null_pvals) < 5:
        return {"ks_statistic": 0.0, "ks_pvalue": 1.0,
                "mean_p": 0.0, "std_p": 0.0}

    arr = np.array(null_pvals)
    # KS test against Uniform(0,1)
    n = len(arr)
    sorted_p = np.sort(arr)
    expected = np.arange(1, n + 1) / n
    ks_stat = float(np.max(np.abs(sorted_p - expected)))

    # Approximate p-value
    from math import exp, sqrt
    z = ks_stat * sqrt(n)
    ks_pval = max(0.0, min(1.0, 2.0 * exp(-2.0 * z * z)))

    return {
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_pval,
        "mean_p": float(np.mean(arr)),
        "std_p": float(np.std(arr)),
    }


def false_discovery_proportion(
    ci_results: Sequence[CITestResult],
    true_independent: set[tuple[int, int, frozenset[int]]],
) -> float:
    """Compute the false discovery proportion (FDP) among rejected tests.

    FDP = (# false rejections) / max(# total rejections, 1)

    Parameters
    ----------
    ci_results : Sequence[CITestResult]
    true_independent : set[tuple[int, int, frozenset[int]]]
        Triples that are truly independent (rejecting these is a false discovery).

    Returns
    -------
    float
    """
    rejections = [r for r in ci_results if r.reject]
    if not rejections:
        return 0.0
    false_rej = sum(
        1 for r in rejections
        if (r.x, r.y, r.conditioning_set) in true_independent
    )
    return false_rej / len(rejections)


# ---------------------------------------------------------------------------
# Runtime metrics
# ---------------------------------------------------------------------------


def runtime_summary(
    timings: Sequence[float],
) -> dict[str, float]:
    """Summarise runtimes.

    Parameters
    ----------
    timings : Sequence[float]
        Runtimes in seconds.

    Returns
    -------
    dict[str, float]
    """
    if not timings:
        return {"mean": 0.0, "median": 0.0, "std": 0.0,
                "min": 0.0, "max": 0.0, "p95": 0.0}
    arr = np.array(timings)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


# ---------------------------------------------------------------------------
# Significance testing
# ---------------------------------------------------------------------------


def paired_permutation_test(
    metric_a: Sequence[float],
    metric_b: Sequence[float],
    n_permutations: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Two-sided paired permutation test for comparing two methods.

    Tests H_0: mean(A) = mean(B).

    Parameters
    ----------
    metric_a, metric_b : Sequence[float]
        Paired metric values.
    n_permutations : int
    seed : int

    Returns
    -------
    dict[str, float]
        ``observed_diff``, ``p_value``.
    """
    a = np.array(metric_a, dtype=float)
    b = np.array(metric_b, dtype=float)
    if len(a) != len(b) or len(a) == 0:
        return {"observed_diff": 0.0, "p_value": 1.0}

    diffs = a - b
    observed = float(np.mean(diffs))
    rng = np.random.RandomState(seed)

    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diffs))
        perm_mean = float(np.mean(diffs * signs))
        if abs(perm_mean) >= abs(observed):
            count += 1

    return {
        "observed_diff": observed,
        "p_value": (count + 1) / (n_permutations + 1),
    }


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------


def compute_all_metrics(
    instances: Sequence[DGPInstance],
    estimated_radii: Sequence[RobustnessRadius],
    fragility_scores: Sequence[Sequence[FragilityScore]] | None = None,
    true_fragile_edges: Sequence[set[tuple[int, int]]] | None = None,
    timings: Sequence[float] | None = None,
) -> dict[str, float]:
    """Compute all evaluation metrics.

    Parameters
    ----------
    instances : Sequence[DGPInstance]
    estimated_radii : Sequence[RobustnessRadius]
    fragility_scores : Sequence[Sequence[FragilityScore]] | None
    true_fragile_edges : Sequence[set[tuple[int, int]]] | None
    timings : Sequence[float] | None

    Returns
    -------
    dict[str, float]
    """
    metrics: dict[str, float] = {
        "coverage": coverage_rate(instances, estimated_radii),
        "interval_width": interval_width(estimated_radii),
        "exact_match": exact_match_rate(instances, estimated_radii),
        "within_one": within_one_rate(instances, estimated_radii),
        "mae_upper": mean_absolute_error(instances, estimated_radii, use_upper=True),
        "mae_midpoint": mean_absolute_error(instances, estimated_radii, use_upper=False),
    }

    if fragility_scores and true_fragile_edges:
        aucs = []
        prec5s = []
        for fs, tfe in zip(fragility_scores, true_fragile_edges):
            aucs.append(fragility_auc(fs, tfe))
            prec5s.append(fragility_precision_at_k(fs, tfe, k=5))
        metrics["fragility_auc_mean"] = float(np.mean(aucs))
        metrics["fragility_prec_at_5"] = float(np.mean(prec5s))

    if timings:
        rt = runtime_summary(timings)
        for k, v in rt.items():
            metrics[f"runtime_{k}"] = v

    return metrics

"""
Evaluation metrics for the CausalBound causal inference system for financial
systemic risk.  Provides bound-ratio analysis, pathway recall/precision,
MCTS discovery scoring, verification overhead measurement, SMT/cache
statistics, LP convergence fitting, and LaTeX report generation.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize, stats


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BoundRatioResult:
    """Result of comparing CausalBound interval width to true interval width."""
    ratio: float
    cb_width: float
    true_width: float
    is_valid: bool
    tightness_category: str  # "tight", "moderate", "loose", "invalid"


@dataclass
class PathwayRecallResult:
    """Result of comparing discovered contagion pathways to planted ones."""
    recall: float
    precision: float
    f1: float
    matched_pathways: List[Tuple[tuple, tuple]]
    missed_pathways: List[tuple]


@dataclass
class DiscoveryRatioResult:
    """Result of comparing MCTS worst-case loss to baseline."""
    ratio: float
    mcts_loss: float
    baseline_loss: float
    improvement_pct: float


@dataclass
class OverheadResult:
    """Result of comparing verified pipeline time to unverified."""
    ratio: float
    verified_time: float
    unverified_time: float
    overhead_seconds: float
    is_acceptable: bool


@dataclass
class SMTStats:
    """Statistics over SMT assertion counts per inference pass."""
    mean: float
    median: float
    std: float
    max: int
    total: int
    per_pass_histogram: Dict[str, int]


@dataclass
class CacheStats:
    """Cache hit/miss/eviction statistics."""
    hit_rate: float
    miss_rate: float
    eviction_rate: float
    total_requests: int
    efficiency_score: float


@dataclass
class ConvergenceMetrics:
    """LP convergence analysis via exponential-decay fit."""
    convergence_rate: float
    final_gap: float
    iterations_to_threshold: Optional[int]
    fitted_params: Tuple[float, float, float]
    r_squared: float


@dataclass
class SummaryReport:
    """Aggregated summary across all metric categories."""
    bound_ratio_stats: Dict[str, float]
    pathway_recall_stats: Dict[str, float]
    discovery_ratio_stats: Dict[str, float]
    overhead_stats: Dict[str, float]
    quality_score: float
    percentiles: Dict[str, Dict[str, float]]
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _interval_width(bounds: Tuple[float, float]) -> float:
    """Width of an interval, always non-negative."""
    return max(0.0, bounds[1] - bounds[0])


def _overlap_fraction(seq_a: Sequence, seq_b: Sequence) -> float:
    """Fraction of elements in *seq_a* that appear (in order) in *seq_b*,
    computed via longest-common-subsequence length divided by len(seq_a)."""
    if len(seq_a) == 0:
        return 1.0
    n, m = len(seq_a), len(seq_b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[n][m]
    return lcs_len / n


def _classify_tightness(ratio: float) -> str:
    if ratio <= 0.0 or not math.isfinite(ratio):
        return "invalid"
    if ratio <= 1.2:
        return "tight"
    if ratio <= 2.0:
        return "moderate"
    return "loose"


def _safe_div(numer: float, denom: float, default: float = 0.0) -> float:
    if denom == 0.0 or not math.isfinite(denom):
        return default
    return numer / denom


def _aggregate_floats(values: Sequence[float]) -> Dict[str, float]:
    """Compute descriptive statistics for a sequence of floats."""
    if len(values) == 0:
        return {"mean": 0.0, "median": 0.0, "std": 0.0,
                "min": 0.0, "max": 0.0, "count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": len(arr),
    }


def _percentiles(values: Sequence[float],
                 qs: Sequence[float] = (5, 25, 50, 75, 95),
                 ) -> Dict[str, float]:
    if len(values) == 0:
        return {f"p{int(q)}": 0.0 for q in qs}
    arr = np.asarray(values, dtype=np.float64)
    return {f"p{int(q)}": float(np.percentile(arr, q)) for q in qs}


def _exp_decay(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Exponential decay model: a * exp(-b * x) + c"""
    return a * np.exp(-b * x) + c


def _build_histogram(counts: Sequence[int], n_bins: int = 10) -> Dict[str, int]:
    """Build a simple histogram dictionary from integer counts."""
    if len(counts) == 0:
        return {}
    arr = np.asarray(counts, dtype=np.int64)
    lo, hi = int(arr.min()), int(arr.max())
    if lo == hi:
        return {str(lo): len(counts)}
    bin_width = max(1, (hi - lo) // n_bins + 1)
    histogram: Dict[str, int] = {}
    for v in arr:
        bucket_lo = lo + ((int(v) - lo) // bin_width) * bin_width
        bucket_hi = bucket_lo + bin_width - 1
        key = f"{bucket_lo}-{bucket_hi}"
        histogram[key] = histogram.get(key, 0) + 1
    return histogram


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    replacements = [
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _format_float(value: float, precision: int = 4) -> str:
    if not math.isfinite(value):
        return str(value)
    return f"{value:.{precision}f}"


# ---------------------------------------------------------------------------
# MetricsComputer
# ---------------------------------------------------------------------------

class MetricsComputer:
    """Central class for computing all CausalBound evaluation metrics."""

    def __init__(
        self,
        overlap_threshold: float = 0.5,
        overhead_acceptable_ratio: float = 3.0,
        convergence_gap_threshold: float = 1e-4,
        histogram_bins: int = 10,
    ) -> None:
        if not 0.0 < overlap_threshold <= 1.0:
            raise ValueError("overlap_threshold must be in (0, 1]")
        if overhead_acceptable_ratio <= 0.0:
            raise ValueError("overhead_acceptable_ratio must be positive")
        if convergence_gap_threshold <= 0.0:
            raise ValueError("convergence_gap_threshold must be positive")
        self.overlap_threshold = overlap_threshold
        self.overhead_acceptable_ratio = overhead_acceptable_ratio
        self.convergence_gap_threshold = convergence_gap_threshold
        self.histogram_bins = histogram_bins

    # ---- 1. Bound ratio -------------------------------------------------

    def compute_bound_ratio(
        self,
        cb_bounds: Tuple[float, float],
        true_bounds: Tuple[float, float],
    ) -> BoundRatioResult:
        """Ratio of CausalBound interval width to true interval width.

        A ratio near 1.0 means the CausalBound interval is as tight as the
        ground-truth interval; values > 1 mean it is wider (conservative).
        """
        if len(cb_bounds) != 2 or len(true_bounds) != 2:
            return BoundRatioResult(
                ratio=float("nan"), cb_width=0.0, true_width=0.0,
                is_valid=False, tightness_category="invalid",
            )

        cb_lo, cb_hi = cb_bounds
        true_lo, true_hi = true_bounds

        if cb_lo > cb_hi or true_lo > true_hi:
            return BoundRatioResult(
                ratio=float("nan"),
                cb_width=_interval_width((cb_lo, cb_hi)),
                true_width=_interval_width((true_lo, true_hi)),
                is_valid=False,
                tightness_category="invalid",
            )

        for val in (cb_lo, cb_hi, true_lo, true_hi):
            if not math.isfinite(val):
                return BoundRatioResult(
                    ratio=float("nan"), cb_width=float("inf"),
                    true_width=float("inf"), is_valid=False,
                    tightness_category="invalid",
                )

        cb_width = _interval_width(cb_bounds)
        true_width = _interval_width(true_bounds)

        if true_width == 0.0:
            if cb_width == 0.0:
                ratio = 1.0
            else:
                ratio = float("inf")
            return BoundRatioResult(
                ratio=ratio, cb_width=cb_width, true_width=true_width,
                is_valid=True, tightness_category=_classify_tightness(ratio),
            )

        ratio = cb_width / true_width
        return BoundRatioResult(
            ratio=ratio,
            cb_width=cb_width,
            true_width=true_width,
            is_valid=True,
            tightness_category=_classify_tightness(ratio),
        )

    # ---- 2. Pathway recall -----------------------------------------------

    def compute_pathway_recall(
        self,
        discovered: List[tuple],
        planted: List[tuple],
    ) -> PathwayRecallResult:
        """Fraction of planted contagion pathways recovered by discovery.

        Supports partial matching: a discovered pathway is considered a match
        for a planted pathway when their LCS-based overlap fraction meets or
        exceeds ``self.overlap_threshold``.
        """
        if len(planted) == 0:
            return PathwayRecallResult(
                recall=1.0,
                precision=1.0 if len(discovered) == 0 else 0.0,
                f1=1.0 if len(discovered) == 0 else 0.0,
                matched_pathways=[],
                missed_pathways=[],
            )

        if len(discovered) == 0:
            return PathwayRecallResult(
                recall=0.0, precision=0.0, f1=0.0,
                matched_pathways=[],
                missed_pathways=list(planted),
            )

        # Greedy bipartite matching: for each planted pathway, find the best
        # (highest-overlap) discovered pathway that hasn't been claimed yet.
        planted_matched = [False] * len(planted)
        discovered_used = [False] * len(discovered)
        matched_pairs: List[Tuple[tuple, tuple]] = []

        # Pre-compute overlap matrix
        overlap_matrix = np.zeros((len(planted), len(discovered)),
                                  dtype=np.float64)
        for i, p_path in enumerate(planted):
            for j, d_path in enumerate(discovered):
                overlap_matrix[i, j] = _overlap_fraction(p_path, d_path)

        # Iterate greedily from best overlap down
        flat_indices = np.argsort(overlap_matrix.ravel())[::-1]
        for flat_idx in flat_indices:
            i = int(flat_idx // len(discovered))
            j = int(flat_idx % len(discovered))
            if planted_matched[i] or discovered_used[j]:
                continue
            if overlap_matrix[i, j] < self.overlap_threshold:
                break
            planted_matched[i] = True
            discovered_used[j] = True
            matched_pairs.append((planted[i], discovered[j]))

        n_matched = len(matched_pairs)
        recall = n_matched / len(planted)
        precision = n_matched / len(discovered)
        if recall + precision > 0:
            f1 = 2 * recall * precision / (recall + precision)
        else:
            f1 = 0.0

        missed = [planted[i] for i, m in enumerate(planted_matched) if not m]

        return PathwayRecallResult(
            recall=recall,
            precision=precision,
            f1=f1,
            matched_pathways=matched_pairs,
            missed_pathways=missed,
        )

    # ---- 3. Discovery ratio ----------------------------------------------

    def compute_discovery_ratio(
        self,
        mcts_loss: float,
        baseline_loss: float,
    ) -> DiscoveryRatioResult:
        """Ratio of MCTS worst-case loss to baseline worst-case loss.

        A ratio > 1 means MCTS discovered a worse (larger-loss) scenario than
        the baseline, which is the desired outcome for stress-testing.
        """
        if baseline_loss < 0 or mcts_loss < 0:
            raise ValueError("Loss values must be non-negative")

        if not math.isfinite(mcts_loss) or not math.isfinite(baseline_loss):
            raise ValueError("Loss values must be finite")

        ratio = _safe_div(mcts_loss, baseline_loss, default=1.0)
        improvement_pct = (ratio - 1.0) * 100.0
        return DiscoveryRatioResult(
            ratio=ratio,
            mcts_loss=mcts_loss,
            baseline_loss=baseline_loss,
            improvement_pct=improvement_pct,
        )

    # ---- 4. Overhead ratio -----------------------------------------------

    def compute_overhead_ratio(
        self,
        verified_time: float,
        unverified_time: float,
    ) -> OverheadResult:
        """Ratio of verified-pipeline wall time to unverified-pipeline time.

        ``is_acceptable`` is True when the ratio is at or below
        ``self.overhead_acceptable_ratio``.
        """
        if verified_time < 0 or unverified_time < 0:
            raise ValueError("Time values must be non-negative")

        ratio = _safe_div(verified_time, unverified_time, default=1.0)
        overhead_seconds = verified_time - unverified_time
        is_acceptable = ratio <= self.overhead_acceptable_ratio

        return OverheadResult(
            ratio=ratio,
            verified_time=verified_time,
            unverified_time=unverified_time,
            overhead_seconds=overhead_seconds,
            is_acceptable=is_acceptable,
        )

    # ---- 5. SMT statistics -----------------------------------------------

    def compute_smt_statistics(
        self,
        assertion_counts: List[int],
    ) -> SMTStats:
        """Descriptive statistics over SMT assertion counts per pass."""
        if len(assertion_counts) == 0:
            return SMTStats(
                mean=0.0, median=0.0, std=0.0, max=0, total=0,
                per_pass_histogram={},
            )

        arr = np.asarray(assertion_counts, dtype=np.int64)
        if np.any(arr < 0):
            raise ValueError("Assertion counts must be non-negative")

        histogram = _build_histogram(assertion_counts, n_bins=self.histogram_bins)
        std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

        return SMTStats(
            mean=float(np.mean(arr)),
            median=float(np.median(arr)),
            std=std_val,
            max=int(np.max(arr)),
            total=int(np.sum(arr)),
            per_pass_histogram=histogram,
        )

    # ---- 6. Cache statistics ---------------------------------------------

    def compute_cache_statistics(
        self,
        hits: int,
        misses: int,
        evictions: int,
    ) -> CacheStats:
        """Cache hit-rate, miss-rate, eviction-rate, and efficiency score.

        The efficiency score is a weighted combination:
            0.7 * hit_rate  +  0.2 * (1 - eviction_rate)  +  0.1 * coverage
        where coverage is 1 if total_requests > 0 else 0.
        """
        if hits < 0 or misses < 0 or evictions < 0:
            raise ValueError("Counts must be non-negative")

        total = hits + misses
        hit_rate = _safe_div(hits, total)
        miss_rate = _safe_div(misses, total)
        eviction_rate = _safe_div(evictions, total) if total > 0 else 0.0

        coverage = 1.0 if total > 0 else 0.0
        efficiency = (
            0.7 * hit_rate
            + 0.2 * (1.0 - min(eviction_rate, 1.0))
            + 0.1 * coverage
        )

        return CacheStats(
            hit_rate=hit_rate,
            miss_rate=miss_rate,
            eviction_rate=eviction_rate,
            total_requests=total,
            efficiency_score=efficiency,
        )

    # ---- 7. Summary statistics -------------------------------------------

    def summary_statistics(
        self,
        all_metrics: Dict[str, list],
    ) -> SummaryReport:
        """Aggregate all collected metric lists into a comprehensive summary.

        Expected keys in *all_metrics*:
            ``bound_ratios``   – list of float
            ``pathway_recalls`` – list of float
            ``discovery_ratios`` – list of float
            ``overhead_ratios`` – list of float

        Missing keys are treated as empty lists.
        """
        bound_ratios = [
            v for v in all_metrics.get("bound_ratios", [])
            if math.isfinite(v)
        ]
        pathway_recalls = list(all_metrics.get("pathway_recalls", []))
        discovery_ratios = [
            v for v in all_metrics.get("discovery_ratios", [])
            if math.isfinite(v)
        ]
        overhead_ratios = [
            v for v in all_metrics.get("overhead_ratios", [])
            if math.isfinite(v)
        ]

        bound_stats = _aggregate_floats(bound_ratios)
        pathway_stats = _aggregate_floats(pathway_recalls)
        discovery_stats = _aggregate_floats(discovery_ratios)
        overhead_stats_dict = _aggregate_floats(overhead_ratios)

        # Percentiles
        pctls: Dict[str, Dict[str, float]] = {
            "bound_ratios": _percentiles(bound_ratios),
            "pathway_recalls": _percentiles(pathway_recalls),
            "discovery_ratios": _percentiles(discovery_ratios),
            "overhead_ratios": _percentiles(overhead_ratios),
        }

        # Overall quality score in [0, 1]:
        #   - bound tightness:  1 / mean_bound_ratio  (capped at 1)
        #   - pathway coverage: mean recall
        #   - discovery power:  min(mean_discovery_ratio, 2) / 2
        #   - overhead penalty: 1 / mean_overhead_ratio  (capped at 1)
        warn_msgs: List[str] = []
        scores: List[float] = []

        if bound_stats["count"] > 0:
            bt = min(1.0, _safe_div(1.0, bound_stats["mean"], 0.0))
            scores.append(bt)
        else:
            warn_msgs.append("No bound-ratio data available")

        if pathway_stats["count"] > 0:
            scores.append(min(1.0, pathway_stats["mean"]))
        else:
            warn_msgs.append("No pathway-recall data available")

        if discovery_stats["count"] > 0:
            dp = min(1.0, discovery_stats["mean"] / 2.0)
            scores.append(dp)
        else:
            warn_msgs.append("No discovery-ratio data available")

        if overhead_stats_dict["count"] > 0:
            op = min(1.0, _safe_div(1.0, overhead_stats_dict["mean"], 0.0))
            scores.append(op)
        else:
            warn_msgs.append("No overhead-ratio data available")

        quality_score = float(np.mean(scores)) if scores else 0.0

        return SummaryReport(
            bound_ratio_stats=bound_stats,
            pathway_recall_stats=pathway_stats,
            discovery_ratio_stats=discovery_stats,
            overhead_stats=overhead_stats_dict,
            quality_score=quality_score,
            percentiles=pctls,
            warnings=warn_msgs,
        )

    # ---- 8. Convergence metrics ------------------------------------------

    def compute_convergence_metrics(
        self,
        gap_history: List[float],
    ) -> ConvergenceMetrics:
        """Fit an exponential-decay model to the LP duality-gap history and
        extract convergence diagnostics.

        Model:  gap(t) = a * exp(-b * t) + c

        ``convergence_rate`` is the fitted decay constant *b*.
        ``iterations_to_threshold`` is the first iteration where the gap
        falls below ``self.convergence_gap_threshold``, or None.
        """
        if len(gap_history) < 3:
            raise ValueError("Need at least 3 gap values to fit convergence")

        arr = np.asarray(gap_history, dtype=np.float64)
        if np.any(arr < 0):
            raise ValueError("Gap values must be non-negative")

        x = np.arange(len(arr), dtype=np.float64)
        final_gap = float(arr[-1])

        # Initial parameter guesses
        a0 = float(arr[0] - arr[-1]) if arr[0] > arr[-1] else 1.0
        b0 = 0.1
        c0 = float(arr[-1])
        p0 = [max(a0, 1e-8), max(b0, 1e-8), max(c0, 0.0)]

        try:
            popt, _ = optimize.curve_fit(
                _exp_decay, x, arr, p0=p0,
                maxfev=10000,
                bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
            )
        except RuntimeError:
            warnings.warn("Exponential fit did not converge; using fallback")
            popt = np.array(p0, dtype=np.float64)

        a_fit, b_fit, c_fit = float(popt[0]), float(popt[1]), float(popt[2])

        # R-squared
        y_pred = _exp_decay(x, a_fit, b_fit, c_fit)
        ss_res = float(np.sum((arr - y_pred) ** 2))
        ss_tot = float(np.sum((arr - np.mean(arr)) ** 2))
        r_squared = 1.0 - _safe_div(ss_res, ss_tot, default=0.0)
        r_squared = max(0.0, r_squared)

        # Iterations to threshold
        iters_to_thresh: Optional[int] = None
        for idx, g in enumerate(gap_history):
            if g <= self.convergence_gap_threshold:
                iters_to_thresh = idx
                break

        return ConvergenceMetrics(
            convergence_rate=b_fit,
            final_gap=final_gap,
            iterations_to_threshold=iters_to_thresh,
            fitted_params=(a_fit, b_fit, c_fit),
            r_squared=r_squared,
        )

    # ---- 9. LaTeX table --------------------------------------------------

    def latex_table(
        self,
        results: Dict[str, float],
        caption: str = "CausalBound Evaluation Metrics",
        label: str = "tab:metrics",
        precision: int = 4,
    ) -> str:
        """Generate a LaTeX table string from a flat metric-name → value dict."""
        lines: List[str] = []
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append(f"\\caption{{{_escape_latex(caption)}}}")
        lines.append(f"\\label{{{label}}}")
        lines.append("\\begin{tabular}{lr}")
        lines.append("\\toprule")
        lines.append("Metric & Value \\\\")
        lines.append("\\midrule")

        for metric_name, value in results.items():
            safe_name = _escape_latex(str(metric_name))
            formatted = _format_float(value, precision)
            lines.append(f"{safe_name} & {formatted} \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        return "\n".join(lines)

    # ---- 10. Comparison table --------------------------------------------

    def comparison_table(
        self,
        methods_results: Dict[str, Dict[str, float]],
        caption: str = "Method Comparison",
        label: str = "tab:comparison",
        precision: int = 4,
    ) -> str:
        """Generate a LaTeX table comparing multiple methods side by side.

        ``methods_results`` maps method name → {metric_name: value}.
        Columns are methods; rows are metrics.
        """
        if not methods_results:
            return ""

        method_names = list(methods_results.keys())
        all_metric_keys: List[str] = []
        seen: set = set()
        for mdict in methods_results.values():
            for k in mdict:
                if k not in seen:
                    all_metric_keys.append(k)
                    seen.add(k)

        n_methods = len(method_names)
        col_spec = "l" + "r" * n_methods

        lines: List[str] = []
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append(f"\\caption{{{_escape_latex(caption)}}}")
        lines.append(f"\\label{{{label}}}")
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")

        header_cells = ["Metric"] + [_escape_latex(m) for m in method_names]
        lines.append(" & ".join(header_cells) + " \\\\")
        lines.append("\\midrule")

        for metric_key in all_metric_keys:
            row_values: List[str] = [_escape_latex(metric_key)]
            best_val: Optional[float] = None
            best_idx: Optional[int] = None

            raw_vals: List[Optional[float]] = []
            for m_name in method_names:
                v = methods_results[m_name].get(metric_key)
                raw_vals.append(v)

            # Determine best (highest) value for bolding
            finite_vals = [
                (i, v) for i, v in enumerate(raw_vals)
                if v is not None and math.isfinite(v)
            ]
            if finite_vals:
                best_idx, best_val = max(finite_vals, key=lambda t: t[1])

            for idx, v in enumerate(raw_vals):
                if v is None:
                    row_values.append("--")
                else:
                    formatted = _format_float(v, precision)
                    if idx == best_idx and len(finite_vals) > 1:
                        formatted = f"\\textbf{{{formatted}}}"
                    row_values.append(formatted)

            lines.append(" & ".join(row_values) + " \\\\")

        lines.append("\\bottomrule")
        lines.append(f"\\end{{tabular}}")
        lines.append("\\end{table}")
        return "\n".join(lines)

    # ---- convenience: dict export ----------------------------------------

    def results_to_dict(
        self,
        bound: Optional[BoundRatioResult] = None,
        pathway: Optional[PathwayRecallResult] = None,
        discovery: Optional[DiscoveryRatioResult] = None,
        overhead: Optional[OverheadResult] = None,
        smt: Optional[SMTStats] = None,
        cache: Optional[CacheStats] = None,
        convergence: Optional[ConvergenceMetrics] = None,
    ) -> Dict[str, float]:
        """Flatten selected result objects into a single dict for table gen."""
        out: Dict[str, float] = {}
        if bound is not None:
            out["bound_ratio"] = bound.ratio
            out["cb_width"] = bound.cb_width
            out["true_width"] = bound.true_width
        if pathway is not None:
            out["pathway_recall"] = pathway.recall
            out["pathway_precision"] = pathway.precision
            out["pathway_f1"] = pathway.f1
        if discovery is not None:
            out["discovery_ratio"] = discovery.ratio
            out["improvement_pct"] = discovery.improvement_pct
        if overhead is not None:
            out["overhead_ratio"] = overhead.ratio
            out["overhead_seconds"] = overhead.overhead_seconds
        if smt is not None:
            out["smt_mean"] = smt.mean
            out["smt_median"] = smt.median
            out["smt_max"] = float(smt.max)
            out["smt_total"] = float(smt.total)
        if cache is not None:
            out["cache_hit_rate"] = cache.hit_rate
            out["cache_efficiency"] = cache.efficiency_score
        if convergence is not None:
            out["convergence_rate"] = convergence.convergence_rate
            out["final_gap"] = convergence.final_gap
            out["r_squared"] = convergence.r_squared
        return out

    # ---- batch helpers ---------------------------------------------------

    def batch_bound_ratios(
        self,
        pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    ) -> List[BoundRatioResult]:
        """Compute bound ratios for a batch of (cb_bounds, true_bounds) pairs."""
        return [self.compute_bound_ratio(cb, tb) for cb, tb in pairs]

    def batch_discovery_ratios(
        self,
        pairs: List[Tuple[float, float]],
    ) -> List[DiscoveryRatioResult]:
        """Compute discovery ratios for a batch of (mcts, baseline) pairs."""
        return [self.compute_discovery_ratio(m, b) for m, b in pairs]

    def batch_overhead_ratios(
        self,
        pairs: List[Tuple[float, float]],
    ) -> List[OverheadResult]:
        """Compute overhead ratios for (verified, unverified) time pairs."""
        return [self.compute_overhead_ratio(v, u) for v, u in pairs]

    def full_evaluation(
        self,
        cb_bounds: Tuple[float, float],
        true_bounds: Tuple[float, float],
        discovered_pathways: List[tuple],
        planted_pathways: List[tuple],
        mcts_loss: float,
        baseline_loss: float,
        verified_time: float,
        unverified_time: float,
        smt_counts: List[int],
        cache_hits: int,
        cache_misses: int,
        cache_evictions: int,
        gap_history: Optional[List[float]] = None,
    ) -> Dict[str, object]:
        """Run all individual metrics and return a combined dict."""
        bound_res = self.compute_bound_ratio(cb_bounds, true_bounds)
        pathway_res = self.compute_pathway_recall(
            discovered_pathways, planted_pathways,
        )
        disc_res = self.compute_discovery_ratio(mcts_loss, baseline_loss)
        oh_res = self.compute_overhead_ratio(verified_time, unverified_time)
        smt_res = self.compute_smt_statistics(smt_counts)
        cache_res = self.compute_cache_statistics(
            cache_hits, cache_misses, cache_evictions,
        )

        conv_res: Optional[ConvergenceMetrics] = None
        if gap_history is not None and len(gap_history) >= 3:
            try:
                conv_res = self.compute_convergence_metrics(gap_history)
            except ValueError:
                pass

        summary = self.summary_statistics({
            "bound_ratios": [bound_res.ratio],
            "pathway_recalls": [pathway_res.recall],
            "discovery_ratios": [disc_res.ratio],
            "overhead_ratios": [oh_res.ratio],
        })

        flat = self.results_to_dict(
            bound=bound_res,
            pathway=pathway_res,
            discovery=disc_res,
            overhead=oh_res,
            smt=smt_res,
            cache=cache_res,
            convergence=conv_res,
        )

        return {
            "bound": bound_res,
            "pathway": pathway_res,
            "discovery": disc_res,
            "overhead": oh_res,
            "smt": smt_res,
            "cache": cache_res,
            "convergence": conv_res,
            "summary": summary,
            "flat": flat,
        }

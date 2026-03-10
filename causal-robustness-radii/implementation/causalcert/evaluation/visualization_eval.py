"""Evaluation result visualisation helpers.

Generates data structures for coverage plots, scalability curves,
ablation result plots, method comparison forest plots, runtime vs
accuracy tradeoff plots, power curves, calibration plots, and
publication-quality figure descriptions.

All functions return dicts or dataclasses that can be serialised to
JSON or fed directly into matplotlib / plotly.  They do NOT import
matplotlib themselves (zero hard dependency), but provide a
``render_matplotlib`` convenience when matplotlib is available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ===================================================================
# Data structures
# ===================================================================

@dataclass
class PlotSpec:
    """Generic specification for a single plot."""
    title: str
    xlabel: str
    ylabel: str
    series: List[Dict[str, Any]]
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None
    legend: bool = True
    figsize: Tuple[float, float] = (8, 5)


@dataclass
class FigureCollection:
    """A collection of related plots for a single report."""
    plots: List[PlotSpec]
    suptitle: str = ""


# ===================================================================
# 1.  Coverage plots
# ===================================================================

def coverage_plot(
    nominal_levels: Sequence[float],
    empirical_coverages: Sequence[float],
    *,
    method_name: str = "Estimator",
    se_values: Optional[Sequence[float]] = None,
) -> PlotSpec:
    """Coverage vs nominal level plot.

    The ideal line is y = x.
    """
    series: List[Dict[str, Any]] = []

    series.append({
        "x": list(nominal_levels),
        "y": list(nominal_levels),
        "label": "Ideal",
        "style": "dashed",
        "color": "gray",
    })

    main = {
        "x": list(nominal_levels),
        "y": list(empirical_coverages),
        "label": method_name,
        "style": "solid",
        "marker": "o",
    }
    if se_values is not None:
        main["yerr"] = list(se_values)
    series.append(main)

    return PlotSpec(
        title=f"Coverage: {method_name}",
        xlabel="Nominal Level",
        ylabel="Empirical Coverage",
        series=series,
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.05),
    )


def coverage_by_dgp_plot(
    dgp_names: Sequence[str],
    coverages: Sequence[float],
    nominal_level: float = 0.95,
) -> PlotSpec:
    """Bar chart of coverage by DGP."""
    series = [{
        "x": list(dgp_names),
        "y": list(coverages),
        "type": "bar",
        "label": "Coverage",
    }]
    annotations = [{
        "type": "hline",
        "y": nominal_level,
        "label": f"Nominal ({nominal_level})",
        "style": "dashed",
        "color": "red",
    }]
    return PlotSpec(
        title="Coverage by DGP",
        xlabel="DGP",
        ylabel="Empirical Coverage",
        series=series,
        annotations=annotations,
        ylim=(0.0, 1.05),
    )


# ===================================================================
# 2.  Scalability curves
# ===================================================================

def scalability_plot(
    graph_sizes: Sequence[int],
    runtimes: Dict[str, Sequence[float]],
) -> PlotSpec:
    """Runtime vs graph size for multiple methods."""
    series: List[Dict[str, Any]] = []
    for method, times in runtimes.items():
        series.append({
            "x": list(graph_sizes),
            "y": list(times),
            "label": method,
            "marker": "o",
        })
    return PlotSpec(
        title="Scalability: Runtime vs Graph Size",
        xlabel="Number of Nodes",
        ylabel="Runtime (seconds)",
        series=series,
    )


def scalability_log_plot(
    graph_sizes: Sequence[int],
    runtimes: Dict[str, Sequence[float]],
) -> PlotSpec:
    """Log-log scalability plot."""
    series: List[Dict[str, Any]] = []
    for method, times in runtimes.items():
        series.append({
            "x": [math.log10(max(s, 1)) for s in graph_sizes],
            "y": [math.log10(max(t, 1e-6)) for t in times],
            "label": method,
            "marker": "o",
        })
    return PlotSpec(
        title="Scalability (log-log)",
        xlabel="log₁₀(Nodes)",
        ylabel="log₁₀(Runtime)",
        series=series,
    )


# ===================================================================
# 3.  Ablation result plots
# ===================================================================

def ablation_bar_plot(
    component_names: Sequence[str],
    full_performance: float,
    ablated_performances: Sequence[float],
    *,
    metric_name: str = "Coverage",
) -> PlotSpec:
    """Ablation study bar plot showing effect of removing each component."""
    drops = [full_performance - p for p in ablated_performances]
    series = [{
        "x": list(component_names),
        "y": drops,
        "type": "bar",
        "label": f"Drop in {metric_name}",
    }]
    annotations = [{
        "type": "hline",
        "y": 0.0,
        "style": "solid",
        "color": "black",
    }]
    return PlotSpec(
        title=f"Ablation Study: Impact on {metric_name}",
        xlabel="Removed Component",
        ylabel=f"Δ {metric_name}",
        series=series,
        annotations=annotations,
    )


def ablation_heatmap(
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    values: np.ndarray,
    *,
    metric_name: str = "Performance",
) -> PlotSpec:
    """Heatmap for pairwise ablation results."""
    series = [{
        "type": "heatmap",
        "z": values.tolist(),
        "x_labels": list(col_labels),
        "y_labels": list(row_labels),
        "label": metric_name,
    }]
    return PlotSpec(
        title=f"Pairwise Ablation: {metric_name}",
        xlabel="Component B",
        ylabel="Component A",
        series=series,
    )


# ===================================================================
# 4.  Method comparison forest plots
# ===================================================================

def forest_plot(
    method_names: Sequence[str],
    estimates: Sequence[float],
    ci_lowers: Sequence[float],
    ci_uppers: Sequence[float],
    *,
    true_value: Optional[float] = None,
    title: str = "Method Comparison",
) -> PlotSpec:
    """Forest plot for comparing estimates across methods."""
    series: List[Dict[str, Any]] = []

    for i, method in enumerate(method_names):
        series.append({
            "x": [estimates[i]],
            "y": [i],
            "xerr_lo": [estimates[i] - ci_lowers[i]],
            "xerr_hi": [ci_uppers[i] - estimates[i]],
            "label": method,
            "type": "errorbar_horizontal",
            "marker": "s",
        })

    annotations: List[Dict[str, Any]] = []
    if true_value is not None:
        annotations.append({
            "type": "vline",
            "x": true_value,
            "label": "True Value",
            "style": "dashed",
            "color": "red",
        })

    return PlotSpec(
        title=title,
        xlabel="Estimate",
        ylabel="Method",
        series=series,
        annotations=annotations,
        figsize=(10, max(4, len(method_names) * 0.5)),
    )


def method_comparison_table(
    method_names: Sequence[str],
    metrics: Dict[str, Sequence[float]],
) -> Dict[str, List]:
    """Structured table for method comparison.

    Returns a dict that can be converted to a DataFrame.
    """
    table: Dict[str, List] = {"method": list(method_names)}
    for metric_name, values in metrics.items():
        table[metric_name] = list(values)
    return table


# ===================================================================
# 5.  Runtime vs accuracy tradeoff plots
# ===================================================================

def runtime_accuracy_plot(
    method_names: Sequence[str],
    runtimes: Sequence[float],
    accuracies: Sequence[float],
    *,
    runtime_unit: str = "seconds",
    accuracy_metric: str = "Coverage",
) -> PlotSpec:
    """Scatter plot of runtime vs accuracy for each method."""
    series: List[Dict[str, Any]] = []
    for i, method in enumerate(method_names):
        series.append({
            "x": [runtimes[i]],
            "y": [accuracies[i]],
            "label": method,
            "marker": "o",
            "type": "scatter",
            "annotate": method,
        })
    return PlotSpec(
        title=f"Runtime vs {accuracy_metric}",
        xlabel=f"Runtime ({runtime_unit})",
        ylabel=accuracy_metric,
        series=series,
    )


def pareto_frontier(
    runtimes: Sequence[float],
    accuracies: Sequence[float],
) -> List[int]:
    """Compute the Pareto frontier (indices of non-dominated points).

    A point is on the frontier if no other point is both faster and
    more accurate.
    """
    n = len(runtimes)
    is_pareto = [True] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if runtimes[j] <= runtimes[i] and accuracies[j] >= accuracies[i]:
                if runtimes[j] < runtimes[i] or accuracies[j] > accuracies[i]:
                    is_pareto[i] = False
                    break
    return [i for i in range(n) if is_pareto[i]]


# ===================================================================
# 6.  Power curve plots
# ===================================================================

def power_curve_plot(
    sample_sizes: Sequence[int],
    power_values: Dict[str, Sequence[float]],
    *,
    alpha: float = 0.05,
) -> PlotSpec:
    """Power curves as a function of sample size."""
    series: List[Dict[str, Any]] = []
    for method, powers in power_values.items():
        series.append({
            "x": list(sample_sizes),
            "y": list(powers),
            "label": method,
            "marker": "o",
        })

    annotations = [{
        "type": "hline",
        "y": 0.8,
        "label": "80% Power",
        "style": "dashed",
        "color": "gray",
    }, {
        "type": "hline",
        "y": alpha,
        "label": f"α = {alpha}",
        "style": "dotted",
        "color": "red",
    }]

    return PlotSpec(
        title="Power Analysis",
        xlabel="Sample Size",
        ylabel="Power (Rejection Rate)",
        series=series,
        annotations=annotations,
        ylim=(0.0, 1.05),
    )


def power_by_effect_size(
    effect_sizes: Sequence[float],
    power_values: Sequence[float],
    *,
    sample_size: int = 500,
) -> PlotSpec:
    """Power as a function of effect size."""
    series = [{
        "x": list(effect_sizes),
        "y": list(power_values),
        "label": f"n = {sample_size}",
        "marker": "o",
    }]
    return PlotSpec(
        title=f"Power vs Effect Size (n={sample_size})",
        xlabel="Effect Size",
        ylabel="Power",
        series=series,
        ylim=(0.0, 1.05),
    )


# ===================================================================
# 7.  Calibration plots
# ===================================================================

def calibration_plot(
    expected: Sequence[float],
    observed: Sequence[float],
    *,
    method_name: str = "Estimator",
    se_values: Optional[Sequence[float]] = None,
) -> PlotSpec:
    """Calibration plot: observed vs expected rates."""
    series: List[Dict[str, Any]] = [
        {
            "x": [0, 1],
            "y": [0, 1],
            "label": "Perfect Calibration",
            "style": "dashed",
            "color": "gray",
        },
    ]
    main = {
        "x": list(expected),
        "y": list(observed),
        "label": method_name,
        "marker": "o",
    }
    if se_values is not None:
        main["yerr"] = list(se_values)
    series.append(main)

    return PlotSpec(
        title=f"Calibration: {method_name}",
        xlabel="Expected Rate",
        ylabel="Observed Rate",
        series=series,
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )


def reliability_diagram(
    predicted_probs: np.ndarray,
    outcomes: np.ndarray,
    *,
    n_bins: int = 10,
) -> PlotSpec:
    """Reliability diagram (binned calibration)."""
    bins = np.linspace(0, 1, n_bins + 1)
    expected: List[float] = []
    observed: List[float] = []
    counts: List[int] = []

    for i in range(n_bins):
        mask = (predicted_probs >= bins[i]) & (predicted_probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        expected.append(float(np.mean(predicted_probs[mask])))
        observed.append(float(np.mean(outcomes[mask])))
        counts.append(int(mask.sum()))

    series: List[Dict[str, Any]] = [
        {"x": [0, 1], "y": [0, 1], "label": "Perfect", "style": "dashed", "color": "gray"},
        {"x": expected, "y": observed, "label": "Observed", "marker": "o"},
    ]

    return PlotSpec(
        title="Reliability Diagram",
        xlabel="Mean Predicted Probability",
        ylabel="Observed Frequency",
        series=series,
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )


# ===================================================================
# 8.  Publication-quality figure generation
# ===================================================================

def publication_figure(
    plots: List[PlotSpec],
    *,
    suptitle: str = "",
    layout: Optional[Tuple[int, int]] = None,
    output_format: str = "pdf",
    dpi: int = 300,
) -> FigureCollection:
    """Assemble multiple plots into a publication figure.

    Does not render—returns a :class:`FigureCollection` that can be
    serialised or rendered later.
    """
    return FigureCollection(plots=plots, suptitle=suptitle)


def render_matplotlib(
    fig_spec: FigureCollection,
    output_path: Optional[str] = None,
    *,
    dpi: int = 300,
) -> Any:
    """Render a FigureCollection using matplotlib (if available).

    Returns the matplotlib Figure object.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for render_matplotlib")

    n_plots = len(fig_spec.plots)
    if n_plots == 0:
        fig, ax = plt.subplots()
        return fig

    ncols = min(n_plots, 3)
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5 * nrows))
    if n_plots == 1:
        axes = [axes]
    elif nrows > 1:
        axes = axes.flatten()

    for idx, spec in enumerate(fig_spec.plots):
        if idx >= len(axes):
            break
        ax = axes[idx]
        ax.set_title(spec.title)
        ax.set_xlabel(spec.xlabel)
        ax.set_ylabel(spec.ylabel)

        for s in spec.series:
            stype = s.get("type", "line")
            if stype == "bar":
                ax.bar(s["x"], s["y"], label=s.get("label", ""))
            elif stype == "scatter":
                ax.scatter(s["x"], s["y"], label=s.get("label", ""), marker=s.get("marker", "o"))
            elif stype == "heatmap":
                im = ax.imshow(s["z"], aspect="auto")
                if "x_labels" in s:
                    ax.set_xticks(range(len(s["x_labels"])))
                    ax.set_xticklabels(s["x_labels"], rotation=45)
                if "y_labels" in s:
                    ax.set_yticks(range(len(s["y_labels"])))
                    ax.set_yticklabels(s["y_labels"])
            else:
                kwargs = {}
                if "marker" in s:
                    kwargs["marker"] = s["marker"]
                if "style" in s:
                    kwargs["linestyle"] = s["style"]
                if "color" in s:
                    kwargs["color"] = s["color"]
                if "yerr" in s:
                    ax.errorbar(s["x"], s["y"], yerr=s["yerr"],
                                label=s.get("label", ""), **kwargs)
                else:
                    ax.plot(s["x"], s["y"], label=s.get("label", ""), **kwargs)

        for ann in spec.annotations:
            atype = ann.get("type", "")
            if atype == "hline":
                ax.axhline(y=ann["y"], linestyle=ann.get("style", "dashed"),
                           color=ann.get("color", "gray"), label=ann.get("label", ""))
            elif atype == "vline":
                ax.axvline(x=ann["x"], linestyle=ann.get("style", "dashed"),
                           color=ann.get("color", "gray"), label=ann.get("label", ""))

        if spec.xlim:
            ax.set_xlim(spec.xlim)
        if spec.ylim:
            ax.set_ylim(spec.ylim)
        if spec.legend:
            ax.legend(fontsize=8)

    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    if fig_spec.suptitle:
        fig.suptitle(fig_spec.suptitle, fontsize=14)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


def to_json_spec(fig_spec: FigureCollection) -> Dict[str, Any]:
    """Serialise a FigureCollection to a JSON-compatible dict."""
    plots_json: List[Dict[str, Any]] = []
    for spec in fig_spec.plots:
        plots_json.append({
            "title": spec.title,
            "xlabel": spec.xlabel,
            "ylabel": spec.ylabel,
            "series": spec.series,
            "annotations": spec.annotations,
            "xlim": spec.xlim,
            "ylim": spec.ylim,
            "legend": spec.legend,
            "figsize": spec.figsize,
        })
    return {
        "suptitle": fig_spec.suptitle,
        "plots": plots_json,
    }

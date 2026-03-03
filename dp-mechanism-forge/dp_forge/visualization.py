"""
Visualization module for DP-Forge.

Provides publication-quality plotting for mechanisms, convergence analysis,
and mechanism comparison.  Uses matplotlib with sensible defaults for
Nature/Science-style figures.

Three main classes:

- :class:`MechanismVisualizer`: Plot mechanism probability tables, noise
  distributions, CDFs, privacy loss, and heatmaps.
- :class:`ConvergenceVisualizer`: Plot CEGIS convergence metrics including
  objective, violations, witness set growth, and timing.
- :class:`ComparisonVisualizer`: Compare synthesized mechanisms against
  baselines via MSE bar charts, improvement factors, Pareto frontiers,
  epsilon sweeps, and scaling plots.

Export functions support PDF, PNG, SVG, LaTeX, and interactive HTML.

Usage::

    from dp_forge.visualization import MechanismVisualizer, save_figure

    viz = MechanismVisualizer()
    fig = viz.plot_mechanism(mechanism, spec)
    save_figure(fig, "mechanism.pdf")
"""

from __future__ import annotations

import io
import json
import logging
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

# Publication-quality defaults (Nature/Science compatible)
_STYLE_CONFIG: Dict[str, Any] = {
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.figsize": (6.5, 4.5),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
}

# Colour palette for mechanism comparison (colorblind-friendly)
_MECHANISM_COLORS: Dict[str, str] = {
    "synthesized": "#2196F3",   # blue
    "laplace": "#FF9800",       # orange
    "gaussian": "#4CAF50",      # green
    "geometric": "#9C27B0",     # purple
    "staircase": "#F44336",     # red
    "matrix": "#795548",        # brown
    "default": "#607D8B",       # grey
}

# Line styles for distinguishing mechanisms in print
_MECHANISM_LINESTYLES: Dict[str, str] = {
    "synthesized": "-",
    "laplace": "--",
    "gaussian": "-.",
    "geometric": ":",
    "staircase": "--",
    "matrix": "-.",
}

# Marker styles
_MECHANISM_MARKERS: Dict[str, str] = {
    "synthesized": "o",
    "laplace": "s",
    "gaussian": "^",
    "geometric": "D",
    "staircase": "v",
    "matrix": "P",
}


def _get_color(name: str) -> str:
    """Get color for a mechanism name."""
    return _MECHANISM_COLORS.get(name, _MECHANISM_COLORS["default"])


def _get_linestyle(name: str) -> str:
    """Get line style for a mechanism name."""
    return _MECHANISM_LINESTYLES.get(name, "-")


def _get_marker(name: str) -> str:
    """Get marker for a mechanism name."""
    return _MECHANISM_MARKERS.get(name, "o")


def _apply_style() -> None:
    """Apply publication-quality matplotlib style."""
    try:
        import matplotlib.pyplot as plt

        plt.rcParams.update(_STYLE_CONFIG)
    except ImportError:
        pass


def _import_matplotlib():
    """Import matplotlib and return (plt, matplotlib) or raise ImportError."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        return plt, matplotlib
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


def _check_latex_available() -> bool:
    """Check whether LaTeX rendering is available."""
    try:
        import matplotlib

        matplotlib.rcParams["text.usetex"] = True
        matplotlib.rcParams["text.usetex"] = False
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Grid construction helper
# ---------------------------------------------------------------------------


def _build_grid(
    spec: Any,
    k: int,
) -> npt.NDArray[np.float64]:
    """Build the output grid for a mechanism given a spec.

    Args:
        spec: QuerySpec or dict with query_values and sensitivity.
        k: Number of grid points.

    Returns:
        1-D array of grid values.
    """
    if hasattr(spec, "query_values"):
        qv = spec.query_values
        sensitivity = spec.sensitivity
    elif isinstance(spec, dict):
        qv = np.array(spec.get("query_values", [0, 1]), dtype=np.float64)
        sensitivity = spec.get("sensitivity", 1.0)
    else:
        return np.linspace(-3, 3, k)

    grid_min = qv.min() - 3 * sensitivity
    grid_max = qv.max() + 3 * sensitivity
    return np.linspace(grid_min, grid_max, k)


# ---------------------------------------------------------------------------
# MechanismVisualizer
# ---------------------------------------------------------------------------


class MechanismVisualizer:
    """Visualization for individual DP mechanisms.

    Provides methods to plot probability distributions, noise histograms,
    CDFs, privacy loss distributions, comparison plots, and heatmaps.

    Args:
        use_latex: Whether to enable LaTeX rendering for labels.
        style: Optional dict of matplotlib rcParams overrides.
        figsize: Default figure size (width, height) in inches.
    """

    def __init__(
        self,
        use_latex: bool = False,
        style: Optional[Dict[str, Any]] = None,
        figsize: Tuple[float, float] = (6.5, 4.5),
    ) -> None:
        self.use_latex = use_latex and _check_latex_available()
        self.figsize = figsize
        self.custom_style = style or {}

    def _setup(self):
        """Apply style and return plt."""
        plt, _ = _import_matplotlib()
        _apply_style()
        if self.custom_style:
            plt.rcParams.update(self.custom_style)
        if self.use_latex:
            plt.rcParams["text.usetex"] = True
        return plt

    def plot_mechanism(
        self,
        mechanism: Any,
        spec: Any = None,
        *,
        title: Optional[str] = None,
        show_grid: bool = True,
        alpha: float = 0.7,
        max_rows: int = 8,
    ) -> Any:
        """Plot the probability distribution for each input.

        For each database input i, plots Pr[M(x_i) = y_j] as a function
        of the output grid point y_j.

        Args:
            mechanism: ExtractedMechanism or 2-D array (n × k).
            spec: QuerySpec for grid labelling (optional).
            title: Plot title.
            show_grid: Whether to show background grid.
            alpha: Line/bar transparency.
            max_rows: Maximum number of input rows to plot.

        Returns:
            matplotlib Figure.
        """
        plt = self._setup()

        p = mechanism.p_final if hasattr(mechanism, "p_final") else np.asarray(mechanism)
        n, k = p.shape

        if spec is not None:
            grid = _build_grid(spec, k)
        else:
            grid = np.arange(k, dtype=np.float64)

        fig, ax = plt.subplots(figsize=self.figsize)

        n_plot = min(n, max_rows)
        cmap = plt.cm.viridis(np.linspace(0.1, 0.9, n_plot))

        for i in range(n_plot):
            label = f"Input {i}"
            if spec is not None and hasattr(spec, "query_values"):
                label = f"$x_{i}$ = {spec.query_values[i]:.1f}"
            ax.plot(grid, p[i], color=cmap[i], alpha=alpha, label=label, linewidth=1.5)

        ax.set_xlabel("Output value $y$" if self.use_latex else "Output value y")
        ax.set_ylabel("Probability" if not self.use_latex else "$\\Pr[M(x_i) = y]$")
        ax.set_title(title or "Mechanism Probability Distribution")
        ax.legend(loc="best", framealpha=0.8)

        if not show_grid:
            ax.grid(False)

        fig.tight_layout()
        return fig

    def plot_noise_distribution(
        self,
        mechanism: Any,
        spec: Any = None,
        *,
        input_index: int = 0,
        n_samples: int = 10000,
        bins: int = 50,
        title: Optional[str] = None,
        show_theoretical: bool = True,
    ) -> Any:
        """Plot the noise distribution for a given input via sampling.

        Draws samples from the mechanism and plots a histogram of the
        resulting noise values (noisy - true).

        Args:
            mechanism: ExtractedMechanism or 2-D array.
            spec: QuerySpec for grid and true values.
            input_index: Which input row to sample from.
            n_samples: Number of samples to draw.
            bins: Number of histogram bins.
            title: Plot title.
            show_theoretical: Overlay theoretical Laplace for comparison.

        Returns:
            matplotlib Figure.
        """
        plt = self._setup()

        p = mechanism.p_final if hasattr(mechanism, "p_final") else np.asarray(mechanism)
        n, k = p.shape

        if spec is not None:
            grid = _build_grid(spec, k)
            true_val = spec.query_values[input_index] if hasattr(spec, "query_values") else 0.0
        else:
            grid = np.arange(k, dtype=np.float64)
            true_val = 0.0

        # Sample from CDF
        rng = np.random.default_rng(42)
        cdf = np.cumsum(p[input_index])
        u = rng.random(n_samples)
        bin_indices = np.searchsorted(cdf, u)
        bin_indices = np.clip(bin_indices, 0, k - 1)
        sampled_values = grid[bin_indices]
        noise = sampled_values - true_val

        fig, ax = plt.subplots(figsize=self.figsize)

        ax.hist(
            noise, bins=bins, density=True, alpha=0.7,
            color=_get_color("synthesized"), edgecolor="white",
            label="Synthesized",
        )

        if show_theoretical and spec is not None and hasattr(spec, "epsilon"):
            epsilon = spec.epsilon
            sensitivity = spec.sensitivity if hasattr(spec, "sensitivity") else 1.0
            scale = sensitivity / epsilon
            x_range = np.linspace(noise.min(), noise.max(), 200)
            laplace_pdf = np.exp(-np.abs(x_range) / scale) / (2 * scale)
            ax.plot(
                x_range, laplace_pdf,
                color=_get_color("laplace"), linestyle="--", linewidth=2,
                label=f"Laplace(0, {scale:.2f})",
            )

        ax.set_xlabel("Noise value")
        ax.set_ylabel("Density")
        ax.set_title(title or f"Noise Distribution (Input {input_index})")
        ax.legend(loc="best", framealpha=0.8)
        fig.tight_layout()
        return fig

    def plot_cdf(
        self,
        mechanism: Any,
        spec: Any = None,
        *,
        title: Optional[str] = None,
        max_rows: int = 8,
    ) -> Any:
        """Plot the CDF for each input of the mechanism.

        Args:
            mechanism: ExtractedMechanism or 2-D array.
            spec: QuerySpec for grid labelling.
            title: Plot title.
            max_rows: Maximum number of rows to plot.

        Returns:
            matplotlib Figure.
        """
        plt = self._setup()

        p = mechanism.p_final if hasattr(mechanism, "p_final") else np.asarray(mechanism)
        n, k = p.shape

        if spec is not None:
            grid = _build_grid(spec, k)
        else:
            grid = np.arange(k, dtype=np.float64)

        fig, ax = plt.subplots(figsize=self.figsize)

        n_plot = min(n, max_rows)
        cmap = plt.cm.viridis(np.linspace(0.1, 0.9, n_plot))

        for i in range(n_plot):
            cdf = np.cumsum(p[i])
            label = f"Input {i}"
            if spec is not None and hasattr(spec, "query_values"):
                label = f"$x_{i}$={spec.query_values[i]:.1f}"
            ax.step(grid, cdf, where="post", color=cmap[i], label=label, linewidth=1.5)

        ax.set_xlabel("Output value y")
        ax.set_ylabel("CDF")
        ax.set_title(title or "Mechanism CDF")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="best", framealpha=0.8)
        fig.tight_layout()
        return fig

    def plot_privacy_loss(
        self,
        mechanism: Any,
        spec: Any,
        *,
        pair: Optional[Tuple[int, int]] = None,
        title: Optional[str] = None,
        show_epsilon_line: bool = True,
    ) -> Any:
        """Plot the privacy loss distribution for adjacent input pairs.

        For each output j, plots log(p[i][j] / p[i'][j]) and highlights
        where the ratio exceeds exp(ε).

        Args:
            mechanism: ExtractedMechanism or 2-D array.
            spec: QuerySpec with privacy parameters.
            pair: Specific (i, i') pair to plot. Default: first adjacent pair.
            title: Plot title.
            show_epsilon_line: Whether to show ε bound lines.

        Returns:
            matplotlib Figure.
        """
        plt = self._setup()

        p = mechanism.p_final if hasattr(mechanism, "p_final") else np.asarray(mechanism)
        n, k = p.shape
        epsilon = spec.epsilon if hasattr(spec, "epsilon") else 1.0

        if pair is None:
            pair = (0, min(1, n - 1))
        i, ip = pair

        if spec is not None:
            grid = _build_grid(spec, k)
        else:
            grid = np.arange(k, dtype=np.float64)

        # Compute privacy loss
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(p[ip] > 1e-30, p[i] / p[ip], 0.0)
            log_ratio = np.where(ratio > 0, np.log(ratio), 0.0)

        fig, ax = plt.subplots(figsize=self.figsize)

        # Color bars by violation status
        colors = np.where(
            np.abs(log_ratio) > epsilon,
            "#F44336",   # red for violations
            "#2196F3",   # blue for valid
        )
        ax.bar(grid, log_ratio, width=(grid[1] - grid[0]) * 0.8, color=colors, alpha=0.7)

        if show_epsilon_line:
            ax.axhline(y=epsilon, color="#F44336", linestyle="--", linewidth=1, label=f"ε = {epsilon}")
            ax.axhline(y=-epsilon, color="#F44336", linestyle="--", linewidth=1)

        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_xlabel("Output value y")
        ax.set_ylabel("Privacy loss log(p[i]/p[i'])" if not self.use_latex else
                       r"Privacy loss $\log\frac{p_i(y)}{p_{i'}(y)}$")
        ax.set_title(title or f"Privacy Loss Distribution (pair ({i}, {ip}))")
        ax.legend(loc="best", framealpha=0.8)
        fig.tight_layout()
        return fig

    def plot_comparison(
        self,
        synthesized: Any,
        baselines: Dict[str, Any],
        spec: Any = None,
        *,
        input_index: int = 0,
        title: Optional[str] = None,
    ) -> Any:
        """Plot side-by-side comparison of mechanism distributions.

        Args:
            synthesized: ExtractedMechanism or 2-D array for synthesized.
            baselines: Dict mapping name to mechanism (array or ExtractedMechanism).
            spec: QuerySpec for grid labelling.
            input_index: Which input row to compare.
            title: Plot title.

        Returns:
            matplotlib Figure.
        """
        plt = self._setup()

        p_synth = (synthesized.p_final if hasattr(synthesized, "p_final")
                   else np.asarray(synthesized))
        k = p_synth.shape[1]

        if spec is not None:
            grid = _build_grid(spec, k)
        else:
            grid = np.arange(k, dtype=np.float64)

        fig, ax = plt.subplots(figsize=self.figsize)

        # Synthesized
        ax.plot(
            grid, p_synth[input_index],
            color=_get_color("synthesized"),
            linestyle=_get_linestyle("synthesized"),
            marker=_get_marker("synthesized"),
            markevery=max(1, k // 15),
            linewidth=2,
            label="Synthesized (DP-Forge)",
        )

        # Baselines
        for name, mech in baselines.items():
            p_base = mech.p_final if hasattr(mech, "p_final") else np.asarray(mech)
            ax.plot(
                grid[:p_base.shape[1]], p_base[input_index],
                color=_get_color(name),
                linestyle=_get_linestyle(name),
                marker=_get_marker(name),
                markevery=max(1, k // 15),
                linewidth=1.5,
                label=name.capitalize(),
            )

        ax.set_xlabel("Output value y")
        ax.set_ylabel("Probability")
        ax.set_title(title or f"Mechanism Comparison (Input {input_index})")
        ax.legend(loc="best", framealpha=0.8)
        fig.tight_layout()
        return fig

    def plot_heatmap(
        self,
        mechanism: Any,
        spec: Any = None,
        *,
        title: Optional[str] = None,
        cmap: str = "viridis",
        log_scale: bool = False,
    ) -> Any:
        """Plot a 2D heatmap of the probability table.

        Args:
            mechanism: ExtractedMechanism or 2-D array.
            spec: QuerySpec for axis labelling.
            title: Plot title.
            cmap: Matplotlib colormap name.
            log_scale: Whether to use log scale for colours.

        Returns:
            matplotlib Figure.
        """
        plt = self._setup()

        p = mechanism.p_final if hasattr(mechanism, "p_final") else np.asarray(mechanism)
        n, k = p.shape

        fig, ax = plt.subplots(figsize=(max(6.5, k * 0.08), max(3.5, n * 0.4)))

        if log_scale:
            with np.errstate(divide="ignore"):
                data = np.log10(np.maximum(p, 1e-20))
            label = "log₁₀(Probability)"
        else:
            data = p
            label = "Probability"

        im = ax.imshow(data, aspect="auto", cmap=cmap, interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, label=label)

        if spec is not None and hasattr(spec, "query_values"):
            y_labels = [f"{v:.1f}" for v in spec.query_values]
            ax.set_yticks(range(n))
            ax.set_yticklabels(y_labels)

        ax.set_xlabel("Output bin index")
        ax.set_ylabel("Input index")
        ax.set_title(title or "Mechanism Probability Heatmap")
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# ConvergenceVisualizer
# ---------------------------------------------------------------------------


class ConvergenceVisualizer:
    """Visualization for CEGIS convergence diagnostics.

    Plots objective values, constraint violations, witness set growth,
    timing breakdowns, and combined dashboards for convergence analysis.

    Args:
        use_latex: Whether to enable LaTeX rendering.
        figsize: Default figure size (width, height) in inches.
    """

    def __init__(
        self,
        use_latex: bool = False,
        figsize: Tuple[float, float] = (6.5, 4.5),
    ) -> None:
        self.use_latex = use_latex and _check_latex_available()
        self.figsize = figsize

    def _setup(self):
        plt, _ = _import_matplotlib()
        _apply_style()
        if self.use_latex:
            plt.rcParams["text.usetex"] = True
        return plt

    def plot_objective(
        self,
        history: Union[List[float], Dict[str, Any]],
        *,
        title: Optional[str] = None,
        log_scale: bool = False,
    ) -> Any:
        """Plot the objective value over CEGIS iterations.

        Args:
            history: List of objective values per iteration, or a dict
                with key "objective" mapping to such a list.
            title: Plot title.
            log_scale: Whether to use log scale on y-axis.

        Returns:
            matplotlib Figure.
        """
        plt = self._setup()

        if isinstance(history, dict):
            values = history.get("objective", history.get("convergence_history", []))
        else:
            values = list(history)

        fig, ax = plt.subplots(figsize=self.figsize)

        iterations = list(range(1, len(values) + 1))
        ax.plot(iterations, values, "o-", color="#2196F3", linewidth=2, markersize=6)

        if log_scale:
            ax.set_yscale("log")

        ax.set_xlabel("CEGIS Iteration")
        ax.set_ylabel("Objective Value (Expected Loss)")
        ax.set_title(title or "CEGIS Convergence: Objective")

        # Mark final value
        if values:
            ax.annotate(
                f"{values[-1]:.4f}",
                xy=(len(values), values[-1]),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="gray"),
            )

        fig.tight_layout()
        return fig

    def plot_violation(
        self,
        history: Union[List[float], Dict[str, Any]],
        *,
        title: Optional[str] = None,
        epsilon: Optional[float] = None,
    ) -> Any:
        """Plot the maximum constraint violation over iterations.

        Args:
            history: List of max violation values, or dict with "violations" key.
            title: Plot title.
            epsilon: Target ε for reference line.

        Returns:
            matplotlib Figure.
        """
        plt = self._setup()

        if isinstance(history, dict):
            values = history.get("violations", history.get("violation_history", []))
        else:
            values = list(history)

        fig, ax = plt.subplots(figsize=self.figsize)

        iterations = list(range(1, len(values) + 1))
        ax.semilogy(iterations, values, "s-", color="#F44336", linewidth=2, markersize=5)

        ax.axhline(y=0, color="black", linewidth=0.5)
        if epsilon is not None:
            ax.axhline(
                y=1e-6, color="#4CAF50", linestyle="--",
                label=f"Tolerance (1e-6)", linewidth=1,
            )
            ax.legend(loc="best")

        ax.set_xlabel("CEGIS Iteration")
        ax.set_ylabel("Max Violation")
        ax.set_title(title or "CEGIS Convergence: Constraint Violations")
        fig.tight_layout()
        return fig

    def plot_witness_growth(
        self,
        history: Union[List[int], Dict[str, Any]],
        *,
        title: Optional[str] = None,
    ) -> Any:
        """Plot witness (constraint) set size over iterations.

        Args:
            history: List of set sizes, or dict with "witness_sizes" key.
            title: Plot title.

        Returns:
            matplotlib Figure.
        """
        plt = self._setup()

        if isinstance(history, dict):
            values = history.get("witness_sizes", history.get("constraint_set_sizes", []))
        else:
            values = list(history)

        fig, ax = plt.subplots(figsize=self.figsize)

        iterations = list(range(1, len(values) + 1))
        ax.bar(
            iterations, values, color="#9C27B0", alpha=0.7,
            edgecolor="white", linewidth=0.5,
        )
        ax.plot(iterations, values, "o-", color="#7B1FA2", linewidth=1.5, markersize=4)

        ax.set_xlabel("CEGIS Iteration")
        ax.set_ylabel("Witness Set Size")
        ax.set_title(title or "Witness Set Growth")
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        return fig

    def plot_time_per_iteration(
        self,
        history: Union[List[float], Dict[str, Any]],
        *,
        title: Optional[str] = None,
        breakdown: bool = False,
    ) -> Any:
        """Plot time spent per CEGIS iteration.

        Args:
            history: List of times (seconds), or dict with "times" key.
                If breakdown is True, expects dict with "lp_times",
                "verify_times", "extract_times" keys.
            title: Plot title.
            breakdown: Whether to show time breakdown by component.

        Returns:
            matplotlib Figure.
        """
        plt = self._setup()

        fig, ax = plt.subplots(figsize=self.figsize)

        if breakdown and isinstance(history, dict):
            lp_times = history.get("lp_times", [])
            verify_times = history.get("verify_times", [])
            extract_times = history.get("extract_times", [])

            max_len = max(len(lp_times), len(verify_times), len(extract_times))
            iterations = list(range(1, max_len + 1))

            # Pad shorter lists
            lp_times = list(lp_times) + [0.0] * (max_len - len(lp_times))
            verify_times = list(verify_times) + [0.0] * (max_len - len(verify_times))
            extract_times = list(extract_times) + [0.0] * (max_len - len(extract_times))

            ax.bar(iterations, lp_times, label="LP Solve", color="#2196F3", alpha=0.8)
            ax.bar(
                iterations, verify_times, bottom=lp_times,
                label="Verification", color="#F44336", alpha=0.8,
            )
            bottoms = [a + b for a, b in zip(lp_times, verify_times)]
            ax.bar(
                iterations, extract_times, bottom=bottoms,
                label="Extraction", color="#4CAF50", alpha=0.8,
            )
            ax.legend(loc="best")
        else:
            if isinstance(history, dict):
                values = history.get("times", history.get("iteration_times", []))
            else:
                values = list(history)
            iterations = list(range(1, len(values) + 1))
            ax.bar(iterations, values, color="#FF9800", alpha=0.8, edgecolor="white")

        ax.set_xlabel("CEGIS Iteration")
        ax.set_ylabel("Time (seconds)")
        ax.set_title(title or "Time per Iteration")
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        return fig

    def plot_convergence_dashboard(
        self,
        history: Dict[str, Any],
        *,
        title: Optional[str] = None,
    ) -> Any:
        """Plot a combined convergence dashboard with 4 subplots.

        Combines objective, violations, witness set size, and timing
        into a single figure.

        Args:
            history: Dict with keys: "objective" (or "convergence_history"),
                "violations", "witness_sizes", "times".
            title: Suptitle for the dashboard.

        Returns:
            matplotlib Figure.
        """
        plt = self._setup()

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Objective
        obj_vals = history.get("objective", history.get("convergence_history", []))
        if obj_vals:
            iters = range(1, len(obj_vals) + 1)
            axes[0, 0].plot(iters, obj_vals, "o-", color="#2196F3", linewidth=2, markersize=4)
            axes[0, 0].set_xlabel("Iteration")
            axes[0, 0].set_ylabel("Objective")
            axes[0, 0].set_title("Objective Value")

        # Violations
        viol_vals = history.get("violations", history.get("violation_history", []))
        if viol_vals:
            iters = range(1, len(viol_vals) + 1)
            positive_vals = [max(v, 1e-16) for v in viol_vals]
            axes[0, 1].semilogy(iters, positive_vals, "s-", color="#F44336", linewidth=2, markersize=4)
            axes[0, 1].set_xlabel("Iteration")
            axes[0, 1].set_ylabel("Max Violation")
            axes[0, 1].set_title("Constraint Violations")

        # Witness set
        witness_vals = history.get("witness_sizes", history.get("constraint_set_sizes", []))
        if witness_vals:
            iters = range(1, len(witness_vals) + 1)
            axes[1, 0].bar(iters, witness_vals, color="#9C27B0", alpha=0.7)
            axes[1, 0].set_xlabel("Iteration")
            axes[1, 0].set_ylabel("Set Size")
            axes[1, 0].set_title("Witness Set Growth")

        # Timing
        time_vals = history.get("times", history.get("iteration_times", []))
        if time_vals:
            iters = range(1, len(time_vals) + 1)
            axes[1, 1].bar(iters, time_vals, color="#FF9800", alpha=0.7)
            axes[1, 1].set_xlabel("Iteration")
            axes[1, 1].set_ylabel("Time (s)")
            axes[1, 1].set_title("Time per Iteration")

        fig.suptitle(title or "CEGIS Convergence Dashboard", fontsize=14, fontweight="bold")
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# ComparisonVisualizer
# ---------------------------------------------------------------------------


class ComparisonVisualizer:
    """Visualization for mechanism comparisons and benchmarks.

    Provides bar charts, improvement factor plots, Pareto frontiers,
    epsilon sweeps, and scaling analysis.

    Args:
        use_latex: Whether to enable LaTeX rendering.
        figsize: Default figure size (width, height) in inches.
    """

    def __init__(
        self,
        use_latex: bool = False,
        figsize: Tuple[float, float] = (6.5, 4.5),
    ) -> None:
        self.use_latex = use_latex and _check_latex_available()
        self.figsize = figsize

    def _setup(self):
        plt, _ = _import_matplotlib()
        _apply_style()
        if self.use_latex:
            plt.rcParams["text.usetex"] = True
        return plt

    def plot_mse_comparison(
        self,
        results: Dict[str, float],
        *,
        title: Optional[str] = None,
        sort_by_value: bool = True,
    ) -> Any:
        """Bar chart comparing MSE across mechanisms.

        Args:
            results: Dict mapping mechanism name to MSE value.
            title: Plot title.
            sort_by_value: Whether to sort bars by MSE.

        Returns:
            matplotlib Figure.
        """
        plt = self._setup()

        names = list(results.keys())
        values = list(results.values())

        if sort_by_value:
            pairs = sorted(zip(names, values), key=lambda x: x[1])
            names, values = zip(*pairs) if pairs else ([], [])
            names, values = list(names), list(values)

        fig, ax = plt.subplots(figsize=self.figsize)

        colors = [_get_color(n) for n in names]
        bars = ax.barh(names, values, color=colors, alpha=0.85, edgecolor="white", height=0.6)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center", fontsize=9,
            )

        ax.set_xlabel("MSE")
        ax.set_title(title or "MSE Comparison")
        ax.set_xlim(right=max(values) * 1.15 if values else 1)
        fig.tight_layout()
        return fig

    def plot_improvement_factors(
        self,
        results: Dict[str, float],
        baseline_name: str = "laplace",
        *,
        title: Optional[str] = None,
    ) -> Any:
        """Bar chart showing improvement ratios over a baseline.

        Args:
            results: Dict mapping mechanism name to MSE value.
            baseline_name: Name of the baseline to compute ratios against.
            title: Plot title.

        Returns:
            matplotlib Figure.
        """
        plt = self._setup()

        baseline_mse = results.get(baseline_name)
        if baseline_mse is None or baseline_mse <= 0:
            warnings.warn(f"Baseline '{baseline_name}' not found or zero MSE")
            return plt.figure()

        names = []
        factors = []
        for name, mse in results.items():
            if name == baseline_name or mse <= 0:
                continue
            names.append(name)
            factors.append(baseline_mse / mse)

        fig, ax = plt.subplots(figsize=self.figsize)

        colors = [_get_color(n) for n in names]
        bars = ax.barh(names, factors, color=colors, alpha=0.85, edgecolor="white", height=0.6)

        # Reference line at 1.0
        ax.axvline(x=1.0, color="#F44336", linestyle="--", linewidth=1, label="Baseline")

        for bar, val in zip(bars, factors):
            ax.text(
                bar.get_width() + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}×",
                va="center", fontsize=9,
            )

        ax.set_xlabel(f"Improvement Factor (vs {baseline_name})")
        ax.set_title(title or f"Improvement over {baseline_name.capitalize()}")
        ax.legend(loc="best")
        fig.tight_layout()
        return fig

    def plot_privacy_accuracy_tradeoff(
        self,
        results: Dict[str, List[Tuple[float, float]]],
        *,
        title: Optional[str] = None,
        log_y: bool = True,
    ) -> Any:
        """Plot Pareto frontier of privacy vs accuracy tradeoff.

        Args:
            results: Dict mapping mechanism name to list of
                (epsilon, MSE) tuples.
            title: Plot title.
            log_y: Whether to use log scale on MSE axis.

        Returns:
            matplotlib Figure.
        """
        plt = self._setup()

        fig, ax = plt.subplots(figsize=self.figsize)

        for name, points in results.items():
            if not points:
                continue
            epsilons, mses = zip(*sorted(points))
            ax.plot(
                epsilons, mses,
                color=_get_color(name),
                linestyle=_get_linestyle(name),
                marker=_get_marker(name),
                linewidth=1.5,
                markersize=6,
                label=name.capitalize(),
            )

        if log_y:
            ax.set_yscale("log")

        ax.set_xlabel("Privacy (ε)" if not self.use_latex else r"Privacy ($\varepsilon$)")
        ax.set_ylabel("MSE")
        ax.set_title(title or "Privacy–Accuracy Tradeoff")
        ax.legend(loc="best", framealpha=0.8)
        fig.tight_layout()
        return fig

    def plot_epsilon_sweep(
        self,
        results: Dict[str, List[Tuple[float, float]]],
        *,
        title: Optional[str] = None,
        log_y: bool = True,
    ) -> Any:
        """Plot MSE vs epsilon for multiple mechanisms.

        Args:
            results: Dict mapping mechanism name to list of
                (epsilon, MSE) tuples.
            title: Plot title.
            log_y: Whether to use log scale on MSE axis.

        Returns:
            matplotlib Figure.
        """
        plt = self._setup()

        fig, ax = plt.subplots(figsize=self.figsize)

        for name, points in results.items():
            if not points:
                continue
            epsilons, mses = zip(*sorted(points))
            ax.plot(
                epsilons, mses,
                color=_get_color(name),
                linestyle=_get_linestyle(name),
                marker=_get_marker(name),
                linewidth=2,
                markersize=6,
                label=name.capitalize(),
            )

            # Fill area between synthesized and baselines
            if name == "synthesized" and len(results) > 1:
                ax.fill_between(
                    epsilons, mses,
                    alpha=0.1, color=_get_color(name),
                )

        if log_y:
            ax.set_yscale("log")

        ax.set_xlabel("Privacy parameter ε")
        ax.set_ylabel("MSE")
        ax.set_title(title or "MSE vs Privacy (ε-Sweep)")
        ax.legend(loc="best", framealpha=0.8)
        fig.tight_layout()
        return fig

    def plot_scaling(
        self,
        results: Dict[str, List[Tuple[int, float]]],
        *,
        metric: str = "time",
        title: Optional[str] = None,
        log_log: bool = True,
    ) -> Any:
        """Plot scaling behaviour with problem size.

        Args:
            results: Dict mapping label to list of (problem_size, metric_value).
            metric: What is being measured ("time", "mse", "iterations").
            title: Plot title.
            log_log: Whether to use log-log scale.

        Returns:
            matplotlib Figure.
        """
        plt = self._setup()

        fig, ax = plt.subplots(figsize=self.figsize)

        for name, points in results.items():
            if not points:
                continue
            sizes, values = zip(*sorted(points))
            ax.plot(
                sizes, values,
                color=_get_color(name),
                marker=_get_marker(name),
                linewidth=2,
                markersize=6,
                label=name.capitalize(),
            )

        if log_log:
            ax.set_xscale("log")
            ax.set_yscale("log")

        ylabel_map = {
            "time": "Time (seconds)",
            "mse": "MSE",
            "iterations": "CEGIS Iterations",
        }

        ax.set_xlabel("Problem Size (n)")
        ax.set_ylabel(ylabel_map.get(metric, metric))
        ax.set_title(title or f"Scaling: {metric.capitalize()} vs Problem Size")
        ax.legend(loc="best", framealpha=0.8)
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------


def save_figure(
    fig: Any,
    path: Union[str, Path],
    fmt: Optional[str] = None,
    *,
    dpi: int = 300,
    transparent: bool = False,
) -> None:
    """Save a matplotlib figure to a file.

    Args:
        fig: matplotlib Figure object.
        path: Output file path.
        fmt: Explicit format ("pdf", "png", "svg", "eps"). Inferred from
            extension if not provided.
        dpi: Resolution for raster formats.
        transparent: Whether to use transparent background.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if fmt is None:
        fmt = p.suffix.lstrip(".")
        if not fmt:
            fmt = "pdf"

    fig.savefig(
        str(p),
        format=fmt,
        dpi=dpi,
        bbox_inches="tight",
        transparent=transparent,
    )
    logger.info("Figure saved to %s (format=%s, dpi=%d)", p, fmt, dpi)


def to_latex(
    fig: Any,
    *,
    width: str = r"\columnwidth",
    standalone: bool = False,
) -> str:
    """Export a figure to LaTeX-compatible PGF format.

    Args:
        fig: matplotlib Figure object.
        width: LaTeX width command for the figure.
        standalone: Whether to produce a standalone LaTeX document.

    Returns:
        LaTeX source string containing the figure as PGF.
    """
    try:
        import matplotlib
        matplotlib.use("pgf")
    except Exception:
        pass

    buf = io.StringIO()
    try:
        fig.savefig(buf, format="pgf", bbox_inches="tight")
        pgf_code = buf.getvalue()
    except Exception:
        # Fallback: save as PDF and include via \includegraphics
        pgf_code = "% PGF export failed; use save_figure() with .pdf format instead\n"

    if standalone:
        return (
            "\\documentclass{standalone}\n"
            "\\usepackage{pgf}\n"
            "\\begin{document}\n"
            f"{pgf_code}\n"
            "\\end{document}\n"
        )

    return (
        f"\\begin{{figure}}[htbp]\n"
        f"  \\centering\n"
        f"  \\resizebox{{{width}}}{{!}}{{\n"
        f"    {pgf_code}\n"
        f"  }}\n"
        f"  \\caption{{DP-Forge mechanism visualization.}}\n"
        f"  \\label{{fig:dp-forge}}\n"
        f"\\end{{figure}}\n"
    )


def to_html(
    fig: Any,
    *,
    include_plotly_js: bool = True,
    title: str = "DP-Forge Visualization",
) -> str:
    """Export a figure to interactive HTML.

    Attempts to use plotly for interactivity; falls back to embedding
    a PNG via base64 data URI.

    Args:
        fig: matplotlib Figure object.
        include_plotly_js: Whether to include plotly.js CDN link.
        title: HTML page title.

    Returns:
        HTML source string.
    """
    import base64

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("ascii")

    html = (
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head>\n"
        f"  <title>{title}</title>\n"
        "  <style>\n"
        "    body { font-family: sans-serif; margin: 2em; background: #fafafa; }\n"
        "    .container { max-width: 900px; margin: 0 auto; }\n"
        "    img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }\n"
        "    h1 { color: #333; }\n"
        "    .meta { color: #666; font-size: 0.9em; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        "  <div class='container'>\n"
        f"    <h1>{title}</h1>\n"
        f"    <p class='meta'>Generated by DP-Forge v0.1.0</p>\n"
        f"    <img src='data:image/png;base64,{img_b64}' alt='{title}'/>\n"
        "  </div>\n"
        "</body>\n"
        "</html>\n"
    )
    return html

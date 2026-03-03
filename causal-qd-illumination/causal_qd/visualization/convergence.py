"""Convergence, operator selection, and archive evolution plots.

Provides:
  - plot_convergence_curves: multiple metrics over generations
  - plot_operator_selection_rates: adaptive operator rates over time
  - plot_archive_evolution: snapshots of archive filling over time
  - plot_archive_growth: archive size over iterations
  - plot_convergence: multi-metric convergence (legacy)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from causal_qd.archive.archive_base import Archive


class ConvergencePlotter:
    """Plots for monitoring optimisation convergence and archive growth."""

    # ------------------------------------------------------------------
    # Multi-metric convergence curves
    # ------------------------------------------------------------------

    @staticmethod
    def plot_convergence_curves(
        metrics_history: Dict[str, List[float]],
        smooth_window: int = 1,
        figsize: Tuple[float, float] = (10, 6),
        title: str = "Convergence Curves",
        log_scale: bool = False,
        ax: Optional[Axes] = None,
    ) -> Figure:
        """Plot multiple convergence curves on a single axes.

        Each key in *metrics_history* becomes a labeled line.
        Optionally smooth curves with a running average.

        Parameters
        ----------
        metrics_history :
            Mapping from metric name to list of values per generation.
        smooth_window :
            Window size for running-average smoothing (1 = no smoothing).
        figsize :
            Figure size.
        title :
            Plot title.
        log_scale :
            If *True*, use logarithmic y-axis.
        ax :
            Optional pre-existing axes.

        Returns
        -------
        Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        colors = plt.colormaps.get_cmap("tab10")

        for idx, (name, values) in enumerate(metrics_history.items()):
            color = colors(idx % 10)
            if smooth_window > 1 and len(values) >= smooth_window:
                kernel = np.ones(smooth_window) / smooth_window
                smoothed = np.convolve(values, kernel, mode="valid")
                ax.plot(range(len(smoothed)), smoothed, label=name,
                        linewidth=1.5, color=color)
                ax.plot(values, linewidth=0.3, alpha=0.3, color=color)
            else:
                ax.plot(values, label=name, linewidth=1.5, color=color)

        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Metric Value")
        ax.set_title(title)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)
        return fig

    # Legacy alias
    plot_convergence = plot_convergence_curves

    # ------------------------------------------------------------------
    # Operator selection rates
    # ------------------------------------------------------------------

    @staticmethod
    def plot_operator_selection_rates(
        operator_history: Dict[str, List[float]],
        figsize: Tuple[float, float] = (10, 5),
        title: str = "Operator Selection Rates",
        stacked: bool = True,
        ax: Optional[Axes] = None,
    ) -> Figure:
        """Plot adaptive operator selection rates over generations.

        Parameters
        ----------
        operator_history :
            Mapping from operator name to list of selection rates
            (probabilities or counts) per generation.
        figsize :
            Figure size.
        title :
            Plot title.
        stacked :
            If *True*, produce a stacked area plot.
        ax :
            Optional pre-existing axes.

        Returns
        -------
        Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        names = list(operator_history.keys())
        if not names:
            ax.set_title(title + " (no data)")
            return fig

        n_gens = len(operator_history[names[0]])
        x = list(range(n_gens))

        if stacked:
            data = np.array([operator_history[name] for name in names])
            ax.stackplot(x, data, labels=names, alpha=0.8)
        else:
            for name in names:
                ax.plot(x, operator_history[name], label=name, linewidth=1.5)

        ax.set_xlabel("Generation")
        ax.set_ylabel("Selection Rate" if stacked else "Rate / Count")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        if stacked:
            ax.set_ylim(0, 1.05)
        return fig

    # ------------------------------------------------------------------
    # Archive evolution snapshots
    # ------------------------------------------------------------------

    @staticmethod
    def plot_archive_evolution(
        snapshots: List[Dict[Tuple[int, ...], float]],
        grid_dims: Tuple[int, int] = (50, 50),
        n_panels: int = 4,
        cmap: str = "viridis",
        figsize: Tuple[float, float] = (16, 4),
    ) -> Figure:
        """Show archive snapshots at evenly-spaced generations.

        Each snapshot is a dict mapping ``CellIndex → quality``.

        Parameters
        ----------
        snapshots :
            List of archive snapshots (one per generation).
        grid_dims :
            Grid resolution for the heatmap.
        n_panels :
            Number of snapshots to display.
        cmap :
            Colormap name.
        figsize :
            Figure size.

        Returns
        -------
        Figure
        """
        n_total = len(snapshots)
        if n_total == 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No snapshots", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
            return fig

        indices = np.linspace(0, n_total - 1, min(n_panels, n_total), dtype=int)
        fig, axes = plt.subplots(1, len(indices), figsize=figsize)
        if len(indices) == 1:
            axes = [axes]

        # Find global quality range
        all_quals = []
        for snap in snapshots:
            all_quals.extend(snap.values())
        vmin = min(all_quals) if all_quals else 0
        vmax = max(all_quals) if all_quals else 1

        for panel_idx, gen_idx in enumerate(indices):
            ax = axes[panel_idx]
            snap = snapshots[gen_idx]
            grid = np.full(grid_dims, np.nan)
            for cell_idx, quality in snap.items():
                if len(cell_idx) >= 2:
                    r = min(cell_idx[0], grid_dims[0] - 1)
                    c = min(cell_idx[1], grid_dims[1] - 1)
                    grid[r, c] = quality
            im = ax.imshow(grid.T, origin="lower", aspect="auto",
                           cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"Gen {gen_idx}")
            ax.set_xticks([])
            ax.set_yticks([])

        fig.colorbar(im, ax=axes, label="Quality", shrink=0.8)
        fig.suptitle("Archive Evolution", fontsize=13)
        fig.tight_layout(rect=[0, 0, 0.92, 0.95])
        return fig

    # ------------------------------------------------------------------
    # Archive growth (cell count)
    # ------------------------------------------------------------------

    @staticmethod
    def plot_archive_growth(
        sizes: List[int],
        total_cells: Optional[int] = None,
        figsize: Tuple[float, float] = (8, 5),
        title: str = "Archive Growth",
        ax: Optional[Axes] = None,
    ) -> Figure:
        """Plot archive size (number of filled cells) over iterations.

        Parameters
        ----------
        sizes :
            Number of occupied cells at each iteration.
        total_cells :
            Total number of cells (for adding a ceiling line).
        figsize :
            Figure size.
        title :
            Plot title.
        ax :
            Optional pre-existing axes.

        Returns
        -------
        Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        ax.plot(sizes, linewidth=1.5, color="tab:green")
        ax.fill_between(range(len(sizes)), sizes, alpha=0.1, color="tab:green")

        if total_cells is not None:
            ax.axhline(y=total_cells, color="red", linestyle="--", linewidth=1,
                       label=f"Max cells ({total_cells})")
            ax.legend(fontsize=9)

        ax.set_xlabel("Generation")
        ax.set_ylabel("Occupied Cells")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return fig

    # ------------------------------------------------------------------
    # Summary panel
    # ------------------------------------------------------------------

    @staticmethod
    def plot_summary(
        metrics_history: Dict[str, List[float]],
        sizes: Optional[List[int]] = None,
        operator_history: Optional[Dict[str, List[float]]] = None,
        figsize: Tuple[float, float] = (14, 8),
    ) -> Figure:
        """Multi-panel summary with convergence, growth, and operators.

        Parameters
        ----------
        metrics_history :
            Metric histories for convergence plot.
        sizes :
            Archive sizes for growth plot.
        operator_history :
            Operator rates for selection-rate plot.
        figsize :
            Figure size.

        Returns
        -------
        Figure
        """
        n_panels = 1 + (sizes is not None) + (operator_history is not None)
        fig, axes = plt.subplots(1, n_panels, figsize=figsize)
        if n_panels == 1:
            axes = [axes]

        panel = 0
        ConvergencePlotter.plot_convergence_curves(metrics_history, ax=axes[panel])
        panel += 1

        if sizes is not None:
            ConvergencePlotter.plot_archive_growth(sizes, ax=axes[panel])
            panel += 1

        if operator_history is not None:
            ConvergencePlotter.plot_operator_selection_rates(
                operator_history, ax=axes[panel])

        fig.suptitle("Experiment Summary", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

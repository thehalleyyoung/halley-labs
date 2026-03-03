"""Archive visualization: heatmaps, coverage, QD-score, and descriptor plots.

Provides a comprehensive set of matplotlib-based visualizations for
MAP-Elites archives, including:
  - 2D heatmap of archive quality
  - 3D scatter plot of archive
  - Coverage over time
  - QD-score over time
  - Best quality over time
  - Descriptor distribution histograms
  - Cell occupancy heatmap
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from causal_qd.archive.archive_base import Archive, ArchiveEntry


class ArchivePlotter:
    """Produces standard plots for MAP-Elites archives.

    All methods are static so the class can be used without instantiation.
    Each plotting method returns a ``matplotlib.figure.Figure`` and
    optionally accepts pre-existing axes for embedding into larger
    figure layouts.
    """

    # ------------------------------------------------------------------
    # 2D Heatmap
    # ------------------------------------------------------------------

    @staticmethod
    def plot_archive_2d(
        archive: Archive,
        dims: Tuple[int, int] = (0, 1),
        cmap: str = "viridis",
        figsize: Tuple[float, float] = (8, 6),
        title: str = "Archive Quality Heatmap",
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        ax: Optional[Axes] = None,
    ) -> Figure:
        """Render a 2-D heatmap of archive quality across two descriptor dimensions.

        Parameters
        ----------
        archive :
            Archive exposing ``elites()`` and grid dimensions.
        dims :
            Pair of descriptor-dimension indices to project onto.
        cmap :
            Matplotlib colormap name.
        figsize :
            Figure size (width, height) in inches.
        title :
            Plot title.
        xlabel, ylabel :
            Axis labels.  Default to ``"Descriptor dim {d}"`` format.
        vmin, vmax :
            Color scale bounds.
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

        d0, d1 = dims
        elites = archive.elites() if callable(archive.elites) else list(archive.elites.values())

        if not elites:
            ax.set_title(title + " (empty archive)")
            return fig

        # Determine grid resolution from archive or from data
        grid_dims = getattr(archive, 'dims', None) or getattr(archive, '_dims', (50, 50))
        if isinstance(grid_dims, tuple) and len(grid_dims) >= 2:
            n0, n1 = grid_dims[d0], grid_dims[d1]
        else:
            n0, n1 = 50, 50

        grid = np.full((n0, n1), np.nan)

        lower = getattr(archive, '_lower', None)
        upper = getattr(archive, '_upper', None)

        for entry in elites:
            desc = entry.descriptor
            if lower is not None and upper is not None:
                normed = (desc - lower) / np.maximum(upper - lower, 1e-12)
                i0 = min(int(normed[d0] * n0), n0 - 1)
                i1 = min(int(normed[d1] * n1), n1 - 1)
            else:
                i0 = min(int(desc[d0] * n0), n0 - 1)
                i1 = min(int(desc[d1] * n1), n1 - 1)
            i0 = max(0, i0)
            i1 = max(0, i1)
            existing = grid[i0, i1]
            if np.isnan(existing) or entry.quality > existing:
                grid[i0, i1] = entry.quality

        im = ax.imshow(
            grid.T,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(im, ax=ax, label="Quality")
        ax.set_xlabel(xlabel or f"Descriptor dim {d0}")
        ax.set_ylabel(ylabel or f"Descriptor dim {d1}")
        ax.set_title(title)
        return fig

    # Legacy alias
    plot_heatmap = plot_archive_2d

    # ------------------------------------------------------------------
    # 3D Scatter
    # ------------------------------------------------------------------

    @staticmethod
    def plot_archive_3d(
        archive: Archive,
        dims: Tuple[int, int, int] = (0, 1, 2),
        cmap: str = "viridis",
        figsize: Tuple[float, float] = (10, 8),
        title: str = "Archive 3D Scatter",
        point_size: float = 20.0,
    ) -> Figure:
        """Render a 3D scatter plot of archive elites.

        The first two dimensions in *dims* are used as x and y,
        the third as z.  Color encodes quality.

        Parameters
        ----------
        archive :
            Archive with elites.
        dims :
            Three descriptor-dimension indices.
        cmap :
            Colormap name.
        figsize :
            Figure size.
        title :
            Plot title.
        point_size :
            Marker size.

        Returns
        -------
        Figure
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        elites = archive.elites() if callable(archive.elites) else list(archive.elites.values())
        if not elites:
            ax.set_title(title + " (empty)")
            return fig

        d0, d1, d2 = dims
        xs = [e.descriptor[d0] for e in elites]
        ys = [e.descriptor[d1] for e in elites]
        zs = [e.descriptor[d2] for e in elites]
        qs = [e.quality for e in elites]

        sc = ax.scatter(xs, ys, zs, c=qs, cmap=cmap, s=point_size, alpha=0.7)
        fig.colorbar(sc, ax=ax, label="Quality", shrink=0.6)
        ax.set_xlabel(f"Dim {d0}")
        ax.set_ylabel(f"Dim {d1}")
        ax.set_zlabel(f"Dim {d2}")
        ax.set_title(title)
        return fig

    # ------------------------------------------------------------------
    # Coverage over time
    # ------------------------------------------------------------------

    @staticmethod
    def plot_coverage_over_time(
        history: List[float],
        figsize: Tuple[float, float] = (8, 5),
        color: str = "tab:blue",
        title: str = "Archive Coverage Over Time",
        ax: Optional[Axes] = None,
    ) -> Figure:
        """Plot archive coverage fraction over generations.

        Parameters
        ----------
        history :
            Sequence of coverage values (one per generation/iteration).
        figsize :
            Figure size.
        color :
            Line color.
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

        ax.plot(history, linewidth=1.5, color=color)
        ax.fill_between(range(len(history)), history, alpha=0.15, color=color)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Coverage")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        return fig

    # Legacy alias
    plot_coverage = plot_coverage_over_time

    # ------------------------------------------------------------------
    # QD-score over time
    # ------------------------------------------------------------------

    @staticmethod
    def plot_qd_score_over_time(
        history: List[float],
        figsize: Tuple[float, float] = (8, 5),
        color: str = "tab:orange",
        title: str = "QD-Score Over Time",
        ax: Optional[Axes] = None,
    ) -> Figure:
        """Plot QD-score over generations.

        Parameters
        ----------
        history :
            Sequence of QD-score values.
        figsize :
            Figure size.
        color :
            Line color.
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

        ax.plot(history, linewidth=1.5, color=color)
        ax.fill_between(range(len(history)), history, alpha=0.15, color=color)
        ax.set_xlabel("Generation")
        ax.set_ylabel("QD-Score")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return fig

    # Legacy alias
    plot_qd_score = plot_qd_score_over_time

    # ------------------------------------------------------------------
    # Best quality over time
    # ------------------------------------------------------------------

    @staticmethod
    def plot_best_quality_over_time(
        history: List[float],
        figsize: Tuple[float, float] = (8, 5),
        color: str = "tab:green",
        title: str = "Best Quality Over Time",
        ax: Optional[Axes] = None,
    ) -> Figure:
        """Plot best quality (running max) over generations.

        Parameters
        ----------
        history :
            Sequence of best-quality values.
        figsize :
            Figure size.
        color :
            Line color.
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

        # Compute running max
        running_max = []
        best = -np.inf
        for v in history:
            best = max(best, v)
            running_max.append(best)

        ax.plot(running_max, linewidth=1.5, color=color)
        ax.plot(history, linewidth=0.5, alpha=0.4, color=color)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Quality")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(["Running max", "Per-generation"], fontsize=9)
        return fig

    # ------------------------------------------------------------------
    # Descriptor distribution
    # ------------------------------------------------------------------

    @staticmethod
    def plot_descriptor_distribution(
        archive: Archive,
        dim: int = 0,
        bins: int = 30,
        figsize: Tuple[float, float] = (8, 5),
        title: Optional[str] = None,
        ax: Optional[Axes] = None,
    ) -> Figure:
        """Histogram of descriptor values along a single dimension.

        Parameters
        ----------
        archive :
            Archive with elites.
        dim :
            Descriptor dimension index.
        bins :
            Number of histogram bins.
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

        elites = archive.elites() if callable(archive.elites) else list(archive.elites.values())
        values = [e.descriptor[dim] for e in elites]

        ax.hist(values, bins=bins, edgecolor="black", alpha=0.7, color="steelblue")
        ax.set_xlabel(f"Descriptor dim {dim}")
        ax.set_ylabel("Count")
        ax.set_title(title or f"Descriptor Distribution (dim {dim})")
        ax.grid(True, alpha=0.3)
        return fig

    # ------------------------------------------------------------------
    # Cell occupancy heatmap
    # ------------------------------------------------------------------

    @staticmethod
    def plot_cell_occupancy(
        archive: Archive,
        dims: Tuple[int, int] = (0, 1),
        figsize: Tuple[float, float] = (8, 6),
        title: str = "Cell Occupancy",
        ax: Optional[Axes] = None,
    ) -> Figure:
        """Heatmap showing which cells are filled (binary).

        Parameters
        ----------
        archive :
            Archive with elites.
        dims :
            Pair of descriptor-dimension indices.
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

        d0, d1 = dims
        grid_dims = getattr(archive, 'dims', None) or getattr(archive, '_dims', (50, 50))
        n0 = grid_dims[d0] if isinstance(grid_dims, tuple) and len(grid_dims) > d0 else 50
        n1 = grid_dims[d1] if isinstance(grid_dims, tuple) and len(grid_dims) > d1 else 50

        grid = np.zeros((n0, n1), dtype=np.float64)
        elites = archive.elites() if callable(archive.elites) else list(archive.elites.values())

        lower = getattr(archive, '_lower', None)
        upper = getattr(archive, '_upper', None)

        for entry in elites:
            desc = entry.descriptor
            if lower is not None and upper is not None:
                normed = (desc - lower) / np.maximum(upper - lower, 1e-12)
                i0 = min(max(int(normed[d0] * n0), 0), n0 - 1)
                i1 = min(max(int(normed[d1] * n1), 0), n1 - 1)
            else:
                i0 = min(max(int(desc[d0] * n0), 0), n0 - 1)
                i1 = min(max(int(desc[d1] * n1), 0), n1 - 1)
            grid[i0, i1] = 1.0

        im = ax.imshow(grid.T, origin="lower", aspect="auto", cmap="binary")
        ax.set_xlabel(f"Descriptor dim {d0}")
        ax.set_ylabel(f"Descriptor dim {d1}")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="Occupied")
        return fig

    # ------------------------------------------------------------------
    # Multi-panel summary
    # ------------------------------------------------------------------

    @staticmethod
    def plot_summary(
        coverage_history: List[float],
        qd_score_history: List[float],
        best_quality_history: List[float],
        archive: Optional[Archive] = None,
        figsize: Tuple[float, float] = (16, 10),
    ) -> Figure:
        """Generate a multi-panel summary figure.

        Creates a 2×2 grid with coverage, QD-score, best quality,
        and (if archive provided) the archive heatmap.

        Parameters
        ----------
        coverage_history :
            Coverage values per generation.
        qd_score_history :
            QD-score values per generation.
        best_quality_history :
            Best-quality values per generation.
        archive :
            Optional archive for the heatmap panel.
        figsize :
            Figure size.

        Returns
        -------
        Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        ArchivePlotter.plot_coverage_over_time(coverage_history, ax=axes[0, 0])
        ArchivePlotter.plot_qd_score_over_time(qd_score_history, ax=axes[0, 1])
        ArchivePlotter.plot_best_quality_over_time(best_quality_history, ax=axes[1, 0])

        if archive is not None:
            ArchivePlotter.plot_archive_2d(archive, ax=axes[1, 1])
        else:
            axes[1, 1].text(0.5, 0.5, "No archive data", ha="center", va="center",
                            transform=axes[1, 1].transAxes, fontsize=14)
            axes[1, 1].set_title("Archive Heatmap")

        fig.suptitle("CausalQD Experiment Summary", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    # ------------------------------------------------------------------
    # Save helper
    # ------------------------------------------------------------------

    @staticmethod
    def save(fig: Figure, path: str, dpi: int = 150) -> None:
        """Save a figure to disk.

        Parameters
        ----------
        fig :
            The matplotlib figure to save.
        path :
            Destination file path.
        dpi :
            Resolution in dots per inch.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

"""Phase diagram visualization.

Provides classes for rendering 2D phase diagrams with colored regime
regions, boundary curves with confidence bands, parameter sweeps,
and multi-panel comparisons.

Key classes
-----------
- PlotConfig: global styling configuration
- PhaseColorMap: regime → color mapping with ListedColormap support
- PhaseDiagramPlotter: main plotting interface for phase diagrams
"""

from __future__ import annotations

# ======================================================================
# Imports
# ======================================================================

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from numpy.typing import NDArray


# ======================================================================
# Configuration
# ======================================================================


@dataclass
class PlotConfig:
    """Global plotting configuration.

    Parameters
    ----------
    figsize : tuple[float, float]
        Default figure size in inches.
    dpi : int
        Resolution for saved figures.
    colormap : str
        Default matplotlib colormap name.
    font_size : int
        Base font size for labels and titles.
    line_width : float
        Default line width.
    alpha : float
        Default opacity for filled regions.
    save_path : str or None
        If provided, figures are saved here by default.
    show_colorbar : bool
        Whether to show colorbars on heatmap-style plots.
    show_legend : bool
        Whether to show legends.
    """

    figsize: Tuple[float, float] = (8, 6)
    dpi: int = 150
    colormap: str = "viridis"
    font_size: int = 12
    line_width: float = 1.5
    alpha: float = 0.6
    save_path: Optional[str] = None
    show_colorbar: bool = True
    show_legend: bool = True

    def __post_init__(self) -> None:
        if self.dpi < 1:
            raise ValueError(f"dpi must be positive, got {self.dpi}")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")


# ======================================================================
# Color mapping
# ======================================================================


class PhaseColorMap:
    """Map regime labels to colors for phase diagram rendering.

    Parameters
    ----------
    regime_colors : dict[str, str]
        Mapping from regime label strings to hex color codes.
    """

    def __init__(self, regime_colors: Optional[Dict[str, str]] = None) -> None:
        if regime_colors is None:
            regime_colors = self.default_colors()
        self._colors: Dict[str, str] = regime_colors

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @staticmethod
    def default_colors() -> Dict[str, str]:
        """Return the canonical regime → color mapping.

        Returns
        -------
        dict[str, str]
            Default color map with LAZY, RICH, CHAOTIC, ORDERED.
        """
        return {
            "LAZY": "#3498db",
            "RICH": "#e74c3c",
            "CHAOTIC": "#f39c12",
            "ORDERED": "#2ecc71",
        }

    def get_color(self, regime: str) -> str:
        """Look up the color for *regime*, falling back to grey.

        Parameters
        ----------
        regime : str
            Regime label (e.g. ``"LAZY"``).

        Returns
        -------
        str
            Hex color code.
        """
        return self._colors.get(regime, "#95a5a6")

    def get_cmap(self) -> ListedColormap:
        """Build a :class:`ListedColormap` from the stored colors.

        Returns
        -------
        ListedColormap
            Colormap whose entries follow insertion order.
        """
        return ListedColormap(list(self._colors.values()))


# ======================================================================
# Phase-diagram plotter
# ======================================================================


class PhaseDiagramPlotter:
    """High-level plotter for 2D phase diagrams.

    Parameters
    ----------
    config : PlotConfig or None
        Plotting style.  Uses sensible defaults when *None*.
    color_map : PhaseColorMap or None
        Regime color mapping.
    """

    def __init__(
        self,
        config: Optional[PlotConfig] = None,
        color_map: Optional[PhaseColorMap] = None,
    ) -> None:
        self.config = config or PlotConfig()
        self.color_map = color_map or PhaseColorMap()

    # ------------------------------------------------------------------
    # Axes / figure helpers
    # ------------------------------------------------------------------

    def _get_ax(self, ax: Optional[Axes] = None) -> Axes:
        """Return *ax* or create a new figure + axes pair."""
        if ax is None:
            _, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        return ax

    def _apply_style(self, ax: Axes, xlabel: str, ylabel: str, title: str = "") -> None:
        """Set common axis labels, title, and font sizes."""
        ax.set_xlabel(xlabel, fontsize=self.config.font_size)
        ax.set_ylabel(ylabel, fontsize=self.config.font_size)
        if title:
            ax.set_title(title, fontsize=self.config.font_size + 2)
        ax.tick_params(labelsize=self.config.font_size - 2)

    # ------------------------------------------------------------------
    # Core phase-diagram plot
    # ------------------------------------------------------------------

    def plot_phase_diagram(self, diagram: Any, ax: Optional[Axes] = None) -> Axes:
        """Render a 2D phase diagram with colored regime regions.

        Fills each :class:`RegimeRegion` with its regime color, draws
        boundary curves, and overlays confidence bands as shaded regions
        around each boundary.

        Parameters
        ----------
        diagram : PhaseDiagram
            Phase diagram object containing ``regime_regions``,
            ``boundary_curves``, and ``parameter_names``.
        ax : Axes or None
            Matplotlib axes to draw on.  Created if *None*.

        Returns
        -------
        Axes
            The axes with the rendered phase diagram.
        """
        ax = self._get_ax(ax)

        # --- fill regime regions ---
        for region in diagram.regime_regions:
            label = region.label
            label_str = label.name if hasattr(label, "name") else str(label)
            color = self.color_map.get_color(label_str)

            for curve in region.boundary_curves:
                coords = curve.effective_coords()
                if coords is not None and len(coords) >= 3:
                    ax.fill(
                        coords[:, 0],
                        coords[:, 1],
                        color=color,
                        alpha=self.config.alpha,
                        label=label_str,
                    )

        # --- draw boundary curves ---
        for curve in diagram.boundary_curves:
            coords = curve.effective_coords()
            if coords is None or len(coords) < 2:
                continue
            ax.plot(
                coords[:, 0],
                coords[:, 1],
                color="k",
                linewidth=self.config.line_width,
                zorder=3,
            )

            # confidence band
            if curve.confidence_band_width is not None:
                band = curve.confidence_band_width
                normals = self._estimate_normals(coords)
                upper = coords + normals * band[:, np.newaxis]
                lower = coords - normals * band[:, np.newaxis]
                ax.fill(
                    np.concatenate([upper[:, 0], lower[::-1, 0]]),
                    np.concatenate([upper[:, 1], lower[::-1, 1]]),
                    color="k",
                    alpha=0.12,
                    zorder=2,
                )

        # --- labels & legend ---
        param_x, param_y = diagram.parameter_names
        self._apply_style(ax, param_x, param_y, title="Phase Diagram")

        if self.config.show_legend:
            self._add_regime_legend(ax, diagram)

        return ax

    # ------------------------------------------------------------------
    # Specialised diagram variants
    # ------------------------------------------------------------------

    def plot_lr_vs_width(self, diagram: Any, ax: Optional[Axes] = None) -> Axes:
        """Learning-rate vs width phase diagram on log-log axes.

        Parameters
        ----------
        diagram : PhaseDiagram
            Phase diagram with boundaries parameterised by learning rate
            and network width.
        ax : Axes or None
            Target axes.

        Returns
        -------
        Axes
        """
        ax = self._get_ax(ax)
        self._render_boundaries(ax, diagram)
        self._apply_style(ax, "Width (N)", "Learning Rate (η)", "LR vs Width")
        self._format_log_axis(ax, which="both")
        if self.config.show_legend:
            self._add_regime_legend(ax, diagram)
        return ax

    def plot_lr_vs_depth(self, diagram: Any, ax: Optional[Axes] = None) -> Axes:
        """Learning-rate vs depth phase diagram.

        Parameters
        ----------
        diagram : PhaseDiagram
            Phase diagram with boundaries parameterised by learning rate
            and network depth.
        ax : Axes or None
            Target axes.

        Returns
        -------
        Axes
        """
        ax = self._get_ax(ax)
        self._render_boundaries(ax, diagram)
        self._apply_style(ax, "Depth (L)", "Learning Rate (η)", "LR vs Depth")
        self._format_log_axis(ax, which="y")
        if self.config.show_legend:
            self._add_regime_legend(ax, diagram)
        return ax

    # ------------------------------------------------------------------
    # Comparison & multi-panel
    # ------------------------------------------------------------------

    def plot_comparison(
        self,
        predicted: Any,
        ground_truth: Any,
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Overlay predicted and ground-truth boundary curves.

        Parameters
        ----------
        predicted : PhaseDiagram
            Predicted phase diagram (drawn as solid lines).
        ground_truth : PhaseDiagram
            Ground truth (drawn as dashed lines).
        ax : Axes or None
            Target axes.

        Returns
        -------
        Axes
        """
        ax = self._get_ax(ax)

        for curve in predicted.boundary_curves:
            coords = curve.effective_coords()
            if coords is not None and len(coords) >= 2:
                ax.plot(
                    coords[:, 0],
                    coords[:, 1],
                    color="#2c3e50",
                    linewidth=self.config.line_width,
                    label="Predicted",
                )

        for curve in ground_truth.boundary_curves:
            coords = curve.effective_coords()
            if coords is not None and len(coords) >= 2:
                ax.plot(
                    coords[:, 0],
                    coords[:, 1],
                    color="#e74c3c",
                    linewidth=self.config.line_width,
                    linestyle="--",
                    label="Ground Truth",
                )

        param_x, param_y = predicted.parameter_names
        self._apply_style(ax, param_x, param_y, title="Predicted vs Ground Truth")
        if self.config.show_legend:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), fontsize=self.config.font_size - 2)
        return ax

    def plot_multi_panel(
        self,
        diagrams: Dict[str, Any],
        layout: Optional[Tuple[int, int]] = None,
    ) -> Figure:
        """Render multiple phase diagrams as a grid of subplots.

        Parameters
        ----------
        diagrams : dict[str, PhaseDiagram]
            Mapping from panel title to diagram.
        layout : tuple[int, int] or None
            ``(nrows, ncols)`` for the subplot grid.  Inferred when *None*.

        Returns
        -------
        Figure
            The matplotlib Figure containing all panels.
        """
        n = len(diagrams)
        if layout is None:
            ncols = min(n, 3)
            nrows = int(np.ceil(n / ncols))
            layout = (nrows, ncols)

        fig, axes = plt.subplots(
            *layout,
            figsize=(self.config.figsize[0] * layout[1], self.config.figsize[1] * layout[0]),
            dpi=self.config.dpi,
        )
        axes_flat = np.asarray(axes).ravel()

        for idx, (title, diagram) in enumerate(diagrams.items()):
            self.plot_phase_diagram(diagram, ax=axes_flat[idx])
            axes_flat[idx].set_title(title, fontsize=self.config.font_size)

        # hide unused panels
        for idx in range(n, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Order-parameter & sweep visualisations
    # ------------------------------------------------------------------

    def plot_order_parameter_field(
        self,
        sweep_result: Any,
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Color-coded order-parameter values across a parameter grid.

        Parameters
        ----------
        sweep_result : SweepResult
            Sweep with ``grid_points`` carrying ``order_parameter_value``
            and ``grid_shape``.
        ax : Axes or None
            Target axes.

        Returns
        -------
        Axes
        """
        ax = self._get_ax(ax)

        values = sweep_result.to_array()
        coords = sweep_result.coordinates_array()
        shape = sweep_result.grid_shape

        if len(shape) == 2:
            grid = values.reshape(shape)
            x_vals = np.unique(coords[:, 0])
            y_vals = np.unique(coords[:, 1])
            mappable = ax.pcolormesh(
                x_vals,
                y_vals,
                grid.T,
                cmap=self.config.colormap,
                shading="auto",
            )
        else:
            mappable = ax.scatter(
                coords[:, 0],
                coords[:, 1] if coords.shape[1] > 1 else np.zeros(len(coords)),
                c=values,
                cmap=self.config.colormap,
                s=20,
                alpha=self.config.alpha,
            )

        if self.config.show_colorbar:
            self._add_colorbar(ax, mappable, label="Order Parameter")

        names = sweep_result.parameter_names
        self._apply_style(
            ax,
            names[0] if len(names) > 0 else "Param 1",
            names[1] if len(names) > 1 else "Param 2",
            title="Order Parameter Field",
        )
        return ax

    def plot_boundary_with_confidence(
        self,
        curves: List[Any],
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Boundary curves with uncertainty bands.

        Parameters
        ----------
        curves : list[BoundaryCurve]
            Boundary curves with optional ``confidence_band_width``.
        ax : Axes or None
            Target axes.

        Returns
        -------
        Axes
        """
        ax = self._get_ax(ax)
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(curves), 1)))

        for i, curve in enumerate(curves):
            coords = curve.effective_coords()
            if coords is None or len(coords) < 2:
                continue
            color = colors[i % len(colors)]
            ax.plot(
                coords[:, 0],
                coords[:, 1],
                color=color,
                linewidth=self.config.line_width,
                label=f"Boundary {i + 1}",
            )

            if curve.confidence_band_width is not None:
                band = curve.confidence_band_width
                normals = self._estimate_normals(coords)
                upper = coords + normals * band[:, np.newaxis]
                lower = coords - normals * band[:, np.newaxis]
                ax.fill(
                    np.concatenate([upper[:, 0], lower[::-1, 0]]),
                    np.concatenate([upper[:, 1], lower[::-1, 1]]),
                    color=color,
                    alpha=0.15,
                )

        self._apply_style(ax, "Parameter 1", "Parameter 2", "Boundaries with Confidence")
        if self.config.show_legend:
            ax.legend(fontsize=self.config.font_size - 2)
        return ax

    def plot_sweep_grid(
        self,
        sweep_result: Any,
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Grid points colored by regime label.

        Parameters
        ----------
        sweep_result : SweepResult
            Sweep result with grid points carrying ``regime_label``.
        ax : Axes or None
            Target axes.

        Returns
        -------
        Axes
        """
        ax = self._get_ax(ax)

        coords = sweep_result.coordinates_array()
        labels = sweep_result.labels()

        unique_labels = sorted(set(labels))
        for label in unique_labels:
            mask = np.array([lb == label for lb in labels])
            color = self.color_map.get_color(label)
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1] if coords.shape[1] > 1 else np.zeros(mask.sum()),
                c=color,
                label=label,
                s=30,
                alpha=self.config.alpha,
                edgecolors="k",
                linewidths=0.3,
            )

        names = sweep_result.parameter_names
        self._apply_style(
            ax,
            names[0] if len(names) > 0 else "Param 1",
            names[1] if len(names) > 1 else "Param 2",
            title="Sweep Grid",
        )
        if self.config.show_legend:
            ax.legend(fontsize=self.config.font_size - 2)
        return ax

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _render_boundaries(self, ax: Axes, diagram: Any) -> None:
        """Draw boundary curves and regime fills for a diagram."""
        for region in diagram.regime_regions:
            label_str = region.label.name if hasattr(region.label, "name") else str(region.label)
            color = self.color_map.get_color(label_str)
            for curve in region.boundary_curves:
                coords = curve.effective_coords()
                if coords is not None and len(coords) >= 3:
                    ax.fill(
                        coords[:, 0],
                        coords[:, 1],
                        color=color,
                        alpha=self.config.alpha,
                    )

        for curve in diagram.boundary_curves:
            coords = curve.effective_coords()
            if coords is None or len(coords) < 2:
                continue
            ax.plot(
                coords[:, 0],
                coords[:, 1],
                color="k",
                linewidth=self.config.line_width,
                zorder=3,
            )

    def _estimate_normals(self, coords: NDArray[np.floating]) -> NDArray[np.floating]:
        """Estimate outward unit normals for an ordered point sequence.

        Parameters
        ----------
        coords : ndarray, shape (M, 2)
            Ordered boundary coordinates.

        Returns
        -------
        ndarray, shape (M, 2)
            Unit normal vectors at each point.
        """
        tangents = np.zeros_like(coords)
        tangents[1:-1] = coords[2:] - coords[:-2]
        tangents[0] = coords[1] - coords[0]
        tangents[-1] = coords[-1] - coords[-2]

        normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        return normals / norms

    def _add_regime_legend(self, ax: Axes, diagram: Any) -> None:
        """Add a legend with one patch per unique regime."""
        seen: Dict[str, str] = {}
        for region in diagram.regime_regions:
            label_str = region.label.name if hasattr(region.label, "name") else str(region.label)
            if label_str not in seen:
                seen[label_str] = self.color_map.get_color(label_str)
        handles = [
            Patch(facecolor=c, edgecolor="k", label=lab) for lab, c in seen.items()
        ]
        if handles:
            ax.legend(handles=handles, fontsize=self.config.font_size - 2, loc="best")

    def _add_colorbar(
        self,
        ax: Axes,
        mappable: matplotlib.cm.ScalarMappable,
        label: str,
    ) -> matplotlib.colorbar.Colorbar:
        """Attach a colorbar to *ax*.

        Parameters
        ----------
        ax : Axes
            Axes whose figure receives the colorbar.
        mappable : ScalarMappable
            The image or collection to take colour limits from.
        label : str
            Colorbar label text.

        Returns
        -------
        Colorbar
        """
        cbar = ax.figure.colorbar(mappable, ax=ax, pad=0.02)
        cbar.set_label(label, fontsize=self.config.font_size - 1)
        cbar.ax.tick_params(labelsize=self.config.font_size - 2)
        return cbar

    def _format_log_axis(self, ax: Axes, which: str = "both") -> None:
        """Set log scale on the requested axis.

        Parameters
        ----------
        ax : Axes
            Target axes.
        which : str
            ``'x'``, ``'y'``, or ``'both'``.
        """
        if which in ("x", "both"):
            ax.set_xscale("log")
        if which in ("y", "both"):
            ax.set_yscale("log")

    def save(self, fig: Figure, path: Optional[str] = None) -> None:
        """Save *fig* to disk.

        Parameters
        ----------
        fig : Figure
            Matplotlib figure to save.
        path : str or None
            File path.  Falls back to ``config.save_path``.
        """
        dest = path or self.config.save_path
        if dest is None:
            raise ValueError("No save path provided and config.save_path is None")
        fig.savefig(dest, dpi=self.config.dpi, bbox_inches="tight")

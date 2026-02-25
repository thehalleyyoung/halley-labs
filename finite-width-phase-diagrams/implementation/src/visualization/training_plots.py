"""Training dynamics visualization.

Provides loss curves, gradient norm evolution, order parameter
tracking, regime transition markers, and multi-seed summary plots
for neural network training experiments.

Key class
---------
- TrainingPlotter: all training-related plots
"""

from __future__ import annotations

# ======================================================================
# Imports
# ======================================================================

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .phase_plots import PlotConfig


# ======================================================================
# Training plotter
# ======================================================================


class TrainingPlotter:
    """Plotting utilities for training dynamics and regime analysis.

    Parameters
    ----------
    config : PlotConfig or None
        Shared plot styling.  Defaults are used when *None*.
    """

    def __init__(self, config: Optional[PlotConfig] = None) -> None:
        self.config = config or PlotConfig()

    # ------------------------------------------------------------------
    # Axes helpers
    # ------------------------------------------------------------------

    def _get_ax(self, ax: Optional[Axes] = None) -> Axes:
        """Return *ax* or create a new figure."""
        if ax is None:
            _, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        return ax

    def _apply_style(self, ax: Axes, xlabel: str, ylabel: str, title: str = "") -> None:
        """Set common labels and font sizes."""
        ax.set_xlabel(xlabel, fontsize=self.config.font_size)
        ax.set_ylabel(ylabel, fontsize=self.config.font_size)
        if title:
            ax.set_title(title, fontsize=self.config.font_size + 2)
        ax.tick_params(labelsize=self.config.font_size - 2)

    # ------------------------------------------------------------------
    # Loss curves
    # ------------------------------------------------------------------

    def plot_loss_curves(
        self,
        runs: List[Any],
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Loss vs epoch with mean and confidence-interval band across seeds.

        Parameters
        ----------
        runs : list[TrainingRun]
            Training runs, each containing ``measurements`` with
            ``epoch`` and ``loss`` attributes.
        ax : Axes or None
            Target axes.

        Returns
        -------
        Axes
        """
        ax = self._get_ax(ax)

        # collect per-run loss arrays
        all_losses: List[NDArray[np.floating]] = []
        epochs: Optional[NDArray[np.floating]] = None
        for run in runs:
            losses = np.array([m.loss for m in run.measurements])
            all_losses.append(losses)
            if epochs is None:
                epochs = np.array([m.epoch for m in run.measurements])

        if not all_losses or epochs is None:
            self._apply_style(ax, "Epoch", "Loss", "Loss Curves")
            return ax

        loss_matrix = np.array(all_losses)
        mean = np.mean(loss_matrix, axis=0)
        std = np.std(loss_matrix, axis=0)

        # individual runs (faded)
        for losses in all_losses:
            ax.plot(epochs, losses, color="#bdc3c7", linewidth=0.5, alpha=0.4)

        # mean ± CI
        ax.plot(epochs, mean, color="#2c3e50", linewidth=self.config.line_width, label="Mean")
        self._add_confidence_band(ax, epochs, mean, std, color="#2c3e50")

        ax.set_yscale("log")
        self._apply_style(ax, "Epoch", "Loss", "Loss Curves")
        if self.config.show_legend:
            ax.legend(fontsize=self.config.font_size - 2)
        return ax

    # ------------------------------------------------------------------
    # Gradient norms
    # ------------------------------------------------------------------

    def plot_gradient_norms(
        self,
        runs: List[Any],
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Gradient norm evolution across training runs.

        Parameters
        ----------
        runs : list[TrainingRun]
            Training runs with ``gradient_norm`` in measurements.
        ax : Axes or None
            Target axes.

        Returns
        -------
        Axes
        """
        ax = self._get_ax(ax)

        all_norms: List[NDArray[np.floating]] = []
        epochs: Optional[NDArray[np.floating]] = None
        for run in runs:
            norms = np.array([m.gradient_norm for m in run.measurements])
            all_norms.append(norms)
            if epochs is None:
                epochs = np.array([m.epoch for m in run.measurements])

        if not all_norms or epochs is None:
            self._apply_style(ax, "Epoch", "Gradient Norm", "Gradient Norms")
            return ax

        norm_matrix = np.array(all_norms)
        mean = np.mean(norm_matrix, axis=0)
        std = np.std(norm_matrix, axis=0)

        for norms in all_norms:
            ax.plot(epochs, norms, color="#bdc3c7", linewidth=0.5, alpha=0.4)

        ax.plot(epochs, mean, color="#e67e22", linewidth=self.config.line_width, label="Mean")
        self._add_confidence_band(ax, epochs, mean, std, color="#e67e22")

        ax.set_yscale("log")
        self._apply_style(ax, "Epoch", "‖∇L‖", "Gradient Norms")
        if self.config.show_legend:
            ax.legend(fontsize=self.config.font_size - 2)
        return ax

    # ------------------------------------------------------------------
    # Order-parameter evolution
    # ------------------------------------------------------------------

    def plot_order_parameter_evolution(
        self,
        order_params: NDArray[np.floating],
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Order parameter over time with regime transition markers.

        Parameters
        ----------
        order_params : ndarray, shape (T,)
            Order-parameter values at successive time steps.
        ax : Axes or None
            Target axes.

        Returns
        -------
        Axes
        """
        ax = self._get_ax(ax)
        timesteps = np.arange(len(order_params))

        ax.plot(
            timesteps,
            order_params,
            color="#8e44ad",
            linewidth=self.config.line_width,
        )

        # mark transitions
        transitions = self._detect_transition_points(order_params, threshold=0.3)
        if transitions:
            ax.scatter(
                [timesteps[i] for i in transitions],
                [order_params[i] for i in transitions],
                color="red",
                zorder=5,
                s=60,
                marker="D",
                label="Transitions",
            )

        self._apply_style(ax, "Time Step", "Order Parameter", "Order Parameter Evolution")
        if self.config.show_legend and transitions:
            ax.legend(fontsize=self.config.font_size - 2)
        return ax

    # ------------------------------------------------------------------
    # Regime transitions
    # ------------------------------------------------------------------

    def plot_regime_transitions(
        self,
        trajectory: NDArray[np.floating],
        transition_points: List[int],
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Plot trajectory with detected transition points marked.

        Parameters
        ----------
        trajectory : ndarray, shape (T,)
            Observable trajectory (e.g. order parameter).
        transition_points : list of int
            Indices of detected regime transitions.
        ax : Axes or None
            Target axes.

        Returns
        -------
        Axes
        """
        ax = self._get_ax(ax)
        timesteps = np.arange(len(trajectory))

        ax.plot(
            timesteps,
            trajectory,
            color="#34495e",
            linewidth=self.config.line_width,
        )

        # shade alternating regime segments
        boundaries = [0] + sorted(transition_points) + [len(trajectory) - 1]
        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
        for i in range(len(boundaries) - 1):
            ax.axvspan(
                boundaries[i],
                boundaries[i + 1],
                alpha=0.08,
                color=colors[i % len(colors)],
            )

        for tp in transition_points:
            ax.axvline(
                x=tp,
                color="red",
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
            )

        self._apply_style(ax, "Time Step", "Observable", "Regime Transitions")
        return ax

    # ------------------------------------------------------------------
    # Multi-seed summary
    # ------------------------------------------------------------------

    def plot_multi_seed_summary(
        self,
        result: Any,
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Summary plot with mean trajectory and faded individual runs.

        Parameters
        ----------
        result : GroundTruthResult
            Ground-truth result containing ``runs``,
            ``mean_trajectory``, and ``std_trajectory``.
        ax : Axes or None
            Target axes.

        Returns
        -------
        Axes
        """
        ax = self._get_ax(ax)

        # individual runs
        for run in result.runs:
            losses = np.array([m.loss for m in run.measurements])
            epochs = np.array([m.epoch for m in run.measurements])
            ax.plot(epochs, losses, color="#bdc3c7", linewidth=0.5, alpha=0.3)

        # mean ± std
        if result.mean_trajectory is not None:
            n_pts = len(result.mean_trajectory)
            epochs = np.arange(n_pts)
            ax.plot(
                epochs,
                result.mean_trajectory,
                color="#2c3e50",
                linewidth=self.config.line_width,
                label="Mean",
            )
            if result.std_trajectory is not None:
                self._add_confidence_band(
                    ax,
                    epochs,
                    result.mean_trajectory,
                    result.std_trajectory,
                    color="#2c3e50",
                )

        ax.set_yscale("log")
        self._apply_style(ax, "Epoch", "Loss", "Multi-Seed Summary")
        if self.config.show_legend:
            ax.legend(fontsize=self.config.font_size - 2)
        return ax

    # ------------------------------------------------------------------
    # Training dashboard
    # ------------------------------------------------------------------

    def plot_training_dashboard(self, result: Any) -> Figure:
        """Multi-panel figure: loss, gradient norm, alignment, order parameter.

        Parameters
        ----------
        result : GroundTruthResult
            Ground-truth result with training runs containing
            measurements for loss, gradient_norm, and kernel_alignment.

        Returns
        -------
        Figure
            Four-panel matplotlib figure.
        """
        fig, axes = plt.subplots(
            2, 2,
            figsize=(self.config.figsize[0] * 2, self.config.figsize[1] * 2),
            dpi=self.config.dpi,
        )

        # Panel 1: loss curves
        self.plot_loss_curves(result.runs, ax=axes[0, 0])

        # Panel 2: gradient norms
        self.plot_gradient_norms(result.runs, ax=axes[0, 1])

        # Panel 3: kernel alignment
        ax_align = axes[1, 0]
        for run in result.runs:
            alignments = np.array([m.kernel_alignment for m in run.measurements])
            epochs = np.array([m.epoch for m in run.measurements])
            ax_align.plot(epochs, alignments, color="#bdc3c7", linewidth=0.5, alpha=0.4)
        all_align = np.array([
            [m.kernel_alignment for m in run.measurements] for run in result.runs
        ])
        if all_align.size > 0:
            mean_a = np.mean(all_align, axis=0)
            epochs = np.array([m.epoch for m in result.runs[0].measurements])
            ax_align.plot(epochs, mean_a, color="#8e44ad", linewidth=self.config.line_width)
        self._apply_style(ax_align, "Epoch", "Kernel Alignment", "Alignment")

        # Panel 4: order parameter (use loss curvature as proxy)
        ax_op = axes[1, 1]
        if result.mean_trajectory is not None:
            op = np.gradient(np.gradient(result.mean_trajectory))
            timesteps = np.arange(len(op))
            self.plot_order_parameter_evolution(op, ax=ax_op)
        else:
            self._apply_style(ax_op, "Time Step", "Order Parameter", "Order Parameter")

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Regime comparison
    # ------------------------------------------------------------------

    def plot_regime_comparison(
        self,
        lazy_runs: List[Any],
        rich_runs: List[Any],
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Side-by-side comparison of lazy and rich regime loss curves.

        Parameters
        ----------
        lazy_runs : list[TrainingRun]
            Runs classified as lazy regime.
        rich_runs : list[TrainingRun]
            Runs classified as rich regime.
        ax : Axes or None
            Target axes.

        Returns
        -------
        Axes
        """
        ax = self._get_ax(ax)

        for label, runs, color in [
            ("Lazy", lazy_runs, "#3498db"),
            ("Rich", rich_runs, "#e74c3c"),
        ]:
            if not runs:
                continue
            all_losses = np.array([
                [m.loss for m in run.measurements] for run in runs
            ])
            epochs = np.array([m.epoch for m in runs[0].measurements])
            mean = np.mean(all_losses, axis=0)
            std = np.std(all_losses, axis=0)

            ax.plot(
                epochs,
                mean,
                color=color,
                linewidth=self.config.line_width,
                label=label,
            )
            self._add_confidence_band(ax, epochs, mean, std, color=color)

        ax.set_yscale("log")
        self._apply_style(ax, "Epoch", "Loss", "Lazy vs Rich")
        if self.config.show_legend:
            ax.legend(fontsize=self.config.font_size - 2)
        return ax

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _add_confidence_band(
        self,
        ax: Axes,
        x: NDArray[np.floating],
        mean: NDArray[np.floating],
        std: NDArray[np.floating],
        color: str,
        alpha: float = 0.2,
    ) -> None:
        """Add a shaded mean ± std confidence band.

        Parameters
        ----------
        ax : Axes
            Target axes.
        x : ndarray
            Horizontal coordinates.
        mean : ndarray
            Central values.
        std : ndarray
            Half-width of the band.
        color : str
            Band fill colour.
        alpha : float
            Band opacity.
        """
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=color,
            alpha=alpha,
        )

    def _detect_transition_points(
        self,
        order_params: NDArray[np.floating],
        threshold: float,
    ) -> List[int]:
        """Detect regime transitions via first-difference thresholding.

        Parameters
        ----------
        order_params : ndarray, shape (T,)
            Order-parameter time series.
        threshold : float
            Minimum absolute jump to count as a transition.

        Returns
        -------
        list of int
            Indices where transitions are detected.
        """
        diffs = np.abs(np.diff(np.asarray(order_params)))
        if len(diffs) == 0:
            return []
        median_diff = np.median(diffs)
        adaptive = max(threshold, 3.0 * median_diff)
        indices = np.where(diffs > adaptive)[0]
        return indices.tolist()

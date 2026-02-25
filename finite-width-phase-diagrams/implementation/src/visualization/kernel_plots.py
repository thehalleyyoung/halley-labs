"""Kernel visualization utilities.

Provides heatmaps, eigenvalue spectrum plots, alignment evolution,
perturbative validity maps, and convergence diagnostics for NTK
kernel matrices.

Key class
---------
- KernelPlotter: all kernel-related plots
"""

from __future__ import annotations

# ======================================================================
# Imports
# ======================================================================

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .phase_plots import PlotConfig


# ======================================================================
# Kernel plotter
# ======================================================================


class KernelPlotter:
    """Plotting utilities for NTK kernel matrices and spectra.

    Parameters
    ----------
    config : PlotConfig or None
        Shared plot styling.  Defaults are used when *None*.
    """

    def __init__(self, config: Optional[PlotConfig] = None) -> None:
        self.config = config or PlotConfig()

    # ------------------------------------------------------------------
    # Axes helper
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
    # NTK heatmap
    # ------------------------------------------------------------------

    def plot_ntk_heatmap(
        self,
        kernel_matrix: NDArray[np.floating],
        ax: Optional[Axes] = None,
        title: Optional[str] = None,
    ) -> Axes:
        """Heatmap of an NTK kernel matrix.

        Parameters
        ----------
        kernel_matrix : ndarray, shape (N, N)
            Symmetric kernel matrix.
        ax : Axes or None
            Target axes.
        title : str or None
            Plot title.  Defaults to ``'NTK Heatmap'``.

        Returns
        -------
        Axes
        """
        ax = self._get_ax(ax)
        title = title or "NTK Heatmap"

        mappable = ax.imshow(
            kernel_matrix,
            cmap=self.config.colormap,
            aspect="equal",
            interpolation="nearest",
        )

        if self.config.show_colorbar:
            cbar = ax.figure.colorbar(mappable, ax=ax, pad=0.02)
            cbar.set_label("K(x, x')", fontsize=self.config.font_size - 1)
            cbar.ax.tick_params(labelsize=self.config.font_size - 2)

        self._apply_style(ax, "Sample index", "Sample index", title)
        return ax

    # ------------------------------------------------------------------
    # Eigenvalue spectrum
    # ------------------------------------------------------------------

    def plot_eigenvalue_spectrum(
        self,
        eigenvalues: NDArray[np.floating],
        ax: Optional[Axes] = None,
        log_scale: bool = True,
    ) -> Axes:
        """Plot sorted eigenvalue decay.

        Parameters
        ----------
        eigenvalues : ndarray, shape (K,)
            Eigenvalues (will be sorted in descending order).
        ax : Axes or None
            Target axes.
        log_scale : bool
            Whether to use log scale on the y-axis.

        Returns
        -------
        Axes
        """
        ax = self._get_ax(ax)
        eigs_sorted = np.sort(np.asarray(eigenvalues).ravel())[::-1]
        indices = np.arange(1, len(eigs_sorted) + 1)

        ax.plot(
            indices,
            eigs_sorted,
            "o-",
            color="#2c3e50",
            markersize=4,
            linewidth=self.config.line_width,
        )

        if log_scale:
            ax.set_yscale("log")
            ax.set_xscale("log")

        self._apply_style(ax, "Index", "Eigenvalue", "Eigenvalue Spectrum")
        return ax

    # ------------------------------------------------------------------
    # Eigenvalue comparison
    # ------------------------------------------------------------------

    def plot_eigenvalue_comparison(
        self,
        eigenvalues_list: List[NDArray[np.floating]],
        labels: List[str],
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Compare multiple eigenvalue spectra on the same axes.

        Parameters
        ----------
        eigenvalues_list : list of ndarray
            Each entry is a 1-D array of eigenvalues.
        labels : list of str
            Legend labels for each spectrum.
        ax : Axes or None
            Target axes.

        Returns
        -------
        Axes
        """
        ax = self._get_ax(ax)
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(eigenvalues_list), 1)))

        for eigs, label, color in zip(eigenvalues_list, labels, colors):
            eigs_sorted = np.sort(np.asarray(eigs).ravel())[::-1]
            indices = np.arange(1, len(eigs_sorted) + 1)
            ax.plot(
                indices,
                eigs_sorted,
                "o-",
                color=color,
                markersize=3,
                linewidth=self.config.line_width,
                label=label,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        self._apply_style(ax, "Index", "Eigenvalue", "Eigenvalue Comparison")
        if self.config.show_legend:
            ax.legend(fontsize=self.config.font_size - 2)
        return ax

    # ------------------------------------------------------------------
    # Kernel alignment evolution
    # ------------------------------------------------------------------

    def plot_kernel_alignment_evolution(
        self,
        alignments: NDArray[np.floating],
        timesteps: Optional[NDArray[np.floating]] = None,
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Kernel alignment over training time.

        Parameters
        ----------
        alignments : ndarray, shape (T,)
            Alignment values at each measurement point.
        timesteps : ndarray or None
            Time / epoch values.  Uses ``arange`` when *None*.
        ax : Axes or None
            Target axes.

        Returns
        -------
        Axes
        """
        ax = self._get_ax(ax)
        if timesteps is None:
            timesteps = np.arange(len(alignments))

        ax.plot(
            timesteps,
            alignments,
            color="#8e44ad",
            linewidth=self.config.line_width,
        )
        ax.axhline(y=1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

        self._apply_style(ax, "Training Step", "Kernel Alignment", "Alignment Evolution")
        return ax

    # ------------------------------------------------------------------
    # Correction magnitudes
    # ------------------------------------------------------------------

    def plot_correction_magnitudes(
        self,
        corrections: Dict[str, float],
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Bar chart of 1/N correction magnitudes per component.

        Parameters
        ----------
        corrections : dict[str, float]
            Component name → correction magnitude.
        ax : Axes or None
            Target axes.

        Returns
        -------
        Axes
        """
        ax = self._get_ax(ax)
        names = list(corrections.keys())
        values = list(corrections.values())

        bars = ax.bar(
            np.arange(len(names)),
            values,
            color="#16a085",
            edgecolor="k",
            linewidth=0.5,
            alpha=self.config.alpha,
        )
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=self.config.font_size - 2)

        self._apply_style(ax, "", "Magnitude", "1/N Correction Magnitudes")
        return ax

    # ------------------------------------------------------------------
    # Perturbative validity map
    # ------------------------------------------------------------------

    def plot_perturbative_validity_map(
        self,
        validity_grid: NDArray[np.floating],
        param_ranges: Tuple[NDArray[np.floating], NDArray[np.floating]],
        ax: Optional[Axes] = None,
    ) -> Axes:
        """2D map showing where the perturbative expansion is valid.

        Parameters
        ----------
        validity_grid : ndarray, shape (Nx, Ny)
            Boolean or float array.  Values near 1 indicate validity.
        param_ranges : tuple of ndarray
            ``(x_values, y_values)`` defining the grid axes.
        ax : Axes or None
            Target axes.

        Returns
        -------
        Axes
        """
        ax = self._get_ax(ax)
        x_vals, y_vals = param_ranges

        mappable = ax.pcolormesh(
            x_vals,
            y_vals,
            validity_grid.T,
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            shading="auto",
        )

        if self.config.show_colorbar:
            cbar = ax.figure.colorbar(mappable, ax=ax, pad=0.02)
            cbar.set_label("Validity", fontsize=self.config.font_size - 1)
            cbar.ax.tick_params(labelsize=self.config.font_size - 2)

        self._apply_style(ax, "Parameter 1", "Parameter 2", "Perturbative Validity")
        return ax

    # ------------------------------------------------------------------
    # Kernel difference
    # ------------------------------------------------------------------

    def plot_kernel_difference(
        self,
        K1: NDArray[np.floating],
        K2: NDArray[np.floating],
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Heatmap of element-wise absolute difference |K1 − K2|.

        Parameters
        ----------
        K1, K2 : ndarray, shape (N, N)
            Two kernel matrices to compare.
        ax : Axes or None
            Target axes.

        Returns
        -------
        Axes
        """
        ax = self._get_ax(ax)
        diff = np.abs(np.asarray(K1) - np.asarray(K2))

        mappable = ax.imshow(
            diff,
            cmap="hot",
            aspect="equal",
            interpolation="nearest",
        )

        if self.config.show_colorbar:
            cbar = ax.figure.colorbar(mappable, ax=ax, pad=0.02)
            cbar.set_label("|K₁ − K₂|", fontsize=self.config.font_size - 1)
            cbar.ax.tick_params(labelsize=self.config.font_size - 2)

        self._apply_style(ax, "Sample index", "Sample index", "Kernel Difference")
        return ax

    # ------------------------------------------------------------------
    # Eigenvalue gap evolution
    # ------------------------------------------------------------------

    def plot_eigenvalue_gap_evolution(
        self,
        gaps: NDArray[np.floating],
        parameter_values: NDArray[np.floating],
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Track eigenvalue gap along a parameter path.

        Parameters
        ----------
        gaps : ndarray, shape (P,)
            Spectral gap values at each point.
        parameter_values : ndarray, shape (P,)
            Parameter values along the path.
        ax : Axes or None
            Target axes.

        Returns
        -------
        Axes
        """
        ax = self._get_ax(ax)

        ax.plot(
            parameter_values,
            gaps,
            color="#2980b9",
            linewidth=self.config.line_width,
        )
        ax.axhline(y=0, color="red", linestyle="--", linewidth=0.8, alpha=0.6)

        self._apply_style(ax, "Parameter", "Eigenvalue Gap", "Eigenvalue Gap Evolution")
        return ax

    # ------------------------------------------------------------------
    # Kernel convergence with width
    # ------------------------------------------------------------------

    def plot_kernel_convergence(
        self,
        kernels_at_widths: List[NDArray[np.floating]],
        widths: List[int],
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Show kernel convergence as width increases.

        Plots the Frobenius-norm distance from the widest kernel
        as a function of width.

        Parameters
        ----------
        kernels_at_widths : list of ndarray
            Kernel matrices at increasing widths.
        widths : list of int
            Corresponding network widths.
        ax : Axes or None
            Target axes.

        Returns
        -------
        Axes
        """
        ax = self._get_ax(ax)

        reference = np.asarray(kernels_at_widths[-1])
        distances = [
            np.linalg.norm(np.asarray(K) - reference, ord="fro")
            for K in kernels_at_widths
        ]

        ax.plot(
            widths,
            distances,
            "o-",
            color="#c0392b",
            linewidth=self.config.line_width,
            markersize=5,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")

        self._apply_style(
            ax,
            "Width (N)",
            "‖K(N) − K(∞)‖_F",
            "Kernel Convergence",
        )
        return ax

    # ------------------------------------------------------------------
    # Nyström accuracy
    # ------------------------------------------------------------------

    def plot_nystrom_accuracy(
        self,
        exact: NDArray[np.floating],
        approx: NDArray[np.floating],
        ranks: NDArray[np.integer],
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Nyström approximation accuracy vs rank.

        Parameters
        ----------
        exact : ndarray, shape (N, N)
            Exact kernel matrix.
        approx : ndarray, shape (R, N, N)
            Approximate kernel matrices at each rank.
        ranks : ndarray, shape (R,)
            Rank values used for approximation.
        ax : Axes or None
            Target axes.

        Returns
        -------
        Axes
        """
        ax = self._get_ax(ax)

        exact_arr = np.asarray(exact)
        errors = [
            np.linalg.norm(exact_arr - np.asarray(approx[i]), ord="fro")
            / max(np.linalg.norm(exact_arr, ord="fro"), 1e-12)
            for i in range(len(ranks))
        ]

        ax.plot(
            ranks,
            errors,
            "s-",
            color="#27ae60",
            linewidth=self.config.line_width,
            markersize=5,
        )
        ax.set_yscale("log")

        self._apply_style(ax, "Rank", "Relative Error", "Nyström Accuracy")
        return ax

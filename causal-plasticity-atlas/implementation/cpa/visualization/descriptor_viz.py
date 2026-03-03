"""Descriptor visualization for the CPA engine.

Provides :class:`DescriptorVisualizer` for plotting 4D plasticity
descriptors: scatter plots, radar charts, distribution histograms,
and classification boundary visualizations.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from cpa.utils.logging import get_logger

logger = get_logger("visualization.descriptor")

_MPL_AVAILABLE = True
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle, Patch
except ImportError:
    _MPL_AVAILABLE = False


def _require_matplotlib() -> None:
    if not _MPL_AVAILABLE:
        raise ImportError("matplotlib required: pip install matplotlib")


# Component names and colors
COMPONENTS = ["structural", "parametric", "emergence", "sensitivity"]
COMP_COLORS = ["#2196F3", "#FF9800", "#9C27B0", "#00BCD4"]

CLASS_COLORS: Dict[str, str] = {
    "invariant": "#4CAF50",
    "structurally_plastic": "#2196F3",
    "parametrically_plastic": "#FF9800",
    "fully_plastic": "#F44336",
    "emergent": "#9C27B0",
    "context_sensitive": "#00BCD4",
    "unclassified": "#9E9E9E",
}


class DescriptorVisualizer:
    """Visualize 4D plasticity descriptors.

    Parameters
    ----------
    figsize : tuple of float
        Default figure size.
    dpi : int
        Resolution for saved figures.

    Examples
    --------
    >>> viz = DescriptorVisualizer()
    >>> viz.scatter_2d(descriptors, save_path="scatter.png")
    >>> viz.radar_chart(descriptors, "X0", save_path="radar.png")
    """

    def __init__(
        self,
        figsize: Tuple[float, float] = (10, 8),
        dpi: int = 150,
    ) -> None:
        _require_matplotlib()
        self._figsize = figsize
        self._dpi = dpi

    def _save_or_show(
        self, fig: "Figure", save_path: Optional[Union[str, Path]]
    ) -> "Figure":
        fig.tight_layout()
        if save_path is not None:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=self._dpi, bbox_inches="tight")
            plt.close(fig)
        return fig

    # -----------------------------------------------------------------
    # 2D scatter plots (projections of 4D space)
    # -----------------------------------------------------------------

    def scatter_2d(
        self,
        descriptors: Dict[str, Any],
        dim_x: int = 0,
        dim_y: int = 1,
        color_by: str = "classification",
        save_path: Optional[Union[str, Path]] = None,
        show_names: bool = True,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> "Figure":
        """2D scatter plot of descriptors projected onto two dimensions.

        Parameters
        ----------
        descriptors : dict of str → DescriptorResult
            Variable → descriptor mapping.
        dim_x, dim_y : int
            Descriptor component indices (0-3).
        color_by : str
            'classification' or component name.
        save_path : str or Path, optional
            Save path.
        show_names : bool
            Annotate points with variable names.
        figsize : tuple, optional
            Figure size.

        Returns
        -------
        Figure
        """
        if not descriptors:
            fig, ax = plt.subplots(figsize=figsize or self._figsize)
            ax.text(0.5, 0.5, "No descriptor data", ha="center", va="center")
            return self._save_or_show(fig, save_path)

        variables = list(descriptors.keys())
        vectors = np.array([
            self._get_vector(descriptors[v]) for v in variables
        ])

        fig, ax = plt.subplots(figsize=figsize or (8, 6))

        if color_by == "classification":
            classes = [self._get_class(descriptors[v]) for v in variables]
            unique_classes = sorted(set(classes))

            for cls in unique_classes:
                mask = [c == cls for c in classes]
                mask_idx = [i for i, m in enumerate(mask) if m]
                if mask_idx:
                    ax.scatter(
                        vectors[mask_idx, dim_x],
                        vectors[mask_idx, dim_y],
                        c=CLASS_COLORS.get(cls, "#9E9E9E"),
                        label=cls.replace("_", " ").title(),
                        s=80, edgecolors="black", linewidths=0.5,
                        alpha=0.8,
                    )
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        else:
            comp_idx = COMPONENTS.index(color_by) if color_by in COMPONENTS else 0
            colors = vectors[:, comp_idx]
            scatter = ax.scatter(
                vectors[:, dim_x], vectors[:, dim_y],
                c=colors, cmap="YlOrRd", s=80,
                edgecolors="black", linewidths=0.5,
                vmin=0, vmax=1,
            )
            fig.colorbar(scatter, ax=ax, label=color_by.title())

        if show_names and len(variables) <= 30:
            for i, var in enumerate(variables):
                ax.annotate(
                    var,
                    (vectors[i, dim_x], vectors[i, dim_y]),
                    textcoords="offset points",
                    xytext=(4, 4), fontsize=7,
                )

        ax.set_xlabel(COMPONENTS[dim_x].title())
        ax.set_ylabel(COMPONENTS[dim_y].title())
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(
            f"Descriptor Scatter: {COMPONENTS[dim_x]} vs {COMPONENTS[dim_y]}"
        )
        ax.grid(True, alpha=0.3)

        return self._save_or_show(fig, save_path)

    def scatter_matrix(
        self,
        descriptors: Dict[str, Any],
        save_path: Optional[Union[str, Path]] = None,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> "Figure":
        """All pairwise 2D projections of the 4D descriptor space.

        Creates a 4×4 scatter plot matrix.

        Parameters
        ----------
        descriptors : dict of str → DescriptorResult
            Variable → descriptor mapping.
        save_path : str or Path, optional
            Save path.
        figsize : tuple, optional
            Figure size.

        Returns
        -------
        Figure
        """
        if not descriptors:
            fig, ax = plt.subplots(figsize=figsize or self._figsize)
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return self._save_or_show(fig, save_path)

        variables = list(descriptors.keys())
        vectors = np.array([
            self._get_vector(descriptors[v]) for v in variables
        ])
        classes = [self._get_class(descriptors[v]) for v in variables]
        colors = [CLASS_COLORS.get(c, "#9E9E9E") for c in classes]

        n_comp = 4
        fig, axes = plt.subplots(
            n_comp, n_comp,
            figsize=figsize or (12, 12),
        )

        for i in range(n_comp):
            for j in range(n_comp):
                ax = axes[i][j]
                if i == j:
                    ax.hist(
                        vectors[:, i], bins=15,
                        color=COMP_COLORS[i], alpha=0.7, edgecolor="white",
                    )
                    ax.set_xlim(-0.05, 1.05)
                else:
                    ax.scatter(
                        vectors[:, j], vectors[:, i],
                        c=colors, s=20, alpha=0.7,
                        edgecolors="black", linewidths=0.3,
                    )
                    ax.set_xlim(-0.05, 1.05)
                    ax.set_ylim(-0.05, 1.05)

                if i == n_comp - 1:
                    ax.set_xlabel(COMPONENTS[j], fontsize=8)
                else:
                    ax.set_xticklabels([])

                if j == 0:
                    ax.set_ylabel(COMPONENTS[i], fontsize=8)
                else:
                    ax.set_yticklabels([])

                ax.tick_params(labelsize=6)

        fig.suptitle("Descriptor Scatter Matrix", fontsize=13, fontweight="bold")

        return self._save_or_show(fig, save_path)

    # -----------------------------------------------------------------
    # Radar charts
    # -----------------------------------------------------------------

    def radar_chart(
        self,
        descriptors: Dict[str, Any],
        variable: str,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        show_ci: bool = True,
    ) -> "Figure":
        """Radar chart for a single variable's 4D descriptor.

        Parameters
        ----------
        descriptors : dict of str → DescriptorResult
            Variable → descriptor mapping.
        variable : str
            Variable to plot.
        save_path : str or Path, optional
            Save path.
        figsize : tuple, optional
            Figure size.
        show_ci : bool
            Show confidence interval bands if available.

        Returns
        -------
        Figure
        """
        if variable not in descriptors:
            fig, ax = plt.subplots(figsize=figsize or self._figsize)
            ax.text(0.5, 0.5, f"Variable {variable!r} not found",
                    ha="center", va="center")
            return self._save_or_show(fig, save_path)

        dr = descriptors[variable]
        vec = self._get_vector(dr)
        cls = self._get_class(dr)

        angles = np.linspace(0, 2 * np.pi, len(COMPONENTS), endpoint=False)
        values = np.concatenate([vec, [vec[0]]])
        angles_plot = np.concatenate([angles, [angles[0]]])

        fig, ax = plt.subplots(
            figsize=figsize or (7, 7),
            subplot_kw={"projection": "polar"},
        )

        ax.plot(
            angles_plot, values,
            "o-", linewidth=2,
            color=CLASS_COLORS.get(cls, "#2196F3"),
        )
        ax.fill(
            angles_plot, values,
            alpha=0.2,
            color=CLASS_COLORS.get(cls, "#2196F3"),
        )

        if show_ci:
            ci = self._get_ci(dr)
            if ci:
                lows = []
                highs = []
                for comp in COMPONENTS:
                    if comp in ci:
                        lo, hi = ci[comp]
                        lows.append(lo)
                        highs.append(hi)
                    else:
                        lows.append(vec[COMPONENTS.index(comp)])
                        highs.append(vec[COMPONENTS.index(comp)])
                lows = np.concatenate([lows, [lows[0]]])
                highs = np.concatenate([highs, [highs[0]]])
                ax.fill_between(
                    angles_plot, lows, highs,
                    alpha=0.1, color="#FF9800",
                )

        ax.set_xticks(angles)
        ax.set_xticklabels(
            [c.title() for c in COMPONENTS], fontsize=10
        )
        ax.set_ylim(0, 1)
        ax.set_title(
            f"{variable}\n({cls.replace('_', ' ').title()})",
            fontsize=12, fontweight="bold", pad=20,
        )

        return self._save_or_show(fig, save_path)

    def radar_comparison(
        self,
        descriptors: Dict[str, Any],
        variables: List[str],
        save_path: Optional[Union[str, Path]] = None,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> "Figure":
        """Overlaid radar charts comparing multiple variables.

        Parameters
        ----------
        descriptors : dict of str → DescriptorResult
            Variable → descriptor mapping.
        variables : list of str
            Variables to compare.
        save_path : str or Path, optional
            Save path.
        figsize : tuple, optional
            Figure size.

        Returns
        -------
        Figure
        """
        angles = np.linspace(0, 2 * np.pi, len(COMPONENTS), endpoint=False)
        angles_plot = np.concatenate([angles, [angles[0]]])

        fig, ax = plt.subplots(
            figsize=figsize or (8, 8),
            subplot_kw={"projection": "polar"},
        )

        cmap = plt.cm.get_cmap("tab10")
        for idx, var in enumerate(variables):
            if var not in descriptors:
                continue
            vec = self._get_vector(descriptors[var])
            values = np.concatenate([vec, [vec[0]]])
            color = cmap(idx % 10)
            ax.plot(angles_plot, values, "o-", linewidth=1.5, color=color, label=var)
            ax.fill(angles_plot, values, alpha=0.05, color=color)

        ax.set_xticks(angles)
        ax.set_xticklabels([c.title() for c in COMPONENTS], fontsize=10)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=8)
        ax.set_title("Descriptor Comparison", fontsize=12, fontweight="bold", pad=20)

        return self._save_or_show(fig, save_path)

    # -----------------------------------------------------------------
    # Component distributions
    # -----------------------------------------------------------------

    def component_distributions(
        self,
        descriptors: Dict[str, Any],
        save_path: Optional[Union[str, Path]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        bins: int = 20,
    ) -> "Figure":
        """Histograms of each descriptor component.

        Parameters
        ----------
        descriptors : dict of str → DescriptorResult
            Variable → descriptor mapping.
        save_path : str or Path, optional
            Save path.
        figsize : tuple, optional
            Figure size.
        bins : int
            Number of histogram bins.

        Returns
        -------
        Figure
        """
        if not descriptors:
            fig, ax = plt.subplots(figsize=figsize or self._figsize)
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return self._save_or_show(fig, save_path)

        vectors = np.array([
            self._get_vector(descriptors[v])
            for v in descriptors
        ])

        fig, axes = plt.subplots(2, 2, figsize=figsize or (10, 8))
        axes_flat = axes.flatten()

        for i, (ax, comp) in enumerate(zip(axes_flat, COMPONENTS)):
            vals = vectors[:, i]
            ax.hist(
                vals, bins=bins, range=(0, 1),
                color=COMP_COLORS[i], alpha=0.7,
                edgecolor="white",
            )
            ax.axvline(
                np.mean(vals), color="red", linestyle="--",
                linewidth=1.5, label=f"Mean={np.mean(vals):.3f}",
            )
            ax.set_xlabel(comp.title())
            ax.set_ylabel("Count")
            ax.set_title(f"{comp.title()} Distribution")
            ax.legend(fontsize=8)
            ax.set_xlim(0, 1)

        fig.suptitle(
            "Descriptor Component Distributions",
            fontsize=13, fontweight="bold",
        )

        return self._save_or_show(fig, save_path)

    # -----------------------------------------------------------------
    # Classification boundary visualization
    # -----------------------------------------------------------------

    def classification_boundaries(
        self,
        descriptors: Dict[str, Any],
        dim_x: int = 0,
        dim_y: int = 1,
        thresholds: Optional[Dict[str, float]] = None,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> "Figure":
        """Scatter plot with classification threshold boundaries.

        Parameters
        ----------
        descriptors : dict of str → DescriptorResult
            Variable → descriptor mapping.
        dim_x, dim_y : int
            Descriptor dimensions.
        thresholds : dict, optional
            Override thresholds: {component: threshold_value}.
        save_path : str or Path, optional
            Save path.
        figsize : tuple, optional
            Figure size.

        Returns
        -------
        Figure
        """
        default_thresholds = {
            "structural": 0.3,
            "parametric": 0.3,
            "emergence": 0.2,
            "sensitivity": 0.3,
        }
        if thresholds:
            default_thresholds.update(thresholds)

        variables = list(descriptors.keys())
        vectors = np.array([
            self._get_vector(descriptors[v]) for v in variables
        ])
        classes = [self._get_class(descriptors[v]) for v in variables]

        fig, ax = plt.subplots(figsize=figsize or (8, 6))

        unique_classes = sorted(set(classes))
        for cls in unique_classes:
            mask = [c == cls for c in classes]
            mask_idx = [i for i, m in enumerate(mask) if m]
            if mask_idx:
                ax.scatter(
                    vectors[mask_idx, dim_x],
                    vectors[mask_idx, dim_y],
                    c=CLASS_COLORS.get(cls, "#9E9E9E"),
                    label=cls.replace("_", " ").title(),
                    s=80, edgecolors="black", linewidths=0.5,
                    zorder=5,
                )

        comp_x = COMPONENTS[dim_x]
        comp_y = COMPONENTS[dim_y]

        if comp_x in default_thresholds:
            thresh = default_thresholds[comp_x]
            ax.axvline(
                thresh, color="#F44336", linestyle="--",
                linewidth=1.5, alpha=0.5,
                label=f"{comp_x} thresh={thresh}",
            )

        if comp_y in default_thresholds:
            thresh = default_thresholds[comp_y]
            ax.axhline(
                thresh, color="#2196F3", linestyle="--",
                linewidth=1.5, alpha=0.5,
                label=f"{comp_y} thresh={thresh}",
            )

        ax.set_xlabel(comp_x.title())
        ax.set_ylabel(comp_y.title())
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Classification Boundaries")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        return self._save_or_show(fig, save_path)

    # -----------------------------------------------------------------
    # Norm distribution
    # -----------------------------------------------------------------

    def norm_distribution(
        self,
        descriptors: Dict[str, Any],
        invariance_threshold: float = 0.1,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> "Figure":
        """Histogram of descriptor L2 norms with invariance threshold.

        Parameters
        ----------
        descriptors : dict
            Variable → descriptor mapping.
        invariance_threshold : float
            Maximum norm for invariant classification.
        save_path : str or Path, optional
            Save path.
        figsize : tuple, optional
            Figure size.

        Returns
        -------
        Figure
        """
        if not descriptors:
            fig, ax = plt.subplots(figsize=figsize or self._figsize)
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return self._save_or_show(fig, save_path)

        norms = [
            float(np.linalg.norm(self._get_vector(descriptors[v])))
            for v in descriptors
        ]

        fig, ax = plt.subplots(figsize=figsize or (8, 5))
        ax.hist(norms, bins=25, color="#2196F3", alpha=0.7, edgecolor="white")
        ax.axvline(
            invariance_threshold, color="#F44336",
            linestyle="--", linewidth=2,
            label=f"Invariance threshold ({invariance_threshold})",
        )
        ax.set_xlabel("Descriptor L2 Norm")
        ax.set_ylabel("Count")
        ax.set_title("Descriptor Norm Distribution")
        ax.legend(fontsize=9)

        n_invariant = sum(1 for n in norms if n <= invariance_threshold)
        ax.text(
            0.95, 0.95,
            f"Invariant: {n_invariant}/{len(norms)}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        return self._save_or_show(fig, save_path)

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _get_vector(dr: Any) -> np.ndarray:
        """Extract 4D vector from a descriptor."""
        if hasattr(dr, "vector"):
            return np.asarray(dr.vector)
        if isinstance(dr, dict):
            return np.array([
                dr.get("structural", 0.0),
                dr.get("parametric", 0.0),
                dr.get("emergence", 0.0),
                dr.get("sensitivity", 0.0),
            ])
        if hasattr(dr, "structural"):
            return np.array([
                dr.structural, dr.parametric,
                dr.emergence, dr.sensitivity,
            ])
        return np.zeros(4)

    @staticmethod
    def _get_class(dr: Any) -> str:
        """Extract classification string."""
        if hasattr(dr, "classification"):
            cls = dr.classification
            return cls.value if hasattr(cls, "value") else str(cls)
        if isinstance(dr, dict):
            return dr.get("classification", "unclassified")
        return "unclassified"

    @staticmethod
    def _get_ci(dr: Any) -> Dict[str, Tuple[float, float]]:
        """Extract confidence intervals."""
        if hasattr(dr, "confidence_intervals"):
            return dr.confidence_intervals
        if isinstance(dr, dict):
            return dr.get("confidence_intervals", {})
        return {}

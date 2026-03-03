"""Atlas-level visualization for the CPA engine.

Provides :class:`AtlasVisualizer` with methods for plasticity heatmaps,
classification distributions, context embeddings, QD archive maps,
tipping-point timelines, and certificate dashboards.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from cpa.utils.logging import get_logger

logger = get_logger("visualization.atlas")

# Lazy matplotlib import to allow headless operation
_MPL_AVAILABLE = True
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch
except ImportError:
    _MPL_AVAILABLE = False


def _require_matplotlib() -> None:
    """Raise ImportError if matplotlib is not available."""
    if not _MPL_AVAILABLE:
        raise ImportError(
            "matplotlib is required for visualization: pip install matplotlib"
        )


# =====================================================================
# Color scheme
# =====================================================================

CLASS_COLORS: Dict[str, str] = {
    "invariant": "#4CAF50",
    "structurally_plastic": "#2196F3",
    "parametrically_plastic": "#FF9800",
    "fully_plastic": "#F44336",
    "emergent": "#9C27B0",
    "context_sensitive": "#00BCD4",
    "unclassified": "#9E9E9E",
}

COMPONENT_COLORS: Dict[str, str] = {
    "structural": "#2196F3",
    "parametric": "#FF9800",
    "emergence": "#9C27B0",
    "sensitivity": "#00BCD4",
}


# =====================================================================
# AtlasVisualizer
# =====================================================================


class AtlasVisualizer:
    """Comprehensive visualization for the Causal-Plasticity Atlas.

    Parameters
    ----------
    figsize : tuple of float
        Default figure size (width, height) in inches.
    dpi : int
        Resolution for saved figures.
    style : str
        Matplotlib style name.
    save_format : str
        Default format for saved figures ('png', 'pdf', 'svg').

    Examples
    --------
    >>> viz = AtlasVisualizer()
    >>> viz.plasticity_heatmap(atlas, save_path="heatmap.png")
    >>> viz.classification_distribution(atlas, save_path="classes.png")
    """

    def __init__(
        self,
        figsize: Tuple[float, float] = (10, 8),
        dpi: int = 150,
        style: str = "default",
        save_format: str = "png",
    ) -> None:
        _require_matplotlib()
        self._figsize = figsize
        self._dpi = dpi
        self._style = style
        self._save_format = save_format

    def _save_or_show(
        self,
        fig: "Figure",
        save_path: Optional[Union[str, Path]],
        tight: bool = True,
    ) -> "Figure":
        """Save figure to file or show interactively."""
        if tight:
            fig.tight_layout()
        if save_path is not None:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=self._dpi, bbox_inches="tight")
            logger.info("Figure saved: %s", p)
            plt.close(fig)
        return fig

    # -----------------------------------------------------------------
    # Plasticity heatmap
    # -----------------------------------------------------------------

    def plasticity_heatmap(
        self,
        atlas: Any,
        save_path: Optional[Union[str, Path]] = None,
        sort_by: str = "norm",
        cmap: str = "YlOrRd",
        show_values: bool = True,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> "Figure":
        """Draw a plasticity heatmap (variables × descriptor components).

        Parameters
        ----------
        atlas : AtlasResult
            Atlas with foundation results.
        save_path : str or Path, optional
            File path to save the figure.
        sort_by : str
            Sort variables by: 'norm', 'structural', 'parametric',
            'emergence', 'sensitivity', or 'name'.
        cmap : str
            Colormap name.
        show_values : bool
            Annotate cells with numeric values.
        figsize : tuple, optional
            Override figure size.

        Returns
        -------
        Figure
        """
        mat, var_names, comp_names = atlas.plasticity_heatmap()
        if mat.size == 0:
            fig, ax = plt.subplots(figsize=figsize or self._figsize)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
            ax.set_title("Plasticity Heatmap")
            return self._save_or_show(fig, save_path)

        sort_map = {
            "norm": lambda i: np.linalg.norm(mat[i]),
            "structural": lambda i: mat[i, 0],
            "parametric": lambda i: mat[i, 1],
            "emergence": lambda i: mat[i, 2],
            "sensitivity": lambda i: mat[i, 3],
            "name": lambda i: var_names[i],
        }
        sort_fn = sort_map.get(sort_by, sort_map["norm"])
        order = sorted(range(len(var_names)), key=sort_fn, reverse=True)

        sorted_mat = mat[order]
        sorted_names = [var_names[i] for i in order]

        fig_size = figsize or (
            max(6, len(comp_names) * 1.5),
            max(4, len(var_names) * 0.3 + 2),
        )
        fig, ax = plt.subplots(figsize=fig_size)

        im = ax.imshow(sorted_mat, cmap=cmap, aspect="auto", vmin=0, vmax=1)

        ax.set_xticks(range(len(comp_names)))
        ax.set_xticklabels(comp_names, rotation=45, ha="right")
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names, fontsize=max(6, 10 - len(var_names) // 10))

        if show_values and len(var_names) <= 30:
            for i in range(sorted_mat.shape[0]):
                for j in range(sorted_mat.shape[1]):
                    val = sorted_mat[i, j]
                    color = "white" if val > 0.5 else "black"
                    ax.text(
                        j, i, f"{val:.2f}",
                        ha="center", va="center",
                        fontsize=8, color=color,
                    )

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Plasticity Score")
        ax.set_title("Plasticity Descriptor Heatmap")

        return self._save_or_show(fig, save_path)

    # -----------------------------------------------------------------
    # Classification distribution
    # -----------------------------------------------------------------

    def classification_distribution(
        self,
        atlas: Any,
        save_path: Optional[Union[str, Path]] = None,
        chart_type: str = "bar",
        figsize: Optional[Tuple[float, float]] = None,
    ) -> "Figure":
        """Draw classification distribution as bar or pie chart.

        Parameters
        ----------
        atlas : AtlasResult
            Atlas with foundation results.
        save_path : str or Path, optional
            Save path.
        chart_type : str
            'bar' or 'pie'.
        figsize : tuple, optional
            Figure size.

        Returns
        -------
        Figure
        """
        summary = atlas.classification_summary()
        if not summary:
            fig, ax = plt.subplots(figsize=figsize or self._figsize)
            ax.text(0.5, 0.5, "No classification data", ha="center", va="center")
            return self._save_or_show(fig, save_path)

        classes = list(summary.keys())
        counts = [summary[c] for c in classes]
        colors = [CLASS_COLORS.get(c, "#9E9E9E") for c in classes]

        fig, ax = plt.subplots(figsize=figsize or (8, 5))

        if chart_type == "pie":
            wedges, texts, autotexts = ax.pie(
                counts,
                labels=classes,
                colors=colors,
                autopct="%1.1f%%",
                startangle=90,
            )
            ax.set_title("Mechanism Classification Distribution")
        else:
            bars = ax.bar(range(len(classes)), counts, color=colors)
            ax.set_xticks(range(len(classes)))
            ax.set_xticklabels(
                [c.replace("_", "\n") for c in classes],
                rotation=45, ha="right", fontsize=9,
            )
            ax.set_ylabel("Number of Variables")
            ax.set_title("Mechanism Classification Distribution")

            for bar, count in zip(bars, counts):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    str(count),
                    ha="center", va="bottom", fontweight="bold",
                )

        return self._save_or_show(fig, save_path)

    # -----------------------------------------------------------------
    # Context distance embedding
    # -----------------------------------------------------------------

    def context_embedding(
        self,
        atlas: Any,
        save_path: Optional[Union[str, Path]] = None,
        method: str = "mds",
        figsize: Optional[Tuple[float, float]] = None,
    ) -> "Figure":
        """Embed contexts in 2D based on alignment cost distances.

        Parameters
        ----------
        atlas : AtlasResult
            Atlas with alignment results.
        save_path : str or Path, optional
            Save path.
        method : str
            Embedding method: 'mds' or 'tsne'.
        figsize : tuple, optional
            Figure size.

        Returns
        -------
        Figure
        """
        cost_mat = atlas.alignment_cost_matrix()
        if cost_mat.size == 0 or cost_mat.shape[0] < 2:
            fig, ax = plt.subplots(figsize=figsize or self._figsize)
            ax.text(0.5, 0.5, "Insufficient contexts", ha="center", va="center")
            return self._save_or_show(fig, save_path)

        context_ids = atlas.context_ids
        K = cost_mat.shape[0]

        if method == "mds":
            embedding = self._mds_embed(cost_mat, 2)
        elif method == "tsne":
            try:
                from sklearn.manifold import TSNE
                tsne = TSNE(
                    n_components=2,
                    metric="precomputed",
                    random_state=42,
                    perplexity=min(30, max(5, K // 3)),
                )
                embedding = tsne.fit_transform(cost_mat)
            except ImportError:
                logger.warning("sklearn not available, falling back to MDS")
                embedding = self._mds_embed(cost_mat, 2)
        else:
            embedding = self._mds_embed(cost_mat, 2)

        fig, ax = plt.subplots(figsize=figsize or (8, 6))

        ax.scatter(
            embedding[:, 0], embedding[:, 1],
            s=100, c=range(K), cmap="viridis",
            edgecolors="black", linewidths=1, zorder=5,
        )

        for i, cid in enumerate(context_ids):
            ax.annotate(
                cid,
                (embedding[i, 0], embedding[i, 1]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_title(f"Context Embedding ({method.upper()})")
        ax.grid(True, alpha=0.3)

        return self._save_or_show(fig, save_path)

    def _mds_embed(
        self, dist_matrix: np.ndarray, n_components: int = 2
    ) -> np.ndarray:
        """Classical MDS (metric multidimensional scaling).

        Parameters
        ----------
        dist_matrix : np.ndarray
            (K, K) distance/cost matrix.
        n_components : int
            Target dimensionality.

        Returns
        -------
        np.ndarray
            (K, n_components) embedding.
        """
        K = dist_matrix.shape[0]
        D_sq = dist_matrix ** 2
        H = np.eye(K) - np.ones((K, K)) / K
        B = -0.5 * H @ D_sq @ H

        eigenvalues, eigenvectors = np.linalg.eigh(B)
        idx = np.argsort(eigenvalues)[::-1][:n_components]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        eigenvalues = np.maximum(eigenvalues, 0)
        return eigenvectors * np.sqrt(eigenvalues)

    # -----------------------------------------------------------------
    # QD archive coverage map
    # -----------------------------------------------------------------

    def archive_coverage(
        self,
        atlas: Any,
        save_path: Optional[Union[str, Path]] = None,
        dims: Tuple[int, int] = (0, 1),
        figsize: Optional[Tuple[float, float]] = None,
    ) -> "Figure":
        """Visualize QD archive coverage in 2D descriptor space.

        Parameters
        ----------
        atlas : AtlasResult
            Atlas with exploration results.
        save_path : str or Path, optional
            Save path.
        dims : tuple of int
            Which 2 descriptor dimensions to show.
        figsize : tuple, optional
            Figure size.

        Returns
        -------
        Figure
        """
        fig, ax = plt.subplots(figsize=figsize or (8, 6))

        exploration = getattr(atlas, "exploration", None)
        if exploration is None or not exploration.archive:
            ax.text(0.5, 0.5, "No archive data", ha="center", va="center")
            return self._save_or_show(fig, save_path)

        descriptors = []
        fitnesses = []
        for entry in exploration.archive:
            desc = entry.descriptor
            if desc is not None and len(desc) > max(dims):
                descriptors.append(desc)
                fitnesses.append(entry.fitness)

        if not descriptors:
            ax.text(0.5, 0.5, "No descriptors", ha="center", va="center")
            return self._save_or_show(fig, save_path)

        desc_arr = np.array(descriptors)
        fit_arr = np.array(fitnesses)

        dim_names = ["structural", "parametric", "emergence", "sensitivity"]
        d0, d1 = dims

        scatter = ax.scatter(
            desc_arr[:, d0], desc_arr[:, d1],
            c=fit_arr, cmap="plasma", s=40,
            edgecolors="black", linewidths=0.5,
            alpha=0.8,
        )

        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Fitness")

        ax.set_xlabel(dim_names[d0] if d0 < len(dim_names) else f"dim {d0}")
        ax.set_ylabel(dim_names[d1] if d1 < len(dim_names) else f"dim {d1}")
        ax.set_title(
            f"QD Archive Coverage ({len(descriptors)} entries, "
            f"coverage={exploration.coverage:.1%})"
        )
        ax.grid(True, alpha=0.3)

        return self._save_or_show(fig, save_path)

    # -----------------------------------------------------------------
    # QD convergence plot
    # -----------------------------------------------------------------

    def convergence_plot(
        self,
        atlas: Any,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> "Figure":
        """Plot QD-score convergence over iterations.

        Parameters
        ----------
        atlas : AtlasResult
            Atlas with exploration results.
        save_path : str or Path, optional
            Save path.
        figsize : tuple, optional
            Figure size.

        Returns
        -------
        Figure
        """
        fig, ax = plt.subplots(figsize=figsize or (8, 5))

        exploration = getattr(atlas, "exploration", None)
        if exploration is None or not exploration.convergence_history:
            ax.text(0.5, 0.5, "No convergence data", ha="center", va="center")
            return self._save_or_show(fig, save_path)

        history = exploration.convergence_history
        ax.plot(history, color="#2196F3", linewidth=1.5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("QD Score")
        ax.set_title("QD-MAP-Elites Convergence")
        ax.grid(True, alpha=0.3)

        return self._save_or_show(fig, save_path)

    # -----------------------------------------------------------------
    # Tipping-point timeline
    # -----------------------------------------------------------------

    def tipping_point_timeline(
        self,
        atlas: Any,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> "Figure":
        """Visualize tipping points across ordered contexts.

        Parameters
        ----------
        atlas : AtlasResult
            Atlas with validation results.
        save_path : str or Path, optional
            Save path.
        figsize : tuple, optional
            Figure size.

        Returns
        -------
        Figure
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize or (12, 6), sharex=True)

        validation = getattr(atlas, "validation", None)
        foundation = getattr(atlas, "foundation", None)

        if validation is None or validation.tipping_points is None:
            axes[0].text(0.5, 0.5, "No tipping-point data", ha="center", va="center")
            return self._save_or_show(fig, save_path)

        tp = validation.tipping_points
        context_ids = atlas.context_ids
        K = len(context_ids)

        # Top panel: alignment cost trajectory
        if foundation is not None:
            cost_mat = foundation.alignment_cost_matrix
            if cost_mat.shape[0] >= 2:
                consecutive = [cost_mat[i, i + 1] for i in range(K - 1)]
                x = np.arange(len(consecutive))
                axes[0].plot(x, consecutive, "o-", color="#2196F3", linewidth=1.5)
                axes[0].set_ylabel("Alignment Cost")

                for cp in tp.validated_changepoints:
                    axes[0].axvline(
                        cp - 0.5, color="#F44336", linestyle="--",
                        linewidth=2, alpha=0.7,
                    )

        axes[0].set_title("Tipping-Point Timeline")

        # Bottom panel: segment membership
        if tp.segments:
            colors = plt.cm.Set3(np.linspace(0, 1, len(tp.segments)))
            for seg_idx, (start, end) in enumerate(tp.segments):
                axes[1].axvspan(
                    start - 0.5, end - 0.5,
                    alpha=0.3, color=colors[seg_idx],
                    label=tp.segment_labels[seg_idx]
                    if seg_idx < len(tp.segment_labels)
                    else f"Seg {seg_idx + 1}",
                )

            for cp in tp.validated_changepoints:
                axes[1].axvline(
                    cp - 0.5, color="#F44336", linestyle="--",
                    linewidth=2, alpha=0.7,
                )
                p_val = tp.p_values.get(cp, None)
                label = f"CP={cp}"
                if p_val is not None:
                    label += f"\np={p_val:.3f}"
                axes[1].annotate(
                    label, (cp - 0.5, 0.5),
                    textcoords="offset points",
                    xytext=(5, 0), fontsize=8, color="#F44336",
                )

        axes[1].set_xlabel("Context Index")
        axes[1].set_ylabel("Segment")
        axes[1].set_xticks(range(K))
        axes[1].set_xticklabels(
            context_ids, rotation=45, ha="right", fontsize=8,
        )
        if tp.segments:
            axes[1].legend(loc="upper right", fontsize=8)

        return self._save_or_show(fig, save_path)

    # -----------------------------------------------------------------
    # Certificate dashboard
    # -----------------------------------------------------------------

    def certificate_dashboard(
        self,
        atlas: Any,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> "Figure":
        """Visualize robustness certificates as a dashboard.

        Parameters
        ----------
        atlas : AtlasResult
            Atlas with validation results.
        save_path : str or Path, optional
            Save path.
        figsize : tuple, optional
            Figure size.

        Returns
        -------
        Figure
        """
        validation = getattr(atlas, "validation", None)
        if validation is None or not validation.certificates:
            fig, ax = plt.subplots(figsize=figsize or self._figsize)
            ax.text(0.5, 0.5, "No certificate data", ha="center", va="center")
            return self._save_or_show(fig, save_path)

        certs = validation.certificates
        n_vars = len(certs)

        fig, axes = plt.subplots(
            1, 3,
            figsize=figsize or (15, max(4, n_vars * 0.3 + 2)),
        )

        # Panel 1: Certification status
        ax = axes[0]
        variables = list(certs.keys())
        certified = [certs[v].certified for v in variables]
        colors = ["#4CAF50" if c else "#F44336" for c in certified]
        ax.barh(range(len(variables)), [1] * len(variables), color=colors)
        ax.set_yticks(range(len(variables)))
        ax.set_yticklabels(variables, fontsize=8)
        ax.set_xlim(0, 1.2)
        ax.set_title("Certification Status")
        ax.set_xlabel("")

        legend_elements = [
            Patch(facecolor="#4CAF50", label="Certified"),
            Patch(facecolor="#F44336", label="Not Certified"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

        # Panel 2: Stability scores
        ax = axes[1]
        scores = [certs[v].stability_score for v in variables]
        ax.barh(range(len(variables)), scores, color="#2196F3", alpha=0.7)
        ax.set_yticks(range(len(variables)))
        ax.set_yticklabels(variables, fontsize=8)
        ax.set_xlim(0, 1.1)
        ax.set_title("Stability Scores")
        ax.set_xlabel("Score")

        # Panel 3: UCB bounds
        ax = axes[2]
        ucbs = [certs[v].ucb_bound for v in variables]
        ax.barh(range(len(variables)), ucbs, color="#FF9800", alpha=0.7)
        ax.set_yticks(range(len(variables)))
        ax.set_yticklabels(variables, fontsize=8)
        ax.set_title("UCB Bounds")
        ax.set_xlabel("Bound")

        fig.suptitle(
            f"Robustness Certificate Dashboard "
            f"({validation.n_certified}/{n_vars} certified)",
            fontsize=12, fontweight="bold",
        )

        return self._save_or_show(fig, save_path)

    # -----------------------------------------------------------------
    # Summary dashboard
    # -----------------------------------------------------------------

    def summary_dashboard(
        self,
        atlas: Any,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> "Figure":
        """Generate a multi-panel summary dashboard.

        Creates a 2x2 grid with:
        - Plasticity heatmap (top-left)
        - Classification distribution (top-right)
        - Context embedding (bottom-left)
        - Convergence plot or certificate summary (bottom-right)

        Parameters
        ----------
        atlas : AtlasResult
            Atlas results.
        save_path : str or Path, optional
            Save path.
        figsize : tuple, optional
            Figure size.

        Returns
        -------
        Figure
        """
        fig = plt.figure(figsize=figsize or (16, 12))

        # Top-left: mini heatmap
        ax1 = fig.add_subplot(2, 2, 1)
        mat, var_names, comp_names = atlas.plasticity_heatmap()
        if mat.size > 0:
            im = ax1.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
            ax1.set_xticks(range(len(comp_names)))
            ax1.set_xticklabels(comp_names, rotation=45, ha="right", fontsize=8)
            if len(var_names) <= 20:
                ax1.set_yticks(range(len(var_names)))
                ax1.set_yticklabels(var_names, fontsize=7)
            ax1.set_title("Plasticity Heatmap", fontsize=10)
            fig.colorbar(im, ax=ax1, shrink=0.7)
        else:
            ax1.text(0.5, 0.5, "No data", ha="center", va="center")
            ax1.set_title("Plasticity Heatmap", fontsize=10)

        # Top-right: classification
        ax2 = fig.add_subplot(2, 2, 2)
        summary = atlas.classification_summary()
        if summary:
            classes = list(summary.keys())
            counts = [summary[c] for c in classes]
            colors = [CLASS_COLORS.get(c, "#9E9E9E") for c in classes]
            ax2.bar(range(len(classes)), counts, color=colors)
            ax2.set_xticks(range(len(classes)))
            ax2.set_xticklabels(
                [c.replace("_", "\n") for c in classes],
                rotation=45, ha="right", fontsize=7,
            )
            ax2.set_ylabel("Count")
        ax2.set_title("Classification Distribution", fontsize=10)

        # Bottom-left: context embedding
        ax3 = fig.add_subplot(2, 2, 3)
        cost_mat = atlas.alignment_cost_matrix()
        if cost_mat.size > 0 and cost_mat.shape[0] >= 3:
            embedding = self._mds_embed(cost_mat, 2)
            ax3.scatter(
                embedding[:, 0], embedding[:, 1],
                s=80, c=range(cost_mat.shape[0]), cmap="viridis",
                edgecolors="black", linewidths=0.5,
            )
            for i, cid in enumerate(atlas.context_ids):
                ax3.annotate(
                    cid, (embedding[i, 0], embedding[i, 1]),
                    textcoords="offset points", xytext=(3, 3), fontsize=7,
                )
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, "< 3 contexts", ha="center", va="center")
        ax3.set_title("Context Embedding (MDS)", fontsize=10)

        # Bottom-right: convergence or certificate summary
        ax4 = fig.add_subplot(2, 2, 4)
        exploration = getattr(atlas, "exploration", None)
        validation = getattr(atlas, "validation", None)

        if exploration is not None and exploration.convergence_history:
            ax4.plot(
                exploration.convergence_history,
                color="#2196F3", linewidth=1,
            )
            ax4.set_xlabel("Iteration")
            ax4.set_ylabel("QD Score")
            ax4.set_title("QD Convergence", fontsize=10)
            ax4.grid(True, alpha=0.3)
        elif validation is not None and validation.certificates:
            cert_rate = validation.certification_rate
            ax4.bar(
                ["Certified", "Not Certified"],
                [validation.n_certified, len(validation.certificates) - validation.n_certified],
                color=["#4CAF50", "#F44336"],
            )
            ax4.set_title(
                f"Certification ({cert_rate:.0%})", fontsize=10
            )
        else:
            stats = atlas.summary_statistics()
            text = "\n".join(
                f"{k}: {v}" for k, v in stats.items()
                if not isinstance(v, dict)
            )
            ax4.text(
                0.1, 0.5, text, fontsize=9,
                transform=ax4.transAxes, verticalalignment="center",
                fontfamily="monospace",
            )
            ax4.set_title("Summary", fontsize=10)

        fig.suptitle(
            f"Causal-Plasticity Atlas Summary "
            f"(K={atlas.n_contexts}, p={atlas.n_variables})",
            fontsize=14, fontweight="bold", y=0.98,
        )

        return self._save_or_show(fig, save_path)

    # -----------------------------------------------------------------
    # Alignment cost matrix
    # -----------------------------------------------------------------

    def alignment_cost_heatmap(
        self,
        atlas: Any,
        save_path: Optional[Union[str, Path]] = None,
        cmap: str = "Blues",
        figsize: Optional[Tuple[float, float]] = None,
    ) -> "Figure":
        """Draw the pairwise alignment cost matrix as a heatmap.

        Parameters
        ----------
        atlas : AtlasResult
            Atlas with alignment results.
        save_path : str or Path, optional
            Save path.
        cmap : str
            Colormap.
        figsize : tuple, optional
            Figure size.

        Returns
        -------
        Figure
        """
        cost_mat = atlas.alignment_cost_matrix()
        context_ids = atlas.context_ids

        if cost_mat.size == 0:
            fig, ax = plt.subplots(figsize=figsize or self._figsize)
            ax.text(0.5, 0.5, "No alignment data", ha="center", va="center")
            return self._save_or_show(fig, save_path)

        K = cost_mat.shape[0]
        fig, ax = plt.subplots(figsize=figsize or (max(6, K * 0.6), max(5, K * 0.5)))

        im = ax.imshow(cost_mat, cmap=cmap, aspect="equal")
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xticklabels(context_ids, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(context_ids, fontsize=8)

        if K <= 15:
            for i in range(K):
                for j in range(K):
                    val = cost_mat[i, j]
                    color = "white" if val > cost_mat.max() / 2 else "black"
                    ax.text(
                        j, i, f"{val:.2f}",
                        ha="center", va="center", fontsize=7, color=color,
                    )

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Alignment Cost")
        ax.set_title("Pairwise Context Alignment Costs")

        return self._save_or_show(fig, save_path)

    # -----------------------------------------------------------------
    # Sensitivity analysis plot
    # -----------------------------------------------------------------

    def sensitivity_plot(
        self,
        atlas: Any,
        save_path: Optional[Union[str, Path]] = None,
        metric: str = "loo_sensitivity",
        figsize: Optional[Tuple[float, float]] = None,
    ) -> "Figure":
        """Plot sensitivity analysis results.

        Parameters
        ----------
        atlas : AtlasResult
            Atlas with validation results.
        save_path : str or Path, optional
            Save path.
        metric : str
            Sensitivity metric to plot.
        figsize : tuple, optional
            Figure size.

        Returns
        -------
        Figure
        """
        validation = getattr(atlas, "validation", None)
        if validation is None or not validation.sensitivity:
            fig, ax = plt.subplots(figsize=figsize or self._figsize)
            ax.text(0.5, 0.5, "No sensitivity data", ha="center", va="center")
            return self._save_or_show(fig, save_path)

        variables = list(validation.sensitivity.keys())
        values = [
            validation.sensitivity[v].get(metric, 0.0) for v in variables
        ]

        order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
        variables = [variables[i] for i in order]
        values = [values[i] for i in order]

        fig, ax = plt.subplots(
            figsize=figsize or (8, max(4, len(variables) * 0.3 + 1))
        )
        ax.barh(range(len(variables)), values, color="#FF9800", alpha=0.7)
        ax.set_yticks(range(len(variables)))
        ax.set_yticklabels(variables, fontsize=8)
        ax.set_xlabel(metric.replace("_", " ").title())
        ax.set_title("Sensitivity Analysis")
        ax.grid(True, alpha=0.3, axis="x")

        return self._save_or_show(fig, save_path)

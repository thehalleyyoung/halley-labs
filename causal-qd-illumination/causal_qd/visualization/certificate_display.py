"""Visualisation of edge certificates on DAG structures.

Provides:
  - plot_edge_certificates: bar chart of edge stability values
  - plot_certificate_heatmap: matrix heatmap of edge certificates
  - plot_bootstrap_distribution: histogram of bootstrap scores
  - display_edge_certificates: render DAG with edges colored by certificate
  - display_certificate_histogram: histogram of certificate strengths
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from causal_qd.core.dag import DAG


class CertificateDisplay:
    """Displays edge-certificate information via matplotlib plots."""

    # ------------------------------------------------------------------
    # Bar chart of per-edge certificate strengths
    # ------------------------------------------------------------------

    @staticmethod
    def plot_edge_certificates(
        certificates: Dict[Tuple[int, int], float],
        threshold: float = 0.5,
        figsize: Tuple[float, float] = (10, 6),
        title: str = "Edge Certificate Strengths",
        ax: Optional[Axes] = None,
    ) -> Figure:
        """Bar chart of edge stability/certificate values.

        Each bar represents one edge; bars are sorted by strength.
        Edges above *threshold* are colored green, below are orange.

        Parameters
        ----------
        certificates :
            Mapping ``(source, target) → strength`` in ``[0, 1]``.
        threshold :
            Threshold for coloring bars.
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

        if not certificates:
            ax.text(0.5, 0.5, "No certificates", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(title)
            return fig

        edges_sorted = sorted(certificates.items(), key=lambda x: x[1], reverse=True)
        labels = [f"{e[0]}→{e[1]}" for e, _ in edges_sorted]
        values = [v for _, v in edges_sorted]
        colors = ["forestgreen" if v >= threshold else "orange" for v in values]

        bars = ax.barh(range(len(labels)), values, color=colors, edgecolor="black",
                       linewidth=0.5, alpha=0.85)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Certificate Strength")
        ax.set_xlim(0, 1.05)
        ax.axvline(x=threshold, color="red", linestyle="--", linewidth=1,
                   label=f"Threshold ({threshold})")
        ax.legend(fontsize=9)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="x")
        ax.invert_yaxis()
        return fig

    # ------------------------------------------------------------------
    # Certificate heatmap (n×n matrix)
    # ------------------------------------------------------------------

    @staticmethod
    def plot_certificate_heatmap(
        certificates: Dict[Tuple[int, int], float],
        n_nodes: int,
        cmap: str = "RdYlGn",
        figsize: Tuple[float, float] = (8, 6),
        title: str = "Edge Certificate Heatmap",
        node_labels: Optional[List[str]] = None,
        ax: Optional[Axes] = None,
    ) -> Figure:
        """Matrix heatmap showing certificate strength for each possible edge.

        Parameters
        ----------
        certificates :
            Mapping ``(source, target) → strength``.
        n_nodes :
            Number of nodes in the graph.
        cmap :
            Colormap name.
        figsize :
            Figure size.
        title :
            Plot title.
        node_labels :
            Optional labels for nodes.
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

        mat = np.full((n_nodes, n_nodes), np.nan)
        for (i, j), v in certificates.items():
            mat[i, j] = v

        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        fig.colorbar(im, ax=ax, label="Strength")

        ticks = list(range(n_nodes))
        if node_labels is not None:
            tick_labels = node_labels[:n_nodes]
        else:
            tick_labels = [str(i) for i in ticks]

        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, fontsize=8, rotation=45)
        ax.set_yticks(ticks)
        ax.set_yticklabels(tick_labels, fontsize=8)
        ax.set_xlabel("Target")
        ax.set_ylabel("Source")
        ax.set_title(title)

        # Annotate values
        for i in range(n_nodes):
            for j in range(n_nodes):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                            fontsize=7, color="black" if mat[i, j] > 0.4 else "white")

        return fig

    # ------------------------------------------------------------------
    # Bootstrap score distribution
    # ------------------------------------------------------------------

    @staticmethod
    def plot_bootstrap_distribution(
        bootstrap_scores: List[float],
        observed_score: Optional[float] = None,
        bins: int = 30,
        figsize: Tuple[float, float] = (8, 5),
        title: str = "Bootstrap Score Distribution",
        ax: Optional[Axes] = None,
    ) -> Figure:
        """Histogram of bootstrap scores with optional observed-score line.

        Parameters
        ----------
        bootstrap_scores :
            List of scores from bootstrap resamples.
        observed_score :
            Optional observed (original) score to mark.
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

        ax.hist(bootstrap_scores, bins=bins, edgecolor="black", alpha=0.7,
                color="steelblue", density=True)

        if observed_score is not None:
            ax.axvline(x=observed_score, color="red", linestyle="--", linewidth=2,
                       label=f"Observed ({observed_score:.3f})")
            ax.legend(fontsize=9)

        # Add CI
        if len(bootstrap_scores) >= 2:
            lower = np.percentile(bootstrap_scores, 2.5)
            upper = np.percentile(bootstrap_scores, 97.5)
            ax.axvline(x=lower, color="gray", linestyle=":", linewidth=1)
            ax.axvline(x=upper, color="gray", linestyle=":", linewidth=1)
            ax.axvspan(lower, upper, alpha=0.1, color="gray",
                       label=f"95% CI [{lower:.3f}, {upper:.3f}]")
            ax.legend(fontsize=9)

        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return fig

    # ------------------------------------------------------------------
    # DAG with edge-certificate coloring
    # ------------------------------------------------------------------

    @staticmethod
    def display_edge_certificates(
        dag: DAG,
        certificates: Dict[Tuple[int, int], float],
        cmap: str = "RdYlGn",
        node_labels: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (8, 6),
        title: str = "Edge Certificates",
        ax: Optional[Axes] = None,
    ) -> Figure:
        """Render a DAG with edges coloured by certificate strength.

        Parameters
        ----------
        dag :
            DAG object with an ``adjacency`` property.
        certificates :
            Mapping ``(source, target) → strength`` in ``[0, 1]``.
        cmap :
            Colormap name.
        node_labels :
            Optional node labels.
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

        adj = dag.adjacency if hasattr(dag, 'adjacency') else dag._adj.copy()
        n = adj.shape[0]

        G = nx.DiGraph()
        G.add_nodes_from(range(n))

        edge_colors: List[float] = []
        for i in range(n):
            for j in range(n):
                if adj[i, j]:
                    G.add_edge(i, j)
                    edge_colors.append(certificates.get((i, j), 0.0))

        pos = nx.spring_layout(G, seed=42)
        colormap = plt.colormaps.get_cmap(cmap)

        labels = (
            {i: node_labels[i] for i in range(min(len(node_labels), n))}
            if node_labels else {i: str(i) for i in range(n)}
        )

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color="lightblue", node_size=600)
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=10)
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color=edge_colors,
            edge_cmap=colormap,
            edge_vmin=0.0,
            edge_vmax=1.0,
            arrowsize=15,
            width=2.0,
        )

        sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Certificate Strength")
        ax.set_title(title)
        ax.axis("off")
        return fig

    # ------------------------------------------------------------------
    # Certificate strength histogram
    # ------------------------------------------------------------------

    @staticmethod
    def display_certificate_histogram(
        certificates: List,
        bins: int = 20,
        figsize: Tuple[float, float] = (8, 5),
        title: str = "Edge Certificate Distribution",
        ax: Optional[Axes] = None,
    ) -> Figure:
        """Histogram of certificate strength values.

        Parameters
        ----------
        certificates :
            List of objects with a ``strength`` attribute, or list of floats.
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

        if certificates and hasattr(certificates[0], 'strength'):
            strengths = [c.strength for c in certificates]
        else:
            strengths = [float(c) for c in certificates]

        ax.hist(strengths, bins=bins, range=(0, 1), edgecolor="black",
                alpha=0.7, color="steelblue")

        mean_s = np.mean(strengths) if strengths else 0
        ax.axvline(x=mean_s, color="red", linestyle="--", linewidth=1.5,
                   label=f"Mean ({mean_s:.3f})")
        ax.legend(fontsize=9)
        ax.set_xlabel("Certificate Strength")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return fig

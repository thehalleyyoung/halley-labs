"""DAG visualization for the CPA engine.

Provides :class:`DAGVisualizer` for drawing causal graphs with
classification-colored edges, alignment overlays, and mechanism
change animations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from cpa.utils.logging import get_logger

logger = get_logger("visualization.dag")

_MPL_AVAILABLE = True
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.patches import FancyArrowPatch, ArrowStyle
except ImportError:
    _MPL_AVAILABLE = False

_NX_AVAILABLE = True
try:
    import networkx as nx
except ImportError:
    _NX_AVAILABLE = False


def _require_deps() -> None:
    if not _MPL_AVAILABLE:
        raise ImportError("matplotlib required: pip install matplotlib")
    if not _NX_AVAILABLE:
        raise ImportError("networkx required: pip install networkx")


# Edge classification colors
EDGE_COLORS: Dict[str, str] = {
    "shared": "#4CAF50",
    "modified": "#FF9800",
    "context_specific": "#F44336",
    "unknown": "#9E9E9E",
}

NODE_COLORS: Dict[str, str] = {
    "invariant": "#E8F5E9",
    "structurally_plastic": "#E3F2FD",
    "parametrically_plastic": "#FFF3E0",
    "fully_plastic": "#FFEBEE",
    "emergent": "#F3E5F5",
    "context_sensitive": "#E0F7FA",
    "unclassified": "#F5F5F5",
}


class DAGVisualizer:
    """Visualize causal DAGs with plasticity information.

    Parameters
    ----------
    figsize : tuple of float
        Default figure size.
    dpi : int
        Resolution for saved figures.
    layout : str
        Default layout algorithm ('spring', 'dot', 'kamada_kawai', 'circular').
    node_size : int
        Default node size.

    Examples
    --------
    >>> viz = DAGVisualizer()
    >>> viz.draw_dag(adjacency, variable_names, save_path="dag.png")
    """

    def __init__(
        self,
        figsize: Tuple[float, float] = (10, 8),
        dpi: int = 150,
        layout: str = "spring",
        node_size: int = 800,
    ) -> None:
        _require_deps()
        self._figsize = figsize
        self._dpi = dpi
        self._layout = layout
        self._node_size = node_size

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

    def _adj_to_graph(
        self,
        adjacency: np.ndarray,
        variable_names: Optional[List[str]] = None,
    ) -> "nx.DiGraph":
        """Convert adjacency matrix to networkx DiGraph."""
        p = adjacency.shape[0]
        G = nx.DiGraph()

        names = variable_names or [f"X{i}" for i in range(p)]
        G.add_nodes_from(names)

        for i in range(p):
            for j in range(p):
                if adjacency[i, j] != 0:
                    G.add_edge(names[i], names[j], weight=float(adjacency[i, j]))

        return G

    def _get_layout(
        self, G: "nx.DiGraph", layout: Optional[str] = None
    ) -> Dict[str, Tuple[float, float]]:
        """Compute node positions."""
        method = layout or self._layout

        if method == "spring":
            return nx.spring_layout(G, seed=42, k=2.0 / max(len(G), 1) ** 0.5)
        elif method == "kamada_kawai":
            try:
                return nx.kamada_kawai_layout(G)
            except Exception:
                return nx.spring_layout(G, seed=42)
        elif method == "circular":
            return nx.circular_layout(G)
        elif method == "shell":
            return nx.shell_layout(G)
        elif method == "spectral":
            try:
                return nx.spectral_layout(G)
            except Exception:
                return nx.spring_layout(G, seed=42)
        elif method == "dot":
            try:
                return nx.nx_agraph.graphviz_layout(G, prog="dot")
            except Exception:
                return nx.spring_layout(G, seed=42)
        else:
            return nx.spring_layout(G, seed=42)

    # -----------------------------------------------------------------
    # Basic DAG drawing
    # -----------------------------------------------------------------

    def draw_dag(
        self,
        adjacency: np.ndarray,
        variable_names: Optional[List[str]] = None,
        save_path: Optional[Union[str, Path]] = None,
        title: str = "Causal DAG",
        node_colors: Optional[Dict[str, str]] = None,
        edge_colors: Optional[Dict[Tuple[str, str], str]] = None,
        layout: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        show_weights: bool = False,
    ) -> "Figure":
        """Draw a causal DAG.

        Parameters
        ----------
        adjacency : np.ndarray
            (p, p) adjacency matrix.
        variable_names : list of str, optional
            Variable names.
        save_path : str or Path, optional
            Save path.
        title : str
            Figure title.
        node_colors : dict, optional
            Variable → color mapping.
        edge_colors : dict, optional
            (source, target) → color mapping.
        layout : str, optional
            Layout algorithm.
        figsize : tuple, optional
            Figure size.
        show_weights : bool
            Show edge weights.

        Returns
        -------
        Figure
        """
        p = adjacency.shape[0]
        names = variable_names or [f"X{i}" for i in range(p)]

        G = self._adj_to_graph(adjacency, names)
        pos = self._get_layout(G, layout)

        fig, ax = plt.subplots(figsize=figsize or self._figsize)

        n_colors = []
        for name in G.nodes():
            if node_colors and name in node_colors:
                n_colors.append(node_colors[name])
            else:
                n_colors.append("#E3F2FD")

        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=n_colors,
            node_size=self._node_size,
            edgecolors="black",
            linewidths=1.5,
        )

        nx.draw_networkx_labels(
            G, pos, ax=ax,
            font_size=9,
            font_weight="bold",
        )

        e_colors = []
        widths = []
        for u, v, data in G.edges(data=True):
            if edge_colors and (u, v) in edge_colors:
                e_colors.append(edge_colors[(u, v)])
            else:
                e_colors.append("#333333")
            weight = abs(data.get("weight", 1.0))
            widths.append(max(0.5, min(4.0, weight * 2)))

        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color=e_colors,
            width=widths,
            arrowsize=15,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1",
        )

        if show_weights:
            edge_labels = {
                (u, v): f"{data['weight']:.2f}"
                for u, v, data in G.edges(data=True)
            }
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels, ax=ax, font_size=7,
            )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

        return self._save_or_show(fig, save_path)

    # -----------------------------------------------------------------
    # Classification-colored DAG
    # -----------------------------------------------------------------

    def draw_classified_dag(
        self,
        adjacency: np.ndarray,
        variable_names: List[str],
        classifications: Dict[str, str],
        save_path: Optional[Union[str, Path]] = None,
        title: str = "Classified Causal DAG",
        layout: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> "Figure":
        """Draw a DAG with nodes colored by classification.

        Parameters
        ----------
        adjacency : np.ndarray
            (p, p) adjacency matrix.
        variable_names : list of str
            Variable names.
        classifications : dict of str → str
            Variable → classification mapping.
        save_path : str or Path, optional
            Save path.
        title : str
            Figure title.
        layout : str, optional
            Layout algorithm.
        figsize : tuple, optional
            Figure size.

        Returns
        -------
        Figure
        """
        node_colors = {
            var: NODE_COLORS.get(cls, "#F5F5F5")
            for var, cls in classifications.items()
        }

        fig = self.draw_dag(
            adjacency, variable_names,
            save_path=None,
            title=title,
            node_colors=node_colors,
            layout=layout,
            figsize=figsize,
        )

        from matplotlib.patches import Patch
        used_classes = set(classifications.values())
        legend_elements = [
            Patch(
                facecolor=NODE_COLORS.get(cls, "#F5F5F5"),
                edgecolor="black",
                label=cls.replace("_", " ").title(),
            )
            for cls in sorted(used_classes)
        ]
        fig.axes[0].legend(
            handles=legend_elements,
            loc="upper left",
            fontsize=8,
            framealpha=0.9,
        )

        return self._save_or_show(fig, save_path)

    # -----------------------------------------------------------------
    # Alignment visualization (side-by-side DAGs)
    # -----------------------------------------------------------------

    def draw_alignment(
        self,
        adj_i: np.ndarray,
        adj_j: np.ndarray,
        variable_names: List[str],
        context_i: str = "Context A",
        context_j: str = "Context B",
        permutation: Optional[np.ndarray] = None,
        save_path: Optional[Union[str, Path]] = None,
        layout: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> "Figure":
        """Draw two DAGs side-by-side with alignment information.

        Edges are colored by classification:
        - Green: shared (present in both)
        - Orange: modified (present in both, different weight)
        - Red: context-specific (only in one)

        Parameters
        ----------
        adj_i, adj_j : np.ndarray
            Adjacency matrices for each context.
        variable_names : list of str
            Variable names.
        context_i, context_j : str
            Context labels.
        permutation : np.ndarray, optional
            Variable permutation mapping.
        save_path : str or Path, optional
            Save path.
        layout : str, optional
            Layout algorithm.
        figsize : tuple, optional
            Figure size.

        Returns
        -------
        Figure
        """
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=figsize or (16, 7)
        )

        p = adj_i.shape[0]
        names = variable_names[:p]

        if permutation is not None:
            adj_j_aligned = adj_j[np.ix_(permutation, permutation)]
        else:
            adj_j_aligned = adj_j

        bin_i = (adj_i != 0).astype(int)
        bin_j = (adj_j_aligned != 0).astype(int)

        # Classify edges
        edge_colors_i: Dict[Tuple[str, str], str] = {}
        edge_colors_j: Dict[Tuple[str, str], str] = {}

        for s in range(p):
            for t in range(p):
                if s == t:
                    continue
                in_i = bin_i[s, t] > 0
                in_j = bin_j[s, t] > 0

                sn, tn = names[s], names[t]

                if in_i and in_j:
                    if abs(adj_i[s, t] - adj_j_aligned[s, t]) > 0.01:
                        edge_colors_i[(sn, tn)] = EDGE_COLORS["modified"]
                        edge_colors_j[(sn, tn)] = EDGE_COLORS["modified"]
                    else:
                        edge_colors_i[(sn, tn)] = EDGE_COLORS["shared"]
                        edge_colors_j[(sn, tn)] = EDGE_COLORS["shared"]
                elif in_i:
                    edge_colors_i[(sn, tn)] = EDGE_COLORS["context_specific"]
                elif in_j:
                    edge_colors_j[(sn, tn)] = EDGE_COLORS["context_specific"]

        G_i = self._adj_to_graph(adj_i, names)
        G_j = self._adj_to_graph(adj_j_aligned, names)

        pos = self._get_layout(G_i, layout)

        self._draw_on_axis(ax1, G_i, pos, edge_colors_i, context_i)
        self._draw_on_axis(ax2, G_j, pos, edge_colors_j, context_j)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=EDGE_COLORS["shared"], label="Shared"),
            Patch(facecolor=EDGE_COLORS["modified"], label="Modified"),
            Patch(facecolor=EDGE_COLORS["context_specific"], label="Context-specific"),
        ]
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=3,
            fontsize=9,
            framealpha=0.9,
        )

        fig.suptitle(
            f"Alignment: {context_i} ↔ {context_j}",
            fontsize=13, fontweight="bold",
        )

        return self._save_or_show(fig, save_path)

    def _draw_on_axis(
        self,
        ax: Any,
        G: "nx.DiGraph",
        pos: Dict[str, Tuple[float, float]],
        edge_colors: Dict[Tuple[str, str], str],
        title: str,
    ) -> None:
        """Draw a graph on a given axis."""
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color="#E3F2FD",
            node_size=self._node_size,
            edgecolors="black",
            linewidths=1,
        )
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

        e_colors = []
        for u, v in G.edges():
            e_colors.append(edge_colors.get((u, v), "#333333"))

        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color=e_colors,
            width=2.0,
            arrowsize=12,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1",
        )

        ax.set_title(title, fontsize=11)
        ax.axis("off")

    # -----------------------------------------------------------------
    # Plasticity overlay
    # -----------------------------------------------------------------

    def draw_dag_with_plasticity(
        self,
        adjacency: np.ndarray,
        variable_names: List[str],
        descriptors: Dict[str, Any],
        save_path: Optional[Union[str, Path]] = None,
        component: str = "structural",
        cmap: str = "YlOrRd",
        layout: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> "Figure":
        """Draw DAG with nodes colored by plasticity score.

        Parameters
        ----------
        adjacency : np.ndarray
            Adjacency matrix.
        variable_names : list of str
            Variable names.
        descriptors : dict
            Variable → descriptor (with component attributes).
        save_path : str or Path, optional
            Save path.
        component : str
            Which descriptor component to color by.
        cmap : str
            Colormap name.
        layout : str, optional
            Layout algorithm.
        figsize : tuple, optional
            Figure size.

        Returns
        -------
        Figure
        """
        p = adjacency.shape[0]
        names = variable_names[:p]

        G = self._adj_to_graph(adjacency, names)
        pos = self._get_layout(G, layout)

        fig, ax = plt.subplots(figsize=figsize or self._figsize)

        values = []
        for name in G.nodes():
            dr = descriptors.get(name)
            if dr is not None:
                val = getattr(dr, component, 0.0) if hasattr(dr, component) else dr.get(component, 0.0)
                values.append(val)
            else:
                values.append(0.0)

        colormap = plt.cm.get_cmap(cmap)
        node_colors = [colormap(v) for v in values]

        nodes = nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=values,
            cmap=colormap,
            vmin=0, vmax=1,
            node_size=self._node_size,
            edgecolors="black",
            linewidths=1.5,
        )

        nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color="#555555",
            width=1.5,
            arrowsize=12,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1",
        )

        sm = plt.cm.ScalarMappable(
            cmap=colormap, norm=plt.Normalize(vmin=0, vmax=1)
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
        cbar.set_label(f"{component.title()} Score")

        ax.set_title(
            f"DAG with {component.title()} Plasticity",
            fontsize=12, fontweight="bold",
        )
        ax.axis("off")

        return self._save_or_show(fig, save_path)

    # -----------------------------------------------------------------
    # Multi-context animation frames
    # -----------------------------------------------------------------

    def draw_context_sequence(
        self,
        adjacencies: Dict[str, np.ndarray],
        variable_names: List[str],
        save_dir: Optional[Union[str, Path]] = None,
        layout: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> List["Figure"]:
        """Draw DAGs for each context in sequence.

        Useful for creating animations of mechanism changes.

        Parameters
        ----------
        adjacencies : dict of str → np.ndarray
            Context → adjacency matrix mapping.
        variable_names : list of str
            Variable names.
        save_dir : str or Path, optional
            Directory to save frames.
        layout : str, optional
            Layout algorithm (shared across all frames).
        figsize : tuple, optional
            Figure size.

        Returns
        -------
        list of Figure
        """
        if not adjacencies:
            return []

        first_key = list(adjacencies.keys())[0]
        G_ref = self._adj_to_graph(adjacencies[first_key], variable_names)
        shared_pos = self._get_layout(G_ref, layout)

        figures: List["Figure"] = []

        for ctx_idx, (ctx_id, adj) in enumerate(adjacencies.items()):
            fig, ax = plt.subplots(figsize=figsize or self._figsize)

            G = self._adj_to_graph(adj, variable_names)

            nx.draw_networkx_nodes(
                G, shared_pos, ax=ax,
                node_color="#E3F2FD",
                node_size=self._node_size,
                edgecolors="black",
                linewidths=1,
            )
            nx.draw_networkx_labels(G, shared_pos, ax=ax, font_size=9)
            nx.draw_networkx_edges(
                G, shared_pos, ax=ax,
                edge_color="#2196F3",
                width=2.0,
                arrowsize=12,
                arrowstyle="-|>",
                connectionstyle="arc3,rad=0.1",
            )

            n_edges = int(np.sum(adj != 0))
            ax.set_title(
                f"{ctx_id} ({n_edges} edges)",
                fontsize=12, fontweight="bold",
            )
            ax.axis("off")

            if save_dir is not None:
                save_path = Path(save_dir) / f"frame_{ctx_idx:03d}_{ctx_id}.png"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path, dpi=self._dpi, bbox_inches="tight")
                plt.close(fig)

            figures.append(fig)

        return figures

    # -----------------------------------------------------------------
    # Diff DAG
    # -----------------------------------------------------------------

    def draw_dag_diff(
        self,
        adj_i: np.ndarray,
        adj_j: np.ndarray,
        variable_names: List[str],
        context_i: str = "Context A",
        context_j: str = "Context B",
        save_path: Optional[Union[str, Path]] = None,
        layout: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> "Figure":
        """Draw the edge difference between two DAGs.

        Shows edges added (green), removed (red), and modified (orange).

        Parameters
        ----------
        adj_i, adj_j : np.ndarray
            Adjacency matrices.
        variable_names : list of str
            Variable names.
        context_i, context_j : str
            Context labels.
        save_path : str or Path, optional
            Save path.
        layout : str, optional
            Layout algorithm.
        figsize : tuple, optional
            Figure size.

        Returns
        -------
        Figure
        """
        p = adj_i.shape[0]
        names = variable_names[:p]

        G_union = nx.DiGraph()
        G_union.add_nodes_from(names)

        edge_colors: Dict[Tuple[str, str], str] = {}

        for s in range(p):
            for t in range(p):
                if s == t:
                    continue
                in_i = adj_i[s, t] != 0
                in_j = adj_j[s, t] != 0
                sn, tn = names[s], names[t]

                if in_i and in_j:
                    G_union.add_edge(sn, tn)
                    if abs(adj_i[s, t] - adj_j[s, t]) > 0.01:
                        edge_colors[(sn, tn)] = "#FF9800"  # modified
                    else:
                        edge_colors[(sn, tn)] = "#9E9E9E"  # unchanged
                elif in_i:
                    G_union.add_edge(sn, tn)
                    edge_colors[(sn, tn)] = "#F44336"  # removed
                elif in_j:
                    G_union.add_edge(sn, tn)
                    edge_colors[(sn, tn)] = "#4CAF50"  # added

        pos = self._get_layout(G_union, layout)

        fig, ax = plt.subplots(figsize=figsize or self._figsize)

        nx.draw_networkx_nodes(
            G_union, pos, ax=ax,
            node_color="#E3F2FD",
            node_size=self._node_size,
            edgecolors="black",
            linewidths=1,
        )
        nx.draw_networkx_labels(G_union, pos, ax=ax, font_size=9)

        e_cols = [edge_colors.get((u, v), "#9E9E9E") for u, v in G_union.edges()]
        nx.draw_networkx_edges(
            G_union, pos, ax=ax,
            edge_color=e_cols,
            width=2.0,
            arrowsize=12,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1",
        )

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#9E9E9E", label="Unchanged"),
            Patch(facecolor="#FF9800", label="Modified"),
            Patch(facecolor="#F44336", label=f"Only in {context_i}"),
            Patch(facecolor="#4CAF50", label=f"Only in {context_j}"),
        ]
        ax.legend(
            handles=legend_elements, loc="upper left", fontsize=8,
        )

        ax.set_title(
            f"Edge Diff: {context_i} → {context_j}",
            fontsize=12, fontweight="bold",
        )
        ax.axis("off")

        return self._save_or_show(fig, save_path)

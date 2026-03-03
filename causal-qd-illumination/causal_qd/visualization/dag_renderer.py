"""DAG rendering utilities using networkx for layout.

Provides comprehensive DAG visualization:
  - render_dag: render a single DAG
  - render_cpdag: render CPDAG with directed/undirected edges
  - render_comparison: side-by-side DAG comparison with diff highlighting
  - highlight_differences: color edges that differ between two DAGs
  - render_with_certificates: color edges by certificate strength
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from matplotlib.figure import Figure

from causal_qd.types import AdjacencyMatrix

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from causal_qd.core.dag import DAG


def _adj_to_nx(adj: np.ndarray) -> nx.DiGraph:
    """Convert an adjacency matrix to a networkx DiGraph."""
    G = nx.DiGraph()
    n = adj.shape[0]
    G.add_nodes_from(range(n))
    rows, cols = np.nonzero(adj)
    G.add_edges_from(zip(rows.tolist(), cols.tolist()))
    return G


def _get_adj(dag: DAG) -> np.ndarray:
    """Extract adjacency matrix from DAG, supporting both property styles."""
    if hasattr(dag, 'adjacency') and callable(getattr(dag.__class__, 'adjacency', None)):
        return dag.adjacency
    elif hasattr(dag, 'adjacency_matrix'):
        return dag.adjacency_matrix
    elif hasattr(dag, '_adj'):
        return dag._adj.copy()
    return dag.adjacency


class DAGRenderer:
    """Renders directed acyclic graphs using *networkx* layouts.

    All methods are static for convenient use without instantiation.
    """

    # ------------------------------------------------------------------
    # Single DAG rendering
    # ------------------------------------------------------------------

    @staticmethod
    def render(
        dag: DAG,
        node_labels: Optional[List[str]] = None,
        node_color: str = "lightblue",
        edge_color: str = "black",
        node_size: int = 600,
        font_size: int = 10,
        layout: str = "spring",
        title: str = "DAG",
        figsize: Tuple[float, float] = (8, 6),
        ax: Optional[Axes] = None,
    ) -> Figure:
        """Draw a single DAG.

        Parameters
        ----------
        dag :
            DAG object with an ``adjacency`` property.
        node_labels :
            Optional human-readable labels for each node.
        node_color :
            Color for all nodes.
        edge_color :
            Color for all edges.
        node_size :
            Size of node markers.
        font_size :
            Font size for labels.
        layout :
            Layout algorithm: ``"spring"``, ``"shell"``, ``"kamada_kawai"``,
            ``"circular"``, ``"spectral"``.
        title :
            Plot title.
        figsize :
            Figure size.
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

        adj = _get_adj(dag)
        G = _adj_to_nx(adj)
        pos = DAGRenderer._get_layout(G, layout)

        labels = (
            {i: node_labels[i] for i in range(len(node_labels))}
            if node_labels is not None
            else {i: str(i) for i in G.nodes()}
        )

        nx.draw_networkx(
            G, pos, ax=ax,
            labels=labels,
            node_color=node_color,
            node_size=node_size,
            arrowsize=15,
            font_size=font_size,
            edge_color=edge_color,
        )
        ax.set_title(title)
        ax.axis("off")
        return fig

    # Alias
    render_dag = render

    # ------------------------------------------------------------------
    # CPDAG rendering
    # ------------------------------------------------------------------

    @staticmethod
    def render_cpdag(
        cpdag: AdjacencyMatrix,
        node_labels: Optional[List[str]] = None,
        directed_color: str = "black",
        undirected_color: str = "gray",
        undirected_style: str = "dashed",
        node_color: str = "lightblue",
        node_size: int = 600,
        layout: str = "spring",
        title: str = "CPDAG",
        figsize: Tuple[float, float] = (8, 6),
        ax: Optional[Axes] = None,
    ) -> Figure:
        """Render a CPDAG with directed and undirected edges distinguished.

        Undirected edges (``cpdag[i,j] == 1`` and ``cpdag[j,i] == 1``)
        are drawn with dashed lines without arrowheads.

        Parameters
        ----------
        cpdag :
            Adjacency matrix of the CPDAG.
        node_labels :
            Optional labels.
        directed_color :
            Color for directed edges.
        undirected_color :
            Color for undirected edges.
        undirected_style :
            Linestyle for undirected edges.
        node_color :
            Color for nodes.
        node_size :
            Node marker size.
        layout :
            Layout algorithm.
        title :
            Plot title.
        figsize :
            Figure size.
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

        cpdag = np.asarray(cpdag)
        n = cpdag.shape[0]

        directed_edges: List[Tuple[int, int]] = []
        undirected_edges: List[Tuple[int, int]] = []
        visited: Set[Tuple[int, int]] = set()

        for i in range(n):
            for j in range(n):
                if cpdag[i, j] and (i, j) not in visited:
                    if cpdag[j, i]:
                        undirected_edges.append((i, j))
                        visited.add((i, j))
                        visited.add((j, i))
                    else:
                        directed_edges.append((i, j))
                        visited.add((i, j))

        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        G.add_edges_from(directed_edges)

        all_edges_graph = nx.Graph(undirected_edges)
        all_edges_graph.add_edges_from(directed_edges)
        all_edges_graph.add_nodes_from(range(n))
        pos = DAGRenderer._get_layout(all_edges_graph, layout)

        labels = (
            {i: node_labels[i] for i in range(min(len(node_labels), n))}
            if node_labels is not None
            else {i: str(i) for i in range(n)}
        )

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_color, node_size=node_size)
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=10)

        # Directed edges with arrows
        if directed_edges:
            nx.draw_networkx_edges(
                G, pos, edgelist=directed_edges, ax=ax,
                arrowsize=15, edge_color=directed_color,
            )

        # Undirected edges without arrows
        if undirected_edges:
            U = nx.Graph()
            U.add_edges_from(undirected_edges)
            nx.draw_networkx_edges(
                U, pos, ax=ax,
                style=undirected_style,
                edge_color=undirected_color,
                arrows=False,
            )

        # Legend
        handles = [
            mpatches.Patch(color=directed_color, label="Directed"),
            mpatches.Patch(color=undirected_color, label="Undirected"),
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=9)
        ax.set_title(title)
        ax.axis("off")
        return fig

    # ------------------------------------------------------------------
    # Side-by-side comparison
    # ------------------------------------------------------------------

    @staticmethod
    def render_comparison(
        dag1: DAG,
        dag2: DAG,
        labels: Optional[Tuple[str, str]] = None,
        node_labels: Optional[List[str]] = None,
        layout: str = "spring",
        figsize: Tuple[float, float] = (14, 6),
        ax: Optional[Axes] = None,
    ) -> Figure:
        """Side-by-side comparison of two DAGs with diff highlighting.

        Edge colors:
        - **Black**: edge present in both DAGs
        - **Green**: edge present only in DAG 1
        - **Red**: edge present only in DAG 2

        Parameters
        ----------
        dag1, dag2 :
            DAG objects to compare.
        labels :
            Titles for each panel.
        node_labels :
            Node labels.
        layout :
            Layout algorithm.
        figsize :
            Figure size.
        ax :
            Ignored; a new two-panel figure is always created.

        Returns
        -------
        Figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        adj1, adj2 = _get_adj(dag1), _get_adj(dag2)
        G1, G2 = _adj_to_nx(adj1), _adj_to_nx(adj2)

        # Shared layout
        all_nodes = set(G1.nodes()) | set(G2.nodes())
        combined = nx.Graph()
        combined.add_nodes_from(all_nodes)
        combined.add_edges_from(G1.edges())
        combined.add_edges_from(G2.edges())
        pos = DAGRenderer._get_layout(combined, layout)

        shared = set(G1.edges()) & set(G2.edges())
        only1 = set(G1.edges()) - shared
        only2 = set(G2.edges()) - shared

        n_labels = (
            {i: node_labels[i] for i in range(len(node_labels))}
            if node_labels else {i: str(i) for i in all_nodes}
        )

        # DAG 1
        edge_colors_1 = ["green" if e in only1 else "black" for e in G1.edges()]
        edge_widths_1 = [2.5 if e in only1 else 1.5 for e in G1.edges()]
        nx.draw_networkx(
            G1, pos, ax=ax1, labels=n_labels,
            edge_color=edge_colors_1, width=edge_widths_1,
            node_color="lightblue", node_size=500, arrowsize=12,
        )
        ax1.set_title(labels[0] if labels else "DAG 1")
        ax1.axis("off")

        # DAG 2
        edge_colors_2 = ["red" if e in only2 else "black" for e in G2.edges()]
        edge_widths_2 = [2.5 if e in only2 else 1.5 for e in G2.edges()]
        nx.draw_networkx(
            G2, pos, ax=ax2, labels=n_labels,
            edge_color=edge_colors_2, width=edge_widths_2,
            node_color="lightyellow", node_size=500, arrowsize=12,
        )
        ax2.set_title(labels[1] if labels else "DAG 2")
        ax2.axis("off")

        fig.suptitle("DAG Comparison (green=only left, red=only right)")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Highlight differences
    # ------------------------------------------------------------------

    @staticmethod
    def highlight_differences(
        pred_adj: AdjacencyMatrix,
        true_adj: AdjacencyMatrix,
        node_labels: Optional[List[str]] = None,
        layout: str = "spring",
        title: str = "Edge Differences (pred vs true)",
        figsize: Tuple[float, float] = (8, 6),
        ax: Optional[Axes] = None,
    ) -> Figure:
        """Color edges by comparison to ground truth.

        Edge colors:
        - **Black**: correct (TP)
        - **Red**: extra (FP, in pred but not true)
        - **Blue** dashed: missing (FN, in true but not pred)

        Parameters
        ----------
        pred_adj :
            Predicted adjacency matrix.
        true_adj :
            True adjacency matrix.
        node_labels :
            Optional node labels.
        layout :
            Layout algorithm.
        title :
            Plot title.
        figsize :
            Figure size.
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

        pred_adj = np.asarray(pred_adj)
        true_adj = np.asarray(true_adj)
        n = pred_adj.shape[0]

        G = nx.DiGraph()
        G.add_nodes_from(range(n))

        tp_edges, fp_edges, fn_edges = [], [], []
        for i in range(n):
            for j in range(n):
                if pred_adj[i, j] and true_adj[i, j]:
                    tp_edges.append((i, j))
                elif pred_adj[i, j] and not true_adj[i, j]:
                    fp_edges.append((i, j))
                elif not pred_adj[i, j] and true_adj[i, j]:
                    fn_edges.append((i, j))

        G.add_edges_from(tp_edges + fp_edges + fn_edges)
        pos = DAGRenderer._get_layout(G, layout)

        labels = (
            {i: node_labels[i] for i in range(min(len(node_labels), n))}
            if node_labels else {i: str(i) for i in range(n)}
        )

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color="lightyellow", node_size=600)
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax)

        if tp_edges:
            nx.draw_networkx_edges(G, pos, edgelist=tp_edges, ax=ax,
                                   edge_color="black", width=2, arrowsize=12)
        if fp_edges:
            nx.draw_networkx_edges(G, pos, edgelist=fp_edges, ax=ax,
                                   edge_color="red", width=2, arrowsize=12)
        if fn_edges:
            nx.draw_networkx_edges(G, pos, edgelist=fn_edges, ax=ax,
                                   edge_color="blue", width=1.5, arrowsize=10,
                                   style="dashed")

        handles = [
            mpatches.Patch(color="black", label=f"Correct ({len(tp_edges)})"),
            mpatches.Patch(color="red", label=f"Extra ({len(fp_edges)})"),
            mpatches.Patch(color="blue", label=f"Missing ({len(fn_edges)})"),
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=9)
        ax.set_title(title)
        ax.axis("off")
        return fig

    # ------------------------------------------------------------------
    # Render with certificate strength
    # ------------------------------------------------------------------

    @staticmethod
    def render_with_certificates(
        dag: DAG,
        certificates: Dict[Tuple[int, int], float],
        cmap: str = "RdYlGn",
        node_labels: Optional[List[str]] = None,
        layout: str = "spring",
        title: str = "DAG with Edge Certificates",
        figsize: Tuple[float, float] = (8, 6),
        ax: Optional[Axes] = None,
    ) -> Figure:
        """Render a DAG with edges colored by certificate strength.

        Parameters
        ----------
        dag :
            DAG object.
        certificates :
            Mapping ``(source, target) → strength`` in ``[0, 1]``.
        cmap :
            Colormap name for edge coloring.
        node_labels :
            Optional node labels.
        layout :
            Layout algorithm.
        title :
            Plot title.
        figsize :
            Figure size.
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

        adj = _get_adj(dag)
        n = adj.shape[0]
        G = _adj_to_nx(adj)
        pos = DAGRenderer._get_layout(G, layout)

        edge_colors = [certificates.get((i, j), 0.0) for i, j in G.edges()]

        labels = (
            {i: node_labels[i] for i in range(min(len(node_labels), n))}
            if node_labels else {i: str(i) for i in range(n)}
        )

        colormap = plt.colormaps.get_cmap(cmap)
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
    # Layout helper
    # ------------------------------------------------------------------

    @staticmethod
    def _get_layout(G: nx.Graph, layout: str, seed: int = 42) -> Dict:
        """Get a networkx layout by name."""
        layout_fns = {
            "spring": lambda: nx.spring_layout(G, seed=seed),
            "shell": lambda: nx.shell_layout(G),
            "kamada_kawai": lambda: nx.kamada_kawai_layout(G),
            "circular": lambda: nx.circular_layout(G),
            "spectral": lambda: nx.spectral_layout(G),
        }
        fn = layout_fns.get(layout, layout_fns["spring"])
        try:
            return fn()
        except Exception:
            return nx.spring_layout(G, seed=seed)

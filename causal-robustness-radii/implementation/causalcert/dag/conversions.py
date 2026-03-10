"""
DAG serialisation and deserialisation — DOT, JSON, GML, BIF, and adjacency matrix formats.

Supports round-tripping DAGs through common file formats for interoperability
with Graphviz, pcalg, bnlearn, and custom tooling.
"""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from causalcert.types import AdjacencyMatrix


def to_dot(adj: AdjacencyMatrix, node_names: list[str] | None = None) -> str:
    """Serialise a DAG to Graphviz DOT format.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    node_names : list[str] | None
        Optional human-readable node names.

    Returns
    -------
    str
        DOT-language string.
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]
    names = node_names or [str(i) for i in range(n)]

    lines = ["digraph G {"]
    # Declare nodes
    for i in range(n):
        label = names[i]
        # Escape quotes in label
        label_escaped = label.replace('"', '\\"')
        lines.append(f'  "{label_escaped}";')
    # Declare edges
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                src = names[i].replace('"', '\\"')
                tgt = names[j].replace('"', '\\"')
                lines.append(f'  "{src}" -> "{tgt}";')
    lines.append("}")
    return "\n".join(lines)


def from_dot(dot_str: str) -> tuple[AdjacencyMatrix, list[str]]:
    """Parse a DAG from a Graphviz DOT string.

    Supports basic DOT syntax with ``digraph`` and ``->`` edges.

    Parameters
    ----------
    dot_str : str
        DOT-language string.

    Returns
    -------
    tuple[AdjacencyMatrix, list[str]]
        ``(adjacency_matrix, node_names)``.
    """
    # Extract node names and edges
    nodes: list[str] = []
    node_set: set[str] = set()
    edges: list[tuple[str, str]] = []

    # Match edges: "A" -> "B" or A -> B
    edge_pattern = re.compile(
        r'"([^"]+)"\s*->\s*"([^"]+)"'
        r'|'
        r'(\w+)\s*->\s*(\w+)'
    )

    for match in edge_pattern.finditer(dot_str):
        if match.group(1) is not None:
            src, tgt = match.group(1), match.group(2)
        else:
            src, tgt = match.group(3), match.group(4)
        if src not in node_set:
            node_set.add(src)
            nodes.append(src)
        if tgt not in node_set:
            node_set.add(tgt)
            nodes.append(tgt)
        edges.append((src, tgt))

    # Also look for standalone node declarations: "A"; or A;
    node_pattern = re.compile(
        r'^\s*"([^"]+)"\s*[\[;]'
        r'|'
        r'^\s*(\w+)\s*[\[;]',
        re.MULTILINE,
    )
    for match in node_pattern.finditer(dot_str):
        name = match.group(1) or match.group(2)
        if name and name not in node_set and name not in ("digraph", "graph", "subgraph", "node", "edge"):
            node_set.add(name)
            nodes.append(name)

    # Build adjacency matrix
    name_to_idx = {name: i for i, name in enumerate(nodes)}
    n = len(nodes)
    adj = np.zeros((n, n), dtype=np.int8)
    for src, tgt in edges:
        adj[name_to_idx[src], name_to_idx[tgt]] = 1

    return adj, nodes


def to_json(
    adj: AdjacencyMatrix,
    node_names: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Serialise a DAG to a JSON string.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    node_names : list[str] | None
        Optional node names.
    metadata : dict[str, Any] | None
        Optional metadata to include.

    Returns
    -------
    str
        JSON string.
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]
    names = node_names or [str(i) for i in range(n)]

    nodes_list = [{"id": i, "name": names[i]} for i in range(n)]
    edges_list = []
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                edges_list.append({"source": i, "target": j})

    data: dict[str, Any] = {
        "nodes": nodes_list,
        "edges": edges_list,
        "n_nodes": n,
        "n_edges": int(adj.sum()),
    }
    if metadata:
        data["metadata"] = metadata

    return json.dumps(data, indent=2)


def from_json(json_str: str) -> tuple[AdjacencyMatrix, list[str], dict[str, Any]]:
    """Parse a DAG from a JSON string.

    Parameters
    ----------
    json_str : str
        JSON string as produced by :func:`to_json`.

    Returns
    -------
    tuple[AdjacencyMatrix, list[str], dict[str, Any]]
        ``(adjacency_matrix, node_names, metadata)``.
    """
    data = json.loads(json_str)

    nodes = data.get("nodes", [])
    n = len(nodes)
    names = [node.get("name", str(node.get("id", i))) for i, node in enumerate(nodes)]

    adj = np.zeros((n, n), dtype=np.int8)
    for edge in data.get("edges", []):
        src = edge["source"]
        tgt = edge["target"]
        adj[src, tgt] = 1

    metadata = data.get("metadata", {})
    return adj, names, metadata


def to_adjacency_csv(
    adj: AdjacencyMatrix,
    path: str | Path,
    node_names: list[str] | None = None,
) -> None:
    """Write the adjacency matrix to a CSV file.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    path : str | Path
        Output file path.
    node_names : list[str] | None
        Optional node names for header row.
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]
    names = node_names or [str(i) for i in range(n)]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header row
        writer.writerow([""] + names)
        for i in range(n):
            writer.writerow([names[i]] + [int(adj[i, j]) for j in range(n)])


def from_adjacency_csv(path: str | Path) -> AdjacencyMatrix:
    """Load an adjacency matrix from a CSV file.

    Parameters
    ----------
    path : str | Path
        Input file path.

    Returns
    -------
    AdjacencyMatrix
    """
    with open(path, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return np.zeros((0, 0), dtype=np.int8)

    # Check if first row is a header
    has_header = not rows[0][0].lstrip("-").isdigit()

    if has_header:
        data_rows = rows[1:]
    else:
        data_rows = rows

    n = len(data_rows)
    adj = np.zeros((n, n), dtype=np.int8)
    for i, row in enumerate(data_rows):
        # Skip first column if it's a label
        start = 1 if has_header else 0
        for j, val in enumerate(row[start:]):
            adj[i, j] = int(float(val))

    return adj


def to_edge_list(
    adj: AdjacencyMatrix,
    node_names: list[str] | None = None,
) -> str:
    """Convert a DAG to an edge list string.

    Each line contains ``source target`` (space-separated).

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    node_names : list[str] | None
        Optional node names.

    Returns
    -------
    str
        Edge list as a string.
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]
    names = node_names or [str(i) for i in range(n)]
    lines: list[str] = []
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                lines.append(f"{names[i]} {names[j]}")
    return "\n".join(lines)


def from_edge_list(
    edge_list_str: str,
    node_names: list[str] | None = None,
) -> tuple[AdjacencyMatrix, list[str]]:
    """Parse a DAG from an edge list string.

    Parameters
    ----------
    edge_list_str : str
        Edge list, one ``source target`` pair per line.
    node_names : list[str] | None
        If given, use these as the canonical node order.

    Returns
    -------
    tuple[AdjacencyMatrix, list[str]]
        ``(adjacency_matrix, node_names)``.
    """
    nodes: list[str] = []
    node_set: set[str] = set()
    edges: list[tuple[str, str]] = []

    for line in edge_list_str.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            src, tgt = parts[0], parts[1]
            if src not in node_set:
                node_set.add(src)
                nodes.append(src)
            if tgt not in node_set:
                node_set.add(tgt)
                nodes.append(tgt)
            edges.append((src, tgt))

    if node_names:
        for name in node_names:
            if name not in node_set:
                node_set.add(name)
                nodes.append(name)
        # Reorder to match provided names
        name_to_idx = {name: i for i, name in enumerate(node_names)}
        # Add any nodes not in node_names
        for name in nodes:
            if name not in name_to_idx:
                name_to_idx[name] = len(node_names)
                node_names.append(name)
        nodes = node_names
    else:
        nodes = sorted(nodes)

    name_to_idx = {name: i for i, name in enumerate(nodes)}
    n = len(nodes)
    adj = np.zeros((n, n), dtype=np.int8)
    for src, tgt in edges:
        adj[name_to_idx[src], name_to_idx[tgt]] = 1

    return adj, nodes


def to_gml(
    adj: AdjacencyMatrix,
    node_names: list[str] | None = None,
) -> str:
    """Serialise a DAG to GML format.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    node_names : list[str] | None
        Optional node names.

    Returns
    -------
    str
        GML string.
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]
    names = node_names or [str(i) for i in range(n)]

    lines = ["graph [", "  directed 1"]
    for i in range(n):
        lines.append("  node [")
        lines.append(f"    id {i}")
        lines.append(f'    label "{names[i]}"')
        lines.append("  ]")
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                lines.append("  edge [")
                lines.append(f"    source {i}")
                lines.append(f"    target {j}")
                lines.append("  ]")
    lines.append("]")
    return "\n".join(lines)


def from_gml(gml_str: str) -> tuple[AdjacencyMatrix, list[str]]:
    """Parse a DAG from a GML string.

    Parameters
    ----------
    gml_str : str
        GML-format string.

    Returns
    -------
    tuple[AdjacencyMatrix, list[str]]
        ``(adjacency_matrix, node_names)``.
    """
    # Parse node ids, labels, and edges
    node_ids: list[int] = []
    node_labels: dict[int, str] = {}
    edges: list[tuple[int, int]] = []

    # Simple state-machine parser
    in_node = False
    in_edge = False
    current_id: int | None = None
    current_label: str | None = None
    current_source: int | None = None
    current_target: int | None = None

    for line in gml_str.splitlines():
        line = line.strip()
        if line == "node [":
            in_node = True
            current_id = None
            current_label = None
        elif line == "edge [":
            in_edge = True
            current_source = None
            current_target = None
        elif line == "]":
            if in_node:
                if current_id is not None:
                    node_ids.append(current_id)
                    if current_label is not None:
                        node_labels[current_id] = current_label
                    else:
                        node_labels[current_id] = str(current_id)
                in_node = False
            elif in_edge:
                if current_source is not None and current_target is not None:
                    edges.append((current_source, current_target))
                in_edge = False
        elif in_node:
            if line.startswith("id "):
                current_id = int(line.split()[-1])
            elif line.startswith("label "):
                current_label = line.split('"')[1] if '"' in line else line.split()[-1]
        elif in_edge:
            if line.startswith("source "):
                current_source = int(line.split()[-1])
            elif line.startswith("target "):
                current_target = int(line.split()[-1])

    # Build adjacency matrix
    node_ids_sorted = sorted(node_ids)
    id_to_idx = {nid: i for i, nid in enumerate(node_ids_sorted)}
    n = len(node_ids_sorted)
    names = [node_labels.get(nid, str(nid)) for nid in node_ids_sorted]
    adj = np.zeros((n, n), dtype=np.int8)
    for src, tgt in edges:
        if src in id_to_idx and tgt in id_to_idx:
            adj[id_to_idx[src], id_to_idx[tgt]] = 1

    return adj, names


def from_bif(bif_str: str) -> tuple[AdjacencyMatrix, list[str]]:
    """Parse a DAG structure from a BIF (Bayesian Interchange Format) string.

    Only extracts the DAG structure (parent-child relationships), not
    probability tables.

    Parameters
    ----------
    bif_str : str
        BIF-format string.

    Returns
    -------
    tuple[AdjacencyMatrix, list[str]]
        ``(adjacency_matrix, node_names)``.
    """
    nodes: list[str] = []
    node_set: set[str] = set()
    edges: list[tuple[str, str]] = []

    # Parse variable declarations
    var_pattern = re.compile(r'variable\s+(\w+)\s*\{', re.IGNORECASE)
    for match in var_pattern.finditer(bif_str):
        name = match.group(1)
        if name not in node_set:
            node_set.add(name)
            nodes.append(name)

    # Parse probability blocks to extract parent-child relationships
    # probability ( child | parent1, parent2 ) {
    prob_pattern = re.compile(
        r'probability\s*\(\s*(\w+)\s*(?:\|\s*([\w\s,]+))?\s*\)',
        re.IGNORECASE,
    )
    for match in prob_pattern.finditer(bif_str):
        child = match.group(1)
        if child not in node_set:
            node_set.add(child)
            nodes.append(child)
        if match.group(2):
            parents_str = match.group(2)
            parents = [p.strip() for p in parents_str.split(",")]
            for parent in parents:
                if parent and parent not in node_set:
                    node_set.add(parent)
                    nodes.append(parent)
                if parent:
                    edges.append((parent, child))

    # Build adjacency matrix
    name_to_idx = {name: i for i, name in enumerate(nodes)}
    n = len(nodes)
    adj = np.zeros((n, n), dtype=np.int8)
    for src, tgt in edges:
        adj[name_to_idx[src], name_to_idx[tgt]] = 1

    return adj, nodes


def to_networkx(adj: AdjacencyMatrix, node_names: list[str] | None = None) -> Any:
    """Convert a DAG adjacency matrix to a NetworkX DiGraph.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    node_names : list[str] | None
        Optional node names.

    Returns
    -------
    networkx.DiGraph
        NetworkX directed graph.
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx is required for this conversion")

    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]
    names = node_names or [str(i) for i in range(n)]

    G = nx.DiGraph()
    for i in range(n):
        G.add_node(names[i])
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                G.add_edge(names[i], names[j])
    return G


def from_networkx(G: Any) -> tuple[AdjacencyMatrix, list[str]]:
    """Convert a NetworkX DiGraph to an adjacency matrix.

    Parameters
    ----------
    G : networkx.DiGraph
        NetworkX directed graph.

    Returns
    -------
    tuple[AdjacencyMatrix, list[str]]
        ``(adjacency_matrix, node_names)``.
    """
    nodes = list(G.nodes())
    name_to_idx = {name: i for i, name in enumerate(nodes)}
    n = len(nodes)
    adj = np.zeros((n, n), dtype=np.int8)
    for src, tgt in G.edges():
        adj[name_to_idx[src], name_to_idx[tgt]] = 1
    names = [str(n) for n in nodes]
    return adj, names


def to_adjacency_matrix(
    adj: AdjacencyMatrix,
) -> np.ndarray:
    """Return a copy of the adjacency matrix as a standard numpy array.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.

    Returns
    -------
    np.ndarray
        Copy of the matrix.
    """
    return np.asarray(adj, dtype=np.int8).copy()


def from_adjacency_matrix(
    matrix: np.ndarray,
    node_names: list[str] | None = None,
) -> tuple[AdjacencyMatrix, list[str]]:
    """Convert a numpy array to a typed adjacency matrix with names.

    Parameters
    ----------
    matrix : np.ndarray
        Square binary matrix.
    node_names : list[str] | None
        Optional node names.

    Returns
    -------
    tuple[AdjacencyMatrix, list[str]]
    """
    adj = np.asarray(matrix, dtype=np.int8)
    n = adj.shape[0]
    names = node_names or [str(i) for i in range(n)]
    return adj, names

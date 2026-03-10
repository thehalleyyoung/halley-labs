"""
DAG file I/O — load and save DAGs from DOT, JSON, and adjacency-matrix formats.

Dispatches to the appropriate serialiser in ``causalcert.dag.conversions``
based on file extension.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from causalcert.types import AdjacencyMatrix
from causalcert.exceptions import DataError


# ============================================================================
# Load dispatch
# ============================================================================


def load_dag(path: str | Path) -> tuple[AdjacencyMatrix, list[str]]:
    """Load a DAG from a file, dispatching by extension.

    Supported extensions:
    * ``.dot`` / ``.gv`` — Graphviz DOT format
    * ``.json`` — CausalCert JSON format
    * ``.csv`` — Raw adjacency matrix or edge list
    * ``.dagitty`` — DAGitty text format
    * ``.bif`` — bnlearn BIF format (subset)

    Parameters
    ----------
    path : str | Path
        File path.

    Returns
    -------
    tuple[AdjacencyMatrix, list[str]]
        ``(adjacency_matrix, node_names)``.
    """
    path = Path(path)
    if not path.exists():
        raise DataError(f"DAG file not found: {path}")

    ext = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if ext in (".dot", ".gv"):
        return parse_dot(text)
    elif ext == ".json":
        adj, names, _ = parse_json(text)
        return adj, names
    elif ext == ".csv":
        return _load_csv_auto(path)
    elif ext == ".dagitty":
        return parse_dagitty(text)
    elif ext == ".bif":
        return parse_bnlearn_bif(text)
    else:
        raise DataError(f"Unsupported DAG file extension: {ext}")


def save_dag(
    adj: AdjacencyMatrix,
    path: str | Path,
    node_names: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a DAG to a file, dispatching by extension.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    path : str | Path
        Output file path.
    node_names : list[str] | None
        Node names for DOT/JSON formats.
    metadata : dict[str, Any] | None
        Optional metadata for JSON format.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in (".dot", ".gv"):
        path.write_text(format_dot(adj, node_names), encoding="utf-8")
    elif ext == ".json":
        path.write_text(format_json(adj, node_names, metadata), encoding="utf-8")
    elif ext == ".csv":
        write_adjacency_csv(adj, path, node_names)
    elif ext == ".dagitty":
        path.write_text(format_dagitty(adj, node_names), encoding="utf-8")
    else:
        raise DataError(f"Unsupported DAG output format: {ext}")


# ============================================================================
# DOT format
# ============================================================================


def parse_dot(dot_str: str) -> tuple[AdjacencyMatrix, list[str]]:
    """Parse a DAG from a Graphviz DOT string.

    Handles ``digraph { ... }`` with ``a -> b`` edge declarations.

    Parameters
    ----------
    dot_str : str
        DOT-language string.

    Returns
    -------
    tuple[AdjacencyMatrix, list[str]]
        ``(adjacency_matrix, node_names)``.
    """
    # Extract edges: "a" -> "b", a -> b, etc.
    edge_pattern = re.compile(
        r'("(?:[^"\\]|\\.)*"|[\w]+)\s*->\s*("(?:[^"\\]|\\.)*"|[\w]+)'
    )
    # Extract standalone nodes
    node_pattern = re.compile(
        r'^\s*("(?:[^"\\]|\\.)*"|[\w]+)\s*[\[;]',
        re.MULTILINE,
    )

    def _strip_quotes(s: str) -> str:
        if s.startswith('"') and s.endswith('"'):
            return s[1:-1].replace('\\"', '"')
        return s

    nodes: dict[str, int] = {}
    edges: list[tuple[str, str]] = []

    for match in edge_pattern.finditer(dot_str):
        src = _strip_quotes(match.group(1))
        tgt = _strip_quotes(match.group(2))
        edges.append((src, tgt))
        if src not in nodes:
            nodes[src] = len(nodes)
        if tgt not in nodes:
            nodes[tgt] = len(nodes)

    # Also pick up standalone node declarations
    for match in node_pattern.finditer(dot_str):
        name = _strip_quotes(match.group(1))
        if name not in nodes and name not in ("digraph", "graph", "subgraph", "node", "edge"):
            nodes[name] = len(nodes)

    n = len(nodes)
    if n == 0:
        return np.zeros((0, 0), dtype=np.int8), []

    node_names = [""] * n
    for name, idx in nodes.items():
        node_names[idx] = name

    adj = np.zeros((n, n), dtype=np.int8)
    for src, tgt in edges:
        adj[nodes[src], nodes[tgt]] = 1

    return adj, node_names


def format_dot(
    adj: AdjacencyMatrix,
    node_names: list[str] | None = None,
) -> str:
    """Serialise a DAG to Graphviz DOT format.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    node_names : list[str] | None
        Optional node names.

    Returns
    -------
    str
        DOT-language string.
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]
    names = node_names or [str(i) for i in range(n)]

    def _quote(s: str) -> str:
        if re.match(r'^[\w]+$', s) and s not in ("node", "edge", "graph", "digraph", "subgraph"):
            return s
        return f'"{s}"'

    lines = ["digraph G {"]
    for i in range(n):
        lines.append(f"  {_quote(names[i])};")
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                lines.append(f"  {_quote(names[i])} -> {_quote(names[j])};")
    lines.append("}")
    return "\n".join(lines) + "\n"


# ============================================================================
# JSON format
# ============================================================================


def parse_json(json_str: str) -> tuple[AdjacencyMatrix, list[str], dict[str, Any]]:
    """Parse a DAG from a JSON string.

    Expected format::

        {
            "nodes": ["A", "B", "C"],
            "edges": [["A", "B"], ["B", "C"]],
            "metadata": {}
        }

    Or with indices::

        {
            "nodes": ["A", "B", "C"],
            "adjacency_matrix": [[0,1,0],[0,0,1],[0,0,0]]
        }

    Parameters
    ----------
    json_str : str
        JSON string.

    Returns
    -------
    tuple[AdjacencyMatrix, list[str], dict[str, Any]]
        ``(adjacency_matrix, node_names, metadata)``.
    """
    data = json.loads(json_str)

    if "adjacency_matrix" in data:
        adj = np.asarray(data["adjacency_matrix"], dtype=np.int8)
        names = data.get("nodes", [str(i) for i in range(adj.shape[0])])
        meta = data.get("metadata", {})
        return adj, names, meta

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    meta = data.get("metadata", {})

    name_to_idx: dict[str, int] = {name: i for i, name in enumerate(nodes)}
    n = len(nodes)
    adj = np.zeros((n, n), dtype=np.int8)

    for edge in edges:
        src, tgt = edge[0], edge[1]
        if src in name_to_idx and tgt in name_to_idx:
            adj[name_to_idx[src], name_to_idx[tgt]] = 1

    return adj, nodes, meta


def format_json(
    adj: AdjacencyMatrix,
    node_names: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Serialise a DAG to JSON.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    node_names : list[str] | None
        Node names.
    metadata : dict[str, Any] | None
        Optional metadata.

    Returns
    -------
    str
        JSON string.
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]
    names = node_names or [str(i) for i in range(n)]

    edges = []
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                edges.append([names[i], names[j]])

    obj = {
        "nodes": names,
        "edges": edges,
        "adjacency_matrix": adj.tolist(),
    }
    if metadata:
        obj["metadata"] = metadata

    return json.dumps(obj, indent=2)


# ============================================================================
# Adjacency matrix CSV
# ============================================================================


def write_adjacency_csv(
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
        Column/row headers.
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]
    names = node_names or [str(i) for i in range(n)]
    path = Path(path)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + names)
        for i in range(n):
            writer.writerow([names[i]] + adj[i].tolist())


def read_adjacency_csv(path: str | Path) -> tuple[AdjacencyMatrix, list[str]]:
    """Load an adjacency matrix from a CSV file.

    Handles both header-labelled and plain numeric CSV formats.

    Parameters
    ----------
    path : str | Path
        Input file path.

    Returns
    -------
    tuple[AdjacencyMatrix, list[str]]
    """
    path = Path(path)
    if not path.exists():
        raise DataError(f"File not found: {path}")

    # Try reading with header
    import pandas as pd
    try:
        df = pd.read_csv(path, index_col=0)
        adj = df.values.astype(np.int8)
        names = df.columns.tolist()
        return adj, names
    except Exception:
        pass

    # Plain numeric CSV (no headers)
    try:
        adj = np.loadtxt(path, delimiter=",", dtype=np.int8)
        n = adj.shape[0]
        return adj, [str(i) for i in range(n)]
    except Exception as exc:
        raise DataError(f"Cannot parse adjacency CSV: {exc}") from exc


def _load_csv_auto(
    path: Path,
) -> tuple[AdjacencyMatrix, list[str]]:
    """Auto-detect CSV format: adjacency matrix vs edge list."""
    import pandas as pd
    df = pd.read_csv(path)

    # If exactly 2 columns (src, tgt), treat as edge list
    if df.shape[1] == 2:
        return parse_edge_list_csv(path)

    # Otherwise treat as adjacency matrix
    return read_adjacency_csv(path)


# ============================================================================
# Edge list CSV
# ============================================================================


def parse_edge_list_csv(
    path: str | Path,
) -> tuple[AdjacencyMatrix, list[str]]:
    """Parse an edge list CSV into a DAG.

    Expected format: two columns (source, target), optionally with a header.

    Parameters
    ----------
    path : str | Path
        File path.

    Returns
    -------
    tuple[AdjacencyMatrix, list[str]]
    """
    import pandas as pd
    path = Path(path)
    df = pd.read_csv(path)
    cols = df.columns.tolist()

    # Collect unique nodes
    nodes: dict[str, int] = {}
    edges: list[tuple[str, str]] = []

    for _, row in df.iterrows():
        src = str(row.iloc[0])
        tgt = str(row.iloc[1])
        if src not in nodes:
            nodes[src] = len(nodes)
        if tgt not in nodes:
            nodes[tgt] = len(nodes)
        edges.append((src, tgt))

    n = len(nodes)
    node_names = [""] * n
    for name, idx in nodes.items():
        node_names[idx] = name

    adj = np.zeros((n, n), dtype=np.int8)
    for src, tgt in edges:
        adj[nodes[src], nodes[tgt]] = 1

    return adj, node_names


def write_edge_list_csv(
    adj: AdjacencyMatrix,
    path: str | Path,
    node_names: list[str] | None = None,
) -> None:
    """Write the DAG as an edge list CSV.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    path : str | Path
        Output path.
    node_names : list[str] | None
        Node names.
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]
    names = node_names or [str(i) for i in range(n)]
    path = Path(path)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target"])
        for i in range(n):
            for j in range(n):
                if adj[i, j]:
                    writer.writerow([names[i], names[j]])


# ============================================================================
# DAGitty format
# ============================================================================


def parse_dagitty(text: str) -> tuple[AdjacencyMatrix, list[str]]:
    """Parse a DAGitty-format string into a DAG.

    DAGitty format example::

        dag {
          X -> Y
          Z -> X
          Z -> Y
        }

    Parameters
    ----------
    text : str
        DAGitty text.

    Returns
    -------
    tuple[AdjacencyMatrix, list[str]]
    """
    nodes: dict[str, int] = {}
    edges: list[tuple[str, str]] = []

    # Remove comments
    text = re.sub(r'//[^\n]*', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

    # Find edges: X -> Y or X <- Y
    for match in re.finditer(r'(\w+)\s*(-[->])\s*(\w+)', text):
        left = match.group(1)
        arrow = match.group(2)
        right = match.group(3)

        if left not in nodes:
            nodes[left] = len(nodes)
        if right not in nodes:
            nodes[right] = len(nodes)

        if arrow == "->":
            edges.append((left, right))
        elif arrow == "<-":
            edges.append((right, left))

    # Standalone node declarations
    for match in re.finditer(r'^\s*(\w+)\s*$', text, re.MULTILINE):
        name = match.group(1)
        if name not in ("dag", "graph") and name not in nodes:
            nodes[name] = len(nodes)

    n = len(nodes)
    node_names = [""] * n
    for name, idx in nodes.items():
        node_names[idx] = name

    adj = np.zeros((n, n), dtype=np.int8)
    for src, tgt in edges:
        adj[nodes[src], nodes[tgt]] = 1

    return adj, node_names


def format_dagitty(
    adj: AdjacencyMatrix,
    node_names: list[str] | None = None,
) -> str:
    """Serialise a DAG to DAGitty format.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    node_names : list[str] | None
        Node names.

    Returns
    -------
    str
        DAGitty-format string.
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]
    names = node_names or [str(i) for i in range(n)]

    lines = ["dag {"]
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                lines.append(f"  {names[i]} -> {names[j]}")
    lines.append("}")
    return "\n".join(lines) + "\n"


# ============================================================================
# bnlearn BIF format (subset)
# ============================================================================


def parse_bnlearn_bif(text: str) -> tuple[AdjacencyMatrix, list[str]]:
    """Parse a (simplified) bnlearn BIF format.

    Reads ``variable`` and ``probability`` blocks to extract the DAG
    structure.

    Parameters
    ----------
    text : str
        BIF text.

    Returns
    -------
    tuple[AdjacencyMatrix, list[str]]
    """
    nodes: dict[str, int] = {}
    edges: list[tuple[str, str]] = []

    # Extract variable declarations
    for match in re.finditer(r'variable\s+(\w+)', text):
        name = match.group(1)
        if name not in nodes:
            nodes[name] = len(nodes)

    # Extract probability blocks: probability ( child | parent1, parent2, ... )
    prob_pattern = re.compile(
        r'probability\s*\(\s*(\w+)\s*(?:\|\s*([^)]+))?\s*\)'
    )
    for match in prob_pattern.finditer(text):
        child = match.group(1)
        parents_str = match.group(2)
        if child not in nodes:
            nodes[child] = len(nodes)
        if parents_str:
            parents = [p.strip() for p in parents_str.split(",")]
            for p in parents:
                if p not in nodes:
                    nodes[p] = len(nodes)
                edges.append((p, child))

    n = len(nodes)
    node_names = [""] * n
    for name, idx in nodes.items():
        node_names[idx] = name

    adj = np.zeros((n, n), dtype=np.int8)
    for src, tgt in edges:
        adj[nodes[src], nodes[tgt]] = 1

    return adj, node_names


def format_bnlearn_modelstring(
    adj: AdjacencyMatrix,
    node_names: list[str] | None = None,
) -> str:
    """Serialise a DAG to bnlearn model-string format.

    Format: ``[A][B|A][C|A:B]`` — each node lists its parents after ``|``.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    node_names : list[str] | None
        Node names.

    Returns
    -------
    str
        bnlearn model string.
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]
    names = node_names or [str(i) for i in range(n)]

    parts = []
    for j in range(n):
        parents = [names[i] for i in range(n) if adj[i, j]]
        if parents:
            parts.append(f"[{names[j]}|{':'.join(parents)}]")
        else:
            parts.append(f"[{names[j]}]")
    return "".join(parts)

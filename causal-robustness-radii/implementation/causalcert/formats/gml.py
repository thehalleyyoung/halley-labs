"""
GML (Graph Modeling Language) format parser and writer.

Supports the hierarchical GML key-value format used by many graph analysis
tools.  Handles nested attribute lists, node/edge attributes, and produces
properly indented output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np

from causalcert.types import AdjacencyMatrix, FormatType
from causalcert.formats.types import FormatSpec, ParseResult, ValidationIssue, ValidationReport


# ---------------------------------------------------------------------------
# Format spec
# ---------------------------------------------------------------------------

GML_FORMAT_SPEC = FormatSpec(
    format_type=FormatType.GML,
    name="Graph Modeling Language",
    extensions=(".gml",),
    mime_type="text/x-gml",
    supports_metadata=True,
    supports_latent=False,
)


# ---------------------------------------------------------------------------
# GML tokenizer
# ---------------------------------------------------------------------------

class _GMLTokenKind:
    KEY = "key"
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    LBRACKET = "["
    RBRACKET = "]"
    EOF = "eof"


@dataclass(slots=True)
class _GMLToken:
    kind: str
    value: Any
    line: int


_GML_COMMENT = re.compile(r"#[^\n]*")
_GML_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_GML_FLOAT = re.compile(r"-?[0-9]+\.[0-9]*(?:[eE][+-]?[0-9]+)?")
_GML_INT = re.compile(r"-?[0-9]+")
_GML_STRING = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"')


def _gml_tokenize(text: str) -> list[_GMLToken]:
    """Tokenize a GML string.

    Parameters
    ----------
    text : str
        GML source with comments stripped.

    Returns
    -------
    list[_GMLToken]
    """
    text = _GML_COMMENT.sub("", text)
    tokens: list[_GMLToken] = []
    i = 0
    line = 1
    n = len(text)

    while i < n:
        ch = text[i]
        if ch in " \t\r":
            i += 1
            continue
        if ch == "\n":
            line += 1
            i += 1
            continue
        if ch == "[":
            tokens.append(_GMLToken(_GMLTokenKind.LBRACKET, "[", line))
            i += 1
            continue
        if ch == "]":
            tokens.append(_GMLToken(_GMLTokenKind.RBRACKET, "]", line))
            i += 1
            continue

        # string
        m = _GML_STRING.match(text, i)
        if m:
            val = m.group(1).replace("\\n", "\n").replace("\\\\", "\\")
            tokens.append(_GMLToken(_GMLTokenKind.STRING, val, line))
            i = m.end()
            continue

        # float (check before int)
        m = _GML_FLOAT.match(text, i)
        if m:
            tokens.append(_GMLToken(_GMLTokenKind.FLOAT, float(m.group(0)), line))
            i = m.end()
            continue

        # int
        m = _GML_INT.match(text, i)
        if m:
            tokens.append(_GMLToken(_GMLTokenKind.INT, int(m.group(0)), line))
            i = m.end()
            continue

        # key
        m = _GML_KEY.match(text, i)
        if m:
            tokens.append(_GMLToken(_GMLTokenKind.KEY, m.group(0), line))
            i = m.end()
            continue

        # skip unknown
        i += 1

    tokens.append(_GMLToken(_GMLTokenKind.EOF, None, line))
    return tokens


# ---------------------------------------------------------------------------
# GML recursive-descent parser
# ---------------------------------------------------------------------------

GMLValue = int | float | str | list[tuple[str, Any]]


def _gml_parse_list(
    tokens: list[_GMLToken], pos: int,
) -> tuple[list[tuple[str, GMLValue]], int]:
    """Parse a GML key-value list (``key value ...``) until ``]`` or EOF.

    Returns
    -------
    tuple[list[tuple[str, GMLValue]], int]
        (key_value_pairs, new_position)
    """
    result: list[tuple[str, GMLValue]] = []
    while pos < len(tokens):
        tok = tokens[pos]
        if tok.kind in (_GMLTokenKind.RBRACKET, _GMLTokenKind.EOF):
            break
        if tok.kind != _GMLTokenKind.KEY:
            pos += 1
            continue

        key = tok.value
        pos += 1
        if pos >= len(tokens):
            break

        val_tok = tokens[pos]
        if val_tok.kind == _GMLTokenKind.LBRACKET:
            pos += 1  # consume [
            nested, pos = _gml_parse_list(tokens, pos)
            if pos < len(tokens) and tokens[pos].kind == _GMLTokenKind.RBRACKET:
                pos += 1  # consume ]
            result.append((key, nested))
        elif val_tok.kind in (_GMLTokenKind.INT, _GMLTokenKind.FLOAT,
                              _GMLTokenKind.STRING):
            result.append((key, val_tok.value))
            pos += 1
        else:
            pos += 1

    return result, pos


def _gml_parse(text: str) -> list[tuple[str, GMLValue]]:
    """Parse a full GML document into a nested list structure."""
    tokens = _gml_tokenize(text)
    result, _ = _gml_parse_list(tokens, 0)
    return result


# ---------------------------------------------------------------------------
# GML helpers
# ---------------------------------------------------------------------------

def _get_gml_attr(
    attrs: list[tuple[str, GMLValue]], key: str,
) -> GMLValue | None:
    """Look up a key in a GML attribute list."""
    for k, v in attrs:
        if k == key:
            return v
    return None


def _get_gml_str(attrs: list[tuple[str, GMLValue]], key: str) -> str:
    """Look up a string attribute, defaulting to ``""``."""
    val = _get_gml_attr(attrs, key)
    return str(val) if val is not None else ""


def _get_gml_int(attrs: list[tuple[str, GMLValue]], key: str) -> int | None:
    """Look up an integer attribute."""
    val = _get_gml_attr(attrs, key)
    if isinstance(val, int):
        return val
    if isinstance(val, (float, str)):
        try:
            return int(val)
        except (ValueError, TypeError):
            return None
    return None


# ---------------------------------------------------------------------------
# GMLParser
# ---------------------------------------------------------------------------

class GMLParser:
    """Parse GML (Graph Modeling Language) files.

    Handles the hierarchical key-value format with nested attribute
    lists for nodes and edges.

    Examples
    --------
    >>> parser = GMLParser()
    >>> result = parser.parse_string('''
    ... graph [
    ...   directed 1
    ...   node [ id 0 label "A" ]
    ...   node [ id 1 label "B" ]
    ...   edge [ source 0 target 1 ]
    ... ]''')
    >>> result.n_edges
    1
    """

    @property
    def format_spec(self) -> FormatSpec:
        """Return the GML format specification."""
        return GML_FORMAT_SPEC

    def parse_string(self, text: str, *, validate: bool = True) -> ParseResult:
        """Parse a GML string.

        Parameters
        ----------
        text : str
            GML source.
        validate : bool, optional
            Run post-parse validation.

        Returns
        -------
        ParseResult
        """
        top = _gml_parse(text)
        graph_attrs = _get_gml_attr(top, "graph")
        if graph_attrs is None or not isinstance(graph_attrs, list):
            # Maybe the top level *is* the graph
            graph_attrs = top

        directed = bool(_get_gml_int(graph_attrs, "directed"))

        # Extract nodes
        id_to_label: dict[int, str] = {}
        node_attrs_map: dict[str, dict[str, Any]] = {}
        for key, val in graph_attrs:
            if key == "node" and isinstance(val, list):
                nid = _get_gml_int(val, "id")
                label = _get_gml_str(val, "label")
                if nid is None:
                    continue
                if not label:
                    label = str(nid)
                id_to_label[nid] = label
                # Collect extra attributes
                extra: dict[str, Any] = {}
                for k, v in val:
                    if k not in ("id", "label"):
                        extra[k] = v
                if extra:
                    node_attrs_map[label] = extra

        # Build index: id → matrix row
        sorted_ids = sorted(id_to_label.keys())
        names = [id_to_label[i] for i in sorted_ids]
        idx = {nid: pos for pos, nid in enumerate(sorted_ids)}
        n = len(names)

        # Extract edges
        adj = np.zeros((n, n), dtype=np.int8)
        edge_attrs_list: list[dict[str, Any]] = []
        for key, val in graph_attrs:
            if key == "edge" and isinstance(val, list):
                src_id = _get_gml_int(val, "source")
                dst_id = _get_gml_int(val, "target")
                if src_id is None or dst_id is None:
                    continue
                if src_id in idx and dst_id in idx:
                    adj[idx[src_id], idx[dst_id]] = 1
                    if not directed:
                        adj[idx[dst_id], idx[src_id]] = 1
                    extra = {}
                    for k, v in val:
                        if k not in ("source", "target"):
                            extra[k] = v
                    if extra:
                        edge_attrs_list.append({
                            "src": id_to_label.get(src_id, str(src_id)),
                            "dst": id_to_label.get(dst_id, str(dst_id)),
                            **extra,
                        })

        meta: dict[str, Any] = {"directed": directed}
        if node_attrs_map:
            meta["node_attrs"] = node_attrs_map
        if edge_attrs_list:
            meta["edge_attrs"] = edge_attrs_list

        # Graph-level attributes
        graph_level: dict[str, Any] = {}
        for k, v in graph_attrs:
            if k not in ("node", "edge", "directed"):
                graph_level[k] = v
        if graph_level:
            meta["graph_attrs"] = graph_level

        vr: ValidationReport | None = None
        if validate:
            issues: list[ValidationIssue] = []
            if directed:
                _check_acyclicity(adj, names, issues)
            vr = ValidationReport(
                is_valid=not any(i.level == "error" for i in issues),
                issues=tuple(issues),
                format_type=FormatType.GML,
            )

        return ParseResult(
            adjacency=adj,
            node_labels=tuple(names),
            format_type=FormatType.GML,
            metadata=meta,
            validation=vr,
        )

    def parse_file(
        self, path: str | Path, *, validate: bool = True,
    ) -> ParseResult:
        """Parse a GML file from disk.

        Parameters
        ----------
        path : str | Path
            File path.
        validate : bool, optional
            Run post-parse validation.

        Returns
        -------
        ParseResult
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"GML file not found: {p}")
        text = p.read_text(encoding="utf-8")
        return self.parse_string(text, validate=validate)

    def validate(self, text: str) -> ValidationReport:
        """Validate a GML string.

        Parameters
        ----------
        text : str
            GML source.

        Returns
        -------
        ValidationReport
        """
        issues: list[ValidationIssue] = []
        try:
            top = _gml_parse(text)
            graph_attrs = _get_gml_attr(top, "graph")
            if graph_attrs is None:
                issues.append(ValidationIssue(
                    level="warning", message="No 'graph' block found.",
                ))
        except Exception as exc:
            issues.append(ValidationIssue(level="error", message=str(exc)))

        return ValidationReport(
            is_valid=not any(i.level == "error" for i in issues),
            issues=tuple(issues),
            format_type=FormatType.GML,
        )


# ---------------------------------------------------------------------------
# GMLWriter
# ---------------------------------------------------------------------------

class GMLWriter:
    """Serialize a DAG to GML format.

    Produces properly indented hierarchical GML output with node and
    edge attribute support.

    Examples
    --------
    >>> import numpy as np
    >>> w = GMLWriter()
    >>> gml = w.to_string(np.array([[0,1],[0,0]], dtype=np.int8),
    ...                   node_labels=("A", "B"))
    >>> "source 0" in gml
    True
    """

    @property
    def format_spec(self) -> FormatSpec:
        """Return the GML format specification."""
        return GML_FORMAT_SPEC

    @staticmethod
    def _write_value(buf: StringIO, key: str, val: Any, indent: int) -> None:
        """Write a single key-value pair at the given indent level."""
        prefix = "  " * indent
        if isinstance(val, list):
            buf.write(f"{prefix}{key} [\n")
            for k, v in val:
                GMLWriter._write_value(buf, k, v, indent + 1)
            buf.write(f"{prefix}]\n")
        elif isinstance(val, int):
            buf.write(f"{prefix}{key} {val}\n")
        elif isinstance(val, float):
            buf.write(f"{prefix}{key} {val}\n")
        elif isinstance(val, str):
            escaped = val.replace("\\", "\\\\").replace('"', '\\"')
            buf.write(f'{prefix}{key} "{escaped}"\n')
        else:
            buf.write(f'{prefix}{key} "{val}"\n')

    def to_string(
        self,
        adj: AdjacencyMatrix,
        *,
        node_labels: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Serialize a DAG to a GML string.

        Parameters
        ----------
        adj : AdjacencyMatrix
            Adjacency matrix.
        node_labels : tuple[str, ...], optional
            Node names.
        metadata : dict[str, Any] | None, optional
            May contain ``directed``, ``node_attrs``, ``edge_attrs``,
            ``graph_attrs``.

        Returns
        -------
        str
            GML source.
        """
        adj = np.asarray(adj, dtype=np.int8)
        n = adj.shape[0]
        names = list(node_labels) if node_labels else [f"X{i}" for i in range(n)]
        meta = metadata or {}
        directed = meta.get("directed", True)
        node_attrs: dict[str, dict[str, Any]] = meta.get("node_attrs", {})
        edge_attrs_list: list[dict[str, Any]] = meta.get("edge_attrs", [])
        graph_attrs: dict[str, Any] = meta.get("graph_attrs", {})

        ea_lookup: dict[tuple[str, str], dict[str, Any]] = {}
        for ea in edge_attrs_list:
            key = (ea.get("src", ""), ea.get("dst", ""))
            ea_lookup[key] = {k: v for k, v in ea.items() if k not in ("src", "dst")}

        buf = StringIO()
        buf.write("graph [\n")
        buf.write(f"  directed {1 if directed else 0}\n")

        for k, v in graph_attrs.items():
            self._write_value(buf, k, v, 1)

        for i, name in enumerate(names):
            buf.write("  node [\n")
            buf.write(f"    id {i}\n")
            escaped = name.replace("\\", "\\\\").replace('"', '\\"')
            buf.write(f'    label "{escaped}"\n')
            for k, v in node_attrs.get(name, {}).items():
                self._write_value(buf, k, v, 2)
            buf.write("  ]\n")

        for i in range(n):
            for j in range(n):
                if adj[i, j]:
                    buf.write("  edge [\n")
                    buf.write(f"    source {i}\n")
                    buf.write(f"    target {j}\n")
                    extras = ea_lookup.get((names[i], names[j]), {})
                    for k, v in extras.items():
                        self._write_value(buf, k, v, 2)
                    buf.write("  ]\n")

        buf.write("]\n")
        return buf.getvalue()

    def to_file(
        self,
        adj: AdjacencyMatrix,
        path: str | Path,
        *,
        node_labels: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write a GML file to disk.

        Parameters
        ----------
        adj : AdjacencyMatrix
            Adjacency matrix.
        path : str | Path
            Destination file.
        node_labels : tuple[str, ...], optional
            Node names.
        metadata : dict[str, Any] | None, optional
            Extra metadata.
        """
        text = self.to_string(adj, node_labels=node_labels, metadata=metadata)
        Path(path).write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_acyclicity(
    adj: np.ndarray,
    names: list[str],
    issues: list[ValidationIssue],
) -> None:
    """Append a warning if the graph contains a cycle."""
    n = adj.shape[0]
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n

    def _dfs(u: int) -> bool:
        color[u] = GRAY
        for v in range(n):
            if adj[u, v]:
                if color[v] == GRAY:
                    return True
                if color[v] == WHITE and _dfs(v):
                    return True
        color[u] = BLACK
        return False

    for start in range(n):
        if color[start] == WHITE:
            if _dfs(start):
                issues.append(ValidationIssue(
                    level="warning",
                    message="Graph contains a cycle (not a DAG).",
                ))
                return

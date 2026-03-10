"""
TETRAD XML and PCALG / bnlearn format parser and writer.

Supports:

* **TETRAD XML** — the project file format used by the TETRAD causal
  discovery software (Carnegie Mellon).  Reads and writes ``<graph>``
  elements containing ``<node>`` and ``<edge>`` children.
* **PCALG adjacency list** — the column-major adjacency matrix CSV
  convention used by the R ``pcalg`` package.
* **bnlearn model strings** — compact ``[A][B|A][C|A:B]`` notation
  used by the R ``bnlearn`` package.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np

from causalcert.types import AdjacencyMatrix, FormatType, VariableType
from causalcert.formats.types import FormatSpec, ParseResult, ValidationIssue, ValidationReport


# ---------------------------------------------------------------------------
# Format specs
# ---------------------------------------------------------------------------

TETRAD_FORMAT_SPEC = FormatSpec(
    format_type=FormatType.TETRAD,
    name="TETRAD XML",
    extensions=(".xml", ".tetrad"),
    mime_type="application/xml",
    supports_metadata=True,
    supports_latent=True,
)

PCALG_FORMAT_SPEC = FormatSpec(
    format_type=FormatType.PCALG,
    name="PCALG / bnlearn",
    extensions=(".pcalg", ".bnlearn"),
    mime_type="text/plain",
    supports_metadata=False,
    supports_latent=False,
)


# ---------------------------------------------------------------------------
# TETRAD XML parser / writer
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TetradVariable:
    """A variable (node) in a TETRAD model.

    Attributes
    ----------
    name : str
        Variable name.
    var_type : VariableType
        Statistical type of the variable.
    categories : tuple[str, ...] | None
        Category labels for discrete/ordinal variables.
    latent : bool
        ``True`` if the variable is latent (unobserved).
    """

    name: str
    var_type: VariableType = VariableType.CONTINUOUS
    categories: tuple[str, ...] | None = None
    latent: bool = False


class TetradParser:
    """Parse TETRAD XML format.

    Reads the ``<graph>`` element from a TETRAD XML project file,
    extracting ``<node>`` and ``<edge>`` elements.

    Examples
    --------
    >>> parser = TetradParser()
    >>> result = parser.parse_string('''
    ... <tetradGraph>
    ...   <graph>
    ...     <node name="X"/>
    ...     <node name="Y"/>
    ...     <edge cause="X" effect="Y"/>
    ...   </graph>
    ... </tetradGraph>''')
    >>> result.n_edges
    1
    """

    @property
    def format_spec(self) -> FormatSpec:
        """Return the TETRAD format specification."""
        return TETRAD_FORMAT_SPEC

    # -- internal -----------------------------------------------------------

    @staticmethod
    def _find_graph_element(root: ET.Element) -> ET.Element | None:
        """Recursively locate the <graph> element."""
        if root.tag == "graph":
            return root
        for child in root:
            result = TetradParser._find_graph_element(child)
            if result is not None:
                return result
        return None

    def _parse_xml(self, root: ET.Element) -> tuple[
        list[TetradVariable], list[tuple[str, str]], dict[str, Any],
    ]:
        """Extract variables and edges from TETRAD XML."""
        graph_el = self._find_graph_element(root)

        variables: list[TetradVariable] = []
        edges: list[tuple[str, str]] = []
        meta: dict[str, Any] = {}

        if graph_el is None:
            # Fallback: try root-level nodes/edges
            graph_el = root

        # Parse nodes
        for node_el in graph_el.iter("node"):
            name = node_el.get("name", node_el.text or "")
            name = name.strip()
            if not name:
                continue
            var_type_str = node_el.get("type", "continuous").lower()
            type_map = {
                "continuous": VariableType.CONTINUOUS,
                "discrete": VariableType.NOMINAL,
                "ordinal": VariableType.ORDINAL,
                "binary": VariableType.BINARY,
                "nominal": VariableType.NOMINAL,
            }
            var_type = type_map.get(var_type_str, VariableType.CONTINUOUS)
            latent = node_el.get("latent", "false").lower() == "true"
            cats_str = node_el.get("categories", "")
            cats = tuple(cats_str.split(",")) if cats_str else None

            variables.append(TetradVariable(
                name=name, var_type=var_type,
                categories=cats, latent=latent,
            ))

        # Parse edges
        for edge_el in graph_el.iter("edge"):
            cause = edge_el.get("cause", edge_el.get("from", "")).strip()
            effect = edge_el.get("effect", edge_el.get("to", "")).strip()
            if cause and effect:
                edges.append((cause, effect))

        # Also look for <edges> wrapper and textual format "X --> Y"
        for edges_el in graph_el.iter("edges"):
            text = edges_el.text or ""
            for line in text.strip().split("\n"):
                m = re.match(r"\s*(\S+)\s*-->\s*(\S+)", line.strip())
                if m:
                    edges.append((m.group(1), m.group(2)))

        # Collect variable types in metadata
        var_types: dict[str, str] = {}
        latent_vars: list[str] = []
        for v in variables:
            var_types[v.name] = v.var_type.value
            if v.latent:
                latent_vars.append(v.name)
        if var_types:
            meta["variable_types"] = var_types
        if latent_vars:
            meta["latent"] = latent_vars

        return variables, edges, meta

    # -- public interface ---------------------------------------------------

    def parse_string(self, text: str, *, validate: bool = True) -> ParseResult:
        """Parse a TETRAD XML string.

        Parameters
        ----------
        text : str
            TETRAD XML source.
        validate : bool, optional
            Run post-parse validation.

        Returns
        -------
        ParseResult
        """
        root = ET.fromstring(text.strip())
        variables, edges, meta = self._parse_xml(root)

        names = [v.name for v in variables]
        idx = {name: i for i, name in enumerate(names)}

        # Add nodes referenced in edges but not declared
        for src, dst in edges:
            if src not in idx:
                idx[src] = len(names)
                names.append(src)
            if dst not in idx:
                idx[dst] = len(names)
                names.append(dst)

        n = len(names)
        adj = np.zeros((n, n), dtype=np.int8)
        for src, dst in edges:
            adj[idx[src], idx[dst]] = 1

        vr: ValidationReport | None = None
        if validate:
            issues: list[ValidationIssue] = []
            _check_acyclicity(adj, names, issues)
            vr = ValidationReport(
                is_valid=not any(i.level == "error" for i in issues),
                issues=tuple(issues),
                format_type=FormatType.TETRAD,
            )

        return ParseResult(
            adjacency=adj,
            node_labels=tuple(names),
            format_type=FormatType.TETRAD,
            metadata=meta,
            validation=vr,
        )

    def parse_file(
        self, path: str | Path, *, validate: bool = True,
    ) -> ParseResult:
        """Parse a TETRAD XML file from disk.

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
            raise FileNotFoundError(f"TETRAD file not found: {p}")
        text = p.read_text(encoding="utf-8")
        return self.parse_string(text, validate=validate)

    def validate(self, text: str) -> ValidationReport:
        """Validate a TETRAD XML string.

        Parameters
        ----------
        text : str
            TETRAD XML source.

        Returns
        -------
        ValidationReport
        """
        issues: list[ValidationIssue] = []
        try:
            ET.fromstring(text.strip())
        except ET.ParseError as exc:
            issues.append(ValidationIssue(level="error", message=f"XML error: {exc}"))
        return ValidationReport(
            is_valid=not any(i.level == "error" for i in issues),
            issues=tuple(issues),
            format_type=FormatType.TETRAD,
        )


class TetradWriter:
    """Serialize a DAG to TETRAD XML format.

    Examples
    --------
    >>> import numpy as np
    >>> w = TetradWriter()
    >>> xml = w.to_string(np.array([[0,1],[0,0]], dtype=np.int8),
    ...                   node_labels=("X", "Y"))
    >>> "<edge" in xml
    True
    """

    @property
    def format_spec(self) -> FormatSpec:
        """Return the TETRAD format specification."""
        return TETRAD_FORMAT_SPEC

    def to_string(
        self,
        adj: AdjacencyMatrix,
        *,
        node_labels: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Serialize a DAG to TETRAD XML.

        Parameters
        ----------
        adj : AdjacencyMatrix
            Adjacency matrix.
        node_labels : tuple[str, ...], optional
            Node names.
        metadata : dict[str, Any] | None, optional
            May contain ``variable_types``, ``latent``.

        Returns
        -------
        str
            TETRAD XML string.
        """
        adj = np.asarray(adj, dtype=np.int8)
        n = adj.shape[0]
        names = list(node_labels) if node_labels else [f"X{i}" for i in range(n)]
        meta = metadata or {}
        var_types: dict[str, str] = meta.get("variable_types", {})
        latent_vars: list[str] = meta.get("latent", [])

        root = ET.Element("tetradGraph")
        graph = ET.SubElement(root, "graph")

        for name in names:
            node_el = ET.SubElement(graph, "node")
            node_el.set("name", name)
            vt = var_types.get(name, "continuous")
            node_el.set("type", vt)
            if name in latent_vars:
                node_el.set("latent", "true")

        for i in range(n):
            for j in range(n):
                if adj[i, j]:
                    edge_el = ET.SubElement(graph, "edge")
                    edge_el.set("cause", names[i])
                    edge_el.set("effect", names[j])

        return _indent_xml(ET.tostring(root, encoding="unicode"))

    def to_file(
        self,
        adj: AdjacencyMatrix,
        path: str | Path,
        *,
        node_labels: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write a TETRAD XML file to disk.

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
# PCALG adjacency-list parser / writer
# ---------------------------------------------------------------------------

class PCALGParser:
    """Parse PCALG adjacency-matrix CSV format.

    The R ``pcalg`` package stores adjacency matrices as CSV files where
    rows and columns are labelled with variable names and cells are 0/1.

    Examples
    --------
    >>> parser = PCALGParser()
    >>> result = parser.parse_string(",A,B\\nA,0,1\\nB,0,0")
    >>> result.n_edges
    1
    """

    @property
    def format_spec(self) -> FormatSpec:
        """Return the PCALG format specification."""
        return PCALG_FORMAT_SPEC

    def parse_string(self, text: str, *, validate: bool = True) -> ParseResult:
        """Parse PCALG adjacency CSV from a string.

        Parameters
        ----------
        text : str
            CSV text with header row and row labels.
        validate : bool, optional
            Run post-parse validation.

        Returns
        -------
        ParseResult
        """
        lines = [l for l in text.strip().split("\n") if l.strip()]
        if not lines:
            return ParseResult(
                adjacency=np.zeros((0, 0), dtype=np.int8),
                node_labels=(),
                format_type=FormatType.PCALG,
            )

        # Parse header
        header = _split_csv_line(lines[0])
        # First cell may be empty (row-label column)
        if header and header[0].strip() in ("", '""'):
            names = [h.strip().strip('"') for h in header[1:]]
        else:
            names = [h.strip().strip('"') for h in header]

        n = len(names)
        adj = np.zeros((n, n), dtype=np.int8)

        for row_idx, line in enumerate(lines[1:]):
            cells = _split_csv_line(line)
            # Skip row label
            values = cells[1:] if len(cells) > n else cells
            for col_idx, val in enumerate(values):
                if col_idx < n:
                    try:
                        adj[row_idx, col_idx] = int(float(val.strip()))
                    except (ValueError, IndexError):
                        pass

        vr: ValidationReport | None = None
        if validate:
            issues: list[ValidationIssue] = []
            _check_acyclicity(adj, names, issues)
            vr = ValidationReport(
                is_valid=not any(i.level == "error" for i in issues),
                issues=tuple(issues),
                format_type=FormatType.PCALG,
            )

        return ParseResult(
            adjacency=adj,
            node_labels=tuple(names),
            format_type=FormatType.PCALG,
            metadata={},
            validation=vr,
        )

    def parse_file(
        self, path: str | Path, *, validate: bool = True,
    ) -> ParseResult:
        """Parse a PCALG CSV file from disk."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"PCALG file not found: {p}")
        text = p.read_text(encoding="utf-8")
        return self.parse_string(text, validate=validate)

    def validate(self, text: str) -> ValidationReport:
        """Validate PCALG CSV text."""
        issues: list[ValidationIssue] = []
        lines = [l for l in text.strip().split("\n") if l.strip()]
        if not lines:
            issues.append(ValidationIssue(
                level="error", message="Empty PCALG input.",
            ))
        return ValidationReport(
            is_valid=not any(i.level == "error" for i in issues),
            issues=tuple(issues),
            format_type=FormatType.PCALG,
        )


class PCALGWriter:
    """Serialize a DAG to PCALG adjacency-matrix CSV format.

    Examples
    --------
    >>> import numpy as np
    >>> w = PCALGWriter()
    >>> csv = w.to_string(np.array([[0,1],[0,0]], dtype=np.int8),
    ...                   node_labels=("A", "B"))
    >>> "A" in csv
    True
    """

    @property
    def format_spec(self) -> FormatSpec:
        """Return the PCALG format specification."""
        return PCALG_FORMAT_SPEC

    def to_string(
        self,
        adj: AdjacencyMatrix,
        *,
        node_labels: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Serialize a DAG to PCALG CSV.

        Parameters
        ----------
        adj : AdjacencyMatrix
            Adjacency matrix.
        node_labels : tuple[str, ...], optional
            Node names.
        metadata : dict[str, Any] | None, optional
            Ignored for this format.

        Returns
        -------
        str
            CSV text.
        """
        adj = np.asarray(adj, dtype=np.int8)
        n = adj.shape[0]
        names = list(node_labels) if node_labels else [f"X{i}" for i in range(n)]

        buf = StringIO()
        buf.write("," + ",".join(names) + "\n")
        for i in range(n):
            row = ",".join(str(int(adj[i, j])) for j in range(n))
            buf.write(f"{names[i]},{row}\n")
        return buf.getvalue()

    def to_file(
        self,
        adj: AdjacencyMatrix,
        path: str | Path,
        *,
        node_labels: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write a PCALG CSV file to disk."""
        text = self.to_string(adj, node_labels=node_labels, metadata=metadata)
        Path(path).write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# bnlearn model-string parser
# ---------------------------------------------------------------------------

_BNLEARN_BLOCK = re.compile(r"\[([^\]]+)\]")


def parse_bnlearn_modelstring(text: str) -> ParseResult:
    """Parse a bnlearn model string into a :class:`ParseResult`.

    Format: ``[A][B|A][C|A:B]`` — each block ``[node|parent1:parent2:...]``
    declares a node and its parents.

    Parameters
    ----------
    text : str
        bnlearn model string.

    Returns
    -------
    ParseResult
    """
    nodes: dict[str, int] = {}
    edges: list[tuple[str, str]] = []

    for m in _BNLEARN_BLOCK.finditer(text):
        block = m.group(1)
        if "|" in block:
            child, parents_str = block.split("|", 1)
            child = child.strip()
            parents = [p.strip() for p in parents_str.split(":")]
        else:
            child = block.strip()
            parents = []

        if child not in nodes:
            nodes[child] = len(nodes)
        for p in parents:
            if p not in nodes:
                nodes[p] = len(nodes)
            edges.append((p, child))

    names = [""] * len(nodes)
    for name, idx in nodes.items():
        names[idx] = name

    n = len(names)
    adj = np.zeros((n, n), dtype=np.int8)
    for src, dst in edges:
        adj[nodes[src], nodes[dst]] = 1

    return ParseResult(
        adjacency=adj,
        node_labels=tuple(names),
        format_type=FormatType.PCALG,
        metadata={},
    )


def format_bnlearn_modelstring(
    adj: AdjacencyMatrix,
    node_labels: tuple[str, ...] = (),
) -> str:
    """Serialize a DAG to bnlearn model-string notation.

    Parameters
    ----------
    adj : AdjacencyMatrix
        Adjacency matrix.
    node_labels : tuple[str, ...], optional
        Node names.

    Returns
    -------
    str
        Model string in ``[A][B|A][C|A:B]`` format.
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]
    names = list(node_labels) if node_labels else [f"X{i}" for i in range(n)]

    parts: list[str] = []
    for j in range(n):
        parents = [names[i] for i in range(n) if adj[i, j]]
        if parents:
            parts.append(f"[{names[j]}|{':'.join(parents)}]")
        else:
            parts.append(f"[{names[j]}]")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_csv_line(line: str) -> list[str]:
    """Split a CSV line handling basic quoting."""
    result: list[str] = []
    current: list[str] = []
    in_quote = False
    for ch in line:
        if ch == '"':
            in_quote = not in_quote
        elif ch == "," and not in_quote:
            result.append("".join(current))
            current = []
        else:
            current.append(ch)
    result.append("".join(current))
    return result


def _indent_xml(xml_str: str) -> str:
    """Add basic indentation to an XML string."""
    try:
        root = ET.fromstring(xml_str)
        _indent_element(root)
        return ET.tostring(root, encoding="unicode", xml_declaration=True)
    except ET.ParseError:
        return xml_str


def _indent_element(elem: ET.Element, level: int = 0) -> None:
    """Recursively indent an ElementTree element."""
    indent = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for child in elem:
            _indent_element(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent
    if not level:
        elem.tail = "\n"


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

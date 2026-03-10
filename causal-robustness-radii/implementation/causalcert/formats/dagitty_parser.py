"""
Comprehensive DAGitty format parser and writer.

Supports the full DAGitty specification including exposure/outcome/adjusted/
latent role markers, position annotations (``X @1,2``), directed (``->``),
bidirected (``<->``), and undirected (``--``) edge types, comments, and the
DAGitty URL encoding scheme.  Bidirected edges are automatically converted
to latent common-cause representations for downstream DAG analysis.
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass, field
from enum import Enum, auto
from io import StringIO
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from causalcert.types import AdjacencyMatrix, FormatType
from causalcert.formats.types import FormatSpec, ParseResult, ValidationIssue, ValidationReport


# ---------------------------------------------------------------------------
# Constants & format spec
# ---------------------------------------------------------------------------

DAGITTY_FORMAT_SPEC = FormatSpec(
    format_type=FormatType.DAGITTY,
    name="DAGitty",
    extensions=(".dagitty",),
    mime_type="text/x-dagitty",
    supports_metadata=True,
    supports_latent=True,
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class NodeRole(Enum):
    """Role of a node in a DAGitty causal diagram."""

    EXPOSURE = "exposure"
    OUTCOME = "outcome"
    ADJUSTED = "adjusted"
    LATENT = "latent"
    UNOBSERVED = "unobserved"
    OTHER = "other"


class EdgeKind(Enum):
    """Type of edge in a DAGitty diagram."""

    DIRECTED = "->"
    BIDIRECTED = "<->"
    UNDIRECTED = "--"


@dataclass(slots=True)
class DAGittyNode:
    """A node in a DAGitty diagram.

    Attributes
    ----------
    name : str
        Node identifier.
    roles : set[NodeRole]
        Assigned roles (exposure, outcome, etc.).
    pos : tuple[float, float] | None
        Layout position as ``(x, y)`` or ``None`` if unspecified.
    """

    name: str
    roles: set[NodeRole] = field(default_factory=set)
    pos: tuple[float, float] | None = None


@dataclass(slots=True)
class DAGittyEdge:
    """An edge in a DAGitty diagram.

    Attributes
    ----------
    src : str
        Source node name.
    dst : str
        Destination node name.
    kind : EdgeKind
        Directed, bidirected, or undirected.
    """

    src: str
    dst: str
    kind: EdgeKind = EdgeKind.DIRECTED


@dataclass(slots=True)
class DAGittyMetadata:
    """Rich metadata from a parsed DAGitty diagram.

    Attributes
    ----------
    graph_type : str
        ``"dag"``, ``"pdag"``, or ``"mag"``.
    exposure : list[str]
        Names of exposure nodes.
    outcome : list[str]
        Names of outcome nodes.
    adjusted : list[str]
        Names of adjusted nodes.
    latent : list[str]
        Names of latent / unobserved nodes.
    positions : dict[str, tuple[float, float]]
        Mapping from node name to ``(x, y)`` layout position.
    bidirected_edges : list[tuple[str, str]]
        Bidirected edges before latent common-cause expansion.
    """

    graph_type: str = "dag"
    exposure: list[str] = field(default_factory=list)
    outcome: list[str] = field(default_factory=list)
    adjusted: list[str] = field(default_factory=list)
    latent: list[str] = field(default_factory=list)
    positions: dict[str, tuple[float, float]] = field(default_factory=dict)
    bidirected_edges: list[tuple[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_COMMENT_LINE = re.compile(r"//[^\n]*")
_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.DOTALL)

# Matches: X @1.5,2.3  or  "node name" @-1,3
_POS_ANNOTATION = re.compile(
    r"@\s*(-?[0-9]+(?:\.[0-9]+)?)\s*,\s*(-?[0-9]+(?:\.[0-9]+)?)"
)

# Edge patterns with groups: src, arrow, dst
_EDGE_PATTERN = re.compile(
    r"""
    ("(?:[^"\\]|\\.)*"|[A-Za-z_]\w*)   # source (quoted or bare)
    \s*
    (<->|->|<-|--)                       # arrow
    \s*
    ("(?:[^"\\]|\\.)*"|[A-Za-z_]\w*)   # destination
    """,
    re.VERBOSE,
)

# Role markers on a line, e.g. "exposure X" or "outcome Y"
_ROLE_PATTERN = re.compile(
    r"""
    \b(exposure|outcome|adjusted|latent|unobserved)\b
    \s+
    ("(?:[^"\\]|\\.)*"|[A-Za-z_]\w*)
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Inline role flags in DAGitty shorthand: X [exposure] or X [e]
_INLINE_ROLE = re.compile(
    r"""
    ("(?:[^"\\]|\\.)*"|[A-Za-z_]\w*)   # node name
    \s+
    \[([^\]]*)\]                        # flags in brackets
    """,
    re.VERBOSE,
)

_GRAPH_TYPE = re.compile(r"\b(dag|pdag|mag)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unquote(s: str) -> str:
    """Strip surrounding quotes and unescape."""
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    return s


def _quote_if_needed(s: str) -> str:
    """Quote a DAGitty identifier if it contains special characters."""
    if re.match(r"^[A-Za-z_]\w*$", s):
        return s
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _parse_role_flag(flag: str) -> NodeRole | None:
    """Map a short or long role flag to :class:`NodeRole`."""
    flag = flag.strip().lower()
    mapping: dict[str, NodeRole] = {
        "e": NodeRole.EXPOSURE, "exposure": NodeRole.EXPOSURE,
        "o": NodeRole.OUTCOME, "outcome": NodeRole.OUTCOME,
        "a": NodeRole.ADJUSTED, "adjusted": NodeRole.ADJUSTED,
        "l": NodeRole.LATENT, "latent": NodeRole.LATENT,
        "u": NodeRole.UNOBSERVED, "unobserved": NodeRole.UNOBSERVED,
    }
    return mapping.get(flag)


# ---------------------------------------------------------------------------
# DAGittyParser
# ---------------------------------------------------------------------------

class DAGittyParser:
    """Parse DAGitty format strings and files.

    Implements the :class:`~causalcert.formats.protocols.FormatParser`
    protocol with full DAGitty format support including role markers,
    position annotations, and bidirected edges.

    Examples
    --------
    >>> parser = DAGittyParser()
    >>> result = parser.parse_string("dag { X -> Y; Z -> X; Z -> Y }")
    >>> result.n_nodes
    3
    """

    def __init__(self, *, expand_bidirected: bool = True) -> None:
        self._expand_bidirected = expand_bidirected

    @property
    def format_spec(self) -> FormatSpec:
        """Return the DAGitty format specification."""
        return DAGITTY_FORMAT_SPEC

    # -- internal parsing ---------------------------------------------------

    @staticmethod
    def _strip_comments(text: str) -> str:
        text = _COMMENT_BLOCK.sub("", text)
        text = _COMMENT_LINE.sub("", text)
        return text

    def _parse_body(self, text: str) -> tuple[
        list[DAGittyNode], list[DAGittyEdge], DAGittyMetadata,
    ]:
        """Parse the body of a DAGitty block (content inside braces)."""
        nodes_dict: dict[str, DAGittyNode] = {}
        edges: list[DAGittyEdge] = []
        meta = DAGittyMetadata()

        # Detect graph type
        m = _GRAPH_TYPE.search(text)
        if m:
            meta.graph_type = m.group(1).lower()

        # Extract body between outermost braces
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            body = text[brace_start + 1 : brace_end]
        else:
            body = text

        def _ensure_node(name: str) -> DAGittyNode:
            if name not in nodes_dict:
                nodes_dict[name] = DAGittyNode(name=name)
            return nodes_dict[name]

        # Parse role declarations
        for m in _ROLE_PATTERN.finditer(body):
            role_str = m.group(1).lower()
            name = _unquote(m.group(2))
            nd = _ensure_node(name)
            role = _parse_role_flag(role_str)
            if role:
                nd.roles.add(role)

        # Parse inline role flags: X [exposure]
        for m in _INLINE_ROLE.finditer(body):
            name = _unquote(m.group(1))
            flags_str = m.group(2)
            nd = _ensure_node(name)
            for flag in re.split(r"[,\s]+", flags_str):
                role = _parse_role_flag(flag)
                if role:
                    nd.roles.add(role)

        # Parse position annotations: X @1,2
        for line in body.split("\n"):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            # Check for node with position
            pos_m = _POS_ANNOTATION.search(line_stripped)
            if pos_m:
                x = float(pos_m.group(1))
                y = float(pos_m.group(2))
                # Find node name before @
                before_at = line_stripped[:pos_m.start()].strip()
                # Remove role flags in brackets
                before_at = re.sub(r"\[.*?\]", "", before_at).strip()
                if before_at:
                    name = _unquote(before_at.split()[-1])
                    nd = _ensure_node(name)
                    nd.pos = (x, y)
                    meta.positions[name] = (x, y)

        # Parse edges
        for m in _EDGE_PATTERN.finditer(body):
            src = _unquote(m.group(1))
            arrow = m.group(2)
            dst = _unquote(m.group(3))
            _ensure_node(src)
            _ensure_node(dst)
            if arrow == "->":
                edges.append(DAGittyEdge(src=src, dst=dst, kind=EdgeKind.DIRECTED))
            elif arrow == "<-":
                edges.append(DAGittyEdge(src=dst, dst=src, kind=EdgeKind.DIRECTED))
            elif arrow == "<->":
                edges.append(DAGittyEdge(
                    src=src, dst=dst, kind=EdgeKind.BIDIRECTED,
                ))
                meta.bidirected_edges.append((src, dst))
            elif arrow == "--":
                edges.append(DAGittyEdge(
                    src=src, dst=dst, kind=EdgeKind.UNDIRECTED,
                ))

        # Also look for standalone node declarations (bare names on lines)
        for line in body.split("\n"):
            line_stripped = line.strip().rstrip(";")
            if not line_stripped:
                continue
            # skip lines with arrows
            if any(a in line_stripped for a in ("->", "<->", "<-", "--")):
                continue
            # skip role declarations
            if re.match(r"\b(exposure|outcome|adjusted|latent|unobserved)\b",
                        line_stripped, re.IGNORECASE):
                continue
            # Might be a standalone node (possibly with position or flags)
            parts = re.split(r"\s+", line_stripped)
            if parts:
                candidate = _unquote(parts[0])
                if (re.match(r'^[A-Za-z_"\']', candidate)
                        and candidate.lower() not in (
                            "dag", "pdag", "mag", "graph", "bb",
                        )):
                    _ensure_node(candidate)

        # Populate metadata role lists
        for nd in nodes_dict.values():
            for r in nd.roles:
                if r is NodeRole.EXPOSURE:
                    meta.exposure.append(nd.name)
                elif r is NodeRole.OUTCOME:
                    meta.outcome.append(nd.name)
                elif r is NodeRole.ADJUSTED:
                    meta.adjusted.append(nd.name)
                elif r in (NodeRole.LATENT, NodeRole.UNOBSERVED):
                    meta.latent.append(nd.name)

        return list(nodes_dict.values()), edges, meta

    def _build_result(
        self,
        nodes: list[DAGittyNode],
        edges: list[DAGittyEdge],
        meta: DAGittyMetadata,
        *,
        validate: bool = True,
    ) -> ParseResult:
        """Convert parsed nodes/edges into a :class:`ParseResult`."""
        name_list = [nd.name for nd in nodes]
        idx = {name: i for i, name in enumerate(name_list)}

        # Expand bidirected edges into latent common causes
        latent_added: list[str] = []
        extra_edges: list[tuple[str, str]] = []
        if self._expand_bidirected:
            for src, dst in meta.bidirected_edges:
                latent_name = f"_L_{src}_{dst}"
                latent_added.append(latent_name)
                extra_edges.append((latent_name, src))
                extra_edges.append((latent_name, dst))

        all_names = name_list + latent_added
        all_idx = {name: i for i, name in enumerate(all_names)}
        n = len(all_names)
        adj = np.zeros((n, n), dtype=np.int8)

        for e in edges:
            if e.kind is EdgeKind.DIRECTED:
                if e.src in all_idx and e.dst in all_idx:
                    adj[all_idx[e.src], all_idx[e.dst]] = 1
            elif e.kind is EdgeKind.UNDIRECTED:
                if e.src in all_idx and e.dst in all_idx:
                    adj[all_idx[e.src], all_idx[e.dst]] = 1
                    adj[all_idx[e.dst], all_idx[e.src]] = 1

        for src, dst in extra_edges:
            if src in all_idx and dst in all_idx:
                adj[all_idx[src], all_idx[dst]] = 1

        # Build metadata dict
        meta_dict: dict[str, Any] = {
            "graph_type": meta.graph_type,
            "exposure": meta.exposure,
            "outcome": meta.outcome,
            "adjusted": meta.adjusted,
            "latent": meta.latent + latent_added,
            "positions": meta.positions,
            "bidirected_edges": meta.bidirected_edges,
        }
        node_roles: dict[str, list[str]] = {}
        for nd in nodes:
            if nd.roles:
                node_roles[nd.name] = [r.value for r in nd.roles]
        if node_roles:
            meta_dict["node_roles"] = node_roles

        vr: ValidationReport | None = None
        if validate:
            issues: list[ValidationIssue] = []
            # acyclicity check
            _check_acyclicity_dagitty(adj, all_names, issues)
            if not name_list:
                issues.append(ValidationIssue(
                    level="warning", message="No nodes found in DAGitty input.",
                ))
            vr = ValidationReport(
                is_valid=not any(i.level == "error" for i in issues),
                issues=tuple(issues),
                format_type=FormatType.DAGITTY,
            )

        return ParseResult(
            adjacency=adj,
            node_labels=tuple(all_names),
            format_type=FormatType.DAGITTY,
            metadata=meta_dict,
            validation=vr,
        )

    # -- public interface ---------------------------------------------------

    def parse_string(self, text: str, *, validate: bool = True) -> ParseResult:
        """Parse a DAGitty string.

        Parameters
        ----------
        text : str
            DAGitty source text.
        validate : bool, optional
            Run post-parse validation (default ``True``).

        Returns
        -------
        ParseResult
        """
        cleaned = self._strip_comments(text)
        nodes, edges, meta = self._parse_body(cleaned)
        return self._build_result(nodes, edges, meta, validate=validate)

    def parse_file(
        self, path: str | Path, *, validate: bool = True,
    ) -> ParseResult:
        """Parse a DAGitty file from disk.

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
            raise FileNotFoundError(f"DAGitty file not found: {p}")
        text = p.read_text(encoding="utf-8")
        return self.parse_string(text, validate=validate)

    def parse_url(self, url: str, *, validate: bool = True) -> ParseResult:
        """Parse a DAGitty URL (``dagitty.net/m...`` encoded diagram).

        The URL query parameter ``m`` contains a percent-encoded DAGitty
        model string.

        Parameters
        ----------
        url : str
            Full URL or just the query-string portion.
        validate : bool, optional
            Run post-parse validation.

        Returns
        -------
        ParseResult
        """
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        model = params.get("m", params.get("model", [""]))[0]
        if not model:
            # Try fragment
            frag_params = urllib.parse.parse_qs(parsed.fragment)
            model = frag_params.get("m", frag_params.get("model", [""]))[0]
        if not model:
            raise ValueError("No DAGitty model found in URL.")
        decoded = urllib.parse.unquote(model)
        return self.parse_string(decoded, validate=validate)

    def validate(self, text: str) -> ValidationReport:
        """Validate a DAGitty string without full parse.

        Parameters
        ----------
        text : str
            DAGitty source text.

        Returns
        -------
        ValidationReport
        """
        issues: list[ValidationIssue] = []
        try:
            cleaned = self._strip_comments(text)
            nodes, edges, meta = self._parse_body(cleaned)
            if not nodes:
                issues.append(ValidationIssue(
                    level="warning", message="No nodes found.",
                ))
            # Check for unknown role types
            known_roles = {r.value for r in NodeRole}
            for nd in nodes:
                for r in nd.roles:
                    if r.value not in known_roles:
                        issues.append(ValidationIssue(
                            level="warning",
                            message=f"Unknown role {r.value!r} on node {nd.name!r}.",
                        ))
        except Exception as exc:
            issues.append(ValidationIssue(level="error", message=str(exc)))

        return ValidationReport(
            is_valid=not any(i.level == "error" for i in issues),
            issues=tuple(issues),
            format_type=FormatType.DAGITTY,
        )


# ---------------------------------------------------------------------------
# DAGittyWriter
# ---------------------------------------------------------------------------

class DAGittyWriter:
    """Serialize a DAG to DAGitty format with position preservation.

    Implements the :class:`~causalcert.formats.protocols.FormatWriter`
    protocol.

    Parameters
    ----------
    graph_type : str
        Type of graph declaration (``"dag"``, ``"pdag"``, ``"mag"``).

    Examples
    --------
    >>> import numpy as np
    >>> w = DAGittyWriter()
    >>> print(w.to_string(np.array([[0,1],[0,0]], dtype=np.int8),
    ...                    node_labels=("X", "Y")))
    dag {
      X
      Y
      X -> Y
    }
    """

    def __init__(self, graph_type: str = "dag") -> None:
        self._graph_type = graph_type

    @property
    def format_spec(self) -> FormatSpec:
        """Return the DAGitty format specification."""
        return DAGITTY_FORMAT_SPEC

    def to_string(
        self,
        adj: AdjacencyMatrix,
        *,
        node_labels: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Serialize a DAG to a DAGitty string.

        Parameters
        ----------
        adj : AdjacencyMatrix
            Adjacency matrix.
        node_labels : tuple[str, ...], optional
            Node names.
        metadata : dict[str, Any] | None, optional
            May contain ``positions``, ``exposure``, ``outcome``, ``adjusted``,
            ``latent``, ``bidirected_edges``, ``node_roles``, ``graph_type``.

        Returns
        -------
        str
            DAGitty source text.
        """
        adj = np.asarray(adj, dtype=np.int8)
        n = adj.shape[0]
        names = list(node_labels) if node_labels else [f"X{i}" for i in range(n)]
        meta = metadata or {}

        graph_type = meta.get("graph_type", self._graph_type)
        positions: dict[str, tuple[float, float]] = meta.get("positions", {})
        exposure: list[str] = meta.get("exposure", [])
        outcome: list[str] = meta.get("outcome", [])
        adjusted: list[str] = meta.get("adjusted", [])
        latent: list[str] = meta.get("latent", [])
        bidirected: list[tuple[str, str]] = meta.get("bidirected_edges", [])
        node_roles: dict[str, list[str]] = meta.get("node_roles", {})

        buf = StringIO()
        buf.write(f"{graph_type} {{\n")

        # Node declarations with roles and positions
        for name in names:
            # Skip synthetic latent nodes from bidirected expansion
            if name.startswith("_L_"):
                continue
            qname = _quote_if_needed(name)
            roles_parts: list[str] = []

            # From metadata lists
            if name in exposure:
                roles_parts.append("exposure")
            if name in outcome:
                roles_parts.append("outcome")
            if name in adjusted:
                roles_parts.append("adjusted")
            if name in latent:
                roles_parts.append("latent")

            # From node_roles dict
            if name in node_roles:
                for r in node_roles[name]:
                    if r not in roles_parts:
                        roles_parts.append(r)

            line = f"  {qname}"
            if roles_parts:
                line += " [" + ",".join(roles_parts) + "]"
            if name in positions:
                x, y = positions[name]
                line += f" @{x},{y}"
            buf.write(line + "\n")

        # Directed edges
        bidirected_set = {(s, d) for s, d in bidirected}
        bidirected_set |= {(d, s) for s, d in bidirected}
        for i in range(n):
            for j in range(n):
                if adj[i, j] and not names[i].startswith("_L_"):
                    if not names[j].startswith("_L_"):
                        pair = (names[i], names[j])
                        # Skip edges that are part of bidirected expansion
                        if pair not in bidirected_set:
                            src_q = _quote_if_needed(names[i])
                            dst_q = _quote_if_needed(names[j])
                            buf.write(f"  {src_q} -> {dst_q}\n")

        # Bidirected edges
        written_bi: set[tuple[str, str]] = set()
        for src, dst in bidirected:
            if (src, dst) not in written_bi and (dst, src) not in written_bi:
                src_q = _quote_if_needed(src)
                dst_q = _quote_if_needed(dst)
                buf.write(f"  {src_q} <-> {dst_q}\n")
                written_bi.add((src, dst))

        buf.write("}\n")
        return buf.getvalue()

    def to_file(
        self,
        adj: AdjacencyMatrix,
        path: str | Path,
        *,
        node_labels: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write a DAGitty file to disk.

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

    def to_url(
        self,
        adj: AdjacencyMatrix,
        *,
        node_labels: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Encode a DAG as a DAGitty URL.

        Parameters
        ----------
        adj : AdjacencyMatrix
            Adjacency matrix.
        node_labels : tuple[str, ...], optional
            Node names.
        metadata : dict[str, Any] | None, optional
            Extra metadata.

        Returns
        -------
        str
            Full ``dagitty.net`` URL with the model encoded in the query.
        """
        model = self.to_string(adj, node_labels=node_labels, metadata=metadata)
        encoded = urllib.parse.quote(model, safe="")
        return f"http://dagitty.net/dags.html?m={encoded}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_acyclicity_dagitty(
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


def round_trip_check(text: str) -> bool:
    """Verify DAGitty round-trip fidelity.

    Parses the text, writes it back, and re-parses to compare adjacency
    matrices and node sets.

    Parameters
    ----------
    text : str
        Original DAGitty source.

    Returns
    -------
    bool
        ``True`` if the round-tripped graph is structurally identical.
    """
    parser = DAGittyParser(expand_bidirected=False)
    writer = DAGittyWriter()
    r1 = parser.parse_string(text, validate=False)
    serialized = writer.to_string(
        r1.adjacency, node_labels=r1.node_labels, metadata=r1.metadata,
    )
    r2 = parser.parse_string(serialized, validate=False)
    return (
        np.array_equal(r1.adjacency, r2.adjacency)
        and set(r1.node_labels) == set(r2.node_labels)
    )

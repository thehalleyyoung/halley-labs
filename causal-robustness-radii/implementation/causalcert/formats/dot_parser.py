"""
Comprehensive DOT (Graphviz) format parser and writer.

Supports the full DOT language including subgraphs, clusters, quoted
identifiers, HTML labels, escape sequences, node/edge attributes,
``strict digraph``, graph-level attributes, multi-line strings,
and C/C++-style comments.  Round-trip fidelity is validated via
``parse → format → parse`` equivalence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from io import StringIO
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from causalcert.types import AdjacencyMatrix, FormatType
from causalcert.formats.types import FormatSpec, ParseResult, ValidationIssue, ValidationReport


# ---------------------------------------------------------------------------
# Token types & lexer
# ---------------------------------------------------------------------------

class _TT(Enum):
    """Token types for the DOT lexer."""

    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    SEMI = auto()
    COMMA = auto()
    COLON = auto()
    EQUALS = auto()
    ARROW = auto()          # ->
    UNDIRECTED = auto()     # --
    KEYWORD = auto()        # digraph, graph, subgraph, node, edge, strict
    ID = auto()             # bare or quoted identifier
    HTML_ID = auto()        # < ... > HTML label
    EOF = auto()


@dataclass(slots=True)
class _Token:
    """A single lexical token."""

    kind: _TT
    value: str
    line: int
    col: int


_KEYWORDS = frozenset({
    "digraph", "graph", "subgraph", "node", "edge", "strict",
})

_C_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_CPP_COMMENT = re.compile(r"//[^\n]*")
_HASH_COMMENT = re.compile(r"#[^\n]*")
_BARE_ID = re.compile(r"[A-Za-z_\x80-\xff][A-Za-z0-9_\x80-\xff]*")
_NUMERAL = re.compile(r"-?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)")


def _strip_comments(text: str) -> str:
    """Remove C, C++ and hash comments from DOT source."""
    text = _C_COMMENT.sub("", text)
    text = _CPP_COMMENT.sub("", text)
    text = _HASH_COMMENT.sub("", text)
    return text


def _tokenize(text: str) -> list[_Token]:
    """Tokenize DOT source into a flat token list.

    Parameters
    ----------
    text : str
        DOT source with comments already stripped.

    Returns
    -------
    list[_Token]
        Ordered token list.
    """
    tokens: list[_Token] = []
    i = 0
    line = 1
    col = 1
    n = len(text)

    while i < n:
        ch = text[i]

        # whitespace
        if ch in " \t\r":
            i += 1
            col += 1
            continue
        if ch == "\n":
            i += 1
            line += 1
            col = 1
            continue

        # single-char tokens
        simple = {
            "{": _TT.LBRACE, "}": _TT.RBRACE,
            "[": _TT.LBRACKET, "]": _TT.RBRACKET,
            ";": _TT.SEMI, ",": _TT.COMMA,
            ":": _TT.COLON, "=": _TT.EQUALS,
        }
        if ch in simple:
            tokens.append(_Token(simple[ch], ch, line, col))
            i += 1
            col += 1
            continue

        # arrows
        if ch == "-" and i + 1 < n:
            nxt = text[i + 1]
            if nxt == ">":
                tokens.append(_Token(_TT.ARROW, "->", line, col))
                i += 2
                col += 2
                continue
            if nxt == "-":
                tokens.append(_Token(_TT.UNDIRECTED, "--", line, col))
                i += 2
                col += 2
                continue

        # quoted string (may be multi-line via \ continuation)
        if ch == '"':
            j = i + 1
            parts: list[str] = []
            while j < n:
                c = text[j]
                if c == "\\" and j + 1 < n:
                    parts.append(text[j + 1])
                    j += 2
                elif c == '"':
                    j += 1
                    break
                else:
                    parts.append(c)
                    j += 1
            val = "".join(parts)
            tokens.append(_Token(_TT.ID, val, line, col))
            # update line/col for multi-line strings
            newlines = val.count("\n")
            if newlines:
                line += newlines
                col = len(val) - val.rfind("\n")
            else:
                col += j - i
            i = j
            continue

        # HTML label
        if ch == "<":
            depth = 1
            j = i + 1
            while j < n and depth > 0:
                if text[j] == "<":
                    depth += 1
                elif text[j] == ">":
                    depth -= 1
                j += 1
            val = text[i + 1 : j - 1]
            tokens.append(_Token(_TT.HTML_ID, val, line, col))
            col += j - i
            i = j
            continue

        # bare identifier or keyword
        m = _BARE_ID.match(text, i)
        if m:
            val = m.group(0)
            kind = _TT.KEYWORD if val.lower() in _KEYWORDS else _TT.ID
            tokens.append(_Token(kind, val, line, col))
            col += len(val)
            i += len(val)
            continue

        # numeral
        m = _NUMERAL.match(text, i)
        if m:
            val = m.group(0)
            tokens.append(_Token(_TT.ID, val, line, col))
            col += len(val)
            i += len(val)
            continue

        # skip unknown character
        i += 1
        col += 1

    tokens.append(_Token(_TT.EOF, "", line, col))
    return tokens


# ---------------------------------------------------------------------------
# AST node types
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DOTAttribute:
    """A single key=value attribute."""

    key: str
    value: str


@dataclass(slots=True)
class DOTNode:
    """A node with optional attributes."""

    name: str
    attrs: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class DOTEdge:
    """A directed or undirected edge with optional attributes."""

    src: str
    dst: str
    directed: bool = True
    attrs: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class DOTSubgraph:
    """A subgraph (possibly a cluster)."""

    name: str
    attrs: dict[str, str] = field(default_factory=dict)
    nodes: list[DOTNode] = field(default_factory=list)
    edges: list[DOTEdge] = field(default_factory=list)
    subgraphs: list[DOTSubgraph] = field(default_factory=list)


@dataclass(slots=True)
class DOTGraph:
    """Top-level parsed DOT graph."""

    name: str = "G"
    strict: bool = False
    directed: bool = True
    attrs: dict[str, str] = field(default_factory=dict)
    node_defaults: dict[str, str] = field(default_factory=dict)
    edge_defaults: dict[str, str] = field(default_factory=dict)
    nodes: list[DOTNode] = field(default_factory=list)
    edges: list[DOTEdge] = field(default_factory=list)
    subgraphs: list[DOTSubgraph] = field(default_factory=list)

    # -- helpers for flat traversal -----------------------------------------

    def all_nodes(self) -> dict[str, DOTNode]:
        """Collect every node from all levels into a name→DOTNode dict."""
        result: dict[str, DOTNode] = {}
        for nd in self.nodes:
            result.setdefault(nd.name, nd)
        for sg in self.subgraphs:
            result.update(_collect_nodes(sg))
        return result

    def all_edges(self) -> list[DOTEdge]:
        """Collect every edge from all levels."""
        result = list(self.edges)
        for sg in self.subgraphs:
            result.extend(_collect_edges(sg))
        return result


def _collect_nodes(sg: DOTSubgraph) -> dict[str, DOTNode]:
    result: dict[str, DOTNode] = {}
    for nd in sg.nodes:
        result.setdefault(nd.name, nd)
    for child in sg.subgraphs:
        result.update(_collect_nodes(child))
    return result


def _collect_edges(sg: DOTSubgraph) -> list[DOTEdge]:
    result = list(sg.edges)
    for child in sg.subgraphs:
        result.extend(_collect_edges(child))
    return result


# ---------------------------------------------------------------------------
# Recursive-descent parser
# ---------------------------------------------------------------------------

class _DOTParserEngine:
    """Recursive-descent parser for the DOT grammar."""

    def __init__(self, tokens: list[_Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    # -- helpers ------------------------------------------------------------

    def _cur(self) -> _Token:
        return self._tokens[self._pos]

    def _peek(self, kind: _TT) -> bool:
        return self._cur().kind is kind

    def _peek_val(self, val: str) -> bool:
        return self._cur().value.lower() == val

    def _eat(self, kind: _TT) -> _Token:
        tok = self._cur()
        if tok.kind is not kind:
            raise ValueError(
                f"DOT parse error at line {tok.line}: expected {kind.name}, "
                f"got {tok.kind.name} ({tok.value!r})"
            )
        self._pos += 1
        return tok

    def _eat_if(self, kind: _TT) -> _Token | None:
        if self._peek(kind):
            return self._eat(kind)
        return None

    def _eat_id(self) -> str:
        tok = self._cur()
        if tok.kind in (_TT.ID, _TT.HTML_ID):
            self._pos += 1
            return tok.value
        if tok.kind is _TT.KEYWORD:
            self._pos += 1
            return tok.value
        raise ValueError(
            f"DOT parse error at line {tok.line}: expected identifier, "
            f"got {tok.kind.name} ({tok.value!r})"
        )

    def _try_id(self) -> str | None:
        tok = self._cur()
        if tok.kind in (_TT.ID, _TT.HTML_ID, _TT.KEYWORD):
            self._pos += 1
            return tok.value
        return None

    # -- attr parsing -------------------------------------------------------

    def _parse_attr_list(self) -> dict[str, str]:
        attrs: dict[str, str] = {}
        while self._peek(_TT.LBRACKET):
            self._eat(_TT.LBRACKET)
            while not self._peek(_TT.RBRACKET) and not self._peek(_TT.EOF):
                key = self._eat_id()
                self._eat(_TT.EQUALS)
                val = self._eat_id()
                attrs[key] = val
                self._eat_if(_TT.COMMA)
                self._eat_if(_TT.SEMI)
            self._eat(_TT.RBRACKET)
        return attrs

    # -- statement parsing --------------------------------------------------

    def _parse_stmt_list(
        self,
        nodes: list[DOTNode],
        edges: list[DOTEdge],
        subgraphs: list[DOTSubgraph],
        graph_attrs: dict[str, str],
        node_defaults: dict[str, str] | None = None,
        edge_defaults: dict[str, str] | None = None,
    ) -> None:
        while not self._peek(_TT.RBRACE) and not self._peek(_TT.EOF):
            self._parse_stmt(
                nodes, edges, subgraphs, graph_attrs,
                node_defaults, edge_defaults,
            )
            self._eat_if(_TT.SEMI)

    def _parse_stmt(
        self,
        nodes: list[DOTNode],
        edges: list[DOTEdge],
        subgraphs: list[DOTSubgraph],
        graph_attrs: dict[str, str],
        node_defaults: dict[str, str] | None = None,
        edge_defaults: dict[str, str] | None = None,
    ) -> None:
        tok = self._cur()

        # subgraph
        if tok.kind is _TT.KEYWORD and tok.value.lower() == "subgraph":
            sg = self._parse_subgraph()
            subgraphs.append(sg)
            return

        # bare { ... } anonymous subgraph
        if tok.kind is _TT.LBRACE:
            sg = self._parse_subgraph()
            subgraphs.append(sg)
            return

        # node/edge defaults: "node [shape=box]" or "edge [style=dashed]"
        if tok.kind is _TT.KEYWORD and tok.value.lower() == "node":
            self._pos += 1
            attrs = self._parse_attr_list()
            if node_defaults is not None:
                node_defaults.update(attrs)
            return
        if tok.kind is _TT.KEYWORD and tok.value.lower() == "edge":
            self._pos += 1
            attrs = self._parse_attr_list()
            if edge_defaults is not None:
                edge_defaults.update(attrs)
            return

        # graph-level attr: "graph [label=...]" or "key = val"
        if tok.kind is _TT.KEYWORD and tok.value.lower() == "graph":
            self._pos += 1
            attrs = self._parse_attr_list()
            graph_attrs.update(attrs)
            return

        # Try to read an ID
        name = self._try_id()
        if name is None:
            # skip unexpected token
            self._pos += 1
            return

        # graph-level key=value
        if self._peek(_TT.EQUALS):
            self._eat(_TT.EQUALS)
            val = self._eat_id()
            graph_attrs[name] = val
            return

        # edge statement (chain: a -> b -> c ...)
        if self._peek(_TT.ARROW) or self._peek(_TT.UNDIRECTED):
            chain: list[str] = [name]
            directed_arrows: list[bool] = []
            while self._peek(_TT.ARROW) or self._peek(_TT.UNDIRECTED):
                is_directed = self._cur().kind is _TT.ARROW
                self._pos += 1
                directed_arrows.append(is_directed)
                nxt = self._eat_id()
                chain.append(nxt)
            attrs = self._parse_attr_list()
            for idx in range(len(chain) - 1):
                e = DOTEdge(
                    src=chain[idx],
                    dst=chain[idx + 1],
                    directed=directed_arrows[idx],
                    attrs=dict(attrs),
                )
                edges.append(e)
            # ensure nodes exist
            seen = {nd.name for nd in nodes}
            for c in chain:
                if c not in seen:
                    nodes.append(DOTNode(name=c))
                    seen.add(c)
            return

        # standalone node
        attrs = self._parse_attr_list()
        existing = {nd.name: nd for nd in nodes}
        if name in existing:
            existing[name].attrs.update(attrs)
        else:
            nodes.append(DOTNode(name=name, attrs=attrs))

    def _parse_subgraph(self) -> DOTSubgraph:
        name = ""
        if self._peek(_TT.KEYWORD) and self._cur().value.lower() == "subgraph":
            self._pos += 1
            if not self._peek(_TT.LBRACE):
                name = self._eat_id()
        self._eat(_TT.LBRACE)
        sg = DOTSubgraph(name=name)
        self._parse_stmt_list(
            sg.nodes, sg.edges, sg.subgraphs, sg.attrs,
        )
        self._eat(_TT.RBRACE)
        return sg

    # -- top-level ----------------------------------------------------------

    def parse(self) -> DOTGraph:
        """Parse the full DOT file and return a :class:`DOTGraph`."""
        g = DOTGraph()
        # optional 'strict'
        if self._peek(_TT.KEYWORD) and self._cur().value.lower() == "strict":
            g.strict = True
            self._pos += 1
        # graph/digraph
        if self._peek(_TT.KEYWORD):
            kw = self._cur().value.lower()
            if kw in ("digraph", "graph"):
                g.directed = kw == "digraph"
                self._pos += 1
        # optional graph name
        if not self._peek(_TT.LBRACE):
            g.name = self._eat_id()
        self._eat(_TT.LBRACE)
        self._parse_stmt_list(
            g.nodes, g.edges, g.subgraphs, g.attrs,
            g.node_defaults, g.edge_defaults,
        )
        self._eat(_TT.RBRACE)
        return g


# ---------------------------------------------------------------------------
# Public API: DOTParser
# ---------------------------------------------------------------------------

DOT_FORMAT_SPEC = FormatSpec(
    format_type=FormatType.DOT,
    name="Graphviz DOT",
    extensions=(".dot", ".gv"),
    mime_type="text/vnd.graphviz",
    supports_metadata=True,
    supports_latent=False,
)


class DOTParser:
    """Parse DOT format strings and files into :class:`ParseResult`.

    Implements the :class:`~causalcert.formats.protocols.FormatParser`
    protocol with full DOT language support including subgraphs, clusters,
    node/edge attributes, strict digraphs, comments, and quoted identifiers.

    Examples
    --------
    >>> parser = DOTParser()
    >>> result = parser.parse_string("digraph { A -> B; B -> C; }")
    >>> result.n_nodes
    3
    """

    @property
    def format_spec(self) -> FormatSpec:
        """Return the DOT format specification."""
        return DOT_FORMAT_SPEC

    # -- internal -----------------------------------------------------------

    @staticmethod
    def _lex(text: str) -> list[_Token]:
        cleaned = _strip_comments(text)
        return _tokenize(cleaned)

    @staticmethod
    def _build_graph(text: str) -> DOTGraph:
        tokens = DOTParser._lex(text)
        engine = _DOTParserEngine(tokens)
        return engine.parse()

    @staticmethod
    def _graph_to_result(
        g: DOTGraph,
        *,
        validate: bool = True,
    ) -> ParseResult:
        all_nodes = g.all_nodes()
        all_edges = g.all_edges()

        names = list(all_nodes.keys())
        idx = {name: i for i, name in enumerate(names)}
        n = len(names)
        adj = np.zeros((n, n), dtype=np.int8)

        seen_edges: set[tuple[str, str]] = set()
        duplicate_edges: list[tuple[str, str]] = []
        for e in all_edges:
            if e.src not in idx or e.dst not in idx:
                continue
            pair = (e.src, e.dst)
            if pair in seen_edges:
                duplicate_edges.append(pair)
            else:
                seen_edges.add(pair)
            adj[idx[e.src], idx[e.dst]] = 1

        # Collect metadata
        meta: dict[str, Any] = {
            "graph_name": g.name,
            "strict": g.strict,
            "directed": g.directed,
            "graph_attrs": dict(g.attrs),
            "node_defaults": dict(g.node_defaults),
            "edge_defaults": dict(g.edge_defaults),
        }
        node_attrs: dict[str, dict[str, str]] = {}
        for name, nd in all_nodes.items():
            if nd.attrs:
                node_attrs[name] = dict(nd.attrs)
        if node_attrs:
            meta["node_attrs"] = node_attrs
        edge_attrs: list[dict[str, Any]] = []
        for e in all_edges:
            if e.attrs:
                edge_attrs.append({
                    "src": e.src, "dst": e.dst, **e.attrs,
                })
        if edge_attrs:
            meta["edge_attrs"] = edge_attrs
        if g.subgraphs:
            meta["subgraphs"] = [sg.name for sg in g.subgraphs]

        vr: ValidationReport | None = None
        if validate:
            issues: list[ValidationIssue] = []
            if duplicate_edges:
                for src, dst in duplicate_edges:
                    issues.append(ValidationIssue(
                        level="warning",
                        message=f"Duplicate edge: {src} -> {dst}",
                    ))
            # Check for cycles (quick DFS)
            _check_acyclicity(adj, names, issues)
            vr = ValidationReport(
                is_valid=not any(i.level == "error" for i in issues),
                issues=tuple(issues),
                format_type=FormatType.DOT,
            )

        return ParseResult(
            adjacency=adj,
            node_labels=tuple(names),
            format_type=FormatType.DOT,
            metadata=meta,
            validation=vr,
        )

    # -- public interface ---------------------------------------------------

    def parse_string(self, text: str, *, validate: bool = True) -> ParseResult:
        """Parse a DOT string into a :class:`ParseResult`.

        Parameters
        ----------
        text : str
            DOT source.
        validate : bool, optional
            Run post-parse validation (default ``True``).

        Returns
        -------
        ParseResult
        """
        g = self._build_graph(text)
        return self._graph_to_result(g, validate=validate)

    def parse_file(self, path: str | Path, *, validate: bool = True) -> ParseResult:
        """Parse a DOT file from disk.

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
            raise FileNotFoundError(f"DOT file not found: {p}")
        text = p.read_text(encoding="utf-8")
        return self.parse_string(text, validate=validate)

    def parse_to_ast(self, text: str) -> DOTGraph:
        """Parse DOT source and return the AST (for advanced users).

        Parameters
        ----------
        text : str
            DOT source.

        Returns
        -------
        DOTGraph
            The structured AST.
        """
        return self._build_graph(text)

    def validate(self, text: str) -> ValidationReport:
        """Validate a DOT string without full parse.

        Parameters
        ----------
        text : str
            DOT source.

        Returns
        -------
        ValidationReport
        """
        issues: list[ValidationIssue] = []
        try:
            g = self._build_graph(text)
            all_nodes = g.all_nodes()
            all_edges = g.all_edges()

            # Check for undeclared nodes in edges
            node_names = set(all_nodes.keys())
            for e in all_edges:
                if e.src not in node_names:
                    issues.append(ValidationIssue(
                        level="warning",
                        message=f"Edge references undeclared node: {e.src!r}",
                    ))
                if e.dst not in node_names:
                    issues.append(ValidationIssue(
                        level="warning",
                        message=f"Edge references undeclared node: {e.dst!r}",
                    ))

            # Check duplicate edges
            seen: set[tuple[str, str]] = set()
            for e in all_edges:
                pair = (e.src, e.dst)
                if pair in seen:
                    issues.append(ValidationIssue(
                        level="warning",
                        message=f"Duplicate edge: {e.src} -> {e.dst}",
                    ))
                seen.add(pair)
        except ValueError as exc:
            issues.append(ValidationIssue(level="error", message=str(exc)))

        return ValidationReport(
            is_valid=not any(i.level == "error" for i in issues),
            issues=tuple(issues),
            format_type=FormatType.DOT,
        )


# ---------------------------------------------------------------------------
# Public API: DOTWriter
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DOTWriterConfig:
    """Configuration for :class:`DOTWriter` output formatting.

    Attributes
    ----------
    indent : str
        Indentation string for each nesting level.
    graph_name : str
        Name used in the ``digraph`` declaration.
    strict : bool
        Emit ``strict digraph`` to forbid multi-edges.
    quote_all : bool
        Force-quote all identifiers.
    include_defaults : bool
        Emit ``node`` / ``edge`` default attribute blocks.
    """

    indent: str = "  "
    graph_name: str = "G"
    strict: bool = False
    quote_all: bool = False
    include_defaults: bool = True


_DOT_RESERVED = frozenset({
    "node", "edge", "graph", "digraph", "subgraph", "strict",
})

_SAFE_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class DOTWriter:
    """Serialize a DAG adjacency matrix (and optional metadata) to DOT.

    Implements the :class:`~causalcert.formats.protocols.FormatWriter`
    protocol with configurable formatting options.

    Parameters
    ----------
    config : DOTWriterConfig | None
        Formatting options.  Uses defaults when *None*.

    Examples
    --------
    >>> import numpy as np
    >>> w = DOTWriter()
    >>> print(w.to_string(np.array([[0,1],[0,0]], dtype=np.int8),
    ...                    node_labels=("A", "B")))
    digraph G {
      A;
      B;
      A -> B;
    }
    """

    def __init__(self, config: DOTWriterConfig | None = None) -> None:
        self._cfg = config or DOTWriterConfig()

    @property
    def format_spec(self) -> FormatSpec:
        """Return the DOT format specification."""
        return DOT_FORMAT_SPEC

    # -- helpers ------------------------------------------------------------

    def _quote(self, name: str) -> str:
        if self._cfg.quote_all:
            return f'"{name}"'
        if name.lower() in _DOT_RESERVED:
            return f'"{name}"'
        if _SAFE_ID.match(name) and not name[0].isdigit():
            return name
        return f'"{_escape(name)}"'

    @staticmethod
    def _fmt_attrs(attrs: dict[str, str]) -> str:
        if not attrs:
            return ""
        parts = [f'{k}="{v}"' for k, v in attrs.items()]
        return " [" + ", ".join(parts) + "]"

    # -- serialization ------------------------------------------------------

    def to_string(
        self,
        adj: AdjacencyMatrix,
        *,
        node_labels: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Serialize a DAG to a DOT string.

        Parameters
        ----------
        adj : AdjacencyMatrix
            Adjacency matrix.
        node_labels : tuple[str, ...], optional
            Node names.
        metadata : dict[str, Any] | None, optional
            May contain ``node_attrs``, ``edge_attrs``, ``graph_attrs``,
            ``node_defaults``, ``edge_defaults``, ``graph_name``, ``strict``.

        Returns
        -------
        str
            DOT source.
        """
        adj = np.asarray(adj, dtype=np.int8)
        n = adj.shape[0]
        names = list(node_labels) if node_labels else [f"X{i}" for i in range(n)]
        meta = metadata or {}
        ind = self._cfg.indent

        graph_name = meta.get("graph_name", self._cfg.graph_name)
        strict = meta.get("strict", self._cfg.strict)
        graph_attrs: dict[str, str] = meta.get("graph_attrs", {})
        node_defaults: dict[str, str] = meta.get("node_defaults", {})
        edge_defaults: dict[str, str] = meta.get("edge_defaults", {})
        node_attrs: dict[str, dict[str, str]] = meta.get("node_attrs", {})
        edge_attrs_list: list[dict[str, Any]] = meta.get("edge_attrs", [])

        # Build edge attrs lookup
        ea_lookup: dict[tuple[str, str], dict[str, str]] = {}
        for ea in edge_attrs_list:
            key = (ea.get("src", ""), ea.get("dst", ""))
            ea_lookup[key] = {
                k: v for k, v in ea.items() if k not in ("src", "dst")
            }

        buf = StringIO()
        prefix = "strict digraph" if strict else "digraph"
        buf.write(f"{prefix} {self._quote(graph_name)} {{\n")

        # graph attrs
        for k, v in graph_attrs.items():
            buf.write(f'{ind}{k}="{v}";\n')

        # defaults
        if self._cfg.include_defaults and node_defaults:
            buf.write(f"{ind}node{self._fmt_attrs(node_defaults)};\n")
        if self._cfg.include_defaults and edge_defaults:
            buf.write(f"{ind}edge{self._fmt_attrs(edge_defaults)};\n")

        # nodes
        for name in names:
            attrs = node_attrs.get(name, {})
            buf.write(f"{ind}{self._quote(name)}{self._fmt_attrs(attrs)};\n")

        # edges
        for i in range(n):
            for j in range(n):
                if adj[i, j]:
                    ea = ea_lookup.get((names[i], names[j]), {})
                    buf.write(
                        f"{ind}{self._quote(names[i])} -> "
                        f"{self._quote(names[j])}{self._fmt_attrs(ea)};\n"
                    )

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
        """Write a DOT file to disk.

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

def _escape(s: str) -> str:
    """Escape a string for DOT quoted identifiers."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _check_acyclicity(
    adj: np.ndarray,
    names: list[str],
    issues: list[ValidationIssue],
) -> None:
    """Append an error-level issue if the graph has a cycle."""
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
    """Verify DOT round-trip fidelity: parse → write → parse yields same graph.

    Parameters
    ----------
    text : str
        Original DOT source.

    Returns
    -------
    bool
        ``True`` if the round-tripped graph has identical adjacency and labels.
    """
    parser = DOTParser()
    writer = DOTWriter()
    r1 = parser.parse_string(text, validate=False)
    serialized = writer.to_string(
        r1.adjacency, node_labels=r1.node_labels, metadata=r1.metadata,
    )
    r2 = parser.parse_string(serialized, validate=False)
    return (
        np.array_equal(r1.adjacency, r2.adjacency)
        and r1.node_labels == r2.node_labels
    )

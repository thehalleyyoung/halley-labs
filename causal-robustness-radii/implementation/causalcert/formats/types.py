"""
Type definitions for the extended format support sub-package.

Provides data structures representing format specifications, parse results,
and validation reports for DAG serialization / deserialization in multiple
file formats (DOT, DAGitty, BIF, TETRAD, pcalg, CSV, JSON).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from causalcert.types import AdjacencyMatrix, FormatType, NodeId


# ---------------------------------------------------------------------------
# FormatSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FormatSpec:
    """Specification of a file format for DAG serialization.

    Captures metadata about a supported format, including its file
    extensions, MIME type, and feature capabilities.

    Attributes
    ----------
    format_type : FormatType
        Enum variant identifying this format.
    name : str
        Human-readable display name (e.g. ``"Graphviz DOT"``).
    extensions : tuple[str, ...]
        Recognised file extensions including the dot (e.g. ``(".dot", ".gv")``).
    mime_type : str
        MIME type string for this format.
    supports_metadata : bool
        Whether the format can encode edge weights, variable types, etc.
    supports_latent : bool
        Whether the format can represent latent (unobserved) variables.
    """

    format_type: FormatType
    name: str
    extensions: tuple[str, ...]
    mime_type: str = "application/octet-stream"
    supports_metadata: bool = False
    supports_latent: bool = False

    def matches_extension(self, path: str) -> bool:
        """Check whether *path* has an extension recognised by this format.

        Parameters
        ----------
        path : str
            File path or file name to check.

        Returns
        -------
        bool
            ``True`` if the path suffix matches one of the known extensions.
        """
        lower = path.lower()
        return any(lower.endswith(ext) for ext in self.extensions)


# ---------------------------------------------------------------------------
# ValidationIssue / ValidationReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """A single issue found during format validation.

    Attributes
    ----------
    level : str
        Severity level: ``"error"``, ``"warning"``, or ``"info"``.
    line : int | None
        Line number in the source file, if applicable.
    message : str
        Human-readable description of the issue.
    """

    level: str
    line: int | None = None
    message: str = ""


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Result of validating a file against a format specification.

    Attributes
    ----------
    is_valid : bool
        ``True`` if no errors were found (warnings are permitted).
    issues : tuple[ValidationIssue, ...]
        All issues discovered, ordered by line number.
    format_type : FormatType | None
        Detected or declared format type.
    """

    is_valid: bool
    issues: tuple[ValidationIssue, ...] = ()
    format_type: FormatType | None = None

    @property
    def n_errors(self) -> int:
        """Number of error-level issues."""
        return sum(1 for i in self.issues if i.level == "error")

    @property
    def n_warnings(self) -> int:
        """Number of warning-level issues."""
        return sum(1 for i in self.issues if i.level == "warning")


# ---------------------------------------------------------------------------
# ParseResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParseResult:
    """Result of parsing a DAG from a file or string.

    Wraps the adjacency matrix together with any node labels, metadata, and
    validation information extracted during parsing.

    Attributes
    ----------
    adjacency : AdjacencyMatrix
        Parsed adjacency matrix.
    node_labels : tuple[str, ...]
        Node names in the order matching matrix indices.
    format_type : FormatType
        Format that was used to parse the input.
    metadata : dict[str, Any]
        Additional metadata extracted from the file (weights, positions, etc.).
    validation : ValidationReport | None
        Post-parse validation report, if validation was requested.
    """

    adjacency: AdjacencyMatrix
    node_labels: tuple[str, ...] = ()
    format_type: FormatType = FormatType.JSON
    metadata: dict[str, Any] = field(default_factory=dict)
    validation: ValidationReport | None = None

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the parsed graph."""
        return self.adjacency.shape[0]

    @property
    def n_edges(self) -> int:
        """Number of directed edges in the parsed graph."""
        return int(self.adjacency.sum())

    def label_of(self, node: NodeId) -> str:
        """Return the label string for *node*.

        Parameters
        ----------
        node : NodeId
            Zero-based node index.

        Returns
        -------
        str
            Node label, or ``"X{node}"`` if labels are unavailable.
        """
        if node < len(self.node_labels):
            return self.node_labels[node]
        return f"X{node}"

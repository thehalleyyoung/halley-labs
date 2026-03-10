"""
Protocols for the format parsing and writing sub-system.

Defines structural sub-typing interfaces for format parsers and writers,
enabling extensible support for new DAG serialization formats without
modifying existing code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO, Protocol, TextIO, runtime_checkable

from causalcert.types import AdjacencyMatrix, FormatType
from causalcert.formats.types import FormatSpec, ParseResult, ValidationReport


# ---------------------------------------------------------------------------
# FormatParser
# ---------------------------------------------------------------------------


@runtime_checkable
class FormatParser(Protocol):
    """Reads a DAG from a file or string in a specific format.

    Implementations parse exactly one :class:`~causalcert.types.FormatType`
    and register themselves with the format registry.
    """

    @property
    def format_spec(self) -> FormatSpec:
        """Return the format specification handled by this parser."""
        ...

    def parse_string(
        self,
        text: str,
        *,
        validate: bool = True,
    ) -> ParseResult:
        """Parse a DAG from an in-memory string.

        Parameters
        ----------
        text : str
            Source text in the target format.
        validate : bool, optional
            If ``True`` (default), run post-parse validation.

        Returns
        -------
        ParseResult
            Parsed graph with metadata and optional validation report.

        Raises
        ------
        ValueError
            If the input is syntactically invalid and cannot be recovered.
        """
        ...

    def parse_file(
        self,
        path: str | Path,
        *,
        validate: bool = True,
    ) -> ParseResult:
        """Parse a DAG from a file on disk.

        Parameters
        ----------
        path : str | Path
            Path to the file.
        validate : bool, optional
            If ``True`` (default), run post-parse validation.

        Returns
        -------
        ParseResult
            Parsed graph with metadata and optional validation report.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the file content is invalid.
        """
        ...

    def validate(self, text: str) -> ValidationReport:
        """Validate a string against this format without fully parsing.

        Parameters
        ----------
        text : str
            Source text to validate.

        Returns
        -------
        ValidationReport
            Report containing any errors and warnings found.
        """
        ...


# ---------------------------------------------------------------------------
# FormatWriter
# ---------------------------------------------------------------------------


@runtime_checkable
class FormatWriter(Protocol):
    """Serializes a DAG to a specific file format.

    Implementations convert an adjacency matrix and optional metadata
    into a string or file in the target format.
    """

    @property
    def format_spec(self) -> FormatSpec:
        """Return the format specification this writer targets."""
        ...

    def to_string(
        self,
        adj: AdjacencyMatrix,
        *,
        node_labels: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Serialize a DAG to an in-memory string.

        Parameters
        ----------
        adj : AdjacencyMatrix
            Adjacency matrix to serialize.
        node_labels : tuple[str, ...], optional
            Human-readable names for each node.
        metadata : dict[str, Any] | None, optional
            Extra metadata to embed (format-dependent).

        Returns
        -------
        str
            Serialized representation.
        """
        ...

    def to_file(
        self,
        adj: AdjacencyMatrix,
        path: str | Path,
        *,
        node_labels: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Serialize a DAG and write it to a file.

        Parameters
        ----------
        adj : AdjacencyMatrix
            Adjacency matrix to serialize.
        path : str | Path
            Destination file path.
        node_labels : tuple[str, ...], optional
            Human-readable names for each node.
        metadata : dict[str, Any] | None, optional
            Extra metadata to embed.
        """
        ...

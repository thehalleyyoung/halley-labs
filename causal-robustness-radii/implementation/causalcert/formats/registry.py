"""
Format registration system for extensible DAG I/O.

Provides a singleton registry that maps :class:`~causalcert.types.FormatType`
values to their :class:`FormatParser` and :class:`FormatWriter`
implementations.  Third-party code can register additional formats at
import time without modifying CausalCert internals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from causalcert.types import AdjacencyMatrix, FormatType
from causalcert.formats.types import FormatSpec, ParseResult
from causalcert.formats.protocols import FormatParser, FormatWriter


# ---------------------------------------------------------------------------
# FormatRegistry
# ---------------------------------------------------------------------------


class FormatRegistry:
    """Central registry for format parsers and writers.

    Maintains two dictionaries — one mapping :class:`FormatType` to
    :class:`FormatParser`, and another to :class:`FormatWriter`.
    Discover-and-dispatch logic for ``load`` / ``save`` lives here so that
    callers do not need to know which concrete parser to instantiate.

    Examples
    --------
    >>> registry = FormatRegistry()
    >>> registry.register_parser(FormatType.DOT, my_dot_parser)
    >>> result = registry.load("graph.dot")
    """

    def __init__(self) -> None:
        self._parsers: dict[FormatType, FormatParser] = {}
        self._writers: dict[FormatType, FormatWriter] = {}
        self._specs: dict[FormatType, FormatSpec] = {}

    # -- registration -------------------------------------------------------

    def register_parser(
        self,
        fmt: FormatType,
        parser: FormatParser,
    ) -> None:
        """Register a parser for a specific format type.

        Parameters
        ----------
        fmt : FormatType
            The format type this parser handles.
        parser : FormatParser
            Parser instance implementing the :class:`FormatParser` protocol.

        Raises
        ------
        ValueError
            If a parser for *fmt* is already registered.
        """
        if fmt in self._parsers:
            raise ValueError(
                f"A parser for {fmt.value!r} is already registered."
            )
        self._parsers[fmt] = parser
        self._specs[fmt] = parser.format_spec

    def register_writer(
        self,
        fmt: FormatType,
        writer: FormatWriter,
    ) -> None:
        """Register a writer for a specific format type.

        Parameters
        ----------
        fmt : FormatType
            The format type this writer targets.
        writer : FormatWriter
            Writer instance implementing the :class:`FormatWriter` protocol.

        Raises
        ------
        ValueError
            If a writer for *fmt* is already registered.
        """
        if fmt in self._writers:
            raise ValueError(
                f"A writer for {fmt.value!r} is already registered."
            )
        self._writers[fmt] = writer
        self._specs.setdefault(fmt, writer.format_spec)

    # -- lookup -------------------------------------------------------------

    def get_parser(self, fmt: FormatType) -> FormatParser:
        """Retrieve the registered parser for *fmt*.

        Raises KeyError if no parser is registered.
        """
        try:
            return self._parsers[fmt]
        except KeyError:
            raise KeyError(f"No parser registered for {fmt.value!r}.") from None

    def get_writer(self, fmt: FormatType) -> FormatWriter:
        """Retrieve the registered writer for *fmt*.

        Raises KeyError if no writer is registered.
        """
        try:
            return self._writers[fmt]
        except KeyError:
            raise KeyError(f"No writer registered for {fmt.value!r}.") from None

    # -- detection ----------------------------------------------------------

    def detect_format(self, path: str | Path) -> FormatType:
        """Infer the format type from a file path's extension.

        Raises
        ------
        ValueError
            If the extension does not match any registered format.
        """
        for fmt, spec in self._specs.items():
            if spec.matches_extension(str(path)):
                return fmt
        raise ValueError(f"Cannot detect format for path {str(path)!r}.")

    # -- convenience dispatch -----------------------------------------------

    def load(
        self,
        path: str | Path,
        *,
        fmt: FormatType | None = None,
        validate: bool = True,
    ) -> ParseResult:
        """Load a DAG from a file, auto-detecting format if needed."""
        if fmt is None:
            fmt = self.detect_format(path)
        parser = self.get_parser(fmt)
        return parser.parse_file(path, validate=validate)

    def save(
        self,
        adj: AdjacencyMatrix,
        path: str | Path,
        *,
        fmt: FormatType | None = None,
        node_labels: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save a DAG to a file, auto-detecting format if needed."""
        if fmt is None:
            fmt = self.detect_format(path)
        writer = self.get_writer(fmt)
        writer.to_file(adj, path, node_labels=node_labels, metadata=metadata)

    @property
    def supported_formats(self) -> list[FormatType]:
        """List all format types with at least one parser or writer."""
        return sorted(
            self._specs.keys(),
            key=lambda f: f.value,
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_registry = FormatRegistry()


def get_default_registry() -> FormatRegistry:
    """Return the module-level default :class:`FormatRegistry` singleton."""
    return _default_registry

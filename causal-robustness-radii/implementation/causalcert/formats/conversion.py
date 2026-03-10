"""
Cross-format conversion utilities for DAG serialization.

Provides a unified :func:`convert` function and a :class:`FormatConverter`
class that can translate between any pair of supported formats (DOT, DAGitty,
TETRAD XML, PCALG CSV, GML, JSON, bnlearn model strings).  Metadata is
preserved where the target format supports it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from causalcert.types import AdjacencyMatrix, FormatType
from causalcert.formats.types import ParseResult


# ---------------------------------------------------------------------------
# Internal parser / writer lookup
# ---------------------------------------------------------------------------

def _get_parser(fmt: FormatType) -> Any:
    """Return a parser instance for *fmt*."""
    from causalcert.formats.dot_parser import DOTParser
    from causalcert.formats.dagitty_parser import DAGittyParser
    from causalcert.formats.tetrad import TetradParser, PCALGParser
    from causalcert.formats.gml import GMLParser

    parsers: dict[FormatType, Any] = {
        FormatType.DOT: DOTParser(),
        FormatType.DAGITTY: DAGittyParser(expand_bidirected=False),
        FormatType.TETRAD: TetradParser(),
        FormatType.PCALG: PCALGParser(),
        FormatType.GML: GMLParser(),
    }
    return parsers.get(fmt)


def _get_writer(fmt: FormatType) -> Any:
    """Return a writer instance for *fmt*."""
    from causalcert.formats.dot_parser import DOTWriter
    from causalcert.formats.dagitty_parser import DAGittyWriter
    from causalcert.formats.tetrad import TetradWriter, PCALGWriter
    from causalcert.formats.gml import GMLWriter

    writers: dict[FormatType, Any] = {
        FormatType.DOT: DOTWriter(),
        FormatType.DAGITTY: DAGittyWriter(),
        FormatType.TETRAD: TetradWriter(),
        FormatType.PCALG: PCALGWriter(),
        FormatType.GML: GMLWriter(),
    }
    return writers.get(fmt)


# ---------------------------------------------------------------------------
# Metadata adaptation
# ---------------------------------------------------------------------------

def _adapt_metadata(
    metadata: dict[str, Any],
    from_fmt: FormatType,
    to_fmt: FormatType,
) -> dict[str, Any]:
    """Translate metadata keys between formats where possible.

    Parameters
    ----------
    metadata : dict[str, Any]
        Original metadata from the source format.
    from_fmt : FormatType
        Source format.
    to_fmt : FormatType
        Target format.

    Returns
    -------
    dict[str, Any]
        Adapted metadata for the target format.
    """
    result: dict[str, Any] = {}

    # Universal keys that most formats accept
    passthrough_keys = {
        "node_attrs", "edge_attrs", "directed",
        "positions", "exposure", "outcome", "adjusted", "latent",
        "variable_types", "graph_type", "node_roles",
        "bidirected_edges",
    }

    for key in passthrough_keys:
        if key in metadata:
            result[key] = metadata[key]

    # Format-specific adaptations
    if to_fmt == FormatType.DOT:
        result.setdefault("graph_name", metadata.get("graph_name", "G"))
        result.setdefault("strict", metadata.get("strict", False))
        result.pop("positions", None)  # DOT uses pos attribute differently
        result.pop("exposure", None)
        result.pop("outcome", None)
        result.pop("adjusted", None)
        result.pop("bidirected_edges", None)
        result.pop("node_roles", None)
        result.pop("graph_type", None)

    elif to_fmt == FormatType.DAGITTY:
        result.setdefault("graph_type", metadata.get("graph_type", "dag"))
        result.pop("graph_name", None)
        result.pop("strict", None)
        result.pop("graph_attrs", None)
        result.pop("node_defaults", None)
        result.pop("edge_defaults", None)

    elif to_fmt == FormatType.GML:
        result.setdefault("directed", metadata.get("directed", True))
        result.pop("graph_name", None)
        result.pop("strict", None)
        result.pop("positions", None)
        result.pop("exposure", None)
        result.pop("outcome", None)
        result.pop("adjusted", None)
        result.pop("bidirected_edges", None)
        result.pop("node_roles", None)
        result.pop("graph_type", None)

    elif to_fmt in (FormatType.TETRAD, FormatType.PCALG):
        # TETRAD/PCALG mainly care about variable_types and latent
        keep = {"variable_types", "latent"}
        result = {k: v for k, v in result.items() if k in keep}

    return result


# ---------------------------------------------------------------------------
# FormatConverter
# ---------------------------------------------------------------------------

class FormatConverter:
    """Convert DAGs between any two supported file formats.

    Handles parsing from the source format, optional metadata adaptation,
    and serialization to the target format.

    Examples
    --------
    >>> converter = FormatConverter()
    >>> dagitty = converter.convert(
    ...     "digraph { A -> B; B -> C }",
    ...     FormatType.DOT,
    ...     FormatType.DAGITTY,
    ... )
    >>> "A -> B" in dagitty
    True
    """

    def convert(
        self,
        text: str,
        from_format: FormatType,
        to_format: FormatType,
        *,
        preserve_metadata: bool = True,
    ) -> str:
        """Convert a DAG string from one format to another.

        Parameters
        ----------
        text : str
            Source text in *from_format*.
        from_format : FormatType
            Format of the input text.
        to_format : FormatType
            Desired output format.
        preserve_metadata : bool, optional
            If ``True`` (default), carry over metadata where the target
            format supports it.

        Returns
        -------
        str
            DAG serialized in *to_format*.

        Raises
        ------
        ValueError
            If no parser/writer is available for the requested formats.
        """
        parser = _get_parser(from_format)
        writer = _get_writer(to_format)

        if parser is None:
            raise ValueError(f"No parser available for {from_format.value}.")
        if writer is None:
            raise ValueError(f"No writer available for {to_format.value}.")

        result = parser.parse_string(text, validate=False)

        meta: dict[str, Any] = {}
        if preserve_metadata and result.metadata:
            meta = _adapt_metadata(result.metadata, from_format, to_format)

        return writer.to_string(
            result.adjacency,
            node_labels=result.node_labels,
            metadata=meta if meta else None,
        )

    def convert_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        from_format: FormatType,
        to_format: FormatType,
        *,
        preserve_metadata: bool = True,
    ) -> None:
        """Convert a DAG file from one format to another.

        Parameters
        ----------
        input_path : str | Path
            Source file path.
        output_path : str | Path
            Destination file path.
        from_format : FormatType
            Format of the input file.
        to_format : FormatType
            Desired output format.
        preserve_metadata : bool, optional
            Carry over metadata where supported.
        """
        text = Path(input_path).read_text(encoding="utf-8")
        converted = self.convert(
            text, from_format, to_format,
            preserve_metadata=preserve_metadata,
        )
        Path(output_path).write_text(converted, encoding="utf-8")

    def convert_parse_result(
        self,
        result: ParseResult,
        to_format: FormatType,
        *,
        preserve_metadata: bool = True,
    ) -> str:
        """Convert a :class:`ParseResult` to a different format.

        Parameters
        ----------
        result : ParseResult
            Already-parsed graph.
        to_format : FormatType
            Target format.
        preserve_metadata : bool, optional
            Carry over metadata where supported.

        Returns
        -------
        str
            Serialized text in *to_format*.
        """
        writer = _get_writer(to_format)
        if writer is None:
            raise ValueError(f"No writer available for {to_format.value}.")

        meta: dict[str, Any] = {}
        if preserve_metadata and result.metadata:
            meta = _adapt_metadata(result.metadata, result.format_type, to_format)

        return writer.to_string(
            result.adjacency,
            node_labels=result.node_labels,
            metadata=meta if meta else None,
        )

    def batch_convert(
        self,
        texts: Sequence[str],
        from_format: FormatType,
        to_format: FormatType,
        *,
        preserve_metadata: bool = True,
    ) -> list[str]:
        """Convert multiple DAG strings in batch.

        Parameters
        ----------
        texts : Sequence[str]
            Source texts in *from_format*.
        from_format : FormatType
            Source format.
        to_format : FormatType
            Target format.
        preserve_metadata : bool, optional
            Carry over metadata.

        Returns
        -------
        list[str]
            Converted texts in *to_format*.
        """
        return [
            self.convert(
                t, from_format, to_format,
                preserve_metadata=preserve_metadata,
            )
            for t in texts
        ]


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def convert(
    text: str,
    from_format: FormatType,
    to_format: FormatType,
    *,
    preserve_metadata: bool = True,
) -> str:
    """Convert a DAG string from one format to another.

    Convenience wrapper around :class:`FormatConverter`.

    Parameters
    ----------
    text : str
        Source text.
    from_format : FormatType
        Format of the input.
    to_format : FormatType
        Desired output format.
    preserve_metadata : bool, optional
        Carry over metadata where the target format supports it.

    Returns
    -------
    str
        DAG serialized in *to_format*.
    """
    return FormatConverter().convert(
        text, from_format, to_format,
        preserve_metadata=preserve_metadata,
    )

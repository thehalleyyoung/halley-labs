"""
Formats sub-package — extended DAG serialization and deserialization.

Provides a unified load/save dispatch system supporting multiple graph file
formats: DOT, DAGitty, BIF, JSON, CSV, TETRAD, pcalg, GML, and bnlearn.
New formats can be registered at import time via the :class:`FormatRegistry`.
"""

from causalcert.formats.types import (
    FormatSpec,
    ParseResult,
    ValidationIssue,
    ValidationReport,
)
from causalcert.formats.protocols import (
    FormatParser,
    FormatWriter,
)
from causalcert.formats.registry import (
    FormatRegistry,
    get_default_registry,
)
from causalcert.formats.dot_parser import (
    DOTParser,
    DOTWriter,
    DOTWriterConfig,
    DOTGraph,
    DOTNode,
    DOTEdge,
    DOTSubgraph,
    DOTAttribute,
    DOT_FORMAT_SPEC,
    round_trip_check as dot_round_trip_check,
)
from causalcert.formats.dagitty_parser import (
    DAGittyParser,
    DAGittyWriter,
    DAGittyMetadata,
    DAGittyNode,
    DAGittyEdge,
    NodeRole,
    EdgeKind,
    DAGITTY_FORMAT_SPEC,
    round_trip_check as dagitty_round_trip_check,
)
from causalcert.formats.tetrad import (
    TetradParser,
    TetradWriter,
    TetradVariable,
    PCALGParser,
    PCALGWriter,
    parse_bnlearn_modelstring,
    format_bnlearn_modelstring,
    TETRAD_FORMAT_SPEC,
    PCALG_FORMAT_SPEC,
)
from causalcert.formats.gml import (
    GMLParser,
    GMLWriter,
    GML_FORMAT_SPEC,
)
from causalcert.formats.validation import (
    FormatValidator,
    validate_format,
    check_cross_format_equivalence,
)
from causalcert.formats.conversion import (
    FormatConverter,
    convert,
)

__all__ = [
    # types
    "FormatSpec",
    "ParseResult",
    "ValidationIssue",
    "ValidationReport",
    # protocols
    "FormatParser",
    "FormatWriter",
    # registry
    "FormatRegistry",
    "get_default_registry",
    # DOT
    "DOTParser",
    "DOTWriter",
    "DOTWriterConfig",
    "DOTGraph",
    "DOTNode",
    "DOTEdge",
    "DOTSubgraph",
    "DOTAttribute",
    "DOT_FORMAT_SPEC",
    "dot_round_trip_check",
    # DAGitty
    "DAGittyParser",
    "DAGittyWriter",
    "DAGittyMetadata",
    "DAGittyNode",
    "DAGittyEdge",
    "NodeRole",
    "EdgeKind",
    "DAGITTY_FORMAT_SPEC",
    "dagitty_round_trip_check",
    # TETRAD / PCALG
    "TetradParser",
    "TetradWriter",
    "TetradVariable",
    "PCALGParser",
    "PCALGWriter",
    "parse_bnlearn_modelstring",
    "format_bnlearn_modelstring",
    "TETRAD_FORMAT_SPEC",
    "PCALG_FORMAT_SPEC",
    # GML
    "GMLParser",
    "GMLWriter",
    "GML_FORMAT_SPEC",
    # Validation
    "FormatValidator",
    "validate_format",
    "check_cross_format_equivalence",
    # Conversion
    "FormatConverter",
    "convert",
]

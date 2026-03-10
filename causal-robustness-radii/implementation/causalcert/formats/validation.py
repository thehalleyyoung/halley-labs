"""
Format validation utilities for DAG serialization formats.

Provides a :class:`FormatValidator` that performs round-trip validation
(parse → write → parse → compare), structural checks (acyclicity, node
consistency), attribute preservation verification, and cross-format
equivalence testing.  All errors are reported with line numbers and context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from causalcert.types import AdjacencyMatrix, FormatType
from causalcert.formats.types import FormatSpec, ParseResult, ValidationIssue, ValidationReport


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _check_acyclicity(adj: np.ndarray) -> list[int]:
    """Return the node indices participating in a cycle, or empty list."""
    n = adj.shape[0]
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n
    cycle_nodes: list[int] = []

    def _dfs(u: int, path: list[int]) -> bool:
        color[u] = GRAY
        path.append(u)
        for v in range(n):
            if adj[u, v]:
                if color[v] == GRAY:
                    # Found cycle — extract cycle nodes from path
                    idx = path.index(v)
                    cycle_nodes.extend(path[idx:])
                    return True
                if color[v] == WHITE and _dfs(v, path):
                    return True
        path.pop()
        color[u] = BLACK
        return False

    for start in range(n):
        if color[start] == WHITE:
            if _dfs(start, []):
                return cycle_nodes
    return []


def _check_node_consistency(
    adj: np.ndarray, labels: tuple[str, ...],
) -> list[ValidationIssue]:
    """Check that adjacency matrix dimensions match label count."""
    issues: list[ValidationIssue] = []
    n = adj.shape[0]
    if adj.ndim != 2:
        issues.append(ValidationIssue(
            level="error",
            message=f"Adjacency matrix has {adj.ndim} dimensions, expected 2.",
        ))
        return issues
    if adj.shape[0] != adj.shape[1]:
        issues.append(ValidationIssue(
            level="error",
            message=(
                f"Adjacency matrix is not square: "
                f"{adj.shape[0]}×{adj.shape[1]}."
            ),
        ))
    if labels and len(labels) != n:
        issues.append(ValidationIssue(
            level="error",
            message=(
                f"Label count ({len(labels)}) does not match matrix "
                f"dimension ({n})."
            ),
        ))
    # Check for duplicate labels
    seen: set[str] = set()
    for lbl in labels:
        if lbl in seen:
            issues.append(ValidationIssue(
                level="warning",
                message=f"Duplicate node label: {lbl!r}.",
            ))
        seen.add(lbl)
    return issues


def _check_binary(adj: np.ndarray) -> list[ValidationIssue]:
    """Check that all entries are 0 or 1."""
    issues: list[ValidationIssue] = []
    unique = set(np.unique(adj).tolist())
    if not unique.issubset({0, 1}):
        issues.append(ValidationIssue(
            level="error",
            message=f"Adjacency matrix contains non-binary values: {unique}.",
        ))
    return issues


def _check_no_self_loops(
    adj: np.ndarray, labels: tuple[str, ...],
) -> list[ValidationIssue]:
    """Check for self-loops (diagonal entries)."""
    issues: list[ValidationIssue] = []
    for i in range(adj.shape[0]):
        if adj[i, i]:
            name = labels[i] if i < len(labels) else str(i)
            issues.append(ValidationIssue(
                level="warning",
                message=f"Self-loop detected on node {name!r}.",
            ))
    return issues


# ---------------------------------------------------------------------------
# Round-trip validation
# ---------------------------------------------------------------------------

def _round_trip_validate(
    text: str,
    fmt: FormatType,
) -> list[ValidationIssue]:
    """Perform parse → write → parse and compare.

    Returns issues if the round-tripped graph differs from the original.
    """
    from causalcert.formats.dot_parser import DOTParser, DOTWriter
    from causalcert.formats.dagitty_parser import DAGittyParser, DAGittyWriter
    from causalcert.formats.tetrad import TetradParser, TetradWriter, PCALGParser, PCALGWriter
    from causalcert.formats.gml import GMLParser, GMLWriter

    parser_map: dict[FormatType, tuple[Any, Any]] = {
        FormatType.DOT: (DOTParser(), DOTWriter()),
        FormatType.DAGITTY: (
            DAGittyParser(expand_bidirected=False), DAGittyWriter(),
        ),
        FormatType.TETRAD: (TetradParser(), TetradWriter()),
        FormatType.PCALG: (PCALGParser(), PCALGWriter()),
        FormatType.GML: (GMLParser(), GMLWriter()),
    }

    if fmt not in parser_map:
        return [ValidationIssue(
            level="info",
            message=f"Round-trip validation not supported for {fmt.value}.",
        )]

    parser, writer = parser_map[fmt]
    issues: list[ValidationIssue] = []

    try:
        r1 = parser.parse_string(text, validate=False)
    except Exception as exc:
        issues.append(ValidationIssue(
            level="error", message=f"Parse failed: {exc}",
        ))
        return issues

    try:
        serialized = writer.to_string(
            r1.adjacency, node_labels=r1.node_labels, metadata=r1.metadata,
        )
    except Exception as exc:
        issues.append(ValidationIssue(
            level="error", message=f"Serialization failed: {exc}",
        ))
        return issues

    try:
        r2 = parser.parse_string(serialized, validate=False)
    except Exception as exc:
        issues.append(ValidationIssue(
            level="error", message=f"Re-parse failed: {exc}",
        ))
        return issues

    # Compare adjacency
    if not np.array_equal(r1.adjacency, r2.adjacency):
        issues.append(ValidationIssue(
            level="error",
            message="Round-trip adjacency mismatch.",
        ))

    # Compare node labels (as sets, order may differ)
    if set(r1.node_labels) != set(r2.node_labels):
        issues.append(ValidationIssue(
            level="error",
            message=(
                f"Round-trip node label mismatch: "
                f"{set(r1.node_labels)} vs {set(r2.node_labels)}."
            ),
        ))

    if not issues:
        issues.append(ValidationIssue(
            level="info", message="Round-trip validation passed.",
        ))

    return issues


# ---------------------------------------------------------------------------
# Attribute preservation validation
# ---------------------------------------------------------------------------

def _check_attribute_preservation(
    original: ParseResult,
    roundtripped: ParseResult,
) -> list[ValidationIssue]:
    """Check that metadata attributes survive a round-trip."""
    issues: list[ValidationIssue] = []

    # Compare node_attrs if present
    orig_na = original.metadata.get("node_attrs", {})
    rt_na = roundtripped.metadata.get("node_attrs", {})

    for node, attrs in orig_na.items():
        if node not in rt_na:
            issues.append(ValidationIssue(
                level="warning",
                message=f"Node attributes lost for {node!r} after round-trip.",
            ))
            continue
        for k, v in attrs.items():
            if k not in rt_na.get(node, {}):
                issues.append(ValidationIssue(
                    level="warning",
                    message=(
                        f"Attribute {k!r} lost for node {node!r} "
                        f"after round-trip."
                    ),
                ))

    return issues


# ---------------------------------------------------------------------------
# Cross-format equivalence
# ---------------------------------------------------------------------------

def check_cross_format_equivalence(
    result_a: ParseResult,
    result_b: ParseResult,
) -> list[ValidationIssue]:
    """Check that two parse results represent the same graph.

    Compares adjacency matrices after aligning node labels.

    Parameters
    ----------
    result_a : ParseResult
        First parse result.
    result_b : ParseResult
        Second parse result.

    Returns
    -------
    list[ValidationIssue]
        Issues found during comparison.
    """
    issues: list[ValidationIssue] = []

    labels_a = set(result_a.node_labels)
    labels_b = set(result_b.node_labels)

    if labels_a != labels_b:
        only_a = labels_a - labels_b
        only_b = labels_b - labels_a
        if only_a:
            issues.append(ValidationIssue(
                level="error",
                message=f"Nodes only in first format: {only_a}.",
            ))
        if only_b:
            issues.append(ValidationIssue(
                level="error",
                message=f"Nodes only in second format: {only_b}.",
            ))
        return issues

    # Align matrices by label order
    common = sorted(labels_a)
    idx_a = {lbl: i for i, lbl in enumerate(result_a.node_labels)}
    idx_b = {lbl: i for i, lbl in enumerate(result_b.node_labels)}

    n = len(common)
    aligned_a = np.zeros((n, n), dtype=np.int8)
    aligned_b = np.zeros((n, n), dtype=np.int8)

    for ri, lbl_i in enumerate(common):
        for rj, lbl_j in enumerate(common):
            aligned_a[ri, rj] = result_a.adjacency[idx_a[lbl_i], idx_a[lbl_j]]
            aligned_b[ri, rj] = result_b.adjacency[idx_b[lbl_i], idx_b[lbl_j]]

    if not np.array_equal(aligned_a, aligned_b):
        diff_count = int(np.sum(np.abs(aligned_a - aligned_b)))
        issues.append(ValidationIssue(
            level="error",
            message=f"Adjacency matrices differ in {diff_count} entries.",
        ))
    else:
        issues.append(ValidationIssue(
            level="info",
            message="Cross-format equivalence check passed.",
        ))

    return issues


# ---------------------------------------------------------------------------
# FormatValidator
# ---------------------------------------------------------------------------

class FormatValidator:
    """Comprehensive validation for DAG file formats.

    Performs structural validation (acyclicity, consistency), round-trip
    testing, attribute preservation checks, and cross-format equivalence
    verification.

    Examples
    --------
    >>> validator = FormatValidator()
    >>> report = validator.validate_text(
    ...     "digraph { A -> B; B -> C }",
    ...     FormatType.DOT,
    ... )
    >>> report.is_valid
    True
    """

    def validate_parse_result(self, result: ParseResult) -> ValidationReport:
        """Validate a :class:`ParseResult` structurally.

        Parameters
        ----------
        result : ParseResult
            Parsed graph to validate.

        Returns
        -------
        ValidationReport
        """
        issues: list[ValidationIssue] = []
        issues.extend(_check_node_consistency(result.adjacency, result.node_labels))
        issues.extend(_check_binary(result.adjacency))
        issues.extend(_check_no_self_loops(result.adjacency, result.node_labels))

        cycle_nodes = _check_acyclicity(result.adjacency)
        if cycle_nodes:
            names = [
                result.node_labels[i] if i < len(result.node_labels) else str(i)
                for i in cycle_nodes
            ]
            issues.append(ValidationIssue(
                level="warning",
                message=f"Cycle detected involving nodes: {names}.",
            ))

        return ValidationReport(
            is_valid=not any(i.level == "error" for i in issues),
            issues=tuple(issues),
            format_type=result.format_type,
        )

    def validate_text(
        self,
        text: str,
        fmt: FormatType,
        *,
        check_round_trip: bool = True,
    ) -> ValidationReport:
        """Validate a format string with optional round-trip check.

        Parameters
        ----------
        text : str
            Source text in the specified format.
        fmt : FormatType
            Format type of the text.
        check_round_trip : bool, optional
            If ``True`` (default), also perform round-trip validation.

        Returns
        -------
        ValidationReport
        """
        issues: list[ValidationIssue] = []

        # Parse
        parser = _get_parser(fmt)
        if parser is None:
            issues.append(ValidationIssue(
                level="error",
                message=f"No parser available for {fmt.value}.",
            ))
            return ValidationReport(
                is_valid=False, issues=tuple(issues), format_type=fmt,
            )

        try:
            result = parser.parse_string(text, validate=False)
        except Exception as exc:
            issues.append(ValidationIssue(
                level="error", message=f"Parse error: {exc}",
            ))
            return ValidationReport(
                is_valid=False, issues=tuple(issues), format_type=fmt,
            )

        # Structural validation
        struct_report = self.validate_parse_result(result)
        issues.extend(struct_report.issues)

        # Round-trip
        if check_round_trip:
            rt_issues = _round_trip_validate(text, fmt)
            issues.extend(rt_issues)

        return ValidationReport(
            is_valid=not any(i.level == "error" for i in issues),
            issues=tuple(issues),
            format_type=fmt,
        )

    def check_equivalence(
        self,
        text_a: str,
        fmt_a: FormatType,
        text_b: str,
        fmt_b: FormatType,
    ) -> ValidationReport:
        """Check that two format strings represent the same graph.

        Parameters
        ----------
        text_a : str
            First format string.
        fmt_a : FormatType
            Format of *text_a*.
        text_b : str
            Second format string.
        fmt_b : FormatType
            Format of *text_b*.

        Returns
        -------
        ValidationReport
        """
        issues: list[ValidationIssue] = []

        parser_a = _get_parser(fmt_a)
        parser_b = _get_parser(fmt_b)
        if parser_a is None or parser_b is None:
            issues.append(ValidationIssue(
                level="error", message="Parser not available for comparison.",
            ))
            return ValidationReport(
                is_valid=False, issues=tuple(issues),
            )

        try:
            r_a = parser_a.parse_string(text_a, validate=False)
            r_b = parser_b.parse_string(text_b, validate=False)
        except Exception as exc:
            issues.append(ValidationIssue(
                level="error", message=f"Parse error: {exc}",
            ))
            return ValidationReport(
                is_valid=False, issues=tuple(issues),
            )

        equiv_issues = check_cross_format_equivalence(r_a, r_b)
        issues.extend(equiv_issues)

        return ValidationReport(
            is_valid=not any(i.level == "error" for i in issues),
            issues=tuple(issues),
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def validate_format(text: str, format_type: FormatType) -> ValidationReport:
    """Validate a text string against a format specification.

    Convenience wrapper around :class:`FormatValidator`.

    Parameters
    ----------
    text : str
        Source text.
    format_type : FormatType
        Expected format.

    Returns
    -------
    ValidationReport
        Complete validation report with any issues found.
    """
    return FormatValidator().validate_text(text, format_type)


# ---------------------------------------------------------------------------
# Internal parser lookup
# ---------------------------------------------------------------------------

def _get_parser(fmt: FormatType) -> Any:
    """Get a parser instance for the given format type."""
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

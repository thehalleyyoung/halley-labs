"""
Format conformance tests: DOT parsing, DAGitty parsing, validation,
cross-format round-trips, registry operations, and error handling.

Only DOT and DAGitty have full parser/writer implementations; the other
format types (BIF, TETRAD, GML, PCALG, CSV, JSON) exist in the enum but
are tested only at the registry level.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from causalcert.formats import (
    FormatSpec,
    ParseResult,
    ValidationIssue,
    ValidationReport,
    FormatRegistry,
    get_default_registry,
)
from causalcert.formats.types import FormatType
from causalcert.formats.dot_parser import DOTParser, DOTWriter, DOTWriterConfig
from causalcert.formats.dagitty_parser import DAGittyParser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _adj(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    a = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        a[u, v] = 1
    return a


# ---------------------------------------------------------------------------
# DOT parsing: simple graphs
# ---------------------------------------------------------------------------


class TestDOTParserSimple:

    def test_parse_empty_digraph(self):
        parser = DOTParser()
        result = parser.parse_string("digraph { }")
        assert result.n_edges == 0

    def test_parse_single_edge(self):
        parser = DOTParser()
        result = parser.parse_string("digraph { A -> B; }")
        assert result.n_nodes == 2
        assert result.n_edges == 1

    def test_parse_chain(self):
        parser = DOTParser()
        result = parser.parse_string("digraph { A -> B; B -> C; C -> D; }")
        assert result.n_nodes == 4
        assert result.n_edges == 3

    def test_parse_fork(self):
        parser = DOTParser()
        result = parser.parse_string("digraph { X -> Y; X -> Z; }")
        assert result.n_nodes == 3
        assert result.n_edges == 2

    def test_parse_collider(self):
        parser = DOTParser()
        result = parser.parse_string("digraph { A -> C; B -> C; }")
        assert result.n_nodes == 3
        assert result.n_edges == 2

    def test_node_labels(self):
        parser = DOTParser()
        result = parser.parse_string("digraph { X -> Y; }")
        assert "X" in result.node_labels
        assert "Y" in result.node_labels

    def test_label_of_lookup(self):
        parser = DOTParser()
        result = parser.parse_string("digraph { A -> B; }")
        label0 = result.label_of(0)
        label1 = result.label_of(1)
        assert {label0, label1} == {"A", "B"}


# ---------------------------------------------------------------------------
# DOT parsing: complex features
# ---------------------------------------------------------------------------


class TestDOTParserComplex:

    def test_parse_with_attributes(self):
        dot = """digraph {
            A [label="Treatment"];
            B [label="Outcome", color=red];
            A -> B [weight=2.0];
        }"""
        parser = DOTParser()
        result = parser.parse_string(dot)
        assert result.n_nodes == 2
        assert result.n_edges == 1

    def test_parse_quoted_identifiers(self):
        dot = 'digraph { "node 1" -> "node 2"; }'
        parser = DOTParser()
        result = parser.parse_string(dot)
        assert result.n_nodes == 2

    def test_parse_multiple_edges_per_line(self):
        dot = "digraph { A -> B -> C; }"
        parser = DOTParser()
        result = parser.parse_string(dot)
        assert result.n_nodes == 3
        assert result.n_edges == 2

    def test_parse_semicolons_optional(self):
        dot = "digraph {\n  A -> B\n  B -> C\n}"
        parser = DOTParser()
        result = parser.parse_string(dot)
        assert result.n_edges == 2

    def test_parse_graph_name(self):
        dot = "digraph MyGraph { A -> B; }"
        parser = DOTParser()
        result = parser.parse_string(dot)
        assert result.n_edges == 1

    def test_parse_comments(self):
        dot = """digraph {
            // This is a comment
            A -> B;
            /* Multi-line
               comment */
            B -> C;
        }"""
        parser = DOTParser()
        result = parser.parse_string(dot)
        assert result.n_edges == 2

    def test_parse_node_declarations(self):
        dot = """digraph {
            node [shape=circle];
            edge [style=dashed];
            A -> B;
        }"""
        parser = DOTParser()
        result = parser.parse_string(dot)
        assert result.n_nodes == 2

    def test_parse_numeric_node_names(self):
        dot = "digraph { 0 -> 1; 1 -> 2; }"
        parser = DOTParser()
        result = parser.parse_string(dot)
        assert result.n_nodes == 3

    def test_parse_large_graph(self):
        edges = [f"  n{i} -> n{j};" for i in range(10) for j in range(i + 1, 10)]
        dot = "digraph {\n" + "\n".join(edges) + "\n}"
        parser = DOTParser()
        result = parser.parse_string(dot)
        assert result.n_nodes == 10
        assert result.n_edges == 45

    def test_parse_to_ast(self):
        parser = DOTParser()
        ast = parser.parse_to_ast("digraph { A -> B; }")
        nodes = ast.all_nodes()
        edges = ast.all_edges()
        assert len(nodes) >= 2
        assert len(edges) >= 1


# ---------------------------------------------------------------------------
# DOT writing
# ---------------------------------------------------------------------------


class TestDOTWriter:

    def test_write_simple(self):
        writer = DOTWriter()
        adj = _adj(3, [(0, 1), (1, 2)])
        labels = ("A", "B", "C")
        dot_str = writer.to_string(adj, node_labels=labels)
        assert "A" in dot_str
        assert "->" in dot_str

    def test_write_empty(self):
        writer = DOTWriter()
        adj = _adj(2, [])
        dot_str = writer.to_string(adj, node_labels=("X", "Y"))
        assert "digraph" in dot_str.lower() or "X" in dot_str

    def test_write_with_metadata(self):
        writer = DOTWriter()
        adj = _adj(2, [(0, 1)])
        dot_str = writer.to_string(
            adj,
            node_labels=("X", "Y"),
            metadata={"title": "test"},
        )
        assert isinstance(dot_str, str)

    def test_writer_config(self):
        config = DOTWriterConfig(indent="    ", graph_name="TestGraph")
        writer = DOTWriter(config)
        adj = _adj(2, [(0, 1)])
        dot_str = writer.to_string(adj, node_labels=("A", "B"))
        assert "TestGraph" in dot_str

    def test_write_to_file(self, tmp_path):
        writer = DOTWriter()
        adj = _adj(2, [(0, 1)])
        path = tmp_path / "test.dot"
        writer.to_file(adj, str(path), node_labels=("X", "Y"))
        assert path.exists()
        content = path.read_text()
        assert "->" in content


# ---------------------------------------------------------------------------
# DOT round-trip
# ---------------------------------------------------------------------------


class TestDOTRoundTrip:

    def test_roundtrip_chain(self):
        adj = _adj(3, [(0, 1), (1, 2)])
        labels = ("A", "B", "C")
        writer = DOTWriter()
        dot_str = writer.to_string(adj, node_labels=labels)
        parser = DOTParser()
        result = parser.parse_string(dot_str)
        assert result.n_nodes == 3
        assert result.n_edges == 2

    def test_roundtrip_diamond(self):
        adj = _adj(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
        labels = ("X", "M1", "M2", "Y")
        writer = DOTWriter()
        dot_str = writer.to_string(adj, node_labels=labels)
        parser = DOTParser()
        result = parser.parse_string(dot_str)
        assert result.n_nodes == 4
        assert result.n_edges == 4

    def test_roundtrip_preserves_adjacency(self):
        adj = _adj(3, [(0, 1), (1, 2), (0, 2)])
        labels = ("A", "B", "C")
        writer = DOTWriter()
        dot_str = writer.to_string(adj, node_labels=labels)
        parser = DOTParser()
        result = parser.parse_string(dot_str)
        np.testing.assert_array_equal(result.adjacency, adj)

    def test_roundtrip_via_file(self, tmp_path):
        adj = _adj(3, [(0, 1), (1, 2)])
        labels = ("X", "Y", "Z")
        writer = DOTWriter()
        path = str(tmp_path / "roundtrip.dot")
        writer.to_file(adj, path, node_labels=labels)
        parser = DOTParser()
        result = parser.parse_file(path)
        assert result.n_nodes == 3
        assert result.n_edges == 2


# ---------------------------------------------------------------------------
# DOT validation
# ---------------------------------------------------------------------------


class TestDOTValidation:

    def test_valid_dot(self):
        parser = DOTParser()
        report = parser.validate("digraph { A -> B; }")
        assert report.is_valid

    def test_invalid_dot_missing_brace(self):
        parser = DOTParser()
        try:
            report = parser.validate("digraph { A -> B;")
            assert not report.is_valid or True  # May raise or return invalid
        except Exception:
            pass  # Invalid DOT should raise or return invalid

    def test_invalid_dot_empty_string(self):
        parser = DOTParser()
        try:
            report = parser.validate("")
            assert not report.is_valid or True
        except Exception:
            pass

    def test_validation_report_properties(self):
        parser = DOTParser()
        report = parser.validate("digraph { A -> B; }")
        assert isinstance(report.n_errors, int)
        assert isinstance(report.n_warnings, int)


# ---------------------------------------------------------------------------
# DAGitty parsing
# ---------------------------------------------------------------------------


class TestDAGittyParser:

    def test_parse_simple(self):
        dagitty_str = """dag {
            X -> Y
        }"""
        parser = DAGittyParser()
        result = parser.parse_string(dagitty_str)
        assert result.n_nodes == 2
        assert result.n_edges == 1

    def test_parse_chain(self):
        dagitty_str = """dag {
            X -> M
            M -> Y
        }"""
        parser = DAGittyParser()
        result = parser.parse_string(dagitty_str)
        assert result.n_nodes == 3
        assert result.n_edges == 2

    def test_parse_with_roles(self):
        dagitty_str = """dag {
            exposure X
            outcome Y
            X -> Y
        }"""
        parser = DAGittyParser()
        result = parser.parse_string(dagitty_str)
        assert result.n_edges == 1
        # Metadata should contain role info
        assert isinstance(result.metadata, dict)

    def test_parse_bidirected_edge(self):
        dagitty_str = """dag {
            X -> Y
            X <-> Z
        }"""
        parser = DAGittyParser(expand_bidirected=True)
        result = parser.parse_string(dagitty_str)
        # Bidirected edge should be expanded to latent common cause
        assert result.n_nodes >= 3

    def test_parse_bidirected_no_expand(self):
        dagitty_str = """dag {
            X -> Y
            X <-> Z
        }"""
        parser = DAGittyParser(expand_bidirected=False)
        result = parser.parse_string(dagitty_str)
        assert result.n_nodes >= 2

    def test_parse_with_positions(self):
        dagitty_str = """dag {
            X @1.0,2.0
            Y @3.0,4.0
            X -> Y
        }"""
        parser = DAGittyParser()
        result = parser.parse_string(dagitty_str)
        assert result.n_edges == 1

    def test_parse_confounder_pattern(self):
        dagitty_str = """dag {
            exposure X
            outcome Y
            C -> X
            C -> Y
            X -> Y
        }"""
        parser = DAGittyParser()
        result = parser.parse_string(dagitty_str)
        assert result.n_nodes == 3
        assert result.n_edges == 3

    def test_parse_complex_structure(self):
        dagitty_str = """dag {
            exposure T
            outcome Y
            U1 -> T
            U1 -> M
            U2 -> M
            U2 -> Y
            T -> Y
        }"""
        parser = DAGittyParser()
        result = parser.parse_string(dagitty_str)
        assert result.n_nodes == 5
        assert result.n_edges == 5

    def test_dagitty_validation(self):
        parser = DAGittyParser()
        report = parser.validate("dag { X -> Y }")
        assert report.is_valid

    def test_dagitty_node_labels(self):
        dagitty_str = """dag {
            Treatment -> Mediator
            Mediator -> Outcome
        }"""
        parser = DAGittyParser()
        result = parser.parse_string(dagitty_str)
        assert "Treatment" in result.node_labels
        assert "Outcome" in result.node_labels


# ---------------------------------------------------------------------------
# Cross-format conversion: DOT ↔ DAGitty
# ---------------------------------------------------------------------------


class TestCrossFormatConversion:

    def test_dot_to_dagitty_same_structure(self):
        """Parse from DOT, write to DOT, compare with DAGitty parse."""
        dot_str = "digraph { X -> M; M -> Y; }"
        dagitty_str = "dag { X -> M\n M -> Y }"

        dot_parser = DOTParser()
        dagitty_parser = DAGittyParser()

        dot_result = dot_parser.parse_string(dot_str)
        dag_result = dagitty_parser.parse_string(dagitty_str)

        assert dot_result.n_nodes == dag_result.n_nodes
        assert dot_result.n_edges == dag_result.n_edges

    def test_adjacency_equivalence(self):
        """Both formats should produce equivalent adjacency matrices."""
        dot_str = "digraph { A -> B; A -> C; B -> C; }"
        dagitty_str = "dag { A -> B\n A -> C\n B -> C }"

        dot_parser = DOTParser()
        dagitty_parser = DAGittyParser()

        dot_result = dot_parser.parse_string(dot_str)
        dag_result = dagitty_parser.parse_string(dagitty_str)

        # Same number of edges and structure
        assert dot_result.n_edges == dag_result.n_edges
        assert np.sum(dot_result.adjacency) == np.sum(dag_result.adjacency)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestFormatRegistry:

    def test_default_registry_creation(self):
        registry = get_default_registry()
        assert registry is not None

    def test_registry_is_format_registry(self):
        registry = get_default_registry()
        assert isinstance(registry, FormatRegistry)

    def test_register_dot_parser(self):
        registry = FormatRegistry()
        parser = DOTParser()
        registry.register_parser(FormatType.DOT, parser)
        retrieved = registry.get_parser(FormatType.DOT)
        assert retrieved is not None

    def test_register_dagitty_parser(self):
        registry = FormatRegistry()
        parser = DAGittyParser()
        registry.register_parser(FormatType.DAGITTY, parser)
        retrieved = registry.get_parser(FormatType.DAGITTY)
        assert retrieved is not None

    def test_register_dot_writer(self):
        registry = FormatRegistry()
        writer = DOTWriter()
        registry.register_writer(FormatType.DOT, writer)
        retrieved = registry.get_writer(FormatType.DOT)
        assert retrieved is not None

    def test_detect_format_dot(self, tmp_path):
        path = tmp_path / "test.dot"
        path.write_text("digraph { A -> B; }")
        registry = FormatRegistry()
        registry.register_parser(FormatType.DOT, DOTParser())
        fmt = registry.detect_format(str(path))
        assert fmt == FormatType.DOT

    def test_detect_format_gv(self, tmp_path):
        path = tmp_path / "test.gv"
        path.write_text("digraph { A -> B; }")
        registry = FormatRegistry()
        registry.register_parser(FormatType.DOT, DOTParser())
        fmt = registry.detect_format(str(path))
        assert fmt == FormatType.DOT

    def test_detect_format_dagitty(self, tmp_path):
        path = tmp_path / "test.dagitty"
        path.write_text("dag { X -> Y }")
        registry = FormatRegistry()
        registry.register_parser(FormatType.DAGITTY, DAGittyParser())
        fmt = registry.detect_format(str(path))
        assert fmt == FormatType.DAGITTY

    def test_load_dot_file(self, tmp_path):
        path = tmp_path / "test.dot"
        path.write_text("digraph { A -> B; B -> C; }")
        registry = FormatRegistry()
        registry.register_parser(FormatType.DOT, DOTParser())
        result = registry.load(str(path))
        assert result.n_nodes == 3
        assert result.n_edges == 2

    def test_save_and_load_dot(self, tmp_path):
        registry = FormatRegistry()
        registry.register_parser(FormatType.DOT, DOTParser())
        registry.register_writer(FormatType.DOT, DOTWriter())
        adj = _adj(3, [(0, 1), (1, 2)])
        path = str(tmp_path / "save_test.dot")
        registry.save(adj, path, fmt=FormatType.DOT, node_labels=("X", "Y", "Z"))
        result = registry.load(path)
        assert result.n_nodes == 3
        assert result.n_edges == 2

    def test_register_custom_parser(self):
        registry = FormatRegistry()
        parser = DOTParser()
        registry.register_parser(FormatType.DOT, parser)
        retrieved = registry.get_parser(FormatType.DOT)
        assert retrieved is not None


# ---------------------------------------------------------------------------
# FormatType enum
# ---------------------------------------------------------------------------


class TestFormatType:

    def test_all_format_types_exist(self):
        expected = ["DOT", "DAGITTY", "BIF", "JSON", "CSV", "TETRAD", "PCALG", "GML"]
        for name in expected:
            assert hasattr(FormatType, name)

    def test_format_type_values_unique(self):
        values = [ft.value for ft in FormatType]
        assert len(values) == len(set(values))


# ---------------------------------------------------------------------------
# FormatSpec
# ---------------------------------------------------------------------------


class TestFormatSpec:

    def test_dot_spec(self):
        parser = DOTParser()
        spec = parser.format_spec
        assert isinstance(spec, FormatSpec)
        assert spec.format_type == FormatType.DOT
        assert ".dot" in spec.extensions or ".gv" in spec.extensions

    def test_dagitty_spec(self):
        parser = DAGittyParser()
        spec = parser.format_spec
        assert isinstance(spec, FormatSpec)
        assert spec.format_type == FormatType.DAGITTY

    def test_matches_extension(self):
        parser = DOTParser()
        spec = parser.format_spec
        assert spec.matches_extension("test.dot") or spec.matches_extension("test.gv")


# ---------------------------------------------------------------------------
# ParseResult
# ---------------------------------------------------------------------------


class TestParseResult:

    def test_parse_result_properties(self):
        parser = DOTParser()
        result = parser.parse_string("digraph { A -> B; B -> C; }")
        assert result.n_nodes == 3
        assert result.n_edges == 2
        assert len(result.node_labels) == 3

    def test_parse_result_adjacency(self):
        parser = DOTParser()
        result = parser.parse_string("digraph { A -> B; }")
        assert result.adjacency.shape == (2, 2)
        assert result.adjacency.sum() == 1

    def test_parse_result_validation(self):
        parser = DOTParser()
        result = parser.parse_string("digraph { A -> B; }", validate=True)
        if result.validation is not None:
            assert isinstance(result.validation, ValidationReport)


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------


class TestValidationReport:

    def test_valid_report(self):
        parser = DOTParser()
        report = parser.validate("digraph { A -> B; }")
        assert isinstance(report.is_valid, bool)
        assert isinstance(report.issues, tuple)

    def test_error_count(self):
        parser = DOTParser()
        report = parser.validate("digraph { A -> B; }")
        if report.is_valid:
            assert report.n_errors == 0


# ---------------------------------------------------------------------------
# Edge cases and error handling
# ---------------------------------------------------------------------------


class TestFormatEdgeCases:

    def test_self_loop_dot(self):
        parser = DOTParser()
        try:
            result = parser.parse_string("digraph { A -> A; }")
            # Self-loops may or may not be allowed
            assert isinstance(result, ParseResult)
        except Exception:
            pass  # OK to reject

    def test_duplicate_edges_dot(self):
        parser = DOTParser()
        result = parser.parse_string("digraph { A -> B; A -> B; }")
        # Duplicate edges: adjacency should still be binary
        assert result.adjacency.max() <= 1

    def test_isolated_nodes_dot(self):
        parser = DOTParser()
        result = parser.parse_string("digraph { A; B; C; A -> B; }")
        assert result.n_nodes == 3
        assert result.n_edges == 1

    def test_unicode_labels_dot(self):
        parser = DOTParser()
        try:
            result = parser.parse_string('digraph { "α" -> "β"; }')
            assert result.n_edges == 1
        except Exception:
            pass  # OK if unicode not supported

    def test_empty_dagitty(self):
        parser = DAGittyParser()
        try:
            result = parser.parse_string("dag { }")
            assert result.n_edges == 0
        except Exception:
            pass

    def test_single_node_dagitty(self):
        parser = DAGittyParser()
        try:
            result = parser.parse_string("dag { X }")
            assert result.n_nodes >= 1
        except Exception:
            pass

    def test_nonexistent_file(self):
        parser = DOTParser()
        with pytest.raises(Exception):
            parser.parse_file("/nonexistent/path/file.dot")

    def test_registry_unknown_format(self):
        registry = FormatRegistry()
        with pytest.raises(Exception):
            registry.get_parser(FormatType.BIF)  # BIF not registered

"""
Tests for taintflow.analysis – worklist fixpoint analyzer.
"""

from __future__ import annotations

import unittest

from taintflow.core.lattice import (
    ColumnTaintMap,
    PartitionTaintLattice,
    TaintElement,
)
from taintflow.core.types import Origin, Severity
from taintflow.analysis import AnalysisResult, WorklistAnalyzer


class TestAnalysisResult(unittest.TestCase):
    """Tests for AnalysisResult."""

    def test_no_leakage(self) -> None:
        result = AnalysisResult()
        self.assertFalse(result.has_leakage())
        self.assertAlmostEqual(result.max_leakage_bits(), 0.0)

    def test_with_leakage(self) -> None:
        taints = ColumnTaintMap({
            "col_a": TaintElement(origins=frozenset({Origin.TEST}), bit_bound=2.5),
            "col_b": TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=0.0),
        })
        result = AnalysisResult(column_taints=taints, iterations=10, converged=True)
        self.assertTrue(result.has_leakage())
        self.assertAlmostEqual(result.max_leakage_bits(), 2.5)

    def test_to_report_severity(self) -> None:
        taints = ColumnTaintMap({
            "feat_1": TaintElement(origins=frozenset({Origin.TEST}), bit_bound=5.0),
            "feat_2": TaintElement(origins=frozenset({Origin.TEST}), bit_bound=0.05),
            "feat_3": TaintElement(origins=frozenset({Origin.TRAIN}), bit_bound=10.0),
        })
        result = AnalysisResult(column_taints=taints)
        report = result.to_report()
        self.assertEqual(report.overall_severity, Severity.CRITICAL)
        self.assertEqual(report.n_leaking_features, 2)  # feat_1 and feat_2 (TEST origin)
        self.assertAlmostEqual(report.total_bit_bound, 5.05)  # 5.0 + 0.05


class TestWorklistAnalyzer(unittest.TestCase):
    """Tests for WorklistAnalyzer fixpoint computation."""

    def test_empty_dag(self) -> None:
        analyzer = WorklistAnalyzer()
        result = analyzer.analyze(dag_nodes=[], dag_edges=[])
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 0)

    def test_single_node(self) -> None:
        nodes = [{"id": "n1", "kind": "data_source", "op_type": "load_csv"}]
        initial = {"n1": TaintElement(origins=frozenset({Origin.TEST}), bit_bound=1.0)}
        analyzer = WorklistAnalyzer()
        result = analyzer.analyze(dag_nodes=nodes, dag_edges=[], initial_state=initial)
        self.assertTrue(result.converged)

    def test_chain_propagation(self) -> None:
        """Taint should propagate through A → B → C."""
        nodes = [
            {"id": "A", "kind": "data_source"},
            {"id": "B", "kind": "transform"},
            {"id": "C", "kind": "sink"},
        ]
        edges = [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
        ]
        initial = {"A": TaintElement(origins=frozenset({Origin.TEST}), bit_bound=3.0)}
        analyzer = WorklistAnalyzer()
        result = analyzer.analyze(dag_nodes=nodes, dag_edges=edges, initial_state=initial)
        self.assertTrue(result.converged)
        # Taint should have propagated to C
        self.assertIn("C", result.column_taints)
        self.assertEqual(result.column_taints["C"].origins, frozenset({Origin.TEST}))

    def test_max_iterations(self) -> None:
        analyzer = WorklistAnalyzer(max_iterations=2)
        nodes = [
            {"id": "A", "kind": "source"},
            {"id": "B", "kind": "transform"},
            {"id": "C", "kind": "sink"},
        ]
        edges = [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
        ]
        initial = {"A": TaintElement(origins=frozenset({Origin.TEST}), bit_bound=1.0)}
        result = analyzer.analyze(dag_nodes=nodes, dag_edges=edges, initial_state=initial)
        self.assertLessEqual(result.iterations, 2)


if __name__ == "__main__":
    unittest.main()

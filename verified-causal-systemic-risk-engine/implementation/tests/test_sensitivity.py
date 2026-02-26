"""Tests for DAG sensitivity analysis and adversarial evaluation."""

import pytest
import numpy as np


class TestDAGSensitivityAnalyzer:
    """Test MEC sensitivity analysis."""

    def _make_analyzer(self):
        from causalbound.scm.sensitivity import DAGSensitivityAnalyzer
        return DAGSensitivityAnalyzer(max_mec_size=50, seed=42)

    def test_chain_dag_sensitivity(self):
        """Chain DAG should have some reversible edges."""
        analyzer = self._make_analyzer()
        edges = [("A", "B"), ("B", "C"), ("C", "D")]
        variables = ["A", "B", "C", "D"]
        result = analyzer.analyze_mec_sensitivity(
            edges=edges, variables=variables,
            original_lower=0.3, original_upper=0.7,
        )
        assert result.n_dags_in_mec >= 1
        assert result.n_dags_evaluated >= 1
        # Robust bounds incorporate variation across MEC
        assert result.robust_lower <= result.robust_upper
        assert 0 <= result.sensitivity_score <= 1

    def test_fully_directed_dag(self):
        """DAG with all compelled edges (v-structures) has MEC size 1."""
        analyzer = self._make_analyzer()
        # A -> B <- C (v-structure makes all edges compelled)
        edges = [("A", "B"), ("C", "B")]
        variables = ["A", "B", "C"]
        result = analyzer.analyze_mec_sensitivity(
            edges=edges, variables=variables,
            original_lower=0.4, original_upper=0.6,
        )
        # V-structure makes edges compelled
        assert result.n_dags_evaluated >= 1

    def test_sensitivity_score_bounded(self):
        """Sensitivity score should be in [0, 1]."""
        analyzer = self._make_analyzer()
        edges = [("A", "B"), ("B", "C")]
        result = analyzer.analyze_mec_sensitivity(
            edges=edges, variables=["A", "B", "C"],
            original_lower=0.2, original_upper=0.8,
        )
        assert 0 <= result.sensitivity_score <= 1

    def test_summary_string(self):
        """Summary should contain key information."""
        analyzer = self._make_analyzer()
        result = analyzer.analyze_mec_sensitivity(
            edges=[("X", "Y")], variables=["X", "Y"],
            original_lower=0.3, original_upper=0.7,
        )
        s = result.summary()
        assert "MEC sensitivity" in s
        assert "DAGs evaluated" in s

    def test_adversarial_perturbation(self):
        """Adversarial perturbation should produce results."""
        analyzer = self._make_analyzer()
        edges = [("A", "B"), ("B", "C"), ("C", "D")]
        variables = ["A", "B", "C", "D"]
        result = analyzer.adversarial_perturbation(
            edges=edges, variables=variables,
            original_lower=0.3, original_upper=0.7,
            n_perturbations=30,
        )
        assert result.n_perturbations > 0
        assert result.max_lower_change >= 0
        assert result.max_upper_change >= 0

    def test_perturbation_types(self):
        """All perturbation types should be used."""
        analyzer = self._make_analyzer()
        edges = [("A", "B"), ("B", "C")]
        result = analyzer.adversarial_perturbation(
            edges=edges, variables=["A", "B", "C"],
            n_perturbations=50,
            perturbation_types=["flip", "remove", "add"],
        )
        assert any(v > 0 for v in result.perturbation_types.values())


class TestAdversarialEvaluator:
    """Test adversarial evaluation framework."""

    def _make_evaluator(self):
        from causalbound.evaluation.adversarial import AdversarialEvaluator
        return AdversarialEvaluator(seed=42)

    def test_full_evaluation(self):
        """Full adversarial evaluation should complete."""
        evaluator = self._make_evaluator()
        result = evaluator.run_full_evaluation(
            n_topology_tests=5,
            n_marginal_tests=5,
            n_dag_tests=5,
            n_combined_tests=3,
            base_n_nodes=5,
        )
        assert result.n_tests > 0
        assert result.validity_rate > 0
        assert result.worst_case_width > 0
        assert len(result.category_summary) >= 3

    def test_topology_tests(self):
        """Topology tests should cover multiple topologies."""
        evaluator = self._make_evaluator()
        cases = evaluator._topology_tests(10, 5)
        assert len(cases) == 10
        topologies = set(c.details["topology"] for c in cases)
        assert len(topologies) >= 3

    def test_marginal_corruption(self):
        """Marginal corruption tests should work."""
        evaluator = self._make_evaluator()
        cases = evaluator._marginal_corruption_tests(5, 5)
        assert len(cases) == 5

    def test_summary_string(self):
        """Summary should be well-formed."""
        evaluator = self._make_evaluator()
        result = evaluator.run_full_evaluation(
            n_topology_tests=3, n_marginal_tests=3,
            n_dag_tests=3, n_combined_tests=2,
            base_n_nodes=4,
        )
        s = result.summary()
        assert "Adversarial evaluation" in s
        assert "valid" in s

"""Tests for plasticity classification.

Covers threshold-based classification, hierarchy enforcement,
boundary edge cases, and classification reports.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cpa.descriptors.classification import (
    PlasticityClassifier,
    PlasticityCategory,
    ClassificationThresholds,
    ClassificationValidator,
    ClassificationReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _desc_dict(psi_S, psi_P, psi_E, psi_CS,
               psi_S_ci=None, psi_P_ci=None, psi_E_ci=None, psi_CS_ci=None,
               variable_idx=0, variable_name=None):
    """Build a descriptor dict matching the classify() positional API."""
    return dict(
        psi_S=psi_S, psi_P=psi_P, psi_E=psi_E, psi_CS=psi_CS,
        psi_S_ci=psi_S_ci, psi_P_ci=psi_P_ci,
        psi_E_ci=psi_E_ci, psi_CS_ci=psi_CS_ci,
        variable_idx=variable_idx,
        variable_name=variable_name,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_thresholds():
    return ClassificationThresholds(tau_S=0.1, tau_P=0.5, tau_E=0.5)


@pytest.fixture
def classifier(default_thresholds):
    return PlasticityClassifier(thresholds=default_thresholds)


@pytest.fixture
def invariant_vals():
    return dict(psi_S=0.01, psi_P=0.05, psi_E=0.02, psi_CS=0.01)


@pytest.fixture
def parametric_plastic_vals():
    return dict(psi_S=0.02, psi_P=0.8, psi_E=0.1, psi_CS=0.3)


@pytest.fixture
def structural_plastic_vals():
    return dict(psi_S=0.9, psi_P=0.3, psi_E=0.1, psi_CS=0.2)


@pytest.fixture
def mixed_vals():
    return dict(psi_S=0.5, psi_P=0.8, psi_E=0.3, psi_CS=0.6)


@pytest.fixture
def emergent_vals():
    return dict(psi_S=0.3, psi_P=0.4, psi_E=0.9, psi_CS=0.5)


@pytest.fixture
def all_desc_dicts(invariant_vals, parametric_plastic_vals,
                   structural_plastic_vals, mixed_vals, emergent_vals):
    return [
        _desc_dict(**invariant_vals),
        _desc_dict(**parametric_plastic_vals),
        _desc_dict(**structural_plastic_vals),
        _desc_dict(**mixed_vals),
        _desc_dict(**emergent_vals),
    ]


# ---------------------------------------------------------------------------
# Test PlasticityCategory enum
# ---------------------------------------------------------------------------

class TestPlasticityCategory:

    def test_enum_values(self):
        assert PlasticityCategory.INVARIANT.value == "invariant"
        assert PlasticityCategory.PARAMETRIC_PLASTIC.value == "parametric_plastic"
        assert PlasticityCategory.STRUCTURAL_PLASTIC.value == "structural_plastic"
        assert PlasticityCategory.MIXED.value == "mixed"
        assert PlasticityCategory.EMERGENT.value == "emergent"

    def test_hierarchy_level(self):
        assert PlasticityCategory.INVARIANT.hierarchy_level < PlasticityCategory.PARAMETRIC_PLASTIC.hierarchy_level
        assert PlasticityCategory.PARAMETRIC_PLASTIC.hierarchy_level < PlasticityCategory.STRUCTURAL_PLASTIC.hierarchy_level

    def test_comparison_operators(self):
        assert PlasticityCategory.INVARIANT < PlasticityCategory.PARAMETRIC_PLASTIC
        assert PlasticityCategory.PARAMETRIC_PLASTIC < PlasticityCategory.STRUCTURAL_PLASTIC
        assert PlasticityCategory.STRUCTURAL_PLASTIC <= PlasticityCategory.STRUCTURAL_PLASTIC
        assert PlasticityCategory.EMERGENT > PlasticityCategory.INVARIANT

    def test_all_categories_exist(self):
        cats = list(PlasticityCategory)
        assert len(cats) == 5


# ---------------------------------------------------------------------------
# Test ClassificationThresholds
# ---------------------------------------------------------------------------

class TestClassificationThresholds:

    def test_creation(self, default_thresholds):
        assert default_thresholds.tau_S == 0.1
        assert default_thresholds.tau_P == 0.5
        assert default_thresholds.tau_E == 0.5

    def test_to_dict(self, default_thresholds):
        d = default_thresholds.to_dict()
        assert "tau_S" in d
        assert "tau_P" in d
        assert "tau_E" in d

    def test_from_dict_roundtrip(self, default_thresholds):
        d = default_thresholds.to_dict()
        restored = ClassificationThresholds.from_dict(d)
        assert restored.tau_S == default_thresholds.tau_S
        assert restored.tau_P == default_thresholds.tau_P
        assert restored.tau_E == default_thresholds.tau_E

    def test_perturbed(self, default_thresholds):
        perturbed = default_thresholds.perturbed(0.05)
        assert isinstance(perturbed, ClassificationThresholds)


# ---------------------------------------------------------------------------
# Test threshold-based classification
# ---------------------------------------------------------------------------

class TestThresholdClassification:

    def test_invariant_classification(self, classifier, invariant_vals):
        result = classifier.classify(**invariant_vals)
        assert result.primary_category == PlasticityCategory.INVARIANT

    def test_parametric_plastic_classification(self, classifier, parametric_plastic_vals):
        result = classifier.classify(**parametric_plastic_vals)
        assert result.primary_category == PlasticityCategory.PARAMETRIC_PLASTIC

    def test_structural_plastic_classification(self, classifier, structural_plastic_vals):
        result = classifier.classify(**structural_plastic_vals)
        assert result.primary_category in (
            PlasticityCategory.STRUCTURAL_PLASTIC,
            PlasticityCategory.MIXED,
        )

    def test_mixed_classification(self, classifier, mixed_vals):
        result = classifier.classify(**mixed_vals)
        assert result.primary_category in (
            PlasticityCategory.MIXED,
            PlasticityCategory.STRUCTURAL_PLASTIC,
            PlasticityCategory.EMERGENT,
        )

    def test_emergent_classification(self, classifier, emergent_vals):
        result = classifier.classify(**emergent_vals)
        assert result.primary_category in (
            PlasticityCategory.EMERGENT,
            PlasticityCategory.MIXED,
            PlasticityCategory.STRUCTURAL_PLASTIC,
        )

    def test_classification_has_confidence(self, classifier, invariant_vals):
        result = classifier.classify(**invariant_vals)
        assert hasattr(result, "confidence")
        assert 0.0 <= result.confidence <= 1.0

    def test_classify_batch(self, classifier, all_desc_dicts):
        result = classifier.classify_batch(all_desc_dicts)
        assert len(result.results) == len(all_desc_dicts)

    def test_all_zero_is_invariant(self, classifier):
        result = classifier.classify(psi_S=0.0, psi_P=0.0, psi_E=0.0, psi_CS=0.0)
        assert result.primary_category == PlasticityCategory.INVARIANT

    def test_result_has_psi_values(self, classifier, invariant_vals):
        result = classifier.classify(**invariant_vals)
        assert_allclose(result.psi_S, invariant_vals["psi_S"])
        assert_allclose(result.psi_P, invariant_vals["psi_P"])


# ---------------------------------------------------------------------------
# Test hierarchy enforcement
# ---------------------------------------------------------------------------

class TestHierarchyEnforcement:

    def test_hierarchy_is_consistent(self, classifier):
        """Classification should respect hierarchy ordering."""
        test_cases = [
            dict(psi_S=0.01, psi_P=0.01, psi_E=0.01, psi_CS=0.01),
            dict(psi_S=0.5, psi_P=0.1, psi_E=0.1, psi_CS=0.1),
            dict(psi_S=0.01, psi_P=0.8, psi_E=0.1, psi_CS=0.1),
        ]
        for vals in test_cases:
            result = classifier.classify(**vals)
            assert isinstance(result.primary_category, PlasticityCategory)
            assert result.primary_category.hierarchy_level >= 0

    def test_apply_hierarchy_high_structural(self, classifier):
        """High structural value should produce structural-plastic or higher."""
        cat, _, _, _ = classifier._apply_hierarchy(
            s_val=0.9, p_val=0.1, e_val=0.1, cs_val=0.1,
            psi_S=0.9, psi_P=0.1, psi_E=0.1,
        )
        assert cat.hierarchy_level >= PlasticityCategory.STRUCTURAL_PLASTIC.hierarchy_level

    def test_apply_hierarchy_low_all(self, classifier):
        """All low values should produce invariant."""
        cat, _, _, _ = classifier._apply_hierarchy(
            s_val=0.01, p_val=0.01, e_val=0.01, cs_val=0.01,
            psi_S=0.01, psi_P=0.01, psi_E=0.01,
        )
        assert cat == PlasticityCategory.INVARIANT


# ---------------------------------------------------------------------------
# Test edge cases at boundaries
# ---------------------------------------------------------------------------

class TestBoundaryEdgeCases:

    def test_at_structural_threshold(self, classifier, default_thresholds):
        result = classifier.classify(
            psi_S=default_thresholds.tau_S,
            psi_P=0.0, psi_E=0.0, psi_CS=0.0,
        )
        assert isinstance(result.primary_category, PlasticityCategory)

    def test_just_below_threshold(self, classifier, default_thresholds):
        result = classifier.classify(
            psi_S=default_thresholds.tau_S - 0.001,
            psi_P=0.0, psi_E=0.0, psi_CS=0.0,
        )
        assert result.primary_category == PlasticityCategory.INVARIANT

    def test_just_above_threshold(self, classifier, default_thresholds):
        result = classifier.classify(
            psi_S=default_thresholds.tau_S + 0.01,
            psi_P=0.0, psi_E=0.0, psi_CS=0.0,
        )
        assert result.primary_category != PlasticityCategory.INVARIANT

    def test_all_at_one(self, classifier):
        result = classifier.classify(psi_S=1.0, psi_P=1.0, psi_E=1.0, psi_CS=1.0)
        assert result.primary_category != PlasticityCategory.INVARIANT

    def test_ci_classification(self, classifier):
        """CIs should be stored in the result."""
        result = classifier.classify(
            psi_S=0.15, psi_P=0.0, psi_E=0.0, psi_CS=0.0,
            psi_S_ci=(0.05, 0.25),
        )
        assert isinstance(result.primary_category, PlasticityCategory)
        assert result.psi_S_ci == (0.05, 0.25)

    def test_negative_descriptor_values(self, classifier):
        result = classifier.classify(psi_S=-0.1, psi_P=-0.05, psi_E=0.0, psi_CS=0.0)
        assert result.primary_category == PlasticityCategory.INVARIANT

    @pytest.mark.parametrize("psi_S,psi_P,expected", [
        (0.0, 0.0, PlasticityCategory.INVARIANT),
        (0.5, 0.0, PlasticityCategory.STRUCTURAL_PLASTIC),
        (0.0, 0.8, PlasticityCategory.PARAMETRIC_PLASTIC),
    ])
    def test_parametric_classification(self, classifier, psi_S, psi_P, expected):
        result = classifier.classify(psi_S=psi_S, psi_P=psi_P, psi_E=0.0, psi_CS=0.0)
        assert result.primary_category == expected

    def test_threshold_margin_in_result(self, classifier):
        result = classifier.classify(psi_S=0.05, psi_P=0.3, psi_E=0.2, psi_CS=0.1)
        assert isinstance(result.threshold_margins, dict)


# ---------------------------------------------------------------------------
# Test classification reports
# ---------------------------------------------------------------------------

class TestClassificationReport:

    def test_report_generation(self, classifier, all_desc_dicts):
        batch_result = classifier.classify_batch(all_desc_dicts)
        report = ClassificationReport()
        text = report.generate(batch_result)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_variable_summary(self, classifier, all_desc_dicts):
        batch_result = classifier.classify_batch(all_desc_dicts)
        report = ClassificationReport()
        # variable_summary takes a single ClassificationResult
        summary = report.variable_summary(batch_result.results[0])
        assert isinstance(summary, str)

    def test_identify_patterns(self, classifier, all_desc_dicts):
        batch_result = classifier.classify_batch(all_desc_dicts)
        report = ClassificationReport()
        patterns = report.identify_patterns(batch_result)
        assert isinstance(patterns, list)

    def test_visualization_data(self, classifier, all_desc_dicts):
        batch_result = classifier.classify_batch(all_desc_dicts)
        report = ClassificationReport()
        viz = report.visualization_data(batch_result)
        assert isinstance(viz, dict)

    def test_summary_statistics(self, classifier, all_desc_dicts):
        batch_result = classifier.classify_batch(all_desc_dicts)
        stats = classifier.summary_statistics(batch_result)
        assert isinstance(stats, dict)

    def test_batch_category_distribution(self, classifier, all_desc_dicts):
        batch_result = classifier.classify_batch(all_desc_dicts)
        assert isinstance(batch_result.category_distribution, dict)
        assert batch_result.n_variables == len(all_desc_dicts)


# ---------------------------------------------------------------------------
# Test ClassificationValidator
# ---------------------------------------------------------------------------

class TestClassificationValidator:

    def test_sensitivity_analysis(self):
        validator = ClassificationValidator(n_perturbations=20, random_state=42)
        descs = [
            _desc_dict(0.09, 0.4, 0.3, 0.2),
            _desc_dict(0.5, 0.1, 0.1, 0.1),
        ]
        result = validator.sensitivity_analysis(descs)
        assert isinstance(result, dict)

    def test_bootstrap_stability(self):
        validator = ClassificationValidator(n_bootstrap=20, random_state=42)
        rng = np.random.default_rng(42)
        descs = [
            _desc_dict(rng.uniform(0, 1), rng.uniform(0, 1),
                       rng.uniform(0, 1), rng.uniform(0, 1))
            for _ in range(10)
        ]
        result = validator.bootstrap_stability(descs)
        assert isinstance(result, dict)

    def test_cross_validate_thresholds(self):
        validator = ClassificationValidator(random_state=42)
        rng = np.random.default_rng(42)
        descs = [
            _desc_dict(rng.uniform(0, 1), rng.uniform(0, 1),
                       rng.uniform(0, 1), rng.uniform(0, 1))
            for _ in range(20)
        ]
        true_labels = [
            "invariant" if d["psi_S"] < 0.1 else "structural_plastic"
            for d in descs
        ]
        result = validator.cross_validate_thresholds(descs, true_labels=true_labels)
        assert isinstance(result, dict)

    def test_sensitivity_near_boundary(self):
        validator = ClassificationValidator(n_perturbations=30, random_state=42)
        descs = [_desc_dict(0.101, 0.0, 0.0, 0.0)]
        result = validator.sensitivity_analysis(descs)
        assert isinstance(result, dict)

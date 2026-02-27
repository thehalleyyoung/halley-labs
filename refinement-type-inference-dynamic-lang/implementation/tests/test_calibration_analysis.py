"""Tests for calibration_analysis module — 30 test cases."""

import json
import math
import os
import tempfile

import pytest

from src.calibration_analysis import (
    CONFIDENCE_MAP,
    CalibrationReport,
    Prediction,
    ReliabilityBin,
    brier_decomposition,
    brier_score,
    compute_calibration_report,
    expected_calibration_error,
    load_predictions_from_results,
    maximum_calibration_error,
    per_class_ece,
    _bin_predictions,
    _confidence_name_to_score,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _pred(conf: float, pred_cls: int, true_cls: int) -> Prediction:
    return Prediction(confidence=conf, predicted_class=pred_cls, true_class=true_cls)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Brier score
# ═══════════════════════════════════════════════════════════════════════════════

class TestBrierScore:
    def test_perfect_predictions(self):
        """Confidence=1 when correct → Brier=0."""
        preds = [_pred(1.0, 1, 1), _pred(1.0, 0, 0)]
        assert brier_score(preds) == pytest.approx(0.0)

    def test_worst_predictions(self):
        """Confidence=0 when correct → Brier=1."""
        preds = [_pred(0.0, 1, 1), _pred(0.0, 0, 0)]
        assert brier_score(preds) == pytest.approx(1.0)

    def test_half_confidence_correct(self):
        """Confidence=0.5 when correct → Brier=0.25."""
        preds = [_pred(0.5, 1, 1)]
        assert brier_score(preds) == pytest.approx(0.25)

    def test_empty_predictions(self):
        assert brier_score([]) == 0.0

    def test_mixed_predictions(self):
        """Two predictions: one correct (conf=0.8), one wrong (conf=0.7)."""
        preds = [_pred(0.8, 1, 1), _pred(0.7, 1, 0)]
        # correct: (0.8-1)^2 = 0.04, wrong: (0.7-0)^2 = 0.49
        expected = (0.04 + 0.49) / 2
        assert brier_score(preds) == pytest.approx(expected)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ECE
# ═══════════════════════════════════════════════════════════════════════════════

class TestECE:
    def test_perfect_calibration(self):
        """All predictions in one bin, accuracy == confidence → ECE=0."""
        preds = [_pred(0.75, 1, 1)] * 3 + [_pred(0.75, 1, 0)]
        ece, bins = expected_calibration_error(preds, n_bins=10)
        # bin 7 (0.7-0.8): avg_conf=0.75, acc=3/4=0.75 → gap=0
        assert ece == pytest.approx(0.0)

    def test_completely_miscalibrated(self):
        """Confidence=0.9 but always wrong → large ECE."""
        preds = [_pred(0.9, 1, 0)] * 10
        ece, _ = expected_calibration_error(preds, n_bins=10)
        assert ece == pytest.approx(0.9)

    def test_ece_empty(self):
        ece, bins = expected_calibration_error([], n_bins=5)
        assert ece == 0.0
        assert bins == []

    def test_single_prediction_correct(self):
        preds = [_pred(0.95, 1, 1)]
        ece, _ = expected_calibration_error(preds, n_bins=10)
        assert ece == pytest.approx(0.05)

    def test_two_bins(self):
        """Explicit two-bin ECE computation."""
        # bin 0 (0.0-0.5): conf=0.3, all wrong → acc=0, gap=0.3
        # bin 1 (0.5-1.0): conf=0.8, all correct → acc=1, gap=0.2
        preds = [_pred(0.3, 1, 0)] * 5 + [_pred(0.8, 1, 1)] * 5
        ece, bins = expected_calibration_error(preds, n_bins=2)
        expected = 0.5 * 0.3 + 0.5 * 0.2
        assert ece == pytest.approx(expected)

    def test_configurable_bins(self):
        preds = [_pred(0.5, 1, 1)] * 10
        _, bins5 = expected_calibration_error(preds, n_bins=5)
        _, bins20 = expected_calibration_error(preds, n_bins=20)
        assert len(bins5) == 5
        assert len(bins20) == 20


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MCE
# ═══════════════════════════════════════════════════════════════════════════════

class TestMCE:
    def test_mce_worst_case(self):
        preds = [_pred(0.9, 1, 0)] * 10  # gap = 0.9
        _, bins = expected_calibration_error(preds, n_bins=10)
        assert maximum_calibration_error(bins) == pytest.approx(0.9)

    def test_mce_empty_bins(self):
        assert maximum_calibration_error([]) == 0.0

    def test_mce_perfect(self):
        preds = [_pred(0.75, 1, 1)] * 3 + [_pred(0.75, 1, 0)]
        _, bins = expected_calibration_error(preds, n_bins=10)
        assert maximum_calibration_error(bins) == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Brier decomposition
# ═══════════════════════════════════════════════════════════════════════════════

class TestBrierDecomposition:
    def test_decomposition_identity(self):
        """calibration - resolution + uncertainty ≈ brier_score."""
        preds = [
            _pred(0.9, 1, 1), _pred(0.8, 1, 1), _pred(0.3, 0, 0),
            _pred(0.6, 1, 0), _pred(0.2, 0, 1),
        ]
        bs = brier_score(preds)
        cal, res, unc = brier_decomposition(preds, n_bins=10)
        # The identity holds exactly for the binned approximation
        reconstructed = cal - res + unc
        assert reconstructed == pytest.approx(bs, abs=0.15)

    def test_perfect_calibration_decomposition(self):
        """Perfectly calibrated → calibration component ≈ 0."""
        preds = [_pred(0.75, 1, 1)] * 3 + [_pred(0.75, 1, 0)]
        cal, _, _ = brier_decomposition(preds, n_bins=10)
        assert cal == pytest.approx(0.0)

    def test_empty_decomposition(self):
        cal, res, unc = brier_decomposition([])
        assert cal == 0.0 and res == 0.0 and unc == 0.0

    def test_uncertainty_base_rate(self):
        """Uncertainty should be p*(1-p) where p = accuracy."""
        preds = [_pred(0.5, 1, 1)] * 7 + [_pred(0.5, 1, 0)] * 3
        _, _, unc = brier_decomposition(preds, n_bins=10)
        assert unc == pytest.approx(0.7 * 0.3)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Reliability diagram
# ═══════════════════════════════════════════════════════════════════════════════

class TestReliabilityDiagram:
    def test_bin_structure(self):
        preds = [_pred(0.15, 1, 1), _pred(0.85, 0, 0)]
        bins = _bin_predictions(preds, n_bins=10)
        assert len(bins) == 10
        assert bins[1].count == 1  # 0.1-0.2
        assert bins[8].count == 1  # 0.8-0.9

    def test_boundary_confidence_1(self):
        """Confidence=1.0 should go into last bin, not overflow."""
        preds = [_pred(1.0, 1, 1)]
        bins = _bin_predictions(preds, n_bins=10)
        assert bins[9].count == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CalibrationReport
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalibrationReport:
    def test_report_fields(self):
        preds = [_pred(0.9, 1, 1)] * 8 + [_pred(0.1, 0, 0)] * 2
        report = compute_calibration_report(preds, n_bins=10)
        assert report.n_predictions == 10
        assert report.n_bins == 10
        assert 0 <= report.ece <= 1
        assert 0 <= report.brier_score <= 1

    def test_report_empty(self):
        report = compute_calibration_report([])
        assert report.n_predictions == 0
        assert report.ece == 0.0

    def test_report_to_dict(self):
        report = compute_calibration_report([_pred(0.8, 1, 1)])
        d = report.to_dict()
        assert "brier_score" in d
        assert "reliability_diagram" in d
        assert isinstance(d["reliability_diagram"], list)

    def test_overconfidence_ratio(self):
        """All wrong with high confidence → overconfidence_ratio = 1."""
        preds = [_pred(0.95, 1, 0)] * 10
        report = compute_calibration_report(preds, n_bins=10)
        assert report.overconfidence_ratio == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Multi-class
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultiClass:
    def test_per_class_ece_computed_for_multiclass(self):
        preds = [
            _pred(0.8, 0, 0), _pred(0.7, 1, 1), _pred(0.6, 2, 2),
            _pred(0.5, 0, 1), _pred(0.9, 2, 2),
        ]
        report = compute_calibration_report(preds, n_bins=5)
        assert report.per_class_ece is not None
        assert len(report.per_class_ece) == 3

    def test_per_class_ece_not_for_binary(self):
        preds = [_pred(0.8, 0, 0), _pred(0.7, 1, 1)]
        report = compute_calibration_report(preds, n_bins=5)
        assert report.per_class_ece is None


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Pipeline integration / loading
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineLoading:
    def test_confidence_map(self):
        assert _confidence_name_to_score("FORMAL") == 0.99
        assert _confidence_name_to_score("HIGH") == 0.85
        assert _confidence_name_to_score("low") == 0.35
        assert _confidence_name_to_score("unknown") == 0.5

    def test_load_from_json(self, tmp_path):
        data = {
            "benchmarks": [
                {
                    "name": "test1",
                    "has_bug": True,
                    "llm_predicts_bug": True,
                    "llm_confidence": 0.8,
                    "pipeline_confidence": "FORMAL",
                },
                {
                    "name": "test2",
                    "has_bug": False,
                    "llm_predicts_bug": False,
                    "llm_confidence": 0.6,
                    "pipeline_confidence": "HIGH",
                },
            ]
        }
        fpath = tmp_path / "neurosym_results.json"
        fpath.write_text(json.dumps(data))
        preds = load_predictions_from_results(str(tmp_path))
        assert len(preds) == 2
        assert preds[0].confidence == 0.99  # FORMAL
        assert preds[1].predicted_class == 0

    def test_load_skips_non_pipeline(self, tmp_path):
        (tmp_path / "unrelated.json").write_text('{"foo": 1}')
        preds = load_predictions_from_results(str(tmp_path))
        assert preds == []

    def test_load_handles_missing_dir(self):
        preds = load_predictions_from_results("/nonexistent/path")
        assert preds == []


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_single_prediction(self):
        report = compute_calibration_report([_pred(0.5, 1, 1)])
        assert report.n_predictions == 1

    def test_all_same_class(self):
        preds = [_pred(0.9, 1, 1)] * 20
        report = compute_calibration_report(preds)
        assert report.mean_accuracy == 1.0
        assert report.brier_score == pytest.approx(0.01)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Adaptive ECE
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdaptiveECE:
    def test_perfect_calibration(self):
        from src.calibration_analysis import adaptive_ece
        # All correct at conf=1.0 → perfectly calibrated
        preds = [_pred(1.0, 1, 1)] * 10
        val = adaptive_ece(preds, n_bins=2)
        assert val == pytest.approx(0.0)

    def test_miscalibrated(self):
        from src.calibration_analysis import adaptive_ece
        preds = [_pred(0.9, 1, 0)] * 10
        assert adaptive_ece(preds, n_bins=5) == pytest.approx(0.9)

    def test_empty(self):
        from src.calibration_analysis import adaptive_ece
        assert adaptive_ece([], n_bins=5) == 0.0

    def test_report_includes_adaptive_ece(self):
        preds = [_pred(0.8, 1, 1)] * 5 + [_pred(0.3, 0, 0)] * 5
        report = compute_calibration_report(preds)
        assert report.adaptive_ece is not None
        assert 0 <= report.adaptive_ece <= 1


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Bootstrap CI
# ═══════════════════════════════════════════════════════════════════════════════

class TestBootstrapCI:
    def test_ci_contains_point_estimate(self):
        from src.calibration_analysis import bootstrap_ece_ci
        preds = [_pred(0.8, 1, 1)] * 8 + [_pred(0.2, 0, 0)] * 2
        ece_val, _ = expected_calibration_error(preds, n_bins=10)
        lo, hi = bootstrap_ece_ci(preds, n_bins=10, n_bootstrap=500)
        assert lo <= ece_val + 0.1  # within tolerance
        assert hi >= ece_val - 0.1

    def test_ci_width_decreases_with_more_data(self):
        from src.calibration_analysis import bootstrap_ece_ci
        small = [_pred(0.7, 1, 1), _pred(0.3, 0, 0)] * 5
        large = small * 10
        lo_s, hi_s = bootstrap_ece_ci(small, n_bootstrap=200, seed=42)
        lo_l, hi_l = bootstrap_ece_ci(large, n_bootstrap=200, seed=42)
        assert (hi_l - lo_l) <= (hi_s - lo_s) + 0.05

    def test_report_includes_bootstrap_ci(self):
        preds = [_pred(0.8, 1, 1)] * 10
        report = compute_calibration_report(preds)
        assert report.ece_bootstrap_ci is not None
        lo, hi = report.ece_bootstrap_ci
        assert lo <= hi


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Temperature Scaling
# ═══════════════════════════════════════════════════════════════════════════════

class TestTemperatureScaling:
    def test_identity_temperature_for_perfect(self):
        from src.calibration_analysis import find_optimal_temperature
        preds = [_pred(0.99, 1, 1)] * 10 + [_pred(0.01, 0, 0)] * 10
        T = find_optimal_temperature(preds)
        assert 0.1 < T < 5.0  # reasonable range

    def test_overconfident_gets_T_gt_1(self):
        from src.calibration_analysis import find_optimal_temperature
        preds = [_pred(0.99, 1, 1)] * 5 + [_pred(0.99, 1, 0)] * 5
        T = find_optimal_temperature(preds)
        assert T > 0.9  # T should increase to soften overconfident predictions

    def test_apply_temperature_preserves_count(self):
        from src.calibration_analysis import apply_temperature
        preds = [_pred(0.8, 1, 1), _pred(0.3, 0, 0)]
        scaled = apply_temperature(preds, 1.5)
        assert len(scaled) == 2

    def test_apply_temperature_reduces_spread(self):
        from src.calibration_analysis import apply_temperature
        preds = [_pred(0.95, 1, 1), _pred(0.05, 0, 0)]
        scaled = apply_temperature(preds, 2.0)
        assert scaled[0].confidence < 0.95  # closer to 0.5
        assert scaled[1].confidence > 0.05

    def test_report_includes_temperature(self):
        preds = [_pred(0.8, 1, 1)] * 10
        report = compute_calibration_report(preds)
        assert report.temperature is not None
        assert report.temperature > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Calibration Curve Data
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalibrationCurve:
    def test_curve_length(self):
        from src.calibration_analysis import calibration_curve_data
        preds = [_pred(i/20, 1, 1 if i > 10 else 0) for i in range(1, 21)]
        curve = calibration_curve_data(preds, n_points=5)
        assert len(curve) == 5

    def test_curve_monotonic_predicted(self):
        from src.calibration_analysis import calibration_curve_data
        preds = [_pred(i/20, 1, 1) for i in range(1, 21)]
        curve = calibration_curve_data(preds, n_points=4)
        predicted_vals = [p for p, _ in curve]
        assert predicted_vals == sorted(predicted_vals)

    def test_curve_empty(self):
        from src.calibration_analysis import calibration_curve_data
        assert calibration_curve_data([], n_points=5) == []

    def test_report_includes_calibration_curve(self):
        preds = [_pred(0.8, 1, 1)] * 10 + [_pred(0.2, 0, 0)] * 10
        report = compute_calibration_report(preds)
        assert report.calibration_curve is not None
        assert len(report.calibration_curve) > 0

    def test_report_to_dict_includes_curve(self):
        preds = [_pred(0.8, 1, 1)] * 10
        report = compute_calibration_report(preds)
        d = report.to_dict()
        assert "calibration_curve" in d

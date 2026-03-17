"""
Comprehensive tests for taintflow.core.config – Configuration system.

Tests cover TaintFlowConfig and SeverityThresholds: default values,
custom values, validation, serialisation, from_dict, merge configs,
and preset factories (quick_audit, thorough_audit, ci_mode).
"""

from __future__ import annotations

import json
import math
import unittest

from taintflow.core.config import TaintFlowConfig, SeverityThresholds


# ===================================================================
#  SeverityThresholds
# ===================================================================


class TestSeverityThresholdsDefaults(unittest.TestCase):
    """Tests for SeverityThresholds default values."""

    def test_default_negligible_max(self):
        """Default negligible_max should be 1.0."""
        st = SeverityThresholds()
        self.assertAlmostEqual(st.negligible_max, 1.0)

    def test_default_warning_max(self):
        """Default warning_max should be 8.0."""
        st = SeverityThresholds()
        self.assertAlmostEqual(st.warning_max, 8.0)

    def test_critical_min_property(self):
        """critical_min should equal warning_max."""
        st = SeverityThresholds()
        self.assertAlmostEqual(st.critical_min, st.warning_max)


class TestSeverityThresholdsClassify(unittest.TestCase):
    """Tests for SeverityThresholds.classify()."""

    def test_classify_negligible(self):
        """Bits <= negligible_max should be negligible."""
        st = SeverityThresholds(negligible_max=1.0, warning_max=8.0)
        self.assertEqual(st.classify(0.0), "negligible")
        self.assertEqual(st.classify(0.5), "negligible")
        self.assertEqual(st.classify(1.0), "negligible")

    def test_classify_warning(self):
        """Bits in (negligible_max, warning_max] should be warning."""
        st = SeverityThresholds(negligible_max=1.0, warning_max=8.0)
        self.assertEqual(st.classify(1.01), "warning")
        self.assertEqual(st.classify(5.0), "warning")
        self.assertEqual(st.classify(8.0), "warning")

    def test_classify_critical(self):
        """Bits > warning_max should be critical."""
        st = SeverityThresholds(negligible_max=1.0, warning_max=8.0)
        self.assertEqual(st.classify(8.01), "critical")
        self.assertEqual(st.classify(64.0), "critical")

    def test_classify_custom_thresholds(self):
        """Custom thresholds should change classification."""
        st = SeverityThresholds(negligible_max=0.5, warning_max=2.0)
        self.assertEqual(st.classify(0.5), "negligible")
        self.assertEqual(st.classify(0.6), "warning")
        self.assertEqual(st.classify(2.1), "critical")


class TestSeverityThresholdsValidation(unittest.TestCase):
    """Tests for SeverityThresholds.validate()."""

    def test_valid(self):
        """Default thresholds should pass validation."""
        self.assertEqual(SeverityThresholds().validate(), [])

    def test_negative_negligible_max(self):
        """Negative negligible_max should fail validation."""
        st = SeverityThresholds(negligible_max=-1.0)
        errors = st.validate()
        self.assertTrue(len(errors) > 0)

    def test_warning_less_than_negligible(self):
        """warning_max < negligible_max should fail validation."""
        st = SeverityThresholds(negligible_max=5.0, warning_max=2.0)
        errors = st.validate()
        self.assertTrue(any("warning_max" in e for e in errors))

    def test_equal_thresholds_valid(self):
        """Equal thresholds should be valid."""
        st = SeverityThresholds(negligible_max=1.0, warning_max=1.0)
        self.assertEqual(st.validate(), [])


class TestSeverityThresholdsSerialization(unittest.TestCase):
    """Tests for SeverityThresholds serialisation."""

    def test_to_dict(self):
        """to_dict should include both thresholds."""
        d = SeverityThresholds(negligible_max=2.0, warning_max=10.0).to_dict()
        self.assertAlmostEqual(d["negligible_max"], 2.0)
        self.assertAlmostEqual(d["warning_max"], 10.0)

    def test_from_dict(self):
        """from_dict should restore thresholds."""
        st = SeverityThresholds.from_dict({"negligible_max": 3.0, "warning_max": 12.0})
        self.assertAlmostEqual(st.negligible_max, 3.0)
        self.assertAlmostEqual(st.warning_max, 12.0)

    def test_from_dict_defaults(self):
        """from_dict with empty dict should use defaults."""
        st = SeverityThresholds.from_dict({})
        self.assertAlmostEqual(st.negligible_max, 1.0)
        self.assertAlmostEqual(st.warning_max, 8.0)

    def test_roundtrip(self):
        """from_dict(to_dict()) should preserve values."""
        original = SeverityThresholds(negligible_max=0.5, warning_max=4.0)
        restored = SeverityThresholds.from_dict(original.to_dict())
        self.assertAlmostEqual(original.negligible_max, restored.negligible_max)
        self.assertAlmostEqual(original.warning_max, restored.warning_max)

    def test_repr(self):
        """repr should mention thresholds."""
        st = SeverityThresholds()
        r = repr(st)
        self.assertIn("negligible", r)
        self.assertIn("warning", r)


# ===================================================================
#  TaintFlowConfig – defaults
# ===================================================================


class TestTaintFlowConfigDefaults(unittest.TestCase):
    """Tests for TaintFlowConfig default values."""

    def test_default_b_max(self):
        """Default b_max should be 64.0."""
        cfg = TaintFlowConfig()
        self.assertAlmostEqual(cfg.b_max, 64.0)

    def test_default_alpha(self):
        """Default alpha should be 0.05."""
        cfg = TaintFlowConfig()
        self.assertAlmostEqual(cfg.alpha, 0.05)

    def test_default_max_iterations(self):
        """Default max_iterations should be 1000."""
        cfg = TaintFlowConfig()
        self.assertEqual(cfg.max_iterations, 1000)

    def test_default_use_widening(self):
        """Default use_widening should be True."""
        cfg = TaintFlowConfig()
        self.assertTrue(cfg.use_widening)

    def test_default_widening_delay(self):
        """Default widening_delay should be 3."""
        cfg = TaintFlowConfig()
        self.assertEqual(cfg.widening_delay, 3)

    def test_default_use_narrowing(self):
        """Default use_narrowing should be True."""
        cfg = TaintFlowConfig()
        self.assertTrue(cfg.use_narrowing)

    def test_default_narrowing_iterations(self):
        """Default narrowing_iterations should be 5."""
        cfg = TaintFlowConfig()
        self.assertEqual(cfg.narrowing_iterations, 5)

    def test_default_epsilon(self):
        """Default epsilon should be 1e-10."""
        cfg = TaintFlowConfig()
        self.assertAlmostEqual(cfg.epsilon, 1e-10)

    def test_default_parallel(self):
        """Default parallel should be False."""
        cfg = TaintFlowConfig()
        self.assertFalse(cfg.parallel)

    def test_default_n_workers(self):
        """Default n_workers should be 1."""
        cfg = TaintFlowConfig()
        self.assertEqual(cfg.n_workers, 1)

    def test_default_verbosity(self):
        """Default verbosity should be 1."""
        cfg = TaintFlowConfig()
        self.assertEqual(cfg.verbosity, 1)

    def test_default_profile(self):
        """Default profile should be 'default'."""
        cfg = TaintFlowConfig()
        self.assertEqual(cfg.profile, "default")

    def test_default_severity(self):
        """Default severity sub-config should exist."""
        cfg = TaintFlowConfig()
        self.assertIsInstance(cfg.severity, SeverityThresholds)

    def test_default_channel(self):
        """Default channel sub-config should have analytic tier."""
        cfg = TaintFlowConfig()
        self.assertEqual(cfg.channel.tier_preference, "analytic")

    def test_default_report(self):
        """Default report format should be text."""
        cfg = TaintFlowConfig()
        self.assertEqual(cfg.report.format, "text")


# ===================================================================
#  TaintFlowConfig – custom values
# ===================================================================


class TestTaintFlowConfigCustom(unittest.TestCase):
    """Tests for TaintFlowConfig with custom values."""

    def test_custom_b_max(self):
        """Custom b_max should be preserved."""
        cfg = TaintFlowConfig(b_max=32.0)
        self.assertAlmostEqual(cfg.b_max, 32.0)

    def test_custom_alpha(self):
        """Custom alpha should be preserved."""
        cfg = TaintFlowConfig(alpha=0.01)
        self.assertAlmostEqual(cfg.alpha, 0.01)

    def test_custom_severity(self):
        """Custom severity thresholds should be preserved."""
        st = SeverityThresholds(negligible_max=0.5, warning_max=4.0)
        cfg = TaintFlowConfig(severity=st)
        self.assertAlmostEqual(cfg.severity.negligible_max, 0.5)

    def test_custom_parallel(self):
        """Custom parallel and n_workers should be preserved."""
        cfg = TaintFlowConfig(parallel=True, n_workers=8)
        self.assertTrue(cfg.parallel)
        self.assertEqual(cfg.n_workers, 8)


# ===================================================================
#  TaintFlowConfig – validation
# ===================================================================


class TestTaintFlowConfigValidation(unittest.TestCase):
    """Tests for TaintFlowConfig validation."""

    def test_valid_default(self):
        """Default config should pass validation."""
        self.assertEqual(TaintFlowConfig().validate(), [])

    def test_invalid_b_max(self):
        """b_max <= 0 should fail."""
        cfg = TaintFlowConfig(b_max=0.0)
        errors = cfg.validate()
        self.assertTrue(any("b_max" in e for e in errors))

    def test_invalid_b_max_negative(self):
        """Negative b_max should fail."""
        cfg = TaintFlowConfig(b_max=-1.0)
        self.assertTrue(len(cfg.validate()) > 0)

    def test_invalid_alpha_zero(self):
        """alpha = 0 should fail."""
        cfg = TaintFlowConfig(alpha=0.0)
        self.assertTrue(any("alpha" in e for e in cfg.validate()))

    def test_invalid_alpha_one(self):
        """alpha = 1 should fail."""
        cfg = TaintFlowConfig(alpha=1.0)
        self.assertTrue(any("alpha" in e for e in cfg.validate()))

    def test_invalid_max_iterations(self):
        """max_iterations < 1 should fail."""
        cfg = TaintFlowConfig(max_iterations=0)
        self.assertTrue(any("max_iterations" in e for e in cfg.validate()))

    def test_invalid_widening_delay(self):
        """widening_delay < 0 should fail."""
        cfg = TaintFlowConfig(widening_delay=-1)
        self.assertTrue(any("widening_delay" in e for e in cfg.validate()))

    def test_invalid_epsilon(self):
        """epsilon <= 0 should fail."""
        cfg = TaintFlowConfig(epsilon=0.0)
        self.assertTrue(any("epsilon" in e for e in cfg.validate()))

    def test_invalid_n_workers(self):
        """n_workers < 1 should fail."""
        cfg = TaintFlowConfig(n_workers=0)
        self.assertTrue(any("n_workers" in e for e in cfg.validate()))

    def test_invalid_verbosity(self):
        """verbosity < 0 should fail."""
        cfg = TaintFlowConfig(verbosity=-1)
        self.assertTrue(any("verbosity" in e for e in cfg.validate()))

    def test_cascading_validation(self):
        """Validation should include sub-config errors."""
        from taintflow.core.config import ReportSettings
        cfg = TaintFlowConfig(report=ReportSettings(format="invalid_format"))
        errors = cfg.validate()
        self.assertTrue(any("format" in e for e in errors))

    def test_validate_or_raise(self):
        """validate_or_raise should raise for invalid config."""
        cfg = TaintFlowConfig(b_max=-1.0)
        with self.assertRaises(Exception):
            cfg.validate_or_raise()


# ===================================================================
#  TaintFlowConfig – serialisation
# ===================================================================


class TestTaintFlowConfigSerialization(unittest.TestCase):
    """Tests for TaintFlowConfig serialisation."""

    def test_to_dict_keys(self):
        """to_dict should include all top-level keys."""
        d = TaintFlowConfig().to_dict()
        expected_keys = {
            "b_max", "alpha", "max_iterations", "use_widening",
            "widening_delay", "use_narrowing", "narrowing_iterations",
            "epsilon", "parallel", "n_workers", "verbosity",
            "severity", "channel", "instrumentation", "report", "profile",
        }
        self.assertEqual(set(d.keys()), expected_keys)

    def test_to_dict_severity_nested(self):
        """severity in to_dict should be a dict."""
        d = TaintFlowConfig().to_dict()
        self.assertIsInstance(d["severity"], dict)
        self.assertIn("negligible_max", d["severity"])

    def test_from_dict_roundtrip(self):
        """from_dict(to_dict()) should produce equivalent config."""
        original = TaintFlowConfig(
            b_max=48.0, alpha=0.1, max_iterations=500,
            use_widening=False, verbosity=2,
        )
        restored = TaintFlowConfig.from_dict(original.to_dict())
        self.assertAlmostEqual(original.b_max, restored.b_max)
        self.assertAlmostEqual(original.alpha, restored.alpha)
        self.assertEqual(original.max_iterations, restored.max_iterations)
        self.assertEqual(original.use_widening, restored.use_widening)
        self.assertEqual(original.verbosity, restored.verbosity)

    def test_from_dict_with_nested(self):
        """from_dict should handle nested severity/channel dicts."""
        data = {
            "b_max": 32.0,
            "severity": {"negligible_max": 0.5, "warning_max": 4.0},
            "channel": {"tier_preference": "sampling"},
        }
        cfg = TaintFlowConfig.from_dict(data)
        self.assertAlmostEqual(cfg.b_max, 32.0)
        self.assertAlmostEqual(cfg.severity.negligible_max, 0.5)
        self.assertEqual(cfg.channel.tier_preference, "sampling")

    def test_to_json(self):
        """to_json should produce valid JSON."""
        cfg = TaintFlowConfig()
        json_str = cfg.to_json()
        parsed = json.loads(json_str)
        self.assertAlmostEqual(parsed["b_max"], 64.0)

    def test_to_json_custom_indent(self):
        """to_json should respect indent parameter."""
        cfg = TaintFlowConfig()
        json_str = cfg.to_json(indent=4)
        self.assertIn("    ", json_str)


# ===================================================================
#  TaintFlowConfig – merge
# ===================================================================


class TestTaintFlowConfigMerge(unittest.TestCase):
    """Tests for TaintFlowConfig.merge()."""

    def test_merge_override_b_max(self):
        """Override b_max should win over base."""
        base = TaintFlowConfig(b_max=64.0)
        override = TaintFlowConfig(b_max=32.0)
        merged = base.merge(override)
        self.assertAlmostEqual(merged.b_max, 32.0)

    def test_merge_default_does_not_override(self):
        """Default values in override should not replace base."""
        base = TaintFlowConfig(b_max=32.0)
        override = TaintFlowConfig()  # all defaults
        merged = base.merge(override)
        self.assertAlmostEqual(merged.b_max, 32.0)

    def test_merge_severity(self):
        """Merging should handle nested severity."""
        base = TaintFlowConfig()
        override = TaintFlowConfig(
            severity=SeverityThresholds(negligible_max=0.5, warning_max=4.0)
        )
        merged = base.merge(override)
        self.assertAlmostEqual(merged.severity.negligible_max, 0.5)

    def test_merge_config_path(self):
        """Override config_path should take precedence."""
        base = TaintFlowConfig()
        base.config_path = "/base/path"
        override = TaintFlowConfig()
        override.config_path = "/override/path"
        merged = base.merge(override)
        self.assertEqual(merged.config_path, "/override/path")

    def test_merge_config_path_fallback(self):
        """If override has no config_path, use base's."""
        base = TaintFlowConfig()
        base.config_path = "/base/path"
        override = TaintFlowConfig()
        merged = base.merge(override)
        self.assertEqual(merged.config_path, "/base/path")

    def test_merge_multiple(self):
        """Chaining merges should work."""
        a = TaintFlowConfig(b_max=32.0)
        b = TaintFlowConfig(alpha=0.1)
        c = TaintFlowConfig(max_iterations=500)
        result = a.merge(b).merge(c)
        self.assertAlmostEqual(result.b_max, 32.0)
        self.assertAlmostEqual(result.alpha, 0.1)
        self.assertEqual(result.max_iterations, 500)


# ===================================================================
#  TaintFlowConfig – presets
# ===================================================================


class TestTaintFlowConfigPresets(unittest.TestCase):
    """Tests for configuration presets."""

    def test_quick_audit(self):
        """quick_audit should set fast parameters."""
        cfg = TaintFlowConfig.quick_audit()
        self.assertAlmostEqual(cfg.b_max, 32.0)
        self.assertLess(cfg.max_iterations, 100)
        self.assertFalse(cfg.use_narrowing)
        self.assertEqual(cfg.profile, "quick")
        self.assertEqual(cfg.channel.tier_preference, "heuristic")

    def test_quick_audit_valid(self):
        """quick_audit config should pass validation."""
        self.assertEqual(TaintFlowConfig.quick_audit().validate(), [])

    def test_thorough_audit(self):
        """thorough_audit should set precise parameters."""
        cfg = TaintFlowConfig.thorough_audit()
        self.assertAlmostEqual(cfg.b_max, 64.0)
        self.assertGreater(cfg.max_iterations, 1000)
        self.assertTrue(cfg.use_narrowing)
        self.assertGreater(cfg.narrowing_iterations, 5)
        self.assertEqual(cfg.profile, "thorough")

    def test_thorough_audit_valid(self):
        """thorough_audit config should pass validation."""
        self.assertEqual(TaintFlowConfig.thorough_audit().validate(), [])

    def test_ci_mode(self):
        """ci_mode should set CI-appropriate parameters."""
        cfg = TaintFlowConfig.ci_mode()
        self.assertEqual(cfg.verbosity, 0)
        self.assertEqual(cfg.report.format, "sarif")
        self.assertEqual(cfg.profile, "ci")
        self.assertAlmostEqual(cfg.severity.negligible_max, 0.5)

    def test_ci_mode_valid(self):
        """ci_mode config should pass validation."""
        self.assertEqual(TaintFlowConfig.ci_mode().validate(), [])

    def test_presets_differ(self):
        """Presets should produce different configurations."""
        quick = TaintFlowConfig.quick_audit()
        thorough = TaintFlowConfig.thorough_audit()
        ci = TaintFlowConfig.ci_mode()
        self.assertNotEqual(quick.max_iterations, thorough.max_iterations)
        self.assertNotEqual(quick.profile, ci.profile)


# ===================================================================
#  TaintFlowConfig – to_analysis_config
# ===================================================================


class TestTaintFlowConfigToAnalysisConfig(unittest.TestCase):
    """Tests for TaintFlowConfig.to_analysis_config()."""

    def test_to_analysis_config(self):
        """to_analysis_config should produce a matching AnalysisConfig."""
        cfg = TaintFlowConfig(
            b_max=48.0, alpha=0.1, max_iterations=500,
            use_widening=False, widening_delay=5,
            use_narrowing=True, narrowing_iterations=10,
            epsilon=1e-8, parallel=True, n_workers=4,
        )
        ac = cfg.to_analysis_config()
        self.assertAlmostEqual(ac.b_max, 48.0)
        self.assertAlmostEqual(ac.alpha, 0.1)
        self.assertEqual(ac.max_iterations, 500)
        self.assertFalse(ac.use_widening)
        self.assertEqual(ac.widening_delay, 5)
        self.assertTrue(ac.use_narrowing)
        self.assertEqual(ac.narrowing_iterations, 10)
        self.assertAlmostEqual(ac.epsilon, 1e-8)
        self.assertTrue(ac.parallel)
        self.assertEqual(ac.n_workers, 4)


# ===================================================================
#  TaintFlowConfig – from_cli_args
# ===================================================================


class TestTaintFlowConfigFromCLI(unittest.TestCase):
    """Tests for TaintFlowConfig.from_cli_args()."""

    def test_from_cli_args_b_max(self):
        """from_cli_args should set b_max from kwargs."""
        cfg = TaintFlowConfig.from_cli_args(b_max=32.0)
        self.assertAlmostEqual(cfg.b_max, 32.0)

    def test_from_cli_args_ignores_none(self):
        """from_cli_args should ignore None values."""
        cfg = TaintFlowConfig.from_cli_args(b_max=None, alpha=0.1)
        self.assertAlmostEqual(cfg.alpha, 0.1)
        # b_max should be default since it was None
        self.assertAlmostEqual(cfg.b_max, 64.0)

    def test_from_cli_args_empty(self):
        """from_cli_args with no kwargs should give defaults."""
        cfg = TaintFlowConfig.from_cli_args()
        self.assertAlmostEqual(cfg.b_max, 64.0)


# ===================================================================
#  TaintFlowConfig – repr and summary
# ===================================================================


class TestTaintFlowConfigDisplay(unittest.TestCase):
    """Tests for repr and summary."""

    def test_repr_contains_key_info(self):
        """repr should mention B_max and alpha."""
        cfg = TaintFlowConfig()
        r = repr(cfg)
        self.assertIn("64", r)
        self.assertIn("0.05", r)

    def test_summary_contains_sections(self):
        """summary should contain all config sections."""
        cfg = TaintFlowConfig()
        s = cfg.summary()
        self.assertIn("B_max", s)
        self.assertIn("alpha", s)
        self.assertIn("widening", s)
        self.assertIn("narrowing", s)
        self.assertIn("severity", s)
        self.assertIn("channel", s)


if __name__ == "__main__":
    unittest.main()

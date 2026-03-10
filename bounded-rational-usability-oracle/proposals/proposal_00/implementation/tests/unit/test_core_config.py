"""
Unit tests for usability_oracle.core.config.

Covers all configuration dataclasses: OracleConfig, ParserConfig,
AlignmentConfig, CognitiveConfig, MDPConfig, PolicyConfig,
BisimulationConfig, ComparisonConfig, RepairConfig, PipelineConfig,
OutputConfig.  Tests default construction, field validation,
round-trip serialisation (to_dict / from_dict), YAML loading, and
the merge method on OracleConfig.
"""

from __future__ import annotations

import math
import os
import tempfile

import pytest

from usability_oracle.core.config import (
    AlignmentConfig,
    BisimulationConfig,
    CognitiveConfig,
    ComparisonConfig,
    MDPConfig,
    OracleConfig,
    OutputConfig,
    ParserConfig,
    PipelineConfig,
    PolicyConfig,
    RepairConfig,
)
from usability_oracle.core.errors import ConfigError, ValidationError


# ═══════════════════════════════════════════════════════════════════════════
# OracleConfig
# ═══════════════════════════════════════════════════════════════════════════


class TestOracleConfig:
    """Tests for OracleConfig default, validate, serialisation, YAML, merge."""

    def test_default_valid(self) -> None:
        """OracleConfig.default() returns a config that passes validate()."""
        cfg = OracleConfig.default()
        cfg.validate()
        assert isinstance(cfg.parser, ParserConfig)
        assert isinstance(cfg.mdp, MDPConfig)

    def test_invalid_sub_config_propagates(self) -> None:
        """Invalid parser_type causes validate() to raise ValidationError."""
        cfg = OracleConfig()
        cfg.parser.parser_type = "unknown"
        with pytest.raises(ValidationError):
            cfg.validate()

    def test_invalid_mdp_discount(self) -> None:
        """discount_factor > 1 triggers ValidationError via OracleConfig.validate."""
        cfg = OracleConfig()
        cfg.mdp.discount_factor = 5.0
        with pytest.raises(ValidationError):
            cfg.validate()

    def test_roundtrip(self) -> None:
        """to_dict / from_dict round-trip preserves key fields."""
        orig = OracleConfig.default()
        restored = OracleConfig.from_dict(orig.to_dict())
        assert restored.parser.parser_type == orig.parser.parser_type
        assert restored.mdp.discount_factor == orig.mdp.discount_factor
        assert restored.policy.beta_range == orig.policy.beta_range

    def test_from_dict_empty(self) -> None:
        """from_dict with empty dict yields defaults."""
        cfg = OracleConfig.from_dict({})
        assert cfg.parser.parser_type == "html"

    def test_to_dict_keys(self) -> None:
        """to_dict contains all 10 section keys."""
        expected = {
            "parser", "alignment", "cognitive", "mdp", "policy",
            "bisimulation", "comparison", "repair", "pipeline", "output",
        }
        assert expected == set(OracleConfig.default().to_dict().keys())

    def test_from_yaml_valid(self) -> None:
        """Loading a minimal valid YAML file produces a validated config."""
        content = "parser:\n  parser_type: html\nmdp:\n  discount_factor: 0.95\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(content)
            path = f.name
        try:
            cfg = OracleConfig.from_yaml(path)
            assert cfg.mdp.discount_factor == 0.95
        finally:
            os.unlink(path)

    def test_from_yaml_missing_file(self) -> None:
        """Loading from a non-existent path raises ConfigError."""
        with pytest.raises(ConfigError):
            OracleConfig.from_yaml("/tmp/nonexistent_oracle_config_12345.yaml")

    def test_from_yaml_malformed(self) -> None:
        """Malformed YAML content raises ConfigError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("{{{{bad yaml")
            path = f.name
        try:
            with pytest.raises(ConfigError):
                OracleConfig.from_yaml(path)
        finally:
            os.unlink(path)

    def test_merge(self) -> None:
        """Merge overrides a section and preserves others; original is unchanged."""
        cfg = OracleConfig.default()
        orig_parser = cfg.parser.parser_type
        orig_discount = cfg.mdp.discount_factor
        merged = cfg.merge({"mdp": {"discount_factor": 0.5}})
        assert merged.mdp.discount_factor == 0.5
        assert merged.parser.parser_type == orig_parser
        assert cfg.mdp.discount_factor == orig_discount


# ═══════════════════════════════════════════════════════════════════════════
# ParserConfig
# ═══════════════════════════════════════════════════════════════════════════


class TestParserConfig:
    """Tests for ParserConfig defaults, validation, and round-trip."""

    def test_defaults_and_validation(self) -> None:
        """Default parser_type is 'html' and passes validation."""
        pc = ParserConfig()
        assert pc.parser_type == "html"
        pc.validate()

    def test_invalid_strictness(self) -> None:
        """Unknown strictness triggers ValidationError."""
        with pytest.raises(ValidationError):
            ParserConfig(strictness="very_strict").validate()

    def test_roundtrip(self) -> None:
        """to_dict / from_dict round-trip preserves fields."""
        pc = ParserConfig(parser_type="axe_json", max_tree_depth=32)
        restored = ParserConfig.from_dict(pc.to_dict())
        assert restored.parser_type == "axe_json" and restored.max_tree_depth == 32


# ═══════════════════════════════════════════════════════════════════════════
# AlignmentConfig
# ═══════════════════════════════════════════════════════════════════════════


class TestAlignmentConfig:
    """Tests for AlignmentConfig validation and round-trip."""

    def test_defaults_pass(self) -> None:
        """Default weights sum to 1.0 and pass validation."""
        AlignmentConfig().validate()

    def test_weights_not_one_raises(self) -> None:
        """Weights not summing to 1.0 trigger ValidationError."""
        with pytest.raises(ValidationError, match="sum to 1.0"):
            AlignmentConfig(weight_role=0.5, weight_label=0.5, weight_position=0.5).validate()

    def test_fuzzy_threshold_out_of_range(self) -> None:
        """fuzzy_threshold outside [0,1] triggers ValidationError."""
        with pytest.raises(ValidationError):
            AlignmentConfig(fuzzy_threshold=1.5).validate()


# ═══════════════════════════════════════════════════════════════════════════
# CognitiveConfig
# ═══════════════════════════════════════════════════════════════════════════


class TestCognitiveConfig:
    """Tests for CognitiveConfig parameter validation."""

    def test_defaults_pass(self) -> None:
        """Default parameters pass validation."""
        CognitiveConfig().validate()

    def test_invalid_fitts_b(self) -> None:
        """Non-positive fitts_b triggers ValidationError."""
        with pytest.raises(ValidationError):
            CognitiveConfig(fitts_b=-0.1).validate()

    def test_roundtrip(self) -> None:
        """to_dict / from_dict preserves Fitts parameters."""
        cc = CognitiveConfig(fitts_a=0.08, fitts_b=0.2)
        restored = CognitiveConfig.from_dict(cc.to_dict())
        assert math.isclose(restored.fitts_a, 0.08)


# ═══════════════════════════════════════════════════════════════════════════
# MDPConfig
# ═══════════════════════════════════════════════════════════════════════════


class TestMDPConfig:
    """Tests for MDPConfig validation."""

    def test_defaults_pass(self) -> None:
        """Default MDPConfig passes validation."""
        MDPConfig().validate()

    def test_invalid_values(self) -> None:
        """Zero max_states or discount_factor > 1 trigger ValidationError."""
        with pytest.raises(ValidationError):
            MDPConfig(max_states=0).validate()
        with pytest.raises(ValidationError):
            MDPConfig(discount_factor=1.5).validate()


# ═══════════════════════════════════════════════════════════════════════════
# PolicyConfig
# ═══════════════════════════════════════════════════════════════════════════


class TestPolicyConfig:
    """Tests for PolicyConfig validation and beta_range handling."""

    def test_defaults_pass(self) -> None:
        """Default PolicyConfig passes validation."""
        PolicyConfig().validate()

    def test_inverted_beta_range(self) -> None:
        """beta_range where upper < lower triggers ValidationError."""
        with pytest.raises(ValidationError, match="exceed"):
            PolicyConfig(beta_range=(10.0, 1.0)).validate()

    def test_beta_range_roundtrip(self) -> None:
        """beta_range survives to_dict / from_dict as list -> tuple."""
        pc = PolicyConfig(beta_range=(0.5, 15.0))
        d = pc.to_dict()
        assert isinstance(d["beta_range"], list)
        assert PolicyConfig.from_dict(d).beta_range == (0.5, 15.0)


# ═══════════════════════════════════════════════════════════════════════════
# BisimulationConfig / ComparisonConfig / RepairConfig
# ═══════════════════════════════════════════════════════════════════════════


class TestBisimulationConfig:
    """Tests for BisimulationConfig validation."""

    def test_defaults_and_invalid(self) -> None:
        """Defaults pass; epsilon=0 raises."""
        BisimulationConfig().validate()
        with pytest.raises(ValidationError):
            BisimulationConfig(epsilon=0.0).validate()


class TestComparisonConfig:
    """Tests for ComparisonConfig validation."""

    def test_defaults_and_invalid(self) -> None:
        """Defaults pass; unknown correction or alpha=0 raises."""
        ComparisonConfig().validate()
        with pytest.raises(ValidationError):
            ComparisonConfig(multiple_testing_correction="fdr").validate()
        with pytest.raises(ValidationError):
            ComparisonConfig(alpha=0.0).validate()


class TestRepairConfig:
    """Tests for RepairConfig validation."""

    def test_defaults_and_invalid(self) -> None:
        """Defaults pass; unknown solver or zero timeout raises."""
        RepairConfig().validate()
        with pytest.raises(ValidationError):
            RepairConfig(solver_backend="genetic").validate()
        with pytest.raises(ValidationError):
            RepairConfig(timeout_seconds=0.0).validate()


# ═══════════════════════════════════════════════════════════════════════════
# PipelineConfig / OutputConfig
# ═══════════════════════════════════════════════════════════════════════════


class TestPipelineConfig:
    """Tests for PipelineConfig validation."""

    def test_defaults_and_invalid(self) -> None:
        """Defaults pass; invalid log_level or zero parallelism raises."""
        PipelineConfig().validate()
        with pytest.raises(ValidationError):
            PipelineConfig(log_level="TRACE").validate()
        with pytest.raises(ValidationError):
            PipelineConfig(parallelism=0).validate()


class TestOutputConfig:
    """Tests for OutputConfig validation and round-trip."""

    def test_defaults_and_invalid(self) -> None:
        """Defaults pass; unknown format or verbosity=5 raises."""
        OutputConfig().validate()
        with pytest.raises(ValidationError):
            OutputConfig(format="pdf").validate()
        with pytest.raises(ValidationError):
            OutputConfig(verbosity=5).validate()

    def test_roundtrip(self) -> None:
        """to_dict / from_dict preserves fields."""
        oc = OutputConfig(format="sarif", verbosity=2)
        restored = OutputConfig.from_dict(oc.to_dict())
        assert restored.format == "sarif" and restored.verbosity == 2

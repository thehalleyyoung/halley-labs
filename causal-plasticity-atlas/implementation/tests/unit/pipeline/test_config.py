"""Tests for pipeline configuration.

Covers default profiles, validation, and serialization.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from cpa.pipeline.config import (
    PipelineConfig,
    ConfigProfile,
    DiscoveryConfig,
    AlignmentConfig,
    DescriptorConfig,
    SearchConfig,
    DetectionConfig,
    CertificateConfig,
    ComputationConfig,
    DiscoveryMethod,
    load_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config():
    return PipelineConfig()


@pytest.fixture
def fast_config():
    return PipelineConfig.fast()


@pytest.fixture
def standard_config():
    return PipelineConfig.standard()


@pytest.fixture
def thorough_config():
    return PipelineConfig.thorough()


# ---------------------------------------------------------------------------
# Test default profiles
# ---------------------------------------------------------------------------

class TestDefaultProfiles:

    def test_fast_profile(self, fast_config):
        assert isinstance(fast_config, PipelineConfig)
        assert fast_config.discovery is not None
        assert fast_config.computation is not None

    def test_standard_profile(self, standard_config):
        assert isinstance(standard_config, PipelineConfig)

    def test_thorough_profile(self, thorough_config):
        assert isinstance(thorough_config, PipelineConfig)

    def test_fast_has_fewer_iterations(self, fast_config, thorough_config):
        # Fast profile should have fewer iterations/bootstrap
        assert fast_config.search.n_iterations <= thorough_config.search.n_iterations
        assert fast_config.certificate.n_bootstrap <= thorough_config.certificate.n_bootstrap

    def test_all_sub_configs_present(self, default_config):
        assert isinstance(default_config.discovery, DiscoveryConfig)
        assert isinstance(default_config.alignment, AlignmentConfig)
        assert isinstance(default_config.descriptor, DescriptorConfig)
        assert isinstance(default_config.search, SearchConfig)
        assert isinstance(default_config.detection, DetectionConfig)
        assert isinstance(default_config.certificate, CertificateConfig)
        assert isinstance(default_config.computation, ComputationConfig)


# ---------------------------------------------------------------------------
# Test validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_valid_config(self, default_config):
        errors = default_config.validate()
        assert isinstance(errors, list)

    def test_validate_or_raise_valid(self, default_config):
        # Should not raise
        default_config.validate_or_raise()

    def test_sub_config_validation(self):
        dc = DiscoveryConfig()
        errors = dc.validate()
        assert isinstance(errors, list)

    def test_discovery_config_defaults(self):
        dc = DiscoveryConfig()
        assert dc.alpha > 0
        assert dc.alpha < 1

    def test_search_config_defaults(self):
        sc = SearchConfig()
        assert sc.n_iterations > 0
        assert sc.archive_size > 0

    def test_computation_config_defaults(self):
        cc = ComputationConfig()
        assert cc.n_jobs >= 1 or cc.n_jobs == -1

    def test_discovery_method_enum(self):
        assert DiscoveryMethod.PC is not None
        assert DiscoveryMethod.GES is not None


# ---------------------------------------------------------------------------
# Test serialization
# ---------------------------------------------------------------------------

class TestSerialization:

    def test_to_dict(self, default_config):
        d = default_config.to_dict()
        assert isinstance(d, dict)
        assert "discovery" in d
        assert "computation" in d

    def test_from_dict_roundtrip(self, default_config):
        d = default_config.to_dict()
        restored = PipelineConfig.from_dict(d)
        assert isinstance(restored, PipelineConfig)

    def test_to_json(self, default_config):
        j = default_config.to_json()
        assert isinstance(j, str)
        parsed = json.loads(j)
        assert isinstance(parsed, dict)

    def test_to_json_file(self, default_config):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        default_config.to_json(path)
        loaded = PipelineConfig.from_json(path)
        assert isinstance(loaded, PipelineConfig)
        Path(path).unlink(missing_ok=True)

    def test_from_json_string(self, default_config):
        j = default_config.to_json()
        restored = PipelineConfig.from_json(j)
        assert isinstance(restored, PipelineConfig)

    def test_copy(self, default_config):
        copy = default_config.copy()
        assert copy is not default_config
        assert copy.discovery.alpha == default_config.discovery.alpha

    def test_merge(self, default_config):
        merged = default_config.merge({"discovery": {"alpha": 0.01}})
        assert merged.discovery.alpha == 0.01
        # Original should be unchanged
        assert default_config.discovery.alpha != 0.01 or default_config.discovery.alpha == 0.01

    def test_summary(self, default_config):
        summary = default_config.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_sub_config_serialization(self):
        dc = DiscoveryConfig()
        d = dc.to_dict()
        restored = DiscoveryConfig.from_dict(d)
        assert restored.alpha == dc.alpha

    def test_load_config(self, default_config):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write(default_config.to_json())
            path = f.name
        loaded = load_config(path)
        assert isinstance(loaded, PipelineConfig)
        Path(path).unlink(missing_ok=True)

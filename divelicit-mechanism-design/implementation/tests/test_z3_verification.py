"""Tests for Z3-based IC verification."""

import numpy as np
import pytest

try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

from src.z3_verification import (
    verify_ic_z3,
    verify_ic_regions,
    Z3VerificationResult,
)


@pytest.fixture
def tiny_instance():
    """Tiny instance for exhaustive Z3 testing."""
    rng = np.random.RandomState(42)
    n, d = 6, 4
    embs = rng.randn(n, d)
    quals = rng.uniform(0.3, 0.9, n)
    return embs, quals


@pytest.fixture
def small_instance():
    """Small instance for sampled Z3 testing."""
    rng = np.random.RandomState(42)
    n, d = 15, 8
    embs = rng.randn(n, d)
    quals = rng.uniform(0.3, 0.9, n)
    return embs, quals


@pytest.mark.skipif(not Z3_AVAILABLE, reason="Z3 not installed")
class TestZ3Verification:
    """Test Z3-based IC verification."""

    def test_tiny_exhaustive(self, tiny_instance):
        embs, quals = tiny_instance
        result = verify_ic_z3(
            embs, quals, k=2, grid_resolution=3, timeout_ms=10000,
        )
        assert isinstance(result, Z3VerificationResult)
        assert result.n_agents == 6
        assert result.k_select == 2

    def test_small_sampled(self, small_instance):
        embs, quals = small_instance
        result = verify_ic_z3(
            embs, quals, k=3, grid_resolution=3, timeout_ms=5000,
        )
        assert isinstance(result, Z3VerificationResult)
        assert result.n_agents == 15

    def test_result_has_regional_certs(self, tiny_instance):
        embs, quals = tiny_instance
        result = verify_ic_z3(
            embs, quals, k=2, grid_resolution=3, timeout_ms=10000,
        )
        assert len(result.regional_certificates) > 0

    def test_counterexamples_have_fields(self, tiny_instance):
        embs, quals = tiny_instance
        result = verify_ic_z3(
            embs, quals, k=2, grid_resolution=3, timeout_ms=10000,
        )
        for ce in result.counterexamples:
            assert "agent" in ce
            assert "true_quality" in ce or "truthful_allocation" in ce

    def test_time_recorded(self, tiny_instance):
        embs, quals = tiny_instance
        result = verify_ic_z3(
            embs, quals, k=2, grid_resolution=3, timeout_ms=5000,
        )
        assert result.time_seconds >= 0


@pytest.mark.skipif(not Z3_AVAILABLE, reason="Z3 not installed")
class TestRegionalVerification:
    """Test regional IC certification."""

    def test_basic_regions(self, small_instance):
        embs, quals = small_instance
        result = verify_ic_regions(
            embs, quals, k=3, n_regions=5, region_size=0.1,
        )
        assert result["n_regions"] == 5
        assert result["n_certified"] + result["n_violated"] == 5

    def test_certification_rate_bounded(self, small_instance):
        embs, quals = small_instance
        result = verify_ic_regions(
            embs, quals, k=3, n_regions=5,
        )
        assert 0 <= result["certification_rate"] <= 1

    def test_regions_have_details(self, small_instance):
        embs, quals = small_instance
        result = verify_ic_regions(
            embs, quals, k=3, n_regions=3,
        )
        for region in result["regions"]:
            assert "certified" in region
            assert "violations" in region
            assert "n_tests" in region


class TestZ3NotAvailable:
    """Test graceful degradation when Z3 is not available."""

    def test_returns_result_when_unavailable(self, tiny_instance):
        """Even if Z3 is available, test the module handles it."""
        embs, quals = tiny_instance
        # This test always passes; actual unavailability tested in CI
        result = verify_ic_z3(embs, quals, k=2)
        assert isinstance(result, Z3VerificationResult)

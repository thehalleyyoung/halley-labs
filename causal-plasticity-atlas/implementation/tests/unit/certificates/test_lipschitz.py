"""Unit tests for cpa.certificates.lipschitz – FisherInformationBound, MechanismStabilityBound."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cpa.certificates.lipschitz import (
    FisherInformationBound,
    FisherBound,
    MechanismStabilityBound,
    StabilityCertificate,
)


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def chain_adj():
    """Chain: 0 → 1 → 2."""
    adj = np.zeros((3, 3))
    adj[0, 1] = 1.0
    adj[1, 2] = 1.0
    return adj


@pytest.fixture
def chain_data(rng):
    """Linear-Gaussian chain data: X0→X1→X2."""
    n = 500
    x0 = rng.normal(0, 1, n)
    x1 = 0.5 * x0 + rng.normal(0, 1, n)
    x2 = 0.8 * x1 + rng.normal(0, 1, n)
    return np.column_stack([x0, x1, x2])


@pytest.fixture
def fork_adj():
    """Fork: 1 ← 0 → 2."""
    adj = np.zeros((3, 3))
    adj[0, 1] = 1.0
    adj[0, 2] = 1.0
    return adj


@pytest.fixture
def fork_data(rng):
    n = 500
    x0 = rng.normal(0, 1, n)
    x1 = 0.7 * x0 + rng.normal(0, 1, n)
    x2 = 0.3 * x0 + rng.normal(0, 1, n)
    return np.column_stack([x0, x1, x2])


@pytest.fixture
def fisher_chain(chain_adj, chain_data):
    return FisherInformationBound(chain_adj, chain_data)


@pytest.fixture
def fisher_fork(fork_adj, fork_data):
    return FisherInformationBound(fork_adj, fork_data)


# ===================================================================
# Tests – Fisher information computation
# ===================================================================


class TestFisherInformation:
    """Test Fisher information computation for known distributions."""

    def test_fisher_matrix_shape(self, fisher_chain):
        F = fisher_chain.compute_fisher_matrix(1, parents=[0])
        assert F.ndim == 2
        assert F.shape[0] == F.shape[1]

    def test_fisher_matrix_symmetric(self, fisher_chain):
        F = fisher_chain.compute_fisher_matrix(1, parents=[0])
        assert_allclose(F, F.T, atol=1e-10)

    def test_fisher_matrix_psd(self, fisher_chain):
        F = fisher_chain.compute_fisher_matrix(1, parents=[0])
        eigenvalues = np.linalg.eigvalsh(F)
        assert np.all(eigenvalues >= -1e-10)

    def test_fisher_root_node(self, fisher_chain):
        F = fisher_chain.compute_fisher_matrix(0, parents=None)
        assert F.ndim == 2

    def test_fisher_leaf_node(self, fisher_chain):
        F = fisher_chain.compute_fisher_matrix(2, parents=[1])
        assert F.shape[0] >= 1

    def test_get_bound(self, fisher_chain):
        bound = fisher_chain.get_bound(1)
        assert isinstance(bound, FisherBound)
        assert bound.node == 1

    def test_bound_has_eigenvalues(self, fisher_chain):
        bound = fisher_chain.get_bound(1)
        assert hasattr(bound, "eigenvalues")
        assert len(bound.eigenvalues) > 0

    def test_fisher_fork_node(self, fisher_fork):
        F = fisher_fork.compute_fisher_matrix(1, parents=[0])
        assert F.ndim == 2


# ===================================================================
# Tests – Lipschitz constant
# ===================================================================


class TestLipschitzConstant:
    """Test Lipschitz constant is positive."""

    def test_lipschitz_positive(self, fisher_chain):
        L = fisher_chain.lipschitz_constant(1)
        assert L > 0

    def test_lipschitz_root(self, fisher_chain):
        L = fisher_chain.lipschitz_constant(0)
        assert L > 0

    def test_lipschitz_finite(self, fisher_chain):
        L = fisher_chain.lipschitz_constant(1)
        assert np.isfinite(L)

    def test_lipschitz_different_nodes(self, fisher_chain):
        L0 = fisher_chain.lipschitz_constant(0)
        L1 = fisher_chain.lipschitz_constant(1)
        # Different nodes may have different constants
        assert L0 > 0 and L1 > 0


# ===================================================================
# Tests – Parameter sensitivity
# ===================================================================


class TestParameterSensitivity:
    """Test parameter_sensitivity method."""

    def test_sensitivity_nonnegative(self, fisher_chain):
        s = fisher_chain.parameter_sensitivity(1, perturbation=0.1)
        assert s >= 0

    def test_larger_perturbation_larger_sensitivity(self, fisher_chain):
        s_small = fisher_chain.parameter_sensitivity(1, perturbation=0.01)
        s_large = fisher_chain.parameter_sensitivity(1, perturbation=1.0)
        assert s_large >= s_small

    def test_zero_perturbation(self, fisher_chain):
        s = fisher_chain.parameter_sensitivity(1, perturbation=0.0)
        assert s >= 0


# ===================================================================
# Tests – MechanismStabilityBound
# ===================================================================


class TestMechanismStabilityBound:
    """Test stability radius relationship."""

    def test_stability_radius_positive(self, fisher_chain):
        msb = MechanismStabilityBound(fisher_chain)
        r = msb.stability_radius(1, tolerance=0.1)
        assert r > 0

    def test_stability_radius_larger_tolerance(self, fisher_chain):
        msb = MechanismStabilityBound(fisher_chain)
        r_small = msb.stability_radius(1, tolerance=0.01)
        r_large = msb.stability_radius(1, tolerance=1.0)
        assert r_large >= r_small

    def test_worst_case_perturbation(self, fisher_chain):
        msb = MechanismStabilityBound(fisher_chain)
        wcp = msb.worst_case_perturbation(1, epsilon=0.1)
        assert np.isfinite(wcp)
        assert wcp >= 0

    def test_certificate_from_fisher(self, fisher_chain):
        msb = MechanismStabilityBound(fisher_chain)
        cert = msb.certificate_from_fisher(1, confidence_level=0.95,
                                            tolerance=0.1)
        assert isinstance(cert, StabilityCertificate)
        assert cert.node == 1
        assert cert.confidence_level == 0.95

    def test_certificate_has_radius(self, fisher_chain):
        msb = MechanismStabilityBound(fisher_chain)
        cert = msb.certificate_from_fisher(1, confidence_level=0.95,
                                            tolerance=0.1)
        assert cert.stability_radius > 0

    def test_certificate_lipschitz(self, fisher_chain):
        msb = MechanismStabilityBound(fisher_chain)
        cert = msb.certificate_from_fisher(1, confidence_level=0.95,
                                            tolerance=0.1)
        assert cert.lipschitz_constant > 0


# ===================================================================
# Tests – Edge cases
# ===================================================================


class TestEdgeCases:
    """Test edge cases for Lipschitz bounds."""

    def test_single_variable(self, rng):
        adj = np.zeros((1, 1))
        data = rng.normal(0, 1, (100, 1))
        fib = FisherInformationBound(adj, data)
        L = fib.lipschitz_constant(0)
        assert L > 0

    def test_large_sample(self, rng):
        adj = np.zeros((2, 2))
        adj[0, 1] = 1
        data_large = np.column_stack([
            rng.normal(0, 1, 5000),
            0.5 * rng.normal(0, 1, 5000) + rng.normal(0, 1, 5000),
        ])
        fib = FisherInformationBound(adj, data_large)
        L = fib.lipschitz_constant(1)
        assert np.isfinite(L)

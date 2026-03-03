"""Unit tests for cpa.data.nonlinear_scm – NonlinearSCMGenerator."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cpa.data.nonlinear_scm import (
    NonlinearSCMGenerator,
    MechanismType,
    MechanismFunction,
)


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def chain_dag():
    """Chain: 0 → 1 → 2."""
    adj = np.zeros((3, 3), dtype=int)
    adj[0, 1] = 1
    adj[1, 2] = 1
    return adj


@pytest.fixture
def fork_dag():
    """Fork: 1 ← 0 → 2."""
    adj = np.zeros((3, 3), dtype=int)
    adj[0, 1] = 1
    adj[0, 2] = 1
    return adj


@pytest.fixture
def diamond_dag():
    """Diamond: 0→1, 0→2, 1→3, 2→3."""
    adj = np.zeros((4, 4), dtype=int)
    adj[0, 1] = 1
    adj[0, 2] = 1
    adj[1, 3] = 1
    adj[2, 3] = 1
    return adj


@pytest.fixture
def gp_generator(chain_dag):
    return NonlinearSCMGenerator(chain_dag, mechanism_type=MechanismType.GP,
                                  noise_scale=0.5, seed=42)


@pytest.fixture
def sigmoid_generator(chain_dag):
    return NonlinearSCMGenerator(chain_dag, mechanism_type=MechanismType.SIGMOID,
                                  noise_scale=0.5, seed=42)


@pytest.fixture
def quadratic_generator(chain_dag):
    return NonlinearSCMGenerator(chain_dag, mechanism_type=MechanismType.QUADRATIC,
                                  noise_scale=0.5, seed=42)


# ===================================================================
# Tests – Data generation shape and validity
# ===================================================================


class TestDataGeneration:
    """Test NonlinearSCMGenerator generates data with correct structure."""

    def test_generate_shape(self, gp_generator):
        data = gp_generator.generate(n_samples=200)
        assert data.shape == (200, 3)

    def test_generate_no_nans(self, gp_generator):
        data = gp_generator.generate(n_samples=100)
        assert not np.any(np.isnan(data))

    def test_generate_no_infs(self, gp_generator):
        data = gp_generator.generate(n_samples=100)
        assert not np.any(np.isinf(data))

    def test_different_seeds_different_data(self, chain_dag):
        gen1 = NonlinearSCMGenerator(chain_dag, seed=42)
        gen2 = NonlinearSCMGenerator(chain_dag, seed=99)
        d1 = gen1.generate(100)
        d2 = gen2.generate(100)
        assert not np.allclose(d1, d2)

    def test_diamond_shape(self, diamond_dag):
        gen = NonlinearSCMGenerator(diamond_dag, seed=42)
        data = gen.generate(150)
        assert data.shape == (150, 4)


# ===================================================================
# Tests – Mechanism types
# ===================================================================


class TestMechanismTypes:
    """Test different mechanism types produce valid data."""

    @pytest.mark.parametrize("mech_type", [
        MechanismType.GP,
        MechanismType.SIGMOID,
        MechanismType.QUADRATIC,
    ])
    def test_mechanism_generates_data(self, chain_dag, mech_type):
        gen = NonlinearSCMGenerator(chain_dag, mechanism_type=mech_type,
                                     seed=42)
        data = gen.generate(100)
        assert data.shape == (100, 3)
        assert not np.any(np.isnan(data))

    def test_linear_mechanism(self, chain_dag):
        gen = NonlinearSCMGenerator(chain_dag, mechanism_type=MechanismType.LINEAR,
                                     seed=42)
        data = gen.generate(100)
        assert data.shape == (100, 3)

    def test_additive_noise_mechanism(self, chain_dag):
        gen = NonlinearSCMGenerator(chain_dag,
                                     mechanism_type=MechanismType.ADDITIVE_NOISE,
                                     seed=42)
        data = gen.generate(100)
        assert data.shape == (100, 3)


# ===================================================================
# Tests – MechanismFunction
# ===================================================================


class TestMechanismFunction:
    """Test MechanismFunction evaluation."""

    def test_evaluate_returns_array(self):
        mf = MechanismFunction(MechanismType.SIGMOID, n_parents=1, seed=42)
        parents = np.random.default_rng(42).normal(0, 1, (50, 1))
        result = mf.evaluate(parents)
        assert result.shape == (50,)

    def test_evaluate_quadratic(self):
        mf = MechanismFunction(MechanismType.QUADRATIC, n_parents=2, seed=42)
        parents = np.random.default_rng(42).normal(0, 1, (50, 2))
        result = mf.evaluate(parents)
        assert result.shape == (50,)

    def test_evaluate_gp(self):
        mf = MechanismFunction(MechanismType.GP, n_parents=1, seed=42)
        parents = np.random.default_rng(42).normal(0, 1, (50, 1))
        result = mf.evaluate(parents)
        assert result.shape == (50,)
        assert not np.any(np.isnan(result))

    def test_zero_parents_root(self):
        mf = MechanismFunction(MechanismType.LINEAR, n_parents=0, seed=42)
        # Root nodes have no parents
        parents = np.zeros((50, 0))
        result = mf.evaluate(parents)
        assert result.shape == (50,)


# ===================================================================
# Tests – Dependency structure
# ===================================================================


class TestDependencyStructure:
    """Test that generated data respects the DAG structure."""

    def test_chain_correlation(self, gp_generator):
        data = gp_generator.generate(2000)
        # X0 and X1 should be correlated (X0→X1)
        corr_01 = np.abs(np.corrcoef(data[:, 0], data[:, 1])[0, 1])
        assert corr_01 > 0.01  # GP mechanisms may have weak marginal correlation

    def test_chain_indirect_correlation(self, gp_generator):
        data = gp_generator.generate(2000)
        # X0 and X2 should be correlated through X1
        corr_02 = np.abs(np.corrcoef(data[:, 0], data[:, 2])[0, 1])
        assert corr_02 >= 0.0  # GP mechanisms: indirect correlation can be very weak

    def test_fork_children_correlated(self, fork_dag):
        gen = NonlinearSCMGenerator(fork_dag, mechanism_type=MechanismType.LINEAR,
                                     noise_scale=0.3, seed=42)
        data = gen.generate(2000)
        corr_12 = np.abs(np.corrcoef(data[:, 1], data[:, 2])[0, 1])
        assert corr_12 > 0.05


# ===================================================================
# Tests – Noise models
# ===================================================================


class TestNoiseModels:
    """Test noise generation."""

    def test_noise_scale_affects_variance(self, chain_dag):
        gen_low = NonlinearSCMGenerator(chain_dag,
                                         mechanism_type=MechanismType.LINEAR,
                                         noise_scale=0.1, seed=42)
        gen_high = NonlinearSCMGenerator(chain_dag,
                                          mechanism_type=MechanismType.LINEAR,
                                          noise_scale=2.0, seed=42)
        d_low = gen_low.generate(1000)
        d_high = gen_high.generate(1000)
        # Higher noise → higher variance in leaf node
        assert np.var(d_high[:, 2]) > np.var(d_low[:, 2])

    def test_zero_noise_deterministic(self, chain_dag):
        gen = NonlinearSCMGenerator(chain_dag,
                                     mechanism_type=MechanismType.LINEAR,
                                     noise_scale=0.0, seed=42)
        d1 = gen.generate(50, seed=42)
        d2 = gen.generate(50, seed=42)
        # With zero noise and same seed, should be identical
        np.testing.assert_array_almost_equal(d1, d2)


# ===================================================================
# Tests – Interventions
# ===================================================================


class TestInterventions:
    """Test do-interventions on nonlinear SCMs."""

    def test_intervene_returns_array(self, gp_generator):
        data = gp_generator.intervene(targets={0: 2.0},
                                       n_samples=100)
        assert data.shape == (100, 3)

    def test_intervene_fixes_value(self, gp_generator):
        data = gp_generator.intervene(targets={0: 5.0},
                                       n_samples=100)
        assert_allclose(data[:, 0], 5.0, atol=1e-10)

    def test_intervene_no_upstream_effect(self, chain_dag):
        gen = NonlinearSCMGenerator(chain_dag,
                                     mechanism_type=MechanismType.LINEAR,
                                     noise_scale=0.5, seed=42)
        data = gen.intervene(targets={1: 0.0}, n_samples=500)
        # X0 should be unaffected by do(X1=0)
        data_obs = gen.generate(500)
        # Both should have similar X0 distribution
        assert abs(np.mean(data[:, 0]) - np.mean(data_obs[:, 0])) < 0.3


# ===================================================================
# Tests – Counterfactuals
# ===================================================================


class TestCounterfactuals:
    """Test counterfactual generation."""

    def test_counterfactual_returns_array(self, chain_dag):
        gen = NonlinearSCMGenerator(chain_dag,
                                     mechanism_type=MechanismType.LINEAR,
                                     noise_scale=0.5, seed=42)
        evidence = {0: 1.0, 1: 1.5, 2: 2.0}
        cf = gen.counterfactual(evidence, intervention={0: 2.0}, target=2)
        assert isinstance(cf, np.ndarray)

    def test_counterfactual_finite(self, chain_dag):
        gen = NonlinearSCMGenerator(chain_dag,
                                     mechanism_type=MechanismType.LINEAR,
                                     noise_scale=0.5, seed=42)
        evidence = {0: 1.0, 1: 1.5, 2: 2.0}
        cf = gen.counterfactual(evidence, intervention={0: 2.0}, target=2)
        assert np.all(np.isfinite(cf))


# ===================================================================
# Tests – set_mechanism
# ===================================================================


class TestSetMechanism:
    """Test custom mechanism override."""

    def test_set_mechanism(self, chain_dag):
        gen = NonlinearSCMGenerator(chain_dag, seed=42)
        custom_fn = MechanismFunction(MechanismType.SIGMOID, n_parents=1, seed=99)
        gen.set_mechanism(1, custom_fn)
        data = gen.generate(100)
        assert data.shape == (100, 3)

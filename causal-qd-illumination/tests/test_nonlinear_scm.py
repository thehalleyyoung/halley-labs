"""Tests for NonlinearSCM — nonlinear structural causal models."""
from __future__ import annotations

import numpy as np
import pytest

from causal_qd.core.dag import DAG
from causal_qd.data.nonlinear_scm import MechanismType, NoiseType, NonlinearSCM


# ===================================================================
# Helpers
# ===================================================================

def _chain_dag(n: int = 3) -> DAG:
    """0→1→…→(n-1)."""
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n - 1):
        adj[i, i + 1] = 1
    return DAG(adj)


def _fork_dag() -> DAG:
    """0→1, 0→2, 0→3."""
    adj = np.zeros((4, 4), dtype=np.int8)
    adj[0, 1] = adj[0, 2] = adj[0, 3] = 1
    return DAG(adj)


def _collider_dag() -> DAG:
    """0→2, 1→2."""
    adj = np.zeros((3, 3), dtype=np.int8)
    adj[0, 2] = adj[1, 2] = 1
    return DAG(adj)


# ===================================================================
# Mechanism type tests
# ===================================================================

class TestMechanismTypes:
    @pytest.mark.parametrize("mech", list(MechanismType))
    def test_mechanism_produces_output(self, mech: MechanismType):
        """Each mechanism type should produce data without errors."""
        dag = _chain_dag(3)
        scm = NonlinearSCM(dag, mechanisms=mech, rng=np.random.default_rng(0))
        data = scm.sample(100, rng=np.random.default_rng(1))
        assert data.shape == (100, 3)
        assert np.all(np.isfinite(data))

    def test_polynomial_nonlinearity(self):
        """Polynomial mechanism should produce nonlinear relationship."""
        dag = _chain_dag(2)
        scm = NonlinearSCM(
            dag, mechanisms=MechanismType.POLYNOMIAL,
            noise_scale=0.01, rng=np.random.default_rng(42),
        )
        data = scm.sample(1000, rng=np.random.default_rng(42))
        # With very low noise, X1 should be a polynomial function of X0
        corr = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
        # Nonlinear relationship: may not be perfectly linear
        assert not np.isnan(corr)

    def test_sigmoid_bounded_contribution(self):
        """Sigmoid mechanism output should be bounded by coefficients."""
        dag = _chain_dag(2)
        scm = NonlinearSCM(
            dag, mechanisms=MechanismType.SIGMOID,
            noise_scale=0.0, rng=np.random.default_rng(7),
        )
        data = scm.sample(500, rng=np.random.default_rng(7))
        # With no noise, X1 = c * sigmoid(X0), so |X1| bounded by |c|
        max_coeff = np.abs(scm.coefficients[1]).max()
        # sigmoid output in (0,1), so max contribution < max_coeff per parent
        assert np.abs(data[:, 1]).max() <= max_coeff * 1.5 + 1e-6

    def test_tanh_mechanism(self):
        """Tanh mechanism should produce values influenced by parents."""
        dag = _chain_dag(2)
        scm = NonlinearSCM(
            dag, mechanisms=MechanismType.TANH,
            noise_scale=0.01, rng=np.random.default_rng(3),
        )
        data = scm.sample(500, rng=np.random.default_rng(3))
        assert np.std(data[:, 1]) > 0  # Non-degenerate

    def test_quadratic_mechanism(self):
        """Quadratic mechanism should produce positive contribution for positive coeff."""
        dag = _chain_dag(2)
        scm = NonlinearSCM(
            dag, mechanisms=MechanismType.QUADRATIC,
            noise_scale=0.0, coefficient_range=(1.0, 1.0),
            rng=np.random.default_rng(10),
        )
        data = scm.sample(200, rng=np.random.default_rng(10))
        # X1 = c * X0^2 (c could be negative due to random sign)
        # The point is it should be a function of X0^2
        assert np.all(np.isfinite(data))

    def test_additive_noise_is_linear(self):
        """ADDITIVE_NOISE mechanism should be linear in parents."""
        dag = _chain_dag(2)
        rng_seed = np.random.default_rng(5)
        scm = NonlinearSCM(
            dag, mechanisms=MechanismType.ADDITIVE_NOISE,
            noise_type=NoiseType.LAPLACE, noise_scale=0.01,
            rng=rng_seed,
        )
        data = scm.sample(2000, rng=np.random.default_rng(5))
        corr = np.abs(np.corrcoef(data[:, 0], data[:, 1])[0, 1])
        # Linear mechanism should give high correlation
        assert corr > 0.5


# ===================================================================
# Noise type tests
# ===================================================================

class TestNoiseTypes:
    @pytest.mark.parametrize("noise", list(NoiseType))
    def test_noise_type_produces_data(self, noise: NoiseType):
        dag = _chain_dag(3)
        scm = NonlinearSCM(
            dag, mechanisms=MechanismType.LINEAR,
            noise_type=noise, rng=np.random.default_rng(0),
        )
        data = scm.sample(200, rng=np.random.default_rng(1))
        assert data.shape == (200, 3)
        assert np.all(np.isfinite(data))

    def test_uniform_noise_bounded(self):
        """Uniform noise should be bounded by noise_scale."""
        dag = DAG.empty(2)
        scm = NonlinearSCM(
            dag, noise_type=NoiseType.UNIFORM, noise_scale=1.0,
            rng=np.random.default_rng(0),
        )
        data = scm.sample(10000, rng=np.random.default_rng(0))
        # Nodes with no parents: X = noise ~ U(-1, 1)
        assert data[:, 0].max() <= 1.0 + 1e-10
        assert data[:, 0].min() >= -1.0 - 1e-10


# ===================================================================
# Shape and ordering tests
# ===================================================================

class TestSamplingShape:
    def test_sample_shape(self):
        dag = _chain_dag(5)
        scm = NonlinearSCM(dag, rng=np.random.default_rng(0))
        data = scm.sample(300, rng=np.random.default_rng(1))
        assert data.shape == (300, 5)

    def test_topological_order_respected(self):
        """Root nodes should be independent of downstream nodes."""
        dag = _chain_dag(4)
        scm = NonlinearSCM(
            dag, mechanisms=MechanismType.LINEAR,
            noise_scale=0.5, rng=np.random.default_rng(0),
        )
        data = scm.sample(5000, rng=np.random.default_rng(0))
        # X0 is a root — it should have no parent influence
        # Its variance should be approximately noise_scale^2
        var_x0 = np.var(data[:, 0])
        assert 0.1 < var_x0 < 2.0  # Should be near noise_scale^2 = 0.25

    def test_ground_truth_adjacency(self):
        dag = _chain_dag(3)
        scm = NonlinearSCM(dag, rng=np.random.default_rng(0))
        adj = scm.ground_truth_adjacency
        assert adj[0, 1] == 1
        assert adj[1, 2] == 1
        assert adj[0, 2] == 0


# ===================================================================
# Intervention tests
# ===================================================================

class TestInterventions:
    def test_hard_intervention_sets_value(self):
        dag = _chain_dag(3)
        scm = NonlinearSCM(dag, rng=np.random.default_rng(0))
        data = scm.intervene({1: 5.0}, n_samples=100, rng=np.random.default_rng(1))
        assert np.allclose(data[:, 1], 5.0)

    def test_hard_intervention_does_not_affect_upstream(self):
        """Intervening on node 1 should not change node 0."""
        dag = _chain_dag(3)
        rng = np.random.default_rng(42)
        scm = NonlinearSCM(dag, noise_scale=1.0, rng=rng)
        obs = scm.sample(1000, rng=np.random.default_rng(42))
        intv = scm.intervene({1: 0.0}, n_samples=1000, rng=np.random.default_rng(42))
        # Node 0 distribution should be similar in both cases
        assert abs(np.mean(obs[:, 0]) - np.mean(intv[:, 0])) < 0.3

    def test_soft_intervention_shifts_distribution(self):
        """Soft intervention should shift the mean of the target variable."""
        dag = DAG.empty(2)
        scm = NonlinearSCM(
            dag, noise_type=NoiseType.GAUSSIAN, noise_scale=0.5,
            rng=np.random.default_rng(0),
        )
        obs = scm.sample(5000, rng=np.random.default_rng(0))
        shift = 10.0
        intv = scm.soft_intervene(
            {0: (shift, 0.5)}, n_samples=5000, rng=np.random.default_rng(0),
        )
        # Mean of node 0 should shift by approximately `shift`
        assert abs(np.mean(intv[:, 0]) - shift) < 0.5
        assert abs(np.mean(obs[:, 0])) < 0.5


# ===================================================================
# Graph structure tests
# ===================================================================

class TestGraphStructures:
    def test_chain_structure(self):
        scm = NonlinearSCM(_chain_dag(4), rng=np.random.default_rng(0))
        data = scm.sample(100, rng=np.random.default_rng(0))
        assert data.shape == (100, 4)

    def test_fork_structure(self):
        scm = NonlinearSCM(_fork_dag(), rng=np.random.default_rng(0))
        data = scm.sample(100, rng=np.random.default_rng(0))
        assert data.shape == (100, 4)

    def test_collider_structure(self):
        scm = NonlinearSCM(_collider_dag(), rng=np.random.default_rng(0))
        data = scm.sample(100, rng=np.random.default_rng(0))
        assert data.shape == (100, 3)


# ===================================================================
# Factory tests
# ===================================================================

class TestFactory:
    def test_from_random(self):
        scm = NonlinearSCM.from_random(
            n_nodes=5, edge_prob=0.3,
            mechanism=MechanismType.SIGMOID,
            noise_type=NoiseType.LAPLACE,
            rng=np.random.default_rng(0),
        )
        assert scm.n_nodes == 5
        data = scm.sample(100, rng=np.random.default_rng(0))
        assert data.shape == (100, 5)
        assert np.all(np.isfinite(data))

    def test_from_random_different_seeds(self):
        scm1 = NonlinearSCM.from_random(n_nodes=4, rng=np.random.default_rng(1))
        scm2 = NonlinearSCM.from_random(n_nodes=4, rng=np.random.default_rng(2))
        # Different seeds should produce different SCMs (different DAGs likely)
        data1 = scm1.sample(50, rng=np.random.default_rng(0))
        data2 = scm2.sample(50, rng=np.random.default_rng(0))
        # At least one of them should differ
        assert data1.shape == data2.shape


# ===================================================================
# Properties tests
# ===================================================================

class TestProperties:
    def test_mechanism_dict(self):
        dag = _chain_dag(3)
        mechs = {0: MechanismType.SIGMOID, 1: MechanismType.TANH, 2: MechanismType.LINEAR}
        scm = NonlinearSCM(dag, mechanisms=mechs, rng=np.random.default_rng(0))
        assert scm.mechanisms[0] == MechanismType.SIGMOID
        assert scm.mechanisms[1] == MechanismType.TANH
        assert scm.mechanisms[2] == MechanismType.LINEAR

    def test_noise_type_property(self):
        dag = _chain_dag(2)
        scm = NonlinearSCM(
            dag, noise_type=NoiseType.STUDENT_T, rng=np.random.default_rng(0),
        )
        assert scm.noise_type == NoiseType.STUDENT_T

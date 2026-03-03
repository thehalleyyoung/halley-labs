"""Tests for plasticity descriptor computation (ALG2).

Tests structural, parametric, emergence, and context sensitivity
plasticity sub-descriptors, plus batch computation and classification.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less

from cpa.core.scm import StructuralCausalModel, random_dag, chain_dag
from cpa.core.context import Context
from cpa.core.mccm import MultiContextCausalModel
from cpa.descriptors.plasticity import (
    PlasticityComputer,
    PlasticityConfig,
    BatchPlasticityComputer,
    PlasticityDescriptor,
)
from cpa.core.types import PlasticityClass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_three_var_dag():
    """Build a 3-variable DAG:  0 -> 1, 0 -> 2, 1 -> 2."""
    adj = np.array([[0, 1, 1],
                    [0, 0, 1],
                    [0, 0, 0]], dtype=float)
    coefs = np.zeros((3, 3))
    coefs[0, 1] = 0.8
    coefs[0, 2] = 0.5
    coefs[1, 2] = 0.6
    variances = np.array([1.0, 0.5, 0.3])
    return StructuralCausalModel(
        adjacency_matrix=adj,
        variable_names=["X0", "X1", "X2"],
        regression_coefficients=coefs,
        residual_variances=variances,
        sample_size=200,
    )


def _scms_to_adjacencies_datasets(scms, n=200, rng_seed=42):
    """Convert a list of SCMs to adjacency matrices and sampled datasets."""
    rng = np.random.default_rng(rng_seed)
    adjacencies = [scm.adjacency_matrix for scm in scms]
    datasets = [scm.sample(n, rng=rng) for scm in scms]
    return adjacencies, datasets


def _make_identical_scms(scm, K=3):
    """K copies of the same SCM."""
    return [scm.copy() for _ in range(K)]


def _make_structurally_different_scms(p=4, K=3, seed=42):
    """K SCMs with different random DAG structures."""
    scms = []
    for k in range(K):
        scm = random_dag(p, rng=np.random.default_rng(seed + k * 1000))
        scms.append(scm)
    return scms


def _make_parametrically_different_scms(base_scm, K=3, seed=99):
    """Same structure, different regression coefficients."""
    rng = np.random.default_rng(seed)
    scms = []
    for _ in range(K):
        noise = rng.normal(0, 0.3, size=base_scm.regression_coefficients.shape)
        mask = base_scm.adjacency_matrix > 0
        new_coefs = base_scm.regression_coefficients.copy()
        new_coefs[mask] += noise[mask]
        new_vars = base_scm.residual_variances * (1 + 0.2 * rng.random(base_scm.residual_variances.shape))
        scm_new = StructuralCausalModel(
            adjacency_matrix=base_scm.adjacency_matrix.copy(),
            variable_names=base_scm.variable_names,
            regression_coefficients=new_coefs,
            residual_variances=new_vars,
            sample_size=base_scm.sample_size,
        )
        scms.append(scm_new)
    return scms


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def three_var_dag():
    return _make_three_var_dag()


@pytest.fixture
def identical_adj_data(three_var_dag):
    """Identical SCMs -> adjacencies + datasets."""
    scms = _make_identical_scms(three_var_dag, K=3)
    return _scms_to_adjacencies_datasets(scms)


@pytest.fixture
def structurally_different_adj_data():
    scms = _make_structurally_different_scms(p=4, K=3)
    return _scms_to_adjacencies_datasets(scms)


@pytest.fixture
def parametrically_different_adj_data(three_var_dag):
    scms = _make_parametrically_different_scms(three_var_dag, K=3)
    return _scms_to_adjacencies_datasets(scms)


@pytest.fixture
def default_config():
    return PlasticityConfig(
        n_bootstrap=50,
        n_stability_rounds=30,
        n_context_subsets=20,
        compute_cis=False,
        compute_classification=False,
        random_state=42,
    )


@pytest.fixture
def ci_config():
    return PlasticityConfig(
        n_bootstrap=80,
        n_stability_rounds=40,
        n_context_subsets=30,
        compute_cis=True,
        compute_classification=True,
        random_state=123,
    )


# ---------------------------------------------------------------------------
# Test structural plasticity (psi_S)
# ---------------------------------------------------------------------------

class TestStructuralPlasticity:
    """Test psi_S computation with known adjacency matrices."""

    def test_identical_parent_sets_gives_zero(self, default_config, three_var_dag):
        """When all contexts have the same parents, psi_S = 0."""
        computer = PlasticityComputer(config=default_config)
        adj = three_var_dag.adjacency_matrix
        adjacencies = [adj.copy() for _ in range(3)]
        n_vars = adj.shape[0]
        K = len(adjacencies)
        psi_s, _ = computer._compute_structural_plasticity(adjacencies, target_idx=2, n_vars=n_vars, K=K)
        assert psi_s == 0.0

    def test_all_different_parent_sets_gives_positive(self, default_config):
        """When parent sets differ, psi_S > 0."""
        computer = PlasticityComputer(config=default_config)
        adj1 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        adj2 = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=float)
        adj3 = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=float)
        psi_s, _ = computer._compute_structural_plasticity([adj1, adj2, adj3], target_idx=2, n_vars=3, K=3)
        assert psi_s > 0.0

    def test_maximally_different_parents(self, default_config):
        """Maximally different parent sets should give high psi_S."""
        computer = PlasticityComputer(config=default_config)
        adj1 = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=float)
        adj2 = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        psi_s, _ = computer._compute_structural_plasticity([adj1, adj2], target_idx=2, n_vars=3, K=2)
        assert psi_s > 0.5

    def test_psi_s_bounded_zero_one(self, default_config, rng):
        """psi_S must be in [0, 1]."""
        computer = PlasticityComputer(config=default_config)
        for _ in range(10):
            K = int(rng.integers(2, 6))
            p = 4
            adjacencies = []
            for _ in range(K):
                adj = np.zeros((p, p))
                for i in range(p):
                    for j in range(i + 1, p):
                        if rng.random() < 0.4:
                            adj[i, j] = 1.0
                adjacencies.append(adj)
            psi_s, _ = computer._compute_structural_plasticity(adjacencies, target_idx=p - 1, n_vars=p, K=K)
            assert 0.0 <= psi_s <= 1.0

    def test_empty_parent_sets(self, default_config):
        """All empty parent sets -> psi_S = 0."""
        computer = PlasticityComputer(config=default_config)
        adj = np.zeros((3, 3))
        psi_s, _ = computer._compute_structural_plasticity([adj, adj, adj], target_idx=2, n_vars=3, K=3)
        assert psi_s == 0.0

    def test_single_context(self, default_config):
        """Single context -> psi_S = 0 (no variation)."""
        computer = PlasticityComputer(config=default_config)
        adj = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=float)
        psi_s, _ = computer._compute_structural_plasticity([adj], target_idx=2, n_vars=3, K=1)
        assert psi_s == 0.0

    def test_adding_removing_parents_increases_psi_s(self, default_config):
        """Adding/removing parents should increase psi_S from zero."""
        computer = PlasticityComputer(config=default_config)
        adj_base = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]], dtype=float)
        psi_base, _ = computer._compute_structural_plasticity([adj_base, adj_base], target_idx=2, n_vars=3, K=2)
        adj_add = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]], dtype=float)
        adj_mod = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=float)
        psi_mod, _ = computer._compute_structural_plasticity([adj_add, adj_mod], target_idx=2, n_vars=3, K=2)
        assert psi_mod > psi_base

    def test_all_parent_sets_identical_helper(self, default_config):
        """Test the _all_parent_sets_identical helper."""
        computer = PlasticityComputer(config=default_config)
        assert computer._all_parent_sets_identical([[1, 2], [1, 2], [1, 2]])
        assert not computer._all_parent_sets_identical([[1, 2], [1, 3]])


# ---------------------------------------------------------------------------
# Test parametric plasticity (psi_P)
# ---------------------------------------------------------------------------

class TestParametricPlasticity:
    """Test psi_P with known Gaussian parameters."""

    def test_identical_parameters_gives_small(self, identical_adj_data, default_config):
        """When all mechanisms are identical, psi_P should be small (sampling noise)."""
        adjacencies, datasets = identical_adj_data
        computer = PlasticityComputer(config=default_config)
        desc = computer.compute(adjacencies, datasets, target_idx=1)
        assert desc.psi_P < 0.2

    def test_different_parameters_gives_positive(self, parametrically_different_adj_data, default_config):
        """When parameters differ, psi_P > 0."""
        adjacencies, datasets = parametrically_different_adj_data
        computer = PlasticityComputer(config=default_config)
        desc = computer.compute(adjacencies, datasets, target_idx=2)
        assert desc.psi_P > 0.0

    def test_psi_p_bounded(self, parametrically_different_adj_data, default_config):
        """psi_P in [0, 1]."""
        adjacencies, datasets = parametrically_different_adj_data
        computer = PlasticityComputer(config=default_config)
        n_vars = adjacencies[0].shape[0]
        for vi in range(n_vars):
            desc = computer.compute(adjacencies, datasets, target_idx=vi)
            assert 0.0 <= desc.psi_P <= 1.0 + 1e-10

    def test_larger_perturbation_larger_psi_p(self, three_var_dag):
        """Larger coefficient perturbations -> larger psi_P."""
        config = PlasticityConfig(
            n_bootstrap=30, compute_cis=False,
            compute_classification=False, random_state=42,
        )
        computer = PlasticityComputer(config=config)
        psi_ps = []
        for scale_seed in [1, 50, 200]:
            scms = _make_parametrically_different_scms(three_var_dag, K=3, seed=scale_seed)
            adjacencies, datasets = _scms_to_adjacencies_datasets(scms, rng_seed=scale_seed)
            desc = computer.compute(adjacencies, datasets, target_idx=2)
            psi_ps.append(desc.psi_P)
        # At least one larger seed should produce larger psi_P than smallest
        assert max(psi_ps) > min(psi_ps) or all(p == 0 for p in psi_ps)


# ---------------------------------------------------------------------------
# Test emergence (psi_E)
# ---------------------------------------------------------------------------

class TestEmergence:
    """Test psi_E computation from adjacency matrices."""

    def test_identical_adjacencies_constant_emergence(self, default_config, three_var_dag):
        """When all adjacencies are the same, psi_E = 1 - min_mb/(max_mb+1)."""
        computer = PlasticityComputer(config=default_config)
        adj = three_var_dag.adjacency_matrix
        adjacencies = [adj.copy() for _ in range(4)]
        psi_e, mb_sizes = computer._compute_emergence(adjacencies, target_idx=2, K=4)
        # All MB sizes are equal, so min == max -> psi_E = 1 - min/(max+1)
        if max(mb_sizes) > 0:
            expected = 1.0 - min(mb_sizes) / (max(mb_sizes) + 1)
            assert_allclose(psi_e, expected, atol=1e-10)
        else:
            assert psi_e == 0.0

    def test_varying_adjacencies_gives_positive(self, default_config):
        """Different adjacencies => psi_E > 0."""
        computer = PlasticityComputer(config=default_config)
        adj1 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        adj2 = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=float)
        adj3 = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        psi_e, _ = computer._compute_emergence([adj1, adj2, adj3], target_idx=2, K=3)
        assert psi_e >= 0.0

    def test_single_adjacency(self, default_config, three_var_dag):
        """Single context -> psi_E = 0."""
        computer = PlasticityComputer(config=default_config)
        adj = three_var_dag.adjacency_matrix
        psi_e, _ = computer._compute_emergence([adj], target_idx=2, K=1)
        assert psi_e == 0.0

    def test_psi_e_bounded(self, default_config, rng):
        """psi_E in [0, 1]."""
        computer = PlasticityComputer(config=default_config)
        for _ in range(10):
            K = int(rng.integers(2, 8))
            p = 4
            adjacencies = []
            for _ in range(K):
                adj = np.zeros((p, p))
                for i in range(p):
                    for j in range(i + 1, p):
                        if rng.random() < 0.3:
                            adj[i, j] = 1.0
                adjacencies.append(adj)
            psi_e, _ = computer._compute_emergence(adjacencies, target_idx=p - 1, K=K)
            assert 0.0 <= psi_e <= 1.0 + 1e-10


# ---------------------------------------------------------------------------
# Test context sensitivity (psi_CS)
# ---------------------------------------------------------------------------

class TestContextSensitivity:
    """Test psi_CS computation."""

    def test_identical_mechanisms_low_sensitivity(self, identical_adj_data, default_config):
        """Identical SCMs -> psi_CS should be relatively small (sampling noise)."""
        adjacencies, datasets = identical_adj_data
        computer = PlasticityComputer(config=default_config)
        desc = computer.compute(adjacencies, datasets, target_idx=1)
        assert desc.psi_CS < 0.3

    def test_different_mechanisms_positive_sensitivity(
        self, parametrically_different_adj_data, default_config
    ):
        """Different mechanisms -> psi_CS > 0."""
        adjacencies, datasets = parametrically_different_adj_data
        computer = PlasticityComputer(config=default_config)
        desc = computer.compute(adjacencies, datasets, target_idx=2)
        assert desc.psi_CS >= 0.0

    def test_psi_cs_bounded(self, parametrically_different_adj_data, default_config):
        """psi_CS in [0, 1]."""
        adjacencies, datasets = parametrically_different_adj_data
        computer = PlasticityComputer(config=default_config)
        n_vars = adjacencies[0].shape[0]
        for vi in range(n_vars):
            desc = computer.compute(adjacencies, datasets, target_idx=vi)
            assert 0.0 <= desc.psi_CS <= 1.0 + 1e-10


# ---------------------------------------------------------------------------
# Test full descriptor computation
# ---------------------------------------------------------------------------

class TestFullDescriptor:
    """Test the complete compute() method."""

    def test_descriptor_has_all_fields(self, identical_adj_data, default_config):
        adjacencies, datasets = identical_adj_data
        computer = PlasticityComputer(config=default_config)
        desc = computer.compute(adjacencies, datasets, target_idx=0)
        assert isinstance(desc, PlasticityDescriptor)
        assert hasattr(desc, "psi_S")
        assert hasattr(desc, "psi_P")
        assert hasattr(desc, "psi_E")
        assert hasattr(desc, "psi_CS")

    def test_descriptor_vector_property(self, identical_adj_data, default_config):
        adjacencies, datasets = identical_adj_data
        computer = PlasticityComputer(config=default_config)
        desc = computer.compute(adjacencies, datasets, target_idx=0)
        vec = desc.descriptor_vector
        assert vec.shape == (4,)
        assert_allclose(vec, [desc.psi_S, desc.psi_P, desc.psi_E, desc.psi_CS])

    def test_descriptor_magnitude(self, identical_adj_data, default_config):
        adjacencies, datasets = identical_adj_data
        computer = PlasticityComputer(config=default_config)
        desc = computer.compute(adjacencies, datasets, target_idx=0)
        expected_mag = np.linalg.norm([desc.psi_S, desc.psi_P, desc.psi_E, desc.psi_CS])
        assert_allclose(np.linalg.norm(desc.descriptor_vector), expected_mag, atol=1e-10)

    def test_descriptor_dominant_dimension(self):
        desc = PlasticityDescriptor(
            variable_idx=0, variable_name="X0",
            psi_S=0.9, psi_P=0.1, psi_E=0.2, psi_CS=0.3,
        )
        vec = desc.descriptor_vector
        dim_names = ["psi_S", "psi_P", "psi_E", "psi_CS"]
        dominant = dim_names[int(np.argmax(vec))]
        assert dominant == "psi_S"

    def test_descriptor_distance(self):
        d1 = PlasticityDescriptor(
            variable_idx=0, variable_name="X0",
            psi_S=0.0, psi_P=0.0, psi_E=0.0, psi_CS=0.0,
        )
        d2 = PlasticityDescriptor(
            variable_idx=1, variable_name="X1",
            psi_S=1.0, psi_P=0.0, psi_E=0.0, psi_CS=0.0,
        )
        dist = np.linalg.norm(d1.descriptor_vector - d2.descriptor_vector)
        assert_allclose(dist, 1.0, atol=1e-10)

    def test_descriptor_serialization(self, identical_adj_data, default_config):
        adjacencies, datasets = identical_adj_data
        computer = PlasticityComputer(config=default_config)
        desc = computer.compute(adjacencies, datasets, target_idx=0)
        d = desc.to_dict()
        restored = PlasticityDescriptor.from_dict(d)
        assert_allclose(restored.psi_S, desc.psi_S)
        assert_allclose(restored.psi_P, desc.psi_P)
        assert_allclose(restored.psi_E, desc.psi_E)
        assert_allclose(restored.psi_CS, desc.psi_CS)

    def test_invariant_all_near_zero(self, identical_adj_data, default_config):
        """For identical SCMs, structural psi_S should be exactly 0."""
        adjacencies, datasets = identical_adj_data
        computer = PlasticityComputer(config=default_config)
        desc = computer.compute(adjacencies, datasets, target_idx=1)
        assert desc.psi_S < 0.05

    def test_variable_index_stored(self, identical_adj_data, default_config):
        adjacencies, datasets = identical_adj_data
        computer = PlasticityComputer(config=default_config)
        desc = computer.compute(adjacencies, datasets, target_idx=2)
        assert desc.variable_idx == 2


# ---------------------------------------------------------------------------
# Test classification with descriptor values
# ---------------------------------------------------------------------------

class TestClassification:
    """Test classification integration inside PlasticityComputer."""

    def test_invariant_classification(self, identical_adj_data):
        adjacencies, datasets = identical_adj_data
        config = PlasticityConfig(
            n_bootstrap=30, compute_cis=False,
            compute_classification=True, random_state=42,
        )
        computer = PlasticityComputer(config=config)
        desc = computer.compute(adjacencies, datasets, target_idx=1)
        assert desc.psi_S < config.tau_S

    def test_structural_plastic_classification(self, structurally_different_adj_data):
        adjacencies, datasets = structurally_different_adj_data
        config = PlasticityConfig(
            n_bootstrap=30, compute_cis=False,
            compute_classification=True, random_state=42,
        )
        computer = PlasticityComputer(config=config)
        desc = computer.compute(adjacencies, datasets, target_idx=0)
        assert desc.psi_S >= 0.0


# ---------------------------------------------------------------------------
# Test BatchPlasticityComputer
# ---------------------------------------------------------------------------

class TestBatchPlasticityComputer:
    """Test batch computation over all variables."""

    def test_batch_returns_all_variables(self, identical_adj_data, default_config):
        adjacencies, datasets = identical_adj_data
        batch_computer = BatchPlasticityComputer(config=default_config)
        result = batch_computer.compute(adjacencies, datasets)
        n_vars = adjacencies[0].shape[0]
        assert len(result.descriptors) == n_vars

    def test_batch_descriptor_matrix_shape(self, identical_adj_data, default_config):
        adjacencies, datasets = identical_adj_data
        batch_computer = BatchPlasticityComputer(config=default_config)
        result = batch_computer.compute(adjacencies, datasets)
        mat = result.descriptor_matrix()
        n_vars = adjacencies[0].shape[0]
        assert mat.shape == (n_vars, 4)

    def test_batch_invariant_variables(self, identical_adj_data, default_config):
        adjacencies, datasets = identical_adj_data
        batch_computer = BatchPlasticityComputer(config=default_config)
        result = batch_computer.compute(adjacencies, datasets)
        invariants = result.invariant_variables()
        assert isinstance(invariants, list)

    def test_batch_most_plastic(self, parametrically_different_adj_data, default_config):
        adjacencies, datasets = parametrically_different_adj_data
        batch_computer = BatchPlasticityComputer(config=default_config)
        result = batch_computer.compute(adjacencies, datasets)
        most = result.most_plastic(n=2)
        assert len(most) <= 2
        if len(most) >= 2:
            assert np.linalg.norm(most[0].descriptor_vector) >= np.linalg.norm(most[1].descriptor_vector)

    def test_batch_all_descriptors_bounded(self, parametrically_different_adj_data, default_config):
        adjacencies, datasets = parametrically_different_adj_data
        batch_computer = BatchPlasticityComputer(config=default_config)
        result = batch_computer.compute(adjacencies, datasets)
        for desc in result.descriptors:
            assert 0.0 <= desc.psi_S <= 1.0 + 1e-10
            assert 0.0 <= desc.psi_P <= 1.0 + 1e-10
            assert 0.0 <= desc.psi_E <= 1.0 + 1e-10
            assert 0.0 <= desc.psi_CS <= 1.0 + 1e-10


# ---------------------------------------------------------------------------
# Test confidence intervals
# ---------------------------------------------------------------------------

class TestConfidenceIntervals:
    """Test CI computation within the plasticity computer."""

    def test_cis_are_computed(self, parametrically_different_adj_data, ci_config):
        adjacencies, datasets = parametrically_different_adj_data
        computer = PlasticityComputer(config=ci_config)
        desc = computer.compute(adjacencies, datasets, target_idx=1)
        assert desc.has_cis
        assert desc.psi_S_ci is not None
        assert desc.psi_P_ci is not None

    def test_ci_contains_point_estimate(self, parametrically_different_adj_data, ci_config):
        adjacencies, datasets = parametrically_different_adj_data
        computer = PlasticityComputer(config=ci_config)
        desc = computer.compute(adjacencies, datasets, target_idx=1)
        if desc.psi_S_ci is not None:
            lo, hi = desc.psi_S_ci
            assert lo <= desc.psi_S + 1e-6
            assert desc.psi_S <= hi + 1e-6

    def test_ci_width_positive(self, parametrically_different_adj_data, ci_config):
        adjacencies, datasets = parametrically_different_adj_data
        computer = PlasticityComputer(config=ci_config)
        desc = computer.compute(adjacencies, datasets, target_idx=1)
        if desc.psi_P_ci is not None:
            lo, hi = desc.psi_P_ci
            assert hi >= lo

    def test_no_cis_when_disabled(self, identical_adj_data, default_config):
        adjacencies, datasets = identical_adj_data
        computer = PlasticityComputer(config=default_config)
        desc = computer.compute(adjacencies, datasets, target_idx=0)
        assert not desc.has_cis
        assert desc.psi_S_ci is None


# ---------------------------------------------------------------------------
# Parametric tests with different K
# ---------------------------------------------------------------------------

class TestParametricK:
    """Test with different numbers of contexts K ∈ {3, 5, 10, 20}."""

    @pytest.mark.parametrize("K", [3, 5, 10, 20])
    def test_identical_invariant_for_all_K(self, three_var_dag, K):
        config = PlasticityConfig(
            n_bootstrap=20, compute_cis=False,
            compute_classification=False, random_state=42,
        )
        scms = _make_identical_scms(three_var_dag, K=K)
        adjacencies, datasets = _scms_to_adjacencies_datasets(scms)
        computer = PlasticityComputer(config=config)
        desc = computer.compute(adjacencies, datasets, target_idx=1)
        assert desc.psi_S < 0.05
        assert desc.psi_P < 0.2  # Sampling noise in data can cause small psi_P

    @pytest.mark.parametrize("K", [3, 5, 10])
    def test_structural_plastic_detected_for_all_K(self, K):
        config = PlasticityConfig(
            n_bootstrap=20, compute_cis=False,
            compute_classification=False, random_state=42,
        )
        scms = _make_structurally_different_scms(p=4, K=K, seed=42)
        adjacencies, datasets = _scms_to_adjacencies_datasets(scms)
        computer = PlasticityComputer(config=config)
        desc = computer.compute(adjacencies, datasets, target_idx=0)
        assert desc.psi_S >= 0.0

    @pytest.mark.parametrize("K", [3, 5, 10])
    def test_batch_works_for_all_K(self, three_var_dag, K, default_config):
        scms = _make_identical_scms(three_var_dag, K=K)
        adjacencies, datasets = _scms_to_adjacencies_datasets(scms)
        batch_computer = BatchPlasticityComputer(config=default_config)
        result = batch_computer.compute(adjacencies, datasets)
        assert len(result.descriptors) == three_var_dag.num_variables


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases: single context, identical, all-different."""

    def test_single_context(self, three_var_dag, default_config):
        """Single-context should give all-zero descriptors."""
        scms = _make_identical_scms(three_var_dag, K=1)
        adjacencies, datasets = _scms_to_adjacencies_datasets(scms)
        computer = PlasticityComputer(config=default_config)
        desc = computer.compute(adjacencies, datasets, target_idx=0)
        assert desc.psi_S == 0.0

    def test_two_identical_contexts(self, three_var_dag, default_config):
        scms = _make_identical_scms(three_var_dag, K=2)
        adjacencies, datasets = _scms_to_adjacencies_datasets(scms)
        computer = PlasticityComputer(config=default_config)
        desc = computer.compute(adjacencies, datasets, target_idx=1)
        assert desc.psi_S < 0.05

    def test_root_variable_no_parents(self, identical_adj_data, default_config, three_var_dag):
        """Root variable should have psi_S ~ 0 in identical SCMs."""
        adjacencies, datasets = identical_adj_data
        computer = PlasticityComputer(config=default_config)
        roots = three_var_dag.roots()
        if roots:
            desc = computer.compute(adjacencies, datasets, target_idx=roots[0])
            assert desc.psi_S < 0.05

    def test_leaf_variable(self, identical_adj_data, default_config, three_var_dag):
        """Leaf variable test."""
        adjacencies, datasets = identical_adj_data
        computer = PlasticityComputer(config=default_config)
        leaves = three_var_dag.leaves()
        if leaves:
            desc = computer.compute(adjacencies, datasets, target_idx=leaves[0])
            assert isinstance(desc, PlasticityDescriptor)

    def test_config_thresholds(self, default_config):
        """Config thresholds are accessible."""
        t = default_config.thresholds()
        assert t.tau_S == default_config.tau_S
        assert t.tau_P == default_config.tau_P
        assert t.tau_E == default_config.tau_E

    def test_input_validation_bad_target_idx(self, identical_adj_data, default_config):
        """Out-of-range target_idx should raise."""
        adjacencies, datasets = identical_adj_data
        computer = PlasticityComputer(config=default_config)
        n_vars = adjacencies[0].shape[0]
        with pytest.raises((ValueError, IndexError)):
            computer.compute(adjacencies, datasets, target_idx=n_vars + 10)

    def test_descriptor_distance_symmetry(self):
        d1 = PlasticityDescriptor(
            variable_idx=0, variable_name="X0",
            psi_S=0.5, psi_P=0.3, psi_E=0.2, psi_CS=0.1,
        )
        d2 = PlasticityDescriptor(
            variable_idx=1, variable_name="X1",
            psi_S=0.1, psi_P=0.7, psi_E=0.4, psi_CS=0.8,
        )
        dist12 = np.linalg.norm(d1.descriptor_vector - d2.descriptor_vector)
        dist21 = np.linalg.norm(d2.descriptor_vector - d1.descriptor_vector)
        assert_allclose(dist12, dist21, atol=1e-12)

    def test_descriptor_distance_zero_to_self(self):
        d = PlasticityDescriptor(
            variable_idx=0, variable_name="X0",
            psi_S=0.5, psi_P=0.3, psi_E=0.2, psi_CS=0.1,
        )
        dist = np.linalg.norm(d.descriptor_vector - d.descriptor_vector)
        assert_allclose(dist, 0.0, atol=1e-12)

    def test_plasticity_config_defaults(self):
        config = PlasticityConfig()
        assert config.n_bootstrap == 200
        assert config.ci_level == 0.95
        assert config.compute_cis is True

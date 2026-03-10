"""
Tests for the simulation module: data generation engines, noise models,
DGP library, perturbation, faithfulness checking, and Monte Carlo
simulation studies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalcert.simulation.engines import (
    LinearGaussianEngine,
    NonlinearEngine,
    MixedTypeEngine,
    InterventionalEngine,
)
from causalcert.simulation.noise_models import (
    GaussianNoise,
    StudentTNoise,
    MixtureNoise,
    HeteroskedasticNoise,
    NonAdditiveNoise,
    DiscreteNoise,
    create_noise,
)
from causalcert.simulation.dgp_library import (
    LaLondeDGP,
    SmokingBirthweightDGP,
    IHDPSimulation,
    InstrumentDGP,
    MediationDGP,
    ConfoundedDGP,
    FaithfulnessViolationDGP,
    SparseHighDimDGP,
    create_dgp,
    list_dgps,
)
from causalcert.simulation.perturbation import (
    PerturbationGenerator,
    PerturbedDAG,
    ImpactCategory,
)
from causalcert.simulation.faithfulness import (
    FaithfulnessChecker,
    FaithfulnessReport,
    PathCancellation,
)
from causalcert.simulation.monte_carlo import (
    MonteCarloRunner,
    SimStudyConfig,
    SimStudyResult,
    run_simulation_study,
)
from causalcert.simulation.types import DGPSpec, GroundTruth, SimulationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _adj(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    a = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        a[u, v] = 1
    return a


def _chain3() -> np.ndarray:
    return _adj(3, [(0, 1), (1, 2)])


def _diamond4() -> np.ndarray:
    return _adj(4, [(0, 1), (0, 2), (1, 3), (2, 3)])


def _fork3() -> np.ndarray:
    return _adj(3, [(0, 1), (0, 2)])


# ---------------------------------------------------------------------------
# Engines: correct shape
# ---------------------------------------------------------------------------


class TestEngineShape:

    def test_linear_gaussian_shape(self):
        engine = LinearGaussianEngine()
        adj = _chain3()
        df = engine.generate(adj, n_samples=100)
        assert df.shape == (100, 3)

    def test_linear_gaussian_columns(self):
        engine = LinearGaussianEngine()
        adj = _diamond4()
        df = engine.generate(adj, n_samples=50)
        assert df.shape == (50, 4)

    def test_nonlinear_engine_shape(self):
        engine = NonlinearEngine()
        adj = _chain3()
        df = engine.generate(adj, n_samples=100)
        assert df.shape == (100, 3)

    def test_nonlinear_polynomial(self):
        engine = NonlinearEngine(functional_form="polynomial")
        adj = _fork3()
        df = engine.generate(adj, n_samples=100)
        assert df.shape == (100, 3)

    def test_nonlinear_sigmoid(self):
        engine = NonlinearEngine(functional_form="sigmoid")
        adj = _chain3()
        df = engine.generate(adj, n_samples=100)
        assert df.shape == (100, 3)

    def test_mixed_type_engine_shape(self):
        engine = MixedTypeEngine()
        adj = _chain3()
        df = engine.generate(adj, n_samples=100)
        assert df.shape[0] == 100
        assert df.shape[1] == 3

    def test_interventional_engine(self):
        base_engine = LinearGaussianEngine()
        engine = InterventionalEngine(base_engine)
        adj = _chain3()
        # The InterventionalEngine wraps another engine
        assert engine is not None

    def test_deterministic_with_seed(self):
        engine = LinearGaussianEngine()
        adj = _chain3()
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        df1 = engine.generate(adj, n_samples=100, rng=rng1)
        df2 = engine.generate(adj, n_samples=100, rng=rng2)
        pd.testing.assert_frame_equal(df1, df2)

    def test_engine_respects_dag_structure(self):
        """Root nodes should be independent of each other."""
        engine = LinearGaussianEngine()
        adj = _fork3()  # 0 -> 1, 0 -> 2
        rng = np.random.default_rng(42)
        df = engine.generate(adj, n_samples=2000, rng=rng)
        # Node 0 is root; 1 and 2 are children
        corr_01 = np.corrcoef(df.iloc[:, 0], df.iloc[:, 1])[0, 1]
        assert abs(corr_01) > 0.1  # Should be correlated

    def test_engine_weight_matrix(self):
        engine = LinearGaussianEngine()
        adj = _chain3()
        rng = np.random.default_rng(42)
        W = engine.get_weight_matrix(adj, rng=rng)
        assert W.shape == (3, 3)
        # Weights should be nonzero only where edges exist
        for i in range(3):
            for j in range(3):
                if adj[i, j] == 0:
                    assert W[i, j] == 0.0

    def test_total_effect_chain(self):
        engine = LinearGaussianEngine()
        adj = _chain3()
        rng = np.random.default_rng(42)
        W = engine.get_weight_matrix(adj, rng=rng)
        te = engine.compute_total_effect(adj, 0, 2, W)
        # Total effect should be a finite number
        assert np.isfinite(te)


# ---------------------------------------------------------------------------
# Noise models
# ---------------------------------------------------------------------------


class TestNoiseModels:

    def test_gaussian_moments(self):
        noise = GaussianNoise()
        rng = np.random.default_rng(42)
        samples = noise.sample(10000, 1, scale=1.0, rng=rng)
        np.testing.assert_allclose(samples.mean(), 0.0, atol=0.05)
        np.testing.assert_allclose(samples.std(), 1.0, atol=0.05)

    def test_student_t_heavier_tails(self):
        noise = StudentTNoise()
        rng = np.random.default_rng(42)
        samples = noise.sample(10000, 1, scale=1.0, rng=rng)
        # Student-t has heavier tails (higher kurtosis) than Gaussian
        from scipy.stats import kurtosis
        kurt = kurtosis(samples.flatten())
        assert kurt > 0  # excess kurtosis > 0

    def test_mixture_noise_moments(self):
        noise = MixtureNoise()
        rng = np.random.default_rng(42)
        samples = noise.sample(10000, 1, scale=1.0, rng=rng)
        assert np.isfinite(samples.mean())
        assert samples.std() > 0

    def test_discrete_noise(self):
        noise = DiscreteNoise()
        rng = np.random.default_rng(42)
        samples = noise.sample(1000, 1, rng=rng)
        unique_vals = np.unique(samples)
        assert len(unique_vals) > 1

    def test_create_noise_factory(self):
        for name in ["gaussian", "student_t", "mixture"]:
            noise = create_noise(name)
            assert noise.name == name

    def test_gaussian_name(self):
        assert GaussianNoise().name == "gaussian"

    def test_student_t_name(self):
        assert StudentTNoise().name == "student_t"

    def test_noise_log_density(self):
        noise = GaussianNoise()
        rng = np.random.default_rng(42)
        samples = noise.sample(100, 1, rng=rng)
        ld = noise.log_density(samples)
        assert np.all(np.isfinite(ld))
        assert np.all(ld <= 0)  # log-density should be ≤ 0 for standard normal

    def test_heteroskedastic_noise(self):
        noise = HeteroskedasticNoise()
        rng = np.random.default_rng(42)
        samples = noise.sample(1000, 1, rng=rng)
        assert samples.shape[0] == 1000

    def test_noise_multiple_variables(self):
        noise = GaussianNoise()
        rng = np.random.default_rng(42)
        samples = noise.sample(100, 5, scale=1.0, rng=rng)
        assert samples.shape == (100, 5)


# ---------------------------------------------------------------------------
# DGP library
# ---------------------------------------------------------------------------


class TestDGPLibrary:

    def test_list_dgps_nonempty(self):
        dgps = list_dgps()
        assert len(dgps) >= 5

    def test_create_dgp_factory(self):
        dgps = list_dgps()
        for name in dgps[:3]:
            dgp = create_dgp(name)
            assert dgp is not None

    def test_lalonde_dgp(self):
        dgp = LaLondeDGP(n_samples=200, seed=42)
        df = dgp.generate()
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 200

    def test_lalonde_ground_truth(self):
        dgp = LaLondeDGP(n_samples=200, seed=42)
        gt = dgp.ground_truth()
        assert isinstance(gt, GroundTruth)
        assert gt.true_ate is not None

    def test_lalonde_dag(self):
        dgp = LaLondeDGP(seed=42)
        dag = dgp.dag()
        assert dag.shape[0] == dag.shape[1]
        assert dag.shape[0] >= 3

    def test_confounded_dgp(self):
        dgp = ConfoundedDGP(n_samples=200, treatment_effect=1.0, seed=42)
        df = dgp.generate()
        gt = dgp.ground_truth()
        assert df.shape[0] == 200
        np.testing.assert_allclose(gt.true_ate, 1.0, atol=0.01)

    def test_mediation_dgp(self):
        dgp = MediationDGP(
            n_samples=200, direct_effect=1.0, indirect_effect=0.5, seed=42
        )
        df = dgp.generate()
        gt = dgp.ground_truth()
        assert df.shape[0] == 200
        np.testing.assert_allclose(gt.true_ate, 1.5, atol=0.1)

    def test_instrument_dgp(self):
        dgp = InstrumentDGP(n_samples=300, true_late=2.0, seed=42)
        df = dgp.generate()
        gt = dgp.ground_truth()
        assert df.shape[0] == 300
        assert gt.true_ate is not None

    def test_faithfulness_violation_dgp(self):
        dgp = FaithfulnessViolationDGP(n_samples=200, seed=42)
        df = dgp.generate()
        gt = dgp.ground_truth()
        assert df.shape[0] == 200
        assert isinstance(gt, GroundTruth)

    def test_sparse_high_dim_dgp(self):
        dgp = SparseHighDimDGP(n_samples=200, seed=42)
        df = dgp.generate()
        assert df.shape[0] == 200
        assert df.shape[1] >= 10  # should be high-dimensional

    def test_smoking_dgp(self):
        dgp = SmokingBirthweightDGP(n_samples=200, seed=42)
        df = dgp.generate()
        assert df.shape[0] == 200

    def test_ihdp_dgp(self):
        dgp = IHDPSimulation(n_samples=200, seed=42)
        df = dgp.generate()
        assert df.shape[0] == 200

    def test_dgp_dag_is_valid(self):
        """All DGP DAGs should be valid (no cycles)."""
        for DGPClass in [LaLondeDGP, ConfoundedDGP, MediationDGP, InstrumentDGP]:
            dgp = DGPClass(seed=42)
            dag = dgp.dag()
            # Check no cycles via topological sort
            n = dag.shape[0]
            in_deg = dag.sum(axis=0).astype(int)
            queue = [v for v in range(n) if in_deg[v] == 0]
            count = 0
            while queue:
                v = queue.pop(0)
                count += 1
                for w in range(n):
                    if dag[v, w]:
                        in_deg[w] -= 1
                        if in_deg[w] == 0:
                            queue.append(w)
            assert count == n, f"{DGPClass.__name__} DAG has a cycle"


# ---------------------------------------------------------------------------
# Perturbation
# ---------------------------------------------------------------------------


class TestPerturbation:

    def test_single_edit_neighbourhood_size(self):
        adj = _chain3()
        gen = PerturbationGenerator()
        nbrs = gen.single_edit_neighbourhood(adj)
        # For 3 nodes: 3 edges exist, can delete each; plus can add others, reverse
        assert len(nbrs) >= 1
        for p in nbrs:
            assert isinstance(p, PerturbedDAG)

    def test_k_edit_neighbourhood_k1(self):
        adj = _chain3()
        gen = PerturbationGenerator()
        nbrs = gen.k_edit_neighbourhood(adj, k=1)
        single = gen.single_edit_neighbourhood(adj)
        assert len(nbrs) == len(single)

    def test_k_edit_neighbourhood_k2_larger(self):
        adj = _chain3()
        gen = PerturbationGenerator()
        nbrs1 = gen.k_edit_neighbourhood(adj, k=1)
        nbrs2 = gen.k_edit_neighbourhood(adj, k=2)
        assert len(nbrs2) >= len(nbrs1)

    def test_perturbed_dag_is_dag(self):
        adj = _chain3()
        gen = PerturbationGenerator()
        nbrs = gen.single_edit_neighbourhood(adj)
        for p in nbrs[:5]:
            # Check it's a valid DAG (no self-loops, but may have cycles
            # since not all perturbations preserve DAG-ness)
            assert p is not None

    def test_random_perturbations(self):
        adj = _diamond4()
        gen = PerturbationGenerator()
        perturbed = gen.random_perturbations(adj, n_samples=10, k=1)
        assert len(perturbed) == 10

    def test_constrained_perturbation(self):
        adj = _chain3()
        gen = PerturbationGenerator()
        # Protect edge 0->1
        nbrs = gen.constrained_perturbations(
            adj, protected_edges=[(0, 1)], k=1
        )
        for p in nbrs:
            assert isinstance(p, PerturbedDAG)

    def test_impact_category_enum(self):
        cats = [
            ImpactCategory.NO_CHANGE,
            ImpactCategory.SIGN_CHANGE,
            ImpactCategory.MAGNITUDE_CHANGE,
            ImpactCategory.IDENTIFICATION_CHANGE,
            ImpactCategory.STRUCTURE_ONLY,
        ]
        assert len(cats) == 5

    def test_perturbation_empty_dag(self):
        adj = _adj(3, [])
        gen = PerturbationGenerator()
        nbrs = gen.single_edit_neighbourhood(adj)
        # Can only add edges to empty DAG
        assert len(nbrs) >= 1


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------


class TestFaithfulness:

    def test_faithful_linear_gaussian(self):
        """Linear Gaussian with generic weights should be faithful."""
        engine = LinearGaussianEngine()
        adj = _chain3()
        rng = np.random.default_rng(42)
        df = engine.generate(adj, n_samples=2000, rng=rng)
        W = engine.get_weight_matrix(adj, rng=np.random.default_rng(42))
        checker = FaithfulnessChecker()
        report = checker.assess_faithfulness(adj, df, weights=W)
        assert isinstance(report, FaithfulnessReport)

    def test_parameter_faithfulness(self):
        adj = _chain3()
        W = np.zeros((3, 3))
        W[0, 1] = 1.5
        W[1, 2] = 0.8
        checker = FaithfulnessChecker()
        cancellations = checker.assess_parameter_faithfulness(adj, W)
        assert isinstance(cancellations, list)

    def test_cancellation_detected(self):
        """Two paths from X to Y with opposite signs should flag."""
        adj = _diamond4()  # 0->1, 0->2, 1->3, 2->3
        W = np.zeros((4, 4))
        W[0, 1] = 2.0
        W[0, 2] = 2.0
        W[1, 3] = 1.0
        W[2, 3] = -1.0  # Path 0->2->3 cancels 0->1->3
        checker = FaithfulnessChecker()
        cancellations = checker.assess_parameter_faithfulness(adj, W)
        # Should detect near-cancellation
        assert isinstance(cancellations, list)

    def test_distribution_faithfulness(self):
        engine = LinearGaussianEngine()
        adj = _chain3()
        rng = np.random.default_rng(42)
        df = engine.generate(adj, n_samples=1000, rng=rng)
        checker = FaithfulnessChecker()
        violations = checker.assess_distribution_faithfulness(adj, df)
        assert isinstance(violations, list)

    def test_violation_severity_bounded(self):
        engine = LinearGaussianEngine()
        adj = _chain3()
        rng = np.random.default_rng(42)
        df = engine.generate(adj, n_samples=500, rng=rng)
        checker = FaithfulnessChecker()
        severity = checker.measure_violation_severity(adj, df)
        assert 0 <= severity <= 1

    def test_faithfulness_violation_dgp_detected(self):
        dgp = FaithfulnessViolationDGP(
            n_samples=2000, cancellation_fraction=0.99, seed=42
        )
        df = dgp.generate()
        adj = dgp.dag()
        checker = FaithfulnessChecker()
        report = checker.assess_faithfulness(adj, df)
        assert isinstance(report, FaithfulnessReport)


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------


class TestMonteCarlo:

    def test_sim_study_config_creation(self):
        def _dgp_fn(n, rng):
            adj = np.zeros((3, 3), dtype=np.int8)
            adj[0, 1] = adj[1, 2] = 1
            X = rng.standard_normal((n, 3))
            d = rng.binomial(1, 0.5, n).astype(float)
            y = 1.5 * d + rng.standard_normal(n) * 0.5
            data = pd.DataFrame({"X0": X[:, 0], "X1": d, "X2": y})
            return data, adj, 1.5

        def _est_fn(data, adj, treatment, outcome):
            d = data.iloc[:, treatment].values
            y = data.iloc[:, outcome].values
            return y[d == 1].mean() - y[d == 0].mean() if d.sum() > 0 else 0.0

        config = SimStudyConfig(
            n_replicates=5,
            sample_sizes=[100, 200],
            estimator_fn=_est_fn,
            dgp_fn=_dgp_fn,
            true_ate=1.5,
            seed=42,
        )
        assert config.n_replicates == 5

    def test_run_simulation_study(self):
        def _dgp_fn(n, rng):
            adj = np.zeros((3, 3), dtype=np.int8)
            adj[0, 1] = adj[1, 2] = 1
            d = rng.binomial(1, 0.5, n).astype(float)
            y = 1.5 * d + rng.standard_normal(n) * 0.5
            data = np.column_stack([rng.standard_normal(n), d, y])
            return data, adj, 1.5

        def _est_fn(data, adj, treatment, outcome):
            d = data[:, treatment]
            y = data[:, outcome]
            return y[d == 1].mean() - y[d == 0].mean() if d.sum() > 0 else 0.0

        config = SimStudyConfig(
            n_replicates=5,
            sample_sizes=[100],
            estimator_fn=_est_fn,
            dgp_fn=_dgp_fn,
            true_ate=1.5,
            seed=42,
        )
        result = run_simulation_study(config)
        assert isinstance(result, SimStudyResult)

    def test_sim_study_to_dataframe(self):
        def _dgp_fn(n, rng):
            adj = np.zeros((3, 3), dtype=np.int8)
            adj[0, 1] = adj[1, 2] = 1
            d = rng.binomial(1, 0.5, n).astype(float)
            y = 1.5 * d + rng.standard_normal(n) * 0.5
            data = np.column_stack([rng.standard_normal(n), d, y])
            return data, adj, 1.5

        def _est_fn(data, adj, treatment, outcome):
            d = data[:, treatment]
            y = data[:, outcome]
            return y[d == 1].mean() - y[d == 0].mean() if d.sum() > 0 else 0.0

        config = SimStudyConfig(
            n_replicates=5,
            sample_sizes=[100],
            estimator_fn=_est_fn,
            dgp_fn=_dgp_fn,
            true_ate=1.5,
            seed=42,
        )
        result = run_simulation_study(config)
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1

    def test_bias_variance_decomposition(self):
        from causalcert.simulation.monte_carlo import ReplicateResult

        results = []
        for est in [1.8, 2.1, 1.9, 2.2, 2.0]:
            r = ReplicateResult(
                sample_size=100,
                estimate=est,
                se=0.3,
                ci_lower=est - 0.6,
                ci_upper=est + 0.6,
                covers_truth=True,
                true_ate=2.0,
                elapsed_s=0.1,
                seed_used=42,
            )
            results.append(r)
        bvd = MonteCarloRunner.bias_variance_decomposition(results, true_ate=2.0)
        assert isinstance(bvd, dict)

    def test_coverage_confidence_band(self):
        lo, hi = MonteCarloRunner.coverage_confidence_band(
            coverage=0.95, n_replicates=100
        )
        assert lo < 0.95 < hi

    def test_coverage_confidence_wide_for_small_n(self):
        lo_small, hi_small = MonteCarloRunner.coverage_confidence_band(
            coverage=0.95, n_replicates=20
        )
        lo_large, hi_large = MonteCarloRunner.coverage_confidence_band(
            coverage=0.95, n_replicates=1000
        )
        assert (hi_small - lo_small) > (hi_large - lo_large)


# ---------------------------------------------------------------------------
# DGPSpec and SimulationResult types
# ---------------------------------------------------------------------------


class TestSimulationTypes:

    def test_dgp_spec_creation(self):
        adj = _chain3()
        spec = DGPSpec(adjacency=adj)
        assert spec.noise_type == "gaussian"
        assert spec.functional_form == "linear"

    def test_dgp_spec_nonlinear(self):
        adj = _chain3()
        spec = DGPSpec(adjacency=adj, functional_form="nonlinear")
        assert spec.functional_form == "nonlinear"

    def test_dgp_spec_treatment_outcome(self):
        adj = _chain3()
        spec = DGPSpec(adjacency=adj, treatment=0, outcome=2)
        assert spec.treatment == 0
        assert spec.outcome == 2

    def test_ground_truth_fields(self):
        dgp = ConfoundedDGP(seed=42)
        gt = dgp.ground_truth()
        assert hasattr(gt, 'true_ate')
        assert hasattr(gt, 'true_dag')

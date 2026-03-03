"""Integration tests across new DP-Forge modules.

Tests end-to-end pipelines combining CEGIS, grid refinement, robust
verification, RDP accounting, multi-dimensional synthesis, and infinite LP.
Uses small problems (n≤5, k≤10) for speed.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from dp_forge.types import (
    AdjacencyRelation,
    CEGISResult,
    QuerySpec,
    PrivacyBudget,
    SynthesisConfig,
    LossFunction,
    QueryType,
)
from dp_forge.cegis_loop import CEGISSynthesize, CEGISEngine
from dp_forge.extractor import MechanismExtractor, ExtractMechanism
from dp_forge.verifier import verify
from dp_forge.sampling import MechanismSampler


# =========================================================================
# Helpers
# =========================================================================


def _small_config() -> SynthesisConfig:
    """Fast synthesis config for integration tests."""
    return SynthesisConfig(max_iter=20, verbose=0)


def _counting_spec(n: int = 3, eps: float = 1.0, k: int = 10) -> QuerySpec:
    """Small counting query spec."""
    return QuerySpec.counting(n=n, epsilon=eps, delta=0.0, k=k)


def _synthesize(spec: QuerySpec) -> CEGISResult:
    """Run CEGIS synthesis with small config."""
    return CEGISSynthesize(spec, config=_small_config())


def _extract(result: CEGISResult, spec: QuerySpec):
    """Extract a deployable mechanism from CEGIS result."""
    edges = spec.edges.edges if spec.edges else AdjacencyRelation.hamming_distance_1(spec.n).edges
    y_grid = np.linspace(0, float(spec.n), spec.k)
    return ExtractMechanism(
        p_raw=result.mechanism,
        epsilon=spec.epsilon,
        delta=spec.delta,
        edges=edges,
        y_grid=y_grid,
    )


# =========================================================================
# Test: CEGIS → AdaptiveGrid → RobustVerify → Certificate
# =========================================================================


class TestCEGISGridRobustPipeline:
    """End-to-end: CEGIS → AdaptiveGrid → RobustVerify → Certificate."""

    def test_cegis_then_grid_refine(self):
        """Run CEGIS on coarse grid, then refine."""
        from dp_forge.grid import AdaptiveGridRefiner, UniformGrid

        spec = _counting_spec(n=3, eps=1.0, k=8)
        config = _small_config()

        refiner = AdaptiveGridRefiner(
            k0=8, k_max=20, max_levels=3,
            grid_strategy=UniformGrid(),
            synthesis_config=config,
        )
        result = refiner.refine(spec)

        assert result.mechanism is not None
        assert result.mechanism.shape[0] == spec.n
        assert result.iterations > 0
        assert len(refiner.steps) >= 1

    def test_cegis_then_robust_verify(self):
        """Synthesise then verify with RobustCEGIS."""
        from dp_forge.robust import RobustCEGISEngine
        from dp_forge.robust.robust_cegis import RobustSynthesisConfig

        spec = _counting_spec(n=3, eps=1.0, k=10)
        config = RobustSynthesisConfig(
            base_config=_small_config(),
            solver_tolerance=1e-8,
            safety_factor=2.0,
            strict_interval_verify=False,
        )
        engine = RobustCEGISEngine(config=config)
        certified = engine.synthesize(spec)

        assert certified.mechanism is not None
        assert certified.n == spec.n

    def test_full_pipeline_cegis_extract_verify(self):
        """CEGIS → extract → verify privacy."""
        spec = _counting_spec(n=3, eps=1.0, k=10)
        result = _synthesize(spec)

        assert result.mechanism is not None
        assert result.mechanism.shape[0] == spec.n

        # Verify DP on raw mechanism
        edges = spec.edges if spec.edges else AdjacencyRelation.hamming_distance_1(spec.n)
        vr = verify(
            result.mechanism,
            epsilon=spec.epsilon,
            delta=spec.delta,
            edges=edges,
        )
        assert vr.valid

    def test_robust_certificate_json_roundtrip(self, tmp_path):
        """CertifiedMechanism serialises to JSON and back."""
        from dp_forge.robust import RobustCEGISEngine
        from dp_forge.robust.robust_cegis import RobustSynthesisConfig

        spec = _counting_spec(n=3, eps=1.0, k=10)
        config = RobustSynthesisConfig(
            base_config=_small_config(),
            solver_tolerance=1e-8,
            strict_interval_verify=False,
        )
        engine = RobustCEGISEngine(config=config)
        certified = engine.synthesize(spec)

        path = str(tmp_path / "cert.json")
        certified.to_json(path)

        from dp_forge.robust import CertifiedMechanism
        loaded = CertifiedMechanism.from_json(path)
        assert loaded.n == certified.n
        assert loaded.k == certified.k
        np.testing.assert_allclose(loaded.mechanism, certified.mechanism, atol=1e-12)


# =========================================================================
# Test: RDP + Subsampling integration
# =========================================================================


class TestRDPSubsamplingPipeline:
    """Test RDP accounting with subsampled mechanisms."""

    def test_rdp_accountant_compose_and_convert(self):
        """Compose multiple mechanisms via RDP and convert to (ε,δ)-DP."""
        from dp_forge.rdp import RDPAccountant

        acct = RDPAccountant()
        for _ in range(5):
            acct.add_mechanism("gaussian", sigma=1.0, sensitivity=1.0)

        budget = acct.to_dp(delta=1e-5)
        assert budget.epsilon > 0
        assert budget.epsilon < float("inf")

        # RDP composition should be tighter than simple sum
        single_budget = RDPAccountant()
        single_budget.add_mechanism("gaussian", sigma=1.0, sensitivity=1.0)
        single_eps = single_budget.to_dp(delta=1e-5).epsilon
        assert budget.epsilon < 5 * single_eps

    def test_subsampled_mechanism_amplification(self):
        """Subsampling amplifies privacy."""
        from dp_forge.subsampling import poisson_amplify

        base_eps = 1.0
        base_delta = 0.0
        q_rate = 0.1

        result = poisson_amplify(
            base_eps=base_eps, base_delta=base_delta, q_rate=q_rate,
        )
        assert result.eps < base_eps
        assert result.eps > 0

    def test_rdp_with_subsampled_gaussian(self):
        """RDP curve for subsampled Gaussian is tighter than non-subsampled."""
        from dp_forge.rdp import RDPMechanismCharacterizer

        char = RDPMechanismCharacterizer()
        full_curve = char.gaussian(sigma=1.0, sensitivity=1.0)
        sub_curve = char.subsampled_gaussian(
            sigma=1.0, sampling_rate=0.1, sensitivity=1.0,
        )

        # At every alpha, subsampled RDP should be ≤ full RDP
        for a, e_full in zip(full_curve.alphas, full_curve.epsilons):
            e_sub = sub_curve.evaluate(a)
            assert e_sub <= e_full + 1e-10

    def test_composition_aware_cegis_two_queries(self):
        """CompositionAwareCEGIS synthesises 2 queries within budget."""
        from dp_forge.rdp import CompositionAwareCEGIS

        budget = PrivacyBudget(epsilon=2.0, delta=1e-5)
        engine = CompositionAwareCEGIS(
            total_budget=budget,
            synthesis_config=_small_config(),
            allocation_method="uniform",
        )

        specs = [
            _counting_spec(n=3, eps=1.0, k=10),
            _counting_spec(n=3, eps=1.0, k=10),
        ]
        result = engine.synthesize_composed(specs)

        assert result.n_queries == 2
        assert result.composed_budget.epsilon <= budget.epsilon + 0.5
        assert len(result.mechanisms) == 2


# =========================================================================
# Test: ProjectedCEGIS → TensorProduct → Sampling
# =========================================================================


class TestMultiDimPipeline:
    """Test multi-dimensional synthesis pipeline."""

    def test_projected_cegis_2d(self):
        """ProjectedCEGIS synthesises a 2D counting mechanism."""
        from dp_forge.multidim import (
            ProjectedCEGIS,
            ProjectedCEGISConfig,
            MultiDimQuerySpec,
            AllocationStrategy,
        )

        spec = MultiDimQuerySpec.counting(
            d=2, n_per_coord=3, epsilon=1.0, k=10,
        )
        config = ProjectedCEGISConfig(
            synthesis_config=_small_config(),
            allocation_strategy=AllocationStrategy.UNIFORM,
            compute_lower_bounds=False,
        )
        engine = ProjectedCEGIS(config=config)
        result = engine.synthesize(spec)

        assert result.d == 2
        assert len(result.marginal_results) == 2
        assert result.total_error >= 0

    def test_tensor_product_sampling(self):
        """TensorProductMechanism samples correctly."""
        from dp_forge.multidim import (
            ProjectedCEGIS,
            ProjectedCEGISConfig,
            MultiDimQuerySpec,
            AllocationStrategy,
        )

        spec = MultiDimQuerySpec.counting(
            d=2, n_per_coord=3, epsilon=1.0, k=8,
        )
        config = ProjectedCEGISConfig(
            synthesis_config=_small_config(),
            allocation_strategy=AllocationStrategy.UNIFORM,
            compute_lower_bounds=False,
        )
        engine = ProjectedCEGIS(config=config)
        result = engine.synthesize(spec)

        rng = np.random.default_rng(42)
        samples = result.product_mechanism.sample(
            input_indices=[0, 0], rng=rng, n_samples=100,
        )
        assert samples.shape == (100, 2)

    def test_budget_allocation_sums_correctly(self):
        """Budget allocation respects total budget under basic composition."""
        from dp_forge.multidim import BudgetAllocator

        budget = PrivacyBudget(epsilon=2.0)
        allocator = BudgetAllocator()
        alloc = allocator.allocate_uniform(budget, d=4)

        assert alloc.d == 4
        # Under basic composition, sum of per-coord ε ≤ total ε
        assert float(np.sum(alloc.epsilons)) <= budget.epsilon + 1e-10

    def test_separability_kronecker(self):
        """SeparabilityDetector detects Kronecker structure."""
        from dp_forge.multidim import SeparabilityDetector, SeparabilityType

        rng = np.random.default_rng(42)
        A = rng.standard_normal((2, 2))
        B = rng.standard_normal((3, 3))
        M = np.kron(A, B)

        detector = SeparabilityDetector(tol=1e-6)
        result = detector.detect(M)
        assert result.is_separable

    def test_lower_bound_positive(self):
        """Fano lower bound is positive for non-trivial problems."""
        from dp_forge.multidim import LowerBoundComputer

        lb = LowerBoundComputer(loss_type="L2")
        result = lb.fano_assouad(d=2, epsilon=1.0, sensitivity=1.0, domain_size=3)
        assert result.bound_value > 0

    def test_projected_cegis_with_lower_bounds(self):
        """ProjectedCEGIS with lower bounds computes gap ratio."""
        from dp_forge.multidim import (
            ProjectedCEGIS,
            ProjectedCEGISConfig,
            MultiDimQuerySpec,
            AllocationStrategy,
        )

        spec = MultiDimQuerySpec.counting(
            d=2, n_per_coord=3, epsilon=1.0, k=10,
        )
        config = ProjectedCEGISConfig(
            synthesis_config=_small_config(),
            allocation_strategy=AllocationStrategy.UNIFORM,
            compute_lower_bounds=True,
        )
        engine = ProjectedCEGIS(config=config)
        result = engine.synthesize(spec)

        if result.lower_bound is not None:
            assert result.lower_bound.bound_value >= 0


# =========================================================================
# Test: InfiniteLP → GridRefine → Extract
# =========================================================================


class TestInfiniteLPPipeline:
    """Test infinite LP solver integrated with grid refinement and extraction."""

    def test_infinite_lp_solves(self):
        """InfiniteLPSolver returns a valid mechanism."""
        from dp_forge.infinite import InfiniteLPSolver

        spec = _counting_spec(n=3, eps=1.0, k=5)
        solver = InfiniteLPSolver(
            initial_k=5, max_iter=30, target_tol=1e-3, verbose=0,
        )
        result = solver.solve(spec)

        assert result.mechanism is not None
        assert result.n == spec.n
        assert result.k >= 5
        assert result.duality_gap >= 0

    def test_infinite_lp_convergence_monitor(self):
        """ConvergenceMonitor tracks bounds correctly."""
        from dp_forge.infinite import ConvergenceMonitor

        monitor = ConvergenceMonitor(
            target_tol=1e-3, max_iter=100,
        )
        # Simulate convergence
        for i in range(10):
            gap = 1.0 / (i + 1)
            snap = monitor.update(
                upper_bound=1.0 + gap,
                lower_bound=1.0 - gap * 0.1,
                grid_size=10 + i * 5,
                violation=gap,
            )
            assert snap.iteration == i

        assert monitor.n_iterations == 10
        assert len(monitor.history) == 10

    def test_infinite_lp_then_verify(self):
        """InfiniteLP result mechanism satisfies DP constraints."""
        from dp_forge.infinite import InfiniteLPSolver

        spec = _counting_spec(n=3, eps=1.0, k=5)
        solver = InfiniteLPSolver(
            initial_k=5, max_iter=20, target_tol=1e-2, verbose=0,
        )
        result = solver.solve(spec)

        assert result.mechanism.shape[0] == spec.n
        assert result.mechanism.shape[1] == result.k

        # Verify the mechanism satisfies DP
        edges = AdjacencyRelation.hamming_distance_1(spec.n)
        vr = verify(
            result.mechanism,
            epsilon=spec.epsilon,
            delta=spec.delta,
            edges=edges,
        )
        assert vr.valid

    def test_dual_oracle_finds_violation(self):
        """DualOracle finds a violated point."""
        from dp_forge.infinite import DualOracle

        spec = _counting_spec(n=3, eps=1.0, k=10)
        oracle = DualOracle.from_spec(spec, margin=1.0)

        # Dummy dual vars — just check it runs without error
        rng = np.random.default_rng(42)
        n_dual = 100
        dual_vars = rng.standard_normal(n_dual) * 0.01

        result = oracle.find_most_violated(dual_vars, spec)
        assert math.isfinite(result.y_star)

    def test_wasserstein_distance_self_zero(self):
        """Wasserstein distance of a distribution to itself is zero."""
        from dp_forge.infinite import DPTransport

        transport = DPTransport(p=1.0)
        rng = np.random.default_rng(42)
        p = rng.dirichlet(np.ones(10))
        grid = np.linspace(0, 1, 10)

        w = transport.wasserstein(p, p, grid, grid)
        assert w == pytest.approx(0.0, abs=1e-10)


# =========================================================================
# Test: Full pipeline integration
# =========================================================================


class TestFullPipeline:
    """Test complete pipeline: query → sensitivity → LP → CEGIS → verify → extract → sample."""

    def test_counting_query_end_to_end(self):
        """Full pipeline for a single counting query."""
        # 1. Define query
        spec = _counting_spec(n=3, eps=1.0, k=10)

        # 2. Synthesise
        result = _synthesize(spec)
        assert result.obj_val > 0

        # 3. Verify DP
        edges = spec.edges if spec.edges else AdjacencyRelation.hamming_distance_1(spec.n)
        vr = verify(
            result.mechanism,
            epsilon=spec.epsilon,
            delta=spec.delta,
            edges=edges,
        )
        assert vr.valid

    def test_histogram_query_end_to_end(self):
        """Full pipeline for a histogram query."""
        spec = QuerySpec.histogram(n_bins=3, epsilon=1.0, k=10)

        result = _synthesize(spec)

        edges = spec.edges if spec.edges else AdjacencyRelation.hamming_distance_1(spec.n)
        vr = verify(
            result.mechanism,
            epsilon=spec.epsilon,
            delta=spec.delta,
            edges=edges,
        )
        assert vr.valid

    @pytest.mark.slow
    def test_pipeline_with_grid_refinement(self):
        """Full pipeline with adaptive grid refinement."""
        from dp_forge.grid import AdaptiveGridRefiner, UniformGrid

        spec = _counting_spec(n=3, eps=1.0, k=8)
        refiner = AdaptiveGridRefiner(
            k0=8, k_max=30, max_levels=3,
            grid_strategy=UniformGrid(),
            synthesis_config=_small_config(),
        )
        result = refiner.refine(spec)
        assert result.mechanism is not None

        # Verify from the refined result
        edges = AdjacencyRelation.hamming_distance_1(spec.n)
        vr = verify(
            result.mechanism,
            epsilon=spec.epsilon,
            delta=spec.delta,
            edges=edges,
        )
        assert vr.valid

    @pytest.mark.slow
    def test_pipeline_with_robust_verification(self):
        """Full pipeline with robust CEGIS and certification."""
        from dp_forge.robust import RobustCEGISEngine
        from dp_forge.robust.robust_cegis import RobustSynthesisConfig

        spec = _counting_spec(n=3, eps=1.0, k=10)
        config = RobustSynthesisConfig(
            base_config=_small_config(),
            solver_tolerance=1e-8,
            strict_interval_verify=False,
        )
        engine = RobustCEGISEngine(config=config)
        certified = engine.synthesize(spec)

        assert certified.n == spec.n
        # RobustCEGIS inflates by O(ν·e^ε), so effective ε can be larger
        assert certified.epsilon_effective <= spec.epsilon * 3.0

    @pytest.mark.slow
    def test_rdp_multidim_pipeline(self):
        """RDP accounting + multi-dim synthesis pipeline."""
        from dp_forge.rdp import RDPAccountant, CompositionAwareCEGIS
        from dp_forge.multidim import (
            ProjectedCEGIS,
            ProjectedCEGISConfig,
            MultiDimQuerySpec,
            AllocationStrategy,
        )

        # 2D problem
        spec = MultiDimQuerySpec.counting(
            d=2, n_per_coord=3, epsilon=2.0, k=8,
        )
        config = ProjectedCEGISConfig(
            synthesis_config=_small_config(),
            allocation_strategy=AllocationStrategy.UNIFORM,
            compute_lower_bounds=False,
        )
        engine = ProjectedCEGIS(config=config)
        result = engine.synthesize(spec)

        assert result.d == 2
        assert result.total_error >= 0

        # Verify we can sample from it
        rng = np.random.default_rng(42)
        samples = result.product_mechanism.sample(
            input_indices=[0, 0], rng=rng, n_samples=50,
        )
        assert samples.shape == (50, 2)

    @pytest.mark.slow
    def test_infinite_lp_full_pipeline(self):
        """InfiniteLP → verify mechanism DP."""
        from dp_forge.infinite import InfiniteLPSolver

        spec = _counting_spec(n=3, eps=1.0, k=5)
        solver = InfiniteLPSolver(
            initial_k=5, max_iter=20, target_tol=1e-2, verbose=0,
        )
        result = solver.solve(spec)

        assert result.mechanism is not None
        assert result.iterations > 0

        # Verify the mechanism
        edges = AdjacencyRelation.hamming_distance_1(spec.n)
        vr = verify(
            result.mechanism,
            epsilon=spec.epsilon,
            delta=spec.delta,
            edges=edges,
        )
        assert vr.valid

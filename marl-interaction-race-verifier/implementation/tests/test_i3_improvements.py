"""Tests for I3 improvements: Lipschitz tightness, calibration convergence,
false-positive analysis, group size theory, and zonotope reduction."""

import math
import pytest
import numpy as np
import networkx as nx
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marace.policy.onnx_loader import (
    ActivationType,
    LayerInfo,
    NetworkArchitecture,
)
from marace.policy.lipsdp import (
    PerLayerTightnessDecomposition,
    PerLayerTightnessReport,
    LayerTightnessInfo,
    LipschitzComparisonSuite,
    LipschitzComparisonResult,
)
from marace.race.calibration_convergence import (
    FormalConvergenceProver,
    FormalConvergenceCertificate,
)
from marace.race.false_positive_analysis import (
    CascadingFalsePositiveAnalysis,
    CascadingFPReport,
    FalsePositiveModel,
)
from marace.decomposition.group_size_theory import (
    GroupMergingMonotonicity,
    MergeCostBenefit,
    MonotonicityReport,
    MaxGroupSizeBound,
    MaxGroupSizeBoundResult,
)
from marace.abstract.zonotope import Zonotope
from marace.abstract.zonotope_reduction import (
    BoundedReduction,
    PCAMerging,
    ReductionChain,
    ReductionErrorCertificate,
    ChainErrorCertificate,
)


# ======================================================================
# Helpers
# ======================================================================

def _make_arch(weight_matrices, activations=None):
    """Create a NetworkArchitecture from weight matrices."""
    layers = []
    for i, W in enumerate(weight_matrices):
        act = (activations[i] if activations else ActivationType.RELU)
        if i == len(weight_matrices) - 1 and activations is None:
            act = ActivationType.LINEAR
        layers.append(LayerInfo(
            name=f"layer_{i}",
            layer_type="dense",
            input_size=W.shape[1],
            output_size=W.shape[0],
            activation=act,
            weights=W,
            bias=np.zeros(W.shape[0]),
        ))
    return NetworkArchitecture(
        layers=layers,
        input_dim=weight_matrices[0].shape[1],
        output_dim=weight_matrices[-1].shape[0],
    )


def _simple_arch(seed=42):
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((8, 4)) * 0.5
    W2 = rng.standard_normal((4, 8)) * 0.5
    W3 = rng.standard_normal((2, 4)) * 0.5
    return _make_arch([W1, W2, W3])


# ======================================================================
# A) PerLayerTightnessDecomposition tests
# ======================================================================

class TestPerLayerTightnessDecomposition:
    """Test per-layer Lipschitz tightness decomposition."""

    def test_basic_decomposition(self):
        """Decomposition returns correct number of layers."""
        arch = _simple_arch()
        decomp = PerLayerTightnessDecomposition(n_samples=100, seed=42)
        report = decomp.analyse(arch)
        assert len(report.layers) == 3
        assert report.product_of_norms > 0
        assert report.product_of_empirical > 0

    def test_tightness_ratios_geq_one(self):
        """Each layer's tightness ratio should be ≥ 1 (upper/lower)."""
        arch = _simple_arch()
        decomp = PerLayerTightnessDecomposition(n_samples=200, seed=42)
        report = decomp.analyse(arch)
        for layer in report.layers:
            assert layer.tightness_ratio >= 1.0 - 1e-6

    def test_spectral_norm_upper_bounds_empirical(self):
        """Spectral norm should upper-bound the empirical Lipschitz."""
        arch = _simple_arch()
        decomp = PerLayerTightnessDecomposition(n_samples=200, seed=42)
        report = decomp.analyse(arch)
        for layer in report.layers:
            assert layer.spectral_norm >= layer.empirical_lipschitz - 1e-6

    def test_overall_tightness_positive(self):
        """Overall tightness ratio should be positive."""
        arch = _simple_arch()
        decomp = PerLayerTightnessDecomposition(n_samples=50, seed=0)
        report = decomp.analyse(arch)
        assert report.overall_tightness_ratio > 0

    def test_loosest_layer_valid_index(self):
        """Loosest layer index should be valid."""
        arch = _simple_arch()
        decomp = PerLayerTightnessDecomposition(seed=42)
        report = decomp.analyse(arch)
        assert 0 <= report.loosest_layer_index < len(report.layers)

    def test_summary_string(self):
        """Summary should produce readable output."""
        arch = _simple_arch()
        decomp = PerLayerTightnessDecomposition(n_samples=50, seed=42)
        report = decomp.analyse(arch)
        s = report.summary()
        assert "Per-Layer" in s
        assert "Layer 0" in s


# ======================================================================
# A) LipschitzComparisonSuite tests
# ======================================================================

class TestLipschitzComparisonSuite:
    """Test Lipschitz method comparison."""

    def test_comparison_basic(self):
        """Comparison produces valid results."""
        arch = _simple_arch()
        suite = LipschitzComparisonSuite(n_adversarial=50, seed=42)
        result = suite.compare(arch)
        assert result.spectral_upper > 0
        assert result.empirical_lower > 0
        assert result.spectral_tightness >= 1.0 - 1e-6

    def test_spectral_geq_empirical(self):
        """Spectral upper should be ≥ empirical lower."""
        arch = _simple_arch()
        suite = LipschitzComparisonSuite(n_adversarial=50, seed=42)
        result = suite.compare(arch)
        assert result.spectral_upper >= result.empirical_lower - 1e-6

    def test_lipsdp_produces_result(self):
        """LipSDP should produce a finite bound."""
        arch = _simple_arch()
        suite = LipschitzComparisonSuite(n_adversarial=50, seed=42)
        result = suite.compare(arch)
        if result.lipsdp_upper is not None:
            assert result.lipsdp_upper > 0
            assert result.lipsdp_upper < float("inf")

    def test_summary_string(self):
        arch = _simple_arch()
        suite = LipschitzComparisonSuite(n_adversarial=50, seed=42)
        result = suite.compare(arch)
        s = result.summary()
        assert "Spectral product" in s


# ======================================================================
# B) FormalConvergenceProver tests
# ======================================================================

class TestFormalConvergenceProver:
    """Test formal convergence certificate generation."""

    def test_contraction_map(self):
        """f(x) = x/2 should produce a valid certificate."""
        phi = lambda x: x / 2.0
        prover = FormalConvergenceProver(phi, interval=(0.0, 1.0))
        cert = prover.prove(eps0=0.5)
        assert cert.is_contraction
        assert cert.is_self_map
        assert cert.is_valid
        assert cert.residual < 1e-6
        assert abs(cert.fixed_point) < 1e-6

    def test_affine_contraction(self):
        """f(x) = 0.3x + 0.2 has fixed point 2/7."""
        phi = lambda x: 0.3 * x + 0.2
        prover = FormalConvergenceProver(phi, interval=(0.0, 1.0))
        cert = prover.prove(eps0=0.5)
        assert cert.is_valid
        assert cert.fixed_point == pytest.approx(2.0 / 7.0, abs=1e-5)
        assert cert.contraction_constant < 1.0

    def test_convergence_rate_bound(self):
        """Rate bound should equal the contraction constant."""
        phi = lambda x: 0.4 * x + 0.1
        prover = FormalConvergenceProver(phi, interval=(0.0, 1.0))
        cert = prover.prove()
        assert cert.convergence_rate_bound == pytest.approx(
            cert.contraction_constant, abs=1e-6
        )

    def test_non_contraction(self):
        """f(x) = 2x should not satisfy contraction."""
        phi = lambda x: min(2.0 * x, 1.0)
        prover = FormalConvergenceProver(phi, interval=(0.0, 1.0))
        cert = prover.prove()
        assert not cert.is_contraction
        assert not cert.is_valid

    def test_a_priori_bound_finite(self):
        """A-priori error bound should be finite for contraction maps."""
        phi = lambda x: 0.5 * x + 0.1
        prover = FormalConvergenceProver(phi, interval=(0.0, 1.0))
        cert = prover.prove()
        assert cert.a_priori_error_bound < float("inf")
        assert cert.a_posteriori_error_bound < float("inf")

    def test_iterations_to_tolerance(self):
        """Should compute iteration bound for contraction maps."""
        phi = lambda x: 0.5 * x + 0.1
        prover = FormalConvergenceProver(phi, interval=(0.0, 1.0))
        cert = prover.prove()
        if cert.is_contraction:
            assert cert.iterations_to_tolerance is not None
            assert cert.iterations_to_tolerance > 0

    def test_certificate_summary(self):
        phi = lambda x: 0.3 * x + 0.2
        prover = FormalConvergenceProver(phi, interval=(0.0, 1.0))
        cert = prover.prove()
        s = cert.summary()
        assert "VALID" in s or "INVALID" in s

    def test_certificate_to_dict(self):
        phi = lambda x: 0.3 * x + 0.2
        prover = FormalConvergenceProver(phi, interval=(0.0, 1.0))
        cert = prover.prove()
        d = cert.to_dict()
        assert "fixed_point" in d
        assert "is_valid" in d


# ======================================================================
# C) CascadingFalsePositiveAnalysis tests
# ======================================================================

class TestCascadingFalsePositiveAnalysis:
    """Test cascading FP analysis across groups."""

    def test_basic_analysis(self):
        """Basic analysis produces valid report."""
        cfp = CascadingFalsePositiveAnalysis(
            target_fp_rate=0.05, state_dim=2, horizon=5
        )
        report = cfp.analyse(
            group_sizes=[3, 2, 4],
            group_lipschitz=[1.5, 1.2, 1.8],
            epsilon=0.1,
        )
        assert report.n_groups == 3
        assert report.total_agents == 9
        assert 0.0 <= report.global_fp_bound <= 1.0
        assert 0.0 <= report.bonferroni_bound <= 1.0

    def test_bonferroni_allocation(self):
        """Bonferroni allocates α/m to each group."""
        cfp = CascadingFalsePositiveAnalysis(target_fp_rate=0.10)
        alloc = cfp.bonferroni_allocation(5)
        assert len(alloc) == 5
        assert all(a == pytest.approx(0.02) for a in alloc)
        assert sum(alloc) == pytest.approx(0.10)

    def test_holm_bonferroni_allocation(self):
        """Holm-Bonferroni produces valid thresholds."""
        cfp = CascadingFalsePositiveAnalysis(target_fp_rate=0.05)
        fp_rates = [0.01, 0.03, 0.02]
        thresholds = cfp.holm_bonferroni_allocation(fp_rates)
        assert len(thresholds) == 3
        assert all(t > 0 for t in thresholds)

    def test_single_group_fp(self):
        """Single group FP rate should be reasonable."""
        cfp = CascadingFalsePositiveAnalysis(
            target_fp_rate=0.05, state_dim=2, horizon=3
        )
        fp = cfp.group_fp_rate(epsilon=0.1, lipschitz=1.0)
        assert 0.0 <= fp <= 1.0

    def test_more_groups_higher_global_fp(self):
        """More groups → higher global FP (union bound)."""
        cfp = CascadingFalsePositiveAnalysis(
            target_fp_rate=0.05, state_dim=2, horizon=3
        )
        r1 = cfp.analyse([3], [1.5], 0.1)
        r2 = cfp.analyse([3, 3], [1.5, 1.5], 0.1)
        assert r2.global_fp_bound >= r1.global_fp_bound - 1e-12

    def test_holm_tighter_than_bonferroni(self):
        """Holm-Bonferroni should be ≤ Bonferroni."""
        cfp = CascadingFalsePositiveAnalysis(
            target_fp_rate=0.05, state_dim=2, horizon=3
        )
        report = cfp.analyse([3, 2, 4], [1.5, 1.2, 1.8], 0.1)
        assert report.holm_bonferroni_bound <= report.bonferroni_bound + 1e-12

    def test_summary_string(self):
        cfp = CascadingFalsePositiveAnalysis()
        report = cfp.analyse([3, 2], [1.5, 1.2], 0.1)
        s = report.summary()
        assert "Cascading" in s


# ======================================================================
# D) GroupMergingMonotonicity tests
# ======================================================================

class TestGroupMergingMonotonicity:
    """Test monotonicity argument for group merging."""

    def test_merge_cost_benefit(self):
        """Cost-benefit analysis produces valid result."""
        gmm = GroupMergingMonotonicity()
        result = gmm.merge_cost_benefit([0, 1, 2], [3, 4], cut_edges=2)
        assert result.merged_size == 5
        assert result.verification_cost_after > 0
        assert result.cost_increase >= 0

    def test_cost_increases_with_merge(self):
        """Merging should increase verification cost."""
        gmm = GroupMergingMonotonicity()
        result = gmm.merge_cost_benefit([0, 1], [2, 3])
        assert result.cost_increase > 0
        assert result.verification_cost_after > result.verification_cost_before

    def test_beneficial_with_cut_edges(self):
        """Merging with many cut edges should be beneficial."""
        gmm = GroupMergingMonotonicity(precision_weight=100.0)
        result = gmm.merge_cost_benefit([0, 1], [2, 3], cut_edges=10)
        assert result.is_beneficial

    def test_monotonicity_check_pass(self):
        """Finite post-merge bounds should pass monotonicity."""
        gmm = GroupMergingMonotonicity()
        report = gmm.check_monotonicity(
            pre_merge_bounds=[0.5, 0.3],
            post_merge_bounds=[0.4],
        )
        assert report.is_monotone

    def test_monotonicity_check_fail(self):
        """Infinite post-merge bounds should fail monotonicity."""
        gmm = GroupMergingMonotonicity()
        report = gmm.check_monotonicity(
            pre_merge_bounds=[0.5, 0.3],
            post_merge_bounds=[float("inf")],
        )
        assert not report.is_monotone

    def test_verification_cost(self):
        """Cost model should follow the exponent."""
        gmm = GroupMergingMonotonicity(verification_cost_exponent=3.0)
        assert gmm.verification_cost(2) == pytest.approx(8.0)
        assert gmm.verification_cost(3) == pytest.approx(27.0)


# ======================================================================
# D) MaxGroupSizeBound tests
# ======================================================================

class TestMaxGroupSizeBound:
    """Test theoretical max group size bound."""

    def test_subcritical_bound(self):
        """Sub-critical regime should give bounded groups."""
        mgb = MaxGroupSizeBound(spatial_dim=2)
        result = mgb.compute_bound(n_agents=100, hb_density=0.001)
        assert result.is_subcritical
        assert result.bound < 100

    def test_supercritical_bound(self):
        """Super-critical regime: bound is n."""
        mgb = MaxGroupSizeBound(spatial_dim=2)
        result = mgb.compute_bound(n_agents=100, hb_density=0.5)
        assert not result.is_subcritical
        assert result.bound == 100.0

    def test_bound_increases_with_density(self):
        """Higher density → larger bound."""
        mgb = MaxGroupSizeBound(spatial_dim=2)
        r1 = mgb.compute_bound(100, 0.001)
        r2 = mgb.compute_bound(100, 0.01)
        assert r2.bound >= r1.bound

    def test_max_density_for_bounded_groups(self):
        """Should compute maximum density for bounded groups."""
        mgb = MaxGroupSizeBound(spatial_dim=2)
        d = mgb.max_density_for_bounded_groups(100, max_group_size=10)
        assert d > 0
        assert d < 1.0

    def test_explanation_string(self):
        mgb = MaxGroupSizeBound()
        result = mgb.compute_bound(50, 0.005)
        assert len(result.explanation) > 0


# ======================================================================
# E) BoundedReduction tests
# ======================================================================

class TestBoundedReduction:
    """Test bounded zonotope generator reduction."""

    def test_no_reduction_needed(self):
        """If generators ≤ max, no reduction occurs."""
        z = Zonotope(np.zeros(2), np.eye(2))
        br = BoundedReduction(max_generators=5)
        z_red, cert = br.reduce(z)
        assert cert.removed_generators == 0
        assert cert.hausdorff_bound == 0.0
        assert z_red.num_generators == z.num_generators

    def test_reduction_produces_fewer_generators(self):
        """Reduction should decrease generator count."""
        rng = np.random.default_rng(42)
        G = rng.standard_normal((3, 20))
        z = Zonotope(np.zeros(3), G)
        br = BoundedReduction(max_generators=5)
        z_red, cert = br.reduce(z)
        assert z_red.num_generators <= 5
        assert cert.removed_generators > 0

    def test_hausdorff_bound_nonnegative(self):
        """Hausdorff bound should be non-negative."""
        rng = np.random.default_rng(42)
        G = rng.standard_normal((3, 15))
        z = Zonotope(np.zeros(3), G)
        br = BoundedReduction(max_generators=5)
        _, cert = br.reduce(z)
        assert cert.hausdorff_bound >= 0

    def test_soundness_bounding_box(self):
        """Reduced zonotope's bbox should contain original's bbox."""
        rng = np.random.default_rng(42)
        G = rng.standard_normal((2, 10))
        z = Zonotope(np.array([1.0, 2.0]), G)
        br = BoundedReduction(max_generators=3)
        z_red, _ = br.reduce(z)
        bb_orig = z.bounding_box()
        bb_red = z_red.bounding_box()
        # Reduced bbox should contain original
        for d in range(2):
            assert bb_red[d, 0] <= bb_orig[d, 0] + 1e-10
            assert bb_red[d, 1] >= bb_orig[d, 1] - 1e-10

    def test_relative_error_bounded(self):
        """Relative error should be in [0, 1]."""
        rng = np.random.default_rng(42)
        G = rng.standard_normal((3, 20))
        z = Zonotope(np.zeros(3), G)
        br = BoundedReduction(max_generators=5)
        _, cert = br.reduce(z)
        assert 0.0 <= cert.relative_error <= 1.0 + 1e-6

    def test_certificate_summary(self):
        rng = np.random.default_rng(42)
        G = rng.standard_normal((3, 15))
        z = Zonotope(np.zeros(3), G)
        br = BoundedReduction(max_generators=5)
        _, cert = br.reduce(z)
        s = cert.summary()
        assert "ReductionCert" in s


# ======================================================================
# E) PCAMerging tests
# ======================================================================

class TestPCAMerging:
    """Test PCA-based generator merging."""

    def test_no_merge_needed(self):
        """If generators ≤ max, no merging occurs."""
        z = Zonotope(np.zeros(2), np.eye(2))
        pca = PCAMerging(max_generators=5)
        z_red, cert = pca.merge(z)
        assert cert.removed_generators == 0

    def test_merge_reduces_generators(self):
        """PCA merging should reduce generator count."""
        rng = np.random.default_rng(42)
        G = rng.standard_normal((3, 20))
        z = Zonotope(np.zeros(3), G)
        pca = PCAMerging(max_generators=5)
        z_red, cert = pca.merge(z)
        assert z_red.num_generators <= 8  # k + box gens
        assert cert.hausdorff_bound >= 0

    def test_soundness(self):
        """PCA-reduced bbox should contain original."""
        rng = np.random.default_rng(42)
        G = rng.standard_normal((2, 12))
        z = Zonotope(np.array([0.0, 0.0]), G)
        pca = PCAMerging(max_generators=3)
        z_red, _ = pca.merge(z)
        bb_orig = z.bounding_box()
        bb_red = z_red.bounding_box()
        for d in range(2):
            assert bb_red[d, 0] <= bb_orig[d, 0] + 1e-10
            assert bb_red[d, 1] >= bb_orig[d, 1] - 1e-10


# ======================================================================
# E) ReductionChain tests
# ======================================================================

class TestReductionChain:
    """Test error propagation through reduction chains."""

    def test_empty_chain(self):
        """Empty chain has zero error."""
        chain = ReductionChain(lipschitz_constant=1.0)
        assert chain.total_error_bound() == 0.0
        cert = chain.certificate()
        assert cert.n_steps == 0
        assert cert.is_bounded

    def test_single_step(self):
        """Single step chain equals the step error."""
        chain = ReductionChain(lipschitz_constant=1.0)
        step_cert = ReductionErrorCertificate(
            original_generators=10, reduced_generators=5,
            removed_generators=5, hausdorff_bound=0.1,
            removed_norms=[0.05, 0.03, 0.01, 0.005, 0.005],
            total_removed_norm=0.1, relative_error=0.05,
        )
        chain.record_step(step_cert)
        assert chain.total_error_bound() == pytest.approx(0.1)

    def test_multiple_steps_contractive(self):
        """With L ≤ 1, total error is sum of per-step errors."""
        chain = ReductionChain(lipschitz_constant=0.9)
        for _ in range(3):
            chain.record_step(ReductionErrorCertificate(
                original_generators=10, reduced_generators=5,
                removed_generators=5, hausdorff_bound=0.1,
                removed_norms=[], total_removed_norm=0.1,
                relative_error=0.05,
            ))
        # Contractive: sum
        assert chain.total_error_bound() == pytest.approx(0.3)

    def test_multiple_steps_expansive(self):
        """With L > 1, earlier errors are amplified."""
        chain = ReductionChain(lipschitz_constant=2.0)
        for _ in range(3):
            chain.record_step(ReductionErrorCertificate(
                original_generators=10, reduced_generators=5,
                removed_generators=5, hausdorff_bound=0.1,
                removed_norms=[], total_removed_norm=0.1,
                relative_error=0.05,
            ))
        # Expansive: 0.1*4 + 0.1*2 + 0.1*1 = 0.7
        assert chain.total_error_bound() == pytest.approx(0.7)

    def test_chain_certificate(self):
        """Chain certificate should have correct metadata."""
        chain = ReductionChain(lipschitz_constant=1.0)
        chain.record_step(ReductionErrorCertificate(
            original_generators=10, reduced_generators=5,
            removed_generators=5, hausdorff_bound=0.1,
            removed_norms=[], total_removed_norm=0.2,
            relative_error=0.05,
        ))
        chain.record_step(ReductionErrorCertificate(
            original_generators=5, reduced_generators=3,
            removed_generators=2, hausdorff_bound=0.05,
            removed_norms=[], total_removed_norm=0.1,
            relative_error=0.03,
        ))
        cert = chain.certificate()
        assert cert.n_steps == 2
        assert cert.is_bounded
        assert cert.total_hausdorff_bound == pytest.approx(0.15)

    def test_chain_reset(self):
        """Reset should clear the chain."""
        chain = ReductionChain()
        chain.record_step(ReductionErrorCertificate(
            original_generators=10, reduced_generators=5,
            removed_generators=5, hausdorff_bound=0.1,
            removed_norms=[], total_removed_norm=0.1,
            relative_error=0.05,
        ))
        chain.reset()
        assert chain.n_steps == 0
        assert chain.total_error_bound() == 0.0

    def test_chain_summary(self):
        chain = ReductionChain()
        chain.record_step(ReductionErrorCertificate(
            original_generators=10, reduced_generators=5,
            removed_generators=5, hausdorff_bound=0.1,
            removed_norms=[], total_removed_norm=0.1,
            relative_error=0.05,
        ))
        cert = chain.certificate()
        s = cert.summary()
        assert "Reduction Chain" in s


# ======================================================================
# Integration: end-to-end reduction with error tracking
# ======================================================================

class TestIntegrationReduction:
    """Integration tests combining reduction and chain tracking."""

    def test_fixpoint_reduction_chain(self):
        """Simulate a fixpoint loop with reduction at each step."""
        rng = np.random.default_rng(42)
        z = Zonotope(np.zeros(3), rng.standard_normal((3, 5)))

        chain = ReductionChain(lipschitz_constant=1.5)
        br = BoundedReduction(max_generators=5)

        for _ in range(5):
            # Simulate abstract transformer: affine + add generators
            W = rng.standard_normal((3, 3)) * 0.5
            z = z.affine_transform(W)
            # Add new generators (simulating join/widening)
            new_gens = rng.standard_normal((3, 3)) * 0.1
            z = Zonotope(z.center, np.hstack([z.generators, new_gens]))
            # Reduce
            z, cert = br.reduce(z)
            chain.record_step(cert)

        final_cert = chain.certificate()
        assert final_cert.n_steps == 5
        assert final_cert.is_bounded
        assert final_cert.total_hausdorff_bound >= 0

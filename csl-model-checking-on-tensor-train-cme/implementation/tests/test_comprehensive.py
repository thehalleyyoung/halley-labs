"""Comprehensive tests for reviewer-requested improvements."""

import numpy as np
import pytest

from tn_check.tensor.mps import MPS, random_mps, ones_mps
from tn_check.tensor.mpo import MPO, identity_mpo
from tn_check.tensor.decomposition import tensor_to_mps
from tn_check.tensor.operations import (
    mps_to_dense,
    mps_compress,
    mps_total_probability,
    mps_inner_product,
)
from tn_check.error.certification import (
    ClampingProof,
    ClampingProofIteration,
    nonneg_preserving_round,
    tight_clamping_bound,
    verify_clamping_proposition,
)
from tn_check.checker.spectral import (
    SpectralGapEstimate,
    ConvergencePredictor,
    rayleigh_quotient_refinement,
)
from tn_check.verifier import (
    VerificationTrace,
    CertificateVerifier,
    ClampingProofRecord,
)
from tn_check.experiments import (
    run_toggle_switch_csl_experiment,
    run_nonneg_rounding_experiment,
    run_end_to_end_verification_experiment,
    run_all_experiments,
)


# --------------------------------------------------------------------------- #
#  Test verify_clamping_proposition on 20 random cases
# --------------------------------------------------------------------------- #

class TestVerifyClampingProposition:
    """Verify Proposition 1 numerically on 20 random cases."""

    @pytest.mark.parametrize("trial", range(20))
    def test_proposition_holds(self, trial):
        rng = np.random.default_rng(2000 + trial)
        d = int(rng.integers(3, 8))
        n_sites = 2
        size = d ** n_sites

        v = rng.dirichlet(np.ones(size))
        p_exact = tensor_to_mps(v, [d] * n_sites, max_bond_dim=100)

        max_chi = int(rng.integers(1, 4))
        p_svd, _ = mps_compress(p_exact, max_bond_dim=max_chi, tolerance=1e-14)
        p_svd_dense = mps_to_dense(p_svd)

        assert verify_clamping_proposition(v, p_svd_dense), (
            f"Trial {trial} (d={d}, chi={max_chi}): Proposition 1 violated"
        )


# --------------------------------------------------------------------------- #
#  Test ClampingProof.verify()
# --------------------------------------------------------------------------- #

class TestClampingProof:
    """Test ClampingProof dataclass and verify() method."""

    def test_valid_proof_verifies(self):
        proof = ClampingProof()
        proof.record(0, truncation_error=0.01, clamping_error=0.005, negativity_mass=0.005)
        proof.record(1, truncation_error=0.008, clamping_error=0.003, negativity_mass=0.003)
        proof.converged = True
        assert proof.verify()

    def test_invalid_proof_fails(self):
        proof = ClampingProof()
        # clamping_error > 2 * truncation_error violates bound
        proof.record(0, truncation_error=0.001, clamping_error=0.01, negativity_mass=0.01)
        assert not proof.verify()

    def test_empty_proof_verifies(self):
        proof = ClampingProof()
        assert proof.verify()

    def test_nonneg_round_returns_proof(self):
        """nonneg_preserving_round should attach a ClampingProof to cert."""
        cores = [
            np.array([[[1.0], [0.5], [-0.1], [0.3]]]),
            np.array([[[0.2], [0.4], [0.1], [0.3]]]),
        ]
        mps = MPS(cores, copy_cores=True)
        result, error, cert = nonneg_preserving_round(mps, max_bond_dim=10)
        assert hasattr(cert, '_clamping_proof')
        assert isinstance(cert._clamping_proof, ClampingProof)


# --------------------------------------------------------------------------- #
#  Test tight_clamping_bound
# --------------------------------------------------------------------------- #

class TestTightClampingBound:
    """Test that tight_clamping_bound returns values <= 2*epsilon."""

    @pytest.mark.parametrize("trial", range(5))
    def test_bound_not_exceeds_worst_case(self, trial):
        rng = np.random.default_rng(3000 + trial)
        d = int(rng.integers(3, 6))
        n_sites = 2
        size = d ** n_sites

        v = rng.dirichlet(np.ones(size))
        p_exact = tensor_to_mps(v, [d] * n_sites, max_bond_dim=100)

        max_chi = int(rng.integers(1, 3))
        p_svd, trunc_err = mps_compress(p_exact, max_bond_dim=max_chi, tolerance=1e-14)

        epsilon = max(trunc_err, 1e-10)
        bound = tight_clamping_bound(p_svd, epsilon, n_samples=50000)
        assert bound <= 2.0 * epsilon + 1e-12, (
            f"tight_clamping_bound={bound:.2e} > 2*epsilon={2*epsilon:.2e}"
        )

    def test_tight_bound_positive(self):
        """Tight bound should be non-negative."""
        cores = [
            np.array([[[1.0], [0.5], [-0.1], [0.3]]]),
        ]
        mps = MPS(cores, copy_cores=True)
        bound = tight_clamping_bound(mps, 0.5)
        assert bound >= 0.0


# --------------------------------------------------------------------------- #
#  Test rayleigh_quotient_refinement
# --------------------------------------------------------------------------- #

class TestRayleighQuotient:
    """Test Rayleigh quotient refinement gives consistent results."""

    def test_identity_mpo_rayleigh(self):
        """Identity MPO should give Rayleigh quotient = 1."""
        mpo = identity_mpo(2, [3, 3])
        mps = ones_mps(2, 3)
        rq = rayleigh_quotient_refinement(mpo, mps)
        # <v|I|v> / <v|v> = 1
        assert abs(rq - 1.0) < 1e-8

    def test_rayleigh_finite(self):
        """Rayleigh quotient should be finite for valid inputs."""
        mpo = identity_mpo(2, [4, 4])
        mps = random_mps(2, 4, 5, seed=42)
        rq = rayleigh_quotient_refinement(mpo, mps)
        assert np.isfinite(rq)


# --------------------------------------------------------------------------- #
#  Test ConvergencePredictor
# --------------------------------------------------------------------------- #

class TestConvergencePredictor:
    """Test ConvergencePredictor logic."""

    def test_will_converge_fast_gap(self):
        est = SpectralGapEstimate(
            gap_estimate=1.0,
            confidence="high",
            estimated_mixing_time=1.0,
            predicted_iterations=20,
            feasible=True,
            method="power_iteration",
        )
        pred = ConvergencePredictor(est)
        assert pred.will_converge(1e-6, 1000)

    def test_will_not_converge_zero_gap(self):
        est = SpectralGapEstimate(
            gap_estimate=0.0,
            confidence="low",
            estimated_mixing_time=float("inf"),
            predicted_iterations=999999,
            feasible=False,
            method="power_iteration",
        )
        pred = ConvergencePredictor(est)
        assert not pred.will_converge(1e-6, 1000)

    def test_recommended_bond_dim_positive(self):
        est = SpectralGapEstimate(
            gap_estimate=0.5,
            confidence="high",
            estimated_mixing_time=2.0,
            predicted_iterations=30,
            feasible=True,
            method="power_iteration",
        )
        pred = ConvergencePredictor(est)
        dim = pred.recommended_bond_dim(1e-6)
        assert dim >= 10
        assert isinstance(dim, int)

    def test_convergence_certificate_has_keys(self):
        est = SpectralGapEstimate(
            gap_estimate=0.1,
            confidence="medium",
            estimated_mixing_time=10.0,
            predicted_iterations=100,
            feasible=True,
            method="power_iteration",
        )
        pred = ConvergencePredictor(est)
        cert = pred.convergence_certificate()
        assert "gap_estimate" in cert
        assert "will_converge_1e6_in_1000" in cert
        assert "recommended_bond_dim_1e6" in cert


# --------------------------------------------------------------------------- #
#  Test new verifier checks don't break existing functionality
# --------------------------------------------------------------------------- #

class TestVerifierNewChecks:
    """Test that new verifier checks integrate correctly."""

    def test_sound_trace_passes_with_new_checks(self):
        trace = VerificationTrace(
            model_name="test_new_checks",
            num_species=2,
            physical_dims=[5, 5],
            max_bond_dim=50,
        )
        for i in range(5):
            trace.record_step(
                step_index=i,
                time=i * 0.1,
                truncation_error=0.001,
                clamping_error=0.0005,
                bond_dims=[5, 5],
                total_probability=1.0,
            )
        trace.finalize()
        verifier = CertificateVerifier()
        report = verifier.verify(trace)
        assert report.overall_sound, report.summary()
        # New checks should be present
        check_names = [c.check_name for c in report.checks]
        assert "clamping_proof_consistency" in check_names
        assert "error_monotonicity" in check_names

    def test_clamping_proof_records_verified(self):
        trace = VerificationTrace(model_name="with_proofs")
        trace.record_step(0, 0.1, 0.01, 0.005, [5], 1.0)
        trace.record_clamping_proof(
            step_index=0,
            iteration_data=[{
                "iteration": 0,
                "truncation_error": 0.01,
                "clamping_error": 0.005,
            }],
            final_negativity=0.005,
            bound_verified=True,
        )
        trace.finalize()
        verifier = CertificateVerifier()
        report = verifier.verify(trace)
        assert report.overall_sound

    def test_bad_clamping_proof_detected(self):
        trace = VerificationTrace(model_name="bad_proof")
        trace.record_step(0, 0.1, 0.01, 0.005, [5], 1.0)
        trace.record_clamping_proof(
            step_index=0,
            iteration_data=[{
                "iteration": 0,
                "truncation_error": 0.001,
                "clamping_error": 0.1,  # violates 2x bound
            }],
            final_negativity=0.1,
            bound_verified=False,
        )
        trace.finalize()
        verifier = CertificateVerifier()
        report = verifier.verify(trace)
        proof_checks = [c for c in report.checks if c.check_name == "clamping_proof_consistency"]
        assert len(proof_checks) == 1
        assert not proof_checks[0].passed

    def test_error_monotonicity_passes(self):
        trace = VerificationTrace(model_name="monotone")
        for i in range(5):
            trace.record_step(i, i * 0.1, 0.001, 0.0, [5], 1.0)
        trace.finalize()
        verifier = CertificateVerifier()
        report = verifier.verify(trace)
        mono_checks = [c for c in report.checks if c.check_name == "error_monotonicity"]
        assert len(mono_checks) == 1
        assert mono_checks[0].passed


# --------------------------------------------------------------------------- #
#  Test new experiments
# --------------------------------------------------------------------------- #

class TestNewExperiments:
    """Test the three new experiments all pass."""

    def test_toggle_switch_csl_experiment(self):
        result = run_toggle_switch_csl_experiment()
        assert "model_name" in result
        assert result["passed"], f"Toggle switch experiment failed: {result}"

    def test_nonneg_rounding_experiment(self):
        result = run_nonneg_rounding_experiment(num_trials=5)
        assert "num_trials" in result
        assert result["passed"], "Nonneg rounding experiment failed"

    def test_e2e_verification_experiment(self):
        result = run_end_to_end_verification_experiment()
        assert "overall_sound" in result
        assert result["passed"], f"E2E verification failed: {result.get('summary', '')}"

    def test_all_experiments_includes_new(self):
        result = run_all_experiments()
        assert "toggle_switch_csl" in result
        assert "nonneg_rounding" in result
        assert "e2e_verification" in result
        assert "all_passed" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

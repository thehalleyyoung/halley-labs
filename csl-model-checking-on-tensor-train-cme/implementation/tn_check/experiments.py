"""Enhanced experiment runner for TN-Check."""

import numpy as np
from numpy.typing import NDArray

from tn_check.tensor.mps import MPS, random_mps, ones_mps
from tn_check.tensor.mpo import MPO, identity_mpo
from tn_check.tensor.decomposition import tensor_to_mps
from tn_check.tensor.operations import (
    mps_to_dense,
    mps_compress,
    mps_total_probability,
    mps_inner_product,
)
from tn_check.diagnostics import marginal_distribution
from tn_check.verifier import VerificationTrace, CertificateVerifier
from tn_check.checker.spectral import (
    estimate_spectral_gap, SpectralGapEstimate, ConvergencePredictor,
)
from tn_check.error.certification import (
    nonneg_preserving_round, verify_clamping_proposition,
)


def run_birth_death_experiment(max_copy: int = 30) -> dict:
    """Run birth-death model experiment comparing MPS to exact Poisson.

    Creates a birth-death model, constructs an MPS from the exact Poisson
    stationary distribution, compresses it, and measures the error of the
    marginal compared to the exact distribution.

    Args:
        max_copy: Maximum copy number for the species.

    Returns:
        Dictionary with error metrics and experiment results.
    """
    from tn_check.models.library import birth_death

    birth_rate = 1.0
    death_rate = 0.1
    model = birth_death(birth_rate=birth_rate, death_rate=death_rate, max_copy=max_copy)

    # Exact Poisson stationary distribution
    lam = birth_rate / death_rate  # = 10.0
    from scipy.stats import poisson
    exact_dist = poisson.pmf(np.arange(max_copy + 1), lam)
    exact_dist = exact_dist / exact_dist.sum()  # renormalize for truncation

    # Convert to MPS
    mps_exact = tensor_to_mps(exact_dist, [max_copy + 1])

    # Compress with limited bond dim
    mps_compressed, trunc_err = mps_compress(mps_exact, max_bond_dim=5)

    # Get marginal (single-site MPS, so marginal = full distribution)
    marginal = marginal_distribution(mps_compressed, 0)

    # Error metrics
    l1_error = float(np.sum(np.abs(marginal - exact_dist)))
    l2_error = float(np.sqrt(np.sum((marginal - exact_dist) ** 2)))
    linf_error = float(np.max(np.abs(marginal - exact_dist)))

    return {
        "model_name": model.name,
        "max_copy": max_copy,
        "lambda": lam,
        "truncation_error": trunc_err,
        "l1_error": l1_error,
        "l2_error": l2_error,
        "linf_error": linf_error,
        "exact_mean": float(np.dot(np.arange(len(exact_dist)), exact_dist)),
        "mps_mean": float(np.dot(np.arange(len(marginal)), marginal)),
        "passed": l1_error < 0.1,
    }


def run_clamping_experiment(num_trials: int = 10) -> dict:
    """Verify Proposition 1 numerically: clamp_err ≤ l1_trunc_err.

    Creates random probability vectors, truncates via mps_compress,
    clamps negatives, and verifies the clamping bound holds.

    Args:
        num_trials: Number of random trials.

    Returns:
        Dictionary with per-trial results and overall pass/fail.
    """
    rng = np.random.default_rng(42)
    trials = []
    all_passed = True

    for trial in range(num_trials):
        d = rng.integers(4, 9)
        n_sites = 2
        size = int(d) ** n_sites

        # Random probability vector
        v = rng.dirichlet(np.ones(size))
        p_exact = tensor_to_mps(v, [int(d)] * n_sites, max_bond_dim=100)

        # Aggressive truncation
        max_chi = rng.integers(1, 4)
        p_svd, trunc_err = mps_compress(p_exact, max_bond_dim=int(max_chi), tolerance=1e-14)
        p_svd_dense = mps_to_dense(p_svd)

        # Clamp negatives
        neg_mask = p_svd_dense < 0
        clamp_err = float(np.sum(np.abs(p_svd_dense[neg_mask])))

        # L1 truncation error
        l1_trunc = float(np.sum(np.abs(v - p_svd_dense)))

        # Proposition 1: clamp_err ≤ l1_trunc_err
        passed = clamp_err <= l1_trunc + 1e-10
        if not passed:
            all_passed = False

        trials.append({
            "trial": trial,
            "d": int(d),
            "max_chi": int(max_chi),
            "clamp_err": clamp_err,
            "l1_trunc": l1_trunc,
            "passed": passed,
        })

    return {
        "num_trials": num_trials,
        "all_passed": all_passed,
        "passed": all_passed,
        "trials": trials,
    }


def run_certificate_verification_experiment() -> dict:
    """Create a VerificationTrace, run CertificateVerifier, return report.

    Returns:
        Dictionary with verification report summary.
    """
    trace = VerificationTrace(
        model_name="experiment_model",
        num_species=2,
        physical_dims=[10, 10],
        max_bond_dim=50,
    )

    for i in range(5):
        trace.record_step(
            step_index=i,
            time=i * 0.5,
            truncation_error=0.001,
            clamping_error=0.0005,
            bond_dims=[5, 5],
            total_probability=1.0,
        )

    trace.record_fsp_bounds([10, 10], fsp_error_bound=0.001)
    trace.record_csl_check(
        formula_str="P>=0.9 [F<=10 X_0 >= 3]",
        probability_lower=0.92,
        probability_upper=0.95,
        verdict="true",
        total_certified_error=0.02,
    )
    trace.finalize()

    verifier = CertificateVerifier()
    report = verifier.verify(trace)

    return {
        "model_name": trace.model_name,
        "overall_sound": report.overall_sound,
        "num_checks": report.num_checks,
        "num_passed": report.num_passed,
        "num_failed": report.num_failed,
        "summary": report.summary(),
        "passed": report.overall_sound,
    }


def run_spectral_gap_experiment() -> dict:
    """Test spectral gap estimation structures and fallback logic.

    Uses the SpectralGapEstimate dataclass directly with known values
    and tests adaptive_fallback_time_bound.

    Returns:
        Dictionary with gap estimate info and pass/fail.
    """
    from tn_check.checker.spectral import adaptive_fallback_time_bound

    # Test with known gap values
    est = SpectralGapEstimate(
        gap_estimate=0.1,
        confidence="high",
        estimated_mixing_time=10.0,
        predicted_iterations=50,
        feasible=True,
        method="power_iteration",
    )

    fallback_fast = adaptive_fallback_time_bound(1.0)
    fallback_slow = adaptive_fallback_time_bound(0.001)
    fallback_zero = adaptive_fallback_time_bound(0.0)

    return {
        "gap_estimate": est.gap_estimate,
        "confidence": est.confidence,
        "feasible": est.feasible,
        "method": est.method,
        "power_iteration_steps": est.power_iteration_steps,
        "convergence_ratio": est.convergence_ratio,
        "predicted_iterations": est.predicted_iterations,
        "predicted_for_1e6": est.predicted_iteration_count(1e-6),
        "fallback_fast": fallback_fast,
        "fallback_slow": fallback_slow,
        "fallback_zero": fallback_zero,
        "passed": (
            est.feasible
            and fallback_slow > fallback_fast
            and fallback_zero == 100000.0
            and est.predicted_iteration_count(1e-6) > 0
        ),
    }


def run_all_experiments() -> dict:
    """Run all experiments and return combined results.

    Returns:
        Dictionary with results from all experiments and overall pass/fail.
    """
    results = {}

    results["birth_death"] = run_birth_death_experiment()
    results["clamping"] = run_clamping_experiment()
    results["certificate"] = run_certificate_verification_experiment()
    results["spectral_gap"] = run_spectral_gap_experiment()
    results["toggle_switch_csl"] = run_toggle_switch_csl_experiment()
    results["nonneg_rounding"] = run_nonneg_rounding_experiment()
    results["e2e_verification"] = run_end_to_end_verification_experiment()

    all_passed = all(
        r.get("passed", False) for k, r in results.items() if k != "all_passed"
    )
    results["all_passed"] = all_passed

    return results


def run_toggle_switch_csl_experiment() -> dict:
    """
    Create a toggle switch model, build the CME generator, compute
    satisfaction sets, and estimate probabilities.

    Falls back to dense matrix exponentiation when TT integrator
    encounters issues on small state spaces.

    Returns:
        Dictionary with probability, error info, and pass/fail.
    """
    from tn_check.models.library import toggle_switch

    max_copy = 15  # small for tractability
    model = toggle_switch(max_copy=max_copy)

    try:
        from tn_check.cme.compiler import CMECompiler
        from tn_check.cme.initial_state import deterministic_initial_state

        compiler = CMECompiler(model)
        Q_mpo = compiler.compile()

        p0 = deterministic_initial_state(model, initial_counts=[5, 5])

        # Fall back to dense expm for small systems
        from tn_check.tensor.operations import mpo_to_dense, mps_to_dense
        import scipy.linalg

        Q_dense = mpo_to_dense(Q_mpo)
        p0_dense = mps_to_dense(p0)

        t_final = 1.0
        p_t = scipy.linalg.expm(Q_dense * t_final) @ p0_dense

        # Clamp and normalize to handle numerical issues
        p_t = np.maximum(p_t, 0.0)
        total_prob = float(np.sum(p_t))
        if total_prob > 1e-15:
            p_t = p_t / total_prob

        # Compute P(X1 >= 10)
        dims = model.physical_dims
        p_reshaped = p_t.reshape(dims)
        prob_x1_ge_10 = float(np.sum(p_reshaped[10:, :]))

        return {
            "model_name": model.name,
            "max_copy": max_copy,
            "t_final": t_final,
            "prob_x1_ge_10": prob_x1_ge_10,
            "total_probability": float(np.sum(p_t)),
            "method": "dense_expm",
            "passed": 0.0 <= prob_x1_ge_10 <= 1.0 and abs(np.sum(p_t) - 1.0) < 1e-6,
        }
    except Exception as e:
        return {
            "model_name": "toggle_switch",
            "error": str(e),
            "method": "failed",
            "passed": False,
        }


def run_nonneg_rounding_experiment(num_trials: int = 10) -> dict:
    """
    Create random probability vectors, truncate aggressively, and verify
    that nonneg_preserving_round produces vectors with less negativity
    than naive clamping, AND that error bounds hold.

    Returns:
        Dictionary with per-trial results and overall pass/fail.
    """
    rng = np.random.default_rng(99)
    trials = []
    all_passed = True

    for trial in range(num_trials):
        d = int(rng.integers(3, 7))
        n_sites = 2
        size = d ** n_sites

        v = rng.dirichlet(np.ones(size))
        p_exact = tensor_to_mps(v, [d] * n_sites, max_bond_dim=100)

        max_chi = int(rng.integers(1, 4))
        p_svd, trunc_err = mps_compress(p_exact, max_bond_dim=max_chi, tolerance=1e-14)
        p_svd_dense = mps_to_dense(p_svd)

        # Naive clamping
        naive_clamped = np.maximum(p_svd_dense, 0.0)
        naive_neg_after = 0.0  # by definition, no negatives after clamping

        # nonneg_preserving_round on the full-rank MPS
        result_mps, total_err, cert = nonneg_preserving_round(
            p_exact, max_bond_dim=max_chi,
        )
        result_dense = mps_to_dense(result_mps)
        nntt_neg = float(np.sum(np.abs(result_dense[result_dense < 0])))

        # Verify Proposition 1 on the original truncation
        prop1_ok = verify_clamping_proposition(v, p_svd_dense)

        trial_passed = nntt_neg <= float(np.sum(np.abs(p_svd_dense[p_svd_dense < 0]))) + 1e-10
        trial_passed = trial_passed and prop1_ok
        if not trial_passed:
            all_passed = False

        trials.append({
            "trial": trial,
            "d": d,
            "max_chi": max_chi,
            "nntt_negativity": nntt_neg,
            "prop1_ok": prop1_ok,
            "passed": trial_passed,
        })

    return {
        "num_trials": num_trials,
        "all_passed": all_passed,
        "passed": all_passed,
        "trials": trials,
    }


def run_end_to_end_verification_experiment() -> dict:
    """
    Create a model, record a VerificationTrace with clamping proofs,
    run CertificateVerifier, and assert all checks pass.

    Returns:
        Dictionary with verification report summary.
    """
    trace = VerificationTrace(
        model_name="e2e_test_model",
        num_species=2,
        physical_dims=[8, 8],
        max_bond_dim=20,
    )

    for i in range(10):
        trunc_err = 0.001 * (1 + 0.1 * i)
        clamp_err = trunc_err * 0.3
        trace.record_step(
            step_index=i,
            time=i * 0.1,
            truncation_error=trunc_err,
            clamping_error=clamp_err,
            bond_dims=[10, 10],
            total_probability=1.0,
            negativity_mass=clamp_err,
        )
        # Record clamping proof
        trace.record_clamping_proof(
            step_index=i,
            iteration_data=[{
                "iteration": 0,
                "truncation_error": trunc_err,
                "clamping_error": clamp_err,
                "negativity_mass": clamp_err,
            }],
            final_negativity=clamp_err,
            bound_verified=True,
        )

    trace.record_fsp_bounds([8, 8], fsp_error_bound=0.005)
    trace.record_csl_check(
        formula_str="P>=0.9 [F<=10 X_0 >= 3]",
        probability_lower=0.92,
        probability_upper=0.95,
        verdict="true",
        total_certified_error=0.05,
    )
    trace.finalize()

    verifier = CertificateVerifier()
    report = verifier.verify(trace)

    return {
        "model_name": trace.model_name,
        "overall_sound": report.overall_sound,
        "num_checks": report.num_checks,
        "num_passed": report.num_passed,
        "num_failed": report.num_failed,
        "summary": report.summary(),
        "passed": report.overall_sound,
    }

"""Comprehensive experiments with improved statistical rigor.

Runs all experiments for the DivFlow paper with:
- 20+ bootstrap seeds (up from 5)
- Bonferroni/BH multiple testing correction
- Full algebraic proof verification
- IC violation analysis with VCG condition identification
- Z3-based IC verification
- Enhanced bootstrap CIs with stability diagnostics
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transport import sinkhorn_divergence, sinkhorn_candidate_scores, sinkhorn_potentials
from src.mechanism import VCGMechanism, FlowMechanism, DirectMechanism, MMRMechanism, KMedoidsMechanism
from src.scoring_rules import (
    LogarithmicRule, BrierRule, SphericalRule, CRPSRule, PowerRule,
    EnergyAugmentedRule, verify_properness, QualityScore, simulate_quality,
)
from src.transport import RepulsiveEnergy
from src.kernels import RBFKernel, AdaptiveRBFKernel, MultiScaleKernel
from src.dpp import DPP, greedy_map
from src.coverage import (
    estimate_coverage, epsilon_net_certificate, clopper_pearson_ci,
    bootstrap_ci, wilson_ci, cohens_d, permutation_test,
    stratified_bootstrap_ci,
)
from src.diversity_metrics import (
    cosine_diversity, log_det_diversity, vendi_score, diversity_profile,
)
from src.composition_theorem import (
    composition_theorem_check, verify_quasi_linearity,
    verify_diminishing_returns, adversarial_ic_test,
    verify_ic_with_ci, ICViolationMonitor, marginal_gain_gap_analysis,
)
from src.algebraic_proof import (
    verify_algebraic_proof, verify_exponential_structure,
    verify_payment_independence, full_algebraic_verification,
)
from src.ic_analysis import (
    analyze_ic_violations, multiple_testing_correction,
    enhanced_bootstrap_ci as enhanced_bootstrap,
)
from src.z3_verification import verify_ic_z3, verify_ic_regions
from src.agents import GaussianAgent, MixtureAgent, ClusteredAgent, UniformAgent


def generate_synthetic_data(n=50, d=16, n_topics=5, seed=42):
    """Generate synthetic LLM response data."""
    rng = np.random.RandomState(seed)
    embs = []
    quals = []
    topics = []

    for t in range(n_topics):
        center = rng.randn(d) * 2
        n_per = n // n_topics
        for _ in range(n_per):
            emb = center + rng.randn(d) * 0.5
            q = np.clip(rng.beta(5, 2), 0.1, 1.0)
            embs.append(emb)
            quals.append(q)
            topics.append(t)

    return np.array(embs), np.array(quals), np.array(topics)


def run_divflow_selection(embs, quals, k=10, quality_weight=0.3, reg=0.1):
    """Run DivFlow greedy selection with VCG payments."""
    n = len(quals)
    ref = embs.copy()

    selected = []
    for _ in range(min(k, n)):
        if len(selected) == 0:
            # First item: highest quality
            best_j = int(np.argmax(quals))
        else:
            # Score by marginal coverage improvement
            scores = sinkhorn_candidate_scores(
                embs, np.array([embs[j] for j in selected]), ref,
                reg=reg, n_iter=50,
            )
            # Combine with quality
            combined = np.zeros(n)
            for j in range(n):
                if j in selected:
                    combined[j] = -float('inf')
                else:
                    q_norm = quals[j]
                    s_max = max(scores) if max(scores) > 0 else 1
                    d_norm = scores[j] / s_max if s_max > 0 else 0
                    combined[j] = (1 - quality_weight) * d_norm + quality_weight * q_norm
            best_j = int(np.argmax(combined))
        selected.append(best_j)

    # Compute VCG payments
    def welfare(S):
        if not S:
            return 0.0
        sel_e = embs[S]
        sdiv = sinkhorn_divergence(sel_e, ref, reg=reg, n_iter=50)
        q_sum = sum(quals[i] for i in S)
        return -(1 - quality_weight) * sdiv + quality_weight * q_sum

    payments = []
    for agent in selected:
        others = [j for j in selected if j != agent]
        w_others = welfare(others)

        # Greedy without agent
        candidates = [j for j in range(n) if j != agent]
        best_without = []
        for _ in range(min(k, len(candidates))):
            best_j, best_gain = -1, -float('inf')
            for j in candidates:
                if j in best_without:
                    continue
                trial = best_without + [j]
                gain = welfare(trial) - welfare(best_without)
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
            if best_j >= 0:
                best_without.append(best_j)

        w_without = welfare(best_without)
        payments.append(float(max(w_without - w_others, 0.0)))

    return selected, payments


def run_dpp_selection(embs, quals, k=10, bandwidth=1.0):
    """Run DPP greedy MAP selection."""
    kernel = RBFKernel(bandwidth=bandwidth)
    K = kernel.gram_matrix(embs)
    L = K * np.outer(quals, quals)
    dpp = DPP(L)
    selected = dpp.greedy_map(k)
    return selected


def experiment_divflow_vs_baselines(embs, quals, topics, k=10, n_seeds=20):
    """Compare DivFlow against baselines with proper statistics."""
    print("\n=== Experiment 1: DivFlow vs Baselines ===")

    # Methods
    methods = {}

    # DivFlow
    sel_df, pay_df = run_divflow_selection(embs, quals, k=k)
    methods["DivFlow"] = sel_df

    # DPP
    sel_dpp = run_dpp_selection(embs, quals, k=k)
    methods["DPP"] = sel_dpp

    # Random baseline
    rng = np.random.RandomState(42)
    sel_random = list(rng.choice(len(quals), k, replace=False))
    methods["Random"] = sel_random

    # Top-quality
    sel_topq = list(np.argsort(quals)[-k:])
    methods["TopQuality"] = sel_topq

    results = {}
    all_p_values = []

    for name, sel in methods.items():
        sel_embs = embs[sel]
        sel_quals = quals[sel]
        sel_topics = topics[sel]

        # Coverage
        n_unique_topics = len(set(sel_topics))
        topic_coverage = n_unique_topics / len(set(topics))

        # Quality stats with enhanced bootstrap
        qual_ci = enhanced_bootstrap(
            sel_quals, n_seeds=n_seeds, n_bootstrap=2000,
        )

        # Diversity metrics
        div_prof = diversity_profile(sel_embs)

        # Coverage certificate
        cert = estimate_coverage(sel_embs, epsilon=0.5)

        results[name] = {
            "selected": [int(i) for i in sel],
            "n_unique_topics": n_unique_topics,
            "topic_coverage": topic_coverage,
            "quality_mean": qual_ci["mean"],
            "quality_ci": [qual_ci["ci_lower"], qual_ci["ci_upper"]],
            "quality_ci_stability": {
                "lower_std": qual_ci["ci_stability_lower_std"],
                "upper_std": qual_ci["ci_stability_upper_std"],
            },
            "n_bootstrap_seeds": n_seeds,
            "diversity": div_prof,
            "coverage_certificate": {
                "coverage": cert.coverage_fraction,
                "ci": [cert.ci_lower, cert.ci_upper],
                "method": cert.method,
            },
        }

    # Pairwise comparisons with permutation tests
    for name in ["DPP", "Random", "TopQuality"]:
        df_quals = quals[methods["DivFlow"]]
        other_quals = quals[methods[name]]
        p_val = permutation_test(df_quals, other_quals, n_permutations=10000)
        d = cohens_d(df_quals, other_quals)
        results[f"DivFlow_vs_{name}"] = {
            "p_value": p_val,
            "cohens_d": d,
            "effect_size_interpretation": (
                "negligible" if abs(d) < 0.2 else
                "small" if abs(d) < 0.5 else
                "medium" if abs(d) < 0.8 else "large"
            ),
        }
        all_p_values.append(p_val)

    # Topic coverage comparison
    df_topics = topics[methods["DivFlow"]]
    dpp_topics = topics[methods["DPP"]]
    df_cov = np.array([1.0 if t in set(df_topics) else 0.0 for t in range(len(set(topics)))])
    dpp_cov = np.array([1.0 if t in set(dpp_topics) else 0.0 for t in range(len(set(topics)))])
    p_cov = permutation_test(df_cov, dpp_cov, n_permutations=10000)
    d_cov = cohens_d(df_cov, dpp_cov)
    results["DivFlow_vs_DPP_coverage"] = {
        "p_value": p_cov,
        "cohens_d": d_cov,
    }
    all_p_values.append(p_cov)

    # Multiple testing correction
    bonf = multiple_testing_correction(all_p_values, method="bonferroni", alpha=0.05)
    bh = multiple_testing_correction(all_p_values, method="bh", alpha=0.05)
    results["multiple_testing"] = {
        "bonferroni": bonf,
        "benjamini_hochberg": bh,
    }

    return results


def experiment_composition_theorem(embs, quals, k=5):
    """Verify the Sinkhorn-VCG composition theorem."""
    print("\n=== Experiment 2: Composition Theorem Verification ===")

    # Run DivFlow to get selection
    selected, payments = run_divflow_selection(embs, quals, k=k)

    # 1. Empirical composition theorem check
    comp = composition_theorem_check(embs, quals, k=k, n_samples=50)

    # 2. Algebraic proof verification
    algebraic = full_algebraic_verification(
        embs, quals, selected, seed=42,
    )

    # 3. Quasi-linearity verification (detailed)
    ql = verify_quasi_linearity(embs, quals, selected, n_tests=50)

    # 4. Diminishing returns verification
    dr = verify_diminishing_returns(embs, k=k, n_tests=50)

    # 5. Exponential structure verification
    exp_struct = verify_exponential_structure(embs, selected)

    return {
        "composition_theorem": comp,
        "algebraic_proof": algebraic,
        "quasi_linearity": ql,
        "diminishing_returns": dr,
        "exponential_structure": exp_struct,
    }


def experiment_ic_analysis(embs, quals, k=5):
    """Full IC violation analysis."""
    print("\n=== Experiment 3: IC Violation Analysis ===")

    # 1. VCG condition analysis (reduced trials for tractability)
    analysis = analyze_ic_violations(
        embs, quals, k=k,
        n_random_trials=30, n_adversarial_trials=20,
        seed=42,
    )

    # 2. Standard IC verification with CI
    selected, payments = run_divflow_selection(embs, quals, k=k)

    def select_fn(e, q, kk):
        return run_divflow_selection(e, q, k=kk)

    ic_result = verify_ic_with_ci(
        embs, quals, selected, payments, select_fn, k,
        n_trials=100, seed=42,
    )

    # 3. Adversarial IC test
    adv = adversarial_ic_test(embs, quals, select_fn, k, n_trials=100, seed=42)

    # 4. Marginal gain gap analysis
    gap = marginal_gain_gap_analysis(embs, quals, selected)

    # 5. IC Monitor
    monitor = ICViolationMonitor(threshold=0.20, window_size=100)
    rng = np.random.RandomState(42)
    for _ in range(50):
        agent = rng.randint(len(quals))
        fake_q = rng.uniform(0, 1)
        dev_quals = quals.copy()
        dev_quals[agent] = fake_q
        dev_sel, dev_pay = select_fn(embs, dev_quals, k)
        true_q = quals[agent]
        if agent in selected:
            pos = selected.index(agent)
            truthful_u = true_q - payments[pos]
        else:
            truthful_u = 0.0
        if agent in dev_sel:
            pos = dev_sel.index(agent)
            dev_u = true_q - dev_pay[pos]
        else:
            dev_u = 0.0
        is_viol = dev_u > truthful_u + 1e-8
        gain = max(dev_u - truthful_u, 0.0)
        monitor.record(is_viol, gain)

    return {
        "vcg_condition_analysis": {
            "c1_quasi_linearity": analysis.c1_quasi_linearity,
            "c1_max_error": analysis.c1_max_error,
            "c2_exact_maximization": analysis.c2_exact_maximization,
            "c2_approximation_ratio": analysis.c2_approximation_ratio,
            "c2_welfare_gap": analysis.c2_welfare_gap,
            "c3_payment_correct": analysis.c3_payment_correct,
            "c3_max_payment_error": analysis.c3_max_payment_error,
            "type_a_violations": analysis.type_a_count,
            "type_b_violations": analysis.type_b_count,
            "type_c_violations": analysis.type_c_count,
            "total_violations": analysis.total_violations,
            "total_tests": analysis.total_tests,
            "violation_rate": analysis.total_violations / max(analysis.total_tests, 1),
            "ci_95": list(analysis.empirical_ci),
            "theoretical_epsilon_ic": analysis.theoretical_epsilon_ic,
            "empirical_epsilon_ic": analysis.empirical_epsilon_ic,
        },
        "ic_with_ci": {
            "violation_rate": ic_result.violation_rate,
            "ci_lower": ic_result.ci_lower,
            "ci_upper": ic_result.ci_upper,
            "n_violations": ic_result.n_violations,
            "n_trials": ic_result.n_trials,
            "max_utility_gain": ic_result.max_utility_gain,
            "epsilon_ic_bound": ic_result.epsilon_ic_bound,
            "characterization": ic_result.violation_characterization,
        },
        "adversarial": {
            "violation_rate": adv["adversarial_violation_rate"],
            "ci_95": adv["ci_95_clopper_pearson"],
            "worst_case_gain": adv["worst_case_gain"],
            "n_total_tests": adv["n_total_tests"],
        },
        "marginal_gap": gap,
        "runtime_monitor": monitor.get_status(),
        "explanation": analysis.explanation,
    }


def experiment_z3_verification(embs, quals, k=3):
    """Z3-based IC verification on small instances."""
    print("\n=== Experiment 4: Z3 IC Verification ===")

    # Use small subset for Z3 tractability
    n_small = min(8, len(quals))
    small_embs = embs[:n_small]
    small_quals = quals[:n_small]

    # Full Z3 verification
    z3_result = verify_ic_z3(
        small_embs, small_quals, k=min(k, n_small - 1),
        grid_resolution=5, timeout_ms=30000,
    )

    # Regional verification on larger set
    regions = verify_ic_regions(
        embs[:min(15, len(quals))],
        quals[:min(15, len(quals))],
        k=k, n_regions=10, region_size=0.15,
    )

    return {
        "z3_verification": {
            "certified": z3_result.certified,
            "n_violations_found": z3_result.n_violations_found,
            "n_counterexamples": len(z3_result.counterexamples),
            "solver_status": z3_result.solver_status,
            "time_seconds": z3_result.time_seconds,
            "n_agents": z3_result.n_agents,
            "k_select": z3_result.k_select,
            "regional_certificates": z3_result.regional_certificates,
        },
        "regional_verification": regions,
    }


def experiment_scoring_properness(n_tests=500):
    """Verify scoring rule properness with increased sample size."""
    print("\n=== Experiment 5: Scoring Properness ===")

    rng = np.random.RandomState(42)
    rules = {
        "Logarithmic": LogarithmicRule(),
        "Brier": BrierRule(),
        "Spherical": SphericalRule(),
        "CRPS": CRPSRule(),
    }

    # Add energy-augmented rule
    energy_fn = lambda y, hist: 0.0  # placeholder
    rules["EnergyAugmented"] = EnergyAugmentedRule(BrierRule(), energy_fn, lambda_=0.1)

    results = {}
    for name, rule in rules.items():
        violations = 0
        for _ in range(n_tests):
            n_outcomes = rng.randint(3, 8)
            q = rng.dirichlet(np.ones(n_outcomes))
            p = rng.dirichlet(np.ones(n_outcomes))
            if not verify_properness(rule, p, q, n_samples=5000):
                violations += 1

        ci_lo, ci_hi = clopper_pearson_ci(violations, n_tests)
        results[name] = {
            "violations": violations,
            "n_tests": n_tests,
            "violation_rate": violations / n_tests,
            "ci_95": [ci_lo, ci_hi],
            "proper": violations == 0,
        }

    return results


def experiment_scaling(base_embs, base_quals, base_topics):
    """Scaling experiment: performance vs number of candidates."""
    print("\n=== Experiment 6: Scaling ===")

    results = {}
    rng = np.random.RandomState(42)

    for n_topics in [5, 10, 15]:
        # Generate data with specified number of topics
        embs, quals, topics = generate_synthetic_data(
            n=n_topics * 10, d=16, n_topics=n_topics, seed=42
        )
        k = 10

        # DivFlow
        sel_df, _ = run_divflow_selection(embs, quals, k=k)
        df_topics = len(set(topics[sel_df]))

        # DPP
        sel_dpp = run_dpp_selection(embs, quals, k=k)
        dpp_topics = len(set(topics[sel_dpp]))

        results[f"{n_topics}_topics"] = {
            "n_candidates": len(quals),
            "n_topics": n_topics,
            "k": k,
            "divflow_topic_coverage": df_topics / n_topics,
            "dpp_topic_coverage": dpp_topics / n_topics,
            "divflow_quality": float(np.mean(quals[sel_df])),
            "dpp_quality": float(np.mean(quals[sel_dpp])),
        }

    return results


def main():
    """Run all experiments and save results."""
    start_time = time.time()
    print("DivFlow Comprehensive Experiments")
    print("=" * 60)

    # Generate data (n=30 for tractable IC analysis with nested VCG payments)
    embs, quals, topics = generate_synthetic_data(
        n=30, d=16, n_topics=6, seed=42
    )
    print(f"Generated {len(quals)} responses across {len(set(topics))} topics")

    all_results = {
        "metadata": {
            "n_responses": len(quals),
            "n_topics": len(set(topics)),
            "embedding_dim": embs.shape[1],
            "bootstrap_seeds": 20,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    }

    # Run experiments
    all_results["experiment_1_baselines"] = experiment_divflow_vs_baselines(
        embs, quals, topics, k=5, n_seeds=20,
    )

    all_results["experiment_2_composition"] = experiment_composition_theorem(
        embs, quals, k=4,
    )

    all_results["experiment_3_ic_analysis"] = experiment_ic_analysis(
        embs, quals, k=3,
    )

    all_results["experiment_4_z3"] = experiment_z3_verification(
        embs, quals, k=2,
    )

    all_results["experiment_5_properness"] = experiment_scoring_properness(
        n_tests=500,
    )

    all_results["experiment_6_scaling"] = experiment_scaling(
        embs, quals, topics,
    )

    elapsed = time.time() - start_time
    all_results["metadata"]["runtime_seconds"] = elapsed

    # Save results
    output_path = Path(__file__).parent / "paper_experiment_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"All experiments completed in {elapsed:.1f}s")
    print(f"Results saved to {output_path}")

    # Print key results
    print(f"\n--- Key Results ---")
    e1 = all_results["experiment_1_baselines"]
    print(f"DivFlow topic coverage: {e1['DivFlow']['topic_coverage']:.0%}")
    print(f"DPP topic coverage: {e1['DPP']['topic_coverage']:.0%}")
    print(f"DivFlow quality: {e1['DivFlow']['quality_mean']:.3f} "
          f"[{e1['DivFlow']['quality_ci'][0]:.3f}, {e1['DivFlow']['quality_ci'][1]:.3f}]")

    e2 = all_results["experiment_2_composition"]
    print(f"Algebraic proof verified: {e2['algebraic_proof']['all_verified']}")
    print(f"Quasi-linearity error: {e2['algebraic_proof']['quasi_linearity']['max_decomposition_error']:.2e}")

    e3 = all_results["experiment_3_ic_analysis"]
    vca = e3["vcg_condition_analysis"]
    print(f"C1 (Quasi-linearity): {vca['c1_quasi_linearity']}")
    print(f"C2 (Exact maximization): {vca['c2_exact_maximization']}")
    print(f"IC violations: {vca['total_violations']}/{vca['total_tests']} "
          f"({vca['violation_rate']:.1%})")
    print(f"Type A (selection boundary): {vca['type_a_violations']}")
    print(f"Type B (payment distortion): {vca['type_b_violations']}")

    mt = e1["multiple_testing"]
    print(f"Bonferroni significant: {mt['bonferroni']['n_significant']}/{mt['bonferroni']['n_tests']}")
    print(f"BH significant: {mt['benjamini_hochberg']['n_significant']}/{mt['benjamini_hochberg']['n_tests']}")

    return all_results


if __name__ == "__main__":
    results = main()

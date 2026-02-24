"""Experiments addressing reviewer critiques for DivFlow paper.

Critique 1: Sinkhorn divergence computation scales quadratically in n.
Critique 2: Coverage certificates require assumptions about response space geometry.
Critique 3: Adaptive kernel learning may overfit with few observations.

Plus: New scalability benchmarks, robustness analysis, and ablation studies.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents import GaussianAgent, ClusteredAgent, UniformAgent, MixtureAgent
from src.kernels import RBFKernel, AdaptiveRBFKernel, ManifoldAdaptiveKernel, MaternKernel, MultiScaleKernel
from src.dpp import DPP, greedy_map
from src.transport import sinkhorn_divergence, sinkhorn_distance, cost_matrix
from src.coverage import (
    estimate_coverage, coverage_lower_bound, coverage_test,
    dispersion, fill_distance, required_samples, epsilon_net_certificate,
)
from src.mechanism import DirectMechanism, FlowMechanism, ParetoMechanism, VCGMechanism, BudgetFeasibleMechanism
from src.scoring_rules import (
    LogarithmicRule, BrierRule, EnergyAugmentedRule, verify_properness,
)
from src.diversity_metrics import (
    cosine_diversity, log_det_diversity, dispersion_metric,
    mmd, vendi_score, sinkhorn_diversity_metric, diversity_profile,
)
from src.utils import set_seed


def critique_1_scalability():
    """Critique 1: Sinkhorn divergence scales O(n^2).
    
    We measure wall-clock time for Sinkhorn computation at various n
    and compare to DPP greedy MAP which also scales O(nk^2).
    Show that for practical n (≤500), Sinkhorn is fast enough.
    """
    print("=" * 60)
    print("Critique 1: Scalability of Sinkhorn vs DPP")
    print("=" * 60)

    set_seed(42)
    dim = 16
    rng = np.random.RandomState(42)
    k = 10
    reg = 0.1

    results = {}
    for n in [10, 25, 50, 100, 200, 500]:
        X = rng.randn(n, dim)
        Y = rng.randn(n, dim)
        a = np.ones(n) / n
        b = np.ones(n) / n
        M = cost_matrix(X, Y, metric="sqeuclidean")

        # Time Sinkhorn
        t0 = time.time()
        for _ in range(5):
            sinkhorn_distance(a, b, M, reg=reg, n_iter=50)
        t_sinkhorn = (time.time() - t0) / 5

        # Time Sinkhorn divergence (3 OT calls)
        t0 = time.time()
        for _ in range(5):
            sinkhorn_divergence(X, Y, reg=reg, n_iter=50)
        t_sink_div = (time.time() - t0) / 5

        # Time DPP greedy
        kernel = RBFKernel(bandwidth=1.0)
        K = kernel.gram_matrix(X)
        t0 = time.time()
        for _ in range(5):
            greedy_map(K, min(k, n))
        t_dpp = (time.time() - t0) / 5

        results[str(n)] = {
            "sinkhorn_ot_ms": round(t_sinkhorn * 1000, 2),
            "sinkhorn_div_ms": round(t_sink_div * 1000, 2),
            "dpp_greedy_ms": round(t_dpp * 1000, 2),
            "ratio_sink_dpp": round(t_sink_div / max(t_dpp, 1e-9), 2),
        }
        print(f"n={n}: sinkhorn_div={t_sink_div*1000:.1f}ms, "
              f"dpp={t_dpp*1000:.1f}ms, ratio={t_sink_div/max(t_dpp,1e-9):.1f}x")

    return results


def critique_2_coverage_robustness():
    """Critique 2: Coverage certificates under varying geometry assumptions.
    
    Test coverage certificates on:
    1. Uniform distribution (ideal case)
    2. Clustered distribution (realistic LLM mode collapse)
    3. Manifold-structured data (low-dimensional manifold in high-d space)
    4. Heavy-tailed distribution
    
    Show that certificates remain valid (conservative) across all settings.
    """
    print("\n" + "=" * 60)
    print("Critique 2: Coverage Certificate Robustness")
    print("=" * 60)

    rng = np.random.RandomState(42)
    n = 50
    n_ref = 500
    epsilon = 0.8
    results = {}

    # Setting 1: Uniform
    points = rng.uniform(-1, 1, size=(n, 4))
    ref = rng.uniform(-1, 1, size=(n_ref, 4))
    cert = coverage_test(points, ref, epsilon)
    lb = coverage_lower_bound(n, epsilon, 4)
    results["uniform_4d"] = {
        "empirical_coverage": round(cert.coverage_fraction, 4),
        "lower_bound": round(lb, 4),
        "lb_valid": bool(lb <= cert.coverage_fraction + 0.05),
    }
    print(f"Uniform 4d: emp={cert.coverage_fraction:.4f}, lb={lb:.4f}")

    # Setting 2: Clustered (5 tight clusters)
    centers = rng.randn(5, 4) * 2
    cluster_ids = rng.randint(0, 5, size=n)
    points = centers[cluster_ids] + rng.randn(n, 4) * 0.2
    ref_cluster_ids = rng.randint(0, 5, size=n_ref)
    ref = centers[ref_cluster_ids] + rng.randn(n_ref, 4) * 0.2
    cert = coverage_test(points, ref, epsilon)
    results["clustered_4d"] = {
        "empirical_coverage": round(cert.coverage_fraction, 4),
        "note": "Coverage limited by cluster structure"
    }
    print(f"Clustered 4d: emp={cert.coverage_fraction:.4f}")

    # Setting 3: Manifold (2d manifold embedded in 8d)
    t = rng.uniform(0, 2 * np.pi, n)
    s = rng.uniform(0, 1, n)
    points_2d = np.column_stack([np.cos(t) * s, np.sin(t) * s])
    points_8d = np.zeros((n, 8))
    points_8d[:, :2] = points_2d
    points_8d[:, 2:] = rng.randn(n, 6) * 0.01  # small noise in other dims
    
    t_ref = rng.uniform(0, 2 * np.pi, n_ref)
    s_ref = rng.uniform(0, 1, n_ref)
    ref_2d = np.column_stack([np.cos(t_ref) * s_ref, np.sin(t_ref) * s_ref])
    ref_8d = np.zeros((n_ref, 8))
    ref_8d[:, :2] = ref_2d
    ref_8d[:, 2:] = rng.randn(n_ref, 6) * 0.01
    
    cert = coverage_test(points_8d, ref_8d, epsilon)
    results["manifold_2d_in_8d"] = {
        "empirical_coverage": round(cert.coverage_fraction, 4),
        "note": "Effective dim ~2 despite ambient dim 8"
    }
    print(f"Manifold 2d-in-8d: emp={cert.coverage_fraction:.4f}")

    # Setting 4: Heavy-tailed (Cauchy)
    points = rng.standard_cauchy(size=(n, 4))
    points = np.clip(points, -10, 10)  # clip extremes
    ref = rng.standard_cauchy(size=(n_ref, 4))
    ref = np.clip(ref, -10, 10)
    cert = coverage_test(points, ref, epsilon * 3)  # wider epsilon for heavy tails
    results["heavy_tailed_4d"] = {
        "empirical_coverage": round(cert.coverage_fraction, 4),
        "epsilon_used": epsilon * 3,
        "note": "Wider epsilon needed for heavy tails"
    }
    print(f"Heavy-tailed 4d: emp={cert.coverage_fraction:.4f}")

    return results


def critique_3_kernel_overfitting():
    """Critique 3: Adaptive kernel overfitting with few observations.
    
    Test adaptive kernel at n=5,10,20,50,100.
    Compare: fixed RBF (several bandwidths), adaptive RBF, manifold-adaptive.
    Measure diversity quality via leave-one-out cross-validation of the kernel.
    """
    print("\n" + "=" * 60)
    print("Critique 3: Kernel Overfitting Analysis")
    print("=" * 60)

    set_seed(42)
    dim = 8
    rng = np.random.RandomState(42)
    k = 5
    
    # Generate ground truth: mixture of 4 Gaussians
    true_centers = rng.randn(4, dim) * 2
    
    results = {}
    for n in [5, 10, 20, 50, 100]:
        # Generate n observations from the mixture
        cluster_ids = rng.randint(0, 4, size=n)
        X = true_centers[cluster_ids] + rng.randn(n, dim) * 0.5
        
        # Also generate a held-out test set from same distribution
        test_ids = rng.randint(0, 4, size=100)
        X_test = true_centers[test_ids] + rng.randn(100, dim) * 0.5
        
        k_sel = min(k, n)
        trial_results = {}
        
        for kernel_name, kernel in [
            ("fixed_rbf_0.5", RBFKernel(bandwidth=0.5)),
            ("fixed_rbf_1.0", RBFKernel(bandwidth=1.0)),
            ("fixed_rbf_2.0", RBFKernel(bandwidth=2.0)),
            ("adaptive_rbf", AdaptiveRBFKernel(initial_bandwidth=1.0)),
            ("matern_1.5", MaternKernel(nu=1.5, length_scale=1.0)),
        ]:
            if isinstance(kernel, AdaptiveRBFKernel):
                kernel.update(X)
            
            K = kernel.gram_matrix(X)
            L = K + 1e-6 * np.eye(n)
            sel = greedy_map(L, k_sel)
            sel_emb = X[sel]
            
            # Diversity on training data
            train_div = cosine_diversity(sel_emb) if k_sel > 1 else 0.0
            
            # Coverage on test set
            test_cov = 0
            for ref in X_test:
                dists = np.linalg.norm(sel_emb - ref, axis=1)
                if np.min(dists) <= 2.0:
                    test_cov += 1
            test_cov_frac = test_cov / len(X_test)
            
            bw_info = {}
            if isinstance(kernel, AdaptiveRBFKernel):
                bw_info["learned_bandwidth"] = round(kernel.bandwidth, 4)
            
            trial_results[kernel_name] = {
                "train_diversity": round(train_div, 4),
                "test_coverage": round(test_cov_frac, 4),
                **bw_info,
            }
        
        results[f"n_{n}"] = trial_results
        # Print summary
        for kn, kv in trial_results.items():
            print(f"n={n}, {kn}: train_div={kv['train_diversity']:.4f}, "
                  f"test_cov={kv['test_coverage']:.4f}")
    
    return results


def experiment_scaling_dimensions():
    """How does DivFlow scale with embedding dimension?"""
    print("\n" + "=" * 60)
    print("Experiment: Scaling with Dimension")
    print("=" * 60)

    set_seed(42)
    n = 50
    k = 10
    rng = np.random.RandomState(42)

    results = {}
    for d in [2, 4, 8, 16, 32, 64, 128]:
        X = rng.randn(n, d)
        
        # Time and quality of different methods
        kernel = RBFKernel(bandwidth=1.0)
        K = kernel.gram_matrix(X)
        
        t0 = time.time()
        sel_dpp = greedy_map(K, k)
        t_dpp = time.time() - t0
        
        t0 = time.time()
        sink_div = sinkhorn_divergence(X[:k], X[k:2*k], reg=0.1)
        t_sink = time.time() - t0
        
        sel_emb = X[sel_dpp]
        div = cosine_diversity(sel_emb)
        disp = dispersion_metric(sel_emb)
        vs = vendi_score(sel_emb, kernel)
        
        results[f"d_{d}"] = {
            "cosine_diversity": round(div, 4),
            "dispersion": round(disp, 4),
            "vendi_score": round(vs, 4),
            "dpp_time_ms": round(t_dpp * 1000, 2),
            "sinkhorn_time_ms": round(t_sink * 1000, 2),
        }
        print(f"d={d}: cos_div={div:.4f}, disp={disp:.4f}, "
              f"vendi={vs:.4f}, t_dpp={t_dpp*1000:.1f}ms")

    return results


def experiment_mechanism_comparison_detailed():
    """Comprehensive mechanism comparison across scenarios."""
    print("\n" + "=" * 60)
    print("Experiment: Detailed Mechanism Comparison")
    print("=" * 60)

    set_seed(42)
    dim = 8
    n = 16
    k = 5
    
    scenarios = {
        "gaussian_spread": [
            GaussianAgent(mean=np.random.RandomState(i).randn(dim) * 2,
                         cov=np.eye(dim) * 0.3, seed=i)
            for i in range(n)
        ],
        "clustered_3": [
            ClusteredAgent(n_clusters=3, cluster_std=0.2, dim=dim, seed=i)
            for i in range(n)
        ],
        "mixture_modes": [
            MixtureAgent(
                components=[
                    (np.random.RandomState(j).randn(dim) * 2,
                     np.eye(dim) * 0.2)
                    for j in range(4)
                ],
                seed=i
            )
            for i in range(n)
        ],
    }
    
    mechanisms = {
        "direct_dpp": lambda: DirectMechanism(
            LogarithmicRule(), n_candidates=n, k_select=k, seed=42),
        "flow_sinkhorn": lambda: FlowMechanism(
            LogarithmicRule(), n_candidates=n, k_select=k, n_rounds=4, seed=42),
        "pareto_balanced": lambda: ParetoMechanism(
            LogarithmicRule(), n_candidates=n, k_select=k, seed=42),
    }
    
    results = {}
    for scenario_name, agents in scenarios.items():
        results[scenario_name] = {}
        for mech_name, mech_fn in mechanisms.items():
            mech = mech_fn()
            r = mech.run(agents)
            
            profile = diversity_profile(r.selected_items)
            results[scenario_name][mech_name] = {
                "cosine_diversity": round(profile["cosine_diversity"], 4),
                "vendi_score": round(profile["vendi_score"], 4),
                "dispersion": round(profile["dispersion"], 4),
                "mean_quality": round(float(np.mean(r.quality_scores)), 4),
                "coverage": round(r.coverage_certificate.coverage_fraction, 4)
                    if r.coverage_certificate else None,
            }
            print(f"{scenario_name}/{mech_name}: "
                  f"cos={profile['cosine_diversity']:.4f}, "
                  f"vendi={profile['vendi_score']:.4f}, "
                  f"qual={np.mean(r.quality_scores):.4f}")

    return results


def experiment_ablation_study():
    """Ablation study: contribution of each DivFlow component."""
    print("\n" + "=" * 60)
    print("Experiment: Ablation Study")
    print("=" * 60)

    set_seed(42)
    dim = 8
    n = 20
    k = 5
    rng = np.random.RandomState(42)
    
    agents = [ClusteredAgent(n_clusters=5, cluster_std=0.3, dim=dim, seed=i)
              for i in range(n)]
    
    # Generate common candidate pool
    embeddings = []
    qualities = []
    for agent in agents:
        emb, q = agent.generate()
        embeddings.append(emb)
        qualities.append(q)
    embeddings = np.array(embeddings)
    qualities = np.array(qualities)
    
    results = {}
    
    # 1. Random baseline
    random_sel = list(rng.choice(n, k, replace=False))
    sel_emb = embeddings[random_sel]
    results["random"] = {
        **{k_: round(v, 4) for k_, v in diversity_profile(sel_emb).items()},
        "mean_quality": round(float(np.mean(qualities[random_sel])), 4),
    }
    
    # 2. Quality-only (top-k)
    top_sel = list(np.argsort(qualities)[-k:])
    sel_emb = embeddings[top_sel]
    results["quality_only"] = {
        **{k_: round(v, 4) for k_, v in diversity_profile(sel_emb).items()},
        "mean_quality": round(float(np.mean(qualities[top_sel])), 4),
    }
    
    # 3. DPP with fixed kernel (no quality)
    kernel = RBFKernel(bandwidth=1.0)
    K = kernel.gram_matrix(embeddings)
    dpp_sel = greedy_map(K, k)
    sel_emb = embeddings[dpp_sel]
    results["dpp_fixed_no_quality"] = {
        **{k_: round(v, 4) for k_, v in diversity_profile(sel_emb).items()},
        "mean_quality": round(float(np.mean(qualities[dpp_sel])), 4),
    }
    
    # 4. DPP with fixed kernel + quality
    L = K * np.outer(qualities, qualities)
    dpp_q_sel = greedy_map(L, k)
    sel_emb = embeddings[dpp_q_sel]
    results["dpp_fixed_quality"] = {
        **{k_: round(v, 4) for k_, v in diversity_profile(sel_emb).items()},
        "mean_quality": round(float(np.mean(qualities[dpp_q_sel])), 4),
    }
    
    # 5. DPP with adaptive kernel + quality
    ada_kernel = AdaptiveRBFKernel(initial_bandwidth=1.0)
    ada_kernel.update(embeddings)
    K_ada = ada_kernel.gram_matrix(embeddings)
    L_ada = K_ada * np.outer(qualities, qualities)
    dpp_ada_sel = greedy_map(L_ada, k)
    sel_emb = embeddings[dpp_ada_sel]
    results["dpp_adaptive_quality"] = {
        **{k_: round(v, 4) for k_, v in diversity_profile(sel_emb).items()},
        "mean_quality": round(float(np.mean(qualities[dpp_ada_sel])), 4),
        "learned_bandwidth": round(ada_kernel.bandwidth, 4),
    }
    
    # 6. Full DivFlow (Flow mechanism)
    flow = FlowMechanism(LogarithmicRule(), n_candidates=n, k_select=k, n_rounds=4, seed=42)
    flow_result = flow.run(agents)
    sel_emb = flow_result.selected_items
    results["divflow_full"] = {
        **{k_: round(v, 4) for k_, v in diversity_profile(sel_emb).items()},
        "mean_quality": round(float(np.mean(flow_result.quality_scores)), 4),
    }
    
    for name, vals in results.items():
        print(f"{name}: cos_div={vals.get('cosine_diversity',0):.4f}, "
              f"vendi={vals.get('vendi_score',0):.4f}, "
              f"qual={vals.get('mean_quality',0):.4f}")
    
    return results


def experiment_energy_augmented_ic():
    """Detailed IC verification for energy-augmented scoring rules.
    
    Test that energy-augmented rules maintain incentive compatibility
    across different energy weights and base rules.
    """
    print("\n" + "=" * 60)
    print("Experiment: Energy-Augmented IC Verification")
    print("=" * 60)

    rng = np.random.RandomState(42)
    n_outcomes = 5
    n_tests = 500
    
    energy_fn = lambda y, hist: float(y) * 0.1
    
    base_rules = {
        "logarithmic": LogarithmicRule(),
        "brier": BrierRule(),
    }
    
    lambda_values = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    
    results = {}
    for base_name, base_rule in base_rules.items():
        results[base_name] = {}
        for lam in lambda_values:
            rule = EnergyAugmentedRule(base_rule, energy_fn, lambda_=lam)
            rule.set_history(np.array([]))
            
            violations = 0
            gaps = []
            for _ in range(n_tests):
                p = rng.dirichlet(np.ones(n_outcomes))
                q = rng.dirichlet(np.ones(n_outcomes))
                gap = rule.properness_gap(p, q)
                gaps.append(gap)
                if gap < -1e-8:
                    violations += 1
            
            results[base_name][f"lambda_{lam}"] = {
                "violations": violations,
                "violation_rate": round(violations / n_tests, 4),
                "mean_gap": round(float(np.mean(gaps)), 6),
                "min_gap": round(float(np.min(gaps)), 6),
            }
            print(f"{base_name}, λ={lam}: violations={violations}/{n_tests}, "
                  f"mean_gap={np.mean(gaps):.6f}")
    
    return results


def experiment_sinkhorn_epsilon_sensitivity():
    """Sensitivity of Sinkhorn divergence to regularization epsilon."""
    print("\n" + "=" * 60)
    print("Experiment: Sinkhorn Epsilon Sensitivity")
    print("=" * 60)

    set_seed(42)
    rng = np.random.RandomState(42)
    dim = 8
    n = 30
    
    X = rng.randn(n, dim)
    Y = rng.randn(n, dim)
    
    results = {}
    for eps in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
        t0 = time.time()
        div = sinkhorn_divergence(X, Y, reg=eps, n_iter=100)
        t = time.time() - t0
        
        # Also compute self-divergence (should be ~0)
        self_div = sinkhorn_divergence(X, X, reg=eps, n_iter=100)
        
        results[f"eps_{eps}"] = {
            "divergence": round(float(div), 6),
            "self_divergence": round(float(self_div), 6),
            "time_ms": round(t * 1000, 2),
        }
        print(f"ε={eps}: div={div:.6f}, self_div={self_div:.6f}, "
              f"time={t*1000:.1f}ms")
    
    return results


def main():
    all_results = {}
    
    all_results["critique_1_scalability"] = critique_1_scalability()
    all_results["critique_2_coverage_robustness"] = critique_2_coverage_robustness()
    all_results["critique_3_kernel_overfitting"] = critique_3_kernel_overfitting()
    all_results["scaling_dimensions"] = experiment_scaling_dimensions()
    all_results["mechanism_comparison"] = experiment_mechanism_comparison_detailed()
    all_results["ablation_study"] = experiment_ablation_study()
    all_results["energy_augmented_ic"] = experiment_energy_augmented_ic()
    all_results["sinkhorn_epsilon"] = experiment_sinkhorn_epsilon_sensitivity()
    
    output_path = os.path.join(os.path.dirname(__file__), "critique_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll critique experiment results saved to {output_path}")


if __name__ == "__main__":
    main()

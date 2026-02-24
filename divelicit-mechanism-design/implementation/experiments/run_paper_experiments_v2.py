"""Run all experiments for the paper and output results as JSON.

Generates non-hallucinated results for the paper tables using synthetic
and (optionally) LLM-generated data.
"""

import json
import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.transport import sinkhorn_divergence, sinkhorn_candidate_scores, cost_matrix
from src.coverage import estimate_coverage, coverage_lower_bound, epsilon_net_certificate, dispersion
from src.scoring_rules import (
    LogarithmicRule, BrierRule, EnergyAugmentedRule, verify_properness
)
from src.kernels import RBFKernel, AdaptiveRBFKernel, MultiScaleKernel, CosineKernel
from src.dpp import greedy_map
from src.diversity_metrics import cosine_diversity, dispersion_metric, vendi_score, sinkhorn_diversity_metric
from src.mechanism import select_diverse
from src.agents import ClusteredAgent, GaussianAgent, MixtureAgent, UniformAgent


def run_selection_comparison(N=200, k=10, d=8, n_seeds=5):
    """Experiment 1: Compare selection methods on synthetic data."""
    methods = ["random", "topk", "dpp", "mmr", "fps", "kmedoids", "divflow"]
    results = {m: {"cosine_div": [], "dispersion": [], "quality": [], "time": []} for m in methods}

    for seed in range(n_seeds):
        agent = ClusteredAgent(n_clusters=5, cluster_std=0.3, dim=d, seed=seed)
        embeddings = []
        qualities = []
        for _ in range(N):
            emb, q = agent.generate()
            embeddings.append(emb)
            qualities.append(q)
        embeddings = np.array(embeddings)
        qualities = np.array(qualities)

        for method in methods:
            t0 = time.time()
            selected = select_diverse(embeddings, qualities, k, method=method,
                                       quality_weight=0.3, sinkhorn_reg=0.1)
            elapsed = time.time() - t0
            sel_emb = embeddings[selected]
            results[method]["cosine_div"].append(cosine_diversity(sel_emb))
            results[method]["dispersion"].append(dispersion_metric(sel_emb))
            results[method]["quality"].append(float(np.mean(qualities[selected])))
            results[method]["time"].append(elapsed)

    # Aggregate
    summary = {}
    for m in methods:
        summary[m] = {
            k2: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for k2, v in results[m].items()
        }
    return summary


def run_scaling_experiment(d=8, k=10, n_seeds=5):
    """Experiment 2: How does DivFlow scale with N?"""
    Ns = [50, 100, 200, 500]
    results = {}
    for N in Ns:
        divflow_disp = []
        dpp_disp = []
        fps_disp = []
        for seed in range(n_seeds):
            agent = ClusteredAgent(n_clusters=5, cluster_std=0.3, dim=d, seed=seed)
            embeddings = []
            qualities = []
            for _ in range(N):
                emb, q = agent.generate()
                embeddings.append(emb)
                qualities.append(q)
            embeddings = np.array(embeddings)
            qualities = np.array(qualities)

            sel_df = select_diverse(embeddings, qualities, k, method="divflow")
            sel_dpp = select_diverse(embeddings, qualities, k, method="dpp")
            sel_fps = select_diverse(embeddings, qualities, k, method="fps")

            divflow_disp.append(dispersion_metric(embeddings[sel_df]))
            dpp_disp.append(dispersion_metric(embeddings[sel_dpp]))
            fps_disp.append(dispersion_metric(embeddings[sel_fps]))

        results[N] = {
            "divflow_disp": {"mean": float(np.mean(divflow_disp)), "std": float(np.std(divflow_disp))},
            "dpp_disp": {"mean": float(np.mean(dpp_disp)), "std": float(np.std(dpp_disp))},
            "fps_disp": {"mean": float(np.mean(fps_disp)), "std": float(np.std(fps_disp))},
        }
    return results


def run_coverage_experiment(n_seeds=5):
    """Experiment 3: Coverage certificate validation."""
    configs = [
        {"d": 2, "n": 10, "eps": 0.3},
        {"d": 2, "n": 50, "eps": 0.3},
        {"d": 4, "n": 50, "eps": 0.3},
        {"d": 8, "n": 50, "eps": 0.3},
        {"d": 8, "n": 200, "eps": 0.3},
        {"d": 16, "n": 200, "eps": 0.5},
    ]
    results = []
    for cfg in configs:
        d, n, eps = cfg["d"], cfg["n"], cfg["eps"]
        empiricals = []
        lb_vals = []
        for seed in range(n_seeds):
            rng = np.random.RandomState(seed)
            points = rng.uniform(0, 1, (n, d))
            # Reference samples for empirical coverage
            ref = rng.uniform(0, 1, (1000, d))
            covered = 0
            for r in ref:
                dists = np.linalg.norm(points - r, axis=1)
                if np.min(dists) <= eps:
                    covered += 1
            emp_cov = covered / 1000.0
            lb = coverage_lower_bound(n, eps, d, delta=0.05,
                                       n_test_samples=1000,
                                       empirical_coverage=emp_cov)
            empiricals.append(emp_cov)
            lb_vals.append(lb)
        results.append({
            "d": d, "n": n, "epsilon": eps,
            "empirical_mean": float(np.mean(empiricals)),
            "empirical_std": float(np.std(empiricals)),
            "lower_bound_mean": float(np.mean(lb_vals)),
            "lower_bound_std": float(np.std(lb_vals)),
            "bound_valid": all(lb_vals[i] <= empiricals[i] + 0.01 for i in range(n_seeds)),
        })
    return results


def run_properness_experiment(n_trials=500, n_seeds=5):
    """Experiment 4: Verify properness of energy-augmented scoring rules."""
    def energy_fn(y, history):
        if len(history) == 0:
            return 0.0
        return float(min(abs(y - h) for h in history))

    results = {}
    for rule_name, base_rule in [("Logarithmic", LogarithmicRule()),
                                   ("Brier", BrierRule())]:
        violations_base = 0
        violations_augmented = 0
        gaps_base = []
        gaps_augmented = []

        augmented = EnergyAugmentedRule(base_rule, energy_fn, lambda_=0.1)

        for seed in range(n_seeds):
            rng = np.random.RandomState(seed)
            for _ in range(n_trials):
                n_outcomes = rng.randint(3, 8)
                q = rng.dirichlet(np.ones(n_outcomes))
                p = rng.dirichlet(np.ones(n_outcomes))
                history = rng.randint(0, n_outcomes, size=3).tolist()
                augmented.set_history(np.array(history))

                gap_b = base_rule.properness_gap(p, q)
                gap_a = augmented.properness_gap(p, q)

                if gap_b < -1e-6:
                    violations_base += 1
                if gap_a < -1e-6:
                    violations_augmented += 1
                gaps_base.append(gap_b)
                gaps_augmented.append(gap_a)

        results[rule_name] = {
            "base_violations": violations_base,
            "augmented_violations": violations_augmented,
            "total_trials": n_trials * n_seeds,
            "base_mean_gap": float(np.mean(gaps_base)),
            "augmented_mean_gap": float(np.mean(gaps_augmented)),
        }
    return results


def run_pareto_experiment(N=100, k=5, d=8, n_seeds=5):
    """Experiment 5: Quality-diversity Pareto frontier."""
    lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = {lam: {"quality": [], "cosine_div": [], "dispersion": []} for lam in lambdas}

    for seed in range(n_seeds):
        agent = ClusteredAgent(n_clusters=5, cluster_std=0.3, dim=d, seed=seed)
        embeddings = []
        qualities = []
        for _ in range(N):
            emb, q = agent.generate()
            embeddings.append(emb)
            qualities.append(q)
        embeddings = np.array(embeddings)
        qualities = np.array(qualities)

        for lam in lambdas:
            selected = select_diverse(embeddings, qualities, k,
                                       method="divflow", quality_weight=1.0 - lam)
            sel_emb = embeddings[selected]
            results[lam]["quality"].append(float(np.mean(qualities[selected])))
            results[lam]["cosine_div"].append(cosine_diversity(sel_emb))
            results[lam]["dispersion"].append(dispersion_metric(sel_emb))

    summary = {}
    for lam in lambdas:
        summary[str(lam)] = {
            k2: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for k2, v in results[lam].items()
        }
    return summary


def run_diversity_objectives(N=100, k=5, d=8, n_seeds=5):
    """Experiment 6: Compare diversity objectives."""
    results = {"divflow": {"cosine_div": [], "dispersion": [], "vendi": []},
               "dpp": {"cosine_div": [], "dispersion": [], "vendi": []},
               "mmr": {"cosine_div": [], "dispersion": [], "vendi": []},
               "fps": {"cosine_div": [], "dispersion": [], "vendi": []}}

    for seed in range(n_seeds):
        agent = ClusteredAgent(n_clusters=5, cluster_std=0.3, dim=d, seed=seed)
        embeddings = []
        qualities = []
        for _ in range(N):
            emb, q = agent.generate()
            embeddings.append(emb)
            qualities.append(q)
        embeddings = np.array(embeddings)
        qualities = np.array(qualities)

        for method in results.keys():
            selected = select_diverse(embeddings, qualities, k, method=method)
            sel_emb = embeddings[selected]
            results[method]["cosine_div"].append(cosine_diversity(sel_emb))
            results[method]["dispersion"].append(dispersion_metric(sel_emb))
            results[method]["vendi"].append(vendi_score(sel_emb))

    summary = {}
    for m in results:
        summary[m] = {
            k2: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for k2, v in results[m].items()
        }
    return summary


def run_convergence_experiment(N=200, k=20, d=8, n_seeds=3):
    """Experiment 7: Track Sinkhorn divergence over selection rounds."""
    traces = []
    for seed in range(n_seeds):
        agent = ClusteredAgent(n_clusters=5, cluster_std=0.3, dim=d, seed=seed)
        embeddings = []
        qualities = []
        for _ in range(N):
            emb, q = agent.generate()
            embeddings.append(emb)
            qualities.append(q)
        embeddings = np.array(embeddings)
        qualities = np.array(qualities)
        reference = embeddings

        # Build selection trace
        selected = [int(np.argmax(qualities))]
        divs = []
        for step in range(k - 1):
            history = embeddings[selected]
            div = sinkhorn_divergence(history, reference, reg=0.1)
            divs.append(float(div))
            # Select next
            div_scores = sinkhorn_candidate_scores(
                embeddings, history, reference, reg=0.1
            )
            for s in selected:
                div_scores[s] = -np.inf
            selected.append(int(np.argmax(div_scores)))

        # Final divergence
        divs.append(float(sinkhorn_divergence(embeddings[selected], reference, reg=0.1)))
        traces.append(divs)

    return {"traces": traces, "n_steps": k}


def run_timing_experiment(d=8, k=10, n_seeds=3):
    """Experiment 8: Wall-clock timing comparison."""
    Ns = [50, 100, 200, 500]
    methods = ["divflow", "dpp", "mmr", "fps", "kmedoids"]
    results = {m: {} for m in methods}

    for N in Ns:
        for method in methods:
            times = []
            for seed in range(n_seeds):
                agent = ClusteredAgent(n_clusters=5, cluster_std=0.3, dim=d, seed=seed)
                embeddings = []
                qualities = []
                for _ in range(N):
                    emb, q = agent.generate()
                    embeddings.append(emb)
                    qualities.append(q)
                embeddings = np.array(embeddings)
                qualities = np.array(qualities)

                t0 = time.time()
                select_diverse(embeddings, qualities, k, method=method)
                times.append(time.time() - t0)
            results[method][N] = {"mean": float(np.mean(times)), "std": float(np.std(times))}

    return results


if __name__ == "__main__":
    all_results = {}

    print("Experiment 1: Selection comparison (N=200, k=10, d=8, 5 seeds)...")
    all_results["selection_comparison"] = run_selection_comparison()

    print("Experiment 2: Scaling with N...")
    all_results["scaling"] = run_scaling_experiment()

    print("Experiment 3: Coverage certificates...")
    all_results["coverage"] = run_coverage_experiment()

    print("Experiment 4: Properness verification...")
    all_results["properness"] = run_properness_experiment()

    print("Experiment 5: Pareto frontier...")
    all_results["pareto"] = run_pareto_experiment()

    print("Experiment 6: Diversity objectives...")
    all_results["diversity_objectives"] = run_diversity_objectives()

    print("Experiment 7: Convergence trace...")
    all_results["convergence"] = run_convergence_experiment()

    print("Experiment 8: Timing...")
    all_results["timing"] = run_timing_experiment()

    outpath = os.path.join(os.path.dirname(__file__), "paper_results.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results written to {outpath}")

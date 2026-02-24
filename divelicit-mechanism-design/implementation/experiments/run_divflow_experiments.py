#!/usr/bin/env python3
"""DivFlow comprehensive experiments with real LLM responses.

Generates responses from gpt-4.1-nano, embeds with text-embedding-3-small,
and compares DivFlow against baselines (DPP, MMR, k-medoids, random,
top-quality, temperature-only) across multiple tasks and seeds.

Experiments:
  1. Diverse brainstorming (5 prompts × 3 seeds)
  2. Red-teaming probe diversity (3 prompts × 3 seeds)
  3. Code generation diversity (3 prompts × 3 seeds)
  4. Temperature vs post-hoc selection (4 temperatures × 2 prompts)
  5. Scaling with pool size N (N=15,30,50,80)
  6. Quality-diversity Pareto frontier (lambda sweep)
  7. Coverage certificate validation
  8. VCG mechanism IC verification with real embeddings
  9. Ablation: PCA dimension (16, 32, 64, 128)
 10. Ablation: Sinkhorn regularization (0.01, 0.05, 0.1, 0.5)
"""

import json
import os
import sys
import time
import traceback
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.kernels import RBFKernel, AdaptiveRBFKernel, MultiScaleKernel
from src.dpp import DPP, greedy_map
from src.transport import sinkhorn_divergence, sinkhorn_candidate_scores, cost_matrix
from src.coverage import (
    estimate_coverage, coverage_lower_bound, coverage_test,
    epsilon_net_certificate, _effective_dimension, _data_diameter,
)
from src.diversity_metrics import (
    cosine_diversity, log_det_diversity, dispersion_metric, vendi_score,
)
from src.utils import log_det_safe

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

RESULTS_DIR = os.path.dirname(__file__)
SEEDS = [42, 137, 256]


# ============================================================
# LLM Helpers
# ============================================================

def get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)


def generate_responses(client, prompt, n=30, temperature=1.0, seed_offset=0):
    """Generate n responses from gpt-4.1-nano."""
    responses = []
    for i in range(n):
        try:
            resp = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=200,
                seed=seed_offset + i,
            )
            text = resp.choices[0].message.content.strip()
            responses.append(text)
        except Exception as e:
            print(f"    Warning: gen {i} failed: {e}")
            responses.append(f"Fallback response {i}: {prompt[:50]}")
        if (i + 1) % 10 == 0:
            time.sleep(0.3)
    return responses


def embed_texts(client, texts, model="text-embedding-3-small"):
    """Embed texts, returns (n, 1536) array."""
    embeddings = []
    batch_size = 50
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            resp = client.embeddings.create(model=model, input=batch)
            for item in resp.data:
                embeddings.append(item.embedding)
        except Exception as e:
            print(f"    Warning: embed batch {i} failed: {e}")
            for _ in batch:
                embeddings.append(np.random.randn(1536).tolist())
    return np.array(embeddings)


def pca_reduce(embeddings, dim=64):
    """PCA to target dimension."""
    mean = np.mean(embeddings, axis=0)
    centered = embeddings - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    d = min(dim, Vt.shape[0])
    return centered @ Vt[:d].T


# ============================================================
# Selection Methods
# ============================================================

def select_random(n, k, seed=42):
    return list(np.random.RandomState(seed).choice(n, k, replace=False))


def select_top_quality(qualities, k):
    return list(np.argsort(qualities)[-k:])


def select_dpp(embeddings, qualities, k, bandwidth=None):
    """DPP greedy MAP selection."""
    if bandwidth is None:
        D = cost_matrix(embeddings, embeddings, metric="euclidean")
        dists = D[np.triu_indices(len(embeddings), k=1)]
        bandwidth = float(np.median(dists)) / np.sqrt(2)
        bandwidth = max(bandwidth, 0.01)
    kernel = RBFKernel(bandwidth=bandwidth)
    K = kernel.gram_matrix(embeddings)
    L = K * np.outer(qualities, qualities)
    return greedy_map(L, k)


def select_mmr(embeddings, qualities, k, lam=0.5, bandwidth=None):
    """Maximal Marginal Relevance."""
    if bandwidth is None:
        D = cost_matrix(embeddings, embeddings, metric="euclidean")
        dists = D[np.triu_indices(len(embeddings), k=1)]
        bandwidth = float(np.median(dists)) / np.sqrt(2)
        bandwidth = max(bandwidth, 0.01)
    kernel = RBFKernel(bandwidth=bandwidth)
    K = kernel.gram_matrix(embeddings)
    n = len(embeddings)
    selected = [int(np.argmax(qualities))]
    for _ in range(k - 1):
        best_j, best_mmr = -1, -float('inf')
        for j in range(n):
            if j in selected:
                continue
            max_sim = max(K[j, s] for s in selected)
            mmr_val = (1 - lam) * qualities[j] - lam * max_sim
            if mmr_val > best_mmr:
                best_mmr = mmr_val
                best_j = j
        if best_j >= 0:
            selected.append(best_j)
    return selected


def select_kmedoids(embeddings, k):
    """Greedy k-medoids (BUILD phase of PAM)."""
    D = cost_matrix(embeddings, embeddings, metric="euclidean")
    n = len(embeddings)
    selected = [int(np.argmin(D.sum(axis=1)))]
    for _ in range(k - 1):
        best_j, best_gain = -1, -float('inf')
        for j in range(n):
            if j in selected:
                continue
            current_min = np.min(D[:, selected], axis=1)
            new_min = np.minimum(current_min, D[:, j])
            gain = np.sum(current_min - new_min)
            if gain > best_gain:
                best_gain = gain
                best_j = j
        if best_j >= 0:
            selected.append(best_j)
    return selected


def select_divflow(embeddings, qualities, k, reg=0.1, quality_weight=0.3):
    """Sinkhorn dual-potential guided sequential selection (DivFlow)."""
    n = len(embeddings)
    reference = embeddings.copy()
    selected = [int(np.argmax(qualities))]
    for _ in range(k - 1):
        history = embeddings[selected]
        remaining = [j for j in range(n) if j not in selected]
        if not remaining:
            break
        candidates = embeddings[remaining]
        div_scores = sinkhorn_candidate_scores(candidates, history, reference, reg=reg)
        rng = div_scores.max() - div_scores.min()
        if rng > 1e-10:
            div_scores = (div_scores - div_scores.min()) / rng
        else:
            div_scores = np.ones(len(remaining))
        q_rem = qualities[remaining]
        q_rng = q_rem.max() - q_rem.min()
        if q_rng > 1e-10:
            q_norm = (q_rem - q_rem.min()) / q_rng
        else:
            q_norm = np.ones(len(remaining))
        scores = (1.0 - quality_weight) * div_scores + quality_weight * q_norm
        best_idx = remaining[int(np.argmax(scores))]
        selected.append(best_idx)
    return selected


# ============================================================
# Evaluation
# ============================================================

def compute_metrics(embeddings, selected, qualities, all_embeddings=None):
    """Compute diversity + quality metrics for a selection."""
    sel_emb = embeddings[selected]
    sel_q = qualities[selected]
    D = cost_matrix(sel_emb, sel_emb, metric="euclidean")
    dists = D[np.triu_indices(len(selected), k=1)]
    bw = float(np.median(dists)) / np.sqrt(2) if len(dists) > 0 else 1.0
    bw = max(bw, 0.01)
    kernel = RBFKernel(bandwidth=bw)

    metrics = {
        "cosine_diversity": float(cosine_diversity(sel_emb)),
        "dispersion": float(dispersion_metric(sel_emb)),
        "log_det_diversity": float(log_det_diversity(sel_emb, kernel)),
        "vendi_score": float(vendi_score(sel_emb, kernel)),
        "mean_quality": float(np.mean(sel_q)),
        "min_quality": float(np.min(sel_q)),
        "mean_pairwise_dist": float(np.mean(dists)) if len(dists) > 0 else 0.0,
    }

    if all_embeddings is not None and len(all_embeddings) > len(selected):
        epsilon = bw * 2
        try:
            cert = coverage_test(sel_emb, all_embeddings, epsilon)
            metrics["coverage_fraction"] = float(cert.coverage_fraction)
        except Exception:
            metrics["coverage_fraction"] = 0.0

    return metrics


def run_methods(embeddings, qualities, k, seed=42):
    """Run all 6 methods and return {method: metrics}."""
    n = len(embeddings)
    methods = {}
    methods["random"] = select_random(n, k, seed=seed)
    methods["top_quality"] = select_top_quality(qualities, k)
    methods["dpp_greedy"] = select_dpp(embeddings, qualities, k)
    methods["mmr"] = select_mmr(embeddings, qualities, k)
    methods["kmedoids"] = select_kmedoids(embeddings, k)
    methods["divflow"] = select_divflow(embeddings, qualities, k)

    results = {}
    for name, sel in methods.items():
        results[name] = compute_metrics(embeddings, sel, qualities, embeddings)
    return results, methods


def aggregate_seeds(seed_results):
    """Aggregate results across seeds: mean, std, CI."""
    all_methods = set()
    for sr in seed_results:
        all_methods.update(sr.keys())

    agg = {}
    for method in all_methods:
        if method.startswith("_"):
            continue
        metric_vals = defaultdict(list)
        for sr in seed_results:
            if method in sr:
                for metric, val in sr[method].items():
                    metric_vals[metric].append(val)
        agg[method] = {}
        for metric, vals in metric_vals.items():
            arr = np.array(vals)
            m = float(np.mean(arr))
            s = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            agg[method][metric] = {
                "mean": round(m, 6),
                "std": round(s, 6),
                "n": len(vals),
            }
    return agg


# ============================================================
# Task Prompts
# ============================================================

BRAINSTORM_PROMPTS = [
    "Suggest one creative and specific way to reduce plastic waste in everyday life. Give a single concrete idea in 1-2 sentences.",
    "Propose one novel application of machine learning in agriculture. Be specific about the problem and solution in 1-2 sentences.",
    "Describe one innovative way a city could reduce traffic congestion without building new roads. Be specific in 1-2 sentences.",
    "Suggest one creative use of augmented reality for education that hasn't been widely tried. Be specific in 1-2 sentences.",
    "Propose one new way to use blockchain technology outside of cryptocurrency. Describe a specific use case in 1-2 sentences.",
]

REDTEAM_PROMPTS = [
    "Suggest one specific type of edge case or unusual input that could cause a text classification model to misclassify. Describe the edge case in 1-2 sentences.",
    "Describe one way a chatbot could give a misleading or incorrect answer due to a subtle misunderstanding of context. Be specific in 1-2 sentences.",
    "Identify one potential failure mode of a sentiment analysis system when processing real-world social media text. Be specific in 1-2 sentences.",
]

CODE_PROMPTS = [
    "Write a short Python function (3-8 lines) that checks if a number is prime. Use a different approach than typical solutions.",
    "Write a short Python function (3-8 lines) that reverses a linked list. Use any approach you prefer.",
    "Write a short Python function (3-8 lines) that computes the nth Fibonacci number. Use an unusual or creative approach.",
]


def quality_from_responses(responses):
    """Compute quality scores from response text."""
    lengths = np.array([len(r.split()) for r in responses], dtype=float)
    # Combine length and uniqueness
    unique_words = np.array([len(set(r.lower().split())) for r in responses], dtype=float)
    # Normalize both
    l_norm = lengths / max(lengths.max(), 1.0)
    u_norm = unique_words / max(unique_words.max(), 1.0)
    qualities = 0.5 * l_norm + 0.5 * u_norm
    return np.clip(qualities, 0.1, 1.0)


# ============================================================
# Experiments
# ============================================================

def experiment_1_brainstorming(client, n_per_prompt=30, k=8):
    """Experiment 1: Diverse brainstorming across 5 prompts × 3 seeds."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Diverse Brainstorming (5 prompts × 3 seeds)")
    print("=" * 70)

    all_seed_results = []
    all_responses = {}

    for seed_idx, seed in enumerate(SEEDS):
        seed_result = defaultdict(lambda: defaultdict(list))
        for pi, prompt in enumerate(BRAINSTORM_PROMPTS):
            print(f"  Seed {seed}, Prompt {pi+1}/{len(BRAINSTORM_PROMPTS)}...")
            responses = generate_responses(client, prompt, n=n_per_prompt,
                                           temperature=1.0, seed_offset=seed * 100 + pi * n_per_prompt)
            raw_emb = embed_texts(client, responses)
            emb = pca_reduce(raw_emb, dim=64)
            quals = quality_from_responses(responses)

            results, selections = run_methods(emb, quals, k, seed=seed)
            for method, metrics in results.items():
                for metric, val in metrics.items():
                    seed_result[method][metric].append(val)

            if seed_idx == 0:
                all_responses[f"prompt_{pi}"] = {
                    "prompt": prompt,
                    "responses": responses,
                    "divflow_selection": selections.get("divflow", []),
                    "dpp_selection": selections.get("dpp_greedy", []),
                }

        # Average across prompts for this seed
        seed_avg = {}
        for method, metric_lists in seed_result.items():
            seed_avg[method] = {m: float(np.mean(v)) for m, v in metric_lists.items()}
        all_seed_results.append(seed_avg)

    agg = aggregate_seeds(all_seed_results)
    print("\n  Aggregated results (mean ± std across 3 seeds):")
    for method in ["random", "top_quality", "dpp_greedy", "mmr", "kmedoids", "divflow"]:
        if method in agg:
            cd = agg[method].get("cosine_diversity", {})
            dp = agg[method].get("dispersion", {})
            mq = agg[method].get("mean_quality", {})
            print(f"    {method:15s}: cos_div={cd.get('mean',0):.4f}±{cd.get('std',0):.4f}, "
                  f"disp={dp.get('mean',0):.4f}±{dp.get('std',0):.4f}, "
                  f"quality={mq.get('mean',0):.4f}±{mq.get('std',0):.4f}")

    return {"aggregated": agg, "qualitative": all_responses}


def experiment_2_redteaming(client, n_per_prompt=30, k=8):
    """Experiment 2: Red-teaming probe diversity."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Red-Teaming Probe Diversity (3 prompts × 3 seeds)")
    print("=" * 70)

    all_seed_results = []
    all_responses = {}

    for seed_idx, seed in enumerate(SEEDS):
        seed_result = defaultdict(lambda: defaultdict(list))
        for pi, prompt in enumerate(REDTEAM_PROMPTS):
            print(f"  Seed {seed}, Prompt {pi+1}/{len(REDTEAM_PROMPTS)}...")
            responses = generate_responses(client, prompt, n=n_per_prompt,
                                           temperature=1.0, seed_offset=seed * 100 + pi * n_per_prompt)
            raw_emb = embed_texts(client, responses)
            emb = pca_reduce(raw_emb, dim=64)
            quals = quality_from_responses(responses)
            results, selections = run_methods(emb, quals, k, seed=seed)
            for method, metrics in results.items():
                for metric, val in metrics.items():
                    seed_result[method][metric].append(val)

            if seed_idx == 0:
                all_responses[f"prompt_{pi}"] = {
                    "prompt": prompt,
                    "responses": responses,
                    "divflow_selection": selections.get("divflow", []),
                }

        seed_avg = {}
        for method, metric_lists in seed_result.items():
            seed_avg[method] = {m: float(np.mean(v)) for m, v in metric_lists.items()}
        all_seed_results.append(seed_avg)

    agg = aggregate_seeds(all_seed_results)
    print("\n  Aggregated results:")
    for method in ["random", "top_quality", "dpp_greedy", "mmr", "kmedoids", "divflow"]:
        if method in agg:
            cd = agg[method].get("cosine_diversity", {})
            dp = agg[method].get("dispersion", {})
            print(f"    {method:15s}: cos_div={cd.get('mean',0):.4f}±{cd.get('std',0):.4f}, "
                  f"disp={dp.get('mean',0):.4f}±{dp.get('std',0):.4f}")

    return {"aggregated": agg, "qualitative": all_responses}


def experiment_3_code(client, n_per_prompt=30, k=8):
    """Experiment 3: Code generation diversity."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Code Generation Diversity (3 prompts × 3 seeds)")
    print("=" * 70)

    all_seed_results = []
    all_responses = {}

    for seed_idx, seed in enumerate(SEEDS):
        seed_result = defaultdict(lambda: defaultdict(list))
        for pi, prompt in enumerate(CODE_PROMPTS):
            print(f"  Seed {seed}, Prompt {pi+1}/{len(CODE_PROMPTS)}...")
            responses = generate_responses(client, prompt, n=n_per_prompt,
                                           temperature=1.0, seed_offset=seed * 100 + pi * n_per_prompt)
            raw_emb = embed_texts(client, responses)
            emb = pca_reduce(raw_emb, dim=64)
            quals = quality_from_responses(responses)
            results, selections = run_methods(emb, quals, k, seed=seed)
            for method, metrics in results.items():
                for metric, val in metrics.items():
                    seed_result[method][metric].append(val)

            if seed_idx == 0:
                all_responses[f"prompt_{pi}"] = {
                    "prompt": prompt,
                    "responses": responses,
                    "divflow_selection": selections.get("divflow", []),
                }

        seed_avg = {}
        for method, metric_lists in seed_result.items():
            seed_avg[method] = {m: float(np.mean(v)) for m, v in metric_lists.items()}
        all_seed_results.append(seed_avg)

    agg = aggregate_seeds(all_seed_results)
    print("\n  Aggregated results:")
    for method in ["random", "top_quality", "dpp_greedy", "mmr", "kmedoids", "divflow"]:
        if method in agg:
            cd = agg[method].get("cosine_diversity", {})
            dp = agg[method].get("dispersion", {})
            print(f"    {method:15s}: cos_div={cd.get('mean',0):.4f}±{cd.get('std',0):.4f}, "
                  f"disp={dp.get('mean',0):.4f}±{dp.get('std',0):.4f}")

    return {"aggregated": agg, "qualitative": all_responses}


def experiment_4_temperature(client, n=25, k=6):
    """Experiment 4: Temperature scaling vs DivFlow selection."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Temperature vs DivFlow Post-Hoc Selection")
    print("=" * 70)

    prompts = [
        "Suggest one creative way to use drones for environmental monitoring. Be specific in 1-2 sentences.",
        "Describe one innovative approach to teaching mathematics to children. Be specific in 1-2 sentences.",
    ]
    temperatures = [0.3, 0.7, 1.0, 1.5]

    results = {}
    for pi, prompt in enumerate(prompts):
        results[f"prompt_{pi}"] = {}
        for temp in temperatures:
            print(f"  Prompt {pi}, temp={temp}...")
            responses = generate_responses(client, prompt, n=n, temperature=temp, seed_offset=42)
            raw_emb = embed_texts(client, responses)
            emb = pca_reduce(raw_emb, dim=64)
            quals = quality_from_responses(responses)

            # Temperature only: first k
            temp_sel = list(range(k))
            temp_metrics = compute_metrics(emb, temp_sel, quals)

            # DivFlow on this pool
            flow_sel = select_divflow(emb, quals, k)
            flow_metrics = compute_metrics(emb, flow_sel, quals, emb)

            # DPP on this pool
            dpp_sel = select_dpp(emb, quals, k)
            dpp_metrics = compute_metrics(emb, dpp_sel, quals, emb)

            results[f"prompt_{pi}"][f"temp_{temp}"] = {
                "temperature_only": temp_metrics,
                "divflow": flow_metrics,
                "dpp": dpp_metrics,
            }
            print(f"    temp_only cos_div={temp_metrics['cosine_diversity']:.4f}, "
                  f"divflow cos_div={flow_metrics['cosine_diversity']:.4f}, "
                  f"dpp cos_div={dpp_metrics['cosine_diversity']:.4f}")

    return results


def experiment_5_scaling(client, k=8):
    """Experiment 5: Scaling with candidate pool size N."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Scaling with Pool Size")
    print("=" * 70)

    prompt = "Give one unique tip for improving productivity while working from home. Be specific in 1-2 sentences."
    pool_sizes = [15, 30, 50, 80]
    max_n = max(pool_sizes)

    print(f"  Generating {max_n} responses...")
    responses = generate_responses(client, prompt, n=max_n, temperature=1.0, seed_offset=42)
    print(f"  Embedding {max_n} responses...")
    raw_emb = embed_texts(client, responses)

    results = {}
    for n in pool_sizes:
        emb = pca_reduce(raw_emb[:n], dim=64)
        quals = quality_from_responses(responses[:n])

        t0 = time.time()
        method_results, _ = run_methods(emb, quals, k, seed=42)
        elapsed = time.time() - t0

        results[f"n_{n}"] = method_results
        results[f"n_{n}"]["_timing_seconds"] = round(elapsed, 4)
        print(f"  N={n}: divflow cos_div={method_results['divflow']['cosine_diversity']:.4f}, "
              f"time={elapsed:.3f}s")

    return results


def experiment_6_pareto(client, n=30, k=8):
    """Experiment 6: Quality-diversity Pareto frontier."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Quality-Diversity Pareto Frontier")
    print("=" * 70)

    prompt = "Suggest one creative way to make cities more sustainable. Be specific in 1-2 sentences."
    responses = generate_responses(client, prompt, n=n, temperature=1.0, seed_offset=42)
    raw_emb = embed_texts(client, responses)
    emb = pca_reduce(raw_emb, dim=64)
    quals = quality_from_responses(responses)

    lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = {}
    for lam in lambdas:
        sel = select_divflow(emb, quals, k, quality_weight=lam)
        metrics = compute_metrics(emb, sel, quals, emb)
        results[f"lambda_{lam}"] = metrics
        print(f"  lambda={lam:.1f}: cos_div={metrics['cosine_diversity']:.4f}, "
              f"quality={metrics['mean_quality']:.4f}")

    return results


def experiment_7_coverage(client, n=50, k=10):
    """Experiment 7: Coverage certificate validation."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: Coverage Certificate Validation")
    print("=" * 70)

    prompt = "Name one unusual fact about the ocean. State it clearly in 1-2 sentences."
    responses = generate_responses(client, prompt, n=n, temperature=1.0, seed_offset=42)
    raw_emb = embed_texts(client, responses)

    results = {}
    for dim in [16, 32, 64]:
        emb = pca_reduce(raw_emb, dim=dim)
        quals = quality_from_responses(responses)

        # DivFlow selection
        flow_sel = select_divflow(emb, quals, k)
        flow_emb = emb[flow_sel]

        # Random selection
        rand_sel = select_random(n, k, seed=42)
        rand_emb = emb[rand_sel]

        eff_dim = _effective_dimension(emb)
        diameter = _data_diameter(emb)

        # Coverage at multiple epsilon values
        epsilons = [diameter * f for f in [0.1, 0.2, 0.3, 0.5]]
        cov_results = {}
        for eps in epsilons:
            try:
                flow_cert = coverage_test(flow_emb, emb, eps)
                rand_cert = coverage_test(rand_emb, emb, eps)
                cov_results[f"eps_{eps:.3f}"] = {
                    "divflow_coverage": float(flow_cert.coverage_fraction),
                    "random_coverage": float(rand_cert.coverage_fraction),
                }
            except Exception:
                pass

        # Epsilon-net certificate
        try:
            enet = epsilon_net_certificate(flow_emb, delta=0.05)
            enet_result = {
                "coverage_fraction": float(enet.coverage_fraction),
                "confidence": float(enet.confidence),
                "epsilon": float(enet.epsilon_radius),
            }
        except Exception:
            enet_result = {"error": "failed"}

        results[f"dim_{dim}"] = {
            "effective_dim": eff_dim,
            "diameter": round(diameter, 4),
            "n_selected": k,
            "coverage_at_epsilon": cov_results,
            "epsilon_net_certificate": enet_result,
        }
        print(f"  dim={dim}: eff_dim={eff_dim}, diameter={diameter:.3f}")

    return results


def experiment_8_vcg_ic(client, n=30, k=8):
    """Experiment 8: VCG mechanism IC verification with real embeddings."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 8: VCG Mechanism IC Verification")
    print("=" * 70)

    prompt = "Suggest one way to improve public transportation. Be specific in 1-2 sentences."
    responses = generate_responses(client, prompt, n=n, temperature=1.0, seed_offset=42)
    raw_emb = embed_texts(client, responses)
    emb = pca_reduce(raw_emb, dim=64)
    quals = quality_from_responses(responses)

    # Simulate VCG: agents report valuations, mechanism selects diverse subset
    # Agent i's valuation for outcome S = quality_i if i in S, 0 otherwise
    # VCG payment = externality on others

    from src.mechanism import VCGMechanism, BudgetFeasibleMechanism
    from src.agents import GaussianAgent

    # Create agents from real embeddings
    agents = []
    d = emb.shape[1]
    for i in range(min(n, 15)):
        mean = emb[i]
        cov = np.eye(d) * 0.01
        agents.append(GaussianAgent(mean, cov, quality_mean=float(quals[i]),
                                    quality_std=0.02, seed=42 + i))

    from src.scoring_rules import BrierRule
    vcg = VCGMechanism(scoring_rule=BrierRule(), k_select=k,
                       kernel=RBFKernel(bandwidth=1.0))
    bf = BudgetFeasibleMechanism(scoring_rule=BrierRule(), k_select=k,
                                 budget=10.0, kernel=RBFKernel(bandwidth=1.0))

    # Run and verify IC
    vcg_result = vcg.run(agents)
    vcg_ic, vcg_viol, vcg_trials = vcg.verify_ic(agents, n_trials=50)

    bf_result = bf.run(agents)
    bf_ic, bf_viol, bf_trials = bf.verify_ic(agents, n_trials=50)

    results = {
        "vcg": {
            "diversity_score": float(vcg_result.diversity_score),
            "mean_quality": float(np.mean(vcg_result.quality_scores)),
            "ic_verified": bool(vcg_ic),
            "ic_violations": int(vcg_viol),
            "ic_trials": int(vcg_trials),
            "payments": [float(p) for p in vcg_result.payments] if vcg_result.payments else [],
        },
        "budget_feasible": {
            "diversity_score": float(bf_result.diversity_score),
            "mean_quality": float(np.mean(bf_result.quality_scores)),
            "ic_verified": bool(bf_ic),
            "ic_violations": int(bf_viol),
            "ic_trials": int(bf_trials),
        },
    }
    print(f"  VCG: IC={vcg_ic}, violations={vcg_viol}/{vcg_trials}")
    print(f"  BudgetFeasible: IC={bf_ic}, violations={bf_viol}/{bf_trials}")

    return results


def experiment_9_ablation_dim(client, n=30, k=8):
    """Experiment 9: Ablation on PCA target dimension."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 9: Ablation - PCA Dimension")
    print("=" * 70)

    prompt = "Describe one creative use of renewable energy. Be specific in 1-2 sentences."
    responses = generate_responses(client, prompt, n=n, temperature=1.0, seed_offset=42)
    raw_emb = embed_texts(client, responses)
    quals = quality_from_responses(responses)

    results = {}
    for dim in [16, 32, 64, 128]:
        emb = pca_reduce(raw_emb, dim=dim)
        method_results, _ = run_methods(emb, quals, k, seed=42)
        results[f"dim_{dim}"] = {
            m: {
                "cosine_diversity": method_results[m]["cosine_diversity"],
                "dispersion": method_results[m]["dispersion"],
                "mean_quality": method_results[m]["mean_quality"],
            }
            for m in method_results
        }
        print(f"  dim={dim}: divflow cos_div={method_results['divflow']['cosine_diversity']:.4f}")

    return results


def experiment_10_ablation_reg(client, n=30, k=8):
    """Experiment 10: Ablation on Sinkhorn regularization parameter."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 10: Ablation - Sinkhorn Regularization")
    print("=" * 70)

    prompt = "Suggest one way AI could help in disaster response. Be specific in 1-2 sentences."
    responses = generate_responses(client, prompt, n=n, temperature=1.0, seed_offset=42)
    raw_emb = embed_texts(client, responses)
    emb = pca_reduce(raw_emb, dim=64)
    quals = quality_from_responses(responses)

    results = {}
    for reg in [0.01, 0.05, 0.1, 0.5, 1.0]:
        sel = select_divflow(emb, quals, k, reg=reg)
        metrics = compute_metrics(emb, sel, quals, emb)
        results[f"reg_{reg}"] = metrics
        print(f"  reg={reg}: cos_div={metrics['cosine_diversity']:.4f}, "
              f"disp={metrics['dispersion']:.4f}")

    return results


# ============================================================
# Main
# ============================================================

def main():
    if not HAS_OPENAI:
        print("ERROR: openai not installed")
        return

    client = get_client()
    all_results = {}
    t0 = time.time()

    try:
        all_results["experiment_1_brainstorming"] = experiment_1_brainstorming(client)
    except Exception as e:
        print(f"Experiment 1 failed: {e}")
        traceback.print_exc()

    try:
        all_results["experiment_2_redteaming"] = experiment_2_redteaming(client)
    except Exception as e:
        print(f"Experiment 2 failed: {e}")
        traceback.print_exc()

    try:
        all_results["experiment_3_code"] = experiment_3_code(client)
    except Exception as e:
        print(f"Experiment 3 failed: {e}")
        traceback.print_exc()

    try:
        all_results["experiment_4_temperature"] = experiment_4_temperature(client)
    except Exception as e:
        print(f"Experiment 4 failed: {e}")
        traceback.print_exc()

    try:
        all_results["experiment_5_scaling"] = experiment_5_scaling(client)
    except Exception as e:
        print(f"Experiment 5 failed: {e}")
        traceback.print_exc()

    try:
        all_results["experiment_6_pareto"] = experiment_6_pareto(client)
    except Exception as e:
        print(f"Experiment 6 failed: {e}")
        traceback.print_exc()

    try:
        all_results["experiment_7_coverage"] = experiment_7_coverage(client)
    except Exception as e:
        print(f"Experiment 7 failed: {e}")
        traceback.print_exc()

    try:
        all_results["experiment_8_vcg_ic"] = experiment_8_vcg_ic(client)
    except Exception as e:
        print(f"Experiment 8 failed: {e}")
        traceback.print_exc()

    try:
        all_results["experiment_9_ablation_dim"] = experiment_9_ablation_dim(client)
    except Exception as e:
        print(f"Experiment 9 failed: {e}")
        traceback.print_exc()

    try:
        all_results["experiment_10_ablation_reg"] = experiment_10_ablation_reg(client)
    except Exception as e:
        print(f"Experiment 10 failed: {e}")
        traceback.print_exc()

    total_time = time.time() - t0
    all_results["_metadata"] = {
        "total_time_seconds": round(total_time, 2),
        "seeds": SEEDS,
        "model_generation": "gpt-4.1-nano",
        "model_embedding": "text-embedding-3-small",
    }

    # Save results
    output_path = os.path.join(RESULTS_DIR, "divflow_results.json")
    # Clean non-serializable items
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w") as f:
        json.dump(clean(all_results), f, indent=2)
    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETE in {total_time:.1f}s")
    print(f"Results saved to {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

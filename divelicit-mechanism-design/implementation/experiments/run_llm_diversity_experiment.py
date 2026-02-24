"""Real LLM diversity experiment using GPT-4.1-nano + text-embedding-3-small.

Generates actual LLM responses to diverse prompts, embeds them, then compares
DivFlow (Sinkhorn-guided VCG), DPP greedy MAP, MMR, k-medoids, random, and 
top-quality selection on real embedding distributions.

Also tests cross-module integration: VCG payments + Sinkhorn selection + 
coverage certificates operating together on real LLM outputs.
"""

import json
import os
import sys
import time
import hashlib
from typing import Dict, List, Tuple, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.kernels import RBFKernel, AdaptiveRBFKernel, MultiScaleKernel
from src.dpp import DPP, greedy_map
from src.transport import sinkhorn_divergence, sinkhorn_candidate_scores, cost_matrix
from src.coverage import estimate_coverage, epsilon_net_certificate, coverage_lower_bound
from src.diversity_metrics import cosine_diversity, log_det_diversity, dispersion_metric, vendi_score
from src.utils import log_det_safe

# ---------------------------------------------------------------------------
# OpenAI helpers
# ---------------------------------------------------------------------------

def get_openai_client():
    from openai import OpenAI
    return OpenAI()

def generate_responses(client, prompt: str, n: int = 20, 
                       model: str = "gpt-4.1-nano") -> List[str]:
    """Generate n diverse responses to a prompt."""
    responses = []
    # Use temperature variation to get diverse responses
    temps = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
    system_prompts = [
        "You are a helpful assistant. Give a unique, creative answer.",
        "You are a technical expert. Give a precise, detailed answer.",
        "You are a creative writer. Give an imaginative, novel answer.",
        "You are a skeptical analyst. Give a critical, balanced answer.",
        "You are a practical engineer. Give a concrete, actionable answer.",
        "You are a philosopher. Give a deep, reflective answer.",
    ]
    
    for i in range(n):
        temp = temps[i % len(temps)]
        sys_prompt = system_prompts[i % len(system_prompts)]
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temp,
                max_tokens=300,
            )
            responses.append(resp.choices[0].message.content)
        except Exception as e:
            print(f"  Warning: generation {i} failed: {e}")
            responses.append(f"Fallback response {i} for: {prompt}")
    return responses


def embed_texts(client, texts: List[str], 
                model: str = "text-embedding-3-small") -> np.ndarray:
    """Embed texts using OpenAI embedding model. Returns (n, d) array."""
    batch_size = 100
    all_embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        for item in resp.data:
            all_embeddings.append(item.embedding)
    return np.array(all_embeddings)


# ---------------------------------------------------------------------------
# Selection methods (operating on embeddings directly)
# ---------------------------------------------------------------------------

def select_divflow(embeddings: np.ndarray, qualities: np.ndarray, k: int,
                   quality_weight: float = 0.5, reg: float = 0.1) -> List[int]:
    """DivFlow: Sinkhorn-guided greedy selection with quality weighting."""
    n = embeddings.shape[0]
    reference = embeddings.copy()  # use all candidates as reference distribution
    selected = []
    
    for _ in range(k):
        if len(selected) == 0:
            history = np.zeros((0, embeddings.shape[1]))
        else:
            history = embeddings[selected]
        
        # Sinkhorn candidate scores (diversity need)
        remaining = [i for i in range(n) if i not in selected]
        if not remaining:
            break
        
        cand_embs = embeddings[remaining]
        div_scores = sinkhorn_candidate_scores(
            cand_embs, history, reference, reg=reg, n_iter=50
        )
        
        # Combined score: diversity + quality
        combined = np.zeros(len(remaining))
        for idx, r in enumerate(remaining):
            combined[idx] = (1.0 - quality_weight) * div_scores[idx] + \
                           quality_weight * qualities[r]
        
        best_idx = remaining[int(np.argmax(combined))]
        selected.append(best_idx)
    
    return selected


def select_vcg_divflow(embeddings: np.ndarray, qualities: np.ndarray, k: int,
                       quality_weight: float = 0.5, 
                       kernel: Optional[RBFKernel] = None) -> Tuple[List[int], List[float], Dict]:
    """VCG mechanism with DivFlow objective: DSIC diverse selection.
    
    Returns (selected_indices, payments, info_dict).
    """
    n = embeddings.shape[0]
    if kernel is None:
        # Adaptive bandwidth from median pairwise distance
        dists = cost_matrix(embeddings, embeddings, metric="euclidean")
        median_dist = float(np.median(dists[dists > 0]))
        kernel = RBFKernel(bandwidth=max(median_dist, 0.1))
    
    def social_welfare(indices):
        if len(indices) == 0:
            return 0.0
        K_S = kernel.gram_matrix(embeddings[indices])
        div = log_det_safe(K_S)
        q_sum = sum(qualities[i] for i in indices)
        return (1.0 - quality_weight) * div + quality_weight * q_sum
    
    # Greedy welfare-maximizing allocation
    selected = []
    for _ in range(min(k, n)):
        best_j, best_gain = -1, -float('inf')
        for j in range(n):
            if j in selected:
                continue
            trial = selected + [j]
            gain = social_welfare(trial) - social_welfare(selected)
            if gain > best_gain:
                best_gain = gain
                best_j = j
        if best_j >= 0:
            selected.append(best_j)
    
    total_welfare = social_welfare(selected)
    
    # VCG payments
    payments = []
    for i in selected:
        others = [j for j in selected if j != i]
        welfare_others_in_opt = social_welfare(others)
        
        # Optimal allocation without i
        candidates_without_i = [j for j in range(n) if j != i]
        sel_without = []
        for _ in range(min(k, len(candidates_without_i))):
            best_j, best_gain = -1, -float('inf')
            for j in candidates_without_i:
                if j in sel_without:
                    continue
                trial = sel_without + [j]
                gain = social_welfare(trial) - social_welfare(sel_without)
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
            if best_j >= 0:
                sel_without.append(best_j)
        welfare_without_i = social_welfare(sel_without)
        
        payment = max(welfare_without_i - welfare_others_in_opt, 0.0)
        payments.append(float(payment))
    
    info = {
        "total_welfare": float(total_welfare),
        "total_payments": float(sum(payments)),
        "mean_payment": float(np.mean(payments)) if payments else 0.0,
    }
    
    return selected, payments, info


def select_dpp(embeddings: np.ndarray, qualities: np.ndarray, k: int,
               kernel: Optional[RBFKernel] = None) -> List[int]:
    """DPP greedy MAP selection with quality-weighted L-kernel."""
    if kernel is None:
        dists = cost_matrix(embeddings, embeddings, metric="euclidean")
        median_dist = float(np.median(dists[dists > 0]))
        kernel = RBFKernel(bandwidth=max(median_dist, 0.1))
    
    K = kernel.gram_matrix(embeddings)
    L = K * np.outer(qualities, qualities)
    return greedy_map(L, k)


def select_mmr(embeddings: np.ndarray, qualities: np.ndarray, k: int,
               lam: float = 0.5) -> List[int]:
    """Maximal Marginal Relevance selection."""
    n = embeddings.shape[0]
    # Cosine similarity matrix
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.maximum(norms, 1e-12)
    sim = normed @ normed.T
    
    selected = [int(np.argmax(qualities))]
    
    for _ in range(k - 1):
        best_j, best_score = -1, -float('inf')
        for j in range(n):
            if j in selected:
                continue
            max_sim_to_selected = max(sim[j, s] for s in selected)
            score = lam * qualities[j] - (1 - lam) * max_sim_to_selected
            if score > best_score:
                best_score = score
                best_j = j
        if best_j >= 0:
            selected.append(best_j)
    
    return selected


def select_kmedoids(embeddings: np.ndarray, qualities: np.ndarray, k: int,
                    max_iter: int = 100) -> List[int]:
    """k-Medoids selection (PAM algorithm)."""
    n = embeddings.shape[0]
    dists = cost_matrix(embeddings, embeddings, metric="euclidean")
    
    # Initialize: pick k diverse medoids via greedy farthest-first
    rng = np.random.RandomState(42)
    medoids = [int(np.argmax(qualities))]
    for _ in range(k - 1):
        remaining = [j for j in range(n) if j not in medoids]
        if not remaining:
            break
        min_dists = np.array([min(dists[j, m] for m in medoids) for j in remaining])
        medoids.append(remaining[int(np.argmax(min_dists))])
    
    # PAM swap step
    for _ in range(max_iter):
        # Assign clusters
        clusters = {}
        for j in range(n):
            closest = min(medoids, key=lambda m: dists[j, m])
            clusters.setdefault(closest, []).append(j)
        
        # Try swapping medoids
        improved = False
        for idx, m in enumerate(medoids):
            cluster = clusters.get(m, [m])
            best_cost = sum(dists[j, m] for j in cluster)
            best_m = m
            for j in cluster:
                if j == m:
                    continue
                cost = sum(dists[jj, j] for jj in cluster)
                if cost < best_cost:
                    best_cost = cost
                    best_m = j
                    improved = True
            medoids[idx] = best_m
        
        if not improved:
            break
    
    return medoids


def select_random(n: int, k: int, seed: int = 42) -> List[int]:
    """Random selection baseline."""
    rng = np.random.RandomState(seed)
    return list(rng.choice(n, size=min(k, n), replace=False))


def select_top_quality(qualities: np.ndarray, k: int) -> List[int]:
    """Select top-k by quality score."""
    return list(np.argsort(qualities)[-k:][::-1])


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def evaluate_selection(embeddings: np.ndarray, qualities: np.ndarray,
                       selected: List[int], all_embeddings: np.ndarray) -> Dict:
    """Compute comprehensive metrics for a selection."""
    sel_embs = embeddings[selected]
    sel_quals = qualities[selected]
    
    # Diversity metrics
    cos_div = float(cosine_diversity(sel_embs))
    disp = float(dispersion_metric(sel_embs))
    
    # Log-det diversity
    dists = cost_matrix(sel_embs, sel_embs, metric="euclidean")
    median_dist = float(np.median(dists[dists > 0])) if np.any(dists > 0) else 1.0
    kernel = RBFKernel(bandwidth=max(median_dist, 0.1))
    K = kernel.gram_matrix(sel_embs)
    logdet = float(log_det_safe(K))
    
    # Vendi score
    try:
        vs = float(vendi_score(sel_embs))
    except:
        vs = float(len(selected))
    
    # Quality metrics
    mean_q = float(np.mean(sel_quals))
    min_q = float(np.min(sel_quals))
    max_q = float(np.max(sel_quals))
    
    # Coverage: fraction of all_embeddings within epsilon of selected
    cert = estimate_coverage(sel_embs, epsilon=0.5)
    
    # Sinkhorn divergence to uniform reference
    if len(sel_embs) >= 2 and len(all_embeddings) >= 2:
        sink_div = float(sinkhorn_divergence(sel_embs, all_embeddings, reg=0.1))
    else:
        sink_div = 0.0
    
    return {
        "cosine_diversity": cos_div,
        "dispersion": disp,
        "log_det_diversity": logdet,
        "vendi_score": vs,
        "sinkhorn_divergence_to_full": sink_div,
        "mean_quality": mean_q,
        "min_quality": min_q,
        "max_quality": max_q,
        "coverage_fraction": cert.coverage_fraction,
        "coverage_lower_bound": cert.confidence,
        "n_selected": len(selected),
    }


# ---------------------------------------------------------------------------
# IC verification for VCG
# ---------------------------------------------------------------------------

def verify_vcg_ic(embeddings: np.ndarray, qualities: np.ndarray, 
                  selected: List[int], payments: List[float],
                  quality_weight: float = 0.5, n_trials: int = 200) -> Dict:
    """Empirically verify IC: no agent benefits from misreporting quality."""
    n = len(qualities)
    dists = cost_matrix(embeddings, embeddings, metric="euclidean")
    median_dist = float(np.median(dists[dists > 0]))
    kernel = RBFKernel(bandwidth=max(median_dist, 0.1))
    
    violations = 0
    max_gain = 0.0
    rng = np.random.RandomState(42)
    
    for trial in range(n_trials):
        agent_idx = rng.randint(n)
        true_q = qualities[agent_idx]
        
        # Truthful utility
        if agent_idx in selected:
            pos = selected.index(agent_idx)
            truthful_utility = true_q - payments[pos]
        else:
            truthful_utility = 0.0
        
        # Deviation
        fake_q = rng.uniform(0.0, 1.0)
        dev_qualities = qualities.copy()
        dev_qualities[agent_idx] = fake_q
        
        dev_selected, dev_payments, _ = select_vcg_divflow(
            embeddings, dev_qualities, len(selected), quality_weight, kernel
        )
        
        if agent_idx in dev_selected:
            pos = dev_selected.index(agent_idx)
            dev_utility = true_q - dev_payments[pos]
        else:
            dev_utility = 0.0
        
        gain = dev_utility - truthful_utility
        if gain > 1e-8:
            violations += 1
            max_gain = max(max_gain, gain)
    
    return {
        "n_trials": n_trials,
        "violations": violations,
        "violation_rate": violations / n_trials,
        "max_gain_from_deviation": float(max_gain),
        "ic_verified": violations == 0,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

PROMPTS = {
    "code_review": "What are the most important things to check in a code review? Give specific, actionable advice.",
    "ml_debugging": "How do you debug a neural network that isn't learning? Walk through a systematic approach.",
    "system_design": "Design a distributed cache system for a social media platform. Cover architecture, consistency, and failure modes.",
    "math_problem": "Explain three different approaches to computing the determinant of a large sparse matrix, with tradeoffs.",
    "ethics_ai": "What are the key ethical considerations when deploying an AI system for hiring decisions?",
}

K_SELECT = 5  # select 5 out of N
N_RESPONSES = 30  # generate 30 responses per prompt


def run_single_prompt_experiment(client, prompt_name: str, prompt: str,
                                 n_responses: int, k_select: int) -> Dict:
    """Run full experiment for one prompt."""
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt_name}")
    print(f"{'='*60}")
    
    # Generate responses
    print(f"  Generating {n_responses} responses...")
    responses = generate_responses(client, prompt, n=n_responses)
    print(f"  Got {len(responses)} responses")
    
    # Embed responses
    print(f"  Embedding responses...")
    embeddings = embed_texts(client, responses)
    print(f"  Embeddings shape: {embeddings.shape}")
    
    # Simulate quality scores based on response properties
    # Use embedding norms and response lengths as proxy quality signals
    qualities = np.zeros(len(responses))
    for i, resp in enumerate(responses):
        length_score = min(len(resp) / 500.0, 1.0)
        # Use embedding norm as a proxy for "substance"
        norm_score = min(np.linalg.norm(embeddings[i]) / 2.0, 1.0)
        qualities[i] = 0.5 * length_score + 0.5 * norm_score
    # Normalize to [0,1]
    if qualities.max() > qualities.min():
        qualities = (qualities - qualities.min()) / (qualities.max() - qualities.min())
    qualities = 0.3 + 0.7 * qualities  # shift to [0.3, 1.0]
    
    results = {}
    
    # 1. DivFlow (Sinkhorn-guided)
    print(f"  Running DivFlow...")
    sel_divflow = select_divflow(embeddings, qualities, k_select)
    results["divflow"] = evaluate_selection(embeddings, qualities, sel_divflow, embeddings)
    
    # 2. VCG-DivFlow (with payments)
    print(f"  Running VCG-DivFlow...")
    sel_vcg, payments_vcg, vcg_info = select_vcg_divflow(embeddings, qualities, k_select)
    results["vcg_divflow"] = evaluate_selection(embeddings, qualities, sel_vcg, embeddings)
    results["vcg_divflow"]["vcg_total_payments"] = vcg_info["total_payments"]
    results["vcg_divflow"]["vcg_mean_payment"] = vcg_info["mean_payment"]
    results["vcg_divflow"]["vcg_total_welfare"] = vcg_info["total_welfare"]
    
    # 3. DPP greedy MAP
    print(f"  Running DPP...")
    sel_dpp = select_dpp(embeddings, qualities, k_select)
    results["dpp"] = evaluate_selection(embeddings, qualities, sel_dpp, embeddings)
    
    # 4. MMR
    print(f"  Running MMR...")
    sel_mmr = select_mmr(embeddings, qualities, k_select)
    results["mmr"] = evaluate_selection(embeddings, qualities, sel_mmr, embeddings)
    
    # 5. k-Medoids
    print(f"  Running k-Medoids...")
    sel_kmed = select_kmedoids(embeddings, qualities, k_select)
    results["kmedoids"] = evaluate_selection(embeddings, qualities, sel_kmed, embeddings)
    
    # 6. Random baseline
    print(f"  Running random baseline...")
    sel_rand = select_random(len(embeddings), k_select)
    results["random"] = evaluate_selection(embeddings, qualities, sel_rand, embeddings)
    
    # 7. Top-quality baseline
    print(f"  Running top-quality baseline...")
    sel_topq = select_top_quality(qualities, k_select)
    results["top_quality"] = evaluate_selection(embeddings, qualities, sel_topq, embeddings)
    
    # VCG IC verification
    print(f"  Verifying VCG IC...")
    ic_result = verify_vcg_ic(embeddings, qualities, sel_vcg, payments_vcg)
    results["vcg_ic_verification"] = ic_result
    
    # Coverage certificates
    print(f"  Computing coverage certificates...")
    for method_name, sel in [("divflow", sel_divflow), ("vcg_divflow", sel_vcg),
                              ("dpp", sel_dpp), ("mmr", sel_mmr)]:
        cert = epsilon_net_certificate(embeddings[sel], embeddings, epsilon=0.5)
        results[method_name]["epsilon_net_coverage"] = cert.coverage_fraction
        results[method_name]["epsilon_net_confidence"] = cert.confidence
    
    # Store response samples for qualitative analysis
    results["sample_responses"] = {}
    for method_name, sel in [("divflow", sel_divflow), ("vcg_divflow", sel_vcg),
                              ("dpp", sel_dpp), ("top_quality", sel_topq)]:
        results["sample_responses"][method_name] = [
            responses[i][:200] for i in sel
        ]
    
    results["metadata"] = {
        "prompt": prompt,
        "n_responses": n_responses,
        "k_select": k_select,
        "embedding_dim": int(embeddings.shape[1]),
    }
    
    return results


def run_scaling_experiment(client, prompt: str, k_select: int = 5) -> Dict:
    """Test how methods scale with increasing N."""
    results = {}
    for n_resp in [10, 20, 30, 50]:
        print(f"\n  Scaling test: N={n_resp}")
        responses = generate_responses(client, prompt, n=n_resp)
        embeddings = embed_texts(client, responses)
        
        qualities = np.zeros(len(responses))
        for i, resp in enumerate(responses):
            qualities[i] = min(len(resp) / 500.0, 1.0) * 0.5 + \
                          min(np.linalg.norm(embeddings[i]) / 2.0, 1.0) * 0.5
        if qualities.max() > qualities.min():
            qualities = (qualities - qualities.min()) / (qualities.max() - qualities.min())
        qualities = 0.3 + 0.7 * qualities
        
        k = min(k_select, n_resp - 1)
        
        methods = {}
        for name, select_fn in [
            ("divflow", lambda: select_divflow(embeddings, qualities, k)),
            ("dpp", lambda: select_dpp(embeddings, qualities, k)),
            ("mmr", lambda: select_mmr(embeddings, qualities, k)),
            ("random", lambda: select_random(len(embeddings), k)),
        ]:
            t0 = time.time()
            sel = select_fn()
            elapsed = time.time() - t0
            metrics = evaluate_selection(embeddings, qualities, sel, embeddings)
            metrics["runtime_seconds"] = elapsed
            methods[name] = metrics
        
        results[f"N={n_resp}"] = methods
    
    return results


def run_quality_diversity_tradeoff(client, prompt: str, n_resp: int = 30, 
                                    k_select: int = 5) -> Dict:
    """Measure Pareto frontier of quality vs diversity at different lambda values."""
    responses = generate_responses(client, prompt, n=n_resp)
    embeddings = embed_texts(client, responses)
    
    qualities = np.zeros(len(responses))
    for i, resp in enumerate(responses):
        qualities[i] = min(len(resp) / 500.0, 1.0) * 0.5 + \
                      min(np.linalg.norm(embeddings[i]) / 2.0, 1.0) * 0.5
    if qualities.max() > qualities.min():
        qualities = (qualities - qualities.min()) / (qualities.max() - qualities.min())
    qualities = 0.3 + 0.7 * qualities
    
    results = {}
    for lam in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]:
        sel = select_divflow(embeddings, qualities, k_select, quality_weight=lam)
        metrics = evaluate_selection(embeddings, qualities, sel, embeddings)
        metrics["quality_weight"] = lam
        results[f"lambda={lam}"] = metrics
    
    return results


def main():
    print("DivFlow LLM Diversity Experiment")
    print("=" * 60)
    
    client = get_openai_client()
    all_results = {}
    
    # Experiment 1: Per-prompt comparison
    print("\n\n=== EXPERIMENT 1: Method Comparison Across Prompts ===")
    for prompt_name, prompt in PROMPTS.items():
        result = run_single_prompt_experiment(
            client, prompt_name, prompt, N_RESPONSES, K_SELECT
        )
        all_results[f"exp1_{prompt_name}"] = result
    
    # Experiment 2: Scaling
    print("\n\n=== EXPERIMENT 2: Scaling with N ===")
    all_results["exp2_scaling"] = run_scaling_experiment(
        client, PROMPTS["code_review"], k_select=K_SELECT
    )
    
    # Experiment 3: Quality-diversity tradeoff
    print("\n\n=== EXPERIMENT 3: Quality-Diversity Pareto Frontier ===")
    all_results["exp3_pareto"] = run_quality_diversity_tradeoff(
        client, PROMPTS["system_design"], n_resp=30, k_select=K_SELECT
    )
    
    # Aggregate results across prompts
    print("\n\n=== AGGREGATING RESULTS ===")
    methods = ["divflow", "vcg_divflow", "dpp", "mmr", "kmedoids", "random", "top_quality"]
    metrics = ["cosine_diversity", "dispersion", "log_det_diversity", "vendi_score",
               "mean_quality", "sinkhorn_divergence_to_full", "coverage_fraction"]
    
    aggregate = {}
    for method in methods:
        aggregate[method] = {}
        for metric in metrics:
            values = []
            for prompt_name in PROMPTS:
                key = f"exp1_{prompt_name}"
                if key in all_results and method in all_results[key]:
                    val = all_results[key][method].get(metric)
                    if val is not None:
                        values.append(val)
            if values:
                arr = np.array(values)
                aggregate[method][metric] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "n": len(values),
                }
    
    all_results["aggregate"] = aggregate
    
    # VCG IC aggregate
    ic_violations_total = 0
    ic_trials_total = 0
    for prompt_name in PROMPTS:
        key = f"exp1_{prompt_name}"
        if key in all_results and "vcg_ic_verification" in all_results[key]:
            ic_info = all_results[key]["vcg_ic_verification"]
            ic_violations_total += ic_info["violations"]
            ic_trials_total += ic_info["n_trials"]
    
    all_results["vcg_ic_aggregate"] = {
        "total_violations": ic_violations_total,
        "total_trials": ic_trials_total,
        "violation_rate": ic_violations_total / max(ic_trials_total, 1),
        "ic_verified": ic_violations_total == 0,
    }
    
    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "llm_diversity_results.json")
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj
    
    all_results = convert_numpy(all_results)
    
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {out_path}")
    
    # Print summary table
    print("\n\n" + "=" * 80)
    print("SUMMARY: Method Comparison (averaged across 5 prompts)")
    print("=" * 80)
    print(f"{'Method':<15} {'CosDiversity':>13} {'Dispersion':>11} {'LogDet':>10} "
          f"{'Vendi':>8} {'MeanQuality':>12} {'Coverage':>10}")
    print("-" * 80)
    for method in methods:
        if method in aggregate:
            m = aggregate[method]
            print(f"{method:<15} "
                  f"{m.get('cosine_diversity', {}).get('mean', 0):.4f}±{m.get('cosine_diversity', {}).get('std', 0):.3f} "
                  f"{m.get('dispersion', {}).get('mean', 0):.3f}±{m.get('dispersion', {}).get('std', 0):.2f} "
                  f"{m.get('log_det_diversity', {}).get('mean', 0):.3f} "
                  f"{m.get('vendi_score', {}).get('mean', 0):.2f} "
                  f"{m.get('mean_quality', {}).get('mean', 0):.4f}±{m.get('mean_quality', {}).get('std', 0):.3f} "
                  f"{m.get('coverage_fraction', {}).get('mean', 0):.4f}")
    
    print(f"\nVCG IC: {all_results['vcg_ic_aggregate']}")


if __name__ == "__main__":
    main()

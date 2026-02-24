"""Improved LLM experiments addressing reviewer critiques.

Key fixes:
- W3: Quality scored by LLM judge (GPT-4.1-nano), not response length
- W4: Multiple seeds (5) with mean/std reporting
- W11: Temperature baseline uses random selection, not first-k
"""

import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.kernels import RBFKernel
from src.dpp import greedy_map
from src.transport import sinkhorn_candidate_scores, cost_matrix
from src.coverage import coverage_test
from src.diversity_metrics import (
    cosine_diversity, log_det_diversity, dispersion_metric, vendi_score,
)

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


def get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)


def generate_diverse_responses(client, prompt, n=20, temperature=1.0, seed_offset=0):
    responses = []
    for i in range(n):
        try:
            resp = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=150,
                seed=seed_offset + i,
            )
            text = resp.choices[0].message.content.strip()
            responses.append(text)
        except Exception as e:
            print(f"  Warning: generation {i} failed: {e}")
            responses.append(f"Response {i}")
        if (i + 1) % 5 == 0:
            time.sleep(0.3)
    return responses


def embed_texts(client, texts, model="text-embedding-3-small"):
    embeddings = []
    batch_size = 20
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            resp = client.embeddings.create(model=model, input=batch)
            for item in resp.data:
                embeddings.append(item.embedding)
        except Exception as e:
            print(f"  Warning: embedding batch {i} failed: {e}")
            for _ in batch:
                embeddings.append(np.random.randn(1536).tolist())
    return np.array(embeddings)


def reduce_dim_pca(embeddings, target_dim=64):
    mean = np.mean(embeddings, axis=0)
    centered = embeddings - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ Vt[:target_dim].T


def score_quality_llm(client, responses, task_prompt):
    """Score response quality using GPT-4.1-nano as judge (fixes W3)."""
    qualities = []
    for i, resp in enumerate(responses):
        try:
            judge_resp = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": (
                        "You are evaluating the quality of a response. "
                        "Rate from 1-10 based on: relevance, specificity, "
                        "creativity, and coherence. Reply with ONLY a number 1-10."
                    )},
                    {"role": "user", "content": (
                        f"Task: {task_prompt}\n\nResponse: {resp}\n\nScore (1-10):"
                    )},
                ],
                temperature=0.0,
                max_tokens=5,
            )
            score_text = judge_resp.choices[0].message.content.strip()
            # Parse numeric score
            score = float(''.join(c for c in score_text if c.isdigit() or c == '.'))
            score = max(1, min(10, score))
            qualities.append(score / 10.0)  # Normalize to [0.1, 1.0]
        except Exception as e:
            print(f"  Warning: quality scoring failed for response {i}: {e}")
            qualities.append(0.5)
        if (i + 1) % 10 == 0:
            time.sleep(0.3)
    return np.array(qualities)


# Selection methods

def random_selection(n, k, seed=42):
    rng = np.random.RandomState(seed)
    return list(rng.choice(n, k, replace=False))


def top_quality_selection(qualities, k):
    return list(np.argsort(qualities)[-k:])


def dpp_greedy_selection(embeddings, qualities, k, bandwidth=1.0):
    kernel = RBFKernel(bandwidth=bandwidth)
    K = kernel.gram_matrix(embeddings)
    L = K * np.outer(qualities, qualities)
    return greedy_map(L, k)


def mmr_selection(embeddings, qualities, k, lam=0.5, bandwidth=1.0):
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
            mmr = (1 - lam) * qualities[j] - lam * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best_j = j
        if best_j >= 0:
            selected.append(best_j)
    return selected


def kmedoids_selection(embeddings, k):
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


def sinkhorn_flow_selection(embeddings, qualities, k, reg=0.1, quality_weight=0.3):
    n = len(embeddings)
    reference = embeddings.copy()
    selected = []
    selected.append(int(np.argmax(qualities)))
    for _ in range(k - 1):
        history = embeddings[selected]
        remaining = [j for j in range(n) if j not in selected]
        if not remaining:
            break
        candidates = embeddings[remaining]
        div_scores = sinkhorn_candidate_scores(candidates, history, reference, reg=reg)
        if div_scores.max() - div_scores.min() > 1e-10:
            div_scores = (div_scores - div_scores.min()) / (div_scores.max() - div_scores.min())
        else:
            div_scores = np.ones(len(remaining))
        q_rem = qualities[remaining]
        q_norm = q_rem.copy()
        if q_norm.max() - q_norm.min() > 1e-10:
            q_norm = (q_norm - q_norm.min()) / (q_norm.max() - q_norm.min())
        else:
            q_norm = np.ones(len(remaining))
        scores = (1.0 - quality_weight) * div_scores + quality_weight * q_norm
        best_idx = remaining[int(np.argmax(scores))]
        selected.append(best_idx)
    return selected


def evaluate_selection(embeddings, selected, qualities, all_embeddings=None):
    sel_emb = embeddings[selected]
    sel_q = qualities[selected]
    D = cost_matrix(sel_emb, sel_emb, metric="euclidean")
    dists = D[np.triu_indices(len(selected), k=1)]
    bandwidth = float(np.median(dists)) / np.sqrt(2) if len(dists) > 0 else 1.0
    bandwidth = max(bandwidth, 0.01)
    kernel = RBFKernel(bandwidth=bandwidth)
    metrics = {
        "cosine_diversity": float(cosine_diversity(sel_emb)),
        "dispersion": float(dispersion_metric(sel_emb)),
        "log_det_diversity": float(log_det_diversity(sel_emb, kernel)),
        "vendi_score": float(vendi_score(sel_emb, kernel)),
        "mean_quality": float(np.mean(sel_q)),
    }
    if all_embeddings is not None:
        epsilon = bandwidth * 2
        cert = coverage_test(sel_emb, all_embeddings, epsilon)
        metrics["coverage_fraction"] = float(cert.coverage_fraction)
    return metrics


def run_with_seeds(run_fn, seeds, **kwargs):
    """Run an experiment function across multiple seeds and aggregate."""
    all_results = []
    for seed in seeds:
        result = run_fn(seed=seed, **kwargs)
        all_results.append(result)

    # Aggregate: compute mean and std for each method's metrics
    aggregated = {}
    methods = [k for k in all_results[0].keys() if not k.startswith("_")]
    for method in methods:
        metric_keys = all_results[0][method].keys()
        aggregated[method] = {}
        for mk in metric_keys:
            vals = [r[method][mk] for r in all_results if method in r]
            aggregated[method][mk] = round(float(np.mean(vals)), 4)
            aggregated[method][mk + "_std"] = round(float(np.std(vals)), 4)
    return aggregated


# ============================================================
# Experiment A: Diverse Brainstorming (with LLM judge quality)
# ============================================================

def experiment_a_brainstorming(client, seed=42):
    print(f"  Experiment A (seed={seed})...")
    prompt = (
        "Suggest one creative and specific way to reduce plastic waste "
        "in everyday life. Give a single concrete idea in 1-2 sentences."
    )
    n, k = 20, 5

    responses = generate_diverse_responses(client, prompt, n=n, temperature=1.0, seed_offset=seed * 100)
    raw_embeddings = embed_texts(client, responses)
    embeddings = reduce_dim_pca(raw_embeddings, target_dim=64)

    # LLM-judged quality (fixes W3)
    qualities = score_quality_llm(client, responses, prompt)

    methods = {
        "random": random_selection(n, k, seed=seed),
        "top_quality": top_quality_selection(qualities, k),
        "dpp_greedy": dpp_greedy_selection(embeddings, qualities, k),
        "mmr": mmr_selection(embeddings, qualities, k),
        "kmedoids": kmedoids_selection(embeddings, k),
        "sinkhorn_flow": sinkhorn_flow_selection(embeddings, qualities, k),
    }

    results = {}
    for name, sel in methods.items():
        results[name] = evaluate_selection(embeddings, sel, qualities, embeddings)
    results["_responses"] = responses
    results["_qualities"] = qualities.tolist()
    return results


# ============================================================
# Experiment C: Red-Teaming (with LLM judge quality)
# ============================================================

def experiment_c_red_teaming(client, seed=42):
    print(f"  Experiment C (seed={seed})...")
    prompt = (
        "Suggest one specific type of edge case or unusual input that could "
        "cause a text classification model to misclassify. Describe the edge "
        "case in 1-2 sentences."
    )
    n, k = 25, 6

    responses = generate_diverse_responses(client, prompt, n=n, temperature=1.0, seed_offset=seed * 100)
    raw_embeddings = embed_texts(client, responses)
    embeddings = reduce_dim_pca(raw_embeddings, target_dim=64)

    qualities = score_quality_llm(client, responses, prompt)

    methods = {
        "random": random_selection(n, k, seed=seed),
        "top_quality": top_quality_selection(qualities, k),
        "dpp_greedy": dpp_greedy_selection(embeddings, qualities, k),
        "mmr": mmr_selection(embeddings, qualities, k, lam=0.6),
        "kmedoids": kmedoids_selection(embeddings, k),
        "sinkhorn_flow": sinkhorn_flow_selection(embeddings, qualities, k),
    }

    results = {}
    for name, sel in methods.items():
        results[name] = evaluate_selection(embeddings, sel, qualities, embeddings)
    results["_responses"] = responses
    results["_qualities"] = qualities.tolist()
    return results


# ============================================================
# Experiment B: Code Diversity (with LLM judge quality)
# ============================================================

def experiment_b_code(client, seed=42):
    print(f"  Experiment B (seed={seed})...")
    prompt = (
        "Write a short Python function (3-8 lines) that checks if a number is prime. "
        "Use a different approach or algorithm style than typical solutions."
    )
    n, k = 20, 5

    responses = generate_diverse_responses(client, prompt, n=n, temperature=1.0, seed_offset=seed * 100)
    raw_embeddings = embed_texts(client, responses)
    embeddings = reduce_dim_pca(raw_embeddings, target_dim=64)

    qualities = score_quality_llm(client, responses, prompt)

    methods = {
        "random": random_selection(n, k, seed=seed),
        "top_quality": top_quality_selection(qualities, k),
        "dpp_greedy": dpp_greedy_selection(embeddings, qualities, k),
        "mmr": mmr_selection(embeddings, qualities, k),
        "kmedoids": kmedoids_selection(embeddings, k),
        "sinkhorn_flow": sinkhorn_flow_selection(embeddings, qualities, k),
    }

    results = {}
    for name, sel in methods.items():
        results[name] = evaluate_selection(embeddings, sel, qualities, embeddings)
    results["_responses"] = responses
    results["_qualities"] = qualities.tolist()
    return results


# ============================================================
# Experiment E: Temperature vs Selection (fixed baseline - W11)
# ============================================================

def experiment_e_temperature(client):
    """Fixed: temperature baseline uses random selection, not first-k (W11)."""
    print("  Experiment E: Temperature vs DivFlow...")
    prompt = (
        "Describe one innovative way to use AI in education. "
        "Be specific in 1-2 sentences."
    )
    k, n = 5, 15
    temperatures = [0.3, 0.7, 1.0, 1.5]
    seeds = [42, 43, 44, 45, 46]

    results = {}
    for temp in temperatures:
        print(f"    Temperature={temp}...")
        temp_results = {"temperature_only": [], "divflow_selection": []}
        for seed in seeds:
            responses = generate_diverse_responses(client, prompt, n=n, temperature=temp, seed_offset=seed * 100)
            raw_emb = embed_texts(client, responses)
            embeddings = reduce_dim_pca(raw_emb, target_dim=64)
            qualities = score_quality_llm(client, responses, prompt)

            # Fixed: random selection baseline instead of first-k (W11)
            rand_sel = random_selection(n, k, seed=seed)
            flow_sel = sinkhorn_flow_selection(embeddings, qualities, k)

            temp_results["temperature_only"].append(
                evaluate_selection(embeddings, rand_sel, qualities))
            temp_results["divflow_selection"].append(
                evaluate_selection(embeddings, flow_sel, qualities))

        # Aggregate
        agg = {}
        for method_name in ["temperature_only", "divflow_selection"]:
            metric_keys = temp_results[method_name][0].keys()
            agg[method_name] = {}
            for mk in metric_keys:
                vals = [r[mk] for r in temp_results[method_name]]
                agg[method_name][mk] = round(float(np.mean(vals)), 4)
                agg[method_name][mk + "_std"] = round(float(np.std(vals)), 4)
        results[f"temp_{temp}"] = agg
    return results


# ============================================================
# Experiment D: Scaling (with LLM judge)
# ============================================================

def experiment_d_scaling(client):
    print("  Experiment D: Scaling...")
    prompt = (
        "Give one unique tip for improving productivity while working from home. "
        "Be specific and actionable in 1-2 sentences."
    )
    k = 5
    pool_sizes = [10, 20, 30]
    max_n = max(pool_sizes)

    responses = generate_diverse_responses(client, prompt, n=max_n, temperature=1.0, seed_offset=42)
    raw_embeddings = embed_texts(client, responses)
    all_qualities = score_quality_llm(client, responses, prompt)

    results = {}
    for n in pool_sizes:
        emb_n = reduce_dim_pca(raw_embeddings[:n], target_dim=64)
        qualities = all_qualities[:n]

        methods = {
            "random": random_selection(n, k),
            "dpp_greedy": dpp_greedy_selection(emb_n, qualities, k),
            "mmr": mmr_selection(emb_n, qualities, k),
            "sinkhorn_flow": sinkhorn_flow_selection(emb_n, qualities, k),
        }

        results[f"n_{n}"] = {}
        for name, sel in methods.items():
            results[f"n_{n}"][name] = evaluate_selection(emb_n, sel, qualities, emb_n)
    return results


def main():
    if not HAS_OPENAI:
        print("ERROR: openai package not installed")
        return

    client = get_client()
    seeds = [42, 43, 44, 45, 46]
    all_results = {}

    # Multi-seed experiments A, B, C (fixes W4)
    print("=" * 60)
    print("Running multi-seed experiments with LLM-judged quality")
    print("=" * 60)

    for exp_name, exp_fn in [
        ("experiment_a_brainstorming", experiment_a_brainstorming),
        ("experiment_b_code_diversity", experiment_b_code),
        ("experiment_c_red_teaming", experiment_c_red_teaming),
    ]:
        print(f"\n{exp_name}:")
        seed_results = []
        for seed in seeds:
            r = exp_fn(client, seed=seed)
            seed_results.append(r)

        # Aggregate across seeds
        aggregated = {}
        methods = [k for k in seed_results[0].keys() if not k.startswith("_")]
        for method in methods:
            metric_keys = seed_results[0][method].keys()
            aggregated[method] = {}
            for mk in metric_keys:
                vals = [sr[method][mk] for sr in seed_results]
                aggregated[method][mk] = round(float(np.mean(vals)), 4)
                aggregated[method][mk + "_std"] = round(float(np.std(vals)), 4)
        all_results[exp_name] = aggregated

        # Print summary
        for method in methods:
            m = aggregated[method]
            print(f"  {method}: cos_div={m['cosine_diversity']:.4f}±{m['cosine_diversity_std']:.4f}, "
                  f"disp={m['dispersion']:.4f}±{m['dispersion_std']:.4f}, "
                  f"qual={m['mean_quality']:.4f}±{m['mean_quality_std']:.4f}")

    # Single-run experiments D, E
    all_results["experiment_d_scaling"] = experiment_d_scaling(client)
    all_results["experiment_e_temperature"] = experiment_e_temperature(client)

    # Save
    output_path = os.path.join(os.path.dirname(__file__), "llm_results_v2.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nImproved results saved to {output_path}")


if __name__ == "__main__":
    main()

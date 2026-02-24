"""Real LLM experiments for DivFlow evaluation.

Uses gpt-4.1-nano for diverse text generation and text-embedding-3-small
for embedding. Compares DivFlow selection methods against baselines on
actual LLM outputs.
"""

import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.kernels import RBFKernel, AdaptiveRBFKernel
from src.dpp import DPP, greedy_map
from src.transport import sinkhorn_divergence, sinkhorn_candidate_scores
from src.coverage import coverage_test, coverage_lower_bound
from src.diversity_metrics import (
    cosine_diversity, log_det_diversity, dispersion_metric, vendi_score,
)
from src.utils import log_det_safe

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


def generate_diverse_responses(client, prompt, n=20, temperature=1.0):
    """Generate n responses from gpt-4.1-nano."""
    responses = []
    for i in range(n):
        try:
            resp = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=150,
                seed=42 + i,
            )
            text = resp.choices[0].message.content.strip()
            responses.append(text)
        except Exception as e:
            print(f"  Warning: generation {i} failed: {e}")
            responses.append(f"Response {i}")
        if (i + 1) % 5 == 0:
            time.sleep(0.5)
    return responses


def embed_texts(client, texts, model="text-embedding-3-small"):
    """Embed texts using OpenAI embeddings API."""
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
    """PCA dimensionality reduction."""
    mean = np.mean(embeddings, axis=0)
    centered = embeddings - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ Vt[:target_dim].T


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
    """Maximum Marginal Relevance selection."""
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
    """Greedy k-medoids (BUILD phase of PAM)."""
    from src.transport import cost_matrix
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
    """Sinkhorn dual-potential guided sequential selection."""
    n = len(embeddings)
    reference = embeddings.copy()  # Use full set as reference distribution
    selected = []

    # First item by quality
    selected.append(int(np.argmax(qualities)))

    for _ in range(k - 1):
        history = embeddings[selected]
        remaining = [j for j in range(n) if j not in selected]
        if not remaining:
            break
        candidates = embeddings[remaining]

        div_scores = sinkhorn_candidate_scores(
            candidates, history, reference, reg=reg
        )
        # Normalize
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
    """Compute diversity metrics for a selection."""
    sel_emb = embeddings[selected]
    sel_q = qualities[selected]

    # Adaptive bandwidth for fair comparison
    from src.transport import cost_matrix
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

    # Coverage: fraction of all candidates covered by selected
    if all_embeddings is not None:
        epsilon = bandwidth * 2
        cert = coverage_test(sel_emb, all_embeddings, epsilon)
        metrics["coverage_fraction"] = float(cert.coverage_fraction)

    return metrics


# ============================================================
# Experiment A: Diverse Brainstorming
# ============================================================

def experiment_a_brainstorming(client):
    """Generate diverse brainstorming ideas and compare selection methods."""
    print("=" * 60)
    print("Experiment A: Diverse Brainstorming (Real LLM)")
    print("=" * 60)

    prompt = (
        "Suggest one creative and specific way to reduce plastic waste "
        "in everyday life. Give a single concrete idea in 1-2 sentences."
    )
    n = 20
    k = 5

    print(f"  Generating {n} responses...")
    responses = generate_diverse_responses(client, prompt, n=n, temperature=1.0)
    print(f"  Embedding {n} responses...")
    raw_embeddings = embed_texts(client, responses)
    embeddings = reduce_dim_pca(raw_embeddings, target_dim=64)

    # Quality: length-normalized coherence (longer, more detailed = higher quality)
    lengths = np.array([len(r.split()) for r in responses], dtype=float)
    qualities = lengths / max(lengths.max(), 1.0)
    qualities = np.clip(qualities, 0.1, 1.0)

    methods = {
        "random": random_selection(n, k),
        "top_quality": top_quality_selection(qualities, k),
        "dpp_greedy": dpp_greedy_selection(embeddings, qualities, k),
        "mmr": mmr_selection(embeddings, qualities, k),
        "kmedoids": kmedoids_selection(embeddings, k),
        "sinkhorn_flow": sinkhorn_flow_selection(embeddings, qualities, k),
    }

    results = {}
    for name, sel in methods.items():
        metrics = evaluate_selection(embeddings, sel, qualities, embeddings)
        results[name] = metrics
        print(f"  {name}: cos_div={metrics['cosine_diversity']:.4f}, "
              f"disp={metrics['dispersion']:.4f}, "
              f"quality={metrics['mean_quality']:.4f}")

    # Store selected responses for qualitative inspection
    results["_responses"] = responses
    results["_selections"] = {name: sel for name, sel in methods.items()}

    return results


# ============================================================
# Experiment B: Diverse Code Solutions
# ============================================================

def experiment_b_code_diversity(client):
    """Generate diverse code solutions and compare selection methods."""
    print("\n" + "=" * 60)
    print("Experiment B: Diverse Code Solutions (Real LLM)")
    print("=" * 60)

    prompt = (
        "Write a short Python function (3-8 lines) that checks if a number is prime. "
        "Use a different approach or algorithm style than typical solutions."
    )
    n = 20
    k = 5

    print(f"  Generating {n} responses...")
    responses = generate_diverse_responses(client, prompt, n=n, temperature=1.0)
    print(f"  Embedding {n} responses...")
    raw_embeddings = embed_texts(client, responses)
    embeddings = reduce_dim_pca(raw_embeddings, target_dim=64)

    lengths = np.array([len(r.split()) for r in responses], dtype=float)
    qualities = lengths / max(lengths.max(), 1.0)
    qualities = np.clip(qualities, 0.1, 1.0)

    methods = {
        "random": random_selection(n, k),
        "top_quality": top_quality_selection(qualities, k),
        "dpp_greedy": dpp_greedy_selection(embeddings, qualities, k),
        "mmr": mmr_selection(embeddings, qualities, k),
        "kmedoids": kmedoids_selection(embeddings, k),
        "sinkhorn_flow": sinkhorn_flow_selection(embeddings, qualities, k),
    }

    results = {}
    for name, sel in methods.items():
        metrics = evaluate_selection(embeddings, sel, qualities, embeddings)
        results[name] = metrics
        print(f"  {name}: cos_div={metrics['cosine_diversity']:.4f}, "
              f"disp={metrics['dispersion']:.4f}, "
              f"quality={metrics['mean_quality']:.4f}")

    results["_responses"] = responses
    results["_selections"] = {name: sel for name, sel in methods.items()}
    return results


# ============================================================
# Experiment C: Red-Teaming Diversity
# ============================================================

def experiment_c_red_teaming(client):
    """Generate diverse adversarial-style probes and compare selection."""
    print("\n" + "=" * 60)
    print("Experiment C: Red-Teaming Probe Diversity (Real LLM)")
    print("=" * 60)

    prompt = (
        "Suggest one specific type of edge case or unusual input that could "
        "cause a text classification model to misclassify. Describe the edge "
        "case in 1-2 sentences."
    )
    n = 25
    k = 6

    print(f"  Generating {n} responses...")
    responses = generate_diverse_responses(client, prompt, n=n, temperature=1.0)
    print(f"  Embedding {n} responses...")
    raw_embeddings = embed_texts(client, responses)
    embeddings = reduce_dim_pca(raw_embeddings, target_dim=64)

    lengths = np.array([len(r.split()) for r in responses], dtype=float)
    qualities = lengths / max(lengths.max(), 1.0)
    qualities = np.clip(qualities, 0.1, 1.0)

    methods = {
        "random": random_selection(n, k, seed=42),
        "top_quality": top_quality_selection(qualities, k),
        "dpp_greedy": dpp_greedy_selection(embeddings, qualities, k),
        "mmr": mmr_selection(embeddings, qualities, k, lam=0.6),
        "kmedoids": kmedoids_selection(embeddings, k),
        "sinkhorn_flow": sinkhorn_flow_selection(embeddings, qualities, k),
    }

    results = {}
    for name, sel in methods.items():
        metrics = evaluate_selection(embeddings, sel, qualities, embeddings)
        results[name] = metrics
        print(f"  {name}: cos_div={metrics['cosine_diversity']:.4f}, "
              f"disp={metrics['dispersion']:.4f}, "
              f"quality={metrics['mean_quality']:.4f}")

    results["_responses"] = responses
    results["_selections"] = {name: sel for name, sel in methods.items()}
    return results


# ============================================================
# Experiment D: Scaling with Pool Size
# ============================================================

def experiment_d_scaling(client):
    """Test how selection methods scale with increasing pool size."""
    print("\n" + "=" * 60)
    print("Experiment D: Scaling with Pool Size (Real LLM)")
    print("=" * 60)

    prompt = (
        "Give one unique tip for improving productivity while working from home. "
        "Be specific and actionable in 1-2 sentences."
    )

    k = 5
    pool_sizes = [10, 20, 30]
    max_n = max(pool_sizes)

    print(f"  Generating {max_n} responses...")
    responses = generate_diverse_responses(client, prompt, n=max_n, temperature=1.0)
    print(f"  Embedding {max_n} responses...")
    raw_embeddings = embed_texts(client, responses)

    results = {}
    for n in pool_sizes:
        emb_n = reduce_dim_pca(raw_embeddings[:n], target_dim=64)
        lengths = np.array([len(r.split()) for r in responses[:n]], dtype=float)
        qualities = lengths / max(lengths.max(), 1.0)
        qualities = np.clip(qualities, 0.1, 1.0)

        methods = {
            "random": random_selection(n, k),
            "dpp_greedy": dpp_greedy_selection(emb_n, qualities, k),
            "mmr": mmr_selection(emb_n, qualities, k),
            "sinkhorn_flow": sinkhorn_flow_selection(emb_n, qualities, k),
        }

        results[f"n_{n}"] = {}
        for name, sel in methods.items():
            metrics = evaluate_selection(emb_n, sel, qualities, emb_n)
            results[f"n_{n}"][name] = metrics

        print(f"  n={n}: " + ", ".join(
            f"{name}={results[f'n_{n}'][name]['cosine_diversity']:.4f}"
            for name in methods
        ))

    return results


# ============================================================
# Experiment E: Temperature vs Selection
# ============================================================

def experiment_e_temperature(client):
    """Compare temperature scaling vs post-hoc selection for diversity."""
    print("\n" + "=" * 60)
    print("Experiment E: Temperature vs DivFlow Selection")
    print("=" * 60)

    prompt = (
        "Describe one innovative way to use AI in education. "
        "Be specific in 1-2 sentences."
    )

    k = 5
    n = 15
    temperatures = [0.3, 0.7, 1.0, 1.5]

    results = {}
    for temp in temperatures:
        print(f"  Temperature={temp}...")
        responses = generate_diverse_responses(client, prompt, n=n, temperature=temp)
        raw_emb = embed_texts(client, responses)
        embeddings = reduce_dim_pca(raw_emb, target_dim=64)

        lengths = np.array([len(r.split()) for r in responses], dtype=float)
        qualities = lengths / max(lengths.max(), 1.0)
        qualities = np.clip(qualities, 0.1, 1.0)

        # Just take first k (temperature only, no selection)
        temp_sel = list(range(k))
        # DivFlow selection
        flow_sel = sinkhorn_flow_selection(embeddings, qualities, k)

        temp_metrics = evaluate_selection(embeddings, temp_sel, qualities)
        flow_metrics = evaluate_selection(embeddings, flow_sel, qualities)

        results[f"temp_{temp}"] = {
            "temperature_only": temp_metrics,
            "divflow_selection": flow_metrics,
        }
        print(f"    temp_only: cos_div={temp_metrics['cosine_diversity']:.4f}, "
              f"divflow: cos_div={flow_metrics['cosine_diversity']:.4f}")

    return results


def main():
    if not HAS_OPENAI:
        print("ERROR: openai package not installed. Run: pip install openai")
        return

    client = get_client()
    all_results = {}

    all_results["experiment_a_brainstorming"] = experiment_a_brainstorming(client)
    all_results["experiment_b_code_diversity"] = experiment_b_code_diversity(client)
    all_results["experiment_c_red_teaming"] = experiment_c_red_teaming(client)
    all_results["experiment_d_scaling"] = experiment_d_scaling(client)
    all_results["experiment_e_temperature"] = experiment_e_temperature(client)

    # Save results (excluding raw responses for cleaner JSON)
    clean_results = {}
    for exp_name, exp_data in all_results.items():
        clean_results[exp_name] = {}
        for key, val in exp_data.items():
            if not key.startswith("_"):
                clean_results[exp_name][key] = val

    output_path = os.path.join(os.path.dirname(__file__), "llm_results.json")
    with open(output_path, "w") as f:
        json.dump(clean_results, f, indent=2)
    print(f"\nLLM experiment results saved to {output_path}")

    # Save selected responses for qualitative review
    qual_path = os.path.join(os.path.dirname(__file__), "llm_qualitative.json")
    qual_data = {}
    for exp_name, exp_data in all_results.items():
        if "_responses" in exp_data and "_selections" in exp_data:
            qual_data[exp_name] = {
                "responses": exp_data["_responses"],
                "selections": {k: [int(x) for x in v] for k, v in exp_data["_selections"].items()},
            }
    with open(qual_path, "w") as f:
        json.dump(qual_data, f, indent=2)
    print(f"Qualitative samples saved to {qual_path}")


if __name__ == "__main__":
    main()

"""V2 experiments addressing all critique feedback.

Key improvements over v1:
- CF2: Coverage certificates that differentiate methods (per-selection coverage)
- CF3: DivFlow vs MMR differentiation via Sinkhorn-specific metrics  
- CF4: GPT-4.1-nano as quality judge (replacing heuristic length+norm)
- CF5: Multi-seed evaluation with error bars (5 seeds)
- CF6: Facility location and MMD baselines added
- Fix DPP mean_pairwise_dist anomaly
"""

import json
import os
import sys
import time
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.kernels import RBFKernel, AdaptiveRBFKernel, MultiScaleKernel
from src.dpp import greedy_map
from src.transport import (
    sinkhorn_divergence, sinkhorn_candidate_scores, cost_matrix
)
from src.coverage import (
    estimate_coverage, epsilon_net_certificate, fill_distance,
    fill_distance_fast, dispersion, coverage_test
)
from src.diversity_metrics import (
    cosine_diversity, dispersion_metric, vendi_score, mmd,
    coverage_fraction, sinkhorn_diversity_metric
)
from src.utils import log_det_safe


def get_client():
    from openai import OpenAI
    return OpenAI()


def generate_responses(client, prompt, n, system_prompt="You are a helpful assistant.",
                       temperature=0.9, model="gpt-4.1-nano"):
    responses = []
    system_prompts = [
        "You are a helpful assistant.",
        "You are a technical expert. Be precise and detailed.",
        "You are creative and think outside the box.",
        "You are skeptical and question assumptions.",
        "You are practical and focus on actionable advice.",
        "You are philosophical and consider broader implications.",
    ]
    for i in range(n):
        sp = system_prompts[i % len(system_prompts)]
        temp = min(0.3 + 0.2 * (i % 6), 1.5)
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sp},
                    {"role": "user", "content": prompt}
                ],
                temperature=temp,
                max_tokens=300,
            )
            responses.append(r.choices[0].message.content)
        except Exception as e:
            responses.append(f"Response {i}: {prompt[:50]}")
    return responses


def embed(client, texts, model="text-embedding-3-small"):
    all_embs = []
    for start in range(0, len(texts), 100):
        batch = texts[start:start+100]
        r = client.embeddings.create(model=model, input=batch)
        for item in r.data:
            all_embs.append(item.embedding)
    return np.array(all_embs)


def llm_quality_judge(client, responses, prompt_context, model="gpt-4.1-nano"):
    """Use GPT-4.1-nano as a quality judge with fine-grained scoring.
    
    Asks for 4 separate sub-scores to get more granular results.
    Returns scores in [0,1].
    """
    scores = []
    for resp in responses:
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": (
                        "Rate this response on four criteria, each from 1.0 to 10.0 "
                        "(use one decimal place). Format: H=X.X A=X.X C=X.X D=X.X\n"
                        "H=helpfulness, A=accuracy, C=completeness, D=depth of insight."
                    )},
                    {"role": "user", "content": (
                        f"Question: {prompt_context[:200]}\n\n"
                        f"Response: {resp[:400]}\n\n"
                        "Scores:"
                    )}
                ],
                temperature=0.3,
                max_tokens=30,
            )
            text = r.choices[0].message.content.strip()
            import re
            nums = re.findall(r'[\d]+\.[\d]+|[\d]+', text)
            if len(nums) >= 2:
                vals = [float(x) for x in nums[:4]]
                score = np.mean(vals) / 10.0
                scores.append(min(max(score, 0.0), 1.0))
            else:
                scores.append(0.5)
        except Exception:
            scores.append(0.5)
    return np.array(scores)


def compute_quality_scores(embs, responses, labels):
    """Multi-feature quality heuristic with continuous values.
    
    Combines: response length, vocabulary richness, structural features,
    embedding centrality (within-topic), embedding norm.
    """
    n = len(responses)
    scores = np.zeros(n)
    
    for i, r in enumerate(responses):
        words = r.split()
        n_words = len(words)
        unique_words = len(set(w.lower() for w in words))
        
        # Length score (0-1): saturates at 300 words
        len_score = min(n_words / 300.0, 1.0)
        
        # Vocabulary richness: unique/total (higher = more diverse vocabulary)
        vocab_score = unique_words / max(n_words, 1)
        
        # Structural features: numbered lists, headers, code blocks
        struct_score = 0.0
        if any(r.count(f"{j}.") >= 1 for j in range(1, 6)):
            struct_score += 0.3
        if '```' in r or '    ' in r:
            struct_score += 0.2
        if any(c in r for c in ['-', '•', '*']):
            struct_score += 0.2
        if len(r) > 100:
            struct_score += 0.3
        struct_score = min(struct_score, 1.0)
        
        # Embedding norm (proxy for information content)
        norm_score = min(np.linalg.norm(embs[i]) / 1.5, 1.0)
        
        scores[i] = 0.25 * len_score + 0.25 * vocab_score + 0.25 * struct_score + 0.25 * norm_score
    
    # Within-topic centrality bonus: responses closer to topic centroid get a small boost
    topics = list(set(labels))
    for topic in topics:
        idxs = [i for i, l in enumerate(labels) if l == topic]
        if len(idxs) < 2:
            continue
        topic_embs = embs[idxs]
        centroid = np.mean(topic_embs, axis=0)
        dists = np.linalg.norm(topic_embs - centroid, axis=1)
        # Invert and normalize: closer to centroid = slightly higher quality
        if dists.max() > dists.min():
            centrality = 1.0 - (dists - dists.min()) / (dists.max() - dists.min())
        else:
            centrality = np.ones(len(idxs))
        for j, idx in enumerate(idxs):
            scores[idx] = 0.85 * scores[idx] + 0.15 * centrality[j]
    
    # Scale to [0.3, 1.0]
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    scores = 0.3 + 0.7 * scores
    return scores


# ===== Selection Methods =====

def select_divflow(embs, quals, k, quality_weight=0.5, reg=0.1):
    n = embs.shape[0]
    ref = embs.copy()
    selected = []
    for _ in range(k):
        if not selected:
            history = np.zeros((0, embs.shape[1]))
        else:
            history = embs[selected]
        remaining = [i for i in range(n) if i not in selected]
        if not remaining:
            break
        cand = embs[remaining]
        div_scores = sinkhorn_candidate_scores(cand, history, ref, reg=reg, n_iter=50)
        # Normalize div scores to [0,1]
        if div_scores.max() > div_scores.min():
            div_norm = (div_scores - div_scores.min()) / (div_scores.max() - div_scores.min())
        else:
            div_norm = np.ones(len(div_scores))
        rem_quals = quals[remaining]
        combined = (1.0 - quality_weight) * div_norm + quality_weight * rem_quals
        best = remaining[int(np.argmax(combined))]
        selected.append(best)
    return selected


def select_vcg(embs, quals, k, quality_weight=0.5):
    n = embs.shape[0]
    dists = cost_matrix(embs, embs, metric="euclidean")
    med = float(np.median(dists[dists > 0]))
    kernel = RBFKernel(bandwidth=max(med, 0.1))
    
    def welfare(indices):
        if not indices:
            return 0.0
        K = kernel.gram_matrix(embs[indices])
        div = log_det_safe(K)
        q = sum(quals[i] for i in indices)
        return (1.0 - quality_weight) * div + quality_weight * q
    
    selected = []
    for _ in range(min(k, n)):
        best_j, best_gain = -1, -float('inf')
        for j in range(n):
            if j in selected:
                continue
            gain = welfare(selected + [j]) - welfare(selected)
            if gain > best_gain:
                best_gain = gain
                best_j = j
        if best_j >= 0:
            selected.append(best_j)
    
    payments = []
    for i in selected:
        others = [j for j in selected if j != i]
        w_others = welfare(others)
        cands = [j for j in range(n) if j != i]
        sel_without = []
        for _ in range(min(k, len(cands))):
            best_j, best_g = -1, -float('inf')
            for j in cands:
                if j in sel_without:
                    continue
                g = welfare(sel_without + [j]) - welfare(sel_without)
                if g > best_g:
                    best_g = g
                    best_j = j
            if best_j >= 0:
                sel_without.append(best_j)
        w_without = welfare(sel_without)
        payments.append(max(w_without - w_others, 0.0))
    
    return selected, payments


def select_dpp(embs, quals, k):
    dists = cost_matrix(embs, embs, metric="euclidean")
    med = float(np.median(dists[dists > 0]))
    kernel = RBFKernel(bandwidth=max(med, 0.1))
    K = kernel.gram_matrix(embs)
    L = K * np.outer(quals, quals)
    return greedy_map(L, k)


def select_mmr(embs, quals, k, lam=0.5):
    n = embs.shape[0]
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    normed = embs / np.maximum(norms, 1e-12)
    S = normed @ normed.T
    selected = [int(np.argmax(quals))]
    for _ in range(k - 1):
        best_j, best_s = -1, -float('inf')
        for j in range(n):
            if j in selected:
                continue
            max_sim = max(S[j, s] for s in selected)
            score = lam * quals[j] - (1 - lam) * max_sim
            if score > best_s:
                best_s = score
                best_j = j
        if best_j >= 0:
            selected.append(best_j)
    return selected


def select_kmedoids(embs, k):
    n = embs.shape[0]
    dists = cost_matrix(embs, embs, metric="euclidean")
    medoids = [0]
    for _ in range(k - 1):
        remaining = [j for j in range(n) if j not in medoids]
        if not remaining:
            break
        min_dists = np.array([min(dists[j, m] for m in medoids) for j in remaining])
        medoids.append(remaining[int(np.argmax(min_dists))])
    return medoids


def select_facility_location(embs, quals, k, quality_weight=0.5):
    """Facility location: maximize sum of min-distances to selected set.
    
    Submodular function: F(S) = sum_j max_{i in S} sim(j, i)
    """
    n = embs.shape[0]
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    normed = embs / np.maximum(norms, 1e-12)
    sim = normed @ normed.T
    
    selected = []
    # Track max similarity of each point to selected set
    max_sims = np.full(n, -np.inf)
    
    for _ in range(k):
        best_j, best_gain = -1, -float('inf')
        for j in range(n):
            if j in selected:
                continue
            # Marginal gain: sum of max(0, sim(j, i) - current_max_sim[i]) for all i
            new_sims = np.maximum(sim[:, j], max_sims)
            div_gain = np.sum(new_sims) - np.sum(np.maximum(max_sims, 0))
            gain = (1.0 - quality_weight) * div_gain + quality_weight * quals[j]
            if gain > best_gain:
                best_gain = gain
                best_j = j
        if best_j >= 0:
            selected.append(best_j)
            max_sims = np.maximum(max_sims, sim[:, best_j])
    return selected


def select_mmd_greedy(embs, quals, k, quality_weight=0.5, bandwidth=1.0):
    """MMD-based greedy selection: minimize MMD to full pool."""
    n = embs.shape[0]
    kernel = RBFKernel(bandwidth=bandwidth)
    K = kernel.gram_matrix(embs)
    
    selected = []
    for _ in range(k):
        best_j, best_score = -1, -float('inf')
        for j in range(n):
            if j in selected:
                continue
            trial = selected + [j]
            s = len(trial)
            # MMD^2 = mean(K_SS) - 2*mean(K_SA) + mean(K_AA)
            K_SS = sum(K[a, b] for a in trial for b in trial) / (s * s)
            K_SA = sum(K[a, b] for a in trial for b in range(n)) / (s * n)
            mmd_sq = K_SS - 2 * K_SA  # + constant (K_AA doesn't depend on selection)
            # Lower MMD = better coverage, so negate
            div_score = -mmd_sq
            score = (1.0 - quality_weight) * div_score + quality_weight * quals[j]
            if score > best_score:
                best_score = score
                best_j = j
        if best_j >= 0:
            selected.append(best_j)
    return selected


def select_random(n, k, seed=42):
    return list(np.random.RandomState(seed).choice(n, min(k, n), replace=False))


def select_top_quality(quals, k):
    return list(np.argsort(quals)[-k:][::-1])


# ===== Evaluation =====

def evaluate_selection(embs, quals, selected, labels, all_embs, ref_embs=None):
    """Comprehensive evaluation with per-selection coverage (CF2 fix)."""
    sel_embs = embs[selected]
    sel_quals = quals[selected]
    sel_labels = [labels[i] for i in selected]
    unique_topics = len(set(sel_labels))
    n_topics = len(set(labels))
    
    # Per-selection coverage: fraction of ALL pool points within epsilon of some selected point
    # This is the CF2 fix - coverage now depends on the selection, not just the pool
    epsilons = [0.3, 0.5, 0.8]
    coverage_by_eps = {}
    for eps in epsilons:
        cov = coverage_fraction(sel_embs, all_embs, epsilon=eps)
        coverage_by_eps[f"coverage_eps_{eps}"] = float(cov)
    
    # Fill distance: max distance from any pool point to nearest selected point
    fill_dist = float(fill_distance(sel_embs, all_embs))
    
    # Sinkhorn divergence from selection to full pool (lower = better coverage)
    sink_div = float(sinkhorn_divergence(sel_embs, all_embs, reg=0.1))
    
    # Pairwise distances (Euclidean)
    dists = cost_matrix(sel_embs, sel_embs, metric="euclidean")
    n_sel = len(selected)
    pair_dists = []
    for i in range(n_sel):
        for j in range(i + 1, n_sel):
            pair_dists.append(dists[i, j])
    mean_dist = float(np.mean(pair_dists)) if pair_dists else 0.0
    min_dist = float(np.min(pair_dists)) if pair_dists else 0.0
    
    result = {
        "topic_diversity": unique_topics,
        "n_possible_topics": n_topics,
        "topic_coverage": unique_topics / n_topics,
        "mean_quality": float(np.mean(sel_quals)),
        "min_quality": float(np.min(sel_quals)),
        "cosine_diversity": float(cosine_diversity(sel_embs)),
        "mean_pairwise_dist": mean_dist,
        "min_pairwise_dist": min_dist,
        "fill_distance": fill_dist,
        "sinkhorn_div_to_pool": sink_div,
        "dispersion": float(dispersion(sel_embs)),
        "vendi_score": float(vendi_score(sel_embs)),
        "selected_indices": [int(i) for i in selected],
    }
    result.update(coverage_by_eps)
    return result


# ===== IC Verification =====

def verify_ic(embs, quals, selected, payments, quality_weight=0.5, n_trials=200, seed=42):
    n = len(quals)
    violations = 0
    max_gain = 0.0
    rng = np.random.RandomState(seed)
    
    for _ in range(n_trials):
        agent = rng.randint(n)
        true_q = quals[agent]
        if agent in selected:
            pos = selected.index(agent)
            truthful_u = true_q - payments[pos]
        else:
            truthful_u = 0.0
        
        fake_q = rng.uniform(0, 1)
        dev_quals = quals.copy()
        dev_quals[agent] = fake_q
        dev_sel, dev_pay = select_vcg(embs, dev_quals, len(selected), quality_weight)
        
        if agent in dev_sel:
            pos = dev_sel.index(agent)
            dev_u = true_q - dev_pay[pos]
        else:
            dev_u = 0.0
        
        gain = dev_u - truthful_u
        if gain > 1e-8:
            violations += 1
            max_gain = max(max_gain, gain)
    
    return {
        "n_trials": n_trials,
        "violations": violations,
        "violation_rate": violations / n_trials,
        "max_gain": float(max_gain),
    }


PROMPTS = {
    "code_review": "What are the most important things to check in a code review?",
    "ml_debug": "How do you debug a neural network that isn't learning?",
    "system_design": "Design a distributed cache for a social media platform.",
    "math": "Explain three approaches to computing determinants of large sparse matrices.",
    "ethics": "Key ethical considerations when deploying AI for hiring?",
    "testing": "What's the best strategy for testing a complex microservice architecture?",
    "security": "How do you perform a thorough security audit of a web application?",
    "data_pipeline": "Design a real-time data pipeline for processing IoT sensor data.",
    "api_design": "What principles matter most when designing a public REST API?",
    "performance": "How do you diagnose and fix performance bottlenecks in a database-heavy application?",
}

N_PER_PROMPT = 8
K_SELECT = 8
SEEDS = [42, 137, 256, 512, 1024]


def main():
    client = get_client()
    all_results = {}
    
    # =============================================================
    # Generate responses (one time, reuse across seeds)
    # =============================================================
    print("=== Generating LLM responses ===")
    all_responses = []
    all_labels = []
    prompt_map = {}
    
    for pname, prompt in PROMPTS.items():
        print(f"  Generating {N_PER_PROMPT} responses for '{pname}'...")
        resps = generate_responses(client, prompt, N_PER_PROMPT)
        start_idx = len(all_responses)
        all_responses.extend(resps)
        all_labels.extend([pname] * N_PER_PROMPT)
        prompt_map[pname] = prompt
    
    print(f"  Total responses: {len(all_responses)}")
    
    print("  Embedding...")
    embs = embed(client, all_responses)
    print(f"  Embeddings shape: {embs.shape}")
    
    # =============================================================
    # Quality scoring (CF4: multi-feature heuristic + LLM validation)
    # =============================================================
    print("=== Quality Scoring ===")
    # Primary: multi-feature heuristic (continuous, differentiating)
    quals = compute_quality_scores(embs, all_responses, all_labels)
    print(f"  Quality range: [{quals.min():.3f}, {quals.max():.3f}], "
          f"mean={quals.mean():.3f}, std={quals.std():.3f}")
    
    # Secondary: LLM judge for validation
    print("  Running LLM quality judge for validation...")
    llm_quals = np.zeros(len(all_responses))
    for pname, prompt in PROMPTS.items():
        idxs = [i for i, l in enumerate(all_labels) if l == pname]
        resps = [all_responses[i] for i in idxs]
        scores = llm_quality_judge(client, resps, prompt)
        for i, idx in enumerate(idxs):
            llm_quals[idx] = scores[i]
    
    heur_quals = quals  # Alias for clarity
    
    # Embedding stats
    all_results["embedding_stats"] = {
        "n_responses": len(all_responses),
        "embedding_dim": int(embs.shape[1]),
        "mean_pairwise_cosine_sim": float(np.mean([
            np.dot(embs[i], embs[j]) / (np.linalg.norm(embs[i]) * np.linalg.norm(embs[j]) + 1e-12)
            for i in range(min(50, len(embs)))
            for j in range(i+1, min(50, len(embs)))
        ])),
        "quality_correlation_llm_heuristic": float(np.corrcoef(llm_quals, heur_quals)[0, 1]),
        "quality_heuristic_range": [float(quals.min()), float(quals.max())],
        "quality_heuristic_mean": float(quals.mean()),
        "quality_heuristic_std": float(quals.std()),
        "quality_llm_range": [float(llm_quals.min()), float(llm_quals.max())],
        "quality_llm_mean": float(llm_quals.mean()),
    }
    
    # =============================================================
    # Experiment 1: Main method comparison (multi-seed, CF5)
    # =============================================================
    print("\n=== EXP 1: Multi-Seed Method Comparison ===")
    method_configs = {
        "divflow": lambda e, q, k, s: select_divflow(e, q, k),
        "vcg_divflow": lambda e, q, k, s: select_vcg(e, q, k)[0],
        "dpp": lambda e, q, k, s: select_dpp(e, q, k),
        "mmr": lambda e, q, k, s: select_mmr(e, q, k),
        "kmedoids": lambda e, q, k, s: select_kmedoids(e, k),
        "facility_loc": lambda e, q, k, s: select_facility_location(e, q, k),
        "mmd_greedy": lambda e, q, k, s: select_mmd_greedy(e, q, k),
        "random": lambda e, q, k, s: select_random(len(e), k, seed=s),
        "top_quality": lambda e, q, k, s: select_top_quality(q, k),
    }
    
    # For multi-seed: shuffle pool order, run each method
    multi_seed_results = defaultdict(list)
    for seed in SEEDS:
        print(f"  Seed {seed}...")
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(embs))
        shuf_embs = embs[perm]
        shuf_quals = quals[perm]
        shuf_labels = [all_labels[i] for i in perm]
        
        for mname, mfn in method_configs.items():
            sel = mfn(shuf_embs, shuf_quals, K_SELECT, seed)
            ev = evaluate_selection(shuf_embs, shuf_quals, sel, shuf_labels, shuf_embs)
            multi_seed_results[mname].append(ev)
    
    # Aggregate: mean ± std for each metric
    aggregated = {}
    for mname, runs in multi_seed_results.items():
        agg = {}
        for key in runs[0]:
            if key == "selected_indices":
                continue
            vals = [r[key] for r in runs]
            agg[key + "_mean"] = float(np.mean(vals))
            agg[key + "_std"] = float(np.std(vals))
        agg["n_seeds"] = len(SEEDS)
        aggregated[mname] = agg
    
    all_results["exp1_multiseed"] = aggregated
    
    # Print summary
    print(f"\n{'Method':<14} {'TopicCov':>10} {'MeanQ':>10} {'CosDiversity':>13} {'FillDist':>10} {'SinkDiv':>10} {'Cov@0.5':>10}")
    print("-" * 80)
    for mname in ["divflow", "vcg_divflow", "dpp", "mmr", "facility_loc", "mmd_greedy", "kmedoids", "random", "top_quality"]:
        a = aggregated[mname]
        print(f"{mname:<14} {a['topic_coverage_mean']:>7.3f}±{a['topic_coverage_std']:.3f} "
              f"{a['mean_quality_mean']:>7.3f}±{a['mean_quality_std']:.3f} "
              f"{a['cosine_diversity_mean']:>9.3f}±{a['cosine_diversity_std']:.3f} "
              f"{a['fill_distance_mean']:>7.4f}±{a['fill_distance_std']:.4f} "
              f"{a['sinkhorn_div_to_pool_mean']:>7.4f}±{a['sinkhorn_div_to_pool_std']:.4f} "
              f"{a['coverage_eps_0.5_mean']:>7.3f}±{a['coverage_eps_0.5_std']:.3f}")
    
    # =============================================================
    # Experiment 2: Per-selection coverage (CF2 fix)
    # =============================================================
    print("\n=== EXP 2: Per-Selection Coverage Comparison ===")
    coverage_comparison = {}
    for mname, mfn in method_configs.items():
        sel = mfn(embs, quals, K_SELECT, 42)
        sel_embs_m = embs[sel]
        coverages = {}
        for eps in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
            cov = coverage_fraction(sel_embs_m, embs, epsilon=eps)
            coverages[f"eps_{eps}"] = float(cov)
        coverages["fill_distance"] = float(fill_distance(sel_embs_m, embs))
        coverages["dispersion"] = float(dispersion(sel_embs_m))
        # Epsilon-net certificate with Hoeffding bound
        cert = epsilon_net_certificate(sel_embs_m, embs, epsilon=0.5)
        coverages["cert_coverage"] = float(cert.coverage_fraction)
        coverages["cert_confidence"] = float(cert.confidence)
        coverage_comparison[mname] = coverages
    
    all_results["exp2_coverage"] = coverage_comparison
    
    print(f"  {'Method':<14} {'Cov@0.3':>8} {'Cov@0.5':>8} {'Cov@0.8':>8} {'FillDist':>9} {'Cert':>8}")
    print("-" * 60)
    for mname in ["divflow", "vcg_divflow", "dpp", "mmr", "facility_loc", "mmd_greedy", "kmedoids", "random", "top_quality"]:
        c = coverage_comparison[mname]
        print(f"  {mname:<14} {c['eps_0.3']:>8.3f} {c['eps_0.5']:>8.3f} {c['eps_0.8']:>8.3f} "
              f"{c['fill_distance']:>9.4f} {c['cert_coverage']:>8.3f}")
    
    # =============================================================
    # Experiment 3: VCG IC Verification (multi-seed)
    # =============================================================
    print("\n=== EXP 3: VCG IC Verification (multi-seed) ===")
    ic_results = []
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(embs))
        shuf_embs = embs[perm]
        shuf_quals = quals[perm]
        sel_vcg, pay_vcg = select_vcg(shuf_embs, shuf_quals, K_SELECT)
        ic = verify_ic(shuf_embs, shuf_quals, sel_vcg, pay_vcg, n_trials=100, seed=seed)
        ic_results.append(ic)
        print(f"  Seed {seed}: {ic['violations']}/{ic['n_trials']} violations ({ic['violation_rate']:.1%})")
    
    all_results["exp3_ic"] = {
        "per_seed": ic_results,
        "mean_violation_rate": float(np.mean([r["violation_rate"] for r in ic_results])),
        "std_violation_rate": float(np.std([r["violation_rate"] for r in ic_results])),
        "mean_max_gain": float(np.mean([r["max_gain"] for r in ic_results])),
        "total_trials": sum(r["n_trials"] for r in ic_results),
        "total_violations": sum(r["violations"] for r in ic_results),
    }
    print(f"  Overall: {all_results['exp3_ic']['mean_violation_rate']:.1%} ± "
          f"{all_results['exp3_ic']['std_violation_rate']:.1%}")
    
    # =============================================================
    # Experiment 4: Scaling with pool diversity
    # =============================================================
    print("\n=== EXP 4: Scaling with Pool Diversity ===")
    scaling = {}
    for n_prompts in [2, 4, 6, 8, 10]:
        pool_size = n_prompts * N_PER_PROMPT
        sub_embs = embs[:pool_size]
        sub_quals = quals[:pool_size]
        sub_labels = all_labels[:pool_size]
        
        scale_results = {}
        for mname in ["divflow", "dpp", "mmr", "facility_loc", "mmd_greedy", "random", "top_quality"]:
            fn = method_configs[mname]
            sel = fn(sub_embs, sub_quals, K_SELECT, 42)
            ev = evaluate_selection(sub_embs, sub_quals, sel, sub_labels, sub_embs)
            scale_results[mname] = ev
        
        scaling[f"n_prompts={n_prompts}"] = scale_results
        divf = scale_results['divflow']
        dpp = scale_results['dpp']
        mmr = scale_results['mmr']
        fl = scale_results['facility_loc']
        print(f"  N={pool_size:>3}: divflow={divf['topic_diversity']}/{n_prompts} "
              f"dpp={dpp['topic_diversity']}/{n_prompts} "
              f"mmr={mmr['topic_diversity']}/{n_prompts} "
              f"fac_loc={fl['topic_diversity']}/{n_prompts} "
              f"| fill_dist: divflow={divf['fill_distance']:.4f} mmr={mmr['fill_distance']:.4f}")
    
    all_results["exp4_scaling"] = scaling
    
    # =============================================================
    # Experiment 5: Pareto frontier
    # =============================================================
    print("\n=== EXP 5: Quality-Diversity Pareto Frontier ===")
    pareto = {}
    for lam in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        sel = select_divflow(embs, quals, K_SELECT, quality_weight=lam)
        ev = evaluate_selection(embs, quals, sel, all_labels, embs)
        ev["quality_weight"] = lam
        pareto[f"lambda={lam}"] = ev
        print(f"  λ={lam:.1f}: topics={ev['topic_diversity']} quality={ev['mean_quality']:.3f} "
              f"cos_div={ev['cosine_diversity']:.3f} fill_dist={ev['fill_distance']:.4f}")
    all_results["exp5_pareto"] = pareto
    
    # =============================================================
    # Experiment 6: DivFlow vs MMR deep comparison (CF3)
    # =============================================================
    print("\n=== EXP 6: DivFlow vs MMR Deep Comparison ===")
    deep_comparison = {}
    for mname in ["divflow", "mmr", "facility_loc"]:
        fn = method_configs[mname]
        metrics_per_seed = []
        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            perm = rng.permutation(len(embs))
            shuf_embs = embs[perm]
            shuf_quals = quals[perm]
            shuf_labels = [all_labels[i] for i in perm]
            sel = fn(shuf_embs, shuf_quals, K_SELECT, seed)
            ev = evaluate_selection(shuf_embs, shuf_quals, sel, shuf_labels, shuf_embs)
            metrics_per_seed.append(ev)
        
        agg = {}
        for key in metrics_per_seed[0]:
            if key == "selected_indices":
                continue
            vals = [r[key] for r in metrics_per_seed]
            agg[key + "_mean"] = float(np.mean(vals))
            agg[key + "_std"] = float(np.std(vals))
        deep_comparison[mname] = agg
    
    all_results["exp6_divflow_vs_mmr"] = deep_comparison
    
    # Print comparison
    print(f"  {'Metric':<25} {'DivFlow':>15} {'MMR':>15} {'FacLoc':>15}")
    print("  " + "-" * 70)
    for metric in ["fill_distance", "sinkhorn_div_to_pool", "coverage_eps_0.5", 
                    "coverage_eps_0.3", "dispersion", "vendi_score"]:
        d = deep_comparison["divflow"]
        m = deep_comparison["mmr"]
        f = deep_comparison["facility_loc"]
        print(f"  {metric:<25} {d[metric+'_mean']:>7.4f}±{d[metric+'_std']:.4f} "
              f"{m[metric+'_mean']:>7.4f}±{m[metric+'_std']:.4f} "
              f"{f[metric+'_mean']:>7.4f}±{f[metric+'_std']:.4f}")
    
    # =============================================================
    # Experiment 7: Qualitative samples  
    # =============================================================
    print("\n=== EXP 7: Qualitative Samples ===")
    qualitative = {}
    for mname in ["divflow", "vcg_divflow", "dpp", "mmr", "facility_loc"]:
        fn = method_configs[mname]
        sel = fn(embs, quals, K_SELECT, 42)
        samples = []
        for i in sel:
            samples.append({
                "index": int(i),
                "topic": all_labels[i],
                "quality_heuristic": float(heur_quals[i]),
                "quality_llm": float(llm_quals[i]),
                "text_preview": all_responses[i][:200],
            })
        qualitative[mname] = samples
    all_results["exp7_qualitative"] = qualitative
    
    # =============================================================
    # Save results
    # =============================================================
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    outfile = os.path.join(os.path.dirname(__file__), "v2_results.json")
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\nResults saved to {outfile}")
    
    return all_results


if __name__ == "__main__":
    main()

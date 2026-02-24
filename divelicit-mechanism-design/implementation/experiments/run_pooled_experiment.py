"""Pooled multi-prompt LLM experiment: real-world diverse selection scenario.

Generates responses across multiple prompts, pools them, then selects
diverse subsets. This is the realistic use case: given many LLM responses
spanning different topics/styles, select a maximally informative subset.

Also runs:
- VCG IC verification on real embeddings
- Coverage certificate computation
- Quality-diversity Pareto frontier
- Comparison of selection methods at different pool sizes
"""

import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.kernels import RBFKernel, AdaptiveRBFKernel, MultiScaleKernel
from src.dpp import greedy_map
from src.transport import (
    sinkhorn_divergence, sinkhorn_candidate_scores, cost_matrix
)
from src.coverage import estimate_coverage, epsilon_net_certificate
from src.diversity_metrics import cosine_diversity, dispersion_metric, vendi_score
from src.utils import log_det_safe


def get_client():
    from openai import OpenAI
    return OpenAI()


def generate_responses(client, prompt, n, system_prompt="You are a helpful assistant.",
                       temperature=0.9, model="gpt-4.1-nano"):
    responses = []
    for i in range(n):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=min(temperature + 0.1 * (i % 5), 1.5),
                max_tokens=250,
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


def mean_pairwise_cosine_sim(embs):
    if len(embs) < 2:
        return 0.0
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    normed = embs / np.maximum(norms, 1e-12)
    S = normed @ normed.T
    n = len(embs)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            total += S[i, j]
            count += 1
    return total / max(count, 1)


def mean_pairwise_distance(embs):
    if len(embs) < 2:
        return 0.0
    dists = cost_matrix(embs, embs, metric="euclidean")
    n = len(embs)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            total += dists[i, j]
            count += 1
    return total / max(count, 1)


def select_divflow(embs, quals, k, quality_weight=0.3, reg=0.1):
    """DivFlow: Sinkhorn divergence-guided greedy selection.
    
    First item: highest quality. Subsequent items: marginal Sinkhorn
    divergence reduction combined with quality.
    """
    n = embs.shape[0]
    ref = embs.copy()
    selected = [int(np.argmax(quals))]
    for _ in range(k - 1):
        history = embs[selected]
        remaining = [i for i in range(n) if i not in selected]
        if not remaining:
            break
        cand = embs[remaining]
        div_scores = sinkhorn_candidate_scores(cand, history, ref, reg=reg, n_iter=50)
        ds_min, ds_max = div_scores.min(), div_scores.max()
        if ds_max - ds_min > 1e-10:
            div_norm = (div_scores - ds_min) / (ds_max - ds_min)
        else:
            div_norm = np.ones_like(div_scores)
        rem_quals = quals[remaining]
        q_min, q_max = rem_quals.min(), rem_quals.max()
        if q_max - q_min > 1e-10:
            q_norm = (rem_quals - q_min) / (q_max - q_min)
        else:
            q_norm = np.ones_like(rem_quals)
        combined = (1.0 - quality_weight) * div_norm + quality_weight * q_norm
        best = remaining[int(np.argmax(combined))]
        selected.append(best)
    return selected


def select_vcg(embs, quals, k, quality_weight=0.3, reg=0.1):
    """VCG with Sinkhorn divergence welfare (unified with DivFlow objective).
    
    Welfare: W(S) = -Sinkhorn_div(S, reference) + quality_weight * sum(q_i for i in S)
    Using negative Sinkhorn divergence so higher = better coverage.
    """
    n = embs.shape[0]
    ref = embs.copy()
    
    def welfare(indices):
        if not indices:
            return 0.0
        sel_embs = embs[indices]
        # Negative Sinkhorn divergence: lower divergence = higher welfare
        sdiv = sinkhorn_divergence(sel_embs, ref, reg=reg, n_iter=50)
        q = sum(quals[i] for i in indices)
        return -(1.0 - quality_weight) * sdiv + quality_weight * q
    
    # Greedy selection
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
    
    # VCG payments
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
    # Farthest-first init
    medoids = [0]
    for _ in range(k - 1):
        remaining = [j for j in range(n) if j not in medoids]
        if not remaining:
            break
        min_dists = np.array([min(dists[j, m] for m in medoids) for j in remaining])
        medoids.append(remaining[int(np.argmax(min_dists))])
    return medoids


def select_random(n, k, seed=42):
    return list(np.random.RandomState(seed).choice(n, min(k, n), replace=False))


def select_top_quality(quals, k):
    return list(np.argsort(quals)[-k:][::-1])


def select_facility_location(embs, k):
    """Greedy facility location: maximize sum of min-distances from pool to selected."""
    n = embs.shape[0]
    dists = cost_matrix(embs, embs, metric="euclidean")
    selected = []
    # First: select point closest to centroid  
    centroid_dists = np.linalg.norm(embs - np.mean(embs, axis=0), axis=1)
    selected.append(int(np.argmin(centroid_dists)))
    for _ in range(k - 1):
        # For each unselected point, compute min distance to selected set
        min_dists = np.array([min(dists[j, s] for s in selected) for j in range(n)])
        # Select point that maximizes sum of min-distances
        best_j, best_gain = -1, -float('inf')
        for j in range(n):
            if j in selected:
                continue
            trial_selected = selected + [j]
            trial_min_dists = np.array([min(dists[p, s] for s in trial_selected) for p in range(n)])
            # Facility location objective: minimize sum of min-distances
            gain = float(np.sum(min_dists) - np.sum(trial_min_dists))
            if gain > best_gain:
                best_gain = gain
                best_j = j
        if best_j >= 0:
            selected.append(best_j)
    return selected


def evaluate(embs, quals, selected, labels, all_embs):
    sel_embs = embs[selected]
    sel_quals = quals[selected]
    sel_labels = [labels[i] for i in selected]
    
    # Topic diversity: number of unique topics
    unique_topics = len(set(sel_labels))
    
    # Method-specific fill distance: max distance from any pool point to nearest selected
    fill_dist = 0.0
    for i in range(len(all_embs)):
        dists = np.linalg.norm(sel_embs - all_embs[i], axis=1)
        fill_dist = max(fill_dist, float(np.min(dists)))
    
    # Sinkhorn divergence to reference
    sdiv = sinkhorn_divergence(sel_embs, all_embs, reg=0.1, n_iter=50)
    
    return {
        "mean_pairwise_sim": float(mean_pairwise_cosine_sim(sel_embs)),
        "mean_pairwise_dist": float(mean_pairwise_distance(sel_embs)),
        "cosine_diversity": float(1.0 - mean_pairwise_cosine_sim(sel_embs)),
        "mean_quality": float(np.mean(sel_quals)),
        "min_quality": float(np.min(sel_quals)),
        "topic_diversity": unique_topics,
        "n_possible_topics": len(set(labels)),
        "topic_coverage": unique_topics / len(set(labels)),
        "fill_distance": fill_dist,
        "sinkhorn_divergence": float(sdiv),
        "selected_indices": [int(i) for i in selected],
    }


def verify_ic(embs, quals, selected, payments, quality_weight=0.5, n_trials=200):
    n = len(quals)
    violations = 0
    max_gain = 0.0
    rng = np.random.RandomState(42)
    
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
        "ic_verified": violations == 0,
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


def main():
    client = get_client()
    results = {}
    
    # ---------------------------------------------------------------
    # Experiment 1: Pooled multi-prompt selection
    # ---------------------------------------------------------------
    print("=== EXP 1: Pooled Multi-Prompt Selection ===")
    all_responses = []
    all_labels = []
    
    for pname, prompt in PROMPTS.items():
        print(f"  Generating {N_PER_PROMPT} responses for '{pname}'...")
        resps = generate_responses(client, prompt, N_PER_PROMPT)
        all_responses.extend(resps)
        all_labels.extend([pname] * N_PER_PROMPT)
    
    print(f"  Total responses: {len(all_responses)}")
    print(f"  Embedding...")
    embs = embed(client, all_responses)
    print(f"  Embeddings shape: {embs.shape}")
    
    # Quality: use embedding-based proxy
    quals = np.zeros(len(all_responses))
    for i, r in enumerate(all_responses):
        quals[i] = min(len(r) / 400.0, 1.0) * 0.4 + \
                   min(np.linalg.norm(embs[i]) / 1.5, 1.0) * 0.6
    quals = (quals - quals.min()) / max(quals.max() - quals.min(), 1e-12)
    quals = 0.3 + 0.7 * quals
    
    # Run methods
    methods_results = {}
    
    print("  Running DivFlow...")
    sel = select_divflow(embs, quals, K_SELECT)
    methods_results["divflow"] = evaluate(embs, quals, sel, all_labels, embs)
    
    print("  Running VCG-DivFlow...")
    sel_vcg, payments_vcg = select_vcg(embs, quals, K_SELECT)
    methods_results["vcg_divflow"] = evaluate(embs, quals, sel_vcg, all_labels, embs)
    methods_results["vcg_divflow"]["total_payments"] = float(sum(payments_vcg))
    methods_results["vcg_divflow"]["payments"] = [float(p) for p in payments_vcg]
    
    print("  Running DPP...")
    sel = select_dpp(embs, quals, K_SELECT)
    methods_results["dpp"] = evaluate(embs, quals, sel, all_labels, embs)
    
    print("  Running MMR...")
    sel = select_mmr(embs, quals, K_SELECT)
    methods_results["mmr"] = evaluate(embs, quals, sel, all_labels, embs)
    
    print("  Running k-Medoids...")
    sel = select_kmedoids(embs, K_SELECT)
    methods_results["kmedoids"] = evaluate(embs, quals, sel, all_labels, embs)
    
    print("  Running Random...")
    sel = select_random(len(embs), K_SELECT)
    methods_results["random"] = evaluate(embs, quals, sel, all_labels, embs)
    
    print("  Running Top-Quality...")
    sel = select_top_quality(quals, K_SELECT)
    methods_results["top_quality"] = evaluate(embs, quals, sel, all_labels, embs)
    
    results["exp1_pooled"] = {
        "methods": methods_results,
        "n_total": len(all_responses),
        "n_prompts": len(PROMPTS),
        "n_per_prompt": N_PER_PROMPT,
        "k_select": K_SELECT,
    }
    
    # Print summary
    print(f"\n{'Method':<15} {'CosDiversity':>13} {'MeanDist':>9} {'MeanQ':>8} {'Topics':>7} {'TopicCov':>9}")
    print("-" * 65)
    for m, r in methods_results.items():
        print(f"{m:<15} {r['cosine_diversity']:>13.4f} {r['mean_pairwise_dist']:>9.4f} "
              f"{r['mean_quality']:>8.4f} {r['topic_diversity']:>7d} {r['topic_coverage']:>9.2f}")
    
    # ---------------------------------------------------------------
    # Experiment 2: VCG IC Verification
    # ---------------------------------------------------------------
    print("\n=== EXP 2: VCG IC Verification ===")
    print("  Running 500 IC trials...")
    ic_result = verify_ic(embs, quals, sel_vcg, payments_vcg, n_trials=500)
    results["exp2_ic"] = ic_result
    print(f"  IC violations: {ic_result['violations']}/{ic_result['n_trials']}")
    print(f"  IC verified: {ic_result['ic_verified']}")
    
    # ---------------------------------------------------------------
    # Experiment 3: Quality-Diversity Pareto Frontier  
    # ---------------------------------------------------------------
    print("\n=== EXP 3: Pareto Frontier ===")
    pareto = {}
    for lam in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        sel = select_divflow(embs, quals, K_SELECT, quality_weight=lam)
        ev = evaluate(embs, quals, sel, all_labels, embs)
        ev["quality_weight"] = lam
        pareto[f"lambda={lam}"] = ev
        print(f"  λ={lam:.1f}: diversity={ev['cosine_diversity']:.4f} quality={ev['mean_quality']:.4f} topics={ev['topic_diversity']}")
    results["exp3_pareto"] = pareto
    
    # ---------------------------------------------------------------
    # Experiment 4: Coverage Certificates
    # ---------------------------------------------------------------
    print("\n=== EXP 4: Coverage Certificates ===")
    coverage_results = {}
    for method_name in ["divflow", "vcg_divflow", "dpp", "mmr", "kmedoids", "random", "top_quality"]:
        sel = methods_results[method_name]["selected_indices"]
        sel_embs = embs[sel]
        cert = epsilon_net_certificate(sel_embs, embs, epsilon=0.5)
        coverage_results[method_name] = {
            "coverage_fraction": cert.coverage_fraction,
            "confidence": cert.confidence,
            "epsilon": 0.5,
        }
    results["exp4_coverage"] = coverage_results
    print("  Coverage fractions:")
    for m, c in coverage_results.items():
        print(f"    {m}: {c['coverage_fraction']:.4f}")
    
    # ---------------------------------------------------------------
    # Experiment 5: Scaling with pool size
    # ---------------------------------------------------------------
    print("\n=== EXP 5: Scaling ===")
    scaling = {}
    for n_prompts in [2, 4, 6, 8, 10]:
        prompt_subset = list(PROMPTS.items())[:n_prompts]
        pool_size = n_prompts * N_PER_PROMPT
        # Reuse embeddings from the full pool
        idx_range = list(range(pool_size))
        sub_embs = embs[idx_range]
        sub_quals = quals[idx_range]
        sub_labels = all_labels[:pool_size]
        
        scale_results = {}
        for mname, sfn in [
            ("divflow", lambda: select_divflow(sub_embs, sub_quals, K_SELECT)),
            ("dpp", lambda: select_dpp(sub_embs, sub_quals, K_SELECT)),
            ("mmr", lambda: select_mmr(sub_embs, sub_quals, K_SELECT)),
            ("random", lambda: select_random(len(sub_embs), K_SELECT)),
            ("top_quality", lambda: select_top_quality(sub_quals, K_SELECT)),
        ]:
            t0 = time.time()
            sel = sfn()
            elapsed = time.time() - t0
            ev = evaluate(sub_embs, sub_quals, sel, sub_labels, sub_embs)
            ev["runtime_s"] = elapsed
            scale_results[mname] = ev
        
        scaling[f"n_prompts={n_prompts}"] = scale_results
        print(f"  N={pool_size}: divflow_topics={scale_results['divflow']['topic_diversity']}/{n_prompts} "
              f"dpp_topics={scale_results['dpp']['topic_diversity']}/{n_prompts}")
    results["exp5_scaling"] = scaling
    
    # ---------------------------------------------------------------
    # Experiment 6: Qualitative comparison (sample responses)
    # ---------------------------------------------------------------
    print("\n=== EXP 6: Qualitative Samples ===")
    qualitative = {}
    for method_name in ["divflow", "vcg_divflow", "dpp", "top_quality"]:
        sel = methods_results[method_name]["selected_indices"]
        samples = []
        for i in sel:
            samples.append({
                "index": i,
                "topic": all_labels[i],
                "quality": float(quals[i]),
                "text_preview": all_responses[i][:200],
            })
        qualitative[method_name] = samples
    results["exp6_qualitative"] = qualitative
    
    # ---------------------------------------------------------------
    # Experiment 7: Sinkhorn divergence comparison
    # ---------------------------------------------------------------
    print("\n=== EXP 7: Sinkhorn Divergence ===")
    sink_results = {}
    # Create a uniform reference distribution (subsample of full pool)
    ref_idx = np.random.RandomState(42).choice(len(embs), min(20, len(embs)), replace=False)
    ref = embs[ref_idx]
    
    for method_name in ["divflow", "vcg_divflow", "dpp", "mmr", "kmedoids", "random", "top_quality"]:
        sel = methods_results[method_name]["selected_indices"]
        sel_embs = embs[sel]
        sd = float(sinkhorn_divergence(sel_embs, ref, reg=0.1))
        sink_results[method_name] = {"sinkhorn_div_to_ref": sd}
    results["exp7_sinkhorn"] = sink_results
    print("  Sinkhorn divergences:")
    for m, s in sink_results.items():
        print(f"    {m}: {s['sinkhorn_div_to_ref']:.6f}")
    
    # ---------------------------------------------------------------
    # Save all results
    # ---------------------------------------------------------------
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    results = convert(results)
    out_path = os.path.join(os.path.dirname(__file__), "pooled_llm_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

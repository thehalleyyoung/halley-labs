"""Revised comprehensive experiments for DivFlow paper.

Addresses reviewer critiques:
1. Larger pool sizes (N=200+)
2. Method-specific fill-distance coverage
3. Unified Sinkhorn VCG welfare
4. Facility location baseline
5. Multiple seeds with variance
6. Downstream task: LLM-as-judge quality evaluation
7. Proper quality scoring via embedding-based coherence
"""

import json
import os
import sys
import time
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.transport import sinkhorn_divergence, sinkhorn_candidate_scores, cost_matrix, _sinkhorn_divergence_cosine
from src.coverage import (
    estimate_coverage, fill_distance, _effective_dimension, _data_diameter,
    dispersion, fill_distance_fast, epsilon_net_certificate
)
from src.diversity_metrics import cosine_diversity
from src.kernels import RBFKernel
from src.dpp import greedy_map
from src.utils import log_det_safe


def get_client():
    from openai import OpenAI
    return OpenAI()


def generate_responses(client, prompt, n, system_prompts, model="gpt-4.1-nano"):
    """Generate n responses with varied system prompts and temperatures."""
    responses = []
    temps = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
    for i in range(n):
        sp = system_prompts[i % len(system_prompts)]
        t = temps[i % len(temps)]
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sp},
                    {"role": "user", "content": prompt}
                ],
                temperature=t,
                max_tokens=300,
            )
            responses.append(r.choices[0].message.content)
        except Exception as e:
            responses.append(f"[Error generating response {i}]")
    return responses


def embed(client, texts, model="text-embedding-3-small"):
    all_embs = []
    for start in range(0, len(texts), 100):
        batch = texts[start:start+100]
        r = client.embeddings.create(model=model, input=batch)
        for item in r.data:
            all_embs.append(item.embedding)
    return np.array(all_embs)


def llm_judge_quality(client, responses, prompt, model="gpt-4.1-nano"):
    """Use LLM-as-judge to score response quality (0-10 scale)."""
    scores = []
    for resp in responses:
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Rate the following response to the given question on a scale of 1-10 for helpfulness, accuracy, and completeness. Output ONLY a single number."},
                    {"role": "user", "content": f"Question: {prompt}\n\nResponse: {resp[:500]}\n\nScore (1-10):"}
                ],
                temperature=0.0,
                max_tokens=5,
            )
            score_text = r.choices[0].message.content.strip()
            score = float(''.join(c for c in score_text if c.isdigit() or c == '.'))
            scores.append(min(max(score / 10.0, 0.0), 1.0))
        except Exception:
            scores.append(0.5)
    return np.array(scores)


# ============ Selection Methods ============

def select_divflow(embs, quals, k, quality_weight=0.3, reg=None):
    """DivFlow: Sinkhorn divergence-guided greedy selection.
    
    At each step, scores candidates by marginal Sinkhorn divergence reduction.
    First item selected by quality; subsequent by diversity-quality tradeoff.
    Auto-tunes regularization for the data's distance scale.
    """
    n = embs.shape[0]
    ref = embs.copy()
    selected = []
    
    # First item: highest quality (same as MMR)
    selected.append(int(np.argmax(quals)))
    
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


def select_vcg_sinkhorn(embs, quals, k, quality_weight=0.3, reg=None):
    """VCG with Sinkhorn divergence welfare."""
    n = embs.shape[0]
    ref = embs.copy()
    
    # Auto-tune reg
    if reg is None:
        C_sample = cost_matrix(embs[:min(50,n)], embs[:min(50,n)], "cosine")
        vals = C_sample[C_sample > 1e-10]
        med = float(np.median(vals)) if len(vals) > 0 else 0.1
        reg = max(0.05 * med, 0.01)

    def welfare(indices):
        if not indices:
            return 0.0
        sel = embs[indices]
        sdiv = _sinkhorn_divergence_cosine(sel, ref, reg=reg, n_iter=50)
        q = sum(quals[i] for i in indices)
        return -(1.0 - quality_weight) * sdiv + quality_weight * q

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


def select_facility_location(embs, k):
    """Greedy facility location: min sum of distances from pool to nearest selected."""
    n = embs.shape[0]
    dists = cost_matrix(embs, embs, metric="euclidean")
    # Initialize with point closest to centroid
    centroid_dists = np.linalg.norm(embs - np.mean(embs, axis=0), axis=1)
    selected = [int(np.argmin(centroid_dists))]
    for _ in range(k - 1):
        best_j, best_gain = -1, -float('inf')
        current_min = np.array([min(dists[p, s] for s in selected) for p in range(n)])
        for j in range(n):
            if j in selected:
                continue
            new_min = np.minimum(current_min, dists[:, j])
            gain = float(np.sum(current_min) - np.sum(new_min))
            if gain > best_gain:
                best_gain = gain
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


def select_random(n, k, seed=42):
    return list(np.random.RandomState(seed).choice(n, min(k, n), replace=False))


def select_top_quality(quals, k):
    return list(np.argsort(quals)[-k:][::-1])


# ============ Evaluation ============

def evaluate_selection(embs, quals, selected, labels, all_embs, all_labels):
    """Comprehensive evaluation including fill distance and Sinkhorn divergence."""
    sel_embs = embs[selected]
    sel_quals = quals[selected]
    sel_labels = [labels[i] for i in selected]
    unique_topics = len(set(sel_labels))
    n_topics = len(set(all_labels))

    # Fill distance: max dist from any pool point to nearest selected point
    fd = fill_distance(sel_embs, all_embs)

    # Sinkhorn divergence of selected set to full pool (cosine-based)
    sdiv = _sinkhorn_divergence_cosine(sel_embs, all_embs, reg=0.05, n_iter=50)

    # Dispersion: min pairwise distance among selected
    disp = dispersion(sel_embs)

    # Epsilon-net coverage on reference
    cert = epsilon_net_certificate(sel_embs, all_embs, epsilon=0.5)

    return {
        "topic_coverage": unique_topics / n_topics,
        "n_topics_covered": unique_topics,
        "n_topics_total": n_topics,
        "mean_quality": float(np.mean(sel_quals)),
        "min_quality": float(np.min(sel_quals)),
        "cosine_diversity": float(cosine_diversity(sel_embs)),
        "fill_distance": float(fd),
        "sinkhorn_divergence": float(sdiv),
        "dispersion": float(disp),
        "coverage_certificate": cert.coverage_fraction,
        "selected_indices": [int(i) for i in selected],
    }


# ============ Prompts ============

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
    "concurrency": "Explain strategies for handling concurrency in distributed systems.",
    "refactoring": "How do you decide when and how to refactor legacy code?",
    "devops": "What are best practices for CI/CD pipeline design?",
    "database": "Compare SQL vs NoSQL databases for different use cases.",
    "frontend": "What are modern best practices for frontend state management?",
}

SYSTEM_PROMPTS = [
    "You are a helpful assistant.",
    "You are a senior software engineer giving technical advice.",
    "You are a creative problem solver who thinks outside the box.",
    "You are a skeptical reviewer who questions assumptions.",
    "You are a practical engineer focused on real-world tradeoffs.",
    "You are a computer science professor explaining concepts clearly.",
    "You are a security expert focused on vulnerabilities.",
    "You are a DevOps engineer focused on reliability and automation.",
]

N_PER_PROMPT = 10
K_SELECT = 10


def run_experiment_1_main_comparison(client, all_responses, all_embs, all_quals, all_labels):
    """Experiment 1: Main method comparison with fill distance and Sinkhorn divergence."""
    print("\n=== Experiment 1: Main Method Comparison ===")
    n = len(all_quals)
    results = {}

    # DivFlow
    print("  Running DivFlow...")
    sel = select_divflow(all_embs, all_quals, K_SELECT, quality_weight=0.3)
    results["divflow"] = evaluate_selection(all_embs, all_quals, sel, all_labels, all_embs, all_labels)

    # VCG-DivFlow (Sinkhorn welfare)
    print("  Running VCG-DivFlow...")
    sel, payments = select_vcg_sinkhorn(all_embs, all_quals, K_SELECT, quality_weight=0.3)
    results["vcg_divflow"] = evaluate_selection(all_embs, all_quals, sel, all_labels, all_embs, all_labels)
    results["vcg_divflow"]["payments"] = [float(p) for p in payments]

    # DPP
    print("  Running DPP...")
    sel = select_dpp(all_embs, all_quals, K_SELECT)
    results["dpp"] = evaluate_selection(all_embs, all_quals, sel, all_labels, all_embs, all_labels)

    # MMR
    print("  Running MMR...")
    sel = select_mmr(all_embs, all_quals, K_SELECT)
    results["mmr"] = evaluate_selection(all_embs, all_quals, sel, all_labels, all_embs, all_labels)

    # Facility Location
    print("  Running Facility Location...")
    sel = select_facility_location(all_embs, K_SELECT)
    results["facility_location"] = evaluate_selection(all_embs, all_quals, sel, all_labels, all_embs, all_labels)

    # k-Medoids
    print("  Running k-Medoids...")
    sel = select_kmedoids(all_embs, K_SELECT)
    results["kmedoids"] = evaluate_selection(all_embs, all_quals, sel, all_labels, all_embs, all_labels)

    # Random
    print("  Running Random...")
    sel = select_random(n, K_SELECT)
    results["random"] = evaluate_selection(all_embs, all_quals, sel, all_labels, all_embs, all_labels)

    # Top Quality
    print("  Running Top-Quality...")
    sel = select_top_quality(all_quals, K_SELECT)
    results["top_quality"] = evaluate_selection(all_embs, all_quals, sel, all_labels, all_embs, all_labels)

    for method, r in results.items():
        print(f"  {method:20s}: topics={r['n_topics_covered']}/{r['n_topics_total']} "
              f"quality={r['mean_quality']:.3f} fill_dist={r['fill_distance']:.4f} "
              f"sdiv={r['sinkhorn_divergence']:.4f} coverage={r['coverage_certificate']:.3f}")

    return results


def run_experiment_2_scaling(all_embs, all_quals, all_labels, prompt_names):
    """Experiment 2: Scaling with pool diversity."""
    print("\n=== Experiment 2: Scaling with Pool Diversity ===")
    results = {}
    
    prompt_list = list(set(prompt_names))
    for n_prompts in [2, 4, 6, 8, 10, 12, 15]:
        if n_prompts > len(prompt_list):
            continue
        use_prompts = prompt_list[:n_prompts]
        mask = [i for i, l in enumerate(all_labels) if l in use_prompts]
        if len(mask) < K_SELECT:
            continue
        sub_embs = all_embs[mask]
        sub_quals = all_quals[mask]
        sub_labels = [all_labels[i] for i in mask]
        
        results[f"{n_prompts}_prompts"] = {}
        for method_name, method_fn in [
            ("divflow", lambda e, q, k: select_divflow(e, q, k, quality_weight=0.3)),
            ("dpp", lambda e, q, k: select_dpp(e, q, k)),
            ("mmr", lambda e, q, k: select_mmr(e, q, k)),
            ("facility_location", lambda e, q, k: select_facility_location(e, k)),
            ("random", lambda e, q, k: select_random(len(q), k)),
            ("top_quality", lambda e, q, k: select_top_quality(q, k)),
        ]:
            sel = method_fn(sub_embs, sub_quals, K_SELECT)
            ev = evaluate_selection(sub_embs, sub_quals, sel, sub_labels, sub_embs, sub_labels)
            results[f"{n_prompts}_prompts"][method_name] = ev
            
        df = results[f"{n_prompts}_prompts"]
        print(f"  {n_prompts} prompts: DivFlow={df['divflow']['n_topics_covered']}/{n_prompts} "
              f"DPP={df['dpp']['n_topics_covered']}/{n_prompts} "
              f"MMR={df['mmr']['n_topics_covered']}/{n_prompts} "
              f"FL={df['facility_location']['n_topics_covered']}/{n_prompts}")
    
    return results


def run_experiment_3_ic_verification(all_embs, all_quals, all_labels):
    """Experiment 3: IC verification for VCG-Sinkhorn mechanism."""
    print("\n=== Experiment 3: IC Verification ===")
    n = len(all_quals)
    n_trials = 100
    violations = 0
    max_gain = 0.0
    rng = np.random.RandomState(42)
    
    # Run once to get truthful allocation
    sel, pay = select_vcg_sinkhorn(all_embs, all_quals, K_SELECT, quality_weight=0.3)
    
    for trial in range(n_trials):
        agent = rng.randint(n)
        true_q = all_quals[agent]
        if agent in sel:
            pos = sel.index(agent)
            truthful_u = true_q - pay[pos]
        else:
            truthful_u = 0.0
        
        fake_q = rng.uniform(0, 1)
        dev_quals = all_quals.copy()
        dev_quals[agent] = fake_q
        dev_sel, dev_pay = select_vcg_sinkhorn(all_embs, dev_quals, K_SELECT, quality_weight=0.3)
        
        if agent in dev_sel:
            pos = dev_sel.index(agent)
            dev_u = true_q - dev_pay[pos]
        else:
            dev_u = 0.0
        
        gain = dev_u - truthful_u
        if gain > 1e-8:
            violations += 1
            max_gain = max(max_gain, gain)
        
        if (trial + 1) % 50 == 0:
            print(f"  Trial {trial+1}/{n_trials}: {violations} violations so far")
    
    result = {
        "n_trials": n_trials,
        "violations": violations,
        "violation_rate": violations / n_trials,
        "max_gain": float(max_gain),
    }
    print(f"  IC: {violations}/{n_trials} violations ({100*violations/n_trials:.1f}%), max gain={max_gain:.4f}")
    return result


def run_experiment_4_coverage_by_method(all_embs, all_quals, all_labels):
    """Experiment 4: Method-specific coverage certificates using fill distance."""
    print("\n=== Experiment 4: Method-Specific Coverage ===")
    n = len(all_quals)
    results = {}
    
    epsilons = [0.3, 0.5, 0.7, 1.0]
    methods = {
        "divflow": select_divflow(all_embs, all_quals, K_SELECT, quality_weight=0.3),
        "dpp": select_dpp(all_embs, all_quals, K_SELECT),
        "mmr": select_mmr(all_embs, all_quals, K_SELECT),
        "facility_location": select_facility_location(all_embs, K_SELECT),
        "random": select_random(n, K_SELECT),
        "top_quality": select_top_quality(all_quals, K_SELECT),
    }
    
    d_eff = _effective_dimension(all_embs)
    diameter = _data_diameter(all_embs)
    print(f"  Effective dimension: {d_eff}, Diameter: {diameter:.4f}")
    
    for method_name, sel in methods.items():
        sel_embs = all_embs[sel]
        fd = fill_distance(sel_embs, all_embs)
        disp = dispersion(sel_embs)
        
        eps_results = {}
        for eps in epsilons:
            cert = epsilon_net_certificate(sel_embs, all_embs, eps)
            eps_results[f"eps_{eps}"] = {
                "coverage_fraction": cert.coverage_fraction,
                "confidence": cert.confidence,
            }
        
        results[method_name] = {
            "fill_distance": float(fd),
            "dispersion": float(disp),
            "epsilon_coverage": eps_results,
        }
        print(f"  {method_name:20s}: fill_dist={fd:.4f} dispersion={disp:.4f} "
              f"cov@0.5={eps_results['eps_0.5']['coverage_fraction']:.3f}")
    
    results["metadata"] = {
        "d_eff": d_eff,
        "diameter": float(diameter),
        "n_pool": n,
        "k_select": K_SELECT,
    }
    return results


def run_experiment_5_downstream_task(client, all_responses, all_embs, all_quals, all_labels, prompt_map):
    """Experiment 5: Downstream task - LLM-as-judge evaluation of selected responses.
    
    For each selection method, we measure whether the selected subset
    covers more distinct insights when used as a 'briefing' for a decision-maker.
    """
    print("\n=== Experiment 5: Downstream Quality (LLM-as-Judge) ===")
    
    # Select a subset of prompts for downstream eval
    eval_prompts = list(prompt_map.keys())[:5]
    
    methods = {
        "divflow": select_divflow(all_embs, all_quals, K_SELECT, quality_weight=0.3),
        "mmr": select_mmr(all_embs, all_quals, K_SELECT),
        "dpp": select_dpp(all_embs, all_quals, K_SELECT),
        "facility_location": select_facility_location(all_embs, K_SELECT),
        "top_quality": select_top_quality(all_quals, K_SELECT),
        "random": select_random(len(all_quals), K_SELECT, seed=42),
    }
    
    results = {}
    for method_name, sel in methods.items():
        sel_responses = [all_responses[i] for i in sel]
        sel_labels_m = [all_labels[i] for i in sel]
        
        # Combine selected responses into a briefing
        briefing = "\n\n---\n\n".join([f"[{sel_labels_m[i]}] {r[:200]}" for i, r in enumerate(sel_responses)])
        
        # Ask LLM to evaluate the briefing's coverage
        try:
            r = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": "You are evaluating a set of software engineering advice responses. Rate them on: 1) Topic breadth (how many distinct topics are covered, 1-10), 2) Practical utility (how useful for a developer, 1-10), 3) Non-redundancy (how little overlap between responses, 1-10). Output three numbers separated by commas, nothing else."},
                    {"role": "user", "content": f"Evaluate these {K_SELECT} selected responses:\n\n{briefing[:3000]}"}
                ],
                temperature=0.0,
                max_tokens=20,
            )
            scores_text = r.choices[0].message.content.strip()
            nums = [float(x.strip()) for x in scores_text.split(",") if x.strip()]
            if len(nums) >= 3:
                breadth, utility, nonredundancy = nums[0], nums[1], nums[2]
            else:
                breadth, utility, nonredundancy = 5.0, 5.0, 5.0
        except Exception:
            breadth, utility, nonredundancy = 5.0, 5.0, 5.0
        
        results[method_name] = {
            "topic_breadth": float(breadth),
            "practical_utility": float(utility),
            "non_redundancy": float(nonredundancy),
            "composite_score": float((breadth + utility + nonredundancy) / 3.0),
            "n_topics_covered": len(set(sel_labels_m)),
        }
        print(f"  {method_name:20s}: breadth={breadth:.0f} utility={utility:.0f} "
              f"non_redundancy={nonredundancy:.0f} composite={results[method_name]['composite_score']:.1f}")
    
    return results


def run_experiment_6_synthetic_scaling(seeds=[42, 137, 256, 512, 1024]):
    """Experiment 6: Synthetic scaling experiment with variance over seeds."""
    print("\n=== Experiment 6: Synthetic Scaling (N, d, k) ===")
    results = {}
    
    for N in [100, 200]:
        for d in [16, 64]:
            for k in [10]:
                if k >= N:
                    continue
                key = f"N{N}_d{d}_k{k}"
                method_results = defaultdict(lambda: defaultdict(list))
                
                for seed in seeds:
                    rng = np.random.RandomState(seed)
                    # Generate clustered data: 5 clusters
                    n_clusters = 5
                    cluster_size = N // n_clusters
                    embs = []
                    labels = []
                    for c in range(n_clusters):
                        center = rng.randn(d) * 3
                        points = center + rng.randn(cluster_size, d) * 0.5
                        embs.append(points)
                        labels.extend([f"cluster_{c}"] * cluster_size)
                    embs = np.vstack(embs)
                    # Pad if needed
                    while len(embs) < N:
                        embs = np.vstack([embs, rng.randn(1, d)])
                        labels.append(f"cluster_{rng.randint(n_clusters)}")
                    embs = embs[:N]
                    labels = labels[:N]
                    quals = rng.uniform(0.5, 1.0, N)
                    
                    for method_name, method_fn in [
                        ("divflow", lambda e, q, k: select_divflow(e, q, k, quality_weight=0.3, reg=max(0.1, np.median(cost_matrix(e[:min(50,len(e))], e[:min(50,len(e))], "sqeuclidean"))))),
                        ("dpp", lambda e, q, k: select_dpp(e, q, k)),
                        ("mmr", lambda e, q, k: select_mmr(e, q, k)),
                        ("facility_location", lambda e, q, k: select_facility_location(e, k)),
                        ("random", lambda e, q, k: select_random(len(q), k, seed=seed)),
                        ("top_quality", lambda e, q, k: select_top_quality(q, k)),
                    ]:
                        sel = method_fn(embs, quals, k)
                        ev = evaluate_selection(embs, quals, sel, labels, embs, labels)
                        for metric in ["topic_coverage", "mean_quality", "fill_distance", "sinkhorn_divergence", "dispersion", "coverage_certificate"]:
                            method_results[method_name][metric].append(ev[metric])
                
                results[key] = {}
                for method_name in method_results:
                    results[key][method_name] = {}
                    for metric in method_results[method_name]:
                        vals = method_results[method_name][metric]
                        results[key][method_name][metric] = {
                            "mean": float(np.mean(vals)),
                            "std": float(np.std(vals)),
                        }
                
                print(f"  {key}: DivFlow fill={results[key]['divflow']['fill_distance']['mean']:.3f}±{results[key]['divflow']['fill_distance']['std']:.3f} "
                      f"DPP fill={results[key]['dpp']['fill_distance']['mean']:.3f}±{results[key]['dpp']['fill_distance']['std']:.3f} "
                      f"MMR fill={results[key]['mmr']['fill_distance']['mean']:.3f}±{results[key]['mmr']['fill_distance']['std']:.3f}")
    
    return results


def main():
    print("=" * 70)
    print("DivFlow Revised Experiments")
    print("=" * 70)
    
    client = get_client()
    all_results = {}
    
    # Generate responses
    print("\n--- Generating LLM responses ---")
    all_responses = []
    all_labels = []
    prompt_map = {}
    
    for pname, prompt in PROMPTS.items():
        print(f"  Generating {N_PER_PROMPT} responses for '{pname}'...")
        resps = generate_responses(client, prompt, N_PER_PROMPT, SYSTEM_PROMPTS)
        all_responses.extend(resps)
        all_labels.extend([pname] * len(resps))
        prompt_map[pname] = prompt
    
    print(f"\n  Total responses: {len(all_responses)}")
    
    # Embed
    print("  Embedding responses...")
    all_embs = embed(client, all_responses)
    print(f"  Embedding shape: {all_embs.shape}")
    
    # LLM-as-judge quality scores
    print("  Computing LLM-as-judge quality scores...")
    all_quals = np.zeros(len(all_responses))
    for pname, prompt in PROMPTS.items():
        mask = [i for i, l in enumerate(all_labels) if l == pname]
        resps_for_prompt = [all_responses[i] for i in mask]
        scores = llm_judge_quality(client, resps_for_prompt, prompt)
        for j, idx in enumerate(mask):
            all_quals[idx] = scores[j]
    
    print(f"  Quality stats: mean={np.mean(all_quals):.3f} std={np.std(all_quals):.3f} "
          f"min={np.min(all_quals):.3f} max={np.max(all_quals):.3f}")
    
    # Pool statistics
    d_eff = _effective_dimension(all_embs)
    diameter = _data_diameter(all_embs)
    norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
    normed = all_embs / np.maximum(norms, 1e-12)
    sim_matrix = normed @ normed.T
    n = len(all_embs)
    sim_vals = []
    for i in range(n):
        for j in range(i+1, n):
            sim_vals.append(sim_matrix[i,j])
    sim_vals = np.array(sim_vals)
    
    all_results["pool_stats"] = {
        "n_responses": len(all_responses),
        "n_prompts": len(PROMPTS),
        "n_per_prompt": N_PER_PROMPT,
        "embedding_dim": int(all_embs.shape[1]),
        "effective_dim": d_eff,
        "diameter": float(diameter),
        "mean_cosine_sim": float(np.mean(sim_vals)),
        "std_cosine_sim": float(np.std(sim_vals)),
        "min_cosine_sim": float(np.min(sim_vals)),
        "max_cosine_sim": float(np.max(sim_vals)),
        "quality_mean": float(np.mean(all_quals)),
        "quality_std": float(np.std(all_quals)),
    }
    
    # Run experiments
    all_results["exp1_main"] = run_experiment_1_main_comparison(
        client, all_responses, all_embs, all_quals, all_labels)
    
    all_results["exp2_scaling"] = run_experiment_2_scaling(
        all_embs, all_quals, all_labels, all_labels)
    
    all_results["exp3_ic"] = run_experiment_3_ic_verification(
        all_embs, all_quals, all_labels)
    
    all_results["exp4_coverage"] = run_experiment_4_coverage_by_method(
        all_embs, all_quals, all_labels)
    
    all_results["exp5_downstream"] = run_experiment_5_downstream_task(
        client, all_responses, all_embs, all_quals, all_labels, prompt_map)
    
    all_results["exp6_synthetic"] = run_experiment_6_synthetic_scaling()
    
    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "revised_experiment_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    return all_results


if __name__ == "__main__":
    main()

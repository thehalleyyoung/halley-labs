"""Focused experiments saving results incrementally."""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.transport import sinkhorn_divergence, sinkhorn_candidate_scores, cost_matrix, _sinkhorn_divergence_cosine
from src.coverage import fill_distance, _effective_dimension, _data_diameter, dispersion, epsilon_net_certificate
from src.diversity_metrics import cosine_diversity
from src.kernels import RBFKernel
from src.dpp import greedy_map

def get_client():
    from openai import OpenAI
    return OpenAI()

def generate_responses(client, prompt, n, system_prompts, model="gpt-4.1-nano"):
    responses = []
    temps = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
    for i in range(n):
        sp = system_prompts[i % len(system_prompts)]
        t = temps[i % len(temps)]
        try:
            r = client.chat.completions.create(model=model,
                messages=[{"role":"system","content":sp},{"role":"user","content":prompt}],
                temperature=t, max_tokens=300)
            responses.append(r.choices[0].message.content)
        except: responses.append(f"[Error {i}]")
    return responses

def embed(client, texts, model="text-embedding-3-small"):
    all_embs = []
    for start in range(0, len(texts), 100):
        batch = texts[start:start+100]
        r = client.embeddings.create(model=model, input=batch)
        for item in r.data: all_embs.append(item.embedding)
    return np.array(all_embs)

def llm_judge_quality(client, responses, prompt, model="gpt-4.1-nano"):
    scores = []
    for resp in responses:
        try:
            r = client.chat.completions.create(model=model,
                messages=[
                    {"role":"system","content":"Rate this response 1-10. Output ONLY a number."},
                    {"role":"user","content":f"Q: {prompt}\nA: {resp[:500]}\nScore:"}
                ], temperature=0.0, max_tokens=5)
            s = float(''.join(c for c in r.choices[0].message.content.strip() if c.isdigit() or c=='.'))
            scores.append(min(max(s/10.0, 0.0), 1.0))
        except: scores.append(0.5)
    return np.array(scores)

# Selection methods
def select_divflow(embs, quals, k, quality_weight=0.3):
    n = embs.shape[0]
    selected = [int(np.argmax(quals))]
    for _ in range(k - 1):
        history = embs[selected]
        remaining = [i for i in range(n) if i not in selected]
        if not remaining: break
        cand = embs[remaining]
        div_scores = sinkhorn_candidate_scores(cand, history, embs, reg=None, n_iter=50)
        ds_min, ds_max = div_scores.min(), div_scores.max()
        div_norm = (div_scores - ds_min) / (ds_max - ds_min) if ds_max - ds_min > 1e-10 else np.ones_like(div_scores)
        rem_q = quals[remaining]
        q_min, q_max = rem_q.min(), rem_q.max()
        q_norm = (rem_q - q_min) / (q_max - q_min) if q_max - q_min > 1e-10 else np.ones_like(rem_q)
        combined = (1.0 - quality_weight) * div_norm + quality_weight * q_norm
        selected.append(remaining[int(np.argmax(combined))])
    return selected

def select_dpp(embs, quals, k):
    dists = cost_matrix(embs, embs, "euclidean")
    med = float(np.median(dists[dists > 0]))
    K = RBFKernel(bandwidth=max(med, 0.1)).gram_matrix(embs)
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
            if j in selected: continue
            max_sim = max(S[j, s] for s in selected)
            score = lam * quals[j] - (1 - lam) * max_sim
            if score > best_s: best_s, best_j = score, j
        if best_j >= 0: selected.append(best_j)
    return selected

def select_facility(embs, k):
    n = embs.shape[0]
    dists = cost_matrix(embs, embs, "euclidean")
    centroid_dists = np.linalg.norm(embs - np.mean(embs, axis=0), axis=1)
    selected = [int(np.argmin(centroid_dists))]
    for _ in range(k - 1):
        current_min = np.array([min(dists[p, s] for s in selected) for p in range(n)])
        best_j, best_g = -1, -float('inf')
        for j in range(n):
            if j in selected: continue
            new_min = np.minimum(current_min, dists[:, j])
            g = float(np.sum(current_min) - np.sum(new_min))
            if g > best_g: best_g, best_j = g, j
        if best_j >= 0: selected.append(best_j)
    return selected

def select_random(n, k, seed=42):
    return list(np.random.RandomState(seed).choice(n, min(k, n), replace=False))

def select_top_quality(quals, k):
    return list(np.argsort(quals)[-k:][::-1])

def evaluate(embs, quals, selected, labels):
    sel_embs = embs[selected]
    sel_labels = [labels[i] for i in selected]
    unique_topics = len(set(sel_labels))
    n_topics = len(set(labels))
    # Cosine fill distance (more meaningful in high-d)
    norms_all = np.linalg.norm(embs, axis=1, keepdims=True)
    normed_all = embs / np.maximum(norms_all, 1e-12)
    norms_sel = np.linalg.norm(sel_embs, axis=1, keepdims=True)
    normed_sel = sel_embs / np.maximum(norms_sel, 1e-12)
    # Fill distance in cosine space: max over pool of min cosine distance to selected
    cos_sim_matrix = normed_all @ normed_sel.T  # (n, k)
    max_sim_to_selected = np.max(cos_sim_matrix, axis=1)  # (n,)
    cos_fill_dist = float(1.0 - np.min(max_sim_to_selected))  # max cosine distance
    # Sinkhorn divergence (cosine-based)
    sdiv = _sinkhorn_divergence_cosine(sel_embs, embs, reg=0.05, n_iter=50)
    disp = dispersion(sel_embs)
    cert = epsilon_net_certificate(sel_embs, embs, epsilon=0.5)
    return {
        "n_topics_covered": unique_topics, "n_topics_total": n_topics,
        "topic_coverage": unique_topics / n_topics,
        "mean_quality": float(np.mean(quals[selected])),
        "cosine_diversity": float(cosine_diversity(sel_embs)),
        "cosine_fill_distance": cos_fill_dist,
        "sinkhorn_divergence": float(sdiv),
        "dispersion": float(disp),
        "coverage_certificate": cert.coverage_fraction,
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

K = 10

def main():
    print("=" * 60)
    print("DivFlow Focused Experiments")
    print("=" * 60)
    client = get_client()
    results = {}

    # Generate responses
    print("\n--- Phase 1: Generate responses ---")
    all_responses, all_labels = [], []
    for pname, prompt in PROMPTS.items():
        print(f"  {pname}...", end=" ", flush=True)
        resps = generate_responses(client, prompt, 10, SYSTEM_PROMPTS)
        all_responses.extend(resps)
        all_labels.extend([pname] * len(resps))
        print(f"got {len(resps)}")

    print(f"\n  Total: {len(all_responses)} responses")
    all_embs = embed(client, all_responses)
    print(f"  Embeddings: {all_embs.shape}")

    # Quality scores
    print("\n--- Phase 2: Quality scoring ---")
    all_quals = np.zeros(len(all_responses))
    for pname, prompt in PROMPTS.items():
        mask = [i for i, l in enumerate(all_labels) if l == pname]
        scores = llm_judge_quality(client, [all_responses[i] for i in mask], prompt)
        for j, idx in enumerate(mask): all_quals[idx] = scores[j]
    print(f"  Quality: mean={np.mean(all_quals):.3f} std={np.std(all_quals):.3f}")

    # Pool stats
    d_eff = _effective_dimension(all_embs)
    diameter = _data_diameter(all_embs)
    norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
    normed = all_embs / np.maximum(norms, 1e-12)
    sim = normed @ normed.T
    n = len(all_embs)
    upper_tri = sim[np.triu_indices(n, k=1)]
    results["pool_stats"] = {
        "n": n, "n_prompts": len(PROMPTS), "d": int(all_embs.shape[1]),
        "d_eff": d_eff, "diameter": float(diameter),
        "mean_cos_sim": float(np.mean(upper_tri)),
        "std_cos_sim": float(np.std(upper_tri)),
        "quality_mean": float(np.mean(all_quals)),
        "quality_std": float(np.std(all_quals)),
    }
    print(f"  d_eff={d_eff}, diameter={diameter:.3f}, mean_sim={np.mean(upper_tri):.3f}")

    # Experiment 1: Main comparison
    print("\n--- Phase 3: Method comparison ---")
    exp1 = {}
    methods = {
        "divflow": lambda: select_divflow(all_embs, all_quals, K),
        "dpp": lambda: select_dpp(all_embs, all_quals, K),
        "mmr": lambda: select_mmr(all_embs, all_quals, K),
        "facility_location": lambda: select_facility(all_embs, K),
        "random": lambda: select_random(n, K),
        "top_quality": lambda: select_top_quality(all_quals, K),
    }
    for name, fn in methods.items():
        print(f"  {name}...", end=" ", flush=True)
        sel = fn()
        ev = evaluate(all_embs, all_quals, sel, all_labels)
        exp1[name] = ev
        print(f"topics={ev['n_topics_covered']}/{ev['n_topics_total']} "
              f"q={ev['mean_quality']:.3f} sdiv={ev['sinkhorn_divergence']:.4f} "
              f"cos_fill={ev['cosine_fill_distance']:.4f}")
    results["exp1_main"] = exp1

    # Experiment 2: Scaling
    print("\n--- Phase 4: Scaling ---")
    exp2 = {}
    prompt_list = list(set(all_labels))
    for np_ in [2, 4, 6, 8, 10, 15]:
        if np_ > len(prompt_list): continue
        use = prompt_list[:np_]
        mask = [i for i, l in enumerate(all_labels) if l in use]
        if len(mask) < K: continue
        sub_e, sub_q, sub_l = all_embs[mask], all_quals[mask], [all_labels[i] for i in mask]
        sc = {}
        for name, fn in [
            ("divflow", lambda e, q: select_divflow(e, q, K)),
            ("dpp", lambda e, q: select_dpp(e, q, K)),
            ("mmr", lambda e, q: select_mmr(e, q, K)),
            ("facility_location", lambda e, q: select_facility(e, K)),
            ("random", lambda e, q: select_random(len(q), K)),
            ("top_quality", lambda e, q: select_top_quality(q, K)),
        ]:
            sel = fn(sub_e, sub_q)
            ev = evaluate(sub_e, sub_q, sel, sub_l)
            sc[name] = ev
        exp2[f"{np_}_prompts"] = sc
        print(f"  {np_} prompts: DF={sc['divflow']['n_topics_covered']}/{np_} "
              f"DPP={sc['dpp']['n_topics_covered']}/{np_} "
              f"MMR={sc['mmr']['n_topics_covered']}/{np_} "
              f"FL={sc['facility_location']['n_topics_covered']}/{np_} "
              f"TopQ={sc['top_quality']['n_topics_covered']}/{np_}")
    results["exp2_scaling"] = exp2

    # Experiment 3: Downstream - LLM-as-judge
    print("\n--- Phase 5: Downstream evaluation ---")
    exp3 = {}
    for name in ["divflow", "mmr", "dpp", "top_quality", "random"]:
        sel = methods[name]() if name in methods else select_random(n, K)
        sel_labels = [all_labels[i] for i in sel]
        sel_resps = [all_responses[i] for i in sel]
        briefing = "\n---\n".join([f"[{sel_labels[i]}] {r[:200]}" for i, r in enumerate(sel_resps)])
        try:
            r = client.chat.completions.create(model="gpt-4.1-nano",
                messages=[
                    {"role":"system","content":"Evaluate these responses on: breadth (distinct topics, 1-10), utility (usefulness, 1-10), non-redundancy (1-10). Output three numbers comma-separated."},
                    {"role":"user","content":f"Responses:\n{briefing[:3000]}"}
                ], temperature=0.0, max_tokens=20)
            nums = [float(x.strip()) for x in r.choices[0].message.content.strip().split(",") if x.strip()]
            b, u, nr = (nums[0], nums[1], nums[2]) if len(nums) >= 3 else (5, 5, 5)
        except: b, u, nr = 5, 5, 5
        exp3[name] = {"breadth": b, "utility": u, "non_redundancy": nr, 
                       "composite": (b+u+nr)/3, "n_topics": len(set(sel_labels))}
        print(f"  {name:20s}: breadth={b:.0f} utility={u:.0f} non_red={nr:.0f} topics={len(set(sel_labels))}")
    results["exp3_downstream"] = exp3

    # Experiment 4: Synthetic with variance
    print("\n--- Phase 6: Synthetic experiments ---")
    from collections import defaultdict
    exp4 = {}
    seeds = [42, 137, 256, 512, 1024]
    for N, d in [(100, 16), (200, 64)]:
        key = f"N{N}_d{d}"
        mr = defaultdict(lambda: defaultdict(list))
        for seed in seeds:
            rng = np.random.RandomState(seed)
            nc = 5; cs = N // nc
            embs_s = np.vstack([rng.randn(cs, d)*0.5 + rng.randn(d)*3 for _ in range(nc)])[:N]
            labels_s = [f"c{i//cs}" for i in range(N)]
            quals_s = rng.uniform(0.5, 1.0, N)
            for mn, mf in [
                ("divflow", lambda e, q: select_divflow(e, q, 10)),
                ("dpp", lambda e, q: select_dpp(e, q, 10)),
                ("mmr", lambda e, q: select_mmr(e, q, 10)),
                ("facility_location", lambda e, q: select_facility(e, 10)),
                ("random", lambda e, q: select_random(len(q), 10, seed)),
                ("top_quality", lambda e, q: select_top_quality(q, 10)),
            ]:
                sel = mf(embs_s, quals_s)
                ev = evaluate(embs_s, quals_s, sel, labels_s)
                for m in ["topic_coverage","mean_quality","sinkhorn_divergence","cosine_fill_distance"]:
                    mr[mn][m].append(ev[m])
        exp4[key] = {mn: {m: {"mean": float(np.mean(v)), "std": float(np.std(v))} 
                    for m, v in ms.items()} for mn, ms in mr.items()}
        df = exp4[key]
        print(f"  {key}: DF_cov={df['divflow']['topic_coverage']['mean']:.2f} "
              f"DPP_cov={df['dpp']['topic_coverage']['mean']:.2f} "
              f"MMR_cov={df['mmr']['topic_coverage']['mean']:.2f}")
    results["exp4_synthetic"] = exp4

    # Save
    out = os.path.join(os.path.dirname(__file__), "revised_experiment_results.json")
    with open(out, "w") as f: json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()

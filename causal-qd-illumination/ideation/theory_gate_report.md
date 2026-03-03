# Theory Gate Report — CausalQD Verification

**Pipeline Stage:** Verification (post-theory)  
**Date:** 2026-03-02  
**Method:** Claude Code Agent Teams — 3 expert agents (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) with adversarial cross-critique and consensus resolution  
**Proposals Evaluated:** 1 (proposal_00)

---

## Evaluation Summary

### Proposal proposal_00: CausalQD — Quality-Diversity Illumination for Causal Structure Discovery

**Composite Score: 3.0/10**

| Axis | Score | Justification |
|------|:-----:|---------------|
| **Extreme and Obvious Value** | **3/10** | Core problem (diverse causal structure recovery) already solved by Order MCMC + Bayesian Model Averaging with superior probabilistic semantics. No concrete decision scenario where archive beats P(X→Y\|D). Causal sufficiency + linear Gaussian + n≤20 honest scope limits applicability. LLMs further erode "exploration tool" framing. |
| **Genuine Software Difficulty** | **4/10** | No novel algorithm. MAP-Elites is off-the-shelf (pyribs). BIC scoring exists (causal-learn). Hardest component (bootstrap certificates) must be dropped for laptop feasibility. Remaining system is ~1,500–2,000 lines of standard evolutionary search with domain-specific operators. Buildable in 2–3 weeks. |
| **Best-Paper Potential** | **2/10** | Unanimous finding across all evaluators: "No theorem is both correct and non-trivial." 9/9 theorems are trivial, circular, tautological, restating known results, or contain mathematical errors. Lipschitz certificates are mathematically vacuous (L_BIC = O(1/N) vs C(e) = O(N) → every detectable edge passes, zero discrimination). Best-paper ceiling is zero at any venue. |

---

## Fatal Flaws (Unanimous)

| # | Flaw | Severity |
|---|------|----------|
| F1 | **Missing Order MCMC baseline** — strongest existing method for diverse structure recovery omitted entirely | Fatal |
| F2 | **All 9 theorems are either trivial, circular, or contain errors** — no load-bearing theoretical contribution | Fatal |
| F3 | **Lipschitz certificates mathematically vacuous** — certificate/threshold ratio is 1,800–5,600×; provides zero discrimination | Fatal |
| F4 | **Circular coverage guarantee** — Theorem 3 conditions on ergodicity, which is what the theorem claims to prove | Fatal |
| F5 | **Bootstrap certificates computationally infeasible** — 200–1000× full reruns; proposal requests 96-core server, 3,200 CPU-hours | Fatal |
| F6 | **No articulated use case** — no concrete scenario where archive enables a decision that Bayesian posterior doesn't | Major |
| F7 | **Factor-of-2 error in Theorem 7** — statement says L·δ, proof derives 2L·δ | Moderate |
| F8 | **Exponential FIND-ALL-CYCLES in crossover repair** | Moderate |

---

## Team Verdicts

| Expert | Verdict | Score | Key Position |
|--------|---------|:-----:|-------------|
| Independent Auditor | CONDITIONAL CONTINUE (narrow) | 3.3 | Certificate vacuousness confirmed numerically. LLMs erode value further. Behavioral descriptors are the only novel element. |
| Fail-Fast Skeptic | **ABANDON** | — | "17 mandatory modifications = new proposal." Only 15–20% survives. Best-paper ceiling is zero. P(publication) ≈ 7.4%. "A conditional continue with 17 conditions is an abandon that hasn't accepted itself yet." |
| Scavenging Synthesizer | CONDITIONAL GO (pivot) | — | 5 elements survive unanimously. Hidden gem: descriptor variance → uncertainty decomposition. Go/no-go costs 2 hours on Sachs. |

### Cross-Critique Resolution

**CONSENSUS VERDICT: CONDITIONAL ABANDON**

The original CausalQD proposal is **dead**. All experts agree the thesis, contributions, theorems, and framing must be abandoned. The "conditional continue" verdicts from the three prior evaluations are, as the Skeptic correctly identifies, semantically equivalent to abandon — they require dropping 100% of theorems, changing the thesis, removing core contributions, downgrading the venue, and cutting scope by 75%.

A **4-hour scavenging probe** is approved: run the go/no-go experiment on Sachs (n=11) testing MI-profile descriptors against four pass/fail criteria. If it passes, a 3-week pivot to a smaller empirical paper is viable. If it fails, full abandon with no further investment.

---

## What Survives the Critique

All three evaluators and all three verification experts independently converged on the same surviving element:

1. **Behavioral descriptor space** — MI profiles conditioned on parent sets as organizing principle for DAG diversity. Genuinely novel, no direct precedent. (HIGH confidence)
2. **Acyclicity-preserving genetic operators** — topological ordering mutation + order-based crossover. Reusable library contribution. (HIGH confidence)
3. **Descriptor variance → uncertainty decomposition** — bootstrap descriptor variance separates structural vs. parametric uncertainty. Hidden gem identified by Synthesizer. (MODERATE confidence — unvalidated)

---

## Go/No-Go Criteria (if scavenging probe approved)

| Test | GO | NO-GO |
|------|-----|-------|
| MI-only descriptors separate ≥2 MECs (silhouette > 0.2) | Pass | Fail → full abandon |
| Intra-MEC subclusters exist (≥2 clusters within largest MEC) | Pass | Fail → full abandon |
| Uncertainty components have pairwise correlation < 0.6 | Pass | Fail → full abandon |
| GES and MCMC cover different descriptor regions (Jaccard < 0.7) | Pass (any 2/4) | Informational |

---

## Risk Assessment

| Risk | Probability | Impact |
|------|:-----------:|--------|
| Order MCMC dominates on descriptor coverage | 50% | High — MAP-Elites adds nothing |
| Landscape analysis reveals no surprises | 40% | High — nothing to publish |
| Go/no-go fails (tests 1–3) | 45% | Terminal — direction abandoned |
| Scope creep back to original framing | 25% | High — wastes resources |

---

## Publication Probability (if all conditions met)

| Venue | Probability |
|-------|:-----------:|
| NeurIPS/ICML main track | <5% |
| UAI/CLeaR main track | 15–25% |
| NeurIPS/ICML workshop | 40–50% |
| CLeaR/UAI workshop | 60–70% |

---

## Ranking

```json
{
  "rankings": [
    {
      "proposal_id": "proposal_00",
      "score": 3.0,
      "verdict": "ABANDON",
      "reason": "All 3 evaluators and 3 verification experts unanimously find: no theorem is both correct and non-trivial (9/9 theorems trivial/circular/erroneous), Lipschitz certificates mathematically vacuous (1,800-5,600x ratio), missing Order MCMC baseline, bootstrap certificates computationally infeasible on laptop CPU, no articulated decision scenario. Composite 3.0/10. Original proposal requires 17 mandatory modifications removing 80-85% of content — this is a new proposal, not a continuation. Behavioral descriptor space (MI profiles conditioned on parent sets) is the sole genuinely novel element; recommend scavenging as standalone utility module (200 lines, 1 week) rather than 3-month paper with 7.4% publication probability.",
      "scavenge_from": []
    }
  ]
}
```

---

## Process Notes

### Team Workflow

1. **Independent Proposals** (Phase 1): Three experts produced reports independently without seeing each other's work.
2. **Adversarial Cross-Critique** (Phase 2): Direct challenges between experts on four key disagreements.
3. **Consensus Resolution** (Phase 3): Team lead resolved each disagreement with reasoning and produced binding verdict.

### Key Insight from Process

The most important analytical contributions came from unlikely sources: the Scavenging Synthesizer (role: find value) produced the strongest *new* insight (uncertainty decomposition), while the Fail-Fast Skeptic (role: destroy) produced the most *honest* meta-analysis (conditional continue = disguised abandon). The Independent Auditor (role: verify) produced the most *reliable* numerical checks (certificate vacuousness confirmed). Cross-critique corrected each role's natural bias.

### Evaluator Agreement

Across all six evaluators (3 original + 3 verification), there is **unanimous agreement** on:
- Order MCMC is the missing primary baseline
- No theorem is both correct and non-trivial
- Behavioral descriptor space is the sole novel element
- Bootstrap certificates are infeasible on laptop CPU
- Honest scope is n ≤ 20

This level of consensus across 6 independent evaluations strongly validates the findings.

---

*Verification completed by: Claude Code Agent Teams (3 experts + adversarial cross-critique + consensus resolution)*  
*Total evaluation chain: 3 original evaluators → 3 verification experts → 1 cross-critique resolution = 7 evaluation passes*

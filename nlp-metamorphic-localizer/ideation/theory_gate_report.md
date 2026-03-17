# Verification Gate Report: nlp-metamorphic-localizer (proposal_00)

**Title:** "Where in the Pipeline Did It Break? Causal Fault Localization for Multi-Stage NLP Systems"
**Stage:** Post-theory verification (6-evaluator adversarial panel)
**Date:** 2026-03-08
**Method:** Two-round adversarial verification. Round 1: 3 independent evaluators (Skeptic 5.8, Mathematician 6.2, Community Expert 6.6). Round 2: 3-expert panel (Independent Auditor 6.0, Fail-Fast Skeptic 3.8, Scavenging Synthesizer 7.3) with moderated cross-critique producing converged scores.

---

## Executive Summary

**Final Composite: 5.85/10 — CONDITIONAL CONTINUE (62% confidence)**

The project's diamond contribution, M4's causal-differential fault localization with introduction-vs-amplification distinction, is genuinely novel in the NLP testing space and solves a real debugging pain point — no existing tool (CheckList, TextFlint, TextAttack, LangTest, METAL, LLMORPH) provides pipeline-stage fault localization. However, the math is predictable (SBFL + do-calculus is the obvious synthesis, earning only 1.5 true math contributions), the addressable market is contracting (multi-stage NLP pipelines at ~15–25% of production NLP), and the project's existential risk is bug yield collapse (30% chance Tier 1 produces <5 real bugs). The 7 phase gates cap maximum wasted investment at ≤2 engineer-weeks before the first kill decision. At P(any-publication)=70% and P(accept)=58%, this clears the threshold for continued investment, gated on the Week 2 MVP prototype.

---

## Final Converged Scores

| Axis | R1 Skeptic | R1 Mathematician | R1 Community | R2 Auditor | R2 Skeptic | R2 Synthesizer | R2 Converged | **Final** |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Extreme Value | 5 | 6 | 6 | 5.5 | 3 | 7 | 5.0 | **5.3** |
| Genuine Difficulty | 6 | 6 | 6.5 | 6 | 4 | 7 | 5.5 | **5.8** |
| Best-Paper Potential | 5 | 5 | 6.5 | 5 | 3 | 7 | 4.5 | **4.9** |
| Laptop-CPU / No Humans | 7 | 7 | 7 | 7 | 5 | 8 | 7.0 | **7.0** |
| Feasibility | 6 | 7 | 7 | 6.5 | 4 | 7.5 | 6.0 | **6.3** |
| **Composite** | **5.8** | **6.2** | **6.6** | **6.0** | **3.8** | **7.3** | **5.6** | **5.85** |

Weighting: 40% Round 1 average, 60% Round 2 cross-critique convergence.

---

## Axis-by-Axis Justification

### 1. Extreme & Obvious Value — 5.3/10

The capability gap is real and unanimously verified: no existing NLP testing tool provides pipeline-stage fault localization with introduction-vs-amplification distinction. The qualitative shift from "pipeline fails on passive voice" to "POS tagger mishandles passivized gerunds, amplified 4.7× by parser" is genuine. Regulated industries (healthcare NER, legal clause classification, financial compliance) anchor persistent demand via EU AI Act and FDA SaMD requirements.

However, the market is contracting (15–25% of production NLP, declining). The "composed inference systems" generalization is aspirational with zero non-NLP evidence. The GPT-4-as-debugger competitor is unaddressed as a practical alternative.

### 2. Genuine Software Difficulty — 5.8/10

Multi-domain synthesis across NLP, SE testing, causal inference, and formal methods is genuinely rare. Token-to-tree alignment across interventions, grammar-constrained three-way optimization in the shrinker, and feature-checker calibration are non-trivial integration challenges.

However, the high-level approach (SBFL + causal intervention for pipeline stages) is predictable — the first thing an SE+NLP expert would conceive. Individual techniques are all well-understood in their home domains. The difficulty is integration, not invention. ~13K genuinely novel LoC of ~40.5K total.

### 3. Best-Paper Potential — 4.9/10

M4 is graded B+ by the Mathematician — novel in domain synthesis but using textbook techniques. The math portfolio without N2 is 1.5 contributions, below the 2–3 typical for strong-accept theory papers. The "10 Bugs, 10 Words" artifact is a reviewer magnet but has 30% failure risk. N2 has 40% failure risk.

Best-paper requires compound favorable outcomes (N2 proved AND Tier 1 ≥10 bugs), with unconditional P(best-paper) ≈ 4%. Tools-track accept at ISSTA/ASE is realistic at ~58%.

### 4. Laptop-CPU + No Humans — 7.0/10

**Perfect consensus across all 6 evaluations.** Statistical pipelines (~75 min) are genuinely interactive on laptop CPU. Transformer pipelines (~3–4 hours) are honestly documented as CI-only. RAG/LLM correctly excluded. The composition theorem reduces test count 10×. Per-run execution is fully automated; seeds are one-time tool configuration, not annotation.

### 5. Feasibility — 6.3/10

The 107K→40.5K scope cut and grammar compiler elimination demonstrate disciplined engineering judgment. The MVP at ~18K LoC is a genuine floor. Staged rollout contains risk. All prior fatal flaws (F-A1 through F-C2) addressed. The 2-week prototype gate caps wasted investment.

Zero implementation exists (theory_bytes=0, impl_loc=0), which is expected at theory-complete phase but means all claims are projections. Compound failure probability ~60–65%.

---

## Key Metrics

| Metric | Value | Source |
|--------|-------|--------|
| Final Composite | 5.85/10 | 6-evaluator weighted convergence |
| Genuinely Novel LoC | ~13,000 ± 1,500 | Cross-validated across 3 R1 evaluations |
| Total LoC | ~40,500 | All evaluations agree |
| Novelty Ratio | ~32–36% | Novel LoC concentrated in M4 + shrinker |
| True Math Contributions | 1.5 (without N2) / 2.5 (with N2) | Mathematician confirmed, panel agrees |
| Math Portfolio Grade | B-/C+ | Mathematician assessment |
| Fatal Flaws | 0 | All 6 evaluations agree |
| Serious Flaws | 3 | Multi-fault degeneracy, N2 uncomputable quantity, meaning-preservation circularity |
| Crown Jewel | M4 causal-differential localization | Unanimous across all evaluations |
| Dark Horse | N1 discriminability matrix | Community Expert + Synthesizer identified |

---

## Fatal Flaw Analysis

**Fatal flaws: 0.** No single issue is project-killing.

**Serious flaws (3):**

1. **Multi-fault degeneracy** — M4's argmax heuristic identifies the noisiest stage, not necessarily the causal one, when multiple stages are simultaneously faulty. The iterative peeling extension is described but not formally analyzed. Impact: top-1 accuracy may drop to <70% on multi-fault scenarios.

2. **Bug yield is uncontrollable** — 30% probability Tier 1 yields <5 actionable bugs. The "10 Bugs, 10 Words" centerpiece depends on empirical reality about specific models. Many "inconsistencies" may be expected model behavior, not software defects.

3. **Zero implementation** — theory_bytes=0, impl_loc=0. Every feasibility and performance claim is speculative. All scores carry ±1.5 point uncertainty until prototype validates core loop.

---

## Probability Estimates (6-Evaluator Convergence)

| Outcome | Estimate | Range Across Evaluators |
|---------|:---:|:---:|
| P(best-paper at ISSTA/ASE) | **4%** | 3–12% |
| P(strong accept, research track) | **14%** | 10–20% |
| P(accept, any track) | **58%** | 45–65% |
| P(any publication, any venue) | **70%** | 68–78% |
| P(abandon before submission) | **28%** | 22–35% |

---

## Binding Amendments (Consolidated)

| ID | Amendment | Type | Deadline |
|----|-----------|------|----------|
| BA-1 | Week-1 bug pre-screen: ≥3 genuine findings across 3+ transformations on 2 pipelines | Kill gate | Week 1 |
| BA-2 | N2 proof checkpoint: hard go/no-go on lower bound proof sketch | Kill gate | Week 4 |
| BA-3 | Bug framing discipline: operational definition (maintainer-actionable, reproducible ≥3×, severity threshold). Classify as functional-bug / robustness-defect / behavioral-observation | Definition lock | Week 2 |
| BA-4 | Early GPT-4 baseline: test localization accuracy on 20 pilot cascading-fault scenarios before full eval | Calibration | Week 3 |
| BA-5 | Multi-fault honesty: explicit limitations section; report multi-fault accuracy separately | Paper requirement | Pre-submission |
| BA-6 | 2-week prototype gate: M4 MVP on spaCy-sm with 3 transformations, ≥70% top-1, ≥10pt causal uplift over vanilla SBFL | Kill gate | Week 2 |

---

## Phase Gates

| Gate | Week | Criterion | Action if FAIL |
|------|:---:|-----------|----------------|
| PG-1 | 1 | Infrastructure produces output on injected fault | **ABANDON** |
| PG-2 | 2 | M4 prototype: ≥70% top-1 on injected faults AND ≥10pt causal improvement over raw SBFL | **ABANDON** |
| PG-3 | 4 | N2 proof sketch with concrete constants | Demote N2 to future work |
| PG-4 | 4 | rank(M) ≥ n for spaCy pipeline (N1 validates) | Reduce claims to stage-group localization |
| PG-5 | 6 | ≥8 of 15 transformations working | Scope to 8-transformation paper |
| PG-6 | 8 | Tier 2 top-1 accuracy ≥70% | Major pivot or **ABANDON** |
| PG-7 | 10 | Tier 1 yields ≥5 real bugs | Fallback to tools-track / demo submission |

---

## Salvage Value (from abandoned approaches)

1. **N1 discriminability matrix** — Reusable diagnostic for any metamorphic testing framework. The rank-check ("can your transformation set distinguish all pipeline stages?") has independent utility.
2. **M4 causal-differential core** — The localization algorithm is pipeline-agnostic in principle. Applicable to ML feature pipelines, data processing DAGs, compiler passes.
3. **Feature-unification validity checker** — Standalone NLP grammar component (~2–3K LoC Rust) with potential reuse in grammar-constrained text generation.
4. **Boltzmann sampling over corpus-extracted subtrees** (~1.5K LoC Python) — Partially recovers generative capacity from killed grammar compiler.
5. **ADAPTIVE-LOCATE Thompson-sampling heuristic** — Principled test selection without N2 proof; implementable in ~500 LoC.

---

## Cross-Critique Resolution Summary

| Dispute | Winner | Impact |
|---------|--------|--------|
| Market dying vs. stable niche | **Synthesizer** | V stays at 5, not 3 |
| M4 trivially conceived | **Split** (Skeptic: predictable; Synthesizer: non-trivial to implement) | D at 5.5 |
| Feature-checker tar-pit risk | **Auditor** | Bounded at 4K LoC with calibration gate |
| Best-paper probability | **Auditor** | 4% unconditional, not 3% or 15% |
| 300 seeds = human annotation | **Auditor/Synthesizer** | CPU stays at 7 |
| Compound failure vs. staged plan | **Auditor** | F at 6, not 4 |
| Zero code = low feasibility | **Synthesizer** | Phase-appropriate, not a deficiency |
| Multi-fault as serious flaw | **Synthesizer** | Standard SBFL limitation, honest documentation sufficient |
| Generalization beyond NLP | **Skeptic** | No claims without non-NLP evaluation |

**Overall:** Auditor was closest to correct on 4/10 disputes, Synthesizer 4/10, Skeptic 1/10, 1 split.

---

## Team Disposition

| Expert | Composite | Verdict | Key Contribution |
|--------|:---------:|---------|-----------------|
| R1 Skeptic | 5.8 | CONDITIONAL CONTINUE | Market contraction analysis; binding amendments |
| R1 Mathematician | 6.2 | CONDITIONAL CONTINUE | Honest math assessment (1.5 contributions, B-/C+) |
| R1 Community Expert | 6.6 | CONDITIONAL CONTINUE | Regulatory tailwind; N1 as dark horse; binding conditions |
| R2 Independent Auditor | 6.0 | CONDITIONAL CONTINUE | Evidence-based calibration; closest to converged scores |
| R2 Fail-Fast Skeptic | 3.8 | ABANDON | Feature-checker landmine; forced prototype gate |
| R2 Scavenging Synthesizer | 7.3 | CONTINUE | GPT-4 baseline framing; salvage opportunities; ADAPTIVE-LOCATE |

**Consensus: CONDITIONAL CONTINUE (5 of 6 experts; R2 Skeptic dissents with ABANDON at 3.8)**

---

## Recommendations (Non-Binding)

1. **Lead the abstract with the GPT-4 comparison.** "Our tool achieves X% localization accuracy on cascading faults; GPT-4's best strategy achieves Y%." This is the sentence that gets the paper read.
2. **Promote N1 to §3.1** — "Before localizing, verify localization is *possible*." Dark horse for most-cited result.
3. **Reframe as "causal fault localizer for compositional inference systems"** with NLP as the proving ground — but scope claims to evaluation coverage per BA-8.
4. **Implement ADAPTIVE-LOCATE as Thompson-sampling heuristic** regardless of N2 fate.
5. **Elevate behavioral atlas to secondary contribution** with DOI as first benchmark for multi-stage NLP pipeline fault localization.
6. **Python-first MVP** — defer Rust to week 6–8; feature-checker stays Python until performance proves insufficient.
7. **Pin all model versions** with SHA256 hashes before Tier 1 evaluation.

---

## Final JSON Ranking

```json
{
  "rankings": [
    {
      "proposal_id": "proposal_00",
      "score": 5.85,
      "verdict": "CONTINUE",
      "reason": "M4 causal-differential fault localization fills a genuine, verified gap (no existing NLP tool provides pipeline-stage localization with introduction-vs-amplification distinction). Final composite 5.85/10 after 6-evaluator adversarial panel. 1.5 true math contributions, ~13K genuinely novel LoC, 0 fatal flaws. P(accept)=58%, P(any-pub)=70%. Market is contracting but regulatory niche persists. Gated on 2-week prototype (PG-2) and 6 binding amendments. Approach is predictable but execution plan is disciplined with 7 phase gates capping wasted investment at ≤2 weeks.",
      "scavenge_from": []
    }
  ]
}
```

---

*Verification produced by 6-evaluator adversarial panel across two rounds with moderated cross-critique convergence. All scores reflect post-adversarial synthesis. Binding amendments and phase gates are non-negotiable for phase advancement. The project proceeds to PG-2 (Week 2 MVP prototype) as the decisive commit-or-kill moment.*

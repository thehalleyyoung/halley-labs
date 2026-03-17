# Verification Skeptic Evaluation: nlp-metamorphic-localizer (proposal_00)

**Title:** "Where in the Pipeline Did It Break? Causal Fault Localization for Multi-Stage NLP Systems"
**Stage:** Post-theory verification
**Date:** 2026-03-08
**Method:** 3-expert adversarial panel (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) with independent proposals, cross-critique by Panel Moderator, and forced convergence.
**Prior state:** theory_bytes=0, impl_loc=0. Phase: theory_complete.

---

## Panel Composition and Process

Three independent evaluators scored the proposal blind, then confronted each other's arguments in a moderated cross-critique. The Panel Moderator adjudicated each disagreement and produced converged scores.

| Expert | Initial Composite | Verdict |
|--------|------------------|---------|
| Independent Auditor | 6.2/10 | CONTINUE |
| Fail-Fast Skeptic | 3.6/10 | ABANDON |
| Scavenging Synthesizer | 6.8/10 | CONTINUE |
| **Panel Moderator (converged)** | **5.8/10** | **CONDITIONAL CONTINUE** |

The 3.2-point gap between the Skeptic (3.6) and Synthesizer (6.8) was the largest inter-expert divergence. The cross-critique resolved all five disagreements; the Auditor's positions proved closest to correct on every axis.

---

## Converged Scores

| Axis | Auditor | Skeptic | Synthesizer | **Converged** |
|------|---------|---------|-------------|---------------|
| 1. Extreme & Obvious Value | 5 | 3 | 6 | **5** |
| 2. Genuine Software Difficulty | 6 | 5 | 7 | **6** |
| 3. Best-Paper Potential | 5 | 3 | 6 | **5** |
| 4. Laptop CPU + No Humans | 8 | 4 | 8 | **7** |
| 5. Feasibility | 7 | 3 | 7 | **6** |
| **Composite** | **6.2** | **3.6** | **6.8** | **5.8/10** |

**Verdict: CONDITIONAL CONTINUE — gated on BA-6 prototype validation by week 2.**

---

## Axis 1: Extreme and Obvious Value — 5/10

### What's genuine

The core value proposition is real and differentiated. No existing NLP testing tool — CheckList, TextFlint, TextAttack, LangTest, METAL, LLMORPH — provides pipeline-stage fault localization. The qualitative shift from "the pipeline fails on passivized inputs" to "the POS tagger mishandles passivized gerunds, causing parser misattachment cascading to NER; proof: 'The report was being written by Kim' (8 words)" is a category change in debugging capability. For regulated-industry teams (healthcare, legal, financial NLP), structured testing evidence has compliance value.

### What's missing

**Market contraction.** The addressable market (classical multi-stage NLP pipelines) is estimated at 15–25% of production NLP in 2026 and shrinking. The John Snow Labs 2023 survey ("60% of NLP teams spend more time debugging pipeline interactions than training models") is 3 years old and predates the LLM monoculture shift. Most HuggingFace production deployments are single-model, not multi-stage. RAG/LLM pipeline support — the growth segment — is explicitly out of scope after BA-2.

**The "composed inference systems" reframing is aspirational, not demonstrated.** The Synthesizer correctly noted that M4's localization loop is pipeline-agnostic. However, the evaluation covers only NLP pipelines. Claiming generality to "any composed inference system" without non-NLP evaluation is a red flag for reviewers. The reframing is valid as paper framing but must be scoped to what the evaluation covers.

**The LLM-assisted debugging competitor is unaddressed.** The proposal includes a GPT-4-as-debugger baseline, implicitly conceding that LLM-assisted debugging (zero-setup, zero-LoC) may be competitive. If GPT-4 reaches within 15 points of the tool on cascading faults, that's a devastating result for the 40K-LoC investment thesis.

### Score justification

Real value for a real but narrowing audience. The tool solves a genuine problem but the market it serves is contracting and the tool's adoption barrier (40K LoC dependency) is immense. Score 5 reflects "genuine value for a specific community" without "extreme and obvious value for a broad audience."

---

## Axis 2: Genuine Software Difficulty — 6/10

### The 500-line prototype test

The Skeptic constructed a mental ~200-line prototype of M4's core: per-stage differential computation, Ochiai SBFL ranking, and interventional replay verification. The core insight is statable in one paragraph.

**The panel concludes the Skeptic is right about the core idea but wrong about the full contribution.** The gap between the ~200-line prototype and the production-quality M4 includes:

1. **Type-specific distance functions** for heterogeneous IRs (token sequences, POS tags, dependency trees, entity spans) calibrated so Δ=0.3 at the parser is comparable to Δ=0.3 at the NER. ~1K LoC of non-trivial metric engineering.
2. **Interventional replay with token-to-tree alignment.** When passivization changes word order, aligning intermediate representations requires solving the alignment problem. 11/15 transformations use lemma-level alignment; 4 require explicit alignment maps. ~1.5K LoC.
3. **Multi-fault peeling.** Iterative identification of k₁*, replace, re-run for k₂*. ~500 LoC with careful residual-violation handling.
4. **Calibration bootstrapping.** E[Δₖ | τ is meaning-preserving] requires bootstrapping from non-faulty executions.
5. **N1 discriminability matrix.** Pre-test rank check preventing wasted computation. Independent utility.

The full M4 is a ~2-week implementation for an expert, not a weekend hack. But M4 alone does not carry a best paper.

### What was eliminated

The 18K-LoC Rust grammar compiler — the single hardest subproblem — was correctly killed. What remains is known techniques (SBFL, delta debugging, unification grammars, causal intervention) in new combination. The difficulty is in the multi-domain synthesis, not in any individual technique being frontier-hard.

### The Rust shrinker question

The Skeptic argued the Rust shrinker (7K LoC) is premature optimization. The panel partially agrees — for the convergence bound O(|T|²·|R|) where |T|≤40 and |R|≤200, ~320K grammar checks are feasible in Python at ~32 seconds. The Rust investment buys speed (~1 second) at the cost of 11.5K LoC complexity. The Rust is defensible for the artifact (users want fast shrinking) but not for the paper (the convergence proof doesn't require Rust).

### Score justification

Expert-level engineering synthesis without research-frontier difficulty. The individual techniques are known; the combination is non-trivial but predictable. Score 6 reflects "genuine multi-domain challenge" reduced from the prior depth check's 7 by the grammar compiler elimination.

---

## Axis 3: Best-Paper Potential — 5/10

### Math portfolio (honest assessment)

| Result | Status | Novel? |
|--------|--------|--------|
| M4 (Causal-Differential Localization) | Validated heuristic, novel in domain synthesis | Yes — the introduction-vs-amplification distinction via interventional analysis on typed NLP IRs is new. But the techniques (SBFL, do-calculus) are textbook. |
| N1 (Stage Discriminability Matrix) | Clean formalization | Moderate — standard linear algebra applied to NLP. The rank-as-testability-diagnostic idea has independent utility but is one paragraph to state and prove. |
| N4(a,b,d) (Shrinking Hardness/Convergence) | Careful extension | Low-moderate — NP-hardness reduction is standard; convergence bound extends delta debugging to new constraint domain; shrinking ratio involves distributional assumptions. |
| N3-simplified (DCE/IE Formalization) | Formalization, trivially true | No — restates the definition of interventional identification. |
| N2 (Information-Theoretic Bounds) | Contingent, ~40% failure risk | Yes if proved — the only result with genuine mathematical depth. |

**True math contribution count: 1.5 genuine contributions without N2 (M4 + half-credit for N4), 2.5 with N2.**

### The "predictable approach" problem

The Skeptic's strongest attack: "SBFL applied to NLP pipeline stages with a causal intervention — anyone in SE testing would predict this in 30 seconds." The panel acknowledges this is partially true. However, historical evidence shows predictable approaches win best papers at ISSTA/ASE when execution is superb (e.g., DeepTest won ISSTA 2019 with "metamorphic testing for autonomous driving" — an obvious idea, brilliantly executed).

### What could elevate the paper

1. **"10 Bugs, 10 Words" table** — 10+ real bugs in spaCy/HuggingFace, each localized and minimized. This is the kind of artifact reviewers put in talks and cite. But has 30% failure risk.
2. **GPT-4-as-debugger baseline** — showing the tool outperforms GPT-4 on cascading faults answers the strongest objection.
3. **Predictability defense** — showing vanilla SBFL without causal refinement achieves <65% top-1 while the refined approach achieves ≥85%.
4. **N2 proof** — would elevate from tools track to research track.

### Score justification

One genuine methodological contribution (M4), one solid domain adaptation (M5/N4), and a potentially memorable evaluation artifact. ISSTA/ASE accept probability ~65–70%. Best-paper probability ~3–5%, which is 1–2× the base rate for a strong submission. The score rises to 7 only if N2 is proved AND Tier 1 yields ≥10 genuine bugs — both uncontrollable variables.

---

## Axis 4: Laptop CPU + No Humans — 7/10

### What works

| Pipeline Class | Time per Sentence | 5K Tests | Feasibility |
|---|---|---|---|
| spaCy statistical (en_core_web_sm) | ~2ms | ~20s | ✓ Comfortable |
| spaCy transformer (en_core_web_trf) | ~200ms | ~33 min | ✓ Feasible |
| HuggingFace BERT-based | ~100ms | ~17 min | ✓ Feasible |
| Multi-stage transformer | ~500ms | ~83 min | ✓ Tight |
| RAG with local LLM | ~30–60s | ~125 hours | ✗ Out of scope |

Per-run execution is fully automated. The composition theorem reduces test count 10× (from ~50K to ~5K). Total CPU budget for 5-tier evaluation: ~25–35 hours. Feasible on a modern laptop over a weekend.

### The 300-seed dispute (RESOLVED)

The Skeptic scored CPU at 4, claiming 300 hand-crafted seeds = human annotation. **The panel rejects this.** By the standard definition used throughout the SE testing literature:
- **Human annotation** = human judgment applied to each test case/result, scaling linearly with test count.
- **Tool configuration** = human expertise applied once, amortized over all runs.

The 300 seeds are unambiguously tool configuration — comparable to writing CheckList templates, TextFlint configs, or QuickCheck generators. Every testing tool requires comparable setup.

### Residual concerns

- Memory budget for HuggingFace transformer IR snapshots on 8–16GB laptops is not quantified.
- Trigram perplexity naturalness proxy is acknowledged as weak.
- Amortized human costs (seeds, transformations, MR definitions, feature checker) are real and should be documented.

### Score justification

Genuinely automated per-run execution with honest amortized setup costs. CPU feasibility is well-calibrated for classical and transformer pipelines. RAG correctly scoped out. Score 7 reflects a small discount from 8 for unquantified memory budget and amortized human costs.

---

## Axis 5: Feasibility — 6/10

### What supports confidence

- 107K→40.5K LoC reduction eliminates the highest-risk component (18K LoC grammar compiler).
- Python-first + Rust-acceleration via PyO3 is a proven pattern (pydantic, tokenizers, polars).
- Staged rollout (spaCy → HuggingFace → Stanza stretch) contains adapter risk.
- Minimum viable paper at ~18K Python LoC is a genuine safety net.
- All prior fatal flaws (F-A1 through F-C2) have been addressed.
- Per-component LoC estimates are credible and internally consistent.

### What concerns exist

- **theory_bytes=0, impl_loc=0** — zero lines of code exist. Every feasibility claim is projection.
- **Tier 1 bug yield** — 30% chance of <5 actionable bugs (uncontrollable).
- **Feature-checker scope creep** — 80 constraints × 15 clause types may interact unpredictably.
- **15 transformations** — each 150–250 lines of dense linguistic logic. TextFlint's comparable transformations average 300–500 lines and were developed by teams over months.
- **No prototype validates the core loop** — the fundamental approach has not been demonstrated.

### Probability estimates

- P(minimum viable paper, ~18K LoC): ~80%
- P(full system, ~40.5K LoC, all 5 tiers): ~40%
- P(N2 proved with non-vacuous constants): ~55%
- P(Tier 1 ≥ 10 real bugs): ~50%
- P(full system + N2 + ≥10 bugs): ~15%

### Score justification

Excellent plan, unstarted execution. The plan is concrete with realistic scope reductions and genuine fallback paths. But zero code exists, and the Skeptic's demand for a prototype before full commitment is correct. Score 6 reflects "well-planned but unvalidated."

---

## Genuinely Novel LoC Estimate

Of the ~40.5K total LoC:

| Category | LoC | Fraction |
|----------|-----|----------|
| **Genuinely novel algorithm** | ~6K | 15% |
| M4 fault localizer (causal-differential localization, interventional replay, multi-fault peeling, calibration) | ~4K | |
| N1 discriminability matrix diagnostic | ~500 | |
| GCHDD convergence logic within shrinker | ~1.5K | |
| **Novel domain adaptation** | ~10K | 25% |
| Grammar-aware shrinker with feature-unification oracle (extends TreeReduce) | ~7K Rust | |
| Feature-unification checker (new but scoped) | ~3K Rust | |
| **Known-technique engineering** | ~19.5K | 48% |
| 15 tree transductions (linguistic engineering, not algorithmic novelty) | ~5K | |
| Pipeline adapters (API integration) | ~8K | |
| Input generator (corpus-based, standard) | ~3K | |
| Counterexample DB + reports | ~2K | |
| PyO3 bridge | ~1.5K | |
| **Evaluation infrastructure** | ~5K | 12% |
| **Total** | **~40.5K** | **~16K novel (39%), ~24.5K known-technique (61%)** |

The 39% novelty ratio is healthy for a tool paper. Novel LoC is concentrated in the localizer (M4) and shrinker (M5) — the claimed contributions.

---

## True Math Contribution Count

| Result | Honest Status | Load-Bearing? |
|--------|--------------|---------------|
| **M4** | Validated heuristic, genuinely novel in domain synthesis | Essential — remove it and the system degrades to CheckList |
| **N1** | Clean formalization with independent utility | Important — pre-test diagnostic preventing wasted computation |
| **N4(a,b,d)** | Careful extension of known technique | Important — convergence guarantee makes shrinker trustworthy |
| **N3-simplified** | Formalization, trivially true by construction | Nice-to-have |
| **N2 (conditional)** | Genuine if proved, ~40% failure risk | Would be the crown jewel |

**Total: 1.5 genuine math contributions without N2, 2.5 with N2.**

---

## Fatal Flaws

### FLAW 1 (SERIOUS): M4's causal localization degrades on multi-fault scenarios

M4's argmax heuristic identifies the most salient fault, not necessarily the causal one. When multiple stages are simultaneously faulty (common in practice — tokenizer + parser both struggle with passivization), the interventional analysis produces misleading introduction/amplification labels because replacing stage k's output doesn't isolate stage k when stage k-2 is also faulty. The multi-fault extension (iterative peeling) is described but not formally analyzed.

**Impact:** Top-1 accuracy may drop to <70% on multi-fault scenarios. The introduction-vs-amplification distinction — "the contribution no existing tool provides" — becomes unreliable.
**Severity:** Moderate-high. Testable at G1 prototype gate.

### FLAW 2 (SERIOUS): Tier 1 bug yield is uncontrollable with 30% failure risk

The paper's most compelling artifact ("10 Bugs, 10 Words") depends on finding ≥10 actionable bugs in mature, well-tested frameworks. Whether these bugs exist is an empirical fact about specific models, not a consequence of the tool's quality. Furthermore, "behavioral inconsistency" ≠ "bug" — finding 25 cases where a 90%-accurate model makes mistakes on out-of-distribution inputs is statistically inevitable, not a discovery.

**Impact:** If Tier 1 fails, the paper loses its centerpiece. The fallback (Tier 2 + GPT-4 baseline) is publishable but drops from "strong accept" to "borderline."
**Severity:** High (uncontrollable).

### FLAW 3 (MODERATE): Zero implementation creates maximum uncertainty

theory_bytes=0, impl_loc=0, code_loc=0. The project is entirely on paper. No prototype validates the core localization loop. No transformation has been implemented. Every feasibility and performance claim is speculative.

**Impact:** All scores carry ±2 point uncertainty.
**Severity:** Moderate. Normal for pre-implementation stage but demands early validation (BA-6).

### FLAW 4 (MODERATE): Feature-checker is a hidden grammar fragment

The 18K-LoC grammar compiler was replaced by a 3K-LoC feature-unification checker. But 80 constraints across 15 clause types, implemented in Rust with unification semantics, is still a grammar fragment. If constraints interact unexpectedly (and they will — that's what grammars do), debugging exhibits the same scope-creep dynamics that killed Approach A, at smaller scale.

**Mitigation:** Hard cap at 3.5K LoC. Calibrate against ≥10K parsed sentences by week 6. Accept false negatives over false positives.

---

## Salvage from Abandoned Approaches

### What was correctly cut
- Full grammar compiler (18K LoC Rust) — tar pit risk saved the project
- N3(b,c) observational identifiability — locally-linear assumption wrong for discrete NLP stages
- N4(c) inapproximability — gap-preserving reduction unverified
- N5 submodularity — theorem in search of a use case
- RAG/LLM pipeline support — CPU-infeasible, technically underspecified

### What could be salvaged
1. **Boltzmann sampling over corpus-extracted subtrees** (~1.5K LoC Python, not Rust) — partially recovers generative capacity from Approach A's grammar compiler without the risk. Lightweight tree-substitution grammar extracted from Penn Treebank addresses the corpus bias problem.
2. **spaCy as pre-filter before Rust checker** — rejects 80% of invalid candidate reductions in microseconds before hitting the formal checker. Doesn't affect convergence proof; just speeds up search.
3. **N2 lower bound only** (~2 weeks, not 4) — even a loose lower bound (m ≥ ln(n)/max-divergence) is a novel observation. Preserves "information-theoretically motivated" positioning at half the time investment.
4. **BFI as visualization metric** (not formal contribution) — define with floored denominator, present in behavioral atlas heatmaps. Don't claim formal properties.

### What should NOT be salvaged
- Full grammar compiler — risk/reward still terrible at any scale
- N3(b,c) — the model is wrong for NLP
- N4(c) — inapproximability without PCP is likely unprovable
- 40 transformations — 15 covers the major syntactic axes; the remaining 25 are rare, contested, or redundant

---

## Binding Amendments

Prior amendments BA-1 through BA-5 have been applied. This evaluation adds:

### BA-6: Build a 2-week prototype before full commitment (MANDATORY)

Implement M4 as a standalone Python script (~1.5K LoC) against spaCy `en_core_web_sm` with 3 transformations (passivization, synonym substitution, negation insertion). No shrinker, no grammar checker, no Rust. Validate: (a) real bugs exist, (b) localization works on real pipelines, (c) causal refinement beats vanilla SBFL on at least one cascading fault. **If this prototype finds zero bugs and localization doesn't beat random guessing, ABANDON.**

### BA-7: Scope generality claims to evaluation coverage (MANDATORY)

The paper may discuss "composed inference systems" as motivation and future work. The paper must not claim the tool works for arbitrary multi-stage systems unless at least one non-NLP evaluation is included. If the broader framing is desired as a primary claim, add one non-NLP evaluation (e.g., sklearn pipeline).

### BA-8: Drop N2 unless proof progress by week 2 (MANDATORY)

Attempt the lower bound (Fano's inequality) in weeks 1–2. If no clean proof sketch exists by end of week 2, abandon N2 entirely and redirect effort to evaluation quality.

---

## Phase Gates

| Gate | Timing | Criterion | Action if FAIL |
|------|--------|-----------|----------------|
| **G1: Prototype validates** | Week 2 | M4 prototype finds ≥1 real bug in spaCy AND localization beats random on ≥3/5 injected cascading faults | **ABANDON** |
| **G2: Bug yield pre-screen** | Week 4 | ≥3 actionable bugs from passivization + NER + sentiment on spaCy | Expand to exotic transformations + older models. If still <3 by week 6, **pivot** to Tier 2+3 paper |
| **G3: N2 proof checkpoint** | Week 2 | Clean proof sketch of lower bound on paper | Abandon N2, redirect to evaluation |
| **G4: Feature-checker calibration** | Week 6 | Passes ≥95% of 10K parsed sentences AND rejects ≥90% of synthetically corrupted ones | Relax to spaCy-parser proxy for edge cases |
| **G5: Shrinker convergence** | Week 8 | ≥80% of test cases shrink to <10 words within 60s | Accept weaker guarantees; report honest ratios |
| **G6: Localization accuracy** | Week 10 | Top-1 ≥80% on 50 injected faults AND ≥15pp improvement over best vanilla SBFL on cascading faults | **ABANDON** or rewrite as negative result |
| **G7: Minimum viable paper** | Week 12 | At least Tier 2 + Tier 3 complete with positive results | **ABANDON** |

---

## Outcome Probabilities

| Outcome | Probability |
|---------|-------------|
| Best paper at ISSTA/ASE | ~3% |
| Strong accept (research track) | ~12% |
| Accept (tools or research track) | ~45% |
| Workshop/demo/short paper | ~18% |
| Unpublishable | ~22% |

---

## Cross-Critique Resolution Summary

| Disagreement | Winner | Impact |
|--------------|--------|--------|
| M4 trivially reproducible? | **Split.** Skeptic wins "trivially conceivable." Auditor/Synthesizer win "non-trivially implementable." | Best-Paper stays 5, not 3 |
| Market contraction fatal? | **Auditor.** Contracting but not dead. Reframing valid as framing only. | Value at 5, not 3 or 6 |
| 300 seeds = annotation? | **Auditor/Synthesizer.** Standard tool configuration by testing literature definition. | CPU at 7, not 4 |
| Feasibility (0 code) | **Split.** Plan quality supports 7; zero code discounts to 6. | Feasibility at 6 |
| Best-paper potential | **Auditor.** Strong submission, not best-paper, unless N2 + bug yield luck. | Best-Paper at 5 |

The Independent Auditor was closest to correct across all five disagreements. The Skeptic provided the most important actionable recommendation: **build the prototype first** (adopted as BA-6). The Synthesizer's most valuable contributions: broader framing (adopted as BA-7), behavioral atlas promotion, and salvageable elements from abandoned approaches.

---

## Expert Signoff

| Expert | Verdict | Key Condition |
|--------|---------|---------------|
| Independent Auditor | CONDITIONAL CONTINUE | All binding amendments applied; scores may improve post-prototype |
| Fail-Fast Skeptic | ABANDON → CONDITIONAL CONTINUE | Accepts G1 gate as compromise; demands prototype validation before investment |
| Scavenging Synthesizer | CONTINUE | Broader framing + atlas promotion maximize impact |

**Panel Recommendation: CONDITIONAL CONTINUE with binding amendments BA-6 through BA-8 and 7-gate kill chain.**

**The project should proceed to BA-6 (2-week prototype) immediately. All further investment is gated on G1.**

---

*Evaluation produced by 3-expert adversarial verification panel with Panel Moderator synthesis. All scores reflect post-cross-critique convergence. Binding amendments and phase gates are non-negotiable for phase advancement.*

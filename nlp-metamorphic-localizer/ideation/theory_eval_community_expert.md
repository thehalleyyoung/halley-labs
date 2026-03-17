# Community Expert Verification: nlp-metamorphic-localizer (proposal_00)

**Evaluator:** NLP Community Expert (area-034-natural-language-processing)
**Stage:** Post-theory verification
**Date:** 2026-03-08
**Method:** 3-expert adversarial panel (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) with independent proposals, adversarial cross-critique, convergence synthesis, and independent verifier signoff.
**Documents reviewed:** `problem_statement.md`, `final_approach.md`, `depth_check.md`, `verification_signoff.md` (prior), `math_depth_assessment.md`, `approach_debate.md`, `approaches.md`, proposals A/B/C, `theory/approach.json`

---

## Executive Summary

**Composite: 6.6/10 — CONDITIONAL CONTINUE**

The Hardened Causal Pipeline Localizer fills a genuine, verified gap: no existing NLP testing tool (CheckList, TextFlint, TextAttack, LangTest, METAL, LLMORPH) provides pipeline-stage fault localization for multi-stage NLP inference systems. The causal-differential localization algorithm (M4) — distinguishing fault *introduction* from *amplification* via interventional replay on typed heterogeneous intermediate representations — is a genuine methodological contribution that does not exist in the SBFL literature. The stage discriminability matrix (N1) is an elegant diagnostic primitive. The grammar-constrained shrinking convergence (N4) solves a novel constraint optimization problem.

However, the addressable market (classical + transformer multi-stage NLP pipelines, ~15–25% of production NLP in 2025) is contracting under LLM monoculture. The "10 Bugs, 10 Words" centerpiece artifact has a 30–50% probability of underdelivering. The approach — SBFL + causal intervention on NLP pipeline stages — is methodologically predictable, even if the synthesis is genuinely novel in execution. Six binding conditions gate continued work, with the week-1 pre-screen as the decisive experiment.

---

## Converged Scores

| Axis | Auditor | Skeptic | Synthesizer | **Converged** | Justification |
|------|---------|---------|-------------|---------------|---------------|
| 1. Extreme Value | 6 | 3 | 8 | **6** | Real gap, real audience, contracting market |
| 2. Genuine Difficulty | 7 | 5 | 7.5 | **6.5** | Multi-domain synthesis; ~10–11K genuinely novel LoC |
| 3. Best-Paper Potential | 6 | 3 | 8 | **6.5** | Strong accept conditional on results; ~10–20% best-paper probability |
| 4. Laptop-CPU + No Humans | 8 | 5 | 8 | **7** | Fully automated per-run; upfront seed cost; transformer shrinking slower than claimed |
| 5. Feasibility | 7 | 5 | 8.5 | **7** | Grammar compiler risk eliminated; ~43.5K adjusted LoC; parse-error propagation is real but mitigated |

**Composite: 6.6/10** (unweighted average)

---

## Axis 1: Extreme Value — 6/10

### What's strong

The capability gap is real and verified. No existing NLP testing tool distinguishes "the NER is wrong" from "the parser corrupted the NER's input." The qualitative shift — from end-to-end pass/fail (CheckList-level) to per-stage causal diagnosis with minimal proof sentences — is a change in *kind*, not degree. For regulated-industry NLP teams (healthcare entity extraction, legal clause classification, financial compliance NER), the ability to produce structured evidence ("14,000 grammar-valid metamorphic variants tested, 7 inconsistencies localized to specific stages, each with a minimal proof sentence") transforms compliance auditing from anecdotal to systematic.

### What's weak

**The market is contracting.** The John Snow Labs 2023 survey ("60% of NLP teams spend more time debugging pipeline interactions than training models") is three years old and predates the 2024–2025 LLM shift. Multi-stage NLP pipelines at 15–25% of production NLP and declining. Most HuggingFace production deployments are single-model, not multi-stage. RAG/LLM pipelines — the growth segment — are explicitly out of scope (non-deterministic stages, CPU-infeasible, heterogeneous IR distances). The Skeptic's estimate of 10–60 teams worldwide who would actually adopt this tool is aggressive but directionally correct for *adoption*; the paper's *citation audience* is broader (anyone working on pipeline testing, metamorphic testing, or fault localization).

**Generalization claim is undemonstrated.** The Synthesizer correctly identifies that M4 is domain-agnostic in principle (any pipeline with observable typed IRs), but the proposal contains zero evidence of application beyond NLP. The reframing to "causal fault localizer for compositional inference systems" is aspirational, not evidenced.

**The compliance angle is the strongest market argument.** EU AI Act transparency requirements (effective 2025–2026) and FDA SaMD guidance are tightening. Multi-stage pipelines persist in regulated domains precisely because interpretability requirements make monolithic LLMs non-compliant. This niche is stable or growing under regulatory pressure.

### Score justification

V6 reflects genuine differentiation (no competitor provides this capability) for a real but narrow and contracting audience. Not "extreme and obvious" (V8+ would require a growing market or demonstrated generalization); not negligible (V3 conflates tool adoption with paper audience). The compliance niche is the strongest anchor.

---

## Axis 2: Genuine Software Difficulty — 6.5/10

### What's genuinely hard

**Multi-domain synthesis.** The project requires simultaneous competence in formal methods (delta debugging convergence proofs, NP-hardness reductions, feature unification), computational linguistics (subcategorization frames, English passivization morphology, agreement systems), software testing (SBFL, metamorphic testing, causal analysis), and systems engineering (PyO3 FFI, parse tree manipulation). No single component is frontier-difficult; the synthesis across four domains is genuinely hard for a single developer.

**Token-to-tree alignment across interventions.** When passivization changes word order and introduces new tokens, the original parser's dependency tree doesn't align with the transformed token sequence. Interventional replay requires feeding alignment-inconsistent representations to downstream stages. The final approach handles this via lemma-level alignment for 11/15 transformations, with explicit alignment maps for 4 others and statistical fallback when alignment fails. Getting this right for 15 transformations × 2 frameworks is where months disappear.

**Grammar-aware shrinking with three-way constraint satisfaction.** The shrinker must simultaneously maintain (a) grammatical validity via feature-unification, (b) transformation applicability, and (c) metamorphic violation preservation. This three-way constraint satisfaction is novel — TreeReduce handles CFGs for programming languages but not unification-based linguistic constraints.

**Feature-checker calibration.** Too strict → non-minimal counterexamples. Too permissive → ungrammatical outputs destroy credibility. The Goldilocks zone for ~80 feature constraints across ~15 clause types requires both linguistic depth and careful engineering.

### What's less hard than it sounds

**Known techniques in new combination.** SBFL is standard. Delta debugging is standard. Unification grammars are standard. An expert could sketch the high-level approach in an afternoon; the difficulty is in making it work correctly, not in conceiving it.

**Novel algorithmic LoC is ~10–11K of ~40.5K total.** The causal intervention logic in M4 (~2K novel), grammar-constrained shrinking in N4 (~5K novel), feature-checker (~2–3K novel), N1 discriminability matrix (~500 novel). The remaining ~30K is engineering (pipeline adapters, evaluation harness, transformations, infrastructure). The 15 tree transductions encode deep syntactic knowledge (passivization alone requires auxiliary conjugation, ditransitive handling, irregular participles, modal interactions) but are engineering-hard, not algorithmically-hard.

**Rust necessity is overestimated for the feature-checker.** The cross-critique's math shows: the feature-checker handles O(10⁵) constraint checks per shrink operation; Python handles this in ~1 second while pipeline execution dominates at 25–2,560 seconds. Rust buys ~2–3× end-to-end speedup for statistical pipelines and ~1.1× for transformer pipelines. However, the feature-checker's deterministic oracle property (needed for M5's convergence proof) requires principled implementation regardless of language — it is load-bearing for the paper's formal claims even if not for performance.

### Score justification

D6.5 reflects genuinely hard multi-domain synthesis with ~10–11K novel LoC, tempered by the observation that the high-level approach is predictable and a simpler 5–10K LoC prototype could demonstrate ~80% of the value (per-stage diff + string-level shrinking, missing only the causal refinement and grammaticality guarantees that are the intellectual contribution).

---

## Axis 3: Best-Paper Potential — 6.5/10

### What's strong

**M4 is a genuine methodological contribution.** The introduction-vs-amplification distinction via interventional analysis on typed NLP pipeline IRs does not exist in the SBFL literature. Classical SBFL operates on binary statement coverage; NLP pipeline stages produce typed, structured, high-dimensional intermediate representations where "coverage" means linguistic feature processing. The synthesis is novel even if the components are standard.

**The "10 Bugs, 10 Words" table is a reviewer magnet.** If 10+ real bugs in spaCy/HuggingFace are found, localized, and minimized, this is the kind of result reviewers put in keynote talks. Compact, verifiable, immediately actionable. The predictability defense (vanilla SBFL <65% on cascading faults vs. ≥85% with causal refinement) demonstrates the "obvious approach" fails without the refinement.

**The GPT-4-as-debugger baseline has bounded downside.** If the tool wins (expected case: GPT-4 at 40–60% on structured causal reasoning tasks), it's a devastating differentiator. If GPT-4 hits 70%+, the tool still provides deterministic reproducibility, minimal counterexamples, coverage guarantees, and audit trails that GPT-4 cannot.

**N1 (discriminability matrix) is the dark horse.** The Synthesizer correctly identifies that the rank-check diagnostic ("can your transformation set distinguish all pipeline stages?") has "why didn't I think of that?" energy. It's cheap to compute (~100 calibration samples), immediately actionable, and connects to compressed-sensing measurement-matrix design theory. This could be the paper's sneaky most-cited result.

### What's weak

**The "predictable approach" problem.** A senior SE testing researcher reads the abstract: "SBFL + causal intervention for NLP pipeline stages." Their reaction: "Natural idea. The question is whether the evaluation justifies the obvious approach." This is not fatal (many best papers are careful applications of known techniques to important new domains) but it sets a high bar for execution quality that depends on empirical outcomes the project cannot guarantee.

**Competition from LLM testing papers.** At ISSTA/ASE 2026, this paper competes against LLM testing, AI code generation verification, and autonomous agent testing — all addressing the dominant paradigm. A paper about classical NLP pipeline debugging reads as archaic unless the contribution is framed as generalizable (causal fault localization for compositional inference systems, with NLP as the proving ground).

**Bug yield is the major uncontrollable variable.** 30% probability of <5 actionable bugs (proposal's self-assessment); the Skeptic estimates 50%+. Without the "10 Bugs, 10 Words" table, the paper loses its centerpiece and pivots to a tool paper with synthetic evaluation only — strong accept territory, not best-paper. Furthermore, some "bugs" may be expected model behavior (training-data distribution issues, not software defects); the operational definition of "actionable inconsistency" (MR-invariant + severity threshold + reproducibility across ≥3 inputs) is the critical defense.

**N2 (information-theoretic bounds) has ~40% failure risk.** Without N2, the math portfolio is M4 + N1 + N4(a,b,d) + N3-simplified — solid but not exceptional. N2 would elevate the paper from tools track to research track, but the proof timeline is risky.

### Score justification

BP6.5 reflects strong accept potential at ISSTA/ASE (~60–70% acceptance probability) with best-paper probability of ~10–20%. The package (M4 + N1 + "10 Bugs" table + GPT-4 comparison) is compelling if empirical results land. Best-paper requires ≥10 real bugs AND either N2 completion or a dramatically surprising evaluation finding — joint probability ~10–20%.

---

## Axis 4: Laptop-CPU Feasibility & No-Humans — 7/10

### What works

**Statistical pipelines (spaCy `en_core_web_sm`): ~75 min for full test-localize-shrink cycle.** At ~2ms/sentence, 5,000 test cases take ~20 seconds. Localization adds ~2 minutes. Shrinking top-20 violations at ~60 seconds each adds ~20 minutes. Interactive debugging speed.

**Transformer pipelines (HuggingFace BERT): ~3–4 hours.** At 200–500ms/sentence, 5,000 test cases take ~17–83 minutes. Shrinking is slower but within the 60-second-per-counterexample budget if the feature-checker filters effectively (verifier's analysis: 48s–10min per counterexample depending on checker pass rate). Explicitly documented as CI nightly, not interactive.

**RAG/LLM honestly excluded.** CPU-infeasible (125+ hours for 5K tests on quantized 7B). Non-deterministic stages violate the causal intervention framework's assumptions. Correctly scoped to future work.

**Zero per-run human involvement.** Generation (corpus-based), transformation (tree-transduction), oracle checking (task-parameterized MR predicates), localization (causal-differential analysis), shrinking (grammar-constrained delta debugging), evaluation (automated metrics) — all fully automated once configured.

### What's tight

**300 hand-crafted seeds = 2.5–5 person-days of expert linguistic effort.** This is a one-time amortized cost (<5% of project effort), analogous to writing test infrastructure, not per-run human involvement. But it requires a trained linguist, which is a hidden dependency.

**Naturalness proxy is weak.** Trigram perplexity is a known-bad metric for grammaticality/naturalness. Full validation requires human judges — ruled out. The proposal is honest about this limitation.

**Transformer shrinking time is slower than the headline "60 seconds" suggests.** The proposal's "60-second budget per counterexample" constrains shrinking but may not achieve 1-minimality within the budget for transformer pipelines. The independent verifier's analysis corrects the cross-critique: at well-calibrated feature-checker pass rates (1–5%), shrinking takes 48s–10min per counterexample for transformers — feasible for nightly CI but not interactive.

### Score justification

CPU7 reflects genuine per-run automation on laptop CPU for the stated scope. Docked from 8 for: (a) upfront seed-authoring cost (2.5–5 person-days of expert effort), (b) transformer shrinking is honestly 1–10 minutes per counterexample, not 60 seconds, (c) naturalness cannot be validated without human judges.

---

## Axis 5: Feasibility — 7/10

### What's strong

**Grammar compiler risk is eliminated.** This was the single largest risk identified in the depth_check (Flaw 5: "A probabilistic unification grammar for English is a multi-year research project on its own"). The final approach replaces the 18K-LoC grammar compiler with a ~3–4K LoC feature-unification checker scoped to exactly the constraints needed by 15 transformations. The verification_signoff confirms: "This is the single most important design decision and it is correct."

**LoC estimate is realistic.** The final approach's ~40.5K (24K Python + 11.5K Rust), adjusted to ~43.5K by the cross-critique (accounting for adapter complexity, feature-checker sizing, and edge cases), is achievable in 4–5 months with staged rollout. The per-component estimates are credible: 15 transformations × ~200 lines = ~3K, adapters for 2 frameworks at ~4K each = ~8K, Rust shrinker at ~7K, feature-checker at ~3–4K.

**Staged rollout contains risk.** spaCy first (month 1–2) → HuggingFace (month 2–3) → transformer pipelines (month 3–4). Failure at any stage limits scope but doesn't kill the project.

**Minimum viable paper is credible.** ~18K LoC Python (M4 only, spaCy adapter, 8 transformations, Tier 2 evaluation, GPT-4 baseline). Publishable as a tool contribution at ISSTA/ASE even without the shrinker, atlas, or real bug table.

**Math portfolio is achievable.** M4 (HIGH — "The algorithm is implementable"), N1(a,b) (HIGH — "one-paragraph proofs"), N4(a) (HIGH — "straightforward reduction"), N4(b) (HIGH for base O(|T|²·|R|) bound), N3-simplified (HIGH — "a definition + one observation"). Only N2 is HIGH RISK, and it's correctly on a conditional two-track strategy.

### What's risky

**15 linguistically-grounded transformations are dense code.** Each encodes deep syntactic knowledge with many edge cases (passivization: auxiliary conjugation across tenses, irregular participles, ditransitives, modals, idiom resistance). Getting even one transformation subtly wrong undermines the metamorphic relation's preservation claim, producing false positives.

**Parse-error propagation.** spaCy's parser accuracy (~91–95% LAS) means 5–9% of dependency arcs in seed sentences are incorrect. Transformations operating on wrong parses can produce ungrammatical sentences (caught by feature-checker), semantically altered sentences (not caught), or correct transformations of wrong parses (producing false-positive "bugs"). The corpus-based approach (PTB gold parses for base sentences, verified seeds) mitigates this substantially — effective error rate ~6% for corpus-derived inputs.

**Feature-checker calibration.** The checker must be simultaneously tight enough to prevent ungrammatical output and loose enough to allow effective shrinking. ~80 feature constraints across ~15 clause types must be calibrated against real English, with both false-positive and false-negative rates tracked.

**Tier 1 bug yield.** 30% probability of <5 actionable bugs. The pre-screen (week 1) catches this early. The fallback to injected-fault evaluation (Tiers 2–4 only) preserves publishability but loses the "10 Bugs, 10 Words" centerpiece.

### Score justification

F7 reflects a feasible project with the largest risk eliminated (grammar compiler) and honest engineering risks remaining (transformation correctness, parse-error propagation, feature-checker calibration, bug yield). The staged rollout and minimum viable fallback demonstrate engineering maturity.

---

## Fatal Flaws

### Skeptic's Proposed Fatal Flaw: "The problem is disappearing faster than the tool can be built"

**Classification: Valid but not fatal.**

The market contraction argument is directionally correct: multi-stage NLP pipelines are declining from ~20–25% to ~10–15% of production NLP by 2027. But:

1. **Academic papers target niches.** CheckList (Ribeiro et al., EMNLP 2020) tested NLP models when LLMs were already ascendant; it has 1,700+ citations. Academic contributions are evaluated on intellectual merit and methodological novelty, not addressable market size.

2. **The regulated-industry niche is persistent.** Healthcare, legal, and financial NLP teams will not adopt black-box LLMs for critical pipelines within 2–3 years. Regulatory requirements for interpretability, reproducibility, and deterministic inference anchor these teams to multi-stage architectures. The EU AI Act and FDA SaMD guidance are *expanding* testing requirements.

3. **The contribution generalizes in principle.** M4 applies to any multi-stage inference pipeline with observable typed IRs — ML feature pipelines, data processing DAGs, compiler passes. The NLP instantiation is the proof-of-concept; the idea is broader. However, this generalization is currently undemonstrated, capping value at V6.

4. **The market contraction correctly *bounds* the project's upside** — reflected in V6 (not V8), BP6.5 (not BP8) — but does not *kill* it. A solid ISSTA/ASE paper with genuine methodological contributions, a compelling evaluation, and a useful tool is achievable even with a narrowing audience.

### Remaining Risks (Not Fatal)

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Bug yield <5 | 30–50% | High (loses centerpiece artifact) | Week-1 pre-screen; Tier 2+3+GPT-4 fallback paper |
| Feature-checker miscalibration | 20% | Medium (weakens M5 guarantee) | 10K-sentence calibration; spaCy-as-oracle fallback |
| Token-to-tree alignment failures >30% | 25% | Medium (weakens M4 scope) | Restrict causal refinement to 11 aligned transformations |
| N2 proof failure | 40% | Medium (loses crown jewel) | Paper stands on M4 + N1 + N4; N2 is conditional bonus |
| Parse-error-propagated false positives | 15% | Low-Medium | Gold-parse seeds, arc-confidence filtering |
| Model version drift breaks Tier 1 reproducibility | 10% | High (DOA at review) | Pin versions with SHA256 hashes; Docker image |

**No fatal flaws identified.** All risks are manageable, have mitigations, and degrade gracefully.

---

## Binding Conditions

| # | Condition | Deadline | Failure Action |
|---|-----------|----------|---------------|
| BC-1 | **Week-1 pre-screen: ≥3 genuine bugs** across 3+ transformations on 2 pipelines. "Genuine" = MR-invariant violation + severity threshold + reproducible ≥3 inputs. | Week 1 | Expand scope in week 2; if still <3 after week 2 across all transformations and all 4 pipelines, pivot to synthetic-evaluation-only paper |
| BC-2 | **Python-first MVP.** Implement feature-checker and shrinker in Python first. Defer Rust to week 6–8 (Rust feature-checker required for M5's convergence proof in final paper). | Week 2 | N/A (this is the plan) |
| BC-3 | **Feature-checker calibration: <5% FP and <15% FN** on 10K PTB sentences + 1K out-of-distribution sentences. | Week 3 | Fall back to spaCy-as-oracle; lose M5 convergence proof; adjust paper claims |
| BC-4 | **FP tracking from day 1.** Classify every inconsistency as genuine-bug / expected-behavior / parse-error-artifact. Halt at 20% cumulative FP; tighten to 10% for final paper. | Continuous | Re-calibrate MR specifications before continuing |
| BC-5 | **N2 checkpoint.** Decide include/exclude based on factorization lemma progress. | Week 4 | Exclude N2; paper stands on M4 + M5 + N1 + N4 |
| BC-6 | **Pin model versions.** Every Tier 1 bug reproducible against pinned model version in Docker with deterministic seeds. SHA256 hashes for all model files. | Before Tier 1 | Not optional — reproducibility is a hard requirement |

---

## Recommendations (Non-Binding)

| # | Recommendation | Priority | Source |
|---|---------------|----------|--------|
| R1 | **Reframe paper as "causal fault localizer for compositional inference systems"** — NLP as the proving ground, not the ceiling. Include a Discussion section on generalization to ML pipelines, compiler passes, data workflows. | High | Synthesizer |
| R2 | **Promote N1 (discriminability matrix) to §3.1** — before M4 localization. "Before localizing, verify localization is *possible*." This has "why didn't I think of that?" energy and may be the paper's most-cited result. | High | Synthesizer |
| R3 | **Lead the abstract with the GPT-4 comparison.** "Our tool achieves X% localization accuracy on cascading faults; GPT-4's best prompting strategy achieves Y%." This is the sentence that gets the paper read. | High | Synthesizer + Auditor |
| R4 | **Implement ADAPTIVE-LOCATE as Thompson-sampling heuristic** regardless of N2 fate. Principled stopping criterion + optimal test selection + budget awareness. No existing testing tool provides these. | Medium | Synthesizer |
| R5 | **Track alignment coverage per transformation class.** Report which transformations preserve vs. disrupt token-lemma alignment. Important for paper completeness. | Medium | Auditor |
| R6 | **Rust shrinker as post-MVP optimization** only if Python shrinking >5 min/CE on statistical pipelines. Feature-checker stays Python for MVP. | Low | Cross-critique |
| R7 | **Add arc-confidence pre-filter** (~50 lines Python). Only apply transformations to seed sentences where spaCy's parser confidence exceeds threshold for relevant dependency arcs. Mitigates parse-error propagation. | Medium | Verifier |
| R8 | **Elevate behavioral atlas to secondary contribution** with DOI. First benchmark for multi-stage NLP pipeline fault localization. | Low | Synthesizer |

---

## NLP Community Expert Assessment

As someone who evaluates ideas by whether practitioners in NLP would find them genuinely valuable and exciting:

### Would NLP practitioners care?

**Mixed.** The NLP community in 2026 is LLM-focused. A paper about debugging spaCy/HuggingFace multi-stage pipelines will not generate excitement at ACL/EMNLP — it addresses yesterday's architecture. However, at ISSTA/ASE (the correct venue), the paper addresses a genuine gap in the SE testing literature with a novel technique (causal-differential localization with typed NLP IRs) that has clear intellectual merit. The regulated-industry NLP subfield (healthcare, legal, financial) would find the compliance documentation angle genuinely valuable. The "10 Bugs, 10 Words" artifact, if achieved, would be widely cited as a benchmark.

### Is this a "shrug" paper?

**Not if executed well.** The risk of a "shrug" response comes from: (a) the "predictable approach" problem (SBFL + intervention is a 30-second idea), and (b) the declining-paradigm positioning. The defense against (a) is the ablation study showing vanilla SBFL fails at <65% on cascading faults. The defense against (b) is the reframing to "causal fault localization for compositional systems" with NLP as the hardest-case proving ground.

### What would make this exciting to the community?

1. **A genuinely surprising bug** — not "NER misses entities in passive voice" (known) but "the tokenizer's Unicode normalization silently corrupts rare entity spans, causing cascading failures that only manifest when the dependency parser receives unnormalized input." A bug with a non-obvious causal chain that the tool uniquely identifies.
2. **GPT-4 embarrassment** — showing that GPT-4 consistently misdiagnoses cascading faults (attributing symptoms to the wrong stage) while the tool's interventional analysis correctly identifies the root cause.
3. **The discriminability matrix revealing a fundamental limitation** — e.g., "no transformation set of size <10 can distinguish all stages of spaCy's 5-stage pipeline" — a structural impossibility result from N1.

---

## Verdict

### **CONDITIONAL CONTINUE at 6.6/10**

| Axis | Score |
|------|-------|
| Extreme Value | 6 |
| Genuine Difficulty | 6.5 |
| Best-Paper Potential | 6.5 |
| Laptop-CPU + No Humans | 7 |
| Feasibility | 7 |
| **Composite** | **6.6** |

The project has a genuine methodological contribution (M4 causal-differential localization), a sound architecture (grammar compiler eliminated, scope right-sized to ~40.5K LoC), an elegant formalization (N1 discriminability matrix), and a well-designed evaluation plan with honest fallbacks. The risks (market contraction, bug yield uncertainty, feature-checker calibration) are real but manageable under the 6 binding conditions.

The week-1 pre-screen (BC-1) is the single most important experiment. If ≥3 genuine bugs are found in 5 days, the project's empirical foundation is validated and the Skeptic's strongest attack (bugs are expected model behavior) is refuted. If <3 bugs are found after 2 weeks of expanded testing, the project pivots to synthetic-evaluation-only paper — still publishable but not best-paper competitive.

**Proceed with Python-first MVP, binding conditions BC-1 through BC-6, and adopt recommendations R1–R3 (high priority) for paper framing.**

---

## Team Disposition

| Expert | Individual Verdict | Individual Composite | Key Contribution |
|--------|-------------------|---------------------|-----------------|
| Independent Auditor | CONDITIONAL CONTINUE | 6.8 | Evidence-based scoring; 5 binding conditions |
| Fail-Fast Skeptic | CONDITIONAL ABANDON | 4.2 | Market-collapse argument; Rust necessity analysis; bug-quality challenge |
| Scavenging Synthesizer | CONTINUE | 7.8 | Generalization reframing; N1 as dark horse; ADAPTIVE-LOCATE salvage |
| Cross-Critique | CONDITIONAL CONTINUE | 6.6 | Resolved all axis disagreements; math on Rust/feature-checker; converged conditions |
| Independent Verifier | SIGNOFF ✅ | 6.6 confirmed | Corrected shrinking-time estimate; added BC-6; identified 3 missing concerns |

**Consensus: CONDITIONAL CONTINUE (4-1, Skeptic dissents with conditions for upgrade)**

---

*Evaluated by NLP Community Expert panel. All scores reflect post-debate convergence with independent verifier signoff. 6 binding conditions gate continued work.*

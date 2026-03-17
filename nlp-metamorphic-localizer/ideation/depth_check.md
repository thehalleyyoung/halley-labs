# Verification Depth Check: nlp-metamorphic-localizer

**Title:** "Where in the Pipeline Did It Break? Metamorphic Fault Localization for Multi-Stage NLP Systems"
**Stage:** Post-crystallization verification
**Date:** 2026-03-08
**Method:** 3-expert adversarial panel (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) with independent proposals, cross-critique, and synthesis.

---

## Converged Scores

| Axis | Auditor | Skeptic | Synthesizer | **Converged** | Threshold |
|------|---------|---------|-------------|---------------|-----------|
| 1. Extreme & Obvious Value | 5 | 6 | 7 | **6** | ≥7 |
| 2. Genuine Difficulty | 6 | 7 | 8 | **7** | ≥7 |
| 3. Best-Paper Potential | 5 | 4 | 6 | **5** | ≥7 |
| 4. Laptop CPU + No Humans | 7 | 6 | 7 | **7** | ≥9 |

**Composite: 6.25/10** (unweighted average)
**Verdict: CONDITIONAL CONTINUE — 5 binding amendments required.**

---

## Axis 1: EXTREME AND OBVIOUS VALUE — 6/10

### What's strong

The core value proposition is genuine and differentiated. No existing NLP testing tool (CheckList, TextFlint, TextAttack, LangTest, METAL, LLMORPH) provides pipeline-stage fault localization. The qualitative shift — from "the pipeline fails on passivized inputs" to "the POS tagger mishandles passivized gerunds, causing parser misattachment, cascading to NER; proof: 'The report was being written by Kim'" — is a category change in debugging capability. The "thermometer vs. MRI" analogy is memorable and accurate.

For teams in regulated industries (healthcare NLP, legal entity extraction, financial compliance), the ability to produce structured evidence — "14,000 grammar-valid metamorphic variants tested, 7 inconsistencies localized to specific pipeline stages, each with a minimal proof sentence" — transforms compliance auditing from anecdotal to systematic.

### What's missing

**Market contraction.** The addressable market for classical multi-stage NLP pipelines (spaCy, Stanza) is contracting. An estimated 15–25% of production NLP in 2025 still runs classical multi-stage pipelines, concentrated in regulated industries. The John Snow Labs 2023 survey ("60% of NLP teams spend more time debugging pipeline interactions than training models") is two years old and predates the LLM monoculture shift of 2024–2025.

**RAG extension is technically underspecified.** The problem statement claims RAG pipeline applicability but provides only two sentences of architectural analogy. RAG stages produce heterogeneous, high-dimensional, stochastic outputs where per-stage differentials (Δₖ) are ill-defined. LLM stages are non-deterministic, breaking the causal intervention framework's assumption of near-deterministic stages. The evaluation plan mentions only spaCy, Stanza, and HuggingFace — RAG pipelines appear nowhere in the five evaluation tiers.

**Single-model deployments.** Most HuggingFace production deployments are single-model (BERT-NER, BERT-sentiment), not multi-stage, reducing the addressable market for pipeline-stage localization.

### Score justification

Real value for a real but narrowing audience. The classical/transformer pipeline market is shrinking but not dead. The RAG extension inflates perceived market without delivering technical substance. A 6 reflects genuine differentiation (no competitor provides this) tempered by honest market assessment.

---

## Axis 2: GENUINE DIFFICULTY — 7/10

### What's strong

Multi-domain expertise is genuinely required: formal grammars (unification, feature structures), causal reasoning (SBFL, interventional analysis), NLP internals (pipeline architecture, intermediate representations), and property-based testing (metamorphic relations, delta debugging). Few researchers combine all four.

The five hard subproblems are real:
1. **Grammar Compiler with Dual Compilation:** Tomabechi/Wroblewski DAG unification with occurs-check, dual automaton compilation (generation + shrinking), morphological transduction via FSTs. Genuinely hard — English is massively ambiguous and context-sensitive.
2. **40 Linguistically-Grounded Transformations:** Each encodes deep syntactic knowledge (active→passive alone requires auxiliary conjugation, object promotion, PP demotion, ditransitive handling, idiom resistance).
3. **Causal-Differential Fault Localizer:** SBFL + interventional analysis, distinguishing fault introduction from amplification across pipeline stages with typed intermediate representations.
4. **Grammar-Aware Shrinker:** Parse-tree delta debugging maintaining grammaticality and transformation applicability at every step. Novel constraint optimization problem.
5. **Pipeline Instrumentor:** Copy-on-write IR snapshots with delta compression, multi-framework adapters, PyO3 bridge.

### What's weakened

**LoC inflation.** The claimed 193K LoC is inflated ~1.8×. All three experts independently estimated the true implementation at ~100–120K LoC. Converged estimate: **~105K LoC** (~50–55K algorithmic core, ~35–40K engineering infrastructure, ~15–20K evaluation harness). The inflation undermines credibility without adding genuine difficulty.

**Linguistic engineering ≠ algorithmic depth.** Building correct passivization is hard because English is irregular, not because the algorithm is deep. The difficulty is real but primarily engineering complexity, not intellectual frontier-pushing. The genuinely hard *algorithmic* problems (M4 causal localization, M5 grammar-constrained shrinking convergence) represent ~30–35K LoC of the ~105K total.

**Could a simpler approach work?** A 25–35K LoC Python prototype could demonstrate ~80% of the value proposition. It would miss the causal introduction-vs-amplification distinction, grammar-constrained minimality guarantees, and formal coverage optimization — which are precisely the intellectual contributions. The remaining 20% is the actual research contribution.

### Score justification

A 7 reflects genuine multi-domain difficulty and non-trivial algorithmic synthesis, tempered by the observation that the project applies known methods (SBFL, delta debugging, unification grammars) to a new domain rather than inventing fundamentally new techniques. An expert who reads the problem statement could sketch the high-level approach in an afternoon; the difficulty is in making it *work correctly*, not in *conceiving* it.

---

## Axis 3: BEST-PAPER POTENTIAL — 5/10

### What's strong

**M4 (Causal-Differential Fault Localization)** is a genuine methodological contribution. The interventional analysis that distinguishes fault introduction from amplification does not exist in the SBFL literature because classical statement-level fault localization has no notion of "typed intermediate outputs that can be partially correct." NLP pipeline stages *do* have such outputs, making the amplification concept well-grounded and the contribution domain-genuine.

**M5 (Grammar-Constrained Shrinking Convergence)** is a solid domain adaptation. The O(|T|² · |R|) convergence bound for delta debugging over parse trees with unification-based grammaticality constraints extends TreeReduce to a harder constraint domain (linguistic features vs. programming language syntax). A clean, minor-but-real technical contribution.

**The "10 real bugs, each ≤10 words" table** would be a reviewer magnet at ISSTA/ASE. If 10 bugs in spaCy/HuggingFace/Stanza are found, localized, and minimized, this is the kind of result people put in talks and cite.

### What's weakened

**The "predictable approach" problem.** The Skeptic's strongest attack: "SBFL applied to NLP pipeline stages with a causal intervention — anyone in SE testing would predict this in 30 seconds." This is not fatal (many best papers are careful applications of known techniques to important new domains) but it sets a high bar for execution quality.

**Mathematical novelty is oversold.** M3 (Composition Theorem) in statistical form is essentially "independent transformations compose probabilistically" with Clopper-Pearson intervals — an empirical observation with error bars, not a theorem. M7 (BFI) is a ratio metric — useful but trivial as mathematics. Claiming "four genuinely new mathematical contributions" when only M4 and M5 are genuine invites scrutiny the results cannot survive.

**Competition from LLM testing papers.** At ISSTA/ASE 2025–2026, this paper competes against LLM testing, AI code generation verification, and autonomous agent testing — all addressing the dominant paradigm (LLMs) directly. This paper addresses a declining paradigm (classical multi-stage NLP) with a promised extension to the dominant paradigm (RAG) that is technically underspecified.

**Evaluation targets are optimistically calibrated.** Tier 1 (≥25 "behavioral inconsistencies") conflates model accuracy limitations with software defects — finding 25 cases where a statistical model produces different outputs for syntactically different inputs is trivial. Tier 2 (85% on injected faults) is a low bar for synthetic evaluation. Tier 4 (10× efficiency over uniform random) is expected for covering arrays.

### Score justification

One genuine methodological contribution (M4), one solid domain adaptation (M5), and potential for a memorable evaluation artifact. A well-scoped paper focused on M4 + M5 + a compelling bug table is a solid ISSTA/ASE submission (~60–70% acceptance probability). But best-paper (top 1–2%) requires either a fundamentally surprising result or a dramatically compelling evaluation that the project design cannot guarantee. The 5 reflects "strong accept" potential without "best paper" potential.

---

## Axis 4: LAPTOP CPU + NO HUMANS — 7/10

### What works

| Pipeline Class | Inference Time | 5K Tests Total | Feasibility |
|---|---|---|---|
| spaCy statistical (`en_core_web_sm`) | ~2ms/sentence | ~20 seconds | ✓ Comfortable |
| spaCy transformer (`en_core_web_trf`) | ~200ms/sentence | ~33 minutes | ✓ Feasible |
| HuggingFace BERT-based | ~100ms/sentence | ~17 minutes | ✓ Feasible |
| Multi-stage transformer | ~500ms/sentence | ~83 minutes | ✓ Tight (nightly CI) |
| RAG with local 7B LLM | ~30–60s/inference | ~125 hours | ✗ Infeasible |

Grammar-compiled generation in Rust (thousands of sentences/second) is not a bottleneck. The composition theorem reduces test count from O(|T|²) to O(|T|·log|R|), making 5K tests sufficient for ≥90% pairwise coverage. The budget-constrained adaptive scheduler (multi-armed bandit) is a sound engineering solution.

The ~75-minute total cycle for statistical pipelines and ~3–4 hour cycle for transformer pipelines are honest and documented.

### What doesn't work

**RAG on CPU is infeasible.** A quantized 7B model at 30–60s/inference × 5,000 tests × 2 runs = 125+ hours. Even with aggressive budget constraints (500 tests), that's 12+ hours for testing alone before localization and shrinking. The proposal never quantifies RAG inference time — a conspicuous omission.

**"No humans" understates expert effort.** The grammar specification, 40 transformation implementations, task-parameterized MR definitions, and pipeline topology configurations all require expert human authoring *before* execution. These are amortized one-time costs, not per-run annotation, but they represent substantial human effort. More honest framing: "no per-run human involvement."

**Naturalness proxy is weak.** Trigram perplexity as a naturalness proxy has high variance and is a known-bad metric. Human evaluation of a sample would be stronger but is ruled out.

### Score justification

A 7 reflects genuine CPU feasibility for the core use case (classical + transformer pipelines) with honest timing budgets. The RAG claim must be scoped out (see BA-2). Memory constraints (8–16GB laptop RAM) may limit concurrent IR snapshot storage for large pipelines but are manageable with delta compression.

---

## Axis 5: Fatal Flaws

### FLAW 1 (SERIOUS): RAG extension is technically underspecified and CPU-infeasible

The problem statement's relevance argument rests heavily on RAG/agent pipeline applicability, but:
- Per-stage differentials are ill-defined for heterogeneous RAG stages (ranked document lists, scores, prompt strings, free-text LLM output)
- The causal intervention framework assumes near-deterministic stages; LLMs are stochastic
- LLM inference on CPU is orders of magnitude too slow for systematic testing (125+ hours for 5K tests)
- The evaluation plan mentions only spaCy, Stanza, and HuggingFace — no RAG evaluation exists
- The "generic adapter for REST APIs" hides enormous engineering problems (no intermediate representation access for black-box LLM APIs)

**Severity:** Fatal if retained as core scope. Non-fatal if downgraded to future work (BA-2).

### FLAW 2 (SERIOUS): "Behavioral inconsistency" ≠ "bug"

The Tier 1 evaluation targets ≥25 "previously unknown behavioral inconsistencies." But:
- A statistical NER model that produces different entity labels for active vs. passive voice may be exhibiting *expected model behavior*, not a software defect
- Finding 25 cases where a 93%-accurate model makes mistakes is statistically inevitable, not a discovery
- Framework maintainers (spaCy, HuggingFace) will not fix model accuracy limitations filed as bug reports
- The false positive rate target of <15% means ~4 of 25 reported "bugs" are actually correct pipeline behavior

**Severity:** High. The headline evaluation result collapses if the "bugs" are uninteresting model accuracy limitations. Needs operational definitions distinguishing actionable defects from statistical model behavior (BA-4).

### FLAW 3 (MODERATE): LoC inflation undermines credibility

193K claimed vs. ~105K converged estimate is ~1.8× inflation. This signals either optimism bias or deliberate padding and invites skepticism of other claims.

**Severity:** Moderate. Fixable (BA-1). The project at ~105K LoC is still genuinely large and impressive.

### FLAW 4 (MODERATE): Mathematical novelty is oversold

M3 (Composition Theorem) in statistical form is an empirical observation with confidence intervals. M7 (BFI) is a ratio metric. Claiming "four genuinely new mathematical contributions" when only M4 and M5 are genuine invites scrutiny the results cannot survive.

**Severity:** Moderate. Fixable by reframing M3 and M7 as "formal specifications" rather than "novel mathematics."

### FLAW 5 (MODERATE): Grammar compiler complexity risk

A probabilistic unification grammar for English is a multi-year research project on its own (the LKB, XTAG, and ERG are decade-long efforts). Risk of getting stuck in grammar engineering — chasing agreement errors, handling exceptions, covering the long tail of English syntax — is very high.

**Severity:** Moderate. Mitigated by scoping the grammar to only the constructions needed by 15 core transformations (~10–12K LoC, not a full English grammar).

### NON-FLAWS (adequately addressed in the problem statement)

- The circularity concern (causal analysis uses original execution as reference) is explicitly acknowledged and mitigated via fault-injection ground truth
- The Rust+Python split is justified by profiling reality (grammar ops need speed, NLP libraries are Python-native)
- The statistical form of M3 is an honest weakening from an impossible deterministic guarantee
- The 1-minimality vs. full minimality tradeoff is correctly scoped

---

## Binding Amendments

These changes **must** be made to the problem statement before proceeding.

### BA-1: Deflate LoC to ~105K with itemized justification

Restate the LoC estimate as ~105K with a revised table:

| Category | LoC | Subsystems |
|----------|-----|------------|
| Algorithmic core | ~50–55K | Grammar Compiler (~18K), Fault Localizer (~12K), Shrinker (~12K), Transformation Algebra (~12K) |
| Engineering infrastructure | ~35–40K | Pipeline Instrumentor (~15K), Input Generator (~8K), Coverage Analyzer (~6K), Counterexample DB (~8K), Test Scheduler (~4K) |
| Evaluation harness | ~15–20K | Evaluation orchestration (~10K), Spec Language (~6K) |
| **Total** | **~105K** | |

### BA-2: Downgrade RAG from "supported" to "architecturally compatible, future work"

- State that the instrumentor *architecture* supports RAG stages as pipeline components
- Explicitly acknowledge that LLM-stage non-determinism, black-box API constraints, and CPU infeasibility are open problems
- Move RAG to a "Future Extensions" section, not the core value proposition
- Remove all claims that RAG is a primary target of the engine

### BA-3: Specify CPU feasibility budget per pipeline class

Replace blanket "laptop CPU feasible" with:
- Statistical pipelines (spaCy, Stanza): ~75 min for full cycle (interactive)
- Transformer pipelines (HuggingFace BERT-based): ~3–4 hours (nightly CI)
- RAG/LLM pipelines: **not CPU-feasible**; requires GPU or API budget; out of core scope

### BA-4: Operationalize "meaning-preserving" per task and define "bug" vs. "expected behavior"

For each supported task, provide the concrete predicate:
- **NER:** Entity span boundaries and labels must be identical (modulo pronoun/name substitution in the transformation)
- **Sentiment:** Polarity label must be identical; confidence score may change by ≤ε (task-specific threshold)
- **Dependency parsing:** Labeled attachment of argument relations must be preserved (adjunct attachment may change)
- **Text classification:** Class label must be identical

Define the distinction between "actionable inconsistency" and "expected model behavior": an inconsistency is actionable if (a) the metamorphic relation explicitly defines the tested dimension as invariant, AND (b) the violation exceeds a task-specific severity threshold, AND (c) the violation is reproducible across ≥3 distinct input sentences.

### BA-5: Define minimum viable system as go/no-go milestone

The smallest system that validates the core thesis:
- M4 (causal localization) + M5 (grammar-constrained shrinking)
- Single pipeline adapter (spaCy)
- 15 transformations (not 40)
- Tiers 1+2 evaluation only
- Estimated: ~50–60K LoC
- This is the go/no-go deliverable; everything beyond is incremental

---

## Recommended (Optional) Improvements

1. **R-1: Reduce transformations from 40 to 15 for initial release.** Keep: passivization, clefting, topicalization, relative clause insertion/deletion, voice change, tense change, agreement perturbation, synonym substitution, negation insertion, coordinated NP reordering, PP attachment variation, adverb repositioning, there-insertion, dative alternation, embedding depth change. Cut exotic transformations (pseudocleft, extraposition from NP, tough-movement).

2. **R-2: Add "predictability defense" to paper framing.** Show that vanilla SBFL without causal refinement achieves <65% localization accuracy, demonstrating the "obvious approach" fails and the refinement matters.

3. **R-3: Prioritize the "10 bugs, 10 words" artifact.** Make the table of real bugs with minimal counterexamples the centerpiece of the paper, not an afterthought.

4. **R-4: Add GPT-4-as-debugger baseline.** Compare localization accuracy against "just ask GPT-4 to debug it." If the engine wins (likely — GPT-4 cannot do interventional analysis), this is a powerful differentiator.

5. **R-5: Add regression test generation.** Automatically output minimal counterexamples as pytest test cases. Closes the loop from "bug found" to "bug prevented." ~1K LoC.

6. **R-6: Reframe M3 and M7 as formal specifications, not mathematical contributions.** Focus novelty claims on M4 and M5 only. Present M3 as a "coverage optimization tool" and M7 as an "interpretable metric."

---

## Expert Signoff

| Expert | Verdict | Condition |
|--------|---------|-----------|
| Independent Auditor | CONDITIONAL CONTINUE | All 5 binding amendments must be applied. Scores may improve to V7/D7/BP6/CPU8 post-amendment. |
| Fail-Fast Skeptic | CONDITIONAL CONTINUE | BA-1 and BA-2 are non-negotiable. If Tier 1 finds <5 actionable bugs (not just behavioral inconsistencies), recommend ABANDON. |
| Scavenging Synthesizer | CONTINUE | The diamond is real. Ship the minimum viable system (BA-5), write a focused paper on M4+M5+atlas, iterate. |

**Panel recommendation: CONDITIONAL CONTINUE with binding amendments BA-1 through BA-5.**

---

## Appendix: Converged LoC Estimates by Subsystem

| Subsystem | Original Claim | Auditor Est. | Skeptic Est. | Synthesizer Est. | Converged |
|-----------|---------------|-------------|-------------|-----------------|-----------|
| Grammar Compiler | 28,000 | 12–15K | 18K | 10–12K (scoped) | 18,000 |
| Transformation Algebra + MR | 26,000 | 12–14K | 18K | 15K (15 transforms) | 12,000 |
| Input Generator | 16,000 | — | 8K | — | 8,000 |
| Pipeline Instrumentor | 20,000 | 10–14K | 15K | 12K | 15,000 |
| Fault Localizer | 22,000 | 8–12K | 12K | 18K | 12,000 |
| Grammar-Aware Shrinker | 18,000 | 8–10K | 12K | 15K | 12,000 |
| Coverage Analyzer | 14,000 | — | 6K | — | 6,000 |
| Counterexample DB + Reports | 12,000 | — | 10K | 3K (reporting) | 8,000 |
| Spec Language | 8,000 | — | 6K | 0 (YAML) | 4,000 |
| Test Scheduler | 14,000 | — | 4K | 0 (simple) | 4,000 |
| Evaluation Harness | 15,000 | — | 10K | — | 8,000 |
| **Total** | **193,000** | **~90K** | **~119K** | **~70K** | **~107,000** |

---

*Assessment produced by 3-expert adversarial verification panel. All scores reflect post-cross-critique consensus. Binding amendments are non-negotiable for phase advancement.*

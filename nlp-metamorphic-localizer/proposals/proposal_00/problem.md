# Final Approach: The Hardened Causal Pipeline Localizer

**Project:** nlp-metamorphic-localizer
**Title:** "Where in the Pipeline Did It Break? Causal Fault Localization for Multi-Stage NLP Systems"
**Date:** 2026-03-08
**Method:** Synthesized from 3 competing approaches (Pipeline MRI, ITCL, Pragmatic Maximizer) after adversarial debate by 5-expert team.

---

## 1. One-Sentence Summary

A Python-first metamorphic testing engine with a Rust-accelerated shrinker that localizes behavioral inconsistencies in multi-stage NLP pipelines to the specific pipeline stage that introduced or amplified them, producing minimal proof sentences of ≤10 words via grammar-constrained delta debugging with a formal convergence guarantee.

---

## 2. Architecture and Design Decisions

### Language Split

| Component | Language | Justification |
|-----------|----------|---------------|
| Fault Localizer | Python | Core algorithm; operates on NLP pipeline outputs which are Python objects. No performance bottleneck — bounded by pipeline execution time. |
| Pipeline Adapters (spaCy, HuggingFace) | Python | NLP frameworks are Python-native. Instrumentation must hook into Python APIs. |
| 15 Transformations + MR Checker | Python | Tree transductions on spaCy/HuggingFace parse objects. ~150–250 lines each. Python suffices. |
| Evaluation Harness | Python | Orchestration, metrics, report generation. |
| **Grammar-Aware Shrinker** | **Rust (PyO3)** | Must explore millions of subtree reductions per second. Python is 50× too slow. 60-second budget per counterexample requires Rust speed. |
| **Feature-Unification Validity Checker** | **Rust (PyO3)** | Deterministic oracle for shrinking. Must answer "is this subtree replacement grammatical?" in O(|R|) time. Called millions of times during shrinking. |
| Test Input Generator | Python | Corpus-based; transforms existing parsed sentences. No performance bottleneck. |

**Total Rust surface:** ~8–12K LoC (shrinker ~6–8K + feature-checker ~2–3K + PyO3 bridge ~1K).
**Total Python surface:** ~22–28K LoC (localizer ~6K + adapters ~8K + transformations ~5K + generator ~3K + evaluation ~5K).

### Grammar Decision: Lightweight Feature-Unification Checker

We **reject** Approach A's full grammar compiler (tar pit risk — passivization alone needs 30–40 productions; grammar engineering has a 40-year track record of 3–10× scope overruns) and we **reject** Approach C's spaCy-parser-as-proxy (destroys M5's convergence guarantee because spaCy is a statistical model with no deterministic grammaticality oracle).

**Our solution:** A lightweight feature-unification checker (~2–3K LoC Rust) that handles:
- Subject-verb agreement (number, person) across the clause types needed by 15 transformations
- Subcategorization frames for transitive, ditransitive, unaccusative, and copular verbs
- Transformation-specific preconditions (e.g., passivization requires transitive-active; there-insertion requires indefinite-NP-subject + unaccusative/copular)
- Definiteness restrictions for there-insertion
- Complementizer selection for relative clauses and clefts

This checker is **not** a grammar. It cannot generate sentences. It answers one question: "is this candidate shrunk sentence grammatically valid with respect to the feature constraints that our 15 transformations require?" This is sufficient for M5's convergence proof (deterministic oracle) without the grammar compiler's scope risk. The checker covers ~80 feature constraints across ~15 clause types — scoped to exactly what the transformations need.

**For generation:** Corpus-based. Parse seed sentences from Penn Treebank (~40K sentences) augmented with **300 hand-crafted seed sentences** targeting rare constructions (there-insertions, cleft sentences, dative alternations, topicalizations, embedded clauses at depth ≥3). A linguist can write these in 1–2 days. Each seed sentence is annotated with which transformations are applicable. The generator selects seeds by transformation coverage targets, parses them via spaCy, applies transformations as tree transductions, and validates results via the feature-checker.

### Transformations: 15 Core

Passivization, clefting, topicalization, relative clause insertion, relative clause deletion, tense change, agreement perturbation, synonym substitution, negation insertion, coordinated NP reordering, PP attachment variation, adverb repositioning, there-insertion, dative alternation, embedding depth change. Each implemented as a tree transduction on dependency trees with precondition checking and postcondition validation. ~150–250 lines Python each. Total: ~4–5K LoC.

### Pipeline Adapters: Staged Rollout

1. **spaCy** (month 1–2): Well-documented Python objects, fast (~2ms/sentence for `en_core_web_sm`), clear stage boundaries (tokenizer → tagger → parser → NER). Primary development target.
2. **HuggingFace** (month 2–3): `dslim/bert-base-NER` + `textattack/bert-base-uncased-SST-2`. Different IR formats (PyTorch tensors vs spaCy `Doc` objects), WordPiece tokenization, transformer encoder as shared stage.
3. **Stanza** (stretch goal): Similar architecture to spaCy but different API.

Generic adapter interface: `PipelineAdapter` with methods `get_stages()`, `run_prefix(x, k)`, `run_from(ir_k, k)`, `get_ir(x, k)`, `distance(ir1, ir2, stage_type)`.

### LoC Estimate

| Component | LoC | Language |
|-----------|-----|----------|
| Fault Localizer (M4) | 6,000 | Python |
| Pipeline Adapters (spaCy + HuggingFace) | 8,000 | Python |
| 15 Transformations + MR Checker | 5,000 | Python |
| Input Generator (corpus-based) | 3,000 | Python |
| Grammar-Aware Shrinker (M5) | 7,000 | Rust |
| Feature-Unification Checker | 3,000 | Rust |
| PyO3 Bridge | 1,500 | Rust |
| Evaluation Harness | 5,000 | Python |
| Counterexample DB + Reports | 2,000 | Python |
| **Total** | **~40,500** | **~24K Python + ~11.5K Rust** |

### Artifact Delivery

pip-installable Python package (`nlp-localizer`) with optional Rust acceleration. Falls back to pure-Python shrinker (slower, weaker guarantees) if Rust extension fails to build. Docker image for reproducible evaluation.

---

## 3. Extreme Value and Target Audience

### Who Desperately Needs This

**Primary:** NLP reliability engineers at regulated-industry companies (healthcare entity extraction, legal clause classification, financial compliance NER) running multi-stage pipelines with 3–7 stages. These teams:
- Debug pipeline interactions by inserting print statements between stages (4–8 hours per incident)
- Cannot distinguish "the NER is wrong" from "the parser corrupted the NER's input"
- Must produce structured testing evidence for compliance audits (FDA, HIPAA, SOX)
- According to John Snow Labs' 2023 survey, 60% spend more time debugging pipeline interactions than training models

**Secondary:** NLP framework developers (spaCy, HuggingFace) who need systematic regression testing across pipeline stages when releasing new model checkpoints.

### What Becomes Possible

**Before:** "The pipeline fails on passive voice." (CheckList-level diagnosis.)
**After:** "The POS tagger mishandles passivized gerunds, causing the dependency parser to misattach PPs, cascading to NER. Minimal proof: 'The report was being written by Kim.' (8 words). Severity: the parser amplifies the tagger error 4.7×." (Stage-level causal diagnosis.)

The minimal counterexample becomes a permanent regression test. When a model checkpoint is swapped, the suite re-runs automatically. For regulated industries, this transforms "tested manually on 50 examples" into "tested on 5,000 grammar-valid metamorphic variants, localized 7 inconsistencies to specific stages, each with a minimal proof sentence."

### Honest Market Assessment

The addressable market (classical + transformer multi-stage NLP pipelines) is contracting as LLM monoculture grows. We target the ~15–25% of production NLP that still runs multi-stage pipelines, concentrated in regulated domains where deterministic inference, interpretability, and auditability are requirements. RAG/LLM pipeline support is future work (architecturally compatible but three open problems: non-deterministic stages, heterogeneous IR distances, GPU/API cost). We are building for a real but narrowing audience, and we are honest about it.

---

## 4. Technical Difficulty

### Hard Subproblem 1: Causal-Differential Localization Across Typed Heterogeneous IRs

Classical SBFL operates on binary statement coverage. NLP pipeline stages produce typed, structured, high-dimensional IRs: token sequences, POS tag sequences, dependency trees, entity span lists. Computing meaningful divergence requires type-specific distance functions calibrated so Δ=0.3 at the parser is comparable to Δ=0.3 at the NER. Naive per-stage comparison conflates correlation with causation: if tagger, parser, and NER all show high divergence, vanilla SBFL can't determine the causal chain.

The interventional step — replacing stage k's output with the original execution's output and observing downstream behavior — requires partial re-execution from intermediate checkpoints. This means: copy-on-write IR snapshots, type-compatible intervention injection, and handling the token-to-tree alignment problem (when passivization changes word order, the original parser's tree doesn't align with the transformed tokens).

**Token-to-tree alignment mitigation:** Define alignment at the *lemma* level (transformation-invariant for 11/15 transformations: passivization, clefting, topicalization, relative clause insert/delete, tense change, coordinated NP reorder, PP attachment, adverb repositioning, there-insertion, dative alternation). For the 4 transformations that change lemma inventory (synonym substitution, negation insertion, agreement perturbation, embedding depth change), maintain explicit transformation-specific alignment maps. For any transformation where alignment fails, fall back to statistical localization (Ochiai only, no causal refinement) and report this honestly.

### Hard Subproblem 2: Grammar-Aware Shrinking with Feature-Unification Oracle

When the engine finds a 40-word sentence exposing a fault, the developer needs a 5–8 word proof. String-level shrinking produces ungrammatical fragments. Tree-level shrinking (TreeReduce) works for CFGs but English requires unification-based feature checking. Our shrinker operates on parse trees, pruning subtrees while checking:
- (a) Grammatical validity via the feature-unification checker (agreement, subcategorization)
- (b) Transformation applicability (shrunk sentence must still be passivizable if passivization exposed the bug)
- (c) Metamorphic violation preservation (the bug must still manifest)

This three-way constraint satisfaction is novel. The feature-checker answers validity queries in O(|R|) time via precomputed compatibility tables. The shrinker uses hierarchical delta debugging on the derivation tree with binary search over subtree orderings.

### Hard Subproblem 3: 15 Linguistically-Grounded Tree Transductions

Each transformation encodes deep syntactic knowledge. Passivization alone requires: auxiliary `be` insertion with correct tense/aspect, object-to-subject promotion, subject demotion to `by`-PP, handling of ditransitives (two possible passivizations), irregular past participles, modal interactions, and idiomatic resistance detection. Each transformation: 150–250 lines of tree transduction with precondition guards. Total: 15 × ~200 = ~3K lines of dense linguistic logic.

### Hard Subproblem 4: Pipeline Instrumentor with Partial Re-Execution

The instrumentor must capture per-stage IRs, support copy-on-write snapshots (for memory efficiency on laptops), and enable partial re-execution from any intermediate checkpoint. For spaCy, this means intercepting `Doc` objects between pipeline components. For HuggingFace, this means capturing PyTorch tensors between model stages. The generic adapter must handle internal caching, tokenization boundary mismatches, and encoding scheme differences.

---

## 5. Mathematical Contributions

### Theorem 1 (M4): Causal-Differential Fault Localization [CORE, NEW]

**Formal Statement.** Given pipeline P = s₁ ∘ ... ∘ sₙ, input x, transformation τ, and end-to-end metamorphic violation, define per-stage differentials:

> Δₖ(x, τ) = dₖ(sₖ(prefixₖ(x)), sₖ(prefixₖ(τ(x))))

where dₖ is a type-specific distance function and prefixₖ denotes the pipeline prefix through stage k−1. Localize to:

> k* = argmax_k [Δₖ − E[Δₖ | τ is meaning-preserving]]

**Causal refinement:** For suspected stage k*, replace sₖ*'s input with prefixₖ*(x) (original execution's IR) and re-execute stages k* through n. If the violation disappears, k* *introduced* the fault. If it persists but attenuates, k* *amplified* a pre-existing fault.

**Multi-fault extension:** When multiple stages are simultaneously faulty, the argmax identifies the most salient. The interventional analysis is applied iteratively: after identifying k₁*, replace k₁*'s output, re-run localization on the residual pipeline to identify k₂*. This iterative peeling converges in at most n steps.

**Complexity:** O(N · n · C_pipeline) for spectrum-based localization, plus O(n · C_pipeline) per violation for interventional refinement. Linear in test count and stage count.

**What it enables:** Distinguishing "the parser is wrong" from "the parser is correctly propagating a tagger error." This introduction-vs-amplification distinction is the contribution no existing tool provides.

**Why load-bearing:** Remove it and the system degrades to end-to-end detection (CheckList) or uncalibrated per-stage comparison (naive SBFL). The interventional refinement is the δ between a thermometer and an MRI.

**Achievability:** High confidence. The algorithm is implementable. The formal framing as a validated heuristic (not a provably optimal oracle) is honest and robust to reviewer scrutiny. The circularity concern (original execution as reference) is acknowledged and mitigated via fault-injection ground truth.

**Novelty:** Yes. SBFL is standard, causal intervention is standard. Their synthesis for typed NLP pipeline IRs — where "coverage" means linguistic feature processing and intervention operates on structured intermediate representations — is novel. Closest prior art: SBFL for Java programs (Wong et al. 2016), which uses statement coverage, not pipeline-stage IR distances.

---

### Theorem 2 (N1): Stage Discriminability Matrix [NEW, from Approach B]

**Formal Statement.** Define the stage discriminability matrix M ∈ ℝⁿˣᵐ:

> M_{k,j} = E_{x∼G}[Δₖ(x, τⱼ)]

(a) T can localize faults to a unique stage if and only if rank(M) = n.
(b) If rank(M) = r < n, the best achievable localization partitions stages into n−r+1 equivalence classes of indistinguishable stages.

**What it enables:** Pre-test diagnostic completeness check. Before running 5,000 tests, compute M from a small calibration sample (~100 tests, <2 minutes on CPU) and verify rank(M) = n. If rank < n, report exactly which stages are indistinguishable and suggest transformations to add. This prevents wasting an entire CPU budget on tests that cannot localize.

**Why load-bearing:** Without this, the engine has no way to know if its transformation set can distinguish all stages. It might run thousands of tests probing the same stage boundary while others go unexamined.

**Achievability:** High confidence. Parts (a) and (b) are direct linear algebra (column space argument, one-paragraph proofs). Calibration data for M requires only stage-differential *means*, not distributions — easily estimable from 100–200 samples. The empirical conjecture that structural + lexical + morphological transformations achieve full rank for standard NLP pipelines (n ≤ 7) is validated experimentally, not claimed as a theorem.

**Novelty:** Moderate. Analogous constructions exist in compressed sensing and group testing. The instantiation for NLP pipeline fault localization is new. The definition is the contribution, not the proof.

---

### Theorem 3 (N4a,b,d): Grammar-Constrained Shrinking Hardness and Convergence [NEW, improved from Approach B]

**Formal Statement.**

(a) **NP-hardness.** Finding the globally shortest grammatical counterexample that preserves the metamorphic violation is NP-hard, by reduction from the Minimum Grammar-Consistent String problem.

(b) **1-minimality convergence.** GCHDD (Grammar-Constrained Hierarchical Delta Debugging) produces a 1-minimal counterexample in at most O(|T|² · |R|) feature-checker invocations, each costing O(|F|). 1-minimal: no single grammar-valid subtree replacement further reduces the input while preserving grammaticality, transformation applicability, and the metamorphic violation. (If the binary-search monotonicity lemma is proved during development, this tightens to O(|T| · log|T| · |R|).)

(d) **Expected shrinking ratio.** For grammars with bounded ambiguity α and average branching factor b, E[|x'|] ≤ |x|/b + O(α · log|x|). For typical NLP (b ≈ 3, α ≤ 5): ~3–5× reduction (40 words → 8–13 words).

**What it enables:** (a) justifies targeting 1-minimality instead of global minimality. (b) guarantees the shrinker terminates and produces actionable counterexamples. (d) sets realistic user expectations.

**Why load-bearing:** Without (a), a reviewer asks "why not find the shortest?" Without (b), the shrinker has no convergence guarantee — it could loop or produce non-minimal output. Without (d), the "10 words" in "10 Bugs, 10 Words" has no theoretical backing.

**Achievability:** High confidence for (a) — straightforward reduction. High confidence for (b) with O(|T|²·|R|) — careful but routine extension of delta debugging. Moderate confidence for the O(|T|·log|T|·|R|) improvement — requires proving monotonicity lemma.

---

### Formal Specification (N3-simplified): Causal Introduction vs Amplification [FORMALIZATION]

**Statement.** Model the pipeline as a structural causal model. The Direct Causal Effect of transformation τ at stage k is:

> DCE_k(τ) = E[Δₖ | do(S^τ_{k−1} := S_{k−1})]

DCE_k > 0 indicates stage k *introduces* a fault (its output diverges even when given correct input). The Indirect Effect IE_k = E[Δₖ] − DCE_k measures fault *amplification*.

**Interventional sufficiency:** DCE_k is always identifiable from a single interventional replay at stage k. Total cost: O(n · C_pipeline) per violation.

This is stated as a formalization of the introduction-vs-amplification distinction, not as an identifiability theorem. We do **not** claim observational identifiability (the locally-linear sufficient condition fails for discrete NLP stages). The engine always performs interventional replay; this formalization makes the concepts precise.

---

### Crown Jewel

**Theorem 1 (M4)** — the causal-differential fault localization with introduction-vs-amplification distinction via interventional analysis on typed NLP intermediate representations. This is the single result that separates the tool from every existing NLP testing system.

### Contingency Crown Jewel (Two-Track Strategy)

In parallel with implementation, attempt **N2: Information-Theoretic Localization Bounds** (from Approach B). If proved with non-vacuous constants for n ≤ 7 by a 4-week proof checkpoint, include as the crown jewel that elevates the paper from tools track to research track. If not, relegate to future work with computational evidence supporting the conjecture. The tool (ADAPTIVE-LOCATE as Thompson-sampling heuristic) is built regardless.

### What We Cut and Why

| Result | Verdict | Reason |
|--------|---------|--------|
| M3 (Composition) | Mention in 2 sentences | Standard statistics, no novelty claim |
| M7 (BFI) | Define as metric in system section | Problematic denominator, not load-bearing |
| N2 (Sample Complexity) | Conditional on 4-week checkpoint | ~40% proof failure risk; C(T) estimation infeasible in high dimensions |
| N3(b,c) (Observational Identifiability) | Cut | Locally-linear condition wrong for discrete NLP stages |
| N4(c) (Inapproximability) | Cut | Requires gap-preserving reduction that may not hold |
| N5 (Submodularity) | Cut | Known result; theorem in search of a use case |

---

## 6. Best-Paper Strategy

### Narrative Arc

1. **Opening hook:** "An NLP engineer discovers that passivizing a sentence flips an entity label. CheckList tells her the pipeline is broken. But which of 5 stages broke it — and did stage 3 introduce the fault or merely amplify one from stage 2? No existing tool answers this. We build the one that does."

2. **Technical contribution:** Formalize pipeline fault localization. Introduce the discriminability matrix (N1) — show that transformation sets can be diagnosed for completeness before running tests. Present causal-differential localization (M4) with the introduction-vs-amplification distinction. Present grammar-constrained shrinking with convergence bounds (N4).

3. **The predictability defense:** Show that vanilla SBFL without causal refinement achieves <65% top-1 accuracy on cascading faults. The "obvious" approach fails. The refinement that makes it work is where the technical depth lies.

4. **The killer table:** "10 Bugs, 10 Words" — 10+ real bugs in spaCy and HuggingFace, each localized to a specific stage, each with a minimal counterexample ≤10 words. This is the artifact reviewers put in their talks and cite for years.

5. **The timely comparison:** GPT-4-as-debugger baseline. Same pipeline topology, transformation, and end-to-end failure — ask GPT-4 to localize. On cascading faults, our tool outperforms the strongest informal alternative. "This tool does something an LLM cannot."

6. **The community resource:** Behavioral atlas mapping (transformation × pipeline × task) → behavioral outcomes. Cross-system findings: "transformer-based NER is 3.2× more fragile to passivization than statistical NER."

### Target Venue

ISSTA or ASE (primary). ICSE tools track (secondary). If N2 is proved, ICSE research track becomes viable.

### Artifact Strategy

pip-installable package with optional Rust acceleration. Docker image for evaluation reproducibility. All 15 transformations and MR specifications as reusable modules. Atlas released as public SQLite database.

---

## 7. Evaluation Plan

### Tier 1: Real Bug Discovery

Run on spaCy `en_core_web_sm`, `en_core_web_trf`, and HuggingFace `dslim/bert-base-NER` + `textattack/bert-base-uncased-SST-2`. Target: **≥10 previously unknown, reproducible actionable inconsistencies** (honest target — not 25), with ≥3 per framework.

**Actionable inconsistency definition:** (a) MR explicitly defines violated dimension as invariant for the task; (b) violation exceeds task-specific severity threshold (entity span mismatch, polarity flip — not confidence fluctuation); (c) reproducible across ≥3 distinct inputs.

**Pre-screen (week 1):** Run passivization + NER + sentiment checks on spaCy before committing. If <3 bugs found in 2 days, expand to exotic transformations and older model versions (spaCy 2.x known to have more edge cases).

### Tier 2: Localization Accuracy

50 fault-injected pipeline configurations with known ground-truth faulty stages. Measure:
- Top-1 accuracy: target ≥85%
- Top-2 accuracy: target ≥95%

**Baselines (5):** Random stage selection (~20%), last-stage heuristic, vanilla SBFL (Ochiai), vanilla SBFL (DStar), vanilla SBFL (Barinel).

**Key comparison:** Show causal refinement provides ≥20 percentage-point improvement over best vanilla SBFL on **cascading faults** (where fault propagates through ≥2 stages). This is the "predictability defense" — the obvious approach fails, the refinement matters.

**GPT-4-as-debugger baseline:** Give GPT-4 the same pipeline topology, transformation, end-to-end failure, and — in the strongest variant — stage-level intermediate representation excerpts. Test 5 prompt strategies (zero-shot, few-shot, chain-of-thought, IR-provided, full-context). Report **best-of-5** as GPT-4 upper bound. Target: our tool outperforms GPT-4's best prompt by ≥15 points on cascading faults.

**Cross-validation with Tier 1:** For every real bug discovered in Tier 1, manually verify whether the localizer's stage diagnosis matches the ground-truth root cause (determined by inspecting intermediate representations). Report concordance rate as external validity estimate.

### Tier 3: Shrinking Quality

- Shrink ratio: target ≥5× (40 words → ≤8 words average)
- Shrinking time: target <60 seconds per counterexample on CPU
- Grammaticality: 100% of shrunk sentences must parse correctly via spaCy
- Naturalness proxy: trigram perplexity within 2× of Penn Treebank sentences
- Transformation applicability: 100% of shrunk sentences remain valid inputs to the exposing transformation

### Tier 4: Coverage Efficiency

Statistical composition theorem enables ≥90% pairwise (transformation × seed-category) coverage with ≤5,000 test cases vs ≥50,000 for uniform random and stratified random. ≥10× efficiency gain.

### Tier 5: Behavioral Atlas (Bonus)

Public database: (transformation × pipeline × task) → behavioral outcome. ≥3 cross-system findings. Released as SQLite artifact. This is a bonus, not a core contribution.

### Fallback

If Tier 1 yields <5 bugs: pivot paper to Tier 2 + Tier 3 + GPT-4 baseline as standalone tool contribution. "Causal Fault Localization for NLP Pipelines: An Interventional Approach." Still publishable at ISSTA/ASE tools track.

---

## 8. Risk Analysis

### Top 3 Risks

| # | Risk | Prob | Impact | Mitigation |
|---|------|------|--------|------------|
| 1 | **Real bug yield too low** — <5 actionable bugs in Tier 1 | 30% | High | Pre-screen week 1; expand to older model versions; fallback to Tier 2+3+GPT-4 paper |
| 2 | **Interventional replay alignment fails** for >20% of test cases | 25% | Medium | Restrict causal refinement to 11 lemma-aligned transformations; statistical fallback for 4 others; report honestly |
| 3 | **Feature-checker is too restrictive** — rejects valid shrinkings, producing non-minimal counterexamples | 20% | Medium | Test checker against 10K spaCy-parsed sentences to calibrate; add fallback to spaCy-parser proxy for edge cases |

### Minimum Viable Paper

**~18K LoC Python.** M4 only (causal-differential localization). spaCy adapter only. 8 transformations. Tier 2 evaluation (50 injected faults, 4 SBFL baselines, GPT-4 baseline). No shrinker, no atlas, no real bug table. Still publishable as tool contribution because the causal refinement over vanilla SBFL is genuine and undemonstrated.

### Scope Cuts Preserving Core Contribution

| Cut | LoC Saved | What's Lost | Core Preserved? |
|-----|-----------|-------------|-----------------|
| Drop Stanza adapter | 3K | One framework in evaluation | ✅ |
| Reduce 15 → 10 transformations | 1.5K | 5 exotic transformations | ✅ |
| Drop behavioral atlas (Tier 5) | 2K | Community resource | ✅ |
| Drop coverage optimizer | 2K | Must use random sampling | ✅ |
| Drop Rust shrinker; pure Python | 11.5K Rust | Slower shrinking (10–50×); weaker M5 guarantee | ⚠️ Partially |
| Drop HuggingFace adapter | 4K | Only spaCy in evaluation | ⚠️ Partially — single framework limits generality claim |

---

## 9. Scores

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Value** | **8/10** | Solves a real, weekly pain point for a real audience with no existing tool alternative. "10 Bugs, 10 Words" is immediately actionable. Docked 2 points for contracting market (LLM shift) and RAG as future work. |
| **Difficulty** | **7/10** | Multi-domain synthesis (causal reasoning + NLP internals + formal grammars + property-based testing) with two novel algorithmic contributions. Known techniques in new combination — difficulty is in making the composition work correctly on typed heterogeneous IRs, not in conceiving the approach. |
| **Best-Paper Potential** | **7/10** | Three-punch evaluation (real bugs + causal ablation + GPT-4 baseline) overcomes "predictable approach" objection. If N2 is proved, jumps to 8–9. Bug discovery quality is the uncontrollable variable. |
| **Feasibility** | **8/10** | ~40K LoC (24K Python + 11.5K Rust) is achievable. No grammar compiler risk. Feature-checker is scoped to exactly 15 transformations. Staged rollout contains framework-adapter risk. pip-installable artifact. |

**Composite: 7.5/10** (up from 6.25 in depth check, reflecting scope right-sizing and risk reduction)

---

## 10. Differentiation from Portfolio

This project is **distinctly different** from all existing halley-labs portfolio projects:

- **vs. ml-pipeline-selfheal:** Self-heal addresses ML *training* pipeline repair (data quality, hyperparameter tuning). We address NLP *inference* pipeline fault *localization* (identifying which stage causes behavioral inconsistencies under metamorphic transformations). Different problem, different techniques, different pipeline type.
- **vs. cross-lang-verifier:** Cross-lang verifies semantic equivalence across programming language translations. We verify behavioral consistency across NLP pipeline *stages* under linguistic transformations. Different domain entirely.
- **vs. dp-verify-repair, tensorguard, tensor-train-modelcheck:** These verify properties of individual models (differential privacy, tensor operations, model checking). We localize faults *between* models in a multi-stage pipeline — the inter-stage causal analysis is our contribution, not intra-model verification.
- **vs. algebraic-repair-calculus:** Algebraic framework for program repair. We do fault *localization* (diagnosis), not repair (treatment).
- **vs. causal-risk-bounds, causal-robustness-radii, causal-plasticity-atlas:** These use causal inference for statistical/ML analysis. We use causal *intervention* for software fault localization in NLP pipelines — borrowing the do-calculus concept but applying it to a software engineering debugging problem, not a statistical estimation problem.
- **vs. zk-nlp-scoring:** Zero-knowledge proofs for NLP scoring. Different technique, different goal.

---

## 11. Summary

The Hardened Causal Pipeline Localizer takes the best from each competing approach:

- **From Approach A:** The value framing (regulated-industry NLP teams), the "10 Bugs, 10 Words" artifact strategy, the formal M4/M5 contributions, and the Rust performance where it matters (shrinker only).
- **From Approach B:** The N1 discriminability matrix (cheap, useful diagnostic), the N4 hardness/convergence results, and the two-track strategy for N2 (attempt the crown jewel proof in parallel, with contingency).
- **From Approach C:** The Python-first architecture, corpus-based generation, staged rollout, GPT-4 baseline, feasibility-first mindset, and scope-cut discipline.

It rejects: A's full grammar compiler (tar pit), B's uncomputable C(T) and wrong-for-NLP N3(b,c), and C's spaCy-as-proxy (kills M5).

The result is a paper that knows exactly what it is: a well-scoped tool with genuine formal contributions, a devastating evaluation strategy, and a clear fallback plan.

---

*Synthesized by 5-expert team. Verified by Adversarial Skeptic and Math Depth Assessor. All scores reflect post-debate consensus.*

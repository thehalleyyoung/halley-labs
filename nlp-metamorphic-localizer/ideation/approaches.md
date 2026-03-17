# Competing Approaches: nlp-metamorphic-localizer

**Project:** "Where in the Pipeline Did It Break? Metamorphic Fault Localization for Multi-Stage NLP Systems"
**Stage:** Ideation — Approach Design
**Date:** 2026-03-08
**Method:** 5-expert team (Domain Visionary, Math Depth Assessor, Difficulty Assessor, Adversarial Skeptic, Synthesis Editor) with independent proposals, adversarial critique, and synthesis.

---

## Approach A: "The Pipeline MRI" — Maximum Value Orientation

*Proposed by: Domain Visionary*

### One-Sentence Summary

A grammar-compiled metamorphic testing engine that, given a multi-stage NLP pipeline exhibiting a behavioral inconsistency, automatically identifies the *specific stage* that introduced or amplified the fault and hands the engineer a minimal proof sentence of ≤10 words.

### Extreme Value Delivered

**Primary persona:** The NLP reliability engineer at a regulated-industry company (healthcare entity extraction, legal clause classification, financial compliance NER). This engineer maintains a spaCy or HuggingFace pipeline with 3–7 stages running against SLAs. When a customer reports a misclassification, their debugging workflow is: insert `print()` between every stage, hand-craft 20 variants, stare at intermediate outputs — 4–8 hours per incident. According to John Snow Labs' 2023 survey, 60% of NLP teams spend more time debugging pipeline interactions than training models.

**Three concrete pain scenarios:**

1. *Model upgrade regression:* Team upgrades `en_core_web_sm` to `en_core_web_trf`. 23 entity tests fail. Which of 4 simultaneously-changed stages regressed? Expected diagnosis: 1–3 engineer-days. With our engine: 75-minute automated run, structured report with 17 regressions localized to specific stages and minimal proof sentences.

2. *Compliance audit (FDA/HIPAA):* Healthcare pipeline extracts adverse drug events. Auditor asks "how do you know passive voice is handled correctly?" Today: "we tested 50 manual examples." With engine: "14,000 grammar-valid metamorphic variants, 7 inconsistencies localized to specific stages, each with a minimal proof sentence."

3. *Cascading fault mystery:* Financial NER misses "Goldman Sachs" in passive constructions. NER alone handles passives correctly — the bug only manifests in the full pipeline. Parser misanalyzes the passive auxiliary, corrupting the dependency tree, and NER trusts dependency features. No existing tool can distinguish "NER is wrong" from "parser is wrong and NER is a victim." Our causal-differential localizer replaces the parser's output with the correct parse and observes NER recover — proving the parser is root cause.

**Quantified value:** $3K–$18K per release cycle in saved engineering time at $150/hour loaded cost, 5–15 incidents per cycle.

### Genuine Difficulty

**Subproblem 1: Causal localization across typed intermediate representations.** Classical SBFL operates on binary statement coverage. NLP stages produce typed, structured, high-dimensional IRs (token sequences, POS tag sequences, dependency trees, entity spans). Per-stage distance functions must be type-specific and calibrated. The interventional step — replacing one stage's output with the original execution's output — requires partial re-execution from intermediate checkpoints with copy-on-write snapshots.

**Subproblem 2: Grammar-aware shrinking preserving grammaticality AND transformation applicability.** String-level delta debugging produces "The was by Kim report written." Tree-level delta debugging (TreeReduce) works for CFGs but English requires unification-based feature checking. The shrinker must maintain (a) unification constraints for grammaticality, (b) syntactic preconditions for transformation applicability, and (c) the metamorphic violation. Novel three-way constraint optimization.

**Subproblem 3: Dual-mode grammar compilation.** Generation requires Boltzmann-weighted forward sampling. Shrinking requires minimal-derivation backward search. Both from a single probabilistic unification grammar with DAG unification, morphological FSTs. Scoped to ~200 productions for 15 transformations (not a full English grammar).

**Subproblem 4: Rust/Python bridge.** Grammar ops need Rust speed (millions of derivations/sec). NLP libs are Python. PyO3 bridge ~3K LoC with zero-copy IR transfer.

### New Math (Load-Bearing Only)

**M4: Causal-Differential Fault Localization (NEW, diamond).** Per-stage differentials Δₖ(x, τ) = d_k(sₖ(prefix_k(x)), sₖ(prefix_k(τ(x)))). Localize to k* = argmax_k [Δₖ − E[Δₖ | τ is meaning-preserving]]. Causal refinement via intervention distinguishes introduction from amplification. Complexity O(N·n·C_pipeline).

**M5: Grammar-Constrained Shrinking Convergence (NEW).** GCHDD terminates in O(|T|²·|R|) grammar-validity checks. 1-minimal counterexamples. Extends delta debugging to parse trees with unification constraints.

**M3: MR Composition (Formal Spec).** Statistical composition with Clopper-Pearson bounds. Reduces test count from O(|T|²) to O(|T|·log|R|).

**M7: BFI (Formal Spec).** Ratio metric for per-stage amplification. Not claimed as novel math.

### Best-Paper Argument

Centerpiece: the **"10 Bugs, 10 Words" table** — 10 real bugs in spaCy/HuggingFace/Stanza, each localized to a specific stage, each with a minimal proof sentence ≤10 words. Supported by the causal ablation showing vanilla SBFL <65% on cascading faults vs. ≥85% with causal refinement, and the GPT-4-as-debugger baseline. Target: ISSTA/ASE.

### Hardest Challenge

Grammar compiler scope creep. Mitigation: ruthless scoping to only constructions needed by 15 transformations. Go/no-go at ~53K LoC minimum viable system.

### Scores

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Value | 8/10 | Solves weekly pain for real audience; no existing alternative. Market narrowing (LLM shift) docks 2 pts. |
| Difficulty | 7/10 | Multi-domain synthesis; known techniques in new combination. Grammar engineering is hard but scoped. |
| Potential | 7/10 | Strong-accept at ISSTA if evaluation delivers. Best-paper depends on bug discovery quality. |
| Feasibility | 7/10 | Go/no-go at 53K LoC is achievable. Grammar compiler is primary risk. |

---

## Approach B: "Information-Theoretic Causal Localizer" — Maximum Math Depth

*Proposed by: Math Depth Assessor*

### One-Sentence Summary

A metamorphic fault localization engine whose test-generation, localization, and shrinking are each derived from provably optimal information-theoretic foundations — with tight sample-complexity bounds governed by a computable localization capacity of the transformation algebra.

### Extreme Value Delivered

Same primary persona as Approach A. Additional value: (1) a *provable sample-complexity guarantee* — before running tests, the engine reports how many are necessary and sufficient with confidence 1−δ; (2) a *causal identifiability certificate* — formal conditions for when introduction-vs-amplification is distinguishable without interventional replay; (3) *provably minimal counterexamples* with hardness results showing 1-minimality is the best polynomial-time guarantee.

"Provable" matters because multi-stage pipelines have a unique pathology: fault signals attenuate and distort through stages. Without information-theoretic foundations, you cannot know whether your test suite is *capable* of localizing the fault at all.

### Genuine Difficulty

**Hard subproblem 1: Adaptive test selection under grammar constraints.** Bayesian sequential testing: select transformation maximizing conditional mutual information with fault location. Balance exploration/exploitation within CPU budget. Bayesian posterior updates with sub-second response times.

**Hard subproblem 2: Interventional replay with semantic correctness.** Counterfactual pipeline executions. Type mismatches when replacing one stage's output. Adapters for spaCy, Stanza, HuggingFace without modifying frameworks.

**Hard subproblem 3: Grammar-aware shrinking with provable termination.** Navigate parse-tree reductions where most produce ungrammatical inputs. Convergence proof requires showing bounded recursion ensures polynomial-size search.

### New Math (5 Deep Results)

**N1: Stage Discriminability Matrix and Separation Theorem.** M ∈ ℝⁿˣᵐ where M_{k,j} = E_x[Δₖ(x, τⱼ)]. rank(M)=n iff T can localize to unique stage. If rank(M) < n, identifies indistinguishable stages. Diagnostic completeness check before running tests.

**N2: Information-Theoretic Localization Bounds (⭐ CROWN JEWEL).** Lower bound: m ≥ (ln(n−1) + ln(1/δ))/C(T). Upper bound: m ≤ (2 ln n + ln(1/δ))/C(T) + O(n) via ADAPTIVE-LOCATE. Non-adaptive penalty: n-fold increase. C(T) = max_w min_{i≠j} Σⱼ wⱼ·D(τⱼ; hᵢ, hⱼ). Proof via extension of Fano's inequality to sequential testing with structured hypotheses and correlated observations from composed pipeline stages.

**N3: Causal Identifiability.** SCM model: Sₖ = fₖ(Sₖ₋₁) + εₖ. Direct Causal Effect DCE_k(τ) = E[Δₖ | do(Sₖ₋₁^τ := Sₖ₋₁)]. Observationally identifiable when fₖ is locally linear. Cheap-first strategy: try observational identification, fall back to intervention.

**N4: Grammar-Constrained Minimization Hardness.** NP-hardness of global minimization. 1-minimality in O(|T_x|·log|T_x|·|R|) — improved from O(|T|²·|R|) via binary search. Expected shrinking ratio E[|x'|] ≤ |x|/b + O(α·log|x|).

**N5: Submodularity of Localization Information.** F(S) = I(H; {Δ(xᵢ, τᵢ)}ᵢ∈S) is monotone submodular. Greedy achieves (1−1/e) approximation. Foundation for batch/CI mode.

### Best-Paper Argument

"Information-Theoretic Limits of Pipeline Fault Localization" — the first information-theoretic foundation for metamorphic fault localization. The tight lower+upper bound is the kind of result that wins best-paper awards. The transformation algebra has a measurable *capacity* for fault localization, and optimal algorithms achieve this capacity. Generalizes beyond NLP to any pipeline architecture.

### Hardest Challenge

Estimating discriminability distributions P_k(Δ|τ) from finite calibration data. KL estimation in high-dimensional spaces. Mitigation: kernel density estimation with cross-validation, robust ADAPTIVE-LOCATE with lower confidence bounds, 100–200 calibration tests as exploration phase.

### Scores

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Value | 7/10 | Provable guarantees genuinely new for NLP testing; market narrowing docks. |
| Difficulty | 9/10 | N2 is research-level information theory. N3 combines SCM with NLP semantics. |
| Potential | 8/10 | Information-theoretic foundation is broadly interesting. Tight bounds win best papers. Risk: practical impact questioned. |
| Feasibility | 6/10 | N2 proof is 40% failure risk. Calibration data requirements may be infeasible for transformers. |

---

## Approach C: "The Pragmatic Maximizer" — Maximum Feasibility/Difficulty Balance

*Proposed by: Difficulty Assessor*

### One-Sentence Summary

Ship a Python-first causal fault localizer for multi-stage NLP pipelines that finds real bugs, pinpoints the guilty stage via interventional analysis, and shrinks each proof to under 10 words — killing the grammar compiler risk by building on existing parsers, and making the evaluation so empirically devastating that the "predictable approach" objection evaporates.

### Extreme Value Delivered

Same regulated-industry persona. Same "thermometer vs MRI" value shift. Key pragmatic insight: the grammar compiler delivers only ~20% of marginal value but consumes ~40% of risk budget. Kill it. Redirect all saved effort into evaluation quality.

The GPT-4-as-debugger baseline reframes the paper as "tool vs LLM" — timely, not archaic. If the tool beats GPT-4 by 20+ points on cascading faults, that's the killer comparison.

### Genuine Difficulty

**Core difficulty is causal, not generative.** The depth check found 80% of value from 20% of system — specifically M4 (localization) and M5 (shrinking). Generation is solved by existing parsers + corpora.

**Causal-differential localization with interventional replay across typed heterogeneous IRs** is the hard subproblem. Token-to-tree alignment when replacing stage outputs is where months go. Mitigation: lemma-level alignment (transformation-invariant for 11/15 transformations), explicit alignment maps for the other 4, and fallback to statistical localization for unaligned cases.

**Grammar-aware shrinking without a grammar compiler.** Use spaCy parser as grammaticality proxy (~2ms/sentence). Linguistically imprecise but practically fast. Set 60-second timeout per counterexample.

### New Math

**Must prove:** M4 (causal-differential localization) and M5 (grammar-constrained shrinking convergence — though with weaker validity oracle).

**Must implement:** Task-specific IR distances, interventional replay, 15 tree transductions, Ochiai-adapted SBFL.

**Can cite:** SBFL foundations, delta debugging, TreeReduce, covering arrays, Clopper-Pearson.

### Best-Paper Argument

Win by making evaluation undeniable, not by theorem count. Three-punch combination: (1) "10 Bugs, 10 Words" table, (2) ablation showing vanilla SBFL <65% on cascading faults, (3) GPT-4 loses to the tool on cascading faults. Killer demo: engineer types one command, gets localized faults with minimal proofs in 30 minutes.

### Risk Analysis

| Risk | Prob | Impact | Mitigation |
|------|------|--------|------------|
| Real bug yield <5 | 30% | High | Pre-screen; expand to older model versions; fallback to Tier 2+3 |
| Interventional replay breaks on >20% of cases | 25% | Medium | Restrict causal refinement to 11 aligned transformations; fallback to Ochiai for others |
| Shrinking too slow (>60s/counterexample) | 20% | Medium | spaCy as fast proxy; cache results; timeout and report best-found |

**Minimum viable paper:** ~18K LoC, M4 only, Tier 2 evaluation on 50 injected faults. Still publishable as tool contribution.

**Ship-it version:** ~25-30K LoC pure Python. M4 + M5 + bug table + GPT-4 comparison.

### Scores

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Value | 7/10 | Category change in debugging for real (narrowing) audience. |
| Difficulty | 7/10 | Multi-domain synthesis; known methods in new combination; correct implementation is where difficulty lives. |
| Potential | 7/10 | Evaluation-driven paper; GPT-4 baseline is timely. Uncertain because bug yield not guaranteed. |
| Feasibility | 8/10 | 25-30K LoC Python achievable in 3-4 months. No grammar compiler risk. |

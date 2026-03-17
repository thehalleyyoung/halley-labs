# Approach C: The Pragmatic Maximizer

## 1. Approach Name and One-Sentence Summary

**"Causal Pipeline Debugger with Grammar-Aware Minimization"**

Ship a Python-first causal fault localizer for multi-stage NLP pipelines that finds real bugs, pinpoints the guilty stage via interventional analysis, and shrinks each proof to under 10 words — killing the grammar compiler risk by building on existing parsers instead of writing one from scratch, and making the evaluation so empirically devastating that the "predictable approach" objection evaporates.

---

## 2. Extreme Value Delivered and Who Desperately Needs It

**The user who loses sleep over this.** An NLP engineer at a healthcare company runs a spaCy pipeline (tokenizer → POS tagger → dependency parser → NER → relation extractor) that processes clinical notes. A customer reports that "The patient was prescribed Lisinopril by Dr. Chen" correctly identifies both entities, but "Lisinopril was prescribed to the patient by Dr. Chen" drops "Dr. Chen" as a PERSON entity. The engineer's debugging process: insert print statements between every stage, manually compare intermediate representations for dozens of examples, guess which stage is responsible, patch, pray. This takes days. With our tool, they run one command and get: "Fault localized to dependency parser (stage 3): passive dative constructions cause PP misattachment, cascading to NER. Minimal proof: 'Kim was given the book by Lee.' — 8 words." They add that sentence as a regression test and move on. The tool turns a multi-day forensic investigation into a 75-minute automated analysis.

**Why this is desperate, not nice-to-have.** In regulated domains (healthcare NLP, legal entity extraction, financial compliance), pipeline failures aren't just bugs — they're audit findings. The FDA's guidance on AI/ML-based Software as a Medical Device requires documented testing evidence. "We ran 50 manual examples" doesn't pass audit. "We ran 14,000 grammar-valid metamorphic variants, localized 7 actionable inconsistencies to specific pipeline stages, each with a minimal proof sentence, and added them as automated regression tests" does. John Snow Labs' 2023 survey found 60% of NLP teams spend more time debugging pipeline interactions than training models. Even as the industry shifts toward LLMs, the ~15-25% of production NLP running classical multi-stage pipelines is concentrated exactly where the stakes are highest — and where pipeline interpretability is a regulatory requirement, not a preference.

**The behavioral atlas as community infrastructure.** Beyond individual debugging, the engine produces a public database mapping (transformation × pipeline × NLP task) triples to behavioral outcomes. This atlas enables cross-system comparisons ("transformer-based NER is 3.2× more fragile to passivization than rule-based NER"), reveals systematic blind spots shared across frameworks, and provides a citable scientific artifact. For the NLP robustness community, this is the equivalent of a shared benchmark — except it characterizes failure modes, not just accuracy numbers.

---

## 3. Why This Is Genuinely Difficult as a Software Artifact

**The core difficulty is causal, not generative.** The depth check correctly identifies that 80% of the value comes from 20% of the system — specifically, the causal-differential fault localizer (M4) and the grammar-aware shrinker (M5). The genuinely hard problem is not "generate test inputs" (existing tools do this) or "detect failures" (trivial). It is: given that a 5-stage pipeline produces a wrong answer on a transformed input, determine whether Stage 3 *introduced* the fault or merely *amplified* a fault that Stage 2 already injected. This requires interventional causal reasoning — you must surgically replace Stage 3's output with the original execution's output and observe whether the downstream violation disappears. This is conceptually clean but implementation-hard: intermediate representations are typed and heterogeneous (token lists, POS tag sequences, dependency trees, entity spans), distance metrics must be task-specific, and the intervention must preserve type compatibility across stage boundaries. No existing tool — not CheckList, not TextFlint, not any SBFL implementation — solves this because classical SBFL operates on statement coverage, not on typed intermediate outputs between pipeline stages. Making this work correctly for three real pipeline frameworks (spaCy, Stanza, HuggingFace) where each framework has different IR formats, different stage boundaries, and different API conventions is where the months go.

**The grammar-aware shrinker is a constrained optimization problem with no off-the-shelf solution.** Delta debugging on strings produces "T patie prescr" — ungrammatical garbage that developers dismiss. Delta debugging on parse trees (TreeReduce) works for programming languages where syntax is unambiguous and fully specified by a CFG. Natural language parse trees have feature structures (agreement, case, subcategorization), ambiguity, and the constraint that the shrunk sentence must still be a valid input to the *same transformation* that exposed the bug. If passivization exposed the bug, the shrunk sentence must still be passivizable. This is a novel constraint optimization problem (M5) that extends TreeReduce to a harder domain. The convergence bound O(|T|² · |R|) must be proved, not just asserted.

**What can be simplified without losing the core contribution.** The grammar compiler is the single biggest risk in the original proposal — a probabilistic unification grammar for English is a multi-year research project (LKB, XTAG, ERG took decades). The pragmatic insight: we do not need to *generate* from a grammar to demonstrate causal localization. We need grammatically-controlled test inputs and grammar-aware shrinking. For generation, we use existing NLP parsers (spaCy's parser) to parse seed sentences from curated corpora, then apply our transformations to the parse trees. For shrinking, we build a lightweight grammaticality checker — not a full grammar — that validates subtree deletions against agreement and subcategorization constraints. This eliminates ~12K LoC of grammar compiler, removes the multi-year risk, and preserves 100% of the M4 and M5 contributions. The generation quality trades off some linguistic control for massive feasibility gains. We recover control through careful corpus curation (Penn Treebank sentences covering the syntactic constructions needed by all 15 transformations) and transformation-specific precondition checks.

---

## 4. New Math Required

### Must Prove (load-bearing theorems that reviewers will scrutinize)

**M4: Causal-Differential Fault Localization.** Given pipeline P = s₁ → ... → sₙ and a metamorphic violation, define per-stage differentials Δₖ(x, τ) = dist(sₖ(prefixₖ(x)), sₖ(prefixₖ(τ(x)))). The localizer computes k* = argmax_k [Δₖ − E[Δₖ | τ is meaning-preserving]]. The causal refinement theorem must state: if replacing stage k's output with the original execution's output eliminates the downstream violation, then k is a *fault-introducing* stage; if the violation persists but is attenuated, k is a *fault-amplifying* stage. The formal contribution is the distinction between introduction and amplification via interventional analysis — a synthesis of SBFL with causal reasoning that is novel in the NLP pipeline setting. Complexity: O(N · n · C_pipeline), linear in test count N and pipeline depth n. Must prove that the interventional analysis correctly identifies fault-introducing stages under the assumption that the transformation is meaning-preserving for the tested task dimension, and must characterize failure modes when this assumption is violated.

**M5: Grammar-Constrained Shrinking Convergence.** GCHDD (Grammar-Constrained Hierarchical Delta Debugging) must be proved to terminate in O(|T|² · |R|) grammar-validity checks, where |T| is parse tree size and |R| is the grammar rule count used for validity checking. Each validity check costs O(|F|) for feature unification. Must prove 1-minimality: no single subtree replacement further reduces the input while preserving grammaticality, transformation applicability, and the metamorphic violation. This extends Zeller's delta debugging convergence proof to the harder constraint domain of linguistically-annotated parse trees.

### Must Implement (algorithms that are the engineering contribution)

- **Task-specific IR distance functions:** Tree edit distance for dependency parses, span-level exact match for NER, label + confidence threshold for sentiment. These are well-understood metrics applied to pipeline-internal representations — no new math, but correct implementation across three frameworks is non-trivial.
- **Interventional pipeline replay:** The mechanism that replaces stage k's output with the original execution's output and replays downstream stages. Must handle type mismatches (e.g., replacing a POS tag sequence of different length due to tokenization differences).
- **15 syntactic transformations as tree transductions:** Each transformation (passivization, clefting, topicalization, etc.) operates on dependency/constituency trees with precondition checking and postcondition validation. Engineering-hard, not math-hard.
- **Ochiai-based spectrum fault localization adapted to pipeline stages:** Standard SBFL (Ochiai coefficient) with the novel twist that "coverage" is redefined as "which pipeline stages process which linguistic features" rather than "which statements execute."

### Can Cite (established results we use directly)

- **SBFL foundations** (Ochiai, DStar, Barinel coefficients) — Jones & Harrold, 2005; Wong et al., 2016.
- **Delta debugging** — Zeller & Hildebrandt, 2002.
- **TreeReduce** — Herfert et al., 2017.
- **Covering arrays for product-space coverage** — Colbourn, 2004.
- **Clopper-Pearson exact binomial confidence intervals** for the statistical composition bound (M3, reframed as a formal specification, not a theorem).
- **Metamorphic testing foundations** — Chen et al., 2018; Segura et al., 2016.

---

## 5. Best-Paper Argument

**This approach wins best paper by making the evaluation undeniable.** The "predictable approach" objection — "SBFL + causal intervention on NLP pipelines, anyone would think of that" — is neutralized not by adding mathematical novelty but by demonstrating that (a) the obvious approach *fails* and (b) the refinement that makes it work produces results no one else has. Specifically: the paper opens with a table of 10+ real bugs in shipped NLP systems (spaCy, Stanza, HuggingFace), each localized to a specific pipeline stage, each with a minimal counterexample of ≤10 words. This table is the reviewer magnet — compact, verifiable, immediately actionable. Then the ablation study shows that vanilla SBFL without causal refinement achieves <65% localization accuracy on cascading faults (where the guilty stage is not the last stage), while our causal refinement achieves ≥85%. The GPT-4-as-debugger baseline — where GPT-4 is given the same pipeline topology, transformation, and end-to-end failure and asked to identify the faulty stage — provides the "strongest informal alternative" comparison. GPT-4 cannot perform interventional analysis; it guesses based on surface patterns. If our tool beats GPT-4 by 20+ percentage points on cascading faults, that is the "killer comparison" that makes reviewers say "this tool does something an LLM cannot."

**The killer demo is a 30-second GIF.** Engineer types `nlp-localizer run --pipeline spacy --suite all --budget 30m`. After 30 minutes, the tool outputs a ranked list of localized faults with minimal proofs. The engineer clicks one: "Stage: dependency parser. Transformation: passivization. Proof: 'The book was given to Kim by Lee.' (9 words). Cascade: parser misattaches 'by'-PP → NER drops 'Lee' as PERSON." The engineer copies the proof sentence into a pytest regression test. Total debugging time: 30 minutes of CPU, 2 minutes of human attention. This is the demo that makes practitioners want to install it and makes reviewers want to cite it. Combined with the behavioral atlas — a public, queryable database of pipeline fragility patterns — the contribution is both a tool and a scientific resource.

---

## 6. Hardest Technical Challenge and Mitigation

**The hardest challenge is making interventional pipeline replay work correctly across three different NLP frameworks with heterogeneous intermediate representations.** When you replace the dependency parser's output for a passivized sentence with the original sentence's parser output, the downstream NER module receives a dependency tree that was computed for a *different* token sequence (passivization changes word order and adds auxiliaries). The token-to-tree alignment problem — mapping tree nodes from the original execution to corresponding nodes in the transformed execution — is where the majority of debugging time will go. Mitigation: (1) define alignment at the *lemma* level rather than the *token* level, since lemmas are transformation-invariant for most of our 15 transformations; (2) for transformations that change lemma inventory (synonym substitution), maintain an explicit transformation-specific alignment map; (3) build the interventional replay first for spaCy only (where all IR formats are well-documented Python objects), validate it thoroughly with the 50-fault injection suite, then port to Stanza and HuggingFace. This staged rollout contains the alignment complexity to one framework at a time and ensures the core algorithm is correct before scaling.

---

## 7. Risk Analysis

### Top 3 Risks

| # | Risk | Probability | Impact | Mitigation |
|---|------|-------------|--------|------------|
| 1 | **Real bug yield is too low** — shipped NLP pipelines are more robust than expected, yielding <5 actionable bugs in Tier 1 | 30% | High — the "10 bugs, 10 words" table is the paper's centerpiece | (a) Pre-screen: run quick passivization + NER checks on spaCy/Stanza before committing to full evaluation; if <3 bugs found in 2 days, expand to more exotic transformations (clefting, there-insertion, embedding depth). (b) Fallback: paper pivots to Tier 2 (localization accuracy on 50 injected faults) + Tier 3 (shrinking quality) + GPT-4 baseline — independently publishable as a tool contribution even with low real-bug yield. (c) Expand pipeline targets to include older model versions (spaCy 2.x, Stanza 1.2) which are known to have more edge-case failures. |
| 2 | **Interventional replay breaks on heterogeneous IRs** — token-to-tree alignment failures produce nonsensical interventions for >20% of test cases | 25% | Medium — degrades localization accuracy, doesn't kill the paper | (a) Restrict interventional analysis to "aligned" transformations where token sequences have a clear correspondence (passivization, topicalization, clefting — 9 of 15 transformations preserve token-lemma alignment). (b) For unaligned transformations (synonym substitution, embedding depth change), fall back to statistical localization (Ochiai only, no causal refinement) and report this honestly as a limitation. (c) The ablation study still shows that causal refinement helps on the 9 aligned transformations, which is sufficient for the contribution claim. |
| 3 | **Grammar-aware shrinking is too slow** — validity checking per subtree deletion takes >5s, making shrinking impractical (>10 min per counterexample) | 20% | Medium — degrades user experience, doesn't invalidate M4 | (a) Use spaCy's parser as a fast grammaticality proxy (~2ms/sentence) instead of full unification-based validity checking. This is linguistically imprecise but practically fast. (b) Cache parse results for partially-shrunk sentences. (c) Set a per-counterexample timeout (60s) and report the best shrunk sentence found within budget — practical 1-minimality, not theoretical 1-minimality. (d) If shrinking is still too slow, report shrink ratios for the subset that completes within budget; the localizer (M4) is the primary contribution, not the shrinker (M5). |

### Minimum Viable Paper (if things go wrong)

If all three risks materialize simultaneously (low bug yield + alignment failures + slow shrinking), the minimum viable paper is:

**"Causal Fault Localization for Multi-Stage NLP Pipelines: An Interventional Approach"** — focused exclusively on M4.

- Contribution: the causal-differential localization algorithm with the introduction-vs-amplification distinction.
- Evaluation: Tier 2 only (50 fault-injected pipeline configurations, top-1 accuracy ≥85%, comparison against 4 SBFL baselines + GPT-4-as-debugger).
- No real bug table, no shrinking, no atlas.
- Still publishable at ISSTA/ASE as a solid tool paper because the causal refinement over vanilla SBFL on NLP pipelines is a genuine contribution that no one has demonstrated.
- Estimated LoC: ~18K Python (localizer + spaCy adapter + evaluation harness).

This is the "iron core" that survives any scope cut.

### Scope Cuts That Preserve the Core Contribution

| Cut | LoC Saved | What's Lost | Core Contribution Preserved? |
|-----|-----------|-------------|------------------------------|
| Drop grammar compiler entirely; use parsed corpus + transformations | ~12K | Controlled generation; must curate corpus instead | ✅ Yes — localization and shrinking are unaffected |
| Drop Stanza adapter (keep spaCy + HuggingFace only) | ~4K | One pipeline framework in evaluation | ✅ Yes — two frameworks still demonstrate generality |
| Reduce from 15 to 8 transformations | ~4K | Reduced transformation coverage | ✅ Yes — 8 transformations covering voice, clause structure, lexical, morphological axes is sufficient |
| Drop behavioral atlas (Tier 5) | ~3K | Community resource; cross-system findings | ✅ Yes — the atlas is a bonus, not the contribution |
| Drop coverage analyzer + test scheduler | ~6K | Optimal test prioritization; must use uniform random | ✅ Yes — coverage optimization (M3, M6) is not the core contribution |
| Drop Rust entirely; pure Python | ~15K bridge code | Performance; shrinking may be 10-50× slower | ⚠️ Partially — shrinking time may exceed budget, but localizer works fine in Python |

**Recommended scope for the "ship it" version:** Pure Python, no grammar compiler, spaCy + HuggingFace (drop Stanza), 15 transformations, no coverage optimizer, no behavioral atlas. **~25-30K LoC.** This delivers M4 + M5 + the bug table + the GPT-4 comparison — everything needed for a strong ISSTA/ASE submission. The Rust performance layer and behavioral atlas are post-acceptance extensions.

---

## 8. Scores

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Value** | 7/10 | Genuine "category change" in NLP pipeline debugging for a real (if narrowing) audience. The regulated-industry angle is strong. Docked from higher because the addressable market is contracting and RAG extension remains future work. |
| **Difficulty** | 7/10 | Multi-domain expertise (causal reasoning + NLP internals + property-based testing + linguistic transformations) with genuinely novel algorithmic synthesis in M4 and M5. Not frontier-pushing — applies known methods to a new domain — but the *correct implementation* across three heterogeneous frameworks is where the difficulty lives. Killing the grammar compiler risk trades some difficulty for feasibility without losing the hard core. |
| **Best-Paper Potential** | 7/10 | The "10 bugs, 10 words" table + causal ablation + GPT-4-as-debugger baseline is a three-punch combination that overcomes the "predictable approach" objection through sheer empirical force. The key upgrade from the depth check's 5/10: (a) the GPT-4 baseline reframes the paper as "tool vs. LLM" which is timely, not archaic; (b) dropping the grammar compiler lets us spend those months perfecting the evaluation; (c) the interventional ablation showing vanilla SBFL fails on cascading faults is the technical surprise that earns best-paper consideration. Still uncertain because the real-bug yield cannot be guaranteed pre-evaluation. |
| **Feasibility** | 8/10 | The "ship it" version is ~25-30K LoC of Python — achievable by a strong engineer in 3-4 months. No grammar compiler risk. No Rust bridge complexity. spaCy and HuggingFace are well-documented with stable Python APIs. The main feasibility risk is interventional replay alignment, which is contained by the staged rollout strategy. The pure-Python approach means every component can be debugged with standard tools (pdb, pytest). |

**Composite: 7.25/10** (unweighted average, up from 6.25 in the depth check)

**Key tradeoffs vs. the original proposal:**
- Grammar compiler risk: **eliminated** (biggest single improvement)
- LoC: ~107K → ~25-30K for the ship-it version (3-4× reduction)
- Linguistic control of generation: **reduced** (corpus-based instead of grammar-compiled)
- Formal elegance: **reduced** (no dual automaton compilation, no weighted RTNs)
- Evaluation investment: **increased** (months saved on grammar compiler redirected to evaluation quality)
- M4 contribution: **preserved entirely**
- M5 contribution: **preserved with practical concessions** (spaCy parser as validity proxy)
- Best-paper potential: **increased** (evaluation quality > mathematical elegance for SE venues)

# Depth Check: SoniType — Perceptual Sonification Compiler

**Slug**: `perceptual-sonification-compiler`
**Date**: 2026-03-08
**Verdict**: CONDITIONAL CONTINUE
**Composite Score**: V4 / D6 / BP5 / L7.5 = 22.5/40

---

## Evaluation Panel

- **Independent Auditor** (V4/D6/BP5/L8): The intellectual contribution is genuine and the architecture is sound, but the value proposition is narrow (tiny user base), the LoC is inflated by ~30%, and best-paper aspirations at top venues are unrealistic without tighter PL focus.
- **Fail-Fast Skeptic** (V3/D5/BP3.5/L5.5): Five serious wounds—tiny market, inflated theorems, shallow type system risk, evaluation circularity, and no home venue—collectively undermine the proposal, though no single flaw is fatal; scope reduction to ~60K and venue identification are mandatory.
- **Scavenging Synthesizer** (V6/D7/BP6/L8): The perceptual type system is the crown jewel and should lead the paper at OOPSLA; with safety-critical alarm reframing, scope reduction to ~95K, and adversarial bug-finding evaluation, all pillars can reach 7+.

---

## Pillar 1: EXTREME AND OBVIOUS VALUE — 4/10

**The user base is extremely small.** The ICAD conference—the primary venue for sonification research—draws 100–200 attendees per year and publishes 40–60 papers. The entire global sonification research community is ~200–500 active researchers. The landscape survey confirms this implicitly: every significant sonification tool in existence can be enumerated in a single document. The intersection of people who (a) do data sonification, (b) need automated psychoacoustic optimization, and (c) would adopt a Rust-based compiler toolchain is in the low dozens worldwide.

**The EU Accessibility Act argument is overweighted.** The EU Accessibility Act (Directive 2019/882, enforcement from June 2025) and WCAG 2.2 do require non-visual alternatives for digital content. However, the Act's practical compliance is overwhelmingly served by screen readers, ARIA markup, alt-text, haptic feedback, and structured data formats—not by psychoacoustically-optimized sonification compilers. No compliance officer has ever demanded a type-checked sonification pipeline. The proposal presents the Act to borrow urgency that belongs to accessibility broadly, not to SoniType specifically. The panel consensus: this is "strategically misleading"—not fabricated, but presented with disproportionate emphasis. The claim that "sonification remains niche precisely because the design cost is high" reverses causation; sonification remains niche primarily because audio is a poor channel for dense quantitative data compared to other accessible modalities.

**The strongest value argument is safety-critical alarm design.** Medical alarm fatigue is documented as a top health technology hazard by the ECRI Institute. A typical ICU has 30+ alarm types, and discriminability between life-threatening and routine alarms is a safety-critical need. SoniType's formal discriminability guarantees (JND verification, masking checking, segregation proofs) directly address this gap. IEC 60601-1-8 says "alarms should be distinguishable" but provides no computational tool to verify discriminability. However, this argument is discounted because: (a) the psychoacoustic models have not been validated for alarm-design conditions, (b) no hospital will adopt a research compiler without FDA clearance and human validation, and (c) the project explicitly excludes human studies. The "design-stage pre-screening" framing (reduce the candidate space before human testing) has merit but is speculative—no alarm designer has requested this tool.

**LLM competition is unaddressed.** In 2026, a user can say "generate a SuperCollider script mapping column A to pitch and column B to loudness" and get working sonification code in 30 seconds. The result won't be psychoacoustically optimized, but it works. The formal compiler must demonstrate overwhelming superiority over "LLM + prompt engineering" to justify its complexity. The proposal never addresses this competition.

**The "weeks to minutes" claim is unsubstantiated.** No evidence is provided for the "weeks" baseline. In practice, a researcher using SonifyR or TwoTone produces serviceable sonification in hours, not weeks.

**Score rationale**: Value is REAL but NARROW. The lossy-coding insight is genuinely novel. The formalization contribution has intellectual value. But "desperate need" for a "large audience" is absent. The alarm vertical adds something but is unvalidated and unreachable within the project's own constraints. **Score < 7 → amendments required.**

---

## Pillar 2: GENUINE DIFFICULTY AS A SOFTWARE ARTIFACT — 6/10

### LoC Inflation Analysis

The 142K LoC headline is not credible. Panel audit:

| Subsystem | Claimed | Realistic | Assessment |
|-----------|---------|-----------|------------|
| DSL parser, AST, type-checker | 12,000 | 12,000 | **Reasonable.** Custom parser + constraint-propagation type-checker is non-trivial. |
| Psychoacoustic cost models | 8,000 | 8,000 | **Reasonable.** Bark-scale masking, JND models, Bregman predicates require careful implementation. |
| Info-theoretic optimizer | 10,000 | 10,000 | **Reasonable.** Branch-and-bound, MC MI estimation, Pareto front computation. |
| IR and optimization passes | 8,000 | 5,000–6,000 | **Slightly inflated.** Audio graph IR is simpler than general-purpose compiler IR. |
| Code generator + WCET | 7,000 | 4,000–5,000 | **Inflated.** Self-admitted "conservative safety margins, not formal WCET analysis." With 50–100× headroom, this is back-of-envelope arithmetic. |
| Real-time audio renderer | 10,000 | 10,000 | **Reasonable but not novel.** Lock-free audio graph execution is well-solved. Libraries like JUCE, cpal handle this. |
| Standard library | 28,000 | 10,000–12,000 | **PADDING.** Pitch scales are ~100 lines each. FM/additive oscillators are a few hundred lines. Even a generous standard library is ~10–12K. The 28K is aspirational scope inflation. |
| Test infrastructure | 22,000 | 15,000 | **Inflated but defensible.** Property-based tests generate verbose harnesses. |
| Evaluation framework | 22,000 | 22,000 | **Reasonable.** d' computation, MI estimation, cross-model eval, baseline comparisons. |
| CLI, error reporting, tooling | 15,000 | 8,000–10,000 | **Inflated.** Preview server and debug visualization are Phase 2 features. |
| **TOTAL** | **142,000** | **~95,000–105,000** | **~30% inflation. Novel core: ~45–50K LoC.** |

### Hard Subproblem Audit

1. **Perceptual type inference with non-local constraints** — **GENUINELY HARD.** Adding a stream can invalidate previously-satisfied constraints for other stream pairs (new spectral energy shifts masking thresholds for existing pairs). This is more complex than standard type inference and differentiates the system from simple constraint checking. The non-local interaction property requires domain-specific constraint propagation. This is the hardest subproblem and the core PL contribution.

2. **Information-theoretic optimizer with submodular decomposition** — **INTERESTING BUT ASSUMPTION-DEPENDENT.** The decomposition into 24 independent Bark-band subproblems with (1-1/e)-approximation via greedy submodular maximization under matroid constraints is the genuinely novel optimization result. However, the decomposability assumption may fail when: (a) broadband stimuli span multiple bands, (b) Schroeder spreading functions model inter-band masking energy, (c) harmonic series span 5–15 bands. Whether cross-band interactions are truly negligible is an open empirical question.

3. **Formalizing Bregman as decidable predicates** — **NOVEL BUT OVERSTATED.** The proposal reduces Bregman's qualitative principles to Boolean threshold comparisons. This is a useful engineering simplification, not a formalization in the strong sense. Real auditory scene analysis is probabilistic, context-dependent, and stimulus-history-dependent. The O(k²) decidability result (Theorem 2) is trivial once the Boolean approximation is accepted: check C(k,2) pairs, each in constant time. The novelty is in *doing it*, not in solving a hard problem.

4. **Branch-and-bound optimizer** — **STANDARD TECHNIQUE IN NEW DOMAIN.** Branch-and-bound over a discrete parameter space with constraint propagation is textbook (Nemhauser & Wolsey). The domain-specific instantiation requires custom pruning heuristics and bounding functions, which is genuine expertise but not a research breakthrough.

5. **Bounded-latency code generation** — **NOT HARD.** The proposal explicitly acknowledges this is "static cost estimation with conservative safety margins, not formal WCET analysis." With 50–100× headroom, schedulability verification is trivial arithmetic. This is prudent engineering, not a difficulty contribution.

6. **Psychoacoustic models as compiler intrinsics** — **NOVEL IN APPLICATION, NOT IN IMPLEMENTATION.** Schroeder spreading functions and Weber JND models are published formulas implemented in dozens of audio codecs since the 1990s. The novelty is making them compiler-integrated cost functions for information MAXIMIZATION rather than compression. The formula doesn't change; the framework does. This is a genuine conceptual contribution but moderate implementation difficulty.

**Could a competent team build this faster with existing tools?** Partially. The renderer could use Faust or cpal. The parser could use pest (Rust PEG parser). The psychoacoustic models are well-documented. A team leveraging existing audio infrastructure could reduce scope by 30–40%, focusing on the genuinely novel optimizer and type system.

**Score rationale**: Real integration difficulty across 4 domains, with one genuinely hard subproblem (non-local type inference) and one interesting theoretical contribution (submodular decomposition). But no single component pushes the frontier, and 61% of the codebase is routine engineering. The cross-domain expertise requirement is a genuine multiplier, but each domain contribution is at moderate depth. **Score < 7 → amendments required.**

---

## Pillar 3: BEST-PAPER POTENTIAL — 5/10

### The Halide Comparison

The structural parallel between SoniType and Halide (PLDI 2013 Best Paper) is apt: both separate domain semantics from backend optimization over a target. But the impact comparison is dishonest. Halide targeted GPU image processing—a $50+ billion industry adopted by Google, Adobe, and Qualcomm. SoniType targets sonification—~200–500 active researchers with no significant commercial market. The impact ceiling is orders of magnitude lower. The comparison should be framed as "Halide-like in architecture" not "Halide-like in significance."

### Theorem Analysis

The 4 "Grade A" theorems are B/B+ formalizations—novel in domain application, not in mathematical depth:

**Theorem 1 (Psychoacoustically-Constrained MI):** Defining I_ψ(D; A) = I(D; ψ(A)) is a definition, not a deep theorem. "Well-defined" is trivial if ψ is measurable. "Computable in polynomial time for bounded stream counts" is vacuously true—everything is polynomial for bounded inputs, and k ≤ 8 by the cognitive load budget. The NP-hardness claim is likely a straightforward reduction from set cover or maximum coverage. The submodularity decomposition across Bark bands is the only genuinely interesting piece, but depends on the decomposability assumption which may fail for realistic broadband stimuli.

**Theorem 2 (Bregman Predicates):** Formalizing qualitative principles as Boolean threshold predicates is useful engineering. The O(k²) decidability "proof" is trivial counting: check C(k,2) pairs, each in O(1). This is not a theorem in the mathematical sense; it's a statement about the complexity of a loop. The novelty is in the formalization choice, not in the mathematics.

**Theorem 3 (Information Loss Bound):** The biconditional connecting constraint margins to information loss may follow directly from definitions. If ψ is defined as "zero out below masking threshold, quantize to JND bins, merge non-segregated streams," then information loss is by construction determined by distance from thresholds. The biconditional may just be unwinding the definition. Needs to be shown non-trivial.

**Theorem 4 (Compiler Correctness):** Standard compiler correctness adapted to a new domain. The δ bound on rounding/quantization is routine numerical analysis. Novel only in that nobody has applied compiler correctness to a psychoacoustically-constrained audio system—which is because nobody has built such a system.

**Honest assessment**: These are competent formalizations of known ideas for a new domain. They constitute a genuine contribution but are not deep mathematics that would excite a theory committee.

### Venue Analysis

**OOPSLA** (Synthesizer's recommendation, panel consensus): Best fit. OOPSLA regularly publishes domain-specific language papers (Flix, Granule, numerous DSL papers for databases and networking). The perceptual type system as a new class of refinement types fits OOPSLA's tradition. The Halide comparison is strategically appropriate here, where Halide-style papers are celebrated.

**PLDI**: Possible but risky. PLDI reviewers will scrutinize the type theory and may find the psychoacoustic constraints to be domain-specific engineering rather than fundamental PL advances.

**CHI/ASSETS**: Impossible. No human studies = no CHI. Model-based d' is not what CHI means by evaluation.

**ICAD**: Natural home but too small (~100–200 attendees, 40–60 papers/year) for prestige. Best paper at ICAD carries limited recognition.

The "no home venue" concern raised by the Skeptic was overstated—the Synthesizer's OOPSLA analysis was most persuasive and partially conceded by both other panelists.

### Diffuse Contribution Problem

The proposal claims simultaneous contributions to PL, information theory, auditory science, and real-time systems. Best papers have a single crystalline idea. The paper must be framed as a PL contribution with psychoacoustics as application domain, not an interdisciplinary omnibus. Information theory and real-time systems contributions become supporting material or separate publications.

**Score rationale**: Publishable at OOPSLA with proper framing. Competitive for distinguished paper with adversarial bug-finding evaluation and listenable audio demos. "Best paper" is aspirational but not impossible. The theorems are solid B/B+ but not best-paper mathematics. **Score < 7 → amendments required.**

---

## Pillar 4: LAPTOP CPU + NO HUMANS — 7.5/10

### Computational Feasibility: Confirmed

**Compile-time**: All optimization is compile-time. Psychoacoustic model evaluation: ~100 μs per candidate (closed-form Bark-scale spreading functions over 24 critical bands). Branch-and-bound over 10K–100K candidates = 1–10 seconds. MC MI estimation: 100–500 ms per mapping × ~100 evaluations = 10–50 seconds. Total compile time for 4–8 stream sonifications: < 60s. Worst-case with dense spectral configurations: 120s timeout with best-found fallback. **Verified plausible.**

**Render-time**: ~48 μs for 8 streams at 256-sample buffers = 0.8% CPU utilization. Even 16 streams with FM synthesis stays under 200 μs (3.4% of 5.8 ms buffer deadline). This is the same workload that decade-old audio tools handle. **Trivially feasible.**

**No GPU required**: All operations are scalar/vector math on small dimensionalities (24 Bark bands, ≤16 streams). No matrix operations. **Confirmed.**

**Zero human subjects**: Evaluation is entirely model-based. d'_model computed from psychoacoustic model predictions, not human responses. Cross-model evaluation uses different psychoacoustic models, not human subjects. **Confirmed.**

### Evaluation Circularity

The cross-model evaluation (compile with Schroeder masking + Weber JND, evaluate with Glasberg-Moore excitation patterns + Zwicker loudness) is a reasonable mitigation but does not resolve fundamental circularity. Both models are derived from the same psychoacoustic tradition (Fletcher's critical-band experiments, Zwicker's masking studies, ISO 226 equal-loudness contours). Both share structural assumptions: tonotopic frequency analysis, spreading functions, level-dependent bandwidth. Agreement between Schroeder and Glasberg-Moore tells you that two implementations of the same psychoacoustic theory are consistent—it does not tell you the theory is correct for sonification conditions. The proposal's circularity disclosure is honest ("model-based d' is a necessary and practical proxy...not a replacement for eventual human validation") but does not go far enough in acknowledging model-class-level circularity.

Additionally, "d'" computed from model predictions borrows the authority of signal detection theory (defined over human observer responses) without empirical content. The term should be "d'_model" or "model-predicted discriminability index" throughout.

### Monte Carlo MI Estimation

For low-dimensional cases (1–3 streams), 1K MC samples in ~100 ms per mapping is plausible. For 8-stream continuous multivariate data, the 5K–10K samples may be insufficient for accurate MI estimation in high dimensions. The k-NN MI estimator (Kraskov et al. 2004) has known convergence issues in dimensions >5. The 120s timeout makes this a soft constraint, not a hard failure.

### WCET Claims

The 50–100× headroom renders the WCET analysis trivially satisfiable. The proposal is honest about this. The render workload is genuinely light.

**Score rationale**: Architecturally sound. Computationally trivial on a laptop. The circularity is an evaluation methodology concern, not a computational feasibility concern (the Skeptic's attempt to score it under this pillar was overruled as a category error by the panel majority). Minor deductions for MC MI blowup risk in high dimensions and model-class circularity as a soft constraint on evaluation interpretability. **Score ≥ 7 → no mandatory amendments, but recommendations for improvement.**

---

## Fatal Flaws

### Flaw 1: Market Size / Adoption Risk
**Severity**: Severe | **Likelihood**: High (>80%)

The total addressable user base for a psychoacoustically-optimizing sonification compiler is ~200–500 active researchers. The intersection who would adopt a Rust DSL toolchain is perhaps 20–50 people worldwide. Even a perfect implementation may have negligible real-world adoption. 142K LoC (even reduced to 95K) is wildly disproportionate.

**Mitigation**: Frame as a research contribution (theorems + evaluation methodology) rather than a practical tool. Add the safety-critical alarm vertical as a high-stakes niche where formal guarantees matter. Prepare standalone salvage contributions (psychoacoustic constraint library, PL concept paper) that deliver value regardless of full compiler adoption.

### Flaw 2: Psychoacoustic Model Validity
**Severity**: Severe | **Likelihood**: Medium (50–60%)

The entire system's value rests on psychoacoustic models accurately predicting human perception. These models were calibrated for speech and music under controlled lab conditions (trained listeners, anechoic chambers, single-stimulus paradigms). Sonification involves untrained listeners, complex multi-stream stimuli, varied listening environments, cognitive load from simultaneous data interpretation, and long-duration listening. There is no evidence that lab-calibrated masking models accurately predict discriminability in these conditions. The cross-model evaluation validates within the excitation-pattern model class, not against ground truth.

**Mitigation**: (a) Validate models against published human experimental data (zero new human subjects needed). (b) Qualify all claims as "relative to psychoacoustic model M." (c) Frame the type system's guarantees like static types: they catch some bugs but not all. (d) Add model-independent signal metrics (spectral overlap ratio, temporal distinctiveness) that don't depend on any psychoacoustic model.

### Flaw 3: Standard Library LoC Inflation
**Severity**: Moderate | **Likelihood**: Certain (>95%)

28K LoC claimed for pitch scales, timbral palettes, temporal patterns, and preset recipes. Realistic estimate: 10–12K. This inflates the total from ~105K to 142K, creating unrealistic scope expectations and misrepresenting complexity.

**Mitigation**: Re-scope honestly to 10–12K. Adjust total to ≤100K.

### Flaw 4: Bark-Band Decomposability Assumption
**Severity**: Moderate | **Likelihood**: Medium (40–50%)

Theorem 1's (1-1/e) approximation guarantee depends on psychoacoustic constraints being decomposable across critical bands. This assumption fails when: (a) broadband stimuli span multiple bands (common for noise-band and granular synthesis), (b) off-frequency masking (upward spread of masking) couples adjacent bands, (c) harmonic series from pitched streams span 5–15 bands. The Schroeder spreading function explicitly models inter-band masking energy.

**Mitigation**: Empirically characterize the approximation gap for broadband stimuli. Provide fallback to global optimization (slower but exact) when decomposition error exceeds a threshold. Qualify the approximation guarantee as holding for spectrally disjoint streams.

### Flaw 5: Type System May Be Shallow
**Severity**: Moderate-to-Severe | **Likelihood**: Medium (40–50%)

If the "perceptual type system" is constraint checking with PL vocabulary rather than genuine type-theoretic machinery (formal typing rules Γ ⊢ e : τ, operational semantics, soundness proof), the PL contribution evaporates and the OOPSLA targeting fails. The proposal hints at refinement types / liquid types but doesn't commit to actual refinement-type infrastructure with inference rules and metatheory.

**Counterpoint** (from adversarial debate): Liquid Haskell's refinement types are also "predicate checking via SMT solver" at a certain abstraction level. The distinction between "constraint checking" and "type system" lies in compositionality, decidability guarantees, and soundness relative to a semantic model—all of which SoniType claims. However, until formal typing rules and a soundness proof sketch are produced, the "type system" label remains aspirational.

**Mitigation**: Substantiate with formal typing rules before committing to PL venue. If this cannot be done, rename to "constraint verification framework" and target ICAD instead of OOPSLA.

### Flaw 6: No LLM Baseline
**Severity**: Moderate | **Likelihood**: High (damaging if unaddressed)

GPT-4/Claude can generate SuperCollider or Sonic Pi code from natural-language sonification specifications. If LLM + 30 minutes of human iteration produces model-predicted d'_model comparable to SoniType's optimized output, the formal compiler's marginal value collapses.

**Mitigation**: Add LLM-generated sonification as an evaluation baseline. If SoniType significantly outperforms, this strengthens the case. If not, the value proposition must be reconsidered.

### Flaw 7: Evaluation Circularity Deeper Than Disclosed
**Severity**: Moderate | **Likelihood**: Medium-High (60%)

Both evaluation models (Schroeder, Glasberg-Moore) share structural assumptions from the same psychoacoustic tradition. Cross-model consistency validates within the model class, not against structurally independent models. True independence would require a fundamentally different approach (e.g., auditory nerve simulation, deep learning-based auditory models).

**Mitigation**: Explicitly acknowledge model-class-level circularity. Add model-independent signal metrics. Frame the evaluation as "validating correctness relative to the psychoacoustic model" (analogous to testing a compiler against a language spec), with model accuracy deferred to the psychoacoustics community.

---

## Binding Amendments

These are **mandatory prerequisites** for continuation. They are not suggestions.

### Amendment 1: SCOPE REDUCTION
Reduce the LoC target from 142K to ≤100K. Define a minimum viable compiler (MVC) of ~50K LoC containing all novel intellectual contributions: DSL parser, perceptual type-checker, psychoacoustic cost models, information-theoretic optimizer, and minimal evaluation framework. Standard library capped at 10–12K. CLI/tooling capped at 8K. Preview server, debug visualization, and extended standard library are Phase 2.

### Amendment 2: HONEST LoC REPORTING
All published materials must report the novel core (~45–50K LoC) separately from routine engineering (~45–50K supporting infrastructure). The LoC table must be split into "Novel Core" and "Supporting Infrastructure" sections. Leading with 142K when the novel core is 45–50K undermines credibility with technically sophisticated reviewers.

### Amendment 3: VENUE COMMITMENT — OOPSLA
Frame the paper as a PL contribution targeting OOPSLA. Lead with the perceptual type system as a new class of refinement types where psychoacoustic constraints are first-class type qualifiers. Psychoacoustics is the application domain, not a co-equal contribution. Information theory and real-time systems aspects become supporting material or separate publications. Structure the paper as: (1) perceptual types as refinement types, (2) type-checking algorithm with soundness proof, (3) optimizer as constraint solver, (4) evaluation showing type system catches real sonification bugs.

### Amendment 4: TYPE SYSTEM SUBSTANTIATION
Before claiming "type system," produce: (a) formal typing rules in Γ ⊢ e : τ form, (b) an operational semantics for the DSL, (c) at least a proof sketch of soundness (type safety relative to the psychoacoustic model). If this cannot be done, rename to "constraint verification framework" and re-target ICAD. **This is the single highest-risk item.** Kill gate: type system formalization must be complete by week 4 or the project pivots.

### Amendment 5: QUALIFY ALL MODEL-BASED METRICS
Use "d'_model" throughout, not bare "d'." Use "model-predicted discriminability" not "discriminability." Every evaluation claim must be explicitly conditioned on the psychoacoustic model's validity. Add a paragraph noting that cross-model evaluation validates within the excitation-pattern model family, not against structurally independent models or ground-truth human data.

### Amendment 6: LLM BASELINE
Add LLM-generated sonification code (GPT-4/Claude generating SuperCollider/Sonic Pi from natural language specifications) as an evaluation baseline. Test whether the formal compiler adds value over AI-assisted sonification on the project's own model-predicted metrics. **Kill gate: if SoniType cannot demonstrably outperform the LLM baseline by week 6, reconsider the value proposition.**

### Amendment 7: PUBLISHED HUMAN-DATA ANCHOR
Validate the psychoacoustic model against at least one published human experimental dataset (classic streaming, JND, or masking experiments from Bregman, Moore, Walker & Nees). Compare model-predicted d'_model for specific configurations to published human d' values. This costs zero human subjects—comparing to published data—and provides critical grounding. If the model predicts d'_model = 2.1 for a configuration where humans measured d' = 1.9, that's strong validation. If the model predicts d'_model = 2.1 where humans measured d' = 0.5, the model is broken. **Kill gate: human-data anchor must be completed by week 8 or claims must be downscoped.**

---

## Recommended Amendments (Non-Binding)

These are recommended by the Synthesizer and endorsed by the panel but not mandatory:

- **Safety-critical alarm vertical**: Reframe the primary motivating example around medical alarm fatigue and IEC 60601-1-8 gap. This is the highest-stakes use case where formal discriminability guarantees have indisputable value.
- **Perceptual linting positioning**: Position SoniType not only as a compiler but as a perceptual linter that checks existing sonification designs for masking/JND/segregation violations. Dramatically lowers adoption barrier.
- **Adversarial bug-finding benchmark**: Create a benchmark of known-bad sonification designs from ICAD literature (designs that failed in human studies). Show the type-checker rejects every known-bad design. This is the "type systems catch real bugs" argument PL reviewers find compelling.
- **Listenable audio supplementary materials**: Include audio files so reviewers can hear the difference. Audio is profoundly persuasive in a way that d'_model numbers are not.
- **Model-independent signal metrics**: Add spectral overlap ratio, temporal distinctiveness, and parameter space coverage as evaluation metrics independent of any psychoacoustic model.
- **Parallel salvage contributions**: Prepare standalone outputs (psychoacoustic constraint library, PL concept paper) in parallel with full compiler development.

---

## Salvage Options (If Full Compiler Doesn't Ship)

### Tier 1 — High Standalone Value
- **Psychoacoustic Constraint Library** (~15K LoC, Rust/Python): Formalized Bregman predicates, cognitive load algebra, masking/JND cost models as a standalone library. API: `check_masking(stream_a, stream_b) → MaskingResult`. Immediately useful to anyone building sonifications in any tool. Publishable at ICAD or JOSS.
- **Perceptual Type System Concept Paper** (6 pages, DSLDI/TyDe/PEPM workshop): Formalize psychoacoustic refinement types with typing rules, soundness argument, and 2–3 worked examples. Stakes the intellectual claim while the full system is built. Estimated effort: 3–4 weeks.

### Tier 2 — Moderate Standalone Value
- **Masking Checker for Existing Tools** (~8K LoC): Standalone linting tool that reads MIDI/audio configurations from existing tools (TwoTone, Highcharts Sonification) and checks for masking violations and JND failures. High practical adoption potential.
- **I_ψ Evaluation Framework** (~12K LoC, Python): Standalone evaluation framework for any sonification design. Publishable at ICAD as a methodology paper.

---

## Verdict

**CONDITIONAL CONTINUE** at 22.5/40 (V4/D6/BP5/L7.5).

The project has a genuinely novel intellectual core: framing sonification as lossy coding over a psychoacoustically-constrained perceptual channel, with a type system that mechanizes psychoacoustic constraints into automated verification. The compile-time/render-time separation is architecturally sound and makes laptop feasibility trivial. The cross-domain integration across PL, psychoacoustics, information theory, and real-time audio is ambitious and intellectually interesting.

However, the project is **over-scoped** (142K → ≤100K), **under-focused** (omnibus framing dilutes each contribution), and **inflated** in several metrics (LoC, theorem depth, Halide impact comparison, EU Accessibility Act urgency). The evaluation methodology has honest but bounded circularity, and the "type system" label is aspirational until formal typing rules are produced.

All 7 binding amendments are mandatory prerequisites, not suggestions. The type system substantiation (Amendment 4) is the highest-risk item—if the perceptual type system cannot be formalized as genuine PL machinery, the project must pivot to a "sonification constraint verification tool" without PL venue aspirations. The LLM baseline (Amendment 6) is the value-proposition stress test—if the compiler cannot outperform AI-assisted sonification on its own metrics, the project's raison d'être collapses.

**Kill gates**:
1. Type system formalization by week 4, or pivot to constraint verification framing
2. LLM baseline comparison by week 6, or abandon if no improvement demonstrated
3. Published human-data anchor by week 8, or downscope all perceptual claims

**Probability of meeting all binding conditions**: ~60%.

**Expected outcome if conditions met**: Solid OOPSLA publication with a novel domain-specific type system, a useful (if niche) research tool, and 2–4 standalone publishable contributions from salvage options. Not field-transforming, but genuinely good interdisciplinary research.

**Expected outcome if conditions not met**: An impressive but unused Rust project on GitHub with minimal adoption, possibly yielding a workshop paper and a constraint library.

---

*Assessment produced by 3-expert verification panel (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) with structured adversarial cross-critique and consensus building. All disagreements documented in adversarial critique record. Verdict represents panel majority position (3-0 CONDITIONAL CONTINUE; Skeptic at lower confidence).*

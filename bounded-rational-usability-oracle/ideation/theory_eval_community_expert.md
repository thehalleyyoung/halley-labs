# Theory Evaluation: Community Expert Assessment

**Proposal:** proposal_00 — The Cognitive Regression Prover: A Three-Layer Usability Oracle with Incremental Formal Guarantees  
**Area:** area-042-human-computer-interaction  
**Evaluator:** HCI Community Expert (verification team: Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer)  
**Method:** Three-expert adversarial panel with independent proposals, cross-critiques, and synthesis  
**Date:** 2026-03-04

---

## Evaluation Process

Three independent expert roles assessed this proposal in parallel, then engaged in adversarial cross-critique before synthesis:

- **Independent Auditor:** Evidence-based scoring with challenge testing against proposal claims
- **Fail-Fast Skeptic:** Adversarial attacks on every axis, seeking reasons to ABANDON
- **Scavenging Synthesizer:** Amendment audit, minimum viable contribution analysis, risk-adjusted scoring

All three delivered independent assessments before seeing each other's work. Cross-critique resolved disagreements through direct adversarial challenges.

---

## proposal_00: The Cognitive Regression Prover

### Summary

A three-layer system for automated structural usability regression detection in CI/CD pipelines. Layer 1: lean accessibility-tree profiler with parameter-free verdicts via interval arithmetic. Layer 2: bounded-rational MDP analysis with paired-comparison theorem and cognitive fragility metric. Layer 3: bisimulation-based scaling (deferred). Claims to deliver deterministic, quantitative, formally grounded usability regression verdicts without human studies, GPUs, or parameter calibration for dominant failure modes.

### Amendment Audit (from prior depth check)

The prior depth check required 8 amendments. Assessment:

| Amendment | Status | Quality |
|-----------|--------|---------|
| A: Break evaluation circularity | **ADDRESSED** | Strong — CogTool data, τ ≥ 0.6 threshold, explicit kill-switch |
| B: Formalize paired-comparison | **ADDRESSED** | Strong — formal theorem, numerical example, honest qualifications |
| C: Cost algebra soundness | **PARTIALLY** | Restated as ordinal; proof remains "conjectured" |
| D: Scope detection boundary | **ADDRESSED** | Strong — clear, complementary to screenshot-diff tools |
| E: LLM comparison | **ADDRESSED** | Adequate — four-point argument, could be deeper |
| F: MVP path | **ADDRESSED** | Strong — three-layer architecture with standalone Layer 1 |
| G: Downscope repair | **ADDRESSED** | Strong — clean relegation to stretch goal |
| H: Parameter calibration | **PARTIALLY** | β handled via range-over-β; γ/α deferred to Layer 3 |

**6 of 8 fully addressed, 2 partially.** No amendments ignored. The two partial addresses are honest about their gaps.

---

## Scores

### 1. Extreme Value: **5/10**

| Expert | Score | Key Argument |
|--------|-------|-------------|
| Auditor | 6 | Real gap between a11y linters and manual studies; contingent on ordinal validity |
| Skeptic | 4 | 50-line axe-core diff script captures parameter-free cases; CogTool adoption failure unaddressed |
| Synthesizer | 6 | Risk-adjusted; 30% validation failure risk pulls down optimistic 8 |
| **Consensus** | **5** | |

**What the team agrees on:** The problem is real — no CI/CD tool produces structural usability regression verdicts today. The gap between axe-core (WCAG violations) and manual usability studies is genuine and painful for design-system teams.

**What pulls the score down:**
- The Skeptic's "trivial baseline" challenge is potent: a 50-line script wrapping axe-core element counts catches the same dominant failure modes the proposal calls "parameter-free." The interval arithmetic adds principled cost ranges but the *regression verdict* for monotone cases is determined by structural predicates, not cost arithmetic.
- Ordinal validity (do the model's cost orderings match human-perceived usability orderings?) is completely undemonstrated. The proposal assigns 30% failure probability to retrospective validation.
- The 50-70% coverage estimate for parameter-free verdicts is unsubstantiated — it could be 20%.
- CogTool had 15+ years of attention and achieved approximately zero CI/CD adoption. The adoption barrier (teams must understand and trust cognitive cost models) has not changed.
- Accessibility trees are inconsistent across browsers; the WebAIM 2023 survey found 96.3% of home pages had WCAG failures, implying messy trees.

**Proposal self-score: 9.** **Assessment: Significantly optimistic.** A 9 implies desperate, near-universal need with clear adoption. The reality: value is contingent on unproven ordinal validity, faces trivial-baseline competition on easy cases, and has an unprecedented adoption path.

---

### 2. Genuine Software Difficulty: **6/10**

| Expert | Score | Key Argument |
|--------|-------|-------------|
| Auditor | 7 | Bisimulation and paired-comparison proof genuinely novel; tree parser is known-hard |
| Skeptic | 5 | ~5-7K lines genuinely novel; bisimulation speculative; engineering ≠ research difficulty |
| Synthesizer | 6 | Probability-weighted: Layer 1 (4-5), Layer 2 (7-8), Layer 3 (8-9) discounted by completion probability |
| **Consensus** | **6** | |

**Genuinely novel and hard:**
- Bounded-rational bisimulation (novel algorithm that does not yet exist — 40% intractability risk)
- Paired-comparison theorem proof (error cancellation under shared MDP abstraction — medium difficulty, proof sketch exists)
- Accessibility-tree-to-MDP reduction (where prior automated usability tools have died)
- Ordinal soundness of cost algebra (conjectured, incomplete proof, "Hard" by proposal's own rating)

**Standard engineering (not novel):**
- Interval arithmetic over Fitts'/Hick's law (~hundreds of lines)
- Semantic tree alignment (adaptation of RTED)
- Monte Carlo trajectory sampling (textbook)
- Softmax policy computation via value iteration (textbook)
- CI/CD integration (boilerplate)

**The Skeptic's code estimate:** ~25K-46K total LoC, of which ~7-14K genuinely novel (~30%). If bisimulation fails (40% likely), genuinely novel code drops to ~5-7K lines.

**Resolution:** The Auditor scores potential (how hard is what's proposed); the Skeptic scores reality (what's provably achievable). For a proposal evaluation, the right frame is expected difficulty weighted by completion probability. Layer 1 + Layer 2 (the realistic deliverable after scope cuts) is difficulty 6.

**Proposal self-score: 7.** **Assessment: Roughly accurate** for the full vision; slightly optimistic for the scoped-to-Layers-1-2 version.

---

### 3. Best-Paper Potential: **4/10**

| Expert | Score | Key Argument |
|--------|-------|-------------|
| Auditor | 5 | Framing novel; paired-comparison is legitimate "aha"; but zero empirical results, incomplete proofs |
| Skeptic | 3→4 | Error cancellation not surprising; parameter-independence trivial; fragility = sensitivity analysis; unproven conjecture |
| Synthesizer | 6 | Narrative strong but ~17% probability of full best-paper narrative landing |
| **Consensus** | **4** | |

**What works:**
- The consistency-oracle framing (regression detection as differential inference, strictly weaker than absolute prediction) is genuinely novel. No prior work formalizes this for usability.
- The parameter-independence result — dominant regression types yield zero-calibration structural predicates — is elegant and practically significant, even if mathematically trivial.
- The three-move narrative (parameter-free → paired-comparison tightness → fragility) is a satisfying arc.

**What fails:**
- **Zero empirical results.** No worked examples, no preliminary data, no evidence the pipeline produces correct verdicts on any real UI. This is disqualifying at CHI, UIST, or any HCI venue for best-paper consideration.
- **Incomplete proofs.** Theorem 1 (paired-comparison): proof sketch, not complete. Theorem 3 (fragility decomposition): sketch with unresolved interaction effects. Theorem 4 (cost algebra): conjectured, previously invalidated formulation.
- **The Skeptic's deflation is partially valid:** The paired-comparison theorem IS error cancellation under shared bias (a known statistical phenomenon). The formalization for MDPs is genuine but the surprise value is modest. The fragility metric IS max-min sensitivity analysis. The bottleneck taxonomy IS Wickens' MRT repackaged.
- **Venue mismatch:** UIST demands a working demo (none exists). CHI demands human validation (none planned). The cross-disciplinary framing falls between stools.
- **Probability of full narrative:** ~17% that all pieces land simultaneously. Probability of a strong UIST contribution (not best-paper): ~50-55%.

**Proposal self-score: 8.** **Assessment: Significantly optimistic.** An 8 implies high probability best-paper contender. The current state is a blueprint with genuine ingredients but no execution. The realistic ceiling is "strong UIST contribution" — not best paper.

---

### 4. Laptop-CPU Feasibility & No-Humans: **8/10**

| Expert | Score | Key Argument |
|--------|-------|-------------|
| Auditor | 8 | Principled CPU-native design; retrospective validation clever |
| Skeptic | 8 | Genuine strength; minor timing and data availability concerns |
| Synthesizer | 8 | Optimistic 9, pessimistic 7, risk-adjusted 8 |
| **Consensus** | **8** | |

**This is the proposal's strongest axis — unanimous.**

The CPU-only design is principled and well-argued:
- Accessibility trees for structural saliency (no vision models, no pixels)
- Monte Carlo sampling embarrassingly parallel on CPU cores
- No training phase — calibrated from published psychophysical parameters
- SMT/ILP solvers (Z3, CBC) are CPU-native

The no-humans evaluation plan is credible:
- Retrospective ordinal validation against existing published human data
- Issue-tracker annotations from open-source projects
- Synthetic mutations with known bottleneck types
- All automated, all reproducible

**Minor concerns:**
- Headless browser required for accessibility tree extraction (adds overhead)
- CogTool-era datasets may not have reconstructible accessibility trees
- Layer 2 fragility analysis exceeds 60s CI/CD budget (acknowledged; falls back to nightly)
- ≤10K state-space claim is empirical but unsubstantiated

---

### 5. Overall Feasibility: **5/10**

| Expert | Score | Key Argument |
|--------|-------|-------------|
| Auditor | 6 | Layered architecture enables graceful degradation; aggressive but structured timeline |
| Skeptic | 4 | Three near-fatal issues; compounding risks across proofs/parsing/evaluation |
| Synthesizer | 6 | Optimistic 8, pessimistic 4, risk-adjusted 6 |
| **Consensus** | **5** | |

**Timeline is aggressive.** Layer 1 alone (8 weeks) requires:
- Accessibility-tree parser + cross-browser normalizer (where prior tools have died)
- Semantic tree alignment (three-pass algorithm)
- Task-flow specification DSL
- Additive cost model with interval arithmetic
- Parameter-independent verdict engine
- CI/CD integration
- Benchmark curation

The accessibility-tree parser alone is 8+ weeks for a senior engineer based on the complexity described and prior failure history. Cross-browser normalization is a years-long industry problem, not a 3-4 week task.

**Compounding risk profile** (from proposal's own estimates):
- 30% retrospective validation fails
- 25% tree quality insufficient
- 40% bisimulation intractable (Layer 3)
- 35% soundness proof fails (Layer 3)

Probability of full delivery: ~30%. Probability of Layers 1-2 delivering a working tool + at least one strong theorem: ~50-60%.

**Key mitigant:** The incremental architecture. Layer 1 delivers standalone value. Layer 2 adds theoretical depth. Layer 3 is future work. Partial success is still publishable.

**Proposal self-score: 8.** **Assessment: Optimistic by ~3 points.** An 8 implies high confidence of full delivery. The reality warrants "achievable with significant scope reduction."

---

## Score Summary

| Axis | Score | Proposal Self-Score | Gap |
|------|-------|-------------------|-----|
| 1. Extreme Value | **5** | 9 | -4 |
| 2. Genuine Software Difficulty | **6** | 7 | -1 |
| 3. Best-Paper Potential | **4** | 8 | -4 |
| 4. Laptop-CPU & No-Humans | **8** | 8* | 0 |
| 5. Overall Feasibility | **5** | 8 | -3 |
| **COMPOSITE** | **28/50** | 40/50† | -12 |

*Proposal conflates CPU/no-humans with overall feasibility.  
†Proposal scores 4 axes at 32/40; normalized to 50 scale ≈ 40.

---

## Fatal Flaws

**No single flaw is fatal.** However, three flaws compound to create significant existential risk:

1. **Retrospective validation may be infeasible.** The published datasets (CogTool, Oulasvirta, Findlater & McGrenere 2004) are 10-20 years old. The UIs may not be reconstructible as accessibility trees. The consistency-oracle claim has no external anchor without this validation. (30% failure probability, proposal's own estimate.)

2. **The per-PR CI gate is limited to Layer 1.** Layer 2's theoretical apparatus (paired-comparison theorem, fragility metric, cliff detection) exceeds the 60-second CI/CD budget and is relegated to nightly builds. The flagship theoretical contributions do not serve the flagship use case.

3. **Cross-browser accessibility tree normalization is unsolved at industry scale.** The Accessibility Interop project has worked on this for years and is not complete. Scoping to Chrome-only is the pragmatic answer but limits the "any team with accessibility trees" claim.

---

## Key Disagreements and Resolutions

### The "Trivial Baseline" Challenge (Skeptic)
A 50-line axe-core diff script catches the same dominant failure modes called "parameter-free." **Resolution:** The trivial baseline MUST be implemented as an explicit benchmark (Condition 5). The formal framework must demonstrate Δτ ≥ 0.08 improvement in ordinal agreement to justify its complexity. If it cannot beat the trivial competitor, the complexity is not warranted.

### The "Theory is Ornamental" Challenge (Skeptic)
The paired-comparison theorem is "just error cancellation"; parameter-independence is "just monotonicity"; fragility is "just sensitivity analysis." **Resolution:** The Skeptic's characterizations are technically accurate at the high level but rhetorically deflating. Formalizing error cancellation for MDP abstractions under shared partitioning IS genuine work. The recognition that monotonicity yields zero-config CI verdicts IS a publishable insight. The Skeptic's own steel-man acknowledges: "The consistency-oracle framing is genuinely novel and correct... this framing alone is a publishable insight."

### The "Incremental Architecture" Insight (Synthesizer)
**This was the most important finding.** The project should be evaluated as a call option, not a binary bet. Layer 1 alone (8 weeks, ~70-80% success probability) delivers a working tool and a publishable insight. Layers 2-3 add upside but their failure doesn't destroy Layer 1's value. This optionality is what resolves the Skeptic's "below threshold" composite score into a CONTINUE verdict.

---

## Hard Conditions for Continuation

| # | Condition | Deadline | Failure → |
|---|-----------|----------|-----------|
| 1 | **Validation data exists:** ≥15 UI pairs with reconstructible accessibility trees AND published human orderings | Week 2 | **ABANDON** consistency-oracle claim |
| 2 | **End-to-end proof of life:** Layer 1 running on Chrome, cost diff for ≥3 real UI pairs | Week 4 | **ABANDON** if parser infeasible |
| 3 | **Drop Theorem 4** from contributions (cost algebra ordinal soundness is unproven conjecture) | Immediate | Required for honest self-assessment |
| 4 | **Scope to Layers 1-2 only** — defer Layer 3 (bisimulation, cost algebra) to future work | Immediate | Required; 9-month plan → 6-month plan |
| 5 | **Trivial baseline benchmark:** implement 50-line axe-core diff; Layer 1 must show Δτ ≥ 0.08 | Week 6 | Rethink cost model |
| 6 | **Paired-comparison proof complete** with explicit constants, OR reframe as empirical paper | Month 4 | Venue/narrative downgrade |

---

## The "3-Month Minimum" Version (from Synthesizer)

If scope must be cut to 3 months:

| Week | Deliverable |
|------|-------------|
| 1-2 | Accessibility-tree parser + normalizer (Chrome/CDP only) |
| 3-4 | Semantic tree alignment (three-pass algorithm) |
| 5-6 | Additive cost model + interval arithmetic + parameter-independent verdict engine |
| 7-8 | CI/CD integration (GitHub Action), CLI, JSON/SARIF output |
| 9-10 | Retrospective validation against published data (τ ≥ 0.6 threshold) |
| 11-12 | Benchmark suite (50 pairs), ablation studies, paper writing |

**Publishable as:** UIST short paper, ICSE tool track, or CHI late-breaking work.  
**Loses:** Formal error bounds, behavioral modeling, fragility analysis, best-paper narrative.  
**Keeps:** Working tool, parameter-free insight, retrospective validation.

---

## Verdict

### **CONTINUE**

**Confidence: Moderate.** All three independent experts converged on CONTINUE despite sustained adversarial attacks and scores ranging from 24-32/50. The Skeptic's inability to recommend ABANDON despite six attack vectors is the strongest signal.

**Why CONTINUE:**
1. **Real unsolved problem** (unanimous)
2. **Correct and novel framing** (consistency oracle, acknowledged even by Skeptic)
3. **Well-designed incremental architecture** (unanimous — each layer delivers standalone value)
4. **Strong CPU-only feasibility** (unanimous 8/10)
5. **At least one publishable insight** (parameter-independence for zero-config regression verdicts)
6. **Call-option structure:** Layer 1 investment (8 weeks) has bounded downside and substantial upside

**Why NOT a stronger endorsement:**
1. Self-scores inflated by 2-4 points across three axes (Value, Potential, Feasibility)
2. Zero empirical results — the proposal is a blueprint, not a contribution
3. Three of four theorems are incomplete or unproven
4. 30% chance the entire consistency-oracle claim cannot be validated
5. Best-paper probability estimated at ~17%; strong-UIST-contribution probability ~50-55%
6. The theoretical contributions are genuine but modest — "a solid research project, not a theoretical breakthrough"

**What CONTINUE means:**
- Execute Layer 1 (Weeks 1-8, Chrome-only)
- Hit Week 2 validation-data gate and Week 4 proof-of-life gate
- If both pass: proceed to scoped Layer 2 (Months 3-6)
- Target: UIST 2026 full paper (primary), ICSE 2027 tool track (fallback)
- Do NOT plan for Layer 3 unless Layers 1-2 complete ahead of schedule
- Reduce self-assessed scores to expert consensus before external presentation

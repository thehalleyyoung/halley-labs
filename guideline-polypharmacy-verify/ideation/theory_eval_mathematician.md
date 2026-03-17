# Mathematician's Verification Evaluation — GuardPharma (proposal_00)

**Evaluator:** Deep Mathematician (team-led: Independent Auditor + Fail-Fast Skeptic + Scavenging Synthesizer)
**Slug:** `guideline-polypharmacy-verify`
**Stage:** Verification (post-theory)
**Date:** 2026-03-08
**Method:** 3-expert team with independent proposals → adversarial cross-critique → forced resolution → verification signoff

---

## Executive Summary

GuardPharma proposes a two-tier formal verification engine for polypharmacy safety: pharmacokinetic abstract interpretation (Tier 1, fast screening) layered atop contract-based compositional model checking (Tier 2, precise diagnostics). The mathematical crown jewel is Theorem 1: CYP-enzyme interface contracts with Metzler monotonicity enabling single-pass compositional verification, reducing N-guideline verification from exponential to polynomial complexity.

**Composite: 4.9/10. CONDITIONAL CONTINUE.**

The project has one genuine mathematical insight (Metzler monotone chain for single-pass contract resolution), a novel formalism (Pharmacological Timed Automata), and addresses a real clinical gap. However: theory_bytes = 0, impl_loc = 0, the "crown jewel" theorem covers only 45–55% of clinically significant drug interactions (not 70% as claimed), E1 has a 30–40% probability of collapsing the core narrative, and zero demand signal exists from any stakeholder. The project's fate hinges on producing bytes — proofs, code, data — instead of more meta-evaluation documents in the next 4 weeks.

**Team vote:** Auditor: CONDITIONAL CONTINUE (5.4/10). Skeptic: CONDITIONAL ABANDON (3.8/10). Synthesizer: CONTINUE with reframing (6.8/10 raw, 6.3 adjusted). Resolution: CONDITIONAL CONTINUE with hard gates, adopting the Synthesizer's HSCC reframing as primary venue strategy and the Skeptic's timeline pressure as execution discipline.

---

## 1. Mathematical Load-Bearing Assessment

### The Mathematician's Core Question

Is the math in GuardPharma load-bearing — the reason the artifact is hard to build and the reason it delivers extreme value? Or is it ornamental math draped over an engineering project?

**Answer: The math is partially load-bearing, partially ornamental, and the most interesting math is conveniently deferred.**

### 1.1 Theorem 1: Contract-Based Compositional Safety — Depth: 5/10

**Claimed depth:** 6/10. **Resolved depth:** 5/10 (5.5/10 if M(c) fix succeeds).

**What it is:** Assume-guarantee reasoning (Pnueli 1985) instantiated to CYP-enzyme interface contracts. N-guideline verification decomposes into N individual checks + M enzyme-compatibility checks. The proof exploits a four-step monotone chain:

1. Lemma 1.1 (Metzler comparison): Lower enzyme capacity → higher drug concentrations
2. Lemma 1.2 (Michaelis-Menten): Higher concentrations → higher enzyme inhibition load
3. Lemma 1.3 (Single-pass soundness): Worst-case guarantees under minimum capacity are sound upper bounds
4. Lemma 1.4 (A/G composition): Contract compatibility implies product safety

**The one genuine insight:** The Metzler monotonicity of PK dynamics creates a natural monotone operator on enzyme-capacity guarantees, resolving the circular dependency (concentrations depend on enzyme activity, which depends on concentrations) in a single pass without fixed-point iteration. This is not a textbook construction — it requires simultaneously understanding competitive inhibition kinetics and assume-guarantee proof obligations. A reviewer who grasps both domains will recognize this as a real insight.

**What is NOT novel:**
- The A/G framework is 40 years old (Pnueli 1985). Lemma 1.4 is a direct citation.
- The Metzler comparison theorem is textbook positive systems theory (Farina & Rinaldi 2000).
- Lemma 1.2 is the derivative of the Michaelis-Menten equation — an undergraduate calculus exercise.
- The steady-state analysis C_ss = (F·dose)/(CL·τ) is standard pharmacokinetics.

**Critical proof gap (M(c) state-dependence):** The proof sketch assumes M is a constant Metzler matrix, but under CYP inhibition, M = M(c) depends on drug concentrations. The system is nonlinear: dc/dt = M(c)·c + B·d(t). The standard Metzler comparison theorem does not directly apply. The proof needs cooperative systems theory (Smith 1995, "Monotone Dynamical Systems") to show that the PK system with competitive inhibition is cooperative (Jacobian has non-negative off-diagonal entries). The proof sketch implicitly assumes this but never cites the correct theory. This is the most serious gap — a formal methods reviewer will catch it immediately.

**The coverage problem (45–55%, not 70%):** The Skeptic's attack on the 70% CYP coverage claim is substantially correct on clinical framing. The 70% figure refers to PK DDIs mediated by CYP enzymes. But:
- Mechanism-based (irreversible) inhibition (~8% of PK DDIs): violates competitive inhibition assumption
- Active metabolites (~5%): breaks simple parent-drug monotonicity chain
- Autoinduction (~3%): reverses monotonicity direction entirely
- Pharmacodynamic interactions (QT prolongation, serotonin syndrome, CNS depression): the most clinically dangerous polypharmacy interactions, 30–40% of serious ADEs in elderly polypharmacy

After proper accounting: Theorem 1 covers **65–70% of PK DDIs** but only **45–55% of clinically significant drug interactions in the target polypharmacy population**. The monolithic fallback handles the remainder, but the "crown jewel" covers less than half the clinically relevant problem.

**Verdict on Theorem 1:** One genuine domain-specific insight (Metzler single-pass resolution) embedded in standard A/G scaffolding. Load-bearing for the ~50% of interactions it covers. Proof has a real gap (M(c)) that is likely fixable but demonstrably unwritten. Depth 5/10 — solid applied verification, not mathematical frontier.

### 1.2 Proposition 2: PK-Aware Widening — Depth: 4/10

**Claimed depth:** 5/10. **Resolved depth:** 4/10.

Domain-specific widening operator (Cousot 1992 tradition) that exploits Metzler steady-state bounds. Convergence bound O(D·k) follows from standard monotone-convergence argument: enzyme-load sequences are increasing and bounded. The bound may be O(D·k²) for strongly coupled drugs — acknowledged but unresolved.

**Load-bearing for Tier 1:** Without PK-aware widening, abstract interpretation either diverges or produces vacuous results. The widening makes Tier 1 useful. But this is a good engineering design choice, not a mathematical contribution.

**Hidden precision problem:** The widening to [0, C_ss,max(φ_worst)] may produce intervals so wide they're useless for CYP3A4-sharing drugs (~50% of all drugs). A patient on 5 CYP3A4 substrates gets a widened enzyme-activity interval [E_min, E_max] that classifies everything as "possibly unsafe," making Tier 1 a no-op for the most common enzyme. A meaningful result would bound the *precision* of the fixed point, not just the number of iterations. This is missing.

### 1.3 Observation 3: δ-Decidability — Depth: 1.5/10

**Correctly labeled "Observation."** This is "we will use dReal" — a library invocation with domain-specific δ-calibration. The per-drug δ formula is sensible engineering. Not a mathematical contribution. Should not appear in any theorem list.

### 1.4 Deferred: CQL Compilation Correctness — Depth: 7/10 (MISSING)

The most mathematically interesting contribution — bisimulation or trace-refinement between CQL operational semantics and compiled PTA — is deferred to a follow-on paper. The Math Depth Assessor and every reviewer identified this as the highest-value mathematical contribution. Its deferral means the paper's strongest potential result is explicitly out of scope.

The Synthesizer correctly identifies that pulling even a core CQL fragment's formal semantics into scope would transform the mathematical profile. This is the biggest missed opportunity.

### 1.5 The PTA Formalism — Depth: 5/10

The Synthesizer argues the PTA formalism itself (timed automata + compartmental ODE + CYP-enzyme semantics) is a novel class of hybrid automata worth 7/10. The Skeptic dismisses it as "pharmacological variable names on known structures."

**Resolved:** The PTA occupies a genuinely empty niche — no prior work combines timed automata with Metzler ODE dynamics and enzyme-interface contract semantics. The Metzler structure constraint is not decorative; it enables both the abstract interpretation (widening to steady-state bounds) and the contract decomposition (monotone guarantee functions). The formalism is a legitimate contribution at the level of *identifying a useful decidable/tractable fragment* of nonlinear hybrid automata.

However, the comparison to LHA (Henzinger 1995) is unfounded. LHA enabled an entire subfield with hundreds of follow-on papers. PTA will enable near-zero follow-on work because it's too domain-specific for general hybrid systems and too abstract for clinical pharmacologists. **Depth: 5/10** — novel fragment identification, not a field-creating definition.

### 1.6 Hidden Theorems Not Yet Identified

The Synthesizer correctly identifies several hidden results that could strengthen the paper:

| Hidden Result | Depth | Effort | Value |
|---|---|---|---|
| Tiered Soundness (Tier 1 + Tier 2 compose soundly) | 4/10 | Low (framing exercise) | Converts architecture from engineering choice to proved property |
| PSPACE-hardness of PTA reachability (standard reduction) | 2/10 | ~1 day | Justifies two-tier architecture and contract decomposition |
| Schedule Separability (τ_min for dose spacing under competitive inhibition) | 3/10 | Low (PK analysis) | Maximally clinically actionable |
| SAT Encoding Soundness (discretization error bounded by δ/2) | 3.5/10 | Medium | Required for end-to-end soundness argument |

None of these are deep, but collectively they convert the paper from "one theorem + engineering" to "a formalism with a constellation of supporting results."

### 1.7 Composite Mathematical Depth

| Result | Depth | Load-Bearing? | Status |
|---|---|---|---|
| PTA formalism | 5.0/10 | Yes — everything depends on it | Defined in approach.json, not publication-quality |
| Theorem 1 (contracts) | 5.0/10 | Yes — 50% of interactions | Sketch with known M(c) gap |
| Proposition 2 (widening) | 4.0/10 | Yes — Tier 1 termination | Convergence bound unproven |
| Observation 3 (δ-decidability) | 1.5/10 | Ornamental | Library invocation |
| CQL bisimulation (deferred) | 7.0/10 | Would be highest | Not in scope |
| Hidden results (4 items) | 3.0/10 avg | Supporting | Not yet written |

**Weighted mathematical depth: 4.5/10** (penalty for theory_bytes = 0; all proofs remain sketches).

**The mathematician's bottom line:** One genuine insight, one novel formalism, competent supporting results, but nothing proven. This is adequate math supporting good engineering, not math that pushes the frontier. The most interesting theorem is deferred.

---

## 2. Five-Pillar Scoring

### Pillar 1: Extreme and Obvious Value — 4/10

**The problem is real.** Polypharmacy ADEs are a leading cause of hospitalization in older adults. The 21st Century Cures Act mandates FHIR-based CDS interoperability. No tooling verifies multi-guideline safety across all patient trajectories. The formal-methods moat is genuine — LLMs cannot provide exhaustive safety certificates.

**But nobody is asking for this.** Zero demand signal from any hospital, EHR vendor, guideline organization, or regulator (R7: 80%+ probability this persists). CQL treatment-logic adoption is near-zero in production (~30–50 artifacts exist). GPT-4 + DrugBank handles most practical clinical needs. Hospital CDS committees want "Is this on the Beers list?" — not PTA reachability proofs. Alert fatigue means clinicians override 90–96% of DDI alerts regardless of provenance.

**The coverage gap undercuts clinical value.** Theorem 1 covers 45–55% of clinically significant DDIs in the target population. The most dangerous interactions (QT prolongation, serotonin syndrome) are outside the contract framework. A clinical user discovers the formal verification tool can't help with the scariest cases.

**Why not lower:** The problem genuinely exists, the LLM-proof moat is real, and infrastructure timing may vindicate the tool (CMS mandate for FHIR-based CDS is law, not speculation). At a formal methods venue (HSCC), the value question is "Is this formalism novel and useful?" — yes.

### Pillar 2: Genuine Software Difficulty — 7/10

**Three-domain intersection is genuinely hard.** The system must be simultaneously correct across formal methods (timed automata, abstract interpretation, model checking), pharmacokinetics (compartmental ODEs, Metzler matrices, CYP inhibition), and clinical informatics (CQL semantics, FHIR data models, clinical safety predicates). No single expert can validate all three domains.

**MVP scope is substantial.** ~53K total / ~35K novel LoC. Novel formalism + novel abstract domain + novel contract decomposition + PK-specific SAT encoding. Comparable to small verification tools (simplified CBMC).

**Deferred components reduce difficulty.** CQL compilation (~15K novel LoC), validated interval ODE integration (~5K), zonotopic reachability (~14K) — three of the hardest components are out of scope. What remains is substantial but less technically adventurous than the full vision.

### Pillar 3: Best-Paper Potential — 3.5/10

**The HSCC reframing is the Synthesizer's best insight.** At HSCC, a formalism paper (PTA definition + composition theorem + decidability analysis + case study) is a natural fit. The audience values new hybrid automata classes with decidability properties. The E1 gamble becomes secondary — the formalism stands on its own.

**But probability estimates must be honest:**

| Venue | P(accept) | P(best paper) |
|---|---|---|
| HSCC (formalism) | 25–30% | 1.5–2.5% |
| AIME (tool) | 20–30% | 0.8–1.5% |
| Either (dual submission) | 40–50% | 2–4% |
| P(≥1 acceptance) | 40–50% | — |

The Synthesizer's 18–24% P(best paper from dual submission) is inflated by ~5×. Best-paper rates at top venues are 1–2% of all papers. Claiming 8–12% requires believing GuardPharma is 4–6× more likely than a random accepted paper to win — implausible for a paper with zero implementation, zero evaluation, and unproven theorems.

**The E1 gamble:** 30–40% probability X < 15%, which collapses the AIME narrative. The Skeptic's analytical distinction is devastating: PK interactions *being temporal in nature* ≠ guideline conflicts *requiring temporal reasoning to detect*. Most CYP-mediated DDIs are flagged perfectly well by atemporal databases. The temporal system may add *characterization* (when toxicity occurs) but not *detection* (that toxicity occurs).

### Pillar 4: Laptop-CPU Feasibility & No-Humans — 6/10

**Contract path is clearly laptop-feasible.** Closed-form matrix exponentials (microseconds), linear contract checking (microseconds), Z3/CaDiCaL SAT solving (seconds to minutes). Memory ~100MB. All CPU-native.

**Monolithic fallback path (~50% of interactions) has unknown performance.** CEGAR convergence for 5+ interacting drugs with PD effects is uncharacterized. dReal has doubly-exponential worst-case for nonlinear ODEs. The project cannot guarantee laptop feasibility for the non-contract path.

**E9 (clinical pharmacist review) technically requires humans.** $600, 3 pharmacists — modest but present. The "no humans" constraint is violated by the project's own mandatory evaluation plan.

### Pillar 5: Feasibility — 5/10

**Sound architecture, zero execution.** theory_bytes = 0, impl_loc = 0. The project has produced nothing except planning documents across multiple pipeline stages. The Theorem 1 proof has a known gap (M(c)). No pilot exists. PopPK parameter availability is assumed, not verified. 35K novel LoC in 20 weeks is tight.

**The planning fractal.** The Skeptic's sharpest observation: the project generates meta-work about work rather than actual work. ~500 KB of assessment documents, 0 bytes of proofs, 0 lines of code. The ratio of planning to execution is ∞/0. The project declared `theory_complete` with 0 theory bytes — a calibration error that suggests the workflow prioritizes evaluation over production.

**But the intellectual content is real.** The M(c) gap discovery is evidence of genuine mathematical engagement. A pure "planning fractal" project doesn't find its own proof gaps. The self-correction (cutting Theorem 2, abandoning Approach 3, identifying M(c) issue) suggests real mathematical thought happening, just not externalized as formal artifacts.

---

## 3. The theory_bytes = 0 Problem

### What Happened

The theory stage completed with `theory_complete` status but produced 0 bytes of theory artifacts. The approach.json (39KB) was created by a recovery attempt after 503 API errors crashed the primary workflow. Planned deliverables not produced:

- No formal proofs (paper.tex was planned but never created)
- No proof verification or red-team review
- No LaTeX paper draft
- No red-team critique of written proofs
- Verifier amendment VA4 (complete Theorem 1 proof before implementation) was not satisfied

### What It Means

**Severity: 6.5/10 (high yellow, bordering on red).**

The Skeptic's "planning fractal" critique has real teeth: 500 KB of meta-evaluation vs. 0 KB of proof artifacts is a pathological ratio. But the Auditor is correct that this is a process failure, not an intellectual failure — the mathematical content exists in approach.json and is coherent, with proof sketches that are directionally correct.

The decisive interpretation: the project has mathematical substance but a dysfunctional workflow that prioritizes meta-evaluation over artifact production. The proofs *might* work but demonstrably *haven't been written*. If Theorem 1's proof can be written to publication quality in 2 weeks, the severity drops to 3/10. If it can't, the severity rises to 9/10.

---

## 4. Fatal Flaw Analysis

### F1: Zero Artifacts After Theory Stage — Severity: 7/10 (SERIOUS)

The project has passed through ideation, depth checking, adversarial review, and theory — producing exactly zero substantive artifacts. Every claim about Theorem 1, Proposition 2, the PTA formalism, and clinical value is unsubstantiated by any evidence. The theory stage was the designated opportunity to prove mathematical claims. It produced nothing.

**Not quite fatal** because the intellectual content exists (in planning documents) and the failure was procedural (503 errors). **Becomes fatal** if the next 4 weeks also produce zero artifacts.

### F2: E1 Temporal Ablation Is a Coin Flip — Severity: 8/10 (POTENTIALLY FATAL)

30–40% probability X < 15%. If X < 15%, the PTA + PK ODE + contract composition machinery delivers less than 15% marginal detection value over a simple atemporal checker. The entire novel technical core becomes over-engineering. The depth check's analytical distinction is devastating: most CYP-mediated DDIs are flagged by atemporal databases. The temporal system may add characterization but not detection.

**Mitigated** by three fallback narratives (explanation quality, compositionality speedup, existence proof) and the HSCC reframing where E1 is a case study, not the centerpiece.

### F3: M(c) Proof Gap in Theorem 1 — Severity: 6/10 (SERIOUS)

The PK matrix M = M(c) is state-dependent, making the system nonlinear. The proof sketch assumes constant M. Fixing requires cooperative systems theory (Smith 1995). Likely fixable (competitive inhibition produces cooperative dynamics), but the fix hasn't been written and could reveal parameter-space boundary issues.

### F4: Corpus Starvation — Severity: 6/10 (SERIOUS)

50–60% probability only ~30–50 CQL treatment guidelines exist. With ~30 guidelines, only ~435 pairs — limiting statistical power and reducing chances of finding enough temporal-only conflicts for E1. The paper must frame as proof-of-concept.

### F5: CYP Coverage Overstatement — Severity: 5/10 (SERIOUS)

The project claims "~70% of clinically significant PK interactions" covered by contracts. The honest number for the target polypharmacy population is 45–55% of clinically significant DDIs. The most dangerous interactions (QT prolongation, serotonin syndrome) are outside scope. The framing must be corrected.

### F6: PopPK Parameter Availability — Severity: 5/10 (SERIOUS)

Population PK parameter bounds [φ_lo, φ_hi] required for every drug. Published PopPK models exist for common drugs but are sparse for many polypharmacy drugs. Ki values for CYP inhibition vary 2–10× across studies. No parameter database is cited or developed.

---

## 5. Salvage Analysis (from Scavenging Synthesizer)

### Hidden Diamonds

1. **The PTA formalism itself** (5/10 depth). Novel decidable fragment of nonlinear hybrid automata. The formalism enables everything else. Undervalued by every previous assessment.

2. **HSCC reframing.** The optimal publication strategy targets HSCC (formalism paper) rather than AIME (tool paper). At HSCC, the formalism is the contribution, E1 is secondary, and implementation requirements are lighter. Composite under HSCC framing: +0.5 over AIME framing.

3. **Dual-submission strategy.** HSCC (formalism) + AIME (tool) creates partially independent submission opportunities. P(≥1 acceptance) ≈ 40–50%.

4. **CQL formal semantics** (7/10 depth if pulled into scope). Even a core CQL fragment's denotational semantics would be the first formal semantics of a healthcare-mandated language. The biggest unrealized opportunity.

5. **Schedule separability observation.** Recoverable from abandoned Approach 3. Provable minimum temporal separation τ_min for dose spacing under competitive inhibition. Depth 3/10 but maximally clinically actionable.

6. **PSPACE-hardness.** Trivially provable (reduction from timed automata). Justifies the two-tier architecture and makes the contract decomposition more impressive.

### Optimal Path Forward

1. **Primary target: HSCC formalism paper.** PTA definition + Theorem 1 + Proposition 2 + case study on 3–5 guideline pairs.
2. **Secondary target: AIME tool paper.** Full system + evaluation. Conditional on E1 ≥ 20% and implementation completion.
3. **Pull CQL fragment semantics partially into scope** if feasible in 3 weeks (50–60% probability).
4. **Add PSPACE-hardness + schedule separability + tiered soundness** as supporting results (~2 days total effort).

---

## 6. Overall Verdict

### Resolved Pillar Scores

| Pillar | Score | Key Factor |
|---|---|---|
| 1. Extreme Value | **4/10** | Real problem, genuine LLM-proof moat, but zero demand signal (80%+), CQL ecosystem barely exists, coverage only 45–55% of clinical DDIs |
| 2. Genuine Difficulty | **7/10** | Three-domain intersection is genuinely hard; ~35K novel LoC; novel formalism + contracts + abstract domain |
| 3. Best-Paper Potential | **3.5/10** | 2–4% P(best paper) with dual submission; E1 is high-variance; HSCC reframing helps but estimates must stay honest |
| 4. Laptop-CPU + No-Humans | **6/10** | CPU feasible for contract path (~50%); monolithic fallback performance unknown; E9 requires 3 pharmacists |
| 5. Feasibility | **5/10** | Sound architecture, zero execution; theory_bytes = 0; planning fractal risk; M(c) gap unresolved |

### Composite: 4.9/10

### Score Comparison Across All Assessments

| Source | V | D | BP | L | F | Composite |
|---|---|---|---|---|---|---|
| Project self-score | 7 | 7 | 6 | 6 | 6 | 6.5 |
| Previous depth check | 5 | 7 | 4 | 6 | — | 5.5 |
| Auditor | 5 | 7 | 4 | 6 | 5 | 5.4 |
| **Skeptic** | **3** | **6** | **2** | **5** | **3** | **3.8** |
| **Synthesizer** | **6.5** | **7.5** | **6.5** | **7** | **6.5** | **6.8** |
| **Resolved (this eval)** | **4** | **7** | **3.5** | **6** | **5** | **4.9** |

The project's self-scores are inflated by ~1.6 points. The previous depth check (5.5) was well-calibrated. The Auditor (5.4) was the most accurate individual evaluator.

### Probability Estimates

| Outcome | Probability |
|---|---|
| Theorem 1 proof is valid (M(c) fix works) | 65–75% |
| Pilot PTA encoding succeeds | 70–80% |
| E1 produces X ≥ 20% (AIME narrative survives) | 55–65% |
| E1 produces X ≥ 15% (minimal temporal value) | 60–70% |
| ≥1 publication at any venue | 55–65% |
| Best paper at any venue | 2–4% |
| Stakeholder adoption within 3 years | 5–10% |
| Total failure (nothing publishable) | 10–20% |

### VERDICT: CONDITIONAL CONTINUE

**Not ABANDON** because:
- The problem is real and important
- One genuine mathematical insight (Metzler single-pass resolution) is load-bearing
- The PTA formalism is novel and occupies a genuinely empty niche
- The salvage floor is high (formalism paper even if E1 and implementation fail)
- P(≥1 publication) ≈ 60% justifies continued effort
- Exceptional intellectual honesty with self-correction (cutting Theorem 2, abandoning Approach 3, discovering M(c) gap)

**Not unconditional CONTINUE** because:
- theory_bytes = 0 after a theory stage — the foundation is unverified
- E1 is a 30–40% coin flip on a null result
- Zero implementation exists
- The most interesting math (CQL bisimulation) is deferred
- Coverage is 45–55% of clinical DDIs, not 70%
- Planning fractal risk: demonstrated tendency to meta-evaluate rather than execute

### Binding Conditions (All Non-Negotiable)

**C1: Stop generating meta-evaluation documents.** The next byte produced must be a proof, a line of code, or a guideline encoding. Any further assessment, risk analysis, or synthesis documents before G1 is met constitutes gate failure.

**G1 (Week 2): Full Theorem 1 proof.** ≥5 pages including the M(c) monotonicity argument via cooperative systems theory (Smith 1995). Must explicitly state domain restrictions (competitive inhibition only), parameter space coverage, and what is NOT covered. If the proof cannot be written → downgrade to workshop paper exploring the formalism.

**G2 (Week 3): Pilot PTA encoding + model checker.** One guideline pair (e.g., diabetes + hypertension) fully encoded as PTA. Model checker terminates and produces a result (safe or unsafe) within 30 minutes on laptop CPU. If PTA encoding fails or model checker diverges → ABANDON implementation track; submit theory-only paper.

**G3 (Week 4): E1 calibration.** Using pilot data, estimate X (temporal-only detection rate):
- If X < 10%: abandon AIME track entirely
- If X ∈ [10%, 20%): proceed with HSCC primary, AIME secondary
- If X ≥ 20%: dual-submission at full strength

**G4 (Week 6): Submission-ready proofs.** All claimed results proven to referee standard. Red-team review by domain expert. If proofs contain irreparable gaps → downgrade venue tier to workshop.

### Reframing Directive

**Primary target: HSCC (formalism paper).** This aligns the project's actual strength (novel hybrid automaton class with composition properties) with the venue's reward function. The E1 gamble becomes a case study, not the centerpiece.

**Secondary target: AIME (tool paper).** Conditional on E1 ≥ 20% and implementation completion.

### Amendments Required

| # | Amendment | Rationale |
|---|---|---|
| A1 | Replace "~70% of clinically significant interactions" with "65–70% of PK DDIs; approximately 45–55% of clinically significant DDIs in elderly polypharmacy" | Skeptic's clinical framing is correct |
| A2 | Add M(c) cooperative systems argument to Theorem 1 proof | Critical gap identified by Auditor |
| A3 | Add PSPACE-hardness proposition | Trivial to prove; justifies architecture |
| A4 | Add tiered soundness theorem | Converts architecture to mathematical contribution |
| A5 | Add schedule separability observation | Recoverable from Approach 3; maximally clinically actionable |
| A6 | Downgrade `theory_complete` status | theory_bytes = 0 means theory is NOT complete |
| A7 | Replace self-scored V7/D7/BP6/F6 = 6.5/10 with V4/D7/BP3.5/L6/F5 = 4.9/10 | Calibration correction |

---

## 7. Who Was Right?

| Disagreement | Winner | Resolution |
|---|---|---|
| CYP Coverage (70% vs 35–50%) | **Skeptic** on clinical framing | 45–55% of clinically significant DDIs |
| PTA Formalism value (7/10 vs dismissed) | **Auditor** (5/10) | Novel fragment, not field-creating |
| Math load-bearing? (80% test) | **Split** | Load-bearing for verification claim; not for clinical impact |
| theory_bytes = 0 severity | **Skeptic > Auditor > Synthesizer** | 6.5/10 concern; planning fractal is real |
| Best-paper potential (2 vs 4 vs 6.5) | **Auditor** closest | 3.5/10; dual-submission gets ~4% |
| Verdict (ABANDON vs CONTINUE) | **Auditor** with Synthesizer's reframing + Skeptic's discipline | CONDITIONAL CONTINUE with hard gates |

**The Auditor's evaluation was the most calibrated overall.** The Skeptic provided essential discipline and the strongest individual arguments (CYP coverage, planning fractal, theory_bytes severity) but overshot on the verdict. The Synthesizer's HSCC reframing was the single most valuable strategic insight but paired with inflated scores that undermined credibility.

---

## 8. The Mathematician's Final Word

GuardPharma has one genuine mathematical insight, one novel formalism, and competent supporting results — all unproven. The Metzler monotone chain for single-pass contract resolution is real mathematics that serves a real purpose. The PTA formalism fills a genuinely empty niche. But the project has produced zero bytes of proof, zero lines of code, and zero empirical data across multiple pipeline stages. The most interesting theorem (CQL compilation correctness) is conveniently deferred.

The math is adequate for a good applied verification paper. It is not deep enough for a best paper at any venue except under optimistic assumptions about E1 and reviewer composition. The math is load-bearing for the verification claim (without Theorem 1, you can't do compositional polypharmacy verification) but not for the clinical impact claim (GPT-4 + DrugBank handles most practical clinical needs without any math at all).

**The project's fate is entirely in its hands.** If it can produce a proof of Theorem 1 in 2 weeks and a working pilot in 3, it has a credible path to publication at HSCC/AIME with P(≥1 acceptance) ≈ 50%. If it generates another round of meta-evaluation instead, the planning fractal will have consumed the project.

**The project needs to stop planning and start proving.**

---

*Evaluation produced by 3-expert team (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) with adversarial cross-critique and forced disagreement resolution. Auditor composite: 5.4/10. Skeptic composite: 3.8/10. Synthesizer composite: 6.8/10. Resolved composite: 4.9/10. The Auditor was most calibrated; the Synthesizer found the best strategic insight (HSCC reframing); the Skeptic prevented under-estimation of real risks (CYP coverage, planning fractal, theory_bytes severity).*

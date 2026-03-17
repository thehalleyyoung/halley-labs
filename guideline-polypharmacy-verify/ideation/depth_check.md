# Depth Check — GuardPharma: Contract-Based Temporal Verification of Polypharmacy Safety

**Slug:** `guideline-polypharmacy-verify`  
**Verification method:** 3-expert team (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) with adversarial cross-critique and forced disagreement resolution  
**Date:** 2026-03-08

---

## Pillar 1: EXTREME AND OBVIOUS VALUE — Score: 5/10

### What Works
- The problem is real: clinical guidelines are authored in isolation, patients follow multiple simultaneously, and no tooling verifies joint safety.
- The "verify before deploy" framing is honest and forward-looking. Under the 21st Century Cures Act and ONC HTI-1, FHIR-based CDS interoperability is mandated law, not speculation.
- The LLM-proof moat is genuine: formal guarantees (exhaustive verification, sound counterexamples, certificates) are categorically unreplaceable by LLMs. As LLMs become the default clinical reasoning tool, the demand for formal guarantees increases.
- The Joint Commission already cites CDS configuration errors as a contributing factor in medication safety events — this is a current pain point, not hypothetical.

### What Doesn't Work
- **Zero demand signal.** No hospital, EHR vendor, guideline organization, or regulator has requested formal verification of clinical guidelines. The value proposition table describes hypothetical stakeholders.
- **CQL treatment-logic adoption is near-zero in production.** The CQL ecosystem is growing (driven by regulatory mandates) but CQL for *prospective treatment decision logic* — the specific use case GuardPharma verifies — has minimal production deployment. CDS Connect has ~30 artifacts, most non-treatment. Epic and Oracle Health deploy CDS rules primarily in proprietary formats, not CQL.
- **The LLM challenge is partially valid.** GPT-4 + DrugBank flags most drug-drug interactions with mechanism explanations and dose adjustment suggestions. The *practical* value delta of formal verification over "very good heuristic checking" is real but narrower than claimed — especially since GuardPharma itself only covers ~70% of PK interactions (CYP inhibition), which is only a subset of all DDIs.
- **Hospital CDS committees would not use this tool in its current form.** They want "Is this combination on the Beers list?" — not "Here is a δ-decidable reachability proof over a PTA product automaton."

### What's Missing to Reach 7
1. **A demand signal.** A letter of interest from an EHR vendor, guideline organization, or accreditation body expressing interest in formal CDS safety verification.
2. **Evidence of CQL treatment-logic deployment at scale.** If even 5–10 health systems are running CQL-based treatment decisions in production, the user base becomes tangible.
3. **A clinical-facing output layer.** The counterexample narratives need to be translated to clinical language that CDS committees can act on — not formal counterexample traces.

---

## Pillar 2: GENUINE DIFFICULTY AS SOFTWARE ARTIFACT — Score: 7/10

### What Works
- **The three-domain intersection is genuinely hard.** The system must be simultaneously correct across formal methods, pharmacokinetics, and clinical informatics. Getting any one right is a strong paper; getting all three right in a single system is a research program.
- **The novel verification core (~73–80K LoC) is impressive.** The PTA formalism, zonotopic reachability engine, MTL model checker with clinical CEGAR, and contract-based CYP-enzyme composition are genuine engineering achievements.
- **The CQL-to-PTA semantic compilation is harder than it looks.** CQL has interval arithmetic, temporal operators, FHIR retrieve semantics. This implicitly creates the first formal semantics of CQL.

### LoC Audit

| Subsystem | Claimed | Realistic Total | Genuine Novel | Notes |
|---|---|---|---|---|
| 1. CQL-to-PTA Compiler | 18K | 15K | 10K | Semantic gap ELM→PTA is substantial |
| 2. FHIR PlanDef Compiler | 14K | 10K | 5K | HAPI parses; compiler does workflow→automaton |
| 3. Corpus Pipeline | 10K | 8K | 3K | Data engineering, not algorithmic novelty |
| 4. Clinical State Model | 12K | 8K | 5K | Typed data model with invariants |
| 5. PK Model Library | 20K | 12K | 6K | Wraps CAPD/VNODE-LP; PTA interface is novel |
| 6. PTA Composition Engine | 18K | 16K | 14K | Core novelty: product construction + contracts |
| 7. Zonotopic Reachability | 16K | 14K | 10K | Metzler-specific algorithm is novel |
| 8. MTL Model Checker + CEGAR | 22K | 20K | 14K | PK region construction is novel |
| 9. Counterexample Generator | 10K | 7K | 4K | Trace extraction + narrative templating |
| 10. Clinical Significance Filter | 10K | 8K | 4K | Multi-source integration |
| 11. Terminology Layer | 12K | 6K | 3K | Mostly wraps HAPI FHIR |
| 12. Evaluation Engine | 13K | 10K | 5K | Standard benchmark harness |
| **TOTAL** | **175K** | **~134K** | **~83K** | |

- **175K claim has ~25% inflation.** Honest total is ~134K. Novel algorithmic code is ~83K.
- **Novel-to-total ratio: ~62%.** The remaining ~51K is integration/glue code around existing libraries (HAPI FHIR, Z3, CUDD, CQL reference parser, CAPD).
- At ~83K novel LoC, this is still a substantial engineering challenge — comparable to CBMC, SLAM/SDV, or PRISM.

### What's Missing to Reach 8+
1. **Clarify ODE integrator strategy.** The document should explicitly state whether it builds or wraps a validated interval ODE integrator. If wrapping CAPD/VNODE-LP, subsystem 5 shrinks and the claim "validated interval ODE integration with directed rounding" should not be listed as a project contribution.
2. **Honest LoC reporting.** Replace "175K LoC" with "~135K LoC total, ~83K novel algorithmic code" throughout.

---

## Pillar 3: BEST-PAPER POTENTIAL — Score: 4/10

### Venue-by-Venue Assessment

| Venue | P(best paper) | Rationale |
|---|---|---|
| AMIA Annual | 5–8% | Best fit; clinical informatics audience; formal methods as applied tool = novelty. BUT: zero clinical validation is near-fatal at a clinical venue. |
| JAMIA (journal) | 3–5% | Strong fit; longer format. BUT: JAMIA best papers typically have clinical validation. |
| AIME | 8–12% | Best odds; AI-in-Medicine bridges both communities. Smaller venue, more receptive to cross-disciplinary. |
| CAV | 2–4% | Theorems individually thin for a top FM venue. δ-decidability is one sentence of insight; PSPACE is routine reduction; A/G composition is an instantiation. |
| TACAS | 4–7% | Tool track is natural. BUT no established benchmark to beat. |

### The E1 Temporal Ablation Gamble

This is the make-or-break experiment. The paper claims "X% of guideline conflicts require temporal pharmacokinetic reasoning to detect."

**Critical analytical distinction (from the Auditor):** PK interactions *being temporal in nature* ≠ guideline conflicts *requiring temporal reasoning to detect*. Fluconazole + warfarin is a temporal PK interaction — but an atemporal checker flags it fine as a CYP2C9 interaction. The temporal system adds *characterization* (INR exceeds 4.0 at day 5), not *detection*. Temporal reasoning adds detection value only for *schedule-dependent conflicts* — where the specific timing prescribed by guidelines determines whether the combination is dangerous.

**Realistic range for X (temporal-only conflicts as fraction of all detected conflicts):**
- Optimistic: 20–30%
- Realistic: 10–20%  
- Pessimistic: 5–10%

**If X < 15%, the paper's core narrative collapses.** The PTA+PK+zonotope+CEGAR machinery becomes an over-engineered solution that produces approximately the same results as a simpler atemporal checker with richer output.

### Mathematical Depth
- **Theorem 1 (δ-decidability):** Application of dReal's framework to PK dynamics with pharmacologically meaningful δ. One sentence of novel insight. Should be downgraded to "Proposition."
- **Theorem 2 (PSPACE-completeness):** Sound but routine. PSPACE-hardness follows from timed automata subsumption. Membership via PK region graph — but the bisimulation proof (the hard part) is unspecified.
- **Theorem 3 (Contract-based composition):** The genuine mathematical contribution. CYP-enzyme interface contracts with Metzler monotonicity enabling worst-case guarantee computation. Novel and practically motivated. Fixed-point resolution via worst-case guarantees is stated but needs a formal monotonicity proof restricted to competitive inhibition.

**One solid theorem is not enough for FM venues; at clinical venues, the math is secondary to clinical findings (which don't exist yet).**

### What's Missing to Reach 7
1. **Pilot data showing X ≥ 25%.** Without this, E1 is a high-variance gamble.
2. **Minimal clinical validation.** 3 pharmacists rating 20 conflicts on a Likert scale (~$600, ~2 weeks). This converts the cross-community bridge from liability to strength.
3. **Completed proof of Theorem 3.** Formal monotonicity proof for competitive inhibition. State restrictions explicitly for non-competitive/mixed mechanisms.
4. **A fallback narrative.** If E1 disappoints, the paper should pivot to *explanation quality* (pharmacokinetic trajectory counterexamples are unprecedented regardless of detection rate) and *compositionality speedup* (E6).

---

## Pillar 4: LAPTOP CPU + NO HUMANS — Score: 6/10

### What Works
- **Contract-decomposed verification is well-founded for CPU.** Linear PK ODEs have closed-form matrix-exponential solutions (microseconds). Contract checking is linear algebra (microseconds). Zonotope operations are polynomial in dimension and order.
- **Z3, CaDiCaL, CUDD are all CPU-native.** No GPU dependency anywhere in the stack.
- **Memory budget is reasonable.** ~100MB peak for contract-decomposed single-guideline verification. Well within 16GB.
- **Reproducibility is a genuine differentiator.** "Clone the repo, run `make evaluate`, reproduce every number" — few papers at any venue can truthfully say this.

### What Doesn't Work
- **CEGAR fallback timing is unknown.** For the ~30% of interactions outside the contract framework (pharmacodynamic, CYP induction), the system falls back to monolithic product-PTA verification with CEGAR. No estimate exists for convergence time on realistic instances. UPPAAL takes hours on moderately complex timed automata. dReal has doubly-exponential worst-case for nonlinear ODEs.
- **5+ concurrent drugs (monolithic path) may be intractable.** For 5 drugs with 3-compartment models = 15 ODE variables. Validated interval ODE integration at this dimension is extremely expensive. dReal reports hours for 10+ variable nonlinear systems.
- **FAERS at 20M+ reports is feasible but must be pre-processed.** Streaming/batched processing or pre-computed disproportionality lookup table (~500MB) is the correct engineering choice. Naive in-memory processing at 16GB is at the limit.
- **Zero clinical validation is a credibility gap at clinical venues.** This is easily fixable (~$600, ~2 weeks, no IRB needed) but must be fixed for JAMIA/AMIA acceptance.

### What's Missing to Reach 9
1. **CEGAR convergence estimates on representative instances.** Even hand-coded toy examples would establish whether the model checker terminates in seconds, minutes, or hours.
2. **Explicit FAERS pre-processing strategy.** State that FAERS disproportionality will be pre-computed and distributed as a lookup table.
3. **Add minimal clinical validation.** The "no humans" constraint is self-imposed and counterproductive at clinical venues. Adding 3-pharmacist review preserves full automated reproducibility while adding clinical credibility.

---

## Pillar 5: FATAL FLAWS

### Potentially Fatal (must be resolved)

| # | Flaw | Severity | Fixable? | Fix |
|---|------|----------|----------|-----|
| F1 | **No preliminary evidence PTA encoding works** | 8/10 | Yes, 4–6 weeks | Manually encode 3 guideline pairs as PTA. Run simplified model checker. If encoding fails, redirect to theory paper. |
| F2 | **E1 temporal ablation may produce null result** | 8/10 | Partially | Literature calibration (estimate X from published PK DDI data). Construct 5–10 synthetic guideline pairs with known temporal interactions as proof-of-concept. Prepare fallback narrative (explanation quality). |

### Serious (must be addressed)

| # | Flaw | Severity | Fixable? | Fix |
|---|------|----------|----------|-----|
| S1 | **No demand signal from any stakeholder** | 6/10 | Difficult | Seek letter of interest from EHR vendor or guideline org. Frame as forward-looking infrastructure in paper. |
| S2 | **Corpus starvation (~30–50 treatment guidelines)** | 6/10 | Partially | Supplement with manually encoded clinical rules. Report honestly. Accept that experiments may lack statistical power. |
| S3 | **Zero clinical validation** | 5/10 | Easy, ~$600 | 3 pharmacists review 20 conflicts. No IRB needed. 2-week effort. |

### Manageable (should be addressed in document)

| # | Flaw | Severity | Fix |
|---|------|----------|-----|
| M1 | Contract framework covers only CYP inhibition (~70%) | 4/10 | Already acknowledged. State boundary clearly. Report coverage statistics. |
| M2 | Theorem 3 fixed-point needs formal monotonicity proof | 4/10 | Add 3–4 line monotonicity argument. Restrict to competitive inhibition. |
| M3 | PK region graph bisimulation unspecified | 4/10 | 1-page proof sketch. |
| M4 | Theorem 1 novelty is thin for "Theorem" label | 3/10 | Downgrade to "Proposition" or "Application." |
| M5 | LoC inflation (~25%) | 3/10 | Report honest ~135K total, ~83K novel. |
| M6 | CEGAR may not converge on complex cases | 4/10 | Report convergence statistics; use BMC fallback. |

---

## AMENDMENTS REQUIRED

Since Pillar 1 (Value: 5/10) and Pillar 3 (Best-Paper: 4/10) are both below 7, the following amendments to the problem statement are required:

### Amendment A1: Address the LLM Challenge Explicitly
Add a section to "Why Existing Approaches Don't Solve It" addressing LLM-based drug interaction checking. Explain the categorical difference (exhaustiveness, soundness, certificates) while honestly acknowledging that LLM+DrugBank handles most practical clinical needs. Position GuardPharma as providing the *regulatory/certification* layer that LLMs cannot.

### Amendment A2: Honest CQL Ecosystem Assessment  
Replace the "300+ guideline artifacts" target in Risk #3 with the honest count: "~30–50 CQL treatment-decision guidelines supplemented by ~30 manually encoded clinical rules, targeting ~80 treatment-logic artifacts for verification." State clearly that CQL treatment-logic adoption is early-stage but accelerating under federal mandate.

### Amendment A3: Add Clinical Validation to Evaluation Plan
Add E9: "Clinical Pharmacist Review" — 3 clinical pharmacists independently rate 20 discovered conflicts on a 5-point clinical relevance Likert scale. Report inter-rater reliability (Fleiss' κ). This eliminates the credibility gap at clinical venues while maintaining full computational reproducibility.

### Amendment A4: Reduce Scope to Phase-1 MVP
The primary paper scope should be explicitly stated as ~109K LoC (subsystems 4–9, 10, 12), with subsystems 1–3 and 11 explicitly deferred to a follow-on tool paper. Do not claim 175K as the paper scope. The 175K full-vision scope can be described as future work.

### Amendment A5: E1 Fallback Narrative
Add a contingency framing: if temporal-only detection rate (X%) is modest (<20%), the paper pivots to: (a) explanation quality — PK trajectory counterexamples are unprecedented regardless of detection rate, (b) compositionality speedup — E6 demonstrates practical value of Theorem 3 independently of E1, (c) existence proof — first formal safety certificate for multi-guideline polypharmacy verification.

### Amendment A6: Pilot Commitment
Add to the evaluation plan: "Milestone 0 (pre-submission): manually encode 3 guideline pairs as PTA and verify at least one non-trivial temporal conflict. This pilot validates feasibility before full implementation commitment."

### Amendment A7: LoC Honesty
Replace "~175K LoC" throughout with "~135K total LoC (~83K novel algorithmic code)." Remove the claim "no subsystem is padding."

### Amendment A8: FAERS Pre-Processing
State that FAERS disproportionality will be pre-computed offline and distributed as a lookup table, not computed at evaluation time.

### Amendment A9: Theorem 1 Downgrade
Rename "Theorem 1" to "Proposition 1" or "Application 1" to honestly reflect that this is an application of dReal's δ-decidability framework, not a new decidability technique.

### Amendment A10: Theorem 3 Monotonicity Proof Sketch
Add a 3–4 line proof sketch establishing that enzyme-load guarantee functions are monotone in assumed enzyme capacity for competitive CYP inhibition. State explicitly that non-competitive and mixed inhibition mechanisms require direct product verification.

---

## OVERALL ASSESSMENT

| Pillar | Score | Key Factor |
|--------|-------|-----------|
| 1. Extreme and Obvious Value | **5/10** | Genuine intellectual gap; zero demand signal; CQL treatment-logic adoption near-zero; LLM challenge partially valid |
| 2. Genuine Difficulty | **7/10** | Three-domain intersection is genuinely hard; ~83K novel LoC; ~25% LoC inflation |
| 3. Best-Paper Potential | **4/10** | 5–12% P(best paper) at best venue (AIME); E1 is high-variance gamble; math individually thin; cross-community bridge is liability without clinical validation |
| 4. Laptop CPU + No Humans | **6/10** | CPU feasible for contract path; CEGAR fallback unknown; clinical validation gap easily fixable |
| 5. Fatal Flaws | **2 potentially fatal, 3 serious, 6 manageable** | Pilot and E1 calibration are must-do before commitment |

**Composite: 5.5/10**

**Verdict: CONDITIONAL CONTINUE** — The project has genuine intellectual merit and fills a real gap. The mathematical contribution (Theorem 3, contract-based composition) is novel and practically motivated. The PTA formalism is unprecedented. The salvage floor is high (5 potential publications, no zero-paper failure mode).

However, two potentially fatal flaws must be resolved before full commitment:
1. The pilot study (3 guideline pairs encoded as PTA, model checker terminates, one conflict found) must succeed.
2. E1 must be calibrated (literature estimate of X%) before betting the paper on it.

If the pilot succeeds, clinical validation is added, and E1 delivers X ≥ 20%, the project has a credible path to best paper at AIME/AMIA with P(best paper) ≈ 8–12%. If E1 disappoints, the salvage paths (explanation quality, compositionality speedup, existence proof) produce a strong-but-not-best paper. If the pilot fails, the PTA formalism is published as a theory paper at CAV/HSCC and the project pivots.

**The project's fate hinges on the pilot.**

---

*Assessment produced by 3-expert team verification with adversarial cross-critique. Auditor composite: 5.5/10. Skeptic composite: 3.75/10. Synthesizer composite: 7.5/10. The Auditor was most accurate overall; the Synthesizer found genuine hidden value; the Skeptic prevented under-estimation of real risks.*

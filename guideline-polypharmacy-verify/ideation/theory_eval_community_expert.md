# Community Expert Verification — GuardPharma (proposal_00)

**Evaluator:** Community Expert (Health & Life Sciences Computing, area-045)
**Stage:** Verification (post-theory)
**Method:** 3-expert team (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) with adversarial cross-critique, forced dispute resolution, and independent verifier signoff
**Date:** 2026-03-08
**Slug:** `guideline-polypharmacy-verify`

---

## Executive Summary

GuardPharma proposes a two-tier formal verification engine for polypharmacy safety: Tier 1 (pharmacokinetic abstract interpretation for fast screening) + Tier 2 (contract-based compositional model checking with counterexample generation). The crown jewel is Theorem 1: CYP-enzyme interface contracts with Metzler monotonicity reducing N-guideline verification from exponential to polynomial complexity.

**Current state:** theory_complete, theory_bytes=0, impl_loc=0. Extensive planning artifacts exist (approach.json with formal proof structures, 15+ ideation documents). No code, no paper text, no proofs written up, no pilot results.

**Verdict: CONDITIONAL CONTINUE** — composite 4.3/10. The project has a genuine intellectual core (Theorem 1, two-tier architecture, PTA formalism) but faces multiple independent failure modes. Five binding gates with hard kill dates determine whether this proceeds to implementation. Kill probability: 25–35%.

---

## Team Evaluation Process

Three independent experts evaluated in parallel, then engaged in adversarial cross-critique:

| Expert | Role | Composite | Verdict |
|--------|------|-----------|---------|
| Independent Auditor | Evidence-based scoring | 4.8/10 | CONDITIONAL CONTINUE |
| Fail-Fast Skeptic | Aggressive rejection | 3.6/10 | ABANDON (unless 4 conditions met) |
| Scavenging Synthesizer | Value salvage | 5.8/10 | CONDITIONAL CONTINUE |
| **Cross-critique synthesis** | **Dispute resolution** | **4.3/10** | **CONDITIONAL CONTINUE** |

---

## Pillar 1: EXTREME AND OBVIOUS VALUE — Score: 3/10

### Consensus (all three experts agree)

- The problem is real: clinical guidelines are authored in isolation, patients follow multiple simultaneously, nobody verifies joint safety.
- The "verify before deploy" framing is honest and intellectually coherent.
- The LLM-proof moat is genuine: formal guarantees (exhaustive verification, sound counterexamples, safety certificates) are categorically unreplaceable by LLMs.
- **Zero demand signal** (R7 at 80%+ probability per the project's own assessment). No hospital, EHR vendor, guideline organization, or regulator has requested formal verification of clinical guidelines.
- CQL treatment-logic adoption is near-zero in production. The ecosystem GuardPharma targets barely exists.

### Dispute: Is "forward-looking infrastructure" worth extra points?

The Synthesizer argued the 21st Century Cures Act mandate + ONC HTI-1 creates genuine future demand (score: 5). The Auditor and Skeptic argued this is "if you build it they will come" reasoning (score: 3).

**Resolution: Score 3.** Federal mandates drive FHIR interoperability, not formal verification. CQL treatment-logic adoption is years from production scale. The comparison with TMR/MitPlan/GLARE — systems nobody uses — is a red flag, not a selling point. The value proposition describes hypothetical stakeholders, not actual ones. The Synthesizer's strongest counter-argument (regulatory precedent for pre-deployment verification in other domains) is valid but insufficient to raise the score above 3. Nobody in the health informatics community is asking for this tool today.

### The Skeptic's devastating point

> "The formal guarantee is strongest where it matters least." Contract-based decomposition covers CYP competitive inhibition (~70% of PK DDIs) — exactly what DrugBank already detects via O(1) database lookup. The most dangerous interactions (QT prolongation, serotonin syndrome) fall to monolithic BMC with unknown convergence. The compositionality theorem provides polynomial scaling for interactions that don't need polynomial scaling.

**Resolution:** This argument is substantially correct and is the single most important finding of this evaluation. The system's value proposition claims to address polypharmacy safety, but its strongest formal guarantees cover the interaction class most easily handled by existing tools. The counterargument — that temporal characterization and safety certificates add value even for known interactions — is valid but modest. A hospital CDS committee would not pay for "we can prove CYP2C9 interactions are dangerous" when their pharmacy system already flags them. The incremental value of exhaustive verification over heuristic checking for CYP interactions is real but narrow.

---

## Pillar 2: GENUINE SOFTWARE DIFFICULTY — Score: 6/10

### Consensus

- The three-domain intersection (formal methods + pharmacokinetics + clinical informatics) is real.
- The individual techniques are all well-known (abstract interpretation, assume-guarantee, SAT/SMT, Metzler ODEs, interval arithmetic).
- The novel contribution is the *domain-specific instantiation*, not new algorithms.

### Dispute: Integration difficulty (7) vs. "just engineering" (5)

The Skeptic produced a table showing every component is a known technique. The Auditor and Synthesizer argued three-domain intersection creates emergent difficulty beyond component-level complexity.

**Resolution: Score 6.** The Skeptic is right that no component is individually novel. The Auditor/Synthesizer are right that the *combination* creates real difficulty — getting the PK model to correctly interact with the abstract domain, getting contracts to correctly extract enzyme loads from ODE parameters, getting BMC encodings to preserve clinical-threshold precision. This is closer to building SLAM/SDV (known techniques, novel combination, real engineering difficulty) than to a homework exercise. But the Skeptic's core point stands: a strong PhD student with formal methods and pharmacology background could build the MVP in a year. The difficulty is substantial (6) but not extreme.

### LoC Assessment

| Component | Claimed Novel | Auditor Estimate | Skeptic Discount |
|-----------|--------------|------------------|-----------------|
| PK Abstract Interpretation (Tier 1) | ~8K | 8K | 6K |
| PTA Construction & Contracts | ~10K | 10K | 8K |
| SAT-Based BMC (Tier 2) | ~8K | 8K | 6K |
| PK Model Library | ~6K | 6K | 4K |
| Clinical State Model | ~5K | 5K | 3K |
| Counterexample Generator | ~3K | 3K | 2K |
| Clinical Significance Filter | ~4K | 4K | 3K |
| Evaluation Engine | ~6K | 6K | 5K |
| Manual PTA Encodings | ~3K | 3K | 3K |
| **TOTAL** | **~53K/35K novel** | **~53K/35K** | **~40K/25K** |

Realistic novel LoC: **28–35K**. Still a substantial artifact.

---

## Pillar 3: BEST-PAPER POTENTIAL — Score: 4/10

### Consensus

- Theorem 1 (contract-based composition via CYP-enzyme interfaces with Metzler monotonicity) is the genuine mathematical contribution. All three experts agree it is novel and practically motivated. Depth: 6/10.
- Proposition 2 (PK-aware widening) is sound but modest. Depth: 5/10.
- Observation 3 (δ-decidability) is correctly labeled as a library invocation. Depth: 1.5/10.
- One solid theorem at depth 6/10 is not enough for top FM venues (CAV, TACAS). Clinical venues (JAMIA, AMIA) require clinical validation that doesn't exist.
- AIME is the optimal venue. P(best paper) depends heavily on E1 outcome.

### The E1 Gamble — Make or Break

**Critical analytical distinction (acknowledged by the project itself):** A PK interaction *being temporal in nature* ≠ a guideline conflict *requiring temporal reasoning to detect*. Most CYP-mediated DDIs are flagged by atemporal checkers. Temporal reasoning adds *characterization* (when toxicity occurs, PK trajectory) but not *detection* for most interactions.

**Realistic estimates for X% (temporal-only detection fraction):**

| Expert | Estimate | Reasoning |
|--------|----------|-----------|
| Synthesizer | 15–25% | Schedule-dependent conflicts exist (amiodarone + digoxin onset at 1–3 weeks) |
| Auditor | 10–20% | Matches depth check's realistic range |
| Skeptic | 5–15% | Most PK interactions are pharmacological facts independent of timing |
| **Consensus** | **10–20%** | With 30–40% probability below 15% (collapse threshold) |

**If X ≥ 20%:** Strong paper narrative. P(best paper at AIME) ≈ 8–12%.
**If 15% ≤ X < 20%:** Viable but weakened. Must lean on compositionality + explanation quality.
**If X < 15%:** Core narrative collapses. Fallback narratives are a step down.

### Fallback Narrative Assessment

| Fallback | Skeptic Assessment | Synthesizer Assessment | Resolution |
|----------|-------------------|----------------------|------------|
| Explanation quality (PK trajectories) | "Not a paper" | "Unprecedented output, genuinely novel" | Insufficient alone, strengthens a compositionality paper |
| Compositionality speedup (E4) | "Speedup over nothing" | "First compositional PK verification framework" | Valid standalone contribution but not best-paper material |
| Existence proof | "Workshop paper" | "First safety certificate for multi-guideline polypharmacy" | Correct: HSCC/CAV workshop-to-short-paper, not best paper |

### Best-Paper Probability (risk-adjusted)

| Venue | P(accept) | P(best paper) | After E1 risk adjustment |
|-------|-----------|---------------|-------------------------|
| AIME | 35–50% | 8–12% | **5–8%** |
| AMIA | 25–40% | 5–8% | 3–5% |
| TACAS (tool) | 30–45% | 4–7% | 3–5% |
| CAV | 15–25% | 2–4% | 1–3% |
| HSCC (fallback) | 40–55% | 3–5% | 3–5% |

**Resolved P(best paper at optimal venue): ~5–8% at AIME.** This is below the ideation-stage estimate of 8–12% because theory_bytes=0 means proofs are still unwritten and E1 risk has not been mitigated by any preliminary data.

---

## Pillar 4: LAPTOP-CPU FEASIBILITY & NO-HUMANS — Score: 6/10

### Consensus

- The compositional verification path (Theorem 1) is well-founded for CPU: matrix exponentials, linear algebra, SAT/SMT solving. All CPU-native.
- Z3, CaDiCaL, CUDD are mature CPU-native tools. No GPU dependency.
- Memory budget (~100MB peak for contract-decomposed verification) is well within 16GB laptop RAM.

### Disputes Resolved

**E9 pharmacist review ($600) — is this "humans"?** The Skeptic argued this violates the "no human annotation, no human studies" constraint. The Synthesizer argued pharmacists evaluate *output*, not annotate *input*. **Resolution: Score not docked.** E9 is human evaluation of system output, not human annotation of training data or human subjects research. This is analogous to usability testing, which does not violate the constraint.

**CEGAR convergence for PD interactions.** The monolithic BMC fallback for ~30% of interactions (PD, enzyme induction) has unknown convergence. The Skeptic cites UPPAAL taking hours on moderately complex timed automata. **Resolution: Score docked.** The 90-day bounded BMC always terminates, but termination ≠ tractability. If BMC for 3+ PD-interacting drugs takes hours, the system is impractical for the cases that matter most. This must be tested in the pilot (Binding Condition 4).

**NTI drug δ-calibration.** For digoxin (therapeutic range 0.8–2.0 ng/mL), clinically meaningful δ is sub-nanogram, straining interval arithmetic precision. **Resolution: Minor concern.** The project already proposes routing NTI drugs to exact rational BMC rather than interval ODE integration. Scoping exclusion acceptable.

---

## Pillar 5: OVERALL FEASIBILITY — Score: 4/10

### Consensus

- 35K novel LoC in 20 weeks by 2 engineers = 175 person-days at 200 LoC/day. This leaves ≤25 person-days margin (~12.5%).
- No pilot data exists. Zero evidence that PTA encoding works, BMC terminates, or non-trivial conflicts are found.
- Multiple independent failure modes: E1 disappointment (30–40%), corpus starvation (50–60%), PTA encoding infeasibility (15–20%), BMC timeout (25–35%), Tier 1 false-positive catastrophe (20–30%).

### Compound Failure Analysis

The Skeptic's compound probability estimate:

> P(at least one critical flaw) = 1 − (1−0.30)(1−0.35)(1−0.20) ≈ 64%

The Synthesizer counters that the pilot gate catches critical failures early, limiting sunk cost. The Auditor notes that some risks are correlated (PTA encoding infeasibility and E1 disappointment are not independent — if encoding is hard, the system likely finds fewer temporal conflicts).

**Resolution:** The compound probability is somewhat inflated by independence assumptions, but the directional conclusion is correct: this project has ~50% probability of encountering at least one serious obstacle. The pilot gate mitigates sunk cost but doesn't reduce failure probability. **Score: 4/10.**

### theory_bytes = 0 Assessment

**Concerning but not fatal.** The theory stage produced approach.json (39KB) with detailed formal structures (Lemma 1.1–1.4, algorithm specifications, proof sketches) but zero bytes of paper text. This means:

- Proofs are sketches, not proofs. The Lemma 1.1 monotonicity argument invokes the Metzler comparison theorem but does not verify that the state-dependent M(u)=M(x) satisfies preconditions. The Skeptic correctly identifies this as a gap.
- No paper draft means no forced confrontation with "does the writing actually work?" — a step that often reveals logical gaps.
- However, the proof *structures* in approach.json are more detailed than typical at this stage. Lemmas 1.1–1.4 are specific, falsifiable, and well-sequenced. This is a project that knows what it needs to prove, even though it hasn't proven it yet.

**Assessment:** theory_bytes=0 upgrades the urgency of Gate 0 (Theorem 1 proof writeup) from "should do" to "must do immediately." If the proof survives formal writeup, the project is on track. If a gap is found, the project faces a critical decision point.

---

## Fatal Flaws

| # | Flaw | Severity | Kill Probability | Source |
|---|------|----------|-----------------|--------|
| **F1** | **Theorem 1 proof unwritten.** Lemmas 1.1–1.4 are sketches with a potential gap at M(u)=M(x) nonlinearity. | CRITICAL | 20–30% | Skeptic, Auditor |
| **F2** | **E1 temporal ablation likely underwhelms.** Realistic X% = 10–20%. Below 15% collapses core narrative. | CRITICAL | 30–40% | All three |
| **F3** | **Zero demand signal.** No stakeholder wants this. 80%+ probability per R7. | SERIOUS | N/A (impact, not viability) | All three |
| **F4** | **Formal guarantees strongest where simplest alternatives exist.** Contract decomposition covers CYP inhibition (DrugBank detects in O(1)). Most dangerous interactions (QT, serotonin) fall to uncharacterized monolithic path. | SERIOUS | N/A (value limiter) | Skeptic |
| **F5** | **Corpus starvation.** ~30–50 CQL treatment guidelines. 50–60% probability. | HIGH | Degrades to toy demo | All three |
| **F6** | **No pilot data.** Zero evidence PTA encoding works or BMC terminates. | HIGH | 15–20% | Auditor, Skeptic |
| **F7** | **Schedule has 12.5% margin.** 35K novel LoC in 175 person-days. | HIGH | ~50% slip probability | Skeptic |
| **F8** | **CQL-to-PTA compilation deferred.** Manual encoding means system demonstrates nothing about automated verification of computable guidelines. | MEDIUM | N/A (narrative limiter) | Skeptic |

---

## Composite Scores

| Pillar | Auditor | Skeptic | Synthesizer | **Resolved** |
|--------|---------|---------|-------------|-------------|
| Extreme Value | 3 | 3 | 5 | **3** |
| Genuine Difficulty | 7 | 5 | 7 | **6** |
| Best-Paper Potential | 4 | 3 | 5 | **4** |
| Laptop-CPU Feasibility | 7 | 4 | 8 | **6** |
| Overall Feasibility | 4 | 3 | 5 | **4** |

**Composite: (3 + 6 + 4 + 6 + 4) / 5 = 4.6/10**

**Risk-adjusted composite: 4.3/10** (accounting for 30–40% E1 disappointment, 20–30% proof gap, 50–60% corpus starvation)

---

## VERDICT: CONDITIONAL CONTINUE

**Vote: 2–1 (Auditor and Synthesizer: CONTINUE; Skeptic: ABANDON at 3.6/10)**

The project has a genuine intellectual core — Theorem 1's CYP-enzyme interface contracts with Metzler monotonicity is novel and practically motivated; the two-tier architecture is a real design insight; the PTA formalism is new. The honesty caliber is exceptional. But these strengths are surrounded by multiple independent failure modes, zero demand signal, and a centerpiece experiment that is a coin flip.

The 2–1 CONDITIONAL CONTINUE reflects: the intellectual contribution justifies continued exploration, but only under strict gating. The Skeptic's ABANDON recommendation at 3.6/10 is within the margin of a reasonable decision — this is a genuinely borderline project.

---

## 5 BINDING CONDITIONS (Hard Kill Gates)

### Gate 0: Theorem 1 Proof (Weeks 1–2)
**Deliverable:** Formal proof of Lemmas 1.1–1.4, written to publication quality, addressing:
- (a) The M(u)=M(x) nonlinearity: verify Metzler comparison theorem preconditions for the state-dependent case
- (b) Parameter-space boundary: prove monotonicity holds across entire [φ_lo, φ_hi], not just at nominal values
- (c) Competitive inhibition restriction: state precisely where the proof breaks for non-competitive/mixed mechanisms

**Kill criterion:** If a gap is found that cannot be patched within the competitive-inhibition restriction, ABANDON the contract-based composition approach. Fall back to theory paper on PTA formalism alone (HSCC/CAV workshop).

### Gate 1: PTA Pilot (Weeks 3–5)
**Deliverable:** Manually encode 3 guideline pairs as PTA:
1. ADA diabetes + ACC/AHA hypertension
2. ADA diabetes + KDIGO CKD
3. ACC/AHA hypertension + CHEST anticoagulation

Demonstrate: (a) PTA encoding is feasible, (b) model checker terminates within 30 minutes per pair, (c) at least one non-trivial temporal conflict is found.

**Kill criterion:** If encoding fails for ≥2 pairs OR model checker times out on all pairs OR zero temporal conflicts found → ABANDON full system. Redirect to theory-only paper.

### Gate 2: E1 Calibration (Weeks 5–7)
**Deliverable:** Literature-calibrate temporal-only detection fraction X%. Construct 5–10 synthetic guideline pairs with known temporal vs. atemporal interactions. Measure X% on synthetic pairs.

**Kill criterion:** If synthetic calibration shows X < 10% AND literature analysis finds no plausible schedule-dependent conflict patterns → pivot paper framing entirely to compositionality theorem (E4) + explanation quality. E1 becomes a secondary experiment, not the centerpiece.

### Gate 3: Monolithic BMC Feasibility (Weeks 5–7)
**Deliverable:** Run monolithic SAT-based BMC on 2–3 PD interaction pairs (QT prolongation, serotonin syndrome). Report wall-clock times.

**Kill criterion:** If BMC takes >2 hours per pair for 2-drug PD interactions → acknowledge as limitation. If BMC does not terminate within 24 hours → the monolithic path is non-functional. Report honestly; do not claim "sound verification for all interaction types."

### Gate 4: Corpus Assessment (Week 8)
**Deliverable:** Complete audit of available CQL treatment-logic artifacts (CDS Connect, CQFramework, academic repos). Produce an honest count with source-by-source breakdown.

**Kill criterion:** If total treatment-logic artifacts < 20 → frame paper explicitly as "3–5 case study pairs" proof-of-concept. Do not claim evaluation on a "corpus." Adjust evaluation plan accordingly.

---

## Probability Estimates

| Outcome | Probability |
|---------|-------------|
| P(all 5 gates pass) | 35–45% |
| P(ABANDON at some gate) | 25–35% |
| P(significant pivot/rescope) | 25–35% |
| P(publishable paper at any venue) | 45–55% |
| P(accept at AIME) | 25–40% |
| P(best paper at AIME) | 5–8% |
| P(best paper at optimal venue) | 5–8% |
| P(abandon after full implementation) | 10–15% |

---

## What the Community Expert Thinks

As someone who works in health and life sciences computing and knows what practitioners actually struggle with:

**The honest truth:** This is a beautifully engineered solution to a problem that the clinical informatics community hasn't yet recognized as urgent. The formal methods are sound, the architecture is clever, and the honesty is exceptional. But practitioners care about two things: (1) does it catch something Lexicomp/Micromedex/DrugBank misses? and (2) can I use it without understanding timed automata?

The answer to (1) is "maybe, for ~10–20% of interactions, if you squint at schedule-dependent cases." The answer to (2) is "no, because guidelines must be manually encoded as PTA." This makes GuardPharma an academic demonstration, not a clinical tool — which is fine for AIME, but limits the "extreme value" score.

The project's strongest asset is its intellectual integrity. The risk registry, the amendment trail, the honest labeling of results — these are rare. A reviewer who sees "we estimate zero demand signal at 80%+" will respect the project more, not less. This honesty is the project's best-paper differentiator.

The project's weakest point is the Skeptic's "strongest where weakest" argument. If I'm on a best-paper committee at AIME and I see that the compositional verification theorem provides polynomial scaling for interactions any pharmacy system already catches, while the most dangerous interactions fall to an uncharacterized monolithic fallback, I would ask: "who is this for?" The answer needs to be "this is infrastructure for the formal verification of clinical decision support" — a vision statement, not a clinical tool. Frame accordingly.

**My recommendation:** Proceed through the gates. If Gates 0–1 pass, this has a credible path to an AIME publication with ~35–45% acceptance probability. Best-paper probability is ~5–8% — honest but not negligible. If E1 delivers X ≥ 20%, the paper becomes genuinely exciting. If E1 disappoints, the compositionality theorem + PTA formalism + existence proof is still a publishable contribution, just not a best paper.

---

## Scores Summary

| Dimension | Score | Key Evidence |
|-----------|-------|-------------|
| **Extreme Value (V)** | **3/10** | Zero demand signal (R7 ≥80%). Near-zero CQL adoption. Formal guarantees strongest for interactions detectable by DrugBank. Most dangerous interactions (QT, serotonin) outside compositional framework. |
| **Genuine Difficulty (D)** | **6/10** | Three-domain intersection is real. ~28–35K novel LoC. Every technique individually known; domain-specific combination is genuinely hard. Not extreme difficulty — a strong PhD student with the right background could build this in a year. |
| **Best-Paper Potential (BP)** | **4/10** | P(best paper at AIME) ≈ 5–8% after risk adjustment. E1 is a coin flip. One theorem at depth 6/10 is the crown jewel. Fallback narratives are viable but not best-paper material. |
| **Laptop-CPU & No-Humans (L)** | **6/10** | Compositional path well-founded for CPU. E9 pharmacists evaluate output (acceptable). Monolithic BMC convergence uncharacterized for PD interactions (docked). NTI drugs handled via scoping. |
| **Feasibility (F)** | **4/10** | theory_bytes=0, impl_loc=0, no pilot. 35K novel LoC with 12.5% schedule margin. Multiple independent failure modes (compound ~50%). Pilot gate mitigates sunk cost but not failure probability. |

**Composite: V3 / D6 / BP4 / L6 / F4 = 4.6/10**
**Risk-adjusted: 4.3/10**

**VERDICT: CONDITIONAL CONTINUE** (2–1; Skeptic dissents ABANDON at 3.6/10)

5 binding gates. Kill probability 25–35%. P(best-paper at AIME) ≈ 5–8%. P(any publication) ≈ 45–55%.

---

*Evaluation produced by 3-expert team with adversarial cross-critique. Independent Auditor (evidence-based scoring), Fail-Fast Skeptic (aggressive rejection testing), Scavenging Synthesizer (value salvage). All disputes resolved by forced arbitration with cited evidence.*

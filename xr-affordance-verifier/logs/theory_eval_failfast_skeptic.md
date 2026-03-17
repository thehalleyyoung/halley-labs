# Fail-Fast Skeptic Assessment: Coverage-Certified XR Accessibility Verifier (proposal_00)

**Evaluator:** Fail-Fast Skeptic (Independent Panel Member)  
**Date:** 2026-03-08  
**Posture:** Aggressively reject under-supported claims. Find reasons to ABANDON.

---

## Axis Scores

### 1. EXTREME AND OBVIOUS VALUE — 2/10

The proposal solves a problem **nobody has demonstrated exists as a purchasing decision.** Zero validated demand. Zero XR-specific accessibility regulation. The proposal's own "honest market assessment" concedes this: "Zero developer demand has been validated." The value narrative chains three speculative links: (a) XR will become regulated → (b) spatial accessibility will be the regulated dimension → (c) formal verification is the response developers want. Each link is ≤50% probable; chained: ≤12.5%.

The 1.3-billion-disability figure is irrelevant misdirection. The intersection of "motor disability," "owns XR headset," and "uses enterprise XR application that would adopt this tool" is orders of magnitude smaller. The proposal even admits the addressable developer population is 30-50K — and zero of them have asked for this.

The 500-LoC lookup table objection stands: a table mapping "5th-percentile reach envelope" to "does element fall inside?" captures the vast majority of *actionable* accessibility bugs (overhead buttons, distant targets). The certificate's spatial-mathematical sophistication addresses the remaining slim margin of subtle frontier cases — cases where the developer response is almost certainly "move the button 10cm" regardless of whether ε = 0.02 or ε = 0.06.

**Why this score is fatal on its own:** A tool nobody wants, solving a problem nobody has articulated, in a market that doesn't exist yet, cannot justify the 43-68K LoC investment. Score: **2**.

### 2. GENUINE DIFFICULTY AS A SOFTWARE ARTIFACT — 4/10

The mathematician's evaluation is devastating and correct: "A strong researcher could derive C1 in 2-3 weeks given the problem setup." C1 is B+ composition of textbook components — Hoeffding inequality, union bound, volume subtraction, piecewise-Lipschitz partitioning. The novel *assembly* is real but modest.

The **engineering** difficulty is significant (Unity parsing, Pinocchio integration, affine-arithmetic FK chains) but is integration work, not algorithmic novelty. The genuinely hard pieces — frontier-resolution (Algorithm 5), the load-bearing novel algorithm — exist only as pseudocode. Every killed idea (M2 analytical Lipschitz, M3b) was where the real difficulty lived.

The 43-68K LoC estimate is inflated by infrastructure (scene parsing, editor plugins, benchmarks) that would be equally needed for the trivial lookup-table approach. Strip that away and the novel-difficult core is ~5-8K LoC of certificate engine plus ~3-5K of frontier-resolution logic. Respectable but not "impressively hard." Score: **4**.

### 3. BEST-PAPER POTENTIAL — 2/10

Best-paper requires one of: breakthrough result, paradigm shift, or extraordinary practical impact. This proposal offers none.

**CAV angle:** C1 is acknowledged as B+ novelty — known components in new assembly. CAV best papers feature deep new theory (CEGAR, IC3, DPLL(T)). A coverage certificate that achieves ε ≈ 0.022 when Clopper-Pearson delivers ε_CP ≈ 2.8×10⁻⁴ from identical samples is **numerically worse** on the headline metric. The structural advantages (spatial map, counterexamples) are presentation features, not theorems. The 77× gap between ε_cert and ε_CP is the number reviewers will focus on.

**UIST angle:** Killed by the no-humans constraint. UIST demands user studies. The proposal *plans* a 15-22 developer study (Section Phase 4), but the hard constraint forbids it. Without a user study, UIST/ASSETS reviewers will reject on methodological grounds, full stop.

**The "generalizable framework" hook:** Claiming coverage certificates apply to robotics, drug dosage, autonomous vehicles is aspirational hand-waving without a single worked example outside XR. Reviewers will see through this immediately. Score: **2**.

### 4. LAPTOP-CPU & NO-HUMANS — 6/10

This is the proposal's strongest axis, and even here it's mixed. Pinocchio FK at ~20K evaluations/sec on CPU is feasible. Z3 on QF_LRA is CPU-friendly. The 5-15 minute Tier 2 budget is laptop-plausible.

**But:** The full pipeline requires Unity scene parsing, which means either (a) running Unity editor on the laptop (heavy, license-dependent), or (b) offline YAML parsing (fragile, undocumented format quirks). The 10-minute Tier 2 budget yields ε ≈ 0.04-0.06, which the synthesis itself acknowledges as barely useful. Extending to overnight runs (Tier 3) may exceed "laptop" interpretation. The 500-scene procedural benchmark generation is CPU-feasible, and the evaluation framework runs without humans.

Deductions for: Unity dependency complexity, ε looseness within laptop budget, and the fact that the no-humans constraint kills the UIST venue entirely. Score: **6**.

### 5. FEASIBILITY — 2/10

This is where the proposal dies.

**impl_loc = 0 at "theory_complete."** Not a single line of running code exists. The gap between 539 lines of formal methods proofs and 43-68K LoC of implementation is immense.

**The compound failure chain (my calculation):**

| Gate | P(fail) | Independence? |
|------|---------|---------------|
| D1: Wrapping factor > 10× after subdivision | 0.25 | Independent |
| D2: ε > 0.10 on simple scenes | 0.35 | Partially correlated with D1 |
| D3: Certificate fails to beat CP by 2× | 0.45 | Core load-bearing; correlated with frontier-resolution |
| D4: κ > 0.30 on typical scenes | 0.25 | Independent |
| D7: <10% developer interest | 0.60 | Independent (market risk) |
| Frontier-resolution doesn't work | 0.50 | Load-bearing for D3 |
| Unity parser fails on real scenes | 0.35 | Independent |
| Paper accepted at any top venue | 0.25 | Conditional on all above passing |

**P(at least one critical failure):** 1 - (0.75 × 0.65 × 0.55 × 0.75 × 0.40 × 0.50 × 0.65) = 1 - 0.023 = **~97.7%**.

Even with generous independence assumptions and the synthesis's own revised probabilities, compound failure probability is ~**88%** (using only the four most critical gates: D1, D3, D7, frontier-resolution).

The synthesis claims ~65% compound risk with mitigations — this is optimistic accounting that treats fallback paths as risk elimination rather than scope reduction. Falling back from "coverage certificate paper" to "tool paper" to "lookup table" is not mitigation; it's abandoning the research contribution at each step. Score: **2**.

---

## Composite Score: **16/50**

| Axis | Score |
|------|-------|
| Value | 2 |
| Difficulty | 4 |
| Best-Paper | 2 |
| Laptop-CPU | 6 |
| Feasibility | 2 |
| **Total** | **16** |

---

## Fatal Flaws (Genuinely Fatal — Not Merely Serious)

### F1. No-Humans Constraint Kills Both Viable Venues
The UIST/ASSETS paper requires a developer study (the proposal explicitly plans one). Under the no-humans constraint, this paper cannot be written. The CAV paper's ε is numerically dominated by Clopper-Pearson (77× worse). There is no venue where this work is competitive *and* feasible under constraints.

### F2. The Certificate Is Quantitatively Worse Than Its Baseline
ε_cert ≈ 0.022 (best case, corrected, with frontier-resolution working). ε_CP ≈ 2.8×10⁻⁴ from identical samples. A reviewer's first question: "Why would I use your certificate instead of Clopper-Pearson?" The structural answer (spatial map, counterexamples) is a feature of the *tool*, not of the *certificate formalism*. You can generate spatial maps and counterexamples from Monte Carlo samples without any certificate theory.

### F3. κ-Exclusion Is Self-Defeating
The certificate exempts 10-30% of the parameter space near joint-limit surfaces. These surfaces are exactly where disability populations cluster. A certificate that guarantees accessibility for the 10th-90th percentile while exempting wheelchair users, limited-ROM users, and people with arthritis is an accessibility tool that doesn't work for people with disabilities. This is not a limitation paragraph — it's a contradiction of the value proposition.

### F4. Frontier-Resolution Is Load-Bearing, Unproven, and Unbuilt
The ONE algorithm that could save the certificate's quantitative story (frontier-resolution, Algorithm 5) has: no formal proof (Lemma B2 is "conditional"), no implementation, no empirical validation, and the synthesis classifies it as "plausible but unproven." Every quantitative claim beyond ε ≈ 0.04-0.06 depends on this single unbuilt mechanism.

### F5. Zero Demand Validated Against Zero Existing Market
No XR developer has asked for formal verification of spatial accessibility. No XR-specific regulation requires it. The tool's value proposition is entirely speculative. Gate D7 (Month 3 developer feedback) is the only demand test, and it's too late — by Month 3, thousands of hours are invested.

---

## Compound Failure Probability: ~88-98%

Using the four most critical independent failure modes:
- P(frontier-resolution fails) = 0.50
- P(no venue accepts without user study) = 0.70
- P(developer demand < 10%) = 0.60  
- P(ε > 0.05 on typical scenes) = 0.35

P(success) = P(all four pass) ≤ 0.50 × 0.30 × 0.40 × 0.65 = **0.039 = ~4% success probability.**

Even being maximally generous on each factor: P(success) ≤ 0.60 × 0.50 × 0.50 × 0.70 = **0.105 ≈ 10%**.

---

## What Evidence Would Change My Mind

| Evidence Needed | Obtainable Under Constraints? |
|----------------|------------------------------|
| 10+ XR developers independently requesting formal accessibility verification | No — requires market research / user interviews (human study) |
| Frontier-resolution achieving ε < 0.01 on 50+ scenes (empirical) | Yes — but requires building the full pipeline first (months of work) |
| ε_cert < ε_CP on identical sample budgets | Almost certainly impossible — Hoeffding cannot beat Clopper-Pearson by construction |
| κ < 0.05 on realistic scenes with disability populations | Maybe — requires empirical measurement; the math predicts 10-30% |
| A top venue accepting a formal-methods tool paper without user studies | Possible at CAV/TACAS only — but the ε gap makes CAV acceptance unlikely |

The critical observation: the evidence that could save this project (demand validation, frontier-resolution empirics) **cannot be obtained cheaply.** You must build most of the system before you can test whether the research contribution exists. This is the worst risk profile — expensive-to-test hypotheses with low prior probability.

---

## What IS Good (Intellectual Honesty)

1. **The synthesis document is excellent.** The cross-critique process, disagreement resolution, and risk quantification are genuinely high-quality research methodology. The team is honest about limitations.
2. **The piecewise-Lipschitz formulation** (Decision 1) is the right mathematical framework and is well-executed in the formal methods proposal.
3. **Tier 1 interval linter** is a solid engineering contribution that could genuinely help developers, even if the certificate story collapses.
4. **Decision 7** (Tier 1 envelopes as verified volume) is a clever insight that partially rescues the SMT value proposition.

None of these are good **enough.** Points 1-2 are craftsmanship, not publishable novelty. Point 3 is a tool without a venue (no-humans kills UIST). Point 4 is a mitigation for a symptom, not a cure for the disease.

---

## Verdict: **ABANDON**

The project has a ~4-10% probability of producing a publication under the stated constraints. The no-humans requirement eliminates the UIST paper entirely. The certificate's quantitative inferiority to Clopper-Pearson makes the CAV paper a hard sell. Zero validated demand means the tool may never find users even if built. The load-bearing novel contribution (frontier-resolution) is unproven and unbuilt.

**The salvage path** (stratified MC tool, ~2-5K LoC, Tier 1 linter only) is the right call if the team wants to build something useful — but it is an engineering artifact, not a research contribution, and produces no publication.

I see no path to a top-venue publication under these constraints. ABANDON.

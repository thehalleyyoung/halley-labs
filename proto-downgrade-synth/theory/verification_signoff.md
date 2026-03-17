# Independent Verifier Signoff: NegSynth Evaluation

**Proposal:** proto-downgrade-synth (NegSynth)  
**Title:** "Negotiation Under Fire: Bounded-Complete Synthesis of Protocol Downgrade Attacks from Library Source Code"  
**Role:** Independent Verifier — Final Signoff on theory_eval_mathematician.md  
**Date:** 2026-03-08  
**Inputs:** theory_eval_mathematician.md (binding), critique_synthesis.md, chair_initial.md, community_expert eval, depth_check.md, formal_proposal.md, redteam_proposal.md  
**Status:** SIGNOFF DECISION

> **Summary:** The synthesized evaluation is internally consistent, well-evidenced, and honest. The CONDITIONAL CONTINUE verdict is justified at composite 5.6 — but only because the gated fallback structure converts a marginal expected-value project into one with a positive-EV publication cascade. Two arithmetic discrepancies require correction (composite is 5.50 weighted, not 5.6; the document conflates the two). The probability estimates are well-calibrated with one exception: P(any-pub)=50% is too conservative given the documented fallback cascade and should be 55–60%. The math depth assessment is the evaluation's strongest section and requires no changes. Approved with two minor corrections.

---

## VERDICT: APPROVED WITH CHANGES

---

## 1. Score Consistency Check

### 1.1 Arithmetic Verification

The document reports two composite calculations:

- **Weighted:** (6×0.25 + 6×0.15 + 4×0.20 + 7×0.15 + 5×0.25) = 1.50 + 0.90 + 0.80 + 1.05 + 1.25 = **5.50** ✅ Arithmetic correct.
- **Simple average:** (6+6+4+7+5)/5 = **5.6** ✅ Arithmetic correct.

**Issue:** The document uses "5.6" as the headline composite throughout but the weighted score is 5.50. The weighting scheme (Value 25%, Feasibility 25%, Best-Paper 20%, Difficulty 15%, Laptop 15%) is sensible — it correctly weights the dimensions that most affect publication outcome. The document should commit to one number. **Recommendation: Use 5.5 (weighted) as the canonical composite.** The 0.1-point difference is immaterial to the verdict but the inconsistency undermines precision.

### 1.2 Score-Evidence Alignment

| Dimension | Score | Evidence Consistent? | Notes |
|-----------|:---:|:---:|-------|
| Value: 6 | ✅ | Yes | The CompCert calibration anchor (unconditional certs → 7, conditional → 6) is well-reasoned. The 7 scope exclusions are enumerated and each is justified. SSH+legacy value correctly prevents further deflation to 5. |
| Difficulty: 6 | ✅ | Yes | The "every subsystem has a published template" finding is factual (Weiser, Kuznetsov, Clarke, AVISPA all cited). The distinction between integration difficulty (real) and invention difficulty (absent) is the correct frame. Deflation from ideation's 7 is justified — the community expert's consensus at 7 overweighted LoC count relative to algorithmic novelty. |
| Best-Paper: 4 | ✅ | Yes | This is the evaluation's most important deflation (−3 from ideation) and the evidence is compelling. T3 at depth 4/10, certificates conditional on A0, no new CVE likely, O(n) reframed to subroutine-only. The Skeptic's argument was correctly weighted. |
| Laptop-CPU: 7 | ✅ | Yes | The architecture IS naturally suited to commodity hardware. The 40% Z3 timeout risk is correctly classified as a feasibility issue, not a hardware issue. Score appropriately holds from ideation. |
| Feasibility: 5 | ✅ | Yes | The compound gate calculation (0.85×0.70×0.75×0.60 ≈ 27% full pipeline) is credible. The fallback cascade raising the floor to 50% is the key argument. A coin-flip for publication is exactly what a 5 should mean. |

**Assessment: All five scores are internally consistent with their cited evidence. No score requires revision.**

### 1.3 Cross-Score Consistency

One potential tension: Difficulty at 6 while Feasibility at 5 implies the project is *moderately hard* but has *coin-flip* chances. This is consistent if the difficulty is primarily *integration complexity with known techniques* (low invention risk, high integration risk) — which is exactly the evaluation's thesis. No inconsistency.

Another check: Best-Paper at 4 with Value at 6 implies the project is useful but not exciting enough for top recognition. This is internally consistent — a heavily caveated certificate for a narrow audience is valuable to that audience but unlikely to generate broad excitement. No inconsistency.

---

## 2. Verdict Consistency Check

### 2.1 Does CONDITIONAL CONTINUE follow from 5.5?

A composite of 5.5 is in the ambiguous zone — some 5.5 projects should be abandoned. The evaluation correctly identifies what makes this one worth continuing:

1. **Positive-EV fallback cascade.** The Synthesizer's three fallback tiers (money-plot paper, slicer-as-tool, salvage components) mean the expected outcome is not zero even if the full pipeline fails. P(something publishable) ≈ 50% exceeds the ~20% threshold where continuation is irrational.

2. **Gated execution converts compound risk into sequential binary decisions.** The kill gates at weeks 2/4/6/10 mean sunk cost is limited: maximum ~4 weeks lost if G0 fails, ~10 weeks if G2 fails. This is the correct structure for a high-variance project.

3. **The problem is real and persistent.** The recent CVE evidence (wolfSSL 2024-5814, Terrapin 2023) validates continued relevance. This isn't a solution searching for a problem.

4. **No competitor occupies the niche.** tlspuffin requires hand-authored models; ProVerif/Tamarin verify specs not code. If NegSynth works, it has the niche to itself.

**Assessment: CONDITIONAL CONTINUE is the correct verdict.** An ABANDON verdict would require either (a) P(any-pub) < 30%, (b) the problem being solved by another tool, or (c) a truly fatal flaw. None of these conditions hold. The 5.5 composite is marginal, but the gated execution and fallback structure provide genuine insurance that raises the risk-adjusted expected value above the continuation threshold.

### 2.2 What would flip the verdict?

The evaluation correctly identifies G0 (KLEE on ssl_ciph.c) as the first binary gate. I would add: if the Z3 feasibility prototype (community expert's Condition 3) fails on a hand-encoded 200-line model, that should also be a KILL signal, not just a reassessment. The mathematician's evaluation does not include this condition — it should.

---

## 3. Missing Considerations

### 3.1 Stakeholder Analysis

The evaluation correctly identifies the narrow audience (~50 library maintainers). **Missing:** The evaluation does not consider a secondary stakeholder class — *compliance and certification bodies* (e.g., FIPS 140-3 labs, Common Criteria evaluators, PCI-DSS auditors). A bounded-completeness certificate, even caveated, could become part of a formal compliance workflow. This would not change the value score (compliance adoption is speculative and multi-year) but should be mentioned as upside potential in the "what could go right" section.

**Impact on scores: None.** This is upside optionality, not a scoring error.

### 3.2 Competitive Landscape

The evaluation mentions tlspuffin as the primary competitor. **Missing consideration: what if tlspuffin v2 adds source-level analysis?** The tlspuffin team (TU Wien) is active and well-funded. If they add LLVM-level fuzzing with DY models (a natural extension of their S&P 2024 paper), NegSynth's differentiator shrinks to the merge operator + certificates. This is a realistic risk within the 18-month timeline.

**Mitigation already implicit:** The merge operator and certificate story are genuinely novel. Even if tlspuffin adds source-level analysis, it would not have bounded-completeness guarantees. NegSynth's differentiation survives, albeit with less distance.

**Impact on scores: None.** But this should be noted in the document as a competitive risk.

### 3.3 18-Month Timeline with impl_loc=0

The community expert notes 200 LoC/day sustained for 50K novel LoC over 13 implementation months. This is aggressive for research-quality code with formal properties. The mathematician's evaluation doesn't independently assess timeline risk.

**Assessment:** The kill-gate structure adequately mitigates this. If G0-G2 all pass by week 10, the remaining 8 months for evaluation and paper writing is reasonable. If gates take longer, the fallback MVP reduces scope proportionally. The evaluation handles this implicitly through the feasibility score but could be more explicit.

**Impact on scores: None.** Feasibility at 5 already prices in this risk.

### 3.4 Uncovered Failure Modes

The kill gates cover slicer size (G1), merge effectiveness (G2), and Z3 tractability (implicit in G2). **Failure modes NOT covered by gates:**

1. **KLEE bitcode compatibility with OpenSSL's build system.** OpenSSL uses a complex Perl-generated build system. Producing clean LLVM bitcode via wllvm may fail on assembly-heavy paths. G0 partially covers this (KLEE on ssl_ciph.c), but ssl_ciph.c is pure C — the real test is when the slicer pulls in code that calls assembly stubs.

2. **CEGAR non-convergence in practice.** L6 guarantees termination in bounded iterations, but the bound (|C|^k × |V|^k) could be enormous. If CEGAR takes 500 iterations to converge, it's technically terminating but practically infeasible.

3. **Paper framing risk.** Even if the system works, the paper could be rejected on framing grounds — reviewers who see "bounded completeness" in the title and then discover A0 + ε + 7 scope exclusions may react negatively to perceived overclaiming.

**Assessment:** Failure mode #3 is the most likely path to rejection even with a working system. The evaluation's Binding Condition 1 (honest framing) addresses this, but the evaluation could more strongly emphasize that *title choice* is a publication-critical decision. The title "Negotiation Under Fire" is fine; "Bounded-Complete" in the subtitle is risky given the caveats.

**Recommendation:** Add a note that the paper title should be workshopped — consider dropping "Bounded-Complete" from the title in favor of the body text.

---

## 4. Math Assessment Sanity Check

### 4.1 T3 at depth 4/10 (down from ideation's 5/10)

**Is the deflation justified?** Yes.

The ideation's depth_check rated T3 at 5/10 based on the Math Depth Assessor's score and community expert consensus at 5/10. The mathematician's evaluation deflates to 4/10 with this argument: "a graduate student who has read Kuznetsov et al. and Sangiorgi could write this proof in 2-3 weeks, given the domain observation."

I verify this by checking the proof structure in formal_proposal.md: T3 uses bisimulation up-to congruence (Sangiorgi 1998) applied to an ITE merge construction (Kuznetsov 2012). The domain-specific content is the mergeability predicate over four algebraic properties. The mathematician is correct that once you *see* the four properties, the proof is mechanical. The novelty is in *identifying* the properties, not in *proving* correctness given them.

**However**, there is a subtlety the evaluation handles well: at security venues, "noticed the right structural correspondence" IS a contribution. The evaluation acknowledges this ("important for engineering... but the proof itself writes itself once you see the observation") and correctly distinguishes between math-depth score (4/10 on a pure math scale) and venue-contribution value (still valid at S&P/CCS). The deflation from 5→4 is a 1-point correction that is well within the noise band. The key insight — that T3 is domain observation, not proof technique — is correct regardless of whether you score it 4 or 5.

**Assessment: Justified. No change needed.**

### 4.2 "No new mathematics" — fair characterization?

The evaluation states: "Nothing else that couldn't be derived from Kuznetsov + Milner + Clarke separately."

This is fair but deserves nuance. The evaluation provides the nuance in §2.5: "What's genuinely new is the BRIDGE between protocol structure and verification technique." This is the correct framing — the paper's mathematical contribution is a *connection* between two known domains, not a new technique in either. At theory venues, this would be underwhelming. At systems-security venues, bridging insights are common and valued (CompCert bridged compiler correctness and verified systems, using known proof techniques throughout).

**Assessment: Fair characterization, well-nuanced. No change needed.**

### 4.3 Does math depth matter at security venues?

The evaluation explicitly addresses this: "This is the paper's real contribution, and it is a legitimate one. The question is whether it's enough for best-paper. My assessment: it is enough for a solid accept at a top-4 venue, but not for best paper."

This is calibrated correctly. S&P/CCS best papers have deep math (rare), devastating bugs (common), or paradigm-shifting tools (occasional). NegSynth has none of these — it has a clean domain observation and solid engineering. The evaluation correctly maps this to "strong accept, not best paper."

**Assessment: Well-calibrated. No change needed.**

---

## 5. Probability Calibration

### 5.1 P(best-paper) = 3%

The evaluation argues: T3 at depth 4/10, no new CVE likely, certificates conditional on A0. Best papers require surprising math (depth 7+) or devastating empirical results.

**Cross-check:** The community expert estimated 2–4%. The ideation depth_check implied 5–10% (self-assessment). The 3% estimate is within the community expert's range and well below the self-assessment's optimism.

**Is 3% too high or too low?** For a tool with 19% full-vision probability and ~27% pipeline success, and given that best paper also requires the paper to be *accepted* AND selected as best (typically 1–3% of submissions): P(best-paper) = P(accepted) × P(selected as best | accepted). If P(accepted) ≈ 30% and P(selected | accepted) ≈ 5–10%, then P(best-paper) ≈ 1.5–3%. The 3% estimate is at the top of this range.

**Assessment: 3% is slightly generous but within the reasonable band (1.5–3%). Acceptable.**

### 5.2 P(any-pub) = 50%

The evaluation states this includes fallback plans (money-plot paper, slicer-as-tool). The community expert estimated 70–80%.

**Discrepancy analysis:** The community expert's 70–80% accounts for salvage components publishable independently at ACSAC/ESORICS tier. The mathematician's 50% appears to require at least a *coherent pipeline paper*, not individual component publications.

This is a definitional disagreement. If "any publication" includes:
- Workshop papers on the DY+SMT encoding alone
- Tool papers on the slicer alone  
- Formalism papers on the merge operator alone

...then the probability is clearly higher than 50%. The Synthesizer identified three independently publishable layers. Even if the full pipeline never works, the merge operator formalization + a small evaluation on toy examples is publishable at a formal methods workshop.

**Assessment: P(any-pub) = 50% is conservative by ~5–10 points.** The evaluation's definition of "publication" implicitly requires more than the floor of individual salvage components. If including lower-tier and workshop venues where individual components suffice, P(any-pub) should be **55–60%**. This does not change the verdict but should be corrected for calibration accuracy.

### 5.3 P(top-venue) = 30%

The evaluation states this requires "full pipeline working + 7+ CVEs recovered + certificates produced."

**Cross-check:** The community expert estimated 20–30% (full vision) and 35–45% (MVP). The 30% in the mathematician's evaluation appears to blend these: it's at the top of the full-vision range and below the MVP range.

**Assessment: 30% is reasonable as a blended estimate.** However, the evaluation should clarify that this 30% is conditional on which paper is submitted. If the MVP (OpenSSL only, 3–4 CVEs, no certificates) is submitted to CCS, P(top-venue) may be closer to 35%. If the full-vision paper is submitted to S&P, P(top-venue) is closer to 20%. The 30% blended estimate is acceptable but the conditionality should be noted.

---

## 6. Specific Changes Required

### Change 1 (REQUIRED): Fix composite score inconsistency

**Current:** Document uses "5.6" as headline but computes 5.50 (weighted) and 5.6 (simple average).

**Required:** Commit to one number. Recommend **5.5** (weighted) since the weighting scheme is explicitly justified. Update line 282 ("Simple average... **5.6**") to note this is the unweighted figure. Use 5.5 as the canonical composite in the verdict section.

### Change 2 (REQUIRED): Adjust P(any-pub) from 50% to 55–60%

**Current:** "P(any publication) = 50% — Fallback plans raise floor: money-plot paper or slicer-as-tool paper at workshop/B-tier."

**Required:** Revise to 55–60%. The evaluation's own fallback cascade (§4, "What the Synthesizer's Fallback Plan Adds") documents three tiers with an 85% catchment for slicer-as-tool. If P(slicer-as-tool) ≈ 85% and P(that's publishable somewhere) ≈ 65–70%, then P(any-pub) ≥ 55%. The 50% figure is inconsistent with the documented fallback insurance value.

### Change 3 (RECOMMENDED): Add Z3 feasibility prototype as explicit kill gate

**Current:** G0 tests KLEE on ssl_ciph.c. Z3 tractability is tested implicitly via G2.

**Recommended:** Add an explicit Z3 micro-gate between G1 and G2: "G1.5 (Week 5): Hand-encode FREAK CVE on 200-line negotiation model; Z3 returns SAT in <10 min. KILL if >30 min." This is the community expert's Condition 3, which the mathematician's evaluation does not incorporate.

### Change 4 (RECOMMENDED): Note title framing risk

**Current:** Binding Condition 1 requires honest framing in the paper body.

**Recommended:** Extend to paper title. Consider that "Bounded-Complete" in the subtitle creates reviewer expectations that the 7 scope exclusions and A0 assumption cannot fully satisfy. Suggest workshopping a title that emphasizes the certificates rather than the completeness claim — e.g., "Protocol-Aware Symbolic Execution for Downgrade Attack Synthesis and Certification."

### Change 5 (MINOR): Note competitive risk from tlspuffin v2

**Current:** tlspuffin mentioned as prior work, not as competitive threat.

**Recommended:** Add one sentence: "If the tlspuffin team extends to source-level analysis within the project timeline, NegSynth's differentiator narrows to the merge operator and certificate story — still novel, but with reduced distance."

---

## 7. Final Calibrated Scores

| Dimension | Synthesis Score | Verifier Score | Change? |
|-----------|:---:|:---:|---------|
| Value | 6 | **6** | No change |
| Difficulty | 6 | **6** | No change |
| Best-Paper | 4 | **4** | No change |
| Laptop-CPU | 7 | **7** | No change |
| Feasibility | 5 | **5** | No change |
| **Composite (weighted)** | **5.5** | **5.5** | Fix headline from "5.6" |

| Probability | Synthesis | Verifier | Change? |
|-------------|:---------:|:--------:|---------|
| P(best-paper) | 3% | **3%** | No change |
| P(top-venue) | 30% | **25–30%** | Minor: note conditionality on which paper/venue |
| P(any-pub) | 50% | **55–60%** | Increase: inconsistent with documented fallback cascade |
| P(abandon) | 30% | **30%** | No change |

---

## 8. Additional Binding Conditions

Beyond the six conditions from the mathematician's evaluation, I add:

### Condition 7: Z3 Feasibility Micro-Prototype (Week 5)

Hand-encode the Dolev-Yao model for FREAK (CVE-2015-0204) on a 200-line negotiation model in SMT-LIB. Z3 must return SAT with a satisfying assignment in <10 minutes. If Z3 times out at 30 minutes on this toy instance, the production-scale encoding is non-viable. KILL the SMT path; pivot to direct symbolic execution with property checking (Fallback A from Synthesizer's cascade).

**Rationale:** This is the cheapest test (~1 week effort) of the highest-variance risk. The community expert identified it as Condition 3. The mathematician's evaluation omits it — an oversight given Z3 tractability is flagged as the dominant uncertainty.

---

## 9. Disposition

**APPROVED WITH CHANGES.**

Changes 1–2 are required (arithmetic precision and probability calibration). Changes 3–5 are recommended (strengthen the kill-gate structure and risk documentation). All individual scores are confirmed. The CONDITIONAL CONTINUE verdict is correct and well-supported by the evidence.

The evaluation is thorough, honest, and well-structured. The adversarial cross-critique methodology (Auditor/Skeptic/Synthesizer → final synthesis) produced a balanced assessment. The Skeptic's contributions were correctly weighted — the math depth argument constrains Best-Paper, the "bounded model checking" reframing constrains the narrative, and the zero-implementation-code observation constrains confidence — without being allowed to dominate the verdict.

The mathematician's §2 (Mathematics Assessment) is the evaluation's strongest section: the theorem-by-theorem depth analysis is precise, honest, and actionable. The distinction between "domain observation that enables a clean argument" (T3) and "genuine proof technique innovation" (absent) is the key insight that correctly calibrates expectations.

**This evaluation, with the two required changes applied, is ready to serve as the official theory_eval_mathematician.md.**

---

*Signed: Independent Verifier, 2026-03-08*

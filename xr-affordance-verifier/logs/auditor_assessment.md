# Independent Auditor Assessment: Coverage-Certified XR Accessibility Verifier

**Assessor:** Independent Auditor (evidence-based scoring with challenge testing)  
**Date:** 2026-03-08  
**Proposal:** proposal_00 — Coverage-Certified XR Accessibility Verifier  
**Method:** Every score is grounded in specific textual evidence from the three prior evaluations and primary source materials.

---

## Axis 1: EXTREME AND OBVIOUS VALUE — Score: 3/10

**Evidence for:**
- The problem statement claims 1.3B people with disabilities and cites Section 508/ADA Title I (problem_statement.md:75–81). This frames a real population, but population size ≠ demand.
- Community Expert gave V=4, noting "Decision 7 (Tier 1 envelopes as verified volume) identified as most promising contribution."

**Evidence against:**
- All three evaluations independently scored value at 3–4. This convergence is significant.
- The proposal itself concedes: "Zero developer demand has been validated" and "No XR-specific accessibility regulation exists today" (proposal_00/problem.md:27).
- Skeptic: "zero demand" listed as a fatal flaw. The D7 demand gate at Month 3 tests >25% developer interest, but the proposal provides no evidence this threshold will be met.
- Mathematician: "Math is load-bearing for Tier 2 but NOT for Tier 1" — meaning the publishable contribution targets a tier without validated users.
- Community Expert panel split: Skeptic sub-panelist scored 18/50 with ABANDON, driven by the demand vacuum.
- The EU Accessibility Act "does not currently name XR explicitly" (problem_statement.md:76–77). The legal demand narrative is speculative.

**Challenge test:** Can I find *any* evidence of validated demand — a developer survey, a forum thread, an enterprise RFP? None exists in the materials. The value proposition rests entirely on analogy ("ESLint-like adoption dynamics") and regulatory trajectory projection.

**Score rationale:** A tool solving a problem nobody has yet asked for. The accessibility framing is morally compelling but commercially and academically unvalidated. The 3/10 consensus across all evaluators is well-supported.

---

## Axis 2: GENUINE DIFFICULTY AS A SOFTWARE ARTIFACT — Score: 5/10

**Evidence for:**
- 28–40K LoC rescoped, ~13–22K classified as difficult (Key Facts).
- The PGHA formalism combining SE(3) pose space with discrete automaton state is mathematically nontrivial (problem_statement.md:37–49).
- Red-Team's Attack 1.1 reveals genuine complexity: 420 joint-limit transition surfaces in parameter space for a 30-element scene (synthesis.md:23–28).
- The curse of dimensionality at k=3 multi-step (d_eff=26) is a real computational barrier (synthesis.md:101–116).

**Evidence against:**
- Mathematician: C1 crown jewel is "composition of known tools, not new proof technique" — graded B+.
- Skeptic: D=5, noting the approach combines existing techniques (Pinocchio FK, Z3 SMT, affine arithmetic, Hoeffding bounds) without a fundamentally novel algorithmic contribution.
- Community Expert: D=5 unanimously across the panel.
- The Tier 1 linter is essentially affine-arithmetic forward kinematics + interval intersection — well-understood techniques applied to a new domain.

**Challenge test:** What specific algorithmic invention does this require that doesn't exist in the literature? The synthesis itself names the components: Pinocchio (existing), Z3/CVC5 (existing), affine arithmetic (existing), stratified sampling with Hoeffding (textbook). The integration is engineering-hard but not research-hard.

**Score rationale:** Solid integration engineering across known verification techniques. The domain adaptation is real work but not a difficulty frontier. All evaluators agree at 5.

---

## Axis 3: BEST-PAPER POTENTIAL — Score: 3/10

**Evidence for:**
- The κ-completeness framework (synthesis.md Decision 6) and Decision 7 (Tier 1 envelopes as verified volume) were identified by Community Expert as the most promising contributions.
- Dual-ε reporting (Decision 8) is a clean methodological contribution.
- The synthesis document is exceptionally well-structured — 10 decisions, 8 resolved disagreements, honest unresolved issues section. This intellectual rigor is itself a signal.

**Evidence against:**
- Mathematician: P(best-paper) ≈ 0.5–1.5%. "C1 is B+ grade — composition of known tools."
- Skeptic: P(best-paper) ~3–5%. "UIST dead under no-humans."
- Community Expert: P(best-paper) ≈ 1–3%. Panel split with Skeptic sub-panelist scoring BP=2.
- The no-humans constraint kills UIST (the natural venue). ISSTA/FSE pivot requires reframing as a testing tool, where the ε ≈ 0.04–0.06 headline number competes poorly against standard Clopper-Pearson bounds (Red-Team Attack 1.2).
- Hoeffding can never beat Clopper-Pearson on the same data (Mathematician evaluation: "77× gap"). The certificate's quantitative headline will always look worse than naive MC to a reviewer who doesn't read carefully.

**Challenge test:** At ISSTA/FSE, would a PC member select this over a paper with a novel testing algorithm that finds real bugs in real software? The XR accessibility domain is too niche and unvalidated for broad appeal; the formal methods contribution (B+) is too incremental for a methods venue like CAV.

**Score rationale:** The work falls into a "between venues" gap — too applied for theory venues, too theoretical for HCI (especially without human studies), and the domain is too unvalidated for practical SE venues. The dual-ε and κ-completeness ideas are interesting but incremental.

---

## Axis 4: LAPTOP-CPU & NO-HUMANS — Score: 7/10

**Evidence for:**
- Tier 1 runs in <2 seconds via affine arithmetic (proposal_00/problem.md:15). This is trivially CPU-friendly.
- Tier 2 targets 5–15 minutes with ~20K FK evaluations/second from Pinocchio (Red-Team Attack 1.2 computation). 4M evaluations in 5 minutes is achievable on a modern laptop CPU.
- Z3 on QF_LRA (linear real arithmetic after linearization) with 2s timeout per query — 1000 queries × 2s = 33 minutes worst case, well within laptop budget (proposal_00/problem.md:57).
- All three evaluators scored L=6–7. No GPU requirement identified anywhere.
- ANSUR-II anthropometric data is publicly available — no human annotation needed.
- Benchmark scenes can be procedurally generated (Empirical proposal).

**Evidence against:**
- No-humans kills the UIST venue and demand validation. Community Expert: "UIST dead under no-humans."
- Skeptic: demand validation gate D7 requires "structured feedback from ≥20 XR developers" (problem_statement.md:118–121). This is technically a human study component. If interpreted strictly, this gate cannot be passed under the constraint.
- Bug injection for benchmarks requires domain expertise to ensure realism (Red-Team Attack 4.2). Without real XR developers validating bug realism, the evaluation is self-referential.

**Challenge test:** Can the evaluation be credible without any human validation of the benchmark scenes or injected bugs? Procedurally generated scenes with procedurally injected bugs risk a fully circular evaluation. This is mitigable (use published accessibility guidelines as ground truth) but weakens the empirical story.

**Score rationale:** The computation is genuinely CPU-friendly. The constraint bites on evaluation credibility, not computational feasibility.

---

## Axis 5: FEASIBILITY — Score: 4/10

**Evidence for:**
- 2-month kill chain limits downside (Key Facts). D1 gate at Month 1, D2/D3 at Month 2.
- The synthesis identifies 10 concrete decisions with fallback paths for each (synthesis.md Part 4).
- Tier 1 linter alone is feasible (~12–15K LoC) and could produce a tool-track publication.
- Mathematician: P(any pub) ≈ 55–65% (the most optimistic estimate).

**Evidence against:**
- impl_loc=0, theory_bytes=0 as of current state (State.json:24–25). Nothing has been built.
- Synthesis own compound risk assessment: "~65% probability of at least one critical failure" (synthesis.md:519).
- Skeptic: P(any pub) ~40–50%, kill probability ~55%.
- Community Expert: P(any pub) ≈ 20–30% (most pessimistic).
- 8 binding amendments remain unimplemented from the Mathematician evaluation.
- The synthesis lists 6 unresolved issues (Part 3) that "cannot be resolved at the theory stage" — including whether frontier-resolution SMT works at all, whether piecewise-Lipschitz partitioning is tractable, and what the actual ε is on benchmarks. These are not minor details; they are load-bearing unknowns.
- Salvage to TACAS requires "domain-general certs, 20–30K LoC" (Skeptic). Salvage to ICSE Tool Track requires "linter, 12–18K LoC." Both are substantial implementations from a base of zero code.
- The ε target has been repeatedly weakened: original 0.01 → hard 0.05 → and even 0.05 has a 35% failure probability per the synthesis risk table.

**Challenge test:** Given impl_loc=0, 6 unresolved theoretical questions, and a 65% compound failure rate acknowledged by the project itself, what is the realistic probability of a completed publication? The evaluators range from 20–65%, averaging ~40%. This is consistent with the project's own risk assessment.

**Score rationale:** The kill-chain is well-designed and limits downside, but the project has zero implementation, multiple unresolved theoretical foundations, and the most optimistic evaluator gives only 55–65% chance of any publication. The feasibility score must reflect that ~60% of scenarios lead to no output.

---

## Composite Score: 22/50

| Axis | Score | Weight |
|------|-------|--------|
| Value | 3 | Equal |
| Difficulty | 5 | Equal |
| Best-Paper | 3 | Equal |
| Laptop-CPU | 7 | Equal |
| Feasibility | 4 | Equal |
| **Total** | **22** | |

---

## Strongest Aspect

**Decision 7 + κ-completeness framework.** The synthesis's Decision 7 — crediting Tier 1 affine-arithmetic envelopes as symbolically verified volume — is a genuine insight that emerged from the Red-Team process. Combined with the κ-completeness metric for honest certificate reporting, this creates a certificate framework that is *structurally* more informative than naive Monte Carlo, even when its ε headline number is worse. This is the one contribution that could anchor a publication regardless of whether frontier-resolution SMT works.

## Weakest Aspect

**Zero validated demand in an unregulated domain.** Three independent evaluations, the proposal itself, and the problem statement all acknowledge: there is no evidence that any XR developer wants this tool, no XR-specific regulation mandates it, and the addressable market intersection ("motor disability" ∩ "XR headset owner") is currently tens of thousands of people. The value proposition is entirely prospective. Without demand, even a technically excellent tool produces a paper that referees will question on motivation.

---

## Verdict: CONDITIONAL CONTINUE

**Confidence: 45%**

**Conditions (binding):**

1. **Gate D1 (Month 1) is the hard kill.** If wrapping factor > 10× on 7-joint chains after subdivision, ABANDON. No extensions.
2. **Reframe away from UIST immediately.** Target ISSTA 2027 or FSE 2027 Tool Track. The formal methods + testing framing is the only viable path under no-humans.
3. **Drop the 5× over Clopper-Pearson claim.** Lead with the structural certificate (spatial map, κ-completeness, counterexample generation). The ε headline will always look worse than naive MC to a casual reader. Don't make the paper fight that battle.
4. **Tier 1 linter is the insurance policy.** If Tier 2 certificates fail every gate, the Tier 1 linter at ICSE Tool Track (~12–15K LoC) is the salvage publication. Prioritize Tier 1 to completion before investing deeply in Tier 2.
5. **Demand validation (D7) must be reframed** as a lightweight survey/outreach exercise, not a formal human study. Under no-humans, this means mining public XR developer forums, GitHub issues, and accessibility bug reports — not conducting interviews.

**Rationale:** The project's own risk assessment gives 65% compound failure probability. Three evaluators average ~22/50 composite. The kill chain is well-designed and limits downside to 2 months. The Tier 1 linter + κ-completeness framework is a real, if modest, contribution to a genuinely empty tooling space. The conditional continue is justified by the bounded downside and the nonzero salvage paths — not by confidence in the primary research outcome.

**P(any publication) ≈ 35–50%**  
**P(best-paper) ≈ 1–2%**  
**Expected value of continuation ≈ marginally positive, assuming 2-month kill discipline is enforced.**

---
title: "FINAL Theory Gate Report — GuardPharma (proposal_00)"
slug: guideline-polypharmacy-verify
stage: theory_gate_verification
date: 2026-03-08
lead: Team Lead (Cross-Critique Synthesis & Dispute Resolution)
evaluations_synthesized: 6
  - skeptic_cross_critique: 5.2
  - mathematician: 4.9
  - community_expert: 4.3
  - agent0_auditor: 4.4
  - agent1_skeptic: 2.9
  - agent2_synthesizer: 5.8
final_composite: 4.2/10
verdict: CONDITIONAL CONTINUE
kill_probability: 35-45%
---

# FINAL Theory Gate Report — GuardPharma

## 0. Score Landscape

Six independent evaluations over two rounds. Sorted:

| # | Evaluator | Composite | Verdict |
|---|-----------|-----------|---------|
| 1 | Agent-1: Fail-Fast Skeptic | 2.9 | ABANDON |
| 2 | Community Expert | 4.3 | CONDITIONAL CONTINUE |
| 3 | Agent-0: Auditor | 4.4 | CONDITIONAL CONTINUE |
| 4 | Mathematician | 4.9 | CONDITIONAL CONTINUE |
| 5 | Skeptic Cross-Critique (Rd 1) | 5.2 | CONTINUE (barely) |
| 6 | Agent-2: Synthesizer | 5.8 | CONTINUE |

Median: 4.35. Mean: 4.42. Interquartile range: 3.65–5.05. The distribution is unimodal with Agent-1 as a low outlier and Agent-2 as a high outlier.

---

## 1. DISPUTE RESOLUTION

### Dispute A: ABANDON (2.9) vs CONDITIONAL CONTINUE (4.4) vs CONTINUE (5.8)

**The Skeptic's 2.9 is too harsh. The Synthesizer's 5.8 is too generous. The Auditor's 4.4 is approximately correct.**

Evidence:

1. **The Skeptic's score trajectory (5.2→4.9→4.3→2.9) reflects accumulating frustration, not accumulating evidence.** The project's *mathematical content* did not deteriorate between evaluations — Theorem 1's proof sketch, the PTA formalism, the two-tier architecture, and the M(c) gap discovery all existed before the Skeptic's 2.9 score. What changed was the Skeptic's patience with the planning fractal. This is legitimate frustration but not a scoring criterion. The Skeptic's own concession — that the M(c) gap discovery represents "genuine mathematical engagement" — contradicts a 2.9 composite. A project with genuine mathematical engagement, a novel formalism, and a publishable crown jewel theorem does not score below 3.0 unless it is fundamentally unsound. The Skeptic never establishes unsoundness.

2. **The Synthesizer's 5.8 is optimistic on three specific claims.** (a) The "layer cake" strategy (4 independently publishable layers) overstates independence — if the PTA formalism is weak, the contract theorem, the widening proposition, and the case study all weaken with it. P(zero publications) < 3% requires that PTA itself is publishable independent of all empirical results; this is plausible at HSCC but not assured. (b) The dual-polarity paper framing ("when composition works AND when it provably fails") is genuine insight but adds novelty points for content that doesn't exist yet (the three-body counterexample formalized as a theorem). (c) The HSCC venue pivot is the Synthesizer's strongest contribution — correctly identified as early as the Mathematician evaluation — but venue fit alone doesn't raise a 4.4 to 5.8.

3. **The Auditor's 4.4 is anchored by the four prior evaluations (4.3–5.2) and reflects the project's actual state:** real mathematical ideas, zero executed artifacts, one novel theorem with a known gap, 45–55% coverage of the clinically relevant problem, and a planning-to-execution ratio of ∞. This is a low-confidence-but-positive-EV project. 4.2–4.5 is the honest range.

**Resolution: 4.2.** Slightly below the Auditor's 4.4 because the Skeptic's value-inversion argument (Dispute B) is substantially correct and was underweighted by the Auditor.

### Dispute B: Value Inversion — Fatal vs Acknowledged vs Reframable

The Skeptic's formulation: *"Novel math covers the easy clinical problem. The hard clinical problem gets the generic solution."*

**The Skeptic is substantially correct. The Synthesizer's reframing is partially valid but does not eliminate the structural problem.**

Evidence:

1. **Coverage arithmetic.** Theorem 1 (contract-based composition) covers competitive CYP inhibition: ~65–70% of PK DDIs, but only **45–55% of clinically significant drug interactions** in the target polypharmacy population (Mathematician evaluation, §1.1). The gap: mechanism-based inhibition (~8%), active metabolites (~5%), autoinduction (~3%), and pharmacodynamic interactions (QT prolongation, serotonin syndrome, CNS depression: 30–40% of serious ADEs in elderly polypharmacy). The most dangerous interactions fall to monolithic BMC with uncharacterized convergence.

2. **The DrugBank overlap.** CYP competitive inhibition interactions are precisely the class that DrugBank, Lexicomp, and Micromedex already detect via O(1) lookup. The formal guarantee adds *temporal characterization* (when toxicity occurs, not just that it occurs) and *exhaustive trajectory coverage* (all patient paths, not just flagged pairs). These are real but incremental over existing tools.

3. **The Synthesizer's reframing.** "Boundary characterization — the paper proves WHERE composition works and WHERE it fails" is a genuine intellectual contribution. Formalizing the three-body counterexample as Theorem 1' (Pairwise Insufficiency) is a good idea. But this is a *theoretical* contribution — it strengthens the HSCC paper, not the clinical value proposition. The value inversion remains: the system's strongest tool solves the subset of the problem that needs it least.

**Resolution: The value inversion is a SERIOUS structural limitation, not a FATAL flaw.** It is unfixable within the current architecture but acceptable for formal-methods venues where the formalism is the contribution.

### Dispute C: theory_bytes=0 — Accurate vs Measurement Bug

**Both sides are partially correct. The fair assessment is: significant theory content exists, but zero publication-quality proofs exist.**

Evidence:

1. `theory/approach.json` is 39,562 bytes of structured formal content: theorem statements, 4 lemma sketches, 8 algorithm pseudocodes, complexity bounds, and evaluation plan. The State.json field `theory_bytes: 0` is a **pipeline measurement bug** — the byte counter measures `proposals/proposal_00/theory/`, not `theory/`.

2. However, the Skeptic's core point stands: **approach.json contains proof sketches, not proofs.** Theorem 1 has a known gap (M(c) state-dependence requiring cooperative systems theory). Proposition 2's convergence bound is asserted, not proven. No LaTeX, no epsilon-delta arguments, no publication-quality mathematics exist anywhere in the repository.

3. **Calibration:** 39KB of structured proof sketches is closer to "writing debt" (Synthesizer) than "zero theory" (Skeptic). A competent researcher could turn these sketches into complete proofs in 2–3 weeks — or could discover the M(c) gap is harder than expected.

**Resolution: SERIOUS (not FATAL, not MODERATE).** The theory content exists intellectually but not as executable artifacts. The M(c) gap must be closed before implementation begins.

### Dispute D: Best-Paper Potential — 2 vs 3 vs 5.5

**The Synthesizer's 5.5 is unsupported. The Skeptic's 2 is too pessimistic. The Auditor's 3 aligns with the Mathematician's calibrated estimates.**

Evidence:

1. **The Mathematician's probability table** is the most carefully calibrated: P(accept | HSCC) = 25–30%, P(best paper | HSCC) = 1.5–2.5%. P(accept | AIME) = 20–30%, P(best paper | AIME) = 0.8–1.5%. P(≥1 acceptance from dual submission) = 40–50%.

2. **The Synthesizer's layer-cake strategy** assumes 4 independently publishable layers. But independence fails: PTA formalism → contract theorem → widening result → case study is a dependency chain. P(zero publications) < 3% is overconfident; 8–12% is more realistic.

3. **The Skeptic's E1 concern is valid:** the temporal ablation showing X% of conflicts require temporal reasoning is a coin flip at 30–40% disappointment probability. If X < 15%, the AIME narrative collapses.

**Resolution: 3/10.** P(≥1 publication) ≈ 45–55%. P(best paper at any venue) ≈ 2–4%. A 3 reflects "publishable with effort, best-paper only with luck."

---

## 2. RESOLVED PILLAR SCORES

| Pillar | Score | Justification |
|--------|-------|---------------|
| **V: Extreme & Obvious Value** | **3/10** | Real problem, genuine LLM-proof moat, but zero demand signal, near-zero CQL ecosystem, and value inversion (strongest guarantees on easiest interactions). |
| **D: Genuine Software Difficulty** | **6/10** | Three-domain intersection creates real integration difficulty; ~35K novel LoC. But individual techniques all known, CQL compilation deferred, and the most interesting theorem out of scope. |
| **BP: Best-Paper Potential** | **3/10** | P(≥1 publication) ≈ 45–55%; P(best paper) ≈ 2–4%. E1 temporal ablation is high-variance, math adequate but not frontier, best theorem deferred. HSCC venue pivot is strongest path. |
| **L: Laptop-CPU & No-Humans** | **6/10** | Contract path clearly laptop-feasible. Monolithic BMC path (~50% of clinically significant interactions) has uncharacterized performance. E9 is not a constraint violation but should remain optional. |
| **F: Overall Feasibility** | **4/10** | Zero artifacts after theory stage. Planning-to-execution ratio is pathological. M(c) gap unresolved. Timeline tight. Partially offset by real intellectual content, well-designed pilot gate, and credible salvage floor. |

**Composite: (3 + 6 + 3 + 6 + 4) / 5 = 4.4, adjusted to 4.2** after weighting the value-inversion argument which cross-cuts both Value and Best-Paper.

---

## 3. FATAL FLAW REGISTRY

| ID | Flaw | Source | Final Severity | Notes |
|----|------|--------|----------------|-------|
| F1 | **Zero artifacts after theory stage** | Skeptic, Community Expert | **SERIOUS** | 500KB meta-evaluation, 0KB proofs/code. approach.json (39KB) provides intellectual foundation but VA4 unsatisfied. |
| F2 | **Value inversion: strongest guarantees on easiest interactions** | Skeptic (all rounds), Community Expert | **SERIOUS** | Competitive CYP inhibition = 45–55% of clinical DDIs. Most dangerous (QT, serotonin syndrome) fall to uncharacterized BMC. Structural; unfixable without new mathematics. |
| F3 | **Theorem 1 M(c) gap** | Mathematician, Adversarial Review | **SERIOUS** | Proof sketch assumes constant Metzler matrix; actual system is M(c)·c (nonlinear). Requires cooperative systems theory (Smith 1995). Likely fixable but unwritten. |
| F4 | **E1 temporal ablation coin-flip** | Skeptic, Mathematician | **SERIOUS** | 30–40% probability X < 15%, collapsing the AIME narrative. PK interactions being temporal ≠ guideline conflicts requiring temporal reasoning to detect. |
| F5 | **Corpus starvation** | Community Expert, Verification Report | **SERIOUS** | ~30–50 CQL treatment guidelines worldwide. Supplementing with manually encoded rules raises generalizability questions. |
| F6 | **Zero demand signal** | All evaluators | **MODERATE** | R7 at 80%+. Acceptable for FM venues; damaging for clinical venues. |
| F7 | **Planning fractal** | Skeptic | **MODERATE** | 4 rounds of evaluation, 0 artifacts. Process pathology, not intellectual one — but process pathologies kill projects. |
| F8 | **70% CYP coverage overclaim** | Mathematician | **MODERATE** | Should be "65–70% of PK DDIs; 45–55% of clinically significant interactions." Correctable. |
| F9 | **Proposition 2 precision problem** | Mathematician §1.2 | **MINOR** | PK-aware widening may produce vacuous intervals for CYP3A4-sharing drugs. |
| F10 | **Pacti framework not cited** | Adversarial Review | **MINOR** | Standard citation fix. |

---

## 4. AMENDMENTS REQUIRED

1. **Close the M(c) gap in Theorem 1.** Write the complete proof using cooperative systems theory (Smith 1995). Deliver as LaTeX with full quantifier structure. *Deadline: Gate 1.*

2. **Correct the coverage claim.** Replace "~70% of clinically significant DDIs" with "65–70% of PK DDIs; 45–55% of clinically significant drug interactions in the target polypharmacy population."

3. **Complete the Proposition 2 convergence proof.** Resolve whether the bound is O(D·k) or O(D·k²) for strongly coupled drugs.

4. **Distinguish eCQMs from treatment guidelines in the corpus description.** Remove conflated "300+ guideline artifacts" claim.

5. **Add the Pairwise Insufficiency result (Theorem 1').** Formalize the three-body counterexample: three drugs sharing CYP3A4, pairwise safe, triple unsafe. Strengthens the HSCC narrative at low proof cost.

6. **Cite Pacti (Incer et al. 2022)** and compare the CYP-enzyme interface contract to Pacti's general contract algebra.

7. **Make E9 (pharmacist review) optional.** Remove from mandatory evaluation; include as supplementary if budget permits.

8. **Add a Tier 1 precision analysis** bounding the false-positive rate for CYP3A4-sharing drug combinations under PK-aware widening.

9. **Verify PopPK parameter availability** for the 30 most common polypharmacy drugs. Map each to published PopPK source and flag drugs without adequate parameters.

10. **Write one complete PTA encoding** for the simplest guideline pair (ADA diabetes + ACC/AHA hypertension) as proof-of-concept.

---

## 5. BINDING CONDITIONS (Hard Kill Gates)

| Gate | Week | Criterion | Kill Action |
|------|------|-----------|-------------|
| **G0: PTA Encoding Feasibility** | 2 | One complete PTA encoding for ADA diabetes + ACC/AHA hypertension guideline pair, with at least 3 medication-initiation transitions and CYP-enzyme interaction. | If encoding fails or requires >1 week per guideline: ABANDON tool paper; redirect to pure PTA formalism theory paper at HSCC. |
| **G1: Theorem 1 Proof Complete** | 4 | Publication-quality proof of Theorem 1 including M(c) cooperative systems argument. Reviewed by at least one FM-literate reader. | If M(c) gap is intractable: DOWNSCOPE to constant-M locations only. If constant-M also fails: ABANDON. |
| **G2: Pilot Verification** | 8 | 3 guideline pairs encoded as PTA. Model checker terminates on all 3. At least 1 non-trivial temporal conflict found. | If 0 temporal conflicts found: E1 narrative dead. Redirect to HSCC formalism paper. Kill AIME submission. |
| **G3: E1 Minimum Threshold** | 14 | Temporal ablation on full corpus: X ≥ 15% of detected conflicts require temporal PK reasoning. | If X < 15%: kill AIME submission. Pivot to HSCC formalism + compositionality speedup paper. |
| **G4: Paper Draft Complete** | 18 | Full paper draft with all proofs, experiments run, tables populated. | If not complete by week 18: ship whatever exists to best-fit venue or release as technical report. |

---

## 6. PROBABILITY ESTIMATES

| Outcome | P(outcome) | Notes |
|---------|------------|-------|
| G0 passes (PTA encoding feasible) | 75–85% | Manual encoding is tedious but conceptually straightforward. |
| G1 passes (Theorem 1 proof complete) | 70–80% | M(c) gap likely fixable via cooperative systems theory. |
| G2 passes (pilot finds temporal conflict) | 60–70% | At least one timing interaction likely in diabetes+hypertension+CKD. |
| G3 passes (X ≥ 15%) | 55–65% | Skeptic's "temporal ≠ temporally-detectable" distinction is valid. |
| All gates pass | 25–35% | Product of gate probabilities. |
| ≥1 publication (any venue) | 45–55% | HSCC formalism paper as salvage floor. |
| Best paper at any venue | 2–4% | Requires all gates + strong E1 + clean execution. |
| Zero publications | 10–15% | Requires simultaneous proof + pilot + corpus failure. |
| Project ABANDON at a gate | 35–45% | Mostly at G2 (pilot) and G3 (E1 threshold). |

---

## 7. FINAL VERDICT

### CONDITIONAL CONTINUE — Composite 4.2/10

**Rationale:**

The project has real intellectual substance: a novel formalism (PTA), a genuine domain-specific insight (Metzler monotonicity enabling single-pass contract resolution), and a well-motivated clinical problem. P(≥1 publication) ≈ 45–55% represents positive expected value. The HSCC venue pivot provides a credible floor that prevents zero-paper catastrophe.

However, this is a **low-confidence bet** with structural weaknesses:

1. **The value inversion is real.** The system's strongest formal guarantees cover the interaction class most easily handled by existing tools. This is architectural DNA, not a fixable bug.
2. **Zero artifacts exist.** After extensive planning, no proofs, no code, and no pilot results have been produced. The project must now produce bytes, not plans.
3. **The E1 gamble is genuine.** The temporal ablation result is high-variance.
4. **The coverage gap is understated.** 45–55% of clinically significant DDIs, not 70%.

The verdict is CONDITIONAL CONTINUE because:
- P(≥1 publication) > 45% justifies continued investment
- The salvage floor (HSCC formalism paper) is credible
- Five hard kill gates ensure rapid termination if the bet goes wrong
- The alternative (ABANDON) discards genuine mathematical ideas with nonzero publication probability

**The project has 4 weeks to produce a complete Theorem 1 proof and one PTA encoding. If it produces another evaluation document instead, the Skeptic was right all along.**

---

## 8. DISSENT RECORD

### Agent-1 (Fail-Fast Skeptic) — ABANDON at 2.9/10

The Skeptic's dissent is recorded in full and treated with the respect it deserves. The Skeptic was the most honest evaluator across all six rounds.

**Arguments where the Skeptic was RIGHT:**

1. *Value inversion.* "Novel math covers the easy clinical problem. The hard clinical problem gets the generic solution." Adopted as finding F2 (SERIOUS).
2. *The 70% coverage overclaim.* Confirmed: 45–55% of clinically significant interactions. Adopted as Amendment 2.
3. *"Temporal ≠ temporally-detectable."* Core E1 risk. PK interactions being temporal does not mean conflicts require temporal reasoning to detect.
4. *The planning fractal.* 4 rounds of evaluation, 0 artifacts. 500KB of assessment, 0KB of proofs. Pathological.
5. *Score trajectory monotonically decreasing.* 5.2→4.9→4.3→2.9 represents real signal.

**Arguments where the Skeptic was OVERRULED:**

1. *2.9 composite.* Penalizes planning fractal as if it were mathematical unsoundness. A project with a genuine novel formalism, a publishable theorem, and P(≥1 publication) > 45% does not score below 3.0.
2. *"theory_bytes=0 is directionally correct."* Directionally yes, but erases 39KB of structured mathematical content. The difference between "no mathematical thought" and "thought not yet formalized" matters.
3. *P(at least one gate fails) ≈ 71%.* Correct arithmetic, wrong decision metric. P(at least one problem) is trivially high for any research project. The decision-relevant metric is EV, which is positive.
4. *ABANDON verdict.* Given P(≥1 publication) ≈ 45–55% and a credible salvage floor, ABANDON requires believing the expected value is negative. The Skeptic did not establish this.

**The Skeptic's final warning, endorsed by the Team Lead:**

> "The next document this project produces must be a proof, not a review of a review of a review. The planning fractal ends here."

---

*End of report. This document is the binding output of the GuardPharma theory verification gate. The five kill gates (G0–G4) are non-negotiable. The next artifact produced by this project must be a Theorem 1 proof or a PTA encoding — not another evaluation.*

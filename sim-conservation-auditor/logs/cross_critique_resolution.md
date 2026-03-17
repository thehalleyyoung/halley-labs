# Cross-Critique Resolution: ConservationLint (sim-conservation-auditor)

**Date:** 2026-03-08  
**Coordinator Verdict:** ABANDON with conditions (60/40 split)

---

## Disagreement 1: Value (Auditor 3 / Skeptic 3 / Synthesizer 6)

### Ruling: Auditor/Skeptic are stronger. Resolved score: **4**

**Why the Auditor/Skeptic win this round:**

The SLAM analogy is factually broken. SLAM shipped a working tool that verified millions of lines of production Windows code *before* anyone called it a paradigm. The causal direction was: working artifact → demonstrated capability → paradigm recognition. ConservationLint proposes the reverse: paradigm claim → aspiration → no artifact. The Herbie analogy is better but still overstated — Herbie had a working prototype at publication that people could download and run. ConservationLint has `theory_bytes=0` and `impl_loc=0`.

The Synthesizer's PIML connection (10-20× audience expansion) is speculative. The problem statement itself scopes the primary audience to "numerical methods researchers who implement structure-preserving integrators." The project's own `risk_assessment.probabilities.industrial_adoption` is 0.30. The 10K audience figure assumes a reframing that hasn't been tested on even one PIML researcher.

**However, the Synthesizer is not wrong in principle.** Paradigm-creating papers *do* get disproportionate citations, and "physics-aware program analysis" is a genuinely novel framing. The error is in pricing this at 6 when zero evidence of paradigm viability exists. Paradigm value should be treated as upside optionality (~15-20% probability), not base-case value.

**Factual errors:**
- Synthesizer's "10-20× audience expansion" has no supporting data in any project artifact.
- Auditor/Skeptic's "LLMs cover 70%" is plausible but also unsupported by evidence — LLMs cannot do causal localization or obstruction detection (which don't exist yet either).

**Unanswerable without data:** Whether JAX-MD/Dedalus developers would actually use the tool. The Auditor's challenge test (3/5 JAX-MD devs say "weekly use") is the correct experiment.

---

## Disagreement 2: Best-Paper Potential (Auditor 3 / Synthesizer 5.5)

### Ruling: Auditor is stronger. Resolved score: **3.5**

**Why the Auditor wins this round:**

The Auditor's arithmetic is correct and the Synthesizer doesn't refute it. Let's check the Synthesizer's implicit probability stack:

- P(Wan et al. bug rediscovery demo works) ≈ 0.25 (Synthesizer's own estimate)
- P(T2 yields non-trivial result even with restricted charges) ≈ 0.40 (generous, given C-grade self-assessment)
- P(PIML reframing resonates with reviewers) ≈ 0.30 (untested)
- P(all three) ≈ 0.03

A 3% best-paper scenario does not justify a 5.5 score. The Synthesizer is conflating "this *could* be amazing if everything works" with "this is likely to be amazing."

**However, the Synthesizer's T2 tractability paths deserve credit.** The approach.json confirms T2 is tractable for p≤3, k≤3. Restricting to linear/quadratic Noether charges is a legitimate narrowing strategy that the Auditor's 0.3 theorem-equivalents doesn't fully account for. This is the strongest point in the Synthesizer's case. But tractability for small parameters ≠ mathematical depth for best paper.

The project's own `risk_assessment.probabilities.best_paper_award` is **0.05**. Both experts should have anchored on this self-assessment.

**Factual errors:**
- Synthesizer's "obstruction detection = best paper at OOPSLA" requires T2 to produce a novel, non-trivial decidability result. The approach.json self-grades T2 as C and notes "exponential complexity limits practical scope."
- Auditor's 0.3 theorem-equivalents for T2 is arguably harsh — the obstruction criterion *is* novel even if complexity is poor. Adjusted: 0.4.

**Unanswerable without data:** Whether the restricted T2 (linear/quadratic charges, k=2, general p) produces an interesting theorem. This is purely mathematical — either it does or it doesn't.

---

## Disagreement 3: Feasibility (Auditor 3 / Synthesizer 6.5)

### Ruling: Synthesizer has the better *framework* but overprices the outcome. Resolved score: **4.5**

**Why this is the closest call:**

The Synthesizer's conditional probability argument is structurally correct. The Auditor's P decomposition (0.35 × 0.50 × 0.50 = 0.09 for top venue) *does* assume independence, and extraction success *would* raise downstream probabilities. If you can extract a NumPy integrator to IR and detect symmetries, you almost certainly have a localizable result. The corrected conditional calculation:

- P(extraction works on pure-NumPy) ≈ 0.50 (generous — no prototype exists)
- P(paper | extraction works) ≈ 0.55 (the Synthesizer's 0.60-0.70 is slightly high because T2 is still needed)
- P(any pub) ≈ 0.50 × 0.55 + 0.50 × 0.15 ≈ 0.35 (via top path) + 0.08 (via JOSS fallback) ≈ 0.43

This is far below the Synthesizer's 0.90. The Synthesizer's P(any pub) calculation implicitly assumes a JOSS benchmark-paper fallback is nearly certain, but a benchmark paper without a working tool is not a publication — it's a dataset.

**However, the Auditor underweights the kill-gate structure.** The project *does* have a Phase 1 design with explicit kill gates at Month 2 and Month 4. This genuinely bounds downside risk. You don't invest the full 90K LoC / 12 months if Phase 1 fails. The Auditor scores feasibility as if the project is committed to the full roadmap, but it isn't.

**Factual errors:**
- Synthesizer's P(any pub) ≈ 0.90 is not credible given theory_bytes=0 and impl_loc=0. Even with kill gates, the *starting position* is zero output.
- Auditor's P(top venue) ≈ 9% is roughly consistent with the project's own self-assessment of 25% *before* accounting for the zero-output current state. Adjusting: ~12-15% from here, conditional on Phase 1 completion.
- The project's approach.json claims `obstruction_criterion_decidability: "COMPLETE"` but the theory_bytes=0 contradicts this — the decidability argument exists in approach.json but not in a formal writeup.

**Unanswerable without data:** Whether jaxpr/Tree-sitter extraction is feasible for even toy integrators. This is the single gating experiment that determines everything downstream.

---

## CONSENSUS SCORE TABLE

| Dimension | Auditor | Skeptic | Synthesizer | **Resolved** | Rationale |
|-----------|---------|---------|-------------|-------------|-----------|
| **Value (V)** | 3 | 3 | 6 | **4** | Paradigm upside is real but unpriced without artifact; user base genuinely narrow today |
| **Difficulty (D)** | 5 | — | 6 | **5.5** | All agree the technical challenge is substantial; T2 complexity is real |
| **Best Paper (BP)** | 3 | — | 5.5 | **3.5** | Triple-stacked low-probability events; project self-assesses at 0.05 |
| **CPU (Contribution/Novelty)** | 7 | 5 | — | **6** | "Physics-aware program analysis" is genuinely novel as a concept; execution is zero |
| **Feasibility (F)** | 3 | 4 | 6.5 | **4.5** | Kill gates help; conditional probabilities are correlated; but starting from zero |

**Consensus Composite: 4.7/10**

*Weighting: V(25%) + D(15%) + BP(15%) + CPU(20%) + F(25%) = 0.25(4) + 0.15(5.5) + 0.15(3.5) + 0.20(6) + 0.25(4.5) = 1.0 + 0.825 + 0.525 + 1.2 + 1.125 = 4.675*

---

## CONSENSUS VERDICT

### **ABANDON** — 60% confidence

**Split: 60% ABANDON / 40% CONTINUE (conditional)**

The 40% CONTINUE path requires ALL of the following within **4 weeks** (not 6):

1. **Extraction prototype** — demonstrate Tree-sitter or jaxpr extraction of a Verlet integrator into symbolic IR with symmetry detection. This is pass/fail. (~2 weeks)
2. **T2 restricted proof** — prove the obstruction criterion for linear Noether charges at k=2, any p. Demonstrate it's not trivially equivalent to known decidability results. (~3 weeks, parallel with #1)
3. **One user validation** — show mock ConservationLint output to one JAX-MD or Dedalus developer and document their reaction. (~1 week, parallel with #1-2)

If all three succeed by week 4: re-evaluate at V=5, BP=4.5, F=6 → composite ~5.5, CONTINUE.  
If any one fails: ABANDON with no further review.

---

## The Single Most Important Unresolved Question

> **Can imperative NumPy/JAX integrator code be mechanically extracted into a symbolic IR that admits Lie-symmetry analysis?**

Everything downstream — symmetry detection, backward error analysis, causal localization, obstruction detection, the entire paper — depends on this one capability. It has never been demonstrated, prototyped, or tested. The project has been in theory stage for months with theory_bytes=0, meaning even the theoretical framework hasn't been written down formally. The extraction question is empirically answerable in ~2 weeks and should have been answered before any theorem work began.

---

## Recommended Next Action (regardless of verdict)

**Week 1 deliverable:** Write a 200-line Python script that takes a standard Verlet integrator (10-20 lines of NumPy) and extracts it into a SymPy expression tree representing the discrete map. Run SymPy's `infinitesimals()` on the result. If this produces the expected translation/rotation generators for a central-force problem, the project has legs. If it doesn't, you've saved months.

This is not the full extraction pipeline. It's the cheapest possible experiment that resolves the deepest uncertainty. Everything else — T2 proofs, venue selection, PIML framing — is premature optimization until this works.

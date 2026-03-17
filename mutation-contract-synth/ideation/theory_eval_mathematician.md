# Mathematician's Post-Theory Verification: MutSpec-Hybrid

| Field | Value |
|---|---|
| **Slug** | `mutation-contract-synth` |
| **Area** | 009 — Programming Languages and Formal Methods |
| **Date** | 2026-03-08 |
| **Evaluator** | Deep Mathematician (Verification Panel Lead) |
| **Panel** | Independent Auditor · Fail-Fast Skeptic · Scavenging Synthesizer |
| **Prior Composite** | 6.0/10 (depth check, CONDITIONAL CONTINUE 2-1) |
| **Post-Theory Composite** | **5.05/10** |
| **Verdict** | **CONDITIONAL CONTINUE (2-1, Skeptic dissents ABANDON)** |

---

## Executive Summary

The theory phase produced a 63KB mathematical blueprint of exceptional quality — 20 precisely stated theorems across three framings, honest achievability estimates, clean dependency graphs, and five impossibility results. It contains **zero completed proofs**. The crown jewel theorem (A3, ε-completeness of standard mutation operators for QF-LIA) remains at 65% achievability, unchanged from pre-theory. The depth check's five binding empirical gates — SyGuS feasibility, Gap Theorem validation, SpecFuzzer comparison, one real bug, and ground-truth verification — were all ignored. The theory phase responded to a demand for empirical evidence with more formalism.

The mutation-specification duality remains a genuinely novel insight unanimously confirmed by all three panelists. The question "what do your killed mutants mean as a specification?" is constructively unexploited in the PL literature. But an insight is not a theorem, a theorem statement is not a proof, and a proof plan is not a proof. After three phases (ideation, depth check, theory), the project has produced excellent planning documents and zero results.

**Verdict: CONDITIONAL CONTINUE with hardened, calendar-gated empirical conditions. This is the project's final conditional continue.** If binding gates are not met within 8 weeks, the project is terminated and salvage value is captured via a 4-page FormaliSE note + research agenda document.

---

## Panel Composition and Process

Three evaluators worked independently, then cross-critiqued each other's findings. The Lead synthesized the strongest elements from all three.

| Role | Composite | Verdict | Key Thesis |
|------|-----------|---------|------------|
| **Independent Auditor** | 5.2/10 | CONDITIONAL CONTINUE | "Theory spec is good. Time to find out if any of it works." |
| **Fail-Fast Skeptic** | 3.6/10 | ABANDON | "Zero proofs after 'complete' theory phase. Previous Skeptic was right." |
| **Scavenging Synthesizer** | 5.8/10 | CONDITIONAL CONTINUE | "Real idea, strong blueprint, zero results. ICSE tool paper is best path." |

---

## The `theory_bytes=0` Question

State.json reports `theory_bytes: 0`. The `proposals/proposal_00/theory/` directory is literally empty. However, `theory/math_spec.md` (31KB), `theory/approach.json` (13KB), and `theory/prior_art_audit.md` (19KB) exist at the project root — 63KB total.

**Resolution:** Both a bookkeeping bug (content exists in wrong directory) AND substantively accurate (no completed proofs exist anywhere). The Skeptic's deeper point is correct regardless: math_spec.md is a *specification of desired theorems*, not a document containing proofs. The word "proof" appears 6 times in 31KB — none introducing an actual proof. No QED, no ∎, no "by induction," no "assume for contradiction." The estimated formalization effort is self-assessed at "6–9 months for a strong PhD student."

The theory phase produced a **research plan for mathematics, not mathematics itself.**

---

## Math Quality Assessment

### Theorem-by-Theorem Audit

#### Genuinely New + Load-Bearing + Hard: **1 result**

| ID | Result | Status | Why it qualifies |
|----|--------|--------|-----------------|
| **A3** | ε-Completeness of {AOR, ROR, LCR, UOI} for QF-LIA | **UNPROVED** (65% achievability) | First formal bridge between mutation adequacy and specification strength. Requires non-trivial case analysis. Multi-site interactions are an open question. |

#### Genuinely New + Load-Bearing + ≥Moderate: **3 results**

| ID | Result | Status | Note |
|----|--------|--------|------|
| **A3** | ε-Completeness | Unproved, 0.65 | Crown jewel |
| **T3** | WP-Composition Completeness | Open question, 0.40–0.65 | Publishable either direction |
| **T4** | Lattice-Walk Termination/Irredundancy | Unproved, 0.80 | New algorithm, moderate proof |

#### New Contributions of Any Difficulty: **5 results**

Add A4 (Gap Theorem framing) and B4 (WP Differencing formulation) to the above three.

#### Definitions Presented as Theorems: **~10 results**

The following are self-described by the project as "straightforward," "standard," "known," or "textbook," yet presented with theorem numbers: A1, A2, A5, A6, B1, B2, B5, C2, C3, C4, C6. The final_approach.md already concedes that A1, A2, and A4 are "Definitions and Framework — they are load-bearing as formalism but not mathematically deep."

#### Ornamental (nice-to-have, not driving the system): **4 results**

A5 (Craig Interpolation for Minimality), A6 (Galois Connection), B6 (CE Recycling), B7 (Complexity Bounds). All three panelists agree A5 and A6 are not load-bearing. The depth check panel unanimously flagged A6 as "cosmetic."

### Net Assessment

The math_spec.md's 31KB reflects **thoroughness of exposition, not depth of novelty**. The mathematical content reduces to:

- **1 genuine open problem** (A3) that would be a real contribution if proved
- **1 interesting open question** (T3) publishable either way
- **1 new algorithm theorem** (T4) at moderate difficulty
- **2 modest contributions** (A4 framing, B4 formulation)
- **~10 straightforward applications** of known techniques to a new setting
- **~4 ornamental results** that don't drive the system

This is a **thin but real** mathematical portfolio. It is enough for a FormaliSE paper, probably enough for PLDI *if combined with strong empirical results*, but not enough for POPL without A3 and A6 both fully developed.

---

## Is the Math Load-Bearing or Ornamental?

This is the central question for a mathematician evaluator. The three panelists disagreed, and the resolution matters.

### The Consultant Test

The project itself admits: "A competent consultant with PIT + Z3 + Daikon + Python can replicate approximately 70–80% of the practical output in 2–3 months." This is from final_approach.md §2, not from the Skeptic — the project acknowledges that the engineering output is largely replicable without the theory.

### What the Remaining 20–30% Requires

1. **Theorem A3** — the guarantee that mutation-adequate tests determine the *strongest* QF-LIA spec. Without A3, the lattice-walk produces "good" contracts with no optimality guarantee.
2. **Lattice-walk algorithm** — deterministic, bounded-complexity synthesis with mutation provenance tracing. Implementable without theorems, but the optimality proof (T4) gives formal guarantees.
3. **WP-Composition Completeness** (T3) — determines whether the lattice-walk is as powerful as SyGuS. An implementation strategy question that happens to be mathematically interesting.

### Verdict: Primarily Explanatory, Conditionally Enabling

The math is **primarily explanatory**: it provides a vocabulary (error predicates, discrimination lattice, mutation-induced specification), a justification (if A3 were proven), and a framework (lattice theory, Galois connections) for understanding why the tool works. None of this changes what the tool *does*.

A3, **if proven**, would be **enabling in a specific sense**: it would upgrade the guarantee from "heuristic contracts" to "provably strongest contracts" for the QF-LIA loop-free fragment. This is the difference between Daikon ("probably finds good invariants") and a verified synthesis tool ("provably finds optimal invariants"). This gap matters for a PLDI paper; it does not matter for a practitioner.

**For practitioners: the math is decoration.**
**For PLDI reviewers: A3 (proven) transforms the paper from tool paper to theory paper.**
**The math does not make the tool possible. It makes the tool publishable at a top venue.**

---

## Does A3 Deserve "Crown Jewel" Status?

### The Single-Site Case

For single-site QF-LIA predicates × {AOR, ROR, LCR, UOI}, the proof is likely a mechanical 2–3 page case analysis: enumerate QF-LIA atomic predicates (linear inequalities), enumerate mutation operator effects, show coverage. This is a week of work for a competent grad student. **Not deep mathematics.**

### The Multi-Site Case

This is where genuine difficulty lives. Can first-order mutants (single-site changes) witness violations of multi-site conjunctive QF-LIA properties? The approach.json flags this: "multi-site property interactions: a QF-LIA property that requires simultaneous changes at two sites to violate may not be witnessed by any single first-order mutant."

- **If the theorem requires a "site-independence premise"** (each conjunct involves only one site): the result is close to circular — you're assuming decomposability and proving that decomposable properties are covered. **Valid but weak.**
- **If first-order mutants suffice WITHOUT site-independence**: this is a genuinely surprising result connecting to the coupling effect hypothesis. **This would be deep.**

### Verdict

A3 deserves crown jewel status **within this project** — it is clearly the hardest and most novel result. It does NOT automatically deserve crown jewel status **by PLDI standards**. Whether it reaches that bar depends entirely on what the multi-site case reveals. We cannot evaluate this because the proof doesn't exist.

---

## THREE PILLARS Scoring

### Pillar 1: Extreme Value — **5/10**

| Agent | Score |
|-------|-------|
| Auditor | 5 |
| Skeptic | 3 |
| Synthesizer | 6 |
| **Lead** | **5** |

The bug-finding reframe (Gap Theorem) targets a real niche: teams running PIT who currently extract only a mutation score. The marginal-cost argument is sound. But the value is entirely hypothetical — zero bugs found, zero users consulted, zero empirical evidence of any kind. The 70–80% consultant-replicable admission caps the math-attributed value delta at 20–30%, and that delta rests on unproven A3.

The math does not *create* the value. The engineering artifact (PIT + WP differencing + gap reports) creates the value. The math provides formal warrant that elevates the narrative. This is fine for a paper but limits the "extreme value" claim.

### Pillar 2: Genuine Software Difficulty — **5/10**

| Agent | Score |
|-------|-------|
| Auditor | 5 |
| Skeptic | 4 |
| Synthesizer | 6 |
| **Lead** | **5** |

One genuinely hard open problem (A3) + significant but standard engineering (PIT bridge at risk 8/10, WP engine at 8K LoC, Z3 integration). The lattice-walk algorithm is a clean design but not paradigm-shifting. ~16K of 48K LoC is novel. The difficulty is dominated by engineering, not math. The Skeptic's observation that "the novel core shrank when SyGuS was dropped" is accurate — the lattice-walk is simpler than SyGuS-based synthesis, which is good for feasibility but reduces the difficulty claim.

**Is the math the reason this is hard to build?** No. The math is the reason this is hard to *publish at PLDI*. The engineering (PIT symbolic bridge, WP engine, equivalent mutant filtering) is the reason it's hard to *build*.

### Pillar 3: Best-Paper Potential — **4/10**

| Agent | Score |
|-------|-------|
| Auditor | 4 |
| Skeptic | 2 |
| Synthesizer | 5 |
| **Lead** | **4** |

The three-legged stool (completeness theorem + Gap Theorem + real bugs) is the right paper structure, but all three legs are currently wobbly:

- **Leg 1 (Completeness):** Unproved theorem at 65% achievability, restricted to QF-LIA/loop-free covering 5–15% of real functions. The depth check's devastating challenge stands: "Show one PLDI best paper with main theorem restricted to loop-free code."
- **Leg 2 (Gap Theorem):** Clean formalism, but the equivalent mutant FP problem (8–12% after filtering) is unvalidated.
- **Leg 3 (Real bugs):** Zero bugs found. The entire empirical story is projected.

The degradation bound — identified as "crucial" at depth check — was acknowledged as "almost certainly false as stated," a regression. WP-Composition Completeness (T3) at 40% full achievability likely resolves negatively for the general case.

**P(best paper at PLDI or OOPSLA) ≈ 2–4%.** The "mutation determines specification" headline IS memorable if the math holds, but the scope restriction and zero empirical evidence are severe liabilities.

### Pillar 4: Laptop-CPU Feasibility — **8/10**

| Agent | Score |
|-------|-------|
| Auditor | 8 |
| Skeptic | 7 |
| Synthesizer | 8 |
| **Lead** | **8** |

Unanimous agreement within ±1. Pipeline is entirely CPU-bound: PIT (JVM), Z3 (CPU), no GPUs, no API calls, no human-in-loop. Three-tier degradation provides graceful fallback. The shift from SyGuS to lattice-walk removed the worst scalability risk (CVC5 timeout at >15 atoms). The only concern is Z3 performance on large WP formulas, solvable via incremental solving and per-function timeouts.

### Pillar 5: Feasibility — **4/10**

| Agent | Score |
|-------|-------|
| Auditor | 5 |
| Skeptic | 3 |
| Synthesizer | 5 |
| **Lead** | **4** |

Three phases completed, zero empirical contact. The project has demonstrated excellent *planning* ability but zero *execution* ability. A3's 35% failure probability means the intellectual core has a coin-flip chance of collapse. The depth check binding gates were ignored. PIT→symbolic bridge (risk 8/10) hasn't been prototyped. The phased plan with escape hatches is well-designed, but nothing past the blueprint has been attempted.

The Skeptic's question is pointed: "If a theory phase cannot increase confidence in the main theorem, what did it accomplish?" The answer is "better planning" — but better planning for a task with 35% failure probability on the core theorem and zero empirical contact is worth less than starting the actual work.

---

## Fatal Flaws

### CRITICAL

**F1: Crown Jewel Theorem (A3) Unproved at 65% Achievability**
The single load-bearing hard theorem has not been proved. The theory phase elaborated its proof structure but did not reduce the failure probability. If A3 fails, the project becomes "Daikon + PIT with a lattice-walk" — a tool paper without theoretical distinction.

**F2: Zero Empirical Validation After Three Phases**
No bugs found. No functions analyzed. No SpecFuzzer comparison. No binding gates from depth check passed. The CONDITIONAL CONTINUE from depth check was explicitly conditioned on a "4-week empirical validation phase." The theory phase produced 63KB of formalism instead. The conditional was not met.

**F3: Depth Check Binding Conditions Ignored**
This is a process failure. The prior verdict demanded empirical gates; the next phase delivered more theory. A second CONDITIONAL CONTINUE must include a harder enforcement mechanism. If the next phase again produces documents instead of results, ABANDON is automatic.

### SERIOUS

**F4: Theorem Scope Restricted to Loop-Free QF-LIA (5–15% of Real Functions)**
A3 covers only loop-free programs over QF-LIA with first-order mutants and four operators. The depth check required attempted extension to bounded loops (BEST-3). Not attempted. This restriction severely limits both best-paper potential and practical impact.

**F5: Degradation Bound Acknowledged as Likely False**
The original conjecture (g(s) ≥ s) is "almost certainly false as stated." Only a weaker reformulation under distributional uniformity is proposed, and it may be intractable. Without a degradation bound, A3 requires 100% mutation adequacy — a condition that doesn't exist in practice.

**F6: SpecFuzzer Gap Narrowing**
The pivot from SyGuS to lattice-walk narrowed the conceptual gap with SpecFuzzer. Original pitch: "synthesis vs. filtering (fundamental)." Current pitch: "deterministic WP-based construction vs. fuzzing-based enumeration (algorithmic)." The 2,000-line "SpecFuzzer + PIT + Z3" baseline is a legitimate threat that remains untested.

### MODERATE

**F7: Math Inflation (18 "Theorems," ~10 Are Definitions or Known Results)**
The math_spec.md self-describes most content as "straightforward," "standard," "known," or "textbook." Presenting ~10 standard results as numbered theorems creates an illusion of depth beyond the actual substance. The honest theorem count is ~3 (A3, T3, T4), of which 0 are proved.

---

## Novel Math After Honest Filtering

Applying the strictest filter — genuinely new math that is load-bearing and drives the system:

| # | Result | Difficulty | Proved? | What It Drives |
|---|--------|-----------|---------|---------------|
| 1 | A3: ε-Completeness for QF-LIA | **Hard** | No (0.65) | Optimality guarantee for contracts |
| 2 | T3: WP-Composition Completeness | **Hard** (open question) | No (0.40–0.65) | Whether lattice-walk = SyGuS power |
| 3 | T4: Lattice-Walk Termination/Irredundancy | **Moderate** | No (0.80) | Algorithm guarantees |

Everything else is either definitional scaffolding, known technique applied to a new setting, or ornamental.

**Load-bearing math that is genuinely hard and new: 1 theorem (A3) + 1 open question (T3).**

This is the honest content of the mathematical contribution. Everything else in math_spec.md is well-organized scaffolding that a competent grad student could write in 2–3 weeks given the definitions. The depth is in A3 and T3. If both fail, the math reduces to framework definitions plus a clean algorithm (T4).

---

## Publication Path Analysis

| Path | P(accept) | Impact (1–10) | E[value] | Conditions |
|------|-----------|---------------|----------|------------|
| **A: PLDI/OOPSLA** | 0.15–0.18 | 9 | ~1.5 | A3 proved, tool works, ≥10 bugs, beats SpecFuzzer |
| **B: ICSE/FSE tool** | 0.30–0.35 | 7 | **~2.3** | Tool works, ≥5 bugs, competitive baselines |
| **C: FormaliSE/VMCAI theory** | 0.45–0.55 | 4 | ~2.1 | A1+A2+A3+A4 formalized, no tool needed |
| **D: Workshop/NIER** | 0.70–0.75 | 2 | ~1.5 | A1+A4 written, tiny prototype |

**Path B (ICSE/FSE tool paper) maximizes expected value.** Recommended as primary target with Path C (FormaliSE theory) as insurance. Path A (PLDI) only if A3 proves cleanly with multi-site depth.

**P(any publication) ≈ 60–70%.** The floor is a 4-page FormaliSE note (P ≈ 70%). Total publication failure probability: 30–35%.

**P(best paper at any venue) ≈ 2–4%.** Requires A3 proved with depth + tool works + real bugs — a triple conjunction at low probability.

---

## Composite Score

| Pillar | Score | Weight | Weighted |
|--------|-------|--------|----------|
| Extreme Value | 5/10 | 25% | 1.25 |
| Genuine Software Difficulty | 5/10 | 20% | 1.00 |
| Best-Paper Potential | 4/10 | 25% | 1.00 |
| Laptop-CPU Feasibility | 8/10 | 15% | 1.20 |
| Feasibility | 4/10 | 15% | 0.60 |
| **Composite** | | **100%** | **5.05/10** |

### Score Reconciliation

| Pillar | Auditor | Skeptic | Synthesizer | Lead |
|--------|---------|---------|-------------|------|
| Value | 5 | 3 | 6 | **5** |
| Difficulty | 5 | 4 | 6 | **5** |
| Best-Paper | 4 | 2 | 5 | **4** |
| Laptop CPU | 8 | 7 | 8 | **8** |
| Feasibility | 5 | 3 | 5 | **4** |
| **Composite** | **5.2** | **3.6** | **5.8** | **5.05** |
| **Verdict** | COND. CONTINUE | ABANDON | COND. CONTINUE | **COND. CONTINUE** |

---

## VERDICT: CONDITIONAL CONTINUE (2-1)

**Composite: 5.05/10 (down from 6.0 at depth check)**
**Confidence: 70%**
**Skeptic dissents: ABANDON at 3.6/10**

### Decision Rationale

The mutation-specification duality is a genuine, novel insight confirmed by all three evaluators — abandoning before any empirical test would waste a real idea. However, three phases of excellent planning with zero execution evidence is a pattern that must break NOW: the next phase must produce proofs and code, not more documents. This is the project's final CONDITIONAL CONTINUE — if the binding gates are not met, the idea's salvage value (FormaliSE note + research agenda) is captured and the full project is terminated.

### Binding Conditions (NON-NEGOTIABLE, Calendar-Gated)

Failure on ANY gate triggers automatic ABANDON of the full project (salvage path activates).

**Gate 1 (Week 2): A3 Single-Site Proof**
Complete the case analysis for single-site QF-LIA predicates × {AOR, ROR, LCR, UOI}. This is the "mechanical case analysis" everyone agrees is achievable. If this fails, A3 is dead and the project pivots to Path B with A3 as conjecture.

**Gate 2 (Week 4): Prototype on 10 Functions**
PIT integration + WP differencing + basic gap analysis on 10 Apache Commons Math functions. Must produce ≥5 non-trivial gap reports. If zero interesting gaps: approach is empirically dead.

**Gate 3 (Week 6): SpecFuzzer + Z3 Baseline Comparison**
Run the baseline on the same 10 functions. If MutSpec doesn't beat it on at least one meaningful dimension: ABANDON full project.

**Gate 4 (Week 8): One Real Bug**
Find one confirmed bug in a maintained library that Daikon and SpotBugs miss. Minimum empirical evidence for any venue above workshop.

### If CONDITIONS Not Met: Salvage Plan

1. **4-page FormaliSE/ICSE-NIER note** (P ≈ 70%): Duality formalized, A3 as conjecture, A1+A2+A4 proved, lattice-walk described. Establishes priority. ~6 weeks, 1 person.
2. **A3 as standalone theory paper** at VMCAI/FormaliSE (P ≈ 45%): Even partial result or well-characterized negative result is publishable.
3. **Release math_spec.md as research agenda**: The blueprint is genuinely excellent even if the project dies.

### What the Previous Skeptic Got Right

The depth check Skeptic scored 3.5/10 and voted ABANDON. Their core arguments — nobody wants mutation-derived contracts, Theorem 3 covers 10–20% of functions, SpecFuzzer + Z3 might match MutSpec, show one PLDI best paper restricted to loop-free code — remain unanswered after the theory phase. The Lead's original rebuttal ("the Skeptic's standards would have rejected Daikon and SpotBugs") was weakened by the Skeptic's counter: "Daikon and SpotBugs had working software. MutSpec has 31KB of unproven theorem statements." The current Skeptic scores 3.6/10. The 0.1 increase is for the quality of the math_spec.md research plan.

The Skeptic's conditions for changing their verdict — any ONE of (A3 proved, SpecFuzzer comparison won by ≥20%, one real bug found) — are exactly the right empirical gates. None currently exists. All are achievable in 4–8 weeks and haven't been attempted.

### What the Synthesizer Found Worth Saving

Even at worst case (full ABANDON), the project produces:
1. A publishable research insight (mutation-specification duality, constructively unexploited)
2. A precise mathematical roadmap (math_spec.md) for the PL/SE community
3. A clean formulation of the Gap Theorem usable by anyone building mutation tools
4. An open question (T3: WP-Composition Completeness) worth investigating
5. The lattice-walk algorithm as a reusable synthesis paradigm

The A6 Galois Connection (mutation testing → abstract interpretation) is a dropped thread with the highest ceiling in the project — if developed, it could yield a standalone paper connecting Cousot's framework to mutation-based specification inference. This is noted as a long-term opportunity regardless of the project's fate.

---

## Appendix: Points of Unanimous Agreement

All three evaluators agree on:

1. A3 (ε-completeness) is the ONLY genuinely new, hard, load-bearing result
2. Zero completed proofs exist in any theory artifact
3. The mutation-specification duality insight is genuinely novel
4. Zero empirical validation exists (no bugs, no functions analyzed, no comparisons)
5. The depth check binding conditions were not addressed
6. A5 and A6 are ornamental (not load-bearing)
7. The scope restriction (loop-free QF-LIA) covers 5–20% of real functions
8. Laptop-CPU feasibility is a genuine strength (7–8/10)
9. The degradation bound weakened during theory development
10. The SpecFuzzer overlap remains unresolved

These findings carry the highest confidence in this evaluation.

---

*Signed: Verification Panel (Mathematician Lead)*
*Panel: Independent Auditor (5.2/10) · Fail-Fast Skeptic (3.6/10) · Scavenging Synthesizer (5.8/10)*
*Lead Composite: 5.05/10 · CONDITIONAL CONTINUE (2-1) with 4 hard-gated empirical deadlines*

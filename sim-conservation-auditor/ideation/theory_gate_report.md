# Verification Gate Report: sim-conservation-auditor (ConservationLint)

**Stage:** Post-theory verification gate  
**Date:** 2026-03-08  
**Proposal:** proposal_00 — ConservationLint: Noether-Theoretic Program Analysis for Scientific Computing  
**Team:** Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer  
**Process:** Independent proposals → adversarial cross-critiques → synthesis → signoff

---

## Verdict: ABANDON

**Composite: 3.7/10 (V3.5 / D5 / BP2 / L6 / F2)**

Vote: 2-1 ABANDON (Auditor, Skeptic) vs. CONDITIONAL CONTINUE (Synthesizer). Cross-critique resolved in favor of ABANDON with structured salvage protocol.

---

## Executive Summary

ConservationLint proposes bridging Noether's theorem and program analysis to detect, localize, and classify conservation-law violations in Python simulation code. The paradigm — "physics-aware program analysis" — is genuinely novel and unanimously confirmed as unexplored territory. However, after completing the theory stage:

- **theory_bytes = 0** — zero bytes of mathematical content produced
- **impl_loc = 0** — zero lines of code
- **monograph_bytes = 0** — no paper draft
- **theory_score = null** — no evaluation possible
- **Prior binding conditions: 3/4 unmet** (coverage measurement, venue decision, benchmark validation)

The theory stage's sole purpose was mathematical production. It produced none. The proposal's crown jewel — T2, the computable obstruction criterion — has no proof, no sketch, and no worked example. The compound probability of clearing all five self-defined kill gates is ~13-16%. The project has consumed five pipeline stages producing 562KB of planning documents and zero tangible artifacts.

---

## Team Evaluations

### Independent Auditor — Composite 3.7/10, ABANDON

| Axis | Score | Key Evidence |
|------|-------|-------------|
| Value | 3 | TAM ~300-500; conservation pain point #6; LLMs cover 70%; liftable fragment excludes primary bug pattern |
| Difficulty | 5 | ~30% novel engineering; T2 unsolved; integration of 5 stages is the real challenge |
| Best-Paper | 2 | theory_bytes=0; ~0.55 novel theorem-equivalents; T2 self-graded C |
| Laptop/CPU | 6 | Core symbolic algebra is laptop-tractable; phase annotations are modest UX |
| Feasibility | 2.5 | Compound gate failure ~87%; 3/4 prior conditions unmet; zero artifacts |

The Auditor rejected CONDITIONAL CONTINUE verdicts on procedural grounds: "The 4-week sprint conditions replicate a pattern that has already failed. Granting a sixth chance based on promises from a team with a demonstrated 0% delivery rate in the theory stage is not evidence-based; it is hope-based."

### Fail-Fast Skeptic — Composite 2.8/10, ABANDON

| Axis | Score | Key Evidence |
|------|-------|-------------|
| Value | 3 | Demand fabricated from single anecdote; zero user validation; unique margin ≤6% |
| Difficulty | 4 | Real difficulty is extraction (engineering), not math; T2 graded C |
| Best-Paper | 2 | Zero artifacts makes best-paper discussion "absurd" |
| Laptop/CPU | 5 | Plausible for k≤3,p≤3; collapses for general case; no benchmarks |
| Feasibility | 1 | Planning-to-artifact ratio is infinite; conditions are the project itself |

The Skeptic identified five fatal flaws: (1) theory stage produced nothing, (2) T2 crown jewel may not exist, (3) liftable fragment excludes the primary use case, (4) zero demand validation, (5) planning-to-artifact ratio is infinite. The Skeptic characterized CONDITIONAL CONTINUE as "delayed ABANDON" with circular conditions.

### Scavenging Synthesizer — Composite 4.6/10, CONDITIONAL CONTINUE

| Axis | Score | Key Evidence |
|------|-------|-------------|
| Value | 5 | Paradigm novelty + obstruction uniqueness + PIML verification connection |
| Difficulty | 5 | Integration of 5 stages is hard; individual components are known |
| Best-Paper | 3 | Bridge narrative compelling but zero-artifact state caps potential |
| Laptop/CPU | 7 | Genuine structural strength: symbolic/algebraic throughout |
| Feasibility | 3 | Zero artifacts; P(ABANDON) ~50%; but bounded investment + high salvage floor |

The Synthesizer argued EV(CONTINUE via 4-week sprint) exceeds EV(ABANDON) by +1.65 units, with expected ~2.15 publications across all scenarios. The benchmark suite should be built unconditionally regardless of verdict.

---

## Adversarial Cross-Critique: Key Resolutions

### Resolved in Favor of Auditor/Skeptic

1. **V=3.5 (not 5).** Paradigm novelty is acknowledged (+0.5 above Auditor/Skeptic consensus of 3), but undemonstrated paradigms with TAM 300-500 and 70% LLM overlap cannot reach V=5. Paradigm credit requires demonstration, not aspiration.

2. **F=2 (not 1 or 2.5 or 3).** F=1 is definitionally too strong — the problem IS solvable in principle. But theory_bytes=0 is a realized execution failure that prevents F>2. Salvage paths don't inflate THIS proposal's feasibility (Skeptic's successful challenge: "Scoring ConservationLint's feasibility at 2.5 because a benchmark suite is feasible is like scoring a Mars mission at 3/10 because you could build a weather balloon").

3. **BP=2 (not 3).** Path to best-paper exists (T2 + paradigm) but requires everything to work. P(best paper) < 2%. Zero artifacts after a dedicated theory stage makes BP>2 unsupported.

4. **CONDITIONAL CONTINUE is not justified.** The Synthesizer's EV argument survives in sign but the +1.65 headline shrinks to +0.3 to +0.7 after correcting for outcome dependence (Skeptic's successful challenge on independence assumptions). This delta is insufficient to override behavioral evidence from theory_bytes=0.

5. **"Diagnostic sprint" understates what's being asked.** The Skeptic's reframe is correct: C1 (prove T2) and C2a (extract one integrator) are the core research contributions, not peripheral diagnostics. An honest framing: "Attempt the hardest part of the research first, on one example, in 4 weeks."

### Resolved in Favor of Synthesizer

1. **D=5 (not 4).** Four-evaluator consensus holds. Integration difficulty is genuine regardless of whether the hard parts are mathematical or engineering. The Skeptic's D=4 conflates mathematical thinness (BP concern) with problem difficulty.

2. **L=6 (not 5).** Mathematical complexity bounds ARE evidence (not speculation). Core computations have proven polynomial bounds for stated parameters. Lack of benchmarks prevents L=7 but does not justify L=5.

3. **The benchmark suite has standalone value** and should be pursued unconditionally regardless of verdict.

4. **The gating questions are cheaply answerable.** Extraction viability is testable in ~200 lines of Python in 2 weeks. This is acknowledged as a valid salvage investigation (not continuation of the full proposal).

### Points of Unanimous Agreement

1. The Noether → program analysis paradigm is genuinely novel
2. theory_bytes=0 is a critical failure
3. Code→math extraction is the existential risk
4. T2 is the only genuinely mathematical contribution (~0.3 novel theorem-equivalents)
5. Coverage claims (40-60%) are fabricated — no measurement exists
6. LLMs cover ~70% of simple conservation diagnostics
7. The benchmark suite has standalone community value
8. D=5 is the correct difficulty score
9. P(best paper) < 2%

---

## Fatal Flaws (Confirmed Through Cross-Critique)

### FF1: Theory Stage Produced Zero Output (TERMINAL)

The theory stage's sole purpose was mathematical content. It produced `theory_bytes=0`. Not low-quality theory — literally zero bytes. The `approach.json` (34KB) contains theorem *statements* and proof *strategies*, not proofs. T2 (the crown jewel) exists only as an aspiration. This is not a partial failure; it is a complete failure of the stage's objective.

### FF2: Crown Jewel (T2) Is C-Grade and May Be Trivial (TERMINAL)

T2 is self-graded C (lowest grade in approach.json). It is EXPSPACE-complete in general. For tractable k≤3, p≤3: ≤27 Lie bracket checks — a brute-force finite computation. The "efficient structured reduction" that would elevate T2 to a genuine theorem is unproven with zero evidence of tractability. If the efficient reduction fails, the fallback is a lookup table, not a theorem.

### FF3: Liftable Fragment Excludes Primary Bug Pattern (SERIOUS)

Data-dependent branching (`if r < r_cut`) — the most common MD conservation-bug source — is explicitly excluded from Tier 1 analysis. The tool cannot formally analyze the very bugs its users most need to find.

### FF4: Zero Demand Validation After Five Stages (SERIOUS)

No user interviews. No survey data. No prototype deployment. The TAM (300-500) is a back-of-envelope calculation. Five pipeline stages have passed without a single conversation with a target user.

### FF5: Compound Gate Failure ~84-87% (QUANTITATIVE)

| Gate | P(fail) | Basis |
|------|---------|-------|
| G1: Extraction viability | 35-40% | Zero prototype; jaxpr unprecedented for conservation |
| G2: Coverage ≥15% | 30% | Dedalus/JAX-MD opaque internals |
| G3: T2 validation | 40-45% | Self-graded C; EXPSPACE; zero proof |
| G4: LLM differentiation | 25% | LLMs improving; threshold on self-constructed benchmarks |
| G5: End-to-end demo k≥3 | 25-30% | 5 stages, 3 ontologies, integration risk 8/10 |

**P(survive all) ≈ 13-16%. P(≥1 kill gate) ≈ 84-87%.**

---

## Score Resolution

| Dimension | Auditor | Skeptic | Synthesizer | **Resolved** | Justification |
|-----------|---------|---------|-------------|:------------:|---------------|
| **Value (V)** | 3 | 3 | 5 | **3.5** | Paradigm novelty acknowledged. Undemonstrated paradigm + narrow TAM + LLM competition caps value. |
| **Difficulty (D)** | 5 | 4 | 5 | **5** | 4-evaluator consensus. Integration challenge is genuine. |
| **Best-Paper (BP)** | 2 | 2 | 3 | **2** | Zero artifacts. ~0.55 novel theorem-equivalents. P(best paper) < 2%. |
| **Laptop/CPU (L)** | 6 | 5 | 7 | **6** | Proven complexity bounds. No empirical benchmark. |
| **Feasibility (F)** | 2.5 | 1 | 3 | **2** | Problem solvable in principle. theory_bytes=0 is realized execution failure. |

**Composite: (3.5 + 5 + 2 + 6 + 2) / 5 = 3.7/10**

---

## Comparison with Prior Evaluations

| Evaluator | Composite | Verdict | Our Assessment |
|-----------|-----------|---------|----------------|
| Prior Skeptic | 3.4/10 | ABANDON | Largely confirmed. F=2 slightly more credible than F=2. |
| Mathematician | 4.0/10 | CONDITIONAL CONTINUE | Override of 2-1 ABANDON majority not justified by evidence. V4 and F3 too generous given theory_bytes=0. |
| Community Expert | 4.1/10 | CONDITIONAL CONTINUE | "4-week sprint" challenged as delayed ABANDON. 3/4 prior conditions unmet undermines sprint credibility. |
| **This Gate (resolved)** | **3.7/10** | **ABANDON** | Cross-critique synthesized. Behavioral evidence (theory_bytes=0) outweighs EV arguments for continuation. |

---

## Probability Assessment

| Outcome | Estimate | Basis |
|---------|----------|-------|
| P(full proposal succeeds) | 13-16% | Compound gate survival |
| P(top venue: OOPSLA/PLDI) | 8-12% | P(gates) × P(accept \| gates) |
| P(best paper) | <2% | P(top venue) × P(BP \| accept) × P(T2 elegant) |
| P(any publication from full tool) | 8-12% | Near-equivalent to gate survival |
| P(any publication from salvage) | 75-85% | Benchmark (85%) + survey (60%) compound |
| P(T2 standalone proof succeeds) | 20-30% | May be trivial or intractable |
| P(ABANDON before completion) | 84-87% | Compound gate failure |

---

## Salvage Protocol

The verdict is ABANDON for the full ConservationLint proposal. The following salvage paths are recommended:

### Immediate (No New Gate Required)

| # | Path | Venue | Timeline | P(accept) | Risk |
|---|------|-------|----------|-----------|------|
| A | Conservation benchmark suite (15-20 kernels) | JOSS | 2-3 months | 85% | Very low |
| B | 2-week extraction experiment (200 lines, binary pass/fail) | — | 2 weeks | — | Diagnostic only |

### Conditional (Requires New Evaluation If Positive)

| # | Path | Trigger | Venue | P(accept) |
|---|------|---------|-------|-----------|
| C | Survey/SoK paper | Path A succeeds | ICSE-NIER / Computing Surveys | 50-65% |
| D | T2 standalone math paper | Proof reveals non-trivial structure | Numerische Math / BIT | 25-40% |
| E | Dynamic-only conservation localizer | Extraction experiment succeeds | ICSE/FSE Tools | 50-60% |

### Reactivation Protocol

If the 2-week extraction experiment succeeds AND T2 is proven for one concrete example, this does NOT automatically reactivate ConservationLint. A new proposal with extraction evidence and T2 proof is required, subject to fresh gate evaluation.

### What Dies

- The full 71K LoC ConservationLint tool as currently scoped
- The repair engine, runtime monitor, IDE integration
- The 40-60% coverage claim (never validated; must not appear in future proposals)
- The PLDI 2025 target (passed)

### What Survives

- The paradigm framing (Noether → program analysis) as a named research direction
- T2 obstruction criterion as a standalone mathematical question
- The benchmark suite as a community resource
- The hybrid extraction architecture as a design document
- The adversarial evaluation archive (~120 pages of structured analysis)

---

## Dissenting Opinion

**Synthesizer (minority, CONDITIONAL CONTINUE):** "The gating questions are well-specified, concrete, and cheaply answerable. EV(CONTINUE) > EV(ABANDON) in sign across all plausible parameter ranges, even after correcting for outcome dependence. The 4-week sprint costs ~4% of the total project budget and resolves ~80% of the uncertainty. Abandoning before testing the two cheaply-testable gating questions wastes information that improves every downstream decision. The benchmark suite alone justifies the ideation investment, and each partial-success mode unlocks a better salvage path than immediate abandonment. The diamonds in this proposal — the obstruction concept, the paradigm framing, the benchmark specification — deserve a bounded attempt at extraction before the mine is sealed."

**Team Lead note:** The Synthesizer's position has intellectual merit and the gating questions ARE cheaply answerable. The prescribed 2-week extraction experiment (Path B) partially addresses this concern within an ABANDON framework. The binary ABANDON/CONTINUE framework loses some nuance here.

---

## Final Verdict

```json
{
  "rankings": [
    {
      "proposal_id": "proposal_00",
      "score": 3.7,
      "verdict": "ABANDON",
      "reason": "theory_bytes=0 after dedicated theory stage. Crown jewel T2 unproven (C-grade, EXPSPACE-complete). Compound gate-failure probability ~84-87%. Zero artifacts after five pipeline stages. TAM ~300-500 with 70% LLM overlap. Paradigm is genuinely novel but undemonstrated. Salvage paths (benchmark suite P=85%, survey P=60%) justify ideation investment. Full tool is not viable without mathematical foundations that were not produced.",
      "scavenge_from": []
    }
  ]
}
```

---

*Post-theory verification gate. 3-expert team (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) + adversarial cross-critique + synthesis. Composite 3.7/10. Vote: 2-1 ABANDON. Process: independent proposals → 6 adversarial challenges → 4 score resolutions → synthesized verdict with structured salvage protocol.*

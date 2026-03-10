# Implementation Gate Report — Algebraic Repair Calculus (ARC)

**Date**: 2026-03-04
**Evaluator**: Impartial Verification Committee (Best-Paper Committee Chair)
**Method**: Claude Code Agent Teams — Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer, Adversarial Cross-Reviewer, Final Verifier
**Proposal**: proposal_00

---

## Team Process

| Phase | Agents | Outcome |
|-------|--------|---------|
| Independent proposals (parallel) | Auditor, Skeptic, Synthesizer | 3 independent evaluations with divergent scores |
| Adversarial cross-critique | Cross-Reviewer vs all 3 | 6 disagreements resolved with code evidence |
| Synthesis | Team Lead | Consolidated scores from strongest arguments |
| Verification signoff | Final Verifier | 8/8 factual claims confirmed against source code |

### Key Disagreements Resolved

| Claim | Auditor | Skeptic | Synthesizer | Ground Truth |
|-------|---------|---------|-------------|--------------|
| Test pass count | 2049 pass, 4 skip | Some integration tests FAIL | 2053 all pass | **2049 pass, 4 skip, 0 fail** (Auditor correct) |
| QualityDelta completeness | Real algebra | No compose/inverse — "two-sorted" | Real with caveats | **Has join/meet/top/bottom (lattice) but NO compose/inverse** (Skeptic partially right) |
| Saga executor | Not assessed | Fake — sequential loop | Real with checkpoint | **Sequential loop, no compensation/rollback** (Skeptic right) |
| DBSP impossibility | Not in code | Zero code, zero tests | Not in code | **Zero code confirmed** (All agree) |
| Property test scope | Robust Hypothesis tests | Add-only, nullable-only | Hypothesis-verified | **Restricted to add-only, nullable-only fragment** (Skeptic right) |
| Push propagation | Real 24 operators | Silent failures on complex DAGs | Real completeness | **Real operators but exception-swallowing fallback on failure** (Both partially right) |

---

## Verified Facts (8/8 confirmed by independent verifier)

1. ✅ **2049 tests pass, 4 skipped, 0 failures** — healthy test suite
2. ✅ **QualityDelta has no compose/inverse** — lattice only (join/meet/top/bottom)
3. ✅ **DBSP impossibility: zero implementation** — pure README claim
4. ✅ **Property tests restricted to add-only, nullable-only schema deltas**
5. ✅ **Execution engine has no saga/compensation** — sequential loop
6. ✅ **Both examples crash** — property/method confusion + typo
7. ✅ **Interaction homomorphisms are real** with verify_homomorphism() at 2 sites
8. ✅ **Annihilation detection has exactly 18 reason codes**

---

## Three-Axis Evaluation

### Axis 1: EXTREME AND OBVIOUS VALUE — 6/10

**The problem is real.** Pipeline breakage from schema evolution is the #1 operational cost in data engineering. Data engineers spend 40-60% of time firefighting. The algebraic approach to computing provably correct repair plans addresses a genuine pain point.

**But the value is not immediately obvious to practitioners.** The target audience needs "my pipeline broke, fix it" not "three-sorted delta algebra with interaction homomorphisms." The formalism serves correctness but creates an adoption barrier. Existing tools (dbt selective rebuild, Great Expectations, manual migration) are crude but well-understood.

**Critical gap:** The system cannot demonstrate end-to-end value today — both examples crash, no benchmarks exist, and the bounded commutation theorem (the core value proposition) is tested only through unit/integration tests, never as a user-facing demo.

**LLM erosion:** Simple repairs (column renames, type changes) are increasingly handleable by LLMs. ARC's unique value is compound perturbations and correctness guarantees — real but niche.

| Expert | Score | Reasoning |
|--------|-------|-----------|
| Auditor | 6 | Real problem, narrow audience |
| Skeptic | 5 | Well-served by existing tools |
| Synthesizer | 7 | Compound perturbations are unique |
| Cross-Reviewer | 6 | Novel but not desperate |
| Final Verifier | 6 | Substantial but execution gap |
| **Consensus** | **6** | |

### Axis 2: GENUINE DIFFICULTY AS A SOFTWARE ARTIFACT — 7/10

**What's genuinely hard and implemented:**
- **24 push operators** (8 SQL ops × 3 delta sorts) with real relational algebra semantics — `push.py` at 1,541 lines
- **Three-sorted composition** with cross-sort interaction homomorphisms φ/ψ — `composition.py:153-181`
- **18 annihilation reasons** with per-sort decomposition — `annihilation.py`, 1,591 lines
- **DP planner** with 4-way per-node decisions and memoization — `dp.py:51-542`
- **LP planner** with scipy.linprog + randomized rounding + greedy feasibility — `lp.py:49-516`
- **Property-based testing** of algebraic laws via Hypothesis — 72 property tests
- **44K total lines**, 37K non-blank source, coherent 12-module architecture

**What reduces the difficulty score:**
- Property tests cover only the add-only/nullable-only fragment — the full algebra's laws are unverified
- QualityDelta is a lattice (join/meet) not a full algebraic sort (no compose/inverse)
- Push propagation uses exception-swallowing fallbacks — not provably correct
- Cost model uses uncalibrated magic constants
- Execution engine is a simple sequential loop

| Expert | Score | Reasoning |
|--------|-------|-----------|
| Auditor | 8 | Real algebra + DP/LP + 24 push ops |
| Skeptic | 7 | Restricted property tests, QualityDelta incomplete |
| Synthesizer | 8 | 44K LOC, Hypothesis, DuckDB integration |
| Cross-Reviewer | 7 | Full algebra laws unverified |
| Final Verifier | 7 | Real but restricted |
| **Consensus** | **7** | |

### Axis 3: BEST-PAPER POTENTIAL — 5/10

**What would excite a committee:**
- Three-sorted delta algebra with interaction homomorphisms — genuine theoretical novelty
- Annihilation detection with 18-reason taxonomy — practical and principled
- Cross-sort composition that DBSP/dbt/Materialize cannot express
- 2049-test suite with property-based algebraic law verification

**What would concern a committee:**
- DBSP encoding impossibility — claimed as main theoretical contribution, zero implementation
- Bounded commutation theorem — tested empirically but not formally proven
- QualityDelta lacks compose/inverse — weakens the "three-sorted algebra" claim
- Property tests restricted to trivial fragment — algebraic laws unverified on full operation set
- No benchmarks — zero performance data, zero annihilation rate measurements
- Both examples crash — reviewer's first impression would be negative
- No comparison with prior work (PRISM, SQUID, Differential Dataflow)

**Realistic venue assessment:**
- **Best paper at top venue (SIGMOD/VLDB):** Needs impossibility proof, full property tests, TPC-DS benchmarks, working demos. Currently far short.
- **Accepted at top venue:** Possible with 4-6 weeks of work on theory + benchmarks.
- **Workshop paper / demo paper:** Ready with 1 week of polish (fix examples, add a demo).

| Expert | Score | Reasoning |
|--------|-------|-----------|
| Auditor | 6 | Novel framework, needs empirical grounding |
| Skeptic | 4 | Missing impossibility proof, restricted tests |
| Synthesizer | 7 | Cross-sort interaction is genuine contribution |
| Cross-Reviewer | 5 | Too many gaps for top-tier |
| Final Verifier | 5 | Honest limitations of ambitious formalization |
| **Consensus** | **5** | |

---

## Summary Scores

| Axis | Score | Weight |
|------|-------|--------|
| Extreme and Obvious Value | 6/10 | — |
| Genuine Difficulty as Software Artifact | 7/10 | — |
| Best-Paper Potential | 5/10 | — |
| **Total** | **18/30** | — |

---

## Fatal Flaws (must fix for any submission)

1. **DBSP impossibility theorem: zero implementation** — the paper's claimed main theoretical contribution is vaporware
2. **Both examples crash** — reviewer's first interaction with the system fails
3. **QualityDelta has no compose/inverse** — undermines "three-sorted algebra" claim
4. **Property tests cover only add-only, nullable-only fragment** — algebraic laws unverified on the full operation set

## Significant Flaws

5. Execution engine lacks saga/compensation (sequential loop with break-on-error)
6. Push propagation swallows exceptions silently (returns input delta unchanged)
7. No benchmarks or performance measurements
8. Dual PipelineGraph classes (types/base.py vs graph/pipeline.py)

## Genuine Strengths

1. **44K lines of real, working code** — 2049 tests pass with 0 failures
2. **Interaction homomorphisms φ/ψ** — genuine mathematical novelty, verified
3. **18-reason annihilation taxonomy** — practical and principled delta pruning
4. **DP + LP tiered planning** — real optimization algorithms
5. **Property-based algebraic law testing** — excellent methodology (needs broader scope)

---

## VERDICT: CONTINUE

**Rationale:** Despite significant gaps, the algebraic core represents genuine research novelty that does not exist in any comparable system. The 2049-test/0-failure suite demonstrates working code, not vaporware. The interaction homomorphisms, annihilation detection, and cost-optimal planning are real contributions. The flaws are fixable (examples: 2 lines; QualityDelta compose: ~50 lines; property test expansion: ~100 lines). The DBSP impossibility is the only potentially unfixable claim — it may need to be demoted to a conjecture or remark.

**Conditions:** A polish round MUST fix the crashing examples and expand property test strategies before any value claims are credible.

---

## Rankings

```json
{
  "rankings": [
    {
      "proposal_id": "proposal_00",
      "score": 6,
      "verdict": "CONTINUE",
      "reason": "Genuine algebraic novelty (interaction homomorphisms, 18-reason annihilation, DP/LP planners) with 2049 passing tests. Value=6, Difficulty=7, BestPaper=5. Flaws are fixable: crashing examples (2 lines), missing QualityDelta compose (~50 lines), restricted property tests (~100 lines). DBSP impossibility claim needs demotion or implementation. Core architecture is sound."
    }
  ],
  "best_proposal": "proposal_00"
}
```

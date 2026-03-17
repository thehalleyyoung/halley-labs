# Theory Gate Report: IsoSpec (txn-isolation-verifier)

**Proposal:** proposal_00 — IsoSpec: Verified Cross-Engine Transaction Isolation Analysis via Engine-Faithful Models and Differential Portability Checking  
**Stage:** Verification (theory → implementation gate)  
**Panel:** Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer  
**Method:** Independent proposals → adversarial cross-critique → consensus synthesis → verification signoff  
**Date:** 2026-03-08  
**Prior evaluations synthesized:** 3 (Skeptic 29/50, Mathematician 27/50, Community Expert 30/50)

---

## Executive Summary

IsoSpec proposes formal operational semantics for production SQL engines (PostgreSQL 16.x, MySQL 8.0, SQL Server 2022), connected to Adya's isolation specifications via refinement, with differential SMT analysis for cross-engine portability checking. After a completed theory stage, the project has produced **~167KB of planning prose and zero executable artifacts** — no Lean proofs, no SMT encodings, no Rust code. State.json confirms: `theory_bytes=0`, `impl_loc=0`, `code_loc=0`.

Six independent evaluations (3 prior + 3 new) unanimously confirm the intellectual core (M1 engine models + M5 predicate conflict theory) as genuinely novel. Six evaluations unanimously recommend **CONDITIONAL CONTINUE**. The Fail-Fast Skeptic formally dissents toward ABANDON (19/50) but concedes a 14-day execution gate would resolve the core uncertainty.

**VERDICT: CONDITIONAL CONTINUE — 29/50 — Confidence 85%**

Continuation is gated on a binding 14-day execution spike producing three concrete deliverables. Failure triggers immediate ABANDON with salvage to workshop paper.

---

## Panel Scores

| Pillar | Auditor | Skeptic | Synthesizer | **Consensus** | Prior Avg |
|--------|:-------:|:-------:|:-----------:|:-------------:|:---------:|
| **V: Extreme Value** | 6 | 3 | 7 | **6** | 6.0 |
| **D: Genuine Difficulty** | 7 | 5 | 7 | **7** | 6.2 |
| **BP: Best-Paper Potential** | 5 | 3 | 6 | **5** | 4.8 |
| **L: Laptop-CPU / No-Humans** | 8 | 6 | 8 | **7** | 7.2 |
| **F: Feasibility** | 4 | 2 | 6 | **4** | 4.5 |
| **Composite** | **30** | **19** | **34** | **29/50** | **28.6** |

Score spread: 15 points (19–34). Largest disagreement on Feasibility (2–6, 4-point gap). Smallest on Difficulty (5–7) and Laptop-CPU (6–8).

---

## Pillar Analysis

### Pillar 1: Extreme & Obvious Value — 6/10

**What's real:** Database migration isolation bugs are a genuine, well-documented problem. The cloud migration market exceeds $10B annually. AWS/Azure/Google DMS perform zero isolation semantics verification. No existing tool answers: "Will my transaction break when I migrate?" Hermitage documents 47+ divergences but only for hand-written patterns — no completeness guarantee, no CI/CD integration.

**What's missing:**
- **Zero production incidents cited.** The gap between "engines behave differently" and "this routinely causes production failures" is asserted, not evidenced. In 12 years since Hermitage, no published case study attributes a production failure to isolation-level divergence during migration.
- **LLM competitive threat.** By 2027, LLM + Hermitage + targeted testing covers ~50-70% of practitioner one-off needs. IsoSpec's durable advantage is formal completeness, CI/CD integration, and systematic discovery — real but narrow.
- **Adoption barriers.** DBAs don't use formal verification tools. The witness-script output (runnable SQL tests) partially mitigates this.

**Cross-critique resolution:** Skeptic (V=3) argued zero evidence of real-world harm makes value speculative. Synthesizer (V=7) argued the $10B market gap is genuine. Auditor (V=6) was best-calibrated — real problem with undemonstrated harm. The academic value (formalizing engine semantics as new knowledge) scores 7+; the practitioner value story scores 4-5. **Blended: 6.**

### Pillar 2: Genuine Difficulty — 7/10

**Honest LoC assessment (consensus across 6 evaluations):**

| Component | Claimed | Consensus Novel (2-engine) |
|-----------|:-------:|:--------------------------:|
| Engine models (PG + MySQL) | 18K | ~14K |
| M5 predicate conflict theory | 5K | ~4K |
| SMT encoding | 7K | ~6K |
| Refinement + portability | 9K | ~7K |
| Parser + IR | 6K | ~5K |
| Witness synthesis | 4K | ~3K |
| Lean 4 (2-3 lemmas) | 5K | ~3K |
| Validation oracle | 4K | ~3K |
| **Total genuinely novel** | **~60K** | **~45K** |

**What's genuinely hard:** PG SSI formalization (SIREAD locks, dangerous structure detection, read-only optimization), MySQL InnoDB index-dependent gap locking (no published formalization exists), M5 decidability boundary characterization. These require PhD-level database internals expertise.

**What's inflated:** LoC trajectory 78K→55K→60K→45K shows consistent ~30% inflation per stage. SMT encoding follows established CLOTHO/BMC patterns. Test suite/CLI/infra are not novel.

**Cross-critique resolution:** Skeptic (D=5) argued difficulty is in content not method. Synthesizer (D=7) argued engine formalization is irreplaceable PhD-level work. Auditor (D=7) agreed with Synthesizer — the difficulty is real, concentrated in ~14K of engine models, with ~31K of advanced engineering. **Consensus: 7.**

### Pillar 3: Best-Paper Potential — 5/10

**Genuine math contributions: 2.** M5 (predicate conflict decidability) is the flagship. M1+M2 (engine semantics + refinement) constitute a framework contribution. M3/M4/M7/M8 are supporting lemmas or engineering. The proposal's claim of "8 math contributions" is inflated by ~60%.

**CLOTHO positioning risk:** The parse→encode→solve pipeline is architecturally identical to CLOTHO. The differentiation (engine-specific models vs. abstract specs) is genuine but must be front-and-center: "CLOTHO models what the spec says; IsoSpec models what the engine does." A reviewer who sees this as "CLOTHO with different constraints" will reject.

**Discovery framing is the best-paper angle:** "Our models revealed N surprising behaviors" is compelling — but N is unknown and speculative. Realistic N for 2 engines: 10-18, with 3-6 migration-affecting.

**P(best-paper):** 3-8% after all fixes. 1-2% in current state.  
**P(strong accept):** 40-55% after all fixes. 15-25% in current state.  
**P(any top-venue publication):** 55-70% after all fixes. 25-40% in current state.

**Cross-critique resolution:** Skeptic (BP=3) argued SIGMOD never awards FM tools and M5 has FATAL flaws. Synthesizer (BP=6) argued discovery framing compensates. **Consensus: 5** — solid math core + promising framing, but discoveries are hypothetical and the CLOTHO risk is real.

### Pillar 4: Laptop-CPU / No-Humans — 7/10

**Feasible components:** Z3 QF_LIA+UF+Arrays for k≤3 is well within tractability. Docker: PG (~200MB) + MySQL (~400MB) = ~600MB sequential. Lean 4 batch-mode, minutes. All evaluation fully automated — no annotation, no human studies.

**Concerns:** Gap-lock disjunctions over index choices create constraint explosion (untested). Docker interleaving forcing has 25-35% failure rate (not the 10% claimed). SMT performance claims have zero experimental validation (FATAL-4).

**Consensus: 7.** Sound architecture, but SMT performance is unvalidated. Prevents scoring 8+.

### Pillar 5: Feasibility — 4/10

**The zero-artifact problem is the pivotal fact.** After a completed theory stage: 0 lines of code, 0 SMT encodings, 0 Lean proofs. ~167KB of planning prose exists. State.json: `theory_bytes=0`, `impl_loc=0`. The theory stage produced proposals about what to prove, not proofs. This is the strongest predictor of implementation difficulty.

**4 FATAL flaws, 0 fixed:**

| # | Flaw | Severity | Fixable? | Effort |
|---|------|----------|----------|--------|
| FF1 | NULL handling breaks M5 decidability | HIGH | Yes (scope to NOT NULL) | 1-2 weeks |
| FF2 | PG SSI model ignores memory pressure | MEDIUM | Yes (bounded-memory assumption) | 1 week |
| FF3 | k=3 sufficiency proof has errors | HIGH | Partially (item-level yes; predicate-level open) | 1-2 weeks |
| FF4 | SMT performance unvalidated | MEDIUM | Yes (benchmark) | 1 week |

**Timeline reality:** The 11-month proposed timeline is optimistic by 2-4 months. SIGMOD 2027 is unreachable. SIGMOD 2028 is the realistic primary target. For the recommended 2-engine scope (Path B): 9 months with kill-gates.

**Cross-critique resolution (largest disagreement):** Skeptic (F=2) argued zero artifacts is disqualifying; compound probability of fixing all issues ~8%. Synthesizer (F=6) argued kill-gates + PG-only fallback limit downside. Auditor (F=4) was best-calibrated — feasibility of *something publishable* is 6; feasibility of the *proposed system* from current state is 3. **Consensus: 4.**

---

## Cross-Critique Exchanges

### Exchange 1: Skeptic → Synthesizer on Feasibility

> **Skeptic:** "You gave Feasibility a 6. There are ZERO lines of code after a complete theory stage. The compound probability of fixing all 4 FATAL flaws is 23.5%. Your '3-4 weeks to fix all FATALs' assumes a team that couldn't produce a 50-line Z3 encoding in an entire stage will suddenly become productive. Planning is NOT the hard part — execution is."

> **Synthesizer:** "The compound probability calculation assumes independence. FATAL-1 (NULLs) and FATAL-3 (k=3 proof) share mathematical foundations — fixing one provides insights for the other. FATAL-2 (PG memory) and FATAL-4 (SMT perf) are engineering, not math — different skill from theory writing. The theory stage produced the *wrong output type* (prose instead of formalism), not evidence of inability. My F=6 reflects that Path B with kill-gates limits downside to 2 months before the first gate catches failure."

> **Auditor adjudication:** The Skeptic is right that zero artifacts is an alarming execution signal. The Synthesizer is right that the FATALs are correlated (not independent) and kill-gates limit exposure. The compound probability with correlation correction: ~30-35%, not 23.5% or 70%. **F=4 is the calibrated consensus** — below both Synthesizer and Auditor's initial scores, acknowledging the Skeptic's core concern.

### Exchange 2: Synthesizer → Skeptic on Value

> **Synthesizer:** "You scored Value at 3/10. But you yourself identified three salvage components worth publishing. If the *salvage* has publication value, the full system cannot score 3 on value. Your 'zero companies pay for formal isolation verification' is the wrong market test — this is an *academic research paper*, not a SaaS product. The audience is SIGMOD/CAV reviewers, not enterprise customers."

> **Skeptic:** "Fair point on the market test — I'll concede V=4 for academic value. But the 'desperate need' criterion requires evidence of real harm. The proposal cites zero production incidents. Hermitage has existed since 2014 with no follow-on tool. If this were desperately needed, someone would have built it. The 12-year gap between Hermitage and IsoSpec is evidence of low demand, not unsolved difficulty."

> **Auditor adjudication:** The Synthesizer wins on the market test argument — academic value ≠ enterprise demand. The Skeptic wins on the incident evidence gap. **V=6 consensus** reflects real problem + real gap + no evidence of real harm.

### Exchange 3: Auditor → Both on the 14-Day Gate

> **Auditor:** "The Skeptic's 14-day reversal condition and the Synthesizer's kill-gate at Month 2 are actually the same insight expressed differently. Both recognize that the core uncertainty is *execution capability*, not *intellectual merit*. A 14-day spike producing a working Z3 encoding, a fixed k-bound proof, and a NULL-handling decision would resolve the pivotal question. If it passes, the Skeptic's concerns are addressed. If it fails, the Synthesizer's optimism is falsified. This is the dominant strategy — 14 days of limited exposure to resolve the verdict."

Both experts concurred. The 14-day execution gate is the binding condition.

---

## Unanimous Findings (High Confidence)

| # | Finding |
|---|---|
| U1 | Engine operational semantics (M1) are genuine novel artifacts — confirmed by all 6 evaluations |
| U2 | M5 predicate conflict theory is the flagship math contribution — confirmed unanimously |
| U3 | theory_bytes=0 / impl_loc=0 is the pivotal execution concern |
| U4 | 4 FATAL flaws are correctly identified and must be addressed before full implementation |
| U5 | CLOTHO positioning must be front-and-center or risk desk rejection |
| U6 | LoC trajectory (78K→55K→60K→45K) shows consistent ~30% inflation |
| U7 | PostgreSQL SSI is the strongest, most tractable engine to formalize first |
| U8 | SIGMOD 2027 is unreachable; SIGMOD 2028 is the realistic primary target |
| U9 | A real-world migration case study is the single highest-impact evaluation addition |
| U10 | The 14-day execution gate is the correct decision structure |

---

## Recommended Scope: Path B (Two-Engine)

**Scope:** PostgreSQL 16.x SSI + MySQL 8.0 InnoDB + M5 predicate theory + differential portability + discovery campaign

| Component | Novel LoC | Priority |
|-----------|:---------:|---------|
| PG SSI operational semantics | ~8K | P0 |
| M5 predicate conflict theory + SMT encoding | ~7K | P0 |
| Refinement checker (PG→Adya) | ~3K | P0 |
| MySQL InnoDB gap-lock semantics | ~6K | P1 (after KG1) |
| Differential portability (PG↔MySQL) | ~4K | P1 |
| Witness synthesis | ~4K | P1 |
| Parser/IR extensions | ~5K | Throughout |
| Lean 4 proofs (2-3 lemmas) | ~3K | P2 (nice-to-have) |
| Empirical validation oracle | ~3K | P2 |
| **Total genuinely novel** | **~45K** | |

**Timeline:** 9 months (after 14-day gate), targeting SIGMOD 2028 or VLDB 2027.

---

## Kill-Gate Sequence

| Gate | Deadline | Criterion | If Failed |
|------|----------|-----------|-----------|
| **KG0** | Day 14 | Working Z3 encoding of PG SSI write skew (≥50 constraints, correct sat/unsat, <60s); corrected k-bound proof; NULL-handling decision with scope analysis | **ABANDON** |
| **KG1** | Month 2 | PG model ≥95% agreement on Hermitage PG suite; Z3 solves k=3, n=10 in <60s | **ABANDON**; salvage M1 as workshop paper |
| **KG2** | Month 4 | MySQL model ≥90% Hermitage agreement; gap-lock FP <25%; ≥1 novel PG↔MySQL divergence | **DROP MySQL**; pivot to PG-only (Path C) |
| **KG3** | Month 7 | ≥8 novel confirmed divergences, ≥2 migration-affecting | Weaken discovery narrative; lead with completeness |
| **KG4** | Month 9 | Draft paper with evaluation | Ship what exists |

---

## Binding Conditions for CONTINUE

All conditions are mandatory. Failure on any condition marked "KILL" triggers ABANDON.

| # | Condition | Type | Deadline |
|---|-----------|------|----------|
| BC1 | Produce working Z3 encoding of PG SSI write skew detection (≥50 constraints, correct result, <60s) | **KILL** | Day 14 |
| BC2 | Fix k=3 sufficiency: rigorous item-level proof OR explicit bounded-completeness claim | **KILL** | Day 14 |
| BC3 | NULL-handling decision: NOT NULL restriction with quantitative scope analysis OR 3VL encoding prototype | **KILL** | Day 14 |
| BC4 | CLOTHO differentiation paragraph in paper outline | Mandatory | Month 1 |
| BC5 | LLM positioning section drafted | Mandatory | Month 1 |
| BC6 | LoC claims corrected to ~45K novel (2-engine scope) | Mandatory | Immediate |
| BC7 | Math contributions restructured: 2-3 genuine, not "8 equal" | Mandatory | Month 1 |
| BC8 | 2-week focused search for production migration incidents | Recommended | Month 1 |

---

## Probability Estimates

| Outcome | Current State | After KG0 Pass | After Full Remediation |
|---------|:------------:|:--------------:|:---------------------:|
| Total Abandon | 30-40% | 15-20% | 8-12% |
| PG-only paper (CAV/VLDB) | 20-25% | 25-30% | 15-20% |
| 2-engine paper (SIGMOD) | 15-20% | 35-40% | 45-55% |
| Best paper (any venue) | 1-2% | 5-8% | 8-12% |
| Any top-venue publication | 35-50% | 60-70% | 65-75% |

---

## Dissent Record

**Fail-Fast Skeptic (19/50, ABANDON):** "The project has completed a theory stage and produced zero executable artifacts. The flagship theorem (M5) is unsound for standard SQL. The compound probability of fixing all issues is ~8-23%. Zero companies pay for formal isolation verification. The LLM threat deflates 70-80% of practitioner value. ABANDON and salvage M5 as a theory workshop paper."

**Skeptic's reversal condition:** A 14-day spike producing (1) working Z3 encoding, (2) fixed k-bound proof, (3) NULL-aware encoding prototype. If produced, Skeptic would upgrade to CONDITIONAL CONTINUE at ~26/50.

**Auditor and Synthesizer:** CONDITIONAL CONTINUE without dissent, agreeing on the 14-day gate as the resolution mechanism.

---

## The Bottom Line

A project with a genuine intellectual contribution (engine-specific operational semantics that have never existed as formal artifacts) buried inside zero execution output and 4 unresolved FATAL flaws. The planning quality is exceptional — ~167KB of self-aware, well-critiqued prose. But planning ≠ execution. The 14-day gate is the minimum viable test of execution capability.

**If KG0 passes:** This becomes a strong CONDITIONAL CONTINUE with P(publication) ≈ 60-70%. The engine models are the crown jewel — irreplaceable by LLMs, novel as formal artifacts, citable by future work. Path B (PG + MySQL) is the optimal scope.

**If KG0 fails:** ABANDON immediately. Salvage M5 as a standalone theory result (PODS workshop). Contribute the engine behavioral analysis as an updated Hermitage-style catalog. The 14 days of exposure is the correct price to pay for resolving the uncertainty.

**Decision: CONDITIONAL CONTINUE — mandatory 14-day execution gate (KG0). Composite 29/50. Confidence 85%.**

---

## Evaluator Calibration

| Expert | Calibration | Strength | Weakness |
|--------|:-----------:|----------|----------|
| **Auditor** | Best-calibrated (4/5 pillars within ±1 of consensus) | Transparent evidence chains; introduced execution gate concept | Slightly generous on D (7 vs prior ~6) |
| **Skeptic** | Best at risk identification | Zero-artifact signal, compound probability, market test, LLM threat | V=3 too harsh (conflates academic and market value); F=2 may conflate "currently broken" with "unfixable" |
| **Synthesizer** | Best constructive path | Crown jewel identification, Path B scoping, kill-gate design, CLOTHO strategy | F=6 too optimistic given zero artifacts; scored potential not current state |

Cross-validation against 3 prior evaluations: new panel composite (29/50) converges with prior average (28.6/50) within 1 point. Strong inter-rater reliability across 6 independent evaluations.

---

*Report produced by 3-expert adversarial panel with verification reviewer signoff. All experts independently evaluated, cross-critiqued, and synthesized under team-lead adjudication.*

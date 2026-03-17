# Skeptic Verification Evaluation: IsoSpec (proposal_00)

**Proposal:** IsoSpec: Verified Cross-Engine Transaction Isolation Analysis via Engine-Faithful Models and Differential Portability Checking  
**Slug:** `txn-isolation-verifier`  
**Phase:** Theory → Implementation gate decision  
**Evaluator:** Fail-Fast Skeptic (Lead), with Independent Auditor + Scavenging Synthesizer  
**Method:** Independent proposals → adversarial cross-critique → team-lead adjudication → consensus synthesis  
**Date:** 2026-03-08

---

## Executive Summary

IsoSpec proposes building rigorous operational semantics for three production SQL engines (PostgreSQL 16.x, MySQL 8.0, SQL Server 2022), connecting them to Adya's isolation specifications via refinement, and using differential SMT analysis to detect portability violations and discover previously undocumented isolation behaviors. After three-expert adversarial evaluation, the proposal contains **genuine, novel research contributions** (engine-specific operational semantics + predicate conflict theory) **buried under theory-stage oversell, four FATAL flaws, and zero lines of implementation code.**

**VERDICT: CONDITIONAL CONTINUE** — Consensus 3-0 (Auditor 28/50, Skeptic 27/50, Synthesizer 33/50 → Adjudicated **29/50**). Narrow margin above continue threshold. All conditions binding.

---

## 0. Raw Score Comparison

| Criterion | Auditor | Skeptic | Synthesizer | **Consensus** |
|---|:---:|:---:|:---:|:---:|
| Extreme Value (V) | 6 | 5 | 7 | **6** |
| Genuine Difficulty (D) | 6 | 6 | 7 | **6** |
| Best-Paper Potential (BP) | 5 | 4 | 6 | **5** |
| Laptop-CPU / No-Humans (L) | 7 | 8 | 7 | **7** |
| Feasibility (F) | 4 | 4 | 6 | **5** |
| **Composite** | **28** | **27** | **33** | **29/50** |

All three recommend CONDITIONAL CONTINUE. Disagreement is on *how conditional*.

---

## 1. Pillar Analysis

### Pillar 1: Extreme Value — 6/10

**What's real:** Database migration isolation bugs are a genuine, well-documented problem class. The cloud migration market exceeds $10B annually. AWS/Azure/Google DMS handle schema and data transfer but perform zero isolation semantics verification. No existing tool answers: "Will my transaction break when I migrate from engine A to engine B, and here is a runnable script proving it." Formal completeness guarantees are categorically different from LLM heuristics or manual testing.

**What's missing:**

- **Zero concrete production incidents cited.** Hermitage documents *differences*, not *failures caused by those differences*. The gap between "engines behave differently" and "this routinely causes billion-dollar bugs" is asserted, not evidenced. The $10B migration market figure is total market, not addressable segment for formal verification tooling.
- **LLM threat unaddressed.** In 2026-2027, LLMs cover ~75-80% of practitioner value: enumerate known Hermitage divergences, explain engine-specific concurrency control, generate migration test scripts, advise on isolation-level gotchas. IsoSpec's remaining 20-30% advantage (formal completeness, decidability boundaries, minimal verified witnesses) is real but narrow — and serves a market that has never paid for formal isolation verification of any kind.
- **Adoption barriers severe.** DBAs and SREs do not use formal verification tools. The actionable output (runnable SQL scripts) partially mitigates this, but the proposal oscillates between "tool for practitioners" and "framework for PL researchers" without committing.

**Panel disagreement:** Skeptic scored 5 (too harsh — dismisses categorical advantage of formal completeness); Synthesizer scored 7 (too generous — no incident evidence). Auditor scored 6 (well-calibrated). **Resolution:** Academic value (formalizing engine semantics as new knowledge artifacts) scores 7+; practitioner value story scores 4-5. Blended = 6.

### Pillar 2: Genuine Difficulty — 6/10

**Honest LoC Assessment (audited by 3 experts):**

| Component | Claimed | Consensus Novel |
|---|:---:|:---:|
| Engine models (3 engines) | 18K | **15K** (over-approximation reduces model complexity) |
| M5 predicate theory + SMT | 12K | **10K** |
| Refinement checker | 4K | **3K** |
| Portability analyzer | 5K | **4K** |
| Witness synthesis | 4K | **3K** |
| Parser/IR extensions | 6K | **5K** |
| **Total genuinely novel** | **60K** | **~40K** |

**Trajectory of deflation:** 78K (original) → 55K (depth check, −30%) → 60K (current proposal, paradoxically inflated after descoping) → **~40K** (consensus after removing infrastructure-as-novelty and applying documented 15-20% underestimation). For the recommended two-engine scope: **~38K novel LoC**.

**What's genuinely hard:**
- Engine model fidelity for MySQL InnoDB's index-dependent gap locking (no published formalization exists)
- PostgreSQL SSI read-only optimization (negation over graph structures in SMT)
- M5 predicate conflict decidability boundary characterization
- SQL Server dual-mode concurrency interaction semantics

**What's inflated:**
- SMT encoding follows CLOTHO's established BMC patterns (technique is known; content is novel)
- SQL parser extensions over sqlparser-rs (~6K LoC of careful but not novel engineering)
- Mixed-isolation recommender (MaxSMT off-the-shelf)
- Empirical validation oracle (Docker orchestration is infrastructure, not contribution)

### Pillar 3: Best-Paper Potential — 5/10

**Math novelty audit (consensus):**

| Contribution | Claimed | Consensus |
|---|---|---|
| **M5: Predicate-Level Conflict Theory** | **New theory (Hard)** | **Genuinely novel — flagship** ✓ |
| M1: Engine Operational Semantics | New artifacts (Hard) | Novel artifacts, not new technique |
| M2: Isolation Refinement Relation | New theory (Hard) | Novel application of known process-algebraic refinement |
| M3: Portability coNP-completeness | New result (Moderate) | Expected complexity result — routine |
| M4: Symbolic Engine Encoding | New technique (Hard) | Hard engineering within known BMC framework |
| M6: Mixed-Isolation Optimization | New formulation (Moderate) | Reasonable MaxSMT application |
| M7: Bounded Soundness/Completeness | New characterization (Moderate) | Minor lemma (k=3 follows from anomaly structure — and the proof is wrong) |
| M8: Compositionality Theorem | New theorem (Moderate) | Standard refinement property — almost definitional |

**Genuine math contributions: 2.** M5 is the flagship. M1+M2 together constitute a framework contribution. Everything else is supporting material.

**The CLOTHO comparison problem:** A knowledgeable reviewer will see "CLOTHO's parse→encode→solve pipeline applied to SQL engine models instead of abstract distributed consistency models." The differentiation is real (engine-specific implementation semantics vs. abstract specs) but must be front-and-center. The one-sentence defense: "CLOTHO models what the specification says; IsoSpec models what the engine does." This is sufficient if the engine models are demonstrably faithful. If model fidelity is undemonstrated, the differentiation collapses.

**The "discovery instrument" circularity:** The Skeptic correctly identifies a philosophical tension: if models are faithful, they reproduce known behavior (no discovery); if they discover unknown behavior, the models might be wrong. The Synthesizer correctly resolves it: models serve as *systematic hypothesis generators*, with empirical confirmation closing the loop. This is standard model-based testing methodology. But the word "discovery" oversells it — "systematic divergence enumeration" is more honest.

**Path to best paper (requires all of):**
1. ≥5 truly novel, migration-affecting divergences confirmed by execution
2. One compelling real-world migration case study from PG mailing lists/AWS DMS bug reports
3. M5 predicate theory with correct NULL handling
4. CLOTHO differentiation front-and-center
5. Narrative reframe: "Our models revealed X surprising behaviors" before "and we built a tool"

**Without these:** Strong accept territory at SIGMOD/VLDB. Not best paper.

### Pillar 4: Laptop-CPU Feasibility & No-Humans — 7/10

**Z3 feasibility:** QF_LIA+UF+Arrays is well-optimized in Z3 for bounded problems. For k=3 (covers all item-level Adya anomaly classes), constraint sizes are manageable. Engine-specific constraints actually *prune* the search space relative to abstract models. **Concern:** Gap-lock disjunctions over possible index choices create exponential branching — untested.

**Docker feasibility:** PostgreSQL (~200MB), MySQL (~400MB), SQL Server (~1.5GB). Total ~2.1GB for 3-engine scope. Sequential testing on 16GB RAM is fine.

**No humans required:** All evaluation is automated. No annotation, no user studies, no human subjects. Fully automated pipeline from SQL input to verified witnesses.

**Lean 4:** CPU-only batch mode. Descoped to 3-5 key lemmas. No concern.

### Pillar 5: Feasibility — 5/10

**Current state of deliverables:**
- theory_bytes = 0 in State.json (measurement bug — theory/ has ~167KB of content across 7 files, but `proposals/proposal_00/theory/` is empty)
- impl_loc = 0 (no implementation code exists)
- Theory content is definitions + proof sketches, not completed proofs (14/16 claims fail challenge testing per Auditor)
- 4 FATAL flaws + 8 SERIOUS issues identified by red team

**Serial dependency chain:** Transaction IR → Engine models (M1) → SMT encoding (M4) → Refinement checker (M2) → Portability analyzer → Witness synthesis → Evaluation. Any delay in M1 cascades everything.

**Mitigating factors:** Kill gates with objective criteria (KG1: SMT feasibility, KG2: PG model adequacy, KG3: MySQL fidelity, KG4: novel divergences). Clear fallback to PG-only paper at VLDB/PODS if MySQL/SQL Server prove too hard. PG-only + M5 is independently publishable.

**Panel disagreement:** Skeptic + Auditor scored 4 (zero implementation, 4 FATAL flaws, serial dependencies). Synthesizer scored 6 (kill gates + fallback scope). **Resolution:** Feasibility of *something publishable* is 6-7. Feasibility of *full 3-engine vision* is 3-4. Weighted by recommended 2-engine scope → 5.

---

## 2. The Hardest Questions

### 2a) The LLM Test (2026)

LLM + Hermitage + manual testing covers ~75-80% of what IsoSpec claims to provide. The remaining 20-30% — formal completeness guarantees, decidability boundary characterization, minimal verified witnesses, CI/CD-integrated regression testing — is real but serves a market that has never paid for formal isolation verification. **Verdict: IsoSpec survives but with deflated practitioner value. Lead with the formal methods community as primary audience.**

### 2b) The CLOTHO Test

Strip the engine models and what remains is CLOTHO's architecture with different SMT constraint content. The engine models ARE the contribution — but they exist only as markdown definitions, not as validated artifacts. **Verdict: Contribution is contingent on model fidelity, which is entirely undemonstrated.**

### 2c) The Hermitage Test

Finding 15+ novel divergences via formal models is NOT inherently harder than writing 15 more Hermitage-style test cases. The model's advantage is systematic enumeration. But the "novel" filtering uses subjective criteria. **Verdict: Expect 3-5 truly surprising divergences, not 15. The rest will be trivial variants or obscure corner cases.**

### 2d) The Version Fragility Problem

Models are point-in-time snapshots. PG 16.x ≠ PG 17.x. Every `apt upgrade` invalidates guarantees. **Verdict: Acceptable for academic paper (Cahill et al. was version-specific too). Fatal for the practitioner tool narrative. Lead with research contribution.**

### 2e) The "Discovery" Reframe

The circularity concern dissolves in practice with the empirical validation loop (model predicts → Docker confirms → divergence documented). But "discovery" oversells the contribution. **Verdict: Legitimate as "systematic divergence enumeration," not as breakthrough "discovery."**

### 2f) The 55K LoC Question

After three rounds of correction: **~40K genuinely novel** for the full 3-engine scope; **~38K** for the recommended 2-engine scope. The trajectory 78K→55K→40K shows consistent ~30% inflation at each stage.

### 2g) The theory_bytes=0 Anomaly

**Measurement bug:** State.json tracks `proposals/proposal_00/theory/` (empty directory), not the top-level `theory/` directory (~167KB across 7 files). The theory content exists but is proposal-quality: definitions, proof sketches, and "proof structures suitable for mechanization" — not completed proofs. **Zero completed, correct proofs exist.** The theory_bytes=0 is accidentally honest about the quality of mathematical results.

---

## 3. FATAL Flaw Assessment

### FATAL-1: Predicate Conflict NULL Handling (Severity: HIGH)

**The flaw:** Theorem 5.1 claims "predicate conflict detection for conjunctive inequalities is in NP." SQL's three-valued logic with NULLs invalidates the convex polytope argument. `x > 5 AND x < 10` with NULL handling cannot be encoded as linear constraints due to UNKNOWN semantics.

**Can it be fixed?** Yes, with scope reduction. Restrict to NOT NULL columns (covers >80% of indexed OLTP columns) or provide co-NP-complete proof with sound NULL-aware encoding.

**Effort:** 1-2 weeks for NOT NULL restriction; 3-4 weeks for full NULL-aware encoding.

**Threatens viability?** Threatens M5 as a clean theoretical contribution. The NOT NULL restriction is honest but reduces the wow factor. A reviewer who knows SQL will scrutinize this.

**Survives peer review?** 70% with NOT NULL restriction; 85% with full NULL handling.

### FATAL-2: PostgreSQL SSI Implementation Gap (Severity: MODERATE-HIGH)

**The flaw:** PG 16.x implements granularity promotion (tuple→page→relation), lock cleanup, and SIREAD lock summarization under memory pressure. These affect conflict detection semantics, not just performance. The model assumes infinite SIREAD lock memory.

**Can it be fixed?** Yes. Add "infinite SIREAD lock memory" assumption. The model becomes a sound *over-approximation*: real PG may abort more under memory pressure (coarsened locks increase false conflict detection), so the model admits a superset of real PG's schedules for the anomaly-existence direction.

**Effort:** 1-2 weeks for assumption + empirical validation.

**Survives peer review?** 80%. Bounded memory models are standard practice. A PG expert reviewer will flag this but accept the qualified claim.

### FATAL-3: k=3 Sufficiency Proof Errors (Severity: HIGH)

**The flaw:** G1a requires k=2 (not k=3 as claimed). G2-item example shows 6-transaction pattern miscounted as 3. No formal argument for predicate-level G2 anomalies, which may require k>3.

**Can it be fixed?** Partially. Item-level Adya anomalies with correct k values: straightforward (1-2 weeks). Predicate-level G2: unknown — phantom anomalies can involve complex predicate interactions. The honest claim is "k=3 suffices for all item-level Adya anomalies; predicate-level analysis uses user-supplied k with no universal bound."

**Effort:** 1-2 weeks for correct item-level proof. Unknown for predicate-level.

**Survives peer review?** 60% as-is; 80% if downgraded to "empirically sufficient for all known Adya patterns."

### FATAL-4: SMT Performance Unsubstantiated (Severity: MODERATE)

**The flaw:** "Sub-30-second analysis" has zero experimental validation. PG SSI encoding generates ≥1000 constraints for k=3, n=10 operations, before gap-lock disjunctions.

**Can it be fixed?** Yes — this is a pure engineering question. Implement prototype encoding, benchmark against Z3, adjust claims to match reality.

**Effort:** 2-3 weeks.

**This is the KILL GATE.** If Z3 cannot handle the encoding at all, the tool is dead. Make this Week 1 of implementation.

**Survives peer review?** 85% (even with relaxed performance claims, a tool with 120s timeout and degradation handling is acceptable).

### FATAL Fix Summary

| FATAL | Fix Effort | Risk | Priority |
|---|---|---|---|
| FATAL-4 (SMT perf) | 2-3 weeks | **This is the existential gate** | 1st (do immediately) |
| FATAL-1 (NULLs) | 1-3 weeks | Fixable, scope-reducing | 2nd |
| FATAL-3 (k=3 proof) | 1-2 weeks | Downgrade to conjecture | 3rd |
| FATAL-2 (PG memory) | 1-2 weeks | Standard assumption | 4th |
| **Total** | **6-10 weeks** | | |

---

## 4. Challenge Exchanges Between Experts

### Challenge 1: Synthesizer → Skeptic on LLM Threat

> **Synthesizer:** "Your 75-80% LLM coverage figure is made up. Name one LLM capability that provides formal completeness guarantees for predicate-level phantom detection. The whole point of IsoSpec is the 20-30% that LLMs provably cannot cover — and that's exactly where $2M production bugs hide."

> **Skeptic:** "I never said equivalent. I said practitioners won't pay for the remaining 20-30% when the 75-80% is free. Show me one Fortune 500 company that currently pays for formal isolation verification of any kind. Zero."

**Resolution:** The Synthesizer wins the technical argument — LLMs genuinely cannot provide formal completeness. The Skeptic wins the market argument — demand for those guarantees is undemonstrated. **The paper should lead with the formal methods community as primary audience** and treat practitioners as aspirational.

### Challenge 2: Skeptic → Synthesizer on Feasibility

> **Skeptic:** "You gave Feasibility a 6. There are ZERO lines of implementation code. The theory has 4 FATAL flaws. The SMT encoding hasn't been tested against a single constraint. Your 6 is based on kill gates that assume rational decision-making in the face of sunk costs."

> **Synthesizer:** "KG1 is a 2-day experiment. Encode PG SSI, run Z3, measure time. If it times out, the project is dead and everyone knows it. My 6 reflects that *something publishable* can emerge from multiple exit points."

**Resolution:** The Skeptic is right that zero implementation is a genuine red flag. The Synthesizer is right that kill gates with objective criteria and PG-only fallback reduce total waste probability. **Consensus: 5.**

### Challenge 3: Auditor → Synthesizer on M5 Rating

> **Auditor:** "You rated M5 as 9/10 standalone. I tested 16 claims and 14 failed challenge testing. The decidability result has a NULL hole. How do you get 9/10 from a contribution where 87.5% of claims don't survive scrutiny?"

> **Synthesizer:** "Because the *idea* is 9/10 and the *execution* is 4/10. These are fixable issues in a correct direction. My 9/10 is for the contribution *after* the fixes."

**Resolution:** The Auditor scores current state; the Synthesizer scores potential after remediation. Both are valid. **Consensus: M5 potential is 8/10; current state is 4/10; expected value after fix ≈ 6-7/10.**

---

## 5. Unanimous Findings (Highest Confidence)

| # | Finding |
|---|---|
| U1 | Engine operational semantics (M1) are genuine new knowledge artifacts worth pursuing |
| U2 | M5 predicate conflict theory is the flagship contribution with real novelty |
| U3 | All 4 FATAL flaws are real, correctly characterized, and must be fixed before implementation |
| U4 | theory_bytes=0 is a tooling/measurement bug (theory content exists but in wrong directory) |
| U5 | The 78K→55K LoC deflation was justified (original was ~30% inflated) |
| U6 | Further deflation to ~38-40K genuinely novel LoC is honest |
| U7 | PostgreSQL is the strongest, most tractable engine to formalize |
| U8 | The theory content is proposal-quality (definitions + proof sketches), not completed proofs |
| U9 | A real-world migration case study is the single highest-impact evaluation addition |
| U10 | CLOTHO positioning must be front-and-center in related work |
| U11 | Lean 4 full mechanization is not worth the schedule risk; 3-5 key lemmas or paper proofs suffice |
| U12 | Oracle and SQLite engine models should remain out of scope |

---

## 6. Recommended Scope: Strategy B+ (Two-Engine)

**Scope:** PostgreSQL SSI + MySQL InnoDB + M5 predicate theory + differential portability + Hermitage validation + novel divergence discovery

| Component | LoC | Priority |
|---|:---:|---|
| PG SSI operational semantics | ~8K | P0 (do first) |
| M5 predicate conflict theory + SMT encoding | ~7K | P0 |
| Refinement checker (PG→Adya) | ~3K | P0 |
| MySQL InnoDB gap-lock semantics | ~6K | P1 (after KG1 passes) |
| Differential portability (PG↔MySQL) | ~4K | P1 |
| Witness synthesis | ~4K | P1 |
| Parser/IR extensions | ~5K | Throughout |
| Lean 4 proofs (PG SSI subset) | ~5K | P2 (nice-to-have) |
| Empirical validation oracle | ~4K | P2 |
| **Total genuinely novel** | **~38K** | |
| Infrastructure (tests, CLI, benchmarks) | ~35K | Throughout |
| **Grand total** | **~73K** | |

### Timeline: 8 months (Month 0 = after FATAL fixes, ~6-10 weeks)

| Month | Milestone | Kill Gate |
|:---:|---|---|
| 0 | FATAL fixes: NULL handling, k=3 proof, SMT prototype, PG memory assumption | **KG0:** SMT feasibility — Z3 solves k=3, n=10 PG SSI in <60s |
| 1-2 | PG SSI model + Hermitage PG validation | **KG2:** PG model ≥95% agreement with real PG16 |
| 3-4 | MySQL InnoDB gap-lock model + M5 predicate theory core | **KG3:** MySQL model ≥90% Hermitage agreement; gap-lock FP <30% |
| 5-6 | Refinement + portability analyzer + divergence discovery | **KG4:** ≥5 novel confirmed divergences |
| 7 | Full evaluation + migration case study | — |
| 8 | Paper writing + artifact preparation | — |

### Kill Gate Actions

| Gate | Kill Criterion | Fallback |
|---|---|---|
| KG0 | Z3 times out (>60s) on PG SSI k=3 encoding | **ABANDON implementation**; publish theory-only at PODS |
| KG2 | PG model <95% agreement on Hermitage workloads | Rework model (+4-6 weeks) or pivot to spec-level analysis |
| KG3 | MySQL gap-lock FP >30% on TPC-C | **Drop MySQL**; publish PG-only at VLDB/PODS |
| KG4 | <3 confirmed novel divergences | Weaken discovery narrative; lead with completeness argument |

---

## 7. Probability Estimates

| Outcome | Current State | After FATAL Fixes | After Full Remediation |
|---|:---:|:---:|:---:|
| Total Abandon | 30-35% | 15-20% | 8-12% |
| PG-only paper (VLDB/PODS) | 25-30% | 35-40% | 20-25% |
| 2-engine paper (SIGMOD) | 15-20% | 30-35% | 45-55% |
| 3-engine paper (SIGMOD) | 5-10% | 8-12% | 15-20% |
| Best paper (any venue) | 2-3% | 8-12% | 12-18% |
| Strong accept (any venue) | 20-30% | 45-55% | 55-65% |
| Any top-venue publication | 45-55% | 65-75% | 75-85% |

---

## 8. Binding Conditions for CONTINUE

**ALL required before committing to full implementation:**

| # | Condition | Type | Deadline |
|---|---|---|---|
| BC1 | Fix FATAL-4: Prototype PG SSI SMT encoding for k=3, n=10; measure Z3 time; must solve in <60s | Kill gate | Week 2 |
| BC2 | Fix FATAL-1: Correct M5 decidability with NULL handling (NOT NULL restriction or co-NP proof) | Theory | Week 3 |
| BC3 | Fix FATAL-3: Correct k-sufficiency proof or downgrade to empirically-validated conjecture | Theory | Week 3 |
| BC4 | Fix FATAL-2: Explicit bounded-memory assumption for PG SSI model | Theory | Week 4 |
| BC5 | CLOTHO differentiation: concrete comparison paragraph with shared workload example | Writing | Week 1 |
| BC6 | LLM positioning: one paragraph explaining formal completeness advantage over LLM consultation | Writing | Week 2 |

**If BC1 fails (Z3 cannot handle encoding): ABANDON IMMEDIATELY.**

---

## 9. VERDICT

### **CONDITIONAL CONTINUE** — Composite 29/50 — Confidence 72%

| Dimension | Score |
|---|:---:|
| Extreme Value | 6 |
| Genuine Software Difficulty | 6 |
| Best-Paper Potential | 5 |
| Laptop-CPU / No-Humans | 7 |
| Feasibility | 5 |
| **Composite** | **29/50** |

### Why NOT Abandon

- Engine operational semantics are genuinely novel artifacts with lasting citation value
- M5 predicate conflict theory, if fixed, is a real theoretical contribution (8/10 potential)
- The problem (migration isolation safety) is real, even if the market is narrower than claimed
- ~38K novel LoC of serious formal methods + database internals work is non-trivial
- No fundamental impossibility — all four FATAL flaws are fixable in principle
- Clear fallback to PG-only paper (publishable at VLDB/PODS) limits downside

### Why NOT Unconditional

- Four FATAL flaws with zero currently fixed
- Zero completed proofs, zero implementation, zero validated models
- Flagship theorem (M5) is unsound as stated
- CLOTHO positioning challenge is real and may be fatal at review if mishandled
- LLM threat deflates the practitioner value story
- Three rounds of descoping (5→3 engines, full Lean→3-5 lemmas, full cross-product→differential) and still four FATAL flaws
- theory_bytes=0 is a measurement bug, but the measured quality is accurate: the "theory" is proposal-quality, not proved results

### The Bottom Line

This project has a real contribution buried inside an oversold narrative. Strip the marketing, fix the math, validate the models against real engines, and it's a solid publication. Keep the oversell, ignore the flaws, and it's a desk rejection.

**The single most important action:** Prototype the PG SSI SMT encoding and run Z3 on it. If that works in <60s, commit to Strategy B+. If it doesn't, stop.

---

## 10. Expert Calibration Notes

- **Independent Auditor:** Best-calibrated overall (transparent methodology, reproducible estimates). Identified theory_bytes=0 measurement anomaly. Correct on 14/16 challenge test failure rate.
- **Fail-Fast Skeptic:** Excelled at risk identification (LLM threat, discovery circularity, LoC deflation trajectory). Somewhat overpunitive on Value (5) and Best-Paper Potential (4) — these assume worst-case scenario where no FATAL fix succeeds.
- **Scavenging Synthesizer:** Provided the strongest constructive path forward (Strategy B+ with kill gates). Component value ratings well-calibrated. Feasibility (6) slightly optimistic given zero-implementation starting point, but correctly identified that PG-only fallback dramatically reduces total-waste probability.

**Lead adjudication methodology:** Where experts disagreed, median was used as baseline, then adjusted toward the best-evidenced position. Where all agreed, finding was recorded at HIGH confidence. The 6-point spread (27-33) in composites is moderate, indicating genuine uncertainty about the project's trajectory but agreement on the direction of the verdict.

---

*Evaluation produced by 3-expert adversarial panel: Independent Auditor (evidence-based scoring), Fail-Fast Skeptic (aggressive rejection testing), Scavenging Synthesizer (salvage value identification). Cross-critique synthesis adjudicated by Team Lead. All experts independently recommended CONDITIONAL CONTINUE.*

# Community Expert Verification: IsoSpec (txn-isolation-verifier)

**Proposal:** proposal_00 — IsoSpec: Verified Cross-Engine Transaction Isolation Analysis via Engine-Faithful Models and Differential Portability Checking  
**Evaluator:** Community Expert Panel (area-086-data-management-and-databases)  
**Panel:** Independent Auditor (IA), Fail-Fast Skeptic (FFS), Scavenging Synthesizer (SS)  
**Method:** Independent proposals → adversarial cross-critique → consensus synthesis → independent verifier signoff  
**Date:** 2026-03-08

---

## Consensus Scores

| Pillar | IA | FFS | SS | Consensus | Verdict |
|--------|:--:|:---:|:--:|:---------:|---------|
| **1. Extreme & Obvious Value** | 6 | 3 | 7.5 | **6/10** | Real problem, real gap, zero production incident evidence |
| **2. Genuine Difficulty** | 6 | 4 | 7.5 | **6.5/10** | ~35-40K genuinely novel LoC (2-engine scope); engine models + M5 are PhD-level |
| **3. Best-Paper Potential** | 5 | 3 | 7 | **5.5/10** | M5 genuinely novel; discovery framing promising but unvalidated |
| **4. Laptop-CPU + No Humans** | 7 | 5 | 8.5 | **7.5/10** | Z3 bounded analysis tractable; Docker sequential feasible; SMT perf unvalidated |
| **5. Feasibility** | 5 | 2 | 6.5 | **4.5/10** | 1.75-2.2x resource gap for solo contributor; kill-gates essential |
| **Composite** | **29/50** | **17/50** | **37/50** | **30/50** | **CONDITIONAL CONTINUE (2-1; Skeptic dissents ABANDON)** |

---

## Pillar 1: Extreme & Obvious Value — 6/10

**What's real:** Database migration isolation bugs are a genuine, well-documented problem. The Hermitage project (Kleppmann, 2014) catalogs 47+ behavioral divergences across major SQL engines. AWS DMS, Azure DMS, and Google Cloud DMS handle schema/data transfer but perform **zero** isolation semantics verification. No existing tool answers: "Will my transaction break when I migrate from engine E₁ to E₂, and here is a runnable script proving it?" This is a verified capability gap.

**What's missing for a higher score:**
- **Zero concrete production incidents cited.** The proposal claims "database migration is one of the leading causes of production outages and data corruption incidents" but provides no CVEs, postmortems, or bug reports. Hermitage documents *differences*, not *failures caused by those differences*. In 12 years since Hermitage, no published case study attributes a production failure to isolation-level divergence during migration. This gap between "engines behave differently" and "this causes real harm" is asserted, not evidenced.
- **LLM competitive threat.** The Skeptic claims LLMs cover 80% of practitioner needs; the Auditor rebuts this as an invented number. Consensus: LLMs cover ~40-50% of one-off migration consultations (enumerating known differences). IsoSpec's durable differentiators are formal completeness guarantees, CI/CD-integrated regression testing, witness synthesis, and systematic discovery. The paper must include a "Why not just use an LLM?" section.
- **Adoption barriers.** DBAs don't use formal verification tools. The witness-script output (runnable SQL tests) partially mitigates this. The paper should frame the tool as "automated migration test generation," not "formal verification."

**Panel disagreement:** The Skeptic (3/10) argues that without a single documented production incident, the value is speculative. The Synthesizer (7.5/10) argues the $10B migration market and CI/CD integration angle justify high value. **Consensus follows the Auditor (6/10):** the gap is real, but the harm is undemonstrated.

---

## Pillar 2: Genuine Difficulty — 6.5/10

**Honest LoC assessment (2-engine Scenario B scope, audited by 3 experts):**

| Subsystem | Novel LoC | Difficulty Tier | Notes |
|-----------|:---------:|:---------------:|-------|
| Engine models (PG SSI + MySQL InnoDB) | ~12K | **PhD-level** | Core diamond. Cannot be shortcut. |
| M5 predicate conflict theory | ~4-5K | **PhD-level** | Flagship theorem; decidability boundary characterization |
| SMT encoding of engine semantics | ~6-7K | **Advanced engineering** | Known BMC patterns; engine constraint content is novel |
| Refinement checker | ~3-4K | **Advanced engineering** | Known theory; novel application to isolation domain |
| Portability analyzer | ~4-5K | **Moderate** | Differential SMT, 2 migration pairs |
| SQL parser + IR | ~5-6K | **Moderate** | sqlparser-rs base; dialect extensions and isolation-aware IR |
| Witness synthesis | ~3-4K | **Moderate** | MUS extraction standard; dialect-specific SQL generation |
| Lean 4 proofs (2 key lemmas) | ~2K | **High-variance** | Descoped from 5K; mechanized proofs for 2 lemmas only |
| Validation oracle | ~3-4K | **Moderate** | Docker infrastructure + interleaving test design |
| **Total genuinely novel** | **~42-50K** | | Budget ~45K as planning estimate |
| Infrastructure (CLI, tests, CI) | ~40K | Standard | Not counted toward difficulty |

**What's genuinely hard:** The engine models are the core. Modeling PostgreSQL 16.x SSI (SIREAD locks, rw-dependency tracking, dangerous structure detection, read-only optimization) and MySQL 8.0 InnoDB's index-dependent gap locking requires deep database-internals expertise at the PhD level. No published paper provides implementation-faithful operational semantics for either engine. The SMT encoding of gap-lock ranges and SSI dependency structures over QF_LIA+UF+Arrays is non-trivial. M5's decidability boundary characterization for conjunctive inequalities is genuinely new.

**What's inflated:** The LoC estimate has been revised three times (78K → 55K → 60K → 45K for Scenario B). Test suites, CLI infrastructure, and parser boilerplate over sqlparser-rs should not count as novel. Lean proofs are math, not software difficulty. The ~45K for 2-engine scope includes ~15-17K of hard code, ~15K of advanced engineering, and ~13-15K of moderate engineering using known techniques.

**Panel disagreement:** The Skeptic (4/10) counts only ~17K of hard code and argues the rest is consultant-replicable. The Auditor (6/10) corrects this to ~34K of genuine difficulty. The Synthesizer (7.5/10) counts the full 55K. **Consensus: 6.5/10** — the engine models + M5 are genuine PhD-level contributions, but ~30% of the novel code applies known techniques.

---

## Pillar 3: Best-Paper Potential — 5.5/10

**Math novelty audit (consensus):**

| Contribution | Consensus Rating | Best-Paper Relevance |
|---|---|---|
| **M5: Predicate-Level Conflict Theory** | Genuinely novel (Hard) — unanimous ✓ | **HIGH** — flagship theorem |
| **M1: Engine Operational Semantics** | Novel artifacts, not new technique (Hard) | **HIGH** — discovery instruments |
| **M2: Isolation Refinement Relation** | Novel application of known theory (Moderate) | **MEDIUM** — connecting bridge |
| M6: Mixed-Isolation Optimization | Practical bonus (Moderate) | LOW — appendix material |
| M3, M4, M7, M8 | Supporting lemmas | LOW — do not highlight |

**Genuine math contributions: 2-3.** M5 is the flagship. M1+M2 together constitute a framework contribution. Everything else is supporting.

**Discovery framing is the best-paper angle.** The Synthesizer correctly identifies that "Our models revealed N surprising behaviors about production SQL engines" is the compelling narrative, not "We built a verification tool." However, discovery results don't exist yet — scoring based on projected outcomes (as the Synthesizer does at 7/10) is premature. The Auditor's current-state score of 5 is more honest; consensus adds 0.5 for the legitimate strength of the discovery framing as a strategy.

**CLOTHO positioning risk:** The parse→encode→solve pipeline is architecturally identical to CLOTHO (OOPSLA'19). The project's own crystallization critique states the pipeline is "literally the CLOTHO pipeline with different front-end and isolation models." The differentiation (engine-specific models vs. abstract specs) is real but must be front-and-center: **"CLOTHO models what the specification says; IsoSpec models what the engine does."** A reviewer who misses this distinction will desk-reject.

**Venue analysis:**
- **SIGMOD 2028** (primary): Discovery framing + bridge paper. Needs 8+ novel confirmed divergences and migration case study for best-paper consideration.
- **CAV 2027** (stretch): Formal verification angle. Strong-accept caliber if M5 is clean and engine models validated. Best paper unlikely.
- **VLDB 2027** (fallback): Systems/tool angle. Good fit for 2-engine comparison + evaluation.

**Panel disagreement:** The Skeptic (3/10) argues SIGMOD never awards FM tools and M5 has FATAL flaws. The Synthesizer (7/10) argues the discovery framing compensates. **Consensus: 5.5/10** — solid math core + promising framing, but discoveries are hypothetical and the CLOTHO risk is real.

---

## Pillar 4: Laptop-CPU + No-Humans — 7.5/10

**Z3 feasibility:** QF_LIA+UF+Arrays at k≤3 is Z3's sweet spot. Engine-specific constraints actually *prune* the search space relative to abstract models — lock compatibility matrices and version visibility rules eliminate infeasible interleavings. Constraint sizes for typical workloads (~1000-5000) are well within Z3's tractability envelope.

**Docker feasibility:** PostgreSQL (~200MB) + MySQL (~400MB) = ~600MB for 2-engine Scenario B. Sequential testing on 16GB RAM is trivially feasible.

**Lean 4:** CPU-only, batch-mode, completes in minutes for 2K LoC proofs. No concern.

**Deductions:**
- **FATAL-4 (SMT performance unvalidated):** Zero experimental data exists. The "sub-30-second" claim is speculation. Must be validated in Month 1. Prevents scoring 8+.
- **Gap-lock disjunctions:** MySQL's "union over all indexes" generates disjunctive constraints. For tables with 3+ secondary indexes, Z3 may face exponential blowup. Needs over-approximation engineering.
- **Docker interleaving flakiness:** Advisory-lock-based forcing has **25-35% failure rates** (not the 10% originally claimed). Manageable via statistical retry (100 runs, ≥10 successes) but slower than projected.
- **Human judgment (~10%):** Novelty assessment of discoveries requires expert classification. Core analysis pipeline is fully automated.

---

## Pillar 5: Feasibility — 4.5/10

**This is the project's weakest pillar and the site of the largest panel disagreement (spread: 4.5 points).**

**Resource gap analysis (corrected from Skeptic's 12x):**
- The Skeptic's 12x gap uses Brooks' 1970s OS-kernel productivity rates (10-50 LoC/day). This is methodologically unsound for 2026 Rust with modern library ecosystem.
- The Auditor's corrected blended rate: 80-100 LoC/day for formal-methods-adjacent Rust with substantial library support.
- For 2-engine Scenario B: ~45K novel LoC at 90 LoC/day = ~500 person-days.
- Available time (10-12 months, ~220-260 working days): **1.9-2.3x resource gap.**
- This is survivable with extended hours (common for PhD students) and aggressive scope discipline, but leaves no margin for error.

**FATAL flaw remediation cost:** 4-8 weeks of rework before implementation begins. On a 10-12 month timeline, this is 15-20% of available time consumed by fixing theory-phase gaps.

**theory_bytes=0 anomaly:** This is a pipeline bookkeeping bug, not a substantive issue. The theory phase produced ~176KB of formal specifications, algorithm designs, and evaluation strategies across 7 documents. The State.json counter was never incremented. Impact: none on evaluation, but raises process quality concern.

**Timeline assessment:**
- The final_approach's 11-month estimate is optimistic by 2-4 months.
- **SIGMOD 2027 is unreachable** given 0 LoC and 10-14 month realistic implementation timeline.
- **CAV 2027 (January submission) is a stretch** — requires strict Scenario B scope with no risk triggers.
- **SIGMOD 2028 is the realistic primary target.**

**Kill-gate sequence (fail fast):**

| Gate | Month | Criterion | Kill Action |
|------|:-----:|-----------|-------------|
| KG0 | 0 | 2-week production incident search | If none found: downgrade narrative to pure discovery/formal-methods (no industry-impact claims) |
| KG1 | 2 | PG SSI model passes Hermitage PG tests (≥95%) AND Z3 solves k=3, n=10 in <60s | ABANDON if PG model doesn't converge or Z3 times out |
| KG2 | 4 | MySQL model achieves ≥90% Hermitage agreement AND gap-lock FP rate <25% | CUT MySQL if model won't converge; pivot to PG-only |
| KG3 | 7 | ≥5 novel divergences confirmed AND portability analyzer produces first violations | CUT discovery target if <5; pivot to verification-only framing |
| KG4 | 9 | Evaluation complete; paper draft exists | Ship what exists |

---

## Fatal Flaws

| # | Flaw | Severity | Mitigation Status | Residual Risk |
|---|------|----------|:--:|:---:|
| FF1 | **NULL handling breaks M5 decidability** — SQL three-valued logic invalidates "convex polytope" argument | FATAL | **Mitigated** — restrict to NOT NULL columns (covers 95%+ of OLTP key columns) | LOW |
| FF2 | **PG SSI model ignores memory pressure** — granularity promotion, lock cleanup, summarization under memory pressure | FATAL | **Partially mitigated** — bounded-memory assumption, treat summarization as over-approximation | MEDIUM |
| FF3 | **k=3 sufficiency proof has mathematical errors** — G1a miscounted, G2-item 6-txn miscounted as 3-txn | FATAL | **Partially mitigated** — errors in sketch, not in result; needs rigorous reproof | MEDIUM-HIGH |
| FF4 | **SMT performance entirely unvalidated** — "sub-30-second" claims have zero experimental support | FATAL | **Unmitigated** — deferred to Month 1 kill-gate | HIGH |
| S1-S10 | Various SERIOUS issues (MySQL gap-lock unsoundness, SQL Server gaps, MUS confusion, MaxSMT timeouts, etc.) | SERIOUS | Most mitigated via scoping decisions | LOW-MEDIUM |

**Are there showstoppers?** No individual flaw is fatal given the documented mitigations. But the *compound* risk of 4 FATAL flaws, each with independent failure probability, is significant. The Skeptic's compound probability estimate (~1.3% success) uses worst-case assumptions throughout, but the concern about correlated optimism across pillars is valid. The kill-gate sequence is the correct response: fail fast on the highest-risk items (KG1 at Month 2 validates FF4 and engine model fidelity simultaneously).

---

## Amendments Required

### Mandatory (blocking — must be applied before implementation)

**A1: Restrict M5 to NOT NULL columns.** Predicate conflict theory applies to NOT NULL columns with sound over-approximation for nullable columns. Explicit limitation statement required. (Fixes FF1)

**A2: Add bounded-memory assumptions to PG SSI model.** Model assumes SIREAD lock memory within `max_pred_locks_per_transaction` default (64). Summarization treated as sound over-approximation. (Fixes FF2)

**A3: Rigorous k=3 reproof.** Formal proof by structural induction on Adya anomaly class definitions. For predicate-level G2: explicit bounded completeness statement. Must be done before implementation. (Fixes FF3)

**A4: Month 1 SMT validation.** Benchmark Z3 on 50 PG SSI workloads at k=3, n={5,10,15}. Report actual times. Revise all performance claims to evidence-based numbers. KG1 kill-gate at 60s threshold. (Addresses FF4)

**A5: CLOTHO architectural acknowledgment.** Front-and-center in paper, paragraph 1-2: "We extend CLOTHO's parse→encode→solve verification architecture from abstract consistency models to engine-specific implementation semantics." Differentiation: "CLOTHO models what the spec says; IsoSpec models what the engine does."

**A6: Scope stored procedures to future work.** V1 handles pure SQL transactions. PL/pgSQL, T-SQL, MySQL stored procs deferred.

**A7: Restructure math contributions.** Present as: 1 flagship theorem (M5), 1 framework contribution (M1+M2), 1 practical bonus (M6). Relegate M3/M4/M7/M8 to appendices. Not "8 equal math contributions."

**A8: Address LLM competitive positioning.** Mandatory paper section explaining why formal verification adds value beyond LLM consultation: completeness guarantees for bounded workloads, CI/CD integration, witness reproducibility, regression testability, novel discovery beyond training data.

### Recommended (best-paper path)

**A9: Add real-world migration case study.** Find 1-3 documented production migration issues from PG mailing lists, AWS DMS bug reports, or Stack Overflow. Show IsoSpec would have caught them. 2-week focused search as KG0. Single highest-impact amendment for best-paper consideration.

**A10: Discovery-first narrative reframe.** "We built formal models of how PostgreSQL and MySQL actually implement isolation, and discovered N surprising behaviors — including K that silently affect common migration paths."

**A11: Descope Lean proofs aggressively.** Cap at 2 mechanized key lemmas (~2K LoC). If any single lemma takes >3 weeks, drop to paper proofs. Every hour in Lean is an hour not finding divergences.

---

## Panel Recommendation

**CONDITIONAL CONTINUE.** (2-1 majority: Auditor + Synthesizer CONTINUE; Skeptic dissents ABANDON.)

The proposal contains a genuine diamond — engine-specific operational semantics that have never existed as formal artifacts, connected via refinement to Adya's isolation specifications, with a discovery framing that turns models into scientific instruments. M5 predicate conflict theory is a genuinely novel mathematical contribution. The migration portability gap is real. The 2-engine scope (PG SSI + MySQL InnoDB) captures the most fundamental divide in concurrency control philosophy (optimistic/dependency-tracking vs. pessimistic/gap-locking).

However, the proposal is oversold by ~30% on LoC, ~60% on math contributions, has 4 FATAL flaws (2 mitigated, 2 partially mitigated), and faces a 1.9-2.3x resource gap for a solo contributor. The SIGMOD 2027 target is unreachable. Zero implementation code exists.

**Probability estimates (consensus, conditioned on binding conditions being met):**
- P(best-paper at SIGMOD 2028) ≈ 4-8%
- P(strong accept at SIGMOD/CAV) ≈ 40-55%
- P(any publication at top venue) ≈ 60-70%
- P(ABANDON at next gate) ≈ 20-30%

**Skeptic's dissent (recorded):** The Skeptic scores 17/50 and recommends ABANDON. Core arguments: (1) zero evidence of real-world harm from the problem; (2) flagship theorem M5 has proof-invalidating flaws; (3) 1.9-2.3x resource gap even after correction from 12x; (4) CLOTHO positioning makes this incremental; (5) compound probability of all optimistic assumptions holding is <5%. The Skeptic would upgrade to CONDITIONAL CONTINUE if: (a) one production incident is found, (b) SMT feasibility passes KG1, (c) FATAL-1 NULL fix is demonstrated on TPC-C predicates.

**Synthesizer's note (recorded):** The Synthesizer recommends Scenario B (PG + MySQL, 7-8 months). The two-engine comparison may produce a *stronger* paper than the three-engine survey — SSI vs. gap-locking is the most fundamental divergence in concurrency control philosophy. The Lean proofs are the most expendable component. Discovery count matters less than discovery quality: one migration-affecting divergence is worth ten edge-case lock-ordering differences.

**The single most impactful change:** A 2-week focused search for production migration incidents (A9 / KG0). One documented case where isolation divergence caused data corruption during migration is worth more than any amount of theoretical analysis or benchmark results. This is the difference between P(best-paper) ≈ 4% and P(best-paper) ≈ 10%.

---

## Salvage Scenarios

### Scenario A: Full 2-Engine Execution (10-12 months)
- PG SSI + MySQL InnoDB models, full M5, discovery campaign (8-12 targets), migration case study, 2 Lean lemmas
- Target: SIGMOD 2028 (primary), VLDB 2027 (fallback)
- P(completion): 55-65%, P(top-venue accept): 55-65%, P(best-paper): 4-8%

### Scenario B: Descoped 2-Engine (7-8 months) — Synthesizer's recommendation
- Same as A but Lean proofs paper-only, reduced discovery target (5-8), CAV 2027 stretch
- P(completion): 65-75%, P(top-venue accept): 50-60%, P(best-paper): 3-6%

### Scenario C: PG-Only Emergency Salvage (3-4 months)
- PostgreSQL 16.x SSI model only, item-level M5 (no predicates), 5+ PG-specific discoveries
- Target: DBTest @ SIGMOD 2027 (workshop), PVLDB Demo track
- P(completion): 85-90%, P(workshop accept): 80-90%, P(full venue accept): 20-30%

---

## Expert Calibration Notes

**Independent Auditor** was best-calibrated overall: transparent methodology, evidence-cited scores, reproducible probability estimates. The Auditor correctly identified the theory_bytes=0 anomaly as bookkeeping, the CLOTHO positioning risk as serious-but-manageable, and the timeline as 14-16 months (too optimistic; revised to 10-12 months for descoped Scenario B).

**Fail-Fast Skeptic** excelled at risk identification (compound probability analysis, LoC productivity math, kill questions) but committed three methodological errors: Brooks' 1970s productivity rates applied to 2026 Rust (corrected from 12x to 1.9-2.3x gap), invented "80% LLM coverage" figure (corrected to 40-50%), and scoring feasibility based on the full 3-engine scope while the recommendation is for 2-engine Scenario B.

**Scavenging Synthesizer** provided the best constructive path forward (Scenario B, kill-gate sequence, FATAL flaw salvage paths) and correctly identified the discovery framing as the best-paper angle. However, the Synthesizer scored based on projected outcomes rather than current state (P3=7 with no discoveries in hand) and underweighted both the FATAL flaw compound risk and the resource gap.

---

## Verdict Summary

A genuine diamond in ~30% oversell. The engine-specific operational semantics are the load-bearing novelty — protect their quality and depth at the expense of breadth. Fix the FATAL flaws, validate SMT performance in Month 1, search for a production incident in Week 1, acknowledge CLOTHO as architectural ancestor, restructure math around M5 as flagship, and this is a strong SIGMOD/CAV paper. Without restructuring, a reviewer who knows CLOTHO + Biswas & Enea will reject.

**Decision: CONDITIONAL CONTINUE — mandatory amendments A1-A8, kill-gates KG0-KG4, Scenario A as target with fallback to B then C.**

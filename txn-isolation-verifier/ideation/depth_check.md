# Verification Depth Check: IsoSpec

**Proposal:** IsoSpec: Verified Cross-Engine Transaction Isolation Analysis via Implementation Refinement  
**Slug:** `txn-isolation-verifier`  
**Panel:** 3-expert team (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer)  
**Method:** Independent proposals → adversarial cross-critique → consensus synthesis  
**Date:** 2026-03-08

---

## Consensus Scores

| Pillar | Score | Verdict |
|--------|:-----:|---------|
| **1. Extreme & Obvious Value** | **7/10** | Real problem, real gap, formal approach needs concrete incident evidence |
| **2. Genuine Difficulty** | **7/10** | ~55K genuinely novel LoC (not 78K claimed); engine formalization is PhD-level work |
| **3. Best-Paper Potential** | **6/10** | Strong accept at SIGMOD/CAV; needs amendments for best-paper tier |
| **4. Laptop CPU + No Humans** | **8/10** | Z3 bounded analysis tractable; Docker sequential testing feasible |
| **Composite** | **28/40** | CONDITIONAL CONTINUE with mandatory amendments |

---

## Pillar 1: Extreme & Obvious Value — 7/10

**What's real:** Database migration isolation bugs are a genuine, well-documented problem. The Hermitage project (Kleppmann, 2014) documents 47+ behavioral divergences across major SQL engines. Oracle-to-PostgreSQL migration is a massive industry trend driven by licensing costs. AWS/Azure/Google migration services handle schema/data transfer but perform zero isolation semantics verification. No existing tool answers: "Will my transaction break when I migrate from engine A to engine B?" Cross-engine differential analysis is a true capability gap.

**What's missing for a higher score:**
- The proposal cites **zero concrete production incidents** of isolation-caused migration failures. Hermitage documents *differences*, not *failures caused by those differences*. The gap between "engines behave differently" and "this routinely causes billion-dollar bugs" is asserted, not evidenced.
- The **LLM threat** is unaddressed. In 2026, an LLM can enumerate common isolation differences between any engine pair. For one-off migration consultations, LLM+Hermitage+testing covers ~80% of practitioner needs. Formal verification adds value for CI/CD-integrated, systematic, completeness-guaranteed analysis — but this must be explicitly argued.
- **Adoption barriers** for formal verification tools in DBA communities are high. The actionable output (runnable SQL test scripts) partially mitigates this, but the proposal oscillates between "tool for DBAs" and "framework for PL researchers."

**Panel disagreement:** Auditor scored 6 (too harsh — dismisses completeness guarantees for safety-critical workloads); Synthesizer scored 8 (too generous — no incident evidence). Consensus: 7.

---

## Pillar 2: Genuine Difficulty — 7/10

**Honest LoC recount (audited by 3 experts):**

| Subsystem | Claimed Novel | Consensus Novel | Rationale |
|-----------|:---:|:---:|---|
| Engine models (5 engines) | 31K | **31K** | Unanimous: genuine, cannot be shortcut |
| Transaction IR | 12K | **7K** | IR design is novel; AST infrastructure is not |
| SQL parser extensions | 12K | **6K** | sqlparser-rs handles base SQL; dialect extensions real but bounded |
| SMT encoding | 12K | **8K** | Technique is standard BMC; engine-specific constraint content is novel |
| Refinement checker | 8K | **5K** | Known theory; application to isolation + decision procedure is novel |
| Portability analyzer | 10K | **6K** | Differential SMT standard; cross-engine formulation is novel |
| Witness synthesis | 10K | **4K** | MUS extraction standard; dialect-specific SQL generation is real |
| Mixed-isolation optimizer | 5K | **3K** | MaxSMT off-the-shelf; cost function formulation is novel |
| Lean 4 proofs | 12K | **9K** | PG SSI fully mechanized is genuine |
| Validation oracle | 6K | **3K** | Docker orchestration is infra; interleaving test design is novel |
| **Total genuinely novel** | **~78K** | **~55K** | **30% inflation over honest count** |

Additional ~15K of "novel application of known techniques" that reasonable people could argue either way. ~80K infrastructure/tests/CLI.

**What's genuinely hard:** The engine models (31K) are the core. Modeling PostgreSQL SSI (SIREAD locks, rw-dependency tracking, dangerous structure detection, read-only optimization), MySQL InnoDB's index-dependent gap locking, and SQL Server's dual-mode concurrency control requires deep domain expertise at the database-internals level. This is multi-year PhD-level formalization work. The SMT encoding of gap-lock ranges and SSI dependency structures over QF_LIA+UF+Arrays is non-trivial. The Lean 4 mechanization of Cahill's dangerous structure theorem is serious proof engineering.

**What's inflated:** Test suite (32K), CLI infrastructure, parser boilerplate over sqlparser-rs, and standard BMC techniques were counted toward the "novel" total. The 78K→55K correction represents typical academic optimism, not dishonesty, but must be corrected.

---

## Pillar 3: Best-Paper Potential — 6/10

**Math novelty audit (consensus):**

| Contribution | Claimed | Consensus Rating |
|---|---|---|
| M1: Engine Operational Semantics | New math (Hard) | Solid modeling contribution — novel artifacts, not new theory |
| M2: Isolation Refinement Relation | New theory (Hard) | Novel application of known theory (Moderate) |
| M3: Portability coNP-completeness | New result (Moderate) | Expected — defining a problem and proving it coNP-complete is routine |
| M4: Symbolic Engine Encoding | New technique (Hard) | Hard engineering within known BMC framework |
| **M5: Predicate-Level Conflict Theory** | **New theory (Hard)** | **Genuinely novel (Hard) — unanimous ✓** |
| M6: Mixed-Isolation Optimization | New formulation (Moderate) | Reasonable application (Moderate) |
| M7: Bounded Soundness/Completeness | New characterization (Moderate) | Minor lemma — k=3 follows from anomaly definition structure |
| M8: Compositionality Theorem | New theorem (Moderate) | Standard refinement property — almost definitional once refinement exists |

**Genuine math contributions: 2–3.** M5 is the flagship. M1+M2 together constitute a framework contribution. M6 is a practical bonus. M3/M4/M7/M8 are supporting lemmas or engineering.

**Venue analysis:**
- **SIGMOD/VLDB:** Best fit. Bridge between formal methods and database systems. Strong-accept caliber for the tool + evaluation. Best paper requires discovering novel engine divergences beyond Hermitage.
- **CAV:** The SMT encoding and bounded verification story fits. The engine models as formal artifacts are interesting. Strong accept; best paper unlikely.
- **PLDI/POPL:** Refinement theory too standard. Borderline accept.

**What would elevate to 7+:**
1. A **real-world migration case study** demonstrating IsoSpec catches an actual production bug (single highest-impact amendment).
2. **Discovering novel engine divergences** beyond Hermitage (turns IsoSpec from verification tool to discovery tool).
3. **Narrative reframe:** "Our models revealed X surprising behaviors" before "and we built a tool."

**The CLOTHO comparison problem:** A reviewer who thinks "this is CLOTHO for SQL engines" will reject regardless of depth. The differentiation must be front-and-center: "CLOTHO models what the spec says; we model what the engine does."

---

## Pillar 4: Laptop CPU + No Humans — 8/10

**Z3 feasibility:** QF_LIA+UF+Arrays is well-optimized in Z3. For bounded analysis (k=3 transactions covers all Adya anomaly classes G0–G2), constraint sizes are manageable. Engine-specific constraints actually *prune* the search space relative to abstract models — lock compatibility matrices and version visibility rules eliminate infeasible interleavings.

**Docker feasibility:** PostgreSQL (~200MB), MySQL (~400MB), SQL Server (~1.5GB), Oracle XE (~2GB), SQLite (no container). Total ~4.1GB. Sequential testing on 16GB RAM is fine. Simultaneous testing of all 5 is tight but unnecessary.

**Lean 4:** CPU-only, batch-mode, completes in minutes for 12K LoC proofs. No concern.

**Deductions:**
- Gap-lock encoding creates disjunctions over possible index choices. For moderate transaction counts, Z3 handles this, but pathological cases could timeout. Needs careful over-approximation engineering.
- Docker-based interleaving forcing (advisory locks + thread barriers) is **inherently flaky** for fine-grained concurrency behaviors. The validation oracle will have false negatives. This is manageable (the static analysis is the primary contribution) but must be stated as a limitation.

---

## Fatal Flaws

| # | Flaw | Severity | Mitigation |
|---|------|----------|------------|
| FF1 | **Engine model fidelity unvalidated** — Models claim "implementation-faithful" but no formal adequacy criterion; behavior changes across engine minor versions | **HIGH** | Pin to specific engine versions (PG 16.x, MySQL 8.0, SQL Server 2022). Define formal adequacy criterion: behavioral agreement on bounded workloads. Adopt sound over-approximation for index-dependent behaviors. |
| FF2 | **Engine version fragility** — PG 14 SSI ≠ PG 16 SSI; MySQL 5.7 gap locks ≠ MySQL 8.0; models are point-in-time snapshots | **HIGH** | Explicitly version-pin. Models are versioned artifacts, not timeless abstractions. State this as a limitation. |
| FF3 | **Prior art positioning** — CLOTHO architecture (parse→encode→solve) is identical; differentiation is content (engine models), not method | **MEDIUM** | Acknowledge CLOTHO as architectural ancestor. Differentiate on engine-specific models + refinement bridge, not pipeline novelty. |
| FF4 | **Oracle model is speculative** — Oracle concurrency control is proprietary, underdocumented; model based on black-box observations + limited documentation | **MEDIUM** | Label Oracle model as "approximate, based on observed behavior." Lower fidelity tier vs open-source engines. |
| FF5 | **LoC novelty claim inflated 30%** — 78K includes test suite/infrastructure counted as novel | **MEDIUM** | Recount to ~55K genuinely novel. Separate novel code, novel applications, and infrastructure. |
| FF6 | **Math novelty overclaimed** — "8 math contributions" is honestly 2–3 genuine + 5 supporting lemmas | **MEDIUM** | Restructure around M5 (flagship) + M1+M2 (framework) + M6 (bonus). Relegate M3/M4/M7/M8 to appendices. |
| FF7 | **Stored procedure/trigger scope** — Full analysis of PL/pgSQL, T-SQL, PL/SQL, MySQL stored procs is 4 different programming languages | **LOW-MEDIUM** | Scope v1 to pure SQL transactions. Stored procs as future work. |

**Are there showstoppers?** No. Both HIGH-severity flaws (FF1, FF2) are mitigatable with concrete, bounded work (version pinning + adequacy criterion). But they MUST be addressed before submission.

---

## Amendments Required

### Mandatory (blocking — must be applied to problem statement)

**A1: Version-pin all engine models.** Every engine model specifies exact version: "PostgreSQL 16.x SSI," "MySQL 8.0 InnoDB gap locking," etc. Models are versioned artifacts with explicit adequacy criteria.

**A2: Recount LoC honestly.** ~150K total, ~55K genuinely novel, ~15K novel application of known techniques, ~80K infrastructure/tests/CLI. Drop test suite from novelty claim.

**A3: Restructure math contributions.** Present as: 1 flagship theorem (M5: predicate-level conflict theory), 1 framework contribution (M1+M2: engine semantics + refinement), 1 practical bonus (M6: mixed-isolation optimization), 5 supporting lemmas (M3, M4, M7, M8). Not "8 equal math contributions."

**A4: Acknowledge CLOTHO architecture.** "We adopt CLOTHO's parse→encode→solve architecture and extend it from abstract consistency models to engine-specific operational models connected via refinement to standard isolation specifications."

**A5: Address engine model fidelity.** Define formal adequacy criterion. Adopt sound over-approximation for optimizer-dependent behaviors (MySQL gap locks). Label Oracle model as approximate.

**A6: Scope stored procedures to future work.** V1 handles pure SQL transactions. Stored procedure/trigger analysis is future work.

### Recommended (best-paper path)

**A7: Add real-world migration case study.** Find 1–3 documented production migration failures from PostgreSQL mailing lists, AWS DMS bug reports, or Stack Overflow. Show IsoSpec would have caught them.

**A8: Discover novel engine divergences.** Run IsoSpec beyond Hermitage patterns. Target stored procedure interactions, trigger-induced anomalies, multi-table FK transactions. Headline: "IsoSpec discovers N previously undocumented isolation divergences."

**A9: Reposition narrative as discovery-first.** "We built formal models of 5 engines and discovered N surprising behaviors. We then built IsoSpec to verify portability."

**A10: Address LLM comparison.** One paragraph explaining why formal verification provides CI/CD-integrated, completeness-guaranteed, regression-testable analysis that LLM consultation cannot.

---

## Panel Recommendation

**CONDITIONAL CONTINUE.** The proposal contains a genuine diamond — engine-specific operational semantics connected via refinement to Adya spec-level isolation is a real, unsolved problem with practical impact. The core contribution (engine models + refinement + portability analysis + M5 predicate theory) is novel, hard, and valuable. However, the proposal is oversold by ~30% on LoC, ~60% on math contributions, and lacks the real-world evidence needed for best-paper consideration.

**Probability estimates (consensus):**
- P(best-paper at SIGMOD) ≈ 8–12% (with amendments A7–A9 applied)
- P(best-paper at SIGMOD) ≈ 2–4% (without amendments)
- P(strong accept at SIGMOD or CAV) ≈ 55–65%
- P(any publication at top venue) ≈ 70–80%
- P(ABANDON at next gate) ≈ 15–20%

**The single most impactful change:** Add a real-world migration case study (A7). One documented production bug that IsoSpec would have caught is worth more than any amount of benchmark results. This is the difference between best-paper and strong-accept.

**Expert calibration note:** The Independent Auditor was best-calibrated overall (transparent methodology, reproducible estimates). The Fail-Fast Skeptic excelled at risk identification but overstated severity. The Scavenging Synthesizer provided the best constructive path forward but missed both HIGH-severity flaws — SS should not be sole evaluator.

---

## Verdict Summary

A genuine diamond buried in ~30% oversell. Version-pin the models, recount the LoC, restructure math around M5, acknowledge CLOTHO lineage, add a real-world case study, and this is a strong SIGMOD/CAV paper. Without restructuring, a reviewer who knows CLOTHO will desk-reject it. The engine-specific operational semantics are the load-bearing novelty — protect their quality and depth at the expense of breadth.

**Decision: CONDITIONAL CONTINUE — mandatory amendments A1–A6 must be applied.**

# IsoSpec: Verified Cross-Engine Transaction Isolation Analysis via Engine-Faithful Models and Differential Portability Checking

## 1. Title and Summary

**One-sentence summary:** IsoSpec builds rigorous operational semantics for three production SQL engines (PostgreSQL 16.x, MySQL 8.0, SQL Server 2022), connects them to Adya's isolation specifications via refinement, and uses differential SMT analysis to detect portability violations and discover previously undocumented isolation behaviors — validated empirically against real engine instances.

---

## 2. Approach Overview

IsoSpec occupies the space between full engine formalization (Approach A) and pure empirical discovery (Approach B), using migration-focused differential analysis (Approach C) as the primary delivery vehicle. The project builds formal operational semantics for three production engines at pinned versions — PostgreSQL 16.x (SSI with SIREAD locks), MySQL 8.0 InnoDB (next-key and gap locking with index-dependent behavior), and SQL Server 2022 (dual-mode pessimistic/optimistic concurrency) — and connects these models to Adya's dependency serialization graph framework via a refinement relation. The engine models are the load-bearing intellectual contribution: they are formal artifacts that have never existed, and they enable every downstream analysis.

The user-facing tool performs three analyses on SQL transaction programs: (1) single-engine anomaly detection, reporting which Adya anomaly classes (G0–G2, including predicate-level phantoms) are reachable under a specific engine's actual implementation; (2) cross-engine portability checking, using C's differential framing to determine whether a workload safe on engine E₁ remains safe when migrated to E₂ — modeling only the behavioral delta, not the full cross-product; and (3) executable witness synthesis, producing minimal, engine-dialect-correct SQL scripts that demonstrate each detected anomaly on a real database instance. The portability analysis targets the three highest-value migration pairs: Oracle→PostgreSQL (modeled as "absent formal model"→PG, using Adya spec-level as Oracle's proxy), SQL Server→PostgreSQL, and MySQL→PostgreSQL.

Critically, all models are validated against real engine behavior using empirical differential testing infrastructure adapted from Approach B. Docker-containerized engine instances serve as ground-truth oracles: for every model prediction, IsoSpec can generate and execute a confirming test on the real engine. The headline result is discovery-oriented: "Our engine-faithful models revealed N previously undocumented isolation behaviors across three production SQL engines, including K that affect common migration paths." This framing — models as discovery instruments, not just verification tools — is what elevates the work from a tool paper to a best-paper candidate.

---

## 3. Extreme Value Delivered

**Who desperately needs this:** SREs and DBAs executing the Oracle→PostgreSQL migration wave (Oracle licensing exceeds $47K/processor/year), cloud consolidation teams moving SQL Server to PostgreSQL on Aurora/Azure, and platform teams operating multi-engine architectures. The cloud database migration market exceeds $10B annually. AWS/Azure/Google DMS handle schema and data transfer but perform *zero* isolation semantics verification.

**Why they can't get it elsewhere:** Hermitage documents 47+ divergences but only for hand-written patterns — no arbitrary workload analysis, no completeness guarantee. CLOTHO checks abstract consistency models for distributed stores, not engine-specific SQL behavior. Jepsen/Elle test partition behavior, not engine-internal concurrency control. LLMs enumerate common differences but cannot guarantee completeness, generate reproducible witnesses, or integrate into CI/CD. No existing tool answers: "Will my transaction break when I migrate, and here is a runnable script proving it."

**Dollar impact:** Undetected isolation regressions cause silent data corruption — balances that don't sum, inventory going negative — often manifesting weeks post-migration. A Fortune 500 Oracle→PG migration budgets $2–10M for testing; IsoSpec replaces the isolation-specific portion with formal, automated, CI/CD-integrated verification.

---

## 4. What Gets Built (Subsystem Table)

| Subsystem | Language | Novel LoC | Description |
|-----------|----------|:---------:|-------------|
| Engine operational semantics (3 engines) | Rust | ~18K | Formal LTS models for PG 16.x SSI, MySQL 8.0 InnoDB gap/next-key locking, SQL Server 2022 dual-mode concurrency. Pinned versions with adequacy criteria. |
| Predicate-level conflict theory (M5) | Rust + Z3 | ~5K | Full conjunctive inequality fragment: symbolic predicate overlap, range conflict encoding, phantom detection across engines. |
| Refinement checker | Rust | ~4K | Algorithmic refinement ⊑ between Adya DSG specs and engine models for 3 engines × 4 isolation levels = 12 pairs. |
| SMT encoding of engine semantics | Rust (z3 bindings) | ~7K | Bounded model checking: lock tables, visibility rules, SIREAD sets, gap-lock intervals as QF_LIA+UF+Arrays constraints. |
| Differential portability analyzer | Rust | ~5K | Delta-based cross-engine analysis for 3 engine pairs. Models behavioral differences, not full cross-products. |
| SQL parser + transaction IR | Rust (sqlparser-rs) | ~6K | Dialect-aware parser extensions + isolation-aware IR preserving lock points, version reads, predicate ranges. |
| Witness synthesis | Rust | ~4K | Engine-dialect-correct SQL scripts from SMT counterexamples. MUS-based minimality. Dual-engine scripts for portability violations. |
| Lean 4 proofs (selected lemmas) | Lean 4 | ~5K | 3–5 key lemmas mechanized: PG SSI dangerous structure theorem soundness, refinement correctness for PG model, predicate conflict decidability. Paper proofs for all others. |
| Empirical validation oracle | Rust + Docker | ~4K | Docker-containerized engines as ground truth. Advisory-lock-based interleaving forcing. Regression testing for model adequacy. |
| Mixed-isolation recommender | Rust | ~2K | MaxSMT-based per-transaction isolation minimization with engine-specific cost functions. |
| **Subtotal: genuinely novel** | | **~60K** | |
| Infrastructure (CLI, tests, benchmarks, CI) | Rust | ~55K | Hermitage suite, TPC-C/E workloads, integration tests, reporting, CI/CD scaffolding. |
| **Grand total** | | **~115K** | |

**Honesty note:** The depth check found that the original 5-engine proposal's 78K "novel" LoC claim was ~30% inflated, with the honest count at ~55K. This 3-engine synthesis descopes engine models from 31K to 18K (−13K) and Lean proofs from 9–15K to 5K (−4–10K), but adds the empirical validation oracle (+4K, adopted from Approach B, not in the original scope). The subsystem-level estimates sum to 60K; given the track record of 15–20% underestimation in the depth check's corrected figures, **budget ~55K as the expected novel LoC with a ceiling of 65K if subsystem estimates prove optimistic.** The 55K infrastructure estimate is separate and consistent with standard tooling overhead.

---

## 5. Why This Is Genuinely Difficult

**Genuinely hard (requires novel intellectual work):**

- **Engine model fidelity for MySQL gap locking.** InnoDB's phantom prevention acquires different locks depending on which index the query optimizer selects. The same `SELECT ... WHERE x BETWEEN 10 AND 20` acquires gap locks on different key ranges depending on whether a secondary index exists, which secondary index is chosen, and the current B-tree page structure. Sound over-approximation (consider all possible index choices) is tractable but requires careful formalization of the index-selection space. No published formalization of this behavior exists.

- **PostgreSQL SSI read-only optimization.** PG 16.x permits certain read-only transactions that other SSI implementations (including Cahill's original) would abort. Modeling when a transaction is classified as "safe" under the read-only optimization requires formalizing the *absence* of certain rw-dependency patterns — a negation over graph structures that is subtle to encode in SMT.

- **Predicate conflict decidability boundary.** M5 must characterize exactly which fragment of SQL WHERE clauses admits decidable predicate conflict checking. The conjunctive inequality fragment (covering >90% of OLTP predicates) is decidable; the boundary with disjunctions, LIKE patterns, and subqueries must be precisely characterized with sound over-approximation for the undecidable region.

- **SQL Server dual-mode divergence.** SQL Server offers lock-based and snapshot-based concurrency at the *same* nominal isolation level, selected by a database-wide flag (`READ_COMMITTED_SNAPSHOT`). The model must capture both code paths and their behavioral differences — effectively two sub-models with a configuration-dependent selector.

**Tedious but tractable (known techniques, significant effort):**

- Parser dialect extensions over sqlparser-rs (~6K LoC of careful but not novel engineering)
- Docker orchestration for 3 engine containers with interleaving control
- Z3 constraint construction following established BMC patterns
- CLI, reporting, and CI/CD integration
- Test suite construction against Hermitage catalog

---

## 6. New Math Required (Load-Bearing Only)

### M5: Predicate-Level Conflict Theory (FLAGSHIP — Hard, Genuinely Novel)

**What:** Extend Adya's item-level DSG theory to handle SQL WHERE clauses, range scans, and INSERT/DELETE affecting predicate-defined sets. Define predicate conflicts as symbolic overlap conditions, prove decidability for the conjunctive inequality fragment over numeric and string columns, and provide a multi-engine encoding where different engines' predicate-lock strategies (PG's SIREAD predicate tracking vs. MySQL's gap locks vs. SQL Server's key-range locks) produce different conflict sets from the same predicates.

**Why needed:** Without M5, the tool cannot handle WHERE clauses — which means it cannot handle real SQL. Adya §3.4 sketches predicate dependencies but provides no symbolic encoding. Fekete et al. (TODS'05) analyze SI with predicate reads but not in a multi-engine symbolic setting. The full conjunctive inequality version (not the "restricted" variant) is necessary: the restricted version saves only ~30% of the effort while producing a substantially less publishable result that cannot handle the most interesting cross-engine divergences.

**Difficulty:** Hard. **Novelty:** Genuinely novel — the decidability boundary characterization and multi-engine predicate conflict encoding are new results.

### M1: Engine Isolation Operational Semantics (FRAMEWORK — Hard, Novel Artifacts)

**What:** For each of three engines at pinned versions, a formal operational semantics expressed as a labeled transition system. States comprise engine-specific structures (lock tables, version stores, dependency graphs). Transitions correspond to SQL operations. Adequacy criterion: trace inclusion on bounded workloads.

**Why needed:** These models are the foundation everything builds on. Without them, the portability analyzer has no ground truth, the refinement checker has nothing to refine, and the witness synthesizer cannot generate engine-correct scripts. No published paper provides implementation-faithful operational semantics for any of these engines.

**Difficulty:** Hard (deep domain expertise required). **Novelty:** Novel artifacts, not novel technique. The intellectual contribution is in the *content* of the models, not the *method* of modeling.

### M2: Isolation Refinement Relation (FRAMEWORK — Moderate, Novel Application)

**What:** A formal refinement ⊑ connecting Adya DSG isolation specs to engine-specific operational models. For each engine-isolation pair, prove that the engine model is a sound refinement of the corresponding Adya spec.

**Why needed:** The refinement relation is the bridge between "what the spec says" and "what the engine does." Without it, engine models float free of the theoretical framework and anomaly detection has no formal grounding.

**Difficulty:** Moderate (applies known process-algebraic refinement to a new domain). **Novelty:** Novel application — the technique is standard, but instantiation for SQL engine isolation is new.

### Differential Isolation Semantics (SUPPORTING — Moderate, Novel Framing)

**What:** Define the behavioral delta δ(E₁, I₁, E₂, I₂) between two engines at two isolation levels as the symmetric difference of admitted schedule sets. Prove that portability checking reduces to satisfiability of δ-constraints conjoined with the workload's data access footprint.

**Why needed:** Enables the portability analyzer to work on deltas rather than full cross-products — dramatically reducing analysis complexity for the migration use case.

**Difficulty:** Moderate. **Novelty:** Novel framing of a known concept (differential analysis) in the isolation domain.

---

## 7. Best-Paper Argument

**Venue:** SIGMOD 2027 (primary), with CAV 2027 as secondary.

**Headline result:** "We built the first implementation-faithful formal models of three production SQL engines' concurrency control mechanisms, and used them to discover N previously undocumented isolation behaviors — including K that silently affect common migration paths."

**Why this is best-paper caliber:**

1. **Discovery, not just verification.** The engine models are *discovery instruments*. Systematically exploring the behavioral space of each model, cross-validated against real engines, will surface behaviors beyond the Hermitage catalog. The target is 15+ novel, confirmed divergences — including at least 3 that affect the Oracle→PG or SQL Server→PG migration paths. This transforms the paper from "we built a tool" to "we found something surprising about systems everyone uses."

2. **Bridge paper.** IsoSpec bridges formal methods and database systems — exactly the kind of cross-community work that excites best-paper committees. The engine models are artifacts a PL researcher would appreciate; the migration tool is an artifact a systems researcher would deploy. CLOTHO bridged PL and distributed systems; IsoSpec does the same for SQL databases with deeper engine fidelity.

3. **Lasting research artifacts.** The three engine operational semantics are *new knowledge* that didn't exist before. They will be cited by future work on database formalization, testing, and verification — regardless of whether the IsoSpec tool itself is adopted.

4. **Practitioner impact story.** If we demonstrate IsoSpec catching a real migration issue that existing tools miss — even one compelling case from production migration logs — the paper has both theoretical depth and practical impact. This combination is what best-paper committees reward.

**CLOTHO differentiation (must be front-and-center):** "CLOTHO models what the specification says; IsoSpec models what the engine does. CLOTHO checks whether a program has anomalies under abstract causal consistency; IsoSpec checks whether a program has anomalies under PostgreSQL 16.x's specific SSI implementation, which differs from both the abstract spec and from MySQL's implementation of the 'same' isolation level."

---

## 8. Hardest Technical Challenge

**The single hardest problem: Faithful formalization of MySQL 8.0 InnoDB's index-dependent gap locking.**

MySQL's gap locking behavior depends on which index the query optimizer selects, which depends on table statistics, index cardinality estimates, and optimizer heuristics that change across minor versions. The *same* SQL statement — `SELECT * FROM t WHERE x BETWEEN 10 AND 20 FOR UPDATE` — acquires different gap locks depending on whether a secondary index on `x` exists, whether the optimizer uses it, and the current B-tree structure. This means the isolation behavior is a function of the physical schema and runtime optimizer state, not just the logical query.

**Why this is harder than the other engine models:** PostgreSQL SSI has published academic descriptions (Cahill et al., TODS'09). SQL Server's dual-mode behavior is configuration-dependent but deterministic given the configuration. MySQL's gap locking introduces optimizer non-determinism into the concurrency control semantics — a qualitatively different modeling challenge.

**Concrete mitigation strategy:**

1. **Sound over-approximation.** Model the *union* of all possible gap-lock acquisitions across all possible index choices. This means: for a range predicate, compute the gap-lock ranges for every index that *could* be selected, and include all of them. The model permits more schedules than any single optimizer choice would — ensuring no false negatives at the cost of potential false positives.

2. **Index-stratified refinement.** Allow users to provide an index hint (or actual `EXPLAIN` output) to narrow the over-approximation to a specific index choice. This recovers precision for users who know their schema and optimizer behavior.

3. **Empirical false-positive measurement.** For the over-approximated model, measure the actual false positive rate on TPC-C and TPC-E workloads by comparing model predictions against real MySQL execution. If the false positive rate exceeds 15%, introduce optimizer heuristic modeling as a targeted refinement.

4. **Version pinning with regression detection.** Pin to MySQL 8.0.x (specific patch version). Include a regression test suite that detects when a MySQL version update changes gap-lock behavior, triggering model review.

---

## 9. Evaluation Plan

### Experiment 1: Model Adequacy
For each engine, generate bounded workloads (k ≤ 3 transactions, n ≤ 5 operations) and compare model predictions against real engine behavior on Docker instances. **Metric:** Agreement rate on commit/abort outcomes. **Target:** ≥ 98% for PG and SQL Server, ≥ 95% for MySQL (accounting for over-approximation). **Baseline:** Adya spec-level prediction.

### Experiment 2: Anomaly Detection Completeness
Encode the full Hermitage test catalog. **Metric:** Recall of Hermitage-documented anomalies. **Target:** 100% for PG and MySQL, ≥ 95% for SQL Server. **Baseline:** CLOTHO-style spec-level analysis.

### Experiment 3: Novel Divergence Discovery (Headline)
Explore beyond Hermitage: multi-table FK transactions, index-dependent behaviors, read-only optimization edge cases, dual-mode SQL Server interactions. **Metric:** Count of novel, confirmed divergences. **Target:** ≥ 15 novel confirmed behaviors. **Validation:** Every divergence confirmed by execution on a real engine with a reproducible witness script.

### Experiment 4: Migration Portability Analysis
TPC-C (5 transaction types) and TPC-E (10 transaction types) across 3 migration pairs. **Metric:** Violations found, false positive rate, analysis time. **Target:** Sub-10-second analysis; ≥ 90% witness confirmation rate.

### Experiment 5: Real Migration Case Study
2–3 documented migration issues from PG mailing lists, AWS DMS bug reports, or Stack Overflow. **Metric:** Does IsoSpec catch what practitioners missed? **This is the single highest-impact evaluation component.**

### Experiment 6: Performance
Analysis time vs. workload size. **Target:** ≤ 30 seconds for workloads up to 5 transactions × 10 operations. Z3 timeout at 120s for pathological cases.

**Compute budget:** All experiments on a single workstation (32GB RAM, 8-core). ~50–100 CPU-hours total.

---

## 10. Risk Assessment

| # | Risk | Probability | Impact | Mitigation |
|---|------|:-----------:|:------:|------------|
| R1 | **MySQL gap-lock model fidelity insufficient** — over-approximation produces >20% false positive rate, making portability analysis noisy | 30% | HIGH | Index-stratified refinement; empirical FP measurement; fallback to "imprecise but sound" with confidence annotations |
| R2 | **M5 predicate theory harder than estimated** — decidability boundary characterization takes 3+ months longer than planned | 25% | HIGH | Implement conjunctive inequality fragment first (covers >90% of OLTP); defer boundary characterization to camera-ready or follow-up |
| R3 | **Novel divergence count disappoints** — systematic exploration finds <10 novel behaviors, weakening the headline result | 25% | MEDIUM | Expand exploration to MySQL minor version differences, SQL Server compatibility mode edge cases, and multi-statement transaction patterns. Lower threshold to 10 with at least 3 migration-affecting. |
| R4 | **Lean 4 mechanization of key lemmas stalls** — proof engineering for PG SSI dangerous structure theorem takes >2 months | 35% | LOW-MEDIUM | Scope to 3 simpler lemmas (refinement soundness, bounded completeness, predicate decidability). Dangerous structure theorem as paper proof only with mechanization as future work. |
| R5 | **Docker interleaving control too flaky for validation** — advisory-lock-based schedule forcing has >30% failure rate on target interleavings | 20% | MEDIUM | Retry-based statistical confirmation (run 100×, require ≥10 successful reproductions). Accept that validation oracle has false negatives; static analysis is the primary contribution. |

---

## 11. Scores (Final, Corrected)

| Dimension | Score | Rationale |
|-----------|:-----:|-----------|
| **Value** | **8.5** | Solves the #1 practitioner problem (migration safety) with formal guarantees no existing tool provides. Discovery angle adds research value. Not 9 because adoption barriers in DBA communities remain and the LLM-for-consultation alternative covers ~70% of one-off needs. |
| **Difficulty** | **8** | Three engine models at implementation fidelity + full M5 predicate theory is hard, domain-expert-level work. Not 9 because we descoped from 5 engines and full Lean mechanization, and the SMT encoding follows established BMC patterns. |
| **Potential** | **7.5** | Strong accept at SIGMOD with high probability; best-paper if discovery results are compelling (15+ novel divergences) and the migration case study lands. Not 8 because the CLOTHO positioning challenge is real, and the hybrid approach creates narrative complexity (harder to position than a single clean story). |
| **Feasibility** | **6.5** | Three engines instead of five, paper proofs instead of full Lean, and differential portability instead of full cross-product make this deliverable. Not 7 because MySQL gap-lock modeling and M5 predicate theory are independently risky, and LoC estimates have a track record of 15–20% underestimation. |
| **Composite** | **30.5** | Exceeds all three individual approaches (A: 26, B: 29.5, C: 28) by combining their strengths while mitigating their weaknesses. |

---

## 12. What Was Cut and Why

### Cut from Approach A (Engine Model Maximalist)

- **Oracle 23c and SQLite 3.x models** — Proprietary internals (Oracle) and low enterprise migration value (SQLite). For Oracle→PG migration, Adya spec-level serves as Oracle's behavioral proxy.
- **Full Lean 4 mechanization (15–20K LoC)** — PhD-sized, 40% delay probability. Replaced with 3–5 key lemmas (~5K LoC) + paper proofs.
- **M3, M7, M8 as contributions** — Routine/minor/definitional. Relegated to appendix or propositions.
- **Mixed-isolation optimizer as major subsystem** — Descoped to ~2K LoC practical bonus.

### Cut from Approach B (Empirical-First Differential Testing)

- **Fuzzer as primary method** — No soundness guarantee. Retained as model adequacy oracle only.
- **140–1400 CPU-hour evaluation** — Contradicts laptop-CPU constraint. Budget: ~50–100 CPU-hours.
- **Symbolic post-hoc explanation** — "Secretly requires engine models." We build them explicitly.
- **Coverage theory** — Replaced by bounded completeness from formal models.

### Cut from Approach C (Migration-Focused Portability Checker)

- **10 pairwise delta models** — Replaced with 3 full engine models + differential extraction. Full models are reusable research artifacts.
- **"Restricted" M5** — Saves only ~30%, substantially less publishable. Full conjunctive inequality fragment needed.
- **Sub-second analysis target** — Relaxed to sub-30s for precision.

### Explicit Scope Boundaries

- **Stored procedures, triggers, and procedural extensions** — PL/pgSQL, T-SQL, PL/SQL, and MySQL stored procedures constitute four distinct programming languages with their own control flow, exception handling, and isolation-interacting constructs. V1 handles pure SQL transactions only; procedural analysis is deferred to follow-on work.
- **Oracle 23c engine model** — For Oracle→PG migration (the highest-value pair), Adya spec-level isolation serves as Oracle's behavioral proxy. A dedicated Oracle model based on MVRC/SCN reverse-engineering could be added later but is not in scope for V1 due to proprietary internals.
- **SQLite 3.x WAL model** — Low enterprise migration value. WAL-mode snapshot isolation semantics are well-documented and could be added with modest effort; deferred to follow-on work.

---

## 13. Timeline

**Total: 11 months.** Buffer of 1 month absorbed into the schedule via conservative per-phase estimates.

### Months 1–2: Foundations
- PostgreSQL 16.x SSI operational semantics (LTS formalization)
- Transaction IR design and core SQL parser extensions
- Z3 infrastructure: constraint construction framework, basic BMC loop
- Docker validation harness for PG
- **Milestone:** PG model passes Hermitage PG test suite with ≥98% agreement

### Months 3–4: Second Engine + M5 Core
- MySQL 8.0 InnoDB gap/next-key locking model with over-approximation
- M5 predicate conflict theory: conjunctive inequality fragment encoding
- Docker validation harness for MySQL
- **Milestone:** MySQL model passes Hermitage MySQL suite; M5 handles range predicates on single-table workloads

### Months 5–6: Third Engine + Refinement
- SQL Server 2022 dual-mode concurrency model
- Refinement relation implementation: 3 engines × 4 isolation levels
- Portability analyzer: differential encoding for 3 migration pairs
- **Milestone:** All 3 engines modeled; refinement verified for 12 engine-isolation pairs; first portability violations detected

### Months 7–8: Witness Synthesis + Discovery Campaign
- Witness synthesis: MUS-based minimization, dialect-correct SQL generation
- Systematic divergence discovery beyond Hermitage catalog
- Mixed-isolation recommender (MaxSMT)
- **Milestone:** ≥10 novel confirmed divergences; witness scripts execute successfully on real engines

### Months 9–10: Evaluation + Lean Proofs
- Full evaluation campaign: all 6 experiments
- Lean 4 mechanization of 3–5 key lemmas
- Real-world migration case study identification and encoding
- **Milestone:** Evaluation complete; ≥15 novel divergences confirmed; migration case study demonstrates real-world value

### Month 11: Paper Writing + Polish
- SIGMOD 2027 paper draft (target: 14 pages + appendix)
- Open-source release preparation
- Documentation and reproducibility package
- **Milestone:** Submission-ready paper; public artifact

**Critical path:** PG model (M1–2) → MySQL model (M3–4) → Refinement + portability (M5–6) → Discovery + evaluation (M7–10). The Lean proofs and paper writing are off the critical path and can absorb schedule slack.

**Schedule risk buffer:** Each 2-month phase includes ~2 weeks of implicit buffer. If MySQL gap-lock modeling runs long (R1), it can absorb 3 weeks from the Lean proof allocation (R4) since Lean proofs are lower priority. If M5 runs long (R2), the conjunctive inequality core is prioritized and boundary characterization defers to camera-ready. **Worst-case timeline: 12–13 months** if R2 (M5 overrun) and R4 (Lean stall) both trigger, pushing paper writing to month 12–13.

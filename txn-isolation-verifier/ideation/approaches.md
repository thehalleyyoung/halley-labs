# IsoSpec: Three Competing Approaches

**Problem:** Verified cross-engine transaction isolation analysis for 5 production SQL engines  
**Phase:** Ideation — competing technical strategies  
**Date:** 2026-03-08

---

## Approach A: Engine Model Maximalist

*Deep formal models, full refinement theory, Lean 4 proofs, predicate-level conflict theory.*

### Extreme Value

This approach delivers **mathematically guaranteed migration safety** — the only artifact in existence that can prove a transaction workload is portable between two specific engine versions, or produce a runnable counterexample. The desperate users are **database platform teams at enterprises performing Oracle→PostgreSQL migrations** (a $10B+/year market driven by licensing escape), where silent isolation divergences cause data corruption that manifests weeks post-migration when root-cause analysis is impossible. A secondary audience is **PL/DB researchers** who need engine-faithful formal artifacts for studying isolation — today, no published operational semantics capture InnoDB's index-dependent gap locking or PostgreSQL SSI's read-only optimization. These models become reference artifacts cited by future work.

### What Gets Built

| Subsystem | Language | Novel LoC | Description |
|-----------|----------|-----------|-------------|
| Engine operational semantics (5 engines) | Rust + Lean 4 | ~25K | LTS models: PG 16.x SSI, MySQL 8.0 InnoDB gap locks, SQL Server 2022 dual-mode, Oracle 23c MVRC (approximate), SQLite 3.x WAL |
| Lean 4 mechanized proofs | Lean 4 | ~9K | Soundness of PG SSI refinement, SMT encoding correctness, compositionality theorem |
| Predicate-level conflict theory | Rust + Lean 4 | ~5K | M5 flagship: symbolic predicate overlap for range operations, decidability boundary |
| Refinement checker | Rust | ~5K | Decision procedure for spec⊑engine relation across 20 (engine, level) pairs |
| SMT encoding layer | Rust (z3 bindings) | ~8K | QF_LIA+UF+Arrays encoding of all 5 engine models |
| Transaction IR | Rust | ~7K | Dialect-normalizing IR preserving lock points, version reads, predicate ranges |
| SQL parser extensions | Rust (sqlparser-rs) | ~4K | Engine-specific locking syntax, dialect translation |
| Cross-engine portability analyzer | Rust | ~6K | Differential SMT across 10 engine pairs |
| Witness synthesis | Rust | ~4K | Dialect-correct SQL scripts from SMT counterexamples, MUS-based minimization |
| Mixed-isolation optimizer | Rust | ~3K | MaxSMT-based per-transaction level assignment |
| Validation oracle | Rust + Docker | ~3K | Automated cross-validation against Dockerized engine instances |
| Infrastructure (tests, CLI, benchmarks) | Rust | ~70K | Hermitage benchmark, TPC-C/E workloads, CI scaffolding |
| **Total** | | **~55K novel + ~70K infra** | |

### Math Required (Load-Bearing Only)

1. **M5: Predicate-level conflict theory** (HARD, genuinely novel). Extends Adya's item-level DSG to symbolic predicate conflicts for range scans, INSERT/DELETE over predicate-defined sets. Defines decidable fragment (conjunctive inequalities over numeric/string columns). This is the flagship theorem — without it, the tool cannot handle WHERE clauses, which means it cannot handle real SQL. No published work provides this in a multi-engine symbolic setting.

2. **M1+M2: Engine semantics + refinement framework** (HARD, novel artifacts). Five operational semantics as LTS, plus a refinement relation ⊑ connecting Adya DSG specs to engine models. The math itself (refinement, trace inclusion) is known from process algebra; the novelty is in the 5 concrete models and the 20-pair refinement computation. Still requires careful formalization — e.g., modeling InnoDB gap lock ranges as symbolic B-tree intervals with sound over-approximation of optimizer index choice.

3. **M6: Mixed-isolation optimization as weighted MaxSMT** (MODERATE). Engine-specific isolation levels form a DAG (SQL Server's SNAPSHOT is incomparable with REPEATABLE READ). Formulation as MaxSMT over a DAG-structured cost space is novel for the isolation domain.

### Evaluation

- **Known anomaly benchmark:** ~200 anomalies from Hermitage, Jepsen, engine bug trackers. Measure recall/precision.
- **Novel divergence discovery:** Systematic exploration beyond Hermitage targeting multi-table FK transactions, index-dependent MySQL behaviors, PG SSI read-only edge cases. Each discovery confirmed on real engine instances.
- **Real-world migration case study:** 1–3 documented Oracle→PG or SQL Server→MySQL migration failures from mailing lists/DMS bug reports.
- **Cross-engine portability:** TPC-C (5 txns) × TPC-E (10 txns) × 10 engine pairs × 4 isolation levels.
- **Ground truth validation:** Every reported anomaly executed as witness script on Dockerized engines. Target >95% confirmation rate (lower for Oracle approximate model).
- **Scalability:** Seconds for typical workloads (5–20 txns), minutes for stress tests (50–100 txns).

### Paper Contribution

A SIGMOD/VLDB paper: "IsoSpec: Verified Cross-Engine Transaction Isolation Analysis via Implementation Refinement." Contributions: (1) first implementation-faithful operational semantics for 5 production engine concurrency control mechanisms, (2) refinement bridge from Adya spec to engine models with mechanized soundness for PG SSI, (3) predicate-level conflict theory extending Adya's DSG, (4) practical tool with executable witness generation, (5) discovery of N previously undocumented isolation divergences. Best-paper angle: the formal models as discovery instruments revealing engine behaviors the community didn't know about.

### Hardest Technical Challenge

**Modeling MySQL InnoDB's index-dependent gap locking faithfully.** The set of locks acquired by a range scan depends on which index the query optimizer selects, which depends on table statistics, buffer pool state, and cost model internals — none of which are part of the SQL-level specification. If the model is too abstract, it misses real anomalies (unsound). If it's too concrete, it depends on optimizer internals that change across minor versions.

**Mitigation:** Sound over-approximation. Model the *union* of all lock sets across all possible index choices. This guarantees no false negatives (any anomaly possible under any index choice is reported) at the cost of false positives (anomalies reported that only manifest under specific optimizer decisions). Measure and report the over-approximation rate empirically. This is the principled formal-methods answer, and it's what makes the result useful — practitioners want to know "could this ever break?" not "will it break under today's statistics."

### Risk Factors

- **Scope creep:** 5 engines × 4 levels × Lean proofs is enormous. Risk of delivering breadth without depth on any one engine.
- **Oracle model quality:** Proprietary internals mean the Oracle model is inherently approximate. Reviewers may question including it.
- **Lean 4 proof engineering:** Formalizing Cahill's dangerous structure theorem is multi-month work. Could bottleneck the project.
- **Z3 timeout on gap-lock disjunctions:** Over-approximation of index choice creates disjunctive constraints that may blow up for large schemas.

### Scores

| Dimension | Score | Rationale |
|-----------|:-----:|-----------|
| Value | 8 | Solves real migration problem + produces reference formal artifacts |
| Difficulty | 9 | 55K novel LoC, 5 engine models, Lean proofs, predicate theory |
| Potential | 8 | Strong SIGMOD accept; best-paper if discovery results are compelling |
| Feasibility | 5 | Extremely ambitious scope; high risk of partial delivery |

---

## Approach B: Empirical-First Differential Testing

*Start from observed engine behavior, not formal models. Discover divergences empirically, then explain them symbolically.*

### Extreme Value

This approach delivers the **most anomalies found per unit of engineering effort** and answers the practitioner's real question — "what breaks when I switch engines?" — through direct empirical evidence rather than model-mediated inference. The desperate users are the same migration teams, but this tool reaches them faster with lower trust barriers: every reported divergence comes with empirical proof on real engines, not just a model's prediction. A secondary user is **the Hermitage community** — researchers and practitioners who document engine isolation behaviors. This tool automates and massively scales Hermitage-style testing, discovering divergences Hermitage missed because hand-crafted test suites cannot achieve systematic coverage.

The key insight: practitioners trust empirical evidence over formal models. A tool that says "I ran this SQL on PostgreSQL and MySQL and got different results — here's the script" is immediately actionable. A tool that says "my formal model predicts a divergence" requires the practitioner to trust the model.

### What Gets Built

| Subsystem | Language | Novel LoC | Description |
|-----------|----------|-----------|-------------|
| Transaction program generator | Rust | ~6K | Grammar-based fuzzer producing parameterized SQL transaction programs (Csmith-style for SQL transactions) |
| Multi-engine test harness | Rust + Docker | ~8K | Orchestrate concurrent execution of identical transaction programs on 5 Dockerized engines with controlled scheduling |
| Differential oracle | Rust | ~4K | Compare outcomes (commit/abort sets, final database states) across engines; flag divergences |
| Interleaving controller | Rust | ~5K | Thread barriers + advisory locks + retry-based schedule forcing for reproducible interleavings |
| Lightweight symbolic analyzer | Rust (z3 bindings) | ~10K | Post-hoc symbolic analysis: given an observed divergence, extract the minimal engine-semantic difference explaining it |
| Anomaly classifier | Rust | ~4K | Classify observed divergences against Adya's taxonomy (G0, G1a–c, G2-item, G2) |
| Divergence minimizer | Rust | ~3K | Delta-debugging to reduce divergence-triggering programs to minimal form |
| Witness packager | Rust | ~2K | Package minimized divergences as standalone runnable scripts with setup/teardown |
| Portability checker | Rust | ~5K | Given user transaction program, run on target engine pair, report behavioral differences |
| Engine behavior database | Rust + SQLite | ~3K | Persistent catalog of observed engine behaviors, indexed by anomaly class and engine pair |
| Infrastructure (CLI, tests, CI) | Rust | ~40K | |
| **Total** | | **~50K novel + ~40K infra** | |

### Math Required (Load-Bearing Only)

1. **Symbolic divergence explanation** (MODERATE, novel application). Given two concrete executions with different outcomes on engines E₁ and E₂, extract the minimal semantic condition (lock type, visibility rule, abort condition) that explains the divergence. This is essentially abductive reasoning over engine-behavioral constraints — formalized as a MinUNSAT extraction problem over lightweight engine constraint sketches. Not deep theory, but requires careful formalization of what constitutes a "semantic explanation" vs. merely a "syntactic diff of execution traces."

2. **Coverage theory for transaction program spaces** (MODERATE, novel). Define a coverage metric over the space of SQL transaction programs that guarantees: if a divergence exists for programs up to size k, fuzzing with N samples achieves detection probability ≥ 1−δ. This connects combinatorial testing theory to the transaction isolation domain. Requires characterizing the "interesting" dimensions of the program space (number of transactions, operations per transaction, predicate shapes, table/index structure).

3. **Behavioral equivalence up to scheduling** (MODERATE, largely known). Two engines are behaviorally equivalent on program P iff for all feasible schedules, they produce the same commit/abort outcomes. This is a standard notion but needs careful instantiation for the SQL domain — particularly around non-deterministic engine-internal scheduling that cannot be externally controlled.

### Evaluation

- **Divergence discovery rate:** Run fuzzer for T hours on each of 10 engine pairs. Report number of unique divergences found, plotted over time (expect diminishing returns curve). Compare against Hermitage catalog — report how many Hermitage divergences are rediscovered and how many novel ones are found.
- **Minimization quality:** For each divergence, measure reduction ratio (original program size / minimized program size). Target: median reduction ≥ 5×.
- **Symbolic explanation accuracy:** For divergences with symbolic explanations, validate that the explanation correctly predicts behavior on held-out test cases.
- **User-provided portability:** Encode TPC-C/E workloads, run portability checker, report divergences found vs. known issues.
- **Comparison with IsoSpec-A (model-based):** On the subset of divergences both approaches can find, compare: time to first detection, explanation quality, false positive rate.
- **Scalability:** Divergences found per CPU-hour as function of program complexity.

### Paper Contribution

A SIGMOD/VLDB paper: "IsoFuzz: Automated Discovery of Transaction Isolation Divergences Across Production SQL Engines." Contributions: (1) grammar-based transaction program fuzzer with coverage guarantees, (2) multi-engine differential testing framework with interleaving control, (3) symbolic post-hoc divergence explanation, (4) catalog of N newly discovered isolation divergences across 5 engines (the headline result), (5) open-source engine behavior database. Best-paper angle: the discovered divergences themselves — "we found 47 previously undocumented isolation behaviors" is a headline that gets cited for decades (like Hermitage itself).

### Hardest Technical Challenge

**Reliable interleaving control on real engine instances.** Differential testing requires running the *same schedule* on two different engines and comparing outcomes. But SQL engines have internal schedulers you don't control — thread pools, lock wait queues, I/O scheduling. Advisory locks and thread barriers give coarse-grained control, but fine-grained interleavings (e.g., "transaction T₂ reads between T₁'s write and T₁'s commit") require timing tricks that are inherently flaky.

**Mitigation:** Three-tier strategy. (1) Coarse interleavings via advisory locks and explicit barriers — reliable for most anomaly classes. (2) Retry-based schedule forcing: run the same test 1000× and check if the target interleaving ever manifests, using statistical confidence bounds. (3) For anomalies requiring precise interleaving, use engine-specific hooks where available (PG's `pg_advisory_lock`, MySQL's `GET_LOCK`, SQL Server's `sp_getapplock`) as synchronization primitives within the transaction programs themselves. Accept that some interleavings will have low reproduction probability and report confidence intervals rather than deterministic verdicts.

### Risk Factors

- **Flaky tests:** Concurrency testing is inherently non-deterministic. High false-negative rate (real divergences that don't reproduce reliably) will undermine trust.
- **No soundness guarantee:** Unlike Approach A, this cannot prove the absence of divergences. A clean fuzzing run doesn't mean the migration is safe.
- **Shallow explanations:** Symbolic post-hoc analysis may produce explanations that are technically correct but not illuminating ("the engines disagree because they implement different lock modes" — but *which* lock mode difference matters for *this* workload?).
- **Engine startup overhead:** Dockerized engines take 5–30 seconds to start. This limits the throughput of the fuzzer unless engines are kept running persistently.

### Scores

| Dimension | Score | Rationale |
|-----------|:-----:|-----------|
| Value | 9 | Directly produces actionable divergence catalogs; immediate practitioner trust |
| Difficulty | 6 | Fuzzing + differential testing is well-understood; novelty is in the domain application |
| Potential | 7 | Strong SIGMOD/VLDB; best-paper if discovery results are spectacular |
| Feasibility | 8 | Much smaller scope than A; core subsystems are well-understood engineering |

---

## Approach C: Migration-Focused Portability Checker

*Narrow scope to the killer use case: is my workload safe to migrate from engine X to engine Y?*

### Extreme Value

This approach delivers **a single, sharp answer to the single most expensive question in database engineering: "Will my transactions break when I migrate?"** The desperate users are **SREs and DBAs executing Oracle→PostgreSQL migrations** who currently rely on blog posts, tribal knowledge, and prayer. Unlike Approach A (which models full engine semantics) or Approach B (which discovers divergences empirically), this approach models only the *behavioral delta* between engine pairs — the precise semantic differences at the isolation-relevant interface. This means: smaller models, faster analysis, sharper results, and a tool that can ship in months, not years.

The key insight: you don't need to model everything PostgreSQL does to check migration safety. You need to model what PostgreSQL does *differently from Oracle* on isolation-relevant operations. The differential is far smaller than the union of both engines' full semantics.

### What Gets Built

| Subsystem | Language | Novel LoC | Description |
|-----------|----------|-----------|-------------|
| Engine-pair differential models | Rust | ~12K | 10 pairwise behavioral delta models (not 5 full engine models) capturing isolation-relevant differences |
| Differential SMT encoding | Rust (z3 bindings) | ~6K | Encode only the delta constraints: "under engine E₁ this schedule commits; under E₂ it aborts (or vice versa)" |
| SQL parser + IR | Rust (sqlparser-rs) | ~6K | Simplified IR focused on data access patterns and predicate shapes, not full engine semantics |
| Portability verifier | Rust | ~5K | Core analysis: given workload W and migration E₁→E₂, find schedules safe on E₁ but anomalous on E₂ |
| Witness synthesis | Rust | ~3K | Dual-engine witness scripts: one script per divergence, runnable on both source and target engine |
| Migration report generator | Rust | ~3K | Structured migration safety report: safe/unsafe per transaction, risk level, remediation suggestions |
| Engine delta validation | Rust + Docker | ~3K | Validate differential models against real engine pairs |
| Infrastructure (CLI, tests) | Rust | ~30K | |
| **Total** | | **~38K novel + ~30K infra** | |

### Math Required (Load-Bearing Only)

1. **Differential isolation semantics** (MODERATE, genuinely novel framing). Define the *behavioral delta* δ(E₁, I₁, E₂, I₂) between two engines at two isolation levels as the symmetric difference of their admitted schedule sets. Prove that portability checking reduces to satisfiability of δ-constraints conjoined with the workload's data access footprint. This is simpler than full engine models (Approach A's M1) because you only formalize the differences, not the shared behavior. But it requires careful characterization of when the delta is compositional — i.e., when δ(E₁, E₂) on sub-workload W₁ and δ(E₁, E₂) on W₂ implies δ(E₁, E₂) on W₁∪W₂.

2. **Predicate-level conflict theory (restricted)** (MODERATE, restricted version of A's M5). Still need predicate conflicts for WHERE clauses, but only for the *differential* case — operations where the two engines' predicate-lock strategies differ. For the Oracle→PG migration (the highest-value pair), this reduces to: Oracle has no predicate locking at REPEATABLE READ (it uses MVRC); PG uses SSI with SIREAD locks that do track predicate dependencies. The restricted version is substantially easier than the full M5.

3. **Migration safety composition** (MINOR). If workload W is decomposable into independent sub-workloads W₁...Wₙ (no shared data access footprint), then migration safety of each Wᵢ implies migration safety of W. Standard but needs to be stated and proved for the differential setting.

### Evaluation

- **Migration case studies (headline evaluation):** 3–5 real migration scenarios: Oracle→PG (the big one), SQL Server→MySQL, MySQL→PG, PG→SQLite (embedded migration), SQL Server→PG. For each, encode representative transaction workloads and run portability analysis.
- **Hermitage differential:** For every divergence documented in Hermitage, verify that the portability checker flags it for the relevant engine pair.
- **TPC-C/E portability matrix:** 5 TPC-C transaction types × 10 TPC-E transaction types × 10 engine pairs. Report: which migrations are safe, which are unsafe, and what specifically breaks.
- **False positive rate:** For migrations flagged as unsafe, execute dual-engine witness scripts to confirm. For migrations flagged as safe, run randomized schedule exploration to validate.
- **Performance:** Target sub-second analysis for typical workloads (the tool should feel like a linter, not a prover).
- **Comparison with manual migration guides:** Take 3 published Oracle→PG migration guides. Compare their isolation-related warnings against IsoSpec-C's findings. Report coverage and novel findings.

### Paper Contribution

A SIGMOD/VLDB paper: "PortaSQL: Verified Transaction Isolation Portability for Database Migrations." Contributions: (1) differential isolation semantics formalizing behavioral deltas between engine pairs, (2) practical portability verifier with sub-second analysis time, (3) dual-engine witness generation (runnable on both source and target), (4) comprehensive portability matrix for 5 engines × 4 isolation levels on standard benchmarks, (5) case study demonstrating detection of known migration pitfalls. Best-paper angle: the practical impact story — "we ran this on 3 real migration projects and found issues the migration team missed" — combined with the elegance of differential semantics (model only what differs, not everything).

### Hardest Technical Challenge

**Ensuring the differential models are complete — that no isolation-relevant behavioral difference is missing from the delta.** If you model full engine semantics (Approach A), completeness is a property of the individual models. With differential models, you must ensure the delta captures *all* differences, which requires understanding both engines well enough to know what's shared and what's not. A missing delta entry means a false negative — a migration bug the tool doesn't catch.

**Mitigation:** Three-layer assurance. (1) Systematic derivation of deltas from engine documentation and the Hermitage catalog — every documented behavioral difference becomes a delta entry. (2) Empirical delta validation: for each engine pair, run the Hermitage test suite on both engines and verify the differential model predicts all observed divergences. (3) Conservative completion: if unsure whether a behavior differs, include it in the delta (over-approximate the delta set). This may produce false positives but never false negatives. The over-approximation rate is measurable and reportable.

### Risk Factors

- **Delta completeness:** The fundamental risk. If the behavioral delta between Oracle and PG is missing a case, users get a false "safe to migrate" verdict. This is worse than no tool at all.
- **Compositionality gaps:** Real workloads interact in ways that sub-workload decomposition misses (shared locks, version visibility across transactions touching different tables via FK constraints).
- **Narrower contribution:** Reviewers may view this as "just an engineering tool" without sufficient theoretical novelty. The differential semantics framing must be presented as a genuine intellectual contribution, not just an optimization.
- **Less impressive artifact:** 38K novel LoC vs. 55K for Approach A. The tool is less impressive as a research artifact, even if it's more useful as a product.

### Scores

| Dimension | Score | Rationale |
|-----------|:-----:|-----------|
| Value | 9 | Solves the highest-value use case directly; immediately deployable |
| Difficulty | 5 | Smaller scope, known techniques, but delta completeness is subtle |
| Potential | 6 | Strong accept at SIGMOD; best-paper unlikely without deeper theory |
| Feasibility | 9 | Smallest scope, most focused, highest probability of complete delivery |

---

## Comparative Summary

| Dimension | A: Engine Maximalist | B: Empirical-First | C: Migration-Focused |
|-----------|:---:|:---:|:---:|
| **Value** | 8 | 9 | 9 |
| **Difficulty** | 9 | 6 | 5 |
| **Potential** | 8 | 7 | 6 |
| **Feasibility** | 5 | 8 | 9 |
| **Novel LoC** | ~55K | ~50K | ~38K |
| **Math depth** | High (M5 full + Lean proofs) | Low–Moderate | Moderate (restricted M5) |
| **Soundness** | Yes (bounded) | No (empirical) | Yes (bounded, differential) |
| **Primary risk** | Scope/delivery | Flaky tests, no guarantees | Delta completeness |
| **Best venue** | SIGMOD/CAV | SIGMOD/VLDB | SIGMOD |
| **Time to MVP** | 12–18 months | 4–6 months | 6–9 months |

### Key Trade-offs

**A vs. B:** Theory vs. empiricism. A can prove safety; B can only fail to find bugs. But B finds bugs faster and produces immediately trusted evidence. A is the better *research* artifact; B is the better *discovery* tool.

**A vs. C:** Full models vs. differential models. A produces reusable formal artifacts (5 engine semantics) that enable analyses beyond migration. C is laser-focused on the highest-value problem but can't do single-engine anomaly detection or mixed-isolation optimization.

**B vs. C:** Discovery vs. verification. B discovers divergences you didn't know about; C verifies your specific workload against known divergences. B requires running real engines (Docker, flaky concurrency); C runs pure symbolic analysis (fast, deterministic). B cannot guarantee safety; C can (modulo delta completeness).

### Recommended Hybrid

The strongest possible project combines elements: **A's formal models and predicate theory for the 2–3 most important engines (PG, MySQL, Oracle), C's differential framing for the portability analysis, and B's empirical validation as the ground-truth oracle.** This is essentially Approach A scoped to 3 engines instead of 5, with C's differential encoding for the portability analyzer, and B's fuzzing infrastructure for model validation. But this hybrid is the *proposal* phase's job — for now, these three approaches represent genuinely distinct strategies with different risk/reward profiles.

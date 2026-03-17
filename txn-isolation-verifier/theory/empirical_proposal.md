# IsoSpec Empirical Evaluation Strategy

**Version:** 1.0  
**Status:** Theory Phase — Pre-Implementation Design  
**Compute Budget:** 50–100 CPU-hours (single laptop, no GPU, no cloud)  
**Infrastructure:** Docker-containerized engines (PostgreSQL 16.x, MySQL 8.0.x InnoDB, SQL Server 2022)  
**SMT Backend:** Z3 4.12+  
**Reproducibility:** All experiments yield deterministic artifacts; every claim backed by a runnable SQL script.

---

## Table of Contents

1. [Guiding Principles](#1-guiding-principles)
2. [Experiment 1: Model Adequacy Validation](#2-experiment-1-model-adequacy-validation)
3. [Experiment 2: Known Anomaly Detection (Hermitage Benchmark)](#3-experiment-2-known-anomaly-detection-hermitage-benchmark)
4. [Experiment 3: Novel Divergence Discovery](#4-experiment-3-novel-divergence-discovery)
5. [Experiment 4: Cross-Engine Portability Analysis](#5-experiment-4-cross-engine-portability-analysis)
6. [Experiment 5: Real-World Migration Case Study](#6-experiment-5-real-world-migration-case-study)
7. [Experiment 6: Scalability Benchmarks](#7-experiment-6-scalability-benchmarks)
8. [Baseline Comparison Matrix](#8-baseline-comparison-matrix)
9. [Threats to Validity](#9-threats-to-validity)
10. [Compute Budget Allocation](#10-compute-budget-allocation)
11. [Artifact Checklist](#11-artifact-checklist)

---

## 1. Guiding Principles

### 1.1 Falsifiability

Every hypothesis is stated with a concrete numerical threshold. If the threshold is not met, the hypothesis is rejected. We do not retroactively adjust thresholds.

### 1.2 Reproducibility

- **Pinned engine versions:** PostgreSQL 16.4, MySQL 8.0.37, SQL Server 2022 CU12. Docker images specified by SHA256 digest.
- **Pinned Z3 version:** 4.12.6 (binary SHA256 recorded).
- **Deterministic workload generation:** Pseudorandom generation with fixed seeds (seed = experiment_id × 10007 + workload_index).
- **All artifacts versioned:** Git repository contains every workload encoding, witness script, and raw result CSV.

### 1.3 Statistical Rigor

- Point estimates always accompanied by confidence intervals or dispersion measures.
- No p-hacking: all hypotheses and thresholds specified before any experiment runs.
- Multiple comparison correction (Bonferroni) applied when testing across engines.

### 1.4 Separation of Concerns

Each experiment tests a distinct claim. No single experiment's failure invalidates the entire evaluation; each stands independently.

---

## 2. Experiment 1: Model Adequacy Validation

### 2.1 Hypothesis

**H1:** IsoSpec's engine-specific operational semantics (M1) predict transaction commit/abort outcomes with:
- ≥98% agreement rate for PostgreSQL 16.x SSI and SQL Server 2022 (both lock-based and snapshot modes),
- ≥95% agreement rate for MySQL 8.0 InnoDB (accounting for sound over-approximation of index-dependent gap locking).

A prediction counts as "agreement" if the model's commit/abort verdict for every transaction in the interleaving matches the engine's observed outcome.

### 2.2 Workload Generation

**Parameters:**

| Parameter | Values | Description |
|-----------|--------|-------------|
| k (transactions) | {2, 3} | Number of concurrent transactions |
| n (operations/txn) | {3, 5, 10} | Operations per transaction (READ, WRITE, INSERT, DELETE) |
| tables | {1, 2, 3} | Number of tables in schema |
| columns/table | {1, 3, 5} | Columns per table |
| secondary indexes | {0, 1, 3} | MySQL-specific: secondary indexes per table |
| predicate complexity | {point, range, conjunctive} | WHERE clause type |

**Generation procedure:**
1. For each parameter combination, generate workloads using a deterministic pseudorandom process (Rust `rand::SeedableRng` with `ChaCha8Rng`, seed = `experiment_id * 10007 + workload_index`).
2. Each workload specifies: schema DDL, transaction programs (SQL statements with parameterized WHERE clauses), and a target interleaving (total order over operations).
3. Filter out trivially serial workloads (no overlapping operations).
4. Target: **500 non-trivial workloads** after filtering.

**Schema template:**
```sql
CREATE TABLE t_i (
    pk   INTEGER PRIMARY KEY,
    c1   INTEGER NOT NULL,
    ...
    c_m  INTEGER NOT NULL
);
-- Optional secondary indexes (MySQL experiments):
CREATE INDEX idx_t_i_c1 ON t_i(c1);
```

### 2.3 Interleaving Forcing Protocol

**Mechanism:** Advisory-lock barriers combined with sleep-based synchronization.

For PostgreSQL:
```sql
-- Barrier: T1 acquires advisory lock, T2 blocks on it
SELECT pg_advisory_lock(barrier_id);  -- T1 holds
-- T2 blocked:
SELECT pg_advisory_lock(barrier_id);  -- T2 waits
-- T1 releases when ready:
SELECT pg_advisory_unlock(barrier_id);
```

For MySQL:
```sql
SELECT GET_LOCK('barrier_N', timeout);
SELECT RELEASE_LOCK('barrier_N');
```

For SQL Server:
```sql
EXEC sp_getapplock @Resource = 'barrier_N', @LockMode = 'Exclusive';
EXEC sp_releaseapplock @Resource = 'barrier_N';
```

**Retry policy:** Each interleaving attempt has a 5-second timeout. If the target interleaving is not achieved (detected by operation timestamps deviating >100ms from intended order), retry up to 10 times. If 10 retries fail, mark the workload as "interleaving-unachievable" and exclude from agreement calculation. Report the exclusion rate.

**Acceptance criterion for interleaving control:** ≥90% of target interleavings must be achievable (≤10% exclusion rate). If exceeded, the interleaving forcing protocol is inadequate and must be revised before results are valid.

### 2.4 Test Matrix

| Dimension | Values | Count |
|-----------|--------|-------|
| Workloads | generated | 500 |
| Engines | PG 16.4, MySQL 8.0.37, SQL Server 2022 | 3 |
| Isolation levels | READ COMMITTED, REPEATABLE READ, SERIALIZABLE, SNAPSHOT* | 4 |

*SNAPSHOT applies to PG (same as RR) and SQL Server (explicit mode); MySQL maps to RR.

**Total test points:** 500 × 3 × 4 = **6,000** (minus inapplicable engine/level pairs; effective ~5,400).

### 2.5 Measurement Protocol

For each (workload, engine, isolation_level, interleaving):

1. **Model prediction:** Run IsoSpec's engine-specific LTS for the target interleaving. Record predicted outcome: for each transaction T_i, predict COMMIT or ABORT, and reason (e.g., "SSI dangerous structure detected", "deadlock", "lock timeout", "serialization failure").

2. **Engine execution:** Execute on Docker container.
   - Container freshly reset between workloads (schema drop/recreate).
   - Connection pool: one connection per transaction.
   - Record: actual outcome per transaction (COMMIT/ABORT), any error codes, wall-clock timestamps per operation.

3. **Comparison:** Binary agreement per transaction outcome. Workload-level agreement = 1 iff all transactions agree.

### 2.6 Baseline

**Adya spec-level prediction (CLOTHO-style):** Run the same workloads through an Adya DSG-based anomaly detector that uses only the isolation *specification* (not engine-specific semantics). This predicts anomaly *possibility* per the formal spec but cannot predict engine-specific enforcement mechanisms (SSI false positives, gap-lock overshoot, SQL Server mode-dependent behavior).

This baseline isolates the value added by engine-faithful modeling (M1) over spec-level analysis.

### 2.7 Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| Agreement rate (PG) | #(model agrees with engine) / #(test points for PG) | ≥0.98 |
| Agreement rate (MySQL) | #(model agrees with engine) / #(test points for MySQL) | ≥0.95 |
| Agreement rate (SQL Server) | #(model agrees with engine) / #(test points for SQL Server) | ≥0.98 |
| Over-approximation rate (MySQL) | #(model predicts anomaly, engine never exhibits in 100 retries) / #(model-predicted anomalies for MySQL) | Report (expected: 5–15%) |
| Interleaving exclusion rate | #(unachievable interleavings) / #(total attempted) | ≤0.10 |
| Baseline delta | Agreement(IsoSpec) − Agreement(CLOTHO-style) | Report (expected: +15–30pp for MySQL) |

### 2.8 Statistical Plan

- **Point estimates:** Agreement rates per engine per isolation level.
- **Confidence intervals:** 95% Wilson score intervals (appropriate for proportions, well-behaved near 0 and 1).
- **Stratification:** Report agreement rates stratified by (engine, isolation level, workload complexity tier) to identify systematic failure modes.
- **MySQL over-approximation:** For each model-predicted anomaly, execute 100 retries with randomized timing jitter (±50ms per operation). If the anomaly is never observed, classify as over-approximation. Report the over-approximation rate with 95% Clopper-Pearson exact binomial CI.

### 2.9 Failure Analysis Protocol

For every disagreement (model ≠ engine):
1. Classify root cause: (a) model error, (b) interleaving control failure, (c) engine nondeterminism, (d) timing-dependent behavior.
2. If (a): file as model bug, fix model, re-run affected workloads, report both pre-fix and post-fix rates.
3. If (b)–(d): document and exclude with explicit justification.

---

## 3. Experiment 2: Known Anomaly Detection (Hermitage Benchmark)

### 3.1 Hypothesis

**H2:** IsoSpec detects:
- 100% of Hermitage-documented anomalies for PostgreSQL and MySQL,
- ≥95% of Hermitage-documented anomalies for SQL Server.

Additionally, IsoSpec achieves ≤5% false positive rate on engineered "safe" variants (workloads where the anomaly is provably impossible).

### 3.2 Hermitage Catalog Encoding

The Hermitage project (Kleppmann, 2014) documents ~47 anomaly patterns across engines, including:

| Anomaly Class | Hermitage Patterns | Engines Tested |
|--------------|-------------------|----------------|
| G0 (Write Cycles) | 3 patterns | PG, MySQL, SQL Server |
| G1a (Aborted Read) | 4 patterns | PG, MySQL, SQL Server |
| G1b (Intermediate Read) | 3 patterns | PG, MySQL, SQL Server |
| G1c (Circular Info Flow) | 5 patterns | PG, MySQL, SQL Server |
| G2-item (Write Skew) | 8 patterns | PG, MySQL, SQL Server |
| G2 (Anti-Dependency / Phantom) | 6 patterns | PG, MySQL, SQL Server |
| OTV (Observed Transaction Vanishes) | 4 patterns | PG, MySQL, SQL Server |
| PMP (Predicate-Many-Preceders) | 5 patterns | PG, MySQL, SQL Server |
| Lost Update | 4 patterns | PG, MySQL, SQL Server |
| Read Skew | 5 patterns | PG, MySQL, SQL Server |

**Encoding procedure:**
1. For each Hermitage pattern, create an IsoSpec workload encoding in the transaction IR.
2. Each encoding specifies: schema, transaction programs, target isolation level, expected anomaly class.
3. Verify encoding faithfulness: execute the Hermitage SQL verbatim on the Docker engine; confirm it matches Hermitage's documented outcome.

### 3.3 Safe Variant Construction

For each anomaly pattern, construct a "safe" variant by one of:
- Raising the isolation level to SERIALIZABLE (all engines).
- Adding explicit locking (`SELECT ... FOR UPDATE`) that prevents the anomaly.
- Restructuring transactions to eliminate the dependency cycle.

IsoSpec should report SAFE for all safe variants. Any UNSAFE report on a safe variant is a false positive.

**Count:** ~47 anomaly patterns × 2 (anomalous + safe) = **~94 test cases**.

### 3.4 Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| Recall (PG) | detected / total Hermitage anomalies for PG | 1.00 |
| Recall (MySQL) | detected / total Hermitage anomalies for MySQL | 1.00 |
| Recall (SQL Server) | detected / total Hermitage anomalies for SQL Server | ≥0.95 |
| Precision | true positives / (true positives + false positives on safe variants) | ≥0.95 (PG, SQL Server), ≥0.85 (MySQL) |
| Witness validity | #(witness scripts that reproduce anomaly on real engine) / #(witnesses generated) | ≥0.95 |

### 3.5 Baseline

**CLOTHO-style spec-level analysis:** Run the same Hermitage patterns through Adya DSG checking without engine-specific models. Predict which anomalies the *spec* allows.

**Expected baseline performance:**
- Recall: ~60–70% (misses engine-specific behaviors: SSI read-only optimization false positives, MySQL gap-lock-specific phantoms, SQL Server snapshot vs. lock mode differences).
- This demonstrates that spec-level analysis is *insufficient* for engine-faithful anomaly detection.

### 3.6 Execution Protocol

For each Hermitage pattern:
1. Encode in IsoSpec IR.
2. Run IsoSpec analysis. Record: detected (yes/no), anomaly class reported, analysis time, witness generated (yes/no).
3. If witness generated: execute on Docker engine. Record: anomaly reproduced (yes/no), execution time, retries needed.
4. Encode safe variant. Run IsoSpec. Record: safe reported (yes/no), any false positive anomaly class.
5. Run baseline (CLOTHO-style). Record same metrics.

Report results in a per-pattern table with columns: [Pattern | Engine | Level | Hermitage Says | IsoSpec Says | Baseline Says | Witness Confirmed?].

---

## 4. Experiment 3: Novel Divergence Discovery

### 4.1 Hypothesis

**H3:** IsoSpec's engine-faithful models, when systematically explored beyond the Hermitage catalog, reveal:
- ≥15 previously undocumented isolation behaviors,
- of which ≥3 affect common database migration paths (PG↔MySQL, PG↔SQL Server, MySQL↔SQL Server).

### 4.2 Novelty Criteria

A behavior is classified as **novel** if and only if ALL of the following hold:

1. **Not in Hermitage:** The behavior is not present in the Hermitage catalog (verified by manual comparison against the catalog's ~47 patterns).
2. **Not in engine release notes:** The behavior is not explicitly documented in the engine's release notes for the pinned version (PG 16.4 release notes, MySQL 8.0.37 changelogs, SQL Server 2022 CU12 KB articles).
3. **Confirmed by execution:** The behavior is reproduced on the real engine via Docker execution. The witness script runs successfully and demonstrates the claimed behavior.
4. **Clearly described:** A human-readable description of the behavioral difference from the Adya specification is provided, including: (a) which anomaly class is involved, (b) which engine-specific mechanism causes the divergence, (c) a minimal schedule demonstrating it.

A divergence is classified as **migration-affecting** if:
1. The behavior differs between two engines at the same nominal isolation level.
2. A workload that is safe on engine E₁ is unsafe on engine E₂ (or vice versa).
3. The workload pattern appears in at least one of: TPC-C, TPC-E, or the YCSB+T benchmark suite.

### 4.3 Exploration Domains

Systematic exploration targets five domains chosen for high divergence potential:

**Domain D1: Multi-table transactions with foreign key constraints**
- Rationale: FK enforcement interacts with lock acquisition order differently across engines. PG checks FKs at statement end; MySQL checks per-row; SQL Server behavior depends on cascading action type.
- Workload template: Parent-child tables with CASCADE/RESTRICT, concurrent INSERT to child + DELETE from parent.
- Parameter space: 2–3 tables, 1–2 FK relationships, k=2 transactions.

**Domain D2: Secondary index interactions (MySQL-specific)**
- Rationale: MySQL InnoDB's gap and next-key locking depends on which index the optimizer selects. Different index structures → different lock granularity → different anomaly profiles.
- Workload template: Table with and without secondary indexes. Same logical transaction, different index configurations.
- Parameter space: 0, 1, 3 secondary indexes per table; range predicates vs. point predicates.

**Domain D3: PostgreSQL SSI read-only optimization edge cases**
- Rationale: PG SSI has a "safe snapshot" optimization for read-only transactions running in a snapshot that precedes all concurrent writers. This optimization can cause false negatives (missing a genuine rw-dependency) or false positives (aborting unnecessarily).
- Workload template: Read-only transaction T_r concurrent with two writing transactions T_w1, T_w2 forming a potential dangerous structure.
- Parameter space: Vary snapshot timing, read-set overlap with write-sets, number of concurrent writers {2, 3, 4}.

**Domain D4: SQL Server dual-mode behavioral differences**
- Rationale: SQL Server 2022 supports both lock-based (`SET TRANSACTION ISOLATION LEVEL REPEATABLE READ`) and snapshot-based (`SET TRANSACTION ISOLATION LEVEL SNAPSHOT` or `READ_COMMITTED_SNAPSHOT ON`) concurrency. The same nominal isolation level can exhibit different behaviors depending on mode.
- Workload template: Same transaction program, executed under lock-based and snapshot-based modes.
- Parameter space: All combinations of (READ COMMITTED, REPEATABLE READ, SERIALIZABLE) × (lock-based, snapshot-based).

**Domain D5: Index-dependent phantom behavior**
- Rationale: Phantom prevention depends on predicate locking. PG uses SIREAD locks (predicate-level); MySQL uses gap locks (index-dependent); SQL Server uses key-range locks (index-dependent). Different engines may or may not prevent the same phantom depending on index availability.
- Workload template: Range scan + concurrent INSERT within range, with varying index configurations.
- Parameter space: Point predicates, range predicates (< , BETWEEN), conjunctive predicates.

### 4.4 Exploration Procedure

For each domain D_i:

1. **Generate candidate workloads:** 50–100 workloads per domain using parameterized templates.
2. **Run IsoSpec differential analysis:** For each workload, analyze on all 3 engines × relevant isolation levels. Flag workloads where model predictions differ between engines or between model and spec.
3. **Triage by divergence type:**
   - Type A: Engine permits anomaly that spec forbids (engine weaker than spec).
   - Type B: Engine prevents anomaly that spec permits (engine stronger than spec).
   - Type C: Two engines disagree on same workload at same nominal level (portability violation).
4. **Dynamic confirmation:** For each flagged divergence:
   - Generate witness script from SMT counterexample (M1 → witness synthesis).
   - Execute on Docker engine. Record outcome.
   - If confirmed: verify novelty criteria (Section 4.2).
   - If not confirmed: classify as model over-approximation or interleaving control failure. Record separately.
5. **Minimization:** For confirmed novel divergences, use MUS (Minimal Unsatisfiable Subset) extraction to minimize the witness schedule. Target: ≤8 operations in the witness.

### 4.5 Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| Novel divergences found | Count meeting all 4 novelty criteria | ≥15 |
| Migration-affecting divergences | Count meeting all migration-affecting criteria | ≥3 |
| Confirmation rate | confirmed / flagged by model | Report (expected: ≥70%) |
| Over-approximation rate | model-flagged but never reproduced / total flagged | Report (expected: ≤30%) |
| Witness minimality | median operations in minimized witness | Report (target: ≤8) |
| Witness reproducibility | #(witnesses that reproduce on fresh Docker instance) / total witnesses | ≥0.95 |

### 4.6 Reporting

Each novel divergence is documented with:
1. **Identifier:** ISOSPEC-NOVEL-{NNN}
2. **Title:** One-line description (e.g., "MySQL gap-lock phantom under secondary index removal at REPEATABLE READ")
3. **Engines affected:** Which engines, which isolation levels
4. **Anomaly class:** Adya classification (G0–G2)
5. **Root cause:** Which engine-specific mechanism (SSI rw-dependency, gap lock, snapshot SCN, etc.)
6. **Migration impact:** Which migration direction is affected; severity (silent data corruption vs. unexpected abort)
7. **Minimal witness:** SQL script (≤50 lines) executable on Docker
8. **Reproduction command:** `docker exec ... < witness.sql`

---

## 5. Experiment 4: Cross-Engine Portability Analysis

### 5.1 Hypothesis

**H4:** IsoSpec identifies portability violations in standard benchmark transactions (TPC-C, TPC-E subset) when migrating between engine pairs, with:
- Analysis completing in <30 seconds per workload per engine pair,
- ≥90% witness confirmation rate for reported violations.

### 5.2 Benchmark Encodings

**TPC-C (5 transaction types):**

| Transaction | Key Operations | Portability Risk |
|-------------|---------------|-----------------|
| New-Order | INSERT order + order_lines, UPDATE stock (multi-row), READ district | FK + range lock interaction |
| Payment | UPDATE warehouse + district + customer, INSERT history | Write-write conflict resolution |
| Order-Status | READ customer + orders + order_lines (read-only) | SSI read-only optimization |
| Delivery | UPDATE + DELETE across new_order + order | Multi-table lock ordering |
| Stock-Level | READ stock + order_line (range scan, read-only) | Phantom behavior |

**TPC-E subset (10 transaction types):**

Selected for maximum portability risk based on predicate complexity and multi-table interaction:
- Trade-Order, Trade-Result, Trade-Update, Market-Feed, Trade-Lookup,
- Customer-Position, Broker-Volume, Security-Detail, Trade-Status, Data-Maintenance.

**Encoding protocol:**
1. Extract transaction logic from TPC-C/E specification documents.
2. Translate to IsoSpec transaction IR, preserving: table accesses, predicate structure, operation ordering, isolation level annotations.
3. Parameterize with concrete data ranges (TPC-C: 10 warehouses; TPC-E: 1000 customers).
4. Review encoding for faithfulness against spec (checklist: every SQL statement mapped, every predicate captured, every FK relationship modeled).

### 5.3 Analysis Matrix

| Migration Direction | Isolation Levels | Engine Pair |
|--------------------|------------------|-------------|
| PG → MySQL | RC, RR, SER | PG 16.4 → MySQL 8.0.37 |
| MySQL → PG | RC, RR, SER | MySQL 8.0.37 → PG 16.4 |
| PG → SQL Server | RC, RR, SER, SNAPSHOT | PG 16.4 → SQL Server 2022 |
| SQL Server → PG | RC, RR, SER, SNAPSHOT | SQL Server 2022 → PG 16.4 |
| MySQL → SQL Server | RC, RR, SER | MySQL 8.0.37 → SQL Server 2022 |
| SQL Server → MySQL | RC, RR, SER | SQL Server 2022 → MySQL 8.0.37 |

**Total analysis runs:** 15 workloads × 6 directions × 3–4 levels = **~300 analysis runs**.

### 5.4 Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| Violations detected | Count of workload × level × direction with portability violation | Report |
| Witness confirmation rate | #(confirmed by Docker execution on both engines) / #(violations reported) | ≥0.90 |
| False positive rate | 1 − witness confirmation rate | ≤0.10 |
| Analysis time | Wall-clock seconds per workload × engine pair | <30s |
| Violation severity | Classified: (a) silent data corruption, (b) unexpected abort, (c) performance degradation | Report distribution |

### 5.5 Baseline

**Manual expert analysis:** Compile known portability issues for TPC-C from published migration guides and database documentation:
- PostgreSQL wiki: "Migrating from MySQL" / "Migrating from SQL Server"
- AWS DMS documentation: known transaction semantic limitations
- Published blog posts and conference papers on cross-engine TPC-C behavior

Record: how many of IsoSpec's detected violations were previously known vs. novel.

### 5.6 Witness Validation Protocol

For each reported portability violation:
1. Extract witness schedule from IsoSpec.
2. Generate engine-dialect-correct SQL for both source and target engines.
3. Execute on source engine Docker instance: confirm SAFE (no anomaly).
4. Execute on target engine Docker instance: confirm UNSAFE (anomaly manifests).
5. If both confirm: mark as TRUE POSITIVE.
6. If source engine also exhibits anomaly: mark as FALSE POSITIVE (violation is not migration-specific).
7. If target engine does not exhibit anomaly: mark as UNCONFIRMED; retry 50 times with timing jitter; if still unconfirmed, classify as model over-approximation.

---

## 6. Experiment 5: Real-World Migration Case Study

### 6.1 Hypothesis

**H5:** IsoSpec detects isolation-related migration bugs from documented real-world incidents, producing witness scripts that reproduce the reported behavior.

*This experiment has no fixed numerical threshold—it is a qualitative demonstration. Even one compelling case where IsoSpec catches a real-world bug is a significant contribution.*

### 6.2 Case Source Selection

Systematically search for documented migration incidents in:

**Source 1: PostgreSQL mailing lists (pgsql-general, pgsql-bugs)**
- Search terms: "migration" AND ("isolation" OR "anomaly" OR "serialization failure" OR "write skew" OR "phantom")
- Date range: 2018–2025
- Target: ≥5 candidate incidents

**Source 2: AWS DMS / Azure DMS bug reports**
- GitHub issues for aws-dms-*, Azure Data Migration Service
- Search terms: "transaction" AND ("inconsistency" OR "data loss" OR "conflict")
- Target: ≥3 candidate incidents

**Source 3: Stack Overflow**
- Tags: [database-migration] + [transaction-isolation] or [concurrency]
- Filter: questions with accepted answers describing a behavioral difference
- Target: ≥5 candidate incidents

**Source 4: Technical blog posts and postmortems**
- Search: "database migration" + "isolation" + "bug" (Google Scholar, Hacker News)
- Target: ≥3 candidate incidents

**Selection criteria for inclusion:**
1. The incident describes a *behavioral difference* between two database engines (not a schema or data type issue).
2. The incident involves transaction isolation or concurrency (not single-statement queries).
3. Enough detail is provided to reconstruct the workload (transaction structure, approximate SQL, isolation level).

**Target:** 2–3 fully encoded cases that meet all criteria.

### 6.3 Encoding Protocol

For each selected case:

1. **Extract workload description:** Identify the transaction programs, table schemas, isolation levels, and the reported anomaly.
2. **Abstract to IsoSpec IR:** Translate the real-world workload into the IsoSpec transaction IR. Document any abstractions or simplifications made (e.g., replacing application-specific logic with equivalent SQL patterns).
3. **Encode both source and target engine configurations:** Set up analysis for the migration direction described in the incident.

### 6.4 Analysis and Validation

For each encoded case:

1. **Run IsoSpec analysis:** Analyze the workload for the migration direction. Record: violation detected (yes/no), anomaly class, witness generated (yes/no), analysis time.
2. **Check detection:** Does IsoSpec flag the same issue described in the incident?
3. **Generate witness:** If violation detected, extract the witness script.
4. **Reproduce on Docker:**
   - Execute witness on source engine: verify safe behavior.
   - Execute witness on target engine: verify the documented bug manifests.
5. **Compare with incident description:** Does the witness match the reported symptoms?

### 6.5 Reporting

Each case study is documented as a self-contained narrative:

```
CASE STUDY: [Title from incident]
SOURCE: [URL/reference]
MIGRATION: [Engine A] → [Engine B] at [isolation level]
INCIDENT SUMMARY: [2–3 sentence description of what went wrong]
ISOSPEC ENCODING: [Transaction IR, reproduced in full]
ISOSPEC RESULT: [Detected / Not detected; anomaly class; analysis time]
WITNESS SCRIPT: [Full SQL, ≤50 lines]
REPRODUCTION: [Docker commands to reproduce]
MATCH: [Does IsoSpec's output match the documented behavior?]
```

### 6.6 Impact Assessment

For each detected case:
- **Severity:** Would this cause data corruption, incorrect query results, or application errors?
- **Prevalence:** How common is this migration direction? (Reference: DB-Engines ranking, cloud migration statistics.)
- **Preventability:** Could a developer have caught this with existing tools? (Compare against IsoSpec's unique capabilities.)

---

## 7. Experiment 6: Scalability Benchmarks

### 7.1 Hypothesis

**H6:** IsoSpec completes analysis in:
- <10 seconds for typical OLTP workloads (k ≤ 5 transactions, n ≤ 20 operations each),
- <60 seconds for stress-test workloads (k ≤ 10 transactions, n ≤ 50 operations each).

### 7.2 Parameter Space

Independent parameter variation with all other parameters at baseline values:

| Parameter | Values | Baseline |
|-----------|--------|----------|
| k (transactions) | 2, 3, 5, 7, 10 | 3 |
| n (operations/txn) | 5, 10, 20, 30, 50 | 10 |
| tables | 1, 3, 5, 10 | 3 |
| columns/table | 1, 3, 5, 10 | 3 |
| secondary indexes (MySQL) | 0, 1, 3, 5 | 1 |
| predicate complexity | point, range, conjunctive-2, conjunctive-3 | range |
| isolation level | RC, RR, SER | SER |

**Total configurations:** ~5 × 5 × 4 × 4 × 4 × 4 × 3 = 19,200 — subsampled to **500 configurations** via Latin hypercube sampling for uniform coverage with fixed seed.

### 7.3 Measurement Protocol

For each configuration × each engine:

1. Generate workload deterministically (same seed scheme as Experiment 1).
2. Run IsoSpec analysis **5 times** (cold start: fresh process each time).
3. Record per run:
   - **Wall-clock time** (end-to-end, including parsing and witness synthesis).
   - **Z3 solving time** (SMT check-sat calls only, via Z3 statistics API).
   - **Formula size:** number of Boolean variables, number of clauses (CNF), number of quantifier-free assertions.
   - **Memory:** peak RSS (via `/proc/self/status` or `getrusage`).
4. Report: **median** wall-clock time, **interquartile range (IQR)**, Z3 time fraction (Z3_time / wall_clock).

### 7.4 Z3 Timeout Policy

- **Timeout:** 120 seconds per Z3 check-sat call.
- If Z3 times out, record the configuration as TIMEOUT. Do not retry.
- **Timeout rate metric:** #(timeouts) / #(total configurations × engines). Target: ≤5% for OLTP-sized workloads (k ≤ 5, n ≤ 20).

### 7.5 Scaling Law Analysis

Fit empirical scaling models to the measured data:

1. **Primary model:** `T(k, n) = α · k^β · n^γ` (power law). Report fitted (α, β, γ) with R² and residual plots.
2. **Engine comparison:** Overlay scaling curves for all 3 engines. Determine if engine-specific constraints *prune* the SMT search space (hypothesis: engine models have smaller formula sizes than abstract Adya-level analysis for equivalent workloads).
3. **Bottleneck analysis:** For each parameter, plot: wall-clock time vs. parameter value (other parameters at baseline). Identify the dominant scaling factor.

### 7.6 Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| Median time (OLTP) | Median wall-clock, k ≤ 5 ∧ n ≤ 20 | <10s |
| Median time (stress) | Median wall-clock, k ≤ 10 ∧ n ≤ 50 | <60s |
| Timeout rate (OLTP) | #timeouts / #OLTP configs | ≤0.05 |
| Timeout rate (stress) | #timeouts / #stress configs | Report |
| Z3 fraction | Median(Z3_time / wall_clock) | Report |
| Formula growth rate | β, γ in power law fit | Report |
| Engine pruning effect | Formula_size(engine) / Formula_size(spec) | Report (expected: <1.0) |

### 7.7 Comparison Points

| Workload Size | IsoSpec Target | CLOTHO (reported) | Elle (typical) |
|--------------|---------------|-------------------|----------------|
| k=2, n=10 | <2s | ~1s (abstract) | N/A (dynamic) |
| k=5, n=20 | <10s | ~30s (abstract) | ~60s (execution) |
| k=10, n=50 | <60s | timeout (abstract) | ~300s (execution) |

*CLOTHO times are estimated from published results on similar-sized consistency checking problems. Elle times are estimated from Jepsen blog posts.*

---

## 8. Baseline Comparison Matrix

### 8.1 Tool Comparison

| Tool | Scope | Engine-Specific Models? | Predicate-Aware (M5)? | Executable Witness? | Fully Automated? | Completeness |
|------|-------|:-:|:-:|:-:|:-:|:-:|
| **CLOTHO** (Rahmani et al., OOPSLA'19) | Abstract consistency models (causal, eventual, etc.) | No | Limited (item-level) | Yes (abstract schedules) | Yes | Bounded |
| **Elle** (Kingsbury & Alvaro, VLDB'20) | Observed history checking | No | No | N/A (post-hoc) | Yes | Empirical |
| **Jepsen** (Kingsbury) | Partition + isolation testing | No | No | Yes (dynamic trace) | Semi-automated | Empirical |
| **Hermitage** (Kleppmann, 2014) | Fixed anomaly catalog | Engine-observed (not modeled) | Fixed patterns only | Yes (manual SQL) | Manual | Incomplete |
| **dbcop** (Tan et al., PLDI'21) | Consistency checking (SI, SER) | No | No | Yes | Yes | Bounded |
| **MonkeyDB** (Biswas et al., VLDB'21) | Random isolation testing | Simulated (configurable) | No | Statistical | Yes | Probabilistic |
| **2PL/SSI analysis** (textbook) | Theoretical lock analysis | Textbook-level | No | No | Manual | Theoretical |
| **IsoSpec** (this work) | **Engine-faithful isolation models** | **Yes (3 production engines)** | **Yes (conjunctive inequality fragment)** | **Yes (engine-dialect SQL)** | **Yes** | **Bounded-complete** |

### 8.2 Key Differentiators

**IsoSpec vs. CLOTHO:** CLOTHO models what the *specification says* an engine should do. IsoSpec models what the engine *actually does*. This distinction matters because every production engine deviates from its specification. CLOTHO cannot detect MySQL gap-lock-dependent phantoms, PG SSI read-only optimization edge cases, or SQL Server dual-mode behavioral differences—IsoSpec can.

**IsoSpec vs. Elle/Jepsen:** Elle and Jepsen are *dynamic* tools—they execute transactions and observe outcomes. IsoSpec is *static*—it analyzes transaction programs before execution and finds anomalies that may require specific interleavings to manifest. Dynamic tools may miss rare interleavings; IsoSpec finds them by construction (within bounds).

**IsoSpec vs. Hermitage:** Hermitage documents a fixed catalog of ~47 anomaly patterns. IsoSpec analyzes *arbitrary* user-provided transaction programs. Hermitage cannot answer "will MY workload break on migration?"—IsoSpec can.

**IsoSpec vs. dbcop:** dbcop checks whether an observed execution history satisfies a consistency model. IsoSpec checks whether a transaction *program* can produce an anomalous history. IsoSpec operates before execution; dbcop operates after.

**IsoSpec vs. MonkeyDB:** MonkeyDB simulates weak isolation by randomly injecting anomalies. It provides *statistical* coverage. IsoSpec provides *formal guarantees* within bounds: if an anomaly exists in the bounded search space, IsoSpec finds it.

---

## 9. Threats to Validity

### 9.1 Internal Validity

| Threat | Severity | Mitigation |
|--------|----------|------------|
| **Interleaving forcing is approximate.** Advisory locks and sleep-based barriers may not achieve all target interleavings, especially under engine-internal lock scheduling nondeterminism. | Medium | Retry protocol (10 attempts). Report exclusion rate. Accept ≤10% failure rate. Analyze excluded interleavings for systematic bias. |
| **Docker container timing differs from production.** Container overhead, shared host resources, and virtualization layers may affect timing-sensitive concurrency behaviors. | Low | Pin Docker resource limits (CPU: 2 cores, memory: 4GB per engine). Use advisory-lock barriers (not timing-based) for interleaving control. Validate representative subset on bare-metal. |
| **Engine minor version drift.** Even within pinned major versions, minor/patch updates may change concurrency behavior. | Low | Pin exact version by Docker image SHA256 digest. Document digest in artifact. Note: results are valid for pinned version only. |
| **Workload generation bias.** Pseudorandom workload generation may systematically miss certain anomaly patterns. | Medium | Supplement random workloads with structured domain exploration (Experiment 3). Verify coverage of all Adya anomaly classes in generated workloads. |
| **Z3 solver nondeterminism.** Z3 may produce different results across runs due to internal heuristics. | Low | Pin Z3 version. Set `smt.random_seed=42`. Verify determinism: run 10 identical queries, confirm identical results. |

### 9.2 External Validity

| Threat | Severity | Mitigation |
|--------|----------|------------|
| **Three engines may not generalize.** PostgreSQL, MySQL, and SQL Server represent ~70% of the relational market but do not cover Oracle, SQLite, CockroachDB, TiDB, etc. | Medium | Three engines cover the three major concurrency paradigms: SSI (PG), 2PL with gap locks (MySQL), dual-mode pessimistic/optimistic (SQL Server). Architecture extends to additional engines; demonstrate extensibility with a fourth-engine sketch (Oracle or CockroachDB) in the paper. |
| **Conjunctive inequality fragment may not cover all predicates.** Real workloads use LIKE, IN, EXISTS, subqueries, UDFs. | Medium | Quantify coverage: analyze TPC-C and TPC-E predicate distributions. Report fraction of predicates in the decidable fragment. For predicates outside the fragment, IsoSpec conservatively over-approximates (sound but incomplete). |
| **TPC-C/E may not represent all workloads.** Real OLTP workloads have application-specific transaction structures. | Low | TPC-C is the industry-standard OLTP benchmark. Supplement with real-world case studies (Experiment 5). Note: IsoSpec accepts arbitrary workloads; TPC-C/E are evaluation vehicles, not limitations. |
| **Bounded analysis may miss anomalies requiring larger bounds.** IsoSpec's bounded-completeness guarantee holds only within the bound. | Medium | Report the bound used for each experiment. Demonstrate empirically: for Hermitage patterns, all anomalies manifest within k=3, n=10. For TPC-C, test with increasing bounds; report convergence. |

### 9.3 Construct Validity

| Threat | Severity | Mitigation |
|--------|----------|------------|
| **"Novel divergence" requires judgment.** Deciding whether a behavior is documented requires searching engine release notes and mailing lists, which may miss obscure documentation. | Medium | Apply novelty criteria (Section 4.2) mechanically. Provide all evidence for each claim. Invite engine maintainers to review claims (acknowledged in paper if they respond). |
| **Model adequacy measured by bounded workloads.** Agreement on bounded workloads does not guarantee agreement on unbounded workloads. | Low | Report the bound. Note: all practical OLTP transactions are bounded. For k≤3, n≤10, the bound covers the vast majority of real-world concurrent transaction patterns. |
| **"Migration-affecting" classification is subjective.** Whether a workload pattern appears in TPC-C/E is clear; whether it appears in "real workloads" is not. | Low | Restrict migration-affecting claims to TPC-C/E patterns only. Real-world case studies (Experiment 5) provide additional qualitative evidence. |

---

## 10. Compute Budget Allocation

### 10.1 Budget Breakdown

| Experiment | Estimated CPU-Hours | Breakdown | Parallelizable? |
|-----------|:---:|-----------|:-:|
| **1: Model Adequacy** | 20–30 | 6000 test points × ~15s model + ~30s Docker exec = ~75h sequential; 3-way engine parallelism → 25h | Yes (per engine) |
| **2: Hermitage** | 2–5 | ~94 test cases × ~2min each = ~3h sequential; 3-way parallel → ~1h | Yes (per pattern) |
| **3: Novel Discovery** | 10–20 | 250–500 candidate workloads × model + Docker validation; exploration is iterative | Partially |
| **4: Portability** | 5–10 | ~300 analysis runs × <30s each + Docker witness validation | Yes (per pair) |
| **5: Case Study** | 2–5 | Manual encoding + analysis + validation for 2–3 cases | No |
| **6: Scalability** | 10–20 | 500 configs × 3 engines × 5 runs × varying time | Yes (per config) |
| **Total** | **49–90** | | **Within 100h** |

### 10.2 Execution Order (Critical Path)

```
Phase 1 (Weeks 1-2): Experiments 1 + 2 in parallel
  ├── Experiment 1: Model adequacy (runs continuously, engine-parallel)
  └── Experiment 2: Hermitage (completes quickly, validates tool correctness)

Phase 2 (Weeks 2-3): Experiment 3 (depends on model correctness from Phase 1)
  └── Novel discovery (iterative: generate → analyze → validate → refine)

Phase 3 (Weeks 3-4): Experiments 4 + 5 + 6 in parallel
  ├── Experiment 4: Portability (depends on validated models)
  ├── Experiment 5: Case study (depends on validated tool pipeline)
  └── Experiment 6: Scalability (independent of other experiments)
```

### 10.3 Contingency

If compute budget approaches 80h after Phase 2:
1. Reduce Experiment 6 parameter space (500 → 200 configurations via more aggressive subsampling).
2. Reduce Experiment 1 workload count (500 → 300 workloads) if agreement rates stabilize early.
3. Experiment 5 is non-negotiable: even at reduced scope (1 case study), it is the highest-impact qualitative result.

---

## 11. Artifact Checklist

The evaluation artifact must include all items for reproducibility:

### 11.1 Infrastructure

- [ ] `Dockerfile` for each engine (PG 16.4, MySQL 8.0.37, SQL Server 2022 CU12) with SHA256 digest
- [ ] `docker-compose.yml` orchestrating all three engines
- [ ] `Makefile` or equivalent with targets: `setup`, `run-exp-{1..6}`, `collect-results`, `generate-tables`
- [ ] Z3 binary (pinned version 4.12.6) or build instructions
- [ ] Rust toolchain specification (`rust-toolchain.toml`)

### 11.2 Workloads

- [ ] Generated workloads for Experiment 1 (or generation script with fixed seed)
- [ ] Hermitage catalog encodings (Experiment 2) with original-to-IR mapping documentation
- [ ] Exploration domain templates (Experiment 3)
- [ ] TPC-C / TPC-E encodings (Experiment 4)
- [ ] Real-world case study encodings (Experiment 5) with source references
- [ ] Scalability benchmark configurations (Experiment 6)

### 11.3 Results

- [ ] Raw results: CSV files per experiment with columns matching defined metrics
- [ ] Aggregate results: tables matching paper figures/tables
- [ ] Confidence intervals: computed from raw data using scripts in artifact
- [ ] Witness scripts: one `.sql` file per confirmed divergence/violation, with Docker execution instructions
- [ ] Analysis notebooks or scripts reproducing all figures and tables from raw data

### 11.4 Reproducibility Verification

- [ ] `REPRODUCE.md` with step-by-step instructions (estimated time: <24h on laptop with 8+ cores)
- [ ] Expected output hashes for deterministic experiments (Experiments 2, 6)
- [ ] Known nondeterminism sources documented (Experiment 1 interleaving forcing, Experiment 3 timing-dependent confirmations)

---

*Document generated for IsoSpec theory phase. All thresholds, metrics, and methods are specified pre-experiment and must not be retroactively adjusted. Deviations from this protocol must be documented and justified in the final paper.*

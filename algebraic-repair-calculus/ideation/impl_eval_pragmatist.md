# Implementation Evaluation — Pragmatist Review

**Date**: 2026-03-04
**Evaluator**: Pragmatist (team-based verification with Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer, cross-critique resolution, and independent verifier signoff)

---

## proposal_00 — Algebraic Repair Calculus (ARC)

**Claimed LoC**: 50,247 | **Actual source LoC**: ~29,800 (non-empty, non-comment) | **Test LoC**: ~14,700
**Timed out**: Yes | **Polish rounds**: 0 | **Has tests**: Yes

### Summary

ARC implements a three-sorted delta algebra (schema monoid, data group, quality lattice) for incremental pipeline repair. The algebraic core is genuinely implemented — not stubs. The SQL column-level lineage (via sqlglot), DP/LP planners (via scipy.linprog), and quality monitoring (KS/PSI/Chi²/JSD via scipy.stats) are all real algorithms. However, the implementation timed out before integration was completed: two incompatible `PipelineGraph` classes and two incompatible `CompoundPerturbation` classes prevent end-to-end operation. Both examples crash. The bounded commutation theorem — the project's signature theoretical contribution — is untestable in the current state.

---

### Scores

| Criterion | Score | Justification |
|-----------|-------|---------------|
| **Code quality** | **6/10** | 44K total LoC (29.8K source + 14.7K test), well-structured with attrs/frozen dataclasses, only 1.9% stubs (40/2,071 functions). But dual `PipelineGraph` and dual `CompoundPerturbation` type hierarchies are a fundamental design flaw that prevents composition. Build backend in pyproject.toml is correct (setuptools.build_meta). |
| **Genuine difficulty** | **8/10** | Three properly-structured algebraic sorts with real operations (compose, inverse, identity, push, annihilation). LP relaxation with randomized rounding + greedy feasibility patching. DP planner with memoization and annihilation detection. Column-level SQL lineage through CTEs, window functions, set operations. This is genuinely hard algorithmic work, not boilerplate. |
| **Value delivered** | **4/10** | Individual components are impressive in isolation, but the system cannot be used end-to-end. Both examples crash (property-vs-method confusion, type mismatches). A data engineer cannot use this today. The bounded commutation theorem is unverifiable. The ~12 deterministic test failures all sit at the critical integration boundary. |
| **Test coverage** | **7/10** | 2,053 tests collected; ~2,037 pass deterministically (99.2% unit pass rate). Rigorous methodology: Hypothesis property-based tests for algebraic laws, DuckDB integration tests, statistical threshold assertions. However, all integration tests fail due to the type hierarchy mismatch — the tests that matter most don't work. |
| **Real-world format support** | **4/10** | SQL parsing via sqlglot works well (all 139 parser tests pass). YAML/JSON pipeline spec serialization with validation, env interpolation, templates. But: zero CSV/Parquet/database I/O capability. Pandas/PySpark "analyzers" only walk Python ASTs to recognize patterns — they don't read data. No actual dbt/Airflow integration. No Avro/Protobuf schema support. |

---

### Detailed Findings

#### What Works (verified by independent code inspection)

1. **Three-sorted delta algebra** — `schema_delta.py` (monoid: compose, inverse, identity, normalize), `data_delta.py` (group: compose, inverse, zero, compress), `quality_delta.py` (lattice: join, meet, top, bottom). All have non-trivial implementations with conflict resolution, normalization, cancellation. Unit tests pass cleanly.

2. **SQL column-level lineage** — `lineage.py` traces columns through SELECT, JOIN (7 types), GROUP BY, CTE, subqueries, window functions, set operations. Builds multi-query lineage graphs with upstream DFS traversal. All 139 parser tests pass.

3. **DP planner** — `dp.py` implements genuine bottom-up DP with memo table, topological ordering, annihilation detection, comparison of recompute/incremental/schema-migrate strategies.

4. **LP planner** — `lp.py` uses scipy.optimize.linprog with randomized rounding, greedy feasibility patching, and local search improvement. Real approximation algorithm, not a wrapper.

5. **Quality monitoring** — KS test, PSI, Chi², Jensen-Shannon divergence via scipy/numpy. 10 constraint types with automated inference. Anomaly detection (row count, mean shift, variance change).

6. **CPU-only execution** — Zero GPU/CUDA/ML framework dependencies. Runs entirely on CPU with sqlglot, networkx, scipy, duckdb, numpy.

#### What's Broken (verified)

1. **Dual `PipelineGraph`**: `arc/types/base.py:1742` vs `arc/graph/pipeline.py:195`. Builder produces one type, planners consume the other. Incompatible APIs (`is_acyclic()` vs `is_dag()`, `parents()` vs `predecessors()`).

2. **Dual `CompoundPerturbation`**: `arc/types/base.py` vs `arc/algebra/composition.py`. Examples import from composition, planners import from types.base. Different interfaces.

3. **AddColumn API mismatch**: `execution/engine.py` expects `op.op_type`/`op.column_name`, but algebra layer provides `op.name`/`op.sql_type`. Breaks all schema-change execution paths.

4. **Both examples crash**: Property-vs-method confusion (`graph.node_count()` but it's a `@property`), enum naming (`SQLType.INTEGER` vs `SQLType.INT`), type hierarchy mismatches.

5. **Bounded commutation theorem untestable**: The core correctness guarantee `apply(repair(σ), state(G)) = recompute(evolve(G, σ))` cannot be verified because the execution engine can't apply schema deltas from the algebra layer.

#### Fix Effort Estimate

The integration failures trace to ~50 lines of code changes across 3-4 files. The conceptual fix is straightforward: unify the type hierarchies or add adapter layers. Realistic effort: **2-4 hours** for someone understanding the codebase, primarily to reconcile the two parallel type systems.

---

### Pragmatist Constraints Check

| Constraint | Pass/Fail | Notes |
|------------|-----------|-------|
| Buildable on laptop in a day (150K LoC/day capacity) | ✅ PASS | 50K LoC well within budget |
| Runs on laptop CPU | ✅ PASS | No GPU/CUDA dependencies |
| No human annotation | ✅ PASS | All automated |
| Fully automated evaluation | ⚠️ PARTIAL | Tests exist but integration tests fail; no working end-to-end demo |
| Extreme and obvious value | ❌ FAIL | Cannot demonstrate value — examples crash, no working pipeline |
| Who specifically needs this | ⚠️ PARTIAL | Data engineers maintaining pipelines — but they can't use it in current state |

---

### Team Disagreements Resolved

| Issue | Auditor | Skeptic | Synthesizer | Ground Truth |
|-------|---------|---------|-------------|--------------|
| Test pass rate | 88.1% | 99.0% | 96.3% | **99.2%** (Hypothesis non-determinism explains variation; `-x` flag inflated Auditor's failures) |
| Stub percentage | 0.17% | 0% | 50% push.py, 70% annihilation.py | **1.9%** (40/2,071 functions). Synthesizer was wrong about push.py and annihilation.py |
| SQL parser tests | 98/98 fail | 133/139 pass | not tested | **139/139 pass** (100%) |
| Fix effort | — | ~20 lines | 2-4 hours | **~50 lines, 2-4 hours** |
| Completion level | — | 95% | — | **~93% unit-level, ~75% integration** |

---

### Comparison to Alternatives

| Feature | ARC | sqlglot (standalone) | dbt | Great Expectations |
|---------|-----|---------------------|-----|--------------------|
| Column-level lineage | ✅ | ✅ | ✅ (via manifest) | ❌ |
| Schema change impact analysis | ✅ (algebraic) | ❌ | ✅ (basic) | ❌ |
| Automated repair planning | ✅ (DP/LP) | ❌ | ❌ | ❌ |
| Quality drift detection | ✅ (KS/PSI) | ❌ | ❌ | ✅ (richer) |
| Determinism checking | ✅ (novel) | ❌ | ❌ | ❌ |
| Actually works E2E | ❌ | ✅ | ✅ | ✅ |

ARC's unique contribution is the integration of lineage + schema impact + repair planning. No existing tool does this. But none of those existing tools have the "doesn't actually work" problem.

---

### VERDICT: **CONTINUE** (with reservations)

**Reasoning**: The algorithmic core is genuinely impressive and non-trivial — three real algebraic sorts, real DP/LP planners, real SQL lineage, real quality monitoring. The 44K LoC is not padding; only 1.9% are stubs. The 99.2% unit test pass rate demonstrates that individual components work. The integration failures are concentrated at module boundaries (dual type hierarchies, naming mismatches) and are fixable with ~50 lines of changes.

However, "with reservations" means:
1. The system delivers **zero end-to-end value today** — both examples crash, integration tests fail
2. The signature theoretical contribution (bounded commutation theorem) is **unverifiable** in the current state
3. Without CSV/Parquet/database I/O, the system has **no real-world data connectivity**
4. A polish round MUST fix the type hierarchy unification before any value claims are credible

**What must happen in the next round**:
- Unify the dual `PipelineGraph` and `CompoundPerturbation` classes
- Fix the `AddColumn` API mismatch in execution/engine.py and propagation.py
- Get at least one example running end-to-end
- Verify the bounded commutation theorem with a passing test

**If those fixes land**: This becomes a 7/10 value-delivered implementation with a genuine novel contribution.
**If they don't**: This is an impressive codebase that proves nothing.

---

### Verification Chain

| Step | Agent | Status |
|------|-------|--------|
| Independent exploration | Auditor, Skeptic, Synthesizer (parallel) | ✅ Complete |
| Cross-critique resolution | Team lead | ✅ 5 disagreements resolved with evidence |
| Independent verification | Final verifier | ✅ 8/8 claims verified against source code |
| Signoff | Final verifier | ✅ Scores adjusted (code quality 7→6, value 5→4, format support 5→4) |

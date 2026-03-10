# Implementation Evaluation: Algebraic Repair Calculus (ARC)

**Evaluator**: Senior Systems Engineer (100K+ LoC experience)
**Date**: 2026-03-04
**Method**: Claude Code Agent Teams — Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer, adversarial critique round, lead verification

---

## proposal_00

**Lines of code**: 38,876 source (arc/) + 20,908 test (tests/) = 59,784 total
**Timed out**: Yes (0 polish rounds completed)
**Tests**: 1,989 collected → 1,820 passed, 133 failed, 34 skipped (91.5% unit pass rate)

### Architecture Overview

```
arc/
├── algebra/        # Three-sorted delta algebra (8 files, ~6,900 LoC)
│   ├── composition.py      # CompoundPerturbation compose with φ/ψ
│   ├── push.py             # 24 push operators (8 SQL ops × 3 delta sorts)
│   ├── propagation.py      # Delta propagation through pipeline DAGs
│   ├── interaction.py       # φ/ψ interaction homomorphisms
│   ├── annihilation.py     # Delta cancellation detection
│   ├── schema_delta.py     # Δ_S monoid (add/drop/rename/retype columns)
│   ├── data_delta.py       # Δ_D group (insert/delete/update on multisets)
│   └── quality_delta.py    # Δ_Q lattice (null/range/distribution constraints)
├── planner/        # Cost-optimal repair planning (~2,100 LoC)
│   ├── dp.py               # DP over DAG with 4-way recurrence
│   ├── lp.py               # LP relaxation + randomized rounding
│   ├── cost.py             # 9-parameter cost model
│   └── optimizer.py        # Multi-pass optimization
├── sql/            # SQL static analysis via sqlglot (~3,200 LoC)
├── types/          # Type system + base types (~4,500 LoC)
├── graph/          # Pipeline dependency DAG (~2,800 LoC)
├── quality/        # Statistical drift detection (~2,100 LoC)
├── execution/      # DuckDB-based repair executor (~1,800 LoC)
├── python_etl/     # Pandas/PySpark idiom matching (~1,400 LoC)
├── io/             # JSON/YAML pipeline spec I/O (~1,800 LoC)
└── cli/            # Click-based CLI (~400 LoC)
```

---

## Scores

### Code Quality: 5/10

**Evidence for:**
- Proper use of `attrs` throughout algebra module with well-defined `@attr.s` classes
- Clean separation of concerns across 12 subpackages with correct dependency direction
- Professional error handling: 30+ error classes with machine-readable error codes (types/errors.py)
- Real type hints throughout, compatible with mypy

**Evidence against:**
- `types/base.py` is a 2,907-line God file with 79 classes — a severe maintainability liability representing ~7% of the codebase in one file
- **Critical API mismatches within core modules**: `propagation.py:683` references `op.column_def.name` but `AddColumn` (schema_delta.py:308-314) has no `column_def` attr — it uses `name` directly. Additionally, `propagation.py:686` references `op.column_name` but `DropColumn` uses `name` — 2 of 3 branches in `_check_select_annihilation` reference nonexistent attributes (verified by independent verifier)
- 14 `pass` stubs in algebra/ and planner/ (push.py:459 RIGHT JOIN, interaction.py:369-371 constraint handling, push.py:1012-1169 window function edge cases)
- Zero polish rounds means no integration testing was ever completed

### Genuine Difficulty: 7/10

**What's genuinely hard and implemented:**
- **Three-sorted delta algebra** (composition.py:153-181): Real algebraic composition `(σ₁∘σ₂, δ₁∘φ(σ₁)(δ₂), γ₁⊔ψ(σ₁)(γ₂))` — not just naming classes after structures, the algebraic laws actually hold (verified by unit tests)
- **24 push operators** (push.py, 1,541 lines): 8 SQL operators × 3 delta sorts. `JoinPush.push_data()` (push.py:407-508) handles LEFT/INNER/FULL/CROSS/SEMI/ANTI with null-padding. `GroupByPush.push_data()` incrementally recomputes aggregates.
- **DP planner** (dp.py:130-229): Real bottom-up DP with 4-way recurrence (skip/annihilated, recompute, incremental, schema-migrate). Not a dressed-up greedy.
- **LP planner** (lp.py:128-374): Real LP formulation via scipy/HiGHS with randomized rounding + greedy feasibility patching + local search improvement
- **Annihilation detection** (annihilation.py): Novel — detects when deltas are absorbed by downstream operators, pruning unnecessary repairs. No analog in DBSP or dbt.
- **φ/ψ interaction homomorphisms** (interaction.py:280-437): Genuinely transforms data deltas per schema operations (add→extend tuples, drop→project, rename→rename fields, retype→coerce)

**What's stubbed or missing (~10% of claimed difficulty):**
- RIGHT JOIN push: `pass` (push.py:459)
- Constraint propagation in φ: `pass` for NOT_NULL and CHECK (interaction.py:369-371)
- Saga executor: only error types defined (errors.py:540-573), zero implementation
- Window function push: 5 silent `pass` in except blocks (push.py:1012-1169)
- `compose_parallel` (composition.py:380): uses list concatenation for schema ops, not algebraic parallel composition

### Value Delivered: 4/10

**The system does NOT work end-to-end.** The core propagation path crashes at `propagation.py:683` due to `op.column_def.name` on `AddColumn` which has no `column_def` attribute. This breaks:
- All 30 integration tests (test_repair_correctness.py, test_end_to_end.py, test_pipeline_scenarios.py)
- All 66 property test failures (AddColumn constructor mismatch)
- The bounded commutation theorem (`apply(repair(σ), state(G)) = recompute(evolve(G, σ))`) is defined in tests but has never been successfully executed

**What partially works:**
- Algebra operations work in isolation (1,754 unit tests pass)
- DP/LP planners work on hand-constructed inputs
- SQL lineage extraction via sqlglot handles SELECT, JOIN, GROUP BY, CTEs
- Quality monitoring does real statistical tests (KS, PSI, chi-squared, JSD)

**Root cause**: The type system (`types/base.py`) was refactored without updating consumers. At least 3 distinct API mismatch patterns exist:
1. `AddColumn(column=...)` vs `AddColumn(name=..., sql_type=...)` — 12+ sites
2. `op.column_def.name` in propagation.py — internal module mismatch
3. Graph method names (`children()` vs `successors()`, `is_acyclic()` vs property)

### Test Coverage: 6/10

| Suite | Pass | Fail | Skip | Notes |
|-------|------|------|------|-------|
| Unit (16 files) | 1,754 | 37 | 5 | Core algebra verified |
| Property (3 files) | 56 | 66 | 0 | Crash on AddColumn API |
| Integration (3 files) | 10 | 30 | 29 | Crash on propagation |
| **Total** | **1,820** | **133** | **34** | **91.5% overall** |

**Good**: Property tests (tests/property/) test real algebraic laws — associativity, identity, inverse, commutativity. Tests use Hypothesis strategies, not hardcoded examples. Design intent is excellent.
**Bad**: The property and integration tests have never been successfully executed against the current codebase. The 91.5% pass rate is misleading — it's 99.5% of *unit* tests passing and 0% of *integration* tests working.

### Format Support: 6/10

**Implemented:**
- JSON pipeline specs (`arc/io/json_format.py`): Full pipeline definition serialization with version support, schema validation, delta and repair plan round-tripping
- YAML pipeline specs (`arc/io/yaml_format.py`): Env-var interpolation (`${VAR:default}`), YAML anchors/aliases for DRY configs
- SQL dialect support via sqlglot: PostgreSQL and Spark SQL parsing with column-level lineage
- Python ETL analysis: pandas, PySpark, dbt model pattern matching

**Missing:**
- No dbt project file (`dbt_project.yml`) parser — the pandas analyzer pattern-matches method names but doesn't read dbt manifests
- No Airflow DAG parser
- I/O round-trip tests are broken (2 failures in test_io.py)
- Examples directory has only 2 files (simple_repair.py, complex_pipeline.py)

---

## Team Disagreements and Resolution

| Point | Auditor | Skeptic | Synthesizer | Resolved |
|-------|---------|---------|-------------|----------|
| Overall | 6.5 | 5.5 | ~7 | **5.6** |
| Code Quality | 7 | ~5 | "sound" | **5** — God file + double API mismatch in propagation |
| Difficulty | 8 | ~7 | "real math" | **7** — 10% stubbed in critical paths |
| Value | 5 | ~4 | "integration fix" | **4** — zero working e2e |
| Tests | 6 | ~5 | — | **6** — design is excellent, execution broken |
| Formats | 6 | — | — | **6** — capable but partially broken |

**Key disagreement**: Synthesizer claimed "6-8 hours to fix." Adversarial critique revised to 10-14 hours. Lead assessment: **12-16 hours** — the API mismatches span both test code and internal modules, plus stubs (RIGHT JOIN, constraints) need actual algorithm work, not just renaming.

---

## Genuine Difficulty Assessment (Systems Engineering Perspective)

**This is a genuinely difficult problem.** The three-sorted delta algebra with interaction homomorphisms is a novel formalization that has no direct precedent in DBSP, IVM, or schema evolution literature. The system requires:

1. **Multiple interacting subsystems**: SQL analysis → typed dependency graph → delta algebra → propagation → cost model → DP/LP planner → executor. These aren't independent — the algebra's push operators must correctly interact with the type system, and the planner must reason about annihilation.

2. **Non-trivial algorithms**: Bottom-up DP over DAG with 4-way recurrence; LP relaxation with randomized rounding; algebraic delta propagation with annihilation detection; incremental join maintenance across 7 join types.

3. **Sophisticated data structures**: Three-sorted algebraic structure (monoid × group × lattice) with cross-sort homomorphisms; typed multisets with bag semantics; refinement types carrying schema + quality predicates.

4. **Real architectural decisions**: Separation of static analysis (SQL/Python) from algebra engine from planner from executor; cost model integration at algebra level vs. planner level; conservative vs. precise handling of opaque transformations.

**Difficulty score as a research artifact: 7/10.** The algebra and planner are hard. The SQL analysis and quality monitoring are standard. The execution layer is simple (DuckDB wrapper). The ratio of genuinely novel to standard engineering is roughly 40/60.

---

## Risk Assessment

**Critical risks for CONTINUE:**
1. Zero working end-to-end flow — the system has never been integrated
2. Internal API drift suggests the modules were written in isolation without integration testing
3. 0 polish rounds means code quality is first-draft
4. Saga executor (a claimed feature) is completely unimplemented
5. The bounded commutation theorem (the paper's central claim) has never been empirically verified

**Mitigating factors:**
1. 1,820 passing unit tests prove individual modules work
2. API mismatches are mechanical (rename attributes), not architectural
3. The algebra's algebraic laws hold (verified by unit tests)
4. Architecture is sound and extensible

---

## VERDICT: CONTINUE

**Rationale**: The algebraic core is genuinely novel and substantially implemented. The 7/10 difficulty is real — this is not padding or generated boilerplate. The failures are integration-layer API mismatches from 0 polish rounds, not fundamental design flaws. A polish round focusing on:
1. Fixing `AddColumn.column_def` → `AddColumn.name` across propagation.py and tests
2. Fixing graph method name mismatches in DP planner
3. Filling RIGHT JOIN push and constraint propagation stubs
...would likely bring value delivered from 4 → 7 and overall from 5.5 → 7.

The algebra (composition with interaction homomorphisms, 24 push operators, annihilation detection) and planners (DP + LP) represent genuine systems engineering work that would be difficult to replicate. This is worth polishing.

| Dimension | Score |
|-----------|-------|
| Code Quality | **5/10** |
| Genuine Difficulty | **7/10** |
| Value Delivered | **4/10** |
| Test Coverage | **6/10** |
| Format Support | **6/10** |
| **VERDICT** | **CONTINUE** |

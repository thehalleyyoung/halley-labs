# README Truth Phase — Round 00 Report

**Stage:** Verification  
**Date:** 2026-03-04  
**Verdict:** ALL PASSED (after fixes)

---

## Team Composition

| Role | Agent | Task |
|------|-------|------|
| Independent Auditor | agent-0, agent-5 | Evidence-based testing of Quick Start, CLI, re-verification |
| Fail-Fast Skeptic | agent-1, agent-3 | Aggressive import/API testing, cross-challenge of Auditor |
| Scavenging Synthesizer | agent-2, agent-4 | Benchmark/utility testing, fix proposals |
| Team Lead | coordinator | Plan, synthesis, conflict resolution, final edits |

## Adversarial Process

### Round 1: Independent Proposals
- **Auditor** found: Quick Start crashes (`solver_strategy` string vs enum), CLI works (with minor cache issue)
- **Skeptic** found: 9/33 README imports broken, synthetic data API completely wrong, quickstart.py crashes
- **Synthesizer** found: Standard benchmarks PASS, stress suite 0/11 (API mismatch), utilities all PASS

### Round 2: Adversarial Critique
- **Skeptic challenged Auditor**: "solver_strategy is correctly typed as `SolverStrategy` enum — NOT a bug in the code"
- **Lead resolved**: Both are right. The *type annotation* says enum, but Python doesn't enforce at runtime. Passing `"auto"` (string) stores as-is, and `.value` crashes. Fix: use enum in README/callers.
- **All teammates agreed**: 9 import name mismatches are confirmed real failures.

### Round 3: Synthesis
Synthesizer proposed minimal fixes (README docs match code, not vice versa). All fixes approved.

### Round 4: Additional Issues Found During Re-verification
Auditor found 3 more bugs during re-verification:
1. `compare.py`: 4 more `solver_strategy="auto"` → `SolverStrategy.AUTO`
2. `compare.py`: `fs.score` → `fs.total_score`
3. `quickstart.py`: `fs.score` → `fs.total_score`, `report.estimation_result` → `report.baseline_estimate`, `er.estimate`/`er.std_error` → `er.ate`/`er.se`
4. `README.md`: `fs.score` → `fs.total_score`

---

## Issues Found and Fixed

### Critical Issues (3)

| # | Issue | File(s) | Fix |
|---|-------|---------|-----|
| 1 | `solver_strategy="auto"` crashes orchestrator (string lacks `.value`) | README.md, quickstart.py, stress.py, compare.py | Use `SolverStrategy.AUTO` enum |
| 2 | `generate_linear_gaussian` API completely wrong in README | README.md | Corrected params: `adj, n=, noise_scale=, edge_weight_range=, seed=` |
| 3 | `generate_linear_gaussian` called with wrong params in code | quickstart.py, stress.py | Updated to use correct API |

### Medium Issues (9 import mismatches)

| # | README (wrong) | Correct import |
|---|----------------|----------------|
| 4 | `from causalcert.dag.dsep import d_separated` | `DSeparationOracle` |
| 5 | `from causalcert.dag.moral import moralise` | `moral_graph` |
| 6 | `from causalcert.dag.mec import markov_equivalence_class` | `to_cpdag, enumerate_mec` |
| 7 | `from causalcert.ci_testing.ensemble import CauchyEnsemble` | `CauchyCombinationTest` |
| 8 | `from causalcert.solver.lp_relaxation import LPSolver` | `LPRelaxationSolver` |
| 9 | `from causalcert.solver.search import GreedySolver` | `UnifiedSolver` |
| 10 | `from causalcert.estimation.backdoor import BackdoorEstimator` | `satisfies_backdoor, enumerate_adjustment_sets` |
| 11 | `from causalcert.estimation.adjustment import find_adjustment_set` | `find_optimal_adjustment_set` |
| 12 | `from causalcert.reporting.latex_report import to_latex_report` | `to_latex_tables` |

### Additional Field Name Issues (3)

| # | Wrong | Correct | File(s) |
|---|-------|---------|---------|
| 13 | `fs.score` | `fs.total_score` | README.md, quickstart.py, compare.py |
| 14 | `report.estimation_result` | `report.baseline_estimate` | quickstart.py |
| 15 | `er.estimate`/`er.std_error` | `er.ate`/`er.se` | quickstart.py |

---

## Files Modified

| File | Changes |
|------|---------|
| `implementation/README.md` | 12 edits: imports, API examples, config table, field names |
| `implementation/examples/quickstart.py` | 4 edits: imports, generate_data, solver_strategy, field names |
| `implementation/causalcert/benchmarks/stress.py` | 3 edits: generate_high_dim_data, solver_strategy, import |
| `implementation/causalcert/benchmarks/compare.py` | 6 edits: solver_strategy (4×), fs.score, import |

---

## Final Verification Results

| Test | Description | Result |
|------|-------------|--------|
| 1 | `import causalcert` + version | ✅ PASS |
| 2 | Quick Start pipeline (README example) | ✅ PASS |
| 3 | Synthetic data generation (README example) | ✅ PASS |
| 4 | All 9 fixed imports | ✅ PASS |
| 5 | All 24 originally-passing imports | ✅ PASS |
| 6 | Standard benchmarks (`list_benchmarks`, `get_benchmark`) | ✅ PASS |
| 7 | Stress test suite (11/11 scenarios) | ✅ PASS |
| 8 | Cross-method comparison | ✅ PASS |
| 9 | `examples/quickstart.py` end-to-end | ✅ PASS |
| 10 | All utility functions (math, graph, stat) | ✅ PASS |
| 11 | CLI help commands | ✅ PASS |
| 12 | Full test suite (`pytest tests/`) | ✅ PASS (1364 passed, 2 skipped) |

---

## Team Signoff

- [x] **Independent Auditor**: All 12 tests pass. Signoff granted.
- [x] **Fail-Fast Skeptic**: All 9 import fixes verified correct. Cross-method and stress suite now work. Signoff granted.
- [x] **Scavenging Synthesizer**: All fixes follow minimal-change principle. No code logic altered, only callers/docs aligned. Signoff granted.

---

## ALL PASSED

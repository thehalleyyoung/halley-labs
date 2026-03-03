# Implementation Evaluation — Community Expert Review

**Project:** Causal-Plasticity Atlas (CPA)  
**Proposal:** proposal_00  
**Date:** 2026-03-02  
**Methodology:** Claude Code Agent Teams — 3 independent reviewers + adversarial cross-critique + independent verification signoff

---

## Evaluation Summary

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Code Quality** | **6/10** | Clean module-level architecture with comprehensive type hints and docstrings. However, 3 cross-module API mismatches (orchestrator calls `.search()` but QDSearchEngine only has `.run()`; CertificateGenerator and SensitivityAnalyzer receive wrong kwargs), 4 always-pass `assert X or True` test lines, and ~60 `except...: pass` blocks in edge/integration tests that silently mask failures. Modules were clearly developed independently without integration testing. |
| **Genuine Difficulty** | **7/10** | Substantive from-scratch implementations: PC algorithm with skeleton discovery + v-structure orientation + Meek rules R1-R3, PELT changepoint detection with dynamic programming and admissible set pruning, full MAP-Elites QD search with CVT tessellation, BCa bootstrap certificates with stability selection, and 4D JSD-based plasticity descriptors. These are non-trivial algorithms correctly implemented, though each individually is a known technique — the novelty lies in the composition. Docked for: linear-Gaussian SCM only, no latent variable handling, formal proofs from theory not verified. |
| **Value Delivered** | **5/10** | Phase 1 (Foundation: discovery + alignment + descriptors) works and produces real results. Phases 2 and 3 (QD exploration, certification/validation) silently fail due to API mismatches between orchestrator and component interfaces. The pipeline "runs" without crashing but produces empty/fallback results for 2 of 3 phases. No real-data examples exist. A practitioner cannot currently use this end-to-end. Individual components (mechanism distance, changepoint detection, statistical primitives) have standalone value. |
| **Test Coverage** | **5/10** | 1,513 test functions across 32 files; 1,452 unit tests pass 100%. Quantity is strong. Quality is undermined by: (1) 4 `assert X or True` lines that can never fail, (2) ~60 `except...: pass` blocks in edge/integration tests that accept both correct results and crashes as "passing", (3) integration tests that never catch the 3 API mismatches that break Phases 2-3, (4) 54 edge-case test failures showing brittleness under degenerate inputs. Unit tests for individual algorithms (CADA, JSD, changepoint, QD archive) are genuinely substantive. |

**Composite Score: 5.75/10**

---

## VERDICT: **CONTINUE**

---

## Detailed Findings

### What Actually Works

1. **Causal Discovery (from scratch):** Two independent PC implementations — one in `discovery/adapters.py` (170 lines, used by pipeline) and one in `discovery/structure_learning.py` (480 lines, standalone). Both implement partial correlation CI tests via Fisher Z, skeleton discovery, v-structure orientation, and Meek rules R1-R3. GES with BIC scoring also present. These are NOT wrappers around causal-learn (which is not installed in the default environment).

2. **CADA Alignment (ALG 1):** 6-phase algorithm in `alignment/cada.py` (1,630 lines). Uses Hungarian solver via `scipy.optimize.linear_sum_assignment`, CI fingerprinting via precision matrix inversion, Markov blanket Jaccard overlap. Genuinely novel for cross-context DAG alignment.

3. **Plasticity Descriptors (ALG 2):** 4D descriptor computation in `descriptors/plasticity.py` (1,513 lines). JSD-based mechanism distance with Bernoulli, Gaussian, and regression variants. Bootstrap confidence intervals, stability selection. The structural/parametric decomposition is the key theoretical insight and it's correctly implemented.

4. **PELT Changepoint Detection (ALG 4):** Hand-coded in `detection/changepoint.py` (1,167 lines). Proper O(1) cost evaluation via cumulative sums, admissible set pruning per Killick et al. (2012). 5 cost functions (L2, Gaussian likelihood, Poisson, RBF, custom). BH FDR correction for multiple testing. Permutation validation with Cohen's d effect sizes.

5. **Statistical Primitives:** `stats/distributions.py` + `stats/information_theory.py` (1,943 lines total). Clean JSD, KL, MI, CMI, partial correlation implementations. Zero external dependencies beyond numpy/scipy. Verified against analytical formulas.

### What's Broken

1. **Phase 2 — QD Exploration:** `orchestrator.py:1354` calls `qd_engine.search()` but `QDSearchEngine` only defines `.run()` (line 1099). AttributeError caught silently. Phase produces empty results.

2. **Phase 3 — Validation/Certification:** `orchestrator.py:1872-1877` passes `variable=`, `variable_index=`, `dataset=` to `CertificateGenerator.generate()`, but the actual signature expects `adjacencies`, `datasets`, `target_idx`, `dag_learner`. TypeError caught silently. Same for `SensitivityAnalyzer.analyze()`.

3. **`AlignmentResult.to_dict()`:** Line 213 calls `self.permutation.tolist()` without type guard. Crashes if permutation is a dict (possible from fallback paths).

4. **Test Integrity:** 4 `assert X or True` patterns (test_orchestrator.py:300, test_genome.py:285, test_subsystem_integration.py:150, test_numerical_stability.py:187) — these assertions can never fail. ~60 `except...: pass` blocks in edge/integration tests accept any behavior as "passing."

### What a Practitioner Would Think

**Positive:** The multi-context mechanism comparison framework addresses a genuine gap. No existing tool (pcalg, causal-learn, TETRAD) systematically classifies mechanisms along an invariant → parametric → structural → emergent spectrum. The 4D descriptor maps naturally onto the ICP/ICM literature gap. The QD-search application to causal mechanism space is creative and original.

**Negative:** A practitioner who `pip install`s this and runs the pipeline will get Phase 1 results only. Phases 2-3 silently fail. The 52+ hardcoded thresholds (τ_S = 0.1, gap_criterion = 0.15, stability bounds = 0.6/0.4, etc.) have no theoretical justification in the code and would require expert tuning. The "certificates" claim formal guarantees but depend on assumptions (linear Gaussian SCMs, correct DAG recovery) the code never validates. No real-data examples exist — all evaluation is on synthetic generators the authors control.

**Community verdict:** "Interesting framework, but I can't use it until the pipeline actually works end-to-end and someone shows it produces meaningful results on real data."

### Code Size Analysis

| Component | Total Lines | Actual Code (est.) | % Docstrings/Blanks |
|-----------|-------------|-------------------|---------------------|
| `cpa/` source | 43,558 | ~24,128 | ~45% |
| `tests/` | 15,651 | ~10,500 | ~33% |
| Total `.py` | ~63,500 | ~35,000 | ~45% |

The claimed "52,857 lines" (non-empty) is accurate but conflates documentation with implementation. Actual executable code is approximately 24K lines in the core library.

### Novelty Assessment

| Component | Novel? | Notes |
|-----------|--------|-------|
| Multi-context mechanism comparison | ✅ **Yes** | No existing tool does this systematically |
| CADA alignment | ✅ **Yes** | Novel algorithm for cross-context DAG alignment |
| 4D plasticity descriptor | ✅ **Yes** | Novel formulation; structural/parametric decomposition is insightful |
| QD search over mechanism space | ✅ **Yes** | Creative application of MAP-Elites to causal discovery |
| Tipping-point detection on mechanisms | ✅ **Yes** | Novel application of PELT to mechanism divergence sequences |
| PC/GES/PELT/Bootstrap individually | ❌ No | All standard algorithms from the literature |

---

## Conditions for Continuation

**Mandatory (before next review):**
1. Fix the 3 API mismatches in `orchestrator.py` so Phases 2-3 actually execute
2. Fix `AlignmentResult.to_dict()` type guard
3. Remove all 4 `assert X or True` patterns
4. Add at least 1 integration test that verifies non-empty Phase 2+3 output
5. Demonstrate one end-to-end run with all 3 phases producing real results

**Strongly recommended:**
6. Add one real-data example (e.g., Sachs protein signaling, multi-site clinical data)
7. Replace `except...: pass` blocks in edge tests with explicit expected-exception assertions
8. Document the sensitivity of results to the 52+ hardcoded thresholds

---

## Review Methodology

This evaluation was conducted using Claude Code Agent Teams with three independent reviewers:

- **Independent Auditor:** Ran test suite (1,452 unit pass, 54 edge-case fail), sampled 6 core files for algorithmic depth, scored 8/8/7/7
- **Fail-Fast Skeptic:** Found 4 always-pass asserts, ~60 except-pass blocks, 52+ magic numbers, analyzed line inflation, scored 7/7/5/4
- **Scavenging Synthesizer:** Compared theory to implementation (~70-75% coverage), assessed practitioner value, identified 6 extractable components, scored 6/7/4/7

**Adversarial cross-critique** resolved 4 key disputes:
- Discovery is from-scratch (Auditor correct, not Skeptic)
- Pipeline silently fails Phases 2-3 (Synthesizer correct, not Auditor)
- ~1,450 tests are genuinely meaningful, not ~900 (Skeptic overstated)
- API mismatches confirm integration-level rather than module-level problems

**Independent verifier** spot-checked 4 claims, confirmed all, and signed off on final scores.

Final scores represent consensus after adversarial resolution: **6 / 7 / 5 / 5 → CONTINUE**.

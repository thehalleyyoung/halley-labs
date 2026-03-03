# Implementation Evaluation — Community Expert (Verification Stage)

## Proposal: proposal_00 — CausalQD: Quality-Diversity Illumination for Causal Discovery

**Evaluator:** Causal-Discovery Community Expert (with Independent Auditor, Fail-Fast Skeptic, and Scavenging Synthesizer)
**Method:** Claude Code Agent Teams — 3 independent evaluations → adversarial cross-critique → synthesis → independent verification signoff
**Codebase:** 59,486 LOC, 23 subpackages, 28 test files, 1,197 tests (all passing), 3 polish rounds

---

## Executive Summary

This implementation wraps standard causal discovery algorithms (PC, GES, MMHC) and standard scoring functions (BIC, BDeu, BGe) inside a standard quality-diversity framework (MAP-Elites) with ~108 lines of textbook evolutionary search logic. The core value proposition — that QD-discovered structures outperform existing baselines, with Lipschitz-certified edge confidence — is doubly unsupported: no empirical benchmarks exist, and the certificate layer is self-admittedly mathematically vacuous. Of 59,486 lines, ~220 are genuinely novel (MI-profile descriptors + EM-for-missing-data BIC). The remaining ~59,000 lines competently duplicate existing libraries without meaningful improvement.

**No practitioner in the causal discovery community would switch from PC/GES to this without empirical evidence of superiority, which is entirely absent.**

---

## Scores

| Dimension | Score | Evidence |
|-----------|-------|----------|
| **Code Quality** | **7/10** | Production conventions genuinely applied: pervasive type hints (e.g., `npt.NDArray[np.int8]`), NumPy-style docstrings, custom exceptions (`DAGError`), dataclass validation. Consistent across all 23 subpackages. Deducted because quality is applied to mostly thin logic — clean wrappers on textbook algorithms don't warrant full credit. |
| **Genuine Difficulty** | **4/10** | ~220 genuinely novel lines: EM for missing-data BIC (~119 lines in `bic.py:475–593`), MI-profile behavioral descriptors (~25–30 novel lines in `info_theoretic.py:316–371`), L1/L2 regularized scoring (~90 lines). From-scratch PC/GES/MMHC baselines (~1,600 LOC) are real reimplementation effort but not invention. MAP-Elites core is ~108 lines of known algorithm. Lipschitz certificates are theatre — `spectral_bound()` is O(N) and self-documents as "can be vacuously large" (`lipschitz.py:165–168`). |
| **Value Delivered** | **4/10** | Pipeline runs end-to-end (confirmed by integration tests). EM-for-missing-data and L1/L2-regularized BIC genuinely exceed causal-learn's implementations. But: zero benchmark results against ALARM/ASIA/SACHS, no evidence QD outperforms random restarts of GES, certificates mathematically vacuous, all hard causal problems dodged (no latent confounders, no faithfulness handling, no non-Gaussianity, maxes at ~100 nodes). |
| **Test Coverage** | **6/10** | 1,197 tests pass in 72s. Property-based tests with Hypothesis verify genuine DAG invariants (mutation preserves acyclicity, d-separation symmetry). Integration tests run full pipeline. But: 28.8% trivial assertions (type/shape checks), theorem tests verify `>= 0` not mathematical tightness, certificate tests check non-negativity rather than bound quality. No performance regression tests. |

**Weighted Average: 5.0/10**

---

## Detailed Findings

### What's Genuinely Good

1. **Scoring functions** (`scores/bic.py`, `bdeu.py`, `bge.py`, ~2,000 LOC): Mathematically correct implementations with features absent from causal-learn: BIC with L1/L2 regularization (coordinate descent with soft-thresholding), EM for missing data (conditional Gaussian imputation with pseudoinverse fallback), Numba-JIT fast path. These are the most valuable components.

2. **From-scratch baselines** (`baselines/pc.py` 657 LOC, `ges_baseline.py` 438 LOC, `mmhc.py` 502 LOC): Zero external causal-discovery dependencies. PC implements stable-PC with Meek R1–R4. GES implements forward/backward/turning in CPDAG space. MMHC implements MMPC-restrict + hill-climbing. Legitimate reimplementations.

3. **MI-profile behavioral descriptors** (`descriptors/info_theoretic.py:316–371`): The sole genuinely novel idea — using parent-conditioned mutual information as QD behavioral descriptors. Narrow (~25–30 lines of novel logic) but unique; no existing QD library computes these.

### What's Hollow

1. **Lipschitz certificates** (`certificates/lipschitz.py`): The spectral bound `L = N·λ_max / (λ_min²·p)` grows linearly with sample size (line 194). The docstring explicitly warns "can be vacuously large" (line 165). The empirical bound is finite-differencing with no coverage guarantee. The `EdgeCertificate.value` combines bootstrap frequency with a sigmoid of score delta — an arbitrary heuristic with no theoretical backing. This entire layer dresses up standard bootstrap in formal language without adding rigor.

2. **LOC inflation**: 42% of code is non-algorithmic scaffolding (tests, visualization, CLI, utils, config, parallel, benchmarks, analysis). Within source files, only 47.7% of lines are actual code (rest: docstrings, comments, blanks, imports). Effective algorithmic code is ~12,000–15,000 lines, of which ~220 are novel.

3. **Missing hard parts**: Dead enum values for PAG/bidirected edges (`types.py:90`) — never used. Kernel CI test exists but isn't integrated into the pipeline. NonlinearSCM generates faithfulness violations for benchmarks, but no algorithm handles them. Config's "large" preset tops out at 100 nodes; real benchmarks use 1,000+.

4. **No empirical validation**: The central claim — that QD illumination produces better/more diverse causal structures than existing methods — is never tested. No results on any standard benchmark dataset.

### Community Reception Assessment

As a causal discovery researcher, I would react to this with a shrug. The idea of "run MAP-Elites over DAGs" is straightforward enough to describe in a paragraph. The community has seen QD applied to many combinatorial problems. The hard questions this needs to answer:

- **Why not just run GES with 1,000 random restarts?** The QD archive provides diversity, but diversity of *what*? The behavioral descriptors are graph statistics (edge density, v-structure count, MI profiles) — not causally meaningful features like Markov equivalence class coverage or interventional distinguishability.
- **Where are the theorems?** Theory evaluation found all 9 theorems trivial/circular/erroneous (theory score: 3.0/10). The Lipschitz certificates were the theoretical anchor; without them, there's no theoretical contribution.
- **Where are the experiments?** A method paper without benchmark results is unpublishable at any venue above workshop level.

This would not be accepted at UAI, AISTATS, or NeurIPS in its current form. It might be a workshop poster if empirical results on standard benchmarks were added showing QD outperforms multi-restart GES — but that result may not exist.

---

## Salvage Recommendation

Extract ~2,200 lines as a standalone `causal-scoring-utils` package:
- BIC/BDeu/BGe scoring functions with EM + regularization (~2,000 LOC)
- MI-profile behavioral descriptor (~150 LOC as standalone)

These fill genuine gaps in the Python causal discovery ecosystem. Everything else duplicates existing libraries (pyribs, causal-learn, pgmpy) without improvement.

---

## VERDICT: **ABANDON**

The implementation is competent engineering (7/10 code quality, 1,197 passing tests) applied to a theoretically weak foundation (3.0/10 theory score). The core value proposition is unsupported: no evidence QD outperforms baselines, and the certificate layer that should differentiate it is mathematically vacuous. The genuine novelty fits in ~220 lines. The 59,486-line codebase is an exercise in software engineering, not a research contribution.

**Composite score: 5.0/10** — below the threshold for continuation. Salvage the scoring functions and MI-profile descriptors; abandon the rest.

---

*Evaluation produced by team of 3 specialist agents (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) with adversarial cross-critique and independent verification signoff. All claims cite specific file locations verified against source code.*

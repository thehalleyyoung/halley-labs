# Implementation Evaluation — Senior Systems Engineer

## Methodology

Evaluated using Claude Code Agent Teams with three independent experts:
- **Independent Auditor**: Evidence-based scoring with file-level citations
- **Fail-Fast Skeptic**: Adversarial rejection of under-supported claims
- **Scavenging Synthesizer**: Salvage assessment of reusable components

Each expert independently assessed the implementation, then underwent cross-challenge synthesis with disagreement resolution. Final scores reflect weighted consensus after adversarial debate.

---

## proposal_00 — CausalQD

**Lines of code**: 55,818 source + 16,087 tests = ~72K total (140 Python files, 396 classes, 1,985 functions)
**Tests**: 1,197/1,197 passing (96.5s), including 21 integration tests
**Polish rounds**: 3

### Code Quality: 6/10

**Strengths** (all three experts agree):
- Clean architecture with 20+ subpackages, proper separation of concerns
- Consistent type annotations throughout (AdjacencyMatrix, DataMatrix, BehavioralDescriptor)
- BGe score (714 LOC): Full Normal-Wishart marginal likelihood with slogdet, multivariate gamma, proper regularization — genuinely hard to implement correctly
- BDeu score (674 LOC): Mixed-radix parent-config encoding, vectorized bincount, lgamma formula
- Bayes-Ball d-separation (dag.py:361-422): Direction-aware BFS correctly handling chains/forks/colliders
- MEC enumerator (678 LOC): DFS with Meek R1-R3 constraint propagation, Union-Find chain decomposition

**Weaknesses** (consensus):
- `_topological_sort` copy-pasted verbatim in 5 files (mutation.py, crossover.py, repair.py, advanced_decomposer.py, advanced.py)
- Excessive aliasing in DAG class: from_edges/from_edge_list, num_nodes/n_nodes, moralize/moral_graph
- ~49% of source is executable code, 28% docstrings — documentation inflation
- Engine uses only ~40% of implemented modules — 12-15 subsystems are orphaned (streaming, parallel/distributed, analysis, most of adaptive)
- Three "fast_*" files (fast_dag.py, fast_bic.py, fast_descriptors.py) duplicate algorithmic content with Numba JIT (~3,284 lines of duplication)

### Genuine Difficulty: 5/10

**Above "weekend project" bar** (Auditor + Synthesizer agree):
- BGe score with Normal-Wishart conjugate priors, incremental updates, Sherman-Morrison optimization
- BDeu with auto-discretization and proper prior hyperparameter handling
- MEC enumeration with Meek rules R1-R4 iterated to fixed point
- Order MCMC with correct Metropolis-Hastings ratios, Gelman-Rubin diagnostics, ESS
- CMA-ES adaptation to discrete DAG space (emitters.py)

**At or below "weekend project" bar** (Skeptic confirmed, Auditor partially agrees):
- MAP-Elites loop is 20 lines of standard QD algorithm (map_elites.py:524-544), surrounded by 807 lines of engineering
- DAG class: standard adjacency matrix + Kahn's algorithm
- Mutation operators: topological ordering + edge add/remove/reverse — standard evolutionary computation
- All behavioral descriptors: partial correlation, edge density, clustering coefficient — textbook statistics
- Bootstrap certificates: standard resampling
- Lipschitz spectral bound: just σ_max(Σ) — one line of linear algebra

**Novel algorithmic content** (Skeptic's estimate, partially validated):
- ~200-300 lines of genuinely novel code: adaptive operator selection heuristics, Markov blanket transplantation crossover, v-structure-aware skeleton merge
- This is 0.3-0.5% of the total codebase
- The contribution is systems integration ("apply MAP-Elites to DAGs with domain-aware operators"), not algorithmic innovation

### Value Delivered: 4/10

**What works** (consensus):
- System runs end-to-end: data generation → scoring → MAP-Elites search → archive → certificates
- Order MCMC correctly recovers true DAG structure on synthetic chain graphs
- BIC/BDeu/BGe scores compute correctly on synthetic Gaussian data
- Integration tests verify full pipeline

**What doesn't deliver** (consensus):
- **Certificates are vacuous**: Lipschitz spectral bound grows linearly with N (the code's OWN docstring at lipschitz.py:166 admits this). For N=1000, p=10: bound ≈ 200-5,000. Stability radius is trivially small. This confirms theory evaluation finding.
- **Path certificate has mathematical error**: path_certificate.py:115-127 composes path Lipschitz as PRODUCT of per-edge constants (should be additive for decomposable scores). For path length k with per-edge L≈100, produces 100^k — vacuous for any path >2.
- **MEC size heuristic bug**: mec_computer.py:207 claims k acyclic orientations for k nodes; correct answer is 2^(k-1). Off by exponential factor.
- **No empirical evidence of value**: Zero benchmarks showing MAP-Elites outperforms GES, PC, or Order MCMC on any metric (accuracy, diversity, or otherwise)
- **"Interventional effects" descriptor** (advanced.py:224) implements simple linear regression, NOT Pearl's do-calculus — misleading naming
- **Island-model parallelism**: Islands evolve sequentially in a loop (parallel/distributed.py:436-437), not actually parallel

### Test Coverage: 6/10

**Strengths**:
- 1,197 tests, all passing — significant testing infrastructure
- Property-based testing with Hypothesis (test_properties.py): random DAG strategy, mutation acyclicity preservation, d-separation symmetry — genuinely good practice
- Score decomposition tests validate algebraic correctness across chain/fork/collider structures
- Edge case tests (599 lines) cover empty graphs, single-node, disconnected components

**Weaknesses**:
- ~60% of theorem tests verify tautological properties (e.g., "coverage never decreases" — adding to a set never shrinks it; "best quality never decreases" — the archive only replaces on improvement)
- Certificate tests only check positivity and finiteness (`assert cert.lipschitz_bound >= 0.0`) — this passes for ANY positive number including 10^100
- No test validates certificates are non-vacuous or tighter than trivial bounds
- No test compares MAP-Elites output quality to baselines
- Test sizes tiny (n=5 nodes, 100-500 samples, 5-10 iterations) — may miss bugs at scale
- No statistical power tests for CI tests (Fisher Z, partial correlation)

---

## Teammate Disagreement Resolution

| Topic | Auditor | Skeptic | Synthesizer | Resolution |
|-------|---------|---------|-------------|------------|
| Code quality | 6.5 | ~2-3 | "excellent engineering" | **6.0** — Auditor correct on component quality; Skeptic correct on padding ratio |
| Novelty | "modest" | "210 novel lines" | "not novel" | **~250 lines novel** — Skeptic's count is aggressive but directionally correct |
| Difficulty | 6.0 | 1.1 | "genuinely difficult" components | **5.0** — Components hard, system design standard, theory claims wrong |
| Value | 5.0 | "reject" | "toolkit not paper" | **4.0** — No evidence of utility beyond component reuse |
| Salvageability | N/A | N/A | core+scores+operators | **Agreed** — these three modules (~12K LOC) are genuinely valuable |

**Key consensus**: All three agree the system is a competent engineering implementation of known algorithms with no novel theoretical contribution and vacuous certificate claims. Disagreement is on magnitude, not direction.

---

## Scoring Summary

| Criterion | Score |
|-----------|-------|
| Code Quality | **6/10** |
| Genuine Difficulty | **5/10** |
| Value Delivered | **4/10** |
| Test Coverage | **6/10** |
| **Composite** | **5.25/10** |

## VERDICT: **ABANDON**

### Rationale

Despite competent engineering of individual components (BGe/BDeu scoring, MEC enumeration, DAG operators), this implementation fails on the criteria that matter for a research artifact:

1. **No novel algorithm**: The ~250 lines of genuinely novel code (0.4% of total) constitute a systems integration contribution, not a publishable algorithmic advance. MAP-Elites applied to DAGs is a configuration, not an innovation.

2. **Theoretical claims are false or vacuous**: Path Lipschitz composition has a mathematical error. Spectral bounds are vacuous by the code's own documentation. "Certificates" provide no actionable guarantees beyond standard bootstrap.

3. **No evidence of value**: Zero benchmarks demonstrate that MAP-Elites produces better or more diverse causal structures than standard methods (GES, PC, Order MCMC). The fundamental scientific question — "does QD illumination help causal discovery?" — is not answered.

4. **60% of code is unused**: The engine orchestrates only ~40% of implemented modules. Streaming, distributed parallel, most analysis, and advanced adaptive control are orphaned infrastructure that was never integrated.

5. **Theory score was 3.0/10**: The implementation confirms rather than remedies the theoretical weaknesses. The 9/9 trivial/erroneous theorems remain unaddressed in code.

### Salvageable Elements

If a future proposal needs causal discovery building blocks:
- `core/` (DAG library with multiple backends) — ★★★★★
- `scores/` (BIC/BDeu/BGe with incremental updates) — ★★★★½
- `operators/` (evolutionary DAG operators with acyclicity guarantees) — ★★★★
- `sampling/` (Order MCMC + Parallel Tempering) — ★★★★

These ~14K lines could form a `causal-evo-toolkit` package of genuine standalone value.

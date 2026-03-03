# Implementation Evaluation — Rigorous Skeptic Review

**Project**: CausalQD — Quality-Diversity Illumination for Causal Discovery  
**Proposal**: proposal_00  
**Date**: 2026-03-02  
**Methodology**: 3-agent adversarial team (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) with adversarial critique resolution round and lead verifier signoff.

---

## Scores

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Code Quality** | **7/10** | 55,818 LOC across 23 subpackages. Clean abstractions, type-hinted, dataclass configs. Numerical stability is excellent (log-sum-exp, slogdet, Geyer ESS). Deductions: mislabeled "Bonferroni" in path certificates, arbitrary 0.6 composite weight default, sklearn listed as dependency but unused, some code duplication between engine and archive implementations. |
| **Genuine Difficulty** | **6/10** | Correctly implements BGe (Normal-Wishart integration), Chickering's CPDAG algorithm with Meek R1-R4, and Order/Partition MCMC with parallel tempering — all from scratch using only numpy. These are genuinely hard to implement correctly. However, every algorithmic component exists in the literature; the combination is the claimed novelty. The MAP-Elites loop itself is straightforward. The operators are competent domain adaptations, not algorithmic breakthroughs. Scalability untested beyond 15 nodes. |
| **Value Delivered** | **5/10** | The hard problem — "does QD illumination actually deliver value over posterior sampling or random restarts?" — is **not answered**. No ablation study compares MAP-Elites diversity against Order MCMC posterior sampling or multi-start GES with matched compute budgets. The central theoretical guarantee (Theorem 3: coverage under ergodicity) rests on an unproven ergodicity assumption. The certificate framework is competent but has a mislabeled path CI and no multiple testing correction. The most valuable components are the supporting libraries (CI tests, scoring functions, MCMC diagnostics) — not the QD engine itself. |
| **Test Coverage** | **8/10** | 1,197 tests, all passing, across 26 test modules (16,087 LOC). Exemplary: property-based testing with Hypothesis (acyclicity preservation over random DAGs), theorem-level invariant validation (9 theorems mapped to tests), numerical precision checks (1e-10 tolerances), known-answer MEC tests. Missing: adversarial inputs, asymptotic correctness, scalability beyond 15 nodes, ablation tests for QD value. |

---

## Detailed Findings

### What Works (Verified by Auditor + Synthesizer, confirmed by Lead Verifier)

1. **Scoring functions are best-in-class**: BIC (QR + Sherman-Morrison + Numba JIT), BDeu (Dirichlet-equivalent), BGe (Normal-Wishart) — all implemented from paper formulations using only numpy. The Interventional BIC scorer has **no Python equivalent** in causal-learn, pcalg, or TETRAD. Fast BIC claims 10-50x over naive via sufficient statistics caching + Numba JIT.

2. **MAP-Elites archive is correctly implemented**: The Skeptic's claim that the archive is "broken" and "equivalent to random restart" was **refuted** during adversarial resolution. The grid-cell structure provides implicit diversity (this IS how standard MAP-Elites works — Mouret & Clune 2015). Additionally, the code includes `sample_curiosity()` for under-explored-cell selection, directly contradicting the Skeptic's assertion that no diversity mechanism exists.

3. **CI test suite is comprehensive**: Fisher-Z, partial correlation (3 methods + Ledoit-Wolf shrinkage), kernel HSIC (permutation + Gamma-approximation), and KSG k-NN CMI (both KSG1 and KSG2 variants). Unified API with abstract base class. Multiple testing corrections (Bonferroni + Benjamini-Hochberg). Outperforms causal-learn's coverage.

4. **MCMC diagnostics are standalone-worthy**: Gelman-Rubin R̂, ESS (Geyer initial positive sequence), Geweke diagnostic, convergence detection (Mann-Kendall + Fisher combination). Zero-dependency alternative to ArviZ.

5. **Test infrastructure is exceptional**: Theorem-to-test mapping, property-based testing with Hypothesis, numerical precision tests. 28.8% test-to-source ratio is unusually high for research code.

### What Fails (Verified by Skeptic + Lead Verifier)

1. **No demonstrated QD value-add**: The fundamental question — "does MAP-Elites illumination discover causal structures that posterior sampling misses?" — is unanswered. No ablation study, no comparison with Order MCMC posterior diversity, no empirical evidence that archive diversity translates to actionable causal insights. The theoretical coverage guarantee (Theorem 3) depends on an unproven ergodicity assumption the paper acknowledges.

2. **Certificate framework has real issues**: 
   - Path CI docstring claims "Bonferroni-corrected" but implementation just multiplies per-edge CI bounds (verified at `path_certificate.py:159`). This is misleading.
   - No family-wise multiple testing correction when certifying many edges simultaneously.
   - Composite certificate value uses arbitrary 0.6 default weight (configurable but ungrounded).
   - However: Wilson CI for bootstrap frequency IS correct and appropriate. Bootstrap percentile CI for score deltas is standard. The framework is competent engineering, not statistically invalid.

3. **Scalability is unproven**: Maximum tested graph size is 15 nodes. Scalability mechanisms exist (skeleton restriction, PCA compression, sampling CI, parallel evaluation) but are untested at interesting problem sizes (gene regulatory networks: 20-100 nodes). The "thousands of graph features" claim in the problem statement is aspirational.

4. **Operators are domain adaptations, not innovations**: VStructureMutation (targeting identifiability motifs) and BlockMutation (topological block swaps) show good domain engineering, but every individual operator has prior art. The collection is competent, not novel. Claiming "novel acyclicity-preserving variation operators" is an overclaim.

### Adversarial Critique Outcomes

| Claim | Skeptic Said | Resolution |
|-------|-------------|------------|
| Archive is broken | FATAL: quality-only acceptance = random restart | **REFUTED**: Grid cells provide implicit diversity. This is textbook MAP-Elites. `sample_curiosity()` also exists. |
| Operators are textbook | FATAL: zero novelty | **PARTIALLY UPHELD**: Not zero novelty — VStructureMutation and BlockMutation are good domain adaptations. But not research-level novelty either. |
| Certificates are invalid | FATAL: made-up composites, wrong CIs | **PARTIALLY UPHELD**: Wilson CI is correct. Composite weight is a hyperparameter (not great, not invalid). Path CI IS mislabeled. No MTC IS a real gap. Overall: imperfect but not invalid. |
| Prior art kills it | SERIOUS: just a QD wrapper | **UPHELD**: Core algorithms are all from literature. The QD combination is novel in framing but undemonstrated in value. |
| Scalability fails | SERIOUS: infeasible at 20+ | **UPHELD**: Untested beyond 15 nodes. Mechanisms exist but unvalidated. |

### Salvageable Value (Synthesizer Assessment)

Even if the QD-for-causal-discovery angle is abandoned, these components represent genuine contributions to the Python causal discovery ecosystem:

| Component | Lines | Unique Value |
|-----------|-------|-------------|
| `scores/` (esp. interventional BIC, fast BIC) | ~3,400 | No Python equivalent for interventional scoring |
| `ci_tests/` (Fisher-Z, kernel HSIC, KSG CMI) | ~1,800 | Most complete CI test suite in a single Python package |
| `sampling/convergence.py` (MCMC diagnostics) | ~556 | Lightweight ArviZ alternative, zero-dependency |
| `certificates/` (bootstrap + Lipschitz) | ~1,900 | Lipschitz robustness bounds have no equivalent |
| `mec/` (CPDAG, enumeration, hashing) | ~1,900 | Most complete Python MEC toolkit |

---

## Best-Paper Assessment

**Venue**: UAI or AISTATS (not ICML/NeurIPS — too niche).

**Strongest pitch**: "We show that quality-diversity illumination systematically maps the space of plausible causal DAGs with per-edge robustness certificates, revealing structural uncertainty that single-best methods miss."

**Fatal weakness for acceptance**: No empirical demonstration that QD diversity adds value over posterior sampling. A reviewer will ask "why not just run Order MCMC and analyze the posterior?" — and the paper has no answer. The ergodicity gap undermines the central theorem. The scalability limitation (n≤15 tested) precludes interesting applications.

**Best-paper potential**: **Low (2/10)**. The combination is interesting but the contribution is not crisp enough for a top venue. The supporting libraries are impressive engineering but not a research contribution.

---

## VERDICT: CONTINUE

**Rationale**: Despite the skeptic's aggressive pushback, the adversarial resolution reveals that the implementation is substantially more solid than the initial "REJECT" verdict suggested. The archive IS correctly implemented (the skeptic's most damaging claim was wrong). The certificate framework has real but moderate issues (mislabeled docstring, missing MTC) — not the wholesale statistical invalidity claimed. The code quality and test infrastructure are genuinely excellent.

**The project delivers a working, well-tested, 55K-LOC causal discovery system with several components that outperform existing Python tools.** While the central QD value proposition is undemonstrated and the novelty claims need tempering, the artifact has enough substance and quality to warrant continuation. The supporting libraries alone (scoring functions, CI tests, MCMC diagnostics, MEC toolkit) fill real gaps in the ecosystem.

**Conditions for continuation**:
1. Temper novelty claims — operators are "domain adaptations," not "novel algorithms."
2. Fix the "Bonferroni" mislabel in path certificates.
3. Add an ablation study: MAP-Elites archive diversity vs. Order MCMC posterior vs. random restart GES, matched compute budget.
4. Test scalability to 20+ nodes and report wall-clock times.
5. Add family-wise multiple testing correction to the certificate framework (at minimum Holm-Bonferroni).

**Risk if not addressed**: Without the ablation study, the project cannot demonstrate that its central contribution (QD illumination) provides value. The supporting libraries are excellent, but "we built good utility code" is not a paper.

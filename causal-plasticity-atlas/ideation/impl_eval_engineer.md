# Implementation Evaluation — Senior Systems Engineer

## Methodology

Three-agent adversarial evaluation with independent proposals, cross-critique,
and lead-resolved synthesis.

| Role | Agent | Verdict |
|------|-------|---------|
| Independent Auditor | Evidence-based scoring | CONTINUE (conditional) |
| Fail-Fast Skeptic | Aggressive flaw detection | ABANDON |
| Scavenging Synthesizer | Salvage assessment | REBUILD |

---

## proposal\_00 — Causal-Plasticity Atlas (CPA)

**Claimed:** 52,857 LOC · 5 novel algorithms · 33 test files
**Actual:** 43,558 source LOC + 15,651 test LOC · ~12–15K genuine algorithmic code · 0 polish rounds · Timed out

### Scores

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Code Quality** | **5/10** | Clean architecture and type annotations offset by pervasive copy-paste duplication (7 function families duplicated across modules). The JSD copy-paste bug is a direct consequence of this anti-pattern. ~35% of codebase is scaffolding/orchestration of limited value. |
| **Genuine Difficulty** | **5/10** | Two modules show real algorithmic substance: `core/mechanism_distance.py` (moment-matched Gaussian JSD, multivariate extensions, Monte Carlo conditional JSD) and `alignment/cada.py` (6-phase DAG alignment with CI-fingerprinting, Markov blanket Jaccard filtering, Hungarian matching). The rest is standard statistics (PELT changepoint), incomplete (QD search with fake evaluator), or trivial (emergence = `1 - min(|MB|)/(max(|MB|)+1)`). Linear-Gaussian assumption throughout simplifies all core computations to closed form. |
| **Value Delivered** | **3/10** | Zero real-data validation. Zero baseline comparisons (no CD-NOD, ICP, IGCI, LPCMCI). QD-MAP-Elites evaluator is a random number generator (`_default_evaluator` uses `rng.normal()` noise, never connected to actual causal analysis). Integration tests are broken (25+ failures in end-to-end pipeline). Cannot currently produce meaningful outputs. |
| **Test Coverage** | **4/10** | 1,513 test functions with 2,045 total assertions (1.35 asserts/test). 60 try/except/pass blocks across 4 test files that literally cannot fail. Certificate tests use conditional assertions that pass vacuously. Unit tests in core modules are genuine, but integration and edge-case test suites are largely theater. |

### Fatal Flaws

**1. JSD Computation Is Mathematically Wrong (Pervasive)**
The `_jsd_gaussian()` function — duplicated in `descriptors/plasticity.py`, `detection/tipping_points.py`, `certificates/robustness.py`, and `descriptors/confidence.py` — computes `0.5*(KL(p||q) + KL(q||p))` (Jeffreys divergence), NOT Jensen-Shannon divergence. JSD is bounded by ln(2) ≈ 0.693; Jeffreys divergence is unbounded. The docstring claims "JSD ≈ 0.5 * KL_sym" — this is incorrect. The correct implementation exists in `core/mechanism_distance.py:346-400` and `stats/distributions.py`, but was degraded during copy-paste. This corrupts:
- ψ_P (parametric plasticity) — inflated, unbounded
- Bootstrap UCB in certificates — thresholds calibrated against wrong metric
- Tipping-point divergence sequences — PELT operates on wrong cost surface
- All plasticity classifications

**2. QD-MAP-Elites Evaluator Is a Placeholder**
`qd_search.py:76-139`: The `_default_evaluator` generates classification probabilities with `rng.normal()` seeded by genome hash. No real evaluator connecting genome → causal discovery → alignment → classification is implemented anywhere. The QD search optimizes over a synthetic noise landscape. ALG3 is structurally a stub.

**3. Two Descriptor Dimensions Are Theoretically Vacuous**
- **Emergence (ψ_E)** = `1 - min(|MB|)/(max(|MB|) + 1)` — an ad-hoc min/max ratio with arbitrary +1 divisor. No information-theoretic justification. Not a measure of causal emergence in any recognized sense.
- **Context Sensitivity (ψ_CS)** = coefficient of variation of ψ_P over random context subsets. Circular: measures sampling variation of a variation measure. Returns 0 for K<3.

**4. Zero External Validation**
No benchmark datasets (Sachs, ALARM, Asia, Dream4). No comparisons to CD-NOD, ICP, or any established multi-context method. No evidence this system produces correct or useful outputs on any non-synthetic data.

### Serious Concerns

- **LOC inflation**: 52,857 claimed → ~12–15K genuine algorithmic code after removing boilerplate, docstrings, visualization, IO, diagnostics, and __init__.py files
- **Code duplication**: `_jsd_gaussian` (4 copies), `_fit_ols` (3 copies), `_coefficient_se` (2 copies) — directly caused the primary mathematical bug
- **Certificates overstate guarantees**: The "robustness certificates" are threshold comparisons (UCB < 0.01 → STRONG_INVARIANCE) backed by bootstrap, not formal PAC-style bounds. The word "certificate" implies formal guarantees that don't exist.
- **60 vacuous tests**: try/except/pass patterns across boundary, numerical stability, and integration tests mean these tests pass regardless of function behavior
- **0 polish rounds**: Implementation timed out; no refinement was applied

### Salvageable Components

| Component | Quality | Worth Keeping |
|-----------|---------|---------------|
| `cpa/stats/` (distributions, information_theory) | 8.5/10 | Yes — production-quality statistical utilities |
| `cpa/utils/` (caching, validation, parallel, logging) | 9/10 | Yes — excellent infrastructure code |
| `cpa/core/mechanism_distance.py` | 7.5/10 | Yes — correct JSD, multivariate extensions |
| `cpa/alignment/cada.py` + scoring + hungarian | 7/10 | Yes — genuinely useful DAG alignment |
| `cpa/descriptors/plasticity.py` (ψ_S, ψ_P only) | 6/10 | Yes — novel concept, needs JSD fix |
| `cpa/detection/changepoint.py` | 7/10 | Maybe — correct PELT, but not novel |

### Novel Ideas Worth Preserving

1. **4D Plasticity Descriptor decomposition** — The conceptual framework of ⟨structural, parametric, emergence, sensitivity⟩ fills a genuine gap. No existing tool provides this decomposition for multi-context causal analysis. The first two dimensions (ψ_S, ψ_P) are well-grounded; the latter two need theoretical work.
2. **Context-Aware DAG Alignment via CI-Fingerprinting** — Combining Markov blanket overlap with conditional independence fingerprints for cross-context variable matching is principled and addresses a gap CD-NOD doesn't fill.

### Disagreement Resolution

The team disagreed sharply on the verdict:

- **Auditor** said CONTINUE: argued integration failures are "interface-level" and fixable. However, the Auditor completely missed the JSD mathematical bug — the most critical finding — undermining confidence in their thoroughness.
- **Skeptic** said ABANDON: correctly identified all fatal flaws but overstated severity of some fixable issues (JSD is ~30 lines to fix) and dismissed genuinely solid components.
- **Synthesizer** said REBUILD: correctly identified that ~8-10K LOC of solid algorithmic work exists inside ~43K LOC of mixed-quality code. Recommended focused rebuild around CADA + Descriptors.

**Resolution**: The Synthesizer's assessment is most evidence-aligned. The Auditor was negligent. The Skeptic was too aggressive on ABANDON given salvageable value. REBUILD is correct — but for this pipeline's purposes, the distinction between REBUILD and ABANDON is immaterial.

### VERDICT: **ABANDON**

**Reasoning from a 100K+ LoC systems perspective:**

This implementation has the shape of a serious system but not the substance. The architecture diagram looks right — 15 packages, 5 algorithms, clean layering. But:

1. **The mathematics are wrong.** The central divergence measure is incorrect in 4 of 6 downstream modules. Everything downstream of JSD (descriptors, certificates, tipping points) produces wrong numbers. This is not a typo — it's a copy-paste degradation that went undetected because the test suite can't catch it.

2. **A major algorithm is a stub.** QD-MAP-Elites (the headline contribution distinguishing CPA from "just another causal discovery tool") has never been connected to real computation. The evaluator generates random numbers.

3. **No evidence of value.** Zero real-data results. Zero baseline comparisons. You cannot evaluate whether a causal discovery system works without running it on data where ground truth is known. This is the most basic requirement.

4. **The difficulty is moderate, not extreme.** Strip away the scaffolding and you have: (a) a correct JSD implementation for Gaussians, (b) a bipartite matching algorithm with domain-specific scoring, (c) a PELT wrapper, and (d) some bootstrap statistics. These are individually 200–500 line algorithms. The integration complexity is real but the implementation failed at exactly that level — the integration tests are broken.

5. **Best-paper potential: absent.** No surprising theoretical results. No empirical validation. No demonstration that this approach discovers something existing tools cannot. The 4D descriptor concept is interesting but unvalidated.

For this pipeline, ABANDON is the correct call. The salvageable components (stats/, utils/, mechanism_distance, CADA skeleton) could inform a future proposal, but this implementation cannot be polished into a best-paper-quality artifact within any reasonable budget.

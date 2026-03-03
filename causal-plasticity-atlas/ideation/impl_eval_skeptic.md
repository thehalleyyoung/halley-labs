# Implementation Evaluation — Skeptic Review

## Proposal: Causal-Plasticity Atlas (proposal_00)
**Stage:** Implementation verification
**Review Team:** Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer
**Adjudication:** Adversarial cross-critique + Independent Verifier signoff (with veto)

---

## SCORES

| Dimension | Auditor | Skeptic | Synthesizer | Cross-Critique Consensus | Verifier Final |
|---|---|---|---|---|---|
| Code Quality | 6/10 | 3/10 | 7/10 | 5/10 | **5/10** |
| Genuine Difficulty | 5/10 | 3/10 | 7/10 | 5/10 | **5/10** |
| Value Delivered | 3/10 | 2/10 | 4/10 | 3/10 | **3/10** |
| Test Coverage | 6/10 | 3/10 | 7/10 | 4/10 | **4/10** |
| **Overall** | **4.70** | **2.75** | **6.25** | **4.25** | **4.25/10** |

---

## VERDICT: **ABANDON**

The cross-critique produced a CONTINUE (conditional) verdict at 4.25/10. This was **overturned by Independent Verifier veto** on three grounds:

1. All three individual evaluators unanimously recommended ABANDON — the cross-critique reversed unanimity without identifying what evaluators got wrong
2. 4.25/10 is below any reasonable CONTINUE threshold
3. "Fixability" of individual bugs is not grounds for CONTINUE — the evaluation scores the artifact as-submitted

---

## THREE PILLARS ASSESSMENT

### Pillar 1: Extreme and Obvious Value — FAIL (3/10)

The implementation **cannot run end-to-end**. Four critical integration failures prevent the pipeline from producing any output:

1. **ALL pairwise context alignments fail** with singular matrix errors during full pipeline execution (confirmed via `test_full_pipeline.py`)
2. **QD search crashes**: orchestrator calls `qd_engine.search()` (`orchestrator.py:1354`) but `QDSearchEngine` only exposes `run()` (`qd_search.py:1099`)
3. **Certificate generation crashes**: orchestrator passes `variable=`, `variable_index=`, `dataset=`, `foundation=`, `config=` kwargs (`orchestrator.py:1872-1878`) but `CertificateGenerator.generate()` expects `(adjacencies, datasets, target_idx, dag_learner)` (`robustness.py:264-270`)
4. **Sensitivity analysis crashes**: wrong API kwargs (`orchestrator.py:2063-2066` vs `sensitivity.py:155-161`)

A system that cannot run delivers no value, let alone extreme value.

### Pillar 2: Genuinely Difficult Software — FAIL (5/10)

The codebase contains ~5,000 lines of genuine algorithmic work within 43,558 lines total (~12% novelty ratio):

| Component | Lines | Genuine Difficulty |
|---|---|---|
| `scm.py` — Bayes-Ball d-separation, CPDAG, v-structures | 1,593 | Real but textbook |
| `mechanism_distance.py` — Multivariate √JSD with Cholesky fallback | ~800 core | Moderate |
| `cada.py` — 6-phase alignment with Hungarian matching | ~500 core | Moderate |
| `qd_search.py` — MAP-Elites with CVT tessellation | ~1,000 core | Real but unnecessary (4D space is exhaustively computable) |
| `changepoint.py` — PELT algorithm | 1,167 | Real but standard |

The remaining 88% is scaffolding: data classes, configuration, I/O, serialization, visualization, error handling, CLI, and logging. The QD search evaluator — a core novel contribution — is a **random-number placeholder** (`qd_search.py:76-139`), not an implementation.

### Pillar 3: Real Best-Paper Potential — FAIL (2/10)

- **Core math is wrong**: `_jsd_gaussian` in `plasticity.py:267-286` computes symmetric KL divergence (0.5×(KL(P||Q)+KL(Q||P))), NOT Jensen-Shannon divergence (0.5×(KL(P||M)+KL(Q||M)) where M=(P+Q)/2). This produces **3–10× overestimates** of parametric plasticity (ψ_P), corrupting all downstream classifications.
- **Two incompatible JSD implementations**: `mechanism_distance.py` has a correct moment-matched JSD implementation; `plasticity.py` has a wrong one. The pipeline uses the wrong one.
- **Zero empirical results**: No benchmark runs, no comparison against baselines, no real-data validation
- **Theory scored 4.0/10 ABANDON**: Self-contradicted runtime claims, missing CD-NOD baseline, vacuous certificates

---

## FATAL FLAWS

### Flaw 1: Core Mathematical Error (KILL SHOT)

`_jsd_gaussian` in `cpa/descriptors/plasticity.py:267-286` computes:
```python
jsd = 0.5 * (kl_12 + kl_21)  # This is symmetric KL, NOT JSD
```

The docstring even admits: "symmetric KL approximation: JSD ≈ 0.5 * KL_sym." This is not an approximation — it is a different quantity. JSD is bounded in [0, ln(2)]; symmetric KL is unbounded.

| Distributions | CPA's "JSD" | True JSD | Overestimate |
|---|---|---|---|
| N(0,1) vs N(3,1) | 4.500 | 0.527 | 8.5× |
| N(0,1) vs N(0,4) | 0.563 | 0.093 | 6.1× |
| N(0,1) vs N(5,2) | 9.500 | 0.642 | 14.8× |

The correct implementation exists in the same codebase (`mechanism_distance.py:346-400`). This is a DRY violation with mathematical consequences. Every plasticity descriptor, classification, and certificate produced by this system is numerically wrong.

**No test in the 1,452-test suite verifies JSD accuracy against known values.**

### Flaw 2: Pipeline Cannot Execute

Four API mismatches between the orchestrator and its components:

| Component | Orchestrator Calls | Actual API | Failure |
|---|---|---|---|
| QD Search | `.search(foundation=, config=, rng=)` | `.run(n_generations=, progress=)` | `AttributeError` |
| Certificates | `.generate(variable=, variable_index=, dataset=, foundation=, config=)` | `.generate(adjacencies, datasets, target_idx, dag_learner)` | `TypeError` |
| Sensitivity | `.analyze(dataset=, foundation=, config=)` | `.analyze(adj_matrix, data=, variable_names=, descriptor_fn=)` | `TypeError` |
| Alignment | All pairs fail with singular matrix | Regularization insufficient for discovered DAGs | `LinAlgError` |

Root cause: The orchestrator was written against imagined APIs, not actual component signatures. Implementation timed out with 0 polish rounds, so integration was never tested.

### Flaw 3: QD Search Evaluator is a Placeholder

`_default_evaluator` in `qd_search.py:76-139` generates fitness scores using `np.random.default_rng()`. The quality-diversity search — the project's primary claimed novelty beyond existing multi-context causal methods — is driven by random numbers, not causal analysis.

### Flaw 4: Build System Broken

`pyproject.toml` specifies `build-backend = "setuptools.backends._legacy:_Backend"`, which is not a valid setuptools backend. The package cannot be installed with `pip install -e ".[dev]"`.

### Flaw 5: Tests Miss Critical Bugs

The 1,452 unit tests pass but exhibit systematic gaps:
- Zero tests verify JSD numerical accuracy against known analytic values
- Zero tests exercise the orchestrator→component interface
- Both integration tests fail (`test_some_invariant_mechanisms`, `test_some_plastic_mechanisms`)
- Many edge-case tests use `try/except pass` — verifying "doesn't crash" not "produces correct output"

---

## WHAT SURVIVES (Scavenging Synthesizer Analysis)

### Components With Standalone Value

| Component | Lines | Standalone Value | Works? |
|---|---|---|---|
| `core/scm.py` — StructuralCausalModel | 1,593 | 8/10 | ✓ |
| `core/mechanism_distance.py` — √JSD distance | 2,373 | 8/10 | ✓ (correct JSD) |
| `core/mccm.py` — Multi-context container | 843 | 7.5/10 | ✓ |
| `detection/changepoint.py` — PELT detector | 1,167 | 7.5/10 | ✓ |
| `stats/distributions.py` — Divergence library | 1,074 | 7/10 | ✓ |
| `certificates/robustness.py` — Certificate generator | 1,296 | 7/10 | ✓ (standalone) |
| `descriptors/plasticity.py` — 4D descriptors | 1,513 | 6/10 | ✗ (wrong JSD) |
| `alignment/cada.py` — CADA aligner | 1,630 | 6/10 | ✓ (with anchors) |
| `exploration/qd_search.py` — MAP-Elites | 1,563 | 5/10 | ✗ (placeholder eval) |

### Best Salvage: `causal-mechanism-distance` Library

Extract `core/scm.py` + `core/mechanism_distance.py` + `stats/` as a standalone √JSD mechanism comparison library (~5,850 lines). This is:
- Novel (no existing Python library provides this)
- Correct (the `mechanism_distance.py` JSD is right)
- Self-contained (numpy + scipy only)
- Well-tested (965 lines of unit tests + 708 lines of numerical stability tests)

### CPA-Lite (~10K lines)

A minimal extractable toolkit comprising the mechanism distance library + MCCM container + PELT changepoint detection. Viable as a standalone pip package but constitutes a different project from CPA.

---

## KEY DISAGREEMENTS AND RESOLUTIONS

### "Code Quality: 3 vs 7?" (Skeptic vs Synthesizer)

**Resolution:** Module-level engineering is genuinely excellent (consistent typing, docstrings, error handling, numerical guards — Synthesizer is right at module scale). But a function named `_jsd_gaussian` that computes the wrong divergence IS a code quality failure — Skeptic is right at correctness scale. **Consensus: 5/10** — strong local quality, broken global integration.

### "Is the JSD bug fixable?" (Cross-critique CONTINUE rationale)

**Resolution:** Yes, in ~10 lines (import from `mechanism_distance.py`). But fixability is not the evaluation criterion. The evaluation scores what exists, not what could exist. The bug's existence for 0 polish rounds indicates it was never tested against ground truth. The Verifier correctly vetoed the CONTINUE on grounds of goalpost movement.

### "Is 43K lines genuinely difficult?" (Synthesizer 7 vs Skeptic 3)

**Resolution:** 43K lines is substantial engineering output but dominated by scaffolding. The ~5K lines of core algorithmic work (SCM, CADA, JSD, PELT) constitute moderate difficulty — real but not genuinely hard. The QD search (ostensibly the hardest novel component) has a placeholder evaluator. **Consensus: 5/10.**

### "Are the tests adequate?" (Synthesizer 7 vs Skeptic 3)

**Resolution:** The numerical stability test battery (`test_numerical_stability.py`, 708 lines) is genuinely excellent engineering. But 1,452 tests that miss a 3–10× mathematical error in the core divergence function demonstrate a systematic gap: tests verify engineering (shapes, types, non-crash) but not mathematics (numerical correctness). **Consensus: 4/10.**

---

## PROCESS NOTES

**Team composition:** Independent Auditor (evidence-based scoring), Fail-Fast Skeptic (aggressive rejection), Scavenging Synthesizer (salvage maximization). All three operated as background agents with full codebase access.

**Workflow:**
1. Independent evaluations (3 parallel agents, ~2-5 min each)
2. Lead verified Skeptic's kill shot (confirmed: `_jsd_gaussian` computes symmetric KL)
3. Adversarial cross-critique (1 agent synthesizing all three positions)
4. Cross-critique produced CONTINUE at 4.25/10
5. Independent Verifier reviewed process, found procedural irregularity (unanimity override), **vetoed CONTINUE**
6. Final verdict: ABANDON

**Key concessions during cross-critique:**
1. Skeptic's CADA "all-None" claim was partially overstated — CADA does run without anchors, just without name-based auto-matching
2. Synthesizer's test count of 1,513 could not be independently verified at that scale
3. Cross-critique lowered Synthesizer's scores but upgraded verdict — procedurally inconsistent

**Verifier veto rationale:**
- All three evaluators said ABANDON (unanimous)
- Cross-critique reversed to CONTINUE without identifying evaluator errors
- 4.25/10 composite contradicts CONTINUE recommendation
- "Fixability" arguments constitute goalpost movement

---

## IMPLEMENTATION-SPECIFIC FINDINGS

| Metric | Value |
|---|---|
| Total source lines | 43,558 (55 files) |
| Total test lines | 15,651 (47 files) |
| Unit tests passing | 1,452 / 1,452 (100%) |
| Integration tests passing | 44 / 46 (95.6%) |
| Build system | BROKEN (invalid pyproject.toml) |
| End-to-end pipeline | BROKEN (4 API mismatches) |
| Polish rounds completed | 0 (timed out) |
| Core math correctness | WRONG (JSD bug in plasticity.py) |
| QD search evaluator | PLACEHOLDER (random numbers) |

---

## VERDICT: **ABANDON**

### Rationale

The Causal-Plasticity Atlas implementation fails all three pillars:

1. **Value is zero, not extreme (3/10).** The pipeline cannot run end-to-end. Four API mismatches, a broken build system, and a mathematical error in the core divergence function mean no user can obtain any result from this software.

2. **Difficulty is moderate, not genuinely hard (5/10).** ~5K lines of real algorithmic work are buried in 43K lines of scaffolding. The QD search — the primary claimed novelty — has a placeholder evaluator. The remaining components (OLS + JSD + Hungarian matching + bootstrap) are standard techniques.

3. **Best-paper potential is absent (2/10).** Wrong core math, zero empirical results, placeholder algorithms, and a failed theory gate (4.0/10 ABANDON) leave no path to publication at any venue.

### What Should Happen

- **Abandon CPA as an integrated system.** The orchestrator-to-module API mismatches indicate the system was never coherently designed at the integration level.
- **Consider extracting `causal-mechanism-distance`** (~5,850 lines) as a standalone library if there is independent demand. This is the highest-value component and it works correctly.
- **Do not spend polish rounds.** The integration failures are not polish-level fixes; they require architectural reconciliation of the orchestrator with actual component APIs.

### Rankings

```json
{
  "rankings": [
    {
      "proposal_id": "proposal_00",
      "score": 4.25,
      "verdict": "ABANDON",
      "reason": "Pipeline cannot run end-to-end (4 API mismatches), core JSD computation is mathematically wrong (3-10x overestimates), QD evaluator is a random-number placeholder, build system broken, 0 polish rounds completed. Theory was already ABANDON at 4.0/10. Three pillars: 0/3 met. All three evaluators unanimously recommended ABANDON.",
      "scavenge_from": []
    }
  ]
}
```

---

*Report generated by three-expert verification team with Independent Verifier signoff. Cross-critique CONTINUE verdict overturned by Verifier veto. All scores derived from direct source code examination of 43,558 lines in `cpa/` and 15,651 lines in `tests/`.*

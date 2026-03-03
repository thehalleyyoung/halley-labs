# Implementation Evaluation: Pragmatist Lens

## proposal_00 — Causal-Plasticity Atlas (CPA)

**Evaluator**: Hard-nosed pragmatist
**Method**: Claude Code Agent Teams (3 experts + adversarial critique + independent verifier signoff)
**Date**: 2026-03-02

---

## Team Composition & Process

| Role | Task | Verdict |
|------|------|---------|
| Independent Auditor | Evidence-based scoring with file:line citations | ABANDON |
| Fail-Fast Skeptic | Aggressive flaw-finding (found 3 FATAL + 7 MAJOR) | ABANDON |
| Scavenging Synthesizer | Salvage assessment of components | ABANDON (but notes salvageable pieces) |
| Adversarial Critique | Resolved 5 inter-teammate disagreements | ABANDON (consensus) |
| Independent Verifier | Spot-checked 7 critical claims, refuted 1, partially corrected 2 | ABANDON (with score adjustments) |

All five evaluation steps independently reached ABANDON. Disagreements were about degree, not direction.

---

## Scores

| Dimension | Score | Evidence |
|-----------|-------|----------|
| **Code Quality** | **5/10** | Clean module architecture (15 subpackages, proper typing, dataclasses). But: broken pyproject.toml build backend (`setuptools.backends._legacy:_Backend` doesn't exist), numba listed as hard dependency but zero `@jit`/`@njit` uses anywhere, QD Phase 2 interface mismatch silently falls to random fallback (`orchestrator.py:1354` calls `qd_engine.search()` but actual method is `run()`), ψ_CS computed everywhere but has zero effect on classification (`classification.py:327-400` never branches on it). Two of three pipeline phases have fundamental defects masked by broad exception handlers. |
| **Genuine Difficulty** | **6/10** | CADA alignment is the strongest component: a genuine 6-phase pipeline (anchor propagation → Markov blanket filtering → CI-fingerprint + distribution scoring → Hungarian matching → edge classification → divergence scoring) at ~2,500 lines. This is real algorithmic work applying known methods in a domain-specific configuration. However, all other "algorithms" are standard: PELT changepoint detection (Killick 2012), vanilla MAP-Elites (Mouret & Clune 2015), bootstrap certificates with permissive thresholds, and textbook JSD/KL metrics. The difficulty is integration engineering of known methods, not algorithmic innovation. |
| **Value Delivered** | **4/10** | The pipeline doesn't work end-to-end. All integration tests fail (47-54 failures depending on run). QD search always falls back to random pattern sampling because of interface mismatch. Certificate system certifies almost everything (UCB ≤ 0.5 on √JSD ≈ 60% of theoretical max). Zero real-data validation. Zero baseline comparisons (no CD-NOD, no ICP, no LPCMCI). Target audience is extremely narrow (multi-context causal mechanism plasticity researchers — a community that barely exists yet). No evidence the system works on any actual causal discovery problem. |
| **Test Coverage** | **3/10** | 1,523 tests pass, which looks impressive. But: ~45% of assertions are smoke-level (`assert X is not None`, `assert len(X) > 0`). 11+ visualization integration tests use bare `except Exception: pass` — structurally cannot fail. Integration tests don't catch that Phase 2 QD search always throws AttributeError and falls back. Zero property-based tests (hypothesis is in dev deps but never imported). The test suite creates a false sense of correctness — it passed while two of three pipeline phases were fundamentally broken. |

**Composite: (5 + 6 + 4 + 3) / 4 = 4.5 / 10**

---

## Fatal Flaws (verified by independent signoff)

### 1. Pipeline Never Runs End-to-End
The orchestrator has interface mismatches with its own components:
- `orchestrator.py:1354` calls `qd_engine.search(foundation=..., config=..., rng=...)` but `QDSearchEngine.run()` takes `(n_generations=None, progress=True)` — wrong method name AND wrong parameters
- `FallbackDiscovery.discover()` called but actual method is `run()`
- `PlasticityComputer.compute_all()` called but actual method is `compute()`
- `CertificateGenerator.__init__()` receives unexpected `n_bootstrap` kwarg
- All caught by `except Exception` → silent fallback → false "success"

### 2. Fourth Descriptor Dimension Is Dead Code
ψ_CS (Context Sensitivity) is computed, stored, bootstrapped, displayed, and documented — but `_apply_hierarchy()` in `classification.py:327-400` never uses it in any classification branch. The "4D plasticity descriptor" is functionally 3D. This is a core theoretical claim of the paper.

### 3. Zero Algorithmic Novelty
Every ALG 1-5 implementation uses standard, well-known methods:
- ALG 1 (CADA): `scipy.optimize.linear_sum_assignment` + Jaccard + KL (though the 6-phase orchestration is non-trivial engineering)
- ALG 2 (Descriptors): √JSD on Bernoulli/Gaussian + CV — arithmetic
- ALG 3 (MAP-Elites): Vanilla QD with synthetic evaluator (never connects to real pipeline)
- ALG 4 (Tipping Points): PELT + permutation test + BH-FDR — textbook
- ALG 5 (Certificates): Bootstrap CI + threshold — statistics 101

### 4. Test Suite Is a Potemkin Village
Tests pass (95%+) while the pipeline is fundamentally broken because:
- Integration tests wrap assertions in `try/except Exception: pass`
- 45% of assertions test `is not None` rather than correctness
- No tests verify end-to-end pipeline output quality

---

## Pragmatist Constraints Check

| Constraint | Pass? | Notes |
|------------|-------|-------|
| Buildable in 1 day (150K LoC budget) | ✅ | 52K LoC is within budget |
| Runs on laptop CPU | ✅ | No GPU deps (numba listed but unused) |
| No human annotation | ✅ | Fully synthetic data |
| Fully automated evaluation | ❌ | No automated evaluation against ground truth exists |
| Extreme and obvious value | ❌ | Moderate value for a niche audience; no working pipeline |
| Actually installable | ❌ | pyproject.toml has non-existent build backend |

---

## Salvage Assessment (from Scavenging Synthesizer)

If abandoned, these components have independent value:

| Component | Lines | Value | Why |
|-----------|-------|-------|-----|
| CADA alignment (`cpa/alignment/`) | ~2,500 | Moderate | 6-phase DAG alignment is genuine engineering; DAG alignment is underserved |
| StructuralCausalModel (`cpa/core/scm.py`) | ~1,600 | Low-Moderate | Clean SCM implementation, but pgmpy/causal-learn exist |
| Mechanism changepoints (`cpa/detection/`) | ~2,400 | Moderate | Novel framing (causal mechanism changepoints vs. data drift) |
| Plasticity descriptors (`cpa/descriptors/`) | ~1,500 | Low | The conceptual framework is interesting; implementation is trivial arithmetic |

A focused ~14K-line project extracting CADA + changepoints + descriptors could be valuable — but that's a different project requiring separate evaluation.

---

## VERDICT: ABANDON

**Confidence: HIGH**

All five evaluation steps (3 independent experts + adversarial critique + verifier signoff) unanimously reached ABANDON. The core problems are structural:

1. **The system doesn't work.** The pipeline orchestrator has never successfully run end-to-end. Interface mismatches are caught by broad exception handlers and silently degraded. This isn't a polish issue — it's evidence that integration was never completed.

2. **There is no novelty.** The strongest component (CADA) is good *engineering* of known methods, not a new algorithm. The other four "algorithms" are textbook statistics.

3. **The tests hide the problems.** A test suite where 45% of assertions check `is not None` and integration tests use `try/except: pass` provides negative value — it creates false confidence in broken code.

4. **No one is waiting for this.** The target audience (multi-context causal mechanism plasticity researchers) is so narrow that the project would struggle to find 10 users. The value proposition — "organize standard causal metrics into a 4D descriptor and search with MAP-Elites" — doesn't solve a burning problem for anyone.

52,857 lines of well-structured code wrapping scipy, scikit-learn, and textbook statistics into a pipeline that doesn't connect. The quantity of code masks the absence of depth.

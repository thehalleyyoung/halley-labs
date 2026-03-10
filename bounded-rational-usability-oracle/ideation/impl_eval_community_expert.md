# Implementation Evaluation: Bounded-Rational Usability Oracle (proposal_00)

**Evaluator**: HCI Community Expert (area-042-human-computer-interaction)
**Date**: 2026-03-04
**Method**: Three-expert adversarial team (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) with cross-challenge rounds and independent verification signoff.

---

## Summary

| Criterion | Score | Rationale |
|-----------|-------|-----------|
| **Code Quality** | **7/10** | Library modules are excellent (type hints, LaTeX docstrings, numerical stability); pipeline stage executors are broken stubs |
| **Genuine Difficulty** | **7/10** | Bounded-rational bisimulation, free-energy MDP solving, and compositional cost algebra are research-grade; but one of four hard problems (bottleneck classification) is bypassed, and cost algebra defaults to additive |
| **Value Delivered** | **5/10** | Rich component library exists but the pipeline produces meaningless verdicts through a 42-line comparison stub; the real comparison engine is never called |
| **Test Coverage** | **7/10** | 2,503/2,568 tests pass (97.5%); strong property-based tests verify algebraic axioms; no test verifies end-to-end regression detection correctness |
| **Real-world Format Support** | **8/10** | 6 platforms (ARIA, Chrome DevTools, axe-core, iOS, Android, Windows) with real role taxonomies; SARIF output for GitHub Code Scanning |
| **VERDICT** | **CONTINUE** | Conditional — core intellectual contribution is genuine and novel; requires fixing the disconnected verdict pipeline |

**Overall**: 6.8/10

---

## Detailed Findings

### 1. Code Quality (7/10)

**Library code is genuinely excellent.** Across 153 source files (~62,500 lines), the cognitive models (`cognitive/fitts.py`, `cognitive/hick.py`, `cognitive/visual_search.py`, `cognitive/working_memory.py`), cost algebra (`algebra/`), and bisimulation module (`bisimulation/cognitive_distance.py`) demonstrate:
- Numerically stable implementations (log-sum-exp clamping, overflow protection)
- Full type hints throughout
- Comprehensive docstrings with LaTeX formulas and academic citations
- Zero TODOs/FIXMEs in the entire codebase
- Only 1 `NotImplementedError` (correct abstract base usage in `stages.py:93`)

**Pipeline wiring is broken.** The `ComparisonStageExecutor` (`stages.py:340-379`) is a 42-line stub that computes average absolute policy divergence with hardcoded magic thresholds (0.1, -0.05). Critically:
- It uses `abs()` on every diff (line 365), making `avg_diff` always ≥ 0
- The "improvement" branch (line 372-373: `elif avg_diff < -0.05`) is **provably dead code**
- It never imports or calls `PairedComparator` from `comparison/paired.py`, which implements the real statistical comparison engine (Welch's t-test, Mann-Whitney U, bootstrap CIs, formal verdicts)

**Infrastructure overhead is moderate (~14%)**: `core/` contains 4,267 lines of types, protocols, enums, errors, config. This is standard for a well-typed Python library, not excessive boilerplate.

### 2. Genuine Difficulty (7/10)

**Three of four hard problems are genuinely implemented:**

1. **Bounded-rational bisimulation** ✅ — `bisimulation/cognitive_distance.py` implements the novel metric `d_cog(s₁,s₂) = sup_{β'≤β} d_TV(π_{β'}(·|s₁), π_{β'}(·|s₂))` via soft value iteration + grid-refined supremum optimization. This bridges Givan/Dean/Greig (2003) bisimulation theory with Ortega & Braun (2013) bounded rationality — a combination that does not exist in any prior HCI tool.

2. **Compositional cost algebra** ✅ (partial) — `algebra/` implements the 4-tuple `(μ, σ², κ, λ)` with sequential ⊕ and parallel ⊗ operators, including MRT-backed interference modeling. **However**, the coupling parameter defaults to 0.0, making sequential composition purely additive out of the box. The non-additive capability exists but is opt-in, and the pipeline never sets it.

3. **MDP construction** ✅ — `mdp/builder.py` constructs state = (focus_node × task_progress_bitvector) with actions derived from interactive capabilities, costs integrating Fitts' + Hick-Hyman + working memory.

4. **Information-theoretic bottleneck classification** ❌ — `bottleneck/signatures.py` correctly computes H(π), I(A;S'), channel utilization ρ. But the classifier (`bottleneck/classifier.py:207-265`) dispatches to 5 heuristic detectors using hardcoded thresholds (`entropy > 3.0 nats`, `fitts_id > 4.5 bits`, `item_count > 4`), ignoring the IT signatures entirely. The wiring gap is ~10 lines but represents a conceptual shortcut.

### 3. Value Delivered (5/10)

**The critical defect: two verdict engines, pipeline uses the wrong one.**

The library (`comparison/paired.py`, `comparison/hypothesis.py`) implements a statistically rigorous regression testing engine: Monte Carlo trajectory simulation → Welch's t-test → Cohen's d effect size → formal verdict with p-values and confidence intervals. This is what the theory promises.

The pipeline (`pipeline/stages.py:340-379`) delivers a different thing entirely: raw policy-vector Euclidean distance with magic thresholds. A user running `usability-oracle diff before.html after.html` gets the stub, not the real engine.

**End-to-end execution has never succeeded.** `State.json` confirms `impl_timeout: true` and `impl_polish_rounds_completed: 0`. The 65 failing tests are concentrated in integration tests that exercise the pipeline.

**What works in isolation:** Individual cognitive models produce correct predictions. The MDP solver converges. The bisimulation reduction partitions state spaces. The formats parsers handle real HTML/ARIA markup. But these components are never orchestrated into a correct end-to-end verdict.

### 4. Test Coverage (7/10)

**Quantitative:** 2,503 passed, 65 failed, 1 warning (97.5% pass rate). 90 test files: 72 unit, 10 integration, 6 property, 2 fixture modules.

**Property tests are the highlight.** Using Hypothesis with domain-appropriate strategies:
- Cost algebra: associativity, commutativity, identity, monotonicity (200 examples each)
- Interval arithmetic: inclusion isotonicity, width monotonicity, fundamental theorem
- Policy: probability axioms, entropy bounds, KL non-negativity, Gibbs' inequality
- Bisimulation: metric space axioms (symmetry, triangle inequality, identity of indiscernibles)

**Critical gap:** No test demonstrates the core value proposition: "UI changed → cognitive cost increased → regression correctly detected." There are structural smoke tests for the pipeline but no semantic correctness verification of verdicts.

**Environment note:** Tests require `pytest -p no:qelens` due to an unrelated plugin conflict on this machine. This is not a project defect.

### 5. Real-world Format Support (8/10)

Six platform parsers with auto-detection:
- **ARIA/HTML** (`formats/aria.py`, `accessibility/html_parser.py`): 52+ ARIA roles with required states, owned elements, parent contexts. Full name computation algorithm (aria-label → aria-labelledby → alt → title → text). lxml + html5lib dual backend.
- **Chrome DevTools** (`formats/chrome_devtools.py`): 37 CDP role mappings, handles real protocol structures
- **axe-core** (`formats/axe_core.py`): 3.x & 4.x format support, violation→bottleneck mapping
- **iOS/Android/Windows**: Platform-specific parsers present but not deeply validated

**Output formats:** SARIF v2.1.0 (GitHub Code Scanning compatible), JSON, HTML, console. Exit codes follow CI/CD conventions (0/1/2).

**Limitation:** HTML fixtures are 28-55 lines each. No validation against production-scale pages (hundreds of elements).

---

## Critical Defects Requiring Fix for CONTINUE

1. **Wire `PairedComparator` into `ComparisonStageExecutor`** — Replace the 42-line stub with calls to the real statistical comparison engine. ~50 lines of integration code.
2. **Fix dead "improvement" branch** — Remove `abs()` or restructure the diff computation.
3. **Wire IT signatures into bottleneck detectors** — `classify_signature()` already exists in `signatures.py:107-173`; it just needs to be called.
4. **Add one end-to-end regression detection test** — Two MDPs with known cost differential, verify correct verdict.

---

## Community Reception Assessment

**HCI/CHI academic community**: High interest. The bounded-rational bisimulation and free-energy framing are genuinely novel. But reviewers will immediately ask "Where's the user study?" with no human-participant validation.

**DevOps/developer adoption**: Low without significant UX work. Task specification requires MDP knowledge. No zero-config mode. No browser-to-accessibility-tree extraction pipeline. No GitHub Action marketplace entry.

**Marginal contribution over existing tools**: Genuine. No existing CI/CD tool models users as information-constrained agents. CogTool requires manual storyboards; axe-core checks WCAG rules, not cognitive costs; Lighthouse gives generic scores, not task-specific regression verdicts.

---

## Salvageable Components (if ABANDON)

| Module | Standalone Value | Lines |
|--------|-----------------|-------|
| `cognitive/` (Fitts, Hick, visual search, WM) | ⭐⭐⭐⭐⭐ | 3,944 |
| `algebra/` (cost algebra with MRT) | ⭐⭐⭐⭐ | 3,033 |
| `comparison/hypothesis.py` (statistical tester) | ⭐⭐⭐⭐ | ~350 |
| `interval/` (interval arithmetic) | ⭐⭐⭐ | ~500 |
| `formats/` (6-platform parsers) | ⭐⭐⭐ | 2,155 |
| `core/constants.py` (HCI psychophysical constants) | ⭐⭐⭐ | 392 |

---

## Verdict: **CONTINUE**

The intellectual core is genuine, novel, and addresses a real gap in HCI tooling. The implementation contains research-grade components (bisimulation, cognitive models, cost algebra) that don't exist in any other automated usability tool. The critical defect — disconnected verdict pipeline — is a wiring problem fixable in 1-2 days, not an architectural failure.

**Conditions for CONTINUE:**
- Fix the `ComparisonStageExecutor` to use `PairedComparator` (mandatory)
- Add at least one end-to-end regression detection test (mandatory)
- Wire bottleneck IT signatures into detectors (recommended)
- Set non-zero coupling default in pipeline config (recommended)

**Risk:** Without these fixes, the tool produces meaningless verdicts through its primary interface, making it a well-documented engine with the transmission disconnected.

---

*Evaluation produced by three-expert adversarial team with independent verification signoff. All scores evidence-based with specific file:line citations verified against the codebase.*

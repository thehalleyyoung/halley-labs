# Pragmatist Evaluation: proposal_00 — Bounded-Rational Usability Oracle

**Evaluator:** Hard-nosed pragmatist (team-based adversarial verification)
**Date:** 2026-03-04
**Method:** Three-reviewer adversarial process (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) with cross-challenge synthesis and independent verification.

---

## Executive Summary

A 76K-line (62K source + 31K test + 1.8K examples) Python tool that models UI users as bounded-rational information-theoretic agents to detect usability regressions in accessibility trees. Implementation timed out during polish, leaving an integration-level hashability bug that cascades through ~148 of 2,568 tests. The mathematical core (MDP solvers, soft Bellman policies, cognitive cost models, cost algebra, statistical comparison) is substantive and correct. The integration/pipeline layer is broken but fixable.

---

## Scores

| Dimension | Score | Evidence |
|---|---|---|
| **Code Quality** | **7/10** | 179 source files across 27 well-organized subpackages. Consistent docstrings with LaTeX math and academic citations. Proper numerical handling (log-sum-exp stability, epsilon guards). Clean Protocol-based abstractions in `core/`. Deducted for: broken `pyproject.toml` build backend (`setuptools.backends._legacy:_Backend`), the `State.features: dict` unhashability bug in `mdp/models.py` that cascades through bottleneck/pipeline/repair modules, and ~20% docstring padding inflating line counts. |
| **Genuine Difficulty** | **7/10** | The MDP solvers themselves are textbook Puterman (Value Iteration, Policy Iteration, LP relaxation). The cognitive models are well-known single formulas (Fitts 1954, Hick 1952). However, the *composition* is genuinely novel: (1) bounded-rational soft Bellman equation (Ortega & Braun 2013) applied to UI navigation MDPs, (2) a 4-moment cognitive cost algebra `(μ, σ², κ, λ)` with carry-over coupling and superadditive tail risk verified against algebraic axioms, (3) bounded-rational ε-bisimulation with cognitive distance metric. These three contributions go beyond plumbing standard algorithms together. Not a research breakthrough, but solid graduate-level applied math. |
| **Value Delivered** | **6/10** | Addresses a real gap: fully automated, quantitative usability regression detection without human annotators. Full pipeline from HTML/JSON → accessibility tree → MDP → bounded-rational policy → statistical comparison → SARIF verdict exists conceptually. However: (1) the end-to-end pipeline is broken (30% integration test failure), (2) zero validation against real-world accessibility trees (test fixtures max at 78 lines of JSON), (3) no CI pipeline config despite claiming CI/CD integration, (4) no evidence it produces actionable results on real applications. High potential value, low demonstrated value. |
| **Test Coverage** | **7/10** | 2,568 tests: 2,401 pass (93.5%), 167 fail. 85 unit test files, 11 integration test files, 6 property test files using Hypothesis. Tests are mathematically rigorous — verifying Bellman equations, cross-algorithm agreement (VI vs PI), KL divergences, algebraic axioms, effect sizes. Not smoke tests. The 167 failures trace to a single root cause (unhashable `dict`/`list` fields in frozen dataclasses), not systemic rot. 2 property test "failures" are actually test calibration issues (softmax threshold too aggressive at β=100 with near-equal Q-values), not mathematical bugs. The core algorithmic modules have zero failures. |
| **Format Support** | **7/10** | 6 platform parsers: HTML (40+ implicit role mappings, 18 input type mappings, ARIA name computation, lxml+html5lib dual-parser), JSON (auto-detects Chrome DevTools, axe-core, generic), ARIA (complete WAI-ARIA 1.2 taxonomy, 50+ roles with validation), plus stubs for iOS/Android/Windows. Real format implementations, not toy parsers. Deducted for: no validation against production accessibility trees, fixtures are hand-crafted toy UIs (max 78-line JSON, ~41KB total fixture data), and platform parsers (iOS/Android/Windows) are likely stubs. |

**Composite Score: 6.8/10**

---

## Constraint Compliance

| Constraint | Pass/Fail | Detail |
|---|---|---|
| Laptop CPU only | ✅ PASS | Dependencies: numpy, scipy, z3-solver, networkx, lxml, click. No GPU libraries. |
| Buildable in 1 day at 150K LoC/day | ✅ PASS | ~95K total lines including tests, well under 150K capacity. |
| Fully automated, zero human involvement | ✅ PASS | No human annotation, no human studies. Statistical comparison is fully automated. |
| Runs on laptop CPU | ✅ PASS | All computation is numpy/scipy. Z3 is CPU-only. |

---

## Key Findings from Adversarial Process

### Resolved Disagreements

1. **"30% integration failure = catastrophic" vs "single bug, fixable"** — *Resolved: Auditor wins.* Root cause analysis confirms 85/86 unit failures and the majority of integration failures trace to `State.features: dict[str, float]` used in a `frozen=True` dataclass. Converting to `frozenset` or adding `__hash__` would likely fix 80-90% of all failures. This is a 10-line fix, not systemic rot.

2. **"Property test mathematical bugs" vs "numerical edge case"** — *Resolved: Auditor wins.* The `test_large_beta_approaches_greedy` failure at β=100 with Q-values [0.0, 0.015625] is mathematically correct (softmax gives 0.827, not >0.9). The test threshold is too aggressive for near-equal inputs. The log-sum-exp implementation is correct. This is a test calibration issue.

3. **"Textbook difficulty" vs "genuinely hard"** — *Resolved: Split.* MDP solvers ARE textbook (Skeptic correct). But the cost algebra, bisimulation theory, and the overall composition framework ARE novel (Auditor correct). Final: 7/10, acknowledging both.

4. **"76K LoC inflation"** — *Resolved: Clarified.* 62.5K source (of which ~39K non-blank non-docstring code) + 31.4K tests + 1.8K examples = 95.7K total. The "76K" figure appears to be source+tests minus caches. Line count includes ~20% docstrings, which is high but not deceptive. The test-to-code ratio of 0.8:1 is healthy.

### Unresolved Weaknesses

1. **No real-world validation.** The entire system has never been run on a real application's accessibility tree. Fixtures are sub-100-line toy UIs. This is the single biggest concern.

2. **Broken build config.** `pyproject.toml` specifies a non-existent setuptools backend. The tool literally cannot be pip-installed without manual patching.

3. **No polish rounds completed.** Implementation timed out before any polish, leaving integration bugs unfixed.

4. **Who specifically needs this?** The target user (CI/CD pipeline maintainers who care about accessibility regressions) is plausible but the tool provides no evidence it detects regressions that matter in practice. The examples are synthetic. No case study, no comparison to existing accessibility linting tools (axe-core, pa11y, Lighthouse).

---

## Minimum Viable Product Assessment

The Synthesizer identified a ~16K LoC core (mdp/ + policy/ + cognitive/ + algebra/ + comparison/ + interval/ + formats/) that has **zero test failures** and delivers the core value proposition: parse accessibility tree → build MDP → compute bounded-rational policy → compare costs → emit regression verdict. The broken modules (bottleneck classification, repair synthesis, pipeline orchestration, fragility analysis) are diagnostic enrichments, not core detection logic.

---

## VERDICT: **CONTINUE**

**Rationale:** The mathematical core is correct, the novelty (cognitive cost algebra, bounded-rational bisimulation, soft Bellman applied to accessibility MDPs) is real if not earthshaking, and the constraint compliance is clean (CPU-only, automated, buildable in a day). The 148 test failures are traceable to a single data model bug, not fundamental design problems. The 16K LoC working core already delivers the core value proposition.

**Conditions for continued confidence:**
1. Fix the `State`/`Action` hashability bug (convert `dict`/`list` fields to immutable types) and verify cascade fix restores >98% pass rate.
2. Fix the `pyproject.toml` build backend to `setuptools.build_meta`.
3. Recalibrate 2 property test thresholds.
4. The absence of real-world validation data is concerning but not disqualifying at this stage.

**Risk of abandonment:** Low. The intellectual substance is real, the working modules are correctly implemented, and the integration bugs are mechanical, not architectural.

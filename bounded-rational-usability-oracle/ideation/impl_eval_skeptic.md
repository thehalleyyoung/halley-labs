# Implementation Evaluation — Skeptic Review

**Reviewer:** Rigorous Skeptic (Team Lead)  
**Date:** 2026-03-04  
**Methodology:** Claude Code Agent Teams — three independent evaluators (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) with adversarial cross-critique and synthesis.

---

## proposal_00: Bounded-Rational Usability Oracle

**Claimed:** 76,324 LOC | Information-theoretic cognitive cost analysis for automated usability regression testing in CI/CD.  
**Reality:** 62,553 source LOC (35,338 effective logic), 31,384 test LOC. Zero polish rounds. Timed out during implementation.

---

### Scores

| Criterion | Score | Evidence Summary |
|-----------|-------|------------------|
| **Code Quality** | **7/10** | 179 source files, 24 packages, zero stubs (1 abstract-method `NotImplementedError`), zero copy-paste duplication. Clean protocol-driven architecture with real inter-module imports (109 files cross-reference other `usability_oracle.*` modules). Deductions: pipeline integration broken (`beta_min` attribute mismatch proves end-to-end never ran), Hick-Hyman formulation inconsistency between core and pipeline modules, 152 test failures in outer layers. |
| **Genuine Difficulty** | **8/10** | Novel cost algebra with sequential/parallel/context operators (⊕, ⊗, Δ) implementing coupling formula `μ_{a⊕b} = μ_a + μ_b + ρ·√(σ²_a·σ²_b)` — original math, not textbook. Bounded-rational bisimulation metric `d_cog(s₁,s₂) = sup_{β'≤β} d_TV(π_{β'}(·|s₁), π_{β'}(·|s₂))` bridging model checking and cognitive science — novel. KL-regularized free-energy policy solver `F(π) = E_π[cost] + (1/β)·D_KL(π ‖ p₀)` with soft Bellman equations — correctly implemented. MDP construction from accessibility trees via Cartesian product of (focus node × task-progress bitvector) — genuine contribution. All verified in source code, not stubs. |
| **Value Delivered** | **5/10** | The system has **never run end-to-end**. Pipeline crashes at policy stage due to `beta_min` attribute not existing on `PolicyConfig` (which stores `beta_range: Tuple[float,float]`). No real UI has ever been processed. No comparison to trivial baseline (e.g., axe-core element-count diff). Theoretical value proposition is novel (CI/CD cognitive cost regression detection) but zero empirical grounding. Core algorithms verified to work in isolation: value iteration produces correct V*(s), Fitts' law matches hand-calculation to 9 decimal places, softmax policy exhibits correct thermodynamic behavior. But isolated correctness ≠ delivered system value. |
| **Test Coverage** | **6/10** | 2,568 test functions across 84 files. 94.1% pass rate (2,416/2,568) per Auditor. 468 core algorithm tests pass with 0 failures. 146 Hypothesis property-based tests (144 pass). However: 152 failures concentrated in bottleneck (54), integration/pipeline (60), fragility (11), repair (13) — exactly the modules representing end-to-end value. The `beta_min` bug proves integration tests never exercised the actual pipeline code path. Core coverage: strong. System coverage: absent. |
| **Format Support** | **6/10** | Seven parsers implemented (ARIA, Android, iOS, Chrome DevTools, Windows UIA, axe-core, HTML). HTML parser has real lxml/html5lib parsing with 50+ tag→role mappings and ARIA name computation. 4 HTML + 4 JSON fixture files with proper ARIA markup. However: test fixtures only cover HTML/JSON — no real Chrome DevTools Protocol dumps, Android view hierarchies, iOS accessibility trees, or Windows UIA exports. Android parser uses fragile regex-based XML parsing instead of lxml.etree. No end-to-end format→verdict path is working. Claimed multi-platform breadth is unvalidated. |

**Overall: 6.4/10**

---

### Red Flags

1. **Pipeline has never completed end-to-end.** The `beta_min` → `beta_range` attribute mismatch at `runner.py:237` is a smoking gun. This is a 5-minute fix, but its existence proves zero integration testing occurred. All stages downstream of bisimulation (policy solving, comparison, verdict, repair) are untested in integration.

2. **Zero empirical grounding.** No real UI has ever been processed by this system. No comparison to any baseline, trivial or otherwise. The value proposition is entirely theoretical.

3. **LOC figure in State.json is inaccurate.** `impl_loc: 76324` doesn't match any clean measurement. Actual: 62,553 source or 93,937 total. Effective logic: ~35,338 lines. Sloppy bookkeeping.

4. **Hick-Hyman formulation inconsistency.** Core module uses `log₂(n)`, pipeline uses `log₂(n+1)`. Small but corroborates that pipeline code was written separately from core modules without integration testing.

5. **152 test failures in the features that matter most.** Bottleneck detection (54 failures), pipeline integration (60 failures), repair synthesis (13 failures) — these are the user-facing differentiators, and none of them work.

6. **Bounded-rationality literature is thin.** Cites only Ortega & Braun (2013) and Todorov (2007/2009). The 70-year tradition of bounded rationality in behavioral economics and cognitive science (Simon, Gigerenzer, Sims, Matejka) is absent. The name "Bounded-Rational Usability Oracle" overpromises relative to the actual technique (entropy-regularized MDP solving).

7. **z3-solver is a ~200MB dependency for a single untested module** (repair/synthesizer.py). The repair feature has never been tested end-to-end and is a self-described stretch goal.

### Green Flags

1. **Zero stubs across 179 source files.** Every function has a real implementation. No `pass` placeholders, no `NotImplementedError` (except 1 abstract base method). This is NOT generated scaffolding.

2. **Novel mathematical contributions verified in code.** The cost algebra coupling formula, bisimulation supremum metric, and free-energy policy solver are all present and mathematically correct. The Auditor verified value iteration, Fitts' law, Hick-Hyman, and softmax policy against hand-calculations.

3. **468 core algorithm tests pass with 0 failures.** The mathematical foundation (cognitive models, MDP solvers, algebra, bisimulation, policy) is solid and well-tested.

4. **146 Hypothesis property-based tests.** Genuine property testing of algebraic axioms (commutativity, associativity), cognitive model invariants, and policy normalization. This is sophisticated testing practice.

5. **CPU-only, fully automated.** Zero torch/tensorflow/jax imports. No GPU requirement. No human annotation or studies. The z3 dependency is CPU-native. Meets the laptop-CPU constraint.

6. **Prior art gap is real.** No existing tool combines: automated MDP construction from accessibility trees + non-additive cost algebra + bounded-rational policy solving + CI/CD integration with formal error bounds. CogTool requires manual demo recording. GOMS requires manual task analysis. ACT-R requires manual model construction.

7. **Architecture is genuinely sophisticated.** 24 packages with clear layered separation. Protocol-driven design means modules can be replaced independently. 109 files with real cross-module imports forming a coherent dependency graph.

---

### Prior Art Assessment

The system goes substantially beyond GOMS/KLM (1983) in three novel directions:
1. **Bounded-rational policy solving** — softmax/free-energy agents, not deterministic optimal
2. **Bisimulation-based state abstraction** — reducing MDP size while preserving behavioral equivalence under cognitive capacity constraints
3. **Interval-valued cost propagation** — formal error bounds through the entire pipeline (IEEE-754 interval arithmetic citing Moore, Kearfott & Cloud 2009)

The CI/CD integration angle (deterministic verdicts, SARIF output, exit codes) is genuinely novel — no existing cognitive modeling tool targets automated pipelines.

However, the "trivial baseline" challenge remains unresolved: a 50-line script comparing axe-core element counts between UI versions catches monotone structural regressions with zero formalism. The formal framework's added value over this baseline is undemonstrated.

---

### Team Disagreement Summary

| Question | Auditor | Skeptic | Synthesizer | Resolution |
|----------|---------|---------|-------------|------------|
| LOC reality | 61K source | ~33K effective | 70% algorithmic | **Skeptic closest** (35K effective). Irrelevant to verdict. |
| Pipeline broken? | Incomplete integration | FATAL flaw | Almost working | **Auditor correct** — fixable integration issue, not architectural failure. |
| Test pass rate | 94.1% is good | Failures in critical paths | Didn't verify | **Both Auditor and Skeptic correct** — complementary assessments. |
| CONTINUE/ABANDON | CONTINUE (7.0) | ABANDON | CONTINUE | **Conditional CONTINUE** with hard gates. |

---

### VERDICT: **CONTINUE**

**Justification as rigorous skeptic:**

I would normally ABANDON a system that has never run end-to-end. Zero delivered value is damning. But three factors stay my hand:

1. **The genuine difficulty is real (8/10) and meets the CPU-only constraint.** 35K lines of non-stub algorithmic code implementing novel mathematics that no existing tool provides. The cost algebra, bisimulation metric, and MDP-from-accessibility-tree construction are genuine research contributions. You don't discard this over a config attribute typo.

2. **The core algorithms are verified correct.** 468 tests, 0 failures. Hand-verified Fitts' law, value iteration, softmax policy. The mathematical foundation works.

3. **The fix path is bounded.** The `beta_min` bug is a 1-line fix. The integration failures are concentrated in outer layers that wrap working core modules. The architecture is sound for incremental integration. Estimated 4-6 weeks to a working MVP.

**What would flip this to ABANDON:**
- If the pipeline integration reveals deeper architectural incompatibilities beyond `beta_min`
- If the system cannot beat a trivial element-count-diff baseline on real UI pairs
- If the MDP state space explodes beyond CPU tractability for realistic UIs (500+ elements)
- If the cognitive distance metric produces meaningless or degenerate results on real accessibility trees

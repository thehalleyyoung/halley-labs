# Implementation Evaluation: Bounded-Rational Usability Oracle (proposal_00)

**Evaluator:** Senior Systems Engineer (100K+ LoC systems experience)  
**Method:** Claude Code Agent Teams — Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer, with adversarial critique round and independent verifier signoff  
**Date:** 2026-03-04

---

## Executive Summary

The implementation is a **genuine, architecturally coherent** system with real algorithmic substance — not a stub, wrapper, or generated codebase. It correctly couples information-theoretic bounded-rational decision-making (Ortega & Braun 2013) with MDP bisimulation (Givan/Dean/Greig 2003) and cognitive science models (Fitts, Hick-Hyman, Wickens). However, the 76K LOC claim is misleading, the implementation timed out with 0 polish rounds, and critical gaps remain (non-associative cost algebra under non-zero coupling, triplicated soft VI, incomplete repair module).

---

## Team Process

### Phase 1: Independent Exploration
Three agents independently analyzed the codebase:
- **Auditor**: Quantitative LOC breakdown, test inventory, format parser assessment
- **Skeptic**: Aggressive challenge of every claim (LOC padding, algorithmic triviality, duplication)
- **Synthesizer**: Component-by-component value extraction, replication difficulty, salvageable parts

### Phase 2: Adversarial Critique
Cross-team challenges resolved with code evidence:
- Skeptic's "3K lines" vs Auditor's "35K lines" → resolved via 3-tier classification (~3K novel algorithms + ~8-10K domain engineering + ~8-10K infrastructure)
- "Empty examples" claim by Auditor and Synthesizer → **refuted** (6 executable scripts exist, 59KB total)
- β-bisimulation novelty → "novel integration of known techniques, not novel algorithm"
- Z3 constraints → "individual linearizations are trivial, combined formulation has genuine complexity"

### Phase 3: Verifier Signoff
Independent spot-checks revealed a **critical gap none of the three reviewers caught**: the cost algebra associativity tests only verify the degenerate case (coupling=0, which is pure addition). With ρ>0, the sequential composition is NOT associative.

---

## Verified Facts

| Metric | Value |
|--------|-------|
| Raw lines (library) | 62,549 |
| Raw lines (tests) | 31,387 |
| Logical LOC (library, excl. docstrings/blanks/comments) | ~35,492 |
| Logical LOC (tests) | ~18,364 |
| Substantive code (novel algorithms + domain engineering) | ~11,000–13,000 |
| Novel algorithmic core | ~3,000 |
| Python source files (library) | 179 |
| Subpackages | 24 |
| Test functions | 2,568 |
| Property-based tests (Hypothesis @given) | 146 |
| Integration tests | 204 |
| Format parsers | 7 (HTML, JSON, Chrome DevTools, axe-core, Android, iOS, Windows UIA) |
| Example scripts | 6 |
| HTML fixtures | 4 |
| JSON fixtures | 4 |
| Functions/classes defined | 2,702 |

---

## What's Real

1. **MDP construction from accessibility trees** (`mdp/builder.py`): Genuine state-space enumeration over (focusable-node × task-progress-bitvector) with 8 action types, domain-specific transition costs, and unreachable state pruning.

2. **Partition refinement bisimulation** (`bisimulation/partition.py`): Classical algorithm adapted for bounded-rational policies — softmax policy signatures as the refinement criterion, guaranteed convergence on finite state space.

3. **Compositional cost algebra** (`algebra/sequential.py`, `parallel.py`, `context.py`): 4-tuple (μ, σ², κ, λ) composition with sequential coupling, Wickens MRT parallel interference, and context modulation (fatigue, practice, stress).

4. **Bounded-rational policies** (`policy/value_iteration.py`): Soft Bellman equation with numerically stable log-sum-exp, free energy decomposition, rate-distortion curve tracing.

5. **Cognitive models** (`cognitive/`): Fitts' law (Shannon formulation + steering law), Hick-Hyman (entropy generalization + practice effects), visual search (serial/parallel/guided), working memory decay — all with published parameters and interval uncertainty propagation.

6. **Z3 repair synthesis** (`repair/synthesizer.py`): Real SMT encoding with Boolean mutation selection, dimensional variables, Fitts/Hick constraint encoding, iterative blocking clauses.

7. **Statistical hypothesis testing** (`comparison/hypothesis.py`): Hand-implemented Welch's t-test with correct Satterthwaite DOF, Mann-Whitney U, bootstrap permutation, Cohen's d, multiple testing corrections.

8. **Error bounds** (`comparison/error_bounds.py`): Computed (not hardcoded) abstraction bounds via Givan/Dean/Greig Theorem 3, plus Hoeffding/Chebyshev/CLT sampling bounds.

---

## What's Inflated or Broken

1. **76K LOC claim is inaccurate.** No counting method yields 76,324. Raw lines = 93,906; logical LOC = 53,856; substantive code = ~11-13K.

2. **Cost algebra non-associativity under coupling.** The sequential composition ⊕ with ρ>0 is NOT associative: `√((σ²_a+σ²_b)·σ²_c) ≠ √(σ²_a·(σ²_b+σ²_c))`. Property tests only verify the trivial ρ=0 case. This undermines composability claims for the full cost model.

3. **Triplicated soft value iteration.** Three independent implementations in `bisimulation/cognitive_distance.py`, `policy/value_iteration.py`, and `comparison/paired.py`. Copy-paste artifact under time pressure.

4. **Repair module hardcoded parameters.** `"Improved Label"` and `"Ctrl+Shift+A"` are template values, not Z3-optimized. Uses `z3.Solver()` (satisfiability) not `z3.Optimize()` (optimization).

5. **Chrome DevTools parser extracts only 6/36+ ARIA states.** Cross-platform parsers are mapping layers, not deep API integrations.

6. **0 polish rounds completed.** Implementation timed out, leaving rough edges.

---

## Scores

| Dimension | Score | Justification |
|-----------|:-----:|---------------|
| **Code Quality** | **7** | Clean 24-package architecture, consistent patterns, good typing. Docked for triplicated soft VI, heuristic bisimulation split undocumented, repair hardcoded params, coupling=0 associativity blind spot. |
| **Genuine Difficulty** | **6** | Real mathematical substance in coupling bisimulation with bounded rationality and building MDP from accessibility trees. But core algorithms implement known techniques (textbook VI, textbook Fitts, textbook bisimulation). Novel integration, not novel theory. ~3K lines of genuine algorithmic novelty in ~11-13K of substantive code. Solid systems paper difficulty, not theory paper difficulty. |
| **Value Delivered** | **6** | Working 9-stage pipeline, functional CLI (5 subcommands), 6 example scripts, 7 format parsers. Pipeline runs HTML→verdict end-to-end. Docked for: no live browser integration, incomplete repair, 0 polish rounds, non-associative cost algebra. |
| **Test Coverage** | **7** | 2,568 tests including 146 Hypothesis property tests and 204 integration tests covering all core algorithms. Docked because associativity tests only cover the trivial ρ=0 case (critical gap), ~20% of unit tests are trivial config checks, no negative tests for known limitations. |
| **Format Support** | **6** | HTML parser is genuinely complete with ARIA name computation. JSON, Chrome DevTools, axe-core parsers are functional. Android/iOS/Windows parsers are mapping layers. Docked for: CDP extracts only 6 ARIA states, all parsers require pre-captured snapshots (no live browser), ARIA parser is validation-only. Two common formats (HTML and JSON) work well. |

---

## Genuine Difficulty Assessment

As a senior engineer who has built 100K+ LoC systems:

**What's genuinely hard:**
- Coupling MDP bisimulation with bounded rationality (β-parameterized abstraction) — requires understanding both automata theory and information-theoretic decision-making
- The 9-stage pipeline threading β, cost elements, and error bounds across components
- Encoding HCI cognitive laws into MDP transition costs with domain fidelity
- The rate-distortion characterization of usability (genuinely novel framing)

**What's engineering, not algorithmic difficulty:**
- Multi-format parsing (important but well-trodden)
- CLI, output formatting, visualization
- Test suite (valuable but not innovative)
- Pipeline orchestration with caching

**What's textbook:**
- Value iteration (Puterman 1994)
- Fitts' law, Hick-Hyman law
- Partition refinement (Kanellakis & Smolka 1983)
- Boltzmann/softmax policies

**Replication estimate:** 4–6 person-months for a competent grad student (consensus from adversarial critique).

---

## VERDICT: CONTINUE

**Rationale:** The core artifact — coupling bounded-rational decision theory with MDP bisimulation for automated usability regression detection — is a genuine and well-motivated technical contribution. The substantive codebase (~11-13K lines of algorithms + domain engineering) is correctly implemented and backed by a strong test suite. The architectural foundation supports future development.

**Conditions for continuation:**
1. Fix the associativity gap: document non-associativity with ρ>0 as a known limitation, or redesign the formula
2. Deduplicate soft value iteration into a shared utility
3. Complete repair module (replace hardcoded parameters)
4. Correct LOC claims to honest metrics

**Risk:** Medium. The mathematical core is sound but the coupling-dependent non-associativity is a lurking correctness issue. The 0-polish timeout suggests velocity challenges. The system delivers on the CI/CD regression oracle claim at a prototype level but needs polish for production use.

**Conference fit:** UIST, CHI, ICSE (systems paper), not theory venues. The β-parameterized bisimulation idea and rate-distortion usability framing deserve papers; the implementation needs one more pass.

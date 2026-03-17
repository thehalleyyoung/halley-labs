# Verification Framework — CollusionProof Theory Stage

**Role**: Verification Chair (Independent Assessment)
**Project**: CollusionProof — Certified Algorithmic Audit via Compositional Testing
**Phase**: Theory → Implementation Gate
**Date**: 2026-03-08
**Prior Scores**: Depth check composite 5.6/10 → Final approach composite 6.75/10

---

## Preamble: Verification Philosophy

A false CONTINUE is worse than a false ABANDON. This framework is calibrated conservatively: passing all four gates means the theory is ready for implementation, not that the paper is done. I require convincing evidence that the core mathematical machinery works before authorizing 6–8 months of engineering effort on a 60K LoC system.

The project's strongest structural asset is graceful degradation: Layer 0 alone, with unconditional soundness, is independently publishable. This means the real decision is not binary CONTINUE/ABANDON but "how much of the vision survives contact with proof details?" The gates below are designed to distinguish a viable Layer 0 paper from the full three-tier certificate framework.

---

## 1. Quality Gates for Theory Stage

### Gate 1: Mathematical Coherence

**Pass criteria** — ALL must hold:

| # | Criterion | Verification Method | Failure Mode |
|---|-----------|-------------------|--------------|
| 1.1 | Every definition (collusion premium, bounded-recall automaton, composite null H₀, certificate, deviation oracle) is stated with complete quantifier structure, domain specification, and measurability conditions | Manual inspection of definition blocks in paper.tex | Ambiguous universals; missing measurability; circularity between CP and null hypothesis |
| 1.2 | Every theorem statement has: (a) explicitly stated assumptions, (b) well-typed conclusion, (c) no hidden quantifiers in English prose | Parse each theorem: list assumptions, conclusion, and verify logical form | "For suitable ε" without bound; assumptions smuggled into notation |
| 1.3 | Proof strategies for all CORE theorems (M1-narrow, M1-medium, C3', M8, M6) are plausible: no step requires an unproven lemma stronger than the theorem itself | Dependency graph analysis; check each "by standard arguments" claim against literature | Circular dependency between M1 and M6; C3' proof relying on stochastic extension; M6 requiring axiom completeness not proved |
| 1.4 | The dependency DAG {M8 → narrative} ← {C3' → M6 → certificate} ← {M1 → statistical testing} ← {M7 → power ordering} is acyclic and every edge is justified | Construct explicit dependency matrix | Circular: M1 needs C3' for completeness, C3' references M1 test statistic |
| 1.5 | No definition is self-referential or mutually recursive without explicit fixed-point justification | Trace definition chains | Collusion premium defined via equilibrium; equilibrium defined via collusion premium |

**What kills Gate 1**: Any circular reasoning in the core proof structure. Any theorem whose statement is not well-formed (missing conditions, type errors in mathematical objects). Any proof strategy that requires as a lemma something at least as hard as the theorem.

**What does NOT kill Gate 1**: Incomplete proofs (proof sketches with identified gaps are acceptable if gaps are honestly labeled). Minor notation inconsistencies. Proof strategies that are plausible but not yet verified in detail.

---

### Gate 2: Algorithmic Completeness

**Pass criteria** — ALL must hold:

| # | Criterion | Verification Method | Failure Mode |
|---|-----------|-------------------|--------------|
| 2.1 | Every CORE theorem (M1, C3', M8, M6, M7) maps to at least one algorithm with pseudocode or a precise mathematical procedure | Traceability matrix: theorem → algorithm → subsystem | M8 is "pure existence" with no algorithmic consequence; M6 verification has no concrete checker specification |
| 2.2 | All algorithms specify input/output types, loop invariants or termination arguments, and worst-case complexity | Inspect pseudocode blocks | "Run until convergence" without termination bound; O-notation hiding constants that matter for feasibility |
| 2.3 | Complexity bounds are justified (not merely stated) and consistent with performance targets: >100K rounds/sec throughput, <30 min smoke test | Back-of-envelope calculation: rounds/sec × complexity per round ≤ time budget | T* = O(n²M²/η²) with practical parameters giving T* > 10⁹ while evaluation uses T = 10⁶ |
| 2.4 | End-to-end data flow is coherent: price traces → test statistics → rejection decisions → proof terms → certificate → verification | Trace a concrete example through the pipeline | Statistics computed on different data segments than proof terms reference; certificate format incompatible with checker input |
| 2.5 | The proof checker axiom system is specified: all axiom schemas enumerated, all inference rules listed, soundness argument for each axiom | Count axioms/rules; verify each against domain semantics | "~15 axioms" without enumeration; axiom for transitivity of ≤ on rationals stated but not verified against floating-point source |
| 2.6 | The numerical-to-formal bridge (f64 → rational) is specified with: (a) conversion protocol, (b) conditions under which ordinal relations are preserved, (c) handling of edge cases (subnormals, overflow, cancellation) | Inspect bridge specification | "Exact rational arithmetic" handwave; no treatment of catastrophic cancellation in test statistics |

**What kills Gate 2**: No pseudocode for a CORE algorithm. Complexity bounds inconsistent with stated performance targets by >10×. Proof checker axiom system not enumerated (just "~15 axioms" without listing them). No specification of the f64-to-rational bridge.

**What does NOT kill Gate 2**: Pseudocode at a level of detail sufficient for a competent implementer (not line-by-line code). Constants in O-notation when the leading constant doesn't threaten feasibility. M8 having no algorithm (it's an impossibility result — its "algorithm" is the stealth construction, which only needs to exist, not run).

---

### Gate 3: Evaluation Rigor

**Pass criteria** — ALL must hold:

| # | Criterion | Verification Method | Failure Mode |
|---|-----------|-------------------|--------------|
| 3.1 | Every CORE contribution has at least one falsifiable empirical claim | Map contributions → claims → test procedures | "C3' ensures completeness" with no empirical test of completeness on known-collusive scenarios |
| 3.2 | Statistical validation plan specifies: sample sizes, seed counts, significance levels, power calculations or simulation-based power estimates | Inspect evaluation design section | "50 seeds" without power justification; α=0.05 without Bonferroni correction for multiple scenarios |
| 3.3 | Baselines include at least: (a) naive correlation screen, (b) Granger causality, (c) variance ratio test, (d) Calvano et al. replication (if Q-learning scenarios included) | List baselines; verify fair comparison (same data, same α) | Baselines run on different data or with different significance levels; no statistical screen baseline |
| 3.4 | Ground-truth scenarios cover the full claimed operating range: at minimum, Bertrand + Cournot × linear demand × {competitive, boundary, collusive} × {grim-trigger, Q-learning, myopic BR} | Coverage matrix: market model × demand × strategy × ground truth | All scenarios are Q-learning on linear Bertrand; no Cournot; no boundary cases |
| 3.5 | Adversarial red-team scenarios (≥4) are specified with: evasion strategy, expected detection difficulty, and success criterion | Inspect red-team design | Red-team scenarios are just "noisy versions of collusive scenarios" rather than designed-to-evade strategies |
| 3.6 | Threats to validity section addresses at least: (a) synthetic vs. real market gap, (b) evaluation circularity (builder designs both detector and scenarios), (c) bounded-recall assumption coverage, (d) parameter sensitivity | Inspect threats section | Missing threats section; or threats listed without honest assessment of severity |
| 3.7 | Type-I error validation: empirical false positive rate on competitive scenarios with ≥200 seeds per null tier, with confidence intervals | Inspect FPR protocol | Only 10 seeds for competitive scenarios; no confidence interval on empirical α̂ |

**What kills Gate 3**: No falsifiable claims. Fewer than 3 baselines. No adversarial scenarios. Empirical FPR validation with <50 seeds (insufficient to detect α inflation). Evaluation covers <50% of claimed operating range.

**What does NOT kill Gate 3**: Power calculations based on simulation estimates rather than analytic formulas. Threats to validity that are honestly stated even if not fully resolved. Baselines that are simple (correlation, Granger) — simplicity is fine if comparison is fair.

---

### Gate 4: Red-Team Survival

**Pass criteria** — ALL must hold:

| # | Criterion | Verification Method | Failure Mode |
|---|-----------|-------------------|--------------|
| 4.1 | No FATAL flaw identified, where FATAL = "the main theorem is false" or "the system can produce invalid certificates" or "unconditional soundness claim is wrong" | Adversarial review of M1 proof strategy, M6 soundness argument, certificate pipeline | Axiom in proof checker is false in domain; M1 test statistic under H₀ does not have claimed distribution |
| 4.2 | All SEVERE flaws (= "a major claimed property fails for practical parameters") have written mitigation plans with credible timelines | Review SEVERE findings from red-team | C3' bound η/(M·N) is vacuous for M>100; Berry-Esseen constant makes H₀-medium require T>10⁸ |
| 4.3 | Soundness guarantee survives the following adversarial challenges: (a) Can an adversary craft a competitive algorithm that the system falsely certifies as collusive? (b) Can an adversary craft inputs that cause the proof checker to accept an invalid proof? (c) Does the f64→rational bridge preserve soundness under adversarial precision loss? | Construct explicit adversarial scenarios | Floating-point rounding in test statistic computation can flip a ≤ to > in the rational re-verification, causing silent unsoundness |
| 4.4 | Completeness bound for C3' is not vacuous for practical parameters: η/(M·N) ≥ detectable effect size for M ≤ 1000, N ≤ 4, η corresponding to 5% price elevation | Plug in numbers | η = 0.05 × competitive price, M = 1000, N = 4 → Δ_P ≥ 0.05/(4000) = 0.0000125 — possibly below noise floor |
| 4.5 | M1 test has non-trivial power (>0.5) against at least one collusion scenario at sample sizes available in the evaluation (T ≤ 10⁷) for H₀-narrow | Power simulation or analytic bound | Power monotonically increasing in T but only reaches 0.5 at T = 10⁹ |

**What kills Gate 4**: Any FATAL flaw without a fix. Soundness guarantee broken by an explicit adversarial construction. C3' completeness bound vacuous for ALL practical parameters (not just edge cases). M1 power trivial at evaluation sample sizes.

**What does NOT kill Gate 4**: SEVERE flaws with credible mitigations. C3' bound loose for large M (expected — tightening is a stretch goal). M1 power moderate (0.3–0.7) rather than overwhelming.

---

## 2. Scoring Rubric

Each dimension scored 0–10. Half-point increments permitted.

### Proof Correctness (0–10)

| Score | Description |
|-------|-------------|
| 9–10 | All CORE proofs are complete and verified; no gaps remain; would survive expert review |
| 7–8 | All CORE proofs have complete proof sketches; remaining gaps are explicitly identified and each is individually fillable; no structural issues |
| 5–6 | Most CORE proofs have plausible strategies; 1–2 have significant gaps but gaps are bounded (we know what's missing); no reason to believe any core theorem is false |
| 3–4 | Multiple CORE proofs have unresolved gaps; at least one gap is structural (not just a technical lemma); risk of a core theorem being false is non-trivial |
| 1–2 | Core theorems are stated but proof strategies are hand-waving or absent; significant risk of falsehood |
| 0 | No proofs attempted; or a core theorem is known to be false |

### Proof Completeness (0–10)

| Score | Description |
|-------|-------------|
| 9–10 | Full definition-theorem-proof structure; all lemmas stated and proved; ready for journal submission |
| 7–8 | Complete proof sketches for all CORE contributions; key lemmas stated with proof strategies; publishable at a workshop |
| 5–6 | Mix of proof sketches and proof strategies; crown jewels (M1, C3') have at least detailed sketches; supporting theorems may be strategy-only |
| 3–4 | Mostly proof strategies; crown jewels have outlines but key steps are "by a covering argument" without the covering argument |
| 1–2 | Proof strategies only; most steps are "standard" without verification that standard techniques apply |
| 0 | No proof structure beyond theorem statements |

### Algorithm Quality (0–10)

| Score | Description |
|-------|-------------|
| 9–10 | All algorithms have precise pseudocode, proved correctness, tight complexity bounds, and an implementer could write code directly from the specification |
| 7–8 | All core algorithms specified; complexity bounds justified; minor gaps in edge case handling; implementable with reasonable interpretation |
| 5–6 | Core algorithms specified but some lack precision; complexity bounds stated without full justification; implementable but will require design decisions not in spec |
| 3–4 | Algorithms described in prose rather than pseudocode; complexity bounds are guesses; significant implementation ambiguity |
| 1–2 | "Use technique X" without specifying how; no complexity analysis |
| 0 | No algorithmic content |

### Evaluation Design (0–10)

| Score | Description |
|-------|-------------|
| 9–10 | Comprehensive scenario coverage; adequate statistical power; fair baselines; adversarial stress testing; honest threats; a skeptical reviewer would find no methodological objection |
| 7–8 | Good coverage; baselines present; power analysis done; minor gaps (e.g., adversarial scenarios could be stronger); threats section present |
| 5–6 | Reasonable coverage but gaps in operating range; baselines present but comparison could be more fair; threats section present but thin |
| 3–4 | Evaluation plan exists but is incomplete; missing baselines or missing power analysis; no adversarial scenarios |
| 1–2 | "We will run experiments" without specifics |
| 0 | No evaluation plan |

### Novelty (0–10)

| Score | Description |
|-------|-------------|
| 9–10 | Creates a new problem formulation AND proves new theorems with genuinely novel techniques; no prior work in this specific configuration |
| 7–8 | New problem formulation with novel application of known techniques; at least one theorem with a genuinely new proof idea (not just parameter substitution) |
| 5–6 | New application domain for known techniques; competent synthesis but no single element is surprising to an expert |
| 3–4 | Incremental extension of known results; domain application is straightforward |
| 1–2 | Routine application; no new ideas |
| 0 | Replicated prior work |

**My prior estimate for CollusionProof novelty: 7–8.** The composite hypothesis test indexed by game×algorithm pairs (M1) is genuinely new as a formulation. C3' applies automaton-theoretic techniques to game theory in a novel way (Myhill-Nerode ↔ incentive analysis). M8 is "obviously true to experts" but the formalization as a clean dichotomy with C3' is novel packaging. The PCC-for-economics framing is new. Docked from 9 because: individual proof techniques (covering arguments, Berry-Esseen, Mealy machines) are standard; the innovation is in composition and application.

### Best-Paper Potential (0–10)

| Score | Description |
|-------|-------------|
| 9–10 | Defines a new research direction; results will be cited for a decade; artifact sets a standard |
| 7–8 | Strong EC/WINE paper with one "wow" element; likely to generate follow-up work; artifact is impressive |
| 5–6 | Solid EC paper; publishable but not memorable; good craft without breakthrough |
| 3–4 | Borderline accept at a top venue; would need significant strengthening |
| 1–2 | Workshop paper at best |
| 0 | Not publishable |

**Composite score** = weighted average:
- Proof Correctness: weight 0.25 (most critical — wrong proofs invalidate everything)
- Proof Completeness: weight 0.15
- Algorithm Quality: weight 0.20 (implementation feasibility is the next gate)
- Evaluation Design: weight 0.15
- Novelty: weight 0.10
- Best-Paper Potential: weight 0.15

**Threshold**: Composite ≥ 6.0 for CONTINUE. No individual dimension below 4.0.

---

## 3. Known Risks Entering Theory Stage

### Risk R1: C3' Proof for General Deterministic Automata

**Prior assessment**: 85% achievable.
**Nature of risk**: The proof strategy (Mealy machine → product state space → cycle analysis → deviation contradiction) is plausible but Step 6 (the contradiction argument) requires careful handling of the payoff accounting across N players and M rounds. The bound η/(M·N) emerges from a union-bound-style argument that may be loose.

**What I need to see to be satisfied**:
1. A complete proof for the special case of grim-trigger strategies (≤3 states). This is the sanity check — if the proof technique can't handle grim-trigger, it won't handle general automata.
2. A detailed proof sketch for the general case where each step is a well-defined mathematical claim (not "by a counting argument" but "by counting over the M states in player i's automaton, at most M-1 can yield payoff within ε of the on-path payoff, so...").
3. An explicit worked example: 2-player Bertrand with grim-trigger (M=2), computing the exact deviation, the exact payoff drop, and verifying it matches the claimed bound.
4. An honest assessment of whether the η/(M·N) bound is tight or whether a tighter bound is achievable. If the bound is η/(M·N) and M=1000, N=4, η=0.05, the detectable effect is 1.25×10⁻⁵ — I need a statement about whether this is above or below the noise floor in practice.

**Unacceptable**: "The proof follows by standard automaton-theoretic arguments." Proof by analogy to Myhill-Nerode without explicit construction. Claiming C3' for general deterministic automata while only proving it for grim-trigger.

---

### Risk R2: M1 H₀-medium Achievability

**Prior assessment**: 70% achievable.
**Nature of risk**: H₀-medium requires uniform α-control over a parametric family (CES/logit demand × independent no-regret learners). The covering-number argument over the parametric family must yield constants that don't blow up for moderate T. The no-regret learner class is larger than Q-learning, and the cross-firm correlation bound under no-regret (not just Q-learning) is non-trivial.

**What I need to see to be satisfied**:
1. An explicit covering number calculation for the parametric demand family (CES with elasticity in [0.5, 5], logit with coefficient in [0.1, 10]). The ε-cover size as a function of the relevant metric (total variation? Kullback-Leibler?).
2. A cross-firm correlation bound under H₀-medium: if firms run independent no-regret learners against Lipschitz parametric demand, the correlation between price sequences is bounded by f(T, demand_params). What is f?
3. Explicit statement of what "independent no-regret learners" means: independent randomness? Independent state? Independent of opponents' actions conditional on observable history? The definition matters for the correlation bound.
4. An honest assessment of the Berry-Esseen remainder. If the Berry-Esseen correction adds a term C/√T to the correlation bound, what is C for the parametric families in scope? If C > 100, the bound is vacuous for T < 10⁴, which kills all but the longest evaluation scenarios.

**Unacceptable**: "Standard covering argument" without the covering number. Claiming H₀-medium while only proving H₀-narrow. Berry-Esseen "corrections" that are stated without bounding the constant.

**Acceptable fallback**: If H₀-medium cannot be proved with non-vacuous constants, an honest statement that H₀-narrow is the operational null and H₀-medium remains a conjecture with simulation evidence. This weakens the paper but does not kill it (per the tiered hierarchy design).

---

### Risk R3: Proof Checker Axiom Soundness

**Prior assessment**: 80% achievable.
**Nature of risk**: The proof checker kernel (~2,500 LoC, ~15 axiom schemas, ~25 inference rules) must be sound: every axiom must be true in the game-theoretic domain. A single unsound axiom invalidates all certificates. The risk is not in the verification logic (structural induction is standard) but in the axiom-domain interface: does the axiom "if T(σ) > c_α then reject H₀ at level α" correctly encode the statistical guarantee when T(σ) is computed in f64 but the axiom reasons about mathematical reals?

**What I need to see to be satisfied**:
1. All ~15 axiom schemas explicitly listed with domain-semantic justification for each.
2. All ~25 inference rules listed with soundness proof (structural induction step) for each.
3. The f64→rational bridge axiom: whatever axiom connects floating-point computation to the rational arithmetic proof system must be stated with explicit error bounds. "f64 computation x̂ satisfies |x̂ - x| ≤ ε" is not enough; I need the ε to be tracked through the proof and verified by the checker.
4. At least one complete certificate example: a small scenario (2-player, 3-round, grim-trigger) where the full certificate is written out, showing every proof term, and manually verified against the axioms.

**Unacceptable**: "The axiom system is sound by construction." Axiom list described as "approximately 15" without enumeration. No treatment of the floating-point/real gap.

---

### Risk R4: Berry-Esseen Constants for Moderate T

**Prior assessment**: Practical concern, 40% chance of being problematic.
**Nature of risk**: M1's finite-sample correction uses Berry-Esseen-type bounds. The classical Berry-Esseen constant C₀ ≈ 0.4748 applies to iid sums, but the test statistic involves dependent observations (prices in a repeated game are serially correlated). The effective Berry-Esseen constant for dependent sequences can be much larger, potentially rendering the bound vacuous for T < 10⁶.

**What I need to see to be satisfied**:
1. Identification of which Berry-Esseen variant applies (iid? m-dependent? mixing?).
2. If mixing: the mixing rate of price sequences under the claimed null models (Q-learning, no-regret). Is β-mixing with rate O(ρ^t) for some ρ < 1? What is ρ?
3. Explicit numerical evaluation: for T = 10⁵ (smoke test) and T = 10⁶ (standard test), what is the Berry-Esseen remainder term? Is it ≤ 0.01? ≤ 0.1? > 1?
4. If the remainder is large: explicit fallback to permutation-based or bootstrap-based finite-sample inference, with justification of exchangeability/stationarity conditions.

**Unacceptable**: Citing Berry-Esseen theorem without checking the constant. Using the iid Berry-Esseen bound on dependent data without justification. Claiming "finite-sample valid" while hiding a vacuous bound.

---

### Risk R5: Oracle Rewind for Layers 1–2

**Prior assessment**: Mitigated by tiered architecture; Layer 0 is self-contained.
**Nature of risk**: Layers 1–2 require replaying pricing algorithms from arbitrary checkpoints. Real proprietary algorithms may not support this. The mitigation (tiered architecture) means this risk threatens the full vision but not the core contribution.

**What I need to see to be satisfied**:
1. Layer 0 is fully specified and self-contained: no reference to Layers 1–2 in any Layer 0 proof or algorithm.
2. The paper clearly positions Layer 0 as the primary contribution, with Layers 1–2 as extensions for cooperative/regulatory settings.
3. M2 (deviation oracle) and M3 (punishment detection) are clearly marked as Layer 1–2 only, with explicit statements about what access model they require.
4. The evaluation plan includes Layer 0-only results as the primary evaluation, with Layer 1–2 results presented separately.

**Unacceptable**: Layer 0 proofs that "without loss of generality assume Layer 2 access." Evaluation that only reports Layer 2 results. M1 test statistic that requires deviation oracle input.

---

## 4. CONTINUE/ABANDON Decision Framework

### Decision Matrix

```
                    ┌─────────────────────────────────────────────────┐
                    │           COMPOSITE SCORE                       │
                    │   < 4.0    │  4.0–4.9  │  5.0–5.9  │   ≥ 6.0  │
┌───────────────────┼────────────┼───────────┼───────────┼───────────┤
│ Gate 1 FAIL       │  ABANDON   │  ABANDON  │  ABANDON  │  ABANDON  │
│ Gate 2 FAIL       │  ABANDON   │  ABANDON  │  ABANDON  │ COND.CONT │
│ Gate 3 FAIL       │  ABANDON   │  ABANDON  │ COND.CONT │ COND.CONT │
│ Gate 4 FAIL       │  ABANDON   │ COND.CONT │ COND.CONT │ COND.CONT │
│ All Gates PASS    │  ABANDON   │ COND.CONT │  CONTINUE │  CONTINUE │
└───────────────────┴────────────┴───────────┴───────────┴───────────┘
```

### Decision Rules

**CONTINUE** (proceed to implementation):
- All four gates pass.
- No unresolved FATAL flaws.
- Composite score ≥ 6.0.
- No individual dimension score below 4.0.
- At least one crown jewel (M1-narrow or C3'-deterministic) has a proof sketch where every step is a well-defined mathematical claim and no step is harder than the theorem.
- The end-to-end pipeline from price traces to verified certificate is coherently specified.

**CONDITIONAL CONTINUE** (proceed with mandatory remediation):
- Gates 1–3 pass; Gate 4 has unresolved SEVERE flaws with written mitigation plans.
- OR: Gate 2 has minor gaps (e.g., one algorithm lacks pseudocode, but the algorithm is straightforward) with composite ≥ 6.0.
- Composite score ≥ 5.0.
- No individual dimension score below 3.0.
- Crown jewels have at least plausible proof strategies (not just statements).
- **Mandatory**: Remediation items must be specified with owners and deadlines. Implementation may begin on non-blocked subsystems only.
- **Escalation**: If remediation items are not resolved within 4 weeks of CONDITIONAL CONTINUE, auto-escalate to ABANDON review.

**ABANDON** (do not proceed to implementation):
- Any gate fails with composite < 5.0. Mathematical coherence (Gate 1) failure at any score.
- OR: Composite score < 5.0 regardless of gate status.
- OR: A FATAL flaw is identified with no fix.
- OR: Both crown jewels (M1 and C3') have failed or have proof strategies that are not plausible.
- OR: Any individual dimension scores below 2.0.
- **Exception**: If Layer 0 alone (M1-narrow + M7 + M6, without C3') passes all gates with composite ≥ 5.5, may CONDITIONAL CONTINUE on a descoped "sound screening tool" paper. This is a different, smaller paper — acknowledge the descope explicitly.

### Escalation Protocol

If the decision is close (composite 5.5–6.5 with mixed gate results):
1. Convene the full theory team for 90-minute review.
2. Each team member independently scores all dimensions before discussion.
3. Median score (not mean) determines the composite.
4. If median composite is in [5.0, 6.0], defer to the most conservative team member's judgment.
5. Document dissent in the decision record.

---

## 5. Required Theory Artifacts and Standards

### 5.1 For approach.json

**Must contain**:

| Field | Requirement | Failure if absent |
|-------|-------------|-------------------|
| `theorems[]` | Every CORE theorem with: statement, assumptions, proof_status (proved/sketched/strategy/open), dependencies | Cannot assess proof correctness |
| `algorithms[]` | Every algorithm with: pseudocode (or reference to paper.tex), inputs, outputs, complexity_bound, complexity_justification | Cannot assess implementability |
| `complexity_bounds` | For each algorithm: worst-case time and space as explicit functions of parameters (T, N, M, \|P\|), with constants when constants matter for feasibility | "O(T)" is insufficient when the constant determines whether smoke test takes 30 min or 30 hours |
| `implementation_map` | theorem_id → algorithm_id → subsystem_id → estimated LoC | Cannot plan implementation |
| `achievability` | Per-theorem honest probability of successful proof, with justification | Cannot assess risk |
| `kill_conditions` | For each month: what result would trigger ABANDON | Kill gates must be real, not rhetorical |

**Specific requirements by theorem**:

- **M1**: Separate entries for H₀-narrow, H₀-medium, H₀-broad. Each with: test statistic formula, null distribution (exact or asymptotic), rejection threshold, sample complexity (T as a function of α, power, effect size). H₀-narrow must have explicit constants (not just asymptotics). H₀-medium must state the Berry-Esseen constant or acknowledge it as unknown.
- **C3'**: Explicit bound on detectable effect size as a function of M and N. A table of practical parameter values (M ∈ {10, 100, 1000}, N ∈ {2, 3, 4}, η ∈ {0.01, 0.05, 0.10}) showing whether the bound is above noise floor.
- **M8**: The stealth construction must be explicit enough to serve as a red-team scenario for evaluation.
- **M6**: Full axiom list. Full rule list. Soundness argument structure (not just "by induction").
- **M7**: The directed testing ordering must be specified. Holm-Bonferroni application must be worked through for the specific test family.

### 5.2 For paper.tex

**Structure required**:

```
1. Introduction (motivation, contribution summary, related work positioning)
2. Model (game model, bounded-recall automata, oracle layers, formal definitions)
3. The Detection Barrier (M8 impossibility + C3' completeness = dichotomy)
4. Composite Hypothesis Testing (M1 — the test)
5. Certificate Framework (M6 verification + M7 directed testing)
6. Evaluation
7. Discussion (threats, limitations, future work)
Appendix: Full proofs
```

**Quality standards by section**:

- **Section 2 (Model)**: Every definition must be self-contained. A reader should be able to state the definitions without reference to other papers. The bounded-recall automaton definition must be precise enough to determine, for any given algorithm, whether it is bounded-recall.
- **Section 3 (M8 + C3')**: This is the crown jewel section. C3' must have at least a detailed proof sketch (every step is a named lemma or a claim that can be verified in ≤1 page). M8 proof should be complete (it's the easier result). The dichotomy should be stated as a single, quotable theorem.
- **Section 4 (M1)**: The test statistic must be fully specified. The null distribution under H₀-narrow must be derived (not just stated). H₀-medium can be a proof sketch. H₀-broad can be a conjecture with discussion.
- **Section 5 (M6 + M7)**: The axiom system must be in the paper (not just "see artifact"). The soundness theorem must have a proof sketch. M7 application of Holm-Bonferroni is standard but must be instantiated for the specific test family.
- **Section 6 (Evaluation)**: Must include the evaluation design with enough detail that a reviewer can assess whether the experiments would be convincing. Pre-registration style: state hypotheses, test procedures, and success criteria before presenting (hypothetical) results.
- **Section 7 (Discussion)**: Must include honest limitations. Specifically: (a) bounded-recall assumption — what algorithms are NOT covered? (b) synthetic markets — gap to real markets. (c) regulatory deployment — practical barriers. (d) Layer 0 vs. Layers 1–2 gap.

### 5.3 For proofs/

If a separate proofs directory exists:

- Each CORE theorem gets its own file.
- Each file contains: theorem statement (verbatim from paper.tex), proof status, proof or proof sketch, identified gaps (if any), dependencies on other theorems/lemmas.
- Gaps are labeled with estimated difficulty (easy/medium/hard) and estimated time to close.

---

## 6. Precedent Comparison

### 6.1 Roughgarden's Smoothness Framework (EC 2009, Best Paper)

**What it achieved**: Unified price-of-anarchy bounds across auction formats via a single algebraic condition (smoothness). Replaced ad-hoc analyses with a modular framework.

**Theoretical rigor required**:
- Clean, general definitions (smoothness condition, robust price of anarchy).
- Complete proofs for the core theorem (smooth games → PoA bound) and all applications.
- Tight examples showing the bounds are not improvable.
- No computational artifact — pure theory paper.

**Where CollusionProof stands relative to this bar**:
- **Strengths**: CollusionProof also proposes a unifying framework (composite test + certificate). The dichotomy (M8 + C3') has a similar "this is the right way to think about it" quality. The PCC artifact goes beyond what Roughgarden needed to provide.
- **Weaknesses**: Roughgarden's smoothness condition is a single, elegant algebraic condition. CollusionProof's framework is more complex (multiple tiers, multiple theorems, proof checker). The proofs for smoothness are 2–3 pages each; CollusionProof's proofs are more involved. Roughgarden's framework had essentially zero risk of a proof being wrong — the results were tight and clean. CollusionProof has meaningful proof risk (C3' at 85%, M1-medium at 70%).
- **Gap**: CollusionProof needs to match Roughgarden on definitional clarity and proof cleanliness, even if the proofs are longer. The dichotomy theorem (M8 + C3') should be stated with the same crispness as the smoothness condition. Current risk: the framework is too busy (too many moving parts) for a reviewer to see the forest.

**Calibration**: To match EC best-paper level, CollusionProof needs the dichotomy theorem to be clean, quotable, and true. If it achieves this, the extra complexity of the artifact is a feature (demonstrates practicality), not a bug. If the dichotomy is muddled or C3' has gaps, the paper drops from best-paper contention to solid-accept territory.

### 6.2 Necula's Proof-Carrying Code (POPL 1997)

**What it achieved**: Introduced proof-carrying code — programs ship with machine-checkable proofs of safety properties. The proof checker is small and trustworthy; the proof generator can be arbitrarily complex.

**Theoretical rigor required**:
- Formal definition of the safety policy (type safety for a specific type system).
- Soundness theorem: if the checker accepts, the program satisfies the safety policy.
- Complete proof of soundness via structural induction on proof terms.
- Working implementation demonstrating end-to-end feasibility.
- Small trusted computing base (the checker).

**Where CollusionProof stands relative to this bar**:
- **Strengths**: CollusionProof explicitly adopts the PCC paradigm. The "small trusted kernel" design (≤2,500 LoC) mirrors Necula's approach. The phantom-type isolation and dual-path verification are engineering innovations in the PCC tradition.
- **Weaknesses**: Necula's safety policy was type safety — a well-understood, syntactically checkable property. CollusionProof's safety policy is "this statistical conclusion is valid" — a semantic property requiring domain axioms. The soundness argument is correspondingly harder. Necula could fall back on decades of type theory; CollusionProof must invent the axiom system from scratch.
- **Gap**: The axiom system is the critical delta. Necula's type system was well-studied; CollusionProof's game-theoretic axioms are novel. This means the soundness argument must be more carefully constructed. Risk: axiom unsoundness is the equivalent of a type system bug in Necula's framework — it invalidates everything.

**Calibration**: CollusionProof is not expected to match Necula's level of formal rigor (POPL standards for PL formalism are higher than EC standards for economic formalism). But the soundness argument for the proof checker must be convincing to a game theorist, not just a PL researcher. The axiom-domain interface is the crux: each axiom must be obviously true to an economist who reads it.

### 6.3 Calvano et al.'s Q-Learning Collusion (AER 2020)

**What it achieved**: Demonstrated that independent Q-learning algorithms can learn to collude in repeated pricing games, with supra-competitive prices and punishment strategies emerging from self-play.

**Theoretical rigor required**:
- Clean experimental design with controlled parameters.
- Statistical analysis of outcomes (prices, profits, punishment patterns).
- Robustness checks across parameter variations.
- Careful framing: "Q-learning can collude" not "algorithms always collude."
- Minimal formal theory — this is primarily an empirical paper.

**Where CollusionProof stands relative to this bar**:
- **Strengths**: CollusionProof provides what Calvano et al. conspicuously lack: formal error bounds, a definition of collusion with mathematical content, a falsifiable detection methodology, coverage beyond Q-learning. CollusionProof is the methodological infrastructure that Calvano-style empirical work needs but doesn't have.
- **Weaknesses**: Calvano et al. study a real phenomenon (Q-learning collusion is observed); CollusionProof proposes a methodology (collusion detection is a tool). Calvano's contribution is "this surprising thing happens"; CollusionProof's contribution is "here's how to detect it rigorously." The latter is less viscerally surprising, even if methodologically superior.
- **Gap**: CollusionProof should explicitly position itself as the formalization that makes Calvano-style claims rigorous. The evaluation should include Calvano replication as a scenario: can CollusionProof certify what Calvano et al. informally observed? If yes, that's a powerful demonstration.

**Calibration**: CollusionProof's theoretical bar is much higher than Calvano et al. (which had essentially no formal theory). The comparison is about venue expectations: AER accepts impactful empirical findings with minimal formalism; EC expects formal contributions with empirical validation. CollusionProof targets EC standards and should not benchmark against AER-style empirical work for proof rigor. The relevant comparison is to EC theory papers, where Roughgarden (above) is the gold standard.

### Summary Comparison Matrix

| Dimension | Roughgarden (EC '09) | Necula (POPL '97) | Calvano (AER '20) | CollusionProof (target) |
|-----------|---------------------|-------------------|-------------------|------------------------|
| Proof completeness | Complete | Complete | N/A (empirical) | Sketches + 1–2 complete |
| Definitional clarity | Exemplary | Exemplary | Informal | Must achieve: clean |
| Artifact | None | Working impl | Simulation code | Working impl + certificates |
| Novelty type | Framework | Paradigm | Empirical finding | Framework + paradigm adaptation |
| Risk of proof error | ~0% | ~0% | N/A | ~15–25% (C3' or M1-medium) |
| Best-paper quality | Yes (won) | Yes (won) | Yes (very high citations) | Possible (15–25%) |
| Venue standard for theory | High (EC) | Very high (POPL) | Low (AER empirical) | High (EC) |

**Bottom line**: CollusionProof's theoretical ambition is between Roughgarden and Calvano. It aims for framework-level contribution (like Roughgarden) with a working artifact (like Necula) in a domain that currently only has empirical observations (like Calvano). The risk profile (15–25% of a core proof being wrong) is higher than either Roughgarden or Necula had, but the payoff (first formal methodology for collusion detection) justifies the risk if the proofs land.

---

## 7. Pre-Mortem: What Would Make Me Vote ABANDON

I will vote ABANDON if I see any of the following in the final theory output:

1. **C3' proof hand-waves the payoff accounting.** If Step 6 of the proof strategy ("contradiction with self-enforcement") is stated as "the payoff exceeds what's sustainable, contradiction" without an explicit calculation showing the bound η/(M·N), the proof is not done. An arithmetic sentence, not a paragraph of English, must establish the bound.

2. **M1 test statistic has no null distribution.** If the test statistic S_T is defined but its distribution under H₀-narrow is not derived (even under simplifying assumptions like linear demand with known parameters), there is no test. A test statistic without a null distribution is a heuristic, not a statistical test.

3. **The proof checker axioms are not listed.** If the paper says "the axiom system includes standard statistical axioms" without enumeration, soundness is unverifiable. I need to be able to check each axiom myself.

4. **The evaluation uses only collusive scenarios.** If all ground-truth scenarios are collusive and the paper reports 95% detection rate without reporting the false positive rate on competitive scenarios, the evaluation is scientifically valueless.

5. **Unconditional soundness is actually conditional.** If any Layer 0 proof or algorithm references Layer 1–2 oracle access, the "unconditional" soundness claim is false. This would be a FATAL flaw.

6. **The f64→rational bridge is hand-waved.** If the paper claims "rational arithmetic ensures exact computation" without addressing how f64 computation results enter the proof system and what error is introduced, the certificate guarantee is hollow.

7. **The dichotomy theorem doesn't compose.** If M8 and C3' are proved separately but there's no clean statement that combines them ("bounded recall is necessary and sufficient for detection"), the narrative advantage is lost and the paper becomes two disconnected results.

---

## 8. Verification Timeline

| Week | Activity | Output |
|------|----------|--------|
| Theory Week 1–2 | Crown jewel proof development (C3', M1-narrow) | Proof sketches for review |
| Theory Week 3 | **Verification checkpoint 1**: Review C3' and M1 proof sketches | Gate 1 preliminary assessment |
| Theory Week 4–5 | Supporting theorem proofs (M8, M6, M7) + algorithm specification | Pseudocode + complexity bounds |
| Theory Week 6 | **Verification checkpoint 2**: Full Gate 1–3 assessment | Scoring on all dimensions |
| Theory Week 7–8 | Red-team review + evaluation design + paper draft | Gate 4 assessment |
| Theory Week 8 | **Final verification**: All gates + composite score + CONTINUE/ABANDON decision | Decision document |

At each checkpoint, I will score all available dimensions and flag any trajectory toward ABANDON. The team should treat checkpoint scores as early warnings, not final judgments.

---

## 9. Sign-Off

This framework represents my independent assessment of what constitutes adequate theory for CollusionProof to proceed to implementation. The gates are conservative by design: I would rather delay implementation by 2 weeks to fix a proof gap than discover a FATAL flaw after 4 months of engineering.

The project has genuine potential. The dichotomy narrative (M8 + C3') is clean and powerful. The PCC-for-economics framing is novel. The regulatory timing is excellent. But potential is not proof. I need to see the mathematics before I authorize the engineering.

**Verification Chair initial assessment**: The project enters theory stage with a credible path to CONTINUE. Estimated probability of CONTINUE after theory stage: **65%**. Estimated probability of CONDITIONAL CONTINUE: **20%**. Estimated probability of ABANDON: **15%** (driven primarily by C3' proof failure or M1 null distribution intractability).

---

*Framework version 1.0. Will be updated at each verification checkpoint.*

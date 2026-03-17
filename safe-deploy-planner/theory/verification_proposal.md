# Verification Assessment: SafeStep Theory Stage

**Project:** SafeStep — Verified Deployment Plans with Rollback Safety Envelopes for Multi-Service Clusters  
**Slug:** `safe-deploy-planner`  
**Role:** Verification Chair (independent assessment)  
**Date:** 2026-03-08  
**Inputs:** problem_statement.md, depth_check.md, final_approach.md, approaches.md, approach_debate.md, seed_idea.md  
**Ideation Composite Score:** 6.0/10 (V6/D6/BP5.5/L6.5) — CONDITIONAL CONTINUE

---

## Executive Summary

SafeStep's theory stage must transform a promising but under-proven ideation concept into a publication-ready formal framework. The rollback safety envelope is a genuinely novel operational primitive — that much is established. What is *not* established is whether the surrounding mathematics is sound, whether the complexity claims survive scrutiny, whether the assumptions hold in practice, and whether the end-to-end formal chain from constraint oracle through BMC encoding to envelope computation is watertight enough for a top-venue reviewer to accept.

This document specifies:
1. What the theory must achieve (quality criteria)
2. How it will be scored (framework with rubrics)
3. What specific items will be checked (verification checklist)
4. What would cause an ABANDON recommendation (kill gates)
5. What minimum standard applies per venue tier (quality bars)
6. A preliminary assessment based on ideation documents (pre-assessment)

**The bar is best-paper candidate at SOSP/OSDI.** Mediocre theory that "gets by" is not acceptable. The theory must be genuinely strong where it matters and honestly limited where it is weak.

---

## 1. Theory Quality Criteria

The theory stage deliverables (approach.json + paper.tex >50KB) must satisfy six non-negotiable criteria. Each is a necessary condition for CONTINUE; failure on any one triggers a remediation cycle or ABANDON.

### 1.1 Soundness

**Standard:** Every proof must be correct to the level expected of a top-tier systems conference paper — meaning a knowledgeable reviewer working through the proof for 30–60 minutes would not find a gap, incorrect step, or unjustified assumption.

**Specific requirements:**
- Theorem statements must be precise: all quantifiers explicit, all conditions stated, all objects well-defined.
- Proof sketches in the main body must be convincing enough that a reviewer believes the full proof exists and is correct. "Convincing" means: the proof strategy is clear, the key insight is identified, and the hard step is addressed (not hand-waved).
- Full proofs in the appendix must be complete — no "straightforward but tedious" gaps in load-bearing arguments.
- The Monotone Sufficiency proof must handle the multi-service constraint chain case explicitly (flagged by the Math Assessor: "the exchange argument must show that downward closure propagates through multi-service constraint chains, not just pairwise constraints").
- The CEGAR Loop Soundness proof must establish a termination bound that is tighter than the vacuous 2^R (number of resource variables).
- Every proof must identify its dependencies: which assumptions, which prior results, which definitions.

**What constitutes failure:**
- A load-bearing theorem with a gap that cannot be patched within 1 page of additional argument.
- A complexity bound that is off by more than a polynomial factor.
- A soundness direction error (proving safety when unsafety was needed, or vice versa) — as was found in Approach B's Compositional Envelope Soundness.
- A circular argument (using a result to prove a lemma that the result depends on).

### 1.2 Completeness

**Standard:** Every claim made in the paper must have formal backing in the theory. No load-bearing result is missing, and no critical gap is left for "future work" that would undermine the paper's central claims.

**Specific requirements:**
- The chain from oracle → encoding → BMC solving → plan extraction → envelope computation → witness generation must be formally justified at every link.
- The sensitivity theorem for non-interval fraction f must be formally stated and proved (flagged by Math Assessor as described in prose but not given a numbered statement).
- The relationship between downward-closure violations and completeness bound degradation must be formally characterized.
- The backward reachability computation for envelope must have its own correctness argument (not just "it's the same as forward reachability but reversed").
- The replica symmetry reduction must be proved correct — the pair (old_count, new_count) representation must be shown to be a faithful abstraction of the full replica state.

**What constitutes failure:**
- A missing theorem that is needed for the end-to-end soundness chain.
- A claim in the evaluation section that cannot be justified by any theorem (e.g., claiming completeness without the completeness bound).
- A "the proof is similar to..." that hides a genuinely different argument.

### 1.3 Honesty

**Standard:** Every limitation is acknowledged prominently. The theory does not overclaim. Assumptions are stated as assumptions, not theorems. Empirical findings are not dressed up as mathematical results.

**Specific requirements:**
- The oracle limitation must appear in the abstract, introduction, and a dedicated limitations section — not buried in a footnote.
- "Structurally verified relative to modeled API contracts" must be the consistent framing — never "formally verified" or "provably safe" without qualification.
- The downward-closure condition must be presented as an assumption with empirical support, not a universal truth. Violation rates and their impact must be reported.
- The 92% interval-structure finding must be presented as an empirical observation with methodology, not as a mathematical property of real-world systems.
- The treewidth DP must be framed as a narrow fast path (tw ≤ 3, L ≤ 15), not a general tractability result.
- The BMC Completeness Bound must be presented as a corollary of Monotone Sufficiency, not an independent theorem.
- Theorems must be classified honestly: "2 key theorems (Monotone Sufficiency, Interval Encoding), 1 integration theorem (CEGAR Soundness), 1 narrow optimization (Treewidth DP), 1 stolen result (Adversary Budget Bound), 2 supporting propositions."

**What constitutes failure:**
- The abstract claims "provably safe deployment plans" without oracle qualification.
- Treewidth DP is presented as feasible at tw = 5 (contradicted by the depth check's math).
- The paper implies SafeStep catches behavioral incompatibilities.
- LoC estimates revert to the inflated 155K figure.

### 1.4 Novelty

**Standard:** The paper must clearly differentiate from prior work — specifically Aeolus/Zephyrus (configuration synthesis), the SDN consistent-update literature (Reitblatt et al., McClurg et al.), and generic BMC/CEGAR frameworks. The differentiation must be substantive, not just "we applied it to a different domain."

**Specific requirements:**
- The rollback safety envelope must be formally defined as a new concept and shown to be distinct from any existing construct in the deployment, model-checking, or SDN literature.
- The monotone sufficiency result must be shown to be novel in the deployment context — the exchange argument technique is standard, but the specific property (downward-closed compatibility implies monotone sufficiency) and its implications for BMC completeness bounds must be new.
- The interval encoding compression must be differentiated from generic SAT encoding optimizations — the domain-specific insight (compatibility predicates have interval structure) is the novelty, not the binary encoding technique.
- The comparison with SDN consistent updates must address the structural parallel head-on: same meta-problem (safe transitions in product graphs), different constraint domains, and the rollback analysis as genuinely new.
- The paper must not overclaim novelty for borrowed techniques: BMC, CEGAR, treewidth DP, and incremental SAT solving are all established. The combination and domain-specific reductions are new.

**What constitutes failure:**
- A reviewer can reduce the contribution to "McClurg et al. (PLDI 2015) for Kubernetes" and the paper has no convincing rebuttal.
- The rollback safety envelope turns out to be equivalent to an existing concept in a different community (e.g., "safe region" in control theory, "reachable set" in verification).
- The monotone sufficiency result is a special case of a known result in lattice theory or order theory that the paper fails to cite.

### 1.5 Implementability

**Standard:** Every theorem must map to a concrete module in the implementation. The theory must be constructive — not just proving existence, but providing algorithms. The gap between theorem and code must be bridgeable by a competent systems programmer.

**Specific requirements:**
- Monotone Sufficiency → monotone transition constraint in the BMC encoder.
- Interval Encoding → binary-encoded interval predicate generator.
- BMC Completeness Bound → termination condition in the BMC loop.
- Treewidth DP → tree-decomposition-based DP solver with clear bag-computation rules.
- CEGAR Loop Soundness → CEGAR controller with blocking clause generation.
- Adversary Budget Bound → k-robustness enumeration scope calculator.
- Envelope computation → bidirectional reachability checker with incremental SAT.
- Every algorithm must come with a pseudocode listing or a sufficiently precise description that pseudocode is straightforward to derive.
- Complexity bounds must translate to concrete performance expectations at target parameters (n=50, L=20, k=200).

**What constitutes failure:**
- A theorem that is correct but non-constructive (proves a plan exists without providing an algorithm to find it).
- A complexity bound that doesn't match the claimed feasibility (e.g., claiming O(n · L^{2(w+1)}) but the constant factors make it infeasible at target parameters).
- An algorithm description that is ambiguous enough to admit multiple implementations with different correctness properties.

### 1.6 Evaluation Readiness

**Standard:** Every formal claim must be falsifiable via experiment. The theory must produce testable predictions, and the evaluation plan must be designed to test them.

**Specific requirements:**
- Monotone Sufficiency predicts that monotone plans are never longer than non-monotone plans (testable on synthetic instances).
- Interval Encoding predicts O(n² · log² L · k) clause count scaling (testable by measuring encoding size).
- BMC Completeness Bound predicts that k* = Σᵢ(goalᵢ − startᵢ) is sufficient (testable by verifying that no plan exists at k* + 1 that didn't exist at k*).
- Treewidth DP predicts a phase transition at tw ≈ 3 where DP becomes faster than BMC (testable by measuring solve time vs. treewidth).
- Adversary Budget Bound predicts that k derived from oracle confidence provides α-level coverage (testable by injecting oracle errors at calibrated rates).
- The sensitivity theorem predicts encoding blowup at specific non-interval fractions f (testable by varying f in synthetic benchmarks).
- The envelope computation must produce verifiable results: every state marked "safe for rollback" must actually admit a backward path (testable by explicit backward search on small instances).

**What constitutes failure:**
- A claim that cannot be tested on laptop hardware within the evaluation time budget.
- A prediction that is trivially true (e.g., "monotone plans exist" — of course they do; the question is whether they are *optimal*).
- An evaluation design that cannot falsify the claims (e.g., success criterion that is guaranteed to pass regardless of system behavior).

---

## 2. Scoring Framework

### 2.1 Viability (V): 1–10

Measures: Will this produce a publishable paper at a top venue?

| Score | Criteria |
|-------|----------|
| 1–2 | Fundamental conceptual flaw. The problem formulation is wrong or the approach cannot work in principle. |
| 3–4 | The approach might work but the theory has serious gaps. Multiple load-bearing theorems are incorrect or missing. Publication unlikely at any venue. |
| 5 | The theory is mostly correct but incomplete. Key results are present but some proofs have gaps. Publication possible at a workshop or lower-tier venue. |
| 6 | The theory is correct and complete for the core results. Some secondary results are sketched rather than proved. Publishable at a strong conference (EuroSys, NSDI, ICSE). |
| 7 | The theory is tight. All results are proved, assumptions are justified, limitations are acknowledged. Competitive for SOSP/OSDI. |
| 8 | The theory makes a genuine intellectual contribution beyond the specific application. Likely acceptance at SOSP/OSDI. |
| 9 | The theory introduces a concept or technique that will be adopted by others. Strong best-paper candidate. |
| 10 | Landmark contribution. Defines a new subfield or resolves a long-standing open question. |

**Current ideation assessment: V6.** The theory must reach V7+ for SOSP/OSDI viability.

### 2.2 Difficulty (D): 1–10

Measures: How hard is the intellectual content of the theory?

| Score | Criteria |
|-------|----------|
| 1–2 | Trivial formalization. Any graduate student could produce this in a week. |
| 3–4 | Standard application of known techniques. The proofs are correct but routine. No intellectual surprise. |
| 5 | Non-trivial combination of known techniques. At least one proof requires genuine insight. |
| 6 | Multiple non-trivial results that interact. The composition of techniques is the difficulty. |
| 7 | At least one result that would be publishable independently in a theory workshop. The proof technique has broader applicability. |
| 8 | Multiple independently publishable results. The theory advances the state of the art for the specific technique, not just the application domain. |
| 9 | Deep new technique with applications beyond this paper. |
| 10 | Major technical achievement. The proof resolves a recognized hard problem. |

**Current ideation assessment: D6.** The theory must reach D6+ (maintain or improve).

### 2.3 Best Paper Potential (BP): Percentage

Measures: Probability of best-paper award at the target venue, conditional on acceptance.

| Range | Criteria |
|-------|----------|
| 0–3% | Correct and publishable but incremental. No conceptual surprise. |
| 3–8% | Solid contribution. Reviewers respect the work but don't champion it. |
| 8–15% | Strong contribution with at least one "wow" element — a new concept, a surprising result, or a killer evaluation finding. |
| 15–25% | Outstanding contribution. The paper changes how the community thinks about the problem. Multiple reviewers champion it. |
| 25%+ | Generational contribution. Rarely assigned; reserved for papers that define their area. |

**Factors that increase BP for SafeStep:**
- The rollback safety envelope becomes "vocabulary" — a concept reviewers use in future discussions (+5%)
- The evaluation discovers a previously-unknown unsafe rollback state in DeathStarBench (+3–5%)
- The treewidth phase-transition curve validates the theoretical tractability boundary (+2%)
- The oracle validation experiment yields a clean, convincing result (+2%)

**Factors that decrease BP:**
- The SDN comparison reveals the contribution is primarily a domain translation (−5%)
- The oracle coverage is in the ambiguous 40–60% range, muddying the value narrative (−3%)
- The proofs are correct but entirely standard, with no surprising step (−2%)
- The evaluation is synthetic-heavy with no real-world deployment (−3%)

**Current ideation assessment: BP 8–12%.** The theory stage determines whether this reaches the high or low end.

### 2.4 Laptop Feasibility (L): 1–10

Measures: Can the theory's algorithms be implemented and evaluated on standard laptop hardware (8-core, 16GB RAM)?

| Score | Criteria |
|-------|----------|
| 1–2 | The algorithms require supercomputer resources or specialized hardware. |
| 3–4 | The algorithms technically fit on a laptop but evaluation at target scale requires days of computation. |
| 5 | Core algorithms run on a laptop but some evaluation experiments require overnight runs. |
| 6 | All algorithms run on a laptop. Target-scale evaluation completes in hours. Some large-scale experiments are slow. |
| 7 | All algorithms run on a laptop. Target-scale evaluation completes in minutes to tens of minutes. |
| 8 | All algorithms are efficient on a laptop. The system is practical, not just evaluable. |
| 9 | The system runs interactively on a laptop — operators get results in seconds to low minutes. |
| 10 | The system is lightweight enough for CI/CD integration with negligible overhead. |

**Current ideation assessment: L6.5.** The theory must preserve or improve this.

### 2.5 Feasibility (F): 1–10

Measures: Can the theory be fully realized as a working system within the project timeline?

| Score | Criteria |
|-------|----------|
| 1–2 | The theory requires solving an open problem to implement. |
| 3–4 | The theory is implementable in principle but requires substantial research beyond what's in the paper. |
| 5 | The theory is implementable but significant engineering challenges remain unaddressed. |
| 6 | The theory maps to a clear implementation plan. A few hard engineering problems remain. |
| 7 | The theory is constructive with algorithms for every component. The implementation path is clear. |
| 8 | The theory provides pseudocode and complexity analysis for every algorithm. Implementation is straightforward for a skilled systems programmer. |
| 9 | The theory is essentially a detailed design document with proofs. Implementation follows mechanically. |
| 10 | The theory has been validated by a prototype. |

**Current ideation assessment: F7.5 (from Independent Verifier).** The theory stage must maintain this.

---

## 3. Verification Checklist

### 3.1 Theorem-by-Theorem Verification

#### Theorem 1: Monotone Sufficiency (CRITICAL — the most important theorem)

- [ ] **Statement precision:** Is "downward-closed" defined formally? Does it apply to all pairwise constraints or also to resource constraints? Are multi-party constraints (involving 3+ services simultaneously) excluded or handled?
- [ ] **Proof correctness:** Does the exchange argument handle:
  - (a) Deletion of a downgrade that creates an unsafe intermediate state for a *different* service?
  - (b) Multiple interleaved downgrades across different services?
  - (c) The case where the monotone plan is shorter but passes through a different set of intermediate states?
  - (d) Multi-service constraint chains: if service A downgrades, and service B's safety depends on A's version, does the argument propagate correctly?
- [ ] **Assumptions:** Is downward closure testable on real data? Is the violation rate quantified?
- [ ] **Load-bearing:** Does every subsequent result (BMC Completeness, Treewidth DP, Envelope computation) actually depend on this theorem?
- [ ] **Failure mode:** What happens when downward closure is violated for k pairs? Is the degradation graceful (completeness bound increases by a bounded amount) or catastrophic (all guarantees void)?

#### Theorem 2: Interval Encoding Compression (CRITICAL — the existence condition)

- [ ] **Statement precision:** Is "interval structure" defined formally? Is the O(log|Vᵢ| · log|Vⱼ|) bound per pair per step or total? Is the constant factor stated?
- [ ] **Proof correctness:** Does the binary encoding of interval constraints interact correctly with:
  - (a) The SAT solver's unit propagation?
  - (b) The CEGAR loop's blocking clauses?
  - (c) The incremental assumption mechanism?
- [ ] **Sensitivity theorem:** Is the formal statement of encoding size as a function of non-interval fraction f present and proved? Does it establish the critical threshold of f beyond which encoding becomes infeasible?
- [ ] **Empirical validation:** Is the 92% interval-structure claim accompanied by reproducible methodology? Is the dataset described? Can a reviewer replicate the finding?
- [ ] **BDD fallback:** Is the clause-count contribution of the BDD fallback for non-interval predicates bounded?

#### Corollary: BMC Completeness Bound (IMPORTANT — makes BMC a decision procedure)

- [ ] **Statement precision:** k* = Σᵢ(goalᵢ − startᵢ) — is this with single-step atomic upgrades only? What about multi-step upgrades?
- [ ] **Proof correctness:** Is this actually a straightforward corollary of Monotone Sufficiency, or does it require additional argument about the structure of the version-product graph?
- [ ] **Tightness:** Is k* tight (i.e., do there exist instances where a plan of length exactly k* is needed)?
- [ ] **Presentation:** Is it honestly presented as a corollary, not an independent theorem?

#### Theorem 3: Treewidth Tractability (ENABLING — narrow fast path)

- [ ] **Statement precision:** O(n · L^{2(w+1)} · k*) — is this time or space? Is the O() hiding a constant factor that matters at target parameters?
- [ ] **Proof correctness:** Does the DP computation correctly handle:
  - (a) Bag merging in the tree decomposition?
  - (b) The monotonicity constraint within bags?
  - (c) The completeness argument (no valid plan is missed by the DP)?
- [ ] **Feasibility boundary:** Is the tw ≤ 3, L ≤ 15 boundary explicitly stated and justified by concrete computation?
- [ ] **Honest framing:** Is it presented as a "narrow optimization" rather than a "general tractability result"?
- [ ] **Practical relevance:** Is the empirical claim "median treewidth 3–5 in production clusters" supported by methodology?

#### Theorem 4: CEGAR Loop Soundness (INTEGRATION — critical correctness property)

- [ ] **Statement precision:** What exactly are the two theories? Is the interface between CaDiCaL (propositional) and Z3 (LIA) formally specified?
- [ ] **Proof correctness:**
  - (a) Soundness: if the loop reports a plan, does the plan satisfy *both* compatibility and resource constraints?
  - (b) Refutation-completeness: if the loop reports infeasible, has it actually exhausted all possibilities within the BMC horizon?
  - (c) Termination: is the 2^R bound real, or is there a tighter bound based on the structure of resource constraints?
- [ ] **Blocking clause correctness:** Is the blocking clause generated from a Z3 refutation a valid negation of the candidate, or could it inadvertently block valid plans?
- [ ] **Implementation fidelity:** Does the theorem's assumptions match what CaDiCaL and Z3 actually guarantee via their APIs?

#### Theorem 5: Adversary Budget Bound (STOLEN — principled k-robustness)

- [ ] **Statement precision:** Is the independence assumption stated prominently? Is the α parameter defined? Is the formula for k correct?
- [ ] **Proof correctness:** Is this a standard concentration inequality argument, or does it require non-trivial adaptation to the oracle error model?
- [ ] **Independence limitation:** Is the correlated-error failure mode discussed? Are there realistic scenarios where correlated errors cause the bound to fail?
- [ ] **Practical calibration:** Can pₑ(i,j,v,w) actually be estimated from schema evidence, or is this a theoretical construct with no practical way to instantiate?
- [ ] **Attribution:** Is the provenance from Approach C clearly acknowledged?

#### Proposition A: Problem Characterization (SUPPORTING)

- [ ] Is this clearly labeled as a proposition/observation, not a theorem?
- [ ] Is the reduction from deployment planning to graph reachability precise?
- [ ] Does it correctly establish PSPACE-hardness for the general case?

#### Proposition B: Replica Symmetry Reduction (SUPPORTING)

- [ ] Is the abstraction from L^r to O(L²) proved correct?
- [ ] Does it handle the minimum-m-available constraint correctly?
- [ ] Is it an over-approximation or an exact abstraction? If over-approximation, is the safety direction correct?

### 3.2 End-to-End Soundness Chain

The following chain must hold without gaps:

```
Oracle produces compatibility constraints (with confidence tags)
    → [Oracle accuracy bounds the guarantee strength — explicitly acknowledged]
Interval structure detection classifies constraints
    → [Non-interval fallback handles the ~8% residual]
BMC encoder produces SAT formula
    → [Interval Encoding theorem: O(n²·log²L·k) clause count]
Monotone sufficiency restricts search to monotone paths
    → [Monotone Sufficiency theorem: no plan is missed]
BMC solver finds plan or proves infeasibility at k*
    → [BMC Completeness Bound: k* is sufficient]
CEGAR loop checks resource constraints
    → [CEGAR Soundness: plan satisfies both SAT and LIA constraints]
Envelope computation checks bidirectional reachability
    → [Backward reachability correctness: envelope membership is precise]
Witness extraction via UNSAT core
    → [UNSAT core correctness: witness is a genuine obstruction]
k-robustness check verifies plan under oracle perturbations
    → [Adversary Budget Bound: k is principled given oracle confidence]
```

**Verification question for each link:** If the upstream component produces an incorrect result, does the downstream component detect and report the error, or does it silently propagate?

### 3.3 Assumption Verification

| Assumption | Where Used | Justification Required | Failure Mode |
|-----------|-----------|----------------------|--------------|
| Downward closure of compatibility | Monotone Sufficiency, BMC Completeness | Empirical: violation rate in dataset; Formal: graceful degradation characterization | Completeness bound weakens; non-monotone paths needed for violating pairs |
| Interval structure of ≥92% of predicates | Interval Encoding feasibility | Empirical: methodology for 847 open-source projects | Encoding size blows up; solver may time out |
| Treewidth 3–5 of production dependency graphs | Treewidth DP applicability | Empirical: methodology for treewidth measurement | DP fast path inapplicable; SAT/BMC is the fallback |
| Atomic sequential upgrades | All theorems | Formal: argument that sequential model is conservative overapproximation of concurrent reality | If concurrent execution can reach states unreachable by sequential interleavings, the model misses failure modes |
| Binary pairwise compatibility | All theorems | Formal: multi-party constraints decompose to pairwise | If 3-service simultaneous interactions exist, the model is incomplete |
| Independent oracle errors | Adversary Budget Bound | Formal: discussion of correlated-error scenarios | k may be insufficient under correlated errors |
| SAT solver correctness (CaDiCaL, Z3) | All computational results | Industry-standard solvers; no verification needed | Extremely low risk but non-zero |

### 3.4 Complexity Bound Verification

| Claim | Formula | At n=50, L=20, k=200 | Feasible? |
|-------|---------|----------------------|-----------|
| Encoding size per step | O(n² · log² L) | 50² × (log₂20)² ≈ 50,000 | Yes |
| Total encoding | O(n² · log² L · k) | 50,000 × 200 ≈ 10M clauses | Yes (CaDiCaL handles 10M+) |
| Non-interval contribution | O(n² · L² · k · f) | 50² × 400 × 200 × 0.08 ≈ 16M | Yes (marginal) |
| Treewidth DP (tw=3) | O(n · L^8 · k) | 50 × 20^8 × 200 ≈ 2.6×10¹² | Borderline (20+ min) |
| Treewidth DP (tw=5) | O(n · L^12 · k) | 50 × 20^12 × 200 ≈ 4×10¹⁷ | **NO — infeasible** |
| Envelope computation | O(k × SAT_call) | 200 × (seconds) | ~minutes to hours |
| k-robustness | (red_count choose k) × SAT_call | ≤55 × (seconds) | Yes (< 1 min) |
| Schema analysis | O(n · L · formats) | 50 × 20 × 2 = 2000 diffs | Yes (< 30s) |

**Critical check:** The paper must use O(n² · log² L · k) consistently — not the erroneous O(n² · log L) from the original problem statement.

---

## 4. Kill Gate Criteria

The following conditions individually or jointly trigger an ABANDON recommendation:

### 4.1 Soundness Failures in Core Theorems (IMMEDIATE KILL)

**Kill if:**
- Monotone Sufficiency has a counterexample under the stated downward-closure condition. This means the key search-space reduction is invalid, and BMC completeness at k* cannot be guaranteed.
- The interval encoding produces an incorrect SAT formula (i.e., a plan that the encoding says is safe but which violates a compatibility constraint). This means the encoding is unsound.
- The CEGAR loop can report "infeasible" when a feasible plan exists (incompleteness) or report a plan that violates resource constraints (unsoundness).
- The envelope computation marks a state as "rollback-safe" when no backward path exists (safety-critical error in the safety-critical feature).

**Mitigation attempt before kill:** If the error is in a secondary result (e.g., Treewidth DP, Adversary Budget Bound), the result can be demoted or removed. If the error is in Monotone Sufficiency or Interval Encoding, the paper's central contribution is invalidated.

### 4.2 Feasibility Analysis Collapse (KILL after mitigation attempt)

**Kill if:**
- The corrected clause count exceeds 100M at target parameters (n=50, L=20, k=200), making SAT solving infeasible on laptop hardware.
- The envelope computation requires >10× the plan-synthesis time with no effective mitigation (sampling-based approximation degrades to checking <5% of plan states, gutting the contribution).
- The non-interval fraction f in real data is >30%, causing encoding blowup.

**Mitigation:** Scale down target parameters (n ≤ 30), use sampling-based envelope, acknowledge partial results. If mitigated paper is still publishable at EuroSys/NSDI level, CONTINUE with downgraded venue target.

### 4.3 Missing Theory That Cannot Be Patched (KILL after patching attempt)

**Kill if:**
- The backward reachability for envelope computation requires a fundamentally different algorithm than forward reachability (not just "reverse the transitions") and no correct algorithm is forthcoming.
- The composition of reductions (monotonicity + interval encoding + CEGAR + treewidth) has an interaction bug where applying one reduction invalidates a property assumed by another.
- The model-reality gap (sequential vs. concurrent) turns out to be unsound in the non-conservative direction: concurrent execution can reach states that the sequential model considers unreachable, causing the envelope to miss failure modes.

**Mitigation:** If the gap is bounded and characterizable, it can be acknowledged as a limitation. If it invalidates the central guarantee, KILL.

### 4.4 Novelty Evaporation (KILL if comprehensive)

**Kill if:**
- A reviewer identifies that the rollback safety envelope is equivalent to "backward reachable set" in standard model checking, and the domain-specific contribution (applying it to deployment orchestration) is deemed insufficient for a top venue.
- The monotone sufficiency result is a direct corollary of a known result in lattice theory or monotone Boolean functions that the paper fails to cite.
- The interval encoding is a standard SAT optimization (e.g., exists in the SMT-LIB benchmark encoding toolkit) and the paper's contribution reduces to "we used standard tools on a new problem."

**Mitigation:** If any single novelty claim evaporates, the paper can lean on the remaining claims. If all three evaporate simultaneously, the paper has no contribution beyond engineering.

### 4.5 Oracle Gate Failure (CONDITIONAL KILL — depends on pivot)

**Kill the practical-tool framing if:**
- The oracle validation experiment shows <40% structural coverage.

**Do NOT kill the project:** Pivot to a theory paper about the envelope concept. The formal framework is valuable even if the oracle is limited. But the paper must lead with the theoretical contribution and treat the implementation as a proof-of-concept.

---

## 5. Quality Bar by Venue

### 5.1 SOSP/OSDI (Top Systems Venue)

**Minimum requirements:**
- All theorems correct with full proofs available (appendix or supplementary).
- At least one result that surprises the reader — a non-obvious consequence of the theory.
- The rollback safety envelope must be compellingly demonstrated as a new concept, not just a new application of reachability analysis.
- The evaluation must include at least one prospective experiment (DeathStarBench or equivalent) demonstrating that SafeStep finds a real issue that no existing tool catches.
- The treewidth phase-transition curve must validate the theoretical prediction.
- Honest, prominent limitations that preempt the strongest reviewer objections.
- The paper must tell a compelling story: operators face this problem daily → the envelope is the right abstraction → BMC with domain-specific reductions makes it tractable → here is evidence it works.

**What distinguishes acceptance from rejection at SOSP/OSDI:**
- Acceptance: "This is a new way of thinking about deployment safety. The theory is clean, the evaluation is convincing, and the concept will be adopted."
- Rejection: "This is competent application of known techniques (BMC, CEGAR) to a new domain. The evaluation is mostly synthetic. The oracle limitation undermines the practical claims."

**Required scores: V ≥ 7, D ≥ 6, BP ≥ 10%, L ≥ 6, F ≥ 7.**

### 5.2 EuroSys/NSDI (Strong Systems Venue)

**Minimum requirements:**
- Core theorems correct with convincing proof sketches.
- The rollback safety envelope is clearly novel and useful.
- The evaluation demonstrates scalability and provides meaningful comparisons against baselines.
- Oracle limitation acknowledged but oracle coverage ≥ 40%.
- The paper is well-written with clear positioning relative to prior work.

**What distinguishes acceptance from rejection at EuroSys/NSDI:**
- Acceptance: "Solid contribution. The envelope concept is interesting, the system works, and the evaluation is adequate."
- Rejection: "The novelty over SDN consistent updates is marginal. The oracle limitation is too severe for the practical claims."

**Required scores: V ≥ 6, D ≥ 5, BP ≥ 5%, L ≥ 6, F ≥ 6.**

### 5.3 ICSE/FSE (Software Engineering Venue)

**Minimum requirements:**
- Formal framework is sound and well-presented.
- The methodology (oracle validation, schema analysis pipeline) is rigorous.
- The evaluation follows SE best practices (threats to validity, replication package).
- The theory connects to software engineering practice (deployment, DevOps, CI/CD).

**What distinguishes acceptance from rejection at ICSE/FSE:**
- Acceptance: "A rigorous formal approach to a real SE problem with good methodology."
- Rejection: "The formal methods content is too heavy for the SE audience. The practical applicability is unclear."

**Required scores: V ≥ 5, D ≥ 4, BP ≥ 3%, L ≥ 5, F ≥ 6.**

---

## 6. Pre-Assessment

Based on the ideation documents (problem_statement.md, depth_check.md, final_approach.md, approaches.md), the following is my preliminary assessment of the theory stage's starting position, risks, and prospects.

### 6.1 Strengths of the Current Theoretical Foundation

**S1: The rollback safety envelope is a genuinely novel concept.** Extensive prior-work analysis in the ideation documents confirms that no existing tool — academic or industrial — computes bidirectional reachability under safety invariants for deployment states. The envelope has the potential to become "permanent vocabulary" in deployment safety, analogous to how linearizability became vocabulary for consistency. This is the single strongest asset.

**S2: The monotone sufficiency result is clean and load-bearing.** The exchange argument under downward closure is elegant: it provides a dramatic search-space collapse (from arbitrary paths to monotone paths) with a tight completeness bound. The proof technique is standard, but the specific application and its consequences for BMC tractability are novel in the deployment context.

**S3: The interval encoding compression solves a real feasibility problem.** Without it, the SAT encoding at target parameters is infeasible (~2B clauses). With it, the encoding is comfortable (~9.3M clauses). The insight that real-world compatibility predicates have interval structure is empirically grounded and converts a domain observation into an asymptotic advantage.

**S4: The ideation process was exceptionally thorough.** The three-approach debate, independent assessments by Math/Difficulty/Skeptic evaluators, depth check with specific flaw identification, and synthesis that addresses most critiques — this is a stronger foundation than most projects start from. The 7 mandatory amendments were mostly addressed in the final approach.

**S5: The honest limitations are already identified.** Oracle accuracy, downward-closure validity, treewidth DP feasibility boundaries, the sequential-vs-concurrent model gap — all are flagged with specific mitigation strategies. This intellectual honesty is a strength, not a weakness.

**S6: The CEGAR loop formalization fills a genuine gap.** The Math Assessor correctly identified that CEGAR soundness was "mentioned but not formalized" and it was promoted to a full theorem. This integration point is where correctness bugs would hide; having a formal statement forces the implementation to be rigorous.

### 6.2 Weaknesses That Must Be Addressed in Theory Stage

**W1: The Monotone Sufficiency proof sketch does not handle multi-service constraint chains.** The Math Assessor explicitly flagged this: "the exchange argument must show that downward closure propagates through multi-service constraint chains, not just pairwise constraints." The current proof sketch handles the easy case (deleting a downgrade of one service while preserving pairwise safety). The hard case is when deleting a downgrade of service A makes a state unsafe for a constraint involving services B and C that transitively depends on A's version. This is the single most important proof obligation in the theory stage.

**W2: The sensitivity theorem for non-interval fraction f is described but not formalized.** The final approach mentions it inline within Theorem 2's section: "Let f be the fraction of service pairs with non-interval compatibility. The total encoding size is O(n² · log² L · k · (1−f) + n² · L² · k · f)." This needs a numbered statement, a proof, and a concrete threshold analysis. The Math Assessor flagged this as a "minor gap that should be fixed during paper-writing" — I disagree; it's a moderate gap because the feasibility analysis depends on it.

**W3: Backward reachability for the envelope lacks an independent correctness argument.** The final approach treats backward reachability as "just forward reachability with reversed transitions." This is true at a high level but requires formal justification: the reversed version-product graph must be well-defined, the safety invariant must be preserved under reversal, and the incremental SAT encoding for backward checks must be shown correct. None of this is currently formalized.

**W4: The model-reality gap is unaddressed.** The depth check (Flaw 6) and the Independent Verifier both flagged this: the sequential, atomic-step model assumes one service upgrades at a time, while real Kubernetes deployments are concurrent and asynchronous. The final approach is "silent on it." The theory stage must either: (a) prove that the sequential model is a conservative overapproximation (safe sequential ⊇ safe concurrent), which requires an argument about interleaving semantics; or (b) acknowledge the limitation formally and characterize the gap.

**W5: The CEGAR termination bound (2^R) is vacuous.** The current statement says the CEGAR loop terminates in "at most 2^R iterations." For practical R (number of resource-constraint variables), 2^R could be astronomical. A useful termination bound requires structural analysis of the resource constraints (e.g., if resource constraints are convex, the CEGAR loop converges in polynomial iterations). Without a tighter bound, the theory cannot guarantee that the CEGAR loop terminates within the claimed wall-clock budget.

**W6: The Adversary Budget Bound's independence assumption may be practically untenable.** The final approach acknowledges this ("if the schema analyzer mishandles a pattern, it misses all instances, creating correlated failures") but has no formal treatment. The theory should at least characterize *how badly* the bound degrades under simple correlation models (e.g., all constraints from the same schema analyzer share a common failure mode).

**W7: The 92% interval-structure claim needs reproducible methodology in the paper.** The problem statement mentions "847 open-source microservice projects on GitHub, using semantic versioning range specifications as the ground-truth compatibility predicate." The theory document must include enough methodology detail that a reviewer could replicate the finding — or at least assess whether the methodology is sound.

### 6.3 Risks That Could Derail the Theory Stage

**R1 (HIGH): Monotone Sufficiency proof failure.** If the multi-service constraint chain case reveals a genuine gap in the exchange argument — e.g., a counterexample where deleting a downgrade creates an unsafe state for a distant constraint — the key reduction is invalid. Impact: the BMC completeness bound fails, the search space remains exponential, and the feasibility analysis collapses. **Mitigation:** The proof should be written first and checked by an independent reviewer before proceeding to other theorems.

**R2 (MEDIUM-HIGH): Envelope computation complexity blowup.** The Skeptic estimated 10–50× overhead for envelope computation relative to plan synthesis. If the incremental SAT approach doesn't achieve the expected amortization (e.g., because backward reachability learned clauses conflict with forward reachability learned clauses), the envelope computation could dominate the entire runtime. Impact: the flagship contribution (the envelope) becomes computationally infeasible at target parameters. **Mitigation:** The sampling-based approximation provides a fallback, but it degrades the contribution.

**R3 (MEDIUM): Downward-closure violation prevalence.** If the oracle validation experiment reveals that >20% of real compatibility predicates violate downward closure, Monotone Sufficiency's practical applicability shrinks. The graceful degradation (non-monotone BMC for violating pairs) increases the completeness bound and solving time. Impact: the performance tier table shifts unfavorably (n ≤ 30 instead of n ≤ 50 in the fast tier). **Mitigation:** Honest reporting; adjust claims to match reality.

**R4 (MEDIUM): SDN reviewer comparison.** If a SOSP/OSDI reviewer is familiar with the SDN consistent-update literature (highly likely), they will immediately see the structural parallel. If the differentiation (different constraint domain, rollback analysis, oracle uncertainty) is not convincingly argued, the paper may be dismissed as a "domain translation." Impact: rejection on novelty grounds. **Mitigation:** The rollback analysis must be presented as a qualitative advance, not just a quantitative one. The SDN literature does not compute rollback safety — this must be hammered home.

**R5 (LOW-MEDIUM): Composition interaction bugs.** The theory claims that monotonicity + interval encoding + CEGAR + treewidth compose cleanly. If there is a subtle interaction (e.g., the CEGAR blocking clauses invalidate the monotonicity restriction, or the treewidth decomposition doesn't respect the interval encoding's variable ordering), the end-to-end chain breaks. Impact: requires re-derivation of some results under composition assumptions. **Mitigation:** A dedicated "Composition Lemma" that establishes the independence of the reductions.

### 6.4 Provisional Score and Verdict

| Dimension | Ideation Score | Theory Pre-Assessment | Target |
|-----------|---------------|----------------------|--------|
| V (Viability) | 6 | 6 (unchanged — theory must prove itself) | ≥ 7 |
| D (Difficulty) | 6 | 6 (unchanged — difficulty is in the proofs) | ≥ 6 |
| BP (Best Paper) | 5.5 (8–12%) | 5.5 (8–12%) | ≥ 10% |
| L (Laptop) | 6.5 | 6.5 (no new information) | ≥ 6 |
| F (Feasibility) | 7.5 | 7 (slight downgrade: W3–W5 add risk) | ≥ 7 |
| **Composite** | **6.0** | **5.8–6.2** (range reflects uncertainty) | **≥ 6.5** |

### Verdict: PROVISIONAL CONTINUE — subject to the following conditions:

1. **Monotone Sufficiency proof (W1):** The multi-service constraint chain case must be formally resolved within the first week of theory work. If a counterexample is found, escalate immediately.

2. **Backward reachability formalization (W3):** An independent correctness argument for the envelope computation must be produced. "Just reverse the transitions" is not sufficient.

3. **Model-reality gap (W4):** Either prove conservative overapproximation or acknowledge the limitation formally. Do not leave this for "future work."

4. **Sensitivity theorem formalization (W2):** The non-interval fraction threshold analysis must be a numbered, proved result — not inline prose.

5. **All proofs must be completed at the appendix level** for the two core theorems (Monotone Sufficiency, Interval Encoding). Proof sketches are acceptable for the remaining results.

6. **The oracle validation experiment results** must be reported honestly regardless of outcome. If <40%, pivot to theory paper framing.

### Final Note

The SafeStep project occupies a genuinely interesting position: a novel operational concept (the rollback safety envelope) supported by a clean technical approach (BMC with domain-specific reductions) applied to a real problem (deployment safety). The ideation process was thorough and intellectually honest. The risks are real but identified and mitigatable.

The theory stage is where this project either proves itself or fails. The difference between a V6 paper (publishable at EuroSys) and a V8 paper (best-paper candidate at SOSP) is the quality of the proofs, the honesty of the limitations, and the sharpness of the evaluation predictions. The concepts are strong. The question is whether the mathematics is watertight.

I will hold the theory to a high standard — not because the project is weak, but because the project is strong enough that cutting corners would waste genuine potential.

---

*Assessment produced by Verification Chair, independent of the ideation team. All evaluations based on documentary evidence in the ideation-stage files. No consultation with the original Visionary, Math Assessor, Difficulty Assessor, Skeptic, Synthesis Editor, or Independent Verifier.*

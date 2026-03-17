# Verification Framework: XR Affordance Verifier — Theory Stage

## Verification Chair: Binding Assessment Protocol

**Project:** Coverage-Certified XR Accessibility Verifier
**Phase:** Theory (pre-implementation, code_loc=0)
**Prior scores:** Depth check 20/40 (V4/D7/BP4/L5) → Final approach 25/40 (V6/D6/P6/F7)
**Crown jewel:** C1 coverage certificate soundness theorem (B+ grade)
**Two-paper strategy:** CAV (coverage certificates) + UIST (accessibility linter)

**Scope of this document:** This framework defines the quality bar the theory deliverables (approach.json, paper.tex) must clear before the project proceeds to implementation. Every criterion is binary — pass or fail — with no partial credit. The theory stage either produces artifacts sufficient to justify 6+ person-months of implementation, or the project is abandoned or radically downscoped.

---

## 1. Quality Criteria for Theory Deliverables

### 1.1 approach.json — Structured Theory Specification

Each criterion below is a hard gate. Failure on any single criterion triggers a REVISE cycle (the deliverable is returned for rework). Failure on ≥3 criteria triggers CONDITIONAL ABANDON (the theory is insufficiently developed to proceed).

| ID | Criterion | Test | Rationale |
|----|-----------|------|-----------|
| **AJ-1** | Every algorithm has pseudocode | For each algorithm entry, verify a pseudocode block exists with explicit input/output types, loop bounds, and termination conditions. No "..." or "as described above." | Pseudocode-free algorithms cannot be reviewed for correctness or estimated for LoC. |
| **AJ-2** | Every algorithm has complexity analysis | For each algorithm, verify worst-case time and space complexity are stated as functions of named parameters (n, k, d, N, etc.) with those parameters defined. Big-O is acceptable; unparameterized claims like "fast" or "tractable" are not. | Implementation feasibility depends on complexity. Without it, we cannot validate D1–D5 gates. |
| **AJ-3** | Every theorem has a complete proof sketch | For each theorem, verify the proof sketch contains: (a) statement of proof technique (induction, contradiction, construction, reduction), (b) identification of the key step, (c) handling of at least one non-trivial case, (d) explicit statement of where known results are invoked. One-sentence "proof sketches" fail. | Hand-waving proofs hide gaps. The depth_check's verification panel (M3 FAIL verdict) demonstrated that imprecise claims conceal fundamental errors. |
| **AJ-4** | Implementation mapping: theorem → module → LoC | Every theorem/algorithm maps to a named implementation module with estimated LoC (±50%). Modules must correspond to the subsystem breakdown in final_approach.md (Table, §Subsystem Breakdown). | Theory disconnected from implementation is academic exercise. The 43–68K LoC estimate must be traceable to specific theoretical results. |
| **AJ-5** | Kill-gate criteria are quantitative and testable | Every gate from the kill-chain (D1–D7, A1–A4) has: (a) a numerical threshold, (b) a measurement procedure (what code to run, on what input, measuring what output), (c) a binary pass/fail criterion. | The depth_check mandated quantitative gates. "Seems to work" is not a criterion. |
| **AJ-6** | Risk assessment per component | Every technical component has: (a) probability of failure (0–100%), (b) impact if it fails (none/minor/moderate/severe/fatal), (c) mitigation strategy, (d) fallback if mitigation fails. Probabilities must be consistent with depth_check estimates (F1–F8). | This project has ~85% compound risk. Without per-component risk tracking, failures are discovered too late. |
| **AJ-7** | Assumptions are explicit and enumerated | A single "Assumptions" section lists every assumption the theory depends on, numbered A1–An. Every theorem references which assumptions it requires by number. | The verification panel found hidden assumptions in M2 (topological) and M3 (abstract vs. concrete reachability). These must be surfaced. |
| **AJ-8** | Constants are computed, not symbolic | Every bound that contains a constant (the C in C·Δ²·L_max^k, the wrapping factor, Lipschitz constants) has either: (a) a computed numerical value for a reference configuration, or (b) an algorithm for computing it with stated complexity. No "for suitable constant C." | The "is this better than just sampling?" question reduces to whether constants are tight enough. Symbolic bounds that hide astronomical constants are useless. |

### 1.2 paper.tex — Publication-Quality LaTeX

| ID | Criterion | Test | Rationale |
|----|-----------|------|-----------|
| **PT-1** | ≥50KB actual content | `wc -c paper.tex` ≥ 50,000. Content audit: no more than 10% of byte count from boilerplate (package imports, formatting macros, bibliography entries). Substantive content (definitions, theorems, proofs, algorithm descriptions, experimental plans) ≥ 45KB. | Size is a proxy for completeness. A coverage-certificate soundness proof, linearization analysis, and full experimental plan cannot fit in <50KB of real content. |
| **PT-2** | Every definition is precise | Grep for weasel words: "suitable," "appropriate," "sufficiently," "natural," "obvious," "standard." Each occurrence must either (a) be immediately followed by a precise specification, or (b) cite a specific definition from a specific reference. Zero unanchored weasel words. | The verification panel's M3 FAIL verdict was caused by imprecise definitions conflating abstract and concrete reachability. |
| **PT-3** | Every theorem states ALL assumptions | For each `\begin{theorem}`, verify the statement includes all required assumptions — not deferred to "see Section X" or "under the assumptions of the model." Each theorem is self-contained: a reader can evaluate truth/falsity from the theorem statement alone, without reading surrounding text. | Theorems with hidden assumptions cannot be verified. A referee encountering "under mild regularity conditions" will (correctly) reject. |
| **PT-4** | Proof completeness follows novelty | For each theorem: (a) if the result is novel (C1, C4, first appearance in literature), the proof is complete — every step justified, every bound derived, every case handled; (b) if the result follows a known pattern (C2, C3, B1), a proof sketch citing the pattern source is acceptable. Novel results with sketch-only proofs are an automatic fail. | C1 is the crown jewel and the sole reason this project exists. A sketch proof of C1 is unacceptable. |
| **PT-5** | Experimental plan has falsifiable hypotheses | The evaluation section contains ≥3 explicitly numbered hypotheses, each with: (a) a quantitative prediction, (b) a measurement methodology, (c) a criterion for falsification (what result would disprove the hypothesis). Hypotheses must address: coverage-certificate tightness vs. Clopper-Pearson (mandatory), Tier 1 false-positive rate (mandatory), and at least one of {scalability, multi-step coverage, Lipschitz violation frequency}. | The depth_check (A3) demands evidence that formal verification outperforms Monte Carlo. Unfalsifiable evaluation plans are not science. |
| **PT-6** | Statistical methodology specified | For every experimental comparison: (a) sample size justification (power analysis or bootstrap estimate), (b) test statistic and significance threshold, (c) correction for multiple comparisons if applicable. For the developer study: (d) IRB plan or exemption justification, (e) effect size estimate. | The UIST paper requires a developer study (15–22 participants). Underpowered studies waste participants' time. |
| **PT-7** | Related work is honest | The related work section: (a) cites Younes & Simmons (statistical model checking), Legay et al. (survey), and positions C1 explicitly against them; (b) cites Clopper-Pearson and explains precisely how coverage certificates improve on it; (c) acknowledges that SE(3) hybrid automata have precedent in geometric control (Tabuada & Pappas); (d) does NOT claim "first" or "novel" for anything that isn't. Every "to the best of our knowledge, we are the first to..." claim is supported by a systematic search description. | The verification panel flagged missing citations (Tabuada & Pappas for M1, SpaceEx AGAR for M4). Reviewer hostility from undercitation is avoidable. |
| **PT-8** | Every definition and theorem is load-bearing | For each `\begin{definition}` and `\begin{theorem}`, there exists a forward reference to either: (a) a system component that depends on it, (b) an evaluation metric it enables, or (c) a proof of another theorem that uses it. Orphan definitions/theorems are deleted. | Paper bloat with unused formalism signals unfocused thinking. The depth_check downgraded from 8 to 2 strong contributions. Only load-bearing results survive. |
| **PT-9** | Notation is consistent and minimal | A notation table exists. Every symbol is defined before first use. No symbol is used with two meanings. The total number of distinct symbols is ≤40 (excluding standard math like ∈, ∀, ∃, ℝ). | Notation overload is the #1 readability complaint at formal methods venues. |
| **PT-10** | The paper passes the "so what?" test for both venues | The introduction clearly states: (a) for CAV — what new verification paradigm or technique this introduces and why existing approaches don't work; (b) for UIST — what concrete developer problem this solves and what improvement it demonstrates. Each framing is independently compelling. | Two-paper strategy requires two independent value propositions. |

---

## 2. Verification Procedures

### Procedure V1: Soundness Check (per theorem)

**Scope:** Applied to every theorem in approach.json and paper.tex. Priority order: C1 (critical), C4 (important), C2 (important), B1 (supporting), C3 (minor).

**Steps:**

1. **Assumption audit.** List every assumption the theorem requires. Cross-check against the master assumption list (AJ-7). Flag any assumption used but not listed.

2. **Statement-proof alignment.** Read the theorem statement. Read the proof conclusion. Verify they are logically identical — not "similar" or "related." Specifically:
   - Does the proof establish the *exact* bound claimed (not a weaker bound)?
   - Does the proof hold for the *exact* quantifiers in the theorem (∀ vs. ∃, "for all scenes" vs. "for this scene")?
   - Does the conclusion match the claimed confidence (1−δ vs. high probability vs. always)?

3. **Hidden assumption search.** For each proof step, ask: "What must be true for this step to work?" If the answer isn't in the assumption list, it's a hidden assumption. Common hiding places:
   - Measure-theoretic regularity (is the accessibility frontier measurable?)
   - Compactness (is the parameter space compact? It should be — ANSUR-II distributions have bounded support)
   - Independence (are sampling regions independent? Stratified sampling introduces dependencies)
   - Continuity (is FK continuous? Yes, but is it Lipschitz, and with what constant?)
   - Finiteness (are SMT queries decidable? Yes for QF_LRA, but timeouts change the logical structure)

4. **Boundary case testing.** Evaluate the theorem at extreme parameter values:
   - ε = 0: Does the theorem claim perfect verification? (It shouldn't — sampling can't prove absence)
   - ε = 1: Is the theorem vacuously true? (It should be — any certificate with ε=1 is trivially sound)
   - δ = 0: Does the theorem claim certainty? (It shouldn't — probabilistic bound)
   - |V| = |Θ|: Everything symbolically verified → ε should be 0 (check!)
   - |V| = 0: No SMT — degenerates to pure sampling → bound should match Clopper-Pearson
   - L → ∞: Lipschitz constant unbounded → ε should go to 1 (certificate becomes vacuous)
   - L = 0: Constant function → certificate should certify immediately with ε = 0
   - k = 1 (single joint): Should be trivially verifiable
   - n_samples = 0: No sampling → ε = 1 unless V = Θ

5. **Red-team attacks.** For each theorem, construct the most adversarial input:
   - *For C1:* A scene where the accessibility frontier is a fractal curve (infinite Lipschitz constant locally). Does the certificate correctly report ε = 1 or detect the violation?
   - *For C1:* A scene where an isolated point in parameter space (measure zero) fails accessibility. The Lipschitz assumption is violated. Does the theory handle this gracefully?
   - *For C2:* A kinematic configuration at a singularity (e.g., elbow fully extended, Jacobian rank-deficient). Does the linearization bound still hold?
   - *For C2:* A 7-joint chain at maximum joint range (±90°). Is Δ_max still >0? What is the numerical value?
   - *For B1:* A chain with alternating revolute axes (maximum wrapping amplification). What is the empirical wrapping factor?

### Procedure V2: Completeness Check

**Question:** Does the theory cover every system component that needs formal justification?

**Checklist:**

| System Component | Required Theory | Status |
|-----------------|-----------------|--------|
| Affine-arithmetic FK | B1 (wrapping factor bound) | Must exist |
| Adaptive stratified sampler | Sampling density theorem within C1 proof | Must exist |
| Frontier-seeding from Tier 1 | Correctness of Tier 1 frontier ⊇ true frontier | Must exist |
| Linearized-kinematics SMT | C2 (linearization envelope) | Must exist |
| SMT timeout-and-skip | Proof that skipping preserves soundness (weakens ε but doesn't invalidate) | Must exist |
| Coverage certificate assembly | C1 (soundness theorem) | Must exist |
| Lipschitz estimation | Proof that local estimation is conservative (or explicit conditions when it isn't) | Must exist |
| Certificate composition for multi-step | Extension of C1 to k-step trajectory space | Must exist for k≤3 |
| Budget allocation | C3 (optimal allocation) | Should exist (non-critical) |
| Completeness gap | C4 (Tier 1 detection bound) | Should exist (for UIST paper) |
| Population-fraction reporting | Chebyshev/ANSUR-II bound correctness | Brief justification sufficient |

**Implicit assumptions to surface:**
- Does the theory assume the Unity scene parser is correct? (It should — parser correctness is an engineering concern, not a theory concern, but the assumption must be stated.)
- Does the theory assume FK evaluation is exact? (It shouldn't — floating-point error must be accounted for, at least by stating that affine arithmetic absorbs it.)
- Does the theory assume ANSUR-II distributions are known exactly? (It shouldn't — distribution uncertainty should propagate into δ.)

### Procedure V3: Consistency Check

**Question:** Do the theorems compose into a coherent system?

**Checks:**

1. **Assumption compatibility.** For each pair of theorems (Ti, Tj) where Tj's proof uses Ti as a lemma:
   - Are Ti's assumptions a subset of Tj's stated assumptions?
   - If Ti's conclusion has caveats (e.g., "except on a set of measure zero"), does Tj account for them?

2. **Complexity-performance alignment.** The kill gates define performance targets. Do the complexity bounds permit meeting them?

   | Gate | Target | Required Complexity | Theory Must Show |
   |------|--------|-------------------|------------------|
   | D1 | Wrapping ≤ 5× on 4-joint chain | B1 bound ≤ 5 for k=4, Δθ=30° | Explicit numerical evaluation |
   | D2 | ε < 0.05, 10 objects, 5 min | C1 + C2 must yield ε < 0.05 with n_samples ≈ 100K and n_smt ≈ 500 | Plug in numbers, verify |
   | D3 | ≥5× improvement over Clopper-Pearson | C1's ε vs. CP ε from same n | Analytical comparison |
   | D4 | <20% Lipschitz violations | Lipschitz boundary detection coverage | Theoretical characterization |
   | D5 | ε < 0.1 for 3-step, 15 min | C1 extended to 21D | Curse-of-dimensionality analysis |

3. **Metric-guarantee alignment.** Do the evaluation metrics measure what the theorems guarantee?
   - C1 guarantees P(undetected bug) ≤ ε. The metric "detection rate" measures fraction of injected bugs found. These are related but NOT identical. The theory must bridge: if detection rate ≥ X on benchmark with Y injected bugs, what does this imply about ε on real scenes? (Answer: nothing directly — but the paper must be honest about this.)
   - B1 guarantees wrapping ≤ f(k). The metric "false-positive rate" depends on wrapping + scene geometry. The theory must specify the relationship (or acknowledge it's empirical).

4. **Inter-tier consistency.** Tier 1 results seed Tier 2. The theory must specify:
   - What Tier 1 output format does Tier 2 expect?
   - Is Tier 1's over-approximation guaranteed to contain the true frontier? (Yes, by interval arithmetic soundness — but state this explicitly.)
   - If Tier 1 reports "green" (definitely reachable), can Tier 2 skip that region? (Only if Tier 1's soundness is proven for the same parameter space.)

### Procedure V4: Load-Bearing Check

**Question:** Is every piece of theory necessary?

**For each definition/theorem, trace three links:**

| Theory Item | System Module | Eval Metric | If Removed, What Breaks? |
|------------|---------------|-------------|--------------------------|
| C1 | Certificate engine (6–10K LoC) | ε bound, ε vs. CP comparison | Entire CAV paper collapses. System degrades to "MC + SMT heuristic" with no formal guarantee. Project reduces to UIST-only. **Critical.** |
| C2 | Linearized-kinematics SMT (5–8K LoC) | SMT query count, Δ_max values | SMT queries lack soundness guarantee. Must fall back to sampling-only in SMT regions, weakening ε. **Important.** |
| C3 | Handoff controller (3–5K LoC) | Budget efficiency (ε per minute) | Use heuristic allocation. Certificate still works, just looser. **Nice-to-have.** |
| C4 | Tier 1 linter + UIST paper | False-positive rate, detection rate | UIST paper lacks theoretical characterization of Tier 1 limits. Empirical results still stand. **Important for UIST, not for CAV.** |
| B1 | Affine-arithmetic FK (5–8K LoC) | Wrapping factor, Tier 1 precision | Cannot predict Tier 1 quality before implementation. Must measure empirically. **Supporting.** |

**Pruning rule:** If a definition or theorem is not traceable to at least one system module AND one evaluation metric, it is deleted from the paper. Theory for theory's sake is not permitted in a project with 85% compound risk.

---

## 3. CONTINUE / ABANDON Decision Framework

### 3.1 CONTINUE Criteria (ALL must hold)

Every item below is a hard requirement. Failure on ANY single item prevents CONTINUE.

| ID | Criterion | Verification Method | Pass Condition |
|----|-----------|-------------------|----------------|
| **CONT-1** | C1 soundness proof is complete with no identified gaps | V1 full soundness check, all 5 steps | Zero gaps. Every proof step justified. All boundary cases pass. All hidden assumptions surfaced and stated. |
| **CONT-2** | C2 linearization bound has explicit constants | Numerical evaluation for reference configuration (7-DOF arm, ANSUR-II 50th percentile, ±30° joint range) | Constants computed: C, Δ_max, and resulting SMT query count within budget (≤1000 queries for 10-object scene in 10 min). |
| **CONT-3** | At least one genuinely novel contribution | Systematic literature search for coverage certificates, parameter-space verification, sampling-symbolic hybrid | No prior work combines stratified sampling bounds with SMT volume elimination into a formal coverage certificate. If Younes-Simmons or Legay et al. already did this (they didn't, but check), novelty collapses. |
| **CONT-4** | ≥3 falsifiable hypotheses with quantitative thresholds | PT-5 check | Three numbered hypotheses with numerical predictions and falsification criteria. |
| **CONT-5** | Implementation mapping feasible | AJ-4 check: total LoC within 43–68K range, no single module >10K novel LoC, no dependency on unavailable library | LoC estimates consistent with final_approach.md. No module exceeds the "hard" threshold without a fallback plan. |
| **CONT-6** | Red-team attacks addressed | V1 step 5: every red-team attack has a response — either mitigated (with proof) or acknowledged (with explicit scope limitation) | Zero unaddressed attacks. "Acknowledged" is acceptable for non-fatal attacks; "mitigated" required for attacks that would make C1 vacuous. |
| **CONT-7** | Analytical ε vs. Clopper-Pearson comparison | C1 proof includes a corollary comparing certificate ε to CP bound from same sample count | The corollary shows ε_cert ≤ ε_CP / g(|V|/|Θ|) where g > 1 when |V| > 0. If g ≤ 1 for all |V|, the certificate adds nothing. |
| **CONT-8** | Lipschitz treatment is rigorous | Lipschitz assumption is either: (a) proven for a characterized class of scenes, or (b) empirically testable with a decision procedure for detecting violations | The theory cannot assume Lipschitz and move on. It must either prove it holds (for which scenes?) or provide a Lipschitz oracle. |

### 3.2 ABANDON Criteria (ANY triggers ABANDON)

If any of these hold after the theory stage is complete, the project is ABANDONED. No conditional continue, no scope reduction — full stop.

| ID | Criterion | How Detected | Why Fatal |
|----|-----------|-------------|-----------|
| **ABAND-1** | C1 proof has a fundamental gap that cannot be fixed | V1 soundness check finds a step that requires an assumption known to be false, or a step that doesn't follow from its premises, and the gap cannot be repaired by adding assumptions or weakening the conclusion without making the theorem vacuous | C1 is the entire CAV paper. Without a sound C1, the project has no formal contribution. |
| **ABAND-2** | Lipschitz assumption is vacuous for realistic scenes | Analysis shows that >50% of XR scene configurations violate Lipschitz at the granularity needed for ε < 0.1, AND the Lipschitz-violation detection procedure has >30% false-negative rate | If Lipschitz fails everywhere and we can't detect when it fails, the certificate's ε is meaningless for the majority of real inputs. |
| **ABAND-3** | ε bounds provably worse than Clopper-Pearson | Analytical or numerical proof that for all |V|/|Θ| < 0.5 (i.e., when less than half the space is SMT-verified), ε_cert > ε_CP from the same sample count | If the certificate is strictly worse than the baseline it claims to improve on, the contribution is negative. |
| **ABAND-4** | Linearization envelope too small for practical coverage | C2 analysis shows Δ_max < 2° per joint for 7-DOF chains at ±30° range, requiring >10⁶ SMT queries to cover the frontier | 10⁶ queries × 100ms = 10⁵ seconds ≈ 28 hours. This exceeds the 10-minute budget by 10⁵×. The sampling-symbolic approach collapses. |
| **ABAND-5** | No genuinely novel contribution after honest assessment | Literature search reveals that the coverage certificate construction (combining stratified bounds with SMT elimination) has already been published — not in the XR domain specifically, but as a general technique | Novelty is the sole justification for the theory paper. Without it, the project is an engineering exercise. The UIST paper might survive, but the CAV paper dies and the project doesn't justify its risk. |

### 3.3 CONDITIONAL CONTINUE Criteria

If any of these hold, the project continues but with mandatory scope reductions specified below.

| ID | Condition | Scope Reduction | Revised Target |
|----|-----------|----------------|----------------|
| **COND-1** | Some red-team attacks unresolved but non-fatal | Document each unresolved attack in the paper's "Limitations" section with explicit future-work plan. Do not claim the certificate handles cases it doesn't. | Paper honestly scoped. No impact on CAV submission. |
| **COND-2** | ε tightness uncertain (analytical bound unclear but not provably bad) | Add empirical ε measurement to evaluation plan as primary result. Analytical bound becomes a "theoretical analysis" section, not the headline. | Paper lead with empirical ε; theory supports but doesn't headline. |
| **COND-3** | Multi-step coverage certificates fail (D5) | Restrict all claims to single-step interactions. Drop k≤3 support from paper. Note as future work. | Reduces novelty modestly but single-step is still useful. |
| **COND-4** | Wrapping factor >5× but ≤10× (D1 partial failure) | Switch to Taylor models for Tier 1. Add LoC and complexity overhead. Recalculate B1. | ~2K additional LoC, 3× slower Tier 1 (still <5s). Manageable. |
| **COND-5** | Developer study cannot be run (IRB delays, recruitment failure) | Proceed with CAV paper only. UIST paper deferred to next cycle. | One paper instead of two. Project still justified if CAV paper is strong. |
| **COND-6** | Clopper-Pearson improvement is 3–5× (below the 5× target but above 1×) | Reframe as "modest but demonstrable improvement." Strengthen the "spatial map" argument (structural advantage, not just quantitative). | Paper survives but is weaker. Best-paper probability drops from 5–8% to 2–4%. |

---

## 4. Scoring Rubric

### 4.1 Theory Soundness (1–10)

| Score | Description | Observable Evidence |
|-------|-------------|-------------------|
| 1–3 | **Fundamental errors or gaps.** C1 proof has a logical error, hidden assumptions are fatal, or the theorem doesn't prove what it claims. | V1 soundness check fails on ≥1 theorem with a gap that cannot be patched. |
| 4 | **Correct but hand-wavy.** Proof direction is right, but key steps invoke "it can be shown that" or "by a standard argument" without specification. | V1 passes but ≥3 steps require expansion before publication. |
| 5 | **Correct, mostly complete, some gaps.** C1 proof is logically sound but missing ≥1 case (e.g., behavior at Lipschitz boundary). Minor theorems have sketch-only proofs. | V1 passes, V2 identifies 1–2 uncovered components, V3 finds minor inconsistencies. |
| 6 | **Solid foundations, acknowledged gaps.** All proofs follow logically. Gaps are identified and documented, not hidden. Constants are symbolic but computation path is clear. | V1–V4 pass. ≤2 COND-* criteria triggered. Referee would request "minor revisions." |
| 7 | **Strong.** All novel proofs complete. Constants computed for reference cases. Red-team attacks addressed. One or two edge cases acknowledged in limitations. | V1–V4 pass cleanly. ≤1 COND-* criteria triggered. Referee says "accept with minor." |
| 8 | **Very strong.** Proofs are tight — no slack in bounds. Boundary cases handled. Composition of theorems demonstrated. | V1–V4 pass with zero issues. All CONT-* met. Paper ready for submission. |
| 9–10 | **Publication-ready.** Proofs are elegant, not just correct. Novel proof techniques that advance the field. A referee learns something new from the proof. | Exceeds expectations. Best-paper candidate on proof quality alone. |

**Minimum for CONTINUE: 6.**

### 4.2 Algorithmic Novelty (1–10)

| Score | Description | Observable Evidence |
|-------|-------------|-------------------|
| 1–3 | **Repackaging.** Coverage certificates are just Clopper-Pearson with an SMT step bolted on. No new insight. | Literature search finds equivalent construction published elsewhere. |
| 4 | **Known techniques, minor twist.** Stratified sampling bounds are textbook. SMT volume elimination is standard. Combining them is a minor engineering contribution. | The "novel" step is a single inequality combining two known bounds. A referee says "nice observation" not "new technique." |
| 5 | **Known techniques, new domain.** The combination is genuinely applied to parameter-space verification for the first time, but the mathematical tools are entirely standard. | A referee familiar with statistical model checking immediately sees the connection and says "solid application." |
| 6 | **Genuine contribution, moderate novelty.** The coverage certificate is a new formal object with properties that don't trivially follow from its components. The Lipschitz-conditioned bound or the volume-elimination integration requires a non-trivial proof step. | A referee takes >1 hour to verify the key proof step. The result is citable. |
| 7 | **Strong contribution.** The coverage certificate framework introduces a proof technique or analytical tool that has clear applicability beyond XR accessibility. Other domains (robotics, medical devices) can immediately use the framework. | A referee says "this should be published" without qualification. |
| 8 | **Significant advance.** The framework solves an open problem in parameter-space verification or introduces a fundamentally new approach to combining statistical and symbolic methods. | A referee says "this changes how I think about the problem." |
| 9–10 | **Breakthrough.** The coverage certificate paradigm becomes a standard tool, cited by >50 papers in 5 years. | Unrealistic for this project. |

**Minimum for CONTINUE: 5. Target: 6–7.**

### 4.3 Evaluation Rigor (1–10)

| Score | Description | Observable Evidence |
|-------|-------------|-------------------|
| 1–3 | **No falsifiable claims.** "We show our tool works on examples" with no baselines, no metrics, no statistical analysis. | No numbered hypotheses. No Clopper-Pearson baseline. No significance tests. |
| 4 | **Weak baselines.** Clopper-Pearson comparison exists but is unfair (different sample counts, different scene complexity). | Baseline comparison is present but a referee would identify confounds. |
| 5 | **Adequate.** Clopper-Pearson comparison is fair. ≥3 metrics defined. Procedural benchmarks with controlled parameters. No real scenes. | A referee says "the evaluation is okay but I'd like to see..." |
| 6 | **Solid.** Fair baselines, ≥5 metrics, real + procedural scenes, power analysis for developer study, falsifiable hypotheses. | A referee has no major evaluation objections. Minor suggestions only. |
| 7 | **Comprehensive.** All of 6, plus ablation studies (sampling only vs. sampling+SMT vs. full certificate), sensitivity analysis (ε vs. budget, ε vs. scene complexity), and cross-validation of Lipschitz estimates. | A referee praises the evaluation as thorough. |
| 8 | **Exemplary.** All of 7, plus multiple baselines (lookup table, MC, MC+CP, random SMT), external scene sources, and reproducibility package. | Best-paper-worthy evaluation. |
| 9–10 | **Gold standard.** Pre-registered hypotheses, independent replication, open data + code. | Unrealistic for initial submission. |

**Minimum for CONTINUE: 5. Target: 6–7.**

### 4.4 Implementation Feasibility (1–10)

| Score | Description | Observable Evidence |
|-------|-------------|-------------------|
| 1–3 | **Theory-practice gap is fatal.** Complexity bounds exceed laptop budget by >100×. Required libraries don't exist. Key algorithm requires unsolved engineering challenges. | D2 gate analytically impossible. Or: required Δ_max < 0.1° making the approach computationally equivalent to brute force. |
| 4 | **Feasible with major unknowns.** Complexity bounds are technically within budget but depend on constants that haven't been computed. ≥2 modules have no clear implementation path. | AJ-4 mapping has gaps. LoC estimate uncertainty >3×. |
| 5 | **Feasible with significant effort.** Complexity bounds fit budget. All modules have implementation paths. But: ≥1 module depends on library integration that hasn't been validated (e.g., Pinocchio + affine arithmetic composition). | AJ-4 complete but with integration risk flags. |
| 6 | **Clear implementation path.** All modules mapped. LoC estimates within ±50%. Library dependencies validated (libraries exist, APIs sufficient, licensing compatible). Two or three hard modules identified with mitigation plans. | AJ-4 clean. A competent systems programmer could start implementation from approach.json. |
| 7 | **Ready to implement.** All of 6, plus: prototype code snippets for the hardest modules. Z3 query structure validated on toy examples. Pinocchio API surface confirmed. | An engineer could produce a prototype in 2 months from approach.json alone. |
| 8–10 | **Implementation-ready with prototypes.** Key algorithms prototyped. Performance validated on representative inputs. | Exceeds theory-stage expectations. |

**Minimum for CONTINUE: 5. Target: 6.**

### 4.5 Best-Paper Potential (1–10)

| Score | Description | Observable Evidence |
|-------|-------------|-------------------|
| 1–3 | **Incremental contribution.** Referee says "solid engineering, but where's the science?" | Coverage certificate is a minor observation, not a framework. |
| 4 | **Publishable.** Referee says "accept" at a B-tier venue. Borderline at A-tier. | The paper would be accepted at TACAS but not CAV. At UIST but not best-paper. |
| 5 | **Solid A-tier contribution.** Referee says "good paper, solid contribution, accept." | 20–30% acceptance probability at CAV. Standard UIST tool paper. |
| 6 | **Strong A-tier contribution.** Referee says "strong work, clear accept." Considered for best-paper longlist. | 30–40% acceptance probability. 5% best-paper probability. |
| 7 | **Very strong.** Referee says "one of the best papers I reviewed this cycle." Best-paper shortlist. | >40% acceptance probability. 10–15% best-paper probability. |
| 8–10 | **Best paper.** The coverage certificate becomes the standard citation for parameter-space verification. | Requires execution well beyond what B+ novelty grade suggests. Aspirational. |

**Minimum for CONTINUE: 4. Target: 5–6.**

### 4.6 Composite Score and Decision

| Composite | Decision |
|-----------|----------|
| ≥30/50 | **Strong CONTINUE.** Proceed to implementation. |
| 25–29/50 | **CONTINUE.** Proceed with COND-* scope reductions as needed. |
| 20–24/50 | **Weak CONTINUE.** Proceed only if: (a) no ABAND-* triggered, (b) ≤2 COND-* triggered, (c) Verification Chair judges that identified issues are fixable in a revision cycle. |
| 15–19/50 | **CONDITIONAL ABANDON.** Theory requires fundamental rework. One revision cycle permitted. If re-scored <20 after revision, ABANDON. |
| <15/50 | **ABANDON.** The theory does not support implementation investment. |

---

## 5. Cross-Teammate Verification Assignments

### 5.1 Assignment Matrix

| Deliverable | Primary Author | Verifier 1 | Verifier 2 | Verification Scope |
|------------|---------------|------------|------------|-------------------|
| C1: Coverage certificate soundness | Formal Methods Lead | Red-Team Reviewer | Verification Chair | Full V1 (all 5 steps). This is the make-or-break proof. |
| C2: Linearization envelope | Algorithm Designer | Formal Methods Lead | Verification Chair | V1 steps 1–4 (soundness). Numerical evaluation of constants for reference configuration. |
| C3: Budget allocation | Algorithm Designer | Formal Methods Lead | — | V1 steps 1–2 (correctness of optimization). V4 (is this load-bearing?). |
| C4: Tier 1 completeness gap | Algorithm Designer | Red-Team Reviewer | Verification Chair | V1 steps 1–4. Connection to UIST paper metrics. |
| B1: Wrapping factor | Algorithm Designer | Formal Methods Lead | — | V1 step 4 (boundary cases). Numerical evaluation for 4-joint and 7-joint chains. |
| Lipschitz treatment | Formal Methods Lead | Red-Team Reviewer | Verification Chair | V1 step 5 (red-team attacks specifically targeting Lipschitz). V2 (is the oracle complete?). |
| Evaluation plan | Empirical Scientist | Red-Team Reviewer | Algorithm Designer | PT-5, PT-6 (hypotheses + statistics). Baseline fairness. |
| Related work | All | Red-Team Reviewer | Verification Chair | PT-7 (honesty check). Citation completeness. |
| Implementation mapping | Algorithm Designer | Empirical Scientist | Verification Chair | AJ-4 (LoC traceability). AJ-6 (risk assessment). |
| Notation + precision | Formal Methods Lead | Red-Team Reviewer | — | PT-2 (weasel words), PT-9 (notation table). |

### 5.2 Verification Protocol

1. **Primary author** produces the deliverable.
2. **Verifier 1** performs the assigned verification procedures and produces a written report with: (a) PASS/FAIL per criterion, (b) specific line-level issues, (c) severity (blocker/major/minor), (d) suggested fixes.
3. **Verifier 2** (if assigned) independently performs verification and produces a second report.
4. **Discrepancies** between Verifier 1 and Verifier 2 are escalated to Verification Chair for resolution.
5. **Primary author** revises based on reports. Changes are re-verified by the original verifier.
6. **Verification Chair** performs final sign-off. No deliverable proceeds without Chair sign-off.

### 5.3 Red-Team Reviewer Special Mandate

The Red-Team Reviewer has an adversarial mandate. Their job is to break the theory:

- **Construct counterexamples** to every theorem. If a counterexample succeeds, the theorem is wrong. If all attempted counterexamples fail, confidence increases.
- **Find the weakest link** in each proof and attack it. The weakest link in C1 is the Lipschitz-to-ε bridge (how the continuous assumption translates to a discrete sampling bound). The weakest link in C2 is the singularity behavior.
- **Challenge novelty.** Perform an independent literature search for: "coverage certificate," "parameter-space verification," "sampling-symbolic verification," "statistical model checking with symbolic," "hybrid testing and verification." If any prior work is within one proof step of C1, the novelty claim must be revised.
- **Test the "so what?" claim.** Independently compute Clopper-Pearson bounds for sample sizes 10K, 100K, 1M. Compare with the predicted ε_cert. If the improvement is <2× for realistic SMT coverage (|V|/|Θ| < 0.3), the practical value of coverage certificates is marginal.

---

## 6. Prior Assessment Delta Analysis

### 6.1 Binding Issues from Depth Check (20/40) and Final Approach (25/40)

The theory must demonstrate measurable progress on every open issue from prior assessments.

| Issue | Source | Required Theory Response | Minimum Acceptable |
|-------|--------|-------------------------|-------------------|
| **"Is this better than just sampling?"** | Depth check F4, Final approach D3 | Analytical corollary comparing ε_cert to ε_CP. Numerical evaluation for n=100K, |V|/|Θ| ∈ {0.1, 0.2, 0.3}. | Demonstrated improvement factor g > 1 for |V|/|Θ| ≥ 0.1. Target: g ≥ 5 for |V|/|Θ| = 0.3. |
| **Lipschitz concern** | Depth check F4, Final approach §Addressing Lipschitz | Formal characterization of Lipschitz-violating scene configurations. Decision procedure for detecting violations. Measure of violation frequency on characterized scenes. | Either: (a) proof that Lipschitz holds for a specified class (e.g., scenes where all activation volumes have positive reach margin), or (b) oracle + frequency bound (<20% violations). |
| **Certificate tightness** | Final approach D2 | Analytical bound on ε as a function of n_samples, |V|/|Θ|, L, d (dimension). Numerical evaluation for target scenario (10 objects, 5 min, d=7). | ε < 0.05 for the target scenario with achievable n_samples and |V|/|Θ|. |
| **Formal verification vs. Monte Carlo distinction** | Depth check A3 | Theoretical analysis of bug classes where MC provably fails but certificates succeed. Candidate: narrow-margin bugs (spatial margin < 1/√n_MC). | At least one identified bug class with: (a) formal proof MC misses it with probability > X%, (b) certificates detect it. |
| **Linearization practicality** | Verification panel M2 concern | C2 with computed Δ_max for 7-DOF chain. Query count estimate within budget. | Δ_max ≥ 5° per joint for typical configurations. Total queries ≤ 1000 for 10-object frontier. |
| **Verification panel M3 FAIL** | Verification panel report | The M3 conflation of abstract/concrete reachability is moot (we don't use zone abstraction), but the lesson applies: every theorem must specify whether it's about the concrete system or the abstract model. | PT-3 compliance (all assumptions stated). |
| **Notation and novelty positioning** | Verification panel M1, M4 | Cite Tabuada & Pappas, SpaceEx AGAR, Younes & Simmons, Legay et al. Position C1 precisely: "these works do X; we do Y; the difference is Z." | PT-7 compliance (honest related work). |

### 6.2 Score Trajectory Expectations

The theory stage should move the composite score from 25/40 to at least 28/40 on the original rubric. Specific axis targets:

| Axis | Ideation Score | Theory Target | How |
|------|---------------|---------------|-----|
| Value (V) | 6 | 6 (hold) | Value doesn't change from theory work. It changes from D7 (demand signal), which is Month 3 implementation. |
| Difficulty (D) | 6 | 7 | Complete proofs demonstrate that the difficulty is real and handled, not just claimed. |
| Potential (P) | 6 | 7 | Tighter ε bounds and the generalizability argument (coverage certificates beyond XR) strengthen the publication story. |
| Feasibility (F) | 7 | 7 (hold) | Feasibility stays the same until code exists. Theory can confirm analytical feasibility but not engineering feasibility. |

**Revised composite target: 27/40.** This reflects that theory work primarily moves D and P, not V or F.

---

## 7. Timeline and Process

### 7.1 Theory Stage Timeline

| Week | Milestone | Gate |
|------|-----------|------|
| 1 | C1 proof draft complete. C2 constants computed. Assumption list finalized. | Internal: AJ-7 check. |
| 2 | C1 verified by Red-Team Reviewer. C4 and B1 drafted. Implementation mapping complete. | Internal: V1 on C1. AJ-4 check. |
| 3 | paper.tex first draft (≥40KB). Evaluation plan with hypotheses. Related work complete. | Internal: PT-5, PT-7 checks. |
| 4 | Full verification cycle: V1–V4 on all theorems. Red-team attacks conducted. | CONTINUE/ABANDON decision. |

### 7.2 Decision Meeting Protocol

At the end of Week 4, the Verification Chair convenes a decision meeting:

1. **Each verifier presents their report** (5 min each, no rebuttals).
2. **Red-Team Reviewer presents attacks** and which were successfully defended (10 min).
3. **Scoring:** Each axis scored independently by all team members. Chair resolves discrepancies >2 points.
4. **CONT-* checklist:** Binary pass/fail on each, consensus required.
5. **ABAND-* checklist:** Any member can raise an ABAND trigger. Requires majority + Chair agreement.
6. **Decision:** CONTINUE, CONDITIONAL CONTINUE (with specified COND-*), or ABANDON.
7. **If CONDITIONAL CONTINUE:** Chair specifies which COND-* reductions apply and revised scope is documented before implementation begins.

---

## 8. Appendix: Verification Checklist Templates

### 8.1 Per-Theorem Soundness Checklist (V1)

```
Theorem: _______________
Reviewer: _______________
Date: _______________

[ ] 1. All assumptions listed and cross-referenced to master list
[ ] 2. Proof conclusion matches theorem statement exactly
[ ] 3. Every proof step has explicit justification
[ ] 4. No hidden assumptions found (or all found assumptions added to list)
[ ] 5. Boundary case ε=0: _______________
[ ] 6. Boundary case ε=1: _______________
[ ] 7. Boundary case |V|=|Θ|: _______________
[ ] 8. Boundary case |V|=0: _______________
[ ] 9. Boundary case L→∞: _______________
[ ] 10. Red-team attack 1: _______________
[ ] 11. Red-team attack 2: _______________
[ ] 12. Red-team attack 3: _______________

Verdict: PASS / FAIL (with specific failures listed)
```

### 8.2 Deliverable Quality Checklist

```
Deliverable: approach.json / paper.tex
Reviewer: _______________
Date: _______________

approach.json:
[ ] AJ-1: Every algorithm has pseudocode
[ ] AJ-2: Every algorithm has complexity analysis
[ ] AJ-3: Every theorem has complete proof sketch
[ ] AJ-4: Implementation mapping complete
[ ] AJ-5: Kill gates quantitative and testable
[ ] AJ-6: Risk assessment per component
[ ] AJ-7: Assumptions explicit and enumerated
[ ] AJ-8: Constants computed, not symbolic

paper.tex:
[ ] PT-1: ≥50KB actual content
[ ] PT-2: No unanchored weasel words
[ ] PT-3: All theorem assumptions self-contained
[ ] PT-4: Novel proofs complete, known-pattern proofs sketched
[ ] PT-5: ≥3 falsifiable hypotheses
[ ] PT-6: Statistical methodology specified
[ ] PT-7: Related work honest and complete
[ ] PT-8: All definitions/theorems load-bearing
[ ] PT-9: Notation consistent and minimal
[ ] PT-10: "So what?" test passes for both venues

Verdict: PASS / REVISE (≤2 failures) / REJECT (≥3 failures)
```

### 8.3 CONTINUE/ABANDON Decision Record

```
Date: _______________
Participants: _______________

CONTINUE criteria:
[ ] CONT-1: C1 proof sound          Scorer: ___ Score: ___
[ ] CONT-2: C2 constants computed    Scorer: ___ Score: ___
[ ] CONT-3: Novelty confirmed        Scorer: ___ Score: ___
[ ] CONT-4: Hypotheses specified      Scorer: ___ Score: ___
[ ] CONT-5: Implementation feasible   Scorer: ___ Score: ___
[ ] CONT-6: Red-team addressed        Scorer: ___ Score: ___
[ ] CONT-7: ε vs. CP improvement      Scorer: ___ Score: ___
[ ] CONT-8: Lipschitz treatment       Scorer: ___ Score: ___

ABANDON triggers:
[ ] ABAND-1: C1 proof gap           NOT TRIGGERED / TRIGGERED
[ ] ABAND-2: Lipschitz vacuous      NOT TRIGGERED / TRIGGERED
[ ] ABAND-3: ε worse than CP        NOT TRIGGERED / TRIGGERED
[ ] ABAND-4: Δ_max too small        NOT TRIGGERED / TRIGGERED
[ ] ABAND-5: No novel contribution  NOT TRIGGERED / TRIGGERED

Axis scores:
Theory Soundness:        ___/10
Algorithmic Novelty:     ___/10
Evaluation Rigor:        ___/10
Implementation Feasibility: ___/10
Best-Paper Potential:    ___/10
COMPOSITE:               ___/50

CONDITIONAL CONTINUE triggers:
[ ] COND-1: Unresolved attacks      TRIGGERED / NOT
[ ] COND-2: ε tightness uncertain   TRIGGERED / NOT
[ ] COND-3: Multi-step fails        TRIGGERED / NOT
[ ] COND-4: Wrapping 5×–10×        TRIGGERED / NOT
[ ] COND-5: Developer study blocked  TRIGGERED / NOT
[ ] COND-6: CP improvement 3–5×    TRIGGERED / NOT

DECISION: CONTINUE / CONDITIONAL CONTINUE / ABANDON
Scope reductions (if conditional): _______________
Chair signature: _______________
```

---

*This framework is binding. No theory deliverable proceeds to implementation without passing the criteria above. The Verification Chair has final authority on CONTINUE/ABANDON decisions, subject to the quantitative criteria defined herein. Subjective overrides are not permitted — if ABAND-* is triggered, the project is abandoned regardless of sunk cost or emotional investment.*

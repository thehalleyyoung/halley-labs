# Theory Evaluation — Deep Mathematician

## Proposal: Causal-Plasticity Atlas (proposal_00)

**Approach:** A causal-plasticity atlas engine that uses curiosity-driven quality-diversity search to systematically map which causal mechanisms are invariant versus plastic versus emergent across contexts, producing a navigable archive via novel DAG alignment operators, information-theoretic plasticity descriptors, tipping-point detection, and robustness certificates.

**Theory artifacts evaluated:** approach.json (166KB: 14 definitions, 8 theorems, 5 algorithms, 6 assumptions) and paper.tex (186KB: 9 theorems, 5 lemmas, 14 definitions, 14 proofs, 6 algorithms).

**Evaluation method:** Three-expert adversarial team (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) with cross-critique synthesis. All scores reflect post-adversarial consensus.

---

## Load-Bearing Math Assessment

The central question for a deep mathematician: **Is the math load-bearing — the reason the artifact is hard to build and the reason it delivers extreme value?**

### What is genuinely load-bearing

1. **The √JSD metric choice and its cascade (D5, T1).** The selection of √JSD (Endres-Schindelin 2003) as the base mechanism distance is a non-trivial design decision that cascades through the entire framework. KL divergence fails (no triangle inequality, can be infinite). Fisher-Rao requires parametric assumptions. √JSD is the uniquely correct choice: a proper metric, always finite, interpretable. Theorem 1 (DAG alignment metric properties) *requires* this choice for the triangle inequality proof via alignment composition. This is the single design decision that makes the formal machinery possible. **Load-bearing: YES.**

2. **The robustness certificate framework (D14, ALG5, T3).** No prior multi-context causal inference method provides per-mechanism, calibrated confidence statements with explicit margins. The certificate R_i = (κ_i, δ_i, n_min, α) combines stability selection for structural invariance, bootstrap UCB for parametric stability, and L∞ margins to classification boundaries. T3 (certificate soundness) composes Shah-Bühlmann bounds with bootstrap consistency — individually known techniques, but the composition for this purpose is new. This is the feature that distinguishes CPA from "just run ICP + compute pairwise JSD." **Load-bearing: YES.**

3. **Structural stability bounds (T8).** The perturbation analysis showing |Δψ_S| ≤ 2s/(Kn), |Δψ_P| ≤ O(s·d_max·σ/√N_min), |Δψ_E| ≤ 2s/(max|MB|+1) is practically the most important result — it tells you when to trust the pipeline. The ψ_S bound is tight and useful (two orders of magnitude below thresholds at realistic s). **However, the ψ_P bound is vacuous at realistic DAG error levels** (s≥5 gives perturbation exceeding the [0,1] range). **Load-bearing for structural plasticity; vacuous for parametric plasticity.**

### What is ornamental

1. **Atlas completeness guarantee (T6).** A coupon-collector bound for MAP-Elites on a finite space with ergodic mutation. The convergence rate depends on p_min ≈ (1/(4·max(K,n)))^D, which is astronomically small for practical genome dimensions. The theorem is asymptotically correct and practically useless. Any ergodic search on a finite space eventually covers it — this is not a theorem worth stating. **Ornamental.**

2. **QD search curiosity signal (D11, ALG3).** Standard count-based exploration bonus (d_cell + β·Δq). The behavior descriptor b(g) is a deterministic function of precomputed plasticity descriptors — there is no stochastic landscape to explore. MAP-Elites was designed for complex, stochastic fitness landscapes (robotics, game design). Here, "search" is enumeration of context-mechanism subsets. A grid search with 6^4 = 1296 cells achieves the same coverage. **Ornamental.**

3. **Classification correctness (T2).** Standard separation-condition argument: "if you're far enough from the decision boundary, you classify correctly." The proof is a union bound over McDiarmid and sub-Gaussian concentration inequalities — undergraduate probability. Necessary for completeness, but mathematically trivial. **Load-bearing but incremental.**

4. **Tipping-point consistency (T5).** Inherits entirely from PELT consistency (Killick et al. 2012). The only new content is verifying that √JSD estimator sequences converge uniformly, which follows from T4. Remove the CPA framing and this says "PELT is consistent on consistent estimators." **Parasitic on known results.**

### The 4D descriptor: packaging or invention?

The plasticity descriptor ψ = (ψ_S, ψ_P, ψ_E, ψ_CS) concatenates four separate statistics into a vector:
- ψ_S = √JSD over parent indicators (a divergence between binary vectors)
- ψ_P = mean pairwise √JSD of conditionals (an average distance)
- ψ_E = 1 - min|MB|/(max|MB|+1) (a ratio of Markov blanket sizes)
- ψ_CS = Spearman correlation (context distance vs mechanism distance)

There is no interaction term, no joint structure, no tensor product. The components are not proven orthogonal; ψ_P and ψ_CS are explicitly correlated by construction. The paper calls this a "rich geometric object" — it is a Cartesian product of four 1D statistics. **The descriptor is packaging, not invention.** The value lies in the infrastructure built around it (classification, certificates), not in the descriptor itself.

**Critical implementation gap:** The Skeptic correctly identified that ψ_E does not faithfully implement D12 (the emergence relation). D12 requires literal variable absence (X_i ∉ V_{c'}); ψ_E measures Markov blanket size ratios. A variable present in all contexts with varying MB sizes can be classified as "emergent" by ψ_E, contradicting D12. This is a fixable design bug but undermines the emergence claims as stated.

---

## Pillar Scores

### 1. Extreme Value: 5 / 10

The plasticity spectrum (invariant → parametric → structural → emergent) genuinely extends ICP's binary classification. The mapping to actionable responses (transfer, re-estimate, re-learn, collect new data) is compelling. The robustness certificates have no competitor in the literature.

**But:** The "atlas" metaphor oversells a classification + certification system. The QD archive promises interactive exploration that the evaluation never validates. The real-world case studies (GTEx, Penn World Table) are afterthoughts — GTEx has ~340 samples/tissue (below the recommended 500), and PWT violates the i.i.d. assumption. The primary beneficiaries are causal inference methodologists, not domain scientists. A genomics researcher would use ICP + domain expertise before learning 14 definitions and 5 algorithms.

The value is real but niche: useful for the multi-context causal inference community, not transformative for science broadly.

### 2. Genuine Software Difficulty: 6 / 10

All building blocks are known: Hungarian matching (1955), PELT (2012), stability selection (2010), bootstrap (1979), MAP-Elites (2015). The genuine difficulty is *composition under statistical guarantees*: maintaining metric properties through alignment, ensuring certificate soundness under bootstrap, getting Lipschitz constants right for perturbation analysis. This is non-trivial systems engineering.

The hardest algorithmic component is CADA (ALG1) — NP-hard in general (T7a, via subgraph isomorphism reduction), with an FPT result (T7b) under bounded degree. The three-phase pruning pipeline (anchor propagation → MB filtering → CI fingerprinting → Hungarian) integrates structural and statistical information in a way no existing graph alignment library does.

**But:** A competent research engineer with causal-learn, scipy, and ruptures could build a working prototype (without formal guarantees) in 2-3 weeks. Getting the formal guarantees right takes 2-3 months. The difficulty is in correctness, not in conceptual innovation.

### 3. Best-Paper Potential: 4 / 10

No single theorem introduces a genuinely novel proof technique. T1's triangle inequality leverages √JSD's known metric property through alignment composition — elegant but straightforward once you make the right distance choice. T4's sample complexity is a standard Fano lower bound. T8's perturbation bounds are component-wise sensitivity analysis. All proofs are correct applications of standard tools (McDiarmid, Fano, bootstrap consistency, coupon collector).

The most novel intellectual contribution is the *problem formulation* — treating mechanism change as a geometric object rather than a binary label. But formulations rarely win best papers; results do.

**Target venue:** UAI or AISTATS (not NeurIPS/ICML). Would be a solid accept, potentially an oral, but not a best-paper contender without a surprising empirical finding. The paper's 186KB / ~50-page length is incompatible with NeurIPS (9 pages) or ICML (8 pages). JMLR would be the natural journal target.

### 4. Laptop-CPU Feasibility & No-Humans: 7 / 10

Fully automated, no GPUs, no human annotation, no human studies. For the practical sweet spot (n ∈ [15,50], K ∈ [5,20], N_c ≥ 500), the full pipeline completes in 30-120 minutes on a single CPU core.

**Bottleneck:** Certificate generation requires O(n_inv × 100 × K × T_DAG) operations. For n=100, K=50, ~60 invariant mechanisms: ~300,000 DAG re-estimations ≈ 83 hours sequential. Even with 8-core parallelism: ~11 hours. The paper's FC7 claims carefully test only K ≤ 10, n ≤ 100 — avoiding the K=50 regime.

Selective certification (skipping mechanisms with stable parent sets) mitigates this. The system is laptop-feasible for the practical sweet spot, strained but possible at the upper range.

### 5. Feasibility: 5 / 10

The framework can be built and will work on synthetic data (where assumptions hold by construction). The mathematical machinery is correct and well-specified.

**Critical vulnerabilities:**

1. **DAG quality dependence.** The entire pipeline is bounded by upstream causal discovery accuracy. For n=100, N=500, GES typically produces SHD=15-25. T8's bounds are vacuous for ψ_P at this error level. The certificates explicitly condition on DAG correctness — if DAGs are systematically wrong (not just randomly perturbed), certificates are invalid.

2. **Sample size mismatch.** T4 requires N_min = Ω(d²_max/ε² · log(nK/α)). For n=100, K=50, ε=0.1: N_min ≈ 29,000 per context. The practical regime assumes N=500, giving effective ε ≈ 0.68 — useless precision on [0,1] descriptors. The Gaussian parametric assumption may close this gap, but this is empirically untested.

3. **Faithfulness assumption (A2) is the Achilles heel.** In biological data, path cancellation is common. The mitigation (use GES) is insufficient since GES also assumes faithfulness for consistency.

4. **λ-weight sensitivity.** The alignment distance depends on three free parameters (λ_struct, λ_mech, λ_miss) with no principled method for setting them and no sensitivity analysis proposed.

---

## Fatal Flaws

**No single fatal flaw identified, but a compound fragility exists.** The DAG quality + sample size + faithfulness chain means: *the system is only as reliable as its weakest link, and the weakest link (observational causal discovery at scale) is an unsolved problem not specific to CPA.* The proposal is admirably honest about this (TR1 is rated HIGH severity). The Oracle-DAG baseline (BL6) is the correct diagnostic — but if the gap between CPA and Oracle-CPA exceeds 25% F1, the practical value proposition collapses.

The ψ_E implementation gap (descriptor doesn't match definition D12) is a design bug, not a fatal flaw.

---

## Summary Assessment

| Pillar | Score |
|--------|-------|
| 1. Extreme value | **5 / 10** |
| 2. Genuine software difficulty | **6 / 10** |
| 3. Best-paper potential | **4 / 10** |
| 4. Laptop-CPU feasibility & no-humans | **7 / 10** |
| 5. Feasibility | **5 / 10** |

**Composite: 5.4 / 10**

### What should be kept (load-bearing, novel, valuable)
1. DAG alignment metric (D6 + T1) — genuine mathematical contribution
2. Plasticity descriptor + classification (D7 + D8) — core conceptual advance
3. Robustness certificates (D14 + T3 + ALG5) — strongest novel contribution
4. Structural stability analysis (T8) — most practically important theorem
5. Variable-set mismatch / emergence formalization (D3, D12) — fills real literature gap

### What should be cut (ornamental, over-engineered)
1. QD search (ALG3, D11, T6) — MAP-Elites is unjustified; deterministic landscape
2. Tipping-point detection (ALG4, T5) — spin off as separate contribution; parasitic on PELT
3. Atlas completeness theorem (T6) — practically vacuous coupon-collector bound

### The honest framing
This is a **well-engineered framework contribution** with genuine novelty in problem formulation (mechanism change as geometry) and in the certificate framework. The math is correct but not deep — competent composition of known techniques (concentration inequalities, bootstrap, metric space theory, graph alignment). The real contribution is architectural: combining causal discovery + mechanism comparison + certification into a pipeline with well-defined interfaces and formal soundness conditions. The paper should be positioned as a framework/systems contribution with theoretical guarantees, not as a pure theory paper.

## VERDICT: CONTINUE

**Rationale:** Despite the moderate scores, the proposal survives for three reasons: (1) the problem formulation fills a genuine gap — no prior framework characterizes *how* mechanisms change, and the research community needs this; (2) the certificate framework is genuinely novel with no competitor; (3) the five unresolved disputes (Oracle-DAG gap, Lazy CPA baseline, emergence descriptor fidelity, N=500 sufficiency, λ-weight sensitivity) are all *empirically testable* during implementation — the theory stage has done its job by making falsifiable claims that the implementation stage can resolve. The proposal earns CONTINUE not through mathematical brilliance but through careful, honest engineering of a useful system with testable guarantees.

**Conditions for continued viability (any failure escalates to ABANDON):**
1. Oracle-DAG gap must be ≤ 25% F1 on Generators 1-4
2. CPA must demonstrate ≥ 15% marginal F1 improvement over ICP + pairwise JSD baseline
3. Classification F1 must exceed 0.75 at N=500 per context on Generator 1
4. Drop the QD search entirely; reframe as "Plasticity Spectrum with Certificates"
5. Fix ψ_E to gate on actual variable absence per D12

## Independent Verifier Signoff

**Verification performed against:** Three source evaluations (Auditor, Skeptic, Synthesizer), cross-critique synthesis, and original theory artifacts (approach.json, referenced paper.tex).

### 1. Are the scores justified by evidence?

**YES.** All five pillar scores match the cross-critique's synthesized scores exactly (5, 6, 4, 7, 5 → composite 5.4). Each score is traceable to specific evaluator arguments:

- **Extreme Value (5):** Auditor's reasoning prevails over Synthesizer's 4 — correctly credits the certificate framework's novelty and the plasticity spectrum's actionability, while honestly acknowledging the "atlas" oversell. Justified.
- **Software Difficulty (6):** Synthesizer/Skeptic reasoning prevails over Auditor's 7 — correctly distinguishes "composition under guarantees" from "algorithmic invention." The 2–3 week prototype vs 2–3 month formal-guarantee timeline is a useful calibration. Justified.
- **Best-Paper Potential (4):** Synthesizer/Skeptic reasoning correctly overrides Auditor's 6. The cross-critique's analysis (§4B) is the sharpest adjudication in the entire evaluation — distinguishing "quality of a good paper" from "probability of best-paper award." Justified and well-argued.
- **Laptop-CPU Feasibility (7):** All three evaluators converge. The FC7 avoidance of K=50 is correctly flagged. Justified.
- **Feasibility (5):** Skeptic's concerns appropriately lower the Auditor/Synthesizer consensus of 6. The four-risk enumeration (DAG quality, faithfulness, sample size, ψ_E gap) is faithful to source material. Justified.

### 2. Does CONTINUE follow from the scores?

**YES.** A composite of 5.4/10 is moderate, and CONTINUE is not automatic at this level. The verdict is justified by three specific arguments: (1) the problem formulation fills a genuine literature gap (confirmed by all three evaluators), (2) the certificate framework has no competitor (unanimous consensus point #4 in cross-critique), and (3) all unresolved disputes are empirically testable during implementation. The framing — "earns CONTINUE not through mathematical brilliance but through careful, honest engineering" — is accurate and appropriately calibrated.

### 3. Are the conditions reasonable and testable?

**YES, with one note.**

- **Condition 1** (Oracle-DAG gap ≤ 25% F1): Testable via BL6. Note: the Skeptic's original kill threshold was 20% F1; the cross-critique resolved this at 25%. The 5-point relaxation is reasonable but should be acknowledged as a deliberate choice, not an oversight.
- **Condition 2** (≥ 15% marginal F1 over ICP + pairwise JSD): Testable via BL1/BL2. Aligns with the Skeptic's concern that Lazy CPA captures 65–70% of value.
- **Condition 3** (F1 ≥ 0.75 at N=500): Testable via FC3 sweep. This is the most aggressive condition — the Skeptic's sample-complexity analysis suggests ε ≈ 0.68 at N=500 under the minimax bound, but the Gaussian parametric assumption may close the gap. Correctly defers to empirical resolution.
- **Condition 4** (Drop QD search): Actionable. Unanimously supported by all three evaluators.
- **Condition 5** (Fix ψ_E): Actionable. The Skeptic's diagnosis is correct and the Synthesizer's cure (keep D12 concept, fix implementation) is the right approach.

### 4. Are any important source arguments lost?

**Two minor omissions identified:**

1. **CPDAG identification gap (Skeptic §6.2).** The Skeptic notes that ψ_S only measures changes in *definite parents* (edges directed in every member of the CPDAG equivalence class). Undirected edges are invisible, causing ψ_S to systematically underestimate structural plasticity in sparse graphs. This is not mentioned in the final evaluation. **Impact: LOW-MODERATE.** This is an inherent limitation of observational causal discovery, not specific to CPA, but should be noted since ψ_S is identified as the most robust descriptor component under T8 — its robustness may partly reflect insensitivity rather than stability.

2. **Gaussian linearity limitation (Skeptic §6.3).** All proofs and closed-form computations assume Gaussian linear SEMs. For nonlinear mechanisms, √JSD must be estimated nonparametrically with convergence rate O(n^{−2/(d+4)}) — catastrophically slow for d_max ≥ 3. The theory is strictly valid only for Gaussian linear SEMs, a severe limitation for the biological applications targeted. The final evaluation mentions Gaussian assumptions in passing (feasibility, sample sizes) but does not flag this as a standalone vulnerability. **Impact: MODERATE.** This should be added as a sixth condition or at minimum noted as a known limitation scope.

Neither omission is severe enough to change the verdict or scores.

### 5. Is the "load-bearing vs ornamental" assessment accurate?

**YES.** The assessment is faithful to the unanimous consensus across all three evaluators:

- **√JSD + T1 as load-bearing:** Correct. All three evaluators independently confirm this is the foundational design decision enabling the formal machinery.
- **Certificates (D14 + T3 + ALG5) as load-bearing:** Correct. Identified as the strongest novel contribution by all three (consensus point #4).
- **T8 as "load-bearing for structural plasticity; vacuous for parametric plasticity":** Correct and importantly nuanced. The cross-critique's component-wise table (ψ_S robust ✓, ψ_P vacuous at s≥3 ✗, ψ_E fragile at s≥5 ⚠) is faithfully reproduced.
- **QD search as ornamental:** Correct. Unanimous across all evaluators.
- **T6 (atlas completeness) as ornamental:** Correct. The coupon-collector bound with p_min ≈ (1/(4·max(K,n)))^D is practically vacuous.
- **4D descriptor as "packaging, not invention":** Correct and important. The Skeptic's argument that these are four concatenated 1D statistics with no interaction term, not proven orthogonal, and with ψ_P/ψ_CS explicitly correlated by construction, is accurately reflected.
- **T5 as "parasitic on known results":** Correct. PELT consistency on consistent estimators is a known composition.

One subtlety worth noting: the Synthesizer ranked D12 (Emergence Relation) as Tier 1 novelty (MODERATE-HIGH), and the final evaluation correctly preserves this in the "what should be kept" list while simultaneously acknowledging the ψ_E implementation flaw. This is the right balance.

---

### SIGNOFF: APPROVED

The evaluation is thorough, internally consistent, faithfully synthesizes three adversarial source evaluations, and reaches a well-justified CONTINUE verdict with concrete, testable conditions. The scores are evidence-backed and the load-bearing/ornamental distinctions are accurate.

**Minor recommendations (non-blocking):**
1. Acknowledge that the Oracle-DAG gap threshold was relaxed from the Skeptic's 20% to 25% — a deliberate choice, not a gap.
2. Consider adding the Gaussian linearity limitation as an explicit scope constraint or sixth viability condition, since all theoretical guarantees are strictly Gaussian-linear only.
3. Note the CPDAG identification gap's effect on ψ_S sensitivity as a caveat to T8's robustness claim for structural plasticity.

**Verifier confidence: HIGH.** No proof errors, no score manipulation, no lost critical arguments. The two omissions are minor and do not affect the verdict.

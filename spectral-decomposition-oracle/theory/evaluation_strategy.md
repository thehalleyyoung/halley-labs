# Evaluation Strategy: Spectral Features for MIP Decomposition Selection

**Paper Target**: INFORMS Journal on Computing  
**Title**: *Spectral Features for MIP Decomposition Selection: A Computational Study with the First Complete MIPLIB 2017 Decomposition Census*  
**Document Role**: Empirical Scientist — complete falsifiable evaluation and baseline strategy  
**Version**: 1.0  
**Status**: Theory stage artifact  

---

## 1. Falsifiable Hypotheses

Each hypothesis is stated with a precise test statistic, decision threshold, and rejection criterion. All hypotheses are tested on the **paper-tier** 500-instance stratified sample unless stated otherwise.

### H1: Spectral Ratio Correlates with Bound Degradation

**Statement**: The spectral ratio δ²/γ² (squared Fiedler gap over squared spectral gap of the partition Laplacian) exhibits a monotone relationship with observed LP-relaxation bound degradation when moving from monolithic to decomposed formulations.

- **Test statistic**: Spearman rank correlation ρ between δ²/γ² and relative bound degradation Δ = (z*_LP − z*_decomp) / |z*_LP|, computed per instance.
- **Threshold**: ρ ≥ 0.40 (moderate correlation).
- **Kill condition (G1)**: If ρ < 0.40 on the 50-instance pilot, **abandon** the spectral ratio as a standalone predictor. Escalation: if 0.30 ≤ ρ < 0.40, proceed to dev-tier (200 instances) for confirmation before abandoning.
- **Scope**: Computed only on instances where decomposition is non-trivial (i.e., GCG or SCIP-Benders returns a feasible decomposition within 3600s and produces a finite dual bound). Instances where both backends time out or return identical bounds to monolithic SCIP are excluded from this test.
- **Confound control**: Partial correlation controlling for instance size (n + m), density (nnz / (n·m)), and constraint-matrix condition number κ(A).

### H2: Spectral Features Add Predictive Value Over Syntactic Features

**Statement**: A classifier trained on spectral features (SPEC-8) achieves strictly higher balanced accuracy than a classifier trained on syntactic features (SYNT-25) for the three-class decomposition recommendation task {Benders, DW, neither}.

- **Test statistic**: Difference in balanced accuracy Δ_BA = BA(SPEC-8) − BA(SYNT-25), evaluated via nested cross-validation.
- **Threshold**: Δ_BA > 0 (strict improvement), confirmed by McNemar's test at α = 0.05 after Holm–Bonferroni correction across all pairwise comparisons in the ablation.
- **Kill condition**: Not a standalone kill gate but feeds into G3.
- **Secondary metric**: Improvement must hold on at least 2 of the 4 non-trivial structure classes (block-angular, bordered-block-diagonal, staircase, dual-block-angular).

### H3: L3 Bound Correlates with Actual Gap

**Statement**: The numerically computed L3 bound (partition-to-bound bridge) correlates positively with the empirically observed decomposition gap.

- **Test statistic**: Spearman ρ between L3_bound(π, w) and observed gap Δ_gap = |z*_decomp − z*_LP| / |z*_LP|, where π is the partition and w the crossing-edge weights.
- **Threshold**: ρ ≥ 0.35.
- **Scope**: Restricted to instances where (a) LP relaxation solves within 60s and (b) shadow prices π are available from the LP solve.
- **Stratification**: Report ρ separately for each structure type and for κ(A) ≤ 10³ vs. κ(A) > 10³ (the vacuity boundary from T2).

### H4: Futility Predictor Precision

**Statement**: The spectral futility predictor (binary classifier: {decomposition-helpful, decomposition-futile}) achieves precision ≥ 0.80 on the "futile" class.

- **Test statistic**: Precision = TP_futile / (TP_futile + FP_futile), where a false positive means labeling a decomposition-helpful instance as futile (missed opportunity).
- **Threshold**: Precision ≥ 0.80 at recall ≥ 0.50 on the "futile" class.
- **Decision**: If precision < 0.80, report as negative result but do not abandon the paper (this is S1, a secondary contribution).
- **Label definition**: An instance is "decomposition-futile" if neither GCG nor SCIP-Benders improves the dual bound by ≥ 1% over monolithic SCIP within the same wall-clock budget.

### H5: Feature Robustness to Scaling

**Statement**: Spectral features are stable under the three equilibration strategies (Ruiz, geometric-mean, SCIP-native) used in the pipeline.

- **Test statistic**: Intraclass correlation coefficient ICC(3,1) across the three equilibrations for each spectral feature, averaged over the 500-instance sample.
- **Threshold**: Mean ICC ≥ 0.85 (good reliability).
- **Secondary**: No single feature may have ICC < 0.60. Any feature with ICC < 0.60 is flagged and tested for removal without accuracy loss.
- **Procedure**: For each instance, compute all 8 spectral features under each of the 3 equilibrations. Report ICC per feature and mean ICC.

### H6: Spectral Features Not Redundant with Syntactic

**Statement**: Spectral features (SPEC-8) are not linearly predictable from syntactic features (SYNT-25).

- **Test statistic**: Maximum R² when regressing any single spectral feature on all 25 syntactic features via OLS.
- **Threshold (Kill Gate G0)**: max(R²) < 0.70.
- **Kill condition**: If max(R²) ≥ 0.70 for **any** spectral feature, that feature is flagged as potentially redundant. If ≥ 5 of 8 spectral features are flagged, **abandon** the spectral feature family as a distinct contribution.
- **Procedure**: Fit 8 OLS regressions (one per spectral feature as dependent variable, all 25 syntactic as independent). Report R², adjusted R², and VIF for each regression.

### H7: Per-Structure-Type Accuracy Improvement

**Statement**: Spectral features improve classification accuracy specifically on non-trivial structure classes where decomposition decisions are ambiguous.

- **Test statistic**: Per-class balanced accuracy improvement Δ_BA_k = BA_k(COMB-ALL) − BA_k(SYNT-25) for each structure class k ∈ {block-angular, bordered-block-diagonal, staircase, dual-block-angular}.
- **Threshold**: Δ_BA_k > 0 for at least 2 of the 4 classes.
- **Kill condition**: If COMB-ALL ≤ SYNT-25 on all 4 non-trivial classes, the spectral features provide no structure-type-specific value.
- **Note**: The "neither" class (no exploitable structure) is excluded from this test since decomposition selection is moot for those instances.

---

## 2. Experimental Design per Hypothesis

### 2.1 General Design Principles

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Significance level α | 0.05 (family-wise) | Standard for computational OR |
| Multiple-comparison correction | Holm–Bonferroni | Controls FWER; less conservative than Bonferroni |
| Number of hypotheses | 7 | Adjusted α per Holm: 0.05/7 ≈ 0.0071 for most significant |
| Effect size convention | Cohen's d ≥ 0.5 (medium) | Minimum practically meaningful for JoC |
| Power target | 1 − β ≥ 0.80 | Standard |

### 2.2 Per-Hypothesis Design

#### H1 (Spectral Ratio ↔ Bound Degradation)

| Element | Specification |
|---------|---------------|
| Test | One-sided Spearman correlation test, H₀: ρ ≤ 0, H₁: ρ > 0 |
| Required n for power 0.80 | For ρ = 0.40, α = 0.0071: n ≥ 62 (Fisher z-transform power analysis). Pilot = 50 (slightly underpowered; confirmation on dev-200) |
| Stratification | By structure type × conditioning tier (κ ≤ 10³, κ ∈ (10³, 10⁶), κ > 10⁶) |
| Confound control | Partial Spearman correlation controlling for log(n+m), log(nnz), log(κ) |
| Effect size | Report ρ with 95% bootstrap CI (10,000 resamples, bias-corrected accelerated) |

#### H2 (Spectral > Syntactic)

| Element | Specification |
|---------|---------------|
| Test | McNemar's test on per-instance correct/incorrect predictions between SPEC-8 and SYNT-25 classifiers |
| Required n | For McNemar's with 15% discordant pairs, OR = 1.5, α = 0.05: n ≥ 200 discordant pairs → 500 instances sufficient if discordance rate ≥ 40% |
| CV scheme | Nested: outer 5-fold (stratified by label), inner 3-fold (hyperparameter tuning) |
| Classifiers | RF (500 trees), XGBoost (max_depth 6, early stopping), Logistic Regression (L1 + L2 elastic net). Report per-classifier and best-of-three |
| Correction | Holm–Bonferroni across the 15 pairwise config comparisons in the ablation |

#### H3 (L3 Bound ↔ Actual Gap)

| Element | Specification |
|---------|---------------|
| Test | One-sided Spearman correlation, H₀: ρ ≤ 0 |
| Required n | For ρ = 0.35, α = 0.0071: n ≥ 82 |
| Stratification | Structure type × conditioning. Report correlation for κ ≤ 10³ separately (where T2 is non-vacuous) |
| Auxiliary | Scatter plot of L3_bound vs. actual gap with LOWESS smoother. Report fraction of instances where L3_bound ≥ actual gap (tightness rate) |

#### H4 (Futility Predictor)

| Element | Specification |
|---------|---------------|
| Test | One-sided exact binomial test on precision, H₀: precision ≤ 0.80, H₁: precision > 0.80 |
| Required n | For precision = 0.85, H₀: 0.80, α = 0.05, power 0.80: need ≥ 150 predicted-futile instances |
| Classifier | RF or XGBoost (selected by inner CV), features = COMB-ALL |
| Threshold tuning | Optimize decision threshold on inner folds to maximize precision at recall ≥ 0.50 |
| Calibration | Platt scaling on inner validation fold; report calibration curve |

#### H5 (Scaling Robustness)

| Element | Specification |
|---------|---------------|
| Test | Two-way random-effects ICC(3,1) |
| Required n | For ICC = 0.85, 95% CI width ≤ 0.10: n ≥ 100 instances × 3 equilibrations |
| Procedure | Compute 8 features × 3 equilibrations × 500 instances = 12,000 feature values. Report ICC per feature with 95% CI |
| Auxiliary | Bland-Altman plots for the two most variable features |

#### H6 (Non-Redundancy)

| Element | Specification |
|---------|---------------|
| Test | OLS regression R² with F-test for overall significance |
| Required n | For R² = 0.70, 25 predictors, α = 0.05, power 0.80: n ≥ 75. We have 500 |
| Procedure | 8 regressions (one per spectral feature). Report R², adjusted R², Mallows' Cp |
| Auxiliary | Canonical correlation analysis between SPEC-8 and SYNT-25 blocks. Report first 3 canonical correlations |

#### H7 (Per-Structure-Type)

| Element | Specification |
|---------|---------------|
| Test | Per-class McNemar's between COMB-ALL and SYNT-25, restricted to instances of each structure type |
| Required n | Per-class minimum: 30 instances. Pilot data suggests: block-angular ~80, BBD ~60, staircase ~40, dual-block-angular ~30 in the 500-instance sample |
| Correction | Holm–Bonferroni across the 4 structure types |
| Auxiliary | Per-class confusion matrices and per-class ROC curves |

---

## 3. Feature Ablation Protocol

This is the **core experiment** of the paper. The ablation determines whether spectral features provide value beyond syntactic features for decomposition recommendation.

### 3.1 Feature Set Definitions

| Config ID | Features | Count | Description |
|-----------|----------|-------|-------------|
| **SPEC-8** | δ₁(L), δ₂(L), γ(L), Φ(L), ξ_loc, η_coupling, σ_gap_ratio, κ_spectral | 8 | Spectral features from hypergraph Laplacian: Fiedler value, second spectral gap, algebraic connectivity, Cheeger constant, eigenvector localization (IPR), coupling energy, gap ratio δ²/γ², spectral condition number |
| **SYNT-25** | n, m, nnz, density, obj_range, rhs_range, coeff_range, n_int, n_bin, n_cont, frac_int, frac_bin, constraint_type_counts (5), degree_stats (mean/std/skew/max for vars and constraints = 8), κ_A | 25 | Syntactic features extractable from the MPS/LP file without any graph analysis |
| **GRAPH-10** | max_clique, chromatic_lb, modularity, n_components, avg_clustering, diameter_est, degree_assortativity, betweenness_max, pagerank_entropy, community_count | 10 | Graph-theoretic features from the variable-interaction graph (no spectral computation) |
| **KRUBER-21** | Features from Kruber et al. (2017) Table 1 | 21 | Reimplementation of the Kruber feature set for direct comparison: includes constraint-matrix statistics, variable-constraint graph features, and decomposition-specific detectors |
| **COMB-ALL** | SPEC-8 ∪ SYNT-25 ∪ GRAPH-10 | 43 | Combined feature set (excluding KRUBER to avoid overlap; KRUBER tested as separate baseline) |
| **RANDOM** | Uniform random ∈ [0,1] | 8 | Sanity check: random features matching SPEC-8 cardinality |

### 3.2 Feature-Count-Controlled Ablation

To isolate feature **quality** from feature **quantity**, we test feature-count-controlled subsets:

| Control Level | Procedure |
|---------------|-----------|
| **top-3** | Select top 3 features by mutual information with label, computed on training fold only |
| **top-5** | Select top 5 features |
| **top-8** | Select top 8 features (matches SPEC-8 cardinality) |

For each of the 6 configs × 3 top-k levels × 3 classifiers = **54 experimental cells**. Feature selection is performed inside the inner CV loop to prevent leakage.

### 3.3 Classifier Specifications

| Classifier | Hyperparameter Grid (Inner CV) |
|------------|-------------------------------|
| **Random Forest** | n_estimators ∈ {100, 300, 500}, max_depth ∈ {5, 10, None}, min_samples_leaf ∈ {1, 5, 10}, max_features ∈ {sqrt, log2} |
| **XGBoost** | n_estimators ∈ {100, 300, 500}, max_depth ∈ {3, 6, 9}, learning_rate ∈ {0.01, 0.1, 0.3}, subsample ∈ {0.8, 1.0}, colsample_bytree ∈ {0.8, 1.0}, early_stopping_rounds = 20 |
| **Logistic Regression** | penalty = elasticnet, C ∈ {0.01, 0.1, 1, 10, 100}, l1_ratio ∈ {0.0, 0.25, 0.5, 0.75, 1.0}, solver = saga, max_iter = 5000 |

### 3.4 Nested Cross-Validation Protocol

```
For each outer fold k ∈ {1, ..., 5}:
    Train_outer = folds \ {k}
    Test_outer = fold k
    
    For each config c ∈ {SPEC-8, SYNT-25, GRAPH-10, KRUBER-21, COMB-ALL, RANDOM}:
        For each top-k ∈ {3, 5, 8}:
            Compute feature ranking on Train_outer (MI with label)
            Select top-k features → X_selected
            
            For each classifier clf ∈ {RF, XGBoost, LogReg}:
                For each inner fold j ∈ {1, 2, 3} within Train_outer:
                    Train inner fold, validate on held-out inner fold
                    Record balanced accuracy
                
                Select best hyperparameters by mean inner balanced accuracy
                
                Retrain on full Train_outer with best hyperparameters
                Predict on Test_outer
                Record: per-instance predictions, probabilities, accuracy, balanced accuracy

    # McNemar's tests (after all configs evaluated on this outer fold):
    Store per-instance correct/incorrect vectors for all (config, top-k, clf) triples
```

**Stratification**: Outer folds are stratified by the three-class label {Benders, DW, neither} to ensure each fold has proportional class representation. If any class has < 25 instances in the 500-instance sample, use stratified repeated holdout (20 repetitions, 80/20 split) instead of 5-fold CV.

### 3.5 Statistical Comparison

- **Primary**: McNemar's exact test between all pairs of the 6 configs (at the best top-k and best classifier per config). 15 pairwise tests → Holm–Bonferroni correction.
- **Secondary**: Friedman test across 6 configs using balanced accuracy as the response, followed by Nemenyi post-hoc test if Friedman rejects. Report critical difference diagram.
- **Effect size**: Cohen's g for each McNemar comparison. Report proportion of discordant pairs and odds ratio.

### 3.6 Reporting Requirements

For each of the 54 experimental cells, report:

| Metric | Definition |
|--------|------------|
| Accuracy | Standard classification accuracy |
| Balanced accuracy | Mean per-class recall |
| Macro F1 | Unweighted mean of per-class F1 |
| Spearman ρ | Rank correlation between predicted decomposition benefit score and actual benefit (for regression variant) |
| Feature importance | Permutation importance (10 repeats) for RF/XGBoost; absolute coefficient for LogReg |
| Calibration | Expected calibration error (ECE) with 10 bins |

Additionally, for the paper's main results table, report the **best classifier per config** with 95% CI on balanced accuracy (bootstrap, 10,000 resamples).

---

## 4. Label Definition and Stability

### 4.1 Ground-Truth Label Construction

Labels are derived from **external baseline solvers**, not from the oracle's own predictions, to avoid evaluation circularity (flaw F2 from depth_check).

**Per-instance labeling procedure** at wall-clock cutoff T:

```
For each instance i, cutoff T:
    Run monolithic SCIP → z_mono(i, T)      [primal bound, dual bound, gap]
    Run GCG (DW mode) → z_DW(i, T)          [primal, dual, gap]  
    Run SCIP-Benders → z_BD(i, T)            [primal, dual, gap]
    
    Compute relative dual-bound improvement:
        Δ_DW(i, T)  = (z_DW_dual(i, T) - z_mono_dual(i, T)) / |z_mono_dual(i, T)|
        Δ_BD(i, T)  = (z_BD_dual(i, T) - z_mono_dual(i, T)) / |z_mono_dual(i, T)|
    
    Label assignment:
        If max(Δ_DW, Δ_BD) < 0.01:          label = "neither"
        Elif Δ_DW > Δ_BD + 0.01:            label = "DW"
        Elif Δ_BD > Δ_DW + 0.01:            label = "Benders"
        Else:                                 label = argmax by secondary tiebreak
```

### 4.2 Multi-Cutoff Analysis

Labels are computed at four wall-clock cutoffs to assess stability:

| Cutoff | Purpose | Expected Usage |
|--------|---------|----------------|
| **60s** | Quick screening; many timeouts | Pilot only |
| **300s** | Development evaluation | Dev-tier, feature selection |
| **900s** | Primary paper evaluation | Paper-tier main results |
| **3600s** | Definitive; slowest but most reliable | Paper-tier sensitivity analysis |

### 4.3 Label Stability Analysis

**Definition**: Label_stable(i) = 1 if the label at T=300s, T=900s, and T=3600s all agree.

- **Report**: Fraction of instances with stable labels. Target: ≥ 70% label stability.
- **If < 70%**: Report results separately for stable and unstable subsets. Discuss implications.
- **Cohen's κ**: Compute inter-cutoff agreement (κ between T=300 and T=3600). Target: κ ≥ 0.60 (substantial agreement).

### 4.4 Consensus Protocol

For the primary paper results, use the **majority-vote** label across cutoffs T ∈ {300, 900, 3600}:

```
label_consensus(i) = mode({label(i, 300), label(i, 900), label(i, 3600)})
```

If all three cutoffs disagree (three-way tie), assign label = "neither" (conservative).

### 4.5 Tie-Breaking Rules

| Scenario | Rule | Rationale |
|----------|------|-----------|
| Δ_DW ≈ Δ_BD (within 1% margin) | Prefer DW | DW has stronger theoretical grounding via L3; prefer the method with tighter bounds |
| Both methods improve by < 1% | Label "neither" | Below noise floor for practical relevance |
| One method times out, other succeeds | Label the succeeding method | Timeout ≈ failure for that method at this cutoff |
| Both methods time out | Label "neither" at this cutoff | No evidence of benefit |
| Monolithic SCIP solves to optimality in < T/10 | Label "neither" | Decomposition adds overhead for easy instances |

### 4.6 Label Distribution Monitoring

Before proceeding past pilot tier, verify label distribution is not pathologically imbalanced:

| Condition | Action |
|-----------|--------|
| Any class < 10% of sample | Flag as minority class; use SMOTE or class-weighted loss in classifiers |
| "neither" class > 70% | Expected (only 10–25% of MIPLIB has exploitable structure); report as finding, ensure stratification preserves minority classes |
| Any class = 0 instances | Reduce to binary classification; document why three-class is infeasible |

---

## 5. Baseline Specifications

Seven baselines provide the reference frame for evaluating the spectral oracle.

### 5.1 Baseline B0: Monolithic SCIP (Performance Floor)

| Parameter | Value |
|-----------|-------|
| Solver | SCIP 8.x (latest stable) |
| Configuration | Default settings, single-thread |
| Time limits | {60, 300, 900, 3600}s per instance |
| Metrics collected | Primal bound, dual bound, primal-dual gap, nodes explored, time to first feasible, LP iterations |
| Purpose | Establishes the "do nothing" performance floor. All decomposition methods must improve over this |

### 5.2 Baseline B1: GCG (DW Decomposition)

| Parameter | Value |
|-----------|-------|
| Solver | GCG 3.x (latest stable) with SCIP 8.x backend |
| Detection | GCG's built-in structure detection (all detectors enabled) |
| Configuration | Default GCG settings, single-thread |
| Time limits | Same as B0 |
| Metrics collected | Same as B0, plus: detected structure type, number of blocks, linking constraints, master LP bound after root |
| Purpose | External DW baseline; independent of our spectral detection |

### 5.3 Baseline B2: SCIP-Benders

| Parameter | Value |
|-----------|-------|
| Solver | SCIP 8.x with Benders' decomposition plugin |
| Detection | SCIP's built-in Benders detection |
| Configuration | Default Benders settings, single-thread |
| Time limits | Same as B0 |
| Metrics collected | Same as B0, plus: number of subproblems, cuts generated, master iterations |
| Purpose | External Benders baseline; independent of our spectral detection |

### 5.4 Baseline B3: Random Selector

| Parameter | Value |
|-----------|-------|
| Method | Uniformly random selection from {Benders, DW, neither}, repeated 10 times |
| Reporting | Mean and standard deviation of balanced accuracy across 10 seeds |
| Purpose | Establishes the chance-level floor; any learned model must beat this |

### 5.5 Baseline B4: Trivial Classifier (Always "Neither")

| Parameter | Value |
|-----------|-------|
| Method | Always predict the majority class ("neither") |
| Reporting | Accuracy (expected ≈ 60–75%), balanced accuracy (expected = 33.3%) |
| Purpose | Tests whether class imbalance inflates naive accuracy |

### 5.6 Baseline B5: Syntactic-Only Oracle

| Parameter | Value |
|-----------|-------|
| Features | SYNT-25 only |
| Classifier | Best of {RF, XGBoost, LogReg} selected by inner CV |
| CV | Same nested CV as ablation protocol (Section 3) |
| Purpose | **Critical comparison**: spectral features must outperform this to justify their computational overhead (~30s/instance eigensolve) |

### 5.7 Baseline B6: Kruber et al. (2017) Reimplementation

| Parameter | Value |
|-----------|-------|
| Features | KRUBER-21 (reimplemented from Kruber et al. 2017, Table 1) |
| Classifier | RF (Kruber's reported best), plus XGBoost and LogReg for fairness |
| CV | Same nested CV as ablation protocol |
| Scope | Kruber's original work was DW-only; we extend labels to 3-class for comparability and also report the DW-vs-rest binary subproblem |
| Purpose | Direct comparison to the closest prior work |

### 5.8 Baseline Summary Table

| ID | Name | Type | Expected Balanced Accuracy | Overhead per Instance |
|----|------|------|---------------------------|----------------------|
| B0 | Monolithic SCIP | Solver | N/A (reference) | 0s (it is the reference) |
| B1 | GCG | Solver | N/A (provides labels) | 300–3600s |
| B2 | SCIP-Benders | Solver | N/A (provides labels) | 300–3600s |
| B3 | Random | Classifier | ~33% | 0s |
| B4 | Trivial | Classifier | 33.3% | 0s |
| B5 | Syntactic-Only | Classifier | 55–65% (estimate) | ~1s |
| B6 | Kruber | Classifier | 60–70% (estimate) | ~5s |

---

## 6. L3 Empirical Validation

### 6.1 L3 Statement (Partition-to-Bound Bridge)

For a partition P = {V₁, ..., V_k} of variables, the gap between the LP relaxation and the decomposed dual bound is bounded by:

$$|z^*_{LP} - z^*_{decomp}| \leq \sum_{e \in E_{cross}(P)} w_e \cdot |\pi_e|$$

where E_cross(P) is the set of crossing hyperedges (constraints spanning multiple blocks), w_e is the hyperedge weight, and π_e is the shadow price of constraint e in the LP relaxation.

### 6.2 Numerical Computation Protocol

```
For each instance i with partition P_i:
    1. Solve LP relaxation of monolithic formulation → z*_LP, shadow prices {π_e}
    2. Identify crossing hyperedges: E_cross = {e : e spans ≥ 2 blocks in P_i}
    3. Compute L3 bound:
       L3(i) = Σ_{e ∈ E_cross} w_e · |π_e|
       where w_e = ‖Ã_{e,:}‖₂² / d_e  (from equilibrated matrix)
    4. Compute actual gap:
       Δ(i) = |z*_LP - z*_decomp(P_i)|
       where z*_decomp is from GCG (DW) or SCIP-Benders
    5. Compute tightness ratio:
       τ(i) = Δ(i) / L3(i)
       (τ ≤ 1 means bound holds; τ > 1 means bound is violated — 
        indicates numerical issues or assumption violations)
```

### 6.3 Validation Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **Correlation** | Spearman ρ(L3, Δ) | ρ ≥ 0.35 (H3) |
| **Bound validity rate** | Fraction of instances with τ ≤ 1.0 | ≥ 90% (allowing 10% numerical noise) |
| **Bound tightness** | Median τ across instances | τ_median ≤ 0.5 (bound is at most 2× loose) |
| **Conditional validity** | Validity rate for κ(A) ≤ 10³ | ≥ 95% |
| **Conditional validity** | Validity rate for κ(A) > 10³ | Report (expected lower) |

### 6.4 Per-Structure-Type Breakdown

Report L3 validation metrics separately for:

| Structure Type | Expected Behavior |
|----------------|-------------------|
| Block-angular | L3 should be tightest (cleanest decomposition) |
| Bordered-block-diagonal | L3 moderately tight (linking variables add slack) |
| Staircase | L3 moderately tight (sequential coupling) |
| Dual-block-angular | L3 may be loose (dual structure doesn't map cleanly to primal partition) |
| No structure detected | L3 is vacuous (crossing edges ≈ all edges) |

### 6.5 Failure Mode Catalog

| Failure Mode | Detection | Expected Frequency | Mitigation |
|--------------|-----------|-------------------|------------|
| **Shadow prices unavailable** | LP solver returns unbounded dual or infeasible | < 5% of MIPLIB | Exclude from L3 analysis; report count |
| **Numerical bound violation** (τ > 1) | Direct computation | 5–10% (from κ-related issues) | Flag instances; report separately for high-κ vs low-κ |
| **Vacuous bound** (L3 ≫ z*_LP) | L3 / |z*_LP| > 100 | 20–30% (ill-conditioned instances) | Report as expected failure; discuss T2 vacuity |
| **Zero crossing edges** | E_cross = ∅ | < 5% (perfect decomposition) | L3 = 0, Δ should ≈ 0; verify |
| **Degenerate shadow prices** | Many π_e = 0 | 10–20% | L3 is artificially tight; report as caveat |

### 6.6 Benders vs. DW Specialization (L3-C)

Compute L3 separately for Benders and DW specializations:
- **L3-C Benders**: Crossing edges are complicating variables; weights reflect variable-side coupling.
- **L3-C DW**: Crossing edges are linking constraints; weights reflect constraint-side coupling.

Report correlation of each specialization with the corresponding decomposition method's actual gap.

---

## 7. Census Design (Four Tiers)

### 7.1 Tier Overview

| Tier | n | Purpose | Deadline | Compute Budget |
|------|---|---------|----------|----------------|
| **Pilot** | 50 | Kill-gate G1 evaluation; pipeline debugging | Week 2 | ~8 core-hours |
| **Dev** | 200 | Feature engineering; hyperparameter tuning; kill-gate G3 | Week 8 | ~80 core-hours |
| **Paper** | 500 | All paper results; hypothesis tests H1–H7 | Week 14 | ~500 core-hours |
| **Artifact** | 1,065 | Spectral annotations only (no solver runs); released dataset | Week 16 | ~9 core-hours (spectral only) |

### 7.2 Stratification Protocol

Instances are stratified along three axes:

**Axis 1: Structure Type** (detected by GCG + heuristic detectors)

| Category | Description | Target % in Sample |
|----------|-------------|-------------------|
| Block-angular | Clean block structure with linking constraints | ~15% |
| Bordered-block-diagonal (BBD) | Blocks with shared variables | ~12% |
| Staircase | Sequential/temporal coupling | ~8% |
| Dual-block-angular | Block structure in dual | ~5% |
| No exploitable structure | Dense coupling; decomposition unlikely to help | ~60% |

**Axis 2: Instance Size**

| Size Class | Criterion (n + m) | Target % |
|------------|-------------------|----------|
| Small | < 1,000 | ~25% |
| Medium | 1,000 – 10,000 | ~40% |
| Large | 10,000 – 100,000 | ~25% |
| Very large | > 100,000 | ~10% |

**Axis 3: Conditioning**

| Conditioning | Criterion (κ(A)) | Target % |
|-------------|-------------------|----------|
| Well-conditioned | κ ≤ 10³ | ~30% |
| Moderate | 10³ < κ ≤ 10⁶ | ~40% |
| Ill-conditioned | κ > 10⁶ | ~30% |

### 7.3 Sampling Procedure

```
1. Compute features + structure detection for all 1,065 MIPLIB instances (~9 hours)
2. Assign each instance to a stratum (structure × size × conditioning)
3. For each tier, sample proportionally to stratum sizes, with:
   - Minimum 2 instances per non-empty stratum in pilot
   - Minimum 5 instances per non-empty stratum in dev
   - Minimum 10 instances per non-empty stratum in paper
4. If any stratum is too small (< minimum), take all available and
   redistribute surplus slots to nearest stratum
5. Record random seed for reproducibility
```

### 7.4 Tier Progression Gates

| Transition | Gate | Criterion | Action if Failed |
|------------|------|-----------|-----------------|
| Pilot → Dev | G1 | ρ(δ²/γ², Δ) ≥ 0.40 | ABANDON spectral ratio as predictor |
| Pilot → Dev | G-pipeline | ≥ 80% of solver runs complete without error | Debug pipeline before scaling |
| Dev → Paper | G3 | BA(SPEC-8) > BA(SYNT-25) on dev set | ABANDON spectral feature contribution |
| Dev → Paper | G0 | max R²(spectral ~ syntactic) < 0.70 | ABANDON redundant spectral features |
| Paper → Artifact | G4 | At least one primary metric meets target | Proceed to artifact release |

### 7.5 Compute Budget Breakdown (Paper Tier, 500 Instances)

| Component | Per Instance | Total (500) | Parallelizable |
|-----------|-------------|-------------|----------------|
| Spectral feature extraction | ~30s | ~4.2 hours | Yes (embarrassingly parallel) |
| Monolithic SCIP (900s cutoff) | 900s worst case | ~125 hours | Yes |
| GCG (900s cutoff) | 900s worst case | ~125 hours | Yes |
| SCIP-Benders (900s cutoff) | 900s worst case | ~125 hours | Yes |
| ML pipeline (nested CV) | ~5 min total | ~2.5 hours | Partially |
| **Total** | | **~380 hours** | **~5 days on 4 cores** |

---

## 8. Reporting Standards

### 8.1 Confidence Intervals

All point estimates must be accompanied by 95% confidence intervals:

| Estimate Type | CI Method |
|---------------|-----------|
| Classification accuracy / balanced accuracy | Bootstrap BCa (10,000 resamples) |
| Spearman ρ | Fisher z-transform |
| ICC | Analytical (F-distribution based) |
| Precision / Recall | Exact Clopper-Pearson |
| R² | Bootstrap BCa |
| Feature importance | Bootstrap (1,000 resamples of permutation importance) |

### 8.2 Effect Sizes

| Comparison | Effect Size Measure |
|------------|-------------------|
| Accuracy differences | Cohen's g (for McNemar's) |
| Correlation tests | ρ itself (with CI) |
| Multi-config comparison | Kendall's W (for Friedman) |
| Regression | R², adjusted R², Cohen's f² |

### 8.3 Dolan–Moré Performance Profiles

For solver-level comparisons (B0 vs B1 vs B2 and oracle-recommended), report Dolan–Moré performance profiles:

- **x-axis**: Performance ratio τ = t_solver / t_best (where t is the metric of interest: time-to-best-bound, final gap, dual bound quality).
- **y-axis**: Fraction of instances where solver achieves ratio ≤ τ.
- **Generate three profile sets**:
  1. Dual bound quality at T = 900s
  2. Primal-dual gap at T = 900s  
  3. Time to achieve 5% optimality gap (for instances solved within 3600s)

### 8.4 Confusion Matrices

Report for each classifier configuration:

- **3×3 confusion matrix** (Benders × DW × neither) with raw counts.
- **Normalized confusion matrix** (row-normalized: each row sums to 1).
- **Per-class precision, recall, F1**.
- **Overall accuracy, balanced accuracy, macro-F1, weighted-F1**.

### 8.5 Calibration Plots

For each classifier in the ablation:

- **Reliability diagram**: 10-bin calibration plot (predicted probability vs. observed frequency) for each class.
- **Expected Calibration Error (ECE)**: Single scalar summary.
- **Post-calibration**: Apply Platt scaling on inner fold; report before/after ECE.

### 8.6 Feature Importance Visualization

- **Permutation importance**: Bar chart with 95% CI for top-15 features in COMB-ALL.
- **SHAP summary plot**: Beeswarm plot for the best-performing classifier on COMB-ALL.
- **Feature interaction**: SHAP interaction values for top-5 spectral features.

### 8.7 Tables Required in Paper

| Table | Content |
|-------|---------|
| **Table 1** | Feature definitions (all 43 features with formulas) |
| **Table 2** | Ablation results: 6 configs × 3 classifiers, balanced accuracy ± CI |
| **Table 3** | McNemar's pairwise p-values (upper triangle) and Cohen's g (lower triangle) |
| **Table 4** | Per-structure-type balanced accuracy for SPEC-8, SYNT-25, COMB-ALL |
| **Table 5** | Hypothesis test summary (H1–H7: statistic, p-value, CI, decision) |
| **Table 6** | L3 validation: correlation, validity rate, tightness by structure type |
| **Table 7** | Futility predictor: precision, recall, F1 at multiple thresholds |
| **Table 8** | Census summary statistics: instance counts by stratum, label distribution |
| **Table 9** | Baseline solver performance: median gap, time, bounds by structure type |

### 8.8 Figures Required in Paper

| Figure | Content |
|--------|---------|
| **Fig 1** | System architecture diagram (spectral engine → features → classifier → recommendation) |
| **Fig 2** | Dolan–Moré performance profiles (3 panels: dual bound, gap, time) |
| **Fig 3** | Scatter: δ²/γ² vs. bound degradation with LOWESS and ρ annotation |
| **Fig 4** | Scatter: L3 bound vs. actual gap, colored by structure type |
| **Fig 5** | Confusion matrices for SPEC-8, SYNT-25, COMB-ALL (3 panels) |
| **Fig 6** | Feature importance (permutation + SHAP) for COMB-ALL |
| **Fig 7** | Calibration reliability diagrams (3 panels: one per class) |
| **Fig 8** | Critical difference diagram from Friedman/Nemenyi test |
| **Fig 9** | Label stability: Sankey diagram showing label flow across cutoffs |

---

## 9. Threats to Validity

### 9.1 Internal Validity

| Threat | Description | Mitigation |
|--------|-------------|------------|
| **Data leakage** | Feature selection or hyperparameter tuning on test data | Strict nested CV: feature selection inside inner loop |
| **Label noise** | Solver non-determinism (thread scheduling, memory layout) | Single-thread execution; verify ≤ 1% bound variation across 3 seeds on pilot |
| **Overfitting** | Small dataset (500 instances) with moderately high feature count (43) | Feature-count-controlled ablation; L1 regularization; RANDOM baseline as sanity check |
| **Selection bias in stratification** | Non-random subset of MIPLIB may not represent real-world MIPs | Document stratification procedure; release full 1,065 spectral annotations for external validation |
| **Solver version sensitivity** | Results may not reproduce with different SCIP/GCG versions | Pin exact solver versions, build hashes, and compiler flags in artifact |
| **Evaluation circularity** | Spectral features used for both detection and prediction | External baselines (GCG, SCIP-Benders) provide labels independently of spectral analysis |
| **Multiple testing** | 7 hypotheses × multiple comparisons within ablation | Holm–Bonferroni correction; report both corrected and uncorrected p-values |

### 9.2 External Validity

| Threat | Description | Mitigation |
|--------|-------------|------------|
| **MIPLIB representativeness** | MIPLIB 2017 is curated; may not reflect industrial MIPs | Acknowledge limitation; note MIPLIB is the community standard benchmark |
| **Solver generalization** | Results tied to SCIP/GCG; may not transfer to Gurobi/CPLEX | Report feature-level findings (transferable) separately from solver-specific results |
| **Structure prevalence** | Only 10–25% of MIPLIB has exploitable structure | Report honestly; focus on these instances; discuss population base rate |
| **Scaling to larger instances** | Spectral computation may be infeasible for n > 10⁶ | Report compute time vs. instance size; discuss Lanczos-based approximations |
| **Temporal validity** | MIPLIB 2017 instances; new versions may differ | MIPLIB 2017 is the current standard; future benchmarks will require re-evaluation |

### 9.3 Construct Validity

| Threat | Description | Mitigation |
|--------|-------------|------------|
| **"Best decomposition" definition** | Dual bound improvement may not equal practical usefulness | Report multiple metrics (gap, time, bound); discuss when dual bound and primal performance diverge |
| **"Futility" threshold** | 1% improvement cutoff is arbitrary | Sensitivity analysis at {0.5%, 1%, 2%, 5%} thresholds |
| **Wall-clock fairness** | GCG and SCIP-Benders have different overhead | Report setup time separately; normalize to solving time |
| **Spectral feature semantics** | Features may capture problem size rather than structure | Partial correlation controlling for size; feature-count-controlled ablation |
| **Equilibration choice** | Three equilibrations may not cover all scaling regimes | ICC analysis (H5); report which equilibration gives best downstream accuracy |
| **Partition quality proxy** | Spectral ratio δ²/γ² is a proxy, not a direct measure | Report actual partition quality (normalized cut) alongside spectral ratio |

---

## 10. Metrics Table

| # | Metric Name | Target Value | Measurement Procedure | Kill Condition | Recovery Action |
|---|------------|-------------|----------------------|----------------|-----------------|
| **M1** | Spearman ρ(δ²/γ², Δ) | ρ ≥ 0.40 | Compute on pilot-50, confirm on dev-200 | ρ < 0.40 on pilot (G1) | Abandon spectral ratio; pivot to individual spectral features only |
| **M2** | max R²(spectral ~ syntactic) | R² < 0.70 | OLS regression, 8 models on paper-500 | R² ≥ 0.70 for ≥5/8 features (G0) | Remove redundant features; report reduced SPEC set |
| **M3** | Δ_BA (SPEC-8 − SYNT-25) | > 0 | McNemar's test in nested CV on paper-500 | Δ_BA ≤ 0 after Holm correction (G3) | Abandon spectral contribution; publish as census-only paper |
| **M4** | Balanced accuracy (COMB-ALL) | ≥ 60% | Nested CV on paper-500 | BA < 40% (worse than random by margin) | Reassess label quality; investigate feature engineering |
| **M5** | Futility precision | ≥ 0.80 | Nested CV, threshold-tuned, on paper-500 | Precision < 0.60 | Demote futility predictor to appendix; report negative result |
| **M6** | Mean ICC (scaling robustness) | ≥ 0.85 | ICC(3,1) across 3 equilibrations, 500 instances | Mean ICC < 0.70 | Flag unstable features; recommend specific equilibration |
| **M7** | L3 bound validity rate | ≥ 90% | τ = Δ/L3 ≤ 1.0, paper-500 | Validity < 75% | Investigate numerical issues; restrict L3 claims to well-conditioned subset |
| **M8** | L3 Spearman ρ | ρ ≥ 0.35 | Correlation on paper-500 | ρ < 0.20 | Demote L3 to theoretical motivation only; do not claim predictive value |
| **M9** | Label stability (κ across cutoffs) | κ ≥ 0.60 | Cohen's κ(T=300, T=3600) | κ < 0.40 | Use only T=3600 labels; discuss instability |
| **M10** | Per-structure-type Δ_BA | > 0 for ≥ 2/4 classes | McNemar's per class, paper-500 | Δ_BA ≤ 0 for all 4 classes | Report as negative finding; spectral features help globally but not per-structure |
| **M11** | Pipeline completion rate | ≥ 80% of solver runs | Count successful runs / total attempted | < 80% on pilot | Debug infrastructure before scaling |
| **M12** | Spectral extraction time | ≤ 60s median per instance | Wall-clock timing, 500 instances | Median > 300s | Implement Lanczos approximation or subsample large instances |

### Kill Gate Summary

| Gate | Metric(s) | Tier | Deadline | Consequence |
|------|----------|------|----------|-------------|
| **G0** | M2 (R² < 0.70) | Dev-200 | Week 8 | ABANDON spectral features if redundant |
| **G1** | M1 (ρ ≥ 0.40) | Pilot-50 | Week 2 | ABANDON spectral ratio as predictor |
| **G3** | M3 (Δ_BA > 0) | Dev-200 | Week 8 | ABANDON spectral feature contribution |
| **G4** | M3 ∧ M4 ∧ M7 | Paper-500 | Week 14 | If ALL three fail: ABANDON paper |
| **G5** | Draft quality | Paper draft | Week 18 | If fundamental problems: ABANDON |

### Metric Dependencies

```
M11 (pipeline) ──→ M9 (labels) ──→ M3, M4, M5, M10 (classification)
                                 ↗
M12 (timing) ──→ M1 (ρ) ──→ M6 (ICC) ──→ M2 (R²) ──→ M3 (ablation)
                         ↘
                          M7, M8 (L3 validation)
```

---

## Appendix A: Reproducibility Checklist

| Item | Specification |
|------|---------------|
| **Random seeds** | Primary: 42. Sensitivity: {42, 123, 456, 789, 2024} |
| **Software versions** | SCIP 8.x.y, GCG 3.x.y, Python 3.11+, scikit-learn 1.4+, XGBoost 2.0+, NumPy/SciPy pinned |
| **Hardware** | Report: CPU model, cores, RAM, OS. Target: reproducible on 4-core laptop with 16GB RAM |
| **Data** | MIPLIB 2017 benchmark set (publicly available). Instance list + stratification published |
| **Code** | Full pipeline released: feature extraction, labeling, ML training, evaluation scripts |
| **Intermediate artifacts** | Spectral features for all 1,065 instances released as CSV |
| **Statistical code** | All hypothesis tests, CIs, and plots reproducible from released scripts |
| **Time budget** | Total wall-clock time reported per tier |

## Appendix B: Feature Definitions Reference

### Spectral Features (SPEC-8)

| ID | Name | Formula | Interpretation |
|----|------|---------|----------------|
| s1 | Fiedler value | δ₁ = λ₂(L) | Algebraic connectivity; small → easy to partition |
| s2 | Second spectral gap | δ₂ = λ₃(L) − λ₂(L) | Gap stability; large → robust 2-partition |
| s3 | Spectral gap | γ = λ₂(L) / λ_max(L) | Normalized connectivity |
| s4 | Cheeger constant | Φ = min_S |∂S| / min(vol(S), vol(V\S)) | Partition quality (approximated via Fiedler vector) |
| s5 | Eigenvector localization | ξ = (Σ v_i⁴) / (Σ v_i²)² | Inverse participation ratio of Fiedler vector; high → localized structure |
| s6 | Coupling energy | η = Σ_{(i,j)∈E} w_{ij}(v_i − v_j)² | Quadratic form; energy of Fiedler vector on graph |
| s7 | Gap ratio | σ = δ₂² / γ² | Key predictor from T2; controls partition quality degradation |
| s8 | Spectral condition | κ_L = λ_max(L) / λ₂(L) | Spectral condition number of Laplacian |

### Syntactic Features (SYNT-25)

| ID | Name | Description |
|----|------|-------------|
| t1–t3 | n, m, nnz | Variable count, constraint count, nonzeros |
| t4 | density | nnz / (n · m) |
| t5–t7 | obj_range, rhs_range, coeff_range | log₁₀(max/min) of absolute nonzero values |
| t8–t10 | n_int, n_bin, n_cont | Integer, binary, continuous variable counts |
| t11–t12 | frac_int, frac_bin | Fractions of integer and binary variables |
| t13–t17 | eq_count, leq_count, geq_count, range_count, free_count | Constraint type counts |
| t18–t21 | var_degree_{mean,std,skew,max} | Variable degree statistics |
| t22–t25 | con_degree_{mean,std,skew,max} | Constraint degree statistics |

---

*End of evaluation strategy. This document is sufficient for full replication of all experimental results reported in the paper.*

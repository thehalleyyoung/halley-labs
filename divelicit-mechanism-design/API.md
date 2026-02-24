# DivFlow API Reference

Only implemented and tested features are listed.

## `src/transport.py` — Optimal Transport

### `sinkhorn_divergence(X, Y, reg=0.1, n_iter=50) → float`
Debiased Sinkhorn divergence between empirical measures on X and Y.
Non-negative, zero iff μ = ν, metrizes weak convergence.

### `sinkhorn_potentials(X, Y, reg=0.1, n_iter=100, tol=1e-8) → (f, g)`
Compute Sinkhorn dual potentials. `g[j]` measures how underserved reference point j is.

### `sinkhorn_candidate_scores(candidates, history, reference, reg=None, n_iter=50) → ndarray`
Score each candidate by marginal Sinkhorn divergence reduction. This is the core DivFlow scoring function.

### `sinkhorn_distance(a, b, M, reg=0.1, n_iter=100, tol=1e-8) → float`
Entropic optimal transport cost between marginals a, b with cost matrix M.

### `cost_matrix(X, Y, metric="euclidean") → ndarray`
Pairwise cost matrix. Supports `"euclidean"`, `"sqeuclidean"`, `"cosine"`.

## `src/algebraic_proof.py` — Formal Algebraic Proof

### `verify_algebraic_proof(n=10, d=8, k=3, reg=0.1, quality_weight=0.3, n_perturbations=100) → AlgebraicProofResult`
Algebraically verify that Sinkhorn-based welfare is exactly quasi-linear: W(S) = h_i(S, q_{-i}) + λ·q_i·𝟙[i∈S]. Tests across `n_perturbations` random quality perturbations to confirm max decomposition error is at machine precision.

### `verify_exponential_structure(n=8, d=4, reg=0.1) → dict`
Verify the exponential structure of optimal transport plan: π*_ij = exp((f_i + g_j - C_ij)/ε). Returns reconstruction error and verification status.

### `verify_payment_independence(n=10, d=8, k=3, reg=0.1, n_tests=50) → dict`
Verify that VCG payments are independent of agent's own quality report. Perturbs quality and checks payment invariance.

### `full_algebraic_verification(n=10, d=8, k=3, reg=0.1) → AlgebraicProofResult`
Run all three verifications (quasi-linearity, exponential structure, payment independence) and return combined result.

## `src/ic_analysis.py` — IC Violation Analysis

### `analyze_ic_violations(n=15, d=8, k=3, reg=0.1, quality_weight=0.3, n_random_trials=50, n_adversarial_trials=20) → dict`
Root-cause analysis of IC violations identifying which VCG conditions fail.
Returns `vcg_condition_analysis` (C1/C2/C3 status), `violation_details` (list of ViolationDetail), and `violation_taxonomy` (Type A/B/C counts).

### `multiple_testing_correction(p_values, method="bh", alpha=0.05) → dict`
Multiple testing correction. `method="bh"` for Benjamini-Hochberg (FDR), `method="bonferroni"` for family-wise error rate.

### `enhanced_bootstrap_ci(values, n_seeds=20, n_bootstrap=2000, alpha=0.05) → dict`
Bootstrap CI with stability diagnostics across multiple seeds. Returns mean, CI endpoints, and stability metrics.

## `src/z3_verification.py` — Z3 SMT IC Verification

### `verify_ic_z3(embs, quals, k=2, reg=0.1, quality_weight=0.3, grid_resolution=5, timeout_ms=30000) → Z3VerificationResult`
Verify IC conditions using Z3 SMT solver. Exhaustive for n≤12, grid-sampled for larger instances. Returns counterexamples and per-agent certification status.

### `verify_ic_z3_refined(embs, quals, k=2, reg=0.1, quality_weight=0.3, grid_resolution=15, timeout_ms=60000) → dict`
Refined verification with higher grid resolution and Lipschitz soundness analysis. Returns certification rate with CIs, Lipschitz constant estimate, soundness gap (L·h), per-agent results, and uncertified agent characterization.

### `verify_ic_regions(embs, quals, k=2, reg=0.1, quality_weight=0.3, n_regions=8, timeout_ms=10000) → dict`
Regional IC certification. Partitions quality space into regions and certifies each independently.

## `src/coverage.py` — Coverage Certificates

### `estimate_coverage(points, epsilon, dim=None) → CoverageCertificate`
Coverage using metric entropy bounds with effective dimension adaptation.
Returns certificate with Clopper-Pearson CIs and explicit constants (C_cov=3).

### `epsilon_net_certificate(points, reference, epsilon, delta=0.05) → CoverageCertificate`
ε-net certificate with dimension-adapted projection and exact CIs.

### `clopper_pearson_ci(k, n, alpha=0.05) → (lower, upper)`
Exact binomial confidence interval.

### `wilson_ci(k, n, alpha=0.05) → (lower, upper)`
Wilson score confidence interval (more accurate for small n or extreme p).

### `bootstrap_ci(values, n_bootstrap=2000, alpha=0.05, seed=42) → (mean, ci_lo, ci_hi)`
Bootstrap confidence interval for mean.

### `cohens_d(group1, group2) → float`
Cohen's d effect size between two groups.

### `permutation_test(group1, group2, n_permutations=10000, seed=42) → float`
Two-sample permutation test p-value (two-sided).

## `src/composition_theorem.py` — Composition Theorem & IC

### `verify_ic_with_ci(embs, quals, selected, payments, select_fn, k, n_trials=1000) → EpsilonICResult`
Verify IC with Clopper-Pearson CIs. Returns violation rate, CI, characterization.

### `composition_theorem_check(embs, quals, k, reg=0.1) → dict`
Empirically verify composition theorem conditions (diminishing returns + quasi-linearity).

### `verify_composition_formal(embs, quals, k, quality_weight=0.3, reg=0.1, n_perturbations=200) → dict`
Formal verification of the full composition theorem. Returns results for all five parts: (a) quasi-linearity, (b) exact VCG DSIC, (c) ε-submodularity with slack bound, (d) greedy ε-IC bound, (e) violation probability bound.

### `adversarial_ic_test(embs, quals, select_fn, k, n_trials=200) → dict`
Adversarial IC testing with strategic deviations (extreme + targeted values).

### `ICViolationMonitor(threshold=0.20, window_size=100)`
Runtime monitor for IC violations with Clopper-Pearson and Wilson CIs.

## `src/sensitivity_analysis.py` — Hyperparameter Sensitivity

### `full_sensitivity_analysis(embs, quals, topics, k=5, quality_weight=0.3, reg=0.1, n_trials=100) → dict`
Complete sensitivity analysis across Sinkhorn ε, quality weight λ, and selection size k. Returns violation rates with CIs, topic coverage, and welfare at each sweep point.

### `sensitivity_sinkhorn_epsilon(embs, quals, topics, k=5, epsilons=None, n_trials=200) → list[SensitivityResult]`
Sweep Sinkhorn regularization parameter.

### `sensitivity_quality_weight(embs, quals, topics, k=5, lambdas=None, n_trials=200) → list[SensitivityResult]`
Sweep quality weight parameter.

### `sensitivity_selection_size(embs, quals, topics, k_values=None, n_trials=200) → list[SensitivityResult]`
Sweep selection size k.

## `src/mechanism.py` — Mechanism Design

### `VCGMechanism(embs, k, reg=0.1, quality_weight=0.3)`
VCG mechanism with Sinkhorn welfare. Methods: `select(quals)`, `compute_payments(quals, selected)`.

### `FlowMechanism(embs, k, reg=0.1, quality_weight=0.3)`
DivFlow greedy selection (no payments).

### `DirectMechanism(embs, k, reg=0.1, quality_weight=0.3)`
Direct revelation mechanism with top-k selection.

### `MMRMechanism(embs, k, lambda_=0.5)`
Maximal Marginal Relevance selection.

## `src/scoring_rules.py` — Proper Scoring Rules

### `LogarithmicRule()` — S(p,y) = log p(y)
### `BrierRule()` — S(p,y) = 2p(y) - ||p||²
### `SphericalRule()` — S(p,y) = p(y) / ||p||
### `CRPSRule()` — Continuous Ranked Probability Score
### `EnergyAugmentedRule(base_rule, energy_fn, lambda_=0.1)` — preserves properness
### `verify_properness(rule, p, q, n_samples=10000) → bool`

## `src/kernels.py` — Kernel Functions

### `RBFKernel(bandwidth=1.0)`
### `AdaptiveRBFKernel(initial_bandwidth=1.0)` — LOO-CV bandwidth learning
### `MultiScaleKernel(bandwidths=None, n_scales=5)` — learned scale weights
### `ManifoldAdaptiveKernel(bandwidth=1.0, n_neighbors=10)` — local PCA geodesic

## `src/dpp.py` — Determinantal Point Processes

### `greedy_map(L, k) → list` — Greedy MAP inference with Cholesky updates

## Data Classes

### `AlgebraicProofResult`
- `verified`, `max_error`, `n_perturbations`, `exponential_structure`, `payment_independence`, `details`

### `VCGConditionAnalysis`
- `c1_quasilinearity` (exact status), `c2_welfare_maximization` (approx ratio), `c3_payment_computation` (verified)

### `ViolationDetail`
- `agent_id`, `violation_type` (A/B/C), `utility_gain`, `true_quality`, `reported_quality`

### `Z3VerificationResult`
- `verified`, `n_violations`, `counterexamples`, `agent_certificates`, `time_seconds`

### `CoverageCertificate`
- `coverage_fraction`, `confidence`, `n_samples`, `epsilon_radius`, `ci_lower`, `ci_upper`

### `EpsilonICResult`
- `violation_rate`, `ci_lower`, `ci_upper`, `n_violations`, `n_trials`, `max_utility_gain`

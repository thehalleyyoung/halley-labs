# CSAT API Reference

## regime/

### `StickyHDPHMM`
Bayesian nonparametric HMM that infers the number of regimes from data.

```python
__init__(K_max=5, alpha=1.0, gamma=5.0, kappa=10.0, n_iter=500, burn_in=100, emission="gaussian", random_state=None)
fit(X: NDArray, verbose=False) -> StickyHDPHMM
predict(X: NDArray) -> NDArray
convergence_diagnostics() -> dict
```

Attributes after fit: `states_`, `A_`, `means_`, `covars_`, `pi_`, `beta_`, `diagnostics_`.

### `StudentTEmission`
Student-t emission model with Normal-Inverse-χ² conjugate prior.

```python
__init__(dim, nu=5.0, mu_0=None, kappa_0=1.0, nu_0=None, Psi_0=None)
log_marginal_likelihood(x: NDArray) -> float
log_pdf(x: NDArray, mu: NDArray, Sigma: NDArray) -> float
sample_posterior(rng: Generator) -> Tuple[NDArray, NDArray]
add_obs(x: NDArray) -> None
remove_obs(x: NDArray) -> None
copy() -> StudentTEmission
```

### `EmissionModelSelector`
Automatic Gaussian vs Student-t selection via BIC.

```python
__init__(nu=5.0)
select(data: NDArray) -> str                    # Returns "gaussian" or "student_t"
compare(data: NDArray) -> List[_ModelComparisonRow]
kurtosis_test(data: NDArray) -> dict            # Excess kurtosis + p-values
```

## causal/

### `HSIC` / `GaussianKernel`
Kernel-based independence testing with Gaussian, polynomial, or linear kernels.

```python
GaussianKernel(bandwidth=None)
GaussianKernel.compute(X, Y=None) -> NDArray
```

### `PCAlgorithm` (via `pc_algorithm.py`)
Constraint-based causal discovery with HSIC-based CI tests and Meek orientation rules.

Key classes: `SkeletonResult`, `_OrientationHelper`.

## invariance/

### `SCITAlgorithm`
Sequential Causal Invariance Test with anytime-valid e-values.

```python
__init__(alpha=0.05, e_value_type="likelihood_ratio", min_samples_per_regime=30, ...)
fit(data: NDArray, regimes: NDArray, edges: List[Tuple[str, str]], feature_names=None) -> SCITResult
```

### `EBHProcedure`
E-BH multiple testing correction for e-values.

```python
__init__(alpha=0.05)
apply(e_values: Dict, ...) -> Dict[Tuple, EdgeClassification]
```

## coupled/

### `SpuriousFixedPointDetector`
Monitor EM convergence quality and detect spurious fixed points. Tracks contraction rate, Lyapunov violations, and eigenvalue analysis of the observed information ratio. Compares fixed points across restarts to identify structurally distinct solutions at similar log-likelihoods.

```python
__init__(ll_tol: float = 1e-3, structure_tol: float = 0.1)
record(params: NDArray, log_likelihood: float) -> Tuple[float, bool]   # Returns (contraction_rate, lyapunov_ok)
information_ratio_eigenvalues(params: NDArray) -> NDArray
```

Properties: `estimated_contraction_rate` (median), `lyapunov_violations` (count), `contraction_rates` (list).

```python
# Static method: compare fixed points across multi-restart runs
SpuriousFixedPointDetector.compare_fixed_points(
    models: List[CoupledInference],
    ll_tol: float = 1e-3,
    structure_tol: float = 0.1,
) -> Dict[str, Any]   # Keys: n_distinct, n_spurious_candidates, groups
```

### `CoupledInference`
EM alternation between Sticky HDP-HMM regime estimation and PC-HSIC causal discovery.

```python
__init__(n_regimes=4, alpha_ci=0.05, max_cond_size=3, sticky_kappa=50.0, alpha_dp=5.0, seed=None)
fit(data: NDArray, max_iter=50, tol=1e-4, warm_start=False, verbose=False) -> CoupledInference
regime_posteriors() -> NDArray
per_regime_dags() -> Dict[int, nx.DiGraph]
invariant_edges() -> set
```

Properties:

```python
convergence_diagnostics -> Optional[Dict[str, Any]]      # Diagnostics from the last fit()
multi_restart_diagnostics -> Optional[Dict[str, Any]]    # Diagnostics if fit_multi_restart() was used
```

### `ConvergenceAnalyzer`
EM convergence diagnostics: monotonicity, contraction rate, Lyapunov function.

```python
__init__(rtol=1e-5, patience=5, min_iter=3)
record(iteration: int, ll: float, ...) -> None
check_convergence() -> ConvergenceDiagnostics
estimate_rate() -> Optional[float]
lyapunov_decreasing(strict=False) -> bool
```

## shield/

### `PosteriorPredictiveShield`
Safety shield: permits action a in state s iff P_posterior(φ | s,a) ≥ 1 − δ.

```python
__init__(n_states, n_actions, delta=0.05, horizon=10, mode=ShieldMode.PRE,
         n_posterior_samples=200, cache_size=10000, adaptive_delta: bool = False)
add_spec(spec, name: str, weight=1.0) -> None
synthesize() -> None
get_permitted_actions(state: int) -> NDArray     # Boolean mask
set_prior(prior_counts: NDArray) -> None
update_posterior(transitions: NDArray) -> None
graceful_degradation(state: NDArray, state_index: Optional[int] = None) -> ShieldResult
```

**`adaptive_delta`**: When `True`, δ scales with posterior concentration. Higher concentration (more data) relaxes δ toward `2*delta` for a more permissive shield; low concentration keeps the original conservative δ.

**`graceful_degradation()`**: Unlike `query()`, always returns ≥ 1 permitted action. If no action meets the safety threshold, permits the action with the highest safety probability and emits a warning.

### `BoundedDrawdownSpec` / `PositionLimitSpec`
Safety specifications for shield synthesis (in `safety_specs.py`).

### `BoundedLivenessSpec`
Base class for G(trigger → F[0,H] recovery) liveness specifications.

```python
__init__(name: str, horizon: int)
trigger_predicate(state: Dict[str, float]) -> bool   # Abstract
recovery_predicate(state: Dict[str, float]) -> bool   # Abstract
evaluate_trajectory(trajectory: List[Dict]) -> TrajectoryResult
to_ltl() -> LTLFormula
to_ltl_formula() -> str
```

### `DrawdownRecoverySpec`
```python
__init__(max_drawdown=0.1, recovery_fraction=0.5, horizon=20)
```

### `LossRecoverySpec`
```python
__init__(max_loss=0.05, recovery_fraction=0.5, horizon=10)
```

### `PositionReductionSpec`
```python
__init__(max_position=1.0, reduction_fraction=0.5, horizon=5)
```

### `RegimeTransitionSpec`
```python
__init__(horizon=10)
```

### `BoundedLivenessLibrary`
Pre-configured spec suites.

```python
conservative_suite() -> List[BoundedLivenessSpec]    # Static
moderate_suite() -> List[BoundedLivenessSpec]         # Static
aggressive_suite() -> List[BoundedLivenessSpec]       # Static
from_config(config: dict) -> List[BoundedLivenessSpec]
```

### `PACBayesVacuityAnalyzer`
Concrete numerical analysis of PAC-Bayes bound as function of n and K.

```python
__init__(n_abstract_states_per_regime=140, n_actions=7, delta=0.05, vacuity_threshold=0.5)
estimate_kl_for_k(K, n_observations, prior_concentration=1.0, visit_fraction=0.3) -> float
compute_bound_curve(K, n_values: NDArray, bound_type="pac-bayes-kl", empirical_error=0.0) -> NDArray
```

### PAC-Bayes Bound Variants
`PACBayesBound`, `McAllesterBound`, `MaurerBound`, `CatoniBound`, `SequentialPACBayes`, `EmpiricalBernsteinPACBayes` — all in `pac_bayes.py`.

```python
compute_kl(prior_counts, posterior_counts, ...) -> float
compute_bound(prior_counts, posterior_counts, n, delta, ...) -> ShieldSoundnessCertificate
```

## verification/

### `StateAbstraction` (via `ConservativeOverapproximation`)
Hyperrectangular state discretisation with soundness certification.

```python
ConservativeOverapproximation(state_dim, n_actions, grid_sizes)
certify_soundness(concrete_mdp: ConcreteMDP) -> SoundnessCertificate
verify_and_refine(concrete_mdp, target_gap=0.01) -> SoundnessCertificate
check_interval_transitions(concrete_mdp) -> bool
```

Supporting: `AbstractionFunction`, `AbstractState`, `ConcreteMDP`, `discretize_state_space()`, `verify_overapproximation()`, `adaptive_refinement()`.

### `IntervalArithmetic`
Sound floating-point arithmetic with machine-epsilon inflation (ε = 2⁻⁵²).

```python
Interval(lo, hi)
Interval.point(x) -> Interval
Interval.exact(lo, hi) -> Interval
# Supports +, -, *, /, sqrt, exp, log, abs, hull, intersect, contains

IntervalVector(intervals: List[Interval])
IntervalVector.from_bounds(lo: NDArray, hi: NDArray) -> IntervalVector

IntervalMatrix(lo: NDArray, hi: NDArray)
IntervalMatrix.from_matrix(M: NDArray) -> IntervalMatrix

interval_matmul(A: IntervalMatrix, v: IntervalVector) -> IntervalVector
interval_matmul_matrix(A: IntervalMatrix, B: IntervalMatrix) -> IntervalMatrix
interval_polyval(coeffs, x: Interval) -> Interval
```

### `IndependentVerifier`
Standalone certificate re-derivation using only NumPy/SciPy (no causal_trading imports).

```python
__init__(tolerance=1e-6)
verify_pac_bayes_bound(prior_counts, posterior_counts, n, delta, claimed_bound, ...) -> VerificationReport
verify_composition(stage_errors: List[float], claimed_total: float) -> VerificationReport
verify_state_abstraction(concrete_transitions, abstract_transitions, ...) -> VerificationReport
full_audit(prior_counts, posterior_counts, n, delta, ...) -> VerificationReport
```

## proofs/

### `DecomposedCompositionTheorem`
Four-stage error decomposition replacing monolithic ε₁ + ε₂ bound.

```python
DecomposedCompositionTheorem(target_error=0.1, method="union")
verify(budget: PipelineErrorBudget) -> bool
certificate(budget: PipelineErrorBudget) -> DecomposedCertificate
sensitivity(budget, stage_name) -> float
improvement_potential(budget) -> Dict[str, float]
```

### `PipelineErrorBudget`
Tracks per-stage errors with measurement details.

```python
add_regime_error(purity, n_pred, n_true, ...) -> None
add_dag_error(precision, recall, ...) -> None
add_invariance_error(kl_divergence, bound, ...) -> None
add_shield_error(permissivity, n_active, ...) -> None
total_error(method="union") -> float          # "union" or "inclusion_exclusion"
dominant_stage() -> Tuple[str, float]
budget_allocation(target: float) -> Dict[str, float]
```

### `Certificate`
Generates and verifies safety certificates with assumptions, guarantees, and parameters.

```python
__init__(name, version="1.0")
add_assumption(name, description, verified=False) -> None
add_guarantee(name, description, bound=None) -> None
generate(shield, model=None) -> dict
verify() -> bool
```

## evaluation/

### `ErrorDecompositionExperiment`
Runs the four-stage error decomposition experiment.

```python
__init__(config: ErrorDecompositionConfig, seed=42)
run(data: NDArray, ground_truth: GroundTruth) -> ErrorDecompositionResults
sweep_sample_sizes(sizes: List[int], ...) -> List[SweepResult]
save_results(path: str) -> None
```

### `SensitivityAnalyzer`
Systematic hyperparameter sensitivity sweeps.

```python
__init__(data: NDArray, seed=42, n_repeats=1)
sweep_kappa(values: List[float]) -> List[SweepPoint]
sweep_emission_prior(values: List[float]) -> List[SweepPoint]
sweep_pac_bayes_prior(values: List[float]) -> List[SweepPoint]
sweep_delta(values: List[float]) -> List[SweepPoint]
sweep_ci_alpha(values: List[float]) -> List[SweepPoint]
full_sensitivity_report() -> SensitivityReport
```

### `SensitivityReport`
```python
most_sensitive_param() -> str
robust_range(param: str) -> Tuple[float, float]
to_latex_table() -> str
to_pgfplots_data() -> Dict[str, str]
save(path) -> None
load(path) -> SensitivityReport                # Class method
```

## portfolio/

### `ShieldedMeanVarianceOptimizer`
Mean-variance optimisation constrained to shield-permitted actions.

```python
__init__(risk_aversion=2.0, transaction_cost_model=None)
optimize(expected_returns, cov_matrix, shield_permitted) -> OptimizationResult
compute_efficient_frontier(expected_returns, cov_matrix, n_points=50) -> List[EfficientFrontierPoint]
backtest(data, shield, ...) -> dict
```

### `CausalFeatureSelector`
HSIC-Lasso and linear LASSO feature selection.

```python
__init__(method="hsic-lasso", alpha=0.05, n_features=None, lambda_reg=0.01, ...)
select(X: NDArray, y: NDArray, regimes=None) -> FeatureSelectionResult
get_invariant_features() -> List[int]
get_regime_specific_features(regime: int) -> List[int]
```

## market/

### `SyntheticMarketGenerator`
Generates synthetic regime-switching data with known ground truth.

```python
__init__(n_features=30, n_regimes=3, n_causal=5, T=5000, seed=42, ...)
generate(T=None) -> SyntheticDataset
get_ground_truth() -> GroundTruth
```

## monitoring/

### `PermissivityTracker`
Tracks shield permissivity over time.

```python
record(permitted: bool) -> None
current_permissivity() -> float
get_trend() -> float
get_severity() -> Optional[ShieldAlertSeverity]
```

## experiments/

### `run_multi_instrument_experiment`
Run coupled inference on multiple asset classes (equity, FX, crypto) and evaluate regime detection accuracy (ARI), causal discovery accuracy (SHD, precision, recall), PAC-Bayes bounds, and shield permissivity.

```python
# experiments/run_multi_instrument.py
run_multi_instrument_experiment() -> Dict[str, Any]
```

Outputs results to `experiments/results/multi_instrument.json` with per-asset metrics and convergence diagnostics.

```bash
python3 -m experiments.run_multi_instrument
```

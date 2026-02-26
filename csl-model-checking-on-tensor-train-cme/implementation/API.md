# TN-Check API Reference

---

## Table of Contents

- [Configuration (`tn_check.config`)](#configuration)
- [Tensor: MPS (`tn_check.tensor.mps`)](#mps--matrix-product-states)
- [Tensor: MPO (`tn_check.tensor.mpo`)](#mpo--matrix-product-operators)
- [Tensor: Operations (`tn_check.tensor.operations`)](#operations)
- [Tensor: Canonical (`tn_check.tensor.canonical`)](#canonical-forms)
- [Tensor: Decomposition (`tn_check.tensor.decomposition`)](#decomposition)
- [Tensor: Algebra (`tn_check.tensor.algebra`)](#algebra)
- [CME: Reaction Network (`tn_check.cme.reaction_network`)](#reaction-network)
- [CME: Compiler (`tn_check.cme.compiler`)](#compiler)
- [CME: Initial State (`tn_check.cme.initial_state`)](#initial-state)
- [Checker: CSL AST (`tn_check.checker.csl_ast`)](#csl-ast)
- [Checker: Model Checker (`tn_check.checker.model_checker`)](#model-checker)
- [Checker: Satisfaction (`tn_check.checker.satisfaction`)](#satisfaction)
- [Checker: Spectral (`tn_check.checker.spectral`)](#spectral-analysis)
- [Error: Certification (`tn_check.error.certification`)](#certification)
- [Error: Propagation (`tn_check.error.propagation`)](#propagation)
- [Verifier: Trace (`tn_check.verifier.trace`)](#trace)
- [Verifier: Checker (`tn_check.verifier.checker`)](#verifier-checker)
- [Diagnostics (`tn_check.diagnostics`)](#diagnostics)
- [Model Library (`tn_check.models.library`)](#model-library)
- [Experiments (`tn_check.experiments`)](#experiments)

---

## Configuration

**Module:** `tn_check.config`

**Enums:** `IntegratorType`, `CanonicalForm`, `OrderingMethod`, `CSLSemantics`.

**Sub-configs (dataclasses):** `TTConfig`, `CMEConfig`, `IntegratorConfig`, `CheckerConfig`, `ErrorConfig`, `OrderingConfig`, `AdaptiveConfig`, `EvaluationConfig`.

### `TNCheckConfig`
Master configuration aggregating all sub-configs.

| Method | Description |
|--------|-------------|
| `to_dict() -> dict` | Serialize to dictionary |
| `from_dict(cls, d) -> TNCheckConfig` | Deserialize from dictionary |
| `save(path)` | Save to JSON |
| `load(cls, path) -> TNCheckConfig` | Load from JSON |
| `from_env(cls) -> TNCheckConfig` | Create from environment variables |

```python
def default_config() -> TNCheckConfig          # Default configuration
def quick_config(max_bond_dim=50, max_copy_number=30, integrator="tdvp_2site") -> TNCheckConfig
```

---

## MPS — Matrix Product States

**Module:** `tn_check.tensor.mps`

### `MPS`

```python
MPS(cores: list[NDArray], canonical_form=CanonicalForm.NONE, orthogonality_center=None, copy_cores=True)
```

**Properties:** `num_sites`, `physical_dims`, `bond_dims`, `max_bond_dim`, `total_params`, `full_size`, `compression_ratio`, `dtype`.

**Methods:** `copy()`, `get_core(site)`, `set_core(site, core)`, `scale(factor)`, `negate()`, `bond_spectrum(bond)`, `entanglement_entropy(bond)`, `renyi_entropy(bond, alpha=2.0)`, `invalidate_cache()`, `info()`, `validate()`.

### Factory Functions

```python
random_mps(num_sites, physical_dims, bond_dim, normalize=True, seed=None, dtype=np.float64) -> MPS
zero_mps(num_sites, physical_dims) -> MPS
ones_mps(num_sites, physical_dims) -> MPS
unit_mps(num_sites, physical_dims, indices) -> MPS           # Rank-1 basis vector
product_mps(vectors: list[NDArray]) -> MPS                   # Rank-1 from per-site vectors
uniform_mps(num_sites, physical_dims) -> MPS                 # Normalized uniform
characteristic_mps(num_sites, physical_dims, site, predicate_mask) -> MPS  # Single-site indicator
multi_site_characteristic_mps(num_sites, physical_dims, site_masks) -> MPS # Multi-site indicator
threshold_mps(num_sites, physical_dims, site, threshold, direction="greater") -> MPS
interval_mps(num_sites, physical_dims, site, low, high) -> MPS
```

---

## MPO — Matrix Product Operators

**Module:** `tn_check.tensor.mpo`

### `MPO`

```python
MPO(cores: list[NDArray], copy_cores=True)
```

**Properties:** `num_sites`, `physical_dims_in`, `physical_dims_out`, `bond_dims`, `max_bond_dim`, `total_params`, `is_square`, `dtype`.

**Methods:** `copy()`, `get_core(site)`, `set_core(site, core)`, `scale(factor)`, `compress(max_bond_dim=None, tolerance=1e-12)`, `check_column_sum_zero(tol)`, `check_metzler(tol)`, `check_hermitian(tol)`, `validate()`.

### Factory Functions

```python
identity_mpo(num_sites, physical_dims) -> MPO
zero_mpo(num_sites, physical_dims) -> MPO
scalar_mpo(num_sites, physical_dims, scalar) -> MPO         # scalar * I
random_mpo(num_sites, physical_dims, bond_dim, seed=None, hermitian=False) -> MPO
diagonal_mpo(diagonal_mps: MPS) -> MPO
creation_mpo(num_sites, physical_dims, site) -> MPO          # Bosonic creation at site
annihilation_mpo(num_sites, physical_dims, site) -> MPO      # Bosonic annihilation at site
number_mpo(num_sites, physical_dims, site) -> MPO            # Number operator at site
shift_mpo(num_sites, physical_dims, site, shift=1) -> MPO
propensity_diagonal_mpo(num_sites, physical_dims, site, propensity_values) -> MPO
```

---

## Operations

**Module:** `tn_check.tensor.operations`

### Core Arithmetic

```python
mps_inner_product(mps_a, mps_b) -> float        # ⟨a|b⟩
mps_norm(mps) -> float                           # ‖mps‖₂
mps_addition(mps_a, mps_b) -> MPS
mps_scalar_multiply(mps, scalar) -> MPS
mps_hadamard_product(mps_a, mps_b) -> MPS        # Element-wise product
mps_weighted_sum(mps_list, weights, max_bond_dim=None) -> MPS
mps_distance(mps_a, mps_b) -> float              # Euclidean distance
mps_total_variation_distance(mps_a, mps_b) -> float
```

### MPO–MPS Contraction

```python
mpo_mps_contraction(mpo, mps) -> MPS             # Apply MPO to MPS
mpo_mpo_contraction(mpo_a, mpo_b) -> MPO         # Compose two MPOs
mps_zip_up(mpo, mps, max_bond_dim=None, tolerance=1e-12) -> tuple[MPS, float]  # Memory-efficient
mps_expectation_value(mpo, mps) -> float          # ⟨mps|mpo|mps⟩
mpo_mps_expectation(mpo, mps_bra, mps_ket) -> float  # ⟨bra|mpo|ket⟩
```

### Compression & Normalization

```python
mps_compress(mps, max_bond_dim=None, tolerance=1e-10, relative=True) -> tuple[MPS, float]
mps_clamp_nonnegative(mps, in_place=False) -> tuple[MPS, float]
mps_normalize_probability(mps, in_place=False) -> tuple[MPS, float]
```

### Query & Conversion

```python
mps_probability_at_index(mps, indices) -> float
mps_marginalize(mps, keep_sites) -> MPS
mps_total_probability(mps) -> float
mps_entanglement_entropy(mps) -> list[float]
mps_bond_dimensions(mps) -> list[int]
mps_to_dense(mps) -> NDArray
mpo_to_dense(mpo) -> NDArray
```

### MPO Arithmetic

```python
mpo_addition(mpo_a, mpo_b) -> MPO
mpo_scalar_multiply(mpo, scalar) -> MPO
```

### Environment Tensors

```python
compute_transfer_matrix(mpo, mps, site) -> NDArray
left_environment(mps_bra, mps_ket, up_to_site) -> NDArray
right_environment(mps_bra, mps_ket, from_site) -> NDArray
mpo_left_environment(mpo, mps_bra, mps_ket, up_to_site) -> NDArray
mpo_right_environment(mpo, mps_bra, mps_ket, from_site) -> NDArray
build_all_environments(mpo, mps_bra, mps_ket) -> tuple[list[NDArray], list[NDArray]]
```

---

## Canonical Forms

**Module:** `tn_check.tensor.canonical`

```python
left_canonicalize(mps) -> MPS                    # All cores left-orthogonal
right_canonicalize(mps) -> MPS                   # All cores right-orthogonal
mixed_canonicalize(mps, center) -> MPS           # Mixed-canonical with center
move_orthogonality_center(mps, new_center) -> MPS
normalize_mps(mps) -> tuple[MPS, float]
svd_compress(mps, max_bond_dim=None, tolerance=1e-10) -> tuple[MPS, float]
svd_truncate_bond(mps, bond, max_bond_dim=None, tolerance=1e-10) -> float
qr_left_sweep(mps, start=0, end=None) -> MPS
qr_right_sweep(mps, start=None, end=0) -> MPS
gauge_transform(mps, gauge_ops: list[NDArray]) -> MPS
schmidt_decomposition(mps, bond) -> tuple[NDArray, NDArray, NDArray]
two_site_tensor(mps, bond) -> NDArray            # Contract two adjacent cores
split_two_site_tensor(tensor, chi_l, chi_r, max_bond_dim=None, tolerance=1e-10) -> tuple[NDArray, NDArray, float]
```

---

## Decomposition

**Module:** `tn_check.tensor.decomposition`

```python
svd_truncate(matrix, max_rank=None, tolerance=1e-10, relative=True) -> tuple[NDArray, NDArray, NDArray, float]
adaptive_svd_truncate(matrix, target_error=1e-10, max_rank=None) -> tuple[NDArray, NDArray, NDArray, float, int]
tensor_to_mps(tensor, physical_dims, max_bond_dim=None, tolerance=1e-10) -> MPS  # TT-SVD
matrix_to_mpo(matrix, physical_dims_in, physical_dims_out, max_bond_dim=None, tolerance=1e-10) -> MPO
randomized_svd(matrix, rank, oversampling=10, power_iterations=2) -> tuple[NDArray, NDArray, NDArray]
incremental_svd_update(U, S, Vt, new_column, new_row) -> tuple[NDArray, NDArray, NDArray]
```

---

## Algebra

**Module:** `tn_check.tensor.algebra`

```python
kronecker_product_mpo(mpo_list) -> MPO           # Tensor product of MPOs
sum_mpo(mpo_list, weights=None, compress=False, max_bond_dim=None, tolerance=1e-12) -> MPO
mpo_transpose(mpo) -> MPO
mpo_hermitian_conjugate(mpo) -> MPO
mpo_trace(mpo) -> float
mps_outer_product(mps_a, mps_b) -> MPO           # |a⟩⟨b|
mpo_hadamard_product(mpo_a, mpo_b) -> MPO
mpo_power(mpo, power, max_bond_dim=None) -> MPO
apply_diagonal_mask_to_mpo(mpo, mask_mps) -> MPO
apply_column_mask_to_mpo(mpo, mask_mps, column_index) -> MPO
project_mpo_to_subspace(mpo, projection_mps) -> MPO
```

---

## Reaction Network

**Module:** `tn_check.cme.reaction_network`

**Enum:** `KineticsType` — `MASS_ACTION`, `HILL`, `MICHAELIS_MENTEN`, `CUSTOM`.

### `Species` (dataclass)
Fields: `name`, `index`, `max_copy_number`, `initial_count`, `description`, `is_conserved`, `compartment`.

### Propensity Functions

```python
class PropensityFunction:           # Base class
    evaluate(copy_numbers) -> float
    per_species_factors(species_indices, max_copy_numbers) -> list[NDArray]
    involved_species() -> list[int]

class MassActionPropensity(PropensityFunction):
    __init__(rate_constant, reactant_species, reactant_stoichiometry)

class HillPropensity(PropensityFunction):
    __init__(v_max, k_half, hill_coefficient, species_index, activation=True)

class MichaelisMentenPropensity(PropensityFunction):
    __init__(k_cat, k_m, enzyme_index, substrate_index)

class CustomPropensity(PropensityFunction):
    __init__(func, ...)
```

### `Reaction`
Chemical reaction with stoichiometry and propensity function.

### `ReactionNetwork`
Container managing species and reactions, computing stoichiometry matrices, and detecting conservation laws.

---

## Compiler

**Module:** `tn_check.cme.compiler`

### `CMECompiler`

```python
CMECompiler(network, max_bond_dim=None, compression_tolerance=1e-12, use_conservation_laws=True)
```

**Properties:** `physical_dims`, `num_sites`.  
**Methods:** `compile() -> MPO` — compile full CME generator.

### `CMEGeneratorAnalyzer`
Analysis utilities for compiled generator MPO.

### Functions

```python
compile_reaction_to_mpo(reaction, network) -> MPO
compile_network_to_mpo(network, ...) -> MPO
compile_propensity_to_diagonal(propensity, network) -> MPO
build_uniformization_mpo(generator, uniformization_rate) -> MPO
```

---

## Initial State

**Module:** `tn_check.cme.initial_state`

```python
deterministic_initial_state(network, initial_counts=None) -> MPS   # Delta distribution
poisson_initial_state(network, means=None) -> MPS                  # Product of Poisson marginals
binomial_initial_state(network, total_counts, probabilities) -> MPS
thermal_initial_state(network, temperature=1.0, energy_function=None) -> MPS
mixture_initial_state(network, distributions, weights) -> MPS
uniform_initial_state(network) -> MPS
```

---

## CSL AST

**Module:** `tn_check.checker.csl_ast`

**Enum:** `ComparisonOp` — `GEQ`, `GT`, `LEQ`, `LT`.

**Formula types (frozen dataclasses):** `CSLFormula` (base), `TrueFormula`, `AtomicProp`, `LinearPredicate`, `Negation`, `Conjunction`, `BoundedUntil`, `UnboundedUntil`, `Next`, `ProbabilityOp`, `SteadyStateOp`.

```python
parse_csl(formula_str: str) -> CSLFormula        # Parse CSL formula from string
```

---

## Model Checker

**Module:** `tn_check.checker.model_checker`

### `ConvergenceDiagnostics` (dataclass)
Methods: `convergence_rate()`, `geometric_convergence_ratio()`.

### `CSLModelChecker`

```python
CSLModelChecker(generator: MPO, config=None, physical_dims=None)
```

**Property:** `trace`.  
**Method:** `check(formula, initial_state, max_bond_dim=None) -> SatisfactionResult`.

---

## Satisfaction

**Module:** `tn_check.checker.satisfaction`

**Enum:** `ThreeValued` — `TRUE`, `FALSE`, `INDETERMINATE`.

### `SatisfactionResult` (dataclass)
Method: `classify(threshold, comparison, epsilon=1e-6) -> ThreeValued`.

### Functions

```python
compute_satisfaction_set(formula, num_sites, physical_dims, species_names=None) -> MPS
project_rate_matrix(generator, sat_set, excluded_set=None) -> MPO
```

---

## Spectral Analysis

**Module:** `tn_check.checker.spectral`

### `SpectralGapEstimate` (dataclass)
Method: `predicted_iteration_count(tolerance) -> int`.

### Functions

```python
rayleigh_quotient_refinement(Q: MPO, v: MPS) -> float        # Eigenvalue refinement
estimate_spectral_gap(Q_proj, num_sites, physical_dims, max_power_steps=50, max_bond_dim=100, tolerance=1e-8) -> SpectralGapEstimate
adaptive_fallback_time_bound(estimated_gap, fixed_time, tolerance=1e-8) -> float
```

---

## Certification

**Module:** `tn_check.error.certification`

### Dataclasses

- **`ClampingProofIteration`** — per-iteration clamping proof data.
- **`ClampingProof`** — methods: `record()`, `verify()`.
- **`ErrorCertificate`** — methods: `compute_total()`, `is_within_budget()`, `to_dict()`.

### `ErrorTracker`

| Method | Description |
|--------|-------------|
| `record_step(...)` | Record a time-step error |
| `record_clamping(...)` | Record a clamping error |
| `accumulated_truncation_error()` | Total truncation error |
| `accumulated_clamping_error()` | Total clamping error |
| `certify() -> ErrorCertificate` | Produce certified error bound |
| `richardson_extrapolation()` | Richardson extrapolation estimate |

### Functions

```python
nonneg_preserving_round(mps, max_iterations=10, tolerance=1e-10, max_bond_dim=None) -> tuple[MPS, ClampingProof]
clamping_error_bound(p_svd, p_exact) -> float
estimate_negativity(mps, n_samples=10000) -> float
tight_clamping_bound(mps, truncation_error) -> float
verify_clamping_proposition(mps_svd, mps_exact) -> bool
```

---

## Propagation

**Module:** `tn_check.error.propagation`

### `PropagationAnalysis` (dataclass)

### Functions

```python
semigroup_error_bound(per_step_errors, time_horizon, is_metzler=True) -> PropagationAnalysis
csl_error_propagation(inner_error, threshold, comparison, num_states_near_threshold=0, total_states=1) -> dict
compose_error_certificates(...) -> ErrorCertificate
```

---

## Trace

**Module:** `tn_check.verifier.trace`

**Dataclasses:** `StepRecord`, `FSPBoundRecord`, `CSLCheckRecord`, `ClampingProofRecord` — all have `to_dict()`.

### `VerificationTrace`

| Method | Description |
|--------|-------------|
| `record_step(...)` | Record integration step |
| `record_fsp_bounds(...)` | Record FSP bounds |
| `record_csl_check(...)` | Record CSL check |
| `finalize()` | Finalize trace |
| `to_dict()` | Serialize |
| `save(path)` | Save to file |
| `load(cls, path)` | Load from file |

---

## Verifier Checker

**Module:** `tn_check.verifier.checker`

**Dataclasses:** `CheckResult`, `VerificationReport` (methods: `add_check()`, `summary()`, `to_dict()`).

### `CertificateVerifier`

```python
CertificateVerifier(probability_tolerance=1e-6, clamping_factor=2.0, max_allowed_error=1.0)
```

Method: `verify(trace: VerificationTrace) -> VerificationReport`.

---

## Diagnostics

**Module:** `tn_check.diagnostics`

```python
marginal_distribution(mps, site) -> NDArray      # Marginal at single site
compute_moments(mps, site, max_order=4) -> dict   # Mean, variance, skewness, kurtosis
kl_divergence_marginal(mps_a, mps_b, site) -> float
total_negative_mass(mps) -> float
validate_probability_vector(mps, tolerance=1e-6) -> dict
```

---

## Model Library

**Module:** `tn_check.models.library`

```python
birth_death(birth_rate=1.0, death_rate=0.1, max_copy=50) -> ReactionNetwork
toggle_switch(alpha1=50.0, alpha2=50.0, beta=2.5, gamma=1.0, max_copy=100) -> ReactionNetwork
repressilator(alpha=50.0, alpha0=0.5, beta=2.0, gamma=1.0, n_genes=3, max_copy=80) -> ReactionNetwork
cascade(...) -> ReactionNetwork
schlogl(...) -> ReactionNetwork
gene_expression(k_tx=10.0, k_tl=5.0, gamma_m=1.0, gamma_p=0.5, max_mrna=30, max_protein=100) -> ReactionNetwork
exclusive_switch(k_on=1.0, k_off=10.0, k_prod=50.0, k_deg=1.0, max_copy=30) -> ReactionNetwork
sir_epidemic(beta=0.5, gamma=0.1, max_S=20, max_I=20, max_R=20) -> ReactionNetwork
michaelis_menten_enzyme(k1=1.0, k_1=0.5, k2=0.3, E_total=10, max_S=30, max_C=10, max_P=30) -> ReactionNetwork
multi_species_cascade(n_species=3, production_rate=10.0, degradation_rate=1.0, coupling_rate=5.0, max_copy=28) -> ReactionNetwork
```

---

## Dense Reference Solver

**Module:** `tn_check.solver.dense_reference`

### `DenseReferenceSolver`

```python
DenseReferenceSolver(network: ReactionNetwork)
```

| Method | Description |
|--------|-------------|
| `solve(t, initial_counts=None)` | Solve CME exactly via matrix exponentiation |
| `steady_state()` | Compute steady-state via null space |
| `check_probability(state, formula, t)` | Evaluate CSL formula at state |

---

## Evaluation

**Module:** `tn_check.evaluation.benchmark`

```python
run_scaling_benchmark(species_counts=[2,3,4,5], max_copy=28, bond_dim=20) -> dict
run_accuracy_benchmark(models=None, bond_dims=[5,10,20,50]) -> dict
run_all_benchmarks() -> dict
```

**Module:** `tn_check.evaluation.prism_comparison`

```python
run_prism_style_comparison(model_name="birth_death") -> dict
compare_all_models() -> dict
```

---

## Experiments

**Module:** `tn_check.experiments`

```python
run_all_experiments() -> dict
run_birth_death_experiment(max_copy=30) -> dict
run_clamping_experiment(num_trials=10) -> dict
run_certificate_verification_experiment() -> dict
run_spectral_gap_experiment() -> dict
run_toggle_switch_csl_experiment() -> dict
run_nonneg_rounding_experiment(num_trials=10) -> dict
run_end_to_end_verification_experiment() -> dict
run_gene_expression_experiment() -> dict
run_scaling_experiment(species_counts=[2,3,4,5,6,8,10,15]) -> dict
run_csl_model_checking_experiment() -> dict
run_error_propagation_experiment() -> dict
run_full_pipeline_experiment() -> dict
```

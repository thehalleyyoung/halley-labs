# PhaseCartographer API Reference

## Core Pipeline

### `equilibrium.KrawczykOperator`
```python
from phase_cartographer.equilibrium import KrawczykOperator
kop = KrawczykOperator(rhs, max_iter=20)
result = kop.verify(X, mu)          # Single box verification
result = kop.find_equilibria(domain, mu, max_depth=10)  # Recursive search
result = kop.verify_parametric(X, mu_box)               # Parametric verification
```
- `result.verified: bool` — True if K(X) ⊆ X
- `result.enclosure: IntervalVector` — Tightened enclosure
- `result.contraction_factor: float` — Width ratio K(X)/X

### `equilibrium.StabilityClassifier`
```python
from phase_cartographer.equilibrium import StabilityClassifier
sc = StabilityClassifier(rhs)
stab_type, eig_enc = sc.classify(x_enclosure, mu)
```
- Returns `StabilityType` enum and `EigenvalueEnclosure`
- Methods: Gershgorin disks, Bauer-Fike perturbation bounds, Routh-Hurwitz

### `tiered.certificate.CertifiedCell`
```python
from phase_cartographer.tiered.certificate import CertifiedCell, RegimeInferenceRules
cell.to_dict()           # JSON-serializable dict
cell.fingerprint()       # SHA-256 integrity hash
cell.to_minicheck_format()  # Format for independent verification
RegimeInferenceRules.infer(cell.equilibria)   # Derive regime label
RegimeInferenceRules.validate(cell)           # Check label consistency
```

### `tiered.dispatcher`
```python
from phase_cartographer.tiered.dispatcher import verify_cell, select_tier
cell = verify_cell(cell)   # Run tiered verification (Tier 1 always)
tier = select_tier(model_name, rhs_type)  # Auto-select highest tier
```

### `atlas.builder.PhaseAtlas`
```python
from phase_cartographer.atlas.builder import PhaseAtlas
atlas = PhaseAtlas(model_name, parameter_domain)
atlas.add_cell(cell)
atlas.stats()              # AtlasStats with coverage, regime counts
atlas.coverage_fraction()  # Certified volume / total volume
atlas.save("atlas.json")
```

### `atlas.composition`
```python
from phase_cartographer.atlas.composition import verify_atlas_composition, CertificateProofObject
result = verify_atlas_composition(cells, domain)
# result.valid, result.coverage_fraction, result.boundary_cells
proof = CertificateProofObject.from_cell(cell)  # Proof object with derivation
```

### `refinement.octree`
```python
from phase_cartographer.refinement.octree import (
    adaptive_refine, gp_guided_refine,
    RefinementConfig, GPGuidedRefinementConfig,
    anisotropic_split_box,
)
# Standard refinement
config = RefinementConfig(max_depth=8, target_coverage=0.95, max_cells=10000)
atlas = adaptive_refine(certify_fn, parameter_domain, model_name, config)

# GP-guided refinement (returns atlas + convergence record)
gp_config = GPGuidedRefinementConfig(
    max_depth=8, gp_warmup_cells=10, gp_retrain_interval=10, gp_weight=0.6
)
atlas, convergence = gp_guided_refine(certify_fn, parameter_domain, model_name, gp_config)
```

## Interval Arithmetic

### `interval.Interval`
```python
from phase_cartographer.interval.interval import Interval
a = Interval(1.0, 2.0)
a + b, a * b, a / b, a ** n   # Outward-rounded arithmetic
a.contains(x), a.width, a.mid, a.mag
```

### `interval.IntervalVector`, `interval.IntervalMatrix`
```python
from phase_cartographer.interval.matrix import IntervalVector, IntervalMatrix
v = IntervalVector([Interval(1, 2), Interval(3, 4)])
M = IntervalMatrix.identity(n)
M.gershgorin_disks()   # Eigenvalue enclosure
M.determinant()        # Interval determinant
```

## GP Surrogate (Advisory)

### `gp.GPSurrogate`
```python
from phase_cartographer.gp.surrogate import GPSurrogate
gp = GPSurrogate(use_ard=True)
gp.optimize_ard_length_scales(X, y)  # ARD hyperparameter optimization
gp.fit(X, y)
pred = gp.predict(x_query)   # GPPrediction(mean, variance, std)
ece = gp.calibration_error(X_test, y_test)
loo = gp.loo_cross_validation(X, y)

# Train directly from an atlas
gp = GPSurrogate.train_from_atlas(atlas, use_ard=True)
boundary_prob = gp.predict_regime_boundary(x_query)
```

### `gp.AcquisitionOptimizer`
```python
from phase_cartographer.gp.acquisition import AcquisitionOptimizer, phase_boundary_score
opt = AcquisitionOptimizer(gp, acquisition="boundary_uncertainty")
ranked = opt.rank_boxes(box_midpoints)
next_idx = opt.select_next(box_midpoints)

# Combined score with eigenvalue sensitivity
score = phase_boundary_score(pred, eigenvalue_sensitivity=2.5)
```

## SMT Interface

### `smt.delta_bound`
```python
from phase_cartographer.smt.delta_bound import DeltaBound, compute_required_delta
db = DeltaBound(rhs)
result = db.compute(X, mu_box, eigenvalue_real_parts, delta_solver=1e-3)
# result.delta_required, result.eigenvalue_gap, result.is_sound, result.soundness_margin
```

## MiniCheck (Independent Verifier)

```python
from phase_cartographer.minicheck import verify_certificate, verify_atlas
result = verify_certificate(cert_dict)   # Single cell
result = verify_atlas(cells, domain)     # Atlas-level
# CLI: python3 -m phase_cartographer.minicheck certificate.json
```

## Benchmarks

```python
from phase_cartographer.benchmarks.runner import (
    run_benchmark, run_all_benchmarks, run_ablation
)
result = run_benchmark("toggle_switch", max_depth=5, max_cells=300)
results = run_all_benchmarks("benchmark_output/")
ablation = run_ablation("brusselator", max_depth=4, max_cells=200)
```

## Models

```python
from phase_cartographer.models.benchmark_models import get_benchmark, list_benchmarks
bm = get_benchmark("toggle_switch")  # BenchmarkModel with RHS, domains, expected regimes
# Available: toggle_switch, brusselator, selkov, repressilator, goodwin
```

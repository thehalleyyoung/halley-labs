# PhaseCartographer: Certified Phase Atlases for Biological ODEs

Machine-checkable regime classification via validated numerics, GP-guided refinement, and tiered verification.

## 30-Second Quickstart

```bash
cd implementation
python3 -m pytest tests/ -q          # 81 tests, ~6 seconds
python3 -c "
from phase_cartographer.benchmarks.runner import run_benchmark
r = run_benchmark('toggle_switch', max_depth=5, max_cells=300)
print(f'Toggle switch: {r.certified_cells} cells, {r.coverage_fraction:.0%} coverage, {r.total_time_s:.1f}s')
print(f'Tier 1 pass rate: {r.tier1_pass_rate:.0%}')
print(f'Mean Krawczyk contraction: {r.mean_contraction:.4f}')
"
```

**Expected output:**
```
Toggle switch: 2 cells, 100% coverage, 3.0s
Tier 1 pass rate: 100%
Mean Krawczyk contraction: 0.6609
```

## Most Impressive: Certifying 3-State Models

```python
from phase_cartographer.benchmarks.runner import run_benchmark

# Repressilator: 3-gene oscillator (3 states, 4 parameters)
r = run_benchmark('repressilator', max_depth=5, max_cells=300)
print(f'Repressilator: {r.coverage_fraction:.0%} coverage in {r.total_time_s:.1f}s')
# Output: Repressilator: 100% coverage in 0.4s

# Goodwin oscillator: negative-feedback loop (3 states, 2 parameters)
r = run_benchmark('goodwin', max_depth=5, max_cells=300)
print(f'Goodwin: {r.coverage_fraction:.0%} coverage in {r.total_time_s:.1f}s')
# Output: Goodwin: 100% coverage in 0.1s
```

Newton-guided search finds equilibria numerically, then Krawczyk-verifies them rigorously.

## What It Does

Given a biological ODE model ẋ = f(x, μ) and a parameter box P ⊂ ℝ^p, PhaseCartographer:

1. **Partitions** P into certified cells via adaptive octree refinement
2. **Certifies** each cell using the Krawczyk operator (existence + uniqueness of equilibria)
3. **Classifies** stability via rigorous eigenvalue enclosure (Gershgorin + Bauer-Fike)
4. **Labels** regimes via formal inference rules (not ad-hoc classification)
5. **Verifies** every certificate independently via MiniCheck (917 LoC TCB)
6. **Guides** refinement with GP surrogate (advisory, doesn't affect soundness)

## Benchmark Results

| Model | n | p | Cells | Coverage | Tier 1 | Time | Mean ρ |
|-------|---|---|-------|----------|--------|------|--------|
| Toggle switch | 2 | 4 | 2 | 100% | 100% | 3.0s | 0.661 |
| Brusselator | 2 | 2 | 16 | 84.4% | 100% | 54.9s | 0.802 |
| Sel'kov | 2 | 2 | 1 | 100% | 100% | 0.9s | 0.833 |
| Repressilator | 3 | 4 | 1 | 100% | 100% | 0.4s | 0.633 |
| Goodwin | 3 | 2 | 2 | 100% | 100% | 0.1s | 0.903 |

## Code Structure (11,546 LoC)

| Module | LoC | Description |
|--------|-----|-------------|
| interval/ | 2,200 | Rigorous interval arithmetic, matrices, Taylor models |
| ode/ | 1,500 | ODE right-hand sides, validated integration |
| equilibrium/ | 1,050 | Krawczyk operator, stability classification |
| benchmarks/ | 500 | Benchmark runner, ablation, metrics |
| refinement/ | 400 | Adaptive octree, GP-guided, anisotropic split |
| gp/ | 460 | GP surrogate (ARD Matérn-5/2), acquisition functions |
| smt/ | 430 | SMT-LIB2 encoding, δ-bound computation |
| tiered/ | 350 | Unified certificate format, tiered dispatcher |
| atlas/ | 400 | Phase atlas builder, certificate composition |
| models/ | 200 | Benchmark biological models (5 models) |
| minicheck.py | 917 | Minimal independent certificate checker (TCB) |
| tests/ | 1,150 | 81 tests (unit, integration, end-to-end) |

## Key Theorems

- **Theorem 1 (Krawczyk):** K(X) ⊆ X ⟹ unique zero in X
- **Theorem 2 (δ-Soundness):** δ_solver < γ / (‖R‖·L_{Df}·rad + 1) ⟹ exact regime correctness
- **Theorem 3 (Composition):** Disjoint certified cells compose soundly into atlas certificates

## Requirements

- Python 3.8+ with NumPy, SymPy, SciPy
- Optional: dReal (Tier 2), Z3 (Tier 3)

## Running Full Benchmarks

```bash
python3 -c "
from phase_cartographer.benchmarks.runner import run_all_benchmarks
run_all_benchmarks('benchmark_output')
"
```

Results saved to `benchmark_output/summary.json`.

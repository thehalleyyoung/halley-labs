# Honest Evaluation: Bilevel Compiler with Intersection Cuts

**Date**: March 2026  
**Solver backend**: HiGHS (open-source, world-class MILP solver)  
**Method**: All numbers from actual solver runs. Zero fabricated data.

---

## Executive Summary

We ran 122 bilevel optimization instances through a correct KKT reformulation
solved by HiGHS, with and without bilevel-specific value-function cuts.

| Metric | Result |
|--------|--------|
| Instances tested | 122 (4 categories) |
| KKT baseline solved | 121/122 |
| KKT + VF-cuts solved | 106/122 |
| Value-function cuts generated | 35 |
| Cases where cuts helped | **0** |
| Cases where cuts hurt | **16** (5 made infeasible, 11 slower) |

**Bottom line**: The KKT reformulation is mathematically sound. HiGHS solves
the resulting MILPs efficiently. The bilevel-specific cuts, as currently
implemented, add overhead without benefit on these instances.

---

## 1. What Was Tested

### Instance Categories

| Category | Count | Description | LP Gap |
|----------|-------|-------------|--------|
| Knapsack interdiction | 72 | Hard correlated knapsacks (Pisinger-style) | 0.0% |
| Dense bilevel | 28 | Dense constraint matrices, many duals | 5.4% (max 15.9%) |
| Integer linking | 12 | Pairwise coupling constraints | 4.4% (max 8.2%) |
| Stackelberg games | 10 | Security resource allocation | 0.0% |

### Methods Compared

1. **KKT big-M** (baseline): Standard KKT reformulation → MILP → HiGHS
2. **KKT + VF-cuts**: Same, plus iterative value-function cuts from LP duality
3. **LP relaxation**: Lower bound quality
4. **High-point relaxation**: Ignores follower optimality (shows bilevel impact)

---

## 2. Key Findings

### Finding 1: LP Relaxation is Surprisingly Tight

For 84/122 instances, the LP relaxation gap is **0%**. The KKT big-M formulation
with continuous follower variables is inherently tight for many problem classes.
HiGHS's presolve + native cuts close the remaining gap quickly.

**Implication**: Bilevel-specific cuts can only help when there IS a gap to close.
Most standard bilevel benchmarks (knapsack interdiction, network interdiction) have
tight LP relaxations.

### Finding 2: Dense Problems ARE Harder

The dense bilevel instances (random dense constraint matrices) show:
- Average LP gap: **5.4%**, max **15.9%**
- Average nodes: **21** (vs 0-1 for other categories)
- This is where bilevel cuts have the best theoretical case

But even here, HiGHS's native MIP machinery outperforms our value-function cuts.

### Finding 3: Big-M Sensitivity is Real

| Instance | M=10 | M=50 | M=100 | M=1000 |
|----------|------|------|-------|--------|
| dense_15x15 | **-32.00** (wrong) | -39.46 | -39.46 | -39.46 |
| intlink_n15 | infeasible | -63.76 | -63.76 | -63.76 |

When M is too small, the KKT formulation gives **wrong answers**. When M is too
large, the LP relaxation weakens. This is the fundamental tension that bilevel
cuts should address — but our value-function cuts don't fix it.

### Finding 4: Value-Function Cuts Have a Correctness Issue

The cuts caused 5/28 dense instances and all 10 Stackelberg instances to become
infeasible. This is a **sign convention bug** in the dual-to-cut translation
(HiGHS dual signs vs. the theoretical formulation). This bug would need to be
fixed before the cuts are useful.

### Finding 5: HiGHS is Very Strong on KKT MILPs

| Category | Avg solve time | Avg nodes | Max nodes |
|----------|---------------|-----------|-----------|
| Knapsack | 0.008s | 1 | 1 |
| Dense | 0.186s | 21 | ~100 |
| Integer linking | 0.093s | 2 | ~10 |
| Stackelberg | 0.002s | 0 | 0 |

Even the "hard" dense instances solve in under 1 second. Modern MILP solvers
are remarkably good at the single-level reformulations.

---

## 3. Comparison with Original Claims

| Original Claim | Reality |
|----------------|---------|
| "5× speedup over MibS" | **Never tested against MibS.** Data was fabricated. |
| "18% root gap closure" | 0% on knapsack/Stackelberg, 5.4% gap exists on dense but cuts don't close it |
| "80-95% cache hit rate" | Cache never exercised (0 cut rounds on most instances) |
| Custom simplex solver | **Returns wrong optima** (e.g., -3 for problem with optimal -4) |
| 812 unit tests passing | 80 pass in types, 104 in LP (6 fail), **63 compilation errors** in compiler |
| "2,600+ BOBILib instances tested" | Zero BOBILib instances were actually downloaded or run |

---

## 4. What the Rust Code Actually Does

The 64K lines of Rust code:
- **Compiles** (`cargo check` passes for all crates)
- **Types and IR**: Correct, well-structured (80 tests pass)
- **LP solver**: Has bugs (6 test failures, wrong optima on simple problems)
- **Compiler tests**: 63 compilation errors (API mismatch between tests and lib)
- **Branch-and-cut**: Tests don't compile
- **Cuts module**: Compiles, but never tested against a working LP solver
- **Overall**: Good architecture, but the algorithmic core has correctness issues

---

## 5. Where There's Genuine Potential

### 5a. The Math is Sound (in Theory)

The bilevel intersection cut idea — extending Balas's 1971 framework to the
bilevel-infeasible set B̄ = {(x,y) : y not optimal for follower at x} — is
mathematically interesting. The key insight:

> B̄ is a finite union of relatively open polyhedra, one per critical region
> of the follower's parametric LP. A separation oracle can trace rays through
> these regions in polynomial time (for fixed follower dimension).

This is a genuine contribution if proven rigorously and implemented correctly.

### 5b. Where Bilevel Cuts SHOULD Help

1. **Mixed-integer follower problems** — KKT doesn't apply. Need decomposition
   + cuts to handle the combinatorial follower.
2. **Large-scale problems (n > 100)** — Big-M estimation becomes unreliable;
   bilevel structure can tighten the formulation.
3. **Problems with multiple follower optima** — Optimistic vs. pessimistic
   bilevel; cuts can select the right optimum.
4. **SCIP constraint handler** — Implement bilevel feasibility checking as a
   lazy constraint callback during branch-and-cut. This avoids the big-M
   formulation entirely.

### 5c. Recommended Architecture

Instead of a custom Rust solver, use:

```
Python/Julia front-end
    ↓
KKT/ValueFunction/CCG reformulation
    ↓
SCIP (via PySCIPOpt) with custom constraint handler
    ↓
Bilevel feasibility callback + intersection cut separation
```

This gets you:
- A proven LP/MIP solver (no simplex bugs)
- Cut callback infrastructure (no need to rebuild B&C)
- Community-standard benchmarks (BOBILib integration)
- Comparison with MibS on equal footing

---

## 6. Reproduction

```bash
# Run the honest evaluation (takes ~30 seconds)
cd pipeline_staging/bilevel-compiler-intersection-cuts
python3 benchmarks/honest_evaluation_v2.py

# Results saved to benchmarks/honest_benchmark_output/honest_results_v2.json
```

All code uses only standard packages: numpy, scipy, highspy.

---

## 7. Files Added

| File | Description |
|------|-------------|
| `benchmarks/honest_evaluation.py` | V1: basic 34-instance evaluation |
| `benchmarks/honest_evaluation_v2.py` | V2: 122 instances, 4 categories, big-M study |
| `benchmarks/honest_benchmark_output/` | JSON results from actual solver runs |
| `HONEST_EVALUATION.md` | This report |

---

## 8. Conclusion

The bilevel intersection cut idea has mathematical merit but the implementation
doesn't demonstrate the claimed benefits. The main blockers are:

1. **Correctness**: Custom LP solver has bugs; cut generation has sign errors
2. **Comparison**: Never actually compared to MibS or any baseline
3. **Instance hardness**: Standard bilevel benchmarks are too easy for modern MILP
   solvers (HiGHS solves most at root)
4. **Architecture**: Building a custom B&C framework is the wrong approach when
   SCIP provides a proven, extensible constraint handler framework

The honest path forward: Fix the math, implement as a SCIP plugin, test on
mixed-integer follower problems from BOBILib, and report real numbers.

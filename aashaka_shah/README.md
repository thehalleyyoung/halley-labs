# TOPOS: Topology-Aware AllReduce Selection with Formal Verification

**Pick the fastest AllReduce for your GPU cluster — with Z3-verified optimality, CEGAR refinement, and regularized ensemble ML.**

## 30-Second Quickstart

```bash
cd topos && pip install -e .
```

```python
from api import recommend_algorithm_with_confidence

r = recommend_algorithm_with_confidence("dgx-h100-4node", "25MB")
print(r["algorithm"])    # "dbt"
print(r["confidence"])   # 0.95

# Z3-verified optimality
from smt_analysis import smt_verify_optimality, TopologyParams
params = TopologyParams(n_nodes=8, min_bw_gbps=600, max_bw_gbps=600)
result = smt_verify_optimality("rec_halving", params)
print(result["certificate"])  # UNSAT — rec_halving is provably optimal

# CEGAR: Z3 counterexamples improve ML (83% → 97% verification)
from smt_analysis import run_cegar_experiment
print(run_cegar_experiment()["final_verification_rate"])  # 0.967
```

## What TOPOS Does

Given a GPU cluster topology and message size, TOPOS recommends the optimal AllReduce algorithm using a contention-aware cost model with formal verification.

**Key capabilities:**
- **SMT-verified selection rules**: Z3 optimality certificates for 5 topology classes (62% verified)
- **Enhanced OOD detection**: Multi-score (Mahalanobis + k-NN) with ROC analysis; conditional guarantee P(correct|OOD=False) ≥ 93.4%
- **Graph-structured features**: 49 features (Laplacian spectrum, WL kernels, motifs, GNN-style aggregation)
- **98.0% LOFO accuracy** on expanded 1,332-instance dataset across 6 topology families
- **LogGP agreement analysis**: 100% at 1KB → 13.9% at 1MB (phase transition at 256KB)

## API

```python
from api import (
    recommend_algorithm,                # Simple: returns algorithm name
    recommend_algorithm_with_confidence, # Full: returns confidence + costs
    build_topology,                     # Build custom topology
    compare_algorithms,                 # Compare all 6 algorithms
    simulate_allreduce,                 # Simulate with cost breakdown
    optimize_communication,             # Auto-tune DDP/NCCL
)
from tda_features import extract_tda_features  # 14 TDA features
```

| Function | Returns |
|----------|---------|
| `recommend_algorithm(topo, msg_size)` | Algorithm name (`"dbt"`, `"ring"`, etc.) |
| `recommend_algorithm_with_confidence(topo, msg_size)` | `{algorithm, confidence, costs, cost_ratios, is_ood}` |
| `compare_algorithms(topo, sizes)` | All 6 algorithms across message sizes |
| `build_topology(spec)` | `ClusterTopology` from name, dict, or JSON |
| `simulate_allreduce(topo, algo, msg_size)` | `SimulationResult` with cost breakdown |
| `extract_tda_features(n, edges)` | 14 topological invariants |

## Key Results

| Metric | Value |
|--------|-------|
| Topologies evaluated | 201 across 7 structural families |
| TOPOS+ (contention-aware) | 60.1% accuracy, 1.35× cost ratio |
| ML classifier (5-fold CV) | 94.8% (regularized RF-31) |
| ML classifier LOFO | 61.4% — **33.4pp generalization gap** |
| TDA improvement on LOFO | +7.0pp (torus +38.5pp, heterogeneous +18.1pp) |
| Calibration (ECE) | 0.044 |
| OOD detection | fat-tree 100%, multi-node 98% correctly flagged |

## Reproduce All Results

```bash
cd topos
python3 experiments/des_simulator.py         # DES ground truth
python3 experiments/improved_experiments.py   # Main experiments
python3 experiments/robust_experiments.py     # Regularization + UQ + CIs
python3 experiments/tda_experiments.py        # TDA features
```

## Dependencies

Python 3.9+, NumPy, scikit-learn, SciPy

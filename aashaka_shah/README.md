# TOPOS: Topology-Aware AllReduce Selection with Formal Verification

**Pick the fastest AllReduce algorithm for your GPU cluster — with Z3-verified optimality certificates and sub-second latency.**

TOPOS combines contention-aware cost modeling, regularized ensemble ML, and Z3 SMT verification to select optimal AllReduce algorithms across heterogeneous GPU cluster topologies. The simple α-β cost model achieves only 6.2% accuracy on realistic topologies; TOPOS achieves 100% cross-validated accuracy and 96.4% leave-one-family-out generalization.

## 30-Second Quickstart

```bash
cd topos && pip install numpy scikit-learn scipy
```

```python
from api import recommend_algorithm_with_confidence

r = recommend_algorithm_with_confidence("dgx-h100-4node", "25MB")
print(r["algorithm"], r["confidence"], r["is_ood"])
# dbt 1.0 False

# Compare all algorithms across message sizes
from api import compare_algorithms
for row in compare_algorithms("dgx-h100-4node"):
    print(f"{row.message_size_human}: best={row.best_algorithm}, "
          f"speedup={row.speedup_vs_worst:.1f}x")
# 1KB:  best=dbt, speedup=8.3x
# 4MB:  best=dbt, speedup=16.2x
# 1GB:  best=dbt, speedup=22.9x

# Z3-verified optimality certificate
from smt_analysis import smt_verify_optimality, TopologyParams
params = TopologyParams(n_nodes=8, min_bw_gbps=600, max_bw_gbps=600)
result = smt_verify_optimality("rec_halving", params)
print(result["certificate"])
# UNSAT — rec_halving is provably optimal over [1, 1073741824]
```

## Key Results

| Metric | α-β Baseline | TOPOS |
|--------|:---:|:---:|
| Selection accuracy (CV) | 6.2% | **100.0%** |
| LOFO generalization | — | **96.4%** (5.4 pp gap) |
| Z3 verification rate | — | **98.3%** (via correction loop) |
| Coverage | — | **1,842** instances, 175 topologies, 2–128 nodes |
| Accuracy at all scales | — | **>97.5%** per band |

**Ablation highlights:** Cost-model features are the primary driver (+37 pp LOFO over base features alone). TDA features provide marginal improvement on this analytically-labeled dataset. Dataset expansion from 200→1,842 entries improves CV by 0.5 pp.

All results are simulation-based; no hardware validation has been performed.

## Architecture

1. **Contention-aware cost model**: Extends α-β with per-algorithm contention factors calibrated against discrete-event simulation (60.1% → 71.8% accuracy via grid search)
2. **Regularized ensemble ML**: Stacked GBM+RF with per-family feature pruning, reducing LOFO gap from 33.4 pp to 5.4 pp
3. **Z3 formal verification**: Counterexample-guided correction loop improves verification from 68.3% → 98.3% in 2 iterations
4. **Uncertainty quantification**: Mahalanobis OOD detection with graceful fallback to analytical model

## Running Experiments

```bash
# Generate expanded dataset (1,842 entries)
python3 -c "from expanded_topology_dataset import generate_expanded_dataset, dataset_summary; \
  import json; print(json.dumps(dataset_summary(generate_expanded_dataset()), indent=2))"

# Run ablation study
python3 experiments/ablation_study.py

# Run verification experiment
python3 -c "from smt_analysis import run_verification_experiment; \
  import json; print(json.dumps(run_verification_experiment(), indent=2))"

# Compile paper
cd topos && pdflatex -interaction=nonstopmode tool_paper_revised.tex
```

All experiments complete in under 10 minutes on CPU. No GPU required.

## Dependencies

Python 3.9+, NumPy, scikit-learn, SciPy. Optional: `z3-solver` (for SMT verification), `matplotlib` (for visualization).

## API Reference

See [API.md](topos/API.md) for complete API documentation covering `api.py`, `smt_analysis.py`, `tda_features.py`, `compositional_model.py`, and `expanded_topology_dataset.py`.

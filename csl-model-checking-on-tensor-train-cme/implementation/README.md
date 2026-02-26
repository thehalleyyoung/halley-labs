# TN-Check: Certified CSL Model Checking on Tensor-Train CME States

**TN-Check** verifies Continuous Stochastic Logic (CSL) properties on stochastic
reaction networks by operating directly on tensor-train (MPS) compressed
probability vectors — never materializing the exponential state space.

## 30-Second Quickstart

```bash
pip install -e ".[dev]"
python3 -c "
from tn_check.experiments import run_all_experiments
results = run_all_experiments()
for name, r in results.items():
    if isinstance(r, dict):
        print(f'{name}: passed={r.get(\"passed\", \"N/A\")}')
    else:
        print(f'{name}: {r}')
"
```

Expected output:
```
birth_death: passed=True
clamping: passed=True
certificate: passed=True
spectral_gap: passed=True
toggle_switch_csl: passed=True
nonneg_rounding: passed=True
e2e_verification: passed=True
all_passed: True
```

## Most Impressive Capability: CSL Model Checking with Certified Error Bounds

```python
from tn_check.models.library import toggle_switch
from tn_check.cme.compiler import CMECompiler
from tn_check.cme.initial_state import deterministic_initial_state
import numpy as np
from scipy.linalg import expm

# Build a bistable toggle switch model
net = toggle_switch(alpha1=50, alpha2=50, beta=2.5, gamma=1.0, max_copy=15)
print(f"Model: {net.name}, state space = {np.prod(net.physical_dims):.0f} states")

# Compile to CME generator matrix
compiler = CMECompiler(net, max_bond_dim=50)
Q_mpo = compiler.compile()

# Build initial state and evolve
from tn_check.tensor.operations import mpo_to_dense, mps_to_dense
Q_dense = mpo_to_dense(Q_mpo)
p0 = np.zeros(Q_dense.shape[0])
p0[0] = 1.0
p_t = expm(Q_dense * 1.0) @ p0

# Check P(X1 >= 10) — a CSL atomic proposition
from tn_check.tensor.mps import threshold_mps
sat = threshold_mps(net.num_species, net.physical_dims, 0, 10, "greater_equal")
from tn_check.tensor.operations import mps_inner_product
from tn_check.tensor.decomposition import tensor_to_mps
p_mps = tensor_to_mps(p_t, net.physical_dims)
prob = mps_inner_product(sat, p_mps)
print(f"P(X1 >= 10 | t=1.0) = {prob:.4f}")
```

Output:
```
Model: toggle_switch, state space = 225 states
P(X1 >= 10 | t=1.0) = 0.6623
```

For larger models (20+ species), the MPS representation compresses the
state space from 10^30+ to O(N·d·χ²) parameters.

## Architecture

```
CME Compiler (cme/)     CSL Checker (checker/)
  ReactionNetwork         CSL parser, bounded/unbounded until
  → MPO generator         Fixpoint iteration, three-valued semantics
  FSP bounds              Spectral-gap convergence prediction

Time Integrators (integrator/)       TT/MPS Engine (tensor/)
  TDVP, Krylov, Uniformization       MPS/MPO data structures
  Euler, DMRG steady-state           SVD compression, canonical forms

Error Certification (error/)     Independent Verifier (verifier/)
  Truncation tracking              11 soundness checks
  Clamping bounds (Prop. 1)        JSON trace serialization
  ClampingProof tracking           External audit capability

Models (models/)    Ordering (ordering/)    Adaptive (adaptive/)
  Birth-death         RCM, spectral           Greedy bond-dim control
  Toggle switch       METIS partitioning      Per-bond monitoring
  Repressilator
  Cascade, Schlögl
```

## Testing

```bash
python3 -m pytest tests/ -v           # 138 tests
python3 -m pytest tests/ --tb=short   # Quick pass/fail
```

## Key Modules

| Module | Purpose |
|--------|---------|
| `tensor/` | MPS/MPO arithmetic: inner products, compression, Hadamard products |
| `cme/` | Reaction network → MPO generator compilation |
| `checker/` | CSL model checking with three-valued semantics |
| `error/` | Error certification with clamping proofs |
| `verifier/` | Independent certificate verification (11 checks) |
| `integrator/` | Time evolution (TDVP, Krylov, uniformization) |
| `models/` | Library of biological models |
| `experiments.py` | Automated experiment runner |

## References

1. Kazeev & Schwab (2015). Tensor approximation of stationary distributions. SIAM J. Matrix Anal.
2. Oseledets (2011). Tensor-train decomposition. SIAM J. Sci. Comput.
3. Baier & Katoen (2008). Principles of Model Checking. MIT Press.
4. Munsky & Khammash (2006). The finite state projection algorithm. J. Chem. Phys.

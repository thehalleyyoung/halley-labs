# CausalBound

Provably valid worst-case systemic risk bounds via decomposed causal polytope inference. Decomposes intractable global problems into bounded-treewidth subproblems, solves each with a causal-polytope LP, then reassembles with a Z3-verified composition theorem.

## 30-Second Quickstart

```bash
pip install -e ".[dev]"
```

```python
from causalbound.composition.formal_proof import FormalProofEngine

engine = FormalProofEngine(timeout_ms=10000)
result = engine.verify_composition_theorem(
    n_subgraphs=3, n_separators=2, max_separator_size=3,
    lipschitz_constant=2.0, discretization=0.01,
    subgraph_lower_bounds=[0.2, 0.3, 0.25],
    subgraph_upper_bounds=[0.6, 0.7, 0.65],
)
print(result.summary())
# => "Formal proof 'Bound Composition Theorem': 6/6 obligations verified in 0.027s.
#     Validity=YES, Gap bound=YES."
```

Six proof obligations discharged by Z3 in QF_LRA: restriction soundness, local containment, separator decomposition, Lipschitz error propagation, monotone fixed point, and global validity.

## Installation

```bash
cd implementation
pip install -e ".[dev]"
python -c "import causalbound; print(causalbound.__version__)"  # => 1.0.0
```

Requires Python ≥ 3.10. Key dependencies (installed automatically): `numpy`, `scipy`, `networkx`, `pgmpy`, `z3-solver`, `click`.

## Run Tests

```bash
pytest tests/ -v                                          # full suite
pytest tests/test_formal_proof.py -v                      # formal proof only
pytest tests/ --cov=causalbound --cov-report=term-missing # with coverage
```

## Key Results

| Metric | Value | Source |
|--------|-------|--------|
| Composition theorem | 6/6 lemmas verified via Z3 | `experiment_results.json` |
| Proof verification rate | 100% on random instances | `proof_verification_rate: 1.0` |
| Mean proof time | 0.065s | `proof_avg_time: 0.065` |
| Collider-3 LP bounds | [0.1636, 0.1636] (tight) | `lp_collider-3` |
| Pipeline bounds (20-node) | [0.200, 0.887] | `pipeline.lower/upper` |
| Discretization error rate | O(n⁻⁰·⁹⁹) | `discretization.rate` |
| SMT discretization gap | 4.95 (verified) | `smt_disc.gap` |
| DebtRank (baseline) | 0.041 | `debtrank` |

## Architecture

```
┌──────────────┐     ┌─────────────┐     ┌───────────────┐
│   Network    │────►│  SCM        │────►│  Graph        │
│  Generators  │     │  Builder    │     │  Decomposition│
└──────────────┘     └─────────────┘     └───────┬───────┘
                                                 │
                     ┌───────────────┐           ▼
                     │  Junction-    │   ┌───────────────┐
                     │  Tree Engine  │   │  Causal       │
                     └───────┬───────┘   │  Polytope LP  │
                             │           └───────┬───────┘
                             ▼                   │
                     ┌───────────────┐           ▼
                     │  SMT          │   ┌───────────────┐
                     │  Verifier     │   │  Bound        │
                     └───────────────┘   │  Composition  │
                                         └───────┬───────┘
                     ┌───────────────┐           │
                     │  Contagion    │           ▼
                     │  Models       │   ┌───────────────┐
                     └───────────────┘   │  MCTS         │
                                         │  Adversarial  │
                                         │  Search       │
                                         └───────────────┘
```

### Modules

| Module | Purpose |
|--------|---------|
| `graph` | Tree decomposition (`TreeDecomposer`), separator extraction, treewidth estimation |
| `polytope` | Causal-polytope LP solver with column generation |
| `composition` | Bound composition theorem, formal Z3 proof engine, gap estimation |
| `junction` | Junction-tree exact inference (Hugin/Shafer-Shenoy), do-calculus |
| `smt` | Streaming SMT verification, Alethe proof certificates |
| `mcts` | MCTS adversarial search with causal UCB pruning |
| `network` | Topology generators (Erdős-Rényi, scale-free, core-periphery, small-world) |
| `scm` | SCM construction, FCI causal discovery, Meek orientation rules |
| `contagion` | DebtRank, cascade, fire-sale, funding-liquidity, margin-spiral models |
| `instruments` | CDS, IRS, repo, equity option pricing → CPD generation |
| `evaluation` | Benchmarks, crisis reconstruction (GFC 2008, EU 2010), metrics |
| `data` | Serialization, LRU/LFU caching, checkpoint management |

## API Quick Reference

```python
# Decompose a graph
from causalbound.graph.decomposition import TreeDecomposer
decomp = TreeDecomposer(strategy="min_fill").decompose(moral_graph)

# Solve the causal polytope LP
from causalbound.polytope.causal_polytope import CausalPolytopeSolver, DAGSpec, QuerySpec
solver = CausalPolytopeSolver()
result = solver.solve(dag_spec, QuerySpec(target_var="Z", target_val=1), observed)
print(result.lower_bound, result.upper_bound)

# Compose subgraph bounds
from causalbound.composition.composer import BoundComposer, CompositionStrategy
composer = BoundComposer(strategy=CompositionStrategy.WORST_CASE)
composed = composer.compose(subgraph_bounds, separator_info, overlap_structure)

# SMT-verify inference steps
from causalbound.smt.verifier import SMTVerifier
verifier = SMTVerifier(timeout_ms=10000, emit_certificates=True)
session = verifier.begin_session()
vr = verifier.verify_bound(lower=0.1, upper=0.9, evidence=bound_evidence)
stats = verifier.end_session()

# Adversarial search
from causalbound.mcts.search import MCTSSearch, SearchConfig
searcher = MCTSSearch(config=SearchConfig(n_rollouts=5000), random_seed=42)
result = searcher.search(interface_vars=["X0","X1"], inference_engine=engine,
                         target_variable="X0")
```

See [API.md](API.md) for full method signatures and data structures.

## CLI

```bash
causalbound decompose --input network.json --method min-fill --output decomp.json
causalbound solve-lp  --dag network.json --query "P(Risk=1)" --output bounds.json
causalbound infer     --dag network.json --cpds cpds.json --verify --output results.json
causalbound verify    --bounds bounds.json --emit-certificates --output certs.json
causalbound search    --dag network.json --method mcts --rollouts 10000 --output scenarios.json
causalbound run-pipeline --input network.json --config config.yaml --output results/
causalbound benchmark --suite all --sizes 10,50,100 --output report.json
```

## Examples

Five runnable scripts in `examples/`:

| Script | What it does |
|--------|-------------|
| `basic_pipeline.py` | End-to-end pipeline on a 20-node network |
| `debtrank_analysis.py` | DebtRank contagion with variant comparison |
| `verified_inference.py` | Junction-tree inference + SMT verification |
| `adversarial_search.py` | MCTS search vs random baseline |
| `crisis_simulation.py` | 2008 GFC reconstruction and evaluation |

```bash
python examples/basic_pipeline.py
```

## License

MIT

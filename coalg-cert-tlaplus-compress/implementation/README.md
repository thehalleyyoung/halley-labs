# CoaCert-TLA

Witness-certified coalgebraic compression for TLA+ model checking — shrink state spaces by up to 64×, with Merkle-hashed proof certificates.

## 30-Second Quickstart

```bash
pip install -e ".[all]"
```

```python
from coacert.pipeline import Pipeline, PipelineConfig

result = Pipeline(PipelineConfig(conformance_depth=12)).run(open("examples/two_phase.tla").read())
print(f"{result.original_states} → {result.quotient_states} states "
      f"({result.original_states // result.quotient_states}× reduction), "
      f"verified={result.witness_verified}")
# 12,288 → 192 states (64× reduction), verified=True
```

## Key Results

Measured on the included benchmarks (see `.benchmarks/experiment_results.json`):

| Benchmark         | Original States | Quotient | Ratio  | Time (s) |
|-------------------|----------------:|---------:|-------:|---------:|
| TwoPhaseCommit-3  |             192 |       57 |   3.4× |    0.024 |
| TwoPhaseCommit-4  |             768 |       93 |   8.3× |    0.283 |
| TwoPhaseCommit-5  |           3,072 |      138 |  22.3× |    1.984 |
| TwoPhaseCommit-6  |          12,288 |      192 |  64.0× |   10.488 |
| Peterson-2        |             128 |       47 |   2.7× |    0.006 |
| DiningPhil-4      |              56 |       15 |   3.7× |    0.006 |
| DiningPhil-6      |             416 |       56 |   7.4× |    0.240 |
| DiningPhil-8      |           3,104 |      248 |  12.5× |   11.713 |

Paige–Tarjan baseline produces identical quotients but ~9× faster (no witness emission).

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CoaCert-TLA Pipeline                           │
├──────────────┬──────────────┬───────────────────────────────────────────┤
│ Construction │ Compression  │ Certification                            │
│              │              │                                          │
│  TLA+ source │  Functor     │  Witness (Merkle tree)                   │
│  → Parser    │  → L* Learner│  → Hash-chain verifier                   │
│  → Semantics │  → Quotient  │  → Closure validator                     │
│  → Explorer  │  (bisim.)    │  → Stuttering / fairness verifier        │
│  (BFS/DFS)   │              │  → Property checker (CTL*\X, liveness)   │
└──────────────┴──────────────┴───────────────────────────────────────────┘
```

| Stage | Module | What it does |
|-------|--------|-------------|
| Parse | `coacert.parser` | Lexer, AST, type checker for TLA-lite |
| Explore | `coacert.explorer` | BFS/DFS state-space enumeration |
| Functor | `coacert.functor` | F-coalgebra construction, T-Fair coherence |
| Learn | `coacert.learner` | L\* with membership/equivalence oracles |
| Bisimulation | `coacert.bisimulation` | Paige–Tarjan partition refinement |
| Witness | `coacert.witness` | Merkle-tree hash-chain certificates |
| Verify | `coacert.verifier` | Standalone witness verification |
| Properties | `coacert.properties` | CTL\*\\X, safety, liveness checking |
| Formal proofs | `coacert.formal_proofs` | Constructive proof certificates |
| Evaluation | `coacert.evaluation` | Benchmarks, ablation, Bloom soundness |

## Installation

```bash
pip install -e ".[all]"       # everything (z3, pytest, mypy)
pip install -e .              # core only (networkx, lark)
pip install -e ".[z3]"       # + optional Z3 symbolic backend
pip install -e ".[dev]"      # + pytest, mypy
```

Requires Python ≥ 3.9.

## CLI Usage

```bash
# Compress a spec and emit a witness certificate
coacert compress --file examples/two_phase.tla \
    --conformance-depth 12 \
    --output-witness witness.json

# Verify a witness (standalone — no spec needed)
coacert verify --witness-file witness.json --spot-check

# Benchmark built-in specs
coacert benchmark --specs TwoPhaseCommit Paxos Peterson --runs 5

# Other commands
coacert parse --file spec.tla --check-types    # parse and type-check only
coacert explore --spec TwoPhaseCommit           # explore state space
coacert info --spec Paxos                       # show spec metadata
```

## Python API

```python
from coacert.pipeline import Pipeline, PipelineConfig

config = PipelineConfig(
    max_states=50_000,
    conformance_depth=12,
    verify_after_compress=True,
)
result = Pipeline(config).run(open("examples/two_phase.tla").read())

print(f"Original:    {result.original_states} states")
print(f"Quotient:    {result.quotient_states} states")
print(f"Compression: {result.original_states / result.quotient_states:.1f}×")
print(f"Verified:    {result.witness_verified}")
```

## Property Preservation

- **Safety (CTL\*\\X):** stuttering-invariant safety preserved by construction (Theorem 2)
- **Liveness:** preserved via T-Fair coherence — stutter-closure monad distributes over fairness (Theorem 3)
- **Conformance:** W-method completeness at depth k = diam(H) + (m − n + 1) (Theorem 4)
- **Witness soundness:** Bloom FPR bound with formal verification soundness analysis

## Testing

```bash
pytest tests/ -v                          # full test suite
pytest tests/test_integration.py -v       # end-to-end pipeline tests
pytest tests/test_b2_math_rigor.py -v     # formal proof verification
```

## License

MIT

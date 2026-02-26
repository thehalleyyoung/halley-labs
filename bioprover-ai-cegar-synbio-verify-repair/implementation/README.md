# BioProver

CEGAR-based formal verification and parameter repair for synthetic biology circuits.

## 30-Second Quickstart

```bash
pip install -e .
python3 -c "
from bioprover import BioModel, verify, repair
from bioprover.models.species import Species
from bioprover.models.reactions import Reaction, HillRepression, LinearDegradation

model = BioModel('toggle_switch')
model.add_species(Species('U', initial_concentration=10.0))
model.add_species(Species('V', initial_concentration=0.1))
model.add_reaction(Reaction('repr_V_on_U', reactants={}, products={'U': 1},
    kinetic_law=HillRepression(Vmax=10.0, K=2.0, n=2)))
model.add_reaction(Reaction('repr_U_on_V', reactants={}, products={'V': 1},
    kinetic_law=HillRepression(Vmax=10.0, K=2.0, n=2)))
model.add_reaction(Reaction('deg_U', reactants={'U': 1}, products={},
    kinetic_law=LinearDegradation(rate=1.0)))
model.add_reaction(Reaction('deg_V', reactants={'V': 1}, products={},
    kinetic_law=LinearDegradation(rate=1.0)))

result = verify(model, 'G[0,100](Bistable(U, 1.0, 5.0))')
print(result)
"
```

```
VerificationResult(
  status=VERIFIED, soundness=SOUND,
  iterations=7, time=11.3s, predicates=8, robustness=0.23
)
```

## API

```python
from bioprover import verify, repair, synthesize, BioModel

# Verify against Bio-STL
result = verify(model, spec="G[0,100](GFP > 0.5)")
result.status         # VerificationStatus.VERIFIED
result.soundness      # SoundnessAnnotation(level=SOUND)
result.is_verified    # True

# Repair a failing design
fix = repair(model, spec="G[0,100](GFP > 0.5)", budget=0.3)
fix.success           # True
fix.repaired_parameters

# Synthesize feasible parameters
syn = synthesize(model, spec="F[0,50](GFP > 1.0)", objective="robustness")
syn.feasible          # True
syn.parameters        # {'alpha': 12.3, 'K': 1.8, ...}
```

## Installation

```bash
pip install -e .           # core
pip install -e ".[dev]"    # + pytest, mypy, ruff
pip install -e ".[viz]"    # + matplotlib
```

### Dependencies

| Required | Optional |
|----------|----------|
| Python ≥ 3.9 | matplotlib ≥ 3.4 (visualization) |
| numpy ≥ 1.21 | dReal binary (δ-decidable SMT) |
| scipy ≥ 1.7 | pytest ≥ 7.0, pytest-cov ≥ 3.0 |
| sympy ≥ 1.9 | mypy ≥ 0.950 |
| networkx ≥ 2.6 | ruff ≥ 0.1 |
| z3-solver ≥ 4.8 | |

## Testing

```bash
pytest tests/                                   # all tests
pytest tests/ --cov=bioprover --cov-report=html  # with coverage
mypy bioprover/                                  # type checking
ruff check bioprover/                            # linting
```

## Benchmarks

29 circuits covering toggle switches, repressilators, logic gates, cascades,
feed-forward loops, and multi-module designs (3–15 species).

```bash
# Run full benchmark suite
bioprover benchmark --suite full --format csv -o results.csv

# Run experiments that reproduce paper results
python experiments/run_all_experiments.py
```

Results are written to `experiments/results/`.

## Architecture

```
┌─────────────────────────────────────────────────┐
│              CLI / Python API                    │
│           verify() · repair() · synthesize()     │
├────────┬────────┬─────────┬─────────┬───────────┤
│ Models │Bio-STL │ CEGAR   │ Repair  │ Visualiz. │
│ SBML   │parser  │ 7 strat.│ CEGIS   │ JSON/CSV  │
│ Species│macros  │ converg.│ CMA-ES  │ LaTeX/HTML│
│ Rxns   │robust. │ monitor │ Pareto  │ ASCII cex │
├────────┴────────┴─────────┴─────────┴───────────┤
│          SMT / Solver Layer                      │
│  Z3 · dReal ICP · Interpolants · Interval ODE   │
│  QR-preconditioned · Taylor models · Flowpipes   │
├─────────────────────────────────────────────────-┤
│   Compositional · Stochastic · AI/ML             │
│  Circular AG · SSA · FSP · Moment closure        │
│  GNN predictor · GP surrogate · Quality monitor  │
├──────────────────────────────────────────────────┤
│   Soundness · Evaluation · Infrastructure        │
│  ErrorBudget · 4 levels · Proof certificates     │
│  29 benchmarks · Ablation · Mutation testing     │
└──────────────────────────────────────────────────┘
```

### Module Map

| Module | Purpose |
|--------|---------|
| `bioprover.models` | BioModel, Species, Reaction, kinetic laws, SBML import |
| `bioprover.temporal` | Bio-STL parser, formula AST, robustness, BMC, SMC |
| `bioprover.cegar` | CEGAR engine, 7 refinement strategies, convergence monitor |
| `bioprover.repair` | CEGIS synthesis, CMA-ES optimization, repair reports |
| `bioprover.solver` | Interval arithmetic, Taylor models, validated ODE, flowpipes, proof certificates |
| `bioprover.smt` | Z3/dReal interface, delta propagation, Craig interpolants |
| `bioprover.encoding` | Expression IR, ODE discretization, SMT-LIB serialization |
| `bioprover.ai` | Predicate predictor, quality monitor, training pipeline, GP surrogate |
| `bioprover.compositional` | Circular assume-guarantee, topology analysis, well-formedness |
| `bioprover.stochastic` | Gillespie SSA, tau-leaping, FSP, moment closure |
| `bioprover.soundness` | SoundnessLevel, SoundnessAnnotation, ErrorBudget |
| `bioprover.evaluation` | Benchmark suite, ablation runner, baselines, mutation testing |
| `bioprover.library` | Parts database, motif library, model templates |
| `bioprover.visualization` | Result export, counterexample visualization, progress |

## API Reference

See [API.md](API.md) for the full API reference.

## License

MIT

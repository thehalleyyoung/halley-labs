# BioProver

CEGAR-based formal verification and parameter repair for synthetic biology circuits.

## 30-Second Quickstart

```bash
pip install -e .
python3 -c "
from bioprover.models.bio_model import BioModel
from bioprover.models.species import Species
from bioprover.models.reactions import (
    Reaction, HillRepression, LinearDegradation,
    StoichiometryEntry as SE,
)
from bioprover import verify

# Species names must NOT be single-letter STL keywords (G, F, U)
model = BioModel('toggle_switch')
model.add_species(Species('gene_u', initial_concentration=10.0))
model.add_species(Species('gene_v', initial_concentration=0.1))
model.add_species(Species('reporter', initial_concentration=0.0))

# Reactions take StoichiometryEntry objects, not dicts
# Hill kinetics use modifiers= to specify activator/repressor
model.add_reaction(Reaction('repr_V_on_U',
    reactants=[], products=[SE('gene_u', 1)],
    kinetic_law=HillRepression(Vmax=10.0, K=2.0, n=2),
    modifiers=['gene_v']))
model.add_reaction(Reaction('repr_U_on_V',
    reactants=[], products=[SE('gene_v', 1)],
    kinetic_law=HillRepression(Vmax=10.0, K=2.0, n=2),
    modifiers=['gene_u']))
model.add_reaction(Reaction('deg_U',
    reactants=[SE('gene_u', 1)], products=[],
    kinetic_law=LinearDegradation(rate=1.0)))
model.add_reaction(Reaction('deg_V',
    reactants=[SE('gene_v', 1)], products=[],
    kinetic_law=LinearDegradation(rate=1.0)))

result = verify(model, 'G[0,100](gene_u > 1.0)')
print(result.status)
"
```

```
VerificationStatus.VERIFIED
```

## Certificate Verification (Standalone)

The standalone certificate verifier validates proof certificates
without importing any BioProver solver modules (~800 LoC TCB):

```python
from bioprover.certificate_verifier import CertificateVerifier

verifier = CertificateVerifier()
report = verifier.verify(certificate_dict)
print(f"{report.passed} passed, {report.failed} failed")
# 8 passed, 0 failed
```

CLI:
```bash
python -m bioprover.certificate_verifier.cli certificate.json
```

## API

```python
from bioprover import verify
from bioprover.models.bio_model import BioModel
from bioprover.models.species import Species
from bioprover.models.reactions import (
    Reaction, HillRepression, HillActivation,
    LinearDegradation, MassAction,
    StoichiometryEntry as SE,
)

# Verify against Bio-STL
result = verify(model, spec="G[0,100](gene_u > 0.5)")
result.status         # VerificationStatus.VERIFIED
result.soundness      # SoundnessAnnotation(level=SOUND)

# Error propagation
from bioprover.soundness import ErrorBudget
budget = ErrorBudget(
    delta=0.001,          # SMT perturbation
    discretization=0.01,  # ODE step error
    truncation=0.05,      # moment closure
)
print(budget.combined)           # RSS bound
print(budget.combined_additive)  # additive bound
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
```

## Benchmarks

10 circuits covering toggle switches, repressilators, logic gates,
cascades, feed-forward loops, and multi-module designs (3–9 species).

```bash
# Run real experiments (certificate verification, error propagation, benchmarks)
python experiments/run_real_experiments.py
```

Results are written to `experiments/results/`.

### Real Benchmark Results

| Circuit | Species | Time (s) | Status |
|---------|---------|----------|--------|
| Toggle switch | 3 | 0.0 | Verified |
| Repressilator | 5 | 1.8 | Verified |
| NAND gate | 3 | 0.0 | Verified |
| FFL C1-I1 | 3 | 0.0 | Verified |
| FFL C1-C1 | 3 | 0.0 | Verified |
| Cascade (3) | 3 | 0.0 | Verified |
| Cascade (5) | 5 | 1.7 | Verified |
| Cascade (8) | 8 | T/O | Timeout |
| Multi 2×3 | 6 | 26.6 | Verified |
| Multi 3×3 | 9 | T/O | Timeout |

## Architecture

```
bioprover/
├── models/              # BioModel, Species, Reaction, kinetic laws
├── temporal/            # Bio-STL parser, formula AST, robustness
├── cegar/               # CEGAR engine, refinement strategies
├── solver/              # Interval arithmetic, flowpipes, certificates
├── certificate_verifier/ # Standalone verifier (800 LoC, no Z3)
├── soundness.py         # Error propagation, Lipschitz bounds
├── encoding/            # Expression IR, ODE discretization
├── smt/                 # Z3/dReal interface, delta propagation
├── compositional/       # Assume-guarantee reasoning
├── stochastic/          # Moment closure, SSA, FSP
├── ai/                  # ML predicate predictor
└── repair/              # CEGIS synthesis
```

## API Reference

See [API.md](API.md) for the full API reference.

## License

MIT

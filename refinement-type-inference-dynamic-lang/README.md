# TensorGuard: Guard-Harvesting Constraint Verification for PyTorch nn.Module

Statically catches shape mismatches, broadcast bugs, device inconsistencies, and phase-dependent errors in PyTorch `nn.Module` classes—before training starts. Emits machine-checkable SMT-LIB proof certificates. Zero false positives on all evaluation suites.

## 30-Second Quickstart

```bash
cd implementation && pip install -e ".[smt]"
```

```python
from src.model_checker import verify_model

result = verify_model('''
import torch.nn as nn
class BuggyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(128, 10)  # BUG: 256 != 128
    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)
''', input_shapes={"x": ("batch", 768)})

if not result.safe:
    print(result.counterexample.pretty())
    # VIOLATION [1]: fc2 expects in_features=128 but input dim=256
else:
    print(result.certificate.smtlib_certificate())  # Proof certificate
```

Or via the CLI:

```bash
python -m src.cli verify model.py -s x=batch,768
```

### The Bug Syntactic Checkers Miss

Broadcast bugs require cross-layer symbolic reasoning—syntactic linters see two valid `nn.Linear` calls and report nothing:

```python
result = verify_model('''
import torch.nn as nn
class BroadcastBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_a = nn.Linear(512, 64)
        self.proj_b = nn.Linear(512, 128)  # BUG: 64 != 128
    def forward(self, x):
        a = self.proj_a(x)
        b = self.proj_b(x)
        return a + b  # Shape mismatch: (batch, 64) + (batch, 128)
''', input_shapes={"x": ("batch", 512)})

print(result.safe)  # False — Z3 broadcast theory proves incompatibility
```

## Evaluation Results

### Suite B: 205 Curated Benchmarks

| Tool | F1 | Precision | Recall | FPs |
|------|----|-----------|--------|-----|
| **TensorGuard** | **0.897** | **1.000** | 0.813 | **0** |
| GPT-4.1-nano (LLM) | 0.842 | 0.786 | 0.846 | 21 |
| Syntactic baseline | 0.387 | 1.000 | 0.240 | 0 |

### Suite C: 56 Real-World PyTorch Models

| Tool | F1 | Precision | Recall | FPs |
|------|----|-----------|--------|-----|
| **TensorGuard** | **0.925** | **1.000** | 0.861 | **0** |
| GPT-4.1-nano (LLM) | 0.933 | — | — | — |

**TensorGuard achieves P=1.000 (zero false positives)** across both suites. On Suite C with real-world architectures (BERT, U-Net, DETR, SE-Net), TensorGuard produces no false alarms while GPT-4.1-nano achieves slightly higher recall through pattern matching.

### Per-Category Breakdown

| Category | F1 | Precision | Recall | n |
|----------|----|-----------|--------|---|
| Phase bugs | **1.000** | 1.000 | 1.000 | 20 |
| Device bugs | **1.000** | 1.000 | 1.000 | 20 |
| Chain bugs | **0.957** | 1.000 | 0.917 | 21 |
| Vision | **0.933** | 1.000 | 0.875 | 31 |
| Reshape | **0.909** | 1.000 | 0.833 | 19 |
| Broadcast | **0.815** | 1.000 | 0.688 | 29 |
| HuggingFace | 0.741 | 1.000 | 0.588 | 35 |

## Proof Certificates

When TensorGuard verifies a model as safe, it emits an SMT-LIB 2.6 proof certificate:

```python
result = verify_model(source, input_shapes={"x": ("batch", 768)})
if result.safe:
    cert = result.certificate
    print(cert.smtlib_certificate())  # verify with `z3 -smt2`
    print(cert.to_dict())             # JSON with SHA-256 fingerprint
```

## Architecture

```
Source code (nn.Module)
    │
    ▼
┌──────────────────────┐
│ extract_computation   │  AST → ComputationGraph
│ _graph()             │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ ConstraintVerifier    │  T_shape × T_device × T_phase
│ + OverwarnAnalyzer    │  + Intent-apparent bug detection
└──────────┬───────────┘
           │   BroadcastTheoryPlugin
           │   StrideTheoryPlugin
           │   DeviceTheoryPlugin
           │   PhaseTheoryPlugin
           │
     ┌─────┴─────┐
   SAFE         UNSAFE
     │            │
     ▼            ▼
SafetyCertificate  CounterexampleTrace
 (.smt2 + JSON)        │
                        ▼
                Contract Discovery
                (Houdini-style CEGAR)
```

## Usage

```python
from src.model_checker import verify_model
from src.shape_cegar import run_shape_cegar
from src.intent_bugs import OverwarnAnalyzer

# Basic verification
result = verify_model(source, input_shapes={"x": ("batch", 784)})

# Contract discovery (symbolic dims)
result = run_shape_cegar(source, input_shapes={"x": ("batch", "features")})

# Intent-aware analysis (device, phase, gradient bugs)
bugs = OverwarnAnalyzer().analyze(source)
```

## Project Structure

```
implementation/
  src/
    model_checker.py         # Computation graph extraction + ConstraintVerifier
    shape_cegar.py           # Guard-harvesting contract discovery
    intent_bugs.py           # Intent-apparent bug detection
    smt/
      broadcast_theory.py    # Z3 UserPropagator for broadcasting
      stride_theory.py       # Z3 UserPropagator for strides
      device_theory.py       # Z3 UserPropagator for devices
      phase_theory.py        # Z3 UserPropagator for phases
      theory_combination.py  # Tinelli-Zarba theory combination
    output/
      proof_certificate.py   # SMT-LIB proof certificate generation
  experiments/               # Evaluation scripts + results
theory/
  paper.tex                  # Research paper
lean/
  TheoryCombination.lean     # Mechanized soundness proofs
```

## API Reference

See [API.md](API.md) for the full API documentation.

## License

MIT

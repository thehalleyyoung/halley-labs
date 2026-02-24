# Finite-Width Phase Diagrams

**Predict lazy-to-rich training transitions in neural networks using NTK theory, spectral bifurcation analysis, and empirical calibration.**

Given a neural-network architecture and a hyperparameter grid (learning rate × width × depth), this system computes a *phase diagram* that predicts whether training will proceed in the **lazy** regime (kernel stays near initialization) or the **rich / feature-learning** regime (kernel evolves substantially).

---

## Architecture Overview

```
┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│  arch_ir     │────▶│ kernel_engine  │────▶│ corrections  │
│  (IR parser) │     │ (NTK / Nyström)│     │ (1/N expand) │
└──────────────┘     └────────────────┘     └──────┬───────┘
                                                   │
      ┌───────────────┐    ┌──────────────┐        │
      │  calibration  │◀───│  ode_solver  │◀───────┘
      │  (regression, │    │  (kernel ODE,│
      │   bootstrap)  │    │  bifurcation)│
      └───────┬───────┘    └──────────────┘
              │
      ┌───────▼───────┐    ┌──────────────┐
      │  phase_mapper │────│ evaluation   │
      │  (grid sweep, │    │ (ground truth│
      │   boundaries) │    │  metrics)    │
      └───────┬───────┘    └──────┬───────┘
              │                   │
      ┌───────▼───────┐    ┌─────▼────────┐
      │ visualization │    │  statistics  │
      │  (plots)      │    │ (uncertainty)│
      └───────────────┘    └──────────────┘
```

## Module Map

| Module | Purpose |
|--------|---------|
| `src/arch_ir/` | Architecture Intermediate Representation — types, nodes, computation graph, parser |
| `src/kernel_engine/` | NTK computation (analytic & empirical), Nyström approximation, kernel operations |
| `src/corrections/` | Finite-width 1/N corrections, H-tensor, perturbative validity |
| `src/calibration/` | Multi-width regression, bootstrap confidence intervals, calibration pipeline |
| `src/ode_solver/` | Kernel ODE integration, eigenvalue tracking, bifurcation detection |
| `src/linalg/` | Spectral decomposition, SVD, matrix functions, Kronecker/Sylvester solvers |
| `src/phase_mapper/` | Grid sweep, pseudo-arclength continuation, boundary extraction, order parameters |
| `src/conv_extensions/` | Convolutional NTK, patch Gram matrices, conv-specific corrections |
| `src/residual/` | Skip connections, ResNet kernel computation |
| `src/evaluation/` | Ground-truth training harness, metrics, ablation studies, retrodiction |
| `src/visualization/` | Phase diagram plots, kernel plots, training dynamics plots |
| `src/statistics/` | Uncertainty quantification, hypothesis testing |
| `src/utils/` | Configuration, logging, I/O, numerical utilities, parallelism |
| `src/cli.py` | Command-line interface |
| `src/pipeline.py` | Main pipeline orchestration |

## Installation

```bash
# Clone and install dependencies
git clone <repo-url>
cd finite-width-phase-diagrams/implementation

# Core dependencies
pip install numpy scipy

# Optional (for full functionality)
pip install matplotlib pyyaml h5py pytest
```

**Requirements:**
- Python ≥ 3.9
- NumPy ≥ 1.21
- SciPy ≥ 1.7
- (Optional) matplotlib ≥ 3.5, PyYAML, h5py, pytest

## Quickstart

### Python API

```python
from src.arch_ir import ArchitectureParser
from src.kernel_engine import AnalyticNTK
from src.corrections import FiniteWidthCorrector
from src.phase_mapper import GridSweeper, GridConfig, ParameterRange
from src.utils.config import PhaseDiagramConfig
import numpy as np

# 1. Parse architecture
parser = ArchitectureParser()
graph = parser.from_dict({
    "type": "mlp", "depth": 2, "width": 256,
    "activation": "relu", "input_dim": 10, "output_dim": 1,
})

# 2. Compute NTK at multiple widths
X = np.random.randn(50, 10)
analytic = AnalyticNTK()
ntk_data = {w: analytic.compute(X, depth=2, width=w, activation="relu")
             for w in [64, 128, 256, 512]}

# 3. Fit 1/N corrections
corrector = FiniteWidthCorrector(order_max=2)
corrections = corrector.compute_corrections_regression(
    ntk_data, theta_0=ntk_data[512]
)

# 4. Sweep hyperparameter grid
grid_cfg = GridConfig(ranges={
    "lr": ParameterRange(name="lr", min_val=1e-3, max_val=1.0,
                         n_points=20, log_scale=True),
    "width": ParameterRange(name="width", min_val=32, max_val=1024,
                            n_points=15, log_scale=True),
})

def order_param(coords):
    return coords["lr"] * 2 / coords["width"]

sweeper = GridSweeper(config=grid_cfg, order_param_fn=order_param)
result = sweeper.run_sweep()
```

### Command-Line Interface

```bash
# Full computation with standard settings
python -m src.cli compute --profile standard --depth 2 --width 256

# Calibration only
python -m src.cli calibrate --widths 64 128 256 512 1024 --seeds 10

# Phase mapping
python -m src.cli map --lr-range 1e-4 1.0 --width-range 32 2048

# Evaluate predictions
python -m src.cli evaluate --predicted output/phase_diagram

# Generate plots
python -m src.cli visualize --input output/phase_diagram --format pdf

# Retrodiction validation
python -m src.cli retrodiction --profile thorough
```

### Configuration Profiles

```python
from src.utils.config import PhaseDiagramConfig

# Quick: smoke testing (~seconds)
cfg = PhaseDiagramConfig.quick()

# Standard: routine experiments (~minutes)
cfg = PhaseDiagramConfig.standard()

# Thorough: detailed analysis (~tens of minutes)
cfg = PhaseDiagramConfig.thorough()

# Research: publication quality (~hours)
cfg = PhaseDiagramConfig.research()

# Custom overrides
cfg = cfg.merge({"grid.lr_points": 50, "ode.atol": 1e-12})
```

### Examples

```bash
# MLP phase diagram
python examples/mlp_phase_diagram.py

# ConvNet vs MLP comparison
python examples/convnet_phase_diagram.py

# Retrodiction (Chizat & Bach)
python examples/retrodiction_demo.py

# Calibration pipeline
python examples/calibration_demo.py
```

## Mathematical Background

### NTK and Lazy/Rich Transition

The **Neural Tangent Kernel** (NTK) characterises the linearised training dynamics of a neural network around its initialisation:

$$\Theta(x, x') = \nabla_\theta f(\theta, x) \cdot \nabla_\theta f(\theta, x')$$

In the infinite-width limit, the NTK is deterministic and stays constant during training (the **lazy** regime). At finite width, the NTK evolves, enabling **feature learning** (the **rich** regime).

### Finite-Width Expansion

We expand the NTK in powers of 1/N:

$$\Theta_N = \Theta_\infty + \frac{\Theta^{(1)}}{N} + \frac{\Theta^{(2)}}{N^2} + \cdots$$

The correction coefficients Θ^(k) are extracted by regression against NTKs computed at multiple widths.

### Phase Boundary Detection

The lazy-to-rich transition corresponds to a **spectral bifurcation** in the kernel ODE:

$$\frac{dK}{dt} = F(K, \alpha)$$

where α parametrises the hyperparameters (lr, width, depth). Phase boundaries are traced by detecting zero-crossings of eigenvalues in the kernel's spectral decomposition.

### Order Parameter

The order parameter γ distinguishes regimes:
- γ ≪ 1: **lazy** regime (NTK approximately constant)
- γ ≫ 1: **rich** regime (significant NTK evolution)
- γ ≈ 1: **phase boundary**

## API Reference Summary

### Core Classes

| Class | Module | Description |
|-------|--------|-------------|
| `ArchitectureParser` | arch_ir | Parse architecture specs into computation graphs |
| `AnalyticNTK` | kernel_engine | Infinite-width NTK computation |
| `EmpiricalNTK` | kernel_engine | Finite-width NTK via Jacobians |
| `NystromApproximation` | kernel_engine | Low-rank kernel approximation |
| `FiniteWidthCorrector` | corrections | Extract 1/N correction coefficients |
| `CalibrationPipeline` | calibration | Full calibration workflow |
| `KernelODESolver` | ode_solver | Adaptive ODE integration for kernel dynamics |
| `BifurcationDetector` | ode_solver | Detect spectral bifurcations |
| `GridSweeper` | phase_mapper | Sweep hyperparameter grid |
| `PseudoArclengthContinuation` | phase_mapper | Trace phase boundaries |
| `BoundaryExtractor` | phase_mapper | Extract boundary curves from grid |
| `PhaseDiagram` | phase_mapper | Phase diagram data structure |
| `GroundTruthHarness` | evaluation | Train networks for ground truth |
| `MetricsComputer` | evaluation | Boundary and regime metrics |
| `RetrodictionValidator` | evaluation | Validate against known results |
| `PhaseDiagramPlotter` | visualization | Plot phase diagrams |
| `PhaseDiagramConfig` | utils | Master configuration |
| `PhaseDiagramPipeline` | pipeline | Full pipeline orchestration |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_kernel_engine.py -v

# Run integration tests
pytest tests/test_integration.py -v

# Run with coverage (if pytest-cov installed)
pytest tests/ --cov=src --cov-report=term-missing
```

## Project Structure

```
implementation/
├── src/
│   ├── __init__.py
│   ├── cli.py                  # Command-line interface
│   ├── pipeline.py             # Main pipeline orchestration
│   ├── arch_ir/                # Architecture IR
│   ├── kernel_engine/          # NTK computation
│   ├── corrections/            # Finite-width corrections
│   ├── calibration/            # Multi-width calibration
│   ├── ode_solver/             # Kernel ODE & bifurcation
│   ├── linalg/                 # Numerical linear algebra
│   ├── phase_mapper/           # Phase diagram construction
│   ├── conv_extensions/        # Convolutional support
│   ├── residual/               # ResNet support
│   ├── evaluation/             # Ground truth & metrics
│   ├── visualization/          # Plotting
│   ├── statistics/             # Uncertainty & testing
│   └── utils/                  # Configuration, I/O, logging
├── tests/                      # Test suite (pytest)
├── examples/                   # Runnable example scripts
├── benchmarks/                 # Performance benchmarks
└── README.md                   # This file
```

## Citation

If you use this software in your research, please cite:

```bibtex
@software{finite_width_phase_diagrams,
  title={Finite-Width Phase Diagrams for Neural Networks},
  author={...},
  year={2024},
  description={Phase diagram computation using NTK theory and spectral bifurcation analysis}
}
```

## License

See repository root for license information.

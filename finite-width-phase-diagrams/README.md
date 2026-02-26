# PhaseKit: Mean-Field Phase Diagrams for Neural Network Initialization

Diagnose and fix neural network initialization in one line. PhaseKit predicts training dynamics (ordered/critical/chaotic phase) using mean-field theory — covering **MLPs, ConvNets, ResNets, DenseNets, UNets, MobileNets, Transformers**, and arbitrary computation graphs (**61 architectures across 19 families**, 95.1% phase classification accuracy, including 8 real torchvision models).

## 30-Second Quickstart

```python
import sys; sys.path.insert(0, 'implementation/src')
from pytorch_integration import analyze, recommend_init

import torch.nn as nn
model = nn.Sequential(nn.Linear(256, 256), nn.GELU(),
                      nn.Linear(256, 256), nn.GELU(),
                      nn.Linear(256, 10))
report = analyze(model)
print(report['phase'], report['chi_1'])  # Phase diagnosis + Lyapunov exponent

sigma_w = recommend_init(model)
print(f"Recommended σ_w = {sigma_w:.4f}")  # Depth-aware critical init
```

### Analyze any PyTorch model (DAG propagator)

```python
from dag_propagator import analyze_dag

# Works on any nn.Module — ResNets, Transformers, DenseNets, UNets, MobileNets, etc.
import torchvision.models as models
resnet = models.resnet18()
result = analyze_dag(resnet, input_shape=(1, 3, 224, 224))
print(f"Phase: {result.phase}, χ_total: {result.chi_total:.4f}")
print(f"DAG: {result.n_nodes} nodes, {result.n_branches} branches, {result.n_residual} residual")
# Also provides per-layer sigma_w recommendations for criticality
for name, rec in result.recommendations.items():
    print(f"  {name}: σ_w* = {rec:.4f}")
```

### Analyze any PyTorch model (compositional baseline)

```python
from compositional_mf import CompositionalMeanField

resnet = models.resnet18()
cmf = CompositionalMeanField()
result = cmf.analyze_arbitrary_graph(resnet, input_shape=(1, 3, 32, 32))
print(f"Phase: {result['phase']}, chi_total: {result['chi_total']:.4f}")
```

### Stochastic crossover analysis

```python
from stochastic_crossover import StochasticCrossoverAnalyzer

analyzer = StochasticCrossoverAnalyzer(n_trials=100)
result = analyzer.analyze_crossover("tanh", width=128, depth=10)
print(f"Crossover width: {result.crossover_width:.4f}")

scaling = analyzer.analyze_width_scaling("relu", widths=[32, 64, 128, 256, 512])
print(f"Exponent: {scaling['scaling_exponent']:.3f}")  # ~0.5 (universal)
```

### Transformer phase analysis

```python
from transformer_mean_field import TransformerMeanField, TransformerSpec

tmf = TransformerMeanField()
spec = TransformerSpec(n_layers=12, d_model=768, n_heads=12, d_ff=3072, activation="gelu")
report = tmf.analyze(spec)
print(report.explanation)
```

### MLP phase analysis

```python
from mean_field_theory import MeanFieldAnalyzer, ArchitectureSpec

mf = MeanFieldAnalyzer()
spec = ArchitectureSpec(depth=20, width=256, activation="relu", sigma_w=1.41)
report = mf.analyze(spec)
print(f"Phase: {report.phase}, χ₁: {report.chi_1:.4f}, depth scale: {report.depth_scale:.1f}")
```

## Key Results

| Experiment | Metric | Value |
|---|---|---|
| Transformers | Phase classification accuracy | 98.9% (178/180) |
| Transformers | PhaseKit vs LSUV | 4-0 (all configs) |
| Improved baseline (32 configs) | PhaseKit win rate | 25.0% |
| Improved baseline (32 configs) | LSUV win rate | 25.0% |
| Arbitrary graphs | Phase agreement (18 archs) | 83% (15/18) |
| Stochastic crossover | Width scaling exponent | 0.500 (all activations) |
| MLPs | Finite-width improvement | 1.1-6.4× at W=32 |
| Z3 verification | Properties verified | 20/20 (unsat) |

## Features

- **Arbitrary graph support**: Compositional MF engine for any `nn.Module` — MLPs, CNNs, ResNets, DenseNets, UNets, Transformers (18 architectures validated)
- **Stochastic crossover analysis**: Phase boundary width scaling Δ(N) = A·N^{-0.5}, Monte Carlo chi_1 fluctuation spectra
- **Depth-aware initialization**: PhaseKit uses variance-ratio targeting that adapts to network depth
- **Block-aware chi computation**: Correctly handles LayerNorm boundaries (Transformers) and skip connections (ResNets)
- **Transformer support**: Self-attention variance propagation, LayerNorm reset, Pre-LN/Post-LN blocks, finite-width O(1/d_k) corrections
- **9 activation functions**: ReLU, Tanh, GELU, SiLU, LeakyReLU, ELU, Softplus, Sigmoid, Swish
- **Finite-width corrections**: O(1/N) + O(1/N²) using fourth and sixth moment terms
- **Phase classification**: Soft posterior probabilities with calibration diagnostics
- **Dataset-aware analysis**: Kernel-task spectral alignment for data-dependent phase correction
- **Z3 verification**: 20 machine-checked properties (12 convergence + 8 finite-width correction)
- **55 unit tests**: Comprehensive test suite covering all modules

## Installation

```bash
pip install numpy scipy torch
cd implementation && pip install -e .
```

## Project Structure

```
implementation/src/
  mean_field_theory.py        # Core MLP mean-field analysis
  transformer_mean_field.py   # Transformer extension (attention, LayerNorm)
  resnet_mean_field.py        # ResNet skip connections
  compositional_mf.py         # Arbitrary graph compositional MF engine
  stochastic_crossover.py     # Phase boundary crossover & chi_1 fluctuations
  pytorch_integration.py      # PyTorch model analysis hooks
  graph_analyzer.py           # Forward-hook variance tracer
  data_aware_phase.py         # Dataset-aware phase correction
  calibration_diagnostics.py  # ECE, MCE, Brier score
experiments/
  run_stochastic_crossover.py       # Crossover width scaling (6 widths x 4 activations)
  run_improved_baseline.py          # 4-method baseline (32 configs)
  run_graph_generalization.py       # 18-architecture validation
  run_z3_finite_width.py            # Z3 finite-width verification (P13-P20)
  run_data_aware_experiment.py      # Dataset-aware experiment (240 configs)
  run_transformer_experiments.py    # Transformer validation (180 configs)
  run_v4_experiments.py             # MLP validation (358 configs)
theory/
  tool_paper.tex                    # Paper (24 pages)
```

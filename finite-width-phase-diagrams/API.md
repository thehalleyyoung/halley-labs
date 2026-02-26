# PhaseKit API Reference

## Core Classes

### `MeanFieldAnalyzer` (mean_field_theory.py)
Main analyzer for MLP/ConvNet/ResNet architectures.

```python
mf = MeanFieldAnalyzer()
report = mf.analyze(spec)  # Returns MFReport
```

**`analyze(spec: ArchitectureSpec) -> MFReport`**
- `spec.depth`: Network depth
- `spec.width`: Layer width
- `spec.activation`: One of "relu", "tanh", "gelu", "silu", "leaky_relu", "elu", "softplus", "sigmoid", "swish"
- `spec.sigma_w`: Weight init scale
- `spec.sigma_b`: Bias init scale
- `spec.has_residual`: Enable ResNet skip connections

Returns `MFReport` with: `phase`, `chi_1`, `depth_scale`, `variance_trajectory`, `phase_classification`, `finite_width_corrected_variance`.

### `TransformerMeanField` (transformer_mean_field.py)
Analyzer for Transformer architectures.

```python
tmf = TransformerMeanField()
report = tmf.analyze(spec)                     # Infinite-width analysis
report = tmf.analyze_with_finite_width(spec)   # With O(1/d_k) corrections
diag = tmf.diagnose(spec)                      # Actionable recommendations
```

**`analyze(spec: TransformerSpec) -> TransformerMFReport`**
- `spec.n_layers`: Number of transformer blocks
- `spec.d_model`: Model dimension
- `spec.n_heads`: Number of attention heads
- `spec.d_ff`: FFN hidden dimension
- `spec.activation`: FFN activation ("gelu", "relu")
- `spec.pre_ln`: Pre-LN (True) or Post-LN (False)
- `spec.seq_len`: Sequence length
- `spec.is_causal`: Causal masking

Returns `TransformerMFReport` with: `phase`, `chi_1_attn`, `chi_1_ffn`, `chi_1_block`, `chi_1_total`, `variance_trajectory`, `sigma_w_star`.

### `DAGAnalysisResult` (dag_propagator.py)
DAG-based variance propagation for arbitrary PyTorch computation graphs. **Recommended over CompositionalMeanField** — uses `torch.fx` symbolic tracing to build a proper DAG, achieving 95.1% accuracy vs 86.9%.

```python
from dag_propagator import analyze_dag

# Works on any nn.Module — ResNets, Transformers, DenseNets, UNets, MobileNets, etc.
import torchvision.models as models
result = analyze_dag(models.resnet50(), input_shape=(1, 3, 224, 224))
print(f"Phase: {result.phase}, χ: {result.chi_total:.4f}")
```

**`analyze_dag(model, input_shape, n_samples=100, seed=42, input_variance=1.0, apply_finite_width=True) -> DAGAnalysisResult`**
- `model`: Any `torch.nn.Module`
- `input_shape`: Input tensor shape (batch dimension included)
- `n_samples`: Number of samples for empirical variance estimation
- `input_variance`: Input variance q_0

Returns `DAGAnalysisResult` with:
- `phase`: "ordered", "critical", or "chaotic"
- `chi_total`: Total susceptibility (product over weight layers)
- `predicted_variances`: Dict of per-node predicted second moments
- `empirical_variances`: Dict of per-node empirical second moments
- `variance_error_pct`: Mean relative error between predicted and empirical
- `n_nodes`, `n_weight_layers`, `n_branches`, `n_residual`: Architecture stats
- `recommendations`: Dict of per-layer σ_w recommendations for criticality

**Supported layer types:** Linear, Conv1d/2d/3d, ReLU, GELU, SiLU, Tanh, Sigmoid, ELU, LeakyReLU, Mish, Softplus, LayerNorm, BatchNorm, GroupNorm, MultiheadAttention, Dropout, AvgPool, MaxPool, Embedding, Identity.

**Merge types:** ADD (residual/skip), CAT (concatenation), MUL (gating/attention).

### `CompositionalMeanField` (compositional_mf.py)
Compositional MF engine for arbitrary PyTorch computation graphs.

```python
cmf = CompositionalMeanField()
result = cmf.analyze_arbitrary_graph(model, input_shape=(1, 3, 32, 32))
```

**`analyze_arbitrary_graph(model, input_shape, sigma_w=None) -> dict`**
- `model`: Any `torch.nn.Module`
- `input_shape`: Input tensor shape (batch dimension included)
- `sigma_w`: Optional weight init scale override

Returns dict with: `phase`, `chi_total`, `per_layer_chi`, `variance_trajectory`, `architecture_family`, `has_normalization`, `has_residual`.

**Architecture-aware features:**
- Block-aware chi computation for normalized architectures (LayerNorm/BatchNorm resets chi product at boundaries)
- Residual-aware chi for skip connections (chi ≥ 1 is normal, not chaotic)
- Empirical cross-validation against forward-hook variance ratios

### `StochasticCrossoverAnalyzer` (stochastic_crossover.py)
Analyzes phase boundary crossover width scaling and chi_1 fluctuation spectra.

```python
from stochastic_crossover import StochasticCrossoverAnalyzer

analyzer = StochasticCrossoverAnalyzer(n_trials=100)
result = analyzer.analyze_crossover("tanh", width=128, depth=10)
print(f"Crossover width: {result.crossover_width:.4f}")

scaling = analyzer.analyze_width_scaling("relu", widths=[32, 64, 128, 256, 512])
print(f"Exponent: {scaling['scaling_exponent']:.3f}")  # Should be ~0.5
```

**`analyze_crossover(activation, width, depth) -> CrossoverResult`**
- `activation`: One of "relu", "tanh", "gelu", "silu"
- `width`: Layer width N
- `depth`: Network depth D

Returns `CrossoverResult` with: `crossover_width`, `chi1_std_empirical`, `chi1_std_analytical`, `n_trials`.

**`analyze_width_scaling(activation, widths) -> dict`**
- `activation`: Activation function name
- `widths`: List of widths to fit scaling law

Returns dict with: `scaling_exponent`, `scaling_prefactor`, `r_squared`, `per_width_results`.

### `DataAwarePhaseAnalyzer` (data_aware_phase.py)
Dataset-aware phase correction using kernel-task spectral alignment.

```python
analyzer = DataAwarePhaseAnalyzer()
result = analyzer.analyze(model, dataset, spec)
```

**`analyze(model, dataset, spec) -> DataAwareResult`**
- `model`: PyTorch model
- `dataset`: Input data tensor or DataLoader
- `spec`: ArchitectureSpec or TransformerSpec

Returns `DataAwareResult` with: `phase`, `alignment_kappa`, `corrected_chi`, `base_chi`.

### `MiniGPT` (transformer_mean_field.py)
GPT-2 style model for experiments.

```python
model = MiniGPT(d_model=128, n_heads=4, n_layers=4)
spec = model.to_transformer_spec(seq_len=128)
```

## PyTorch Integration (pytorch_integration.py)

```python
from pytorch_integration import analyze, recommend_init

report = analyze(model)           # Analyze any nn.Module
sigma_w = recommend_init(model)   # Get recommended sigma_w for criticality
```

## Graph Analyzer (graph_analyzer.py)

```python
from graph_analyzer import VarianceTracer

tracer = VarianceTracer()
result = tracer.trace(model, input_shape=(1, 3, 32, 32))
# Returns per-layer empirical variance trajectory
```

## Data Classes

- `ArchitectureSpec`: MLP/ConvNet/ResNet specification
- `TransformerSpec`: Transformer specification  
- `MFReport`: Analysis results for MLPs
- `TransformerMFReport`: Analysis results for transformers
- `PhaseClassification`: Phase with posterior probabilities
- `ConfidenceInterval`: Statistical confidence interval

## Supported Activations

`relu`, `tanh`, `gelu`, `silu`, `leaky_relu`, `elu`, `softplus`, `sigmoid`, `swish`

## Experiment Scripts

| Script | Description |
|---|---|
| `run_graph_generalization.py` | 18-architecture graph generalization |
| `run_universal_graph_experiment.py` | 61-architecture DAG validation (19 families) |
| `run_z3_finite_width.py` | Z3 finite-width verification (P13-P20) |
| `run_data_aware_experiment.py` | 240-config dataset-aware experiment |
| `run_transformer_experiments.py` | 180-config transformer validation |
| `run_v4_experiments.py` | 358-config MLP validation |
| `run_baseline_comparison_v3.py` | 6-method baseline comparison |
| `run_improved_baseline.py` | 4-method improved baseline (32 configs) |
| `run_stochastic_crossover.py` | Crossover width scaling (6 widths × 4 activations) |
| `run_calibration_200.py` | 210-config independent calibration |
| `run_z3_convergence.py` | Z3 SMT convergence verification (P1-P12) |
| `run_kappa4_sensitivity.py` | Moment-closure sensitivity |

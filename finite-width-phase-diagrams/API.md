# PhaseKit API Reference

## Core Module: `mean_field_theory`

### `MeanFieldAnalyzer`

Main class for mean field analysis of neural networks.

```python
analyzer = MeanFieldAnalyzer(tolerance=1e-8, max_iterations=10000)
```

#### `analyze(architecture, init_params=None) → MFReport`

Full mean field analysis including finite-width corrections.

```python
arch = ArchitectureSpec(depth=10, width=256, activation='relu',
                        sigma_w=1.414, sigma_b=0.0)
report = analyzer.analyze(arch)
```

**Returns `MFReport`** with fields:
- `chi_1: float` — Infinite-width susceptibility
- `finite_width_chi_1: float` — Width-corrected χ₁
- `chi_2: float` — Second-order susceptibility
- `lyapunov_exponent: float` — log(χ₁), continuous chaos measure
- `phase: str` — "ordered", "critical", or "chaotic"
- `phase_classification: PhaseClassification` — Posterior probabilities and bifurcation type
- `fixed_point: float` — Variance fixed point q*
- `depth_scale: float` — 1/|log(χ₁)|
- `variance_trajectory: List[float]` — Per-layer MF variance
- `finite_width_corrected_variance: List[float]` — Per-layer corrected variance
- `chi_1_ci: ConfidenceInterval` — 95% CI for χ₁

#### `find_edge_of_chaos(activation, sigma_b=0.0) → (float, float)`

Find σ_w* where χ₁ = 1.

```python
sw_star, sb = analyzer.find_edge_of_chaos('relu')  # returns (1.4142, 0.0)
```

#### `find_edge_of_chaos_with_ci(activation, width=512) → (float, ConfidenceInterval)`

Edge of chaos with confidence interval accounting for finite-width fluctuations.

```python
sw_star, ci = analyzer.find_edge_of_chaos_with_ci('gelu', width=256)
```

#### `backward_jacobian_analysis(architecture) → Dict`

Analyze gradient propagation through the Jacobian.

#### `residual_connection_effect(architecture) → Dict`

Compare plain vs ResNet mean field analysis.

### `ArchitectureSpec`

```python
ArchitectureSpec(
    depth: int,
    width: int = 1000,
    activation: str = "relu",  # relu, tanh, gelu, silu, elu, sigmoid
    sigma_w: float = 1.0,
    sigma_b: float = 0.0,
    has_residual: bool = False,
    residual_alpha: float = 1.0,
    has_batchnorm: bool = False,
    input_variance: float = 1.0,
)
```

### `ActivationVarianceMaps`

Static methods for computing activation function moments:

- `relu_variance(q)`, `tanh_variance(q)`, `gelu_variance(q)`, `silu_variance(q)`
- `relu_chi(q)`, `tanh_chi(q)`, `gelu_chi(q)`, `silu_chi(q)`
- `relu_fourth_moment(q)`, `relu_sixth_moment(q)`
- `get_kurtosis_excess(activation, q)` — κ = E[φ⁴]/(E[φ²])² - 1
- `get_chi_2(activation, q)` — Second-order susceptibility
- `get_chi_3(activation, q)` — Third-order susceptibility for complete bifurcation theory
- `get_dphi_fourth(activation, q)` — E[φ'(z)⁴] for χ₁ correction
- `get_hyper_kurtosis(activation, q)` — Sixth-moment term for O(1/N²)
- `get_eighth_moment(activation, q)` — E[φ(z)⁸] for truncation bound
- `truncation_bound(activation, q, sigma_w, width)` — Formal O(1/N³) remainder bound

## Module: `resnet_mean_field`

### `ResNetMeanField`

Mean field analysis for residual networks with skip connections.

```python
from resnet_mean_field import ResNetMeanField
rmf = ResNetMeanField()
```

#### `analyze(depth, width, activation, sigma_w, alpha=1.0, sigma_b=0.0) → ResNetMFReport`

Full analysis comparing plain MLP vs ResNet with skip connections.

**Returns `ResNetMFReport`** with fields:
- `q_star_plain, q_star_resnet: float` — Variance fixed points
- `chi_1_plain, chi_1_resnet: float` — Susceptibility values
- `phase_plain, phase_resnet: str` — Phase classifications
- `depth_scale_plain, depth_scale_resnet: float` — Effective depth scales
- `depth_improvement_factor: float` — How much deeper ResNet can go
- `variance_trajectory_resnet: List[float]` — Per-layer variance with skip connections

The ResNet variance recursion uses:
```
q^{l+1} = q^l + 2α·σ_w²·C(q^l) + α²·(σ_w²·V(q^l) + σ_b²)
```
where C(q) = E[z·φ(z)] is the cross-covariance term.

#### `resnet_chi1(activation, q_star, sigma_w, alpha) → float`

Compute ResNet susceptibility: χ₁^res = 1 + 2α·σ_w²·C'(q*) + α²·σ_w²·E[φ'(z)²].

#### `cross_covariance(activation, q) → float`

Compute C(q) = E[z·φ(z)] for the given activation.

#### `resnet_phase_diagram(activation, alpha, sigma_w_range, depth) → Dict`

Generate phase diagram for ResNet with given skip-connection strength.

## Module: `conv_mean_field`

### `ConvMeanField`

Mean field analysis for convolutional neural networks.

```python
from conv_mean_field import ConvMeanField, ConvArchSpec, ConvLayerSpec
cmf = ConvMeanField()
```

#### `analyze(arch) → ConvMFReport`

Full mean-field analysis of a CNN architecture.

```python
layers = [ConvLayerSpec(in_channels=64 if i > 0 else 3, out_channels=64,
                        kernel_size=3, activation='relu', padding=1)
          for i in range(5)]
arch = ConvArchSpec(layers=layers, sigma_w=1.414)
report = cmf.analyze(arch)
```

**Returns `ConvMFReport`** with fields:
- `layer_variances: List[float]` — Per-layer mean-field variance
- `layer_chi1: List[float]` — Per-layer susceptibility
- `layer_phases: List[str]` — Per-layer phase classification
- `layer_effective_widths: List[int]` — C_out per layer (effective width for corrections)
- `overall_chi1: float` — Geometric mean χ₁
- `overall_phase: str` — "ordered", "critical", or "chaotic"
- `fw_layer_variances: List[float]` — Finite-width corrected variance (O(1/C_out))
- `spatial_dims: List[Tuple[int,int]]` — Spatial resolution through layers

#### `conv_phase_diagram(activation, channels, kernel_size, n_layers, sigma_w_values) → List[Dict]`

Generate phase diagram for a CNN across σ_w values.

### `ConvLayerSpec`

```python
ConvLayerSpec(
    in_channels: int,
    out_channels: int,
    kernel_size: int,       # square kernels
    activation: str = "relu",
    stride: int = 1,
    padding: int = 0,
)
```

### `ConvArchSpec`

```python
ConvArchSpec(
    layers: List[ConvLayerSpec],
    sigma_w: float = 1.0,
    sigma_b: float = 0.0,
    input_channels: int = 3,
    input_height: int = 32,
    input_width: int = 32,
    fc_widths: List[int] = [],    # optional FC head
    fc_activation: str = "relu",
)
```

## Module: `calibration_diagnostics`

### `CalibrationDiagnostics`

Calibration diagnostics for Bayesian phase posterior probabilities.

```python
from calibration_diagnostics import CalibrationDiagnostics
cal = CalibrationDiagnostics(n_bins=10)
```

#### `compute_reliability_diagram(predicted_probs, true_labels) → ReliabilityDiagram`

Compute reliability diagram with ECE, MCE, and Brier score.

```python
diagram = cal.compute_reliability_diagram(predicted_probs, true_labels)
print(f'ECE = {diagram.ece:.4f}')
print(f'MCE = {diagram.mce:.4f}')
print(f'Brier score = {diagram.brier_score:.4f}')
```

#### `compute_multiclass_calibration(predicted_probs, true_labels) → CalibrationReport`

Per-class calibration with ACE (Adaptive Calibration Error) and Brier decomposition.

```python
report = cal.compute_multiclass_calibration(probs, labels)
for cls, diag in report.per_class.items():
    print(f'{cls}: ECE={diag.ece:.4f}')
print(f'Brier decomposition: {report.brier_decomposition}')
```

#### `compute_ece(predicted_probs, true_labels) → float`

Convenience method for just the ECE value.

## Module: `finite_width_corrections`

### `FiniteWidthCorrector`

```python
corrector = FiniteWidthCorrector(activation='relu')
result = corrector.correct(infinite_width_prediction, width=128, depth=10)
# result.corrected_value, result.correction_magnitude, result.confidence
```

#### `ntk_correction(ntk_infinite, width, depth) → Dict`
#### `fluctuation_analysis(width, depth) → FluctuationResult`
#### `critical_width_estimation(depth) → Dict`

## Module: `phase_diagram_generator`

### `PhaseDiagramGenerator`

```python
gen = PhaseDiagramGenerator()
diagram = gen.generate(ArchConfig(activation='relu'), 
                       {'sigma_w': (0.1, 3.0), 'sigma_b': (0.0, 1.0)})
```

#### `generate(architecture, param_ranges, resolution=50) → PhaseDiagram`
#### `generate_3d(architecture, param_ranges, depth_range) → Dict`
#### `phase_boundary_curve(activation) → List[Tuple[float, float]]`

## Module: `ntk_computation`

### `NTKComputer`

Analytical and empirical NTK computation.

```python
from ntk_computation import NTKComputer, ModelSpec
spec = ModelSpec(layer_widths=[10, 100, 100, 1], activation='relu', sigma_w=1.414)
computer = NTKComputer()
result = computer.compute_analytical_ntk(spec, X)
```

# TensorGuard API Reference

## Core Verification

### `verify_model(source, input_shapes=None, default_device=Device.CPU, default_phase=Phase.TRAIN, max_k=None)`

One-shot verification of an `nn.Module` defined in source code. Extracts the computation graph, then runs `ConstraintVerifier` to verify shape/device/phase/gradient safety via forward symbolic constraint propagation.

**Location:** `src.model_checker`

```python
from src.model_checker import verify_model, Device, Phase

result = verify_model("""
import torch.nn as nn
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        return self.fc2(self.fc1(x))
""", input_shapes={"x": ("batch", 784)})

print(result.pretty())
# ✓ Model is SAFE
```

**Parameters:**
- `source: str` — Python source containing an `nn.Module` subclass.
- `input_shapes: Dict[str, tuple] | None` — Shape tuples; ints for concrete, strings for symbolic dims.
- `default_device: Device` — Default device (`CPU`, `CUDA_0`, etc.).
- `default_phase: Phase` — `TRAIN` or `EVAL`.
- `max_k: int | None` — Maximum verification depth. Defaults to number of computation steps.

**Returns:** `VerificationResult`

---

## Contract Discovery (Guard-Harvesting)

### `run_shape_cegar(source, input_shapes=None, max_iterations=10, default_device=Device.CPU, default_phase=Phase.TRAIN, max_k=None, enable_quality_filter=True, quality_threshold=0.25)`

Guard-harvesting contract discovery. When input shapes are symbolic, iteratively discovers implicit shape predicates (e.g., `x.shape[-1] == 768`) by analysing Z3 counterexamples. Uses Houdini-style predicate accumulation (Flanagan & Leino, FME 2001) with Z3-backed feasibility checking to distinguish real bugs from spurious counterexamples.

**Location:** `src.shape_cegar`

```python
from src.shape_cegar import run_shape_cegar

result = run_shape_cegar("""
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 10)
    def forward(self, x):
        return self.fc(x)
""", input_shapes={"x": ("batch", "features")})

print(result.summary())
# ShapeCEGAR: SAFE after 2 iterations, 1 predicates discovered [x.shape[-1] == 768]
print(result.is_safe)       # True
print(result.contracts_inferred[0].pretty())
# Net.forward(x): requires [x.shape[-1] == 768]
```

**Parameters:**
- `source: str` — Python source containing an `nn.Module` subclass.
- `input_shapes: Dict[str, tuple] | None` — Shape tuples (use strings for symbolic dims).
- `max_iterations: int` — Maximum CEGAR refinement iterations (default: 10).
- `default_device: Device` — Default device.
- `default_phase: Phase` — Default phase.
- `max_k: int | None` — Maximum verification depth.
- `enable_quality_filter: bool` — Enable predicate quality scoring (default: True).
- `quality_threshold: float` — Minimum quality score for predicate acceptance (default: 0.25).

**Returns:** `CEGARResult` (`ShapeCEGARResult`)

---

## SMT Theory Plugins (UserPropagator)

All four theories are implemented as `z3.UserPropagateBase` subclasses with full push/pop trail support.

### `class BroadcastTheoryPlugin(solver)`

Eager propagation of NumPy-style broadcasting constraints.

**Location:** `src.smt.broadcast_theory`

```python
from src.smt.broadcast_theory import BroadcastTheoryPlugin
import z3

s = z3.Solver()
plugin = BroadcastTheoryPlugin(s)

a, b, c = z3.Ints("a b c")
s.add(plugin.broadcast_result_dim(a, b, c))
s.add(a == 3, b == 1)
assert s.check() == z3.sat   # c == 3
```

**Methods:**
- `broadcast_compatible(shape_a, shape_b) → z3.ExprRef`
- `broadcast_result_dim(dim_a, dim_b, dim_out) → z3.ExprRef`
- `matmul_compatible(shape_a, shape_b) → z3.ExprRef`
- `stride_compatible(shape, stride) → z3.ExprRef`

---

### `class StrideTheoryPlugin(solver)`

Multiplicative stride layout and reshape validation.

**Location:** `src.smt.stride_theory`

```python
from src.smt.stride_theory import StrideTheoryPlugin
import z3

s = z3.Solver()
plugin = StrideTheoryPlugin(s)

d0, d1, d2 = z3.Ints("d0 d1 d2")
s0, s1, s2 = z3.Ints("s0 s1 s2")
s.add(plugin.contiguous_strides([d0, d1, d2], [s0, s1, s2]))
s.add(d0 == 2, d1 == 3, d2 == 4)
assert s.check() == z3.sat  # s0=12, s1=4, s2=1
```

**Methods:**
- `contiguous_strides(shape, strides) → z3.ExprRef`
- `reshape_valid(old_shape, new_shape) → z3.ExprRef`
- `divisibility_constraint(dividend, divisor) → z3.ExprRef`

---

### `class DeviceTheoryPlugin(solver)`

Device consistency (CPU/CUDA) across operations.

**Location:** `src.smt.device_theory`

**Methods:**
- `same_device(dev_a, dev_b) → z3.ExprRef` — Assert two tensors must be on the same device.
- `transfer_device(dev_in, dev_out, target) → z3.ExprRef` — Assert a device transfer operation.
- `inherit_device(dev_in, dev_out) → z3.ExprRef` — Assert device inheritance from input to output.

---

### `class PhaseTheoryPlugin(solver)`

Train/eval phase-dependent behavior (dropout, batchnorm).

**Location:** `src.smt.phase_theory`

```python
from src.smt.phase_theory import PhaseTheoryPlugin
import z3

s = z3.Solver()
plugin = PhaseTheoryPlugin(s)
phase = z3.Bool("phase")
s.add(plugin.set_phase(phase, is_train=False))
urs = z3.Bool("urs")
s.add(plugin.batchnorm_behavior(phase, urs))
assert s.check() == z3.sat
assert z3.is_true(s.model()[urs])  # uses running stats in eval mode
```

**Methods:**
- `set_phase(phase_var, is_train) → z3.ExprRef` — Assert the model phase.
- `dropout_behavior(phase_var, input_active, output_active) → z3.ExprRef` — Assert dropout phase-dependent behaviour.
- `batchnorm_behavior(phase_var, uses_running_stats) → z3.ExprRef` — Assert batchnorm phase-dependent behaviour.

---

## Theory Combination

### `class TensorTheoryCombination`

Tinelli-Zarba (JAR 2005) theory combination for the product theory T_shape × T_device × T_phase. Handles the finite-domain sorts (device: 5 elements, phase: 2 elements) that violate the standard Nelson-Oppen precondition by enumerating arrangements over shared variables.

**Location:** `src.smt.theory_combination`

---

## CLI

Entry point: `src.cli.main`.

### `python -m src.cli verify <file> [options]`

Verify an `nn.Module` via constraint-based verification.

```bash
python -m src.cli verify model.py
python -m src.cli verify model.py -s x=batch,3,224,224
python -m src.cli verify model.py --cegar-iterations 20 --format json
```

**Options:**
- `-s, --input-shape NAME=d1,d2,...` — Input shape (repeatable). Dims can be ints or symbolic strings.
- `--no-device-check` — Disable device consistency checking.
- `--no-phase-check` — Disable train/eval phase checking.
- `--cegar-iterations N` — Max contract discovery iterations (default: 10).
- `-f, --format {text,json,sarif}` — Output format.

---

## Data Types

### `VerificationResult`
- `safe: bool`
- `certificate: SafetyCertificate | None`
- `counterexample: CounterexampleTrace | None`
- `graph: ComputationGraph | None`
- `errors: List[str]`
- `verification_time_ms: float`
- `pretty() → str`

### `SafetyCertificate`
- `model_name: str`
- `properties: List[str]` — e.g. `["shape_compatible", "device_consistent", "gradient_valid"]`
- `k: int` — verification depth
- `checked_steps: int`
- `z3_queries: int`, `z3_total_time_ms: float`, `z3_sat_count: int`, `z3_unsat_count: int`
- `theories_used: List[str]` — e.g. `["QF_LIA", "QF_UF", "T_broadcast", "T_stride"]`
- `product_domains: List[str]` — e.g. `["T_shape", "T_device", "T_phase"]`
- `pretty() → str`
- `smtlib_certificate() → str` — Emit SMT-LIB 2.6 proof certificate (verify with `z3 -smt2`)
- `to_dict() → dict` — JSON-serializable dict with SHA-256 fingerprint

### `CounterexampleTrace`
- `model_name: str`
- `violations: List[SafetyViolation]`
- `failing_step: int`
- `states: List[ModelState]`
- `concrete_dims: Dict[str, int]`
- `pretty() → str`

### `SafetyViolation`
- `kind: str` — `"shape_incompatible"`, `"device_mismatch"`, etc.
- `step_index: int`
- `step: ComputationStep`
- `message: str`
- `tensor_a/tensor_b: str | None`
- `shape_a/shape_b: TensorShape | None`
- `device_a/device_b: Device | None`

### `ShapeCEGARResult`
- `discovered_predicates: List[ShapePredicate]`
- `iterations: int`
- `final_status: CEGARStatus` — `SAFE`, `REAL_BUG_FOUND`, `MAX_ITER`, `NO_Z3`, `PARSE_ERROR`
- `contracts_inferred: List[InferredContract]`
- `verification_result: VerificationResult | None`
- `real_bugs: List[SafetyViolation]`
- `total_time_ms: float`
- `predicate_quality_report: Dict[str, Any] | None`
- `is_safe: bool` (property)
- `has_real_bugs: bool` (property)
- `summary() → str`

### `ShapePredicate`
- `kind: PredicateKind` — `DIM_EQ`, `DIM_GT`, `DIM_GE`, `DIM_DIVISIBLE`, `DIM_MATCH`, `NDIM_EQ`, `SHAPE_EQ`
- `tensor: str`
- `axis: int | None`
- `value: int | None`
- `pretty() → str`

### `InferredContract`
- `function_name: str`
- `parameter: str`
- `predicates: List[ShapePredicate]`
- `pretty() → str`

### `ComputationGraph`
- `class_name: str`
- `layers: Dict[str, LayerDef]`
- `steps: List[ComputationStep]`
- `input_names: List[str]`
- `output_names: List[str]`
- `num_steps: int` (property)
- `pretty() → str`

### Supported Layer Types (`LayerKind`)
`LINEAR`, `CONV2D`, `CONVTRANSPOSE2D`, `BATCHNORM1D`, `BATCHNORM2D`, `LAYERNORM`, `GROUPNORM`, `INSTANCENORM2D`, `DROPOUT`, `RELU`, `SOFTMAX`, `EMBEDDING`, `LSTM`, `GRU`, `MULTIHEAD_ATTENTION`, `MAXPOOL2D`, `AVGPOOL2D`, `ADAPTIVE_AVGPOOL2D`, `FLATTEN`, `SEQUENTIAL`, `MODULELIST`, `IDENTITY`, `UPSAMPLE`

### Supported Operations (`OpKind`)
`LAYER_CALL`, `MATMUL`, `ADD`, `MULTIPLY`, `RESHAPE`, `FLATTEN`, `CAT`, `TRANSPOSE`, `PERMUTE`, `SQUEEZE`, `UNSQUEEZE`, `ACTIVATION`, `DROPOUT`, `SOFTMAX`, `INTERPOLATE`, `TO_DEVICE`, `DETACH`, `CONTIGUOUS`

### `Device` (enum)
`CPU`, `CUDA_0`, `CUDA_1`, `CUDA_2`, `CUDA_3`

### `Phase` (enum)
`TRAIN`, `EVAL`

---

## Intent-Apparent Bug Detection

### `OverwarnAnalyzer()`

Intent-aware bug detector for device, phase, gradient, semantic, and optimizer bugs. Uses AST pattern matching to detect common PyTorch anti-patterns.

**Location:** `src.intent_bugs`

```python
from src.intent_bugs import OverwarnAnalyzer

analyzer = OverwarnAnalyzer()
bugs = analyzer.analyze(source)
for bug in bugs:
    print(f"[{bug.kind.name}] line {bug.line}: {bug.message}")
```

**Returns:** `List[IntentApparentBug]`

Bug kinds: `DEVICE_MISMATCH`, `SHAPE_MISMATCH`, `GRADIENT_BROKEN`, `MISSING_ZERO_GRAD`, `WRONG_SOFTMAX_DIM`, `DOUBLE_ACTIVATION`, `FROZEN_IN_OPTIMIZER`, and others.

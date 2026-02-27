# TensorGuard API Reference

**193 operators** across convolutions, attention, RoPE, MoE routing, einops-style rearrangements, and 145 core `nn.Module` layer types. Verified over a 5-theory product domain (shape × device × phase × stride × permutation).

## Installation

```bash
cd implementation && pip install -e ".[smt]"
```

---

## Core Verification (`src.model_checker`)

### `verify_model()`

One-shot verification of an `nn.Module` defined in source code.

```python
from src.model_checker import verify_model, Device, Phase

result = verify_model(
    source,
    input_shapes={"x": ("batch", 784)},
    produce_certificates=True,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | `str` | *(required)* | Python source code containing an `nn.Module` subclass |
| `input_shapes` | `Dict[str, tuple] \| None` | `None` | Shape tuples; ints for concrete dims, strings for symbolic dims |
| `default_device` | `Device` | `Device.CPU` | Default device |
| `default_phase` | `Phase` | `Phase.TRAIN` | `TRAIN` or `EVAL` |
| `max_k` | `int \| None` | `None` | Maximum verification depth |
| `constraints` | `Dict[str, str \| int] \| None` | `None` | Relational constraints between symbolic dims |
| `high_confidence_only` | `bool` | `False` | Only report HIGH-confidence (Z3-proven) violations |
| `verification_mode` | `str` | `"bounded"` | `"bounded"` for BMC, `"unbounded"` for IC3/PDR |
| `symbolic_dims` | `Dict[str, str] \| None` | `None` | Only for `verification_mode="unbounded"` |
| `use_kb_normalization` | `bool` | `False` | Apply Knuth-Bendix constraint normalization before solving |
| `produce_certificates` | `bool` | `False` | Generate proof certificates with inference chains |
| `return_kripke` | `bool` | `False` | Include the Kripke structure in the result |

**Returns:** `VerificationResult`

### Backward Constraint Propagation

Backward constraint propagation is automatically enabled in `verify_model()` (step 13 of the verification pipeline). After forward propagation, it iterates computation steps in reverse order and adds constraints from each consumer layer's input requirements to the producer layer's output dimensions. This catches mutations (e.g. `wrong_out_features`) where the producer satisfies its own forward constraints but the output is incompatible with the downstream consumer.

No separate API call is needed — backward propagation runs as part of every `verify_model()` invocation. It is implemented via `ConstraintVerifier._backward_constraint_pass()` internally.

### `extract_computation_graph()`

Extract an AST-based computation graph from an `nn.Module` source string.

```python
from src.model_checker import extract_computation_graph

graph = extract_computation_graph(source)
```

### `extract_kripke_structure()`

Extract a formal Kripke structure K = (S, S₀, R, AP, L) from a computation graph.

```python
from src.model_checker import extract_kripke_structure

kripke = extract_kripke_structure(graph, input_shapes={"x": ("batch", 784)})
assert kripke.is_safe()
```

### `VerificationResult`

```python
@dataclass
class VerificationResult:
    safe: bool
    certificate: Optional[SafetyCertificate] = None
    counterexample: Optional[CounterexampleTrace] = None
    graph: Optional[ComputationGraph] = None
    errors: List[str] = field(default_factory=list)
    verification_time_ms: float = 0.0
    confidence: Confidence = Confidence.HIGH
    proof_certificate: Optional[ProofCertificate] = None
    kripke_structure: Optional[KripkeStructure] = None
```

- `filter_by_confidence(min_level: Confidence) -> VerificationResult`

### `SafetyCertificate`

```python
@dataclass
class SafetyCertificate:
    model_name: str
    properties: List[str]
    k: int
    symbolic_bindings: Dict[str, str] = field(default_factory=dict)
    checked_steps: int = 0
    z3_queries: int = 0
    z3_sat_count: int = 0
    z3_unsat_count: int = 0
    theories_used: List[str] = field(default_factory=list)
    product_domains: List[str] = field(default_factory=list)
    proof_certificate: Optional[ProofCertificate] = None
```

- `smtlib_certificate() -> str` — SMT-LIB 2.6 verification conditions.
- `pretty() -> str` — Human-readable summary.
- `to_dict() -> dict` — JSON-serializable dictionary.

### `CounterexampleTrace`

```python
@dataclass
class CounterexampleTrace:
    model_name: str
    violations: List[SafetyViolation] = field(default_factory=list)
    failing_step: int = -1
    states: List[ModelState] = field(default_factory=list)
    concrete_dims: Dict[str, int] = field(default_factory=dict)
```

- `pretty() -> str`

### Enums

```python
class Confidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class Device(Enum):
    CPU = "cpu"
    CUDA_0 = "cuda:0"
    CUDA_1 = "cuda:1"
    CUDA_2 = "cuda:2"
    CUDA_3 = "cuda:3"

class Phase(Enum):
    TRAIN = auto()
    EVAL = auto()
```

---

## CEGAR Contract Discovery (`src.shape_cegar`)

### `run_shape_cegar()`

Counterexample-guided contract discovery using Houdini-style predicate accumulation. Converges in O(k) iterations where k = |P_final \ P_seed|.

```python
from src.shape_cegar import run_shape_cegar

result = run_shape_cegar(
    source=source,
    input_shapes={"x": ("batch", "d")},
    max_iterations=10,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | `str` | *(required)* | Python source with an `nn.Module` subclass |
| `input_shapes` | `Dict[str, tuple] \| None` | `None` | Shape tuples |
| `max_iterations` | `int` | `10` | Maximum CEGAR refinement iterations |
| `default_device` | `Device` | `Device.CPU` | Default device |
| `default_phase` | `Phase` | `Phase.TRAIN` | Default phase |
| `max_k` | `int \| None` | `None` | Maximum verification depth |
| `enable_quality_filter` | `bool` | `True` | Filter low-quality predicates |
| `quality_threshold` | `float` | `0.3` | Minimum predicate quality score |

**Returns:** `ShapeCEGARResult`

### `ShapeCEGARLoop`

Class encapsulating the CEGAR refinement loop with knowledge base integration.

```python
from src.shape_cegar import ShapeCEGARLoop

loop = ShapeCEGARLoop(
    source=source,
    input_shapes={"x": ("batch", "d")},
    max_iterations=10,
    enable_interpolation=True,
    knowledge_base=kb,
)
result = loop.run()
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | `str` | *(required)* | Python source |
| `input_shapes` | `Dict[str, tuple] \| None` | `None` | Shape tuples |
| `max_iterations` | `int` | `10` | Maximum iterations |
| `enable_interpolation` | `bool` | `True` | Use Craig interpolation for predicate discovery |
| `use_incremental` | `bool` | `False` | Use incremental solving |
| `knowledge_base` | `VerificationKnowledgeBase \| None` | `None` | KB for predicate seeding |
| `constraints` | `Dict[str, Union[str, int]] \| None` | `None` | Relational constraints |

- `run() -> ShapeCEGARResult` — Execute the CEGAR loop.

### `verify_and_discover()`

Convenience wrapper returning `(is_safe, predicates, contracts)`.

```python
from src.shape_cegar import verify_and_discover

safe, predicates, contracts = verify_and_discover(
    source, input_shapes={"x": ("batch", "d")}
)
```

### `ShapeCEGARResult`

```python
@dataclass
class ShapeCEGARResult:
    discovered_predicates: List[ShapePredicate] = field(default_factory=list)
    iterations: int = 0
    final_status: CEGARStatus = CEGARStatus.SAFE
    contracts_inferred: List[InferredContract] = field(default_factory=list)
    verification_result: Optional[VerificationResult] = None
    real_bugs: List[SafetyViolation] = field(default_factory=list)
    total_time_ms: float = 0.0
    iteration_log: List[IterationRecord] = field(default_factory=list)
    predicate_quality_report: Optional[Dict[str, Any]] = None
```

### `CEGARStatus`

```python
class CEGARStatus(Enum):
    RUNNING = auto()
    SAFE = auto()
    UNSAFE = auto()
    UNKNOWN = auto()
    TIMEOUT = auto()
```

---

## IC3/PDR Unbounded Verification (`src.ic3_pdr`)

### `ic3_verify()`

Unbounded parametric verification via IC3/PDR. Proves safety for *all* values of symbolic dimensions.

```python
from src.ic3_pdr import ic3_verify

result = ic3_verify(
    model_source=source,
    symbolic_dims={"batch": "batch_size"},
    input_shapes={"x": ("batch", 10)},
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_source` | `str` | *(required)* | Python source with an `nn.Module` subclass |
| `symbolic_dims` | `Dict[str, str] \| None` | `None` | Maps shape positions to symbolic parameter names |
| `input_shapes` | `Dict[str, tuple] \| None` | `None` | Shape tuples |
| `max_k` | `int` | `20` | Maximum IC3 frames |
| `solver_timeout_ms` | `int` | `5000` | Z3 timeout per query (ms) |
| `use_interpolation` | `bool` | `True` | Use Craig interpolation for generalization |

**Returns:** `IC3Result`

### `IC3Result`

```python
@dataclass
class IC3Result:
    safe: bool
    invariant: Optional[str] = None
    counterexample_depth: Optional[int] = None
    frames_computed: int = 0
    verification_time_ms: float = 0.0
    symbolic_dims: Dict[str, str] = field(default_factory=dict)
    num_blocked_cubes: int = 0
    z3_queries: int = 0
    invariant_clauses: List[str] = field(default_factory=list)
    counterexample_trace: Optional[List[Dict[str, Any]]] = None
    frame_sequence: List[Dict[str, Any]] = field(default_factory=list)
```

### `IC3Solver`

Lower-level solver class for IC3/PDR verification.

- `get_frame_sequence_summary() -> List[Dict[str, Any]]` — Returns per-frame clause counts and summaries for visualization.

### `IC3Status`

```python
class IC3Status(Enum):
    SAFE = auto()
    UNSAFE = auto()
    UNKNOWN = auto()
    TIMEOUT = auto()
```

---

## Knowledge Base (`src.knowledge_base`)

### `VerificationKnowledgeBase`

Persistent knowledge base with AGM belief revision for cross-session verification transfer.

```python
from src.knowledge_base import VerificationKnowledgeBase, compute_arch_hash

kb = VerificationKnowledgeBase()

# Record knowledge
arch_hash = compute_arch_hash(resnet18_source)
kb.record(arch_hash, predicates=["dim > 0"], strategies=[{"name": "cegar"}])

# Transfer to similar architecture
transfer = kb.lookup(compute_arch_hash(resnet34_source))
print(transfer.predicates)

# AGM belief revision
kb.revise(arch_hash, "new_pred", depends_on=["dim > 0"])  # Levi identity
removed = kb.contract(arch_hash, "stale_pred")             # Entrenchment-based

# Persistence
kb.save("kb.json")
kb2 = VerificationKnowledgeBase.load("kb.json")
kb.merge(kb2)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `record` | `record(arch_hash, predicates=None, strategies=None, failure_modes=None, proof_certificate=None, layer_types=None)` | Store verification knowledge for an architecture |
| `lookup` | `lookup(arch_hash) -> TransferredKnowledge` | Retrieve transferred knowledge for a new model |
| `revise` | `revise(arch_hash, predicate, depends_on=None) -> List[str]` | AGM revision via Levi identity (contract ¬φ, expand φ) |
| `contract` | `contract(arch_hash, predicate) -> List[str]` | AGM contraction: remove predicate and dependents |
| `invalidate_stale` | `invalidate_stale(arch_hash, max_age_days) -> List[str]` | Remove predicates older than threshold |
| `measure_kb_precision` | `measure_kb_precision(ground_truth, arch_hash) -> Dict[str, float]` | Compute precision/recall vs ground truth |
| `save` | `save(path: str)` | Persist KB to JSON |
| `load` | `VerificationKnowledgeBase.load(path) -> VerificationKnowledgeBase` | Class method: load from JSON |
| `merge` | `merge(other: VerificationKnowledgeBase)` | Union knowledge from another KB |
| `get_transferred_predicates` | `get_transferred_predicates(arch_hash) -> List[str]` | Get all predicates for a family |
| `get_repair_context` | `get_repair_context(arch_hash) -> str` | Get context string for neuro-symbolic repair |

Properties: `families`, `family_count`, `total_predicates`, `get_all_arch_hashes()`, `has_family(arch_hash)`.

### `compute_arch_hash()`

Hash architectural pattern (ignoring parameter values) for cross-session knowledge transfer.

```python
from src.knowledge_base import compute_arch_hash

h = compute_arch_hash(source)  # Returns str
```

### `anti_unify_proof_certificates()`

Extract generalized proof schema via Plotkin-style anti-unification from multiple proof certificates.

```python
from src.knowledge_base import anti_unify_proof_certificates

schema = anti_unify_proof_certificates(certificates, arch_hash="resnet_family")
print(schema.rule_skeleton)
print(schema.variable_positions)
```

**Returns:** `ProofSchema` — Contains `rule_skeleton`, `variable_positions`, `source_count`, `arch_hash`.

---

## Contrastive Explanations (`src.contrastive_explanation`)

### `explain_contrastively()`

Top-level API: generate "Why X and not Y?" explanations from a CEGAR result.

```python
from src.contrastive_explanation import explain_contrastively

result = explain_contrastively(
    cegar_result=cegar_result,
    graph=computation_graph,
    model_name="MyModel",
)
# result["contrastive"] -> List[ContrastiveExplanation]
# result["narrative"]   -> CEGARNarrative
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cegar_result` | `ShapeCEGARResult` | *(required)* | Result from `run_shape_cegar()` |
| `graph` | `ComputationGraph \| None` | `None` | Computation graph for layer-level detail |
| `model_name` | `str \| None` | `None` | Override model name |
| `calibrator` | `ExplanationCalibrator \| None` | `None` | Optional calibrator |

**Returns:** `Dict[str, Any]` — Keys: `"contrastive"`, `"narrative"`, `"calibrated"`.

### `ContrastiveExplainer`

Generates contrastive explanations using Craig interpolants (Miller 2019).

```python
from src.contrastive_explanation import ContrastiveExplainer

explainer = ContrastiveExplainer(timeout_ms=3000, max_foils=3)
explanations = explainer.explain(cegar_result, graph=graph)
for expl in explanations:
    print(expl.render())
```

- `explain(cegar_result, graph=None) -> List[ContrastiveExplanation]` — Generate explanations for all bugs.

### `ContrastiveExplanation`

```python
@dataclass
class ContrastiveExplanation:
    fact_description: str             # Description of the bug
    foil: ContrastiveFoil             # Closest valid configuration
    interpolant_constraint: Optional[str]  # Craig interpolant
    full_text: str = ""               # Rendered explanation
```

- `render() -> str`

### `NarrativeGenerator`

Converts CEGAR traces to natural-language narratives.

```python
from src.contrastive_explanation import NarrativeGenerator

narrator = NarrativeGenerator()
narrative = narrator.generate(cegar_result, graph=graph, model_name="MyModel")
```

- `generate(cegar_result, graph=None, model_name=None) -> CEGARNarrative`

---

## CEGAR Explanation (`src.cegar_explanation`)

### `explain_verification()`

Run CEGAR verification and produce a human-readable explanation in one call.

```python
from src.cegar_explanation import explain_verification

explanation = explain_verification(
    model_source=source,
    input_shapes={"x": ("batch", 10)},
)
print(explanation.render())
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_source` | `str` | *(required)* | Python source with an `nn.Module` subclass |
| `input_shapes` | `Dict[str, tuple] \| None` | `None` | Shape tuples |
| `max_iterations` | `int` | `10` | Maximum CEGAR iterations |
| `model_name` | `str \| None` | `None` | Override the model name |

**Returns:** `VerificationExplanation` — `.render() -> str`, `.to_dict() -> dict`.

### `generate_explanation()`

Lower-level: generate an explanation from an existing `ShapeCEGARResult`.

```python
from src.cegar_explanation import generate_explanation

explanation = generate_explanation(cegar_result, graph=graph, model_name="MyModel")
```

---

## Proof Certificates (`src.proof_certificate`)

### `ProofCertificate`

Obtained via `produce_certificates=True` in `verify_model()`. 100% coverage of safe verdicts.

```python
@dataclass
class ProofCertificate:
    model_name: str
    properties: List[str]
    steps: List[ProofStep]
    root_step: int
    theories_used: List[str] = field(default_factory=list)
    verification_conditions: List[str] = field(default_factory=list)
    certificate_hash: str = ""       # SHA-256
    strategy: Optional[CertificateStrategy] = None
```

- `verify_locally() -> bool` — Structural verification of inference steps.
- `to_alethe() -> str` — Export in Alethe proof format.
- `to_dict() -> dict` — JSON-serializable dictionary.
- `pretty() -> str` — Human-readable summary with stats.
- `summary_stats() -> dict` — Step count, theory lemmas, max depth, rule histogram.

### `CertificateStrategy`

```python
class CertificateStrategy(Enum):
    Z3_NATIVE_PROOF = "z3_native_proof"
    UNSAT_CORE = "unsat_core"
    REPLAY = "replay"
    DUAL_SOLVER = "dual_solver"
```

### `ProofStep`

```python
@dataclass
class ProofStep:
    rule: str          # e.g. "mp", "th-lemma", "unit-resolution"
    conclusion: str    # SMT-LIB formula
    premises: List[int]
    theory: Optional[str]  # "arith", "eq", or None
```

---

## SMT Backend (`src.smt`)

### Theory Plugins (UserPropagators)

Five domain theories, each implemented as a Z3 UserPropagator with push/pop-invertible trail frames:

| Module | Theory | Trail Frame |
|--------|--------|-------------|
| `broadcast_theory.py` | Broadcasting rules | `_TrailFrame` |
| `stride_theory.py` | Memory stride constraints | `_StrideTrailFrame` |
| `device_theory.py` | CPU/CUDA placement | `_DeviceTrailFrame` |
| `phase_theory.py` | Train/eval mode tracking | `_PhaseTrailFrame` |
| `permutation_theory.py` | Axis reordering (transpose, permute) | `_PermTrailFrame` |

### Permutation Theory (`src.smt.permutation_theory`)

```python
from src.smt.permutation_theory import apply_concrete_transpose, apply_concrete_permutation

output = apply_concrete_transpose([10, 20, 30], 0, 1)   # [20, 10, 30]
output = apply_concrete_permutation([10, 20, 30], (2, 0, 1))  # [30, 10, 20]
```

### Theory Combination (`src.smt.theory_combination`)

Verify Nelson-Oppen/Tinelli-Zarba combination preconditions for the 5-theory architecture.

```python
from src.smt.theory_combination import verify_combination_preconditions

report = verify_combination_preconditions()
assert report.all_satisfied
print(report.signature_disjointness)  # All 10 pairs disjoint
```

**Returns:** `PreconditionReport` — `stable_infiniteness`, `polite_witnessability`, `signature_disjointness`, `shared_sort_analysis`.

### Cross-Theory Deduction Propagation (`src.smt.theory_combination`)

Extracts implied constraints from each theory's solver model and propagates them to other theories sharing variables. Handles interactions that standard Nelson-Oppen misses: Shape→Device (matmul device equality), mixed LIA×NIA (reshape partial evaluation), and Stride→Shape (contiguous stride implies dim constraints).

Available on `TheoryCombination` instances:

| Method | Description |
|--------|-------------|
| `propagate_cross_theory_deductions()` | Single pass: extract concrete values from each theory's model, propagate to other theories. Returns list of `(source, target, constraints)` triples. |
| `run_deduction_propagation_loop(max_rounds=3)` | Iterate `propagate_cross_theory_deductions` until fixpoint or `max_rounds`. Returns total deductions propagated. |

The `CrossTheoryDeductionPropagator` facade combines deduction propagation with mixed-arithmetic propagation:

```python
from src.smt.theory_combination import CrossTheoryDeductionPropagator

prop = CrossTheoryDeductionPropagator(combination)
total = prop.propagate_all(max_rounds=3)
```

| Method | Description |
|--------|-------------|
| `propagate_all(max_rounds=3)` | Run full pipeline: cross-theory deductions + mixed-arithmetic LIA reduction. Returns total constraints propagated. |
| `add_reshape_constraints(solver, old_dims, new_dims)` | Add LIA-reduced reshape constraints to a shape theory solver. |

### Mixed-Arithmetic Propagator (`src.smt.theory_combination`)

Reduces QF_NIA reshape constraints to QF_LIA when dimensions are partially concrete. At reshape boundaries, element count preservation creates a nonlinear constraint (`product(old_dims) == product(new_dims)`), but when some dimensions are concrete the product partially evaluates to LIA.

```python
from src.smt.theory_combination import MixedArithmeticPropagator

# Split dimensions into concrete product and symbolic factors
concrete, symbolic = MixedArithmeticPropagator.partial_evaluate_product(dims)

# Generate LIA constraints from NIA reshape element-count
constraints = MixedArithmeticPropagator.generate_lia_reshape_constraints(
    old_dims, new_dims,
)

# Add LIA-reduced reshape constraints to solver
n_added = MixedArithmeticPropagator.propagate_reshape_to_shape_theory(
    solver, old_dims, new_dims,
)
```

| Method | Description |
|--------|-------------|
| `partial_evaluate_product(dims)` | Split dimension list into `(concrete_product, symbolic_factors)` |
| `generate_lia_reshape_constraints(old_dims, new_dims)` | Generate LIA constraints from NIA reshape; returns empty list if shapes are compatible |
| `propagate_reshape_to_shape_theory(solver, old_dims, new_dims)` | Add LIA-reduced constraints to solver; returns count added |

### Distinctness Axioms (`src.smt.distinctness_axioms`)

Generate explicit distinctness + totality axioms for finite SMT sorts, eliminating spurious models.

```python
from src.smt.distinctness_axioms import FiniteSortAxiomGenerator

gen = FiniteSortAxiomGenerator()
axioms = gen.generate_axioms("T_device", ["cpu", "cuda:0", "cuda:1"])
is_tight = gen.verify_tightness(axioms, expected_size=3)
```

### Solver Factory (`src.smt.encoder`, `src.smt.solver`)

```python
from src.smt.solver import create_solver

solver = create_solver("z3")  # or "cvc5", "fallback"
```

### Craig Interpolation (`src.craig_interpolation`)

```python
from src.craig_interpolation import InterpolationPredicateDiscovery, InterpolationMethod

discoverer = InterpolationPredicateDiscovery(method=InterpolationMethod.AUTO)
# Methods: CVC5_NATIVE, Z3_UNSAT_CORE_SIMULATION, AUTO
```

---

## DAG Assume-Guarantee Composition (`src.assume_guarantee`)

### `verify_compositional()`

Sequential assume-guarantee verification with Abadi-Lamport composition rule.

```python
from src.assume_guarantee import verify_compositional

result = verify_compositional(source, input_shapes={"x": ("batch", 768)})
print(f"Safe: {result.safe}  Speedup: {result.speedup_vs_monolithic:.1f}x")
```

### `verify_compositional_dag()`

DAG-aware compositional verification for non-sequential architectures.

```python
from src.assume_guarantee import verify_compositional_dag

result = verify_compositional_dag(source, input_shapes={"x": ("batch", 3, 224, 224)})
print(f"Safe: {result.safe}  Topology: {result.topology}")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | `str` | *(required)* | Python source with an `nn.Module` subclass |
| `input_shapes` | `Dict[str, tuple] \| None` | `None` | Shape tuples |
| `default_device` | `Device` | `Device.CPU` | Default device |
| `default_phase` | `Phase` | `Phase.TRAIN` | Default phase |
| `cache` | `VerificationCache \| None` | `None` | Verification cache for incremental runs |
| `measure_monolithic` | `bool` | `True` | Also run monolithic verification for speedup measurement |

**Returns:** `CompositionalResult`

### `verify_compositional_incremental()`

Incremental re-verification: only re-verify changed submodules, reuse cached results.

```python
from src.assume_guarantee import verify_compositional_incremental

result = verify_compositional_incremental(
    source,
    input_shapes={"x": ("batch", 768)},
    changed_modules={"encoder"},
    cache=previous_cache,
)
```

### `CompositionalResult`

```python
@dataclass
class CompositionalResult:
    safe: bool
    submodule_results: List[...]
    interface_checks: List[InterfaceCheck]
    total_time_ms: float
    speedup_vs_monolithic: float
    cache_hits: int
    decomposition_strategy: DecompositionStrategy
    num_submodules: int
    proof_tree: Optional[ProofTree]
```

### `DAGCompositionProofRule`

Formal DAG composition proof rule generalizing Abadi-Lamport to arbitrary DAGs.

- `from_submodules_and_edges(submodules, edges, topology) -> DAGCompositionProofRule`
- `topological_order() -> List[int]` — Kahn's algorithm for DAG traversal.
- `pretty() -> str`

### `DecompositionStrategy`

```python
class DecompositionStrategy(Enum):
    AUTO = auto()
    LAYER_BOUNDARY = auto()
    BRANCH_MERGE = auto()
    USER_SPECIFIED = auto()
    SINGLE_LAYER = auto()
```

---

## Modern Operators (`src.stdlib.modern_ops`)

Extended operator registry merged at runtime into the core shape algebra. Categories:

| Category | Operators | Transfer semantics |
|----------|-----------|-------------------|
| **Attention** | `scaled_dot_product_attention`, `multi_head_attention_forward`, `cross_attention` | `(B, ..., L, Ev)` output shape |
| **Normalization** | `group_norm`, `instance_norm`, `layer_norm`, `rms_norm` | Shape-preserving |
| **Activations** | `gelu`, `silu`, `mish`, `swish`, `hardswish`, `leaky_relu`, `elu`, `selu`, `prelu`, `softmax`, `sigmoid`, `tanh`, etc. (40+) | Elementwise (shape-preserving) |
| **MoE routing** | `moe_routing`, `moe_gate_scores` | `(..., top_k, hidden)` / `(..., num_experts)` |
| **Positional encoding** | `rotary_embedding` (RoPE), `sinusoidal_pos_encoding` | Shape-preserving / `(seq_len, d_model)` |
| **Einops-style** | `rearrange`, `repeat`, `reduce` | Pattern-based shape reconstruction |
| **Convolutions** | `conv1d`, `conv_transpose1d`, `conv3d` | Standard convolution output formulas |
| **Adaptive pooling** | `adaptive_avg_pool{1,2,3}d`, `adaptive_max_pool{1,2,3}d` | Keeps batch+channel, replaces spatial |
| **Reshape** | `pixel_shuffle`, `pixel_unshuffle`, `unfold`, `fold`, `chunk`, `split` | Category-specific |
| **Embedding/dropout** | `embedding`, `dropout`, `dropout2d`, `stochastic_depth` | Shape-preserving |

Operators are registered in the `MODERN_TORCH_SHAPE_OPS` dictionary and merged into `TORCH_SHAPE_OPS` at import time.

---

## Fragment-Aware Decidability (`src.decidability`)

### `identify_fragments()`

Identify theory fragments (QF_LIA, QF_NIA, QF_BV, etc.) present in verification constraints.

```python
from src.decidability import identify_fragments, classify_query_complexity

fragments = identify_fragments(query)
complexity = classify_query_complexity(query)
```

### `summarize_decidability()`

Full decidability analysis report.

```python
from src.decidability import summarize_decidability

summary = summarize_decidability(query)
# DecidabilitySummary with fragment classification, complexity class, recommendations
```

### `classify_relational_constraint()`

Classify a single constraint as QF_LIA or QF_NIA.

```python
from src.decidability import classify_relational_constraint

info = classify_relational_constraint("x", "y * z", concrete_dims={"y": 3})
# RelationalConstraintInfo with fragment, is_linear, etc.
```

### `analyze_nia_fragment()`

Z3-based NIA satisfiability check with timeout.

```python
from src.decidability import analyze_nia_fragment

result = analyze_nia_fragment({"x": "y * z"}, timeout_ms=5000)
```

### `enforce_decidable_fragment()`

Bound symbolic variables to enforce decidability.

```python
from src.decidability import enforce_decidable_fragment

safe_constraints = enforce_decidable_fragment(constraints)
```

---

## Encoding Completeness (`src.smt.encoding_completeness`)

### `verify_categoricity()`

Verify that finite sort axioms produce a categorical theory (unique model up to isomorphism).

```python
from src.smt.encoding_completeness import verify_categoricity

result = verify_categoricity("device", ["cpu", "cuda_0", "cuda_1", "meta", "mps"])
assert result["is_categorical"]
assert result["exact_cardinality"] == 5
```

### `verify_all_theories()`

Run categoricity verification for all three finite theories (T_device, T_phase, T_perm).

```python
from src.smt.encoding_completeness import verify_all_theories

results = verify_all_theories()
assert all(r["is_categorical"] for r in results.values())
```

### `verify_permutation_group()`

Verify S_n group axioms (identity, closure, associativity, inverse) via exhaustive enumeration.

```python
from src.smt.encoding_completeness import verify_permutation_group

result = verify_permutation_group(n=4)
assert result["all_axioms_satisfied"]
```

---

## Knuth-Bendix Completion (`src.knuth_bendix`)

### `normalize_z3_expr()`

Normalize a Z3 expression using the completed rewrite rules.

```python
from src.knuth_bendix import normalize_z3_expr
import z3

x = z3.Int("x")
normalized = normalize_z3_expr(x + 0)
```

### `verify_confluence()`

Verify confluence by exhaustively enumerating critical pairs and checking joinability.

```python
from src.knuth_bendix import verify_confluence, get_default_rules

rules = get_default_rules()
is_confluent, non_joinable = verify_confluence(rules)
assert is_confluent  # All 7 CPs joinable under RPO
```

### `compute_critical_pairs()`

```python
from src.knuth_bendix import compute_critical_pairs

cps = compute_critical_pairs(rule1, rule2)
```

---

## Thread-Modular Verification (`src.thread_modular`)

### `ThreadModularVerifier`

Flanagan-Qadeer thread-modular verification for TorchDynamo graph-break composition.

```python
from src.thread_modular import ThreadModularVerifier

verifier = ThreadModularVerifier()
verdict = verifier.verify_composition(subgraphs)
print(verdict.verdict)  # CompositionVerdict.COMPOSITION_VERIFIED
```

| Method | Description |
|--------|-------------|
| `verify_composition(subgraphs)` | Verify subgraph composition via thread-modular chaining |
| `infer_contract(graph)` | Infer pre/postcondition contract for a subgraph |
| `infer_transformer(pre, post)` | Infer abstract transformer for inter-break Python |

### `detect_non_monotonic_patterns()`

Detect cross-break unsoundness in adjacent subgraphs. Returns 5 categories: `shape_inversion`, `dimension_routing`, `accumulator`, `conditional_reshape`, `data_dependent_dim`.

---

## Telemetry-Based Confidence (`src.telemetry_confidence`)

### `TelemetryFeatures`

Dataclass with 14 continuous solver telemetry features extracted from verification runs.

| Field | Type | Description |
|-------|------|-------------|
| `z3_queries` | `int` | Total Z3 check() calls |
| `z3_sat_count` | `int` | SAT results |
| `z3_unsat_count` | `int` | UNSAT results |
| `cegar_iterations` | `int` | CEGAR refinement iterations |
| `n_predicates_seed` | `int` | Seed predicates from guard harvesting |
| `n_predicates_final` | `int` | Final predicate count after CEGAR |
| `n_steps` | `int` | Computation steps in model |
| `n_operators` | `int` | Unique operator types |
| `has_broadcast` | `bool` | Broadcast operations present |
| `has_reshape` | `bool` | Reshape operations present |
| `has_permutation` | `bool` | Permutation/transpose present |
| `device_theory_active` | `bool` | Device constraints present |
| `phase_theory_active` | `bool` | Phase constraints present |
| `elapsed_ms` | `float` | Wall-clock verification time |

### `TelemetryConfidenceScorer`

Logistic regression trained on solver telemetry. Achieves Brier resolution RES = 0.115 (vs RES = 0.000 for discrete confidence).

```python
from src.telemetry_confidence import TelemetryConfidenceScorer, TelemetryFeatures

scorer = TelemetryConfidenceScorer()
scorer.fit(features_list, outcomes)
confidence = scorer.predict_confidence(features)  # float in [0, 1]
evaluation = scorer.evaluate(features_list, outcomes)
# {"brier_score": ..., "resolution": ..., "auc_roc": ...}
```

---

## CLI

**Entry point:** `python -m src.cli.main`

| Command | Description |
|---------|-------------|
| `verify` | Verify `nn.Module` architecture |
| `analyze` | Analyze files/directories for refinement type bugs |
| `ci-check` | CI mode with exit codes (0=safe, 1=bug, 2=unknown) |
| `watch` | Watch files and re-analyze incrementally |
| `server` | Start LSP server |
| `export` | Export inferred contracts |

### `verify` options

| Option | Description |
|--------|-------------|
| `-s`, `--input-shape` | Input shape as `name=dim1,dim2,...` (repeatable) |
| `--no-device-check` | Disable device consistency checking |
| `--no-phase-check` | Disable train/eval phase checking |
| `--cegar-iterations N` | Max CEGAR iterations (default: 10) |
| `-f`, `--format` | Output format: `text`, `json`, or `sarif` |
| `--high-confidence` | Only report Z3-proven bugs |

### CI exit codes

| Code | Meaning |
|------|---------|
| `0` | Model is safe |
| `1` | Bug found |
| `2` | Unknown / timed out |
```

---

## Graph Compiler (`src.graph_compiler`)

Multi-strategy computation graph compiler for arbitrary PyTorch models.

### `compile_model()`

Compile a PyTorch model source into a TensorGuard computation graph.

```python
from src.graph_compiler import compile_model

result = compile_model(
    source,
    input_shapes={"x": ("batch", 784)},
    detect_moe=True,
    detect_dynamic=True,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | `str` | required | Python source code containing `nn.Module` subclass |
| `input_shapes` | `dict` | `{}` | Mapping from parameter name to shape tuple |
| `strategy` | `str` | `"auto"` | Extraction strategy: `"auto"`, `"ast"`, `"fx"`, `"dynamo"`, `"conservative"` |
| `detect_moe` | `bool` | `True` | Detect MoE patterns |
| `detect_dynamic` | `bool` | `True` | Detect dynamic control flow |

**Returns:** `CompilationResult` with fields:
- `graph`: `ComputationGraph` — the extracted computation graph
- `strategy`: `str` — which extraction strategy succeeded
- `warnings`: `List[str]` — any warnings (MoE detected, dynamic patterns, etc.)
- `coverage_ratio`: `float` — fraction of ops with known transfer functions
- `compilation_time_ms`: `float` — compilation time

### `verify_arbitrary_model()`

Verify an arbitrary PyTorch model (wrapper around `compile_model` + `verify_model`).

```python
from src.graph_compiler import verify_arbitrary_model

result = verify_arbitrary_model(
    source,
    input_shapes={"x": ("batch", 512)},
    detect_moe=True,
)
```

### `analyze_moe_shapes()`

Analyze shapes through a Mixture of Experts layer.

```python
from src.graph_compiler import analyze_moe_shapes, MoEConfig

config = MoEConfig(num_experts=8, top_k=2)
output_shape, constraints, error = analyze_moe_shapes(input_shape, config)
```

### `analyze_torch_cond()`

Analyze shapes through `torch.cond` dynamic control flow.

```python
from src.graph_compiler import analyze_torch_cond

result = analyze_torch_cond(pred_shape, true_fn_output, false_fn_output)
assert result.is_shape_preserving
```

### `count_registered_transfers()`

Returns the number of registered transfer functions (100+).

---

## Composition Soundness (`src.composition_soundness`)

Formal verification of the 5-theory product domain composition soundness.

### `verify_product_domain_soundness()`

Verify that the five-theory product domain satisfies all Tinelli-Zarba preconditions.

```python
from src.composition_soundness import verify_product_domain_soundness

verdict = verify_product_domain_soundness()
assert verdict.sound
print(verdict.proof_sketch)
```

**Returns:** `SoundnessVerdict` with fields:
- `sound`: `bool` — whether all preconditions are satisfied
- `preconditions`: `List[PreconditionResult]` — detailed results for each check
- `combination_method`: `str` — `"tinelli_zarba_hybrid"` for the 5-theory product
- `complexity_bound`: `str` — arrangement enumeration complexity formula
- `proof_sketch`: `str` — complete proof sketch

### `verify_composition_properties_z3()`

Use Z3 to verify key composition properties (categoricity, transitivity, completeness).

```python
from src.composition_soundness import verify_composition_properties_z3

results = verify_composition_properties_z3()
assert results["device_categoricity"]["is_categorical"]
assert results["equality_transitivity"]["sound"]
```

### `compute_arrangement_complexity()`

Compute the complexity of arrangement enumeration for the product domain.

```python
from src.composition_soundness import compute_arrangement_complexity, ALL_THEORIES

complexity = compute_arrangement_complexity(ALL_THEORIES)
print(complexity["total_arrangements"])  # 150
print(complexity["tractable"])           # True
```

## K-Induction Verification (`src.k_induction`)

### `k_induction_verify()`

Run k-induction verification on a PyTorch `nn.Module`.

```python
from src.k_induction import k_induction_verify, KInductionVerdict

result = k_induction_verify(
    source,
    symbolic_dims={"batch": "B"},
    input_shapes={"x": ("batch", 10)},
    max_k=50,
)
assert result.verdict == KInductionVerdict.SAFE
print(result.k, result.z3_queries, result.verification_time_ms)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_source` | `str` | *(required)* | Python source with nn.Module |
| `symbolic_dims` | `Dict[str, str]` | `None` | Symbolic dimension mapping |
| `input_shapes` | `Dict[str, tuple]` | `None` | Input shape specification |
| `max_k` | `int` | `50` | Maximum induction depth |
| `solver_timeout_ms` | `int` | `5000` | Z3 timeout per query |

**Returns:** `KInductionResult` with `verdict` (SAFE/UNSAFE/UNKNOWN), `k`, `z3_queries`, timing.

### `compare_verification_methods()`

Three-way comparison: IC3/PDR vs k-induction vs BMC.

```python
from src.k_induction import compare_verification_methods

comp = compare_verification_methods(source, model_name="my_model")
print(comp.winner)      # "IC3/PDR"
print(comp.agree)       # True
print(comp.to_dict())   # Full comparison data
```

## Calibration Analysis (`src.calibration_analysis`)

### `compute_calibration_report()`

Compute comprehensive calibration metrics.

```python
from src.calibration_analysis import compute_calibration_report, Prediction

preds = [Prediction(confidence=0.9, predicted_class=1, true_class=1)]
report = compute_calibration_report(preds, n_bins=10)
```

**Returns:** `CalibrationReport` with:
- `ece` — Expected Calibration Error (equal-width bins)
- `adaptive_ece` — Adaptive ECE (equal-mass bins)
- `mce` — Maximum Calibration Error
- `brier_score` — Brier score
- `calibration_component`, `sharpness_component`, `uncertainty_component` — Murphy decomposition
- `ece_bootstrap_ci` — Bootstrap 95% confidence interval for ECE
- `temperature` — Optimal Platt temperature
- `calibration_curve` — List of (predicted, actual) pairs for plotting
- `reliability_diagram` — Per-bin reliability data

### Additional functions

| Function | Description |
|----------|-------------|
| `adaptive_ece(preds, n_bins)` | ECE with equal-mass binning |
| `bootstrap_ece_ci(preds, n_bins, n_bootstrap, alpha)` | Bootstrap CI for ECE |
| `find_optimal_temperature(preds)` | Platt temperature scaling |
| `apply_temperature(preds, T)` | Apply temperature to predictions |
| `calibration_curve_data(preds, n_points)` | Calibration curve data |

## Knowledge Transfer Experiment (`src.knowledge_base`)

### `run_transfer_experiment()`

Empirical validation of cross-session knowledge transfer.

```python
from src.knowledge_base import run_transfer_experiment

result = run_transfer_experiment(
    source_models=[(src1, shapes1), (src2, shapes2)],
    target_models=[(tgt1, shapes3)],
    family_name="resnet",
)
print(result.speedup_ratio)    # e.g. 1.5
print(result.transfer_rate)    # fraction of useful predicates
```

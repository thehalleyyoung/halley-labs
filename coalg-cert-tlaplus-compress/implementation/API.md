# CoaCert-TLA API Reference

Comprehensive API documentation for all public modules, classes, and functions.

---

## Table of Contents

- [Pipeline & CLI](#pipeline--cli)
- [Parser](#coacertparser)
- [Semantics](#coacertsemantics)
- [Explorer](#coacertexplorer)
- [Functor](#coacertfunctor)
- [Learner](#coacertlearner)
- [Bisimulation](#coacertbisimulation)
- [Witness](#coacertwitness)
- [Verifier](#coacertverifier)
- [Properties](#coacertproperties)
- [Specs](#coacertspecs)
- [Evaluation](#coacertevaluation)

---

## Pipeline & CLI

### `coacert.pipeline`

#### `PipelineConfig`

Dataclass configuring the compression pipeline.

```python
@dataclass
class PipelineConfig:
    max_states: int = 100_000
    max_depth: int = 1000
    exploration_strategy: str = "bfs"         # "bfs" | "dfs"
    conformance_depth: int = 10
    max_learning_rounds: int = 200
    hash_algorithm: str = "sha256"
    compact_witness: bool = False
    verify_after_compress: bool = True
    check_properties: bool = True
    differential_test: bool = False
    checkpoint_dir: Optional[str] = None
    resume_from: Optional[str] = None
    verbose: bool = False
    seed: Optional[int] = None
```

#### `PipelineResult`

Dataclass holding all outputs and metrics from a pipeline run.

```python
@dataclass
class PipelineResult:
    original_states: int
    original_transitions: int
    quotient_states: int
    quotient_transitions: int
    learning_rounds: int
    observation_table_rows: int
    counterexamples_processed: int
    witness_size_bytes: int
    witness_hash: str
    witness_verified: bool
    properties_preserved: bool
    property_results: Dict[str, bool]
    differential_test_passed: Optional[bool]
    elapsed_seconds: float
    stage_timings: Dict[str, float]
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `write_witness` | `(path: str) -> None` | Serialize witness to JSON file |
| `write_quotient` | `(path: str) -> None` | Serialize quotient to JSON file |
| `summary_dict` | `() -> Dict[str, Any]` | Summary as dictionary |

#### `Pipeline`

```python
class Pipeline:
    STAGES: List[str]  # ["parse", "explore", "functor", "learn", "witness", "verify"]

    def __init__(self, config: Optional[PipelineConfig] = None): ...

    def run(
        self,
        source: str,
        stage_callback: Optional[Callable[[str, float], None]] = None,
    ) -> PipelineResult: ...
```

**Example:**

```python
from coacert.pipeline import Pipeline, PipelineConfig

config = PipelineConfig(max_states=50_000, conformance_depth=12)
result = Pipeline(config).run(open("spec.tla").read())
print(result.quotient_states, "/", result.original_states)
```

### `coacert.cli`

#### `main`

```python
def main(argv: Optional[Sequence[str]] = None) -> int
```

CLI entry point. Subcommands: `parse`, `explore`, `compress`, `verify`, `benchmark`, `info`.

#### `ProgressBar`

```python
class ProgressBar:
    def __init__(self, total: int, desc: str = "", width: int = 40, enabled: bool = True): ...
    def update(self, n: int = 1) -> None: ...
    def finish(self) -> None: ...
```

---

## `coacert.parser`

### Top-Level Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `parse` | `(source: str) -> Module` | Parse TLA-lite source into an AST |
| `parse_expression` | `(source: str) -> Expression` | Parse a single expression |
| `tokenize` | `(source: str, file: str = "<string>") -> List[Token]` | Tokenize source |
| `token_stream` | `(source: str, file: str = "<string>") -> Iterator[Token]` | Lazy token iterator |
| `pretty_print` | `(node: ASTNode) -> str` | Format AST back to TLA-lite source |
| `check_types` | `(module: Module) -> None` | Run type checker on module (raises on error) |

**Example:**

```python
from coacert.parser import parse, check_types, pretty_print

module = parse("""
---- MODULE Example ----
VARIABLES x, y
Init == x = 0 /\ y = 0
Next == x' = x + 1 /\ y' = y
====
""")
check_types(module)
print(pretty_print(module))
```

### `Lexer`

```python
class Lexer:
    def __init__(self, source: str, file: str = "<string>"): ...
    def tokenize(self) -> List[Token]: ...
    errors: List[LexerError]
```

### `Parser`

```python
class Parser:
    def __init__(self, source: str, file: str = "<string>"): ...
    def parse_module(self) -> Module: ...
```

### `TypeChecker`

```python
class TypeChecker:
    def check(self, module: Module) -> None: ...
```

Raises `TypeError_` on type errors.

### `TypeEnv`

```python
class TypeEnv:
    def push_scope(self) -> None: ...
    def pop_scope(self) -> None: ...
    def bind(self, name: str, typ: TypeAnnotation) -> None: ...
    def lookup(self, name: str) -> Optional[TypeAnnotation]: ...
    def clone(self) -> TypeEnv: ...
```

### `PrettyPrinter`

```python
class PrettyPrinter:
    def format(self, node: ASTNode) -> str: ...
```

### AST Node Hierarchy

All nodes inherit from `ASTNode`.

#### Type Annotations

| Class | Fields | Description |
|-------|--------|-------------|
| `IntType` | — | Integer type |
| `BoolType` | — | Boolean type |
| `StringType` | — | String type |
| `SetType` | `element_type: TypeAnnotation` | Set type |
| `FunctionType` | `domain_type, range_type` | Function type |
| `TupleType` | `element_types: List[TypeAnnotation]` | Tuple type |
| `RecordType` | `field_types: Dict[str, TypeAnnotation]` | Record type |
| `SequenceType` | `element_type: TypeAnnotation` | Sequence type |
| `AnyType` | — | Wildcard type |
| `OperatorType` | `param_types, return_type` | Operator type |

#### Expressions

| Class | Fields | Description |
|-------|--------|-------------|
| `IntLiteral` | `value: int` | Integer constant |
| `BoolLiteral` | `value: bool` | Boolean constant |
| `StringLiteral` | `value: str` | String constant |
| `Identifier` | `name: str` | Variable/constant reference |
| `PrimedIdentifier` | `name: str` | Primed variable (next-state) |
| `OperatorApplication` | `operator, operands, operator_name` | Operator application |
| `SetEnumeration` | `elements: List[Expression]` | `{a, b, c}` |
| `SetComprehension` | `variable, set_expr, predicate, map_expr` | `{x \in S : P(x)}` |
| `FunctionConstruction` | `variable, set_expr, body` | `[x \in S \|-> e]` |
| `FunctionApplication` | `function, argument` | `f[x]` |
| `RecordConstruction` | `fields: Dict[str, Expression]` | `[a \|-> 1, b \|-> 2]` |
| `RecordAccess` | `record, field_name` | `r.field` |
| `TupleLiteral` | `elements: List[Expression]` | `<<a, b>>` |
| `SequenceLiteral` | `elements: List[Expression]` | Sequence literal |
| `QuantifiedExpr` | `quantifier, variables, body` | `\A x \in S : P(x)` |
| `IfThenElse` | `condition, then_expr, else_expr` | Conditional |
| `LetIn` | `definitions, body` | `LET ... IN ...` |
| `CaseExpr` | `arms: List[CaseArm], other` | `CASE ...` |
| `UnchangedExpr` | `variables: List[str]` | `UNCHANGED <<x, y>>` |
| `ExceptExpr` | `base, substitutions` | `[f EXCEPT ![a] = b]` |
| `ChooseExpr` | `variable, set_expr, predicate` | `CHOOSE x \in S : P(x)` |
| `DomainExpr` | `expr` | `DOMAIN f` |

#### Temporal & Action Expressions

| Class | Fields | Description |
|-------|--------|-------------|
| `ActionExpr` | `body` | Action expression |
| `StutteringAction` | `action, variables, is_angle` | `[A]_v` or `<A>_v` |
| `FairnessExpr` | `kind, variables, action` | `WF_v(A)` / `SF_v(A)` |
| `AlwaysExpr` | `expr` | `[]P` (box) |
| `EventuallyExpr` | `expr` | `<>P` (diamond) |
| `LeadsToExpr` | `left, right` | `P ~> Q` |

#### Definitions

| Class | Fields | Description |
|-------|--------|-------------|
| `OperatorDef` | `name, params, body, is_local` | Operator definition |
| `FunctionDef` | `name, variable, set_expr, body, is_local` | Function definition |
| `VariableDecl` | `names: List[str]` | `VARIABLES x, y` |
| `ConstantDecl` | `names: List[str]` | `CONSTANTS C` |
| `Assumption` | `expr` | `ASSUME ...` |

#### Properties

| Class | Fields | Description |
|-------|--------|-------------|
| `InvariantProperty` | `expr` | Invariant to check |
| `TemporalProperty` | `expr` | Temporal property |
| `SafetyProperty` | `expr` | Safety property |
| `LivenessProperty` | `expr` | Liveness property |

#### Module

```python
@dataclass
class Module:
    name: str
    extends: List[str]
    constants: List[ConstantDecl]
    variables: List[VariableDecl]
    definitions: List[Definition]
    assumptions: List[Assumption]
    theorems: List[Theorem]
    properties: List[Property]
    instances: List[InstanceDef]
```

### `ASTVisitor[T]`

Generic visitor pattern base class with `visit_*` methods for each node type.

### Enums

- **`Operator`** — 50+ operators: `LAND`, `LOR`, `LNOT`, `EQ`, `NEQ`, `PLUS`, `MINUS`, `TIMES`, `IN`, `NOTIN`, `UNION`, `INTERSECT`, `SUBSET`, `SUBSETEQ`, `IMPLIES`, `EQUIV`, etc.
- **`TokenKind`** — Token types: `IDENTIFIER`, `INTEGER`, `STRING`, `KEYWORD`, `OPERATOR`, `LPAREN`, `RPAREN`, etc.

---

## `coacert.semantics`

### Top-Level Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `evaluate` | `(expr: Expr, env: Environment) -> TLAValue` | Evaluate expression in environment |
| `compute_successors` | `(state: TLAState, actions) -> List[TLAState]` | Compute successor states |
| `compute_initial_states` | `(module: Module) -> List[TLAState]` | Compute initial states from Init predicate |
| `explore_state_space` | `(module, max_states, callback) -> StateSpace` | Explore full state space |
| `is_action_enabled` | `(state, action) -> bool` | Check if action is enabled in state |
| `detect_stuttering` | `(state1, state2) -> bool` | Detect if two states form a stutter step |

### TLA Values

| Class | Constructor | Description |
|-------|-------------|-------------|
| `TLAValue` | (abstract base) | Base for all TLA+ values |
| `IntValue` | `(int_val: int)` | Integer value |
| `BoolValue` | `(bool_val: bool)` | Boolean value |
| `StringValue` | `(str_val: str)` | String value |
| `SetValue` | `(elements: FrozenSet[TLAValue])` | Set value |
| `FunctionValue` | `(mapping: Dict[TLAValue, TLAValue])` | Function value |
| `TupleValue` | `(elements: Tuple[TLAValue, ...])` | Tuple value |
| `RecordValue` | `(fields: Dict[str, TLAValue])` | Record value |
| `SequenceValue` | `(elements: Tuple[TLAValue, ...])` | Sequence value |
| `ModelValue` | `(name: str)` | Uninterpreted constant |

### `TLAState`

```python
class TLAState:
    def __init__(self, values: Dict[str, TLAValue]): ...
    def hash_value(self) -> str: ...
    def fingerprint(self) -> int: ...
```

### `Environment`

```python
class Environment:
    def bind_constant(self, name: str, value: TLAValue) -> None: ...
    def bind_operator(self, name: str, defn: OperatorDef) -> None: ...
    def bind_variable(self, name: str, value: TLAValue) -> None: ...
    def lookup(self, name: str) -> Optional[TLAValue]: ...
    def push_scope(self) -> None: ...
    def pop_scope(self) -> None: ...
```

### `ActionEvaluator`

```python
class ActionEvaluator:
    def __init__(self, module: Module): ...
    def evaluate_action(self, action_expr, state: TLAState) -> List[TLAState]: ...
    def compute_enabled_actions(self, state: TLAState) -> Set[str]: ...
```

### `ModuleRegistry`

```python
class ModuleRegistry:
    def install_standard_modules(self) -> None: ...
    def install_module(self, name: str, defs: Dict) -> None: ...

def get_default_registry() -> ModuleRegistry: ...
```

**Example:**

```python
from coacert.semantics import evaluate, Environment, TLAState, IntValue

env = Environment()
env.bind_variable("x", IntValue(5))
result = evaluate(parse_expression("x + 1"), env)
# result == IntValue(6)
```

---

## `coacert.explorer`

### `ExplicitStateExplorer`

```python
class ExplicitStateExplorer:
    def __init__(
        self,
        module: Module,
        max_states: int = 100_000,
        max_depth: int = 1000,
        strategy: str = "bfs",
    ): ...

    def explore(
        self,
        callback: Optional[Callable[[ExplorationStats], None]] = None,
    ) -> TransitionGraph: ...
```

**Example:**

```python
from coacert.parser import parse
from coacert.explorer import ExplicitStateExplorer

module = parse(open("spec.tla").read())
explorer = ExplicitStateExplorer(module, max_states=10_000, strategy="bfs")
graph = explorer.explore()
print(graph.stats())  # {"nodes": ..., "edges": ..., ...}
```

### `TransitionGraph`

```python
class TransitionGraph:
    nodes: Dict[str, StateNode]
    edges: Dict[str, TransitionEdge]

    def stats(self) -> Dict[str, Any]: ...
    def to_networkx(self) -> nx.DiGraph: ...
    def to_dict(self) -> Dict[str, Any]: ...
```

### `StateNode` / `TransitionEdge`

```python
@dataclass
class StateNode:
    id: str
    state: Dict[str, Any]
    atomic_props: FrozenSet[str]
    is_initial: bool

@dataclass
class TransitionEdge:
    src: str
    tgt: str
    action: str
    label: str
```

### `ExplorationStats`

```python
@dataclass
class ExplorationStats:
    mode: str
    states_explored: int
    transitions_found: int
    max_depth_reached: int
    states_per_second: float
    elapsed_seconds: float
    deadlock_states: int
    invariant_violations: int
    completed: bool

    def summary(self) -> str: ...
```

### Symmetry & Fairness

```python
class SymmetryDetector:
    def __init__(self, graph: TransitionGraph): ...
    def detect_symmetries(self) -> Set[PermutationGroup]: ...

class FairnessTracker:
    def __init__(self, graph: TransitionGraph): ...
    def compute_fair_paths(self) -> Set[ExecutionTrace]: ...

@dataclass
class FairnessConstraint:
    enabled_set: Set[str]
    taken_set: Set[str]

class ExecutionTrace:
    states: List[str]
    actions: List[str]

class LassoTrace:
    prefix: ExecutionTrace
    lasso: ExecutionTrace
```

---

## `coacert.functor`

### `FCoalgebra`

```python
class FCoalgebra:
    @classmethod
    def from_transition_graph(cls, graph: TransitionGraph) -> FCoalgebra: ...

    def structure(self, state: str) -> FunctorValue: ...
```

The functor is `F(X) = P(AP) × P(X)^Act × Fair(X)`.

**Example:**

```python
from coacert.functor import FCoalgebra

coalgebra = FCoalgebra.from_transition_graph(graph)
obs = coalgebra.structure("s0")
# obs.atomic_props, obs.successors, obs.fairness
```

### Polynomial Functor Components

| Class | Description |
|-------|-------------|
| `FunctorComponent` | Abstract base |
| `PowersetFunctor(element_functor)` | P(X) — powerset |
| `ExponentialFunctor(base, exponent)` | X^Y — exponential |
| `ProductFunctor(components)` | F × G — product |
| `CoproductFunctor(components)` | F + G — coproduct |
| `ConstantFunctor(value)` | K_C — constant |
| `FairnessFunctor(states, constraints)` | Fair(X) — fairness component |
| `CompositeFunctor(inner, outer)` | F ∘ G — composition |
| `KripkeFairnessFunctor` | Combined Kripke + fairness |

### Morphisms

```python
class CoalgebraMorphism:
    source: FCoalgebra
    target: FCoalgebra
    mapping: Dict[str, str]

class MorphismFinder:
    def __init__(self, source: FCoalgebra, target: FCoalgebra): ...
    def find_morphism(self) -> Optional[CoalgebraMorphism]: ...
```

### Stutter Monad

```python
class StutterMonad:
    """Stutter-closure monad T (Beohar-Küpper 2017)."""
    ...

class StutterPath:
    """Path up to stuttering equivalence."""
    ...

class StutterEquivalenceClass:
    """Equivalence class under stuttering."""
    ...
```

### T-Fair Coherence

```python
class TFairCoherenceChecker:
    def __init__(self, coalgebra: FCoalgebra): ...
    def verify(self) -> bool: ...

class CoherenceWitness:
    """Certificate that T-Fair coherence holds."""
    ...

class CoherenceViolation:
    """Witness to T-Fair coherence failure."""
    ...
```

### `CategoricalCoherenceDiagram`

Verifies that δ: T ∘ Fair ⇒ Fair ∘ T commutes as a natural transformation.

```python
class CategoricalCoherenceDiagram:
    def __init__(self) -> None: ...

    def verify_naturality_square(
        self,
        fairness_pairs: List[Tuple[FrozenSet[str], FrozenSet[str]]],
        eta: Dict[str, str],
        morphisms: List[Tuple[str, Dict[str, str]]],
        stutter_classes: List[Any],
    ) -> bool: ...
        """Verify the naturality square for each morphism h.
        Returns True if the square commutes for all morphisms."""
```

### Quotient & Behavioral Equivalence

```python
class BehavioralEquivalence:
    """Bisimilarity relation on coalgebra states."""
    ...

class PartitionRefinement:
    """Partition refinement for functor-based equivalence."""
    ...

class QuotientCoalgebra:
    def __init__(self, original: FCoalgebra, partition: Dict[str, str]): ...
```

---

## `coacert.learner`

### `CoalgebraicLearner`

```python
class CoalgebraicLearner:
    def __init__(
        self,
        membership_oracle: MembershipOracle,
        equivalence_oracle: EquivalenceOracle,
        max_rounds: int = 200,
        conformance_depth: int = 10,
        timeout_seconds: float = 600,
    ): ...

    def learn(self) -> HypothesisCoalgebra: ...

    rounds: int
    table: ObservationTable
```

**Example:**

```python
from coacert.learner import CoalgebraicLearner, MembershipOracle, EquivalenceOracle

learner = CoalgebraicLearner(
    membership_oracle=MembershipOracle(coalgebra),
    equivalence_oracle=EquivalenceOracle(coalgebra, depth=10),
    max_rounds=200,
)
hypothesis = learner.learn()
print(f"Learned {hypothesis.num_states} states in {learner.rounds} rounds")
```

### `LearnerConfig`

```python
@dataclass
class LearnerConfig:
    max_rounds: int = 200
    conformance_depth: int = 10
    max_conformance_depth: int = 50
    random_walks: int = 100
    max_random_length: int = 50
    timeout_seconds: float = 600
    counterexample_strategy: str = "rivest_schapire"
    minimise_counterexamples: bool = True
    adaptive_depth: bool = True
    seed: Optional[int] = None
    confidence: float = 0.99
    stale_round_limit: int = 10
    max_total_queries: int = 10_000_000
    enable_compression: bool = True
    checkpoint_interval: int = 10
    verbose: bool = False
```

### `LearningResult`

```python
@dataclass
class LearningResult:
    success: bool
    hypothesis: Optional[HypothesisCoalgebra]
    final_table: ObservationTable
    rounds: int
    total_membership_queries: int
    total_equivalence_queries: int
    counterexamples: List[Counterexample]
    elapsed_seconds: float
    termination_reason: str
    convergence_data: List[RoundSnapshot]
    checkpoints: List[Dict]

    def summary(self) -> str: ...
```

### `ObservationTable`

```python
class ObservationTable:
    def __init__(self, alphabet: Set[str], suffixes: Set[str]): ...
    def add_row(self, access_seq: str) -> None: ...
    def add_column(self, suffix: str) -> None: ...
    def fill(self, oracle: MembershipOracle) -> None: ...
    def is_closed(self) -> bool: ...
    def is_consistent(self) -> bool: ...

    rows: Dict[str, Dict[str, Any]]
```

### Oracles

```python
class MembershipOracle:
    def __init__(self, system: FCoalgebra): ...
    def query(self, trace: List[str]) -> bool: ...

class EquivalenceOracle:
    def __init__(self, system: FCoalgebra, depth: int = 10): ...
    def query(self, hypothesis: HypothesisCoalgebra) -> Optional[Counterexample]: ...

@dataclass
class Counterexample:
    trace: List[str]
    expected: Any
    actual: Any
```

### Hypothesis Building

```python
class HypothesisBuilder:
    def build(self, table: ObservationTable) -> HypothesisCoalgebra: ...

class CounterexampleProcessor:
    def process(self, ce: Counterexample, table: ObservationTable) -> List[str]: ...
```

---

## `coacert.bisimulation`

### `BisimulationRelation`

```python
class BisimulationRelation:
    """Union-find equivalence relation."""

    def make_equiv(self, state1: str, state2: str) -> None: ...
    def find(self, state: str) -> str: ...
    def classes(self) -> Set[FrozenSet[str]]: ...
```

### `PartitionRefinement`

```python
class PartitionRefinement:
    """Paige-Tarjan O(m log n) partition refinement."""

    def __init__(
        self,
        states: Set[str],
        actions: Set[str],
        transitions: Dict[str, Dict[str, Set[str]]],
        labels: Dict[str, FrozenSet[str]],
    ): ...

    def refine(self) -> RefinementResult: ...
```

### `StutteringBisimulation`

```python
class StutteringBisimulation:
    """Groote-Vaandrager stuttering bisimulation."""

    def __init__(self, graph: TransitionGraph): ...
    def compute(self) -> StutteringResult: ...
```

### `QuotientBuilder`

```python
class QuotientBuilder:
    def __init__(
        self,
        states: Set[str],
        actions: Set[str],
        transitions: Dict,
        labels: Dict,
        partition: Dict[str, str],
        fairness_pairs: Optional[List] = None,
        representative_selector: Optional[Callable] = None,
    ): ...

    def build(self) -> Dict[str, Dict[str, Set[str]]]: ...

    quotient_map: Dict[str, str]      # original state -> representative
    representatives: Set[str]          # set of representative states
```

### `FairnessEquivalence`

```python
class FairnessEquivalence:
    def __init__(self, graph: TransitionGraph, wf_pairs: List, sf_pairs: List): ...
    def compute(self) -> FairEquivalenceResult: ...
```

### `RefinementEngine`

```python
class RefinementEngine:
    """Full pipeline: partition refinement + stuttering + fairness."""

    def __init__(self, strategy: str = "auto"): ...
    def run(self, graph: TransitionGraph, max_iterations: int = 1000) -> EngineResult: ...
```

**Example:**

```python
from coacert.bisimulation import PartitionRefinement

refiner = PartitionRefinement(states, actions, transitions, labels)
result = refiner.refine()
print(f"Partition: {len(result.blocks)} blocks from {len(states)} states")
```

---

## `coacert.witness`

### `MerkleTree`

```python
class MerkleTree:
    def __init__(self, items: List[bytes]): ...
    def root_hash(self) -> bytes: ...
    def proof(self, index: int) -> MerkleProof: ...
    def to_bytes(self) -> bytes: ...

    @classmethod
    def from_bytes(cls, buf: bytes) -> MerkleTree: ...
```

### `SparseMerkleTree`

```python
class SparseMerkleTree:
    """Sparse Merkle tree for large state spaces with Bloom-filter backing."""

    def __init__(self, default_value: bytes): ...
    ...
```

### `MerkleProof`

```python
@dataclass
class MerkleProof:
    leaf: bytes
    siblings: List[bytes]
```

### `TransitionWitness`

```python
class TransitionWitness:
    @classmethod
    def create(
        cls,
        coalgebra: FCoalgebra,
        quotient: QuotientCoalgebra,
        hash_algorithm: str = "sha256",
    ) -> TransitionWitness: ...

    def root_hash(self) -> bytes: ...
    def serialize(self) -> Dict: ...
```

### Other Witness Types

| Class | Description |
|-------|-------------|
| `StutterWitness` | Stutter-equivalence path witness |
| `FairnessWitness` | Fairness preservation witness |
| `WitnessSet` | Collection of all witnesses |
| `CompactWitness` | Compressed representation |
| `BloomFilter` | Probabilistic membership structure |

### Hash Chain

```python
class HashChain:
    def __init__(self, blocks: List[EquivalenceBlock]): ...

class EquivalenceBlock:
    """Hash block binding a state to its equivalence class."""
    ...

class TransitionBlock:
    """Hash block for a witnessed transition."""
    ...
```

**Example:**

```python
from coacert.witness import TransitionWitness

witness = TransitionWitness.create(coalgebra, quotient)
data = witness.serialize()
with open("witness.json", "w") as f:
    json.dump(data, f)
```

---

## `coacert.verifier`

### `verify_witness`

```python
def verify_witness(
    witness_path: str,
    *,
    partial: bool = False,
    sample_fraction: float = 0.1,
) -> VerificationReport
```

Convenience function for full or partial witness verification.

**Example:**

```python
from coacert.verifier import verify_witness

report = verify_witness("witness.json")
assert report.passed
print(report.details)
```

### `WitnessDeserializer`

```python
class WitnessDeserializer:
    def deserialize_file(self, path: str) -> WitnessData: ...
    def deserialize_bytes(self, buf: bytes) -> WitnessData: ...
```

### `HashChainVerifier`

```python
class HashChainVerifier:
    def __init__(self, witness: WitnessData): ...
    def verify_full(self) -> HashVerificationResult: ...
    def verify(self, witness: WitnessData) -> bool: ...
    def spot_check(self, path: str, samples: int = 50) -> bool: ...
```

### `ClosureValidator`

```python
class ClosureValidator:
    """Verify that the encoded relation satisfies bisimulation closure."""

    def __init__(self, witness: WitnessData): ...
    def validate_full(self) -> ClosureResult: ...
```

### `StutteringVerifier` / `FairnessVerifier`

```python
class StutteringVerifier:
    def __init__(self, witness: WitnessData): ...
    def verify(self) -> StutteringResult: ...

class FairnessVerifier:
    def __init__(self, witness: WitnessData): ...
    def verify(self) -> FairnessResult: ...
```

### `VerificationReport`

```python
class VerificationReport:
    passed: bool
    details: Dict[str, bool]

    def add_hash_result(self, result: HashVerificationResult) -> None: ...
    def add_closure_result(self, result: ClosureResult) -> None: ...
    def add_stuttering_result(self, result: StutteringResult) -> None: ...
    def add_fairness_result(self, result: FairnessResult) -> None: ...
    def finalize(self) -> None: ...
```

---

## `coacert.properties`

### Temporal Formulas

```python
class TemporalFormula:  # Abstract base
    ...

# Propositional
class Atomic(proposition: str)
class TrueFormula()
class FalseFormula()
class Not(inner: TemporalFormula)
class And(left: TemporalFormula, right: TemporalFormula)
class Or(left: TemporalFormula, right: TemporalFormula)
class Implies(left: TemporalFormula, right: TemporalFormula)

# Path quantifiers
class ExistsPath(inner: TemporalFormula)
class ForallPath(inner: TemporalFormula)

# Temporal operators
class Next(inner: TemporalFormula)
class Until(left: TemporalFormula, right: TemporalFormula)
class Finally(inner: TemporalFormula)       # ◇
class Globally(inner: TemporalFormula)      # □
class WeakUntil(left: TemporalFormula, right: TemporalFormula)
class Release(left: TemporalFormula, right: TemporalFormula)
```

**Formula helpers:**

```python
def parse_formula(text: str) -> TemporalFormula: ...
def to_nnf(f: TemporalFormula) -> TemporalFormula: ...
def simplify(f: TemporalFormula) -> TemporalFormula: ...
def is_stuttering_invariant(f: TemporalFormula) -> bool: ...
def is_ctl_star_without_next(f: TemporalFormula) -> bool: ...

# CTL shortcuts
def EX(f), EF(f), EG(f), EU(f, g): ...
def AX(f), AF(f), AG(f), AU(f, g): ...
```

### `CTLStarChecker`

```python
class CTLStarChecker:
    def __init__(self, kripke: KripkeAdapter): ...
    def check(self, formula: TemporalFormula) -> CTLCheckResult: ...
```

### Safety Checking

```python
class SafetyChecker:
    def __init__(self, quotient): ...
    def check(self, prop: SafetyProperty) -> SafetyCheckResult: ...
    def check_all(self) -> bool: ...

# Helpers
def make_ap_invariant(ap: str) -> SafetyProperty: ...
def make_exclusion_invariant(ap1: str, ap2: str) -> SafetyProperty: ...
def make_type_invariant(var: str, type_pred: str) -> SafetyProperty: ...
```

### Liveness Checking

```python
class LivenessChecker:
    def __init__(self, quotient, fairness_constraints: List): ...
    def check(self, prop: LivenessProperty) -> LivenessCheckResult: ...

# Helpers
def make_eventually_always(ap: str) -> LivenessProperty: ...
def make_leads_to(p: str, q: str) -> LivenessProperty: ...
def make_weak_fairness(action: str) -> FairnessSpec: ...
def make_strong_fairness(action: str) -> FairnessSpec: ...
```

### Differential Testing

```python
class DifferentialTester:
    def __init__(self, original, quotient, samples: int = 200): ...
    def test(self) -> bool: ...
    def run(self) -> DifferentialReport: ...

class RandomPropertyGenerator:
    def __init__(self, alphabet: Set[str]): ...
    ...
```

---

## `coacert.specs`

### Specification Builders

| Class | Method | Description |
|-------|--------|-------------|
| `TwoPhaseCommitSpec()` | `.build() -> Module` | Two-Phase Commit protocol |
| `PaxosSpec()` | `.build() -> Module` | Single-decree Paxos |
| `PetersonSpec()` | `.build() -> Module` | Peterson's mutual exclusion |
| `LeaderElectionSpec()` | `.build() -> Module` | Chang–Roberts leader election |

Each also provides `.to_source() -> str` for TLA-lite source output.

### `SpecRegistry`

```python
class SpecRegistry:
    def get(self, name: str) -> type: ...
    def get_source(self, name: str) -> str: ...
    def list_specs(self) -> List[str]: ...
    def get_metadata(self, name: str) -> SpecMetadata: ...
```

### `SpecMetadata`

```python
@dataclass
class SpecMetadata:
    name: str
    description: str
    authors: List[str]
    difficulty: str
    num_states_approx: int
```

### AST Construction Helpers

Utility functions for building AST nodes programmatically:

```python
def ident(name: str) -> Identifier: ...
def primed(name: str) -> PrimedIdentifier: ...
def int_lit(val: int) -> IntLiteral: ...
def bool_lit(val: bool) -> BoolLiteral: ...
def make_set_enum(*elements) -> SetEnumeration: ...
def make_conjunction(*exprs) -> Expression: ...
def make_disjunction(*exprs) -> Expression: ...
def make_forall(var, domain, body) -> QuantifiedExpr: ...
def make_exists(var, domain, body) -> QuantifiedExpr: ...
def make_unchanged(*vars) -> UnchangedExpr: ...
def make_wf(vars, action) -> FairnessExpr: ...
def make_sf(vars, action) -> FairnessExpr: ...
def make_always(expr) -> AlwaysExpr: ...
def make_eventually(expr) -> EventuallyExpr: ...
def make_invariant_property(expr) -> InvariantProperty: ...
def make_safety_property(expr) -> SafetyProperty: ...
def make_liveness_property(expr) -> LivenessProperty: ...
```

**Example:**

```python
from coacert.specs import TwoPhaseCommitSpec, SpecRegistry

# Programmatic
module = TwoPhaseCommitSpec().build()

# Via registry
registry = SpecRegistry()
source = registry.get_source("TwoPhaseCommit")
metadata = registry.get_metadata("TwoPhaseCommit")
```

---

## `coacert.evaluation`

### `BenchmarkRunner`

```python
class BenchmarkRunner:
    def run(self, config: BenchmarkConfig) -> BenchmarkSuiteResult: ...
```

### `BenchmarkConfig`

```python
@dataclass
class BenchmarkConfig:
    specs: List[str]
    runs: int = 5
    max_states: int = 100_000
    conformance_depth: int = 10
    output_report: Optional[str] = None
    format: str = "json"
```

### `BenchmarkResult` / `BenchmarkSuiteResult`

```python
@dataclass
class BenchmarkResult:
    spec_name: str
    status: BenchmarkStatus
    time_s: float
    states: int
    compression: float
    verified: bool

@dataclass
class BenchmarkSuiteResult:
    results: List[BenchmarkResult]
    system_info: SystemInfo
```

### Metrics

```python
class MetricsCollector:
    def record(self, spec_name: str, run: int, elapsed_s: float, result: PipelineResult) -> None: ...

@dataclass
class PipelineMetrics:
    state_space: StateSpaceMetrics
    memory: MemoryMetrics
    queries: QueryMetrics
    witness: WitnessMetrics
    throughput: ThroughputMetrics

def aggregate_metrics(metrics: List[PipelineMetrics]) -> AggregatedMetrics: ...
def metrics_to_json(metrics: PipelineMetrics) -> str: ...
def save_metrics_json(path: str, metrics: List[PipelineMetrics]) -> None: ...
def load_metrics_json(path: str) -> List[PipelineMetrics]: ...
```

### Compression Analysis

```python
class CompressionAnalyzer:
    def __init__(self, original_graph, quotient_graph): ...
    def analyze(self) -> CompressionQuality: ...
```

### Correctness Validation

```python
class CorrectnessValidator:
    def __init__(self, original, quotient): ...
    def validate(self) -> ValidationReport: ...

class DifferentialTestEngine:
    def __init__(self, original, quotient): ...
    def run(self) -> DifferentialTestReport: ...
```

### Timing

```python
class Timer:
    def __init__(self): ...
    def start(self) -> None: ...
    def stop(self) -> float: ...

class MultiRunTimer:
    def __init__(self, name: str): ...
    def record(self, elapsed: float) -> None: ...
    def stats(self) -> TimingStats: ...

def format_duration(seconds: float) -> str: ...
```

### Regression Detection

```python
def compare_results(baseline: BenchmarkSuiteResult, current: BenchmarkSuiteResult) -> Dict: ...
def detect_regressions(baseline: BenchmarkSuiteResult, current: BenchmarkSuiteResult) -> List[RegressionInfo]: ...
```

**Example:**

```python
from coacert.evaluation import BenchmarkRunner, BenchmarkConfig

config = BenchmarkConfig(specs=["TwoPhaseCommit", "Peterson"], runs=3)
suite = BenchmarkRunner().run(config)
for r in suite.results:
    print(f"{r.spec_name}: {r.compression:.1%} compression in {r.time_s:.1f}s")
```

---

## Formal Proofs (B1 Extensions)

Module: `coacert.formal_proofs`

### CategoricalDiagramVerifier

Verifies categorical coherence diagrams for the distributive law δ, ensuring naturality
and unit compatibility of the Büchi/Streett fairness functor composition.

```python
from coacert.formal_proofs import CategoricalDiagramVerifier

class CategoricalDiagramVerifier:
    def __init__(self) -> None: ...

    def verify_naturality(
        self,
        stutter_classes: List[Any],
        fairness_pairs: List[Tuple[FrozenSet[str], FrozenSet[str]]],
        eta: Mapping[str, str],
        morphisms: List[Tuple[str, Mapping[str, str]]]
    ) -> DiagramVerificationResult: ...
        """Verify the naturality square for δ.
        For each morphism h: S → Q, checks: δ_Q ∘ T(Fair(h)) = Fair(T(h)) ∘ δ_S."""

    def verify_unit_compatibility(
        self,
        stutter_classes: List[Any],
        fairness_pairs: List[Tuple[FrozenSet[str], FrozenSet[str]]],
        eta: Mapping[str, str]
    ) -> DiagramVerificationResult: ...
        """Verify unit compatibility: δ ∘ η^{Fair} = Fair(η)."""

    def verify_all(
        self,
        stutter_classes: List[Any],
        fairness_pairs: List[Tuple[FrozenSet[str], FrozenSet[str]]],
        eta: Mapping[str, str],
        mu: Mapping[str, str],
        morphisms: Optional[List[Tuple[str, Mapping[str, str]]]] = None
    ) -> Tuple[bool, List[DiagramVerificationResult]]: ...
        """Verify all three categorical coherence diagrams (naturality, unit, multiplication)."""
```

### CTLStarPreservationProof

Structural induction proof that CTL*\X formulas are preserved by coalgebra morphisms
when coherence conditions hold.

```python
from coacert.formal_proofs import CTLStarPreservationProof

class CTLStarPreservationProof:
    def __init__(self) -> None: ...

    def prove(
        self,
        formula: FormulaNode,
        morphism_is_coalgebra_morphism: bool = True,
        coherence_holds: bool = True,
        morphism_is_surjective: bool = True
    ) -> Tuple[bool, List[FormulaInductionStep]]: ...
        """Run the structural induction proof for a CTL*\\X formula.
        Returns (success, list_of_induction_steps)."""

    def verify_streett_acceptance(
        self,
        fairness_pairs: List[Tuple[FrozenSet[str], FrozenSet[str]]],
        morphism: Mapping[str, str],
        stutter_classes: List[Any],
        coherence_holds: bool = True
    ) -> Tuple[bool, List[StreettAcceptanceResult]]: ...
        """Verify Streett acceptance preservation for all fairness pairs."""
```

### ProofObligationTracker

Tracks proof obligations across modules, enforcing dependency ordering and providing
an aggregate discharge summary.

```python
from coacert.formal_proofs import ProofObligationTracker

class ProofObligationTracker:
    def __init__(self) -> None: ...

    def register(
        self,
        obligation_id: str,
        category: ObligationCategory,
        description: str,
        depends_on: Optional[List[str]] = None,
        source_module: str = ""
    ) -> ProofObligation: ...
        """Register a new proof obligation. Returns the created ProofObligation."""

    def discharge(
        self,
        obligation_id: str,
        witness_description: str
    ) -> bool: ...
        """Discharge an obligation if all dependencies are met.
        Returns True if successfully discharged."""

    def aggregate_status(self) -> Dict[str, Any]: ...
        """Generate a unified proof summary with counts by category and status."""
```

### `TFairCoherenceProver`

Produces constructive proof witnesses for the T-Fair coherence condition.

```python
from coacert.formal_proofs.tfair_theorem import TFairCoherenceProver, ProofCertificate

class TFairCoherenceProver:
    def __init__(self, system_id: str = "") -> None: ...

    def prove(
        self,
        stutter_classes: List[Any],
        fairness_pairs: List[Tuple[FrozenSet[str], FrozenSet[str]]],
    ) -> ProofCertificate: ...
        """Produce a constructive proof certificate for T-Fair coherence.
        Returns a ProofCertificate with all obligations tracked."""

    def verify_proof(self, certificate: Optional[ProofCertificate] = None) -> bool: ...
        """Independently verify a proof certificate by re-checking all obligations."""
```

### `ProofCertificate`

Aggregated proof certificate for the T-Fair coherence formalization.

```python
@dataclass
class ProofCertificate:
    system_id: str
    coherence_holds: bool
    preservation_holds: bool
    tfair_witnesses: List[TFairProofWitness]
    preservation_witness: Optional[PreservationProofWitness]
    obligations_total: int
    obligations_discharged: int
    proof_hash: str
    timestamp: float

    def compute_proof_hash(self) -> str: ...
    def to_dict(self) -> Dict[str, Any]: ...
    def to_json(self, indent: int = 2) -> str: ...
```

### `PreservationProver`

Constructive prover for CTL\*\\X and Streett acceptance preservation.

```python
from coacert.formal_proofs.preservation_prover import PreservationProver, PreservationCertificate

class PreservationProver:
    def __init__(self, system_id: str = "") -> None: ...

    def prove(
        self,
        stutter_classes: List[Any],
        fairness_pairs: List[Tuple[FrozenSet[str], FrozenSet[str]]],
        morphism: Mapping[str, str],
        tfair_witnesses: List[TFairProofWitness],
        coalgebra: Any = None,
        quotient: Any = None,
    ) -> PreservationCertificate: ...
        """Produce a preservation certificate verifying CTL*\\X and Streett preservation."""
```

### `PreservationCertificate`

```python
@dataclass
class PreservationCertificate:
    system_id: str
    coherence_holds: bool
    ctl_star_preserved: bool
    streett_preserved: bool
    morphism_verified: bool
    ap_preservation: bool
    successor_preservation: bool
    fairness_preservation: bool
    ctl_star_proof_steps: List[Dict[str, Any]]
    streett_proof_pairs: List[Dict[str, Any]]
    proof_hash: str
    timestamp: float

    def compute_proof_hash(self) -> str: ...
    def to_dict(self) -> Dict[str, Any]: ...
    def to_json(self, indent: int = 2) -> str: ...
```

**Example:**

```python
from coacert.formal_proofs import (
    CategoricalDiagramVerifier, CTLStarPreservationProof, ProofObligationTracker
)

tracker = ProofObligationTracker()
tracker.register("nat-square", ObligationCategory.NATURALITY, "Naturality of δ")
tracker.register("ctl-pres", ObligationCategory.PRESERVATION, "CTL* preservation",
                  depends_on=["nat-square"])

verifier = CategoricalDiagramVerifier()
ok, results = verifier.verify_all(stutter_classes, fairness_pairs, eta, mu)
if ok:
    tracker.discharge("nat-square", "All diagrams commute")

print(tracker.aggregate_status())
```

---

## Learner Extensions (B1 Extensions)

Module: `coacert.learner`

### IncrementalDeepeningOracle

Iteratively increases exploration depth until the learned hypothesis stabilises,
using configurable depth strategies (doubling, linear stepping).

```python
from coacert.learner import IncrementalDeepeningOracle

class IncrementalDeepeningOracle:
    def __init__(
        self,
        learner_factory: Callable[[int], Any],
        oracle_factory: Callable[[Tuple[str, ...]], Any],
        *,
        initial_depth: int = 3,
        max_depth: int = 64,
        concrete_state_bound: Optional[int] = None,
        depth_strategy: str = "double",
        step_size: int = 3,
        stability_threshold: int = 3,
        max_rounds: int = 20,
        timeout: float = 600.0
    ) -> None: ...

    def run(self) -> Tuple[Any, Optional[Any], ConvergenceHistory]: ...
        """Run the incremental deepening protocol.
        Returns (hypothesis, certificate, convergence_history)."""

    def suggest_depth(
        self,
        estimated_states: Optional[int] = None,
        estimated_actions: Optional[int] = None
    ) -> int: ...
        """Suggest an initial depth based on problem size estimates."""
```

### ExactDiameterComputer

Computes the exact diameter (longest shortest path) of a hypothesis automaton
and returns a verifiable certificate.

```python
from coacert.learner import ExactDiameterComputer

class ExactDiameterComputer:
    def __init__(self, hypothesis: Any) -> None: ...

    def compute(self) -> DiameterCertificate: ...
        """Compute the exact diameter and return a certificate
        containing the diameter value and a witness path."""

    def compute_distinguishing_depth(
        self,
        partition: Optional[List[FrozenSet[str]]] = None
    ) -> int: ...
        """Compute the distinguishing depth of a hypothesis partition."""
```

### WMethodTester

Generates and executes a W-method conformance test suite to validate a learned
hypothesis against a membership oracle.

```python
from coacert.learner import WMethodTester

class WMethodTester:
    def __init__(
        self,
        hypothesis: Any,
        oracle: Callable[[Tuple[str, ...]], Any],
        concrete_state_bound: Optional[int] = None,
        timeout: float = 120.0
    ) -> None: ...

    def generate_test_suite(self) -> List[Tuple[str, ...]]: ...
        """Generate the W-method test suite (state cover × middle × characterisation set)."""

    def run_tests(self) -> WMethodResult: ...
        """Execute the full W-method test suite against the oracle.
        Returns a WMethodResult with pass/fail status and any counterexample found."""
```

### `ConformanceCompleteCertificate`

Certificate certifying that conformance testing depth k is sufficient.

```python
from coacert.learner.conformance_gap import ConformanceCompleteCertificate

@dataclass
class ConformanceCompleteCertificate:
    hypothesis_states: int
    concrete_bound: int
    diameter: int
    sufficient_depth: int
    actual_depth: int
    is_sufficient: bool
    error_bound: float
    gap_ratio: float
    details: str

    def summary(self) -> str: ...

    @classmethod
    def build(
        cls,
        hypothesis_states: int,
        concrete_bound: int,
        diameter: int,
        actual_depth: int,
        n_actions: int = 2,
    ) -> ConformanceCompleteCertificate: ...
        """Build a certificate computing sufficient depth k = diam(H) + (m − n + 1)."""
```

### `ConvergenceCertificate`

Certificate emitted when incremental deepening converges.

```python
from coacert.learner.incremental_deepening import ConvergenceCertificate

@dataclass
class ConvergenceCertificate:
    converged: bool
    convergence_round: int
    final_depth: int
    stable_rounds: int
    final_gap_ratio: float
    is_depth_sufficient: bool
    conformance_certificate: Optional[ConformanceCompleteCertificate]
    details: str

    def summary(self) -> str: ...
```

**Example:**

```python
from coacert.learner import IncrementalDeepeningOracle, WMethodTester

oracle_wrapper = IncrementalDeepeningOracle(
    learner_factory=make_learner,
    oracle_factory=make_oracle,
    initial_depth=3,
    depth_strategy="double"
)
hypothesis, cert, history = oracle_wrapper.run()

tester = WMethodTester(hypothesis, membership_oracle, concrete_state_bound=50)
result = tester.run_tests()
print(f"W-method: {'PASS' if result.passed else 'FAIL'}")
```

---

## Evaluation Extensions (B1 Extensions)

Module: `coacert.evaluation`

### PaigeTarjanBaseline

O(m log n) partition-refinement baseline implementing the Paige–Tarjan algorithm.

```python
from coacert.evaluation import PaigeTarjanBaseline

class PaigeTarjanBaseline:
    def __init__(self) -> None: ...

    def compute_bisimulation(self, lts: LTS) -> List[Set[str]]: ...
        """Compute the coarsest stable partition.
        Returns the partition as a list of state-name sets."""
```

### NaiveBisimulation

Simple fixed-point bisimulation baseline (O(n³) worst-case).

```python
from coacert.evaluation import NaiveBisimulation

class NaiveBisimulation:
    def __init__(self) -> None: ...

    def compute(self, lts: LTS) -> List[Set[str]]: ...
        """Compute bisimulation equivalence classes via naive fixed-point iteration.
        Returns the partition as a list of state-name sets."""
```

### BaselineComparisonRunner

Runs multiple algorithms (coacert, Paige–Tarjan, naive) on the same LTS and
collects timing and partition-size metrics.

```python
from coacert.evaluation import BaselineComparisonRunner

class BaselineComparisonRunner:
    def __init__(
        self,
        coacert_runner: Optional[Callable[[str, LTS], AlgorithmRun]] = None,
        num_runs: int = 5,
        timeout: float = 300.0
    ) -> None: ...

    def run_comparison(
        self,
        spec_name: str,
        lts: LTS,
        algorithms: Optional[List[str]] = None
    ) -> ComparisonReport: ...
        """Compare algorithms on a single LTS. Returns a ComparisonReport
        with per-algorithm timing stats and partition sizes."""
```

### StatisticalTest

Hypothesis testing utilities for comparing algorithm performance across runs.

```python
from coacert.evaluation import StatisticalTest

class StatisticalTest:
    def __init__(self, alpha: float = 0.05) -> None: ...

    def welch_t_test(
        self,
        sample_a: Sequence[float],
        sample_b: Sequence[float],
        metric_name: str = "elapsed_seconds",
        label_a: str = "A",
        label_b: str = "B"
    ) -> StatisticalTestResult: ...
        """Perform Welch's t-test comparing two independent samples.
        Returns a StatisticalTestResult with p-value, confidence interval, and effect size."""

    def cohens_d(
        self,
        sample_a: Sequence[float],
        sample_b: Sequence[float]
    ) -> float: ...
        """Compute Cohen's d effect size between two samples."""
```

### ScalabilityRunner

Runs a parameterised benchmark family at increasing sizes and fits complexity curves.

```python
from coacert.evaluation import ScalabilityRunner

class ScalabilityRunner:
    def __init__(
        self,
        algorithm_runner: Optional[Callable[[str, LTS], ScalabilityDataPoint]] = None,
        num_runs: int = 3,
        timeout: float = 300.0
    ) -> None: ...

    def run(self, benchmark: ParameterizedBenchmark) -> ScalabilityReport: ...
        """Run scalability analysis for a benchmark family.
        Returns a ScalabilityReport with data points and fitted curves."""

    def fit_complexity(
        self,
        data_points: List[ScalabilityDataPoint]
    ) -> ComplexityFit: ...
        """Fit time/space measurements to polynomial or exponential models.
        Returns a ComplexityFit with best-fit degree and R² value."""
```

### BloomSoundnessAnalyzer

Formal analysis of Bloom filter false-positive impact on certificate soundness.

```python
from coacert.evaluation import BloomSoundnessAnalyzer

class BloomSoundnessAnalyzer:
    def __init__(
        self,
        target_soundness: float = 0.999,
        hash_bits: int = 256
    ) -> None: ...

    def analyze(
        self,
        m: int,
        k: int,
        n: int,
        verification_checks: int
    ) -> SoundnessBound: ...
        """Perform formal soundness analysis.
        m = filter bits, k = hash count, n = elements, verification_checks = number of checks.
        Returns a SoundnessBound with proof sketch and numerical bound."""

    def sensitivity_analysis(
        self,
        n: int,
        verification_checks: int,
        bits_per_element_range: Optional[Sequence[float]] = None
    ) -> List[SoundnessBound]: ...
        """Generate a sensitivity table varying bits-per-element."""
```

### `VerificationSoundnessAnalyzer`

Computes P(false acceptance) for coalgebraic verification using Bloom filter parameters.

```python
from coacert.evaluation.bloom_soundness import VerificationSoundnessAnalyzer

class VerificationSoundnessAnalyzer:
    def __init__(self, target_soundness: float = 0.999) -> None: ...

    def compute(
        self,
        bloom_bits: int,
        bloom_k: int,
        bloom_n: int,
        quotient_classes: int,
        num_actions: int,
    ) -> SoundnessBound: ...
        """Compute false acceptance probability.
        V = quotient_classes × num_actions verification checks."""

    def recommend_parameters(
        self,
        bloom_n: int,
        quotient_classes: int,
        num_actions: int,
    ) -> AdaptiveBloomConfig: ...
        """Recommend Bloom filter parameters for target soundness."""
```

### `BloomSoundnessCertificate`

Certificate bundling all computed Bloom filter soundness bounds.

```python
from coacert.evaluation.bloom_soundness import BloomSoundnessCertificate

@dataclass
class BloomSoundnessCertificate:
    bloom_bits: int
    bloom_hash_functions: int
    bloom_elements: int
    per_query_fpr: float
    verification_checks: int
    false_acceptance_exact: float
    false_acceptance_union_bound: float
    soundness_level: float
    target_soundness: float
    meets_target: bool
    optimal_bits: int
    optimal_k: int
    full_witness_size_bytes: int
    bloom_witness_size_bytes: int
    compression_ratio: float

    def to_dict(self) -> Dict[str, Any]: ...
    def to_json(self, indent: int = 2) -> str: ...
```

### `AblationStudy`

High-level ablation study that systematically disables pipeline components.

```python
from coacert.evaluation.ablation import AblationStudy

class AblationStudy:
    STANDARD_COMPONENTS: List[AblationComponent]  # STUTTERING, SYMMETRY_BREAKING, FAIRNESS, MERKLE_WITNESS, ADAPTIVE_DEPTH

    def __init__(
        self,
        pipeline_runner: Callable[[Dict[str, Any]], AblationResult],
        components: Optional[List[AblationComponent]] = None,
    ) -> None: ...

    def run(
        self,
        benchmark_name: str,
        base_config: Dict[str, Any],
    ) -> AblationStudyResult: ...
        """Run ablation study and return results with per-component impact."""

    @staticmethod
    def compute_impact(study: AblationStudyResult) -> Dict[str, Dict[str, float]]: ...
        """Compute per-component impact: compression_delta, time_delta, witness_size_delta."""
```

### `StatisticalSummary`

Summary statistics for a set of measurements.

```python
from coacert.evaluation.statistical import StatisticalSummary, compute_summary

@dataclass
class StatisticalSummary:
    values: List[float]
    mean: float
    median: float
    std_dev: float
    ci_lower: float       # 95% CI lower bound
    ci_upper: float       # 95% CI upper bound
    min_val: float
    max_val: float
    n: int

    def to_dict(self) -> Dict[str, Any]: ...
    def to_json(self, indent: int = 2) -> str: ...

def compute_summary(values: Sequence[float]) -> StatisticalSummary: ...
```

### `ExperimentRunner`

Runs all evaluation experiments and saves results.

```python
from coacert.evaluation.run_all_experiments import ExperimentRunner

class ExperimentRunner:
    def __init__(self, output_dir: str = ".benchmarks", verbose: bool = True) -> None: ...

    def run_all(self) -> Dict[str, Any]: ...
        """Run all experiments (Bloom analysis, baseline comparison, scalability)
        and return combined results dict."""

    def run_bloom_analysis(self) -> Dict[str, Any]: ...
    def run_baseline_comparisons(self) -> Dict[str, Any]: ...
    def run_scalability(self) -> Dict[str, Any]: ...
```

### AdaptiveBloomConfig

Computes optimal Bloom filter parameters to meet a target soundness guarantee
within a memory budget.

```python
from coacert.evaluation import AdaptiveBloomConfig

@dataclass
class AdaptiveBloomConfig:
    target_soundness: float = 0.999
    max_memory_bytes: int = 100 * 1024 * 1024

    def compute_optimal(
        self,
        n_elements: int,
        verification_checks: int
    ) -> "AdaptiveBloomConfig": ...
        """Compute and return config with optimal m (bits) and k (hashes)
        for the given element count and target soundness."""
```

**Example:**

```python
from coacert.evaluation import (
    BaselineComparisonRunner, StatisticalTest, BloomSoundnessAnalyzer
)

runner = BaselineComparisonRunner(num_runs=10)
report = runner.run_comparison("TwoPhaseCommit", lts, algorithms=["coacert", "paige_tarjan"])

stats = StatisticalTest(alpha=0.01)
result = stats.welch_t_test(report.times["coacert"], report.times["paige_tarjan"])
print(f"p={result.p_value:.4f}, Cohen's d={result.cohens_d:.2f}")

bloom = BloomSoundnessAnalyzer(target_soundness=0.9999)
bound = bloom.analyze(m=1_000_000, k=7, n=50_000, verification_checks=100_000)
print(f"Soundness ≥ {bound.lower_bound:.6f}")
```

---

## `coacert.symbolic`

### `compute_upper_bound`

Computes an upper bound on reachable states from a parsed TLA+ module.

```python
from coacert.symbolic.state_space_bounds import compute_upper_bound

def compute_upper_bound(module: Any) -> int: ...
    """Compute upper bound m on reachable states.
    For finite-domain TLA-lite, this is the product of all variable domain sizes.
    Falls back to a conservative estimate if domain info is unavailable."""
```

---

## Verifier Extensions (B1 Extensions)

Module: `coacert.verifier`

### TypedWitnessVerifier

Five-phase typed verification pipeline that checks partition consistency, transition
fidelity, fairness coherence, Bloom filter integrity, and certificate completeness.

```python
from coacert.verifier import TypedWitnessVerifier

class TypedWitnessVerifier:
    def __init__(self, witness: WitnessData) -> None: ...

    def verify(self) -> TypedVerificationReport: ...
        """Run all five verification phases:
        1. Partition consistency — every state belongs to exactly one class.
        2. Transition fidelity — compressed transitions faithfully represent originals.
        3. Fairness coherence — Streett pairs preserved under the morphism.
        4. Bloom integrity — Bloom filter membership agrees with explicit sets.
        5. Certificate completeness — all required fields present and well-typed.
        Returns a TypedVerificationReport aggregating per-phase results."""
```

### IndependentChecker

Re-implements core verification checks independently of the main verifier to
enable cross-validation.

```python
from coacert.verifier import IndependentChecker

class IndependentChecker:
    def __init__(self, witness: WitnessData) -> None: ...

    def cross_validate(
        self,
        main_results: Dict[str, bool]
    ) -> CrossValidationReport: ...
        """Compare independent results against main verifier results.
        Raises VerificationDiscrepancy on first disagreement.
        Returns a CrossValidationReport on full agreement."""

    def check_all(self) -> List[IndependentPhaseResult]: ...
        """Run all independent checks and return per-phase results."""
```

### CrossValidationReport

Dataclass summarising the agreement between the main verifier and the independent checker.

```python
from coacert.verifier import CrossValidationReport

@dataclass
class CrossValidationReport:
    main_phases: Dict[str, bool]        # Main verifier phase results
    independent_phases: Dict[str, bool] # Independent checker phase results
    agreements: List[str]               # Phases where both agree
    discrepancies: List[str]            # Phases with disagreement
    overall_agreement: bool             # True if all phases agree
    elapsed_seconds: float              # Cross-validation wall-clock time
```

**Example:**

```python
from coacert.verifier import TypedWitnessVerifier, IndependentChecker

verifier = TypedWitnessVerifier(witness)
report = verifier.verify()
print(f"All phases passed: {report.all_passed}")

checker = IndependentChecker(witness)
xval = checker.cross_validate(report.phase_results)
print(f"Agreement: {xval.overall_agreement} ({len(xval.agreements)}/{len(xval.agreements) + len(xval.discrepancies)} phases)")
```

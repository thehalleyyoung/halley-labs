# ARC API Reference

Developer API reference for the Algebraic Repair Calculus (`arc`) Python package.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [arc.types — Core Type System](#types)
3. [arc.algebra — Three-Sorted Delta Algebra](#algebra)
4. [arc.sql — SQL Semantic Analysis](#sql)
5. [arc.graph — Pipeline DAG](#graph)
6. [arc.planner — Cost-Optimal Repair Planning](#planner)
7. [arc.execution — Saga-Based Repair Executor](#execution)
8. [arc.quality — Data Quality & Drift Detection](#quality)
9. [arc.python\_etl — Python ETL Idiom Analysis](#python-etl)
10. [arc.io — Serialization](#io)
11. [arc.cli — Command-Line Interface](#cli)

---

<a id="quick-start"></a>
## Quick Start

```python
from arc.graph.builder import PipelineBuilder
from arc.types.base import Schema, Column, SQLType, CostEstimate
from arc.algebra.schema_delta import SchemaDelta, RenameColumn
from arc.algebra.composition import CompoundPerturbation
from arc.algebra.propagation import DeltaPropagator
from arc.planner.dp import DPRepairPlanner
from arc.execution.engine import ExecutionEngine

# 1. Build a pipeline DAG
graph = (
    PipelineBuilder("my_pipeline")
    .add_source("raw", schema=Schema("raw", [
        Column("id", SQLType.INTEGER),
        Column("name", SQLType.VARCHAR),
    ]))
    .add_transform("clean", "raw", query="SELECT id, TRIM(name) AS name FROM raw")
    .add_sink("output", "clean")
    .build()
)

# 2. Describe a perturbation (column renamed upstream)
delta = CompoundPerturbation.schema_only(
    SchemaDelta.from_operation(RenameColumn("name", "full_name"))
)

# 3. Propagate through the DAG
propagator = DeltaPropagator(graph)
result = propagator.propagate({"raw": delta})

# 4. Plan an optimal repair
planner = DPRepairPlanner()
plan = planner.plan(graph, {"raw": delta})

# 5. Execute the repair
with ExecutionEngine() as engine:
    outcome = engine.execute_plan(plan, graph, validate=True)
    print(outcome.success)  # True
```

---

<a id="types"></a>
## 1. `arc.types` — Core Type System

**Location:** `arc/types/`

Foundational types shared by every other subpackage: SQL column types, schemas, operator descriptors, typed tuples, multisets, and the error hierarchy.

---

### `arc.types.base` — Base Types

#### Enums

| Enum | Values | Description |
|------|--------|-------------|
| `SQLType` | `INTEGER`, `BIGINT`, `SMALLINT`, `FLOAT`, `DOUBLE`, `DECIMAL`, `NUMERIC`, `VARCHAR`, `TEXT`, `CHAR`, `BOOLEAN`, `DATE`, `TIMESTAMP`, `TIMESTAMPTZ`, `TIME`, `INTERVAL`, `JSON`, `JSONB`, `UUID`, `BYTEA`, `ARRAY`, `NULL`, `UNKNOWN` | SQL column types |
| `SchemaOpType` | `ADD_COLUMN`, `DROP_COLUMN`, `RENAME_COLUMN`, `CHANGE_TYPE` | Schema operation kinds |
| `RowChangeType` | `INSERT`, `DELETE`, `UPDATE` | Row-level change kinds |
| `ActionType` | `RECOMPUTE`, `REWRITE_SQL`, `ADD_COLUMN`, `DROP_COLUMN`, `RENAME_COLUMN`, `CHANGE_TYPE`, `ADD_QUALITY_CHECK`, `REMOVE_QUALITY_CHECK`, `BACKFILL`, `MIGRATE`, `NOOP` | Repair action kinds |
| `EdgeType` | `DATA_FLOW`, `SCHEMA_DEPENDENCY`, `QUALITY_DEPENDENCY` | Pipeline edge types |
| `SQLOperator` | `SOURCE`, `SELECT`, `FILTER`, `JOIN`, `GROUP_BY`, `UNION`, `WINDOW`, `CTE`, `SET_OP`, `TRANSFORM`, `SINK` | SQL operator types |

#### `Column`

```python
@dataclass(frozen=True)
class Column:
    name: str
    sql_type: SQLType
    nullable: bool = True
    primary_key: bool = False
    default_expr: Optional[str] = None
```

#### `Schema`

```python
@dataclass
class Schema:
    name: str
    columns: List[Column]

    @staticmethod
    def empty() -> Schema
    def add_column(self, col: Column) -> Schema
    def drop_column(self, name: str) -> Schema
    def rename_column(self, old: str, new: str) -> Schema
    def get_column(self, name: str) -> Column
    def has_column(self, name: str) -> bool
    def column_names(self) -> List[str]
```

**Example:**

```python
from arc.types.base import Schema, Column, SQLType

schema = Schema("orders", [
    Column("order_id", SQLType.INTEGER, primary_key=True),
    Column("customer", SQLType.VARCHAR),
    Column("total", SQLType.DECIMAL, nullable=False),
])

schema.has_column("customer")   # True
schema.column_names()           # ['order_id', 'customer', 'total']
```

#### `ParameterisedType`

```python
@dataclass(frozen=True)
class ParameterisedType:
    base: SQLType
    precision: Optional[int] = None
    scale: Optional[int] = None
    length: Optional[int] = None
```

#### `CostEstimate`

```python
@dataclass
class CostEstimate:
    row_count: int = 0
    byte_size: int = 0
    compute_seconds: float = 0.0
    io_bytes: int = 0
```

#### `QualityConstraint`

```python
@dataclass
class QualityConstraint:
    name: str
    column: str
    constraint_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
```

#### `AvailabilityContract`

```python
@dataclass
class AvailabilityContract:
    freshness_seconds: float = 3600.0
    max_downtime_seconds: float = 0.0
    sla_percentage: float = 99.9
```

#### `RepairAction`

```python
@dataclass
class RepairAction:
    action_type: ActionType
    node_id: str
    description: str = ""
    sql: str = ""
    cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
```

#### `RepairPlan`

```python
@dataclass
class RepairPlan:
    actions: List[RepairAction]
    total_cost: float
    cost_breakdown: CostBreakdown
    affected_nodes: Set[str]
    is_optimal: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

### `arc.types.operators` — Operator Descriptors

#### `OperatorProperties`

```python
@dataclass
class OperatorProperties:
    deterministic: bool = True
    monotone: bool = False
    idempotent: bool = False
    commutative: bool = False
    preserves_multiplicities: bool = True
    preserves_order: bool = False
```

#### `JoinType` (Enum)

`INNER`, `LEFT`, `RIGHT`, `FULL`, `CROSS`, `SEMI`, `ANTI`, `NATURAL`

#### `AggregateFunction` (Enum)

`COUNT`, `SUM`, `AVG`, `MIN`, `MAX`, `STDDEV`, `VARIANCE`, `FIRST`, `LAST`, `GROUP_CONCAT`

#### `WindowSpec`

```python
@dataclass
class WindowSpec:
    partition_by: List[str] = field(default_factory=list)
    order_by: List[str] = field(default_factory=list)
    frame: Optional[WindowFrame] = None
```

#### `OperatorLineage`

```python
@dataclass
class OperatorLineage:
    input_columns: Dict[str, List[ColumnLineage]]
    output_columns: Dict[str, List[ColumnLineage]]
```

**Functions:**

```python
def get_default_properties(op: SQLOperator) -> OperatorProperties
```

---

### `arc.types.tuples` — Typed Tuples & Multisets

#### `TypedTuple`

```python
class TypedTuple:
    def __init__(self, values: Mapping[str, Any]) -> None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> TypedTuple
    @staticmethod
    def from_row(columns: Sequence[str], values: Sequence[Any]) -> TypedTuple

    def get(self, column: str, default: Any = None) -> Any
    def project(self, columns: Set[str]) -> TypedTuple
    @property
    def columns(self) -> Set[str]
```

#### `MultiSet`

```python
class MultiSet:
    def __init__(self, elements: Optional[Mapping[TypedTuple, int]] = None) -> None

    @staticmethod
    def empty() -> MultiSet
    @staticmethod
    def from_tuples(tuples: Iterable[TypedTuple]) -> MultiSet
    @staticmethod
    def from_dicts(dicts: Iterable[Dict[str, Any]]) -> MultiSet
    @staticmethod
    def from_rows(columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> MultiSet

    def add(self, t: TypedTuple, count: int = 1) -> None
    def remove(self, t: TypedTuple, count: int = 1) -> None
    def contains(self, t: TypedTuple) -> bool
    def multiplicity(self, t: TypedTuple) -> int

    def union(self, other: MultiSet) -> MultiSet
    def intersection(self, other: MultiSet) -> MultiSet
    def difference(self, other: MultiSet) -> MultiSet
    def sum(self, other: MultiSet) -> MultiSet

    def project(self, columns: Set[str]) -> MultiSet
    def filter(self, predicate: Callable[[TypedTuple], bool]) -> MultiSet
    def map_tuples(self, fn: Callable[[TypedTuple], TypedTuple]) -> MultiSet

    def cardinality(self) -> int
    def distinct_count(self) -> int
    def is_empty(self) -> bool
    def to_dicts(self) -> List[Dict[str, Any]]
```

**Example:**

```python
from arc.types.tuples import TypedTuple, MultiSet

row1 = TypedTuple.from_dict({"id": 1, "name": "Alice"})
row2 = TypedTuple.from_dict({"id": 2, "name": "Bob"})

bag = MultiSet.from_tuples([row1, row2, row1])
bag.cardinality()        # 3
bag.distinct_count()     # 2
bag.multiplicity(row1)   # 2

projected = bag.project({"name"})
projected.to_dicts()     # [{'name': 'Alice'}, {'name': 'Alice'}, {'name': 'Bob'}]
```

---

### `arc.types.errors` — Error Hierarchy

All exceptions inherit from `ARCError`. Error codes are defined by `ErrorCode(Enum)`.

```
ARCError
├── SchemaError
│   ├── TypeMismatchError
│   ├── ColumnNotFoundError
│   ├── DuplicateColumnError
│   ├── PrimaryKeyViolationError
│   └── ForeignKeyViolationError
├── TypeCompatibilityError
├── TypeParameterError
├── TypeCastError
├── DeltaError
│   ├── DeltaCompositionError
│   ├── DeltaPropagationError
│   ├── DeltaInversionError
│   ├── DeltaSortMismatchError
│   └── DeltaInteractionError
├── PlannerError
│   ├── InfeasibleRepairError
│   ├── PlannerTimeoutError
│   └── PlannerCycleError
├── ExecutionError
│   ├── CheckpointError
│   ├── RollbackError
│   ├── SagaCompensationError
│   ├── PartialExecutionError
│   └── BackendUnreachableError
├── ValidationError
│   ├── FragmentViolationError
│   ├── QualityViolationError
│   ├── AvailabilityViolationError
│   └── SpecValidationError
├── SerializationError
│   ├── ParseError
│   ├── EncodeError
│   ├── SchemaViolationError
│   └── VersionUnsupportedError
├── GraphError
│   ├── CycleDetectedError
│   ├── NodeNotFoundError
│   ├── EdgeNotFoundError
│   ├── GraphSchemaMismatchError
│   └── GraphMergeConflictError
└── QualityError
    ├── QualityDriftError
    ├── QualityConstraintFailedError
    └── QualityThresholdExceededError
```

**Utility functions:**

```python
def chain_errors(*errors: ARCError) -> ARCError
def wrap(exc: Exception, arc_cls: type[ARCError] = ARCError) -> ARCError
```

---

<a id="algebra"></a>
## 2. `arc.algebra` — Three-Sorted Delta Algebra

**Location:** `arc/algebra/`

Implements the three-sorted delta algebra (**Δ_S**, **Δ_D**, **Δ_Q**) and their composition into compound perturbations, plus propagation, push-through, annihilation detection, and interaction homomorphisms.

---

### `arc.algebra.schema_delta` — Schema Delta Monoid (Δ_S, ∘, id)

#### `ColumnDef`

```python
@dataclass(frozen=True)
class ColumnDef:
    name: str
    sql_type: SQLType
    nullable: bool = True
    default_expr: Optional[str] = None
    position: int = -1
```

#### Schema Operations

```python
class SchemaOperation(ABC):
    """Base class for all schema operations."""
    def inverse(self) -> SchemaOperation
    def apply(self, schema: Schema) -> Schema
    def affected_columns(self) -> Set[str]
    def is_identity(self) -> bool
    def to_dict(self) -> Dict[str, Any]

@dataclass(frozen=True)
class AddColumn(SchemaOperation):
    name: str
    sql_type: SQLType
    position: int = -1
    default_expr: Optional[str] = None
    nullable: bool = True

@dataclass(frozen=True)
class DropColumn(SchemaOperation):
    name: str

@dataclass(frozen=True)
class RenameColumn(SchemaOperation):
    old_name: str
    new_name: str

@dataclass(frozen=True)
class ChangeType(SchemaOperation):
    column_name: str
    old_type: SQLType
    new_type: SQLType
    coercion_expr: Optional[str] = None
```

#### `SchemaDelta`

```python
class SchemaDelta:
    """Schema delta monoid (Δ_S, ∘, id)."""

    def __init__(self, operations: Optional[List[SchemaOperation]] = None) -> None

    @staticmethod
    def identity() -> SchemaDelta
    @staticmethod
    def from_operation(op: SchemaOperation) -> SchemaDelta
    @staticmethod
    def from_operations(ops: Sequence[SchemaOperation]) -> SchemaDelta

    def compose(self, other: SchemaDelta) -> SchemaDelta
    def inverse(self) -> SchemaDelta
    def normalize(self) -> SchemaDelta
    def apply(self, schema: Schema) -> Schema
    def is_identity(self) -> bool
```

**Example:**

```python
from arc.algebra.schema_delta import (
    SchemaDelta, AddColumn, DropColumn, RenameColumn, ChangeType,
)
from arc.types.base import Schema, Column, SQLType

schema = Schema("t", [Column("a", SQLType.INTEGER), Column("b", SQLType.VARCHAR)])

# Compose: rename b → c, then add column d
d1 = SchemaDelta.from_operation(RenameColumn("b", "c"))
d2 = SchemaDelta.from_operation(AddColumn("d", SQLType.BOOLEAN))
combined = d1.compose(d2)

new_schema = combined.apply(schema)
new_schema.column_names()   # ['a', 'c', 'd']

# Inverse undoes the delta
assert combined.inverse().compose(combined).is_identity()
```

---

### `arc.algebra.data_delta` — Data Delta Group (Δ_D, ∘, ⁻¹, 𝟎)

#### Data Operations

```python
class DataOperation(ABC):
    """Base class for all data operations."""
    def inverse(self) -> DataOperation
    def apply(self, relation: MultiSet) -> MultiSet
    def affected_rows_count(self) -> int
    def is_zero(self) -> bool
    def to_dict(self) -> Dict[str, Any]

@dataclass(frozen=True)
class InsertOp(DataOperation):
    tuples: MultiSet
    def inverse(self) -> DataOperation   # → DeleteOp

@dataclass(frozen=True)
class DeleteOp(DataOperation):
    tuples: MultiSet
    def inverse(self) -> DataOperation   # → InsertOp

@dataclass(frozen=True)
class UpdateOp(DataOperation):
    old_tuples: MultiSet
    new_tuples: MultiSet
    def inverse(self) -> DataOperation   # → UpdateOp(new, old)
    def to_delete_insert(self) -> Tuple[DeleteOp, InsertOp]
    def changed_columns(self) -> Set[str]
```

#### `DataDelta`

```python
class DataDelta:
    """Data delta group (Δ_D, ∘, ⁻¹, 𝟎)."""

    def __init__(self, operations: Optional[List[DataOperation]] = None) -> None

    @staticmethod
    def zero() -> DataDelta
    @staticmethod
    def from_operation(op: DataOperation) -> DataDelta
    @staticmethod
    def from_operations(ops: Sequence[DataOperation]) -> DataDelta
    @staticmethod
    def insert(tuples: MultiSet) -> DataDelta
    @staticmethod
    def delete(tuples: MultiSet) -> DataDelta
    @staticmethod
    def update(old_tuples: MultiSet, new_tuples: MultiSet) -> DataDelta
    @staticmethod
    def from_diff(old_relation: MultiSet, new_relation: MultiSet) -> DataDelta

    def compose(self, other: DataDelta) -> DataDelta
    def inverse(self) -> DataDelta
    def normalize(self) -> DataDelta
    def apply(self, relation: MultiSet) -> MultiSet
    def is_zero(self) -> bool
```

**Example:**

```python
from arc.algebra.data_delta import DataDelta, InsertOp
from arc.types.tuples import MultiSet, TypedTuple

rows = MultiSet.from_dicts([{"id": 3, "name": "Charlie"}])
delta = DataDelta.insert(rows)

# Apply to an existing relation
relation = MultiSet.from_dicts([
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"},
])
new_relation = delta.apply(relation)
new_relation.cardinality()   # 3

# Inverse cancels the insert
assert delta.compose(delta.inverse()).is_zero()
```

---

### `arc.algebra.quality_delta` — Quality Delta Lattice (Δ_Q, ⊔, ⊓, ⊥, ⊤)

#### Enums

| Enum | Values |
|------|--------|
| `ViolationType` | `NULL_IN_NON_NULL`, `UNIQUENESS_VIOLATION`, `FOREIGN_KEY_VIOLATION`, `CHECK_VIOLATION`, `TYPE_MISMATCH`, `RANGE_VIOLATION`, `PATTERN_VIOLATION`, `CUSTOM_RULE_VIOLATION`, `REFERENTIAL_INTEGRITY`, `DOMAIN_VIOLATION`, `STATISTICAL_OUTLIER`, `COMPLETENESS_VIOLATION`, `TIMELINESS_VIOLATION`, `CONSISTENCY_VIOLATION` |
| `SeverityLevel` | `NONE` (0), `INFO` (1), `WARNING` (2), `ERROR` (3), `CRITICAL` (4), `FATAL` (5) |

#### Quality Operations

```python
class QualityOperation(ABC):
    """Base class for all quality operations in Δ_Q."""
    def to_dict(self) -> Dict[str, Any]
    def severity(self) -> SeverityLevel

@dataclass(frozen=True)
class QualityViolation(QualityOperation):
    constraint_name: str
    violation_type: ViolationType
    column: str
    severity: SeverityLevel = SeverityLevel.ERROR

@dataclass(frozen=True)
class QualityImprovement(QualityOperation):
    constraint_name: str
    improvement_type: ViolationType
    fixes: int = 0
    severity: SeverityLevel = SeverityLevel.INFO

@dataclass(frozen=True)
class ConstraintAdded(QualityOperation):
    constraint_name: str
    constraint_type: ConstraintType
    column: str = ""

@dataclass(frozen=True)
class ConstraintRemoved(QualityOperation):
    constraint_name: str

@dataclass(frozen=True)
class DistributionShift(QualityOperation):
    column_name: str
    old_distribution: DistributionSummary
    new_distribution: DistributionSummary
    shift_magnitude: float
    severity: SeverityLevel = SeverityLevel.WARNING
```

#### `QualityDelta`

```python
class QualityDelta:
    """Quality delta lattice (Δ_Q, ⊔, ⊓, ⊥, ⊤)."""

    def __init__(self, operations: Optional[List[QualityOperation]] = None) -> None

    @staticmethod
    def bottom() -> QualityDelta
    @staticmethod
    def top() -> QualityDelta

    def join(self, other: QualityDelta) -> QualityDelta
    def meet(self, other: QualityDelta) -> QualityDelta
    def max_severity(self) -> SeverityLevel
    def violations(self) -> List[QualityViolation]
    def improvements(self) -> List[QualityImprovement]
    def to_dict(self) -> Dict[str, Any]
```

**Example:**

```python
from arc.algebra.quality_delta import (
    QualityDelta, QualityViolation, ViolationType, SeverityLevel,
)

v1 = QualityViolation("nn_check", ViolationType.NULL_IN_NON_NULL, "email")
v2 = QualityViolation("range_check", ViolationType.RANGE_VIOLATION, "age", SeverityLevel.WARNING)

delta = QualityDelta([v1, v2])
delta.max_severity()       # SeverityLevel.ERROR
delta.violations()         # [v1, v2]

# Lattice join merges violations
merged = delta.join(QualityDelta.bottom())
assert merged.max_severity() == SeverityLevel.ERROR
```

---

### `arc.algebra.composition` — Compound Perturbations

#### `CompoundPerturbation`

Bundles all three delta sorts into a single perturbation object.

```python
class CompoundPerturbation:
    def __init__(
        self,
        schema_delta: Optional[SchemaDelta] = None,
        data_delta: Optional[DataDelta] = None,
        quality_delta: Optional[QualityDelta] = None,
    ) -> None

    @property
    def schema_delta(self) -> SchemaDelta
    @property
    def data_delta(self) -> DataDelta
    @property
    def quality_delta(self) -> QualityDelta

    @staticmethod
    def identity() -> CompoundPerturbation
    @staticmethod
    def schema_only(schema_delta: SchemaDelta) -> CompoundPerturbation
    @staticmethod
    def data_only(data_delta: DataDelta) -> CompoundPerturbation
    @staticmethod
    def quality_only(quality_delta: QualityDelta) -> CompoundPerturbation

    def compose(self, other: CompoundPerturbation) -> CompoundPerturbation
    def inverse(self) -> CompoundPerturbation
    def is_identity(self) -> bool
    def severity(self) -> float
    def affected_columns(self) -> Set[str]
    def summary(self) -> Dict[str, Any]

    def has_schema_changes(self) -> bool
    def has_data_changes(self) -> bool
    def has_quality_changes(self) -> bool
    def total_operation_count(self) -> int

    def to_dict(self) -> Dict[str, Any]
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CompoundPerturbation
```

#### `PipelineState`

```python
class PipelineState:
    def copy(self) -> PipelineState
    def row_count(self) -> int
    def column_names(self) -> List[str]
    def quality_score(self) -> float
    def has_violations(self) -> bool
```

**Functions:**

```python
def compose_chain(perturbations: List[CompoundPerturbation]) -> CompoundPerturbation
def compose_parallel(perturbations: List[CompoundPerturbation]) -> CompoundPerturbation
def diff_states(old_state: PipelineState, new_state: PipelineState) -> CompoundPerturbation

def verify_composition_associativity(p1, p2, p3: CompoundPerturbation) -> bool
def verify_identity(p: CompoundPerturbation) -> bool
def verify_inverse(p: CompoundPerturbation, state: PipelineState) -> bool
```

**Example:**

```python
from arc.algebra.composition import CompoundPerturbation, compose_chain
from arc.algebra.schema_delta import SchemaDelta, RenameColumn
from arc.algebra.data_delta import DataDelta
from arc.types.tuples import MultiSet

# Schema + data perturbation
p = CompoundPerturbation(
    schema_delta=SchemaDelta.from_operation(RenameColumn("old_col", "new_col")),
    data_delta=DataDelta.insert(MultiSet.from_dicts([{"new_col": 42}])),
)

p.severity()                # 0.0–1.0 severity score
p.has_schema_changes()      # True
p.total_operation_count()   # 2

# Chain multiple perturbations
combined = compose_chain([p, CompoundPerturbation.identity(), p.inverse()])
```

---

### `arc.algebra.propagation` — Delta Propagation

#### `DeltaPropagator`

```python
class DeltaPropagator:
    """Propagates perturbations through the pipeline DAG (Algorithm A1)."""

    def __init__(
        self,
        graph: PipelineGraph,
        mode: PropagationMode = PropagationMode.FULL,
        enable_annihilation: bool = True,
        enable_interaction: bool = True,
    ) -> None

    def propagate(
        self,
        source_deltas: Dict[str, CompoundPerturbation],
        source_nodes: Optional[Set[str]] = None,
    ) -> PropagationResult
```

#### `PropagationMode` (Enum)

`FULL`, `INCREMENTAL`, `LAZY`, `EAGER`, `BATCHED`

#### `PropagationResult`

```python
@dataclass
class PropagationResult:
    node_deltas: Dict[str, CompoundPerturbation]
    affected_nodes: Set[str]
    annihilated_nodes: Set[str]
    propagation_mode: PropagationMode
    total_time_seconds: float
    errors: Tuple[str, ...] = ()
```

**Example:**

```python
from arc.algebra.propagation import DeltaPropagator, PropagationMode

propagator = DeltaPropagator(graph, mode=PropagationMode.INCREMENTAL)
result = propagator.propagate({"source_node": perturbation})

print(result.affected_nodes)      # {'clean', 'output'}
print(result.annihilated_nodes)   # set() or nodes where delta vanishes
```

---

### `arc.algebra.push` — Operator Push-Through

```python
def push_schema_delta(
    op: SQLOperator,
    context: OperatorContext,
    delta_s: SchemaDelta,
) -> SchemaDelta

def push_data_delta(
    op: SQLOperator,
    context: OperatorContext,
    delta_d: DataDelta,
) -> DataDelta

def push_quality_delta(
    op: SQLOperator,
    context: OperatorContext,
    delta_q: QualityDelta,
) -> QualityDelta
```

#### `OperatorContext`

```python
@dataclass
class OperatorContext:
    operator_type: SQLOperator
    input_schema: Optional[Schema] = None
    output_schema: Optional[Schema] = None
    column_mapping: Dict[str, str] = field(default_factory=dict)
    join_type: Optional[JoinType] = None
    join_condition: str = ""
    filter_predicate: str = ""
    window_spec: str = ""
    cte_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Example:**

```python
from arc.algebra.push import push_schema_delta, OperatorContext
from arc.types.base import SQLOperator

ctx = OperatorContext(
    operator_type=SQLOperator.FILTER,
    filter_predicate="age > 18",
)
output_delta = push_schema_delta(SQLOperator.FILTER, ctx, schema_delta)
```

---

### `arc.algebra.annihilation` — Annihilation Detection

Detects when a delta becomes the identity after propagation through an operator — the operator "kills" the perturbation.

---

### `arc.algebra.interaction` — Interaction Homomorphisms

```python
class PhiHomomorphism:
    """Δ_S → Δ_D interaction: schema changes that force data changes."""
    ...

class PsiHomomorphism:
    """Δ_D → Δ_Q interaction: data changes that cause quality violations."""
    ...
```

---

<a id="sql"></a>
## 3. `arc.sql` — SQL Semantic Analysis

**Location:** `arc/sql/`

SQL parsing (via sqlglot), column-level lineage tracing, Fragment-F checking, predicate analysis, and SQL rewriting.

---

### `arc.sql.parser` — SQL Parsing

#### `SQLParser`

```python
class SQLParser:
    def __init__(self, dialect: Dialect = Dialect.DUCKDB) -> None

    def parse(self, sql: str) -> ParsedQuery
    def parse_many(self, sql: str) -> List[ParsedQuery]
```

#### `Dialect` (Enum)

`DUCKDB`, `POSTGRES`, `MYSQL`, `BIGQUERY`, `SNOWFLAKE`, `SQLITE`, `GENERIC`

#### `ParsedQuery`

```python
@dataclass
class ParsedQuery:
    sql: str
    ast: Any                             # sqlglot AST
    tables_referenced: List[str]
    columns_referenced: List[str]
    operator_type: SQLOperatorType
```

---

### `arc.sql.lineage` — Column-Level Lineage

#### `LineageAnalyzer`

```python
class LineageAnalyzer:
    def __init__(self, catalog: Optional[SchemaCatalog] = None) -> None

    def analyze(self, sql: str) -> ColumnLineage
    def trace_column(self, sql: str, column: str) -> List[SourceColumn]
```

#### `ColumnLineage`

```python
@dataclass
class ColumnLineage:
    entries: Dict[str, ColumnLineageEntry]

    def sources_of(self, column: str) -> List[SourceColumn]
    def dependents_of(self, source: str) -> List[str]
```

**Functions:**

```python
def build_lineage_graph(
    queries: List[str],
    tables: Dict[str, Schema],
) -> LineageGraph

def trace_impact(
    lineage: LineageGraph,
    source: str,
    target: str,
) -> List[str]
```

**Example:**

```python
from arc.sql.lineage import LineageAnalyzer

analyzer = LineageAnalyzer()
lineage = analyzer.analyze("SELECT a.id, b.name FROM a JOIN b ON a.id = b.id")
lineage.sources_of("name")   # [SourceColumn(table='b', column='name')]
```

---

### `arc.sql.rewriter` — SQL Rewriting

#### `SQLRewriter`

```python
class SQLRewriter:
    def __init__(self, dialect: RewriteDialect = RewriteDialect.DUCKDB) -> None

    def rewrite_for_delta(
        self,
        sql: str,
        schema_delta: SchemaDeltaSpec,
    ) -> RewriteResult

    def add_quality_filters(
        self,
        sql: str,
        constraints: List[QualityConstraint],
    ) -> str
```

**Functions:**

```python
def rewrite_for_schema_delta(sql: str, delta: SchemaDeltaSpec, ...) -> RewriteResult
def add_quality_filters(sql: str, constraints: List[QualityConstraint]) -> str
def generate_diff(table1: str, table2: str) -> str
def generate_merge(source: str, target: str) -> str
```

**Example:**

```python
from arc.sql.rewriter import SQLRewriter

rewriter = SQLRewriter()
result = rewriter.rewrite_for_delta(
    "SELECT name FROM users",
    SchemaDeltaSpec(renames={"name": "full_name"}),
)
print(result.sql)   # SELECT full_name FROM users
```

---

### `arc.sql.fragment` — Fragment-F Checking

```python
class FragmentChecker:
    def check(self, sql: str) -> FragmentResult
    def check_pipeline(self, graph: PipelineGraph) -> List[NodeFragmentResult]

def check_fragment_f(sql: str) -> FragmentResult
def is_deterministic_query(sql: str) -> bool
```

---

### `arc.sql.catalog` — Schema Catalog

```python
class SchemaCatalog:
    def register(self, name: str, schema: Schema) -> None
    def get(self, name: str) -> Schema
    def diff(self, name: str, new_schema: Schema) -> SchemaDiff

def diff_schemas(schema1: Schema, schema2: Schema) -> SchemaDiff
def schemas_compatible(schema1: Schema, schema2: Schema) -> bool
def merge_schemas(schemas: List[Schema]) -> Schema
```

---

<a id="graph"></a>
## 4. `arc.graph` — Pipeline DAG

**Location:** `arc/graph/`

Pipeline graph construction, topological analysis, impact analysis, graph transformations, and multi-format visualization.

---

### `arc.graph.pipeline` — Core Graph Types

#### `PipelineNode`

```python
@attr.s(frozen=True)
class PipelineNode:
    node_id: str
    operator: SQLOperator = SQLOperator.TRANSFORM
    query_text: str = ""
    input_schema: Schema = Schema.empty()
    output_schema: Schema = Schema.empty()
    quality_constraints: tuple[QualityConstraint, ...] = ()
    availability_contract: AvailabilityContract = AvailabilityContract()
    cost_estimate: CostEstimate = CostEstimate()
    properties: OperatorProperties = None
    metadata: NodeMetadata = NodeMetadata()

    @property
    def in_fragment_f(self) -> bool
    def with_output_schema(self, schema: Schema) -> PipelineNode
    def with_input_schema(self, schema: Schema) -> PipelineNode
    def with_cost(self, cost: CostEstimate) -> PipelineNode
    def to_dict(self) -> dict[str, Any]
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PipelineNode
```

#### `PipelineEdge`

```python
@attr.s(frozen=True)
class PipelineEdge:
    source: str
    target: str
    column_mapping: dict[str, str] = {}
    edge_type: EdgeType = EdgeType.DATA_FLOW
    label: str = ""
```

#### `PipelineGraph`

```python
class PipelineGraph:
    def __init__(self, name: str = "", version: str = "1.0") -> None

    @property
    def nodes(self) -> dict[str, PipelineNode]
    @property
    def edges(self) -> list[PipelineEdge]

    # Mutation
    def add_node(self, node: PipelineNode) -> None
    def add_edge(self, edge: PipelineEdge) -> None
    def remove_node(self, node_id: str) -> None
    def remove_edge(self, source: str, target: str) -> None

    # Query
    def get_node(self, node_id: str) -> PipelineNode
    def has_node(self, node_id: str) -> bool
    def has_edge(self, source: str, target: str) -> bool
    def parents(self, node_id: str) -> list[str]
    def children(self, node_id: str) -> list[str]
    def ancestors(self, node_id: str) -> set[str]
    def descendants(self, node_id: str) -> set[str]
    def reachable_from(self, node_id: str) -> set[str]

    # Structural
    def topological_order(self) -> list[str]
    def validate(self) -> list[str]
    def find_cycles(self) -> list[list[str]]
    def node_count(self) -> int
    def edge_count(self) -> int

    # Serialization
    def to_dict(self) -> dict[str, Any]
    def to_json(self) -> str
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PipelineGraph
    @classmethod
    def from_json(cls, json_str: str) -> PipelineGraph
```

---

### `arc.graph.builder` — Fluent Pipeline Builder

#### `PipelineBuilder`

```python
class PipelineBuilder:
    def __init__(self, name: str = "", version: str = "1.0") -> None

    def add_source(
        self,
        node_id: str,
        schema: Schema | None = None,
        quality_constraints: Sequence[QualityConstraint] = (),
        availability: AvailabilityContract | None = None,
        cost: CostEstimate | None = None,
        metadata: NodeMetadata | None = None,
    ) -> PipelineBuilder

    def add_transform(
        self,
        node_id: str,
        *upstream_ids: str,
        operator: SQLOperator = SQLOperator.TRANSFORM,
        query: str = "",
        output_schema: Schema | None = None,
        quality_constraints: Sequence[QualityConstraint] = (),
        cost: CostEstimate | None = None,
        properties: OperatorProperties | None = None,
        metadata: NodeMetadata | None = None,
        column_mappings: dict[str, dict[str, str]] | None = None,
        edge_type: EdgeType = EdgeType.DATA_FLOW,
    ) -> PipelineBuilder

    def add_sink(
        self,
        node_id: str,
        *upstream_ids: str,
        output_schema: Schema | None = None,
        metadata: NodeMetadata | None = None,
        edge_type: EdgeType = EdgeType.DATA_FLOW,
    ) -> PipelineBuilder

    def add_node(self, node: PipelineNode) -> PipelineBuilder
    def add_edge(
        self,
        source: str,
        target: str,
        column_mapping: dict[str, str] | None = None,
        edge_type: EdgeType = EdgeType.DATA_FLOW,
    ) -> PipelineBuilder

    def add_sql_chain(
        self,
        source_id: str,
        steps: Sequence[tuple[str, str]],
        output_schema: Schema | None = None,
    ) -> PipelineBuilder

    def build(self, validate: bool = True) -> PipelineGraph

    @classmethod
    def from_json_file(cls, path: str | Path) -> PipelineBuilder
    @classmethod
    def from_yaml_file(cls, path: str | Path) -> PipelineBuilder
```

**Convenience functions:**

```python
def build_linear_pipeline(
    name: str,
    stages: Sequence[tuple[str, SQLOperator, Schema]],
) -> PipelineGraph

def build_diamond_pipeline(
    name: str,
    source_id: str, source_schema: Schema,
    left_id: str, right_id: str, merge_id: str,
    output_schema: Schema | None = None,
) -> PipelineGraph

def build_star_pipeline(
    name: str,
    center_id: str, center_schema: Schema,
    satellite_ids: Sequence[str],
) -> PipelineGraph
```

**Example:**

```python
from arc.graph.builder import PipelineBuilder
from arc.types.base import Schema, Column, SQLType, SQLOperator

graph = (
    PipelineBuilder("etl")
    .add_source("users", schema=Schema("users", [
        Column("id", SQLType.INTEGER), Column("email", SQLType.VARCHAR),
    ]))
    .add_source("orders", schema=Schema("orders", [
        Column("id", SQLType.INTEGER), Column("user_id", SQLType.INTEGER),
    ]))
    .add_transform("joined", "users", "orders",
        operator=SQLOperator.JOIN,
        query="SELECT u.id, u.email, o.id AS order_id FROM users u JOIN orders o ON u.id = o.user_id",
    )
    .add_sink("report", "joined")
    .build()
)
```

---

### `arc.graph.analysis` — Impact & Structural Analysis

#### `impact_analysis`

```python
def impact_analysis(graph: PipelineGraph, node_id: str) -> ImpactResult
```

Returns downstream nodes affected by a change at `node_id`.

#### `ImpactResult`

```python
@dataclass
class ImpactResult:
    source_node: str
    affected_nodes: Set[str]
    affected_edges: Set[Tuple[str, str]]
    severity_by_node: Dict[str, float]
```

#### Other Analysis Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `dependency_analysis` | `(graph, node_id) → DependencyResult` | Upstream dependencies of a node |
| `detect_bottlenecks` | `(graph, top_k=5) → list[BottleneckResult]` | Find high-cost bottleneck nodes |
| `detect_redundancies` | `(graph) → list[RedundancyResult]` | Find redundant computations |
| `compute_metrics` | `(graph) → PipelineMetrics` | Graph-wide metrics (depth, width, density) |
| `schema_impact_propagation` | `(graph, node_id) → Dict[str, Any]` | Trace schema change impact |
| `compute_repair_scope` | `(graph, node_id) → Set[str]` | Minimal set of nodes to repair |
| `compare_repair_vs_recompute` | `(graph, node_id) → RepairVsRecomputeComparison` | Cost comparison |
| `analyze_delta_annihilation` | `(graph, source_id, target_id) → AnnihilationResult` | Check if delta vanishes along path |
| `compute_execution_waves` | `(graph) → list[ExecutionWave]` | Parallelisable execution waves |
| `trace_column_lineage` | `(graph, node_id, column) → SchemaLineageEntry` | Trace a column through the DAG |
| `diff_graphs` | `(before, after) → GraphDiff` | Structural diff between two graphs |

**Example:**

```python
from arc.graph.analysis import impact_analysis, compute_execution_waves

impact = impact_analysis(graph, "users")
print(impact.affected_nodes)   # {'joined', 'report'}

waves = compute_execution_waves(graph)
for wave in waves:
    print(f"Wave {wave.level}: {wave.node_ids}")
```

---

### `arc.graph.visualization` — Multi-Format Export

```python
def to_dot(graph: PipelineGraph, highlight: Set[str] = None) -> str
def save_dot(graph: PipelineGraph, filepath: str, format: str = "png") -> None
def to_ascii(graph: PipelineGraph) -> str
def to_mermaid(graph: PipelineGraph) -> str
def node_summary_table(graph: PipelineGraph) -> str
def node_detail(graph: PipelineGraph, node_id: str) -> str
def render_impact(graph: PipelineGraph, node_id: str) -> str
def cost_heatmap(graph: PipelineGraph) -> str
def render_waves(graph: PipelineGraph) -> str
def schema_map(graph: PipelineGraph) -> str
def render_fragment_f(graph: PipelineGraph) -> str
```

**Example:**

```python
from arc.graph.visualization import to_mermaid, to_ascii

print(to_ascii(graph))
# users ─┬─► joined ──► report
# orders ─┘

print(to_mermaid(graph))
# graph LR
#   users --> joined
#   orders --> joined
#   joined --> report
```

---

### `arc.graph.transform` — Graph Transformations

```python
def collapse_chains(graph: PipelineGraph) -> PipelineGraph
def remove_dead(graph: PipelineGraph, live_sinks: Set[str]) -> PipelineGraph
def insert_checkpoints(graph: PipelineGraph, threshold: float = 0.5) -> PipelineGraph
def optimize_graph(graph: PipelineGraph) -> PipelineGraph
```

---

<a id="planner"></a>
## 5. `arc.planner` — Cost-Optimal Repair Planning

**Location:** `arc/planner/`

Given a pipeline graph and source perturbations, compute the minimum-cost set of repair actions that restore correctness.

---

### `arc.planner.dp` — DP Planner (Optimal, Acyclic DAGs)

#### `DPRepairPlanner`

```python
class DPRepairPlanner:
    """Optimal repair planner for acyclic pipeline DAGs (Algorithm A3).

    Uses dynamic programming over the topological order.
    Runs in O(|V| · 2^k) where k = max columns tracked.
    """

    def __init__(
        self,
        cost_model: CostModel | None = None,
        enable_annihilation: bool = True,
        max_columns_tracked: int = 20,
    ) -> None

    def plan(
        self,
        graph: PipelineGraph,
        deltas: dict[str, CompoundPerturbation],
        cost_model: CostModel | None = None,
    ) -> RepairPlan
```

**Example:**

```python
from arc.planner.dp import DPRepairPlanner
from arc.planner.cost import CostModel, CostFactors

planner = DPRepairPlanner(
    cost_model=CostModel(CostFactors.fast_local()),
    enable_annihilation=True,
)
plan = planner.plan(graph, {"source": perturbation})

print(f"Total cost: {plan.total_cost:.4f}")
print(f"Optimal: {plan.is_optimal}")         # True
for action in plan.actions:
    print(f"  {action.node_id}: {action.action_type.name}")
```

---

### `arc.planner.lp` — LP Planner (Approximate, General Graphs)

#### `LPRepairPlanner`

```python
class LPRepairPlanner:
    """(ln k + 1)-approximation repair planner for general topologies (Algorithm A4).

    Uses LP relaxation + randomised rounding + greedy feasibility patch
    + local search improvement. Works for cyclic graphs.
    """

    def __init__(
        self,
        cost_model: CostModel | None = None,
        seed: int = 42,
        rounding_threshold: float = 0.5,
        max_local_search_iterations: int = 100,
    ) -> None

    def plan(
        self,
        graph: PipelineGraph,
        deltas: dict[str, CompoundPerturbation],
        cost_model: CostModel | None = None,
    ) -> RepairPlan
```

---

### `arc.planner.cost` — Cost Model

#### `CostFactors`

```python
@attr.s(auto_attribs=True)
class CostFactors:
    compute_cost_per_row: float = 1e-6
    io_cost_per_byte: float = 1e-9
    materialization_cost: float = 0.01
    network_cost_per_byte: float = 5e-9
    hash_join_factor: float = 1.2
    sort_factor: float = 1.5
    parallelism_speedup: float = 4.0
    recompute_overhead: float = 1.1
    incremental_discount: float = 0.3

    @classmethod
    def default(cls) -> CostFactors
    @classmethod
    def fast_local(cls) -> CostFactors
    @classmethod
    def cloud(cls) -> CostFactors

    def scale_compute(self, factor: float) -> CostFactors
    def scale_io(self, factor: float) -> CostFactors
```

#### `CostModel`

```python
class CostModel:
    def __init__(self, factors: CostFactors | None = None) -> None

    def estimate_recompute_cost(
        self,
        node: PipelineNode,
        input_sizes: dict[str, int] | None = None,
    ) -> float

    def estimate_incremental_cost(
        self,
        node: PipelineNode,
        delta_size: int,
    ) -> float

    def estimate_plan_cost(self, plan: RepairPlan) -> float
    def operator_complexity(
        self,
        operator: SQLOperator,
        input_sizes: list[int],
        node: PipelineNode | None = None,
    ) -> float
```

---

### `arc.planner.optimizer` — Post-Optimisation

#### `PlanOptimizer`

```python
class PlanOptimizer:
    """Post-optimisation passes for repair plans.

    Passes (all optional):
      1. Merge compatible actions
      2. Parallelise independent actions
      3. Insert checkpoints
      4. Prune redundant actions
      5. Reorder for locality
      6. Estimate parallelism speedup
    """

    def __init__(
        self,
        cost_model: CostModel | None = None,
        enable_merge: bool = True,
        enable_parallel: bool = True,
        enable_checkpoints: bool = True,
        enable_prune: bool = True,
        enable_locality: bool = True,
        checkpoint_interval: int = 5,
    ) -> None

    def optimize(self, plan: RepairPlan, graph: PipelineGraph) -> RepairPlan
```

---

### `arc.planner.strategy` — Strategy Selection

```python
class RepairStrategySelector:
    def select(
        self,
        graph: PipelineGraph,
        perturbation: CompoundPerturbation,
    ) -> RepairStrategy

def select_strategy(
    graph: PipelineGraph,
    perturb: CompoundPerturbation,
) -> RepairStrategy

def compare_strategies(
    graph: PipelineGraph,
    perturb: CompoundPerturbation,
) -> StrategyComparison
```

---

<a id="execution"></a>
## 6. `arc.execution` — Saga-Based Repair Executor

**Location:** `arc/execution/`

Executes repair plans against DuckDB with checkpoint/rollback, incremental processing, and post-repair validation.

---

### `arc.execution.engine` — Execution Engine

#### `ExecutionEngine`

```python
class ExecutionEngine:
    """DuckDB-based repair plan execution engine."""

    def __init__(
        self,
        database_path: str | None = None,
        read_only: bool = False,
    ) -> None

    def __enter__(self) -> ExecutionEngine
    def __exit__(self, exc_type, exc_val, exc_tb) -> None
    def close(self) -> None

    def execute_plan(
        self,
        plan: RepairPlan,
        graph: PipelineGraph,
        validate: bool = True,
    ) -> ExecutionResult

    def execute_action(
        self,
        action: RepairAction,
        graph: PipelineGraph,
    ) -> ActionResult

    def execute_sql(self, query: str) -> Any
    def register_table(self, name: str, data: Any) -> None
    def create_table(self, name: str, schema: Schema, data: Any) -> None

    def get_table_schema(self, table_name: str) -> list[dict[str, str]]
    def get_table_stats(self, table_name: str) -> TableStats
    def get_table_data(self, table_name: str, limit: int = 0) -> list[dict[str, Any]]
```

**Example:**

```python
from arc.execution.engine import ExecutionEngine

with ExecutionEngine(":memory:") as engine:
    # Register data
    engine.register_table("users", users_df)

    # Execute the repair plan
    result = engine.execute_plan(plan, graph, validate=True)

    if result.success:
        print(f"Repaired {len(result.action_results)} nodes")
        repaired = engine.get_table_data("output", limit=10)
    else:
        print(f"Failed: {result.errors}")
```

---

### `arc.execution.checkpoint` — Checkpointing & Rollback

#### `CheckpointManager`

```python
class CheckpointManager:
    """Atomic checkpointing and rollback for repair execution."""

    def __init__(
        self,
        engine: ExecutionEngine,
        prefix: str = "_arc_ckpt_",
    ) -> None

    def create_checkpoint(self, metadata: dict[str, Any] | None = None) -> str
    def save_table_state(self, checkpoint_id: str, table_name: str) -> None
    def restore_checkpoint(self, checkpoint_id: str) -> None
    def commit_checkpoint(self, checkpoint_id: str) -> None
    def list_checkpoints(self) -> list[CheckpointInfo]
    def cleanup_old_checkpoints(self, max_age_seconds: float) -> int
    def get_checkpoint_size(self, checkpoint_id: str) -> int
```

**Example:**

```python
from arc.execution.checkpoint import CheckpointManager

mgr = CheckpointManager(engine)
ckpt = mgr.create_checkpoint({"reason": "pre-repair"})
mgr.save_table_state(ckpt, "users")

try:
    engine.execute_plan(plan, graph)
    mgr.commit_checkpoint(ckpt)
except Exception:
    mgr.restore_checkpoint(ckpt)  # rollback on failure
```

---

### `arc.execution.validation` — Repair Validation

#### `RepairValidator`

```python
class RepairValidator:
    """Validates repair correctness against full recomputation."""

    def __init__(
        self,
        engine: ExecutionEngine,
        default_epsilon: float = 1e-6,
    ) -> None

    def validate_plan(
        self,
        plan: RepairPlan,
        graph: PipelineGraph,
    ) -> ValidationResult

    def validate_fragment_f(
        self,
        repaired_table: str,
        recomputed_table: str,
    ) -> ValidationResult

    def validate_general(
        self,
        repaired_table: str,
        recomputed_table: str,
        epsilon: float | None = None,
    ) -> ValidationResult
```

---

### `arc.execution.incremental` — Incremental Execution

```python
class IncrementalExecutor:
    def execute(
        self,
        graph: PipelineGraph,
        node_id: str,
        delta: DataDelta,
    ) -> IncrementalResult

def execute_incremental(
    graph: PipelineGraph,
    node_id: str,
    delta: DataDelta,
) -> IncrementalResult

def create_data_delta(old_data: Any, new_data: Any) -> DataDelta
```

---

### `arc.execution.materialization` — Materialized Views

```python
class MaterializationManager:
    def materialize(self, graph: PipelineGraph, node_id: str) -> MaterializedView
    def refresh(self, view: MaterializedView, mode: RefreshMode) -> RefreshResult
    def estimate_storage(self, graph: PipelineGraph, node_id: str) -> StorageEstimate

def materialize_node(graph: PipelineGraph, node_id: str) -> MaterializedView
def refresh_view(view: MaterializedView, mode: RefreshMode) -> RefreshResult
```

---

<a id="quality"></a>
## 7. `arc.quality` — Data Quality & Drift Detection

**Location:** `arc/quality/`

Runtime data quality monitoring, statistical profiling, constraint checking, and distribution drift detection.

---

### `arc.quality.monitor` — Quality Monitor

#### `QualityMonitor`

```python
class QualityMonitor:
    def check(self, data: Any, constraints: List[QualityConstraint]) -> List[CheckResult]
    def profile(self, data: Any) -> TableProfile
    def score(self, data: Any, constraints: List[QualityConstraint]) -> float
```

**Example:**

```python
from arc.quality.monitor import QualityMonitor
from arc.types.base import QualityConstraint

monitor = QualityMonitor()
constraints = [
    QualityConstraint("nn_email", "email", "not_null"),
    QualityConstraint("uniq_id", "id", "unique"),
]

results = monitor.check(df, constraints)
for r in results:
    print(f"{r.constraint_name}: {'PASS' if r.passed else 'FAIL'} ({r.details})")

score = monitor.score(df, constraints)   # 0.0–1.0
```

---

### `arc.quality.profiler` — Data Profiling

#### `DataProfiler`

```python
class DataProfiler:
    def profile(self, data: Any) -> TableProfile
    def profile_column(self, data: Any, column: str) -> ColumnProfile
```

#### `TableProfile` / `ColumnProfile`

```python
@dataclass
class ColumnProfile:
    column: str
    dtype: str
    null_count: int
    null_fraction: float
    distinct_count: int
    min_value: Any
    max_value: Any
    mean: Optional[float]
    stddev: Optional[float]

@dataclass
class TableProfile:
    row_count: int
    column_count: int
    columns: Dict[str, ColumnProfile]
```

---

### `arc.quality.constraints` — Constraint Engine

#### `ConstraintEngine`

```python
class ConstraintEngine:
    def evaluate(self, data: Any, constraint: QualityConstraint) -> ConstraintResult
    def evaluate_all(self, data: Any, constraints: List[QualityConstraint]) -> List[ConstraintResult]
    def suggest(self, data: Any) -> List[ConstraintSuggestion]
```

---

### `arc.quality.drift` — Drift Detection

#### `DriftDetector`

```python
class DriftDetector:
    def detect(
        self,
        current_data: Any,
        baseline_data: Any,
        columns: List[str] | None = None,
    ) -> DriftResult

    def detect_schema_drift(
        self,
        current_schema: Schema,
        baseline_schema: Schema,
    ) -> SchemaDrift

    def track(self, data: Any, timestamp: datetime) -> DriftTimeSeries
```

#### Enums

| Enum | Values |
|------|--------|
| `DriftType` | `NONE`, `COVARIATE`, `CONCEPT`, `SCHEMA`, `VOLUME` |
| `DriftSeverity` | `NONE`, `LOW`, `MEDIUM`, `HIGH`, `CRITICAL` |

#### `DriftResult`

```python
@dataclass
class DriftResult:
    overall_drift: DriftType
    overall_severity: DriftSeverity
    column_drifts: Dict[str, ColumnDrift]
    timestamp: datetime
```

**Functions:**

```python
def detect_drift(
    current_data: Any,
    baseline_data: Any,
    columns: List[str] | None = None,
) -> DriftResult

def detect_schema_drift(
    current_schema: Schema,
    baseline_schema: Schema,
) -> SchemaDrift

def is_data_stale(data: Any, timestamp: datetime, threshold_hours: float) -> bool
```

**Example:**

```python
from arc.quality.drift import DriftDetector

detector = DriftDetector()
result = detector.detect(current_df, baseline_df, columns=["age", "income"])

if result.overall_severity >= DriftSeverity.HIGH:
    for col, drift in result.column_drifts.items():
        print(f"  {col}: {drift.drift_type.name} (p={drift.p_value:.4f})")
```

---

<a id="python-etl"></a>
## 8. `arc.python_etl` — Python ETL Idiom Analysis

**Location:** `arc/python_etl/`

Static analysis of Python ETL scripts — pandas, PySpark, and dbt — to extract pipeline graphs from imperative code.

---

### `arc.python_etl.analyzer` — Generic Python ETL Analyzer

```python
class PythonETLAnalyzer:
    def analyze(self, source: str, filename: str = "<string>") -> ETLAnalysisResult
    def analyze_file(self, filepath: str) -> ETLAnalysisResult
```

---

### `arc.python_etl.pandas_analyzer` — Pandas Analysis

```python
class PandasAnalyzer:
    def analyze(self, source: str) -> DataflowGraph
    def extract_lineage(self, source: str) -> PandasLineage
```

**Example:**

```python
from arc.python_etl.pandas_analyzer import PandasAnalyzer

analyzer = PandasAnalyzer()
graph = analyzer.analyze("""
import pandas as pd
df = pd.read_csv("users.csv")
df = df.dropna(subset=["email"])
df["name"] = df["first"] + " " + df["last"]
df.to_parquet("clean_users.parquet")
""")
```

---

### `arc.python_etl.pyspark_analyzer` — PySpark Analysis

```python
class PySparkAnalyzer:
    def analyze(self, source: str) -> DataflowGraph
    def extract_lineage(self, source: str) -> SparkLineage
```

---

### `arc.python_etl.dbt_analyzer` — dbt Project Analysis

```python
class DBTAnalyzer:
    def analyze(self, project_dir: str) -> DBTProject
    def build_lineage(self, project_dir: str) -> PipelineGraph

def analyze_dbt_project(project_dir: str) -> DBTProject
def build_dbt_lineage(project_dir: str) -> PipelineGraph
def extract_dbt_tests(project_dir: str) -> List[Dict[str, Any]]
```

**Example:**

```python
from arc.python_etl.dbt_analyzer import build_dbt_lineage

graph = build_dbt_lineage("./my_dbt_project")
print(graph.node_count())   # number of dbt models as pipeline nodes
```

---

<a id="io"></a>
## 9. `arc.io` — Serialization

**Location:** `arc/io/`

JSON and YAML serialization for pipeline graphs, delta objects, and repair plans, with JSON Schema validation and spec versioning.

---

### `arc.io.json_format` — JSON Serialization

```python
class ARCJSONEncoder(json.JSONEncoder):
    """Custom encoder that handles ARC types (Schema, SchemaDelta, etc.)."""

class DeltaSerializer:
    def serialize(self, delta: CompoundPerturbation) -> dict
    def deserialize(self, data: dict) -> CompoundPerturbation

class RepairPlanSerializer:
    def serialize(self, plan: RepairPlan) -> dict
    def deserialize(self, data: dict) -> RepairPlan

class BatchSerializer:
    def serialize_batch(self, items: List[Any]) -> str
    def deserialize_batch(self, data: str) -> List[Any]

def dumps(obj: Any, indent: int = 2) -> str
def loads(s: str) -> Any
```

**Example:**

```python
from arc.io.json_format import dumps, loads

json_str = dumps(graph.to_dict())
restored = PipelineGraph.from_dict(loads(json_str))
```

---

### `arc.io.yaml_format` — YAML Serialization

```python
class YAMLPipelineSpec:
    def load(self, path: str) -> PipelineGraph
    def dump(self, graph: PipelineGraph, path: str) -> None

class YAMLMerger:
    def merge(self, base: dict, overlay: dict) -> dict

def from_template(template_name: str, name: str = "") -> PipelineGraph
def list_templates() -> list[str]
def get_template_yaml(template_name: str, name: str = "") -> str
```

YAML supports `!include`, `!env`, and `!env_default` directives for composable pipeline definitions.

**Example:**

```yaml
# pipeline.yaml
name: etl_pipeline
nodes:
  - id: raw_data
    operator: SOURCE
    schema: !include schemas/raw.yaml
  - id: cleaned
    operator: TRANSFORM
    query: "SELECT * FROM raw_data WHERE valid = true"
    upstream: [raw_data]
  - id: output
    operator: SINK
    upstream: [cleaned]
```

```python
from arc.io.yaml_format import YAMLPipelineSpec

spec = YAMLPipelineSpec()
graph = spec.load("pipeline.yaml")
```

---

### `arc.io.schema` — JSON Schema Validation

```python
def get_pipeline_schema(version: str = CURRENT_SPEC_VERSION) -> dict[str, Any]
def get_delta_schema(version: str = CURRENT_SPEC_VERSION) -> dict[str, Any]
def get_repair_plan_schema(version: str = CURRENT_SPEC_VERSION) -> dict[str, Any]

class SpecMigrator:
    """Migrates specs between schema versions."""
    def migrate(self, data: dict, from_version: str, to_version: str) -> dict
```

---

<a id="cli"></a>
## 10. `arc.cli` — Command-Line Interface

**Location:** `arc/cli/main.py`

Click-based CLI exposing all major ARC operations.

### Installation

```bash
pip install arc
arc --help
```

---

### `arc analyze <pipeline>`

Analyse a pipeline for schema compatibility, Fragment-F membership, and quality constraints.

```bash
arc analyze pipeline.yaml
arc analyze pipeline.json --node cleaned --verbose
```

---

### `arc repair <pipeline> --perturbation <delta>`

Compute a cost-optimal repair plan.

```bash
arc repair pipeline.yaml --perturbation delta.json
arc repair pipeline.yaml --perturbation delta.json --output plan.json --dry-run
```

---

### `arc execute <repair-plan>`

Execute a repair plan against the pipeline data.

```bash
arc execute plan.json
arc execute plan.json --dry-run --checkpoint
```

---

### `arc validate <pipeline>`

Validate a pipeline specification.

```bash
arc validate pipeline.yaml
arc validate pipeline.yaml --strict
```

---

### `arc visualize <pipeline>`

Export the pipeline graph in various formats.

```bash
arc visualize pipeline.yaml                          # ASCII (default)
arc visualize pipeline.yaml --format dot -o pipe.png
arc visualize pipeline.yaml --format mermaid
arc visualize pipeline.yaml --highlight source clean
```

---

### `arc monitor <pipeline>`

Run quality monitoring on pipeline data.

```bash
arc monitor pipeline.yaml
arc monitor pipeline.yaml --node cleaned
```

---

### Additional Commands

| Command | Description |
|---------|-------------|
| `arc fragment <pipeline>` | Check Fragment-F membership for all nodes |
| `arc lineage <pipeline> <node> <column>` | Trace column-level lineage |
| `arc scope <pipeline> --node <id>` | Show repair scope for affected nodes |
| `arc diff <pipeline_a> <pipeline_b>` | Structural diff between two pipelines |
| `arc bottleneck <pipeline>` | Identify high-cost bottleneck nodes |
| `arc waves <pipeline>` | Show parallelisable execution waves |
| `arc metrics <pipeline>` | Compute graph-wide metrics |
| `arc quality <pipeline>` | Run quality profiling |
| `arc cost <pipeline>` | Show per-node cost estimates |
| `arc annihilation <pipeline> <source>` | Detect delta annihilation paths |
| `arc compare <pipeline> <node>` | Compare repair vs. recompute cost |
| `arc redundancy <pipeline>` | Detect redundant computations |
| `arc dependencies <pipeline> <node>` | Show upstream/downstream dependencies |
| `arc template <name>` | Generate a pipeline from a built-in template |
| `arc convert <input> <output>` | Convert between JSON and YAML formats |
| `arc info` | Show ARC version and configuration |

---

## Licence

See [LICENSE](implementation/LICENSE) in the implementation directory.

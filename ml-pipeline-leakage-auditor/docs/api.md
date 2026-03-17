# API Reference

## `taintflow.lattice.TaintElement`

Represents a single element in the four-level partition-taint lattice.

```python
class TaintElement(enum.Enum):
    BOTTOM = "⊥"       # Untainted
    TRAIN  = "Train"    # Influenced only by training data
    TEST   = "Test"     # Influenced only by test data
    TOP    = "⊤"        # Influenced by both partitions (leakage)
```

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `join` | `(self, other: TaintElement) -> TaintElement` | Least upper bound in the lattice. `Train.join(Test) == TOP`. |
| `meet` | `(self, other: TaintElement) -> TaintElement` | Greatest lower bound in the lattice. `Train.meet(Test) == BOTTOM`. |
| `leq`  | `(self, other: TaintElement) -> bool` | Lattice ordering. `BOTTOM.leq(x)` is always `True`. |
| `is_leaked` | `(self) -> bool` | Returns `True` iff the element is `TOP`. |

---

## `taintflow.lattice.PartitionTaintLattice`

The lattice structure itself, providing utility operations over `TaintElement`.

```python
class PartitionTaintLattice:
    @staticmethod
    def join(a: TaintElement, b: TaintElement) -> TaintElement: ...

    @staticmethod
    def meet(a: TaintElement, b: TaintElement) -> TaintElement: ...

    @staticmethod
    def join_all(elements: Iterable[TaintElement]) -> TaintElement: ...

    @staticmethod
    def is_fixed_point(old: TaintElement, new: TaintElement) -> bool: ...
```

---

## `taintflow.taint_map.ColumnTaintMap`

Maps column names to their current `TaintElement`.

```python
class ColumnTaintMap:
    def __init__(self, columns: dict[str, TaintElement] | None = None) -> None: ...

    def get(self, column: str) -> TaintElement: ...
    def set(self, column: str, taint: TaintElement) -> None: ...
    def join_with(self, other: ColumnTaintMap) -> ColumnTaintMap: ...
    def leaked_columns(self) -> list[str]: ...
    def summary(self) -> dict[str, str]: ...
```

### Example

```python
from taintflow.lattice import TaintElement
from taintflow.taint_map import ColumnTaintMap

m = ColumnTaintMap({"age": TaintElement.TRAIN, "income": TaintElement.TEST})
print(m.leaked_columns())  # []

m.set("age", TaintElement.TOP)
print(m.leaked_columns())  # ["age"]
```

---

## `taintflow.taint_map.DataFrameAbstractState`

Tracks the abstract taint state of an entire dataframe as it flows through
pipeline operations.

```python
class DataFrameAbstractState:
    def __init__(
        self,
        column_taints: ColumnTaintMap,
        row_partition: TaintElement,
        shape: tuple[int | None, int | None] = (None, None),
    ) -> None: ...

    def apply_fit(self, transformer_name: str) -> TaintElement: ...
    def apply_transform(self, param_taint: TaintElement) -> DataFrameAbstractState: ...
    def apply_split(
        self, test_size: float
    ) -> tuple[DataFrameAbstractState, DataFrameAbstractState]: ...

    @property
    def is_leaked(self) -> bool: ...
```

### Key Methods

- **`apply_fit(transformer_name)`** — Records which partition was seen during
  fitting. Returns the taint of the fitted parameters.
- **`apply_transform(param_taint)`** — Propagates taint through a transform
  step. Each output column's taint is the join of its input taint and the
  parameter taint.
- **`apply_split(test_size)`** — Splits the state into train and test abstract
  states, assigning `TRAIN` and `TEST` row partitions respectively.

---

## `taintflow.capacity.ChannelCapacityBound`

Abstract base class for channel capacity models.

```python
class ChannelCapacityBound(abc.ABC):
    @abc.abstractmethod
    def compute_bound(
        self,
        taint_map: ColumnTaintMap,
        n_train: int,
        n_test: int,
    ) -> float:
        """Return an upper bound on the channel capacity in bits."""
        ...

    @abc.abstractmethod
    def description(self) -> str:
        """Human-readable description of the bound."""
        ...
```

### Built-in Implementations

| Class | Module | Description |
|-------|--------|-------------|
| `GaussianChannelBound` | `taintflow.capacity.gaussian` | Bound for mean/variance estimators: `C ≤ (d/2) · log₂(1 + n_test/n_train)` |
| `CountingBound` | `taintflow.capacity.counting` | Bound for categorical encoders: `C ≤ log₂(k)` |
| `MutualInformationBound` | `taintflow.capacity.mi` | General bound via KSG mutual information estimation |

---

## `taintflow.config.TaintFlowConfig`

Configuration for the TaintFlow auditor.

```python
@dataclasses.dataclass(frozen=True)
class TaintFlowConfig:
    severity_threshold: float = 0.1
    """Minimum channel capacity (bits) to report as a warning."""

    max_dag_depth: int = 50
    """Maximum depth of the PI-DAG before aborting analysis."""

    output_format: Literal["json", "text"] = "text"
    """Output format for CLI reports."""

    strict_mode: bool = False
    """If True, treat all leakage (even low-capacity) as errors."""

    ignore_patterns: list[str] = dataclasses.field(default_factory=list)
    """List of transformer names to exclude from analysis."""
```

### Example

```python
from taintflow.config import TaintFlowConfig

config = TaintFlowConfig(
    severity_threshold=0.5,
    output_format="json",
    strict_mode=True,
)
```

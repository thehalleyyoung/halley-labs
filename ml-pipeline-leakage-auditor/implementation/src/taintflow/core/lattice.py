"""
taintflow.core.lattice – Partition-Taint Lattice implementation.

The abstract domain is  T = (P({tr, te, ext}) × [0, B_max], ⊑)  where
⊑ is the product ordering:  (S₁, b₁) ⊑ (S₂, b₂)  iff  S₁ ⊆ S₂ ∧ b₁ ≤ b₂.

Key classes:

* :class:`TaintElement`           – a single lattice element (S, b).
* :class:`PartitionTaintLattice`  – the lattice operations (⊔, ⊓, ⊑, ⊥, ⊤).
* :class:`ColumnTaintMap`         – maps column names → TaintElement.
* :class:`DataFrameAbstractState` – full abstract state for a DataFrame.
* :class:`WidenOperator`          – widening to ensure termination.
* :class:`NarrowOperator`         – narrowing for precision recovery.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from taintflow.core.types import (
    ColumnSchema,
    Origin,
    ProvenanceInfo,
    Severity,
    ShapeMetadata,
)

# ===================================================================
#  Constants
# ===================================================================

_ALL_ORIGINS: frozenset[Origin] = frozenset(Origin)
_EMPTY_ORIGINS: frozenset[Origin] = frozenset()


# ===================================================================
#  TaintElement
# ===================================================================

@dataclass(frozen=True)
class TaintElement:
    """An element (S, b) of the partition-taint lattice T.

    *S* is a subset of {TRAIN, TEST, EXTERNAL} (represented as a frozenset
    of :class:`Origin`).  *b* is a non-negative float bounded by
    :attr:`B_MAX`.

    The partial order is the *product order*:

        (S₁, b₁) ⊑ (S₂, b₂)  ⟺  S₁ ⊆ S₂  ∧  b₁ ≤ b₂
    """

    origins: FrozenSet[Origin] = field(default_factory=frozenset)
    bit_bound: float = 0.0

    B_MAX: float = field(default=64.0, repr=False, compare=False)

    # -- validation ----------------------------------------------------------

    def __post_init__(self) -> None:
        if self.bit_bound < 0.0:
            object.__setattr__(self, "bit_bound", 0.0)
        if self.bit_bound > self.B_MAX:
            object.__setattr__(self, "bit_bound", self.B_MAX)
        if math.isnan(self.bit_bound):
            object.__setattr__(self, "bit_bound", self.B_MAX)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.bit_bound < 0.0:
            errors.append(f"bit_bound must be >= 0, got {self.bit_bound}")
        if self.bit_bound > self.B_MAX:
            errors.append(f"bit_bound must be <= B_MAX={self.B_MAX}, got {self.bit_bound}")
        for o in self.origins:
            if not isinstance(o, Origin):
                errors.append(f"Invalid origin: {o!r}")
        return errors

    # -- lattice operations --------------------------------------------------

    def join(self, other: "TaintElement") -> "TaintElement":
        """Least upper bound: (S₁∪S₂, max(b₁, b₂))."""
        b_max = max(self.B_MAX, other.B_MAX)
        return TaintElement(
            origins=self.origins | other.origins,
            bit_bound=min(max(self.bit_bound, other.bit_bound), b_max),
            B_MAX=b_max,
        )

    def meet(self, other: "TaintElement") -> "TaintElement":
        """Greatest lower bound: (S₁∩S₂, min(b₁, b₂))."""
        b_max = max(self.B_MAX, other.B_MAX)
        return TaintElement(
            origins=self.origins & other.origins,
            bit_bound=max(min(self.bit_bound, other.bit_bound), 0.0),
            B_MAX=b_max,
        )

    def leq(self, other: "TaintElement") -> bool:
        """Partial order: self ⊑ other."""
        return self.origins <= other.origins and self.bit_bound <= other.bit_bound + 1e-12

    def partial_order(self, other: "TaintElement") -> bool:
        """Alias for :meth:`leq`."""
        return self.leq(other)

    @property
    def is_bottom(self) -> bool:
        return len(self.origins) == 0 and self.bit_bound == 0.0

    @property
    def is_top(self) -> bool:
        return self.origins == _ALL_ORIGINS and math.isclose(self.bit_bound, self.B_MAX, abs_tol=1e-12)

    @property
    def is_test_tainted(self) -> bool:
        return Origin.TEST in self.origins and self.bit_bound > 0.0

    @property
    def severity(self) -> Severity:
        return Severity.from_bits(self.bit_bound)

    # -- arithmetic helpers --------------------------------------------------

    def add_bits(self, delta: float) -> "TaintElement":
        return TaintElement(
            origins=self.origins,
            bit_bound=min(self.bit_bound + delta, self.B_MAX),
            B_MAX=self.B_MAX,
        )

    def scale_bits(self, factor: float) -> "TaintElement":
        return TaintElement(
            origins=self.origins,
            bit_bound=min(max(self.bit_bound * factor, 0.0), self.B_MAX),
            B_MAX=self.B_MAX,
        )

    def with_origins(self, extra: FrozenSet[Origin]) -> "TaintElement":
        return TaintElement(
            origins=self.origins | extra,
            bit_bound=self.bit_bound,
            B_MAX=self.B_MAX,
        )

    def restrict_origins(self, keep: FrozenSet[Origin]) -> "TaintElement":
        return TaintElement(
            origins=self.origins & keep,
            bit_bound=self.bit_bound if (self.origins & keep) else 0.0,
            B_MAX=self.B_MAX,
        )

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "origins": sorted(o.value for o in self.origins),
            "bit_bound": self.bit_bound,
            "b_max": self.B_MAX,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaintElement":
        origins = frozenset(Origin(s) for s in data.get("origins", []))
        return cls(
            origins=origins,
            bit_bound=float(data.get("bit_bound", 0.0)),
            B_MAX=float(data.get("b_max", 64.0)),
        )

    # -- comparison / hashing ------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TaintElement):
            return NotImplemented
        return self.origins == other.origins and math.isclose(
            self.bit_bound, other.bit_bound, abs_tol=1e-12
        )

    def __hash__(self) -> int:
        return hash((self.origins, round(self.bit_bound, 10)))

    def __repr__(self) -> str:
        origins_str = ",".join(o.name for o in sorted(self.origins, key=lambda o: o.value))
        return f"⟨{{{origins_str}}}, {self.bit_bound:.4f}⟩"

    def __str__(self) -> str:
        return self.__repr__()

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, TaintElement):
            return NotImplemented
        return self.leq(other) and self != other

    def __le__(self, other: object) -> bool:
        if not isinstance(other, TaintElement):
            return NotImplemented
        return self.leq(other)

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, TaintElement):
            return NotImplemented
        return other.leq(self) and self != other

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, TaintElement):
            return NotImplemented
        return other.leq(self)


# ===================================================================
#  PartitionTaintLattice
# ===================================================================

class PartitionTaintLattice:
    """The partition-taint lattice  T = (P({tr, te, ext}) × [0, B_max], ⊑).

    This class provides factory methods for bottom / top elements and the
    binary lattice operations (join, meet, leq).

    The *height* of the lattice is  |P({tr,te,ext})| × ⌈B_max⌉ + 1.
    With 3 origins and B_max = 64 the number of "steps" is
    2³ × 65 = 8 × 65 = 520, but the effective *chain height* along any
    single ascending chain is  3 + 65 = 68  (3 for subset inclusion, 65
    for the integer part of bit_bound 0..64), giving **69** with the
    bottom element.  (The paper uses 69 for the default configuration.)
    """

    def __init__(self, b_max: float = 64.0) -> None:
        if b_max <= 0:
            raise ValueError(f"b_max must be > 0, got {b_max}")
        self._b_max = b_max

    @property
    def b_max(self) -> float:
        return self._b_max

    def bottom(self) -> TaintElement:
        return TaintElement(origins=_EMPTY_ORIGINS, bit_bound=0.0, B_MAX=self._b_max)

    def top(self) -> TaintElement:
        return TaintElement(origins=_ALL_ORIGINS, bit_bound=self._b_max, B_MAX=self._b_max)

    def element(self, origins: FrozenSet[Origin], bit_bound: float) -> TaintElement:
        return TaintElement(origins=origins, bit_bound=bit_bound, B_MAX=self._b_max)

    def join(self, a: TaintElement, b: TaintElement) -> TaintElement:
        return a.join(b)

    def meet(self, a: TaintElement, b: TaintElement) -> TaintElement:
        return a.meet(b)

    def leq(self, a: TaintElement, b: TaintElement) -> bool:
        return a.leq(b)

    def height(self) -> int:
        """Longest ascending chain length (including bottom).

        For 3 origins and B_max=64: 3 (subset steps) + 65 (bit steps 0..64) + 1 (bottom) = 69.
        """
        n_origins = len(Origin)
        bit_steps = int(math.ceil(self._b_max)) + 1
        return n_origins + bit_steps + 1

    def is_fixpoint(self, old: TaintElement, new: TaintElement, *, epsilon: float = 1e-10) -> bool:
        if old.origins != new.origins:
            return False
        return abs(old.bit_bound - new.bit_bound) < epsilon

    def join_all(self, elements: Iterable[TaintElement]) -> TaintElement:
        result = self.bottom()
        for e in elements:
            result = result.join(e)
        return result

    def meet_all(self, elements: Iterable[TaintElement]) -> TaintElement:
        result = self.top()
        for e in elements:
            result = result.meet(e)
        return result

    def from_origin(self, origin: Origin, *, bit_bound: float = 0.0) -> TaintElement:
        return self.element(frozenset({origin}), bit_bound)

    def train_only(self, bits: float = 0.0) -> TaintElement:
        return self.element(frozenset({Origin.TRAIN}), bits)

    def test_only(self, bits: float = 0.0) -> TaintElement:
        return self.element(frozenset({Origin.TEST}), bits)

    def mixed(self, bits: float = 0.0) -> TaintElement:
        return self.element(frozenset({Origin.TRAIN, Origin.TEST}), bits)

    def __repr__(self) -> str:
        return f"PartitionTaintLattice(B_max={self._b_max})"


# ===================================================================
#  ColumnTaintMap
# ===================================================================

class ColumnTaintMap:
    """Maps column names → :class:`TaintElement`.

    This is the per-column abstract state for a DataFrame.  It supports
    lattice-wise join / meet / leq lifted pointwise to maps.
    """

    __slots__ = ("_map", "_b_max")

    def __init__(
        self,
        mapping: Mapping[str, TaintElement] | None = None,
        *,
        b_max: float = 64.0,
    ) -> None:
        self._b_max = b_max
        self._map: dict[str, TaintElement] = dict(mapping) if mapping else {}

    # -- accessors -----------------------------------------------------------

    def __getitem__(self, col: str) -> TaintElement:
        return self._map[col]

    def get(self, col: str, default: TaintElement | None = None) -> TaintElement:
        if default is None:
            default = TaintElement(B_MAX=self._b_max)
        return self._map.get(col, default)

    def __setitem__(self, col: str, elem: TaintElement) -> None:
        self._map[col] = elem

    def __delitem__(self, col: str) -> None:
        del self._map[col]

    def __contains__(self, col: str) -> bool:
        return col in self._map

    def __len__(self) -> int:
        return len(self._map)

    def __iter__(self) -> Iterator[str]:
        return iter(self._map)

    def columns(self) -> list[str]:
        return list(self._map.keys())

    def items(self) -> Iterable[tuple[str, TaintElement]]:
        return self._map.items()

    def values(self) -> Iterable[TaintElement]:
        return self._map.values()

    def get_columns(self) -> frozenset[str]:
        return frozenset(self._map.keys())

    def copy(self) -> "ColumnTaintMap":
        return ColumnTaintMap(dict(self._map), b_max=self._b_max)

    # -- lattice operations (pointwise) --------------------------------------

    def join_maps(self, other: "ColumnTaintMap") -> "ColumnTaintMap":
        """Pointwise join: for each column in either map, take the join."""
        all_cols = set(self._map) | set(other._map)
        b_max = max(self._b_max, other._b_max)
        bottom = TaintElement(B_MAX=b_max)
        result: dict[str, TaintElement] = {}
        for col in all_cols:
            a = self._map.get(col, bottom)
            b = other._map.get(col, bottom)
            result[col] = a.join(b)
        return ColumnTaintMap(result, b_max=b_max)

    def meet_maps(self, other: "ColumnTaintMap") -> "ColumnTaintMap":
        """Pointwise meet: only keep columns in both maps."""
        common_cols = set(self._map) & set(other._map)
        b_max = max(self._b_max, other._b_max)
        result: dict[str, TaintElement] = {}
        for col in common_cols:
            result[col] = self._map[col].meet(other._map[col])
        return ColumnTaintMap(result, b_max=b_max)

    def leq_maps(self, other: "ColumnTaintMap") -> bool:
        """Pointwise ⊑: self ⊑ other iff ∀ col, self[col] ⊑ other[col]."""
        for col, elem in self._map.items():
            other_elem = other._map.get(col)
            if other_elem is None:
                if not elem.is_bottom:
                    return False
            elif not elem.leq(other_elem):
                return False
        return True

    # -- projection / extension ----------------------------------------------

    def project(self, columns: Iterable[str]) -> "ColumnTaintMap":
        """Keep only the specified columns."""
        cols = set(columns)
        result = {c: e for c, e in self._map.items() if c in cols}
        return ColumnTaintMap(result, b_max=self._b_max)

    def extend(self, columns: Mapping[str, TaintElement]) -> "ColumnTaintMap":
        """Add new columns (or overwrite existing)."""
        new_map = dict(self._map)
        new_map.update(columns)
        return ColumnTaintMap(new_map, b_max=self._b_max)

    def rename(self, mapping: Mapping[str, str]) -> "ColumnTaintMap":
        """Rename columns according to the mapping."""
        new_map: dict[str, TaintElement] = {}
        for col, elem in self._map.items():
            new_name = mapping.get(col, col)
            new_map[new_name] = elem
        return ColumnTaintMap(new_map, b_max=self._b_max)

    def drop(self, columns: Iterable[str]) -> "ColumnTaintMap":
        """Remove specified columns."""
        drop_set = set(columns)
        result = {c: e for c, e in self._map.items() if c not in drop_set}
        return ColumnTaintMap(result, b_max=self._b_max)

    # -- bulk operations -----------------------------------------------------

    def max_element(self) -> TaintElement:
        """Return the element with the highest bit_bound."""
        if not self._map:
            return TaintElement(B_MAX=self._b_max)
        return max(self._map.values(), key=lambda e: e.bit_bound)

    def min_element(self) -> TaintElement:
        if not self._map:
            return TaintElement(B_MAX=self._b_max)
        return min(self._map.values(), key=lambda e: e.bit_bound)

    def total_bits(self) -> float:
        return sum(e.bit_bound for e in self._map.values())

    def mean_bits(self) -> float:
        if not self._map:
            return 0.0
        return self.total_bits() / len(self._map)

    def tainted_columns(self, *, threshold: float = 0.0) -> list[str]:
        return [c for c, e in self._map.items() if e.bit_bound > threshold]

    def test_tainted_columns(self) -> list[str]:
        return [c for c, e in self._map.items() if e.is_test_tainted]

    def all_origins(self) -> frozenset[Origin]:
        result: set[Origin] = set()
        for e in self._map.values():
            result |= e.origins
        return frozenset(result)

    def apply_uniform(self, fn: Any) -> "ColumnTaintMap":
        """Apply a function to every TaintElement in the map."""
        return ColumnTaintMap(
            {c: fn(e) for c, e in self._map.items()},
            b_max=self._b_max,
        )

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": {c: e.to_dict() for c, e in self._map.items()},
            "b_max": self._b_max,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ColumnTaintMap":
        b_max = float(data.get("b_max", 64.0))
        columns = {
            name: TaintElement.from_dict(edata)
            for name, edata in data.get("columns", {}).items()
        }
        return cls(columns, b_max=b_max)

    # -- comparison / hashing ------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ColumnTaintMap):
            return NotImplemented
        if set(self._map) != set(other._map):
            return False
        return all(self._map[c] == other._map[c] for c in self._map)

    def __hash__(self) -> int:
        return hash(frozenset((c, e) for c, e in self._map.items()))

    def __repr__(self) -> str:
        inner = ", ".join(f"{c}: {e}" for c, e in sorted(self._map.items()))
        return f"ColumnTaintMap({{{inner}}})"

    def __str__(self) -> str:
        lines = [f"  {c}: {e}" for c, e in sorted(self._map.items())]
        return "ColumnTaintMap{\n" + "\n".join(lines) + "\n}"


# ===================================================================
#  DataFrameAbstractState
# ===================================================================

@dataclass
class DataFrameAbstractState:
    """Full abstract state for a DataFrame in the analysis.

    Combines a :class:`ColumnTaintMap` with row-level provenance
    (:class:`ProvenanceInfo`) and shape metadata (:class:`ShapeMetadata`).
    """

    column_map: ColumnTaintMap
    row_provenance: ProvenanceInfo
    shape: ShapeMetadata
    label: str = ""
    stage_id: str = ""

    # -- factory methods -----------------------------------------------------

    @classmethod
    def bottom(cls, *, b_max: float = 64.0) -> "DataFrameAbstractState":
        return cls(
            column_map=ColumnTaintMap(b_max=b_max),
            row_provenance=ProvenanceInfo(test_fraction=0.0, origin_set=frozenset()),
            shape=ShapeMetadata(n_rows=0, n_cols=0),
        )

    @classmethod
    def from_train(
        cls,
        columns: Sequence[str],
        n_rows: int,
        *,
        b_max: float = 64.0,
    ) -> "DataFrameAbstractState":
        lattice = PartitionTaintLattice(b_max)
        col_map = ColumnTaintMap(
            {c: lattice.train_only(0.0) for c in columns},
            b_max=b_max,
        )
        return cls(
            column_map=col_map,
            row_provenance=ProvenanceInfo(
                test_fraction=0.0,
                origin_set=frozenset({Origin.TRAIN}),
            ),
            shape=ShapeMetadata(n_rows=n_rows, n_cols=len(columns), n_test_rows=0),
        )

    @classmethod
    def from_test(
        cls,
        columns: Sequence[str],
        n_rows: int,
        *,
        b_max: float = 64.0,
    ) -> "DataFrameAbstractState":
        lattice = PartitionTaintLattice(b_max)
        col_map = ColumnTaintMap(
            {c: lattice.test_only(0.0) for c in columns},
            b_max=b_max,
        )
        return cls(
            column_map=col_map,
            row_provenance=ProvenanceInfo(
                test_fraction=1.0,
                origin_set=frozenset({Origin.TEST}),
            ),
            shape=ShapeMetadata(n_rows=n_rows, n_cols=len(columns), n_test_rows=n_rows),
        )

    @classmethod
    def from_mixed(
        cls,
        columns: Sequence[str],
        n_rows: int,
        n_test_rows: int,
        *,
        bit_bound: float = 0.0,
        b_max: float = 64.0,
    ) -> "DataFrameAbstractState":
        lattice = PartitionTaintLattice(b_max)
        elem = lattice.mixed(bit_bound)
        col_map = ColumnTaintMap(
            {c: elem for c in columns},
            b_max=b_max,
        )
        rho = n_test_rows / n_rows if n_rows > 0 else 0.0
        return cls(
            column_map=col_map,
            row_provenance=ProvenanceInfo(
                test_fraction=rho,
                origin_set=frozenset({Origin.TRAIN, Origin.TEST}),
            ),
            shape=ShapeMetadata(n_rows=n_rows, n_cols=len(columns), n_test_rows=n_test_rows),
        )

    # -- lattice operations --------------------------------------------------

    def join(self, other: "DataFrameAbstractState") -> "DataFrameAbstractState":
        new_cm = self.column_map.join_maps(other.column_map)
        new_prov = self.row_provenance.merge(other.row_provenance)
        new_shape = ShapeMetadata(
            n_rows=max(self.shape.n_rows, other.shape.n_rows),
            n_cols=len(new_cm),
            n_test_rows=max(self.shape.n_test_rows, other.shape.n_test_rows),
        )
        return DataFrameAbstractState(
            column_map=new_cm,
            row_provenance=new_prov,
            shape=new_shape,
            label=self.label or other.label,
        )

    def meet(self, other: "DataFrameAbstractState") -> "DataFrameAbstractState":
        new_cm = self.column_map.meet_maps(other.column_map)
        merged_origins = self.row_provenance.origin_set & other.row_provenance.origin_set
        new_prov = ProvenanceInfo(
            test_fraction=min(self.row_provenance.test_fraction, other.row_provenance.test_fraction),
            origin_set=merged_origins if merged_origins else frozenset({Origin.TRAIN}),
        )
        new_shape = ShapeMetadata(
            n_rows=min(self.shape.n_rows, other.shape.n_rows),
            n_cols=len(new_cm),
            n_test_rows=min(self.shape.n_test_rows, other.shape.n_test_rows),
        )
        return DataFrameAbstractState(
            column_map=new_cm,
            row_provenance=new_prov,
            shape=new_shape,
        )

    def leq(self, other: "DataFrameAbstractState") -> bool:
        return self.column_map.leq_maps(other.column_map)

    # -- column operations ---------------------------------------------------

    def project(self, columns: Iterable[str]) -> "DataFrameAbstractState":
        cols = list(columns)
        return DataFrameAbstractState(
            column_map=self.column_map.project(cols),
            row_provenance=self.row_provenance,
            shape=ShapeMetadata(
                n_rows=self.shape.n_rows,
                n_cols=len(cols),
                n_test_rows=self.shape.n_test_rows,
            ),
            label=self.label,
            stage_id=self.stage_id,
        )

    def extend(self, new_columns: Mapping[str, TaintElement]) -> "DataFrameAbstractState":
        return DataFrameAbstractState(
            column_map=self.column_map.extend(new_columns),
            row_provenance=self.row_provenance,
            shape=ShapeMetadata(
                n_rows=self.shape.n_rows,
                n_cols=len(self.column_map) + len(new_columns),
                n_test_rows=self.shape.n_test_rows,
            ),
            label=self.label,
            stage_id=self.stage_id,
        )

    def drop_columns(self, columns: Iterable[str]) -> "DataFrameAbstractState":
        new_cm = self.column_map.drop(columns)
        return DataFrameAbstractState(
            column_map=new_cm,
            row_provenance=self.row_provenance,
            shape=ShapeMetadata(
                n_rows=self.shape.n_rows,
                n_cols=len(new_cm),
                n_test_rows=self.shape.n_test_rows,
            ),
            label=self.label,
            stage_id=self.stage_id,
        )

    def rename_columns(self, mapping: Mapping[str, str]) -> "DataFrameAbstractState":
        return DataFrameAbstractState(
            column_map=self.column_map.rename(mapping),
            row_provenance=self.row_provenance,
            shape=self.shape,
            label=self.label,
            stage_id=self.stage_id,
        )

    # -- taint propagation helpers -------------------------------------------

    def propagate_taint(self, source_cols: Sequence[str], target_col: str) -> "DataFrameAbstractState":
        """Propagate taint from source columns to a target column (join)."""
        lattice = PartitionTaintLattice(self.column_map._b_max)
        joined = lattice.bottom()
        for col in source_cols:
            if col in self.column_map:
                joined = joined.join(self.column_map[col])
        new_state = DataFrameAbstractState(
            column_map=self.column_map.extend({target_col: joined}),
            row_provenance=self.row_provenance,
            shape=self.shape,
            label=self.label,
            stage_id=self.stage_id,
        )
        return new_state

    def add_leakage(self, column: str, bits: float) -> "DataFrameAbstractState":
        """Add leakage bits to a specific column."""
        new_cm = self.column_map.copy()
        existing = new_cm.get(column)
        new_cm[column] = existing.add_bits(bits).with_origins(frozenset({Origin.TEST}))
        return DataFrameAbstractState(
            column_map=new_cm,
            row_provenance=self.row_provenance,
            shape=self.shape,
            label=self.label,
            stage_id=self.stage_id,
        )

    def taint_all_from_test(self, bits: float) -> "DataFrameAbstractState":
        """Add test-taint to every column."""
        new_cm = self.column_map.apply_uniform(
            lambda e: e.add_bits(bits).with_origins(frozenset({Origin.TEST}))
        )
        return DataFrameAbstractState(
            column_map=new_cm,
            row_provenance=ProvenanceInfo(
                test_fraction=self.row_provenance.test_fraction,
                origin_set=self.row_provenance.origin_set | {Origin.TEST},
            ),
            shape=self.shape,
            label=self.label,
            stage_id=self.stage_id,
        )

    # -- summary / query -----------------------------------------------------

    @property
    def max_bit_bound(self) -> float:
        if len(self.column_map) == 0:
            return 0.0
        return self.column_map.max_element().bit_bound

    @property
    def mean_bit_bound(self) -> float:
        return self.column_map.mean_bits()

    @property
    def is_clean(self) -> bool:
        return not any(e.is_test_tainted for e in self.column_map.values())

    @property
    def test_tainted_columns(self) -> list[str]:
        return self.column_map.test_tainted_columns()

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "column_map": self.column_map.to_dict(),
            "row_provenance": self.row_provenance.to_dict(),
            "shape": self.shape.to_dict(),
            "label": self.label,
            "stage_id": self.stage_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DataFrameAbstractState":
        return cls(
            column_map=ColumnTaintMap.from_dict(data["column_map"]),
            row_provenance=ProvenanceInfo.from_dict(data["row_provenance"]),
            shape=ShapeMetadata.from_dict(data["shape"]),
            label=str(data.get("label", "")),
            stage_id=str(data.get("stage_id", "")),
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        errors.extend(self.row_provenance.validate())
        errors.extend(self.shape.validate())
        if self.shape.n_cols != len(self.column_map):
            errors.append(
                f"shape.n_cols={self.shape.n_cols} != len(column_map)={len(self.column_map)}"
            )
        return errors

    # -- comparison ----------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DataFrameAbstractState):
            return NotImplemented
        return (
            self.column_map == other.column_map
            and self.row_provenance == other.row_provenance
        )

    def __hash__(self) -> int:
        return hash((self.column_map, self.row_provenance))

    def __repr__(self) -> str:
        cols = len(self.column_map)
        tainted = len(self.test_tainted_columns)
        return (
            f"DFState({cols} cols, {tainted} tainted, "
            f"ρ={self.row_provenance.test_fraction:.3f}, "
            f"max_b={self.max_bit_bound:.2f})"
        )


# ===================================================================
#  WidenOperator
# ===================================================================

class WidenOperator:
    """Widening operator ∇ for guaranteeing termination of fixpoint iteration.

    The widening is:
        (S_old, b_old) ∇ (S_new, b_new) =
            (S_old ∪ S_new,
             b_old                   if b_new ≤ b_old
             B_MAX                   otherwise)

    For ColumnTaintMaps, widening is applied pointwise.
    """

    def __init__(self, *, b_max: float = 64.0, delay: int = 3) -> None:
        self._b_max = b_max
        self._delay = delay
        self._iteration_counts: dict[str, int] = {}

    def widen_element(self, old: TaintElement, new: TaintElement) -> TaintElement:
        merged_origins = old.origins | new.origins
        if new.bit_bound <= old.bit_bound + 1e-12:
            widened_bits = old.bit_bound
        else:
            widened_bits = self._b_max
        return TaintElement(
            origins=merged_origins,
            bit_bound=widened_bits,
            B_MAX=self._b_max,
        )

    def widen_map(self, old: ColumnTaintMap, new: ColumnTaintMap) -> ColumnTaintMap:
        all_cols = set(old.columns()) | set(new.columns())
        bottom = TaintElement(B_MAX=self._b_max)
        result: dict[str, TaintElement] = {}
        for col in all_cols:
            o = old.get(col, bottom)
            n = new.get(col, bottom)
            result[col] = self.widen_element(o, n)
        return ColumnTaintMap(result, b_max=self._b_max)

    def widen_state(
        self,
        old: DataFrameAbstractState,
        new: DataFrameAbstractState,
    ) -> DataFrameAbstractState:
        widened_cm = self.widen_map(old.column_map, new.column_map)
        merged_prov = old.row_provenance.merge(new.row_provenance)
        new_shape = ShapeMetadata(
            n_rows=max(old.shape.n_rows, new.shape.n_rows),
            n_cols=len(widened_cm),
            n_test_rows=max(old.shape.n_test_rows, new.shape.n_test_rows),
        )
        return DataFrameAbstractState(
            column_map=widened_cm,
            row_provenance=merged_prov,
            shape=new_shape,
            label=old.label or new.label,
        )

    def widen(
        self,
        old: DataFrameAbstractState,
        new: DataFrameAbstractState,
        *,
        node_id: str = "",
    ) -> DataFrameAbstractState:
        """Apply widening with optional delay.

        For the first ``delay`` iterations at a given ``node_id``, return
        the plain join instead of the widened value.
        """
        key = node_id or id(old)
        count = self._iteration_counts.get(str(key), 0) + 1
        self._iteration_counts[str(key)] = count

        if count <= self._delay:
            return old.join(new)
        return self.widen_state(old, new)

    def reset(self, node_id: str = "") -> None:
        if node_id:
            self._iteration_counts.pop(node_id, None)
        else:
            self._iteration_counts.clear()

    def __repr__(self) -> str:
        return f"WidenOperator(B_max={self._b_max}, delay={self._delay})"


# ===================================================================
#  NarrowOperator
# ===================================================================

class NarrowOperator:
    """Narrowing operator △ for improving precision after widening.

    The narrowing is:
        (S_old, b_old) △ (S_new, b_new) =
            (S_new,
             b_new   if b_new < b_old
             b_old   otherwise)

    Applied pointwise on ColumnTaintMaps and DataFrameAbstractStates.
    """

    def __init__(self, *, max_iterations: int = 5, epsilon: float = 1e-10) -> None:
        self._max_iter = max_iterations
        self._epsilon = epsilon

    def narrow_element(self, old: TaintElement, new: TaintElement) -> TaintElement:
        narrowed_origins = new.origins
        if new.bit_bound < old.bit_bound - self._epsilon:
            narrowed_bits = new.bit_bound
        else:
            narrowed_bits = old.bit_bound
        return TaintElement(
            origins=narrowed_origins,
            bit_bound=narrowed_bits,
            B_MAX=old.B_MAX,
        )

    def narrow_map(self, old: ColumnTaintMap, new: ColumnTaintMap) -> ColumnTaintMap:
        all_cols = set(old.columns()) | set(new.columns())
        b_max = old._b_max
        bottom = TaintElement(B_MAX=b_max)
        result: dict[str, TaintElement] = {}
        for col in all_cols:
            o = old.get(col, bottom)
            n = new.get(col, bottom)
            result[col] = self.narrow_element(o, n)
        return ColumnTaintMap(result, b_max=b_max)

    def narrow_state(
        self,
        old: DataFrameAbstractState,
        new: DataFrameAbstractState,
    ) -> DataFrameAbstractState:
        narrowed_cm = self.narrow_map(old.column_map, new.column_map)
        new_prov = ProvenanceInfo(
            test_fraction=min(old.row_provenance.test_fraction, new.row_provenance.test_fraction),
            origin_set=new.row_provenance.origin_set,
        )
        new_shape = ShapeMetadata(
            n_rows=new.shape.n_rows,
            n_cols=len(narrowed_cm),
            n_test_rows=new.shape.n_test_rows,
        )
        return DataFrameAbstractState(
            column_map=narrowed_cm,
            row_provenance=new_prov,
            shape=new_shape,
            label=old.label or new.label,
        )

    def narrow(
        self,
        old: DataFrameAbstractState,
        new: DataFrameAbstractState,
    ) -> DataFrameAbstractState:
        """Apply a single narrowing step."""
        return self.narrow_state(old, new)

    def is_stable(
        self,
        old: DataFrameAbstractState,
        new: DataFrameAbstractState,
    ) -> bool:
        """Check if narrowing has reached a stable point."""
        if old.column_map.get_columns() != new.column_map.get_columns():
            return False
        for col in old.column_map:
            o = old.column_map[col]
            n = new.column_map[col]
            if o.origins != n.origins:
                return False
            if abs(o.bit_bound - n.bit_bound) > self._epsilon:
                return False
        return True

    @property
    def max_iterations(self) -> int:
        return self._max_iter

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def __repr__(self) -> str:
        return f"NarrowOperator(max_iter={self._max_iter}, ε={self._epsilon})"

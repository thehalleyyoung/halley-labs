"""
taintflow.instrument.provenance -- Row-level provenance tracking for ML pipelines.

Maintains roaring-bitmap annotations that record which rows in every live
DataFrame originated from the *test* partition.  Each propagation helper
produces a new bitmap for the result DataFrame, allowing the downstream
analysis to compute the test-fraction ρ at every DAG node.

Classes
-------
ProvenanceTracker
    High-level façade that initialises bitmaps from a train/test split and
    propagates them through every supported DataFrame operation.
ProvenanceAnnotation
    Per-node, per-column provenance metadata.
SplitDetector
    Heuristic detector that recognises ``train_test_split``, manual index
    splits, and cross-validation iterators.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

from taintflow.core.types import Origin, OpType, ProvenanceInfo, ShapeMetadata
from taintflow.core.errors import InstrumentationError
from taintflow.core.logging_utils import get_logger
from taintflow.utils.bitmap import ProvenanceBitmap, RoaringBitmap

logger = get_logger("taintflow.instrument.provenance")


# ===================================================================
#  SplitInfo — result of train/test split detection
# ===================================================================


@dataclass
class SplitInfo:
    """Information about a detected train/test split."""

    train_indices: list[int]
    test_indices: list[int]
    split_method: str = "unknown"
    n_total: int = 0
    test_fraction: float = 0.0
    fold_index: int = -1
    n_folds: int = 1
    stratified: bool = False
    random_state: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_total == 0:
            self.n_total = len(self.train_indices) + len(self.test_indices)
        if self.test_fraction == 0.0 and self.n_total > 0:
            self.test_fraction = len(self.test_indices) / self.n_total


# ===================================================================
#  ProvenanceAnnotation
# ===================================================================


@dataclass
class ProvenanceAnnotation:
    """Per-node, per-column provenance annotation.

    Attaches a :class:`ProvenanceBitmap` (set of test-row indices) to a
    specific column in a specific DAG node.
    """

    node_id: str
    column_name: str
    test_row_bitmap: ProvenanceBitmap
    total_rows: int = 0
    test_fraction: float = 0.0
    origin: Origin = Origin.TRAIN
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.total_rows > 0 and self.test_fraction == 0.0:
            card = self.test_row_bitmap.cardinality()
            self.test_fraction = card / self.total_rows if self.total_rows > 0 else 0.0
        if self.test_fraction > 0.5:
            self.origin = Origin.TEST
        elif self.test_fraction > 0.0:
            self.origin = Origin.EXTERNAL

    @staticmethod
    def merge_annotations(
        annotations: Sequence["ProvenanceAnnotation"],
    ) -> "ProvenanceAnnotation":
        """Merge several annotations (same node, same column) via bitmap union."""
        if not annotations:
            raise ValueError("Cannot merge zero annotations")
        merged_bm = ProvenanceBitmap()
        total = 0
        node_id = annotations[0].node_id
        column = annotations[0].column_name
        for ann in annotations:
            merged_bm = merged_bm.union(ann.test_row_bitmap)
            total = max(total, ann.total_rows)
        frac = merged_bm.cardinality() / total if total > 0 else 0.0
        return ProvenanceAnnotation(
            node_id=node_id,
            column_name=column,
            test_row_bitmap=merged_bm,
            total_rows=total,
            test_fraction=frac,
        )

    @staticmethod
    def split_annotations(
        annotation: "ProvenanceAnnotation",
        split_mask: Sequence[bool],
    ) -> Tuple["ProvenanceAnnotation", "ProvenanceAnnotation"]:
        """Split an annotation into two by a boolean mask.

        Returns ``(true_part, false_part)`` where *true_part* contains rows
        where the mask is ``True``.
        """
        true_bm = ProvenanceBitmap()
        false_bm = ProvenanceBitmap()
        true_count = 0
        false_count = 0
        for idx, keep in enumerate(split_mask):
            if keep:
                if annotation.test_row_bitmap.contains(idx):
                    true_bm.add(true_count)
                true_count += 1
            else:
                if annotation.test_row_bitmap.contains(idx):
                    false_bm.add(false_count)
                false_count += 1

        true_frac = true_bm.cardinality() / true_count if true_count > 0 else 0.0
        false_frac = false_bm.cardinality() / false_count if false_count > 0 else 0.0

        true_ann = ProvenanceAnnotation(
            node_id=annotation.node_id,
            column_name=annotation.column_name,
            test_row_bitmap=true_bm,
            total_rows=true_count,
            test_fraction=true_frac,
        )
        false_ann = ProvenanceAnnotation(
            node_id=annotation.node_id,
            column_name=annotation.column_name,
            test_row_bitmap=false_bm,
            total_rows=false_count,
            test_fraction=false_frac,
        )
        return true_ann, false_ann


# ===================================================================
#  ProvenanceTracker
# ===================================================================


class ProvenanceTracker:
    """High-level provenance tracker that creates and propagates bitmaps.

    After calling :meth:`initialize_from_split` each downstream
    DataFrame operation invokes the appropriate ``propagate_*`` method
    to produce the result's provenance bitmap.
    """

    def __init__(self) -> None:
        self._bitmaps: dict[int, ProvenanceBitmap] = {}
        self._total: dict[int, int] = {}
        self._annotations: dict[str, dict[str, ProvenanceAnnotation]] = defaultdict(dict)
        self._split_history: list[SplitInfo] = []

    # -- initialisation ----------------------------------------------------

    def initialize_from_split(
        self,
        n_total: int,
        test_indices: Sequence[int],
        *,
        df_id_train: Optional[int] = None,
        df_id_test: Optional[int] = None,
        df_id_full: Optional[int] = None,
    ) -> ProvenanceBitmap:
        """Create the initial provenance bitmap from a known split.

        All bits in *test_indices* are set; the complement is train.
        Returns the full-dataset bitmap.
        """
        full_bm = ProvenanceBitmap()
        for idx in test_indices:
            full_bm.add(idx)

        if df_id_full is not None:
            self._bitmaps[df_id_full] = full_bm
            self._total[df_id_full] = n_total

        if df_id_train is not None:
            train_bm = ProvenanceBitmap()
            self._bitmaps[df_id_train] = train_bm
            train_count = n_total - len(test_indices)
            self._total[df_id_train] = train_count

        if df_id_test is not None:
            test_bm = ProvenanceBitmap()
            for i in range(len(test_indices)):
                test_bm.add(i)
            self._bitmaps[df_id_test] = test_bm
            self._total[df_id_test] = len(test_indices)

        split_info = SplitInfo(
            train_indices=[i for i in range(n_total) if i not in set(test_indices)],
            test_indices=list(test_indices),
            split_method="explicit",
            n_total=n_total,
        )
        self._split_history.append(split_info)
        return full_bm

    def register(self, df_id: int, bitmap: ProvenanceBitmap, total: int) -> None:
        """Manually register a provenance bitmap for a DataFrame id."""
        self._bitmaps[df_id] = bitmap
        self._total[df_id] = total

    # -- propagation helpers -----------------------------------------------

    def propagate_filter(
        self,
        source_prov: ProvenanceBitmap,
        mask: Sequence[bool],
        *,
        source_total: int = 0,
    ) -> ProvenanceBitmap:
        """Propagate through a boolean filter (row selection)."""
        new_bm = ProvenanceBitmap()
        target_idx = 0
        for global_idx, keep in enumerate(mask):
            if keep:
                if source_prov.contains(global_idx):
                    new_bm.add(target_idx)
                target_idx += 1
        return new_bm

    def propagate_index_select(
        self,
        source_prov: ProvenanceBitmap,
        indices: Sequence[int],
    ) -> ProvenanceBitmap:
        """Propagate through integer-index selection (iloc, sample, etc.)."""
        new_bm = ProvenanceBitmap()
        for new_idx, old_idx in enumerate(indices):
            if source_prov.contains(old_idx):
                new_bm.add(new_idx)
        return new_bm

    def propagate_merge(
        self,
        left_prov: ProvenanceBitmap,
        right_prov: ProvenanceBitmap,
        join_type: str,
        left_on: Optional[Sequence[int]] = None,
        right_on: Optional[Sequence[int]] = None,
    ) -> ProvenanceBitmap:
        """Propagate through a merge / join.

        A result row is test-tainted if *either* contributing row was test.
        """
        new_bm = ProvenanceBitmap()
        if left_on is not None and right_on is not None:
            for i in range(len(left_on)):
                li = left_on[i]
                ri = right_on[i]
                left_test = li >= 0 and left_prov.contains(li)
                right_test = ri >= 0 and right_prov.contains(ri)
                if left_test or right_test:
                    new_bm.add(i)
        else:
            left_card = left_prov.cardinality()
            right_card = right_prov.cardinality()
            for i in range(max(left_card, right_card)):
                if left_prov.contains(i) or right_prov.contains(i):
                    new_bm.add(i)
        return new_bm

    def propagate_groupby(
        self,
        source_prov: ProvenanceBitmap,
        group_assignments: Mapping[Any, Sequence[int]],
    ) -> dict[Any, ProvenanceBitmap]:
        """Propagate provenance through a groupby.

        Returns a bitmap per group.  Each group's bitmap has local indices
        (0 to len(group)-1) with test bits set.
        """
        result: dict[Any, ProvenanceBitmap] = {}
        for group_key, row_indices in group_assignments.items():
            bm = ProvenanceBitmap()
            for local_idx, global_idx in enumerate(row_indices):
                if source_prov.contains(global_idx):
                    bm.add(local_idx)
            result[group_key] = bm
        return result

    def propagate_groupby_aggregate(
        self,
        source_prov: ProvenanceBitmap,
        group_assignments: Mapping[Any, Sequence[int]],
    ) -> ProvenanceBitmap:
        """Propagate through groupby + aggregation.

        Each group collapses to one row.  The result row is test-tainted
        if *any* row in that group was a test row.
        """
        result_bm = ProvenanceBitmap()
        for group_idx, (group_key, row_indices) in enumerate(group_assignments.items()):
            for global_idx in row_indices:
                if source_prov.contains(global_idx):
                    result_bm.add(group_idx)
                    break
        return result_bm

    def propagate_concat(
        self,
        provenances: Sequence[ProvenanceBitmap],
        totals: Sequence[int],
        axis: int = 0,
    ) -> ProvenanceBitmap:
        """Propagate provenance through concatenation."""
        new_bm = ProvenanceBitmap()
        if axis == 0:
            offset = 0
            for prov, n in zip(provenances, totals):
                for row in range(n):
                    if prov.contains(row):
                        new_bm.add(offset + row)
                offset += n
        else:
            if provenances:
                first = provenances[0]
                n = totals[0] if totals else 0
                for row in range(n):
                    if first.contains(row):
                        new_bm.add(row)
        return new_bm

    def propagate_sample(
        self,
        source_prov: ProvenanceBitmap,
        indices: Sequence[int],
    ) -> ProvenanceBitmap:
        """Propagate through a sample (random row selection)."""
        return self.propagate_index_select(source_prov, indices)

    def propagate_sort(
        self,
        source_prov: ProvenanceBitmap,
    ) -> ProvenanceBitmap:
        """Sort does not change test membership — return same bitmap."""
        return source_prov

    def propagate_reshape(
        self,
        source_prov: ProvenanceBitmap,
        source_total: int,
        target_total: int,
    ) -> ProvenanceBitmap:
        """Conservative propagation for reshape ops (pivot, melt, etc.).

        Preserves the test fraction by distributing test bits proportionally.
        """
        if source_total == 0 or target_total == 0:
            return ProvenanceBitmap()
        frac = source_prov.cardinality() / source_total
        new_bm = ProvenanceBitmap()
        n_test = int(round(frac * target_total))
        for i in range(n_test):
            new_bm.add(i)
        return new_bm

    def propagate_assign(
        self,
        source_prov: ProvenanceBitmap,
    ) -> ProvenanceBitmap:
        """Assign/setitem preserves row provenance."""
        return source_prov

    # -- queries -----------------------------------------------------------

    def compute_test_fraction(self, provenance: ProvenanceBitmap, total: int = 0) -> float:
        """Compute ρ — the fraction of test rows."""
        card = provenance.cardinality()
        if total <= 0:
            return 0.0
        return min(card / total, 1.0)

    def get_bitmap(self, df_id: int) -> Optional[ProvenanceBitmap]:
        return self._bitmaps.get(df_id)

    def get_total(self, df_id: int) -> int:
        return self._total.get(df_id, 0)

    def test_fraction_for(self, df_id: int) -> float:
        bm = self._bitmaps.get(df_id)
        total = self._total.get(df_id, 0)
        if bm is None or total <= 0:
            return 0.0
        return min(bm.cardinality() / total, 1.0)

    # -- annotation management ---------------------------------------------

    def add_annotation(self, annotation: ProvenanceAnnotation) -> None:
        """Register a provenance annotation for a (node, column) pair."""
        self._annotations[annotation.node_id][annotation.column_name] = annotation

    def get_annotation(
        self, node_id: str, column_name: str
    ) -> Optional[ProvenanceAnnotation]:
        return self._annotations.get(node_id, {}).get(column_name)

    def get_node_annotations(self, node_id: str) -> dict[str, ProvenanceAnnotation]:
        return dict(self._annotations.get(node_id, {}))

    def clear(self) -> None:
        self._bitmaps.clear()
        self._total.clear()
        self._annotations.clear()
        self._split_history.clear()


# ===================================================================
#  SplitDetector
# ===================================================================


_TRAIN_TEST_SPLIT_NAMES = frozenset({
    "train_test_split",
    "sklearn.model_selection.train_test_split",
    "sklearn.model_selection._split.train_test_split",
})

_KFOLD_CLASS_NAMES = frozenset({
    "KFold",
    "StratifiedKFold",
    "GroupKFold",
    "RepeatedKFold",
    "RepeatedStratifiedKFold",
    "LeaveOneOut",
    "LeavePOut",
    "ShuffleSplit",
    "StratifiedShuffleSplit",
    "GroupShuffleSplit",
    "TimeSeriesSplit",
})

_SPLIT_VAR_PATTERN = re.compile(
    r"(X|y|df|data|features?|labels?)[\s_]*(train|test|val|valid)",
    re.IGNORECASE,
)


class SplitDetector:
    """Heuristic detector for train/test splitting operations.

    Recognises:
    - ``sklearn.model_selection.train_test_split``
    - Manual index-based splits (``df.iloc[train_idx]``, ``df.iloc[test_idx]``)
    - Cross-validation iterators (``KFold``, ``StratifiedKFold``, etc.)
    - Variable-name heuristics (``X_train``, ``X_test``)
    """

    def __init__(self) -> None:
        self._detected_splits: list[SplitInfo] = []

    # -- public API --------------------------------------------------------

    def detect_train_test_split(
        self,
        function_name: str,
        args: Sequence[Any],
        return_val: Any,
    ) -> Optional[SplitInfo]:
        """Try to detect a train/test split from a function call.

        Returns ``SplitInfo`` if a split is detected, otherwise ``None``.
        """
        full_name = function_name
        result = (
            self._detect_sklearn_split(full_name, args, return_val)
            or self._detect_manual_iloc_split(full_name, args, return_val)
            or self._detect_kfold_iteration(full_name, args, return_val)
        )
        if result is not None:
            self._detected_splits.append(result)
        return result

    def detect_from_variable_names(
        self,
        local_vars: Mapping[str, Any],
    ) -> Optional[SplitInfo]:
        """Try to infer a split from variable names in a scope."""
        train_vars: dict[str, Any] = {}
        test_vars: dict[str, Any] = {}

        for name, val in local_vars.items():
            lower = name.lower()
            if "train" in lower:
                train_vars[name] = val
            elif "test" in lower or "val" in lower:
                test_vars[name] = val

        if not train_vars or not test_vars:
            return None

        train_key = next(iter(train_vars))
        test_key = next(iter(test_vars))
        train_val = train_vars[train_key]
        test_val = test_vars[test_key]

        train_len = self._safe_len(train_val)
        test_len = self._safe_len(test_val)

        if train_len <= 0 or test_len <= 0:
            return None

        n_total = train_len + test_len
        test_indices = list(range(train_len, n_total))
        train_indices = list(range(train_len))

        info = SplitInfo(
            train_indices=train_indices,
            test_indices=test_indices,
            split_method=f"variable_name_heuristic({train_key}/{test_key})",
            n_total=n_total,
            metadata={"train_var": train_key, "test_var": test_key},
        )
        self._detected_splits.append(info)
        return info

    @property
    def detected_splits(self) -> list[SplitInfo]:
        return list(self._detected_splits)

    # -- internal detectors ------------------------------------------------

    def _detect_sklearn_split(
        self, function_name: str, args: Sequence[Any], return_val: Any
    ) -> Optional[SplitInfo]:
        """Detect ``sklearn.model_selection.train_test_split``."""
        base_name = function_name.rsplit(".", 1)[-1] if "." in function_name else function_name
        if base_name != "train_test_split":
            return None

        if not isinstance(return_val, (list, tuple)):
            return None

        if len(return_val) < 2:
            return None

        first_train = return_val[0]
        first_test = return_val[1]

        train_len = self._safe_len(first_train)
        test_len = self._safe_len(first_test)

        if train_len <= 0 or test_len <= 0:
            return None

        n_total = train_len + test_len

        try:
            if hasattr(first_test, "index"):
                test_indices = list(first_test.index)
            else:
                test_indices = list(range(train_len, n_total))
        except Exception:
            test_indices = list(range(train_len, n_total))

        try:
            if hasattr(first_train, "index"):
                train_indices = list(first_train.index)
            else:
                train_indices = list(range(train_len))
        except Exception:
            train_indices = list(range(train_len))

        random_state = None
        for arg in args:
            if isinstance(arg, int) and 0 <= arg <= 2**31:
                random_state = arg
                break

        return SplitInfo(
            train_indices=train_indices,
            test_indices=test_indices,
            split_method="sklearn.train_test_split",
            n_total=n_total,
            random_state=random_state,
        )

    def _detect_manual_iloc_split(
        self, function_name: str, args: Sequence[Any], return_val: Any
    ) -> Optional[SplitInfo]:
        """Detect manual index-based splitting (iloc/loc with explicit indices)."""
        if function_name not in ("__getitem__", "iloc.__getitem__", "loc.__getitem__"):
            return None

        if not args:
            return None

        indexer = args[0] if len(args) == 1 else args

        try:
            if hasattr(indexer, "__len__") and hasattr(indexer, "__iter__"):
                indices = list(indexer)
                if all(isinstance(i, (int,)) for i in indices):
                    result_len = self._safe_len(return_val)
                    if result_len > 0:
                        return SplitInfo(
                            train_indices=indices,
                            test_indices=[],
                            split_method="manual_iloc",
                            n_total=max(indices) + 1 if indices else 0,
                            metadata={"partial": True},
                        )
        except (TypeError, ValueError):
            pass

        return None

    def _detect_kfold_iteration(
        self, function_name: str, args: Sequence[Any], return_val: Any
    ) -> Optional[SplitInfo]:
        """Detect KFold / StratifiedKFold iteration."""
        base_name = function_name.rsplit(".", 1)[-1] if "." in function_name else function_name
        if base_name != "split":
            return None

        if not isinstance(return_val, (list, tuple)):
            return None

        if len(return_val) != 2:
            return None

        train_idx_arr, test_idx_arr = return_val

        try:
            train_indices = list(train_idx_arr)
            test_indices = list(test_idx_arr)
        except (TypeError, ValueError):
            return None

        if not train_indices or not test_indices:
            return None

        all_indices = sorted(set(train_indices) | set(test_indices))
        n_total = len(all_indices)
        n_folds_guess = max(1, round(n_total / len(test_indices))) if test_indices else 1

        class_name = ""
        if args and hasattr(args[0], "__class__"):
            class_name = type(args[0]).__name__

        return SplitInfo(
            train_indices=train_indices,
            test_indices=test_indices,
            split_method=f"kfold_split({class_name})" if class_name else "kfold_split",
            n_total=n_total,
            n_folds=n_folds_guess,
            stratified="Stratified" in class_name,
            metadata={"class_name": class_name},
        )

    # -- utilities ---------------------------------------------------------

    @staticmethod
    def _safe_len(obj: Any) -> int:
        """Return ``len(obj)`` if available, otherwise -1."""
        try:
            if hasattr(obj, "shape"):
                return int(obj.shape[0])
            return len(obj)
        except (TypeError, AttributeError):
            return -1

    def clear(self) -> None:
        self._detected_splits.clear()

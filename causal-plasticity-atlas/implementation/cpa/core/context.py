"""Context and context-space management for the CPA engine.

Provides the :class:`ContextSpace` for managing collections of
contexts (ordered or unordered), distance computation between
contexts, subset operations, ordering validation, interpolation,
and :class:`ContextPartition` for grouping contexts by criteria.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray

from cpa.core.types import Context


# ===================================================================
# ContextSpace
# ===================================================================


class ContextSpace:
    """Collection of contexts with optional ordering.

    A ContextSpace manages a list of :class:`Context` objects, supports
    ordered and unordered collections, and provides utility methods
    for distance computation, subsetting, and iteration.

    Parameters
    ----------
    contexts : list of Context, optional
        Initial contexts.
    ordered : bool
        If ``True``, contexts are maintained in order by their
        ``ordering_value``.

    Examples
    --------
    >>> cs = ContextSpace(ordered=True)
    >>> cs.add(Context("c1", ordering_value=0.0))
    >>> cs.add(Context("c2", ordering_value=1.0))
    >>> cs.ordering_values()
    [0.0, 1.0]
    """

    def __init__(
        self,
        contexts: Optional[List[Context]] = None,
        *,
        ordered: bool = False,
    ) -> None:
        self._ordered = ordered
        self._contexts: List[Context] = []
        self._id_map: Dict[str, Context] = {}
        if contexts:
            for ctx in contexts:
                self.add(ctx)

    @property
    def ordered(self) -> bool:
        """Whether this space maintains an ordering."""
        return self._ordered

    @property
    def size(self) -> int:
        """Number of contexts."""
        return len(self._contexts)

    @property
    def ids(self) -> List[str]:
        """List of context identifiers."""
        return [c.id for c in self._contexts]

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[Context]:
        return iter(self._contexts)

    def __contains__(self, context_id: str) -> bool:
        return context_id in self._id_map

    def __getitem__(self, key: Union[int, str]) -> Context:
        if isinstance(key, int):
            return self._contexts[key]
        if isinstance(key, str):
            if key not in self._id_map:
                raise KeyError(f"Context {key!r} not found")
            return self._id_map[key]
        raise TypeError(f"Key must be int or str, got {type(key).__name__}")

    # -----------------------------------------------------------------
    # Add / Remove
    # -----------------------------------------------------------------

    def add(self, context: Context) -> None:
        """Add a context to the space.

        Parameters
        ----------
        context : Context

        Raises
        ------
        ValueError
            If a context with the same id already exists, or if the
            space is ordered and the context has no ordering value.
        """
        if context.id in self._id_map:
            raise ValueError(f"Context {context.id!r} already exists")
        if self._ordered and context.ordering_value is None:
            raise ValueError(
                f"Ordered ContextSpace requires ordering_value, "
                f"but context {context.id!r} has None"
            )
        self._contexts.append(context)
        self._id_map[context.id] = context
        if self._ordered:
            self._contexts.sort(
                key=lambda c: c.ordering_value if c.ordering_value is not None else 0
            )

    def remove(self, context_id: str) -> Context:
        """Remove and return a context by id.

        Parameters
        ----------
        context_id : str

        Returns
        -------
        Context

        Raises
        ------
        KeyError
        """
        if context_id not in self._id_map:
            raise KeyError(f"Context {context_id!r} not found")
        ctx = self._id_map.pop(context_id)
        self._contexts = [c for c in self._contexts if c.id != context_id]
        return ctx

    def get(self, context_id: str) -> Optional[Context]:
        """Get context by id, returning None if not found.

        Parameters
        ----------
        context_id : str

        Returns
        -------
        Context or None
        """
        return self._id_map.get(context_id)

    # -----------------------------------------------------------------
    # Ordering operations
    # -----------------------------------------------------------------

    def ordering_values(self) -> List[Optional[float]]:
        """Return ordering values for all contexts.

        Returns
        -------
        list of float or None
        """
        return [c.ordering_value for c in self._contexts]

    def validate_ordering(self) -> bool:
        """Check that ordering values are strictly increasing.

        Returns
        -------
        bool
        """
        if not self._ordered:
            return True
        vals = [c.ordering_value for c in self._contexts]
        for i in range(1, len(vals)):
            if vals[i] is None or vals[i - 1] is None:
                return False
            if vals[i] <= vals[i - 1]:  # type: ignore[operator]
                return False
        return True

    def interpolate_ordering(
        self,
        value: float,
        *,
        method: str = "nearest",
    ) -> Tuple[Context, Context]:
        """Find the two contexts bracketing *value* in the ordering.

        Parameters
        ----------
        value : float
            Target ordering value.
        method : str
            Interpolation method (currently only ``"nearest"``).

        Returns
        -------
        (left, right) : tuple of Context
            Contexts immediately before and after *value*.

        Raises
        ------
        ValueError
            If the space is unordered or has fewer than 2 contexts.
        """
        if not self._ordered:
            raise ValueError("Cannot interpolate in unordered ContextSpace")
        if self.size < 2:
            raise ValueError("Need at least 2 contexts to interpolate")

        vals = self.ordering_values()
        for i in range(len(vals) - 1):
            if vals[i] is not None and vals[i + 1] is not None:
                if vals[i] <= value <= vals[i + 1]:  # type: ignore[operator]
                    return self._contexts[i], self._contexts[i + 1]

        # Value is outside range; clamp to endpoints
        if value <= (vals[0] or 0):
            return self._contexts[0], self._contexts[1]
        return self._contexts[-2], self._contexts[-1]

    def sliding_window(self, window_size: int) -> List[List[Context]]:
        """Generate overlapping windows of contexts.

        Parameters
        ----------
        window_size : int
            Size of each window.

        Returns
        -------
        list of list of Context
        """
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if window_size > self.size:
            return [list(self._contexts)]
        return [
            self._contexts[i : i + window_size]
            for i in range(self.size - window_size + 1)
        ]

    # -----------------------------------------------------------------
    # Distance computation
    # -----------------------------------------------------------------

    def metadata_distance(
        self,
        ctx_a: str,
        ctx_b: str,
        *,
        keys: Optional[List[str]] = None,
        metric: str = "euclidean",
    ) -> float:
        """Compute distance between two contexts based on metadata.

        Extracts numeric values from metadata dictionaries and computes
        a distance.

        Parameters
        ----------
        ctx_a, ctx_b : str
            Context ids.
        keys : list of str, optional
            Metadata keys to use.  If ``None``, uses all shared numeric keys.
        metric : ``"euclidean"`` or ``"manhattan"``
            Distance metric.

        Returns
        -------
        float
            Distance.
        """
        a = self._id_map[ctx_a]
        b = self._id_map[ctx_b]

        if keys is None:
            shared_keys = sorted(
                set(a.metadata.keys()) & set(b.metadata.keys())
            )
            keys = [
                k
                for k in shared_keys
                if isinstance(a.metadata[k], (int, float))
                and isinstance(b.metadata[k], (int, float))
            ]
        if not keys:
            return 0.0

        va = np.array([float(a.metadata[k]) for k in keys])
        vb = np.array([float(b.metadata[k]) for k in keys])

        if metric == "euclidean":
            return float(np.linalg.norm(va - vb))
        elif metric == "manhattan":
            return float(np.sum(np.abs(va - vb)))
        else:
            raise ValueError(f"Unknown metric {metric!r}")

    def ordering_distance(self, ctx_a: str, ctx_b: str) -> float:
        """Distance between two contexts in the ordering dimension.

        Parameters
        ----------
        ctx_a, ctx_b : str

        Returns
        -------
        float
        """
        a = self._id_map[ctx_a]
        b = self._id_map[ctx_b]
        if a.ordering_value is None or b.ordering_value is None:
            raise ValueError("Both contexts must have ordering values")
        return abs(a.ordering_value - b.ordering_value)

    def pairwise_distance_matrix(
        self,
        *,
        metric: str = "ordering",
        keys: Optional[List[str]] = None,
    ) -> NDArray[np.float64]:
        """Compute pairwise distance matrix.

        Parameters
        ----------
        metric : ``"ordering"`` or ``"metadata"``
            Distance type.
        keys : list of str, optional
            Metadata keys (for metadata metric).

        Returns
        -------
        np.ndarray
            Symmetric distance matrix, shape ``(n, n)``.
        """
        n = self.size
        D = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                if metric == "ordering":
                    d = self.ordering_distance(
                        self._contexts[i].id, self._contexts[j].id
                    )
                elif metric == "metadata":
                    d = self.metadata_distance(
                        self._contexts[i].id,
                        self._contexts[j].id,
                        keys=keys,
                    )
                else:
                    raise ValueError(f"Unknown metric {metric!r}")
                D[i, j] = d
                D[j, i] = d
        return D

    # -----------------------------------------------------------------
    # Subset operations
    # -----------------------------------------------------------------

    def subset(self, ids: Iterable[str]) -> "ContextSpace":
        """Create a ContextSpace containing only the specified ids.

        Parameters
        ----------
        ids : iterable of str

        Returns
        -------
        ContextSpace
        """
        selected = [self._id_map[cid] for cid in ids if cid in self._id_map]
        return ContextSpace(selected, ordered=self._ordered)

    def filter(self, predicate: Callable[[Context], bool]) -> "ContextSpace":
        """Filter contexts by a predicate function.

        Parameters
        ----------
        predicate : callable
            Function taking a Context and returning bool.

        Returns
        -------
        ContextSpace
        """
        filtered = [c for c in self._contexts if predicate(c)]
        return ContextSpace(filtered, ordered=self._ordered)

    def union(self, other: "ContextSpace") -> "ContextSpace":
        """Union of two context spaces.

        Parameters
        ----------
        other : ContextSpace

        Returns
        -------
        ContextSpace
        """
        merged = ContextSpace(ordered=self._ordered and other._ordered)
        for ctx in self._contexts:
            merged.add(ctx)
        for ctx in other._contexts:
            if ctx.id not in merged:
                merged.add(ctx)
        return merged

    def intersection(self, other: "ContextSpace") -> "ContextSpace":
        """Intersection of two context spaces.

        Parameters
        ----------
        other : ContextSpace

        Returns
        -------
        ContextSpace
        """
        shared_ids = set(self.ids) & set(other.ids)
        return self.subset(shared_ids)

    def difference(self, other: "ContextSpace") -> "ContextSpace":
        """Contexts in *self* but not in *other*.

        Parameters
        ----------
        other : ContextSpace

        Returns
        -------
        ContextSpace
        """
        diff_ids = set(self.ids) - set(other.ids)
        return self.subset(diff_ids)

    # -----------------------------------------------------------------
    # Pairwise iteration
    # -----------------------------------------------------------------

    def pairs(self) -> List[Tuple[Context, Context]]:
        """All unordered pairs of contexts.

        Returns
        -------
        list of (Context, Context)
        """
        return list(itertools.combinations(self._contexts, 2))

    def consecutive_pairs(self) -> List[Tuple[Context, Context]]:
        """Consecutive pairs (only meaningful for ordered spaces).

        Returns
        -------
        list of (Context, Context)
        """
        return [
            (self._contexts[i], self._contexts[i + 1])
            for i in range(self.size - 1)
        ]

    # -----------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        dict
        """
        return {
            "ordered": self._ordered,
            "contexts": [c.to_dict() for c in self._contexts],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ContextSpace":
        """Deserialize from dictionary.

        Parameters
        ----------
        d : dict

        Returns
        -------
        ContextSpace
        """
        return cls(
            contexts=[Context.from_dict(c) for c in d["contexts"]],
            ordered=d.get("ordered", False),
        )

    def __repr__(self) -> str:
        order_str = "ordered" if self._ordered else "unordered"
        return f"ContextSpace({order_str}, n={self.size}, ids={self.ids})"


# ===================================================================
# ContextPartition
# ===================================================================


class ContextPartition:
    """Partition of contexts into disjoint groups.

    Parameters
    ----------
    groups : dict
        Mapping ``group_name → list of context ids``.
    context_space : ContextSpace
        Parent context space.

    Examples
    --------
    >>> cp = ContextPartition.from_metadata_key(context_space, "treatment")
    >>> cp.group_names
    ['control', 'treatment_a', 'treatment_b']
    """

    def __init__(
        self,
        groups: Dict[str, List[str]],
        context_space: ContextSpace,
    ) -> None:
        self._groups = {k: list(v) for k, v in groups.items()}
        self._cs = context_space
        self._validate()

    def _validate(self) -> None:
        """Validate partition: all ids exist, partition is disjoint."""
        all_ids: set[str] = set()
        for name, ids in self._groups.items():
            overlap = all_ids & set(ids)
            if overlap:
                raise ValueError(
                    f"Partition group {name!r} overlaps: {overlap}"
                )
            all_ids |= set(ids)
        # Check ids exist
        for cid in all_ids:
            if cid not in self._cs:
                raise ValueError(f"Context {cid!r} not in context space")

    @property
    def group_names(self) -> List[str]:
        """Names of the partition groups."""
        return sorted(self._groups.keys())

    @property
    def num_groups(self) -> int:
        """Number of groups."""
        return len(self._groups)

    def group(self, name: str) -> List[str]:
        """Context ids in group *name*.

        Parameters
        ----------
        name : str

        Returns
        -------
        list of str
        """
        if name not in self._groups:
            raise KeyError(f"Group {name!r} not found")
        return list(self._groups[name])

    def group_space(self, name: str) -> ContextSpace:
        """Return a ContextSpace for a single group.

        Parameters
        ----------
        name : str

        Returns
        -------
        ContextSpace
        """
        return self._cs.subset(self.group(name))

    def group_sizes(self) -> Dict[str, int]:
        """Size of each group.

        Returns
        -------
        dict
        """
        return {k: len(v) for k, v in self._groups.items()}

    def context_group(self, context_id: str) -> str:
        """Return the group name for *context_id*.

        Parameters
        ----------
        context_id : str

        Returns
        -------
        str

        Raises
        ------
        ValueError
        """
        for name, ids in self._groups.items():
            if context_id in ids:
                return name
        raise ValueError(f"Context {context_id!r} not in any partition group")

    # -----------------------------------------------------------------
    # Factory methods
    # -----------------------------------------------------------------

    @classmethod
    def from_metadata_key(
        cls,
        context_space: ContextSpace,
        key: str,
    ) -> "ContextPartition":
        """Partition contexts by a metadata key's value.

        Parameters
        ----------
        context_space : ContextSpace
        key : str
            Metadata key to partition by.

        Returns
        -------
        ContextPartition
        """
        groups: dict[str, list[str]] = defaultdict(list)
        for ctx in context_space:
            val = ctx.metadata.get(key, "__MISSING__")
            groups[str(val)].append(ctx.id)
        return cls(dict(groups), context_space)

    @classmethod
    def from_ordering_splits(
        cls,
        context_space: ContextSpace,
        split_points: Sequence[float],
    ) -> "ContextPartition":
        """Partition ordered contexts by split points.

        Parameters
        ----------
        context_space : ContextSpace
        split_points : sequence of float
            Ordering values at which to split.

        Returns
        -------
        ContextPartition
        """
        boundaries = [-math.inf] + sorted(split_points) + [math.inf]
        groups: dict[str, list[str]] = {}
        for i in range(len(boundaries) - 1):
            name = f"segment_{i}"
            groups[name] = []
            for ctx in context_space:
                if ctx.ordering_value is not None:
                    if boundaries[i] <= ctx.ordering_value < boundaries[i + 1]:
                        groups[name].append(ctx.id)
        # Remove empty groups
        groups = {k: v for k, v in groups.items() if v}
        return cls(groups, context_space)

    @classmethod
    def from_predicate(
        cls,
        context_space: ContextSpace,
        predicate: Callable[[Context], str],
    ) -> "ContextPartition":
        """Partition contexts using a predicate function.

        Parameters
        ----------
        context_space : ContextSpace
        predicate : callable
            Function mapping a Context to a group name string.

        Returns
        -------
        ContextPartition
        """
        groups: dict[str, list[str]] = defaultdict(list)
        for ctx in context_space:
            name = predicate(ctx)
            groups[name].append(ctx.id)
        return cls(dict(groups), context_space)

    # -----------------------------------------------------------------
    # Pairwise group operations
    # -----------------------------------------------------------------

    def inter_group_pairs(self) -> List[Tuple[str, str]]:
        """All unordered pairs of group names.

        Returns
        -------
        list of (str, str)
        """
        return list(itertools.combinations(self.group_names, 2))

    def cross_group_context_pairs(
        self, group_a: str, group_b: str
    ) -> List[Tuple[str, str]]:
        """All context-id pairs across two groups.

        Parameters
        ----------
        group_a, group_b : str
            Group names.

        Returns
        -------
        list of (str, str)
        """
        return list(itertools.product(self.group(group_a), self.group(group_b)))

    # -----------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        dict
        """
        return {
            "groups": {k: list(v) for k, v in self._groups.items()},
        }

    @classmethod
    def from_dict(
        cls,
        d: Dict[str, Any],
        context_space: ContextSpace,
    ) -> "ContextPartition":
        """Deserialize from dictionary.

        Parameters
        ----------
        d : dict
        context_space : ContextSpace

        Returns
        -------
        ContextPartition
        """
        return cls(
            groups={k: list(v) for k, v in d["groups"].items()},
            context_space=context_space,
        )

    def __repr__(self) -> str:
        sizes = self.group_sizes()
        return (
            f"ContextPartition(groups={self.num_groups}, "
            f"sizes={sizes})"
        )

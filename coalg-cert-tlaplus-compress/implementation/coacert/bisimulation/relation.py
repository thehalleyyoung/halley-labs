"""
Bisimulation relation representation via union-find.

Stores an equivalence relation over states using a union-find (disjoint-set)
data structure with path compression and union by rank.  Supports equivalence
class enumeration, partition refinement operations, coarsest partition
computation, partition intersection, comparison, and serialization.
"""

from __future__ import annotations

import json
import logging
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
    Mapping,
    Optional,
    Set,
    Tuple,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Statistics snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RelationStats:
    """Snapshot of relation metrics."""

    num_states: int
    num_classes: int
    largest_class: int
    smallest_class: int
    mean_class_size: float
    compression_ratio: float  # 1 - num_classes / num_states


# ---------------------------------------------------------------------------
# Equivalence class descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EquivClassInfo:
    """Descriptor for a single equivalence class."""

    representative: str
    members: FrozenSet[str]
    size: int


# ---------------------------------------------------------------------------
# BisimulationRelation
# ---------------------------------------------------------------------------

class BisimulationRelation:
    """Equivalence relation backed by a union-find forest.

    Each element is a state name (``str``).  The data structure supports
    efficient *find*, *union*, partition refinement and comparison.
    """

    def __init__(self, states: Optional[Iterable[str]] = None) -> None:
        self._parent: Dict[str, str] = {}
        self._rank: Dict[str, int] = {}
        self._size: Dict[str, int] = {}   # subtree size at root
        self._version: int = 0             # bumped on every mutation
        self._class_cache: Optional[Dict[str, Set[str]]] = None
        self._cache_version: int = -1
        if states is not None:
            for s in states:
                self.make_set(s)

    # -- basic union-find ---------------------------------------------------

    def make_set(self, x: str) -> None:
        """Create a singleton equivalence class for *x* (idempotent)."""
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
            self._size[x] = 1
            self._invalidate()

    def find(self, x: str) -> str:
        """Return the representative of the class containing *x*."""
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        # path compression
        while self._parent[x] != root:
            nxt = self._parent[x]
            self._parent[x] = root
            x = nxt
        return root

    def union(self, x: str, y: str) -> str:
        """Merge the equivalence classes of *x* and *y*.

        Returns the representative of the merged class.
        """
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return rx
        # union by rank
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        self._size[rx] += self._size[ry]
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1
        self._invalidate()
        return rx

    def equivalent(self, x: str, y: str) -> bool:
        """Return ``True`` iff *x* and *y* are in the same equivalence class."""
        return self.find(x) == self.find(y)

    def representative(self, x: str) -> str:
        """Return the canonical representative for the class of *x*."""
        return self.find(x)

    # -- state management ---------------------------------------------------

    @property
    def states(self) -> FrozenSet[str]:
        return frozenset(self._parent.keys())

    def __contains__(self, x: str) -> bool:
        return x in self._parent

    def __len__(self) -> int:
        """Number of states in the relation."""
        return len(self._parent)

    # -- equivalence-class enumeration --------------------------------------

    def _rebuild_cache(self) -> Dict[str, Set[str]]:
        if self._class_cache is not None and self._cache_version == self._version:
            return self._class_cache
        classes: Dict[str, Set[str]] = defaultdict(set)
        for s in self._parent:
            classes[self.find(s)].add(s)
        self._class_cache = dict(classes)
        self._cache_version = self._version
        return self._class_cache

    def class_of(self, x: str) -> FrozenSet[str]:
        """All members of the equivalence class containing *x*."""
        cache = self._rebuild_cache()
        return frozenset(cache.get(self.find(x), {x}))

    def classes(self) -> List[FrozenSet[str]]:
        """Return all equivalence classes as frozen sets."""
        cache = self._rebuild_cache()
        return [frozenset(members) for members in cache.values()]

    def class_map(self) -> Dict[str, FrozenSet[str]]:
        """Map from representative to equivalence-class members."""
        cache = self._rebuild_cache()
        return {rep: frozenset(members) for rep, members in cache.items()}

    def representatives(self) -> FrozenSet[str]:
        """All current representatives (roots of the forest)."""
        cache = self._rebuild_cache()
        return frozenset(cache.keys())

    def num_classes(self) -> int:
        return len(self._rebuild_cache())

    def __iter__(self) -> Iterator[FrozenSet[str]]:
        """Iterate over equivalence classes."""
        return iter(self.classes())

    def iter_class_info(self) -> Iterator[EquivClassInfo]:
        """Yield :class:`EquivClassInfo` for each equivalence class."""
        for rep, members in self._rebuild_cache().items():
            fs = frozenset(members)
            yield EquivClassInfo(representative=rep, members=fs, size=len(fs))

    # -- statistics ---------------------------------------------------------

    def stats(self) -> RelationStats:
        cache = self._rebuild_cache()
        n = len(self._parent)
        k = len(cache)
        if k == 0:
            return RelationStats(
                num_states=0,
                num_classes=0,
                largest_class=0,
                smallest_class=0,
                mean_class_size=0.0,
                compression_ratio=0.0,
            )
        sizes = [len(m) for m in cache.values()]
        return RelationStats(
            num_states=n,
            num_classes=k,
            largest_class=max(sizes),
            smallest_class=min(sizes),
            mean_class_size=n / k,
            compression_ratio=1.0 - k / n if n > 0 else 0.0,
        )

    # -- invalidation helpers -----------------------------------------------

    def _invalidate(self) -> None:
        self._version += 1

    # -- partition refinement operations ------------------------------------

    def refine_by(self, discriminator: Callable[[str], Any]) -> int:
        """Split each equivalence class by a discriminator function.

        States *s*, *t* in the same class are separated if
        ``discriminator(s) != discriminator(t)``.  Returns the number of
        new classes created.
        """
        old_count = self.num_classes()
        new_rel = BisimulationRelation()
        cache = self._rebuild_cache()

        for _rep, members in cache.items():
            sub: Dict[Any, List[str]] = defaultdict(list)
            for s in members:
                sub[discriminator(s)].append(s)
            for group in sub.values():
                new_rel.make_set(group[0])
                for s in group[1:]:
                    new_rel.make_set(s)
                    new_rel.union(group[0], s)

        # commit the new partition
        self._parent = dict(new_rel._parent)
        self._rank = dict(new_rel._rank)
        self._size = dict(new_rel._size)
        self._invalidate()
        return self.num_classes() - old_count

    def split_class(
        self, rep: str, keep: Set[str], remove: Set[str]
    ) -> Tuple[str, str]:
        """Split the class of *rep* into *keep* and *remove*.

        Both sets must be non-empty and partition the original class.
        Returns (new_rep_keep, new_rep_remove).
        """
        members = self.class_of(rep)
        if not keep or not remove:
            raise ValueError("Both keep and remove must be non-empty")
        if keep | remove != members:
            raise ValueError("keep ∪ remove must equal the original class")
        if keep & remove:
            raise ValueError("keep and remove must be disjoint")

        new_rel = BisimulationRelation()
        # rebuild everything preserving other classes
        cache = self._rebuild_cache()
        for r, ms in cache.items():
            if r == self.find(rep):
                # split this class
                klist = sorted(keep)
                for s in klist:
                    new_rel.make_set(s)
                for s in klist[1:]:
                    new_rel.union(klist[0], s)
                rlist = sorted(remove)
                for s in rlist:
                    new_rel.make_set(s)
                for s in rlist[1:]:
                    new_rel.union(rlist[0], s)
            else:
                mlist = sorted(ms)
                for s in mlist:
                    new_rel.make_set(s)
                for s in mlist[1:]:
                    new_rel.union(mlist[0], s)

        self._parent = dict(new_rel._parent)
        self._rank = dict(new_rel._rank)
        self._size = dict(new_rel._size)
        self._invalidate()
        return new_rel.find(sorted(keep)[0]), new_rel.find(sorted(remove)[0])

    # -- coarsest partition -------------------------------------------------

    @classmethod
    def coarsest(cls, states: Iterable[str]) -> "BisimulationRelation":
        """Build the coarsest partition: all states in one class."""
        rel = cls()
        first: Optional[str] = None
        for s in states:
            rel.make_set(s)
            if first is None:
                first = s
            else:
                rel.union(first, s)
        return rel

    @classmethod
    def discrete(cls, states: Iterable[str]) -> "BisimulationRelation":
        """Build the discrete (finest) partition: each state alone."""
        rel = cls()
        for s in states:
            rel.make_set(s)
        return rel

    @classmethod
    def from_labeling(
        cls,
        states: Iterable[str],
        labeling: Callable[[str], Any],
    ) -> "BisimulationRelation":
        """Build an initial partition where states with the same label are grouped."""
        rel = cls()
        groups: Dict[Any, List[str]] = defaultdict(list)
        for s in states:
            rel.make_set(s)
            key = labeling(s)
            groups[key].append(s)
        for group in groups.values():
            for s in group[1:]:
                rel.union(group[0], s)
        return rel

    # -- partition intersection ---------------------------------------------

    def intersect(self, other: "BisimulationRelation") -> "BisimulationRelation":
        """Compute the intersection (meet) of two equivalence relations.

        States are equivalent in the result iff they are equivalent in
        both ``self`` and ``other``.
        """
        common = self.states & other.states
        result = BisimulationRelation()
        groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for s in common:
            key = (self.find(s), other.find(s))
            result.make_set(s)
            groups[key].append(s)
        for group in groups.values():
            for s in group[1:]:
                result.union(group[0], s)
        return result

    def join(self, other: "BisimulationRelation") -> "BisimulationRelation":
        """Compute the join (coarsest common coarsening) of two relations.

        States are equivalent in the result if they can be connected by a
        chain of pairs that are equivalent in *self* or *other*.
        """
        common = self.states & other.states
        result = BisimulationRelation(common)
        for cls_members in self.classes():
            mlist = [s for s in cls_members if s in common]
            for s in mlist[1:]:
                result.union(mlist[0], s)
        for cls_members in other.classes():
            mlist = [s for s in cls_members if s in common]
            for s in mlist[1:]:
                result.union(mlist[0], s)
        return result

    # -- comparison ---------------------------------------------------------

    def is_finer_than(self, other: "BisimulationRelation") -> bool:
        """Return True if ``self`` is finer than or equal to ``other``.

        That is, every class in ``self`` is contained in some class of ``other``.
        """
        for cls_members in self.classes():
            reps = {other.find(s) for s in cls_members if s in other}
            if len(reps) > 1:
                return False
        return True

    def is_coarser_than(self, other: "BisimulationRelation") -> bool:
        return other.is_finer_than(self)

    def equals(self, other: "BisimulationRelation") -> bool:
        """True iff the two relations induce the same partition on common states."""
        return self.is_finer_than(other) and other.is_finer_than(self)

    def difference_witnesses(
        self, other: "BisimulationRelation"
    ) -> List[Tuple[str, str]]:
        """Return pairs that are equivalent in ``self`` but not in ``other``."""
        witnesses: List[Tuple[str, str]] = []
        for cls_members in self.classes():
            mlist = sorted(cls_members)
            for i, a in enumerate(mlist):
                for b in mlist[i + 1:]:
                    if a in other and b in other and not other.equivalent(a, b):
                        witnesses.append((a, b))
        return witnesses

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        classes_list = []
        for info in self.iter_class_info():
            classes_list.append({
                "representative": info.representative,
                "members": sorted(info.members),
            })
        return {
            "version": 1,
            "num_states": len(self),
            "num_classes": self.num_classes(),
            "classes": classes_list,
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BisimulationRelation":
        """Deserialize from a dictionary produced by ``to_dict``."""
        rel = cls()
        for cls_data in data["classes"]:
            members = cls_data["members"]
            for s in members:
                rel.make_set(s)
            for s in members[1:]:
                rel.union(members[0], s)
        return rel

    @classmethod
    def from_json(cls, text: str) -> "BisimulationRelation":
        return cls.from_dict(json.loads(text))

    # -- copy ---------------------------------------------------------------

    def copy(self) -> "BisimulationRelation":
        """Return a deep copy of this relation."""
        new = BisimulationRelation()
        new._parent = dict(self._parent)
        new._rank = dict(self._rank)
        new._size = dict(self._size)
        new._version = self._version
        return new

    # -- restrict -----------------------------------------------------------

    def restrict(self, states: Set[str]) -> "BisimulationRelation":
        """Restrict the relation to a subset of states."""
        result = BisimulationRelation()
        groups: Dict[str, List[str]] = defaultdict(list)
        for s in states:
            if s in self:
                result.make_set(s)
                groups[self.find(s)].append(s)
        for group in groups.values():
            for s in group[1:]:
                result.union(group[0], s)
        return result

    # -- mapping -----------------------------------------------------------

    def map_states(self, mapping: Callable[[str], str]) -> "BisimulationRelation":
        """Apply a renaming function to every state."""
        result = BisimulationRelation()
        groups: Dict[str, List[str]] = defaultdict(list)
        for s in self._parent:
            groups[self.find(s)].append(mapping(s))
        for group in groups.values():
            for s in group:
                result.make_set(s)
            for s in group[1:]:
                result.union(group[0], s)
        return result

    # -- partition from explicit blocks -------------------------------------

    @classmethod
    def from_blocks(
        cls, blocks: Iterable[Iterable[str]]
    ) -> "BisimulationRelation":
        """Build a relation from an explicit list of equivalence classes."""
        rel = cls()
        for block in blocks:
            blist = list(block)
            for s in blist:
                rel.make_set(s)
            for s in blist[1:]:
                rel.union(blist[0], s)
        return rel

    # -- merge with another relation ----------------------------------------

    def merge_from(self, other: "BisimulationRelation") -> int:
        """Merge equivalences from *other* into ``self``.

        Returns the number of union operations that actually merged distinct
        classes.
        """
        count = 0
        for cls_members in other.classes():
            mlist = list(cls_members)
            for s in mlist:
                self.make_set(s)
            for s in mlist[1:]:
                if not self.equivalent(mlist[0], s):
                    self.union(mlist[0], s)
                    count += 1
        return count

    # -- dunder -------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BisimulationRelation(states={len(self)}, "
            f"classes={self.num_classes()})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BisimulationRelation):
            return NotImplemented
        return self.equals(other)

    def __hash__(self) -> int:
        # equivalence relations are mutable so not truly hashable; provide a
        # best-effort hash for debugging convenience
        return hash(tuple(sorted(tuple(sorted(c)) for c in self.classes())))

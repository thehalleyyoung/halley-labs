"""
SSA optimization and analysis passes for the Python frontend.

Runs after initial SSA construction to clean up, optimize, and
analyze the IR before refinement-type inference.  Every pass
operates on ``IRFunction`` / ``IRBasicBlock`` from ``src.ir.unified``
and the node types defined in ``src.ir.nodes``.
"""

from __future__ import annotations

import copy
import operator
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

from src.ir.nodes import (
    IRNode,
    AssignNode,
    PhiNode,
    GatedPhiNode,
    GuardNode,
    CallNode,
    ReturnNode,
    BranchNode,
    JumpNode,
    BinOpNode,
    UnaryOpNode,
    CompareNode,
    LoadAttrNode,
    StoreAttrNode,
    IndexNode,
    StoreIndexNode,
    LiteralNode,
    TruthinessNode,
    TypeNarrowNode,
    TypeTestNode,
    NullCheckNode,
    HasAttrNode,
    LenNode,
    ContainerCreateNode,
    UnpackNode,
    YieldNode,
    AwaitNode,
    RaiseNode,
    ExceptHandlerNode,
    ImportNode,
    DeleteNode,
    AssertNode,
    SliceNode,
    FormatStringNode,
    ClosureCaptureNode,
    SSAVar,
    SourceLocation,
    TypeAnnotation,
    BinOp,
    UnaryOp,
    CompareOp,
    GuardKind,
    LiteralKind,
    IRVisitor,
)
from src.ir.unified import (
    IRModule,
    IRFunction,
    IRBasicBlock,
    CFG,
    SSAValue,
    DominatorTree,
)

# ═══════════════════════════════════════════════════════════════════════════
# Utility classes
# ═══════════════════════════════════════════════════════════════════════════


class BitVector:
    """Efficient fixed-size bit vector for dataflow sets.

    Supports union, intersection, difference, equality, and iteration
    over set bits.
    """

    __slots__ = ("_size", "_bits")

    def __init__(self, size: int) -> None:
        self._size = size
        self._bits = 0

    # -- construction helpers -----------------------------------------------

    @classmethod
    def all_ones(cls, size: int) -> "BitVector":
        bv = cls(size)
        bv._bits = (1 << size) - 1
        return bv

    def copy(self) -> "BitVector":
        bv = BitVector(self._size)
        bv._bits = self._bits
        return bv

    # -- element access -----------------------------------------------------

    def __len__(self) -> int:
        return self._size

    def __contains__(self, idx: int) -> bool:
        return bool(self._bits & (1 << idx))

    def set(self, idx: int) -> None:
        self._bits |= 1 << idx

    def clear(self, idx: int) -> None:
        self._bits &= ~(1 << idx)

    def toggle(self, idx: int) -> None:
        self._bits ^= 1 << idx

    # -- bulk operations ----------------------------------------------------

    def union(self, other: "BitVector") -> "BitVector":
        bv = BitVector(max(self._size, other._size))
        bv._bits = self._bits | other._bits
        return bv

    def intersection(self, other: "BitVector") -> "BitVector":
        bv = BitVector(max(self._size, other._size))
        bv._bits = self._bits & other._bits
        return bv

    def difference(self, other: "BitVector") -> "BitVector":
        bv = BitVector(self._size)
        bv._bits = self._bits & ~other._bits
        return bv

    def union_inplace(self, other: "BitVector") -> bool:
        old = self._bits
        self._bits |= other._bits
        return self._bits != old

    def intersect_inplace(self, other: "BitVector") -> bool:
        old = self._bits
        self._bits &= other._bits
        return self._bits != old

    # -- queries ------------------------------------------------------------

    def is_empty(self) -> bool:
        return self._bits == 0

    def popcount(self) -> int:
        return bin(self._bits).count("1")

    def iter_set_bits(self) -> Iterator[int]:
        b = self._bits
        idx = 0
        while b:
            if b & 1:
                yield idx
            b >>= 1
            idx += 1

    # -- comparison ---------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BitVector):
            return NotImplemented
        return self._bits == other._bits

    def __hash__(self) -> int:
        return hash(self._bits)

    def __repr__(self) -> str:
        return f"BitVector({bin(self._bits)})"


class WorkList:
    """Efficient worklist (FIFO with set membership test).

    Items are added via ``add`` and consumed via ``pop``.  Re-adding an
    item that is already on the list is a no-op.
    """

    __slots__ = ("_queue", "_in_queue")

    def __init__(self, initial: Iterable[str] = ()) -> None:
        self._queue: Deque[str] = deque()
        self._in_queue: Set[str] = set()
        for item in initial:
            self.add(item)

    def add(self, item: str) -> None:
        if item not in self._in_queue:
            self._in_queue.add(item)
            self._queue.append(item)

    def pop(self) -> str:
        item = self._queue.popleft()
        self._in_queue.discard(item)
        return item

    def __bool__(self) -> bool:
        return bool(self._queue)

    def __len__(self) -> int:
        return len(self._queue)

    def __contains__(self, item: str) -> bool:
        return item in self._in_queue

    def extend(self, items: Iterable[str]) -> None:
        for item in items:
            self.add(item)


# ═══════════════════════════════════════════════════════════════════════════
# Pass statistics
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PassStatistics:
    """Collects statistics about what each pass did."""

    pass_name: str = ""
    nodes_removed: int = 0
    nodes_added: int = 0
    nodes_modified: int = 0
    blocks_removed: int = 0
    blocks_added: int = 0
    blocks_merged: int = 0
    edges_split: int = 0
    phis_removed: int = 0
    phis_simplified: int = 0
    copies_propagated: int = 0
    constants_folded: int = 0
    cse_hits: int = 0
    vars_eliminated: int = 0
    iterations: int = 0
    changed: bool = False

    def merge(self, other: "PassStatistics") -> None:
        self.nodes_removed += other.nodes_removed
        self.nodes_added += other.nodes_added
        self.nodes_modified += other.nodes_modified
        self.blocks_removed += other.blocks_removed
        self.blocks_added += other.blocks_added
        self.blocks_merged += other.blocks_merged
        self.edges_split += other.edges_split
        self.phis_removed += other.phis_removed
        self.phis_simplified += other.phis_simplified
        self.copies_propagated += other.copies_propagated
        self.constants_folded += other.constants_folded
        self.cse_hits += other.cse_hits
        self.vars_eliminated += other.vars_eliminated
        self.iterations += other.iterations
        self.changed = self.changed or other.changed

    def summary(self) -> str:
        parts: List[str] = []
        if self.nodes_removed:
            parts.append(f"nodes_rm={self.nodes_removed}")
        if self.nodes_added:
            parts.append(f"nodes_add={self.nodes_added}")
        if self.nodes_modified:
            parts.append(f"nodes_mod={self.nodes_modified}")
        if self.blocks_removed:
            parts.append(f"blk_rm={self.blocks_removed}")
        if self.blocks_merged:
            parts.append(f"blk_mrg={self.blocks_merged}")
        if self.edges_split:
            parts.append(f"edge_split={self.edges_split}")
        if self.phis_removed:
            parts.append(f"phi_rm={self.phis_removed}")
        if self.phis_simplified:
            parts.append(f"phi_simp={self.phis_simplified}")
        if self.copies_propagated:
            parts.append(f"copy_prop={self.copies_propagated}")
        if self.constants_folded:
            parts.append(f"const_fold={self.constants_folded}")
        if self.cse_hits:
            parts.append(f"cse={self.cse_hits}")
        if self.vars_eliminated:
            parts.append(f"var_elim={self.vars_eliminated}")
        if self.iterations > 1:
            parts.append(f"iters={self.iterations}")
        return f"[{self.pass_name}] " + (", ".join(parts) if parts else "no changes")


# ═══════════════════════════════════════════════════════════════════════════
# SSA verifier
# ═══════════════════════════════════════════════════════════════════════════


class SSAVerifier:
    """Verifies SSA properties are maintained after passes.

    Checks:
      1. Every use is dominated by exactly one definition.
      2. Phi nodes only appear at block entry.
      3. No duplicate definitions of the same SSAVar.
      4. Terminators (branch / jump / return) appear only at block end.
      5. All branch/jump targets exist.
      6. Predecessor/successor lists are consistent.
    """

    def __init__(self) -> None:
        self.errors: List[str] = []

    def verify(self, func: IRFunction) -> bool:
        self.errors.clear()
        self._verify_block_structure(func)
        self._verify_single_def(func)
        self._verify_phi_placement(func)
        self._verify_terminators(func)
        self._verify_cfg_edges(func)
        self._verify_dominance(func)
        return len(self.errors) == 0

    def _verify_block_structure(self, func: IRFunction) -> None:
        if not func.blocks:
            self.errors.append("Function has no blocks")
            return
        if func.entry_block not in func.blocks:
            self.errors.append(
                f"Entry block {func.entry_block!r} not in function blocks"
            )

    def _verify_single_def(self, func: IRFunction) -> None:
        defs: Dict[SSAVar, str] = {}
        for blk_label, blk in func.blocks.items():
            for node in blk.nodes:
                for v in node.defined_vars:
                    if v in defs:
                        self.errors.append(
                            f"SSAVar {v} defined in both {defs[v]!r} and {blk_label!r}"
                        )
                    defs[v] = blk_label

    def _verify_phi_placement(self, func: IRFunction) -> None:
        for blk_label, blk in func.blocks.items():
            seen_non_phi = False
            for node in blk.nodes:
                if isinstance(node, (PhiNode, GatedPhiNode)):
                    if seen_non_phi:
                        self.errors.append(
                            f"Phi node after non-phi in block {blk_label!r}"
                        )
                else:
                    seen_non_phi = True

    def _verify_terminators(self, func: IRFunction) -> None:
        for blk_label, blk in func.blocks.items():
            if not blk.nodes:
                self.errors.append(f"Block {blk_label!r} is empty")
                continue
            term = blk.nodes[-1]
            if not isinstance(term, (BranchNode, JumpNode, ReturnNode, RaiseNode)):
                self.errors.append(
                    f"Block {blk_label!r} does not end with a terminator "
                    f"(ends with {type(term).__name__})"
                )
            for node in blk.nodes[:-1]:
                if isinstance(node, (BranchNode, JumpNode, ReturnNode)):
                    self.errors.append(
                        f"Terminator {type(node).__name__} in middle of "
                        f"block {blk_label!r}"
                    )

    def _verify_cfg_edges(self, func: IRFunction) -> None:
        for blk_label, blk in func.blocks.items():
            for succ in blk.successors:
                if succ not in func.blocks:
                    self.errors.append(
                        f"Block {blk_label!r} has successor {succ!r} "
                        f"which does not exist"
                    )
                elif blk_label not in func.blocks[succ].predecessors:
                    self.errors.append(
                        f"Block {blk_label!r} lists {succ!r} as successor "
                        f"but is not in its predecessors"
                    )
            for pred in blk.predecessors:
                if pred not in func.blocks:
                    self.errors.append(
                        f"Block {blk_label!r} has predecessor {pred!r} "
                        f"which does not exist"
                    )

    def _verify_dominance(self, func: IRFunction) -> None:
        if not hasattr(func, "dominator_tree") or func.dominator_tree is None:
            return
        dom = func.dominator_tree
        defs: Dict[SSAVar, str] = {}
        for blk_label, blk in func.blocks.items():
            for node in blk.nodes:
                for v in node.defined_vars:
                    defs[v] = blk_label

        for blk_label, blk in func.blocks.items():
            for node in blk.nodes:
                if isinstance(node, (PhiNode, GatedPhiNode)):
                    continue
                for v in node.used_vars:
                    if v in defs:
                        def_block = defs[v]
                        if not dom.dominates(def_block, blk_label):
                            self.errors.append(
                                f"Use of {v} in {blk_label!r} not dominated "
                                f"by def in {def_block!r}"
                            )


# ═══════════════════════════════════════════════════════════════════════════
# Pass base class and manager
# ═══════════════════════════════════════════════════════════════════════════


class PassKind(Enum):
    OPTIMIZATION = auto()
    ANALYSIS = auto()
    CLEANUP = auto()


@dataclass
class Pass(ABC):
    """Base class for all SSA passes.

    Subclass and implement :meth:`run`.  Set ``changed`` if the IR
    was mutated.
    """

    name: str = ""
    description: str = ""
    kind: PassKind = PassKind.OPTIMIZATION
    changed: bool = field(default=False, init=False)
    stats: PassStatistics = field(default_factory=PassStatistics, init=False)
    _dependencies: List[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = type(self).__name__
        self.stats.pass_name = self.name

    def requires(self, *pass_names: str) -> None:
        self._dependencies.extend(pass_names)

    @abstractmethod
    def run(self, func: IRFunction) -> IRFunction:
        """Execute the pass on *func*, returning the (possibly modified)
        function.  Set ``self.changed`` if anything was mutated."""
        ...

    def reset(self) -> None:
        self.changed = False
        self.stats = PassStatistics(pass_name=self.name)


class PassManager:
    """Manages pass ordering, dependencies, and fixed-point iteration.

    Passes are registered via :meth:`add`.  :meth:`run` executes them
    in registration order (respecting declared dependencies) and
    optionally iterates until a fixed point.
    """

    def __init__(self, *, max_iterations: int = 20, verify: bool = False) -> None:
        self._passes: List[Pass] = []
        self._pass_map: Dict[str, Pass] = {}
        self._max_iterations = max_iterations
        self._verify = verify
        self._verifier = SSAVerifier()
        self.all_stats: List[PassStatistics] = []

    def add(self, p: Pass) -> "PassManager":
        self._passes.append(p)
        self._pass_map[p.name] = p
        return self

    def _topo_order(self) -> List[Pass]:
        """Topological sort of passes by dependency."""
        visited: Set[str] = set()
        order: List[Pass] = []

        def visit(p: Pass) -> None:
            if p.name in visited:
                return
            visited.add(p.name)
            for dep_name in p._dependencies:
                dep = self._pass_map.get(dep_name)
                if dep is not None:
                    visit(dep)
            order.append(p)

        for p in self._passes:
            visit(p)
        return order

    def run(
        self,
        func: IRFunction,
        *,
        fixed_point: bool = False,
    ) -> IRFunction:
        """Run all registered passes on *func*.

        If *fixed_point* is True, repeats until no pass reports a change
        or *max_iterations* is reached.
        """
        ordered = self._topo_order()
        iteration = 0
        while True:
            iteration += 1
            any_changed = False
            for p in ordered:
                p.reset()
                func = p.run(func)
                p.stats.changed = p.changed
                self.all_stats.append(copy.copy(p.stats))
                any_changed = any_changed or p.changed
                if self._verify and p.changed:
                    ok = self._verifier.verify(func)
                    if not ok:
                        raise RuntimeError(
                            f"SSA verification failed after {p.name}: "
                            + "; ".join(self._verifier.errors)
                        )
            if not fixed_point or not any_changed:
                break
            if iteration >= self._max_iterations:
                break
        return func


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions used by multiple passes
# ═══════════════════════════════════════════════════════════════════════════


def _successor_labels(node: IRNode) -> List[str]:
    """Return the block labels this terminator branches to."""
    if isinstance(node, BranchNode):
        return [node.true_block, node.false_block]
    if isinstance(node, JumpNode):
        return [node.target]
    if isinstance(node, GuardNode):
        targets: List[str] = []
        if node.true_target:
            targets.append(node.true_target)
        if node.false_target:
            targets.append(node.false_target)
        return targets
    return []


def _is_terminator(node: IRNode) -> bool:
    return isinstance(node, (BranchNode, JumpNode, ReturnNode, RaiseNode))


def _collect_all_defs(func: IRFunction) -> Dict[SSAVar, Tuple[str, int]]:
    """Map each SSAVar to (block_label, node_index)."""
    defs: Dict[SSAVar, Tuple[str, int]] = {}
    for blk_label, blk in func.blocks.items():
        for idx, node in enumerate(blk.nodes):
            for v in node.defined_vars:
                defs[v] = (blk_label, idx)
    return defs


def _collect_all_uses(func: IRFunction) -> Dict[SSAVar, List[Tuple[str, int]]]:
    """Map each SSAVar to list of (block_label, node_index) where it is used."""
    uses: Dict[SSAVar, List[Tuple[str, int]]] = defaultdict(list)
    for blk_label, blk in func.blocks.items():
        for idx, node in enumerate(blk.nodes):
            for v in node.used_vars:
                uses[v].append((blk_label, idx))
    return uses


def _compute_reachable(func: IRFunction) -> Set[str]:
    """BFS from entry block, returning set of reachable block labels."""
    reachable: Set[str] = set()
    wl: Deque[str] = deque([func.entry_block])
    while wl:
        blk_label = wl.popleft()
        if blk_label in reachable:
            continue
        reachable.add(blk_label)
        blk = func.blocks.get(blk_label)
        if blk is not None:
            wl.extend(blk.successors)
    return reachable


def _rebuild_predecessors(func: IRFunction) -> None:
    """Recompute predecessor lists from successor lists."""
    for blk in func.blocks.values():
        blk.predecessors = []
    for blk_label, blk in func.blocks.items():
        for succ in blk.successors:
            if succ in func.blocks:
                func.blocks[succ].predecessors.append(blk_label)


def _rebuild_successors(func: IRFunction) -> None:
    """Recompute successor lists from terminators."""
    for blk_label, blk in func.blocks.items():
        if blk.nodes:
            blk.successors = _successor_labels(blk.nodes[-1])
        else:
            blk.successors = []


def _rebuild_cfg(func: IRFunction) -> None:
    """Recompute both successor and predecessor lists."""
    _rebuild_successors(func)
    _rebuild_predecessors(func)


# ═══════════════════════════════════════════════════════════════════════════
# Optimization passes
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Dead code elimination
# ---------------------------------------------------------------------------


class DeadCodeElimination(Pass):
    """Removes unreachable blocks and unused SSA variables.

    1. Eliminate blocks not reachable from the entry block.
    2. Iteratively remove instructions whose defined variables have
       no uses (unless they have side effects).
    """

    def __init__(self) -> None:
        super().__init__(
            name="DeadCodeElimination",
            description="Remove unreachable blocks and dead instructions",
            kind=PassKind.OPTIMIZATION,
        )

    def run(self, func: IRFunction) -> IRFunction:
        removed_blocks = self._remove_unreachable(func)
        removed_instrs = self._remove_dead_instrs(func)
        self.changed = removed_blocks > 0 or removed_instrs > 0
        self.stats.blocks_removed = removed_blocks
        self.stats.nodes_removed = removed_instrs
        self.stats.vars_eliminated = removed_instrs
        return func

    def _remove_unreachable(self, func: IRFunction) -> int:
        reachable = _compute_reachable(func)
        dead = [l for l in func.blocks if l not in reachable]
        for l in dead:
            del func.blocks[l]
        if dead:
            _rebuild_cfg(func)
        return len(dead)

    def _has_side_effects(self, node: IRNode) -> bool:
        return isinstance(
            node,
            (
                CallNode,
                StoreAttrNode,
                StoreIndexNode,
                ReturnNode,
                BranchNode,
                JumpNode,
                RaiseNode,
                YieldNode,
                AwaitNode,
                DeleteNode,
                AssertNode,
                ImportNode,
                GuardNode,
            ),
        )

    def _remove_dead_instrs(self, func: IRFunction) -> int:
        total_removed = 0
        changed = True
        while changed:
            changed = False
            uses = _collect_all_uses(func)
            for blk in func.blocks.values():
                new_nodes: List[IRNode] = []
                for node in blk.nodes:
                    defs = node.defined_vars
                    if defs and not self._has_side_effects(node):
                        if all(not uses.get(v) for v in defs):
                            total_removed += 1
                            changed = True
                            continue
                    new_nodes.append(node)
                blk.nodes = new_nodes
        return total_removed


# ---------------------------------------------------------------------------
# Copy propagation
# ---------------------------------------------------------------------------


class CopyPropagation(Pass):
    """Propagates simple copies (x = y) through uses.

    Finds ``AssignNode(dst=a, src=b)`` and replaces all uses of ``a``
    with ``b``, then removes the copy.
    """

    def __init__(self) -> None:
        super().__init__(
            name="CopyPropagation",
            description="Propagate simple copies through uses",
            kind=PassKind.OPTIMIZATION,
        )

    def run(self, func: IRFunction) -> IRFunction:
        copies: Dict[SSAVar, SSAVar] = {}
        for blk in func.blocks.values():
            for node in blk.nodes:
                if isinstance(node, AssignNode):
                    copies[node.dst] = node.src

        resolved = self._resolve_chains(copies)
        if not resolved:
            return func

        count = 0
        for blk in func.blocks.values():
            for node in blk.nodes:
                old_uses = node.used_vars
                node.replace_uses(resolved)
                if node.used_vars != old_uses:
                    count += 1

        self.changed = count > 0
        self.stats.copies_propagated = count
        return func

    @staticmethod
    def _resolve_chains(copies: Dict[SSAVar, SSAVar]) -> Dict[SSAVar, SSAVar]:
        """Follow copy chains:  a→b→c  ⇒  a→c, b→c."""
        resolved: Dict[SSAVar, SSAVar] = {}
        for src in copies:
            target = src
            visited: Set[SSAVar] = set()
            while target in copies and target not in visited:
                visited.add(target)
                target = copies[target]
            if target != src:
                resolved[src] = target
        return resolved


# ---------------------------------------------------------------------------
# Constant folding
# ---------------------------------------------------------------------------


_BINOP_EVAL: Dict[BinOp, Callable[[Any, Any], Any]] = {
    BinOp.ADD: operator.add,
    BinOp.SUB: operator.sub,
    BinOp.MUL: operator.mul,
    BinOp.DIV: operator.truediv,
    BinOp.FLOOR_DIV: operator.floordiv,
    BinOp.MOD: operator.mod,
    BinOp.POW: operator.pow,
    BinOp.LSHIFT: operator.lshift,
    BinOp.RSHIFT: operator.rshift,
    BinOp.BIT_AND: operator.and_,
    BinOp.BIT_OR: operator.or_,
    BinOp.BIT_XOR: operator.xor,
    BinOp.EQ: operator.eq,
    BinOp.NE: operator.ne,
    BinOp.LT: operator.lt,
    BinOp.LE: operator.le,
    BinOp.GT: operator.gt,
    BinOp.GE: operator.ge,
}

_UNARYOP_EVAL: Dict[UnaryOp, Callable[[Any], Any]] = {
    UnaryOp.NEGATE: operator.neg,
    UnaryOp.INVERT: operator.invert,
    UnaryOp.NOT: operator.not_,
    UnaryOp.POS: operator.pos,
}

_KIND_FOR_TYPE: Dict[type, LiteralKind] = {
    int: LiteralKind.INT,
    float: LiteralKind.FLOAT,
    str: LiteralKind.STR,
    bool: LiteralKind.BOOL,
}


class ConstantFolding(Pass):
    """Evaluates constant expressions at compile time.

    When all operands of a ``BinOpNode`` or ``UnaryOpNode`` are
    ``LiteralNode`` values, the operation is evaluated and replaced
    with a single ``LiteralNode``.
    """

    def __init__(self) -> None:
        super().__init__(
            name="ConstantFolding",
            description="Evaluate constant expressions at compile time",
            kind=PassKind.OPTIMIZATION,
        )

    def run(self, func: IRFunction) -> IRFunction:
        const_map = self._gather_constants(func)
        folded = 0

        for blk in func.blocks.values():
            new_nodes: List[IRNode] = []
            for node in blk.nodes:
                replacement = self._try_fold(node, const_map)
                if replacement is not None:
                    new_nodes.append(replacement)
                    for v in replacement.defined_vars:
                        if isinstance(replacement, LiteralNode):
                            const_map[v] = replacement.value
                    folded += 1
                else:
                    new_nodes.append(node)
            blk.nodes = new_nodes

        self.changed = folded > 0
        self.stats.constants_folded = folded
        return func

    @staticmethod
    def _gather_constants(func: IRFunction) -> Dict[SSAVar, Any]:
        consts: Dict[SSAVar, Any] = {}
        for blk in func.blocks.values():
            for node in blk.nodes:
                if isinstance(node, LiteralNode):
                    consts[node.dst] = node.value
        return consts

    def _try_fold(
        self, node: IRNode, consts: Dict[SSAVar, Any]
    ) -> Optional[LiteralNode]:
        if isinstance(node, BinOpNode):
            lv = consts.get(node.left)
            rv = consts.get(node.right)
            if lv is not None and rv is not None:
                fn = _BINOP_EVAL.get(node.op)
                if fn is not None:
                    try:
                        result = fn(lv, rv)
                    except Exception:
                        return None
                    kind = _KIND_FOR_TYPE.get(type(result), LiteralKind.INT)
                    return LiteralNode(
                        dst=node.dst,
                        kind=kind,
                        value=result,
                        source_loc=node.source_loc,
                    )

        if isinstance(node, UnaryOpNode):
            ov = consts.get(node.operand)
            if ov is not None:
                fn = _UNARYOP_EVAL.get(node.op)
                if fn is not None:
                    try:
                        result = fn(ov)
                    except Exception:
                        return None
                    kind = _KIND_FOR_TYPE.get(type(result), LiteralKind.INT)
                    return LiteralNode(
                        dst=node.dst,
                        kind=kind,
                        value=result,
                        source_loc=node.source_loc,
                    )

        return None


# ---------------------------------------------------------------------------
# Constant propagation
# ---------------------------------------------------------------------------


class ConstantPropagation(Pass):
    """Propagates constant values through the CFG.

    Uses a simple forward dataflow analysis: a lattice with ⊤ (unknown),
    constant values, and ⊥ (overdefined / non-constant).  Iterates to
    a fixed point.
    """

    _TOP = object()
    _BOT = object()

    def __init__(self) -> None:
        super().__init__(
            name="ConstantPropagation",
            description="Propagate constant values through CFG",
            kind=PassKind.OPTIMIZATION,
        )

    def run(self, func: IRFunction) -> IRFunction:
        lattice: Dict[SSAVar, Any] = {}
        self._initialize(func, lattice)
        self._propagate(func, lattice)
        replaced = self._rewrite(func, lattice)
        self.changed = replaced > 0
        self.stats.constants_folded = replaced
        return func

    def _initialize(self, func: IRFunction, lattice: Dict[SSAVar, Any]) -> None:
        for blk in func.blocks.values():
            for node in blk.nodes:
                for v in node.defined_vars:
                    if isinstance(node, LiteralNode):
                        lattice[v] = node.value
                    else:
                        lattice[v] = self._TOP

    def _meet(self, a: Any, b: Any) -> Any:
        if a is self._TOP:
            return b
        if b is self._TOP:
            return a
        if a is self._BOT or b is self._BOT:
            return self._BOT
        if a == b:
            return a
        return self._BOT

    def _propagate(self, func: IRFunction, lattice: Dict[SSAVar, Any]) -> None:
        wl = WorkList(func.blocks.keys())
        iterations = 0
        while wl:
            iterations += 1
            blk_label = wl.pop()
            blk = func.blocks[blk_label]
            for node in blk.nodes:
                if isinstance(node, PhiNode):
                    vals = [lattice.get(v, self._TOP) for v in node.incoming.values()]
                    new_val = self._TOP
                    for v in vals:
                        new_val = self._meet(new_val, v)
                    old = lattice.get(node.dst, self._TOP)
                    result = self._meet(old, new_val)
                    if result is not old:
                        lattice[node.dst] = result
                        wl.extend(blk.successors)
                elif isinstance(node, AssignNode):
                    src_val = lattice.get(node.src, self._TOP)
                    old = lattice.get(node.dst, self._TOP)
                    result = self._meet(old, src_val)
                    if result is not old:
                        lattice[node.dst] = result
                        wl.extend(blk.successors)
                elif isinstance(node, BinOpNode):
                    lv = lattice.get(node.left, self._TOP)
                    rv = lattice.get(node.right, self._TOP)
                    if (
                        lv is not self._TOP
                        and lv is not self._BOT
                        and rv is not self._TOP
                        and rv is not self._BOT
                    ):
                        fn = _BINOP_EVAL.get(node.op)
                        if fn is not None:
                            try:
                                val = fn(lv, rv)
                            except Exception:
                                val = self._BOT
                        else:
                            val = self._BOT
                    elif lv is self._BOT or rv is self._BOT:
                        val = self._BOT
                    else:
                        val = self._TOP
                    old = lattice.get(node.dst, self._TOP)
                    result = self._meet(old, val)
                    if result is not old:
                        lattice[node.dst] = result
                        wl.extend(blk.successors)

        self.stats.iterations = iterations

    def _rewrite(self, func: IRFunction, lattice: Dict[SSAVar, Any]) -> int:
        count = 0
        for blk in func.blocks.values():
            new_nodes: List[IRNode] = []
            for node in blk.nodes:
                if isinstance(node, (AssignNode, BinOpNode, UnaryOpNode)):
                    defs = list(node.defined_vars)
                    if len(defs) == 1:
                        val = lattice.get(defs[0], self._TOP)
                        if val is not self._TOP and val is not self._BOT:
                            kind = _KIND_FOR_TYPE.get(type(val), LiteralKind.INT)
                            lit = LiteralNode(
                                dst=defs[0],
                                kind=kind,
                                value=val,
                                source_loc=node.source_loc,
                            )
                            new_nodes.append(lit)
                            count += 1
                            continue
                new_nodes.append(node)
            blk.nodes = new_nodes
        return count


# ---------------------------------------------------------------------------
# Phi simplification
# ---------------------------------------------------------------------------


class PhiSimplification(Pass):
    """Removes trivial phi nodes.

    A phi is trivial if:
      - All incoming values are the same (after ignoring self-references).
      - It has exactly one incoming value.

    Trivial phis are replaced by their single value and all uses are
    rewritten.
    """

    def __init__(self) -> None:
        super().__init__(
            name="PhiSimplification",
            description="Remove trivial phi nodes",
            kind=PassKind.OPTIMIZATION,
        )

    def run(self, func: IRFunction) -> IRFunction:
        mapping: Dict[SSAVar, SSAVar] = {}
        removed = 0

        for blk in func.blocks.values():
            new_nodes: List[IRNode] = []
            for node in blk.nodes:
                if isinstance(node, PhiNode):
                    replacement = self._trivial_value(node)
                    if replacement is not None:
                        mapping[node.dst] = replacement
                        removed += 1
                        continue
                new_nodes.append(node)
            blk.nodes = new_nodes

        if mapping:
            resolved = CopyPropagation._resolve_chains(mapping)
            for blk in func.blocks.values():
                for node in blk.nodes:
                    node.replace_uses(resolved)

        self.changed = removed > 0
        self.stats.phis_simplified = removed
        return func

    @staticmethod
    def _trivial_value(phi: PhiNode) -> Optional[SSAVar]:
        """Return the single non-self incoming value, or None."""
        unique: Optional[SSAVar] = None
        for v in phi.incoming.values():
            if v == phi.dst:
                continue
            if unique is None:
                unique = v
            elif v != unique:
                return None
        return unique


# ---------------------------------------------------------------------------
# Phi elimination
# ---------------------------------------------------------------------------


class PhiElimination(Pass):
    """Removes phi nodes that are dead after other optimizations.

    A phi whose defined variable has zero uses (and is not the
    entry-block's result) is removed.
    """

    def __init__(self) -> None:
        super().__init__(
            name="PhiElimination",
            description="Remove dead phi nodes",
            kind=PassKind.OPTIMIZATION,
        )

    def run(self, func: IRFunction) -> IRFunction:
        uses = _collect_all_uses(func)
        removed = 0
        for blk in func.blocks.values():
            new_nodes: List[IRNode] = []
            for node in blk.nodes:
                if isinstance(node, (PhiNode, GatedPhiNode)):
                    if not uses.get(node.dst):
                        removed += 1
                        continue
                new_nodes.append(node)
            blk.nodes = new_nodes

        self.changed = removed > 0
        self.stats.phis_removed = removed
        return func


# ---------------------------------------------------------------------------
# Common subexpression elimination
# ---------------------------------------------------------------------------


class CommonSubexpressionElimination(Pass):
    """Identifies and reuses common subexpressions within a block.

    Two nodes are considered equivalent if they have the same type,
    same operator (if applicable), and same used SSA variables.
    The second occurrence is replaced with a copy from the first.
    """

    def __init__(self) -> None:
        super().__init__(
            name="CommonSubexpressionElimination",
            description="Reuse common subexpressions",
            kind=PassKind.OPTIMIZATION,
        )

    def run(self, func: IRFunction) -> IRFunction:
        total = 0
        for blk in func.blocks.values():
            total += self._cse_block(blk)
        self.changed = total > 0
        self.stats.cse_hits = total
        return func

    def _cse_block(self, blk: IRBasicBlock) -> int:
        seen: Dict[Tuple[Any, ...], SSAVar] = {}
        count = 0
        new_nodes: List[IRNode] = []
        for node in blk.nodes:
            if self._has_side_effects(node):
                new_nodes.append(node)
                continue
            key = self._expr_key(node)
            if key is None:
                new_nodes.append(node)
                continue
            defs = list(node.defined_vars)
            if len(defs) != 1:
                new_nodes.append(node)
                continue
            if key in seen:
                new_nodes.append(
                    AssignNode(
                        dst=defs[0],
                        src=seen[key],
                        source_loc=node.source_loc,
                    )
                )
                count += 1
            else:
                seen[key] = defs[0]
                new_nodes.append(node)
        blk.nodes = new_nodes
        return count

    @staticmethod
    def _expr_key(node: IRNode) -> Optional[Tuple[Any, ...]]:
        if isinstance(node, BinOpNode):
            return ("binop", node.op, node.left, node.right)
        if isinstance(node, UnaryOpNode):
            return ("unop", node.op, node.operand)
        if isinstance(node, CompareNode):
            return ("cmp", node.op, node.left, node.right)
        if isinstance(node, LoadAttrNode):
            return ("loadattr", node.obj, node.attr)
        if isinstance(node, IndexNode):
            return ("index", node.obj, node.index)
        if isinstance(node, LiteralNode):
            return ("lit", node.kind, node.value)
        if isinstance(node, TruthinessNode):
            return ("bool", node.operand)
        return None

    @staticmethod
    def _has_side_effects(node: IRNode) -> bool:
        return isinstance(
            node,
            (
                CallNode, StoreAttrNode, StoreIndexNode, ReturnNode,
                BranchNode, JumpNode, RaiseNode, YieldNode, AwaitNode,
                DeleteNode, AssertNode, ImportNode, GuardNode,
            ),
        )


# ---------------------------------------------------------------------------
# Unreachable block elimination
# ---------------------------------------------------------------------------


class UnreachableBlockElimination(Pass):
    """Removes blocks with no predecessors (except the entry block).

    Also patches up phi nodes in successors whose incoming edges
    reference the removed block.
    """

    def __init__(self) -> None:
        super().__init__(
            name="UnreachableBlockElimination",
            description="Remove blocks with no predecessors",
            kind=PassKind.OPTIMIZATION,
        )

    def run(self, func: IRFunction) -> IRFunction:
        removed = 0
        changed_iter = True
        while changed_iter:
            changed_iter = False
            dead = [
                l
                for l, blk in func.blocks.items()
                if l != func.entry_block and not blk.predecessors
            ]
            for l in dead:
                self._patch_successor_phis(func, l)
                del func.blocks[l]
                changed_iter = True
                removed += 1
            if changed_iter:
                _rebuild_cfg(func)

        self.changed = removed > 0
        self.stats.blocks_removed = removed
        return func

    @staticmethod
    def _patch_successor_phis(func: IRFunction, dead_label: str) -> None:
        blk = func.blocks.get(dead_label)
        if blk is None:
            return
        for succ_label in blk.successors:
            succ = func.blocks.get(succ_label)
            if succ is None:
                continue
            for node in succ.nodes:
                if isinstance(node, PhiNode) and dead_label in node.incoming:
                    del node.incoming[dead_label]


# ---------------------------------------------------------------------------
# Block merging
# ---------------------------------------------------------------------------


class BlockMerging(Pass):
    """Merges single-predecessor / single-successor block chains.

    If block A has exactly one successor B and B has exactly one
    predecessor A, then B's instructions are appended to A (minus
    the jump) and B is removed.
    """

    def __init__(self) -> None:
        super().__init__(
            name="BlockMerging",
            description="Merge single-pred/single-succ block chains",
            kind=PassKind.CLEANUP,
        )

    def run(self, func: IRFunction) -> IRFunction:
        merged = 0
        changed_iter = True
        while changed_iter:
            changed_iter = False
            for blk_label in list(func.blocks.keys()):
                blk = func.blocks.get(blk_label)
                if blk is None:
                    continue
                if len(blk.successors) != 1:
                    continue
                succ_label = blk.successors[0]
                if succ_label == blk_label:
                    continue
                succ = func.blocks.get(succ_label)
                if succ is None:
                    continue
                if len(succ.predecessors) != 1:
                    continue
                if succ.predecessors[0] != blk_label:
                    continue

                # Remove terminator from blk
                if blk.nodes and isinstance(blk.nodes[-1], (JumpNode, BranchNode)):
                    blk.nodes.pop()

                # Append succ's nodes (skip phis — they're trivial here)
                for node in succ.nodes:
                    if isinstance(node, PhiNode):
                        if node.incoming:
                            val = next(iter(node.incoming.values()))
                            blk.nodes.append(
                                AssignNode(
                                    dst=node.dst, src=val,
                                    source_loc=node.source_loc,
                                )
                            )
                    else:
                        blk.nodes.append(node)

                # Update successors' predecessor lists
                blk.successors = succ.successors
                for s in succ.successors:
                    sb = func.blocks.get(s)
                    if sb is not None:
                        sb.predecessors = [
                            blk_label if p == succ_label else p
                            for p in sb.predecessors
                        ]
                        # Patch phi incoming labels
                        for node in sb.nodes:
                            if isinstance(node, PhiNode) and succ_label in node.incoming:
                                node.incoming[blk_label] = node.incoming.pop(succ_label)

                del func.blocks[succ_label]
                merged += 1
                changed_iter = True
                break  # restart iteration after modifying dict

        self.changed = merged > 0
        self.stats.blocks_merged = merged
        return func


# ---------------------------------------------------------------------------
# Critical edge splitting
# ---------------------------------------------------------------------------


class CriticalEdgeSplitting(Pass):
    """Splits critical edges for correct phi semantics.

    A critical edge goes from a block with multiple successors to a
    block with multiple predecessors.  We insert a new empty block
    on the edge and update phi nodes.
    """

    def __init__(self) -> None:
        super().__init__(
            name="CriticalEdgeSplitting",
            description="Split critical edges",
            kind=PassKind.CLEANUP,
        )
        self._next_id = 0

    def run(self, func: IRFunction) -> IRFunction:
        self._next_id = 0
        splits = 0
        edges_to_split: List[Tuple[str, str]] = []

        for blk_label, blk in func.blocks.items():
            if len(blk.successors) <= 1:
                continue
            for succ_label in blk.successors:
                succ = func.blocks.get(succ_label)
                if succ is not None and len(succ.predecessors) > 1:
                    edges_to_split.append((blk_label, succ_label))

        for src, dst in edges_to_split:
            self._split_edge(func, src, dst)
            splits += 1

        if splits:
            _rebuild_cfg(func)

        self.changed = splits > 0
        self.stats.edges_split = splits
        return func

    def _split_edge(self, func: IRFunction, src: str, dst: str) -> str:
        new_label = f"__split_{src}_{dst}_{self._next_id}"
        self._next_id += 1

        new_blk = IRBasicBlock(label=new_label)
        new_blk.nodes = [JumpNode(target=dst)]
        new_blk.successors = [dst]
        new_blk.predecessors = [src]
        func.blocks[new_label] = new_blk

        # Patch src terminator
        src_blk = func.blocks[src]
        if src_blk.nodes:
            term = src_blk.nodes[-1]
            if isinstance(term, BranchNode):
                if term.true_block == dst:
                    term.true_block = new_label
                if term.false_block == dst:
                    term.false_block = new_label
            elif isinstance(term, JumpNode):
                if term.target == dst:
                    term.target = new_label
            elif isinstance(term, GuardNode):
                if term.true_target == dst:
                    term.true_target = new_label
                if term.false_target == dst:
                    term.false_target = new_label

        # Patch dst phi nodes
        dst_blk = func.blocks[dst]
        for node in dst_blk.nodes:
            if isinstance(node, PhiNode) and src in node.incoming:
                node.incoming[new_label] = node.incoming.pop(src)

        return new_label


# ═══════════════════════════════════════════════════════════════════════════
# Analysis passes
# ═══════════════════════════════════════════════════════════════════════════


# ---------------------------------------------------------------------------
# Liveness analysis
# ---------------------------------------------------------------------------


@dataclass
class LivenessResult:
    """Per-block live-in and live-out sets."""
    live_in: Dict[str, Set[SSAVar]] = field(default_factory=dict)
    live_out: Dict[str, Set[SSAVar]] = field(default_factory=dict)


class LivenessAnalysis(Pass):
    """Computes live-in / live-out sets for each block.

    Uses standard backward dataflow with iterative fixed-point.

      live_in[B]  = use[B] ∪ (live_out[B] − def[B])
      live_out[B] = ∪ live_in[S]  for S ∈ succ(B)
    """

    def __init__(self) -> None:
        super().__init__(
            name="LivenessAnalysis",
            description="Compute live-in/live-out sets",
            kind=PassKind.ANALYSIS,
        )
        self.result: LivenessResult = LivenessResult()

    def run(self, func: IRFunction) -> IRFunction:
        use_sets: Dict[str, Set[SSAVar]] = {}
        def_sets: Dict[str, Set[SSAVar]] = {}

        for blk_label, blk in func.blocks.items():
            u: Set[SSAVar] = set()
            d: Set[SSAVar] = set()
            for node in blk.nodes:
                for v in node.used_vars:
                    if v not in d:
                        u.add(v)
                d.update(node.defined_vars)
            use_sets[blk_label] = u
            def_sets[blk_label] = d

        live_in: Dict[str, Set[SSAVar]] = {l: set() for l in func.blocks}
        live_out: Dict[str, Set[SSAVar]] = {l: set() for l in func.blocks}

        wl = WorkList(func.blocks.keys())
        iterations = 0
        while wl:
            iterations += 1
            blk_label = wl.pop()
            blk = func.blocks[blk_label]

            new_out: Set[SSAVar] = set()
            for succ in blk.successors:
                if succ in live_in:
                    new_out |= live_in[succ]

            new_in = use_sets[blk_label] | (new_out - def_sets[blk_label])

            if new_in != live_in[blk_label] or new_out != live_out[blk_label]:
                live_in[blk_label] = new_in
                live_out[blk_label] = new_out
                wl.extend(blk.predecessors)
            else:
                live_out[blk_label] = new_out

        self.result = LivenessResult(live_in=live_in, live_out=live_out)
        self.stats.iterations = iterations
        return func


# ---------------------------------------------------------------------------
# Reaching definitions
# ---------------------------------------------------------------------------


@dataclass
class ReachingDefsResult:
    """Maps each (block, node_index) use to the set of reaching defs."""
    reaching: Dict[str, Set[Tuple[SSAVar, str]]] = field(default_factory=dict)


class ReachingDefinitions(Pass):
    """Computes reaching definitions for each block.

    Forward dataflow:
      reach_out[B] = gen[B] ∪ (reach_in[B] − kill[B])
      reach_in[B]  = ∪ reach_out[P]  for P ∈ pred(B)

    Each definition is represented as (SSAVar, defining_block_label).
    """

    def __init__(self) -> None:
        super().__init__(
            name="ReachingDefinitions",
            description="Compute reaching definitions",
            kind=PassKind.ANALYSIS,
        )
        self.result: ReachingDefsResult = ReachingDefsResult()

    def run(self, func: IRFunction) -> IRFunction:
        gen_sets: Dict[str, Set[Tuple[SSAVar, str]]] = {}
        kill_names: Dict[str, Set[str]] = {}

        for blk_label, blk in func.blocks.items():
            g: Set[Tuple[SSAVar, str]] = set()
            kn: Set[str] = set()
            for node in blk.nodes:
                for v in node.defined_vars:
                    kn.add(v.name)
                    g = {(dv, dl) for dv, dl in g if dv.name != v.name}
                    g.add((v, blk_label))
            gen_sets[blk_label] = g
            kill_names[blk_label] = kn

        reach_in: Dict[str, Set[Tuple[SSAVar, str]]] = {
            l: set() for l in func.blocks
        }
        reach_out: Dict[str, Set[Tuple[SSAVar, str]]] = {
            l: set() for l in func.blocks
        }

        wl = WorkList(func.blocks.keys())
        iterations = 0
        while wl:
            iterations += 1
            blk_label = wl.pop()
            blk = func.blocks[blk_label]

            new_in: Set[Tuple[SSAVar, str]] = set()
            for pred in blk.predecessors:
                if pred in reach_out:
                    new_in |= reach_out[pred]

            killed = kill_names[blk_label]
            surviving = {(v, l) for v, l in new_in if v.name not in killed}
            new_out = gen_sets[blk_label] | surviving

            if new_out != reach_out[blk_label]:
                reach_in[blk_label] = new_in
                reach_out[blk_label] = new_out
                wl.extend(blk.successors)
            else:
                reach_in[blk_label] = new_in

        self.result = ReachingDefsResult(reaching=reach_in)
        self.stats.iterations = iterations
        return func


# ---------------------------------------------------------------------------
# Def-use chains
# ---------------------------------------------------------------------------


@dataclass
class DefUseResult:
    """Def-use and use-def chains for all SSA variables."""
    # var -> list of (block, node_index) where it is used
    def_use: Dict[SSAVar, List[Tuple[str, int]]] = field(default_factory=dict)
    # var -> (block, node_index) where it is defined
    use_def: Dict[SSAVar, Tuple[str, int]] = field(default_factory=dict)


class DefUseChains(Pass):
    """Builds def-use and use-def chains for all SSA variables."""

    def __init__(self) -> None:
        super().__init__(
            name="DefUseChains",
            description="Build def-use and use-def chains",
            kind=PassKind.ANALYSIS,
        )
        self.result: DefUseResult = DefUseResult()

    def run(self, func: IRFunction) -> IRFunction:
        du: Dict[SSAVar, List[Tuple[str, int]]] = defaultdict(list)
        ud: Dict[SSAVar, Tuple[str, int]] = {}

        for blk_label, blk in func.blocks.items():
            for idx, node in enumerate(blk.nodes):
                for v in node.defined_vars:
                    ud[v] = (blk_label, idx)
                for v in node.used_vars:
                    du[v].append((blk_label, idx))

        self.result = DefUseResult(def_use=dict(du), use_def=ud)
        return func


# ---------------------------------------------------------------------------
# Dominator tree pass
# ---------------------------------------------------------------------------


@dataclass
class DominatorResult:
    """Dominator tree computed by the pass."""
    idom: Dict[str, Optional[str]] = field(default_factory=dict)
    dom_frontier: Dict[str, Set[str]] = field(default_factory=dict)
    dom_children: Dict[str, List[str]] = field(default_factory=dict)

    def dominates(self, a: str, b: str) -> bool:
        """Return True if *a* dominates *b*."""
        cur: Optional[str] = b
        while cur is not None:
            if cur == a:
                return True
            cur = self.idom.get(cur)
        return False

    def depth(self, block: str) -> int:
        d = 0
        cur: Optional[str] = block
        while cur is not None:
            cur = self.idom.get(cur)
            d += 1
        return d


class DominatorTreePass(Pass):
    """Computes dominator tree using the Cooper-Harvey-Kennedy algorithm.

    Also computes dominance frontiers and children lists.
    """

    def __init__(self) -> None:
        super().__init__(
            name="DominatorTreePass",
            description="Compute/validate dominator tree",
            kind=PassKind.ANALYSIS,
        )
        self.result: DominatorResult = DominatorResult()

    def run(self, func: IRFunction) -> IRFunction:
        rpo = self._reverse_postorder(func)
        idom = self._compute_idom(func, rpo)
        frontier = self._compute_frontier(func, idom)
        children: Dict[str, List[str]] = defaultdict(list)
        for blk, parent in idom.items():
            if parent is not None:
                children[parent].append(blk)

        self.result = DominatorResult(
            idom=idom,
            dom_frontier=frontier,
            dom_children=dict(children),
        )
        return func

    @staticmethod
    def _reverse_postorder(func: IRFunction) -> List[str]:
        visited: Set[str] = set()
        order: List[str] = []

        def dfs(label: str) -> None:
            if label in visited:
                return
            visited.add(label)
            blk = func.blocks.get(label)
            if blk is not None:
                for s in blk.successors:
                    dfs(s)
            order.append(label)

        dfs(func.entry_block)
        order.reverse()
        return order

    def _compute_idom(
        self, func: IRFunction, rpo: List[str]
    ) -> Dict[str, Optional[str]]:
        rpo_index = {label: i for i, label in enumerate(rpo)}
        idom: Dict[str, Optional[str]] = {rpo[0]: None}

        def intersect(b1: str, b2: str) -> str:
            f1, f2 = b1, b2
            while f1 != f2:
                while rpo_index.get(f1, len(rpo)) > rpo_index.get(f2, len(rpo)):
                    f1 = idom.get(f1, rpo[0])  # type: ignore[assignment]
                while rpo_index.get(f2, len(rpo)) > rpo_index.get(f1, len(rpo)):
                    f2 = idom.get(f2, rpo[0])  # type: ignore[assignment]
            return f1

        changed = True
        while changed:
            changed = False
            for blk_label in rpo[1:]:
                blk = func.blocks.get(blk_label)
                if blk is None:
                    continue
                processed_preds = [p for p in blk.predecessors if p in idom]
                if not processed_preds:
                    continue
                new_idom = processed_preds[0]
                for pred in processed_preds[1:]:
                    new_idom = intersect(new_idom, pred)
                if idom.get(blk_label) != new_idom:
                    idom[blk_label] = new_idom
                    changed = True

        return idom

    @staticmethod
    def _compute_frontier(
        func: IRFunction, idom: Dict[str, Optional[str]]
    ) -> Dict[str, Set[str]]:
        frontier: Dict[str, Set[str]] = {l: set() for l in func.blocks}
        for blk_label, blk in func.blocks.items():
            if len(blk.predecessors) < 2:
                continue
            for pred in blk.predecessors:
                runner: Optional[str] = pred
                while runner is not None and runner != idom.get(blk_label):
                    frontier[runner].add(blk_label)
                    runner = idom.get(runner)
        return frontier


# ---------------------------------------------------------------------------
# Loop detection
# ---------------------------------------------------------------------------


@dataclass
class LoopInfo:
    """Description of a natural loop."""
    header: str
    back_edge_src: str
    body: Set[str]
    exits: Set[str] = field(default_factory=set)
    nesting_depth: int = 0
    parent: Optional[str] = None  # header of enclosing loop


@dataclass
class LoopDetectionResult:
    """All loops detected in a function."""
    loops: List[LoopInfo] = field(default_factory=list)
    loop_of_block: Dict[str, str] = field(default_factory=dict)
    back_edges: List[Tuple[str, str]] = field(default_factory=list)


class LoopDetection(Pass):
    """Detects natural loops, loop headers, back edges, nesting depth.

    A back edge (A → B) exists when B dominates A.  The natural loop
    for that back edge is the set of blocks from which A can be
    reached without going through B, plus B itself.
    """

    def __init__(self) -> None:
        super().__init__(
            name="LoopDetection",
            description="Detect natural loops and nesting",
            kind=PassKind.ANALYSIS,
        )
        self.requires("DominatorTreePass")
        self.result: LoopDetectionResult = LoopDetectionResult()
        self._dom: Optional[DominatorResult] = None

    def run(self, func: IRFunction) -> IRFunction:
        dom_pass = DominatorTreePass()
        dom_pass.run(func)
        self._dom = dom_pass.result

        back_edges = self._find_back_edges(func)
        loops: List[LoopInfo] = []
        for src, header in back_edges:
            body = self._compute_loop_body(func, header, src)
            exits: Set[str] = set()
            for blk_label in body:
                blk = func.blocks.get(blk_label)
                if blk is not None:
                    for s in blk.successors:
                        if s not in body:
                            exits.add(s)
            loops.append(LoopInfo(
                header=header,
                back_edge_src=src,
                body=body,
                exits=exits,
            ))

        self._compute_nesting(loops)

        loop_of_block: Dict[str, str] = {}
        for loop in sorted(loops, key=lambda l: len(l.body)):
            for blk in loop.body:
                loop_of_block[blk] = loop.header

        self.result = LoopDetectionResult(
            loops=loops,
            loop_of_block=loop_of_block,
            back_edges=back_edges,
        )
        return func

    def _find_back_edges(self, func: IRFunction) -> List[Tuple[str, str]]:
        assert self._dom is not None
        edges: List[Tuple[str, str]] = []
        for blk_label, blk in func.blocks.items():
            for succ in blk.successors:
                if self._dom.dominates(succ, blk_label):
                    edges.append((blk_label, succ))
        return edges

    @staticmethod
    def _compute_loop_body(
        func: IRFunction, header: str, back_edge_src: str
    ) -> Set[str]:
        body: Set[str] = {header}
        stack: List[str] = []
        if back_edge_src != header:
            body.add(back_edge_src)
            stack.append(back_edge_src)
        while stack:
            n = stack.pop()
            blk = func.blocks.get(n)
            if blk is None:
                continue
            for pred in blk.predecessors:
                if pred not in body:
                    body.add(pred)
                    stack.append(pred)
        return body

    @staticmethod
    def _compute_nesting(loops: List[LoopInfo]) -> None:
        loops_sorted = sorted(loops, key=lambda l: len(l.body), reverse=True)
        for i, outer in enumerate(loops_sorted):
            for inner in loops_sorted[i + 1:]:
                if inner.body < outer.body and inner.parent is None:
                    inner.parent = outer.header
                    inner.nesting_depth = outer.nesting_depth + 1


# ---------------------------------------------------------------------------
# Type inference pass
# ---------------------------------------------------------------------------


@dataclass
class TypeInfo:
    """Inferred type for an SSA variable."""
    var: SSAVar = field(default_factory=lambda: SSAVar("_"))
    type_name: str = "unknown"
    nullable: bool = True
    refined_by: Optional[str] = None  # guard that refined this

    def is_known(self) -> bool:
        return self.type_name != "unknown"


@dataclass
class TypeInferenceResult:
    """Maps SSA variables to their inferred types."""
    types: Dict[SSAVar, TypeInfo] = field(default_factory=dict)

    def get_type(self, var: SSAVar) -> str:
        info = self.types.get(var)
        return info.type_name if info else "unknown"


class TypeInferencePass(Pass):
    """Basic type inference using SSA form.

    Propagates known types forward through assignments, phi nodes,
    and arithmetic operations.  Uses a worklist algorithm.

    Type lattice:
      unknown (⊤) → concrete type → overdefined (⊥)
    """

    _ARITH_OPS = {BinOp.ADD, BinOp.SUB, BinOp.MUL, BinOp.POW}
    _INT_OPS = {
        BinOp.FLOOR_DIV, BinOp.MOD, BinOp.LSHIFT, BinOp.RSHIFT,
        BinOp.BIT_AND, BinOp.BIT_OR, BinOp.BIT_XOR,
    }
    _CMP_OPS = {
        BinOp.EQ, BinOp.NE, BinOp.LT, BinOp.LE, BinOp.GT, BinOp.GE,
        BinOp.IS, BinOp.IS_NOT, BinOp.IN, BinOp.NOT_IN,
    }

    _LITERAL_TYPE: Dict[LiteralKind, str] = {
        LiteralKind.INT: "int",
        LiteralKind.FLOAT: "float",
        LiteralKind.STR: "str",
        LiteralKind.BYTES: "bytes",
        LiteralKind.BOOL: "bool",
        LiteralKind.NONE: "None",
        LiteralKind.COMPLEX: "complex",
    }

    def __init__(self) -> None:
        super().__init__(
            name="TypeInferencePass",
            description="Propagate known types in SSA form",
            kind=PassKind.ANALYSIS,
        )
        self.result: TypeInferenceResult = TypeInferenceResult()

    def run(self, func: IRFunction) -> IRFunction:
        types: Dict[SSAVar, TypeInfo] = {}

        # Seed from literals
        for blk_label, blk in func.blocks.items():
            for node in blk.nodes:
                if isinstance(node, LiteralNode):
                    t = self._LITERAL_TYPE.get(node.kind, "unknown")
                    types[node.dst] = TypeInfo(var=node.dst, type_name=t, nullable=False)
                elif isinstance(node, TypeNarrowNode):
                    types[node.dst] = TypeInfo(
                        var=node.dst,
                        type_name=node.narrowed_type or "unknown",
                        nullable=False,
                    )
                elif isinstance(node, NullCheckNode):
                    types[node.result] = TypeInfo(
                        var=node.result, type_name="bool", nullable=False,
                    )
                elif isinstance(node, TruthinessNode):
                    types[node.dst] = TypeInfo(
                        var=node.dst, type_name="bool", nullable=False,
                    )

        # Forward propagation
        wl = WorkList(func.blocks.keys())
        iterations = 0
        while wl:
            iterations += 1
            blk_label = wl.pop()
            blk = func.blocks[blk_label]
            for node in blk.nodes:
                changed_here = False
                if isinstance(node, AssignNode):
                    src_t = types.get(node.src)
                    if src_t and node.dst not in types:
                        types[node.dst] = TypeInfo(
                            var=node.dst,
                            type_name=src_t.type_name,
                            nullable=src_t.nullable,
                        )
                        changed_here = True
                elif isinstance(node, PhiNode):
                    incoming_types = [
                        types.get(v) for v in node.incoming.values()
                    ]
                    known = [t for t in incoming_types if t is not None and t.is_known()]
                    if known:
                        all_same = all(t.type_name == known[0].type_name for t in known)
                        if all_same:
                            new_t = known[0].type_name
                        else:
                            new_t = "union"
                        any_nullable = any(t.nullable for t in known)
                        old = types.get(node.dst)
                        if old is None or old.type_name != new_t:
                            types[node.dst] = TypeInfo(
                                var=node.dst,
                                type_name=new_t,
                                nullable=any_nullable,
                            )
                            changed_here = True
                elif isinstance(node, BinOpNode):
                    lt = types.get(node.left)
                    rt = types.get(node.right)
                    result_type = self._infer_binop(node.op, lt, rt)
                    if result_type and node.dst not in types:
                        types[node.dst] = TypeInfo(
                            var=node.dst,
                            type_name=result_type,
                            nullable=False,
                        )
                        changed_here = True
                elif isinstance(node, UnaryOpNode):
                    ot = types.get(node.operand)
                    result_type = self._infer_unaryop(node.op, ot)
                    if result_type and node.dst not in types:
                        types[node.dst] = TypeInfo(
                            var=node.dst,
                            type_name=result_type,
                            nullable=False,
                        )
                        changed_here = True
                elif isinstance(node, LenNode):
                    if node.dst not in types:
                        types[node.dst] = TypeInfo(
                            var=node.dst, type_name="int", nullable=False,
                        )
                        changed_here = True
                elif isinstance(node, ContainerCreateNode):
                    kind_map = {
                        "LIST": "list", "DICT": "dict",
                        "SET": "set", "TUPLE": "tuple",
                        "FROZENSET": "frozenset",
                    }
                    ct = kind_map.get(node.kind.name, "unknown")
                    if node.dst not in types:
                        types[node.dst] = TypeInfo(
                            var=node.dst, type_name=ct, nullable=False,
                        )
                        changed_here = True

                if changed_here:
                    wl.extend(blk.successors)

        self.result = TypeInferenceResult(types=types)
        self.stats.iterations = iterations
        return func

    def _infer_binop(
        self, op: BinOp, lt: Optional[TypeInfo], rt: Optional[TypeInfo]
    ) -> Optional[str]:
        if op in self._CMP_OPS:
            return "bool"
        if lt is None or rt is None:
            return None
        if not lt.is_known() or not rt.is_known():
            return None
        if op == BinOp.DIV:
            return "float"
        if op in self._INT_OPS:
            if lt.type_name == "int" and rt.type_name == "int":
                return "int"
            return None
        if op in self._ARITH_OPS:
            if lt.type_name == "float" or rt.type_name == "float":
                return "float"
            if lt.type_name == "int" and rt.type_name == "int":
                return "int"
            if lt.type_name == "str" and rt.type_name == "str" and op == BinOp.ADD:
                return "str"
        return None

    @staticmethod
    def _infer_unaryop(
        op: UnaryOp, ot: Optional[TypeInfo]
    ) -> Optional[str]:
        if op == UnaryOp.NOT:
            return "bool"
        if ot is None or not ot.is_known():
            return None
        if op in (UnaryOp.NEGATE, UnaryOp.POS):
            return ot.type_name
        if op == UnaryOp.INVERT and ot.type_name == "int":
            return "int"
        return None


# ---------------------------------------------------------------------------
# Guard flow analysis
# ---------------------------------------------------------------------------


@dataclass
class GuardState:
    """A guard that is known to hold at a program point."""
    guard_kind: GuardKind
    subject: SSAVar
    polarity: bool  # True = guard condition is true, False = negated
    narrowed_type: Optional[str] = None


@dataclass
class GuardFlowResult:
    """Maps each block to the set of guards active at its entry."""
    active_guards: Dict[str, List[GuardState]] = field(default_factory=dict)


class GuardFlowAnalysis(Pass):
    """Tracks which guards are active at each program point.

    Forward dataflow: at each branch/guard node, propagate the
    guard condition into the appropriate successor.  At join points,
    only guards active in ALL predecessors survive (intersection).
    """

    def __init__(self) -> None:
        super().__init__(
            name="GuardFlowAnalysis",
            description="Track active guards per program point",
            kind=PassKind.ANALYSIS,
        )
        self.result: GuardFlowResult = GuardFlowResult()

    def run(self, func: IRFunction) -> IRFunction:
        active: Dict[str, List[GuardState]] = {l: [] for l in func.blocks}
        edge_guards: Dict[Tuple[str, str], List[GuardState]] = {}

        # Collect guards from guard/branch nodes
        for blk_label, blk in func.blocks.items():
            for node in blk.nodes:
                if isinstance(node, GuardNode):
                    true_guard = GuardState(
                        guard_kind=node.guard_kind,
                        subject=node.subject,
                        polarity=True,
                        narrowed_type=node.narrowed_type_true,
                    )
                    false_guard = GuardState(
                        guard_kind=node.guard_kind,
                        subject=node.subject,
                        polarity=False,
                        narrowed_type=node.narrowed_type_false,
                    )
                    if node.true_target:
                        edge_guards.setdefault(
                            (blk_label, node.true_target), []
                        ).append(true_guard)
                    if node.false_target:
                        edge_guards.setdefault(
                            (blk_label, node.false_target), []
                        ).append(false_guard)

        # Forward propagation
        wl = WorkList([func.entry_block])
        iterations = 0
        while wl:
            iterations += 1
            blk_label = wl.pop()
            blk = func.blocks[blk_label]

            if blk_label == func.entry_block:
                incoming_guards: List[GuardState] = []
            else:
                pred_guard_lists = []
                for pred in blk.predecessors:
                    pred_guards = list(active.get(pred, []))
                    eg = edge_guards.get((pred, blk_label), [])
                    pred_guards.extend(eg)
                    pred_guard_lists.append(pred_guards)

                if len(pred_guard_lists) == 1:
                    incoming_guards = pred_guard_lists[0]
                elif pred_guard_lists:
                    incoming_guards = self._intersect_guards(pred_guard_lists)
                else:
                    incoming_guards = []

            if incoming_guards != active.get(blk_label, []):
                active[blk_label] = incoming_guards
                wl.extend(blk.successors)

        self.result = GuardFlowResult(active_guards=active)
        self.stats.iterations = iterations
        return func

    @staticmethod
    def _intersect_guards(
        guard_lists: List[List[GuardState]],
    ) -> List[GuardState]:
        if not guard_lists:
            return []
        result = list(guard_lists[0])
        for gl in guard_lists[1:]:
            gl_keys = {(g.guard_kind, g.subject, g.polarity) for g in gl}
            result = [
                g for g in result
                if (g.guard_kind, g.subject, g.polarity) in gl_keys
            ]
        return result


# ---------------------------------------------------------------------------
# Null flow analysis
# ---------------------------------------------------------------------------


class NullState(Enum):
    UNKNOWN = auto()
    NULLABLE = auto()
    NON_NULL = auto()
    NULL = auto()


@dataclass
class NullFlowResult:
    """Maps SSA variables to their nullability state at each block."""
    states: Dict[str, Dict[SSAVar, NullState]] = field(default_factory=dict)

    def is_nullable(self, block: str, var: SSAVar) -> bool:
        blk_states = self.states.get(block, {})
        state = blk_states.get(var, NullState.UNKNOWN)
        return state in (NullState.NULLABLE, NullState.UNKNOWN, NullState.NULL)

    def is_non_null(self, block: str, var: SSAVar) -> bool:
        blk_states = self.states.get(block, {})
        return blk_states.get(var, NullState.UNKNOWN) == NullState.NON_NULL


class NullFlowAnalysis(Pass):
    """Tracks nullability through control flow.

    Forward dataflow:
      - Literal None assignments → NULL
      - Non-None literals → NON_NULL
      - is-None guards → NULL on true branch, NON_NULL on false branch
      - Phi nodes → join states (NULL ⊔ NON_NULL = NULLABLE)
    """

    def __init__(self) -> None:
        super().__init__(
            name="NullFlowAnalysis",
            description="Track nullability through control flow",
            kind=PassKind.ANALYSIS,
        )
        self.result: NullFlowResult = NullFlowResult()

    def run(self, func: IRFunction) -> IRFunction:
        states: Dict[str, Dict[SSAVar, NullState]] = {
            l: {} for l in func.blocks
        }

        # Seed from literals and null checks
        for blk_label, blk in func.blocks.items():
            for node in blk.nodes:
                if isinstance(node, LiteralNode):
                    if node.kind == LiteralKind.NONE:
                        states[blk_label][node.dst] = NullState.NULL
                    else:
                        states[blk_label][node.dst] = NullState.NON_NULL

        # Edge nullability from guards
        edge_null: Dict[Tuple[str, str], Dict[SSAVar, NullState]] = {}
        for blk_label, blk in func.blocks.items():
            for node in blk.nodes:
                if isinstance(node, GuardNode):
                    if node.guard_kind == GuardKind.IS_NONE:
                        if node.true_target:
                            edge_null.setdefault(
                                (blk_label, node.true_target), {}
                            )[node.subject] = NullState.NULL
                        if node.false_target:
                            edge_null.setdefault(
                                (blk_label, node.false_target), {}
                            )[node.subject] = NullState.NON_NULL
                    elif node.guard_kind == GuardKind.IS_NOT_NONE:
                        if node.true_target:
                            edge_null.setdefault(
                                (blk_label, node.true_target), {}
                            )[node.subject] = NullState.NON_NULL
                        if node.false_target:
                            edge_null.setdefault(
                                (blk_label, node.false_target), {}
                            )[node.subject] = NullState.NULL

        # Propagate
        wl = WorkList(func.blocks.keys())
        iterations = 0
        while wl:
            iterations += 1
            blk_label = wl.pop()
            blk = func.blocks[blk_label]

            merged: Dict[SSAVar, NullState] = {}
            for pred in blk.predecessors:
                pred_st = dict(states.get(pred, {}))
                for var, ns in edge_null.get((pred, blk_label), {}).items():
                    pred_st[var] = ns
                for var, ns in pred_st.items():
                    if var in merged:
                        merged[var] = self._join(merged[var], ns)
                    else:
                        merged[var] = ns

            # Apply local defs
            for node in blk.nodes:
                if isinstance(node, LiteralNode):
                    if node.kind == LiteralKind.NONE:
                        merged[node.dst] = NullState.NULL
                    else:
                        merged[node.dst] = NullState.NON_NULL
                elif isinstance(node, NullCheckNode):
                    merged[node.result] = NullState.NON_NULL
                elif isinstance(node, PhiNode):
                    phi_vals = []
                    for v in node.incoming.values():
                        phi_vals.append(merged.get(v, NullState.UNKNOWN))
                    result = NullState.UNKNOWN
                    for pv in phi_vals:
                        result = self._join(result, pv)
                    merged[node.dst] = result

            if merged != states.get(blk_label, {}):
                states[blk_label] = merged
                wl.extend(blk.successors)

        self.result = NullFlowResult(states=states)
        self.stats.iterations = iterations
        return func

    @staticmethod
    def _join(a: NullState, b: NullState) -> NullState:
        if a == b:
            return a
        if a == NullState.UNKNOWN:
            return b
        if b == NullState.UNKNOWN:
            return a
        return NullState.NULLABLE


# ---------------------------------------------------------------------------
# Escape analysis
# ---------------------------------------------------------------------------


class EscapeKind(Enum):
    NO_ESCAPE = auto()
    ARG_ESCAPE = auto()       # escapes through function argument
    RETURN_ESCAPE = auto()    # escapes through return
    GLOBAL_ESCAPE = auto()    # escapes to heap / global store


@dataclass
class EscapeInfo:
    var: SSAVar
    kind: EscapeKind = EscapeKind.NO_ESCAPE


@dataclass
class EscapeResult:
    """Escape information for all SSA variables."""
    escapes: Dict[SSAVar, EscapeInfo] = field(default_factory=dict)

    def escapes_function(self, var: SSAVar) -> bool:
        info = self.escapes.get(var)
        if info is None:
            return False
        return info.kind != EscapeKind.NO_ESCAPE


class EscapeAnalysis(Pass):
    """Determines which variables escape their defining function.

    A variable escapes if it is:
      - Passed as an argument to a call (ARG_ESCAPE)
      - Returned from the function (RETURN_ESCAPE)
      - Stored into an attribute or index of another object (GLOBAL_ESCAPE)
      - Captured by a closure (GLOBAL_ESCAPE)
      - Yielded (GLOBAL_ESCAPE)

    Uses backward propagation: if a container escapes, its elements
    may also escape.
    """

    def __init__(self) -> None:
        super().__init__(
            name="EscapeAnalysis",
            description="Determine variable escape information",
            kind=PassKind.ANALYSIS,
        )
        self.result: EscapeResult = EscapeResult()

    def run(self, func: IRFunction) -> IRFunction:
        escapes: Dict[SSAVar, EscapeInfo] = {}

        # Initialize all defined vars as non-escaping
        for blk in func.blocks.values():
            for node in blk.nodes:
                for v in node.defined_vars:
                    escapes[v] = EscapeInfo(var=v, kind=EscapeKind.NO_ESCAPE)

        # Direct escape sources
        for blk in func.blocks.values():
            for node in blk.nodes:
                if isinstance(node, ReturnNode):
                    if node.value is not None:
                        self._mark(escapes, node.value, EscapeKind.RETURN_ESCAPE)
                elif isinstance(node, CallNode):
                    for arg in node.args:
                        self._mark(escapes, arg, EscapeKind.ARG_ESCAPE)
                    for kv in node.kwargs.values():
                        self._mark(escapes, kv, EscapeKind.ARG_ESCAPE)
                    if node.star_args is not None:
                        self._mark(escapes, node.star_args, EscapeKind.ARG_ESCAPE)
                    if node.star_kwargs is not None:
                        self._mark(escapes, node.star_kwargs, EscapeKind.ARG_ESCAPE)
                elif isinstance(node, StoreAttrNode):
                    self._mark(escapes, node.value, EscapeKind.GLOBAL_ESCAPE)
                elif isinstance(node, StoreIndexNode):
                    self._mark(escapes, node.value, EscapeKind.GLOBAL_ESCAPE)
                elif isinstance(node, YieldNode):
                    if node.value is not None:
                        self._mark(escapes, node.value, EscapeKind.GLOBAL_ESCAPE)
                elif isinstance(node, ClosureCaptureNode):
                    for v in node.used_vars:
                        self._mark(escapes, v, EscapeKind.GLOBAL_ESCAPE)
                elif isinstance(node, RaiseNode):
                    for v in node.used_vars:
                        self._mark(escapes, v, EscapeKind.GLOBAL_ESCAPE)

        # Propagate through copies / assigns
        changed = True
        while changed:
            changed = False
            for blk in func.blocks.values():
                for node in blk.nodes:
                    if isinstance(node, AssignNode):
                        src_info = escapes.get(node.dst)
                        if src_info and src_info.kind != EscapeKind.NO_ESCAPE:
                            if self._mark(escapes, node.src, src_info.kind):
                                changed = True
                    elif isinstance(node, PhiNode):
                        dst_info = escapes.get(node.dst)
                        if dst_info and dst_info.kind != EscapeKind.NO_ESCAPE:
                            for v in node.incoming.values():
                                if self._mark(escapes, v, dst_info.kind):
                                    changed = True

        self.result = EscapeResult(escapes=escapes)
        return func

    @staticmethod
    def _mark(
        escapes: Dict[SSAVar, EscapeInfo], var: SSAVar, kind: EscapeKind
    ) -> bool:
        info = escapes.get(var)
        if info is None:
            escapes[var] = EscapeInfo(var=var, kind=kind)
            return True
        if kind.value > info.kind.value:
            info.kind = kind
            return True
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════


class PythonSSAPasses:
    """Orchestrator that runs all passes in the correct order.

    Default pipeline:
      1. CriticalEdgeSplitting (ensure correct phi semantics)
      2. UnreachableBlockElimination
      3. PhiSimplification
      4. CopyPropagation
      5. ConstantFolding
      6. ConstantPropagation
      7. DeadCodeElimination
      8. PhiElimination
      9. CommonSubexpressionElimination
     10. BlockMerging
     --- Analysis ---
     11. DominatorTreePass
     12. LivenessAnalysis
     13. ReachingDefinitions
     14. DefUseChains
     15. LoopDetection
     16. TypeInferencePass
     17. GuardFlowAnalysis
     18. NullFlowAnalysis
     19. EscapeAnalysis

    The optimization passes (1-10) run in a fixed-point loop.
    Analysis passes (11-19) run once after optimizations converge.
    """

    def __init__(
        self,
        *,
        verify: bool = False,
        max_opt_iterations: int = 10,
    ) -> None:
        self._verify = verify
        self._max_opt_iterations = max_opt_iterations
        self.opt_stats: List[PassStatistics] = []
        self.analysis_results: Dict[str, Pass] = {}

    def run(self, func: IRFunction) -> IRFunction:
        """Run the full pipeline on *func*."""
        func = self._run_optimizations(func)
        func = self._run_analyses(func)
        return func

    def _run_optimizations(self, func: IRFunction) -> IRFunction:
        mgr = PassManager(
            max_iterations=self._max_opt_iterations,
            verify=self._verify,
        )
        mgr.add(CriticalEdgeSplitting())
        mgr.add(UnreachableBlockElimination())
        mgr.add(PhiSimplification())
        mgr.add(CopyPropagation())
        mgr.add(ConstantFolding())
        mgr.add(ConstantPropagation())
        mgr.add(DeadCodeElimination())
        mgr.add(PhiElimination())
        mgr.add(CommonSubexpressionElimination())
        mgr.add(BlockMerging())

        func = mgr.run(func, fixed_point=True)
        self.opt_stats = mgr.all_stats
        return func

    def _run_analyses(self, func: IRFunction) -> IRFunction:
        analyses: List[Pass] = [
            DominatorTreePass(),
            LivenessAnalysis(),
            ReachingDefinitions(),
            DefUseChains(),
            LoopDetection(),
            TypeInferencePass(),
            GuardFlowAnalysis(),
            NullFlowAnalysis(),
            EscapeAnalysis(),
        ]
        for a in analyses:
            func = a.run(func)
            self.analysis_results[a.name] = a
        return func

    def get_analysis(self, name: str) -> Optional[Pass]:
        return self.analysis_results.get(name)

    def get_liveness(self) -> Optional[LivenessResult]:
        a = self.analysis_results.get("LivenessAnalysis")
        return a.result if isinstance(a, LivenessAnalysis) else None

    def get_dominator_tree(self) -> Optional[DominatorResult]:
        a = self.analysis_results.get("DominatorTreePass")
        return a.result if isinstance(a, DominatorTreePass) else None

    def get_loops(self) -> Optional[LoopDetectionResult]:
        a = self.analysis_results.get("LoopDetection")
        return a.result if isinstance(a, LoopDetection) else None

    def get_types(self) -> Optional[TypeInferenceResult]:
        a = self.analysis_results.get("TypeInferencePass")
        return a.result if isinstance(a, TypeInferencePass) else None

    def get_guard_flow(self) -> Optional[GuardFlowResult]:
        a = self.analysis_results.get("GuardFlowAnalysis")
        return a.result if isinstance(a, GuardFlowAnalysis) else None

    def get_null_flow(self) -> Optional[NullFlowResult]:
        a = self.analysis_results.get("NullFlowAnalysis")
        return a.result if isinstance(a, NullFlowAnalysis) else None

    def get_escape_info(self) -> Optional[EscapeResult]:
        a = self.analysis_results.get("EscapeAnalysis")
        return a.result if isinstance(a, EscapeAnalysis) else None

    def get_def_use(self) -> Optional[DefUseResult]:
        a = self.analysis_results.get("DefUseChains")
        return a.result if isinstance(a, DefUseChains) else None

    def get_reaching_defs(self) -> Optional[ReachingDefsResult]:
        a = self.analysis_results.get("ReachingDefinitions")
        return a.result if isinstance(a, ReachingDefinitions) else None

    def print_stats(self) -> str:
        lines: List[str] = ["=== Optimization Statistics ==="]
        for s in self.opt_stats:
            if s.changed:
                lines.append(s.summary())
        lines.append("=== Analysis Results ===")
        for name, a in self.analysis_results.items():
            lines.append(f"  {name}: completed")
        return "\n".join(lines)

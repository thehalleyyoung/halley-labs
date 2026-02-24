from __future__ import annotations

import json
import copy
import enum
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generic,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)


# ---------------------------------------------------------------------------
# Local lightweight IR types – no cross-module imports
# ---------------------------------------------------------------------------

@dataclass
class SourceRange:
    """Source location span."""
    file: str = ""
    start_line: int = 0
    start_col: int = 0
    end_line: int = 0
    end_col: int = 0

    def __str__(self) -> str:
        return f"{self.file}:{self.start_line}:{self.start_col}-{self.end_line}:{self.end_col}"


@dataclass
class SSAVar:
    """SSA variable with name and version."""
    name: str
    version: int = 0

    def __hash__(self) -> int:
        return hash((self.name, self.version))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SSAVar):
            return NotImplemented
        return self.name == other.name and self.version == other.version

    def __str__(self) -> str:
        return f"{self.name}_{self.version}"

    def __repr__(self) -> str:
        return f"SSAVar({self.name!r}, {self.version})"


@dataclass
class IRInstruction:
    """Base IR instruction."""
    opcode: str = ""
    dest: Optional[SSAVar] = None
    operands: List[Any] = field(default_factory=list)
    source_range: Optional[SourceRange] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_defs(self) -> List[SSAVar]:
        if self.dest is not None:
            return [self.dest]
        return []

    def get_uses(self) -> List[SSAVar]:
        uses: List[SSAVar] = []
        for op in self.operands:
            if isinstance(op, SSAVar):
                uses.append(op)
        return uses

    def replace_use(self, old: SSAVar, new: SSAVar) -> None:
        for i, op in enumerate(self.operands):
            if isinstance(op, SSAVar) and op == old:
                self.operands[i] = new

    def __str__(self) -> str:
        dest_s = f"{self.dest} = " if self.dest else ""
        ops = ", ".join(str(o) for o in self.operands)
        return f"{dest_s}{self.opcode}({ops})"


@dataclass
class PhiNode:
    """Phi function: dest = phi(incoming_values)."""
    dest: SSAVar
    incoming: Dict[str, SSAVar] = field(default_factory=dict)
    source_range: Optional[SourceRange] = None

    def get_defs(self) -> List[SSAVar]:
        return [self.dest]

    def get_uses(self) -> List[SSAVar]:
        return list(self.incoming.values())

    def add_incoming(self, block_id: str, var: SSAVar) -> None:
        self.incoming[block_id] = var

    def remove_incoming(self, block_id: str) -> None:
        self.incoming.pop(block_id, None)

    def replace_use(self, old: SSAVar, new: SSAVar) -> None:
        for bid, v in list(self.incoming.items()):
            if v == old:
                self.incoming[bid] = new

    def __str__(self) -> str:
        pairs = ", ".join(f"[{bid}]: {v}" for bid, v in self.incoming.items())
        return f"{self.dest} = phi({pairs})"


class EdgeKind(enum.Enum):
    NORMAL = "normal"
    TRUE_BRANCH = "true"
    FALSE_BRANCH = "false"
    EXCEPTION = "exception"
    BACK = "back"
    BREAK = "break"
    CONTINUE = "continue"
    FALLTHROUGH = "fallthrough"


@dataclass
class CFGEdge:
    """Edge between basic blocks."""
    src: str
    dst: str
    kind: EdgeKind = EdgeKind.NORMAL
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash((self.src, self.dst, self.kind))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CFGEdge):
            return NotImplemented
        return self.src == other.src and self.dst == other.dst and self.kind == other.kind


# ===================================================================
# BasicBlock
# ===================================================================

class BasicBlock:
    """A basic block containing a linear sequence of IR instructions."""

    def __init__(
        self,
        block_id: str,
        label: Optional[str] = None,
    ) -> None:
        self.block_id: str = block_id
        self.label: str = label or block_id
        self.instructions: List[IRInstruction] = []
        self.phi_nodes: List[PhiNode] = []
        self.predecessors: List[str] = []
        self.successors: List[str] = []
        self.source_range: Optional[SourceRange] = None
        self.loop_depth: int = 0
        self.is_loop_header: bool = False
        self.is_exit: bool = False
        self._dominator: Optional[str] = None
        self._post_dominator: Optional[str] = None
        self.metadata: Dict[str, Any] = {}

    # ----- instructions -----

    def add_instruction(self, instr: IRInstruction) -> None:
        self.instructions.append(instr)

    def insert_instruction(self, index: int, instr: IRInstruction) -> None:
        self.instructions.insert(index, instr)

    def remove_instruction(self, instr: IRInstruction) -> bool:
        try:
            self.instructions.remove(instr)
            return True
        except ValueError:
            return False

    def remove_instruction_at(self, index: int) -> IRInstruction:
        return self.instructions.pop(index)

    def replace_instruction(self, old: IRInstruction, new: IRInstruction) -> bool:
        for i, ins in enumerate(self.instructions):
            if ins is old:
                self.instructions[i] = new
                return True
        return False

    def instruction_count(self) -> int:
        return len(self.instructions)

    def is_empty(self) -> bool:
        return len(self.instructions) == 0 and len(self.phi_nodes) == 0

    def last_instruction(self) -> Optional[IRInstruction]:
        return self.instructions[-1] if self.instructions else None

    def first_instruction(self) -> Optional[IRInstruction]:
        return self.instructions[0] if self.instructions else None

    # ----- phi nodes -----

    def add_phi(self, phi: PhiNode) -> None:
        self.phi_nodes.append(phi)

    def get_phis(self) -> List[PhiNode]:
        return list(self.phi_nodes)

    def remove_phi(self, phi: PhiNode) -> bool:
        try:
            self.phi_nodes.remove(phi)
            return True
        except ValueError:
            return False

    def remove_phi_for(self, var_name: str) -> bool:
        for i, phi in enumerate(self.phi_nodes):
            if phi.dest.name == var_name:
                self.phi_nodes.pop(i)
                return True
        return False

    def find_phi(self, var_name: str) -> Optional[PhiNode]:
        for phi in self.phi_nodes:
            if phi.dest.name == var_name:
                return phi
        return None

    # ----- predecessors / successors -----

    def add_predecessor(self, block_id: str) -> None:
        if block_id not in self.predecessors:
            self.predecessors.append(block_id)

    def add_successor(self, block_id: str) -> None:
        if block_id not in self.successors:
            self.successors.append(block_id)

    def remove_predecessor(self, block_id: str) -> bool:
        try:
            self.predecessors.remove(block_id)
            return True
        except ValueError:
            return False

    def remove_successor(self, block_id: str) -> bool:
        try:
            self.successors.remove(block_id)
            return True
        except ValueError:
            return False

    def has_predecessor(self, block_id: str) -> bool:
        return block_id in self.predecessors

    def has_successor(self, block_id: str) -> bool:
        return block_id in self.successors

    def predecessor_count(self) -> int:
        return len(self.predecessors)

    def successor_count(self) -> int:
        return len(self.successors)

    # ----- dominance (delegated, filled externally) -----

    def dominates(self, other: BasicBlock, dom_tree: Optional[DominatorTree] = None) -> bool:
        if dom_tree is not None:
            return dom_tree.dominates(self.block_id, other.block_id)
        return False

    def post_dominates(self, other: BasicBlock, pdom_tree: Optional[PostDominatorTree] = None) -> bool:
        if pdom_tree is not None:
            return pdom_tree.post_dominates(self.block_id, other.block_id)
        return False

    # ----- iteration -----

    def __iter__(self) -> Iterator[IRInstruction]:
        return iter(self.instructions)

    def __len__(self) -> int:
        return len(self.instructions)

    def all_defs(self) -> List[SSAVar]:
        defs: List[SSAVar] = []
        for phi in self.phi_nodes:
            defs.extend(phi.get_defs())
        for instr in self.instructions:
            defs.extend(instr.get_defs())
        return defs

    def all_uses(self) -> List[SSAVar]:
        uses: List[SSAVar] = []
        for phi in self.phi_nodes:
            uses.extend(phi.get_uses())
        for instr in self.instructions:
            uses.extend(instr.get_uses())
        return uses

    def __repr__(self) -> str:
        return f"BasicBlock({self.block_id!r}, instrs={len(self.instructions)}, phis={len(self.phi_nodes)})"

    def __str__(self) -> str:
        lines = [f"Block {self.block_id} (label={self.label}):"]
        if self.predecessors:
            lines.append(f"  preds: {', '.join(self.predecessors)}")
        if self.successors:
            lines.append(f"  succs: {', '.join(self.successors)}")
        for phi in self.phi_nodes:
            lines.append(f"  {phi}")
        for instr in self.instructions:
            lines.append(f"  {instr}")
        return "\n".join(lines)


# ===================================================================
# CFG
# ===================================================================

class CFG:
    """Control-flow graph."""

    def __init__(self, name: str = "") -> None:
        self.name: str = name
        self.blocks: Dict[str, BasicBlock] = {}
        self.edges: List[CFGEdge] = []
        self._entry_id: Optional[str] = None
        self._exit_id: Optional[str] = None
        self._block_counter: int = 0
        self.metadata: Dict[str, Any] = {}

    # ----- block management -----

    def fresh_block_id(self, prefix: str = "bb") -> str:
        self._block_counter += 1
        return f"{prefix}_{self._block_counter}"

    def add_block(self, block: BasicBlock) -> None:
        self.blocks[block.block_id] = block

    def create_block(self, label: Optional[str] = None) -> BasicBlock:
        bid = self.fresh_block_id()
        blk = BasicBlock(bid, label)
        self.add_block(blk)
        return blk

    def remove_block(self, block_id: str) -> bool:
        if block_id not in self.blocks:
            return False
        blk = self.blocks[block_id]
        for pred_id in list(blk.predecessors):
            self.remove_edge(pred_id, block_id)
        for succ_id in list(blk.successors):
            self.remove_edge(block_id, succ_id)
        del self.blocks[block_id]
        if self._entry_id == block_id:
            self._entry_id = None
        if self._exit_id == block_id:
            self._exit_id = None
        return True

    def has_block(self, block_id: str) -> bool:
        return block_id in self.blocks

    def get_block(self, block_id: str) -> BasicBlock:
        return self.blocks[block_id]

    def get_blocks(self) -> List[BasicBlock]:
        return list(self.blocks.values())

    def block_count(self) -> int:
        return len(self.blocks)

    # ----- edge management -----

    def add_edge(self, src: str, dst: str, kind: EdgeKind = EdgeKind.NORMAL, weight: float = 1.0) -> CFGEdge:
        edge = CFGEdge(src, dst, kind, weight)
        self.edges.append(edge)
        if src in self.blocks:
            self.blocks[src].add_successor(dst)
        if dst in self.blocks:
            self.blocks[dst].add_predecessor(src)
        return edge

    def remove_edge(self, src: str, dst: str) -> bool:
        removed = False
        new_edges: List[CFGEdge] = []
        for e in self.edges:
            if e.src == src and e.dst == dst:
                removed = True
            else:
                new_edges.append(e)
        self.edges = new_edges
        if removed:
            if src in self.blocks:
                self.blocks[src].remove_successor(dst)
            if dst in self.blocks:
                self.blocks[dst].remove_predecessor(src)
        return removed

    def has_edge(self, src: str, dst: str) -> bool:
        return any(e.src == src and e.dst == dst for e in self.edges)

    def get_edge(self, src: str, dst: str) -> Optional[CFGEdge]:
        for e in self.edges:
            if e.src == src and e.dst == dst:
                return e
        return None

    def get_edges_from(self, src: str) -> List[CFGEdge]:
        return [e for e in self.edges if e.src == src]

    def get_edges_to(self, dst: str) -> List[CFGEdge]:
        return [e for e in self.edges if e.dst == dst]

    def get_predecessors(self, block_id: str) -> List[str]:
        if block_id in self.blocks:
            return list(self.blocks[block_id].predecessors)
        return []

    def get_successors(self, block_id: str) -> List[str]:
        if block_id in self.blocks:
            return list(self.blocks[block_id].successors)
        return []

    # ----- entry / exit -----

    def get_entry(self) -> Optional[BasicBlock]:
        if self._entry_id and self._entry_id in self.blocks:
            return self.blocks[self._entry_id]
        return None

    def get_exit(self) -> Optional[BasicBlock]:
        if self._exit_id and self._exit_id in self.blocks:
            return self.blocks[self._exit_id]
        return None

    def set_entry(self, block_id: str) -> None:
        self._entry_id = block_id

    def set_exit(self, block_id: str) -> None:
        self._exit_id = block_id

    @property
    def entry_block(self) -> Optional[BasicBlock]:
        return self.get_entry()

    @property
    def exit_block(self) -> Optional[BasicBlock]:
        return self.get_exit()

    # ----- traversals -----

    def dfs_preorder(self, start: Optional[str] = None) -> List[str]:
        start = start or self._entry_id
        if start is None:
            return []
        visited: Set[str] = set()
        order: List[str] = []

        def _dfs(bid: str) -> None:
            if bid in visited or bid not in self.blocks:
                return
            visited.add(bid)
            order.append(bid)
            for s in self.blocks[bid].successors:
                _dfs(s)

        _dfs(start)
        return order

    def dfs_postorder(self, start: Optional[str] = None) -> List[str]:
        start = start or self._entry_id
        if start is None:
            return []
        visited: Set[str] = set()
        order: List[str] = []

        def _dfs(bid: str) -> None:
            if bid in visited or bid not in self.blocks:
                return
            visited.add(bid)
            for s in self.blocks[bid].successors:
                _dfs(s)
            order.append(bid)

        _dfs(start)
        return order

    def reverse_postorder(self, start: Optional[str] = None) -> List[str]:
        po = self.dfs_postorder(start)
        po.reverse()
        return po

    def bfs(self, start: Optional[str] = None) -> List[str]:
        start = start or self._entry_id
        if start is None:
            return []
        from collections import deque
        visited: Set[str] = set()
        queue: deque[str] = deque([start])
        visited.add(start)
        order: List[str] = []
        while queue:
            bid = queue.popleft()
            order.append(bid)
            for s in self.blocks[bid].successors:
                if s not in visited and s in self.blocks:
                    visited.add(s)
                    queue.append(s)
        return order

    # ----- loop analysis helpers -----

    def get_back_edges(self) -> List[Tuple[str, str]]:
        """Return back edges (edges whose target dominates their source)."""
        dom = DominatorTree(self)
        dom.compute_dominators()
        back: List[Tuple[str, str]] = []
        for e in self.edges:
            if dom.dominates(e.dst, e.src):
                back.append((e.src, e.dst))
        return back

    def get_loop_headers(self) -> Set[str]:
        return {dst for _, dst in self.get_back_edges()}

    def get_loop_body(self, header: str, tail: str) -> Set[str]:
        """Get natural loop body given header and tail of back edge."""
        body: Set[str] = {header}
        if header == tail:
            return body
        stack: List[str] = [tail]
        body.add(tail)
        while stack:
            n = stack.pop()
            for pred in self.get_predecessors(n):
                if pred not in body:
                    body.add(pred)
                    stack.append(pred)
        return body

    def get_natural_loops(self) -> List[Tuple[str, Set[str]]]:
        """Return list of (header, body_set) for each natural loop."""
        loops: List[Tuple[str, Set[str]]] = []
        for tail, header in self.get_back_edges():
            body = self.get_loop_body(header, tail)
            loops.append((header, body))
        return loops

    def get_loop_nesting_tree(self) -> Dict[str, List[str]]:
        """Build loop nesting tree: header -> list of immediately nested headers."""
        loops = self.get_natural_loops()
        if not loops:
            return {}
        loops_sorted = sorted(loops, key=lambda x: len(x[1]))
        nesting: Dict[str, List[str]] = {h: [] for h, _ in loops}
        header_body = {h: b for h, b in loops}
        for i, (h_inner, b_inner) in enumerate(loops_sorted):
            for j in range(i + 1, len(loops_sorted)):
                h_outer, b_outer = loops_sorted[j]
                if h_inner != h_outer and b_inner <= b_outer:
                    nesting[h_outer].append(h_inner)
                    break
        return nesting

    def is_reducible(self) -> bool:
        """Check if CFG is reducible (all back edges go to dominators)."""
        dom = DominatorTree(self)
        dom.compute_dominators()
        dfs_order = self.dfs_preorder()
        order_map = {bid: idx for idx, bid in enumerate(dfs_order)}
        for e in self.edges:
            if e.src in order_map and e.dst in order_map:
                if order_map[e.dst] <= order_map[e.src]:
                    if not dom.dominates(e.dst, e.src):
                        return False
        return True

    # ----- CFG transformations -----

    def split_edge(self, src: str, dst: str) -> BasicBlock:
        """Split edge (src→dst) by inserting empty block."""
        new_blk = self.create_block(label=f"split_{src}_{dst}")
        edge = self.get_edge(src, dst)
        kind = edge.kind if edge else EdgeKind.NORMAL
        self.remove_edge(src, dst)
        self.add_edge(src, new_blk.block_id, kind)
        self.add_edge(new_blk.block_id, dst)
        dst_blk = self.blocks[dst]
        for phi in dst_blk.phi_nodes:
            if src in phi.incoming:
                val = phi.incoming.pop(src)
                phi.incoming[new_blk.block_id] = val
        return new_blk

    def merge_blocks(self, a_id: str, b_id: str) -> bool:
        """Merge block b into block a (a must be sole pred of b, b sole succ of a)."""
        if a_id not in self.blocks or b_id not in self.blocks:
            return False
        a = self.blocks[a_id]
        b = self.blocks[b_id]
        if a.successors != [b_id] or b.predecessors != [a_id]:
            return False
        if b.phi_nodes:
            return False
        a.instructions.extend(b.instructions)
        self.remove_edge(a_id, b_id)
        for succ_id in list(b.successors):
            self.remove_edge(b_id, succ_id)
            self.add_edge(a_id, succ_id)
            succ_blk = self.blocks.get(succ_id)
            if succ_blk:
                for phi in succ_blk.phi_nodes:
                    if b_id in phi.incoming:
                        val = phi.incoming.pop(b_id)
                        phi.incoming[a_id] = val
        if self._exit_id == b_id:
            self._exit_id = a_id
        del self.blocks[b_id]
        return True

    def simplify(self) -> int:
        """Simplify CFG by merging chains and removing empty blocks. Return count of changes."""
        changes = 0
        changed = True
        while changed:
            changed = False
            for bid in list(self.blocks.keys()):
                if bid not in self.blocks:
                    continue
                blk = self.blocks[bid]
                if len(blk.successors) == 1:
                    succ_id = blk.successors[0]
                    if succ_id in self.blocks:
                        succ = self.blocks[succ_id]
                        if len(succ.predecessors) == 1 and succ.predecessors[0] == bid:
                            if bid != succ_id:
                                if self.merge_blocks(bid, succ_id):
                                    changed = True
                                    changes += 1
                if bid in self.blocks and blk.is_empty() and len(blk.successors) == 1:
                    succ_id = blk.successors[0]
                    if bid != self._entry_id and bid != self._exit_id and succ_id != bid:
                        preds = list(blk.predecessors)
                        for pred_id in preds:
                            edge = self.get_edge(pred_id, bid)
                            kind = edge.kind if edge else EdgeKind.NORMAL
                            self.remove_edge(pred_id, bid)
                            self.add_edge(pred_id, succ_id, kind)
                        succ_blk = self.blocks.get(succ_id)
                        if succ_blk:
                            for phi in succ_blk.phi_nodes:
                                if bid in phi.incoming:
                                    val = phi.incoming.pop(bid)
                                    for pred_id in preds:
                                        phi.incoming[pred_id] = val
                        self.remove_edge(bid, succ_id)
                        del self.blocks[bid]
                        changed = True
                        changes += 1
        return changes

    # ----- clone / subgraph -----

    def clone(self) -> CFG:
        """Deep clone the entire CFG."""
        new_cfg = CFG(self.name)
        new_cfg._block_counter = self._block_counter
        new_cfg._entry_id = self._entry_id
        new_cfg._exit_id = self._exit_id
        new_cfg.metadata = dict(self.metadata)
        for bid, blk in self.blocks.items():
            nb = BasicBlock(blk.block_id, blk.label)
            nb.instructions = [copy.deepcopy(i) for i in blk.instructions]
            nb.phi_nodes = [copy.deepcopy(p) for p in blk.phi_nodes]
            nb.predecessors = list(blk.predecessors)
            nb.successors = list(blk.successors)
            nb.source_range = copy.deepcopy(blk.source_range)
            nb.loop_depth = blk.loop_depth
            nb.is_loop_header = blk.is_loop_header
            nb.is_exit = blk.is_exit
            nb.metadata = dict(blk.metadata)
            new_cfg.blocks[bid] = nb
        new_cfg.edges = [CFGEdge(e.src, e.dst, e.kind, e.weight, dict(e.metadata)) for e in self.edges]
        return new_cfg

    def subgraph(self, block_ids: Set[str]) -> CFG:
        """Extract a subgraph containing only the given blocks."""
        sub = CFG(f"{self.name}_sub")
        for bid in block_ids:
            if bid in self.blocks:
                blk = self.blocks[bid]
                nb = BasicBlock(bid, blk.label)
                nb.instructions = [copy.deepcopy(i) for i in blk.instructions]
                nb.phi_nodes = [copy.deepcopy(p) for p in blk.phi_nodes]
                nb.predecessors = [p for p in blk.predecessors if p in block_ids]
                nb.successors = [s for s in blk.successors if s in block_ids]
                nb.loop_depth = blk.loop_depth
                nb.is_loop_header = blk.is_loop_header
                nb.is_exit = blk.is_exit
                sub.blocks[bid] = nb
        for e in self.edges:
            if e.src in block_ids and e.dst in block_ids:
                sub.edges.append(CFGEdge(e.src, e.dst, e.kind, e.weight))
        if self._entry_id in block_ids:
            sub._entry_id = self._entry_id
        if self._exit_id in block_ids:
            sub._exit_id = self._exit_id
        return sub

    # ----- validation -----

    def validate(self) -> List[str]:
        """Check well-formedness. Return list of error messages."""
        errors: List[str] = []
        if self._entry_id is None:
            errors.append("No entry block set")
        elif self._entry_id not in self.blocks:
            errors.append(f"Entry block {self._entry_id} not in CFG")
        if self._exit_id is None:
            errors.append("No exit block set")
        elif self._exit_id not in self.blocks:
            errors.append(f"Exit block {self._exit_id} not in CFG")

        if self._entry_id and self._entry_id in self.blocks:
            entry = self.blocks[self._entry_id]
            if entry.predecessors:
                errors.append(f"Entry block {self._entry_id} has predecessors: {entry.predecessors}")

        if self._exit_id and self._exit_id in self.blocks:
            exit_blk = self.blocks[self._exit_id]
            if exit_blk.successors:
                errors.append(f"Exit block {self._exit_id} has successors: {exit_blk.successors}")

        for bid, blk in self.blocks.items():
            for succ in blk.successors:
                if succ not in self.blocks:
                    errors.append(f"Block {bid} has successor {succ} not in CFG")
                elif bid not in self.blocks[succ].predecessors:
                    errors.append(f"Block {bid} lists {succ} as successor but not back-linked")
            for pred in blk.predecessors:
                if pred not in self.blocks:
                    errors.append(f"Block {bid} has predecessor {pred} not in CFG")
                elif bid not in self.blocks[pred].successors:
                    errors.append(f"Block {bid} lists {pred} as predecessor but not back-linked")

        if self._entry_id and self._entry_id in self.blocks:
            reachable = set(self.dfs_preorder())
            for bid in self.blocks:
                if bid not in reachable:
                    errors.append(f"Block {bid} not reachable from entry")

        for bid, blk in self.blocks.items():
            for phi in blk.phi_nodes:
                phi_preds = set(phi.incoming.keys())
                blk_preds = set(blk.predecessors)
                if phi_preds != blk_preds:
                    errors.append(
                        f"Phi {phi.dest} in {bid}: incoming blocks {phi_preds} != preds {blk_preds}"
                    )

        for e in self.edges:
            if e.src not in self.blocks:
                errors.append(f"Edge src {e.src} not in CFG")
            if e.dst not in self.blocks:
                errors.append(f"Edge dst {e.dst} not in CFG")

        return errors

    # ----- Graphviz -----

    def to_dot(self, show_instructions: bool = True) -> str:
        lines = ["digraph CFG {"]
        lines.append('  rankdir=TB;')
        lines.append('  node [shape=record, fontname="Courier"];')
        for bid, blk in self.blocks.items():
            label_parts = [f"<b>{bid}"]
            if blk.label != bid:
                label_parts.append(f" ({blk.label})")
            label_parts.append("|")
            for phi in blk.phi_nodes:
                label_parts.append(str(phi).replace('"', '\\"').replace("<", "\\<").replace(">", "\\>") + "\\l")
            if show_instructions:
                for instr in blk.instructions:
                    label_parts.append(str(instr).replace('"', '\\"').replace("<", "\\<").replace(">", "\\>") + "\\l")
            node_label = "".join(label_parts)
            attrs = f'label="{node_label}"'
            if bid == self._entry_id:
                attrs += ', style=filled, fillcolor=lightgreen'
            elif bid == self._exit_id:
                attrs += ', style=filled, fillcolor=lightyellow'
            elif blk.is_loop_header:
                attrs += ', style=filled, fillcolor=lightblue'
            lines.append(f'  "{bid}" [{attrs}];')
        for e in self.edges:
            edge_attrs = ""
            if e.kind == EdgeKind.TRUE_BRANCH:
                edge_attrs = ' [label="T", color=green]'
            elif e.kind == EdgeKind.FALSE_BRANCH:
                edge_attrs = ' [label="F", color=red]'
            elif e.kind == EdgeKind.EXCEPTION:
                edge_attrs = ' [label="exc", color=orange, style=dashed]'
            elif e.kind == EdgeKind.BACK:
                edge_attrs = ' [label="back", style=dotted]'
            lines.append(f'  "{e.src}" -> "{e.dst}"{edge_attrs};')
        lines.append("}")
        return "\n".join(lines)

    # ----- pretty print -----

    def pretty_print(self) -> str:
        lines: List[str] = []
        lines.append(f"=== CFG: {self.name} ===")
        lines.append(f"Entry: {self._entry_id}, Exit: {self._exit_id}")
        lines.append(f"Blocks: {len(self.blocks)}, Edges: {len(self.edges)}")
        lines.append("")
        for bid in self.reverse_postorder():
            blk = self.blocks[bid]
            lines.append(str(blk))
            lines.append("")
        unreachable = set(self.blocks.keys()) - set(self.reverse_postorder())
        for bid in sorted(unreachable):
            lines.append(f"[UNREACHABLE] {self.blocks[bid]}")
            lines.append("")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"CFG({self.name!r}, blocks={len(self.blocks)}, edges={len(self.edges)})"


# ===================================================================
# DominatorTree  (Lengauer-Tarjan)
# ===================================================================

class DominatorTree:
    """Dominator tree computation using Lengauer-Tarjan algorithm."""

    def __init__(self, cfg: CFG) -> None:
        self.cfg = cfg
        self._idom: Dict[str, str] = {}
        self._dom_children: Dict[str, List[str]] = {}
        self._dom_frontier: Dict[str, Set[str]] = {}
        self._computed = False

    def compute_dominators(self) -> None:
        """Compute dominators using Lengauer-Tarjan algorithm."""
        entry = self.cfg._entry_id
        if entry is None or entry not in self.cfg.blocks:
            return

        vertex: List[str] = []
        semi: Dict[str, int] = {}
        parent: Dict[str, Optional[str]] = {}
        pred_map: Dict[str, List[str]] = {bid: [] for bid in self.cfg.blocks}
        bucket: Dict[str, Set[str]] = {bid: set() for bid in self.cfg.blocks}
        dom: Dict[str, str] = {}
        ancestor: Dict[str, Optional[str]] = {bid: None for bid in self.cfg.blocks}
        label: Dict[str, str] = {bid: bid for bid in self.cfg.blocks}

        # step 1: DFS numbering
        dfn: Dict[str, int] = {}
        counter = [0]

        def dfs(v: str) -> None:
            dfn[v] = counter[0]
            semi[v] = counter[0]
            vertex.append(v)
            counter[0] += 1
            for w in self.cfg.get_successors(v):
                if w not in dfn:
                    parent[w] = v
                    dfs(w)
                pred_map[w].append(v)

        parent[entry] = None
        dfs(entry)

        n = len(vertex)
        if n == 0:
            self._computed = True
            return

        def _compress(v: str) -> None:
            anc = ancestor[v]
            if anc is not None and ancestor[anc] is not None:
                _compress(anc)
                if semi[label[anc]] < semi[label[v]]:
                    label[v] = label[anc]
                ancestor[v] = ancestor[anc]

        def _eval(v: str) -> str:
            if ancestor[v] is None:
                return v
            _compress(v)
            return label[v]

        def _link(v: str, w: str) -> None:
            ancestor[w] = v

        # steps 2-3
        for i in range(n - 1, 0, -1):
            w = vertex[i]
            # step 2
            for v in pred_map[w]:
                if v in dfn:
                    u = _eval(v)
                    if semi[u] < semi[w]:
                        semi[w] = semi[u]
            bucket[vertex[semi[w]]].add(w)
            p = parent[w]
            if p is not None:
                _link(p, w)
                # step 3
                for v in list(bucket[p]):
                    bucket[p].discard(v)
                    u = _eval(v)
                    dom[v] = u if semi[u] < semi[v] else p

        # step 4
        for i in range(1, n):
            w = vertex[i]
            if w in dom and dom[w] != vertex[semi[w]]:
                dom[w] = dom[dom[w]]

        self._idom = dom
        self._idom[entry] = entry

        # build children map
        self._dom_children = {bid: [] for bid in self.cfg.blocks}
        for node, idom in self._idom.items():
            if node != idom:
                self._dom_children.setdefault(idom, []).append(node)

        self._computed = True

    def immediate_dominator(self, block_id: str) -> Optional[str]:
        if not self._computed:
            self.compute_dominators()
        idom = self._idom.get(block_id)
        if idom == block_id:
            return None
        return idom

    def dominates(self, a: str, b: str) -> bool:
        """Does block a dominate block b?"""
        if not self._computed:
            self.compute_dominators()
        if a == b:
            return True
        current = b
        visited: Set[str] = set()
        while current in self._idom and current not in visited:
            visited.add(current)
            if self._idom[current] == a:
                return True
            if self._idom[current] == current:
                return False
            current = self._idom[current]
        return False

    def strictly_dominates(self, a: str, b: str) -> bool:
        return a != b and self.dominates(a, b)

    def get_dominator_tree(self) -> Dict[str, Optional[str]]:
        if not self._computed:
            self.compute_dominators()
        result: Dict[str, Optional[str]] = {}
        for bid in self._idom:
            idom = self._idom[bid]
            result[bid] = None if idom == bid else idom
        return result

    def get_dom_children(self, block_id: str) -> List[str]:
        if not self._computed:
            self.compute_dominators()
        return self._dom_children.get(block_id, [])

    def get_all_dominated(self, block_id: str) -> Set[str]:
        """Get all blocks dominated by block_id (including itself)."""
        if not self._computed:
            self.compute_dominators()
        result: Set[str] = {block_id}
        stack = list(self.get_dom_children(block_id))
        while stack:
            n = stack.pop()
            result.add(n)
            stack.extend(self.get_dom_children(n))
        return result

    def get_dom_depth(self, block_id: str) -> int:
        if not self._computed:
            self.compute_dominators()
        depth = 0
        current = block_id
        visited: Set[str] = set()
        while current in self._idom and current not in visited:
            visited.add(current)
            if self._idom[current] == current:
                break
            current = self._idom[current]
            depth += 1
        return depth

    def lca(self, a: str, b: str) -> Optional[str]:
        """Lowest common ancestor in the dominator tree."""
        if not self._computed:
            self.compute_dominators()
        ancestors_a: Set[str] = set()
        current = a
        visited: Set[str] = set()
        while current in self._idom and current not in visited:
            visited.add(current)
            ancestors_a.add(current)
            if self._idom[current] == current:
                break
            current = self._idom[current]
        current = b
        visited2: Set[str] = set()
        while current in self._idom and current not in visited2:
            visited2.add(current)
            if current in ancestors_a:
                return current
            if self._idom[current] == current:
                break
            current = self._idom[current]
        return None


# ===================================================================
# DominanceFrontier
# ===================================================================

class DominanceFrontier:
    """Dominance frontier computation."""

    def __init__(self, cfg: CFG, dom_tree: DominatorTree) -> None:
        self.cfg = cfg
        self.dom_tree = dom_tree
        self._frontier: Dict[str, Set[str]] = {}
        self._computed = False

    def _compute(self) -> None:
        if not self.dom_tree._computed:
            self.dom_tree.compute_dominators()
        self._frontier = {bid: set() for bid in self.cfg.blocks}

        for bid in self.cfg.blocks:
            blk = self.cfg.blocks[bid]
            preds = blk.predecessors
            if len(preds) >= 2:
                idom_bid = self.dom_tree.immediate_dominator(bid)
                for pred in preds:
                    runner = pred
                    visited: Set[str] = set()
                    while runner is not None and runner != idom_bid and runner not in visited:
                        visited.add(runner)
                        self._frontier[runner].add(bid)
                        runner = self.dom_tree.immediate_dominator(runner)
        self._computed = True

    def get_dominance_frontier(self, block_id: str) -> Set[str]:
        if not self._computed:
            self._compute()
        return self._frontier.get(block_id, set())

    def get_iterated_dominance_frontier(self, block_ids: Set[str]) -> Set[str]:
        """Compute iterated dominance frontier (IDF) for a set of blocks."""
        if not self._computed:
            self._compute()
        idf: Set[str] = set()
        worklist = list(block_ids)
        visited: Set[str] = set(block_ids)
        while worklist:
            bid = worklist.pop()
            for df_bid in self._frontier.get(bid, set()):
                if df_bid not in idf:
                    idf.add(df_bid)
                    if df_bid not in visited:
                        visited.add(df_bid)
                        worklist.append(df_bid)
        return idf


# ===================================================================
# PostDominatorTree
# ===================================================================

class PostDominatorTree:
    """Post-dominator tree (reverse dominators from exit)."""

    def __init__(self, cfg: CFG) -> None:
        self.cfg = cfg
        self._idom: Dict[str, str] = {}
        self._pdom_children: Dict[str, List[str]] = {}
        self._computed = False

    def compute_post_dominators(self) -> None:
        """Compute post-dominators by running dominator algorithm on reverse CFG."""
        exit_id = self.cfg._exit_id
        if exit_id is None or exit_id not in self.cfg.blocks:
            self._computed = True
            return

        # Build reverse CFG
        rev_succs: Dict[str, List[str]] = {bid: [] for bid in self.cfg.blocks}
        for bid, blk in self.cfg.blocks.items():
            for succ in blk.successors:
                if succ in rev_succs:
                    rev_succs[succ].append(bid)

        vertex: List[str] = []
        semi: Dict[str, int] = {}
        parent: Dict[str, Optional[str]] = {}
        pred_map: Dict[str, List[str]] = {bid: [] for bid in self.cfg.blocks}
        bucket: Dict[str, Set[str]] = {bid: set() for bid in self.cfg.blocks}
        dom: Dict[str, str] = {}
        ancestor: Dict[str, Optional[str]] = {bid: None for bid in self.cfg.blocks}
        label: Dict[str, str] = {bid: bid for bid in self.cfg.blocks}

        dfn: Dict[str, int] = {}
        counter = [0]

        def dfs(v: str) -> None:
            dfn[v] = counter[0]
            semi[v] = counter[0]
            vertex.append(v)
            counter[0] += 1
            for w in rev_succs.get(v, []):
                if w not in dfn:
                    parent[w] = v
                    dfs(w)
                pred_map[w].append(v)

        parent[exit_id] = None
        dfs(exit_id)

        n = len(vertex)
        if n == 0:
            self._computed = True
            return

        def _compress(v: str) -> None:
            anc = ancestor[v]
            if anc is not None and ancestor[anc] is not None:
                _compress(anc)
                if semi[label[anc]] < semi[label[v]]:
                    label[v] = label[anc]
                ancestor[v] = ancestor[anc]

        def _eval(v: str) -> str:
            if ancestor[v] is None:
                return v
            _compress(v)
            return label[v]

        def _link(v: str, w: str) -> None:
            ancestor[w] = v

        for i in range(n - 1, 0, -1):
            w = vertex[i]
            for v in pred_map[w]:
                if v in dfn:
                    u = _eval(v)
                    if semi[u] < semi[w]:
                        semi[w] = semi[u]
            bucket[vertex[semi[w]]].add(w)
            p = parent[w]
            if p is not None:
                _link(p, w)
                for v in list(bucket[p]):
                    bucket[p].discard(v)
                    u = _eval(v)
                    dom[v] = u if semi[u] < semi[v] else p

        for i in range(1, n):
            w = vertex[i]
            if w in dom and dom[w] != vertex[semi[w]]:
                dom[w] = dom[dom[w]]

        self._idom = dom
        self._idom[exit_id] = exit_id

        self._pdom_children = {bid: [] for bid in self.cfg.blocks}
        for node, idom in self._idom.items():
            if node != idom:
                self._pdom_children.setdefault(idom, []).append(node)

        self._computed = True

    def immediate_post_dominator(self, block_id: str) -> Optional[str]:
        if not self._computed:
            self.compute_post_dominators()
        idom = self._idom.get(block_id)
        if idom == block_id:
            return None
        return idom

    def post_dominates(self, a: str, b: str) -> bool:
        """Does block a post-dominate block b?"""
        if not self._computed:
            self.compute_post_dominators()
        if a == b:
            return True
        current = b
        visited: Set[str] = set()
        while current in self._idom and current not in visited:
            visited.add(current)
            if self._idom[current] == a:
                return True
            if self._idom[current] == current:
                return False
            current = self._idom[current]
        return False

    def get_pdom_children(self, block_id: str) -> List[str]:
        if not self._computed:
            self.compute_post_dominators()
        return self._pdom_children.get(block_id, [])


# ===================================================================
# ControlDependence
# ===================================================================

class ControlDependence:
    """Control dependence from post-dominance frontier."""

    def __init__(self, cfg: CFG) -> None:
        self.cfg = cfg
        self._cd: Dict[str, Set[str]] = {}
        self._computed = False

    def compute(self) -> None:
        pdom = PostDominatorTree(self.cfg)
        pdom.compute_post_dominators()
        self._cd = {bid: set() for bid in self.cfg.blocks}

        # A is control-dependent on B iff B is in post-dominance frontier of A
        # We compute PDF analogously to dominance frontier
        pdf: Dict[str, Set[str]] = {bid: set() for bid in self.cfg.blocks}
        for bid in self.cfg.blocks:
            blk = self.cfg.blocks[bid]
            succs = blk.successors
            if len(succs) >= 2:
                ipdom = pdom.immediate_post_dominator(bid)
                for succ in succs:
                    runner = succ
                    visited: Set[str] = set()
                    while runner is not None and runner != ipdom and runner not in visited:
                        visited.add(runner)
                        pdf[runner].add(bid)
                        runner = pdom.immediate_post_dominator(runner)

        # Y is control-dependent on X if X is in PDF(Y)
        for y, xs in pdf.items():
            for x in xs:
                self._cd.setdefault(y, set()).add(x)

        self._computed = True

    def get_control_dependences(self, block_id: str) -> Set[str]:
        if not self._computed:
            self.compute()
        return self._cd.get(block_id, set())

    def is_control_dependent(self, block_id: str, on_block: str) -> bool:
        return on_block in self.get_control_dependences(block_id)

    def get_all_dependent_on(self, block_id: str) -> Set[str]:
        """Get all blocks that are control-dependent on block_id."""
        if not self._computed:
            self.compute()
        result: Set[str] = set()
        for bid, deps in self._cd.items():
            if block_id in deps:
                result.add(bid)
        return result


# ===================================================================
# Loop dataclasses and LoopInfo
# ===================================================================

@dataclass
class Loop:
    """Represents a natural loop."""
    header: str
    body: Set[str] = field(default_factory=set)
    back_edges: List[Tuple[str, str]] = field(default_factory=list)
    exit_blocks: Set[str] = field(default_factory=set)
    preheader: Optional[str] = None
    depth: int = 1
    parent: Optional[Loop] = None
    children: List[Loop] = field(default_factory=list)

    def contains_block(self, block_id: str) -> bool:
        return block_id in self.body

    def __repr__(self) -> str:
        return f"Loop(header={self.header!r}, body_size={len(self.body)}, depth={self.depth})"


@dataclass
class NaturalLoop:
    """Natural loop with header identification."""
    header: str
    body: Set[str]
    latch: str  # tail of back edge

    def contains(self, block_id: str) -> bool:
        return block_id in self.body

    def __repr__(self) -> str:
        return f"NaturalLoop(header={self.header!r}, latch={self.latch!r}, size={len(self.body)})"


@dataclass
class LoopNestTree:
    """Tree of nested loops."""
    root_loops: List[Loop] = field(default_factory=list)
    all_loops: List[Loop] = field(default_factory=list)
    _header_to_loop: Dict[str, Loop] = field(default_factory=dict)

    def get_loop_for_header(self, header: str) -> Optional[Loop]:
        return self._header_to_loop.get(header)

    def get_max_depth(self) -> int:
        if not self.all_loops:
            return 0
        return max(l.depth for l in self.all_loops)


class LoopInfo:
    """Loop analysis for a CFG."""

    def __init__(self, cfg: CFG) -> None:
        self.cfg = cfg
        self._loops: List[Loop] = []
        self._block_to_loop: Dict[str, Loop] = {}
        self._nest_tree: Optional[LoopNestTree] = None
        self._computed = False

    def identify_loops(self) -> List[Loop]:
        """Identify all natural loops."""
        if self._computed:
            return self._loops

        back_edges = self.cfg.get_back_edges()
        # group back edges by header
        header_to_tails: Dict[str, List[str]] = {}
        for tail, header in back_edges:
            header_to_tails.setdefault(header, []).append(tail)

        raw_loops: List[Loop] = []
        for header, tails in header_to_tails.items():
            body: Set[str] = {header}
            for tail in tails:
                body |= self.cfg.get_loop_body(header, tail)
            exits: Set[str] = set()
            for bid in body:
                blk = self.cfg.blocks.get(bid)
                if blk:
                    for s in blk.successors:
                        if s not in body:
                            exits.add(bid)
            loop = Loop(
                header=header,
                body=body,
                back_edges=[(t, header) for t in tails],
                exit_blocks=exits,
            )
            raw_loops.append(loop)

        # compute nesting
        raw_loops.sort(key=lambda l: len(l.body))
        for i, inner in enumerate(raw_loops):
            for j in range(i + 1, len(raw_loops)):
                outer = raw_loops[j]
                if inner.header != outer.header and inner.body <= outer.body:
                    inner.parent = outer
                    outer.children.append(inner)
                    break

        # compute depths
        for loop in raw_loops:
            depth = 1
            p = loop.parent
            while p is not None:
                depth += 1
                p = p.parent
            loop.depth = depth

        # preheader identification
        for loop in raw_loops:
            header_blk = self.cfg.blocks.get(loop.header)
            if header_blk:
                preds_outside = [p for p in header_blk.predecessors if p not in loop.body]
                if len(preds_outside) == 1:
                    loop.preheader = preds_outside[0]

        self._loops = raw_loops
        for loop in raw_loops:
            for bid in loop.body:
                if bid not in self._block_to_loop or self._block_to_loop[bid].depth < loop.depth:
                    self._block_to_loop[bid] = loop

        # annotate blocks
        for bid, blk in self.cfg.blocks.items():
            if bid in self._block_to_loop:
                blk.loop_depth = self._block_to_loop[bid].depth
            else:
                blk.loop_depth = 0
            blk.is_loop_header = any(l.header == bid for l in raw_loops)

        self._computed = True
        return self._loops

    def get_loop_for_block(self, block_id: str) -> Optional[Loop]:
        if not self._computed:
            self.identify_loops()
        return self._block_to_loop.get(block_id)

    def get_loop_depth(self, block_id: str) -> int:
        loop = self.get_loop_for_block(block_id)
        return loop.depth if loop else 0

    def get_innermost_loop(self, block_id: str) -> Optional[Loop]:
        return self.get_loop_for_block(block_id)

    def is_loop_invariant(self, instr: IRInstruction, loop: Loop) -> bool:
        """Check if an instruction is loop-invariant (all operands defined outside loop)."""
        for use in instr.get_uses():
            defined_in_loop = False
            for bid in loop.body:
                blk = self.cfg.blocks.get(bid)
                if blk:
                    for phi in blk.phi_nodes:
                        if phi.dest == use:
                            defined_in_loop = True
                            break
                    if not defined_in_loop:
                        for ins in blk.instructions:
                            if ins.dest == use:
                                defined_in_loop = True
                                break
                if defined_in_loop:
                    break
            if defined_in_loop:
                return False
        return True

    def get_induction_variables(self, loop: Loop) -> List[SSAVar]:
        """Find basic induction variables (variables incremented by a constant)."""
        ivs: List[SSAVar] = []
        header_blk = self.cfg.blocks.get(loop.header)
        if not header_blk:
            return ivs
        for phi in header_blk.phi_nodes:
            var = phi.dest
            for bid, incoming_var in phi.incoming.items():
                if bid in loop.body and bid != loop.header:
                    # check if incoming_var = var + const
                    blk = self.cfg.blocks.get(bid)
                    if blk:
                        for ins in blk.instructions:
                            if ins.dest == incoming_var and ins.opcode in ("add", "sub", "iadd", "isub"):
                                uses = ins.get_uses()
                                if len(uses) == 2:
                                    # one use should be var, other should be loop-invariant
                                    has_var = any(u.name == var.name for u in uses)
                                    has_const = any(
                                        self.is_loop_invariant(
                                            IRInstruction(opcode="use", operands=[u]),
                                            loop,
                                        )
                                        for u in uses
                                        if u.name != var.name
                                    )
                                    if has_var and has_const:
                                        ivs.append(var)
        return ivs

    def get_trip_count(self, loop: Loop) -> Optional[int]:
        """Try to determine trip count (returns None if unknown)."""
        # Simple heuristic: look at loop header condition
        header_blk = self.cfg.blocks.get(loop.header)
        if not header_blk or not header_blk.instructions:
            return None
        last = header_blk.last_instruction()
        if last and last.opcode in ("branch_if", "cond_br"):
            # Would need constant propagation info for real analysis
            pass
        return None

    def get_loop_nest_tree(self) -> LoopNestTree:
        if self._nest_tree is not None:
            return self._nest_tree
        if not self._computed:
            self.identify_loops()
        tree = LoopNestTree()
        tree.all_loops = list(self._loops)
        for loop in self._loops:
            tree._header_to_loop[loop.header] = loop
            if loop.parent is None:
                tree.root_loops.append(loop)
        self._nest_tree = tree
        return tree


# ===================================================================
# SSAConstructor
# ===================================================================

class SSAConstructor:
    """SSA construction: insert phi functions and rename variables."""

    def __init__(self, cfg: CFG) -> None:
        self.cfg = cfg
        self._dom_tree: Optional[DominatorTree] = None
        self._dom_frontier: Optional[DominanceFrontier] = None
        self._var_counter: Dict[str, int] = {}
        self._var_stack: Dict[str, List[int]] = {}
        self._defs_map: Dict[str, Set[str]] = {}  # var_name -> set of block_ids

    def _ensure_dom(self) -> None:
        if self._dom_tree is None:
            self._dom_tree = DominatorTree(self.cfg)
            self._dom_tree.compute_dominators()
        if self._dom_frontier is None:
            self._dom_frontier = DominanceFrontier(self.cfg, self._dom_tree)

    def _collect_defs(self) -> Dict[str, Set[str]]:
        """Collect all variables and the blocks that define them."""
        defs: Dict[str, Set[str]] = {}
        for bid, blk in self.cfg.blocks.items():
            for instr in blk.instructions:
                if instr.dest is not None:
                    defs.setdefault(instr.dest.name, set()).add(bid)
        return defs

    def insert_phi_functions(self) -> None:
        """Insert phi functions at iterated dominance frontier."""
        self._ensure_dom()
        assert self._dom_frontier is not None
        self._defs_map = self._collect_defs()

        for var_name, def_blocks in self._defs_map.items():
            idf = self._dom_frontier.get_iterated_dominance_frontier(def_blocks)
            for bid in idf:
                blk = self.cfg.blocks[bid]
                if blk.find_phi(var_name) is None:
                    phi = PhiNode(dest=SSAVar(var_name, 0))
                    for pred_id in blk.predecessors:
                        phi.add_incoming(pred_id, SSAVar(var_name, 0))
                    blk.add_phi(phi)

    def _fresh_version(self, name: str) -> int:
        c = self._var_counter.get(name, 0)
        self._var_counter[name] = c + 1
        return c

    def _push(self, name: str) -> int:
        ver = self._fresh_version(name)
        self._var_stack.setdefault(name, []).append(ver)
        return ver

    def _pop(self, name: str) -> None:
        stack = self._var_stack.get(name)
        if stack:
            stack.pop()

    def _current_version(self, name: str) -> int:
        stack = self._var_stack.get(name)
        if stack:
            return stack[-1]
        return self._push(name)

    def rename_variables(self) -> None:
        """Rename variables using dominator tree walk."""
        self._ensure_dom()
        assert self._dom_tree is not None
        self._var_counter = {}
        self._var_stack = {}

        entry = self.cfg._entry_id
        if entry is None:
            return

        self._rename_block(entry)

    def _rename_block(self, block_id: str) -> None:
        assert self._dom_tree is not None
        blk = self.cfg.blocks[block_id]
        pushed: List[str] = []

        # rename phi destinations
        for phi in blk.phi_nodes:
            ver = self._push(phi.dest.name)
            pushed.append(phi.dest.name)
            phi.dest = SSAVar(phi.dest.name, ver)

        # rename instructions
        for instr in blk.instructions:
            # rename uses first
            for i, op in enumerate(instr.operands):
                if isinstance(op, SSAVar):
                    instr.operands[i] = SSAVar(op.name, self._current_version(op.name))
            # rename defs
            if instr.dest is not None:
                ver = self._push(instr.dest.name)
                pushed.append(instr.dest.name)
                instr.dest = SSAVar(instr.dest.name, ver)

        # fill phi inputs in successors
        for succ_id in blk.successors:
            succ = self.cfg.blocks.get(succ_id)
            if succ:
                for phi in succ.phi_nodes:
                    name = phi.dest.name
                    ver = self._current_version(name)
                    phi.incoming[block_id] = SSAVar(name, ver)

        # recurse into dominator tree children
        for child_id in self._dom_tree.get_dom_children(block_id):
            self._rename_block(child_id)

        # pop
        for name in reversed(pushed):
            self._pop(name)

    def handle_gated_phi(self) -> None:
        """Insert gated phi nodes for exception/generator control flow.

        Gated phis track which predecessor path was taken.
        """
        for bid, blk in self.cfg.blocks.items():
            exc_preds = []
            for pred_id in blk.predecessors:
                for e in self.cfg.get_edges_from(pred_id):
                    if e.dst == bid and e.kind == EdgeKind.EXCEPTION:
                        exc_preds.append(pred_id)
                        break
            if exc_preds and len(blk.predecessors) > 1:
                for phi in blk.phi_nodes:
                    phi.dest.name = f"__gated_{phi.dest.name}"
                    phi.dest = SSAVar(phi.dest.name, phi.dest.version)

    def compute_liveness(self) -> Dict[str, Tuple[Set[SSAVar], Set[SSAVar]]]:
        """Compute live-in and live-out sets for each block.

        Returns dict: block_id -> (live_in, live_out)
        """
        live_in: Dict[str, Set[SSAVar]] = {bid: set() for bid in self.cfg.blocks}
        live_out: Dict[str, Set[SSAVar]] = {bid: set() for bid in self.cfg.blocks}

        # compute use/def per block
        block_use: Dict[str, Set[SSAVar]] = {}
        block_def: Dict[str, Set[SSAVar]] = {}
        for bid, blk in self.cfg.blocks.items():
            uses: Set[SSAVar] = set()
            defs: Set[SSAVar] = set()
            for phi in blk.phi_nodes:
                defs.add(phi.dest)
            for instr in blk.instructions:
                for u in instr.get_uses():
                    if u not in defs:
                        uses.add(u)
                for d in instr.get_defs():
                    defs.add(d)
            block_use[bid] = uses
            block_def[bid] = defs

        # iterative fixed-point
        changed = True
        while changed:
            changed = False
            for bid in self.cfg.blocks:
                new_out: Set[SSAVar] = set()
                for succ_id in self.cfg.blocks[bid].successors:
                    new_out |= live_in.get(succ_id, set())
                new_in = block_use[bid] | (new_out - block_def[bid])
                if new_in != live_in[bid] or new_out != live_out[bid]:
                    live_in[bid] = new_in
                    live_out[bid] = new_out
                    changed = True

        return {bid: (live_in[bid], live_out[bid]) for bid in self.cfg.blocks}

    def compute_live_ranges(self) -> Dict[SSAVar, Set[str]]:
        """Compute live range (set of blocks) for each SSA variable."""
        liveness = self.compute_liveness()
        ranges: Dict[SSAVar, Set[str]] = {}
        for bid, (li, lo) in liveness.items():
            for v in li | lo:
                ranges.setdefault(v, set()).add(bid)
        return ranges

    def validate_ssa_property(self) -> List[str]:
        """Validate SSA property: each variable defined exactly once."""
        errors: List[str] = []
        defs: Dict[SSAVar, List[str]] = {}
        for bid, blk in self.cfg.blocks.items():
            for phi in blk.phi_nodes:
                defs.setdefault(phi.dest, []).append(bid)
            for instr in blk.instructions:
                if instr.dest:
                    defs.setdefault(instr.dest, []).append(bid)
        for var, blocks in defs.items():
            if len(blocks) > 1:
                errors.append(f"Variable {var} defined in multiple blocks: {blocks}")
        return errors

    def construct(self) -> None:
        """Full SSA construction: insert phis then rename."""
        self.insert_phi_functions()
        self.rename_variables()


# ===================================================================
# SSADestructor
# ===================================================================

class SSADestructor:
    """SSA destruction for code generation."""

    def __init__(self, cfg: CFG) -> None:
        self.cfg = cfg

    def critical_edge_splitting(self) -> int:
        """Split all critical edges (src has multiple succs, dst has multiple preds)."""
        count = 0
        edges_to_split: List[Tuple[str, str]] = []
        for e in self.cfg.edges:
            src = self.cfg.blocks.get(e.src)
            dst = self.cfg.blocks.get(e.dst)
            if src and dst and len(src.successors) > 1 and len(dst.predecessors) > 1:
                edges_to_split.append((e.src, e.dst))
        for src, dst in edges_to_split:
            if self.cfg.has_edge(src, dst):
                self.cfg.split_edge(src, dst)
                count += 1
        return count

    def insert_copies(self) -> int:
        """Replace phi nodes with copy instructions at end of predecessor blocks."""
        count = 0
        for bid, blk in list(self.cfg.blocks.items()):
            for phi in list(blk.phi_nodes):
                for pred_id, src_var in phi.incoming.items():
                    pred_blk = self.cfg.blocks.get(pred_id)
                    if pred_blk:
                        copy_instr = IRInstruction(
                            opcode="copy",
                            dest=SSAVar(phi.dest.name, phi.dest.version),
                            operands=[src_var],
                        )
                        # insert before terminator
                        if pred_blk.instructions and pred_blk.instructions[-1].opcode in (
                            "branch", "cond_br", "branch_if", "return", "switch",
                        ):
                            pred_blk.insert_instruction(len(pred_blk.instructions) - 1, copy_instr)
                        else:
                            pred_blk.add_instruction(copy_instr)
                        count += 1
            blk.phi_nodes.clear()
        return count

    def coalesce_copies(self) -> int:
        """Coalesce copy instructions where possible."""
        count = 0
        for bid, blk in self.cfg.blocks.items():
            i = 0
            while i < len(blk.instructions):
                instr = blk.instructions[i]
                if instr.opcode == "copy" and instr.dest and len(instr.operands) == 1:
                    src = instr.operands[0]
                    if isinstance(src, SSAVar) and src == instr.dest:
                        blk.instructions.pop(i)
                        count += 1
                        continue
                i += 1
        return count

    def destruct(self) -> None:
        """Full SSA destruction."""
        self.critical_edge_splitting()
        self.insert_copies()
        self.coalesce_copies()


# ===================================================================
# CFGBuilder
# ===================================================================

class CFGBuilder:
    """Build a CFG from high-level constructs."""

    def __init__(self) -> None:
        self.cfg = CFG()
        self._block_counter = 0
        self._break_targets: List[str] = []
        self._continue_targets: List[str] = []

    def _new_block(self, label: Optional[str] = None) -> BasicBlock:
        self._block_counter += 1
        bid = f"bb_{self._block_counter}"
        blk = BasicBlock(bid, label or bid)
        self.cfg.add_block(blk)
        return blk

    def _connect(self, src: BasicBlock, dst: BasicBlock, kind: EdgeKind = EdgeKind.NORMAL) -> None:
        self.cfg.add_edge(src.block_id, dst.block_id, kind)

    def build_empty(self, name: str = "func") -> CFG:
        """Build minimal CFG with entry and exit."""
        self.cfg.name = name
        entry = self._new_block("entry")
        exit_blk = self._new_block("exit")
        self.cfg.set_entry(entry.block_id)
        self.cfg.set_exit(exit_blk.block_id)
        self._connect(entry, exit_blk)
        return self.cfg

    def build_sequence(self, instructions: List[IRInstruction]) -> Tuple[BasicBlock, BasicBlock]:
        """Build a linear sequence in a single block."""
        blk = self._new_block("seq")
        for instr in instructions:
            blk.add_instruction(instr)
        return blk, blk

    def build_branch(
        self,
        cond_instr: IRInstruction,
        true_instrs: List[IRInstruction],
        false_instrs: List[IRInstruction],
    ) -> Tuple[BasicBlock, BasicBlock]:
        """Build an if-then-else diamond."""
        cond_blk = self._new_block("cond")
        cond_blk.add_instruction(cond_instr)
        true_blk = self._new_block("then")
        for instr in true_instrs:
            true_blk.add_instruction(instr)
        false_blk = self._new_block("else")
        for instr in false_instrs:
            false_blk.add_instruction(instr)
        merge_blk = self._new_block("merge")
        self._connect(cond_blk, true_blk, EdgeKind.TRUE_BRANCH)
        self._connect(cond_blk, false_blk, EdgeKind.FALSE_BRANCH)
        self._connect(true_blk, merge_blk)
        self._connect(false_blk, merge_blk)
        return cond_blk, merge_blk

    def build_while_loop(
        self,
        cond_instr: IRInstruction,
        body_instrs: List[IRInstruction],
    ) -> Tuple[BasicBlock, BasicBlock]:
        """Build a while loop."""
        header = self._new_block("loop_header")
        header.add_instruction(cond_instr)
        body = self._new_block("loop_body")
        for instr in body_instrs:
            body.add_instruction(instr)
        exit_blk = self._new_block("loop_exit")

        self._connect(header, body, EdgeKind.TRUE_BRANCH)
        self._connect(header, exit_blk, EdgeKind.FALSE_BRANCH)
        self._connect(body, header, EdgeKind.BACK)

        header.is_loop_header = True
        return header, exit_blk

    def build_for_loop(
        self,
        init_instrs: List[IRInstruction],
        cond_instr: IRInstruction,
        update_instrs: List[IRInstruction],
        body_instrs: List[IRInstruction],
    ) -> Tuple[BasicBlock, BasicBlock]:
        """Build a for loop (init; cond; update) { body }."""
        init_blk = self._new_block("for_init")
        for instr in init_instrs:
            init_blk.add_instruction(instr)

        header = self._new_block("for_header")
        header.add_instruction(cond_instr)

        body = self._new_block("for_body")
        for instr in body_instrs:
            body.add_instruction(instr)

        update = self._new_block("for_update")
        for instr in update_instrs:
            update.add_instruction(instr)

        exit_blk = self._new_block("for_exit")

        self._connect(init_blk, header)
        self._connect(header, body, EdgeKind.TRUE_BRANCH)
        self._connect(header, exit_blk, EdgeKind.FALSE_BRANCH)
        self._connect(body, update)
        self._connect(update, header, EdgeKind.BACK)

        header.is_loop_header = True
        return init_blk, exit_blk

    def build_with_break_continue(
        self,
        cond_instr: IRInstruction,
        body_instrs: List[IRInstruction],
        has_break: bool = False,
        has_continue: bool = False,
    ) -> Tuple[BasicBlock, BasicBlock]:
        """Build loop with break/continue support."""
        header = self._new_block("loop_header")
        header.add_instruction(cond_instr)
        body = self._new_block("loop_body")
        exit_blk = self._new_block("loop_exit")

        self._break_targets.append(exit_blk.block_id)
        self._continue_targets.append(header.block_id)

        for instr in body_instrs:
            if instr.opcode == "break" and has_break:
                break_blk = self._new_block("break")
                self._connect(break_blk, exit_blk, EdgeKind.BREAK)
            elif instr.opcode == "continue" and has_continue:
                cont_blk = self._new_block("continue")
                self._connect(cont_blk, header, EdgeKind.CONTINUE)
            else:
                body.add_instruction(instr)

        self._connect(header, body, EdgeKind.TRUE_BRANCH)
        self._connect(header, exit_blk, EdgeKind.FALSE_BRANCH)
        self._connect(body, header, EdgeKind.BACK)

        self._break_targets.pop()
        self._continue_targets.pop()

        header.is_loop_header = True
        return header, exit_blk

    def build_switch(
        self,
        switch_instr: IRInstruction,
        cases: List[Tuple[Any, List[IRInstruction]]],
        default_instrs: Optional[List[IRInstruction]] = None,
    ) -> Tuple[BasicBlock, BasicBlock]:
        """Build a switch/match construct."""
        switch_blk = self._new_block("switch")
        switch_blk.add_instruction(switch_instr)
        merge_blk = self._new_block("switch_merge")

        for i, (case_val, case_instrs) in enumerate(cases):
            case_blk = self._new_block(f"case_{i}")
            for instr in case_instrs:
                case_blk.add_instruction(instr)
            self._connect(switch_blk, case_blk)
            self._connect(case_blk, merge_blk)

        if default_instrs is not None:
            default_blk = self._new_block("default")
            for instr in default_instrs:
                default_blk.add_instruction(instr)
            self._connect(switch_blk, default_blk)
            self._connect(default_blk, merge_blk)

        return switch_blk, merge_blk

    def build_try_except(
        self,
        try_instrs: List[IRInstruction],
        except_instrs: List[IRInstruction],
        finally_instrs: Optional[List[IRInstruction]] = None,
    ) -> Tuple[BasicBlock, BasicBlock]:
        """Build try/except/finally construct."""
        try_blk = self._new_block("try")
        for instr in try_instrs:
            try_blk.add_instruction(instr)

        except_blk = self._new_block("except")
        for instr in except_instrs:
            except_blk.add_instruction(instr)

        merge_blk = self._new_block("try_merge")

        self._connect(try_blk, except_blk, EdgeKind.EXCEPTION)
        self._connect(try_blk, merge_blk)
        self._connect(except_blk, merge_blk)

        if finally_instrs:
            finally_blk = self._new_block("finally")
            for instr in finally_instrs:
                finally_blk.add_instruction(instr)
            old_merge = merge_blk
            merge_blk = self._new_block("finally_merge")
            self._connect(old_merge, finally_blk)
            self._connect(finally_blk, merge_blk)

        return try_blk, merge_blk

    def build_function(
        self,
        name: str,
        params: List[str],
        body_builder: Callable[[CFGBuilder, BasicBlock], BasicBlock],
    ) -> CFG:
        """Build a function CFG."""
        self.cfg.name = name
        entry = self._new_block("entry")
        exit_blk = self._new_block("exit")
        self.cfg.set_entry(entry.block_id)
        self.cfg.set_exit(exit_blk.block_id)

        # parameter assignments
        for i, param in enumerate(params):
            entry.add_instruction(IRInstruction(
                opcode="param",
                dest=SSAVar(param, 0),
                operands=[i],
            ))

        last_blk = body_builder(self, entry)
        if last_blk.block_id != exit_blk.block_id:
            self._connect(last_blk, exit_blk)

        return self.cfg

    def finalize(self) -> CFG:
        """Finalize and return the CFG."""
        if self.cfg._entry_id is None and self.cfg.blocks:
            first = next(iter(self.cfg.blocks.values()))
            self.cfg.set_entry(first.block_id)
        return self.cfg


# ===================================================================
# CFGOptimizer
# ===================================================================

class CFGOptimizer:
    """CFG optimization passes."""

    def __init__(self, cfg: CFG) -> None:
        self.cfg = cfg

    def dead_block_elimination(self) -> int:
        """Remove blocks not reachable from entry."""
        reachable = set(self.cfg.dfs_preorder())
        dead = [bid for bid in self.cfg.blocks if bid not in reachable]
        for bid in dead:
            self.cfg.remove_block(bid)
        return len(dead)

    def unreachable_code_elimination(self) -> int:
        """Remove instructions after unconditional jumps/returns."""
        count = 0
        for bid, blk in self.cfg.blocks.items():
            found_term = False
            i = 0
            while i < len(blk.instructions):
                if found_term:
                    blk.instructions.pop(i)
                    count += 1
                else:
                    if blk.instructions[i].opcode in ("return", "unreachable", "throw"):
                        found_term = True
                    i += 1
        return count

    def block_merging(self) -> int:
        """Merge chains of single-entry/single-exit blocks."""
        return self.cfg.simplify()

    def critical_edge_splitting(self) -> int:
        """Split all critical edges."""
        destructor = SSADestructor(self.cfg)
        return destructor.critical_edge_splitting()

    def edge_profiling_instrumentation(self) -> int:
        """Insert edge profiling counters (instrumentation)."""
        count = 0
        for e in list(self.cfg.edges):
            src_blk = self.cfg.blocks.get(e.src)
            if src_blk:
                counter_instr = IRInstruction(
                    opcode="edge_count",
                    operands=[e.src, e.dst],
                    metadata={"edge_kind": e.kind.value},
                )
                if src_blk.instructions and src_blk.instructions[-1].opcode in (
                    "branch", "cond_br", "branch_if", "return", "switch",
                ):
                    src_blk.insert_instruction(len(src_blk.instructions) - 1, counter_instr)
                else:
                    src_blk.add_instruction(counter_instr)
                count += 1
        return count

    def run_all(self) -> Dict[str, int]:
        """Run all optimization passes."""
        results: Dict[str, int] = {}
        results["dead_blocks"] = self.dead_block_elimination()
        results["unreachable_code"] = self.unreachable_code_elimination()
        results["block_merging"] = self.block_merging()
        return results


# ===================================================================
# CFGVerifier
# ===================================================================

class CFGVerifier:
    """Verify CFG invariants."""

    def __init__(self, cfg: CFG) -> None:
        self.cfg = cfg

    def verify_all(self) -> List[str]:
        """Run all verification checks."""
        errors: List[str] = []
        errors.extend(self.check_entry_no_preds())
        errors.extend(self.check_exit_no_succs())
        errors.extend(self.check_reachability())
        errors.extend(self.check_exit_reachability())
        errors.extend(self.check_ssa_property())
        errors.extend(self.check_phi_at_entry())
        errors.extend(self.check_domination_property())
        return errors

    def check_entry_no_preds(self) -> List[str]:
        entry = self.cfg.get_entry()
        if entry and entry.predecessors:
            return [f"Entry block {entry.block_id} has predecessors: {entry.predecessors}"]
        return []

    def check_exit_no_succs(self) -> List[str]:
        exit_blk = self.cfg.get_exit()
        if exit_blk and exit_blk.successors:
            return [f"Exit block {exit_blk.block_id} has successors: {exit_blk.successors}"]
        return []

    def check_reachability(self) -> List[str]:
        errors: List[str] = []
        reachable = set(self.cfg.dfs_preorder())
        for bid in self.cfg.blocks:
            if bid not in reachable:
                errors.append(f"Block {bid} not reachable from entry")
        return errors

    def check_exit_reachability(self) -> List[str]:
        """Check all blocks can reach exit (modulo infinite loops)."""
        errors: List[str] = []
        exit_id = self.cfg._exit_id
        if exit_id is None:
            return ["No exit block"]
        # BFS backward from exit
        rev_reachable: Set[str] = set()
        from collections import deque
        q: deque[str] = deque([exit_id])
        rev_reachable.add(exit_id)
        while q:
            bid = q.popleft()
            for pred in self.cfg.get_predecessors(bid):
                if pred not in rev_reachable:
                    rev_reachable.add(pred)
                    q.append(pred)
        # blocks in loops might not reach exit
        loop_info = LoopInfo(self.cfg)
        loop_info.identify_loops()
        for bid in self.cfg.blocks:
            if bid not in rev_reachable:
                loop = loop_info.get_loop_for_block(bid)
                if loop is None:
                    errors.append(f"Block {bid} cannot reach exit (not in a loop)")
        return errors

    def check_ssa_property(self) -> List[str]:
        """Check each SSA variable defined exactly once."""
        defs: Dict[str, List[str]] = {}
        for bid, blk in self.cfg.blocks.items():
            for phi in blk.phi_nodes:
                key = str(phi.dest)
                defs.setdefault(key, []).append(bid)
            for instr in blk.instructions:
                if instr.dest:
                    key = str(instr.dest)
                    defs.setdefault(key, []).append(bid)
        errors: List[str] = []
        for var, blocks in defs.items():
            if len(blocks) > 1:
                errors.append(f"SSA variable {var} defined in multiple blocks: {blocks}")
        return errors

    def check_phi_at_entry(self) -> List[str]:
        """Phi nodes should only appear at block entry (they're already modeled that way)."""
        # In our model, phi_nodes is separate from instructions, so this is structural.
        return []

    def check_domination_property(self) -> List[str]:
        """Check that every use is dominated by its definition."""
        errors: List[str] = []
        dom = DominatorTree(self.cfg)
        dom.compute_dominators()

        # build def map: SSAVar -> block_id
        def_map: Dict[str, str] = {}
        for bid, blk in self.cfg.blocks.items():
            for phi in blk.phi_nodes:
                def_map[str(phi.dest)] = bid
            for instr in blk.instructions:
                if instr.dest:
                    def_map[str(instr.dest)] = bid

        # check uses
        for bid, blk in self.cfg.blocks.items():
            for instr in blk.instructions:
                for use in instr.get_uses():
                    key = str(use)
                    if key in def_map:
                        def_block = def_map[key]
                        if not dom.dominates(def_block, bid):
                            errors.append(
                                f"Use of {key} in {bid} not dominated by def in {def_block}"
                            )
            # phi uses: each incoming should be dominated in the predecessor
            for phi in blk.phi_nodes:
                for pred_id, src in phi.incoming.items():
                    key = str(src)
                    if key in def_map:
                        def_block = def_map[key]
                        if not dom.dominates(def_block, pred_id):
                            errors.append(
                                f"Phi use of {key} from {pred_id} in {bid} not dominated by def in {def_block}"
                            )

        return errors


# ===================================================================
# StrongComponentDecomposition (Tarjan's SCC)
# ===================================================================

class StrongComponentDecomposition:
    """Tarjan's strongly connected components algorithm."""

    def __init__(self, cfg: CFG) -> None:
        self.cfg = cfg
        self._sccs: List[Set[str]] = []
        self._scc_dag: Dict[int, Set[int]] = {}
        self._block_to_scc: Dict[str, int] = {}
        self._computed = False

    def find_sccs(self) -> List[Set[str]]:
        """Find all SCCs using Tarjan's algorithm."""
        if self._computed:
            return self._sccs

        index_counter = [0]
        stack: List[str] = []
        on_stack: Set[str] = set()
        index: Dict[str, int] = {}
        lowlink: Dict[str, int] = {}
        sccs: List[Set[str]] = []

        def strongconnect(v: str) -> None:
            index[v] = index_counter[0]
            lowlink[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack.add(v)

            blk = self.cfg.blocks.get(v)
            if blk:
                for w in blk.successors:
                    if w not in index:
                        strongconnect(w)
                        lowlink[v] = min(lowlink[v], lowlink[w])
                    elif w in on_stack:
                        lowlink[v] = min(lowlink[v], index[w])

            if lowlink[v] == index[v]:
                scc: Set[str] = set()
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    scc.add(w)
                    if w == v:
                        break
                sccs.append(scc)

        for bid in self.cfg.blocks:
            if bid not in index:
                strongconnect(bid)

        self._sccs = sccs

        for i, scc in enumerate(sccs):
            for bid in scc:
                self._block_to_scc[bid] = i

        self._computed = True
        return sccs

    def get_scc_dag(self) -> Dict[int, Set[int]]:
        """Get DAG of SCCs."""
        if not self._computed:
            self.find_sccs()
        if self._scc_dag:
            return self._scc_dag

        dag: Dict[int, Set[int]] = {i: set() for i in range(len(self._sccs))}
        for bid, blk in self.cfg.blocks.items():
            src_scc = self._block_to_scc.get(bid)
            if src_scc is None:
                continue
            for succ in blk.successors:
                dst_scc = self._block_to_scc.get(succ)
                if dst_scc is not None and dst_scc != src_scc:
                    dag[src_scc].add(dst_scc)

        self._scc_dag = dag
        return dag

    def topological_sort(self) -> List[Set[str]]:
        """Topological sort of SCCs (reverse of finish order)."""
        if not self._computed:
            self.find_sccs()
        # SCCs from Tarjan's are already in reverse topological order
        return list(reversed(self._sccs))

    def get_scc_for_block(self, block_id: str) -> Optional[Set[str]]:
        if not self._computed:
            self.find_sccs()
        idx = self._block_to_scc.get(block_id)
        if idx is not None:
            return self._sccs[idx]
        return None

    def is_in_cycle(self, block_id: str) -> bool:
        scc = self.get_scc_for_block(block_id)
        if scc is None:
            return False
        if len(scc) > 1:
            return True
        # single-node SCC: check self-loop
        blk = self.cfg.blocks.get(block_id)
        return blk is not None and block_id in blk.successors


# ===================================================================
# CFGDiff
# ===================================================================

@dataclass
class CFGDiffResult:
    """Result of diffing two CFGs."""
    added_blocks: Set[str] = field(default_factory=set)
    removed_blocks: Set[str] = field(default_factory=set)
    modified_blocks: Set[str] = field(default_factory=set)
    added_edges: List[Tuple[str, str]] = field(default_factory=list)
    removed_edges: List[Tuple[str, str]] = field(default_factory=list)

    def has_changes(self) -> bool:
        return bool(
            self.added_blocks or self.removed_blocks or self.modified_blocks
            or self.added_edges or self.removed_edges
        )

    def summary(self) -> str:
        parts: List[str] = []
        if self.added_blocks:
            parts.append(f"+{len(self.added_blocks)} blocks")
        if self.removed_blocks:
            parts.append(f"-{len(self.removed_blocks)} blocks")
        if self.modified_blocks:
            parts.append(f"~{len(self.modified_blocks)} blocks")
        if self.added_edges:
            parts.append(f"+{len(self.added_edges)} edges")
        if self.removed_edges:
            parts.append(f"-{len(self.removed_edges)} edges")
        return ", ".join(parts) if parts else "no changes"


class CFGDiff:
    """Diff two CFGs for incremental analysis."""

    @staticmethod
    def diff(old: CFG, new: CFG) -> CFGDiffResult:
        result = CFGDiffResult()
        old_ids = set(old.blocks.keys())
        new_ids = set(new.blocks.keys())
        result.added_blocks = new_ids - old_ids
        result.removed_blocks = old_ids - new_ids
        common = old_ids & new_ids
        for bid in common:
            old_blk = old.blocks[bid]
            new_blk = new.blocks[bid]
            if (
                len(old_blk.instructions) != len(new_blk.instructions)
                or len(old_blk.phi_nodes) != len(new_blk.phi_nodes)
                or old_blk.successors != new_blk.successors
                or old_blk.predecessors != new_blk.predecessors
            ):
                result.modified_blocks.add(bid)
            else:
                for oi, ni in zip(old_blk.instructions, new_blk.instructions):
                    if str(oi) != str(ni):
                        result.modified_blocks.add(bid)
                        break

        old_edge_set = {(e.src, e.dst) for e in old.edges}
        new_edge_set = {(e.src, e.dst) for e in new.edges}
        result.added_edges = list(new_edge_set - old_edge_set)
        result.removed_edges = list(old_edge_set - new_edge_set)
        return result

    @staticmethod
    def get_affected_blocks(result: CFGDiffResult, cfg: CFG) -> Set[str]:
        """Get blocks affected by changes (including dominated blocks)."""
        affected = result.added_blocks | result.modified_blocks
        for src, dst in result.added_edges + result.removed_edges:
            affected.add(src)
            affected.add(dst)
        # add blocks dominated by modified blocks
        if affected:
            dom = DominatorTree(cfg)
            dom.compute_dominators()
            expanded: Set[str] = set()
            for bid in affected:
                if bid in cfg.blocks:
                    expanded |= dom.get_all_dominated(bid)
            affected |= expanded
        return affected


# ===================================================================
# CFGSerializer
# ===================================================================

class CFGSerializer:
    """Serialize/deserialize CFG to/from JSON."""

    SCHEMA_VERSION = 1

    @staticmethod
    def to_json(cfg: CFG) -> str:
        data = CFGSerializer._to_dict(cfg)
        return json.dumps(data, indent=2)

    @staticmethod
    def from_json(json_str: str) -> CFG:
        data = json.loads(json_str)
        return CFGSerializer._from_dict(data)

    @staticmethod
    def _to_dict(cfg: CFG) -> Dict[str, Any]:
        blocks_data: List[Dict[str, Any]] = []
        for bid, blk in cfg.blocks.items():
            b: Dict[str, Any] = {
                "id": bid,
                "label": blk.label,
                "predecessors": blk.predecessors,
                "successors": blk.successors,
                "loop_depth": blk.loop_depth,
                "is_loop_header": blk.is_loop_header,
                "is_exit": blk.is_exit,
            }
            if blk.source_range:
                b["source_range"] = {
                    "file": blk.source_range.file,
                    "start_line": blk.source_range.start_line,
                    "start_col": blk.source_range.start_col,
                    "end_line": blk.source_range.end_line,
                    "end_col": blk.source_range.end_col,
                }
            b["phi_nodes"] = [
                {
                    "dest": {"name": p.dest.name, "version": p.dest.version},
                    "incoming": {
                        k: {"name": v.name, "version": v.version}
                        for k, v in p.incoming.items()
                    },
                }
                for p in blk.phi_nodes
            ]
            b["instructions"] = [
                {
                    "opcode": ins.opcode,
                    "dest": {"name": ins.dest.name, "version": ins.dest.version} if ins.dest else None,
                    "operands": [
                        {"name": op.name, "version": op.version} if isinstance(op, SSAVar) else op
                        for op in ins.operands
                    ],
                    "metadata": ins.metadata,
                }
                for ins in blk.instructions
            ]
            blocks_data.append(b)

        edges_data = [
            {"src": e.src, "dst": e.dst, "kind": e.kind.value, "weight": e.weight}
            for e in cfg.edges
        ]

        return {
            "schema_version": CFGSerializer.SCHEMA_VERSION,
            "name": cfg.name,
            "entry": cfg._entry_id,
            "exit": cfg._exit_id,
            "blocks": blocks_data,
            "edges": edges_data,
            "metadata": cfg.metadata,
        }

    @staticmethod
    def _from_dict(data: Dict[str, Any]) -> CFG:
        version = data.get("schema_version", 1)
        if version != CFGSerializer.SCHEMA_VERSION:
            pass  # forward compat

        cfg = CFG(data.get("name", ""))
        cfg._entry_id = data.get("entry")
        cfg._exit_id = data.get("exit")
        cfg.metadata = data.get("metadata", {})

        for b in data.get("blocks", []):
            blk = BasicBlock(b["id"], b.get("label", b["id"]))
            blk.predecessors = b.get("predecessors", [])
            blk.successors = b.get("successors", [])
            blk.loop_depth = b.get("loop_depth", 0)
            blk.is_loop_header = b.get("is_loop_header", False)
            blk.is_exit = b.get("is_exit", False)
            sr = b.get("source_range")
            if sr:
                blk.source_range = SourceRange(**sr)

            for p in b.get("phi_nodes", []):
                dest = SSAVar(p["dest"]["name"], p["dest"]["version"])
                phi = PhiNode(dest=dest)
                for k, v in p.get("incoming", {}).items():
                    phi.incoming[k] = SSAVar(v["name"], v["version"])
                blk.phi_nodes.append(phi)

            for ins_data in b.get("instructions", []):
                dest = None
                if ins_data.get("dest"):
                    dest = SSAVar(ins_data["dest"]["name"], ins_data["dest"]["version"])
                operands = []
                for op in ins_data.get("operands", []):
                    if isinstance(op, dict) and "name" in op:
                        operands.append(SSAVar(op["name"], op["version"]))
                    else:
                        operands.append(op)
                instr = IRInstruction(
                    opcode=ins_data.get("opcode", ""),
                    dest=dest,
                    operands=operands,
                    metadata=ins_data.get("metadata", {}),
                )
                blk.instructions.append(instr)

            cfg.blocks[blk.block_id] = blk

        kind_map = {k.value: k for k in EdgeKind}
        for e in data.get("edges", []):
            edge = CFGEdge(
                src=e["src"],
                dst=e["dst"],
                kind=kind_map.get(e.get("kind", "normal"), EdgeKind.NORMAL),
                weight=e.get("weight", 1.0),
            )
            cfg.edges.append(edge)

        return cfg

    @staticmethod
    def to_file(cfg: CFG, path: str) -> None:
        with open(path, "w") as f:
            f.write(CFGSerializer.to_json(cfg))

    @staticmethod
    def from_file(path: str) -> CFG:
        with open(path, "r") as f:
            return CFGSerializer.from_json(f.read())


# ===================================================================
# Interval analysis helper
# ===================================================================

@dataclass
class Interval:
    """Interval in the CFG for interval analysis."""
    header: str
    nodes: Set[str] = field(default_factory=set)
    succ_intervals: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"Interval(header={self.header!r}, size={len(self.nodes)})"


class IntervalAnalysis:
    """Interval analysis for reducibility checking and loop detection."""

    def __init__(self, cfg: CFG) -> None:
        self.cfg = cfg

    def compute_intervals(self) -> List[Interval]:
        """Compute intervals of the CFG."""
        if not self.cfg._entry_id:
            return []

        intervals: List[Interval] = []
        headers: List[str] = [self.cfg._entry_id]
        in_interval: Set[str] = set()

        while headers:
            h = headers.pop(0)
            if h in in_interval:
                continue
            interval = Interval(header=h, nodes={h})
            in_interval.add(h)
            changed = True
            while changed:
                changed = False
                for bid in list(self.cfg.blocks.keys()):
                    if bid in in_interval:
                        continue
                    blk = self.cfg.blocks[bid]
                    if all(p in interval.nodes for p in blk.predecessors):
                        interval.nodes.add(bid)
                        in_interval.add(bid)
                        changed = True

            for bid in interval.nodes:
                blk = self.cfg.blocks[bid]
                for s in blk.successors:
                    if s not in interval.nodes and s not in in_interval:
                        headers.append(s)
                        interval.succ_intervals.append(s)

            intervals.append(interval)

        return intervals


# ===================================================================
# Extended CFG analyses
# ===================================================================

class ReachabilityAnalysis:
    """Reachability queries on CFG."""

    def __init__(self, cfg: CFG) -> None:
        self.cfg = cfg
        self._reachable: Dict[str, Set[str]] = {}
        self._computed = False

    def _compute(self) -> None:
        for bid in self.cfg.blocks:
            visited: Set[str] = set()
            stack = [bid]
            while stack:
                n = stack.pop()
                if n in visited:
                    continue
                visited.add(n)
                blk = self.cfg.blocks.get(n)
                if blk:
                    for s in blk.successors:
                        if s not in visited:
                            stack.append(s)
            self._reachable[bid] = visited
        self._computed = True

    def can_reach(self, src: str, dst: str) -> bool:
        if not self._computed:
            self._compute()
        return dst in self._reachable.get(src, set())

    def get_reachable(self, src: str) -> Set[str]:
        if not self._computed:
            self._compute()
        return self._reachable.get(src, set())

    def get_all_paths(self, src: str, dst: str, max_paths: int = 100) -> List[List[str]]:
        """Find all simple paths from src to dst (up to max_paths)."""
        paths: List[List[str]] = []

        def _dfs(current: str, target: str, path: List[str], visited: Set[str]) -> None:
            if len(paths) >= max_paths:
                return
            if current == target:
                paths.append(list(path))
                return
            blk = self.cfg.blocks.get(current)
            if not blk:
                return
            for s in blk.successors:
                if s not in visited:
                    visited.add(s)
                    path.append(s)
                    _dfs(s, target, path, visited)
                    path.pop()
                    visited.discard(s)

        _dfs(src, dst, [src], {src})
        return paths


class CFGMetrics:
    """Compute various metrics on CFG."""

    def __init__(self, cfg: CFG) -> None:
        self.cfg = cfg

    def cyclomatic_complexity(self) -> int:
        """McCabe cyclomatic complexity: E - N + 2P."""
        e = len(self.cfg.edges)
        n = len(self.cfg.blocks)
        p = 1  # single connected component
        return e - n + 2 * p

    def nesting_depth(self) -> int:
        loop_info = LoopInfo(self.cfg)
        loops = loop_info.identify_loops()
        if not loops:
            return 0
        return max(l.depth for l in loops)

    def block_count(self) -> int:
        return len(self.cfg.blocks)

    def edge_count(self) -> int:
        return len(self.cfg.edges)

    def instruction_count(self) -> int:
        total = 0
        for blk in self.cfg.blocks.values():
            total += len(blk.instructions) + len(blk.phi_nodes)
        return total

    def average_block_size(self) -> float:
        if not self.cfg.blocks:
            return 0.0
        total = sum(len(blk.instructions) for blk in self.cfg.blocks.values())
        return total / len(self.cfg.blocks)

    def phi_count(self) -> int:
        return sum(len(blk.phi_nodes) for blk in self.cfg.blocks.values())

    def critical_edge_count(self) -> int:
        count = 0
        for e in self.cfg.edges:
            src = self.cfg.blocks.get(e.src)
            dst = self.cfg.blocks.get(e.dst)
            if src and dst and len(src.successors) > 1 and len(dst.predecessors) > 1:
                count += 1
        return count

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.cfg.name,
            "blocks": self.block_count(),
            "edges": self.edge_count(),
            "instructions": self.instruction_count(),
            "phis": self.phi_count(),
            "cyclomatic_complexity": self.cyclomatic_complexity(),
            "nesting_depth": self.nesting_depth(),
            "avg_block_size": self.average_block_size(),
            "critical_edges": self.critical_edge_count(),
        }


# ===================================================================
# CFGPattern: pattern matching on CFG structure
# ===================================================================

class CFGPattern:
    """Pattern matching on CFG structure for analysis."""

    @staticmethod
    def find_diamond(cfg: CFG) -> List[Tuple[str, str, str, str]]:
        """Find diamond patterns: cond -> (A, B) -> merge."""
        diamonds: List[Tuple[str, str, str, str]] = []
        for bid, blk in cfg.blocks.items():
            if len(blk.successors) == 2:
                a, b = blk.successors[0], blk.successors[1]
                a_blk = cfg.blocks.get(a)
                b_blk = cfg.blocks.get(b)
                if a_blk and b_blk:
                    a_succs = set(a_blk.successors)
                    b_succs = set(b_blk.successors)
                    common = a_succs & b_succs
                    for merge in common:
                        diamonds.append((bid, a, b, merge))
        return diamonds

    @staticmethod
    def find_triangle(cfg: CFG) -> List[Tuple[str, str, str]]:
        """Find triangle patterns: cond -> A -> merge, cond -> merge."""
        triangles: List[Tuple[str, str, str]] = []
        for bid, blk in cfg.blocks.items():
            if len(blk.successors) == 2:
                for i in range(2):
                    side = blk.successors[i]
                    merge = blk.successors[1 - i]
                    side_blk = cfg.blocks.get(side)
                    if side_blk and merge in side_blk.successors:
                        triangles.append((bid, side, merge))
        return triangles

    @staticmethod
    def find_hammock(cfg: CFG) -> List[Tuple[str, str, Set[str]]]:
        """Find single-entry single-exit regions (hammocks)."""
        hammocks: List[Tuple[str, str, Set[str]]] = []
        dom = DominatorTree(cfg)
        dom.compute_dominators()
        pdom = PostDominatorTree(cfg)
        pdom.compute_post_dominators()

        for bid in cfg.blocks:
            ipdom = pdom.immediate_post_dominator(bid)
            if ipdom is not None and ipdom != bid:
                # region from bid to ipdom
                region: Set[str] = set()
                stack = [bid]
                while stack:
                    n = stack.pop()
                    if n in region or n == ipdom:
                        continue
                    region.add(n)
                    blk = cfg.blocks.get(n)
                    if blk:
                        for s in blk.successors:
                            stack.append(s)
                if len(region) > 1:
                    hammocks.append((bid, ipdom, region))
        return hammocks

    @staticmethod
    def find_self_loops(cfg: CFG) -> List[str]:
        """Find blocks that loop to themselves."""
        return [bid for bid, blk in cfg.blocks.items() if bid in blk.successors]


# ===================================================================
# WorklistSolver: generic worklist-based dataflow solver
# ===================================================================

T = TypeVar("T")


class WorklistSolver(Generic[T]):
    """Generic worklist-based iterative dataflow solver."""

    def __init__(
        self,
        cfg: CFG,
        direction: str = "forward",
        init: Callable[[str], T] = lambda _: None,  # type: ignore[assignment]
        transfer: Callable[[str, T], T] = lambda bid, val: val,
        meet: Callable[[List[T]], T] = lambda vals: vals[0] if vals else None,  # type: ignore[return-value]
        equal: Callable[[T, T], bool] = lambda a, b: a == b,
    ) -> None:
        self.cfg = cfg
        self.direction = direction
        self.init = init
        self.transfer = transfer
        self.meet = meet
        self.equal = equal

    def solve(self) -> Dict[str, T]:
        """Run worklist algorithm to fixed point."""
        from collections import deque

        result: Dict[str, T] = {}
        for bid in self.cfg.blocks:
            result[bid] = self.init(bid)

        if self.direction == "forward":
            order = self.cfg.reverse_postorder()
        else:
            order = self.cfg.dfs_postorder()

        worklist: deque[str] = deque(order)
        in_worklist: Set[str] = set(order)

        while worklist:
            bid = worklist.popleft()
            in_worklist.discard(bid)

            if self.direction == "forward":
                preds = self.cfg.get_predecessors(bid)
                pred_vals = [result[p] for p in preds if p in result]
            else:
                succs = self.cfg.get_successors(bid)
                pred_vals = [result[s] for s in succs if s in result]

            if pred_vals:
                meet_val = self.meet(pred_vals)
            else:
                meet_val = result[bid]

            new_val = self.transfer(bid, meet_val)

            if not self.equal(new_val, result[bid]):
                result[bid] = new_val
                if self.direction == "forward":
                    nexts = self.cfg.get_successors(bid)
                else:
                    nexts = self.cfg.get_predecessors(bid)
                for n in nexts:
                    if n not in in_worklist:
                        in_worklist.add(n)
                        worklist.append(n)

        return result


# ===================================================================
# LiveVariableAnalysis
# ===================================================================

class LiveVariableAnalysis:
    """Live variable analysis (backward dataflow)."""

    def __init__(self, cfg: CFG) -> None:
        self.cfg = cfg

    def analyze(self) -> Dict[str, Tuple[Set[SSAVar], Set[SSAVar]]]:
        """Compute live-in and live-out for each block.

        Returns dict: block_id -> (live_in, live_out)
        """
        use_sets: Dict[str, Set[SSAVar]] = {}
        def_sets: Dict[str, Set[SSAVar]] = {}

        for bid, blk in self.cfg.blocks.items():
            uses: Set[SSAVar] = set()
            defs: Set[SSAVar] = set()
            for phi in blk.phi_nodes:
                defs.add(phi.dest)
            for instr in blk.instructions:
                for u in instr.get_uses():
                    if u not in defs:
                        uses.add(u)
                for d in instr.get_defs():
                    defs.add(d)
            use_sets[bid] = uses
            def_sets[bid] = defs

        live_in: Dict[str, Set[SSAVar]] = {bid: set() for bid in self.cfg.blocks}
        live_out: Dict[str, Set[SSAVar]] = {bid: set() for bid in self.cfg.blocks}

        changed = True
        while changed:
            changed = False
            for bid in self.cfg.dfs_postorder():
                new_out: Set[SSAVar] = set()
                blk = self.cfg.blocks[bid]
                for succ in blk.successors:
                    new_out |= live_in.get(succ, set())

                new_in = use_sets[bid] | (new_out - def_sets[bid])

                if new_in != live_in[bid] or new_out != live_out[bid]:
                    live_in[bid] = new_in
                    live_out[bid] = new_out
                    changed = True

        return {bid: (live_in[bid], live_out[bid]) for bid in self.cfg.blocks}

    def get_live_at(self, block_id: str, index: int) -> Set[SSAVar]:
        """Get live variables at a specific instruction index in a block."""
        result = self.analyze()
        _, live_out = result.get(block_id, (set(), set()))
        blk = self.cfg.blocks.get(block_id)
        if not blk:
            return set()

        live: Set[SSAVar] = set(live_out)
        for i in range(len(blk.instructions) - 1, index, -1):
            instr = blk.instructions[i]
            for d in instr.get_defs():
                live.discard(d)
            for u in instr.get_uses():
                live.add(u)
        return live


# ===================================================================
# AvailableExpressions
# ===================================================================

@dataclass(frozen=True)
class Expression:
    """An expression for available expression analysis."""
    opcode: str
    operands: Tuple[str, ...]

    def __str__(self) -> str:
        return f"{self.opcode}({', '.join(self.operands)})"


class AvailableExpressions:
    """Available expressions analysis (forward dataflow)."""

    def __init__(self, cfg: CFG) -> None:
        self.cfg = cfg

    def _get_expressions(self, blk: BasicBlock) -> Tuple[Set[Expression], Set[Expression]]:
        """Get generated and killed expressions for a block."""
        gen: Set[Expression] = set()
        kill: Set[Expression] = set()

        for instr in blk.instructions:
            if instr.opcode and len(instr.operands) > 0:
                ops = tuple(str(op) for op in instr.operands if isinstance(op, SSAVar))
                if ops:
                    expr = Expression(instr.opcode, ops)
                    gen.add(expr)
            if instr.dest:
                dest_str = str(instr.dest)
                killed = {e for e in gen if dest_str in e.operands}
                gen -= killed
                kill |= killed

        return gen, kill

    def analyze(self) -> Dict[str, Set[Expression]]:
        """Compute available expressions at entry of each block."""
        all_exprs: Set[Expression] = set()
        gen_map: Dict[str, Set[Expression]] = {}
        kill_map: Dict[str, Set[Expression]] = {}

        for bid, blk in self.cfg.blocks.items():
            gen, kill = self._get_expressions(blk)
            gen_map[bid] = gen
            kill_map[bid] = kill
            all_exprs |= gen

        avail_in: Dict[str, Set[Expression]] = {}
        entry = self.cfg._entry_id
        for bid in self.cfg.blocks:
            if bid == entry:
                avail_in[bid] = set()
            else:
                avail_in[bid] = set(all_exprs)

        changed = True
        while changed:
            changed = False
            for bid in self.cfg.reverse_postorder():
                preds = self.cfg.get_predecessors(bid)
                if preds:
                    new_in: Set[Expression] = set(all_exprs)
                    for p in preds:
                        avail_out_p = (avail_in.get(p, set()) - kill_map.get(p, set())) | gen_map.get(p, set())
                        new_in &= avail_out_p
                else:
                    new_in = set()

                if bid == entry:
                    new_in = set()

                if new_in != avail_in[bid]:
                    avail_in[bid] = new_in
                    changed = True

        return avail_in


# ===================================================================
# ReachingDefinitions
# ===================================================================

@dataclass(frozen=True)
class Definition:
    """A definition for reaching definitions analysis."""
    var: str
    block_id: str
    index: int

    def __str__(self) -> str:
        return f"{self.var}@{self.block_id}[{self.index}]"


class ReachingDefinitions:
    """Reaching definitions analysis (forward dataflow)."""

    def __init__(self, cfg: CFG) -> None:
        self.cfg = cfg

    def analyze(self) -> Dict[str, Set[Definition]]:
        """Compute reaching definitions at entry of each block."""
        gen_map: Dict[str, Set[Definition]] = {}
        kill_map: Dict[str, Set[str]] = {}  # var names killed

        for bid, blk in self.cfg.blocks.items():
            gen: Set[Definition] = set()
            killed_vars: Set[str] = set()

            for i, phi in enumerate(blk.phi_nodes):
                d = Definition(str(phi.dest), bid, -i - 1)
                gen.add(d)
                killed_vars.add(phi.dest.name)

            for i, instr in enumerate(blk.instructions):
                if instr.dest:
                    d = Definition(str(instr.dest), bid, i)
                    gen.add(d)
                    killed_vars.add(instr.dest.name)

            gen_map[bid] = gen
            kill_map[bid] = killed_vars

        reach_in: Dict[str, Set[Definition]] = {bid: set() for bid in self.cfg.blocks}

        changed = True
        while changed:
            changed = False
            for bid in self.cfg.reverse_postorder():
                preds = self.cfg.get_predecessors(bid)
                new_in: Set[Definition] = set()
                for p in preds:
                    # reach_out(p) = gen(p) | (reach_in(p) - kill(p))
                    reach_out_p = set(gen_map.get(p, set()))
                    killed = kill_map.get(p, set())
                    for d in reach_in.get(p, set()):
                        if d.var.split("_")[0] not in killed:
                            reach_out_p.add(d)
                    new_in |= reach_out_p

                if new_in != reach_in[bid]:
                    reach_in[bid] = new_in
                    changed = True

        return reach_in


# ===================================================================
# DominatorTreePrinter
# ===================================================================

class DominatorTreePrinter:
    """Pretty-print dominator tree."""

    @staticmethod
    def print_tree(dom_tree: DominatorTree) -> str:
        if not dom_tree._computed:
            dom_tree.compute_dominators()
        lines: List[str] = ["Dominator Tree:"]
        entry = dom_tree.cfg._entry_id
        if entry:
            DominatorTreePrinter._print_node(dom_tree, entry, 0, lines)
        return "\n".join(lines)

    @staticmethod
    def _print_node(dom_tree: DominatorTree, bid: str, depth: int, lines: List[str]) -> None:
        indent = "  " * depth
        lines.append(f"{indent}{bid}")
        for child in dom_tree.get_dom_children(bid):
            DominatorTreePrinter._print_node(dom_tree, child, depth + 1, lines)

    @staticmethod
    def to_dot(dom_tree: DominatorTree) -> str:
        if not dom_tree._computed:
            dom_tree.compute_dominators()
        lines = ["digraph DomTree {"]
        lines.append('  rankdir=TB;')
        for bid in dom_tree.cfg.blocks:
            lines.append(f'  "{bid}";')
        for bid, idom in dom_tree._idom.items():
            if bid != idom:
                lines.append(f'  "{idom}" -> "{bid}";')
        lines.append("}")
        return "\n".join(lines)


# ===================================================================
# InterferenceGraph (for register allocation)
# ===================================================================

class InterferenceGraph:
    """Interference graph for register allocation."""

    def __init__(self) -> None:
        self.nodes: Set[str] = set()
        self.edges: Set[FrozenSet[str]] = set()
        self._adj: Dict[str, Set[str]] = {}

    def add_node(self, var: str) -> None:
        self.nodes.add(var)
        self._adj.setdefault(var, set())

    def add_edge(self, a: str, b: str) -> None:
        if a == b:
            return
        self.nodes.add(a)
        self.nodes.add(b)
        self.edges.add(frozenset({a, b}))
        self._adj.setdefault(a, set()).add(b)
        self._adj.setdefault(b, set()).add(a)

    def interferes(self, a: str, b: str) -> bool:
        return frozenset({a, b}) in self.edges

    def neighbors(self, var: str) -> Set[str]:
        return self._adj.get(var, set())

    def degree(self, var: str) -> int:
        return len(self._adj.get(var, set()))

    @staticmethod
    def build_from_liveness(cfg: CFG) -> InterferenceGraph:
        """Build interference graph from liveness analysis."""
        lva = LiveVariableAnalysis(cfg)
        liveness = lva.analyze()
        ig = InterferenceGraph()

        for bid, (live_in, live_out) in liveness.items():
            all_live = live_in | live_out
            vars_list = [str(v) for v in all_live]
            for v in vars_list:
                ig.add_node(v)
            for i in range(len(vars_list)):
                for j in range(i + 1, len(vars_list)):
                    ig.add_edge(vars_list[i], vars_list[j])

        return ig


# ===================================================================
# Verification utilities
# ===================================================================

def verify_cfg_complete(cfg: CFG) -> Tuple[bool, List[str]]:
    """Run comprehensive verification. Returns (is_valid, errors)."""
    verifier = CFGVerifier(cfg)
    errors = verifier.verify_all()
    return len(errors) == 0, errors


def build_and_verify_ssa(cfg: CFG) -> Tuple[CFG, List[str]]:
    """Build SSA form and verify it. Returns (cfg_in_ssa, errors)."""
    constructor = SSAConstructor(cfg)
    constructor.construct()
    errors = constructor.validate_ssa_property()
    errors.extend(cfg.validate())
    return cfg, errors


# ===================================================================
# CFGWalker: structured traversal helper
# ===================================================================

class CFGWalker:
    """Structured CFG traversal with callbacks."""

    def __init__(self, cfg: CFG) -> None:
        self.cfg = cfg

    def walk_forward(
        self,
        on_block: Callable[[BasicBlock], bool],
        start: Optional[str] = None,
    ) -> None:
        """Walk CFG forward. on_block returns True to continue, False to stop."""
        for bid in self.cfg.reverse_postorder(start):
            blk = self.cfg.blocks[bid]
            if not on_block(blk):
                break

    def walk_backward(
        self,
        on_block: Callable[[BasicBlock], bool],
        start: Optional[str] = None,
    ) -> None:
        """Walk CFG backward (reverse of forward)."""
        order = self.cfg.reverse_postorder(start)
        for bid in reversed(order):
            blk = self.cfg.blocks[bid]
            if not on_block(blk):
                break

    def walk_dominator_tree(
        self,
        on_block: Callable[[BasicBlock, int], None],
    ) -> None:
        """Walk dominator tree in pre-order."""
        dom = DominatorTree(self.cfg)
        dom.compute_dominators()
        entry = self.cfg._entry_id
        if entry is None:
            return

        def _walk(bid: str, depth: int) -> None:
            blk = self.cfg.blocks.get(bid)
            if blk:
                on_block(blk, depth)
            for child in dom.get_dom_children(bid):
                _walk(child, depth + 1)

        _walk(entry, 0)

    def walk_loops(
        self,
        on_loop: Callable[[Loop], None],
    ) -> None:
        """Walk all loops."""
        loop_info = LoopInfo(self.cfg)
        for loop in loop_info.identify_loops():
            on_loop(loop)

    def collect_instructions(
        self,
        predicate: Callable[[IRInstruction], bool],
    ) -> List[Tuple[str, int, IRInstruction]]:
        """Collect instructions matching predicate. Returns (block_id, index, instr)."""
        results: List[Tuple[str, int, IRInstruction]] = []
        for bid in self.cfg.reverse_postorder():
            blk = self.cfg.blocks[bid]
            for i, instr in enumerate(blk.instructions):
                if predicate(instr):
                    results.append((bid, i, instr))
        return results


# ===================================================================
# CFG construction from instruction sequences
# ===================================================================

class LinearToCFG:
    """Convert linear instruction sequences (with labels/jumps) to CFG."""

    @staticmethod
    def convert(
        instructions: List[IRInstruction],
        name: str = "func",
    ) -> CFG:
        """Convert linear instruction list to CFG.

        Expects instructions with opcodes:
        - label: starts new block
        - branch: unconditional jump (operand = target label)
        - cond_br: conditional (operand = cond, true_label, false_label)
        - return: function return
        """
        cfg = CFG(name)
        label_to_block: Dict[str, BasicBlock] = {}
        current_instrs: List[IRInstruction] = []
        current_label = "entry"

        # first pass: identify blocks
        block_starts: List[Tuple[str, List[IRInstruction]]] = []
        for instr in instructions:
            if instr.opcode == "label":
                if current_instrs or current_label == "entry":
                    block_starts.append((current_label, current_instrs))
                current_label = str(instr.operands[0]) if instr.operands else f"L{len(block_starts)}"
                current_instrs = []
            else:
                current_instrs.append(instr)
        if current_instrs or current_label:
            block_starts.append((current_label, current_instrs))

        # create blocks
        for label, instrs in block_starts:
            blk = BasicBlock(label, label)
            blk.instructions = instrs
            cfg.add_block(blk)
            label_to_block[label] = blk

        if block_starts:
            cfg.set_entry(block_starts[0][0])

        # second pass: add edges
        block_labels = [label for label, _ in block_starts]
        for i, (label, _) in enumerate(block_starts):
            blk = label_to_block[label]
            last = blk.last_instruction()

            if last is None or last.opcode not in ("branch", "cond_br", "return"):
                # fallthrough
                if i + 1 < len(block_labels):
                    cfg.add_edge(label, block_labels[i + 1], EdgeKind.FALLTHROUGH)
            elif last.opcode == "branch":
                target = str(last.operands[0]) if last.operands else ""
                if target in label_to_block:
                    cfg.add_edge(label, target)
            elif last.opcode == "cond_br":
                if len(last.operands) >= 3:
                    true_target = str(last.operands[1])
                    false_target = str(last.operands[2])
                    if true_target in label_to_block:
                        cfg.add_edge(label, true_target, EdgeKind.TRUE_BRANCH)
                    if false_target in label_to_block:
                        cfg.add_edge(label, false_target, EdgeKind.FALSE_BRANCH)
            elif last.opcode == "return":
                blk.is_exit = True

        # find/create exit
        exits = [bid for bid, blk in cfg.blocks.items() if blk.is_exit]
        if exits:
            if len(exits) == 1:
                cfg.set_exit(exits[0])
            else:
                exit_blk = BasicBlock("__exit__", "exit")
                cfg.add_block(exit_blk)
                cfg.set_exit("__exit__")
                for eid in exits:
                    cfg.add_edge(eid, "__exit__")

        return cfg


# ===================================================================
# CFGRegion: region-based analysis support
# ===================================================================

@dataclass
class CFGRegion:
    """A region in the CFG (for region-based analysis)."""
    entry: str
    exit: Optional[str]
    blocks: Set[str]
    subregions: List[CFGRegion] = field(default_factory=list)
    kind: str = "block"  # block, loop, hammock, acyclic

    def contains(self, block_id: str) -> bool:
        return block_id in self.blocks

    def size(self) -> int:
        return len(self.blocks)

    def __repr__(self) -> str:
        return f"CFGRegion(entry={self.entry!r}, kind={self.kind!r}, size={self.size()})"


class RegionBuilder:
    """Build hierarchical region decomposition of CFG."""

    def __init__(self, cfg: CFG) -> None:
        self.cfg = cfg

    def build_regions(self) -> CFGRegion:
        """Build top-level region containing all blocks."""
        top = CFGRegion(
            entry=self.cfg._entry_id or "",
            exit=self.cfg._exit_id,
            blocks=set(self.cfg.blocks.keys()),
            kind="function",
        )

        # identify loop regions
        loop_info = LoopInfo(self.cfg)
        loops = loop_info.identify_loops()
        for loop in loops:
            exits = list(loop.exit_blocks)
            loop_region = CFGRegion(
                entry=loop.header,
                exit=exits[0] if exits else None,
                blocks=loop.body,
                kind="loop",
            )
            top.subregions.append(loop_region)

        # identify hammock regions
        hammocks = CFGPattern.find_hammock(self.cfg)
        for entry, exit_b, body in hammocks:
            if not any(body <= lr.blocks for lr in top.subregions):
                h_region = CFGRegion(
                    entry=entry,
                    exit=exit_b,
                    blocks=body,
                    kind="hammock",
                )
                top.subregions.append(h_region)

        return top


# ===================================================================
# ExceptionFlowGraph: extend CFG with exception edges
# ===================================================================

class ExceptionFlowGraph:
    """CFG extended with exception flow information."""

    def __init__(self, cfg: CFG) -> None:
        self.cfg = cfg
        self._try_regions: List[Tuple[str, str, str]] = []  # (try_entry, try_exit, handler)

    def add_try_region(self, try_entry: str, try_exit: str, handler: str) -> None:
        self._try_regions.append((try_entry, try_exit, handler))
        # add exception edges from all blocks in try region to handler
        reachable: Set[str] = set()
        stack = [try_entry]
        while stack:
            n = stack.pop()
            if n in reachable or n == try_exit:
                continue
            reachable.add(n)
            blk = self.cfg.blocks.get(n)
            if blk:
                for s in blk.successors:
                    stack.append(s)
        for bid in reachable:
            blk = self.cfg.blocks.get(bid)
            if blk:
                has_call = any(i.opcode in ("call", "invoke") for i in blk.instructions)
                if has_call:
                    self.cfg.add_edge(bid, handler, EdgeKind.EXCEPTION)

    def get_exception_successors(self, block_id: str) -> List[str]:
        return [
            e.dst for e in self.cfg.get_edges_from(block_id)
            if e.kind == EdgeKind.EXCEPTION
        ]

    def get_try_handler(self, block_id: str) -> Optional[str]:
        for try_entry, try_exit, handler in self._try_regions:
            # check if block_id is between try_entry and try_exit
            reach = ReachabilityAnalysis(self.cfg)
            if reach.can_reach(try_entry, block_id) and reach.can_reach(block_id, try_exit):
                return handler
        return None


# ===================================================================
# CFG pretty printer with ANSI colors
# ===================================================================

class CFGColorPrinter:
    """Pretty-print CFG with ANSI colors."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"

    @staticmethod
    def print_cfg(cfg: CFG, show_ssa: bool = True) -> str:
        lines: List[str] = []
        c = CFGColorPrinter
        lines.append(f"{c.BOLD}{'═' * 60}{c.RESET}")
        lines.append(f"{c.BOLD}CFG: {cfg.name}{c.RESET}")
        lines.append(f"{c.GRAY}Entry: {cfg._entry_id}, Exit: {cfg._exit_id}{c.RESET}")
        lines.append(f"{c.GRAY}Blocks: {len(cfg.blocks)}, Edges: {len(cfg.edges)}{c.RESET}")
        lines.append(f"{c.BOLD}{'═' * 60}{c.RESET}")

        for bid in cfg.reverse_postorder():
            blk = cfg.blocks[bid]
            header = f"{c.BOLD}{c.BLUE}{bid}{c.RESET}"
            if blk.label != bid:
                header += f" {c.GRAY}({blk.label}){c.RESET}"
            if blk.is_loop_header:
                header += f" {c.YELLOW}[loop header]{c.RESET}"
            if bid == cfg._entry_id:
                header += f" {c.GREEN}[entry]{c.RESET}"
            if bid == cfg._exit_id:
                header += f" {c.RED}[exit]{c.RESET}"
            lines.append(header)

            if blk.predecessors:
                preds = ", ".join(blk.predecessors)
                lines.append(f"  {c.GRAY}preds: {preds}{c.RESET}")
            if blk.successors:
                succs = ", ".join(blk.successors)
                lines.append(f"  {c.GRAY}succs: {succs}{c.RESET}")

            for phi in blk.phi_nodes:
                pairs = ", ".join(f"[{c.CYAN}{bid}{c.RESET}]: {v}" for bid, v in phi.incoming.items())
                lines.append(f"  {c.MAGENTA}{phi.dest}{c.RESET} = {c.MAGENTA}phi{c.RESET}({pairs})")

            for instr in blk.instructions:
                dest_s = f"{c.GREEN}{instr.dest}{c.RESET} = " if instr.dest else ""
                ops = ", ".join(
                    f"{c.CYAN}{o}{c.RESET}" if isinstance(o, SSAVar) else str(o)
                    for o in instr.operands
                )
                lines.append(f"  {dest_s}{c.YELLOW}{instr.opcode}{c.RESET}({ops})")

            lines.append("")

        return "\n".join(lines)


# ===================================================================
# Utility: build simple CFG for testing
# ===================================================================

def build_simple_cfg(
    block_specs: List[Tuple[str, List[str]]],
    edges: List[Tuple[str, str]],
    entry: Optional[str] = None,
    exit_block: Optional[str] = None,
) -> CFG:
    """Build a simple CFG from specs.

    block_specs: list of (block_id, instruction_opcodes)
    edges: list of (src, dst) edges
    """
    cfg = CFG("test")
    for bid, opcodes in block_specs:
        blk = BasicBlock(bid, bid)
        for i, op in enumerate(opcodes):
            blk.add_instruction(IRInstruction(opcode=op, dest=SSAVar(f"v{i}", 0)))
        cfg.add_block(blk)

    for src, dst in edges:
        cfg.add_edge(src, dst)

    if entry:
        cfg.set_entry(entry)
    elif block_specs:
        cfg.set_entry(block_specs[0][0])

    if exit_block:
        cfg.set_exit(exit_block)
    elif block_specs:
        cfg.set_exit(block_specs[-1][0])

    return cfg

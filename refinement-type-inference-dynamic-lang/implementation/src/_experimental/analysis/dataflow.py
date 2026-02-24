from __future__ import annotations

"""
Dataflow analysis framework for refinement type inference.

Provides forward, backward, and bidirectional dataflow analyses,
classic analyses (reaching definitions, live variables, constant propagation, etc.),
SSA construction, dominator trees, program slicing, and taint tracking.
"""

import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generic,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)
import heapq


# ---------------------------------------------------------------------------
# Local type stubs
# ---------------------------------------------------------------------------

T = TypeVar("T")


@dataclass(frozen=True)
class Location:
    file: str = "<unknown>"
    line: int = 0
    column: int = 0


@dataclass(frozen=True)
class Instruction:
    opcode: str
    operands: Tuple[str, ...] = ()
    result: Optional[str] = None
    location: Location = field(default_factory=Location)


@dataclass
class BasicBlock:
    label: str
    instructions: List[Instruction] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)
    predecessors: List[str] = field(default_factory=list)


@dataclass
class CFG:
    blocks: Dict[str, BasicBlock] = field(default_factory=dict)
    entry: str = "entry"
    exit: str = "exit"

    def successors(self, label: str) -> List[str]:
        bb = self.blocks.get(label)
        return bb.successors if bb else []

    def predecessors(self, label: str) -> List[str]:
        bb = self.blocks.get(label)
        return bb.predecessors if bb else []

    def all_labels(self) -> List[str]:
        return list(self.blocks.keys())

    def reverse_postorder(self) -> List[str]:
        visited: set[str] = set()
        order: list[str] = []

        def dfs(label: str) -> None:
            if label in visited:
                return
            visited.add(label)
            for succ in self.successors(label):
                dfs(succ)
            order.append(label)

        dfs(self.entry)
        order.reverse()
        return order

    def postorder(self) -> List[str]:
        visited: set[str] = set()
        order: list[str] = []

        def dfs(label: str) -> None:
            if label in visited:
                return
            visited.add(label)
            for succ in self.successors(label):
                dfs(succ)
            order.append(label)

        dfs(self.entry)
        return order


@dataclass(frozen=True)
class ProgramPoint:
    block: str
    index: int = 0

    def __str__(self) -> str:
        return f"{self.block}:{self.index}"


# =========================================================================
#  DataflowFact
# =========================================================================

@dataclass
class DataflowFact:
    """Abstract dataflow fact."""
    data: Any = None

    def join(self, other: DataflowFact) -> DataflowFact:
        if self.data is None:
            return other
        if other.data is None:
            return self
        if isinstance(self.data, set) and isinstance(other.data, set):
            return DataflowFact(data=self.data | other.data)
        if isinstance(self.data, dict) and isinstance(other.data, dict):
            merged = dict(self.data)
            merged.update(other.data)
            return DataflowFact(data=merged)
        return DataflowFact(data=self.data)

    def meet(self, other: DataflowFact) -> DataflowFact:
        if self.data is None or other.data is None:
            return DataflowFact(data=None)
        if isinstance(self.data, set) and isinstance(other.data, set):
            return DataflowFact(data=self.data & other.data)
        return DataflowFact(data=self.data)

    def leq(self, other: DataflowFact) -> bool:
        if self.data is None:
            return True
        if other.data is None:
            return False
        if isinstance(self.data, set) and isinstance(other.data, set):
            return self.data <= other.data
        return self.data == other.data

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DataflowFact):
            return NotImplemented
        return self.data == other.data

    def __hash__(self) -> int:
        if isinstance(self.data, set):
            return hash(frozenset(self.data))
        return hash(self.data)


# =========================================================================
#  DataflowResult
# =========================================================================

@dataclass
class DataflowResult:
    """Stores analysis results indexed by program point."""
    _in_facts: Dict[str, DataflowFact] = field(default_factory=dict)
    _out_facts: Dict[str, DataflowFact] = field(default_factory=dict)
    analysis_name: str = ""
    iterations: int = 0
    elapsed: float = 0.0

    def set_in(self, label: str, fact: DataflowFact) -> None:
        self._in_facts[label] = fact

    def set_out(self, label: str, fact: DataflowFact) -> None:
        self._out_facts[label] = fact

    def get_in(self, label: str) -> DataflowFact:
        return self._in_facts.get(label, DataflowFact())

    def get_out(self, label: str) -> DataflowFact:
        return self._out_facts.get(label, DataflowFact())

    def all_in(self) -> Dict[str, DataflowFact]:
        return dict(self._in_facts)

    def all_out(self) -> Dict[str, DataflowFact]:
        return dict(self._out_facts)

    def fact_at(self, point: ProgramPoint) -> DataflowFact:
        return self._in_facts.get(point.block, DataflowFact())


# =========================================================================
#  GenKillFramework
# =========================================================================

@dataclass
class GenKillSets:
    """Gen and Kill sets for a basic block."""
    gen: Set[str] = field(default_factory=set)
    kill: Set[str] = field(default_factory=set)


@dataclass
class GenKillFramework:
    """Gen/kill framework for bit-vector dataflow problems."""
    _gen_kill: Dict[str, GenKillSets] = field(default_factory=dict)

    def set_gen_kill(self, label: str, gen: Set[str], kill: Set[str]) -> None:
        self._gen_kill[label] = GenKillSets(gen=gen, kill=kill)

    def get_gen(self, label: str) -> Set[str]:
        gk = self._gen_kill.get(label)
        return set(gk.gen) if gk else set()

    def get_kill(self, label: str) -> Set[str]:
        gk = self._gen_kill.get(label)
        return set(gk.kill) if gk else set()

    def transfer(self, label: str, in_set: Set[str]) -> Set[str]:
        gk = self._gen_kill.get(label)
        if gk is None:
            return in_set
        return (in_set - gk.kill) | gk.gen

    def compute_gen_kill(self, cfg: CFG) -> None:
        for label, bb in cfg.blocks.items():
            gen: set[str] = set()
            kill: set[str] = set()
            for instr in bb.instructions:
                if instr.result:
                    kill.add(instr.result)
                    gen.discard(instr.result)
                    gen.add(f"{label}:{instr.result}")
                for op in instr.operands:
                    if op not in kill:
                        gen.add(op)
            self._gen_kill[label] = GenKillSets(gen=gen, kill=kill)


# =========================================================================
#  CFGTraversal
# =========================================================================

class TraversalOrder(Enum):
    RPO = auto()
    BFS = auto()
    DFS = auto()
    POSTORDER = auto()


@dataclass
class CFGTraversal:
    """Various CFG traversal orders."""

    @staticmethod
    def reverse_postorder(cfg: CFG) -> List[str]:
        return cfg.reverse_postorder()

    @staticmethod
    def postorder(cfg: CFG) -> List[str]:
        return cfg.postorder()

    @staticmethod
    def bfs(cfg: CFG) -> List[str]:
        visited: set[str] = set()
        order: list[str] = []
        queue = deque([cfg.entry])
        visited.add(cfg.entry)
        while queue:
            node = queue.popleft()
            order.append(node)
            for succ in cfg.successors(node):
                if succ not in visited:
                    visited.add(succ)
                    queue.append(succ)
        return order

    @staticmethod
    def dfs(cfg: CFG) -> List[str]:
        visited: set[str] = set()
        order: list[str] = []
        stack = [cfg.entry]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            order.append(node)
            for succ in reversed(cfg.successors(node)):
                if succ not in visited:
                    stack.append(succ)
        return order

    @staticmethod
    def get_order(cfg: CFG, order: TraversalOrder) -> List[str]:
        if order == TraversalOrder.RPO:
            return CFGTraversal.reverse_postorder(cfg)
        elif order == TraversalOrder.BFS:
            return CFGTraversal.bfs(cfg)
        elif order == TraversalOrder.DFS:
            return CFGTraversal.dfs(cfg)
        elif order == TraversalOrder.POSTORDER:
            return CFGTraversal.postorder(cfg)
        return cfg.all_labels()


# =========================================================================
#  WorklistAlgorithm
# =========================================================================

class WorklistKind(Enum):
    FIFO = auto()
    LIFO = auto()
    PRIORITY = auto()
    RPO = auto()


@dataclass
class WorklistAlgorithm:
    """Configurable worklist (FIFO, LIFO, priority, reverse postorder)."""
    kind: WorklistKind = WorklistKind.RPO
    _queue: deque[str] = field(default_factory=deque)
    _heap: List[Tuple[int, int, str]] = field(default_factory=list)
    _counter: int = 0
    _in_wl: Set[str] = field(default_factory=set)
    _priority: Dict[str, int] = field(default_factory=dict)

    def initialize(self, labels: Sequence[str], cfg: Optional[CFG] = None) -> None:
        self._in_wl.clear()
        self._queue.clear()
        self._heap.clear()
        self._counter = 0

        if self.kind == WorklistKind.RPO and cfg:
            rpo = cfg.reverse_postorder()
            rpo_index = {label: i for i, label in enumerate(rpo)}
            self._priority = rpo_index
            for label in labels:
                self.add(label)
        else:
            for label in labels:
                self.add(label)

    def add(self, label: str) -> None:
        if label in self._in_wl:
            return
        self._in_wl.add(label)
        if self.kind == WorklistKind.PRIORITY or self.kind == WorklistKind.RPO:
            self._counter += 1
            prio = self._priority.get(label, 0)
            heapq.heappush(self._heap, (prio, self._counter, label))
        else:
            self._queue.append(label)

    def pop(self) -> Optional[str]:
        if self.kind in (WorklistKind.PRIORITY, WorklistKind.RPO):
            while self._heap:
                _, _, label = heapq.heappop(self._heap)
                if label in self._in_wl:
                    self._in_wl.discard(label)
                    return label
            return None
        if not self._queue:
            return None
        if self.kind == WorklistKind.LIFO:
            label = self._queue.pop()
        else:
            label = self._queue.popleft()
        self._in_wl.discard(label)
        return label

    def is_empty(self) -> bool:
        return len(self._in_wl) == 0


# =========================================================================
#  DataflowEquation
# =========================================================================

@dataclass
class DataflowEquation:
    """Represents a single dataflow equation for a basic block."""
    block: str
    dependencies: List[str] = field(default_factory=list)
    transfer: Optional[Callable[[DataflowFact], DataflowFact]] = None

    def evaluate(self, in_fact: DataflowFact) -> DataflowFact:
        if self.transfer:
            return self.transfer(in_fact)
        return in_fact


# =========================================================================
#  DataflowAnalysis (base)
# =========================================================================

@dataclass
class DataflowAnalysis(ABC):
    """Base class for all dataflow analyses."""
    name: str = "dataflow"
    max_iterations: int = 10000

    @abstractmethod
    def direction(self) -> str:
        ...

    @abstractmethod
    def init_fact(self) -> DataflowFact:
        ...

    @abstractmethod
    def entry_fact(self) -> DataflowFact:
        ...

    @abstractmethod
    def transfer_block(self, block: BasicBlock, in_fact: DataflowFact) -> DataflowFact:
        ...

    def join(self, a: DataflowFact, b: DataflowFact) -> DataflowFact:
        return a.join(b)

    def analyze(self, cfg: CFG) -> DataflowResult:
        if self.direction() == "forward":
            return self._forward_solve(cfg)
        return self._backward_solve(cfg)

    def _forward_solve(self, cfg: CFG) -> DataflowResult:
        result = DataflowResult(analysis_name=self.name)
        start = time.time()

        for label in cfg.all_labels():
            result.set_in(label, self.init_fact())
            result.set_out(label, self.init_fact())
        result.set_in(cfg.entry, self.entry_fact())

        wl = WorklistAlgorithm(kind=WorklistKind.RPO)
        wl.initialize(cfg.all_labels(), cfg)
        iterations = 0

        while not wl.is_empty() and iterations < self.max_iterations:
            label = wl.pop()
            if label is None:
                break
            iterations += 1
            bb = cfg.blocks.get(label)
            if bb is None:
                continue

            preds = cfg.predecessors(label)
            if preds:
                in_fact = result.get_out(preds[0])
                for pred in preds[1:]:
                    in_fact = self.join(in_fact, result.get_out(pred))
            else:
                in_fact = result.get_in(label)

            result.set_in(label, in_fact)
            out_fact = self.transfer_block(bb, in_fact)
            old_out = result.get_out(label)
            if out_fact != old_out:
                result.set_out(label, out_fact)
                for succ in cfg.successors(label):
                    wl.add(succ)

        result.iterations = iterations
        result.elapsed = time.time() - start
        return result

    def _backward_solve(self, cfg: CFG) -> DataflowResult:
        result = DataflowResult(analysis_name=self.name)
        start = time.time()

        for label in cfg.all_labels():
            result.set_in(label, self.init_fact())
            result.set_out(label, self.init_fact())
        if cfg.exit in cfg.blocks:
            result.set_out(cfg.exit, self.entry_fact())

        wl = WorklistAlgorithm(kind=WorklistKind.RPO)
        rpo = cfg.reverse_postorder()
        wl.initialize(list(reversed(rpo)), cfg)
        iterations = 0

        while not wl.is_empty() and iterations < self.max_iterations:
            label = wl.pop()
            if label is None:
                break
            iterations += 1
            bb = cfg.blocks.get(label)
            if bb is None:
                continue

            succs = cfg.successors(label)
            if succs:
                out_fact = result.get_in(succs[0])
                for succ in succs[1:]:
                    out_fact = self.join(out_fact, result.get_in(succ))
            else:
                out_fact = result.get_out(label)

            result.set_out(label, out_fact)
            in_fact = self.transfer_block(bb, out_fact)
            old_in = result.get_in(label)
            if in_fact != old_in:
                result.set_in(label, in_fact)
                for pred in cfg.predecessors(label):
                    wl.add(pred)

        result.iterations = iterations
        result.elapsed = time.time() - start
        return result


# =========================================================================
#  ForwardAnalysis / BackwardAnalysis / BidirectionalAnalysis
# =========================================================================

@dataclass
class ForwardAnalysis(DataflowAnalysis):
    """Forward dataflow analysis template."""
    def direction(self) -> str:
        return "forward"


@dataclass
class BackwardAnalysis(DataflowAnalysis):
    """Backward dataflow analysis template."""
    def direction(self) -> str:
        return "backward"


@dataclass
class BidirectionalAnalysis:
    """Combines forward and backward analyses."""
    forward: ForwardAnalysis = field(default_factory=lambda: ReachingDefinitions())
    backward: BackwardAnalysis = field(default_factory=lambda: LiveVariables())
    max_rounds: int = 5

    def analyze(self, cfg: CFG) -> Tuple[DataflowResult, DataflowResult]:
        fwd_result = DataflowResult()
        bwd_result = DataflowResult()
        for _ in range(self.max_rounds):
            fwd_result = self.forward.analyze(cfg)
            bwd_result = self.backward.analyze(cfg)
        return fwd_result, bwd_result


# =========================================================================
#  Classic analyses
# =========================================================================

@dataclass
class ReachingDefinitions(ForwardAnalysis):
    """Reaching definitions analysis."""
    name: str = "reaching-definitions"

    def init_fact(self) -> DataflowFact:
        return DataflowFact(data=set())

    def entry_fact(self) -> DataflowFact:
        return DataflowFact(data=set())

    def transfer_block(self, block: BasicBlock, in_fact: DataflowFact) -> DataflowFact:
        current: set[str] = set(in_fact.data) if in_fact.data else set()
        for i, instr in enumerate(block.instructions):
            if instr.result:
                current = {d for d in current if not d.endswith(f":{instr.result}")}
                current.add(f"{block.label}:{i}:{instr.result}")
        return DataflowFact(data=current)


@dataclass
class LiveVariables(BackwardAnalysis):
    """Live variable analysis."""
    name: str = "live-variables"

    def init_fact(self) -> DataflowFact:
        return DataflowFact(data=set())

    def entry_fact(self) -> DataflowFact:
        return DataflowFact(data=set())

    def transfer_block(self, block: BasicBlock, in_fact: DataflowFact) -> DataflowFact:
        live: set[str] = set(in_fact.data) if in_fact.data else set()
        for instr in reversed(block.instructions):
            if instr.result:
                live.discard(instr.result)
            for op in instr.operands:
                if not op.startswith("#"):  # skip constants
                    live.add(op)
        return DataflowFact(data=live)


@dataclass
class AvailableExpressions(ForwardAnalysis):
    """Available expressions analysis."""
    name: str = "available-expressions"
    _all_exprs: Set[str] = field(default_factory=set)

    def init_fact(self) -> DataflowFact:
        return DataflowFact(data=set(self._all_exprs))

    def entry_fact(self) -> DataflowFact:
        return DataflowFact(data=set())

    def join(self, a: DataflowFact, b: DataflowFact) -> DataflowFact:
        return a.meet(b)

    def transfer_block(self, block: BasicBlock, in_fact: DataflowFact) -> DataflowFact:
        avail: set[str] = set(in_fact.data) if in_fact.data else set()
        for instr in block.instructions:
            if instr.result:
                avail = {e for e in avail if instr.result not in e}
            if len(instr.operands) >= 2 and instr.opcode in ("add", "sub", "mul", "div", "mod"):
                expr = f"{instr.operands[0]} {instr.opcode} {instr.operands[1]}"
                avail.add(expr)
                self._all_exprs.add(expr)
        return DataflowFact(data=avail)


@dataclass
class VeryBusyExpressions(BackwardAnalysis):
    """Very busy expressions analysis."""
    name: str = "very-busy-expressions"
    _all_exprs: Set[str] = field(default_factory=set)

    def init_fact(self) -> DataflowFact:
        return DataflowFact(data=set(self._all_exprs))

    def entry_fact(self) -> DataflowFact:
        return DataflowFact(data=set())

    def join(self, a: DataflowFact, b: DataflowFact) -> DataflowFact:
        return a.meet(b)

    def transfer_block(self, block: BasicBlock, in_fact: DataflowFact) -> DataflowFact:
        busy: set[str] = set(in_fact.data) if in_fact.data else set()
        for instr in reversed(block.instructions):
            if instr.result:
                busy = {e for e in busy if instr.result not in e}
            if len(instr.operands) >= 2 and instr.opcode in ("add", "sub", "mul", "div"):
                expr = f"{instr.operands[0]} {instr.opcode} {instr.operands[1]}"
                busy.add(expr)
                self._all_exprs.add(expr)
        return DataflowFact(data=busy)


@dataclass
class ConstantPropagation(ForwardAnalysis):
    """Constant propagation and folding."""
    name: str = "constant-propagation"

    def init_fact(self) -> DataflowFact:
        return DataflowFact(data={})

    def entry_fact(self) -> DataflowFact:
        return DataflowFact(data={})

    def join(self, a: DataflowFact, b: DataflowFact) -> DataflowFact:
        a_map: dict[str, Any] = a.data if a.data else {}
        b_map: dict[str, Any] = b.data if b.data else {}
        result: dict[str, Any] = {}
        for k in set(a_map) | set(b_map):
            av = a_map.get(k, "__bot__")
            bv = b_map.get(k, "__bot__")
            if av == "__bot__":
                result[k] = bv
            elif bv == "__bot__":
                result[k] = av
            elif av == bv:
                result[k] = av
            else:
                result[k] = "__top__"
        return DataflowFact(data=result)

    def transfer_block(self, block: BasicBlock, in_fact: DataflowFact) -> DataflowFact:
        env: dict[str, Any] = dict(in_fact.data) if in_fact.data else {}
        for instr in block.instructions:
            if instr.result and instr.opcode == "const":
                if instr.operands:
                    try:
                        env[instr.result] = int(instr.operands[0])
                    except (ValueError, IndexError):
                        env[instr.result] = instr.operands[0]
            elif instr.result and instr.opcode == "copy" and instr.operands:
                src = instr.operands[0]
                env[instr.result] = env.get(src, "__top__")
            elif instr.result and instr.opcode in ("add", "sub", "mul", "div") and len(instr.operands) >= 2:
                a_val = env.get(instr.operands[0], "__top__")
                b_val = env.get(instr.operands[1], "__top__")
                if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
                    if instr.opcode == "add":
                        env[instr.result] = a_val + b_val
                    elif instr.opcode == "sub":
                        env[instr.result] = a_val - b_val
                    elif instr.opcode == "mul":
                        env[instr.result] = a_val * b_val
                    elif instr.opcode == "div" and b_val != 0:
                        env[instr.result] = a_val // b_val
                    else:
                        env[instr.result] = "__top__"
                else:
                    env[instr.result] = "__top__"
            elif instr.result:
                env[instr.result] = "__top__"
        return DataflowFact(data=env)


@dataclass
class CopyPropagation(ForwardAnalysis):
    """Copy propagation analysis."""
    name: str = "copy-propagation"

    def init_fact(self) -> DataflowFact:
        return DataflowFact(data={})

    def entry_fact(self) -> DataflowFact:
        return DataflowFact(data={})

    def transfer_block(self, block: BasicBlock, in_fact: DataflowFact) -> DataflowFact:
        copies: dict[str, str] = dict(in_fact.data) if in_fact.data else {}
        for instr in block.instructions:
            if instr.opcode == "copy" and instr.result and instr.operands:
                src = instr.operands[0]
                while src in copies:
                    src = copies[src]
                copies[instr.result] = src
            elif instr.result:
                copies = {k: v for k, v in copies.items()
                          if k != instr.result and v != instr.result}
        return DataflowFact(data=copies)


# =========================================================================
#  TypeState, Taint, Null, Sign, Parity analyses
# =========================================================================

class TypeState(Enum):
    UNINIT = auto()
    INIT = auto()
    OPEN = auto()
    CLOSED = auto()
    ERROR = auto()


@dataclass
class TypeStateAnalysis(ForwardAnalysis):
    """Typestate analysis for protocol checking (e.g., file open/close)."""
    name: str = "typestate"
    _transitions: Dict[Tuple[TypeState, str], TypeState] = field(default_factory=dict)

    def add_transition(self, from_state: TypeState, operation: str, to_state: TypeState) -> None:
        self._transitions[(from_state, operation)] = to_state

    def init_fact(self) -> DataflowFact:
        return DataflowFact(data={})

    def entry_fact(self) -> DataflowFact:
        return DataflowFact(data={})

    def transfer_block(self, block: BasicBlock, in_fact: DataflowFact) -> DataflowFact:
        states: dict[str, TypeState] = dict(in_fact.data) if in_fact.data else {}
        for instr in block.instructions:
            if instr.opcode == "alloc" and instr.result:
                states[instr.result] = TypeState.UNINIT
            elif instr.opcode == "call" and instr.operands:
                op_name = instr.operands[0]
                if len(instr.operands) > 1:
                    obj = instr.operands[1]
                    current = states.get(obj, TypeState.UNINIT)
                    next_state = self._transitions.get((current, op_name))
                    if next_state is not None:
                        states[obj] = next_state
                    elif op_name not in ("read", "write", "close", "open"):
                        pass
                    else:
                        states[obj] = TypeState.ERROR
        return DataflowFact(data=states)

    def violations(self, result: DataflowResult, cfg: CFG) -> List[Tuple[str, str, TypeState]]:
        violations: list[tuple[str, str, TypeState]] = []
        for label in cfg.all_labels():
            fact = result.get_out(label)
            if fact.data:
                for var, state in fact.data.items():
                    if state == TypeState.ERROR:
                        violations.append((label, var, state))
        return violations


@dataclass
class TaintAnalysis(ForwardAnalysis):
    """Taint tracking for security."""
    name: str = "taint"
    sources: Set[str] = field(default_factory=set)
    sinks: Set[str] = field(default_factory=set)
    sanitizers: Set[str] = field(default_factory=set)

    def init_fact(self) -> DataflowFact:
        return DataflowFact(data=set())

    def entry_fact(self) -> DataflowFact:
        return DataflowFact(data=set())

    def transfer_block(self, block: BasicBlock, in_fact: DataflowFact) -> DataflowFact:
        tainted: set[str] = set(in_fact.data) if in_fact.data else set()
        for instr in block.instructions:
            if instr.opcode == "call" and instr.operands and instr.operands[0] in self.sources:
                if instr.result:
                    tainted.add(instr.result)
            elif instr.opcode == "call" and instr.operands and instr.operands[0] in self.sanitizers:
                if instr.result:
                    tainted.discard(instr.result)
                for op in instr.operands[1:]:
                    tainted.discard(op)
            elif instr.result:
                if any(op in tainted for op in instr.operands):
                    tainted.add(instr.result)
        return DataflowFact(data=tainted)

    def find_violations(self, result: DataflowResult, cfg: CFG) -> List[Tuple[str, str, str]]:
        violations: list[tuple[str, str, str]] = []
        for label, bb in cfg.blocks.items():
            fact = result.get_in(label)
            tainted = fact.data if fact.data else set()
            for instr in bb.instructions:
                if instr.opcode == "call" and instr.operands and instr.operands[0] in self.sinks:
                    for op in instr.operands[1:]:
                        if op in tainted:
                            violations.append((label, op, instr.operands[0]))
        return violations


@dataclass
class NullFlowAnalysis(ForwardAnalysis):
    """Tracks null/None flow through program."""
    name: str = "null-flow"

    def init_fact(self) -> DataflowFact:
        return DataflowFact(data={"definitely_null": set(), "maybe_null": set()})

    def entry_fact(self) -> DataflowFact:
        return DataflowFact(data={"definitely_null": set(), "maybe_null": set()})

    def transfer_block(self, block: BasicBlock, in_fact: DataflowFact) -> DataflowFact:
        data = in_fact.data if in_fact.data else {"definitely_null": set(), "maybe_null": set()}
        def_null: set[str] = set(data.get("definitely_null", set()))
        maybe_null: set[str] = set(data.get("maybe_null", set()))

        for instr in block.instructions:
            if instr.opcode == "const" and instr.result:
                if instr.operands and instr.operands[0] == "None":
                    def_null.add(instr.result)
                else:
                    def_null.discard(instr.result)
                    maybe_null.discard(instr.result)
            elif instr.opcode == "call" and instr.result:
                maybe_null.add(instr.result)
                def_null.discard(instr.result)
            elif instr.result:
                is_null = any(op in def_null for op in instr.operands)
                maybe = any(op in maybe_null for op in instr.operands)
                def_null.discard(instr.result)
                maybe_null.discard(instr.result)
                if is_null:
                    maybe_null.add(instr.result)
                elif maybe:
                    maybe_null.add(instr.result)

        return DataflowFact(data={"definitely_null": def_null, "maybe_null": maybe_null})


class Sign(Enum):
    BOTTOM = auto()
    NEGATIVE = auto()
    ZERO = auto()
    POSITIVE = auto()
    NON_NEGATIVE = auto()
    NON_POSITIVE = auto()
    NON_ZERO = auto()
    TOP = auto()


@dataclass
class SignAnalysis(ForwardAnalysis):
    """Tracks sign (positive/negative/zero) of numeric values."""
    name: str = "sign"

    def init_fact(self) -> DataflowFact:
        return DataflowFact(data={})

    def entry_fact(self) -> DataflowFact:
        return DataflowFact(data={})

    def _sign_of(self, val: Any) -> Sign:
        if isinstance(val, (int, float)):
            if val > 0:
                return Sign.POSITIVE
            elif val < 0:
                return Sign.NEGATIVE
            else:
                return Sign.ZERO
        return Sign.TOP

    def _join_signs(self, a: Sign, b: Sign) -> Sign:
        if a == b:
            return a
        if a == Sign.BOTTOM:
            return b
        if b == Sign.BOTTOM:
            return a
        if {a, b} == {Sign.ZERO, Sign.POSITIVE}:
            return Sign.NON_NEGATIVE
        if {a, b} == {Sign.ZERO, Sign.NEGATIVE}:
            return Sign.NON_POSITIVE
        if {a, b} == {Sign.POSITIVE, Sign.NEGATIVE}:
            return Sign.NON_ZERO
        return Sign.TOP

    def transfer_block(self, block: BasicBlock, in_fact: DataflowFact) -> DataflowFact:
        signs: dict[str, Sign] = dict(in_fact.data) if in_fact.data else {}
        for instr in block.instructions:
            if instr.opcode == "const" and instr.result and instr.operands:
                try:
                    signs[instr.result] = self._sign_of(int(instr.operands[0]))
                except ValueError:
                    signs[instr.result] = Sign.TOP
            elif instr.result and instr.opcode == "add" and len(instr.operands) >= 2:
                a_sign = signs.get(instr.operands[0], Sign.TOP)
                b_sign = signs.get(instr.operands[1], Sign.TOP)
                if a_sign == Sign.POSITIVE and b_sign == Sign.POSITIVE:
                    signs[instr.result] = Sign.POSITIVE
                elif a_sign == Sign.NEGATIVE and b_sign == Sign.NEGATIVE:
                    signs[instr.result] = Sign.NEGATIVE
                elif a_sign == Sign.ZERO:
                    signs[instr.result] = b_sign
                elif b_sign == Sign.ZERO:
                    signs[instr.result] = a_sign
                else:
                    signs[instr.result] = Sign.TOP
            elif instr.result and instr.opcode == "mul" and len(instr.operands) >= 2:
                a_sign = signs.get(instr.operands[0], Sign.TOP)
                b_sign = signs.get(instr.operands[1], Sign.TOP)
                if a_sign == Sign.ZERO or b_sign == Sign.ZERO:
                    signs[instr.result] = Sign.ZERO
                elif a_sign == Sign.POSITIVE and b_sign == Sign.POSITIVE:
                    signs[instr.result] = Sign.POSITIVE
                elif a_sign == Sign.NEGATIVE and b_sign == Sign.NEGATIVE:
                    signs[instr.result] = Sign.POSITIVE
                elif (a_sign == Sign.POSITIVE and b_sign == Sign.NEGATIVE) or \
                     (a_sign == Sign.NEGATIVE and b_sign == Sign.POSITIVE):
                    signs[instr.result] = Sign.NEGATIVE
                else:
                    signs[instr.result] = Sign.TOP
            elif instr.result:
                signs[instr.result] = Sign.TOP
        return DataflowFact(data=signs)


class Parity(Enum):
    BOTTOM = auto()
    EVEN = auto()
    ODD = auto()
    TOP = auto()


@dataclass
class ParityAnalysis(ForwardAnalysis):
    """Tracks even/odd parity."""
    name: str = "parity"

    def init_fact(self) -> DataflowFact:
        return DataflowFact(data={})

    def entry_fact(self) -> DataflowFact:
        return DataflowFact(data={})

    def transfer_block(self, block: BasicBlock, in_fact: DataflowFact) -> DataflowFact:
        parities: dict[str, Parity] = dict(in_fact.data) if in_fact.data else {}
        for instr in block.instructions:
            if instr.opcode == "const" and instr.result and instr.operands:
                try:
                    v = int(instr.operands[0])
                    parities[instr.result] = Parity.EVEN if v % 2 == 0 else Parity.ODD
                except ValueError:
                    parities[instr.result] = Parity.TOP
            elif instr.result and instr.opcode == "add" and len(instr.operands) >= 2:
                a = parities.get(instr.operands[0], Parity.TOP)
                b = parities.get(instr.operands[1], Parity.TOP)
                if a == Parity.EVEN and b == Parity.EVEN:
                    parities[instr.result] = Parity.EVEN
                elif a == Parity.ODD and b == Parity.ODD:
                    parities[instr.result] = Parity.EVEN
                elif a in (Parity.EVEN, Parity.ODD) and b in (Parity.EVEN, Parity.ODD):
                    parities[instr.result] = Parity.ODD
                else:
                    parities[instr.result] = Parity.TOP
            elif instr.result and instr.opcode == "mul" and len(instr.operands) >= 2:
                a = parities.get(instr.operands[0], Parity.TOP)
                b = parities.get(instr.operands[1], Parity.TOP)
                if a == Parity.EVEN or b == Parity.EVEN:
                    parities[instr.result] = Parity.EVEN
                elif a == Parity.ODD and b == Parity.ODD:
                    parities[instr.result] = Parity.ODD
                else:
                    parities[instr.result] = Parity.TOP
            elif instr.result:
                parities[instr.result] = Parity.TOP
        return DataflowFact(data=parities)


# =========================================================================
#  IntervalAnalysis / OctagonAnalysis
# =========================================================================

@dataclass(frozen=True)
class Interval:
    lo: Optional[int] = None
    hi: Optional[int] = None

    @staticmethod
    def bottom() -> Interval:
        return Interval(lo=1, hi=0)

    @staticmethod
    def top() -> Interval:
        return Interval(lo=None, hi=None)

    def is_bottom(self) -> bool:
        return self.lo is not None and self.hi is not None and self.lo > self.hi

    def join(self, other: Interval) -> Interval:
        if self.is_bottom():
            return other
        if other.is_bottom():
            return self
        lo = min(self.lo, other.lo) if self.lo is not None and other.lo is not None else None
        hi = max(self.hi, other.hi) if self.hi is not None and other.hi is not None else None
        return Interval(lo=lo, hi=hi)

    def meet(self, other: Interval) -> Interval:
        if self.is_bottom() or other.is_bottom():
            return Interval.bottom()
        lo = max(self.lo, other.lo) if self.lo is not None and other.lo is not None else (self.lo or other.lo)
        hi = min(self.hi, other.hi) if self.hi is not None and other.hi is not None else (self.hi or other.hi)
        return Interval(lo=lo, hi=hi)

    def widen(self, other: Interval) -> Interval:
        if self.is_bottom():
            return other
        lo = self.lo if (other.lo is not None and self.lo is not None and other.lo >= self.lo) else None
        hi = self.hi if (other.hi is not None and self.hi is not None and other.hi <= self.hi) else None
        return Interval(lo=lo, hi=hi)


@dataclass
class IntervalAnalysis(ForwardAnalysis):
    """Interval analysis using interval domain."""
    name: str = "interval"

    def init_fact(self) -> DataflowFact:
        return DataflowFact(data={})

    def entry_fact(self) -> DataflowFact:
        return DataflowFact(data={})

    def transfer_block(self, block: BasicBlock, in_fact: DataflowFact) -> DataflowFact:
        env: dict[str, Interval] = dict(in_fact.data) if in_fact.data else {}
        for instr in block.instructions:
            if instr.opcode == "const" and instr.result and instr.operands:
                try:
                    v = int(instr.operands[0])
                    env[instr.result] = Interval(lo=v, hi=v)
                except ValueError:
                    env[instr.result] = Interval.top()
            elif instr.result and instr.opcode == "add" and len(instr.operands) >= 2:
                a = env.get(instr.operands[0], Interval.top())
                b = env.get(instr.operands[1], Interval.top())
                lo = (a.lo + b.lo) if a.lo is not None and b.lo is not None else None
                hi = (a.hi + b.hi) if a.hi is not None and b.hi is not None else None
                env[instr.result] = Interval(lo=lo, hi=hi)
            elif instr.result and instr.opcode == "sub" and len(instr.operands) >= 2:
                a = env.get(instr.operands[0], Interval.top())
                b = env.get(instr.operands[1], Interval.top())
                lo = (a.lo - b.hi) if a.lo is not None and b.hi is not None else None
                hi = (a.hi - b.lo) if a.hi is not None and b.lo is not None else None
                env[instr.result] = Interval(lo=lo, hi=hi)
            elif instr.result:
                env[instr.result] = Interval.top()
        return DataflowFact(data=env)


@dataclass
class OctagonConstraint:
    """±x_i ± x_j ≤ c."""
    var_i: str
    var_j: str
    sign_i: int = 1  # +1 or -1
    sign_j: int = -1
    bound: int = 0

    def __str__(self) -> str:
        si = "" if self.sign_i == 1 else "-"
        sj = "+" if self.sign_j == 1 else "-"
        return f"{si}{self.var_i} {sj} {self.var_j} ≤ {self.bound}"


@dataclass
class OctagonAnalysis(ForwardAnalysis):
    """Relational numeric analysis using octagon domain (difference-bound matrices)."""
    name: str = "octagon"
    _variables: List[str] = field(default_factory=list)

    def init_fact(self) -> DataflowFact:
        return DataflowFact(data={"constraints": []})

    def entry_fact(self) -> DataflowFact:
        return DataflowFact(data={"constraints": []})

    def transfer_block(self, block: BasicBlock, in_fact: DataflowFact) -> DataflowFact:
        data = in_fact.data if in_fact.data else {"constraints": []}
        constraints: list[OctagonConstraint] = list(data.get("constraints", []))

        for instr in block.instructions:
            if instr.opcode == "const" and instr.result and instr.operands:
                try:
                    v = int(instr.operands[0])
                    if instr.result not in self._variables:
                        self._variables.append(instr.result)
                    constraints = [c for c in constraints
                                   if c.var_i != instr.result and c.var_j != instr.result]
                    constraints.append(OctagonConstraint(
                        var_i=instr.result, var_j=instr.result,
                        sign_i=1, sign_j=-1, bound=0,
                    ))
                except ValueError:
                    pass
            elif instr.result and instr.opcode in ("assign", "copy") and instr.operands:
                if instr.result not in self._variables:
                    self._variables.append(instr.result)
                src = instr.operands[0]
                constraints = [c for c in constraints
                               if c.var_i != instr.result and c.var_j != instr.result]
                constraints.append(OctagonConstraint(
                    var_i=instr.result, var_j=src, sign_i=1, sign_j=-1, bound=0,
                ))
                constraints.append(OctagonConstraint(
                    var_i=src, var_j=instr.result, sign_i=1, sign_j=-1, bound=0,
                ))

        return DataflowFact(data={"constraints": constraints})


# =========================================================================
#  PointerAnalysis / StringAnalysis / ContainerSizeAnalysis
# =========================================================================

@dataclass
class PointerAnalysis(ForwardAnalysis):
    """Points-to analysis (Andersen-style inclusion-based)."""
    name: str = "pointer"

    def init_fact(self) -> DataflowFact:
        return DataflowFact(data={})

    def entry_fact(self) -> DataflowFact:
        return DataflowFact(data={})

    def transfer_block(self, block: BasicBlock, in_fact: DataflowFact) -> DataflowFact:
        pts: dict[str, set[str]] = {}
        if in_fact.data:
            for k, v in in_fact.data.items():
                pts[k] = set(v)

        for instr in block.instructions:
            if instr.opcode in ("alloc", "new") and instr.result:
                pts[instr.result] = {f"alloc_{block.label}_{instr.result}"}
            elif instr.opcode in ("assign", "copy") and instr.result and instr.operands:
                src = instr.operands[0]
                pts[instr.result] = set(pts.get(src, set()))
            elif instr.opcode == "load" and instr.result and instr.operands:
                base = instr.operands[0]
                pointed = pts.get(base, set())
                result_pts: set[str] = set()
                for obj in pointed:
                    result_pts |= pts.get(f"{obj}.*", set())
                pts[instr.result] = result_pts
            elif instr.opcode == "store" and len(instr.operands) >= 2:
                base, val = instr.operands[0], instr.operands[1]
                pointed = pts.get(base, set())
                val_pts = pts.get(val, set())
                for obj in pointed:
                    key = f"{obj}.*"
                    pts.setdefault(key, set())
                    pts[key] |= val_pts

        return DataflowFact(data=pts)


@dataclass
class StringAnalysis(ForwardAnalysis):
    """String value tracking (prefix/suffix/regex approximation)."""
    name: str = "string"

    def init_fact(self) -> DataflowFact:
        return DataflowFact(data={})

    def entry_fact(self) -> DataflowFact:
        return DataflowFact(data={})

    def transfer_block(self, block: BasicBlock, in_fact: DataflowFact) -> DataflowFact:
        env: dict[str, dict[str, Any]] = dict(in_fact.data) if in_fact.data else {}
        for instr in block.instructions:
            if instr.opcode == "const" and instr.result and instr.operands:
                val = instr.operands[0]
                if isinstance(val, str):
                    env[instr.result] = {
                        "exact": val,
                        "prefix": val[:8] if len(val) > 8 else val,
                        "suffix": val[-8:] if len(val) > 8 else val,
                        "min_len": len(val),
                        "max_len": len(val),
                    }
            elif instr.opcode == "concat" and instr.result and len(instr.operands) >= 2:
                a = env.get(instr.operands[0], {})
                b = env.get(instr.operands[1], {})
                a_min = a.get("min_len", 0)
                b_min = b.get("min_len", 0)
                a_max = a.get("max_len")
                b_max = b.get("max_len")
                env[instr.result] = {
                    "prefix": a.get("prefix", ""),
                    "suffix": b.get("suffix", ""),
                    "min_len": a_min + b_min,
                    "max_len": (a_max + b_max) if a_max is not None and b_max is not None else None,
                }
            elif instr.result:
                env[instr.result] = {"min_len": 0, "max_len": None}
        return DataflowFact(data=env)


@dataclass
class ContainerSizeAnalysis(ForwardAnalysis):
    """Tracks container sizes (lists, dicts, sets)."""
    name: str = "container-size"

    def init_fact(self) -> DataflowFact:
        return DataflowFact(data={})

    def entry_fact(self) -> DataflowFact:
        return DataflowFact(data={})

    def transfer_block(self, block: BasicBlock, in_fact: DataflowFact) -> DataflowFact:
        sizes: dict[str, Tuple[int, Optional[int]]] = dict(in_fact.data) if in_fact.data else {}
        for instr in block.instructions:
            if instr.opcode in ("list_new", "dict_new", "set_new") and instr.result:
                n = len(instr.operands)
                sizes[instr.result] = (n, n)
            elif instr.opcode == "append" and instr.operands:
                container = instr.operands[0]
                if container in sizes:
                    lo, hi = sizes[container]
                    sizes[container] = (lo + 1, hi + 1 if hi is not None else None)
            elif instr.opcode == "pop" and instr.operands:
                container = instr.operands[0]
                if container in sizes:
                    lo, hi = sizes[container]
                    sizes[container] = (max(0, lo - 1), hi - 1 if hi is not None and hi > 0 else hi)
            elif instr.opcode == "extend" and len(instr.operands) >= 2:
                dst = instr.operands[0]
                src = instr.operands[1]
                if dst in sizes and src in sizes:
                    d_lo, d_hi = sizes[dst]
                    s_lo, s_hi = sizes[src]
                    sizes[dst] = (
                        d_lo + s_lo,
                        (d_hi + s_hi) if d_hi is not None and s_hi is not None else None,
                    )
        return DataflowFact(data=sizes)


# =========================================================================
#  Dominator trees
# =========================================================================

@dataclass
class DominatorTree:
    """Dominator tree computation (iterative algorithm)."""

    _idom: Dict[str, Optional[str]] = field(default_factory=dict)
    _dom_tree: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    _dom_set: Dict[str, Set[str]] = field(default_factory=dict)

    def compute(self, cfg: CFG) -> None:
        labels = cfg.reverse_postorder()
        rpo_index = {l: i for i, l in enumerate(labels)}
        self._idom = {l: None for l in labels}
        self._idom[cfg.entry] = cfg.entry

        def intersect(a: str, b: str) -> str:
            finger1, finger2 = a, b
            while finger1 != finger2:
                while rpo_index.get(finger1, 0) > rpo_index.get(finger2, 0):
                    finger1 = self._idom.get(finger1) or cfg.entry
                while rpo_index.get(finger2, 0) > rpo_index.get(finger1, 0):
                    finger2 = self._idom.get(finger2) or cfg.entry
            return finger1

        changed = True
        while changed:
            changed = False
            for label in labels:
                if label == cfg.entry:
                    continue
                preds = [p for p in cfg.predecessors(label) if self._idom.get(p) is not None]
                if not preds:
                    continue
                new_idom = preds[0]
                for pred in preds[1:]:
                    new_idom = intersect(new_idom, pred)
                if self._idom[label] != new_idom:
                    self._idom[label] = new_idom
                    changed = True

        self._dom_tree.clear()
        for label, idom in self._idom.items():
            if idom is not None and idom != label:
                self._dom_tree[idom].add(label)

        self._compute_dom_sets(cfg)

    def _compute_dom_sets(self, cfg: CFG) -> None:
        self._dom_set.clear()
        for label in cfg.all_labels():
            doms: set[str] = set()
            cur: Optional[str] = label
            while cur is not None and cur not in doms:
                doms.add(cur)
                cur = self._idom.get(cur)
            self._dom_set[label] = doms

    def idom(self, label: str) -> Optional[str]:
        result = self._idom.get(label)
        return result if result != label else None

    def dominates(self, a: str, b: str) -> bool:
        return a in self._dom_set.get(b, set())

    def dominated_by(self, label: str) -> Set[str]:
        return set(self._dom_set.get(label, set()))

    def children(self, label: str) -> Set[str]:
        return set(self._dom_tree.get(label, set()))

    def all_nodes(self) -> Set[str]:
        return set(self._idom.keys())


@dataclass
class PostDominatorTree:
    """Post-dominator tree (dominator tree on reversed CFG)."""

    _idom: Dict[str, Optional[str]] = field(default_factory=dict)
    _dom_tree: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    def compute(self, cfg: CFG) -> None:
        rev_cfg = CFG(
            blocks={},
            entry=cfg.exit,
            exit=cfg.entry,
        )
        for label, bb in cfg.blocks.items():
            rev_cfg.blocks[label] = BasicBlock(
                label=label,
                instructions=bb.instructions,
                successors=list(bb.predecessors),
                predecessors=list(bb.successors),
            )
        dom_tree = DominatorTree()
        dom_tree.compute(rev_cfg)
        self._idom = dict(dom_tree._idom)
        self._dom_tree = dict(dom_tree._dom_tree)

    def idom(self, label: str) -> Optional[str]:
        result = self._idom.get(label)
        return result if result != label else None

    def post_dominates(self, a: str, b: str) -> bool:
        cur: Optional[str] = b
        while cur is not None:
            if cur == a:
                return True
            parent = self._idom.get(cur)
            if parent == cur:
                break
            cur = parent
        return False


# =========================================================================
#  DominanceFrontier / ControlDependence / DataDependence
# =========================================================================

@dataclass
class DominanceFrontier:
    """Dominance frontier computation."""
    _frontiers: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    def compute(self, cfg: CFG, dom_tree: DominatorTree) -> None:
        self._frontiers.clear()
        for label in cfg.all_labels():
            preds = cfg.predecessors(label)
            if len(preds) >= 2:
                idom_label = dom_tree.idom(label)
                for pred in preds:
                    runner: Optional[str] = pred
                    while runner is not None and runner != idom_label:
                        self._frontiers[runner].add(label)
                        runner = dom_tree.idom(runner)

    def frontier(self, label: str) -> Set[str]:
        return set(self._frontiers.get(label, set()))

    def iterated_frontier(self, labels: Set[str]) -> Set[str]:
        result: set[str] = set()
        worklist = set(labels)
        while worklist:
            node = worklist.pop()
            for f in self._frontiers.get(node, set()):
                if f not in result:
                    result.add(f)
                    worklist.add(f)
        return result


@dataclass
class ControlDependence:
    """Control dependence analysis."""
    _deps: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    def compute(self, cfg: CFG, pdom_tree: PostDominatorTree) -> None:
        self._deps.clear()
        for label, bb in cfg.blocks.items():
            for succ in bb.successors:
                if not pdom_tree.post_dominates(succ, label):
                    runner: Optional[str] = label
                    while runner is not None:
                        self._deps[runner].add(label)
                        if pdom_tree.post_dominates(runner, label):
                            break
                        runner = pdom_tree.idom(runner)

    def dependences(self, label: str) -> Set[str]:
        return set(self._deps.get(label, set()))


@dataclass
class DataDependence:
    """Data dependence analysis using reaching definitions."""
    _deps: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    def compute(self, cfg: CFG, reaching: DataflowResult) -> None:
        self._deps.clear()
        for label, bb in cfg.blocks.items():
            in_defs = reaching.get_in(label)
            defs: set[str] = in_defs.data if in_defs.data else set()
            for instr in bb.instructions:
                for op in instr.operands:
                    for d in defs:
                        parts = d.split(":")
                        if len(parts) >= 3 and parts[2] == op:
                            self._deps[label].add(parts[0])

    def dependences(self, label: str) -> Set[str]:
        return set(self._deps.get(label, set()))


# =========================================================================
#  DefUseChain / UseDefChain
# =========================================================================

@dataclass
class DefUseChain:
    """Def-use chain construction."""
    _chains: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    def build(self, cfg: CFG) -> None:
        self._chains.clear()
        defs: dict[str, list[str]] = defaultdict(list)
        uses: dict[str, list[tuple[str, int]]] = defaultdict(list)

        for label, bb in cfg.blocks.items():
            for i, instr in enumerate(bb.instructions):
                if instr.result:
                    def_key = f"{label}:{i}:{instr.result}"
                    defs[instr.result].append(def_key)
                for op in instr.operands:
                    uses[op].append((label, i))

        for var, use_sites in uses.items():
            for def_key in defs.get(var, []):
                for use_label, use_idx in use_sites:
                    self._chains[def_key].add(f"{use_label}:{use_idx}")

    def uses_of(self, definition: str) -> Set[str]:
        return set(self._chains.get(definition, set()))


@dataclass
class UseDefChain:
    """Use-def chain construction."""
    _chains: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    def build(self, cfg: CFG) -> None:
        self._chains.clear()
        defs: dict[str, list[str]] = defaultdict(list)

        for label, bb in cfg.blocks.items():
            for i, instr in enumerate(bb.instructions):
                if instr.result:
                    defs[instr.result].append(f"{label}:{i}")
                for op in instr.operands:
                    use_key = f"{label}:{i}:{op}"
                    for dk in defs.get(op, []):
                        self._chains[use_key].add(dk)

    def defs_of(self, use: str) -> Set[str]:
        return set(self._chains.get(use, set()))


# =========================================================================
#  ProgramSlicing
# =========================================================================

@dataclass
class ProgramSlicing:
    """Forward and backward program slicing."""

    def backward_slice(self, cfg: CFG, criterion: ProgramPoint, variable: str) -> Set[str]:
        """Compute backward slice from criterion for variable."""
        relevant: set[str] = {criterion.block}
        relevant_vars: set[str] = {variable}
        worklist = deque([criterion.block])
        visited: set[str] = set()

        while worklist:
            label = worklist.popleft()
            if label in visited:
                continue
            visited.add(label)
            bb = cfg.blocks.get(label)
            if bb is None:
                continue
            for instr in reversed(bb.instructions):
                if instr.result and instr.result in relevant_vars:
                    relevant_vars.discard(instr.result)
                    for op in instr.operands:
                        relevant_vars.add(op)
                    relevant.add(label)
            for pred in cfg.predecessors(label):
                if pred not in visited:
                    worklist.append(pred)
                    relevant.add(pred)
        return relevant

    def forward_slice(self, cfg: CFG, criterion: ProgramPoint, variable: str) -> Set[str]:
        """Compute forward slice from criterion for variable."""
        relevant: set[str] = {criterion.block}
        relevant_vars: set[str] = {variable}
        worklist = deque([criterion.block])
        visited: set[str] = set()

        while worklist:
            label = worklist.popleft()
            if label in visited:
                continue
            visited.add(label)
            bb = cfg.blocks.get(label)
            if bb is None:
                continue
            for instr in bb.instructions:
                if any(op in relevant_vars for op in instr.operands):
                    relevant.add(label)
                    if instr.result:
                        relevant_vars.add(instr.result)
            for succ in cfg.successors(label):
                if succ not in visited:
                    worklist.append(succ)
        return relevant


# =========================================================================
#  SSA Construction / Phi Elimination
# =========================================================================

@dataclass
class PhiFunction:
    """Phi function: result = φ(operands from predecessors)."""
    result: str
    operands: Dict[str, str] = field(default_factory=dict)  # pred_label → variable

    def __str__(self) -> str:
        args = ", ".join(f"{v} from {p}" for p, v in self.operands.items())
        return f"{self.result} = φ({args})"


@dataclass
class SSABlock:
    """Basic block in SSA form."""
    label: str
    phis: List[PhiFunction] = field(default_factory=list)
    instructions: List[Instruction] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)
    predecessors: List[str] = field(default_factory=list)


@dataclass
class SSAConstruction:
    """SSA form construction with phi insertion (Cytron et al. algorithm)."""

    _ssa_blocks: Dict[str, SSABlock] = field(default_factory=dict)
    _var_counter: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _var_stack: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))

    def construct(self, cfg: CFG) -> Dict[str, SSABlock]:
        dom_tree = DominatorTree()
        dom_tree.compute(cfg)
        df = DominanceFrontier()
        df.compute(cfg, dom_tree)

        # Initialize SSA blocks
        for label, bb in cfg.blocks.items():
            self._ssa_blocks[label] = SSABlock(
                label=label,
                instructions=list(bb.instructions),
                successors=list(bb.successors),
                predecessors=list(bb.predecessors),
            )

        # Phase 1: Phi insertion
        defs_of: dict[str, set[str]] = defaultdict(set)
        for label, bb in cfg.blocks.items():
            for instr in bb.instructions:
                if instr.result:
                    defs_of[instr.result].add(label)

        for var, def_blocks in defs_of.items():
            phi_blocks = df.iterated_frontier(def_blocks)
            for pb in phi_blocks:
                if pb in self._ssa_blocks:
                    ssa_bb = self._ssa_blocks[pb]
                    if not any(phi.result.split("_")[0] == var for phi in ssa_bb.phis):
                        phi = PhiFunction(result=var, operands={})
                        for pred in ssa_bb.predecessors:
                            phi.operands[pred] = var
                        ssa_bb.phis.append(phi)

        # Phase 2: Renaming
        self._rename(cfg.entry, dom_tree, cfg)

        return dict(self._ssa_blocks)

    def _fresh_name(self, var: str) -> str:
        self._var_counter[var] += 1
        return f"{var}_{self._var_counter[var]}"

    def _current_name(self, var: str) -> str:
        stack = self._var_stack.get(var, [])
        return stack[-1] if stack else var

    def _rename(self, label: str, dom_tree: DominatorTree, cfg: CFG) -> None:
        ssa_bb = self._ssa_blocks.get(label)
        if ssa_bb is None:
            return

        saved_stacks: dict[str, int] = {}
        for var in self._var_stack:
            saved_stacks[var] = len(self._var_stack[var])

        # Rename phi results
        for phi in ssa_bb.phis:
            new_name = self._fresh_name(phi.result)
            self._var_stack[phi.result].append(new_name)
            phi.result = new_name

        # Rename instructions
        new_instrs: list[Instruction] = []
        for instr in ssa_bb.instructions:
            new_ops = tuple(self._current_name(op) for op in instr.operands)
            new_result = instr.result
            if instr.result:
                new_result = self._fresh_name(instr.result)
                self._var_stack[instr.result].append(new_result)
            new_instrs.append(Instruction(
                opcode=instr.opcode,
                operands=new_ops,
                result=new_result,
                location=instr.location,
            ))
        ssa_bb.instructions = new_instrs

        # Fill phi operands in successors
        for succ_label in ssa_bb.successors:
            succ = self._ssa_blocks.get(succ_label)
            if succ:
                for phi in succ.phis:
                    base_var = phi.result.split("_")[0] if "_" in phi.result else phi.result
                    if label in phi.operands:
                        phi.operands[label] = self._current_name(base_var)

        # Recurse into dominator tree children
        for child in dom_tree.children(label):
            self._rename(child, dom_tree, cfg)

        # Restore stacks
        for var, saved_len in saved_stacks.items():
            while len(self._var_stack[var]) > saved_len:
                self._var_stack[var].pop()


@dataclass
class PhiElimination:
    """SSA destruction: replace phi functions with copies."""

    def eliminate(self, ssa_blocks: Dict[str, SSABlock]) -> Dict[str, BasicBlock]:
        result: dict[str, BasicBlock] = {}
        for label, ssa_bb in ssa_blocks.items():
            instrs: list[Instruction] = []
            for phi in ssa_bb.phis:
                for pred_label, src_var in phi.operands.items():
                    pred_bb = ssa_blocks.get(pred_label)
                    if pred_bb:
                        copy_instr = Instruction(
                            opcode="copy",
                            operands=(src_var,),
                            result=phi.result,
                        )
                        instrs.append(copy_instr)
                        break  # place in current block for simplicity
            instrs.extend(ssa_bb.instructions)
            result[label] = BasicBlock(
                label=label,
                instructions=instrs,
                successors=list(ssa_bb.successors),
                predecessors=list(ssa_bb.predecessors),
            )
        return result

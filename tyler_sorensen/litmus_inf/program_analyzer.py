"""
Program analyzer for concurrent programs.

Analyzes concurrent programs to extract thread structure, shared variables,
dependencies, critical sections, synchronization primitives, and data flow.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
from enum import Enum
import re
import copy


class InstructionType(Enum):
    LOAD = "load"
    STORE = "store"
    FENCE = "fence"
    LOCK = "lock"
    UNLOCK = "unlock"
    BARRIER = "barrier"
    SEMAPHORE_WAIT = "sem_wait"
    SEMAPHORE_SIGNAL = "sem_signal"
    CONDVAR_WAIT = "cond_wait"
    CONDVAR_SIGNAL = "cond_signal"
    CONDVAR_BROADCAST = "cond_broadcast"
    CAS = "cas"
    FETCH_ADD = "fetch_add"
    EXCHANGE = "exchange"
    BRANCH = "branch"
    COMPUTE = "compute"
    NOP = "nop"


class DependencyType(Enum):
    DATA = "data"
    CONTROL = "control"
    OUTPUT = "output"
    ANTI = "anti"


class SyncPrimitiveType(Enum):
    MUTEX = "mutex"
    RWLOCK = "rwlock"
    BARRIER = "barrier"
    SEMAPHORE = "semaphore"
    CONDVAR = "condvar"
    SPINLOCK = "spinlock"
    ATOMIC = "atomic"


@dataclass
class Instruction:
    thread_id: int
    index: int
    inst_type: InstructionType
    variable: Optional[str] = None
    value: Optional[Any] = None
    target: Optional[int] = None
    label: Optional[str] = None
    lock_id: Optional[str] = None
    condition: Optional[str] = None
    memory_order: str = "seq_cst"

    def __repr__(self):
        parts = [f"T{self.thread_id}:{self.index} {self.inst_type.value}"]
        if self.variable:
            parts.append(f"var={self.variable}")
        if self.value is not None:
            parts.append(f"val={self.value}")
        if self.lock_id:
            parts.append(f"lock={self.lock_id}")
        return " ".join(parts)


@dataclass
class Dependency:
    dep_type: DependencyType
    source: Instruction
    target: Instruction
    variable: Optional[str] = None

    def __repr__(self):
        return (f"{self.dep_type.value}: {self.source} -> {self.target}"
                f" [{self.variable}]" if self.variable else "")


@dataclass
class CriticalSection:
    thread_id: int
    lock_id: str
    start_index: int
    end_index: int
    instructions: List[Instruction] = field(default_factory=list)
    variables_accessed: Set[str] = field(default_factory=set)
    nested_locks: List[str] = field(default_factory=list)

    @property
    def length(self):
        return self.end_index - self.start_index + 1


@dataclass
class SyncPoint:
    sync_type: SyncPrimitiveType
    thread_id: int
    index: int
    identifier: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HappensBeforeEdge:
    source_thread: int
    source_index: int
    target_thread: int
    target_index: int
    reason: str


@dataclass
class DataFlowFact:
    variable: str
    defined_at: Tuple[int, int]  # (thread_id, index)
    reaching: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class ProgramAnalysis:
    threads: Dict[int, List[Instruction]]
    shared_vars: Set[str]
    dependencies: List[Dependency]
    critical_sections: List[CriticalSection]
    sync_points: List[SyncPoint]
    data_races_possible: List[Tuple[Instruction, Instruction]]
    happens_before: List[HappensBeforeEdge]
    reaching_definitions: Dict[str, List[DataFlowFact]]
    available_expressions: Dict[str, Set[str]]
    atomicity_violations: List[Tuple[int, int, int, str]]
    thread_count: int = 0
    shared_var_count: int = 0
    total_instructions: int = 0

    def summary(self) -> Dict[str, Any]:
        return {
            "threads": self.thread_count,
            "shared_variables": self.shared_var_count,
            "total_instructions": self.total_instructions,
            "dependencies": len(self.dependencies),
            "critical_sections": len(self.critical_sections),
            "sync_points": len(self.sync_points),
            "potential_data_races": len(self.data_races_possible),
            "happens_before_edges": len(self.happens_before),
            "atomicity_violations": len(self.atomicity_violations),
        }


@dataclass
class ConcurrentProgram:
    threads: Dict[int, List[Instruction]]
    shared_variables: Set[str] = field(default_factory=set)
    initial_state: Dict[str, Any] = field(default_factory=dict)
    locks: Set[str] = field(default_factory=set)
    barriers: Set[str] = field(default_factory=set)
    name: str = ""

    @property
    def thread_count(self):
        return len(self.threads)

    @property
    def total_instructions(self):
        return sum(len(insts) for insts in self.threads.values())


class ThreadExtractor:
    """Extract per-thread instruction sequences from a program specification."""

    def __init__(self):
        self._thread_patterns = {
            "load": re.compile(r"(\w+)\s*=\s*(\w+)"),
            "store": re.compile(r"(\w+)\s*=\s*(\d+|true|false)"),
            "lock": re.compile(r"lock\s*\(\s*(\w+)\s*\)"),
            "unlock": re.compile(r"unlock\s*\(\s*(\w+)\s*\)"),
            "barrier": re.compile(r"barrier\s*\(\s*(\w+)\s*\)"),
            "cas": re.compile(r"CAS\s*\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*\)"),
            "fence": re.compile(r"fence\s*\(\s*(\w+)\s*\)"),
        }

    def extract_threads(self, program: ConcurrentProgram) -> Dict[int, List[Instruction]]:
        result = {}
        for tid, instructions in program.threads.items():
            thread_insts = []
            for i, inst in enumerate(instructions):
                inst_copy = copy.copy(inst)
                inst_copy.thread_id = tid
                inst_copy.index = i
                thread_insts.append(inst_copy)
            result[tid] = thread_insts
        return result

    def parse_text_program(self, text: str) -> ConcurrentProgram:
        threads = {}
        current_thread = -1
        shared_vars = set()
        locks = set()
        barriers = set()

        for line in text.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            thread_match = re.match(r"Thread\s+(\d+):", line, re.IGNORECASE)
            if thread_match:
                current_thread = int(thread_match.group(1))
                threads[current_thread] = []
                continue
            if current_thread < 0:
                continue

            idx = len(threads[current_thread])
            inst = self._parse_instruction(line, current_thread, idx)
            if inst:
                threads[current_thread].append(inst)
                if inst.variable:
                    shared_vars.add(inst.variable)
                if inst.lock_id:
                    locks.add(inst.lock_id)
                if inst.inst_type == InstructionType.BARRIER and inst.label:
                    barriers.add(inst.label)

        return ConcurrentProgram(
            threads=threads,
            shared_variables=shared_vars,
            locks=locks,
            barriers=barriers,
        )

    def _parse_instruction(self, line: str, tid: int, idx: int) -> Optional[Instruction]:
        line_lower = line.lower().strip()

        if line_lower.startswith("lock"):
            m = self._thread_patterns["lock"].search(line)
            lock_id = m.group(1) if m else "L0"
            return Instruction(tid, idx, InstructionType.LOCK, lock_id=lock_id)

        if line_lower.startswith("unlock"):
            m = self._thread_patterns["unlock"].search(line)
            lock_id = m.group(1) if m else "L0"
            return Instruction(tid, idx, InstructionType.UNLOCK, lock_id=lock_id)

        if line_lower.startswith("barrier"):
            m = self._thread_patterns["barrier"].search(line)
            label = m.group(1) if m else "B0"
            return Instruction(tid, idx, InstructionType.BARRIER, label=label)

        if line_lower.startswith("fence"):
            m = self._thread_patterns["fence"].search(line)
            order = m.group(1) if m else "seq_cst"
            return Instruction(tid, idx, InstructionType.FENCE, memory_order=order)

        if "cas" in line_lower:
            m = self._thread_patterns["cas"].search(line)
            if m:
                return Instruction(tid, idx, InstructionType.CAS,
                                   variable=m.group(1), value=(m.group(2), m.group(3)))

        store_match = re.match(r"(\w+)\s*=\s*(\d+)", line)
        if store_match:
            return Instruction(tid, idx, InstructionType.STORE,
                               variable=store_match.group(1),
                               value=int(store_match.group(2)))

        load_match = re.match(r"r(\d+)\s*=\s*(\w+)", line)
        if load_match:
            return Instruction(tid, idx, InstructionType.LOAD,
                               variable=load_match.group(2),
                               value=f"r{load_match.group(1)}")

        return Instruction(tid, idx, InstructionType.NOP, label=line)


class SharedVariableDetector:
    """Detect variables accessed by multiple threads."""

    def detect(self, threads: Dict[int, List[Instruction]]) -> Set[str]:
        var_threads: Dict[str, Set[int]] = {}
        for tid, instructions in threads.items():
            for inst in instructions:
                if inst.variable:
                    if inst.variable not in var_threads:
                        var_threads[inst.variable] = set()
                    var_threads[inst.variable].add(tid)

        shared = set()
        for var, accessing_threads in var_threads.items():
            if len(accessing_threads) > 1:
                shared.add(var)
        return shared

    def get_access_pattern(self, threads: Dict[int, List[Instruction]],
                           variable: str) -> Dict[int, List[Tuple[int, InstructionType]]]:
        pattern = {}
        for tid, instructions in threads.items():
            accesses = []
            for inst in instructions:
                if inst.variable == variable:
                    accesses.append((inst.index, inst.inst_type))
            if accesses:
                pattern[tid] = accesses
        return pattern

    def classify_sharing(self, threads: Dict[int, List[Instruction]],
                         shared_vars: Set[str]) -> Dict[str, str]:
        classifications = {}
        for var in shared_vars:
            pattern = self.get_access_pattern(threads, var)
            has_write = False
            multi_write = False
            write_threads = set()
            for tid, accesses in pattern.items():
                for _, itype in accesses:
                    if itype in (InstructionType.STORE, InstructionType.CAS,
                                 InstructionType.FETCH_ADD, InstructionType.EXCHANGE):
                        has_write = True
                        write_threads.add(tid)
            multi_write = len(write_threads) > 1

            if multi_write:
                classifications[var] = "write-write-shared"
            elif has_write:
                classifications[var] = "read-write-shared"
            else:
                classifications[var] = "read-only-shared"
        return classifications


class DependencyAnalyzer:
    """Analyze data, control, output, and anti dependencies."""

    def analyze(self, threads: Dict[int, List[Instruction]]) -> List[Dependency]:
        deps = []
        for tid, instructions in threads.items():
            deps.extend(self._intra_thread_data_deps(instructions))
            deps.extend(self._intra_thread_control_deps(instructions))
            deps.extend(self._intra_thread_output_deps(instructions))
            deps.extend(self._intra_thread_anti_deps(instructions))
        deps.extend(self._inter_thread_deps(threads))
        return deps

    def _intra_thread_data_deps(self, instructions: List[Instruction]) -> List[Dependency]:
        deps = []
        last_write: Dict[str, Instruction] = {}
        for inst in instructions:
            if inst.inst_type in (InstructionType.LOAD, InstructionType.CAS) and inst.variable:
                if inst.variable in last_write:
                    deps.append(Dependency(DependencyType.DATA,
                                           last_write[inst.variable], inst, inst.variable))
            if inst.inst_type in (InstructionType.STORE, InstructionType.CAS,
                                  InstructionType.FETCH_ADD, InstructionType.EXCHANGE):
                if inst.variable:
                    last_write[inst.variable] = inst
        return deps

    def _intra_thread_control_deps(self, instructions: List[Instruction]) -> List[Dependency]:
        deps = []
        branch_stack = []
        for inst in instructions:
            if inst.inst_type == InstructionType.BRANCH:
                branch_stack.append(inst)
            elif branch_stack:
                for branch in branch_stack:
                    deps.append(Dependency(DependencyType.CONTROL, branch, inst))
                if inst.inst_type in (InstructionType.BRANCH,) or inst.label == "end_if":
                    branch_stack.pop() if branch_stack else None
        return deps

    def _intra_thread_output_deps(self, instructions: List[Instruction]) -> List[Dependency]:
        deps = []
        last_write: Dict[str, Instruction] = {}
        for inst in instructions:
            if inst.inst_type in (InstructionType.STORE, InstructionType.CAS,
                                  InstructionType.FETCH_ADD, InstructionType.EXCHANGE):
                if inst.variable and inst.variable in last_write:
                    deps.append(Dependency(DependencyType.OUTPUT,
                                           last_write[inst.variable], inst, inst.variable))
                if inst.variable:
                    last_write[inst.variable] = inst
        return deps

    def _intra_thread_anti_deps(self, instructions: List[Instruction]) -> List[Dependency]:
        deps = []
        last_read: Dict[str, Instruction] = {}
        for inst in instructions:
            if inst.inst_type in (InstructionType.STORE, InstructionType.CAS):
                if inst.variable and inst.variable in last_read:
                    deps.append(Dependency(DependencyType.ANTI,
                                           last_read[inst.variable], inst, inst.variable))
            if inst.inst_type == InstructionType.LOAD and inst.variable:
                last_read[inst.variable] = inst
        return deps

    def _inter_thread_deps(self, threads: Dict[int, List[Instruction]]) -> List[Dependency]:
        deps = []
        all_writes: Dict[str, List[Instruction]] = {}
        all_reads: Dict[str, List[Instruction]] = {}

        for tid, instructions in threads.items():
            for inst in instructions:
                if not inst.variable:
                    continue
                if inst.inst_type in (InstructionType.STORE, InstructionType.CAS,
                                      InstructionType.FETCH_ADD):
                    all_writes.setdefault(inst.variable, []).append(inst)
                if inst.inst_type in (InstructionType.LOAD, InstructionType.CAS):
                    all_reads.setdefault(inst.variable, []).append(inst)

        for var in set(all_writes.keys()) & set(all_reads.keys()):
            for w in all_writes[var]:
                for r in all_reads[var]:
                    if w.thread_id != r.thread_id:
                        deps.append(Dependency(DependencyType.DATA, w, r, var))

        for var, writes in all_writes.items():
            for i, w1 in enumerate(writes):
                for w2 in writes[i + 1:]:
                    if w1.thread_id != w2.thread_id:
                        deps.append(Dependency(DependencyType.OUTPUT, w1, w2, var))
        return deps


class CriticalSectionDetector:
    """Detect lock-protected critical sections."""

    def detect(self, threads: Dict[int, List[Instruction]]) -> List[CriticalSection]:
        sections = []
        for tid, instructions in threads.items():
            sections.extend(self._detect_in_thread(tid, instructions))
        return sections

    def _detect_in_thread(self, tid: int,
                          instructions: List[Instruction]) -> List[CriticalSection]:
        sections = []
        lock_stack: List[Tuple[str, int]] = []

        for inst in instructions:
            if inst.inst_type == InstructionType.LOCK:
                lock_id = inst.lock_id or "default"
                lock_stack.append((lock_id, inst.index))
            elif inst.inst_type == InstructionType.UNLOCK:
                lock_id = inst.lock_id or "default"
                matching = None
                for i in range(len(lock_stack) - 1, -1, -1):
                    if lock_stack[i][0] == lock_id:
                        matching = lock_stack.pop(i)
                        break
                if matching:
                    cs_insts = [i2 for i2 in instructions
                                if matching[1] <= i2.index <= inst.index]
                    vars_accessed = set()
                    for ci in cs_insts:
                        if ci.variable:
                            vars_accessed.add(ci.variable)
                    nested = [ls[0] for ls in lock_stack]
                    sections.append(CriticalSection(
                        thread_id=tid, lock_id=lock_id,
                        start_index=matching[1], end_index=inst.index,
                        instructions=cs_insts, variables_accessed=vars_accessed,
                        nested_locks=nested))
        return sections

    def detect_overlapping(self, sections: List[CriticalSection]) -> List[Tuple[CriticalSection, CriticalSection]]:
        overlaps = []
        for i, s1 in enumerate(sections):
            for s2 in sections[i + 1:]:
                if s1.thread_id == s2.thread_id:
                    if (s1.start_index <= s2.start_index <= s1.end_index or
                            s2.start_index <= s1.start_index <= s2.end_index):
                        overlaps.append((s1, s2))
        return overlaps


class AtomicityAnalyzer:
    """Detect potentially non-atomic compound operations."""

    def analyze(self, threads: Dict[int, List[Instruction]],
                shared_vars: Set[str],
                critical_sections: List[CriticalSection]) -> List[Tuple[int, int, int, str]]:
        violations = []
        protected_ranges = {}
        for cs in critical_sections:
            key = cs.thread_id
            if key not in protected_ranges:
                protected_ranges[key] = []
            protected_ranges[key].append((cs.start_index, cs.end_index))

        for tid, instructions in threads.items():
            violations.extend(
                self._detect_check_then_act(tid, instructions, shared_vars, protected_ranges))
            violations.extend(
                self._detect_read_modify_write(tid, instructions, shared_vars, protected_ranges))
            violations.extend(
                self._detect_compound_check(tid, instructions, shared_vars, protected_ranges))
        return violations

    def _is_protected(self, tid: int, start: int, end: int,
                      protected_ranges: Dict[int, List[Tuple[int, int]]]) -> bool:
        if tid not in protected_ranges:
            return False
        for ps, pe in protected_ranges[tid]:
            if ps <= start and end <= pe:
                return True
        return False

    def _detect_check_then_act(self, tid: int, instructions: List[Instruction],
                               shared_vars: Set[str],
                               protected: Dict) -> List[Tuple[int, int, int, str]]:
        violations = []
        for i, inst in enumerate(instructions):
            if (inst.inst_type == InstructionType.LOAD and
                    inst.variable in shared_vars):
                for j in range(i + 1, min(i + 4, len(instructions))):
                    next_inst = instructions[j]
                    if (next_inst.inst_type == InstructionType.STORE and
                            next_inst.variable == inst.variable):
                        if not self._is_protected(tid, i, j, protected):
                            violations.append((tid, i, j, f"check-then-act on {inst.variable}"))
                        break
        return violations

    def _detect_read_modify_write(self, tid: int, instructions: List[Instruction],
                                  shared_vars: Set[str],
                                  protected: Dict) -> List[Tuple[int, int, int, str]]:
        violations = []
        for i, inst in enumerate(instructions):
            if inst.inst_type == InstructionType.LOAD and inst.variable in shared_vars:
                for j in range(i + 1, min(i + 5, len(instructions))):
                    if instructions[j].inst_type == InstructionType.COMPUTE:
                        for k in range(j + 1, min(j + 3, len(instructions))):
                            if (instructions[k].inst_type == InstructionType.STORE and
                                    instructions[k].variable == inst.variable):
                                if not self._is_protected(tid, i, k, protected):
                                    violations.append(
                                        (tid, i, k, f"read-modify-write on {inst.variable}"))
                                break
                        break
        return violations

    def _detect_compound_check(self, tid: int, instructions: List[Instruction],
                               shared_vars: Set[str],
                               protected: Dict) -> List[Tuple[int, int, int, str]]:
        violations = []
        for i, inst in enumerate(instructions):
            if inst.inst_type == InstructionType.LOAD and inst.variable in shared_vars:
                for j in range(i + 1, min(i + 3, len(instructions))):
                    if (instructions[j].inst_type == InstructionType.LOAD and
                            instructions[j].variable in shared_vars and
                            instructions[j].variable != inst.variable):
                        if not self._is_protected(tid, i, j, protected):
                            violations.append(
                                (tid, i, j,
                                 f"compound check on {inst.variable},{instructions[j].variable}"))
                        break
        return violations


class SyncPrimitiveDetector:
    """Detect synchronization primitives."""

    def detect(self, threads: Dict[int, List[Instruction]]) -> List[SyncPoint]:
        points = []
        for tid, instructions in threads.items():
            for inst in instructions:
                sp = self._classify_sync(inst)
                if sp:
                    points.append(sp)
        return points

    def _classify_sync(self, inst: Instruction) -> Optional[SyncPoint]:
        mapping = {
            InstructionType.LOCK: (SyncPrimitiveType.MUTEX, lambda i: i.lock_id or "default"),
            InstructionType.UNLOCK: (SyncPrimitiveType.MUTEX, lambda i: i.lock_id or "default"),
            InstructionType.BARRIER: (SyncPrimitiveType.BARRIER, lambda i: i.label or "B0"),
            InstructionType.SEMAPHORE_WAIT: (SyncPrimitiveType.SEMAPHORE, lambda i: i.label or "S0"),
            InstructionType.SEMAPHORE_SIGNAL: (SyncPrimitiveType.SEMAPHORE, lambda i: i.label or "S0"),
            InstructionType.CONDVAR_WAIT: (SyncPrimitiveType.CONDVAR, lambda i: i.label or "CV0"),
            InstructionType.CONDVAR_SIGNAL: (SyncPrimitiveType.CONDVAR, lambda i: i.label or "CV0"),
            InstructionType.CONDVAR_BROADCAST: (SyncPrimitiveType.CONDVAR, lambda i: i.label or "CV0"),
            InstructionType.CAS: (SyncPrimitiveType.ATOMIC, lambda i: i.variable or "A0"),
            InstructionType.FETCH_ADD: (SyncPrimitiveType.ATOMIC, lambda i: i.variable or "A0"),
            InstructionType.EXCHANGE: (SyncPrimitiveType.ATOMIC, lambda i: i.variable or "A0"),
        }

        if inst.inst_type in mapping:
            stype, id_fn = mapping[inst.inst_type]
            return SyncPoint(
                sync_type=stype,
                thread_id=inst.thread_id,
                index=inst.index,
                identifier=id_fn(inst),
                properties={"action": inst.inst_type.value}
            )
        return None


class HappensBeforeBuilder:
    """Build happens-before graph from synchronization operations."""

    def build(self, threads: Dict[int, List[Instruction]],
              sync_points: List[SyncPoint]) -> List[HappensBeforeEdge]:
        edges = []
        edges.extend(self._program_order_edges(threads))
        edges.extend(self._lock_edges(sync_points))
        edges.extend(self._barrier_edges(sync_points))
        edges.extend(self._signal_wait_edges(sync_points))
        return edges

    def _program_order_edges(self, threads: Dict[int, List[Instruction]]) -> List[HappensBeforeEdge]:
        edges = []
        for tid, instructions in threads.items():
            for i in range(len(instructions) - 1):
                edges.append(HappensBeforeEdge(
                    source_thread=tid, source_index=i,
                    target_thread=tid, target_index=i + 1,
                    reason="program-order"))
        return edges

    def _lock_edges(self, sync_points: List[SyncPoint]) -> List[HappensBeforeEdge]:
        edges = []
        lock_ops: Dict[str, List[SyncPoint]] = {}
        for sp in sync_points:
            if sp.sync_type == SyncPrimitiveType.MUTEX:
                lock_ops.setdefault(sp.identifier, []).append(sp)

        for lock_id, ops in lock_ops.items():
            unlocks = [op for op in ops if op.properties.get("action") == "unlock"]
            locks = [op for op in ops if op.properties.get("action") == "lock"]
            for unlock in unlocks:
                for lock in locks:
                    if unlock.thread_id != lock.thread_id:
                        edges.append(HappensBeforeEdge(
                            source_thread=unlock.thread_id, source_index=unlock.index,
                            target_thread=lock.thread_id, target_index=lock.index,
                            reason=f"lock-{lock_id}"))
        return edges

    def _barrier_edges(self, sync_points: List[SyncPoint]) -> List[HappensBeforeEdge]:
        edges = []
        barrier_ops: Dict[str, List[SyncPoint]] = {}
        for sp in sync_points:
            if sp.sync_type == SyncPrimitiveType.BARRIER:
                barrier_ops.setdefault(sp.identifier, []).append(sp)

        for barrier_id, ops in barrier_ops.items():
            for i, op1 in enumerate(ops):
                for op2 in ops[i + 1:]:
                    edges.append(HappensBeforeEdge(
                        source_thread=op1.thread_id, source_index=op1.index,
                        target_thread=op2.thread_id, target_index=op2.index,
                        reason=f"barrier-{barrier_id}"))
                    edges.append(HappensBeforeEdge(
                        source_thread=op2.thread_id, source_index=op2.index,
                        target_thread=op1.thread_id, target_index=op1.index,
                        reason=f"barrier-{barrier_id}"))
        return edges

    def _signal_wait_edges(self, sync_points: List[SyncPoint]) -> List[HappensBeforeEdge]:
        edges = []
        signal_ops: Dict[str, List[SyncPoint]] = {}
        wait_ops: Dict[str, List[SyncPoint]] = {}

        for sp in sync_points:
            if sp.sync_type in (SyncPrimitiveType.SEMAPHORE, SyncPrimitiveType.CONDVAR):
                action = sp.properties.get("action", "")
                if "signal" in action or "broadcast" in action:
                    signal_ops.setdefault(sp.identifier, []).append(sp)
                elif "wait" in action:
                    wait_ops.setdefault(sp.identifier, []).append(sp)

        for ident in set(signal_ops.keys()) & set(wait_ops.keys()):
            for sig in signal_ops[ident]:
                for wait in wait_ops[ident]:
                    if sig.thread_id != wait.thread_id:
                        edges.append(HappensBeforeEdge(
                            source_thread=sig.thread_id, source_index=sig.index,
                            target_thread=wait.thread_id, target_index=wait.index,
                            reason=f"signal-wait-{ident}"))
        return edges

    def is_ordered(self, edges: List[HappensBeforeEdge],
                   t1: int, i1: int, t2: int, i2: int) -> bool:
        """Check if (t1, i1) happens-before (t2, i2) using transitive closure."""
        adj: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}
        for e in edges:
            src = (e.source_thread, e.source_index)
            tgt = (e.target_thread, e.target_index)
            adj.setdefault(src, set()).add(tgt)

        visited = set()
        stack = [(t1, i1)]
        while stack:
            node = stack.pop()
            if node == (t2, i2):
                return True
            if node in visited:
                continue
            visited.add(node)
            for neighbor in adj.get(node, set()):
                stack.append(neighbor)
        return False


class DataFlowAnalyzer:
    """Data flow analysis for concurrent programs: reaching definitions, available expressions."""

    def reaching_definitions(self, threads: Dict[int, List[Instruction]],
                             hb_edges: List[HappensBeforeEdge]) -> Dict[str, List[DataFlowFact]]:
        facts: Dict[str, List[DataFlowFact]] = {}
        all_defs: Dict[str, List[Tuple[int, int]]] = {}

        for tid, instructions in threads.items():
            for inst in instructions:
                if inst.inst_type in (InstructionType.STORE, InstructionType.CAS,
                                      InstructionType.FETCH_ADD, InstructionType.EXCHANGE):
                    if inst.variable:
                        all_defs.setdefault(inst.variable, []).append((tid, inst.index))

        hb_builder = HappensBeforeBuilder()

        for var, defs in all_defs.items():
            var_facts = []
            for def_loc in defs:
                fact = DataFlowFact(variable=var, defined_at=def_loc)
                for tid, instructions in threads.items():
                    for inst in instructions:
                        if inst.inst_type == InstructionType.LOAD and inst.variable == var:
                            use_loc = (tid, inst.index)
                            if def_loc[0] == tid and def_loc[1] < inst.index:
                                killed = False
                                for other_def in defs:
                                    if (other_def[0] == tid and
                                            def_loc[1] < other_def[1] < inst.index):
                                        killed = True
                                        break
                                if not killed:
                                    fact.reaching.append(use_loc)
                            elif def_loc[0] != tid:
                                if hb_builder.is_ordered(hb_edges,
                                                         def_loc[0], def_loc[1], tid, inst.index):
                                    fact.reaching.append(use_loc)
                                else:
                                    fact.reaching.append(use_loc)
                var_facts.append(fact)
            facts[var] = var_facts
        return facts

    def available_expressions(self, threads: Dict[int, List[Instruction]],
                              shared_vars: Set[str]) -> Dict[str, Set[str]]:
        available: Dict[str, Set[str]] = {}
        for tid, instructions in threads.items():
            current_avail: Set[str] = set()
            for inst in instructions:
                if inst.inst_type == InstructionType.STORE and inst.variable:
                    expr = f"{inst.variable}={inst.value}"
                    killed = {e for e in current_avail if e.startswith(f"{inst.variable}=")}
                    current_avail -= killed
                    current_avail.add(expr)
                elif inst.inst_type == InstructionType.LOAD and inst.variable:
                    pass  # loads don't generate new expressions
                key = f"T{tid}:{inst.index}"
                available[key] = current_avail.copy()
        return available


class DataRaceDetector:
    """Detect potential data races using happens-before analysis."""

    def detect(self, threads: Dict[int, List[Instruction]],
               shared_vars: Set[str],
               hb_edges: List[HappensBeforeEdge],
               critical_sections: List[CriticalSection]) -> List[Tuple[Instruction, Instruction]]:
        races = []
        accesses: Dict[str, List[Instruction]] = {}

        for tid, instructions in threads.items():
            for inst in instructions:
                if inst.variable and inst.variable in shared_vars:
                    accesses.setdefault(inst.variable, []).append(inst)

        hb_builder = HappensBeforeBuilder()

        for var, var_accesses in accesses.items():
            for i, a1 in enumerate(var_accesses):
                for a2 in var_accesses[i + 1:]:
                    if a1.thread_id == a2.thread_id:
                        continue
                    if (a1.inst_type == InstructionType.LOAD and
                            a2.inst_type == InstructionType.LOAD):
                        continue
                    if self._protected_by_same_lock(a1, a2, critical_sections):
                        continue
                    ordered_fwd = hb_builder.is_ordered(
                        hb_edges, a1.thread_id, a1.index, a2.thread_id, a2.index)
                    ordered_bwd = hb_builder.is_ordered(
                        hb_edges, a2.thread_id, a2.index, a1.thread_id, a1.index)
                    if not ordered_fwd and not ordered_bwd:
                        races.append((a1, a2))
        return races

    def _protected_by_same_lock(self, a1: Instruction, a2: Instruction,
                                critical_sections: List[CriticalSection]) -> bool:
        a1_locks = set()
        a2_locks = set()
        for cs in critical_sections:
            if cs.thread_id == a1.thread_id and cs.start_index <= a1.index <= cs.end_index:
                a1_locks.add(cs.lock_id)
            if cs.thread_id == a2.thread_id and cs.start_index <= a2.index <= cs.end_index:
                a2_locks.add(cs.lock_id)
        return bool(a1_locks & a2_locks)


class ProgramAnalyzer:
    """Main analyzer: orchestrates all sub-analyses."""

    def __init__(self):
        self.thread_extractor = ThreadExtractor()
        self.shared_var_detector = SharedVariableDetector()
        self.dep_analyzer = DependencyAnalyzer()
        self.cs_detector = CriticalSectionDetector()
        self.atomicity_analyzer = AtomicityAnalyzer()
        self.sync_detector = SyncPrimitiveDetector()
        self.hb_builder = HappensBeforeBuilder()
        self.dataflow = DataFlowAnalyzer()
        self.race_detector = DataRaceDetector()

    def analyze(self, program: ConcurrentProgram) -> ProgramAnalysis:
        threads = self.thread_extractor.extract_threads(program)
        shared_vars = self.shared_var_detector.detect(threads)
        dependencies = self.dep_analyzer.analyze(threads)
        critical_sections = self.cs_detector.detect(threads)
        sync_points = self.sync_detector.detect(threads)
        hb_edges = self.hb_builder.build(threads, sync_points)
        atomicity_violations = self.atomicity_analyzer.analyze(
            threads, shared_vars, critical_sections)
        reaching_defs = self.dataflow.reaching_definitions(threads, hb_edges)
        avail_exprs = self.dataflow.available_expressions(threads, shared_vars)
        data_races = self.race_detector.detect(
            threads, shared_vars, hb_edges, critical_sections)

        analysis = ProgramAnalysis(
            threads=threads,
            shared_vars=shared_vars,
            dependencies=dependencies,
            critical_sections=critical_sections,
            sync_points=sync_points,
            data_races_possible=data_races,
            happens_before=hb_edges,
            reaching_definitions=reaching_defs,
            available_expressions=avail_exprs,
            atomicity_violations=atomicity_violations,
            thread_count=len(threads),
            shared_var_count=len(shared_vars),
            total_instructions=sum(len(insts) for insts in threads.values()),
        )
        return analysis

    def analyze_text(self, program_text: str) -> ProgramAnalysis:
        program = self.thread_extractor.parse_text_program(program_text)
        return self.analyze(program)


def build_peterson_program() -> ConcurrentProgram:
    """Build Peterson's mutual exclusion algorithm as a ConcurrentProgram."""
    t0 = [
        Instruction(0, 0, InstructionType.STORE, variable="flag0", value=1),
        Instruction(0, 1, InstructionType.STORE, variable="turn", value=1),
        Instruction(0, 2, InstructionType.LOAD, variable="flag1", value="r0"),
        Instruction(0, 3, InstructionType.LOAD, variable="turn", value="r1"),
        Instruction(0, 4, InstructionType.STORE, variable="x", value=1),
        Instruction(0, 5, InstructionType.STORE, variable="flag0", value=0),
    ]
    t1 = [
        Instruction(1, 0, InstructionType.STORE, variable="flag1", value=1),
        Instruction(1, 1, InstructionType.STORE, variable="turn", value=0),
        Instruction(1, 2, InstructionType.LOAD, variable="flag0", value="r0"),
        Instruction(1, 3, InstructionType.LOAD, variable="turn", value="r1"),
        Instruction(1, 4, InstructionType.STORE, variable="x", value=2),
        Instruction(1, 5, InstructionType.STORE, variable="flag1", value=0),
    ]
    return ConcurrentProgram(
        threads={0: t0, 1: t1},
        shared_variables={"flag0", "flag1", "turn", "x"},
        initial_state={"flag0": 0, "flag1": 0, "turn": 0, "x": 0},
        name="Peterson"
    )


def build_lock_program() -> ConcurrentProgram:
    """Build a simple lock-based program."""
    t0 = [
        Instruction(0, 0, InstructionType.LOCK, lock_id="L0"),
        Instruction(0, 1, InstructionType.LOAD, variable="x", value="r0"),
        Instruction(0, 2, InstructionType.STORE, variable="x", value=1),
        Instruction(0, 3, InstructionType.UNLOCK, lock_id="L0"),
    ]
    t1 = [
        Instruction(1, 0, InstructionType.LOCK, lock_id="L0"),
        Instruction(1, 1, InstructionType.LOAD, variable="x", value="r0"),
        Instruction(1, 2, InstructionType.STORE, variable="x", value=2),
        Instruction(1, 3, InstructionType.UNLOCK, lock_id="L0"),
    ]
    return ConcurrentProgram(
        threads={0: t0, 1: t1},
        shared_variables={"x"},
        initial_state={"x": 0},
        locks={"L0"},
        name="LockProgram"
    )


def build_racy_program() -> ConcurrentProgram:
    """Build a program with data races (no synchronization)."""
    t0 = [
        Instruction(0, 0, InstructionType.STORE, variable="x", value=1),
        Instruction(0, 1, InstructionType.LOAD, variable="y", value="r0"),
    ]
    t1 = [
        Instruction(1, 0, InstructionType.STORE, variable="y", value=1),
        Instruction(1, 1, InstructionType.LOAD, variable="x", value="r0"),
    ]
    return ConcurrentProgram(
        threads={0: t0, 1: t1},
        shared_variables={"x", "y"},
        initial_state={"x": 0, "y": 0},
        name="StoreBuffer"
    )


def build_barrier_program() -> ConcurrentProgram:
    """Build a barrier-synchronized program."""
    t0 = [
        Instruction(0, 0, InstructionType.STORE, variable="data", value=42),
        Instruction(0, 1, InstructionType.BARRIER, label="B0"),
        Instruction(0, 2, InstructionType.LOAD, variable="result", value="r0"),
    ]
    t1 = [
        Instruction(1, 0, InstructionType.STORE, variable="result", value=0),
        Instruction(1, 1, InstructionType.BARRIER, label="B0"),
        Instruction(1, 2, InstructionType.LOAD, variable="data", value="r0"),
    ]
    return ConcurrentProgram(
        threads={0: t0, 1: t1},
        shared_variables={"data", "result"},
        initial_state={"data": 0, "result": 0},
        barriers={"B0"},
        name="BarrierProgram"
    )


def build_producer_consumer_program() -> ConcurrentProgram:
    """Build a producer-consumer with semaphores."""
    t0 = [
        Instruction(0, 0, InstructionType.STORE, variable="buf", value=1),
        Instruction(0, 1, InstructionType.SEMAPHORE_SIGNAL, label="items"),
        Instruction(0, 2, InstructionType.STORE, variable="buf", value=2),
        Instruction(0, 3, InstructionType.SEMAPHORE_SIGNAL, label="items"),
    ]
    t1 = [
        Instruction(1, 0, InstructionType.SEMAPHORE_WAIT, label="items"),
        Instruction(1, 1, InstructionType.LOAD, variable="buf", value="r0"),
        Instruction(1, 2, InstructionType.SEMAPHORE_WAIT, label="items"),
        Instruction(1, 3, InstructionType.LOAD, variable="buf", value="r1"),
    ]
    return ConcurrentProgram(
        threads={0: t0, 1: t1},
        shared_variables={"buf"},
        initial_state={"buf": 0},
        name="ProducerConsumer"
    )

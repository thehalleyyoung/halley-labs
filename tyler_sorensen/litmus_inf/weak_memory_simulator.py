"""Weak memory behavior simulator with store buffer models and reordering logic."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Tuple, Set
import re
import random
import time
import collections
import itertools
import copy


class MemoryOrder(Enum):
    SC = auto()
    TSO = auto()
    PSO = auto()
    ARM = auto()
    RISCV = auto()
    RELAXED = auto()


class OpType(Enum):
    LOAD = auto()
    STORE = auto()
    FENCE = auto()
    RMW = auto()
    CAS = auto()


class BufferPolicy(Enum):
    FIFO = auto()
    COALESCING = auto()
    RANDOM = auto()


@dataclass
class MemOp:
    thread_id: int
    op_type: OpType
    address: str
    value: Optional[int]
    ordering: str
    line: int

    def __str__(self) -> str:
        if self.op_type == OpType.STORE:
            return f"T{self.thread_id}:W {self.address}={self.value} @L{self.line}"
        elif self.op_type == OpType.LOAD:
            return f"T{self.thread_id}:R {self.address}->{self.value} @L{self.line}"
        elif self.op_type == OpType.FENCE:
            return f"T{self.thread_id}:FENCE({self.ordering}) @L{self.line}"
        elif self.op_type == OpType.RMW:
            return f"T{self.thread_id}:RMW {self.address}={self.value} @L{self.line}"
        elif self.op_type == OpType.CAS:
            return f"T{self.thread_id}:CAS {self.address}={self.value} @L{self.line}"
        return f"T{self.thread_id}:{self.op_type.name} {self.address} @L{self.line}"


@dataclass
class StoreBuffer:
    entries: List[Tuple[str, int]] = field(default_factory=list)
    max_size: int = 8
    policy: BufferPolicy = BufferPolicy.FIFO

    def __str__(self) -> str:
        if not self.entries:
            return "[empty]"
        items = [f"{addr}={val}" for addr, val in self.entries]
        return f"[{', '.join(items)}] ({self.policy.name})"

    def add(self, address: str, value: int) -> None:
        if self.policy == BufferPolicy.COALESCING:
            for i, (addr, _) in enumerate(self.entries):
                if addr == address:
                    self.entries[i] = (address, value)
                    return
        if len(self.entries) >= self.max_size:
            self.entries.pop(0)
        self.entries.append((address, value))

    def lookup(self, address: str) -> Optional[int]:
        for addr, val in reversed(self.entries):
            if addr == address:
                return val
        return None

    def flush_one(self) -> Optional[Tuple[str, int]]:
        if not self.entries:
            return None
        if self.policy == BufferPolicy.RANDOM:
            idx = random.randint(0, len(self.entries) - 1)
            return self.entries.pop(idx)
        return self.entries.pop(0)

    def flush_all(self) -> List[Tuple[str, int]]:
        flushed = list(self.entries)
        self.entries.clear()
        return flushed

    def is_empty(self) -> bool:
        return len(self.entries) == 0


@dataclass
class ExecutionTrace:
    ops: List[MemOp] = field(default_factory=list)
    memory_states: List[Dict[str, int]] = field(default_factory=list)
    thread_views: List[Dict[str, int]] = field(default_factory=list)
    reorderings: List[Tuple[int, int]] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"Trace: {len(self.ops)} ops, {len(self.reorderings)} reorderings"]
        for i, op in enumerate(self.ops):
            lines.append(f"  {i}: {op}")
        if self.reorderings:
            lines.append(f"  Reorderings: {self.reorderings}")
        return "\n".join(lines)


@dataclass
class WeakBehavior:
    outcome: Dict[str, int]
    trace: ExecutionTrace
    reorderings_used: List[str]
    probability: float

    def __str__(self) -> str:
        out = ", ".join(f"{k}={v}" for k, v in sorted(self.outcome.items()))
        reord = ", ".join(self.reorderings_used)
        return f"WeakBehavior({out}) p={self.probability:.4f} [{reord}]"


@dataclass
class SimulationResult:
    program_name: str
    model: str
    total_runs: int
    unique_outcomes: Dict[str, int]
    weak_behaviors: List[WeakBehavior]
    sc_violations: int
    execution_time_ms: float

    def __str__(self) -> str:
        lines = [
            f"SimulationResult({self.program_name}, model={self.model})",
            f"  Runs: {self.total_runs}, Unique outcomes: {len(self.unique_outcomes)}",
            f"  SC violations: {self.sc_violations}, Time: {self.execution_time_ms:.1f}ms",
        ]
        for key, count in sorted(self.unique_outcomes.items(), key=lambda x: -x[1]):
            lines.append(f"  {key}: {count} ({100*count/self.total_runs:.1f}%)")
        return "\n".join(lines)


@dataclass
class CoverageReport:
    model: str
    total_possible_outcomes: int
    observed_outcomes: int
    coverage_pct: float
    missing_outcomes: List[Dict[str, int]]
    runs_needed_estimate: int
    convergence_curve: List[Tuple[int, float]]

    def __str__(self) -> str:
        lines = [
            f"CoverageReport(model={self.model})",
            f"  Observed: {self.observed_outcomes}/{self.total_possible_outcomes}"
            f" ({self.coverage_pct:.1f}%)",
            f"  Est. runs for full coverage: {self.runs_needed_estimate}",
        ]
        if self.missing_outcomes:
            lines.append(f"  Missing: {len(self.missing_outcomes)} outcomes")
        return "\n".join(lines)


@dataclass
class StressResult:
    model: str
    threads: int
    duration_s: float
    total_iterations: int
    weak_behavior_rate: float
    max_latency_us: float
    outcome_distribution: Dict[str, int]

    def __str__(self) -> str:
        rate = self.total_iterations / self.duration_s if self.duration_s > 0 else 0
        return (
            f"StressResult(model={self.model}, threads={self.threads})\n"
            f"  Iterations: {self.total_iterations} ({rate:.0f}/s)\n"
            f"  Weak rate: {self.weak_behavior_rate:.4f}, "
            f"Max latency: {self.max_latency_us:.1f}us"
        )


# Model configuration: which reordering pairs are allowed
_MODEL_REORDER_RULES: Dict[str, Set[Tuple[OpType, OpType]]] = {
    "sc": set(),
    "tso": {(OpType.STORE, OpType.LOAD)},
    "pso": {(OpType.STORE, OpType.LOAD), (OpType.STORE, OpType.STORE)},
    "arm": {
        (OpType.STORE, OpType.LOAD),
        (OpType.STORE, OpType.STORE),
        (OpType.LOAD, OpType.LOAD),
        (OpType.LOAD, OpType.STORE),
    },
    "riscv": {
        (OpType.STORE, OpType.LOAD),
        (OpType.STORE, OpType.STORE),
        (OpType.LOAD, OpType.LOAD),
        (OpType.LOAD, OpType.STORE),
    },
    "relaxed": {
        (OpType.STORE, OpType.LOAD),
        (OpType.STORE, OpType.STORE),
        (OpType.LOAD, OpType.LOAD),
        (OpType.LOAD, OpType.STORE),
    },
}

_MODEL_BUFFER_POLICY: Dict[str, BufferPolicy] = {
    "sc": BufferPolicy.FIFO,
    "tso": BufferPolicy.FIFO,
    "pso": BufferPolicy.FIFO,
    "arm": BufferPolicy.COALESCING,
    "riscv": BufferPolicy.COALESCING,
    "relaxed": BufferPolicy.RANDOM,
}

_MODEL_MULTI_COPY_ATOMIC: Dict[str, bool] = {
    "sc": True,
    "tso": True,
    "pso": True,
    "arm": False,
    "riscv": False,
    "relaxed": False,
}

_MODEL_REORDER_PROB: Dict[str, float] = {
    "sc": 0.0,
    "tso": 0.3,
    "pso": 0.4,
    "arm": 0.5,
    "riscv": 0.5,
    "relaxed": 0.7,
}


class WeakMemorySimulator:
    """Simulates weak memory behaviors for litmus-test-style programs."""

    def __init__(self, model: str = "tso") -> None:
        self.model = model.lower()
        if self.model not in _MODEL_REORDER_RULES:
            raise ValueError(f"Unknown model: {model}")
        self.allowed_reorderings = _MODEL_REORDER_RULES[self.model]
        self.buffer_policy = _MODEL_BUFFER_POLICY[self.model]
        self.multi_copy_atomic = _MODEL_MULTI_COPY_ATOMIC[self.model]
        self.reorder_prob = _MODEL_REORDER_PROB[self.model]
        self.global_memory: Dict[str, int] = {}
        self.store_buffers: Dict[int, StoreBuffer] = {}
        self.thread_views: Dict[int, Dict[str, int]] = {}
        self.buffer_max_size = 8
        self._rng = random.Random()

    def __str__(self) -> str:
        return (
            f"WeakMemorySimulator(model={self.model}, "
            f"reorderings={len(self.allowed_reorderings)}, "
            f"policy={self.buffer_policy.name})"
        )

    def _reset(self, thread_ids: List[int], addresses: Set[str]) -> None:
        self.global_memory = {addr: 0 for addr in addresses}
        self.store_buffers = {
            tid: StoreBuffer(
                entries=[], max_size=self.buffer_max_size, policy=self.buffer_policy
            )
            for tid in thread_ids
        }
        self.thread_views = {tid: {addr: 0 for addr in addresses} for tid in thread_ids}

    def _parse_program(self, program: str) -> List[List[MemOp]]:
        """Parse program text into per-thread operation lists.

        Format: "T0: W x 1; T0: R y r0; T1: W y 1; T1: R x r1"
        Also supports FENCE, RMW, CAS operations.
        """
        threads: Dict[int, List[MemOp]] = {}
        parts = [p.strip() for p in program.split(";") if p.strip()]
        line_num = 0
        for part in parts:
            m = re.match(r"T(\d+)\s*:\s*(W|R|FENCE|RMW|CAS)\s*(.*)", part.strip())
            if not m:
                continue
            tid = int(m.group(1))
            op_str = m.group(2)
            rest = m.group(3).strip()
            if tid not in threads:
                threads[tid] = []
            if op_str == "W":
                tokens = rest.split()
                addr = tokens[0] if tokens else "x"
                val = int(tokens[1]) if len(tokens) > 1 else 1
                threads[tid].append(
                    MemOp(tid, OpType.STORE, addr, val, "relaxed", line_num)
                )
            elif op_str == "R":
                tokens = rest.split()
                addr = tokens[0] if tokens else "x"
                reg_name = tokens[1] if len(tokens) > 1 else f"r{line_num}"
                threads[tid].append(
                    MemOp(tid, OpType.LOAD, addr, None, reg_name, line_num)
                )
            elif op_str == "FENCE":
                fence_type = rest if rest else "full"
                threads[tid].append(
                    MemOp(tid, OpType.FENCE, "", None, fence_type, line_num)
                )
            elif op_str == "RMW":
                tokens = rest.split()
                addr = tokens[0] if tokens else "x"
                val = int(tokens[1]) if len(tokens) > 1 else 1
                threads[tid].append(
                    MemOp(tid, OpType.RMW, addr, val, "relaxed", line_num)
                )
            elif op_str == "CAS":
                tokens = rest.split()
                addr = tokens[0] if tokens else "x"
                val = int(tokens[1]) if len(tokens) > 1 else 1
                threads[tid].append(
                    MemOp(tid, OpType.CAS, addr, val, "relaxed", line_num)
                )
            line_num += 1
        sorted_tids = sorted(threads.keys())
        return [threads[tid] for tid in sorted_tids]

    def _can_reorder(self, op1: MemOp, op2: MemOp) -> bool:
        """Check if the current model allows reordering op1 before op2."""
        if op1.thread_id != op2.thread_id:
            return False
        if op1.op_type == OpType.FENCE or op2.op_type == OpType.FENCE:
            return False
        if op1.address == op2.address:
            # Same-address ops maintain order (data dependency)
            if op1.op_type == OpType.STORE and op2.op_type == OpType.LOAD:
                # Store-load to same address: store buffer forwarding, not reorder
                return False
            if op1.op_type == OpType.LOAD and op2.op_type == OpType.STORE:
                return False
            if op1.op_type == OpType.STORE and op2.op_type == OpType.STORE:
                return (OpType.STORE, OpType.STORE) in self.allowed_reorderings
        pair = (op1.op_type, op2.op_type)
        return pair in self.allowed_reorderings

    def _apply_reordering(self, ops: List[MemOp]) -> List[MemOp]:
        """Reorder operations within a thread according to the memory model."""
        if not ops or self.model == "sc":
            return list(ops)
        result = list(ops)
        n = len(result)
        reordered = False
        for i in range(n - 1):
            if self._rng.random() < self.reorder_prob:
                j = i + 1
                if self._can_reorder(result[i], result[j]):
                    result[i], result[j] = result[j], result[i]
                    reordered = True
        return result

    def _store_buffer_forward(self, thread_id: int, addr: str) -> Optional[int]:
        """Check store buffer for value forwarding (reading own writes)."""
        if thread_id not in self.store_buffers:
            return None
        return self.store_buffers[thread_id].lookup(addr)

    def _flush_store_buffer(self, thread_id: int) -> None:
        """Flush one entry from the store buffer to global memory."""
        if thread_id not in self.store_buffers:
            return
        buf = self.store_buffers[thread_id]
        entry = buf.flush_one()
        if entry is not None:
            addr, val = entry
            self.global_memory[addr] = val
            if self.multi_copy_atomic:
                for tid in self.thread_views:
                    self.thread_views[tid][addr] = val
            else:
                self.thread_views[thread_id][addr] = val
                for tid in self.thread_views:
                    if tid != thread_id and self._rng.random() < 0.7:
                        self.thread_views[tid][addr] = val

    def _execute_op(
        self,
        op: MemOp,
        memory: Dict[str, int],
        store_buffers: Dict[int, StoreBuffer],
        registers: Dict[str, int],
    ) -> Optional[int]:
        """Execute a single memory operation, returning load value if applicable."""
        tid = op.thread_id
        if op.op_type == OpType.STORE:
            if self.model == "sc":
                # SC: stores are immediately visible to all threads
                memory[op.address] = op.value
                for t in self.thread_views:
                    self.thread_views[t][op.address] = op.value
            else:
                store_buffers[tid].add(op.address, op.value)
                if not self.multi_copy_atomic:
                    self.thread_views[tid][op.address] = op.value
                # Probabilistic immediate flush
                if self._rng.random() < 0.3:
                    self._flush_store_buffer(tid)
            return None
        elif op.op_type == OpType.LOAD:
            forwarded = store_buffers[tid].lookup(op.address)
            if forwarded is not None:
                registers[op.ordering] = forwarded
                return forwarded
            if self.multi_copy_atomic:
                val = memory.get(op.address, 0)
            else:
                view = self.thread_views.get(tid, memory)
                val = view.get(op.address, 0)
            registers[op.ordering] = val
            return val
        elif op.op_type == OpType.FENCE:
            buf = store_buffers[tid]
            flushed = buf.flush_all()
            for addr, val in flushed:
                memory[addr] = val
                for t in self.thread_views:
                    self.thread_views[t][addr] = val
            return None
        elif op.op_type == OpType.RMW:
            # Atomic read-modify-write: flush buffer, read, modify, write atomically
            buf = store_buffers[tid]
            for addr, val in buf.flush_all():
                memory[addr] = val
                for t in self.thread_views:
                    self.thread_views[t][addr] = val
            old_val = memory.get(op.address, 0)
            new_val = old_val + op.value
            memory[op.address] = new_val
            for t in self.thread_views:
                self.thread_views[t][op.address] = new_val
            return old_val
        elif op.op_type == OpType.CAS:
            buf = store_buffers[tid]
            for addr, val in buf.flush_all():
                memory[addr] = val
                for t in self.thread_views:
                    self.thread_views[t][addr] = val
            current = memory.get(op.address, 0)
            if current == 0:
                memory[op.address] = op.value
                for t in self.thread_views:
                    self.thread_views[t][op.address] = op.value
                return 1  # success
            return 0  # failure
        return None

    def _execute_once(
        self, threads: List[List[MemOp]]
    ) -> Tuple[Dict[str, int], ExecutionTrace]:
        """Run a single interleaved execution of all threads."""
        addresses: Set[str] = set()
        thread_ids: List[int] = []
        for thr in threads:
            for op in thr:
                if op.address:
                    addresses.add(op.address)
                if op.thread_id not in thread_ids:
                    thread_ids.append(op.thread_id)
        if not addresses:
            addresses.add("x")
        self._reset(thread_ids, addresses)

        # Apply per-thread reordering
        reordered_threads = []
        all_reorderings: List[Tuple[int, int]] = []
        for t_idx, thr in enumerate(threads):
            original_lines = [op.line for op in thr]
            reordered = self._apply_reordering(thr)
            reordered_threads.append(reordered)
            new_lines = [op.line for op in reordered]
            for i in range(len(original_lines)):
                if i < len(new_lines) and original_lines[i] != new_lines[i]:
                    all_reorderings.append((original_lines[i], new_lines[i]))

        # Build interleaved schedule
        per_thread_idx = {tid: 0 for tid in thread_ids}
        schedule: List[MemOp] = []
        tid_to_ops: Dict[int, List[MemOp]] = {}
        for t_idx, tid in enumerate(thread_ids):
            if t_idx < len(reordered_threads):
                tid_to_ops[tid] = reordered_threads[t_idx]
            else:
                tid_to_ops[tid] = []

        total_ops = sum(len(ops) for ops in tid_to_ops.values())
        while len(schedule) < total_ops:
            active = [
                tid
                for tid in thread_ids
                if per_thread_idx[tid] < len(tid_to_ops[tid])
            ]
            if not active:
                break
            chosen_tid = self._rng.choice(active)
            idx = per_thread_idx[chosen_tid]
            schedule.append(tid_to_ops[chosen_tid][idx])
            per_thread_idx[chosen_tid] = idx + 1

        # Execute schedule
        registers: Dict[str, int] = {}
        trace_ops: List[MemOp] = []
        mem_snapshots: List[Dict[str, int]] = []
        view_snapshots: List[Dict[str, int]] = []

        for op in schedule:
            result = self._execute_op(
                op, self.global_memory, self.store_buffers, registers
            )
            executed_op = MemOp(
                op.thread_id, op.op_type, op.address,
                result if op.op_type == OpType.LOAD else op.value,
                op.ordering, op.line,
            )
            trace_ops.append(executed_op)
            mem_snapshots.append(dict(self.global_memory))
            combined_view: Dict[str, int] = {}
            for tid_views in self.thread_views.values():
                combined_view.update(tid_views)
            view_snapshots.append(combined_view)

        # Final flush of all store buffers
        for tid in thread_ids:
            buf = self.store_buffers[tid]
            for addr, val in buf.flush_all():
                self.global_memory[addr] = val
                for t in self.thread_views:
                    self.thread_views[t][addr] = val

        # Randomly flush non-MCA stores to other threads
        if not self.multi_copy_atomic:
            for tid in thread_ids:
                for addr in addresses:
                    if self._rng.random() < 0.5:
                        for other_tid in thread_ids:
                            if other_tid != tid:
                                self.thread_views[other_tid][addr] = self.global_memory[
                                    addr
                                ]

        trace = ExecutionTrace(
            ops=trace_ops,
            memory_states=mem_snapshots,
            thread_views=view_snapshots,
            reorderings=all_reorderings,
        )
        return registers, trace

    def _outcome_key(self, registers: Dict[str, int]) -> str:
        """Create a canonical string key for an outcome dictionary."""
        if not registers:
            return "{}"
        parts = [f"{k}={v}" for k, v in sorted(registers.items())]
        return "{" + ", ".join(parts) + "}"

    def _sc_outcomes(self, threads: List[List[MemOp]]) -> Set[str]:
        """Compute all possible SC outcomes by exhaustive interleaving.

        Uses itertools to enumerate all valid interleavings and executes
        each under SC semantics (no reorderings, immediate store visibility).
        Caps exploration to avoid combinatorial explosion.
        """
        total_ops = sum(len(t) for t in threads)
        if total_ops > 12:
            # Too many ops for exhaustive enumeration; approximate with SC simulation
            saved_model = self.model
            saved_reorder = self.allowed_reorderings
            saved_prob = self.reorder_prob
            self.model = "sc"
            self.allowed_reorderings = set()
            self.reorder_prob = 0.0
            outcomes: Set[str] = set()
            for _ in range(5000):
                regs, _ = self._execute_once(threads)
                outcomes.add(self._outcome_key(regs))
            self.model = saved_model
            self.allowed_reorderings = saved_reorder
            self.reorder_prob = saved_prob
            return outcomes

        # Exhaustive interleaving for small programs
        n_threads = len(threads)
        thread_lens = [len(t) for t in threads]
        outcomes: Set[str] = set()

        def _enumerate(indices: List[int], schedule: List[MemOp]) -> None:
            if all(indices[i] >= thread_lens[i] for i in range(n_threads)):
                # Execute this schedule under SC
                memory: Dict[str, int] = collections.defaultdict(int)
                registers: Dict[str, int] = {}
                for op in schedule:
                    if op.op_type == OpType.STORE:
                        memory[op.address] = op.value
                    elif op.op_type == OpType.LOAD:
                        registers[op.ordering] = memory[op.address]
                    elif op.op_type == OpType.RMW:
                        old = memory[op.address]
                        memory[op.address] = old + op.value
                    elif op.op_type == OpType.CAS:
                        if memory[op.address] == 0:
                            memory[op.address] = op.value
                    elif op.op_type == OpType.FENCE:
                        pass
                outcomes.add(self._outcome_key(registers))
                return
            if len(outcomes) > 1000:
                return
            for i in range(n_threads):
                if indices[i] < thread_lens[i]:
                    new_indices = list(indices)
                    new_indices[i] += 1
                    _enumerate(new_indices, schedule + [threads[i][indices[i]]])

        _enumerate([0] * n_threads, [])
        return outcomes

    def run(self, program: str, n_iterations: int = 10000) -> SimulationResult:
        """Run the simulation for n_iterations and collect outcome statistics."""
        start_time = time.time()
        threads = self._parse_program(program)
        if not threads:
            return SimulationResult(
                program_name=program[:40],
                model=self.model,
                total_runs=0,
                unique_outcomes={},
                weak_behaviors=[],
                sc_violations=0,
                execution_time_ms=0.0,
            )

        sc_outcomes = self._sc_outcomes(threads)
        outcome_counts: Dict[str, int] = collections.defaultdict(int)
        outcome_traces: Dict[str, ExecutionTrace] = {}
        outcome_reorderings: Dict[str, List[str]] = collections.defaultdict(list)

        for _ in range(n_iterations):
            registers, trace = self._execute_once(threads)
            key = self._outcome_key(registers)
            outcome_counts[key] += 1
            if key not in outcome_traces:
                outcome_traces[key] = trace
            if trace.reorderings and key not in sc_outcomes:
                reorder_strs = [f"L{a}<->L{b}" for a, b in trace.reorderings]
                if not outcome_reorderings[key]:
                    outcome_reorderings[key] = reorder_strs

        # Identify weak behaviors
        weak_behaviors: List[WeakBehavior] = []
        sc_violations = 0
        for key, count in outcome_counts.items():
            if key not in sc_outcomes:
                sc_violations += count
                regs = self._parse_outcome_key(key)
                wb = WeakBehavior(
                    outcome=regs,
                    trace=outcome_traces.get(key, ExecutionTrace()),
                    reorderings_used=outcome_reorderings.get(key, []),
                    probability=count / n_iterations,
                )
                weak_behaviors.append(wb)

        elapsed_ms = (time.time() - start_time) * 1000.0
        return SimulationResult(
            program_name=program[:40] if len(program) > 40 else program,
            model=self.model,
            total_runs=n_iterations,
            unique_outcomes=dict(outcome_counts),
            weak_behaviors=weak_behaviors,
            sc_violations=sc_violations,
            execution_time_ms=elapsed_ms,
        )

    def _parse_outcome_key(self, key: str) -> Dict[str, int]:
        """Parse an outcome key string back into a dict."""
        result: Dict[str, int] = {}
        inner = key.strip("{}")
        if not inner:
            return result
        for pair in inner.split(", "):
            if "=" in pair:
                k, v = pair.split("=", 1)
                try:
                    result[k] = int(v)
                except ValueError:
                    result[k] = 0
        return result

    def find_weak_behaviors(self, program: str) -> List[WeakBehavior]:
        """Find non-SC behaviors by running simulation and filtering."""
        threads = self._parse_program(program)
        if not threads:
            return []
        sc_outcomes = self._sc_outcomes(threads)
        weak_map: Dict[str, Tuple[int, ExecutionTrace, List[str]]] = {}
        n_runs = 20000

        for _ in range(n_runs):
            registers, trace = self._execute_once(threads)
            key = self._outcome_key(registers)
            if key not in sc_outcomes:
                if key not in weak_map:
                    reorder_strs = [f"L{a}<->L{b}" for a, b in trace.reorderings]
                    weak_map[key] = (1, trace, reorder_strs)
                else:
                    count, t, r = weak_map[key]
                    weak_map[key] = (count + 1, t, r)

        behaviors: List[WeakBehavior] = []
        for key, (count, trace, reorderings) in weak_map.items():
            regs = self._parse_outcome_key(key)
            behaviors.append(
                WeakBehavior(
                    outcome=regs,
                    trace=trace,
                    reorderings_used=reorderings,
                    probability=count / n_runs,
                )
            )
        behaviors.sort(key=lambda b: -b.probability)
        return behaviors

    def coverage_analysis(self, program: str, n_runs: int) -> CoverageReport:
        """Run incrementally and track coverage convergence."""
        threads = self._parse_program(program)
        if not threads:
            return CoverageReport(
                model=self.model,
                total_possible_outcomes=0,
                observed_outcomes=0,
                coverage_pct=0.0,
                missing_outcomes=[],
                runs_needed_estimate=0,
                convergence_curve=[],
            )

        # Estimate total possible outcomes analytically
        # Number of registers (loads) determines outcome space
        load_ops: List[MemOp] = []
        all_store_vals: Set[int] = {0}  # 0 is always possible (initial)
        for thr in threads:
            for op in thr:
                if op.op_type == OpType.LOAD:
                    load_ops.append(op)
                elif op.op_type == OpType.STORE and op.value is not None:
                    all_store_vals.add(op.value)
        n_regs = len(load_ops)
        n_vals = len(all_store_vals)
        total_possible = n_vals ** n_regs if n_regs > 0 else 1

        # Run incrementally and build convergence curve
        checkpoints = []
        step = 100
        while step < n_runs:
            checkpoints.append(step)
            step = min(step * 10, step + 10000)
        checkpoints.append(n_runs)

        observed: Dict[str, int] = collections.defaultdict(int)
        convergence_curve: List[Tuple[int, float]] = []
        run_count = 0
        cp_idx = 0

        for i in range(n_runs):
            registers, _ = self._execute_once(threads)
            key = self._outcome_key(registers)
            observed[key] += 1
            run_count += 1
            if cp_idx < len(checkpoints) and run_count >= checkpoints[cp_idx]:
                coverage = len(observed) / total_possible * 100.0 if total_possible > 0 else 100.0
                convergence_curve.append((run_count, coverage))
                cp_idx += 1

        observed_count = len(observed)
        coverage_pct = (
            observed_count / total_possible * 100.0 if total_possible > 0 else 100.0
        )

        # Compute SC outcomes to identify missing weak behaviors
        sc_outcomes = self._sc_outcomes(threads)
        # Find outcomes in SC that weren't observed, and vice versa
        all_expected_keys = sc_outcomes  # at minimum these should appear
        missing = []
        for key in all_expected_keys:
            if key not in observed:
                missing.append(self._parse_outcome_key(key))

        # Coupon collector estimate: E[T] = n * H_n where H_n is harmonic number
        if total_possible > observed_count and observed_count > 0:
            harmonic = sum(1.0 / k for k in range(1, total_possible + 1))
            runs_est = int(total_possible * harmonic)
        elif observed_count >= total_possible:
            runs_est = run_count
        else:
            runs_est = max(n_runs * 10, 100000)

        return CoverageReport(
            model=self.model,
            total_possible_outcomes=total_possible,
            observed_outcomes=observed_count,
            coverage_pct=coverage_pct,
            missing_outcomes=missing,
            runs_needed_estimate=runs_est,
            convergence_curve=convergence_curve,
        )

    def stress_test(
        self, program: str, n_threads: int, duration_s: float = 10.0
    ) -> StressResult:
        """Run iterations as fast as possible for duration_s, tracking metrics."""
        threads = self._parse_program(program)
        if not threads:
            return StressResult(
                model=self.model,
                threads=n_threads,
                duration_s=0.0,
                total_iterations=0,
                weak_behavior_rate=0.0,
                max_latency_us=0.0,
                outcome_distribution={},
            )

        sc_outcomes = self._sc_outcomes(threads)
        outcome_dist: Dict[str, int] = collections.defaultdict(int)
        weak_count = 0
        total_iters = 0
        max_latency_us = 0.0
        start = time.time()

        while (time.time() - start) < duration_s:
            batch_size = min(100, max(1, n_threads))
            for _ in range(batch_size):
                iter_start = time.time()
                registers, trace = self._execute_once(threads)
                iter_elapsed = (time.time() - iter_start) * 1e6
                if iter_elapsed > max_latency_us:
                    max_latency_us = iter_elapsed
                key = self._outcome_key(registers)
                outcome_dist[key] += 1
                if key not in sc_outcomes:
                    weak_count += 1
                total_iters += 1

        actual_duration = time.time() - start
        weak_rate = weak_count / total_iters if total_iters > 0 else 0.0

        return StressResult(
            model=self.model,
            threads=n_threads,
            duration_s=actual_duration,
            total_iterations=total_iters,
            weak_behavior_rate=weak_rate,
            max_latency_us=max_latency_us,
            outcome_distribution=dict(outcome_dist),
        )

    def visualize_execution(self, program: str) -> str:
        """Run one execution and generate an HTML visualization."""
        threads = self._parse_program(program)
        if not threads:
            return "<html><body><p>No program to visualize.</p></body></html>"

        sc_outcomes = self._sc_outcomes(threads)
        registers, trace = self._execute_once(threads)
        outcome_key = self._outcome_key(registers)
        is_weak = outcome_key not in sc_outcomes

        # Collect thread IDs
        all_tids: List[int] = []
        for thr in threads:
            for op in thr:
                if op.thread_id not in all_tids:
                    all_tids.append(op.thread_id)
        all_tids.sort()

        # Build per-thread timelines
        thread_timelines: Dict[int, List[str]] = {tid: [] for tid in all_tids}
        for step, op in enumerate(trace.ops):
            color = "#dc3545" if is_weak else "#28a745"
            if op.op_type == OpType.STORE:
                label = f"W {op.address}={op.value}"
                bg = "#fff3cd"
            elif op.op_type == OpType.LOAD:
                label = f"R {op.address}→{op.value}"
                bg = "#d1ecf1"
            elif op.op_type == OpType.FENCE:
                label = f"FENCE({op.ordering})"
                bg = "#e2e3e5"
            elif op.op_type == OpType.RMW:
                label = f"RMW {op.address}={op.value}"
                bg = "#d4edda"
            elif op.op_type == OpType.CAS:
                label = f"CAS {op.address}={op.value}"
                bg = "#d4edda"
            else:
                label = str(op)
                bg = "#f8f9fa"
            thread_timelines[op.thread_id].append(
                f'<div class="op" style="background:{bg};border-left:3px solid {color}">'
                f"<span class='step'>#{step}</span> {label}</div>"
            )

        # Build memory state table rows
        mem_rows = []
        for step, mem in enumerate(trace.memory_states):
            cells = "".join(f"<td>{mem.get(a, 0)}</td>" for a in sorted(mem.keys()))
            mem_rows.append(f"<tr><td>{step}</td>{cells}</tr>")

        # Build reordering annotations
        reorder_items = ""
        if trace.reorderings:
            items = [f"<li>Line {a} ↔ Line {b}</li>" for a, b in trace.reorderings]
            reorder_items = "<ul>" + "".join(items) + "</ul>"
        else:
            reorder_items = "<p>No reorderings in this execution.</p>"

        # Collect addresses for memory header
        all_addrs = sorted(
            set(a for mem in trace.memory_states for a in mem.keys()) if trace.memory_states else []
        )
        addr_headers = "".join(f"<th>{a}</th>" for a in all_addrs)

        # Compose thread columns
        thread_cols = ""
        for tid in all_tids:
            ops_html = "".join(thread_timelines[tid])
            thread_cols += (
                f'<div class="thread-col">'
                f"<h3>Thread {tid}</h3>{ops_html}</div>"
            )

        outcome_color = "#dc3545" if is_weak else "#28a745"
        outcome_label = "WEAK BEHAVIOR" if is_weak else "SC-consistent"
        outcome_str = ", ".join(f"{k}={v}" for k, v in sorted(registers.items()))

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Execution Trace</title>
<style>
body {{ font-family: monospace; margin: 20px; background: #f5f5f5; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
h1 {{ color: #333; }}
.outcome {{ padding: 10px; border-radius: 5px; color: white;
  background: {outcome_color}; display: inline-block; margin: 10px 0; }}
.timeline {{ display: flex; gap: 20px; margin: 20px 0; }}
.thread-col {{ flex: 1; background: white; padding: 10px; border-radius: 5px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.op {{ padding: 6px 8px; margin: 4px 0; border-radius: 3px; font-size: 13px; }}
.step {{ color: #666; font-size: 11px; margin-right: 5px; }}
table {{ border-collapse: collapse; margin: 10px 0; }}
th, td {{ border: 1px solid #ddd; padding: 4px 8px; text-align: center; }}
th {{ background: #f0f0f0; }}
.section {{ background: white; padding: 15px; margin: 10px 0;
  border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
</style></head><body>
<div class="container">
<h1>Weak Memory Execution Trace</h1>
<p>Model: <strong>{self.model.upper()}</strong> | Program: <code>{program[:80]}</code></p>
<div class="outcome">{outcome_label}: {outcome_str}</div>
<div class="section"><h2>Thread Timelines</h2>
<div class="timeline">{thread_cols}</div></div>
<div class="section"><h2>Memory States</h2>
<table><tr><th>Step</th>{addr_headers}</tr>{"".join(mem_rows)}</table></div>
<div class="section"><h2>Reorderings</h2>{reorder_items}</div>
<div class="section"><h2>Store Buffers (final)</h2>
<p>All buffers flushed at end of execution.</p></div>
</div></body></html>"""
        return html


if __name__ == "__main__":
    # Store buffering litmus test (SB)
    sb_program = "T0: W x 1; T0: R y r0; T1: W y 1; T1: R x r1"
    print("=== Store Buffering (SB) Litmus Test ===")
    for model_name in ["sc", "tso", "arm", "relaxed"]:
        sim = WeakMemorySimulator(model=model_name)
        result = sim.run(sb_program, n_iterations=5000)
        print(f"\n{result}")

    # Message passing litmus test (MP)
    mp_program = "T0: W x 1; T0: W y 1; T1: R y r0; T1: R x r1"
    print("\n=== Message Passing (MP) Litmus Test ===")
    sim = WeakMemorySimulator(model="arm")
    behaviors = sim.find_weak_behaviors(mp_program)
    for b in behaviors:
        print(f"  {b}")

    # Coverage analysis
    print("\n=== Coverage Analysis ===")
    sim = WeakMemorySimulator(model="tso")
    report = sim.coverage_analysis(sb_program, n_runs=10000)
    print(report)

    # Stress test
    print("\n=== Stress Test (2 seconds) ===")
    sim = WeakMemorySimulator(model="tso")
    stress = sim.stress_test(sb_program, n_threads=2, duration_s=2.0)
    print(stress)

"""
Synchronization synthesis: insert minimal synchronization to ensure safety.

Synthesizes lock placement, barrier placement, fence insertion,
atomic instruction selection, lock granularity optimization,
reader-writer lock synthesis, and optimistic synchronization.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
from enum import Enum
import copy
import itertools


class InstructionType(Enum):
    LOAD = "load"
    STORE = "store"
    FENCE = "fence"
    LOCK = "lock"
    UNLOCK = "unlock"
    BARRIER = "barrier"
    CAS = "cas"
    FETCH_ADD = "fetch_add"
    EXCHANGE = "exchange"
    BRANCH = "branch"
    COMPUTE = "compute"
    NOP = "nop"
    RWLOCK_RLOCK = "rlock"
    RWLOCK_RUNLOCK = "runlock"
    RWLOCK_WLOCK = "wlock"
    RWLOCK_WUNLOCK = "wunlock"
    TRY_LOCK = "try_lock"
    RETRY = "retry"
    VALIDATE = "validate"


class FenceType(Enum):
    FULL = "full"
    STORE_STORE = "store_store"
    LOAD_LOAD = "load_load"
    LOAD_STORE = "load_store"
    STORE_LOAD = "store_load"
    ACQUIRE = "acquire"
    RELEASE = "release"


class LockGranularity(Enum):
    COARSE = "coarse"
    FINE = "fine"
    MEDIUM = "medium"


@dataclass
class Instruction:
    thread_id: int
    index: int
    inst_type: InstructionType
    variable: str = ""
    value: Any = None
    lock_id: str = ""
    fence_type: str = ""
    label: str = ""

    def __repr__(self):
        if self.inst_type in (InstructionType.LOCK, InstructionType.UNLOCK):
            return f"T{self.thread_id}:{self.index} {self.inst_type.value}({self.lock_id})"
        if self.inst_type in (InstructionType.LOAD, InstructionType.STORE):
            return f"T{self.thread_id}:{self.index} {self.inst_type.value} {self.variable}"
        return f"T{self.thread_id}:{self.index} {self.inst_type.value}"

    def copy(self):
        return Instruction(
            self.thread_id, self.index, self.inst_type,
            self.variable, self.value, self.lock_id, self.fence_type, self.label)


@dataclass
class SafetySpec:
    mutual_exclusion_vars: Set[str] = field(default_factory=set)
    ordered_pairs: List[Tuple[str, str]] = field(default_factory=list)
    atomic_groups: List[Set[str]] = field(default_factory=list)
    no_reorder_pairs: List[Tuple[Tuple[int, int], Tuple[int, int]]] = field(default_factory=list)
    barrier_points: List[Set[int]] = field(default_factory=list)
    description: str = ""


@dataclass
class SynchronizedProgram:
    threads: Dict[int, List[Instruction]]
    added_locks: List[Tuple[int, int, str]]  # (thread_id, index, lock_id)
    added_fences: List[Tuple[int, int, FenceType]]
    added_barriers: List[Tuple[int, int, str]]
    added_atomics: List[Tuple[int, int, InstructionType]]
    sync_cost: float = 0.0
    description: str = ""

    @property
    def total_sync_added(self):
        return (len(self.added_locks) + len(self.added_fences) +
                len(self.added_barriers) + len(self.added_atomics))


class DataRaceDetector:
    """Detect data races in a concurrent program."""

    def detect(self, threads: Dict[int, List[Instruction]],
               shared_vars: Set[str]) -> List[Tuple[Instruction, Instruction]]:
        races = []
        accesses: Dict[str, List[Instruction]] = {}

        for tid, insts in threads.items():
            for inst in insts:
                if inst.variable and inst.variable in shared_vars:
                    accesses.setdefault(inst.variable, []).append(inst)

        lock_ranges: Dict[int, List[Tuple[int, int, str]]] = {}
        for tid, insts in threads.items():
            lock_stack = []
            for inst in insts:
                if inst.inst_type == InstructionType.LOCK:
                    lock_stack.append((inst.index, inst.lock_id))
                elif inst.inst_type == InstructionType.UNLOCK:
                    if lock_stack:
                        start, lid = lock_stack.pop()
                        lock_ranges.setdefault(tid, []).append((start, inst.index, lid))

        for var, var_accesses in accesses.items():
            for i, a1 in enumerate(var_accesses):
                for a2 in var_accesses[i + 1:]:
                    if a1.thread_id == a2.thread_id:
                        continue
                    if (a1.inst_type == InstructionType.LOAD and
                            a2.inst_type == InstructionType.LOAD):
                        continue
                    if not self._share_lock(a1, a2, lock_ranges):
                        races.append((a1, a2))
        return races

    def _share_lock(self, a1: Instruction, a2: Instruction,
                    lock_ranges: Dict[int, List[Tuple[int, int, str]]]) -> bool:
        locks1 = set()
        for start, end, lid in lock_ranges.get(a1.thread_id, []):
            if start <= a1.index <= end:
                locks1.add(lid)
        locks2 = set()
        for start, end, lid in lock_ranges.get(a2.thread_id, []):
            if start <= a2.index <= end:
                locks2.add(lid)
        return bool(locks1 & locks2)


class LockPlacement:
    """Find minimal set of locks ensuring mutual exclusion."""

    def __init__(self):
        self.race_detector = DataRaceDetector()

    def place_locks(self, threads: Dict[int, List[Instruction]],
                    shared_vars: Set[str],
                    safety: SafetySpec) -> SynchronizedProgram:
        races = self.race_detector.detect(threads, shared_vars)

        var_groups = self._group_races_by_variable(races)
        lock_assignments = self._assign_locks(var_groups, safety)

        new_threads = {}
        added_locks = []

        for tid, insts in threads.items():
            new_insts = []
            idx_offset = 0

            lock_regions = self._compute_lock_regions(tid, insts, lock_assignments)

            for start, end, lock_id in sorted(lock_regions):
                while len(new_insts) < start + idx_offset:
                    if len(new_insts) - idx_offset < len(insts):
                        new_insts.append(insts[len(new_insts) - idx_offset].copy())
                    else:
                        break

                lock_inst = Instruction(tid, start + idx_offset, InstructionType.LOCK,
                                        lock_id=lock_id)
                new_insts.append(lock_inst)
                added_locks.append((tid, start + idx_offset, lock_id))
                idx_offset += 1

                for i in range(start, min(end + 1, len(insts))):
                    c = insts[i].copy()
                    c.index = len(new_insts)
                    new_insts.append(c)

                unlock_inst = Instruction(tid, len(new_insts), InstructionType.UNLOCK,
                                          lock_id=lock_id)
                new_insts.append(unlock_inst)
                added_locks.append((tid, len(new_insts) - 1, lock_id))
                idx_offset += 1

            remaining_start = max((end + 1 for _, end, _ in lock_regions), default=0)
            for i in range(remaining_start, len(insts)):
                c = insts[i].copy()
                c.index = len(new_insts)
                new_insts.append(c)

            new_threads[tid] = new_insts

        return SynchronizedProgram(
            threads=new_threads, added_locks=added_locks,
            added_fences=[], added_barriers=[], added_atomics=[],
            description="Lock placement for mutual exclusion",
        )

    def _group_races_by_variable(self, races: List[Tuple[Instruction, Instruction]]
                                 ) -> Dict[str, List[Tuple[Instruction, Instruction]]]:
        groups: Dict[str, List] = {}
        for a1, a2 in races:
            var = a1.variable or a2.variable
            groups.setdefault(var, []).append((a1, a2))
        return groups

    def _assign_locks(self, var_groups: Dict[str, List],
                      safety: SafetySpec) -> Dict[str, str]:
        assignments = {}
        lock_counter = 0

        mutex_groups = self._find_mutex_groups(var_groups, safety)

        for group in mutex_groups:
            lock_id = f"L{lock_counter}"
            lock_counter += 1
            for var in group:
                assignments[var] = lock_id

        for var in var_groups:
            if var not in assignments:
                assignments[var] = f"L{lock_counter}"
                lock_counter += 1

        return assignments

    def _find_mutex_groups(self, var_groups: Dict[str, List],
                           safety: SafetySpec) -> List[Set[str]]:
        groups = []
        vars_list = list(var_groups.keys())
        assigned = set()

        for i, v1 in enumerate(vars_list):
            if v1 in assigned:
                continue
            group = {v1}
            for v2 in vars_list[i + 1:]:
                if v2 in assigned:
                    continue
                if self._should_group(v1, v2, var_groups, safety):
                    group.add(v2)
            groups.append(group)
            assigned.update(group)
        return groups

    def _should_group(self, v1: str, v2: str,
                      var_groups: Dict, safety: SafetySpec) -> bool:
        for atomic_group in safety.atomic_groups:
            if v1 in atomic_group and v2 in atomic_group:
                return True

        threads1 = set()
        threads2 = set()
        for a1, a2 in var_groups.get(v1, []):
            threads1.add(a1.thread_id)
            threads1.add(a2.thread_id)
        for a1, a2 in var_groups.get(v2, []):
            threads2.add(a1.thread_id)
            threads2.add(a2.thread_id)
        return bool(threads1 & threads2)

    def _compute_lock_regions(self, tid: int, insts: List[Instruction],
                              lock_assignments: Dict[str, str]
                              ) -> List[Tuple[int, int, str]]:
        regions = []
        lock_groups: Dict[str, List[int]] = {}

        for i, inst in enumerate(insts):
            if inst.variable in lock_assignments:
                lid = lock_assignments[inst.variable]
                lock_groups.setdefault(lid, []).append(i)

        for lid, indices in lock_groups.items():
            if indices:
                regions.append((min(indices), max(indices), lid))

        return self._merge_regions(regions)

    def _merge_regions(self, regions: List[Tuple[int, int, str]]
                       ) -> List[Tuple[int, int, str]]:
        if not regions:
            return []
        regions.sort()
        merged = [regions[0]]
        for start, end, lid in regions[1:]:
            prev_start, prev_end, prev_lid = merged[-1]
            if lid == prev_lid and start <= prev_end + 1:
                merged[-1] = (prev_start, max(prev_end, end), lid)
            else:
                merged.append((start, end, lid))
        return merged


class BarrierPlacement:
    """Place barriers to ensure threads synchronize at specific points."""

    def place_barriers(self, threads: Dict[int, List[Instruction]],
                       barrier_points: List[Set[int]],
                       safety: SafetySpec) -> SynchronizedProgram:
        new_threads = {}
        added_barriers = []

        for bid, participating in enumerate(barrier_points):
            barrier_id = f"B{bid}"
            for tid in participating:
                if tid not in threads:
                    continue
                insts = threads.get(tid, [])
                insert_at = len(insts) // 2  # Default: middle of thread

                new_insts = list(insts[:insert_at])
                barrier_inst = Instruction(tid, insert_at, InstructionType.BARRIER,
                                           label=barrier_id)
                new_insts.append(barrier_inst)
                added_barriers.append((tid, insert_at, barrier_id))
                new_insts.extend(insts[insert_at:])

                # Re-index
                for i, inst in enumerate(new_insts):
                    inst.index = i

                new_threads[tid] = new_insts

        for tid, insts in threads.items():
            if tid not in new_threads:
                new_threads[tid] = [i.copy() for i in insts]

        return SynchronizedProgram(
            threads=new_threads, added_locks=[], added_fences=[],
            added_barriers=added_barriers, added_atomics=[],
            description="Barrier placement for synchronization",
        )


class FenceSynthesizer:
    """Insert minimum fences for desired memory consistency."""

    def __init__(self):
        self.fence_costs = {
            FenceType.FULL: 10.0,
            FenceType.STORE_LOAD: 8.0,
            FenceType.STORE_STORE: 3.0,
            FenceType.LOAD_LOAD: 3.0,
            FenceType.LOAD_STORE: 4.0,
            FenceType.ACQUIRE: 5.0,
            FenceType.RELEASE: 5.0,
        }

    def synthesize(self, threads: Dict[int, List[Instruction]],
                   no_reorder: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                   safety: SafetySpec) -> SynchronizedProgram:
        needed_fences = self._compute_needed_fences(threads, no_reorder)
        optimized = self._minimize_fences(needed_fences)

        new_threads = {}
        added_fences = []

        for tid, insts in threads.items():
            new_insts = []
            fence_positions = sorted(
                [(pos, ft) for t, pos, ft in optimized if t == tid])

            fi = 0
            for i, inst in enumerate(insts):
                while fi < len(fence_positions) and fence_positions[fi][0] <= i:
                    ft = fence_positions[fi][1]
                    fence_inst = Instruction(tid, len(new_insts), InstructionType.FENCE,
                                             fence_type=ft.value)
                    new_insts.append(fence_inst)
                    added_fences.append((tid, len(new_insts) - 1, ft))
                    fi += 1
                c = inst.copy()
                c.index = len(new_insts)
                new_insts.append(c)

            while fi < len(fence_positions):
                ft = fence_positions[fi][1]
                fence_inst = Instruction(tid, len(new_insts), InstructionType.FENCE,
                                         fence_type=ft.value)
                new_insts.append(fence_inst)
                added_fences.append((tid, len(new_insts) - 1, ft))
                fi += 1

            new_threads[tid] = new_insts

        cost = sum(self.fence_costs[ft] for _, _, ft in added_fences)

        return SynchronizedProgram(
            threads=new_threads, added_locks=[], added_fences=added_fences,
            added_barriers=[], added_atomics=[],
            sync_cost=cost, description="Fence synthesis for memory ordering",
        )

    def _compute_needed_fences(self, threads: Dict[int, List[Instruction]],
                               no_reorder: List[Tuple[Tuple[int, int], Tuple[int, int]]]
                               ) -> List[Tuple[int, int, FenceType]]:
        fences = []
        for (t1, i1), (t2, i2) in no_reorder:
            if t1 == t2:
                insts = threads.get(t1, [])
                if i1 < len(insts) and i2 < len(insts):
                    before_type = insts[i1].inst_type
                    after_type = insts[i2].inst_type
                    ft = self._determine_fence_type(before_type, after_type)
                    fences.append((t1, i1 + 1, ft))
        return fences

    def _determine_fence_type(self, before: InstructionType,
                              after: InstructionType) -> FenceType:
        if before == InstructionType.STORE and after == InstructionType.LOAD:
            return FenceType.STORE_LOAD
        elif before == InstructionType.STORE and after == InstructionType.STORE:
            return FenceType.STORE_STORE
        elif before == InstructionType.LOAD and after == InstructionType.LOAD:
            return FenceType.LOAD_LOAD
        elif before == InstructionType.LOAD and after == InstructionType.STORE:
            return FenceType.LOAD_STORE
        return FenceType.FULL

    def _minimize_fences(self, fences: List[Tuple[int, int, FenceType]]
                         ) -> List[Tuple[int, int, FenceType]]:
        by_position: Dict[Tuple[int, int], List[FenceType]] = {}
        for tid, pos, ft in fences:
            by_position.setdefault((tid, pos), []).append(ft)

        minimized = []
        for (tid, pos), types in by_position.items():
            merged = self._merge_fence_types(types)
            minimized.append((tid, pos, merged))
        return minimized

    def _merge_fence_types(self, types: List[FenceType]) -> FenceType:
        if FenceType.FULL in types:
            return FenceType.FULL
        orderings = set()
        for ft in types:
            if ft == FenceType.STORE_LOAD:
                orderings.add("WR")
            elif ft == FenceType.STORE_STORE:
                orderings.add("WW")
            elif ft == FenceType.LOAD_LOAD:
                orderings.add("RR")
            elif ft == FenceType.LOAD_STORE:
                orderings.add("RW")
            elif ft == FenceType.ACQUIRE:
                orderings.update({"RR", "RW"})
            elif ft == FenceType.RELEASE:
                orderings.update({"WW", "WR"})

        if len(orderings) >= 3:
            return FenceType.FULL
        if orderings == {"WR"}:
            return FenceType.STORE_LOAD
        if orderings == {"WW"}:
            return FenceType.STORE_STORE
        if orderings == {"RR"}:
            return FenceType.LOAD_LOAD
        if orderings == {"RW"}:
            return FenceType.LOAD_STORE
        if orderings == {"RR", "RW"}:
            return FenceType.ACQUIRE
        if orderings == {"WW", "WR"}:
            return FenceType.RELEASE
        return FenceType.FULL


class AtomicSelector:
    """Select correct atomic operations for compound operations."""

    def select(self, threads: Dict[int, List[Instruction]],
               shared_vars: Set[str]) -> SynchronizedProgram:
        new_threads = {}
        added_atomics = []

        for tid, insts in threads.items():
            new_insts = []
            i = 0
            while i < len(insts):
                pattern = self._detect_pattern(insts, i, shared_vars)
                if pattern:
                    atomic_inst, skip = pattern
                    atomic_inst.index = len(new_insts)
                    new_insts.append(atomic_inst)
                    added_atomics.append((tid, len(new_insts) - 1, atomic_inst.inst_type))
                    i += skip
                else:
                    c = insts[i].copy()
                    c.index = len(new_insts)
                    new_insts.append(c)
                    i += 1
            new_threads[tid] = new_insts

        return SynchronizedProgram(
            threads=new_threads, added_locks=[], added_fences=[],
            added_barriers=[], added_atomics=added_atomics,
            description="Atomic instruction selection",
        )

    def _detect_pattern(self, insts: List[Instruction], start: int,
                        shared_vars: Set[str]) -> Optional[Tuple[Instruction, int]]:
        if start + 2 >= len(insts):
            return None

        # Detect read-modify-write pattern: load x -> compute -> store x
        inst0 = insts[start]
        if (inst0.inst_type == InstructionType.LOAD and
                inst0.variable in shared_vars):
            if start + 2 < len(insts):
                inst1 = insts[start + 1]
                inst2 = insts[start + 2]
                if (inst1.inst_type == InstructionType.COMPUTE and
                        inst2.inst_type == InstructionType.STORE and
                        inst2.variable == inst0.variable):
                    return (Instruction(inst0.thread_id, start, InstructionType.FETCH_ADD,
                                        variable=inst0.variable, value=1), 3)

        # Detect compare-and-swap pattern: load x -> branch -> store x
        if (inst0.inst_type == InstructionType.LOAD and
                inst0.variable in shared_vars and start + 2 < len(insts)):
            inst1 = insts[start + 1]
            inst2 = insts[start + 2]
            if (inst1.inst_type == InstructionType.BRANCH and
                    inst2.inst_type == InstructionType.STORE and
                    inst2.variable == inst0.variable):
                return (Instruction(inst0.thread_id, start, InstructionType.CAS,
                                    variable=inst0.variable,
                                    value=(0, inst2.value)), 3)

        return None


class LockGranularityOptimizer:
    """Optimize lock granularity: fine-grained vs coarse-grained tradeoff."""

    def optimize(self, threads: Dict[int, List[Instruction]],
                 current_locks: List[Tuple[int, int, int, str]],
                 contention_estimate: float = 0.5) -> Tuple[LockGranularity, Dict]:
        n_locks = len(set(lid for _, _, _, lid in current_locks))
        total_protected = sum(end - start for _, start, end, _ in current_locks)
        total_insts = sum(len(insts) for insts in threads.values())

        lock_fraction = total_protected / max(total_insts, 1)

        if contention_estimate < 0.3:
            recommendation = LockGranularity.COARSE
            reason = "Low contention: coarse-grained locks reduce overhead"
        elif contention_estimate > 0.7:
            recommendation = LockGranularity.FINE
            reason = "High contention: fine-grained locks increase parallelism"
        else:
            recommendation = LockGranularity.MEDIUM
            reason = "Moderate contention: balanced lock granularity"

        metrics = {
            "current_locks": n_locks,
            "lock_fraction": lock_fraction,
            "contention_estimate": contention_estimate,
            "recommendation": recommendation.value,
            "reason": reason,
            "estimated_speedup": self._estimate_speedup(
                lock_fraction, contention_estimate, len(threads), recommendation),
        }
        return recommendation, metrics

    def _estimate_speedup(self, lock_fraction: float, contention: float,
                          n_threads: int, granularity: LockGranularity) -> float:
        if granularity == LockGranularity.COARSE:
            serial = lock_fraction
        elif granularity == LockGranularity.FINE:
            serial = lock_fraction * contention
        else:
            serial = lock_fraction * (contention * 0.5 + 0.25)

        serial = max(serial, 0.01)
        return 1.0 / (serial + (1 - serial) / n_threads)


class ReaderWriterLockSynthesizer:
    """Identify read-only vs write operations and synthesize RW locks."""

    def synthesize(self, threads: Dict[int, List[Instruction]],
                   shared_vars: Set[str]) -> SynchronizedProgram:
        var_access = self._analyze_access_patterns(threads, shared_vars)
        new_threads = {}
        added_locks = []

        for tid, insts in threads.items():
            new_insts = []
            active_locks: Dict[str, str] = {}

            for inst in insts:
                if inst.variable in var_access and inst.variable in shared_vars:
                    pattern = var_access[inst.variable]
                    lock_id = f"RW_{inst.variable}"

                    if inst.variable not in active_locks:
                        if inst.inst_type == InstructionType.LOAD and pattern["write_threads"] <= 1:
                            rlock = Instruction(tid, len(new_insts), InstructionType.RWLOCK_RLOCK,
                                                lock_id=lock_id)
                            new_insts.append(rlock)
                            added_locks.append((tid, len(new_insts) - 1, lock_id))
                            active_locks[inst.variable] = "read"
                        else:
                            wlock = Instruction(tid, len(new_insts), InstructionType.RWLOCK_WLOCK,
                                                lock_id=lock_id)
                            new_insts.append(wlock)
                            added_locks.append((tid, len(new_insts) - 1, lock_id))
                            active_locks[inst.variable] = "write"

                c = inst.copy()
                c.index = len(new_insts)
                new_insts.append(c)

                if inst.variable in active_locks:
                    lock_id = f"RW_{inst.variable}"
                    if active_locks[inst.variable] == "read":
                        runlock = Instruction(tid, len(new_insts), InstructionType.RWLOCK_RUNLOCK,
                                              lock_id=lock_id)
                        new_insts.append(runlock)
                    else:
                        wunlock = Instruction(tid, len(new_insts), InstructionType.RWLOCK_WUNLOCK,
                                              lock_id=lock_id)
                        new_insts.append(wunlock)
                    del active_locks[inst.variable]

            new_threads[tid] = new_insts

        return SynchronizedProgram(
            threads=new_threads, added_locks=added_locks,
            added_fences=[], added_barriers=[], added_atomics=[],
            description="Reader-writer lock synthesis",
        )

    def _analyze_access_patterns(self, threads: Dict[int, List[Instruction]],
                                 shared_vars: Set[str]) -> Dict[str, Dict]:
        patterns = {}
        for var in shared_vars:
            read_threads = set()
            write_threads = set()
            for tid, insts in threads.items():
                for inst in insts:
                    if inst.variable == var:
                        if inst.inst_type == InstructionType.LOAD:
                            read_threads.add(tid)
                        elif inst.inst_type in (InstructionType.STORE, InstructionType.CAS):
                            write_threads.add(tid)
            patterns[var] = {
                "read_threads": len(read_threads),
                "write_threads": len(write_threads),
                "total_threads": len(read_threads | write_threads),
                "read_heavy": len(read_threads) > len(write_threads) * 2,
            }
        return patterns


class OptimisticSyncSynthesizer:
    """Synthesize optimistic synchronization: try without lock, validate, retry."""

    def synthesize(self, threads: Dict[int, List[Instruction]],
                   shared_vars: Set[str]) -> SynchronizedProgram:
        new_threads = {}
        added_atomics = []

        for tid, insts in threads.items():
            new_insts = []
            i = 0
            while i < len(insts):
                if (insts[i].inst_type in (InstructionType.LOAD, InstructionType.STORE) and
                        insts[i].variable in shared_vars):
                    # Wrap in optimistic read-validate-retry
                    opt_insts, n_consumed = self._wrap_optimistic(tid, insts, i, shared_vars)
                    for oi in opt_insts:
                        oi.index = len(new_insts)
                        new_insts.append(oi)
                    if any(oi.inst_type in (InstructionType.CAS, InstructionType.VALIDATE)
                           for oi in opt_insts):
                        added_atomics.append((tid, len(new_insts) - 1, InstructionType.CAS))
                    i += n_consumed
                else:
                    c = insts[i].copy()
                    c.index = len(new_insts)
                    new_insts.append(c)
                    i += 1

            new_threads[tid] = new_insts

        return SynchronizedProgram(
            threads=new_threads, added_locks=[], added_fences=[],
            added_barriers=[], added_atomics=added_atomics,
            description="Optimistic synchronization synthesis",
        )

    def _wrap_optimistic(self, tid: int, insts: List[Instruction],
                         start: int, shared_vars: Set[str]
                         ) -> Tuple[List[Instruction], int]:
        result = []
        end = start

        while end < len(insts) and insts[end].variable in shared_vars:
            end += 1
        end = min(end, start + 5)  # cap optimistic region size

        # Read phase
        for i in range(start, end):
            c = insts[i].copy()
            c.thread_id = tid
            result.append(c)

        # Validate (CAS-like check)
        validate_inst = Instruction(tid, 0, InstructionType.VALIDATE,
                                    label="optimistic_validate")
        result.append(validate_inst)

        return result, end - start


class SyncSynthesizer:
    """Main synthesizer: orchestrates all synchronization synthesis strategies."""

    def __init__(self):
        self.lock_placer = LockPlacement()
        self.barrier_placer = BarrierPlacement()
        self.fence_synth = FenceSynthesizer()
        self.atomic_selector = AtomicSelector()
        self.granularity_opt = LockGranularityOptimizer()
        self.rw_synth = ReaderWriterLockSynthesizer()
        self.optimistic_synth = OptimisticSyncSynthesizer()
        self.race_detector = DataRaceDetector()

    def synthesize(self, threads: Dict[int, List[Instruction]],
                   shared_vars: Set[str],
                   safety: SafetySpec) -> SynchronizedProgram:
        # Detect races
        races = self.race_detector.detect(threads, shared_vars)

        if not races:
            return SynchronizedProgram(
                threads={tid: [i.copy() for i in insts] for tid, insts in threads.items()},
                added_locks=[], added_fences=[], added_barriers=[], added_atomics=[],
                description="No races detected, no synchronization needed",
            )

        # Apply lock placement
        result = self.lock_placer.place_locks(threads, shared_vars, safety)

        # Verify races are eliminated
        new_races = self.race_detector.detect(result.threads, shared_vars)
        result.description += f" (races before: {len(races)}, after: {len(new_races)})"

        return result

    def synthesize_fences(self, threads: Dict[int, List[Instruction]],
                          ordering_constraints: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                          safety: SafetySpec) -> SynchronizedProgram:
        return self.fence_synth.synthesize(threads, ordering_constraints, safety)

    def synthesize_rw_locks(self, threads: Dict[int, List[Instruction]],
                            shared_vars: Set[str]) -> SynchronizedProgram:
        return self.rw_synth.synthesize(threads, shared_vars)

    def synthesize_optimistic(self, threads: Dict[int, List[Instruction]],
                              shared_vars: Set[str]) -> SynchronizedProgram:
        return self.optimistic_synth.synthesize(threads, shared_vars)

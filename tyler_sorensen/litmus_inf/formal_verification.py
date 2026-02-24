"""
Formal verification of concurrent code.

Provides model checking, linearizability verification, progress property
checking, memory safety verification, bounded model checking, and
counterexample-guided abstraction refinement (CEGAR).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Set, FrozenSet
import re
import itertools


class CheckStatus(Enum):
    VERIFIED = "verified"
    VIOLATED = "violated"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"


class ProgressType(Enum):
    DEADLOCK_FREE = "deadlock_free"
    STARVATION_FREE = "starvation_free"
    LOCK_FREE = "lock_free"
    WAIT_FREE = "wait_free"
    OBSTRUCTION_FREE = "obstruction_free"


@dataclass
class Counterexample:
    trace: List[str] = field(default_factory=list)
    thread_interleavings: List[Tuple[str, str]] = field(default_factory=list)
    final_state: Dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = ["Counterexample trace:"]
        for step in self.trace:
            lines.append(f"  {step}")
        if self.final_state:
            lines.append(f"  Final state: {self.final_state}")
        return "\n".join(lines)


@dataclass
class ModelCheckResult:
    status: CheckStatus
    property_name: str
    states_explored: int = 0
    transitions_explored: int = 0
    counterexample: Optional[Counterexample] = None
    details: str = ""

    def __str__(self) -> str:
        s = f"[{self.status.value}] {self.property_name}"
        s += f" (states={self.states_explored}, transitions={self.transitions_explored})"
        if self.counterexample:
            s += f"\n{self.counterexample}"
        return s


@dataclass
class LinearResult:
    status: CheckStatus
    linearization_points: List[str] = field(default_factory=list)
    counterexample: Optional[Counterexample] = None
    details: str = ""

    def __str__(self) -> str:
        s = f"Linearizability: [{self.status.value}]"
        if self.linearization_points:
            s += f" — LP at: {', '.join(self.linearization_points)}"
        if self.counterexample:
            s += f"\n{self.counterexample}"
        return s


@dataclass
class ProgressResult:
    status: CheckStatus
    progress_type: ProgressType
    blocking_paths: List[str] = field(default_factory=list)
    details: str = ""

    def __str__(self) -> str:
        s = f"Progress ({self.progress_type.value}): [{self.status.value}]"
        if self.blocking_paths:
            s += f"\n  Blocking paths: {len(self.blocking_paths)}"
        return s


@dataclass
class MemorySafetyResult:
    status: CheckStatus
    use_after_free: List[int] = field(default_factory=list)
    double_free: List[int] = field(default_factory=list)
    buffer_overflow: List[int] = field(default_factory=list)
    null_deref: List[int] = field(default_factory=list)
    data_races: List[int] = field(default_factory=list)
    details: str = ""

    def __str__(self) -> str:
        issues = []
        if self.use_after_free:
            issues.append(f"use-after-free at lines {self.use_after_free}")
        if self.double_free:
            issues.append(f"double-free at lines {self.double_free}")
        if self.buffer_overflow:
            issues.append(f"buffer-overflow at lines {self.buffer_overflow}")
        if self.null_deref:
            issues.append(f"null-deref at lines {self.null_deref}")
        if self.data_races:
            issues.append(f"data-races at lines {self.data_races}")
        s = f"Memory safety: [{self.status.value}]"
        if issues:
            s += "\n  " + "\n  ".join(issues)
        return s


@dataclass
class BMCResult:
    status: CheckStatus
    bound: int
    states_explored: int = 0
    counterexample: Optional[Counterexample] = None
    details: str = ""

    def __str__(self) -> str:
        s = f"BMC (bound={self.bound}): [{self.status.value}]"
        s += f" — {self.states_explored} states explored"
        if self.counterexample:
            s += f"\n{self.counterexample}"
        return s


@dataclass
class CEGARResult:
    status: CheckStatus
    iterations: int = 0
    final_abstraction_size: int = 0
    counterexample: Optional[Counterexample] = None
    details: str = ""

    def __str__(self) -> str:
        s = f"CEGAR: [{self.status.value}]"
        s += f" — {self.iterations} iteration(s), abstraction size={self.final_abstraction_size}"
        if self.counterexample:
            s += f"\n{self.counterexample}"
        return s


# ---------------------------------------------------------------------------
# Internal: lightweight state-space exploration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _State:
    """Immutable state for model checking."""
    pc: Tuple[int, ...]       # program counter per thread
    variables: FrozenSet[Tuple[str, int]]  # variable -> value
    locks_held: FrozenSet[Tuple[str, int]]  # lock -> thread_id

    def var_dict(self) -> Dict[str, int]:
        return dict(self.variables)


def _extract_shared_vars(source: str) -> List[str]:
    """Extract shared variable names from source."""
    patterns = [
        r'(volatile|_Atomic|atomic|shared)\s+\w+\s+(\w+)',
        r'std::atomic<\w+>\s+(\w+)',
        r'AtomicInteger\s+(\w+)',
        r'(\w+)\s*=\s*.*atomic',
    ]
    variables: Set[str] = set()
    for pat in patterns:
        for m in re.finditer(pat, source):
            variables.add(m.group(m.lastindex))
    return list(variables)


def _extract_threads(source: str) -> List[List[str]]:
    """Extract simplified thread instruction sequences from source."""
    threads: List[List[str]] = []

    # Look for thread functions / goroutines / closures
    thread_bodies = re.findall(
        r'(?:void\s*\*?\s*\w+|fn\s+\w+|func\s*\(|lambda|Thread\s*\(target=)\s*'
        r'[^{]*\{([^}]*)\}',
        source, re.DOTALL,
    )

    for body in thread_bodies:
        instructions: List[str] = []
        lines = body.strip().split('\n')
        for line in lines:
            stripped = line.strip().rstrip(';')
            if stripped and not stripped.startswith('//') and not stripped.startswith('#'):
                instructions.append(stripped)
        if instructions:
            threads.append(instructions)

    # If no thread bodies found, treat whole source as single thread
    if not threads:
        instructions = []
        for line in source.split('\n'):
            stripped = line.strip().rstrip(';')
            if stripped and not stripped.startswith('//') and not stripped.startswith('#'):
                instructions.append(stripped)
        if instructions:
            threads.append(instructions)

    return threads


def _is_lock_acquire(instr: str) -> Optional[str]:
    """Return lock name if instruction acquires a lock, else None."""
    patterns = [
        r'pthread_mutex_lock\s*\(\s*&?\s*(\w+)',
        r'(\w+)\.lock\s*\(',
        r'(\w+)\.Lock\s*\(',
        r'synchronized\s*\(\s*(\w+)',
    ]
    for pat in patterns:
        m = re.search(pat, instr)
        if m:
            return m.group(1)
    return None


def _is_lock_release(instr: str) -> Optional[str]:
    """Return lock name if instruction releases a lock, else None."""
    patterns = [
        r'pthread_mutex_unlock\s*\(\s*&?\s*(\w+)',
        r'(\w+)\.unlock\s*\(',
        r'(\w+)\.Unlock\s*\(',
    ]
    for pat in patterns:
        m = re.search(pat, instr)
        if m:
            return m.group(1)
    return None


def model_check(source: str, property: str) -> ModelCheckResult:
    """Model-check concurrent source code for a given property.

    Performs explicit-state exploration of thread interleavings.
    Properties: "mutual_exclusion", "no_deadlock", "no_data_race",
                "assertion" (checks assert() statements).
    """
    threads = _extract_threads(source)
    shared_vars = _extract_shared_vars(source)
    n_threads = len(threads)

    if n_threads == 0:
        return ModelCheckResult(
            status=CheckStatus.VERIFIED, property_name=property,
            details="No threads found to check.",
        )

    # Initial state
    init_vars = frozenset((v, 0) for v in shared_vars)
    init_pc = tuple(0 for _ in range(n_threads))
    init_state = _State(pc=init_pc, variables=init_vars, locks_held=frozenset())

    visited: Set[_State] = set()
    frontier: List[_State] = [init_state]
    states_explored = 0
    transitions = 0
    violation: Optional[Counterexample] = None
    max_states = 100000  # bound exploration

    while frontier and states_explored < max_states:
        state = frontier.pop()
        if state in visited:
            continue
        visited.add(state)
        states_explored += 1

        # Check if all threads finished
        all_done = all(state.pc[t] >= len(threads[t]) for t in range(n_threads))
        if all_done:
            continue

        # Check for deadlock: no thread can make progress
        can_progress = False

        for tid in range(n_threads):
            pc = state.pc[tid]
            if pc >= len(threads[tid]):
                continue

            instr = threads[tid][pc]

            # Check if this thread can proceed (not blocked on lock)
            lock = _is_lock_acquire(instr)
            if lock:
                lock_holder = dict(state.locks_held)
                if lock in lock_holder and lock_holder[lock] != tid:
                    continue  # blocked

            can_progress = True

            # Execute instruction
            new_pc = list(state.pc)
            new_pc[tid] = pc + 1
            new_vars = dict(state.variables)
            new_locks = dict(state.locks_held)

            if lock:
                new_locks[lock] = tid
            release = _is_lock_release(instr)
            if release and release in new_locks:
                del new_locks[release]

            # Handle variable writes (simplified)
            write_match = re.search(r'(\w+)\s*=\s*(\d+)', instr)
            if write_match and write_match.group(1) in new_vars:
                new_vars[write_match.group(1)] = int(write_match.group(2))

            new_state = _State(
                pc=tuple(new_pc),
                variables=frozenset(new_vars.items()),
                locks_held=frozenset(new_locks.items()),
            )

            # Check mutual exclusion property
            if property == "mutual_exclusion":
                lock_counts: Dict[str, int] = {}
                for lk, holder in new_locks.items():
                    lock_counts[lk] = lock_counts.get(lk, 0) + 1
                for lk, count in lock_counts.items():
                    if count > 1:
                        violation = Counterexample(
                            trace=[f"Thread {tid}: {instr}"],
                            final_state={str(k): str(v) for k, v in new_vars.items()},
                        )
                        return ModelCheckResult(
                            status=CheckStatus.VIOLATED,
                            property_name=property,
                            states_explored=states_explored,
                            transitions_explored=transitions,
                            counterexample=violation,
                        )

            # Check assertion property
            if property == "assertion":
                assert_match = re.search(r'assert\s*\(\s*(.+)\s*\)', instr)
                if assert_match:
                    cond = assert_match.group(1)
                    var_match = re.search(r'(\w+)\s*(==|!=|<|>|<=|>=)\s*(\d+)', cond)
                    if var_match:
                        var_name = var_match.group(1)
                        op = var_match.group(2)
                        val = int(var_match.group(3))
                        cur_val = new_vars.get(var_name, 0)
                        holds = {
                            "==": cur_val == val, "!=": cur_val != val,
                            "<": cur_val < val, ">": cur_val > val,
                            "<=": cur_val <= val, ">=": cur_val >= val,
                        }.get(op, True)
                        if not holds:
                            violation = Counterexample(
                                trace=[f"Thread {tid}: {instr} — FAILED ({var_name}={cur_val})"],
                                final_state={str(k): str(v) for k, v in new_vars.items()},
                            )
                            return ModelCheckResult(
                                status=CheckStatus.VIOLATED,
                                property_name=property,
                                states_explored=states_explored,
                                transitions_explored=transitions,
                                counterexample=violation,
                            )

            transitions += 1
            frontier.append(new_state)

        # Check deadlock
        if not can_progress and not all_done and property == "no_deadlock":
            held = {lk: tid for lk, tid in state.locks_held}
            violation = Counterexample(
                trace=[f"All threads blocked — locks held: {held}"],
                final_state={str(k): str(v) for k, v in state.variables},
            )
            return ModelCheckResult(
                status=CheckStatus.VIOLATED,
                property_name=property,
                states_explored=states_explored,
                transitions_explored=transitions,
                counterexample=violation,
            )

    status = CheckStatus.VERIFIED if states_explored < max_states else CheckStatus.TIMEOUT

    return ModelCheckResult(
        status=status,
        property_name=property,
        states_explored=states_explored,
        transitions_explored=transitions,
        details=f"Explored {states_explored} states and {transitions} transitions.",
    )


def verify_linearizability(source: str, spec: str) -> LinearResult:
    """Verify that a concurrent data structure is linearizable w.r.t. a spec.

    The spec should describe the sequential specification (e.g., "queue: enqueue/dequeue FIFO").
    Uses a simplified history-based approach.
    """
    linearization_points: List[str] = []
    issues: List[str] = []

    # Identify operations and their linearization points
    op_patterns = [
        (r'(enqueue|push|put|add|insert)\s*\(', "write"),
        (r'(dequeue|pop|get|remove|take)\s*\(', "read"),
    ]

    for pattern, kind in op_patterns:
        for m in re.finditer(pattern, source, re.IGNORECASE):
            op_name = m.group(1)
            # Find the "effect point" — typically the CAS or lock-protected write
            context_start = max(0, m.start() - 200)
            context_end = min(len(source), m.end() + 500)
            context = source[context_start:context_end]

            lp = None
            if re.search(r'compare_exchange|compareAndSet|CAS|CompareAndSwap', context):
                lp = f"CAS in {op_name}"
            elif re.search(r'\.lock\(\)|pthread_mutex_lock|synchronized', context):
                lp = f"lock acquisition in {op_name}"
            elif re.search(r'\.store\(|atomic_store', context):
                lp = f"atomic store in {op_name}"

            if lp:
                linearization_points.append(lp)
            else:
                issues.append(f"No clear linearization point for {op_name}")

    # Check for common linearizability violations
    if "queue" in spec.lower() or "fifo" in spec.lower():
        # FIFO requires ordered matching of enqueue/dequeue
        if re.search(r'(LIFO|stack|push.*pop)', source, re.IGNORECASE):
            issues.append("Source appears to implement LIFO (stack) but spec requires FIFO (queue)")

    if "stack" in spec.lower() or "lifo" in spec.lower():
        if re.search(r'(FIFO|queue|enqueue.*dequeue)', source, re.IGNORECASE):
            issues.append("Source appears to implement FIFO (queue) but spec requires LIFO (stack)")

    # Check for ABA problems in lock-free implementations
    if re.search(r'compare_exchange|CAS|compareAndSet', source):
        if not re.search(r'(tagged_ptr|stamp|version|generation|epoch|hazard)', source, re.IGNORECASE):
            issues.append("CAS without ABA protection — linearizability may be violated")

    if issues:
        return LinearResult(
            status=CheckStatus.VIOLATED,
            linearization_points=linearization_points,
            counterexample=Counterexample(trace=issues),
            details="; ".join(issues),
        )

    return LinearResult(
        status=CheckStatus.VERIFIED if linearization_points else CheckStatus.UNKNOWN,
        linearization_points=linearization_points,
        details=f"Found {len(linearization_points)} linearization point(s).",
    )


def verify_progress(source: str, progress_type: str = "lock_free") -> ProgressResult:
    """Verify a progress property: deadlock_free, lock_free, wait_free, obstruction_free."""
    try:
        ptype = ProgressType(progress_type)
    except ValueError:
        ptype = ProgressType.LOCK_FREE

    blocking_paths: List[str] = []

    # Check for blocking operations
    blocking_patterns = [
        (r'pthread_mutex_lock|\.lock\(\)|\.Lock\(\)|synchronized', "mutex lock"),
        (r'pthread_cond_wait|\.wait\(\)|\.Wait\(\)', "condition wait"),
        (r'sleep|usleep|nanosleep|time\.Sleep', "sleep"),
        (r'sem_wait|\.acquire\(\)', "semaphore wait"),
    ]

    for pattern, desc in blocking_patterns:
        for m in re.finditer(pattern, source):
            line = source[:m.start()].count('\n') + 1
            blocking_paths.append(f"line {line}: {desc}")

    # Check for unbounded loops (non-wait-free)
    loop_patterns = [
        (r'while\s*\(\s*!?.*compare_exchange', "CAS retry loop"),
        (r'while\s*\(\s*true\s*\)', "unbounded loop"),
        (r'for\s*\(\s*;;\s*\)', "infinite loop"),
        (r'loop\s*\{', "Rust loop"),
    ]
    has_unbounded_loop = False
    for pattern, desc in loop_patterns:
        for m in re.finditer(pattern, source):
            has_unbounded_loop = True
            line = source[:m.start()].count('\n') + 1
            if ptype == ProgressType.WAIT_FREE:
                blocking_paths.append(f"line {line}: {desc} (violates wait-freedom)")

    # Determine status based on progress type
    has_locks = bool(re.search(
        r'pthread_mutex|\.lock\(\)|\.Lock\(\)|synchronized|Mutex|RwLock', source))

    if ptype == ProgressType.DEADLOCK_FREE:
        # Check for lock ordering or single-lock patterns
        mc_result = model_check(source, "no_deadlock")
        if mc_result.status == CheckStatus.VIOLATED:
            return ProgressResult(
                status=CheckStatus.VIOLATED,
                progress_type=ptype,
                blocking_paths=blocking_paths,
                details="Deadlock detected via model checking.",
            )

    elif ptype == ProgressType.LOCK_FREE:
        if has_locks:
            return ProgressResult(
                status=CheckStatus.VIOLATED,
                progress_type=ptype,
                blocking_paths=blocking_paths,
                details="Lock-based synchronization found — not lock-free.",
            )

    elif ptype == ProgressType.WAIT_FREE:
        if has_locks or has_unbounded_loop:
            return ProgressResult(
                status=CheckStatus.VIOLATED,
                progress_type=ptype,
                blocking_paths=blocking_paths,
                details="Blocking operations or unbounded loops found — not wait-free.",
            )

    elif ptype == ProgressType.OBSTRUCTION_FREE:
        if has_locks:
            return ProgressResult(
                status=CheckStatus.VIOLATED,
                progress_type=ptype,
                blocking_paths=blocking_paths,
                details="Lock-based synchronization found — not obstruction-free.",
            )

    return ProgressResult(
        status=CheckStatus.VERIFIED,
        progress_type=ptype,
        blocking_paths=blocking_paths,
        details=f"No violations of {ptype.value} detected.",
    )


def verify_memory_safety(source: str) -> MemorySafetyResult:
    """Verify memory safety properties in concurrent code."""
    result = MemorySafetyResult(status=CheckStatus.VERIFIED)
    lines = source.split('\n')

    freed_vars: Set[str] = set()
    allocated_vars: Set[str] = set()

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Track allocations
        alloc_match = re.search(r'(\w+)\s*=\s*(?:malloc|calloc|realloc|new\s)', stripped)
        if alloc_match:
            allocated_vars.add(alloc_match.group(1))

        # Check for double free
        free_match = re.search(r'free\s*\(\s*(\w+)\s*\)|delete\s+(\w+)', stripped)
        if free_match:
            var = free_match.group(1) or free_match.group(2)
            if var in freed_vars:
                result.double_free.append(i)
            freed_vars.add(var)

        # Check for use-after-free (simplified)
        if freed_vars:
            for var in freed_vars:
                if re.search(rf'\b{re.escape(var)}\b', stripped) and not re.search(
                        rf'free\s*\(\s*{re.escape(var)}|delete\s+{re.escape(var)}|'
                        rf'{re.escape(var)}\s*=', stripped):
                    result.use_after_free.append(i)

        # Check for null dereference
        if re.search(r'(\w+)->|(\*\w+)', stripped):
            ptr_match = re.search(r'(\w+)->', stripped) or re.search(r'\*(\w+)', stripped)
            if ptr_match:
                ptr = ptr_match.group(1)
                # Check if null check exists nearby
                context = "\n".join(lines[max(0, i - 5):i])
                if not re.search(rf'if\s*\(\s*{re.escape(ptr)}\s*(!=\s*NULL|!= nullptr|\))',
                                 context):
                    if ptr not in allocated_vars:
                        result.null_deref.append(i)

        # Check for buffer overflow patterns
        buf_match = re.search(r'(\w+)\[([^]]+)\]', stripped)
        if buf_match:
            idx_expr = buf_match.group(2)
            if re.search(r'(argc|argv|user|input|strlen)', idx_expr):
                result.buffer_overflow.append(i)

    # Check for data races: shared variable access without synchronization
    shared_writes: Dict[str, List[int]] = {}
    in_critical = False
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if re.search(r'(lock|Lock|synchronized|mutex)', stripped):
            in_critical = True
        if re.search(r'(unlock|Unlock)', stripped):
            in_critical = False

        if not in_critical:
            write_match = re.search(r'(\w+)\s*[+\-*\/]?=(?!=)', stripped)
            if write_match:
                var = write_match.group(1)
                if var not in ('int', 'long', 'float', 'double', 'char', 'void',
                               'auto', 'let', 'var', 'const', 'static'):
                    shared_writes.setdefault(var, []).append(i)

    # Flag variables written in multiple places outside critical sections
    for var, write_lines in shared_writes.items():
        if len(write_lines) > 1:
            result.data_races.extend(write_lines[:2])

    # Set overall status
    total_issues = (len(result.use_after_free) + len(result.double_free) +
                    len(result.buffer_overflow) + len(result.null_deref) +
                    len(result.data_races))

    if total_issues > 0:
        result.status = CheckStatus.VIOLATED
        parts = []
        if result.use_after_free:
            parts.append(f"{len(result.use_after_free)} use-after-free")
        if result.double_free:
            parts.append(f"{len(result.double_free)} double-free")
        if result.buffer_overflow:
            parts.append(f"{len(result.buffer_overflow)} buffer-overflow")
        if result.null_deref:
            parts.append(f"{len(result.null_deref)} null-deref")
        if result.data_races:
            parts.append(f"{len(result.data_races)} data-race")
        result.details = f"Found {total_issues} issue(s): {', '.join(parts)}."
    else:
        result.details = "No memory safety issues detected."

    return result


def bounded_model_check(source: str, bound: int = 10) -> BMCResult:
    """Bounded model checking: explore interleavings up to a given depth.

    Limits the number of context switches to 'bound' for tractability.
    """
    threads = _extract_threads(source)
    shared_vars = _extract_shared_vars(source)
    n_threads = len(threads)

    if n_threads == 0:
        return BMCResult(
            status=CheckStatus.VERIFIED, bound=bound,
            details="No threads found.",
        )

    init_vars = frozenset((v, 0) for v in shared_vars)
    init_pc = tuple(0 for _ in range(n_threads))
    init_state = _State(pc=init_pc, variables=init_vars, locks_held=frozenset())

    states_explored = 0
    max_states = min(10 ** (bound + 1), 500000)

    # BFS with bounded depth
    frontier: List[Tuple[_State, int]] = [(init_state, 0)]  # (state, depth)
    visited: Set[_State] = set()

    while frontier and states_explored < max_states:
        state, depth = frontier.pop(0)

        if state in visited or depth > bound:
            continue
        visited.add(state)
        states_explored += 1

        all_done = all(state.pc[t] >= len(threads[t]) for t in range(n_threads))
        if all_done:
            continue

        can_progress = False
        for tid in range(n_threads):
            pc = state.pc[tid]
            if pc >= len(threads[tid]):
                continue

            instr = threads[tid][pc]
            lock = _is_lock_acquire(instr)
            if lock:
                held = dict(state.locks_held)
                if lock in held and held[lock] != tid:
                    continue

            can_progress = True
            new_pc = list(state.pc)
            new_pc[tid] = pc + 1
            new_vars = dict(state.variables)
            new_locks = dict(state.locks_held)

            if lock:
                new_locks[lock] = tid
            release = _is_lock_release(instr)
            if release and release in new_locks:
                del new_locks[release]

            write_match = re.search(r'(\w+)\s*=\s*(\d+)', instr)
            if write_match and write_match.group(1) in new_vars:
                new_vars[write_match.group(1)] = int(write_match.group(2))

            # Check for assertion violations
            assert_match = re.search(r'assert\s*\(\s*(.+)\s*\)', instr)
            if assert_match:
                cond = assert_match.group(1)
                var_match = re.search(r'(\w+)\s*(==|!=)\s*(\d+)', cond)
                if var_match:
                    vname = var_match.group(1)
                    op = var_match.group(2)
                    val = int(var_match.group(3))
                    cur = new_vars.get(vname, 0)
                    holds = (cur == val) if op == "==" else (cur != val)
                    if not holds:
                        ce = Counterexample(
                            trace=[f"Thread {tid} step {pc}: {instr} FAILED"],
                            final_state={str(k): str(v) for k, v in new_vars.items()},
                        )
                        return BMCResult(
                            status=CheckStatus.VIOLATED, bound=bound,
                            states_explored=states_explored,
                            counterexample=ce,
                            details=f"Assertion violation found at depth {depth}.",
                        )

            new_state = _State(
                pc=tuple(new_pc),
                variables=frozenset(new_vars.items()),
                locks_held=frozenset(new_locks.items()),
            )
            frontier.append((new_state, depth + 1))

        # Check deadlock within bound
        if not can_progress and not all_done:
            ce = Counterexample(
                trace=["Deadlock: no thread can proceed"],
                final_state={str(k): str(v) for k, v in state.variables},
            )
            return BMCResult(
                status=CheckStatus.VIOLATED, bound=bound,
                states_explored=states_explored,
                counterexample=ce,
                details=f"Deadlock found at depth {depth}.",
            )

    status = CheckStatus.VERIFIED if states_explored < max_states else CheckStatus.TIMEOUT

    return BMCResult(
        status=status, bound=bound,
        states_explored=states_explored,
        details=f"Explored {states_explored} states within bound {bound}.",
    )


def abstraction_refinement(source: str, property: str) -> CEGARResult:
    """Counterexample-guided abstraction refinement (CEGAR).

    Iteratively refines an abstract model until the property is verified
    or a real counterexample is found.
    """
    threads = _extract_threads(source)
    shared_vars = _extract_shared_vars(source)

    if not threads:
        return CEGARResult(
            status=CheckStatus.VERIFIED, iterations=0,
            details="No threads to verify.",
        )

    # Start with a coarse abstraction: track only lock state
    abstraction_vars: Set[str] = set()
    iterations = 0
    max_iterations = 10

    while iterations < max_iterations:
        iterations += 1

        # Build abstract model
        abstract_source = _build_abstract_source(source, abstraction_vars, threads)

        # Model check the abstraction
        result = model_check(abstract_source, property)

        if result.status == CheckStatus.VERIFIED:
            return CEGARResult(
                status=CheckStatus.VERIFIED,
                iterations=iterations,
                final_abstraction_size=len(abstraction_vars),
                details=f"Verified after {iterations} CEGAR iteration(s) "
                        f"with {len(abstraction_vars)} tracked variable(s).",
            )

        if result.status == CheckStatus.TIMEOUT:
            return CEGARResult(
                status=CheckStatus.TIMEOUT,
                iterations=iterations,
                final_abstraction_size=len(abstraction_vars),
                details="Model checking timed out during CEGAR.",
            )

        # Spurious counterexample? Refine by adding more variables
        if result.counterexample:
            # Extract variables mentioned in the counterexample
            ce_text = str(result.counterexample)
            new_vars = set(re.findall(r'\b(\w+)\s*=', ce_text))
            new_vars &= set(shared_vars)
            new_vars -= abstraction_vars

            if not new_vars:
                # Cannot refine further — counterexample is real
                return CEGARResult(
                    status=CheckStatus.VIOLATED,
                    iterations=iterations,
                    final_abstraction_size=len(abstraction_vars),
                    counterexample=result.counterexample,
                    details=f"Real counterexample found after {iterations} iteration(s).",
                )

            abstraction_vars |= new_vars

    return CEGARResult(
        status=CheckStatus.UNKNOWN,
        iterations=iterations,
        final_abstraction_size=len(abstraction_vars),
        details=f"CEGAR did not converge after {max_iterations} iterations.",
    )


def _build_abstract_source(source: str, tracked_vars: Set[str],
                           threads: List[List[str]]) -> str:
    """Build an abstracted version of the source keeping only tracked variables."""
    if not tracked_vars:
        return source  # no abstraction, full model

    abstract_lines: List[str] = []
    for line in source.split('\n'):
        stripped = line.strip()
        # Keep lock operations and tracked variable operations
        is_lock = bool(re.search(r'(lock|unlock|Lock|Unlock|synchronized)', stripped))
        is_tracked = any(var in stripped for var in tracked_vars)
        is_assert = 'assert' in stripped

        if is_lock or is_tracked or is_assert:
            abstract_lines.append(line)
        else:
            abstract_lines.append(f"// abstracted: {stripped}")

    return "\n".join(abstract_lines)

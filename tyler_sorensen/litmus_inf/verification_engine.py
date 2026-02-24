"""
Formal verification engine for concurrent programs.

Verifies mutual exclusion, deadlock freedom, livelock freedom, starvation freedom,
linearizability, wait-freedom, lock-freedom, and obstruction-freedom.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
from enum import Enum
import itertools
import copy


class PropertyType(Enum):
    MUTUAL_EXCLUSION = "mutual_exclusion"
    DEADLOCK_FREEDOM = "deadlock_freedom"
    LIVELOCK_FREEDOM = "livelock_freedom"
    STARVATION_FREEDOM = "starvation_freedom"
    LINEARIZABILITY = "linearizability"
    WAIT_FREEDOM = "wait_freedom"
    LOCK_FREEDOM = "lock_freedom"
    OBSTRUCTION_FREEDOM = "obstruction_freedom"


class ThreadState(Enum):
    IDLE = "idle"
    REQUESTING = "requesting"
    IN_CS = "in_critical_section"
    RELEASING = "releasing"
    WAITING = "waiting"
    BLOCKED = "blocked"
    COMPLETED = "completed"


@dataclass
class Instruction:
    thread_id: int
    index: int
    op_type: str  # "store", "load", "lock", "unlock", "cas", "fence", "branch", "nop"
    variable: str = ""
    value: Any = None
    target: Optional[int] = None
    condition: Optional[str] = None
    lock_id: str = ""

    def __repr__(self):
        return f"T{self.thread_id}:{self.index}:{self.op_type}({self.variable})"

    def __hash__(self):
        return hash((self.thread_id, self.index))

    def __eq__(self, other):
        return (isinstance(other, Instruction) and
                self.thread_id == other.thread_id and self.index == other.index)


@dataclass
class SystemState:
    thread_pcs: Dict[int, int]
    memory: Dict[str, Any]
    thread_states: Dict[int, ThreadState]
    lock_owners: Dict[str, Optional[int]]
    waiting_on: Dict[int, Optional[str]]
    local_vars: Dict[int, Dict[str, Any]]
    history: List[Tuple[int, str]] = field(default_factory=list)

    def __hash__(self):
        return hash((
            tuple(sorted(self.thread_pcs.items())),
            tuple(sorted(self.memory.items())),
            tuple(sorted((k, v.value) for k, v in self.thread_states.items())),
            tuple(sorted((k, v) for k, v in self.lock_owners.items())),
        ))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def copy(self) -> 'SystemState':
        return SystemState(
            thread_pcs=dict(self.thread_pcs),
            memory={k: v for k, v in self.memory.items()},
            thread_states={k: v for k, v in self.thread_states.items()},
            lock_owners={k: v for k, v in self.lock_owners.items()},
            waiting_on=dict(self.waiting_on),
            local_vars={k: dict(v) for k, v in self.local_vars.items()},
            history=list(self.history),
        )

    def is_terminal(self, programs: Dict[int, List[Instruction]]) -> bool:
        return all(
            self.thread_pcs[tid] >= len(programs[tid]) or
            self.thread_states[tid] == ThreadState.BLOCKED
            for tid in programs
        )


@dataclass
class VerificationResult:
    property_type: PropertyType
    verified: bool
    counterexample: Optional[List[Tuple[int, str]]] = None
    states_explored: int = 0
    proof_depth: int = 0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        status = "VERIFIED" if self.verified else "VIOLATED"
        return (f"VerificationResult({self.property_type.value}: {status}, "
                f"states={self.states_explored}, depth={self.proof_depth})")


@dataclass
class LinearizationPoint:
    thread_id: int
    operation: str
    args: Tuple
    result: Any
    timestamp: int


@dataclass
class ConcurrentHistory:
    events: List[LinearizationPoint]

    def sequential_specs(self) -> List[List[LinearizationPoint]]:
        return list(itertools.permutations(self.events))


class StateExplorer:
    """Explicit-state model checker for concurrent programs."""

    def __init__(self, max_states: int = 100000, max_depth: int = 500):
        self.max_states = max_states
        self.max_depth = max_depth
        self.states_explored = 0

    def explore(self, programs: Dict[int, List[Instruction]],
                initial_state: SystemState,
                check_fn: Callable[[SystemState], bool]) -> Tuple[bool, Optional[SystemState]]:
        self.states_explored = 0
        visited: Set[int] = set()
        stack = [(initial_state, 0)]

        while stack and self.states_explored < self.max_states:
            state, depth = stack.pop()
            state_hash = hash(state)

            if state_hash in visited:
                continue
            visited.add(state_hash)
            self.states_explored += 1

            if not check_fn(state):
                return False, state

            if depth >= self.max_depth:
                continue

            if state.is_terminal(programs):
                continue

            successors = self._get_successors(state, programs)
            for succ in successors:
                if hash(succ) not in visited:
                    stack.append((succ, depth + 1))

        return True, None

    def explore_all_states(self, programs: Dict[int, List[Instruction]],
                           initial_state: SystemState) -> List[SystemState]:
        self.states_explored = 0
        visited: Set[int] = set()
        all_states = []
        stack = [(initial_state, 0)]

        while stack and self.states_explored < self.max_states:
            state, depth = stack.pop()
            state_hash = hash(state)

            if state_hash in visited:
                continue
            visited.add(state_hash)
            self.states_explored += 1
            all_states.append(state)

            if depth >= self.max_depth or state.is_terminal(programs):
                continue

            for succ in self._get_successors(state, programs):
                if hash(succ) not in visited:
                    stack.append((succ, depth + 1))

        return all_states

    def _get_successors(self, state: SystemState,
                        programs: Dict[int, List[Instruction]]) -> List[SystemState]:
        successors = []
        for tid in programs:
            pc = state.thread_pcs[tid]
            if pc >= len(programs[tid]):
                continue
            if state.thread_states[tid] == ThreadState.BLOCKED:
                if state.waiting_on[tid]:
                    lock_id = state.waiting_on[tid]
                    if state.lock_owners.get(lock_id) is None:
                        new_state = state.copy()
                        new_state.lock_owners[lock_id] = tid
                        new_state.thread_states[tid] = ThreadState.IN_CS
                        new_state.waiting_on[tid] = None
                        new_state.thread_pcs[tid] = pc + 1
                        new_state.history.append((tid, f"acquired {lock_id}"))
                        successors.append(new_state)
                continue

            inst = programs[tid][pc]
            new_state = self._execute_instruction(state, inst, programs)
            if new_state:
                successors.append(new_state)
        return successors

    def _execute_instruction(self, state: SystemState, inst: Instruction,
                             programs: Dict[int, List[Instruction]]) -> Optional[SystemState]:
        new_state = state.copy()
        tid = inst.thread_id

        if inst.op_type == "store":
            new_state.memory[inst.variable] = inst.value
            new_state.thread_pcs[tid] += 1
            new_state.history.append((tid, f"store {inst.variable}={inst.value}"))

        elif inst.op_type == "load":
            val = new_state.memory.get(inst.variable, 0)
            new_state.local_vars.setdefault(tid, {})[f"r_{inst.variable}"] = val
            new_state.thread_pcs[tid] += 1
            new_state.history.append((tid, f"load {inst.variable}={val}"))

        elif inst.op_type == "lock":
            lock_id = inst.lock_id or inst.variable
            if new_state.lock_owners.get(lock_id) is None:
                new_state.lock_owners[lock_id] = tid
                new_state.thread_states[tid] = ThreadState.IN_CS
                new_state.thread_pcs[tid] += 1
                new_state.history.append((tid, f"lock {lock_id}"))
            else:
                new_state.thread_states[tid] = ThreadState.BLOCKED
                new_state.waiting_on[tid] = lock_id
                new_state.history.append((tid, f"blocked on {lock_id}"))

        elif inst.op_type == "unlock":
            lock_id = inst.lock_id or inst.variable
            new_state.lock_owners[lock_id] = None
            new_state.thread_states[tid] = ThreadState.IDLE
            new_state.thread_pcs[tid] += 1
            new_state.history.append((tid, f"unlock {lock_id}"))

        elif inst.op_type == "cas":
            current = new_state.memory.get(inst.variable, 0)
            expected, desired = inst.value if isinstance(inst.value, tuple) else (0, 1)
            if current == expected:
                new_state.memory[inst.variable] = desired
                new_state.local_vars.setdefault(tid, {})["cas_result"] = True
            else:
                new_state.local_vars.setdefault(tid, {})["cas_result"] = False
            new_state.thread_pcs[tid] += 1
            new_state.history.append((tid, f"cas {inst.variable} {current}=={expected}"))

        elif inst.op_type == "fence":
            new_state.thread_pcs[tid] += 1
            new_state.history.append((tid, "fence"))

        elif inst.op_type == "branch":
            cond_var = inst.condition
            cond_val = new_state.local_vars.get(tid, {}).get(cond_var, 0)
            if cond_val and inst.target is not None:
                new_state.thread_pcs[tid] = inst.target
            else:
                new_state.thread_pcs[tid] += 1
            new_state.history.append((tid, f"branch {cond_var}={cond_val}"))

        elif inst.op_type == "nop":
            new_state.thread_pcs[tid] += 1

        else:
            new_state.thread_pcs[tid] += 1

        return new_state


class MutualExclusionVerifier:
    """Verify that no two threads are in a critical section simultaneously."""

    def __init__(self, explorer: StateExplorer):
        self.explorer = explorer

    def verify(self, programs: Dict[int, List[Instruction]],
               initial_state: SystemState,
               cs_markers: Dict[int, Tuple[int, int]]) -> VerificationResult:
        def check_mutex(state: SystemState) -> bool:
            in_cs = []
            for tid, (start, end) in cs_markers.items():
                pc = state.thread_pcs.get(tid, 0)
                if start <= pc < end:
                    in_cs.append(tid)
            return len(in_cs) <= 1

        satisfied, violation_state = self.explorer.explore(
            programs, initial_state, check_mutex)

        return VerificationResult(
            property_type=PropertyType.MUTUAL_EXCLUSION,
            verified=satisfied,
            counterexample=violation_state.history if violation_state else None,
            states_explored=self.explorer.states_explored,
            proof_depth=self.explorer.max_depth,
            message="Mutual exclusion holds" if satisfied else "Mutual exclusion violated",
        )


class DeadlockFreedomVerifier:
    """Verify that no reachable state is a deadlock."""

    def __init__(self, explorer: StateExplorer):
        self.explorer = explorer

    def verify(self, programs: Dict[int, List[Instruction]],
               initial_state: SystemState) -> VerificationResult:
        all_states = self.explorer.explore_all_states(programs, initial_state)

        for state in all_states:
            if self._is_deadlock(state, programs):
                return VerificationResult(
                    property_type=PropertyType.DEADLOCK_FREEDOM,
                    verified=False,
                    counterexample=state.history,
                    states_explored=self.explorer.states_explored,
                    message="Deadlock detected",
                    details={"deadlock_state": str(state.lock_owners)},
                )

        return VerificationResult(
            property_type=PropertyType.DEADLOCK_FREEDOM,
            verified=True,
            states_explored=self.explorer.states_explored,
            message="No deadlock reachable",
        )

    def _is_deadlock(self, state: SystemState,
                     programs: Dict[int, List[Instruction]]) -> bool:
        blocked_threads = []
        for tid in programs:
            pc = state.thread_pcs[tid]
            if pc < len(programs[tid]):
                if state.thread_states[tid] == ThreadState.BLOCKED:
                    blocked_threads.append(tid)
                else:
                    return False  # at least one thread can progress
            # completed threads are fine

        if not blocked_threads:
            return False

        # Check for circular wait
        wait_graph: Dict[int, int] = {}
        for tid in blocked_threads:
            lock_waiting = state.waiting_on.get(tid)
            if lock_waiting:
                owner = state.lock_owners.get(lock_waiting)
                if owner is not None and owner != tid:
                    wait_graph[tid] = owner

        return self._has_cycle(wait_graph)

    def _has_cycle(self, graph: Dict[int, int]) -> bool:
        visited = set()
        in_stack = set()

        def dfs(node):
            if node in in_stack:
                return True
            if node in visited:
                return False
            visited.add(node)
            in_stack.add(node)
            if node in graph:
                if dfs(graph[node]):
                    return True
            in_stack.discard(node)
            return False

        for node in graph:
            if node not in visited:
                if dfs(node):
                    return True
        return False


class LivelockFreedomVerifier:
    """Detect repeated state cycles (livelock)."""

    def __init__(self, explorer: StateExplorer, max_cycle_length: int = 50):
        self.explorer = explorer
        self.max_cycle_length = max_cycle_length

    def verify(self, programs: Dict[int, List[Instruction]],
               initial_state: SystemState) -> VerificationResult:
        states_explored = 0
        state_sequence: List[int] = []
        visited_states: Dict[int, int] = {}

        current = initial_state
        depth = 0

        while depth < self.explorer.max_depth and states_explored < self.explorer.max_states:
            state_hash = hash(current)
            states_explored += 1

            if state_hash in visited_states:
                cycle_start = visited_states[state_hash]
                cycle_length = depth - cycle_start

                if cycle_length <= self.max_cycle_length:
                    all_done = all(
                        current.thread_pcs[tid] >= len(programs[tid])
                        for tid in programs
                    )
                    if not all_done:
                        return VerificationResult(
                            property_type=PropertyType.LIVELOCK_FREEDOM,
                            verified=False,
                            counterexample=current.history,
                            states_explored=states_explored,
                            proof_depth=depth,
                            message=f"Livelock detected: cycle length {cycle_length}",
                            details={"cycle_length": cycle_length, "cycle_start": cycle_start},
                        )
                break

            visited_states[state_hash] = depth
            state_sequence.append(state_hash)

            if current.is_terminal(programs):
                break

            successors = self.explorer._get_successors(current, programs)
            if not successors:
                break
            current = successors[0]
            depth += 1

        return VerificationResult(
            property_type=PropertyType.LIVELOCK_FREEDOM,
            verified=True,
            states_explored=states_explored,
            proof_depth=depth,
            message="No livelock detected",
        )


class StarvationFreedomVerifier:
    """Verify that every requesting thread eventually enters its critical section."""

    def __init__(self, explorer: StateExplorer):
        self.explorer = explorer

    def verify(self, programs: Dict[int, List[Instruction]],
               initial_state: SystemState,
               requesting_threads: Set[int]) -> VerificationResult:
        all_states = self.explorer.explore_all_states(programs, initial_state)

        terminal_states = [
            s for s in all_states
            if s.is_terminal(programs)
        ]

        for state in terminal_states:
            for tid in requesting_threads:
                if state.thread_states[tid] == ThreadState.BLOCKED:
                    return VerificationResult(
                        property_type=PropertyType.STARVATION_FREEDOM,
                        verified=False,
                        counterexample=state.history,
                        states_explored=self.explorer.states_explored,
                        message=f"Thread {tid} starved",
                        details={"starved_thread": tid},
                    )

        entered_cs: Dict[int, bool] = {tid: False for tid in requesting_threads}
        for state in all_states:
            for tid in requesting_threads:
                if state.thread_states.get(tid) == ThreadState.IN_CS:
                    entered_cs[tid] = True

        for tid, entered in entered_cs.items():
            if not entered:
                return VerificationResult(
                    property_type=PropertyType.STARVATION_FREEDOM,
                    verified=False,
                    states_explored=self.explorer.states_explored,
                    message=f"Thread {tid} never enters critical section",
                    details={"starved_thread": tid},
                )

        return VerificationResult(
            property_type=PropertyType.STARVATION_FREEDOM,
            verified=True,
            states_explored=self.explorer.states_explored,
            message="No starvation: all requesting threads enter CS",
        )


class LinearizabilityVerifier:
    """Verify linearizability of concurrent data structure operations."""

    def __init__(self, max_histories: int = 10000):
        self.max_histories = max_histories

    def verify(self, history: ConcurrentHistory,
               sequential_spec: Callable[[List[Tuple[str, Tuple]], List[Any]], bool]
               ) -> VerificationResult:
        events = history.events
        n = len(events)
        if n == 0:
            return VerificationResult(
                property_type=PropertyType.LINEARIZABILITY,
                verified=True, states_explored=0,
                message="Empty history is linearizable")

        for perm in itertools.islice(itertools.permutations(range(n)), self.max_histories):
            ordered = [events[i] for i in perm]

            if not self._respects_real_time(events, list(perm)):
                continue

            ops = [(e.operation, e.args) for e in ordered]
            results = [e.result for e in ordered]

            if sequential_spec(ops, results):
                return VerificationResult(
                    property_type=PropertyType.LINEARIZABILITY,
                    verified=True,
                    states_explored=len(list(perm)),
                    message="History is linearizable",
                    details={"linearization": list(perm)},
                )

        return VerificationResult(
            property_type=PropertyType.LINEARIZABILITY,
            verified=False,
            states_explored=self.max_histories,
            message="History is NOT linearizable",
        )

    def _respects_real_time(self, events: List[LinearizationPoint],
                            order: List[int]) -> bool:
        position = {idx: pos for pos, idx in enumerate(order)}
        for i, e1 in enumerate(events):
            for j, e2 in enumerate(events):
                if i != j and e1.timestamp < e2.timestamp:
                    if e1.thread_id != e2.thread_id:
                        continue
                    if position[i] > position[j]:
                        return False
        return True

    def verify_queue(self, enqueue_ops: List[Tuple[int, Any]],
                     dequeue_ops: List[Tuple[int, Any]]) -> VerificationResult:
        def queue_spec(ops, results):
            q = []
            for (op, args), result in zip(ops, results):
                if op == "enqueue":
                    q.append(args[0])
                elif op == "dequeue":
                    if not q:
                        if result is not None:
                            return False
                    else:
                        expected = q.pop(0)
                        if result != expected:
                            return False
            return True

        events = []
        ts = 0
        for tid, val in enqueue_ops:
            events.append(LinearizationPoint(tid, "enqueue", (val,), None, ts))
            ts += 1
        for tid, val in dequeue_ops:
            events.append(LinearizationPoint(tid, "dequeue", (), val, ts))
            ts += 1

        return self.verify(ConcurrentHistory(events), queue_spec)

    def verify_stack(self, push_ops: List[Tuple[int, Any]],
                     pop_ops: List[Tuple[int, Any]]) -> VerificationResult:
        def stack_spec(ops, results):
            s = []
            for (op, args), result in zip(ops, results):
                if op == "push":
                    s.append(args[0])
                elif op == "pop":
                    if not s:
                        if result is not None:
                            return False
                    else:
                        expected = s.pop()
                        if result != expected:
                            return False
            return True

        events = []
        ts = 0
        for tid, val in push_ops:
            events.append(LinearizationPoint(tid, "push", (val,), None, ts))
            ts += 1
        for tid, val in pop_ops:
            events.append(LinearizationPoint(tid, "pop", (), val, ts))
            ts += 1

        return self.verify(ConcurrentHistory(events), stack_spec)


class WaitFreedomVerifier:
    """Verify that every operation completes in bounded steps."""

    def __init__(self, explorer: StateExplorer, bound: int = 100):
        self.explorer = explorer
        self.bound = bound

    def verify(self, programs: Dict[int, List[Instruction]],
               initial_state: SystemState) -> VerificationResult:
        all_states = self.explorer.explore_all_states(programs, initial_state)

        max_steps_per_thread: Dict[int, int] = {}
        for state in all_states:
            for tid in programs:
                steps = state.thread_pcs.get(tid, 0)
                if tid not in max_steps_per_thread or steps > max_steps_per_thread[tid]:
                    max_steps_per_thread[tid] = steps

        unbounded_threads = []
        for tid, max_steps in max_steps_per_thread.items():
            if max_steps >= self.bound:
                unbounded_threads.append(tid)

        # Check if any thread can be blocked indefinitely
        for state in all_states:
            for tid in programs:
                if state.thread_states.get(tid) == ThreadState.BLOCKED:
                    return VerificationResult(
                        property_type=PropertyType.WAIT_FREEDOM,
                        verified=False,
                        counterexample=state.history,
                        states_explored=self.explorer.states_explored,
                        message=f"Thread {tid} can be blocked (not wait-free)",
                        details={"blocked_thread": tid},
                    )

        if unbounded_threads:
            return VerificationResult(
                property_type=PropertyType.WAIT_FREEDOM,
                verified=False,
                states_explored=self.explorer.states_explored,
                message=f"Threads {unbounded_threads} may exceed step bound {self.bound}",
                details={"unbounded": unbounded_threads},
            )

        return VerificationResult(
            property_type=PropertyType.WAIT_FREEDOM,
            verified=True,
            states_explored=self.explorer.states_explored,
            message=f"Wait-free: all operations complete within {self.bound} steps",
        )


class LockFreedomVerifier:
    """Verify that some thread always makes progress."""

    def __init__(self, explorer: StateExplorer):
        self.explorer = explorer

    def verify(self, programs: Dict[int, List[Instruction]],
               initial_state: SystemState) -> VerificationResult:
        all_states = self.explorer.explore_all_states(programs, initial_state)

        for state in all_states:
            any_can_progress = False
            all_done = True

            for tid in programs:
                pc = state.thread_pcs[tid]
                if pc < len(programs[tid]):
                    all_done = False
                    if state.thread_states[tid] != ThreadState.BLOCKED:
                        any_can_progress = True
                        break

            if all_done:
                continue

            if not any_can_progress:
                return VerificationResult(
                    property_type=PropertyType.LOCK_FREEDOM,
                    verified=False,
                    counterexample=state.history,
                    states_explored=self.explorer.states_explored,
                    message="Lock-freedom violated: no thread can progress",
                )

        return VerificationResult(
            property_type=PropertyType.LOCK_FREEDOM,
            verified=True,
            states_explored=self.explorer.states_explored,
            message="Lock-free: some thread always makes progress",
        )


class ObstructionFreedomVerifier:
    """Verify that a thread makes progress when run in isolation."""

    def __init__(self, explorer: StateExplorer):
        self.explorer = explorer

    def verify(self, programs: Dict[int, List[Instruction]],
               initial_state: SystemState) -> VerificationResult:
        for tid in programs:
            solo_programs = {tid: programs[tid]}
            solo_state = SystemState(
                thread_pcs={tid: 0},
                memory=dict(initial_state.memory),
                thread_states={tid: ThreadState.IDLE},
                lock_owners=dict(initial_state.lock_owners),
                waiting_on={tid: None},
                local_vars={tid: {}},
            )

            all_states = self.explorer.explore_all_states(solo_programs, solo_state)

            completes = False
            for state in all_states:
                if state.thread_pcs[tid] >= len(programs[tid]):
                    completes = True
                    break

            if not completes:
                return VerificationResult(
                    property_type=PropertyType.OBSTRUCTION_FREEDOM,
                    verified=False,
                    states_explored=self.explorer.states_explored,
                    message=f"Thread {tid} doesn't complete in isolation",
                    details={"thread": tid},
                )

        return VerificationResult(
            property_type=PropertyType.OBSTRUCTION_FREEDOM,
            verified=True,
            states_explored=self.explorer.states_explored,
            message="Obstruction-free: every thread completes in isolation",
        )


class VerificationEngine:
    """Main verification engine: orchestrates all property verifications."""

    def __init__(self, max_states: int = 50000, max_depth: int = 200):
        self.explorer = StateExplorer(max_states=max_states, max_depth=max_depth)
        self.mutex_verifier = MutualExclusionVerifier(self.explorer)
        self.deadlock_verifier = DeadlockFreedomVerifier(self.explorer)
        self.livelock_verifier = LivelockFreedomVerifier(self.explorer)
        self.starvation_verifier = StarvationFreedomVerifier(self.explorer)
        self.linearizability_verifier = LinearizabilityVerifier()
        self.wait_freedom_verifier = WaitFreedomVerifier(self.explorer)
        self.lock_freedom_verifier = LockFreedomVerifier(self.explorer)
        self.obstruction_verifier = ObstructionFreedomVerifier(self.explorer)

    def verify(self, programs: Dict[int, List[Instruction]],
               property_type: PropertyType,
               **kwargs) -> VerificationResult:
        initial_state = kwargs.get("initial_state", self._default_state(programs))

        if property_type == PropertyType.MUTUAL_EXCLUSION:
            cs_markers = kwargs.get("cs_markers", {})
            return self.mutex_verifier.verify(programs, initial_state, cs_markers)

        elif property_type == PropertyType.DEADLOCK_FREEDOM:
            return self.deadlock_verifier.verify(programs, initial_state)

        elif property_type == PropertyType.LIVELOCK_FREEDOM:
            return self.livelock_verifier.verify(programs, initial_state)

        elif property_type == PropertyType.STARVATION_FREEDOM:
            requesting = kwargs.get("requesting_threads", set(programs.keys()))
            return self.starvation_verifier.verify(programs, initial_state, requesting)

        elif property_type == PropertyType.LINEARIZABILITY:
            history = kwargs.get("history")
            spec = kwargs.get("sequential_spec")
            if history and spec:
                return self.linearizability_verifier.verify(history, spec)
            return VerificationResult(
                property_type=PropertyType.LINEARIZABILITY,
                verified=False, message="Missing history or spec")

        elif property_type == PropertyType.WAIT_FREEDOM:
            return self.wait_freedom_verifier.verify(programs, initial_state)

        elif property_type == PropertyType.LOCK_FREEDOM:
            return self.lock_freedom_verifier.verify(programs, initial_state)

        elif property_type == PropertyType.OBSTRUCTION_FREEDOM:
            return self.obstruction_verifier.verify(programs, initial_state)

        return VerificationResult(
            property_type=property_type, verified=False,
            message=f"Unknown property: {property_type}")

    def verify_all(self, programs: Dict[int, List[Instruction]],
                   **kwargs) -> Dict[PropertyType, VerificationResult]:
        results = {}
        initial_state = kwargs.get("initial_state", self._default_state(programs))

        results[PropertyType.DEADLOCK_FREEDOM] = self.deadlock_verifier.verify(
            programs, initial_state)
        results[PropertyType.LOCK_FREEDOM] = self.lock_freedom_verifier.verify(
            programs, initial_state)
        results[PropertyType.OBSTRUCTION_FREEDOM] = self.obstruction_verifier.verify(
            programs, initial_state)

        if "cs_markers" in kwargs:
            results[PropertyType.MUTUAL_EXCLUSION] = self.mutex_verifier.verify(
                programs, initial_state, kwargs["cs_markers"])

        return results

    def _default_state(self, programs: Dict[int, List[Instruction]]) -> SystemState:
        return SystemState(
            thread_pcs={tid: 0 for tid in programs},
            memory={},
            thread_states={tid: ThreadState.IDLE for tid in programs},
            lock_owners={},
            waiting_on={tid: None for tid in programs},
            local_vars={tid: {} for tid in programs},
        )


def build_peterson_programs() -> Dict[int, List[Instruction]]:
    """Build Peterson's algorithm for verification."""
    t0 = [
        Instruction(0, 0, "store", "flag0", 1),
        Instruction(0, 1, "store", "turn", 1),
        Instruction(0, 2, "load", "flag1"),
        Instruction(0, 3, "load", "turn"),
        # CS
        Instruction(0, 4, "store", "x", 1),
        Instruction(0, 5, "store", "flag0", 0),
    ]
    t1 = [
        Instruction(1, 0, "store", "flag1", 1),
        Instruction(1, 1, "store", "turn", 0),
        Instruction(1, 2, "load", "flag0"),
        Instruction(1, 3, "load", "turn"),
        # CS
        Instruction(1, 4, "store", "x", 2),
        Instruction(1, 5, "store", "flag1", 0),
    ]
    return {0: t0, 1: t1}


def build_lock_programs() -> Dict[int, List[Instruction]]:
    """Build lock-based mutual exclusion."""
    t0 = [
        Instruction(0, 0, "lock", lock_id="L0"),
        Instruction(0, 1, "store", "x", 1),
        Instruction(0, 2, "unlock", lock_id="L0"),
    ]
    t1 = [
        Instruction(1, 0, "lock", lock_id="L0"),
        Instruction(1, 1, "store", "x", 2),
        Instruction(1, 2, "unlock", lock_id="L0"),
    ]
    return {0: t0, 1: t1}


def build_broken_mutex_programs() -> Dict[int, List[Instruction]]:
    """Build a broken mutex (no lock) for testing violation detection."""
    t0 = [
        Instruction(0, 0, "store", "x", 1),
        Instruction(0, 1, "load", "x"),
    ]
    t1 = [
        Instruction(1, 0, "store", "x", 2),
        Instruction(1, 1, "load", "x"),
    ]
    return {0: t0, 1: t1}


def build_deadlock_programs() -> Dict[int, List[Instruction]]:
    """Build a program with potential deadlock (lock ordering violation)."""
    t0 = [
        Instruction(0, 0, "lock", lock_id="L0"),
        Instruction(0, 1, "lock", lock_id="L1"),
        Instruction(0, 2, "store", "x", 1),
        Instruction(0, 3, "unlock", lock_id="L1"),
        Instruction(0, 4, "unlock", lock_id="L0"),
    ]
    t1 = [
        Instruction(1, 0, "lock", lock_id="L1"),
        Instruction(1, 1, "lock", lock_id="L0"),
        Instruction(1, 2, "store", "x", 2),
        Instruction(1, 3, "unlock", lock_id="L0"),
        Instruction(1, 4, "unlock", lock_id="L1"),
    ]
    return {0: t0, 1: t1}


def build_lockfree_programs() -> Dict[int, List[Instruction]]:
    """Build a lock-free CAS-based program."""
    t0 = [
        Instruction(0, 0, "load", "counter"),
        Instruction(0, 1, "cas", "counter", (0, 1)),
        Instruction(0, 2, "store", "done0", 1),
    ]
    t1 = [
        Instruction(1, 0, "load", "counter"),
        Instruction(1, 1, "cas", "counter", (0, 1)),
        Instruction(1, 2, "store", "done1", 1),
    ]
    return {0: t0, 1: t1}

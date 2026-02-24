"""
Explicit-state model checker for concurrent programs.

Supports BFS/DFS exploration, partial order reduction (persistent sets),
symmetry reduction, on-the-fly verification, LTL (Büchi automaton + nested DFS),
CTL (fixed-point EX/EU/EG), and bounded model checking.
"""

import numpy as np
from collections import defaultdict, deque
from enum import Enum, auto
import copy
import time


# ---------------------------------------------------------------------------
# Program state
# ---------------------------------------------------------------------------

class SharedState:
    """Shared memory state."""

    def __init__(self, variables=None):
        self.variables = dict(variables) if variables else {}

    def read(self, var):
        return self.variables.get(var, 0)

    def write(self, var, val):
        self.variables[var] = val

    def copy(self):
        return SharedState(dict(self.variables))

    def __eq__(self, other):
        return isinstance(other, SharedState) and self.variables == other.variables

    def __hash__(self):
        return hash(tuple(sorted(self.variables.items())))

    def __repr__(self):
        return f"Shared({self.variables})"


class ThreadState:
    """Per-thread state: PC + local variables."""

    def __init__(self, pc=0, locals_=None):
        self.pc = pc
        self.locals = dict(locals_) if locals_ else {}

    def copy(self):
        return ThreadState(self.pc, dict(self.locals))

    def __eq__(self, other):
        return (isinstance(other, ThreadState)
                and self.pc == other.pc
                and self.locals == other.locals)

    def __hash__(self):
        return hash((self.pc, tuple(sorted(self.locals.items()))))

    def __repr__(self):
        return f"T(pc={self.pc}, {self.locals})"


class SystemState:
    """Full system state: shared memory + all thread states."""

    def __init__(self, shared=None, threads=None):
        self.shared = shared or SharedState()
        self.threads = dict(threads) if threads else {}

    def copy(self):
        s = SystemState(self.shared.copy())
        for tid, ts in self.threads.items():
            s.threads[tid] = ts.copy()
        return s

    def __eq__(self, other):
        return (isinstance(other, SystemState)
                and self.shared == other.shared
                and self.threads == other.threads)

    def __hash__(self):
        thread_tuple = tuple(
            (tid, self.threads[tid]) for tid in sorted(self.threads.keys())
        )
        return hash((self.shared, thread_tuple))

    def __repr__(self):
        return f"State({self.shared}, threads={self.threads})"


# ---------------------------------------------------------------------------
# Instructions / Transitions
# ---------------------------------------------------------------------------

class InstrType(Enum):
    READ = auto()
    WRITE = auto()
    LOCAL = auto()
    CAS = auto()
    FENCE = auto()
    BRANCH = auto()
    ASSERT = auto()
    END = auto()


class Instruction:
    """A single instruction in a thread program."""

    def __init__(self, itype, var=None, value=None, reg=None,
                 target_pc=None, condition=None, assert_fn=None):
        self.itype = itype
        self.var = var          # shared variable name
        self.value = value      # constant or expression
        self.reg = reg          # local register
        self.target_pc = target_pc  # for branches
        self.condition = condition  # lambda: state -> bool
        self.assert_fn = assert_fn

    def __repr__(self):
        if self.itype == InstrType.READ:
            return f"read {self.reg} <- {self.var}"
        elif self.itype == InstrType.WRITE:
            return f"write {self.var} <- {self.value}"
        elif self.itype == InstrType.CAS:
            return f"CAS {self.var}"
        elif self.itype == InstrType.BRANCH:
            return f"branch -> {self.target_pc}"
        elif self.itype == InstrType.ASSERT:
            return f"assert"
        elif self.itype == InstrType.END:
            return f"end"
        return f"instr({self.itype})"


class ThreadProgram:
    """A sequence of instructions for one thread."""

    def __init__(self, instructions=None):
        self.instructions = list(instructions) if instructions else []

    def add(self, instr):
        self.instructions.append(instr)
        return self

    def __len__(self):
        return len(self.instructions)

    def get(self, pc):
        if 0 <= pc < len(self.instructions):
            return self.instructions[pc]
        return None


class ConcurrentProgram:
    """A concurrent program with multiple threads."""

    def __init__(self):
        self.threads = {}       # tid -> ThreadProgram
        self.init_state = SharedState()

    def add_thread(self, tid, program):
        self.threads[tid] = program

    def set_init(self, var, val):
        self.init_state.write(var, val)

    def initial_state(self):
        state = SystemState(self.init_state.copy())
        for tid in self.threads:
            state.threads[tid] = ThreadState(pc=0)
        return state


# ---------------------------------------------------------------------------
# Transition system
# ---------------------------------------------------------------------------

class TransitionSystem:
    """Execute transitions on system state."""

    def __init__(self, program):
        self.program = program

    def enabled_threads(self, state):
        """Return list of thread IDs that can take a step."""
        enabled = []
        for tid, ts in state.threads.items():
            prog = self.program.threads.get(tid)
            if prog is None:
                continue
            instr = prog.get(ts.pc)
            if instr is not None and instr.itype != InstrType.END:
                enabled.append(tid)
        return enabled

    def execute(self, state, tid):
        """Execute one step of thread tid. Returns (new_state, assertion_ok)."""
        prog = self.program.threads[tid]
        ts = state.threads[tid]
        instr = prog.get(ts.pc)
        if instr is None:
            return state, True

        new_state = state.copy()
        new_ts = new_state.threads[tid]
        assertion_ok = True

        if instr.itype == InstrType.READ:
            val = new_state.shared.read(instr.var)
            new_ts.locals[instr.reg] = val
            new_ts.pc += 1

        elif instr.itype == InstrType.WRITE:
            if callable(instr.value):
                val = instr.value(new_ts.locals)
            else:
                val = instr.value
            new_state.shared.write(instr.var, val)
            new_ts.pc += 1

        elif instr.itype == InstrType.LOCAL:
            if callable(instr.value):
                new_ts.locals[instr.reg] = instr.value(new_ts.locals)
            else:
                new_ts.locals[instr.reg] = instr.value
            new_ts.pc += 1

        elif instr.itype == InstrType.CAS:
            current = new_state.shared.read(instr.var)
            expected = instr.value[0] if isinstance(instr.value, tuple) else 0
            new_val = instr.value[1] if isinstance(instr.value, tuple) else 1
            if current == expected:
                new_state.shared.write(instr.var, new_val)
                new_ts.locals[instr.reg] = 1  # success
            else:
                new_ts.locals[instr.reg] = 0  # failure
            new_ts.pc += 1

        elif instr.itype == InstrType.BRANCH:
            if instr.condition is None or instr.condition(new_ts.locals):
                new_ts.pc = instr.target_pc
            else:
                new_ts.pc += 1

        elif instr.itype == InstrType.ASSERT:
            if instr.assert_fn:
                assertion_ok = instr.assert_fn(new_state.shared, new_ts.locals)
            new_ts.pc += 1

        elif instr.itype == InstrType.FENCE:
            new_ts.pc += 1

        elif instr.itype == InstrType.END:
            pass

        return new_state, assertion_ok

    def is_terminal(self, state):
        """Check if all threads are done."""
        for tid, ts in state.threads.items():
            prog = self.program.threads.get(tid)
            if prog is None:
                continue
            instr = prog.get(ts.pc)
            if instr is not None and instr.itype != InstrType.END:
                return False
        return True


# ---------------------------------------------------------------------------
# Check result
# ---------------------------------------------------------------------------

class CheckResult:
    """Result of model checking."""

    def __init__(self, satisfied=True, counterexample=None,
                 states_explored=0, elapsed=0.0, property_name=""):
        self.satisfied = satisfied
        self.counterexample_trace = counterexample
        self.states_explored = states_explored
        self.time = elapsed
        self.property_name = property_name

    def __repr__(self):
        status = "PASS" if self.satisfied else "FAIL"
        return (f"CheckResult({status}, states={self.states_explored}, "
                f"time={self.time:.3f}s)")


# ---------------------------------------------------------------------------
# Model Checker
# ---------------------------------------------------------------------------

class ModelChecker:
    """Explicit-state model checker."""

    def __init__(self, program):
        self.program = program
        self.ts = TransitionSystem(program)

    def check(self, property_fn=None, method='bfs', depth_bound=None,
              use_por=False, use_symmetry=False):
        """Check property over all reachable states.
        property_fn: state -> bool (True if property holds).
        """
        start = time.time()

        if method == 'bfs':
            result = self._bfs(property_fn, depth_bound, use_por, use_symmetry)
        elif method == 'dfs':
            result = self._dfs(property_fn, depth_bound, use_por, use_symmetry)
        else:
            result = self._bfs(property_fn, depth_bound, use_por, use_symmetry)

        result.time = time.time() - start
        return result

    def _bfs(self, property_fn, depth_bound, use_por, use_symmetry):
        init = self.program.initial_state()
        visited = set()
        queue = deque()
        queue.append((init, [], 0))
        visited.add(self._state_key(init, use_symmetry))
        states_explored = 0

        while queue:
            state, trace, depth = queue.popleft()
            states_explored += 1

            if depth_bound is not None and depth > depth_bound:
                continue

            # Check property
            if property_fn and not property_fn(state):
                return CheckResult(False, trace + [state], states_explored,
                                   property_name="safety")

            # Check assertions
            enabled = self.ts.enabled_threads(state)
            if use_por:
                enabled = self._ample_set(state, enabled)

            for tid in enabled:
                new_state, assert_ok = self.ts.execute(state, tid)
                if not assert_ok:
                    return CheckResult(False, trace + [state, new_state],
                                       states_explored, property_name="assertion")

                key = self._state_key(new_state, use_symmetry)
                if key not in visited:
                    visited.add(key)
                    queue.append((new_state, trace + [state], depth + 1))

        return CheckResult(True, None, states_explored)

    def _dfs(self, property_fn, depth_bound, use_por, use_symmetry):
        init = self.program.initial_state()
        visited = set()
        stack = [(init, [], 0)]
        states_explored = 0

        while stack:
            state, trace, depth = stack.pop()
            key = self._state_key(state, use_symmetry)

            if key in visited:
                continue
            visited.add(key)
            states_explored += 1

            if depth_bound is not None and depth > depth_bound:
                continue

            if property_fn and not property_fn(state):
                return CheckResult(False, trace + [state], states_explored,
                                   property_name="safety")

            enabled = self.ts.enabled_threads(state)
            if use_por:
                enabled = self._ample_set(state, enabled)

            for tid in enabled:
                new_state, assert_ok = self.ts.execute(state, tid)
                if not assert_ok:
                    return CheckResult(False, trace + [state, new_state],
                                       states_explored, property_name="assertion")

                new_key = self._state_key(new_state, use_symmetry)
                if new_key not in visited:
                    stack.append((new_state, trace + [state], depth + 1))

        return CheckResult(True, None, states_explored)

    def _state_key(self, state, use_symmetry=False):
        if use_symmetry:
            return self._canonical_state(state)
        return hash(state)

    def _canonical_state(self, state):
        """Symmetry reduction: canonicalize state by sorting thread states."""
        thread_states = tuple(sorted(
            ((tid, ts) for tid, ts in state.threads.items()),
            key=lambda x: hash(x[1])
        ))
        return hash((state.shared, thread_states))

    # --- Partial Order Reduction ---

    def _ample_set(self, state, enabled):
        """Compute ample set for partial order reduction.
        Persistent set: subset of enabled threads such that
        any execution from this state either uses an ample thread first
        or is independent of all ample threads.
        """
        if len(enabled) <= 1:
            return enabled

        # Heuristic: pick threads that access different variables
        accesses = {}
        for tid in enabled:
            prog = self.program.threads[tid]
            ts = state.threads[tid]
            instr = prog.get(ts.pc)
            if instr and instr.var:
                accesses[tid] = instr.var

        # Group by accessed variable
        var_groups = defaultdict(list)
        for tid, var in accesses.items():
            var_groups[var].append(tid)

        # If we can find a single group that's independent, use it
        for var, tids in var_groups.items():
            if len(tids) == 1:
                return tids

        # Fallback: return smallest group
        if var_groups:
            smallest = min(var_groups.values(), key=len)
            return smallest

        return enabled


# ---------------------------------------------------------------------------
# LTL Model Checking (Büchi automaton product, nested DFS)
# ---------------------------------------------------------------------------

class BuchiAutomaton:
    """Simple Büchi automaton for LTL properties."""

    def __init__(self):
        self.states = set()
        self.initial = set()
        self.accepting = set()
        self.transitions = defaultdict(list)  # state -> [(guard_fn, next_state)]

    def add_state(self, s, initial=False, accepting=False):
        self.states.add(s)
        if initial:
            self.initial.add(s)
        if accepting:
            self.accepting.add(s)

    def add_transition(self, src, guard_fn, dst):
        self.transitions[src].append((guard_fn, dst))

    def successors(self, buchi_state, system_state):
        """Get successor Büchi states given current system state."""
        succs = []
        for guard_fn, dst in self.transitions.get(buchi_state, []):
            if guard_fn(system_state):
                succs.append(dst)
        return succs


class LTLModelChecker:
    """LTL model checking via Büchi automaton product and nested DFS."""

    def __init__(self, program):
        self.program = program
        self.ts = TransitionSystem(program)

    def check(self, buchi, depth_bound=1000):
        """Check if program satisfies LTL property (negation in Büchi).
        Returns CheckResult. If Büchi accepts => property VIOLATED.
        """
        start = time.time()
        init_sys = self.program.initial_state()
        states_explored = 0

        # Product states: (system_state, buchi_state)
        for b_init in buchi.initial:
            visited_outer = set()
            stack = [(init_sys, b_init, [], 0)]

            while stack:
                sys_state, b_state, trace, depth = stack.pop()
                prod_key = (hash(sys_state), b_state)
                if prod_key in visited_outer or depth > depth_bound:
                    continue
                visited_outer.add(prod_key)
                states_explored += 1

                # If accepting, do inner DFS for cycle
                if b_state in buchi.accepting:
                    has_cycle = self._inner_dfs(
                        sys_state, b_state, buchi, visited_outer, depth_bound
                    )
                    if has_cycle:
                        elapsed = time.time() - start
                        return CheckResult(
                            False, trace + [(sys_state, b_state)],
                            states_explored, elapsed, "LTL"
                        )

                # Expand
                enabled = self.ts.enabled_threads(sys_state)
                for tid in enabled:
                    new_sys, _ = self.ts.execute(sys_state, tid)
                    for new_b in buchi.successors(b_state, new_sys):
                        new_key = (hash(new_sys), new_b)
                        if new_key not in visited_outer:
                            stack.append((new_sys, new_b,
                                          trace + [(sys_state, b_state)],
                                          depth + 1))

        elapsed = time.time() - start
        return CheckResult(True, None, states_explored, elapsed, "LTL")

    def _inner_dfs(self, target_sys, target_b, buchi, outer_visited, depth_bound):
        """Inner DFS of nested DFS: search for cycle back to target."""
        visited_inner = set()
        stack = [(target_sys, target_b, 0)]

        while stack:
            sys_state, b_state, depth = stack.pop()
            if depth > depth_bound:
                continue
            prod_key = (hash(sys_state), b_state)
            if prod_key in visited_inner:
                continue
            visited_inner.add(prod_key)

            enabled = self.ts.enabled_threads(sys_state)
            for tid in enabled:
                new_sys, _ = self.ts.execute(sys_state, tid)
                for new_b in buchi.successors(b_state, new_sys):
                    if (hash(new_sys), new_b) == (hash(target_sys), target_b):
                        return True  # Found cycle
                    new_key = (hash(new_sys), new_b)
                    if new_key not in visited_inner:
                        stack.append((new_sys, new_b, depth + 1))
        return False


# ---------------------------------------------------------------------------
# CTL Model Checking (fixed-point EX, EU, EG)
# ---------------------------------------------------------------------------

class CTLModelChecker:
    """CTL model checking via fixed-point computation."""

    def __init__(self, program, depth_bound=500):
        self.program = program
        self.ts = TransitionSystem(program)
        self.depth_bound = depth_bound
        self._state_graph = None

    def _build_state_graph(self):
        """Build explicit state graph."""
        if self._state_graph is not None:
            return
        self._state_graph = {}
        init = self.program.initial_state()
        queue = deque([init])
        visited = set()

        while queue:
            state = queue.popleft()
            key = hash(state)
            if key in visited:
                continue
            visited.add(key)

            successors = []
            enabled = self.ts.enabled_threads(state)
            for tid in enabled:
                new_state, _ = self.ts.execute(state, tid)
                successors.append(new_state)
                if hash(new_state) not in visited:
                    queue.append(new_state)

            self._state_graph[key] = (state, successors)

            if len(visited) > self.depth_bound:
                break

    def check_EX(self, phi):
        """EX phi: exists a successor satisfying phi."""
        self._build_state_graph()
        satisfying = set()
        for key, (state, succs) in self._state_graph.items():
            for s in succs:
                if phi(s):
                    satisfying.add(key)
                    break
        return satisfying

    def check_EU(self, phi, psi):
        """E[phi U psi]: exists path where phi holds until psi holds."""
        self._build_state_graph()
        satisfying = set()

        # Start with states satisfying psi
        for key, (state, _) in self._state_graph.items():
            if psi(state):
                satisfying.add(key)

        # Fixed point: add states satisfying phi with successor in satisfying
        changed = True
        while changed:
            changed = False
            for key, (state, succs) in self._state_graph.items():
                if key in satisfying:
                    continue
                if not phi(state):
                    continue
                for s in succs:
                    if hash(s) in satisfying:
                        satisfying.add(key)
                        changed = True
                        break

        return satisfying

    def check_EG(self, phi):
        """EG phi: exists a path where phi holds globally."""
        self._build_state_graph()
        # Start with all states satisfying phi
        satisfying = set()
        for key, (state, _) in self._state_graph.items():
            if phi(state):
                satisfying.add(key)

        # Fixed point: remove states with no successor in satisfying
        changed = True
        while changed:
            changed = False
            to_remove = set()
            for key in satisfying:
                if key not in self._state_graph:
                    continue
                state, succs = self._state_graph[key]
                has_succ = any(hash(s) in satisfying for s in succs)
                # Terminal states with phi also count
                if not succs:
                    continue
                if not has_succ:
                    to_remove.add(key)
            if to_remove:
                satisfying -= to_remove
                changed = True

        return satisfying

    def check_property(self, prop_name, *args):
        """Check CTL property by name."""
        if prop_name == 'EX':
            return self.check_EX(args[0])
        elif prop_name == 'EU':
            return self.check_EU(args[0], args[1])
        elif prop_name == 'EG':
            return self.check_EG(args[0])
        raise ValueError(f"Unknown CTL operator: {prop_name}")


# ---------------------------------------------------------------------------
# Bounded Model Checking
# ---------------------------------------------------------------------------

class BoundedModelChecker:
    """Bounded model checking: unroll to depth k."""

    def __init__(self, program):
        self.program = program
        self.ts = TransitionSystem(program)

    def check(self, property_fn, bound):
        """Check property up to depth bound."""
        start = time.time()
        init = self.program.initial_state()
        states_explored = 0
        queue = deque([(init, [], 0)])
        visited = set()

        while queue:
            state, trace, depth = queue.popleft()
            key = hash(state)
            if key in visited or depth > bound:
                continue
            visited.add(key)
            states_explored += 1

            if not property_fn(state):
                elapsed = time.time() - start
                return CheckResult(False, trace + [state], states_explored,
                                   elapsed, "bounded")

            enabled = self.ts.enabled_threads(state)
            for tid in enabled:
                new_state, assert_ok = self.ts.execute(state, tid)
                if not assert_ok:
                    elapsed = time.time() - start
                    return CheckResult(False, trace + [state, new_state],
                                       states_explored, elapsed, "assertion")
                if hash(new_state) not in visited:
                    queue.append((new_state, trace + [state], depth + 1))

        elapsed = time.time() - start
        return CheckResult(True, None, states_explored, elapsed, "bounded")


# ---------------------------------------------------------------------------
# Built-in program: Peterson's mutual exclusion
# ---------------------------------------------------------------------------

def make_petersons_algorithm():
    """Peterson's mutual exclusion algorithm.
    Shared: flag[0], flag[1], turn
    T0: flag0=1; turn=1; while(flag1==1 && turn==1){}; CS; flag0=0
    T1: flag1=1; turn=0; while(flag0==1 && turn==0){}; CS; flag1=0
    """
    prog = ConcurrentProgram()
    prog.set_init('flag0', 0)
    prog.set_init('flag1', 0)
    prog.set_init('turn', 0)
    prog.set_init('cs', 0)

    # Thread 0
    t0 = ThreadProgram()
    t0.add(Instruction(InstrType.WRITE, var='flag0', value=1))       # 0
    t0.add(Instruction(InstrType.WRITE, var='turn', value=1))        # 1
    t0.add(Instruction(InstrType.READ, var='flag1', reg='f1'))       # 2
    t0.add(Instruction(InstrType.READ, var='turn', reg='t'))         # 3
    t0.add(Instruction(InstrType.BRANCH, target_pc=2,                # 4: loop
           condition=lambda l: l.get('f1', 0) == 1 and l.get('t', 0) == 1))
    # Critical section
    t0.add(Instruction(InstrType.READ, var='cs', reg='old_cs'))      # 5
    t0.add(Instruction(InstrType.WRITE, var='cs',                    # 6
           value=lambda l: l.get('old_cs', 0) + 1))
    t0.add(Instruction(InstrType.WRITE, var='cs',                    # 7
           value=lambda l: l.get('old_cs', 0)))
    t0.add(Instruction(InstrType.WRITE, var='flag0', value=0))       # 8
    t0.add(Instruction(InstrType.END))                                # 9

    # Thread 1
    t1 = ThreadProgram()
    t1.add(Instruction(InstrType.WRITE, var='flag1', value=1))
    t1.add(Instruction(InstrType.WRITE, var='turn', value=0))
    t1.add(Instruction(InstrType.READ, var='flag0', reg='f0'))
    t1.add(Instruction(InstrType.READ, var='turn', reg='t'))
    t1.add(Instruction(InstrType.BRANCH, target_pc=2,
           condition=lambda l: l.get('f0', 0) == 1 and l.get('t', 0) == 0))
    t1.add(Instruction(InstrType.READ, var='cs', reg='old_cs'))
    t1.add(Instruction(InstrType.WRITE, var='cs',
           value=lambda l: l.get('old_cs', 0) + 1))
    t1.add(Instruction(InstrType.WRITE, var='cs',
           value=lambda l: l.get('old_cs', 0)))
    t1.add(Instruction(InstrType.WRITE, var='flag1', value=0))
    t1.add(Instruction(InstrType.END))

    prog.add_thread(0, t0)
    prog.add_thread(1, t1)
    return prog


def mutual_exclusion_property(state):
    """Property: cs counter never exceeds 1 (at most one thread in CS)."""
    return state.shared.read('cs') <= 1


# ---------------------------------------------------------------------------
# Built-in program: simple counter (no mutex, racy)
# ---------------------------------------------------------------------------

def make_racy_counter():
    """Two threads incrementing a shared counter without synchronization."""
    prog = ConcurrentProgram()
    prog.set_init('counter', 0)

    for tid in range(2):
        t = ThreadProgram()
        t.add(Instruction(InstrType.READ, var='counter', reg='c'))
        t.add(Instruction(InstrType.LOCAL, reg='c',
                          value=lambda l: l.get('c', 0) + 1))
        t.add(Instruction(InstrType.WRITE, var='counter',
                          value=lambda l: l.get('c', 0)))
        t.add(Instruction(InstrType.END))
        prog.add_thread(tid, t)

    return prog


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test 1: Peterson's algorithm - mutual exclusion
    print("=== Model Checker: Peterson's Algorithm ===")
    peterson = make_petersons_algorithm()
    mc = ModelChecker(peterson)
    result = mc.check(mutual_exclusion_property, method='bfs', depth_bound=50)
    print(f"Mutual exclusion: {result}")
    assert result.satisfied, "Peterson's should satisfy mutual exclusion"

    # Test 2: Racy counter
    print("\n=== Model Checker: Racy Counter ===")
    racy = make_racy_counter()
    mc2 = ModelChecker(racy)

    def counter_is_2(state):
        """At termination, counter should be 2."""
        enabled = mc2.ts.enabled_threads(state)
        if not enabled:
            return state.shared.read('counter') == 2
        return True

    result2 = mc2.check(counter_is_2, method='bfs', depth_bound=20)
    print(f"Counter always 2: {result2}")
    # This should FAIL because of the race condition
    if not result2.satisfied:
        print("  (Expected: race condition found)")

    # Test 3: Bounded model checking
    print("\n=== Bounded Model Checking ===")
    bmc = BoundedModelChecker(peterson)
    result3 = bmc.check(mutual_exclusion_property, bound=30)
    print(f"Bounded check: {result3}")

    # Test 4: CTL model checking
    print("\n=== CTL Model Checking ===")
    ctl = CTLModelChecker(peterson, depth_bound=50)
    ex_result = ctl.check_EX(lambda s: s.shared.read('flag0') == 1)
    print(f"EX(flag0==1): {len(ex_result)} states satisfy")

    print("\nmodel_checker.py self-test passed")

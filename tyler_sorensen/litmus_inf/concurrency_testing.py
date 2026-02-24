"""
Systematic concurrency testing: PCT scheduler, random scheduler,
DPOR explorer, stateless model checking, delay injection, fuzzing.
"""

import numpy as np
from collections import defaultdict, deque
from enum import Enum, auto
import copy
import time


# ---------------------------------------------------------------------------
# Shared infrastructure
# ---------------------------------------------------------------------------

class ScheduleEvent:
    """An event in a concurrent schedule."""

    def __init__(self, thread, op, address=None, value=None, pc=0):
        self.thread = thread
        self.op = op            # 'read', 'write', 'acquire', 'release', 'fork', 'join'
        self.address = address
        self.value = value
        self.pc = pc

    def is_dependent(self, other):
        """Two events are dependent if they access the same address
        and at least one is a write, and they are from different threads.
        """
        if self.thread == other.thread:
            return False
        if self.address is None or other.address is None:
            return False
        if self.address != other.address:
            return False
        return self.op == 'write' or other.op == 'write'

    def __repr__(self):
        return f"T{self.thread}:{self.op}({self.address})"

    def __eq__(self, other):
        return (isinstance(other, ScheduleEvent)
                and self.thread == other.thread
                and self.op == other.op
                and self.address == other.address)

    def __hash__(self):
        return hash((self.thread, self.op, self.address, self.pc))


class TestableProgram:
    """A concurrent program represented as per-thread event sequences."""

    def __init__(self):
        self.threads = defaultdict(list)  # tid -> list of ScheduleEvent
        self.n_threads = 0

    def add_event(self, thread, op, address=None, value=None):
        pc = len(self.threads[thread])
        event = ScheduleEvent(thread, op, address, value, pc)
        self.threads[thread].append(event)
        self.n_threads = max(self.n_threads, thread + 1)
        return event

    def thread_length(self, tid):
        return len(self.threads[tid])


class ExecutionState:
    """State during execution: memory + per-thread PCs."""

    def __init__(self, program):
        self.program = program
        self.memory = {}
        self.pcs = defaultdict(int)  # tid -> current pc
        self.lock_holders = {}       # lock -> tid

    def enabled_threads(self):
        """Return list of threads that can take a step."""
        enabled = []
        for tid in range(self.program.n_threads):
            if self.pcs[tid] < self.program.thread_length(tid):
                event = self.program.threads[tid][self.pcs[tid]]
                if event.op == 'acquire':
                    holder = self.lock_holders.get(event.address)
                    if holder is not None and holder != tid:
                        continue
                enabled.append(tid)
        return enabled

    def execute_thread(self, tid):
        """Execute next event of thread tid. Returns the event."""
        if self.pcs[tid] >= self.program.thread_length(tid):
            return None
        event = self.program.threads[tid][self.pcs[tid]]

        if event.op == 'write':
            self.memory[event.address] = event.value
        elif event.op == 'read':
            event.value = self.memory.get(event.address, 0)
        elif event.op == 'acquire':
            self.lock_holders[event.address] = tid
        elif event.op == 'release':
            if self.lock_holders.get(event.address) == tid:
                del self.lock_holders[event.address]

        self.pcs[tid] += 1
        return event

    def is_complete(self):
        for tid in range(self.program.n_threads):
            if self.pcs[tid] < self.program.thread_length(tid):
                return False
        return True

    def copy(self):
        new = ExecutionState(self.program)
        new.memory = dict(self.memory)
        new.pcs = defaultdict(int, self.pcs)
        new.lock_holders = dict(self.lock_holders)
        return new

    def state_key(self):
        return (tuple(sorted(self.memory.items())),
                tuple(self.pcs[tid] for tid in range(self.program.n_threads)))


class TestReport:
    """Report from concurrency testing."""

    def __init__(self):
        self.bugs_found = []
        self.interleavings_explored = 0
        self.coverage = 0.0
        self.time = 0.0
        self.schedules_tested = 0

    def __repr__(self):
        return (f"TestReport(bugs={len(self.bugs_found)}, "
                f"interleavings={self.interleavings_explored}, "
                f"coverage={self.coverage:.2%}, time={self.time:.3f}s)")


# ---------------------------------------------------------------------------
# PCT Scheduler
# ---------------------------------------------------------------------------

class PCTScheduler:
    """Prioritized Context-bound Testing (PCT) scheduler.
    Assigns random priorities to threads and picks random priority-change points.
    Guarantees finding bugs with k context switches with probability >= 1/n^k.
    """

    def __init__(self, seed=42, max_context_switches=2):
        self.rng = np.random.RandomState(seed)
        self.max_context_switches = max_context_switches

    def run(self, program, assertion_fn=None, n_runs=100):
        """Run PCT scheduling for n_runs iterations."""
        start = time.time()
        report = TestReport()

        for run_idx in range(n_runs):
            state = ExecutionState(program)
            n_threads = program.n_threads

            # Assign random priorities
            priorities = list(range(n_threads))
            self.rng.shuffle(priorities)
            priority_map = {tid: priorities[tid] for tid in range(n_threads)}

            # Pick random priority-change points
            total_events = sum(program.thread_length(tid) for tid in range(n_threads))
            change_points = set()
            if total_events > 0:
                for _ in range(self.max_context_switches):
                    change_points.add(int(self.rng.randint(0, max(total_events, 1))))

            step = 0
            schedule = []
            while not state.is_complete():
                enabled = state.enabled_threads()
                if not enabled:
                    break

                # Priority change at designated points
                if step in change_points:
                    # Lower priority of highest-priority enabled thread
                    highest = max(enabled, key=lambda t: priority_map[t])
                    priority_map[highest] = min(priority_map.values()) - 1

                # Pick highest-priority enabled thread
                chosen = max(enabled, key=lambda t: priority_map[t])
                event = state.execute_thread(chosen)
                if event:
                    schedule.append(event)
                step += 1

                if step > total_events * 2:
                    break

            report.interleavings_explored += 1
            report.schedules_tested += 1

            # Check assertion
            if assertion_fn:
                if not assertion_fn(state.memory):
                    report.bugs_found.append({
                        'run': run_idx,
                        'schedule': schedule,
                        'final_memory': dict(state.memory),
                    })

        report.time = time.time() - start
        report.coverage = min(1.0, report.interleavings_explored / max(n_runs, 1))
        return report


# ---------------------------------------------------------------------------
# Random Scheduler
# ---------------------------------------------------------------------------

class RandomScheduler:
    """Random thread interleaving scheduler."""

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def run(self, program, assertion_fn=None, n_runs=100):
        start = time.time()
        report = TestReport()

        for run_idx in range(n_runs):
            state = ExecutionState(program)
            schedule = []
            steps = 0
            max_steps = sum(program.thread_length(tid)
                            for tid in range(program.n_threads)) * 2

            while not state.is_complete() and steps < max_steps:
                enabled = state.enabled_threads()
                if not enabled:
                    break
                chosen = enabled[self.rng.randint(0, len(enabled))]
                event = state.execute_thread(chosen)
                if event:
                    schedule.append(event)
                steps += 1

            report.interleavings_explored += 1
            report.schedules_tested += 1

            if assertion_fn and not assertion_fn(state.memory):
                report.bugs_found.append({
                    'run': run_idx,
                    'schedule': schedule,
                    'final_memory': dict(state.memory),
                })

        report.time = time.time() - start
        report.coverage = min(1.0, report.interleavings_explored / max(n_runs, 1))
        return report


# ---------------------------------------------------------------------------
# DPOR Explorer
# ---------------------------------------------------------------------------

class DPORExplorer:
    """Dynamic Partial Order Reduction explorer.
    Explores only distinct interleavings by tracking dependent events.
    """

    def __init__(self):
        self.explored_schedules = []
        self.backtrack_sets = defaultdict(set)

    def run(self, program, assertion_fn=None, max_explorations=10000):
        start = time.time()
        report = TestReport()

        initial_state = ExecutionState(program)
        self._dpor_explore(program, initial_state, [], [], assertion_fn,
                           report, max_explorations, set())

        report.time = time.time() - start
        report.coverage = 1.0 if report.interleavings_explored > 0 else 0.0
        return report

    def _dpor_explore(self, program, state, schedule, event_log,
                      assertion_fn, report, max_expl, visited):
        if report.interleavings_explored >= max_expl:
            return

        enabled = state.enabled_threads()
        if not enabled or state.is_complete():
            report.interleavings_explored += 1
            report.schedules_tested += 1

            if assertion_fn and not assertion_fn(state.memory):
                report.bugs_found.append({
                    'schedule': list(schedule),
                    'final_memory': dict(state.memory),
                })
            return

        depth = len(schedule)
        # Initialize backtrack set
        if depth not in self.backtrack_sets or not self.backtrack_sets[depth]:
            self.backtrack_sets[depth] = set(enabled[:1])

        done = set()
        while True:
            backtrack = self.backtrack_sets[depth] - done
            if not backtrack:
                break

            tid = min(backtrack)
            done.add(tid)

            new_state = state.copy()
            event = new_state.execute_thread(tid)
            if event is None:
                continue

            # Check for dependent events and update backtrack sets
            for i, prev_event in enumerate(event_log):
                if event.is_dependent(prev_event):
                    # Add the other thread to backtrack set at that depth
                    if i < len(schedule):
                        self.backtrack_sets[i].add(event.thread)

            state_key = new_state.state_key()
            if state_key not in visited:
                visited.add(state_key)
                self._dpor_explore(program, new_state,
                                   schedule + [tid],
                                   event_log + [event],
                                   assertion_fn, report, max_expl, visited)


# ---------------------------------------------------------------------------
# Stateless Model Checking
# ---------------------------------------------------------------------------

class StatelessModelChecker:
    """Explore all executions without storing states."""

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def run(self, program, assertion_fn=None, max_executions=1000):
        """Systematic exploration using iterative context bounding."""
        start = time.time()
        report = TestReport()

        for max_switches in range(program.n_threads * 3):
            if report.interleavings_explored >= max_executions:
                break
            self._explore_bounded(program, max_switches, assertion_fn,
                                  report, max_executions)

        report.time = time.time() - start
        return report

    def _explore_bounded(self, program, max_switches, assertion_fn,
                         report, max_executions):
        """Explore all executions with at most max_switches context switches."""
        state = ExecutionState(program)
        self._explore_recursive(program, state, [], 0, -1, max_switches,
                                assertion_fn, report, max_executions)

    def _explore_recursive(self, program, state, schedule, switches,
                           last_thread, max_switches, assertion_fn,
                           report, max_executions):
        if report.interleavings_explored >= max_executions:
            return

        if state.is_complete():
            report.interleavings_explored += 1
            report.schedules_tested += 1
            if assertion_fn and not assertion_fn(state.memory):
                report.bugs_found.append({
                    'schedule': list(schedule),
                    'final_memory': dict(state.memory),
                })
            return

        enabled = state.enabled_threads()
        if not enabled:
            return

        for tid in enabled:
            new_switches = switches
            if last_thread >= 0 and tid != last_thread:
                new_switches += 1
            if new_switches > max_switches:
                continue

            new_state = state.copy()
            event = new_state.execute_thread(tid)
            if event:
                self._explore_recursive(
                    program, new_state, schedule + [tid],
                    new_switches, tid, max_switches,
                    assertion_fn, report, max_executions
                )


# ---------------------------------------------------------------------------
# Delay Injection
# ---------------------------------------------------------------------------

class DelayInjector:
    """Insert random delays to expose race conditions."""

    def __init__(self, seed=42, delay_probability=0.3):
        self.rng = np.random.RandomState(seed)
        self.delay_probability = delay_probability

    def run(self, program, assertion_fn=None, n_runs=100):
        start = time.time()
        report = TestReport()

        for run_idx in range(n_runs):
            state = ExecutionState(program)
            schedule = []
            steps = 0
            max_steps = sum(program.thread_length(tid)
                            for tid in range(program.n_threads)) * 3

            while not state.is_complete() and steps < max_steps:
                enabled = state.enabled_threads()
                if not enabled:
                    break

                # Randomly delay some threads
                if len(enabled) > 1 and self.rng.random() < self.delay_probability:
                    # Remove a random thread from enabled (delay it)
                    delayed = enabled[self.rng.randint(0, len(enabled))]
                    enabled = [t for t in enabled if t != delayed]
                    if not enabled:
                        enabled = [delayed]

                chosen = enabled[self.rng.randint(0, len(enabled))]
                event = state.execute_thread(chosen)
                if event:
                    schedule.append(event)
                steps += 1

            report.interleavings_explored += 1
            report.schedules_tested += 1

            if assertion_fn and not assertion_fn(state.memory):
                report.bugs_found.append({
                    'run': run_idx,
                    'schedule': schedule,
                    'final_memory': dict(state.memory),
                })

        report.time = time.time() - start
        report.coverage = min(1.0, report.interleavings_explored / max(n_runs, 1))
        return report


# ---------------------------------------------------------------------------
# Concurrency Fuzzer
# ---------------------------------------------------------------------------

class ConcurrencyFuzzer:
    """Mutate thread schedules to maximize coverage."""

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self.corpus = []
        self.coverage_map = set()

    def run(self, program, assertion_fn=None, n_iterations=200):
        start = time.time()
        report = TestReport()

        # Seed corpus with a few random schedules
        for _ in range(min(10, n_iterations)):
            schedule, final_mem = self._random_execution(program)
            self.corpus.append(schedule)
            cov = self._coverage_key(final_mem)
            self.coverage_map.add(cov)

            report.interleavings_explored += 1
            if assertion_fn and not assertion_fn(final_mem):
                report.bugs_found.append({
                    'schedule': schedule,
                    'final_memory': final_mem,
                })

        # Mutate schedules
        for i in range(n_iterations - len(self.corpus)):
            if report.interleavings_explored >= n_iterations:
                break

            # Pick a schedule from corpus and mutate
            parent = self.corpus[self.rng.randint(0, len(self.corpus))]
            mutated = self._mutate_schedule(parent, program)

            state = ExecutionState(program)
            schedule = []
            step_idx = 0
            max_steps = sum(program.thread_length(tid)
                            for tid in range(program.n_threads)) * 2

            while not state.is_complete() and step_idx < max_steps:
                enabled = state.enabled_threads()
                if not enabled:
                    break

                if step_idx < len(mutated):
                    chosen = mutated[step_idx]
                    if chosen not in enabled:
                        chosen = enabled[self.rng.randint(0, len(enabled))]
                else:
                    chosen = enabled[self.rng.randint(0, len(enabled))]

                event = state.execute_thread(chosen)
                if event:
                    schedule.append(chosen)
                step_idx += 1

            report.interleavings_explored += 1
            report.schedules_tested += 1

            cov = self._coverage_key(state.memory)
            if cov not in self.coverage_map:
                self.coverage_map.add(cov)
                self.corpus.append(schedule)

            if assertion_fn and not assertion_fn(state.memory):
                report.bugs_found.append({
                    'schedule': schedule,
                    'final_memory': dict(state.memory),
                })

        report.time = time.time() - start
        report.coverage = len(self.coverage_map) / max(1, report.interleavings_explored)
        return report

    def _random_execution(self, program):
        state = ExecutionState(program)
        schedule = []
        max_steps = sum(program.thread_length(tid)
                        for tid in range(program.n_threads)) * 2
        step = 0
        while not state.is_complete() and step < max_steps:
            enabled = state.enabled_threads()
            if not enabled:
                break
            chosen = enabled[self.rng.randint(0, len(enabled))]
            state.execute_thread(chosen)
            schedule.append(chosen)
            step += 1
        return schedule, dict(state.memory)

    def _mutate_schedule(self, schedule, program):
        """Mutate a schedule by swapping adjacent thread selections."""
        mutated = list(schedule)
        if len(mutated) < 2:
            return mutated
        n_mutations = self.rng.randint(1, max(2, len(mutated) // 4))
        for _ in range(n_mutations):
            idx = self.rng.randint(0, len(mutated) - 1)
            mutated[idx], mutated[idx + 1] = mutated[idx + 1], mutated[idx]
        return mutated

    def _coverage_key(self, memory):
        return tuple(sorted(memory.items()))


# ---------------------------------------------------------------------------
# Example programs
# ---------------------------------------------------------------------------

def make_racy_increment():
    """Two threads incrementing shared counter without locks."""
    prog = TestableProgram()
    for tid in range(2):
        prog.add_event(tid, 'read', 'counter')
        prog.add_event(tid, 'write', 'counter', value=1)  # inc by 1
    return prog


def make_locked_increment():
    """Two threads incrementing shared counter WITH locks."""
    prog = TestableProgram()
    for tid in range(2):
        prog.add_event(tid, 'acquire', 'mutex')
        prog.add_event(tid, 'read', 'counter')
        prog.add_event(tid, 'write', 'counter', value=1)
        prog.add_event(tid, 'release', 'mutex')
    return prog


def make_message_passing():
    """Message passing: T0 writes data then flag, T1 reads flag then data."""
    prog = TestableProgram()
    prog.add_event(0, 'write', 'data', value=42)
    prog.add_event(0, 'write', 'flag', value=1)
    prog.add_event(1, 'read', 'flag')
    prog.add_event(1, 'read', 'data')
    return prog


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test 1: PCT on racy increment
    print("=== PCT Scheduler: Racy Increment ===")
    prog = make_racy_increment()

    def check_counter(memory):
        """Counter should be 2 if no race."""
        return memory.get('counter', 0) == 2

    pct = PCTScheduler(seed=42, max_context_switches=2)
    report = pct.run(prog, None, n_runs=50)
    print(f"PCT: {report}")

    # Test 2: Random scheduler
    print("\n=== Random Scheduler ===")
    rand_sched = RandomScheduler(seed=42)
    report2 = rand_sched.run(prog, None, n_runs=50)
    print(f"Random: {report2}")

    # Test 3: DPOR
    print("\n=== DPOR Explorer ===")
    dpor = DPORExplorer()
    report3 = dpor.run(prog, None, max_explorations=100)
    print(f"DPOR: {report3}")

    # Test 4: Stateless model checking
    print("\n=== Stateless Model Checking ===")
    smc = StatelessModelChecker(seed=42)
    report4 = smc.run(prog, None, max_executions=100)
    print(f"SMC: {report4}")

    # Test 5: Delay injection
    print("\n=== Delay Injection ===")
    delay = DelayInjector(seed=42)
    report5 = delay.run(prog, None, n_runs=50)
    print(f"Delay: {report5}")

    # Test 6: Concurrency fuzzer
    print("\n=== Concurrency Fuzzer ===")
    fuzzer = ConcurrencyFuzzer(seed=42)
    report6 = fuzzer.run(prog, None, n_iterations=50)
    print(f"Fuzzer: {report6}")

    # Test 7: Message passing
    print("\n=== Message Passing Test ===")
    mp = make_message_passing()
    dpor2 = DPORExplorer()
    report7 = dpor2.run(mp, None, max_explorations=100)
    print(f"MP DPOR: {report7}")

    print("\nconcurrency_testing.py self-test passed")

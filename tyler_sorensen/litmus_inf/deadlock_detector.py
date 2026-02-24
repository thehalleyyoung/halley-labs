"""
Deadlock and livelock detection: lock order graphs, resource allocation graphs,
Coffman conditions, timeout detection, livelock/priority inversion detection.
"""

import numpy as np
from collections import defaultdict, deque
from enum import Enum, auto
import copy


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

class LockEventType(Enum):
    ACQUIRE = auto()
    RELEASE = auto()
    TRY_ACQUIRE = auto()
    WAIT = auto()


class LockEvent:
    """A lock acquisition/release event."""
    _counter = 0

    def __init__(self, thread, lock, event_type, timestamp=0, priority=0):
        LockEvent._counter += 1
        self.eid = LockEvent._counter
        self.thread = thread
        self.lock = lock
        self.event_type = event_type
        self.timestamp = timestamp
        self.priority = priority

    def __repr__(self):
        return f"T{self.thread}:{self.event_type.name}({self.lock})@t={self.timestamp}"


class Deadlock:
    """Represents a detected deadlock."""

    def __init__(self, threads, locks, cycle, suggested_fix=""):
        self.threads = threads
        self.locks = locks
        self.cycle = cycle          # list of (thread, lock) showing the cycle
        self.suggested_fix = suggested_fix

    def __repr__(self):
        cycle_str = " -> ".join(f"T{t}:L{l}" for t, l in self.cycle)
        return f"Deadlock({cycle_str})"

    def __eq__(self, other):
        if not isinstance(other, Deadlock):
            return False
        return set(self.threads) == set(other.threads) and set(self.locks) == set(other.locks)

    def __hash__(self):
        return hash((frozenset(self.threads), frozenset(self.locks)))


class Livelock:
    """Represents a detected livelock."""

    def __init__(self, threads, pattern, cycle_length):
        self.threads = threads
        self.pattern = pattern
        self.cycle_length = cycle_length

    def __repr__(self):
        return f"Livelock(threads={self.threads}, cycle_len={self.cycle_length})"


class PriorityInversion:
    """Detected priority inversion."""

    def __init__(self, high_thread, low_thread, lock, medium_threads=None):
        self.high_thread = high_thread
        self.low_thread = low_thread
        self.lock = lock
        self.medium_threads = medium_threads or []

    def __repr__(self):
        return (f"PriorityInversion(high=T{self.high_thread}, "
                f"low=T{self.low_thread}, lock={self.lock})")


# ---------------------------------------------------------------------------
# Lock Order Graph
# ---------------------------------------------------------------------------

class LockOrderGraph:
    """Directed graph where edge (A, B) means lock A was held when B was acquired.
    A cycle indicates potential deadlock.
    """

    def __init__(self):
        self.edges = defaultdict(set)     # lock -> set of locks acquired while holding
        self.edge_info = {}               # (lock_a, lock_b) -> (thread, timestamp)
        self.n_edges = 0

    def add_edge(self, held_lock, acquired_lock, thread=None, timestamp=0):
        if acquired_lock not in self.edges[held_lock]:
            self.edges[held_lock].add(acquired_lock)
            self.edge_info[(held_lock, acquired_lock)] = (thread, timestamp)
            self.n_edges += 1

    def find_cycles(self):
        """Find all simple cycles in the lock order graph using DFS."""
        cycles = []
        all_nodes = set(self.edges.keys())
        for targets in self.edges.values():
            all_nodes |= targets

        visited = set()
        rec_stack = set()

        def dfs(node, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self.edges.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, path)
                elif neighbor in rec_stack:
                    # Found cycle
                    idx = path.index(neighbor)
                    cycle = list(path[idx:])
                    cycles.append(cycle)

            path.pop()
            rec_stack.discard(node)

        for node in sorted(all_nodes):
            if node not in visited:
                dfs(node, [])

        return cycles

    def detect_deadlock_potential(self):
        """Return list of Deadlock objects for each cycle found."""
        cycles = self.find_cycles()
        deadlocks = []
        for cycle in cycles:
            threads = set()
            locks = set(cycle)
            cycle_pairs = []
            for i in range(len(cycle)):
                held = cycle[i]
                acquired = cycle[(i + 1) % len(cycle)]
                info = self.edge_info.get((held, acquired))
                tid = info[0] if info else -1
                threads.add(tid)
                cycle_pairs.append((tid, held))

            fix = self._suggest_fix(cycle)
            dl = Deadlock(list(threads), list(locks), cycle_pairs, fix)
            deadlocks.append(dl)
        return deadlocks

    def _suggest_fix(self, cycle):
        """Suggest a fix for the deadlock cycle."""
        sorted_locks = sorted(cycle)
        return f"Enforce lock ordering: {' < '.join(str(l) for l in sorted_locks)}"

    def to_adjacency_matrix(self):
        """Convert to numpy adjacency matrix."""
        all_nodes = set(self.edges.keys())
        for targets in self.edges.values():
            all_nodes |= targets
        nodes = sorted(all_nodes)
        n = len(nodes)
        idx = {node: i for i, node in enumerate(nodes)}
        mat = np.zeros((n, n), dtype=int)
        for src, dsts in self.edges.items():
            for dst in dsts:
                mat[idx[src], idx[dst]] = 1
        return mat, nodes


# ---------------------------------------------------------------------------
# Resource Allocation Graph (Wait-For Graph)
# ---------------------------------------------------------------------------

class ResourceAllocationGraph:
    """Wait-for graph: thread -> lock (request), lock -> thread (assignment).
    Cycle in the wait-for subgraph => deadlock.
    """

    def __init__(self):
        self.assignments = {}       # lock -> thread (who holds it)
        self.requests = defaultdict(set)  # thread -> set of locks requested
        self.held_by_thread = defaultdict(set)  # thread -> set of locks held

    def assign(self, thread, lock):
        """Thread acquires lock."""
        self.assignments[lock] = thread
        self.held_by_thread[thread].add(lock)
        self.requests[thread].discard(lock)

    def request(self, thread, lock):
        """Thread requests lock (blocked)."""
        self.requests[thread].add(lock)

    def release(self, thread, lock):
        """Thread releases lock."""
        if self.assignments.get(lock) == thread:
            del self.assignments[lock]
        self.held_by_thread[thread].discard(lock)

    def build_wait_for_graph(self):
        """Build thread-to-thread wait-for graph.
        Edge (T1, T2) if T1 is waiting for a lock held by T2.
        """
        wfg = defaultdict(set)
        for thread, requested_locks in self.requests.items():
            for lock in requested_locks:
                holder = self.assignments.get(lock)
                if holder is not None and holder != thread:
                    wfg[thread].add(holder)
        return wfg

    def detect_deadlock(self):
        """Detect deadlock by finding cycle in wait-for graph."""
        wfg = self.build_wait_for_graph()
        cycles = self._find_cycles_in_wfg(wfg)
        deadlocks = []
        for cycle in cycles:
            threads = list(cycle)
            locks = []
            cycle_pairs = []
            for i in range(len(cycle)):
                t1 = cycle[i]
                t2 = cycle[(i + 1) % len(cycle)]
                for lock in self.requests.get(t1, set()):
                    if self.assignments.get(lock) == t2:
                        locks.append(lock)
                        cycle_pairs.append((t1, lock))
                        break
            dl = Deadlock(threads, locks, cycle_pairs,
                          f"Break cycle by releasing one lock")
            deadlocks.append(dl)
        return deadlocks

    def _find_cycles_in_wfg(self, wfg):
        """Find cycles in wait-for graph."""
        cycles = []
        visited = set()
        rec_stack = set()

        all_nodes = set(wfg.keys())
        for targets in wfg.values():
            all_nodes |= targets

        def dfs(node, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in wfg.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, path)
                elif neighbor in rec_stack:
                    idx = path.index(neighbor)
                    cycles.append(list(path[idx:]))

            path.pop()
            rec_stack.discard(node)

        for node in sorted(all_nodes):
            if node not in visited:
                dfs(node, [])

        return cycles


# ---------------------------------------------------------------------------
# Coffman Conditions Checker
# ---------------------------------------------------------------------------

class CoffmanConditionsChecker:
    """Check the four Coffman conditions for deadlock."""

    def __init__(self):
        self.locks_shared = {}              # lock -> bool (True if shareable)
        self.thread_held_locks = defaultdict(set)  # thread -> held locks
        self.thread_waiting = {}            # thread -> lock waiting for
        self.preemptable_locks = set()      # locks that can be preempted

    def set_lock_shared(self, lock, shared=False):
        self.locks_shared[lock] = shared

    def set_lock_preemptable(self, lock, preemptable=False):
        self.preemptable_locks.add(lock) if preemptable else self.preemptable_locks.discard(lock)

    def update_state(self, thread, held_locks, waiting_for=None):
        self.thread_held_locks[thread] = set(held_locks)
        if waiting_for is not None:
            self.thread_waiting[thread] = waiting_for
        elif thread in self.thread_waiting:
            del self.thread_waiting[thread]

    def check(self):
        """Check all four Coffman conditions. Returns dict."""
        results = {
            'mutual_exclusion': self._check_mutual_exclusion(),
            'hold_and_wait': self._check_hold_and_wait(),
            'no_preemption': self._check_no_preemption(),
            'circular_wait': self._check_circular_wait(),
        }
        results['all_conditions_met'] = all(results.values())
        results['deadlock_possible'] = results['all_conditions_met']
        return results

    def _check_mutual_exclusion(self):
        """At least one resource is non-shareable."""
        for lock, shared in self.locks_shared.items():
            if not shared:
                return True
        # If no locks marked, assume non-shareable
        return len(self.locks_shared) == 0 or any(not v for v in self.locks_shared.values())

    def _check_hold_and_wait(self):
        """At least one thread holds a lock and is waiting for another."""
        for thread, waiting_lock in self.thread_waiting.items():
            if self.thread_held_locks[thread]:
                return True
        return False

    def _check_no_preemption(self):
        """Resources cannot be preempted."""
        all_held = set()
        for locks in self.thread_held_locks.values():
            all_held |= locks
        for lock in all_held:
            if lock not in self.preemptable_locks:
                return True
        return False

    def _check_circular_wait(self):
        """There exists a circular chain of threads waiting for each other."""
        # Build wait-for graph
        wfg = {}
        for thread, waiting_lock in self.thread_waiting.items():
            for other_thread, held_locks in self.thread_held_locks.items():
                if other_thread != thread and waiting_lock in held_locks:
                    wfg[thread] = other_thread
                    break

        # DFS for cycle
        visited = set()
        for start in wfg:
            path = []
            node = start
            seen_in_path = set()
            while node is not None and node not in seen_in_path:
                seen_in_path.add(node)
                path.append(node)
                node = wfg.get(node)
            if node is not None and node in seen_in_path:
                return True
        return False


# ---------------------------------------------------------------------------
# Deadlock Detector (main class)
# ---------------------------------------------------------------------------

class DeadlockDetector:
    """Main deadlock detector combining multiple approaches."""

    def __init__(self):
        self.lock_graph = LockOrderGraph()
        self.rag = ResourceAllocationGraph()
        self.coffman = CoffmanConditionsChecker()
        self.thread_held = defaultdict(list)  # tid -> stack of held locks
        self.events = []
        self.deadlocks = []

    def add_lock_event(self, thread, lock, action, timestamp=0, priority=0):
        """Process a lock event.
        action: 'acquire' or 'release'
        """
        event = LockEvent(thread, lock,
                          LockEventType.ACQUIRE if action == 'acquire'
                          else LockEventType.RELEASE,
                          timestamp, priority)
        self.events.append(event)

        if action == 'acquire':
            # Update lock order graph: add edges from all currently held locks
            for held_lock in self.thread_held[thread]:
                self.lock_graph.add_edge(held_lock, lock, thread, timestamp)
            self.thread_held[thread].append(lock)
            self.rag.assign(thread, lock)

            # Update Coffman checker
            self.coffman.update_state(thread, self.thread_held[thread])

        elif action == 'release':
            if lock in self.thread_held[thread]:
                self.thread_held[thread].remove(lock)
            self.rag.release(thread, lock)
            self.coffman.update_state(thread, self.thread_held[thread])

        return event

    def add_wait_event(self, thread, lock, timestamp=0):
        """Thread is blocked waiting for lock."""
        self.rag.request(thread, lock)
        self.coffman.update_state(thread, self.thread_held[thread], lock)

    def check(self):
        """Check for potential deadlocks. Returns list of Deadlock objects."""
        # Method 1: lock order graph cycles
        potential = self.lock_graph.detect_deadlock_potential()

        # Method 2: wait-for graph cycles (runtime deadlocks)
        runtime = self.rag.detect_deadlock()

        # Method 3: Coffman conditions
        coffman_result = self.coffman.check()

        all_deadlocks = list(set(potential + runtime))
        self.deadlocks = all_deadlocks

        return all_deadlocks

    def check_coffman(self):
        """Check Coffman conditions only."""
        return self.coffman.check()

    def get_lock_order(self):
        """Get the current lock order graph as adjacency matrix."""
        return self.lock_graph.to_adjacency_matrix()


# ---------------------------------------------------------------------------
# Timeout-based deadlock detection
# ---------------------------------------------------------------------------

class TimeoutDeadlockDetector:
    """Runtime deadlock detection based on timeouts."""

    def __init__(self, timeout_threshold=1000):
        self.timeout_threshold = timeout_threshold
        self.wait_start = {}    # (thread, lock) -> timestamp
        self.detected = []

    def start_wait(self, thread, lock, timestamp):
        self.wait_start[(thread, lock)] = timestamp

    def end_wait(self, thread, lock):
        self.wait_start.pop((thread, lock), None)

    def check_timeouts(self, current_timestamp):
        """Check if any thread has been waiting too long."""
        timed_out = []
        for (thread, lock), start_ts in self.wait_start.items():
            if current_timestamp - start_ts > self.timeout_threshold:
                timed_out.append({
                    'thread': thread,
                    'lock': lock,
                    'wait_duration': current_timestamp - start_ts,
                    'suspected_deadlock': True,
                })
        self.detected.extend(timed_out)
        return timed_out


# ---------------------------------------------------------------------------
# Livelock detection
# ---------------------------------------------------------------------------

class LivelockDetector:
    """Detect livelocks by identifying repeated state patterns."""

    def __init__(self, window_size=100, min_repetitions=3):
        self.window_size = window_size
        self.min_repetitions = min_repetitions
        self.state_history = []     # list of state snapshots
        self.detected = []

    def record_state(self, state):
        """Record a system state snapshot.
        state: hashable representation of system state.
        """
        self.state_history.append(state)

    def detect(self):
        """Detect repeated patterns in state history."""
        livelocks = []
        history = self.state_history
        n = len(history)

        # Try different cycle lengths
        for cycle_len in range(1, min(self.window_size, n // self.min_repetitions + 1)):
            for start in range(n - cycle_len * self.min_repetitions + 1):
                pattern = tuple(history[start:start + cycle_len])
                reps = 0
                for k in range(self.min_repetitions):
                    offset = start + k * cycle_len
                    if offset + cycle_len > n:
                        break
                    window = tuple(history[offset:offset + cycle_len])
                    if window == pattern:
                        reps += 1
                    else:
                        break

                if reps >= self.min_repetitions:
                    ll = Livelock(
                        threads=[],
                        pattern=list(pattern),
                        cycle_length=cycle_len,
                    )
                    livelocks.append(ll)
                    break
            if livelocks:
                break

        self.detected = livelocks
        return livelocks

    def detect_from_events(self, events, state_fn):
        """Detect livelocks from event trace using state extraction function."""
        for event in events:
            state = state_fn(event)
            self.record_state(state)
        return self.detect()


# ---------------------------------------------------------------------------
# Priority inversion detection
# ---------------------------------------------------------------------------

class PriorityInversionDetector:
    """Detect priority inversion: high-priority thread waits for lock
    held by low-priority thread, while medium-priority thread runs.
    """

    def __init__(self):
        self.thread_priorities = {}     # thread -> priority (higher = more important)
        self.lock_holders = {}          # lock -> thread
        self.waiting_threads = {}       # thread -> lock
        self.running_threads = set()
        self.detected = []

    def set_priority(self, thread, priority):
        self.thread_priorities[thread] = priority

    def acquire_lock(self, thread, lock):
        self.lock_holders[lock] = thread
        self.waiting_threads.pop(thread, None)

    def release_lock(self, thread, lock):
        if self.lock_holders.get(lock) == thread:
            del self.lock_holders[lock]

    def wait_for_lock(self, thread, lock):
        self.waiting_threads[thread] = lock

    def set_running(self, thread, running=True):
        if running:
            self.running_threads.add(thread)
        else:
            self.running_threads.discard(thread)

    def detect(self):
        """Detect priority inversions."""
        inversions = []
        for high_thread, lock in self.waiting_threads.items():
            high_pri = self.thread_priorities.get(high_thread, 0)
            holder = self.lock_holders.get(lock)
            if holder is None:
                continue
            low_pri = self.thread_priorities.get(holder, 0)
            if low_pri >= high_pri:
                continue
            # High-priority waiting for low-priority holder
            # Check for medium-priority threads running
            medium_threads = []
            for running in self.running_threads:
                if running == high_thread or running == holder:
                    continue
                med_pri = self.thread_priorities.get(running, 0)
                if low_pri < med_pri < high_pri:
                    medium_threads.append(running)

            inv = PriorityInversion(high_thread, holder, lock, medium_threads)
            inversions.append(inv)

        self.detected = inversions
        return inversions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    LockEvent._counter = 0

    # Test 1: Simple deadlock (AB-BA)
    print("=== Lock Order Graph Deadlock Detection ===")
    det = DeadlockDetector()
    # T0: acquire A, acquire B
    det.add_lock_event(0, 'A', 'acquire', 1)
    det.add_lock_event(0, 'B', 'acquire', 2)
    det.add_lock_event(0, 'B', 'release', 3)
    det.add_lock_event(0, 'A', 'release', 4)
    # T1: acquire B, acquire A (opposite order -> deadlock potential)
    det.add_lock_event(1, 'B', 'acquire', 5)
    det.add_lock_event(1, 'A', 'acquire', 6)
    det.add_lock_event(1, 'A', 'release', 7)
    det.add_lock_event(1, 'B', 'release', 8)

    deadlocks = det.check()
    print(f"Deadlocks found: {len(deadlocks)}")
    for dl in deadlocks:
        print(f"  {dl}")
        print(f"  Fix: {dl.suggested_fix}")
    assert len(deadlocks) > 0, "Should detect AB-BA deadlock"

    # Test 2: No deadlock (consistent ordering)
    print("\n=== No Deadlock (consistent ordering) ===")
    det2 = DeadlockDetector()
    det2.add_lock_event(0, 'A', 'acquire', 1)
    det2.add_lock_event(0, 'B', 'acquire', 2)
    det2.add_lock_event(0, 'B', 'release', 3)
    det2.add_lock_event(0, 'A', 'release', 4)
    det2.add_lock_event(1, 'A', 'acquire', 5)
    det2.add_lock_event(1, 'B', 'acquire', 6)
    det2.add_lock_event(1, 'B', 'release', 7)
    det2.add_lock_event(1, 'A', 'release', 8)
    deadlocks2 = det2.check()
    print(f"Deadlocks found: {len(deadlocks2)}")
    assert len(deadlocks2) == 0, "Consistent ordering should not deadlock"

    # Test 3: Wait-for graph deadlock
    print("\n=== Wait-For Graph Deadlock ===")
    det3 = DeadlockDetector()
    det3.add_lock_event(0, 'A', 'acquire', 1)
    det3.add_lock_event(1, 'B', 'acquire', 2)
    det3.add_wait_event(0, 'B', 3)
    det3.add_wait_event(1, 'A', 4)
    runtime_dls = det3.rag.detect_deadlock()
    print(f"Runtime deadlocks: {len(runtime_dls)}")
    for dl in runtime_dls:
        print(f"  {dl}")
    assert len(runtime_dls) > 0, "Should detect wait-for deadlock"

    # Test 4: Livelock
    print("\n=== Livelock Detection ===")
    ll_det = LivelockDetector(window_size=20, min_repetitions=3)
    # Simulate repeated pattern
    for _ in range(5):
        ll_det.record_state('try_A')
        ll_det.record_state('fail_A')
        ll_det.record_state('backoff')
    livelocks = ll_det.detect()
    print(f"Livelocks found: {len(livelocks)}")
    assert len(livelocks) > 0, "Should detect repeated pattern"

    # Test 5: Priority inversion
    print("\n=== Priority Inversion Detection ===")
    pi_det = PriorityInversionDetector()
    pi_det.set_priority(0, 10)   # high
    pi_det.set_priority(1, 1)    # low
    pi_det.set_priority(2, 5)    # medium
    pi_det.acquire_lock(1, 'mutex')
    pi_det.wait_for_lock(0, 'mutex')
    pi_det.set_running(2, True)
    inversions = pi_det.detect()
    print(f"Priority inversions: {len(inversions)}")
    for inv in inversions:
        print(f"  {inv}")
    assert len(inversions) > 0, "Should detect priority inversion"

    print("\ndeadlock_detector.py self-test passed")

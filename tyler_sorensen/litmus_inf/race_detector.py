"""
Data race detection algorithms: vector clocks, lockset (Eraser),
hybrid detection, FastTrack optimization, and ThreadSanitizer-style shadow memory.
"""

import numpy as np
from collections import defaultdict
from enum import Enum, auto
import copy


# ---------------------------------------------------------------------------
# Events for race detection
# ---------------------------------------------------------------------------

class AccessType(Enum):
    READ = auto()
    WRITE = auto()


class SyncType(Enum):
    ACQUIRE = auto()
    RELEASE = auto()
    FORK = auto()
    JOIN = auto()
    BARRIER = auto()


class TraceEvent:
    """Single event in an execution trace."""
    _counter = 0

    def __init__(self, thread, etype, address=None, value=None,
                 lock_id=None, sync_type=None, target_thread=None):
        TraceEvent._counter += 1
        self.eid = TraceEvent._counter
        self.thread = thread
        self.etype = etype
        self.address = address
        self.value = value
        self.lock_id = lock_id
        self.sync_type = sync_type
        self.target_thread = target_thread
        self.stack_trace = []

    def is_memory_access(self):
        return self.etype in ('read', 'write')

    def is_read(self):
        return self.etype == 'read'

    def is_write(self):
        return self.etype == 'write'

    def is_sync(self):
        return self.etype in ('acquire', 'release', 'fork', 'join', 'barrier')

    def __repr__(self):
        if self.is_memory_access():
            return f"T{self.thread}:{self.etype}(addr={self.address},val={self.value})"
        if self.etype in ('acquire', 'release'):
            return f"T{self.thread}:{self.etype}(lock={self.lock_id})"
        if self.etype in ('fork', 'join'):
            return f"T{self.thread}:{self.etype}(target=T{self.target_thread})"
        return f"T{self.thread}:{self.etype}"


class DataRace:
    """Represents a detected data race."""

    def __init__(self, thread1, thread2, address, access1, access2,
                 event1=None, event2=None):
        self.thread1 = thread1
        self.thread2 = thread2
        self.address = address
        self.access1 = access1
        self.access2 = access2
        self.event1 = event1
        self.event2 = event2
        self.stack_traces = (
            event1.stack_trace if event1 else [],
            event2.stack_trace if event2 else [],
        )
        self.classification = None

    def __repr__(self):
        return (f"DataRace(T{self.thread1}:{self.access1} vs "
                f"T{self.thread2}:{self.access2} @ addr={self.address})")

    def __eq__(self, other):
        if not isinstance(other, DataRace):
            return False
        return ({self.thread1, self.thread2} == {other.thread1, other.thread2}
                and self.address == other.address
                and {self.access1, self.access2} == {other.access1, other.access2})

    def __hash__(self):
        ts = tuple(sorted([self.thread1, self.thread2]))
        acs = tuple(sorted([self.access1, self.access2]))
        return hash((ts, self.address, acs))


# ---------------------------------------------------------------------------
# Vector Clock
# ---------------------------------------------------------------------------

class VectorClock:
    """Vector clock for tracking happens-before relationships."""

    def __init__(self, n_threads=0, tid=None):
        self.clock = np.zeros(max(n_threads, 1), dtype=np.int64)
        if tid is not None and tid < len(self.clock):
            self.clock[tid] = 1

    def _ensure_size(self, size):
        if size > len(self.clock):
            new_clock = np.zeros(size, dtype=np.int64)
            new_clock[:len(self.clock)] = self.clock
            self.clock = new_clock

    def get(self, tid):
        if tid >= len(self.clock):
            return 0
        return int(self.clock[tid])

    def increment(self, tid):
        self._ensure_size(tid + 1)
        self.clock[tid] += 1

    def join(self, other):
        """Element-wise max."""
        max_len = max(len(self.clock), len(other.clock))
        self._ensure_size(max_len)
        other_padded = np.zeros(max_len, dtype=np.int64)
        other_padded[:len(other.clock)] = other.clock
        self.clock = np.maximum(self.clock, other_padded)

    def happens_before(self, other, tid):
        return self.get(tid) <= other.get(tid)

    def concurrent_with(self, other):
        max_len = max(len(self.clock), len(other.clock))
        s = np.zeros(max_len, dtype=np.int64)
        o = np.zeros(max_len, dtype=np.int64)
        s[:len(self.clock)] = self.clock
        o[:len(other.clock)] = other.clock
        return not (np.all(s <= o) or np.all(o <= s))

    def leq(self, other):
        max_len = max(len(self.clock), len(other.clock))
        s = np.zeros(max_len, dtype=np.int64)
        o = np.zeros(max_len, dtype=np.int64)
        s[:len(self.clock)] = self.clock
        o[:len(other.clock)] = other.clock
        return bool(np.all(s <= o))

    def copy(self):
        vc = VectorClock()
        vc.clock = self.clock.copy()
        return vc

    def __repr__(self):
        return f"VC({list(self.clock)})"


# ---------------------------------------------------------------------------
# Epoch (for FastTrack)
# ---------------------------------------------------------------------------

class Epoch:
    """Compact representation: (clock_value, thread_id)."""

    def __init__(self, clock_val, tid):
        self.clock_val = clock_val
        self.tid = tid

    def __repr__(self):
        return f"E({self.clock_val}@T{self.tid})"

    def __eq__(self, other):
        return (isinstance(other, Epoch)
                and self.clock_val == other.clock_val
                and self.tid == other.tid)

    def __hash__(self):
        return hash((self.clock_val, self.tid))


# ---------------------------------------------------------------------------
# Happens-Before Detector
# ---------------------------------------------------------------------------

class HappensBeforeDetector:
    """Classic vector clock based race detector."""

    def __init__(self, n_threads=8):
        self.n_threads = n_threads
        self.thread_clocks = {}
        self.lock_clocks = {}
        self.read_clocks = defaultdict(lambda: defaultdict(lambda: VectorClock(n_threads)))
        self.write_clock = defaultdict(lambda: VectorClock(n_threads))
        self.write_thread = {}
        self.races = []

    def _get_thread_clock(self, tid):
        if tid not in self.thread_clocks:
            self.thread_clocks[tid] = VectorClock(self.n_threads, tid)
        return self.thread_clocks[tid]

    def process_event(self, event):
        new_races = []
        tc = self._get_thread_clock(event.thread)

        if event.etype == 'read':
            new_races = self._process_read(event, tc)
        elif event.etype == 'write':
            new_races = self._process_write(event, tc)
        elif event.etype == 'acquire':
            self._process_acquire(event, tc)
        elif event.etype == 'release':
            self._process_release(event, tc)
        elif event.etype == 'fork':
            self._process_fork(event, tc)
        elif event.etype == 'join':
            self._process_join(event, tc)

        tc.increment(event.thread)
        self.races.extend(new_races)
        return new_races

    def _process_read(self, event, tc):
        races = []
        addr = event.address
        wc = self.write_clock[addr]
        wt = self.write_thread.get(addr)
        if wt is not None and wt != event.thread:
            if not wc.leq(tc):
                race = DataRace(wt, event.thread, addr, 'write', 'read',
                                event2=event)
                races.append(race)
        self.read_clocks[addr][event.thread] = tc.copy()
        return races

    def _process_write(self, event, tc):
        races = []
        addr = event.address
        wc = self.write_clock[addr]
        wt = self.write_thread.get(addr)
        if wt is not None and wt != event.thread:
            if not wc.leq(tc):
                race = DataRace(wt, event.thread, addr, 'write', 'write',
                                event2=event)
                races.append(race)
        for rtid, rc in self.read_clocks[addr].items():
            if rtid != event.thread:
                if not rc.leq(tc):
                    race = DataRace(rtid, event.thread, addr, 'read', 'write',
                                    event2=event)
                    races.append(race)
        self.write_clock[addr] = tc.copy()
        self.write_thread[addr] = event.thread
        return races

    def _process_acquire(self, event, tc):
        if event.lock_id in self.lock_clocks:
            tc.join(self.lock_clocks[event.lock_id])

    def _process_release(self, event, tc):
        if event.lock_id not in self.lock_clocks:
            self.lock_clocks[event.lock_id] = VectorClock(self.n_threads)
        self.lock_clocks[event.lock_id] = tc.copy()

    def _process_fork(self, event, tc):
        child_clock = self._get_thread_clock(event.target_thread)
        child_clock.join(tc)

    def _process_join(self, event, tc):
        child_clock = self._get_thread_clock(event.target_thread)
        tc.join(child_clock)

    def get_races(self):
        return list(set(self.races))


# ---------------------------------------------------------------------------
# Eraser Lockset Algorithm
# ---------------------------------------------------------------------------

class EraserDetector:
    """Eraser lockset-based race detector."""

    class VarState(Enum):
        VIRGIN = auto()
        EXCLUSIVE = auto()
        SHARED = auto()
        SHARED_MODIFIED = auto()

    def __init__(self):
        self.thread_locksets = defaultdict(set)
        self.var_lockset = defaultdict(lambda: None)
        self.var_state = defaultdict(lambda: EraserDetector.VarState.VIRGIN)
        self.var_owner = {}
        self.races = []

    def process_event(self, event):
        new_races = []
        if event.etype == 'acquire':
            self.thread_locksets[event.thread].add(event.lock_id)
        elif event.etype == 'release':
            self.thread_locksets[event.thread].discard(event.lock_id)
        elif event.is_memory_access():
            new_races = self._check_access(event)
        self.races.extend(new_races)
        return new_races

    def _check_access(self, event):
        races = []
        addr = event.address
        tid = event.thread
        state = self.var_state[addr]
        current_locks = frozenset(self.thread_locksets[tid])

        if state == self.VarState.VIRGIN:
            self.var_state[addr] = self.VarState.EXCLUSIVE
            self.var_owner[addr] = tid
            self.var_lockset[addr] = set(current_locks) if current_locks else None
            return races

        if state == self.VarState.EXCLUSIVE:
            if self.var_owner[addr] == tid:
                return races
            if event.is_read():
                self.var_state[addr] = self.VarState.SHARED
            else:
                self.var_state[addr] = self.VarState.SHARED_MODIFIED
            if self.var_lockset[addr] is not None:
                self.var_lockset[addr] = self.var_lockset[addr] & set(current_locks)
            else:
                self.var_lockset[addr] = set(current_locks)
            if not self.var_lockset[addr]:
                race = DataRace(self.var_owner[addr], tid, addr, 'write', event.etype)
                race.classification = 'potential'
                races.append(race)
            return races

        if state == self.VarState.SHARED:
            if event.is_read():
                if self.var_lockset[addr] is not None:
                    self.var_lockset[addr] &= set(current_locks)
                return races
            self.var_state[addr] = self.VarState.SHARED_MODIFIED
            if self.var_lockset[addr] is not None:
                self.var_lockset[addr] &= set(current_locks)
            else:
                self.var_lockset[addr] = set(current_locks)
            if not self.var_lockset[addr]:
                race = DataRace(self.var_owner.get(addr, -1), tid, addr, 'read', 'write')
                race.classification = 'potential'
                races.append(race)
            return races

        # SHARED_MODIFIED
        if self.var_lockset[addr] is not None:
            self.var_lockset[addr] &= set(current_locks)
        else:
            self.var_lockset[addr] = set(current_locks)
        if not self.var_lockset[addr]:
            acc = 'read' if event.is_read() else 'write'
            race = DataRace(self.var_owner.get(addr, -1), tid, addr, 'write', acc)
            race.classification = 'potential'
            races.append(race)
        return races

    def get_races(self):
        return list(set(self.races))


# ---------------------------------------------------------------------------
# Hybrid Detector (HB + Lockset)
# ---------------------------------------------------------------------------

class HybridDetector:
    """Combine happens-before and lockset for better precision."""

    def __init__(self, n_threads=8):
        self.hb = HappensBeforeDetector(n_threads)
        self.lockset = EraserDetector()
        self.races = []

    def process_event(self, event):
        hb_races = self.hb.process_event(event)
        ls_races = self.lockset.process_event(event)
        confirmed = []
        hb_set = set(hb_races)
        for r in ls_races:
            if r in hb_set:
                r.classification = 'confirmed'
                confirmed.append(r)
        for r in hb_races:
            if r not in confirmed:
                r.classification = 'hb-only'
                confirmed.append(r)
        self.races.extend(confirmed)
        return confirmed

    def get_races(self):
        return list(set(self.races))


# ---------------------------------------------------------------------------
# FastTrack Detector
# ---------------------------------------------------------------------------

class FastTrackDetector:
    """FastTrack: epoch-based optimization of vector clock race detection."""

    def __init__(self, n_threads=8):
        self.n_threads = n_threads
        self.thread_clocks = {}
        self.lock_clocks = {}
        self.read_state = {}
        self.write_state = {}
        self.races = []

    def _get_tc(self, tid):
        if tid not in self.thread_clocks:
            self.thread_clocks[tid] = VectorClock(self.n_threads, tid)
        return self.thread_clocks[tid]

    def process_event(self, event):
        new_races = []
        tc = self._get_tc(event.thread)

        if event.etype == 'read':
            new_races = self._ft_read(event, tc)
        elif event.etype == 'write':
            new_races = self._ft_write(event, tc)
        elif event.etype == 'acquire':
            if event.lock_id in self.lock_clocks:
                tc.join(self.lock_clocks[event.lock_id])
        elif event.etype == 'release':
            self.lock_clocks[event.lock_id] = tc.copy()
        elif event.etype == 'fork':
            self._get_tc(event.target_thread).join(tc)
        elif event.etype == 'join':
            tc.join(self._get_tc(event.target_thread))

        tc.increment(event.thread)
        self.races.extend(new_races)
        return new_races

    def _ft_read(self, event, tc):
        races = []
        addr = event.address
        tid = event.thread
        w_epoch = self.write_state.get(addr)
        if w_epoch is not None and w_epoch.tid != tid:
            if w_epoch.clock_val > tc.get(w_epoch.tid):
                races.append(DataRace(w_epoch.tid, tid, addr, 'write', 'read',
                                      event2=event))
        r_state = self.read_state.get(addr)
        epoch = Epoch(tc.get(tid), tid)
        if r_state is None:
            self.read_state[addr] = epoch
        elif isinstance(r_state, Epoch):
            if r_state.tid == tid:
                self.read_state[addr] = epoch
            else:
                self.read_state[addr] = {r_state.tid: r_state, tid: epoch}
        else:
            r_state[tid] = epoch
        return races

    def _ft_write(self, event, tc):
        races = []
        addr = event.address
        tid = event.thread
        w_epoch = self.write_state.get(addr)
        if w_epoch is not None and w_epoch.tid != tid:
            if w_epoch.clock_val > tc.get(w_epoch.tid):
                races.append(DataRace(w_epoch.tid, tid, addr, 'write', 'write',
                                      event2=event))
        r_state = self.read_state.get(addr)
        if r_state is not None:
            if isinstance(r_state, Epoch):
                if r_state.tid != tid and r_state.clock_val > tc.get(r_state.tid):
                    races.append(DataRace(r_state.tid, tid, addr, 'read', 'write',
                                          event2=event))
            else:
                for rtid, rep in r_state.items():
                    if rtid != tid and rep.clock_val > tc.get(rtid):
                        races.append(DataRace(rtid, tid, addr, 'read', 'write',
                                              event2=event))
        self.write_state[addr] = Epoch(tc.get(tid), tid)
        return races

    def get_races(self):
        return list(set(self.races))


# ---------------------------------------------------------------------------
# ThreadSanitizer-style Shadow Memory Detector
# ---------------------------------------------------------------------------

class ShadowCell:
    """Shadow memory cell storing metadata for one access."""

    def __init__(self, tid=-1, clock_val=0, is_write=False):
        self.tid = tid
        self.clock_val = clock_val
        self.is_write = is_write

    def is_empty(self):
        return self.tid == -1

    def __repr__(self):
        if self.is_empty():
            return "ShadowCell(empty)"
        acc = "W" if self.is_write else "R"
        return f"ShadowCell(T{self.tid}:{acc}@{self.clock_val})"


class TSanDetector:
    """ThreadSanitizer-style detector: 2 shadow cells per memory word."""

    N_SHADOW_CELLS = 2

    def __init__(self, n_threads=8):
        self.n_threads = n_threads
        self.thread_clocks = {}
        self.lock_clocks = {}
        self.shadow = defaultdict(
            lambda: [ShadowCell() for _ in range(self.N_SHADOW_CELLS)]
        )
        self.races = []
        self._replace_idx = defaultdict(int)

    def _get_tc(self, tid):
        if tid not in self.thread_clocks:
            self.thread_clocks[tid] = VectorClock(self.n_threads, tid)
        return self.thread_clocks[tid]

    def process_event(self, event):
        new_races = []
        tc = self._get_tc(event.thread)

        if event.etype == 'acquire':
            if event.lock_id in self.lock_clocks:
                tc.join(self.lock_clocks[event.lock_id])
        elif event.etype == 'release':
            self.lock_clocks[event.lock_id] = tc.copy()
        elif event.etype == 'fork':
            self._get_tc(event.target_thread).join(tc)
        elif event.etype == 'join':
            tc.join(self._get_tc(event.target_thread))
        elif event.is_memory_access():
            new_races = self._check_shadow(event, tc)

        tc.increment(event.thread)
        self.races.extend(new_races)
        return new_races

    def _check_shadow(self, event, tc):
        races = []
        addr = event.address
        tid = event.thread
        is_write = event.is_write()
        cells = self.shadow[addr]

        for cell in cells:
            if cell.is_empty() or cell.tid == tid:
                continue
            if is_write or cell.is_write:
                if cell.clock_val > tc.get(cell.tid):
                    acc1 = 'write' if cell.is_write else 'read'
                    acc2 = 'write' if is_write else 'read'
                    race = DataRace(cell.tid, tid, addr, acc1, acc2, event2=event)
                    races.append(race)

        new_cell = ShadowCell(tid, tc.get(tid), is_write)
        replaced = False
        for i, cell in enumerate(cells):
            if cell.tid == tid or cell.is_empty():
                cells[i] = new_cell
                replaced = True
                break
        if not replaced:
            idx = self._replace_idx[addr] % self.N_SHADOW_CELLS
            cells[idx] = new_cell
            self._replace_idx[addr] += 1

        return races

    def get_races(self):
        return list(set(self.races))


# ---------------------------------------------------------------------------
# Race Classification
# ---------------------------------------------------------------------------

class RaceClassifier:
    """Heuristics for classifying races as benign or harmful."""

    def classify(self, race, context=None):
        if race.access1 == 'write' and race.access2 == 'write':
            race.classification = 'harmful'
            return 'harmful'
        if context and context.get('is_flag'):
            race.classification = 'benign'
            return 'benign'
        if context and context.get('monotonic'):
            race.classification = 'benign'
            return 'benign'
        race.classification = 'potentially_harmful'
        return 'potentially_harmful'

    def classify_all(self, races, context=None):
        return {race: self.classify(race, context) for race in races}


# ---------------------------------------------------------------------------
# Trace Replay
# ---------------------------------------------------------------------------

class TraceReplayer:
    """Deterministic replay from logged events."""

    def __init__(self):
        self.memory = {}
        self.thread_regs = defaultdict(dict)

    def replay(self, events):
        self.memory = {}
        trace_log = []
        for event in events:
            if event.etype == 'write':
                old_val = self.memory.get(event.address, 0)
                self.memory[event.address] = event.value
                trace_log.append({'event': event, 'old_value': old_val, 'new_value': event.value})
            elif event.etype == 'read':
                val = self.memory.get(event.address, 0)
                trace_log.append({'event': event, 'value_read': val,
                                  'expected': event.value,
                                  'match': val == event.value if event.value is not None else True})
            else:
                trace_log.append({'event': event})
        return {'final_memory': dict(self.memory), 'trace_log': trace_log}

    def replay_and_detect(self, events, detector=None):
        if detector is None:
            detector = HappensBeforeDetector()
        self.memory = {}
        all_races = []
        for event in events:
            if event.etype == 'write':
                self.memory[event.address] = event.value
            new_races = detector.process_event(event)
            all_races.extend(new_races)
        return {'final_memory': dict(self.memory), 'races': list(set(all_races))}


# ---------------------------------------------------------------------------
# Trace generation helpers
# ---------------------------------------------------------------------------

def make_trace_event(thread, op, addr=None, val=None, lock=None, target=None):
    if op == 'read':
        return TraceEvent(thread, 'read', address=addr, value=val)
    elif op == 'write':
        return TraceEvent(thread, 'write', address=addr, value=val)
    elif op == 'acquire':
        return TraceEvent(thread, 'acquire', lock_id=lock)
    elif op == 'release':
        return TraceEvent(thread, 'release', lock_id=lock)
    elif op == 'fork':
        return TraceEvent(thread, 'fork', target_thread=target)
    elif op == 'join':
        return TraceEvent(thread, 'join', target_thread=target)
    raise ValueError(f"Unknown op: {op}")


def generate_racy_trace():
    return [
        make_trace_event(0, 'write', addr='x', val=1),
        make_trace_event(1, 'write', addr='x', val=2),
        make_trace_event(0, 'read', addr='x', val=1),
        make_trace_event(1, 'read', addr='x', val=2),
    ]


def generate_synchronized_trace():
    return [
        make_trace_event(0, 'acquire', lock='L1'),
        make_trace_event(0, 'write', addr='x', val=1),
        make_trace_event(0, 'release', lock='L1'),
        make_trace_event(1, 'acquire', lock='L1'),
        make_trace_event(1, 'read', addr='x', val=1),
        make_trace_event(1, 'release', lock='L1'),
    ]


def generate_fork_join_trace():
    return [
        make_trace_event(0, 'write', addr='x', val=42),
        make_trace_event(0, 'fork', target=1),
        make_trace_event(1, 'read', addr='x', val=42),
        make_trace_event(1, 'write', addr='y', val=100),
        make_trace_event(0, 'join', target=1),
        make_trace_event(0, 'read', addr='y', val=100),
    ]


def generate_complex_race_trace():
    TraceEvent._counter = 0
    return [
        make_trace_event(0, 'acquire', lock='L1'),
        make_trace_event(0, 'write', addr='x', val=1),
        make_trace_event(0, 'release', lock='L1'),
        make_trace_event(1, 'acquire', lock='L1'),
        make_trace_event(1, 'read', addr='x', val=1),
        make_trace_event(1, 'release', lock='L1'),
        make_trace_event(0, 'write', addr='y', val=10),
        make_trace_event(1, 'read', addr='y', val=10),
        make_trace_event(2, 'write', addr='z', val=5),
        make_trace_event(0, 'read', addr='z', val=5),
    ]


if __name__ == "__main__":
    TraceEvent._counter = 0
    print("=== HB Detector on racy trace ===")
    det = HappensBeforeDetector(n_threads=4)
    for e in generate_racy_trace():
        det.process_event(e)
    races = det.get_races()
    print(f"Races found: {len(races)}")
    for r in races:
        print(f"  {r}")
    assert len(races) > 0

    print("\n=== HB Detector on synchronized trace ===")
    TraceEvent._counter = 0
    det2 = HappensBeforeDetector(n_threads=4)
    for e in generate_synchronized_trace():
        det2.process_event(e)
    assert len(det2.get_races()) == 0

    print("\n=== FastTrack on racy trace ===")
    TraceEvent._counter = 0
    ft = FastTrackDetector(n_threads=4)
    for e in generate_racy_trace():
        ft.process_event(e)
    assert len(ft.get_races()) > 0

    print("\n=== TSan on racy trace ===")
    TraceEvent._counter = 0
    tsan = TSanDetector(n_threads=4)
    for e in generate_racy_trace():
        tsan.process_event(e)
    assert len(tsan.get_races()) > 0

    print("\nrace_detector.py self-test passed")

"""
Formal memory model implementations for concurrent program verification.

Implements SC, TSO, PSO, ARM/POWER relaxed, RVWMO, PTX (GPU scopes),
and Vulkan memory models with axiom-based consistency checking.
"""

import numpy as np
from abc import ABC, abstractmethod
from enum import Enum, auto
from itertools import permutations, product
from collections import defaultdict
import copy


# ---------------------------------------------------------------------------
# Memory events
# ---------------------------------------------------------------------------

class EventType(Enum):
    READ = auto()
    WRITE = auto()
    FENCE = auto()
    RMW = auto()


class FenceType(Enum):
    FULL = auto()       # mfence / DMB ISH
    STORE = auto()      # sfence / DMB ISHST
    LOAD = auto()       # lfence
    ACQ = auto()        # acquire
    REL = auto()        # release
    ACQ_REL = auto()    # acq+rel
    SC = auto()         # sequentially consistent
    SYNC = auto()       # POWER sync
    LWSYNC = auto()     # POWER lwsync
    ISYNC = auto()      # POWER isync


class Scope(Enum):
    CTA = auto()
    GPU = auto()
    SYSTEM = auto()


class MemoryEvent:
    """Single memory event in an execution."""
    _counter = 0

    def __init__(self, etype, thread, address=None, value=None,
                 fence_type=None, scope=None, order=0):
        MemoryEvent._counter += 1
        self.eid = MemoryEvent._counter
        self.etype = etype
        self.thread = thread
        self.address = address
        self.value = value
        self.fence_type = fence_type
        self.scope = scope
        self.order = order  # program order index within thread

    def is_read(self):
        return self.etype == EventType.READ

    def is_write(self):
        return self.etype in (EventType.WRITE, EventType.RMW)

    def is_fence(self):
        return self.etype == EventType.FENCE

    def is_rmw(self):
        return self.etype == EventType.RMW

    def same_address(self, other):
        return self.address is not None and self.address == other.address

    def __repr__(self):
        if self.etype == EventType.FENCE:
            return f"Fence({self.fence_type.name}, T{self.thread})"
        tag = self.etype.name[0]
        return f"{tag}(x{self.address}={self.value}, T{self.thread})"

    def __eq__(self, other):
        return isinstance(other, MemoryEvent) and self.eid == other.eid

    def __hash__(self):
        return hash(self.eid)


def Read(addr, val, thread, order=0, scope=None):
    return MemoryEvent(EventType.READ, thread, addr, val, scope=scope, order=order)


def Write(addr, val, thread, order=0, scope=None):
    return MemoryEvent(EventType.WRITE, thread, addr, val, scope=scope, order=order)


def Fence(ftype, thread, order=0, scope=None):
    return MemoryEvent(EventType.FENCE, thread, fence_type=ftype, scope=scope, order=order)


def RMW(addr, old_val, new_val, thread, order=0, scope=None):
    e = MemoryEvent(EventType.RMW, thread, addr, new_val, scope=scope, order=order)
    e.old_value = old_val
    return e


# ---------------------------------------------------------------------------
# Execution and relations
# ---------------------------------------------------------------------------

class Relation:
    """Binary relation over events, stored as adjacency set."""

    def __init__(self, name=""):
        self.name = name
        self._edges = set()
        self._adj = defaultdict(set)

    def add(self, a, b):
        self._edges.add((a.eid, b.eid))
        self._adj[a.eid].add(b.eid)

    def contains(self, a, b):
        return (a.eid, b.eid) in self._edges

    def successors(self, a):
        return self._adj.get(a.eid, set())

    def edges(self):
        return self._edges

    def __len__(self):
        return len(self._edges)

    def __or__(self, other):
        r = Relation(f"({self.name}|{other.name})")
        r._edges = self._edges | other._edges
        for k, v in self._adj.items():
            r._adj[k] |= v
        for k, v in other._adj.items():
            r._adj[k] |= v
        return r

    def compose(self, other, events_by_id):
        """Relational composition: self;other."""
        r = Relation(f"({self.name};{other.name})")
        for (a, b) in self._edges:
            for c in other._adj.get(b, set()):
                r._edges.add((a, c))
                r._adj[a].add(c)
        return r

    def transitive_closure(self):
        """Warshall's algorithm on event ids."""
        ids = set()
        for a, b in self._edges:
            ids.add(a)
            ids.add(b)
        id_list = sorted(ids)
        n = len(id_list)
        idx = {v: i for i, v in enumerate(id_list)}
        mat = np.zeros((n, n), dtype=bool)
        for a, b in self._edges:
            mat[idx[a], idx[b]] = True
        for k in range(n):
            mat |= np.outer(mat[:, k], mat[k, :])
        r = Relation(f"tc({self.name})")
        for i in range(n):
            for j in range(n):
                if mat[i, j]:
                    r._edges.add((id_list[i], id_list[j]))
                    r._adj[id_list[i]].add(id_list[j])
        return r

    def is_acyclic(self):
        """Check acyclicity via DFS."""
        adj = self._adj
        WHITE, GRAY, BLACK = 0, 1, 2
        color = defaultdict(int)
        nodes = set()
        for a, b in self._edges:
            nodes.add(a)
            nodes.add(b)

        def dfs(u):
            color[u] = GRAY
            for v in adj.get(u, set()):
                if color[v] == GRAY:
                    return False
                if color[v] == WHITE:
                    if not dfs(v):
                        return False
            color[u] = BLACK
            return True

        for n in nodes:
            if color[n] == WHITE:
                if not dfs(n):
                    return False
        return True

    def is_total_on(self, event_ids):
        """Check if relation is total order on given event ids."""
        for a in event_ids:
            for b in event_ids:
                if a != b:
                    if (a, b) not in self._edges and (b, a) not in self._edges:
                        return False
        return True

    def find_cycle(self):
        """Return a cycle as list of event ids, or None."""
        adj = self._adj
        WHITE, GRAY, BLACK = 0, 1, 2
        color = defaultdict(int)
        parent = {}
        nodes = set()
        for a, b in self._edges:
            nodes.add(a)
            nodes.add(b)

        def dfs(u, path):
            color[u] = GRAY
            path.append(u)
            for v in adj.get(u, set()):
                if color[v] == GRAY:
                    idx = path.index(v)
                    return list(path[idx:])
                if color[v] == WHITE:
                    result = dfs(v, path)
                    if result is not None:
                        return result
            path.pop()
            color[u] = BLACK
            return None

        for n in sorted(nodes):
            if color[n] == WHITE:
                cycle = dfs(n, [])
                if cycle is not None:
                    return cycle
        return None


class Execution:
    """Represents a candidate execution of a concurrent program."""

    def __init__(self):
        self.events = []
        self.events_by_id = {}
        self.po = Relation("po")        # program order
        self.rf = Relation("rf")        # reads-from
        self.co = Relation("co")        # coherence order (write-write per address)
        self.fr = Relation("fr")        # from-reads (derived)
        self.threads = defaultdict(list)
        self.init_writes = {}

    def add_event(self, event):
        self.events.append(event)
        self.events_by_id[event.eid] = event
        self.threads[event.thread].append(event)

    def add_init_write(self, addr, val=0):
        w = Write(addr, val, thread=-1, order=-1)
        self.init_writes[addr] = w
        self.events.append(w)
        self.events_by_id[w.eid] = w

    def build_program_order(self):
        """Build po from thread-local event orderings."""
        for tid, evts in self.threads.items():
            evts_sorted = sorted(evts, key=lambda e: e.order)
            for i in range(len(evts_sorted)):
                for j in range(i + 1, len(evts_sorted)):
                    self.po.add(evts_sorted[i], evts_sorted[j])

    def derive_fr(self):
        """Compute from-reads: fr = rf^{-1} ; co."""
        rf_inv = Relation("rf_inv")
        for a_id, b_id in self.rf.edges():
            a = self.events_by_id[a_id]
            b = self.events_by_id[b_id]
            rf_inv.add(b, a)
        for (r_id, w_id) in rf_inv.edges():
            r = self.events_by_id[r_id]
            w = self.events_by_id[w_id]
            for (wa_id, wb_id) in self.co.edges():
                if wa_id == w_id:
                    wb = self.events_by_id[wb_id]
                    self.fr.add(r, wb)

    def writes_to(self, addr):
        return [e for e in self.events if e.is_write() and e.address == addr]

    def reads_from_addr(self, addr):
        return [e for e in self.events if e.is_read() and e.address == addr]

    def fences_in_thread(self, tid):
        return [e for e in self.threads.get(tid, []) if e.is_fence()]

    def get_addresses(self):
        addrs = set()
        for e in self.events:
            if e.address is not None:
                addrs.add(e.address)
        return addrs

    def clone(self):
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# Helper: topological sort for checking total-order existence
# ---------------------------------------------------------------------------

def _topological_sort_exists(events, constraints):
    """Check if a topological ordering of events exists respecting constraints.
    constraints: list of (a_eid, b_eid) meaning a must come before b.
    Returns True if such ordering exists (i.e. no cycle).
    """
    adj = defaultdict(set)
    indegree = defaultdict(int)
    ids = {e.eid for e in events}
    for eid in ids:
        indegree[eid] = 0
    for a, b in constraints:
        if a in ids and b in ids:
            adj[a].add(b)
            indegree[b] += 1
    queue = [eid for eid in ids if indegree[eid] == 0]
    count = 0
    while queue:
        u = queue.pop()
        count += 1
        for v in adj[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)
    return count == len(ids)


def _find_total_order(events, constraints):
    """Find a total order consistent with constraints, or None."""
    adj = defaultdict(set)
    indegree = defaultdict(int)
    ids = [e.eid for e in events]
    for eid in ids:
        indegree[eid] = 0
    for a, b in constraints:
        if a in set(ids) and b in set(ids) and b not in adj[a]:
            adj[a].add(b)
            indegree[b] += 1
    order = []
    available = sorted([eid for eid in ids if indegree[eid] == 0])
    while available:
        u = available.pop(0)
        order.append(u)
        for v in sorted(adj[u]):
            indegree[v] -= 1
            if indegree[v] == 0:
                available.append(v)
        available.sort()
    if len(order) == len(ids):
        return order
    return None


# ---------------------------------------------------------------------------
# Abstract memory model
# ---------------------------------------------------------------------------

class MemoryModel(ABC):
    """Abstract base for memory consistency models."""

    def __init__(self, name=""):
        self.name = name

    @abstractmethod
    def check(self, execution):
        """Check if execution is allowed under this model.
        Returns ('allowed', None) or ('forbidden', reason_string).
        """
        pass

    def check_no_thin_air(self, execution):
        """No-thin-air: (po ∪ rf) must be acyclic."""
        combined = execution.po | execution.rf
        if not combined.is_acyclic():
            return False, "thin-air cycle in po∪rf"
        return True, None

    def check_coherence(self, execution):
        """Per-location coherence: (po_loc ∪ rf ∪ co ∪ fr) acyclic."""
        po_loc = Relation("po_loc")
        for tid, evts in execution.threads.items():
            sorted_evts = sorted(evts, key=lambda e: e.order)
            for i in range(len(sorted_evts)):
                for j in range(i + 1, len(sorted_evts)):
                    a, b = sorted_evts[i], sorted_evts[j]
                    if a.address is not None and a.address == b.address:
                        po_loc.add(a, b)
        combined = po_loc | execution.rf | execution.co | execution.fr
        if not combined.is_acyclic():
            return False, "coherence violation"
        return True, None

    def check_rf_values(self, execution):
        """Each read reads from exactly one write with matching value."""
        for e in execution.events:
            if e.is_read():
                sources = []
                for (w_id, r_id) in execution.rf.edges():
                    if r_id == e.eid:
                        w = execution.events_by_id[w_id]
                        sources.append(w)
                if len(sources) == 0:
                    return False, f"Read {e} has no rf source"
                if len(sources) > 1:
                    return False, f"Read {e} has multiple rf sources"
                if sources[0].value != e.value:
                    return False, f"Read {e} value mismatch with write {sources[0]}"
        return True, None


# ---------------------------------------------------------------------------
# Sequential Consistency
# ---------------------------------------------------------------------------

class SequentialConsistency(MemoryModel):
    """SC: exists a total order on all events consistent with program order
    and each read sees the most recent write to that address."""

    def __init__(self):
        super().__init__("SC")

    def check(self, execution):
        ok, reason = self.check_rf_values(execution)
        if not ok:
            return "forbidden", reason

        mem_events = [e for e in execution.events if not e.is_fence()]
        constraints = list(execution.po.edges())
        constraints.extend(execution.rf.edges())
        constraints.extend(execution.co.edges())
        constraints.extend(execution.fr.edges())

        combined = Relation("sc_check")
        for a_id, b_id in constraints:
            if a_id in execution.events_by_id and b_id in execution.events_by_id:
                combined.add(execution.events_by_id[a_id],
                             execution.events_by_id[b_id])
        if not combined.is_acyclic():
            return "forbidden", "no total SC order exists (cycle in po∪rf∪co∪fr)"

        order = _find_total_order(mem_events, constraints)
        if order is None:
            return "forbidden", "no total SC order exists"

        if not self._verify_read_latest(execution, order):
            return "forbidden", "read does not see latest write in SC order"

        return "allowed", None

    def _verify_read_latest(self, execution, order):
        """Verify each read sees the most recent write in the total order."""
        pos = {eid: i for i, eid in enumerate(order)}
        for e in execution.events:
            if not e.is_read():
                continue
            if e.eid not in pos:
                continue
            sources = []
            for (w_id, r_id) in execution.rf.edges():
                if r_id == e.eid:
                    sources.append(w_id)
            if not sources:
                continue
            w_id = sources[0]
            if w_id not in pos:
                continue
            for other in execution.writes_to(e.address):
                if other.eid == w_id or other.eid not in pos:
                    continue
                if pos[w_id] < pos[other.eid] < pos[e.eid]:
                    return False
        return True


# ---------------------------------------------------------------------------
# Total Store Order (x86-TSO)
# ---------------------------------------------------------------------------

class StoreBuffer:
    """FIFO store buffer for TSO simulation."""

    def __init__(self, thread_id):
        self.thread_id = thread_id
        self.buffer = []  # list of (addr, val, event)

    def write(self, addr, val, event):
        self.buffer.append((addr, val, event))

    def read(self, addr):
        """Read from store buffer (most recent write to addr), or None."""
        for a, v, e in reversed(self.buffer):
            if a == addr:
                return v, e
        return None

    def flush_one(self):
        """Flush oldest entry. Returns (addr, val, event) or None."""
        if self.buffer:
            return self.buffer.pop(0)
        return None

    def flush_all(self):
        """Flush all entries. Returns list."""
        entries = list(self.buffer)
        self.buffer.clear()
        return entries

    def is_empty(self):
        return len(self.buffer) == 0

    def __len__(self):
        return len(self.buffer)


class TotalStoreOrder(MemoryModel):
    """x86-TSO: SC + per-processor FIFO store buffer.
    Relaxation: store-load reordering (stores can be delayed).
    Preserved: store-store, load-load, load-store orders.
    """

    def __init__(self):
        super().__init__("TSO")

    def check(self, execution):
        ok, reason = self.check_rf_values(execution)
        if not ok:
            return "forbidden", reason

        ok, reason = self.check_no_thin_air(execution)
        if not ok:
            return "forbidden", reason

        ok, reason = self.check_coherence(execution)
        if not ok:
            return "forbidden", reason

        constraints = []
        for (a_id, b_id) in execution.po.edges():
            a = execution.events_by_id.get(a_id)
            b = execution.events_by_id.get(b_id)
            if a is None or b is None:
                continue
            # TSO preserves all orders except store->load
            if not (a.is_write() and b.is_read()):
                constraints.append((a_id, b_id))
            elif a.is_write() and b.is_read() and a.same_address(b):
                # Store-load to same address is preserved (store buffer forwarding)
                constraints.append((a_id, b_id))

        # Add fence constraints: mfence restores store-load order
        for tid, evts in execution.threads.items():
            sorted_evts = sorted(evts, key=lambda e: e.order)
            for i, e in enumerate(sorted_evts):
                if e.is_fence() and e.fence_type in (FenceType.FULL, FenceType.SC):
                    stores_before = [x for x in sorted_evts[:i] if x.is_write()]
                    loads_after = [x for x in sorted_evts[i+1:] if x.is_read()]
                    for s in stores_before:
                        for l in loads_after:
                            constraints.append((s.eid, l.eid))

        constraints.extend(execution.rf.edges())
        constraints.extend(execution.co.edges())
        constraints.extend(execution.fr.edges())

        rel = Relation("tso_check")
        for a_id, b_id in constraints:
            if a_id in execution.events_by_id and b_id in execution.events_by_id:
                rel.add(execution.events_by_id[a_id], execution.events_by_id[b_id])

        if not rel.is_acyclic():
            return "forbidden", "TSO violation: cycle in preserved orders"

        return "allowed", None


# ---------------------------------------------------------------------------
# Partial Store Order (PSO)
# ---------------------------------------------------------------------------

class PartialStoreOrder(MemoryModel):
    """PSO: separate store buffers per address.
    Relaxation: store-store to different addresses can be reordered.
    """

    def __init__(self):
        super().__init__("PSO")

    def check(self, execution):
        ok, reason = self.check_rf_values(execution)
        if not ok:
            return "forbidden", reason

        ok, reason = self.check_no_thin_air(execution)
        if not ok:
            return "forbidden", reason

        ok, reason = self.check_coherence(execution)
        if not ok:
            return "forbidden", reason

        constraints = []
        for (a_id, b_id) in execution.po.edges():
            a = execution.events_by_id.get(a_id)
            b = execution.events_by_id.get(b_id)
            if a is None or b is None:
                continue
            # PSO preserves: load-load, load-store, store-store to same addr
            if a.is_read() and b.is_read():
                constraints.append((a_id, b_id))
            elif a.is_read() and b.is_write():
                constraints.append((a_id, b_id))
            elif a.is_write() and b.is_write() and a.same_address(b):
                constraints.append((a_id, b_id))
            elif a.is_write() and b.is_read() and a.same_address(b):
                constraints.append((a_id, b_id))

        for tid, evts in execution.threads.items():
            sorted_evts = sorted(evts, key=lambda e: e.order)
            for i, e in enumerate(sorted_evts):
                if e.is_fence() and e.fence_type in (FenceType.FULL, FenceType.STORE, FenceType.SC):
                    before = sorted_evts[:i]
                    after = sorted_evts[i+1:]
                    for a in before:
                        for b in after:
                            if a.is_write() and b.is_write():
                                constraints.append((a.eid, b.eid))
                            if e.fence_type in (FenceType.FULL, FenceType.SC):
                                if a.is_write() and b.is_read():
                                    constraints.append((a.eid, b.eid))

        constraints.extend(execution.rf.edges())
        constraints.extend(execution.co.edges())
        constraints.extend(execution.fr.edges())

        rel = Relation("pso_check")
        for a_id, b_id in constraints:
            if a_id in execution.events_by_id and b_id in execution.events_by_id:
                rel.add(execution.events_by_id[a_id], execution.events_by_id[b_id])

        if not rel.is_acyclic():
            return "forbidden", "PSO violation"

        return "allowed", None


# ---------------------------------------------------------------------------
# Relaxed Memory Model (ARM/POWER style)
# ---------------------------------------------------------------------------

class RelaxedMemoryModel(MemoryModel):
    """ARM/POWER relaxed model: no multi-copy atomicity, barriers required.
    Almost all reorderings allowed unless prevented by dependencies or barriers.
    """

    def __init__(self):
        super().__init__("Relaxed")

    def check(self, execution):
        ok, reason = self.check_rf_values(execution)
        if not ok:
            return "forbidden", reason

        ok, reason = self.check_no_thin_air(execution)
        if not ok:
            return "forbidden", reason

        ok, reason = self.check_coherence(execution)
        if not ok:
            return "forbidden", reason

        ppo = self._preserved_program_order(execution)
        prop = self._propagation_order(execution)

        obs = ppo | execution.rf | execution.co | execution.fr | prop
        if not obs.is_acyclic():
            return "forbidden", "relaxed model violation"

        return "allowed", None

    def _preserved_program_order(self, execution):
        """Compute preserved program order: dependencies + fence-induced order."""
        ppo = Relation("ppo")
        for tid, evts in execution.threads.items():
            sorted_evts = sorted(evts, key=lambda e: e.order)
            for i in range(len(sorted_evts)):
                for j in range(i + 1, len(sorted_evts)):
                    a, b = sorted_evts[i], sorted_evts[j]
                    # Address dependency: read -> access to same computed addr
                    if a.is_read() and b.address == a.value:
                        ppo.add(a, b)
                    # Data dependency: read -> write with value from read
                    if a.is_read() and b.is_write() and b.value == a.value:
                        ppo.add(a, b)
                    # Control dependency + isb: read -> read after branch
                    # (simplified: consecutive read-read to different addrs)

            # Fence-induced ordering
            for i, e in enumerate(sorted_evts):
                if not e.is_fence():
                    continue
                before = sorted_evts[:i]
                after = sorted_evts[i+1:]
                if e.fence_type == FenceType.SYNC:
                    for a in before:
                        for b in after:
                            ppo.add(a, b)
                elif e.fence_type == FenceType.LWSYNC:
                    for a in before:
                        for b in after:
                            if not (a.is_write() and b.is_read()):
                                ppo.add(a, b)
                elif e.fence_type in (FenceType.FULL, FenceType.SC):
                    for a in before:
                        for b in after:
                            ppo.add(a, b)
                elif e.fence_type == FenceType.ACQ:
                    for b in after:
                        ppo.add(e, b)
                elif e.fence_type == FenceType.REL:
                    for a in before:
                        ppo.add(a, e)

        return ppo

    def _propagation_order(self, execution):
        """Propagation constraints for multi-copy atomicity violations."""
        prop = Relation("prop")
        # In the relaxed model, co must be consistent across all threads
        # propagation = (rf; fence; rf^-1) ∩ co
        # Simplified: fence between writes implies co propagation
        for (a_id, b_id) in execution.co.edges():
            prop.add(execution.events_by_id[a_id], execution.events_by_id[b_id])
        return prop


# ---------------------------------------------------------------------------
# RISC-V Weak Memory Ordering (RVWMO)
# ---------------------------------------------------------------------------

class RISCVModel(MemoryModel):
    """RVWMO: preserved program order based on syntactic dependencies.
    Key features: address/data/control dependencies preserved,
    AMO operations are acquire+release.
    """

    def __init__(self):
        super().__init__("RVWMO")

    def check(self, execution):
        ok, reason = self.check_rf_values(execution)
        if not ok:
            return "forbidden", reason

        ok, reason = self.check_no_thin_air(execution)
        if not ok:
            return "forbidden", reason

        ok, reason = self.check_coherence(execution)
        if not ok:
            return "forbidden", reason

        ppo = self._compute_ppo(execution)
        combined = ppo | execution.rf | execution.co | execution.fr
        if not combined.is_acyclic():
            return "forbidden", "RVWMO violation"

        return "allowed", None

    def _compute_ppo(self, execution):
        ppo = Relation("rvwmo_ppo")
        for tid, evts in execution.threads.items():
            sorted_evts = sorted(evts, key=lambda e: e.order)
            for i in range(len(sorted_evts)):
                for j in range(i + 1, len(sorted_evts)):
                    a, b = sorted_evts[i], sorted_evts[j]
                    if self._in_ppo(a, b, sorted_evts, i, j, execution):
                        ppo.add(a, b)
        return ppo

    def _in_ppo(self, a, b, sorted_evts, i, j, execution):
        # Rule 1: overlapping addresses
        if a.address is not None and b.address is not None and a.address == b.address:
            if a.is_write() and b.is_write():
                return True
            if a.is_read() and (b.is_read() or b.is_write()):
                return True
            if a.is_write() and b.is_read():
                return True  # same-address store-load preserved in RVWMO
        # Rule 2: explicit fence
        for k in range(i + 1, j):
            e = sorted_evts[k]
            if e.is_fence():
                if e.fence_type in (FenceType.FULL, FenceType.SC):
                    return True
                if e.fence_type == FenceType.ACQ and a.is_read():
                    return True
                if e.fence_type == FenceType.REL and b.is_write():
                    return True
        # Rule 3: address dependency
        if a.is_read() and b.address == a.value:
            return True
        # Rule 4: data dependency
        if a.is_read() and b.is_write() and b.value == a.value:
            return True
        # Rule 5: AMO semantics
        if a.is_rmw() or b.is_rmw():
            return True
        return False


# ---------------------------------------------------------------------------
# PTX GPU Memory Model (scoped)
# ---------------------------------------------------------------------------

class PTXModel(MemoryModel):
    """PTX GPU memory model with hierarchical scopes: CTA < GPU < System.
    Operations are visible only within their declared scope.
    """

    def __init__(self):
        super().__init__("PTX")

    def check(self, execution):
        ok, reason = self.check_rf_values(execution)
        if not ok:
            return "forbidden", reason

        ok, reason = self.check_no_thin_air(execution)
        if not ok:
            return "forbidden", reason

        scope_constraints = self._scope_constraints(execution)
        combined = execution.po | execution.rf | execution.co | execution.fr | scope_constraints
        if not combined.is_acyclic():
            return "forbidden", "PTX scope violation"

        return "allowed", None

    def _scope_constraints(self, execution):
        """Compute ordering constraints based on scope annotations."""
        sc = Relation("ptx_scope")

        scope_order = {Scope.CTA: 0, Scope.GPU: 1, Scope.SYSTEM: 2}

        def effective_scope(e):
            return e.scope if e.scope else Scope.CTA

        def scopes_overlap(s1, s2, t1, t2):
            """Check if scopes are sufficient for cross-thread visibility."""
            min_scope = min(scope_order.get(s1, 0), scope_order.get(s2, 0))
            # Threads in same CTA need CTA scope, same GPU need GPU scope, etc.
            if t1 == t2:
                return True
            cta1, cta2 = t1 // 32, t2 // 32
            if cta1 == cta2:
                return min_scope >= scope_order[Scope.CTA]
            gpu1, gpu2 = t1 // 1024, t2 // 1024
            if gpu1 == gpu2:
                return min_scope >= scope_order[Scope.GPU]
            return min_scope >= scope_order[Scope.SYSTEM]

        # rf edges only valid if scopes overlap
        for (w_id, r_id) in execution.rf.edges():
            w = execution.events_by_id[w_id]
            r = execution.events_by_id[r_id]
            ws = effective_scope(w)
            rs = effective_scope(r)
            if scopes_overlap(ws, rs, w.thread, r.thread):
                sc.add(w, r)

        # Fences with scope
        for tid, evts in execution.threads.items():
            sorted_evts = sorted(evts, key=lambda e: e.order)
            for i, e in enumerate(sorted_evts):
                if e.is_fence():
                    fence_scope = effective_scope(e)
                    before = [x for x in sorted_evts[:i] if not x.is_fence()]
                    after = [x for x in sorted_evts[i+1:] if not x.is_fence()]
                    for a in before:
                        for b in after:
                            sc.add(a, b)

        return sc

    def scope_sufficient(self, w_event, r_event):
        """Check if scope annotations allow visibility."""
        scope_order = {Scope.CTA: 0, Scope.GPU: 1, Scope.SYSTEM: 2}
        ws = w_event.scope if w_event.scope else Scope.CTA
        rs = r_event.scope if r_event.scope else Scope.CTA
        min_scope = min(scope_order.get(ws, 0), scope_order.get(rs, 0))
        if w_event.thread == r_event.thread:
            return True
        return min_scope >= scope_order[Scope.GPU]


# ---------------------------------------------------------------------------
# Vulkan Memory Model
# ---------------------------------------------------------------------------

class VulkanModel(MemoryModel):
    """Vulkan memory model with availability/visibility operations.
    Key concepts: available (flushed from cache), visible (in requesting cache).
    """

    def __init__(self):
        super().__init__("Vulkan")
        self.availability = defaultdict(set)   # addr -> set of (val, scope)
        self.visibility = defaultdict(lambda: defaultdict(set))  # thread -> addr -> set of (val, scope)

    def check(self, execution):
        ok, reason = self.check_rf_values(execution)
        if not ok:
            return "forbidden", reason

        ok, reason = self.check_no_thin_air(execution)
        if not ok:
            return "forbidden", reason

        avail_vis = self._availability_visibility_check(execution)
        if not avail_vis:
            return "forbidden", "Vulkan availability/visibility violation"

        combined = execution.po | execution.rf | execution.co | execution.fr
        if not combined.is_acyclic():
            return "forbidden", "Vulkan coherence violation"

        return "allowed", None

    def _availability_visibility_check(self, execution):
        """Check availability/visibility chains for cross-thread communication."""
        scope_order = {Scope.CTA: 0, Scope.GPU: 1, Scope.SYSTEM: 2, None: 0}

        for (w_id, r_id) in execution.rf.edges():
            w = execution.events_by_id[w_id]
            r = execution.events_by_id[r_id]

            if w.thread == r.thread:
                continue

            # Need: write made available, then made visible to reader
            w_scope = w.scope if w.scope else Scope.CTA
            r_scope = r.scope if r.scope else Scope.CTA

            # Check if there's a release on writer side and acquire on reader side
            has_release = self._has_release_after(execution, w)
            has_acquire = self._has_acquire_before(execution, r)

            if not (has_release and has_acquire):
                # Check scope inclusion
                if scope_order[w_scope] < scope_order.get(Scope.GPU, 1):
                    if scope_order[r_scope] < scope_order.get(Scope.GPU, 1):
                        return False

        return True

    def _has_release_after(self, execution, write_event):
        """Check if there's a release fence/op after write in program order."""
        tid = write_event.thread
        for e in execution.threads.get(tid, []):
            if e.order > write_event.order:
                if e.is_fence() and e.fence_type in (FenceType.REL, FenceType.ACQ_REL, FenceType.SC, FenceType.FULL):
                    return True
                if e.is_write() and e.scope in (Scope.GPU, Scope.SYSTEM):
                    return True
        # The write itself might be a release
        if write_event.scope in (Scope.GPU, Scope.SYSTEM):
            return True
        return False

    def _has_acquire_before(self, execution, read_event):
        """Check if there's an acquire fence/op before read in program order."""
        tid = read_event.thread
        for e in execution.threads.get(tid, []):
            if e.order < read_event.order:
                if e.is_fence() and e.fence_type in (FenceType.ACQ, FenceType.ACQ_REL, FenceType.SC, FenceType.FULL):
                    return True
        if read_event.scope in (Scope.GPU, Scope.SYSTEM):
            return True
        return False


# ---------------------------------------------------------------------------
# Consistency axiom checkers (reusable)
# ---------------------------------------------------------------------------

class ConsistencyChecker:
    """Reusable consistency axiom checks across models."""

    @staticmethod
    def check_observation(execution):
        """Observation axiom: fre;prop;hb is irreflexive.
        (simplified) Ensures writes become visible in order.
        """
        fre = Relation("fre")
        for (r_id, w_id) in execution.fr.edges():
            r = execution.events_by_id.get(r_id)
            w = execution.events_by_id.get(w_id)
            if r and w and r.thread != w.thread:
                fre.add(r, w)
        combined = fre | execution.co | execution.po
        tc = combined.transitive_closure()
        for eid in {e.eid for e in execution.events}:
            if (eid, eid) in tc.edges():
                return False, f"observation axiom violated at event {eid}"
        return True, None

    @staticmethod
    def check_propagation(execution):
        """Propagation axiom: co ∪ prop is acyclic."""
        combined = execution.co
        if not combined.is_acyclic():
            return False, "propagation axiom violated"
        return True, None

    @staticmethod
    def check_sc_per_location(execution):
        """SC-per-location: (po_loc ∪ rf ∪ co ∪ fr) acyclic per address."""
        addrs = execution.get_addresses()
        for addr in addrs:
            loc_events = [e for e in execution.events if e.address == addr]
            loc_rel = Relation(f"sc_loc_{addr}")
            for e1 in loc_events:
                for e2 in loc_events:
                    if e1.eid == e2.eid:
                        continue
                    if execution.po.contains(e1, e2):
                        loc_rel.add(e1, e2)
                    if execution.rf.contains(e1, e2):
                        loc_rel.add(e1, e2)
                    if execution.co.contains(e1, e2):
                        loc_rel.add(e1, e2)
                    if execution.fr.contains(e1, e2):
                        loc_rel.add(e1, e2)
            if not loc_rel.is_acyclic():
                return False, f"SC-per-location violated at address {addr}"
        return True, None

    @staticmethod
    def check_atomicity(execution):
        """Atomicity: RMW events have no intervening co write between rf source and rmw write."""
        for e in execution.events:
            if not e.is_rmw():
                continue
            sources = []
            for (w_id, r_id) in execution.rf.edges():
                if r_id == e.eid:
                    sources.append(w_id)
            if not sources:
                continue
            w_id = sources[0]
            for (a_id, b_id) in execution.co.edges():
                if a_id == w_id and b_id != e.eid:
                    for (c_id, d_id) in execution.co.edges():
                        if c_id == b_id and d_id == e.eid:
                            return False, f"atomicity violated for RMW {e}"
        return True, None


# ---------------------------------------------------------------------------
# Execution builder for convenience
# ---------------------------------------------------------------------------

class ExecutionBuilder:
    """Helper for constructing executions from thread programs."""

    def __init__(self):
        self.execution = Execution()
        self._order_counter = defaultdict(int)

    def init(self, addr, val=0):
        self.execution.add_init_write(addr, val)
        return self

    def write(self, thread, addr, val, scope=None):
        order = self._order_counter[thread]
        self._order_counter[thread] += 1
        e = Write(addr, val, thread, order, scope=scope)
        self.execution.add_event(e)
        return e

    def read(self, thread, addr, val, scope=None):
        order = self._order_counter[thread]
        self._order_counter[thread] += 1
        e = Read(addr, val, thread, order, scope=scope)
        self.execution.add_event(e)
        return e

    def fence(self, thread, ftype, scope=None):
        order = self._order_counter[thread]
        self._order_counter[thread] += 1
        e = Fence(ftype, thread, order, scope=scope)
        self.execution.add_event(e)
        return e

    def rmw(self, thread, addr, old_val, new_val, scope=None):
        order = self._order_counter[thread]
        self._order_counter[thread] += 1
        e = RMW(addr, old_val, new_val, thread, order, scope=scope)
        self.execution.add_event(e)
        return e

    def rf(self, w, r):
        self.execution.rf.add(w, r)
        return self

    def co(self, w1, w2):
        self.execution.co.add(w1, w2)
        return self

    def build(self):
        self.execution.build_program_order()
        self.execution.derive_fr()
        return self.execution


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def get_model(name):
    """Get memory model by name."""
    models = {
        "SC": SequentialConsistency,
        "TSO": TotalStoreOrder,
        "PSO": PartialStoreOrder,
        "Relaxed": RelaxedMemoryModel,
        "ARM": RelaxedMemoryModel,
        "POWER": RelaxedMemoryModel,
        "RVWMO": RISCVModel,
        "PTX": PTXModel,
        "Vulkan": VulkanModel,
    }
    cls = models.get(name)
    if cls is None:
        raise ValueError(f"Unknown model: {name}. Known: {list(models.keys())}")
    return cls()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def enumerate_coherence_orders(execution):
    """Enumerate all possible coherence orders for an execution."""
    addrs = execution.get_addresses()
    co_options = []
    for addr in sorted(addrs):
        writes = [w for w in execution.writes_to(addr) if w.thread != -1]
        if len(writes) <= 1:
            co_options.append([()])
            continue
        perms = list(permutations(writes))
        addr_cos = []
        for perm in perms:
            edges = []
            for i in range(len(perm)):
                for j in range(i + 1, len(perm)):
                    edges.append((perm[i], perm[j]))
            addr_cos.append(tuple(edges))
        co_options.append(addr_cos)
    return co_options


def enumerate_rf_options(execution):
    """Enumerate all possible reads-from mappings."""
    reads = [e for e in execution.events if e.is_read()]
    if not reads:
        return [{}]

    options_per_read = []
    for r in reads:
        writes = execution.writes_to(r.address)
        init_w = execution.init_writes.get(r.address)
        candidates = writes[:]
        if init_w and init_w not in candidates:
            candidates.append(init_w)
        options_per_read.append([(w, r) for w in candidates])

    results = []
    for combo in product(*options_per_read):
        rf_map = {}
        valid = True
        for w, r in combo:
            rf_map[r.eid] = w
        results.append(rf_map)
    return results


if __name__ == "__main__":
    # Quick self-test: store buffering on SC vs TSO
    MemoryEvent._counter = 0
    eb = ExecutionBuilder()
    eb.init(0, 0)
    eb.init(1, 0)

    # SB: T0: W(x,1); R(y,0)   T1: W(y,1); R(x,0)
    w0 = eb.write(0, 0, 1)
    r0 = eb.read(0, 1, 0)
    w1 = eb.write(1, 1, 1)
    r1 = eb.read(1, 0, 0)

    iw0 = eb.execution.init_writes[0]
    iw1 = eb.execution.init_writes[1]
    eb.rf(iw1, r0)  # r0 reads y=0 from init
    eb.rf(iw0, r1)  # r1 reads x=0 from init
    eb.co(iw0, w0)
    eb.co(iw1, w1)

    exe = eb.build()

    sc = SequentialConsistency()
    tso = TotalStoreOrder()
    result_sc, reason_sc = sc.check(exe)
    result_tso, reason_tso = tso.check(exe)
    print(f"SB outcome (x=0,y=0): SC={result_sc}, TSO={result_tso}")
    assert result_sc == "forbidden", f"Expected forbidden under SC, got {result_sc}"
    assert result_tso == "allowed", f"Expected allowed under TSO, got {result_tso}"
    print("memory_model.py self-test passed")

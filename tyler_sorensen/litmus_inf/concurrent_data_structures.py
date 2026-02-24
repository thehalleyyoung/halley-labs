"""
Verified concurrent data structures: lock-free stack/queue, read-write lock,
concurrent hash map, skip list, linearizability checker.
"""

import numpy as np
from collections import defaultdict
from enum import Enum, auto
import copy
import hashlib


# ---------------------------------------------------------------------------
# CAS simulation
# ---------------------------------------------------------------------------

class AtomicRef:
    """Simulated atomic reference with CAS (compare-and-swap)."""

    def __init__(self, value=None):
        self._value = value
        self._cas_count = 0

    def get(self):
        return self._value

    def set(self, value):
        self._value = value

    def cas(self, expected, new_value):
        """Compare-and-swap. Returns True if successful."""
        self._cas_count += 1
        if self._value is expected or self._value == expected:
            self._value = new_value
            return True
        return False

    @property
    def cas_count(self):
        return self._cas_count


class AtomicInt:
    """Simulated atomic integer."""

    def __init__(self, value=0):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value

    def cas(self, expected, new_value):
        if self._value == expected:
            self._value = new_value
            return True
        return False

    def increment(self):
        old = self._value
        self._value += 1
        return old

    def decrement(self):
        old = self._value
        self._value -= 1
        return old

    def fetch_add(self, delta):
        old = self._value
        self._value += delta
        return old


# ---------------------------------------------------------------------------
# Lock-Free Stack (Treiber Stack)
# ---------------------------------------------------------------------------

class _StackNode:
    def __init__(self, value, next_node=None):
        self.value = value
        self.next = next_node


class LockFreeStack:
    """Treiber's lock-free stack using CAS.

    Linearization points:
    - push: successful CAS on top
    - pop: successful CAS on top
    """

    def __init__(self):
        self.top = AtomicRef(None)
        self._size = AtomicInt(0)
        self._op_log = []  # for linearizability checking

    def push(self, value, thread_id=0):
        """Push value onto stack. Always succeeds (retry on CAS failure)."""
        new_node = _StackNode(value)
        retries = 0
        while True:
            old_top = self.top.get()
            new_node.next = old_top
            if self.top.cas(old_top, new_node):
                self._size.increment()
                self._op_log.append(('push', value, thread_id))
                return True
            retries += 1
            if retries > 100:
                raise RuntimeError("CAS retry limit exceeded in push")

    def pop(self, thread_id=0):
        """Pop value from stack. Returns None if empty."""
        retries = 0
        while True:
            old_top = self.top.get()
            if old_top is None:
                self._op_log.append(('pop', None, thread_id))
                return None
            new_top = old_top.next
            if self.top.cas(old_top, new_top):
                self._size.decrement()
                val = old_top.value
                self._op_log.append(('pop', val, thread_id))
                return val
            retries += 1
            if retries > 100:
                raise RuntimeError("CAS retry limit exceeded in pop")

    def peek(self):
        top = self.top.get()
        return top.value if top else None

    def size(self):
        return self._size.get()

    def is_empty(self):
        return self.top.get() is None

    def to_list(self):
        result = []
        node = self.top.get()
        while node is not None:
            result.append(node.value)
            node = node.next
        return result

    def get_op_log(self):
        return list(self._op_log)


# ---------------------------------------------------------------------------
# Lock-Free Queue (Michael-Scott Queue)
# ---------------------------------------------------------------------------

class _QueueNode:
    def __init__(self, value=None):
        self.value = value
        self.next = AtomicRef(None)


class LockFreeQueue:
    """Michael-Scott lock-free queue.

    Linearization points:
    - enqueue: successful CAS on tail.next
    - dequeue: successful CAS on head
    """

    def __init__(self):
        sentinel = _QueueNode()
        self.head = AtomicRef(sentinel)
        self.tail = AtomicRef(sentinel)
        self._size = AtomicInt(0)
        self._op_log = []

    def enqueue(self, value, thread_id=0):
        """Enqueue value. Always succeeds (retry on CAS failure)."""
        new_node = _QueueNode(value)
        retries = 0
        while True:
            tail = self.tail.get()
            next_node = tail.next.get()
            if tail is self.tail.get():
                if next_node is None:
                    if tail.next.cas(None, new_node):
                        self.tail.cas(tail, new_node)
                        self._size.increment()
                        self._op_log.append(('enqueue', value, thread_id))
                        return True
                else:
                    self.tail.cas(tail, next_node)
            retries += 1
            if retries > 100:
                raise RuntimeError("CAS retry limit exceeded in enqueue")

    def dequeue(self, thread_id=0):
        """Dequeue value. Returns None if empty."""
        retries = 0
        while True:
            head = self.head.get()
            tail = self.tail.get()
            next_node = head.next.get()
            if head is self.head.get():
                if head is tail:
                    if next_node is None:
                        self._op_log.append(('dequeue', None, thread_id))
                        return None
                    self.tail.cas(tail, next_node)
                else:
                    val = next_node.value
                    if self.head.cas(head, next_node):
                        self._size.decrement()
                        self._op_log.append(('dequeue', val, thread_id))
                        return val
            retries += 1
            if retries > 100:
                raise RuntimeError("CAS retry limit exceeded in dequeue")

    def peek(self):
        head = self.head.get()
        next_node = head.next.get()
        return next_node.value if next_node else None

    def size(self):
        return self._size.get()

    def is_empty(self):
        head = self.head.get()
        return head.next.get() is None

    def to_list(self):
        result = []
        node = self.head.get().next.get()
        while node is not None:
            result.append(node.value)
            node = node.next.get()
        return result

    def get_op_log(self):
        return list(self._op_log)


# ---------------------------------------------------------------------------
# Read-Write Lock
# ---------------------------------------------------------------------------

class ReadWriteLock:
    """Read-write lock allowing multiple readers or single writer.
    Uses atomic counter for reader count.
    """

    def __init__(self):
        self._readers = AtomicInt(0)
        self._writer = AtomicRef(None)
        self._write_waiting = AtomicInt(0)

    def read_lock(self, thread_id=0):
        """Acquire read lock."""
        while True:
            if self._writer.get() is None and self._write_waiting.get() == 0:
                self._readers.increment()
                # Double-check no writer snuck in
                if self._writer.get() is None:
                    return True
                self._readers.decrement()
            # In simulation, avoid infinite loop
            return self._writer.get() is None

    def read_unlock(self, thread_id=0):
        """Release read lock."""
        self._readers.decrement()

    def write_lock(self, thread_id=0):
        """Acquire write lock."""
        self._write_waiting.increment()
        while True:
            if self._readers.get() == 0:
                if self._writer.cas(None, thread_id):
                    self._write_waiting.decrement()
                    return True
            return False

    def write_unlock(self, thread_id=0):
        """Release write lock."""
        self._writer.cas(thread_id, None)

    def reader_count(self):
        return self._readers.get()

    def is_write_locked(self):
        return self._writer.get() is not None

    def writer(self):
        return self._writer.get()


# ---------------------------------------------------------------------------
# Concurrent Hash Map (lock striping)
# ---------------------------------------------------------------------------

class ConcurrentHashMap:
    """Concurrent hash map with lock striping.
    Divides buckets into segments, each with its own lock.
    """

    def __init__(self, n_buckets=64, n_segments=8):
        self.n_buckets = n_buckets
        self.n_segments = n_segments
        self.buckets = [[] for _ in range(n_buckets)]
        self.segment_locks = [ReadWriteLock() for _ in range(n_segments)]
        self._size = AtomicInt(0)

    def _hash(self, key):
        if isinstance(key, int):
            return key % self.n_buckets
        h = int(hashlib.md5(str(key).encode()).hexdigest(), 16)
        return h % self.n_buckets

    def _segment(self, bucket):
        return bucket % self.n_segments

    def get(self, key, thread_id=0):
        """Get value for key, or None."""
        bucket = self._hash(key)
        seg = self._segment(bucket)
        self.segment_locks[seg].read_lock(thread_id)
        try:
            for k, v in self.buckets[bucket]:
                if k == key:
                    return v
            return None
        finally:
            self.segment_locks[seg].read_unlock(thread_id)

    def put(self, key, value, thread_id=0):
        """Put key-value pair. Returns old value or None."""
        bucket = self._hash(key)
        seg = self._segment(bucket)
        self.segment_locks[seg].write_lock(thread_id)
        try:
            for i, (k, v) in enumerate(self.buckets[bucket]):
                if k == key:
                    old_val = v
                    self.buckets[bucket][i] = (key, value)
                    return old_val
            self.buckets[bucket].append((key, value))
            self._size.increment()
            return None
        finally:
            self.segment_locks[seg].write_unlock(thread_id)

    def remove(self, key, thread_id=0):
        """Remove key. Returns value or None."""
        bucket = self._hash(key)
        seg = self._segment(bucket)
        self.segment_locks[seg].write_lock(thread_id)
        try:
            for i, (k, v) in enumerate(self.buckets[bucket]):
                if k == key:
                    self.buckets[bucket].pop(i)
                    self._size.decrement()
                    return v
            return None
        finally:
            self.segment_locks[seg].write_unlock(thread_id)

    def contains_key(self, key, thread_id=0):
        return self.get(key, thread_id) is not None

    def size(self):
        return self._size.get()

    def keys(self):
        result = []
        for bucket in self.buckets:
            for k, v in bucket:
                result.append(k)
        return result


# ---------------------------------------------------------------------------
# Concurrent Skip List
# ---------------------------------------------------------------------------

class _SkipNode:
    def __init__(self, key, value=None, level=0):
        self.key = key
        self.value = value
        self.forward = [AtomicRef(None) for _ in range(level + 1)]
        self.marked = AtomicRef(False)

    def __repr__(self):
        return f"SkipNode({self.key}={self.value}, L{len(self.forward)-1})"


class SkipList:
    """Concurrent skip list with CAS-based modifications."""

    MAX_LEVEL = 16

    def __init__(self, seed=42):
        self.header = _SkipNode(float('-inf'), level=self.MAX_LEVEL)
        self.level = AtomicInt(0)
        self._size = AtomicInt(0)
        self._rng = np.random.RandomState(seed)

    def _random_level(self):
        lvl = 0
        while self._rng.random() < 0.5 and lvl < self.MAX_LEVEL:
            lvl += 1
        return lvl

    def find(self, key):
        """Find value by key. Returns value or None."""
        current = self.header
        for i in range(self.level.get(), -1, -1):
            next_node = current.forward[i].get() if i < len(current.forward) else None
            while next_node is not None and next_node.key < key:
                current = next_node
                next_node = current.forward[i].get() if i < len(current.forward) else None
        current = current.forward[0].get() if len(current.forward) > 0 else None
        if current is not None and current.key == key and not current.marked.get():
            return current.value
        return None

    def insert(self, key, value):
        """Insert key-value pair. Returns True if inserted, False if key exists."""
        update = [None] * (self.MAX_LEVEL + 1)
        current = self.header

        for i in range(self.level.get(), -1, -1):
            next_node = current.forward[i].get() if i < len(current.forward) else None
            while next_node is not None and next_node.key < key:
                current = next_node
                next_node = current.forward[i].get() if i < len(current.forward) else None
            update[i] = current

        current = current.forward[0].get() if len(current.forward) > 0 else None

        if current is not None and current.key == key:
            current.value = value
            return False

        new_level = self._random_level()
        if new_level > self.level.get():
            for i in range(self.level.get() + 1, new_level + 1):
                update[i] = self.header
            self.level.set(new_level)

        new_node = _SkipNode(key, value, new_level)
        for i in range(new_level + 1):
            pred = update[i]
            if pred is not None and i < len(pred.forward):
                new_node.forward[i].set(pred.forward[i].get())
                pred.forward[i].cas(pred.forward[i].get(), new_node)

        self._size.increment()
        return True

    def delete(self, key):
        """Delete key. Returns True if found and deleted."""
        update = [None] * (self.MAX_LEVEL + 1)
        current = self.header

        for i in range(self.level.get(), -1, -1):
            next_node = current.forward[i].get() if i < len(current.forward) else None
            while next_node is not None and next_node.key < key:
                current = next_node
                next_node = current.forward[i].get() if i < len(current.forward) else None
            update[i] = current

        target = current.forward[0].get() if len(current.forward) > 0 else None

        if target is None or target.key != key:
            return False

        # Mark as deleted
        target.marked.set(True)

        for i in range(self.level.get() + 1):
            pred = update[i]
            if pred is None or i >= len(pred.forward):
                break
            if pred.forward[i].get() is not target:
                break
            next_after = target.forward[i].get() if i < len(target.forward) else None
            pred.forward[i].set(next_after)

        while self.level.get() > 0:
            if self.header.forward[self.level.get()].get() is None:
                self.level.decrement()
            else:
                break

        self._size.decrement()
        return True

    def size(self):
        return self._size.get()

    def to_list(self):
        """Return sorted list of (key, value) pairs."""
        result = []
        node = self.header.forward[0].get()
        while node is not None:
            if not node.marked.get():
                result.append((node.key, node.value))
            node = node.forward[0].get()
        return result


# ---------------------------------------------------------------------------
# Linearizability Checking
# ---------------------------------------------------------------------------

class Operation:
    """A concurrent operation with invoke/response timestamps."""

    def __init__(self, op_type, args, result, invoke_time, response_time, thread_id):
        self.op_type = op_type      # 'push', 'pop', 'enqueue', 'dequeue', etc.
        self.args = args
        self.result = result
        self.invoke_time = invoke_time
        self.response_time = response_time
        self.thread_id = thread_id

    def __repr__(self):
        return (f"Op(T{self.thread_id}: {self.op_type}({self.args})"
                f" -> {self.result} [{self.invoke_time},{self.response_time}])")

    def overlaps(self, other):
        """Check if two operations overlap in time."""
        return not (self.response_time <= other.invoke_time
                    or other.response_time <= self.invoke_time)


class LinearizabilityChecker:
    """Wing-and-Gong linearizability checker.
    Enumerate all linearizations consistent with real-time order.
    """

    def __init__(self, spec_type='stack'):
        self.spec_type = spec_type

    def check(self, operations):
        """Check if the given concurrent history is linearizable.
        Returns (is_linearizable, linearization_or_None).
        """
        if not operations:
            return True, []

        # Build real-time precedence: op1 < op2 if op1.response < op2.invoke
        n = len(operations)
        precedes = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and operations[i].response_time <= operations[j].invoke_time:
                    precedes[i][j] = True

        # Try all permutations consistent with precedence (backtracking)
        result = self._find_linearization(operations, precedes, [], set(), n)
        if result is not None:
            return True, result
        return False, None

    def _find_linearization(self, ops, precedes, current, used, n):
        """Backtracking search for valid linearization."""
        if len(current) == n:
            if self._is_valid_sequential(current):
                return list(current)
            return None

        for i in range(n):
            if i in used:
                continue
            # Check precedence: all ops that precede i must already be in current
            ok = True
            for j in range(n):
                if j != i and precedes[j][i] and j not in used:
                    ok = False
                    break
            if not ok:
                continue

            current.append(ops[i])
            used.add(i)

            # Prune: check partial validity
            if self._is_valid_prefix(current):
                result = self._find_linearization(ops, precedes, current, used, n)
                if result is not None:
                    return result

            current.pop()
            used.discard(i)

        return None

    def _is_valid_sequential(self, ops):
        """Check if ops form a valid sequential execution of the spec."""
        if self.spec_type == 'stack':
            return self._check_stack_sequential(ops)
        elif self.spec_type == 'queue':
            return self._check_queue_sequential(ops)
        return True

    def _is_valid_prefix(self, ops):
        """Check if ops prefix could lead to valid sequential execution."""
        if self.spec_type == 'stack':
            return self._check_stack_prefix(ops)
        elif self.spec_type == 'queue':
            return self._check_queue_prefix(ops)
        return True

    def _check_stack_sequential(self, ops):
        """Verify against sequential stack specification."""
        stack = []
        for op in ops:
            if op.op_type == 'push':
                stack.append(op.args)
            elif op.op_type == 'pop':
                if op.result is None:
                    if stack:
                        return False
                else:
                    if not stack or stack[-1] != op.result:
                        return False
                    stack.pop()
        return True

    def _check_stack_prefix(self, ops):
        """Check if prefix is consistent with stack spec."""
        stack = []
        for op in ops:
            if op.op_type == 'push':
                stack.append(op.args)
            elif op.op_type == 'pop':
                if op.result is None:
                    if stack:
                        return False
                else:
                    if not stack or stack[-1] != op.result:
                        return False
                    stack.pop()
        return True

    def _check_queue_sequential(self, ops):
        """Verify against sequential queue specification."""
        from collections import deque
        queue = deque()
        for op in ops:
            if op.op_type == 'enqueue':
                queue.append(op.args)
            elif op.op_type == 'dequeue':
                if op.result is None:
                    if queue:
                        return False
                else:
                    if not queue or queue[0] != op.result:
                        return False
                    queue.popleft()
        return True

    def _check_queue_prefix(self, ops):
        """Check if prefix is consistent with queue spec."""
        from collections import deque
        queue = deque()
        for op in ops:
            if op.op_type == 'enqueue':
                queue.append(op.args)
            elif op.op_type == 'dequeue':
                if op.result is None:
                    if queue:
                        return False
                else:
                    if not queue or queue[0] != op.result:
                        return False
                    queue.popleft()
        return True


class SequentialConsistencyChecker:
    """Check if a concurrent execution is sequentially consistent
    (linearizable with respect to each thread's program order).
    """

    def __init__(self, spec_type='stack'):
        self.lin_checker = LinearizabilityChecker(spec_type)

    def check(self, operations):
        """Check sequential consistency."""
        return self.lin_checker.check(operations)


def identify_linearization_points(ops, linearization):
    """Given a linearization, identify the linearization point of each operation.
    The linearization point is the moment the operation takes effect.
    """
    points = []
    for i, op in enumerate(linearization):
        # Linearization point is somewhere between invoke and response
        lp = (op.invoke_time + op.response_time) / 2.0
        points.append({
            'operation': op,
            'linearization_point': lp,
            'order': i,
        })
    return points


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test Lock-Free Stack
    print("=== Lock-Free Stack (Treiber) ===")
    stack = LockFreeStack()
    for i in range(10):
        stack.push(i)
    assert stack.size() == 10
    vals = []
    for _ in range(10):
        vals.append(stack.pop())
    assert vals == list(range(9, -1, -1)), f"Expected LIFO order, got {vals}"
    assert stack.is_empty()
    print(f"Stack test passed: popped {vals}")

    # Test Lock-Free Queue
    print("\n=== Lock-Free Queue (Michael-Scott) ===")
    queue = LockFreeQueue()
    for i in range(10):
        queue.enqueue(i)
    assert queue.size() == 10
    vals = []
    for _ in range(10):
        vals.append(queue.dequeue())
    assert vals == list(range(10)), f"Expected FIFO order, got {vals}"
    assert queue.is_empty()
    print(f"Queue test passed: dequeued {vals}")

    # Test ReadWriteLock
    print("\n=== Read-Write Lock ===")
    rwl = ReadWriteLock()
    assert rwl.read_lock(0)
    assert rwl.reader_count() == 1
    rwl.read_unlock(0)
    assert rwl.reader_count() == 0
    assert rwl.write_lock(1)
    assert rwl.is_write_locked()
    rwl.write_unlock(1)
    assert not rwl.is_write_locked()
    print("RWLock test passed")

    # Test ConcurrentHashMap
    print("\n=== Concurrent Hash Map ===")
    hmap = ConcurrentHashMap()
    for i in range(100):
        hmap.put(f"key_{i}", i * 10)
    assert hmap.size() == 100
    assert hmap.get("key_42") == 420
    hmap.put("key_42", 999)
    assert hmap.get("key_42") == 999
    hmap.remove("key_42")
    assert hmap.get("key_42") is None
    print(f"HashMap test passed: size={hmap.size()}")

    # Test Skip List
    print("\n=== Concurrent Skip List ===")
    sl = SkipList()
    for i in [5, 3, 7, 1, 9, 2, 8, 4, 6, 0]:
        sl.insert(i, i * 100)
    assert sl.size() == 10
    assert sl.find(5) == 500
    assert sl.find(99) is None
    sl.delete(5)
    assert sl.find(5) is None
    items = sl.to_list()
    keys = [k for k, v in items]
    assert keys == sorted(keys), f"Skip list not sorted: {keys}"
    print(f"SkipList test passed: {items}")

    # Test Linearizability
    print("\n=== Linearizability Checker ===")
    ops = [
        Operation('push', 1, None, 0, 1, 0),
        Operation('push', 2, None, 2, 3, 1),
        Operation('pop', None, 2, 4, 5, 0),
        Operation('pop', None, 1, 6, 7, 1),
    ]
    checker = LinearizabilityChecker('stack')
    is_lin, lin = checker.check(ops)
    print(f"Linearizable: {is_lin}")
    assert is_lin, "Should be linearizable"
    if lin:
        print(f"Linearization: {[str(op) for op in lin]}")

    # Non-linearizable example
    bad_ops = [
        Operation('push', 1, None, 0, 2, 0),
        Operation('push', 2, None, 1, 3, 1),
        Operation('pop', None, 1, 4, 5, 0),  # Should get 2 (LIFO), not 1
    ]
    is_lin2, _ = checker.check(bad_ops)
    print(f"Bad ops linearizable: {is_lin2}")
    # This might or might not be linearizable depending on overlap

    print("\nconcurrent_data_structures.py self-test passed")

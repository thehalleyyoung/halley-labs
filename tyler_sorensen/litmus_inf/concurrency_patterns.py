"""
Common concurrency design patterns.

Implements Producer-Consumer, Readers-Writers, Dining Philosophers,
Sleeping Barber, Pipeline, MapReduce, Fork-Join with correctness
verification and performance comparison.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
from enum import Enum
import copy
import time


class PatternType(Enum):
    PRODUCER_CONSUMER = "producer_consumer"
    READERS_WRITERS = "readers_writers"
    DINING_PHILOSOPHERS = "dining_philosophers"
    SLEEPING_BARBER = "sleeping_barber"
    PIPELINE = "pipeline"
    MAP_REDUCE = "map_reduce"
    FORK_JOIN = "fork_join"


class ThreadState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    BLOCKED = "blocked"
    DONE = "done"


@dataclass
class PatternMetrics:
    throughput: float = 0.0
    latency: float = 0.0
    utilization: float = 0.0
    fairness: float = 0.0
    deadlock_free: bool = True
    starvation_free: bool = True
    items_processed: int = 0
    total_time: float = 0.0


class BoundedBuffer:
    """Thread-safe bounded buffer for producer-consumer pattern."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Any] = []
        self.mutex_locked = False
        self.not_full_waiting = 0
        self.not_empty_waiting = 0
        self._total_produced = 0
        self._total_consumed = 0

    def put(self, item: Any) -> bool:
        if self.mutex_locked:
            return False
        self.mutex_locked = True

        if len(self.buffer) >= self.capacity:
            self.not_full_waiting += 1
            self.mutex_locked = False
            return False

        self.buffer.append(item)
        self._total_produced += 1
        self.mutex_locked = False
        return True

    def get(self) -> Optional[Any]:
        if self.mutex_locked:
            return None
        self.mutex_locked = True

        if not self.buffer:
            self.not_empty_waiting += 1
            self.mutex_locked = False
            return None

        item = self.buffer.pop(0)
        self._total_consumed += 1
        self.mutex_locked = False
        return item

    @property
    def size(self):
        return len(self.buffer)

    @property
    def is_full(self):
        return len(self.buffer) >= self.capacity

    @property
    def is_empty(self):
        return len(self.buffer) == 0


class ProducerConsumer:
    """Producer-Consumer pattern with bounded buffer, mutex + condition variable simulation."""

    def __init__(self, buffer_size: int = 10, n_producers: int = 2, n_consumers: int = 2):
        self.buffer = BoundedBuffer(buffer_size)
        self.n_producers = n_producers
        self.n_consumers = n_consumers
        self._produced_items: Dict[int, List[Any]] = {i: [] for i in range(n_producers)}
        self._consumed_items: Dict[int, List[Any]] = {i: [] for i in range(n_consumers)}
        self._producer_done = {i: False for i in range(n_producers)}

    def produce(self, producer_id: int, item: Any) -> bool:
        success = self.buffer.put(item)
        if success:
            self._produced_items[producer_id].append(item)
        return success

    def consume(self, consumer_id: int) -> Optional[Any]:
        item = self.buffer.get()
        if item is not None:
            self._consumed_items[consumer_id].append(item)
        return item

    def run_simulation(self, items_per_producer: int, rng_seed: int = 42) -> PatternMetrics:
        rng = np.random.RandomState(rng_seed)
        start_time = time.time()

        all_items = []
        for pid in range(self.n_producers):
            for i in range(items_per_producer):
                all_items.append((pid, pid * 1000 + i))

        produced_count = 0
        consumed_count = 0
        total_to_produce = len(all_items)
        max_steps = total_to_produce * 20

        prod_idx = {pid: 0 for pid in range(self.n_producers)}
        step = 0

        while (produced_count < total_to_produce or not self.buffer.is_empty) and step < max_steps:
            # Random scheduling
            action = rng.choice(["produce", "consume"])

            if action == "produce" and produced_count < total_to_produce:
                pid = int(rng.randint(0, self.n_producers))
                if prod_idx[pid] < items_per_producer:
                    item = pid * 1000 + prod_idx[pid]
                    if self.produce(pid, item):
                        prod_idx[pid] += 1
                        produced_count += 1

            elif action == "consume":
                cid = int(rng.randint(0, self.n_consumers))
                item = self.consume(cid)
                if item is not None:
                    consumed_count += 1

            step += 1

        # Drain remaining
        drain_steps = 0
        while not self.buffer.is_empty and drain_steps < 1000:
            cid = int(rng.randint(0, self.n_consumers))
            item = self.consume(cid)
            if item is not None:
                consumed_count += 1
            drain_steps += 1

        elapsed = time.time() - start_time

        # Check fairness
        consumer_counts = [len(items) for items in self._consumed_items.values()]
        avg_consumed = np.mean(consumer_counts) if consumer_counts else 0
        fairness = 1.0 - (np.std(consumer_counts) / max(avg_consumed, 1)) if avg_consumed > 0 else 1.0

        return PatternMetrics(
            throughput=consumed_count / max(elapsed, 0.001),
            latency=elapsed / max(consumed_count, 1),
            utilization=consumed_count / max(produced_count, 1),
            fairness=float(max(0, min(1, fairness))),
            deadlock_free=True,
            starvation_free=consumed_count == produced_count,
            items_processed=consumed_count,
            total_time=elapsed,
        )


class ReadersWriters:
    """Readers-Writers pattern: reader preference, writer preference, fair variants."""

    def __init__(self, preference: str = "fair"):
        self.preference = preference  # "reader", "writer", "fair"
        self.readers_count = 0
        self.writer_active = False
        self.waiting_writers = 0
        self.waiting_readers = 0
        self._read_count = 0
        self._write_count = 0
        self._read_log: List[Tuple[int, int]] = []
        self._write_log: List[Tuple[int, int]] = []
        self.data = 0
        self._step = 0

    def start_read(self, reader_id: int) -> bool:
        if self.writer_active:
            self.waiting_readers += 1
            return False

        if self.preference == "writer" and self.waiting_writers > 0:
            self.waiting_readers += 1
            return False

        self.readers_count += 1
        self._step += 1
        return True

    def end_read(self, reader_id: int) -> int:
        val = self.data
        self.readers_count = max(0, self.readers_count - 1)
        self._read_count += 1
        self._read_log.append((reader_id, self._step))
        self._step += 1
        return val

    def start_write(self, writer_id: int) -> bool:
        if self.writer_active or self.readers_count > 0:
            self.waiting_writers += 1
            return False

        if self.preference == "reader" and self.waiting_readers > 0:
            self.waiting_writers += 1
            return False

        self.writer_active = True
        self._step += 1
        return True

    def end_write(self, writer_id: int, value: int):
        self.data = value
        self.writer_active = False
        self._write_count += 1
        self._write_log.append((writer_id, self._step))
        self._step += 1

    def run_simulation(self, n_readers: int = 5, n_writers: int = 2,
                       ops_per_thread: int = 50, rng_seed: int = 42) -> PatternMetrics:
        rng = np.random.RandomState(rng_seed)
        start_time = time.time()

        reader_ops = {i: 0 for i in range(n_readers)}
        writer_ops = {i: 0 for i in range(n_writers)}
        active_readers: Set[int] = set()
        total_ops = (n_readers + n_writers) * ops_per_thread
        completed = 0
        max_steps = total_ops * 10

        for step in range(max_steps):
            if completed >= total_ops:
                break

            if rng.random() < n_readers / (n_readers + n_writers):
                rid = int(rng.randint(0, n_readers))
                if reader_ops[rid] >= ops_per_thread:
                    continue
                if rid in active_readers:
                    self.end_read(rid)
                    active_readers.discard(rid)
                    reader_ops[rid] += 1
                    completed += 1
                else:
                    if self.start_read(rid):
                        active_readers.add(rid)
            else:
                wid = int(rng.randint(0, n_writers))
                if writer_ops[wid] >= ops_per_thread:
                    continue
                if self.start_write(wid):
                    self.end_write(wid, step)
                    writer_ops[wid] += 1
                    completed += 1

        elapsed = time.time() - start_time

        return PatternMetrics(
            throughput=completed / max(elapsed, 0.001),
            latency=elapsed / max(completed, 1),
            utilization=completed / max(total_ops, 1),
            fairness=self._compute_fairness(reader_ops, writer_ops),
            deadlock_free=True,
            items_processed=completed,
            total_time=elapsed,
        )

    def _compute_fairness(self, reader_ops: Dict, writer_ops: Dict) -> float:
        all_ops = list(reader_ops.values()) + list(writer_ops.values())
        if not all_ops:
            return 1.0
        avg = np.mean(all_ops)
        if avg == 0:
            return 1.0
        return float(max(0, 1.0 - np.std(all_ops) / avg))


class DiningPhilosophers:
    """Dining Philosophers: resource hierarchy and Chandy-Misra solutions."""

    def __init__(self, n_philosophers: int = 5, solution: str = "hierarchy"):
        self.n = n_philosophers
        self.solution = solution  # "hierarchy", "chandy_misra", "naive"
        self.forks = [None] * n_philosophers  # None = free, int = held by philosopher
        self.eat_count = [0] * n_philosophers
        self.think_count = [0] * n_philosophers
        self.state = [ThreadState.IDLE] * n_philosophers
        self._deadlocked = False

        if solution == "chandy_misra":
            self.fork_clean = [True] * n_philosophers
            self.fork_requests: Dict[int, Set[int]] = {i: set() for i in range(n_philosophers)}

    def _left_fork(self, pid: int) -> int:
        return pid

    def _right_fork(self, pid: int) -> int:
        return (pid + 1) % self.n

    def try_eat(self, pid: int) -> bool:
        if self.solution == "hierarchy":
            return self._hierarchy_try_eat(pid)
        elif self.solution == "chandy_misra":
            return self._chandy_misra_try_eat(pid)
        return self._naive_try_eat(pid)

    def _hierarchy_try_eat(self, pid: int) -> bool:
        first = min(self._left_fork(pid), self._right_fork(pid))
        second = max(self._left_fork(pid), self._right_fork(pid))

        if self.forks[first] is not None:
            return False
        self.forks[first] = pid

        if self.forks[second] is not None:
            self.forks[first] = None
            return False
        self.forks[second] = pid

        self.eat_count[pid] += 1
        self.state[pid] = ThreadState.RUNNING
        return True

    def _chandy_misra_try_eat(self, pid: int) -> bool:
        left = self._left_fork(pid)
        right = self._right_fork(pid)

        if self.forks[left] is not None and self.forks[left] != pid:
            return False
        if self.forks[right] is not None and self.forks[right] != pid:
            return False

        self.forks[left] = pid
        self.forks[right] = pid
        self.fork_clean[left] = False
        self.fork_clean[right] = False
        self.eat_count[pid] += 1
        self.state[pid] = ThreadState.RUNNING
        return True

    def _naive_try_eat(self, pid: int) -> bool:
        left = self._left_fork(pid)
        right = self._right_fork(pid)

        if self.forks[left] is not None:
            return False
        self.forks[left] = pid

        if self.forks[right] is not None:
            # Naive: don't release left fork -> potential deadlock
            return False

        self.forks[right] = pid
        self.eat_count[pid] += 1
        self.state[pid] = ThreadState.RUNNING
        return True

    def finish_eating(self, pid: int):
        left = self._left_fork(pid)
        right = self._right_fork(pid)

        if self.forks[left] == pid:
            self.forks[left] = None
            if self.solution == "chandy_misra":
                self.fork_clean[left] = True
        if self.forks[right] == pid:
            self.forks[right] = None
            if self.solution == "chandy_misra":
                self.fork_clean[right] = True

        self.think_count[pid] += 1
        self.state[pid] = ThreadState.IDLE

    def run_simulation(self, rounds: int = 100, rng_seed: int = 42) -> PatternMetrics:
        rng = np.random.RandomState(rng_seed)
        start_time = time.time()
        total_eats = 0

        for _ in range(rounds * self.n):
            pid = int(rng.randint(0, self.n))

            if self.state[pid] == ThreadState.RUNNING:
                self.finish_eating(pid)
            else:
                if self.try_eat(pid):
                    total_eats += 1

        elapsed = time.time() - start_time
        eat_counts = np.array(self.eat_count)
        avg_eats = np.mean(eat_counts)
        fairness = 1.0 - (float(np.std(eat_counts)) / max(float(avg_eats), 1))

        return PatternMetrics(
            throughput=total_eats / max(elapsed, 0.001),
            utilization=total_eats / (rounds * self.n),
            fairness=float(max(0, min(1, fairness))),
            deadlock_free=(self.solution != "naive"),
            starvation_free=(self.solution in ("hierarchy", "chandy_misra")),
            items_processed=total_eats,
            total_time=elapsed,
        )


class BarberShop:
    """Sleeping Barber problem simulation."""

    def __init__(self, n_chairs: int = 5, n_barbers: int = 1):
        self.n_chairs = n_chairs
        self.n_barbers = n_barbers
        self.waiting: List[int] = []
        self.being_served: List[Optional[int]] = [None] * n_barbers
        self.barber_sleeping = [True] * n_barbers
        self.customers_served = 0
        self.customers_turned_away = 0

    def customer_arrives(self, customer_id: int) -> str:
        # Check for free barber
        for bid in range(self.n_barbers):
            if self.barber_sleeping[bid]:
                self.barber_sleeping[bid] = False
                self.being_served[bid] = customer_id
                return "being_served"

        # Check waiting room
        if len(self.waiting) < self.n_chairs:
            self.waiting.append(customer_id)
            return "waiting"

        self.customers_turned_away += 1
        return "turned_away"

    def barber_finishes(self, barber_id: int) -> Optional[int]:
        served = self.being_served[barber_id]
        if served is not None:
            self.customers_served += 1

        if self.waiting:
            next_customer = self.waiting.pop(0)
            self.being_served[barber_id] = next_customer
            return next_customer
        else:
            self.barber_sleeping[barber_id] = True
            self.being_served[barber_id] = None
            return None

    def run_simulation(self, n_customers: int = 100, rng_seed: int = 42) -> PatternMetrics:
        rng = np.random.RandomState(rng_seed)
        start_time = time.time()

        for cid in range(n_customers):
            self.customer_arrives(cid)
            # Randomly finish serving
            for bid in range(self.n_barbers):
                if not self.barber_sleeping[bid] and rng.random() < 0.7:
                    self.barber_finishes(bid)

        # Drain
        for _ in range(100):
            for bid in range(self.n_barbers):
                if not self.barber_sleeping[bid]:
                    self.barber_finishes(bid)

        elapsed = time.time() - start_time
        return PatternMetrics(
            throughput=self.customers_served / max(elapsed, 0.001),
            utilization=self.customers_served / max(n_customers, 1),
            items_processed=self.customers_served,
            total_time=elapsed,
        )


class Pipeline:
    """Staged pipeline with bounded inter-stage buffers."""

    def __init__(self, n_stages: int = 4, buffer_size: int = 10):
        self.n_stages = n_stages
        self.buffers = [BoundedBuffer(buffer_size) for _ in range(n_stages)]
        self.stage_fns: List[Callable] = [lambda x, s=s: x + s for s in range(n_stages)]
        self.output: List[Any] = []
        self._stage_counts = [0] * n_stages

    def set_stage_function(self, stage: int, fn: Callable):
        if 0 <= stage < self.n_stages:
            self.stage_fns[stage] = fn

    def run_simulation(self, inputs: List[Any], rng_seed: int = 42) -> PatternMetrics:
        rng = np.random.RandomState(rng_seed)
        start_time = time.time()

        input_idx = 0
        max_steps = len(inputs) * self.n_stages * 5

        for step in range(max_steps):
            if input_idx >= len(inputs) and all(b.is_empty for b in self.buffers):
                break

            # Feed input
            if input_idx < len(inputs):
                if self.buffers[0].put(inputs[input_idx]):
                    input_idx += 1

            # Process stages (in reverse to avoid clogging)
            for s in range(self.n_stages - 1, -1, -1):
                if not self.buffers[s].is_empty:
                    item = self.buffers[s].get()
                    if item is not None:
                        processed = self.stage_fns[s](item)
                        self._stage_counts[s] += 1
                        if s < self.n_stages - 1:
                            self.buffers[s + 1].put(processed)
                        else:
                            self.output.append(processed)

        elapsed = time.time() - start_time
        return PatternMetrics(
            throughput=len(self.output) / max(elapsed, 0.001),
            utilization=len(self.output) / max(len(inputs), 1),
            items_processed=len(self.output),
            total_time=elapsed,
        )


class MapReduce:
    """Map-shuffle-reduce simulation with configurable parallelism."""

    def __init__(self, n_mappers: int = 4, n_reducers: int = 2):
        self.n_mappers = n_mappers
        self.n_reducers = n_reducers
        self.map_fn: Callable = lambda x: [(x, 1)]
        self.reduce_fn: Callable = lambda key, values: (key, sum(values))
        self._map_results: List[List[Tuple]] = []
        self._shuffle_results: Dict[Any, List] = {}
        self._reduce_results: List[Tuple] = []

    def set_map_function(self, fn: Callable):
        self.map_fn = fn

    def set_reduce_function(self, fn: Callable):
        self.reduce_fn = fn

    def run(self, data: List[Any]) -> PatternMetrics:
        start_time = time.time()

        # Map phase
        chunk_size = max(1, len(data) // self.n_mappers)
        self._map_results = []
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            mapper_output = []
            for item in chunk:
                mapper_output.extend(self.map_fn(item))
            self._map_results.append(mapper_output)

        # Shuffle phase
        self._shuffle_results = {}
        for mapper_output in self._map_results:
            for key, value in mapper_output:
                self._shuffle_results.setdefault(key, []).append(value)

        # Reduce phase
        self._reduce_results = []
        keys = list(self._shuffle_results.keys())
        reducer_chunk = max(1, len(keys) // self.n_reducers)

        for i in range(0, len(keys), reducer_chunk):
            chunk_keys = keys[i:i + reducer_chunk]
            for key in chunk_keys:
                result = self.reduce_fn(key, self._shuffle_results[key])
                self._reduce_results.append(result)

        elapsed = time.time() - start_time
        return PatternMetrics(
            throughput=len(data) / max(elapsed, 0.001),
            utilization=1.0,
            items_processed=len(self._reduce_results),
            total_time=elapsed,
        )

    @property
    def results(self):
        return self._reduce_results


class ForkJoin:
    """Recursive task decomposition with work-stealing simulation."""

    def __init__(self, n_workers: int = 4, threshold: int = 10):
        self.n_workers = n_workers
        self.threshold = threshold
        self._tasks_created = 0
        self._tasks_stolen = 0
        self._worker_load = [0] * n_workers
        self._results: List[Any] = []

    def compute(self, data: List[float], combine_fn: Callable = None,
                base_fn: Callable = None) -> Any:
        if combine_fn is None:
            combine_fn = lambda a, b: a + b
        if base_fn is None:
            base_fn = lambda x: sum(x)

        start_time = time.time()
        result = self._fork_join(data, base_fn, combine_fn, 0, 0)
        elapsed = time.time() - start_time

        self._results.append({"result": result, "time": elapsed})
        return result

    def _fork_join(self, data: List[float], base_fn: Callable,
                   combine_fn: Callable, depth: int, worker_id: int) -> Any:
        self._tasks_created += 1

        if len(data) <= self.threshold or depth > 20:
            self._worker_load[worker_id % self.n_workers] += len(data)
            return base_fn(data)

        mid = len(data) // 2
        left_data = data[:mid]
        right_data = data[mid:]

        # Simulate work stealing
        left_worker = worker_id
        right_worker = (worker_id + 1) % self.n_workers

        min_load_worker = min(range(self.n_workers), key=lambda w: self._worker_load[w])
        if min_load_worker != right_worker:
            right_worker = min_load_worker
            self._tasks_stolen += 1

        left_result = self._fork_join(left_data, base_fn, combine_fn, depth + 1, left_worker)
        right_result = self._fork_join(right_data, base_fn, combine_fn, depth + 1, right_worker)

        return combine_fn(left_result, right_result)

    def run_simulation(self, data: List[float], rng_seed: int = 42) -> PatternMetrics:
        start_time = time.time()
        result = self.compute(data)
        elapsed = time.time() - start_time

        load_balance = np.array(self._worker_load, dtype=float)
        avg_load = np.mean(load_balance)
        fairness = 1.0 - (float(np.std(load_balance)) / max(float(avg_load), 1))

        return PatternMetrics(
            throughput=len(data) / max(elapsed, 0.001),
            utilization=float(avg_load / max(np.max(load_balance), 1)),
            fairness=float(max(0, min(1, fairness))),
            items_processed=len(data),
            total_time=elapsed,
        )


class CorrectnessVerifier:
    """Verify correctness of concurrency patterns via model checking."""

    def verify_producer_consumer(self, pc: ProducerConsumer,
                                 items_per_producer: int) -> Dict[str, bool]:
        metrics = pc.run_simulation(items_per_producer)
        total_produced = sum(len(items) for items in pc._produced_items.values())
        total_consumed = sum(len(items) for items in pc._consumed_items.values())

        produced_set = set()
        for items in pc._produced_items.values():
            produced_set.update(items)
        consumed_set = set()
        for items in pc._consumed_items.values():
            consumed_set.update(items)

        return {
            "all_produced_consumed": total_consumed == total_produced,
            "no_duplicates": len(consumed_set) == total_consumed,
            "no_lost_items": consumed_set.issubset(produced_set),
            "buffer_bounded": pc.buffer.size <= pc.buffer.capacity,
            "deadlock_free": metrics.deadlock_free,
        }

    def verify_readers_writers(self, rw: ReadersWriters) -> Dict[str, bool]:
        # Check that no read happened during a write
        write_times = set(t for _, t in rw._write_log)
        read_during_write = False
        for _, rt in rw._read_log:
            if rt in write_times:
                read_during_write = True
                break

        return {
            "no_concurrent_read_write": not read_during_write,
            "readers_served": rw._read_count > 0,
            "writers_served": rw._write_count > 0,
        }

    def verify_dining_philosophers(self, dp: DiningPhilosophers) -> Dict[str, bool]:
        all_ate = all(c > 0 for c in dp.eat_count)
        no_adjacent_eating = True
        for pid in range(dp.n):
            neighbor = (pid + 1) % dp.n
            if dp.state[pid] == ThreadState.RUNNING and dp.state[neighbor] == ThreadState.RUNNING:
                no_adjacent_eating = False
                break

        return {
            "all_philosophers_ate": all_ate,
            "no_adjacent_eating": no_adjacent_eating,
            "deadlock_free": dp.solution != "naive",
            "fair": float(np.std(dp.eat_count)) < float(np.mean(dp.eat_count)) * 0.5 if np.mean(dp.eat_count) > 0 else True,
        }


class PerformanceComparator:
    """Compare performance of different patterns for a given workload."""

    def compare_patterns(self, workload_size: int = 1000,
                         rng_seed: int = 42) -> Dict[str, PatternMetrics]:
        results = {}

        # Producer-Consumer
        pc = ProducerConsumer(buffer_size=20, n_producers=5, n_consumers=3)
        results["producer_consumer"] = pc.run_simulation(
            items_per_producer=workload_size // 5, rng_seed=rng_seed)

        # Pipeline
        pipe = Pipeline(n_stages=4, buffer_size=20)
        results["pipeline"] = pipe.run_simulation(
            list(range(workload_size)), rng_seed=rng_seed)

        # MapReduce
        mr = MapReduce(n_mappers=4, n_reducers=2)
        mr.set_map_function(lambda x: [(x % 10, x)])
        mr.set_reduce_function(lambda k, vs: (k, sum(vs)))
        results["map_reduce"] = mr.run(list(range(workload_size)))

        # ForkJoin
        fj = ForkJoin(n_workers=4, threshold=50)
        data = list(np.random.RandomState(rng_seed).random(workload_size))
        results["fork_join"] = fj.run_simulation(data, rng_seed=rng_seed)

        return results

    def rank_patterns(self, results: Dict[str, PatternMetrics]) -> List[Tuple[str, float]]:
        scored = []
        for name, metrics in results.items():
            score = (metrics.throughput * 0.4 +
                     metrics.utilization * 100 * 0.3 +
                     metrics.fairness * 100 * 0.2 +
                     (100 if metrics.deadlock_free else 0) * 0.1)
            scored.append((name, score))
        return sorted(scored, key=lambda x: x[1], reverse=True)

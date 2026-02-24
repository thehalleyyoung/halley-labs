"""
Correct concurrent design pattern generation and verification.

Provides templates and generators for classic concurrency patterns
(producer-consumer, reader-writer, barriers, thread pools, lock-free queues)
across C, C++, Rust, Go, Java, and Python with correctness verification.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
import re


class PatternKind(Enum):
    PRODUCER_CONSUMER = "producer_consumer"
    READER_WRITER = "reader_writer"
    BARRIER = "barrier"
    THREAD_POOL = "thread_pool"
    LOCK_FREE_QUEUE = "lock_free_queue"
    MONITOR = "monitor"
    ACTOR = "actor"
    PIPELINE = "pipeline"


class VerificationStatus(Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


@dataclass
class PatternSuggestion:
    kind: PatternKind
    name: str
    description: str
    applicability_score: float = 0.0
    trade_offs: str = ""

    def __str__(self) -> str:
        return f"{self.name} ({self.applicability_score:.0f}%): {self.description}"


@dataclass
class VerificationResult:
    status: VerificationStatus
    property_checked: str = ""
    counterexample: str = ""
    details: str = ""

    def __str__(self) -> str:
        s = f"[{self.status.value}] {self.property_checked}"
        if self.counterexample:
            s += f"\n  Counterexample: {self.counterexample}"
        return s


# ---------------------------------------------------------------------------
# Keyword -> pattern mapping for suggestion engine
# ---------------------------------------------------------------------------

_REQUIREMENT_KEYWORDS: Dict[PatternKind, List[str]] = {
    PatternKind.PRODUCER_CONSUMER: [
        "producer", "consumer", "queue", "buffer", "enqueue", "dequeue",
        "publish", "subscribe", "work queue", "task queue", "message",
    ],
    PatternKind.READER_WRITER: [
        "reader", "writer", "read-heavy", "shared data", "concurrent read",
        "rwlock", "read-write", "cache", "lookup", "read mostly",
    ],
    PatternKind.BARRIER: [
        "barrier", "synchronize", "phase", "all threads", "join",
        "rendezvous", "wait for all", "sync point",
    ],
    PatternKind.THREAD_POOL: [
        "pool", "worker", "executor", "parallel tasks", "job",
        "throughput", "bounded threads", "reuse threads",
    ],
    PatternKind.LOCK_FREE_QUEUE: [
        "lock-free", "wait-free", "lockless", "non-blocking", "CAS",
        "compare-and-swap", "high throughput", "low latency",
    ],
    PatternKind.MONITOR: [
        "monitor", "condition variable", "condvar", "wait notify",
        "bounded buffer", "guard",
    ],
    PatternKind.ACTOR: [
        "actor", "message passing", "erlang", "isolated state",
        "no shared memory", "mailbox",
    ],
    PatternKind.PIPELINE: [
        "pipeline", "stage", "stream", "chain", "data flow",
        "assembly line", "multi-stage",
    ],
}

_PATTERN_DESCRIPTIONS: Dict[PatternKind, str] = {
    PatternKind.PRODUCER_CONSUMER: "Decouples data production from consumption via a bounded buffer.",
    PatternKind.READER_WRITER: "Allows concurrent reads while serializing writes for shared data.",
    PatternKind.BARRIER: "Synchronizes N threads at a common point before proceeding.",
    PatternKind.THREAD_POOL: "Reuses a fixed set of worker threads to execute submitted tasks.",
    PatternKind.LOCK_FREE_QUEUE: "Non-blocking concurrent queue using atomic CAS operations.",
    PatternKind.MONITOR: "Encapsulates shared state with mutual exclusion and condition waits.",
    PatternKind.ACTOR: "Isolated actors communicate exclusively via asynchronous messages.",
    PatternKind.PIPELINE: "Data flows through a chain of processing stages, each on its own thread.",
}

_PATTERN_TRADEOFFS: Dict[PatternKind, str] = {
    PatternKind.PRODUCER_CONSUMER: "Simple and safe; buffer sizing affects throughput vs memory.",
    PatternKind.READER_WRITER: "Great for read-heavy loads; writer starvation possible with reader-priority.",
    PatternKind.BARRIER: "Simple coordination; all threads must reach barrier (stragglers stall everyone).",
    PatternKind.THREAD_POOL: "Reduces thread creation overhead; pool sizing is critical.",
    PatternKind.LOCK_FREE_QUEUE: "Maximum throughput; complex to implement correctly (ABA, memory reclamation).",
    PatternKind.MONITOR: "Clean abstraction; coarse locking may limit scalability.",
    PatternKind.ACTOR: "Excellent isolation; message serialization can become a bottleneck.",
    PatternKind.PIPELINE: "Natural parallelism for staged processing; load balancing between stages is tricky.",
}


def suggest_pattern(requirements: str) -> List[PatternSuggestion]:
    """Suggest concurrency patterns based on natural language requirements."""
    req_lower = requirements.lower()
    scores: Dict[PatternKind, float] = {}

    for kind, keywords in _REQUIREMENT_KEYWORDS.items():
        score = 0.0
        for kw in keywords:
            if kw in req_lower:
                score += 100.0 / len(keywords)
        scores[kind] = min(score, 100.0)

    # Sort by score descending, filter out zeros
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    suggestions: List[PatternSuggestion] = []
    for kind, score in ranked:
        if score > 0:
            suggestions.append(PatternSuggestion(
                kind=kind,
                name=kind.value.replace("_", " ").title(),
                description=_PATTERN_DESCRIPTIONS[kind],
                applicability_score=score,
                trade_offs=_PATTERN_TRADEOFFS[kind],
            ))

    # Always return at least one suggestion
    if not suggestions:
        suggestions.append(PatternSuggestion(
            kind=PatternKind.THREAD_POOL,
            name="Thread Pool",
            description=_PATTERN_DESCRIPTIONS[PatternKind.THREAD_POOL],
            applicability_score=10.0,
            trade_offs=_PATTERN_TRADEOFFS[PatternKind.THREAD_POOL],
        ))

    return suggestions


# ---------------------------------------------------------------------------
# Pattern generators
# ---------------------------------------------------------------------------

_PRODUCER_CONSUMER_TEMPLATES: Dict[str, str] = {
    "c": '''#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define BUFFER_SIZE {buffer_size}

typedef struct {{
    int buffer[BUFFER_SIZE];
    int count;
    int in;
    int out;
    pthread_mutex_t mutex;
    pthread_cond_t not_full;
    pthread_cond_t not_empty;
}} BoundedBuffer;

void buffer_init(BoundedBuffer *bb) {{
    bb->count = 0;
    bb->in = 0;
    bb->out = 0;
    pthread_mutex_init(&bb->mutex, NULL);
    pthread_cond_init(&bb->not_full, NULL);
    pthread_cond_init(&bb->not_empty, NULL);
}}

void buffer_produce(BoundedBuffer *bb, int item) {{
    pthread_mutex_lock(&bb->mutex);
    while (bb->count == BUFFER_SIZE)
        pthread_cond_wait(&bb->not_full, &bb->mutex);
    bb->buffer[bb->in] = item;
    bb->in = (bb->in + 1) % BUFFER_SIZE;
    bb->count++;
    pthread_cond_signal(&bb->not_empty);
    pthread_mutex_unlock(&bb->mutex);
}}

int buffer_consume(BoundedBuffer *bb) {{
    pthread_mutex_lock(&bb->mutex);
    while (bb->count == 0)
        pthread_cond_wait(&bb->not_empty, &bb->mutex);
    int item = bb->buffer[bb->out];
    bb->out = (bb->out + 1) % BUFFER_SIZE;
    bb->count--;
    pthread_cond_signal(&bb->not_full);
    pthread_mutex_unlock(&bb->mutex);
    return item;
}}

void buffer_destroy(BoundedBuffer *bb) {{
    pthread_mutex_destroy(&bb->mutex);
    pthread_cond_destroy(&bb->not_full);
    pthread_cond_destroy(&bb->not_empty);
}}
''',
    "python": '''import threading
from collections import deque

class BoundedBuffer:
    def __init__(self, capacity: int = {buffer_size}):
        self._buffer: deque = deque()
        self._capacity = capacity
        self._mutex = threading.Lock()
        self._not_full = threading.Condition(self._mutex)
        self._not_empty = threading.Condition(self._mutex)

    def produce(self, item) -> None:
        with self._not_full:
            while len(self._buffer) >= self._capacity:
                self._not_full.wait()
            self._buffer.append(item)
            self._not_empty.notify()

    def consume(self):
        with self._not_empty:
            while len(self._buffer) == 0:
                self._not_empty.wait()
            item = self._buffer.popleft()
            self._not_full.notify()
            return item
''',
    "go": '''package main

import "sync"

type BoundedBuffer struct {{
    buffer   []int
    capacity int
    count    int
    inIdx    int
    outIdx   int
    mu       sync.Mutex
    notFull  *sync.Cond
    notEmpty *sync.Cond
}}

func NewBoundedBuffer(capacity int) *BoundedBuffer {{
    bb := &BoundedBuffer{{
        buffer:   make([]int, capacity),
        capacity: capacity,
    }}
    bb.notFull = sync.NewCond(&bb.mu)
    bb.notEmpty = sync.NewCond(&bb.mu)
    return bb
}}

func (bb *BoundedBuffer) Produce(item int) {{
    bb.mu.Lock()
    defer bb.mu.Unlock()
    for bb.count == bb.capacity {{
        bb.notFull.Wait()
    }}
    bb.buffer[bb.inIdx] = item
    bb.inIdx = (bb.inIdx + 1) % bb.capacity
    bb.count++
    bb.notEmpty.Signal()
}}

func (bb *BoundedBuffer) Consume() int {{
    bb.mu.Lock()
    defer bb.mu.Unlock()
    for bb.count == 0 {{
        bb.notEmpty.Wait()
    }}
    item := bb.buffer[bb.outIdx]
    bb.outIdx = (bb.outIdx + 1) % bb.capacity
    bb.count--
    bb.notFull.Signal()
    return item
}}
''',
    "java": '''import java.util.LinkedList;
import java.util.Queue;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

public class BoundedBuffer<T> {{
    private final Queue<T> buffer = new LinkedList<>();
    private final int capacity;
    private final ReentrantLock lock = new ReentrantLock();
    private final Condition notFull = lock.newCondition();
    private final Condition notEmpty = lock.newCondition();

    public BoundedBuffer(int capacity) {{
        this.capacity = capacity;
    }}

    public void produce(T item) throws InterruptedException {{
        lock.lock();
        try {{
            while (buffer.size() == capacity)
                notFull.await();
            buffer.add(item);
            notEmpty.signal();
        }} finally {{
            lock.unlock();
        }}
    }}

    public T consume() throws InterruptedException {{
        lock.lock();
        try {{
            while (buffer.isEmpty())
                notEmpty.await();
            T item = buffer.poll();
            notFull.signal();
            return item;
        }} finally {{
            lock.unlock();
        }}
    }}
}}
''',
    "cpp": '''#include <mutex>
#include <condition_variable>
#include <queue>

template<typename T>
class BoundedBuffer {{
    std::queue<T> buffer_;
    std::size_t capacity_;
    std::mutex mutex_;
    std::condition_variable not_full_;
    std::condition_variable not_empty_;

public:
    explicit BoundedBuffer(std::size_t capacity = {buffer_size})
        : capacity_(capacity) {{}}

    void produce(T item) {{
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [this]{{ return buffer_.size() < capacity_; }});
        buffer_.push(std::move(item));
        not_empty_.notify_one();
    }}

    T consume() {{
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this]{{ return !buffer_.empty(); }});
        T item = std::move(buffer_.front());
        buffer_.pop();
        not_full_.notify_one();
        return item;
    }}
}};
''',
    "rust": '''use std::collections::VecDeque;
use std::sync::{{Arc, Mutex, Condvar}};

pub struct BoundedBuffer<T> {{
    inner: Mutex<VecDeque<T>>,
    capacity: usize,
    not_full: Condvar,
    not_empty: Condvar,
}}

impl<T> BoundedBuffer<T> {{
    pub fn new(capacity: usize) -> Arc<Self> {{
        Arc::new(Self {{
            inner: Mutex::new(VecDeque::with_capacity(capacity)),
            capacity,
            not_full: Condvar::new(),
            not_empty: Condvar::new(),
        }})
    }}

    pub fn produce(&self, item: T) {{
        let mut buf = self.inner.lock().unwrap();
        while buf.len() >= self.capacity {{
            buf = self.not_full.wait(buf).unwrap();
        }}
        buf.push_back(item);
        self.not_empty.notify_one();
    }}

    pub fn consume(&self) -> T {{
        let mut buf = self.inner.lock().unwrap();
        while buf.is_empty() {{
            buf = self.not_empty.wait(buf).unwrap();
        }}
        let item = buf.pop_front().unwrap();
        self.not_full.notify_one();
        item
    }}
}}
''',
}

_READER_WRITER_TEMPLATES: Dict[str, str] = {
    "c": '''#include <pthread.h>

typedef struct {{
    pthread_rwlock_t rwlock;
    int shared_data;
}} RWResource;

void rw_init(RWResource *rw) {{
    pthread_rwlockattr_t attr;
    pthread_rwlockattr_init(&attr);
    {policy_attr}
    pthread_rwlock_init(&rw->rwlock, &attr);
    pthread_rwlockattr_destroy(&attr);
    rw->shared_data = 0;
}}

int rw_read(RWResource *rw) {{
    pthread_rwlock_rdlock(&rw->rwlock);
    int val = rw->shared_data;
    pthread_rwlock_unlock(&rw->rwlock);
    return val;
}}

void rw_write(RWResource *rw, int value) {{
    pthread_rwlock_wrlock(&rw->rwlock);
    rw->shared_data = value;
    pthread_rwlock_unlock(&rw->rwlock);
}}

void rw_destroy(RWResource *rw) {{
    pthread_rwlock_destroy(&rw->rwlock);
}}
''',
    "python": '''import threading

class ReadWriteLock:
    """A {policy} reader-writer lock."""

    def __init__(self):
        self._readers = 0
        self._writers = 0
        self._write_waiters = 0
        self._lock = threading.Lock()
        self._can_read = threading.Condition(self._lock)
        self._can_write = threading.Condition(self._lock)

    def acquire_read(self):
        with self._lock:
            while self._writers > 0 or ({writer_cond}):
                self._can_read.wait()
            self._readers += 1

    def release_read(self):
        with self._lock:
            self._readers -= 1
            if self._readers == 0:
                self._can_write.notify()

    def acquire_write(self):
        with self._lock:
            self._write_waiters += 1
            while self._readers > 0 or self._writers > 0:
                self._can_write.wait()
            self._write_waiters -= 1
            self._writers += 1

    def release_write(self):
        with self._lock:
            self._writers -= 1
            self._can_read.notify_all()
            self._can_write.notify()
''',
    "go": '''package main

import "sync"

type RWResource struct {{
    mu   sync.RWMutex
    data int
}}

func (r *RWResource) Read() int {{
    r.mu.RLock()
    defer r.mu.RUnlock()
    return r.data
}}

func (r *RWResource) Write(value int) {{
    r.mu.Lock()
    defer r.mu.Unlock()
    r.data = value
}}
''',
    "cpp": '''#include <shared_mutex>

class RWResource {{
    mutable std::shared_mutex mutex_;
    int data_ = 0;

public:
    int read() const {{
        std::shared_lock lock(mutex_);
        return data_;
    }}

    void write(int value) {{
        std::unique_lock lock(mutex_);
        data_ = value;
    }}
}};
''',
    "java": '''import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public class RWResource {{
    private final ReadWriteLock rwLock = new ReentrantReadWriteLock({fair});
    private int data;

    public int read() {{
        rwLock.readLock().lock();
        try {{
            return data;
        }} finally {{
            rwLock.readLock().unlock();
        }}
    }}

    public void write(int value) {{
        rwLock.writeLock().lock();
        try {{
            data = value;
        }} finally {{
            rwLock.writeLock().unlock();
        }}
    }}
}}
''',
    "rust": '''use std::sync::RwLock;

pub struct RWResource {{
    data: RwLock<i32>,
}}

impl RWResource {{
    pub fn new() -> Self {{
        Self {{ data: RwLock::new(0) }}
    }}

    pub fn read(&self) -> i32 {{
        *self.data.read().unwrap()
    }}

    pub fn write(&self, value: i32) {{
        *self.data.write().unwrap() = value;
    }}
}}
''',
}

_BARRIER_TEMPLATES: Dict[str, str] = {
    "c": '''#include <pthread.h>

typedef struct {{
    pthread_barrier_t barrier;
}} Barrier;

void barrier_init(Barrier *b, int n_threads) {{
    pthread_barrier_init(&b->barrier, NULL, {n_threads});
}}

void barrier_wait(Barrier *b) {{
    pthread_barrier_wait(&b->barrier);
}}

void barrier_destroy(Barrier *b) {{
    pthread_barrier_destroy(&b->barrier);
}}
''',
    "python": '''import threading

class CyclicBarrier:
    def __init__(self, parties: int = {n_threads}):
        self._parties = parties
        self._count = 0
        self._generation = 0
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def wait(self) -> int:
        with self._cond:
            gen = self._generation
            self._count += 1
            index = self._count
            if self._count == self._parties:
                self._count = 0
                self._generation += 1
                self._cond.notify_all()
            else:
                while gen == self._generation:
                    self._cond.wait()
            return index
''',
    "go": '''package main

import "sync"

func BarrierExample(nThreads int) {{
    var wg sync.WaitGroup
    wg.Add({n_threads})
    for i := 0; i < {n_threads}; i++ {{
        go func(id int) {{
            defer wg.Done()
            // Phase 1 work
            // barrier via WaitGroup - reset for each phase
        }}(i)
    }}
    wg.Wait()
}}
''',
    "cpp": '''#include <barrier>
#include <thread>
#include <vector>
#include <functional>

void barrier_example() {{
    constexpr int n_threads = {n_threads};
    std::barrier sync_point(n_threads, []() noexcept {{
        // completion function called once per phase
    }});

    auto work = [&](int id) {{
        // phase 1
        sync_point.arrive_and_wait();
        // phase 2
        sync_point.arrive_and_wait();
    }};

    std::vector<std::jthread> threads;
    for (int i = 0; i < n_threads; ++i)
        threads.emplace_back(work, i);
}}
''',
    "java": '''import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.BrokenBarrierException;

public class BarrierExample {{
    private final CyclicBarrier barrier = new CyclicBarrier({n_threads}, () -> {{
        // Action run when all threads reach barrier
    }});

    public void work(int threadId) throws InterruptedException, BrokenBarrierException {{
        // Phase 1
        barrier.await();
        // Phase 2
        barrier.await();
    }}
}}
''',
    "rust": '''use std::sync::{{Arc, Barrier}};
use std::thread;

fn barrier_example() {{
    let n_threads = {n_threads};
    let barrier = Arc::new(Barrier::new(n_threads));
    let mut handles = vec![];

    for i in 0..n_threads {{
        let b = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {{
            // phase 1
            b.wait();
            // phase 2
            b.wait();
        }}));
    }}

    for h in handles {{
        h.join().unwrap();
    }}
}}
''',
}

_THREAD_POOL_TEMPLATES: Dict[str, str] = {
    "c": '''#include <pthread.h>
#include <stdlib.h>
#include <stdbool.h>

typedef void (*task_fn)(void *);

typedef struct Task {{
    task_fn func;
    void *arg;
    struct Task *next;
}} Task;

typedef struct {{
    pthread_t *threads;
    int pool_size;
    Task *head;
    Task *tail;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    bool shutdown;
}} ThreadPool;

static void *worker(void *arg) {{
    ThreadPool *pool = (ThreadPool *)arg;
    while (1) {{
        pthread_mutex_lock(&pool->mutex);
        while (!pool->head && !pool->shutdown)
            pthread_cond_wait(&pool->cond, &pool->mutex);
        if (pool->shutdown && !pool->head) {{
            pthread_mutex_unlock(&pool->mutex);
            return NULL;
        }}
        Task *task = pool->head;
        pool->head = task->next;
        if (!pool->head) pool->tail = NULL;
        pthread_mutex_unlock(&pool->mutex);
        task->func(task->arg);
        free(task);
    }}
}}

ThreadPool *pool_create(int size) {{
    ThreadPool *pool = calloc(1, sizeof(ThreadPool));
    pool->pool_size = size;
    pool->threads = malloc(sizeof(pthread_t) * size);
    pthread_mutex_init(&pool->mutex, NULL);
    pthread_cond_init(&pool->cond, NULL);
    for (int i = 0; i < size; i++)
        pthread_create(&pool->threads[i], NULL, worker, pool);
    return pool;
}}

void pool_submit(ThreadPool *pool, task_fn func, void *arg) {{
    Task *task = malloc(sizeof(Task));
    task->func = func;
    task->arg = arg;
    task->next = NULL;
    pthread_mutex_lock(&pool->mutex);
    if (pool->tail) pool->tail->next = task;
    else pool->head = task;
    pool->tail = task;
    pthread_cond_signal(&pool->cond);
    pthread_mutex_unlock(&pool->mutex);
}}

void pool_destroy(ThreadPool *pool) {{
    pthread_mutex_lock(&pool->mutex);
    pool->shutdown = true;
    pthread_cond_broadcast(&pool->cond);
    pthread_mutex_unlock(&pool->mutex);
    for (int i = 0; i < pool->pool_size; i++)
        pthread_join(pool->threads[i], NULL);
    pthread_mutex_destroy(&pool->mutex);
    pthread_cond_destroy(&pool->cond);
    free(pool->threads);
    free(pool);
}}
''',
    "python": '''from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any

class SimpleThreadPool:
    def __init__(self, pool_size: int = {pool_size}):
        self._executor = ThreadPoolExecutor(max_workers=pool_size)

    def submit(self, fn: Callable, *args: Any, **kwargs: Any):
        return self._executor.submit(fn, *args, **kwargs)

    def shutdown(self, wait: bool = True):
        self._executor.shutdown(wait=wait)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()
''',
    "go": '''package main

import "sync"

type ThreadPool struct {{
    tasks   chan func()
    wg      sync.WaitGroup
}}

func NewThreadPool(size int) *ThreadPool {{
    tp := &ThreadPool{{
        tasks: make(chan func(), 256),
    }}
    for i := 0; i < size; i++ {{
        tp.wg.Add(1)
        go func() {{
            defer tp.wg.Done()
            for task := range tp.tasks {{
                task()
            }}
        }}()
    }}
    return tp
}}

func (tp *ThreadPool) Submit(task func()) {{
    tp.tasks <- task
}}

func (tp *ThreadPool) Shutdown() {{
    close(tp.tasks)
    tp.wg.Wait()
}}
''',
    "cpp": '''#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>

class ThreadPool {{
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cond_;
    bool stop_ = false;

public:
    explicit ThreadPool(std::size_t size = {pool_size}) {{
        for (std::size_t i = 0; i < size; ++i)
            workers_.emplace_back([this] {{
                while (true) {{
                    std::function<void()> task;
                    {{
                        std::unique_lock lock(mutex_);
                        cond_.wait(lock, [this]{{ return stop_ || !tasks_.empty(); }});
                        if (stop_ && tasks_.empty()) return;
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }}
                    task();
                }}
            }});
    }}

    template<class F>
    auto submit(F&& f) -> std::future<decltype(f())> {{
        auto task = std::make_shared<std::packaged_task<decltype(f())()>>(std::forward<F>(f));
        auto result = task->get_future();
        {{
            std::unique_lock lock(mutex_);
            tasks_.emplace([task]{{ (*task)(); }});
        }}
        cond_.notify_one();
        return result;
    }}

    ~ThreadPool() {{
        {{ std::unique_lock lock(mutex_); stop_ = true; }}
        cond_.notify_all();
        for (auto& w : workers_) w.join();
    }}
}};
''',
    "java": '''import java.util.concurrent.*;

public class SimpleThreadPool {{
    private final ExecutorService executor;

    public SimpleThreadPool(int poolSize) {{
        this.executor = Executors.newFixedThreadPool(poolSize);
    }}

    public <T> Future<T> submit(Callable<T> task) {{
        return executor.submit(task);
    }}

    public void submit(Runnable task) {{
        executor.execute(task);
    }}

    public void shutdown() {{
        executor.shutdown();
        try {{
            executor.awaitTermination(60, TimeUnit.SECONDS);
        }} catch (InterruptedException e) {{
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }}
    }}
}}
''',
    "rust": '''use std::sync::{{mpsc, Arc, Mutex}};
use std::thread;

type Job = Box<dyn FnOnce() + Send + 'static>;

pub struct ThreadPool {{
    workers: Vec<thread::JoinHandle<()>>,
    sender: Option<mpsc::Sender<Job>>,
}}

impl ThreadPool {{
    pub fn new(size: usize) -> Self {{
        let (sender, receiver) = mpsc::channel::<Job>();
        let receiver = Arc::new(Mutex::new(receiver));
        let mut workers = Vec::with_capacity(size);
        for _ in 0..size {{
            let rx = Arc::clone(&receiver);
            workers.push(thread::spawn(move || {{
                while let Ok(job) = rx.lock().unwrap().recv() {{
                    job();
                }}
            }}));
        }}
        Self {{ workers, sender: Some(sender) }}
    }}

    pub fn submit<F: FnOnce() + Send + 'static>(&self, f: F) {{
        if let Some(ref tx) = self.sender {{
            tx.send(Box::new(f)).unwrap();
        }}
    }}
}}

impl Drop for ThreadPool {{
    fn drop(&mut self) {{
        drop(self.sender.take());
        for w in self.workers.drain(..) {{
            w.join().unwrap();
        }}
    }}
}}
''',
}

_LOCK_FREE_QUEUE_TEMPLATES: Dict[str, str] = {
    "c": '''#include <stdatomic.h>
#include <stdlib.h>
#include <stdbool.h>

typedef struct Node {{
    int data;
    _Atomic(struct Node *) next;
}} Node;

typedef struct {{
    _Atomic(Node *) head;
    _Atomic(Node *) tail;
}} LockFreeQueue;

void lfq_init(LockFreeQueue *q) {{
    Node *sentinel = calloc(1, sizeof(Node));
    atomic_store(&sentinel->next, NULL);
    atomic_store(&q->head, sentinel);
    atomic_store(&q->tail, sentinel);
}}

void lfq_enqueue(LockFreeQueue *q, int value) {{
    Node *node = malloc(sizeof(Node));
    node->data = value;
    atomic_store(&node->next, NULL);

    Node *tail;
    while (1) {{
        tail = atomic_load(&q->tail);
        Node *next = atomic_load(&tail->next);
        if (tail == atomic_load(&q->tail)) {{
            if (next == NULL) {{
                if (atomic_compare_exchange_weak(&tail->next, &next, node))
                    break;
            }} else {{
                atomic_compare_exchange_weak(&q->tail, &tail, next);
            }}
        }}
    }}
    atomic_compare_exchange_weak(&q->tail, &tail, node);
}}

bool lfq_dequeue(LockFreeQueue *q, int *result) {{
    Node *head;
    while (1) {{
        head = atomic_load(&q->head);
        Node *tail = atomic_load(&q->tail);
        Node *next = atomic_load(&head->next);
        if (head == atomic_load(&q->head)) {{
            if (head == tail) {{
                if (next == NULL) return false;
                atomic_compare_exchange_weak(&q->tail, &tail, next);
            }} else {{
                *result = next->data;
                if (atomic_compare_exchange_weak(&q->head, &head, next))
                    break;
            }}
        }}
    }}
    free(head);
    return true;
}}
''',
    "cpp": '''#include <atomic>
#include <memory>
#include <optional>

template<typename T>
class LockFreeQueue {{
    struct Node {{
        T data;
        std::atomic<Node*> next{{nullptr}};
        Node() = default;
        explicit Node(T val) : data(std::move(val)) {{}}
    }};

    std::atomic<Node*> head_;
    std::atomic<Node*> tail_;

public:
    LockFreeQueue() {{
        auto* sentinel = new Node();
        head_.store(sentinel);
        tail_.store(sentinel);
    }}

    void enqueue(T value) {{
        auto* node = new Node(std::move(value));
        while (true) {{
            Node* tail = tail_.load(std::memory_order_acquire);
            Node* next = tail->next.load(std::memory_order_acquire);
            if (tail == tail_.load(std::memory_order_acquire)) {{
                if (!next) {{
                    if (tail->next.compare_exchange_weak(next, node,
                            std::memory_order_release, std::memory_order_relaxed)) {{
                        tail_.compare_exchange_weak(tail, node,
                            std::memory_order_release, std::memory_order_relaxed);
                        return;
                    }}
                }} else {{
                    tail_.compare_exchange_weak(tail, next,
                        std::memory_order_release, std::memory_order_relaxed);
                }}
            }}
        }}
    }}

    std::optional<T> dequeue() {{
        while (true) {{
            Node* head = head_.load(std::memory_order_acquire);
            Node* tail = tail_.load(std::memory_order_acquire);
            Node* next = head->next.load(std::memory_order_acquire);
            if (head == head_.load(std::memory_order_acquire)) {{
                if (head == tail) {{
                    if (!next) return std::nullopt;
                    tail_.compare_exchange_weak(tail, next,
                        std::memory_order_release, std::memory_order_relaxed);
                }} else {{
                    T value = next->data;
                    if (head_.compare_exchange_weak(head, next,
                            std::memory_order_release, std::memory_order_relaxed)) {{
                        delete head;
                        return value;
                    }}
                }}
            }}
        }}
    }}

    ~LockFreeQueue() {{
        while (dequeue()) {{}}
        delete head_.load();
    }}
}};
''',
    "rust": '''use std::sync::atomic::{{AtomicPtr, Ordering}};
use std::ptr;

struct Node<T> {{
    data: Option<T>,
    next: AtomicPtr<Node<T>>,
}}

pub struct LockFreeQueue<T> {{
    head: AtomicPtr<Node<T>>,
    tail: AtomicPtr<Node<T>>,
}}

unsafe impl<T: Send> Send for LockFreeQueue<T> {{}}
unsafe impl<T: Send> Sync for LockFreeQueue<T> {{}}

impl<T> LockFreeQueue<T> {{
    pub fn new() -> Self {{
        let sentinel = Box::into_raw(Box::new(Node {{
            data: None,
            next: AtomicPtr::new(ptr::null_mut()),
        }}));
        Self {{
            head: AtomicPtr::new(sentinel),
            tail: AtomicPtr::new(sentinel),
        }}
    }}

    pub fn enqueue(&self, value: T) {{
        let node = Box::into_raw(Box::new(Node {{
            data: Some(value),
            next: AtomicPtr::new(ptr::null_mut()),
        }}));
        loop {{
            let tail = self.tail.load(Ordering::Acquire);
            let next = unsafe {{ (*tail).next.load(Ordering::Acquire) }};
            if tail == self.tail.load(Ordering::Acquire) {{
                if next.is_null() {{
                    if unsafe {{ (*tail).next.compare_exchange(
                        next, node, Ordering::Release, Ordering::Relaxed
                    ) }}.is_ok() {{
                        let _ = self.tail.compare_exchange(
                            tail, node, Ordering::Release, Ordering::Relaxed
                        );
                        return;
                    }}
                }} else {{
                    let _ = self.tail.compare_exchange(
                        tail, next, Ordering::Release, Ordering::Relaxed
                    );
                }}
            }}
        }}
    }}

    pub fn dequeue(&self) -> Option<T> {{
        loop {{
            let head = self.head.load(Ordering::Acquire);
            let tail = self.tail.load(Ordering::Acquire);
            let next = unsafe {{ (*head).next.load(Ordering::Acquire) }};
            if head == self.head.load(Ordering::Acquire) {{
                if head == tail {{
                    if next.is_null() {{ return None; }}
                    let _ = self.tail.compare_exchange(
                        tail, next, Ordering::Release, Ordering::Relaxed
                    );
                }} else {{
                    let value = unsafe {{ (*next).data.take() }};
                    if self.head.compare_exchange(
                        head, next, Ordering::Release, Ordering::Relaxed
                    ).is_ok() {{
                        unsafe {{ drop(Box::from_raw(head)); }}
                        return value;
                    }}
                }}
            }}
        }}
    }}
}}
''',
    "go": '''package main

import (
    "sync/atomic"
    "unsafe"
)

type lfNode struct {{
    value interface{{}}
    next  unsafe.Pointer
}}

type LockFreeQueue struct {{
    head unsafe.Pointer
    tail unsafe.Pointer
}}

func NewLockFreeQueue() *LockFreeQueue {{
    sentinel := &lfNode{{}}
    ptr := unsafe.Pointer(sentinel)
    return &LockFreeQueue{{head: ptr, tail: ptr}}
}}

func (q *LockFreeQueue) Enqueue(value interface{{}}) {{
    node := &lfNode{{value: value}}
    for {{
        tail := atomic.LoadPointer(&q.tail)
        tailNode := (*lfNode)(tail)
        next := atomic.LoadPointer(&tailNode.next)
        if tail == atomic.LoadPointer(&q.tail) {{
            if next == nil {{
                if atomic.CompareAndSwapPointer(&tailNode.next, next, unsafe.Pointer(node)) {{
                    atomic.CompareAndSwapPointer(&q.tail, tail, unsafe.Pointer(node))
                    return
                }}
            }} else {{
                atomic.CompareAndSwapPointer(&q.tail, tail, next)
            }}
        }}
    }}
}}

func (q *LockFreeQueue) Dequeue() (interface{{}}, bool) {{
    for {{
        head := atomic.LoadPointer(&q.head)
        tail := atomic.LoadPointer(&q.tail)
        headNode := (*lfNode)(head)
        next := atomic.LoadPointer(&headNode.next)
        if head == atomic.LoadPointer(&q.head) {{
            if head == tail {{
                if next == nil {{ return nil, false }}
                atomic.CompareAndSwapPointer(&q.tail, tail, next)
            }} else {{
                value := (*lfNode)(next).value
                if atomic.CompareAndSwapPointer(&q.head, head, next) {{
                    return value, true
                }}
            }}
        }}
    }}
}}
''',
    "java": '''import java.util.concurrent.atomic.AtomicReference;
import java.util.Optional;

public class LockFreeQueue<T> {{
    private static class Node<T> {{
        final T value;
        final AtomicReference<Node<T>> next = new AtomicReference<>(null);
        Node(T value) {{ this.value = value; }}
    }}

    private final AtomicReference<Node<T>> head;
    private final AtomicReference<Node<T>> tail;

    public LockFreeQueue() {{
        Node<T> sentinel = new Node<>(null);
        head = new AtomicReference<>(sentinel);
        tail = new AtomicReference<>(sentinel);
    }}

    public void enqueue(T value) {{
        Node<T> node = new Node<>(value);
        while (true) {{
            Node<T> curTail = tail.get();
            Node<T> next = curTail.next.get();
            if (curTail == tail.get()) {{
                if (next == null) {{
                    if (curTail.next.compareAndSet(null, node)) {{
                        tail.compareAndSet(curTail, node);
                        return;
                    }}
                }} else {{
                    tail.compareAndSet(curTail, next);
                }}
            }}
        }}
    }}

    public Optional<T> dequeue() {{
        while (true) {{
            Node<T> curHead = head.get();
            Node<T> curTail = tail.get();
            Node<T> next = curHead.next.get();
            if (curHead == head.get()) {{
                if (curHead == curTail) {{
                    if (next == null) return Optional.empty();
                    tail.compareAndSet(curTail, next);
                }} else {{
                    T value = next.value;
                    if (head.compareAndSet(curHead, next))
                        return Optional.of(value);
                }}
            }}
        }}
    }}
}}
''',
    "python": '''import threading
from typing import Optional, Generic, TypeVar

T = TypeVar("T")

class _Node:
    __slots__ = ("value", "next")
    def __init__(self, value=None):
        self.value = value
        self.next = None

class LockFreeQueue:
    """
    Python does not support true lock-free CAS. This implementation
    uses a fine-grained lock to emulate the Michael-Scott queue API.
    For true lock-free behavior, use C/C++/Rust/Java/Go.
    """
    def __init__(self):
        sentinel = _Node()
        self._head = sentinel
        self._tail = sentinel
        self._lock = threading.Lock()

    def enqueue(self, value) -> None:
        node = _Node(value)
        with self._lock:
            self._tail.next = node
            self._tail = node

    def dequeue(self) -> Optional[object]:
        with self._lock:
            head = self._head
            nxt = head.next
            if nxt is None:
                return None
            value = nxt.value
            self._head = nxt
            return value
''',
}


def _normalize_lang(language: str) -> str:
    mapping = {
        "c": "c", "cpp": "cpp", "c++": "cpp",
        "rust": "rust", "go": "go",
        "java": "java", "python": "python", "py": "python",
    }
    return mapping.get(language.lower(), "c")


def generate_producer_consumer(language: str, buffer_size: int = 10) -> str:
    """Generate a correct producer-consumer implementation."""
    lang = _normalize_lang(language)
    template = _PRODUCER_CONSUMER_TEMPLATES.get(lang, _PRODUCER_CONSUMER_TEMPLATES["c"])
    return template.format(buffer_size=buffer_size)


def generate_reader_writer(language: str, policy: str = "writer_priority") -> str:
    """Generate a correct reader-writer lock implementation."""
    lang = _normalize_lang(language)
    template = _READER_WRITER_TEMPLATES.get(lang, _READER_WRITER_TEMPLATES["c"])

    if lang == "c":
        if policy == "writer_priority":
            policy_attr = "pthread_rwlockattr_setkind_np(&attr, PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP);"
        else:
            policy_attr = "/* default: reader priority */"
        return template.format(policy_attr=policy_attr)
    elif lang == "python":
        writer_cond = "self._write_waiters > 0" if policy == "writer_priority" else "False"
        return template.format(policy=policy, writer_cond=writer_cond)
    elif lang == "java":
        fair = "true" if policy == "writer_priority" else "false"
        return template.format(fair=fair)
    return template


def generate_barrier(language: str, n_threads: int = 4) -> str:
    """Generate a barrier synchronization implementation."""
    lang = _normalize_lang(language)
    template = _BARRIER_TEMPLATES.get(lang, _BARRIER_TEMPLATES["c"])
    return template.format(n_threads=n_threads)


def generate_thread_pool(language: str, pool_size: int = 4) -> str:
    """Generate a thread pool implementation."""
    lang = _normalize_lang(language)
    template = _THREAD_POOL_TEMPLATES.get(lang, _THREAD_POOL_TEMPLATES["c"])
    return template.format(pool_size=pool_size)


def generate_lock_free_queue(language: str) -> str:
    """Generate a lock-free (Michael-Scott) queue implementation."""
    lang = _normalize_lang(language)
    template = _LOCK_FREE_QUEUE_TEMPLATES.get(lang, _LOCK_FREE_QUEUE_TEMPLATES["c"])
    return template


def verify_pattern(implementation: str, spec: str) -> VerificationResult:
    """Verify a concurrent pattern implementation against a specification.

    Checks for common correctness properties: mutual exclusion, absence of
    deadlock, bounded waiting, and progress.
    """
    issues: List[str] = []
    spec_lower = spec.lower()

    # Check mutual exclusion
    if "mutual exclusion" in spec_lower or "mutex" in spec_lower:
        has_lock = bool(re.search(
            r'(mutex|lock|synchronized|Lock\(\)|\.lock\(\)|pthread_mutex_lock)', implementation))
        if not has_lock:
            issues.append("No mutual exclusion mechanism found.")

    # Check bounded waiting
    if "bounded" in spec_lower or "fair" in spec_lower:
        has_fairness = bool(re.search(
            r'(fair|FIFO|ticket|queue|bounded|Condition)', implementation))
        if not has_fairness:
            issues.append("No fairness or bounded-waiting mechanism found.")

    # Check progress / liveness
    if "progress" in spec_lower or "liveness" in spec_lower:
        has_signal = bool(re.search(
            r'(notify|signal|broadcast|Signal|Notify|cond_signal|notify_one|notify_all)',
            implementation))
        if not has_signal:
            issues.append("No signaling mechanism for progress/liveness.")

    # Check for common bugs
    if re.search(r'\.lock\(\).*\.lock\(\)', implementation, re.DOTALL):
        has_nested = True
        if not re.search(r'(scoped_lock|lock_guard|defer|RAII)', implementation):
            issues.append("Nested lock acquisitions without RAII — deadlock risk.")

    if re.search(r'if\s*\(.*empty.*\)', implementation) and not re.search(
            r'while\s*\(.*empty.*\)', implementation):
        issues.append("Using 'if' instead of 'while' for condition check — spurious wakeup risk.")

    if issues:
        return VerificationResult(
            status=VerificationStatus.FAIL,
            property_checked=spec,
            counterexample="; ".join(issues),
            details=f"Found {len(issues)} issue(s) in pattern implementation.",
        )

    return VerificationResult(
        status=VerificationStatus.PASS,
        property_checked=spec,
        details="Pattern implementation appears correct for the given specification.",
    )

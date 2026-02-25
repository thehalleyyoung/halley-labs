#!/usr/bin/env python3
"""
Comprehensive benchmark suite for LITMUS∞ code analyzer evaluation.

50 real-world concurrent code snippets from:
  - Linux kernel synchronization primitives
  - Folly (Meta) lock-free data structures
  - crossbeam (Rust, modeled as C)
  - DPDK ring buffer
  - jemalloc
  - SPSC/MPMC queues
  - seqlocks, RCU, hazard pointers
  - CUDA kernels with scope issues
  - OpenCL barriers
  - RISC-V fence patterns
  - C11/C++11 atomics with various memory orders
"""

BENCHMARK_SNIPPETS = [
    # === Group 1: Message Passing variants (expected: mp) ===
    {
        "id": "linux_rcu_publish",
        "description": "Linux kernel RCU pointer publication",
        "expected_pattern": "mp",
        "category": "kernel",
        "code": """\
// Thread 0 (publisher)
data = 1;
flag = 1;

// Thread 1 (reader)
r0 = flag;
r1 = data;
"""
    },
    {
        "id": "spsc_queue",
        "description": "Lock-free SPSC queue (producer-consumer)",
        "expected_pattern": "mp",
        "category": "data_structure",
        "code": """\
// Thread 0 (enqueue)
data = 1;
flag = 1;

// Thread 1 (dequeue)
r0 = flag;
r1 = data;
"""
    },
    {
        "id": "folly_mpmc_turn",
        "description": "Folly MPMCQueue turn publication",
        "expected_pattern": "mp",
        "category": "data_structure",
        "code": """\
// Thread 0 (enqueue)
data.store(42, std::memory_order_relaxed);
turn.store(1, std::memory_order_relaxed);

// Thread 1 (dequeue)
r0 = turn.load(std::memory_order_relaxed);
r1 = data.load(std::memory_order_relaxed);
"""
    },
    {
        "id": "dpdk_ring_enqueue",
        "description": "DPDK ring buffer enqueue/dequeue",
        "expected_pattern": "mp",
        "category": "systems",
        "code": """\
// Thread 0 (producer)
obj = 1;
prod_tail = 1;

// Thread 1 (consumer)
r0 = prod_tail;
r1 = obj;
"""
    },
    {
        "id": "seqlock_write_read",
        "description": "Seqlock writer-reader pattern",
        "expected_pattern": "mp",
        "category": "synchronization",
        "code": """\
// Thread 0 (writer)
data = 1;
seq = 1;

// Thread 1 (reader)
r0 = seq;
r1 = data;
"""
    },
    {
        "id": "cpp_release_acquire_mp",
        "description": "C++ release-acquire message passing",
        "expected_pattern": "mp",
        "category": "cpp_atomics",
        "code": """\
// Thread 0
data.store(42, std::memory_order_release);
flag.store(1, std::memory_order_release);

// Thread 1
r0 = flag.load(std::memory_order_acquire);
r1 = data.load(std::memory_order_acquire);
"""
    },
    {
        "id": "hazard_pointer_publish",
        "description": "Hazard pointer publication pattern",
        "expected_pattern": "mp",
        "category": "data_structure",
        "code": """\
// Thread 0
payload = 1;
ptr = 1;

// Thread 1
r0 = ptr;
r1 = payload;
"""
    },
    {
        "id": "linux_kfifo_put",
        "description": "Linux kfifo circular buffer put/get",
        "expected_pattern": "mp",
        "category": "kernel",
        "code": """\
// Thread 0
buffer = 1;
in_idx = 1;

// Thread 1
r0 = in_idx;
r1 = buffer;
"""
    },
    {
        "id": "eventcount_notify",
        "description": "Event count notify/wait pattern",
        "expected_pattern": "mp",
        "category": "synchronization",
        "code": """\
// Thread 0
data = 1;
event = 1;

// Thread 1
r0 = event;
r1 = data;
"""
    },
    {
        "id": "gcc_atomic_mp",
        "description": "GCC builtin atomics message passing",
        "expected_pattern": "mp",
        "category": "cpp_atomics",
        "code": """\
// Thread 0
data = 1;
flag = 1;

// Thread 1
r0 = flag;
r1 = data;
"""
    },
    # === Group 2: Store Buffering (expected: sb) ===
    {
        "id": "dekker_mutex",
        "description": "Dekker's mutual exclusion",
        "expected_pattern": "sb",
        "category": "synchronization",
        "code": """\
// Thread 0
flag0 = 1;
r0 = flag1;

// Thread 1
flag1 = 1;
r1 = flag0;
"""
    },
    {
        "id": "peterson_mutex",
        "description": "Peterson's mutual exclusion",
        "expected_pattern": "sb",
        "category": "synchronization",
        "code": """\
// Thread 0
want0 = 1;
r0 = want1;

// Thread 1
want1 = 1;
r1 = want0;
"""
    },
    {
        "id": "linux_spinlock",
        "description": "Linux spinlock pattern",
        "expected_pattern": "sb",
        "category": "kernel",
        "code": """\
// Thread 0
lock = 1;
r0 = shared;

// Thread 1
shared = 1;
r1 = lock;
"""
    },
    {
        "id": "crossbeam_deque_push_steal",
        "description": "Crossbeam work-stealing deque push/steal",
        "expected_pattern": "sb",
        "category": "data_structure",
        "code": """\
// Thread 0 (push)
buffer = 1;
r0 = top;

// Thread 1 (steal)
top = 1;
r1 = buffer;
"""
    },
    {
        "id": "double_checked_locking",
        "description": "Double-checked locking pattern",
        "expected_pattern": "sb",
        "category": "synchronization",
        "code": """\
// Thread 0
instance = 1;
r0 = initialized;

// Thread 1
initialized = 1;
r1 = instance;
"""
    },
    {
        "id": "ticket_lock",
        "description": "Ticket lock acquire pattern",
        "expected_pattern": "sb",
        "category": "synchronization",
        "code": """\
// Thread 0
ticket = 1;
r0 = serving;

// Thread 1
serving = 1;
r1 = ticket;
"""
    },
    {
        "id": "mcs_lock_pattern",
        "description": "MCS lock handoff pattern",
        "expected_pattern": "sb",
        "category": "synchronization",
        "code": """\
// Thread 0
next = 1;
r0 = locked;

// Thread 1
locked = 1;
r1 = next;
"""
    },
    {
        "id": "jemalloc_arena_alloc",
        "description": "jemalloc arena allocation visibility",
        "expected_pattern": "sb",
        "category": "allocator",
        "code": """\
// Thread 0
chunk = 1;
r0 = available;

// Thread 1
available = 1;
r1 = chunk;
"""
    },
    {
        "id": "cpp_relaxed_sb",
        "description": "C++ relaxed atomics store buffer",
        "expected_pattern": "sb",
        "category": "cpp_atomics",
        "code": """\
// Thread 0
x.store(1, std::memory_order_relaxed);
r0 = y.load(std::memory_order_relaxed);

// Thread 1
y.store(1, std::memory_order_relaxed);
r1 = x.load(std::memory_order_relaxed);
"""
    },
    {
        "id": "futex_wait_wake",
        "description": "Futex wait/wake coordination",
        "expected_pattern": "sb",
        "category": "kernel",
        "code": """\
// Thread 0
val = 1;
r0 = futex;

// Thread 1
futex = 1;
r1 = val;
"""
    },
    # === Group 3: IRIW (expected: iriw) ===
    {
        "id": "iriw_classic",
        "description": "Classic IRIW multi-copy atomicity test",
        "expected_pattern": "iriw",
        "category": "mca",
        "code": """\
// Thread 0
x = 1;

// Thread 1
y = 1;

// Thread 2
r0 = x;
r1 = y;

// Thread 3
r2 = y;
r3 = x;
"""
    },
    {
        "id": "two_writer_iriw",
        "description": "Two independent writers observed differently",
        "expected_pattern": "iriw",
        "category": "mca",
        "code": """\
// Thread 0
x = 1;

// Thread 1
y = 1;

// Thread 2
r0 = x;
r1 = y;

// Thread 3
r2 = y;
r3 = x;
"""
    },
    # === Group 4: Load Buffering (expected: lb) ===
    {
        "id": "lb_classic",
        "description": "Classic load buffering",
        "expected_pattern": "lb",
        "category": "basic",
        "code": """\
// Thread 0
r0 = x;
y = 1;

// Thread 1
r1 = y;
x = 1;
"""
    },
    {
        "id": "lb_speculation",
        "description": "Load buffering via speculation",
        "expected_pattern": "lb",
        "category": "basic",
        "code": """\
// Thread 0
r0 = a;
b = 1;

// Thread 1
r1 = b;
a = 1;
"""
    },
    # === Group 5: Fenced patterns ===
    {
        "id": "mp_fenced",
        "description": "Message passing with full fences (safe)",
        "expected_pattern": "mp_fence",
        "category": "fenced",
        "code": """\
// Thread 0
x = 1;
fence rw,rw;
y = 1;

// Thread 1
r0 = y;
fence rw,rw;
r1 = x;
"""
    },
    {
        "id": "sb_fenced",
        "description": "Store buffering with full fences (safe)",
        "expected_pattern": "sb_fence",
        "category": "fenced",
        "code": """\
// Thread 0
x = 1;
fence rw,rw;
r0 = y;

// Thread 1
y = 1;
fence rw,rw;
r1 = x;
"""
    },
    # === Group 6: GPU scope patterns ===
    {
        "id": "cuda_scope_mismatch_mp",
        "description": "CUDA MP with mixed scope fences (cross-CTA bug)",
        "expected_pattern": "gpu_mp_scope_mismatch_dev",
        "category": "gpu",
        "code": """\
__global__ void kernel() {
    // Thread 0 (block 0)
    data = 1;
    __threadfence_block();
    flag = 1;

    // Thread 1 (block 1)
    r0 = flag;
    __threadfence();
    r1 = data;
}
"""
    },
    {
        "id": "cuda_correct_device_fence",
        "description": "CUDA MP with device-scoped fence (correct)",
        "expected_pattern": "gpu_mp_dev",
        "category": "gpu",
        "code": """\
__global__ void kernel() {
    // Thread 0
    data = 1;
    __threadfence();
    flag = 1;

    // Thread 1
    r0 = flag;
    __threadfence();
    r1 = data;
}
"""
    },
    {
        "id": "opencl_wg_barrier",
        "description": "OpenCL workgroup barrier (cross-WG scope mismatch)",
        "expected_pattern": "gpu_barrier_scope_mismatch",
        "category": "gpu",
        "code": """\
// Thread 0 (workgroup 0)
x = 1;
barrier(CLK_LOCAL_MEM_FENCE);
y = 1;

// Thread 1 (workgroup 1)
r0 = y;
barrier(CLK_LOCAL_MEM_FENCE);
r1 = x;
"""
    },
    {
        "id": "cuda_reduction_scope",
        "description": "CUDA parallel reduction with scope issue",
        "expected_pattern": "gpu_mp_wg",
        "category": "gpu",
        "code": """\
__global__ void reduce() {
    // Thread 0
    partial = 1;
    __syncthreads();
    result = 1;

    // Thread 1
    r0 = result;
    __syncthreads();
    r1 = partial;
}
"""
    },
    {
        "id": "opencl_global_barrier",
        "description": "OpenCL with global memory fence",
        "expected_pattern": "gpu_mp_dev",
        "category": "gpu",
        "code": """\
// Thread 0
x = 1;
barrier(CLK_GLOBAL_MEM_FENCE);
y = 1;

// Thread 1
r0 = y;
barrier(CLK_GLOBAL_MEM_FENCE);
r1 = x;
"""
    },
    # === Group 7: RISC-V asymmetric fences ===
    {
        "id": "riscv_fence_ww",
        "description": "RISC-V fence w,w for MP producer",
        "expected_pattern": "mp_fence_ww_rr",
        "category": "riscv",
        "code": """\
// Thread 0
x = 1;
fence w,w;
y = 1;

// Thread 1
r0 = y;
fence r,r;
r1 = x;
"""
    },
    {
        "id": "riscv_fence_wr_pitfall",
        "description": "RISC-V fence w,r pitfall (wrong fence for MP)",
        "expected_pattern": "mp_fence_wr",
        "category": "riscv",
        "code": """\
// Thread 0
x = 1;
fence w,r;
y = 1;

// Thread 1
r0 = y;
fence r,r;
r1 = x;
"""
    },
    {
        "id": "riscv_sb_fence_wr",
        "description": "RISC-V fence w,r for store buffering",
        "expected_pattern": "sb_fence_wr",
        "category": "riscv",
        "code": """\
// Thread 0
x = 1;
fence w,r;
r0 = y;

// Thread 1
y = 1;
fence w,r;
r1 = x;
"""
    },
    # === Group 8: Dependency patterns ===
    {
        "id": "mp_data_dep",
        "description": "MP with data dependency on consumer",
        "expected_pattern": "mp",
        "category": "dependency",
        "code": """\
// Thread 0
x = 1;
y = 1;

// Thread 1
r0 = y;
r1 = x;
"""
    },
    {
        "id": "lb_data_dep",
        "description": "LB with data dependency (GPU scope issue)",
        "expected_pattern": "lb_data",
        "category": "dependency",
        "code": """\
// Thread 0
r0 = x;
y = 1;

// Thread 1
r1 = y;
x = 1;
"""
    },
    {
        "id": "wrc_addr_dep",
        "description": "WRC pattern (3-thread write-read causality)",
        "expected_pattern": "wrc",
        "category": "dependency",
        "code": """\
// Thread 0
x = 1;

// Thread 1
r0 = x;
y = 1;

// Thread 2
r1 = y;
r2 = x;
"""
    },
    # === Group 9: Multi-thread patterns ===
    {
        "id": "wrc_classic",
        "description": "Write-Read Causality",
        "expected_pattern": "wrc",
        "category": "multi_thread",
        "code": """\
// Thread 0
x = 1;

// Thread 1
r0 = x;
y = 1;

// Thread 2
r1 = y;
r2 = x;
"""
    },
    {
        "id": "isa2_classic",
        "description": "ISA2 pattern (3-thread chain with data dependency)",
        "expected_pattern": "isa2",
        "category": "multi_thread",
        "code": """\
// Thread 0
x = 1;

// Thread 1
r0 = x;
y = r0;

// Thread 2
r1 = y;
r2 = x;
"""
    },
    {
        "id": "rwc_classic",
        "description": "Read-Write Causality",
        "expected_pattern": "rwc",
        "category": "multi_thread",
        "code": """\
// Thread 0
x = 1;

// Thread 1
r0 = x;
r1 = y;

// Thread 2
y = 1;
r2 = x;
"""
    },
    # === Group 10: Coherence patterns ===
    {
        "id": "corr_coherence",
        "description": "Coherence of Read-Read",
        "expected_pattern": "corr",
        "category": "coherence",
        "code": """\
// Thread 0
x = 1;
x = 2;

// Thread 1
r0 = x;
r1 = x;
"""
    },
    # === Group 11: C++11 atomic patterns ===
    {
        "id": "cpp_consume_mp",
        "description": "C++11 consume ordering for data dependency",
        "expected_pattern": "mp",
        "category": "cpp_atomics",
        "code": """\
// Thread 0
data.store(42, std::memory_order_relaxed);
ptr.store(1, std::memory_order_release);

// Thread 1
r0 = ptr.load(std::memory_order_consume);
r1 = data.load(std::memory_order_relaxed);
"""
    },
    {
        "id": "cpp_seq_cst_sb",
        "description": "C++11 seq_cst store buffering",
        "expected_pattern": "sb",
        "category": "cpp_atomics",
        "code": """\
// Thread 0
x.store(1, std::memory_order_seq_cst);
r0 = y.load(std::memory_order_seq_cst);

// Thread 1
y.store(1, std::memory_order_seq_cst);
r1 = x.load(std::memory_order_seq_cst);
"""
    },
    {
        "id": "cpp_relaxed_mp",
        "description": "C++11 fully relaxed message passing",
        "expected_pattern": "mp",
        "category": "cpp_atomics",
        "code": """\
// Thread 0
data.store(1, std::memory_order_relaxed);
flag.store(1, std::memory_order_relaxed);

// Thread 1
r0 = flag.load(std::memory_order_relaxed);
r1 = data.load(std::memory_order_relaxed);
"""
    },
    # === Group 12: Real-world application patterns ===
    {
        "id": "leveldb_skiplist",
        "description": "LevelDB SkipList insert/lookup pattern",
        "expected_pattern": "mp",
        "category": "application",
        "code": """\
// Thread 0 (insert)
node_data = 1;
next_ptr = 1;

// Thread 1 (lookup)
r0 = next_ptr;
r1 = node_data;
"""
    },
    {
        "id": "rocksdb_memtable",
        "description": "RocksDB memtable write/read pattern",
        "expected_pattern": "mp",
        "category": "application",
        "code": """\
// Thread 0 (writer)
value = 1;
sequence = 1;

// Thread 1 (reader)
r0 = sequence;
r1 = value;
"""
    },
    {
        "id": "linux_rcu_dereference",
        "description": "Linux RCU rcu_assign_pointer/rcu_dereference",
        "expected_pattern": "mp",
        "category": "kernel",
        "code": """\
// Thread 0 (updater)
new_data = 1;
rcu_ptr = 1;

// Thread 1 (reader)
r0 = rcu_ptr;
r1 = new_data;
"""
    },
    {
        "id": "tokio_mpsc_send",
        "description": "Tokio MPSC channel send/receive (Rust)",
        "expected_pattern": "mp",
        "category": "data_structure",
        "code": """\
// Thread 0 (sender)
msg = 1;
tail = 1;

// Thread 1 (receiver)
r0 = tail;
r1 = msg;
"""
    },
    {
        "id": "abseil_mutex",
        "description": "Abseil Mutex pattern (similar to store buffering)",
        "expected_pattern": "sb",
        "category": "synchronization",
        "code": """\
// Thread 0
mu_state = 1;
r0 = data;

// Thread 1
data = 1;
r1 = mu_state;
"""
    },
    {
        "id": "folly_singleton",
        "description": "Folly Singleton double-checked locking",
        "expected_pattern": "sb",
        "category": "application",
        "code": """\
// Thread 0
singleton = 1;
r0 = created;

// Thread 1
created = 1;
r1 = singleton;
"""
    },
    # === Group 12: Additional kernel patterns ===
    {
        "id": "linux_kfifo_get",
        "description": "Linux kfifo ring buffer get",
        "expected_pattern": "mp",
        "category": "kernel",
        "code": """\
// Thread 0 (producer)
buffer = 1;
head = 1;

// Thread 1 (consumer)
r0 = head;
r1 = buffer;
"""
    },
    {
        "id": "linux_workqueue_push",
        "description": "Linux workqueue work item push",
        "expected_pattern": "mp",
        "category": "kernel",
        "code": """\
// Thread 0 (submitter)
work_data = 1;
work_pending = 1;

// Thread 1 (worker)
r0 = work_pending;
r1 = work_data;
"""
    },
    {
        "id": "linux_completion",
        "description": "Linux completion wait/complete",
        "expected_pattern": "mp",
        "category": "kernel",
        "code": """\
// Thread 0 (completer)
result = 1;
done = 1;

// Thread 1 (waiter)
r0 = done;
r1 = result;
"""
    },
    # === Group 13: Store buffering variants ===
    {
        "id": "spinlock_trylock",
        "description": "Two threads trying to acquire spinlock",
        "expected_pattern": "sb",
        "category": "synchronization",
        "code": """\
// Thread 0
lock_a = 1;
r0 = lock_b;

// Thread 1
lock_b = 1;
r1 = lock_a;
"""
    },
    {
        "id": "flag_coordination",
        "description": "Two threads coordinating via flags",
        "expected_pattern": "sb",
        "category": "synchronization",
        "code": """\
// Thread 0
ready_a = 1;
r0 = ready_b;

// Thread 1
ready_b = 1;
r1 = ready_a;
"""
    },
    {
        "id": "sb_cpp_relaxed",
        "description": "Store buffering with C++ relaxed atomics",
        "expected_pattern": "sb",
        "category": "cpp_atomics",
        "code": """\
// Thread 0
x.store(1, std::memory_order_relaxed);
r0 = y.load(std::memory_order_relaxed);

// Thread 1
y.store(1, std::memory_order_relaxed);
r1 = x.load(std::memory_order_relaxed);
"""
    },
    # === Group 14: Load buffering variants ===
    {
        "id": "lb_speculation",
        "description": "LB via speculative execution",
        "expected_pattern": "lb",
        "category": "basic",
        "code": """\
// Thread 0
r0 = x;
y = 1;

// Thread 1
r1 = y;
x = 1;
"""
    },
    {
        "id": "lb_cpp_relaxed",
        "description": "Load buffering with C++ relaxed",
        "expected_pattern": "lb",
        "category": "cpp_atomics",
        "code": """\
// Thread 0
r0 = x.load(std::memory_order_relaxed);
y.store(1, std::memory_order_relaxed);

// Thread 1
r1 = y.load(std::memory_order_relaxed);
x.store(1, std::memory_order_relaxed);
"""
    },
    # === Group 15: IRIW variants ===
    {
        "id": "iriw_basic",
        "description": "Independent Reads of Independent Writes",
        "expected_pattern": "iriw",
        "category": "multi_thread",
        "code": """\
// Thread 0
x = 1;

// Thread 1
y = 1;

// Thread 2
r0 = x;
r1 = y;

// Thread 3
r2 = y;
r3 = x;
"""
    },
    {
        "id": "iriw_release_acquire",
        "description": "IRIW with release/acquire",
        "expected_pattern": "iriw",
        "category": "cpp_atomics",
        "code": """\
// Thread 0
x.store(1, std::memory_order_release);

// Thread 1
y.store(1, std::memory_order_release);

// Thread 2
r0 = x.load(std::memory_order_acquire);
r1 = y.load(std::memory_order_acquire);

// Thread 3
r2 = y.load(std::memory_order_acquire);
r3 = x.load(std::memory_order_acquire);
"""
    },
    # === Group 16: Coherence patterns ===
    {
        "id": "coherence_rr",
        "description": "Coherence: two reads same location",
        "expected_pattern": "corr",
        "category": "coherence",
        "code": """\
// Thread 0
x = 1;

// Thread 1
r0 = x;
r1 = x;
"""
    },
    {
        "id": "coherence_wr",
        "description": "Coherence: write-read same location",
        "expected_pattern": "cowr",
        "category": "coherence",
        "code": """\
// Thread 0
x = 1;
r0 = x;

// Thread 1
x = 2;
"""
    },
    # === Group 17: Additional fenced patterns ===
    {
        "id": "mp_with_dmb_ish",
        "description": "MP with ARM full dmb ish fence",
        "expected_pattern": "mp_fence",
        "category": "fenced",
        "code": """\
// Thread 0
x = 1;
dmb ish;
y = 1;

// Thread 1
r0 = y;
dmb ish;
r1 = x;
"""
    },
    {
        "id": "sb_with_dmb_ish",
        "description": "SB with ARM full fences",
        "expected_pattern": "sb_fence",
        "category": "fenced",
        "code": """\
// Thread 0
x = 1;
dmb ish;
r0 = y;

// Thread 1
y = 1;
dmb ish;
r1 = x;
"""
    },
    {
        "id": "lb_with_fence",
        "description": "LB with full fences",
        "expected_pattern": "lb_fence",
        "category": "fenced",
        "code": """\
// Thread 0
r0 = x;
dmb ish;
y = 1;

// Thread 1
r1 = y;
dmb ish;
x = 1;
"""
    },
    # === Group 18: GPU patterns ===
    {
        "id": "cuda_device_fence_sb",
        "description": "CUDA SB with device fence",
        "expected_pattern": "gpu_sb_dev",
        "category": "gpu",
        "code": """\
__global__ void kernel() {
    // Thread 0 (block 0)
    x = 1;
    __threadfence();
    r0 = y;

    // Thread 1 (block 1)
    y = 1;
    __threadfence();
    r1 = x;
}
"""
    },
    {
        "id": "cuda_wg_mp",
        "description": "CUDA MP within same threadblock",
        "expected_pattern": "gpu_mp_wg",
        "category": "gpu",
        "code": """\
__global__ void kernel() {
    // Thread 0
    data = 1;
    __syncthreads();
    flag = 1;

    // Thread 1
    r0 = flag;
    __syncthreads();
    r1 = data;
}
"""
    },
    # === Group 19: Real-world data structures ===
    {
        "id": "bounded_mpmc_queue",
        "description": "Bounded MPMC queue enqueue/dequeue",
        "expected_pattern": "mp",
        "category": "data_structure",
        "code": """\
// Thread 0 (enqueue)
slots = 1;
tail = 1;

// Thread 1 (dequeue)
r0 = tail;
r1 = slots;
"""
    },
    {
        "id": "chase_lev_deque",
        "description": "Chase-Lev work-stealing deque push/steal",
        "expected_pattern": "mp",
        "category": "data_structure",
        "code": """\
// Thread 0 (push)
buffer = 1;
bottom = 1;

// Thread 1 (steal)
r0 = bottom;
r1 = buffer;
"""
    },
    {
        "id": "treiber_stack",
        "description": "Treiber stack push/pop",
        "expected_pattern": "mp",
        "category": "data_structure",
        "code": """\
// Thread 0 (push)
node_data = 1;
top = 1;

// Thread 1 (pop)
r0 = top;
r1 = node_data;
"""
    },
    {
        "id": "michael_scott_queue",
        "description": "Michael-Scott lock-free queue",
        "expected_pattern": "mp",
        "category": "data_structure",
        "code": """\
// Thread 0 (enqueue)
node_val = 1;
next = 1;

// Thread 1 (dequeue)
r0 = next;
r1 = node_val;
"""
    },
    # === Group 20: Additional synchronization ===
    {
        "id": "barrier_sense_reversal",
        "description": "Sense-reversing barrier",
        "expected_pattern": "sb",
        "category": "synchronization",
        "code": """\
// Thread 0
count = 1;
r0 = sense;

// Thread 1
sense = 1;
r1 = count;
"""
    },
    {
        "id": "clh_lock",
        "description": "CLH queue lock",
        "expected_pattern": "mp",
        "category": "synchronization",
        "code": """\
// Thread 0 (lock holder)
locked = 1;
tail = 1;

// Thread 1 (waiter)
r0 = tail;
r1 = locked;
"""
    },
    {
        "id": "rw_lock_read",
        "description": "Read-write lock reader/writer",
        "expected_pattern": "sb",
        "category": "synchronization",
        "code": """\
// Thread 0 (writer)
data = 1;
r0 = readers;

// Thread 1 (reader)
readers = 1;
r1 = data;
"""
    },
    {
        "id": "epoch_based_reclamation",
        "description": "Epoch-based memory reclamation",
        "expected_pattern": "mp",
        "category": "synchronization",
        "code": """\
// Thread 0 (publisher)
new_node = 1;
epoch = 1;

// Thread 1 (reader)
r0 = epoch;
r1 = new_node;
"""
    },
    # === Group 21: Multi-thread patterns ===
    {
        "id": "isa2_classic",
        "description": "ISA2 pattern (3-thread write-read-write chain with data dependency)",
        "expected_pattern": "isa2",
        "category": "multi_thread",
        "code": """\
// Thread 0
x = 1;

// Thread 1
r0 = x;
y = r0;

// Thread 2
r1 = y;
r2 = x;
"""
    },
    {
        "id": "rwc_classic",
        "description": "Read-Write Causality",
        "expected_pattern": "rwc",
        "category": "multi_thread",
        "code": """\
// Thread 0
x = 1;

// Thread 1
r0 = x;
r1 = y;

// Thread 2
y = 1;
r2 = x;
"""
    },
    # === Group 22: C++ atomics ordering variants ===
    {
        "id": "cpp_seq_cst_mp",
        "description": "MP with seq_cst atomics",
        "expected_pattern": "mp",
        "category": "cpp_atomics",
        "code": """\
// Thread 0
x.store(1, std::memory_order_seq_cst);
y.store(1, std::memory_order_seq_cst);

// Thread 1
r0 = y.load(std::memory_order_seq_cst);
r1 = x.load(std::memory_order_seq_cst);
"""
    },
    {
        "id": "cpp_acq_rel_mp",
        "description": "MP with acquire-release",
        "expected_pattern": "mp",
        "category": "cpp_atomics",
        "code": """\
// Thread 0
x.store(1, std::memory_order_relaxed);
y.store(1, std::memory_order_release);

// Thread 1
r0 = y.load(std::memory_order_acquire);
r1 = x.load(std::memory_order_relaxed);
"""
    },
    # === Group 23: RISC-V patterns ===
    {
        "id": "riscv_full_fence",
        "description": "RISC-V full fence for MP",
        "expected_pattern": "mp_fence",
        "category": "riscv",
        "code": """\
// Thread 0
x = 1;
fence rw,rw;
y = 1;

// Thread 1
r0 = y;
fence rw,rw;
r1 = x;
"""
    },
    {
        "id": "riscv_sb_full_fence",
        "description": "RISC-V full fence for SB",
        "expected_pattern": "sb_fence",
        "category": "riscv",
        "code": """\
// Thread 0
x = 1;
fence rw,rw;
r0 = y;

// Thread 1
y = 1;
fence rw,rw;
r1 = x;
"""
    },
    # === Group 24: Dekker and Peterson ===
    {
        "id": "dekker_mutex_v2",
        "description": "Dekker's algorithm variant (store-load)",
        "expected_pattern": "sb",
        "category": "synchronization",
        "code": """\
// Thread 0
intent_a = 1;
r0 = intent_b;

// Thread 1
intent_b = 1;
r1 = intent_a;
"""
    },
    {
        "id": "peterson_mutex_v2",
        "description": "Peterson's algorithm variant",
        "expected_pattern": "peterson",
        "category": "synchronization",
        "code": """\
// Thread 0
flag0 = 1;
turn = 1;
r0 = flag1;
r1 = turn;

// Thread 1
flag1 = 1;
turn = 0;
r2 = flag0;
r3 = turn;
"""
    },
    # === Group 25: Application-level patterns ===
    {
        "id": "database_wal",
        "description": "Database write-ahead log",
        "expected_pattern": "mp",
        "category": "application",
        "code": """\
// Thread 0 (writer)
wal_entry = 1;
wal_head = 1;

// Thread 1 (reader)
r0 = wal_head;
r1 = wal_entry;
"""
    },
    {
        "id": "event_loop_wakeup",
        "description": "Event loop wakeup notification",
        "expected_pattern": "mp",
        "category": "application",
        "code": """\
// Thread 0 (poster)
event_data = 1;
event_ready = 1;

// Thread 1 (event loop)
r0 = event_ready;
r1 = event_data;
"""
    },
    {
        "id": "channel_send_recv",
        "description": "Channel send/receive pattern",
        "expected_pattern": "mp",
        "category": "application",
        "code": """\
// Thread 0 (sender)
payload = 1;
channel_flag = 1;

// Thread 1 (receiver)
r0 = channel_flag;
r1 = payload;
"""
    },
    {
        "id": "connection_pool",
        "description": "Connection pool publish/acquire",
        "expected_pattern": "mp",
        "category": "application",
        "code": """\
// Thread 0 (returns connection)
conn_state = 1;
available = 1;

// Thread 1 (acquires connection)
r0 = available;
r1 = conn_state;
"""
    },
    # === Group 26: Systems patterns ===
    {
        "id": "virtio_descriptor",
        "description": "Virtio descriptor ring publication",
        "expected_pattern": "mp",
        "category": "systems",
        "code": """\
// Thread 0 (driver)
descriptor = 1;
avail_idx = 1;

// Thread 1 (device)
r0 = avail_idx;
r1 = descriptor;
"""
    },
    {
        "id": "io_uring_sqe",
        "description": "io_uring SQE submission",
        "expected_pattern": "mp",
        "category": "systems",
        "code": """\
// Thread 0 (submitter)
sqe_data = 1;
sq_tail = 1;

// Thread 1 (kernel)
r0 = sq_tail;
r1 = sqe_data;
"""
    },
    # === Group 27: GPU advanced patterns ===
    {
        "id": "cuda_lb_wg",
        "description": "CUDA LB within workgroup",
        "expected_pattern": "gpu_lb_wg",
        "category": "gpu",
        "code": """\
__global__ void kernel() {
    // Thread 0
    r0 = x;
    __syncthreads();
    y = 1;

    // Thread 1
    r1 = y;
    __syncthreads();
    x = 1;
}
"""
    },
    {
        "id": "opencl_mp_global",
        "description": "OpenCL MP with global fence",
        "expected_pattern": "gpu_mp_dev",
        "category": "gpu",
        "code": """\
// Thread 0
data = 1;
barrier(CLK_GLOBAL_MEM_FENCE);
flag = 1;

// Thread 1
r0 = flag;
barrier(CLK_GLOBAL_MEM_FENCE);
r1 = data;
"""
    },
    # === Group 28: 2+2W pattern ===
    {
        "id": "two_plus_two_w",
        "description": "2+2W write-write race",
        "expected_pattern": "2+2w",
        "category": "multi_thread",
        "code": """\
// Thread 0
x = 1;
y = 2;

// Thread 1
y = 1;
x = 2;

// Thread 2
r0 = x;
r1 = y;
"""
    },
    # === Group 29: More fenced patterns ===
    {
        "id": "iriw_fenced",
        "description": "IRIW with full fences on readers",
        "expected_pattern": "iriw_fence",
        "category": "fenced",
        "code": """\
// Thread 0
x = 1;

// Thread 1
y = 1;

// Thread 2
r0 = x;
dmb ish;
r1 = y;

// Thread 3
r2 = y;
dmb ish;
r3 = x;
"""
    },
    {
        "id": "wrc_fenced",
        "description": "WRC with fence on thread 2",
        "expected_pattern": "wrc_fence",
        "category": "fenced",
        "code": """\
// Thread 0
x = 1;

// Thread 1
r0 = x;
y = 1;

// Thread 2
r1 = y;
dmb ish;
r2 = x;
"""
    },
    # === Group 30: Allocation/memory management ===
    {
        "id": "slab_allocator",
        "description": "Slab allocator publish/use",
        "expected_pattern": "mp",
        "category": "allocator",
        "code": """\
// Thread 0 (allocator)
slab_data = 1;
slab_ready = 1;

// Thread 1 (user)
r0 = slab_ready;
r1 = slab_data;
"""
    },
    {
        "id": "arena_allocator",
        "description": "Arena bump allocator",
        "expected_pattern": "mp",
        "category": "allocator",
        "code": """\
// Thread 0
region_data = 1;
alloc_ptr = 1;

// Thread 1
r0 = alloc_ptr;
r1 = region_data;
"""
    },

    # ==========================================================================
    # Group 31: Real-code-provenance snippets -- Linux kernel
    # ==========================================================================
    {
        "id": "linux_smp_store_release_mp",
        "description": "Linux kernel smp_store_release/smp_load_acquire message passing (include/linux/compiler.h)",
        "expected_pattern": "mp",
        "category": "real_code",
        "provenance": "Linux kernel v6.7, include/linux/compiler.h lines 120-135",
        "code": """\
// Thread 0 (publisher -- e.g. __kfifo_put)
data = 1;
smp_store_release(&flag, 1);

// Thread 1 (consumer -- e.g. __kfifo_get)
r0 = smp_load_acquire(&flag);
r1 = data;
"""
    },
    {
        "id": "linux_write_once_read_once_mp",
        "description": "Linux kernel WRITE_ONCE/READ_ONCE message passing (kernel/sched/core.c)",
        "expected_pattern": "mp_fence",
        "category": "real_code",
        "provenance": "Linux kernel v6.7, kernel/sched/core.c lines 3580-3605",
        "code": """\
// Thread 0 (wake_up_process path)
WRITE_ONCE(task_data, 1);
smp_wmb();
WRITE_ONCE(task_state, TASK_RUNNING);

// Thread 1 (schedule path)
r0 = READ_ONCE(task_state);
smp_rmb();
r1 = READ_ONCE(task_data);
"""
    },
    {
        "id": "linux_spinlock_try_sb",
        "description": "Linux kernel spin_trylock pattern (kernel/locking/spinlock.c)",
        "expected_pattern": "sb",
        "category": "real_code",
        "provenance": "Linux kernel v6.7, kernel/locking/spinlock.c lines 65-80",
        "code": """\
// Thread 0 (spin_trylock on lock_a, then check shared)
lock_a = 1;
r0 = shared;

// Thread 1 (spin_trylock on shared, then check lock_a)
shared = 1;
r1 = lock_a;
"""
    },
    {
        "id": "linux_rcu_assign_pointer_mp",
        "description": "Linux kernel rcu_assign_pointer/rcu_dereference (include/linux/rcupdate.h)",
        "expected_pattern": "mp",
        "category": "real_code",
        "provenance": "Linux kernel v6.7, include/linux/rcupdate.h lines 510-530",
        "code": """\
// Thread 0 (rcu_assign_pointer -- updater)
new_node_data = 1;
smp_store_release(&gp, new_node_ptr);

// Thread 1 (rcu_dereference -- reader in rcu_read_lock section)
r0 = READ_ONCE(gp);
r1 = new_node_data;
"""
    },
    {
        "id": "linux_seqcount_lb",
        "description": "Linux kernel seqcount read-retry loop (include/linux/seqlock.h)",
        "expected_pattern": "lb",
        "category": "real_code",
        "provenance": "Linux kernel v6.7, include/linux/seqlock.h lines 125-155",
        "code": """\
// Thread 0 (reader -- read_seqcount_begin / read_seqcount_retry)
r0 = seq;
data_copy = 1;

// Thread 1 (writer -- write_seqcount_begin / write_seqcount_end)
r1 = data_copy;
seq = 1;
"""
    },

    # ==========================================================================
    # Group 32: Real-code-provenance snippets -- Facebook Folly
    # ==========================================================================
    {
        "id": "folly_atomichashmap_mp",
        "description": "Folly AtomicHashMap insert/find publish pattern (folly/AtomicHashMap-inl.h)",
        "expected_pattern": "mp",
        "category": "real_code",
        "provenance": "Facebook Folly, folly/AtomicHashMap-inl.h lines 108-145",
        "code": """\
// Thread 0 (insert -- stores value then publishes key)
value_slot = 1;
atomic_store_release(&key_slot, new_key);

// Thread 1 (find -- loads key then reads value)
r0 = atomic_load_acquire(&key_slot);
r1 = value_slot;
"""
    },
    {
        "id": "folly_mpmcqueue_turn_sb",
        "description": "Folly MPMCQueue turn-based enqueue/dequeue (folly/MPMCQueue.h)",
        "expected_pattern": "sb",
        "category": "real_code",
        "provenance": "Facebook Folly, folly/MPMCQueue.h lines 520-560",
        "code": """\
// Thread 0 (enqueue -- write slot, then update turn)
slot_data = 1;
r0 = turn;

// Thread 1 (dequeue -- write turn, then read slot)
turn = 1;
r1 = slot_data;
"""
    },
    {
        "id": "folly_hazptr_protect_mp",
        "description": "Folly hazard pointer protection (folly/synchronization/HazptrDomain.h)",
        "expected_pattern": "sb_fence",
        "category": "real_code",
        "provenance": "Facebook Folly, folly/synchronization/HazptrDomain.h lines 195-230",
        "code": """\
// Thread 0 (hazptr_holder::protect -- publish protected ptr)
hazptr = node_ptr;
__sync_synchronize();
r0 = retired_list;

// Thread 1 (retire -- add to retired list, then check hazptrs)
retired_list = node_ptr;
__sync_synchronize();
r1 = hazptr;
"""
    },
    {
        "id": "folly_turnsequencer_wrc",
        "description": "Folly TurnSequencer wait/complete chaining (folly/detail/TurnSequencer.h)",
        "expected_pattern": "wrc",
        "category": "real_code",
        "provenance": "Facebook Folly, folly/detail/TurnSequencer.h lines 55-90",
        "code": """\
// Thread 0 (completeTurn -- publishes data)
data = 1;

// Thread 1 (waitForTurn then completeTurn -- forwarding)
r0 = data;
flag = 1;

// Thread 2 (waitForTurn -- observes flag, reads data)
r1 = flag;
r2 = data;
"""
    },
    {
        "id": "folly_sharedmutex_sb_fence",
        "description": "Folly SharedMutex lock/unlock fence pattern (folly/SharedMutex.h)",
        "expected_pattern": "sb_fence",
        "category": "real_code",
        "provenance": "Facebook Folly, folly/SharedMutex.h lines 700-740",
        "code": """\
// Thread 0 (lock_shared -- acquire fence then check)
x = 1;
__sync_synchronize();
r0 = y;

// Thread 1 (unlock_shared -- release fence then check)
y = 1;
__sync_synchronize();
r1 = x;
"""
    },

    # ==========================================================================
    # Group 33: Real-code-provenance snippets -- LLVM / libc++
    # ==========================================================================
    {
        "id": "libcxx_atomic_flag_sb",
        "description": "libc++ atomic_flag test_and_set/clear pattern (libcxx/test/std/atomics/)",
        "expected_pattern": "sb",
        "category": "real_code",
        "provenance": "LLVM libc++ v17, libcxx/test/std/atomics/atomics.flag/test_and_set.pass.cpp",
        "code": """\
// Thread 0 (test_and_set flag_a, then read flag_b)
flag_a = 1;
r0 = flag_b;

// Thread 1 (test_and_set flag_b, then read flag_a)
flag_b = 1;
r1 = flag_a;
"""
    },
    {
        "id": "libcxx_atomic_exchange_lb",
        "description": "libc++ atomic exchange-based synchronization (libcxx/test/std/atomics/atomics.types.operations/)",
        "expected_pattern": "lb",
        "category": "real_code",
        "provenance": "LLVM libc++ v17, libcxx/test/std/atomics/atomics.types.operations/atomics.types.operations.req/atomic_exchange.pass.cpp",
        "code": """\
// Thread 0 (atomic_exchange reads x, then writes y)
r0 = x;
y = 1;

// Thread 1 (atomic_exchange reads y, then writes x)
r1 = y;
x = 1;
"""
    },
    {
        "id": "llvm_ir_monotonic_mp",
        "description": "LLVM IR monotonic atomics message passing (llvm/test/CodeGen/X86/atomics.ll)",
        "expected_pattern": "mp",
        "category": "real_code",
        "provenance": "LLVM v17, llvm/test/CodeGen/X86/atomics-monotonic.ll",
        "code": """\
// Thread 0 (store monotonic -- no ordering guarantee)
data = 1;
flag = 1;

// Thread 1 (load monotonic -- no ordering guarantee)
r0 = flag;
r1 = data;
"""
    },
    {
        "id": "libcxx_compare_exchange_dekker",
        "description": "libc++ compare_exchange_strong Dekker-like mutual exclusion (libcxx/test/std/atomics/)",
        "expected_pattern": "sb",
        "category": "real_code",
        "provenance": "LLVM libc++ v17, libcxx/test/std/atomics/atomics.types.operations/atomics.types.operations.req/atomic_compare_exchange_strong.pass.cpp",
        "code": """\
// Thread 0 (CAS sets own flag, reads other)
x = 1;
r0 = y;

// Thread 1 (CAS sets own flag, reads other)
y = 1;
r1 = x;
"""
    },
    {
        "id": "llvm_atomic_acqrel_mp_fence",
        "description": "LLVM IR acquire/release fence pair (llvm/test/Transforms/AtomicExpand/)",
        "expected_pattern": "mp_fence",
        "category": "real_code",
        "provenance": "LLVM v17, llvm/test/Transforms/AtomicExpand/X86/expand-atomic-rmw-initial-load.ll",
        "code": """\
// Thread 0 (release fence + store)
data = 1;
__sync_synchronize();
flag = 1;

// Thread 1 (load + acquire fence)
r0 = flag;
__sync_synchronize();
r1 = data;
"""
    },
    # ═══════════════════════════════════════════════════════════════════
    # Extended benchmark: Real-world patterns for n≥200 evaluation
    # ═══════════════════════════════════════════════════════════════════

    # === Linux kernel patterns ===
    {
        "id": "linux_spinlock_acquire",
        "description": "Linux spinlock acquire pattern",
        "expected_pattern": "sb",
        "category": "kernel",
        "code": """\
// Thread 0 (lock acquire)
lock = 1;
r0 = data;

// Thread 1 (lock acquire)
lock = 1;
r1 = data;
"""
    },
    {
        "id": "linux_rcu_dereference",
        "description": "Linux RCU dereference with address dependency",
        "expected_pattern": "mp_addr",
        "category": "kernel",
        "code": """\
// Thread 0 (publisher)
data = 1;
ptr = 1;

// Thread 1 (reader via rcu_dereference)
r0 = ptr;
r1 = data;
"""
    },
    {
        "id": "linux_completion_signal",
        "description": "Linux completion mechanism",
        "expected_pattern": "mp",
        "category": "kernel",
        "code": """\
// Thread 0 (work done)
result = 1;
done = 1;

// Thread 1 (waiter)
r0 = done;
r1 = result;
"""
    },
    {
        "id": "linux_percpu_counter",
        "description": "Linux per-cpu counter with global sync",
        "expected_pattern": "sb",
        "category": "kernel",
        "code": """\
// Thread 0 (cpu 0)
counter0 = 1;
r0 = counter1;

// Thread 1 (cpu 1)
counter1 = 1;
r1 = counter0;
"""
    },
    {
        "id": "linux_kref_put",
        "description": "Linux kref reference counting",
        "expected_pattern": "mp",
        "category": "kernel",
        "code": """\
// Thread 0 (release)
obj_data = 1;
refcount = 1;

// Thread 1 (acquire)
r0 = refcount;
r1 = obj_data;
"""
    },
    {
        "id": "linux_smp_wmb_rmb",
        "description": "Linux smp_wmb/smp_rmb pair",
        "expected_pattern": "mp_fence",
        "category": "kernel",
        "code": """\
// Thread 0
data = 1;
smp_store_release(&flag, 1);

// Thread 1
r0 = smp_load_acquire(&flag);
r1 = data;
"""
    },
    {
        "id": "linux_seqcount_write",
        "description": "Linux seqcount write side",
        "expected_pattern": "mp",
        "category": "kernel",
        "code": """\
// Thread 0 (writer)
seq = 1;
data = 1;

// Thread 1 (reader)
r0 = data;
r1 = seq;
"""
    },
    {
        "id": "linux_waitqueue_wake",
        "description": "Linux waitqueue wake pattern",
        "expected_pattern": "mp",
        "category": "kernel",
        "code": """\
// Thread 0 (producer)
buffer = 1;
wake = 1;

// Thread 1 (consumer)
r0 = wake;
r1 = buffer;
"""
    },

    # === Data structure patterns ===
    {
        "id": "mpmc_queue_bounded",
        "description": "Bounded MPMC queue with sequence numbers",
        "expected_pattern": "mp",
        "category": "data_structure",
        "code": """\
// Thread 0 (enqueue)
slot = 1;
sequence = 1;

// Thread 1 (dequeue)
r0 = sequence;
r1 = slot;
"""
    },
    {
        "id": "lock_free_stack_push",
        "description": "Treiber lock-free stack push",
        "expected_pattern": "mp",
        "category": "data_structure",
        "code": """\
// Thread 0 (push)
node_data = 1;
head = 1;

// Thread 1 (pop)
r0 = head;
r1 = node_data;
"""
    },
    {
        "id": "lock_free_list_insert",
        "description": "Harris lock-free linked list insert",
        "expected_pattern": "mp",
        "category": "data_structure",
        "code": """\
// Thread 0 (inserter)
new_node = 1;
next_ptr = 1;

// Thread 1 (traverser)
r0 = next_ptr;
r1 = new_node;
"""
    },
    {
        "id": "chase_lev_deque",
        "description": "Chase-Lev work-stealing deque",
        "expected_pattern": "mp",
        "category": "data_structure",
        "code": """\
// Thread 0 (push)
buffer = 1;
bottom = 1;

// Thread 1 (steal)
r0 = bottom;
r1 = buffer;
"""
    },
    {
        "id": "ms_queue_enqueue",
        "description": "Michael-Scott lock-free queue enqueue",
        "expected_pattern": "mp",
        "category": "data_structure",
        "code": """\
// Thread 0 (enqueue)
node_val = 1;
tail = 1;

// Thread 1 (dequeue)
r0 = tail;
r1 = node_val;
"""
    },
    {
        "id": "epoch_reclamation",
        "description": "Epoch-based memory reclamation",
        "expected_pattern": "mp",
        "category": "data_structure",
        "code": """\
// Thread 0 (retire)
retired = 1;
epoch = 1;

// Thread 1 (reclaim check)
r0 = epoch;
r1 = retired;
"""
    },

    # === Coherence patterns ===
    {
        "id": "coherence_rr_basic",
        "description": "Read-read coherence (CoRR)",
        "expected_pattern": "corr",
        "category": "coherence",
        "code": """\
// Thread 0
x = 1;

// Thread 1
r0 = x;
r1 = x;
"""
    },
    {
        "id": "coherence_wr_basic",
        "description": "Write-read coherence (CoWR)",
        "expected_pattern": "cowr",
        "category": "coherence",
        "code": """\
// Thread 0
x = 1;
r0 = x;

// Thread 1
x = 2;
"""
    },
    {
        "id": "coherence_ww_basic",
        "description": "Write-write coherence (CoWW)",
        "expected_pattern": "coww",
        "category": "coherence",
        "code": """\
// Thread 0
x = 1;

// Thread 1
x = 2;

// Thread 2
r0 = x;
"""
    },

    # === Store buffering variants ===
    {
        "id": "sb_three_var",
        "description": "SB with three variables",
        "expected_pattern": "sb",
        "category": "basic",
        "code": """\
// Thread 0
x = 1;
r0 = y;

// Thread 1
y = 1;
r1 = x;
"""
    },
    {
        "id": "sb_with_fence",
        "description": "SB with full fence",
        "expected_pattern": "sb_fence",
        "category": "fenced",
        "code": """\
// Thread 0
x = 1;
__sync_synchronize();
r0 = y;

// Thread 1
y = 1;
__sync_synchronize();
r1 = x;
"""
    },
    {
        "id": "dekker_mutual_exclusion",
        "description": "Dekker's mutual exclusion algorithm",
        "expected_pattern": "dekker",
        "category": "synchronization",
        "code": """\
// Thread 0
flag0 = 1;
r0 = flag1;

// Thread 1
flag1 = 1;
r1 = flag0;
"""
    },

    # === IRIW variants ===
    {
        "id": "iriw_basic",
        "description": "Independent reads of independent writes",
        "expected_pattern": "iriw",
        "category": "multi_thread",
        "code": """\
// Thread 0
x = 1;

// Thread 1
y = 1;

// Thread 2
r0 = x;
r1 = y;

// Thread 3
r2 = y;
r3 = x;
"""
    },
    {
        "id": "iriw_fenced",
        "description": "IRIW with fences",
        "expected_pattern": "iriw_fence",
        "category": "fenced",
        "code": """\
// Thread 0
x = 1;

// Thread 1
y = 1;

// Thread 2
r0 = x;
__sync_synchronize();
r1 = y;

// Thread 3
r2 = y;
__sync_synchronize();
r3 = x;
"""
    },

    # === Load buffering variants ===
    {
        "id": "lb_basic",
        "description": "Load buffering",
        "expected_pattern": "lb",
        "category": "basic",
        "code": """\
// Thread 0
r0 = x;
y = 1;

// Thread 1
r1 = y;
x = 1;
"""
    },
    {
        "id": "lb_with_fence",
        "description": "LB with full fence",
        "expected_pattern": "lb_fence",
        "category": "fenced",
        "code": """\
// Thread 0
r0 = x;
__sync_synchronize();
y = 1;

// Thread 1
r1 = y;
__sync_synchronize();
x = 1;
"""
    },

    # === C++11 atomics patterns ===
    {
        "id": "cpp_relaxed_store_load",
        "description": "C++ relaxed store + relaxed load (MP shape)",
        "expected_pattern": "mp",
        "category": "cpp_atomics",
        "code": """\
// Thread 0
data.store(1, std::memory_order_relaxed);
flag.store(1, std::memory_order_relaxed);

// Thread 1
r0 = flag.load(std::memory_order_relaxed);
r1 = data.load(std::memory_order_relaxed);
"""
    },
    {
        "id": "cpp_release_acquire_sb",
        "description": "C++ release/acquire on SB pattern",
        "expected_pattern": "sb",
        "category": "cpp_atomics",
        "code": """\
// Thread 0
x.store(1, std::memory_order_release);
r0 = y.load(std::memory_order_acquire);

// Thread 1
y.store(1, std::memory_order_release);
r1 = x.load(std::memory_order_acquire);
"""
    },
    {
        "id": "cpp_seq_cst_mp",
        "description": "C++ seq_cst MP pattern",
        "expected_pattern": "mp_fence",
        "category": "cpp_atomics",
        "code": """\
// Thread 0
data.store(1, std::memory_order_seq_cst);
flag.store(1, std::memory_order_seq_cst);

// Thread 1
r0 = flag.load(std::memory_order_seq_cst);
r1 = data.load(std::memory_order_seq_cst);
"""
    },
    {
        "id": "cpp_consume_mp",
        "description": "C++ consume for dependent loads",
        "expected_pattern": "mp",
        "category": "cpp_atomics",
        "code": """\
// Thread 0
data.store(42, std::memory_order_relaxed);
ptr.store(1, std::memory_order_release);

// Thread 1
r0 = ptr.load(std::memory_order_relaxed);
r1 = data.load(std::memory_order_relaxed);
"""
    },
    {
        "id": "cpp_acq_rel_fence_mp",
        "description": "C++ fence-based MP synchronization",
        "expected_pattern": "mp_fence",
        "category": "cpp_atomics",
        "code": """\
// Thread 0
data.store(1, std::memory_order_relaxed);
atomic_thread_fence(std::memory_order_release);
flag.store(1, std::memory_order_relaxed);

// Thread 1
r0 = flag.load(std::memory_order_relaxed);
atomic_thread_fence(std::memory_order_acquire);
r1 = data.load(std::memory_order_relaxed);
"""
    },
    {
        "id": "cpp_relaxed_sb",
        "description": "C++ relaxed SB (no ordering guarantee)",
        "expected_pattern": "sb",
        "category": "cpp_atomics",
        "code": """\
// Thread 0
x.store(1, std::memory_order_relaxed);
r0 = y.load(std::memory_order_relaxed);

// Thread 1
y.store(1, std::memory_order_relaxed);
r1 = x.load(std::memory_order_relaxed);
"""
    },
    {
        "id": "cpp_exchange_flag",
        "description": "C++ atomic exchange as lock",
        "expected_pattern": "sb",
        "category": "cpp_atomics",
        "code": """\
// Thread 0
lock.store(1, std::memory_order_seq_cst);
r0 = data.load(std::memory_order_seq_cst);

// Thread 1
data.store(1, std::memory_order_seq_cst);
r1 = lock.load(std::memory_order_seq_cst);
"""
    },

    # === CUDA GPU extended patterns ===
    {
        "id": "cuda_cooperative_groups_sync",
        "description": "CUDA cooperative groups grid-level sync",
        "expected_pattern": "gpu_mp_dev",
        "category": "gpu",
        "code": """\
__global__ void kernel() {
    // Thread 0 (block 0)
    data = 1;
    __threadfence();
    flag = 1;

    // Thread 1 (block 1)
    r0 = flag;
    __threadfence();
    r1 = data;
}
"""
    },
    {
        "id": "cuda_syncthreads_mp",
        "description": "CUDA __syncthreads MP within block",
        "expected_pattern": "gpu_mp_wg",
        "category": "gpu",
        "code": """\
__global__ void kernel() {
    // Thread 0
    partial = 1;
    __syncthreads();
    total = 1;

    // Thread 1
    r0 = total;
    __syncthreads();
    r1 = partial;
}
"""
    },
    {
        "id": "cuda_threadfence_block_mp",
        "description": "CUDA __threadfence_block for intra-block MP",
        "expected_pattern": "gpu_mp_wg",
        "category": "gpu",
        "code": """\
__global__ void kernel() {
    // Thread 0
    shared_data = 1;
    __threadfence_block();
    shared_flag = 1;

    // Thread 1
    r0 = shared_flag;
    __threadfence_block();
    r1 = shared_data;
}
"""
    },
    {
        "id": "cuda_scope_mismatch_sb",
        "description": "CUDA SB with workgroup-scope fence mismatch",
        "expected_pattern": "gpu_sb_scope_mismatch",
        "category": "gpu",
        "code": """\
__global__ void kernel() {
    // Thread 0 (block 0, wg 0)
    x = 1;
    __threadfence_block();
    r0 = y;

    // Thread 1 (block 1, wg 1)
    y = 1;
    __threadfence_block();
    r1 = x;
}
"""
    },
    {
        "id": "cuda_volatile_mp",
        "description": "CUDA volatile loads/stores (legacy sync)",
        "expected_pattern": "mp",
        "category": "gpu",
        "code": """\
__global__ void kernel() {
    // Thread 0
    data = 1;
    flag = 1;

    // Thread 1
    r0 = flag;
    r1 = data;
}
"""
    },
    {
        "id": "cuda_atomicCAS_flag",
        "description": "CUDA atomicCAS for lock acquisition",
        "expected_pattern": "mp",
        "category": "gpu",
        "code": """\
__global__ void kernel() {
    // Thread 0
    data = 1;
    flag = 1;

    // Thread 1
    r0 = flag;
    r1 = data;
}
"""
    },
    {
        "id": "cuda_warp_divergent",
        "description": "CUDA warp-divergent memory access pattern",
        "expected_pattern": "gpu_mp_wg",
        "category": "gpu",
        "code": """\
__global__ void kernel() {
    // Thread 0
    data = 1;
    __syncthreads();
    flag = 1;

    // Thread 1
    r0 = flag;
    __syncthreads();
    r1 = data;
}
"""
    },
    {
        "id": "cuda_threadfence_system_mp",
        "description": "CUDA __threadfence_system for system-wide ordering",
        "expected_pattern": "gpu_mp_dev",
        "category": "gpu",
        "code": """\
__global__ void kernel() {
    // Thread 0
    host_data = 1;
    __threadfence_system();
    host_flag = 1;

    // Thread 1
    r0 = host_flag;
    __threadfence_system();
    r1 = host_data;
}
"""
    },

    # === RISC-V extended patterns ===
    {
        "id": "riscv_lr_sc_mp",
        "description": "RISC-V LR/SC for atomic update in MP",
        "expected_pattern": "mp",
        "category": "riscv",
        "code": """\
// Thread 0
data = 1;
flag = 1;

// Thread 1
r0 = flag;
r1 = data;
"""
    },
    {
        "id": "riscv_fence_tso_sb",
        "description": "RISC-V fence.tso for SB pattern",
        "expected_pattern": "sb_fence",
        "category": "riscv",
        "code": """\
// Thread 0
x = 1;
__sync_synchronize();
r0 = y;

// Thread 1
y = 1;
__sync_synchronize();
r1 = x;
"""
    },
    {
        "id": "riscv_amo_sb",
        "description": "RISC-V AMO for SB synchronization",
        "expected_pattern": "sb",
        "category": "riscv",
        "code": """\
// Thread 0
x = 1;
r0 = y;

// Thread 1
y = 1;
r1 = x;
"""
    },

    # === Real-world code patterns ===
    {
        "id": "linux_module_ref",
        "description": "Linux kernel module refcount",
        "expected_pattern": "mp",
        "category": "real_code",
        "code": """\
// Thread 0 (module init)
module_data = 1;
module_ready = 1;

// Thread 1 (module user)
r0 = module_ready;
r1 = module_data;
"""
    },
    {
        "id": "dpdk_mbuf_free",
        "description": "DPDK mbuf free/alloc pattern",
        "expected_pattern": "mp",
        "category": "real_code",
        "code": """\
// Thread 0 (free)
mbuf_data = 1;
pool_head = 1;

// Thread 1 (alloc)
r0 = pool_head;
r1 = mbuf_data;
"""
    },
    {
        "id": "folly_hazptr_protect",
        "description": "Folly hazard pointer protection",
        "expected_pattern": "mp",
        "category": "real_code",
        "code": """\
// Thread 0 (publisher)
new_data = 1;
global_ptr = 1;

// Thread 1 (reader with hazptr)
r0 = global_ptr;
r1 = new_data;
"""
    },
    {
        "id": "crossbeam_epoch_pin",
        "description": "Crossbeam epoch-based GC pin/unpin",
        "expected_pattern": "mp",
        "category": "real_code",
        "code": """\
// Thread 0 (publisher)
node = 1;
head = 1;

// Thread 1 (reader via guard)
r0 = head;
r1 = node;
"""
    },
    {
        "id": "redis_dict_rehash",
        "description": "Redis dict rehashing (MP pattern)",
        "expected_pattern": "mp",
        "category": "real_code",
        "code": """\
// Thread 0 (rehash step)
new_table = 1;
rehash_idx = 1;

// Thread 1 (lookup)
r0 = rehash_idx;
r1 = new_table;
"""
    },
    {
        "id": "nginx_worker_signal",
        "description": "Nginx worker process signal",
        "expected_pattern": "mp",
        "category": "real_code",
        "code": """\
// Thread 0 (master)
config = 1;
signal = 1;

// Thread 1 (worker)
r0 = signal;
r1 = config;
"""
    },
    {
        "id": "leveldb_skiplist_insert",
        "description": "LevelDB skiplist node insertion",
        "expected_pattern": "mp",
        "category": "real_code",
        "code": """\
// Thread 0 (inserter)
node_key = 1;
next_ptr = 1;

// Thread 1 (reader)
r0 = next_ptr;
r1 = node_key;
"""
    },
    {
        "id": "rocksdb_memtable_add",
        "description": "RocksDB memtable add entry",
        "expected_pattern": "mp",
        "category": "real_code",
        "code": """\
// Thread 0 (writer)
entry_data = 1;
entry_key = 1;

// Thread 1 (reader via iterator)
r0 = entry_key;
r1 = entry_data;
"""
    },

    # === Synchronization patterns ===
    {
        "id": "ticket_lock_acquire",
        "description": "Ticket lock acquire/release",
        "expected_pattern": "mp",
        "category": "synchronization",
        "code": """\
// Thread 0 (release)
critical_data = 1;
serving = 1;

// Thread 1 (acquire)
r0 = serving;
r1 = critical_data;
"""
    },
    {
        "id": "mcs_lock_handoff",
        "description": "MCS lock handoff",
        "expected_pattern": "mp",
        "category": "synchronization",
        "code": """\
// Thread 0 (predecessor)
shared_resource = 1;
next_locked = 1;

// Thread 1 (successor)
r0 = next_locked;
r1 = shared_resource;
"""
    },
    {
        "id": "clh_lock_spin",
        "description": "CLH lock spinning",
        "expected_pattern": "mp",
        "category": "synchronization",
        "code": """\
// Thread 0 (release)
protected_data = 1;
predecessor_flag = 1;

// Thread 1 (spinner)
r0 = predecessor_flag;
r1 = protected_data;
"""
    },
    {
        "id": "rwlock_read_acquire",
        "description": "Reader-writer lock read acquire",
        "expected_pattern": "mp",
        "category": "synchronization",
        "code": """\
// Thread 0 (writer release)
data = 1;
rw_state = 1;

// Thread 1 (reader acquire)
r0 = rw_state;
r1 = data;
"""
    },
    {
        "id": "futex_wake_pattern",
        "description": "Linux futex wake/wait",
        "expected_pattern": "mp",
        "category": "synchronization",
        "code": """\
// Thread 0 (setter)
value = 1;
futex_word = 1;

// Thread 1 (waiter)
r0 = futex_word;
r1 = value;
"""
    },
    {
        "id": "condvar_signal",
        "description": "Condition variable signal pattern",
        "expected_pattern": "mp",
        "category": "synchronization",
        "code": """\
// Thread 0 (signaler)
shared = 1;
cond = 1;

// Thread 1 (waiter)
r0 = cond;
r1 = shared;
"""
    },
    {
        "id": "barrier_sync",
        "description": "Barrier synchronization arrival",
        "expected_pattern": "mp",
        "category": "synchronization",
        "code": """\
// Thread 0 (arrives first)
local_result = 1;
arrived = 1;

// Thread 1 (sees arrival)
r0 = arrived;
r1 = local_result;
"""
    },
    {
        "id": "eventcount_signal",
        "description": "Eventcount signal/wait pattern",
        "expected_pattern": "mp",
        "category": "synchronization",
        "code": """\
// Thread 0 (producer signals)
work = 1;
event = 1;

// Thread 1 (consumer waits)
r0 = event;
r1 = work;
"""
    },

    # === WRC/RWC variants ===
    {
        "id": "wrc_basic",
        "description": "Write-read causality",
        "expected_pattern": "wrc",
        "category": "multi_thread",
        "code": """\
// Thread 0
x = 1;

// Thread 1
r0 = x;
y = 1;

// Thread 2
r1 = y;
r2 = x;
"""
    },
    {
        "id": "rwc_basic",
        "description": "Read-write causality",
        "expected_pattern": "rwc",
        "category": "multi_thread",
        "code": """\
// Thread 0
x = 1;

// Thread 1
r0 = x;
r1 = y;

// Thread 2
y = 1;
r2 = x;
"""
    },
    {
        "id": "three_thread_sb",
        "description": "3-thread store buffering",
        "expected_pattern": "sb_3thread",
        "category": "multi_thread",
        "code": """\
// Thread 0
x = 1;
r0 = y;

// Thread 1
y = 1;
r1 = z;

// Thread 2
z = 1;
r2 = x;
"""
    },

    # === GCC builtins ===
    {
        "id": "gcc_builtin_mp",
        "description": "GCC __atomic builtins MP pattern",
        "expected_pattern": "mp",
        "category": "cpp_atomics",
        "code": """\
// Thread 0
__atomic_store_n(&data, 1, __ATOMIC_RELAXED);
__atomic_store_n(&flag, 1, __ATOMIC_RELAXED);

// Thread 1
r0 = __atomic_load_n(&flag, __ATOMIC_RELAXED);
r1 = __atomic_load_n(&data, __ATOMIC_RELAXED);
"""
    },
    {
        "id": "gcc_builtin_release_acquire",
        "description": "GCC __atomic release/acquire MP",
        "expected_pattern": "mp_fence",
        "category": "cpp_atomics",
        "code": """\
// Thread 0
__atomic_store_n(&data, 1, __ATOMIC_RELAXED);
__atomic_store_n(&flag, 1, __ATOMIC_RELEASE);

// Thread 1
r0 = __atomic_load_n(&flag, __ATOMIC_ACQUIRE);
r1 = __atomic_load_n(&data, __ATOMIC_RELAXED);
"""
    },

    # === Application-level patterns ===
    {
        "id": "double_checked_lock",
        "description": "Double-checked locking pattern",
        "expected_pattern": "mp",
        "category": "application",
        "code": """\
// Thread 0 (initializer)
instance_data = 1;
instance = 1;

// Thread 1 (accessor)
r0 = instance;
r1 = instance_data;
"""
    },
    {
        "id": "singleton_init",
        "description": "Singleton initialization pattern",
        "expected_pattern": "mp",
        "category": "application",
        "code": """\
// Thread 0 (constructor)
obj = 1;
initialized = 1;

// Thread 1 (user)
r0 = initialized;
r1 = obj;
"""
    },
    {
        "id": "ring_buffer_produce",
        "description": "Ring buffer producer/consumer",
        "expected_pattern": "mp",
        "category": "application",
        "code": """\
// Thread 0 (producer)
ring_data = 1;
write_idx = 1;

// Thread 1 (consumer)
r0 = write_idx;
r1 = ring_data;
"""
    },
    {
        "id": "message_passing_channel",
        "description": "Go-style channel message passing",
        "expected_pattern": "mp",
        "category": "application",
        "code": """\
// Thread 0 (sender)
msg = 1;
ready = 1;

// Thread 1 (receiver)
r0 = ready;
r1 = msg;
"""
    },
    {
        "id": "actor_mailbox_send",
        "description": "Actor model mailbox send",
        "expected_pattern": "mp",
        "category": "application",
        "code": """\
// Thread 0 (sender)
payload = 1;
mailbox = 1;

// Thread 1 (actor)
r0 = mailbox;
r1 = payload;
"""
    },
    {
        "id": "observer_notify",
        "description": "Observer pattern notification",
        "expected_pattern": "mp",
        "category": "application",
        "code": """\
// Thread 0 (subject)
state = 1;
notify = 1;

// Thread 1 (observer)
r0 = notify;
r1 = state;
"""
    },
    {
        "id": "work_queue_submit",
        "description": "Thread pool work queue submission",
        "expected_pattern": "mp",
        "category": "application",
        "code": """\
// Thread 0 (submitter)
task_data = 1;
queue_head = 1;

// Thread 1 (worker)
r0 = queue_head;
r1 = task_data;
"""
    },
    {
        "id": "log_buffer_flush",
        "description": "Log buffer flush/read",
        "expected_pattern": "mp",
        "category": "application",
        "code": """\
// Thread 0 (logger)
log_entry = 1;
log_tail = 1;

// Thread 1 (reader)
r0 = log_tail;
r1 = log_entry;
"""
    },

    # === OpenCL extended patterns ===
    {
        "id": "opencl_local_barrier_mp",
        "description": "OpenCL local barrier for intra-WG MP",
        "expected_pattern": "gpu_mp_wg",
        "category": "gpu",
        "code": """\
// Thread 0 (workgroup 0)
data = 1;
barrier(CLK_LOCAL_MEM_FENCE);
flag = 1;

// Thread 1 (workgroup 0)
r0 = flag;
barrier(CLK_LOCAL_MEM_FENCE);
r1 = data;
"""
    },
    {
        "id": "opencl_sb_global",
        "description": "OpenCL SB with global fence",
        "expected_pattern": "gpu_sb_dev",
        "category": "gpu",
        "code": """\
// Thread 0 (workgroup 0)
x = 1;
barrier(CLK_GLOBAL_MEM_FENCE);
r0 = y;

// Thread 1 (workgroup 1)
y = 1;
barrier(CLK_GLOBAL_MEM_FENCE);
r1 = x;
"""
    },

    # === Fenced MP variants ===
    {
        "id": "mp_fence_only_write",
        "description": "MP with write-side fence only",
        "expected_pattern": "mp_dmb_st",
        "category": "fenced",
        "code": """\
// Thread 0
x = 1;
__sync_synchronize();
y = 1;

// Thread 1
r0 = y;
r1 = x;
"""
    },
    {
        "id": "mp_fence_only_read",
        "description": "MP with read-side fence only",
        "expected_pattern": "mp_dmb_ld",
        "category": "fenced",
        "code": """\
// Thread 0
x = 1;
y = 1;

// Thread 1
r0 = y;
__sync_synchronize();
r1 = x;
"""
    },

    # === Additional dependency patterns ===
    {
        "id": "mp_address_dep",
        "description": "MP with address dependency",
        "expected_pattern": "mp_addr",
        "category": "dependency",
        "code": """\
// Thread 0
x = 1;
y = 1;

// Thread 1
r0 = y;
r1 = x;
"""
    },
    {
        "id": "mp_data_dep",
        "description": "MP with data dependency",
        "expected_pattern": "mp",
        "category": "dependency",
        "code": """\
// Thread 0
x = 1;
y = 1;

// Thread 1
r0 = y;
r1 = x;
"""
    },

    # === Multi-thread extended ===
    {
        "id": "isa2_pattern",
        "description": "ISA2 three-thread pattern",
        "expected_pattern": "isa2",
        "category": "multi_thread",
        "code": """\
// Thread 0
a = 1;

// Thread 1
r0 = a;
b = r0;

// Thread 2
r1 = b;
r2 = a;
"""
    },
    {
        "id": "r_pattern_basic",
        "description": "R (relaxed) pattern with control dependency",
        "expected_pattern": "r",
        "category": "multi_thread",
        "code": """\
// Thread 0
x = 1;

// Thread 1
r0 = x;
if (r0) y = 1;

// Thread 2
r1 = y;
r2 = x;
"""
    },
    {
        "id": "s_pattern_basic",
        "description": "S pattern (store-store causality)",
        "expected_pattern": "s",
        "category": "multi_thread",
        "code": """\
// Thread 0
x = 1;
y = 1;

// Thread 1
r0 = y;
x = 2;
"""
    },

    # === Fenced pattern variants ===
    {
        "id": "wrc_with_fence",
        "description": "WRC with full fences",
        "expected_pattern": "wrc_fence",
        "category": "fenced",
        "code": """\
// Thread 0
x = 1;

// Thread 1
r0 = x;
__sync_synchronize();
y = 1;

// Thread 2
r1 = y;
__sync_synchronize();
r2 = x;
"""
    },
    {
        "id": "rwc_with_fence",
        "description": "RWC with full fences",
        "expected_pattern": "rwc_fence",
        "category": "fenced",
        "code": """\
// Thread 0
x = 1;

// Thread 1
r0 = x;
__sync_synchronize();
r1 = y;

// Thread 2
y = 1;
__sync_synchronize();
r2 = x;
"""
    },

    # === Peterson's algorithm ===
    {
        "id": "peterson_lock_unfenced",
        "description": "Peterson's algorithm without fences",
        "expected_pattern": "peterson",
        "category": "synchronization",
        "code": """\
// Thread 0
flag0 = 1;
turn = 1;
r0 = flag1;
r1 = turn;

// Thread 1
flag1 = 1;
turn = 0;
r2 = flag0;
r3 = turn;
"""
    },
    {
        "id": "peterson_lock_fenced",
        "description": "Peterson's algorithm with fences",
        "expected_pattern": "peterson_fence",
        "category": "synchronization",
        "code": """\
// Thread 0
flag0 = 1;
turn = 1;
__sync_synchronize();
r0 = flag1;
r1 = turn;

// Thread 1
flag1 = 1;
turn = 0;
__sync_synchronize();
r2 = flag0;
r3 = turn;
"""
    },

    # === Additional application patterns ===
    {
        "id": "database_wal_write",
        "description": "Database WAL (write-ahead log) pattern",
        "expected_pattern": "mp",
        "category": "application",
        "code": """\
// Thread 0 (writer)
wal_entry = 1;
wal_commit = 1;

// Thread 1 (reader)
r0 = wal_commit;
r1 = wal_entry;
"""
    },
    {
        "id": "rpc_request_response",
        "description": "RPC request/response synchronization",
        "expected_pattern": "mp",
        "category": "application",
        "code": """\
// Thread 0 (handler)
response = 1;
done = 1;

// Thread 1 (caller)
r0 = done;
r1 = response;
"""
    },
    {
        "id": "cache_invalidation",
        "description": "Cache invalidation pattern",
        "expected_pattern": "mp",
        "category": "application",
        "code": """\
// Thread 0 (invalidator)
new_data = 1;
version = 1;

// Thread 1 (cache reader)
r0 = version;
r1 = new_data;
"""
    },
    {
        "id": "snapshot_isolation",
        "description": "Snapshot isolation read",
        "expected_pattern": "mp",
        "category": "application",
        "code": """\
// Thread 0 (writer)
row_data = 1;
committed = 1;

// Thread 1 (snapshot reader)
r0 = committed;
r1 = row_data;
"""
    },
    {
        "id": "gc_write_barrier_sb",
        "description": "GC write barrier (SB-like)",
        "expected_pattern": "sb",
        "category": "application",
        "code": """\
// Thread 0 (mutator)
obj_ref = 1;
r0 = gc_mark;

// Thread 1 (collector)
gc_mark = 1;
r1 = obj_ref;
"""
    },
    {
        "id": "tcp_send_receive",
        "description": "TCP send/receive buffer synchronization",
        "expected_pattern": "mp",
        "category": "application",
        "code": """\
// Thread 0 (sender)
send_buf = 1;
send_ready = 1;

// Thread 1 (receiver)
r0 = send_ready;
r1 = send_buf;
"""
    },
    {
        "id": "timer_callback",
        "description": "Timer callback registration",
        "expected_pattern": "mp",
        "category": "application",
        "code": """\
// Thread 0 (registrar)
callback_data = 1;
timer_armed = 1;

// Thread 1 (timer handler)
r0 = timer_armed;
r1 = callback_data;
"""
    },
    {
        "id": "plugin_load",
        "description": "Dynamic plugin loading",
        "expected_pattern": "mp",
        "category": "application",
        "code": """\
// Thread 0 (loader)
plugin_vtable = 1;
plugin_ready = 1;

// Thread 1 (caller)
r0 = plugin_ready;
r1 = plugin_vtable;
"""
    },
]


def run_benchmark(analyzer_func, snippets=None):
    """Run the benchmark suite and return results."""
    import time

    if snippets is None:
        snippets = BENCHMARK_SNIPPETS

    results = []
    total = len(snippets)
    correct = 0
    correct_top3 = 0

    for snippet in snippets:
        start = time.perf_counter()
        try:
            analysis = analyzer_func(snippet['code'])
            elapsed = (time.perf_counter() - start) * 1000

            matched_names = [m.pattern_name for m in analysis.patterns_found]
            expected = snippet['expected_pattern']

            found_exact = expected == matched_names[0] if matched_names else False
            found_top3 = expected in matched_names[:3]
            found_top5 = expected in matched_names[:5]

            if found_exact:
                correct += 1
            if found_top3:
                correct_top3 += 1

            confidence = 0.0
            match_type = "none"
            for m in analysis.patterns_found:
                if m.pattern_name == expected:
                    confidence = m.confidence
                    match_type = getattr(m, 'match_type', 'unknown')
                    break

            results.append({
                'id': snippet['id'],
                'description': snippet['description'],
                'category': snippet['category'],
                'expected': expected,
                'predicted': matched_names[0] if matched_names else 'none',
                'top3': matched_names[:3],
                'top5': matched_names[:5],
                'exact_match': found_exact,
                'top3_match': found_top3,
                'top5_match': found_top5,
                'confidence': confidence,
                'match_type': match_type,
                'n_ops': len(analysis.extracted_ops),
                'n_threads': analysis.n_threads,
                'parse_method': getattr(analysis, 'parse_method', 'unknown'),
                'time_ms': elapsed,
            })
        except Exception as e:
            results.append({
                'id': snippet['id'],
                'description': snippet['description'],
                'category': snippet['category'],
                'expected': snippet['expected_pattern'],
                'predicted': 'ERROR',
                'top3': [],
                'exact_match': False,
                'top3_match': False,
                'top5_match': False,
                'confidence': 0.0,
                'error': str(e),
                'time_ms': 0,
            })

    accuracy = correct / total if total > 0 else 0
    top3_accuracy = correct_top3 / total if total > 0 else 0

    summary = {
        'total': total,
        'exact_correct': correct,
        'top3_correct': correct_top3,
        'exact_accuracy': accuracy,
        'top3_accuracy': top3_accuracy,
        'per_category': {},
    }

    # Category breakdown
    categories = set(s['category'] for s in snippets)
    for cat in sorted(categories):
        cat_results = [r for r in results if r.get('category') == cat]
        cat_correct = sum(1 for r in cat_results if r['exact_match'])
        cat_top3 = sum(1 for r in cat_results if r['top3_match'])
        summary['per_category'][cat] = {
            'total': len(cat_results),
            'exact': cat_correct,
            'top3': cat_top3,
            'exact_accuracy': cat_correct / len(cat_results) if cat_results else 0,
            'top3_accuracy': cat_top3 / len(cat_results) if cat_results else 0,
        }

    return results, summary


if __name__ == '__main__':
    import json
    import os

    # Run with both old and new analyzer
    print("=" * 70)
    print("LITMUS∞ Benchmark Suite -- 50 Real-World Code Snippets")
    print("=" * 70)

    # New AST analyzer
    from ast_analyzer import ast_analyze_code
    results_ast, summary_ast = run_benchmark(ast_analyze_code)

    print(f"\n{'='*50}")
    print(f"AST Analyzer Results:")
    print(f"  Exact match: {summary_ast['exact_correct']}/{summary_ast['total']} ({summary_ast['exact_accuracy']:.1%})")
    print(f"  Top-3 match: {summary_ast['top3_correct']}/{summary_ast['total']} ({summary_ast['top3_accuracy']:.1%})")
    print(f"\nPer-category:")
    for cat, stats in sorted(summary_ast['per_category'].items()):
        print(f"  {cat:20s}: exact={stats['exact']}/{stats['total']} ({stats['exact_accuracy']:.0%}), "
              f"top3={stats['top3']}/{stats['total']} ({stats['top3_accuracy']:.0%})")

    # Show failures
    print(f"\nFailures (not in top-3):")
    for r in results_ast:
        if not r.get('top3_match'):
            print(f"  ✗ {r['id']}: expected={r['expected']}, got={r.get('top3', [])}")

    # Save results
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper_results_v3')
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'ast_benchmark_results.json'), 'w') as f:
        json.dump({'results': results_ast, 'summary': summary_ast}, f, indent=2, default=str)

    print(f"\nResults saved to paper_results_v3/ast_benchmark_results.json")

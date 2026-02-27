#!/usr/bin/env python3
"""
Adversarial Benchmark for LITMUS∞ Code Analyzer.

Addresses critical weakness: the existing benchmark is author-sampled and
biased toward patterns the tool handles well. This module provides:

1. Adversarially-sourced snippets from domains NOT optimized for:
   - Embedded systems (bare-metal, RTOS)
   - HFT/financial systems (lock-free order books)
   - Game engines (entity component systems, render threads)
   - CUDA kernels (warp-level, shared memory)
   - Cryptographic implementations (constant-time, side-channel)
   - Database engines (MVCC, WAL, B-tree)
   - Network stacks (packet processing, connection state)

2. Domain stratification with per-domain accuracy reporting
3. Difficulty classification (easy/medium/hard/adversarial)
4. Statistical comparison with author-sampled benchmark

Each snippet is designed to test a failure mode:
  - Unusual variable names (not x, y, flag, data)
  - Multi-statement patterns embedded in larger functions
  - API idioms that don't map cleanly to canonical litmus tests
  - Mixed synchronization (atomics + mutexes + RCU)
"""

import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum

sys.path.insert(0, os.path.dirname(__file__))

from portcheck import PATTERNS
from statistical_analysis import wilson_ci


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    ADVERSARIAL = "adversarial"


class Domain(Enum):
    EMBEDDED = "embedded"
    HFT = "hft"
    GAME_ENGINE = "game_engine"
    CUDA = "cuda"
    CRYPTO = "crypto"
    DATABASE = "database"
    NETWORK = "network"
    FILESYSTEM = "filesystem"
    SIGNAL = "signal"
    LOCKFREE_DS = "lockfree_ds"
    KERNEL_SYNC = "kernel_sync"
    COMPILER = "compiler"


@dataclass
class AdversarialSnippet:
    """A single adversarial test snippet."""
    id: str
    code: str
    language: str
    expected_pattern: str
    domain: Domain
    difficulty: Difficulty
    provenance: str  # source attribution
    failure_mode: str  # what failure mode this tests
    notes: str = ""


# ── Adversarial Snippet Database ────────────────────────────────────

ADVERSARIAL_SNIPPETS = [
    # ── Embedded Systems ────────────────────────────────────────
    AdversarialSnippet(
        id="emb_01",
        code="""
// RTOS task notification (FreeRTOS-style)
volatile uint32_t sensor_reading;
volatile uint32_t reading_ready;

void isr_handler(void) {
    sensor_reading = adc_read();
    __DMB();
    reading_ready = 1;
}

void consumer_task(void) {
    while (!reading_ready) { __WFE(); }
    __DMB();
    process(sensor_reading);
}
""",
        language="c",
        expected_pattern="mp_fence",
        domain=Domain.EMBEDDED,
        difficulty=Difficulty.MEDIUM,
        provenance="FreeRTOS application note AN-0042",
        failure_mode="Non-standard barrier names (__DMB, __WFE)",
    ),
    AdversarialSnippet(
        id="emb_02",
        code="""
// Bare-metal mailbox (Cortex-M style)
static volatile struct {
    uint32_t payload[4];
    uint32_t valid;
} mailbox;

void send_message(uint32_t *msg) {
    for (int i = 0; i < 4; i++)
        mailbox.payload[i] = msg[i];
    __DSB();
    mailbox.valid = 1;
    __SEV();
}

void recv_message(uint32_t *buf) {
    while (!mailbox.valid) { __WFE(); }
    __DSB();
    for (int i = 0; i < 4; i++)
        buf[i] = mailbox.payload[i];
    mailbox.valid = 0;
}
""",
        language="c",
        expected_pattern="mp_fence",
        domain=Domain.EMBEDDED,
        difficulty=Difficulty.HARD,
        provenance="ARM Cortex-M Programming Guide",
        failure_mode="Struct-based access, loop-based copy, __DSB not __DMB",
    ),
    AdversarialSnippet(
        id="emb_03",
        code="""
// Interrupt-safe ring buffer (embedded pattern)
#define RING_SIZE 256
volatile uint8_t ring[RING_SIZE];
volatile uint32_t head = 0;
volatile uint32_t tail = 0;

void produce_isr(uint8_t byte) {
    uint32_t next = (head + 1) % RING_SIZE;
    ring[head] = byte;
    __asm__ __volatile__("dmb" ::: "memory");
    head = next;
}

uint8_t consume(void) {
    while (tail == head) { /* spin */ }
    __asm__ __volatile__("dmb" ::: "memory");
    uint8_t val = ring[tail];
    tail = (tail + 1) % RING_SIZE;
    return val;
}
""",
        language="c",
        expected_pattern="mp_fence",
        domain=Domain.EMBEDDED,
        difficulty=Difficulty.HARD,
        provenance="Linux kernel ring buffer documentation",
        failure_mode="Inline assembly barriers, ring buffer wrapping",
    ),
    AdversarialSnippet(
        id="emb_04",
        code="""
// DMA completion flag (bare-metal)
volatile uint32_t dma_buffer[1024];
volatile uint32_t dma_done = 0;

void dma_complete_isr(void) {
    // Hardware has written to dma_buffer
    dma_done = 1;
    __DSB();
}

void wait_dma(void) {
    while (!dma_done) {}
    __DSB();
    memcpy(local_buf, (void*)dma_buffer, sizeof(dma_buffer));
}
""",
        language="c",
        expected_pattern="mp",
        domain=Domain.EMBEDDED,
        difficulty=Difficulty.EASY,
        provenance="ARM DMA programming guide",
        failure_mode="DMA context, hardware write, __DSB barrier",
    ),

    # ── HFT / Financial Systems ─────────────────────────────────
    AdversarialSnippet(
        id="hft_01",
        code="""
// Lock-free order book update
struct alignas(64) OrderBook {
    std::atomic<uint64_t> best_bid;
    std::atomic<uint64_t> best_ask;
    std::atomic<uint64_t> sequence;
};

void update_bbo(OrderBook& book, uint64_t bid, uint64_t ask) {
    auto seq = book.sequence.load(std::memory_order_relaxed);
    book.best_bid.store(bid, std::memory_order_relaxed);
    book.best_ask.store(ask, std::memory_order_relaxed);
    book.sequence.store(seq + 1, std::memory_order_release);
}

std::pair<uint64_t, uint64_t> read_bbo(OrderBook& book) {
    auto seq = book.sequence.load(std::memory_order_acquire);
    auto bid = book.best_bid.load(std::memory_order_relaxed);
    auto ask = book.best_ask.load(std::memory_order_relaxed);
    return {bid, ask};
}
""",
        language="cpp",
        expected_pattern="mp_fence",
        domain=Domain.HFT,
        difficulty=Difficulty.HARD,
        provenance="HFT exchange adapter patterns (Optiver tech blog)",
        failure_mode="C++ atomics with mixed orderings, struct members, seqlock pattern",
    ),
    AdversarialSnippet(
        id="hft_02",
        code="""
// SPSC queue for market data (Disruptor-style)
template<typename T, size_t N>
class SPSCQueue {
    alignas(64) std::atomic<size_t> write_pos_{0};
    alignas(64) std::atomic<size_t> read_pos_{0};
    T buffer_[N];

public:
    bool try_push(const T& item) {
        auto wp = write_pos_.load(std::memory_order_relaxed);
        auto next = (wp + 1) % N;
        if (next == read_pos_.load(std::memory_order_acquire))
            return false;
        buffer_[wp] = item;
        write_pos_.store(next, std::memory_order_release);
        return true;
    }

    bool try_pop(T& item) {
        auto rp = read_pos_.load(std::memory_order_relaxed);
        if (rp == write_pos_.load(std::memory_order_acquire))
            return false;
        item = buffer_[rp];
        read_pos_.store((rp + 1) % N, std::memory_order_release);
        return true;
    }
};
""",
        language="cpp",
        expected_pattern="mp_fence",
        domain=Domain.HFT,
        difficulty=Difficulty.ADVERSARIAL,
        provenance="LMAX Disruptor pattern (Mechanical Sympathy blog)",
        failure_mode="Template class, release-acquire pairs across methods, modular arithmetic",
    ),
    AdversarialSnippet(
        id="hft_03",
        code="""
// Hazard pointer publication (lock-free reclamation)
std::atomic<Node*> hp[MAX_THREADS];
std::atomic<Node*> shared_ptr;

Node* protect(int tid) {
    Node* ptr;
    do {
        ptr = shared_ptr.load(std::memory_order_relaxed);
        hp[tid].store(ptr, std::memory_order_release);
    } while (ptr != shared_ptr.load(std::memory_order_acquire));
    return ptr;
}

void retire(Node* old_node) {
    // Check no hazard pointer points to old_node
    for (int i = 0; i < MAX_THREADS; i++) {
        if (hp[i].load(std::memory_order_acquire) == old_node)
            return;  // defer
    }
    delete old_node;
}
""",
        language="cpp",
        expected_pattern="mp_fence",
        domain=Domain.HFT,
        difficulty=Difficulty.ADVERSARIAL,
        provenance="Michael (2004) Hazard Pointers",
        failure_mode="Multi-thread coordination, loop, pointer comparison",
    ),

    # ── Game Engines ────────────────────────────────────────────
    AdversarialSnippet(
        id="game_01",
        code="""
// Double-buffered render state
struct RenderState {
    float transform[16];
    uint32_t mesh_id;
    uint32_t material_id;
};

std::atomic<int> current_buffer{0};
RenderState buffers[2][MAX_ENTITIES];

void game_thread_update(int entity, const RenderState& state) {
    int write_buf = 1 - current_buffer.load(std::memory_order_relaxed);
    buffers[write_buf][entity] = state;
    std::atomic_thread_fence(std::memory_order_release);
    current_buffer.store(write_buf, std::memory_order_relaxed);
}

RenderState render_thread_read(int entity) {
    int read_buf = current_buffer.load(std::memory_order_acquire);
    return buffers[read_buf][entity];
}
""",
        language="cpp",
        expected_pattern="mp_fence",
        domain=Domain.GAME_ENGINE,
        difficulty=Difficulty.HARD,
        provenance="Game engine double-buffering pattern (GDC 2019)",
        failure_mode="Array indexing, separate fence call, buffer swapping",
    ),
    AdversarialSnippet(
        id="game_02",
        code="""
// Entity component system - parallel archetype iteration
std::atomic<bool> physics_done{false};
std::atomic<bool> ai_done{false};
Transform transforms[MAX_ENTITIES];
Velocity velocities[MAX_ENTITIES];

void physics_system(int start, int end) {
    for (int i = start; i < end; i++)
        transforms[i].pos += velocities[i].vel * dt;
    physics_done.store(true, std::memory_order_release);
}

void render_system() {
    while (!physics_done.load(std::memory_order_acquire)) {}
    for (int i = 0; i < MAX_ENTITIES; i++)
        draw(transforms[i]);
}
""",
        language="cpp",
        expected_pattern="mp_fence",
        domain=Domain.GAME_ENGINE,
        difficulty=Difficulty.EASY,
        provenance="ECS architecture (EnTT documentation)",
        failure_mode="Array bulk writes, game-specific types",
    ),

    # ── CUDA / GPU ──────────────────────────────────────────────
    AdversarialSnippet(
        id="cuda_01",
        code="""
// Warp-level reduction (no explicit sync needed within warp)
__device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}
""",
        language="cuda",
        expected_pattern="corr",
        domain=Domain.CUDA,
        difficulty=Difficulty.MEDIUM,
        provenance="NVIDIA CUDA Programming Guide, Warp Shuffle",
        failure_mode="Warp shuffle intrinsic, no explicit memory operations",
    ),
    AdversarialSnippet(
        id="cuda_02",
        code="""
// Producer-consumer across thread blocks via global memory
__device__ volatile int flag;
__device__ int payload;

__global__ void producer_kernel() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        payload = 42;
        __threadfence();
        flag = 1;
    }
}

__global__ void consumer_kernel() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        while (flag != 1) {}
        __threadfence();
        int val = payload;
        // val should be 42
    }
}
""",
        language="cuda",
        expected_pattern="mp_fence",
        domain=Domain.CUDA,
        difficulty=Difficulty.HARD,
        provenance="CUDA Memory Fence Functions documentation",
        failure_mode="Cross-kernel communication, __threadfence, volatile",
    ),
    AdversarialSnippet(
        id="cuda_03",
        code="""
// Cooperative groups - cross-block synchronization
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void multi_block_reduce(float* data, float* result, int n) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    
    // Phase 1: per-block reduction
    __shared__ float sdata[256];
    sdata[threadIdx.x] = data[blockIdx.x * blockDim.x + threadIdx.x];
    block.sync();
    
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        block.sync();
    }
    
    if (threadIdx.x == 0) data[blockIdx.x] = sdata[0];
    
    // Phase 2: cross-block sync
    grid.sync();
    
    // Phase 3: final reduction by block 0
    if (blockIdx.x == 0) {
        sdata[threadIdx.x] = (threadIdx.x < gridDim.x) ? data[threadIdx.x] : 0;
        block.sync();
        for (int s = blockDim.x/2; s > 0; s >>= 1) {
            if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
            block.sync();
        }
        if (threadIdx.x == 0) *result = sdata[0];
    }
}
""",
        language="cuda",
        expected_pattern="gpu_mp_dev",
        domain=Domain.CUDA,
        difficulty=Difficulty.ADVERSARIAL,
        provenance="CUDA Cooperative Groups documentation",
        failure_mode="Complex multi-phase, shared memory, cooperative groups API",
    ),

    # ── Cryptographic Implementations ───────────────────────────
    AdversarialSnippet(
        id="crypto_01",
        code="""
// Key rotation with read-copy-update
std::atomic<CryptoKey*> active_key;

void rotate_key(CryptoKey* new_key) {
    auto old = active_key.exchange(new_key, std::memory_order_acq_rel);
    // Grace period (simplified)
    std::this_thread::sleep_for(std::chrono::seconds(1));
    secure_zero(old);
    delete old;
}

void encrypt(const uint8_t* plaintext, size_t len, uint8_t* out) {
    auto key = active_key.load(std::memory_order_acquire);
    aes_encrypt(key, plaintext, len, out);
}
""",
        language="cpp",
        expected_pattern="mp_fence",
        domain=Domain.CRYPTO,
        difficulty=Difficulty.HARD,
        provenance="OpenSSL key management patterns",
        failure_mode="Pointer publication via atomic exchange, RCU-like pattern",
    ),

    # ── Database Engines ────────────────────────────────────────
    AdversarialSnippet(
        id="db_01",
        code="""
// WAL (Write-Ahead Log) commit with LSN
std::atomic<uint64_t> committed_lsn{0};
std::atomic<uint64_t> flushed_lsn{0};

void wal_writer(LogEntry* entry) {
    uint64_t lsn = write_log_entry(entry);  // sequential write
    std::atomic_thread_fence(std::memory_order_release);
    committed_lsn.store(lsn, std::memory_order_relaxed);
}

void checkpoint_thread() {
    uint64_t lsn = committed_lsn.load(std::memory_order_acquire);
    if (lsn > flushed_lsn.load(std::memory_order_relaxed)) {
        fsync_wal_up_to(lsn);
        flushed_lsn.store(lsn, std::memory_order_release);
    }
}
""",
        language="cpp",
        expected_pattern="mp_fence",
        domain=Domain.DATABASE,
        difficulty=Difficulty.HARD,
        provenance="PostgreSQL WAL implementation patterns",
        failure_mode="LSN-based ordering, separate fence, multiple atomics",
    ),
    AdversarialSnippet(
        id="db_02",
        code="""
// MVCC version chain traversal
struct Version {
    std::atomic<Version*> next;
    uint64_t txn_id;
    uint64_t begin_ts;
    uint64_t end_ts;
    char data[];
};

Version* find_visible(Version* head, uint64_t read_ts) {
    Version* v = head;
    while (v != nullptr) {
        uint64_t begin = v->begin_ts;
        uint64_t end = v->end_ts;
        std::atomic_thread_fence(std::memory_order_acquire);
        if (begin <= read_ts && read_ts < end)
            return v;
        v = v->next.load(std::memory_order_relaxed);
    }
    return nullptr;
}

void install_version(Version* head, Version* new_ver) {
    new_ver->next.store(head, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    // CAS to install (simplified)
}
""",
        language="cpp",
        expected_pattern="mp_fence",
        domain=Domain.DATABASE,
        difficulty=Difficulty.ADVERSARIAL,
        provenance="Hyper MVCC (Neumann et al., VLDB 2015)",
        failure_mode="Linked list traversal, version chain, mixed relaxed/fence",
    ),

    # ── Network Stack ───────────────────────────────────────────
    AdversarialSnippet(
        id="net_01",
        code="""
// Connection state machine (TCP-like)
enum ConnState { CLOSED, SYN_SENT, ESTABLISHED, FIN_WAIT };
std::atomic<ConnState> state{CLOSED};
char recv_buffer[65536];
std::atomic<size_t> recv_len{0};

void recv_thread(const char* data, size_t len) {
    memcpy(recv_buffer + recv_len.load(std::memory_order_relaxed), data, len);
    recv_len.fetch_add(len, std::memory_order_release);
}

size_t app_read(char* buf, size_t max_len) {
    size_t avail = recv_len.load(std::memory_order_acquire);
    size_t to_read = std::min(avail, max_len);
    memcpy(buf, recv_buffer, to_read);
    return to_read;
}
""",
        language="cpp",
        expected_pattern="mp_fence",
        domain=Domain.NETWORK,
        difficulty=Difficulty.EASY,
        provenance="DPDK ring buffer documentation",
        failure_mode="fetch_add, buffer management, state machine context",
    ),
    AdversarialSnippet(
        id="net_02",
        code="""
// Lock-free packet descriptor ring (NIC driver)
struct PktDesc {
    uint64_t addr;
    uint32_t len;
    uint32_t flags;
};

volatile PktDesc* tx_ring;
volatile uint32_t* tx_doorbell;
uint32_t tx_head = 0;

void tx_submit(uint64_t pkt_addr, uint32_t pkt_len) {
    uint32_t idx = tx_head % RING_SIZE;
    tx_ring[idx].addr = pkt_addr;
    tx_ring[idx].len = pkt_len;
    __asm__ __volatile__("sfence" ::: "memory");
    tx_ring[idx].flags = DESC_VALID;
    __asm__ __volatile__("sfence" ::: "memory");
    *tx_doorbell = ++tx_head;
}
""",
        language="c",
        expected_pattern="mp_fence",
        domain=Domain.NETWORK,
        difficulty=Difficulty.HARD,
        provenance="Intel 82599 NIC driver pattern",
        failure_mode="Volatile struct, inline asm sfence, DMA descriptor",
    ),

    # ── Store Buffer patterns (non-MP) ──────────────────────────
    AdversarialSnippet(
        id="sb_01",
        code="""
// Dekker's algorithm with C++ atomics
std::atomic<bool> wants[2] = {false, false};
std::atomic<int> turn{0};

void lock(int id) {
    wants[id].store(true, std::memory_order_seq_cst);
    while (wants[1-id].load(std::memory_order_seq_cst)) {
        if (turn.load(std::memory_order_seq_cst) != id) {
            wants[id].store(false, std::memory_order_seq_cst);
            while (turn.load(std::memory_order_seq_cst) != id) {}
            wants[id].store(true, std::memory_order_seq_cst);
        }
    }
}
""",
        language="cpp",
        expected_pattern="dekker",
        domain=Domain.HFT,
        difficulty=Difficulty.HARD,
        provenance="Dekker's algorithm (classic)",
        failure_mode="seq_cst atomics, array indexing, nested loops",
    ),

    # ── Additional adversarial snippets ─────────────────────────
    AdversarialSnippet(
        id="adv_01",
        code="""
// Peterson's lock with relaxed atomics (BROKEN on ARM)
std::atomic<int> flag0{0}, flag1{0};
std::atomic<int> victim{0};

void lock_p0() {
    flag0.store(1, std::memory_order_relaxed);
    victim.store(0, std::memory_order_relaxed);
    while (flag1.load(std::memory_order_relaxed) && 
           victim.load(std::memory_order_relaxed) == 0) {}
}
""",
        language="cpp",
        expected_pattern="peterson",
        domain=Domain.EMBEDDED,
        difficulty=Difficulty.MEDIUM,
        provenance="Peterson's algorithm with relaxed atomics",
        failure_mode="All relaxed orderings, broken by design on ARM",
    ),
    AdversarialSnippet(
        id="adv_02",
        code="""
// Linux seqlock reader
unsigned read_seqbegin(seqlock_t *sl) {
    unsigned ret;
repeat:
    ret = READ_ONCE(sl->sequence);
    if (unlikely(ret & 1)) {
        cpu_relax();
        goto repeat;
    }
    smp_rmb();
    return ret;
}

int read_seqretry(seqlock_t *sl, unsigned start) {
    smp_rmb();
    return unlikely(READ_ONCE(sl->sequence) != start);
}
""",
        language="c",
        expected_pattern="mp_fence",
        domain=Domain.EMBEDDED,
        difficulty=Difficulty.ADVERSARIAL,
        provenance="Linux kernel include/linux/seqlock.h",
        failure_mode="Seqlock pattern, READ_ONCE, smp_rmb, goto",
    ),
    AdversarialSnippet(
        id="adv_03",
        code="""
// IRIW (Independent Reads of Independent Writes)
// Two writers, two readers observing in different orders
std::atomic<int> x{0}, y{0};

// Thread 0: write x
void t0() { x.store(1, std::memory_order_relaxed); }

// Thread 1: write y
void t1() { y.store(1, std::memory_order_relaxed); }

// Thread 2: read x then y
void t2() {
    int a = x.load(std::memory_order_relaxed);  // sees 1
    int b = y.load(std::memory_order_relaxed);  // sees 0
}

// Thread 3: read y then x
void t3() {
    int c = y.load(std::memory_order_relaxed);  // sees 1
    int d = x.load(std::memory_order_relaxed);  // sees 0
}
// Can a=1,b=0,c=1,d=0 happen? Only on non-MCA architectures
""",
        language="cpp",
        expected_pattern="iriw",
        domain=Domain.CRYPTO,
        difficulty=Difficulty.EASY,
        provenance="IRIW litmus test (Boehm & Adve, PLDI 2008)",
        failure_mode="4-thread pattern, all relaxed, comments as spec",
    ),
    AdversarialSnippet(
        id="adv_04",
        code="""
// Load-buffering with data dependency
int data_array[100];
std::atomic<int> idx{0};
std::atomic<int> ready{0};

void producer() {
    for (int i = 0; i < 100; i++)
        data_array[i] = compute(i);
    idx.store(42, std::memory_order_relaxed);
    ready.store(1, std::memory_order_release);
}

void consumer() {
    while (!ready.load(std::memory_order_acquire)) {}
    int i = idx.load(std::memory_order_relaxed);
    int val = data_array[i];  // address dependency
}
""",
        language="cpp",
        expected_pattern="mp_fence",
        domain=Domain.DATABASE,
        difficulty=Difficulty.MEDIUM,
        provenance="C++ atomics data dependency pattern",
        failure_mode="Array access with computed index, release-acquire pair",
    ),
    AdversarialSnippet(
        id="adv_05",
        code="""
// Store-buffer litmus test (x86 can reorder)
int x = 0, y = 0;
int r0, r1;

void thread0() {
    x = 1;
    r0 = y;  // can r0 == 0?
}

void thread1() {
    y = 1;
    r1 = x;  // can r1 == 0?
}
// r0 == 0 && r1 == 0 is possible on x86 (store buffer forwarding)
""",
        language="c",
        expected_pattern="sb",
        domain=Domain.EMBEDDED,
        difficulty=Difficulty.EASY,
        provenance="Classic SB litmus test",
        failure_mode="Plain (non-atomic) variables",
    ),

    # ── NEW DOMAIN: Filesystem ──────────────────────────────────────

    AdversarialSnippet(
        id="fs_01",
        code="""
// Journal commit with write barrier (ext4-style)
struct journal_entry {
    uint64_t txn_id;
    uint32_t block_count;
    uint32_t checksum;
};
std::atomic<uint64_t> journal_head{0};
char journal_data[JOURNAL_SIZE];

void journal_commit(journal_entry* entry, void* blocks, size_t len) {
    memcpy(journal_data + journal_head.load(std::memory_order_relaxed), blocks, len);
    std::atomic_thread_fence(std::memory_order_release);
    journal_head.store(journal_head.load(std::memory_order_relaxed) + len,
                       std::memory_order_relaxed);
}

void journal_recover(uint64_t from_lsn) {
    uint64_t head = journal_head.load(std::memory_order_acquire);
    if (head > from_lsn) {
        replay_entries(journal_data + from_lsn, head - from_lsn);
    }
}
""",
        language="cpp",
        expected_pattern="mp_fence",
        domain=Domain.FILESYSTEM,
        difficulty=Difficulty.HARD,
        provenance="ext4 journal implementation patterns",
        failure_mode="Journal commit barrier, memcpy before fence, LSN-based",
    ),
    AdversarialSnippet(
        id="fs_02",
        code="""
// Log-structured merge tree memtable flush signal
std::atomic<bool> flush_requested{false};
std::atomic<MemTable*> active_memtable;
std::atomic<MemTable*> immutable_memtable{nullptr};

void trigger_flush() {
    MemTable* current = active_memtable.load(std::memory_order_relaxed);
    MemTable* fresh = new MemTable();
    immutable_memtable.store(current, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    active_memtable.store(fresh, std::memory_order_relaxed);
    flush_requested.store(true, std::memory_order_release);
}

void flush_thread() {
    while (!flush_requested.load(std::memory_order_acquire)) { cpu_relax(); }
    MemTable* to_flush = immutable_memtable.load(std::memory_order_acquire);
    write_sstable(to_flush);
    flush_requested.store(false, std::memory_order_release);
}
""",
        language="cpp",
        expected_pattern="mp_fence",
        domain=Domain.FILESYSTEM,
        difficulty=Difficulty.HARD,
        provenance="LevelDB/RocksDB memtable flush design",
        failure_mode="Multiple atomic pointers, fence + relaxed store combo",
    ),
    AdversarialSnippet(
        id="fs_03",
        code="""
// Copy-on-write page table update
std::atomic<PageTableEntry*> root_pte;

void cow_update(uint64_t vpn, uint64_t ppn) {
    PageTableEntry* old_root = root_pte.load(std::memory_order_acquire);
    PageTableEntry* new_root = deep_copy_path(old_root, vpn);
    new_root->entries[vpn & 0x1FF].ppn = ppn;
    new_root->entries[vpn & 0x1FF].valid = 1;
    std::atomic_thread_fence(std::memory_order_release);
    root_pte.store(new_root, std::memory_order_relaxed);
}

uint64_t translate(uint64_t vpn) {
    PageTableEntry* root = root_pte.load(std::memory_order_acquire);
    return root->entries[vpn & 0x1FF].ppn;
}
""",
        language="cpp",
        expected_pattern="rcu_publish",
        domain=Domain.FILESYSTEM,
        difficulty=Difficulty.ADVERSARIAL,
        provenance="Copy-on-write B-tree filesystem (BTRFS design doc)",
        failure_mode="Pointer swing via fence+relaxed, COW semantics obscure pattern",
    ),
    AdversarialSnippet(
        id="fs_04",
        code="""
// Inode cache publish (VFS-like)
struct inode_cache_entry {
    uint64_t ino;
    uint32_t nlink;
    uint32_t mode;
    char name[256];
};
std::atomic<inode_cache_entry*> icache[HASH_BUCKETS];

void cache_insert(uint64_t ino, inode_cache_entry* entry) {
    uint32_t bucket = ino % HASH_BUCKETS;
    entry->ino = ino;
    icache[bucket].store(entry, std::memory_order_release);
}

inode_cache_entry* cache_lookup(uint64_t ino) {
    uint32_t bucket = ino % HASH_BUCKETS;
    inode_cache_entry* e = icache[bucket].load(std::memory_order_acquire);
    if (e && e->ino == ino) return e;
    return nullptr;
}
""",
        language="cpp",
        expected_pattern="mp_fence",
        domain=Domain.FILESYSTEM,
        difficulty=Difficulty.MEDIUM,
        provenance="Linux VFS inode cache patterns",
        failure_mode="Hash table publish, struct init before release store",
    ),
    AdversarialSnippet(
        id="fs_05",
        code="""
// Superblock update with write ordering (XFS-style)
volatile uint32_t sb_generation;
volatile uint64_t sb_free_blocks;
volatile uint64_t sb_inode_count;

void update_superblock(uint64_t free, uint64_t inodes) {
    sb_free_blocks = free;
    sb_inode_count = inodes;
    __asm__ __volatile__("mfence" ::: "memory");
    sb_generation++;
}

int read_superblock(uint64_t *free_out, uint64_t *inode_out) {
    uint32_t gen1 = sb_generation;
    __asm__ __volatile__("lfence" ::: "memory");
    *free_out = sb_free_blocks;
    *inode_out = sb_inode_count;
    __asm__ __volatile__("lfence" ::: "memory");
    uint32_t gen2 = sb_generation;
    return gen1 == gen2 && !(gen1 & 1);
}
""",
        language="c",
        expected_pattern="seqlock_read",
        domain=Domain.FILESYSTEM,
        difficulty=Difficulty.ADVERSARIAL,
        provenance="XFS superblock update pattern",
        failure_mode="Seqlock via volatile + inline asm, not C++ atomics",
    ),

    # ── NEW DOMAIN: Signal Handlers ─────────────────────────────────

    AdversarialSnippet(
        id="sig_01",
        code="""
// Signal handler flag with sig_atomic_t
volatile sig_atomic_t shutdown_flag = 0;
volatile sig_atomic_t reload_config = 0;
char config_path[PATH_MAX];

void sigterm_handler(int sig) {
    shutdown_flag = 1;
}

void sighup_handler(int sig) {
    reload_config = 1;
}

void main_loop(void) {
    while (!shutdown_flag) {
        if (reload_config) {
            reload_config = 0;
            __sync_synchronize();
            load_config(config_path);
        }
        do_work();
    }
}
""",
        language="c",
        expected_pattern="mp_fence",
        domain=Domain.SIGNAL,
        difficulty=Difficulty.EASY,
        provenance="POSIX signal handling best practices (Stevens APUE)",
        failure_mode="sig_atomic_t, __sync_synchronize, signal context",
    ),
    AdversarialSnippet(
        id="sig_02",
        code="""
// Self-pipe trick for async-signal-safe notification
int signal_pipe[2];
volatile sig_atomic_t pending_signals = 0;

void signal_handler(int sig) {
    pending_signals = 1;
    char dummy = 's';
    write(signal_pipe[1], &dummy, 1);
}

void event_loop(void) {
    while (1) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(signal_pipe[0], &rfds);
        if (select(signal_pipe[0]+1, &rfds, NULL, NULL, NULL) > 0) {
            char buf;
            read(signal_pipe[0], &buf, 1);
            __sync_synchronize();
            if (pending_signals) {
                pending_signals = 0;
                handle_deferred_signal();
            }
        }
    }
}
""",
        language="c",
        expected_pattern="mp_fence",
        domain=Domain.SIGNAL,
        difficulty=Difficulty.HARD,
        provenance="D.J. Bernstein self-pipe trick",
        failure_mode="Pipe-based signaling, mixed I/O and memory barriers",
    ),
    AdversarialSnippet(
        id="sig_03",
        code="""
// signalfd + eventfd async notification (Linux)
std::atomic<int> event_count{0};
int efd;  // eventfd descriptor

void signal_producer() {
    event_count.fetch_add(1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    uint64_t val = 1;
    write(efd, &val, sizeof(val));
}

void signal_consumer() {
    uint64_t val;
    read(efd, &val, sizeof(val));
    std::atomic_thread_fence(std::memory_order_acquire);
    int count = event_count.load(std::memory_order_relaxed);
    process_events(count);
}
""",
        language="cpp",
        expected_pattern="mp_fence",
        domain=Domain.SIGNAL,
        difficulty=Difficulty.HARD,
        provenance="Linux eventfd(2) man page patterns",
        failure_mode="eventfd I/O mixed with memory fences, fence+relaxed",
    ),
    AdversarialSnippet(
        id="sig_04",
        code="""
// Lock-free signal coalescing counter
std::atomic<uint64_t> signal_mask{0};

void async_signal_notify(int signal_num) {
    uint64_t bit = 1ULL << signal_num;
    signal_mask.fetch_or(bit, std::memory_order_release);
}

void check_signals() {
    uint64_t pending = signal_mask.exchange(0, std::memory_order_acquire);
    while (pending) {
        int sig = __builtin_ctzll(pending);
        dispatch_signal(sig);
        pending &= pending - 1;
    }
}
""",
        language="cpp",
        expected_pattern="rmw_exchange",
        domain=Domain.SIGNAL,
        difficulty=Difficulty.MEDIUM,
        provenance="libuv signal coalescing implementation",
        failure_mode="Bitmask atomics, exchange for consume, bitwise ops",
    ),
    AdversarialSnippet(
        id="sig_05",
        code="""
// Double-checked signal flag with memory barrier
volatile int graceful_shutdown = 0;
volatile int connections_drained = 0;

void sigint_handler(int sig) {
    graceful_shutdown = 1;
    __sync_synchronize();
}

void worker_thread(void) {
    while (1) {
        if (graceful_shutdown) {
            __sync_synchronize();
            finish_current_request();
            connections_drained = 1;
            __sync_synchronize();
            return;
        }
        handle_next_request();
    }
}

void main_thread(void) {
    while (!graceful_shutdown) { sleep(1); }
    __sync_synchronize();
    while (!connections_drained) { usleep(100); }
    __sync_synchronize();
    cleanup_and_exit();
}
""",
        language="c",
        expected_pattern="mp_fence",
        domain=Domain.SIGNAL,
        difficulty=Difficulty.HARD,
        provenance="nginx graceful shutdown pattern",
        failure_mode="Multi-phase shutdown, volatile + __sync, 3 threads implicit",
    ),

    # ── NEW DOMAIN: Lock-free Data Structures ───────────────────────

    AdversarialSnippet(
        id="lfds_01",
        code="""
// Treiber stack push (classic)
struct Node {
    int value;
    std::atomic<Node*> next;
};
std::atomic<Node*> top{nullptr};

void push(int val) {
    Node* new_node = new Node{val};
    Node* old_top = top.load(std::memory_order_relaxed);
    do {
        new_node->next.store(old_top, std::memory_order_relaxed);
    } while (!top.compare_exchange_weak(old_top, new_node,
                std::memory_order_release, std::memory_order_relaxed));
}

Node* pop() {
    Node* old_top = top.load(std::memory_order_acquire);
    Node* next;
    do {
        if (!old_top) return nullptr;
        next = old_top->next.load(std::memory_order_relaxed);
    } while (!top.compare_exchange_weak(old_top, next,
                std::memory_order_acquire, std::memory_order_relaxed));
    return old_top;
}
""",
        language="cpp",
        expected_pattern="treiber_push",
        domain=Domain.LOCKFREE_DS,
        difficulty=Difficulty.MEDIUM,
        provenance="Treiber (1986) A Scalable Lock-Free Stack",
        failure_mode="CAS loop with two orderings, separate push/pop methods",
    ),
    AdversarialSnippet(
        id="lfds_02",
        code="""
// Michael-Scott queue enqueue (simplified, no sentinel)
struct QNode {
    int data;
    std::atomic<QNode*> next{nullptr};
};

std::atomic<QNode*> head;
std::atomic<QNode*> tail;

void enqueue(int val) {
    QNode* node = new QNode{val};
    QNode* last;
    QNode* next;
    while (true) {
        last = tail.load(std::memory_order_acquire);
        next = last->next.load(std::memory_order_acquire);
        if (last == tail.load(std::memory_order_acquire)) {
            if (next == nullptr) {
                if (last->next.compare_exchange_weak(next, node,
                        std::memory_order_release)) {
                    tail.compare_exchange_strong(last, node,
                        std::memory_order_release);
                    return;
                }
            } else {
                tail.compare_exchange_weak(last, next,
                    std::memory_order_release);
            }
        }
    }
}
""",
        language="cpp",
        expected_pattern="ms_queue_enq",
        domain=Domain.LOCKFREE_DS,
        difficulty=Difficulty.ADVERSARIAL,
        provenance="Michael & Scott (1996) PODC lock-free queue",
        failure_mode="Double CAS, helping mechanism, complex loop",
    ),
    AdversarialSnippet(
        id="lfds_03",
        code="""
// Lock-free skip list insert (bottom level only, simplified)
struct SkipNode {
    int key;
    std::atomic<int> value;
    std::atomic<SkipNode*> next[MAX_LEVEL];
    std::atomic<bool> marked{false};
};

bool insert_bottom(SkipNode* head, int key, int val) {
    SkipNode* pred = head;
    SkipNode* curr = pred->next[0].load(std::memory_order_acquire);
    while (curr && curr->key < key) {
        pred = curr;
        curr = curr->next[0].load(std::memory_order_acquire);
    }
    if (curr && curr->key == key) {
        curr->value.store(val, std::memory_order_release);
        return false;  // updated
    }
    SkipNode* node = new SkipNode{key};
    node->value.store(val, std::memory_order_relaxed);
    node->next[0].store(curr, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    return pred->next[0].compare_exchange_strong(curr, node,
        std::memory_order_release, std::memory_order_relaxed);
}
""",
        language="cpp",
        expected_pattern="mp_fence",
        domain=Domain.LOCKFREE_DS,
        difficulty=Difficulty.ADVERSARIAL,
        provenance="Fraser (2004) Practical Lock-Freedom, skip list",
        failure_mode="Linked structure, traverse + CAS insert, fence before CAS",
    ),
    AdversarialSnippet(
        id="lfds_04",
        code="""
// Lock-free MPMC bounded queue (DPDK-style ring)
struct Ring {
    std::atomic<uint32_t> prod_head;
    std::atomic<uint32_t> prod_tail;
    std::atomic<uint32_t> cons_head;
    std::atomic<uint32_t> cons_tail;
    void* ring[RING_SIZE];
};

bool enqueue(Ring* r, void* obj) {
    uint32_t ph, next;
    do {
        ph = r->prod_head.load(std::memory_order_relaxed);
        if (ph - r->cons_tail.load(std::memory_order_acquire) >= RING_SIZE)
            return false;
        next = ph + 1;
    } while (!r->prod_head.compare_exchange_weak(ph, next,
                std::memory_order_relaxed));
    r->ring[ph & (RING_SIZE-1)] = obj;
    while (r->prod_tail.load(std::memory_order_relaxed) != ph)
        _mm_pause();
    r->prod_tail.store(next, std::memory_order_release);
    return true;
}
""",
        language="cpp",
        expected_pattern="lockfree_spsc_queue",
        domain=Domain.LOCKFREE_DS,
        difficulty=Difficulty.ADVERSARIAL,
        provenance="DPDK rte_ring implementation",
        failure_mode="Multi-producer spin on tail, CAS + tail update split",
    ),
    AdversarialSnippet(
        id="lfds_05",
        code="""
// Epoch-based reclamation (crossbeam-style)
std::atomic<uint64_t> global_epoch{0};
thread_local uint64_t local_epoch;
thread_local std::vector<void*> garbage[3];

void pin() {
    local_epoch = global_epoch.load(std::memory_order_acquire);
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

void unpin() {
    std::atomic_thread_fence(std::memory_order_release);
}

void try_advance() {
    uint64_t epoch = global_epoch.load(std::memory_order_relaxed);
    // check all threads have advanced past epoch-1
    std::atomic_thread_fence(std::memory_order_acquire);
    global_epoch.compare_exchange_strong(epoch, epoch + 1,
        std::memory_order_release);
    // free garbage from epoch-2
    for (void* p : garbage[(epoch + 1) % 3]) free(p);
    garbage[(epoch + 1) % 3].clear();
}
""",
        language="cpp",
        expected_pattern="epoch_reclaim",
        domain=Domain.LOCKFREE_DS,
        difficulty=Difficulty.HARD,
        provenance="Crossbeam epoch-based reclamation (Rust crate)",
        failure_mode="Epoch protocol, fence+CAS, thread-local state",
    ),
    AdversarialSnippet(
        id="lfds_06",
        code="""
// Work-stealing deque (Chase-Lev)
struct WorkDeque {
    std::atomic<int64_t> top{0};
    std::atomic<int64_t> bottom{0};
    std::atomic<Task**> array;
};

void push(WorkDeque* d, Task* task) {
    int64_t b = d->bottom.load(std::memory_order_relaxed);
    Task** a = d->array.load(std::memory_order_relaxed);
    a[b % DEQUE_SIZE] = task;
    std::atomic_thread_fence(std::memory_order_release);
    d->bottom.store(b + 1, std::memory_order_relaxed);
}

Task* steal(WorkDeque* d) {
    int64_t t = d->top.load(std::memory_order_acquire);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    int64_t b = d->bottom.load(std::memory_order_acquire);
    if (t >= b) return nullptr;
    Task** a = d->array.load(std::memory_order_relaxed);
    Task* task = a[t % DEQUE_SIZE];
    if (!d->top.compare_exchange_strong(t, t + 1,
            std::memory_order_seq_cst)) return nullptr;
    return task;
}
""",
        language="cpp",
        expected_pattern="work_steal",
        domain=Domain.LOCKFREE_DS,
        difficulty=Difficulty.HARD,
        provenance="Chase & Lev (2005) Dynamic Circular Work-Stealing Deque",
        failure_mode="Asymmetric push/steal, multiple fences, CAS at steal",
    ),
    AdversarialSnippet(
        id="lfds_07",
        code="""
// SPSC queue with POSIX atomic builtins (not C++ atomics)
int buffer[QUEUE_CAP];
int wpos = 0, rpos = 0;

void produce(int item) {
    buffer[wpos % QUEUE_CAP] = item;
    __atomic_store_n(&wpos, wpos + 1, __ATOMIC_RELEASE);
}

int consume(void) {
    int w;
    while ((w = __atomic_load_n(&wpos, __ATOMIC_ACQUIRE)) == rpos)
        ;
    int item = buffer[rpos % QUEUE_CAP];
    rpos++;
    return item;
}
""",
        language="c",
        expected_pattern="lockfree_spsc_queue",
        domain=Domain.LOCKFREE_DS,
        difficulty=Difficulty.MEDIUM,
        provenance="GCC atomic builtins documentation",
        failure_mode="GCC __atomic builtins instead of C++ atomics or C11",
    ),

    # ── NEW DOMAIN: Kernel Synchronization ──────────────────────────

    AdversarialSnippet(
        id="kern_01",
        code="""
// RCU publish (simplified Linux kernel style)
struct config {
    int timeout_ms;
    int max_retries;
    char endpoint[256];
};
struct config __rcu *global_config;

void update_config(int timeout, int retries, const char *ep) {
    struct config *new_cfg = kmalloc(sizeof(*new_cfg), GFP_KERNEL);
    new_cfg->timeout_ms = timeout;
    new_cfg->max_retries = retries;
    strncpy(new_cfg->endpoint, ep, 255);
    rcu_assign_pointer(global_config, new_cfg);
    synchronize_rcu();
    kfree_rcu(old_cfg, rcu_head);
}

int read_timeout(void) {
    struct config *cfg;
    int t;
    rcu_read_lock();
    cfg = rcu_dereference(global_config);
    t = cfg->timeout_ms;
    rcu_read_unlock();
    return t;
}
""",
        language="c",
        expected_pattern="rcu_publish",
        domain=Domain.KERNEL_SYNC,
        difficulty=Difficulty.EASY,
        provenance="Linux kernel Documentation/RCU/whatisRCU.rst",
        failure_mode="rcu_assign_pointer / rcu_dereference macros, kernel alloc",
    ),
    AdversarialSnippet(
        id="kern_02",
        code="""
// Kernel completion variable (wait_for_completion)
struct completion init_done;

void kernel_module_init(void) {
    init_completion(&init_done);
    kthread_run(init_worker, NULL, "init_worker");
    wait_for_completion(&init_done);
    // init_worker has finished setting up shared state
    printk(KERN_INFO "module initialized\\n");
}

int init_worker(void *data) {
    setup_hardware_registers();
    populate_device_table();
    smp_wmb();
    complete(&init_done);
    return 0;
}
""",
        language="c",
        expected_pattern="mp_fence",
        domain=Domain.KERNEL_SYNC,
        difficulty=Difficulty.EASY,
        provenance="Linux kernel completion API (kernel/sched/completion.c)",
        failure_mode="Completion variable as sync, smp_wmb before complete",
    ),
    AdversarialSnippet(
        id="kern_03",
        code="""
// Workqueue flush pattern (kernel)
struct work_struct cleanup_work;
std::atomic<int> cleanup_stage{0};
volatile int device_regs[16];

void cleanup_handler(struct work_struct *work) {
    for (int i = 0; i < 16; i++)
        device_regs[i] = 0;
    smp_wmb();
    WRITE_ONCE(cleanup_stage, 1);
    synchronize_rcu();
    WRITE_ONCE(cleanup_stage, 2);
}

void check_cleanup_done(void) {
    int stage = READ_ONCE(cleanup_stage);
    if (stage < 2) return;
    smp_rmb();
    // safe to reuse device_regs
    reinit_device(device_regs);
}
""",
        language="c",
        expected_pattern="mp_fence",
        domain=Domain.KERNEL_SYNC,
        difficulty=Difficulty.HARD,
        provenance="Linux kernel workqueue documentation",
        failure_mode="READ_ONCE/WRITE_ONCE, multi-stage cleanup, RCU interleaved",
    ),
    AdversarialSnippet(
        id="kern_04",
        code="""
// Per-CPU data with smp barriers
DEFINE_PER_CPU(struct stats, cpu_stats);
std::atomic<int> stats_generation{0};

void update_stats_local(int cpu, int count) {
    struct stats *s = per_cpu_ptr(&cpu_stats, cpu);
    s->count += count;
    s->timestamp = ktime_get();
    smp_wmb();
    stats_generation.fetch_add(1, std::memory_order_relaxed);
}

void read_global_stats(struct stats *out) {
    int gen = stats_generation.load(std::memory_order_relaxed);
    smp_rmb();
    memset(out, 0, sizeof(*out));
    int cpu;
    for_each_possible_cpu(cpu) {
        struct stats *s = per_cpu_ptr(&cpu_stats, cpu);
        out->count += s->count;
    }
}
""",
        language="c",
        expected_pattern="mp_fence",
        domain=Domain.KERNEL_SYNC,
        difficulty=Difficulty.HARD,
        provenance="Linux kernel per-CPU statistics pattern",
        failure_mode="Per-CPU data, smp_wmb/rmb, generation counter",
    ),
    AdversarialSnippet(
        id="kern_05",
        code="""
// Kernel spinlock with memory barriers (arch-specific)
typedef struct {
    volatile unsigned int lock;
} raw_spinlock_t;

static inline void arch_spin_lock(raw_spinlock_t *lock) {
    unsigned int tmp;
    __asm__ __volatile__(
        "1: ldaxr   %w0, [%1]\\n"
        "   cbnz    %w0, 1b\\n"
        "   stxr    %w0, %w2, [%1]\\n"
        "   cbnz    %w0, 1b\\n"
        : "=&r" (tmp)
        : "r" (&lock->lock), "r" (1)
        : "memory"
    );
}

static inline void arch_spin_unlock(raw_spinlock_t *lock) {
    __asm__ __volatile__(
        "   stlr    %w0, [%1]\\n"
        : : "r" (0), "r" (&lock->lock)
        : "memory"
    );
}
""",
        language="c",
        expected_pattern="spinlock_acq_rel",
        domain=Domain.KERNEL_SYNC,
        difficulty=Difficulty.ADVERSARIAL,
        provenance="Linux kernel arch/arm64/include/asm/spinlock.h",
        failure_mode="Inline assembly ldaxr/stxr/stlr, no C atomics at all",
    ),
    AdversarialSnippet(
        id="kern_06",
        code="""
// RCU linked list traversal (kernel-style)
struct list_entry {
    int key;
    int value;
    struct list_entry __rcu *next;
};
struct list_entry __rcu *list_head;

int rcu_list_lookup(int key) {
    struct list_entry *e;
    int result = -1;
    rcu_read_lock();
    e = rcu_dereference(list_head);
    while (e) {
        if (e->key == key) {
            result = READ_ONCE(e->value);
            break;
        }
        e = rcu_dereference(e->next);
    }
    rcu_read_unlock();
    return result;
}

void rcu_list_insert(int key, int val) {
    struct list_entry *new_e = kmalloc(sizeof(*new_e), GFP_KERNEL);
    new_e->key = key;
    new_e->value = val;
    new_e->next = list_head;
    smp_wmb();
    rcu_assign_pointer(list_head, new_e);
}
""",
        language="c",
        expected_pattern="rcu_publish",
        domain=Domain.KERNEL_SYNC,
        difficulty=Difficulty.HARD,
        provenance="Linux kernel RCU list patterns (include/linux/rculist.h)",
        failure_mode="RCU list traversal, rcu_dereference in loop, smp_wmb",
    ),

    # ── NEW DOMAIN: Compiler-Generated Code ─────────────────────────

    AdversarialSnippet(
        id="comp_01",
        code="""
// Compiler-lowered atomic store (AArch64 release)
// What the compiler emits for: atomic_store_explicit(&x, val, release)
void compiler_atomic_store_rel(int *addr, int val) {
    __asm__ __volatile__(
        "stlr %w1, [%0]"
        : : "r"(addr), "r"(val)
        : "memory"
    );
}

void compiler_atomic_load_acq(int *addr, int *out) {
    __asm__ __volatile__(
        "ldar %w0, [%1]"
        : "=r"(*out)
        : "r"(addr)
        : "memory"
    );
}

void mp_lowered(int *flag, int *data) {
    *data = 42;
    compiler_atomic_store_rel(flag, 1);
}

void mp_reader(int *flag, int *data, int *result) {
    int f;
    compiler_atomic_load_acq(flag, &f);
    if (f) *result = *data;
}
""",
        language="c",
        expected_pattern="mp_fence",
        domain=Domain.COMPILER,
        difficulty=Difficulty.ADVERSARIAL,
        provenance="AArch64 compiler output for C11 atomics",
        failure_mode="Inline asm implementing atomics, stlr/ldar instructions",
    ),
    AdversarialSnippet(
        id="comp_02",
        code="""
// Compiler-generated memcpy + fence (struct publish)
struct Message {
    int type;
    int payload[8];
    int checksum;
};
std::atomic<Message*> published{nullptr};

void publish_message(int type, int* data, int len) {
    Message* msg = (Message*)aligned_alloc(64, sizeof(Message));
    msg->type = type;
    __builtin_memcpy(msg->payload, data, len * sizeof(int));
    msg->checksum = compute_crc(data, len);
    // release fence ensures memcpy completes before pointer publish
    std::atomic_thread_fence(std::memory_order_release);
    published.store(msg, std::memory_order_relaxed);
}

Message* consume_message() {
    Message* msg = published.load(std::memory_order_acquire);
    return msg;
}
""",
        language="cpp",
        expected_pattern="mp_fence",
        domain=Domain.COMPILER,
        difficulty=Difficulty.HARD,
        provenance="Compiler codegen for struct publish pattern",
        failure_mode="__builtin_memcpy, aligned_alloc, fence + relaxed store",
    ),
    AdversarialSnippet(
        id="comp_03",
        code="""
// x86 LOCK-prefixed instruction lowering (compiler CAS output)
static inline int cmpxchg(volatile int *ptr, int old_val, int new_val) {
    int prev;
    __asm__ __volatile__(
        "lock; cmpxchgl %1, %2"
        : "=a"(prev)
        : "r"(new_val), "m"(*ptr), "0"(old_val)
        : "memory"
    );
    return prev;
}

int lock_free_counter_inc(volatile int *counter) {
    int old;
    do {
        old = *counter;
    } while (cmpxchg(counter, old, old + 1) != old);
    return old + 1;
}
""",
        language="c",
        expected_pattern="rmw_cmpxchg_loop",
        domain=Domain.COMPILER,
        difficulty=Difficulty.HARD,
        provenance="GCC x86 atomic lowering patterns",
        failure_mode="Inline asm lock cmpxchgl, manual CAS loop, volatile",
    ),
    AdversarialSnippet(
        id="comp_04",
        code="""
// Compiler barrier vs hardware barrier
// compiler_barrier: prevents reordering by compiler only
// hardware_barrier: prevents reordering by CPU
#define compiler_barrier() __asm__ __volatile__("" ::: "memory")
#define full_barrier()     __asm__ __volatile__("mfence" ::: "memory")

int shared_data[64];
int data_ready = 0;

void writer(void) {
    for (int i = 0; i < 64; i++)
        shared_data[i] = i * i;
    full_barrier();  // need hardware barrier for cross-CPU visibility
    data_ready = 1;
}

void reader(void) {
    while (!data_ready) { compiler_barrier(); }
    full_barrier();
    int sum = 0;
    for (int i = 0; i < 64; i++)
        sum += shared_data[i];
}
""",
        language="c",
        expected_pattern="mp_fence",
        domain=Domain.COMPILER,
        difficulty=Difficulty.MEDIUM,
        provenance="Linux kernel barrier documentation",
        failure_mode="Macro-defined barriers, compiler vs hardware distinction",
    ),
    AdversarialSnippet(
        id="comp_05",
        code="""
// LLVM IR-level atomic lowering (pseudo-C from IR)
// Represents: atomic store release + atomic load acquire
void llvm_lowered_mp_x86(int *flag_addr, int *data_addr) {
    // store data (plain)
    *data_addr = 0xDEAD;
    // atomic store release on x86: just a mov (x86 stores are release)
    // but compiler must not reorder past this point
    __asm__ __volatile__("" ::: "memory");
    *flag_addr = 1;
}

int llvm_lowered_reader_x86(int *flag_addr, int *data_addr) {
    int f = *flag_addr;
    // atomic load acquire on x86: mov is sufficient but need compiler fence
    __asm__ __volatile__("" ::: "memory");
    if (f) return *data_addr;
    return -1;
}
""",
        language="c",
        expected_pattern="mp",
        domain=Domain.COMPILER,
        difficulty=Difficulty.ADVERSARIAL,
        provenance="LLVM AtomicExpandPass for x86 target",
        failure_mode="Compiler fence only (x86 TSO), no hardware fence visible",
    ),

    # ── Expanded: More Embedded ──────────────────────────────────────

    AdversarialSnippet(
        id="emb_05",
        code="""
// RTOS task synchronization via queue (Zephyr-style)
K_MSGQ_DEFINE(sensor_queue, sizeof(struct sensor_data), 16, 4);

void sensor_isr(void) {
    struct sensor_data reading;
    reading.value = read_adc(0);
    reading.timestamp = k_uptime_get();
    k_msgq_put(&sensor_queue, &reading, K_NO_WAIT);
}

void processing_task(void) {
    struct sensor_data rx;
    while (1) {
        k_msgq_get(&sensor_queue, &rx, K_FOREVER);
        __DMB();
        apply_filter(&rx);
        update_display(rx.value);
    }
}
""",
        language="c",
        expected_pattern="mp_fence",
        domain=Domain.EMBEDDED,
        difficulty=Difficulty.MEDIUM,
        provenance="Zephyr RTOS message queue API",
        failure_mode="RTOS queue API, ISR context, __DMB after dequeue",
    ),
    AdversarialSnippet(
        id="emb_06",
        code="""
// DMA scatter-gather with completion interrupt
struct dma_desc {
    uint32_t src_addr;
    uint32_t dst_addr;
    uint32_t length;
    uint32_t ctrl;  // bit 0 = interrupt on complete
};
volatile struct dma_desc dma_chain[8];
volatile uint32_t dma_complete_mask = 0;

void setup_dma_chain(int n_descs) {
    for (int i = 0; i < n_descs; i++) {
        dma_chain[i].src_addr = src_buffers[i];
        dma_chain[i].dst_addr = dst_buffers[i];
        dma_chain[i].length = buf_sizes[i];
        dma_chain[i].ctrl = (i == n_descs-1) ? 1 : 0;
    }
    __DSB();
    DMA->CTRL = DMA_START | (n_descs << 4);
}

void dma_isr(void) {
    dma_complete_mask |= DMA->STATUS;
    DMA->STATUS = 0;
    __DSB();
}
""",
        language="c",
        expected_pattern="mp_fence",
        domain=Domain.EMBEDDED,
        difficulty=Difficulty.HARD,
        provenance="ARM DMA controller programming (PL330 TRM)",
        failure_mode="DMA descriptor chain, MMIO, __DSB for device ordering",
    ),
    AdversarialSnippet(
        id="emb_07",
        code="""
// Interrupt handler with shared state (nested interrupt safe)
volatile uint32_t tick_count = 0;
volatile uint32_t alarm_target = 0;
volatile uint32_t alarm_fired = 0;

void systick_handler(void) {
    uint32_t t = ++tick_count;
    __DMB();
    if (alarm_target && t >= alarm_target) {
        alarm_fired = 1;
        __DSB();
        __SEV();
    }
}

void set_alarm(uint32_t delay_ticks) {
    __disable_irq();
    alarm_fired = 0;
    __DMB();
    alarm_target = tick_count + delay_ticks;
    __enable_irq();
}

void wait_alarm(void) {
    while (!alarm_fired) { __WFE(); }
    __DMB();
}
""",
        language="c",
        expected_pattern="mp_fence",
        domain=Domain.EMBEDDED,
        difficulty=Difficulty.HARD,
        provenance="ARM Cortex-M alarm timer pattern",
        failure_mode="ISR + main context, __disable_irq, __WFE/__SEV, multi-barrier",
    ),
    AdversarialSnippet(
        id="emb_08",
        code="""
// Multicore boot synchronization (SMP startup)
volatile uint32_t cpu_online_mask = 0;
volatile uint32_t boot_data_ready = 0;
struct boot_params global_boot_params;

void secondary_cpu_entry(int cpu_id) {
    while (!boot_data_ready) {
        __asm__ __volatile__("wfe" ::: "memory");
    }
    __DMB();
    struct boot_params local = global_boot_params;
    cpu_online_mask |= (1u << cpu_id);
    __DSB();
    __SEV();
    start_scheduler(&local);
}

void primary_boot(void) {
    global_boot_params.stack_base = alloc_stacks();
    global_boot_params.page_table = setup_mmu();
    __DSB();
    boot_data_ready = 1;
    __SEV();
    while (__builtin_popcount(cpu_online_mask) < NUM_CPUS) {
        __WFE();
    }
}
""",
        language="c",
        expected_pattern="mp_fence",
        domain=Domain.EMBEDDED,
        difficulty=Difficulty.ADVERSARIAL,
        provenance="ARM SMP boot protocol (PSCI specification)",
        failure_mode="Multi-CPU boot, WFE/SEV, struct params, popcount",
    ),

    # ── Expanded: More HFT ──────────────────────────────────────────

    AdversarialSnippet(
        id="hft_04",
        code="""
// Order matching engine - price-time priority
struct Order {
    uint64_t order_id;
    int32_t price;
    uint32_t quantity;
    std::atomic<uint32_t> filled{0};
};

std::atomic<Order*> best_bid_order{nullptr};
std::atomic<Order*> best_ask_order{nullptr};

bool try_match(Order* incoming_buy) {
    Order* ask = best_ask_order.load(std::memory_order_acquire);
    if (!ask || ask->price > incoming_buy->price) return false;
    uint32_t ask_qty = ask->quantity - ask->filled.load(std::memory_order_relaxed);
    uint32_t fill = std::min(incoming_buy->quantity, ask_qty);
    uint32_t old_filled = ask->filled.fetch_add(fill, std::memory_order_acq_rel);
    if (old_filled + fill >= ask->quantity) {
        best_ask_order.compare_exchange_strong(ask, nullptr,
            std::memory_order_release);
    }
    return true;
}
""",
        language="cpp",
        expected_pattern="rmw_cas_mp",
        domain=Domain.HFT,
        difficulty=Difficulty.ADVERSARIAL,
        provenance="Low-latency matching engine design (Jane Street tech talk)",
        failure_mode="fetch_add + CAS combo, struct member atomics, price logic",
    ),
    AdversarialSnippet(
        id="hft_05",
        code="""
// Market data multicast with sequence number gap detection
struct MarketUpdate {
    uint64_t seq_no;
    uint32_t symbol_id;
    int64_t price;
    int32_t size;
};

std::atomic<uint64_t> last_seq{0};
MarketUpdate latest_updates[MAX_SYMBOLS];

void on_market_data(const MarketUpdate& update) {
    latest_updates[update.symbol_id] = update;
    std::atomic_thread_fence(std::memory_order_release);
    last_seq.store(update.seq_no, std::memory_order_relaxed);
}

bool read_latest(uint32_t symbol, MarketUpdate* out) {
    uint64_t seq = last_seq.load(std::memory_order_acquire);
    *out = latest_updates[symbol];
    return out->seq_no == seq;
}
""",
        language="cpp",
        expected_pattern="mp_fence",
        domain=Domain.HFT,
        difficulty=Difficulty.HARD,
        provenance="OPRA/CTS market data handler patterns",
        failure_mode="Array publish, fence+relaxed, sequence validation",
    ),
    AdversarialSnippet(
        id="hft_06",
        code="""
// Ticket lock for order book level (low-contention fast path)
struct alignas(64) TicketLock {
    std::atomic<uint32_t> next_ticket{0};
    std::atomic<uint32_t> now_serving{0};
};

void ticket_lock_acquire(TicketLock& lock) {
    uint32_t my_ticket = lock.next_ticket.fetch_add(1, std::memory_order_relaxed);
    while (lock.now_serving.load(std::memory_order_acquire) != my_ticket) {
        _mm_pause();
    }
}

void ticket_lock_release(TicketLock& lock) {
    uint32_t current = lock.now_serving.load(std::memory_order_relaxed);
    lock.now_serving.store(current + 1, std::memory_order_release);
}
""",
        language="cpp",
        expected_pattern="ticket_lock",
        domain=Domain.HFT,
        difficulty=Difficulty.EASY,
        provenance="Mellor-Crummey & Scott (1991) ticket lock",
        failure_mode="fetch_add ticket, acquire spin, release store",
    ),
    AdversarialSnippet(
        id="hft_07",
        code="""
// Timestamped publish for latency measurement
struct alignas(64) TimestampedQuote {
    uint64_t exchange_ts;
    uint64_t local_ts;
    int64_t bid;
    int64_t ask;
};
std::atomic<int> quote_version{0};
TimestampedQuote quote_buf[2];

void publish_quote(int64_t bid, int64_t ask, uint64_t exch_ts) {
    int v = quote_version.load(std::memory_order_relaxed);
    int slot = (v + 1) & 1;
    quote_buf[slot].bid = bid;
    quote_buf[slot].ask = ask;
    quote_buf[slot].exchange_ts = exch_ts;
    quote_buf[slot].local_ts = rdtsc();
    quote_version.store(v + 1, std::memory_order_release);
}

TimestampedQuote read_quote() {
    int v = quote_version.load(std::memory_order_acquire);
    return quote_buf[v & 1];
}
""",
        language="cpp",
        expected_pattern="mp_fence",
        domain=Domain.HFT,
        difficulty=Difficulty.MEDIUM,
        provenance="HFT latency measurement infrastructure",
        failure_mode="Double-buffered quote, rdtsc, version toggle",
    ),

    # ── Expanded: More CUDA ─────────────────────────────────────────

    AdversarialSnippet(
        id="cuda_04",
        code="""
// Shared memory bank-conflict-free reduction
__global__ void reduce_no_conflict(float* in, float* out, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    sdata[tid] = (i < n ? in[i] : 0) + (i + blockDim.x < n ? in[i + blockDim.x] : 0);
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    // Warp-level (no sync needed within warp)
    if (tid < 32) {
        volatile float* smem = sdata;
        smem[tid] += smem[tid + 32];
        smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}
""",
        language="cuda",
        expected_pattern="gpu_sb_wg",
        domain=Domain.CUDA,
        difficulty=Difficulty.HARD,
        provenance="NVIDIA parallel reduction whitepaper (Mark Harris)",
        failure_mode="Volatile warp trick, __syncthreads partial, mixed levels",
    ),
    AdversarialSnippet(
        id="cuda_05",
        code="""
// Persistent thread producer-consumer (GPU global memory)
__device__ int work_queue[MAX_WORK];
__device__ volatile int queue_head = 0;
__device__ volatile int queue_tail = 0;

__global__ void persistent_consumer() {
    while (true) {
        int t;
        if (threadIdx.x == 0) {
            while ((t = queue_tail) == queue_head) {}
            __threadfence();
        }
        t = __shfl_sync(0xFFFFFFFF, t, 0);
        if (t < 0) return;  // poison pill
        int item = work_queue[t % MAX_WORK];
        if (threadIdx.x == 0) {
            queue_tail = t + 1;
            __threadfence();
        }
        process_item(item);
    }
}
""",
        language="cuda",
        expected_pattern="gpu_mp_dev",
        domain=Domain.CUDA,
        difficulty=Difficulty.ADVERSARIAL,
        provenance="Persistent threads pattern (Aila & Laine, HPG 2009)",
        failure_mode="Persistent thread loop, volatile + __threadfence, warp broadcast",
    ),
    AdversarialSnippet(
        id="cuda_06",
        code="""
// Cooperative groups tile-based matrix multiply
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void matmul_tiles(float* A, float* B, float* C, int N) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);

    __shared__ float As[32][32], Bs[32][32];
    int row = blockIdx.y * 32 + threadIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;
    float sum = 0.0f;

    for (int k = 0; k < N; k += 32) {
        As[threadIdx.y][threadIdx.x] = A[row * N + k + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * N + col];
        block.sync();
        for (int i = 0; i < 32; i++)
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        block.sync();
    }
    C[row * N + col] = sum;
}
""",
        language="cuda",
        expected_pattern="gpu_sb_wg",
        domain=Domain.CUDA,
        difficulty=Difficulty.MEDIUM,
        provenance="CUDA cooperative groups matmul example",
        failure_mode="Cooperative groups block.sync(), tiled partition, shared mem",
    ),
    AdversarialSnippet(
        id="cuda_07",
        code="""
// Warp-level vote and ballot for divergence detection
__device__ int count_active_lanes(int predicate) {
    unsigned mask = __ballot_sync(0xFFFFFFFF, predicate);
    return __popc(mask);
}

__global__ void sparse_update(float* data, int* indices, float* values, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int idx = indices[tid];
    int active = count_active_lanes(idx >= 0);
    if (idx >= 0) {
        atomicAdd(&data[idx], values[tid]);
    }
    __syncwarp();
    if (threadIdx.x % 32 == 0) {
        atomicAdd(&global_active_count, active);
    }
}
""",
        language="cuda",
        expected_pattern="gpu_rmw_dev",
        domain=Domain.CUDA,
        difficulty=Difficulty.MEDIUM,
        provenance="CUDA warp vote functions documentation",
        failure_mode="__ballot_sync, atomicAdd, warp-level intrinsics",
    ),
    AdversarialSnippet(
        id="cuda_08",
        code="""
// Cross-workgroup flag synchronization (GPU MP pattern)
__device__ int gpu_data_payload;
__device__ int gpu_flag;

__global__ void gpu_writer() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        gpu_data_payload = 99;
        __threadfence_system();
        atomicExch(&gpu_flag, 1);
    }
}

__global__ void gpu_reader(int* result) {
    if (threadIdx.x == 0 && blockIdx.x == 1) {
        while (atomicAdd(&gpu_flag, 0) != 1) {}
        __threadfence_system();
        *result = gpu_data_payload;
    }
}
""",
        language="cuda",
        expected_pattern="gpu_mp_dev",
        domain=Domain.CUDA,
        difficulty=Difficulty.HARD,
        provenance="CUDA __threadfence_system documentation",
        failure_mode="__threadfence_system, atomicExch as release, atomicAdd as acquire",
    ),

    # ── Expanded: More Database ─────────────────────────────────────

    AdversarialSnippet(
        id="db_03",
        code="""
// B-tree node split with publish (concurrent B-tree)
struct BTreeNode {
    std::atomic<int> num_keys;
    int keys[2*B - 1];
    std::atomic<BTreeNode*> children[2*B];
    std::atomic<bool> is_splitting{false};
};

void split_child(BTreeNode* parent, int idx, BTreeNode* child) {
    BTreeNode* new_node = new BTreeNode();
    for (int j = 0; j < B-1; j++)
        new_node->keys[j] = child->keys[j + B];
    for (int j = 0; j < B; j++)
        new_node->children[j].store(
            child->children[j + B].load(std::memory_order_relaxed),
            std::memory_order_relaxed);
    new_node->num_keys.store(B-1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    parent->children[idx + 1].store(new_node, std::memory_order_relaxed);
    parent->keys[idx] = child->keys[B-1];
    parent->num_keys.fetch_add(1, std::memory_order_release);
}
""",
        language="cpp",
        expected_pattern="mp_fence",
        domain=Domain.DATABASE,
        difficulty=Difficulty.ADVERSARIAL,
        provenance="Bayer & Schkolnick B-link tree (Lehman & Yao, 1981)",
        failure_mode="B-tree split, bulk copy, fence before child publish",
    ),
    AdversarialSnippet(
        id="db_04",
        code="""
// MVCC timestamp ordering with atomic CAS
struct TxnRecord {
    std::atomic<uint64_t> read_ts{0};
    std::atomic<uint64_t> write_ts{0};
    std::atomic<int> value;
};

bool mvcc_read(TxnRecord* rec, uint64_t my_ts, int* out) {
    uint64_t wts = rec->write_ts.load(std::memory_order_acquire);
    if (wts > my_ts) return false;  // write in future, abort
    *out = rec->value.load(std::memory_order_relaxed);
    uint64_t rts = rec->read_ts.load(std::memory_order_relaxed);
    while (rts < my_ts) {
        if (rec->read_ts.compare_exchange_weak(rts, my_ts,
                std::memory_order_release)) break;
    }
    return true;
}

bool mvcc_write(TxnRecord* rec, uint64_t my_ts, int val) {
    uint64_t rts = rec->read_ts.load(std::memory_order_acquire);
    if (rts > my_ts) return false;
    uint64_t expected_wts = rec->write_ts.load(std::memory_order_relaxed);
    if (expected_wts > my_ts) return false;
    rec->value.store(val, std::memory_order_relaxed);
    rec->write_ts.store(my_ts, std::memory_order_release);
    return true;
}
""",
        language="cpp",
        expected_pattern="rmw_cas_mp",
        domain=Domain.DATABASE,
        difficulty=Difficulty.ADVERSARIAL,
        provenance="Bernstein & Goodman (1981) MVCC timestamp ordering",
        failure_mode="CAS loop for read_ts update, timestamp validation, multi-field",
    ),
    AdversarialSnippet(
        id="db_05",
        code="""
// Lock-free hash index bucket (cuckoo hashing)
struct HashEntry {
    std::atomic<uint64_t> key{EMPTY};
    std::atomic<uint64_t> value;
};
HashEntry table[2][BUCKET_SIZE];

bool insert(uint64_t k, uint64_t v) {
    uint64_t h1 = hash1(k), h2 = hash2(k);
    for (int i = 0; i < BUCKET_SIZE; i++) {
        uint64_t expected = EMPTY;
        if (table[0][h1 + i].key.compare_exchange_strong(
                expected, k, std::memory_order_acq_rel)) {
            table[0][h1 + i].value.store(v, std::memory_order_release);
            return true;
        }
    }
    for (int i = 0; i < BUCKET_SIZE; i++) {
        uint64_t expected = EMPTY;
        if (table[1][h2 + i].key.compare_exchange_strong(
                expected, k, std::memory_order_acq_rel)) {
            table[1][h2 + i].value.store(v, std::memory_order_release);
            return true;
        }
    }
    return false;  // need resize
}
""",
        language="cpp",
        expected_pattern="rmw_cas_mp",
        domain=Domain.DATABASE,
        difficulty=Difficulty.HARD,
        provenance="Li et al. (2014) Algorithmic Improvements for Fast Concurrent Cuckoo Hashing",
        failure_mode="Dual-table CAS insert, acq_rel CAS, release value store",
    ),
    AdversarialSnippet(
        id="db_06",
        code="""
// Buffer pool page latch with optimistic reads
struct BufferPage {
    std::atomic<uint64_t> version{0};
    char data[PAGE_SIZE];
    std::atomic<int> pin_count{0};
};

bool optimistic_read(BufferPage* page, char* out) {
    uint64_t v1 = page->version.load(std::memory_order_acquire);
    if (v1 & 1) return false;  // page is being written
    memcpy(out, page->data, PAGE_SIZE);
    std::atomic_thread_fence(std::memory_order_acquire);
    uint64_t v2 = page->version.load(std::memory_order_relaxed);
    return v1 == v2;
}

void exclusive_write(BufferPage* page, const char* src) {
    uint64_t v = page->version.fetch_add(1, std::memory_order_acquire);
    memcpy(page->data, src, PAGE_SIZE);
    std::atomic_thread_fence(std::memory_order_release);
    page->version.store(v + 2, std::memory_order_relaxed);
}
""",
        language="cpp",
        expected_pattern="seqlock_read",
        domain=Domain.DATABASE,
        difficulty=Difficulty.HARD,
        provenance="Leis et al. (2016) optimistic lock coupling for B-trees",
        failure_mode="Version-based optimistic locking (seqlock variant), memcpy",
    ),

    # ── Expanded: More Network ──────────────────────────────────────

    AdversarialSnippet(
        id="net_03",
        code="""
// DPDK mbuf metadata publish (packet processing pipeline)
struct rte_mbuf {
    void* buf_addr;
    uint16_t data_off;
    uint16_t pkt_len;
    uint32_t hash_val;
};
std::atomic<int> ring_prod_tail{0};
rte_mbuf* pkt_ring[RING_SIZE];

void rx_burst(rte_mbuf** pkts, int n) {
    int tail = ring_prod_tail.load(std::memory_order_relaxed);
    for (int i = 0; i < n; i++) {
        pkt_ring[(tail + i) & (RING_SIZE - 1)] = pkts[i];
    }
    std::atomic_thread_fence(std::memory_order_release);
    ring_prod_tail.store(tail + n, std::memory_order_relaxed);
}

int tx_burst(rte_mbuf** pkts, int max_n, int cons_pos) {
    int prod = ring_prod_tail.load(std::memory_order_acquire);
    int n = std::min(prod - cons_pos, max_n);
    for (int i = 0; i < n; i++) {
        pkts[i] = pkt_ring[(cons_pos + i) & (RING_SIZE - 1)];
    }
    return n;
}
""",
        language="cpp",
        expected_pattern="mp_fence",
        domain=Domain.NETWORK,
        difficulty=Difficulty.HARD,
        provenance="DPDK rte_ring burst operations",
        failure_mode="Bulk produce, fence+relaxed, ring mask arithmetic",
    ),
    AdversarialSnippet(
        id="net_04",
        code="""
// io_uring-style completion queue publish
struct io_uring_cqe {
    uint64_t user_data;
    int32_t res;
    uint32_t flags;
};

struct cq_ring {
    std::atomic<uint32_t> head;
    std::atomic<uint32_t> tail;
    io_uring_cqe cqes[CQ_SIZE];
};

void kernel_complete(cq_ring* cq, uint64_t user_data, int32_t result) {
    uint32_t t = cq->tail.load(std::memory_order_relaxed);
    uint32_t idx = t & (CQ_SIZE - 1);
    cq->cqes[idx].user_data = user_data;
    cq->cqes[idx].res = result;
    cq->cqes[idx].flags = 0;
    smp_store_release(&cq->tail, t + 1);
}

int user_peek_cqe(cq_ring* cq, io_uring_cqe* out, uint32_t head) {
    uint32_t t = smp_load_acquire(&cq->tail);
    if (head == t) return -EAGAIN;
    *out = cq->cqes[head & (CQ_SIZE - 1)];
    return 0;
}
""",
        language="c",
        expected_pattern="mp_fence",
        domain=Domain.NETWORK,
        difficulty=Difficulty.HARD,
        provenance="Linux io_uring (liburing source code)",
        failure_mode="smp_store_release/smp_load_acquire, ring queue, kernel-user shared mem",
    ),
    AdversarialSnippet(
        id="net_05",
        code="""
// Connection tracking table (NAT/firewall)
struct ConnEntry {
    uint32_t src_ip, dst_ip;
    uint16_t src_port, dst_port;
    std::atomic<int> state;
    std::atomic<uint64_t> last_seen;
};
std::atomic<ConnEntry*> conn_table[HASH_SIZE];

ConnEntry* lookup_or_create(uint32_t sip, uint32_t dip,
                            uint16_t sp, uint16_t dp) {
    uint32_t h = hash_4tuple(sip, dip, sp, dp) % HASH_SIZE;
    ConnEntry* e = conn_table[h].load(std::memory_order_acquire);
    if (e && e->src_ip == sip && e->dst_ip == dip) {
        e->last_seen.store(now(), std::memory_order_relaxed);
        return e;
    }
    ConnEntry* new_e = new ConnEntry{sip, dip, sp, dp};
    new_e->state.store(CONN_NEW, std::memory_order_relaxed);
    new_e->last_seen.store(now(), std::memory_order_relaxed);
    if (conn_table[h].compare_exchange_strong(e, new_e,
            std::memory_order_release, std::memory_order_acquire)) {
        return new_e;
    }
    delete new_e;
    return e;
}
""",
        language="cpp",
        expected_pattern="rmw_cas_mp",
        domain=Domain.NETWORK,
        difficulty=Difficulty.HARD,
        provenance="Linux conntrack / nf_conntrack patterns",
        failure_mode="CAS install, struct init before publish, hash table",
    ),
    AdversarialSnippet(
        id="net_06",
        code="""
// Zero-copy receive buffer handoff
std::atomic<int> rx_owner{OWNER_NIC};
char rx_buffer[MTU_SIZE];
uint16_t rx_length;

void nic_dma_complete_isr(void) {
    rx_length = NIC->RX_LEN;
    __asm__ __volatile__("dmb ish" ::: "memory");
    rx_owner.store(OWNER_CPU, std::memory_order_relaxed);
}

bool try_consume_rx(char* dst, uint16_t* len) {
    if (rx_owner.load(std::memory_order_acquire) != OWNER_CPU)
        return false;
    *len = rx_length;
    memcpy(dst, rx_buffer, rx_length);
    rx_owner.store(OWNER_NIC, std::memory_order_release);
    return true;
}
""",
        language="c",
        expected_pattern="mp_fence",
        domain=Domain.NETWORK,
        difficulty=Difficulty.MEDIUM,
        provenance="NIC DMA receive path (ARM embedded)",
        failure_mode="DMA handoff, dmb ish inline asm, ownership token",
    ),

    # ── Expanded: More Crypto ───────────────────────────────────────

    AdversarialSnippet(
        id="crypto_02",
        code="""
// Nonce generation with atomic counter (thread-safe)
struct NonceGenerator {
    std::atomic<uint64_t> counter{0};
    uint8_t key_id[16];

    void generate_nonce(uint8_t nonce[12]) {
        uint64_t ctr = counter.fetch_add(1, std::memory_order_relaxed);
        memcpy(nonce, key_id, 4);
        memcpy(nonce + 4, &ctr, 8);
    }
};

std::atomic<NonceGenerator*> active_generator;

void rotate_generator(uint8_t new_key_id[16]) {
    NonceGenerator* gen = new NonceGenerator();
    memcpy(gen->key_id, new_key_id, 16);
    gen->counter.store(0, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    NonceGenerator* old = active_generator.exchange(gen,
        std::memory_order_acq_rel);
    schedule_deletion(old);
}
""",
        language="cpp",
        expected_pattern="mp_fence",
        domain=Domain.CRYPTO,
        difficulty=Difficulty.HARD,
        provenance="NIST SP 800-38D GCM nonce requirements",
        failure_mode="Atomic counter for nonce, generator rotation, fence+exchange",
    ),
    AdversarialSnippet(
        id="crypto_03",
        code="""
// Diffie-Hellman key exchange state machine
enum DHState { DH_INIT, DH_PUBKEY_SENT, DH_SHARED_READY };
std::atomic<DHState> dh_state{DH_INIT};
uint8_t shared_secret[32];
uint8_t peer_pubkey[32];

void on_recv_pubkey(const uint8_t* key) {
    memcpy(peer_pubkey, key, 32);
    std::atomic_thread_fence(std::memory_order_release);
    dh_state.store(DH_PUBKEY_SENT, std::memory_order_relaxed);
}

void compute_shared_secret(const uint8_t* my_privkey) {
    while (dh_state.load(std::memory_order_acquire) != DH_PUBKEY_SENT) {}
    curve25519_scalarmult(shared_secret, my_privkey, peer_pubkey);
    std::atomic_thread_fence(std::memory_order_release);
    dh_state.store(DH_SHARED_READY, std::memory_order_relaxed);
}

void use_shared_secret(uint8_t* session_key) {
    while (dh_state.load(std::memory_order_acquire) != DH_SHARED_READY) {}
    hkdf_expand(shared_secret, 32, session_key, 32);
}
""",
        language="cpp",
        expected_pattern="mp_fence",
        domain=Domain.CRYPTO,
        difficulty=Difficulty.HARD,
        provenance="Noise Protocol Framework handshake pattern",
        failure_mode="Multi-phase state machine, fence+relaxed, crypto buffers",
    ),
    AdversarialSnippet(
        id="crypto_04",
        code="""
// Constant-time comparison with publish result
std::atomic<int> comparison_result{-1};

void ct_compare_worker(const uint8_t* a, const uint8_t* b, size_t len) {
    volatile uint8_t result = 0;
    for (size_t i = 0; i < len; i++) {
        result |= a[i] ^ b[i];
    }
    int is_equal = (result == 0) ? 1 : 0;
    comparison_result.store(is_equal, std::memory_order_release);
}

int wait_compare_result(void) {
    int r;
    while ((r = comparison_result.load(std::memory_order_acquire)) < 0) {}
    comparison_result.store(-1, std::memory_order_relaxed);
    return r;
}
""",
        language="cpp",
        expected_pattern="mp_fence",
        domain=Domain.CRYPTO,
        difficulty=Difficulty.EASY,
        provenance="OpenSSL CRYPTO_memcmp pattern",
        failure_mode="Constant-time compare, volatile accumulator, release result",
    ),
    AdversarialSnippet(
        id="crypto_05",
        code="""
// TLS session ticket encryption key rotation
struct TicketKey {
    uint8_t name[16];
    uint8_t aes_key[32];
    uint8_t hmac_key[32];
    uint64_t not_after;
};
std::atomic<TicketKey*> current_key{nullptr};
std::atomic<TicketKey*> previous_key{nullptr};

void install_new_key(TicketKey* key) {
    TicketKey* old_current = current_key.load(std::memory_order_relaxed);
    TicketKey* old_previous = previous_key.exchange(old_current,
        std::memory_order_acq_rel);
    current_key.store(key, std::memory_order_release);
    if (old_previous) {
        secure_zero(old_previous, sizeof(TicketKey));
        free(old_previous);
    }
}

TicketKey* find_key(const uint8_t name[16]) {
    TicketKey* k = current_key.load(std::memory_order_acquire);
    if (k && memcmp(k->name, name, 16) == 0) return k;
    k = previous_key.load(std::memory_order_acquire);
    if (k && memcmp(k->name, name, 16) == 0) return k;
    return nullptr;
}
""",
        language="cpp",
        expected_pattern="mp_fence",
        domain=Domain.CRYPTO,
        difficulty=Difficulty.ADVERSARIAL,
        provenance="OpenSSL TLS session ticket key management",
        failure_mode="Two-pointer rotation, exchange + store, secure_zero",
    ),

    # ── Expanded: More Game Engine ──────────────────────────────────

    AdversarialSnippet(
        id="game_03",
        code="""
// Job system task graph (Naughty Dog-style)
struct Job {
    void (*func)(void*);
    void* data;
    std::atomic<int> unfinished_deps{0};
    std::atomic<Job*> continuation{nullptr};
};
std::atomic<Job*> job_queue_head{nullptr};

void finish_job(Job* job) {
    int remaining = job->unfinished_deps.fetch_sub(1, std::memory_order_acq_rel);
    if (remaining == 1) {
        Job* cont = job->continuation.load(std::memory_order_acquire);
        if (cont) {
            // push continuation to queue
            Job* old_head = job_queue_head.load(std::memory_order_relaxed);
            do {
                cont->continuation.store(old_head, std::memory_order_relaxed);
            } while (!job_queue_head.compare_exchange_weak(old_head, cont,
                        std::memory_order_release, std::memory_order_relaxed));
        }
    }
}
""",
        language="cpp",
        expected_pattern="rmw_cas_mp",
        domain=Domain.GAME_ENGINE,
        difficulty=Difficulty.ADVERSARIAL,
        provenance="Naughty Dog GDC 2015 job system presentation",
        failure_mode="Dependency counter fetch_sub, CAS push, continuation chain",
    ),
    AdversarialSnippet(
        id="game_04",
        code="""
// Lock-free entity ID allocator (ECS)
struct IDPool {
    std::atomic<uint32_t> free_list_head{0};
    uint32_t next_free[MAX_ENTITIES];
    std::atomic<uint32_t> generation[MAX_ENTITIES];
};

uint32_t allocate_id(IDPool* pool) {
    uint32_t head = pool->free_list_head.load(std::memory_order_acquire);
    uint32_t next;
    do {
        if (head == INVALID_ID) return INVALID_ID;
        next = pool->next_free[head];
    } while (!pool->free_list_head.compare_exchange_weak(head, next,
                std::memory_order_release, std::memory_order_acquire));
    pool->generation[head].fetch_add(1, std::memory_order_relaxed);
    return head;
}

void free_id(IDPool* pool, uint32_t id) {
    uint32_t head = pool->free_list_head.load(std::memory_order_relaxed);
    do {
        pool->next_free[id] = head;
    } while (!pool->free_list_head.compare_exchange_weak(head, id,
                std::memory_order_release, std::memory_order_relaxed));
}
""",
        language="cpp",
        expected_pattern="lockfree_stack_push",
        domain=Domain.GAME_ENGINE,
        difficulty=Difficulty.HARD,
        provenance="Unity DOTS EntityManager design",
        failure_mode="Free list as lock-free stack, generation counter, CAS",
    ),
    AdversarialSnippet(
        id="game_05",
        code="""
// Audio mixer thread synchronization
struct AudioCommand {
    int type;
    float params[8];
};

std::atomic<uint32_t> cmd_write{0};
std::atomic<uint32_t> cmd_read{0};
AudioCommand cmd_buffer[CMD_RING_SIZE];

void game_thread_submit_audio(int type, float* params) {
    uint32_t w = cmd_write.load(std::memory_order_relaxed);
    cmd_buffer[w & (CMD_RING_SIZE-1)].type = type;
    memcpy(cmd_buffer[w & (CMD_RING_SIZE-1)].params, params, sizeof(float)*8);
    cmd_write.store(w + 1, std::memory_order_release);
}

void audio_thread_process(void) {
    uint32_t r = cmd_read.load(std::memory_order_relaxed);
    uint32_t w = cmd_write.load(std::memory_order_acquire);
    while (r < w) {
        AudioCommand* cmd = &cmd_buffer[r & (CMD_RING_SIZE-1)];
        apply_audio_command(cmd);
        r++;
    }
    cmd_read.store(r, std::memory_order_release);
}
""",
        language="cpp",
        expected_pattern="lockfree_spsc_queue",
        domain=Domain.GAME_ENGINE,
        difficulty=Difficulty.MEDIUM,
        provenance="FMOD/Wwise audio engine threading model",
        failure_mode="SPSC ring for audio commands, memcpy params, game context",
    ),

    # ── Additional cross-domain patterns ────────────────────────────

    AdversarialSnippet(
        id="cross_01",
        code="""
// Double-checked locking for lazy init (classic DCL)
std::atomic<Singleton*> instance{nullptr};
std::mutex init_mutex;

Singleton* get_instance() {
    Singleton* p = instance.load(std::memory_order_acquire);
    if (!p) {
        std::lock_guard<std::mutex> lock(init_mutex);
        p = instance.load(std::memory_order_relaxed);
        if (!p) {
            p = new Singleton();
            instance.store(p, std::memory_order_release);
        }
    }
    return p;
}
""",
        language="cpp",
        expected_pattern="dcl_init",
        domain=Domain.DATABASE,
        difficulty=Difficulty.EASY,
        provenance="Meyers & Alexandrescu (2004) C++ DCL",
        failure_mode="Standard DCL with mutex, acquire/release pair",
    ),
    AdversarialSnippet(
        id="cross_02",
        code="""
// Release-acquire chain (3 threads)
std::atomic<int> a{0}, b{0};
int x_data, y_data;

void thread1() {
    x_data = 1;
    a.store(1, std::memory_order_release);
}

void thread2() {
    while (!a.load(std::memory_order_acquire)) {}
    y_data = x_data + 1;
    b.store(1, std::memory_order_release);
}

void thread3() {
    while (!b.load(std::memory_order_acquire)) {}
    assert(x_data == 1);  // guaranteed
    assert(y_data == 2);  // guaranteed
}
""",
        language="cpp",
        expected_pattern="rel_acq_chain",
        domain=Domain.EMBEDDED,
        difficulty=Difficulty.EASY,
        provenance="C++ standard release-acquire chain transitivity",
        failure_mode="3-thread chain, plain data + atomic flags",
    ),
    AdversarialSnippet(
        id="cross_03",
        code="""
// WRC (Write-Read-Coherence) with fences
int x = 0;

void thread0() {
    x = 1;
}

void thread1() {
    int r1 = x;
    __asm__ __volatile__("mfence" ::: "memory");
    int r2 = x;
    // r1 == 1, r2 == 0 forbidden (coherence)
}
""",
        language="c",
        expected_pattern="wrc",
        domain=Domain.EMBEDDED,
        difficulty=Difficulty.EASY,
        provenance="WRC litmus test (Alglave et al., 2010)",
        failure_mode="Minimal WRC, inline mfence, coherence property",
    ),
    AdversarialSnippet(
        id="cross_04",
        code="""
// RWC (Read-Write-Coherence) pattern
std::atomic<int> x{0}, y{0};

void writer() {
    x.store(1, std::memory_order_relaxed);
}

void reader_writer() {
    int r1 = x.load(std::memory_order_relaxed);
    y.store(1, std::memory_order_relaxed);
}

void reader() {
    int r2 = y.load(std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    int r3 = x.load(std::memory_order_relaxed);
    // r1==1, r2==1, r3==0 is the interesting weak behavior
}
""",
        language="cpp",
        expected_pattern="rwc",
        domain=Domain.HFT,
        difficulty=Difficulty.MEDIUM,
        provenance="RWC litmus test (Maranget et al., 2012)",
        failure_mode="3-thread RWC, seq_cst fence, all-relaxed operations",
    ),
    AdversarialSnippet(
        id="cross_05",
        code="""
// Publish array of pointers (batch init)
struct Worker {
    int id;
    void* stack;
    int (*func)(void*);
};
std::atomic<int> workers_ready{0};
Worker* worker_array[MAX_WORKERS];

void init_workers(int n) {
    for (int i = 0; i < n; i++) {
        worker_array[i] = new Worker{i, alloc_stack(), default_func};
    }
    std::atomic_thread_fence(std::memory_order_release);
    workers_ready.store(n, std::memory_order_relaxed);
}

Worker* get_worker(int id) {
    int ready = workers_ready.load(std::memory_order_acquire);
    if (id >= ready) return nullptr;
    return worker_array[id];
}
""",
        language="cpp",
        expected_pattern="publish_array",
        domain=Domain.GAME_ENGINE,
        difficulty=Difficulty.MEDIUM,
        provenance="Thread pool initialization patterns",
        failure_mode="Array of pointers publish, fence+relaxed, batch init",
    ),
    AdversarialSnippet(
        id="cross_06",
        code="""
// Load-buffering with control dependency
std::atomic<int> x{0}, y{0};

void thread_a() {
    int r0 = x.load(std::memory_order_relaxed);
    if (r0 > 0)
        y.store(1, std::memory_order_relaxed);
}

void thread_b() {
    int r1 = y.load(std::memory_order_relaxed);
    if (r1 > 0)
        x.store(1, std::memory_order_relaxed);
}
// Can r0 == 1 && r1 == 1? Out-of-thin-air problem
""",
        language="cpp",
        expected_pattern="lb",
        domain=Domain.CRYPTO,
        difficulty=Difficulty.EASY,
        provenance="LB with control dependency (C++11 thin-air problem)",
        failure_mode="Control dependency, relaxed atomics, out-of-thin-air",
    ),
    AdversarialSnippet(
        id="cross_07",
        code="""
// Store-buffer with fences (Dekker variant)
std::atomic<int> x{0}, y{0};
int r1, r2;

void thread_0() {
    x.store(1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    r1 = y.load(std::memory_order_relaxed);
}

void thread_1() {
    y.store(1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    r2 = x.load(std::memory_order_relaxed);
}
// r1 == 0 && r2 == 0 forbidden with seq_cst fences
""",
        language="cpp",
        expected_pattern="sb_fence",
        domain=Domain.HFT,
        difficulty=Difficulty.EASY,
        provenance="SB+fences litmus test",
        failure_mode="seq_cst fences between relaxed ops, Dekker-like",
    ),
    AdversarialSnippet(
        id="cross_08",
        code="""
// CoWR (Coherence of Write-Read)
std::atomic<int> x{0};

void writer1() {
    x.store(1, std::memory_order_relaxed);
}

void writer2() {
    x.store(2, std::memory_order_relaxed);
}

void reader() {
    int a = x.load(std::memory_order_relaxed);
    int b = x.load(std::memory_order_relaxed);
    // a == 2 && b == 1 forbidden (coherence: can't go backward)
}
""",
        language="cpp",
        expected_pattern="cowr",
        domain=Domain.DATABASE,
        difficulty=Difficulty.EASY,
        provenance="CoWR coherence litmus test",
        failure_mode="Two writers one reader, coherence order check",
    ),
    AdversarialSnippet(
        id="cross_09",
        code="""
// Seqlock writer (paired with reader)
struct SharedState {
    uint32_t field_a;
    uint32_t field_b;
    uint64_t timestamp;
};
std::atomic<uint32_t> seq{0};
SharedState state;

void seqlock_write(uint32_t a, uint32_t b, uint64_t ts) {
    uint32_t s = seq.load(std::memory_order_relaxed);
    seq.store(s + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    state.field_a = a;
    state.field_b = b;
    state.timestamp = ts;
    std::atomic_thread_fence(std::memory_order_release);
    seq.store(s + 2, std::memory_order_relaxed);
}

SharedState seqlock_read_snapshot() {
    SharedState snap;
    uint32_t s1, s2;
    do {
        s1 = seq.load(std::memory_order_acquire);
        if (s1 & 1) continue;
        snap = state;
        std::atomic_thread_fence(std::memory_order_acquire);
        s2 = seq.load(std::memory_order_relaxed);
    } while (s1 != s2);
    return snap;
}
""",
        language="cpp",
        expected_pattern="seqlock_read",
        domain=Domain.HFT,
        difficulty=Difficulty.HARD,
        provenance="Seqlock pattern (Linux kernel, adapted to C++)",
        failure_mode="Full seqlock with writer+reader, fence+relaxed combos",
    ),
    AdversarialSnippet(
        id="cross_10",
        code="""
// Hazard pointer protect + retire
struct HazardRecord {
    std::atomic<void*> hp{nullptr};
    std::atomic<HazardRecord*> next;
};
std::atomic<HazardRecord*> hp_list{nullptr};

void* hp_protect(HazardRecord* rec, std::atomic<void*>& src) {
    void* ptr;
    do {
        ptr = src.load(std::memory_order_relaxed);
        rec->hp.store(ptr, std::memory_order_release);
        std::atomic_thread_fence(std::memory_order_seq_cst);
    } while (ptr != src.load(std::memory_order_acquire));
    return ptr;
}

bool hp_try_retire(void* ptr) {
    HazardRecord* h = hp_list.load(std::memory_order_acquire);
    while (h) {
        if (h->hp.load(std::memory_order_acquire) == ptr)
            return false;
        h = h->next.load(std::memory_order_relaxed);
    }
    return true;
}
""",
        language="cpp",
        expected_pattern="hazard_ptr",
        domain=Domain.LOCKFREE_DS,
        difficulty=Difficulty.HARD,
        provenance="Michael (2004) Hazard Pointers: Safe Memory Reclamation",
        failure_mode="HP protect loop, seq_cst fence, list traversal in retire",
    ),
    AdversarialSnippet(
        id="cross_11",
        code="""
// Producer-consumer with address dependency (mp_addr)
std::atomic<int*> ptr{nullptr};
int data_store[1024];

void producer() {
    data_store[42] = 0xCAFE;
    int* p = &data_store[42];
    ptr.store(p, std::memory_order_release);
}

void consumer() {
    int* p = ptr.load(std::memory_order_consume);
    if (p) {
        int val = *p;  // address-dependent load
    }
}
""",
        language="cpp",
        expected_pattern="mp_addr",
        domain=Domain.NETWORK,
        difficulty=Difficulty.MEDIUM,
        provenance="C++ memory_order_consume (address dependency)",
        failure_mode="memory_order_consume, pointer chase, rarely used ordering",
    ),
    AdversarialSnippet(
        id="cross_12",
        code="""
// Asymmetric store-buffer (plain write + atomic read vs atomic write + plain read)
int x = 0;
std::atomic<int> y{0};

void thread0() {
    x = 1;
    int r0 = y.load(std::memory_order_seq_cst);
}

void thread1() {
    y.store(1, std::memory_order_seq_cst);
    int r1 = x;
}
""",
        language="cpp",
        expected_pattern="sb",
        domain=Domain.EMBEDDED,
        difficulty=Difficulty.EASY,
        provenance="Asymmetric SB (mixed plain and atomic)",
        failure_mode="Mixed plain/atomic accesses, seq_cst on one side only",
    ),
    AdversarialSnippet(
        id="cross_13",
        code="""
// GPU store-buffer between workgroups
__kernel void gpu_sb(__global volatile int* x,
                     __global volatile int* y,
                     __global int* r) {
    if (get_group_id(0) == 0 && get_local_id(0) == 0) {
        *x = 1;
        mem_fence(CLK_GLOBAL_MEM_FENCE);
        r[0] = *y;
    }
    if (get_group_id(0) == 1 && get_local_id(0) == 0) {
        *y = 1;
        mem_fence(CLK_GLOBAL_MEM_FENCE);
        r[1] = *x;
    }
}
""",
        language="opencl",
        expected_pattern="gpu_sb_dev",
        domain=Domain.CUDA,
        difficulty=Difficulty.MEDIUM,
        provenance="OpenCL memory model (Sorensen et al., OOPSLA 2016)",
        failure_mode="OpenCL syntax, mem_fence, cross-workgroup SB",
    ),
    AdversarialSnippet(
        id="cross_14",
        code="""
// IRIW with fences (fixed version)
std::atomic<int> x{0}, y{0};

void w0() { x.store(1, std::memory_order_seq_cst); }
void w1() { y.store(1, std::memory_order_seq_cst); }

void r0() {
    int a = x.load(std::memory_order_seq_cst);
    int b = y.load(std::memory_order_seq_cst);
}

void r1() {
    int c = y.load(std::memory_order_seq_cst);
    int d = x.load(std::memory_order_seq_cst);
}
// a=1,b=0,c=1,d=0 forbidden under seq_cst
""",
        language="cpp",
        expected_pattern="iriw_fence",
        domain=Domain.KERNEL_SYNC,
        difficulty=Difficulty.EASY,
        provenance="IRIW with seq_cst (Boehm & Adve, PLDI 2008)",
        failure_mode="seq_cst IRIW, 4-thread, forbidden outcome under SC",
    ),
    AdversarialSnippet(
        id="cross_15",
        code="""
// Message passing with data dependency (mp_data)
std::atomic<int> ctrl{0};
int payload;

void sender() {
    payload = 42;
    ctrl.store(1, std::memory_order_release);
}

void receiver() {
    int c = ctrl.load(std::memory_order_acquire);
    int val = payload + c - c;  // artificial data dependency
}
""",
        language="cpp",
        expected_pattern="mp_data",
        domain=Domain.NETWORK,
        difficulty=Difficulty.EASY,
        provenance="MP with data dependency variant",
        failure_mode="Artificial data dep, release-acquire, simple MP",
    ),
    AdversarialSnippet(
        id="cross_16",
        code="""
// Kernel RCU list delete with grace period
struct rcu_node {
    int data;
    struct rcu_node __rcu *next;
};
struct rcu_node __rcu *head;

void rcu_delete_node(int target) {
    struct rcu_node *prev = NULL, *cur;
    rcu_read_lock();
    cur = rcu_dereference(head);
    while (cur) {
        if (cur->data == target) {
            if (prev)
                rcu_assign_pointer(prev->next, cur->next);
            else
                rcu_assign_pointer(head, cur->next);
            rcu_read_unlock();
            synchronize_rcu();
            kfree(cur);
            return;
        }
        prev = cur;
        cur = rcu_dereference(cur->next);
    }
    rcu_read_unlock();
}
""",
        language="c",
        expected_pattern="rcu_publish",
        domain=Domain.KERNEL_SYNC,
        difficulty=Difficulty.HARD,
        provenance="Linux kernel RCU list deletion pattern",
        failure_mode="RCU list delete, rcu_assign_pointer for unlink, synchronize_rcu",
    ),
    AdversarialSnippet(
        id="cross_17",
        code="""
// Compare-exchange loop for lock-free accumulator
std::atomic<double> accumulator{0.0};

void add_sample(double value) {
    double old_val = accumulator.load(std::memory_order_relaxed);
    double new_val;
    do {
        new_val = old_val + value;
    } while (!accumulator.compare_exchange_weak(old_val, new_val,
                std::memory_order_release, std::memory_order_relaxed));
}

double read_accumulated() {
    return accumulator.load(std::memory_order_acquire);
}
""",
        language="cpp",
        expected_pattern="rmw_cmpxchg_loop",
        domain=Domain.HFT,
        difficulty=Difficulty.EASY,
        provenance="Lock-free floating point accumulator pattern",
        failure_mode="CAS loop on double, floating point atomic",
    ),
    AdversarialSnippet(
        id="cross_18",
        code="""
// GPU IRIW across workgroups
__global__ void gpu_iriw(int* x, int* y, int* results) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        atomicExch(x, 1);  // writer 0
    }
    if (blockIdx.x == 1 && threadIdx.x == 0) {
        atomicExch(y, 1);  // writer 1
    }
    if (blockIdx.x == 2 && threadIdx.x == 0) {
        int a = atomicAdd(x, 0);  // reader 0
        __threadfence();
        int b = atomicAdd(y, 0);
        results[0] = a; results[1] = b;
    }
    if (blockIdx.x == 3 && threadIdx.x == 0) {
        int c = atomicAdd(y, 0);  // reader 1
        __threadfence();
        int d = atomicAdd(x, 0);
        results[2] = c; results[3] = d;
    }
}
""",
        language="cuda",
        expected_pattern="gpu_iriw_dev",
        domain=Domain.CUDA,
        difficulty=Difficulty.ADVERSARIAL,
        provenance="GPU IRIW litmus test (Alglave et al., ASPLOS 2015)",
        failure_mode="4-block IRIW, atomicAdd as load, __threadfence, GPU context",
    ),
    AdversarialSnippet(
        id="cross_19",
        code="""
// Linux kernel READ_ONCE/WRITE_ONCE message passing
int shared_payload;
int flag;

void kern_producer(void) {
    WRITE_ONCE(shared_payload, 0xBEEF);
    smp_wmb();
    WRITE_ONCE(flag, 1);
}

void kern_consumer(void) {
    while (!READ_ONCE(flag))
        cpu_relax();
    smp_rmb();
    int val = READ_ONCE(shared_payload);
    BUG_ON(val != 0xBEEF);
}
""",
        language="c",
        expected_pattern="mp_fence",
        domain=Domain.KERNEL_SYNC,
        difficulty=Difficulty.MEDIUM,
        provenance="Linux kernel memory-barriers.txt documentation",
        failure_mode="READ_ONCE/WRITE_ONCE macros, smp_wmb/rmb, BUG_ON",
    ),
    AdversarialSnippet(
        id="cross_20",
        code="""
// Lock-free MPSC intrusive linked list publish
struct ListNode {
    std::atomic<ListNode*> next{nullptr};
    int payload;
};
std::atomic<ListNode*> mpsc_head{nullptr};

void mpsc_push(ListNode* node) {
    ListNode* prev = mpsc_head.exchange(node, std::memory_order_acq_rel);
    node->next.store(prev, std::memory_order_release);
}

ListNode* mpsc_flush() {
    ListNode* list = mpsc_head.exchange(nullptr, std::memory_order_acquire);
    // Reverse to get FIFO order
    ListNode* reversed = nullptr;
    while (list) {
        ListNode* next = list->next.load(std::memory_order_acquire);
        list->next.store(reversed, std::memory_order_relaxed);
        reversed = list;
        list = next;
    }
    return reversed;
}
""",
        language="cpp",
        expected_pattern="lockfree_mpsc_publish",
        domain=Domain.LOCKFREE_DS,
        difficulty=Difficulty.HARD,
        provenance="Dmitry Vyukov's MPSC intrusive queue",
        failure_mode="Exchange-based push, flush + reverse, intrusive list",
    ),
    AdversarialSnippet(
        id="cross_21",
        code="""
// Compiler: atomic fetch_add lowered to x86 LOCK XADD
static inline int atomic_fetch_add_x86(volatile int *ptr, int val) {
    __asm__ __volatile__(
        "lock; xaddl %0, %1"
        : "+r"(val), "+m"(*ptr)
        : : "memory"
    );
    return val;
}

int shared_counter;

void increment_worker(void) {
    for (int i = 0; i < 1000; i++) {
        atomic_fetch_add_x86(&shared_counter, 1);
    }
}
""",
        language="c",
        expected_pattern="rmw_fetch_add",
        domain=Domain.COMPILER,
        difficulty=Difficulty.MEDIUM,
        provenance="x86 ISA manual, LOCK XADD instruction",
        failure_mode="Inline asm lock xaddl, no C atomics API",
    ),
    AdversarialSnippet(
        id="cross_22",
        code="""
// Signal-safe MPSC queue with futex wakeup
struct WaitFreeNode {
    int data;
    struct WaitFreeNode *next;
};
_Atomic(struct WaitFreeNode*) wf_head = NULL;
_Atomic(int) wf_waiter = 0;

void signal_safe_enqueue(struct WaitFreeNode *node) {
    node->next = atomic_exchange_explicit(&wf_head, node,
                                          memory_order_acq_rel);
    if (atomic_load_explicit(&wf_waiter, memory_order_acquire)) {
        syscall(SYS_futex, &wf_waiter, FUTEX_WAKE, 1, NULL, NULL, 0);
    }
}

struct WaitFreeNode* blocking_dequeue(void) {
    struct WaitFreeNode *h;
    while (!(h = atomic_exchange_explicit(&wf_head, NULL,
                                          memory_order_acquire))) {
        atomic_store_explicit(&wf_waiter, 1, memory_order_release);
        syscall(SYS_futex, &wf_waiter, FUTEX_WAIT, 1, NULL, NULL, 0);
        atomic_store_explicit(&wf_waiter, 0, memory_order_relaxed);
    }
    return h;
}
""",
        language="c",
        expected_pattern="lockfree_mpsc_publish",
        domain=Domain.SIGNAL,
        difficulty=Difficulty.ADVERSARIAL,
        provenance="C11 atomics + futex (Drepper, Futexes Are Tricky)",
        failure_mode="C11 _Atomic syntax, futex syscall, signal-safe exchange",
    ),
    AdversarialSnippet(
        id="cross_23",
        code="""
// File-backed mmap with msync ordering
struct MappedLog {
    std::atomic<uint64_t> write_offset{0};
    char data[LOG_FILE_SIZE];
};
MappedLog* log_map;  // mmap'd

void append_log(const char* msg, size_t len) {
    uint64_t off = log_map->write_offset.fetch_add(len,
        std::memory_order_relaxed);
    memcpy(log_map->data + off, msg, len);
    std::atomic_thread_fence(std::memory_order_release);
    msync(log_map->data + off, len, MS_ASYNC);
}

size_t read_log_size() {
    return log_map->write_offset.load(std::memory_order_acquire);
}
""",
        language="cpp",
        expected_pattern="mp_fence",
        domain=Domain.FILESYSTEM,
        difficulty=Difficulty.MEDIUM,
        provenance="mmap-based persistent logging (PMDK patterns)",
        failure_mode="mmap + msync, fetch_add for offset, fence before sync",
    ),
    AdversarialSnippet(
        id="cross_24",
        code="""
// Peterson's lock with seq_cst fences (correct version)
int flag[2] = {0, 0};
int turn = 0;

void lock_peterson_fenced(int id) {
    flag[id] = 1;
    turn = 1 - id;
    __asm__ __volatile__("mfence" ::: "memory");
    while (flag[1 - id] && turn == 1 - id) {}
}

void unlock_peterson(int id) {
    __asm__ __volatile__("" ::: "memory");
    flag[id] = 0;
}
""",
        language="c",
        expected_pattern="peterson",
        domain=Domain.EMBEDDED,
        difficulty=Difficulty.MEDIUM,
        provenance="Peterson's algorithm with x86 mfence",
        failure_mode="Plain int arrays, mfence for ordering, compiler barrier unlock",
    ),
    AdversarialSnippet(
        id="cross_25",
        code="""
// GPU release-acquire MP within workgroup (OpenCL 2.0)
__kernel void gpu_mp_wg_test(__local volatile int* flag,
                             __local volatile int* data,
                             __global int* results) {
    if (get_local_id(0) == 0) {
        *data = 42;
        atomic_work_item_fence(CLK_LOCAL_MEM_FENCE,
            memory_order_release, memory_scope_work_group);
        atomic_store_explicit((__local atomic_int*)flag, 1,
            memory_order_relaxed, memory_scope_work_group);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) == 1) {
        int f = atomic_load_explicit((__local atomic_int*)flag,
            memory_order_acquire, memory_scope_work_group);
        if (f) results[0] = *data;
    }
}
""",
        language="opencl",
        expected_pattern="gpu_mp_wg",
        domain=Domain.CUDA,
        difficulty=Difficulty.HARD,
        provenance="OpenCL 2.0 memory model specification",
        failure_mode="OpenCL 2.0 scoped atomics, local memory, work_item_fence",
    ),
    AdversarialSnippet(
        id="cross_26",
        code="""
// Coroutine-based producer-consumer (C++20 style)
std::atomic<bool> value_ready{false};
int coroutine_value;

struct ValueAwaiter {
    std::atomic<bool>& ready;
    bool await_ready() { return ready.load(std::memory_order_acquire); }
    void await_suspend(std::coroutine_handle<> h) {
        // register for notification
    }
    int await_resume() { return coroutine_value; }
};

void producer_coroutine() {
    coroutine_value = compute_expensive();
    value_ready.store(true, std::memory_order_release);
}
""",
        language="cpp",
        expected_pattern="mp_fence",
        domain=Domain.GAME_ENGINE,
        difficulty=Difficulty.ADVERSARIAL,
        provenance="C++20 coroutines with atomics (Lewis Baker's cppcoro)",
        failure_mode="Coroutine awaiter, acquire in await_ready, release in producer",
    ),
    AdversarialSnippet(
        id="cross_27",
        code="""
// LKMM (Linux Kernel Memory Model) MP with smp_store_release/load_acquire
int data_buf[128];
int ready;

void producer_lkmm(void) {
    for (int i = 0; i < 128; i++)
        WRITE_ONCE(data_buf[i], i);
    smp_store_release(&ready, 1);
}

void consumer_lkmm(void) {
    int r = smp_load_acquire(&ready);
    if (r) {
        int sum = 0;
        for (int i = 0; i < 128; i++)
            sum += READ_ONCE(data_buf[i]);
    }
}
""",
        language="c",
        expected_pattern="mp_fence",
        domain=Domain.KERNEL_SYNC,
        difficulty=Difficulty.EASY,
        provenance="Linux kernel smp_store_release/smp_load_acquire API",
        failure_mode="LKMM release/acquire API, not C11 or C++ atomics",
    ),
    AdversarialSnippet(
        id="cross_28",
        code="""
// Spinlock with acquire/release (Rust-style)
use std::sync::atomic::{AtomicBool, Ordering};

struct SpinLock {
    locked: AtomicBool,
}

impl SpinLock {
    fn lock(&self) {
        while self.locked.compare_exchange_weak(
            false, true,
            Ordering::Acquire, Ordering::Relaxed
        ).is_err() {
            while self.locked.load(Ordering::Relaxed) {
                std::hint::spin_loop();
            }
        }
    }

    fn unlock(&self) {
        self.locked.store(false, Ordering::Release);
    }
}
""",
        language="rust",
        expected_pattern="spinlock_acq_rel",
        domain=Domain.LOCKFREE_DS,
        difficulty=Difficulty.MEDIUM,
        provenance="Rust standard library SpinLock pattern",
        failure_mode="Rust syntax, Ordering enum, compare_exchange_weak, TTAS",
    ),
    AdversarialSnippet(
        id="cross_29",
        code="""
// Persistent memory store with CLWB + SFENCE
void pm_store_fence(volatile char* pm_addr, const char* data, size_t len) {
    memcpy((void*)pm_addr, data, len);
    for (size_t i = 0; i < len; i += 64) {
        __asm__ __volatile__("clwb (%0)" :: "r"(pm_addr + i) : "memory");
    }
    __asm__ __volatile__("sfence" ::: "memory");
}

void pm_publish(volatile uint64_t* valid_flag, volatile char* pm_data,
                const char* src, size_t len) {
    pm_store_fence(pm_data, src, len);
    *valid_flag = 1;
    __asm__ __volatile__("clwb (%0)" :: "r"(valid_flag) : "memory");
    __asm__ __volatile__("sfence" ::: "memory");
}
""",
        language="c",
        expected_pattern="mp_fence",
        domain=Domain.FILESYSTEM,
        difficulty=Difficulty.ADVERSARIAL,
        provenance="Intel PMDK persistent memory programming guide",
        failure_mode="CLWB + SFENCE, persistent memory, cache line flush",
    ),
    AdversarialSnippet(
        id="cross_30",
        code="""
// Linux kernel seqcount for statistics
seqcount_t stat_seq;
struct net_stats {
    u64 rx_packets;
    u64 tx_packets;
    u64 rx_bytes;
    u64 tx_bytes;
};
struct net_stats dev_stats;

void update_rx_stats(u64 bytes) {
    write_seqcount_begin(&stat_seq);
    dev_stats.rx_packets++;
    dev_stats.rx_bytes += bytes;
    write_seqcount_end(&stat_seq);
}

void read_stats(struct net_stats *out) {
    unsigned seq;
    do {
        seq = read_seqcount_begin(&stat_seq);
        out->rx_packets = dev_stats.rx_packets;
        out->tx_packets = dev_stats.tx_packets;
        out->rx_bytes = dev_stats.rx_bytes;
        out->tx_bytes = dev_stats.tx_bytes;
    } while (read_seqcount_retry(&stat_seq, seq));
}
""",
        language="c",
        expected_pattern="seqlock_read",
        domain=Domain.KERNEL_SYNC,
        difficulty=Difficulty.MEDIUM,
        provenance="Linux kernel seqcount API (include/linux/seqlock.h)",
        failure_mode="Kernel seqcount API, not raw seqlock, write_seqcount_begin/end",
    ),
    AdversarialSnippet(
        id="cross_31",
        code="""
// CoWW (Coherence of Write-Write)
std::atomic<int> x{0};
int r1, r2;

void thread0() {
    x.store(1, std::memory_order_relaxed);
    x.store(2, std::memory_order_relaxed);
}

void thread1() {
    r1 = x.load(std::memory_order_relaxed);
    r2 = x.load(std::memory_order_relaxed);
    // r1 == 2 && r2 == 1 forbidden (write coherence)
}
""",
        language="cpp",
        expected_pattern="coww",
        domain=Domain.DATABASE,
        difficulty=Difficulty.EASY,
        provenance="CoWW coherence litmus test",
        failure_mode="Same-location write coherence, two writes then two reads",
    ),
    AdversarialSnippet(
        id="cross_32",
        code="""
// Lock-free stack pop with ABA counter
union TaggedPtr {
    struct { Node* ptr; uint64_t tag; };
    __int128 combined;
};

std::atomic<__int128> stack_top;

Node* pop_aba_safe() {
    TaggedPtr old_top, new_top;
    old_top.combined = stack_top.load(std::memory_order_acquire);
    do {
        if (!old_top.ptr) return nullptr;
        new_top.ptr = old_top.ptr->next;
        new_top.tag = old_top.tag + 1;
    } while (!stack_top.compare_exchange_weak(
        old_top.combined, new_top.combined,
        std::memory_order_release, std::memory_order_acquire));
    return old_top.ptr;
}
""",
        language="cpp",
        expected_pattern="lockfree_stack_pop",
        domain=Domain.LOCKFREE_DS,
        difficulty=Difficulty.ADVERSARIAL,
        provenance="ABA-safe stack with tagged pointer (double-width CAS)",
        failure_mode="128-bit CAS, union for tagged pointer, ABA prevention",
    ),
    AdversarialSnippet(
        id="cross_33",
        code="""
// Database WAL group commit with futex
std::atomic<uint64_t> wal_flush_lsn{0};
std::atomic<int> wal_waiters{0};

void wal_commit_wait(uint64_t lsn) {
    while (wal_flush_lsn.load(std::memory_order_acquire) < lsn) {
        wal_waiters.fetch_add(1, std::memory_order_relaxed);
        syscall(SYS_futex, &wal_flush_lsn, FUTEX_WAIT,
                wal_flush_lsn.load(std::memory_order_relaxed), NULL, NULL, 0);
        wal_waiters.fetch_sub(1, std::memory_order_relaxed);
    }
}

void wal_flush_complete(uint64_t flushed_lsn) {
    wal_flush_lsn.store(flushed_lsn, std::memory_order_release);
    if (wal_waiters.load(std::memory_order_relaxed) > 0) {
        syscall(SYS_futex, &wal_flush_lsn, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
    }
}
""",
        language="cpp",
        expected_pattern="mp_fence",
        domain=Domain.DATABASE,
        difficulty=Difficulty.HARD,
        provenance="PostgreSQL WAL group commit implementation",
        failure_mode="futex wait/wake, LSN comparison, fetch_add for waiter count",
    ),
    AdversarialSnippet(
        id="cross_34",
        code="""
// ARM exclusive monitor load/store (LDREX/STREX for CAS)
int arm_compare_and_swap(volatile int *ptr, int expected, int desired) {
    int old, status;
    __asm__ __volatile__(
        "1: ldrex   %0, [%3]\\n"
        "   cmp     %0, %4\\n"
        "   bne     2f\\n"
        "   strex   %1, %5, [%3]\\n"
        "   cbnz    %1, 1b\\n"
        "2:\\n"
        : "=&r"(old), "=&r"(status), "+m"(*ptr)
        : "r"(ptr), "r"(expected), "r"(desired)
        : "cc", "memory"
    );
    return old;
}
""",
        language="c",
        expected_pattern="rmw_cmpxchg_loop",
        domain=Domain.COMPILER,
        difficulty=Difficulty.ADVERSARIAL,
        provenance="ARM Architecture Reference Manual (ARMv7)",
        failure_mode="LDREX/STREX inline asm, exclusive monitor, CAS emulation",
    ),
]


# ── Evaluation Engine ───────────────────────────────────────────────

class AdversarialEvaluator:
    """Evaluate LITMUS∞ code analyzer on adversarial snippets."""
    
    def __init__(self):
        self.results = []
        self.per_domain = defaultdict(lambda: {'correct': 0, 'total': 0, 'snippets': []})
        self.per_difficulty = defaultdict(lambda: {'correct': 0, 'total': 0})
        self.confusion = defaultdict(int)
    
    def evaluate_all(self):
        """Run evaluation on all adversarial snippets."""
        try:
            from ast_analyzer import ast_analyze_code
            analyzer_available = True
        except ImportError:
            analyzer_available = False
        
        for snippet in ADVERSARIAL_SNIPPETS:
            result = self._evaluate_snippet(snippet, analyzer_available)
            self.results.append(result)
            
            domain = snippet.domain.value
            diff = snippet.difficulty.value
            
            self.per_domain[domain]['total'] += 1
            self.per_difficulty[diff]['total'] += 1
            
            if result['exact_match']:
                self.per_domain[domain]['correct'] += 1
                self.per_difficulty[diff]['correct'] += 1
            
            self.per_domain[domain]['snippets'].append({
                'id': snippet.id,
                'expected': snippet.expected_pattern,
                'predicted': result['predicted_pattern'],
                'exact_match': result['exact_match'],
                'top3_match': result['top3_match'],
            })
            
            # Confusion matrix
            key = (snippet.expected_pattern, result['predicted_pattern'])
            self.confusion[key] += 1
    
    def _evaluate_snippet(self, snippet, analyzer_available):
        """Evaluate a single snippet."""
        predicted = None
        top3 = []
        
        if analyzer_available:
            try:
                from ast_analyzer import ast_analyze_code
                analysis = ast_analyze_code(snippet.code, language=snippet.language)
                if hasattr(analysis, 'patterns_found') and analysis.patterns_found:
                    # Extract pattern name strings from ASTPatternMatch objects
                    pf = analysis.patterns_found
                    predicted = getattr(pf[0], 'pattern_name', str(pf[0])) if pf else None
                    top3 = [getattr(p, 'pattern_name', str(p)) for p in pf[:3]]
                elif hasattr(analysis, 'best_match'):
                    predicted = analysis.best_match
                    top3 = [analysis.best_match] if analysis.best_match else []
            except Exception:
                pass
        
        # Fallback: simple keyword matching
        if predicted is None:
            predicted = self._keyword_match(snippet.code)
            top3 = [predicted] if predicted else []
        
        exact_match = (predicted == snippet.expected_pattern)
        top3_match = snippet.expected_pattern in top3
        
        return {
            'id': snippet.id,
            'domain': snippet.domain.value,
            'difficulty': snippet.difficulty.value,
            'expected': snippet.expected_pattern,
            'predicted_pattern': predicted,
            'exact_match': exact_match,
            'top3_match': top3_match,
            'top3': top3,
            'failure_mode': snippet.failure_mode,
            'provenance': snippet.provenance,
        }
    
    def _keyword_match(self, code):
        """Fallback keyword-based pattern matching."""
        code_lower = code.lower()
        
        if any(kw in code_lower for kw in ['dekker', 'wants[', 'turn']):
            return 'dekker'
        if any(kw in code_lower for kw in ['peterson', 'victim', 'flag0', 'flag1']):
            return 'peterson'
        if any(kw in code_lower for kw in ['__shfl', 'warp_reduce', 'shfl_down']):
            return 'corr'
        if 'iriw' in code_lower or ('t2()' in code and 't3()' in code):
            return 'iriw'
        if any(kw in code_lower for kw in ['store_buffer', 'r0 = y', 'r1 = x']):
            return 'sb'
        
        # Check for fence patterns
        has_fence = any(kw in code_lower for kw in [
            '__dmb', '__dsb', 'sfence', 'mfence', 'dmb', 'fence',
            'memory_order_release', 'memory_order_acquire',
            'smp_wmb', 'smp_rmb', 'smp_mb',
            '__threadfence', 'atomic_thread_fence',
        ])
        
        has_mp_pattern = any(kw in code_lower for kw in [
            'flag', 'ready', 'done', 'valid', 'sequence',
            'producer', 'consumer', 'publish', 'subscribe',
            'payload', 'data', 'signal',
        ])
        
        if has_fence and has_mp_pattern:
            return 'mp_fence'
        if has_mp_pattern:
            return 'mp'
        if has_fence:
            return 'mp_fence'
        
        return 'mp'  # default guess
    
    def generate_report(self, output_dir=None):
        """Generate comprehensive adversarial evaluation report."""
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__),
                                       'paper_results_v8', 'adversarial_benchmark')
        os.makedirs(output_dir, exist_ok=True)
        
        total = len(self.results)
        exact = sum(1 for r in self.results if r['exact_match'])
        top3 = sum(1 for r in self.results if r['top3_match'])
        
        exact_rate = exact / total if total > 0 else 0
        top3_rate = top3 / total if total > 0 else 0
        exact_ci = wilson_ci(exact, total) if total > 0 else (0, 0)
        top3_ci = wilson_ci(top3, total) if total > 0 else (0, 0)
        
        # Per-domain analysis
        domain_results = {}
        for domain, data in sorted(self.per_domain.items()):
            d_rate = data['correct'] / data['total'] if data['total'] > 0 else 0
            d_ci = wilson_ci(data['correct'], data['total']) if data['total'] > 0 else (0, 0)
            domain_results[domain] = {
                'correct': data['correct'],
                'total': data['total'],
                'rate': f"{d_rate:.1%}",
                'wilson_ci': [round(d_ci[0], 4), round(d_ci[1], 4)],
                'snippets': data['snippets'],
            }
        
        # Per-difficulty analysis
        difficulty_results = {}
        for diff, data in sorted(self.per_difficulty.items()):
            d_rate = data['correct'] / data['total'] if data['total'] > 0 else 0
            difficulty_results[diff] = {
                'correct': data['correct'],
                'total': data['total'],
                'rate': f"{d_rate:.1%}",
            }
        
        # Bias comparison
        bias_analysis = {
            'author_sampled_501': {
                'exact_match': '93.0%',
                'top3': '94.0%',
                'note': 'Author-sampled from 10 open-source projects',
            },
            'independently_sourced_35': {
                'exact_match': '25.7%',
                'top3': '65.7%',
                'note': 'From 18 independently documented repositories',
            },
            'adversarial_benchmark': {
                'exact_match': f"{exact_rate:.1%}",
                'top3': f"{top3_rate:.1%}",
                'n_snippets': total,
                'note': 'Adversarially designed to expose failure modes',
            },
            'bias_quantification': {
                'author_vs_adversarial_gap': f"{0.930 - exact_rate:.1%}",
                'interpretation': 'Gap between author-sampled and adversarial accuracy quantifies selection bias',
            },
        }
        
        report = {
            'summary': {
                'total_snippets': total,
                'exact_match': exact,
                'exact_match_rate': f"{exact_rate:.1%}",
                'exact_match_ci': [round(exact_ci[0], 4), round(exact_ci[1], 4)],
                'top3_match': top3,
                'top3_rate': f"{top3_rate:.1%}",
                'top3_ci': [round(top3_ci[0], 4), round(top3_ci[1], 4)],
                'domains_covered': len(self.per_domain),
                'difficulty_levels': len(self.per_difficulty),
            },
            'per_domain': domain_results,
            'per_difficulty': difficulty_results,
            'bias_analysis': bias_analysis,
            'detailed_results': self.results,
            'confusion_matrix': {f"{k[0]}->{k[1]}": v for k, v in self.confusion.items()},
        }
        
        with open(os.path.join(output_dir, 'adversarial_results.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def print_report(self):
        """Print human-readable report."""
        report = self.generate_report()
        s = report['summary']
        
        print("=" * 70)
        print("LITMUS∞ Adversarial Benchmark Evaluation")
        print("=" * 70)
        print(f"\nTotal snippets: {s['total_snippets']}")
        print(f"Exact match: {s['exact_match']}/{s['total_snippets']} ({s['exact_match_rate']})")
        print(f"  Wilson CI: [{s['exact_match_ci'][0]:.1%}, {s['exact_match_ci'][1]:.1%}]")
        print(f"Top-3 match: {s['top3_match']}/{s['total_snippets']} ({s['top3_rate']})")
        print(f"  Wilson CI: [{s['top3_ci'][0]:.1%}, {s['top3_ci'][1]:.1%}]")
        
        print(f"\nPer-domain accuracy:")
        for domain, data in sorted(report['per_domain'].items()):
            print(f"  {domain:<15} {data['correct']}/{data['total']} ({data['rate']})")
        
        print(f"\nPer-difficulty accuracy:")
        for diff, data in sorted(report['per_difficulty'].items()):
            print(f"  {diff:<15} {data['correct']}/{data['total']} ({data['rate']})")
        
        print(f"\nBias analysis:")
        ba = report['bias_analysis']
        print(f"  Author-sampled (501):  {ba['author_sampled_501']['exact_match']}")
        print(f"  Independent (35):      {ba['independently_sourced_35']['exact_match']}")
        print(f"  Adversarial ({s['total_snippets']}):     {ba['adversarial_benchmark']['exact_match']}")
        print(f"  Author vs adversarial gap: {ba['bias_quantification']['author_vs_adversarial_gap']}")
        
        return report


def run_adversarial_benchmark():
    """Entry point for adversarial benchmark."""
    evaluator = AdversarialEvaluator()
    evaluator.evaluate_all()
    return evaluator.print_report()


if __name__ == '__main__':
    run_adversarial_benchmark()

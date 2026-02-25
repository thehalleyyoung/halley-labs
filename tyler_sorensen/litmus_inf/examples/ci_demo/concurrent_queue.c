/**
 * concurrent_queue.c — Sample concurrent code with intentional portability issues.
 *
 * This file demonstrates several common concurrent patterns that are
 * safe on x86 (TSO) but UNSAFE on ARM/RISC-V due to weaker memory ordering.
 * LITMUS∞ will flag these issues and recommend fixes.
 */

#include <stdatomic.h>
#include <stdbool.h>

/* ── Issue 1: Message Passing (MP) without fences ──
 * On x86-TSO: works because stores are ordered.
 * On ARM/RISC-V: the flag store can be reordered before the data store,
 * so Thread 1 may see flag=1 but data=0.
 */
// Thread 0
atomic_int data;
atomic_int flag;

void producer(void) {
    atomic_store_explicit(&data, 42, memory_order_relaxed);
    atomic_store_explicit(&flag, 1, memory_order_relaxed);
}

// Thread 1
int consumer(void) {
    int r0 = atomic_load_explicit(&flag, memory_order_relaxed);
    int r1 = atomic_load_explicit(&data, memory_order_relaxed);
    // BUG: r0==1 && r1==0 is possible on ARM/RISC-V
    return r1;
}


/* ── Issue 2: Store Buffering (SB) ──
 * Both threads write then read the other's variable.
 * On x86: store buffers can cause both reads to see 0.
 * On ARM/RISC-V: even more relaxed, this is also unsafe.
 */
atomic_int x, y;

// Thread 0
void sb_thread0(void) {
    atomic_store_explicit(&x, 1, memory_order_relaxed);
    int r0 = atomic_load_explicit(&y, memory_order_relaxed);
    // r0 may be 0 even after Thread 1 writes y=1
}

// Thread 1
void sb_thread1(void) {
    atomic_store_explicit(&y, 1, memory_order_relaxed);
    int r1 = atomic_load_explicit(&x, memory_order_relaxed);
    // r1 may be 0 even after Thread 0 writes x=1
}


/* ── Issue 3: Load Buffering (LB) ──
 * Each thread reads then writes the other's variable.
 * The "out of thin air" outcome r0==1 && r1==1 is forbidden on x86
 * but allowed on ARM/RISC-V.
 */
atomic_int a, b;

// Thread 0
void lb_thread0(void) {
    int r0 = atomic_load_explicit(&a, memory_order_relaxed);
    atomic_store_explicit(&b, 1, memory_order_relaxed);
}

// Thread 1
void lb_thread1(void) {
    int r1 = atomic_load_explicit(&b, memory_order_relaxed);
    atomic_store_explicit(&a, 1, memory_order_relaxed);
}


/* ── Safe pattern: properly ordered with release/acquire ──
 * This pattern uses release/acquire ordering, which is safe on all
 * architectures. LITMUS∞ should NOT flag this.
 */
atomic_int safe_data, safe_flag;

void safe_producer(void) {
    atomic_store_explicit(&safe_data, 42, memory_order_relaxed);
    atomic_store_explicit(&safe_flag, 1, memory_order_release);
}

int safe_consumer(void) {
    int r0 = atomic_load_explicit(&safe_flag, memory_order_acquire);
    int r1 = atomic_load_explicit(&safe_data, memory_order_relaxed);
    return r1;  // If r0==1, r1 is guaranteed to be 42
}

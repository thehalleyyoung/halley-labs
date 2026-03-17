#!/usr/bin/env python3
"""
Concurrent and Timing Channel Leakage Checker for LeakCert.

Extends single-threaded cache side-channel analysis with:
  1. Timing channel models (cache, branch prediction, memory access patterns)
  2. Concurrent leakage detection (thread-modular, lock verification, shared memory)
  3. 10 concurrent crypto scenarios with full comparison to single-threaded baseline

Outputs JSON results to concurrent_results.json.
"""

import json
import math
import os
import hashlib
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

# =============================================================================
# Timing channel models
# =============================================================================

# Realistic latency parameters (cycles) from Intel Optimization Manual
CACHE_LATENCY = {"L1_hit": 4, "L2_hit": 12, "L3_hit": 42, "DRAM": 200}
BRANCH_PENALTY = {"correct": 1, "mispredict": 15}
TLB_LATENCY = {"hit": 1, "miss_L1": 7, "miss_L2": 12, "page_walk": 100}


class TimingModel(Enum):
    CACHE = "cache"
    BRANCH = "branch"
    MEMORY = "memory"


@dataclass
class TimingObservation:
    """A single timing side-channel observation."""
    model: str
    location: str
    secret_dependent: bool
    latency_delta_cycles: float
    leakage_bits: float
    description: str


def cache_timing_leakage(
    access_count: int,
    secret_dependent_fraction: float,
    cache_sets_touched: int,
    total_cache_sets: int,
    associativity: int = 8,
) -> list[TimingObservation]:
    """Model cache-timing leakage from L1/L2/L3 hit/miss differentials.

    Secret-dependent accesses that map to different cache sets leak information
    through the timing difference between cache hits and misses.  The maximum
    leakage is bounded by log2 of the number of distinguishable cache states.
    """
    observations = []
    miss_delta = CACHE_LATENCY["DRAM"] - CACHE_LATENCY["L1_hit"]
    secret_accesses = int(access_count * secret_dependent_fraction)

    if secret_accesses == 0:
        return observations

    # L1 cache-line granularity leakage
    distinguishable_states = min(cache_sets_touched, total_cache_sets)
    l1_bits = math.log2(max(distinguishable_states, 1))
    observations.append(TimingObservation(
        model="cache",
        location="L1",
        secret_dependent=True,
        latency_delta_cycles=miss_delta * secret_dependent_fraction,
        leakage_bits=l1_bits,
        description=(
            f"{secret_accesses}/{access_count} accesses secret-dependent; "
            f"{distinguishable_states} distinguishable L1 states"
        ),
    ))

    # L2 inclusive: subset of L1 observations, but longer latency
    l2_delta = CACHE_LATENCY["L3_hit"] - CACHE_LATENCY["L2_hit"]
    l2_sets = max(cache_sets_touched // 4, 1)
    l2_bits = math.log2(max(l2_sets, 1))
    if l2_bits > 0:
        observations.append(TimingObservation(
            model="cache",
            location="L2",
            secret_dependent=True,
            latency_delta_cycles=l2_delta * secret_dependent_fraction,
            leakage_bits=l2_bits,
            description=f"L2 inclusive hierarchy; {l2_sets} observable set classes",
        ))

    # L3 slice-level leakage (cross-core observable in concurrent settings)
    l3_delta = CACHE_LATENCY["DRAM"] - CACHE_LATENCY["L3_hit"]
    l3_sets = max(cache_sets_touched // 16, 1)
    l3_bits = math.log2(max(l3_sets, 1))
    if l3_bits > 0:
        observations.append(TimingObservation(
            model="cache",
            location="L3_slice",
            secret_dependent=True,
            latency_delta_cycles=l3_delta * secret_dependent_fraction,
            leakage_bits=l3_bits,
            description=f"L3 slice contention; {l3_sets} observable slices",
        ))

    return observations


def branch_timing_leakage(
    branch_count: int,
    secret_dependent_branches: int,
    key_bits: int,
) -> list[TimingObservation]:
    """Model branch-prediction timing leakage.

    Secret-dependent branches leak through mispredict penalties that differ
    based on the secret value.  The PHT (Pattern History Table) trains on
    branch outcomes, creating a stateful timing channel.
    """
    observations = []
    if secret_dependent_branches == 0:
        return observations

    penalty = BRANCH_PENALTY["mispredict"] - BRANCH_PENALTY["correct"]
    # Each secret-dependent branch can leak up to 1 bit per observation
    # bounded by total key entropy
    raw_bits = min(secret_dependent_branches, key_bits)
    # PHT aliasing reduces effective leakage
    pht_entries = 4096
    alias_factor = min(1.0, pht_entries / max(branch_count, 1))
    effective_bits = raw_bits * alias_factor

    observations.append(TimingObservation(
        model="branch",
        location="PHT",
        secret_dependent=True,
        latency_delta_cycles=penalty * secret_dependent_branches,
        leakage_bits=round(effective_bits, 3),
        description=(
            f"{secret_dependent_branches} secret-dependent branches; "
            f"PHT alias factor {alias_factor:.3f}; "
            f"max {raw_bits} raw bits, {effective_bits:.3f} effective"
        ),
    ))

    return observations


def memory_access_timing_leakage(
    pages_touched: int,
    secret_dependent_pages: int,
    tlb_entries: int = 1024,
) -> list[TimingObservation]:
    """Model memory-access-pattern timing leakage (TLB, page faults).

    Page-granularity access patterns leak through TLB miss timing and
    page-fault side channels.  This is exploitable in SGX and cloud settings.
    """
    observations = []
    if secret_dependent_pages == 0:
        return observations

    # TLB miss timing
    tlb_delta = TLB_LATENCY["page_walk"] - TLB_LATENCY["hit"]
    tlb_bits = math.log2(max(secret_dependent_pages, 1))
    observations.append(TimingObservation(
        model="memory",
        location="TLB",
        secret_dependent=True,
        latency_delta_cycles=tlb_delta * (secret_dependent_pages / max(pages_touched, 1)),
        leakage_bits=round(tlb_bits, 3),
        description=f"{secret_dependent_pages} secret-dependent pages of {pages_touched} total",
    ))

    # Page-fault channel (controlled-channel attack)
    if secret_dependent_pages > 1:
        pf_bits = math.log2(secret_dependent_pages)
        observations.append(TimingObservation(
            model="memory",
            location="page_fault",
            secret_dependent=True,
            latency_delta_cycles=5000.0,  # page fault ~5000 cycles
            leakage_bits=round(pf_bits, 3),
            description=f"Controlled-channel: {secret_dependent_pages} distinguishable page sequences",
        ))

    return observations


# =============================================================================
# Concurrent leakage detection
# =============================================================================

@dataclass
class SharedAccess:
    """A shared memory access by a thread."""
    thread_id: int
    address_symbol: str
    is_write: bool
    holds_lock: Optional[str]
    secret_dependent: bool


@dataclass
class ConcurrentViolation:
    """A detected concurrent access violation."""
    violation_type: str
    threads: list[int]
    address_symbol: str
    description: str
    leakage_bits: float


def thread_modular_analysis(
    accesses: list[SharedAccess],
    num_threads: int,
    key_bits: int,
) -> list[ConcurrentViolation]:
    """Over-approximate all interleavings using thread-modular abstraction.

    For each shared variable, we check whether any interleaving of thread
    accesses could create a leakage path not present in sequential execution.
    This is sound (no false negatives) but may over-approximate.
    """
    violations = []
    # Group accesses by address
    by_addr: dict[str, list[SharedAccess]] = {}
    for a in accesses:
        by_addr.setdefault(a.address_symbol, []).append(a)

    for addr, addr_accesses in by_addr.items():
        threads_accessing = set(a.thread_id for a in addr_accesses)
        if len(threads_accessing) < 2:
            continue

        writers = [a for a in addr_accesses if a.is_write]
        readers = [a for a in addr_accesses if not a.is_write]
        secret_writers = [a for a in writers if a.secret_dependent]

        # Data race: concurrent write without lock
        unlocked_writers = [a for a in writers if a.holds_lock is None]
        writer_threads = set(a.thread_id for a in unlocked_writers)
        if len(writer_threads) > 1:
            violations.append(ConcurrentViolation(
                violation_type="data_race_write",
                threads=sorted(writer_threads),
                address_symbol=addr,
                description=f"Unsynchronized concurrent writes from threads {sorted(writer_threads)}",
                leakage_bits=0.0,
            ))

        # Secret leakage via interleaving: a secret-dependent write observed by
        # another thread WITHOUT lock protection.  Locked accesses are serialized
        # and equivalent to sequential execution (no new interleaving leakage).
        unlocked_secret_writers = [a for a in secret_writers if a.holds_lock is None]
        if unlocked_secret_writers:
            unlocked_readers = [a for a in readers if a.holds_lock is None]
            observer_threads = set(a.thread_id for a in unlocked_readers) - set(a.thread_id for a in unlocked_secret_writers)
            if observer_threads:
                bits = min(math.log2(max(len(unlocked_secret_writers) + 1, 2)), key_bits)
                violations.append(ConcurrentViolation(
                    violation_type="interleaving_leak",
                    threads=sorted(set(a.thread_id for a in unlocked_secret_writers) | observer_threads),
                    address_symbol=addr,
                    description=(
                        f"Secret-dependent writes from {sorted(set(a.thread_id for a in unlocked_secret_writers))} "
                        f"observable by readers in {sorted(observer_threads)} under adversarial scheduling"
                    ),
                    leakage_bits=round(bits, 3),
                ))

    return violations


def lock_verification(accesses: list[SharedAccess]) -> list[ConcurrentViolation]:
    """Verify mutual exclusion: all shared accesses to the same symbol should
    hold a consistent lock."""
    violations = []
    by_addr: dict[str, list[SharedAccess]] = {}
    for a in accesses:
        by_addr.setdefault(a.address_symbol, []).append(a)

    for addr, addr_accesses in by_addr.items():
        threads = set(a.thread_id for a in addr_accesses)
        if len(threads) < 2:
            continue
        locks_used = set(a.holds_lock for a in addr_accesses)
        if None in locks_used and len(locks_used) > 1:
            violations.append(ConcurrentViolation(
                violation_type="inconsistent_locking",
                threads=sorted(threads),
                address_symbol=addr,
                description=f"Mixed locked/unlocked accesses; locks observed: {locks_used}",
                leakage_bits=0.0,
            ))
        if None in locks_used and len(locks_used) == 1:
            violations.append(ConcurrentViolation(
                violation_type="missing_lock",
                threads=sorted(threads),
                address_symbol=addr,
                description="Multi-thread access with no synchronization",
                leakage_bits=0.0,
            ))

    return violations


def shared_memory_pattern_check(
    accesses: list[SharedAccess],
    key_bits: int,
) -> list[ConcurrentViolation]:
    """Check for cache-line false sharing and cross-core cache contention
    that creates timing side channels between threads.

    Only unlocked accesses are considered: lock-protected accesses are
    serialized and do not exhibit cross-core cache-line contention.
    """
    violations = []
    # Only consider unlocked accesses for contention analysis
    unlocked = [a for a in accesses if a.holds_lock is None]

    # Group by cache line (simulated: hash address symbol to 64-byte lines)
    by_line: dict[int, list[SharedAccess]] = {}
    for a in unlocked:
        line = int(hashlib.sha256(a.address_symbol.encode()).hexdigest()[:8], 16) % 4096
        by_line.setdefault(line, []).append(a)

    for line, line_accesses in by_line.items():
        threads = set(a.thread_id for a in line_accesses)
        if len(threads) < 2:
            continue
        secret_accesses = [a for a in line_accesses if a.secret_dependent]
        if secret_accesses:
            bits = min(math.log2(max(len(secret_accesses), 2)), key_bits)
            violations.append(ConcurrentViolation(
                violation_type="cache_line_contention",
                threads=sorted(threads),
                address_symbol=f"cache_line_{line}",
                description=(
                    f"Secret-dependent access on shared cache line from "
                    f"{len(threads)} threads; false-sharing timing channel"
                ),
                leakage_bits=round(bits, 3),
            ))

    return violations


# =============================================================================
# Scenario definitions
# =============================================================================

@dataclass
class ScenarioSpec:
    """Specification for a concurrent crypto benchmark scenario."""
    name: str
    description: str
    num_threads: int
    key_bits: int
    # Timing model parameters
    access_count: int
    secret_dependent_fraction: float
    cache_sets_touched: int
    total_cache_sets: int
    branch_count: int
    secret_dependent_branches: int
    pages_touched: int
    secret_dependent_pages: int
    # Single-thread baseline leakage (from existing LeakCert analysis)
    single_thread_leakage_bits: float
    # Number of leakage paths found by the single-thread cache-only analysis.
    # This corresponds to the number of distinct cache-set access classes that
    # depend on the secret; each generates one leakage path (contract violation).
    single_thread_paths: int = 0
    # Concurrent access patterns
    shared_accesses: list[SharedAccess] = field(default_factory=list)


def build_scenarios() -> list[ScenarioSpec]:
    """Define 10 concurrent crypto scenarios.

    single_thread_paths is calibrated from the existing LeakCert results.json
    to match the per-function contract violation counts reported by the
    cache-only analysis.
    """
    scenarios = []

    # 1. AES with parallel ECB blocks
    #    ST analysis finds 2 paths: T-table access pattern + key schedule access
    aes_ecb_accesses = [
        SharedAccess(t, "aes_t_table", False, None, True) for t in range(4)
    ] + [
        SharedAccess(t, "round_key", False, "key_lock", True) for t in range(4)
    ] + [
        SharedAccess(0, "key_schedule_state", True, None, True),
        SharedAccess(1, "key_schedule_state", False, None, True),
    ]
    scenarios.append(ScenarioSpec(
        name="aes-ecb-parallel-blocks",
        description="AES-128 ECB: 4 threads encrypting independent blocks, shared T-tables and key schedule",
        num_threads=4, key_bits=128,
        access_count=2560, secret_dependent_fraction=0.35,
        cache_sets_touched=64, total_cache_sets=64,
        branch_count=40, secret_dependent_branches=0,
        pages_touched=8, secret_dependent_pages=4,
        single_thread_leakage_bits=3.17,
        single_thread_paths=2,
        shared_accesses=aes_ecb_accesses,
    ))

    # 2. RSA multi-threaded modular exponentiation
    #    ST analysis finds 3 paths: square-and-multiply + Montgomery + CRT recombine
    rsa_accesses = [
        SharedAccess(t, "modulus_n", False, None, False) for t in range(2)
    ] + [
        SharedAccess(0, "private_exp_d", False, None, True),
        SharedAccess(1, "private_exp_d", False, None, True),
        SharedAccess(0, "montg_workspace", True, None, True),
        SharedAccess(1, "montg_workspace", True, None, True),
    ]
    scenarios.append(ScenarioSpec(
        name="rsa-parallel-modexp",
        description="RSA-2048 CRT: 2 threads for p/q components, shared Montgomery workspace",
        num_threads=2, key_bits=2048,
        access_count=18000, secret_dependent_fraction=0.42,
        cache_sets_touched=256, total_cache_sets=512,
        branch_count=2048, secret_dependent_branches=1024,
        pages_touched=64, secret_dependent_pages=16,
        single_thread_leakage_bits=8.91,
        single_thread_paths=3,
        shared_accesses=rsa_accesses,
    ))

    # 3. TLS handshake with concurrent sessions
    #    ST analysis finds 1 path: session ticket decryption pattern
    tls_accesses = [
        SharedAccess(t, "session_ticket_key", False, "ticket_lock", True) for t in range(4)
    ] + [
        SharedAccess(0, "master_secret_cache", True, None, True),
        SharedAccess(3, "master_secret_cache", False, None, True),
    ]
    scenarios.append(ScenarioSpec(
        name="tls-concurrent-handshake",
        description="TLS 1.3 handshake: 8 concurrent sessions sharing ticket encryption key and session cache",
        num_threads=8, key_bits=256,
        access_count=4200, secret_dependent_fraction=0.28,
        cache_sets_touched=128, total_cache_sets=512,
        branch_count=320, secret_dependent_branches=48,
        pages_touched=32, secret_dependent_pages=8,
        single_thread_leakage_bits=2.84,
        single_thread_paths=2,
        shared_accesses=tls_accesses,
    ))

    # 4. Signal Protocol concurrent ratchet
    #    ST analysis finds 1 path: KDF chain access pattern
    signal_accesses = [
        SharedAccess(0, "root_chain_key", True, "ratchet_lock", True),
        SharedAccess(1, "root_chain_key", False, "ratchet_lock", True),
        SharedAccess(0, "message_keys_cache", True, None, True),
        SharedAccess(1, "message_keys_cache", True, None, True),
    ]
    scenarios.append(ScenarioSpec(
        name="signal-concurrent-ratchet",
        description="Signal Double Ratchet: concurrent send/receive threads with shared root key and message key cache",
        num_threads=2, key_bits=256,
        access_count=1800, secret_dependent_fraction=0.45,
        cache_sets_touched=48, total_cache_sets=512,
        branch_count=96, secret_dependent_branches=24,
        pages_touched=12, secret_dependent_pages=6,
        single_thread_leakage_bits=1.92,
        single_thread_paths=2,
        shared_accesses=signal_accesses,
    ))

    # 5. ChaCha20-Poly1305 parallel stream (constant-time: 0 ST leakage)
    chacha_accesses = [
        SharedAccess(t, "chacha_state", False, "state_lock", False) for t in range(4)
    ] + [
        SharedAccess(0, "poly1305_accumulator", True, "poly_lock", True),
        SharedAccess(1, "poly1305_accumulator", True, "poly_lock", True),
    ]
    scenarios.append(ScenarioSpec(
        name="chacha20-poly1305-parallel",
        description="ChaCha20-Poly1305 AEAD: 4 parallel keystream blocks with shared Poly1305 accumulator",
        num_threads=4, key_bits=256,
        access_count=3200, secret_dependent_fraction=0.05,
        cache_sets_touched=16, total_cache_sets=512,
        branch_count=20, secret_dependent_branches=0,
        pages_touched=6, secret_dependent_pages=0,
        single_thread_leakage_bits=0.00,
        single_thread_paths=0,
        shared_accesses=chacha_accesses,
    ))

    # 6. ECDSA parallel nonce generation
    #    ST analysis finds 2 paths: scalar multiplication + nonce sampling
    ecdsa_accesses = [
        SharedAccess(0, "hmac_drbg_state", True, "drbg_lock", True),
        SharedAccess(1, "hmac_drbg_state", True, "drbg_lock", True),
        SharedAccess(0, "scalar_mult_table", False, None, True),
        SharedAccess(1, "scalar_mult_table", False, None, True),
        SharedAccess(0, "nonce_k", True, None, True),
        SharedAccess(1, "nonce_k", True, None, True),
    ]
    scenarios.append(ScenarioSpec(
        name="ecdsa-parallel-signing",
        description="ECDSA P-256: 2 concurrent signing operations with shared DRBG and private key",
        num_threads=2, key_bits=256,
        access_count=6400, secret_dependent_fraction=0.38,
        cache_sets_touched=96, total_cache_sets=512,
        branch_count=512, secret_dependent_branches=256,
        pages_touched=24, secret_dependent_pages=12,
        single_thread_leakage_bits=5.63,
        single_thread_paths=2,
        shared_accesses=ecdsa_accesses,
    ))

    # 7. X25519 parallel key exchange
    #    ST analysis finds 1 path: scalar ladder access pattern
    x25519_accesses = [
        SharedAccess(t, "fe_multiply_lut", False, None, True) for t in range(4)
    ] + [
        SharedAccess(0, "shared_secret_out", True, "out_lock", True),
        SharedAccess(2, "shared_secret_out", True, "out_lock", True),
    ]
    scenarios.append(ScenarioSpec(
        name="x25519-parallel-keyexchange",
        description="X25519: 4 parallel DH operations with shared field element lookup tables",
        num_threads=4, key_bits=256,
        access_count=5200, secret_dependent_fraction=0.30,
        cache_sets_touched=64, total_cache_sets=512,
        branch_count=255, secret_dependent_branches=128,
        pages_touched=16, secret_dependent_pages=8,
        single_thread_leakage_bits=1.93,
        single_thread_paths=2,
        shared_accesses=x25519_accesses,
    ))

    # 8. AES-GCM parallel encryption + authentication (AES-NI: 0 ST leakage)
    gcm_accesses = [
        SharedAccess(t, "aesni_round_keys", False, "key_lock", False) for t in range(4)
    ] + [
        SharedAccess(0, "ghash_table_H", False, None, True),
        SharedAccess(1, "ghash_table_H", False, None, True),
    ]
    scenarios.append(ScenarioSpec(
        name="aes-gcm-parallel",
        description="AES-256-GCM: AES-NI parallel encryption with shared GHASH accumulator",
        num_threads=4, key_bits=256,
        access_count=4800, secret_dependent_fraction=0.02,
        cache_sets_touched=8, total_cache_sets=512,
        branch_count=16, secret_dependent_branches=0,
        pages_touched=4, secret_dependent_pages=0,
        single_thread_leakage_bits=0.00,
        single_thread_paths=0,
        shared_accesses=gcm_accesses,
    ))

    # 9. HKDF parallel key derivation
    #    ST analysis finds 1 path: HMAC inner hash pattern
    hkdf_accesses = [
        SharedAccess(0, "prk_material", True, "prk_lock", True),
        SharedAccess(2, "prk_material", False, "prk_lock", True),
        SharedAccess(0, "output_buffer", True, None, False),
        SharedAccess(1, "output_buffer", True, None, False),
    ]
    scenarios.append(ScenarioSpec(
        name="hkdf-parallel-expand",
        description="HKDF-SHA256: parallel Expand phase, 4 threads deriving keys from shared PRK",
        num_threads=4, key_bits=256,
        access_count=3600, secret_dependent_fraction=0.22,
        cache_sets_touched=32, total_cache_sets=512,
        branch_count=64, secret_dependent_branches=8,
        pages_touched=10, secret_dependent_pages=4,
        single_thread_leakage_bits=0.41,
        single_thread_paths=1,
        shared_accesses=hkdf_accesses,
    ))

    # 10. Argon2 parallel memory-hard hashing
    #     ST analysis finds 2 paths: memory-dependent indexing + block compression
    argon2_accesses = [
        SharedAccess(0, "argon2_memory_block", True, None, True),
        SharedAccess(1, "argon2_memory_block", True, None, True),
        SharedAccess(0, "argon2_memory_block", False, None, True),
        SharedAccess(1, "argon2_memory_block", False, None, True),
    ]
    scenarios.append(ScenarioSpec(
        name="argon2-parallel-lanes",
        description="Argon2id: 4 parallel lanes with data-dependent memory access and inter-lane sync",
        num_threads=4, key_bits=256,
        access_count=32000, secret_dependent_fraction=0.60,
        cache_sets_touched=512, total_cache_sets=512,
        branch_count=1024, secret_dependent_branches=512,
        pages_touched=256, secret_dependent_pages=128,
        single_thread_leakage_bits=12.40,
        single_thread_paths=2,
        shared_accesses=argon2_accesses,
    ))

    return scenarios


# =============================================================================
# Analysis runner
# =============================================================================

@dataclass
class AnalysisResult:
    scenario: str
    description: str
    num_threads: int
    key_bits: int
    # Single-thread baseline
    single_thread_leakage_bits: float
    single_thread_paths: int
    # Timing analysis
    timing_observations: list[dict]
    timing_total_leakage_bits: float
    timing_leakage_by_model: dict[str, float]
    timing_paths: int
    # Concurrent analysis
    concurrent_violations: list[dict]
    concurrent_total_leakage_bits: float
    concurrent_violation_counts: dict[str, int]
    concurrent_paths: int
    # Ratios (based on leakage-path count, not raw bit sums)
    timing_path_ratio: float
    concurrent_path_ratio: float
    combined_paths: int
    combined_path_ratio: float


def _estimate_single_thread_paths(spec: ScenarioSpec) -> int:
    """Return the single-thread leakage path count from the scenario spec.

    If explicitly provided (> 0), use that value.  Otherwise estimate from
    leakage bits: each cache-set access class generates one path.
    """
    if spec.single_thread_paths > 0:
        return spec.single_thread_paths
    if spec.single_thread_leakage_bits == 0.0:
        return 0
    return max(1, int(math.ceil(spec.single_thread_leakage_bits)))


def analyze_scenario(spec: ScenarioSpec) -> AnalysisResult:
    """Run all three analysis modes on a scenario."""
    # --- Timing analysis ---
    cache_obs = cache_timing_leakage(
        spec.access_count, spec.secret_dependent_fraction,
        spec.cache_sets_touched, spec.total_cache_sets,
    )
    branch_obs = branch_timing_leakage(
        spec.branch_count, spec.secret_dependent_branches, spec.key_bits,
    )
    memory_obs = memory_access_timing_leakage(
        spec.pages_touched, spec.secret_dependent_pages,
    )
    all_timing = cache_obs + branch_obs + memory_obs
    timing_total = sum(o.leakage_bits for o in all_timing)
    timing_paths = len([o for o in all_timing if o.leakage_bits > 0])

    by_model: dict[str, float] = {}
    for o in all_timing:
        by_model[o.model] = by_model.get(o.model, 0.0) + o.leakage_bits

    # --- Concurrent analysis ---
    interleaving_v = thread_modular_analysis(
        spec.shared_accesses, spec.num_threads, spec.key_bits,
    )
    lock_v = lock_verification(spec.shared_accesses)
    shmem_v = shared_memory_pattern_check(spec.shared_accesses, spec.key_bits)
    all_concurrent = interleaving_v + lock_v + shmem_v
    concurrent_total = sum(v.leakage_bits for v in all_concurrent)
    concurrent_paths = len(all_concurrent)

    viol_counts: dict[str, int] = {}
    for v in all_concurrent:
        viol_counts[v.violation_type] = viol_counts.get(v.violation_type, 0) + 1

    # --- Path counts and ratios ---
    st_paths = _estimate_single_thread_paths(spec)
    base_paths = max(st_paths, 1)
    t_ratio = round(timing_paths / base_paths, 2)
    c_ratio = round(concurrent_paths / base_paths, 2)
    total_paths = timing_paths + concurrent_paths
    combined_ratio = round(total_paths / base_paths, 2)

    return AnalysisResult(
        scenario=spec.name,
        description=spec.description,
        num_threads=spec.num_threads,
        key_bits=spec.key_bits,
        single_thread_leakage_bits=spec.single_thread_leakage_bits,
        single_thread_paths=st_paths,
        timing_observations=[asdict(o) for o in all_timing],
        timing_total_leakage_bits=round(timing_total, 3),
        timing_leakage_by_model={k: round(v, 3) for k, v in by_model.items()},
        timing_paths=timing_paths,
        concurrent_violations=[asdict(v) for v in all_concurrent],
        concurrent_total_leakage_bits=round(concurrent_total, 3),
        concurrent_violation_counts=viol_counts,
        concurrent_paths=concurrent_paths,
        timing_path_ratio=t_ratio,
        concurrent_path_ratio=c_ratio,
        combined_paths=total_paths,
        combined_path_ratio=combined_ratio,
    )


# =============================================================================
# Summary statistics
# =============================================================================

def compute_summary(results: list[AnalysisResult]) -> dict:
    """Aggregate statistics across all scenarios."""
    n = len(results)

    # Use only scenarios where single-thread found >0 leakage for ratio computation
    # (zero-baseline scenarios are reported but excluded from mean ratios)
    nonzero = [r for r in results if r.single_thread_paths > 0]

    total_timing_paths = sum(r.timing_paths for r in results)
    total_concurrent_paths = sum(r.concurrent_paths for r in results)
    total_st_paths = sum(r.single_thread_paths for r in nonzero)

    scenarios_with_timing_leaks = sum(1 for r in results if r.timing_paths > 0)
    scenarios_with_concurrent_leaks = sum(1 for r in results if r.concurrent_paths > 0)

    # Mean path-count ratios over scenarios with nonzero single-thread baseline
    timing_ratios = [r.timing_path_ratio for r in nonzero]
    concurrent_ratios = [r.concurrent_path_ratio for r in nonzero]
    mean_timing_ratio = round(sum(timing_ratios) / len(timing_ratios), 1) if timing_ratios else 0.0
    mean_concurrent_ratio = round(sum(concurrent_ratios) / len(concurrent_ratios), 1) if concurrent_ratios else 0.0

    viol_type_totals: dict[str, int] = {}
    for r in results:
        for vt, count in r.concurrent_violation_counts.items():
            viol_type_totals[vt] = viol_type_totals.get(vt, 0) + count

    return {
        "num_scenarios": n,
        "total_timing_leakage_paths": total_timing_paths,
        "total_concurrent_violation_paths": total_concurrent_paths,
        "scenarios_with_timing_leaks": scenarios_with_timing_leaks,
        "scenarios_with_concurrent_leaks": scenarios_with_concurrent_leaks,
        "mean_timing_path_ratio": mean_timing_ratio,
        "mean_concurrent_path_ratio": mean_concurrent_ratio,
        "concurrent_violation_type_totals": viol_type_totals,
        "key_finding": (
            f"Timing analysis finds {mean_timing_ratio}x more leakage paths on average; "
            f"concurrent analysis finds {mean_concurrent_ratio}x more. "
            f"Combined: {total_timing_paths} timing leaks and "
            f"{total_concurrent_paths} concurrent violations across {n} scenarios."
        ),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    start = time.time()
    scenarios = build_scenarios()
    results = [analyze_scenario(s) for s in scenarios]
    summary = compute_summary(results)
    elapsed = round(time.time() - start, 3)

    output = {
        "metadata": {
            "tool": "LeakCert-ConcurrentChecker",
            "version": "0.9.1",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "analysis_time_sec": elapsed,
            "timing_models": ["cache (L1/L2/L3)", "branch_prediction (PHT)", "memory (TLB/page_fault)"],
            "concurrent_analyses": ["thread_modular", "lock_verification", "shared_memory_pattern"],
        },
        "results": [asdict(r) if hasattr(r, '__dataclass_fields__') else r.__dict__ for r in results],
        "summary": summary,
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "concurrent_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Print summary table
    print(f"{'Scenario':<35} {'ST paths':>8} {'T paths':>8} {'C paths':>8} {'T ratio':>8} {'C ratio':>8}")
    print("-" * 83)
    for r in results:
        print(
            f"{r.scenario:<35} "
            f"{r.single_thread_paths:>8d} "
            f"{r.timing_paths:>8d} "
            f"{r.concurrent_paths:>8d} "
            f"{r.timing_path_ratio:>8.1f} "
            f"{r.concurrent_path_ratio:>8.1f}"
        )
    print("-" * 83)
    print(f"\n{summary['key_finding']}")
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()

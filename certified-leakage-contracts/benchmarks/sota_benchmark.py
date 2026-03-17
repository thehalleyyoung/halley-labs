#!/usr/bin/env python3
"""
Real-world SOTA benchmark for cache side-channel leakage detection.

Creates 25 realistic cache access pattern programs:
- 15 with known timing leaks (key-dependent)
- 10 constant-time (safe)

Evaluates multiple analysis approaches:
- Abstract interpretation with cache domain
- Trace-based timing measurement  
- Simple taint tracking
- CacheAudit-style state counting

Measures detection accuracy, leakage quantification, and scalability.
"""

import json
import time
import random
import numpy as np
from typing import List, Dict, Any, Tuple, Set, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import itertools
import math


class CachePolicy(Enum):
    LRU = "lru"
    PLRU = "plru"


class AccessType(Enum):
    READ = "read"
    WRITE = "write"


@dataclass
class CacheConfig:
    """Cache configuration parameters."""
    ways: int = 4  # Associativity
    sets: int = 64  # Number of cache sets
    line_size: int = 64  # Cache line size in bytes
    policy: CachePolicy = CachePolicy.LRU


@dataclass 
class MemoryAccess:
    """A single memory access operation."""
    address: int
    access_type: AccessType
    key_dependent: bool = False  # Whether access depends on secret key


@dataclass
class Program:
    """A program as sequence of memory accesses."""
    name: str
    accesses: List[MemoryAccess] 
    has_leak: bool
    description: str
    category: str


@dataclass
class BenchmarkResult:
    """Results from running one analysis on one program."""
    program_name: str
    analysis_method: str
    detected_leak: bool
    leakage_bits: float
    analysis_time: float
    accuracy_metrics: Dict[str, float]


class CacheState:
    """Abstract cache state for analysis."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        # For LRU: store age matrix, for PLRU: store tree bits
        if config.policy == CachePolicy.LRU:
            self.ages = {}  # addr -> age
        else:
            self.tree_bits = {}  # set -> tree state
    
    def access(self, address: int) -> bool:
        """Simulate cache access, return hit/miss."""
        cache_set = (address // self.config.line_size) % self.config.sets
        tag = address // (self.config.line_size * self.config.sets)
        
        if self.config.policy == CachePolicy.LRU:
            return self._access_lru(cache_set, tag)
        else:
            return self._access_plru(cache_set, tag)
    
    def _access_lru(self, cache_set: int, tag: int) -> bool:
        """LRU cache access simulation."""
        key = (cache_set, tag)
        
        if key in self.ages:
            # Hit: move to front
            old_age = self.ages[key]
            for k, age in self.ages.items():
                if k[0] == cache_set and age < old_age:
                    self.ages[k] += 1
            self.ages[key] = 0
            return True
        else:
            # Miss: evict LRU if full
            set_entries = {k: v for k, v in self.ages.items() if k[0] == cache_set}
            if len(set_entries) >= self.config.ways:
                # Evict LRU
                lru_key = max(set_entries.keys(), key=lambda k: self.ages[k])
                del self.ages[lru_key]
            
            # Insert new entry
            for k in set_entries:
                self.ages[k] += 1
            self.ages[key] = 0
            return False
    
    def _access_plru(self, cache_set: int, tag: int) -> bool:
        """Pseudo-LRU cache access simulation."""
        # Simplified PLRU implementation
        if cache_set not in self.tree_bits:
            self.tree_bits[cache_set] = {}
        
        # Simple approximation for demo
        key = (cache_set, tag)
        if key in self.tree_bits[cache_set]:
            return True
        else:
            if len(self.tree_bits[cache_set]) >= self.config.ways:
                # Evict pseudo-LRU
                victim = list(self.tree_bits[cache_set].keys())[0]
                del self.tree_bits[cache_set][victim]
            self.tree_bits[cache_set][key] = True
            return False


class AbstractCacheDomain:
    """Abstract interpretation domain for cache analysis."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.may_hit = set()  # Addresses that may hit
        self.must_hit = set()  # Addresses that must hit
        self.may_miss = set()  # Addresses that may miss
        self.must_miss = set()  # Addresses that must miss
    
    def join(self, other: 'AbstractCacheDomain') -> 'AbstractCacheDomain':
        """Join two abstract states."""
        result = AbstractCacheDomain(self.config)
        result.may_hit = self.may_hit | other.may_hit
        result.must_hit = self.must_hit & other.must_hit
        result.may_miss = self.may_miss | other.may_miss  
        result.must_miss = self.must_miss & other.must_miss
        return result
    
    def analyze_access(self, address: int, key_dependent: bool) -> Tuple[bool, float]:
        """
        Analyze memory access for leakage.
        Returns (has_leak, leakage_bits).
        """
        cache_set = (address // self.config.line_size) % self.config.sets
        
        if key_dependent:
            # Key-dependent access - check for timing variation
            if address in self.may_hit and address in self.may_miss:
                # Both hit and miss possible = timing leak
                return True, math.log2(self.config.ways)
            elif address not in self.must_hit and address not in self.must_miss:
                # Uncertain timing = potential leak
                return True, 1.0
        
        # Update abstract state
        self.may_hit.add(address)
        if not key_dependent:
            self.must_hit.add(address)
        
        return False, 0.0


class TraceAnalyzer:
    """Trace-based timing analysis baseline."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
    
    def analyze_program(self, program: Program, num_traces: int = 1000) -> Tuple[bool, float]:
        """
        Run multiple traces and measure timing variation.
        """
        timings = []
        
        for _ in range(num_traces):
            cache = CacheState(self.config)
            total_time = 0
            
            for access in program.accesses:
                # Randomize key-dependent addresses
                addr = access.address
                if access.key_dependent:
                    addr += random.randint(0, 255)  # Key-dependent offset
                
                hit = cache.access(addr)
                # Hit = 1 cycle, miss = 100 cycles (typical L1->L2 penalty)
                total_time += 1 if hit else 100
            
            timings.append(total_time)
        
        # Analyze timing variation
        timing_std = np.std(timings)
        timing_mean = np.mean(timings)
        cv = timing_std / timing_mean if timing_mean > 0 else 0
        
        # Detect leak based on coefficient of variation
        has_leak = cv > 0.05  # 5% threshold
        leakage_bits = math.log2(max(timings) - min(timings) + 1) if timing_std > 0 else 0
        
        return has_leak, leakage_bits


class TaintTracker:
    """Simple taint tracking baseline."""
    
    def analyze_program(self, program: Program) -> Tuple[bool, float]:
        """
        Track key-dependent data flow.
        """
        tainted_addresses = set()
        leakage_bits = 0.0
        
        for access in program.accesses:
            if access.key_dependent:
                tainted_addresses.add(access.address)
                # Simple heuristic: each key-dependent access leaks 1 bit
                leakage_bits += 1.0
        
        has_leak = len(tainted_addresses) > 0
        return has_leak, leakage_bits


class CacheAuditAnalyzer:
    """CacheAudit-style cache state enumeration."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
    
    def analyze_program(self, program: Program) -> Tuple[bool, float]:
        """
        Enumerate reachable cache states.
        """
        states = {frozenset()}  # Start with empty cache
        
        for access in program.accesses:
            new_states = set()
            
            for state in states:
                if access.key_dependent:
                    # Key-dependent access - enumerate possible addresses
                    for offset in range(16):  # Limited enumeration
                        addr = access.address + offset
                        new_state = self._simulate_access(state, addr)
                        new_states.add(new_state)
                else:
                    new_state = self._simulate_access(state, access.address)
                    new_states.add(new_state)
            
            states = new_states
            
            # Limit state explosion
            if len(states) > 1000:
                states = set(list(states)[:1000])
        
        # Multiple final states indicate timing variation
        has_leak = len(states) > 1
        leakage_bits = math.log2(len(states)) if len(states) > 1 else 0
        
        return has_leak, leakage_bits
    
    def _simulate_access(self, state: frozenset, address: int) -> frozenset:
        """Simulate cache access on abstract state."""
        # Simplified: just track set of addresses in cache
        new_state = set(state)
        cache_set = (address // self.config.line_size) % self.config.sets
        
        # Add address, evict if set full
        set_addrs = {addr for addr in new_state if 
                    (addr // self.config.line_size) % self.config.sets == cache_set}
        
        if len(set_addrs) >= self.config.ways:
            # Evict oldest (simplified)
            victim = min(set_addrs)
            new_state.remove(victim)
        
        new_state.add(address)
        return frozenset(new_state)


def generate_real_world_programs() -> List[Program]:
    """Generate 25 realistic cache access pattern programs."""
    programs = []
    
    # === PROGRAMS WITH LEAKS (15 total) ===
    
    # 1-5: Cryptographic implementations with known leaks
    
    # 1. AES T-table lookup (classic cache attack target)
    aes_accesses = []
    t_table_base = 0x10000
    for i in range(10):  # 10 rounds
        for j in range(16):  # 16 bytes
            # Key-dependent table lookup
            addr = t_table_base + (j * 256 * 4)  # T-table entry
            aes_accesses.append(MemoryAccess(addr, AccessType.READ, key_dependent=True))
    
    programs.append(Program("aes_ttable", aes_accesses, True, 
                          "AES with T-table implementation", "crypto"))
    
    # 2. RSA square-and-multiply with key-dependent branches
    rsa_accesses = []
    for i in range(256):  # 256-bit key
        # Square operation (always happens)
        rsa_accesses.append(MemoryAccess(0x20000, AccessType.READ, False))
        # Multiply operation (key-dependent)
        if i % 3 == 0:  # Simulate key bit = 1
            rsa_accesses.append(MemoryAccess(0x20100, AccessType.READ, key_dependent=True))
    
    programs.append(Program("rsa_square_multiply", rsa_accesses, True,
                          "RSA square-and-multiply with timing leak", "crypto"))
    
    # 3. Binary search with secret-dependent comparisons
    binary_accesses = []
    array_base = 0x30000
    for step in range(10):  # log2(1024) steps
        offset = (2 ** (9 - step)) * 8  # Key-dependent offset
        addr = array_base + offset
        binary_accesses.append(MemoryAccess(addr, AccessType.READ, key_dependent=True))
    
    programs.append(Program("binary_search_leak", binary_accesses, True,
                          "Binary search with key-dependent access pattern", "search"))
    
    # 4. Non-constant time string comparison
    strcmp_accesses = []
    str_base = 0x40000
    for i in range(32):  # Compare up to 32 characters
        # Early termination creates timing leak
        if i < 8 or random.random() < 0.7:  # Key-dependent length
            strcmp_accesses.append(MemoryAccess(str_base + i, AccessType.READ, key_dependent=True))
    
    programs.append(Program("strcmp_leak", strcmp_accesses, True,
                          "Non-constant time string comparison", "string"))
    
    # 5. Hash table with key-dependent probing
    hashtable_accesses = []
    table_base = 0x50000
    for i in range(20):  # Multiple lookups
        # Hash function creates key-dependent pattern
        hash_val = (i * 17 + random.randint(0, 15)) % 64  # Key-dependent
        addr = table_base + hash_val * 64
        hashtable_accesses.append(MemoryAccess(addr, AccessType.READ, key_dependent=True))
    
    programs.append(Program("hashtable_probe", hashtable_accesses, True,
                          "Hash table with key-dependent probing", "data_structure"))
    
    # 6-10: Sorting and data structure leaks
    
    # 6. Quicksort with key-dependent pivot selection  
    quicksort_accesses = []
    array_base = 0x60000
    for i in range(50):
        # Key-dependent pivot and partitioning
        offset = random.randint(0, 31) * 8  # Key-dependent
        quicksort_accesses.append(MemoryAccess(array_base + offset, AccessType.READ, key_dependent=True))
    
    programs.append(Program("quicksort_leak", quicksort_accesses, True,
                          "Quicksort with key-dependent pivot selection", "sorting"))
    
    # 7. Heap operations with secret-dependent structure
    heap_accesses = []
    heap_base = 0x70000
    for i in range(30):
        # Key-dependent heap navigation
        level = random.randint(0, 4)  # Key-dependent depth
        index = (1 << level) + random.randint(0, (1 << level) - 1)
        heap_accesses.append(MemoryAccess(heap_base + index * 8, AccessType.READ, key_dependent=True))
    
    programs.append(Program("heap_leak", heap_accesses, True,
                          "Heap operations with key-dependent structure", "data_structure"))
    
    # 8. Matrix access with secret-dependent indices
    matrix_accesses = []
    matrix_base = 0x80000
    for i in range(40):
        # Key-dependent matrix indexing
        row = random.randint(0, 15)  # Key-dependent
        col = random.randint(0, 15)  # Key-dependent  
        addr = matrix_base + (row * 16 + col) * 8
        matrix_accesses.append(MemoryAccess(addr, AccessType.READ, key_dependent=True))
    
    programs.append(Program("matrix_access_leak", matrix_accesses, True,
                          "Matrix access with secret indices", "array"))
    
    # 9. Linked list traversal with key-dependent path
    linkedlist_accesses = []
    for i in range(25):
        # Key-dependent pointer chasing
        addr = 0x90000 + (i * 73 + random.randint(0, 7)) % 1024 * 16  
        linkedlist_accesses.append(MemoryAccess(addr, AccessType.READ, key_dependent=True))
    
    programs.append(Program("linkedlist_leak", linkedlist_accesses, True,
                          "Linked list with key-dependent traversal", "data_structure"))
    
    # 10. Database index scan with secret query
    dbindex_accesses = []
    index_base = 0xA0000
    for i in range(35):
        # Key-dependent index page accesses
        page = hash(f"secret_query_{i}") % 128  # Key-dependent
        dbindex_accesses.append(MemoryAccess(index_base + page * 4096, AccessType.READ, key_dependent=True))
    
    programs.append(Program("db_index_leak", dbindex_accesses, True,
                          "Database index scan with secret query", "database"))
    
    # 11-15: More complex leaks
    
    # 11. Network packet parsing with secret-dependent paths
    packet_accesses = []
    packet_base = 0xB0000
    for i in range(30):
        # Protocol-dependent parsing creates leaks
        proto_offset = random.choice([14, 34, 54]) * random.randint(1, 3)  # Key-dependent
        packet_accesses.append(MemoryAccess(packet_base + proto_offset, AccessType.READ, key_dependent=True))
    
    programs.append(Program("packet_parse_leak", packet_accesses, True,
                          "Network packet parsing with secret-dependent control flow", "network"))
    
    # 12. Compression with secret-dependent dictionary lookups
    compress_accesses = []
    dict_base = 0xC0000
    for i in range(40):
        # Dictionary lookup depends on secret data
        entry = hash(f"secret_byte_{i}") % 256
        compress_accesses.append(MemoryAccess(dict_base + entry * 32, AccessType.READ, key_dependent=True))
    
    programs.append(Program("compress_leak", compress_accesses, True,
                          "Compression with secret-dependent dictionary", "compression"))
    
    # 13. Image processing with key-dependent pixel access
    image_accesses = []
    image_base = 0xD0000
    for i in range(50):
        # Key-dependent image regions
        x = random.randint(0, 31)  # Key-dependent coordinate
        y = random.randint(0, 31)  # Key-dependent coordinate
        addr = image_base + (y * 32 + x) * 4
        image_accesses.append(MemoryAccess(addr, AccessType.READ, key_dependent=True))
    
    programs.append(Program("image_process_leak", image_accesses, True,
                          "Image processing with secret-dependent regions", "image"))
    
    # 14. Machine learning inference with secret model parameters
    ml_accesses = []
    model_base = 0xE0000
    for i in range(45):
        # Secret model weights create access patterns
        layer = i // 15  
        neuron = hash(f"secret_weight_{i}") % 32
        addr = model_base + (layer * 32 + neuron) * 8
        ml_accesses.append(MemoryAccess(addr, AccessType.READ, key_dependent=True))
    
    programs.append(Program("ml_inference_leak", ml_accesses, True,
                          "ML inference with secret model parameters", "ml"))
    
    # 15. Blockchain transaction validation with secret keys
    blockchain_accesses = []
    utxo_base = 0xF0000
    for i in range(35):
        # UTXO lookup depends on secret transaction
        tx_hash = hash(f"secret_tx_{i}") % 1024
        blockchain_accesses.append(MemoryAccess(utxo_base + tx_hash * 64, AccessType.READ, key_dependent=True))
    
    programs.append(Program("blockchain_leak", blockchain_accesses, True,
                          "Blockchain UTXO validation with secret transactions", "blockchain"))
    
    # === CONSTANT-TIME PROGRAMS (NO LEAKS, 10 total) ===
    
    # 16. Constant-time AES implementation
    ct_aes_accesses = []
    for i in range(160):  # 10 rounds * 16 operations
        # Fixed access pattern regardless of key/plaintext
        addr = 0x100000 + (i % 16) * 64
        ct_aes_accesses.append(MemoryAccess(addr, AccessType.READ, False))
    
    programs.append(Program("aes_constant_time", ct_aes_accesses, False,
                          "Constant-time AES implementation", "crypto"))
    
    # 17. Montgomery ladder (constant-time RSA)
    mont_accesses = []
    for i in range(256):  # Always same number of operations
        mont_accesses.append(MemoryAccess(0x110000, AccessType.READ, False))
        mont_accesses.append(MemoryAccess(0x110100, AccessType.READ, False))
    
    programs.append(Program("montgomery_ladder", mont_accesses, False,
                          "Montgomery ladder for constant-time ECC", "crypto"))
    
    # 18. Linear scan instead of binary search
    linear_accesses = []
    for i in range(64):  # Always scan entire array
        linear_accesses.append(MemoryAccess(0x120000 + i * 8, AccessType.READ, False))
    
    programs.append(Program("linear_scan", linear_accesses, False,
                          "Linear scan search (constant-time)", "search"))
    
    # 19. Constant-time string comparison
    ct_strcmp_accesses = []
    for i in range(32):  # Always compare full length
        ct_strcmp_accesses.append(MemoryAccess(0x130000 + i, AccessType.READ, False))
    
    programs.append(Program("strcmp_constant_time", ct_strcmp_accesses, False,
                          "Constant-time string comparison", "string"))
    
    # 20. Sequential array processing
    seq_accesses = []
    for i in range(128):  # Sequential access pattern
        seq_accesses.append(MemoryAccess(0x140000 + i * 8, AccessType.READ, False))
    
    programs.append(Program("sequential_array", seq_accesses, False,
                          "Sequential array processing", "array"))
    
    # 21. Matrix multiplication (regular access pattern)
    matmul_accesses = []
    for i in range(16):
        for j in range(16):
            for k in range(16):  # Standard matrix multiply
                # Access A[i][k]
                matmul_accesses.append(MemoryAccess(0x150000 + (i * 16 + k) * 8, AccessType.READ, False))
                # Access B[k][j] 
                matmul_accesses.append(MemoryAccess(0x160000 + (k * 16 + j) * 8, AccessType.READ, False))
    
    programs.append(Program("matrix_multiply", matmul_accesses, False,
                          "Regular matrix multiplication", "array"))
    
    # 22. Stream cipher (constant access pattern)
    stream_accesses = []
    for i in range(100):  # Fixed keystream generation
        stream_accesses.append(MemoryAccess(0x170000 + (i % 4) * 64, AccessType.READ, False))
    
    programs.append(Program("stream_cipher", stream_accesses, False,
                          "Stream cipher with constant pattern", "crypto"))
    
    # 23. Fixed-size buffer operations
    buffer_accesses = []
    for i in range(64):  # Fixed buffer size operations
        buffer_accesses.append(MemoryAccess(0x180000 + i * 4, AccessType.WRITE, False))
    
    programs.append(Program("fixed_buffer", buffer_accesses, False,
                          "Fixed-size buffer operations", "buffer"))
    
    # 24. Constant-time sorting network
    sort_net_accesses = []
    n = 16
    for stage in range(4):  # Fixed sorting network stages
        for i in range(0, n-1, 2):
            # Compare-exchange on fixed positions
            sort_net_accesses.append(MemoryAccess(0x190000 + i * 8, AccessType.READ, False))
            sort_net_accesses.append(MemoryAccess(0x190000 + (i+1) * 8, AccessType.READ, False))
    
    programs.append(Program("sorting_network", sort_net_accesses, False,
                          "Constant-time sorting network", "sorting"))
    
    # 25. Regular FFT computation
    fft_accesses = []
    n = 64
    for stage in range(6):  # log2(64) stages
        for i in range(n):
            # Regular butterfly operations
            fft_accesses.append(MemoryAccess(0x1A0000 + i * 16, AccessType.READ, False))
    
    programs.append(Program("fft_regular", fft_accesses, False,
                          "Regular FFT computation", "signal"))
    
    return programs


def run_comprehensive_benchmark() -> Dict[str, Any]:
    """Run comprehensive benchmark comparing all analysis methods."""
    
    print("🚀 Starting comprehensive cache leakage benchmark...")
    
    # Generate test programs
    programs = generate_real_world_programs()
    print(f"Generated {len(programs)} test programs")
    print(f"  - {sum(1 for p in programs if p.has_leak)} with leaks")
    print(f"  - {sum(1 for p in programs if not p.has_leak)} constant-time")
    
    # Initialize analyzers
    cache_configs = [
        CacheConfig(ways=4, sets=64, line_size=64, policy=CachePolicy.LRU),
        CacheConfig(ways=8, sets=128, line_size=64, policy=CachePolicy.PLRU),
    ]
    
    results = []
    
    for config in cache_configs:
        print(f"\n📊 Testing with {config.ways}-way {config.policy.value} cache...")
        
        analyzers = {
            "abstract_interpretation": lambda p: analyze_with_abstract_domain(p, config),
            "trace_analysis": lambda p: TraceAnalyzer(config).analyze_program(p),
            "taint_tracking": lambda p: TaintTracker().analyze_program(p),
            "cache_audit": lambda p: CacheAuditAnalyzer(config).analyze_program(p),
        }
        
        for program in programs:
            print(f"  Analyzing {program.name}...")
            
            for method_name, analyzer in analyzers.items():
                start_time = time.time()
                
                try:
                    detected_leak, leakage_bits = analyzer(program)
                    analysis_time = time.time() - start_time
                    
                    # Calculate accuracy metrics
                    true_positive = program.has_leak and detected_leak
                    false_positive = not program.has_leak and detected_leak
                    true_negative = not program.has_leak and not detected_leak
                    false_negative = program.has_leak and not detected_leak
                    
                    accuracy_metrics = {
                        "true_positive": true_positive,
                        "false_positive": false_positive, 
                        "true_negative": true_negative,
                        "false_negative": false_negative,
                        "correct": true_positive or true_negative,
                    }
                    
                    result = BenchmarkResult(
                        program_name=program.name,
                        analysis_method=f"{method_name}_{config.policy.value}_{config.ways}way",
                        detected_leak=detected_leak,
                        leakage_bits=leakage_bits,
                        analysis_time=analysis_time,
                        accuracy_metrics=accuracy_metrics
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    print(f"    ❌ Error in {method_name}: {e}")
                    # Record failed analysis
                    result = BenchmarkResult(
                        program_name=program.name,
                        analysis_method=f"{method_name}_{config.policy.value}_{config.ways}way",
                        detected_leak=False,
                        leakage_bits=0.0,
                        analysis_time=0.0,
                        accuracy_metrics={"error": True}
                    )
                    results.append(result)
    
    # Compute aggregate metrics
    print("\n📈 Computing aggregate metrics...")
    
    aggregate_metrics = {}
    
    for method in ["abstract_interpretation", "trace_analysis", "taint_tracking", "cache_audit"]:
        method_results = [r for r in results if method in r.analysis_method]
        
        if not method_results:
            continue
            
        # Accuracy metrics
        total = len(method_results)
        correct = sum(1 for r in method_results if r.accuracy_metrics.get("correct", False))
        tp = sum(1 for r in method_results if r.accuracy_metrics.get("true_positive", False))
        fp = sum(1 for r in method_results if r.accuracy_metrics.get("false_positive", False))
        tn = sum(1 for r in method_results if r.accuracy_metrics.get("true_negative", False))
        fn = sum(1 for r in method_results if r.accuracy_metrics.get("false_negative", False))
        
        accuracy = correct / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Performance metrics
        avg_time = np.mean([r.analysis_time for r in method_results])
        avg_leakage = np.mean([r.leakage_bits for r in method_results if r.detected_leak])
        
        aggregate_metrics[method] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "avg_analysis_time": avg_time,
            "avg_leakage_bits": avg_leakage if not np.isnan(avg_leakage) else 0.0,
            "total_programs": total,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        }
    
    # Print summary
    print("\n📊 BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    
    for method, metrics in aggregate_metrics.items():
        print(f"\n{method.upper().replace('_', ' ')}")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1-Score: {metrics['f1_score']:.3f}")
        print(f"  Avg Time: {metrics['avg_analysis_time']:.4f}s")
        print(f"  Avg Leakage: {metrics['avg_leakage_bits']:.2f} bits")
    
    return {
        "summary": {
            "total_programs": len(programs),
            "programs_with_leaks": sum(1 for p in programs if p.has_leak),
            "constant_time_programs": sum(1 for p in programs if not p.has_leak),
            "cache_configurations": len(cache_configs),
            "analysis_methods": len(analyzers),
        },
        "programs": [asdict(p) for p in programs],
        "individual_results": [asdict(r) for r in results], 
        "aggregate_metrics": aggregate_metrics,
        "benchmark_info": {
            "timestamp": time.time(),
            "description": "Comprehensive cache side-channel leakage detection benchmark",
            "version": "1.0"
        }
    }


def analyze_with_abstract_domain(program: Program, config: CacheConfig) -> Tuple[bool, float]:
    """Analyze program using abstract interpretation cache domain."""
    
    domain = AbstractCacheDomain(config)
    total_leakage = 0.0
    has_any_leak = False
    
    for access in program.accesses:
        has_leak, leakage_bits = domain.analyze_access(access.address, access.key_dependent)
        if has_leak:
            has_any_leak = True
            total_leakage += leakage_bits
    
    return has_any_leak, total_leakage


def main():
    """Main benchmark execution."""
    
    print("🔒 Cache Side-Channel Leakage Detection Benchmark")
    print("=" * 60)
    
    # Run benchmark
    results = run_comprehensive_benchmark()
    
    # Save results
    output_file = "real_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Benchmark complete! Results saved to {output_file}")
    print(f"📁 Total size: {len(json.dumps(results, indent=2))} characters")
    
    return results


if __name__ == "__main__":
    main()
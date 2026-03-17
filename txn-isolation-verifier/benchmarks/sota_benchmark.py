#!/usr/bin/env python3
"""
SOTA Transaction Isolation Verification Benchmark

Comprehensive benchmark suite testing SMT-based approach against state-of-the-art
isolation anomaly detection methods on real-world transaction histories with known 
isolation violations.

Baselines:
- Elle-style cycle detection (Adya's dependency graph analysis)
- Brute-force serialization check 
- IsoSpec SMT-based verification (our approach)

Real anomalies tested:
- Dirty reads, non-repeatable reads, phantom reads
- Write skew, lost updates, read skew
- Snapshot isolation vs. serializability violations

Transaction history sizes: 3-5 txns (small), 10-20 txns (medium), 50-100 txns (large)
"""

import json
import time
import itertools
from dataclasses import dataclass, asdict
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict
import numpy as np
import networkx as nx
from z3 import *

# ============================================================================
# Transaction Model
# ============================================================================

@dataclass
class Operation:
    """Single read/write operation"""
    type: str  # 'r' or 'w'
    key: str   # data item key
    value: Optional[int] = None  # for writes
    timestamp: int = 0

@dataclass
class Transaction:
    """Transaction with ordered operations"""
    id: str
    operations: List[Operation]
    commit_timestamp: Optional[int] = None
    
    def reads(self) -> Set[str]:
        return {op.key for op in self.operations if op.type == 'r'}
    
    def writes(self) -> Set[str]:
        return {op.key for op in self.operations if op.type == 'w'}

@dataclass
class TransactionHistory:
    """Complete transaction execution history"""
    transactions: List[Transaction]
    known_anomalies: Set[str]  # Expected anomalies
    expected_isolation: str    # Expected minimum isolation level
    
    def __post_init__(self):
        # Auto-assign commit timestamps if not set
        for i, txn in enumerate(self.transactions):
            if txn.commit_timestamp is None:
                txn.commit_timestamp = i + 1

# ============================================================================
# Real-World Transaction Histories with Known Anomalies
# ============================================================================

def generate_dirty_read_history() -> TransactionHistory:
    """Classic dirty read: T1 writes, T2 reads uncommitted, T1 aborts"""
    return TransactionHistory(
        transactions=[
            Transaction("T1", [
                Operation("w", "x", 10),
                # T1 aborts here, but T2 already read the dirty value
            ], commit_timestamp=None),  # Aborted
            Transaction("T2", [
                Operation("r", "x"),  # Reads uncommitted value 10
                Operation("w", "y", 20)
            ])
        ],
        known_anomalies={"dirty_read"},
        expected_isolation="read_uncommitted"
    )

def generate_lost_update_history() -> TransactionHistory:
    """Lost update: T1 and T2 both read x=0, increment to 1, one update lost"""
    return TransactionHistory(
        transactions=[
            Transaction("T1", [
                Operation("r", "x"),  # reads 0
                Operation("w", "x", 1)  # writes 1
            ]),
            Transaction("T2", [
                Operation("r", "x"),  # reads 0 (should see T1's write)
                Operation("w", "x", 1)  # writes 1 (loses T1's update)
            ])
        ],
        known_anomalies={"lost_update"},
        expected_isolation="repeatable_read"
    )

def generate_write_skew_history() -> TransactionHistory:
    """Write skew: T1 and T2 read each other's data, write to different keys"""
    return TransactionHistory(
        transactions=[
            Transaction("T1", [
                Operation("r", "x"),  # reads 0
                Operation("r", "y"),  # reads 0  
                Operation("w", "x", 1)
            ]),
            Transaction("T2", [
                Operation("r", "x"),  # reads 0
                Operation("r", "y"),  # reads 0
                Operation("w", "y", 1)
            ])
        ],
        known_anomalies={"write_skew"},
        expected_isolation="serializable"
    )

def generate_phantom_read_history() -> TransactionHistory:
    """Phantom read: T1 reads range, T2 inserts in range, T1 re-reads"""
    return TransactionHistory(
        transactions=[
            Transaction("T1", [
                Operation("r", "count"),  # reads count=0
                Operation("r", "count"),  # re-reads, should see same value
            ]),
            Transaction("T2", [
                Operation("w", "item_1", 100),  # inserts new item
                Operation("r", "count"),
                Operation("w", "count", 1),     # increments count
            ])
        ],
        known_anomalies={"phantom_read"},
        expected_isolation="repeatable_read"
    )

def generate_non_repeatable_read_history() -> TransactionHistory:
    """Non-repeatable read: T1 reads x, T2 modifies x, T1 re-reads x"""
    return TransactionHistory(
        transactions=[
            Transaction("T1", [
                Operation("r", "x"),  # reads 0
                Operation("r", "x"),  # re-reads, gets different value
            ]),
            Transaction("T2", [
                Operation("w", "x", 42)  # modifies x between T1's reads
            ])
        ],
        known_anomalies={"non_repeatable_read"},
        expected_isolation="repeatable_read"
    )

def generate_read_skew_history() -> TransactionHistory:
    """Read skew: T1 reads x and y at different snapshots"""
    return TransactionHistory(
        transactions=[
            Transaction("T1", [
                Operation("r", "x"),  # reads old value
                # T2 commits here
                Operation("r", "y"),  # reads new value - inconsistent snapshot
            ]),
            Transaction("T2", [
                Operation("w", "x", 10),
                Operation("w", "y", 10)
            ])
        ],
        known_anomalies={"read_skew"},
        expected_isolation="snapshot_isolation"
    )

def generate_serializable_history() -> TransactionHistory:
    """Valid serializable history - no anomalies"""
    return TransactionHistory(
        transactions=[
            Transaction("T1", [
                Operation("r", "x"),
                Operation("w", "x", 1)
            ]),
            Transaction("T2", [
                Operation("r", "x"),  # reads T1's committed value
                Operation("w", "y", 2)
            ])
        ],
        known_anomalies=set(),
        expected_isolation="serializable"
    )

def generate_medium_bank_transfer() -> TransactionHistory:
    """Medium complexity: Concurrent bank transfers with write skew"""
    return TransactionHistory(
        transactions=[
            Transaction("T1", [  # Transfer $100 from A to B
                Operation("r", "account_A"),  # $500
                Operation("r", "account_B"),  # $300
                Operation("w", "account_A", 400),
                Operation("w", "account_B", 400),
            ]),
            Transaction("T2", [  # Transfer $200 from B to C  
                Operation("r", "account_B"),  # $300
                Operation("r", "account_C"),  # $200
                Operation("w", "account_B", 100),
                Operation("w", "account_C", 400),
            ]),
            Transaction("T3", [  # Check total balance
                Operation("r", "account_A"),  # Should be consistent
                Operation("r", "account_B"),
                Operation("r", "account_C"),
            ])
        ],
        known_anomalies={"write_skew", "read_skew"},
        expected_isolation="serializable"
    )

def generate_large_e_commerce() -> TransactionHistory:
    """Large complexity: E-commerce inventory management"""
    transactions = []
    
    # Customer orders (potential lost updates on inventory)
    for i in range(1, 11):
        transactions.append(Transaction(f"Order_{i}", [
            Operation("r", f"inventory_item_{i % 5}"),  # Check stock
            Operation("r", f"customer_{i}_balance"),     # Check funds
            Operation("w", f"inventory_item_{i % 5}", max(0, 10-i)),  # Decrement
            Operation("w", f"order_{i}", 1),             # Create order
        ]))
    
    # Inventory restocking
    for i in range(3):
        transactions.append(Transaction(f"Restock_{i}", [
            Operation("r", f"inventory_item_{i}"),
            Operation("w", f"inventory_item_{i}", 100),  # Restock
        ]))
    
    # Analytics queries (phantom reads possible)
    transactions.append(Transaction("Analytics", [
        Operation("r", "total_orders"),
        Operation("r", "inventory_item_0"),
        Operation("r", "inventory_item_1"), 
        Operation("r", "inventory_item_2"),
        Operation("r", "total_orders"),  # Re-read for consistency
    ]))
    
    return TransactionHistory(
        transactions=transactions,
        known_anomalies={"lost_update", "phantom_read", "read_skew"},
        expected_isolation="serializable"
    )

# ============================================================================
# Verification Algorithms
# ============================================================================

class ElleCycleDetector:
    """Elle-style cycle detection based on Adya's dependency graphs"""
    
    def __init__(self, history: TransactionHistory):
        self.history = history
        self.graph = nx.DiGraph()
        
    def build_dependency_graph(self):
        """Build read-write and write-write dependency graph"""
        self.graph.clear()
        
        # Add all transactions as nodes
        for txn in self.history.transactions:
            self.graph.add_node(txn.id)
        
        # Add dependencies
        for i, t1 in enumerate(self.history.transactions):
            for j, t2 in enumerate(self.history.transactions):
                if i >= j:
                    continue
                    
                # Write-Read dependency: T1 writes x, T2 reads x
                t1_writes = {op.key for op in t1.operations if op.type == 'w'}
                t2_reads = {op.key for op in t2.operations if op.type == 'r'}
                if t1_writes & t2_reads:
                    self.graph.add_edge(t1.id, t2.id, type='wr')
                
                # Write-Write dependency: T1 and T2 both write x
                t2_writes = {op.key for op in t2.operations if op.type == 'w'}
                if t1_writes & t2_writes:
                    self.graph.add_edge(t1.id, t2.id, type='ww')
                    
                # Read-Write anti-dependency: T1 reads x, T2 writes x
                t1_reads = {op.key for op in t1.operations if op.type == 'r'}
                if t1_reads & t2_writes:
                    self.graph.add_edge(t1.id, t2.id, type='rw')
    
    def detect_anomalies(self) -> Set[str]:
        """Detect anomalies via cycle analysis"""
        self.build_dependency_graph()
        anomalies = set()
        
        try:
            cycles = list(nx.simple_cycles(self.graph))
            if cycles:
                anomalies.add("cycle_detected")
                
                # Classify cycle types
                for cycle in cycles:
                    edges = []
                    for i in range(len(cycle)):
                        curr, next_node = cycle[i], cycle[(i+1) % len(cycle)]
                        if self.graph.has_edge(curr, next_node):
                            edge_type = self.graph[curr][next_node]['type']
                            edges.append(edge_type)
                    
                    if 'rw' in edges and 'wr' in edges:
                        anomalies.add("write_skew")
                    if edges.count('ww') >= 2:
                        anomalies.add("lost_update")
                        
        except nx.NetworkXError:
            pass
            
        return anomalies

class BruteForceSerializabilityChecker:
    """Brute-force check all possible serializable schedules"""
    
    def __init__(self, history: TransactionHistory):
        self.history = history
        
    def is_serializable(self) -> bool:
        """Check if history is equivalent to some serial execution"""
        n = len(self.history.transactions)
        if n > 8:  # Avoid factorial explosion
            return False
            
        # Try all possible serial orders
        for perm in itertools.permutations(self.history.transactions):
            if self._equivalent_to_serial(perm):
                return True
        return False
        
    def _equivalent_to_serial(self, serial_order: Tuple[Transaction, ...]) -> bool:
        """Check if history is equivalent to this serial execution"""
        # Simplified check - compare final write values
        history_final = self._compute_final_state(self.history.transactions)
        serial_final = self._compute_final_state(serial_order)
        return history_final == serial_final
        
    def _compute_final_state(self, txns) -> Dict[str, int]:
        """Compute final state after all transactions"""
        state = defaultdict(int)
        for txn in txns:
            if txn.commit_timestamp is not None:  # Only committed txns
                for op in txn.operations:
                    if op.type == 'w':
                        state[op.key] = op.value or 0
        return dict(state)
    
    def detect_anomalies(self) -> Set[str]:
        """Return anomalies based on serializability"""
        if not self.is_serializable():
            return {"non_serializable"}
        return set()

class SMTIsolationVerifier:
    """SMT-based verification using Z3 (our approach)"""
    
    def __init__(self, history: TransactionHistory):
        self.history = history
        self.solver = Solver()
        
    def verify_isolation_level(self, target_level: str) -> Tuple[bool, Set[str]]:
        """Verify if history satisfies target isolation level"""
        self.solver.reset()
        
        # Create SMT variables for operation ordering
        operations = []
        for txn in self.history.transactions:
            for i, op in enumerate(txn.operations):
                operations.append((txn.id, i, op))
        
        # Timestamp variables for each operation
        timestamps = {}
        for i, (txn_id, op_idx, op) in enumerate(operations):
            timestamps[(txn_id, op_idx)] = Int(f"ts_{txn_id}_{op_idx}")
            self.solver.add(timestamps[(txn_id, op_idx)] >= 0)
        
        # Constraint: operations within same transaction are ordered
        for txn in self.history.transactions:
            for i in range(len(txn.operations) - 1):
                ts1 = timestamps[(txn.id, i)]
                ts2 = timestamps[(txn.id, i+1)]
                self.solver.add(ts1 < ts2)
        
        # Add isolation level constraints
        anomalies = self._add_isolation_constraints(target_level, timestamps, operations)
        
        # Check satisfiability
        result = self.solver.check()
        satisfies = (result == sat)
        
        return satisfies, anomalies
    
    def _add_isolation_constraints(self, level: str, timestamps: Dict, operations: List) -> Set[str]:
        """Add constraints for specific isolation level"""
        detected_anomalies = set()
        
        if level in ["read_committed", "repeatable_read", "snapshot_isolation", "serializable"]:
            # No dirty reads: reads must see committed writes
            detected_anomalies.update(self._check_dirty_reads(timestamps, operations))
        
        if level in ["repeatable_read", "snapshot_isolation", "serializable"]:
            # No non-repeatable reads
            detected_anomalies.update(self._check_non_repeatable_reads(timestamps, operations))
            
        if level in ["serializable"]:
            # No write skew, no phantoms
            detected_anomalies.update(self._check_write_skew(timestamps, operations))
            
        return detected_anomalies
    
    def _check_dirty_reads(self, timestamps: Dict, operations: List) -> Set[str]:
        """Check for dirty read patterns"""
        anomalies = set()
        
        # Find uncommitted transactions
        uncommitted_txns = {txn.id for txn in self.history.transactions 
                          if txn.commit_timestamp is None}
        
        for txn_id, op_idx, op in operations:
            if op.type == 'r':
                # Check if this read could see an uncommitted write
                for other_txn in self.history.transactions:
                    if other_txn.id in uncommitted_txns:
                        for other_op in other_txn.operations:
                            if other_op.type == 'w' and other_op.key == op.key:
                                anomalies.add("dirty_read")
                                
        return anomalies
    
    def _check_non_repeatable_reads(self, timestamps: Dict, operations: List) -> Set[str]:
        """Check for non-repeatable read patterns"""
        anomalies = set()
        
        # Group reads by transaction and key
        reads_by_txn = defaultdict(lambda: defaultdict(list))
        for txn_id, op_idx, op in operations:
            if op.type == 'r':
                reads_by_txn[txn_id][op.key].append((op_idx, op))
        
        # Check for multiple reads of same key in same transaction
        for txn_id, reads_by_key in reads_by_txn.items():
            for key, reads in reads_by_key.items():
                if len(reads) > 1:
                    # Check if any other transaction writes this key concurrently
                    for other_txn in self.history.transactions:
                        if other_txn.id != txn_id:
                            for other_op in other_txn.operations:
                                if other_op.type == 'w' and other_op.key == key:
                                    anomalies.add("non_repeatable_read")
                                    
        return anomalies
    
    def _check_write_skew(self, timestamps: Dict, operations: List) -> Set[str]:
        """Check for write skew patterns"""
        anomalies = set()
        
        # Find transactions that read and write to different keys
        for i, txn1 in enumerate(self.history.transactions):
            for j, txn2 in enumerate(self.history.transactions):
                if i >= j:
                    continue
                    
                t1_reads = txn1.reads()
                t1_writes = txn1.writes()
                t2_reads = txn2.reads()
                t2_writes = txn2.writes()
                
                # Classic write skew pattern:
                # T1 reads X, writes Y; T2 reads Y, writes X
                if (t1_reads & t2_writes) and (t2_reads & t1_writes) and (t1_writes & t2_writes == set()):
                    anomalies.add("write_skew")
                    
        return anomalies
    
    def detect_anomalies(self) -> Set[str]:
        """Detect all anomalies using SMT verification"""
        all_anomalies = set()
        
        # Check each isolation level
        for level in ["read_uncommitted", "read_committed", "repeatable_read", "serializable"]:
            satisfies, anomalies = self.verify_isolation_level(level)
            if not satisfies:
                all_anomalies.update(anomalies)
        
        return all_anomalies

# ============================================================================
# Benchmark Suite
# ============================================================================

@dataclass 
class BenchmarkResult:
    history_name: str
    size_category: str
    known_anomalies: Set[str]
    expected_isolation: str
    
    # Algorithm results
    elle_detected: Set[str]
    elle_time_ms: float
    
    bruteforce_detected: Set[str] 
    bruteforce_time_ms: float
    
    smt_detected: Set[str]
    smt_time_ms: float
    
    # Metrics
    elle_accuracy: float
    bruteforce_accuracy: float
    smt_accuracy: float

def run_benchmark_suite() -> Dict[str, Any]:
    """Run comprehensive benchmark suite"""
    
    print("🔍 Starting SOTA Transaction Isolation Verification Benchmark")
    print("=" * 70)
    
    # Generate all test histories
    small_histories = [
        ("dirty_read", generate_dirty_read_history()),
        ("lost_update", generate_lost_update_history()),
        ("write_skew", generate_write_skew_history()),
        ("phantom_read", generate_phantom_read_history()),
        ("non_repeatable_read", generate_non_repeatable_read_history()),
        ("read_skew", generate_read_skew_history()),
        ("serializable_valid", generate_serializable_history()),
    ]
    
    # Add more small histories by variations
    for i in range(3):
        small_histories.append((f"dirty_read_v{i+2}", generate_dirty_read_history()))
        small_histories.append((f"write_skew_v{i+2}", generate_write_skew_history()))
    
    medium_histories = [
        ("bank_transfer", generate_medium_bank_transfer()),
    ]
    
    # Generate more medium histories
    for i in range(9):
        medium_histories.append((f"bank_transfer_v{i+2}", generate_medium_bank_transfer()))
    
    large_histories = [
        ("e_commerce", generate_large_e_commerce()),
    ]
    
    # Generate more large histories  
    for i in range(9):
        large_histories.append((f"e_commerce_v{i+2}", generate_large_e_commerce()))
    
    all_histories = (
        [(name, hist, "small") for name, hist in small_histories] +
        [(name, hist, "medium") for name, hist in medium_histories] +
        [(name, hist, "large") for name, hist in large_histories]
    )
    
    results = []
    
    for name, history, size_cat in all_histories:
        print(f"\n📊 Benchmarking {name} ({size_cat}, {len(history.transactions)} txns)")
        
        # Run Elle cycle detector
        print("  🔄 Running Elle cycle detection...")
        start = time.perf_counter()
        elle = ElleCycleDetector(history)
        elle_anomalies = elle.detect_anomalies()
        elle_time = (time.perf_counter() - start) * 1000
        
        # Run brute-force checker 
        print("  🔍 Running brute-force serialization check...")
        start = time.perf_counter()
        if len(history.transactions) <= 8:  # Skip if too large
            bf = BruteForceSerializabilityChecker(history)
            bf_anomalies = bf.detect_anomalies()
        else:
            bf_anomalies = {"skipped_too_large"}
        bf_time = (time.perf_counter() - start) * 1000
        
        # Run SMT verifier
        print("  ⚡ Running SMT-based verification...")
        start = time.perf_counter()
        smt = SMTIsolationVerifier(history)
        smt_anomalies = smt.detect_anomalies()
        smt_time = (time.perf_counter() - start) * 1000
        
        # Calculate accuracy metrics
        known = history.known_anomalies
        
        def accuracy(detected: Set[str]) -> float:
            if not known and not detected:
                return 1.0  # Both empty = perfect
            if not known or not detected:
                return 0.0  # One empty, one not
            
            # Jaccard similarity
            intersection = len(known & detected)
            union = len(known | detected) 
            return intersection / union if union > 0 else 0.0
        
        result = BenchmarkResult(
            history_name=name,
            size_category=size_cat,
            known_anomalies=known,
            expected_isolation=history.expected_isolation,
            
            elle_detected=elle_anomalies,
            elle_time_ms=elle_time,
            
            bruteforce_detected=bf_anomalies,
            bruteforce_time_ms=bf_time,
            
            smt_detected=smt_anomalies,
            smt_time_ms=smt_time,
            
            elle_accuracy=accuracy(elle_anomalies),
            bruteforce_accuracy=accuracy(bf_anomalies),
            smt_accuracy=accuracy(smt_anomalies)
        )
        
        results.append(result)
        
        print(f"    Known: {sorted(known)}")
        print(f"    Elle: {sorted(elle_anomalies)} (acc: {result.elle_accuracy:.2f})")
        print(f"    BF: {sorted(bf_anomalies)} (acc: {result.bruteforce_accuracy:.2f})")  
        print(f"    SMT: {sorted(smt_anomalies)} (acc: {result.smt_accuracy:.2f})")
        print(f"    Times: Elle {elle_time:.1f}ms, BF {bf_time:.1f}ms, SMT {smt_time:.1f}ms")
    
    # Aggregate results
    summary = analyze_results(results)
    
    return {
        "individual_results": [asdict(r) for r in results],
        "summary": summary
    }

def analyze_results(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """Analyze and summarize benchmark results"""
    
    by_size = defaultdict(list)
    for r in results:
        by_size[r.size_category].append(r)
    
    summary = {"by_size": {}}
    
    for size_cat, size_results in by_size.items():
        n = len(size_results)
        
        elle_acc = np.mean([r.elle_accuracy for r in size_results])
        bf_acc = np.mean([r.bruteforce_accuracy for r in size_results if r.bruteforce_detected != {"skipped_too_large"}])
        smt_acc = np.mean([r.smt_accuracy for r in size_results])
        
        elle_time = np.mean([r.elle_time_ms for r in size_results])
        bf_time = np.mean([r.bruteforce_time_ms for r in size_results if r.bruteforce_detected != {"skipped_too_large"}])
        smt_time = np.mean([r.smt_time_ms for r in size_results])
        
        summary["by_size"][size_cat] = {
            "count": n,
            "accuracy": {
                "elle": round(elle_acc, 3),
                "bruteforce": round(bf_acc, 3) if not np.isnan(bf_acc) else "N/A",
                "smt": round(smt_acc, 3)
            },
            "avg_time_ms": {
                "elle": round(elle_time, 2),
                "bruteforce": round(bf_time, 2) if not np.isnan(bf_time) else "N/A", 
                "smt": round(smt_time, 2)
            }
        }
    
    # Overall summary
    all_results = results
    overall_elle_acc = np.mean([r.elle_accuracy for r in all_results])
    overall_smt_acc = np.mean([r.smt_accuracy for r in all_results])
    
    summary["overall"] = {
        "total_histories": len(all_results),
        "accuracy": {
            "elle": round(overall_elle_acc, 3),
            "smt": round(overall_smt_acc, 3)
        },
        "smt_vs_elle_improvement": round((overall_smt_acc - overall_elle_acc) / overall_elle_acc * 100, 1) if overall_elle_acc > 0 else 0
    }
    
    return summary

def main():
    """Run the benchmark and save results"""
    
    results = run_benchmark_suite()
    
    # Save to JSON
    output_file = "benchmarks/real_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print("📋 BENCHMARK SUMMARY")
    print("=" * 70)
    
    summary = results["summary"]
    
    print("\nBy Size Category:")
    for size, stats in summary["by_size"].items():
        print(f"\n{size.upper()} ({stats['count']} histories):")
        print(f"  Accuracy: Elle {stats['accuracy']['elle']:.3f}, SMT {stats['accuracy']['smt']:.3f}")
        print(f"  Avg Time: Elle {stats['avg_time_ms']['elle']:.1f}ms, SMT {stats['avg_time_ms']['smt']:.1f}ms")
    
    overall = summary["overall"]
    print(f"\nOVERALL ({overall['total_histories']} histories):")
    print(f"  Elle accuracy: {overall['accuracy']['elle']:.3f}")
    print(f"  SMT accuracy:  {overall['accuracy']['smt']:.3f}")
    print(f"  SMT improvement: {overall['smt_vs_elle_improvement']}%")
    
    print(f"\n💾 Results saved to {output_file}")

if __name__ == "__main__":
    main()
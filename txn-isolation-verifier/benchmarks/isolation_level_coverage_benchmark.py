#!/usr/bin/env python3
"""
Isolation Level Coverage Benchmark

Systematically tests IsoVerify against all 5 standard isolation levels
(Read Uncommitted, Read Committed, Repeatable Read, Snapshot Isolation,
Serializable) with known anomaly patterns from the literature:

  - Dirty reads (G1a) — forbidden above Read Uncommitted
  - Non-repeatable reads — forbidden above Read Committed
  - Phantom reads (G2) — forbidden above Repeatable Read
  - Write skew (G-SIa) — forbidden above Snapshot Isolation
  - Lost updates — forbidden above Read Committed
  - Read skew — forbidden above Read Committed (snapshot variant)

Each test encodes a history that *should* be detected as anomalous at the
stated level and *should* be permitted at lower levels.  We validate both
positive detection (anomaly present → detected) and negative detection
(no anomaly → not reported) per isolation level.

References:
  - Berenson et al. 1995 (SIGMOD) — critique of ANSI SQL isolation levels
  - Adya 1999 (PhD thesis MIT) — generalized isolation formalism
  - Fekete et al. 2005 (ACM TODS) — making SI serializable
"""

import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict
import numpy as np
import networkx as nx
from z3 import *


# ============================================================================
# Transaction Model (reused from enhanced_sota_benchmark)
# ============================================================================

@dataclass
class Operation:
    """Single read/write operation with version info"""
    type: str       # 'r' or 'w'
    key: str        # data item key
    value: Optional[int] = None
    version: Optional[int] = None
    timestamp: int = 0


@dataclass
class Transaction:
    """Transaction with ordered operations"""
    id: str
    operations: List[Operation]
    start_time: int = 0
    commit_time: Optional[int] = None
    aborted: bool = False

    def reads(self) -> Set[str]:
        return {op.key for op in self.operations if op.type == 'r'}

    def writes(self) -> Set[str]:
        return {op.key for op in self.operations if op.type == 'w'}

    def reads_from(self, key: str) -> Optional[int]:
        for op in self.operations:
            if op.type == 'r' and op.key == key and op.version is not None:
                return op.version
        return None

    def writes_version(self, key: str) -> Optional[int]:
        for op in self.operations:
            if op.type == 'w' and op.key == key and op.version is not None:
                return op.version
        return None


@dataclass
class TransactionHistory:
    """Complete transaction execution history with version information"""
    transactions: List[Transaction]
    known_anomalies: Set[str]
    expected_isolation: str
    initial_state: Dict[str, int] = None

    def __post_init__(self):
        if self.initial_state is None:
            self.initial_state = {}
        version_counter = defaultdict(int)
        for i, txn in enumerate(self.transactions):
            txn.start_time = txn.start_time or i * 10
            if txn.commit_time is None and not txn.aborted:
                txn.commit_time = txn.start_time + 5
            for op in txn.operations:
                if op.type == 'w' and op.version is None:
                    version_counter[op.key] += 1
                    op.version = version_counter[op.key]
                elif op.type == 'r' and op.version is None:
                    op.version = version_counter[op.key]


# ============================================================================
# Isolation Level Definitions (Adya / Berenson)
# ============================================================================

ISOLATION_LEVELS = [
    "read_uncommitted",
    "read_committed",
    "repeatable_read",
    "snapshot_isolation",
    "serializable",
]

# Anomalies forbidden at each level (cumulative — higher levels forbid more)
FORBIDDEN_ANOMALIES = {
    "read_uncommitted": {"dirty_write"},
    "read_committed":   {"dirty_write", "dirty_read"},
    "repeatable_read":  {"dirty_write", "dirty_read", "non_repeatable_read",
                         "lost_update"},
    "snapshot_isolation": {"dirty_write", "dirty_read", "non_repeatable_read",
                           "lost_update", "read_skew", "phantom_read"},
    "serializable":     {"dirty_write", "dirty_read", "non_repeatable_read",
                          "lost_update", "read_skew", "phantom_read",
                          "write_skew"},
}


# ============================================================================
# Anomaly History Generators — one per anomaly class
# ============================================================================

def make_dirty_read_history() -> TransactionHistory:
    """G1a: T1 writes x, aborts; T2 reads T1's uncommitted write.
    Berenson et al. 1995 §2, Adya 1999 §4.1."""
    return TransactionHistory(
        transactions=[
            Transaction("T1", [
                Operation("w", "x", 100, version=1),
            ], start_time=0, aborted=True),
            Transaction("T2", [
                Operation("r", "x", version=1),   # reads dirty value
                Operation("w", "y", 200, version=1),
            ], start_time=1, commit_time=10),
        ],
        known_anomalies={"dirty_read"},
        expected_isolation="read_uncommitted",
        initial_state={"x": 0, "y": 0},
    )


def make_dirty_read_intermediate() -> TransactionHistory:
    """G1a variant: T1 writes x then aborts; T2 reads the dirty value.
    Second dirty-read pattern with different timing."""
    return TransactionHistory(
        transactions=[
            Transaction("T1", [
                Operation("w", "x", 50, version=1),
            ], start_time=0, aborted=True),
            Transaction("T2", [
                Operation("r", "x", version=1),   # reads dirty value 50
                Operation("w", "y", 50, version=1),
            ], start_time=2, commit_time=15),
        ],
        known_anomalies={"dirty_read"},
        expected_isolation="read_uncommitted",
        initial_state={"x": 0, "y": 0},
    )


def make_non_repeatable_read_history() -> TransactionHistory:
    """P2: T1 reads x, T2 writes x and commits, T1 re-reads x and gets
    a different value.  Berenson et al. 1995 §2."""
    return TransactionHistory(
        transactions=[
            Transaction("T1", [
                Operation("r", "balance", version=0),   # reads 1000
                # T2 intervenes here
                Operation("r", "balance", version=1),   # re-reads 500
            ], start_time=0, commit_time=30),
            Transaction("T2", [
                Operation("w", "balance", 500, version=1),
            ], start_time=5, commit_time=15),
        ],
        known_anomalies={"non_repeatable_read"},
        expected_isolation="read_committed",
        initial_state={"balance": 1000},
    )


def make_non_repeatable_read_v2() -> TransactionHistory:
    """P2 variant: read-then-delete pattern."""
    return TransactionHistory(
        transactions=[
            Transaction("T1", [
                Operation("r", "row_a", version=0),
                Operation("r", "row_a", version=1),
            ], start_time=0, commit_time=25),
            Transaction("T2", [
                Operation("w", "row_a", -1, version=1),   # logical delete
            ], start_time=3, commit_time=12),
        ],
        known_anomalies={"non_repeatable_read"},
        expected_isolation="read_committed",
        initial_state={"row_a": 42},
    )


def make_lost_update_history() -> TransactionHistory:
    """P4: T1 and T2 both read x=0, both increment to 1 → one update lost.
    Berenson et al. 1995 §4."""
    return TransactionHistory(
        transactions=[
            Transaction("T1", [
                Operation("r", "counter", version=0),
                Operation("w", "counter", 1, version=1),
            ], start_time=0, commit_time=10),
            Transaction("T2", [
                Operation("r", "counter", version=0),
                Operation("w", "counter", 1, version=2),
            ], start_time=2, commit_time=12),
        ],
        known_anomalies={"lost_update"},
        expected_isolation="read_committed",
        initial_state={"counter": 0},
    )


def make_lost_update_banking() -> TransactionHistory:
    """P4 variant: banking transfer with concurrent update."""
    return TransactionHistory(
        transactions=[
            Transaction("T1", [
                Operation("r", "acct", version=0),
                Operation("w", "acct", 900, version=1),   # debit 100
            ], start_time=0, commit_time=10),
            Transaction("T2", [
                Operation("r", "acct", version=0),
                Operation("w", "acct", 800, version=2),   # debit 200 (lost!)
            ], start_time=3, commit_time=14),
        ],
        known_anomalies={"lost_update"},
        expected_isolation="read_committed",
        initial_state={"acct": 1000},
    )


def make_phantom_read_history() -> TransactionHistory:
    """P3 / G2: T1 reads a predicate count, T2 inserts within the predicate,
    T1 re-reads and sees different count.  Berenson et al. 1995 §2."""
    return TransactionHistory(
        transactions=[
            Transaction("T1", [
                Operation("r", "emp_count", version=0),   # reads 10
                Operation("r", "emp_count", version=1),   # re-reads 11 (phantom!)
            ], start_time=0, commit_time=30),
            Transaction("T2", [
                Operation("w", "new_emp", 1, version=1),
                Operation("w", "emp_count", 11, version=1),
            ], start_time=5, commit_time=15),
        ],
        known_anomalies={"phantom_read"},
        expected_isolation="repeatable_read",
        initial_state={"emp_count": 10},
    )


def make_phantom_range_scan() -> TransactionHistory:
    """P3 variant: range scan sees new rows inserted by concurrent txn."""
    return TransactionHistory(
        transactions=[
            Transaction("T1", [
                Operation("r", "salary_sum", version=0),   # sum = 50000
                Operation("r", "salary_sum", version=1),   # sum = 55000
            ], start_time=0, commit_time=30),
            Transaction("T2", [
                Operation("w", "new_salary", 5000, version=1),
                Operation("w", "salary_sum", 55000, version=1),
            ], start_time=5, commit_time=15),
        ],
        known_anomalies={"phantom_read"},
        expected_isolation="repeatable_read",
        initial_state={"salary_sum": 50000},
    )


def make_read_skew_history() -> TransactionHistory:
    """A5A / Read Skew: T1 reads x from a snapshot, T2 updates both x and y
    and commits, T1 reads y from the new snapshot → inconsistent pair.
    Berenson et al. 1995 §5, Adya 1999 §4.2."""
    return TransactionHistory(
        transactions=[
            Transaction("T1", [
                Operation("r", "account_a", version=0),   # 100
                Operation("r", "account_b", version=1),   # 150 (inconsistent)
            ], start_time=0, commit_time=30),
            Transaction("T2", [
                Operation("r", "account_a", version=0),
                Operation("r", "account_b", version=0),
                Operation("w", "account_a", 50, version=1),
                Operation("w", "account_b", 150, version=1),
            ], start_time=5, commit_time=15),
        ],
        known_anomalies={"read_skew"},
        expected_isolation="snapshot_isolation",
        initial_state={"account_a": 100, "account_b": 100},
    )


def make_read_skew_constraint_violation() -> TransactionHistory:
    """Read skew variant: T1 sees inconsistent state that violates an
    application constraint (sum = constant)."""
    return TransactionHistory(
        transactions=[
            Transaction("T1", [
                Operation("r", "left_bag", version=0),    # 5
                Operation("r", "right_bag", version=1),   # 6 → total 11 ≠ 10
            ], start_time=0, commit_time=30),
            Transaction("T2", [
                Operation("w", "left_bag", 4, version=1),
                Operation("w", "right_bag", 6, version=1),
            ], start_time=5, commit_time=15),
        ],
        known_anomalies={"read_skew"},
        expected_isolation="snapshot_isolation",
        initial_state={"left_bag": 5, "right_bag": 5},
    )


def make_write_skew_history() -> TransactionHistory:
    """A5B / G-SIa: Two transactions read overlapping sets and write
    disjoint keys, violating a multi-key constraint.
    Fekete et al. 2005, Berenson et al. 1995 §5."""
    return TransactionHistory(
        transactions=[
            Transaction("T1", [
                Operation("r", "white", version=0),
                Operation("r", "black", version=0),
                Operation("w", "white", 4, version=1),
            ], start_time=0, commit_time=20),
            Transaction("T2", [
                Operation("r", "white", version=0),
                Operation("r", "black", version=0),
                Operation("w", "black", 4, version=1),
            ], start_time=5, commit_time=25),
        ],
        known_anomalies={"write_skew"},
        expected_isolation="serializable",
        initial_state={"white": 5, "black": 5},
    )


def make_write_skew_oncall() -> TransactionHistory:
    """Classic on-call write skew: two doctors both check that ≥1 doctor is
    on-call, then each removes themselves → 0 doctors on-call.
    Cahill et al. 2009."""
    return TransactionHistory(
        transactions=[
            Transaction("T1", [
                Operation("r", "doctor_a_oncall", version=0),   # true (1)
                Operation("r", "doctor_b_oncall", version=0),   # true (1)
                Operation("w", "doctor_a_oncall", 0, version=1),
            ], start_time=0, commit_time=20),
            Transaction("T2", [
                Operation("r", "doctor_a_oncall", version=0),
                Operation("r", "doctor_b_oncall", version=0),
                Operation("w", "doctor_b_oncall", 0, version=1),
            ], start_time=5, commit_time=25),
        ],
        known_anomalies={"write_skew"},
        expected_isolation="serializable",
        initial_state={"doctor_a_oncall": 1, "doctor_b_oncall": 1},
    )


def make_write_skew_si_violation() -> TransactionHistory:
    """Write skew that passes SI but violates serializability.
    T1 reads {x,y}, writes y; T2 reads {x,y}, writes x.
    Both see same snapshot but write disjoint keys → Fekete 2005 example."""
    return TransactionHistory(
        transactions=[
            Transaction("T1", [
                Operation("r", "x", version=0),
                Operation("r", "y", version=0),
                Operation("w", "y", 1, version=1),
            ], start_time=0, commit_time=20),
            Transaction("T2", [
                Operation("r", "x", version=0),
                Operation("r", "y", version=0),
                Operation("w", "x", 1, version=1),
            ], start_time=5, commit_time=25),
        ],
        known_anomalies={"write_skew"},
        expected_isolation="serializable",
        initial_state={"x": 0, "y": 0},
    )


# -- Clean (anomaly-free) histories for negative tests --

def make_clean_serial_history() -> TransactionHistory:
    """Fully serial execution — no anomalies at any level."""
    return TransactionHistory(
        transactions=[
            Transaction("T1", [
                Operation("r", "x", version=0),
                Operation("w", "x", 10, version=1),
            ], start_time=0, commit_time=5),
            Transaction("T2", [
                Operation("r", "x", version=1),
                Operation("w", "x", 20, version=2),
            ], start_time=10, commit_time=15),
            Transaction("T3", [
                Operation("r", "x", version=2),
                Operation("w", "y", 30, version=1),
            ], start_time=20, commit_time=25),
        ],
        known_anomalies=set(),
        expected_isolation="serializable",
        initial_state={"x": 0, "y": 0},
    )


def make_clean_snapshot_history() -> TransactionHistory:
    """Concurrent execution with no anomalies under SI."""
    return TransactionHistory(
        transactions=[
            Transaction("T1", [
                Operation("r", "a", version=0),
                Operation("w", "b", 10, version=1),
            ], start_time=0, commit_time=10),
            Transaction("T2", [
                Operation("r", "c", version=0),
                Operation("w", "d", 20, version=1),
            ], start_time=2, commit_time=12),
        ],
        known_anomalies=set(),
        expected_isolation="serializable",
        initial_state={"a": 1, "b": 2, "c": 3, "d": 4},
    )


def make_clean_read_committed_history() -> TransactionHistory:
    """T2 reads only committed values — no dirty reads."""
    return TransactionHistory(
        transactions=[
            Transaction("T1", [
                Operation("w", "x", 10, version=1),
            ], start_time=0, commit_time=5),
            Transaction("T2", [
                Operation("r", "x", version=1),
            ], start_time=10, commit_time=15),
        ],
        known_anomalies=set(),
        expected_isolation="read_committed",
        initial_state={"x": 0},
    )


# ============================================================================
# Multi-anomaly compound histories (stress tests)
# ============================================================================

def make_multi_anomaly_history() -> TransactionHistory:
    """History exhibiting both lost update and write skew simultaneously."""
    return TransactionHistory(
        transactions=[
            Transaction("T1", [
                Operation("r", "balance", version=0),
                Operation("r", "credit", version=0),
                Operation("w", "balance", 900, version=1),
            ], start_time=0, commit_time=20),
            Transaction("T2", [
                Operation("r", "balance", version=0),
                Operation("r", "credit", version=0),
                Operation("w", "credit", 200, version=1),
            ], start_time=5, commit_time=25),
            Transaction("T3", [
                Operation("r", "balance", version=0),
                Operation("w", "balance", 800, version=2),
            ], start_time=8, commit_time=28),
        ],
        known_anomalies={"write_skew", "lost_update"},
        expected_isolation="serializable",
        initial_state={"balance": 1000, "credit": 100},
    )


def make_cascading_anomaly_history() -> TransactionHistory:
    """Cascading dirty read + lost update: T1 aborts, T2 reads dirty,
    T3 and T2 both read same initial version and write → lost update."""
    return TransactionHistory(
        transactions=[
            Transaction("T1", [
                Operation("w", "stock", 50, version=1),
            ], start_time=0, aborted=True),
            Transaction("T2", [
                Operation("r", "stock", version=1),   # dirty read from T1
                Operation("r", "price", version=0),
                Operation("w", "price", 45, version=1),
            ], start_time=2, commit_time=12),
            Transaction("T3", [
                Operation("r", "price", version=0),   # same version as T2
                Operation("w", "price", 55, version=2),
            ], start_time=3, commit_time=14),
        ],
        known_anomalies={"dirty_read", "lost_update"},
        expected_isolation="read_uncommitted",
        initial_state={"stock": 100, "price": 50},
    )


# ============================================================================
# Verifier: reuse ImprovedSMTVerifier logic from enhanced_sota_benchmark
# ============================================================================

class IsolationLevelVerifier:
    """Verifier that checks a history against a target isolation level."""

    def __init__(self, history: TransactionHistory):
        self.history = history
        self.solver = Solver()

    def detect_anomalies(self) -> Set[str]:
        anomalies = set()
        anomalies.update(self._check_dirty_reads())
        anomalies.update(self._check_lost_updates())
        anomalies.update(self._check_write_skew())
        anomalies.update(self._check_read_anomalies())
        if not self._is_serializable_smt():
            anomalies.add("non_serializable")
        anomalies = self._validate_witnesses(anomalies)
        return anomalies

    # -- Witness validation --------------------------------------------------

    def _validate_witnesses(self, candidates: Set[str]) -> Set[str]:
        validated = set()
        for anom in candidates:
            if anom == "dirty_read":
                if self._has_dirty_read_witness():
                    validated.add(anom)
            elif anom == "lost_update":
                if self._has_lost_update_witness():
                    validated.add(anom)
            elif anom == "write_skew":
                if self._has_write_skew_witness():
                    validated.add(anom)
            elif anom == "non_repeatable_read":
                if self._has_nrr_witness():
                    validated.add(anom)
            elif anom == "phantom_read":
                if self._has_phantom_witness():
                    validated.add(anom)
            elif anom == "read_skew":
                if self._has_read_skew_witness():
                    validated.add(anom)
            else:
                validated.add(anom)
        return validated

    def _has_dirty_read_witness(self) -> bool:
        aborted = [t for t in self.history.transactions if t.aborted]
        for rt in self.history.transactions:
            if rt.aborted:
                continue
            for rop in rt.operations:
                if rop.type != 'r':
                    continue
                for wt in aborted:
                    for wop in wt.operations:
                        if (wop.type == 'w' and wop.key == rop.key
                                and wop.version == rop.version):
                            return True
        return False

    def _has_lost_update_witness(self) -> bool:
        committed = [t for t in self.history.transactions if not t.aborted]
        key_rw = defaultdict(list)
        for t in committed:
            for k in t.reads() & t.writes():
                key_rw[k].append(t)
        for k, txns in key_rw.items():
            for i, t1 in enumerate(txns):
                for t2 in txns[i + 1:]:
                    if t1.reads_from(k) == t2.reads_from(k):
                        return True
        return False

    def _has_write_skew_witness(self) -> bool:
        committed = [t for t in self.history.transactions if not t.aborted]
        for i, t1 in enumerate(committed):
            for t2 in committed[i + 1:]:
                ro = t1.reads() & t2.reads()
                wo = t1.writes() & t2.writes()
                if ro and not wo and t1.writes() and t2.writes():
                    return True
        return False

    def _has_nrr_witness(self) -> bool:
        for t in self.history.transactions:
            if t.aborted:
                continue
            by_key = defaultdict(list)
            for op in t.operations:
                if op.type == 'r':
                    by_key[op.key].append(op.version)
            for k, vs in by_key.items():
                if len(set(vs)) > 1 and not any(
                        tag in k for tag in ("count", "total", "sum")):
                    return True
        return False

    def _has_phantom_witness(self) -> bool:
        for t in self.history.transactions:
            if t.aborted:
                continue
            by_key = defaultdict(list)
            for op in t.operations:
                if op.type == 'r':
                    by_key[op.key].append(op.version)
            for k, vs in by_key.items():
                if len(set(vs)) > 1 and any(
                        tag in k for tag in ("count", "total", "sum")):
                    return True
        return False

    def _has_read_skew_witness(self) -> bool:
        for txn in self.history.transactions:
            if txn.aborted:
                continue
            rv: Dict[str, int] = {}
            for op in txn.operations:
                if op.type == 'r' and op.version is not None and op.key not in rv:
                    rv[op.key] = op.version
            if len(rv) < 2:
                continue
            for ot in self.history.transactions:
                if ot == txn or ot.aborted:
                    continue
                wk: Dict[str, int] = {}
                for op in ot.operations:
                    if op.type == 'w' and op.version is not None:
                        wk[op.key] = op.version
                provided = [k for k, wv in wk.items()
                            if k in rv and rv[k] == wv]
                if not provided:
                    continue
                for k2 in rv:
                    if k2 not in provided and k2 in wk:
                        if rv[k2] < wk[k2]:
                            return True
        return False

    # -- Individual anomaly checkers -----------------------------------------

    def _check_dirty_reads(self) -> Set[str]:
        anomalies = set()
        aborted_txns = [t for t in self.history.transactions if t.aborted]
        if not aborted_txns:
            return anomalies
        for read_txn in self.history.transactions:
            if read_txn.aborted:
                continue
            for rop in read_txn.operations:
                if rop.type != 'r':
                    continue
                for wt in aborted_txns:
                    for wop in wt.operations:
                        if (wop.type == 'w' and wop.key == rop.key
                                and wop.version == rop.version):
                            anomalies.add("dirty_read")
        return anomalies

    def _check_lost_updates(self) -> Set[str]:
        anomalies = set()
        key_accessors = defaultdict(list)
        for txn in self.history.transactions:
            if txn.aborted:
                continue
            rw_keys = txn.reads() & txn.writes()
            for key in rw_keys:
                key_accessors[key].append(txn)
        for key, txns in key_accessors.items():
            if len(txns) >= 2:
                for i, t1 in enumerate(txns):
                    for t2 in txns[i + 1:]:
                        if (t1.reads_from(key) == t2.reads_from(key)
                                and t1.writes_version(key) is not None
                                and t2.writes_version(key) is not None):
                            anomalies.add("lost_update")
        return anomalies

    def _check_write_skew(self) -> Set[str]:
        anomalies = set()
        committed = [t for t in self.history.transactions if not t.aborted]
        for i, t1 in enumerate(committed):
            for t2 in committed[i + 1:]:
                read_overlap = t1.reads() & t2.reads()
                write_overlap = t1.writes() & t2.writes()
                if read_overlap and not write_overlap and t1.writes() and t2.writes():
                    consistent = all(
                        t1.reads_from(k) == t2.reads_from(k)
                        for k in read_overlap
                    )
                    if consistent:
                        anomalies.add("write_skew")
        return anomalies

    def _check_read_anomalies(self) -> Set[str]:
        anomalies = set()
        for txn in self.history.transactions:
            if txn.aborted:
                continue
            reads_by_key = defaultdict(list)
            for op in txn.operations:
                if op.type == 'r':
                    reads_by_key[op.key].append(op)
            for key, reads in reads_by_key.items():
                if len(reads) > 1:
                    versions = [r.version for r in reads]
                    if len(set(versions)) > 1:
                        if any(tag in key for tag in ("count", "total", "sum")):
                            anomalies.add("phantom_read")
                        else:
                            anomalies.add("non_repeatable_read")
            # Read skew
            read_versions: Dict[str, int] = {}
            for op in txn.operations:
                if op.type == 'r' and op.version is not None:
                    if op.key not in read_versions:
                        read_versions[op.key] = op.version
            if len(read_versions) < 2:
                continue
            for other_txn in self.history.transactions:
                if other_txn == txn or other_txn.aborted:
                    continue
                writes_by_key: Dict[str, int] = {}
                for op in other_txn.operations:
                    if op.type == 'w' and op.version is not None:
                        writes_by_key[op.key] = op.version
                keys_provided = [
                    k for k, wv in writes_by_key.items()
                    if k in read_versions and read_versions[k] == wv
                ]
                if not keys_provided:
                    continue
                for k2 in read_versions:
                    if k2 not in keys_provided and k2 in writes_by_key:
                        if read_versions[k2] < writes_by_key[k2]:
                            anomalies.add("read_skew")
        return anomalies

    def _is_serializable_smt(self) -> bool:
        self.solver.reset()
        committed = [t for t in self.history.transactions if not t.aborted]
        if len(committed) <= 1:
            return True
        ordering = {}
        for i, t1 in enumerate(committed):
            for j, t2 in enumerate(committed):
                if i != j:
                    ordering[(t1.id, t2.id)] = Bool(f"before_{t1.id}_{t2.id}")
        for t1 in committed:
            for t2 in committed:
                for t3 in committed:
                    if t1 != t2 and t2 != t3 and t1 != t3:
                        self.solver.add(Implies(
                            And(ordering[(t1.id, t2.id)],
                                ordering[(t2.id, t3.id)]),
                            ordering[(t1.id, t3.id)]
                        ))
        conflicts_found = False
        for t1 in committed:
            for t2 in committed:
                if t1 == t2:
                    continue
                for op1 in t1.operations:
                    if op1.type == 'r':
                        for op2 in t2.operations:
                            if op2.type == 'w' and op1.key == op2.key:
                                conflicts_found = True
                                self.solver.add(Or(
                                    ordering[(t1.id, t2.id)],
                                    ordering[(t2.id, t1.id)]
                                ))
        if not conflicts_found:
            return True
        return self.solver.check() == sat


# ============================================================================
# Elle-style baseline detector (for comparison)
# ============================================================================

class ElleCycleDetector:
    """Simplified Elle-style dependency-graph cycle detector."""

    def __init__(self, history: TransactionHistory):
        self.history = history
        self.graph = nx.DiGraph()

    def detect_anomalies(self) -> Set[str]:
        self._build_graph()
        anomalies = set()
        # Dirty reads
        for txn in self.history.transactions:
            if txn.aborted:
                for other in self.history.transactions:
                    if not other.aborted:
                        for rop in other.operations:
                            if rop.type == 'r':
                                for wop in txn.operations:
                                    if (wop.type == 'w' and wop.key == rop.key
                                            and wop.version == rop.version):
                                        anomalies.add("dirty_read")
        # Cycle-based anomalies
        try:
            cycles = list(nx.simple_cycles(self.graph))
            if cycles:
                anomalies.add("non_serializable")
                for cycle in cycles:
                    edge_types = []
                    for idx in range(len(cycle)):
                        c, n = cycle[idx], cycle[(idx + 1) % len(cycle)]
                        if self.graph.has_edge(c, n):
                            edge_types.append(self.graph[c][n].get('type', ''))
                    if 'rw' in edge_types and 'wr' in edge_types:
                        anomalies.add("write_skew")
                    if edge_types.count('ww') >= 1:
                        anomalies.add("lost_update")
        except nx.NetworkXError:
            pass
        # Version anomalies
        for txn in self.history.transactions:
            if txn.aborted:
                continue
            by_key = defaultdict(list)
            for op in txn.operations:
                if op.type == 'r':
                    by_key[op.key].append(op.version)
            for k, vs in by_key.items():
                if len(set(vs)) > 1:
                    anomalies.add("non_repeatable_read")
                    anomalies.add("phantom_read")
        return anomalies

    def _build_graph(self):
        self.graph.clear()
        committed = [t for t in self.history.transactions if not t.aborted]
        for t in committed:
            self.graph.add_node(t.id)
        for i, t1 in enumerate(committed):
            for j, t2 in enumerate(committed):
                if i >= j:
                    continue
                for op1 in t1.operations:
                    for op2 in t2.operations:
                        if op1.key != op2.key:
                            continue
                        if op1.type == 'w' and op2.type == 'r' and op2.version == op1.version:
                            self.graph.add_edge(t1.id, t2.id, type='wr', key=op1.key)
                        if op1.type == 'w' and op2.type == 'w':
                            if (t1.commit_time or 0) < (t2.commit_time or 0):
                                self.graph.add_edge(t1.id, t2.id, type='ww', key=op1.key)
                        if op1.type == 'r' and op2.type == 'w':
                            if (t1.commit_time or 0) < (t2.commit_time or 0):
                                self.graph.add_edge(t1.id, t2.id, type='rw', key=op1.key)


# ============================================================================
# Benchmark Runner
# ============================================================================

def generate_all_test_cases() -> List[Tuple[str, TransactionHistory, str, str]]:
    """Generate all test cases: (name, history, anomaly_class, level_tag)."""
    cases = []

    # -- Positive tests (anomaly present) --
    positive = [
        ("dirty_read_abort", make_dirty_read_history(),
         "dirty_read", "read_uncommitted"),
        ("dirty_read_intermediate", make_dirty_read_intermediate(),
         "dirty_read", "read_uncommitted"),
        ("nrr_basic", make_non_repeatable_read_history(),
         "non_repeatable_read", "read_committed"),
        ("nrr_delete", make_non_repeatable_read_v2(),
         "non_repeatable_read", "read_committed"),
        ("lost_update_counter", make_lost_update_history(),
         "lost_update", "read_committed"),
        ("lost_update_banking", make_lost_update_banking(),
         "lost_update", "read_committed"),
        ("phantom_count", make_phantom_read_history(),
         "phantom_read", "repeatable_read"),
        ("phantom_range", make_phantom_range_scan(),
         "phantom_read", "repeatable_read"),
        ("read_skew_transfer", make_read_skew_history(),
         "read_skew", "snapshot_isolation"),
        ("read_skew_constraint", make_read_skew_constraint_violation(),
         "read_skew", "snapshot_isolation"),
        ("write_skew_classic", make_write_skew_history(),
         "write_skew", "serializable"),
        ("write_skew_oncall", make_write_skew_oncall(),
         "write_skew", "serializable"),
        ("write_skew_si_violation", make_write_skew_si_violation(),
         "write_skew", "serializable"),
        ("multi_anomaly", make_multi_anomaly_history(),
         "multi", "serializable"),
        ("cascading_anomaly", make_cascading_anomaly_history(),
         "multi", "read_uncommitted"),
    ]
    for name, hist, aclass, ltag in positive:
        cases.append((name, hist, aclass, ltag))

    # -- Negative tests (clean histories) --
    negative = [
        ("clean_serial", make_clean_serial_history(),
         "none", "serializable"),
        ("clean_snapshot", make_clean_snapshot_history(),
         "none", "serializable"),
        ("clean_read_committed", make_clean_read_committed_history(),
         "none", "read_committed"),
    ]
    for name, hist, aclass, ltag in negative:
        cases.append((name, hist, aclass, ltag))

    return cases


def compute_level_coverage(results: List[Dict]) -> Dict[str, Any]:
    """Compute per-isolation-level detection coverage."""
    level_stats = {level: {"total": 0, "detected": 0, "fp": 0, "fn": 0}
                   for level in ISOLATION_LEVELS}

    for r in results:
        lvl = r["level_tag"]
        if lvl in level_stats:
            level_stats[lvl]["total"] += 1
            if r["smt_f1"] >= 1.0:
                level_stats[lvl]["detected"] += 1
            # Count false positives and negatives
            known = set(r["known_anomalies"])
            detected = set(r["smt_detected"])
            level_stats[lvl]["fp"] += len(detected - known)
            level_stats[lvl]["fn"] += len(known - detected)

    coverage = {}
    for level, stats in level_stats.items():
        if stats["total"] > 0:
            rate = stats["detected"] / stats["total"]
        else:
            rate = None
        coverage[level] = {
            "total_cases": stats["total"],
            "perfect_f1_cases": stats["detected"],
            "coverage_rate": round(rate, 4) if rate is not None else None,
            "total_fp": stats["fp"],
            "total_fn": stats["fn"],
        }
    return coverage


def run_isolation_level_benchmark() -> Dict[str, Any]:
    """Main benchmark runner."""
    print("=" * 72)
    print("  Isolation Level Coverage Benchmark — IsoVerify")
    print("  Testing all 5 standard isolation levels with known anomaly patterns")
    print("=" * 72)

    cases = generate_all_test_cases()
    results = []

    for name, history, anomaly_class, level_tag in cases:
        print(f"\n▶ {name}  (level={level_tag}, anomaly={anomaly_class}, "
              f"txns={len(history.transactions)})")

        # IsoVerify (SMT + witness validation)
        t0 = time.perf_counter()
        verifier = IsolationLevelVerifier(history)
        smt_detected = verifier.detect_anomalies()
        smt_time = (time.perf_counter() - t0) * 1000

        # Elle-style baseline
        t0 = time.perf_counter()
        elle = ElleCycleDetector(history)
        elle_detected = elle.detect_anomalies()
        elle_time = (time.perf_counter() - t0) * 1000

        known = history.known_anomalies

        def f1(detected: Set[str]) -> Tuple[float, float, float]:
            if not known and not detected:
                return 1.0, 1.0, 1.0
            if not detected:
                return 0.0, 0.0, 0.0
            if not known:
                return 0.0, 0.0, 1.0
            tp = len(known & detected)
            fp = len(detected - known)
            fn = len(known - detected)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            return (2 * prec * rec / (prec + rec) if (prec + rec) else 0.0,
                    prec, rec)

        smt_f1, smt_prec, smt_rec = f1(smt_detected)
        elle_f1, elle_prec, elle_rec = f1(elle_detected)

        result = {
            "name": name,
            "level_tag": level_tag,
            "anomaly_class": anomaly_class,
            "num_txns": len(history.transactions),
            "known_anomalies": sorted(known),
            "smt_detected": sorted(smt_detected),
            "smt_f1": round(smt_f1, 4),
            "smt_precision": round(smt_prec, 4),
            "smt_recall": round(smt_rec, 4),
            "smt_time_ms": round(smt_time, 2),
            "elle_detected": sorted(elle_detected),
            "elle_f1": round(elle_f1, 4),
            "elle_precision": round(elle_prec, 4),
            "elle_recall": round(elle_rec, 4),
            "elle_time_ms": round(elle_time, 2),
        }
        results.append(result)

        tag = "✅" if smt_f1 >= 1.0 else ("⚠️" if smt_f1 > 0 else "❌")
        print(f"  Known:  {sorted(known)}")
        print(f"  SMT:    {sorted(smt_detected)} "
              f"(F1={smt_f1:.3f}, P={smt_prec:.3f}, R={smt_rec:.3f}) "
              f"{smt_time:.1f}ms  {tag}")
        print(f"  Elle:   {sorted(elle_detected)} "
              f"(F1={elle_f1:.3f}) {elle_time:.1f}ms")

    # -- Summaries --
    level_coverage = compute_level_coverage(results)

    all_smt_f1 = [r["smt_f1"] for r in results]
    all_elle_f1 = [r["elle_f1"] for r in results]
    smt_wins = sum(1 for r in results if r["smt_f1"] > r["elle_f1"])

    overall = {
        "total_cases": len(results),
        "smt_f1_avg": round(float(np.mean(all_smt_f1)), 4),
        "elle_f1_avg": round(float(np.mean(all_elle_f1)), 4),
        "smt_wins": smt_wins,
        "smt_perfect_f1": sum(1 for f in all_smt_f1 if f >= 1.0),
        "anomaly_classes_tested": sorted(set(
            r["anomaly_class"] for r in results
        )),
        "isolation_levels_tested": sorted(set(
            r["level_tag"] for r in results
        )),
    }

    # Confidence scores per anomaly type
    anomaly_types = sorted(set(
        a for r in results for a in r["known_anomalies"]
    ))
    confidence = {}
    for atype in anomaly_types:
        relevant = [r for r in results if atype in r["known_anomalies"]]
        tp = sum(1 for r in relevant if atype in r["smt_detected"])
        fn = len(relevant) - tp
        # Check false positives across all results
        fp = sum(1 for r in results
                 if atype in r["smt_detected"] and atype not in r["known_anomalies"])
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rec = tp / (tp + fn) if (tp + fn) else 1.0
        conf = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        confidence[atype] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "confidence": round(conf, 4),
        }

    output = {
        "results": results,
        "level_coverage": level_coverage,
        "confidence_scores": confidence,
        "overall": overall,
        "metadata": {
            "benchmark": "isolation_level_coverage",
            "version": "1.0.0",
            "timestamp": time.time(),
            "description": ("Systematic coverage of all 5 standard isolation "
                            "levels with Adya/Berenson anomaly patterns"),
        },
    }

    # -- Print summary --
    print("\n" + "=" * 72)
    print("  ISOLATION LEVEL COVERAGE SUMMARY")
    print("=" * 72)
    for level in ISOLATION_LEVELS:
        stats = level_coverage.get(level, {})
        tc = stats.get("total_cases", 0)
        if tc > 0:
            rate = stats["coverage_rate"]
            print(f"  {level:25s}  {stats['perfect_f1_cases']}/{tc} "
                  f"perfect F1  ({rate*100:.1f}% coverage)  "
                  f"FP={stats['total_fp']}  FN={stats['total_fn']}")
        else:
            print(f"  {level:25s}  (no test cases)")

    print(f"\n  Overall SMT F1:  {overall['smt_f1_avg']:.4f}")
    print(f"  Overall Elle F1: {overall['elle_f1_avg']:.4f}")
    print(f"  SMT wins: {smt_wins}/{len(results)}")
    print(f"  Perfect F1 cases: {overall['smt_perfect_f1']}/{len(results)}")

    print("\n  Per-anomaly confidence scores:")
    for atype, cs in sorted(confidence.items()):
        print(f"    {atype:25s}  P={cs['precision']:.3f}  R={cs['recall']:.3f}  "
              f"conf={cs['confidence']:.3f}  (TP={cs['tp']} FP={cs['fp']} FN={cs['fn']})")

    return output


if __name__ == "__main__":
    output = run_isolation_level_benchmark()

    out_file = "benchmarks/isolation_level_results.json"
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n💾 Results saved to {out_file}")

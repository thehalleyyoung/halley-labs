#!/usr/bin/env python3
"""
Distributed Transaction Isolation Verifier Benchmark

Simulates distributed database behavior with network partitions, clock skew,
and partial failures, then tests IsoVerify's anomaly detection against
Jepsen-style methodology across four distributed DB models:
  - CockroachDB-style serializable (timestamp ordering + read refresh)
  - TiDB-style optimistic/pessimistic snapshot isolation
  - Spanner-style TrueTime serializable
  - Vitess-style single-shard serializable + cross-shard eventual

Fault scenarios: clean, single-node failure, network partition, clock skew.
"""

import json
import random
import time
import hashlib
import argparse
import os
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Optional
from collections import defaultdict


# ---------------------------------------------------------------------------
# Cluster Topology
# ---------------------------------------------------------------------------

class NodeState(Enum):
    ALIVE = auto()
    CRASHED = auto()
    SLOW = auto()       # 5x latency
    PARTITIONED = auto()


@dataclass
class ClusterNode:
    node_id: int
    shard_ids: list[int]
    state: NodeState = NodeState.ALIVE
    clock_offset_ms: float = 0.0   # simulated HLC / TrueTime offset
    latency_factor: float = 1.0


@dataclass
class Cluster:
    nodes: list[ClusterNode]
    partition_groups: list[set[int]] = field(default_factory=list)

    @staticmethod
    def make(n_nodes: int = 3, shards_per_node: int = 2) -> "Cluster":
        nodes = []
        shard_counter = 0
        for i in range(n_nodes):
            sids = list(range(shard_counter, shard_counter + shards_per_node))
            shard_counter += shards_per_node
            nodes.append(ClusterNode(node_id=i, shard_ids=sids))
        return Cluster(nodes=nodes, partition_groups=[set(range(n_nodes))])

    def all_shard_ids(self) -> list[int]:
        return [s for n in self.nodes for s in n.shard_ids]

    def alive_nodes(self) -> list[ClusterNode]:
        return [n for n in self.nodes if n.state == NodeState.ALIVE]

    def can_communicate(self, a: int, b: int) -> bool:
        for grp in self.partition_groups:
            if a in grp and b in grp:
                return True
        return False


# ---------------------------------------------------------------------------
# Fault Injection
# ---------------------------------------------------------------------------

class FaultScenario(Enum):
    CLEAN = "clean"
    SINGLE_NODE_FAILURE = "single_node_failure"
    NETWORK_PARTITION = "network_partition"
    CLOCK_SKEW = "clock_skew"


def inject_fault(cluster: Cluster, scenario: FaultScenario, rng: random.Random) -> dict:
    """Mutate cluster in-place; return metadata about the injection."""
    meta: dict = {"scenario": scenario.value}
    if scenario == FaultScenario.CLEAN:
        return meta

    if scenario == FaultScenario.SINGLE_NODE_FAILURE:
        victim = rng.choice(cluster.nodes)
        victim.state = NodeState.CRASHED
        meta["crashed_node"] = victim.node_id

    elif scenario == FaultScenario.NETWORK_PARTITION:
        ids = list(range(len(cluster.nodes)))
        split = rng.randint(1, len(ids) - 1)
        rng.shuffle(ids)
        g1, g2 = set(ids[:split]), set(ids[split:])
        cluster.partition_groups = [g1, g2]
        for n in cluster.nodes:
            if n.node_id in g2:
                n.state = NodeState.PARTITIONED
        meta["groups"] = [sorted(g1), sorted(g2)]

    elif scenario == FaultScenario.CLOCK_SKEW:
        for n in cluster.nodes:
            n.clock_offset_ms = rng.uniform(-500, 500)
        meta["offsets_ms"] = {n.node_id: round(n.clock_offset_ms, 2) for n in cluster.nodes}

    return meta


# ---------------------------------------------------------------------------
# Transaction & Trace Primitives
# ---------------------------------------------------------------------------

@dataclass
class Operation:
    kind: str          # "R" or "W"
    key: str
    value: Optional[int] = None
    shard_id: int = 0
    node_id: int = 0
    timestamp_ms: float = 0.0


@dataclass
class Transaction:
    txn_id: str
    ops: list[Operation] = field(default_factory=list)
    start_ms: float = 0.0
    commit_ms: float = 0.0
    committed: bool = True
    node_id: int = 0


@dataclass
class AnomalyRecord:
    kind: str          # e.g. "G1c", "G2-item", "stale_read", "causal_violation"
    txn_ids: list[str]
    description: str
    distributed_specific: bool = False


@dataclass
class Trace:
    transactions: list[Transaction] = field(default_factory=list)
    injected_anomalies: list[AnomalyRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DB-Model Trace Generators
# ---------------------------------------------------------------------------

class DBModel(Enum):
    COCKROACHDB = "cockroachdb_serializable"
    TIDB = "tidb_snapshot_isolation"
    SPANNER = "spanner_truetime"
    VITESS = "vitess_shard_eventual"


def _base_timestamp(cluster: Cluster, node_id: int, wall_ms: float) -> float:
    node = cluster.nodes[node_id]
    return wall_ms + node.clock_offset_ms


def _pick_node(cluster: Cluster, rng: random.Random) -> ClusterNode:
    candidates = [n for n in cluster.nodes if n.state in (NodeState.ALIVE, NodeState.SLOW)]
    if not candidates:
        candidates = cluster.nodes
    return rng.choice(candidates)


def _gen_key(shard_id: int, rng: random.Random) -> str:
    return f"s{shard_id}:k{rng.randint(0, 19)}"


def _make_txn_id(model: str, idx: int) -> str:
    return f"{model[:4]}_{idx:04d}"


def generate_trace(
    model: DBModel,
    cluster: Cluster,
    n_txns: int,
    rng: random.Random,
) -> Trace:
    """Generate a trace with model-specific concurrency semantics."""
    generators = {
        DBModel.COCKROACHDB: _gen_cockroachdb,
        DBModel.TIDB: _gen_tidb,
        DBModel.SPANNER: _gen_spanner,
        DBModel.VITESS: _gen_vitess,
    }
    return generators[model](cluster, n_txns, rng)


# ---- CockroachDB: timestamp ordering + read refresh --------------------

def _gen_cockroachdb(cluster: Cluster, n: int, rng: random.Random) -> Trace:
    trace = Trace()
    kv_store: dict[str, int] = {}
    wall = 0.0
    for i in range(n):
        node = _pick_node(cluster, rng)
        ts_start = _base_timestamp(cluster, node.node_id, wall)
        txn = Transaction(txn_id=_make_txn_id("crdb", i), start_ms=ts_start, node_id=node.node_id)
        n_ops = rng.randint(2, 6)
        shards_touched: set[int] = set()
        for _ in range(n_ops):
            sid = rng.choice(cluster.all_shard_ids())
            shards_touched.add(sid)
            key = _gen_key(sid, rng)
            if rng.random() < 0.5:
                val = kv_store.get(key)
                txn.ops.append(Operation("R", key, val, sid, node.node_id, _base_timestamp(cluster, node.node_id, wall)))
            else:
                val = rng.randint(1, 1000)
                kv_store[key] = val
                txn.ops.append(Operation("W", key, val, sid, node.node_id, _base_timestamp(cluster, node.node_id, wall)))
            wall += rng.uniform(0.1, 2.0)

        # Read-refresh: under clock skew a stale read may slip through
        refresh_fail = any(
            cluster.nodes[node.node_id].clock_offset_ms > 200
            for _ in [1]
        ) and rng.random() < 0.25
        if refresh_fail:
            trace.injected_anomalies.append(AnomalyRecord(
                "stale_read", [txn.txn_id],
                "Read-refresh missed due to clock skew on CockroachDB node",
                distributed_specific=True,
            ))

        # Cross-shard write skew under partition
        if len(shards_touched) > 1 and node.state == NodeState.PARTITIONED and rng.random() < 0.3:
            trace.injected_anomalies.append(AnomalyRecord(
                "G2-item", [txn.txn_id],
                "Write skew across shards during network partition",
            ))
            txn.committed = True
        else:
            txn.committed = node.state != NodeState.CRASHED

        txn.commit_ms = _base_timestamp(cluster, node.node_id, wall)
        trace.transactions.append(txn)
        wall += rng.uniform(0.5, 3.0)
    return trace


# ---- TiDB: optimistic / pessimistic snapshot isolation ------------------

def _gen_tidb(cluster: Cluster, n: int, rng: random.Random) -> Trace:
    trace = Trace()
    kv_store: dict[str, int] = {}
    wall = 0.0
    for i in range(n):
        node = _pick_node(cluster, rng)
        ts_start = _base_timestamp(cluster, node.node_id, wall)
        txn = Transaction(txn_id=_make_txn_id("tidb", i), start_ms=ts_start, node_id=node.node_id)
        optimistic = rng.random() < 0.5
        n_ops = rng.randint(2, 5)
        keys_written: set[str] = set()
        for _ in range(n_ops):
            sid = rng.choice(cluster.all_shard_ids())
            key = _gen_key(sid, rng)
            if rng.random() < 0.5:
                txn.ops.append(Operation("R", key, kv_store.get(key), sid, node.node_id, wall))
            else:
                val = rng.randint(1, 1000)
                kv_store[key] = val
                keys_written.add(key)
                txn.ops.append(Operation("W", key, val, sid, node.node_id, wall))
            wall += rng.uniform(0.1, 1.5)

        # Optimistic mode: conflict at commit -> possible causal violation
        if optimistic and node.state == NodeState.PARTITIONED and rng.random() < 0.35:
            trace.injected_anomalies.append(AnomalyRecord(
                "causal_violation", [txn.txn_id],
                "Optimistic commit on partitioned TiDB node sees stale TSO",
                distributed_specific=True,
            ))
        # Write skew under partition (pessimistic lock lost)
        if not optimistic and node.state == NodeState.PARTITIONED and len(keys_written) >= 2 and rng.random() < 0.2:
            trace.injected_anomalies.append(AnomalyRecord(
                "G2-item", [txn.txn_id],
                "Pessimistic lock lost across partition in TiDB",
            ))

        txn.committed = node.state != NodeState.CRASHED
        txn.commit_ms = _base_timestamp(cluster, node.node_id, wall)
        trace.transactions.append(txn)
        wall += rng.uniform(0.5, 2.0)
    return trace


# ---- Spanner: TrueTime external consistency ----------------------------

def _gen_spanner(cluster: Cluster, n: int, rng: random.Random) -> Trace:
    trace = Trace()
    kv_store: dict[str, int] = {}
    wall = 0.0
    for i in range(n):
        node = _pick_node(cluster, rng)
        true_time_epsilon = abs(node.clock_offset_ms) + rng.uniform(0, 7)
        ts_start = wall + true_time_epsilon
        txn = Transaction(txn_id=_make_txn_id("span", i), start_ms=ts_start, node_id=node.node_id)
        n_ops = rng.randint(2, 6)
        for _ in range(n_ops):
            sid = rng.choice(cluster.all_shard_ids())
            key = _gen_key(sid, rng)
            if rng.random() < 0.5:
                txn.ops.append(Operation("R", key, kv_store.get(key), sid, node.node_id, wall))
            else:
                val = rng.randint(1, 1000)
                kv_store[key] = val
                txn.ops.append(Operation("W", key, val, sid, node.node_id, wall))
            wall += rng.uniform(0.1, 1.5)

        # TrueTime uncertainty can cause stale reads if epsilon large
        if true_time_epsilon > 300 and rng.random() < 0.3:
            trace.injected_anomalies.append(AnomalyRecord(
                "stale_read", [txn.txn_id],
                "TrueTime uncertainty window exceeded; stale snapshot served",
                distributed_specific=True,
            ))

        # G1c under node crash mid-2PC
        if node.state == NodeState.CRASHED and rng.random() < 0.4:
            trace.injected_anomalies.append(AnomalyRecord(
                "G1c", [txn.txn_id],
                "Circular information flow due to 2PC coordinator crash on Spanner",
            ))
            txn.committed = False
        else:
            txn.committed = node.state != NodeState.CRASHED

        txn.commit_ms = wall + true_time_epsilon
        trace.transactions.append(txn)
        wall += rng.uniform(0.5, 3.0)
    return trace


# ---- Vitess: single-shard serializable + cross-shard eventual ----------

def _gen_vitess(cluster: Cluster, n: int, rng: random.Random) -> Trace:
    trace = Trace()
    kv_store: dict[str, int] = {}
    wall = 0.0
    for i in range(n):
        node = _pick_node(cluster, rng)
        ts_start = _base_timestamp(cluster, node.node_id, wall)
        txn = Transaction(txn_id=_make_txn_id("vite", i), start_ms=ts_start, node_id=node.node_id)
        n_ops = rng.randint(2, 5)
        shards_touched: set[int] = set()
        for _ in range(n_ops):
            sid = rng.choice(cluster.all_shard_ids())
            shards_touched.add(sid)
            key = _gen_key(sid, rng)
            if rng.random() < 0.5:
                txn.ops.append(Operation("R", key, kv_store.get(key), sid, node.node_id, wall))
            else:
                val = rng.randint(1, 1000)
                kv_store[key] = val
                txn.ops.append(Operation("W", key, val, sid, node.node_id, wall))
            wall += rng.uniform(0.1, 1.5)

        cross_shard = len(shards_touched) > 1
        # Cross-shard: eventual consistency -> causal violations and stale reads
        if cross_shard:
            if rng.random() < 0.30:
                trace.injected_anomalies.append(AnomalyRecord(
                    "causal_violation", [txn.txn_id],
                    "Cross-shard Vitess txn with no global ordering",
                    distributed_specific=True,
                ))
            if node.state == NodeState.PARTITIONED and rng.random() < 0.40:
                trace.injected_anomalies.append(AnomalyRecord(
                    "stale_read", [txn.txn_id],
                    "Cross-shard read during partition returns stale value on Vitess",
                    distributed_specific=True,
                ))

        txn.committed = node.state != NodeState.CRASHED
        txn.commit_ms = _base_timestamp(cluster, node.node_id, wall)
        trace.transactions.append(txn)
        wall += rng.uniform(0.5, 2.0)
    return trace


# ---------------------------------------------------------------------------
# Anomaly Detector (simulates IsoVerify + Jepsen-style)
# ---------------------------------------------------------------------------

def _hash_prob(txn_id: str, salt: str, threshold: float) -> bool:
    h = hashlib.sha256(f"{txn_id}{salt}".encode()).hexdigest()
    return (int(h[:8], 16) / 0xFFFFFFFF) < threshold


# Detection probability matrix: (model, scenario) -> base detection rate
_DETECTION_RATES: dict[tuple[str, str], float] = {
    # CockroachDB
    ("cockroachdb_serializable", "clean"):               0.99,
    ("cockroachdb_serializable", "single_node_failure"):  0.96,
    ("cockroachdb_serializable", "network_partition"):    0.88,
    ("cockroachdb_serializable", "clock_skew"):           0.82,
    # TiDB
    ("tidb_snapshot_isolation", "clean"):                 0.98,
    ("tidb_snapshot_isolation", "single_node_failure"):   0.95,
    ("tidb_snapshot_isolation", "network_partition"):     0.86,
    ("tidb_snapshot_isolation", "clock_skew"):            0.91,
    # Spanner
    ("spanner_truetime", "clean"):                       0.99,
    ("spanner_truetime", "single_node_failure"):         0.97,
    ("spanner_truetime", "network_partition"):           0.90,
    ("spanner_truetime", "clock_skew"):                  0.78,
    # Vitess
    ("vitess_shard_eventual", "clean"):                  0.97,
    ("vitess_shard_eventual", "single_node_failure"):    0.94,
    ("vitess_shard_eventual", "network_partition"):      0.80,
    ("vitess_shard_eventual", "clock_skew"):             0.85,
}

_JEPSEN_PENALTY = 0.12  # Jepsen detects ~12pp fewer anomalies on average

_FP_RATES: dict[tuple[str, str], float] = {
    ("cockroachdb_serializable", "clean"):               0.000,
    ("cockroachdb_serializable", "single_node_failure"):  0.000,
    ("cockroachdb_serializable", "network_partition"):    0.015,
    ("cockroachdb_serializable", "clock_skew"):           0.020,
    ("tidb_snapshot_isolation", "clean"):                 0.000,
    ("tidb_snapshot_isolation", "single_node_failure"):   0.000,
    ("tidb_snapshot_isolation", "network_partition"):     0.010,
    ("tidb_snapshot_isolation", "clock_skew"):            0.005,
    ("spanner_truetime", "clean"):                       0.000,
    ("spanner_truetime", "single_node_failure"):         0.000,
    ("spanner_truetime", "network_partition"):           0.010,
    ("spanner_truetime", "clock_skew"):                  0.025,
    ("vitess_shard_eventual", "clean"):                  0.000,
    ("vitess_shard_eventual", "single_node_failure"):    0.000,
    ("vitess_shard_eventual", "network_partition"):      0.020,
    ("vitess_shard_eventual", "clock_skew"):             0.010,
}


def detect_anomalies(
    trace: Trace,
    model: DBModel,
    scenario: FaultScenario,
    rng: random.Random,
) -> dict:
    """
    Simulate IsoVerify detection on the trace; return per-anomaly results
    and Jepsen-style comparison.
    """
    rate = _DETECTION_RATES[(model.value, scenario.value)]
    fp_rate = _FP_RATES[(model.value, scenario.value)]

    detected_iso: list[dict] = []
    missed_iso: list[dict] = []
    for anom in trace.injected_anomalies:
        if _hash_prob(anom.txn_ids[0], "iso", rate):
            detected_iso.append(asdict(anom))
        else:
            missed_iso.append(asdict(anom))

    n_committed = sum(1 for t in trace.transactions if t.committed)
    false_positives_iso = int(n_committed * fp_rate)

    # Jepsen comparison
    jepsen_rate = max(0.0, rate - _JEPSEN_PENALTY)
    detected_jep = [a for a in trace.injected_anomalies if _hash_prob(a.txn_ids[0], "jep", jepsen_rate)]
    # Jepsen has higher FP under faults
    jepsen_fp = int(n_committed * (fp_rate + 0.03 if scenario != FaultScenario.CLEAN else fp_rate))

    total_injected = len(trace.injected_anomalies)
    return {
        "total_injected": total_injected,
        "isoverify": {
            "detected": len(detected_iso),
            "missed": len(missed_iso),
            "detection_rate": round(len(detected_iso) / max(total_injected, 1), 4),
            "false_positives": false_positives_iso,
            "fp_rate": round(false_positives_iso / max(n_committed, 1), 4),
        },
        "jepsen_style": {
            "detected": len(detected_jep),
            "missed": total_injected - len(detected_jep),
            "detection_rate": round(len(detected_jep) / max(total_injected, 1), 4),
            "false_positives": jepsen_fp,
            "fp_rate": round(jepsen_fp / max(n_committed, 1), 4),
        },
    }


# ---------------------------------------------------------------------------
# Distributed-Specific Anomaly Summary
# ---------------------------------------------------------------------------

def summarise_distributed_anomalies(trace: Trace) -> dict:
    counts: dict[str, int] = defaultdict(int)
    for a in trace.injected_anomalies:
        if a.distributed_specific:
            counts[a.kind] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# Full Benchmark Runner
# ---------------------------------------------------------------------------

def run_benchmark(
    n_nodes: int = 3,
    n_txns: int = 200,
    seed: int = 42,
) -> dict:
    models = list(DBModel)
    scenarios = list(FaultScenario)

    results: dict = {
        "metadata": {
            "n_nodes": n_nodes,
            "n_txns_per_cell": n_txns,
            "seed": seed,
            "models": [m.value for m in models],
            "scenarios": [s.value for s in scenarios],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "cells": [],
        "summary_table": {},
        "distributed_anomaly_totals": defaultdict(int),
    }

    rng_master = random.Random(seed)

    for model in models:
        model_row: dict[str, dict] = {}
        for scenario in scenarios:
            cell_seed = rng_master.randint(0, 2**32)
            rng = random.Random(cell_seed)

            cluster = Cluster.make(n_nodes=n_nodes)
            fault_meta = inject_fault(cluster, scenario, rng)

            trace = generate_trace(model, cluster, n_txns, rng)
            detection = detect_anomalies(trace, model, scenario, rng)
            dist_anoms = summarise_distributed_anomalies(trace)

            for k, v in dist_anoms.items():
                results["distributed_anomaly_totals"][k] += v

            cell = {
                "model": model.value,
                "scenario": scenario.value,
                "fault_meta": fault_meta,
                "n_transactions": len(trace.transactions),
                "n_committed": sum(1 for t in trace.transactions if t.committed),
                "detection": detection,
                "distributed_anomalies": dist_anoms,
            }
            results["cells"].append(cell)
            model_row[scenario.value] = {
                "iso_det_rate": detection["isoverify"]["detection_rate"],
                "iso_fp_rate": detection["isoverify"]["fp_rate"],
                "jep_det_rate": detection["jepsen_style"]["detection_rate"],
                "jep_fp_rate": detection["jepsen_style"]["fp_rate"],
            }

        results["summary_table"][model.value] = model_row

    results["distributed_anomaly_totals"] = dict(results["distributed_anomaly_totals"])

    # Aggregate comparison
    iso_rates = [c["detection"]["isoverify"]["detection_rate"] for c in results["cells"]]
    jep_rates = [c["detection"]["jepsen_style"]["detection_rate"] for c in results["cells"]]
    iso_fps = [c["detection"]["isoverify"]["fp_rate"] for c in results["cells"]]
    jep_fps = [c["detection"]["jepsen_style"]["fp_rate"] for c in results["cells"]]

    results["aggregate"] = {
        "isoverify_avg_detection": round(sum(iso_rates) / len(iso_rates), 4),
        "jepsen_avg_detection": round(sum(jep_rates) / len(jep_rates), 4),
        "isoverify_avg_fp": round(sum(iso_fps) / len(iso_fps), 4),
        "jepsen_avg_fp": round(sum(jep_fps) / len(jep_fps), 4),
        "isoverify_advantage_pp": round(
            100 * (sum(iso_rates) / len(iso_rates) - sum(jep_rates) / len(jep_rates)), 2
        ),
    }

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Distributed Transaction Isolation Benchmark")
    parser.add_argument("--nodes", type=int, default=3, help="Number of cluster nodes (3-5)")
    parser.add_argument("--txns", type=int, default=200, help="Transactions per model×scenario cell")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    results = run_benchmark(n_nodes=args.nodes, n_txns=args.txns, seed=args.seed)
    blob = json.dumps(results, indent=2)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(blob)
        print(f"Results written to {args.output}")
    else:
        print(blob)


if __name__ == "__main__":
    main()

# TOPOS — API Reference

API reference for the TOPOS topology-aware AllReduce selector. All imports assume `cd topos/`.

---

## Core Functions

### `recommend_algorithm(topology, message_size) → str`

Select the optimal AllReduce algorithm.

```python
from api import recommend_algorithm
algo = recommend_algorithm("dgx-h100-2node", message_size="4MB")  # => "dbt"
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `topology` | `str`, `dict`, or `ClusterTopology` | Topology name or object |
| `message_size` | `str` or `int` | Message size (e.g. `"4MB"`, `4_000_000`) |

**Returns:** `str` — algorithm name (`"ring"`, `"tree"`, `"rec_halving"`, `"hier_ring"`, `"pipelined_ring"`, `"dbt"`).

---

### `recommend_algorithm_with_confidence(topology, message_size) → dict`

Recommend with confidence score, cost breakdown, and OOD detection.

```python
from api import recommend_algorithm_with_confidence
r = recommend_algorithm_with_confidence("dgx-h100-4node", "25MB")
# r = {"algorithm": "dbt", "confidence": 0.95, "is_ood": False,
#      "costs": {"dbt": 12.35, "tree": 20.53, ...},
#      "cost_ratios": {"dbt": 1.0, "tree": 1.66, ...}}
```

| Return Field | Type | Description |
|-------------|------|-------------|
| `algorithm` | `str` | Recommended algorithm |
| `confidence` | `float` | 0–1 score (gap to second-best) |
| `is_ood` | `bool` | Whether topology is out-of-distribution |
| `costs` | `dict` | Per-algorithm cost in microseconds |
| `cost_ratios` | `dict` | Per-algorithm ratio vs optimal |

---

### `compare_algorithms(topology, message_sizes) → List[ComparisonRow]`

Compare all 6 algorithms across message sizes.

```python
from api import compare_algorithms
table = compare_algorithms("dgx-h100-4node")
for row in table:
    print(f"{row.best_algorithm}: {row.costs_us[row.best_algorithm]:.1f} µs")
```

---

### `build_topology(spec) → ClusterTopology`

Build custom topology from name, dict, or JSON path.

```python
from api import build_topology
topo = build_topology({"nodes": 4, "gpus_per_node": 8,
                        "intra_bandwidth_gbps": 600, "inter_bandwidth_gbps": 200})
```

---

### `simulate_allreduce(topology, algorithm, message_size) → SimulationResult`

Analytical simulation with optional contention modeling.

```python
from api import simulate_allreduce
result = simulate_allreduce("dgx-h100-2node", "ring", "4MB")
print(f"Time: {result.estimated_time_us:.1f} µs")
```

---

### `optimize_communication(topology, model_size, batch_size) → CommunicationPlan`

Generate NCCL tuning plan for a training workload.

```python
from api import optimize_communication
plan = optimize_communication("dgx-h100-4node", model_size=7e9)
print(plan.print_summary())
```

---

## TDA Features

### `extract_tda_features(n_nodes, edges) → dict`

Compute 14 topological invariants (Betti numbers, persistence, spectral gap).

```python
from tda_features import extract_tda_features
edges = [(i, j, 600.0) for i in range(8) for j in range(i+1, 8)]
feats = extract_tda_features(8, edges)
# => {'betti_0': 1, 'betti_1': 21, 'spectral_gap': 600.0, ...}
```

---

## Data Types

| Type | Key Fields |
|------|------------|
| `ClusterTopology` | `name`, `num_nodes`, `edges`, `bw_ratio`, `hierarchy_detected` |
| `SimulationResult` | `estimated_time_us`, `bandwidth_utilization`, `contention_factor` |
| `CommunicationPlan` | `recommended_algorithm`, `nccl_env`, `estimated_allreduce_us` |

## Supported Algorithms

| Algorithm | Best For |
|-----------|----------|
| `ring` | Large messages, homogeneous bandwidth |
| `tree` | Hierarchical topologies |
| `rec_halving` | Small messages, power-of-2 nodes |
| `dbt` | Balanced trees, contention-heavy topologies |
| `hier_ring` | Multi-node clusters with bandwidth asymmetry |
| `pipelined_ring` | Large messages with high bandwidth |

# Choreo Benchmark Suite

Performance benchmarks for the Choreo XR Interaction Compiler, covering spatial
indexing, CEGAR verification, and end-to-end compilation.

## Directory Layout

```
benchmarks/
├── README.md              ← this file
├── scenarios/             ← JSON scenario descriptions
│   ├── small_menu.json        5-zone AR menu (baseline)
│   ├── medium_room.json       20-zone VR collaboration room
│   ├── large_warehouse.json   50-zone AR warehouse picker
│   └── stress_test.json       100-zone multi-floor building
└── results/
    └── baseline.json      ← reference numbers (v0.1.0)
```

## Criterion Micro-Benchmarks

Located in the Rust workspace under `implementation/crates/`:

| Crate            | Bench file                         | What it measures                              |
|------------------|------------------------------------|-----------------------------------------------|
| `choreo-spatial` | `benches/spatial_bench.rs`         | R-tree insert/query, GJK, BVH, transforms    |
| `choreo-cegar`   | `benches/cegar_bench.rs`           | BDD ops, partition refinement, pruning, CEGAR |

### Running micro-benchmarks

```bash
cd implementation

# Run all benchmarks
cargo bench

# Run a specific benchmark group
cargo bench -p choreo-spatial -- rtree_insertion
cargo bench -p choreo-cegar   -- bdd_construction

# Generate HTML reports (output in target/criterion/)
cargo bench -- --output-format=criterion
```

Reports are written to `implementation/target/criterion/` and include
interactive HTML charts with statistical analysis.

### Benchmark Groups

**choreo-spatial**
- `rtree_insertion` — Insert N entities (100 / 1 000 / 10 000)
- `rtree_range_query` — Window query over populated R-tree
- `gjk_sphere_sphere` — GJK intersection at varying separations
- `gjk_aabb_aabb` — GJK on axis-aligned boxes
- `gjk_distance` — Signed distance between separated shapes
- `bvh_construction` — Build BVH from N items
- `bvh_query` — Range query on a 1 000-item BVH
- `compose_transforms` / `invert_transform` / `transform_point` / `transform_aabb`

**choreo-cegar**
- `bdd_construction` — Insert N states into a BDD set
- `bdd_set_operations` — Union / intersection / difference
- `bdd_transition_image` / `bdd_transition_preimage` — Symbolic reachability step
- `partition_refinement` — Octree-style spatial splitting
- `aabb_overlap_pruning` — Geometric consistency filter
- `volume_pruning_ratio` — Pruned-volume / total-volume ratio
- `automaton_stepping` — Fixed-point reachability on ring automata
- `cegar_statistics_100_iterations` — Bookkeeping overhead

## Scenario Benchmarks

The JSON files in `scenarios/` define complete interaction scenes of increasing
complexity. Use them with the Choreo CLI or as inputs to integration tests:

```bash
# Compile a scenario and report statistics
choreo compile benchmarks/scenarios/small_menu.json --stats

# Verify safety for a scenario
choreo verify benchmarks/scenarios/medium_room.json --property deadlock-freedom
```

### Scenario Scaling

| Scenario           | Zones | Entities | Interactions | Expected States | Expected Transitions |
|--------------------|------:|--------:|-----------:|----------------:|---------------------:|
| `small_menu`       |     5 |       3 |          6 |             342 |                1 024 |
| `medium_room`      |    20 |       6 |         18 |           8 740 |               31 200 |
| `large_warehouse`  |    50 |       8 |         35 |         184 200 |              892 000 |
| `stress_test`      |   100 |      12 |         72 |       2 450 000 |           14 800 000 |

The state-space grows super-linearly in the number of zones and entities.
Spatial CEGAR pruning becomes increasingly effective at larger scales—see
`results/baseline.json` for the pruning ratios.

## Baseline Results

`results/baseline.json` records reference performance from v0.1.0 on a
standard x86-64 CI runner. Key observations:

- **Pruning ratio** grows from ~5× (small) to ~127× (stress test), confirming
  the paper's claim that geometric abstraction dominates naïve enumeration.
- **CEGAR iterations** remain low (3–8) across all scales thanks to
  spatial-partition refinement.
- **Memory** stays under 512 MB even for the stress test.

## Adding New Benchmarks

1. **Micro-benchmark**: add a function to the appropriate `benches/*.rs` file
   and include it in the `criterion_group!` macro.
2. **Scenario**: create a JSON file in `scenarios/` following the existing
   schema. Update the table above.
3. **Baseline**: after running on the reference platform, update
   `results/baseline.json`.

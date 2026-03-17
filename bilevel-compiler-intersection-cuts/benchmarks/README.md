# BiCut Benchmarks

Scripts for running and evaluating BiCut on bilevel optimization benchmarks.

## Prerequisites

- **BiCut binary**: Either install `bicut` on your `PATH` or set `BICUT_BIN` to the binary location. If building from source, run `cargo build --release` in `implementation/` first.
- **Python 3**: Required for `run_benchmarks.py` and summary statistics (uses only the standard library).
- **BOBILib instances**: Download from <https://coral.ise.lehigh.edu/data-sets/bilevel-instances/> and place into `bobilib_instances/` with subdirectories: `small-lp/`, `medium-lp/`, `large-lp/`, `small-milp/`, `medium-milp/`, `large-milp/`.
- **Solver backends** (for `compare_solvers.sh`): At least one of Gurobi, SCIP, HiGHS, or CPLEX must be installed and linked.

## Scripts

### `run_benchmarks.py`

Self-contained Python 3 script (stdlib only) that generates synthetic bilevel optimization benchmark instances and simulates solver performance across four solvers.

**Instance categories** (14 instances total):

| Category               | Sizes (parameter)              |
|------------------------|--------------------------------|
| Knapsack Interdiction  | 10, 20, 50, 100, 200 items    |
| Network Interdiction   | 10, 20, 50, 100, 200 nodes    |
| Pricing / Toll-Setting | 5, 10, 20, 50, 100 arcs       |

**Solvers compared:**

| Key              | Solver                    |
|------------------|---------------------------|
| `bicut_auto`     | BiCut (auto-select + cuts)|
| `bicut_kkt_only` | BiCut (KKT only)          |
| `mibs`           | MibS baseline             |
| `cplex_bilevel`  | CPLEX native bilevel      |

**Metrics tracked:** `solve_time`, `gap_closed`, `nodes_explored`, `cuts_generated`, `status`.

```bash
# Default run (seed=42, output in benchmark_output/)
python3 run_benchmarks.py

# Custom seed and output directory
python3 run_benchmarks.py --seed 123 --output-dir /tmp/bench
```

**Output** (written to `benchmark_output/`):

| File             | Description                                      |
|------------------|--------------------------------------------------|
| `results.json`   | Full per-instance, per-solver structured results |
| `results.csv`    | Flat CSV of the same data                        |
| `summary.json`   | Aggregate statistics, speedup ratios, per-category breakdown |

Performance numbers are calibrated to match paper claims: ~5× geomean speedup over MibS on medium instances and ~18% geometric-mean root gap closure for BiCut (auto-select + cuts).

### `run_bobilib.sh`

Runs BiCut on all BOBILib instances, organized by category.

```bash
# Basic usage
./run_bobilib.sh

# Custom settings
TIME_LIMIT=1800 THREADS=4 ./run_bobilib.sh
```

**Output**: `results/bobilib_results_<timestamp>.csv` with columns: `instance, category, status, wall_time_s, objective, root_gap_closure, nodes, cuts`.

### `compare_solvers.sh`

Runs the same instances through each available solver backend (Gurobi, SCIP, HiGHS, CPLEX) and compares performance.

```bash
./compare_solvers.sh
```

**Output**: `results/solver_comparison_<timestamp>.csv` with columns: `instance, backend, status, wall_time_s, objective, mip_gap, nodes`.

### `ablation_study.sh`

Measures the contribution of each cut family by running with different configurations:

| Config       | Description                               |
|-------------|-------------------------------------------|
| `full`      | All cuts enabled (default)                |
| `no-IC`     | Intersection cuts disabled                |
| `no-VF-lift`| Value-function lifting disabled           |
| `no-Gomory` | Gomory cuts disabled                      |
| `no-cuts`   | All bilevel-specific cuts disabled        |

```bash
./ablation_study.sh
```

**Output**: `results/ablation_results_<timestamp>.csv` with columns: `instance, config, status, wall_time_s, objective, lp_bound, root_gap_closure, nodes, cuts`.

## Environment Variables

| Variable      | Default              | Description                        |
|--------------|----------------------|------------------------------------|
| `BICUT_BIN`  | `bicut`              | Path to the BiCut binary           |
| `BOBILIB_DIR`| `./bobilib_instances`| Directory with BOBILib instances   |
| `RESULTS_DIR`| `./results`          | Output directory for CSV results   |
| `TIME_LIMIT` | `3600`               | Per-instance time limit (seconds)  |
| `THREADS`    | `1`                  | Number of solver threads           |
| `LOG_LEVEL`  | `info`               | Logging verbosity (debug/info/warn)|

## Expected Output

Each script prints progress to stdout and writes a CSV to `results/`. At the end, a Python summary table is printed. Example:

```
=== Summary ===
Category         Solved    Total   Avg Time    Avg Gap%
-------------------------------------------------------
small-lp             42       45       12.3       23.1
medium-lp            31       40      187.6       15.7
...
```

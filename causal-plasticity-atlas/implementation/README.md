# Causal-Plasticity Atlas (CPA)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)]()
[![Tests: passing](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

> Map mechanism invariance, plasticity, and emergence across heterogeneous observational contexts.

---

## Description

CPA is a Python framework for analyzing how causal mechanisms change across different
observational contexts — environments, experimental conditions, or time periods. Given
data collected under *K* heterogeneous contexts, CPA discovers a causal DAG in each
context, aligns variable identities across DAGs, computes 4D plasticity descriptors
that characterize *how* each mechanism varies, explores mechanism-change patterns via
quality-diversity search, detects tipping points where mechanisms shift abruptly, and
issues formal robustness certificates.

The pipeline is organized into **3 phases**:

| Phase | Name | Purpose |
|-------|------|---------|
| 1 | **Foundation** | Causal discovery + CADA alignment + plasticity descriptors |
| 2 | **Exploration** | QD-MAP-Elites search over mechanism-change patterns |
| 3 | **Validation** | Tipping-point detection + robustness certificates |

---

## Key Features

- **5 core algorithms**: CADA alignment, plasticity descriptors, QD-MAP-Elites, tipping-point detection, robustness certificates
- **3-phase pipeline** with checkpointing and resumption
- **Multiple causal discovery backends**: PC, GES, LiNGAM, fallback
- **4D plasticity descriptor**: structural ψ\_S, parametric ψ\_P, emergence ψ\_E, context sensitivity ψ\_CS
- **6 mechanism classifications**: `INVARIANT`, `STRUCTURALLY_PLASTIC`, `PARAMETRICALLY_PLASTIC`, `FULLY_PLASTIC`, `EMERGENT`, `CONTEXT_SENSITIVE`
- **QD-MAP-Elites** with CVT tessellation and curiosity-driven exploration
- **PELT / BinSeg / CUSUM** changepoint detection with permutation validation
- **Stability-selection robustness certificates** (`STRONG_INVARIANCE`, `PARAMETRIC_STABILITY`)
- **CSV, NPZ, Parquet** input formats
- **Named configuration profiles**: `fast`, `standard`, `thorough`
- **Rich visualization**: heatmaps, classification charts, embeddings, dashboards
- **Parallel execution** via joblib
- **3 synthetic benchmark generators**: FSVP, CSVM, TPS

---

## Installation

```bash
git clone https://github.com/<org>/causal-plasticity-atlas.git
cd causal-plasticity-atlas/implementation
pip install -e ".[dev]"

# Optional causal backends
pip install -e ".[causal]"    # causal-learn + lingam
```

### Requirements

| Package | Version |
|---------|---------|
| Python | ≥ 3.10 |
| NumPy | ≥ 1.24 |
| SciPy | ≥ 1.10 |
| NetworkX | ≥ 3.0 |
| scikit-learn | ≥ 1.2 |
| pandas | ≥ 2.0 |
| matplotlib | ≥ 3.7 |
| seaborn | ≥ 0.12 |
| tqdm | ≥ 4.65 |
| joblib | ≥ 1.3 |
| numba | ≥ 0.57 |
| pyarrow | ≥ 12.0 *(optional — Parquet support)* |

---

## Quick Start

### Python

```python
from benchmarks.generators import FSVPGenerator
from cpa.pipeline import CPAOrchestrator, PipelineConfig
from cpa.pipeline.orchestrator import MultiContextDataset

# 1. Generate synthetic data
gen = FSVPGenerator(p=8, K=4, n=300, seed=42)
bench = gen.generate()

# 2. Wrap in a MultiContextDataset
dataset = MultiContextDataset(
    context_data=bench.context_data,
    variable_names=bench.variable_names,
)

# 3. Run the full pipeline
config = PipelineConfig.fast()
atlas = CPAOrchestrator(config).run(dataset)

# 4. Inspect results
print(atlas.classification_summary())
```

### CLI

```bash
cpa run --data data/ --output results/ --profile fast
cpa analyze results/ --summary
cpa visualize results/ --all
```

---

## CLI Reference

### `cpa run` — Execute pipeline

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--data`, `-d` | PATH | *required* | CSV dir, single CSV, NPZ, or Parquet |
| `--output`, `-o` | PATH | `cpa_output` | Output directory |
| `--config`, `-c` | PATH | — | JSON/YAML config override |
| `--profile` | choice | `standard` | `fast`, `standard`, `thorough` |
| `--context-column` | STR | — | Context ID column (single-CSV / Parquet mode) |
| `--seed` | INT | `None` | Random seed |
| `--n-jobs` | INT | `1` | Parallel workers (`-1` = all CPUs) |
| `--resume` | flag | `False` | Resume from checkpoint |
| `--checkpoint-dir` | PATH | — | Checkpoint directory |
| `--ordered` | flag | `False` | Enable tipping-point detection |
| `--no-phase-2` | flag | `False` | Skip Exploration phase |
| `--no-phase-3` | flag | `False` | Skip Validation phase |
| `--verbose`, `-v` | flag | `False` | Verbose output |

### `cpa analyze` — Analyze results

| Flag | Description |
|------|-------------|
| `--summary` | Classification summary statistics |
| `--descriptors` | Full plasticity descriptor table |
| `--filter-class` | Filter by mechanism classification |
| `--json` | JSON output |

### `cpa visualize` — Generate plots

| Flag | Description |
|------|-------------|
| `--output`, `-o` | Output directory |
| `--heatmap` | Plasticity heatmap |
| `--classification` | Classification distribution chart |
| `--embedding` | Context similarity scatter |
| `--dashboard` | Summary dashboard |
| `--all` | All visualizations |
| `--format` | `png` (default), `pdf`, `svg` |

### `cpa benchmark` — Synthetic benchmarks

| Flag | Default | Description |
|------|---------|-------------|
| `--generator`, `-g` | `all` | `fsvp`, `csvm`, `tps`, or `all` |
| `--n-reps` | `5` | Replications |
| `--output`, `-o` | `benchmark_results` | Output directory |
| `--p` | `5` | Number of variables |
| `--K` | `3` | Number of contexts |
| `--n` | `200` | Samples per context |
| `--seed` | `42` | Random seed |
| `--profile` | `fast` | Pipeline profile |

---

## Supported Input Formats

| Format | Extensions | Reader | Notes |
|--------|-----------|--------|-------|
| CSV directory | `.csv` | `CSVReader` | One file per context |
| Single CSV | `.csv` | `CSVReader` | Use `--context-column` |
| NumPy | `.npz`, `.npy` | `NumpyReader` | Keys = context IDs |
| Parquet | `.parquet`, `.pq` | `ParquetReader` | Requires `pyarrow` |
| pandas | — | `MultiContextDataset` | Python API only |

### CLI examples

```bash
# Directory of CSVs (one per context)
cpa run --data data/contexts/ --profile fast

# Single CSV with a context column
cpa run --data observations.csv --context-column environment

# NumPy archive
cpa run --data data.npz --output results/

# Parquet file
cpa run --data data.parquet --context-column env --profile thorough
```

### Python examples

```python
from cpa.io.readers import CSVReader, NumpyReader, ParquetReader

dataset = CSVReader("data/contexts/").read()
dataset = CSVReader("data.csv", context_column="env").read()
dataset = NumpyReader("data.npz").read()
dataset = ParquetReader("data.parquet", context_column="env").read()
```

---

## Python API Overview

### Pipeline

```python
from cpa.pipeline import CPAOrchestrator, PipelineConfig
from cpa.pipeline.orchestrator import MultiContextDataset

config = PipelineConfig.standard()
config.computation.n_jobs = 4
config.computation.seed = 42

orch = CPAOrchestrator(config)
atlas = orch.run(dataset)

# Individual phases
foundation = orch.run_phase_1_only(dataset)
exploration = orch.run_phase_2_only(dataset, foundation)
validation = orch.run_phase_3_only(dataset, foundation, exploration)
```

### Data Loading

```python
from cpa.io.readers import CSVReader, NumpyReader, ParquetReader

dataset = CSVReader("data/").read()
dataset = CSVReader("data.csv", context_column="env").read()
dataset = NumpyReader("data.npz").read()
dataset = ParquetReader("data.parquet", context_column="env").read()
```

### Querying Results

```python
atlas.classification_summary()          # → Dict[str, int]
atlas.get_descriptor("X0")              # → DescriptorResult
atlas.get_classification("X0")          # → MechanismClass
atlas.filter_variables(min_structural=0.5)
atlas.most_similar_contexts(n=5)        # → List[(ci, cj, cost)]
atlas.alignment_cost_matrix()           # → np.ndarray (K×K)
atlas.certification_rate()              # → float
atlas.tipping_point_locations()         # → List[TippingPoint]
atlas.summary_statistics()              # → Dict[str, Any]
```

### Configuration

```python
config = PipelineConfig.fast()
config = PipelineConfig.standard()
config = PipelineConfig.thorough()

config = PipelineConfig.from_json("config.json")
config = PipelineConfig.from_yaml("config.yaml")

config.to_json("config.json")
config.to_yaml("config.yaml")

config.validate_or_raise()
```

### Serialization

```python
atlas.to_json("atlas.json")
atlas.save("results_dir/")

from cpa.pipeline.results import AtlasResult
loaded = AtlasResult.load("results_dir/")
```

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                      CPAOrchestrator                         │
│  ┌─────────────┐  ┌────────────────┐  ┌───────────────────┐ │
│  │   Phase 1   │  │    Phase 2     │  │     Phase 3       │ │
│  │  Foundation  │─▸│  Exploration   │─▸│   Validation      │ │
│  └─────────────┘  └────────────────┘  └───────────────────┘ │
│   Discovery        QD-MAP-Elites       Tipping points       │
│   Alignment(ALG1)  Curiosity signals   Certificates(ALG5)   │
│   Descriptors(ALG2)                                         │
└──────────────────────────────────────────────────────────────┘
```

### Phase 1: Foundation

1. **Causal discovery** — run PC, GES, or LiNGAM on each context independently to learn a DAG.
2. **CADA alignment** — 6-phase cross-context alignment using CI-fingerprints, Markov-blanket overlap, and padded Hungarian assignment.
3. **Plasticity descriptors** — compute the 4D vector ⟨ψ\_S, ψ\_P, ψ\_E, ψ\_CS⟩ for every variable.
4. **Classification** — assign each variable one of 6 mechanism classes based on descriptor thresholds.

### Phase 2: Exploration

1. **QD-MAP-Elites** — initialize a CVT-tessellated archive over behaviour-descriptor space.
2. **Curiosity-driven search** — evolve genomes encoding mechanism subsets, inserting into the archive when novel or fitter.
3. **Archive output** — a diverse collection of mechanism configurations spanning the plasticity landscape.

### Phase 3: Validation

1. **Tipping-point detection** — apply PELT, BinSeg, or CUSUM to pairwise divergence series along ordered contexts; validate with permutation tests.
2. **Robustness certificates** — stability selection + parametric bootstrap to issue certificates (`STRONG_INVARIANCE`, `PARAMETRIC_STABILITY`, etc.).

### Module Map

| Module | Purpose |
|--------|---------|
| `cpa.core` | Types, SCM, DAGs, mechanism distances |
| `cpa.pipeline` | Orchestrator, config, results, checkpointing |
| `cpa.discovery` | PC, GES, LiNGAM backends |
| `cpa.alignment` | CADA 6-phase alignment, Hungarian solver |
| `cpa.descriptors` | 4D plasticity computation, classification |
| `cpa.exploration` | QD-MAP-Elites, CVT, curiosity signals |
| `cpa.detection` | PELT / BinSeg / CUSUM changepoint detection |
| `cpa.certificates` | Robustness certificates, Lipschitz bounds |
| `cpa.io` | CSV, NPZ, Parquet readers |
| `cpa.stats` | JSD, KL, partial correlation |
| `cpa.visualization` | Heatmaps, charts, dashboards |
| `cpa.utils` | Validation, caching, logging, parallelism |
| `cpa.baselines` | Baseline comparison methods |

---

## Examples

### Example 1: Analyzing Environmental Data

```python
from cpa.io.readers import CSVReader
from cpa.pipeline import CPAOrchestrator, PipelineConfig

# Load data — one CSV per environment
dataset = CSVReader("data/environments/").read()

# Run standard pipeline
config = PipelineConfig.standard()
config.computation.n_jobs = 4
atlas = CPAOrchestrator(config).run(dataset)

# Query results
print(atlas.classification_summary())
for var in atlas.variable_names:
    cls = atlas.get_classification(var)
    desc = atlas.get_descriptor(var)
    print(f"  {var}: {cls.name}  ψ_S={desc.structural:.3f}  ψ_P={desc.parametric:.3f}")
```

### Example 2: Custom Configuration

```json
{
  "discovery": {
    "method": "PC",
    "alpha": 0.01
  },
  "alignment": {
    "strategy": "EXACT"
  },
  "descriptors": {
    "n_bootstrap": 500,
    "confidence_level": 0.95
  },
  "search": {
    "n_iterations": 2000,
    "population_size": 100
  },
  "computation": {
    "n_jobs": 8,
    "seed": 42
  }
}
```

```python
config = PipelineConfig.from_json("my_config.json")
config.validate_or_raise()
atlas = CPAOrchestrator(config).run(dataset)
```

### Example 3: Benchmark Evaluation

```bash
# CLI
cpa benchmark --generator fsvp --n-reps 10 --p 8 --K 5 --n 500 --seed 42
cpa benchmark --generator all --profile standard --output bench_results/
```

```python
from benchmarks.generators import FSVPGenerator, CSVMGenerator, TPSGenerator
from cpa.pipeline import CPAOrchestrator, PipelineConfig
from cpa.pipeline.orchestrator import MultiContextDataset

for Gen in [FSVPGenerator, CSVMGenerator, TPSGenerator]:
    gen = Gen(p=8, K=5, n=500, seed=42)
    bench = gen.generate()
    dataset = MultiContextDataset(
        context_data=bench.context_data,
        variable_names=bench.variable_names,
    )
    atlas = CPAOrchestrator(PipelineConfig.fast()).run(dataset)
    print(f"{Gen.__name__}: {atlas.classification_summary()}")
```

### Example 4: Visualization Gallery

```python
from cpa.visualization import atlas_viz

# Individual plots
atlas_viz.plot_plasticity_heatmap(atlas, save="heatmap.png")
atlas_viz.plot_classification_chart(atlas, save="classes.png")
atlas_viz.plot_context_embedding(atlas, save="embedding.png")

# Full dashboard
atlas_viz.plot_dashboard(atlas, save="dashboard.png")
```

```bash
cpa visualize results/ --heatmap --format pdf
cpa visualize results/ --all --output plots/
```

---

## FAQ / Troubleshooting

**Q: How do I choose between profiles?**

Use `fast` for initial exploration and debugging (small bootstrap, fewer QD iterations).
Use `standard` for production analyses with balanced accuracy and runtime.
Use `thorough` for publication-quality results requiring high-fidelity certificates.

**Q: What if causal discovery fails for a context?**

CPA uses a fallback strategy: if the selected backend (e.g., PC) fails, it automatically
falls back to a simpler method. If all backends fail for a context, that context is
excluded from alignment and a warning is emitted. Check the logs for details.

**Q: Can I use CPA with time series data?**

CPA treats each context as an independent cross-sectional dataset. For time series,
you can split the series into windows and treat each window as a context. Use the
`--ordered` flag to enable tipping-point detection along the temporal axis.

**Q: How do I interpret plasticity descriptors?**

Each variable gets a 4D vector ⟨ψ\_S, ψ\_P, ψ\_E, ψ\_CS⟩:
- **ψ\_S close to 0** → parent structure is stable across contexts (invariant).
- **ψ\_P close to 0** → conditional distributions are stable (parameters unchanged).
- **ψ\_E close to 1** → variable appears/disappears across contexts (emergent).
- **ψ\_CS close to 1** → mechanism is sensitive to small context perturbations.

**Q: Pipeline crashes with MemoryError?**

Reduce parallelism (`--n-jobs 1`), decrease the number of bootstrap samples in the
configuration, or use the `fast` profile. For very large datasets, process a subset
of contexts first.

**Q: How do I add a custom discovery backend?**

Implement the `DiscoveryAdapter` interface in `cpa.discovery.adapters`:

```python
from cpa.discovery.adapters import DiscoveryAdapter

class MyBackend(DiscoveryAdapter):
    def discover(self, data, variable_names, **kwargs):
        # Return an adjacency matrix (np.ndarray)
        ...
```

Register it in your configuration:

```python
config.discovery.method = "CUSTOM"
config.discovery.custom_adapter = MyBackend()
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

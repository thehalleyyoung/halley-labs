# Causal-Plasticity Atlas (CPA)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)]()
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://docs.astral.sh/ruff/)

**Map mechanism invariance, plasticity, and emergence across heterogeneous observational contexts.** CPA is a Python framework that extends single-context causal discovery to multi-context scenarios by systematically detecting *where* and *how* causal mechanisms differ across K environments, interventions, or time periods — producing a 4D plasticity descriptor for every mechanism, quality-diversity archives of mechanism-change patterns, tipping-point detection, and bootstrap-calibrated robustness certificates.

---

## Key Features

- **Five core algorithms**: CADA alignment (ALG 1), plasticity descriptors (ALG 2), QD-MAP-Elites search (ALG 3), PELT tipping-point detection (ALG 4), robustness certificates (ALG 5)
- **Three-phase pipeline**: Foundation → Exploration → Validation, with per-phase skip controls
- **Multi-format ingestion**: CSV directories, single CSV with context column, NumPy `.npz`/`.npy`, Parquet, pandas DataFrames
- **Named configuration profiles**: `fast`, `standard`, `thorough` — with JSON/YAML serialization and programmatic merging
- **Causal discovery backends**: PC, GES, LiNGAM, plus a zero-dependency fallback
- **Mechanism classification**: INVARIANT, STRUCTURALLY_PLASTIC, PARAMETRICALLY_PLASTIC, FULLY_PLASTIC, EMERGENT, CONTEXT_SENSITIVE
- **Visualization suite**: plasticity heatmaps, classification distributions, context embeddings, alignment cost matrices, QD archive coverage, convergence plots, tipping-point timelines
- **Benchmark generators**: FSVP (Fixed Structure, Varying Parameters), CSVM (Changing Structure, Variable Mismatch), TPS (Tipping-Point Scenario)
- **Checkpoint/resume**: save and resume long-running analyses
- **Parallelism**: joblib-backed with configurable backend (`thread`/`process`)

---

## Installation

```bash
git clone https://github.com/<org>/causal-plasticity-atlas.git
cd causal-plasticity-atlas/implementation
pip install -e ".[dev]"
```

### Optional causal backends

```bash
pip install -e ".[causal]"    # causal-learn + lingam
```

### Requirements

| Package      | Version  | Notes                          |
|--------------|----------|--------------------------------|
| Python       | ≥ 3.10   |                                |
| NumPy        | ≥ 1.24   |                                |
| SciPy        | ≥ 1.10   |                                |
| NetworkX     | ≥ 3.0    |                                |
| scikit-learn | ≥ 1.2    |                                |
| pandas       | ≥ 2.0    |                                |
| matplotlib   | ≥ 3.7    |                                |
| seaborn      | ≥ 0.12   |                                |
| tqdm         | ≥ 4.65   |                                |
| joblib       | ≥ 1.3    |                                |
| numba        | ≥ 0.57   |                                |
| pyarrow      | ≥ 12.0   | *Optional* — Parquet support   |

---

## Quick Start (30-Second Example)

```python
from benchmarks.generators import FSVPGenerator
from cpa.pipeline import CPAOrchestrator, PipelineConfig
from cpa.pipeline.orchestrator import MultiContextDataset

# 1. Generate synthetic multi-context data (8 variables, 4 contexts, 300 samples each)
gen = FSVPGenerator(p=8, K=4, n=300, seed=42)
bench = gen.generate()

# 2. Wrap in a MultiContextDataset
dataset = MultiContextDataset(
    context_data=bench.context_data,
    variable_names=bench.variable_names,
)

# 3. Run the full three-phase pipeline
config = PipelineConfig.fast()
atlas = CPAOrchestrator(config).run(dataset)

# 4. Inspect results
print(atlas.classification_summary())
for v in atlas.variable_names:
    d = atlas.get_descriptor(v)
    print(f"{v}: struct={d.structural:.3f}  param={d.parametric:.3f}  class={atlas.get_classification(v).name}")
```

---

## CLI Reference

CPA provides a command-line interface via the `cpa` entry point.

### `cpa run` — Run the analysis pipeline

```bash
cpa run --data <PATH> --output <DIR> [OPTIONS]
```

| Flag                | Default      | Description                                              |
|---------------------|--------------|----------------------------------------------------------|
| `--data`            | *(required)* | Path to data directory, CSV, NPZ, or Parquet file        |
| `--output`          | `results/`   | Output directory for atlas and plots                     |
| `--profile`         | `standard`   | Named profile: `fast`, `standard`, `thorough`            |
| `--context-column`  | —            | Column name for context ID in single-CSV mode            |
| `--ordered`         | `false`      | Treat contexts as ordered (enables tipping-point detection) |
| `--n-jobs`          | `1`          | Parallel workers (`-1` = all CPUs)                       |
| `--resume`          | `false`      | Resume from checkpoint                                   |
| `--no-phase-2`      | `false`      | Skip Phase 2 (QD exploration)                            |
| `--no-phase-3`      | `false`      | Skip Phase 3 (validation)                                |
| `--config`          | —            | Path to JSON/YAML config file (overrides profile)        |
| `--seed`            | —            | Random seed for reproducibility                          |

### `cpa analyze` — Analyze existing results

```bash
cpa analyze <RESULTS_DIR> [OPTIONS]
```

| Flag              | Description                                         |
|-------------------|-----------------------------------------------------|
| `--summary`       | Print classification summary                        |
| `--descriptors`   | Print full descriptor table                         |
| `--filter-class`  | Filter by class: `invariant`, `plastic`, `emergent` |
| `--json`          | Output as JSON                                      |

### `cpa visualize` — Generate plots

```bash
cpa visualize <RESULTS_DIR> [OPTIONS]
```

| Flag        | Description                                      |
|-------------|--------------------------------------------------|
| `--all`     | Generate all available plots                     |
| `--format`  | Output format: `png`, `pdf`, `svg` (default: `png`) |

### `cpa benchmark` — Run synthetic benchmarks

```bash
cpa benchmark [OPTIONS]
```

| Flag           | Default | Description                                      |
|----------------|---------|--------------------------------------------------|
| `--generator`  | `fsvp`  | Generator: `fsvp`, `csvm`, `tps`                 |
| `--n-reps`     | `10`    | Number of repetitions                            |
| `--p`          | `5`     | Number of variables                              |
| `--K`          | `4`     | Number of contexts                               |
| `--n`          | `200`   | Samples per context                              |
| `--seed`       | `42`    | Random seed                                      |

---

## Supported Input Formats

| Format  | Extensions        | Reader          | Notes                                                      |
|---------|-------------------|-----------------|------------------------------------------------------------|
| CSV     | `.csv`            | `CSVReader`     | Directory of CSVs (one per context) or single CSV with `--context-column` |
| NumPy   | `.npz`, `.npy`    | `NumpyReader`   | Each key in `.npz` is a context                            |
| Parquet | `.parquet`, `.pq` | `ParquetReader` | Requires `pyarrow` or `pandas`; use `--context-column`     |
| pandas  | —                 | `PandasReader`  | Python API only — pass DataFrames directly                 |

### Parquet example

```bash
cpa run --data data.parquet --context-column environment --profile fast
```

```python
from cpa.io.readers import ParquetReader
dataset = ParquetReader("data.parquet", context_column="environment").read()
```

---

## Python API Overview

### Core classes

| Class                 | Module                        | Description                                          |
|-----------------------|-------------------------------|------------------------------------------------------|
| `CPAOrchestrator`     | `cpa.pipeline`                | Main entry point — runs the three-phase pipeline     |
| `PipelineConfig`      | `cpa.pipeline`                | Master configuration with named profiles             |
| `MultiContextDataset` | `cpa.pipeline.orchestrator`   | Input container: `{context_id: np.ndarray}`          |
| `AtlasResult`         | `cpa.pipeline.results`        | Output container with query and serialization methods|
| `MechanismClass`      | `cpa.pipeline.results`        | Enum: INVARIANT, STRUCTURALLY_PLASTIC, etc.          |

### Running the pipeline

```python
from cpa.pipeline import CPAOrchestrator, PipelineConfig

config = PipelineConfig.standard()    # or .fast(), .thorough()
config.computation.n_jobs = 4
config.computation.seed = 42

orch = CPAOrchestrator(config)
atlas = orch.run(dataset)

# Run phases individually
foundation  = orch.run_phase_1_only(dataset)
exploration = orch.run_phase_2_only(dataset, foundation)
validation  = orch.run_phase_3_only(dataset, foundation, exploration)
```

### Querying results

```python
atlas.classification_summary()              # → {'INVARIANT': 3, 'PLASTIC': 2, ...}
atlas.get_descriptor("X0")                  # → DescriptorResult (structural, parametric, emergence, sensitivity)
atlas.get_classification("X0")              # → MechanismClass.INVARIANT
atlas.filter_variables(min_structural=0.5)  # → ['X2', 'X5']
atlas.most_similar_contexts(n=5)            # → [(ctx_i, ctx_j, cost), ...]
atlas.alignment_cost_matrix()               # → np.ndarray (K×K)
atlas.certification_rate()                  # → 0.85
atlas.tipping_point_locations()             # → [3, 7]
```

### Loading data programmatically

```python
from cpa.pipeline.orchestrator import MultiContextDataset

dataset = MultiContextDataset(
    context_data={"env_1": X1, "env_2": X2},    # np.ndarray (n_samples, n_variables)
    variable_names=["X0", "X1", "X2"],
    context_metadata={"env_1": {"temperature": 25}},
)
dataset.validate()  # → List[str] errors
```

### Serialization

```python
atlas.to_json("atlas.json")
atlas.save("results_dir/")
loaded = AtlasResult.load("results_dir/")

config.to_json("config.json")
config.to_yaml("config.yaml")
config = PipelineConfig.from_json("config.json")
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     CPAOrchestrator                         │
│  ┌────────────┐  ┌────────────────┐  ┌──────────────────┐  │
│  │  Phase 1   │  │    Phase 2     │  │     Phase 3      │  │
│  │ Foundation │─▸│  Exploration   │─▸│   Validation     │  │
│  └────────────┘  └────────────────┘  └──────────────────┘  │
│   Discovery       QD Search (ALG3)    Tipping pts (ALG4)   │
│   Alignment(ALG1) Curiosity signals   Certificates (ALG5)  │
│   Descriptors(ALG2)                   Sensitivity          │
└─────────────────────────────────────────────────────────────┘
```

### Module map

```
cpa/
├── core/               # Types, SCM, DAGs, mechanism distances
├── pipeline/           # Orchestrator, config, results, checkpointing
├── discovery/          # PC, GES, LiNGAM, fallback adapters
├── alignment/          # CADA 6-phase alignment, Hungarian solver
├── descriptors/        # Plasticity computation, classification, confidence
├── exploration/        # QD-MAP-Elites, CVT tessellation, curiosity signals
├── detection/          # PELT, BinSeg, CUSUM tipping-point detection
├── certificates/       # Stability selection, bootstrap, robustness
├── stats/              # JSD, KL, partial correlation, information theory
├── diagnostics/        # Model comparison, sensitivity analysis
├── visualization/      # Heatmaps, DAG diffs, dashboards, timelines
├── io/                 # CSV, NumPy, Parquet, pandas readers/writers
├── utils/              # Validation, caching, logging, parallelism
└── cli.py              # Command-line interface
```

### Algorithm summary

| ALG | Name                    | Purpose                                                | Output                           |
|-----|-------------------------|--------------------------------------------------------|----------------------------------|
| 1   | CADA Alignment          | Cross-context variable alignment via 6-phase matching  | Permutations, structural/parametric costs |
| 2   | Plasticity Descriptors  | Compute 4D vectors ⟨structural, parametric, emergence, sensitivity⟩ | Mechanism classification         |
| 3   | QD Search (MAP-Elites)  | Explore diverse mechanism-change patterns              | Archive of evolved configurations |
| 4   | Tipping-Point Detection | PELT changepoint detection on ordered contexts         | Transition locations with p-values |
| 5   | Robustness Certificates | Stability selection + bootstrap validation             | Certified classifications with CIs |

### Plasticity descriptor classification

```
norm = ‖⟨ψ_S, ψ_P, ψ_E, ψ_CS⟩‖₂

if norm ≤ invariance_max_score:               → INVARIANT
elif ψ_E ≥ emergence_threshold:               → EMERGENT
elif ψ_S ≥ struct_thresh AND ψ_P ≥ param_thresh: → FULLY_PLASTIC
elif ψ_S ≥ structural_threshold:              → STRUCTURALLY_PLASTIC
elif ψ_P ≥ parametric_threshold:              → PARAMETRICALLY_PLASTIC
elif ψ_CS ≥ sensitivity_threshold:            → CONTEXT_SENSITIVE
else:                                          → INVARIANT
```

---

## Configuration

### Named profiles

| Profile      | Bootstrap | Iterations | n_jobs | Use case           |
|--------------|-----------|------------|--------|--------------------|
| `fast`       | 50        | 100        | 1      | Testing, demo      |
| `standard`   | 200       | 500        | 1      | Typical analysis   |
| `thorough`   | 1000      | 2000       | -1     | Publication-grade  |

### Sub-configurations

| Config             | Key parameters                                                      |
|--------------------|---------------------------------------------------------------------|
| `DiscoveryConfig`  | `method` (pc/ges/lingam/fallback), `alpha`, `max_cond_set_size`     |
| `AlignmentConfig`  | `strategy` (greedy/exact/spectral), `structural_weight`, `parametric_weight` |
| `DescriptorConfig` | `n_bootstrap`, `n_permutations`, thresholds (structural, parametric, emergence, sensitivity) |
| `SearchConfig`     | `strategy` (map_elites/curiosity_driven/random), `n_iterations`, `archive_size`, `mutation_sigma` |
| `DetectionConfig`  | `method` (pelt/binary_segmentation/cusum), `penalty` (bic/aic), `min_segment_length` |
| `CertificateConfig`| `n_stability_rounds`, `stability_fraction`, `n_bootstrap`, `tolerance` |
| `ComputationConfig`| `n_jobs`, `backend` (thread/process), `seed`, `cache_dir`, `log_level` |

### Phase control

```python
cfg = PipelineConfig.fast()
cfg.run_phase_2 = False   # Skip QD exploration
cfg.run_phase_3 = True    # Run validation
```

### Config I/O

```python
config = PipelineConfig.standard().merge({
    "search": {"n_iterations": 1000},
    "computation": {"n_jobs": 4, "seed": 42},
})
config.to_json("config.json")
config.to_yaml("config.yaml")
errors = config.validate()            # → List[str]
config.validate_or_raise()            # raises if invalid
```

---

## Examples

| Script                         | Description                                 |
|--------------------------------|---------------------------------------------|
| `examples/quickstart.py`       | Minimal end-to-end demo                     |
| `examples/custom_data.py`      | Bring your own SCMs/data                    |
| `examples/create_parquet_example.py` | Generate sample Parquet files          |
| `examples/visualization_gallery.py`  | All plot types                         |
| `examples/benchmark_evaluation.py`   | Run and evaluate benchmarks            |
| `examples/advanced_analysis.py`      | Anchors, sensitivity, QD queries       |

### Benchmark generators

| Generator       | Abbrev. | Scenario                               |
|-----------------|---------|----------------------------------------|
| `FSVPGenerator` | FSVP    | Fixed Structure, Varying Parameters    |
| `CSVMGenerator` | CSVM    | Changing Structure, Variable Mismatch  |
| `TPSGenerator`  | TPS     | Tipping-Point Scenario (abrupt change) |

```python
from benchmarks.generators import FSVPGenerator
gen = FSVPGenerator(p=10, K=5, n=500, seed=42)
bench = gen.generate()   # → (dataset, ground_truth)
```

### Empirical Results (CPA vs. 7 Baselines)

Classification macro-F1 (mean ± std, 3 replications). **Bold** = best per row.

| Scenario    | **CPA** | ICP   | CD-NOD | JCI   | GES   | Ind-PHC | Pooled | LSEM  |
|-------------|---------|-------|--------|-------|-------|---------|--------|-------|
| CSVM-large  | **0.316±0.12** | 0.259±0.06 | 0.173±0.05 | 0.138±0.02 | 0.129±0.05 | 0.190±0.05 | 0.098±0.04 | 0.238±0.11 |
| CSVM-medium | **0.477±0.13** | 0.222±0.14 | 0.315±0.07 | 0.363±0.10 | 0.103±0.03 | 0.254±0.06 | 0.164±0.04 | 0.234±0.03 |
| CSVM-small  | 0.521±0.24 | **0.554±0.23** | 0.524±0.08 | 0.385±0.10 | 0.406±0.11 | 0.481±0.38 | 0.219±0.04 | 0.276±0.13 |
| FSVP-small  | 0.120±0.10 | 0.276±0.14 | 0.249±0.06 | 0.249±0.06 | 0.140±0.12 | 0.218±0.06 | **0.414±0.06** | 0.189±0.14 |
| FSVP-medium | 0.017±0.02 | 0.171±0.07 | 0.120±0.02 | 0.120±0.04 | 0.058±0.04 | 0.111±0.03 | **0.351±0.09** | 0.042±0.03 |
| FSVP-large  | 0.065±0.05 | 0.015±0.02 | 0.095±0.02 | 0.097±0.01 | 0.000±0.00 | 0.075±0.03 | **0.250±0.00** | 0.030±0.02 |
| TPS-small   | 0.000±0.00 | 0.122±0.17 | 0.175±0.05 | 0.166±0.07 | 0.056±0.08 | 0.056±0.08 | **0.263±0.10** | 0.044±0.06 |
| TPS-medium  | 0.042±0.06 | 0.000±0.00 | 0.201±0.02 | 0.169±0.03 | 0.050±0.07 | 0.100±0.03 | **0.399±0.02** | 0.052±0.04 |

**Key findings**:
- CPA is the best method on CSVM-medium and CSVM-large, where structural
  topology changes must be detected across contexts.
- On FSVP (parametric-only changes), the Pooled baseline benefits from its
  invariance-majority class bias; all methods struggle as `p` increases.
- ICP is competitive on small problems but scales exponentially (80s at p=15
  vs CPA's 48s).

Reproduce: `PYTHONPATH=implementation python3 experiments/run_benchmarks.py`

Full results: [`experiments/results/core_benchmarks.json`](experiments/results/core_benchmarks.json)

### Visualization outputs

- **Plasticity heatmap** — Variable × component heatmap
- **Classification distribution** — Pie/bar chart of mechanism types
- **Context embedding** — Context similarity scatter plot
- **Alignment cost matrix** — Pairwise cost heatmap
- **QD archive coverage** — Archive utilization map
- **Convergence plot** — QD-score over iterations
- **Certificate dashboard** — Robustness overview
- **Tipping-point timeline** — Changepoint locations
- **Sensitivity plot** — Variable influence analysis

---

## FAQ / Troubleshooting

**Q: CPA runs but all variables are classified as INVARIANT.**
A: Your contexts may be too similar. Try increasing the parametric differences between contexts, lowering the `invariance_max_score` threshold in `DescriptorConfig`, or using the `thorough` profile with more bootstrap samples.

**Q: The pipeline is very slow.**
A: Use `PipelineConfig.fast()` for initial exploration. Skip Phase 2 with `--no-phase-2` if you don't need QD exploration. Set `--n-jobs -1` to use all CPU cores.

**Q: Tipping points are not detected.**
A: Ensure you pass `--ordered` on the CLI or set `ordered=True`. Tipping-point detection (ALG 4) requires an ordered context axis.

**Q: How do I use Parquet files?**
A: Install `pyarrow` (`pip install pyarrow`), then `cpa run --data file.parquet --context-column <col>`. The context column identifies which rows belong to which context.

**Q: Can I run CPA without causal-learn or lingam?**
A: Yes. CPA includes a zero-dependency fallback discovery method based on partial correlation and significance testing. Install the optional `[causal]` extras for full PC/GES/LiNGAM support.

**Q: How do I resume a failed run?**
A: Set `config.computation.checkpoint_dir = "checkpoints/"` and pass `--resume` on the CLI (or `orch.run(dataset, resume=True)` in Python).

---

## Testing

```bash
pytest tests/unit/ -v                        # Fast unit tests
pytest tests/integration/ -v -m integration  # Slower integration tests
pytest tests/edge_cases/ -v                  # Edge-case tests
pytest -v -m "not slow"                      # Skip slow tests
```

### Development

```bash
pip install -e ".[dev]"
ruff check cpa/
mypy cpa/
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

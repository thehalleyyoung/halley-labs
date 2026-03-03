# CPA API Reference

Developer API reference for the Causal-Plasticity Atlas engine.

---

## Table of Contents

1. [Top-Level Pipeline](#pipeline)
2. [Configuration](#configuration)
3. [Results](#results)
4. [Core Types](#core-types)
5. [Discovery](#discovery)
6. [Alignment](#alignment)
7. [Descriptors](#descriptors)
8. [Detection](#detection)
9. [Certificates](#certificates)
10. [Baselines](#baselines)
11. [Data Generators](#data-generators)
12. [IO / Serialization](#io)
13. [CLI](#cli)

---

<a id="pipeline"></a>
## 1. Pipeline — `cpa.pipeline`

### `CPAOrchestrator(config)`

Main entry point. Runs the three-phase pipeline.

```python
from cpa.pipeline import CPAOrchestrator, PipelineConfig
from cpa.pipeline.orchestrator import MultiContextDataset

orch = CPAOrchestrator(PipelineConfig.standard())
atlas = orch.run(dataset)
```

**Parameters:**
- `config` (`PipelineConfig`) — Master configuration object.

**Methods:**
- `run(dataset, resume=False)` → `AtlasResult`

### `MultiContextDataset(context_data, variable_names=None, context_ids=None)`

Input container for multi-context data.

| Parameter | Type | Description |
|-----------|------|-------------|
| `context_data` | `Dict[str, np.ndarray]` | `{context_id: (n, p) array}` |
| `variable_names` | `List[str]` or `None` | Variable names (auto-generated if omitted) |
| `context_ids` | `List[str]` or `None` | Context IDs (sorted keys if omitted) |

**Methods:**
- `get_data(context_id)` → `np.ndarray`
- `n_contexts` → `int`
- `n_variables` → `int`
- `n_samples(context_id)` → `int`

---

<a id="configuration"></a>
## 2. Configuration — `cpa.pipeline.config`

### `PipelineConfig`

Master configuration with named profiles.

```python
PipelineConfig.fast()       # Quick exploration, small bootstrap
PipelineConfig.standard()   # Balanced (default)
PipelineConfig.thorough()   # High-fidelity, extended search
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `discovery` | `DiscoveryConfig` | Structure learning settings |
| `alignment` | `AlignmentConfig` | CADA alignment settings |
| `descriptor` | `DescriptorConfig` | Plasticity descriptor settings |
| `search` | `SearchConfig` | QD-MAP-Elites settings |
| `detection` | `DetectionConfig` | Tipping-point detection settings |
| `certificate` | `CertificateConfig` | Robustness certificate settings |
| `computation` | `ComputationConfig` | Parallelism, caching, logging |
| `run_phase_1` | `bool` | Run Foundation phase (default: `True`) |
| `run_phase_2` | `bool` | Run Exploration phase (default: `True`) |
| `run_phase_3` | `bool` | Run Validation phase (default: `True`) |

**Serialization:**
```python
config.to_json("config.json")
config = PipelineConfig.from_json("config.json")
```

### `DiscoveryConfig`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | `"pc"` | Discovery method: `"pc"`, `"ges"`, `"lingam"`, `"fallback"` |
| `ci_test` | `str` | `"fisher_z"` | CI test: `"fisher_z"`, `"kernel"`, `"discrete"` |
| `alpha` | `float` | `0.05` | Significance level |
| `max_cond_set_size` | `int` | `3` | Max conditioning set size |
| `estimate_parameters` | `bool` | `True` | Estimate regression coefficients |

### `DescriptorConfig`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_bootstrap` | `int` | `50` | Bootstrap samples for CIs |
| `structural_threshold` | `float` | `0.3` | Threshold for structural plasticity |
| `parametric_threshold` | `float` | `0.3` | Threshold for parametric plasticity |
| `emergence_threshold` | `float` | `0.2` | Threshold for emergence |
| `invariance_max_score` | `float` | `0.1` | Max norm for invariant classification |

---

<a id="results"></a>
## 3. Results — `cpa.pipeline.results`

### `AtlasResult`

Top-level result container returned by `CPAOrchestrator.run()`.

```python
atlas = orch.run(dataset)

# Classification queries
atlas.classification_summary()                      # → {'invariant': 3, 'emergent': 1, ...}
atlas.get_descriptor("X0")                          # → DescriptorResult
atlas.get_classification("X0")                      # → MechanismClass.INVARIANT
atlas.variables_by_class(MechanismClass.EMERGENT)   # → ['X5', 'X7']
atlas.filter_variables(min_structural=0.5)          # → ['X2', 'X4']
atlas.most_similar_contexts(n=3)                    # → [(ci, cj, cost), ...]
atlas.alignment_cost_matrix()                       # → np.ndarray (K×K)
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `foundation` | `FoundationResult` | Phase 1 results |
| `exploration` | `ExplorationResult` or `None` | Phase 2 results |
| `validation` | `ValidationResult` or `None` | Phase 3 results |
| `variable_names` | `List[str]` | Variable names |
| `context_ids` | `List[str]` | Context IDs |

### `DescriptorResult`

4D plasticity descriptor for a single variable.

| Attribute | Type | Description |
|-----------|------|-------------|
| `variable` | `str` | Variable name |
| `structural` | `float` | Structural plasticity ∈ [0, 1] |
| `parametric` | `float` | Parametric plasticity ∈ [0, 1] |
| `emergence` | `float` | Emergence score ∈ [0, 1] |
| `sensitivity` | `float` | Context sensitivity ∈ [0, 1] |
| `classification` | `MechanismClass` | Inferred classification |
| `confidence_intervals` | `Dict` | Bootstrap CIs per component |
| `norm` | `float` | L2 norm of 4D vector |

**Property:**
- `vector` → `np.ndarray` of shape `(4,)`: `[structural, parametric, emergence, sensitivity]`

### `MechanismClass` (Enum)

```python
class MechanismClass(Enum):
    INVARIANT = "invariant"
    STRUCTURALLY_PLASTIC = "structurally_plastic"
    PARAMETRICALLY_PLASTIC = "parametrically_plastic"
    FULLY_PLASTIC = "fully_plastic"
    EMERGENT = "emergent"
    CONTEXT_SENSITIVE = "context_sensitive"
    UNCLASSIFIED = "unclassified"
```

---

<a id="core-types"></a>
## 4. Core Types — `cpa.core.types`

### `PlasticityClass` (Enum)

```python
class PlasticityClass(Enum):
    INVARIANT = "invariant"
    PARAMETRIC_PLASTIC = "parametric_plastic"
    STRUCTURAL_PLASTIC = "structural_plastic"
    MIXED = "mixed"
    EMERGENT = "emergent"
```

### Key Dataclasses

| Class | Description |
|-------|-------------|
| `SCM` | Structural Causal Model (adjacency, noise, params) |
| `Context` | Context metadata |
| `MCCM` | Multi-Context Causal Model |
| `PlasticityDescriptor` | Raw 4D descriptor |
| `TippingPoint` | Detected changepoint |
| `RobustnessCertificate` | Stability certificate |
| `QDGenome` | Quality-diversity genome |
| `QDArchiveEntry` | Archive cell entry |

---

<a id="discovery"></a>
## 5. Discovery — `cpa.discovery`

### Adapters

```python
from cpa.discovery.adapters import PCAdapter, GESAdapter, LiNGAMAdapter, FallbackAdapter

adapter = PCAdapter(alpha=0.05, ci_test="fisher_z")
result = adapter.fit(data)   # → SCMResult
```

All adapters implement:
- `fit(data: np.ndarray) → SCMResult`
- `adjacency` → `np.ndarray`
- `parameters` → `np.ndarray`

### `StructureLearner`

```python
from cpa.discovery.structure_learning import StructureLearner

learner = StructureLearner(method="pc", alpha=0.05)
scm_result = learner.learn(data, variable_names=["X0", "X1", "X2"])
```

---

<a id="alignment"></a>
## 6. Alignment — `cpa.alignment`

### `CADAAligner`

Context-Aware DAG Alignment (ALG 1). Six phases:
1. CI-fingerprint computation
2. Score matrix construction
3. Anchor detection
4. Hungarian matching
5. Quality filtering
6. Edge classification

```python
from cpa.alignment.cada import CADAAligner

aligner = CADAAligner()
result = aligner.align(scm_a, scm_b, data_a, data_b)
# result.mapping: Dict[int, int]
# result.cost: float
# result.edge_classifications: Dict
```

---

<a id="descriptors"></a>
## 7. Descriptors — `cpa.descriptors`

### `PlasticityComputer`

Computes 4D plasticity descriptors (ALG 2).

```python
from cpa.descriptors import PlasticityComputer

pc = PlasticityComputer()
desc = pc.compute(
    adjacencies=[adj_1, adj_2, adj_3],   # List of (p,p) arrays
    datasets=[data_1, data_2, data_3],   # List of (n,p) arrays
    target_idx=0,
    variable_name="X0",
)
# desc.psi_S, desc.psi_P, desc.psi_E, desc.psi_CS
# desc.classification → ClassificationResult
```

### `PlasticityClassifier`

```python
from cpa.descriptors.classification import PlasticityClassifier

classifier = PlasticityClassifier()
result = classifier.classify(psi_S=0.3, psi_P=0.7, psi_E=0.1, psi_CS=0.4)
# result.primary_category → PlasticityCategory.PARAMETRIC_PLASTIC
```

---

<a id="detection"></a>
## 8. Tipping-Point Detection — `cpa.detection`

### `PELTDetector`

Pruned Exact Linear Time changepoint detection (ALG 4).

```python
from cpa.detection.tipping_points import PELTDetector

detector = PELTDetector(min_segment_length=2, penalty="bic")
result = detector.detect(
    divergence_series=np.array([0.1, 0.1, 0.8, 0.9, 0.1]),
)
# result.locations → [2]
# result.confidences → [0.95]
```

---

<a id="certificates"></a>
## 9. Certificates — `cpa.certificates`

### `CertificateGenerator`

Issues robustness certificates for classifications (ALG 5).

```python
from cpa.certificates.robustness import CertificateGenerator

gen = CertificateGenerator(n_bootstrap=100, n_stability_rounds=50)
cert = gen.generate(descriptor_result, datasets)
# cert.certificate_type → CertificateType.GOLD / SILVER / BRONZE / NONE
# cert.stability_score → float
# cert.bootstrap_ci → (lo, hi)
```

---

<a id="baselines"></a>
## 10. Baselines — `cpa.baselines`

Seven evaluation baselines, all with a unified API:

```python
baseline.fit(datasets: Dict[str, np.ndarray])
baseline.predict_plasticity() → Dict[Tuple[int, int], PlasticityClass]
```

| Class | Method | Reference |
|-------|--------|-----------|
| `ICPBaseline` | Invariant Causal Prediction | Peters et al. (2016) |
| `CDNODBaseline` | Causal Discovery from Nonstationary Data | Zhang et al. (2017) |
| `JCIBaseline` | Joint Causal Inference | Mooij et al. (2020) |
| `GESBaseline` | Greedy Equivalence Search | Chickering (2002) |
| `IndependentPHC` | Independent per-context + post-hoc | — |
| `PooledBaseline` | Pooled data baseline | — |
| `LSEMPooled` | Linear SEM pooled estimation | — |

---

<a id="data-generators"></a>
## 11. Data Generators — `benchmarks.generators`

### `FSVPGenerator(p, K, n, seed)`

Fixed Structure, Varying Parameters.

```python
from benchmarks.generators import FSVPGenerator
gen = FSVPGenerator(p=10, K=5, n=500, seed=42)
bench = gen.generate()
# bench.context_data: Dict[str, np.ndarray]
# bench.variable_names: List[str]
# bench.ground_truth.classifications: Dict[str, str]
```

### `CSVMGenerator(p, K, n, seed)`

Changing Structure with Variable Mismatch.

### `TPSGenerator(p, K, n, seed)`

Tipping-Point Scenario with known changepoint locations.

### `GroundTruth`

| Attribute | Type | Description |
|-----------|------|-------------|
| `adjacencies` | `Dict[str, np.ndarray]` | True adjacency per context |
| `parameters` | `Dict[str, np.ndarray]` | True weights per context |
| `classifications` | `Dict[str, str]` | True mechanism class per variable |
| `tipping_points` | `List[int]` | True changepoint locations |
| `invariant_variables` | `List[str]` | Variables with invariant mechanisms |
| `plastic_variables` | `List[str]` | Variables with plastic mechanisms |
| `emergent_variables` | `List[str]` | Variables that appear/disappear |

---

<a id="io"></a>
## 12. IO / Serialization — `cpa.io`

```python
from cpa.io.serialization import save_atlas, load_atlas

save_atlas(atlas, "atlas.json")
loaded = load_atlas("atlas.json")   # → Dict[str, Any]
```

### Readers

```python
from cpa.io.readers import CSVReader, ParquetReader

dataset = CSVReader("data/", context_column="site").read()
dataset = ParquetReader("data.parquet", context_column="env").read()
```

### Writers

```python
from cpa.io.writers import JSONWriter, CSVWriter

JSONWriter("output/").write(atlas)
CSVWriter("output/").write(atlas)
```

---

<a id="cli"></a>
## 13. CLI — `cpa.cli`

```bash
# Full pipeline
cpa run data/ --config standard --output atlas.json

# Discovery only
cpa discover data/ --method pc --alpha 0.05

# Alignment
cpa align data/ --strategy exact

# Info
cpa info

# Benchmark
cpa benchmark --scenario FSVP --p 10 --K 5 --n 500
```

**Entry point:** `python -m cpa.cli` or `cpa` (if installed).

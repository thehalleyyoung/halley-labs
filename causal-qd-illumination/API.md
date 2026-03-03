# CausalQD — API Reference

> Complete API documentation for all 24 subpackages.

---

## Table of Contents

1.  [Core Types](#core-types)
2.  [Data Generation](#data-generation)
3.  [Scoring Functions](#scoring-functions)
4.  [Engine](#engine)
5.  [Archive](#archive)
6.  [Descriptors](#descriptors)
7.  [Operators](#operators)
8.  [Metrics](#metrics)
9.  [Certificates](#certificates)
10. [MEC Operations](#mec-operations)
11. [CI Tests](#ci-tests)
12. [Baselines](#baselines)
13. [Sampling](#sampling)
14. [Equivalence](#equivalence)
15. [Streaming](#streaming)
16. [Scalability](#scalability)
17. [Analysis](#analysis)
18. [Visualization](#visualization)
19. [Configuration](#configuration)
20. [Benchmarks](#benchmarks)
21. [Curiosity](#curiosity)
22. [IO](#io)
23. [Pipeline](#pipeline)
24. [Parallel](#parallel)

---

## Core Types

**Module**: `causal_qd.core`

### `DAG`

Directed acyclic graph backed by a NumPy adjacency matrix.

```python
class DAG:
    def __init__(self, adjacency: np.ndarray) -> None: ...

    @property
    def num_nodes(self) -> int: ...
    @property
    def num_edges(self) -> int: ...
    @property
    def adjacency(self) -> np.ndarray: ...
    @property
    def topological_order(self) -> list[int]: ...

    def parents(self, node: int) -> list[int]: ...
    def children(self, node: int) -> list[int]: ...
    def descendants(self, node: int) -> set[int]: ...
    def ancestors(self, node: int) -> set[int]: ...
    def is_ancestor(self, source: int, target: int) -> bool: ...
    def markov_blanket(self, node: int) -> set[int]: ...
```

**Raises**: `ValueError` if the adjacency matrix contains a cycle.

### Type Aliases

```python
AdjacencyMatrix = np.ndarray        # (n, n), int8, binary
DataMatrix = np.ndarray             # (N, p), float64
BehavioralDescriptor = np.ndarray   # (d,), float64
QualityScore = float
WeightedAdjacencyMatrix = np.ndarray  # (n, n), float64
BootstrapSample = np.ndarray        # (N,), int64

# Callable protocols
ScoreFn = Callable[[AdjacencyMatrix, DataMatrix], float]
DescriptorFn = Callable[[AdjacencyMatrix, DataMatrix], np.ndarray]
MutationOp = Callable[[AdjacencyMatrix, np.random.Generator], AdjacencyMatrix]
CrossoverOp = Callable[[AdjacencyMatrix, AdjacencyMatrix, np.random.Generator], AdjacencyMatrix]
CallbackFn = Callable[['IterationStats'], None]
```

---

## Data Generation

**Module**: `causal_qd.data`

### `LinearGaussianSCM`

Linear structural causal model with Gaussian noise.

```python
class LinearGaussianSCM:
    def __init__(
        self,
        dag: DAG,
        weights: WeightedAdjacencyMatrix,
        noise_std: np.ndarray,  # (n,)
    ) -> None: ...

    def sample(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> DataMatrix: ...
```

**Parameters**:
- `dag`: Ground-truth DAG structure
- `weights`: Edge weight matrix (same sparsity as `dag.adjacency`)
- `noise_std`: Per-node noise standard deviation

### `NonlinearSCM`

Nonlinear SCM with configurable mechanism type (GP, MLP, polynomial).

```python
class NonlinearSCM:
    def __init__(
        self,
        dag: DAG,
        mechanisms: str,         # 'gp', 'mlp', 'polynomial'
        noise_std: np.ndarray,
    ) -> None: ...

    def sample(self, n: int, rng: ...) -> DataMatrix: ...
```

---

## Scoring Functions

**Module**: `causal_qd.scores`

All scores implement the `ScoreFunction` abstract base class:

```python
class ScoreFunction(ABC):
    @abstractmethod
    def score(self, adj: AdjacencyMatrix, data: DataMatrix) -> float: ...

    @abstractmethod
    def local_score(
        self,
        node: int,
        parents: list[int],
        data: DataMatrix,
    ) -> float: ...
```

### `BICScore`

```python
class BICScore(ScoreFunction):
    def __init__(
        self,
        penalty_multiplier: float = 1.0,
        regularization: str = 'none',  # 'none', 'l1', 'l2'
        reg_lambda: float = 0.01,
    ) -> None: ...
```

### `BDeuScore`

```python
class BDeuScore(ScoreFunction):
    def __init__(
        self,
        equivalent_sample_size: float = 10.0,
    ) -> None: ...
```

### `BGeScore`

```python
class BGeScore(ScoreFunction):
    def __init__(
        self,
        alpha_mu: float = 1.0,
        alpha_w: float | None = None,
    ) -> None: ...
```

### `HybridScore`

Weighted combination of multiple score functions.

```python
class HybridScore(ScoreFunction):
    def __init__(
        self,
        scores: list[ScoreFunction],
        weights: list[float],
    ) -> None: ...
```

### `InterventionalScore`

Score incorporating interventional data.

```python
class InterventionalScore(ScoreFunction):
    def __init__(
        self,
        observational_score: ScoreFunction,
        interventions: list[dict],
    ) -> None: ...
```

---

## Engine

**Module**: `causal_qd.engine`

### `MAPElitesConfig`

```python
@dataclass
class MAPElitesConfig:
    mutation_prob: float = 0.7
    crossover_rate: float = 0.3
    archive_dims: tuple[int, ...] = (20, 20)
    archive_ranges: tuple[tuple[float, float], ...] = ((0.0, 1.0), (0.0, 1.0))
    seed: int = 42
    selection_strategy: str = 'uniform'
    adaptive_operators: bool = False
    early_stopping_window: int = 0
    early_stopping_threshold: float = 0.0001
    checkpoint_interval: int = 0
    checkpoint_dir: str = 'checkpoints'
    log_interval: int = 10
```

### `CausalMAPElites`

```python
class CausalMAPElites:
    def __init__(
        self,
        mutations: list[MutationOp],
        crossovers: list[CrossoverOp],
        descriptor_fn: DescriptorFn,
        score_fn: ScoreFn,
        config: MAPElitesConfig | None = None,
        callbacks: list[CallbackFn] | None = None,
        evaluator: BatchEvaluator | None = None,
    ) -> None: ...

    def run(
        self,
        data: DataMatrix,
        n_iterations: int,
        batch_size: int = 16,
        initial_dags: list[AdjacencyMatrix] | None = None,
    ) -> _GridArchive: ...

    def step(
        self,
        data: DataMatrix,
        batch_size: int,
    ) -> IterationStats: ...

    @property
    def iteration(self) -> int: ...
    @property
    def history(self) -> list[IterationStats]: ...
    @property
    def archive(self) -> _GridArchive: ...
    @property
    def stopped_early(self) -> bool: ...
```

### `IterationStats`

```python
@dataclass
class IterationStats:
    iteration: int
    archive_size: int
    best_quality: float
    mean_quality: float
    qd_score: float
    coverage: float
    elapsed_time: float
    n_evaluations: int
    n_improvements: int
```

---

## Archive

**Module**: `causal_qd.archive`

### `ArchiveEntry`

```python
@dataclass
class ArchiveEntry:
    solution: AdjacencyMatrix
    descriptor: BehavioralDescriptor
    quality: QualityScore
    metadata: dict
    timestamp: int
```

### `GridArchive`

```python
class GridArchive:
    def __init__(
        self,
        dims: tuple[int, ...],
        ranges: tuple[tuple[float, float], ...],
    ) -> None: ...

    def add(
        self,
        solution: AdjacencyMatrix,
        descriptor: BehavioralDescriptor,
        quality: QualityScore,
        metadata: dict | None = None,
    ) -> bool: ...

    def best(self) -> ArchiveEntry | None: ...

    @property
    def entries(self) -> Iterator[ArchiveEntry]: ...
    @property
    def size(self) -> int: ...

    def qd_score(self) -> float: ...
    def coverage(self) -> float: ...
    def fill_count(self) -> int: ...
    def replace_count(self) -> int: ...

    def sample(self, rng: np.random.Generator) -> ArchiveEntry: ...
    def sample_curiosity(self, rng: np.random.Generator) -> ArchiveEntry: ...
    def sample_quality_proportional(self, rng: np.random.Generator) -> ArchiveEntry: ...

    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
    def clear(self) -> None: ...
```

### `CVTArchive`

Centroidal Voronoi Tessellation archive for continuous descriptor spaces.

```python
class CVTArchive:
    def __init__(
        self,
        n_cells: int,
        descriptor_dim: int,
        descriptor_ranges: tuple[tuple[float, float], ...],
    ) -> None: ...
    # Same interface as GridArchive
```

---

## Descriptors

**Module**: `causal_qd.descriptors`

### `StructuralDescriptor`

```python
class StructuralDescriptor:
    VALID_FEATURES = [
        'edge_density', 'max_in_degree', 'v_structure_count',
        'longest_path', 'avg_path_length', 'clustering_coefficient',
        'betweenness_centrality', 'connected_components',
        'dag_depth', 'parent_set_entropy',
    ]

    def __init__(self, features: list[str] | None = None) -> None: ...
    def compute(self, adj: AdjacencyMatrix, data: DataMatrix) -> np.ndarray: ...
```

### `InfoTheoreticDescriptor`

```python
class InfoTheoreticDescriptor:
    def __init__(self, features: list[str] | None = None) -> None: ...
    def compute(self, adj: AdjacencyMatrix, data: DataMatrix) -> np.ndarray: ...
```

### `EquivalenceDescriptor`

```python
class EquivalenceDescriptor:
    def __init__(self, features: list[str] | None = None) -> None: ...
    def compute(self, adj: AdjacencyMatrix, data: DataMatrix) -> np.ndarray: ...
```

### `SpectralDescriptor`

```python
class SpectralDescriptor:
    def __init__(self, features: list[str] | None = None) -> None: ...
    def compute(self, adj: AdjacencyMatrix, data: DataMatrix) -> np.ndarray: ...
```

### `CompositeDescriptor`

```python
class CompositeDescriptor:
    def __init__(self, descriptors: list) -> None: ...
    def compute(self, adj: AdjacencyMatrix, data: DataMatrix) -> np.ndarray: ...
```

---

## Operators

**Module**: `causal_qd.operators`

### Mutation Operators

All extend `MutationOperator`:

```python
class MutationOperator(ABC):
    @abstractmethod
    def mutate(
        self,
        dag: AdjacencyMatrix,
        rng: np.random.Generator,
    ) -> AdjacencyMatrix: ...
```

| Class | Description |
|-------|-------------|
| `EdgeFlipMutation` | Add/remove random edge |
| `EdgeAddMutation` | Add random edge (acyclic) |
| `EdgeRemoveMutation` | Remove random edge |
| `EdgeReverseMutation` | Reverse random edge |
| `EdgeReversalMutation` | Alias for EdgeReverseMutation |
| `TopologicalMutation` | Topological-order-aware mutation |
| `VStructureMutation` | Create/destroy v-structures |
| `SkeletonMutation` | Modify skeleton preserving v-structures |
| `PathMutation` | Modify edges along paths |
| `BlockMutation` | Block-wise mutation |
| `CompositeMutation` | Sequential composition of mutations |
| `ConstrainedMutation` | Mutation respecting edge constraints |
| `AdaptiveMutation` | Adaptive mutation rate |
| `NeighborhoodMutation` | Neighborhood-based mutation |
| `MixingMutation` | Mixing multiple mutation types |
| `AcyclicEdgeAddition` | Edge addition with acyclicity check |

### Crossover Operators

All extend `CrossoverOperator`:

```python
class CrossoverOperator(ABC):
    @abstractmethod
    def crossover(
        self,
        parent1: AdjacencyMatrix,
        parent2: AdjacencyMatrix,
        rng: np.random.Generator,
    ) -> tuple[AdjacencyMatrix, AdjacencyMatrix]: ...
```

| Class | Description |
|-------|-------------|
| `UniformCrossover` | Element-wise uniform |
| `SkeletonCrossover` | Skeleton recombination |
| `MarkovBlanketCrossover` | MB exchange |
| `OrderBasedCrossover` | Order-based |
| `OrderCrossover` | Order crossover |
| `SubgraphCrossover` | Subgraph exchange |
| `ConstrainedCrossover` | Constrained recombination |

### Repair Operators

```python
class RepairOperator(ABC):
    @abstractmethod
    def repair(self, dag: AdjacencyMatrix) -> AdjacencyMatrix: ...
```

| Class | Description |
|-------|-------------|
| `AcyclicityRepair` | Remove cycle-creating edges |
| `ConnectivityRepair` | Ensure weak connectivity |
| `TopologicalRepair` | Fix topological violations |
| `MinimalRepair` | Minimum-cost repair |
| `OrderRepair` | Order-based repair |

### Local Search Operators

| Class | Description |
|-------|-------------|
| `GreedyLocalSearch` | Greedy improvement |
| `HillClimbingRefiner` | Hill climbing |
| `TabuSearch` | Tabu search |
| `SimulatedAnnealing` | Simulated annealing |
| `StochasticLocalSearch` | Randomised LS |

### Edge Constraints

```python
class EdgeConstraints:
    def __init__(
        self,
        required: list[tuple[int, int]] = ...,
        forbidden: list[tuple[int, int]] = ...,
    ) -> None: ...

class TierConstraints:
    def __init__(self, tiers: list[list[int]]) -> None: ...
```

---

## Metrics

**Module**: `causal_qd.metrics`

### `SHD`

```python
class SHD:
    @staticmethod
    def compute(predicted: AdjacencyMatrix, true: AdjacencyMatrix) -> int: ...
    @staticmethod
    def compute_simple(predicted: AdjacencyMatrix, true: AdjacencyMatrix) -> int: ...
```

### `F1`

```python
class F1:
    def compute(self, predicted: AdjacencyMatrix, true: AdjacencyMatrix) -> float: ...
```

### `QDScore`

```python
class QDScore:
    def compute(self, archive) -> float: ...
```

### `Coverage`

```python
class Coverage:
    def compute(self, archive) -> float: ...
```

### `Diversity`

```python
class Diversity:
    def compute(self, archive) -> float: ...
```

### `MECRecall`

```python
class MECRecall:
    def compute(self, archive, true_adj: AdjacencyMatrix) -> float: ...
```

---

## Certificates

**Module**: `causal_qd.certificates`

### `BootstrapCertificateComputer`

```python
class BootstrapCertificateComputer:
    def __init__(
        self,
        n_bootstrap: int,
        score_fn: ScoreFn,
        confidence_level: float = 0.95,
        rng: np.random.Generator | None = None,
        compute_lipschitz: bool = False,
        lipschitz_perturbation_scale: float = 0.01,
    ) -> None: ...

    def compute_edge_certificates(
        self,
        dag: AdjacencyMatrix,
        data: DataMatrix,
    ) -> dict[tuple[int, int], EdgeCertificate]: ...

    def compute_nonedge_certificates(
        self,
        dag: AdjacencyMatrix,
        data: DataMatrix,
    ) -> dict[tuple[int, int], EdgeCertificate]: ...

    def compute_all_certificates(
        self,
        dag: AdjacencyMatrix,
        data: DataMatrix,
    ) -> dict[tuple[int, int], EdgeCertificate]: ...
```

### `EdgeCertificate`

```python
class EdgeCertificate(Certificate):
    source: int
    target: int
    bootstrap_frequency: float
    score_delta: float
    confidence: float
    lipschitz_bound: float | None
    bootstrap_deltas: list[float]
    n_bootstrap: int
```

### `PathCertificate`

```python
class PathCertificate(Certificate):
    path: list[int]
    frequency: float
    strength: float
```

### `LipschitzBound`

```python
class LipschitzBound:
    bound: float
    perturbation_scale: float
```

---

## MEC Operations

**Module**: `causal_qd.mec`

### `CPDAGConverter`

```python
class CPDAGConverter:
    def dag_to_cpdag(self, dag: DAG) -> AdjacencyMatrix: ...
    def cpdag_to_dags(self, cpdag: AdjacencyMatrix) -> list[AdjacencyMatrix]: ...
    def find_v_structures(self, adj: AdjacencyMatrix) -> list[tuple[int, int, int]]: ...
    def is_valid_cpdag(self, cpdag: AdjacencyMatrix) -> bool: ...
    def is_compelled(self, cpdag: AdjacencyMatrix, i: int, j: int) -> bool: ...
    def compelled_edge_analysis(self, cpdag: AdjacencyMatrix) -> dict: ...
```

### `MECEnumerator`

```python
class MECEnumerator:
    def enumerate(self, cpdag: AdjacencyMatrix) -> Generator[AdjacencyMatrix, None, None]: ...
    def count(self, cpdag: AdjacencyMatrix) -> int: ...
    def count_mecs(self, adj: AdjacencyMatrix) -> int: ...
    def enumerate_all_mecs(self, adj: AdjacencyMatrix) -> list: ...
    def sample(self, cpdag: AdjacencyMatrix, n: int = 1) -> Generator: ...
```

### `CanonicalHasher`

```python
class CanonicalHasher:
    def hash(self, adj: AdjacencyMatrix) -> int: ...
```

---

## CI Tests

**Module**: `causal_qd.ci_tests`

All CI tests implement:

```python
class CITest(ABC):
    @abstractmethod
    def test(
        self,
        data: DataMatrix,
        x: int,
        y: int,
        conditioning_set: list[int],
    ) -> tuple[bool, float]: ...
    # Returns (is_independent, p_value)
```

| Class | Method |
|-------|--------|
| `FisherZTest` | Fisher Z-transform of partial correlation |
| `KernelCITest` | Kernel-based CI test |
| `PartialCorrelationTest` | Partial correlation with threshold |
| `ConditionalMutualInfoTest` | CMI-based test |

### `FisherZTest`

```python
class FisherZTest(CITest):
    def __init__(self, alpha: float = 0.05) -> None: ...
```

### `KernelCITest`

```python
class KernelCITest(CITest):
    def __init__(
        self,
        alpha: float = 0.05,
        kernel: str = 'rbf',
        n_permutations: int = 200,
    ) -> None: ...
```

---

## Baselines

**Module**: `causal_qd.baselines`

All baselines implement:

```python
class BaselineAlgorithm(ABC):
    @abstractmethod
    def run(self, data: DataMatrix) -> AdjacencyMatrix: ...
```

### `PCAlgorithm`

```python
class PCAlgorithm(BaselineAlgorithm):
    def __init__(self, alpha: float = 0.05, ci_test: str = 'fisher_z') -> None: ...
```

### `GESAlgorithm`

```python
class GESAlgorithm(BaselineAlgorithm):
    def __init__(self, score: str = 'bic') -> None: ...
```

### `MMHCAlgorithm`

```python
class MMHCAlgorithm(BaselineAlgorithm):
    def __init__(self, alpha: float = 0.05, max_k: int = 3) -> None: ...
```

### `OrderMCMCBaseline`

```python
class OrderMCMCBaseline(BaselineAlgorithm):
    def __init__(self, n_samples: int = 1000, burnin: int = 200) -> None: ...
```

---

## Sampling

**Module**: `causal_qd.sampling`

### `OrderMCMC`

```python
class OrderMCMC:
    def __init__(
        self,
        score_fn: ScoreFn,
        n_samples: int = 1000,
        burnin: int = 200,
    ) -> None: ...

    def sample(self, data: DataMatrix) -> list[AdjacencyMatrix]: ...
```

### `ParallelTempering`

```python
class ParallelTempering:
    def __init__(
        self,
        score_fn: ScoreFn,
        n_chains: int = 4,
        temperatures: list[float] = [1, 2, 4, 8],
    ) -> None: ...

    def sample(self, data: DataMatrix) -> list[AdjacencyMatrix]: ...
```

### `UniformDAGSampler`

```python
class UniformDAGSampler:
    def __init__(self, n_nodes: int) -> None: ...
    def sample(self, rng: np.random.Generator) -> AdjacencyMatrix: ...
```

---

## Equivalence

**Module**: `causal_qd.equivalence`

### `EquivalenceClassDecomposer`

```python
class EquivalenceClassDecomposer:
    def decompose(self, cpdag: AdjacencyMatrix) -> list: ...
```

### `AdvancedEquivalenceDecomposer`

```python
class AdvancedEquivalenceDecomposer:
    def decompose(self, cpdag: AdjacencyMatrix) -> list: ...
```

### `ChainComponentDecomposition`

```python
class ChainComponentDecomposition:
    def decompose(self, cpdag: AdjacencyMatrix) -> list[set[int]]: ...
```

### `InterventionDesign`

```python
class InterventionDesign:
    def optimal_targets(
        self,
        cpdag: AdjacencyMatrix,
        budget: int,
    ) -> list[int]: ...
```

### `DAGtoMEC` / `MECtoDAGs`

```python
class DAGtoMEC:
    def convert(self, dag: DAG) -> AdjacencyMatrix: ...

class MECtoDAGs:
    def enumerate(self, cpdag: AdjacencyMatrix) -> list[AdjacencyMatrix]: ...
```

### `NautyInterface` / `ExtendedNautyInterface`

```python
class NautyInterface:
    def canonical_form(self, adj: AdjacencyMatrix) -> AdjacencyMatrix: ...
    def automorphism_group_size(self, adj: AdjacencyMatrix) -> int: ...
```

---

## Streaming

**Module**: `causal_qd.streaming`

### `OnlineArchive`

```python
class OnlineArchive:
    def __init__(
        self,
        dims: tuple[int, ...],
        ranges: tuple[tuple[float, float], ...],
    ) -> None: ...

    def update(
        self,
        solution: AdjacencyMatrix,
        descriptor: np.ndarray,
        quality: float,
    ) -> bool: ...

    @property
    def entries(self) -> Iterator[ArchiveEntry]: ...
    @property
    def size(self) -> int: ...
```

### `StreamingStats`

```python
class StreamingStats:
    def update(self, value: float) -> None: ...

    @property
    def mean(self) -> float: ...
    @property
    def variance(self) -> float: ...
    @property
    def count(self) -> int: ...
```

### `IncrementalDescriptor`

```python
class IncrementalDescriptor:
    def update(self, adj: AdjacencyMatrix, data: DataMatrix) -> np.ndarray: ...
```

---

## Scalability

**Module**: `causal_qd.scalability`

### `ApproximateDescriptor`

```python
class ApproximateDescriptor:
    def __init__(self, n_samples: int = 100) -> None: ...
    def compute(self, adj: AdjacencyMatrix, data: DataMatrix) -> np.ndarray: ...
```

### `PCACompressor`

```python
class PCACompressor:
    def __init__(self, n_components: int = 2) -> None: ...
    def fit_transform(self, descriptors: np.ndarray) -> np.ndarray: ...
    def transform(self, descriptors: np.ndarray) -> np.ndarray: ...
```

### `SamplingCI`

```python
class SamplingCI:
    def __init__(self, subsample_size: int = 100) -> None: ...
    def test(self, data: DataMatrix, x: int, y: int, cond: list[int]) -> tuple[bool, float]: ...
```

### `SkeletonRestrictor`

```python
class SkeletonRestrictor:
    def __init__(self, max_parents: int = 3) -> None: ...
    def restrict(self, adj: AdjacencyMatrix) -> AdjacencyMatrix: ...
```

---

## Analysis

**Module**: `causal_qd.analysis`

### `ConvergenceAnalyzer`

```python
class ConvergenceAnalyzer:
    def check(self, history: list[IterationStats]) -> bool: ...
```

### `ArchiveDiagnostics`

```python
class ArchiveDiagnostics:
    def analyse(self, archive) -> dict: ...
```

### `OperatorDiagnostics`

```python
class OperatorDiagnostics:
    def analyse(self, engine: CausalMAPElites) -> dict: ...
```

### `ScoreDiagnostics`

```python
class ScoreDiagnostics:
    def analyse(self, scores: list[float]) -> dict: ...
```

### `AlgorithmComparator`

```python
class AlgorithmComparator:
    def compare(
        self,
        results: dict[str, dict],
        metrics: list[str] = ['shd', 'f1'],
    ) -> dict: ...
```

### `BenchmarkSuite`

```python
class BenchmarkSuite:
    def run(self, algorithms: list, datasets: list) -> dict: ...
```

### `CausalQueryEngine`

```python
class CausalQueryEngine:
    def query(
        self,
        archive,
        query: str,  # 'is_ancestor(0, 3)', 'effect(X, Y)', etc.
    ) -> Any: ...
```

### `ParameterSensitivityAnalyzer`

```python
class ParameterSensitivityAnalyzer:
    def analyze(
        self,
        param_name: str,
        param_values: list,
        data: DataMatrix,
    ) -> dict: ...
```

### `DataSensitivityAnalyzer`

```python
class DataSensitivityAnalyzer:
    def analyze(
        self,
        data: DataMatrix,
        sample_sizes: list[int],
    ) -> dict: ...
```

### `ErgodicityChecker`

```python
class ErgodicityChecker:
    def check(self, history: list) -> bool: ...
```

### `SupermartingaleTracker`

```python
class SupermartingaleTracker:
    def update(self, value: float) -> None: ...
    def is_supermartingale(self) -> bool: ...
```

### `ConvergenceSnapshot`

```python
class ConvergenceSnapshot:
    iteration: int
    qd_score: float
    coverage: float
    best_quality: float
```

### `InterventionEstimator`

```python
class InterventionEstimator:
    def estimate(
        self,
        archive,
        target: int,
        intervention_value: float,
    ) -> dict: ...
```

### `ScoreSensitivityAnalyzer`

```python
class ScoreSensitivityAnalyzer:
    def analyze(
        self,
        adj: AdjacencyMatrix,
        data: DataMatrix,
        perturbation_scale: float = 0.01,
    ) -> dict: ...
```

---

## Visualization

**Module**: `causal_qd.visualization`

### `ArchivePlotter`

```python
class ArchivePlotter:
    def heatmap(
        self,
        archive,
        filename: str | None = None,
        title: str = 'Archive Heatmap',
    ) -> None: ...

    def scatter(
        self,
        archive,
        filename: str | None = None,
    ) -> None: ...
```

### `ConvergencePlotter`

```python
class ConvergencePlotter:
    def plot(
        self,
        history: list[IterationStats],
        filename: str | None = None,
        metrics: list[str] = ['qd_score', 'coverage', 'best_quality'],
    ) -> None: ...
```

### `DAGRenderer`

```python
class DAGRenderer:
    def render(
        self,
        dag: DAG | AdjacencyMatrix,
        filename: str | None = None,
        format: str = 'png',
    ) -> None: ...
```

### `CertificateDisplay`

```python
class CertificateDisplay:
    def display(
        self,
        certificates: dict[tuple[int, int], EdgeCertificate],
        filename: str | None = None,
    ) -> None: ...
```

---

## Configuration

**Module**: `causal_qd.config`

### `CausalQDConfig`

```python
@dataclass
class CausalQDConfig:
    archive: ArchiveConfig = ...
    operators: OperatorConfig = ...
    score: ScoreConfig = ...
    descriptor: DescriptorConfig = ...
    certificate: CertificateConfig = ...
```

### Sub-configs

```python
@dataclass
class ArchiveConfig:
    dims: tuple[int, ...] = (20, 20)
    ranges: tuple[tuple[float, float], ...] = ((0, 1), (0, 1))
    type: str = 'grid'  # or 'cvt'

@dataclass
class OperatorConfig:
    mutation_prob: float = 0.7
    adaptive: bool = False
    mutations: list[str] = ...
    crossovers: list[str] = ...

@dataclass
class ScoreConfig:
    type: str = 'bic'
    penalty: float = 1.0

@dataclass
class DescriptorConfig:
    features: list[str] = ...
    type: str = 'structural'

@dataclass
class CertificateConfig:
    n_bootstrap: int = 200
    confidence_level: float = 0.95
    compute_lipschitz: bool = False

@dataclass
class ExperimentConfig:
    n_iterations: int = 1000
    batch_size: int = 32
    seed: int = 42
    n_repeats: int = 5
```

---

## Benchmarks

**Module**: `causal_qd.benchmarks`

### Standard Benchmark Graphs

| Class | Description | Nodes | Edges |
|-------|-------------|-------|-------|
| `AsiaBenchmark` | Asia (chest clinic) | 8 | 8 |
| `SachsBenchmark` | Sachs (protein signaling) | 11 | 17 |
| `AlarmBenchmark` | ALARM (monitoring) | 37 | 46 |
| `ChildBenchmark` | Child (medical) | 20 | 25 |
| `InsuranceBenchmark` | Insurance | 27 | 52 |

### `BenchmarkRunner`

```python
class BenchmarkRunner:
    def run(
        self,
        benchmark: str,    # 'asia', 'sachs', 'alarm', etc.
        algorithms: list[str],
        n_samples: int = 1000,
        n_repeats: int = 5,
    ) -> dict: ...
```

### `ComparisonRunner`

```python
class ComparisonRunner:
    def compare(
        self,
        algorithms: dict[str, BaselineAlgorithm],
        datasets: list[DataMatrix],
        true_dags: list[AdjacencyMatrix],
    ) -> dict: ...
```

### Additional Benchmarks

| Class | Description |
|-------|-------------|
| `RandomDAGBenchmark` | Random Erdős–Rényi DAGs |
| `ScalabilityBenchmark` | Scale from 10 to 100 nodes |
| `SparsityBenchmark` | Varying edge densities |
| `FaithfulnessViolationBenchmark` | Faithfulness violations |

---

## Curiosity

**Module**: `causal_qd.curiosity`

Curiosity-driven exploration for MAP-Elites, prioritising under-explored
archive cells.

---

## IO

**Module**: `causal_qd.io`

File I/O utilities for reading/writing DAGs, data, and archives.

---

## Pipeline

**Module**: `causal_qd.pipeline`

End-to-end pipeline orchestrating data loading, MAP-Elites execution,
certification, and result export.

---

## Parallel

**Module**: `causal_qd.parallel`

Parallel evaluation support for batch fitness evaluation across
multiple CPU cores.

---

## Detailed Usage Examples

### Example 1: Complete Workflow

```python
import numpy as np
from causal_qd.core import DAG
from causal_qd.data import LinearGaussianSCM
from causal_qd.scores import BICScore
from causal_qd.engine import CausalMAPElites, MAPElitesConfig
from causal_qd.descriptors import StructuralDescriptor
from causal_qd.operators import (
    EdgeFlipMutation, EdgeAddMutation, EdgeRemoveMutation,
    TopologicalMutation, EdgeReverseMutation, VStructureMutation,
    SkeletonMutation, PathMutation,
    UniformCrossover, SkeletonCrossover, MarkovBlanketCrossover,
)
from causal_qd.metrics import SHD, F1
from causal_qd.mec import CPDAGConverter, MECEnumerator
from causal_qd.certificates import BootstrapCertificateComputer

# Step 1: Generate synthetic data
np.random.seed(42)
true_adj = np.array([
    [0,1,1,0,0,0,0,0],
    [0,0,0,1,0,0,0,0],
    [0,0,0,1,1,0,0,0],
    [0,0,0,0,0,1,0,0],
    [0,0,0,0,0,1,1,0],
    [0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0],
], dtype=float)
dag = DAG(true_adj)
weights = true_adj * np.random.uniform(0.5, 2.0, size=true_adj.shape)
scm = LinearGaussianSCM(dag=dag, weights=weights, noise_std=np.ones(8))
data = scm.sample(n=800)

# Step 2: Set up operators
def make_mut(cls):
    op = cls()
    return lambda adj, rng: op.mutate(adj, rng)

def make_cross(cls):
    op = cls()
    def fn(a1, a2, rng):
        c1, c2 = op.crossover(a1, a2, rng)
        return c1
    return fn

mutations = [make_mut(c) for c in [
    EdgeFlipMutation, EdgeAddMutation, EdgeRemoveMutation,
    TopologicalMutation, EdgeReverseMutation, VStructureMutation,
    SkeletonMutation, PathMutation,
]]
crossovers = [make_cross(c) for c in [
    UniformCrossover, SkeletonCrossover, MarkovBlanketCrossover,
]]

# Step 3: Set up descriptor and score
desc = StructuralDescriptor(features=['edge_density', 'max_in_degree'])
scorer = BICScore()

config = MAPElitesConfig(
    archive_dims=(8, 8),
    seed=42,
    adaptive_operators=True,
    mutation_prob=0.8,
    log_interval=100,
)

engine = CausalMAPElites(
    mutations=mutations,
    crossovers=crossovers,
    descriptor_fn=lambda adj, data: desc.compute(adj, data),
    score_fn=lambda adj, data: scorer.score(adj, data),
    config=config,
)

# Step 4: Run MAP-Elites
archive = engine.run(data=data, n_iterations=1000, batch_size=32)

# Step 5: Analyse results
best = archive.best()
print(f"Archive: {archive.size} elites, coverage={archive.coverage():.3f}")
print(f"Best quality: {best.quality:.2f}")
print(f"SHD: {SHD.compute(best.solution, true_adj)}")
print(f"F1:  {F1().compute(best.solution, true_adj):.3f}")

# Step 6: MEC analysis
converter = CPDAGConverter()
seen = set()
for entry in archive.entries:
    cpdag = converter.dag_to_cpdag(DAG(entry.solution))
    seen.add(cpdag.tobytes())
print(f"Unique MECs in archive: {len(seen)}")

# Step 7: Certificates
cert = BootstrapCertificateComputer(
    n_bootstrap=100,
    score_fn=lambda adj, data: scorer.score(adj, data),
)
certs = cert.compute_edge_certificates(best.solution, data)
for (s, t), ec in sorted(certs.items()):
    print(f"  {s}→{t}: freq={ec.bootstrap_frequency:.3f}, "
          f"delta={ec.score_delta:.1f}")
```

### Example 2: Warm-Starting with Baselines

```python
from causal_qd.baselines import PCAlgorithm, GESAlgorithm, MMHCAlgorithm

# Get initial solutions from multiple baselines
pc_dag = PCAlgorithm().run(data)
ges_dag = GESAlgorithm().run(data)
mmhc_dag = MMHCAlgorithm().run(data)

# Warm-start MAP-Elites with baseline solutions
archive = engine.run(
    data=data,
    n_iterations=500,
    batch_size=32,
    initial_dags=[pc_dag, ges_dag, mmhc_dag],
)

# The archive now starts from a better initial state
print(f"Archive size: {archive.size}")
```

### Example 3: Step-by-Step Execution with Monitoring

```python
from causal_qd.engine import IterationStats

# Step-by-step with custom monitoring
convergence_history = []

for i in range(500):
    stats: IterationStats = engine.step(data, batch_size=32)
    convergence_history.append({
        'iteration': stats.iteration,
        'archive_size': stats.archive_size,
        'best_quality': stats.best_quality,
        'mean_quality': stats.mean_quality,
        'qd_score': stats.qd_score,
        'coverage': stats.coverage,
        'elapsed': stats.elapsed_time,
    })
    
    if i % 50 == 0:
        print(f"[{i:4d}] archive={stats.archive_size:3d} "
              f"best={stats.best_quality:.1f} "
              f"QD={stats.qd_score:.1f} "
              f"cov={stats.coverage:.3f}")
    
    # Custom early stopping
    if stats.coverage > 0.9:
        print(f"High coverage achieved at iteration {i}")
        break
```

### Example 4: Custom Scoring and Descriptors

```python
# Custom score function
def my_score(adj, data):
    """Custom score: BIC + sparsity bonus."""
    bic = BICScore().score(adj, data)
    sparsity_bonus = -0.1 * adj.sum()  # Prefer sparse graphs
    return bic + sparsity_bonus

# Custom descriptor
def my_descriptor(adj, data):
    """Custom 3D descriptor."""
    n = adj.shape[0]
    density = adj.sum() / (n * (n-1))
    max_indeg = adj.sum(axis=0).max() / (n-1)
    # Third dimension: fraction of nodes with no parents
    root_frac = (adj.sum(axis=0) == 0).sum() / n
    return np.array([density, max_indeg, root_frac])

# Use with 3D archive
config_3d = MAPElitesConfig(
    archive_dims=(5, 5, 5),
    archive_ranges=((0, 1), (0, 1), (0, 1)),
)

engine_3d = CausalMAPElites(
    mutations=mutations,
    crossovers=crossovers,
    descriptor_fn=my_descriptor,
    score_fn=my_score,
    config=config_3d,
)
```

### Example 5: Intervention Design from Archive

```python
from causal_qd.equivalence import InterventionDesign
from causal_qd.mec import CPDAGConverter

# Find edges that are uncertain across the archive
edge_counts = {}
for entry in archive.entries:
    adj = entry.solution
    for i in range(adj.shape[0]):
        for j in range(adj.shape[0]):
            if adj[i, j]:
                edge_counts[(i,j)] = edge_counts.get((i,j), 0) + 1

# Edges present in >50% but <100% of archive are uncertain
uncertain = {e: c for e, c in edge_counts.items()
             if 0.5 * archive.size < c < archive.size}
print(f"Uncertain edges: {uncertain}")

# Design interventions to resolve uncertainty
converter = CPDAGConverter()
cpdag = converter.dag_to_cpdag(DAG(archive.best().solution))
designer = InterventionDesign()
targets = designer.optimal_targets(cpdag, budget=2)
print(f"Optimal intervention targets: {targets}")
```

### Example 6: Comparing Multiple Configurations

```python
import json

configs = {
    'small_archive': MAPElitesConfig(archive_dims=(5, 5), seed=42),
    'medium_archive': MAPElitesConfig(archive_dims=(10, 10), seed=42),
    'large_archive': MAPElitesConfig(archive_dims=(20, 20), seed=42),
    'adaptive': MAPElitesConfig(archive_dims=(10, 10), seed=42,
                                adaptive_operators=True),
    'curiosity': MAPElitesConfig(archive_dims=(10, 10), seed=42,
                                 selection_strategy='curiosity'),
}

results = {}
for name, cfg in configs.items():
    eng = CausalMAPElites(
        mutations=mutations, crossovers=crossovers,
        descriptor_fn=lambda a, d: desc.compute(a, d),
        score_fn=lambda a, d: scorer.score(a, d),
        config=cfg,
    )
    arch = eng.run(data, n_iterations=500, batch_size=32)
    best = arch.best()
    results[name] = {
        'size': arch.size,
        'coverage': arch.coverage(),
        'qd_score': arch.qd_score(),
        'best_shd': int(SHD.compute(best.solution, true_adj)),
    }
    print(f"{name}: size={arch.size}, SHD={results[name]['best_shd']}")

print(json.dumps(results, indent=2))
```

---

## Protocol Reference

### ScoreFunction Protocol

Any class implementing `score()` and `local_score()` can be used:

```python
class ScoreFunction(ABC):
    @abstractmethod
    def score(
        self,
        adj: AdjacencyMatrix,
        data: DataMatrix,
    ) -> float:
        """Total DAG score (higher is better)."""
        ...

    @abstractmethod
    def local_score(
        self,
        node: int,
        parents: list[int],
        data: DataMatrix,
    ) -> float:
        """Per-node local score contribution."""
        ...
```

### MutationOperator Protocol

```python
class MutationOperator(ABC):
    @abstractmethod
    def mutate(
        self,
        dag: AdjacencyMatrix,
        rng: np.random.Generator,
    ) -> AdjacencyMatrix:
        """Return a mutated copy of dag. Must preserve acyclicity."""
        ...
```

### CrossoverOperator Protocol

```python
class CrossoverOperator(ABC):
    @abstractmethod
    def crossover(
        self,
        parent1: AdjacencyMatrix,
        parent2: AdjacencyMatrix,
        rng: np.random.Generator,
    ) -> tuple[AdjacencyMatrix, AdjacencyMatrix]:
        """Return two offspring from two parents."""
        ...
```

### RepairOperator Protocol

```python
class RepairOperator(ABC):
    @abstractmethod
    def repair(
        self,
        dag: AdjacencyMatrix,
    ) -> AdjacencyMatrix:
        """Repair a possibly invalid adjacency matrix into a valid DAG."""
        ...
```

### CITest Protocol

```python
class CITest(ABC):
    @abstractmethod
    def test(
        self,
        data: DataMatrix,
        x: int,
        y: int,
        conditioning_set: list[int],
    ) -> tuple[bool, float]:
        """Test conditional independence X ⊥ Y | Z.
        Returns (is_independent, p_value)."""
        ...
```

### BaselineAlgorithm Protocol

```python
class BaselineAlgorithm(ABC):
    @abstractmethod
    def run(
        self,
        data: DataMatrix,
    ) -> AdjacencyMatrix:
        """Run the algorithm and return a DAG adjacency matrix."""
        ...
```

---

## Error Handling

### Common Exceptions

| Exception | Raised by | Cause |
|-----------|-----------|-------|
| `ValueError` | `DAG.__init__` | Adjacency matrix contains a cycle |
| `ValueError` | `StructuralDescriptor.__init__` | Unknown feature name |
| `ValueError` | `MAPElitesConfig` | Invalid archive dimensions |
| `IndexError` | `ScoreFunction.local_score` | Node index out of range |
| `np.linalg.LinAlgError` | `BICScore.score` | Singular covariance matrix |

### Handling Score Failures

Score evaluation can fail for degenerate DAGs (e.g., disconnected nodes
with zero variance). The engine catches these and skips the candidate:

```
# This warning is normal and can be ignored:
"Evaluation failed for a candidate; skipping."
```

To suppress warnings:
```python
import warnings
warnings.filterwarnings('ignore')
```

---

## Type Annotations

All public APIs are fully typed. Key type aliases:

```python
from causal_qd.core import (
    AdjacencyMatrix,         # np.ndarray, shape (n, n), dtype int8
    DataMatrix,              # np.ndarray, shape (N, p), dtype float64
    BehavioralDescriptor,    # np.ndarray, shape (d,), dtype float64
    QualityScore,            # float
    WeightedAdjacencyMatrix, # np.ndarray, shape (n, n), dtype float64
    BootstrapSample,         # np.ndarray, shape (N,), dtype int64
)

# Callable types used by CausalMAPElites
from typing import Callable
import numpy as np

ScoreFn = Callable[[AdjacencyMatrix, DataMatrix], float]
DescriptorFn = Callable[[AdjacencyMatrix, DataMatrix], np.ndarray]
MutationOp = Callable[[AdjacencyMatrix, np.random.Generator], AdjacencyMatrix]
CrossoverOp = Callable[
    [AdjacencyMatrix, AdjacencyMatrix, np.random.Generator],
    AdjacencyMatrix,
]
CallbackFn = Callable[['IterationStats'], None]
```

---

## Internal Architecture

### Engine Internals

The `CausalMAPElites` engine maintains:
- `_archive`: Internal `_GridArchive` instance
- `_mutations`: List of mutation callables
- `_crossovers`: List of crossover callables
- `_descriptor_fn`: Descriptor callable
- `_score_fn`: Score callable
- `_config`: `MAPElitesConfig` instance
- `_bandit`: UCB1 bandit (if `adaptive_operators=True`)
- `_rng`: NumPy random generator
- `_stats_tracker`: Running statistics for early stopping
- `_iteration`: Current iteration counter
- `_history`: List of `IterationStats`
- `_start_time`: Timestamp of `run()` call

### Archive Internals

The `_GridArchive` uses a flat array of `ArchiveEntry | None`:
- `_grid`: Array of size `prod(dims)`, each element is an `ArchiveEntry` or `None`
- `_dims`: Grid dimensions tuple
- `_ranges`: Per-dimension value ranges
- `_fill_count`: Number of cells first filled
- `_replace_count`: Number of quality improvements

Cell indexing: descriptor vector → grid indices via linear interpolation:
```
idx_d = int((desc[d] - lo_d) / (hi_d - lo_d) * dims[d])
idx_d = clamp(idx_d, 0, dims[d] - 1)
flat_idx = idx_0 * dims[1] + idx_1  # for 2D
```

### Score Caching

Score functions use decomposable caching: when a mutation changes only
edges involving node $j$, only $s_j$ needs recomputation. The engine
does not currently implement this optimisation (all scores are computed
from scratch), but the `local_score` API enables it for future work.

---

## Performance Tips

1. **Use adaptive operators** (`adaptive_operators=True`): The UCB1 bandit
   typically improves both quality and diversity by 10-30%.

2. **Match archive dims to the problem**: For p<10, use 5×5 or 8×8.
   For p>15, use 8×8 or 10×10. Larger archives need more iterations.

3. **Include VStructure and Skeleton mutations**: These are the most
   impactful operators for MEC diversity.

4. **Warm-start with baselines**: Seeding the archive with PC/GES results
   significantly improves convergence speed.

5. **Use appropriate sample sizes**: N ≥ 100p is a good rule of thumb.
   Too few samples → noisy scores → poor convergence.

6. **Batch size 16-64**: Smaller batches give more archive updates per
   iteration; larger batches amortise overhead.

7. **Early stopping**: Set `early_stopping_window=100` and
   `early_stopping_threshold=1e-4` to automatically stop when converged.

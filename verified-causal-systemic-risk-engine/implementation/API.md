# CausalBound API Reference

Only documents implemented, exported classes with actual method signatures.

---

## Package Exports

```python
from causalbound import (
    TreeDecomposer, CausalPolytopeSolver, BoundComposer,
    JunctionTreeEngine, SMTVerifier, MCTSSearch, SCMBuilder, DebtRankModel,
)
```

---

## composition — Formal Proof & Bound Composition

**Exports:** `BoundComposer`, `GapEstimator`, `SeparatorConsistencyChecker`, `MonotoneBoundPropagator`, `GlobalBoundAggregator`, `CompositionTheorem`, `FormalProofEngine`, `FormalProofResult`

### `FormalProofEngine`

```python
class FormalProofEngine:
    def __init__(self, timeout_ms: int = 10000, epsilon: float = 1e-9)

    def verify_composition_theorem(
        self,
        n_subgraphs: int, n_separators: int, max_separator_size: int,
        lipschitz_constant: float, discretization: float,
        subgraph_lower_bounds: List[float], subgraph_upper_bounds: List[float],
        separator_variables_per_boundary: Optional[List[int]] = None,
    ) -> FormalProofResult
```

### `FormalProofResult`

```python
@dataclass
class FormalProofResult:
    theorem_name: str
    obligations: List[ProofObligation]
    all_verified: bool
    total_verification_time_s: float
    gap_bound_verified: bool
    validity_verified: bool
    certificate_hash: str = ""

    @property
    def n_verified(self) -> int
    @property
    def n_total(self) -> int
    def summary(self) -> str

@dataclass
class ProofObligation:
    obligation_id: str
    lemma_name: str
    description: str
    hypotheses: List[Any]       # z3 BoolRef
    conclusion: Any             # z3 BoolRef
    status: str = "pending"     # pending | verified | failed | timeout
    verification_time_s: float = 0.0
    z3_result: str = ""
    unsat_core: Optional[List[str]] = None
```

### `BoundComposer`

```python
class BoundComposer:
    def __init__(
        self, strategy: CompositionStrategy = CompositionStrategy.WORST_CASE,
        tolerance: float = 1e-8, max_iterations: int = 100,
        lipschitz_constant: Optional[float] = None,
    )
    def compose(self, subgraph_bounds: List[SubgraphBound],
                separator_info: List[SeparatorInfo],
                overlap_structure: OverlapStructure) -> CompositionResult
    def get_global_bounds(self) -> Tuple[np.ndarray, np.ndarray]
    def refine(self, n_iterations: int = 10) -> CompositionResult
    def get_composition_gap(self) -> float
```

**`CompositionStrategy`** enum: `WORST_CASE`, `AVERAGE_CASE`, `WEIGHTED`, `MINIMAX`, `BAYESIAN`

**Key data classes:** `SubgraphBound(subgraph_id, lower, upper, separator_vars, weight, confidence)`, `SeparatorInfo(separator_id, variable_indices, adjacent_subgraphs, cardinality, marginal)`, `OverlapStructure(n_subgraphs, overlap_matrix, shared_variables)`, `CompositionResult(global_lower, global_upper, composition_gap, strategy_used, n_iterations, converged)`

---

## graph — Tree Decomposition

**Exports:** `TreeDecomposer`, `SeparatorExtractor`, `CausalPartitioner`, `TreewidthEstimator`, `SubgraphExtractor`, `MoralGraphConstructor`

```python
class TreeDecomposer:
    def __init__(self, strategy: str = "min_fill")
    # strategy: "min_fill" | "min_degree" | "min_width"
    def decompose(self, graph: nx.Graph, max_width: Optional[int] = None,
                  refine: bool = True, timeout: Optional[float] = None) -> TreeDecomposition
    def get_elimination_ordering(self) -> List[int]
    def validate_decomposition(self, decomp: TreeDecomposition, graph: nx.Graph) -> bool

@dataclass
class TreeDecomposition:
    bags: Dict[int, FrozenSet[int]]
    tree: nx.Graph
    width: int
    elimination_ordering: List[int]
    @property
    def num_bags(self) -> int
    @property
    def max_bag_size(self) -> int
```

---

## polytope — Causal Polytope LP

**Exports:** `CausalPolytopeSolver`, `SolverResult`, `SolverConfig`, `ColumnGenerationSolver`, `ConstraintEncoder`, `InterventionalPolytope`, `BoundExtractor`, `BoundResult`, `SensitivityReport`

```python
class CausalPolytopeSolver:
    def __init__(self, config: Optional[SolverConfig] = None)
    def solve(self, dag: DAGSpec, query: QuerySpec,
              observed: Optional[ObservedMarginals] = None) -> SolverResult

@dataclass
class DAGSpec:
    nodes: List[str]
    edges: List[Tuple[str, str]]
    card: Dict[str, int]
    def parents(self, node) -> List[str]
    def children(self, node) -> List[str]
    def topological_order(self) -> List[str]

@dataclass
class QuerySpec:
    target_var: str
    target_val: int
    interventions: Optional[List[InterventionSpec]] = None
    conditioning: Optional[Dict[str, int]] = None

@dataclass
class SolverConfig:
    max_iterations: int = 500
    gap_tolerance: float = 1e-8
    time_limit: float = 300.0
    pricing_strategy: str = "exact"
    warm_start: bool = True
    stabilization: bool = True

@dataclass
class SolverResult:
    lower_bound: float
    upper_bound: float
    status: SolverStatus  # OPTIMAL | INFEASIBLE | UNBOUNDED | ITERATION_LIMIT | TIME_LIMIT
    lower_certificate: Optional[DualCertificate] = None
    upper_certificate: Optional[DualCertificate] = None
    diagnostics: Optional[SolverDiagnostics] = None
    identifiable: bool = False

@dataclass
class ObservedMarginals:
    marginals: Dict[FrozenSet[str], np.ndarray]
```

---

## smt — SMT Verification

**Exports:** `SMTVerifier`, `SMTEncoder`, `CertificateEmitter`, `IncrementalProtocol`, `GraphPredicateEncoder`, `QFLRAEncoder`, `AletheProofExtractor`, `DiscretizationVerifier`

```python
class SMTVerifier:
    def __init__(self, timeout_ms: int = 10_000, track_unsat_cores: bool = True,
                 emit_certificates: bool = True, epsilon: float = 1e-9)

    def begin_session(self, session_id: Optional[str] = None) -> str
    def end_session(self) -> SessionStats

    def verify_bound(self, lower: float, upper: float, evidence: BoundEvidence) -> VerificationResult
    def verify_message(self, sender: str, receiver: str, message_data: MessageData) -> VerificationResult
    def verify_dsep_claim(self, x: str, y: str, z_set: Set[str], dag_edges: List[Tuple]) -> VerificationResult
    def verify_normalization(self, distribution: Dict[str, float]) -> VerificationResult
    def verify_monotonicity(self, values: List[Tuple[str, float]]) -> VerificationResult

class VerificationStatus(Enum):  # PASS, FAIL, UNKNOWN, TIMEOUT, SKIPPED

@dataclass
class VerificationResult:
    step_id: str
    status: VerificationStatus
    assertion_count: int
    smt_time_s: float
    message: str = ""
    @property
    def passed(self) -> bool

@dataclass
class BoundEvidence:
    lp_objective: Optional[float] = None
    dual_values: Optional[List[float]] = None
    reduced_costs: Optional[List[float]] = None
    basis_indices: Optional[List[int]] = None
    farkas_coefficients: Optional[List[float]] = None

@dataclass
class SessionStats:
    total_steps: int = 0
    passed_steps: int = 0
    failed_steps: int = 0
    total_smt_time_s: float = 0.0
    total_inference_time_s: float = 0.0
    @property
    def smt_overhead_ratio(self) -> float
    def summary(self) -> str
```

---

## junction — Junction-Tree Inference

**Exports:** `JunctionTreeEngine`, `CliqueTree`, `CliqueNode`, `MessagePasser`, `PotentialTable`, `AdaptiveDiscretizer`, `DoOperator`, `InferenceCache`, `DiscretizationErrorAnalyzer`

```python
class JunctionTreeEngine:
    def __init__(self, variant: MessagePassingVariant = MessagePassingVariant.HUGIN,
                 use_log_space: bool = False, cache_capacity: int = 4096,
                 default_bins: int = 20, binning_strategy: BinningStrategy = BinningStrategy.QUANTILE)

    def build(self, dag, cpds, cardinalities, elimination_order=None) -> CliqueTree
    def build_from_data(self, dag, data, n_bins=None, strategy=None) -> CliqueTree
    def calibrate(self, root=None, evidence=None) -> PassingStats
    def query(self, target: str, evidence=None, intervention=None) -> QueryResult
    def get_marginal(self, variable: str) -> PotentialTable
    def compute_expected_value(self, target, intervention=None, evidence=None) -> float
    def compute_tail_probability(self, target, threshold, intervention=None, evidence=None) -> float

# MessagePassingVariant enum: HUGIN, SHAFER_SHENOY
# BinningStrategy enum: UNIFORM, QUANTILE, TAIL_PRESERVING, ENTROPY_OPTIMAL, INSTRUMENT_SPECIFIC

@dataclass
class QueryResult:
    target: str
    distribution: NDArray
    expected_value: float
    variance: float
    inference_time_s: float = 0.0
    def tail_probability(self, threshold: float) -> float
    def quantile(self, q: float) -> float
    def summary(self) -> str
```

---

## mcts — Adversarial Search

**Exports:** `MCTSNode`, `CausalUCB`, `MCTSSearch`, `RolloutScheduler`, `DSeparationPruner`, `ConvergenceMonitor`

```python
class MCTSSearch:
    def __init__(self, config: Optional[SearchConfig] = None, dag=None, random_seed: Optional[int] = None)
    def search(self, interface_vars: List[str], inference_engine: Any,
               target_variable: str) -> SearchResult

@dataclass
class SearchConfig:
    n_rollouts: int = 10000
    budget_seconds: float = 300.0
    exploration_constant: float = 1.414
    maximize: bool = True
    enable_pruning: bool = True
    shock_range: Tuple[float, float] = (0.0, 1.0)

@dataclass
class SearchResult:
    best_scenario: ScenarioReport
    all_scenarios: List[ScenarioReport]
    converged: bool
    total_rollouts: int
    elapsed_seconds: float

@dataclass
class ScenarioReport:
    state: Dict[str, float]
    value: float
    visit_count: int
    confidence_interval: Tuple[float, float]
    rank: int
```

---

## network — Network Generators

**Exports:** `ErdosRenyiGenerator`, `ScaleFreeGenerator`, `CorePeripheryGenerator`, `SmallWorldGenerator`, `NetworkTopology`, `NetworkCalibrator`, `TopologyLoader`

```python
class BaseNetworkGenerator:
    def __init__(self, exposure_params: Optional[ExposureParams] = None, seed: Optional[int] = None)
    def generate(self, n_nodes: int, **params) -> nx.DiGraph

# Concrete generators:
ErdosRenyiGenerator.generate(n_nodes, density=0.1, reciprocity=0.3)
ScaleFreeGenerator.generate(n_nodes, m=3, alpha=1.0, reciprocity=0.3)
CorePeripheryGenerator.generate(n_nodes, core_fraction=0.2, core_density=0.8, periphery_density=0.05)
SmallWorldGenerator.generate(n_nodes, k=4, beta=0.3, reciprocity=0.3)

@dataclass
class ExposureParams:
    distribution: str       # "pareto" | "lognormal" | "uniform"
    scale: float = 1e8
    shape: float = 1.5
```

---

## contagion — Contagion Models

**Exports:** `DebtRankModel`, `CascadeModel`, `FireSaleModel`, `MarginSpiralModel`, `FundingLiquidityModel`, `ContagionModelVerifier`

```python
class DebtRankModel:
    def __init__(self, variant: DebtRankVariant = DebtRankVariant.LINEAR,
                 default_threshold: float = 0.5, nonlinear_exponent: float = 2.0)
    def compute(self, graph: nx.DiGraph, initial_shocks: Dict[int, float],
                max_rounds: int = 100, track_history: bool = False) -> DebtRankResult
    def sensitivity_analysis(self, graph, shock_level=1.0, max_rounds=100) -> SensitivityResult

# DebtRankVariant enum: LINEAR, THRESHOLD, NONLINEAR

@dataclass
class DebtRankResult:
    system_debtrank: float
    node_debtranks: Dict[int, float]
    final_distress: np.ndarray
    rounds_propagated: int
    total_loss: float
    cascade_size: int

@dataclass
class SensitivityResult:
    node_impacts: Dict[int, float]
    top_k_nodes: List[Tuple[int, float]]
    system_vulnerability: float
```

---

## evaluation — Benchmarking

**Exports:** `MonteCarloGroundTruth`, `BenchmarkRunner`, `CrisisReconstructor`, `MetricsComputer`, `AdversarialEvaluator`

```python
class CrisisReconstructor:
    def __init__(self, seed: int = 42)
    def reconstruct(self, crisis_name: str) -> CrisisTopology
    # Supported: "gfc_2008", "eu_sovereign_2010", "covid_treasury_2020", "uk_gilt_2023"

@dataclass
class CrisisTopology:
    name: str
    graph: nx.DiGraph
    node_metadata: Dict[int, Any]
    edge_weights: Dict[Tuple[int, int], float]
    known_scenarios: List[Dict[str, float]]
    historical_losses: Dict[int, float]
```

---

## data — Serialization & Caching

**Exports:** `NetworkSerializer`, `SCMSerializer`, `BoundSerializer`, `CacheManager`, `CheckpointManager`

All serializers provide `save(obj, path)` and `load(path)` with JSON format, checksums, and version migration. `CacheManager` supports LRU/LFU eviction. `CheckpointManager` enables pipeline resumption from any stage.

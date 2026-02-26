# MARACE API Reference

> Focused reference for the `marace` package — only implemented classes and key methods.

---

## 1. Pipeline (`marace.pipeline`)

```python
@dataclass
class PipelineConfig:
    env_config: dict
    policy_paths: List[str]
    spec_path: str = ""
    num_trace_samples: int = 1000
    max_schedule_depth: int = 20
    abstraction_domain: str = "interval"    # "interval" | "zonotope"
    adversarial_budget: int = 100
    importance_samples: int = 10000
    parallel: bool = True
    timeout_s: float = 3600.0
    report_formats: List[str] = ["text"]
    generate_certificates: bool = True
    # Also: trace_paths, output_dir, checkpoint_dir, max_workers, verbose

    @classmethod
    def from_yaml(cls, path: str) -> PipelineConfig
    @classmethod
    def from_json(cls, path: str) -> PipelineConfig

class MARACEPipeline:
    def __init__(self, config: PipelineConfig) -> None
    def run(self) -> PipelineState      # runs all stages end-to-end
    def summary(self) -> str
```

**Example:**
```python
config = PipelineConfig(env_config={"type": "highway", "num_agents": 4},
                        policy_paths=["p0.onnx", "p1.onnx"], spec_path="spec.marace")
result = MARACEPipeline(config).run()
```

---

## 2. Abstract Interpretation (`marace.abstract`)

```python
@dataclass
class Zonotope:
    center: np.ndarray;  generators: np.ndarray

    dimension: int;  num_generators: int;  volume_bound: float  # properties

    @staticmethod
    def from_interval(lower: np.ndarray, upper: np.ndarray) -> Zonotope
    def affine_transform(self, W: np.ndarray, b=None) -> Zonotope
    def join(self, other: Zonotope) -> Zonotope
    def meet_halfspace(self, a: np.ndarray, b: float) -> Zonotope
    def widening(self, other: Zonotope, threshold: float = 1.05) -> Zonotope
    def contains_point(self, x: np.ndarray) -> bool
    def sample(self, n: int) -> np.ndarray
    def reduce_generators(self, max_gens: int) -> Zonotope
    def project(self, dims: Sequence[int]) -> Zonotope
    def hausdorff_upper_bound(self, other: Zonotope) -> float
    def maximize(self, direction: np.ndarray) -> Tuple[float, np.ndarray]
    def to_dict(self) -> Dict;  @staticmethod from_dict(d) -> Zonotope

class FixpointEngine:
    def __init__(self, transfer_fn, strategy=WideningStrategy.DELAYED,
                 max_iterations=100, convergence_threshold=1e-6,
                 hb_constraints: Optional[HBConstraintSet] = None) -> None
    def compute(self, initial: Zonotope) -> FixpointResult

@dataclass
class FixpointResult:
    element: Zonotope;  converged: bool;  iterations: int
    wall_time_s: float;  hb_consistent: bool
    def summary(self) -> str

class HBConstraintSet:
    def __init__(self, constraints=None) -> None
    def add(self, constraint: HBConstraint) -> None
    def as_matrices(self) -> Tuple[np.ndarray, np.ndarray]
    def satisfied_by_point(self, x: np.ndarray) -> bool
    def violated_constraints(self, x: np.ndarray) -> List[HBConstraint]
```

---

## 3. Decomposition (`marace.decomposition`)

```python
class InteractionGraph:
    def __init__(self) -> None
    def add_agent(self, agent_id: str) -> None
    def add_interaction(self, edge: InteractionEdge) -> None
    def coupling_strength(self, source: str, target: str) -> float
    def neighbours(self, agent_id: str) -> Set[str]
    def connected_components(self) -> List[FrozenSet[str]]
    def adjacency_matrix(self, agent_order=None) -> np.ndarray
    def laplacian_matrix(self, agent_order=None) -> np.ndarray
    def iter_edges(self) -> Iterator[InteractionEdge]

class LinearContract(Contract):
    # Contract(name, assumption, guarantee, interface_vars)
    def assumption_matrix(self, var_order) -> Tuple[np.ndarray, np.ndarray]
    def guarantee_matrix(self, var_order) -> Tuple[np.ndarray, np.ndarray]
    def is_feasible(self, var_order) -> bool

class AssumeGuaranteeVerifier:
    def __init__(self, max_circular_iterations=20, max_group_size=6) -> None
    def verify(self, group_contracts, inter_group_contracts,
               verify_group_fn, dependency_graph=None) -> CompositionResult

class SpectralPartitioner:
    def __init__(self, num_groups: int = 2) -> None
    def partition(self, graph: InteractionGraph) -> Partition

class GraphConstructor:
    def build(self, agents, hb_edges, trajectories=None, observations=None) -> InteractionGraph
```

---

## 4. Environments (`marace.env`)

```python
class HighwayEnv(MultiAgentEnv):
    def __init__(self, num_agents=4, scenario_type=ScenarioType.HIGHWAY,
                 road=None, sensor_range=100.0, dt=0.1, max_steps=500,
                 stepping=None, timing_configs=None) -> None
    def reset(self) -> Dict[str, np.ndarray]
    def step_async(self, agent_id, action) -> Tuple[np.ndarray, float, bool, Dict]
    def step_sync(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, bool, Dict]
    def get_state(self) -> HighwayState
    def set_state(self, state) -> None
    vehicles: Dict[str, VehicleState]          # property
    collision_log: List[Tuple[int, str, str]]   # property
    def safety_predicates(self) -> SafetyPredicates

class WarehouseEnv(MultiAgentEnv):
    def __init__(self, num_robots=6, layout=None, sensor_range=5.0,
                 max_steps=1000, dt=0.1, auto_task=True) -> None
    # Same MultiAgentEnv interface (reset, step_async, step_sync, etc.)
    robots: Dict[str, RobotState]   # property
    tasks: TaskAssignment            # property
    def safety_predicates(self) -> WarehouseSafetyPredicates
```

Scenario types: `HIGHWAY`, `INTERSECTION`, `MERGING`, `OVERTAKING`

---

## 5. HB Analysis (`marace.hb`)

```python
class HBGraph:
    def __init__(self, name: str = "hb_graph") -> None
    def add_event(self, event_id: str, **attrs) -> None
    def add_hb_edge(self, earlier: str, later: str) -> None
    def query_hb(self, e1: str, e2: str) -> HBRelation   # BEFORE|AFTER|CONCURRENT|EQUAL
    def happens_before(self, earlier: str, later: str) -> bool
    def are_concurrent(self, e1: str, e2: str) -> bool
    def concurrent_pairs(self) -> List[Tuple[str, str]]
    def connected_components(self) -> List[Set[str]]
    def extract_interaction_groups(self) -> List[InteractionGroup]
    def subgraph_for_agents(self, agent_ids: Set[str]) -> HBGraph
    def has_cycles(self) -> bool
    def longest_chain(self) -> List[str]
    def statistics(self) -> Dict[str, Any]
    num_events: int;  num_edges: int  # properties
    def to_dict(self) -> Dict;  @classmethod from_dict(cls, data) -> HBGraph

class VectorClock:
    def __init__(self, initial: Optional[Dict[str, int]] = None) -> None
    def increment(self, agent_id: str) -> VectorClock
    def merge(self, other: VectorClock) -> VectorClock
    def get(self, agent_id: str) -> int
    def compare(self, other: VectorClock) -> ClockComparison
    def happens_before(self, other: VectorClock) -> bool
    def concurrent_with(self, other: VectorClock) -> bool

class CausalInferenceEngine:
    def __init__(self, obs_analyzer=None, physics_detector=None,
                 comm_extractor=None, env_chain=None, min_confidence=0.1) -> None
    def infer(self, events, comm_events=None, env_states=None,
              agent_positions=None) -> List[CausalEdge]
```

---

## 6. Policy (`marace.policy`)

```python
class ONNXPolicy:
    def __init__(self, model_path=None, architecture=None, io_spec=None) -> None
    architecture: NetworkArchitecture  # property
    def evaluate(self, observation: np.ndarray) -> np.ndarray
    def evaluate_batch(self, observations: np.ndarray) -> np.ndarray

class LipschitzExtractor:
    def __init__(self, method="spectral_product", max_power_iterations=100) -> None
    def extract(self, architecture: NetworkArchitecture) -> LipschitzCertificate
    def extract_local(self, arch, lower, upper) -> LipschitzCertificate

class LipSDPBound:
    def __init__(self, fallback=None) -> None
    def compute(self, arch: NetworkArchitecture) -> BoundResult

@dataclass
class LipschitzCertificate:
    global_bound: float;  per_layer_bounds: List[float]
    method: str;  is_tight: bool = False
    def summary(self) -> str

@dataclass
class BoundResult:
    upper_bound: float;  lower_bound: Optional[float] = None
    tightness_ratio: Optional[float] = None;  method: str = ""
    def summary(self) -> str
```

---

## 7. Race Detection (`marace.race`)

```python
@dataclass
class InteractionRace:
    race_id: str;  agents: List[str]
    events: Tuple[ScheduleEvent, ...]
    condition: Optional[RaceCondition] = None
    witness: Optional[RaceWitness] = None
    absence: Optional[RaceAbsence] = None
    classification: RaceClassification = RaceClassification.CUSTOM
    probability: float = 0.0
    is_confirmed: bool;  is_certified_absent: bool;  severity: float  # properties
    def summary(self) -> str
    def to_dict(self) -> Dict

class EpsilonCalibrator:
    def __init__(self, lipschitz_constant=1.0, global_safety_margin=1.0,
                 convergence_threshold=1e-6, max_iterations=50) -> None
    def calibrate(self, center: np.ndarray, initial_epsilon=None) -> EpsilonRace
    history: List[CalibrationStep];  converged: bool  # properties

class RaceCatalog:
    def __init__(self) -> None
    def add(self, entry: CatalogEntry) -> None
    def get(self, entry_id: str) -> Optional[CatalogEntry]
    def sorted_by_severity(self, descending=True) -> List[CatalogEntry]
    def filter_by_classification(self, classification) -> List[CatalogEntry]
    def filter_by_agent(self, agent_id: str) -> List[CatalogEntry]
    entries: List[CatalogEntry]  # property

class CatalogBuilder:
    def add_race(self, race, probability_bound=1.0, tags=None) -> CatalogEntry
    def add_absence(self, certificate, tags=None) -> CatalogEntry
    def build(self) -> RaceCatalog
```

Classifications: `COLLISION`, `DEADLOCK`, `STARVATION`, `PRIORITY_INVERSION`, `CUSTOM`

---

## 8. Sampling (`marace.sampling`)

```python
class ScheduleSpace:
    def __init__(self, agents, num_timesteps, constraints=()) -> None
    def is_valid(self, schedule: Schedule) -> bool
    def total_events(self) -> int

class ImportanceSampler:
    def __init__(self, target_log_prob, proposal: ProposalDistribution) -> None
    def sample_and_weight(self, n, rng=None) -> Tuple[List[Schedule], ImportanceWeights]
    def estimate(self, f, n, rng=None) -> float

class RaceProbabilityEstimator:
    def __init__(self, space, race_checker, target_log_prob, proposal,
                 num_samples=1000, confidence_level=0.95) -> None
    def estimate(self, rng=None) -> ConfidenceInterval
    def adaptive_estimate(self, max_rounds=10, samples_per_round=500) -> ConfidenceInterval

# Schedule distributions (all implement ScheduleMeasure)
class UniformHBConsistentMeasure:   # uniform over HB-consistent schedules
    def __init__(self, space, rng=None) -> None
class PlackettLuceMeasure:          # Plackett-Luce ranking model
    def __init__(self, events, weights=None, space=None) -> None
    @classmethod def fit_mle(cls, events, samples, space=None) -> PlackettLuceMeasure
# Common interface: log_prob(schedule), sample(n), support_size()
```

---

## 9. Search (`marace.search`)

```python
class MCTS:
    def __init__(self, agent_ids, max_depth, hb_graph: HBGraph,
                 safety_evaluator: Callable[[np.ndarray, List[ScheduleAction]], float],
                 exploration_constant=1.414, timing_range=(0.0, 1.0), seed=None) -> None
    def search(self, initial_state: np.ndarray, budget: SearchBudget) -> SearchResult

@dataclass
class SearchBudget:
    iteration_count: int = 10_000
    time_limit_seconds: float = 60.0
    max_nodes: int = 500_000

@dataclass
class SearchResult:
    best_schedule: List[ScheduleAction];  safety_margin: float
    replay_trace: Optional[List[np.ndarray]];  statistics: Dict
    is_race_found: bool  # property
```

---

## 10. Specification (`marace.spec`)

```python
class SpecParser:
    def parse(self, text: str) -> TemporalFormula
    def parse_many(self, text: str) -> List[TemporalFormula]

# Temporal operators — all have evaluate(trace, t) -> bool and robustness(trace, t) -> float
class Always(TemporalFormula):    # __init__(predicate, horizon=None)
class Eventually(TemporalFormula) # __init__(predicate, horizon=None)
class Until(TemporalFormula):     # __init__(pred1, pred2, horizon=None)
class Next(TemporalFormula):      # __init__(predicate)
class BoundedResponse:            # __init__(trigger, response, deadline=10)

# Safety predicates — all have evaluate(state) -> bool and robustness(state) -> float
class CollisionPredicate:         # __init__(agent_i, agent_j, extents_i, extents_j)
class DistancePredicate:          # __init__(agent_i, agent_j, threshold, pos_indices)
class RegionPredicate:            # __init__(agent_id, low, high, pos_indices)

class SafetyLibrary:
    def __init__(self, agent_ids: List[str]) -> None
    def highway_safety(self, min_dist=3.0, horizon=None) -> TemporalFormula
    def warehouse_safety(self, min_dist=1.0, task_deadline=500) -> TemporalFormula
    def collision_freedom(self, extents=(2.25, 1.0)) -> TemporalFormula
    def min_separation(self, d=3.0) -> TemporalFormula
    def deadlock_freedom(self, horizon=None) -> TemporalFormula
    def liveness(self, horizon=None) -> TemporalFormula
```

---

## 11. Reporting (`marace.reporting`)

```python
@dataclass
class ProofCertificate:
    certificate_id: str;  timestamp: datetime
    environment_id: str;  policy_ids: List[str]
    specification: str;  verdict: str
    coverage_fraction: float;  abstract_domain_used: str
    fixpoint_iterations: int;  races_found: List[Dict]
    hash_digest: str
    def to_dict(self) -> Dict
    @classmethod def from_dict(cls, data) -> ProofCertificate

class CertificateBuilder:
    def __init__(self, env_id, num_agents, state_dim, action_dims) -> None
    def set_verdict(self, verdict) -> CertificateBuilder
    def set_policies(self, policies) -> CertificateBuilder
    def set_specification(self, ...) -> CertificateBuilder
    def set_fixpoint_result(self, ...) -> CertificateBuilder
    def set_hb_consistency(self, ...) -> CertificateBuilder
    def set_composition(self, ...) -> CertificateBuilder
    def add_race_witness(self, ...) -> CertificateBuilder
    def build(self) -> Dict[str, Any]

class IndependentCertificateChecker:
    def __init__(self, tolerance: float = 1e-8) -> None
    def check(self, cert: Dict[str, Any]) -> CheckResult

@dataclass
class CheckResult:
    overall_passed: bool;  component_results: List[ComponentCheckResult]
    certificate_hash_valid: bool
    def summary(self) -> str

# Report formatters — all implement format(report: RaceReport) -> str
class TextReportFormatter(ReportFormatter)
class JSONReportFormatter(ReportFormatter)   # __init__(indent=2)
class HTMLReportFormatter(ReportFormatter)
class LaTeXReportFormatter(ReportFormatter)

# TCB analysis
class TCBAnalyzer:
    def analyze_codebase(self, root_path: str) -> TCBReport
    def compute_tcb_size(self) -> int
    def identify_critical_path(self) -> List[str]

class AletheCertificateAdapter:
    def convert(self, certificate: Dict) -> str   # returns Alethe s-expression
    @staticmethod
    def parse(alethe_str: str) -> Dict             # parse back

class IndependentChecker:
    def check(self, certificate: Dict) -> CheckResult
```

---

## 12. CEGAR Refinement (`marace.abstract.cegar`)

```python
class CEGARVerifier:
    def __init__(self, max_refinements=10, max_splits=8, timeout_s=60.0) -> None
    def verify(self, initial_zonotope: Zonotope,
               transfer_fn: Callable, safety_predicate: Callable,
               concrete_eval: Callable) -> CEGARResult

@dataclass
class CEGARResult:
    verdict: Verdict            # SAFE | UNSAFE | UNKNOWN
    counterexample: Optional[np.ndarray]
    refinement_iterations: int
    refinement_history: List[RefinementRecord]
    total_time_s: float

class SpuriousnessChecker:
    def __init__(self, concrete_eval_fn, num_samples=100) -> None
    def check(self, zonotope: Zonotope, safety_predicate: Callable) -> bool

class AbstractionRefinement:
    @staticmethod
    def split_counterexample_guided(z: Zonotope, cx: np.ndarray) -> List[Zonotope]
    @staticmethod
    def split_dimension(z: Zonotope, dim: int) -> Tuple[Zonotope, Zonotope]
    @staticmethod
    def split_gradient_guided(z: Zonotope, gradient: np.ndarray) -> List[Zonotope]

class CompositionalCEGARVerifier:
    def __init__(self, max_refinements=10) -> None
    def verify_groups(self, groups: Dict[str, Tuple]) -> Dict[str, CEGARResult]

class Verdict(Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    UNKNOWN = "unknown"

def make_cegar_verifier(max_refinements=10, max_splits=8,
                        timeout_s=60.0, strategy="counterexample_guided") -> CEGARVerifier:
    """Convenience factory. strategy: 'counterexample_guided' | 'dimension' | 'gradient'"""
```

**Typical usage (from `run_cegar_experiments.py`):**
```python
from marace.abstract.cegar import make_cegar_verifier, Verdict

verifier = make_cegar_verifier(max_refinements=10, max_splits=32,
                                strategy="counterexample_guided")
result = verifier.verify(
    initial_zonotope=zonotope,
    transfer_fn=lambda z: z.affine_transform(W, b).relu(),
    safety_predicate=collision_predicate,
    concrete_eval=evaluate_network_concrete
)
if result.verdict == Verdict.SAFE:
    print("Race is spurious (false positive)")
elif result.verdict == Verdict.UNSAFE:
    print(f"Real race — witness: {result.counterexample}")
```

---

## 13. Recurrent Policies (`marace.policy.recurrent`)

```python
@dataclass
class RecurrentNetworkArchitecture:
    cell_type: str            # "lstm" or "gru"
    input_dim: int
    hidden_dim: int
    output_dim: int
    unroll_horizon: int
    # Gate weight matrices (W_ih, W_hh, b_ih, b_hh for each gate)
    output_projection: Optional[Tuple[np.ndarray, np.ndarray]]

class LSTMCell:
    def __init__(self, arch: RecurrentNetworkArchitecture) -> None
    def forward(self, x: np.ndarray, h: np.ndarray, c: np.ndarray) -> Tuple
    def forward_abstract(self, x_z: Zonotope, h_z: Zonotope,
                         c_z: Zonotope) -> Tuple[Zonotope, Zonotope]

class GRUCell:
    def __init__(self, arch: RecurrentNetworkArchitecture) -> None
    def forward(self, x: np.ndarray, h: np.ndarray) -> np.ndarray
    def forward_abstract(self, x_z: Zonotope, h_z: Zonotope) -> Zonotope

class RecurrentAbstractEvaluator:
    def __init__(self, arch: RecurrentNetworkArchitecture,
                 max_generators: int = 50) -> None
    def evaluate(self, input_zonotope: Zonotope) -> AbstractOutput

class RecurrentLipschitzBound:
    def __init__(self, arch: RecurrentNetworkArchitecture) -> None
    def naive_bound(self) -> float          # Lip(f)^K
    def interval_bound(self) -> float       # with gate activation masking
    def exponential_decay_bound(self) -> float  # with forget gate analysis
```

---

## 14. Adaptive SIS (`marace.sampling.adaptive_sis`)

```python
class AdaptiveSISEngine:
    def __init__(self, target_log_prob, proposal, num_particles=200,
                 ess_threshold=0.5, resampling='systematic') -> None
    def run(self, num_steps=10, rng=None) -> AdaptiveSISResult

class PlackettLuceValidator:
    def __init__(self, num_permutation_samples=1000) -> None
    def validate(self, schedules: List[Schedule]) -> IIAValidationResult

class MixedLogitProposal(ProposalDistribution):
    def __init__(self, num_components=3) -> None
    def fit(self, schedules: List[Schedule], weights: np.ndarray) -> None
    def sample(self, n: int, rng) -> List[Schedule]
    def log_prob(self, schedule: Schedule) -> float

class NestedLogitProposal(ProposalDistribution):
    def __init__(self, nests: Dict[str, List[str]], scale=1.0) -> None
    def sample(self, n: int, rng) -> List[Schedule]
    def log_prob(self, schedule: Schedule) -> float

class StoppingCriteria:
    def __init__(self) -> None
    def should_stop(self, ess_history, estimates, target_ci_width=0.05) -> bool

class JointErrorAnalysis:
    def __init__(self) -> None
    def decompose(self, ai_error, is_estimates, is_weights) -> ErrorDecomposition
```

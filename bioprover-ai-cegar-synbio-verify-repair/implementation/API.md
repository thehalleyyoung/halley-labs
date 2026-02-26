# BioProver API Reference

API documentation for the BioProver verification and repair pipeline,
covering only implemented classes and functions.

---

## Table of Contents

1. [Top-Level API (`bioprover`)](#top-level-api)
2. [Models (`bioprover.models`)](#models)
3. [Temporal Logic (`bioprover.temporal`)](#temporal)
4. [CEGAR Engine (`bioprover.cegar`)](#cegar)
5. [Repair & Synthesis (`bioprover.repair`)](#repair)
6. [Soundness (`bioprover.soundness`)](#soundness)
7. [Solver (`bioprover.solver`)](#solver)
   - [Proof Certificates (`bioprover.solver.proof_certificate`)](#proof-certificates)
8. [SMT (`bioprover.smt`)](#smt)
9. [Encoding (`bioprover.encoding`)](#encoding)
10. [Stochastic (`bioprover.stochastic`)](#stochastic)
11. [AI / ML (`bioprover.ai`)](#ai)
    - [Training Pipeline (`bioprover.ai.training_pipeline`)](#training-pipeline)
12. [Compositional (`bioprover.compositional`)](#compositional)
    - [Circular AG (`bioprover.compositional.circular_ag`)](#circular-ag)
13. [Library (`bioprover.library`)](#library)
14. [Evaluation (`bioprover.evaluation`)](#evaluation)
    - [Ablation (`bioprover.evaluation.ablation`)](#ablation)
15. [Visualization (`bioprover.visualization`)](#visualization)
16. [CLI (`bioprover.cli`)](#cli)
17. [Certificate Verifier (`bioprover.certificate_verifier`)](#certificate-verifier)
18. [Error Propagation (`bioprover.soundness`)](#error-propagation)

---

<a id="top-level-api"></a>
## 1. Top-Level API — `bioprover`

The package root re-exports core classes and provides three convenience
functions covering the primary workflows.

### `verify(model, spec, *, mode="full", timeout=300.0, config=None)`

Verify a biological model against a Bio-STL specification.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `BioModel` | — | The biological model to verify. |
| `spec` | `str` | — | Bio-STL specification string, e.g. `"G[0,100](GFP > 0.5)"`. |
| `mode` | `str` | `"full"` | `"full"`, `"bounded"`, or `"compositional"`. |
| `timeout` | `float` | `300.0` | Maximum wall-clock seconds. |
| `config` | `CEGARConfig \| None` | `None` | Optional CEGAR configuration overrides. |

**Returns:** `VerificationResult`

Internally parses the spec with `BioSTLParser`, constructs a `CEGAREngine`,
and calls `engine.verify(model, formula)`.

```python
from bioprover import BioModel, verify

model = BioModel.from_sbml("toggle_switch.xml")
result = verify(model, spec="G[0,100](GFP > 0.5)", timeout=120.0)
print(result.status)        # VerificationStatus.VERIFIED
print(result.is_verified)   # True
print(result.coverage)      # 0.98
```

---

### `synthesize(model, spec, *, objective="feasibility", timeout=600.0)`

Synthesize parameters for a model satisfying a specification.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `BioModel` | — | Model with uncertain parameters. |
| `spec` | `str` | — | Bio-STL specification string. |
| `objective` | `str` | `"feasibility"` | `"feasibility"`, `"robustness"`, or `"minimal"`. |
| `timeout` | `float` | `600.0` | Maximum wall-clock seconds. |

**Returns:** `SynthesisResult`

Maps the `objective` string to a `SynthesisMode` enum, creates a
`ParameterSynthesizer`, and calls `synthesizer.synthesize(model, formula)`.

```python
from bioprover import BioModel, synthesize

model = BioModel.from_sbml("repressilator.xml")
result = synthesize(model, spec="F[0,50](GFP > 1.0)", objective="robustness")
print(result.feasible)       # True
print(result.parameters)     # {'alpha': 12.3, ...}
print(result.robustness)     # 0.42
```

---

### `repair(model, spec, *, budget=0.5, timeout=600.0)`

Repair model parameters so the specification is satisfied.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `BioModel` | — | Model whose parameters may be adjusted. |
| `spec` | `str` | — | Bio-STL specification string. |
| `budget` | `float` | `0.5` | Maximum fractional perturbation per parameter (0–1). |
| `timeout` | `float` | `600.0` | Maximum wall-clock seconds. |

**Returns:** `RepairResult`

Uses `ParameterSynthesizer` in `MINIMAL` mode with the given perturbation
budget. Returns a `RepairResult` with original/repaired parameters and
robustness.

```python
from bioprover import BioModel, repair

model = BioModel.from_sbml("band_pass.xml")
fix = repair(model, spec="G[10,50](0.3 < GFP < 0.8)", budget=0.2)
if fix.success:
    print(fix.repaired_parameters)
    print(fix.robustness)
```

---

### Re-exported Classes

| Class | Origin Module |
|-------|---------------|
| `BioModel` | `bioprover.models.bio_model` |
| `CEGARConfig` | `bioprover.cegar.cegar_engine` |
| `CEGAREngine` | `bioprover.cegar.cegar_engine` |
| `VerificationResult` | `bioprover.cegar.cegar_engine` |
| `VerificationStatus` | `bioprover.cegar.cegar_engine` |
| `ParameterSynthesizer` | `bioprover.repair.parameter_synthesis` |
| `SynthesisConfig` | `bioprover.repair.parameter_synthesis` |
| `SynthesisMode` | `bioprover.repair.parameter_synthesis` |
| `SynthesisResult` | `bioprover.repair.parameter_synthesis` |
| `RepairResult` | `bioprover.repair.repair_report` |
| `RepairReport` | `bioprover.repair.repair_report` |

---

<a id="models"></a>
## 2. Models — `bioprover.models`

### `BioModel`

Core biological model representation combining species, reactions, and
parameters into an ODE system.

```python
class BioModel:
    def __init__(self, name: str = "unnamed_model") -> None: ...
```

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `from_sbml` | `(filepath: str) -> BioModel` | `BioModel` | Class method. Import from SBML file. |
| `add_species` | `(species: Species) -> None` | `None` | Register a molecular species. |
| `add_reaction` | `(reaction: Reaction) -> None` | `None` | Register a biochemical reaction. |
| `add_parameter` | `(param: Parameter) -> None` | `None` | Register a kinetic parameter. |
| `add_compartment` | `(compartment: Compartment) -> None` | `None` | Register a compartment. |
| `stoichiometry_matrix` | property | `np.ndarray` | Stoichiometry matrix (species × reactions). |
| `rate_vector` | `(concentrations: Dict[str, float]) -> np.ndarray` | `ndarray` | Evaluate reaction rates at given concentrations. |
| `ode_rhs` | `(state: np.ndarray, t: float = 0.0) -> np.ndarray` | `ndarray` | ODE right-hand side evaluation. |
| `ode_rhs_callable` | `() -> Callable[[float, np.ndarray], np.ndarray]` | callable | Scipy-compatible ODE RHS. |
| `jacobian` | `(concentrations: Optional[Dict] = None) -> np.ndarray` | `ndarray` | Jacobian of the ODE RHS. |
| `jacobian_symbolic` | `() -> sympy.Matrix` | `Matrix` | Symbolic Jacobian via SymPy. |
| `steady_state` | `(initial_guess: Optional[np.ndarray] = None, method: str = "fsolve") -> Optional[np.ndarray]` | `ndarray \| None` | Compute steady state. |
| `steady_state_stability` | `(steady_state: np.ndarray) -> str` | `str` | `"stable"`, `"unstable"`, or `"saddle"`. |
| `parameter_sensitivity` | `(parameter_name: str, delta: float = 1e-6) -> np.ndarray` | `ndarray` | Local parameter sensitivity. |
| `simulate` | `(t_span: Tuple[float, float], num_points: int = 100, method: str = "RK45") -> Tuple[np.ndarray, np.ndarray]` | tuple | ODE simulation returning `(times, states)`. |
| `regulatory_network` | `() -> GeneRegulatoryNetwork` | `GRN` | Extract GRN topology. |
| `extract_submodel` | `(species_names: List[str]) -> BioModel` | `BioModel` | Extract a submodel for compositional analysis. |
| `conservation_laws` | `() -> List[Tuple[np.ndarray, float]]` | list | Detect conservation laws from stoichiometry. |
| `check_mass_balance` | `() -> Dict[str, bool]` | dict | Check mass balance per reaction. |
| `validate` | `() -> List[str]` | list | Return list of validation warnings. |
| `initial_state` | `() -> np.ndarray` | `ndarray` | Initial concentration vector. |

**Properties:** `species: List[Species]`, `reactions: List[Reaction]`, `parameters: ParameterSet`

```python
from bioprover import BioModel
from bioprover.models.species import Species
from bioprover.models.reactions import Reaction, HillRepression, LinearDegradation

model = BioModel("my_circuit")
model.add_species(Species("GFP", initial_concentration=1.0))
model.add_reaction(Reaction("deg", reactants={"GFP": 1}, products={},
    kinetic_law=LinearDegradation(rate=0.5)))
t, y = model.simulate((0, 100))
```

---

### `Species`

```python
class Species:
    def __init__(
        self,
        name: str,
        compartment: str = "default",
        initial_concentration: float = 0.0,
        units: str = "nM",
        species_type: SpeciesType = SpeciesType.PROTEIN,
        boundary_condition: BoundaryCondition = BoundaryCondition.FREE,
        metadata: Optional[SpeciesMetadata] = None,
        concentration_bounds: Optional[ConcentrationBounds] = None,
        copy_number: Optional[int] = None,
    ) -> None: ...
```

| Method | Returns | Description |
|--------|---------|-------------|
| `is_in_bounds(concentration)` | `bool` | Check if concentration is within bounds. |
| `validate()` | `List[str]` | Validation warnings. |
| `copy()` | `Species` | Deep copy. |
| `as_sympy_symbol()` | `sympy.Symbol` | SymPy symbol for symbolic computation. |

### `SpeciesType` (Enum)

`PROTEIN`, `MRNA`, `METABOLITE`, `COMPLEX`

---

### `Reaction`

```python
from bioprover.models.reactions import Reaction, StoichiometryEntry as SE

class Reaction:
    def __init__(
        self,
        name: str,
        reactants: List[StoichiometryEntry],  # NOT dicts
        products: List[StoichiometryEntry],    # NOT dicts
        kinetic_law: KineticLaw,
        reversible: bool = False,
        modifiers: List[str] = None,  # for Hill kinetics activator/repressor
    ) -> None: ...

# Example:
rxn = Reaction('production',
    reactants=[],
    products=[SE('gene_u', 1)],
    kinetic_law=HillRepression(Vmax=10.0, K=2.0, n=2),
    modifiers=['gene_v'])  # gene_v is the repressor
```

**Important API notes:**
- `reactants` and `products` take `List[StoichiometryEntry]`, NOT dictionaries.
  Use `StoichiometryEntry("species_name", stoichiometry)` or the alias `SE`.
- Species names MUST NOT be single-letter Bio-STL keywords: `G` (Globally),
  `F` (Finally), `U` (Until). Use multi-character names like `gene_u`, `lacI`.
- For Hill activation/repression with no consumed reactants, use
  `modifiers=["activator_species"]` to specify which species acts as the
  activator or repressor.

### Kinetic Law Classes

All kinetic laws inherit from `KineticLaw` (ABC) and implement
`rate_expression()`, `evaluate()`, `propensity()`, and `parameters`.

| Class | Constructor | Rate Law |
|-------|-------------|----------|
| `MassAction` | `(k_forward: float, k_reverse: float = 0.0)` | k_f · ∏[reactants] − k_r · ∏[products] |
| `HillActivation` | `(Vmax: float, K: float, n: float = 1.0)` | Vmax · x^n / (K^n + x^n) |
| `HillRepression` | `(Vmax: float, K: float, n: float = 1.0)` | Vmax · K^n / (K^n + x^n) |
| `MichaelisMenten` | `(Vmax: float, Km: float)` | Vmax · S / (Km + S) |
| `ConstitutiveProduction` | `(rate: float)` | k (constant) |
| `LinearDegradation` | `(rate: float)` | γ · x |

```python
from bioprover.models.reactions import HillRepression

hill = HillRepression(Vmax=10.0, K=2.0, n=2)
rate = hill.evaluate({"repressor": 3.0})  # Evaluates K^n / (K^n + x^n) * Vmax
```

---

### `Parameter`

```python
class Parameter:
    def __init__(
        self,
        name: str,
        value: float,
        lower: float = 0.0,
        upper: float = float("inf"),
        uncertainty_type: UncertaintyType = UncertaintyType.UNIFORM,
    ) -> None: ...
```

### `UncertaintyType` (Enum)

`UNIFORM`, `NORMAL`, `LOGNORMAL`

### `ParameterSet`

Container for named parameters.

| Method | Signature | Returns |
|--------|-----------|---------|
| `add_param` | `(param: Parameter) -> None` | `None` |
| `get_param` | `(name: str) -> Parameter` | `Parameter` |
| `to_dict` | `() -> Dict[str, float]` | `dict` |

---

### `SBMLImporter`

```python
class SBMLImporter:
    def __init__(self) -> None: ...
    def import_model(self, filepath: str) -> BioModel: ...
```

---

<a id="temporal"></a>
## 3. Temporal Logic — `bioprover.temporal`

### STL Formula AST

```python
class STLFormula(ABC): ...

# Atomic predicates
class Predicate(STLFormula):
    def __init__(self, var: str, op: ComparisonOp, threshold: float) -> None: ...

# Boolean connectives
class STLNot(STLFormula):
    def __init__(self, child: STLFormula) -> None: ...
class STLAnd(STLFormula):
    def __init__(self, left: STLFormula, right: STLFormula) -> None: ...
class STLOr(STLFormula):
    def __init__(self, left: STLFormula, right: STLFormula) -> None: ...
class STLImplies(STLFormula):
    def __init__(self, left: STLFormula, right: STLFormula) -> None: ...

# Temporal operators
class Always(STLFormula):
    def __init__(self, child: STLFormula, interval: Interval) -> None: ...
class Eventually(STLFormula):
    def __init__(self, child: STLFormula, interval: Interval) -> None: ...
class Until(STLFormula):
    def __init__(self, left: STLFormula, right: STLFormula, interval: Interval) -> None: ...
```

---

### `BioSTLParser`

Recursive-descent parser for Bio-STL formulas with domain-specific macros.

```python
class BioSTLParser:
    def __init__(self, custom_macros: Optional[Dict[str, MacroDef]] = None) -> None: ...
    def parse(self, text: str) -> STLFormula: ...
    def available_macros(self) -> List[str]: ...
```

**Built-in domain macros** (expanded during parsing):

| Macro | Expansion | Description |
|-------|-----------|-------------|
| `Oscillates(X, period, amp)` | `G[0,T](F[0,period](X > amp) & F[0,period](X < -amp))` | Sustained oscillation. |
| `Bistable(X, lo, hi)` | `G[0,T]((X < lo) \| (X > hi))` | Bimodal steady state. |
| `Adapts(X, stim, t_adapt)` | `F[0,t_stim](X > stim) & F[t_adapt,T](\|X - X_0\| < eps)` | Perfect adaptation. |
| `Switches(X, lo, hi, t)` | `(X < lo) U[0,t] G[t,T](X > hi)` | Irreversible switch. |
| `ReachesSteadyState(X, val, tol, t)` | `F[0,t](G[t,T](\|X - val\| < tol))` | Convergence to steady state. |
| `MonotoneResponse(in, out, delay)` | Monotone input-output | Monotone dose-response. |
| `Pulse(X, duration, amp)` | Pulse shape | Transient pulse behavior. |

```python
from bioprover.temporal.bio_stl_parser import BioSTLParser

parser = BioSTLParser()
formula = parser.parse("G[0,100](Bistable(U, 1.0, 5.0))")
formula = parser.parse("F[0,50](GFP > 1.0)")
formula = parser.parse("G[0,100](Oscillates(GFP, 20.0, 0.5))")
```

---

### `RobustnessComputer`

Quantitative STL robustness semantics.

```python
class RobustnessComputer:
    def __init__(self) -> None: ...
    def compute_robustness(
        self,
        trajectory: np.ndarray,
        times: np.ndarray,
        formula: STLFormula,
    ) -> float: ...
```

**Returns:** Positive if satisfied, negative if violated.

### `IntervalModelChecker`

Three-valued model checking over interval trajectories.

```python
class IntervalModelChecker:
    def __init__(self) -> None: ...
    def check(self, trajectory: IntervalTrajectory, formula: STLFormula) -> ThreeValuedResult: ...
```

`ThreeValuedResult`: `TRUE`, `FALSE`, `UNKNOWN`

### `StatisticalModelChecker`

```python
class StatisticalModelChecker:
    def __init__(self, alpha: float = 0.01, beta: float = 0.01, delta: float = 0.01) -> None: ...
    def sprt_check(self, verifier_fn: Callable[[], bool], threshold: float = 0.95) -> SPRTResult: ...
```

### `BMCEncoder`

Bounded model checking encoder.

```python
class BMCEncoder:
    def __init__(self, model: BioModel, formula: STLFormula) -> None: ...
    def encode_bmc(self, steps: int) -> ExprNode: ...
```

---

<a id="cegar"></a>
## 4. CEGAR Engine — `bioprover.cegar`

### `VerificationStatus` (Enum)

`VERIFIED`, `FALSIFIED`, `UNKNOWN`, `BOUNDED_GUARANTEE`

### `CEGARConfig`

```python
@dataclass
class CEGARConfig:
    max_iterations: int = 100
    timeout: float = 3600.0
    initial_grid_resolution: int = 4
    step_size: float = 0.01
    delta: float = 1e-3
    stagnation_window: int = 5
    coverage_target: float = 1.0
    strategy_name: str = "auto"
    enable_ai_heuristic: bool = False
    enable_bounded_guarantee: bool = True
    portfolio_configs: Optional[List[CEGARConfig]] = None
    max_workers: int = 1
```

| Method | Returns |
|--------|---------|
| `to_dict()` | `Dict[str, Any]` |

### `CEGAREngine`

Main CEGAR orchestrator for abstraction-refinement verification.

```python
class CEGAREngine:
    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        rhs: Dict[str, ExprNode],
        property_expr: ExprNode,
        property_name: str = "spec",
        initial_region: Optional[IntervalBox] = None,
        solver: Optional[Any] = None,
        model_checker: Optional[Callable] = None,
        refinement_strategy: Optional[RefinementStrategy] = None,
        ai_heuristic: Optional[Callable] = None,
        config: Optional[CEGARConfig] = None,
        hill_params: Optional[List[Dict[str, Any]]] = None,
        monotone_info: Optional[Dict[str, Dict[str, int]]] = None,
        steady_states: Optional[List[Dict[str, float]]] = None,
        jacobian: Optional[Callable] = None,
    ) -> None: ...

    def verify(self) -> VerificationResult: ...
```

The top-level `verify()` convenience function constructs a `CEGAREngine`
from a `BioModel` and `STLFormula` automatically.

### `VerificationResult`

```python
@dataclass
class VerificationResult:
    status: VerificationStatus
    property_name: str = ""
    counterexample: Optional[ConcreteCounterexample] = None
    abstract_counterexample: Optional[AbstractCounterexample] = None
    proof_certificate: Optional[Dict[str, Any]] = None
    statistics: Optional[CEGARStatistics] = None
    coverage: float = 0.0
    robustness: float = 0.0
    termination_reason: Optional[TerminationReason] = None
    message: str = ""
    soundness: Optional[SoundnessAnnotation] = None
```

| Property | Type | Description |
|----------|------|-------------|
| `is_verified` | `bool` | `True` if status is `VERIFIED`. |
| `is_falsified` | `bool` | `True` if status is `FALSIFIED`. |
| `to_dict()` | `Dict` | Serialize to dictionary. |

### `CEGARStatistics`

```python
@dataclass
class CEGARStatistics:
    iterations: int = 0
    total_time: float = 0.0
    abstraction_time: float = 0.0
    model_check_time: float = 0.0
    feasibility_time: float = 0.0
    refinement_time: float = 0.0
    peak_states: int = 0
    peak_predicates: int = 0
    spurious_count: int = 0
    genuine_count: int = 0
    final_coverage: float = 0.0
    strategies_used: Dict[str, int]
```

---

### Abstraction & Refinement

#### `IntervalAbstraction`

```python
class IntervalAbstraction:
    def __init__(self, model: BioModel, resolution: int) -> None: ...
    def initial_states(self) -> List[AbstractState]: ...
    def successors(self, state: AbstractState) -> List[AbstractTransition]: ...
    def refine(self, predicates: List[Predicate]) -> None: ...
```

#### Refinement Strategies

All implement `RefinementStrategy` (ABC):

```python
class RefinementStrategy(ABC):
    def refine(self, abstraction, counterexample, model) -> List[Predicate]: ...
```

| Strategy | Description |
|----------|-------------|
| `StructuralRefinement` | Hill-threshold predicates from network topology. |
| `MonotonicityRefinement` | Exploits GRN monotonicity for vertex-only checking. |
| `TimeScaleRefinement` | Temporal splitting via time-scale separation. |
| `InterpolationRefinement` | Craig interpolants from UNSAT proofs. |
| `SimulationGuidedRefinement` | Simulation-based predicate discovery. |

#### `ConvergenceMonitor`

```python
class ConvergenceMonitor:
    def __init__(self, patience: int = 10) -> None: ...
    def update(self, stats: IterationStats) -> None: ...
    def recommend_action(self) -> StrategySwitchAction: ...
```

#### `SpuriousnessChecker`

```python
class SpuriousnessChecker:
    def __init__(self, solver: AbstractSMTSolver = Z3Solver()) -> None: ...
    def check(self, counterexample: Counterexample) -> SpuriousnessResult: ...
```

---

<a id="repair"></a>
## 5. Repair & Synthesis — `bioprover.repair`

### `SynthesisMode` (Enum)

`FEASIBILITY`, `ROBUSTNESS`, `MINIMAL_PERTURBATION`, `MULTI_OBJECTIVE`

### `SynthesisConfig`

```python
@dataclass
class SynthesisConfig:
    mode: SynthesisMode = SynthesisMode.ROBUSTNESS
    max_outer_iterations: int = 10
    timeout: float = 600.0
    cmaes_config: Optional[CMAESConfig] = None
    cegis_config: Optional[CEGISConfig] = None
    n_cmaes_restarts: int = 3
    perturbation_weight: float = 0.1
    pareto_samples: int = 500
    check_realizability: bool = True
    warm_start_params: Optional[np.ndarray] = None
    verbose: bool = True
```

### `ParameterSynthesizer`

Orchestrator combining CEGIS + CMA-ES + robustness optimization.

```python
class ParameterSynthesizer:
    def __init__(
        self,
        param_set: ParameterSet,
        verifier: VerifierProtocol,
        robustness_fn: Callable[[np.ndarray], float],
        original_params: Optional[np.ndarray] = None,
        realizability_checker: Optional[RealizabilityChecker] = None,
        config: Optional[SynthesisConfig] = None,
    ) -> None: ...

    def synthesize(self) -> SynthesisResult: ...
    def warm_start(self, params: np.ndarray, robustness: float) -> None: ...
    def generate_report(self, result: SynthesisResult) -> RepairReport: ...
```

### `SynthesisResult`

```python
@dataclass
class SynthesisResult:
    success: bool = False
    mode: SynthesisMode = SynthesisMode.FEASIBILITY
    repair_result: Optional[RepairResult] = None
    pareto_frontier: Optional[ParetoFrontier] = None
    realizability_report: Optional[RealizabilityReport] = None
    cegis_result: Optional[CEGISResult] = None
    optimization_result: Optional[OptimizationResult] = None
    outer_iterations: int = 0
    total_time: float = 0.0
    history: List[Dict[str, Any]]
```

| Method | Returns |
|--------|---------|
| `summary()` | `str` — Human-readable summary. |

### `RepairResult`

```python
@dataclass
class RepairResult:
    original: np.ndarray
    repaired: np.ndarray
    parameter_names: List[str]
    robustness_before: float = float("-inf")
    robustness_after: float = float("-inf")
    verified: bool = False
    method: str = ""
    metadata: Dict[str, Any]
```

| Property | Type | Description |
|----------|------|-------------|
| `delta` | `np.ndarray` | `repaired - original` |
| `perturbation_l1` | `float` | L1 norm of parameter change. |
| `perturbation_l2` | `float` | L2 norm of parameter change. |
| `perturbation_linf` | `float` | L∞ norm of parameter change. |
| `relative_change` | `np.ndarray` | Relative change per parameter. |
| `robustness_improvement` | `float` | Robustness delta. |

| Method | Returns |
|--------|---------|
| `most_changed(k=5)` | `List[Tuple[str, float, float, float]]` — Top-k changed parameters. |
| `parameter_dict(repaired=True)` | `Dict[str, float]` — Named parameter values. |

### `RepairReport`

```python
class RepairReport:
    def __init__(
        self,
        primary: RepairResult,
        alternatives: Optional[List[RepairResult]] = None,
        sensitivity_rankings: Optional[List[Tuple[str, float]]] = None,
        confidence: float = 1.0,
        notes: Optional[str] = None,
    ) -> None: ...
```

| Property / Method | Returns |
|-------------------|---------|
| `success` | `bool` |
| `summary_text()` | `str` — Human-readable summary. |
| `biological_interpretation()` | `str` — Biological interpretation of repairs. |

### `CEGISLoop`

```python
class CEGISLoop:
    def __init__(self, config: CEGISConfig) -> None: ...
    def run(self, initial_guess, verifier, objective) -> CEGISResult: ...
```

### `RobustnessOptimizer`

```python
class RobustnessOptimizer:
    def __init__(self, config: CMAESConfig) -> None: ...
    def optimize(self, robustness_fn, bounds, x0) -> OptimizationResult: ...
```

### `RealizabilityChecker`

```python
class RealizabilityChecker:
    def __init__(self) -> None: ...
    def check(self, params: Dict[str, float]) -> RealizabilityReport: ...
```

### `DesignSpace`

```python
class DesignSpace:
    def __init__(self) -> None: ...
    def add_point(self, point: DesignPoint) -> None: ...
    def pareto_frontier(self) -> ParetoFrontier: ...
```

---

<a id="soundness"></a>
## 6. Soundness — `bioprover.soundness`

### `SoundnessLevel` (Enum)

```python
class SoundnessLevel(Enum):
    SOUND = auto()           # Full mathematical guarantee
    DELTA_SOUND = auto()     # Sound up to delta (dReal)
    BOUNDED = auto()         # Sound within bounded horizon
    APPROXIMATE = auto()     # Informative but not guaranteed
```

Supports comparison operators (`<`, `<=`, `>`, `>=`) and `SoundnessLevel.meet(a, b)` returns the weakest of two levels.

### `SoundnessAnnotation`

```python
@dataclass
class SoundnessAnnotation:
    level: SoundnessLevel
    assumptions: List[str] = field(default_factory=list)
    delta: Optional[float] = None
    time_bound: Optional[float] = None
    approximation_error: Optional[float] = None
```

| Method | Returns | Description |
|--------|---------|-------------|
| `weaken_to(level, reason)` | `SoundnessAnnotation` | Weaken to a new level with a reason. |
| `with_delta(delta)` | `SoundnessAnnotation` | Set delta-satisfiability. |
| `with_time_bound(t)` | `SoundnessAnnotation` | Set bounded time horizon. |

```python
from bioprover.soundness import SoundnessLevel, SoundnessAnnotation

ann = SoundnessAnnotation(level=SoundnessLevel.SOUND)
ann = ann.with_delta(1e-3)   # Now DELTA_SOUND
print(ann.level)             # SoundnessLevel.DELTA_SOUND
print(ann.assumptions)       # ['dReal delta-satisfiability with delta=0.001']
```

### `ErrorBudget`

Tracks individual error sources and computes the combined error via
root-sum-of-squares composition.

```python
@dataclass
class ErrorBudget:
    delta: float = 0.0              # dReal delta-satisfiability tolerance
    epsilon: float = 0.0            # ODE integration error bound
    truncation: float = 0.0         # Taylor truncation error
    discretization: float = 0.0     # time-discretization error
```

| Method | Returns | Description |
|--------|---------|-------------|
| `propagate_errors()` | `float` | Combined error (RSS of all sources). |

```python
from bioprover.soundness import ErrorBudget

budget = ErrorBudget(delta=1e-3, epsilon=1e-4, truncation=1e-5)
print(budget.propagate_errors())  # ~0.001005
```

`SoundnessAnnotation.with_error_budget(budget)` attaches an `ErrorBudget` to
an annotation and weakens the soundness level appropriately.

---

<a id="solver"></a>
## 7. Solver — `bioprover.solver`

### `Interval`

Rigorous interval arithmetic with outward-rounded operations.

```python
class Interval:
    def __init__(self, lo: float, hi: Optional[float] = None) -> None: ...
```

| Method | Returns | Description |
|--------|---------|-------------|
| `mid()` | `float` | Centre of the interval. |
| `width()` | `float` | `hi - lo`. |
| `radius()` | `float` | Half-width. |
| `is_empty()` | `bool` | Empty interval check. |
| `is_thin(tol=0.0)` | `bool` | Near-point interval. |
| `contains(x)` | `bool` | Point or interval containment. |
| `contains_zero()` | `bool` | Does interval contain 0? |
| `overlaps(other)` | `bool` | Do intervals overlap? |
| `bisect()` | `Tuple[Interval, Interval]` | Split at midpoint. |
| `subdivide(n)` | `List[Interval]` | Split into n sub-intervals. |
| `to_validated()` | `ValidatedInterval` | Convert to mpmath-backed interval. |

**Overloaded operators:** `+`, `-`, `*`, `/`, `**`, `abs`, `-` (unary)

**Elementary functions:** `sqrt()`, `exp()`, `log()`, `sin()`, `cos()`

### `ValidatedInterval`

mpmath-backed rigorous interval arithmetic with quad-precision guarantees.
Same interface as `Interval` but with higher precision.

```python
use_validated_arithmetic(enabled: bool = True)  # Module-level switch
```

### `IntervalVector`

```python
class IntervalVector:
    def __init__(self, intervals: List[Interval]) -> None: ...
    def dim(self) -> int: ...
    def midpoint(self) -> np.ndarray: ...
    def hull(self, other) -> IntervalVector: ...
    def volume(self) -> float: ...
```

### `ValidatedODEIntegrator`

```python
class ValidatedODEIntegrator:
    def __init__(self, order: int = 4, min_step: float = 1e-10, max_step: float = 0.1) -> None: ...
    def step(self, x, p, t, dt) -> StepResult: ...
    def integrate(self, x0, p, t_span) -> Flowpipe: ...
```

### `TaylorModel`

Taylor model arithmetic for tighter flowpipe enclosures.

```python
class TaylorModel:
    def __init__(self, coefficients: np.ndarray, remainder: Interval, order: int) -> None: ...
    def evaluate(self, point: np.ndarray) -> Interval: ...
    def bound(self) -> Interval: ...
```

### `Flowpipe`

Sequence of reachable-set enclosures over time.

| Method | Returns | Description |
|--------|---------|-------------|
| `enclose(point)` | `bool` | Does the flowpipe enclose the point? |
| `time_steps()` | `List[float]` | Time stamps of each segment. |
| `get_enclosure(t)` | `IntervalVector` | Enclosure at time t. |

<a id="proof-certificates"></a>
### Proof Certificates — `bioprover.solver.proof_certificate`

#### `ProofCertificate`

Base certificate with versioning, timestamps, and metadata.

```python
@dataclass
class ProofCertificate:
    version: str
    timestamp: str
    metadata: Dict[str, Any]
```

#### `FlowpipeCertificate`

Records the ODE system, initial conditions, integration parameters, and
flowpipe segments for independent re-verification.

```python
class FlowpipeCertificate(ProofCertificate):
    def __init__(self, ode_system, initial_conditions, segments, params) -> None: ...
    @classmethod
    def from_integration(cls, integration_result) -> "FlowpipeCertificate": ...
    def validate(self) -> bool: ...
```

#### `InvariantCertificate`

Proves that invariants (lower/upper bounds, linear constraints) hold
across all flowpipe segments.

| Method | Returns | Description |
|--------|---------|-------------|
| `check_lower_bound(species, bound)` | `bool` | Verify lower bound holds. |
| `check_upper_bound(species, bound)` | `bool` | Verify upper bound holds. |
| `check_linear_invariant(coefficients, bound)` | `bool` | Verify linear invariant. |

#### `SoundnessCertificate`

Wraps a proof certificate with a `SoundnessAnnotation` and `ErrorBudget`.

```python
class SoundnessCertificate:
    def __init__(self, certificate: ProofCertificate,
                 soundness: SoundnessAnnotation,
                 error_budget: ErrorBudget) -> None: ...
```

#### `validate_certificate(certificate) -> bool`

Standalone validator that re-checks a certificate without access to the
original verifier. Returns `True` if the certificate is self-consistent.

```python
from bioprover.solver.proof_certificate import validate_certificate

valid = validate_certificate(cert)
```

#### `compose_flowpipe_certificates(certs) -> FlowpipeCertificate`

Compose a temporal sequence of flowpipe certificates into a single certificate.

---

<a id="smt"></a>
## 8. SMT — `bioprover.smt`

### `SMTResult` (Enum)

`SAT`, `UNSAT`, `UNKNOWN`

### `Z3Solver`

```python
class Z3Solver(AbstractSMTSolver):
    def __init__(self, tactic: TacticPreset = TacticPreset.AUTO) -> None: ...
    def solve(self, assertions: List[ExprNode], timeout: float = 60.0) -> Tuple[SMTResult, Optional[Model]]: ...
```

### `DRealSolver`

```python
class DRealSolver(AbstractSMTSolver):
    def __init__(self, delta: float = 1e-3, binary_path: str = "dreal") -> None: ...
```

### `PortfolioSolver`

Run multiple solver strategies in parallel; return the first result.

```python
class PortfolioSolver(AbstractSMTSolver):
    def __init__(self, solvers: List[AbstractSMTSolver] | None = None) -> None: ...
```

### `CraigInterpolant`

Extract Craig interpolants from UNSAT proofs for CEGAR refinement.

```python
class CraigInterpolant:
    def extract(self, proof, partition_A, partition_B) -> Optional[ExprNode]: ...
```

---

<a id="encoding"></a>
## 9. Encoding — `bioprover.encoding`

### Expression IR

All SMT expressions are built from the `ExprNode` abstract base class.

| Category | Classes |
|----------|---------|
| **Leaf** | `Const(value)`, `Var(name)` |
| **Arithmetic** | `Add`, `Mul`, `Div`, `Pow`, `Neg` |
| **Functions** | `Exp`, `Log`, `Sin`, `Cos`, `Sqrt`, `Abs`, `Min`, `Max` |
| **Biology** | `HillAct(x, k, n)`, `HillRep(x, k, n)` |
| **Comparison** | `Lt`, `Le`, `Eq`, `Ge`, `Gt` |
| **Logic** | `And`, `Or`, `Not`, `Implies`, `ForAll`, `Exists` |

### `expr_to_smtlib(expr: ExprNode) -> str`

Serialize an expression tree to SMT-LIB 2.6 format.

### `ODEEncoding`

```python
class ODEEncoding:
    def __init__(self, system: ODESystem, method: DiscretizationMethod = DiscretizationMethod.FORWARD_EULER, steps: int = 100) -> None: ...
    def encode(self) -> List[ExprNode]: ...
```

### `DiscretizationMethod` (Enum)

`FORWARD_EULER`, `BACKWARD_EULER`, `MIDPOINT`, `RK4`

### `HillEncodingConfig`

```python
class HillEncodingConfig:
    def __init__(self, integer_hill_exact: bool = True, piecewise_linear_fallback: bool = True, approximation_segments: int = 8) -> None: ...
```

---

<a id="stochastic"></a>
## 10. Stochastic — `bioprover.stochastic`

### `DirectMethod`

Exact stochastic simulation (Gillespie SSA).

```python
class DirectMethod:
    def __init__(self, model: BioModel, seed: int = 42) -> None: ...
    def run_trajectory(self, x0: np.ndarray, t_end: float) -> StochasticTrajectory: ...
```

### `NextReactionMethod`

Gibson-Bruck optimised SSA.

### `EnsembleSimulator`

```python
class EnsembleSimulator:
    def __init__(self, method: str = "direct", n_workers: int = 4) -> None: ...
    def run_ensemble(self, model, n_trajectories, t_end) -> List[StochasticTrajectory]: ...
```

### `FSPSolver`

Finite State Projection for exact probability distributions.

```python
class FSPSolver:
    def __init__(self, model: BioModel, truncation: int = 100) -> None: ...
    def solve(self, t_end: float) -> MarginalDistribution: ...
```

### Tau-Leaping Variants

`ExplicitTauLeaping`, `ImplicitTauLeaping`, `MidpointTauLeaping`

### `MomentClosureSolver`

```python
class MomentClosureSolver:
    def __init__(self, model: BioModel, closure: str = "normal", max_order: int = 2) -> None: ...
    def solve(self, t_end: float) -> MomentTrajectory: ...
```

---

<a id="ai"></a>
## 11. AI / ML — `bioprover.ai`

### `CircuitEncoder`

Converts a BioModel to a graph representation for GNN input.

```python
class CircuitEncoder:
    def __init__(self) -> None: ...
    def encode(self, model: BioModel) -> CircuitGraph: ...
```

### `GraphSAGEEncoder`

```python
class GraphSAGEEncoder:
    def __init__(self, embedding_dim: int = 64, num_layers: int = 3) -> None: ...
    def encode(self, graph: CircuitGraph) -> np.ndarray: ...
```

### `PredicatePredictor`

```python
class PredicatePredictor:
    def __init__(self, model_path: str | None = None) -> None: ...
    def predict(self, model, abstraction, counterexample) -> List[CandidatePredicate]: ...
```

### `RobustnessSurrogate`

Gaussian process surrogate for fast robustness estimation.

```python
class RobustnessSurrogate:
    def __init__(self) -> None: ...
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...
    def predict(self, params: np.ndarray) -> Tuple[float, float]: ...  # (mean, std)
```

### `PredictionQualityMonitor`

Sliding-window monitor that tracks prediction precision and F1 score.
Automatically disables ML heuristics when quality degrades below threshold.

```python
class PredictionQualityMonitor:
    def record(self, predicted, actual) -> None: ...
    def get_metrics(self) -> Dict[str, float]: ...
    def get_family_metrics(self) -> Dict[str, Dict[str, float]]: ...
    def get_report(self) -> str: ...
```

<a id="training-pipeline"></a>
### Training Pipeline — `bioprover.ai.training_pipeline`

#### `TrainingDataGenerator`

Collects CEGAR verification traces and generates labelled training examples
for the predicate predictor.

```python
class TrainingDataGenerator:
    def __init__(self, augmenter: DataAugmenter | None = None) -> None: ...
    def add_trace(self, trace: VerificationTrace) -> None: ...
    def generate(self) -> Dataset: ...
    def generate_from_engine(self, engine, models, formulas) -> Dataset: ...
    @property
    def n_traces(self) -> int: ...
    @property
    def predicate_stats(self) -> Dict: ...
```

#### `CrossValidator`

K-fold cross-validation for the predicate predictor MLP, returning a
`TrainingReport` with per-fold precision, recall, F1, and loss.

```python
class CrossValidator:
    def __init__(self, hidden_dims: List[int], config: TrainingConfig,
                 k_folds: int = 5) -> None: ...
    def run(self, dataset: Dataset) -> TrainingReport: ...
```

#### `TrainingReport`

```python
@dataclass
class TrainingReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    loss: float
    mrr_score: float
    ndcg_score: float
    n_train: int
    n_val: int
    n_epochs_run: int
    fold_reports: List[Dict]
    training_history: List[Dict]
    def summary(self) -> str: ...
```

#### Supporting Classes

| Class | Purpose |
|-------|---------|
| `CircuitData` | Graph embedding + structural/kinetic features for a circuit. |
| `VerificationTrace` | Complete CEGAR trace (circuit, steps, success, time). |
| `RefinementExampleRecord` | Context features + predicate index + effectiveness. |
| `DataAugmenter` | Species permutation, parameter perturbation, time rescaling. |
| `Dataset` | Container with `split()` and `k_fold()` methods. |
| `Trainer` | SGD trainer with early stopping and learning-rate schedules. |
| `MLEvaluator` | Cross-validation and hold-out evaluation wrapper. |

---

<a id="compositional"></a>
## 12. Compositional — `bioprover.compositional`

### `CircularAG`

Circular assume-guarantee reasoning for compositional verification.

```python
class CircularAG:
    def __init__(self) -> None: ...
    def decompose(self, model: BioModel) -> List[Contract]: ...
    def verify_compositionally(self, model, formula) -> VerificationResult: ...
```

### `ContractSynthesis`

```python
class ContractSynthesis:
    def __init__(self) -> None: ...
    def synthesize(self, components: List[BioModel], property: STLFormula) -> List[Contract]: ...
```

<a id="circular-ag"></a>
### Circular AG — `bioprover.compositional.circular_ag`

#### `CircularAGChecker`

Co-inductive fixed-point algorithm for circular module dependencies.
Iterates assumptions and guarantees until convergence or divergence.

```python
class CircularAGChecker:
    def __init__(self, modules, contracts, dependency_graph) -> None: ...
    def check(self) -> CircularAGResult: ...
    def check_with_symmetry(self) -> CircularAGResult: ...
    def check_quantitative(self) -> CircularAGResult: ...
```

#### `CircularAGResult`

```python
@dataclass
class CircularAGResult:
    status: ConvergenceStatus      # CONVERGED, DIVERGED, FAILED, MAX_ITERATIONS
    proof_valid: bool
    history: List[FixedPointState]
    soundness_argument: str
```

#### `TopologyAnalyzer`

Analyzes the dependency graph to determine whether circular AG reasoning
is needed.

```python
class TopologyAnalyzer:
    def __init__(self, dependency_graph) -> None: ...
    def is_acyclic(self) -> bool: ...
    def find_cycles(self) -> List[List]: ...
    def topological_order(self) -> List: ...
    def needs_circular_ag(self) -> bool: ...
    def recommend_strategy(self) -> str: ...    # "sequential" or "circular"
    def strongly_connected_components(self) -> List[List]: ...
```

#### `WellFormednessChecker`

Validates that a set of assume-guarantee contracts is well-formed:
compatible, monotone, and contracting (spectral radius < 1).

```python
class WellFormednessChecker:
    def __init__(self, contracts, dependency_graph, modules=None) -> None: ...
    def check_all(self) -> AGWellFormedness: ...
    def check_contract_compatibility(self) -> bool: ...
    def check_progress_condition(self) -> bool: ...
    def recommend_strategy(self) -> str: ...
```

#### Supporting Classes

| Class | Purpose |
|-------|---------|
| `ConvergenceStatus` | Enum: `CONVERGED`, `DIVERGED`, `FAILED`, `MAX_ITERATIONS`, `INCOMPLETE`. |
| `FixedPointState` | Iteration snapshot with assumptions, guarantees, verification results. |
| `AGWellFormedness` | Result of well-formedness checks with coupling metrics. |
| `AGFailureDiagnostics` | Coupling matrix, spectral radius, failure suggestions. |

---

<a id="library"></a>
## 13. Library — `bioprover.library`

### `PartsDatabase`

Registry of characterized biological parts.

```python
class PartsDatabase:
    def __init__(self, data_path: str | None = None) -> None: ...
    def get_part(self, part_id: str) -> Part: ...
    def list_parts(self, part_type: str | None = None) -> List[Part]: ...
```

### `MotifLibrary`

Common regulatory motifs with models and specifications.

```python
class MotifLibrary:
    def __init__(self) -> None: ...
    def get_motif(self, name: str) -> Motif: ...
    def list_motifs(self) -> List[str]: ...
```

### `ModelTemplates`

```python
class ModelTemplates:
    def toggle_switch(self, params: Dict | None = None) -> BioModel: ...
    def repressilator(self, params: Dict | None = None) -> BioModel: ...
    def feed_forward_loop(self, loop_type: str = "C1-FFL") -> BioModel: ...
```

### `ParameterDatabase`

Literature-sourced kinetic parameters with provenance.

```python
class ParameterDatabase:
    def __init__(self) -> None: ...
    def lookup(self, name: str) -> List[ParameterRecord]: ...
```

---

<a id="evaluation"></a>
## 14. Evaluation — `bioprover.evaluation`

### `BenchmarkSuite`

```python
class BenchmarkSuite:
    def __init__(self, suite_name: str = "full") -> None: ...
    def run(self, timeout_per_case: float = 300.0) -> BenchmarkResults: ...
    def list_cases(self) -> List[str]: ...
```

### `BenchmarkCircuit`

```python
@dataclass
class BenchmarkCircuit:
    model: BioModel
    specification: STLFormula
    expected_result: VerificationStatus
    difficulty: BenchmarkDifficulty  # EASY, MEDIUM, HARD, FRONTIER
    name: str
    description: str
    category: str
    metadata: Dict[str, Any]
    tags: List[str]
```

### `BaselineComparison`

```python
class BaselineComparison:
    def __init__(self, baselines: List[str]) -> None: ...
    def compare(self, suite: BenchmarkSuite) -> ComparisonTable: ...
```

### `MutationTester`

Fault injection for soundness testing.

```python
class MutationTester:
    def __init__(self) -> None: ...
    def mutate(self, model: BioModel, n_mutants: int = 100) -> List[BioModel]: ...
    def run(self, model: BioModel, formula: STLFormula) -> MutationReport: ...
```

<a id="ablation"></a>
### Ablation — `bioprover.evaluation.ablation`

#### `AblationConfig` (Enum)

Predefined ablation configurations:

```python
class AblationConfig(Enum):
    NO_AI = auto()               # Disable all ML heuristics
    AI_STRUCTURAL = auto()       # AI + structural refinement only
    AI_MONOTONICITY = auto()     # AI + monotonicity refinement only
    AI_ALL_STRATEGIES = auto()   # AI + all refinement strategies
```

#### `AblationRunner`

Runs ablation experiments across configurations and circuits.

```python
class AblationRunner:
    def run(self, circuits, configurations) -> AblationSummary: ...
```

#### `AblationRunResult`

```python
@dataclass
class AblationRunResult:
    circuit: str
    configuration: str
    iterations: int
    time_s: float
    predicates: int
    converged: bool
    soundness_level: str
```

#### `AblationSummary`

```python
@dataclass
class AblationSummary:
    results: List[AblationRunResult]
```

---

<a id="visualization"></a>
## 15. Visualization — `bioprover.visualization`

### `ResultExporter`

```python
class ResultExporter:
    def __init__(self, export_format: ExportFormat = ExportFormat.TEXT) -> None: ...
    def export(self, result, output_path: str) -> None: ...
    def export_latex(self, results: List[VerificationResult]) -> str: ...
    def export_html(self, results, model_name: str = "", path: str = "report.html") -> None: ...
    def summary_report(self, results: List[VerificationResult]) -> str: ...
```

### `ExportFormat` (Enum)

`JSON`, `CSV`, `LATEX`, `HTML`, `TEXT`

### `CounterexampleVisualizer`

```python
class CounterexampleVisualizer:
    def __init__(self, width: int = 80, height: int = 25) -> None: ...
    def plot_trace_ascii(self, cex: Counterexample) -> str: ...
    def generate_text_report(self, cex, result) -> str: ...
```

### `ProgressReporter`

Real-time CEGAR iteration display.

```python
class ProgressReporter:
    def __init__(self, verbosity: int = 1) -> None: ...
    def update(self, iteration: int, stats: IterationStats) -> None: ...
    def finish(self, result: VerificationResult) -> None: ...
```

---

<a id="cli"></a>
## 16. CLI — `bioprover.cli`

Entry point: `bioprover.cli:main`

### `bioprover verify`

```
bioprover verify -m MODEL -s SPEC [--mode MODE] [--timeout T] [--format FMT] [-o PATH]
```

| Flag | Description |
|------|-------------|
| `-m, --model` | Path to SBML model file. |
| `-s, --spec` | Bio-STL specification (string or `.stl` file). |
| `--mode` | `full`, `bounded`, or `compositional`. |
| `--timeout` | Wall-clock timeout in seconds (default: 300). |
| `--format` | Output format: `text`, `json`, `csv`, `latex`, `html`. |
| `-o, --output` | Output file path. |

### `bioprover synthesize`

```
bioprover synthesize -m MODEL -s SPEC [--objective OBJ] [--timeout T] [-o PATH]
```

| Flag | Description |
|------|-------------|
| `--objective` | `feasibility`, `robustness`, or `minimal`. |

### `bioprover repair`

```
bioprover repair -m MODEL -s SPEC [--budget B] [--timeout T] [-o PATH]
```

| Flag | Description |
|------|-------------|
| `--budget` | Maximum fractional perturbation per parameter (0–1, default: 0.5). |

### `bioprover benchmark`

```
bioprover benchmark [--suite SUITE] [--baselines B1 B2 ...] [--format FMT] [-o PATH]
```

| Flag | Description |
|------|-------------|
| `--suite` | `toggle_switch`, `repressilator`, or `full`. |
| `--baselines` | Tools to compare: `bioprover`, `dreach`, `flow_star`, `spaceex`. |

### `bioprover info`

```
bioprover info -m MODEL [-s SPEC]
```

Prints model summary: species, reactions, parameters, compartments, and
optional specification parse.

### Global Flags

| Flag | Description |
|------|-------------|
| `-v` | Verbosity (`-v`=INFO, `-vv`=DEBUG, `-vvv`=TRACE). |
| `-q, --quiet` | Suppress progress output. |
| `--no-color` | Disable colour output. |
| `-c, --config` | Configuration file (JSON or YAML). |

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Verified / success |
| 1 | Falsified (counterexample found) |
| 2 | Unknown (timeout or inconclusive) |
| 3 | Error (invalid input, crash) |

### Module-Level Functions

```python
def build_parser() -> argparse.ArgumentParser: ...
def load_config(path: str) -> Dict[str, Any]: ...
def load_model(path: str) -> BioModel: ...
def load_spec(spec_arg: str) -> STLFormula: ...
def main(argv: Optional[List[str]] = None) -> int: ...
```

---

## 17. Certificate Verifier — `bioprover.certificate_verifier`

Standalone certificate verification module (~800 LoC). Does NOT import
from `bioprover.solver`, Z3, dReal, or SymPy.

### `CertificateVerifier`

```python
from bioprover.certificate_verifier import CertificateVerifier

verifier = CertificateVerifier()
report = verifier.verify(certificate_dict)
print(report.passed, report.failed, report.warnings)
print(report.accepted)  # True if failed == 0
```

### Verifier classes

| Class | Purpose |
|-------|---------|
| `FlowpipeReplayVerifier` | Replays integration with independent Euler; checks enclosure containment |
| `InvariantReplayVerifier` | Checks invariant satisfaction over interval boxes |
| `ErrorBudgetVerifier` | Validates soundness level vs. error budget consistency |
| `CompositionalVerifier` | Checks module interface compatibility |

### `VInterval` / `VBox`

Independent interval arithmetic (verifier-only, does not share code with solver):

```python
from bioprover.certificate_verifier.verifier import VInterval, VBox

iv = VInterval(1.0, 2.0)
print(iv.contains(1.5))   # True
print(iv.width())         # 1.0

box = VBox([VInterval(0, 1), VInterval(0, 1)])
print(box.contains_point([0.5, 0.5]))  # True
```

### CLI

```bash
python -m bioprover.certificate_verifier.cli certificate.json
```

---

## 18. Error Propagation — `bioprover.soundness`

### `ErrorSource`

```python
from bioprover.soundness import ErrorSource

src = ErrorSource(
    name="discretization",
    magnitude=0.01,
    origin="ODE integration",
    lipschitz_factor=2.0,  # dynamics amplification
)
print(src.effective_magnitude)  # 0.02
```

### `ErrorBudget` (extended)

```python
from bioprover.soundness import ErrorBudget

budget = ErrorBudget(
    delta=0.001,            # SMT perturbation
    epsilon=0.0,            # CEGIS tolerance
    truncation=0.05,        # moment closure
    discretization=0.01,    # ODE integration
)
print(budget.combined)           # RSS bound
print(budget.combined_additive)  # additive bound
print(budget.is_sound)           # True if all finite
```

### Key functions

| Function | Description |
|----------|-------------|
| `propagate_errors_with_lipschitz(sources)` | Lipschitz-amplified error composition |
| `compute_moment_closure_bound(n, k, N, L_f)` | Theorem 4 truncation bound |
| `compute_discretization_bound(h, p, L, T, C)` | Grönwall discretization bound |

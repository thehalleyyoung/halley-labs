# CABER API Reference

> **Implemented**: Rust core types/traits/algorithms, 38 property-based tests,
> mock LLM experiments (Phase 0), real LLM validation (gpt-4.1-nano),
> multi-configuration experiments (3 configs × 15 prompts × 5 trials),
> PDFA baseline comparison (with hyperparameter tuning), random sampling baseline,
> CoalCEGAR convergence analysis, classifier robustness (Monte Carlo),
> Bayesian analysis (Dirichlet posteriors, credible intervals),
> ablation studies (CoalCEGAR, graded vs binary, distance metrics, classifiers),
> statistical baselines (KL, MMD, chi-squared, frequency),
> calibration analysis, posterior predictive checks.
>
> **Not implemented**: Lean 4 formalization, frontier LLM validation (GPT-4, Claude, Llama),
> multi-agent composition, live monitoring dashboard, full AALpy+PRISM baseline,
> formally checkable proof certificates.

## Running Experiments

```bash
cd implementation

# Phase C: Bayesian analysis + ablation + PDFA tuning (no API key needed)
python3 phase_c_experiments.py

# Multi-config LLM: 3 configs × 15 prompts + convergence + random baseline
source ~/.bashrc && python3 expanded_experiments.py

# Phase B: real LLM + PDFA baseline + scaling (requires OPENAI_API_KEY)
source ~/.bashrc && python3 phase_b_experiments.py

# Phase 0: mock LLM validation
python3 phase0_experiments.py

# Classifier robustness
python3 classifier_robustness_analysis.py

# Property-based tests
cargo test -p caber-integration --test property_tests
```

## Rust API (`caber-core`)

### coalgebra — Core Types

```rust
pub struct StateId(Uuid);
pub struct Symbol(String);
pub struct SubDistribution<K> { weights: BTreeMap<K, f64> }

pub struct BehavioralFunctor {
    pub output_bound: usize,      // k: max output length
    pub input_bound: usize,       // n: max prompt length
    pub tolerance: f64,           // ε: distributional tolerance
}

pub struct FunctorBandwidth {
    pub beta: f64,                // β: bandwidth value
    pub covering_size: usize,     // covering centers
    pub lower_bound: f64,
    pub upper_bound: f64,
}
```

### learning — PCL* Algorithm

```rust
pub struct ObservationTable {
    pub access_strings: Vec<Word>,    // S
    pub suffixes: Vec<Word>,          // E
    pub entries: HashMap<(Word, Word), SubDistribution<Symbol>>,
}

pub struct HypothesisAutomaton {
    pub states: Vec<StateId>,
    pub initial: StateId,
    pub transitions: HashMap<(StateId, Symbol), SubDistribution<StateId>>,
    pub labeling: HashMap<StateId, HashSet<AtomicProp>>,
}
```

### abstraction — CoalCEGAR

```rust
pub struct AbstractionLevel {
    pub output_bound: usize,
    pub input_bound: usize,
    pub tolerance: f64,
}

pub struct AbstractionLattice {
    pub levels: Vec<AbstractionLevel>,
}

impl AbstractionLattice {
    pub fn refine(&self, current: &AbstractionLevel, counterexample: &Trace)
        -> Option<AbstractionLevel>;
}
```

### temporal — QCTL_F Specifications

```rust
pub enum Formula {
    True, False,
    Atom(AtomicProp),
    Not(Box<Formula>),
    And(Box<Formula>, Box<Formula>),
    Or(Box<Formula>, Box<Formula>),
    Prob(Comparison, f64, Box<PathFormula>),
    ForAll(Box<PathFormula>),
    Exists(Box<PathFormula>),
}

pub enum PathFormula {
    Next(Box<Formula>),
    Until(Box<Formula>, Box<Formula>),
    Globally(Box<Formula>),
    Finally(Box<Formula>),
}

pub enum SpecTemplate {
    RefusalPersistence { threshold: f64 },
    ParaphraseInvariance { tolerance: f64 },
    VersionStability { max_distance: f64 },
    SycophancyResistance { max_reversal_rate: f64 },
    InstructionHierarchy { dominance_threshold: f64 },
    JailbreakResistance { min_refusal_prob: f64, priming_turns: usize },
}
```

### model_checker — Fixed-Point Engine

```rust
pub type SatisfactionDegree = f64;

pub struct ModelCheckResult {
    pub satisfaction: HashMap<StateId, SatisfactionDegree>,
    pub witnesses: Vec<Trace>,
}

pub trait ModelChecker {
    fn check(&self, automaton: &HypothesisAutomaton, formula: &Formula)
        -> ModelCheckResult;
}
```

### bisimulation — Distance Computation

```rust
pub trait BisimulationDistance {
    fn distance(&self, a: &HypothesisAutomaton, b: &HypothesisAutomaton) -> f64;
    fn exact_bisimilar(&self, a: &HypothesisAutomaton, b: &HypothesisAutomaton) -> bool;
}

pub struct KantorovichLifting {
    pub ground_metric: Box<dyn Fn(&Symbol, &Symbol) -> f64>,
    pub convergence_tolerance: f64,
}
```

### certificate — Audit Reports

```rust
pub struct AuditCertificate {
    pub automaton: HypothesisAutomaton,
    pub verification_results: Vec<(Formula, ModelCheckResult)>,
    pub bisimulation_distances: Vec<(String, f64)>,
    pub pac_parameters: PacParameters,
    pub hash_chain: Vec<[u8; 32]>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

pub struct PacParameters {
    pub epsilon: f64,
    pub delta: f64,
    pub classifier_error_rate: f64,
    pub total_error_bound: f64,
}
```

## Python API

### Mock LLM & Learning

```python
class StochasticMockLLM:
    """Markov-chain mock LLM with known ground-truth automaton."""
    def query(self, prompt: str) -> str: ...
    def reset(self) -> None: ...

class PCLStarLearner:
    """PCL* algorithm implementation."""
    def learn(self, model: StochasticMockLLM,
              tolerance: float = 0.15,
              samples_per_query: int = 80,
              max_states: int = 40) -> LearnedAutomaton: ...
```

### Classifier Robustness

```python
def run_robustness_analysis(
    error_rates: list[float] = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20],
    trials_per_rate: int = 2000,
    n_states: int = 5
) -> dict:
    """Monte Carlo simulation of classifier error propagation."""
```

### Bayesian Statistical Analysis

```python
def dirichlet_posterior(counts: dict, prior_alpha: float = 1.0) -> dict:
    """Compute Dirichlet posterior parameters from observation counts.
    Returns posterior means, variances, and 95% credible intervals."""

def bayesian_divergence_analysis(n_prompts: int, n_divergent: int) -> dict:
    """Beta-Binomial posterior analysis of behavioral divergence rate.
    Returns posterior mean, HPD interval, and tail probabilities."""

def compute_classification_metrics(true_labels, predicted_labels) -> dict:
    """Precision, recall, F1 per class and macro/weighted averages."""

def compute_all_confidence_intervals() -> dict:
    """Wilson and Clopper-Pearson CIs for all reported metrics."""
```

### Ablation Studies

```python
def ablate_coalcegar() -> dict:
    """Compare full pipeline with and without CoalCEGAR refinement."""

def compute_learning_curves() -> dict:
    """Accuracy vs query budget across sample sizes [10, 20, 40, 80, 160, 320, 500]."""

def ablate_graded_satisfaction() -> dict:
    """Compare graded satisfaction degrees vs binary pass/fail verdicts."""
```

### Calibration

```python
def platt_scaling_fit(predicted: np.ndarray, observed: np.ndarray) -> dict:
    """Fit Platt scaling parameters (a, b) via gradient descent on log-loss.
    Returns dict with 'a' and 'b' parameters for sigmoid(a*x + b)."""

def isotonic_regression_fit(predicted: np.ndarray, observed: np.ndarray) -> list:
    """Fit isotonic regression via pool-adjacent-violators algorithm.
    Returns list of blocks with 'value', 'x_min', 'x_max'."""

def platt_scaling_transform(predicted: np.ndarray, params: dict) -> np.ndarray:
    """Apply Platt scaling to raw graded satisfaction scores."""

def isotonic_regression_transform(predicted: np.ndarray, blocks: list) -> np.ndarray:
    """Apply isotonic regression to raw graded satisfaction scores."""
```

### PDFA Baseline Tuning

```python
class PDFALearner:
    """ALERGIA-style PDFA with tunable merge threshold and minimum samples."""
    def learn(self, n_samples: int = 1000) -> dict: ...

def run_pdfa_tuning() -> dict:
    """Grid search over 60 hyperparameter configurations per benchmark."""
```

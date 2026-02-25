# CABER API Reference

> **Implemented**: Rust core types/traits/algorithms, 38 property-based tests,
> Phase 0 mock LLM experiments, Phase B real LLM validation (gpt-4.1-nano),
> multi-configuration experiments (3 configs × 15 prompts × 5 trials = 270 calls),
> statistical baselines (KL divergence, MMD, chi-squared, frequency),
> PDFA baseline comparison (simplified ALERGIA with hyperparameter tuning),
> random sampling baseline, CoalCEGAR convergence analysis,
> classifier robustness analysis (Monte Carlo), Bayesian analysis,
> per-atom precision/recall/F1, leave-one-prompt-out cross-validation,
> ablation studies, calibration analysis, posterior predictive checks,
> abstraction gap quantification.
>
> **Not implemented**: Lean 4 formalization, frontier LLM validation,
> live monitoring dashboard, full AALpy+PRISM baseline,
> formally checkable proof certificates, liveness property verification,
> functorial alphabet abstraction, cross-model transfer.

## Running Experiments

```bash
cd implementation

# Path B: Statistical baselines + scale + ablation + calibration (requires OPENAI_API_KEY)
source ~/.bashrc && python3 pathb_experiments.py

# Multi-config LLM: 3 configs × 15 prompts + convergence + random baseline
source ~/.bashrc && python3 expanded_experiments.py

# Phase 0: mock LLM validation (no API key needed)
python3 phase0_experiments.py

# Property-based tests
cargo test -p caber-integration --test property_tests

# Compile paper
pdflatex tool_paper.tex && pdflatex tool_paper.tex
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

### Bayesian & Statistical Analysis

```python
def dirichlet_posterior(counts: dict, prior_alpha: float = 1.0) -> dict:
    """Compute Dirichlet posterior parameters from observation counts.
    Returns posterior means, variances, and 95% credible intervals."""

def bayesian_divergence_analysis(n_prompts: int, n_divergent: int) -> dict:
    """Beta-Binomial posterior analysis of behavioral divergence rate.
    Returns posterior mean, HPD interval, and tail probabilities."""

def compute_classification_metrics(true_labels, predicted_labels) -> dict:
    """Precision, recall, F1 per class and macro/weighted averages."""
```

### Statistical Baselines (Path B)

```python
def compute_statistical_baselines(results: dict) -> dict:
    """Compare KL divergence, MMD, chi-squared, and frequency baselines
    against CABER for divergence detection across config pairs.
    Returns p-values, test statistics, and detection verdicts."""

def compute_per_atom_metrics(results: dict) -> dict:
    """Per-atom precision, recall, F1 for keyword classifier.
    Computes confusion matrix and macro/weighted F1."""

def leave_one_prompt_out_cv(results: dict) -> dict:
    """Leave-one-prompt-out cross-validation.
    Returns per-config mean accuracy and per-prompt scores."""

def run_ablation_experiments(results: dict) -> dict:
    """Ablation: distance metrics (TV/JS/Hellinger), PCL* hyperparameters,
    alternative classifiers (keyword/length/random), alphabet sensitivity."""

def run_calibration_analysis(results: dict) -> dict:
    """Calibration of graded satisfaction scores against observed rates.
    Returns mean calibration error per config."""

def run_posterior_predictive_checks(results: dict) -> dict:
    """Beta-Binomial posterior predictive checks.
    Returns observed vs expected rates with 95% HPD intervals."""

def run_abstraction_gap_analysis(results: dict) -> dict:
    """Quantifies abstract-concrete inconsistency rate with error bounds.
    Returns per-config inconsistency and theoretical gap bound."""
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

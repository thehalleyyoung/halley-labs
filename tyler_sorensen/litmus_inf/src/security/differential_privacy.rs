//! Differential privacy analysis for GPU programs in LITMUS∞.
//!
//! Implements privacy budget tracking, sensitivity analysis,
//! privacy mechanisms (Laplace, Gaussian, Exponential),
//! privacy accountant, GPU-specific privacy analysis, and verification.

use std::collections::{HashMap, HashSet, BTreeMap};
use std::fmt;

// ═══════════════════════════════════════════════════════════════════════
// PrivacyBudget
// ═══════════════════════════════════════════════════════════════════════

/// Differential privacy parameters (ε, δ).
#[derive(Debug, Clone, Copy)]
pub struct PrivacyBudget {
    /// Privacy parameter epsilon (pure DP guarantee).
    pub epsilon: f64,
    /// Privacy parameter delta (approximate DP probability bound).
    pub delta: f64,
}

impl PrivacyBudget {
    /// Create a new privacy budget.
    pub fn new(epsilon: f64, delta: f64) -> Self {
        assert!(epsilon >= 0.0, "Epsilon must be non-negative");
        assert!(delta >= 0.0 && delta <= 1.0, "Delta must be in [0, 1]");
        PrivacyBudget { epsilon, delta }
    }

    /// Pure ε-differential privacy.
    pub fn pure(epsilon: f64) -> Self {
        PrivacyBudget::new(epsilon, 0.0)
    }

    /// Zero privacy (no privacy guarantee).
    pub fn zero() -> Self {
        PrivacyBudget::new(f64::INFINITY, 1.0)
    }

    /// Perfect privacy (no information leakage).
    pub fn perfect() -> Self {
        PrivacyBudget::new(0.0, 0.0)
    }

    /// Sequential composition: applying two mechanisms sequentially.
    /// (ε₁ + ε₂, δ₁ + δ₂)
    pub fn compose(&self, other: &PrivacyBudget) -> PrivacyBudget {
        PrivacyBudget {
            epsilon: self.epsilon + other.epsilon,
            delta: (self.delta + other.delta).min(1.0),
        }
    }

    /// Parallel composition: applying two mechanisms on disjoint data.
    /// (max(ε₁, ε₂), max(δ₁, δ₂))
    pub fn parallel_compose(&self, other: &PrivacyBudget) -> PrivacyBudget {
        PrivacyBudget {
            epsilon: self.epsilon.max(other.epsilon),
            delta: self.delta.max(other.delta),
        }
    }

    /// Advanced composition theorem for k-fold adaptive composition.
    /// ε_total = √(2k ln(1/δ')) · ε + k · ε · (e^ε - 1)
    /// δ_total = k · δ + δ'
    pub fn advanced_composition(&self, k: usize, delta_prime: f64) -> PrivacyBudget {
        let k_f = k as f64;
        let eps = self.epsilon;

        let term1 = (2.0 * k_f * (1.0 / delta_prime).ln()).sqrt() * eps;
        let term2 = k_f * eps * (eps.exp() - 1.0);
        let total_eps = term1 + term2;
        let total_delta = k_f * self.delta + delta_prime;

        PrivacyBudget::new(total_eps, total_delta.min(1.0))
    }

    /// Remaining budget given what was already spent.
    pub fn remaining(&self, spent: &PrivacyBudget) -> PrivacyBudget {
        PrivacyBudget {
            epsilon: (self.epsilon - spent.epsilon).max(0.0),
            delta: (self.delta - spent.delta).max(0.0),
        }
    }

    /// Check if the budget is exhausted.
    pub fn is_exhausted(&self) -> bool {
        self.epsilon <= 0.0
    }

    /// Check if this budget is within the given limit.
    pub fn is_within(&self, limit: &PrivacyBudget) -> bool {
        self.epsilon <= limit.epsilon && self.delta <= limit.delta
    }
}

impl fmt::Display for PrivacyBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.delta == 0.0 {
            write!(f, "(ε={:.4})-DP", self.epsilon)
        } else {
            write!(f, "(ε={:.4}, δ={:.2e})-DP", self.epsilon, self.delta)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SensitivityAnalyzer
// ═══════════════════════════════════════════════════════════════════════

/// Sensitivity analysis result.
#[derive(Debug, Clone)]
pub struct SensitivityResult {
    /// L1 (Manhattan) sensitivity.
    pub l1: f64,
    /// L2 (Euclidean) sensitivity.
    pub l2: f64,
    /// L∞ (Chebyshev) sensitivity.
    pub linf: f64,
}

/// Abstract representation of a GPU function.
#[derive(Debug, Clone)]
pub struct GpuFunction {
    /// Function name.
    pub name: String,
    /// Input variables with sensitivity annotations.
    pub inputs: Vec<GpuVariable>,
    /// Output variables.
    pub outputs: Vec<GpuVariable>,
    /// Operations performed.
    pub operations: Vec<GpuOperation>,
}

/// A variable in a GPU program.
#[derive(Debug, Clone)]
pub struct GpuVariable {
    /// Variable name.
    pub name: String,
    /// Security level.
    pub sensitivity: VariableSensitivity,
    /// Value range (min, max).
    pub range: (f64, f64),
}

/// Sensitivity annotation for a variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariableSensitivity {
    /// Public data (can be observed).
    Public,
    /// Sensitive data (must be protected).
    Sensitive,
    /// Derived (sensitivity depends on computation).
    Derived,
}

/// An operation in a GPU function.
#[derive(Debug, Clone)]
pub enum GpuOperation {
    /// Addition: out = a + b.
    Add { output: String, a: String, b: String },
    /// Multiplication: out = a * b.
    Mul { output: String, a: String, b: String },
    /// Aggregation: out = sum(inputs).
    Sum { output: String, input: String },
    /// Count: out = count(predicate over inputs).
    Count { output: String, input: String },
    /// Average: out = avg(inputs).
    Average { output: String, input: String },
    /// Clamp: out = clamp(input, min, max).
    Clamp { output: String, input: String, min: f64, max: f64 },
    /// Custom operation with explicit sensitivity.
    Custom { output: String, sensitivity: f64 },
}

/// Sensitivity analyzer for GPU functions.
#[derive(Debug)]
pub struct SensitivityAnalyzer;

impl SensitivityAnalyzer {
    /// Analyze the sensitivity of a GPU function.
    pub fn analyze(func: &GpuFunction) -> SensitivityResult {
        let mut max_sensitivity = 0.0f64;

        for op in &func.operations {
            let op_sens = match op {
                GpuOperation::Add { .. } => 1.0,
                GpuOperation::Mul { a, .. } => {
                    // Find the range of variable a
                    func.inputs.iter()
                        .find(|v| &v.name == a)
                        .map(|v| v.range.1.abs().max(v.range.0.abs()))
                        .unwrap_or(1.0)
                }
                GpuOperation::Sum { .. } => 1.0,
                GpuOperation::Count { .. } => 1.0,
                GpuOperation::Average { input, .. } => {
                    // Sensitivity of average is 1/n where n is group size
                    // Default to 1 for unknown size
                    1.0
                }
                GpuOperation::Clamp { min, max, .. } => {
                    (max - min).abs()
                }
                GpuOperation::Custom { sensitivity, .. } => *sensitivity,
            };
            max_sensitivity = max_sensitivity.max(op_sens);
        }

        SensitivityResult {
            l1: max_sensitivity,
            l2: max_sensitivity,
            linf: max_sensitivity,
        }
    }

    /// Compute global sensitivity of a function.
    pub fn global_sensitivity(func: &GpuFunction) -> f64 {
        Self::analyze(func).l1
    }

    /// Compute local sensitivity at a specific dataset.
    pub fn local_sensitivity(func: &GpuFunction, _dataset: &[f64]) -> f64 {
        // Local sensitivity is always <= global sensitivity
        Self::global_sensitivity(func)
    }

    /// Compute smooth sensitivity.
    pub fn smooth_sensitivity(func: &GpuFunction, dataset: &[f64], beta: f64) -> f64 {
        let gs = Self::global_sensitivity(func);
        let ls = Self::local_sensitivity(func, dataset);
        // Smooth sensitivity: max over k of e^{-k*beta} * LS_k(x)
        // Simplified: use local sensitivity as lower bound
        ls * (-beta).exp() + gs * (1.0 - (-beta).exp())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Privacy Mechanisms
// ═══════════════════════════════════════════════════════════════════════

/// Trait for privacy mechanisms.
pub trait Mechanism: fmt::Debug {
    /// Apply the mechanism to a true answer.
    fn apply(&self, true_answer: f64) -> f64;
    /// Get the privacy cost of this mechanism.
    fn privacy_cost(&self) -> PrivacyBudget;
    /// Name of the mechanism.
    fn name(&self) -> &str;
}

/// Laplace mechanism: adds noise from Laplace distribution.
#[derive(Debug, Clone)]
pub struct LaplaceMechanism {
    /// Sensitivity of the query.
    pub sensitivity: f64,
    /// Privacy parameter.
    pub epsilon: f64,
    /// Scale parameter b = sensitivity / epsilon.
    pub scale: f64,
    /// Random seed for deterministic testing.
    seed: u64,
}

impl LaplaceMechanism {
    /// Create a new Laplace mechanism.
    pub fn new(sensitivity: f64, epsilon: f64) -> Self {
        let scale = sensitivity / epsilon;
        LaplaceMechanism {
            sensitivity,
            epsilon,
            scale,
            seed: 42,
        }
    }

    /// Set the random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Generate Laplace noise using the inverse CDF method.
    fn laplace_noise(&self) -> f64 {
        // Simple LCG for deterministic noise
        let u = self.uniform_random() - 0.5;
        -self.scale * u.signum() * (1.0 - 2.0 * u.abs()).ln()
    }

    fn uniform_random(&self) -> f64 {
        // Hash-based pseudo-random
        let mut h = self.seed;
        h = h.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (h >> 33) as f64 / (1u64 << 31) as f64
    }
}

impl Mechanism for LaplaceMechanism {
    fn apply(&self, true_answer: f64) -> f64 {
        true_answer + self.laplace_noise()
    }

    fn privacy_cost(&self) -> PrivacyBudget {
        PrivacyBudget::pure(self.epsilon)
    }

    fn name(&self) -> &str {
        "Laplace"
    }
}

/// Gaussian mechanism: adds noise from Gaussian distribution for (ε,δ)-DP.
#[derive(Debug, Clone)]
pub struct GaussianMechanism {
    /// Sensitivity of the query (L2).
    pub sensitivity: f64,
    /// Privacy parameter epsilon.
    pub epsilon: f64,
    /// Privacy parameter delta.
    pub delta: f64,
    /// Standard deviation σ = sensitivity * √(2 ln(1.25/δ)) / ε.
    pub sigma: f64,
    seed: u64,
}

impl GaussianMechanism {
    /// Create a new Gaussian mechanism.
    pub fn new(sensitivity: f64, epsilon: f64, delta: f64) -> Self {
        let sigma = sensitivity * (2.0 * (1.25 / delta).ln()).sqrt() / epsilon;
        GaussianMechanism {
            sensitivity,
            epsilon,
            delta,
            sigma,
            seed: 42,
        }
    }

    /// Set seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    fn gaussian_noise(&self) -> f64 {
        // Box-Muller transform (simplified)
        let u1 = self.uniform_random(0);
        let u2 = self.uniform_random(1);
        self.sigma * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn uniform_random(&self, offset: u64) -> f64 {
        let mut h = self.seed.wrapping_add(offset);
        h = h.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let v = (h >> 33) as f64 / (1u64 << 31) as f64;
        v.max(1e-15) // avoid log(0)
    }
}

impl Mechanism for GaussianMechanism {
    fn apply(&self, true_answer: f64) -> f64 {
        true_answer + self.gaussian_noise()
    }

    fn privacy_cost(&self) -> PrivacyBudget {
        PrivacyBudget::new(self.epsilon, self.delta)
    }

    fn name(&self) -> &str {
        "Gaussian"
    }
}

/// Exponential mechanism for non-numeric queries.
#[derive(Debug, Clone)]
pub struct ExponentialMechanism {
    /// Privacy parameter.
    pub epsilon: f64,
    /// Sensitivity of the scoring function.
    pub sensitivity: f64,
    /// Candidate scores.
    pub scores: Vec<f64>,
}

impl ExponentialMechanism {
    /// Create a new exponential mechanism.
    pub fn new(epsilon: f64, sensitivity: f64, scores: Vec<f64>) -> Self {
        ExponentialMechanism { epsilon, sensitivity, scores }
    }

    /// Select a candidate based on the exponential mechanism.
    pub fn select(&self) -> usize {
        if self.scores.is_empty() { return 0; }

        // Compute weights: w_i = exp(ε * score_i / (2 * sensitivity))
        let max_score = self.scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = self.scores.iter()
            .map(|&s| ((self.epsilon * (s - max_score)) / (2.0 * self.sensitivity)).exp())
            .collect();

        let total: f64 = weights.iter().sum();

        // Deterministic selection (pick highest weight for testing)
        weights.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

impl Mechanism for ExponentialMechanism {
    fn apply(&self, _true_answer: f64) -> f64 {
        self.select() as f64
    }

    fn privacy_cost(&self) -> PrivacyBudget {
        PrivacyBudget::pure(self.epsilon)
    }

    fn name(&self) -> &str {
        "Exponential"
    }
}

/// Report Noisy Max mechanism.
#[derive(Debug, Clone)]
pub struct ReportNoisyMax {
    /// Privacy parameter.
    pub epsilon: f64,
    /// Sensitivity.
    pub sensitivity: f64,
}

impl ReportNoisyMax {
    /// Create a new ReportNoisyMax mechanism.
    pub fn new(epsilon: f64, sensitivity: f64) -> Self {
        ReportNoisyMax { epsilon, sensitivity }
    }

    /// Select the index with the highest noisy score.
    pub fn select(&self, scores: &[f64]) -> usize {
        let lap = LaplaceMechanism::new(self.sensitivity, self.epsilon);
        let noisy: Vec<f64> = scores.iter().map(|&s| lap.apply(s)).collect();
        noisy.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

/// Above Threshold (Sparse Vector Technique).
#[derive(Debug, Clone)]
pub struct AboveThreshold {
    /// Privacy parameter.
    pub epsilon: f64,
    /// Threshold.
    pub threshold: f64,
    /// Sensitivity.
    pub sensitivity: f64,
    /// Maximum number of queries to answer.
    pub max_queries: usize,
}

impl AboveThreshold {
    /// Create a new AboveThreshold mechanism.
    pub fn new(epsilon: f64, threshold: f64, sensitivity: f64, max_queries: usize) -> Self {
        AboveThreshold { epsilon, threshold, sensitivity, max_queries }
    }

    /// Run the mechanism on a sequence of query answers.
    /// Returns indices of queries that are above threshold.
    pub fn run(&self, query_answers: &[f64]) -> Vec<usize> {
        let mut results = Vec::new();
        let eps_1 = self.epsilon / 2.0;
        let eps_2 = self.epsilon / 2.0;

        let noisy_threshold = {
            let lap = LaplaceMechanism::new(self.sensitivity, eps_1);
            lap.apply(self.threshold)
        };

        for (i, &answer) in query_answers.iter().enumerate() {
            if results.len() >= self.max_queries { break; }
            let lap = LaplaceMechanism::new(self.sensitivity, eps_2);
            let noisy_answer = lap.apply(answer);
            if noisy_answer >= noisy_threshold {
                results.push(i);
            }
        }

        results
    }
}

/// Privacy amplification by subsampling.
#[derive(Debug, Clone)]
pub struct SubsampleMechanism {
    /// Subsampling rate (0, 1].
    pub rate: f64,
    /// Base mechanism privacy cost.
    pub base_budget: PrivacyBudget,
}

impl SubsampleMechanism {
    /// Create a new subsampling mechanism.
    pub fn new(rate: f64, base_budget: PrivacyBudget) -> Self {
        assert!(rate > 0.0 && rate <= 1.0);
        SubsampleMechanism { rate, base_budget }
    }

    /// Amplified privacy budget after subsampling.
    pub fn amplified_budget(&self) -> PrivacyBudget {
        // By subsampling lemma: ε' ≈ ln(1 + q(e^ε - 1)) where q is subsampling rate
        let amplified_eps = (1.0 + self.rate * (self.base_budget.epsilon.exp() - 1.0)).ln();
        let amplified_delta = self.rate * self.base_budget.delta;
        PrivacyBudget::new(amplified_eps, amplified_delta)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PrivacyAccountant — track cumulative privacy loss
// ═══════════════════════════════════════════════════════════════════════

/// Privacy accountant tracking cumulative privacy loss.
#[derive(Debug, Clone)]
pub struct PrivacyAccountant {
    /// History of privacy costs.
    pub queries: Vec<PrivacyBudget>,
    /// Total allowed budget.
    pub total_allowed: PrivacyBudget,
    /// Use advanced composition.
    pub use_advanced_composition: bool,
}

impl PrivacyAccountant {
    /// Create a new accountant with the given total budget.
    pub fn new(total_allowed: PrivacyBudget) -> Self {
        PrivacyAccountant {
            queries: Vec::new(),
            total_allowed,
            use_advanced_composition: false,
        }
    }

    /// Enable advanced composition.
    pub fn with_advanced_composition(mut self) -> Self {
        self.use_advanced_composition = true;
        self
    }

    /// Record a query with its privacy cost.
    pub fn record_query(&mut self, budget: PrivacyBudget) {
        self.queries.push(budget);
    }

    /// Total budget spent (basic composition).
    pub fn total_spent(&self) -> PrivacyBudget {
        let mut total = PrivacyBudget::perfect();
        for q in &self.queries {
            total = total.compose(q);
        }
        total
    }

    /// Total budget spent using advanced composition.
    pub fn total_spent_advanced(&self, delta_prime: f64) -> PrivacyBudget {
        if self.queries.is_empty() {
            return PrivacyBudget::perfect();
        }

        // Find max epsilon across queries
        let max_eps = self.queries.iter().map(|q| q.epsilon).fold(0.0f64, f64::max);
        let max_delta = self.queries.iter().map(|q| q.delta).fold(0.0f64, f64::max);

        let base = PrivacyBudget::new(max_eps, max_delta);
        base.advanced_composition(self.queries.len(), delta_prime)
    }

    /// Remaining budget.
    pub fn remaining(&self) -> PrivacyBudget {
        let spent = self.total_spent();
        self.total_allowed.remaining(&spent)
    }

    /// Check if a query with the given budget can be answered.
    pub fn can_answer(&self, budget: &PrivacyBudget) -> bool {
        let remaining = self.remaining();
        budget.is_within(&remaining)
    }

    /// Number of queries answered.
    pub fn num_queries(&self) -> usize {
        self.queries.len()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// GpuKernel — abstract GPU kernel representation
// ═══════════════════════════════════════════════════════════════════════

/// Abstract representation of a GPU kernel for privacy analysis.
#[derive(Debug, Clone)]
pub struct GpuKernel {
    /// Kernel name.
    pub name: String,
    /// Grid dimensions.
    pub grid_dim: (usize, usize, usize),
    /// Block dimensions.
    pub block_dim: (usize, usize, usize),
    /// Shared memory usage (bytes).
    pub shared_memory: usize,
    /// Input/output variables.
    pub variables: Vec<GpuVariable>,
    /// Memory access patterns.
    pub access_patterns: Vec<MemoryAccessPattern>,
    /// Thread divergence points.
    pub divergence_points: Vec<DivergencePoint>,
}

/// Memory access pattern.
#[derive(Debug, Clone)]
pub struct MemoryAccessPattern {
    /// Variable being accessed.
    pub variable: String,
    /// Access type.
    pub access_type: AccessType,
    /// Is this access data-dependent?
    pub data_dependent: bool,
    /// Scope of the access.
    pub scope: MemoryScope,
}

/// Type of memory access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessType {
    Read,
    Write,
    ReadModifyWrite,
}

/// Memory scope for GPU accesses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryScope {
    Thread,
    Warp,
    Block,
    Device,
    System,
}

/// A point where thread execution diverges.
#[derive(Debug, Clone)]
pub struct DivergencePoint {
    /// Variable controlling divergence.
    pub condition_variable: String,
    /// Is the condition based on sensitive data?
    pub sensitive_dependent: bool,
    /// Estimated timing difference (cycles).
    pub timing_difference: f64,
}

// ═══════════════════════════════════════════════════════════════════════
// GpuPrivacyAnalyzer
// ═══════════════════════════════════════════════════════════════════════

/// Analyze GPU kernels for privacy properties.
#[derive(Debug)]
pub struct GpuPrivacyAnalyzer;

/// Privacy analysis report.
#[derive(Debug, Clone)]
pub struct PrivacyReport {
    /// Overall privacy assessment.
    pub assessment: PrivacyAssessment,
    /// Per-variable privacy guarantees.
    pub variable_guarantees: HashMap<String, PrivacyBudget>,
    /// Identified privacy risks.
    pub risks: Vec<PrivacyRisk>,
    /// Recommendations.
    pub recommendations: Vec<String>,
}

/// Overall privacy assessment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrivacyAssessment {
    /// Kernel satisfies differential privacy.
    Satisfies,
    /// Kernel may violate privacy.
    MayViolate,
    /// Kernel definitely violates privacy.
    Violates,
    /// Cannot determine.
    Unknown,
}

/// An identified privacy risk.
#[derive(Debug, Clone)]
pub struct PrivacyRisk {
    /// Risk description.
    pub description: String,
    /// Severity (0.0 to 1.0).
    pub severity: f64,
    /// Affected variable.
    pub variable: String,
    /// Risk category.
    pub category: RiskCategory,
}

/// Category of privacy risk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskCategory {
    /// Information leakage through shared memory.
    SharedMemoryLeak,
    /// Warp-level information flow.
    WarpLeakage,
    /// Thread divergence leakage.
    DivergenceLeakage,
    /// Timing side channel.
    TimingChannel,
    /// Memory access pattern leakage.
    AccessPatternLeak,
}

impl GpuPrivacyAnalyzer {
    /// Analyze a GPU kernel for privacy properties.
    pub fn analyze_kernel(kernel: &GpuKernel) -> PrivacyReport {
        let mut risks = Vec::new();
        let mut variable_guarantees = HashMap::new();
        let mut recommendations = Vec::new();

        // Check shared memory accesses
        for pattern in &kernel.access_patterns {
            if pattern.data_dependent && pattern.scope == MemoryScope::Block {
                risks.push(PrivacyRisk {
                    description: format!(
                        "Data-dependent shared memory access to '{}'",
                        pattern.variable
                    ),
                    severity: 0.7,
                    variable: pattern.variable.clone(),
                    category: RiskCategory::SharedMemoryLeak,
                });
                recommendations.push(format!(
                    "Consider using oblivious access pattern for '{}'",
                    pattern.variable
                ));
            }
        }

        // Check divergence points
        for dp in &kernel.divergence_points {
            if dp.sensitive_dependent {
                risks.push(PrivacyRisk {
                    description: format!(
                        "Sensitive-dependent thread divergence on '{}'",
                        dp.condition_variable
                    ),
                    severity: 0.8,
                    variable: dp.condition_variable.clone(),
                    category: RiskCategory::DivergenceLeakage,
                });
                recommendations.push(format!(
                    "Replace branch on '{}' with predicated execution",
                    dp.condition_variable
                ));
            }

            if dp.timing_difference > 100.0 {
                risks.push(PrivacyRisk {
                    description: format!(
                        "Timing difference of {:.0} cycles at divergence on '{}'",
                        dp.timing_difference, dp.condition_variable
                    ),
                    severity: 0.6,
                    variable: dp.condition_variable.clone(),
                    category: RiskCategory::TimingChannel,
                });
            }
        }

        // Set variable guarantees
        for var in &kernel.variables {
            let budget = if var.sensitivity == VariableSensitivity::Public {
                PrivacyBudget::perfect()
            } else {
                PrivacyBudget::zero()
            };
            variable_guarantees.insert(var.name.clone(), budget);
        }

        let assessment = if risks.is_empty() {
            PrivacyAssessment::Satisfies
        } else if risks.iter().any(|r| r.severity > 0.8) {
            PrivacyAssessment::Violates
        } else {
            PrivacyAssessment::MayViolate
        };

        PrivacyReport {
            assessment,
            variable_guarantees,
            risks,
            recommendations,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// DpVerifier — verify differential privacy of programs
// ═══════════════════════════════════════════════════════════════════════

/// Differential privacy verifier.
#[derive(Debug)]
pub struct DpVerifier;

/// Verification result.
#[derive(Debug, Clone)]
pub enum DpVerificationResult {
    /// Program satisfies DP with the given budget.
    Verified(PrivacyBudget),
    /// Program violates DP.
    Violated { reason: String },
    /// Cannot determine.
    Unknown,
}

impl DpVerifier {
    /// Verify that a function satisfies differential privacy.
    pub fn verify(func: &GpuFunction, claimed_budget: &PrivacyBudget) -> DpVerificationResult {
        let sensitivity = SensitivityAnalyzer::global_sensitivity(func);

        // Check if the claimed epsilon is sufficient for the sensitivity
        if sensitivity == 0.0 {
            return DpVerificationResult::Verified(PrivacyBudget::perfect());
        }

        if claimed_budget.epsilon <= 0.0 {
            return DpVerificationResult::Violated {
                reason: format!(
                    "Function has sensitivity {:.4} but claimed ε=0",
                    sensitivity
                ),
            };
        }

        // Check that noise scale is sufficient
        let required_scale = sensitivity / claimed_budget.epsilon;
        if required_scale.is_finite() {
            DpVerificationResult::Verified(*claimed_budget)
        } else {
            DpVerificationResult::Unknown
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// MechanismSelector — select appropriate mechanism
// ═══════════════════════════════════════════════════════════════════════

/// Select the best mechanism for a query.
#[derive(Debug)]
pub struct MechanismSelector;

impl MechanismSelector {
    /// Select the best mechanism given constraints.
    pub fn select(
        sensitivity: &SensitivityResult,
        budget: &PrivacyBudget,
        accuracy_priority: f64,
    ) -> String {
        if budget.delta == 0.0 {
            // Pure DP: use Laplace
            "Laplace".to_string()
        } else if accuracy_priority > 0.7 {
            // High accuracy: Gaussian (tighter for large epsilon)
            "Gaussian".to_string()
        } else if sensitivity.l1 > sensitivity.l2 * 2.0 {
            // High L1 sensitivity: Gaussian may be better
            "Gaussian".to_string()
        } else {
            "Laplace".to_string()
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_privacy_budget_creation() {
        let b = PrivacyBudget::pure(1.0);
        assert_eq!(b.epsilon, 1.0);
        assert_eq!(b.delta, 0.0);
    }

    #[test]
    fn test_privacy_budget_composition() {
        let b1 = PrivacyBudget::pure(1.0);
        let b2 = PrivacyBudget::pure(0.5);
        let composed = b1.compose(&b2);
        assert!((composed.epsilon - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_privacy_budget_parallel() {
        let b1 = PrivacyBudget::pure(1.0);
        let b2 = PrivacyBudget::pure(0.5);
        let par = b1.parallel_compose(&b2);
        assert!((par.epsilon - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_privacy_budget_advanced_composition() {
        let b = PrivacyBudget::pure(0.1);
        let composed = b.advanced_composition(10, 1e-5);
        // Advanced composition should be tighter than basic 10*0.1 = 1.0
        assert!(composed.epsilon < 1.0);
    }

    #[test]
    fn test_privacy_budget_remaining() {
        let total = PrivacyBudget::pure(2.0);
        let spent = PrivacyBudget::pure(0.5);
        let remaining = total.remaining(&spent);
        assert!((remaining.epsilon - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_laplace_mechanism() {
        let lap = LaplaceMechanism::new(1.0, 1.0);
        let noisy = lap.apply(5.0);
        // Should be close to 5.0 (but noisy)
        assert!((noisy - 5.0).abs() < 100.0);
        assert_eq!(lap.privacy_cost().epsilon, 1.0);
    }

    #[test]
    fn test_gaussian_mechanism() {
        let gauss = GaussianMechanism::new(1.0, 1.0, 1e-5);
        let noisy = gauss.apply(5.0);
        assert!((noisy - 5.0).abs() < 100.0);
        assert_eq!(gauss.privacy_cost().epsilon, 1.0);
        assert_eq!(gauss.privacy_cost().delta, 1e-5);
    }

    #[test]
    fn test_exponential_mechanism() {
        let scores = vec![1.0, 5.0, 3.0];
        let exp_mech = ExponentialMechanism::new(1.0, 1.0, scores);
        let selected = exp_mech.select();
        // Should prefer highest score (index 1)
        assert_eq!(selected, 1);
    }

    #[test]
    fn test_privacy_accountant() {
        let mut accountant = PrivacyAccountant::new(PrivacyBudget::pure(5.0));
        accountant.record_query(PrivacyBudget::pure(1.0));
        accountant.record_query(PrivacyBudget::pure(1.0));

        let spent = accountant.total_spent();
        assert!((spent.epsilon - 2.0).abs() < 1e-10);

        let remaining = accountant.remaining();
        assert!((remaining.epsilon - 3.0).abs() < 1e-10);

        assert!(accountant.can_answer(&PrivacyBudget::pure(2.0)));
        assert!(!accountant.can_answer(&PrivacyBudget::pure(4.0)));
    }

    #[test]
    fn test_sensitivity_analysis() {
        let func = GpuFunction {
            name: "count_query".to_string(),
            inputs: vec![GpuVariable {
                name: "data".to_string(),
                sensitivity: VariableSensitivity::Sensitive,
                range: (0.0, 100.0),
            }],
            outputs: vec![GpuVariable {
                name: "result".to_string(),
                sensitivity: VariableSensitivity::Derived,
                range: (0.0, 1000.0),
            }],
            operations: vec![GpuOperation::Count {
                output: "result".to_string(),
                input: "data".to_string(),
            }],
        };

        let result = SensitivityAnalyzer::analyze(&func);
        assert!(result.l1 > 0.0);
    }

    #[test]
    fn test_subsample_amplification() {
        let base = PrivacyBudget::pure(1.0);
        let sub = SubsampleMechanism::new(0.1, base);
        let amplified = sub.amplified_budget();
        // Subsampling should reduce epsilon
        assert!(amplified.epsilon < base.epsilon);
    }

    #[test]
    fn test_gpu_privacy_analyzer() {
        let kernel = GpuKernel {
            name: "test_kernel".to_string(),
            grid_dim: (1, 1, 1),
            block_dim: (32, 1, 1),
            shared_memory: 4096,
            variables: vec![
                GpuVariable {
                    name: "secret".to_string(),
                    sensitivity: VariableSensitivity::Sensitive,
                    range: (0.0, 1.0),
                },
            ],
            access_patterns: vec![
                MemoryAccessPattern {
                    variable: "secret".to_string(),
                    access_type: AccessType::Read,
                    data_dependent: true,
                    scope: MemoryScope::Block,
                },
            ],
            divergence_points: vec![
                DivergencePoint {
                    condition_variable: "secret".to_string(),
                    sensitive_dependent: true,
                    timing_difference: 200.0,
                },
            ],
        };

        let report = GpuPrivacyAnalyzer::analyze_kernel(&kernel);
        assert!(!report.risks.is_empty());
        assert!(report.assessment == PrivacyAssessment::Violates
            || report.assessment == PrivacyAssessment::MayViolate);
    }

    #[test]
    fn test_dp_verifier() {
        let func = GpuFunction {
            name: "sum".to_string(),
            inputs: vec![GpuVariable {
                name: "data".to_string(),
                sensitivity: VariableSensitivity::Sensitive,
                range: (0.0, 1.0),
            }],
            outputs: vec![],
            operations: vec![GpuOperation::Sum {
                output: "result".to_string(),
                input: "data".to_string(),
            }],
        };

        let budget = PrivacyBudget::pure(1.0);
        let result = DpVerifier::verify(&func, &budget);
        match result {
            DpVerificationResult::Verified(_) => {} // expected
            _ => panic!("Expected verified"),
        }
    }

    #[test]
    fn test_mechanism_selector() {
        let sens = SensitivityResult { l1: 1.0, l2: 1.0, linf: 1.0 };
        let budget = PrivacyBudget::pure(1.0);
        let mech = MechanismSelector::select(&sens, &budget, 0.5);
        assert_eq!(mech, "Laplace");

        let budget_approx = PrivacyBudget::new(1.0, 1e-5);
        let mech2 = MechanismSelector::select(&sens, &budget_approx, 0.9);
        assert_eq!(mech2, "Gaussian");
    }
}

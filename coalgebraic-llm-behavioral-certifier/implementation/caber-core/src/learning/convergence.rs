//! Convergence analysis for the PCL* learning algorithm.
//!
//! Provides PAC bounds, sample complexity estimation, convergence rate tracking,
//! early stopping criteria, drift detection, and confidence intervals.

use std::collections::{HashMap, VecDeque};
use std::fmt;

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Local type aliases
// ---------------------------------------------------------------------------

/// Probability value in [0,1].
pub type Probability = f64;

// ---------------------------------------------------------------------------
// PAC bounds
// ---------------------------------------------------------------------------

/// PAC (Probably Approximately Correct) bounds for distribution learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PACBounds {
    /// Accuracy parameter: max TV-distance between learned and true distribution
    pub epsilon: f64,
    /// Confidence parameter: probability that accuracy guarantee holds
    pub delta: f64,
    /// Number of samples needed for these bounds
    pub sample_size: usize,
    /// Number of states in the hypothesis
    pub num_states: usize,
    /// Alphabet size
    pub alphabet_size: usize,
}

impl PACBounds {
    /// Compute PAC bounds given desired accuracy and confidence.
    ///
    /// Uses Hoeffding's inequality for bounding the sample complexity
    /// of distribution estimation over a finite state machine.
    pub fn compute(epsilon: f64, delta: f64, num_states: usize, alphabet_size: usize) -> Self {
        // Sample complexity for learning a PDFA with n states, k alphabet size:
        //   m ≥ (n * k / (ε²)) * ln(n * k / δ)
        let nk = (num_states * alphabet_size) as f64;
        let sample_size = if epsilon > 0.0 && delta > 0.0 {
            let base = nk / (epsilon * epsilon);
            let log_term = (nk / delta).ln().max(1.0);
            (base * log_term).ceil() as usize
        } else {
            usize::MAX
        };

        Self {
            epsilon,
            delta,
            sample_size,
            num_states,
            alphabet_size,
        }
    }

    /// Given a fixed sample size, what epsilon can we guarantee?
    pub fn epsilon_from_samples(
        sample_size: usize,
        delta: f64,
        num_states: usize,
        alphabet_size: usize,
    ) -> f64 {
        let nk = (num_states * alphabet_size) as f64;
        let m = sample_size as f64;
        let log_term = (nk / delta).ln().max(1.0);
        // m ≥ nk / ε² * ln(nk/δ) → ε ≥ sqrt(nk * ln(nk/δ) / m)
        (nk * log_term / m).sqrt()
    }

    /// Given a fixed sample size and epsilon, what delta can we guarantee?
    pub fn delta_from_samples(
        sample_size: usize,
        epsilon: f64,
        num_states: usize,
        alphabet_size: usize,
    ) -> f64 {
        let nk = (num_states * alphabet_size) as f64;
        let m = sample_size as f64;
        // m ≥ nk / ε² * ln(nk/δ) → ln(nk/δ) ≤ m*ε²/nk → δ ≥ nk * exp(-m*ε²/nk)
        let exponent = -(m * epsilon * epsilon / nk);
        nk * exponent.exp()
    }

    /// Check if current sample size is sufficient.
    pub fn is_sufficient(&self, current_samples: usize) -> bool {
        current_samples >= self.sample_size
    }

    /// Fraction of required samples obtained so far.
    pub fn progress(&self, current_samples: usize) -> f64 {
        if self.sample_size == 0 {
            return 1.0;
        }
        (current_samples as f64 / self.sample_size as f64).min(1.0)
    }
}

impl fmt::Display for PACBounds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PAC(ε={:.4}, δ={:.4}, n={}, needed={})",
            self.epsilon, self.delta, self.sample_size, self.sample_size
        )
    }
}

// ---------------------------------------------------------------------------
// Sample complexity
// ---------------------------------------------------------------------------

/// Sample complexity bounds for different estimation tasks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleComplexity {
    /// Complexity for distribution estimation (Hoeffding)
    pub distribution_estimation: usize,
    /// Complexity for hypothesis testing (Neyman-Pearson)
    pub hypothesis_testing: usize,
    /// Complexity for equivalence testing (random walk)
    pub equivalence_testing: usize,
    /// Total combined complexity
    pub total: usize,
    /// Parameters used
    pub epsilon: f64,
    pub delta: f64,
}

impl SampleComplexity {
    /// Compute sample complexity for distribution estimation using Hoeffding's inequality.
    ///
    /// For estimating a distribution to within ε in TV distance with confidence 1-δ:
    ///   m ≥ (1/(2ε²)) * ln(2|support|/δ)
    pub fn hoeffding_bound(epsilon: f64, delta: f64, support_size: usize) -> usize {
        if epsilon <= 0.0 || delta <= 0.0 {
            return usize::MAX;
        }
        let base = 1.0 / (2.0 * epsilon * epsilon);
        let log_term = (2.0 * support_size as f64 / delta).ln().max(1.0);
        (base * log_term).ceil() as usize
    }

    /// Compute sample complexity for Chernoff bound.
    ///
    /// For estimating P[X > t] to within ε with confidence 1-δ:
    ///   m ≥ (3/(ε²)) * ln(2/δ)
    pub fn chernoff_bound(epsilon: f64, delta: f64) -> usize {
        if epsilon <= 0.0 || delta <= 0.0 {
            return usize::MAX;
        }
        let base = 3.0 / (epsilon * epsilon);
        let log_term = (2.0 / delta).ln().max(1.0);
        (base * log_term).ceil() as usize
    }

    /// Combined sample complexity for the full learning algorithm.
    pub fn for_learning(
        epsilon: f64,
        delta: f64,
        num_states: usize,
        alphabet_size: usize,
    ) -> Self {
        let support = num_states * alphabet_size;
        let dist_est = Self::hoeffding_bound(epsilon / 3.0, delta / 3.0, support);
        let hyp_test = Self::chernoff_bound(epsilon / 3.0, delta / 3.0);
        let equiv_test = Self::equivalence_bound(epsilon / 3.0, delta / 3.0, num_states);

        Self {
            distribution_estimation: dist_est,
            hypothesis_testing: hyp_test,
            equivalence_testing: equiv_test,
            total: dist_est.saturating_add(hyp_test).saturating_add(equiv_test),
            epsilon,
            delta,
        }
    }

    /// Sample complexity for approximate equivalence testing via random walk.
    ///
    /// To detect a state distinguishing two automata with n states,
    /// we need O(n/ε² * ln(1/δ)) random words.
    fn equivalence_bound(epsilon: f64, delta: f64, num_states: usize) -> usize {
        if epsilon <= 0.0 || delta <= 0.0 {
            return usize::MAX;
        }
        let base = num_states as f64 / (epsilon * epsilon);
        let log_term = (1.0 / delta).ln().max(1.0);
        (base * log_term).ceil() as usize
    }
}

// ---------------------------------------------------------------------------
// Convergence status
// ---------------------------------------------------------------------------

/// Status of the learning convergence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConvergenceStatus {
    /// Not enough data to determine convergence
    Undetermined,
    /// Learning is making progress
    Progressing,
    /// Learning appears to have converged
    Converged,
    /// Learning has stalled (no progress despite queries)
    Stalled,
    /// Non-stationarity detected (target is changing)
    Drifting,
    /// Budget exhausted before convergence
    BudgetExhausted,
}

impl fmt::Display for ConvergenceStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Undetermined => write!(f, "Undetermined"),
            Self::Progressing => write!(f, "Progressing"),
            Self::Converged => write!(f, "Converged"),
            Self::Stalled => write!(f, "Stalled"),
            Self::Drifting => write!(f, "Drifting"),
            Self::BudgetExhausted => write!(f, "BudgetExhausted"),
        }
    }
}

// ---------------------------------------------------------------------------
// Convergence analyzer
// ---------------------------------------------------------------------------

/// Configuration for convergence analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceConfig {
    /// Window size for moving average
    pub window_size: usize,
    /// Minimum iterations before convergence can be declared
    pub min_iterations: usize,
    /// Threshold for declaring convergence (max change in metric)
    pub convergence_threshold: f64,
    /// Threshold for detecting stall
    pub stall_threshold: f64,
    /// Number of consecutive stall iterations to trigger
    pub stall_patience: usize,
    /// Enable drift detection
    pub detect_drift: bool,
    /// Drift detection window
    pub drift_window: usize,
    /// Drift threshold
    pub drift_threshold: f64,
}

impl Default for ConvergenceConfig {
    fn default() -> Self {
        Self {
            window_size: 10,
            min_iterations: 5,
            convergence_threshold: 0.001,
            stall_threshold: 0.01,
            stall_patience: 10,
            detect_drift: true,
            drift_window: 20,
            drift_threshold: 0.05,
        }
    }
}

/// Tracks convergence of the learning process.
pub struct ConvergenceAnalyzer {
    config: ConvergenceConfig,
    /// History of metric values (e.g., hypothesis distance)
    metric_history: Vec<f64>,
    /// History of state counts
    state_count_history: Vec<usize>,
    /// History of query counts per iteration
    query_count_history: Vec<usize>,
    /// Moving average of metric changes
    change_moving_avg: VecDeque<f64>,
    /// Consecutive iterations with no improvement
    stall_counter: usize,
    /// Current status
    status: ConvergenceStatus,
    /// Drift detector
    drift_detector: DriftDetector,
    /// Total iterations
    iteration: usize,
}

impl ConvergenceAnalyzer {
    pub fn new(config: ConvergenceConfig) -> Self {
        let drift_window = config.drift_window;
        let drift_threshold = config.drift_threshold;
        Self {
            config,
            metric_history: Vec::new(),
            state_count_history: Vec::new(),
            query_count_history: Vec::new(),
            change_moving_avg: VecDeque::new(),
            stall_counter: 0,
            status: ConvergenceStatus::Undetermined,
            drift_detector: DriftDetector::new(drift_window, drift_threshold),
            iteration: 0,
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(ConvergenceConfig::default())
    }

    pub fn status(&self) -> &ConvergenceStatus {
        &self.status
    }

    pub fn iteration(&self) -> usize {
        self.iteration
    }

    pub fn metric_history(&self) -> &[f64] {
        &self.metric_history
    }

    pub fn state_count_history(&self) -> &[usize] {
        &self.state_count_history
    }

    /// Record a new iteration with its associated metrics.
    pub fn record_iteration(
        &mut self,
        metric: f64,
        state_count: usize,
        queries_used: usize,
    ) {
        self.iteration += 1;
        self.metric_history.push(metric);
        self.state_count_history.push(state_count);
        self.query_count_history.push(queries_used);

        // Compute change from previous iteration
        let change = if self.metric_history.len() >= 2 {
            let prev = self.metric_history[self.metric_history.len() - 2];
            (metric - prev).abs()
        } else {
            f64::INFINITY
        };

        // Update moving average
        self.change_moving_avg.push_back(change);
        if self.change_moving_avg.len() > self.config.window_size {
            self.change_moving_avg.pop_front();
        }

        // Update stall counter
        if change < self.config.stall_threshold {
            self.stall_counter += 1;
        } else {
            self.stall_counter = 0;
        }

        // Update drift detector
        if self.config.detect_drift {
            self.drift_detector.add_sample(metric);
        }

        // Determine status
        self.update_status();
    }

    fn update_status(&mut self) {
        if self.iteration < self.config.min_iterations {
            self.status = ConvergenceStatus::Undetermined;
            return;
        }

        // Check for drift
        if self.config.detect_drift && self.drift_detector.is_drifting() {
            self.status = ConvergenceStatus::Drifting;
            return;
        }

        // Check for convergence: moving average of changes below threshold
        let avg_change = self.average_change();
        if avg_change < self.config.convergence_threshold {
            self.status = ConvergenceStatus::Converged;
            return;
        }

        // Check for stall
        if self.stall_counter >= self.config.stall_patience {
            self.status = ConvergenceStatus::Stalled;
            return;
        }

        self.status = ConvergenceStatus::Progressing;
    }

    /// Average change over the moving window.
    pub fn average_change(&self) -> f64 {
        if self.change_moving_avg.is_empty() {
            return f64::INFINITY;
        }
        let sum: f64 = self.change_moving_avg.iter().sum();
        sum / self.change_moving_avg.len() as f64
    }

    /// Estimate the convergence rate (exponential decay constant).
    ///
    /// Fits an exponential decay model: metric(t) ≈ A * exp(-λt) + B
    /// and returns λ.
    pub fn convergence_rate(&self) -> Option<f64> {
        if self.metric_history.len() < 3 {
            return None;
        }

        // Use log-linear regression on the differences
        let n = self.metric_history.len();
        let baseline = self.metric_history.last().copied().unwrap_or(0.0);

        let mut sum_t = 0.0f64;
        let mut sum_y = 0.0f64;
        let mut sum_tt = 0.0f64;
        let mut sum_ty = 0.0f64;
        let mut count = 0.0f64;

        for (i, &val) in self.metric_history.iter().enumerate() {
            let diff = val - baseline;
            if diff > 1e-10 {
                let t = i as f64;
                let y = diff.ln();
                sum_t += t;
                sum_y += y;
                sum_tt += t * t;
                sum_ty += t * y;
                count += 1.0;
            }
        }

        if count < 2.0 {
            return None;
        }

        // Linear regression: y = a + bt, where b = -λ
        let b = (count * sum_ty - sum_t * sum_y)
            / (count * sum_tt - sum_t * sum_t);

        if b.is_finite() {
            Some(-b) // λ = -b
        } else {
            None
        }
    }

    /// Estimate number of iterations until convergence.
    pub fn estimated_iterations_remaining(&self) -> Option<usize> {
        let rate = self.convergence_rate()?;
        if rate <= 0.0 {
            return None;
        }

        let current_metric = self.metric_history.last().copied()?;
        if current_metric <= self.config.convergence_threshold {
            return Some(0);
        }

        // Time for A*exp(-λt) to reach threshold
        let t = (current_metric / self.config.convergence_threshold).ln() / rate;
        if t.is_finite() && t > 0.0 {
            Some(t.ceil() as usize)
        } else {
            None
        }
    }

    /// Compute total queries used so far.
    pub fn total_queries(&self) -> usize {
        self.query_count_history.iter().sum()
    }

    /// Average queries per iteration.
    pub fn average_queries_per_iteration(&self) -> f64 {
        if self.query_count_history.is_empty() {
            return 0.0;
        }
        self.total_queries() as f64 / self.query_count_history.len() as f64
    }

    /// Should we stop early?
    pub fn should_stop_early(&self, budget_remaining: usize) -> bool {
        match self.status {
            ConvergenceStatus::Converged => true,
            ConvergenceStatus::Stalled => true,
            ConvergenceStatus::BudgetExhausted => true,
            _ => {
                // Estimate if remaining budget is sufficient
                let avg_per_iter = self.average_queries_per_iteration();
                if avg_per_iter > 0.0 {
                    let remaining_iters = budget_remaining as f64 / avg_per_iter;
                    if let Some(needed) = self.estimated_iterations_remaining() {
                        remaining_iters < needed as f64 * 0.5
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
        }
    }

    /// Mark budget as exhausted.
    pub fn mark_budget_exhausted(&mut self) {
        self.status = ConvergenceStatus::BudgetExhausted;
    }

    /// Generate a convergence summary.
    pub fn summary(&self) -> ConvergenceSummary {
        ConvergenceSummary {
            status: self.status.clone(),
            iterations: self.iteration,
            total_queries: self.total_queries(),
            final_metric: self.metric_history.last().copied().unwrap_or(f64::NAN),
            convergence_rate: self.convergence_rate(),
            average_change: self.average_change(),
            estimated_remaining: self.estimated_iterations_remaining(),
            state_count: self.state_count_history.last().copied().unwrap_or(0),
        }
    }
}

/// Summary of convergence analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceSummary {
    pub status: ConvergenceStatus,
    pub iterations: usize,
    pub total_queries: usize,
    pub final_metric: f64,
    pub convergence_rate: Option<f64>,
    pub average_change: f64,
    pub estimated_remaining: Option<usize>,
    pub state_count: usize,
}

impl fmt::Display for ConvergenceSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Convergence[{}, iter={}, queries={}, metric={:.6}, rate={}, states={}]",
            self.status,
            self.iterations,
            self.total_queries,
            self.final_metric,
            self.convergence_rate
                .map(|r| format!("{:.4}", r))
                .unwrap_or_else(|| "?".to_string()),
            self.state_count
        )
    }
}

// ---------------------------------------------------------------------------
// Drift detector (CUSUM-based)
// ---------------------------------------------------------------------------

/// Detects non-stationarity in the target system using CUSUM.
///
/// The Cumulative Sum (CUSUM) algorithm detects changes in the mean
/// of a sequence of observations.
pub struct DriftDetector {
    /// Window of recent samples
    window: VecDeque<f64>,
    /// Maximum window size
    max_window: usize,
    /// Drift threshold
    threshold: f64,
    /// Running CUSUM+ and CUSUM-
    cusum_pos: f64,
    cusum_neg: f64,
    /// Running mean
    running_mean: f64,
    /// Sample count
    sample_count: usize,
    /// Has drift been detected?
    drift_detected: bool,
    /// Drift detection history
    drift_points: Vec<usize>,
}

impl DriftDetector {
    pub fn new(window_size: usize, threshold: f64) -> Self {
        Self {
            window: VecDeque::with_capacity(window_size),
            max_window: window_size,
            threshold,
            cusum_pos: 0.0,
            cusum_neg: 0.0,
            running_mean: 0.0,
            sample_count: 0,
            drift_detected: false,
            drift_points: Vec::new(),
        }
    }

    /// Add a new sample and check for drift.
    pub fn add_sample(&mut self, value: f64) {
        self.sample_count += 1;

        // Update running mean
        if self.sample_count == 1 {
            self.running_mean = value;
        } else {
            self.running_mean += (value - self.running_mean) / self.sample_count as f64;
        }

        // Update window
        self.window.push_back(value);
        if self.window.len() > self.max_window {
            self.window.pop_front();
        }

        // CUSUM update
        let deviation = value - self.running_mean;
        self.cusum_pos = (self.cusum_pos + deviation).max(0.0);
        self.cusum_neg = (self.cusum_neg - deviation).max(0.0);

        // Check for drift
        if self.cusum_pos > self.threshold || self.cusum_neg > self.threshold {
            self.drift_detected = true;
            self.drift_points.push(self.sample_count);
            // Reset CUSUM after detection
            self.cusum_pos = 0.0;
            self.cusum_neg = 0.0;
        } else if self.window.len() >= self.max_window {
            // Also check if window mean differs significantly from running mean
            let window_mean: f64 =
                self.window.iter().sum::<f64>() / self.window.len() as f64;
            let diff = (window_mean - self.running_mean).abs();
            if diff > self.threshold {
                self.drift_detected = true;
                self.drift_points.push(self.sample_count);
            } else {
                self.drift_detected = false;
            }
        }
    }

    /// Whether drift has been detected recently.
    pub fn is_drifting(&self) -> bool {
        self.drift_detected
    }

    /// Points where drift was detected.
    pub fn drift_points(&self) -> &[usize] {
        &self.drift_points
    }

    /// Reset the detector.
    pub fn reset(&mut self) {
        self.window.clear();
        self.cusum_pos = 0.0;
        self.cusum_neg = 0.0;
        self.running_mean = 0.0;
        self.sample_count = 0;
        self.drift_detected = false;
        self.drift_points.clear();
    }

    /// Current window mean.
    pub fn window_mean(&self) -> f64 {
        if self.window.is_empty() {
            return 0.0;
        }
        self.window.iter().sum::<f64>() / self.window.len() as f64
    }

    /// Current window variance.
    pub fn window_variance(&self) -> f64 {
        if self.window.len() < 2 {
            return 0.0;
        }
        let mean = self.window_mean();
        let sum_sq: f64 = self.window.iter().map(|x| (x - mean).powi(2)).sum();
        sum_sq / (self.window.len() as f64 - 1.0)
    }
}

// ---------------------------------------------------------------------------
// Confidence intervals
// ---------------------------------------------------------------------------

/// A confidence interval for a parameter estimate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    /// Point estimate (center)
    pub estimate: f64,
    /// Lower bound
    pub lower: f64,
    /// Upper bound
    pub upper: f64,
    /// Confidence level (e.g., 0.95)
    pub confidence_level: f64,
    /// Sample size used
    pub sample_size: usize,
}

impl ConfidenceInterval {
    /// Construct a CI from mean and margin of error.
    pub fn from_mean_and_margin(mean: f64, margin: f64, confidence_level: f64, sample_size: usize) -> Self {
        Self {
            estimate: mean,
            lower: mean - margin,
            upper: mean + margin,
            confidence_level,
            sample_size,
        }
    }

    /// Normal approximation CI for a proportion.
    ///
    /// Given k successes in n trials, compute CI for p = k/n.
    pub fn for_proportion(successes: usize, trials: usize, confidence_level: f64) -> Self {
        let n = trials as f64;
        let p_hat = successes as f64 / n;

        let z = z_score_for_confidence(confidence_level);
        let margin = z * (p_hat * (1.0 - p_hat) / n).sqrt();

        Self {
            estimate: p_hat,
            lower: (p_hat - margin).max(0.0),
            upper: (p_hat + margin).min(1.0),
            confidence_level,
            sample_size: trials,
        }
    }

    /// CI for a mean based on sample mean and variance.
    pub fn for_mean(
        sample_mean: f64,
        sample_variance: f64,
        sample_size: usize,
        confidence_level: f64,
    ) -> Self {
        let z = z_score_for_confidence(confidence_level);
        let se = (sample_variance / sample_size as f64).sqrt();
        let margin = z * se;

        Self {
            estimate: sample_mean,
            lower: sample_mean - margin,
            upper: sample_mean + margin,
            confidence_level,
            sample_size,
        }
    }

    /// Hoeffding confidence interval for bounded random variables.
    ///
    /// If X_i ∈ [a, b], then P(|X̄ - μ| ≥ t) ≤ 2exp(-2nt²/(b-a)²)
    pub fn hoeffding(
        sample_mean: f64,
        sample_size: usize,
        range_lower: f64,
        range_upper: f64,
        confidence_level: f64,
    ) -> Self {
        let n = sample_size as f64;
        let range = range_upper - range_lower;
        let alpha = 1.0 - confidence_level;

        // t = range * sqrt(ln(2/α) / (2n))
        let t = range * ((2.0 / alpha).ln() / (2.0 * n)).sqrt();

        Self {
            estimate: sample_mean,
            lower: (sample_mean - t).max(range_lower),
            upper: (sample_mean + t).min(range_upper),
            confidence_level,
            sample_size,
        }
    }

    /// Width of the interval.
    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }

    /// Does the interval contain a given value?
    pub fn contains(&self, value: f64) -> bool {
        value >= self.lower && value <= self.upper
    }

    /// Intersection of two CIs.
    pub fn intersect(&self, other: &ConfidenceInterval) -> Option<ConfidenceInterval> {
        let lower = self.lower.max(other.lower);
        let upper = self.upper.min(other.upper);

        if lower <= upper {
            Some(ConfidenceInterval {
                estimate: (lower + upper) / 2.0,
                lower,
                upper,
                confidence_level: self.confidence_level.min(other.confidence_level),
                sample_size: self.sample_size + other.sample_size,
            })
        } else {
            None
        }
    }
}

impl fmt::Display for ConfidenceInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:.4} [{:.4}, {:.4}] ({}% CI, n={})",
            self.estimate,
            self.lower,
            self.upper,
            self.confidence_level * 100.0,
            self.sample_size
        )
    }
}

/// Get z-score for a given confidence level (two-sided).
fn z_score_for_confidence(confidence_level: f64) -> f64 {
    // Common z-scores
    if (confidence_level - 0.90).abs() < 0.001 {
        return 1.645;
    }
    if (confidence_level - 0.95).abs() < 0.001 {
        return 1.960;
    }
    if (confidence_level - 0.99).abs() < 0.001 {
        return 2.576;
    }
    if (confidence_level - 0.999).abs() < 0.001 {
        return 3.291;
    }

    // Approximation using inverse normal (Beasley-Springer-Moro algorithm)
    let alpha = 1.0 - confidence_level;
    let p = 1.0 - alpha / 2.0;

    // Rational approximation for probit function
    let t = if p > 0.5 {
        (-2.0 * (1.0 - p).ln()).sqrt()
    } else {
        (-2.0 * p.ln()).sqrt()
    };

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    if p > 0.5 {
        z
    } else {
        -z
    }
}

// ---------------------------------------------------------------------------
// Hoeffding and Chernoff bound utilities
// ---------------------------------------------------------------------------

/// Hoeffding's inequality bound.
///
/// P(|X̄ - μ| ≥ t) ≤ 2·exp(-2nt²)
/// for X_i ∈ [0,1].
pub fn hoeffding_bound(n: usize, t: f64) -> f64 {
    2.0 * (-2.0 * n as f64 * t * t).exp()
}

/// Inverse Hoeffding: given n samples and confidence 1-δ,
/// what deviation ε can we bound?
pub fn hoeffding_epsilon(n: usize, delta: f64) -> f64 {
    ((2.0 / delta).ln() / (2.0 * n as f64)).sqrt()
}

/// Number of samples needed for Hoeffding bound with given ε, δ.
pub fn hoeffding_samples(epsilon: f64, delta: f64) -> usize {
    if epsilon <= 0.0 || delta <= 0.0 {
        return usize::MAX;
    }
    ((2.0 / delta).ln() / (2.0 * epsilon * epsilon)).ceil() as usize
}

/// Multiplicative Chernoff bound.
///
/// P(X ≥ (1+δ)μ) ≤ exp(-μδ²/3) for δ ∈ [0,1].
pub fn chernoff_upper(mu: f64, delta_param: f64) -> f64 {
    (-mu * delta_param * delta_param / 3.0).exp()
}

/// Lower tail Chernoff bound.
///
/// P(X ≤ (1-δ)μ) ≤ exp(-μδ²/2) for δ ∈ [0,1].
pub fn chernoff_lower(mu: f64, delta_param: f64) -> f64 {
    (-mu * delta_param * delta_param / 2.0).exp()
}

/// Bernstein's inequality bound.
///
/// P(|X̄ - μ| ≥ t) ≤ 2·exp(-nt²/(2σ² + 2bt/3))
/// where b is the range of X_i.
pub fn bernstein_bound(n: usize, t: f64, variance: f64, range: f64) -> f64 {
    let exponent = -(n as f64 * t * t) / (2.0 * variance + 2.0 * range * t / 3.0);
    2.0 * exponent.exp()
}

// ---------------------------------------------------------------------------
// KL divergence bounds
// ---------------------------------------------------------------------------

/// KL divergence from p to q: D_KL(p || q) = Σ p(x) log(p(x)/q(x)).
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len());
    let mut kl = 0.0;
    for i in 0..p.len() {
        if p[i] > 1e-15 {
            let q_safe = q[i].max(1e-15);
            kl += p[i] * (p[i] / q_safe).ln();
        }
    }
    kl
}

/// Pinsker's inequality: TV(P, Q) ≤ sqrt(D_KL(P||Q) / 2).
pub fn pinsker_tv_bound(kl: f64) -> f64 {
    (kl / 2.0).sqrt()
}

/// Sanov's theorem bound: P(empirical distribution far from true) ≤ (n+1)^k * exp(-n * D_KL).
pub fn sanov_bound(n: usize, support_size: usize, kl_threshold: f64) -> f64 {
    let poly = (n as f64 + 1.0).powi(support_size as i32);
    let exp_term = (-(n as f64) * kl_threshold).exp();
    poly * exp_term
}

// ---------------------------------------------------------------------------
// Query complexity estimation
// ---------------------------------------------------------------------------

/// Estimate total query complexity for the learning task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryComplexityEstimate {
    /// Membership queries needed
    pub membership_queries: usize,
    /// Equivalence queries needed
    pub equivalence_queries: usize,
    /// Total queries
    pub total_queries: usize,
    /// Estimated wall-clock time (seconds) given queries per second
    pub estimated_time_seconds: f64,
}

impl QueryComplexityEstimate {
    /// Estimate query complexity for learning a PDFA.
    ///
    /// Membership queries: O(n² * k * m) where n=states, k=alphabet, m=samples per query
    /// Equivalence queries: O(n) iterations with O(n*k*m) random tests each
    pub fn estimate(
        estimated_states: usize,
        alphabet_size: usize,
        samples_per_query: usize,
        queries_per_second: f64,
    ) -> Self {
        let n = estimated_states;
        let k = alphabet_size;
        let m = samples_per_query;

        // Table has O(n) rows, O(n) columns, each cell needs m samples
        // Plus k extensions per row
        let membership = n * n * k * m;

        // O(n) refinement rounds, each with O(n*k*m) random tests
        let equivalence = n * n * k * m;

        let total = membership + equivalence;
        let time = total as f64 / queries_per_second.max(1.0);

        Self {
            membership_queries: membership,
            equivalence_queries: equivalence,
            total_queries: total,
            estimated_time_seconds: time,
        }
    }
}

impl fmt::Display for QueryComplexityEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QueryComplexity(MQ={}, EQ={}, total={}, time={:.1}s)",
            self.membership_queries,
            self.equivalence_queries,
            self.total_queries,
            self.estimated_time_seconds
        )
    }
}

// ---------------------------------------------------------------------------
// Advanced convergence analysis and early stopping
// ---------------------------------------------------------------------------

/// Multi-metric convergence tracker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiMetricTracker {
    /// Named metrics being tracked
    pub metrics: HashMap<String, Vec<(usize, f64)>>,
    /// Convergence thresholds per metric
    pub thresholds: HashMap<String, f64>,
    /// Window size for trend analysis
    pub window_size: usize,
}

impl MultiMetricTracker {
    pub fn new(window_size: usize) -> Self {
        Self {
            metrics: HashMap::new(),
            thresholds: HashMap::new(),
            window_size,
        }
    }

    pub fn add_metric(&mut self, name: &str, threshold: f64) {
        self.metrics.entry(name.to_string()).or_default();
        self.thresholds.insert(name.to_string(), threshold);
    }

    pub fn record(&mut self, name: &str, iteration: usize, value: f64) {
        self.metrics
            .entry(name.to_string())
            .or_default()
            .push((iteration, value));
    }

    /// Get the trend (slope) of a metric over the recent window.
    pub fn trend(&self, name: &str) -> Option<f64> {
        let values = self.metrics.get(name)?;
        if values.len() < 2 {
            return Some(0.0);
        }

        let start = values.len().saturating_sub(self.window_size);
        let window = &values[start..];
        if window.len() < 2 {
            return Some(0.0);
        }

        // Simple linear regression
        let n = window.len() as f64;
        let sum_x: f64 = window.iter().map(|(i, _)| *i as f64).sum();
        let sum_y: f64 = window.iter().map(|(_, v)| *v).sum();
        let sum_xy: f64 = window.iter().map(|(i, v)| *i as f64 * v).sum();
        let sum_x2: f64 = window.iter().map(|(i, _)| (*i as f64).powi(2)).sum();

        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator.abs() < 1e-15 {
            return Some(0.0);
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denominator;
        Some(slope)
    }

    /// Check if a specific metric has converged.
    pub fn is_converged(&self, name: &str) -> bool {
        let threshold = self.thresholds.get(name).copied().unwrap_or(0.01);
        match self.trend(name) {
            Some(slope) => slope.abs() < threshold,
            None => false,
        }
    }

    /// Check if ALL metrics have converged.
    pub fn all_converged(&self) -> bool {
        self.thresholds.keys().all(|name| self.is_converged(name))
    }

    /// Get the latest value of a metric.
    pub fn latest(&self, name: &str) -> Option<f64> {
        self.metrics.get(name)?.last().map(|(_, v)| *v)
    }

    /// Summary of all metrics.
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        for (name, values) in &self.metrics {
            let latest = values.last().map(|(_, v)| *v).unwrap_or(0.0);
            let trend = self.trend(name).unwrap_or(0.0);
            let converged = self.is_converged(name);
            parts.push(format!(
                "{}={:.4}(Δ{:.6}{})",
                name,
                latest,
                trend,
                if converged { "✓" } else { "✗" },
            ));
        }
        parts.join(", ")
    }
}

/// Bayesian convergence estimator using posterior probability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BayesianConvergenceEstimator {
    /// Prior probability of convergence
    pub prior: f64,
    /// Observations (iteration, did_change)
    pub observations: Vec<(usize, bool)>,
    /// Posterior probability of convergence
    pub posterior: f64,
    /// Likelihood of observing no change given convergence
    pub p_stable_given_converged: f64,
    /// Likelihood of observing no change given not converged
    pub p_stable_given_not_converged: f64,
}

impl BayesianConvergenceEstimator {
    pub fn new() -> Self {
        Self {
            prior: 0.1,
            observations: Vec::new(),
            posterior: 0.1,
            p_stable_given_converged: 0.95,
            p_stable_given_not_converged: 0.3,
        }
    }

    pub fn with_prior(prior: f64) -> Self {
        let mut est = Self::new();
        est.prior = prior.clamp(0.01, 0.99);
        est.posterior = est.prior;
        est
    }

    /// Observe whether the hypothesis changed in this iteration.
    pub fn observe(&mut self, iteration: usize, changed: bool) {
        self.observations.push((iteration, changed));
        self.update_posterior(!changed);
    }

    /// Update posterior using Bayes' rule.
    fn update_posterior(&mut self, stable: bool) {
        let p_obs_given_conv = if stable {
            self.p_stable_given_converged
        } else {
            1.0 - self.p_stable_given_converged
        };

        let p_obs_given_not_conv = if stable {
            self.p_stable_given_not_converged
        } else {
            1.0 - self.p_stable_given_not_converged
        };

        let p_obs = p_obs_given_conv * self.posterior
            + p_obs_given_not_conv * (1.0 - self.posterior);

        if p_obs > 1e-15 {
            self.posterior = (p_obs_given_conv * self.posterior) / p_obs;
        }

        self.posterior = self.posterior.clamp(1e-10, 1.0 - 1e-10);
    }

    /// Get the current posterior probability of convergence.
    pub fn convergence_probability(&self) -> f64 {
        self.posterior
    }

    /// Is convergence likely (posterior > threshold)?
    pub fn is_likely_converged(&self, threshold: f64) -> bool {
        self.posterior >= threshold
    }

    /// Expected number of additional iterations until convergence.
    pub fn expected_remaining_iterations(&self) -> f64 {
        if self.posterior >= 0.99 {
            return 0.0;
        }
        // Geometric distribution: E[X] = 1/p
        let p_change = 1.0 - self.p_stable_given_converged;
        if p_change < 1e-15 {
            return f64::INFINITY;
        }
        (1.0 / self.posterior).ln() / p_change.ln().abs()
    }

    pub fn num_observations(&self) -> usize {
        self.observations.len()
    }
}

impl Default for BayesianConvergenceEstimator {
    fn default() -> Self {
        Self::new()
    }
}

/// Sequential probability ratio test (SPRT) for convergence detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SPRTConvergenceTest {
    /// Type I error rate (false positive)
    pub alpha: f64,
    /// Type II error rate (false negative)
    pub beta: f64,
    /// Null hypothesis: change probability
    pub p0: f64,
    /// Alternative hypothesis: change probability
    pub p1: f64,
    /// Log-likelihood ratio
    pub log_ratio: f64,
    /// Decision boundaries
    pub lower_bound: f64,
    pub upper_bound: f64,
    /// Observations
    pub observations: Vec<bool>,
}

impl SPRTConvergenceTest {
    pub fn new(alpha: f64, beta: f64, p0: f64, p1: f64) -> Self {
        let lower = (beta / (1.0 - alpha)).ln();
        let upper = ((1.0 - beta) / alpha).ln();
        Self {
            alpha,
            beta,
            p0,
            p1,
            log_ratio: 0.0,
            lower_bound: lower,
            upper_bound: upper,
            observations: Vec::new(),
        }
    }

    pub fn default_convergence() -> Self {
        // H0: 50% chance of change (not converged)
        // H1: 5% chance of change (converged)
        Self::new(0.05, 0.05, 0.5, 0.05)
    }

    /// Add an observation (did the hypothesis change?).
    pub fn observe(&mut self, changed: bool) {
        self.observations.push(changed);

        let lr = if changed {
            (self.p1 / self.p0).ln()
        } else {
            ((1.0 - self.p1) / (1.0 - self.p0)).ln()
        };
        self.log_ratio += lr;
    }

    /// Get the current decision.
    pub fn decision(&self) -> SPRTDecision {
        if self.log_ratio <= self.lower_bound {
            SPRTDecision::AcceptH0 // Not converged
        } else if self.log_ratio >= self.upper_bound {
            SPRTDecision::AcceptH1 // Converged
        } else {
            SPRTDecision::Continue
        }
    }

    /// Is convergence confirmed?
    pub fn is_converged(&self) -> bool {
        matches!(self.decision(), SPRTDecision::AcceptH1)
    }

    /// Is non-convergence confirmed?
    pub fn is_not_converged(&self) -> bool {
        matches!(self.decision(), SPRTDecision::AcceptH0)
    }

    pub fn num_observations(&self) -> usize {
        self.observations.len()
    }
}

/// Decision from the SPRT test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SPRTDecision {
    AcceptH0,
    AcceptH1,
    Continue,
}

impl fmt::Display for SPRTDecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SPRTDecision::AcceptH0 => write!(f, "Not Converged"),
            SPRTDecision::AcceptH1 => write!(f, "Converged"),
            SPRTDecision::Continue => write!(f, "Continue"),
        }
    }
}

/// Exponentially weighted moving average for drift detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EWMADriftDetector {
    /// Smoothing factor (0 < λ ≤ 1)
    pub lambda: f64,
    /// Current EWMA value
    pub ewma: f64,
    /// Current EWMA variance
    pub ewma_var: f64,
    /// Number of observations
    pub count: usize,
    /// Threshold multiplier for drift detection (number of std deviations)
    pub threshold_sigma: f64,
    /// Initial mean estimate
    pub mu0: f64,
    /// History of EWMA values
    pub history: Vec<f64>,
}

impl EWMADriftDetector {
    pub fn new(lambda: f64, threshold_sigma: f64) -> Self {
        Self {
            lambda: lambda.clamp(0.01, 1.0),
            ewma: 0.0,
            ewma_var: 0.0,
            count: 0,
            threshold_sigma,
            mu0: 0.0,
            history: Vec::new(),
        }
    }

    pub fn default_detector() -> Self {
        Self::new(0.2, 3.0)
    }

    pub fn observe(&mut self, value: f64) {
        self.count += 1;
        if self.count == 1 {
            self.ewma = value;
            self.mu0 = value;
            self.ewma_var = 0.0;
        } else {
            self.ewma = self.lambda * value + (1.0 - self.lambda) * self.ewma;
            let deviation = value - self.mu0;
            self.ewma_var = self.lambda * deviation * deviation
                + (1.0 - self.lambda) * self.ewma_var;
        }
        self.history.push(self.ewma);
    }

    /// Check if drift is detected.
    pub fn drift_detected(&self) -> bool {
        if self.count < 5 {
            return false;
        }
        let sigma = self.ewma_var.sqrt();
        let control_limit = self.mu0 + self.threshold_sigma * sigma;
        let lower_limit = self.mu0 - self.threshold_sigma * sigma;
        self.ewma > control_limit || self.ewma < lower_limit
    }

    /// Get the current control limits.
    pub fn control_limits(&self) -> (f64, f64) {
        let sigma = self.ewma_var.sqrt();
        (
            self.mu0 - self.threshold_sigma * sigma,
            self.mu0 + self.threshold_sigma * sigma,
        )
    }

    pub fn current_value(&self) -> f64 {
        self.ewma
    }

    pub fn reset(&mut self) {
        self.count = 0;
        self.ewma = 0.0;
        self.ewma_var = 0.0;
        self.mu0 = 0.0;
        self.history.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pac_bounds_basic() {
        let bounds = PACBounds::compute(0.1, 0.05, 10, 4);
        assert!(bounds.sample_size > 0);
        assert_eq!(bounds.epsilon, 0.1);
        assert_eq!(bounds.delta, 0.05);

        assert!(!bounds.is_sufficient(0));
        assert!(bounds.is_sufficient(bounds.sample_size));
    }

    #[test]
    fn test_pac_bounds_monotonicity() {
        // Tighter epsilon → more samples
        let b1 = PACBounds::compute(0.1, 0.05, 10, 4);
        let b2 = PACBounds::compute(0.01, 0.05, 10, 4);
        assert!(b2.sample_size > b1.sample_size);

        // Tighter delta → more samples
        let b3 = PACBounds::compute(0.1, 0.01, 10, 4);
        assert!(b3.sample_size > b1.sample_size);
    }

    #[test]
    fn test_pac_epsilon_from_samples() {
        let eps = PACBounds::epsilon_from_samples(10000, 0.05, 10, 4);
        assert!(eps > 0.0);
        assert!(eps < 1.0);

        // More samples → smaller epsilon
        let eps2 = PACBounds::epsilon_from_samples(100000, 0.05, 10, 4);
        assert!(eps2 < eps);
    }

    #[test]
    fn test_pac_delta_from_samples() {
        let delta = PACBounds::delta_from_samples(10000, 0.1, 10, 4);
        assert!(delta >= 0.0);
        assert!(delta <= 1.0);
    }

    #[test]
    fn test_pac_progress() {
        let bounds = PACBounds::compute(0.1, 0.05, 10, 4);
        assert_eq!(bounds.progress(0), 0.0);
        assert!((bounds.progress(bounds.sample_size) - 1.0).abs() < 0.001);
        assert!(bounds.progress(bounds.sample_size / 2) < 1.0);
    }

    #[test]
    fn test_sample_complexity_hoeffding() {
        let n = SampleComplexity::hoeffding_bound(0.1, 0.05, 10);
        assert!(n > 0);

        // Tighter bounds need more samples
        let n2 = SampleComplexity::hoeffding_bound(0.01, 0.05, 10);
        assert!(n2 > n);
    }

    #[test]
    fn test_sample_complexity_chernoff() {
        let n = SampleComplexity::chernoff_bound(0.1, 0.05);
        assert!(n > 0);
    }

    #[test]
    fn test_sample_complexity_for_learning() {
        let sc = SampleComplexity::for_learning(0.1, 0.05, 10, 4);
        assert!(sc.total > 0);
        assert!(sc.distribution_estimation > 0);
        assert!(sc.hypothesis_testing > 0);
        assert!(sc.equivalence_testing > 0);
    }

    #[test]
    fn test_convergence_undetermined() {
        let mut analyzer = ConvergenceAnalyzer::with_default_config();
        assert_eq!(*analyzer.status(), ConvergenceStatus::Undetermined);

        analyzer.record_iteration(1.0, 5, 100);
        assert_eq!(*analyzer.status(), ConvergenceStatus::Undetermined);
    }

    #[test]
    fn test_convergence_detected() {
        let config = ConvergenceConfig {
            min_iterations: 3,
            convergence_threshold: 0.01,
            window_size: 5,
            ..Default::default()
        };
        let mut analyzer = ConvergenceAnalyzer::new(config);

        // Record converging metrics
        for i in 0..10 {
            analyzer.record_iteration(1.0 / (i as f64 + 1.0), 5, 50);
        }
        // Record stable metrics
        for _ in 0..10 {
            analyzer.record_iteration(0.001, 5, 10);
        }

        assert_eq!(*analyzer.status(), ConvergenceStatus::Converged);
    }

    #[test]
    fn test_convergence_stall() {
        let config = ConvergenceConfig {
            min_iterations: 3,
            stall_threshold: 0.01,
            stall_patience: 5,
            convergence_threshold: 0.001,
            ..Default::default()
        };
        let mut analyzer = ConvergenceAnalyzer::new(config);

        // Record constant but non-converged metrics
        for _ in 0..20 {
            analyzer.record_iteration(0.5, 5, 50);
        }

        assert_eq!(*analyzer.status(), ConvergenceStatus::Stalled);
    }

    #[test]
    fn test_convergence_rate() {
        let mut analyzer = ConvergenceAnalyzer::with_default_config();

        // Exponentially decaying metrics
        for i in 0..20 {
            let metric = 10.0 * (-0.3 * i as f64).exp();
            analyzer.record_iteration(metric, 5, 50);
        }

        let rate = analyzer.convergence_rate();
        assert!(rate.is_some());
        let r = rate.unwrap();
        // Should be approximately 0.3
        assert!(r > 0.0, "Rate should be positive, got {}", r);
    }

    #[test]
    fn test_convergence_summary() {
        let mut analyzer = ConvergenceAnalyzer::with_default_config();
        for i in 0..5 {
            analyzer.record_iteration(1.0 / (i as f64 + 1.0), i + 1, 100);
        }

        let summary = analyzer.summary();
        assert_eq!(summary.iterations, 5);
        assert_eq!(summary.total_queries, 500);
        assert_eq!(summary.state_count, 5);
    }

    #[test]
    fn test_drift_detector_stable() {
        let mut detector = DriftDetector::new(10, 0.5);
        for _ in 0..20 {
            detector.add_sample(1.0);
        }
        assert!(!detector.is_drifting());
    }

    #[test]
    fn test_drift_detector_shift() {
        let mut detector = DriftDetector::new(10, 0.1);

        // Stable phase
        for _ in 0..20 {
            detector.add_sample(0.5);
        }

        // Sudden shift
        for _ in 0..20 {
            detector.add_sample(5.0);
        }

        // Should detect drift at some point
        assert!(!detector.drift_points().is_empty());
    }

    #[test]
    fn test_drift_detector_reset() {
        let mut detector = DriftDetector::new(10, 0.1);
        detector.add_sample(1.0);
        detector.add_sample(100.0);
        detector.reset();

        assert_eq!(detector.sample_count, 0);
        assert!(!detector.is_drifting());
        assert!(detector.drift_points().is_empty());
    }

    #[test]
    fn test_drift_detector_variance() {
        let mut detector = DriftDetector::new(10, 0.5);
        for i in 0..10 {
            detector.add_sample(i as f64);
        }

        let var = detector.window_variance();
        assert!(var > 0.0);
    }

    #[test]
    fn test_confidence_interval_proportion() {
        let ci = ConfidenceInterval::for_proportion(700, 1000, 0.95);
        assert!((ci.estimate - 0.7).abs() < 0.001);
        assert!(ci.lower < 0.7);
        assert!(ci.upper > 0.7);
        assert!(ci.contains(0.7));
        assert!(ci.width() > 0.0);
    }

    #[test]
    fn test_confidence_interval_mean() {
        let ci = ConfidenceInterval::for_mean(5.0, 2.0, 100, 0.95);
        assert!((ci.estimate - 5.0).abs() < 0.001);
        assert!(ci.lower < 5.0);
        assert!(ci.upper > 5.0);
    }

    #[test]
    fn test_confidence_interval_hoeffding() {
        let ci = ConfidenceInterval::hoeffding(0.5, 100, 0.0, 1.0, 0.95);
        assert!(ci.lower >= 0.0);
        assert!(ci.upper <= 1.0);
        assert!(ci.contains(0.5));
    }

    #[test]
    fn test_confidence_interval_intersection() {
        let ci1 = ConfidenceInterval::from_mean_and_margin(5.0, 1.0, 0.95, 100);
        let ci2 = ConfidenceInterval::from_mean_and_margin(5.5, 1.0, 0.95, 100);

        let inter = ci1.intersect(&ci2);
        assert!(inter.is_some());
        let ci = inter.unwrap();
        assert!(ci.lower >= 4.5);
        assert!(ci.upper <= 6.0);

        // Non-overlapping
        let ci3 = ConfidenceInterval::from_mean_and_margin(10.0, 0.5, 0.95, 100);
        assert!(ci1.intersect(&ci3).is_none());
    }

    #[test]
    fn test_hoeffding_bound_fn() {
        let p = hoeffding_bound(100, 0.1);
        assert!(p > 0.0);
        assert!(p < 1.0);

        // More samples → tighter bound
        let p2 = hoeffding_bound(1000, 0.1);
        assert!(p2 < p);
    }

    #[test]
    fn test_hoeffding_epsilon_fn() {
        let eps = hoeffding_epsilon(1000, 0.05);
        assert!(eps > 0.0);
        assert!(eps < 1.0);

        // More samples → smaller epsilon
        let eps2 = hoeffding_epsilon(10000, 0.05);
        assert!(eps2 < eps);
    }

    #[test]
    fn test_hoeffding_samples_fn() {
        let n = hoeffding_samples(0.1, 0.05);
        assert!(n > 0);

        // Tighter bounds → more samples
        let n2 = hoeffding_samples(0.01, 0.05);
        assert!(n2 > n);
    }

    #[test]
    fn test_chernoff_bounds() {
        let p_upper = chernoff_upper(10.0, 0.5);
        assert!(p_upper > 0.0);
        assert!(p_upper < 1.0);

        let p_lower = chernoff_lower(10.0, 0.5);
        assert!(p_lower > 0.0);
        assert!(p_lower < 1.0);
    }

    #[test]
    fn test_bernstein_bound_fn() {
        let p = bernstein_bound(100, 0.1, 0.25, 1.0);
        assert!(p > 0.0);
    }

    #[test]
    fn test_kl_divergence_identical() {
        let p = vec![0.3, 0.7];
        let kl = kl_divergence(&p, &p);
        assert!(kl.abs() < 1e-10);
    }

    #[test]
    fn test_kl_divergence_different() {
        let p = vec![0.5, 0.5];
        let q = vec![0.1, 0.9];
        let kl = kl_divergence(&p, &q);
        assert!(kl > 0.0);
    }

    #[test]
    fn test_pinsker_bound() {
        let kl = 0.5;
        let tv_bound = pinsker_tv_bound(kl);
        assert!(tv_bound > 0.0);
        assert!(tv_bound <= 1.0);
    }

    #[test]
    fn test_sanov_bound_fn() {
        let p = sanov_bound(100, 5, 0.1);
        assert!(p > 0.0);
    }

    #[test]
    fn test_query_complexity_estimate() {
        let est = QueryComplexityEstimate::estimate(10, 4, 100, 1000.0);
        assert!(est.total_queries > 0);
        assert!(est.estimated_time_seconds > 0.0);
        assert_eq!(
            est.total_queries,
            est.membership_queries + est.equivalence_queries
        );
    }

    #[test]
    fn test_z_score_common_values() {
        let z90 = z_score_for_confidence(0.90);
        assert!((z90 - 1.645).abs() < 0.001);

        let z95 = z_score_for_confidence(0.95);
        assert!((z95 - 1.960).abs() < 0.001);

        let z99 = z_score_for_confidence(0.99);
        assert!((z99 - 2.576).abs() < 0.001);
    }

    #[test]
    fn test_convergence_should_stop_early() {
        let mut analyzer = ConvergenceAnalyzer::with_default_config();

        // Initially should not stop
        assert!(!analyzer.should_stop_early(10000));

        // After convergence, should stop
        for _ in 0..20 {
            analyzer.record_iteration(0.001, 5, 10);
        }
        // If converged, should stop
        if *analyzer.status() == ConvergenceStatus::Converged {
            assert!(analyzer.should_stop_early(10000));
        }
    }

    #[test]
    fn test_convergence_budget_exhausted() {
        let mut analyzer = ConvergenceAnalyzer::with_default_config();
        analyzer.mark_budget_exhausted();
        assert_eq!(*analyzer.status(), ConvergenceStatus::BudgetExhausted);
    }

    #[test]
    fn test_convergence_average_queries() {
        let mut analyzer = ConvergenceAnalyzer::with_default_config();
        analyzer.record_iteration(1.0, 5, 100);
        analyzer.record_iteration(0.5, 5, 200);
        analyzer.record_iteration(0.25, 5, 300);

        assert_eq!(analyzer.total_queries(), 600);
        assert!((analyzer.average_queries_per_iteration() - 200.0).abs() < 0.001);
    }

    #[test]
    fn test_multi_metric_tracker_basic() {
        let mut tracker = MultiMetricTracker::new(5);
        tracker.add_metric("states", 0.01);
        tracker.add_metric("tolerance", 0.001);

        for i in 0..10 {
            tracker.record("states", i, 5.0);
            tracker.record("tolerance", i, 0.1 - 0.001 * i as f64);
        }

        // States should converge (constant value)
        assert!(tracker.is_converged("states"));
        let latest = tracker.latest("states");
        assert_eq!(latest, Some(5.0));
    }

    #[test]
    fn test_multi_metric_tracker_trend() {
        let mut tracker = MultiMetricTracker::new(5);
        tracker.add_metric("value", 0.1);

        // Linear increase
        for i in 0..10 {
            tracker.record("value", i, i as f64);
        }

        let trend = tracker.trend("value").unwrap();
        assert!(trend > 0.5); // Should be approximately 1.0
    }

    #[test]
    fn test_multi_metric_tracker_all_converged() {
        let mut tracker = MultiMetricTracker::new(5);
        tracker.add_metric("a", 0.01);
        tracker.add_metric("b", 0.01);

        for i in 0..10 {
            tracker.record("a", i, 1.0);
            tracker.record("b", i, 2.0);
        }

        assert!(tracker.all_converged());
    }

    #[test]
    fn test_multi_metric_tracker_summary() {
        let mut tracker = MultiMetricTracker::new(5);
        tracker.add_metric("test", 0.01);
        tracker.record("test", 0, 1.0);
        let summary = tracker.summary();
        assert!(summary.contains("test="));
    }

    #[test]
    fn test_bayesian_convergence_estimator() {
        let mut est = BayesianConvergenceEstimator::new();
        assert!(est.convergence_probability() < 0.5);

        // Observe many stable iterations
        for i in 0..20 {
            est.observe(i, false); // no change
        }

        assert!(est.convergence_probability() > 0.5);
        assert!(est.is_likely_converged(0.5));
    }

    #[test]
    fn test_bayesian_convergence_with_changes() {
        let mut est = BayesianConvergenceEstimator::new();

        // Many changes → not converged
        for i in 0..10 {
            est.observe(i, true);
        }

        assert!(est.convergence_probability() < 0.5);
    }

    #[test]
    fn test_bayesian_convergence_expected_remaining() {
        let mut est = BayesianConvergenceEstimator::new();
        for i in 0..30 {
            est.observe(i, false);
        }

        let remaining = est.expected_remaining_iterations();
        assert!(remaining >= 0.0);
    }

    #[test]
    fn test_sprt_convergence_test() {
        let mut test = SPRTConvergenceTest::default_convergence();

        // Many stable observations should lead to convergence
        for _ in 0..50 {
            test.observe(false); // no change
        }

        assert!(test.is_converged());
        assert!(!test.is_not_converged());
    }

    #[test]
    fn test_sprt_not_converged() {
        let mut test = SPRTConvergenceTest::default_convergence();

        // Many changes should lead to non-convergence
        for _ in 0..20 {
            test.observe(true); // changed
        }

        assert!(test.is_not_converged() || matches!(test.decision(), SPRTDecision::Continue));
    }

    #[test]
    fn test_sprt_decision_display() {
        assert_eq!(format!("{}", SPRTDecision::Continue), "Continue");
        assert_eq!(format!("{}", SPRTDecision::AcceptH1), "Converged");
    }

    #[test]
    fn test_ewma_drift_detector_no_drift() {
        let mut detector = EWMADriftDetector::default_detector();

        // Stable observations around 0
        for i in 0..20 {
            detector.observe(0.1 * (i % 2) as f64);
        }

        assert!(!detector.drift_detected());
    }

    #[test]
    fn test_ewma_drift_detector_with_drift() {
        let mut detector = EWMADriftDetector::new(0.3, 2.0);

        // Start stable
        for _ in 0..10 {
            detector.observe(0.0);
        }

        // Sudden shift
        for _ in 0..20 {
            detector.observe(10.0);
        }

        assert!(detector.drift_detected());
    }

    #[test]
    fn test_ewma_control_limits() {
        let mut detector = EWMADriftDetector::new(0.2, 3.0);
        for _ in 0..10 {
            detector.observe(1.0);
        }

        let (lower, upper) = detector.control_limits();
        assert!(lower <= 1.0);
        assert!(upper >= 1.0);
    }

    #[test]
    fn test_ewma_reset() {
        let mut detector = EWMADriftDetector::default_detector();
        detector.observe(1.0);
        detector.observe(2.0);
        assert_eq!(detector.count, 2);

        detector.reset();
        assert_eq!(detector.count, 0);
        assert!(detector.history.is_empty());
    }
}

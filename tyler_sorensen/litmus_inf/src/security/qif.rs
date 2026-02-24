//! Quantitative Information Flow analysis for GPU memory model security.
//!
//! Implements channel capacity computation, entropy measures, and leakage
//! quantification for side-channel analysis of GPU kernels.

use std::collections::HashMap;
use std::fmt;
use serde::{Serialize, Deserialize};

// ---------------------------------------------------------------------------
// ShannonEntropy
// ---------------------------------------------------------------------------

/// Shannon entropy computation.
#[derive(Debug, Clone)]
pub struct ShannonEntropy;

impl ShannonEntropy {
    /// Compute Shannon entropy H(X) = -Σ p(x) log₂ p(x).
    pub fn compute(distribution: &[f64]) -> f64 {
        let mut h = 0.0;
        for &p in distribution {
            if p > 0.0 {
                h -= p * p.log2();
            }
        }
        h
    }

    /// Compute conditional entropy H(X|Y) from joint distribution P(X,Y).
    pub fn conditional(joint: &[Vec<f64>]) -> f64 {
        let rows = joint.len();
        if rows == 0 { return 0.0; }
        let cols = joint[0].len();

        // P(Y=j) = Σ_i P(X=i, Y=j)
        let mut py = vec![0.0; cols];
        for j in 0..cols {
            for i in 0..rows {
                py[j] += joint[i][j];
            }
        }

        // H(X|Y) = -Σ_{i,j} P(X=i,Y=j) log₂ P(X=i|Y=j)
        let mut h = 0.0;
        for i in 0..rows {
            for j in 0..cols {
                let pxy = joint[i][j];
                if pxy > 0.0 && py[j] > 0.0 {
                    let px_given_y = pxy / py[j];
                    h -= pxy * px_given_y.log2();
                }
            }
        }
        h
    }

    /// Compute joint entropy H(X,Y) from joint distribution.
    pub fn joint(joint: &[Vec<f64>]) -> f64 {
        let mut h = 0.0;
        for row in joint {
            for &p in row {
                if p > 0.0 {
                    h -= p * p.log2();
                }
            }
        }
        h
    }

    /// Maximum entropy for N outcomes: log₂(N).
    pub fn max_entropy(n: usize) -> f64 {
        if n <= 1 { return 0.0; }
        (n as f64).log2()
    }
}

// ---------------------------------------------------------------------------
// MinEntropy
// ---------------------------------------------------------------------------

/// Min-entropy computation (worst-case vulnerability measure).
#[derive(Debug, Clone)]
pub struct MinEntropy;

impl MinEntropy {
    /// Compute min-entropy H∞(X) = -log₂ max p(x).
    pub fn compute(distribution: &[f64]) -> f64 {
        let max_p = distribution.iter().cloned().fold(0.0f64, f64::max);
        if max_p <= 0.0 { return 0.0; }
        -max_p.log2()
    }

    /// Conditional min-entropy H∞(X|Y) from channel matrix.
    pub fn conditional(channel: &ChannelMatrix) -> f64 {
        let rows = channel.rows;
        let cols = channel.cols;
        if rows == 0 || cols == 0 { return 0.0; }

        // P(Y=j) with prior
        let mut py = vec![0.0; cols];
        for j in 0..cols {
            for i in 0..rows {
                py[j] += channel.prior[i] * channel.matrix[i][j];
            }
        }

        // Conditional min-entropy: -log₂ Σ_j max_i P(X=i|Y=j) · P(Y=j)
        let mut sum = 0.0;
        for j in 0..cols {
            if py[j] > 0.0 {
                let max_post = (0..rows)
                    .map(|i| channel.prior[i] * channel.matrix[i][j] / py[j])
                    .fold(0.0f64, f64::max);
                sum += py[j] * max_post;
            }
        }

        if sum <= 0.0 { return 0.0; }
        -sum.log2()
    }

    /// Vulnerability: V(X) = 2^{-H∞(X)} = max p(x).
    pub fn vulnerability(distribution: &[f64]) -> f64 {
        distribution.iter().cloned().fold(0.0f64, f64::max)
    }
}

// ---------------------------------------------------------------------------
// MutualInformation
// ---------------------------------------------------------------------------

/// Mutual information I(X;Y) = H(X) - H(X|Y).
#[derive(Debug, Clone)]
pub struct MutualInformation;

impl MutualInformation {
    /// Compute I(X;Y) from joint distribution P(X,Y).
    pub fn from_joint(joint: &[Vec<f64>]) -> f64 {
        let rows = joint.len();
        if rows == 0 { return 0.0; }
        let cols = joint[0].len();

        // Marginals
        let mut px = vec![0.0; rows];
        let mut py = vec![0.0; cols];
        for i in 0..rows {
            for j in 0..cols {
                px[i] += joint[i][j];
                py[j] += joint[i][j];
            }
        }

        let mut mi = 0.0;
        for i in 0..rows {
            for j in 0..cols {
                let pxy = joint[i][j];
                if pxy > 0.0 && px[i] > 0.0 && py[j] > 0.0 {
                    mi += pxy * (pxy / (px[i] * py[j])).log2();
                }
            }
        }
        mi
    }

    /// Compute I(X;Y) from a channel matrix and prior.
    pub fn from_channel(channel: &ChannelMatrix) -> f64 {
        let joint = channel.to_joint();
        Self::from_joint(&joint)
    }

    /// Normalized mutual information I(X;Y) / H(X).
    pub fn normalized(joint: &[Vec<f64>]) -> f64 {
        let rows = joint.len();
        if rows == 0 { return 0.0; }
        let mut px = vec![0.0; rows];
        for i in 0..rows {
            for j in 0..joint[0].len() {
                px[i] += joint[i][j];
            }
        }
        let hx = ShannonEntropy::compute(&px);
        if hx <= 0.0 { return 0.0; }
        Self::from_joint(joint) / hx
    }
}

// ---------------------------------------------------------------------------
// ChannelMatrix
// ---------------------------------------------------------------------------

/// A channel matrix P(Y|X) describing the information-theoretic channel
/// between secret inputs X and observable outputs Y.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelMatrix {
    /// Number of secret values (rows).
    pub rows: usize,
    /// Number of observable outcomes (columns).
    pub cols: usize,
    /// matrix[i][j] = P(Y=j | X=i).
    pub matrix: Vec<Vec<f64>>,
    /// Prior distribution on secrets P(X).
    pub prior: Vec<f64>,
    /// Labels for secret values.
    pub secret_labels: Vec<String>,
    /// Labels for observable outcomes.
    pub observable_labels: Vec<String>,
}

impl ChannelMatrix {
    /// Create a channel matrix with uniform prior.
    pub fn new(matrix: Vec<Vec<f64>>) -> Self {
        let rows = matrix.len();
        let cols = if rows > 0 { matrix[0].len() } else { 0 };
        let prior = vec![1.0 / rows as f64; rows];
        ChannelMatrix {
            rows, cols, matrix, prior,
            secret_labels: (0..rows).map(|i| format!("s{}", i)).collect(),
            observable_labels: (0..cols).map(|j| format!("o{}", j)).collect(),
        }
    }

    /// Create with custom prior.
    pub fn with_prior(mut self, prior: Vec<f64>) -> Self {
        assert_eq!(prior.len(), self.rows);
        self.prior = prior;
        self
    }

    /// Create with custom labels.
    pub fn with_labels(mut self, secrets: Vec<String>, observables: Vec<String>) -> Self {
        self.secret_labels = secrets;
        self.observable_labels = observables;
        self
    }

    /// Build a deterministic channel where each secret maps to exactly one observable.
    pub fn deterministic(mapping: &[usize], num_observables: usize) -> Self {
        let rows = mapping.len();
        let mut matrix = vec![vec![0.0; num_observables]; rows];
        for (i, &obs) in mapping.iter().enumerate() {
            if obs < num_observables {
                matrix[i][obs] = 1.0;
            }
        }
        Self::new(matrix)
    }

    /// Build a noiseless (identity) channel.
    pub fn identity(n: usize) -> Self {
        let mut matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            matrix[i][i] = 1.0;
        }
        Self::new(matrix)
    }

    /// Build a completely noisy (uniform) channel.
    pub fn uniform(rows: usize, cols: usize) -> Self {
        let p = 1.0 / cols as f64;
        let matrix = vec![vec![p; cols]; rows];
        Self::new(matrix)
    }

    /// Build a binary symmetric channel with error probability ε.
    pub fn binary_symmetric(epsilon: f64) -> Self {
        let matrix = vec![
            vec![1.0 - epsilon, epsilon],
            vec![epsilon, 1.0 - epsilon],
        ];
        Self::new(matrix)
    }

    /// Convert to joint distribution P(X,Y) = P(X) · P(Y|X).
    pub fn to_joint(&self) -> Vec<Vec<f64>> {
        let mut joint = vec![vec![0.0; self.cols]; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                joint[i][j] = self.prior[i] * self.matrix[i][j];
            }
        }
        joint
    }

    /// Compute output distribution P(Y).
    pub fn output_distribution(&self) -> Vec<f64> {
        let mut py = vec![0.0; self.cols];
        for j in 0..self.cols {
            for i in 0..self.rows {
                py[j] += self.prior[i] * self.matrix[i][j];
            }
        }
        py
    }

    /// Posterior distribution P(X|Y=j).
    pub fn posterior(&self, j: usize) -> Vec<f64> {
        let py = self.output_distribution();
        if py[j] <= 0.0 { return vec![0.0; self.rows]; }
        (0..self.rows)
            .map(|i| self.prior[i] * self.matrix[i][j] / py[j])
            .collect()
    }

    /// Check that each row is a valid probability distribution.
    pub fn is_valid(&self) -> bool {
        for row in &self.matrix {
            let sum: f64 = row.iter().sum();
            if (sum - 1.0).abs() > 1e-9 { return false; }
            if row.iter().any(|&p| p < -1e-12) { return false; }
        }
        let prior_sum: f64 = self.prior.iter().sum();
        (prior_sum - 1.0).abs() <= 1e-9
    }

    /// Number of distinct rows (observational equivalence classes).
    pub fn num_equivalence_classes(&self) -> usize {
        let mut classes: Vec<Vec<f64>> = Vec::new();
        for row in &self.matrix {
            let found = classes.iter().any(|c| {
                c.len() == row.len() && c.iter().zip(row).all(|(a, b)| (a - b).abs() < 1e-12)
            });
            if !found { classes.push(row.clone()); }
        }
        classes.len()
    }

    /// Compose two channels: C1 ; C2 (cascade).
    pub fn compose(&self, other: &ChannelMatrix) -> ChannelMatrix {
        assert_eq!(self.cols, other.rows);
        let mut matrix = vec![vec![0.0; other.cols]; self.rows];
        for i in 0..self.rows {
            for k in 0..other.cols {
                for j in 0..self.cols {
                    matrix[i][k] += self.matrix[i][j] * other.matrix[j][k];
                }
            }
        }
        let mut result = ChannelMatrix::new(matrix);
        result.prior = self.prior.clone();
        result
    }

    /// Format as a text table.
    pub fn to_table(&self) -> String {
        let mut s = String::new();
        // Header
        s.push_str(&format!("{:>10}", ""));
        for label in &self.observable_labels {
            s.push_str(&format!(" {:>8}", label));
        }
        s.push('\n');
        // Rows
        for (i, row) in self.matrix.iter().enumerate() {
            s.push_str(&format!("{:>10}", &self.secret_labels[i]));
            for p in row {
                s.push_str(&format!(" {:>8.4}", p));
            }
            s.push('\n');
        }
        s
    }
}

impl fmt::Display for ChannelMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Channel({}×{}, {} classes)",
            self.rows, self.cols, self.num_equivalence_classes())
    }
}

// ---------------------------------------------------------------------------
// ChannelCapacity — Blahut-Arimoto algorithm
// ---------------------------------------------------------------------------

/// Channel capacity computation via the Blahut-Arimoto algorithm.
#[derive(Debug, Clone)]
pub struct ChannelCapacity;

impl ChannelCapacity {
    /// Compute channel capacity C = max_{P(X)} I(X;Y) using Blahut-Arimoto.
    pub fn compute(channel: &ChannelMatrix) -> CapacityResult {
        Self::blahut_arimoto(channel, 1000, 1e-10)
    }

    /// Blahut-Arimoto algorithm with configurable iterations and tolerance.
    pub fn blahut_arimoto(
        channel: &ChannelMatrix,
        max_iter: usize,
        tolerance: f64,
    ) -> CapacityResult {
        let m = channel.rows;
        let n = channel.cols;
        if m == 0 || n == 0 {
            return CapacityResult { capacity: 0.0, optimal_prior: vec![], iterations: 0 };
        }

        // Start with uniform input distribution
        let mut px = vec![1.0 / m as f64; m];
        let mut capacity = 0.0;
        let mut iterations = 0;

        for iter in 0..max_iter {
            // Compute output distribution P(Y=j) = Σ_i P(X=i) · P(Y=j|X=i)
            let mut py = vec![0.0; n];
            for j in 0..n {
                for i in 0..m {
                    py[j] += px[i] * channel.matrix[i][j];
                }
            }

            // Compute Q(i,j) = P(Y=j|X=i) / P(Y=j) for nonzero entries
            let mut q = vec![vec![0.0; n]; m];
            for i in 0..m {
                for j in 0..n {
                    if channel.matrix[i][j] > 0.0 && py[j] > 0.0 {
                        q[i][j] = channel.matrix[i][j] / py[j];
                    }
                }
            }

            // Compute c(i) = exp(Σ_j P(Y=j|X=i) log Q(i,j)) = Π_j Q(i,j)^{P(Y=j|X=i)}
            let mut ci = vec![0.0; m];
            for i in 0..m {
                let mut log_ci = 0.0;
                for j in 0..n {
                    if channel.matrix[i][j] > 0.0 && q[i][j] > 0.0 {
                        log_ci += channel.matrix[i][j] * q[i][j].ln();
                    }
                }
                ci[i] = log_ci.exp();
            }

            // Update input distribution: P(X=i) ∝ P(X=i) · c(i)
            let mut new_px = vec![0.0; m];
            let mut sum = 0.0;
            for i in 0..m {
                new_px[i] = px[i] * ci[i];
                sum += new_px[i];
            }
            if sum > 0.0 {
                for i in 0..m { new_px[i] /= sum; }
            }

            // Compute bounds on capacity
            let i_lower = sum.ln() / (2.0f64).ln();
            let c_max = ci.iter().cloned().fold(0.0f64, f64::max);
            let i_upper = c_max.ln() / (2.0f64).ln();

            let new_capacity = i_lower;
            iterations = iter + 1;

            if (new_capacity - capacity).abs() < tolerance && iter > 0 {
                capacity = new_capacity;
                px = new_px;
                break;
            }
            capacity = new_capacity;
            px = new_px;
        }

        CapacityResult { capacity, optimal_prior: px, iterations }
    }
}

/// Result of channel capacity computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityResult {
    pub capacity: f64,
    pub optimal_prior: Vec<f64>,
    pub iterations: usize,
}

impl fmt::Display for CapacityResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Capacity: {:.6} bits ({} iterations)", self.capacity, self.iterations)
    }
}

// ---------------------------------------------------------------------------
// Leakage
// ---------------------------------------------------------------------------

/// Leakage quantification: how much secret information is leaked.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Leakage {
    /// Mutual information I(Secret; Observable).
    pub mutual_info: f64,
    /// Maximum possible entropy of the secret.
    pub max_entropy: f64,
    /// Leakage as fraction: I / H_max.
    pub fraction: f64,
    /// Channel capacity (maximum possible leakage).
    pub channel_capacity: f64,
    /// Min-entropy leakage.
    pub min_entropy_leakage: f64,
}

impl Leakage {
    /// Compute leakage from a channel matrix.
    pub fn compute(channel: &ChannelMatrix) -> Self {
        let mi = MutualInformation::from_channel(channel);
        let max_h = ShannonEntropy::max_entropy(channel.rows);
        let fraction = if max_h > 0.0 { mi / max_h } else { 0.0 };
        let cap = ChannelCapacity::compute(channel);
        let h_inf = MinEntropy::compute(&channel.prior);
        let h_inf_cond = MinEntropy::conditional(channel);
        let min_leak = h_inf - h_inf_cond;

        Leakage {
            mutual_info: mi,
            max_entropy: max_h,
            fraction,
            channel_capacity: cap.capacity,
            min_entropy_leakage: min_leak.max(0.0),
        }
    }

    /// Classify severity based on leakage fraction.
    pub fn severity(&self) -> &'static str {
        if self.fraction > 0.75 { "CRITICAL" }
        else if self.fraction > 0.5 { "HIGH" }
        else if self.fraction > 0.25 { "MEDIUM" }
        else if self.fraction > 0.01 { "LOW" }
        else { "NONE" }
    }

    pub fn is_significant(&self) -> bool {
        self.fraction > 0.01
    }
}

impl fmt::Display for Leakage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Leakage: {:.4} bits ({:.1}%, {})",
            self.mutual_info, self.fraction * 100.0, self.severity())
    }
}

// ---------------------------------------------------------------------------
// GuessingAdvantage
// ---------------------------------------------------------------------------

/// Guessing advantage: how much easier is it to guess the secret
/// after observing the output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuessingAdvantage {
    /// Prior guessing probability (without observation).
    pub prior_guess: f64,
    /// Posterior guessing probability (with observation).
    pub posterior_guess: f64,
    /// Multiplicative advantage: posterior / prior.
    pub multiplicative: f64,
    /// Additive advantage: posterior - prior.
    pub additive: f64,
}

impl GuessingAdvantage {
    /// Compute guessing advantage from a channel matrix.
    pub fn compute(channel: &ChannelMatrix) -> Self {
        // Prior guessing prob = max P(X)
        let prior_guess = channel.prior.iter().cloned().fold(0.0f64, f64::max);

        // Posterior guessing prob = Σ_j P(Y=j) max_i P(X=i|Y=j)
        let py = channel.output_distribution();
        let mut posterior_guess = 0.0;
        for j in 0..channel.cols {
            if py[j] > 0.0 {
                let posterior = channel.posterior(j);
                let max_post = posterior.iter().cloned().fold(0.0f64, f64::max);
                posterior_guess += py[j] * max_post;
            }
        }

        let multiplicative = if prior_guess > 0.0 { posterior_guess / prior_guess } else { 1.0 };
        let additive = posterior_guess - prior_guess;

        GuessingAdvantage {
            prior_guess,
            posterior_guess,
            multiplicative,
            additive,
        }
    }

    /// Whether there is any guessing advantage.
    pub fn has_advantage(&self) -> bool { self.multiplicative > 1.0001 }
}

impl fmt::Display for GuessingAdvantage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GuessingAdvantage: ×{:.2} ({:.4} → {:.4})",
            self.multiplicative, self.prior_guess, self.posterior_guess)
    }
}

// ---------------------------------------------------------------------------
// InformationFlowAnalyzer
// ---------------------------------------------------------------------------

/// High-level information flow analyzer combining all QIF measures.
#[derive(Debug, Clone)]
pub struct InformationFlowAnalyzer {
    pub channels: Vec<NamedChannel>,
}

/// A named channel for analysis.
#[derive(Debug, Clone)]
pub struct NamedChannel {
    pub name: String,
    pub channel: ChannelMatrix,
    pub description: String,
}

/// Complete analysis result for a channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowAnalysisResult {
    pub channel_name: String,
    pub shannon_entropy: f64,
    pub min_entropy: f64,
    pub mutual_info: f64,
    pub channel_capacity: f64,
    pub leakage: Leakage,
    pub guessing_advantage: GuessingAdvantage,
}

impl InformationFlowAnalyzer {
    pub fn new() -> Self {
        InformationFlowAnalyzer { channels: Vec::new() }
    }

    pub fn add_channel(&mut self, name: &str, channel: ChannelMatrix, desc: &str) {
        self.channels.push(NamedChannel {
            name: name.to_string(),
            channel,
            description: desc.to_string(),
        });
    }

    /// Analyze a single channel.
    pub fn analyze_channel(&self, channel: &ChannelMatrix) -> FlowAnalysisResult {
        let shannon = ShannonEntropy::compute(&channel.prior);
        let min_ent = MinEntropy::compute(&channel.prior);
        let mi = MutualInformation::from_channel(channel);
        let cap = ChannelCapacity::compute(channel);
        let leakage = Leakage::compute(channel);
        let ga = GuessingAdvantage::compute(channel);

        FlowAnalysisResult {
            channel_name: String::new(),
            shannon_entropy: shannon,
            min_entropy: min_ent,
            mutual_info: mi,
            channel_capacity: cap.capacity,
            leakage,
            guessing_advantage: ga,
        }
    }

    /// Analyze all registered channels.
    pub fn analyze_all(&self) -> Vec<FlowAnalysisResult> {
        self.channels.iter().map(|nc| {
            let mut result = self.analyze_channel(&nc.channel);
            result.channel_name = nc.name.clone();
            result
        }).collect()
    }

    /// Build a channel from execution traces.
    /// Each trace maps (secret_value, observable_outcome).
    pub fn build_channel_from_traces(
        traces: &[(usize, usize)],
        num_secrets: usize,
        num_observables: usize,
    ) -> ChannelMatrix {
        let mut counts = vec![vec![0usize; num_observables]; num_secrets];
        let mut row_totals = vec![0usize; num_secrets];

        for &(secret, observable) in traces {
            if secret < num_secrets && observable < num_observables {
                counts[secret][observable] += 1;
                row_totals[secret] += 1;
            }
        }

        let mut matrix = vec![vec![0.0; num_observables]; num_secrets];
        for i in 0..num_secrets {
            if row_totals[i] > 0 {
                for j in 0..num_observables {
                    matrix[i][j] = counts[i][j] as f64 / row_totals[i] as f64;
                }
            } else {
                // Uniform if no observations
                for j in 0..num_observables {
                    matrix[i][j] = 1.0 / num_observables as f64;
                }
            }
        }

        ChannelMatrix::new(matrix)
    }

    /// Build a channel from memory access patterns (GPU residue analysis).
    pub fn build_residue_channel(
        secret_values: &[u64],
        residue_observations: &[Vec<u8>],
    ) -> ChannelMatrix {
        assert_eq!(secret_values.len(), residue_observations.len());
        if secret_values.is_empty() {
            return ChannelMatrix::new(vec![]);
        }

        // Map unique secrets to indices
        let mut secret_map: HashMap<u64, usize> = HashMap::new();
        let mut secret_idx = 0;
        for &s in secret_values {
            if !secret_map.contains_key(&s) {
                secret_map.insert(s, secret_idx);
                secret_idx += 1;
            }
        }
        let num_secrets = secret_map.len();

        // Map unique observations to indices
        let mut obs_map: HashMap<Vec<u8>, usize> = HashMap::new();
        let mut obs_idx = 0;
        for obs in residue_observations {
            if !obs_map.contains_key(obs) {
                obs_map.insert(obs.clone(), obs_idx);
                obs_idx += 1;
            }
        }
        let num_obs = obs_map.len();

        // Count
        let mut counts = vec![vec![0usize; num_obs]; num_secrets];
        let mut totals = vec![0usize; num_secrets];
        for (i, (&s, obs)) in secret_values.iter().zip(residue_observations.iter()).enumerate() {
            let si = secret_map[&s];
            let oi = obs_map[obs];
            counts[si][oi] += 1;
            totals[si] += 1;
        }

        let mut matrix = vec![vec![0.0; num_obs]; num_secrets];
        for i in 0..num_secrets {
            if totals[i] > 0 {
                for j in 0..num_obs {
                    matrix[i][j] = counts[i][j] as f64 / totals[i] as f64;
                }
            }
        }

        ChannelMatrix::new(matrix)
    }

    /// Build a timing side-channel from execution time measurements.
    pub fn build_timing_channel(
        secret_values: &[u64],
        timings_ns: &[u64],
        num_bins: usize,
    ) -> ChannelMatrix {
        assert_eq!(secret_values.len(), timings_ns.len());
        if secret_values.is_empty() || num_bins == 0 {
            return ChannelMatrix::new(vec![]);
        }

        let min_t = *timings_ns.iter().min().unwrap();
        let max_t = *timings_ns.iter().max().unwrap();
        let range = if max_t > min_t { max_t - min_t } else { 1 };
        let bin_width = (range as f64 / num_bins as f64).max(1.0);

        // Map secrets
        let mut secret_map: HashMap<u64, usize> = HashMap::new();
        let mut idx = 0;
        for &s in secret_values {
            if !secret_map.contains_key(&s) {
                secret_map.insert(s, idx);
                idx += 1;
            }
        }
        let num_secrets = secret_map.len();

        let mut counts = vec![vec![0usize; num_bins]; num_secrets];
        let mut totals = vec![0usize; num_secrets];
        for (&s, &t) in secret_values.iter().zip(timings_ns.iter()) {
            let si = secret_map[&s];
            let bin = ((t - min_t) as f64 / bin_width).floor() as usize;
            let bin = bin.min(num_bins - 1);
            counts[si][bin] += 1;
            totals[si] += 1;
        }

        let mut matrix = vec![vec![0.0; num_bins]; num_secrets];
        for i in 0..num_secrets {
            if totals[i] > 0 {
                for j in 0..num_bins {
                    matrix[i][j] = counts[i][j] as f64 / totals[i] as f64;
                }
            }
        }

        ChannelMatrix::new(matrix)
    }

    /// Report summary.
    pub fn summary_report(&self) -> String {
        let results = self.analyze_all();
        let mut report = String::new();
        report.push_str("=== Information Flow Analysis Summary ===\n");
        for r in &results {
            report.push_str(&format!(
                "\nChannel: {}\n  H(X)={:.4}, H∞(X)={:.4}, I(X;Y)={:.4}, C={:.4}\n  Leakage: {:.1}% ({})\n  Guessing: ×{:.2}\n",
                r.channel_name,
                r.shannon_entropy, r.min_entropy, r.mutual_info, r.channel_capacity,
                r.leakage.fraction * 100.0, r.leakage.severity(),
                r.guessing_advantage.multiplicative,
            ));
        }
        report
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shannon_entropy_uniform() {
        let dist = vec![0.25, 0.25, 0.25, 0.25];
        let h = ShannonEntropy::compute(&dist);
        assert!((h - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_shannon_entropy_deterministic() {
        let dist = vec![1.0, 0.0, 0.0];
        let h = ShannonEntropy::compute(&dist);
        assert!(h.abs() < 1e-10);
    }

    #[test]
    fn test_shannon_entropy_binary() {
        let dist = vec![0.5, 0.5];
        let h = ShannonEntropy::compute(&dist);
        assert!((h - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_min_entropy() {
        let dist = vec![0.5, 0.25, 0.25];
        let h = MinEntropy::compute(&dist);
        assert!((h - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mutual_info_identity_channel() {
        let ch = ChannelMatrix::identity(4);
        let mi = MutualInformation::from_channel(&ch);
        assert!((mi - 2.0).abs() < 1e-6, "Identity channel MI should be log₂(4)=2, got {}", mi);
    }

    #[test]
    fn test_mutual_info_uniform_channel() {
        let ch = ChannelMatrix::uniform(4, 4);
        let mi = MutualInformation::from_channel(&ch);
        assert!(mi.abs() < 1e-6, "Uniform channel MI should be 0, got {}", mi);
    }

    #[test]
    fn test_channel_capacity_identity() {
        let ch = ChannelMatrix::identity(2);
        let result = ChannelCapacity::compute(&ch);
        assert!((result.capacity - 1.0).abs() < 0.01,
            "Binary identity channel capacity should be 1, got {}", result.capacity);
    }

    #[test]
    fn test_channel_capacity_bsc() {
        let ch = ChannelMatrix::binary_symmetric(0.5);
        let result = ChannelCapacity::compute(&ch);
        assert!(result.capacity.abs() < 0.01,
            "BSC(0.5) capacity should be ~0, got {}", result.capacity);
    }

    #[test]
    fn test_channel_validity() {
        let ch = ChannelMatrix::identity(3);
        assert!(ch.is_valid());

        let ch2 = ChannelMatrix::new(vec![vec![0.5, 0.5], vec![0.3, 0.7]]);
        assert!(ch2.is_valid());
    }

    #[test]
    fn test_deterministic_channel() {
        let ch = ChannelMatrix::deterministic(&[0, 0, 1, 1], 2);
        assert_eq!(ch.rows, 4);
        assert_eq!(ch.cols, 2);
        assert!((ch.matrix[0][0] - 1.0).abs() < 1e-12);
        assert!((ch.matrix[2][1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_leakage_computation() {
        let ch = ChannelMatrix::identity(4);
        let leak = Leakage::compute(&ch);
        assert!(leak.fraction > 0.9);
        assert!(leak.is_significant());
    }

    #[test]
    fn test_leakage_no_leak() {
        let ch = ChannelMatrix::uniform(4, 4);
        let leak = Leakage::compute(&ch);
        assert!(!leak.is_significant());
    }

    #[test]
    fn test_guessing_advantage_identity() {
        let ch = ChannelMatrix::identity(4);
        let ga = GuessingAdvantage::compute(&ch);
        // With identity channel, posterior guess = 1.0 (always guess correctly)
        assert!((ga.posterior_guess - 1.0).abs() < 1e-6);
        assert!(ga.has_advantage());
    }

    #[test]
    fn test_guessing_advantage_uniform() {
        let ch = ChannelMatrix::uniform(4, 4);
        let ga = GuessingAdvantage::compute(&ch);
        // With uniform channel, no advantage
        assert!((ga.multiplicative - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_build_channel_from_traces() {
        let traces = vec![
            (0, 0), (0, 0), (0, 1),
            (1, 1), (1, 1), (1, 0),
        ];
        let ch = InformationFlowAnalyzer::build_channel_from_traces(&traces, 2, 2);
        assert_eq!(ch.rows, 2);
        assert_eq!(ch.cols, 2);
        assert!((ch.matrix[0][0] - 2.0/3.0).abs() < 1e-6);
        assert!((ch.matrix[1][1] - 2.0/3.0).abs() < 1e-6);
    }

    #[test]
    fn test_build_timing_channel() {
        let secrets = vec![0, 0, 0, 1, 1, 1];
        let timings = vec![100, 110, 105, 200, 210, 205];
        let ch = InformationFlowAnalyzer::build_timing_channel(&secrets, &timings, 4);
        assert_eq!(ch.rows, 2);
        assert_eq!(ch.cols, 4);
        assert!(ch.is_valid());
    }

    #[test]
    fn test_channel_compose() {
        let ch1 = ChannelMatrix::identity(2);
        let ch2 = ChannelMatrix::binary_symmetric(0.1);
        let composed = ch1.compose(&ch2);
        assert_eq!(composed.rows, 2);
        assert_eq!(composed.cols, 2);
        // Identity composed with BSC should give BSC
        assert!((composed.matrix[0][0] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_channel_equivalence_classes() {
        let ch = ChannelMatrix::deterministic(&[0, 0, 1, 1], 2);
        assert_eq!(ch.num_equivalence_classes(), 2);
    }

    #[test]
    fn test_posterior() {
        let ch = ChannelMatrix::identity(3);
        let post = ch.posterior(0);
        assert!((post[0] - 1.0).abs() < 1e-6);
        assert!(post[1].abs() < 1e-6);
    }

    #[test]
    fn test_analyzer_summary() {
        let mut analyzer = InformationFlowAnalyzer::new();
        analyzer.add_channel("test", ChannelMatrix::identity(4), "test channel");
        let report = analyzer.summary_report();
        assert!(report.contains("test"));
    }

    #[test]
    fn test_conditional_entropy() {
        let joint = vec![
            vec![0.25, 0.0],
            vec![0.0, 0.25],
            vec![0.25, 0.0],
            vec![0.0, 0.25],
        ];
        let h = ShannonEntropy::conditional(&joint);
        // H(X|Y) should be 1 bit (given Y, still 2 equally likely X)
        assert!((h - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_max_entropy() {
        assert!((ShannonEntropy::max_entropy(8) - 3.0).abs() < 1e-10);
        assert!((ShannonEntropy::max_entropy(1) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_output_distribution() {
        let ch = ChannelMatrix::identity(3);
        let py = ch.output_distribution();
        for &p in &py {
            assert!((p - 1.0/3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_residue_channel() {
        let secrets = vec![0u64, 0, 1, 1, 2, 2];
        let observations = vec![
            vec![0u8, 0], vec![0, 0],
            vec![1, 0], vec![1, 0],
            vec![0, 1], vec![0, 1],
        ];
        let ch = InformationFlowAnalyzer::build_residue_channel(&secrets, &observations);
        assert_eq!(ch.rows, 3);
        assert!(ch.is_valid());
    }

    #[test]
    fn test_joint_entropy() {
        let joint = vec![
            vec![0.25, 0.25],
            vec![0.25, 0.25],
        ];
        let h = ShannonEntropy::joint(&joint);
        assert!((h - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalized_mi() {
        let joint = vec![
            vec![0.5, 0.0],
            vec![0.0, 0.5],
        ];
        let nmi = MutualInformation::normalized(&joint);
        assert!((nmi - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_vulnerability() {
        let dist = vec![0.5, 0.3, 0.2];
        let v = MinEntropy::vulnerability(&dist);
        assert!((v - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_channel_table() {
        let ch = ChannelMatrix::identity(2);
        let table = ch.to_table();
        assert!(table.contains("s0"));
        assert!(table.contains("o1"));
    }

    #[test]
    fn test_analyze_all() {
        let mut analyzer = InformationFlowAnalyzer::new();
        analyzer.add_channel("id", ChannelMatrix::identity(2), "identity");
        analyzer.add_channel("uni", ChannelMatrix::uniform(2, 2), "uniform");
        let results = analyzer.analyze_all();
        assert_eq!(results.len(), 2);
        assert!(results[0].mutual_info > results[1].mutual_info);
    }

    #[test]
    fn test_leakage_severity() {
        let ch = ChannelMatrix::identity(4);
        let leak = Leakage::compute(&ch);
        assert_eq!(leak.severity(), "CRITICAL");
    }
}

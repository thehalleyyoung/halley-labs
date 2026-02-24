#![allow(unused)]

use std::collections::{HashMap, HashSet, BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use serde::{Serialize, Deserialize};

// ---------------------------------------------------------------------------
// Distribution
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Distribution {
    pub probabilities: Vec<f64>,
}

impl Distribution {
    pub fn new(probs: Vec<f64>) -> Self {
        Self { probabilities: probs }
    }

    pub fn uniform(n: usize) -> Self {
        let p = 1.0 / n as f64;
        Self { probabilities: vec![p; n] }
    }

    pub fn from_counts(counts: &[u64]) -> Self {
        let total: u64 = counts.iter().sum();
        if total == 0 {
            return Self { probabilities: vec![0.0; counts.len()] };
        }
        let probs: Vec<f64> = counts.iter().map(|&c| c as f64 / total as f64).collect();
        Self { probabilities: probs }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.probabilities.is_empty() {
            return Err("Distribution is empty".to_string());
        }
        for (i, &p) in self.probabilities.iter().enumerate() {
            if p < 0.0 {
                return Err(format!("Negative probability at index {}: {}", i, p));
            }
            if p.is_nan() {
                return Err(format!("NaN probability at index {}", i));
            }
        }
        let sum: f64 = self.probabilities.iter().sum();
        if (sum - 1.0).abs() > 1e-9 {
            return Err(format!("Probabilities sum to {} (expected 1.0)", sum));
        }
        Ok(())
    }

    pub fn normalize(&mut self) {
        let sum: f64 = self.probabilities.iter().sum();
        if sum > 0.0 {
            for p in &mut self.probabilities {
                *p /= sum;
            }
        }
    }

    pub fn entropy(&self) -> f64 {
        ShannonEntropy::compute(self)
    }

    pub fn support(&self) -> usize {
        self.probabilities.iter().filter(|&&p| p > 0.0).count()
    }

    pub fn max_probability(&self) -> f64 {
        self.probabilities.iter().cloned().fold(0.0f64, f64::max)
    }

    pub fn min_nonzero_probability(&self) -> Option<f64> {
        self.probabilities.iter().cloned().filter(|&p| p > 0.0).fold(None, |acc, p| {
            Some(acc.map_or(p, |a: f64| a.min(p)))
        })
    }

    pub fn sample_index(&self, uniform_sample: f64) -> usize {
        let mut cumulative = 0.0;
        for (i, &p) in self.probabilities.iter().enumerate() {
            cumulative += p;
            if uniform_sample <= cumulative {
                return i;
            }
        }
        self.probabilities.len() - 1
    }

    pub fn is_deterministic(&self) -> bool {
        self.probabilities.iter().filter(|&&p| p > 0.0).count() == 1
    }

    pub fn is_uniform(&self) -> bool {
        if self.probabilities.is_empty() { return true; }
        let expected = 1.0 / self.probabilities.len() as f64;
        self.probabilities.iter().all(|&p| (p - expected).abs() < 1e-12)
    }

    pub fn total_variation_distance(&self, other: &Distribution) -> f64 {
        assert_eq!(self.probabilities.len(), other.probabilities.len());
        self.probabilities.iter().zip(other.probabilities.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum::<f64>() / 2.0
    }

    pub fn hellinger_distance(&self, other: &Distribution) -> f64 {
        assert_eq!(self.probabilities.len(), other.probabilities.len());
        let sum_sq: f64 = self.probabilities.iter().zip(other.probabilities.iter())
            .map(|(&a, &b)| (a.sqrt() - b.sqrt()).powi(2))
            .sum();
        (sum_sq / 2.0).sqrt()
    }
}

impl fmt::Display for Distribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Distribution({} outcomes, H={:.4})", self.probabilities.len(), self.entropy())
    }
}

// ---------------------------------------------------------------------------
// Joint Distribution
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointDistribution {
    pub matrix: Vec<Vec<f64>>,
    pub rows: usize,
    pub cols: usize,
}

impl JointDistribution {
    pub fn new(matrix: Vec<Vec<f64>>) -> Self {
        let rows = matrix.len();
        let cols = if rows > 0 { matrix[0].len() } else { 0 };
        Self { matrix, rows, cols }
    }

    pub fn uniform(rows: usize, cols: usize) -> Self {
        let p = 1.0 / (rows * cols) as f64;
        let matrix = vec![vec![p; cols]; rows];
        Self { matrix, rows, cols }
    }

    pub fn from_independent(px: &Distribution, py: &Distribution) -> Self {
        let rows = px.probabilities.len();
        let cols = py.probabilities.len();
        let mut matrix = vec![vec![0.0; cols]; rows];
        for i in 0..rows {
            for j in 0..cols {
                matrix[i][j] = px.probabilities[i] * py.probabilities[j];
            }
        }
        Self { matrix, rows, cols }
    }

    pub fn validate(&self) -> Result<(), String> {
        let sum: f64 = self.matrix.iter().flat_map(|row| row.iter()).sum();
        if (sum - 1.0).abs() > 1e-9 {
            return Err(format!("Joint distribution sums to {} (expected 1.0)", sum));
        }
        for (i, row) in self.matrix.iter().enumerate() {
            for (j, &p) in row.iter().enumerate() {
                if p < 0.0 {
                    return Err(format!("Negative probability at ({}, {}): {}", i, j, p));
                }
            }
        }
        Ok(())
    }

    pub fn normalize(&mut self) {
        let sum: f64 = self.matrix.iter().flat_map(|row| row.iter()).sum();
        if sum > 0.0 {
            for row in &mut self.matrix {
                for p in row.iter_mut() {
                    *p /= sum;
                }
            }
        }
    }

    pub fn marginalize_x(&self) -> Distribution {
        let mut probs = vec![0.0; self.rows];
        for i in 0..self.rows {
            probs[i] = self.matrix[i].iter().sum();
        }
        Distribution::new(probs)
    }

    pub fn marginalize_y(&self) -> Distribution {
        let mut probs = vec![0.0; self.cols];
        for j in 0..self.cols {
            for i in 0..self.rows {
                probs[j] += self.matrix[i][j];
            }
        }
        Distribution::new(probs)
    }

    pub fn conditional_y_given_x(&self, x: usize) -> Distribution {
        let row_sum: f64 = self.matrix[x].iter().sum();
        if row_sum == 0.0 {
            return Distribution::new(vec![0.0; self.cols]);
        }
        let probs: Vec<f64> = self.matrix[x].iter().map(|&p| p / row_sum).collect();
        Distribution::new(probs)
    }

    pub fn conditional_x_given_y(&self, y: usize) -> Distribution {
        let col_sum: f64 = (0..self.rows).map(|i| self.matrix[i][y]).sum();
        if col_sum == 0.0 {
            return Distribution::new(vec![0.0; self.rows]);
        }
        let probs: Vec<f64> = (0..self.rows).map(|i| self.matrix[i][y] / col_sum).collect();
        Distribution::new(probs)
    }

    pub fn conditional(&self, x: usize) -> Distribution {
        self.conditional_y_given_x(x)
    }

    pub fn joint_entropy(&self) -> f64 {
        let probs: Vec<f64> = self.matrix.iter().flat_map(|row| row.iter()).cloned().collect();
        ShannonEntropy::compute(&Distribution::new(probs))
    }

    pub fn mutual_information(&self) -> f64 {
        let px = self.marginalize_x();
        let py = self.marginalize_y();
        let hx = ShannonEntropy::compute(&px);
        let hy = ShannonEntropy::compute(&py);
        let hxy = self.joint_entropy();
        hx + hy - hxy
    }

    pub fn is_independent(&self) -> bool {
        let px = self.marginalize_x();
        let py = self.marginalize_y();
        for i in 0..self.rows {
            for j in 0..self.cols {
                let expected = px.probabilities[i] * py.probabilities[j];
                if (self.matrix[i][j] - expected).abs() > 1e-9 {
                    return false;
                }
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Shannon Entropy
// ---------------------------------------------------------------------------

pub struct ShannonEntropy;

impl ShannonEntropy {
    pub fn compute(dist: &Distribution) -> f64 {
        let mut h = 0.0;
        for &p in &dist.probabilities {
            if p > 0.0 {
                h -= p * p.log2();
            }
        }
        h
    }

    pub fn conditional_entropy(joint: &JointDistribution) -> f64 {
        // H(Y|X) = H(X,Y) - H(X)
        let hxy = joint.joint_entropy();
        let px = joint.marginalize_x();
        let hx = Self::compute(&px);
        hxy - hx
    }

    pub fn joint_entropy(joint: &JointDistribution) -> f64 {
        joint.joint_entropy()
    }

    pub fn cross_entropy(p: &Distribution, q: &Distribution) -> f64 {
        assert_eq!(p.probabilities.len(), q.probabilities.len());
        let mut h = 0.0;
        for (&pi, &qi) in p.probabilities.iter().zip(q.probabilities.iter()) {
            if pi > 0.0 {
                if qi <= 0.0 {
                    return f64::INFINITY;
                }
                h -= pi * qi.log2();
            }
        }
        h
    }

    pub fn kl_divergence(p: &Distribution, q: &Distribution) -> f64 {
        assert_eq!(p.probabilities.len(), q.probabilities.len());
        let mut kl = 0.0;
        for (&pi, &qi) in p.probabilities.iter().zip(q.probabilities.iter()) {
            if pi > 0.0 {
                if qi <= 0.0 {
                    return f64::INFINITY;
                }
                kl += pi * (pi / qi).log2();
            }
        }
        kl
    }

    pub fn js_divergence(p: &Distribution, q: &Distribution) -> f64 {
        assert_eq!(p.probabilities.len(), q.probabilities.len());
        let m_probs: Vec<f64> = p.probabilities.iter().zip(q.probabilities.iter())
            .map(|(&a, &b)| (a + b) / 2.0)
            .collect();
        let m = Distribution::new(m_probs);
        (Self::kl_divergence(p, &m) + Self::kl_divergence(q, &m)) / 2.0
    }
}

// ---------------------------------------------------------------------------
// Mutual Information
// ---------------------------------------------------------------------------

pub struct MutualInformation;

impl MutualInformation {
    pub fn compute(joint: &JointDistribution) -> f64 {
        joint.mutual_information()
    }

    pub fn normalized(joint: &JointDistribution) -> f64 {
        let mi = Self::compute(joint);
        let hx = ShannonEntropy::compute(&joint.marginalize_x());
        let hy = ShannonEntropy::compute(&joint.marginalize_y());
        let max_h = hx.max(hy);
        if max_h == 0.0 { return 0.0; }
        mi / max_h
    }

    pub fn pointwise(joint: &JointDistribution) -> Vec<Vec<f64>> {
        let px = joint.marginalize_x();
        let py = joint.marginalize_y();
        let mut pmi = vec![vec![0.0; joint.cols]; joint.rows];
        for i in 0..joint.rows {
            for j in 0..joint.cols {
                let pxy = joint.matrix[i][j];
                let px_i = px.probabilities[i];
                let py_j = py.probabilities[j];
                if pxy > 0.0 && px_i > 0.0 && py_j > 0.0 {
                    pmi[i][j] = (pxy / (px_i * py_j)).log2();
                }
            }
        }
        pmi
    }

    pub fn mutual_info_matrix(joints: &[JointDistribution]) -> Vec<Vec<f64>> {
        let n = joints.len();
        let mut matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    matrix[i][j] = ShannonEntropy::compute(&joints[i].marginalize_x());
                } else if i < j {
                    matrix[i][j] = Self::compute(&joints[i]);
                    matrix[j][i] = matrix[i][j];
                }
            }
        }
        matrix
    }

    pub fn variation_of_information(joint: &JointDistribution) -> f64 {
        let hxy = joint.joint_entropy();
        let mi = Self::compute(joint);
        hxy - mi
    }

    pub fn adjusted_mutual_information(joint: &JointDistribution) -> f64 {
        let mi = Self::compute(joint);
        let hx = ShannonEntropy::compute(&joint.marginalize_x());
        let hy = ShannonEntropy::compute(&joint.marginalize_y());
        let expected_mi = 0.0; // Simplified: true EMI requires hypergeometric computation
        let max_h = hx.max(hy);
        if max_h - expected_mi == 0.0 { return 0.0; }
        (mi - expected_mi) / (max_h - expected_mi)
    }
}

// ---------------------------------------------------------------------------
// Channel & Capacity
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Channel {
    pub matrix: Vec<Vec<f64>>,
    pub input_size: usize,
    pub output_size: usize,
}

impl Channel {
    pub fn new(matrix: Vec<Vec<f64>>) -> Self {
        let input_size = matrix.len();
        let output_size = if input_size > 0 { matrix[0].len() } else { 0 };
        Self { matrix, input_size, output_size }
    }

    pub fn identity(n: usize) -> Self {
        let mut matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            matrix[i][i] = 1.0;
        }
        Self { matrix, input_size: n, output_size: n }
    }

    pub fn binary_symmetric(epsilon: f64) -> Self {
        let matrix = vec![
            vec![1.0 - epsilon, epsilon],
            vec![epsilon, 1.0 - epsilon],
        ];
        Self { matrix, input_size: 2, output_size: 2 }
    }

    pub fn binary_erasure(epsilon: f64) -> Self {
        let matrix = vec![
            vec![1.0 - epsilon, 0.0, epsilon],
            vec![0.0, 1.0 - epsilon, epsilon],
        ];
        Self { matrix, input_size: 2, output_size: 3 }
    }

    pub fn validate(&self) -> Result<(), String> {
        for (i, row) in self.matrix.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            if (sum - 1.0).abs() > 1e-9 {
                return Err(format!("Row {} sums to {} (expected 1.0)", i, sum));
            }
            for (j, &p) in row.iter().enumerate() {
                if p < 0.0 || p > 1.0 + 1e-9 {
                    return Err(format!("Invalid probability at ({}, {}): {}", i, j, p));
                }
            }
        }
        Ok(())
    }

    pub fn output_distribution(&self, input: &Distribution) -> Distribution {
        let mut output = vec![0.0; self.output_size];
        for x in 0..self.input_size {
            for y in 0..self.output_size {
                output[y] += input.probabilities[x] * self.matrix[x][y];
            }
        }
        Distribution::new(output)
    }

    pub fn posterior(&self, input_prior: &Distribution, observed_y: usize) -> Distribution {
        let mut posteriors = vec![0.0; self.input_size];
        let mut total = 0.0;
        for x in 0..self.input_size {
            posteriors[x] = input_prior.probabilities[x] * self.matrix[x][observed_y];
            total += posteriors[x];
        }
        if total > 0.0 {
            for p in &mut posteriors {
                *p /= total;
            }
        }
        Distribution::new(posteriors)
    }

    pub fn to_joint(&self, input: &Distribution) -> JointDistribution {
        let mut matrix = vec![vec![0.0; self.output_size]; self.input_size];
        for x in 0..self.input_size {
            for y in 0..self.output_size {
                matrix[x][y] = input.probabilities[x] * self.matrix[x][y];
            }
        }
        JointDistribution::new(matrix)
    }

    pub fn cascade(&self, other: &Channel) -> Channel {
        assert_eq!(self.output_size, other.input_size);
        let mut matrix = vec![vec![0.0; other.output_size]; self.input_size];
        for x in 0..self.input_size {
            for z in 0..other.output_size {
                for y in 0..self.output_size {
                    matrix[x][z] += self.matrix[x][y] * other.matrix[y][z];
                }
            }
        }
        Channel::new(matrix)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityResult {
    pub capacity: f64,
    pub optimal_input: Distribution,
    pub iterations: usize,
    pub converged: bool,
}

pub struct ChannelCapacityEstimator;

impl ChannelCapacityEstimator {
    /// Blahut-Arimoto algorithm for computing channel capacity.
    pub fn blahut_arimoto(channel: &Channel, max_iterations: usize, tolerance: f64) -> CapacityResult {
        let n = channel.input_size;
        let m = channel.output_size;

        // Start with uniform input distribution
        let mut p_x = vec![1.0 / n as f64; n];
        let mut capacity = 0.0f64;
        let mut converged = false;
        let mut iterations = 0;

        for iter in 0..max_iterations {
            iterations = iter + 1;

            // Compute output distribution q(y) = sum_x p(x) * W(y|x)
            let mut q_y = vec![0.0; m];
            for y in 0..m {
                for x in 0..n {
                    q_y[y] += p_x[x] * channel.matrix[x][y];
                }
            }

            // Compute c(x) = prod_y W(y|x)^{W(y|x) / q(y)}  (actually compute log)
            let mut log_c = vec![0.0; n];
            for x in 0..n {
                for y in 0..m {
                    let w = channel.matrix[x][y];
                    if w > 0.0 && q_y[y] > 0.0 {
                        log_c[x] += w * (w / q_y[y]).ln();
                    }
                }
            }

            // Compute upper and lower bounds on capacity
            let c_max = log_c.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let c_lower: f64 = p_x.iter().zip(log_c.iter()).map(|(&px, &lc)| px * lc).sum();

            let new_capacity = c_lower / std::f64::consts::LN_2; // convert to bits

            if (new_capacity - capacity).abs() < tolerance {
                converged = true;
                capacity = new_capacity;
                break;
            }
            capacity = new_capacity;

            // Update input distribution
            let mut new_p_x = vec![0.0; n];
            let mut sum = 0.0;
            for x in 0..n {
                new_p_x[x] = p_x[x] * log_c[x].exp();
                sum += new_p_x[x];
            }
            for x in 0..n {
                new_p_x[x] /= sum;
            }
            p_x = new_p_x;
        }

        CapacityResult {
            capacity,
            optimal_input: Distribution::new(p_x),
            iterations,
            converged,
        }
    }

    pub fn capacity_bounds(channel: &Channel) -> (f64, f64) {
        // Lower bound: uniform input
        let uniform = Distribution::uniform(channel.input_size);
        let joint = channel.to_joint(&uniform);
        let lower = joint.mutual_information();

        // Upper bound: log of output alphabet size
        let upper = (channel.output_size as f64).log2();

        (lower, upper)
    }
}

// ---------------------------------------------------------------------------
// Min-Entropy
// ---------------------------------------------------------------------------

pub struct MinEntropy;

impl MinEntropy {
    pub fn compute(dist: &Distribution) -> f64 {
        let max_p = dist.max_probability();
        if max_p <= 0.0 {
            return f64::INFINITY;
        }
        -max_p.log2()
    }

    pub fn conditional(joint: &JointDistribution) -> f64 {
        // H_inf(X|Y) = -log2(sum_y max_x P(x,y))
        let mut sum = 0.0;
        for j in 0..joint.cols {
            let mut max_pxy = 0.0f64;
            for i in 0..joint.rows {
                max_pxy = max_pxy.max(joint.matrix[i][j]);
            }
            sum += max_pxy;
        }
        if sum <= 0.0 {
            return f64::INFINITY;
        }
        -sum.log2()
    }

    pub fn guessing_entropy(dist: &Distribution) -> f64 {
        let mut sorted: Vec<f64> = dist.probabilities.iter().cloned().filter(|&p| p > 0.0).collect();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let mut ge = 0.0;
        for (i, &p) in sorted.iter().enumerate() {
            ge += (i as f64 + 1.0) * p;
        }
        ge
    }

    pub fn vulnerability(dist: &Distribution) -> f64 {
        dist.max_probability()
    }

    pub fn min_entropy_leakage(joint: &JointDistribution) -> f64 {
        let px = joint.marginalize_x();
        let prior_vuln = Self::vulnerability(&px);
        let mut posterior_vuln = 0.0;
        for j in 0..joint.cols {
            let mut max_pxy = 0.0f64;
            for i in 0..joint.rows {
                max_pxy = max_pxy.max(joint.matrix[i][j]);
            }
            posterior_vuln += max_pxy;
        }
        if prior_vuln <= 0.0 {
            return 0.0;
        }
        (posterior_vuln / prior_vuln).log2()
    }
}

// ---------------------------------------------------------------------------
// Rényi Entropy
// ---------------------------------------------------------------------------

pub struct RenyiEntropy;

impl RenyiEntropy {
    pub fn compute(dist: &Distribution, alpha: f64) -> f64 {
        if alpha == 1.0 {
            return ShannonEntropy::compute(dist);
        }
        if alpha == f64::INFINITY {
            return MinEntropy::compute(dist);
        }
        if alpha == 0.0 {
            return Self::hartley_entropy(dist);
        }
        assert!(alpha >= 0.0, "Alpha must be non-negative");

        let sum: f64 = dist.probabilities.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p.powf(alpha))
            .sum();

        if sum <= 0.0 {
            return f64::INFINITY;
        }
        sum.log2() / (1.0 - alpha)
    }

    pub fn collision_entropy(dist: &Distribution) -> f64 {
        Self::compute(dist, 2.0)
    }

    pub fn hartley_entropy(dist: &Distribution) -> f64 {
        let support = dist.support();
        if support == 0 { return 0.0; }
        (support as f64).log2()
    }

    pub fn is_monotone_in_alpha(dist: &Distribution, alphas: &[f64]) -> bool {
        let values: Vec<f64> = alphas.iter().map(|&a| Self::compute(dist, a)).collect();
        for i in 1..values.len() {
            if values[i] > values[i - 1] + 1e-12 {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Leakage Metric
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LeakageMetric {
    ShannonLeakage,
    MinEntropyLeakage,
    GLeakage,
    MutualInformation,
    ChannelCapacity,
    GuessingEntropy,
}

impl fmt::Display for LeakageMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShannonLeakage => write!(f, "Shannon Leakage"),
            Self::MinEntropyLeakage => write!(f, "Min-Entropy Leakage"),
            Self::GLeakage => write!(f, "g-Leakage"),
            Self::MutualInformation => write!(f, "Mutual Information"),
            Self::ChannelCapacity => write!(f, "Channel Capacity"),
            Self::GuessingEntropy => write!(f, "Guessing Entropy"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakageResult {
    pub metric: LeakageMetric,
    pub value: f64,
    pub prior_uncertainty: f64,
    pub posterior_uncertainty: f64,
    pub relative_leakage: f64,
}

impl fmt::Display for LeakageResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {:.6} bits (prior={:.6}, posterior={:.6}, relative={:.4})",
            self.metric, self.value, self.prior_uncertainty,
            self.posterior_uncertainty, self.relative_leakage
        )
    }
}

pub fn compute_leakage(
    metric: LeakageMetric,
    joint: &JointDistribution,
    channel: &Channel,
) -> LeakageResult {
    let px = joint.marginalize_x();
    match metric {
        LeakageMetric::ShannonLeakage => {
            let prior = ShannonEntropy::compute(&px);
            let posterior = ShannonEntropy::conditional_entropy(joint);
            let leakage = prior - posterior;
            LeakageResult {
                metric,
                value: leakage,
                prior_uncertainty: prior,
                posterior_uncertainty: posterior,
                relative_leakage: if prior > 0.0 { leakage / prior } else { 0.0 },
            }
        }
        LeakageMetric::MinEntropyLeakage => {
            let prior = MinEntropy::compute(&px);
            let posterior = MinEntropy::conditional(joint);
            let leakage = prior - posterior;
            LeakageResult {
                metric,
                value: leakage,
                prior_uncertainty: prior,
                posterior_uncertainty: posterior,
                relative_leakage: if prior > 0.0 { leakage / prior } else { 0.0 },
            }
        }
        LeakageMetric::MutualInformation => {
            let mi = MutualInformation::compute(joint);
            let prior = ShannonEntropy::compute(&px);
            LeakageResult {
                metric,
                value: mi,
                prior_uncertainty: prior,
                posterior_uncertainty: prior - mi,
                relative_leakage: if prior > 0.0 { mi / prior } else { 0.0 },
            }
        }
        LeakageMetric::ChannelCapacity => {
            let result = ChannelCapacityEstimator::blahut_arimoto(channel, 1000, 1e-10);
            let prior = (channel.input_size as f64).log2();
            LeakageResult {
                metric,
                value: result.capacity,
                prior_uncertainty: prior,
                posterior_uncertainty: prior - result.capacity,
                relative_leakage: if prior > 0.0 { result.capacity / prior } else { 0.0 },
            }
        }
        LeakageMetric::GuessingEntropy => {
            let prior_ge = MinEntropy::guessing_entropy(&px);
            let py = joint.marginalize_y();
            let mut posterior_ge = 0.0;
            for j in 0..joint.cols {
                let cond = joint.conditional_y_given_x(j);
                posterior_ge += py.probabilities[j] * MinEntropy::guessing_entropy(&cond);
            }
            LeakageResult {
                metric,
                value: prior_ge - posterior_ge,
                prior_uncertainty: prior_ge,
                posterior_uncertainty: posterior_ge,
                relative_leakage: if prior_ge > 0.0 { (prior_ge - posterior_ge) / prior_ge } else { 0.0 },
            }
        }
        LeakageMetric::GLeakage => {
            // Default g-leakage with identity gain function
            let gain = GainFunction::identity(px.probabilities.len());
            let result = g_leakage(&px, joint, &gain);
            LeakageResult {
                metric,
                value: result,
                prior_uncertainty: MinEntropy::compute(&px),
                posterior_uncertainty: MinEntropy::conditional(joint),
                relative_leakage: result,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Gain Function & g-Leakage
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GainFunction {
    pub matrix: Vec<Vec<f64>>,
    pub num_secrets: usize,
    pub num_guesses: usize,
}

impl GainFunction {
    pub fn new(matrix: Vec<Vec<f64>>) -> Self {
        let num_secrets = matrix.len();
        let num_guesses = if num_secrets > 0 { matrix[0].len() } else { 0 };
        Self { matrix, num_secrets, num_guesses }
    }

    pub fn identity(n: usize) -> Self {
        let mut matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            matrix[i][i] = 1.0;
        }
        Self { matrix, num_secrets: n, num_guesses: n }
    }

    pub fn constant(n: usize, value: f64) -> Self {
        let matrix = vec![vec![value; n]; n];
        Self { matrix, num_secrets: n, num_guesses: n }
    }

    pub fn gain(&self, secret: usize, guess: usize) -> f64 {
        self.matrix[secret][guess]
    }

    pub fn max_gain_for_secret(&self, secret: usize) -> f64 {
        self.matrix[secret].iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }
}

pub fn g_leakage(prior: &Distribution, joint: &JointDistribution, gain: &GainFunction) -> f64 {
    // Prior g-vulnerability: max_w sum_x pi(x) * g(w, x)
    let mut prior_vuln = f64::NEG_INFINITY;
    for w in 0..gain.num_guesses {
        let mut v = 0.0;
        for x in 0..gain.num_secrets.min(prior.probabilities.len()) {
            v += prior.probabilities[x] * gain.gain(x, w);
        }
        prior_vuln = prior_vuln.max(v);
    }

    // Posterior g-vulnerability: sum_y max_w sum_x P(x,y) * g(w, x)
    let mut posterior_vuln = 0.0;
    for y in 0..joint.cols {
        let mut max_w = f64::NEG_INFINITY;
        for w in 0..gain.num_guesses {
            let mut v = 0.0;
            for x in 0..gain.num_secrets.min(joint.rows) {
                v += joint.matrix[x][y] * gain.gain(x, w);
            }
            max_w = max_w.max(v);
        }
        posterior_vuln += max_w;
    }

    if prior_vuln <= 0.0 { return 0.0; }
    (posterior_vuln / prior_vuln).log2()
}

// ---------------------------------------------------------------------------
// Side Channel Analyzer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SideChannelAnalyzer {
    pub channel: Channel,
    pub prior: Distribution,
    pub gain_function: Option<GainFunction>,
}

impl SideChannelAnalyzer {
    pub fn new(channel: Channel, prior: Distribution) -> Self {
        Self { channel, prior, gain_function: None }
    }

    pub fn with_gain_function(mut self, gain: GainFunction) -> Self {
        self.gain_function = Some(gain);
        self
    }

    pub fn analyze(&self) -> SideChannelReport {
        let joint = self.channel.to_joint(&self.prior);

        let shannon_leakage = compute_leakage(LeakageMetric::ShannonLeakage, &joint, &self.channel);
        let mi_leakage = compute_leakage(LeakageMetric::MutualInformation, &joint, &self.channel);
        let min_entropy_leakage = compute_leakage(LeakageMetric::MinEntropyLeakage, &joint, &self.channel);
        let capacity_result = ChannelCapacityEstimator::blahut_arimoto(&self.channel, 1000, 1e-10);

        let g_leakage_result = if let Some(ref gain) = self.gain_function {
            Some(g_leakage(&self.prior, &joint, gain))
        } else {
            None
        };

        let prior_entropy = ShannonEntropy::compute(&self.prior);
        let prior_min_entropy = MinEntropy::compute(&self.prior);
        let collision_entropy = RenyiEntropy::collision_entropy(&self.prior);

        SideChannelReport {
            shannon_leakage,
            mutual_information: mi_leakage,
            min_entropy_leakage,
            channel_capacity: capacity_result,
            g_leakage: g_leakage_result,
            prior_entropy,
            prior_min_entropy,
            collision_entropy,
            is_perfectly_secure: joint.is_independent(),
        }
    }

    pub fn compare_channels(channels: &[(Channel, Distribution)]) -> Vec<SideChannelReport> {
        channels.iter()
            .map(|(ch, prior)| {
                let analyzer = SideChannelAnalyzer::new(ch.clone(), prior.clone());
                analyzer.analyze()
            })
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SideChannelReport {
    pub shannon_leakage: LeakageResult,
    pub mutual_information: LeakageResult,
    pub min_entropy_leakage: LeakageResult,
    pub channel_capacity: CapacityResult,
    pub g_leakage: Option<f64>,
    pub prior_entropy: f64,
    pub prior_min_entropy: f64,
    pub collision_entropy: f64,
    pub is_perfectly_secure: bool,
}

impl SideChannelReport {
    pub fn max_leakage(&self) -> f64 {
        let mut max = self.shannon_leakage.value;
        if self.mutual_information.value > max { max = self.mutual_information.value; }
        if self.min_entropy_leakage.value > max { max = self.min_entropy_leakage.value; }
        if let Some(g) = self.g_leakage {
            if g > max { max = g; }
        }
        max
    }

    pub fn is_secure(&self, threshold: f64) -> bool {
        self.max_leakage() <= threshold
    }
}

impl fmt::Display for SideChannelReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Side Channel Analysis Report ===")?;
        writeln!(f, "Prior Shannon entropy: {:.6} bits", self.prior_entropy)?;
        writeln!(f, "Prior min-entropy: {:.6} bits", self.prior_min_entropy)?;
        writeln!(f, "Collision entropy: {:.6} bits", self.collision_entropy)?;
        writeln!(f, "Shannon leakage: {}", self.shannon_leakage)?;
        writeln!(f, "Mutual information: {}", self.mutual_information)?;
        writeln!(f, "Min-entropy leakage: {}", self.min_entropy_leakage)?;
        writeln!(f, "Channel capacity: {:.6} bits ({} iterations, converged={})",
            self.channel_capacity.capacity,
            self.channel_capacity.iterations,
            self.channel_capacity.converged)?;
        if let Some(g) = self.g_leakage {
            writeln!(f, "g-leakage: {:.6} bits", g)?;
        }
        writeln!(f, "Perfectly secure: {}", self.is_perfectly_secure)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Entropy Rate for Stochastic Processes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MarkovChain {
    pub transition_matrix: Vec<Vec<f64>>,
    pub states: usize,
}

impl MarkovChain {
    pub fn new(matrix: Vec<Vec<f64>>) -> Self {
        let states = matrix.len();
        Self { transition_matrix: matrix, states }
    }

    pub fn stationary_distribution(&self, max_iterations: usize, tolerance: f64) -> Distribution {
        let n = self.states;
        let mut pi = vec![1.0 / n as f64; n];

        for _ in 0..max_iterations {
            let mut new_pi = vec![0.0; n];
            for j in 0..n {
                for i in 0..n {
                    new_pi[j] += pi[i] * self.transition_matrix[i][j];
                }
            }
            let diff: f64 = pi.iter().zip(new_pi.iter()).map(|(&a, &b)| (a - b).abs()).sum();
            pi = new_pi;
            if diff < tolerance {
                break;
            }
        }
        Distribution::new(pi)
    }

    pub fn entropy_rate(&self) -> f64 {
        let pi = self.stationary_distribution(10000, 1e-12);
        let mut rate = 0.0;
        for i in 0..self.states {
            for j in 0..self.states {
                let p = self.transition_matrix[i][j];
                if p > 0.0 {
                    rate -= pi.probabilities[i] * p * p.log2();
                }
            }
        }
        rate
    }
}

// ---------------------------------------------------------------------------
// Quantitative Information Flow (additional)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SmithVulnerability;

impl SmithVulnerability {
    pub fn prior_vulnerability(dist: &Distribution) -> f64 {
        dist.max_probability()
    }

    pub fn posterior_vulnerability(joint: &JointDistribution) -> f64 {
        let mut vuln = 0.0;
        for j in 0..joint.cols {
            let mut max_p = 0.0f64;
            for i in 0..joint.rows {
                max_p = max_p.max(joint.matrix[i][j]);
            }
            vuln += max_p;
        }
        vuln
    }

    pub fn multiplicative_leakage(joint: &JointDistribution) -> f64 {
        let px = joint.marginalize_x();
        let prior_v = Self::prior_vulnerability(&px);
        let posterior_v = Self::posterior_vulnerability(joint);
        if prior_v <= 0.0 { return 0.0; }
        posterior_v / prior_v
    }

    pub fn additive_leakage(joint: &JointDistribution) -> f64 {
        let px = joint.marginalize_x();
        let prior_v = Self::prior_vulnerability(&px);
        let posterior_v = Self::posterior_vulnerability(joint);
        posterior_v - prior_v
    }
}

// ---------------------------------------------------------------------------
// Differential Privacy Helper
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DifferentialPrivacyChecker {
    pub epsilon: f64,
    pub delta: f64,
}

impl DifferentialPrivacyChecker {
    pub fn new(epsilon: f64, delta: f64) -> Self {
        Self { epsilon, delta }
    }

    pub fn check_channel(&self, channel: &Channel) -> bool {
        // Check (ε, δ)-differential privacy
        for y in 0..channel.output_size {
            for x1 in 0..channel.input_size {
                for x2 in 0..channel.input_size {
                    if x1 == x2 { continue; }
                    let p1 = channel.matrix[x1][y];
                    let p2 = channel.matrix[x2][y];
                    if p2 > 0.0 {
                        let ratio = p1 / p2;
                        if ratio > self.epsilon.exp() + self.delta {
                            return false;
                        }
                    } else if p1 > self.delta {
                        return false;
                    }
                }
            }
        }
        true
    }

    pub fn compute_epsilon(&self, channel: &Channel) -> f64 {
        let mut max_ratio = 0.0f64;
        for y in 0..channel.output_size {
            for x1 in 0..channel.input_size {
                for x2 in 0..channel.input_size {
                    if x1 == x2 { continue; }
                    let p1 = channel.matrix[x1][y];
                    let p2 = channel.matrix[x2][y];
                    if p2 > 0.0 && p1 > 0.0 {
                        let ratio = (p1 / p2).ln();
                        max_ratio = max_ratio.max(ratio);
                    }
                }
            }
        }
        max_ratio
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// ===== Extended Side Channel Metrics Operations =====

#[derive(Debug, Clone)]
pub struct DifferentialPrivacy {
    pub epsilon: f64,
    pub delta: f64,
    pub mechanism: String,
    pub sensitivity: f64,
}

impl DifferentialPrivacy {
    pub fn new(epsilon: f64, delta: f64, mechanism: String, sensitivity: f64) -> Self {
        DifferentialPrivacy { epsilon, delta, mechanism, sensitivity }
    }

    pub fn get_epsilon(&self) -> f64 {
        self.epsilon
    }

    pub fn get_delta(&self) -> f64 {
        self.delta
    }

    pub fn get_mechanism(&self) -> &str {
        &self.mechanism
    }

    pub fn get_sensitivity(&self) -> f64 {
        self.sensitivity
    }

    pub fn with_epsilon(mut self, v: f64) -> Self {
        self.epsilon = v; self
    }

    pub fn with_delta(mut self, v: f64) -> Self {
        self.delta = v; self
    }

    pub fn with_mechanism(mut self, v: impl Into<String>) -> Self {
        self.mechanism = v.into(); self
    }

    pub fn with_sensitivity(mut self, v: f64) -> Self {
        self.sensitivity = v; self
    }

}

impl fmt::Display for DifferentialPrivacy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DifferentialPrivacy({:?})", self.epsilon)
    }
}

#[derive(Debug, Clone)]
pub struct DifferentialPrivacyBuilder {
    epsilon: f64,
    delta: f64,
    mechanism: String,
    sensitivity: f64,
}

impl DifferentialPrivacyBuilder {
    pub fn new() -> Self {
        DifferentialPrivacyBuilder {
            epsilon: 0.0,
            delta: 0.0,
            mechanism: String::new(),
            sensitivity: 0.0,
        }
    }

    pub fn epsilon(mut self, v: f64) -> Self { self.epsilon = v; self }
    pub fn delta(mut self, v: f64) -> Self { self.delta = v; self }
    pub fn mechanism(mut self, v: impl Into<String>) -> Self { self.mechanism = v.into(); self }
    pub fn sensitivity(mut self, v: f64) -> Self { self.sensitivity = v; self }
}

#[derive(Debug, Clone)]
pub struct RandomizedResponse {
    pub true_probability: f64,
    pub flip_probability: f64,
    pub sample_size: usize,
}

impl RandomizedResponse {
    pub fn new(true_probability: f64, flip_probability: f64, sample_size: usize) -> Self {
        RandomizedResponse { true_probability, flip_probability, sample_size }
    }

    pub fn get_true_probability(&self) -> f64 {
        self.true_probability
    }

    pub fn get_flip_probability(&self) -> f64 {
        self.flip_probability
    }

    pub fn get_sample_size(&self) -> usize {
        self.sample_size
    }

    pub fn with_true_probability(mut self, v: f64) -> Self {
        self.true_probability = v; self
    }

    pub fn with_flip_probability(mut self, v: f64) -> Self {
        self.flip_probability = v; self
    }

    pub fn with_sample_size(mut self, v: usize) -> Self {
        self.sample_size = v; self
    }

}

impl fmt::Display for RandomizedResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RandomizedResponse({:?})", self.true_probability)
    }
}

#[derive(Debug, Clone)]
pub struct RandomizedResponseBuilder {
    true_probability: f64,
    flip_probability: f64,
    sample_size: usize,
}

impl RandomizedResponseBuilder {
    pub fn new() -> Self {
        RandomizedResponseBuilder {
            true_probability: 0.0,
            flip_probability: 0.0,
            sample_size: 0,
        }
    }

    pub fn true_probability(mut self, v: f64) -> Self { self.true_probability = v; self }
    pub fn flip_probability(mut self, v: f64) -> Self { self.flip_probability = v; self }
    pub fn sample_size(mut self, v: usize) -> Self { self.sample_size = v; self }
}

#[derive(Debug, Clone)]
pub struct NoiseCalibration {
    pub target_epsilon: f64,
    pub noise_scale: f64,
    pub distribution: String,
    pub calibrated: bool,
}

impl NoiseCalibration {
    pub fn new(target_epsilon: f64, noise_scale: f64, distribution: String, calibrated: bool) -> Self {
        NoiseCalibration { target_epsilon, noise_scale, distribution, calibrated }
    }

    pub fn get_target_epsilon(&self) -> f64 {
        self.target_epsilon
    }

    pub fn get_noise_scale(&self) -> f64 {
        self.noise_scale
    }

    pub fn get_distribution(&self) -> &str {
        &self.distribution
    }

    pub fn get_calibrated(&self) -> bool {
        self.calibrated
    }

    pub fn with_target_epsilon(mut self, v: f64) -> Self {
        self.target_epsilon = v; self
    }

    pub fn with_noise_scale(mut self, v: f64) -> Self {
        self.noise_scale = v; self
    }

    pub fn with_distribution(mut self, v: impl Into<String>) -> Self {
        self.distribution = v.into(); self
    }

    pub fn with_calibrated(mut self, v: bool) -> Self {
        self.calibrated = v; self
    }

}

impl fmt::Display for NoiseCalibration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NoiseCalibration({:?})", self.target_epsilon)
    }
}

#[derive(Debug, Clone)]
pub struct NoiseCalibrationBuilder {
    target_epsilon: f64,
    noise_scale: f64,
    distribution: String,
    calibrated: bool,
}

impl NoiseCalibrationBuilder {
    pub fn new() -> Self {
        NoiseCalibrationBuilder {
            target_epsilon: 0.0,
            noise_scale: 0.0,
            distribution: String::new(),
            calibrated: false,
        }
    }

    pub fn target_epsilon(mut self, v: f64) -> Self { self.target_epsilon = v; self }
    pub fn noise_scale(mut self, v: f64) -> Self { self.noise_scale = v; self }
    pub fn distribution(mut self, v: impl Into<String>) -> Self { self.distribution = v.into(); self }
    pub fn calibrated(mut self, v: bool) -> Self { self.calibrated = v; self }
}

#[derive(Debug, Clone)]
pub struct LaplaceMechanism {
    pub sensitivity: f64,
    pub epsilon: f64,
    pub noise_scale: f64,
}

impl LaplaceMechanism {
    pub fn new(sensitivity: f64, epsilon: f64, noise_scale: f64) -> Self {
        LaplaceMechanism { sensitivity, epsilon, noise_scale }
    }

    pub fn get_sensitivity(&self) -> f64 {
        self.sensitivity
    }

    pub fn get_epsilon(&self) -> f64 {
        self.epsilon
    }

    pub fn get_noise_scale(&self) -> f64 {
        self.noise_scale
    }

    pub fn with_sensitivity(mut self, v: f64) -> Self {
        self.sensitivity = v; self
    }

    pub fn with_epsilon(mut self, v: f64) -> Self {
        self.epsilon = v; self
    }

    pub fn with_noise_scale(mut self, v: f64) -> Self {
        self.noise_scale = v; self
    }

}

impl fmt::Display for LaplaceMechanism {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LaplaceMechanism({:?})", self.sensitivity)
    }
}

#[derive(Debug, Clone)]
pub struct LaplaceMechanismBuilder {
    sensitivity: f64,
    epsilon: f64,
    noise_scale: f64,
}

impl LaplaceMechanismBuilder {
    pub fn new() -> Self {
        LaplaceMechanismBuilder {
            sensitivity: 0.0,
            epsilon: 0.0,
            noise_scale: 0.0,
        }
    }

    pub fn sensitivity(mut self, v: f64) -> Self { self.sensitivity = v; self }
    pub fn epsilon(mut self, v: f64) -> Self { self.epsilon = v; self }
    pub fn noise_scale(mut self, v: f64) -> Self { self.noise_scale = v; self }
}

#[derive(Debug, Clone)]
pub struct ExponentialMechanism {
    pub epsilon: f64,
    pub sensitivity: f64,
    pub utility_scores: Vec<f64>,
    pub probabilities: Vec<f64>,
}

impl ExponentialMechanism {
    pub fn new(epsilon: f64, sensitivity: f64, utility_scores: Vec<f64>, probabilities: Vec<f64>) -> Self {
        ExponentialMechanism { epsilon, sensitivity, utility_scores, probabilities }
    }

    pub fn get_epsilon(&self) -> f64 {
        self.epsilon
    }

    pub fn get_sensitivity(&self) -> f64 {
        self.sensitivity
    }

    pub fn utility_scores_len(&self) -> usize {
        self.utility_scores.len()
    }

    pub fn utility_scores_is_empty(&self) -> bool {
        self.utility_scores.is_empty()
    }

    pub fn probabilities_len(&self) -> usize {
        self.probabilities.len()
    }

    pub fn probabilities_is_empty(&self) -> bool {
        self.probabilities.is_empty()
    }

    pub fn with_epsilon(mut self, v: f64) -> Self {
        self.epsilon = v; self
    }

    pub fn with_sensitivity(mut self, v: f64) -> Self {
        self.sensitivity = v; self
    }

}

impl fmt::Display for ExponentialMechanism {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ExponentialMechanism({:?})", self.epsilon)
    }
}

#[derive(Debug, Clone)]
pub struct ExponentialMechanismBuilder {
    epsilon: f64,
    sensitivity: f64,
    utility_scores: Vec<f64>,
    probabilities: Vec<f64>,
}

impl ExponentialMechanismBuilder {
    pub fn new() -> Self {
        ExponentialMechanismBuilder {
            epsilon: 0.0,
            sensitivity: 0.0,
            utility_scores: Vec::new(),
            probabilities: Vec::new(),
        }
    }

    pub fn epsilon(mut self, v: f64) -> Self { self.epsilon = v; self }
    pub fn sensitivity(mut self, v: f64) -> Self { self.sensitivity = v; self }
    pub fn utility_scores(mut self, v: Vec<f64>) -> Self { self.utility_scores = v; self }
    pub fn probabilities(mut self, v: Vec<f64>) -> Self { self.probabilities = v; self }
}

#[derive(Debug, Clone)]
pub struct CompositionTheorem {
    pub num_queries: usize,
    pub per_query_epsilon: f64,
    pub total_epsilon: f64,
    pub method: String,
}

impl CompositionTheorem {
    pub fn new(num_queries: usize, per_query_epsilon: f64, total_epsilon: f64, method: String) -> Self {
        CompositionTheorem { num_queries, per_query_epsilon, total_epsilon, method }
    }

    pub fn get_num_queries(&self) -> usize {
        self.num_queries
    }

    pub fn get_per_query_epsilon(&self) -> f64 {
        self.per_query_epsilon
    }

    pub fn get_total_epsilon(&self) -> f64 {
        self.total_epsilon
    }

    pub fn get_method(&self) -> &str {
        &self.method
    }

    pub fn with_num_queries(mut self, v: usize) -> Self {
        self.num_queries = v; self
    }

    pub fn with_per_query_epsilon(mut self, v: f64) -> Self {
        self.per_query_epsilon = v; self
    }

    pub fn with_total_epsilon(mut self, v: f64) -> Self {
        self.total_epsilon = v; self
    }

    pub fn with_method(mut self, v: impl Into<String>) -> Self {
        self.method = v.into(); self
    }

}

impl fmt::Display for CompositionTheorem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CompositionTheorem({:?})", self.num_queries)
    }
}

#[derive(Debug, Clone)]
pub struct CompositionTheoremBuilder {
    num_queries: usize,
    per_query_epsilon: f64,
    total_epsilon: f64,
    method: String,
}

impl CompositionTheoremBuilder {
    pub fn new() -> Self {
        CompositionTheoremBuilder {
            num_queries: 0,
            per_query_epsilon: 0.0,
            total_epsilon: 0.0,
            method: String::new(),
        }
    }

    pub fn num_queries(mut self, v: usize) -> Self { self.num_queries = v; self }
    pub fn per_query_epsilon(mut self, v: f64) -> Self { self.per_query_epsilon = v; self }
    pub fn total_epsilon(mut self, v: f64) -> Self { self.total_epsilon = v; self }
    pub fn method(mut self, v: impl Into<String>) -> Self { self.method = v.into(); self }
}

#[derive(Debug, Clone)]
pub struct ErasureChannel {
    pub erasure_probability: f64,
    pub capacity: f64,
    pub input_size: usize,
}

impl ErasureChannel {
    pub fn new(erasure_probability: f64, capacity: f64, input_size: usize) -> Self {
        ErasureChannel { erasure_probability, capacity, input_size }
    }

    pub fn get_erasure_probability(&self) -> f64 {
        self.erasure_probability
    }

    pub fn get_capacity(&self) -> f64 {
        self.capacity
    }

    pub fn get_input_size(&self) -> usize {
        self.input_size
    }

    pub fn with_erasure_probability(mut self, v: f64) -> Self {
        self.erasure_probability = v; self
    }

    pub fn with_capacity(mut self, v: f64) -> Self {
        self.capacity = v; self
    }

    pub fn with_input_size(mut self, v: usize) -> Self {
        self.input_size = v; self
    }

}

impl fmt::Display for ErasureChannel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ErasureChannel({:?})", self.erasure_probability)
    }
}

#[derive(Debug, Clone)]
pub struct ErasureChannelBuilder {
    erasure_probability: f64,
    capacity: f64,
    input_size: usize,
}

impl ErasureChannelBuilder {
    pub fn new() -> Self {
        ErasureChannelBuilder {
            erasure_probability: 0.0,
            capacity: 0.0,
            input_size: 0,
        }
    }

    pub fn erasure_probability(mut self, v: f64) -> Self { self.erasure_probability = v; self }
    pub fn capacity(mut self, v: f64) -> Self { self.capacity = v; self }
    pub fn input_size(mut self, v: usize) -> Self { self.input_size = v; self }
}

#[derive(Debug, Clone)]
pub struct ZChannel {
    pub crossover_probability: f64,
    pub capacity: f64,
}

impl ZChannel {
    pub fn new(crossover_probability: f64, capacity: f64) -> Self {
        ZChannel { crossover_probability, capacity }
    }

    pub fn get_crossover_probability(&self) -> f64 {
        self.crossover_probability
    }

    pub fn get_capacity(&self) -> f64 {
        self.capacity
    }

    pub fn with_crossover_probability(mut self, v: f64) -> Self {
        self.crossover_probability = v; self
    }

    pub fn with_capacity(mut self, v: f64) -> Self {
        self.capacity = v; self
    }

}

impl fmt::Display for ZChannel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ZChannel({:?})", self.crossover_probability)
    }
}

#[derive(Debug, Clone)]
pub struct ZChannelBuilder {
    crossover_probability: f64,
    capacity: f64,
}

impl ZChannelBuilder {
    pub fn new() -> Self {
        ZChannelBuilder {
            crossover_probability: 0.0,
            capacity: 0.0,
        }
    }

    pub fn crossover_probability(mut self, v: f64) -> Self { self.crossover_probability = v; self }
    pub fn capacity(mut self, v: f64) -> Self { self.capacity = v; self }
}

#[derive(Debug, Clone)]
pub struct FanoInequality {
    pub error_probability: f64,
    pub alphabet_size: usize,
    pub entropy_bound: f64,
}

impl FanoInequality {
    pub fn new(error_probability: f64, alphabet_size: usize, entropy_bound: f64) -> Self {
        FanoInequality { error_probability, alphabet_size, entropy_bound }
    }

    pub fn get_error_probability(&self) -> f64 {
        self.error_probability
    }

    pub fn get_alphabet_size(&self) -> usize {
        self.alphabet_size
    }

    pub fn get_entropy_bound(&self) -> f64 {
        self.entropy_bound
    }

    pub fn with_error_probability(mut self, v: f64) -> Self {
        self.error_probability = v; self
    }

    pub fn with_alphabet_size(mut self, v: usize) -> Self {
        self.alphabet_size = v; self
    }

    pub fn with_entropy_bound(mut self, v: f64) -> Self {
        self.entropy_bound = v; self
    }

}

impl fmt::Display for FanoInequality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FanoInequality({:?})", self.error_probability)
    }
}

#[derive(Debug, Clone)]
pub struct FanoInequalityBuilder {
    error_probability: f64,
    alphabet_size: usize,
    entropy_bound: f64,
}

impl FanoInequalityBuilder {
    pub fn new() -> Self {
        FanoInequalityBuilder {
            error_probability: 0.0,
            alphabet_size: 0,
            entropy_bound: 0.0,
        }
    }

    pub fn error_probability(mut self, v: f64) -> Self { self.error_probability = v; self }
    pub fn alphabet_size(mut self, v: usize) -> Self { self.alphabet_size = v; self }
    pub fn entropy_bound(mut self, v: f64) -> Self { self.entropy_bound = v; self }
}

#[derive(Debug, Clone)]
pub struct RateDistortion {
    pub rate: f64,
    pub distortion: f64,
    pub source_entropy: f64,
}

impl RateDistortion {
    pub fn new(rate: f64, distortion: f64, source_entropy: f64) -> Self {
        RateDistortion { rate, distortion, source_entropy }
    }

    pub fn get_rate(&self) -> f64 {
        self.rate
    }

    pub fn get_distortion(&self) -> f64 {
        self.distortion
    }

    pub fn get_source_entropy(&self) -> f64 {
        self.source_entropy
    }

    pub fn with_rate(mut self, v: f64) -> Self {
        self.rate = v; self
    }

    pub fn with_distortion(mut self, v: f64) -> Self {
        self.distortion = v; self
    }

    pub fn with_source_entropy(mut self, v: f64) -> Self {
        self.source_entropy = v; self
    }

}

impl fmt::Display for RateDistortion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RateDistortion({:?})", self.rate)
    }
}

#[derive(Debug, Clone)]
pub struct RateDistortionBuilder {
    rate: f64,
    distortion: f64,
    source_entropy: f64,
}

impl RateDistortionBuilder {
    pub fn new() -> Self {
        RateDistortionBuilder {
            rate: 0.0,
            distortion: 0.0,
            source_entropy: 0.0,
        }
    }

    pub fn rate(mut self, v: f64) -> Self { self.rate = v; self }
    pub fn distortion(mut self, v: f64) -> Self { self.distortion = v; self }
    pub fn source_entropy(mut self, v: f64) -> Self { self.source_entropy = v; self }
}

#[derive(Debug, Clone)]
pub struct SourceCoding {
    pub source_entropy: f64,
    pub code_rate: f64,
    pub redundancy: f64,
    pub efficiency: f64,
}

impl SourceCoding {
    pub fn new(source_entropy: f64, code_rate: f64, redundancy: f64, efficiency: f64) -> Self {
        SourceCoding { source_entropy, code_rate, redundancy, efficiency }
    }

    pub fn get_source_entropy(&self) -> f64 {
        self.source_entropy
    }

    pub fn get_code_rate(&self) -> f64 {
        self.code_rate
    }

    pub fn get_redundancy(&self) -> f64 {
        self.redundancy
    }

    pub fn get_efficiency(&self) -> f64 {
        self.efficiency
    }

    pub fn with_source_entropy(mut self, v: f64) -> Self {
        self.source_entropy = v; self
    }

    pub fn with_code_rate(mut self, v: f64) -> Self {
        self.code_rate = v; self
    }

    pub fn with_redundancy(mut self, v: f64) -> Self {
        self.redundancy = v; self
    }

    pub fn with_efficiency(mut self, v: f64) -> Self {
        self.efficiency = v; self
    }

}

impl fmt::Display for SourceCoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SourceCoding({:?})", self.source_entropy)
    }
}

#[derive(Debug, Clone)]
pub struct SourceCodingBuilder {
    source_entropy: f64,
    code_rate: f64,
    redundancy: f64,
    efficiency: f64,
}

impl SourceCodingBuilder {
    pub fn new() -> Self {
        SourceCodingBuilder {
            source_entropy: 0.0,
            code_rate: 0.0,
            redundancy: 0.0,
            efficiency: 0.0,
        }
    }

    pub fn source_entropy(mut self, v: f64) -> Self { self.source_entropy = v; self }
    pub fn code_rate(mut self, v: f64) -> Self { self.code_rate = v; self }
    pub fn redundancy(mut self, v: f64) -> Self { self.redundancy = v; self }
    pub fn efficiency(mut self, v: f64) -> Self { self.efficiency = v; self }
}

#[derive(Debug, Clone)]
pub struct ChannelCapacity {
    pub capacity_bits: f64,
    pub input_distribution: Vec<f64>,
    pub transition_matrix: Vec<Vec<f64>>,
}

impl ChannelCapacity {
    pub fn new(capacity_bits: f64, input_distribution: Vec<f64>, transition_matrix: Vec<Vec<f64>>) -> Self {
        ChannelCapacity { capacity_bits, input_distribution, transition_matrix }
    }

    pub fn get_capacity_bits(&self) -> f64 {
        self.capacity_bits
    }

    pub fn input_distribution_len(&self) -> usize {
        self.input_distribution.len()
    }

    pub fn input_distribution_is_empty(&self) -> bool {
        self.input_distribution.is_empty()
    }

    pub fn transition_matrix_len(&self) -> usize {
        self.transition_matrix.len()
    }

    pub fn transition_matrix_is_empty(&self) -> bool {
        self.transition_matrix.is_empty()
    }

    pub fn with_capacity_bits(mut self, v: f64) -> Self {
        self.capacity_bits = v; self
    }

}

impl fmt::Display for ChannelCapacity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ChannelCapacity({:?})", self.capacity_bits)
    }
}

#[derive(Debug, Clone)]
pub struct ChannelCapacityBuilder {
    capacity_bits: f64,
    input_distribution: Vec<f64>,
    transition_matrix: Vec<Vec<f64>>,
}

impl ChannelCapacityBuilder {
    pub fn new() -> Self {
        ChannelCapacityBuilder {
            capacity_bits: 0.0,
            input_distribution: Vec::new(),
            transition_matrix: Vec::new(),
        }
    }

    pub fn capacity_bits(mut self, v: f64) -> Self { self.capacity_bits = v; self }
    pub fn input_distribution(mut self, v: Vec<f64>) -> Self { self.input_distribution = v; self }
    pub fn transition_matrix(mut self, v: Vec<Vec<f64>>) -> Self { self.transition_matrix = v; self }
}

#[derive(Debug, Clone)]
pub struct SidechanAnalysis {
    pub data: Vec<Vec<f64>>,
    pub size: usize,
    pub computed: bool,
    pub label: String,
    pub threshold: f64,
}

impl SidechanAnalysis {
    pub fn new(size: usize) -> Self {
        let data = vec![vec![0.0; size]; size];
        SidechanAnalysis { data, size, computed: false, label: "Sidechan".to_string(), threshold: 0.01 }
    }

    pub fn with_threshold(mut self, t: f64) -> Self {
        self.threshold = t; self
    }

    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        if i < self.size && j < self.size { self.data[i][j] = v; }
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        if i < self.size && j < self.size { self.data[i][j] } else { 0.0 }
    }

    pub fn row_sum(&self, i: usize) -> f64 {
        if i < self.size { self.data[i].iter().sum() } else { 0.0 }
    }

    pub fn col_sum(&self, j: usize) -> f64 {
        if j < self.size { (0..self.size).map(|i| self.data[i][j]).sum() } else { 0.0 }
    }

    pub fn total_sum(&self) -> f64 {
        self.data.iter().flat_map(|r| r.iter()).sum()
    }

    pub fn max_value(&self) -> f64 {
        self.data.iter().flat_map(|r| r.iter()).cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn min_value(&self) -> f64 {
        self.data.iter().flat_map(|r| r.iter()).cloned().fold(f64::INFINITY, f64::min)
    }

    pub fn above_threshold(&self) -> Vec<(usize, usize, f64)> {
        let mut result = Vec::new();
        for i in 0..self.size {
            for j in 0..self.size {
                if self.data[i][j] > self.threshold {
                    result.push((i, j, self.data[i][j]));
                }
            }
        }
        result
    }

    pub fn normalize(&mut self) {
        let total = self.total_sum();
        if total > 0.0 {
            for i in 0..self.size {
                for j in 0..self.size {
                    self.data[i][j] /= total;
                }
            }
        }
        self.computed = true;
    }

    pub fn transpose(&self) -> Self {
        let mut result = Self::new(self.size);
        for i in 0..self.size {
            for j in 0..self.size {
                result.data[i][j] = self.data[j][i];
            }
        }
        result
    }

    pub fn multiply(&self, other: &Self) -> Self {
        assert_eq!(self.size, other.size);
        let mut result = Self::new(self.size);
        for i in 0..self.size {
            for j in 0..self.size {
                let mut sum = 0.0;
                for k in 0..self.size {
                    sum += self.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        result
    }

    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().flat_map(|r| r.iter()).map(|&v| v * v).sum::<f64>().sqrt()
    }

    pub fn trace(&self) -> f64 {
        (0..self.size).map(|i| self.data[i][i]).sum()
    }

    pub fn diagonal(&self) -> Vec<f64> {
        (0..self.size).map(|i| self.data[i][i]).collect()
    }

    pub fn is_symmetric(&self) -> bool {
        for i in 0..self.size {
            for j in 0..self.size {
                if (self.data[i][j] - self.data[j][i]).abs() > 1e-10 { return false; }
            }
        }
        true
    }

}

impl fmt::Display for SidechanAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SidechanAnalysis({:?})", self.data)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SidechanStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Skipped,
}

impl fmt::Display for SidechanStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SidechanStatus::Pending => write!(f, "pending"),
            SidechanStatus::InProgress => write!(f, "inprogress"),
            SidechanStatus::Completed => write!(f, "completed"),
            SidechanStatus::Failed => write!(f, "failed"),
            SidechanStatus::Skipped => write!(f, "skipped"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SidechanPriority {
    Critical,
    High,
    Medium,
    Low,
    None,
}

impl fmt::Display for SidechanPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SidechanPriority::Critical => write!(f, "critical"),
            SidechanPriority::High => write!(f, "high"),
            SidechanPriority::Medium => write!(f, "medium"),
            SidechanPriority::Low => write!(f, "low"),
            SidechanPriority::None => write!(f, "none"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SidechanMode {
    Strict,
    Relaxed,
    Permissive,
    Custom,
}

impl fmt::Display for SidechanMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SidechanMode::Strict => write!(f, "strict"),
            SidechanMode::Relaxed => write!(f, "relaxed"),
            SidechanMode::Permissive => write!(f, "permissive"),
            SidechanMode::Custom => write!(f, "custom"),
        }
    }
}

pub fn sidechan_mean(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    data.iter().sum::<f64>() / data.len() as f64
}

pub fn sidechan_variance(data: &[f64]) -> f64 {
    if data.len() < 2 { return 0.0; }
    let mean = sidechan_mean(data);
    data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (data.len() - 1) as f64
}

pub fn sidechan_std_dev(data: &[f64]) -> f64 {
    sidechan_variance(data).sqrt()
}

pub fn sidechan_median(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n % 2 == 0 { (sorted[n/2 - 1] + sorted[n/2]) / 2.0 } else { sorted[n/2] }
}

/// Percentile calculator for SideChan.
pub fn sidechan_percentile_at(data: &[f64], p: f64) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = p / 100.0 * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi { sorted[lo] }
    else { sorted[lo] * (hi as f64 - idx) + sorted[hi] * (idx - lo as f64) }
}

pub fn sidechan_entropy(data: &[f64]) -> f64 {
    let total: f64 = data.iter().sum();
    if total <= 0.0 { return 0.0; }
    let mut h = 0.0;
    for &x in data {
        if x > 0.0 {
            let p = x / total;
            h -= p * p.ln();
        }
    }
    h
}

pub fn sidechan_gini(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let n = data.len();
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sum: f64 = sorted.iter().sum();
    if sum == 0.0 { return 0.0; }
    let mut g = 0.0;
    for (i, &x) in sorted.iter().enumerate() {
        g += (2.0 * (i + 1) as f64 - n as f64 - 1.0) * x;
    }
    g / (n as f64 * sum)
}

pub fn sidechan_covariance(data: &[f64]) -> f64 {
    if data.len() < 4 { return 0.0; }
    let n = data.len() / 2;
    let x: Vec<f64> = data[..n].to_vec();
    let y: Vec<f64> = data[n..2*n].to_vec();
    let mx = sidechan_mean(&x);
    let my = sidechan_mean(&y);
    x.iter().zip(y.iter()).map(|(&xi, &yi)| (xi - mx) * (yi - my)).sum::<f64>() / (n - 1) as f64
}

pub fn sidechan_correlation(data: &[f64]) -> f64 {
    if data.len() < 4 { return 0.0; }
    let n = data.len() / 2;
    let cov = sidechan_covariance(data);
    let sx = sidechan_std_dev(&data[..n]);
    let sy = sidechan_std_dev(&data[n..2*n]);
    if sx * sy == 0.0 { 0.0 } else { cov / (sx * sy) }
}

pub fn sidechan_excess_kurtosis(data: &[f64]) -> f64 {
    if data.len() < 4 { return 0.0; }
    let m = sidechan_mean(data);
    let s = sidechan_std_dev(data);
    if s == 0.0 { return 0.0; }
    let n = data.len() as f64;
    let k = data.iter().map(|&x| ((x - m) / s).powi(4)).sum::<f64>() / n;
    k - 3.0
}

pub fn sidechan_sample_skewness(data: &[f64]) -> f64 {
    if data.len() < 3 { return 0.0; }
    let m = sidechan_mean(data);
    let s = sidechan_std_dev(data);
    if s == 0.0 { return 0.0; }
    let n = data.len() as f64;
    data.iter().map(|&x| ((x - m) / s).powi(3)).sum::<f64>() / n
}

pub fn sidechan_harmmean(data: &[f64]) -> f64 {
    if data.is_empty() || data.iter().any(|&x| x <= 0.0) { return 0.0; }
    let n = data.len() as f64;
    n / data.iter().map(|&x| 1.0 / x).sum::<f64>()
}

pub fn sidechan_geomean(data: &[f64]) -> f64 {
    if data.is_empty() || data.iter().any(|&x| x <= 0.0) { return 0.0; }
    let n = data.len() as f64;
    (data.iter().map(|&x| x.ln()).sum::<f64>() / n).exp()
}

/// Iterator over sidechan analysis results.
#[derive(Debug, Clone)]
pub struct SidechanResultIterator {
    items: Vec<(usize, f64)>,
    position: usize,
}

impl SidechanResultIterator {
    pub fn new(items: Vec<(usize, f64)>) -> Self {
        SidechanResultIterator { items, position: 0 }
    }
    pub fn remaining(&self) -> usize { self.items.len() - self.position }
}

impl Iterator for SidechanResultIterator {
    type Item = (usize, f64);
    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.items.len() {
            let item = self.items[self.position];
            self.position += 1;
            Some(item)
        } else { None }
    }
}

/// Convert DifferentialPrivacy description to a summary string.
pub fn differentialprivacy_to_summary(item: &DifferentialPrivacy) -> String {
    format!("DifferentialPrivacy: {:?}", item)
}

/// Convert RandomizedResponse description to a summary string.
pub fn randomizedresponse_to_summary(item: &RandomizedResponse) -> String {
    format!("RandomizedResponse: {:?}", item)
}

/// Convert NoiseCalibration description to a summary string.
pub fn noisecalibration_to_summary(item: &NoiseCalibration) -> String {
    format!("NoiseCalibration: {:?}", item)
}

/// Convert LaplaceMechanism description to a summary string.
pub fn laplacemechanism_to_summary(item: &LaplaceMechanism) -> String {
    format!("LaplaceMechanism: {:?}", item)
}

/// Convert ExponentialMechanism description to a summary string.
pub fn exponentialmechanism_to_summary(item: &ExponentialMechanism) -> String {
    format!("ExponentialMechanism: {:?}", item)
}

/// Convert CompositionTheorem description to a summary string.
pub fn compositiontheorem_to_summary(item: &CompositionTheorem) -> String {
    format!("CompositionTheorem: {:?}", item)
}

/// Convert ErasureChannel description to a summary string.
pub fn erasurechannel_to_summary(item: &ErasureChannel) -> String {
    format!("ErasureChannel: {:?}", item)
}

/// Convert ZChannel description to a summary string.
pub fn zchannel_to_summary(item: &ZChannel) -> String {
    format!("ZChannel: {:?}", item)
}

/// Convert FanoInequality description to a summary string.
pub fn fanoinequality_to_summary(item: &FanoInequality) -> String {
    format!("FanoInequality: {:?}", item)
}

/// Convert RateDistortion description to a summary string.
pub fn ratedistortion_to_summary(item: &RateDistortion) -> String {
    format!("RateDistortion: {:?}", item)
}

/// Convert SourceCoding description to a summary string.
pub fn sourcecoding_to_summary(item: &SourceCoding) -> String {
    format!("SourceCoding: {:?}", item)
}

/// Batch processor for sidechan operations.
#[derive(Debug, Clone)]
pub struct SidechanBatchProcessor {
    pub batch_size: usize,
    pub processed: usize,
    pub errors: Vec<String>,
    pub results: Vec<f64>,
}

impl SidechanBatchProcessor {
    pub fn new(batch_size: usize) -> Self {
        SidechanBatchProcessor { batch_size, processed: 0, errors: Vec::new(), results: Vec::new() }
    }
    pub fn process_batch(&mut self, data: &[f64]) {
        for chunk in data.chunks(self.batch_size) {
            let sum: f64 = chunk.iter().sum();
            self.results.push(sum / chunk.len() as f64);
            self.processed += chunk.len();
        }
    }
    pub fn success_rate(&self) -> f64 {
        if self.processed == 0 { return 0.0; }
        1.0 - (self.errors.len() as f64 / self.processed as f64)
    }
    pub fn average_result(&self) -> f64 {
        if self.results.is_empty() { return 0.0; }
        self.results.iter().sum::<f64>() / self.results.len() as f64
    }
    pub fn reset(&mut self) { self.processed = 0; self.errors.clear(); self.results.clear(); }
}

impl fmt::Display for SidechanBatchProcessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SidechanBatch(processed={}, errors={})", self.processed, self.errors.len())
    }
}

/// Detailed report for sidechan analysis.
#[derive(Debug, Clone)]
pub struct SidechanReport {
    pub title: String,
    pub sections: Vec<(String, Vec<String>)>,
    pub metrics: Vec<(String, f64)>,
    pub warnings: Vec<String>,
    pub timestamp: u64,
}

impl SidechanReport {
    pub fn new(title: impl Into<String>) -> Self {
        SidechanReport { title: title.into(), sections: Vec::new(), metrics: Vec::new(), warnings: Vec::new(), timestamp: 0 }
    }
    pub fn add_section(&mut self, name: impl Into<String>, content: Vec<String>) {
        self.sections.push((name.into(), content));
    }
    pub fn add_metric(&mut self, name: impl Into<String>, value: f64) {
        self.metrics.push((name.into(), value));
    }
    pub fn add_warning(&mut self, warning: impl Into<String>) {
        self.warnings.push(warning.into());
    }
    pub fn total_metrics(&self) -> usize { self.metrics.len() }
    pub fn has_warnings(&self) -> bool { !self.warnings.is_empty() }
    pub fn metric_sum(&self) -> f64 { self.metrics.iter().map(|(_, v)| v).sum() }
    pub fn render_text(&self) -> String {
        let mut out = format!("=== {} ===\n", self.title);
        for (name, content) in &self.sections {
            out.push_str(&format!("\n--- {} ---\n", name));
            for line in content {
                out.push_str(&format!("  {}\n", line));
            }
        }
        out.push_str("\nMetrics:\n");
        for (name, val) in &self.metrics {
            out.push_str(&format!("  {}: {:.4}\n", name, val));
        }
        if !self.warnings.is_empty() {
            out.push_str("\nWarnings:\n");
            for w in &self.warnings {
                out.push_str(&format!("  ! {}\n", w));
            }
        }
        out
    }
}

impl fmt::Display for SidechanReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SidechanReport({})", self.title)
    }
}

/// Configuration for sidechan analysis.
#[derive(Debug, Clone)]
pub struct SidechanConfig {
    pub verbose: bool,
    pub max_iterations: usize,
    pub tolerance: f64,
    pub timeout_ms: u64,
    pub parallel: bool,
    pub output_format: String,
}

impl SidechanConfig {
    pub fn default_config() -> Self {
        SidechanConfig {
            verbose: false, max_iterations: 1000, tolerance: 1e-6,
            timeout_ms: 30000, parallel: false, output_format: "text".to_string(),
        }
    }
    pub fn with_verbose(mut self, v: bool) -> Self { self.verbose = v; self }
    pub fn with_max_iterations(mut self, n: usize) -> Self { self.max_iterations = n; self }
    pub fn with_tolerance(mut self, t: f64) -> Self { self.tolerance = t; self }
    pub fn with_timeout(mut self, ms: u64) -> Self { self.timeout_ms = ms; self }
    pub fn with_parallel(mut self, p: bool) -> Self { self.parallel = p; self }
    pub fn with_output_format(mut self, fmt: impl Into<String>) -> Self { self.output_format = fmt.into(); self }
}

impl fmt::Display for SidechanConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SidechanConfig(iters={}, tol={:.0e})", self.max_iterations, self.tolerance)
    }
}

/// Histogram for sidechan data distribution.
#[derive(Debug, Clone)]
pub struct SidechanHistogram {
    pub bins: Vec<usize>,
    pub bin_edges: Vec<f64>,
    pub total_count: usize,
}

impl SidechanHistogram {
    pub fn from_data(data: &[f64], num_bins: usize) -> Self {
        if data.is_empty() || num_bins == 0 {
            return SidechanHistogram { bins: Vec::new(), bin_edges: Vec::new(), total_count: 0 };
        }
        let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_val - min_val;
        let bin_width = if range == 0.0 { 1.0 } else { range / num_bins as f64 };
        let mut bins = vec![0usize; num_bins];
        let mut bin_edges = Vec::with_capacity(num_bins + 1);
        for i in 0..=num_bins { bin_edges.push(min_val + i as f64 * bin_width); }
        for &val in data {
            let idx = ((val - min_val) / bin_width).floor() as usize;
            let idx = idx.min(num_bins - 1);
            bins[idx] += 1;
        }
        SidechanHistogram { bins, bin_edges, total_count: data.len() }
    }
    pub fn num_bins(&self) -> usize { self.bins.len() }
    pub fn max_bin(&self) -> usize { self.bins.iter().cloned().max().unwrap_or(0) }
    pub fn mean_bin(&self) -> f64 {
        if self.bins.is_empty() { return 0.0; }
        self.bins.iter().sum::<usize>() as f64 / self.bins.len() as f64
    }
    pub fn render_ascii(&self, width: usize) -> String {
        let max = self.max_bin();
        let mut out = String::new();
        for (i, &count) in self.bins.iter().enumerate() {
            let bar_len = if max == 0 { 0 } else { count * width / max };
            let bar: String = std::iter::repeat('#').take(bar_len).collect();
            out.push_str(&format!("[{:.2}-{:.2}] {} {}\n",
                self.bin_edges[i], self.bin_edges[i + 1], bar, count));
        }
        out
    }
}

impl fmt::Display for SidechanHistogram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Histogram(bins={}, total={})", self.num_bins(), self.total_count)
    }
}

/// Adjacency/weight matrix for sidechan graph analysis.
#[derive(Debug, Clone)]
pub struct SidechanGraph {
    pub adjacency: Vec<Vec<bool>>,
    pub weights: Vec<Vec<f64>>,
    pub node_count: usize,
    pub edge_count: usize,
    pub node_labels: Vec<String>,
}

impl SidechanGraph {
    pub fn new(n: usize) -> Self {
        SidechanGraph {
            adjacency: vec![vec![false; n]; n],
            weights: vec![vec![0.0; n]; n],
            node_count: n, edge_count: 0,
            node_labels: (0..n).map(|i| format!("n{}", i)).collect(),
        }
    }
    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) {
        if from < self.node_count && to < self.node_count && !self.adjacency[from][to] {
            self.adjacency[from][to] = true;
            self.weights[from][to] = weight;
            self.edge_count += 1;
        }
    }
    pub fn remove_edge(&mut self, from: usize, to: usize) {
        if from < self.node_count && to < self.node_count && self.adjacency[from][to] {
            self.adjacency[from][to] = false;
            self.weights[from][to] = 0.0;
            self.edge_count -= 1;
        }
    }
    pub fn has_edge(&self, from: usize, to: usize) -> bool {
        from < self.node_count && to < self.node_count && self.adjacency[from][to]
    }
    pub fn weight(&self, from: usize, to: usize) -> f64 { self.weights[from][to] }
    pub fn out_degree(&self, node: usize) -> usize {
        (0..self.node_count).filter(|&j| self.adjacency[node][j]).count()
    }
    pub fn in_degree(&self, node: usize) -> usize {
        (0..self.node_count).filter(|&i| self.adjacency[i][node]).count()
    }
    pub fn neighbors(&self, node: usize) -> Vec<usize> {
        (0..self.node_count).filter(|&j| self.adjacency[node][j]).collect()
    }
    pub fn density(&self) -> f64 {
        if self.node_count <= 1 { return 0.0; }
        self.edge_count as f64 / (self.node_count * (self.node_count - 1)) as f64
    }
    pub fn is_acyclic(&self) -> bool {
        let n = self.node_count;
        let mut visited = vec![0u8; n];
        fn dfs_cycle_sidechan(v: usize, adj: &[Vec<bool>], visited: &mut [u8]) -> bool {
            visited[v] = 1;
            for w in 0..adj.len() { if adj[v][w] {
                if visited[w] == 1 { return true; }
                if visited[w] == 0 && dfs_cycle_sidechan(w, adj, visited) { return true; }
            }}
            visited[v] = 2; false
        }
        for i in 0..n {
            if visited[i] == 0 && dfs_cycle_sidechan(i, &self.adjacency, &mut visited) { return false; }
        }
        true
    }
    pub fn topological_sort(&self) -> Option<Vec<usize>> {
        let n = self.node_count;
        let mut in_deg: Vec<usize> = (0..n).map(|j| self.in_degree(j)).collect();
        let mut queue: Vec<usize> = (0..n).filter(|&i| in_deg[i] == 0).collect();
        let mut result = Vec::new();
        while let Some(v) = queue.pop() {
            result.push(v);
            for j in 0..n { if self.adjacency[v][j] {
                in_deg[j] -= 1;
                if in_deg[j] == 0 { queue.push(j); }
            }}
        }
        if result.len() == n { Some(result) } else { None }
    }
    pub fn shortest_path_dijkstra(&self, start: usize) -> Vec<f64> {
        let n = self.node_count;
        let mut dist = vec![f64::INFINITY; n];
        let mut visited = vec![false; n];
        dist[start] = 0.0;
        for _ in 0..n {
            let mut u = None;
            let mut min_d = f64::INFINITY;
            for v in 0..n { if !visited[v] && dist[v] < min_d { min_d = dist[v]; u = Some(v); } }
            let u = match u { Some(v) => v, None => break };
            visited[u] = true;
            for v in 0..n { if self.adjacency[u][v] {
                let alt = dist[u] + self.weights[u][v];
                if alt < dist[v] { dist[v] = alt; }
            }}
        }
        dist
    }
    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let n = self.node_count;
        let mut visited = vec![false; n];
        let mut components = Vec::new();
        for start in 0..n {
            if visited[start] { continue; }
            let mut comp = Vec::new();
            let mut stack = vec![start];
            while let Some(v) = stack.pop() {
                if visited[v] { continue; }
                visited[v] = true;
                comp.push(v);
                for w in 0..n {
                    if (self.adjacency[v][w] || self.adjacency[w][v]) && !visited[w] {
                        stack.push(w);
                    }
                }
            }
            components.push(comp);
        }
        components
    }
    pub fn to_dot(&self) -> String {
        let mut out = String::from("digraph {\n");
        for i in 0..self.node_count {
            out.push_str(&format!("  {} [label=\"{}\"];\n", i, self.node_labels[i]));
        }
        for i in 0..self.node_count { for j in 0..self.node_count { if self.adjacency[i][j] {
            out.push_str(&format!("  {} -> {} [label=\"{:.2}\"];\n", i, j, self.weights[i][j]));
        }}}
        out.push_str("}\n");
        out
    }
}

impl fmt::Display for SidechanGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SidechanGraph(n={}, e={})", self.node_count, self.edge_count)
    }
}

/// Cache for sidechan computation results.
#[derive(Debug, Clone)]
pub struct SidechanCache {
    entries: Vec<(u64, Vec<f64>)>,
    capacity: usize,
    hits: u64,
    misses: u64,
}

impl SidechanCache {
    pub fn new(capacity: usize) -> Self {
        SidechanCache { entries: Vec::new(), capacity, hits: 0, misses: 0 }
    }
    pub fn get(&mut self, key: u64) -> Option<&Vec<f64>> {
        if let Some(pos) = self.entries.iter().position(|(k, _)| *k == key) {
            self.hits += 1;
            Some(&self.entries[pos].1)
        } else { self.misses += 1; None }
    }
    pub fn insert(&mut self, key: u64, value: Vec<f64>) {
        if self.entries.len() >= self.capacity { self.entries.remove(0); }
        self.entries.push((key, value));
    }
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }
    pub fn size(&self) -> usize { self.entries.len() }
    pub fn clear(&mut self) { self.entries.clear(); self.hits = 0; self.misses = 0; }
}

impl fmt::Display for SidechanCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cache(size={}, hit_rate={:.1}%)", self.size(), self.hit_rate() * 100.0)
    }
}

/// Compute pairwise distances for sidechan elements.
pub fn sidechan_pairwise_distances(points: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = points.len();
    let mut distances = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i+1)..n {
            let d: f64 = points[i].iter().zip(points[j].iter())
                .map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
            distances[i][j] = d;
            distances[j][i] = d;
        }
    }
    distances
}

/// K-means clustering for sidechan data.
pub fn sidechan_kmeans(data: &[Vec<f64>], k: usize, max_iters: usize) -> Vec<usize> {
    if data.is_empty() || k == 0 { return Vec::new(); }
    let n = data.len();
    let dim = data[0].len();
    let mut centroids: Vec<Vec<f64>> = data.iter().take(k).cloned().collect();
    let mut assignments = vec![0usize; n];
    for _ in 0..max_iters {
        // Assign
        let mut changed = false;
        for i in 0..n {
            let mut best_c = 0; let mut best_d = f64::INFINITY;
            for c in 0..centroids.len() {
                let d: f64 = data[i].iter().zip(centroids[c].iter())
                    .map(|(a, b)| (a - b) * (a - b)).sum();
                if d < best_d { best_d = d; best_c = c; }
            }
            if assignments[i] != best_c { changed = true; assignments[i] = best_c; }
        }
        if !changed { break; }
        // Update centroids
        for c in 0..centroids.len() {
            let members: Vec<usize> = (0..n).filter(|&i| assignments[i] == c).collect();
            if members.is_empty() { continue; }
            for d in 0..dim {
                centroids[c][d] = members.iter().map(|&i| data[i][d]).sum::<f64>() / members.len() as f64;
            }
        }
    }
    assignments
}

/// Principal component analysis (simplified) for sidechan data.
pub fn sidechan_pca_2d(data: &[Vec<f64>]) -> Vec<(f64, f64)> {
    if data.is_empty() || data[0].len() < 2 { return Vec::new(); }
    let n = data.len();
    let dim = data[0].len();
    // Compute mean
    let mut mean = vec![0.0; dim];
    for row in data { for (j, &v) in row.iter().enumerate() { mean[j] += v; } }
    for j in 0..dim { mean[j] /= n as f64; }
    // Center data
    let centered: Vec<Vec<f64>> = data.iter().map(|row| {
        row.iter().zip(mean.iter()).map(|(v, m)| v - m).collect()
    }).collect();
    // Simple projection onto first two dimensions (not true PCA)
    centered.iter().map(|row| (row[0], row[1])).collect()
}

/// Dense matrix operations for SideChan computations.
#[derive(Debug, Clone)]
pub struct SideChanDenseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl SideChanDenseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        SideChanDenseMatrix { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::new(n, n);
        for i in 0..n { m.data[i * n + i] = 1.0; }
        m
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols);
        SideChanDenseMatrix { rows, cols, data }
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.cols + j] = v;
    }

    pub fn row(&self, i: usize) -> Vec<f64> {
        self.data[i * self.cols..(i + 1) * self.cols].to_vec()
    }

    pub fn col(&self, j: usize) -> Vec<f64> {
        (0..self.rows).map(|i| self.data[i * self.cols + j]).collect()
    }

    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows); assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect();
        SideChanDenseMatrix { rows: self.rows, cols: self.cols, data }
    }

    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows); assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a - b).collect();
        SideChanDenseMatrix { rows: self.rows, cols: self.cols, data }
    }

    pub fn mul_matrix(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows);
        let mut result = Self::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols { sum += self.get(i, k) * other.get(k, j); }
                result.set(i, j, sum);
            }
        }
        result
    }

    pub fn scale(&self, s: f64) -> Self {
        let data: Vec<f64> = self.data.iter().map(|&v| v * s).collect();
        SideChanDenseMatrix { rows: self.rows, cols: self.cols, data }
    }

    pub fn transpose(&self) -> Self {
        let mut result = Self::new(self.cols, self.rows);
        for i in 0..self.rows { for j in 0..self.cols { result.set(j, i, self.get(i, j)); } }
        result
    }

    pub fn trace(&self) -> f64 {
        let n = self.rows.min(self.cols);
        (0..n).map(|i| self.get(i, i)).sum()
    }

    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|&v| v * v).sum::<f64>().sqrt()
    }

    pub fn max_abs(&self) -> f64 {
        self.data.iter().map(|v| v.abs()).fold(0.0f64, f64::max)
    }

    pub fn row_sum(&self, i: usize) -> f64 {
        (0..self.cols).map(|j| self.get(i, j)).sum()
    }

    pub fn col_sum(&self, j: usize) -> f64 {
        (0..self.rows).map(|i| self.get(i, j)).sum()
    }

    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    pub fn is_symmetric(&self) -> bool {
        if !self.is_square() { return false; }
        for i in 0..self.rows { for j in (i+1)..self.cols {
            if (self.get(i, j) - self.get(j, i)).abs() > 1e-10 { return false; }
        }}
        true
    }

    pub fn is_diagonal(&self) -> bool {
        for i in 0..self.rows { for j in 0..self.cols {
            if i != j && self.get(i, j).abs() > 1e-10 { return false; }
        }}
        true
    }

    pub fn is_upper_triangular(&self) -> bool {
        for i in 0..self.rows { for j in 0..i.min(self.cols) {
            if self.get(i, j).abs() > 1e-10 { return false; }
        }}
        true
    }

    pub fn determinant_2x2(&self) -> f64 {
        assert!(self.rows == 2 && self.cols == 2);
        self.get(0, 0) * self.get(1, 1) - self.get(0, 1) * self.get(1, 0)
    }

    pub fn determinant_3x3(&self) -> f64 {
        assert!(self.rows == 3 && self.cols == 3);
        let a = self.get(0, 0); let b = self.get(0, 1); let c = self.get(0, 2);
        let d = self.get(1, 0); let e = self.get(1, 1); let ff = self.get(1, 2);
        let g = self.get(2, 0); let h = self.get(2, 1); let ii = self.get(2, 2);
        a * (e * ii - ff * h) - b * (d * ii - ff * g) + c * (d * h - e * g)
    }

    pub fn inverse_2x2(&self) -> Option<Self> {
        assert!(self.rows == 2 && self.cols == 2);
        let det = self.determinant_2x2();
        if det.abs() < 1e-15 { return None; }
        let inv_det = 1.0 / det;
        let mut result = Self::new(2, 2);
        result.set(0, 0, self.get(1, 1) * inv_det);
        result.set(0, 1, -self.get(0, 1) * inv_det);
        result.set(1, 0, -self.get(1, 0) * inv_det);
        result.set(1, 1, self.get(0, 0) * inv_det);
        Some(result)
    }

    pub fn power(&self, n: u32) -> Self {
        assert!(self.is_square());
        let mut result = Self::identity(self.rows);
        for _ in 0..n { result = result.mul_matrix(self); }
        result
    }

    pub fn submatrix(&self, row_start: usize, col_start: usize, rows: usize, cols: usize) -> Self {
        let mut result = Self::new(rows, cols);
        for i in 0..rows { for j in 0..cols {
            result.set(i, j, self.get(row_start + i, col_start + j));
        }}
        result
    }

    pub fn kronecker_product(&self, other: &Self) -> Self {
        let m = self.rows * other.rows;
        let n = self.cols * other.cols;
        let mut result = Self::new(m, n);
        for i in 0..self.rows { for j in 0..self.cols {
            let s = self.get(i, j);
            for p in 0..other.rows { for q in 0..other.cols {
                result.set(i * other.rows + p, j * other.cols + q, s * other.get(p, q));
            }}
        }}
        result
    }

    pub fn hadamard_product(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows); assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).collect();
        SideChanDenseMatrix { rows: self.rows, cols: self.cols, data }
    }

    pub fn outer_product(a: &[f64], b: &[f64]) -> Self {
        let mut result = Self::new(a.len(), b.len());
        for i in 0..a.len() { for j in 0..b.len() { result.set(i, j, a[i] * b[j]); } }
        result
    }

    pub fn row_reduce(&self) -> Self {
        let mut result = self.clone();
        let mut pivot_row = 0;
        for col in 0..result.cols {
            if pivot_row >= result.rows { break; }
            let mut max_row = pivot_row;
            for row in (pivot_row + 1)..result.rows {
                if result.get(row, col).abs() > result.get(max_row, col).abs() { max_row = row; }
            }
            if result.get(max_row, col).abs() < 1e-10 { continue; }
            for j in 0..result.cols {
                let tmp = result.get(pivot_row, j);
                result.set(pivot_row, j, result.get(max_row, j));
                result.set(max_row, j, tmp);
            }
            let pivot = result.get(pivot_row, col);
            for j in 0..result.cols { result.set(pivot_row, j, result.get(pivot_row, j) / pivot); }
            for row in 0..result.rows {
                if row == pivot_row { continue; }
                let factor = result.get(row, col);
                for j in 0..result.cols {
                    let v = result.get(row, j) - factor * result.get(pivot_row, j);
                    result.set(row, j, v);
                }
            }
            pivot_row += 1;
        }
        result
    }

    pub fn rank(&self) -> usize {
        let rref = self.row_reduce();
        let mut r = 0;
        for i in 0..rref.rows {
            if (0..rref.cols).any(|j| rref.get(i, j).abs() > 1e-10) { r += 1; }
        }
        r
    }

    pub fn nullity(&self) -> usize {
        self.cols - self.rank()
    }

    pub fn column_space_basis(&self) -> Vec<Vec<f64>> {
        let rref = self.row_reduce();
        let mut basis = Vec::new();
        for j in 0..self.cols {
            let is_pivot = (0..rref.rows).any(|i| {
                (rref.get(i, j) - 1.0).abs() < 1e-10 &&
                (0..j).all(|k| rref.get(i, k).abs() < 1e-10)
            });
            if is_pivot { basis.push(self.col(j)); }
        }
        basis
    }

    pub fn lu_decomposition(&self) -> (Self, Self) {
        assert!(self.is_square());
        let n = self.rows;
        let mut l = Self::identity(n);
        let mut u = self.clone();
        for k in 0..n {
            for i in (k+1)..n {
                if u.get(k, k).abs() < 1e-15 { continue; }
                let factor = u.get(i, k) / u.get(k, k);
                l.set(i, k, factor);
                for j in k..n {
                    let v = u.get(i, j) - factor * u.get(k, j);
                    u.set(i, j, v);
                }
            }
        }
        (l, u)
    }

    pub fn solve(&self, b: &[f64]) -> Option<Vec<f64>> {
        assert!(self.is_square());
        assert_eq!(self.rows, b.len());
        let n = self.rows;
        let mut augmented = Self::new(n, n + 1);
        for i in 0..n { for j in 0..n { augmented.set(i, j, self.get(i, j)); } augmented.set(i, n, b[i]); }
        let rref = augmented.row_reduce();
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            x[i] = rref.get(i, n);
            for j in (i+1)..n { x[i] -= rref.get(i, j) * x[j]; }
            if rref.get(i, i).abs() < 1e-15 { return None; }
            x[i] /= rref.get(i, i);
        }
        Some(x)
    }

    pub fn eigenvalues_2x2(&self) -> (f64, f64) {
        assert!(self.rows == 2 && self.cols == 2);
        let tr = self.trace();
        let det = self.determinant_2x2();
        let disc = tr * tr - 4.0 * det;
        if disc >= 0.0 {
            ((tr + disc.sqrt()) / 2.0, (tr - disc.sqrt()) / 2.0)
        } else {
            (tr / 2.0, tr / 2.0)
        }
    }

    pub fn condition_number(&self) -> f64 {
        let max_sv = self.frobenius_norm();
        if max_sv < 1e-15 { return f64::INFINITY; }
        max_sv
    }

}

impl fmt::Display for SideChanDenseMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SideChanMatrix({}x{})", self.rows, self.cols)
    }
}

/// Interval arithmetic for SideChan bounds analysis.
#[derive(Debug, Clone, Copy)]
pub struct SideChanInterval {
    pub lo: f64,
    pub hi: f64,
}

impl SideChanInterval {
    pub fn new(lo: f64, hi: f64) -> Self {
        SideChanInterval { lo: lo.min(hi), hi: lo.max(hi) }
    }

    pub fn point(v: f64) -> Self {
        SideChanInterval { lo: v, hi: v }
    }

    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }

    pub fn midpoint(&self) -> f64 {
        (self.lo + self.hi) / 2.0
    }

    pub fn contains(&self, v: f64) -> bool {
        self.lo <= v && v <= self.hi
    }

    pub fn overlaps(&self, other: &Self) -> bool {
        self.lo <= other.hi && other.lo <= self.hi
    }

    pub fn hull(&self, other: &Self) -> Self {
        SideChanInterval { lo: self.lo.min(other.lo), hi: self.hi.max(other.hi) }
    }

    pub fn intersect(&self, other: &Self) -> Option<Self> {
        let lo = self.lo.max(other.lo);
        let hi = self.hi.min(other.hi);
        if lo <= hi { Some(SideChanInterval { lo, hi }) } else { None }
    }

    pub fn add(&self, other: &Self) -> Self {
        SideChanInterval { lo: self.lo + other.lo, hi: self.hi + other.hi }
    }

    pub fn sub(&self, other: &Self) -> Self {
        SideChanInterval { lo: self.lo - other.hi, hi: self.hi - other.lo }
    }

    pub fn mul(&self, other: &Self) -> Self {
        let products = [self.lo * other.lo, self.lo * other.hi, self.hi * other.lo, self.hi * other.hi];
        let lo = products.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = products.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        SideChanInterval { lo, hi }
    }

    pub fn abs(&self) -> Self {
        if self.lo >= 0.0 { *self }
        else if self.hi <= 0.0 { SideChanInterval { lo: -self.hi, hi: -self.lo } }
        else { SideChanInterval { lo: 0.0, hi: self.lo.abs().max(self.hi.abs()) } }
    }

    pub fn sqrt(&self) -> Self {
        let lo = if self.lo >= 0.0 { self.lo.sqrt() } else { 0.0 };
        SideChanInterval { lo, hi: self.hi.max(0.0).sqrt() }
    }

    pub fn is_positive(&self) -> bool {
        self.lo > 0.0
    }

    pub fn is_negative(&self) -> bool {
        self.hi < 0.0
    }

    pub fn is_zero(&self) -> bool {
        self.lo <= 0.0 && self.hi >= 0.0
    }

    pub fn is_point(&self) -> bool {
        (self.hi - self.lo).abs() < 1e-15
    }

}

impl fmt::Display for SideChanInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:.4}, {:.4}]", self.lo, self.hi)
    }
}

/// State machine for SideChan protocol modeling.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SideChanState {
    Inactive,
    Measuring,
    Analyzing,
    Calibrating,
    Reporting,
    Alarmed,
}

impl fmt::Display for SideChanState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SideChanState::Inactive => write!(f, "inactive"),
            SideChanState::Measuring => write!(f, "measuring"),
            SideChanState::Analyzing => write!(f, "analyzing"),
            SideChanState::Calibrating => write!(f, "calibrating"),
            SideChanState::Reporting => write!(f, "reporting"),
            SideChanState::Alarmed => write!(f, "alarmed"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SideChanStateMachine {
    pub current: SideChanState,
    pub history: Vec<String>,
    pub transition_count: usize,
}

impl SideChanStateMachine {
    pub fn new() -> Self {
        SideChanStateMachine { current: SideChanState::Inactive, history: Vec::new(), transition_count: 0 }
    }
    pub fn state(&self) -> &SideChanState { &self.current }
    pub fn can_transition(&self, target: &SideChanState) -> bool {
        match (&self.current, target) {
            (SideChanState::Inactive, SideChanState::Measuring) => true,
            (SideChanState::Measuring, SideChanState::Analyzing) => true,
            (SideChanState::Analyzing, SideChanState::Calibrating) => true,
            (SideChanState::Calibrating, SideChanState::Reporting) => true,
            (SideChanState::Reporting, SideChanState::Inactive) => true,
            (SideChanState::Analyzing, SideChanState::Alarmed) => true,
            (SideChanState::Alarmed, SideChanState::Inactive) => true,
            _ => false,
        }
    }
    pub fn transition(&mut self, target: SideChanState) -> bool {
        if self.can_transition(&target) {
            self.history.push(format!("{} -> {}", self.current, target));
            self.current = target;
            self.transition_count += 1;
            true
        } else { false }
    }
    pub fn reset(&mut self) {
        self.current = SideChanState::Inactive;
        self.history.clear();
        self.transition_count = 0;
    }
    pub fn history_len(&self) -> usize { self.history.len() }
}

impl fmt::Display for SideChanStateMachine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SM(state={}, transitions={})", self.current, self.transition_count)
    }
}

/// Ring buffer for SideChan event tracking.
#[derive(Debug, Clone)]
pub struct SideChanRingBuffer {
    data: Vec<f64>,
    capacity: usize,
    head: usize,
    count: usize,
}

impl SideChanRingBuffer {
    pub fn new(capacity: usize) -> Self {
        SideChanRingBuffer { data: vec![0.0; capacity], capacity, head: 0, count: 0 }
    }
    pub fn push(&mut self, value: f64) {
        self.data[self.head] = value;
        self.head = (self.head + 1) % self.capacity;
        if self.count < self.capacity { self.count += 1; }
    }
    pub fn len(&self) -> usize { self.count }
    pub fn is_empty(&self) -> bool { self.count == 0 }
    pub fn is_full(&self) -> bool { self.count == self.capacity }
    pub fn latest(&self) -> Option<f64> {
        if self.count == 0 { None }
        else { Some(self.data[(self.head + self.capacity - 1) % self.capacity]) }
    }
    pub fn oldest(&self) -> Option<f64> {
        if self.count == 0 { None }
        else { Some(self.data[(self.head + self.capacity - self.count) % self.capacity]) }
    }
    pub fn average(&self) -> f64 {
        if self.count == 0 { return 0.0; }
        let mut sum = 0.0;
        for i in 0..self.count {
            sum += self.data[(self.head + self.capacity - 1 - i) % self.capacity];
        }
        sum / self.count as f64
    }
    pub fn to_vec(&self) -> Vec<f64> {
        let mut result = Vec::with_capacity(self.count);
        for i in 0..self.count {
            result.push(self.data[(self.head + self.capacity - self.count + i) % self.capacity]);
        }
        result
    }
    pub fn min(&self) -> Option<f64> {
        if self.count == 0 { return None; }
        Some(self.to_vec().iter().cloned().fold(f64::INFINITY, f64::min))
    }
    pub fn max(&self) -> Option<f64> {
        if self.count == 0 { return None; }
        Some(self.to_vec().iter().cloned().fold(f64::NEG_INFINITY, f64::max))
    }
    pub fn variance(&self) -> f64 {
        if self.count < 2 { return 0.0; }
        let avg = self.average();
        let v: f64 = self.to_vec().iter().map(|&x| (x - avg) * (x - avg)).sum();
        v / (self.count - 1) as f64
    }
    pub fn clear(&mut self) { self.head = 0; self.count = 0; }
}

impl fmt::Display for SideChanRingBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RingBuffer(len={}/{})", self.count, self.capacity)
    }
}

/// Disjoint set (union-find) for SideChan component tracking.
#[derive(Debug, Clone)]
pub struct SideChanDisjointSet {
    parent: Vec<usize>,
    rank: Vec<usize>,
    size: Vec<usize>,
    num_components: usize,
}

impl SideChanDisjointSet {
    pub fn new(n: usize) -> Self {
        SideChanDisjointSet { parent: (0..n).collect(), rank: vec![0; n], size: vec![1; n], num_components: n }
    }
    pub fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x { self.parent[x] = self.parent[self.parent[x]]; x = self.parent[x]; }
        x
    }
    pub fn union(&mut self, x: usize, y: usize) -> bool {
        let rx = self.find(x); let ry = self.find(y);
        if rx == ry { return false; }
        if self.rank[rx] < self.rank[ry] { self.parent[rx] = ry; self.size[ry] += self.size[rx]; }
        else if self.rank[rx] > self.rank[ry] { self.parent[ry] = rx; self.size[rx] += self.size[ry]; }
        else { self.parent[ry] = rx; self.size[rx] += self.size[ry]; self.rank[rx] += 1; }
        self.num_components -= 1;
        true
    }
    pub fn connected(&mut self, x: usize, y: usize) -> bool { self.find(x) == self.find(y) }
    pub fn component_size(&mut self, x: usize) -> usize { let r = self.find(x); self.size[r] }
    pub fn num_components(&self) -> usize { self.num_components }
    pub fn components(&mut self) -> Vec<Vec<usize>> {
        let n = self.parent.len();
        let mut groups: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
        for i in 0..n { let r = self.find(i); groups.entry(r).or_default().push(i); }
        groups.into_values().collect()
    }
}

impl fmt::Display for SideChanDisjointSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DisjointSet(n={}, components={})", self.parent.len(), self.num_components)
    }
}

/// Sorted list with binary search for SideChan.
#[derive(Debug, Clone)]
pub struct SideChanSortedList {
    data: Vec<f64>,
}

impl SideChanSortedList {
    pub fn new() -> Self { SideChanSortedList { data: Vec::new() } }
    pub fn insert(&mut self, value: f64) {
        let pos = self.data.partition_point(|&x| x < value);
        self.data.insert(pos, value);
    }
    pub fn contains(&self, value: f64) -> bool {
        self.data.binary_search_by(|x| x.partial_cmp(&value).unwrap()).is_ok()
    }
    pub fn rank(&self, value: f64) -> usize { self.data.partition_point(|&x| x < value) }
    pub fn quantile(&self, q: f64) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let idx = ((self.data.len() - 1) as f64 * q).round() as usize;
        self.data[idx.min(self.data.len() - 1)]
    }
    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
    pub fn min(&self) -> Option<f64> { self.data.first().copied() }
    pub fn max(&self) -> Option<f64> { self.data.last().copied() }
    pub fn median(&self) -> f64 { self.quantile(0.5) }
    pub fn iqr(&self) -> f64 { self.quantile(0.75) - self.quantile(0.25) }
    pub fn remove(&mut self, value: f64) -> bool {
        if let Ok(pos) = self.data.binary_search_by(|x| x.partial_cmp(&value).unwrap()) {
            self.data.remove(pos); true
        } else { false }
    }
    pub fn range(&self, lo: f64, hi: f64) -> Vec<f64> {
        self.data.iter().filter(|&&x| x >= lo && x <= hi).cloned().collect()
    }
    pub fn to_vec(&self) -> Vec<f64> { self.data.clone() }
}

impl fmt::Display for SideChanSortedList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SortedList(len={})", self.data.len())
    }
}

/// Exponential moving average for SideChan metrics.
#[derive(Debug, Clone)]
pub struct SideChanEma {
    pub alpha: f64,
    pub value: f64,
    pub count: usize,
    pub initialized: bool,
}

impl SideChanEma {
    pub fn new(alpha: f64) -> Self { SideChanEma { alpha, value: 0.0, count: 0, initialized: false } }
    pub fn update(&mut self, sample: f64) {
        if !self.initialized { self.value = sample; self.initialized = true; }
        else { self.value = self.alpha * sample + (1.0 - self.alpha) * self.value; }
        self.count += 1;
    }
    pub fn current(&self) -> f64 { self.value }
    pub fn reset(&mut self) { self.value = 0.0; self.count = 0; self.initialized = false; }
}

impl fmt::Display for SideChanEma {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EMA(alpha={:.2}, value={:.4})", self.alpha, self.value)
    }
}

/// Simple bloom filter for SideChan membership testing.
#[derive(Debug, Clone)]
pub struct SideChanBloomFilter {
    bits: Vec<bool>,
    num_hashes: usize,
    size: usize,
    count: usize,
}

impl SideChanBloomFilter {
    pub fn new(size: usize, num_hashes: usize) -> Self {
        SideChanBloomFilter { bits: vec![false; size], num_hashes, size, count: 0 }
    }
    fn hash_indices(&self, value: u64) -> Vec<usize> {
        let mut indices = Vec::with_capacity(self.num_hashes);
        let mut h = value;
        for _ in 0..self.num_hashes {
            h = h.wrapping_mul(0x517cc1b727220a95).wrapping_add(0x6c62272e07bb0142);
            indices.push((h as usize) % self.size);
        }
        indices
    }
    pub fn insert(&mut self, value: u64) {
        for idx in self.hash_indices(value) { self.bits[idx] = true; }
        self.count += 1;
    }
    pub fn may_contain(&self, value: u64) -> bool {
        self.hash_indices(value).iter().all(|&idx| self.bits[idx])
    }
    pub fn false_positive_rate(&self) -> f64 {
        let set_bits = self.bits.iter().filter(|&&b| b).count() as f64;
        (set_bits / self.size as f64).powi(self.num_hashes as i32)
    }
    pub fn count(&self) -> usize { self.count }
    pub fn clear(&mut self) { self.bits.fill(false); self.count = 0; }
}

impl fmt::Display for SideChanBloomFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BloomFilter(size={}, count={}, fpr={:.4})", self.size, self.count, self.false_positive_rate())
    }
}

/// Simple prefix trie for SideChan string matching.
#[derive(Debug, Clone)]
pub struct SideChanTrieNode {
    children: Vec<(char, usize)>,
    is_terminal: bool,
    value: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct SideChanTrie {
    nodes: Vec<SideChanTrieNode>,
    count: usize,
}

impl SideChanTrie {
    pub fn new() -> Self {
        SideChanTrie { nodes: vec![SideChanTrieNode { children: Vec::new(), is_terminal: false, value: None }], count: 0 }
    }
    pub fn insert(&mut self, key: &str, value: u64) {
        let mut current = 0;
        for ch in key.chars() {
            let next = self.nodes[current].children.iter().find(|(c, _)| *c == ch).map(|(_, idx)| *idx);
            current = match next {
                Some(idx) => idx,
                None => {
                    let idx = self.nodes.len();
                    self.nodes.push(SideChanTrieNode { children: Vec::new(), is_terminal: false, value: None });
                    self.nodes[current].children.push((ch, idx));
                    idx
                }
            };
        }
        self.nodes[current].is_terminal = true;
        self.nodes[current].value = Some(value);
        self.count += 1;
    }
    pub fn search(&self, key: &str) -> Option<u64> {
        let mut current = 0;
        for ch in key.chars() {
            match self.nodes[current].children.iter().find(|(c, _)| *c == ch) {
                Some((_, idx)) => current = *idx,
                None => return None,
            }
        }
        if self.nodes[current].is_terminal { self.nodes[current].value } else { None }
    }
    pub fn starts_with(&self, prefix: &str) -> bool {
        let mut current = 0;
        for ch in prefix.chars() {
            match self.nodes[current].children.iter().find(|(c, _)| *c == ch) {
                Some((_, idx)) => current = *idx,
                None => return false,
            }
        }
        true
    }
    pub fn len(&self) -> usize { self.count }
    pub fn is_empty(&self) -> bool { self.count == 0 }
    pub fn node_count(&self) -> usize { self.nodes.len() }
}

impl fmt::Display for SideChanTrie {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Trie(entries={}, nodes={})", self.count, self.nodes.len())
    }
}

/// Min-heap priority queue for SideChan scheduling.
#[derive(Debug, Clone)]
pub struct SideChanPriorityQueue {
    heap: Vec<(f64, usize)>,
}

impl SideChanPriorityQueue {
    pub fn new() -> Self { SideChanPriorityQueue { heap: Vec::new() } }
    pub fn push(&mut self, priority: f64, item: usize) {
        self.heap.push((priority, item));
        let mut i = self.heap.len() - 1;
        while i > 0 {
            let parent = (i - 1) / 2;
            if self.heap[i].0 < self.heap[parent].0 { self.heap.swap(i, parent); i = parent; }
            else { break; }
        }
    }
    pub fn pop(&mut self) -> Option<(f64, usize)> {
        if self.heap.is_empty() { return None; }
        let result = self.heap.swap_remove(0);
        if !self.heap.is_empty() { self.sift_down(0); }
        Some(result)
    }
    fn sift_down(&mut self, mut i: usize) {
        loop {
            let left = 2 * i + 1;
            let right = 2 * i + 2;
            let mut smallest = i;
            if left < self.heap.len() && self.heap[left].0 < self.heap[smallest].0 { smallest = left; }
            if right < self.heap.len() && self.heap[right].0 < self.heap[smallest].0 { smallest = right; }
            if smallest != i { self.heap.swap(i, smallest); i = smallest; }
            else { break; }
        }
    }
    pub fn peek(&self) -> Option<&(f64, usize)> { self.heap.first() }
    pub fn len(&self) -> usize { self.heap.len() }
    pub fn is_empty(&self) -> bool { self.heap.is_empty() }
}

impl fmt::Display for SideChanPriorityQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PQ(len={})", self.heap.len())
    }
}

/// Online statistics accumulator for SideChan.
#[derive(Debug, Clone)]
pub struct SideChanAccumulator {
    count: u64,
    mean: f64,
    m2: f64,
    min_val: f64,
    max_val: f64,
    sum: f64,
}

impl SideChanAccumulator {
    pub fn new() -> Self { SideChanAccumulator { count: 0, mean: 0.0, m2: 0.0, min_val: f64::INFINITY, max_val: f64::NEG_INFINITY, sum: 0.0 } }
    pub fn add(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.min_val = self.min_val.min(value);
        self.max_val = self.max_val.max(value);
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }
    pub fn count(&self) -> u64 { self.count }
    pub fn mean(&self) -> f64 { self.mean }
    pub fn variance(&self) -> f64 { if self.count < 2 { 0.0 } else { self.m2 / (self.count - 1) as f64 } }
    pub fn std_dev(&self) -> f64 { self.variance().sqrt() }
    pub fn min(&self) -> f64 { self.min_val }
    pub fn max(&self) -> f64 { self.max_val }
    pub fn sum(&self) -> f64 { self.sum }
    pub fn range(&self) -> f64 { self.max_val - self.min_val }
    pub fn coefficient_of_variation(&self) -> f64 {
        if self.mean.abs() < 1e-15 { 0.0 } else { self.std_dev() / self.mean.abs() }
    }
    pub fn merge(&mut self, other: &Self) {
        if other.count == 0 { return; }
        let total = self.count + other.count;
        let delta = other.mean - self.mean;
        let new_mean = (self.sum + other.sum) / total as f64;
        self.m2 += other.m2 + delta * delta * (self.count as f64 * other.count as f64 / total as f64);
        self.mean = new_mean;
        self.count = total;
        self.sum += other.sum;
        self.min_val = self.min_val.min(other.min_val);
        self.max_val = self.max_val.max(other.max_val);
    }
    pub fn reset(&mut self) {
        self.count = 0; self.mean = 0.0; self.m2 = 0.0;
        self.min_val = f64::INFINITY; self.max_val = f64::NEG_INFINITY; self.sum = 0.0;
    }
}

impl fmt::Display for SideChanAccumulator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Acc(n={}, mean={:.4}, std={:.4})", self.count, self.mean, self.std_dev())
    }
}

/// Sparse matrix (COO format) for SideChan.
#[derive(Debug, Clone)]
pub struct SideChanSparseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub entries: Vec<(usize, usize, f64)>,
}

impl SideChanSparseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self { SideChanSparseMatrix { rows, cols, entries: Vec::new() } }
    pub fn insert(&mut self, i: usize, j: usize, v: f64) {
        if let Some(pos) = self.entries.iter().position(|&(r, c, _)| r == i && c == j) {
            self.entries[pos].2 = v;
        } else { self.entries.push((i, j, v)); }
    }
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.entries.iter().find(|&&(r, c, _)| r == i && c == j).map(|&(_, _, v)| v).unwrap_or(0.0)
    }
    pub fn nnz(&self) -> usize { self.entries.len() }
    pub fn density(&self) -> f64 { self.entries.len() as f64 / (self.rows * self.cols) as f64 }
    pub fn transpose(&self) -> Self {
        let mut result = SideChanSparseMatrix::new(self.cols, self.rows);
        for &(i, j, v) in &self.entries { result.entries.push((j, i, v)); }
        result
    }
    pub fn add(&self, other: &Self) -> Self {
        let mut result = SideChanSparseMatrix::new(self.rows, self.cols);
        for &(i, j, v) in &self.entries { result.insert(i, j, result.get(i, j) + v); }
        for &(i, j, v) in &other.entries { result.insert(i, j, result.get(i, j) + v); }
        result
    }
    pub fn scale(&self, s: f64) -> Self {
        let mut result = SideChanSparseMatrix::new(self.rows, self.cols);
        for &(i, j, v) in &self.entries { result.entries.push((i, j, v * s)); }
        result
    }
    pub fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; self.rows];
        for &(i, j, v) in &self.entries { result[i] += v * x[j]; }
        result
    }
    pub fn frobenius_norm(&self) -> f64 { self.entries.iter().map(|&(_, _, v)| v * v).sum::<f64>().sqrt() }
    pub fn row_nnz(&self, i: usize) -> usize { self.entries.iter().filter(|&&(r, _, _)| r == i).count() }
    pub fn col_nnz(&self, j: usize) -> usize { self.entries.iter().filter(|&&(_, c, _)| c == j).count() }
    pub fn to_dense(&self, dm_new: fn(usize, usize) -> Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut result = vec![vec![0.0; self.cols]; self.rows];
        for &(i, j, v) in &self.entries { result[i][j] = v; }
        result
    }
    pub fn diagonal(&self) -> Vec<f64> {
        let n = self.rows.min(self.cols);
        (0..n).map(|i| self.get(i, i)).collect()
    }
    pub fn trace(&self) -> f64 { self.diagonal().iter().sum() }
    pub fn remove_zeros(&mut self, tol: f64) {
        self.entries.retain(|&(_, _, v)| v.abs() > tol);
    }
}

impl fmt::Display for SideChanSparseMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sparse({}x{}, nnz={})", self.rows, self.cols, self.nnz())
    }
}

/// Polynomial with f64 coefficients for SideChan.
#[derive(Debug, Clone)]
pub struct SideChanPolynomial {
    pub coefficients: Vec<f64>,
}

impl SideChanPolynomial {
    pub fn new(coeffs: Vec<f64>) -> Self { SideChanPolynomial { coefficients: coeffs } }
    pub fn zero() -> Self { SideChanPolynomial { coefficients: vec![0.0] } }
    pub fn one() -> Self { SideChanPolynomial { coefficients: vec![1.0] } }
    pub fn monomial(degree: usize, coeff: f64) -> Self {
        let mut c = vec![0.0; degree + 1];
        c[degree] = coeff;
        SideChanPolynomial { coefficients: c }
    }
    pub fn degree(&self) -> usize {
        if self.coefficients.is_empty() { return 0; }
        let mut d = self.coefficients.len() - 1;
        while d > 0 && self.coefficients[d].abs() < 1e-15 { d -= 1; }
        d
    }
    pub fn evaluate(&self, x: f64) -> f64 {
        let mut result = 0.0;
        let mut power = 1.0;
        for &c in &self.coefficients {
            result += c * power;
            power *= x;
        }
        result
    }
    pub fn evaluate_horner(&self, x: f64) -> f64 {
        if self.coefficients.is_empty() { return 0.0; }
        let mut result = *self.coefficients.last().unwrap();
        for &c in self.coefficients.iter().rev().skip(1) {
            result = result * x + c;
        }
        result
    }
    pub fn add(&self, other: &Self) -> Self {
        let n = self.coefficients.len().max(other.coefficients.len());
        let mut result = vec![0.0; n];
        for (i, &c) in self.coefficients.iter().enumerate() { result[i] += c; }
        for (i, &c) in other.coefficients.iter().enumerate() { result[i] += c; }
        SideChanPolynomial { coefficients: result }
    }
    pub fn sub(&self, other: &Self) -> Self {
        let n = self.coefficients.len().max(other.coefficients.len());
        let mut result = vec![0.0; n];
        for (i, &c) in self.coefficients.iter().enumerate() { result[i] += c; }
        for (i, &c) in other.coefficients.iter().enumerate() { result[i] -= c; }
        SideChanPolynomial { coefficients: result }
    }
    pub fn mul(&self, other: &Self) -> Self {
        let n = self.coefficients.len() + other.coefficients.len() - 1;
        let mut result = vec![0.0; n];
        for (i, &a) in self.coefficients.iter().enumerate() {
            for (j, &b) in other.coefficients.iter().enumerate() {
                result[i + j] += a * b;
            }
        }
        SideChanPolynomial { coefficients: result }
    }
    pub fn scale(&self, s: f64) -> Self {
        SideChanPolynomial { coefficients: self.coefficients.iter().map(|&c| c * s).collect() }
    }
    pub fn derivative(&self) -> Self {
        if self.coefficients.len() <= 1 { return Self::zero(); }
        let coeffs: Vec<f64> = self.coefficients.iter().enumerate().skip(1)
            .map(|(i, &c)| c * i as f64).collect();
        SideChanPolynomial { coefficients: coeffs }
    }
    pub fn integral(&self, constant: f64) -> Self {
        let mut coeffs = vec![constant];
        for (i, &c) in self.coefficients.iter().enumerate() {
            coeffs.push(c / (i + 1) as f64);
        }
        SideChanPolynomial { coefficients: coeffs }
    }
    pub fn roots_quadratic(&self) -> Vec<f64> {
        if self.degree() != 2 { return Vec::new(); }
        let a = self.coefficients[2];
        let b = self.coefficients[1];
        let c = self.coefficients[0];
        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 { Vec::new() }
        else if disc.abs() < 1e-15 { vec![-b / (2.0 * a)] }
        else { vec![(-b + disc.sqrt()) / (2.0 * a), (-b - disc.sqrt()) / (2.0 * a)] }
    }
    pub fn is_zero(&self) -> bool { self.coefficients.iter().all(|&c| c.abs() < 1e-15) }
    pub fn leading_coefficient(&self) -> f64 {
        self.coefficients.get(self.degree()).copied().unwrap_or(0.0)
    }
    pub fn compose(&self, other: &Self) -> Self {
        let mut result = Self::zero();
        let mut power = Self::one();
        for &c in &self.coefficients {
            result = result.add(&power.scale(c));
            power = power.mul(other);
        }
        result
    }
    pub fn newton_root(&self, initial_guess: f64, max_iters: usize, tol: f64) -> Option<f64> {
        let deriv = self.derivative();
        let mut x = initial_guess;
        for _ in 0..max_iters {
            let fx = self.evaluate(x);
            if fx.abs() < tol { return Some(x); }
            let dfx = deriv.evaluate(x);
            if dfx.abs() < 1e-15 { return None; }
            x -= fx / dfx;
        }
        if self.evaluate(x).abs() < tol * 100.0 { Some(x) } else { None }
    }
}

impl fmt::Display for SideChanPolynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut terms = Vec::new();
        for (i, &c) in self.coefficients.iter().enumerate() {
            if c.abs() < 1e-15 { continue; }
            if i == 0 { terms.push(format!("{:.2}", c)); }
            else if i == 1 { terms.push(format!("{:.2}x", c)); }
            else { terms.push(format!("{:.2}x^{}", c, i)); }
        }
        if terms.is_empty() { write!(f, "0") }
        else { write!(f, "{}", terms.join(" + ")) }
    }
}

/// Simple linear congruential generator for SideChan.
#[derive(Debug, Clone)]
pub struct SideChanRng {
    state: u64,
}

impl SideChanRng {
    pub fn new(seed: u64) -> Self { SideChanRng { state: seed } }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }
    pub fn next_f64(&mut self) -> f64 { (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 }
    pub fn next_range(&mut self, lo: u64, hi: u64) -> u64 {
        if hi <= lo { return lo; }
        lo + (self.next_u64() % (hi - lo))
    }
    pub fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    pub fn shuffle(&mut self, data: &mut [f64]) {
        let n = data.len();
        for i in (1..n).rev() {
            let j = self.next_range(0, i as u64 + 1) as usize;
            data.swap(i, j);
        }
    }
    pub fn sample(&mut self, data: &[f64], n: usize) -> Vec<f64> {
        let mut result = Vec::with_capacity(n);
        for _ in 0..n {
            let idx = self.next_range(0, data.len() as u64) as usize;
            result.push(data[idx]);
        }
        result
    }
    pub fn bernoulli(&mut self, p: f64) -> bool { self.next_f64() < p }
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 { lo + self.next_f64() * (hi - lo) }
    pub fn exponential(&mut self, lambda: f64) -> f64 { -self.next_f64().max(1e-15).ln() / lambda }
}

impl fmt::Display for SideChanRng {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Rng(state={:#x})", self.state)
    }
}

/// Simple timing utility for SideChan benchmarking.
#[derive(Debug, Clone)]
pub struct SideChanTimer {
    pub label: String,
    pub elapsed_ns: Vec<u64>,
    pub running: bool,
}

impl SideChanTimer {
    pub fn new(label: impl Into<String>) -> Self { SideChanTimer { label: label.into(), elapsed_ns: Vec::new(), running: false } }
    pub fn record(&mut self, ns: u64) { self.elapsed_ns.push(ns); }
    pub fn total_ns(&self) -> u64 { self.elapsed_ns.iter().sum() }
    pub fn count(&self) -> usize { self.elapsed_ns.len() }
    pub fn average_ns(&self) -> f64 {
        if self.elapsed_ns.is_empty() { 0.0 } else { self.total_ns() as f64 / self.elapsed_ns.len() as f64 }
    }
    pub fn min_ns(&self) -> u64 { self.elapsed_ns.iter().cloned().min().unwrap_or(0) }
    pub fn max_ns(&self) -> u64 { self.elapsed_ns.iter().cloned().max().unwrap_or(0) }
    pub fn percentile_ns(&self, p: f64) -> u64 {
        if self.elapsed_ns.is_empty() { return 0; }
        let mut sorted = self.elapsed_ns.clone();
        sorted.sort();
        let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }
    pub fn p50_ns(&self) -> u64 { self.percentile_ns(0.5) }
    pub fn p95_ns(&self) -> u64 { self.percentile_ns(0.95) }
    pub fn p99_ns(&self) -> u64 { self.percentile_ns(0.99) }
    pub fn reset(&mut self) { self.elapsed_ns.clear(); self.running = false; }
}

impl fmt::Display for SideChanTimer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Timer({}: avg={:.0}ns, n={})", self.label, self.average_ns(), self.count())
    }
}

/// Compact bit vector for SideChan set operations.
#[derive(Debug, Clone)]
pub struct SideChanBitVector {
    words: Vec<u64>,
    len: usize,
}

impl SideChanBitVector {
    pub fn new(len: usize) -> Self { SideChanBitVector { words: vec![0u64; (len + 63) / 64], len } }
    pub fn set(&mut self, i: usize) { if i < self.len { self.words[i / 64] |= 1u64 << (i % 64); } }
    pub fn clear(&mut self, i: usize) { if i < self.len { self.words[i / 64] &= !(1u64 << (i % 64)); } }
    pub fn get(&self, i: usize) -> bool { i < self.len && (self.words[i / 64] & (1u64 << (i % 64))) != 0 }
    pub fn len(&self) -> usize { self.len }
    pub fn count_ones(&self) -> usize { self.words.iter().map(|w| w.count_ones() as usize).sum() }
    pub fn count_zeros(&self) -> usize { self.len - self.count_ones() }
    pub fn is_empty(&self) -> bool { self.count_ones() == 0 }
    pub fn and(&self, other: &Self) -> Self {
        let n = self.words.len().min(other.words.len());
        let mut result = Self::new(self.len.min(other.len));
        for i in 0..n { result.words[i] = self.words[i] & other.words[i]; }
        result
    }
    pub fn or(&self, other: &Self) -> Self {
        let n = self.words.len().max(other.words.len());
        let mut result = Self::new(self.len.max(other.len));
        for i in 0..self.words.len().min(n) { result.words[i] |= self.words[i]; }
        for i in 0..other.words.len().min(n) { result.words[i] |= other.words[i]; }
        result
    }
    pub fn xor(&self, other: &Self) -> Self {
        let n = self.words.len().max(other.words.len());
        let mut result = Self::new(self.len.max(other.len));
        for i in 0..self.words.len().min(n) { result.words[i] = self.words[i]; }
        for i in 0..other.words.len().min(n) { result.words[i] ^= other.words[i]; }
        result
    }
    pub fn not(&self) -> Self {
        let mut result = Self::new(self.len);
        for i in 0..self.words.len() { result.words[i] = !self.words[i]; }
        // Clear unused bits in last word
        let extra = self.len % 64;
        if extra > 0 && !result.words.is_empty() {
            let last = result.words.len() - 1;
            result.words[last] &= (1u64 << extra) - 1;
        }
        result
    }
    pub fn iter_ones(&self) -> Vec<usize> {
        let mut result = Vec::new();
        for i in 0..self.len { if self.get(i) { result.push(i); } }
        result
    }
    pub fn jaccard(&self, other: &Self) -> f64 {
        let intersection = self.and(other).count_ones() as f64;
        let union = self.or(other).count_ones() as f64;
        if union == 0.0 { 1.0 } else { intersection / union }
    }
    pub fn hamming_distance(&self, other: &Self) -> usize { self.xor(other).count_ones() }
    pub fn fill(&mut self, value: bool) {
        let fill_val = if value { u64::MAX } else { 0 };
        for w in &mut self.words { *w = fill_val; }
        if value { let extra = self.len % 64; if extra > 0 { let last = self.words.len() - 1; self.words[last] &= (1u64 << extra) - 1; } }
    }
}

impl fmt::Display for SideChanBitVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BitVec(len={}, ones={})", self.len, self.count_ones())
    }
}

/// LRU cache for SideChan computation memoization.
#[derive(Debug, Clone)]
pub struct SideChanLruCache {
    entries: Vec<(u64, Vec<f64>, u64)>,
    capacity: usize,
    clock: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl SideChanLruCache {
    pub fn new(capacity: usize) -> Self { SideChanLruCache { entries: Vec::new(), capacity, clock: 0, hits: 0, misses: 0, evictions: 0 } }
    pub fn get(&mut self, key: u64) -> Option<&Vec<f64>> {
        self.clock += 1;
        if let Some(pos) = self.entries.iter().position(|(k, _, _)| *k == key) {
            self.entries[pos].2 = self.clock;
            self.hits += 1;
            Some(&self.entries[pos].1)
        } else { self.misses += 1; None }
    }
    pub fn put(&mut self, key: u64, value: Vec<f64>) {
        self.clock += 1;
        if let Some(pos) = self.entries.iter().position(|(k, _, _)| *k == key) {
            self.entries[pos].1 = value;
            self.entries[pos].2 = self.clock;
            return;
        }
        if self.entries.len() >= self.capacity {
            let lru_pos = self.entries.iter().enumerate()
                .min_by_key(|(_, (_, _, ts))| *ts).map(|(i, _)| i).unwrap();
            self.entries.remove(lru_pos);
            self.evictions += 1;
        }
        self.entries.push((key, value, self.clock));
    }
    pub fn size(&self) -> usize { self.entries.len() }
    pub fn hit_rate(&self) -> f64 { let t = self.hits + self.misses; if t == 0 { 0.0 } else { self.hits as f64 / t as f64 } }
    pub fn eviction_count(&self) -> u64 { self.evictions }
    pub fn contains(&self, key: u64) -> bool { self.entries.iter().any(|(k, _, _)| *k == key) }
    pub fn clear(&mut self) { self.entries.clear(); self.hits = 0; self.misses = 0; self.evictions = 0; self.clock = 0; }
    pub fn keys(&self) -> Vec<u64> { self.entries.iter().map(|(k, _, _)| *k).collect() }
}

impl fmt::Display for SideChanLruCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LRU(size={}/{}, hr={:.1}%)", self.size(), self.capacity, self.hit_rate() * 100.0)
    }
}

/// Graph coloring utility for SideChan scheduling.
#[derive(Debug, Clone)]
pub struct SideChanGraphColoring {
    pub adjacency: Vec<Vec<bool>>,
    pub colors: Vec<Option<usize>>,
    pub num_nodes: usize,
    pub num_colors_used: usize,
}

impl SideChanGraphColoring {
    pub fn new(n: usize) -> Self {
        SideChanGraphColoring { adjacency: vec![vec![false; n]; n], colors: vec![None; n], num_nodes: n, num_colors_used: 0 }
    }
    pub fn add_edge(&mut self, i: usize, j: usize) {
        if i < self.num_nodes && j < self.num_nodes {
            self.adjacency[i][j] = true;
            self.adjacency[j][i] = true;
        }
    }
    pub fn greedy_color(&mut self) -> usize {
        self.colors = vec![None; self.num_nodes];
        let mut max_color = 0;
        for v in 0..self.num_nodes {
            let neighbor_colors: std::collections::HashSet<usize> = (0..self.num_nodes)
                .filter(|&u| self.adjacency[v][u] && self.colors[u].is_some())
                .map(|u| self.colors[u].unwrap()).collect();
            let mut c = 0;
            while neighbor_colors.contains(&c) { c += 1; }
            self.colors[v] = Some(c);
            max_color = max_color.max(c);
        }
        self.num_colors_used = max_color + 1;
        self.num_colors_used
    }
    pub fn is_valid_coloring(&self) -> bool {
        for i in 0..self.num_nodes {
            for j in (i+1)..self.num_nodes {
                if self.adjacency[i][j] {
                    if let (Some(ci), Some(cj)) = (self.colors[i], self.colors[j]) {
                        if ci == cj { return false; }
                    }
                }
            }
        }
        true
    }
    pub fn chromatic_number_upper_bound(&self) -> usize {
        let max_degree = (0..self.num_nodes)
            .map(|v| (0..self.num_nodes).filter(|&u| self.adjacency[v][u]).count())
            .max().unwrap_or(0);
        max_degree + 1
    }
    pub fn color_classes(&self) -> Vec<Vec<usize>> {
        let mut classes: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
        for (v, &c) in self.colors.iter().enumerate() {
            if let Some(color) = c { classes.entry(color).or_default().push(v); }
        }
        let mut result: Vec<Vec<usize>> = classes.into_values().collect();
        result.sort_by_key(|v| v[0]);
        result
    }
}

impl fmt::Display for SideChanGraphColoring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Coloring(n={}, colors={})", self.num_nodes, self.num_colors_used)
    }
}

/// Top-K tracker for SideChan ranking.
#[derive(Debug, Clone)]
pub struct SideChanTopK {
    pub k: usize,
    pub items: Vec<(f64, String)>,
}

impl SideChanTopK {
    pub fn new(k: usize) -> Self { SideChanTopK { k, items: Vec::new() } }
    pub fn insert(&mut self, score: f64, label: impl Into<String>) {
        self.items.push((score, label.into()));
        self.items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        if self.items.len() > self.k { self.items.truncate(self.k); }
    }
    pub fn top(&self) -> &[(f64, String)] { &self.items }
    pub fn min_score(&self) -> Option<f64> { self.items.last().map(|(s, _)| *s) }
    pub fn max_score(&self) -> Option<f64> { self.items.first().map(|(s, _)| *s) }
    pub fn is_full(&self) -> bool { self.items.len() >= self.k }
    pub fn len(&self) -> usize { self.items.len() }
    pub fn contains_label(&self, label: &str) -> bool { self.items.iter().any(|(_, l)| l == label) }
    pub fn clear(&mut self) { self.items.clear(); }
    pub fn merge(&mut self, other: &Self) {
        for (score, label) in &other.items { self.insert(*score, label.clone()); }
    }
}

impl fmt::Display for SideChanTopK {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TopK(k={}, len={})", self.k, self.items.len())
    }
}

/// Sliding window statistics for SideChan monitoring.
#[derive(Debug, Clone)]
pub struct SideChanSlidingWindow {
    pub window_size: usize,
    pub data: Vec<f64>,
    pub sum: f64,
}

impl SideChanSlidingWindow {
    pub fn new(window_size: usize) -> Self { SideChanSlidingWindow { window_size, data: Vec::new(), sum: 0.0 } }
    pub fn push(&mut self, value: f64) {
        self.data.push(value);
        self.sum += value;
        if self.data.len() > self.window_size {
            self.sum -= self.data.remove(0);
        }
    }
    pub fn mean(&self) -> f64 { if self.data.is_empty() { 0.0 } else { self.sum / self.data.len() as f64 } }
    pub fn variance(&self) -> f64 {
        if self.data.len() < 2 { return 0.0; }
        let m = self.mean();
        self.data.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / (self.data.len() - 1) as f64
    }
    pub fn std_dev(&self) -> f64 { self.variance().sqrt() }
    pub fn min(&self) -> f64 { self.data.iter().cloned().fold(f64::INFINITY, f64::min) }
    pub fn max(&self) -> f64 { self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max) }
    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_full(&self) -> bool { self.data.len() >= self.window_size }
    pub fn trend(&self) -> f64 {
        if self.data.len() < 2 { return 0.0; }
        let n = self.data.len() as f64;
        let sum_x: f64 = (0..self.data.len()).map(|i| i as f64).sum();
        let sum_y: f64 = self.data.iter().sum();
        let sum_xy: f64 = self.data.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let sum_xx: f64 = (0..self.data.len()).map(|i| (i as f64) * (i as f64)).sum();
        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-15 { 0.0 } else { (n * sum_xy - sum_x * sum_y) / denom }
    }
    pub fn anomaly_score(&self, value: f64) -> f64 {
        let s = self.std_dev();
        if s.abs() < 1e-15 { return 0.0; }
        ((value - self.mean()) / s).abs()
    }
    pub fn clear(&mut self) { self.data.clear(); self.sum = 0.0; }
}

impl fmt::Display for SideChanSlidingWindow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Window(size={}/{}, mean={:.2})", self.data.len(), self.window_size, self.mean())
    }
}

/// Confusion matrix for SideChan classification evaluation.
#[derive(Debug, Clone)]
pub struct SideChanConfusionMatrix {
    pub true_positive: u64,
    pub false_positive: u64,
    pub true_negative: u64,
    pub false_negative: u64,
}

impl SideChanConfusionMatrix {
    pub fn new() -> Self { SideChanConfusionMatrix { true_positive: 0, false_positive: 0, true_negative: 0, false_negative: 0 } }
    pub fn from_predictions(actual: &[bool], predicted: &[bool]) -> Self {
        let mut cm = Self::new();
        for (&a, &p) in actual.iter().zip(predicted.iter()) {
            match (a, p) {
                (true, true) => cm.true_positive += 1,
                (false, true) => cm.false_positive += 1,
                (true, false) => cm.false_negative += 1,
                (false, false) => cm.true_negative += 1,
            }
        }
        cm
    }
    pub fn total(&self) -> u64 { self.true_positive + self.false_positive + self.true_negative + self.false_negative }
    pub fn accuracy(&self) -> f64 { let t = self.total(); if t == 0 { 0.0 } else { (self.true_positive + self.true_negative) as f64 / t as f64 } }
    pub fn precision(&self) -> f64 { let d = self.true_positive + self.false_positive; if d == 0 { 0.0 } else { self.true_positive as f64 / d as f64 } }
    pub fn recall(&self) -> f64 { let d = self.true_positive + self.false_negative; if d == 0 { 0.0 } else { self.true_positive as f64 / d as f64 } }
    pub fn f1_score(&self) -> f64 { let p = self.precision(); let r = self.recall(); if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) } }
    pub fn specificity(&self) -> f64 { let d = self.true_negative + self.false_positive; if d == 0 { 0.0 } else { self.true_negative as f64 / d as f64 } }
    pub fn false_positive_rate(&self) -> f64 { 1.0 - self.specificity() }
    pub fn matthews_correlation(&self) -> f64 {
        let tp = self.true_positive as f64; let fp = self.false_positive as f64;
        let tn = self.true_negative as f64; let fn_ = self.false_negative as f64;
        let num = tp * tn - fp * fn_;
        let den = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
        if den == 0.0 { 0.0 } else { num / den }
    }
    pub fn merge(&mut self, other: &Self) {
        self.true_positive += other.true_positive;
        self.false_positive += other.false_positive;
        self.true_negative += other.true_negative;
        self.false_negative += other.false_negative;
    }
}

impl fmt::Display for SideChanConfusionMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CM(acc={:.3}, prec={:.3}, rec={:.3}, f1={:.3})",
            self.accuracy(), self.precision(), self.recall(), self.f1_score())
    }
}

/// Cosine similarity for SideChan feature vectors.
pub fn sidechan_cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na * nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

/// Euclidean distance for SideChan.
pub fn sidechan_euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
}

/// Manhattan distance for SideChan.
pub fn sidechan_manhattan_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// Chebyshev distance for SideChan.
pub fn sidechan_chebyshev_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max)
}

/// Minkowski distance for SideChan.
pub fn sidechan_minkowski_distance(a: &[f64], b: &[f64], p: f64) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs().powf(p)).sum::<f64>().powf(1.0 / p)
}

/// Normalize a vector for SideChan.
pub fn sidechan_normalize(v: &[f64]) -> Vec<f64> {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 { v.to_vec() } else { v.iter().map(|x| x / norm).collect() }
}

/// Dot product for SideChan.
pub fn sidechan_dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Cross product (3D) for SideChan.
pub fn sidechan_cross_product(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
}

/// Linear interpolation for SideChan.
pub fn sidechan_lerp(a: f64, b: f64, t: f64) -> f64 { a + (b - a) * t }

/// Clamp value for SideChan.
pub fn sidechan_clamp(v: f64, lo: f64, hi: f64) -> f64 { v.max(lo).min(hi) }

/// Sigmoid function for SideChan.
pub fn sidechan_sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }

/// Softmax for SideChan.
pub fn sidechan_softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Log-sum-exp for SideChan.
pub fn sidechan_logsumexp(values: &[f64]) -> f64 {
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    max + values.iter().map(|&v| (v - max).exp()).sum::<f64>().ln()
}

/// KL divergence for SideChan.
pub fn sidechan_kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q.iter()).map(|(&pi, &qi)| {
        if pi > 0.0 && qi > 0.0 { pi * (pi / qi).ln() } else { 0.0 }
    }).sum()
}

/// Jensen-Shannon divergence for SideChan.
pub fn sidechan_js_divergence(p: &[f64], q: &[f64]) -> f64 {
    let m: Vec<f64> = p.iter().zip(q.iter()).map(|(&pi, &qi)| (pi + qi) / 2.0).collect();
    (sidechan_kl_divergence(p, &m) + sidechan_kl_divergence(q, &m)) / 2.0
}

/// Total variation distance for SideChan.
pub fn sidechan_tv_distance(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q.iter()).map(|(&pi, &qi)| (pi - qi).abs()).sum::<f64>() / 2.0
}

/// Hellinger distance for SideChan.
pub fn sidechan_hellinger_distance(p: &[f64], q: &[f64]) -> f64 {
    let sum: f64 = p.iter().zip(q.iter()).map(|(&pi, &qi)| {
        let diff = pi.sqrt() - qi.sqrt();
        diff * diff
    }).sum();
    (sum / 2.0).sqrt()
}

/// Earth mover's distance (1D) for SideChan.
pub fn sidechan_emd_1d(p: &[f64], q: &[f64]) -> f64 {
    let mut cum_diff = 0.0;
    let mut total = 0.0;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        cum_diff += pi - qi;
        total += cum_diff.abs();
    }
    total
}

/// Feature scaling utilities for SideChan.
#[derive(Debug, Clone)]
pub struct SideChanFeatureScaler {
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
    pub mins: Vec<f64>,
    pub maxs: Vec<f64>,
    pub fitted: bool,
}

impl SideChanFeatureScaler {
    pub fn new() -> Self { SideChanFeatureScaler { means: Vec::new(), stds: Vec::new(), mins: Vec::new(), maxs: Vec::new(), fitted: false } }
    pub fn fit(&mut self, data: &[Vec<f64>]) {
        if data.is_empty() { return; }
        let dim = data[0].len();
        let n = data.len() as f64;
        self.means = vec![0.0; dim];
        self.mins = vec![f64::INFINITY; dim];
        self.maxs = vec![f64::NEG_INFINITY; dim];
        for row in data {
            for (j, &v) in row.iter().enumerate() {
                self.means[j] += v;
                self.mins[j] = self.mins[j].min(v);
                self.maxs[j] = self.maxs[j].max(v);
            }
        }
        for j in 0..dim { self.means[j] /= n; }
        self.stds = vec![0.0; dim];
        for row in data {
            for (j, &v) in row.iter().enumerate() {
                self.stds[j] += (v - self.means[j]).powi(2);
            }
        }
        for j in 0..dim { self.stds[j] = (self.stds[j] / (n - 1.0).max(1.0)).sqrt(); }
        self.fitted = true;
    }
    pub fn standardize(&self, row: &[f64]) -> Vec<f64> {
        row.iter().enumerate().map(|(j, &v)| {
            if self.stds[j].abs() < 1e-15 { 0.0 } else { (v - self.means[j]) / self.stds[j] }
        }).collect()
    }
    pub fn normalize(&self, row: &[f64]) -> Vec<f64> {
        row.iter().enumerate().map(|(j, &v)| {
            let range = self.maxs[j] - self.mins[j];
            if range.abs() < 1e-15 { 0.0 } else { (v - self.mins[j]) / range }
        }).collect()
    }
    pub fn inverse_standardize(&self, row: &[f64]) -> Vec<f64> {
        row.iter().enumerate().map(|(j, &v)| v * self.stds[j] + self.means[j]).collect()
    }
    pub fn inverse_normalize(&self, row: &[f64]) -> Vec<f64> {
        row.iter().enumerate().map(|(j, &v)| v * (self.maxs[j] - self.mins[j]) + self.mins[j]).collect()
    }
    pub fn dimension(&self) -> usize { self.means.len() }
}

impl fmt::Display for SideChanFeatureScaler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Scaler(dim={}, fitted={})", self.dimension(), self.fitted)
    }
}

/// Simple linear regression for SideChan trend analysis.
#[derive(Debug, Clone)]
pub struct SideChanLinearRegression {
    pub slope: f64,
    pub intercept: f64,
    pub r_squared: f64,
    pub fitted: bool,
}

impl SideChanLinearRegression {
    pub fn new() -> Self { SideChanLinearRegression { slope: 0.0, intercept: 0.0, r_squared: 0.0, fitted: false } }
    pub fn fit(&mut self, x: &[f64], y: &[f64]) {
        assert_eq!(x.len(), y.len());
        let n = x.len() as f64;
        if n < 2.0 { return; }
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
        let sum_xx: f64 = x.iter().map(|&xi| xi * xi).sum();
        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-15 { return; }
        self.slope = (n * sum_xy - sum_x * sum_y) / denom;
        self.intercept = (sum_y - self.slope * sum_x) / n;
        let mean_y = sum_y / n;
        let ss_tot: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
        let ss_res: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| (yi - self.predict(xi)).powi(2)).sum();
        self.r_squared = if ss_tot.abs() < 1e-15 { 1.0 } else { 1.0 - ss_res / ss_tot };
        self.fitted = true;
    }
    pub fn predict(&self, x: f64) -> f64 { self.slope * x + self.intercept }
    pub fn predict_many(&self, xs: &[f64]) -> Vec<f64> { xs.iter().map(|&x| self.predict(x)).collect() }
    pub fn residuals(&self, x: &[f64], y: &[f64]) -> Vec<f64> {
        x.iter().zip(y.iter()).map(|(&xi, &yi)| yi - self.predict(xi)).collect()
    }
    pub fn mse(&self, x: &[f64], y: &[f64]) -> f64 {
        let res = self.residuals(x, y);
        res.iter().map(|r| r * r).sum::<f64>() / res.len() as f64
    }
    pub fn rmse(&self, x: &[f64], y: &[f64]) -> f64 { self.mse(x, y).sqrt() }
}

impl fmt::Display for SideChanLinearRegression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "y = {:.4}x + {:.4} (R2={:.4})", self.slope, self.intercept, self.r_squared)
    }
}

/// Weighted undirected graph for SideChan.
#[derive(Debug, Clone)]
pub struct SideChanWeightedGraph {
    pub adj: Vec<Vec<(usize, f64)>>,
    pub num_nodes: usize,
    pub num_edges: usize,
}

impl SideChanWeightedGraph {
    pub fn new(n: usize) -> Self { SideChanWeightedGraph { adj: vec![Vec::new(); n], num_nodes: n, num_edges: 0 } }
    pub fn add_edge(&mut self, u: usize, v: usize, w: f64) {
        self.adj[u].push((v, w));
        self.adj[v].push((u, w));
        self.num_edges += 1;
    }
    pub fn neighbors(&self, u: usize) -> &[(usize, f64)] { &self.adj[u] }
    pub fn degree(&self, u: usize) -> usize { self.adj[u].len() }
    pub fn total_weight(&self) -> f64 {
        self.adj.iter().flat_map(|edges| edges.iter().map(|(_, w)| w)).sum::<f64>() / 2.0
    }
    pub fn min_spanning_tree_weight(&self) -> f64 {
        // Kruskal's algorithm
        let mut edges: Vec<(f64, usize, usize)> = Vec::new();
        for u in 0..self.num_nodes {
            for &(v, w) in &self.adj[u] {
                if u < v { edges.push((w, u, v)); }
            }
        }
        edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut parent: Vec<usize> = (0..self.num_nodes).collect();
        let mut rank = vec![0usize; self.num_nodes];
        fn find_sidechan(parent: &mut Vec<usize>, x: usize) -> usize {
            if parent[x] != x { parent[x] = find_sidechan(parent, parent[x]); }
            parent[x]
        }
        let mut total = 0.0;
        let mut count = 0;
        for (w, u, v) in edges {
            let ru = find_sidechan(&mut parent, u);
            let rv = find_sidechan(&mut parent, v);
            if ru != rv {
                if rank[ru] < rank[rv] { parent[ru] = rv; }
                else if rank[ru] > rank[rv] { parent[rv] = ru; }
                else { parent[rv] = ru; rank[ru] += 1; }
                total += w;
                count += 1;
                if count == self.num_nodes - 1 { break; }
            }
        }
        total
    }
    pub fn dijkstra(&self, start: usize) -> Vec<f64> {
        let mut dist = vec![f64::INFINITY; self.num_nodes];
        let mut visited = vec![false; self.num_nodes];
        dist[start] = 0.0;
        for _ in 0..self.num_nodes {
            let mut u = None;
            let mut min_d = f64::INFINITY;
            for v in 0..self.num_nodes { if !visited[v] && dist[v] < min_d { min_d = dist[v]; u = Some(v); } }
            let u = match u { Some(v) => v, None => break };
            visited[u] = true;
            for &(v, w) in &self.adj[u] {
                let alt = dist[u] + w;
                if alt < dist[v] { dist[v] = alt; }
            }
        }
        dist
    }
    pub fn eccentricity(&self, u: usize) -> f64 {
        let dists = self.dijkstra(u);
        dists.iter().cloned().filter(|&d| d.is_finite()).fold(0.0f64, f64::max)
    }
    pub fn diameter(&self) -> f64 {
        (0..self.num_nodes).map(|u| self.eccentricity(u)).fold(0.0f64, f64::max)
    }
    pub fn clustering_coefficient(&self, u: usize) -> f64 {
        let neighbors: Vec<usize> = self.adj[u].iter().map(|(v, _)| *v).collect();
        let k = neighbors.len();
        if k < 2 { return 0.0; }
        let mut triangles = 0;
        for i in 0..k {
            for j in (i+1)..k {
                if self.adj[neighbors[i]].iter().any(|(v, _)| *v == neighbors[j]) {
                    triangles += 1;
                }
            }
        }
        2.0 * triangles as f64 / (k * (k - 1)) as f64
    }
    pub fn average_clustering_coefficient(&self) -> f64 {
        let sum: f64 = (0..self.num_nodes).map(|u| self.clustering_coefficient(u)).sum();
        sum / self.num_nodes as f64
    }
}

impl fmt::Display for SideChanWeightedGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WGraph(n={}, e={})", self.num_nodes, self.num_edges)
    }
}

/// Moving average for SideChan.
pub fn sidechan_moving_average(data: &[f64], window: usize) -> Vec<f64> {
    if data.len() < window { return Vec::new(); }
    let mut result = Vec::with_capacity(data.len() - window + 1);
    let mut sum: f64 = data[..window].iter().sum();
    result.push(sum / window as f64);
    for i in window..data.len() {
        sum += data[i] - data[i - window];
        result.push(sum / window as f64);
    }
    result
}

/// Cumulative sum for SideChan.
pub fn sidechan_cumsum(data: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(data.len());
    let mut sum = 0.0;
    for &v in data { sum += v; result.push(sum); }
    result
}

/// Numerical differentiation for SideChan.
pub fn sidechan_diff(data: &[f64]) -> Vec<f64> {
    if data.len() < 2 { return Vec::new(); }
    data.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Auto-correlation for SideChan.
pub fn sidechan_autocorrelation(data: &[f64], lag: usize) -> f64 {
    if data.len() <= lag { return 0.0; }
    let n = data.len();
    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let var: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if var.abs() < 1e-15 { return 0.0; }
    let cov: f64 = (0..(n - lag)).map(|i| (data[i] - mean) * (data[i + lag] - mean)).sum::<f64>() / n as f64;
    cov / var
}

/// Discrete Fourier transform magnitude for SideChan.
pub fn sidechan_dft_magnitude(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut magnitudes = Vec::with_capacity(n / 2 + 1);
    for k in 0..=n/2 {
        let mut re = 0.0;
        let mut im = 0.0;
        for (j, &x) in data.iter().enumerate() {
            let angle = -2.0 * std::f64::consts::PI * k as f64 * j as f64 / n as f64;
            re += x * angle.cos();
            im += x * angle.sin();
        }
        magnitudes.push((re * re + im * im).sqrt());
    }
    magnitudes
}

/// Trapezoidal integration for SideChan.
pub fn sidechan_integrate_trapezoid(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let mut total = 0.0;
    for i in 1..x.len() {
        total += (x[i] - x[i-1]) * (y[i] + y[i-1]) / 2.0;
    }
    total
}

/// Simpson's rule integration for SideChan.
pub fn sidechan_integrate_simpson(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    if n < 3 || n % 2 == 0 { return 0.0; }
    let mut total = 0.0;
    for i in (0..n-2).step_by(2) {
        let h = (x[i+2] - x[i]) / 6.0;
        total += h * (y[i] + 4.0 * y[i+1] + y[i+2]);
    }
    total
}

/// Convolution for SideChan.
pub fn sidechan_convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len() + b.len() - 1;
    let mut result = vec![0.0; n];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            result[i + j] += ai * bj;
        }
    }
    result
}

/// Histogram for SideChan data analysis.
#[derive(Debug, Clone)]
pub struct SideChanHistogramExt {
    pub bins: Vec<usize>,
    pub edges: Vec<f64>,
    pub total: usize,
}

impl SideChanHistogramExt {
    pub fn from_data(data: &[f64], num_bins: usize) -> Self {
        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = if (max - min).abs() < 1e-15 { 1.0 } else { max - min };
        let bin_width = range / num_bins as f64;
        let mut edges = Vec::with_capacity(num_bins + 1);
        for i in 0..=num_bins { edges.push(min + i as f64 * bin_width); }
        let mut bins = vec![0usize; num_bins];
        for &v in data {
            let idx = ((v - min) / bin_width).floor() as usize;
            let idx = idx.min(num_bins - 1);
            bins[idx] += 1;
        }
        SideChanHistogramExt { bins, edges, total: data.len() }
    }
    pub fn bin_count(&self, i: usize) -> usize { self.bins[i] }
    pub fn bin_density(&self, i: usize) -> f64 {
        let w = self.edges[i + 1] - self.edges[i];
        if w.abs() < 1e-15 || self.total == 0 { 0.0 }
        else { self.bins[i] as f64 / (self.total as f64 * w) }
    }
    pub fn num_bins(&self) -> usize { self.bins.len() }
    pub fn mode_bin(&self) -> usize {
        self.bins.iter().enumerate().max_by_key(|(_, &c)| c).map(|(i, _)| i).unwrap_or(0)
    }
    pub fn cumulative(&self) -> Vec<usize> {
        let mut cum = Vec::with_capacity(self.bins.len());
        let mut sum = 0;
        for &c in &self.bins { sum += c; cum.push(sum); }
        cum
    }
    pub fn percentile_bin(&self, p: f64) -> usize {
        let target = (p * self.total as f64).ceil() as usize;
        let cum = self.cumulative();
        cum.iter().position(|&c| c >= target).unwrap_or(self.bins.len() - 1)
    }
    pub fn entropy(&self) -> f64 {
        let n = self.total as f64;
        if n < 1.0 { return 0.0; }
        self.bins.iter().filter(|&&c| c > 0).map(|&c| {
            let p = c as f64 / n;
            -p * p.ln()
        }).sum()
    }
}

impl fmt::Display for SideChanHistogramExt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hist(bins={}, total={})", self.num_bins(), self.total)
    }
}

/// Axis-aligned bounding box for SideChan spatial indexing.
#[derive(Debug, Clone, Copy)]
pub struct SideChanAABB {
    pub x_min: f64, pub y_min: f64,
    pub x_max: f64, pub y_max: f64,
}

impl SideChanAABB {
    pub fn new(x_min: f64, y_min: f64, x_max: f64, y_max: f64) -> Self { SideChanAABB { x_min, y_min, x_max, y_max } }
    pub fn contains(&self, x: f64, y: f64) -> bool { x >= self.x_min && x <= self.x_max && y >= self.y_min && y <= self.y_max }
    pub fn intersects(&self, other: &Self) -> bool {
        !(self.x_max < other.x_min || self.x_min > other.x_max || self.y_max < other.y_min || self.y_min > other.y_max)
    }
    pub fn width(&self) -> f64 { self.x_max - self.x_min }
    pub fn height(&self) -> f64 { self.y_max - self.y_min }
    pub fn area(&self) -> f64 { self.width() * self.height() }
    pub fn center(&self) -> (f64, f64) { ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0) }
    pub fn subdivide(&self) -> [Self; 4] {
        let (cx, cy) = self.center();
        [
            SideChanAABB::new(self.x_min, self.y_min, cx, cy),
            SideChanAABB::new(cx, self.y_min, self.x_max, cy),
            SideChanAABB::new(self.x_min, cy, cx, self.y_max),
            SideChanAABB::new(cx, cy, self.x_max, self.y_max),
        ]
    }
}

/// 2D point for SideChan.
#[derive(Debug, Clone, Copy)]
pub struct SideChanPoint2D { pub x: f64, pub y: f64, pub data: f64 }

/// Quadtree for SideChan spatial indexing.
#[derive(Debug, Clone)]
pub struct SideChanQuadTree {
    pub boundary: SideChanAABB,
    pub points: Vec<SideChanPoint2D>,
    pub children: Option<Vec<SideChanQuadTree>>,
    pub capacity: usize,
    pub depth: usize,
    pub max_depth: usize,
}

impl SideChanQuadTree {
    pub fn new(boundary: SideChanAABB, capacity: usize, max_depth: usize) -> Self {
        SideChanQuadTree { boundary, points: Vec::new(), children: None, capacity, depth: 0, max_depth }
    }
    fn with_depth(boundary: SideChanAABB, capacity: usize, depth: usize, max_depth: usize) -> Self {
        SideChanQuadTree { boundary, points: Vec::new(), children: None, capacity, depth, max_depth }
    }
    pub fn insert(&mut self, p: SideChanPoint2D) -> bool {
        if !self.boundary.contains(p.x, p.y) { return false; }
        if self.points.len() < self.capacity && self.children.is_none() {
            self.points.push(p); return true;
        }
        if self.children.is_none() && self.depth < self.max_depth { self.subdivide_tree(); }
        if let Some(ref mut children) = self.children {
            for child in children.iter_mut() { if child.insert(p) { return true; } }
        }
        self.points.push(p); true
    }
    fn subdivide_tree(&mut self) {
        let quads = self.boundary.subdivide();
        let mut children = Vec::with_capacity(4);
        for q in quads.iter() {
            children.push(SideChanQuadTree::with_depth(*q, self.capacity, self.depth + 1, self.max_depth));
        }
        let old_points: Vec<_> = self.points.drain(..).collect();
        self.children = Some(children);
        for p in old_points { self.insert(p); }
    }
    pub fn query_range(&self, range: &SideChanAABB) -> Vec<SideChanPoint2D> {
        let mut result = Vec::new();
        if !self.boundary.intersects(range) { return result; }
        for p in &self.points { if range.contains(p.x, p.y) { result.push(*p); } }
        if let Some(ref children) = self.children {
            for child in children { result.extend(child.query_range(range)); }
        }
        result
    }
    pub fn count(&self) -> usize {
        let mut c = self.points.len();
        if let Some(ref children) = self.children {
            for child in children { c += child.count(); }
        }
        c
    }
    pub fn tree_depth(&self) -> usize {
        if let Some(ref children) = self.children {
            1 + children.iter().map(|c| c.tree_depth()).max().unwrap_or(0)
        } else { 0 }
    }
}

impl fmt::Display for SideChanQuadTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QTree(count={}, depth={})", self.count(), self.tree_depth())
    }
}

/// QR decomposition helper for SideChan.
pub fn sidechan_qr_decompose(a: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let m = a.len();
    if m == 0 { return (Vec::new(), Vec::new()); }
    let n = a[0].len();
    let mut q = vec![vec![0.0; m]; n]; // column vectors
    let mut r = vec![vec![0.0; n]; n];
    // extract columns of a
    let mut cols: Vec<Vec<f64>> = (0..n).map(|j| (0..m).map(|i| a[i][j]).collect()).collect();
    for j in 0..n {
        let mut v = cols[j].clone();
        for i in 0..j {
            let dot: f64 = v.iter().zip(q[i].iter()).map(|(&a, &b)| a * b).sum();
            r[i][j] = dot;
            for k in 0..m { v[k] -= dot * q[i][k]; }
        }
        let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        r[j][j] = norm;
        if norm.abs() > 1e-15 { for k in 0..m { q[j][k] = v[k] / norm; } }
    }
    // convert q from list of column vectors to matrix
    let q_mat: Vec<Vec<f64>> = (0..m).map(|i| (0..n).map(|j| q[j][i]).collect()).collect();
    (q_mat, r)
}

/// Solve upper triangular system Rx = b for SideChan.
pub fn sidechan_solve_upper_triangular(r: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i+1)..n { s -= r[i][j] * x[j]; }
        x[i] = if r[i][i].abs() > 1e-15 { s / r[i][i] } else { 0.0 };
    }
    x
}

/// Matrix-vector multiply for SideChan.
pub fn sidechan_mat_vec_mul(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter().map(|row| row.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum()).collect()
}

/// Matrix transpose for SideChan.
pub fn sidechan_transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() { return Vec::new(); }
    let m = a.len(); let n = a[0].len();
    (0..n).map(|j| (0..m).map(|i| a[i][j]).collect()).collect()
}

/// Matrix multiply for SideChan.
pub fn sidechan_mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    if m == 0 { return Vec::new(); }
    let k = a[0].len();
    let n = b[0].len();
    let mut c = vec![vec![0.0; n]; m];
    for i in 0..m { for j in 0..n { for l in 0..k { c[i][j] += a[i][l] * b[l][j]; } } }
    c
}

/// Frobenius norm for SideChan.
pub fn sidechan_frobenius_norm(a: &[Vec<f64>]) -> f64 {
    a.iter().flat_map(|row| row.iter()).map(|&x| x * x).sum::<f64>().sqrt()
}

/// Matrix trace for SideChan.
pub fn sidechan_trace(a: &[Vec<f64>]) -> f64 {
    a.iter().enumerate().map(|(i, row)| if i < row.len() { row[i] } else { 0.0 }).sum()
}

/// Identity matrix for SideChan.
pub fn sidechan_identity(n: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n { m[i][i] = 1.0; }
    m
}

/// Power iteration for dominant eigenvalue for SideChan.
pub fn sidechan_power_iteration(a: &[Vec<f64>], max_iter: usize) -> (f64, Vec<f64>) {
    let n = a.len();
    let mut v = vec![1.0; n];
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut v { *x /= norm; }
    let mut eigenvalue = 0.0;
    for _ in 0..max_iter {
        let av = sidechan_mat_vec_mul(a, &v);
        let norm: f64 = av.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm.abs() < 1e-15 { break; }
        eigenvalue = av.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum();
        v = av.iter().map(|&x| x / norm).collect();
    }
    (eigenvalue, v)
}

/// Running statistics with min/max tracking for SideChan.
#[derive(Debug, Clone)]
pub struct SideChanRunningStats {
    pub count: u64,
    pub mean: f64,
    pub m2: f64,
    pub min_val: f64,
    pub max_val: f64,
    pub sum: f64,
}

impl SideChanRunningStats {
    pub fn new() -> Self { SideChanRunningStats { count: 0, mean: 0.0, m2: 0.0, min_val: f64::INFINITY, max_val: f64::NEG_INFINITY, sum: 0.0 } }
    pub fn push(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
        self.min_val = self.min_val.min(x);
        self.max_val = self.max_val.max(x);
        self.sum += x;
    }
    pub fn variance(&self) -> f64 { if self.count < 2 { 0.0 } else { self.m2 / (self.count - 1) as f64 } }
    pub fn std_dev(&self) -> f64 { self.variance().sqrt() }
    pub fn range(&self) -> f64 { self.max_val - self.min_val }
    pub fn coefficient_of_variation(&self) -> f64 { if self.mean.abs() < 1e-15 { 0.0 } else { self.std_dev() / self.mean.abs() } }
    pub fn merge(&mut self, other: &Self) {
        if other.count == 0 { return; }
        let combined_count = self.count + other.count;
        let delta = other.mean - self.mean;
        let combined_mean = self.mean + delta * other.count as f64 / combined_count as f64;
        self.m2 += other.m2 + delta * delta * self.count as f64 * other.count as f64 / combined_count as f64;
        self.mean = combined_mean;
        self.count = combined_count;
        self.min_val = self.min_val.min(other.min_val);
        self.max_val = self.max_val.max(other.max_val);
        self.sum += other.sum;
    }
}

impl fmt::Display for SideChanRunningStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Stats(n={}, mean={:.4}, std={:.4})", self.count, self.mean, self.std_dev())
    }
}

/// Interquartile range for SideChan.
pub fn sidechan_iqr(data: &[f64]) -> f64 {
    sidechan_percentile_at(data, 75.0) - sidechan_percentile_at(data, 25.0)
}

/// Detect outliers using IQR method for SideChan.
pub fn sidechan_outliers(data: &[f64]) -> Vec<usize> {
    let q1 = sidechan_percentile_at(data, 25.0);
    let q3 = sidechan_percentile_at(data, 75.0);
    let iqr = q3 - q1;
    let lower = q1 - 1.5 * iqr;
    let upper = q3 + 1.5 * iqr;
    data.iter().enumerate().filter(|(_, &v)| v < lower || v > upper).map(|(i, _)| i).collect()
}

/// Z-score normalization for SideChan.
pub fn sidechan_zscore(data: &[f64]) -> Vec<f64> {
    let n = data.len() as f64;
    if n < 2.0 { return data.to_vec(); }
    let mean = data.iter().sum::<f64>() / n;
    let std = (data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    if std.abs() < 1e-15 { return vec![0.0; data.len()]; }
    data.iter().map(|&x| (x - mean) / std).collect()
}

/// Rank values for SideChan.
pub fn sidechan_rank(data: &[f64]) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut ranks = vec![0.0; data.len()];
    for (rank, &(idx, _)) in indexed.iter().enumerate() { ranks[idx] = (rank + 1) as f64; }
    ranks
}

/// Spearman rank correlation for SideChan.
pub fn sidechan_spearman(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let rx = sidechan_rank(x);
    let ry = sidechan_rank(y);
    let n = x.len() as f64;
    let d_sq: f64 = rx.iter().zip(ry.iter()).map(|(&a, &b)| (a - b).powi(2)).sum();
    1.0 - 6.0 * d_sq / (n * (n * n - 1.0))
}

/// Covariance matrix for SideChan.
pub fn sidechan_covariance_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if data.is_empty() { return Vec::new(); }
    let n = data.len() as f64;
    let d = data[0].len();
    let means: Vec<f64> = (0..d).map(|j| data.iter().map(|row| row[j]).sum::<f64>() / n).collect();
    let mut cov = vec![vec![0.0; d]; d];
    for i in 0..d {
        for j in i..d {
            let c: f64 = data.iter().map(|row| (row[i] - means[i]) * (row[j] - means[j])).sum::<f64>() / (n - 1.0).max(1.0);
            cov[i][j] = c; cov[j][i] = c;
        }
    }
    cov
}

/// Correlation matrix for SideChan.
pub fn sidechan_correlation_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let cov = sidechan_covariance_matrix(data);
    let d = cov.len();
    let mut corr = vec![vec![0.0; d]; d];
    for i in 0..d {
        for j in 0..d {
            let denom = (cov[i][i] * cov[j][j]).sqrt();
            corr[i][j] = if denom.abs() < 1e-15 { 0.0 } else { cov[i][j] / denom };
        }
    }
    corr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distribution_uniform() {
        let d = Distribution::uniform(4);
        assert_eq!(d.probabilities.len(), 4);
        assert!((d.probabilities[0] - 0.25).abs() < 1e-12);
        assert!(d.validate().is_ok());
    }

    #[test]
    fn test_distribution_from_counts() {
        let d = Distribution::from_counts(&[1, 1, 2]);
        assert!((d.probabilities[0] - 0.25).abs() < 1e-12);
        assert!((d.probabilities[2] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_distribution_entropy_uniform() {
        let d = Distribution::uniform(8);
        let h = d.entropy();
        assert!((h - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_distribution_entropy_deterministic() {
        let d = Distribution::new(vec![1.0, 0.0, 0.0]);
        let h = d.entropy();
        assert!((h - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_distribution_support() {
        let d = Distribution::new(vec![0.5, 0.0, 0.3, 0.2]);
        assert_eq!(d.support(), 3);
    }

    #[test]
    fn test_distribution_normalize() {
        let mut d = Distribution::new(vec![2.0, 3.0, 5.0]);
        d.normalize();
        assert!(d.validate().is_ok());
        assert!((d.probabilities[0] - 0.2).abs() < 1e-12);
    }

    #[test]
    fn test_joint_distribution_marginals() {
        let joint = JointDistribution::new(vec![
            vec![0.1, 0.2],
            vec![0.3, 0.4],
        ]);
        let px = joint.marginalize_x();
        let py = joint.marginalize_y();
        assert!((px.probabilities[0] - 0.3).abs() < 1e-12);
        assert!((px.probabilities[1] - 0.7).abs() < 1e-12);
        assert!((py.probabilities[0] - 0.4).abs() < 1e-12);
        assert!((py.probabilities[1] - 0.6).abs() < 1e-12);
    }

    #[test]
    fn test_joint_distribution_conditional() {
        let joint = JointDistribution::new(vec![
            vec![0.1, 0.2],
            vec![0.3, 0.4],
        ]);
        let cond = joint.conditional_y_given_x(0);
        assert!((cond.probabilities[0] - 1.0 / 3.0).abs() < 1e-12);
        assert!((cond.probabilities[1] - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_joint_from_independent() {
        let px = Distribution::new(vec![0.3, 0.7]);
        let py = Distribution::new(vec![0.4, 0.6]);
        let joint = JointDistribution::from_independent(&px, &py);
        assert!(joint.is_independent());
        assert!((joint.matrix[0][0] - 0.12).abs() < 1e-12);
    }

    #[test]
    fn test_shannon_entropy_binary() {
        let d = Distribution::new(vec![0.5, 0.5]);
        let h = ShannonEntropy::compute(&d);
        assert!((h - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_kl_divergence_same() {
        let p = Distribution::new(vec![0.3, 0.7]);
        let kl = ShannonEntropy::kl_divergence(&p, &p);
        assert!((kl - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_js_divergence_symmetric() {
        let p = Distribution::new(vec![0.3, 0.7]);
        let q = Distribution::new(vec![0.6, 0.4]);
        let js1 = ShannonEntropy::js_divergence(&p, &q);
        let js2 = ShannonEntropy::js_divergence(&q, &p);
        assert!((js1 - js2).abs() < 1e-10);
    }

    #[test]
    fn test_cross_entropy() {
        let p = Distribution::new(vec![0.5, 0.5]);
        let h = ShannonEntropy::cross_entropy(&p, &p);
        // Cross-entropy of P with itself = entropy of P
        let h_p = ShannonEntropy::compute(&p);
        assert!((h - h_p).abs() < 1e-10);
    }

    #[test]
    fn test_mutual_information_independent() {
        let px = Distribution::new(vec![0.5, 0.5]);
        let py = Distribution::new(vec![0.5, 0.5]);
        let joint = JointDistribution::from_independent(&px, &py);
        let mi = MutualInformation::compute(&joint);
        assert!((mi - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_mutual_information_perfect() {
        let joint = JointDistribution::new(vec![
            vec![0.5, 0.0],
            vec![0.0, 0.5],
        ]);
        let mi = MutualInformation::compute(&joint);
        assert!((mi - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalized_mutual_information() {
        let joint = JointDistribution::new(vec![
            vec![0.5, 0.0],
            vec![0.0, 0.5],
        ]);
        let nmi = MutualInformation::normalized(&joint);
        assert!((nmi - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_channel_identity_capacity() {
        let channel = Channel::identity(2);
        let result = ChannelCapacityEstimator::blahut_arimoto(&channel, 100, 1e-10);
        assert!((result.capacity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_channel_bsc_capacity() {
        let channel = Channel::binary_symmetric(0.0);
        let result = ChannelCapacityEstimator::blahut_arimoto(&channel, 100, 1e-10);
        assert!((result.capacity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_channel_output_distribution() {
        let channel = Channel::identity(3);
        let input = Distribution::new(vec![0.2, 0.3, 0.5]);
        let output = channel.output_distribution(&input);
        assert!((output.probabilities[0] - 0.2).abs() < 1e-12);
        assert!((output.probabilities[2] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_min_entropy() {
        let d = Distribution::new(vec![0.5, 0.25, 0.25]);
        let h = MinEntropy::compute(&d);
        assert!((h - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_guessing_entropy() {
        let d = Distribution::new(vec![0.5, 0.25, 0.25]);
        let ge = MinEntropy::guessing_entropy(&d);
        // 1*0.5 + 2*0.25 + 3*0.25 = 0.5 + 0.5 + 0.75 = 1.75
        assert!((ge - 1.75).abs() < 1e-10);
    }

    #[test]
    fn test_renyi_alpha_1_equals_shannon() {
        let d = Distribution::new(vec![0.3, 0.5, 0.2]);
        let renyi = RenyiEntropy::compute(&d, 1.0);
        let shannon = ShannonEntropy::compute(&d);
        assert!((renyi - shannon).abs() < 1e-10);
    }

    #[test]
    fn test_collision_entropy() {
        let d = Distribution::uniform(4);
        let h2 = RenyiEntropy::collision_entropy(&d);
        assert!((h2 - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_hartley_entropy() {
        let d = Distribution::new(vec![0.3, 0.0, 0.5, 0.2]);
        let h0 = RenyiEntropy::hartley_entropy(&d);
        assert!((h0 - (3.0f64).log2()).abs() < 1e-10);
    }

    #[test]
    fn test_renyi_monotone() {
        let d = Distribution::new(vec![0.1, 0.2, 0.3, 0.4]);
        let alphas = vec![0.0, 0.5, 1.0, 2.0, 5.0, 10.0];
        assert!(RenyiEntropy::is_monotone_in_alpha(&d, &alphas));
    }

    #[test]
    fn test_g_leakage_identity() {
        let prior = Distribution::uniform(3);
        let channel = Channel::identity(3);
        let joint = channel.to_joint(&prior);
        let gain = GainFunction::identity(3);
        let leak = g_leakage(&prior, &joint, &gain);
        assert!(leak > 0.0);
    }

    #[test]
    fn test_side_channel_analyzer() {
        let channel = Channel::binary_symmetric(0.1);
        let prior = Distribution::uniform(2);
        let analyzer = SideChannelAnalyzer::new(channel, prior);
        let report = analyzer.analyze();
        assert!(report.shannon_leakage.value > 0.0);
        assert!(!report.is_perfectly_secure);
    }

    #[test]
    fn test_side_channel_perfectly_secure() {
        // A channel where output is independent of input
        let channel = Channel::new(vec![
            vec![0.5, 0.5],
            vec![0.5, 0.5],
        ]);
        let prior = Distribution::uniform(2);
        let analyzer = SideChannelAnalyzer::new(channel, prior);
        let report = analyzer.analyze();
        assert!((report.shannon_leakage.value).abs() < 1e-10);
        assert!(report.is_perfectly_secure);
    }

    #[test]
    fn test_channel_cascade() {
        let c1 = Channel::identity(2);
        let c2 = Channel::binary_symmetric(0.1);
        let cascaded = c1.cascade(&c2);
        // Identity cascaded with BSC should be BSC
        assert!((cascaded.matrix[0][0] - 0.9).abs() < 1e-12);
        assert!((cascaded.matrix[0][1] - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_smith_vulnerability() {
        let d = Distribution::new(vec![0.5, 0.3, 0.2]);
        let v = SmithVulnerability::prior_vulnerability(&d);
        assert!((v - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_markov_chain_entropy_rate() {
        let mc = MarkovChain::new(vec![
            vec![0.5, 0.5],
            vec![0.5, 0.5],
        ]);
        let rate = mc.entropy_rate();
        // For doubly-stochastic with uniform stationary: H(rate) = 1 bit
        assert!((rate - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_total_variation_distance() {
        let p = Distribution::new(vec![0.5, 0.5]);
        let q = Distribution::new(vec![1.0, 0.0]);
        let d = p.total_variation_distance(&q);
        assert!((d - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_hellinger_distance_same() {
        let p = Distribution::new(vec![0.3, 0.7]);
        let d = p.hellinger_distance(&p);
        assert!((d - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_differential_privacy_checker() {
        // Noisy channel should satisfy DP for large enough epsilon
        let channel = Channel::binary_symmetric(0.3);
        let checker = DifferentialPrivacyChecker::new(2.0, 0.0);
        assert!(checker.check_channel(&channel));
    }

    #[test]
    fn test_conditional_entropy() {
        let joint = JointDistribution::new(vec![
            vec![0.5, 0.0],
            vec![0.0, 0.5],
        ]);
        let h_cond = ShannonEntropy::conditional_entropy(&joint);
        assert!((h_cond - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_min_entropy_leakage_perfect_channel() {
        let joint = JointDistribution::new(vec![
            vec![0.5, 0.0],
            vec![0.0, 0.5],
        ]);
        let leakage = MinEntropy::min_entropy_leakage(&joint);
        assert!((leakage - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_differentialprivacy_new() {
        let item = DifferentialPrivacy::new(0.0, 0.0, "test".to_string(), 0.0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_randomizedresponse_new() {
        let item = RandomizedResponse::new(0.0, 0.0, 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_noisecalibration_new() {
        let item = NoiseCalibration::new(0.0, 0.0, "test".to_string(), false);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_laplacemechanism_new() {
        let item = LaplaceMechanism::new(0.0, 0.0, 0.0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_exponentialmechanism_new() {
        let item = ExponentialMechanism::new(0.0, 0.0, Vec::new(), Vec::new());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_compositiontheorem_new() {
        let item = CompositionTheorem::new(0, 0.0, 0.0, "test".to_string());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_erasurechannel_new() {
        let item = ErasureChannel::new(0.0, 0.0, 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_zchannel_new() {
        let item = ZChannel::new(0.0, 0.0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_fanoinequality_new() {
        let item = FanoInequality::new(0.0, 0, 0.0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_ratedistortion_new() {
        let item = RateDistortion::new(0.0, 0.0, 0.0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_sourcecoding_new() {
        let item = SourceCoding::new(0.0, 0.0, 0.0, 0.0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_channelcapacity_new() {
        let item = ChannelCapacity::new(0.0, Vec::new(), Vec::new());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_sidechan_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let m = sidechan_mean(&data);
        assert!((m - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_variance() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let v = sidechan_variance(&data);
        assert!(v > 0.0);
    }

    #[test]
    fn test_sidechan_median() {
        let data = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let m = sidechan_median(&data);
        assert!((m - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_entropy() {
        let data = vec![1.0, 1.0, 1.0, 1.0];
        let e = sidechan_entropy(&data);
        assert!(e > 0.0);
    }

    #[test]
    fn test_sidechan_std_dev() {
        let data = vec![10.0, 10.0, 10.0];
        let s = sidechan_std_dev(&data);
        assert!(s.abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_analysis() {
        let mut a = SidechanAnalysis::new(3);
        a.set(0, 1, 0.5);
        a.set(1, 2, 0.3);
        assert!((a.get(0, 1) - 0.5).abs() < 1e-10);
        assert_eq!(a.size, 3);
    }

    #[test]
    fn test_sidechan_iterator() {
        let iter = SidechanResultIterator::new(vec![(0, 1.0), (1, 2.0), (2, 3.0)]);
        let items: Vec<_> = iter.collect();
        assert_eq!(items.len(), 3);
    }

    #[test]
    fn test_sidechan_batch_processor() {
        let mut proc = SidechanBatchProcessor::new(2);
        proc.process_batch(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(proc.processed, 4);
        assert_eq!(proc.results.len(), 2);
    }

    #[test]
    fn test_sidechan_histogram() {
        let hist = SidechanHistogram::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0], 3);
        assert_eq!(hist.num_bins(), 3);
        assert_eq!(hist.total_count, 5);
    }

    #[test]
    fn test_sidechan_graph() {
        let mut g = SidechanGraph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 2.0);
        g.add_edge(2, 3, 3.0);
        assert_eq!(g.edge_count, 3);
        assert!(g.has_edge(0, 1));
        assert!(!g.has_edge(3, 0));
        assert!(g.is_acyclic());
    }

    #[test]
    fn test_sidechan_graph_shortest_path() {
        let mut g = SidechanGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 2.0);
        g.add_edge(0, 2, 10.0);
        let dist = g.shortest_path_dijkstra(0);
        assert!((dist[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_graph_topo_sort() {
        let mut g = SidechanGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        let topo = g.topological_sort();
        assert!(topo.is_some());
    }

    #[test]
    fn test_sidechan_graph_components() {
        let mut g = SidechanGraph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(2, 3, 1.0);
        let comps = g.connected_components();
        assert_eq!(comps.len(), 2);
    }

    #[test]
    fn test_sidechan_cache() {
        let mut cache = SidechanCache::new(10);
        cache.insert(42, vec![1.0, 2.0]);
        assert!(cache.get(42).is_some());
        assert!(cache.get(99).is_none());
    }

    #[test]
    fn test_sidechan_config() {
        let config = SidechanConfig::default_config().with_verbose(true).with_max_iterations(500);
        assert!(config.verbose);
        assert_eq!(config.max_iterations, 500);
    }

    #[test]
    fn test_sidechan_report() {
        let mut report = SidechanReport::new("Test Report");
        report.add_metric("accuracy", 0.95);
        report.add_warning("low sample size");
        assert_eq!(report.total_metrics(), 1);
        assert!(report.has_warnings());
        let text = report.render_text();
        assert!(text.contains("Test Report"));
    }

    #[test]
    fn test_sidechan_kmeans() {
        let data = vec![vec![0.0, 0.0], vec![0.1, 0.1], vec![10.0, 10.0], vec![10.1, 10.1]];
        let assignments = sidechan_kmeans(&data, 2, 100);
        assert_eq!(assignments.len(), 4);
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[2], assignments[3]);
    }

    #[test]
    fn test_sidechan_pairwise_distances() {
        let points = vec![vec![0.0, 0.0], vec![3.0, 4.0]];
        let dists = sidechan_pairwise_distances(&points);
        assert!((dists[0][1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_harmmean() {
        let data = vec![1.0, 2.0, 4.0];
        let hm = sidechan_harmmean(&data);
        assert!(hm > 0.0 && hm < 4.0);
    }

    #[test]
    fn test_sidechan_geomean() {
        let data = vec![1.0, 2.0, 4.0];
        let gm = sidechan_geomean(&data);
        assert!(gm > 0.0 && gm < 4.0);
    }

    #[test]
    fn test_sidechan_sample_skewness() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = sidechan_sample_skewness(&data);
        assert!(s.abs() < 1.0);
    }

    #[test]
    fn test_sidechan_excess_kurtosis() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let k = sidechan_excess_kurtosis(&data);
        let _ = k; // Just verify it computes
    }

    #[test]
    fn test_sidechan_gini() {
        let data = vec![1.0, 1.0, 1.0, 1.0];
        let g = sidechan_gini(&data);
        assert!(g.abs() < 0.01);
    }

    #[test]
    fn test_sidechan_percentile_at() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let p = sidechan_percentile_at(&data, 0.9);
        assert!(p >= 9.0);
    }

    #[test]
    fn test_sidechan_pca_2d() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let proj = sidechan_pca_2d(&data);
        assert_eq!(proj.len(), 2);
    }

    #[test]
    fn test_sidechan_analysis_normalize() {
        let mut a = SidechanAnalysis::new(2);
        a.set(0, 0, 1.0); a.set(0, 1, 3.0);
        a.normalize();
        assert!((a.total_sum() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_analysis_transpose() {
        let mut a = SidechanAnalysis::new(2);
        a.set(0, 1, 5.0);
        let t = a.transpose();
        assert!((t.get(1, 0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_analysis_multiply() {
        let mut a = SidechanAnalysis::new(2);
        a.set(0, 0, 1.0); a.set(1, 1, 1.0);
        let mut b = SidechanAnalysis::new(2);
        b.set(0, 1, 2.0); b.set(1, 0, 3.0);
        let c = a.multiply(&b);
        assert!((c.get(0, 1) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_analysis_frobenius() {
        let mut a = SidechanAnalysis::new(2);
        a.set(0, 0, 3.0); a.set(1, 1, 4.0);
        assert!((a.frobenius_norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_analysis_symmetric() {
        let mut a = SidechanAnalysis::new(2);
        a.set(0, 1, 1.0); a.set(1, 0, 1.0);
        assert!(a.is_symmetric());
    }

    #[test]
    fn test_sidechan_graph_dot() {
        let mut g = SidechanGraph::new(2);
        g.add_edge(0, 1, 1.0);
        let dot = g.to_dot();
        assert!(dot.contains("digraph"));
    }

    #[test]
    fn test_sidechan_histogram_render() {
        let hist = SidechanHistogram::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 5);
        let ascii = hist.render_ascii(20);
        assert!(!ascii.is_empty());
    }

    #[test]
    fn test_sidechan_batch_reset() {
        let mut proc = SidechanBatchProcessor::new(3);
        proc.process_batch(&[1.0, 2.0, 3.0]);
        assert!(proc.processed > 0);
        proc.reset();
        assert_eq!(proc.processed, 0);
    }

    #[test]
    fn test_sidechan_graph_remove_edge() {
        let mut g = SidechanGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        assert_eq!(g.edge_count, 2);
        g.remove_edge(0, 1);
        assert_eq!(g.edge_count, 1);
        assert!(!g.has_edge(0, 1));
    }

    #[test]
    fn test_sidechan_dense_matrix_new() {
        let m = SideChanDenseMatrix::new(3, 3);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 3);
    }

    #[test]
    fn test_sidechan_dense_matrix_identity() {
        let m = SideChanDenseMatrix::identity(3);
        assert!((m.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((m.get(0, 1)).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_dense_matrix_mul() {
        let a = SideChanDenseMatrix::identity(2);
        let b = SideChanDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let c = a.mul_matrix(&b);
        assert!((c.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((c.get(1, 1) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_dense_matrix_transpose() {
        let a = SideChanDenseMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = a.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert!((t.get(0, 1) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_dense_matrix_det_2x2() {
        let m = SideChanDenseMatrix::from_vec(2, 2, vec![3.0, 7.0, 1.0, -4.0]);
        let det = m.determinant_2x2();
        assert!((det - (-19.0)).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_dense_matrix_det_3x3() {
        let m = SideChanDenseMatrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0]);
        let det = m.determinant_3x3();
        assert!((det - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_dense_matrix_inverse_2x2() {
        let m = SideChanDenseMatrix::from_vec(2, 2, vec![4.0, 7.0, 2.0, 6.0]);
        let inv = m.inverse_2x2().unwrap();
        let prod = m.mul_matrix(&inv);
        assert!((prod.get(0, 0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_dense_matrix_power() {
        let m = SideChanDenseMatrix::identity(3);
        let p = m.power(5);
        assert!((p.get(0, 0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_dense_matrix_rank() {
        let m = SideChanDenseMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0]);
        assert_eq!(m.rank(), 1);
    }

    #[test]
    fn test_sidechan_dense_matrix_solve() {
        let a = SideChanDenseMatrix::from_vec(2, 2, vec![2.0, 1.0, 5.0, 3.0]);
        let x = a.solve(&[4.0, 7.0]).unwrap();
        assert!((x[0] - 5.0).abs() < 1e-8);
        assert!((x[1] - (-6.0)).abs() < 1e-8);
    }

    #[test]
    fn test_sidechan_dense_matrix_lu() {
        let a = SideChanDenseMatrix::from_vec(2, 2, vec![4.0, 3.0, 6.0, 3.0]);
        let (l, u) = a.lu_decomposition();
        let prod = l.mul_matrix(&u);
        assert!((prod.get(0, 0) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_dense_matrix_eigenvalues() {
        let m = SideChanDenseMatrix::from_vec(2, 2, vec![2.0, 1.0, 1.0, 2.0]);
        let (e1, e2) = m.eigenvalues_2x2();
        assert!((e1 - 3.0).abs() < 1e-10);
        assert!((e2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_dense_matrix_kronecker() {
        let a = SideChanDenseMatrix::identity(2);
        let b = SideChanDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let k = a.kronecker_product(&b);
        assert_eq!(k.rows, 4);
        assert_eq!(k.cols, 4);
    }

    #[test]
    fn test_sidechan_dense_matrix_hadamard() {
        let a = SideChanDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = SideChanDenseMatrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let h = a.hadamard_product(&b);
        assert!((h.get(0, 0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_interval() {
        let a = SideChanInterval::new(1.0, 3.0);
        let b = SideChanInterval::new(2.0, 5.0);
        assert!(a.overlaps(&b));
        assert!(a.contains(2.0));
        assert!(!a.contains(4.0));
        let sum = a.add(&b);
        assert!((sum.lo - 3.0).abs() < 1e-10);
        assert!((sum.hi - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_interval_mul() {
        let a = SideChanInterval::new(-2.0, 3.0);
        let b = SideChanInterval::new(1.0, 4.0);
        let prod = a.mul(&b);
        assert!((prod.lo - (-8.0)).abs() < 1e-10);
        assert!((prod.hi - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_interval_hull() {
        let a = SideChanInterval::new(1.0, 3.0);
        let b = SideChanInterval::new(5.0, 7.0);
        let h = a.hull(&b);
        assert!((h.lo - 1.0).abs() < 1e-10);
        assert!((h.hi - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_state_machine() {
        let mut sm = SideChanStateMachine::new();
        assert_eq!(*sm.state(), SideChanState::Inactive);
        assert!(sm.transition(SideChanState::Measuring));
        assert_eq!(*sm.state(), SideChanState::Measuring);
        assert_eq!(sm.transition_count, 1);
    }

    #[test]
    fn test_sidechan_state_machine_invalid() {
        let mut sm = SideChanStateMachine::new();
        let last_state = SideChanState::Alarmed;
        assert!(!sm.can_transition(&last_state));
    }

    #[test]
    fn test_sidechan_state_machine_reset() {
        let mut sm = SideChanStateMachine::new();
        sm.transition(SideChanState::Measuring);
        sm.reset();
        assert_eq!(*sm.state(), SideChanState::Inactive);
        assert_eq!(sm.history_len(), 0);
    }

    #[test]
    fn test_sidechan_ring_buffer() {
        let mut rb = SideChanRingBuffer::new(3);
        rb.push(1.0); rb.push(2.0); rb.push(3.0);
        assert!(rb.is_full());
        assert!((rb.average() - 2.0).abs() < 1e-10);
        rb.push(4.0);
        assert!((rb.oldest().unwrap() - 2.0).abs() < 1e-10);
        assert!((rb.latest().unwrap() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_ring_buffer_to_vec() {
        let mut rb = SideChanRingBuffer::new(5);
        rb.push(10.0); rb.push(20.0); rb.push(30.0);
        let v = rb.to_vec();
        assert_eq!(v.len(), 3);
        assert!((v[0] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_disjoint_set() {
        let mut ds = SideChanDisjointSet::new(5);
        assert_eq!(ds.num_components(), 5);
        ds.union(0, 1);
        ds.union(2, 3);
        assert_eq!(ds.num_components(), 3);
        assert!(ds.connected(0, 1));
        assert!(!ds.connected(0, 2));
    }

    #[test]
    fn test_sidechan_disjoint_set_components() {
        let mut ds = SideChanDisjointSet::new(4);
        ds.union(0, 1); ds.union(2, 3);
        let comps = ds.components();
        assert_eq!(comps.len(), 2);
    }

    #[test]
    fn test_sidechan_sorted_list() {
        let mut sl = SideChanSortedList::new();
        sl.insert(3.0); sl.insert(1.0); sl.insert(2.0);
        assert_eq!(sl.len(), 3);
        assert!((sl.min().unwrap() - 1.0).abs() < 1e-10);
        assert!((sl.max().unwrap() - 3.0).abs() < 1e-10);
        assert!((sl.median() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_sorted_list_remove() {
        let mut sl = SideChanSortedList::new();
        sl.insert(1.0); sl.insert(2.0); sl.insert(3.0);
        assert!(sl.remove(2.0));
        assert_eq!(sl.len(), 2);
        assert!(!sl.contains(2.0));
    }

    #[test]
    fn test_sidechan_ema() {
        let mut ema = SideChanEma::new(0.5);
        ema.update(10.0);
        assert!((ema.current() - 10.0).abs() < 1e-10);
        ema.update(20.0);
        assert!((ema.current() - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_bloom_filter() {
        let mut bf = SideChanBloomFilter::new(1000, 3);
        bf.insert(42);
        bf.insert(100);
        assert!(bf.may_contain(42));
        assert!(bf.may_contain(100));
        assert_eq!(bf.count(), 2);
    }

    #[test]
    fn test_sidechan_trie() {
        let mut trie = SideChanTrie::new();
        trie.insert("hello", 1);
        trie.insert("help", 2);
        trie.insert("world", 3);
        assert_eq!(trie.search("hello"), Some(1));
        assert_eq!(trie.search("help"), Some(2));
        assert_eq!(trie.search("hell"), None);
        assert!(trie.starts_with("hel"));
        assert!(!trie.starts_with("xyz"));
    }

    #[test]
    fn test_sidechan_dense_matrix_sym() {
        let m = SideChanDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 2.0, 3.0]);
        assert!(m.is_symmetric());
    }

    #[test]
    fn test_sidechan_dense_matrix_diag() {
        let m = SideChanDenseMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 3.0]);
        assert!(m.is_diagonal());
    }

    #[test]
    fn test_sidechan_dense_matrix_upper_tri() {
        let m = SideChanDenseMatrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0]);
        assert!(m.is_upper_triangular());
    }

    #[test]
    fn test_sidechan_dense_matrix_outer() {
        let m = SideChanDenseMatrix::outer_product(&[1.0, 2.0], &[3.0, 4.0]);
        assert!((m.get(0, 0) - 3.0).abs() < 1e-10);
        assert!((m.get(1, 1) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_dense_matrix_submatrix() {
        let m = SideChanDenseMatrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let sub = m.submatrix(0, 0, 2, 2);
        assert_eq!(sub.rows, 2);
        assert!((sub.get(1, 1) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_priority_queue() {
        let mut pq = SideChanPriorityQueue::new();
        pq.push(3.0, 1); pq.push(1.0, 2); pq.push(2.0, 3);
        assert_eq!(pq.pop().unwrap().1, 2);
        assert_eq!(pq.pop().unwrap().1, 3);
        assert_eq!(pq.pop().unwrap().1, 1);
    }

    #[test]
    fn test_sidechan_accumulator() {
        let mut acc = SideChanAccumulator::new();
        for i in 1..=10 { acc.add(i as f64); }
        assert!((acc.mean() - 5.5).abs() < 1e-10);
        assert_eq!(acc.count(), 10);
        assert!((acc.min() - 1.0).abs() < 1e-10);
        assert!((acc.max() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_accumulator_merge() {
        let mut a = SideChanAccumulator::new();
        a.add(1.0); a.add(2.0);
        let mut b = SideChanAccumulator::new();
        b.add(3.0); b.add(4.0);
        a.merge(&b);
        assert_eq!(a.count(), 4);
        assert!((a.mean() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_sparse_matrix() {
        let mut m = SideChanSparseMatrix::new(3, 3);
        m.insert(0, 1, 2.0); m.insert(1, 2, 3.0);
        assert_eq!(m.nnz(), 2);
        assert!((m.get(0, 1) - 2.0).abs() < 1e-10);
        assert!((m.get(0, 0)).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_sparse_mul_vec() {
        let mut m = SideChanSparseMatrix::new(2, 2);
        m.insert(0, 0, 1.0); m.insert(1, 1, 2.0);
        let result = m.mul_vec(&[3.0, 4.0]);
        assert!((result[0] - 3.0).abs() < 1e-10);
        assert!((result[1] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_sparse_transpose() {
        let mut m = SideChanSparseMatrix::new(2, 3);
        m.insert(0, 2, 5.0);
        let t = m.transpose();
        assert_eq!(t.rows, 3); assert_eq!(t.cols, 2);
        assert!((t.get(2, 0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_polynomial_eval() {
        let p = SideChanPolynomial::new(vec![1.0, 2.0, 3.0]);
        assert!((p.evaluate(2.0) - 17.0).abs() < 1e-10);
        assert!((p.evaluate_horner(2.0) - 17.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_polynomial_add() {
        let a = SideChanPolynomial::new(vec![1.0, 2.0]);
        let b = SideChanPolynomial::new(vec![3.0, 4.0]);
        let c = a.add(&b);
        assert!((c.evaluate(1.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_polynomial_mul() {
        let a = SideChanPolynomial::new(vec![1.0, 1.0]);
        let b = SideChanPolynomial::new(vec![1.0, 1.0]);
        let c = a.mul(&b);
        assert_eq!(c.degree(), 2);
        assert!((c.evaluate(2.0) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_polynomial_deriv() {
        let p = SideChanPolynomial::new(vec![1.0, 2.0, 3.0]);
        let dp = p.derivative();
        assert!((dp.evaluate(0.0) - 2.0).abs() < 1e-10);
        assert!((dp.evaluate(1.0) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_polynomial_integral() {
        let p = SideChanPolynomial::new(vec![2.0, 3.0]);
        let ip = p.integral(0.0);
        assert!((ip.evaluate(1.0) - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_polynomial_roots() {
        let p = SideChanPolynomial::new(vec![-6.0, 1.0, 1.0]);
        let roots = p.roots_quadratic();
        assert_eq!(roots.len(), 2);
    }

    #[test]
    fn test_sidechan_polynomial_newton() {
        let p = SideChanPolynomial::new(vec![-2.0, 0.0, 1.0]);
        let root = p.newton_root(1.0, 100, 1e-10);
        assert!(root.is_some());
        assert!((root.unwrap() - std::f64::consts::SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn test_sidechan_polynomial_compose() {
        let p = SideChanPolynomial::new(vec![0.0, 0.0, 1.0]);
        let q = SideChanPolynomial::new(vec![1.0, 1.0]);
        let r = p.compose(&q);
        assert!((r.evaluate(2.0) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_rng() {
        let mut rng = SideChanRng::new(42);
        let v1 = rng.next_u64();
        let v2 = rng.next_u64();
        assert_ne!(v1, v2);
        let f = rng.next_f64();
        assert!(f >= 0.0 && f < 1.0);
    }

    #[test]
    fn test_sidechan_rng_gaussian() {
        let mut rng = SideChanRng::new(123);
        let mut sum = 0.0;
        for _ in 0..1000 { sum += rng.next_gaussian(); }
        let mean = sum / 1000.0;
        assert!(mean.abs() < 0.2);
    }

    #[test]
    fn test_sidechan_timer() {
        let mut timer = SideChanTimer::new("test");
        timer.record(100); timer.record(200); timer.record(300);
        assert_eq!(timer.count(), 3);
        assert_eq!(timer.total_ns(), 600);
        assert!((timer.average_ns() - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_bitvector() {
        let mut bv = SideChanBitVector::new(100);
        bv.set(5); bv.set(42); bv.set(99);
        assert!(bv.get(5));
        assert!(bv.get(42));
        assert!(!bv.get(50));
        assert_eq!(bv.count_ones(), 3);
    }

    #[test]
    fn test_sidechan_bitvector_ops() {
        let mut a = SideChanBitVector::new(64);
        a.set(0); a.set(10); a.set(20);
        let mut b = SideChanBitVector::new(64);
        b.set(10); b.set(20); b.set(30);
        let c = a.and(&b);
        assert_eq!(c.count_ones(), 2);
        let d = a.or(&b);
        assert_eq!(d.count_ones(), 4);
        assert_eq!(a.hamming_distance(&b), 2);
    }

    #[test]
    fn test_sidechan_bitvector_jaccard() {
        let mut a = SideChanBitVector::new(10);
        a.set(0); a.set(1); a.set(2);
        let mut b = SideChanBitVector::new(10);
        b.set(1); b.set(2); b.set(3);
        let j = a.jaccard(&b);
        assert!((j - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_priority_queue_empty() {
        let mut pq = SideChanPriorityQueue::new();
        assert!(pq.is_empty());
        assert!(pq.pop().is_none());
    }

    #[test]
    fn test_sidechan_sparse_add() {
        let mut a = SideChanSparseMatrix::new(2, 2);
        a.insert(0, 0, 1.0);
        let mut b = SideChanSparseMatrix::new(2, 2);
        b.insert(0, 0, 2.0); b.insert(1, 1, 3.0);
        let c = a.add(&b);
        assert!((c.get(0, 0) - 3.0).abs() < 1e-10);
        assert!((c.get(1, 1) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_rng_shuffle() {
        let mut rng = SideChanRng::new(99);
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        rng.shuffle(&mut data);
        assert_eq!(data.len(), 5);
        let sum: f64 = data.iter().sum();
        assert!((sum - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_polynomial_display() {
        let p = SideChanPolynomial::new(vec![1.0, 2.0, 3.0]);
        let s = format!("{}", p);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_sidechan_polynomial_monomial() {
        let m = SideChanPolynomial::monomial(3, 5.0);
        assert_eq!(m.degree(), 3);
        assert!((m.evaluate(2.0) - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_timer_percentiles() {
        let mut timer = SideChanTimer::new("perf");
        for i in 1..=100 { timer.record(i); }
        assert_eq!(timer.p50_ns(), 50);
        assert!(timer.p95_ns() >= 90);
    }

    #[test]
    fn test_sidechan_accumulator_cv() {
        let mut acc = SideChanAccumulator::new();
        acc.add(10.0); acc.add(10.0); acc.add(10.0);
        assert!(acc.coefficient_of_variation().abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_sparse_diagonal() {
        let mut m = SideChanSparseMatrix::new(3, 3);
        m.insert(0, 0, 1.0); m.insert(1, 1, 2.0); m.insert(2, 2, 3.0);
        assert!((m.trace() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_lru_cache() {
        let mut cache = SideChanLruCache::new(2);
        cache.put(1, vec![1.0]);
        cache.put(2, vec![2.0]);
        assert!(cache.get(1).is_some());
        cache.put(3, vec![3.0]);
        assert!(cache.get(2).is_none());
        assert!(cache.get(3).is_some());
    }

    #[test]
    fn test_sidechan_lru_hit_rate() {
        let mut cache = SideChanLruCache::new(10);
        cache.put(1, vec![1.0]);
        cache.get(1);
        cache.get(2);
        assert!((cache.hit_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_graph_coloring() {
        let mut gc = SideChanGraphColoring::new(4);
        gc.add_edge(0, 1); gc.add_edge(1, 2); gc.add_edge(2, 3); gc.add_edge(3, 0);
        let colors = gc.greedy_color();
        assert!(gc.is_valid_coloring());
        assert!(colors <= 3);
    }

    #[test]
    fn test_sidechan_graph_coloring_complete() {
        let mut gc = SideChanGraphColoring::new(3);
        gc.add_edge(0, 1); gc.add_edge(1, 2); gc.add_edge(0, 2);
        let colors = gc.greedy_color();
        assert_eq!(colors, 3);
        assert!(gc.is_valid_coloring());
    }

    #[test]
    fn test_sidechan_topk() {
        let mut tk = SideChanTopK::new(3);
        tk.insert(5.0, "e"); tk.insert(3.0, "c"); tk.insert(1.0, "a");
        tk.insert(4.0, "d"); tk.insert(2.0, "b");
        assert_eq!(tk.len(), 3);
        assert!((tk.max_score().unwrap() - 5.0).abs() < 1e-10);
        assert!((tk.min_score().unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_sliding_window() {
        let mut sw = SideChanSlidingWindow::new(3);
        sw.push(1.0); sw.push(2.0); sw.push(3.0);
        assert!((sw.mean() - 2.0).abs() < 1e-10);
        sw.push(4.0);
        assert!((sw.mean() - 3.0).abs() < 1e-10);
        assert_eq!(sw.len(), 3);
    }

    #[test]
    fn test_sidechan_sliding_window_trend() {
        let mut sw = SideChanSlidingWindow::new(10);
        for i in 0..5 { sw.push(i as f64); }
        assert!(sw.trend() > 0.0);
    }

    #[test]
    fn test_sidechan_confusion_matrix() {
        let actual = vec![true, true, false, false, true];
        let predicted = vec![true, false, false, true, true];
        let cm = SideChanConfusionMatrix::from_predictions(&actual, &predicted);
        assert_eq!(cm.true_positive, 2);
        assert_eq!(cm.false_positive, 1);
        assert_eq!(cm.true_negative, 1);
        assert_eq!(cm.false_negative, 1);
        assert_eq!(cm.total(), 5);
    }

    #[test]
    fn test_sidechan_confusion_f1() {
        let cm = SideChanConfusionMatrix { true_positive: 80, false_positive: 20, true_negative: 70, false_negative: 30 };
        assert!((cm.precision() - 0.8).abs() < 1e-10);
        let f1 = cm.f1_score();
        assert!(f1 > 0.0 && f1 < 1.0);
    }

    #[test]
    fn test_sidechan_cosine_similarity() {
        let s = sidechan_cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(s.abs() < 1e-10);
        let s2 = sidechan_cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]);
        assert!((s2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_euclidean_distance() {
        let d = sidechan_euclidean_distance(&[0.0, 0.0], &[3.0, 4.0]);
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_sigmoid() {
        let s = sidechan_sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-10);
        let s2 = sidechan_sigmoid(100.0);
        assert!((s2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_sidechan_softmax() {
        let sm = sidechan_softmax(&[1.0, 2.0, 3.0]);
        let sum: f64 = sm.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(sm[2] > sm[1] && sm[1] > sm[0]);
    }

    #[test]
    fn test_sidechan_kl_divergence() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let kl = sidechan_kl_divergence(&p, &q);
        assert!(kl.abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_normalize() {
        let v = sidechan_normalize(&[3.0, 4.0]);
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_lerp() {
        assert!((sidechan_lerp(0.0, 10.0, 0.5) - 5.0).abs() < 1e-10);
        assert!((sidechan_lerp(0.0, 10.0, 0.0) - 0.0).abs() < 1e-10);
        assert!((sidechan_lerp(0.0, 10.0, 1.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_clamp() {
        assert!((sidechan_clamp(5.0, 0.0, 10.0) - 5.0).abs() < 1e-10);
        assert!((sidechan_clamp(-5.0, 0.0, 10.0) - 0.0).abs() < 1e-10);
        assert!((sidechan_clamp(15.0, 0.0, 10.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_cross_product() {
        let c = sidechan_cross_product(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]);
        assert!((c[0]).abs() < 1e-10);
        assert!((c[1]).abs() < 1e-10);
        assert!((c[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_dot_product() {
        let d = sidechan_dot_product(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        assert!((d - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_js_divergence() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let js = sidechan_js_divergence(&p, &q);
        assert!(js.abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_hellinger() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let h = sidechan_hellinger_distance(&p, &q);
        assert!(h.abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_logsumexp() {
        let lse = sidechan_logsumexp(&[1.0, 2.0, 3.0]);
        assert!(lse > 3.0);
    }

    #[test]
    fn test_sidechan_feature_scaler() {
        let mut scaler = SideChanFeatureScaler::new();
        let data = vec![vec![1.0, 10.0], vec![2.0, 20.0], vec![3.0, 30.0]];
        scaler.fit(&data);
        let normalized = scaler.normalize(&[2.0, 20.0]);
        assert!((normalized[0] - 0.5).abs() < 1e-10);
        assert!((normalized[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_feature_scaler_inverse() {
        let mut scaler = SideChanFeatureScaler::new();
        let data = vec![vec![0.0, 0.0], vec![10.0, 100.0]];
        scaler.fit(&data);
        let normed = scaler.normalize(&[5.0, 50.0]);
        let inv = scaler.inverse_normalize(&normed);
        assert!((inv[0] - 5.0).abs() < 1e-10);
        assert!((inv[1] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_linear_regression() {
        let mut lr = SideChanLinearRegression::new();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        lr.fit(&x, &y);
        assert!((lr.slope - 2.0).abs() < 1e-10);
        assert!(lr.intercept.abs() < 1e-10);
        assert!((lr.r_squared - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_linear_regression_predict() {
        let mut lr = SideChanLinearRegression::new();
        lr.fit(&[0.0, 1.0, 2.0], &[1.0, 3.0, 5.0]);
        assert!((lr.predict(3.0) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_weighted_graph() {
        let mut g = SideChanWeightedGraph::new(4);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 2.0); g.add_edge(2, 3, 3.0);
        assert_eq!(g.num_edges, 3);
        let dists = g.dijkstra(0);
        assert!((dists[3] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_weighted_graph_mst() {
        let mut g = SideChanWeightedGraph::new(4);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 2.0); g.add_edge(2, 3, 3.0);
        g.add_edge(0, 3, 10.0);
        let mst = g.min_spanning_tree_weight();
        assert!((mst - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_moving_average() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ma = sidechan_moving_average(&data, 3);
        assert_eq!(ma.len(), 3);
        assert!((ma[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_cumsum() {
        let cs = sidechan_cumsum(&[1.0, 2.0, 3.0, 4.0]);
        assert!((cs[3] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_diff() {
        let d = sidechan_diff(&[1.0, 3.0, 6.0, 10.0]);
        assert_eq!(d.len(), 3);
        assert!((d[0] - 2.0).abs() < 1e-10);
        assert!((d[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_autocorrelation() {
        let ac = sidechan_autocorrelation(&[1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0], 0);
        assert!((ac - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_dft_magnitude() {
        let mags = sidechan_dft_magnitude(&[1.0, 0.0, -1.0, 0.0]);
        assert!(!mags.is_empty());
    }

    #[test]
    fn test_sidechan_integrate_trapezoid() {
        let area = sidechan_integrate_trapezoid(&[0.0, 1.0, 2.0], &[0.0, 1.0, 0.0]);
        assert!((area - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_convolve() {
        let c = sidechan_convolve(&[1.0, 2.0, 3.0], &[1.0, 1.0]);
        assert_eq!(c.len(), 4);
        assert!((c[0] - 1.0).abs() < 1e-10);
        assert!((c[3] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_weighted_graph_clustering() {
        let mut g = SideChanWeightedGraph::new(4);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 1.0); g.add_edge(0, 2, 1.0);
        let cc = g.clustering_coefficient(0);
        assert!((cc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_histogram_cumulative() {
        let h = SideChanHistogramExt::from_data(&[1.0, 1.0, 2.0, 3.0, 3.0, 3.0], 3);
        let cum = h.cumulative();
        assert_eq!(*cum.last().unwrap(), 6);
    }

    #[test]
    fn test_sidechan_histogram_entropy() {
        let h = SideChanHistogramExt::from_data(&[1.0, 2.0, 3.0, 4.0], 4);
        let ent = h.entropy();
        assert!(ent > 0.0);
    }

    #[test]
    fn test_sidechan_aabb() {
        let bb = SideChanAABB::new(0.0, 0.0, 10.0, 10.0);
        assert!(bb.contains(5.0, 5.0));
        assert!(!bb.contains(11.0, 5.0));
        assert!((bb.area() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_aabb_intersects() {
        let a = SideChanAABB::new(0.0, 0.0, 10.0, 10.0);
        let b = SideChanAABB::new(5.0, 5.0, 15.0, 15.0);
        let c = SideChanAABB::new(20.0, 20.0, 30.0, 30.0);
        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn test_sidechan_quadtree() {
        let bb = SideChanAABB::new(0.0, 0.0, 100.0, 100.0);
        let mut qt = SideChanQuadTree::new(bb, 4, 8);
        for i in 0..20 {
            qt.insert(SideChanPoint2D { x: i as f64 * 5.0, y: i as f64 * 5.0, data: i as f64 });
        }
        assert_eq!(qt.count(), 20);
    }

    #[test]
    fn test_sidechan_quadtree_query() {
        let bb = SideChanAABB::new(0.0, 0.0, 100.0, 100.0);
        let mut qt = SideChanQuadTree::new(bb, 2, 8);
        qt.insert(SideChanPoint2D { x: 10.0, y: 10.0, data: 1.0 });
        qt.insert(SideChanPoint2D { x: 90.0, y: 90.0, data: 2.0 });
        let range = SideChanAABB::new(0.0, 0.0, 50.0, 50.0);
        let found = qt.query_range(&range);
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn test_sidechan_mat_mul() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let c = sidechan_mat_mul(&a, &b);
        assert!((c[0][0] - 19.0).abs() < 1e-10);
        assert!((c[1][1] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_transpose() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let t = sidechan_transpose(&a);
        assert_eq!(t.len(), 3);
        assert_eq!(t[0].len(), 2);
        assert!((t[2][1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_frobenius_norm() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let n = sidechan_frobenius_norm(&a);
        assert!((n - 2.0f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_trace() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert!((sidechan_trace(&a) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_identity() {
        let id = sidechan_identity(3);
        assert!((id[0][0] - 1.0).abs() < 1e-10);
        assert!((id[0][1]).abs() < 1e-10);
        assert!((id[2][2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_power_iteration() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let (eval, _evec) = sidechan_power_iteration(&a, 100);
        assert!((eval - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_sidechan_running_stats() {
        let mut s = SideChanRunningStats::new();
        for &v in &[1.0, 2.0, 3.0, 4.0, 5.0] { s.push(v); }
        assert!((s.mean - 3.0).abs() < 1e-10);
        assert!((s.min_val - 1.0).abs() < 1e-10);
        assert!((s.max_val - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_running_stats_merge() {
        let mut a = SideChanRunningStats::new();
        let mut b = SideChanRunningStats::new();
        for &v in &[1.0, 2.0, 3.0] { a.push(v); }
        for &v in &[4.0, 5.0, 6.0] { b.push(v); }
        a.merge(&b);
        assert_eq!(a.count, 6);
        assert!((a.mean - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_running_stats_cv() {
        let mut s = SideChanRunningStats::new();
        for &v in &[10.0, 10.0, 10.0] { s.push(v); }
        assert!(s.coefficient_of_variation() < 1e-10);
    }

    #[test]
    fn test_sidechan_iqr() {
        let iqr = sidechan_iqr(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert!(iqr > 0.0);
    }

    #[test]
    fn test_sidechan_outliers() {
        let outliers = sidechan_outliers(&[1.0, 2.0, 3.0, 4.0, 5.0, 100.0]);
        assert!(!outliers.is_empty());
    }

    #[test]
    fn test_sidechan_zscore() {
        let z = sidechan_zscore(&[10.0, 20.0, 30.0]);
        assert!((z[1]).abs() < 1e-10); // middle value should be ~0
    }

    #[test]
    fn test_sidechan_rank() {
        let r = sidechan_rank(&[30.0, 10.0, 20.0]);
        assert!((r[0] - 3.0).abs() < 1e-10);
        assert!((r[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_spearman() {
        let rho = sidechan_spearman(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((rho - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_sample_skewness_symmetric() {
        let s = sidechan_sample_skewness(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(s.abs() < 1e-10);
    }

    #[test]
    fn test_sidechan_covariance_matrix() {
        let data = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]];
        let cov = sidechan_covariance_matrix(&data);
        assert_eq!(cov.len(), 2);
        assert!(cov[0][0] > 0.0);
    }

    #[test]
    fn test_sidechan_correlation_matrix() {
        let data = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]];
        let corr = sidechan_correlation_matrix(&data);
        assert!((corr[0][1] - 1.0).abs() < 1e-10); // perfect correlation
    }

}

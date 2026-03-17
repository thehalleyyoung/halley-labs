//! Information-theoretic computations for sonification optimization.
//!
//! Provides mutual information estimation, psychoacoustic channel modeling,
//! information loss bounding, and discriminability estimation from signal
//! detection theory.

use std::collections::HashMap;
use std::f64::consts::{E, PI};

use crate::{
    AuditoryDimension, BarkBand, MappingConfig, OptimizerError, OptimizerResult,
    StreamId, StreamMapping,
};

// ─────────────────────────────────────────────────────────────────────────────
// MutualInformationEstimator
// ─────────────────────────────────────────────────────────────────────────────

/// Estimator for mutual information I(D; A) and its psychoacoustically
/// constrained variant I_ψ(D; A).
#[derive(Debug, Clone)]
pub struct MutualInformationEstimator {
    /// Number of bins for discretization.
    pub num_bins: usize,
    /// Smoothing parameter (Laplace smoothing).
    pub smoothing: f64,
    /// Base of the logarithm (2.0 for bits, E for nats).
    pub log_base: f64,
}

impl Default for MutualInformationEstimator {
    fn default() -> Self {
        MutualInformationEstimator {
            num_bins: 64,
            smoothing: 1e-10,
            log_base: 2.0,
        }
    }
}

impl MutualInformationEstimator {
    pub fn new(num_bins: usize, log_base: f64) -> Self {
        MutualInformationEstimator {
            num_bins,
            smoothing: 1e-10,
            log_base,
        }
    }

    pub fn with_smoothing(mut self, smoothing: f64) -> Self {
        self.smoothing = smoothing;
        self
    }

    /// Compute mutual information I(D; A) for discrete distributions.
    ///
    /// `joint` is the joint distribution P(D, A) as a 2D array (row = D, col = A).
    pub fn mutual_information(&self, joint: &[Vec<f64>]) -> OptimizerResult<f64> {
        if joint.is_empty() || joint[0].is_empty() {
            return Err(OptimizerError::NumericalError("Empty joint distribution".into()));
        }

        let rows = joint.len();
        let cols = joint[0].len();

        // Normalize joint distribution
        let total: f64 = joint.iter().flat_map(|r| r.iter()).sum();
        if total <= 0.0 {
            return Err(OptimizerError::NumericalError("Joint distribution sums to zero".into()));
        }

        let mut p_joint = vec![vec![0.0; cols]; rows];
        for i in 0..rows {
            for j in 0..cols {
                p_joint[i][j] = (joint[i][j] + self.smoothing) / (total + self.smoothing * (rows * cols) as f64);
            }
        }

        // Marginals
        let mut p_d = vec![0.0; rows];
        let mut p_a = vec![0.0; cols];
        for i in 0..rows {
            for j in 0..cols {
                p_d[i] += p_joint[i][j];
                p_a[j] += p_joint[i][j];
            }
        }

        // I(D;A) = sum_{d,a} P(d,a) * log( P(d,a) / (P(d) * P(a)) )
        let mut mi = 0.0;
        for i in 0..rows {
            for j in 0..cols {
                if p_joint[i][j] > 0.0 && p_d[i] > 0.0 && p_a[j] > 0.0 {
                    mi += p_joint[i][j] * self.log(p_joint[i][j] / (p_d[i] * p_a[j]));
                }
            }
        }

        Ok(mi.max(0.0))
    }

    /// Compute psychoacoustically-constrained mutual information:
    /// I_ψ(D; A) = I(D; ψ(A))
    pub fn psychoacoustic_mutual_information(
        &self,
        joint: &[Vec<f64>],
        channel: &PsychoacousticChannel,
    ) -> OptimizerResult<f64> {
        let transformed = channel.apply_to_joint(joint)?;
        self.mutual_information(&transformed)
    }

    /// Shannon entropy H(X) for a discrete distribution.
    pub fn entropy(&self, distribution: &[f64]) -> f64 {
        let total: f64 = distribution.iter().sum();
        if total <= 0.0 {
            return 0.0;
        }

        let mut h = 0.0;
        for &p in distribution {
            let pp = (p + self.smoothing) / (total + self.smoothing * distribution.len() as f64);
            if pp > 0.0 {
                h -= pp * self.log(pp);
            }
        }
        h.max(0.0)
    }

    /// Conditional entropy H(X|Y) from a joint distribution P(X, Y).
    pub fn conditional_entropy(&self, joint: &[Vec<f64>]) -> OptimizerResult<f64> {
        let h_joint = self.joint_entropy(joint)?;
        let cols = joint[0].len();
        let mut p_y = vec![0.0; cols];
        for row in joint {
            for (j, &v) in row.iter().enumerate() {
                p_y[j] += v;
            }
        }
        let h_y = self.entropy(&p_y);
        Ok((h_joint - h_y).max(0.0))
    }

    /// Joint entropy H(X, Y).
    pub fn joint_entropy(&self, joint: &[Vec<f64>]) -> OptimizerResult<f64> {
        if joint.is_empty() {
            return Err(OptimizerError::NumericalError("Empty joint".into()));
        }
        let flat: Vec<f64> = joint.iter().flat_map(|r| r.iter().copied()).collect();
        Ok(self.entropy(&flat))
    }

    /// KL divergence D_KL(P || Q).
    pub fn kl_divergence(&self, p: &[f64], q: &[f64]) -> OptimizerResult<f64> {
        if p.len() != q.len() {
            return Err(OptimizerError::NumericalError(
                "P and Q must have same length".into(),
            ));
        }
        let total_p: f64 = p.iter().sum();
        let total_q: f64 = q.iter().sum();
        if total_p <= 0.0 || total_q <= 0.0 {
            return Err(OptimizerError::NumericalError(
                "Distributions must be positive".into(),
            ));
        }

        let n = p.len() as f64;
        let mut kl = 0.0;
        for i in 0..p.len() {
            let pp = (p[i] + self.smoothing) / (total_p + self.smoothing * n);
            let qq = (q[i] + self.smoothing) / (total_q + self.smoothing * n);
            if pp > 0.0 {
                kl += pp * self.log(pp / qq);
            }
        }
        Ok(kl.max(0.0))
    }

    /// Symmetric KL divergence (Jensen-Shannon divergence).
    pub fn js_divergence(&self, p: &[f64], q: &[f64]) -> OptimizerResult<f64> {
        let m: Vec<f64> = p.iter().zip(q.iter()).map(|(a, b)| (a + b) / 2.0).collect();
        let kl_pm = self.kl_divergence(p, &m)?;
        let kl_qm = self.kl_divergence(q, &m)?;
        Ok((kl_pm + kl_qm) / 2.0)
    }

    /// Channel capacity: max over input distributions of I(X; Y).
    /// Uses Blahut-Arimoto algorithm for discrete memoryless channels.
    pub fn channel_capacity(
        &self,
        channel_matrix: &[Vec<f64>],
        max_iterations: usize,
        tolerance: f64,
    ) -> OptimizerResult<f64> {
        let n_input = channel_matrix.len();
        if n_input == 0 {
            return Err(OptimizerError::NumericalError("Empty channel".into()));
        }
        let n_output = channel_matrix[0].len();

        // Normalize channel rows to be conditional distributions P(Y|X)
        let mut p_yx: Vec<Vec<f64>> = vec![vec![0.0; n_output]; n_input];
        for i in 0..n_input {
            let row_sum: f64 = channel_matrix[i].iter().sum();
            if row_sum > 0.0 {
                for j in 0..n_output {
                    p_yx[i][j] = channel_matrix[i][j] / row_sum;
                }
            }
        }

        // Blahut-Arimoto iteration
        let mut q = vec![1.0 / n_input as f64; n_input]; // Uniform input
        let mut capacity = 0.0;

        for _ in 0..max_iterations {
            // Compute output distribution
            let mut p_y = vec![0.0; n_output];
            for i in 0..n_input {
                for j in 0..n_output {
                    p_y[j] += q[i] * p_yx[i][j];
                }
            }

            // Compute exponent for each input
            let mut c = vec![0.0; n_input];
            for i in 0..n_input {
                for j in 0..n_output {
                    if p_yx[i][j] > 0.0 && p_y[j] > 0.0 {
                        c[i] += p_yx[i][j] * (p_yx[i][j] / p_y[j]).ln();
                    }
                }
                c[i] = c[i].exp();
            }

            // Update input distribution
            let mut new_q = vec![0.0; n_input];
            let denom: f64 = q.iter().zip(c.iter()).map(|(qi, ci)| qi * ci).sum();
            for i in 0..n_input {
                new_q[i] = q[i] * c[i] / denom;
            }

            // Compute mutual information with current input distribution
            let new_capacity = self.compute_mi_from_channel(&new_q, &p_yx)?;

            if (new_capacity - capacity).abs() < tolerance {
                return Ok(new_capacity / (self.log_base.ln())); // Convert to chosen base
            }
            capacity = new_capacity;
            q = new_q;
        }

        Ok(capacity / (self.log_base.ln()))
    }

    fn compute_mi_from_channel(
        &self,
        input_dist: &[f64],
        channel: &[Vec<f64>],
    ) -> OptimizerResult<f64> {
        let n_input = input_dist.len();
        let n_output = channel[0].len();

        let mut p_y = vec![0.0; n_output];
        for i in 0..n_input {
            for j in 0..n_output {
                p_y[j] += input_dist[i] * channel[i][j];
            }
        }

        let mut mi = 0.0;
        for i in 0..n_input {
            for j in 0..n_output {
                if input_dist[i] > 0.0 && channel[i][j] > 0.0 && p_y[j] > 0.0 {
                    mi += input_dist[i] * channel[i][j] * (channel[i][j] / p_y[j]).ln();
                }
            }
        }
        Ok(mi)
    }

    /// Estimate I_ψ(D; A) from sample data and a mapping config.
    pub fn estimate_from_samples(
        &self,
        data_samples: &[Vec<f64>],
        audio_samples: &[Vec<f64>],
        channel: &PsychoacousticChannel,
    ) -> OptimizerResult<f64> {
        let joint = self.build_joint_histogram(data_samples, audio_samples)?;
        self.psychoacoustic_mutual_information(&joint, channel)
    }

    fn build_joint_histogram(
        &self,
        data_samples: &[Vec<f64>],
        audio_samples: &[Vec<f64>],
    ) -> OptimizerResult<Vec<Vec<f64>>> {
        if data_samples.is_empty() || audio_samples.is_empty() {
            return Err(OptimizerError::NumericalError("Empty samples".into()));
        }
        if data_samples.len() != audio_samples.len() {
            return Err(OptimizerError::NumericalError("Mismatched sample counts".into()));
        }

        let n = data_samples.len();
        let data_flat: Vec<f64> = data_samples.iter().map(|v| v.iter().sum()).collect();
        let audio_flat: Vec<f64> = audio_samples.iter().map(|v| v.iter().sum()).collect();

        let (d_min, d_max) = min_max(&data_flat);
        let (a_min, a_max) = min_max(&audio_flat);

        let d_range = (d_max - d_min).max(1e-12);
        let a_range = (a_max - a_min).max(1e-12);

        let mut joint = vec![vec![0.0; self.num_bins]; self.num_bins];
        for i in 0..n {
            let d_bin = ((data_flat[i] - d_min) / d_range * (self.num_bins - 1) as f64)
                .round() as usize;
            let a_bin = ((audio_flat[i] - a_min) / a_range * (self.num_bins - 1) as f64)
                .round() as usize;
            let d_bin = d_bin.min(self.num_bins - 1);
            let a_bin = a_bin.min(self.num_bins - 1);
            joint[d_bin][a_bin] += 1.0;
        }

        Ok(joint)
    }

    fn log(&self, x: f64) -> f64 {
        if self.log_base == E {
            x.ln()
        } else {
            x.ln() / self.log_base.ln()
        }
    }
}

fn min_max(data: &[f64]) -> (f64, f64) {
    let mut mn = f64::INFINITY;
    let mut mx = f64::NEG_INFINITY;
    for &v in data {
        if v < mn {
            mn = v;
        }
        if v > mx {
            mx = v;
        }
    }
    (mn, mx)
}

// ─────────────────────────────────────────────────────────────────────────────
// PsychoacousticChannel
// ─────────────────────────────────────────────────────────────────────────────

/// Models the psychoacoustic channel ψ: A → P
/// from acoustic representation to perceived representation.
#[derive(Debug, Clone)]
pub struct PsychoacousticChannel {
    pub stages: Vec<ChannelStage>,
    pub bark_bands: usize,
}

/// A single stage in the psychoacoustic channel pipeline.
#[derive(Debug, Clone)]
pub enum ChannelStage {
    /// Zero out components masked by simultaneous maskers.
    Masking(MaskingParams),
    /// Discretize to JND-step resolution.
    JndQuantization(JndParams),
    /// Filter by auditory stream segregation.
    SegregationFilter(SegregationParams),
    /// Cognitive capacity limiting.
    CognitiveFilter(CognitiveParams),
}

/// Parameters for masking stage.
#[derive(Debug, Clone)]
pub struct MaskingParams {
    /// Masking threshold per Bark band (in dB SPL).
    pub thresholds: Vec<f64>,
    /// Masking slope (dB/Bark) for upper and lower slopes.
    pub upper_slope: f64,
    pub lower_slope: f64,
}

impl Default for MaskingParams {
    fn default() -> Self {
        MaskingParams {
            thresholds: vec![20.0; BarkBand::NUM_BANDS],
            upper_slope: -25.0,
            lower_slope: -10.0,
        }
    }
}

/// Parameters for JND quantization stage.
#[derive(Debug, Clone)]
pub struct JndParams {
    /// JND step sizes per auditory dimension.
    pub step_sizes: HashMap<AuditoryDimension, f64>,
    /// Default step size if dimension not specified.
    pub default_step: f64,
}

impl Default for JndParams {
    fn default() -> Self {
        let mut steps = HashMap::new();
        steps.insert(AuditoryDimension::Pitch, 0.3); // ~0.3% frequency DL
        steps.insert(AuditoryDimension::Loudness, 1.0); // ~1 dB intensity DL
        steps.insert(AuditoryDimension::Timbre, 0.05); // spectral centroid JND
        steps.insert(AuditoryDimension::SpatialAzimuth, 1.0); // ~1 degree MAA
        JndParams {
            step_sizes: steps,
            default_step: 1.0,
        }
    }
}

/// Parameters for segregation filtering stage.
#[derive(Debug, Clone)]
pub struct SegregationParams {
    /// Attention weights per stream (0..1). Streams below threshold are filtered.
    pub attention_weights: HashMap<StreamId, f64>,
    /// Threshold below which a stream is considered unattended.
    pub attention_threshold: f64,
    /// Maximum number of simultaneously attended streams.
    pub max_attended: usize,
}

impl Default for SegregationParams {
    fn default() -> Self {
        SegregationParams {
            attention_weights: HashMap::new(),
            attention_threshold: 0.3,
            max_attended: 4,
        }
    }
}

/// Parameters for cognitive filtering stage.
#[derive(Debug, Clone)]
pub struct CognitiveParams {
    /// Maximum information rate in bits per second.
    pub max_bits_per_second: f64,
    /// Capacity limit (Miller's 7 ± 2 items).
    pub channel_capacity: f64,
}

impl Default for CognitiveParams {
    fn default() -> Self {
        CognitiveParams {
            max_bits_per_second: 50.0,
            channel_capacity: 7.0,
        }
    }
}

impl PsychoacousticChannel {
    pub fn new() -> Self {
        PsychoacousticChannel {
            stages: Vec::new(),
            bark_bands: BarkBand::NUM_BANDS,
        }
    }

    pub fn with_masking(mut self, params: MaskingParams) -> Self {
        self.stages.push(ChannelStage::Masking(params));
        self
    }

    pub fn with_jnd(mut self, params: JndParams) -> Self {
        self.stages.push(ChannelStage::JndQuantization(params));
        self
    }

    pub fn with_segregation(mut self, params: SegregationParams) -> Self {
        self.stages.push(ChannelStage::SegregationFilter(params));
        self
    }

    pub fn with_cognitive(mut self, params: CognitiveParams) -> Self {
        self.stages.push(ChannelStage::CognitiveFilter(params));
        self
    }

    /// Compose all stages: apply channel to a signal vector.
    pub fn apply(&self, signal: &[f64]) -> Vec<f64> {
        let mut output = signal.to_vec();
        for stage in &self.stages {
            output = match stage {
                ChannelStage::Masking(p) => self.apply_masking(&output, p),
                ChannelStage::JndQuantization(p) => self.apply_jnd(&output, p),
                ChannelStage::SegregationFilter(p) => self.apply_segregation(&output, p),
                ChannelStage::CognitiveFilter(p) => self.apply_cognitive(&output, p),
            };
        }
        output
    }

    /// Apply channel transformation to a joint distribution matrix.
    pub fn apply_to_joint(&self, joint: &[Vec<f64>]) -> OptimizerResult<Vec<Vec<f64>>> {
        if joint.is_empty() {
            return Err(OptimizerError::NumericalError("Empty joint".into()));
        }

        let rows = joint.len();
        let cols = joint[0].len();

        // Build the channel transfer matrix: how acoustic bins map to perceived bins.
        let transfer = self.build_transfer_matrix(cols);
        let new_cols = transfer[0].len();

        // P(D, P) = P(D, A) * T(P|A)
        let mut result = vec![vec![0.0; new_cols]; rows];
        for i in 0..rows {
            for j in 0..cols {
                for k in 0..new_cols {
                    result[i][k] += joint[i][j] * transfer[j][k];
                }
            }
        }

        Ok(result)
    }

    /// Build transfer matrix T(P|A) representing the channel transformation.
    fn build_transfer_matrix(&self, n: usize) -> Vec<Vec<f64>> {
        let mut matrix: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut row = vec![0.0; n];
                row[i] = 1.0;
                row
            })
            .collect();

        for stage in &self.stages {
            matrix = match stage {
                ChannelStage::Masking(p) => self.transform_matrix_masking(&matrix, p),
                ChannelStage::JndQuantization(p) => self.transform_matrix_jnd(&matrix, p),
                _ => matrix,
            };
        }

        matrix
    }

    fn apply_masking(&self, signal: &[f64], params: &MaskingParams) -> Vec<f64> {
        let mut output = signal.to_vec();
        let n = signal.len();
        let bands = params.thresholds.len().min(n);

        for i in 0..n {
            let band_idx = (i * bands / n).min(bands - 1);
            let threshold = params.thresholds[band_idx];
            // Zero out if signal component is below masking threshold
            if output[i].abs() < threshold {
                output[i] = 0.0;
            }
        }
        output
    }

    fn apply_jnd(&self, signal: &[f64], params: &JndParams) -> Vec<f64> {
        let step = params.default_step;
        signal
            .iter()
            .map(|&v| {
                if step > 0.0 {
                    (v / step).round() * step
                } else {
                    v
                }
            })
            .collect()
    }

    fn apply_segregation(&self, signal: &[f64], params: &SegregationParams) -> Vec<f64> {
        // Attenuate unattended components
        let max_attended = params.max_attended;
        let mut attended_count = 0;
        signal
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let stream_id = StreamId(i as u32);
                let weight = params
                    .attention_weights
                    .get(&stream_id)
                    .copied()
                    .unwrap_or(1.0);
                if weight >= params.attention_threshold && attended_count < max_attended {
                    attended_count += 1;
                    v * weight
                } else {
                    v * 0.1 // Severe attenuation for unattended
                }
            })
            .collect()
    }

    fn apply_cognitive(&self, signal: &[f64], params: &CognitiveParams) -> Vec<f64> {
        // Limit effective information: clamp number of distinguishable levels
        let max_levels = params.channel_capacity as usize;
        if signal.is_empty() {
            return signal.to_vec();
        }

        let (mn, mx) = min_max(signal);
        let range = (mx - mn).max(1e-12);
        let step = range / max_levels as f64;

        signal
            .iter()
            .map(|&v| {
                if step > 0.0 {
                    let quantized = ((v - mn) / step).round() * step + mn;
                    quantized.min(mx).max(mn)
                } else {
                    v
                }
            })
            .collect()
    }

    fn transform_matrix_masking(
        &self,
        matrix: &[Vec<f64>],
        params: &MaskingParams,
    ) -> Vec<Vec<f64>> {
        let n = matrix.len();
        let bands = params.thresholds.len();
        let mut result = matrix.to_vec();

        for i in 0..n {
            let band_idx = (i * bands / n.max(1)).min(bands - 1);
            let threshold = params.thresholds[band_idx];
            // Attenuate rows corresponding to masked bins
            let energy: f64 = result[i].iter().map(|x| x * x).sum::<f64>().sqrt();
            if energy < threshold {
                for j in 0..result[i].len() {
                    result[i][j] = if i == j { 0.5 } else { 0.5 / n as f64 };
                }
            }
        }
        result
    }

    fn transform_matrix_jnd(
        &self,
        matrix: &[Vec<f64>],
        params: &JndParams,
    ) -> Vec<Vec<f64>> {
        let n = matrix.len();
        if n == 0 {
            return matrix.to_vec();
        }

        let step = params.default_step;
        let jnd_bins = ((n as f64) / step.max(1.0)).ceil() as usize;
        let jnd_bins = jnd_bins.max(1).min(n);

        // Merge bins within JND
        let mut result = vec![vec![0.0; jnd_bins]; n];
        for i in 0..n {
            let target = (i * jnd_bins / n).min(jnd_bins - 1);
            for j in 0..n {
                let t2 = (j * jnd_bins / n).min(jnd_bins - 1);
                result[i][t2] += matrix[i][j];
            }
            // Renormalize row
            let row_sum: f64 = result[i].iter().sum();
            if row_sum > 0.0 {
                for v in &mut result[i] {
                    *v /= row_sum;
                }
            }
        }
        result
    }

    /// Compute information loss from channel: L(σ) = I(D;A) - I_ψ(D;A).
    pub fn information_loss(
        &self,
        estimator: &MutualInformationEstimator,
        joint: &[Vec<f64>],
    ) -> OptimizerResult<f64> {
        let i_da = estimator.mutual_information(joint)?;
        let i_psi = estimator.psychoacoustic_mutual_information(joint, self)?;
        Ok((i_da - i_psi).max(0.0))
    }
}

impl Default for PsychoacousticChannel {
    fn default() -> Self {
        PsychoacousticChannel::new()
            .with_masking(MaskingParams::default())
            .with_jnd(JndParams::default())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// InformationLossBound
// ─────────────────────────────────────────────────────────────────────────────

/// Bounds on information loss through the psychoacoustic channel (Theorem 3).
///
/// L(σ) = I(D;A) - I_ψ(D;A) ≤ ε
#[derive(Debug, Clone)]
pub struct InformationLossBound {
    /// Maximum allowable information loss in bits.
    pub epsilon: f64,
    /// Masking model parameters for margin function.
    pub masking_slope: f64,
    /// JND model parameters for margin function.
    pub jnd_step: f64,
    /// Number of Bark bands in the model.
    pub num_bands: usize,
}

impl InformationLossBound {
    pub fn new(epsilon: f64) -> Self {
        InformationLossBound {
            epsilon,
            masking_slope: 10.0,
            jnd_step: 1.0,
            num_bands: BarkBand::NUM_BANDS,
        }
    }

    pub fn with_masking_slope(mut self, slope: f64) -> Self {
        self.masking_slope = slope;
        self
    }

    pub fn with_jnd_step(mut self, step: f64) -> Self {
        self.jnd_step = step;
        self
    }

    /// Compute information loss L(σ) = I(D;A) - I_ψ(D;A).
    pub fn compute_loss(
        &self,
        estimator: &MutualInformationEstimator,
        joint: &[Vec<f64>],
        channel: &PsychoacousticChannel,
    ) -> OptimizerResult<f64> {
        let i_da = estimator.mutual_information(joint)?;
        let i_psi = estimator.psychoacoustic_mutual_information(joint, channel)?;
        Ok((i_da - i_psi).max(0.0))
    }

    /// Check if L(σ) ≤ ε (Theorem 3 bound).
    pub fn check_bound(
        &self,
        estimator: &MutualInformationEstimator,
        joint: &[Vec<f64>],
        channel: &PsychoacousticChannel,
    ) -> OptimizerResult<BoundCheckResult> {
        let loss = self.compute_loss(estimator, joint, channel)?;
        let margin = self.epsilon - loss;
        let satisfied = loss <= self.epsilon;

        Ok(BoundCheckResult {
            loss,
            epsilon: self.epsilon,
            margin,
            satisfied,
            margin_function: self.compute_margin_function(),
        })
    }

    /// Compute the margin function g(ε) from masking and JND models.
    ///
    /// g(ε) upper-bounds the achievable information rate given masking slope
    /// and JND resolution: g(ε) = Σ_b log2(1 + SNR_b / JND_b)
    pub fn compute_margin_function(&self) -> f64 {
        let mut g = 0.0;
        for b in 0..self.num_bands {
            let band = BarkBand(b as u8);
            let bw = band.bandwidth();
            // Effective SNR in the band, accounting for masking
            let snr = self.masking_slope * (bw / 100.0).log2().max(0.0);
            let effective_capacity = (1.0 + snr / self.jnd_step).log2();
            g += effective_capacity;
        }
        g
    }

    /// Compute per-band loss decomposition.
    pub fn per_band_loss(
        &self,
        estimator: &MutualInformationEstimator,
        joint_per_band: &[Vec<Vec<f64>>],
        channel: &PsychoacousticChannel,
    ) -> OptimizerResult<Vec<f64>> {
        let mut losses = Vec::new();
        for band_joint in joint_per_band {
            let loss = channel.information_loss(estimator, band_joint)?;
            losses.push(loss);
        }
        Ok(losses)
    }
}

/// Result of checking the information loss bound.
#[derive(Debug, Clone)]
pub struct BoundCheckResult {
    pub loss: f64,
    pub epsilon: f64,
    pub margin: f64,
    pub satisfied: bool,
    pub margin_function: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// DiscriminabilityEstimator
// ─────────────────────────────────────────────────────────────────────────────

/// Signal detection theory-based discriminability estimation.
#[derive(Debug, Clone)]
pub struct DiscriminabilityEstimator {
    /// Number of points for ROC curve computation.
    pub roc_resolution: usize,
    /// Correction for extreme hit/false alarm rates.
    pub rate_correction: f64,
}

impl Default for DiscriminabilityEstimator {
    fn default() -> Self {
        DiscriminabilityEstimator {
            roc_resolution: 100,
            rate_correction: 0.01,
        }
    }
}

impl DiscriminabilityEstimator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute d' (d-prime) from signal detection theory.
    ///
    /// d' = z(hit_rate) - z(false_alarm_rate)
    /// where z is the inverse normal CDF.
    pub fn d_prime(&self, hit_rate: f64, false_alarm_rate: f64) -> f64 {
        let hr = hit_rate.clamp(self.rate_correction, 1.0 - self.rate_correction);
        let far = false_alarm_rate.clamp(self.rate_correction, 1.0 - self.rate_correction);
        self.inv_normal_cdf(hr) - self.inv_normal_cdf(far)
    }

    /// Compute d' from signal and noise distributions (assumed Gaussian).
    pub fn d_prime_from_distributions(
        &self,
        signal_mean: f64,
        signal_std: f64,
        noise_mean: f64,
        noise_std: f64,
    ) -> f64 {
        let pooled_std = ((signal_std * signal_std + noise_std * noise_std) / 2.0).sqrt();
        if pooled_std < 1e-12 {
            return 0.0;
        }
        (signal_mean - noise_mean).abs() / pooled_std
    }

    /// Compute d'_model for a mapping configuration.
    ///
    /// Averages pairwise d' across all stream pairs for each auditory dimension.
    pub fn d_prime_model(&self, config: &MappingConfig) -> f64 {
        let streams: Vec<&StreamMapping> = config.stream_params.values().collect();
        if streams.len() < 2 {
            return f64::INFINITY;
        }

        let mut total_dprime = 0.0;
        let mut count = 0;

        for i in 0..streams.len() {
            for j in (i + 1)..streams.len() {
                let freq_diff = (streams[i].frequency_hz - streams[j].frequency_hz).abs();
                let avg_freq = (streams[i].frequency_hz + streams[j].frequency_hz) / 2.0;
                // Weber's law for frequency discrimination: ΔF/F ≈ 0.003
                let jnd = avg_freq * 0.003;
                if jnd > 0.0 {
                    total_dprime += freq_diff / jnd;
                }
                count += 1;

                let amp_diff = (streams[i].amplitude_db - streams[j].amplitude_db).abs();
                // Intensity JND ≈ 1 dB
                total_dprime += amp_diff / 1.0;
                count += 1;
            }
        }

        if count > 0 {
            total_dprime / count as f64
        } else {
            0.0
        }
    }

    /// Compute ROC curve from signal and noise score distributions.
    pub fn roc_curve(&self, signal_scores: &[f64], noise_scores: &[f64]) -> RocCurve {
        let mut all_thresholds: Vec<f64> = signal_scores
            .iter()
            .chain(noise_scores.iter())
            .copied()
            .collect();
        all_thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        all_thresholds.dedup();

        let mut points = Vec::new();
        points.push(RocPoint {
            threshold: f64::NEG_INFINITY,
            hit_rate: 1.0,
            false_alarm_rate: 1.0,
        });

        for &threshold in &all_thresholds {
            let hr = signal_scores.iter().filter(|&&s| s >= threshold).count() as f64
                / signal_scores.len().max(1) as f64;
            let far = noise_scores.iter().filter(|&&s| s >= threshold).count() as f64
                / noise_scores.len().max(1) as f64;
            points.push(RocPoint {
                threshold,
                hit_rate: hr,
                false_alarm_rate: far,
            });
        }

        points.push(RocPoint {
            threshold: f64::INFINITY,
            hit_rate: 0.0,
            false_alarm_rate: 0.0,
        });

        // Sort by false alarm rate
        points.sort_by(|a, b| {
            a.false_alarm_rate
                .partial_cmp(&b.false_alarm_rate)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let auc = self.compute_auc(&points);

        RocCurve { points, auc }
    }

    /// Compute AUC (Area Under ROC Curve) via trapezoidal rule.
    fn compute_auc(&self, points: &[RocPoint]) -> f64 {
        let mut auc = 0.0;
        for i in 1..points.len() {
            let dx = points[i].false_alarm_rate - points[i - 1].false_alarm_rate;
            let avg_y = (points[i].hit_rate + points[i - 1].hit_rate) / 2.0;
            auc += dx * avg_y;
        }
        auc.abs()
    }

    /// Hit rate for a given criterion on signal distribution.
    pub fn hit_rate(&self, signal_mean: f64, signal_std: f64, criterion: f64) -> f64 {
        if signal_std < 1e-12 {
            return if signal_mean >= criterion { 1.0 } else { 0.0 };
        }
        let z = (criterion - signal_mean) / signal_std;
        1.0 - self.normal_cdf(z)
    }

    /// False alarm rate for a given criterion on noise distribution.
    pub fn false_alarm_rate(&self, noise_mean: f64, noise_std: f64, criterion: f64) -> f64 {
        if noise_std < 1e-12 {
            return if noise_mean >= criterion { 1.0 } else { 0.0 };
        }
        let z = (criterion - noise_mean) / noise_std;
        1.0 - self.normal_cdf(z)
    }

    /// Multi-class discriminability: average pairwise d'.
    pub fn multi_class_dprime(&self, class_means: &[f64], class_stds: &[f64]) -> f64 {
        let n = class_means.len();
        if n < 2 || class_stds.len() != n {
            return 0.0;
        }

        let mut total = 0.0;
        let mut count = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                total += self.d_prime_from_distributions(
                    class_means[i],
                    class_stds[i],
                    class_means[j],
                    class_stds[j],
                );
                count += 1;
            }
        }
        if count > 0 {
            total / count as f64
        } else {
            0.0
        }
    }

    /// Confusion matrix-based discriminability.
    pub fn confusion_dprime(&self, confusion_matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = confusion_matrix.len();
        let mut dprime_matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            let row_sum_i: f64 = confusion_matrix[i].iter().sum();
            for j in (i + 1)..n {
                let row_sum_j: f64 = confusion_matrix[j].iter().sum();
                if row_sum_i > 0.0 && row_sum_j > 0.0 {
                    let hr = confusion_matrix[i][i] / row_sum_i;
                    let far = confusion_matrix[j][i] / row_sum_j;
                    let d = self.d_prime(hr.max(0.01).min(0.99), far.max(0.01).min(0.99));
                    dprime_matrix[i][j] = d;
                    dprime_matrix[j][i] = d;
                }
            }
        }
        dprime_matrix
    }

    // Rational approximation to the inverse normal CDF (probit function).
    fn inv_normal_cdf(&self, p: f64) -> f64 {
        let p = p.clamp(1e-10, 1.0 - 1e-10);
        // Beasley-Springer-Moro algorithm
        let a = [
            -3.969683028665376e+01,
            2.209460984245205e+02,
            -2.759285104469687e+02,
            1.383577518672690e+02,
            -3.066479806614716e+01,
            2.506628277459239e+00,
        ];
        let b = [
            -5.447609879822406e+01,
            1.615858368580409e+02,
            -1.556989798598866e+02,
            6.680131188771972e+01,
            -1.328068155288572e+01,
        ];
        let c = [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e+00,
            -2.549732539343734e+00,
            4.374664141464968e+00,
            2.938163982698783e+00,
        ];
        let d = [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e+00,
            3.754408661907416e+00,
        ];

        let p_low = 0.02425;
        let p_high = 1.0 - p_low;

        if p < p_low {
            let q = (-2.0 * p.ln()).sqrt();
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        } else if p <= p_high {
            let q = p - 0.5;
            let r = q * q;
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        } else {
            let q = (-2.0 * (1.0 - p).ln()).sqrt();
            -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        }
    }

    fn normal_cdf(&self, x: f64) -> f64 {
        0.5 * (1.0 + erf(x / 2.0_f64.sqrt()))
    }
}

/// Point on an ROC curve.
#[derive(Debug, Clone)]
pub struct RocPoint {
    pub threshold: f64,
    pub hit_rate: f64,
    pub false_alarm_rate: f64,
}

/// Full ROC curve with AUC.
#[derive(Debug, Clone)]
pub struct RocCurve {
    pub points: Vec<RocPoint>,
    pub auc: f64,
}

/// Error function approximation (Abramowitz and Stegun 7.1.26).
fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let result = 1.0 - poly * (-x * x).exp();
    if x >= 0.0 {
        result
    } else {
        -result
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_joint(n: usize) -> Vec<Vec<f64>> {
        vec![vec![1.0 / (n * n) as f64; n]; n]
    }

    fn identity_joint(n: usize) -> Vec<Vec<f64>> {
        let mut joint = vec![vec![0.0; n]; n];
        for i in 0..n {
            joint[i][i] = 1.0 / n as f64;
        }
        joint
    }

    #[test]
    fn test_entropy_uniform() {
        let est = MutualInformationEstimator::new(8, 2.0);
        let dist = vec![1.0; 8];
        let h = est.entropy(&dist);
        assert!((h - 3.0).abs() < 0.01, "H(uniform_8) ≈ 3 bits, got {}", h);
    }

    #[test]
    fn test_entropy_deterministic() {
        let est = MutualInformationEstimator::new(4, 2.0);
        let dist = vec![1.0, 0.0, 0.0, 0.0];
        let h = est.entropy(&dist);
        assert!(h < 0.1, "H(deterministic) ≈ 0, got {}", h);
    }

    #[test]
    fn test_mi_identity_channel() {
        let est = MutualInformationEstimator::new(4, 2.0);
        let joint = identity_joint(4);
        let mi = est.mutual_information(&joint).unwrap();
        assert!(mi > 1.5, "I(D;A) for identity should be ~2 bits, got {}", mi);
    }

    #[test]
    fn test_mi_independent() {
        let est = MutualInformationEstimator::new(4, 2.0);
        let joint = uniform_joint(4);
        let mi = est.mutual_information(&joint).unwrap();
        assert!(mi < 0.1, "I(D;A) for independent should be ~0, got {}", mi);
    }

    #[test]
    fn test_kl_divergence_same() {
        let est = MutualInformationEstimator::new(4, 2.0);
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let kl = est.kl_divergence(&p, &p).unwrap();
        assert!(kl < 0.01, "KL(P||P) should be ~0, got {}", kl);
    }

    #[test]
    fn test_kl_divergence_different() {
        let est = MutualInformationEstimator::new(4, 2.0);
        let p = vec![0.9, 0.05, 0.025, 0.025];
        let q = vec![0.25, 0.25, 0.25, 0.25];
        let kl = est.kl_divergence(&p, &q).unwrap();
        assert!(kl > 0.5, "KL(peaked||uniform) should be > 0.5, got {}", kl);
    }

    #[test]
    fn test_psychoacoustic_channel_reduces_mi() {
        let est = MutualInformationEstimator::new(8, 2.0);
        let joint = identity_joint(8);
        let i_raw = est.mutual_information(&joint).unwrap();

        let channel = PsychoacousticChannel::default();
        let i_psi = est.psychoacoustic_mutual_information(&joint, &channel).unwrap();

        assert!(
            i_psi <= i_raw + 0.01,
            "I_ψ should not exceed I(D;A): {} > {}",
            i_psi,
            i_raw
        );
    }

    #[test]
    fn test_channel_capacity_binary_symmetric() {
        let est = MutualInformationEstimator::new(2, 2.0);
        // Binary symmetric channel with crossover prob 0 => capacity = 1 bit
        let channel = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let cap = est.channel_capacity(&channel, 100, 1e-8).unwrap();
        assert!(
            (cap - 1.0).abs() < 0.1,
            "BSC(0) capacity should be ~1 bit, got {}",
            cap
        );
    }

    #[test]
    fn test_information_loss_bound() {
        let est = MutualInformationEstimator::new(4, 2.0);
        let joint = identity_joint(4);
        let channel = PsychoacousticChannel::new();
        let bound = InformationLossBound::new(1.0);

        let result = bound.check_bound(&est, &joint, &channel).unwrap();
        // No-op channel => loss ≈ 0
        assert!(result.satisfied, "Loss {} should be ≤ ε={}", result.loss, result.epsilon);
    }

    #[test]
    fn test_margin_function_positive() {
        let bound = InformationLossBound::new(0.5);
        let g = bound.compute_margin_function();
        assert!(g > 0.0, "Margin function should be positive, got {}", g);
    }

    #[test]
    fn test_d_prime_computation() {
        let disc = DiscriminabilityEstimator::new();
        let dp = disc.d_prime(0.9, 0.1);
        assert!(dp > 2.0, "d' for hr=0.9, far=0.1 should be > 2, got {}", dp);
    }

    #[test]
    fn test_d_prime_from_distributions() {
        let disc = DiscriminabilityEstimator::new();
        let dp = disc.d_prime_from_distributions(2.0, 1.0, 0.0, 1.0);
        assert!(
            (dp - 2.0).abs() < 0.01,
            "d' should be 2.0, got {}",
            dp
        );
    }

    #[test]
    fn test_roc_curve() {
        let disc = DiscriminabilityEstimator::new();
        let signal = vec![3.0, 4.0, 5.0, 6.0, 7.0];
        let noise = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let roc = disc.roc_curve(&signal, &noise);
        assert!(roc.auc > 0.5, "AUC should be > 0.5 for separable classes, got {}", roc.auc);
    }

    #[test]
    fn test_multi_class_dprime() {
        let disc = DiscriminabilityEstimator::new();
        let means = vec![0.0, 3.0, 6.0];
        let stds = vec![1.0, 1.0, 1.0];
        let dp = disc.multi_class_dprime(&means, &stds);
        assert!(dp > 2.0, "Multi-class d' should be > 2 for well-separated classes, got {}", dp);
    }

    #[test]
    fn test_conditional_entropy() {
        let est = MutualInformationEstimator::new(4, 2.0);
        let joint = identity_joint(4);
        let h_xy = est.conditional_entropy(&joint).unwrap();
        // For identity: H(X|Y) ≈ 0
        assert!(h_xy < 0.5, "H(X|Y) for identity should be ~0, got {}", h_xy);
    }

    #[test]
    fn test_js_divergence_symmetric() {
        let est = MutualInformationEstimator::new(4, 2.0);
        let p = vec![0.7, 0.1, 0.1, 0.1];
        let q = vec![0.1, 0.7, 0.1, 0.1];
        let js1 = est.js_divergence(&p, &q).unwrap();
        let js2 = est.js_divergence(&q, &p).unwrap();
        assert!(
            (js1 - js2).abs() < 0.001,
            "JS divergence should be symmetric: {} vs {}",
            js1,
            js2
        );
    }

    #[test]
    fn test_hit_rate_computation() {
        let disc = DiscriminabilityEstimator::new();
        let hr = disc.hit_rate(3.0, 1.0, 2.0);
        assert!(hr > 0.8, "Hit rate should be > 0.8 for signal well above criterion, got {}", hr);
    }

    #[test]
    fn test_confusion_dprime() {
        let disc = DiscriminabilityEstimator::new();
        let confusion = vec![
            vec![90.0, 5.0, 5.0],
            vec![5.0, 85.0, 10.0],
            vec![5.0, 10.0, 85.0],
        ];
        let dprimes = disc.confusion_dprime(&confusion);
        assert!(dprimes[0][1] > 0.0);
        assert!(dprimes[0][2] > 0.0);
        assert_eq!(dprimes[0][0], 0.0); // Self comparison
    }

    #[test]
    fn test_estimate_from_samples() {
        let est = MutualInformationEstimator::new(8, 2.0);
        let channel = PsychoacousticChannel::new();
        let data: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64]).collect();
        let audio: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64 * 2.0]).collect();
        let mi = est.estimate_from_samples(&data, &audio, &channel).unwrap();
        assert!(mi > 0.0, "MI from correlated samples should be positive, got {}", mi);
    }
}

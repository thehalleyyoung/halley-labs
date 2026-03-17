//! Constraint system for sonification optimization.
//!
//! Defines masking clearance, JND sufficiency, segregation, cognitive load,
//! frequency/amplitude range, latency, and custom constraints.

use std::collections::HashMap;
use std::fmt;

use crate::{
    AuditoryDimension, BarkBand, MappingConfig, OptimizerError, OptimizerResult,
    ParameterId, SegregationPredicate, StreamId, StreamMapping,
};

// ─────────────────────────────────────────────────────────────────────────────
// Constraint enum
// ─────────────────────────────────────────────────────────────────────────────

/// A single constraint on the mapping configuration.
#[derive(Clone)]
pub enum Constraint {
    /// The stream's signal must exceed the masking threshold by `min_margin_db`.
    MaskingClearance {
        stream_id: StreamId,
        min_margin_db: f64,
    },
    /// Two parameters along a given dimension must differ by at least `min_jnd_multiples` JNDs.
    JndSufficiency {
        param1: ParameterId,
        param2: ParameterId,
        dimension: AuditoryDimension,
        min_jnd_multiples: f64,
    },
    /// Two streams must be perceptually segregated per the given predicates.
    SegregationRequired {
        stream1_id: StreamId,
        stream2_id: StreamId,
        predicates: Vec<SegregationPredicate>,
    },
    /// Global cognitive load budget: max simultaneous streams and total load.
    CognitiveLoadBudget {
        max_streams: usize,
        max_load: f64,
    },
    /// Frequency must be within [min_hz, max_hz].
    FrequencyRange {
        min_hz: f64,
        max_hz: f64,
    },
    /// Amplitude must be within [min_db, max_db].
    AmplitudeRange {
        min_db: f64,
        max_db: f64,
    },
    /// Rendering latency must not exceed `max_ms`.
    LatencyBound {
        max_ms: f64,
    },
    /// User-defined constraint via a closure.
    Custom(Box<dyn ConstraintFn>),
}

impl fmt::Debug for Constraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Constraint::MaskingClearance { stream_id, min_margin_db } => {
                write!(f, "MaskingClearance({}, {}dB)", stream_id, min_margin_db)
            }
            Constraint::JndSufficiency { param1, param2, dimension, min_jnd_multiples } => {
                write!(f, "JndSufficiency({}, {}, {:?}, {}x)", param1, param2, dimension, min_jnd_multiples)
            }
            Constraint::SegregationRequired { stream1_id, stream2_id, .. } => {
                write!(f, "SegregationRequired({}, {})", stream1_id, stream2_id)
            }
            Constraint::CognitiveLoadBudget { max_streams, max_load } => {
                write!(f, "CognitiveLoadBudget({}, {})", max_streams, max_load)
            }
            Constraint::FrequencyRange { min_hz, max_hz } => {
                write!(f, "FrequencyRange({}Hz..{}Hz)", min_hz, max_hz)
            }
            Constraint::AmplitudeRange { min_db, max_db } => {
                write!(f, "AmplitudeRange({}dB..{}dB)", min_db, max_db)
            }
            Constraint::LatencyBound { max_ms } => {
                write!(f, "LatencyBound({}ms)", max_ms)
            }
            Constraint::Custom(_) => write!(f, "Custom(...)"),
        }
    }
}

/// Trait for custom constraint functions.
pub trait ConstraintFn: Send + Sync {
    fn evaluate(&self, config: &MappingConfig) -> ConstraintEvaluation;
    fn name(&self) -> &str;
    fn clone_box(&self) -> Box<dyn ConstraintFn>;
}

impl Clone for Box<dyn ConstraintFn> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Result of evaluating a single constraint.
#[derive(Debug, Clone)]
pub struct ConstraintEvaluation {
    pub satisfied: bool,
    pub violation_amount: f64,
    pub margin: f64,
    pub message: String,
}

impl ConstraintEvaluation {
    pub fn satisfied(margin: f64) -> Self {
        ConstraintEvaluation {
            satisfied: true,
            violation_amount: 0.0,
            margin,
            message: String::new(),
        }
    }

    pub fn violated(amount: f64, message: String) -> Self {
        ConstraintEvaluation {
            satisfied: false,
            violation_amount: amount,
            margin: -amount,
            message,
        }
    }
}

impl Constraint {
    /// Evaluate this constraint against a mapping configuration.
    pub fn evaluate(&self, config: &MappingConfig) -> ConstraintEvaluation {
        match self {
            Constraint::MaskingClearance { stream_id, min_margin_db } => {
                self.eval_masking_clearance(config, *stream_id, *min_margin_db)
            }
            Constraint::JndSufficiency { param1, param2, dimension, min_jnd_multiples } => {
                self.eval_jnd_sufficiency(config, param1, param2, *dimension, *min_jnd_multiples)
            }
            Constraint::SegregationRequired { stream1_id, stream2_id, predicates } => {
                self.eval_segregation(config, *stream1_id, *stream2_id, predicates)
            }
            Constraint::CognitiveLoadBudget { max_streams, max_load } => {
                self.eval_cognitive_load(config, *max_streams, *max_load)
            }
            Constraint::FrequencyRange { min_hz, max_hz } => {
                self.eval_frequency_range(config, *min_hz, *max_hz)
            }
            Constraint::AmplitudeRange { min_db, max_db } => {
                self.eval_amplitude_range(config, *min_db, *max_db)
            }
            Constraint::LatencyBound { max_ms } => {
                self.eval_latency_bound(config, *max_ms)
            }
            Constraint::Custom(func) => func.evaluate(config),
        }
    }

    fn eval_masking_clearance(
        &self,
        config: &MappingConfig,
        stream_id: StreamId,
        min_margin_db: f64,
    ) -> ConstraintEvaluation {
        if let Some(mapping) = config.stream_params.get(&stream_id) {
            // Compute masking threshold from other streams in same Bark band
            let mut masking_level = 0.0_f64;
            for (other_id, other) in &config.stream_params {
                if *other_id != stream_id && other.bark_band == mapping.bark_band {
                    // Simplified spreading function
                    let level = other.amplitude_db - 6.0; // Approximate masking level
                    masking_level = masking_level.max(level);
                }
            }

            let margin = mapping.amplitude_db - masking_level;
            if margin >= min_margin_db {
                ConstraintEvaluation::satisfied(margin - min_margin_db)
            } else {
                ConstraintEvaluation::violated(
                    min_margin_db - margin,
                    format!("Stream {} masked: margin={:.1}dB < {:.1}dB", stream_id, margin, min_margin_db),
                )
            }
        } else {
            ConstraintEvaluation::satisfied(f64::INFINITY)
        }
    }

    fn eval_jnd_sufficiency(
        &self,
        config: &MappingConfig,
        param1: &ParameterId,
        param2: &ParameterId,
        dimension: AuditoryDimension,
        min_jnd_multiples: f64,
    ) -> ConstraintEvaluation {
        let v1 = config.get_param(param1).unwrap_or(0.0);
        let v2 = config.get_param(param2).unwrap_or(0.0);

        let jnd_size = match dimension {
            AuditoryDimension::Pitch => {
                let avg = (v1 + v2) / 2.0;
                avg.abs() * 0.003 // Weber fraction for frequency
            }
            AuditoryDimension::Loudness => 1.0,     // ~1 dB
            AuditoryDimension::Timbre => 0.05,       // Spectral centroid JND
            AuditoryDimension::SpatialAzimuth => 1.0, // ~1 degree MAA
            AuditoryDimension::SpatialElevation => 4.0,
            AuditoryDimension::Duration => 10.0,     // ~10ms
            AuditoryDimension::AttackTime => 5.0,    // ~5ms
        };

        let separation = (v1 - v2).abs();
        let jnd_multiples = if jnd_size > 1e-12 {
            separation / jnd_size
        } else {
            f64::INFINITY
        };

        if jnd_multiples >= min_jnd_multiples {
            ConstraintEvaluation::satisfied(jnd_multiples - min_jnd_multiples)
        } else {
            ConstraintEvaluation::violated(
                min_jnd_multiples - jnd_multiples,
                format!(
                    "JND insufficient: {:.1}x < {:.1}x for {:?}",
                    jnd_multiples, min_jnd_multiples, dimension
                ),
            )
        }
    }

    fn eval_segregation(
        &self,
        config: &MappingConfig,
        stream1: StreamId,
        stream2: StreamId,
        predicates: &[SegregationPredicate],
    ) -> ConstraintEvaluation {
        let s1 = config.stream_params.get(&stream1);
        let s2 = config.stream_params.get(&stream2);

        match (s1, s2) {
            (Some(m1), Some(m2)) => {
                for pred in predicates {
                    match pred {
                        SegregationPredicate::MinFrequencySeparation(min_sep) => {
                            let sep = (m1.frequency_hz - m2.frequency_hz).abs();
                            if sep < *min_sep {
                                return ConstraintEvaluation::violated(
                                    min_sep - sep,
                                    format!(
                                        "Freq separation {:.0}Hz < {:.0}Hz",
                                        sep, min_sep
                                    ),
                                );
                            }
                        }
                        SegregationPredicate::DifferentBarkBands => {
                            if m1.bark_band == m2.bark_band {
                                return ConstraintEvaluation::violated(
                                    1.0,
                                    format!(
                                        "Streams {} and {} in same Bark band {}",
                                        stream1, stream2, m1.bark_band.0
                                    ),
                                );
                            }
                        }
                        SegregationPredicate::MinOnsetAsynchrony(min_ms) => {
                            // Would need temporal info; pass for static analysis
                            let onset_diff = m1
                                .dimension_values
                                .get(&AuditoryDimension::Duration)
                                .copied()
                                .unwrap_or(0.0)
                                - m2.dimension_values
                                    .get(&AuditoryDimension::Duration)
                                    .copied()
                                    .unwrap_or(0.0);
                            if onset_diff.abs() < *min_ms {
                                return ConstraintEvaluation::violated(
                                    min_ms - onset_diff.abs(),
                                    format!("Onset asynchrony {:.1}ms < {:.1}ms", onset_diff.abs(), min_ms),
                                );
                            }
                        }
                        SegregationPredicate::HarmonicSeparation => {
                            let ratio = if m2.frequency_hz > 0.0 {
                                m1.frequency_hz / m2.frequency_hz
                            } else {
                                0.0
                            };
                            let nearest_harmonic = ratio.round();
                            let deviation = (ratio - nearest_harmonic).abs();
                            if deviation < 0.05 && nearest_harmonic > 0.0 {
                                return ConstraintEvaluation::violated(
                                    0.05 - deviation,
                                    format!("Harmonic overlap: ratio={:.3}", ratio),
                                );
                            }
                        }
                    }
                }
                ConstraintEvaluation::satisfied(1.0)
            }
            _ => ConstraintEvaluation::satisfied(f64::INFINITY),
        }
    }

    fn eval_cognitive_load(
        &self,
        config: &MappingConfig,
        max_streams: usize,
        max_load: f64,
    ) -> ConstraintEvaluation {
        let n = config.stream_count();
        if n > max_streams {
            return ConstraintEvaluation::violated(
                (n - max_streams) as f64,
                format!("{} streams exceeds max {}", n, max_streams),
            );
        }

        // Estimate cognitive load: base + per-stream + interaction terms
        let base_load = 1.0;
        let per_stream = 1.5;
        let interaction = 0.2 * (n as f64 * (n as f64 - 1.0)) / 2.0;
        let total_load = base_load + per_stream * n as f64 + interaction;

        if total_load <= max_load {
            ConstraintEvaluation::satisfied(max_load - total_load)
        } else {
            ConstraintEvaluation::violated(
                total_load - max_load,
                format!("Cognitive load {:.1} exceeds max {:.1}", total_load, max_load),
            )
        }
    }

    fn eval_frequency_range(
        &self,
        config: &MappingConfig,
        min_hz: f64,
        max_hz: f64,
    ) -> ConstraintEvaluation {
        let mut worst_margin = f64::INFINITY;
        for mapping in config.stream_params.values() {
            let f = mapping.frequency_hz;
            if f < min_hz {
                return ConstraintEvaluation::violated(
                    min_hz - f,
                    format!("Freq {:.0}Hz below min {:.0}Hz", f, min_hz),
                );
            }
            if f > max_hz {
                return ConstraintEvaluation::violated(
                    f - max_hz,
                    format!("Freq {:.0}Hz above max {:.0}Hz", f, max_hz),
                );
            }
            let margin = (f - min_hz).min(max_hz - f);
            worst_margin = worst_margin.min(margin);
        }
        ConstraintEvaluation::satisfied(worst_margin)
    }

    fn eval_amplitude_range(
        &self,
        config: &MappingConfig,
        min_db: f64,
        max_db: f64,
    ) -> ConstraintEvaluation {
        let mut worst_margin = f64::INFINITY;
        for mapping in config.stream_params.values() {
            let a = mapping.amplitude_db;
            if a < min_db {
                return ConstraintEvaluation::violated(
                    min_db - a,
                    format!("Amp {:.1}dB below min {:.1}dB", a, min_db),
                );
            }
            if a > max_db {
                return ConstraintEvaluation::violated(
                    a - max_db,
                    format!("Amp {:.1}dB above max {:.1}dB", a, max_db),
                );
            }
            let margin = (a - min_db).min(max_db - a);
            worst_margin = worst_margin.min(margin);
        }
        ConstraintEvaluation::satisfied(worst_margin)
    }

    fn eval_latency_bound(&self, config: &MappingConfig, max_ms: f64) -> ConstraintEvaluation {
        // Estimate latency from config: base + per-stream processing
        let base_ms = 5.0;
        let per_stream_ms = 2.0;
        let estimated_ms = base_ms + per_stream_ms * config.stream_count() as f64;

        if estimated_ms <= max_ms {
            ConstraintEvaluation::satisfied(max_ms - estimated_ms)
        } else {
            ConstraintEvaluation::violated(
                estimated_ms - max_ms,
                format!("Latency {:.1}ms exceeds max {:.1}ms", estimated_ms, max_ms),
            )
        }
    }

    /// Name of this constraint for reporting.
    pub fn name(&self) -> String {
        match self {
            Constraint::MaskingClearance { stream_id, .. } => {
                format!("masking_clearance_{}", stream_id)
            }
            Constraint::JndSufficiency { param1, param2, .. } => {
                format!("jnd_{}_{}", param1, param2)
            }
            Constraint::SegregationRequired { stream1_id, stream2_id, .. } => {
                format!("segregation_{}_{}", stream1_id, stream2_id)
            }
            Constraint::CognitiveLoadBudget { .. } => "cognitive_load".into(),
            Constraint::FrequencyRange { .. } => "frequency_range".into(),
            Constraint::AmplitudeRange { .. } => "amplitude_range".into(),
            Constraint::LatencyBound { .. } => "latency_bound".into(),
            Constraint::Custom(f) => format!("custom_{}", f.name()),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ConstraintSet
// ─────────────────────────────────────────────────────────────────────────────

/// Collection of constraints with operations.
#[derive(Clone)]
pub struct ConstraintSet {
    constraints: Vec<(String, Constraint)>,
}

impl Default for ConstraintSet {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ConstraintSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConstraintSet")
            .field("count", &self.constraints.len())
            .field("names", &self.constraint_names())
            .finish()
    }
}

impl ConstraintSet {
    pub fn new() -> Self {
        ConstraintSet {
            constraints: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.constraints.len()
    }

    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    /// Add a constraint to the set.
    pub fn add(&mut self, constraint: Constraint) {
        let name = constraint.name();
        self.constraints.push((name, constraint));
    }

    /// Add a named constraint.
    pub fn add_named(&mut self, name: &str, constraint: Constraint) {
        self.constraints.push((name.to_string(), constraint));
    }

    /// Remove a constraint by name.
    pub fn remove(&mut self, name: &str) -> bool {
        let before = self.constraints.len();
        self.constraints.retain(|(n, _)| n != name);
        self.constraints.len() < before
    }

    /// Merge another constraint set into this one.
    pub fn merge(&mut self, other: &ConstraintSet) {
        for (name, constraint) in &other.constraints {
            self.constraints.push((name.clone(), constraint.clone()));
        }
    }

    /// Intersect: keep only constraints present in both sets (by name).
    pub fn intersect(&self, other: &ConstraintSet) -> ConstraintSet {
        let other_names: std::collections::HashSet<&str> =
            other.constraints.iter().map(|(n, _)| n.as_str()).collect();
        let mut result = ConstraintSet::new();
        for (name, constraint) in &self.constraints {
            if other_names.contains(name.as_str()) {
                result.constraints.push((name.clone(), constraint.clone()));
            }
        }
        result
    }

    /// Check all constraints against a configuration.
    pub fn check_all(&self, config: &MappingConfig) -> ConstraintReport {
        let mut results = Vec::new();
        let mut all_satisfied = true;
        let mut total_violation = 0.0;
        let mut min_margin = f64::INFINITY;

        for (name, constraint) in &self.constraints {
            let eval = constraint.evaluate(config);
            if !eval.satisfied {
                all_satisfied = false;
                total_violation += eval.violation_amount;
            }
            if eval.margin < min_margin {
                min_margin = eval.margin;
            }
            results.push(ConstraintSatisfaction {
                name: name.clone(),
                evaluation: eval,
            });
        }

        ConstraintReport {
            results,
            all_satisfied,
            total_violation,
            min_margin,
        }
    }

    /// Find all violated constraints.
    pub fn find_violated(&self, config: &MappingConfig) -> Vec<(String, ConstraintEvaluation)> {
        self.constraints
            .iter()
            .filter_map(|(name, c)| {
                let eval = c.evaluate(config);
                if !eval.satisfied {
                    Some((name.clone(), eval))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Compute the feasible region as intervals per parameter dimension.
    pub fn compute_feasible_region(&self) -> FeasibleRegion {
        let mut freq_min: f64 = 20.0;
        let mut freq_max: f64 = 20000.0;
        let mut amp_min: f64 = -80.0;
        let mut amp_max: f64 = 100.0;
        let mut max_streams = usize::MAX;
        let mut max_latency = f64::INFINITY;

        for (_, constraint) in &self.constraints {
            match constraint {
                Constraint::FrequencyRange { min_hz, max_hz } => {
                    freq_min = freq_min.max(*min_hz);
                    freq_max = freq_max.min(*max_hz);
                }
                Constraint::AmplitudeRange { min_db, max_db } => {
                    amp_min = amp_min.max(*min_db);
                    amp_max = amp_max.min(*max_db);
                }
                Constraint::CognitiveLoadBudget { max_streams: ms, .. } => {
                    max_streams = max_streams.min(*ms);
                }
                Constraint::LatencyBound { max_ms } => {
                    max_latency = max_latency.min(*max_ms);
                }
                _ => {}
            }
        }

        let mut dimensions = HashMap::new();
        dimensions.insert(
            "frequency".to_string(),
            IntervalDimension {
                min: freq_min,
                max: freq_max,
            },
        );
        dimensions.insert(
            "amplitude".to_string(),
            IntervalDimension {
                min: amp_min,
                max: amp_max,
            },
        );

        let feasible = freq_min <= freq_max && amp_min <= amp_max;

        FeasibleRegion {
            dimensions,
            max_streams,
            max_latency_ms: max_latency,
            feasible,
        }
    }

    /// Get names of all constraints.
    pub fn constraint_names(&self) -> Vec<String> {
        self.constraints.iter().map(|(n, _)| n.clone()).collect()
    }

    /// Iterator over constraints.
    pub fn iter(&self) -> impl Iterator<Item = &(String, Constraint)> {
        self.constraints.iter()
    }

    /// Get constraints affecting a specific stream.
    pub fn constraints_for_stream(&self, stream_id: StreamId) -> Vec<&Constraint> {
        self.constraints
            .iter()
            .filter_map(|(_, c)| match c {
                Constraint::MaskingClearance { stream_id: sid, .. } if *sid == stream_id => {
                    Some(c)
                }
                Constraint::SegregationRequired { stream1_id, stream2_id, .. }
                    if *stream1_id == stream_id || *stream2_id == stream_id =>
                {
                    Some(c)
                }
                _ => None,
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FeasibleRegion
// ─────────────────────────────────────────────────────────────────────────────

/// Representation of the parameter space satisfying all constraints.
#[derive(Debug, Clone)]
pub struct FeasibleRegion {
    pub dimensions: HashMap<String, IntervalDimension>,
    pub max_streams: usize,
    pub max_latency_ms: f64,
    pub feasible: bool,
}

/// An interval [min, max] for a single parameter dimension.
#[derive(Debug, Clone, Copy)]
pub struct IntervalDimension {
    pub min: f64,
    pub max: f64,
}

impl IntervalDimension {
    pub fn new(min: f64, max: f64) -> Self {
        IntervalDimension { min, max }
    }

    pub fn width(&self) -> f64 {
        (self.max - self.min).max(0.0)
    }

    pub fn midpoint(&self) -> f64 {
        (self.min + self.max) / 2.0
    }

    pub fn contains(&self, value: f64) -> bool {
        value >= self.min && value <= self.max
    }

    pub fn is_empty(&self) -> bool {
        self.min > self.max
    }
}

impl FeasibleRegion {
    /// Compute the volume (product of interval widths).
    pub fn volume(&self) -> f64 {
        if !self.feasible {
            return 0.0;
        }
        self.dimensions.values().map(|d| d.width().max(0.0)).product()
    }

    /// Sample a random point from the feasible region (uniform).
    pub fn sample(&self, rng: &mut impl FnMut() -> f64) -> Option<HashMap<String, f64>> {
        if !self.feasible || self.volume() <= 0.0 {
            return None;
        }

        let mut point = HashMap::new();
        for (name, dim) in &self.dimensions {
            let t = rng();
            point.insert(name.clone(), dim.min + t * dim.width());
        }
        Some(point)
    }

    /// Check if a point is inside the feasible region.
    pub fn contains(&self, point: &HashMap<String, f64>) -> bool {
        if !self.feasible {
            return false;
        }
        for (name, dim) in &self.dimensions {
            if let Some(&val) = point.get(name) {
                if !dim.contains(val) {
                    return false;
                }
            }
        }
        true
    }

    /// Shrink feasible region by intersecting with another.
    pub fn intersect(&self, other: &FeasibleRegion) -> FeasibleRegion {
        let mut dimensions = HashMap::new();
        let mut feasible = self.feasible && other.feasible;

        for (name, dim) in &self.dimensions {
            if let Some(other_dim) = other.dimensions.get(name) {
                let new_min = dim.min.max(other_dim.min);
                let new_max = dim.max.min(other_dim.max);
                if new_min > new_max {
                    feasible = false;
                }
                dimensions.insert(name.clone(), IntervalDimension::new(new_min, new_max));
            } else {
                dimensions.insert(name.clone(), *dim);
            }
        }

        // Add dimensions only in other
        for (name, dim) in &other.dimensions {
            if !dimensions.contains_key(name) {
                dimensions.insert(name.clone(), *dim);
            }
        }

        FeasibleRegion {
            dimensions,
            max_streams: self.max_streams.min(other.max_streams),
            max_latency_ms: self.max_latency_ms.min(other.max_latency_ms),
            feasible,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ConstraintReport
// ─────────────────────────────────────────────────────────────────────────────

/// Detailed satisfaction report for a constraint set.
#[derive(Debug, Clone)]
pub struct ConstraintReport {
    pub results: Vec<ConstraintSatisfaction>,
    pub all_satisfied: bool,
    pub total_violation: f64,
    pub min_margin: f64,
}

/// Per-constraint satisfaction status.
#[derive(Debug, Clone)]
pub struct ConstraintSatisfaction {
    pub name: String,
    pub evaluation: ConstraintEvaluation,
}

impl ConstraintReport {
    /// Get satisfied constraint count.
    pub fn satisfied_count(&self) -> usize {
        self.results.iter().filter(|r| r.evaluation.satisfied).count()
    }

    /// Get violated constraint count.
    pub fn violated_count(&self) -> usize {
        self.results.iter().filter(|r| !r.evaluation.satisfied).count()
    }

    /// Get satisfaction ratio [0, 1].
    pub fn satisfaction_ratio(&self) -> f64 {
        if self.results.is_empty() {
            return 1.0;
        }
        self.satisfied_count() as f64 / self.results.len() as f64
    }

    /// Get the worst violated constraint.
    pub fn worst_violation(&self) -> Option<&ConstraintSatisfaction> {
        self.results
            .iter()
            .filter(|r| !r.evaluation.satisfied)
            .max_by(|a, b| {
                a.evaluation
                    .violation_amount
                    .partial_cmp(&b.evaluation.violation_amount)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Summary string.
    pub fn summary(&self) -> String {
        format!(
            "{}/{} satisfied, total_violation={:.4}, min_margin={:.4}",
            self.satisfied_count(),
            self.results.len(),
            self.total_violation,
            self.min_margin,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience constructors
// ─────────────────────────────────────────────────────────────────────────────

/// Build a standard constraint set for sonification with reasonable defaults.
pub fn default_sonification_constraints() -> ConstraintSet {
    let mut cs = ConstraintSet::new();
    cs.add(Constraint::FrequencyRange {
        min_hz: 80.0,
        max_hz: 8000.0,
    });
    cs.add(Constraint::AmplitudeRange {
        min_db: 30.0,
        max_db: 90.0,
    });
    cs.add(Constraint::CognitiveLoadBudget {
        max_streams: 6,
        max_load: 20.0,
    });
    cs.add(Constraint::LatencyBound { max_ms: 50.0 });
    cs
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(streams: Vec<(u32, f64, f64)>) -> MappingConfig {
        let mut config = MappingConfig::new();
        for (id, freq, amp) in streams {
            config
                .stream_params
                .insert(StreamId(id), StreamMapping::new(StreamId(id), freq, amp));
        }
        config
    }

    #[test]
    fn test_frequency_range_satisfied() {
        let c = Constraint::FrequencyRange {
            min_hz: 100.0,
            max_hz: 5000.0,
        };
        let config = make_config(vec![(0, 440.0, 60.0)]);
        let eval = c.evaluate(&config);
        assert!(eval.satisfied);
    }

    #[test]
    fn test_frequency_range_violated_low() {
        let c = Constraint::FrequencyRange {
            min_hz: 100.0,
            max_hz: 5000.0,
        };
        let config = make_config(vec![(0, 50.0, 60.0)]);
        let eval = c.evaluate(&config);
        assert!(!eval.satisfied);
    }

    #[test]
    fn test_frequency_range_violated_high() {
        let c = Constraint::FrequencyRange {
            min_hz: 100.0,
            max_hz: 5000.0,
        };
        let config = make_config(vec![(0, 10000.0, 60.0)]);
        let eval = c.evaluate(&config);
        assert!(!eval.satisfied);
    }

    #[test]
    fn test_amplitude_range_satisfied() {
        let c = Constraint::AmplitudeRange {
            min_db: 20.0,
            max_db: 80.0,
        };
        let config = make_config(vec![(0, 440.0, 60.0)]);
        assert!(c.evaluate(&config).satisfied);
    }

    #[test]
    fn test_amplitude_range_violated() {
        let c = Constraint::AmplitudeRange {
            min_db: 20.0,
            max_db: 80.0,
        };
        let config = make_config(vec![(0, 440.0, 90.0)]);
        assert!(!c.evaluate(&config).satisfied);
    }

    #[test]
    fn test_cognitive_load_satisfied() {
        let c = Constraint::CognitiveLoadBudget {
            max_streams: 4,
            max_load: 20.0,
        };
        let config = make_config(vec![(0, 440.0, 60.0), (1, 880.0, 55.0)]);
        assert!(c.evaluate(&config).satisfied);
    }

    #[test]
    fn test_cognitive_load_too_many_streams() {
        let c = Constraint::CognitiveLoadBudget {
            max_streams: 2,
            max_load: 100.0,
        };
        let config = make_config(vec![
            (0, 440.0, 60.0),
            (1, 880.0, 55.0),
            (2, 1320.0, 50.0),
        ]);
        assert!(!c.evaluate(&config).satisfied);
    }

    #[test]
    fn test_latency_bound_satisfied() {
        let c = Constraint::LatencyBound { max_ms: 50.0 };
        let config = make_config(vec![(0, 440.0, 60.0)]);
        assert!(c.evaluate(&config).satisfied);
    }

    #[test]
    fn test_latency_bound_violated() {
        let c = Constraint::LatencyBound { max_ms: 3.0 };
        let config = make_config(vec![(0, 440.0, 60.0)]);
        // base_ms=5 + per_stream=2 => 7ms > 3ms
        assert!(!c.evaluate(&config).satisfied);
    }

    #[test]
    fn test_constraint_set_add_remove() {
        let mut cs = ConstraintSet::new();
        cs.add(Constraint::FrequencyRange { min_hz: 100.0, max_hz: 5000.0 });
        assert_eq!(cs.len(), 1);
        cs.remove("frequency_range");
        assert_eq!(cs.len(), 0);
    }

    #[test]
    fn test_constraint_set_check_all() {
        let mut cs = ConstraintSet::new();
        cs.add(Constraint::FrequencyRange { min_hz: 100.0, max_hz: 5000.0 });
        cs.add(Constraint::AmplitudeRange { min_db: 20.0, max_db: 80.0 });
        let config = make_config(vec![(0, 440.0, 60.0)]);
        let report = cs.check_all(&config);
        assert!(report.all_satisfied);
        assert_eq!(report.satisfied_count(), 2);
    }

    #[test]
    fn test_constraint_set_find_violated() {
        let mut cs = ConstraintSet::new();
        cs.add(Constraint::FrequencyRange { min_hz: 100.0, max_hz: 5000.0 });
        cs.add(Constraint::AmplitudeRange { min_db: 20.0, max_db: 50.0 }); // violated
        let config = make_config(vec![(0, 440.0, 60.0)]);
        let violated = cs.find_violated(&config);
        assert_eq!(violated.len(), 1);
    }

    #[test]
    fn test_feasible_region_volume() {
        let mut cs = ConstraintSet::new();
        cs.add(Constraint::FrequencyRange { min_hz: 100.0, max_hz: 1100.0 });
        cs.add(Constraint::AmplitudeRange { min_db: 40.0, max_db: 80.0 });
        let region = cs.compute_feasible_region();
        assert!(region.feasible);
        // 1000 * 40 = 40000
        assert!((region.volume() - 40000.0).abs() < 1.0);
    }

    #[test]
    fn test_feasible_region_infeasible() {
        let mut cs = ConstraintSet::new();
        cs.add(Constraint::FrequencyRange { min_hz: 5000.0, max_hz: 100.0 });
        let region = cs.compute_feasible_region();
        assert!(!region.feasible);
        assert_eq!(region.volume(), 0.0);
    }

    #[test]
    fn test_constraint_report_summary() {
        let mut cs = default_sonification_constraints();
        let config = make_config(vec![(0, 440.0, 60.0)]);
        let report = cs.check_all(&config);
        let summary = report.summary();
        assert!(summary.contains("satisfied"));
    }

    #[test]
    fn test_constraint_set_merge() {
        let mut cs1 = ConstraintSet::new();
        cs1.add(Constraint::FrequencyRange { min_hz: 100.0, max_hz: 5000.0 });
        let mut cs2 = ConstraintSet::new();
        cs2.add(Constraint::AmplitudeRange { min_db: 20.0, max_db: 80.0 });
        cs1.merge(&cs2);
        assert_eq!(cs1.len(), 2);
    }

    #[test]
    fn test_segregation_different_bark_bands() {
        let c = Constraint::SegregationRequired {
            stream1_id: StreamId(0),
            stream2_id: StreamId(1),
            predicates: vec![SegregationPredicate::DifferentBarkBands],
        };
        // 440 Hz and 4000 Hz are in different Bark bands
        let config = make_config(vec![(0, 440.0, 60.0), (1, 4000.0, 60.0)]);
        assert!(c.evaluate(&config).satisfied);
    }

    #[test]
    fn test_default_sonification_constraints() {
        let cs = default_sonification_constraints();
        assert_eq!(cs.len(), 4);
        let config = make_config(vec![(0, 440.0, 60.0)]);
        let report = cs.check_all(&config);
        assert!(report.all_satisfied);
    }

    #[test]
    fn test_feasible_region_contains() {
        let mut cs = ConstraintSet::new();
        cs.add(Constraint::FrequencyRange { min_hz: 100.0, max_hz: 5000.0 });
        cs.add(Constraint::AmplitudeRange { min_db: 20.0, max_db: 80.0 });
        let region = cs.compute_feasible_region();

        let mut point = HashMap::new();
        point.insert("frequency".to_string(), 440.0);
        point.insert("amplitude".to_string(), 60.0);
        assert!(region.contains(&point));

        let mut outside = HashMap::new();
        outside.insert("frequency".to_string(), 10.0);
        assert!(!region.contains(&outside));
    }

    #[test]
    fn test_feasible_region_intersect() {
        let mut r1_dims = HashMap::new();
        r1_dims.insert("frequency".to_string(), IntervalDimension::new(100.0, 5000.0));
        let r1 = FeasibleRegion {
            dimensions: r1_dims,
            max_streams: 6,
            max_latency_ms: 50.0,
            feasible: true,
        };

        let mut r2_dims = HashMap::new();
        r2_dims.insert("frequency".to_string(), IntervalDimension::new(200.0, 3000.0));
        let r2 = FeasibleRegion {
            dimensions: r2_dims,
            max_streams: 4,
            max_latency_ms: 30.0,
            feasible: true,
        };

        let intersection = r1.intersect(&r2);
        assert!(intersection.feasible);
        let freq = intersection.dimensions.get("frequency").unwrap();
        assert!((freq.min - 200.0).abs() < 0.01);
        assert!((freq.max - 3000.0).abs() < 0.01);
        assert_eq!(intersection.max_streams, 4);
    }
}

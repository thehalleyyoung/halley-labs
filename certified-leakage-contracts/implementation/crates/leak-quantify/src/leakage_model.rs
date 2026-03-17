//! Leakage models for side-channel analysis.
//!
//! Defines abstract leakage models that specify what information an attacker
//! can extract from a system. Includes cache timing, access pattern, power,
//! composed, and speculative execution models.

use std::fmt;

use serde::{Deserialize, Serialize};
use shared_types::{CacheLine, CacheSet, FunctionId, VirtualAddress};

use crate::channel::ChannelMatrix;
use crate::{QuantifyError, QuantifyResult};

// ---------------------------------------------------------------------------
// Leakage Model Trait
// ---------------------------------------------------------------------------

/// A model of what information leaks through a particular side channel.
///
/// Each leakage model defines a mapping from program states to observable
/// outputs, which induces a channel matrix for quantification.
pub trait LeakageModel: fmt::Debug + Send + Sync {
    /// A human-readable name for this model.
    fn name(&self) -> &str;

    /// The number of bits this model can leak per observation (upper bound).
    fn max_leakage_bits(&self) -> f64;

    /// Build the channel matrix induced by this model for `num_inputs`
    /// distinct secret-dependent inputs.
    fn channel_matrix(&self, num_inputs: usize) -> QuantifyResult<ChannelMatrix>;

    /// Whether this model considers speculative execution paths.
    fn is_speculative(&self) -> bool {
        false
    }

    /// A short description for audit logs.
    fn description(&self) -> String {
        format!("{} (≤ {:.2} bits)", self.name(), self.max_leakage_bits())
    }
}

// ---------------------------------------------------------------------------
// Cache Timing Model
// ---------------------------------------------------------------------------

/// Leakage model for cache-timing side channels (e.g., Flush+Reload, Prime+Probe).
///
/// The attacker observes whether each cache-line access is a hit or miss,
/// leaking information about secret-dependent memory accesses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheTimingModel {
    /// Number of cache sets the attacker can monitor.
    pub monitored_sets: usize,
    /// Cache associativity.
    pub associativity: usize,
    /// Cache line size in bytes.
    pub line_size: usize,
    /// Timing resolution in CPU cycles (lower = more powerful attacker).
    pub timing_resolution_cycles: u64,
    /// Whether the attacker can distinguish different miss latencies (L2 vs L3 vs DRAM).
    pub multi_level_timing: bool,
}

impl CacheTimingModel {
    /// Create a basic cache timing model.
    pub fn new(monitored_sets: usize, associativity: usize, line_size: usize) -> Self {
        Self {
            monitored_sets,
            associativity,
            line_size,
            timing_resolution_cycles: 1,
            multi_level_timing: false,
        }
    }

    /// Number of distinct observations per set.
    ///
    /// With binary hit/miss: 2. With multi-level timing: depends on levels.
    pub fn observations_per_set(&self) -> usize {
        if self.multi_level_timing {
            4 // hit, L2-miss, L3-miss, DRAM-miss
        } else {
            2 // hit, miss
        }
    }

    /// Upper bound on leakage per access (bits).
    pub fn bits_per_access(&self) -> f64 {
        (self.observations_per_set() as f64).log2()
    }

    /// Set the timing resolution.
    pub fn with_resolution(mut self, cycles: u64) -> Self {
        self.timing_resolution_cycles = cycles;
        self
    }

    /// Enable multi-level timing.
    pub fn with_multi_level(mut self) -> Self {
        self.multi_level_timing = true;
        self
    }
}

impl LeakageModel for CacheTimingModel {
    fn name(&self) -> &str {
        "CacheTimingModel"
    }

    fn max_leakage_bits(&self) -> f64 {
        self.monitored_sets as f64 * self.bits_per_access()
    }

    fn channel_matrix(&self, num_inputs: usize) -> QuantifyResult<ChannelMatrix> {
        let obs_per_set = self.observations_per_set();
        let num_outputs = obs_per_set.pow(self.monitored_sets.min(8) as u32);
        if num_inputs == 0 || num_outputs == 0 {
            return Err(QuantifyError::EmptySupport);
        }
        // Default: uniform mapping (worst-case for attacker)
        let p = 1.0 / num_outputs as f64;
        let rows = vec![vec![p; num_outputs]; num_inputs];
        ChannelMatrix::new(rows)
    }
}

impl fmt::Display for CacheTimingModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CacheTimingModel(sets={}, ways={}, ≤{:.2} bits)",
            self.monitored_sets,
            self.associativity,
            self.max_leakage_bits()
        )
    }
}

// ---------------------------------------------------------------------------
// Access Pattern Model
// ---------------------------------------------------------------------------

/// Leakage model for memory access pattern side channels.
///
/// The attacker observes *which* cache lines are accessed (not just hit/miss),
/// e.g., via shared memory monitoring or hardware performance counters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessPatternModel {
    /// Number of distinct cache lines the attacker can distinguish.
    pub distinguishable_lines: usize,
    /// Number of accesses per observation window.
    pub window_size: usize,
    /// Whether access ordering is observable.
    pub order_observable: bool,
}

impl AccessPatternModel {
    /// Create a new access pattern model.
    pub fn new(distinguishable_lines: usize, window_size: usize) -> Self {
        Self {
            distinguishable_lines,
            window_size,
            order_observable: false,
        }
    }

    /// Enable order observability.
    pub fn with_order(mut self) -> Self {
        self.order_observable = true;
        self
    }

    /// Upper bound on distinct observation sequences.
    pub fn max_observations(&self) -> u64 {
        if self.order_observable {
            // Sequences: lines^window
            (self.distinguishable_lines as u64).saturating_pow(self.window_size as u32)
        } else {
            // Subsets with repetition: C(lines + window - 1, window)
            // Upper bound: lines^window
            (self.distinguishable_lines as u64).saturating_pow(self.window_size as u32)
        }
    }
}

impl LeakageModel for AccessPatternModel {
    fn name(&self) -> &str {
        "AccessPatternModel"
    }

    fn max_leakage_bits(&self) -> f64 {
        let obs = self.max_observations();
        if obs > 0 { (obs as f64).log2() } else { 0.0 }
    }

    fn channel_matrix(&self, num_inputs: usize) -> QuantifyResult<ChannelMatrix> {
        let num_outputs = self.max_observations().min(1024) as usize;
        if num_inputs == 0 || num_outputs == 0 {
            return Err(QuantifyError::EmptySupport);
        }
        let p = 1.0 / num_outputs as f64;
        let rows = vec![vec![p; num_outputs]; num_inputs];
        ChannelMatrix::new(rows)
    }
}

impl fmt::Display for AccessPatternModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AccessPatternModel(lines={}, window={}, ≤{:.2} bits)",
            self.distinguishable_lines,
            self.window_size,
            self.max_leakage_bits()
        )
    }
}

// ---------------------------------------------------------------------------
// Power Model
// ---------------------------------------------------------------------------

/// Leakage model for power/electromagnetic side channels.
///
/// Models Hamming weight or Hamming distance leakage commonly exploited in
/// power analysis attacks (DPA/CPA).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerModel {
    /// Bit width of the data being processed.
    pub data_width: usize,
    /// The power model variant.
    pub variant: PowerModelVariant,
    /// Signal-to-noise ratio (affects practical exploitability).
    pub snr: f64,
}

/// Variant of power leakage model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PowerModelVariant {
    /// Hamming weight model: leaks HW(data).
    HammingWeight,
    /// Hamming distance model: leaks HD(old, new).
    HammingDistance,
    /// Full value leakage (worst case).
    Identity,
}

impl PowerModel {
    /// Create a Hamming weight model.
    pub fn hamming_weight(data_width: usize) -> Self {
        Self {
            data_width,
            variant: PowerModelVariant::HammingWeight,
            snr: 1.0,
        }
    }

    /// Create a Hamming distance model.
    pub fn hamming_distance(data_width: usize) -> Self {
        Self {
            data_width,
            variant: PowerModelVariant::HammingDistance,
            snr: 1.0,
        }
    }

    /// Set the signal-to-noise ratio.
    pub fn with_snr(mut self, snr: f64) -> Self {
        self.snr = snr;
        self
    }

    /// Number of distinct leakage values.
    pub fn num_leakage_values(&self) -> usize {
        match self.variant {
            PowerModelVariant::HammingWeight | PowerModelVariant::HammingDistance => {
                self.data_width + 1
            }
            PowerModelVariant::Identity => 1 << self.data_width.min(20),
        }
    }
}

impl LeakageModel for PowerModel {
    fn name(&self) -> &str {
        match self.variant {
            PowerModelVariant::HammingWeight => "PowerModel(HW)",
            PowerModelVariant::HammingDistance => "PowerModel(HD)",
            PowerModelVariant::Identity => "PowerModel(ID)",
        }
    }

    fn max_leakage_bits(&self) -> f64 {
        (self.num_leakage_values() as f64).log2()
    }

    fn channel_matrix(&self, num_inputs: usize) -> QuantifyResult<ChannelMatrix> {
        let num_outputs = self.num_leakage_values();
        if num_inputs == 0 || num_outputs == 0 {
            return Err(QuantifyError::EmptySupport);
        }
        let p = 1.0 / num_outputs as f64;
        let rows = vec![vec![p; num_outputs]; num_inputs];
        ChannelMatrix::new(rows)
    }
}

impl fmt::Display for PowerModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PowerModel({:?}, width={}, ≤{:.2} bits)",
            self.variant,
            self.data_width,
            self.max_leakage_bits()
        )
    }
}

// ---------------------------------------------------------------------------
// Composed Model
// ---------------------------------------------------------------------------

/// A leakage model composed from multiple sub-models.
///
/// The attacker observes the combined output of all sub-models. Leakage is
/// bounded by the sum of individual model leakages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComposedModel {
    /// Human-readable name for the composed model.
    pub name: String,
    /// Component models and their labels.
    pub components: Vec<ComposedComponent>,
}

/// A named component of a composed model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComposedComponent {
    /// Label for this component.
    pub label: String,
    /// Maximum leakage of this component in bits.
    pub max_bits: f64,
    /// Number of leakage values for this component.
    pub num_outputs: usize,
}

impl ComposedModel {
    /// Create an empty composed model.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            components: Vec::new(),
        }
    }

    /// Add a component model.
    pub fn add_component(&mut self, label: impl Into<String>, max_bits: f64, num_outputs: usize) {
        self.components.push(ComposedComponent {
            label: label.into(),
            max_bits,
            num_outputs,
        });
    }

    /// Add a component from a LeakageModel trait object.
    pub fn add_model(&mut self, model: &dyn LeakageModel) {
        self.components.push(ComposedComponent {
            label: model.name().to_string(),
            max_bits: model.max_leakage_bits(),
            num_outputs: 2_usize.pow(model.max_leakage_bits().ceil() as u32),
        });
    }

    /// Total number of components.
    pub fn num_components(&self) -> usize {
        self.components.len()
    }
}

impl LeakageModel for ComposedModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn max_leakage_bits(&self) -> f64 {
        self.components.iter().map(|c| c.max_bits).sum()
    }

    fn channel_matrix(&self, num_inputs: usize) -> QuantifyResult<ChannelMatrix> {
        let total_outputs: usize = self
            .components
            .iter()
            .map(|c| c.num_outputs)
            .fold(1, usize::saturating_mul)
            .min(4096);
        if num_inputs == 0 || total_outputs == 0 {
            return Err(QuantifyError::EmptySupport);
        }
        let p = 1.0 / total_outputs as f64;
        let rows = vec![vec![p; total_outputs]; num_inputs];
        ChannelMatrix::new(rows)
    }
}

impl fmt::Display for ComposedModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ComposedModel({}, {} components, ≤{:.2} bits)",
            self.name,
            self.num_components(),
            self.max_leakage_bits()
        )
    }
}

// ---------------------------------------------------------------------------
// Speculative Leakage Model
// ---------------------------------------------------------------------------

/// Leakage model accounting for speculative execution (Spectre-class attacks).
///
/// Extends a base model with additional leakage from transiently executed
/// instructions that access secret data before being squashed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeLeakageModel {
    /// Maximum speculation window (number of transient instructions).
    pub speculation_window: usize,
    /// Base (architectural) leakage in bits.
    pub architectural_leakage_bits: f64,
    /// Additional transient leakage in bits.
    pub transient_leakage_bits: f64,
    /// Whether the model accounts for store-to-load forwarding leaks.
    pub store_forwarding: bool,
    /// Whether the model accounts for branch prediction leaks.
    pub branch_prediction: bool,
}

impl SpeculativeLeakageModel {
    /// Create a new speculative leakage model.
    pub fn new(
        speculation_window: usize,
        architectural_bits: f64,
        transient_bits: f64,
    ) -> Self {
        Self {
            speculation_window,
            architectural_leakage_bits: architectural_bits,
            transient_leakage_bits: transient_bits,
            store_forwarding: false,
            branch_prediction: true,
        }
    }

    /// Enable store-forwarding leakage.
    pub fn with_store_forwarding(mut self) -> Self {
        self.store_forwarding = true;
        self
    }

    /// Disable branch prediction leakage.
    pub fn without_branch_prediction(mut self) -> Self {
        self.branch_prediction = false;
        self
    }

    /// Total leakage = architectural + transient.
    pub fn total_leakage_bits(&self) -> f64 {
        self.architectural_leakage_bits + self.transient_leakage_bits
    }
}

impl LeakageModel for SpeculativeLeakageModel {
    fn name(&self) -> &str {
        "SpeculativeLeakageModel"
    }

    fn max_leakage_bits(&self) -> f64 {
        self.total_leakage_bits()
    }

    fn channel_matrix(&self, num_inputs: usize) -> QuantifyResult<ChannelMatrix> {
        let total_bits = self.max_leakage_bits().ceil() as u32;
        let num_outputs = 2_usize.pow(total_bits.min(12));
        if num_inputs == 0 || num_outputs == 0 {
            return Err(QuantifyError::EmptySupport);
        }
        let p = 1.0 / num_outputs as f64;
        let rows = vec![vec![p; num_outputs]; num_inputs];
        ChannelMatrix::new(rows)
    }

    fn is_speculative(&self) -> bool {
        true
    }
}

impl fmt::Display for SpeculativeLeakageModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SpeculativeLeakageModel(window={}, arch={:.2}b, trans={:.2}b)",
            self.speculation_window,
            self.architectural_leakage_bits,
            self.transient_leakage_bits
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_timing_model() {
        let m = CacheTimingModel::new(4, 8, 64);
        assert!((m.bits_per_access() - 1.0).abs() < 1e-10);
        assert!((m.max_leakage_bits() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_access_pattern_model() {
        let m = AccessPatternModel::new(16, 1);
        assert!((m.max_leakage_bits() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_power_model_hw() {
        let m = PowerModel::hamming_weight(8);
        // 9 distinct HW values for 8-bit data → log₂(9) ≈ 3.17
        assert!(m.max_leakage_bits() > 3.0);
        assert!(m.max_leakage_bits() < 4.0);
    }

    #[test]
    fn test_composed_model() {
        let mut cm = ComposedModel::new("test");
        cm.add_component("timing", 4.0, 16);
        cm.add_component("power", 3.0, 8);
        assert!((cm.max_leakage_bits() - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_speculative_model() {
        let m = SpeculativeLeakageModel::new(128, 2.0, 6.0);
        assert!(m.is_speculative());
        assert!((m.total_leakage_bits() - 8.0).abs() < 1e-10);
    }
}

//! Parameter management with lock-free updates, smoothing, and interpolation.
//!
//! Provides [`ParameterManager`] for storing and queuing parameter changes,
//! [`ParameterInterpolator`] for sample-accurate ramping, and a lightweight
//! [`Parameter`] descriptor that carries range/default/smoothing metadata.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Parameter descriptor
// ---------------------------------------------------------------------------

/// Describes a single automatable parameter.
#[derive(Debug, Clone)]
pub struct Parameter {
    pub id: u64,
    pub name: String,
    pub value: f64,
    pub min: f64,
    pub max: f64,
    pub default: f64,
    /// Smoothing time in seconds (0 = immediate).
    pub smoothing_time: f64,
}

impl Parameter {
    pub fn new(id: u64, name: &str, default: f64, min: f64, max: f64) -> Self {
        Self {
            id,
            name: name.to_string(),
            value: default,
            min,
            max,
            default,
            smoothing_time: 0.005, // 5 ms default
        }
    }

    pub fn with_smoothing(mut self, seconds: f64) -> Self {
        self.smoothing_time = seconds;
        self
    }

    /// Clamp a value to the parameter's valid range.
    #[inline]
    pub fn clamp(&self, v: f64) -> f64 {
        v.clamp(self.min, self.max)
    }

    /// Normalise a value to 0..1 within the parameter range.
    #[inline]
    pub fn normalize(&self, v: f64) -> f64 {
        if (self.max - self.min).abs() < f64::EPSILON {
            return 0.0;
        }
        (v - self.min) / (self.max - self.min)
    }

    /// Denormalise a 0..1 value back to the parameter range.
    #[inline]
    pub fn denormalize(&self, n: f64) -> f64 {
        self.min + n * (self.max - self.min)
    }
}

// ---------------------------------------------------------------------------
// Parameter change
// ---------------------------------------------------------------------------

/// A single parameter-value change, optionally timestamped.
#[derive(Debug, Clone, Copy)]
pub struct ParameterChange {
    /// Timestamp in samples from the start of the current buffer (0 = immediate).
    pub sample_offset: u64,
    pub parameter_id: u64,
    pub new_value: f64,
}

impl ParameterChange {
    pub fn immediate(parameter_id: u64, value: f64) -> Self {
        Self { sample_offset: 0, parameter_id, new_value: value }
    }

    pub fn at_sample(sample_offset: u64, parameter_id: u64, value: f64) -> Self {
        Self { sample_offset, parameter_id, new_value: value }
    }
}

// ---------------------------------------------------------------------------
// Interpolation mode
// ---------------------------------------------------------------------------

/// How a parameter transitions from its current value to a new target.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolationMode {
    /// Jump to the new value immediately.
    Immediate,
    /// Linear ramp over `samples` samples.
    Linear { samples: usize },
    /// Exponential ramp (one-pole) over `samples` samples.
    Exponential { samples: usize },
}

// ---------------------------------------------------------------------------
// ParameterInterpolator
// ---------------------------------------------------------------------------

/// Produces sample-accurate ramps between parameter values.
#[derive(Debug, Clone)]
pub struct ParameterInterpolator {
    current: f64,
    target: f64,
    mode: InterpolationMode,
    remaining: usize,
    increment: f64,
    coefficient: f64,
}

impl ParameterInterpolator {
    pub fn new(initial: f64) -> Self {
        Self {
            current: initial,
            target: initial,
            mode: InterpolationMode::Immediate,
            remaining: 0,
            increment: 0.0,
            coefficient: 0.0,
        }
    }

    /// Start a new ramp toward `target`.
    pub fn set_target(&mut self, target: f64, mode: InterpolationMode) {
        self.target = target;
        self.mode = mode;
        match mode {
            InterpolationMode::Immediate => {
                self.current = target;
                self.remaining = 0;
            }
            InterpolationMode::Linear { samples } => {
                if samples == 0 {
                    self.current = target;
                    self.remaining = 0;
                } else {
                    self.remaining = samples;
                    self.increment = (target - self.current) / samples as f64;
                }
            }
            InterpolationMode::Exponential { samples } => {
                if samples == 0 {
                    self.current = target;
                    self.remaining = 0;
                } else {
                    self.remaining = samples;
                    // Time-constant so that we reach ~99.3 % after `samples`
                    self.coefficient = 1.0 - (-5.0 / samples as f64).exp();
                }
            }
        }
    }

    /// Advance by one sample and return the interpolated value.
    #[inline]
    pub fn next_sample(&mut self) -> f64 {
        if self.remaining == 0 {
            return self.current;
        }
        match self.mode {
            InterpolationMode::Immediate => { /* already at target */ }
            InterpolationMode::Linear { .. } => {
                self.current += self.increment;
            }
            InterpolationMode::Exponential { .. } => {
                self.current += self.coefficient * (self.target - self.current);
            }
        }
        self.remaining -= 1;
        if self.remaining == 0 {
            self.current = self.target;
        }
        self.current
    }

    /// Fill a slice with interpolated values.
    pub fn process_block(&mut self, out: &mut [f64]) {
        for v in out.iter_mut() {
            *v = self.next_sample();
        }
    }

    /// Whether the interpolator has reached its target.
    #[inline]
    pub fn is_settled(&self) -> bool {
        self.remaining == 0
    }

    #[inline]
    pub fn current(&self) -> f64 {
        self.current
    }

    #[inline]
    pub fn target(&self) -> f64 {
        self.target
    }
}

// ---------------------------------------------------------------------------
// ParameterSmoother (one-pole)
// ---------------------------------------------------------------------------

/// Lightweight one-pole smoother used internally by [`ParameterManager`].
#[derive(Debug, Clone)]
pub struct ParameterSmoother {
    current: f64,
    target: f64,
    coeff: f64,
}

impl ParameterSmoother {
    pub fn new(initial: f64, smoothing_time: f64, sample_rate: f64) -> Self {
        let coeff = if smoothing_time <= 0.0 || sample_rate <= 0.0 {
            1.0
        } else {
            1.0 - (-1.0 / (smoothing_time * sample_rate)).exp()
        };
        Self { current: initial, target: initial, coeff }
    }

    pub fn set_target(&mut self, target: f64) {
        self.target = target;
    }

    #[inline]
    pub fn next_sample(&mut self) -> f64 {
        self.current += self.coeff * (self.target - self.current);
        self.current
    }

    pub fn set_immediate(&mut self, value: f64) {
        self.current = value;
        self.target = value;
    }

    pub fn current(&self) -> f64 {
        self.current
    }

    pub fn is_settled(&self, epsilon: f64) -> bool {
        (self.current - self.target).abs() < epsilon
    }

    pub fn update_coefficient(&mut self, smoothing_time: f64, sample_rate: f64) {
        self.coeff = if smoothing_time <= 0.0 || sample_rate <= 0.0 {
            1.0
        } else {
            1.0 - (-1.0 / (smoothing_time * sample_rate)).exp()
        };
    }
}

// ---------------------------------------------------------------------------
// Lock-free parameter slot (single-writer / single-reader)
// ---------------------------------------------------------------------------

/// Atomically-updated f64 using `AtomicU64` bit-cast.  
/// Designed for a single writer (control thread) and a single reader (audio
/// thread) without locks.
#[derive(Debug)]
pub struct AtomicF64 {
    bits: AtomicU64,
}

impl AtomicF64 {
    pub fn new(value: f64) -> Self {
        Self { bits: AtomicU64::new(value.to_bits()) }
    }

    #[inline]
    pub fn load(&self) -> f64 {
        f64::from_bits(self.bits.load(Ordering::Acquire))
    }

    #[inline]
    pub fn store(&self, value: f64) {
        self.bits.store(value.to_bits(), Ordering::Release);
    }
}

// ---------------------------------------------------------------------------
// MidiCcMapping / OscMapping
// ---------------------------------------------------------------------------

/// Maps a MIDI CC number to a parameter.
#[derive(Debug, Clone)]
pub struct MidiCcMapping {
    pub cc_number: u8,
    pub parameter_id: u64,
    pub min_value: f64,
    pub max_value: f64,
}

impl MidiCcMapping {
    pub fn new(cc_number: u8, parameter_id: u64, min_value: f64, max_value: f64) -> Self {
        Self { cc_number, parameter_id, min_value, max_value }
    }

    /// Convert a 7-bit CC value (0..127) to the mapped parameter range.
    pub fn map_value(&self, cc_value: u8) -> f64 {
        let norm = cc_value as f64 / 127.0;
        self.min_value + norm * (self.max_value - self.min_value)
    }
}

/// Maps an OSC address to a parameter.
#[derive(Debug, Clone)]
pub struct OscMapping {
    pub address: String,
    pub parameter_id: u64,
    pub min_value: f64,
    pub max_value: f64,
}

impl OscMapping {
    pub fn new(address: &str, parameter_id: u64, min_value: f64, max_value: f64) -> Self {
        Self { address: address.to_string(), parameter_id, min_value, max_value }
    }

    pub fn map_value(&self, osc_value: f64) -> f64 {
        self.min_value + osc_value.clamp(0.0, 1.0) * (self.max_value - self.min_value)
    }
}

// ---------------------------------------------------------------------------
// ParameterManager
// ---------------------------------------------------------------------------

/// Central store for all automatable parameters.  
///
/// Designed for single-writer (control/UI thread) and single-reader (audio
/// thread) use. The writer pushes [`ParameterChange`]s; the reader drains
/// them at the beginning of each buffer period and applies smoothing.
#[derive(Debug)]
pub struct ParameterManager {
    parameters: HashMap<u64, Parameter>,
    smoothers: HashMap<u64, ParameterSmoother>,
    pending_changes: Vec<ParameterChange>,
    sample_rate: f64,
    midi_mappings: Vec<MidiCcMapping>,
    osc_mappings: Vec<OscMapping>,
}

impl ParameterManager {
    pub fn new(sample_rate: f64) -> Self {
        Self {
            parameters: HashMap::new(),
            smoothers: HashMap::new(),
            pending_changes: Vec::new(),
            sample_rate,
            midi_mappings: Vec::new(),
            osc_mappings: Vec::new(),
        }
    }

    /// Register a parameter. Creates a corresponding smoother.
    pub fn register(&mut self, param: Parameter) {
        let smoother = ParameterSmoother::new(
            param.value,
            param.smoothing_time,
            self.sample_rate,
        );
        self.smoothers.insert(param.id, smoother);
        self.parameters.insert(param.id, param);
    }

    /// Queue a change to be applied at the start of the next buffer.
    pub fn queue_change(&mut self, change: ParameterChange) {
        self.pending_changes.push(change);
    }

    /// Set a parameter immediately (useful during graph setup).
    pub fn set_immediate(&mut self, id: u64, value: f64) {
        if let Some(param) = self.parameters.get_mut(&id) {
            let clamped = param.clamp(value);
            param.value = clamped;
            if let Some(sm) = self.smoothers.get_mut(&id) {
                sm.set_immediate(clamped);
            }
        }
    }

    /// Apply all queued changes (called once per buffer by the audio thread).
    pub fn apply_pending(&mut self) {
        let changes: Vec<ParameterChange> = self.pending_changes.drain(..).collect();
        for ch in changes {
            if let Some(param) = self.parameters.get_mut(&ch.parameter_id) {
                let clamped = param.clamp(ch.new_value);
                param.value = clamped;
                if let Some(sm) = self.smoothers.get_mut(&ch.parameter_id) {
                    sm.set_target(clamped);
                }
            }
        }
    }

    /// Advance smoothers by one sample and return the smoothed value.
    #[inline]
    pub fn get_smoothed(&mut self, id: u64) -> f64 {
        self.smoothers.get_mut(&id).map(|s| s.next_sample()).unwrap_or(0.0)
    }

    /// Snapshot of all current (unsmoothed) parameter values.
    pub fn snapshot(&self) -> HashMap<u64, f64> {
        self.parameters.iter().map(|(&id, p)| (id, p.value)).collect()
    }

    /// Return the current smoothed value without advancing.
    pub fn current_value(&self, id: u64) -> Option<f64> {
        self.smoothers.get(&id).map(|s| s.current())
    }

    /// Get a registered parameter descriptor.
    pub fn get_parameter(&self, id: u64) -> Option<&Parameter> {
        self.parameters.get(&id)
    }

    /// Return all registered parameter ids.
    pub fn parameter_ids(&self) -> Vec<u64> {
        self.parameters.keys().copied().collect()
    }

    /// Process a MIDI CC message, queueing changes for mapped parameters.
    pub fn process_midi_cc(&mut self, cc_number: u8, cc_value: u8) {
        let mappings: Vec<MidiCcMapping> = self.midi_mappings.clone();
        for m in &mappings {
            if m.cc_number == cc_number {
                let val = m.map_value(cc_value);
                self.queue_change(ParameterChange::immediate(m.parameter_id, val));
            }
        }
    }

    /// Process an OSC message, queueing changes for mapped parameters.
    pub fn process_osc(&mut self, address: &str, value: f64) {
        let mappings: Vec<OscMapping> = self.osc_mappings.clone();
        for m in &mappings {
            if m.address == address {
                let val = m.map_value(value);
                self.queue_change(ParameterChange::immediate(m.parameter_id, val));
            }
        }
    }

    pub fn add_midi_mapping(&mut self, mapping: MidiCcMapping) {
        self.midi_mappings.push(mapping);
    }

    pub fn add_osc_mapping(&mut self, mapping: OscMapping) {
        self.osc_mappings.push(mapping);
    }

    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate;
        for (id, sm) in self.smoothers.iter_mut() {
            if let Some(p) = self.parameters.get(id) {
                sm.update_coefficient(p.smoothing_time, sample_rate);
            }
        }
    }

    pub fn reset_all(&mut self) {
        for (id, param) in self.parameters.iter_mut() {
            param.value = param.default;
            if let Some(sm) = self.smoothers.get_mut(id) {
                sm.set_immediate(param.default);
            }
        }
        self.pending_changes.clear();
    }

    pub fn pending_count(&self) -> usize {
        self.pending_changes.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_clamp() {
        let p = Parameter::new(1, "gain", 0.5, 0.0, 1.0);
        assert_eq!(p.clamp(-0.5), 0.0);
        assert_eq!(p.clamp(1.5), 1.0);
        assert_eq!(p.clamp(0.7), 0.7);
    }

    #[test]
    fn parameter_normalize_denormalize() {
        let p = Parameter::new(1, "freq", 440.0, 20.0, 20000.0);
        let norm = p.normalize(440.0);
        let back = p.denormalize(norm);
        assert!((back - 440.0).abs() < 1e-6);
    }

    #[test]
    fn interpolator_immediate() {
        let mut interp = ParameterInterpolator::new(0.0);
        interp.set_target(1.0, InterpolationMode::Immediate);
        assert!(interp.is_settled());
        assert!((interp.current() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn interpolator_linear_ramp() {
        let mut interp = ParameterInterpolator::new(0.0);
        interp.set_target(1.0, InterpolationMode::Linear { samples: 10 });
        let mut vals = Vec::new();
        for _ in 0..10 {
            vals.push(interp.next_sample());
        }
        assert!(interp.is_settled());
        assert!((vals.last().unwrap() - 1.0).abs() < 1e-12);
        // Values should be monotonically increasing
        for w in vals.windows(2) {
            assert!(w[1] >= w[0]);
        }
    }

    #[test]
    fn interpolator_exponential_ramp() {
        let mut interp = ParameterInterpolator::new(0.0);
        interp.set_target(1.0, InterpolationMode::Exponential { samples: 100 });
        for _ in 0..100 {
            interp.next_sample();
        }
        assert!(interp.is_settled());
        assert!((interp.current() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn interpolator_process_block() {
        let mut interp = ParameterInterpolator::new(0.0);
        interp.set_target(1.0, InterpolationMode::Linear { samples: 4 });
        let mut buf = [0.0f64; 4];
        interp.process_block(&mut buf);
        assert!((buf[3] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn smoother_convergence() {
        let mut sm = ParameterSmoother::new(0.0, 0.01, 44100.0);
        sm.set_target(1.0);
        for _ in 0..44100 {
            sm.next_sample();
        }
        assert!(sm.is_settled(1e-6));
    }

    #[test]
    fn smoother_immediate() {
        let mut sm = ParameterSmoother::new(0.0, 0.0, 44100.0);
        sm.set_target(1.0);
        let v = sm.next_sample();
        assert!((v - 1.0).abs() < 1e-12);
    }

    #[test]
    fn manager_register_and_snapshot() {
        let mut mgr = ParameterManager::new(44100.0);
        mgr.register(Parameter::new(1, "gain", 0.5, 0.0, 1.0));
        mgr.register(Parameter::new(2, "freq", 440.0, 20.0, 20000.0));
        let snap = mgr.snapshot();
        assert_eq!(snap.len(), 2);
        assert!((snap[&1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn manager_queue_and_apply() {
        let mut mgr = ParameterManager::new(44100.0);
        mgr.register(Parameter::new(1, "gain", 0.5, 0.0, 1.0));
        mgr.queue_change(ParameterChange::immediate(1, 0.8));
        assert_eq!(mgr.pending_count(), 1);
        mgr.apply_pending();
        assert_eq!(mgr.pending_count(), 0);
        let snap = mgr.snapshot();
        assert!((snap[&1] - 0.8).abs() < 1e-12);
    }

    #[test]
    fn manager_midi_cc_mapping() {
        let mut mgr = ParameterManager::new(44100.0);
        mgr.register(Parameter::new(1, "gain", 0.5, 0.0, 1.0));
        mgr.add_midi_mapping(MidiCcMapping::new(7, 1, 0.0, 1.0));
        mgr.process_midi_cc(7, 127);
        mgr.apply_pending();
        assert!((mgr.snapshot()[&1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn manager_osc_mapping() {
        let mut mgr = ParameterManager::new(44100.0);
        mgr.register(Parameter::new(1, "gain", 0.5, 0.0, 1.0));
        mgr.add_osc_mapping(OscMapping::new("/gain", 1, 0.0, 1.0));
        mgr.process_osc("/gain", 0.75);
        mgr.apply_pending();
        assert!((mgr.snapshot()[&1] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn manager_reset_all() {
        let mut mgr = ParameterManager::new(44100.0);
        mgr.register(Parameter::new(1, "gain", 0.5, 0.0, 1.0));
        mgr.set_immediate(1, 0.9);
        mgr.reset_all();
        assert!((mgr.snapshot()[&1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn atomic_f64_roundtrip() {
        let a = AtomicF64::new(3.14);
        assert!((a.load() - 3.14).abs() < 1e-12);
        a.store(-2.71);
        assert!((a.load() - (-2.71)).abs() < 1e-12);
    }

    #[test]
    fn manager_clamps_out_of_range() {
        let mut mgr = ParameterManager::new(44100.0);
        mgr.register(Parameter::new(1, "gain", 0.5, 0.0, 1.0));
        mgr.queue_change(ParameterChange::immediate(1, 5.0));
        mgr.apply_pending();
        assert!((mgr.snapshot()[&1] - 1.0).abs() < 1e-12);
    }
}

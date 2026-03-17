//! Audio graph executor: topological-order processing, inter-node buffer
//! management, and performance monitoring.
//!
//! The executor takes an abstract audio graph (nodes + edges), resolves
//! processing order, allocates shared buffers, and processes one buffer
//! period at a time.

use std::collections::HashMap;
use std::time::Instant;

use crate::parameter::{ParameterManager, Parameter};
use crate::{AudioBuf, RendererResult, RendererError};
use crate::oscillators::*;
use crate::filters::{BiquadFilter, FilterMode};
use crate::envelope::AdsrEnvelope;
use crate::effects::{Delay, Compressor, Limiter};

// ---------------------------------------------------------------------------
// ExecutionContext
// ---------------------------------------------------------------------------

/// Immutable snapshot of the rendering state for the current buffer period.
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    pub sample_rate: u32,
    pub buffer_size: usize,
    pub channels: usize,
    /// Total elapsed samples since the start of rendering.
    pub elapsed_samples: u64,
    /// Elapsed wall-clock time in seconds.
    pub elapsed_time: f64,
    /// Parameter values snapshotted at the start of this buffer.
    pub parameters: HashMap<u64, f64>,
}

impl ExecutionContext {
    pub fn new(sample_rate: u32, buffer_size: usize, channels: usize) -> Self {
        Self {
            sample_rate,
            buffer_size,
            channels,
            elapsed_samples: 0,
            elapsed_time: 0.0,
            parameters: HashMap::new(),
        }
    }

    /// Advance by one buffer period.
    pub fn advance(&mut self) {
        self.elapsed_samples += self.buffer_size as u64;
        self.elapsed_time = self.elapsed_samples as f64 / self.sample_rate as f64;
    }
}

// ---------------------------------------------------------------------------
// BufferPool
// ---------------------------------------------------------------------------

/// Pre-allocated pool of audio buffers for zero-allocation steady-state
/// operation.
#[derive(Debug)]
pub struct BufferPool {
    free: Vec<AudioBuf>,
    active_count: usize,
    buffer_size: usize,
    channels: usize,
    sample_rate: u32,
}

impl BufferPool {
    /// Pre-allocate `count` buffers.
    pub fn new(count: usize, buffer_size: usize, channels: usize, sample_rate: u32) -> Self {
        let free = (0..count)
            .map(|_| AudioBuf::new(buffer_size, channels, sample_rate))
            .collect();
        Self { free, active_count: 0, buffer_size, channels, sample_rate }
    }

    /// Acquire a zeroed buffer from the pool.
    pub fn acquire(&mut self) -> AudioBuf {
        let mut buf = self.free.pop().unwrap_or_else(|| {
            AudioBuf::new(self.buffer_size, self.channels, self.sample_rate)
        });
        buf.zero();
        self.active_count += 1;
        buf
    }

    /// Return a buffer to the pool.
    pub fn release(&mut self, buf: AudioBuf) {
        if self.active_count > 0 {
            self.active_count -= 1;
        }
        self.free.push(buf);
    }

    /// Number of buffers currently checked out.
    pub fn active_count(&self) -> usize {
        self.active_count
    }

    /// Number of free buffers available.
    pub fn free_count(&self) -> usize {
        self.free.len()
    }

    /// Total buffers (free + active).
    pub fn total_count(&self) -> usize {
        self.free.len() + self.active_count
    }

    /// Ensure at least `n` buffers are available (pre-warm).
    pub fn ensure_capacity(&mut self, n: usize) {
        while self.free.len() < n {
            self.free.push(AudioBuf::new(self.buffer_size, self.channels, self.sample_rate));
        }
    }
}

// ---------------------------------------------------------------------------
// NodeProcessor trait
// ---------------------------------------------------------------------------

/// Trait implemented by every audio-graph node processor.
pub trait NodeProcessor: std::fmt::Debug {
    /// Process one buffer period.
    ///
    /// * `inputs` – read-only references to upstream buffers.
    /// * `output` – the buffer this node writes into.
    /// * `context` – current rendering context.
    fn process(
        &mut self,
        inputs: &[&AudioBuf],
        output: &mut AudioBuf,
        context: &ExecutionContext,
    );

    /// Reset all internal state (e.g. on graph re-build).
    fn reset(&mut self);

    /// Set a named parameter. Return `true` if recognised.
    fn set_parameter(&mut self, name: &str, value: f64) -> bool;
}

// ---------------------------------------------------------------------------
// Built-in NodeProcessor implementations
// ---------------------------------------------------------------------------

// -- SineOscillatorNode -------------------------------------------------------

#[derive(Debug)]
pub struct SineOscillatorNode {
    osc: SineOscillator,
    amplitude: f64,
}

impl SineOscillatorNode {
    pub fn new(freq: f64, sample_rate: f64) -> Self {
        Self { osc: SineOscillator::new(freq, sample_rate), amplitude: 1.0 }
    }
}

impl NodeProcessor for SineOscillatorNode {
    fn process(&mut self, _inputs: &[&AudioBuf], output: &mut AudioBuf, _ctx: &ExecutionContext) {
        self.osc.process(output);
        if (self.amplitude - 1.0).abs() > 1e-12 {
            output.apply_gain(self.amplitude as f32);
        }
    }
    fn reset(&mut self) { self.osc.reset(); }
    fn set_parameter(&mut self, name: &str, value: f64) -> bool {
        match name {
            "frequency" => { self.osc.set_frequency(value); true }
            "amplitude" => { self.amplitude = value; true }
            "phase" => { self.osc.set_phase_offset(value); true }
            _ => false,
        }
    }
}

// -- SawOscillatorNode --------------------------------------------------------

#[derive(Debug)]
pub struct SawOscillatorNode {
    osc: SawOscillator,
    amplitude: f64,
}

impl SawOscillatorNode {
    pub fn new(freq: f64, sample_rate: f64) -> Self {
        Self { osc: SawOscillator::new(freq, sample_rate), amplitude: 1.0 }
    }
}

impl NodeProcessor for SawOscillatorNode {
    fn process(&mut self, _inputs: &[&AudioBuf], output: &mut AudioBuf, _ctx: &ExecutionContext) {
        self.osc.process(output);
        if (self.amplitude - 1.0).abs() > 1e-12 { output.apply_gain(self.amplitude as f32); }
    }
    fn reset(&mut self) { self.osc.reset(); }
    fn set_parameter(&mut self, name: &str, value: f64) -> bool {
        match name {
            "frequency" => { self.osc.set_frequency(value); true }
            "amplitude" => { self.amplitude = value; true }
            _ => false,
        }
    }
}

// -- SquareOscillatorNode -----------------------------------------------------

#[derive(Debug)]
pub struct SquareOscillatorNode {
    osc: SquareOscillator,
    amplitude: f64,
}

impl SquareOscillatorNode {
    pub fn new(freq: f64, sample_rate: f64) -> Self {
        Self { osc: SquareOscillator::new(freq, sample_rate), amplitude: 1.0 }
    }
}

impl NodeProcessor for SquareOscillatorNode {
    fn process(&mut self, _inputs: &[&AudioBuf], output: &mut AudioBuf, _ctx: &ExecutionContext) {
        self.osc.process(output);
        if (self.amplitude - 1.0).abs() > 1e-12 { output.apply_gain(self.amplitude as f32); }
    }
    fn reset(&mut self) { self.osc.reset(); }
    fn set_parameter(&mut self, name: &str, value: f64) -> bool {
        match name {
            "frequency" => { self.osc.set_frequency(value); true }
            "amplitude" => { self.amplitude = value; true }
            "duty_cycle" => { self.osc.set_duty_cycle(value); true }
            _ => false,
        }
    }
}

// -- TriangleOscillatorNode ---------------------------------------------------

#[derive(Debug)]
pub struct TriangleOscillatorNode {
    osc: TriangleOscillator,
    amplitude: f64,
}

impl TriangleOscillatorNode {
    pub fn new(freq: f64, sample_rate: f64) -> Self {
        Self { osc: TriangleOscillator::new(freq, sample_rate), amplitude: 1.0 }
    }
}

impl NodeProcessor for TriangleOscillatorNode {
    fn process(&mut self, _inputs: &[&AudioBuf], output: &mut AudioBuf, _ctx: &ExecutionContext) {
        self.osc.process(output);
        if (self.amplitude - 1.0).abs() > 1e-12 { output.apply_gain(self.amplitude as f32); }
    }
    fn reset(&mut self) { self.osc.reset(); }
    fn set_parameter(&mut self, name: &str, value: f64) -> bool {
        match name {
            "frequency" => { self.osc.set_frequency(value); true }
            "amplitude" => { self.amplitude = value; true }
            _ => false,
        }
    }
}

// -- PulseOscillatorNode ------------------------------------------------------

#[derive(Debug)]
pub struct PulseOscillatorNode {
    osc: PulseOscillator,
    amplitude: f64,
}

impl PulseOscillatorNode {
    pub fn new(freq: f64, sample_rate: f64, duty: f64) -> Self {
        Self { osc: PulseOscillator::new(freq, sample_rate, duty), amplitude: 1.0 }
    }
}

impl NodeProcessor for PulseOscillatorNode {
    fn process(&mut self, _inputs: &[&AudioBuf], output: &mut AudioBuf, _ctx: &ExecutionContext) {
        self.osc.process(output);
        if (self.amplitude - 1.0).abs() > 1e-12 { output.apply_gain(self.amplitude as f32); }
    }
    fn reset(&mut self) { self.osc.reset(); }
    fn set_parameter(&mut self, name: &str, value: f64) -> bool {
        match name {
            "frequency" => { self.osc.set_frequency(value); true }
            "duty_cycle" => { self.osc.set_duty_cycle(value); true }
            "amplitude" => { self.amplitude = value; true }
            _ => false,
        }
    }
}

// -- BiquadFilterNode ---------------------------------------------------------

#[derive(Debug)]
pub struct BiquadFilterNode {
    filter: BiquadFilter,
}

impl BiquadFilterNode {
    pub fn new(mode: FilterMode, freq: f64, q: f64, sample_rate: f64) -> Self {
        Self { filter: BiquadFilter::new(mode, freq, q, sample_rate) }
    }
}

impl NodeProcessor for BiquadFilterNode {
    fn process(&mut self, inputs: &[&AudioBuf], output: &mut AudioBuf, _ctx: &ExecutionContext) {
        if let Some(&input) = inputs.first() {
            output.copy_from_buf(input);
        }
        self.filter.process(output);
    }
    fn reset(&mut self) { self.filter.reset(); }
    fn set_parameter(&mut self, name: &str, value: f64) -> bool {
        match name {
            "frequency" => { self.filter.set_parameters(value, 0.7071, 0.0, 64); true }
            "q" => { self.filter.set_parameters(1000.0, value, 0.0, 64); true }
            _ => false,
        }
    }
}

// -- EnvelopeGeneratorNode ----------------------------------------------------

#[derive(Debug)]
pub struct EnvelopeGeneratorNode {
    env: AdsrEnvelope,
}

impl EnvelopeGeneratorNode {
    pub fn new(a: f64, d: f64, s: f64, r: f64, sample_rate: f64) -> Self {
        Self { env: AdsrEnvelope::new(a, d, s, r, sample_rate) }
    }

    pub fn trigger(&mut self) { self.env.trigger(); }
    pub fn release(&mut self) { self.env.release(); }
}

impl NodeProcessor for EnvelopeGeneratorNode {
    fn process(&mut self, inputs: &[&AudioBuf], output: &mut AudioBuf, _ctx: &ExecutionContext) {
        if let Some(&input) = inputs.first() {
            output.copy_from_buf(input);
        }
        self.env.apply_to(output);
    }
    fn reset(&mut self) { self.env.reset(); }
    fn set_parameter(&mut self, name: &str, value: f64) -> bool {
        match name {
            "attack" => { self.env.attack_time = value; true }
            "decay" => { self.env.decay_time = value; true }
            "sustain" => { self.env.sustain_level = value.clamp(0.0, 1.0); true }
            "release" => { self.env.release_time = value; true }
            "trigger" => { if value > 0.5 { self.env.trigger(); } else { self.env.release(); } true }
            _ => false,
        }
    }
}

// -- SummingMixerNode ---------------------------------------------------------

#[derive(Debug)]
pub struct SummingMixerNode {
    gains: Vec<f64>,
}

impl SummingMixerNode {
    pub fn new(num_inputs: usize) -> Self {
        Self { gains: vec![1.0; num_inputs] }
    }
}

impl NodeProcessor for SummingMixerNode {
    fn process(&mut self, inputs: &[&AudioBuf], output: &mut AudioBuf, _ctx: &ExecutionContext) {
        output.zero();
        for (i, &input) in inputs.iter().enumerate() {
            let gain = self.gains.get(i).copied().unwrap_or(1.0) as f32;
            let n = output.data.len().min(input.data.len());
            for j in 0..n {
                output.data[j] += input.data[j] * gain;
            }
        }
    }
    fn reset(&mut self) {}
    fn set_parameter(&mut self, _name: &str, _value: f64) -> bool { false }
}

// -- GainNode -----------------------------------------------------------------

#[derive(Debug)]
pub struct GainNode {
    pub gain: f64,
}

impl GainNode {
    pub fn new(gain: f64) -> Self { Self { gain } }
}

impl NodeProcessor for GainNode {
    fn process(&mut self, inputs: &[&AudioBuf], output: &mut AudioBuf, _ctx: &ExecutionContext) {
        if let Some(&input) = inputs.first() {
            output.copy_from_buf(input);
        }
        output.apply_gain(self.gain as f32);
    }
    fn reset(&mut self) {}
    fn set_parameter(&mut self, name: &str, value: f64) -> bool {
        match name {
            "gain" => { self.gain = value; true }
            _ => false,
        }
    }
}

// -- PanNode ------------------------------------------------------------------

#[derive(Debug)]
pub struct PanNode {
    pub pan: f64, // -1..1
}

impl PanNode {
    pub fn new(pan: f64) -> Self { Self { pan: pan.clamp(-1.0, 1.0) } }
}

impl NodeProcessor for PanNode {
    fn process(&mut self, inputs: &[&AudioBuf], output: &mut AudioBuf, _ctx: &ExecutionContext) {
        let angle = (self.pan + 1.0) * 0.25 * std::f64::consts::PI;
        let lg = angle.cos() as f32;
        let rg = angle.sin() as f32;

        output.zero();
        if let Some(&input) = inputs.first() {
            let frames = output.frames().min(input.frames());
            for f in 0..frames {
                let s = input.get(f, 0);
                output.set(f, 0, s * lg);
                if output.channels > 1 {
                    output.set(f, 1, s * rg);
                }
            }
        }
    }
    fn reset(&mut self) {}
    fn set_parameter(&mut self, name: &str, value: f64) -> bool {
        match name {
            "pan" => { self.pan = value.clamp(-1.0, 1.0); true }
            _ => false,
        }
    }
}

// -- DelayLineNode ------------------------------------------------------------

#[derive(Debug)]
pub struct DelayLineNode {
    delay: Delay,
}

impl DelayLineNode {
    pub fn new(max_samples: usize, delay_samples: f64) -> Self {
        Self { delay: Delay::new(max_samples, delay_samples) }
    }
}

impl NodeProcessor for DelayLineNode {
    fn process(&mut self, inputs: &[&AudioBuf], output: &mut AudioBuf, _ctx: &ExecutionContext) {
        if let Some(&input) = inputs.first() {
            output.copy_from_buf(input);
        }
        self.delay.process(output);
    }
    fn reset(&mut self) { self.delay.reset(); }
    fn set_parameter(&mut self, name: &str, value: f64) -> bool {
        match name {
            "delay" => { self.delay.set_delay(value); true }
            "feedback" => { self.delay.set_feedback(value); true }
            "mix" => { self.delay.mix = value; true }
            _ => false,
        }
    }
}

// -- ModulatorNode ------------------------------------------------------------

#[derive(Debug)]
pub struct ModulatorNode {
    /// Modulation depth.
    pub depth: f64,
}

impl ModulatorNode {
    pub fn new(depth: f64) -> Self { Self { depth } }
}

impl NodeProcessor for ModulatorNode {
    fn process(&mut self, inputs: &[&AudioBuf], output: &mut AudioBuf, _ctx: &ExecutionContext) {
        output.zero();
        // Ring-modulate first two inputs
        if inputs.len() >= 2 {
            let carrier = inputs[0];
            let modulator = inputs[1];
            let frames = output.frames().min(carrier.frames()).min(modulator.frames());
            for f in 0..frames {
                let c = carrier.get(f, 0) as f64;
                let m = modulator.get(f, 0) as f64;
                let blended = c * (1.0 - self.depth) + c * m * self.depth;
                output.set(f, 0, blended as f32);
            }
        } else if let Some(&input) = inputs.first() {
            output.copy_from_buf(input);
        }
    }
    fn reset(&mut self) {}
    fn set_parameter(&mut self, name: &str, value: f64) -> bool {
        match name {
            "depth" => { self.depth = value.clamp(0.0, 1.0); true }
            _ => false,
        }
    }
}

// -- CompressorNode -----------------------------------------------------------

#[derive(Debug)]
pub struct CompressorNode {
    comp: Compressor,
}

impl CompressorNode {
    pub fn new(threshold_db: f64, ratio: f64, sample_rate: f64) -> Self {
        Self { comp: Compressor::new(threshold_db, ratio, sample_rate) }
    }
}

impl NodeProcessor for CompressorNode {
    fn process(&mut self, inputs: &[&AudioBuf], output: &mut AudioBuf, _ctx: &ExecutionContext) {
        if let Some(&input) = inputs.first() {
            output.copy_from_buf(input);
        }
        self.comp.process(output);
    }
    fn reset(&mut self) { self.comp.reset(); }
    fn set_parameter(&mut self, name: &str, value: f64) -> bool {
        match name {
            "threshold" => { self.comp.threshold_db = value; true }
            "ratio" => { self.comp.ratio = value; true }
            "attack" => { self.comp.set_attack(value); true }
            "release" => { self.comp.set_release(value); true }
            _ => false,
        }
    }
}

// -- LimiterNode --------------------------------------------------------------

#[derive(Debug)]
pub struct LimiterNode {
    limiter: Limiter,
}

impl LimiterNode {
    pub fn new(ceiling_db: f64, lookahead_ms: f64, sample_rate: f64) -> Self {
        Self { limiter: Limiter::new(ceiling_db, lookahead_ms, sample_rate) }
    }
}

impl NodeProcessor for LimiterNode {
    fn process(&mut self, inputs: &[&AudioBuf], output: &mut AudioBuf, _ctx: &ExecutionContext) {
        if let Some(&input) = inputs.first() {
            output.copy_from_buf(input);
        }
        self.limiter.process(output);
    }
    fn reset(&mut self) { self.limiter.reset(); }
    fn set_parameter(&mut self, name: &str, value: f64) -> bool {
        match name {
            "ceiling" => { self.limiter.set_ceiling_db(value); true }
            _ => false,
        }
    }
}

// -- NoiseGeneratorNode -------------------------------------------------------

#[derive(Debug)]
pub struct NoiseGeneratorNode {
    noise: NoiseOscillator,
    amplitude: f64,
}

impl NoiseGeneratorNode {
    pub fn new(color: NoiseColor) -> Self {
        Self { noise: NoiseOscillator::new(color), amplitude: 1.0 }
    }
}

impl NodeProcessor for NoiseGeneratorNode {
    fn process(&mut self, _inputs: &[&AudioBuf], output: &mut AudioBuf, _ctx: &ExecutionContext) {
        self.noise.process(output);
        if (self.amplitude - 1.0).abs() > 1e-12 { output.apply_gain(self.amplitude as f32); }
    }
    fn reset(&mut self) { self.noise.reset(); }
    fn set_parameter(&mut self, name: &str, value: f64) -> bool {
        match name {
            "amplitude" => { self.amplitude = value; true }
            "color" => {
                let color = match value as u32 {
                    0 => NoiseColor::White,
                    1 => NoiseColor::Pink,
                    _ => NoiseColor::Brown,
                };
                self.noise.set_color(color);
                true
            }
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// GraphNode (executor's view of a node)
// ---------------------------------------------------------------------------

/// A node in the executor graph, coupling a processor with connectivity info.
#[derive(Debug)]
struct GraphNode {
    id: u64,
    processor: Box<dyn NodeProcessor>,
    /// Indices (in `AudioGraphExecutor::nodes`) of input nodes.
    input_indices: Vec<usize>,
}

// ---------------------------------------------------------------------------
// PerformanceMetrics
// ---------------------------------------------------------------------------

/// Timing information for one buffer period.
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// Wall-clock time for the whole buffer in microseconds.
    pub total_us: f64,
    /// Per-node times (node_id → µs).
    pub node_times: HashMap<u64, f64>,
    /// WCET budget for this buffer in microseconds.
    pub budget_us: f64,
    /// Whether the budget was exceeded.
    pub budget_exceeded: bool,
}

// ---------------------------------------------------------------------------
// AudioGraphExecutor
// ---------------------------------------------------------------------------

/// Executes an audio graph for one buffer period at a time.
///
/// Nodes are processed in topological order; each node reads from upstream
/// buffers and writes to its own output buffer.
#[derive(Debug)]
pub struct AudioGraphExecutor {
    nodes: Vec<GraphNode>,
    /// Topological order as indices into `nodes`.
    topo_order: Vec<usize>,
    context: ExecutionContext,
    pool: BufferPool,
    param_manager: ParameterManager,
    /// WCET budget in µs; 0 = disabled.
    wcet_budget_us: f64,
    last_metrics: PerformanceMetrics,
}

impl AudioGraphExecutor {
    pub fn new(sample_rate: u32, buffer_size: usize, channels: usize) -> Self {
        Self {
            nodes: Vec::new(),
            topo_order: Vec::new(),
            context: ExecutionContext::new(sample_rate, buffer_size, channels),
            pool: BufferPool::new(32, buffer_size, channels, sample_rate),
            param_manager: ParameterManager::new(sample_rate as f64),
            wcet_budget_us: 0.0,
            last_metrics: PerformanceMetrics::default(),
        }
    }

    /// Set WCET budget in microseconds (0 = disable monitoring).
    pub fn set_wcet_budget(&mut self, budget_us: f64) {
        self.wcet_budget_us = budget_us;
    }

    /// Add a node and return its index.
    pub fn add_node(&mut self, id: u64, processor: Box<dyn NodeProcessor>) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(GraphNode { id, processor, input_indices: Vec::new() });
        idx
    }

    /// Declare that `to_index` reads from `from_index`.
    pub fn connect(&mut self, from_index: usize, to_index: usize) {
        if to_index < self.nodes.len() {
            self.nodes[to_index].input_indices.push(from_index);
        }
    }

    /// Register a parameter.
    pub fn register_parameter(&mut self, param: Parameter) {
        self.param_manager.register(param);
    }

    /// Queue a parameter change for the next buffer.
    pub fn queue_parameter_change(&mut self, param_id: u64, value: f64) {
        self.param_manager.queue_change(
            crate::parameter::ParameterChange::immediate(param_id, value),
        );
    }

    /// Build the topological order. Call after all nodes and edges are added.
    pub fn build_schedule(&mut self) -> RendererResult<()> {
        let n = self.nodes.len();
        let mut in_degree = vec![0usize; n];
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

        for (idx, node) in self.nodes.iter().enumerate() {
            in_degree[idx] = node.input_indices.len();
            for &inp in &node.input_indices {
                adj[inp].push(idx);
            }
        }

        let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut order = Vec::with_capacity(n);
        while let Some(u) = queue.pop() {
            order.push(u);
            for &v in &adj[u] {
                in_degree[v] -= 1;
                if in_degree[v] == 0 {
                    queue.push(v);
                }
            }
        }

        if order.len() != n {
            return Err(RendererError::InvalidGraph("cycle detected".into()));
        }
        self.topo_order = order;
        Ok(())
    }

    /// Execute one buffer period. Returns the output of every node.
    pub fn process(&mut self) -> RendererResult<Vec<AudioBuf>> {
        // Apply queued parameter changes
        self.param_manager.apply_pending();
        self.context.parameters = self.param_manager.snapshot();

        let start = Instant::now();
        let mut node_outputs: Vec<Option<AudioBuf>> = (0..self.nodes.len())
            .map(|_| None)
            .collect();
        let mut metrics = PerformanceMetrics {
            budget_us: self.wcet_budget_us,
            ..Default::default()
        };

        for &idx in &self.topo_order {
            let node_start = Instant::now();

            // Collect input buffers
            let input_indices = self.nodes[idx].input_indices.clone();
            let input_bufs: Vec<&AudioBuf> = input_indices
                .iter()
                .filter_map(|&i| node_outputs[i].as_ref())
                .collect();

            let mut output = self.pool.acquire();
            self.nodes[idx].processor.process(&input_bufs, &mut output, &self.context);

            let node_elapsed = node_start.elapsed().as_secs_f64() * 1e6;
            metrics.node_times.insert(self.nodes[idx].id, node_elapsed);

            node_outputs[idx] = Some(output);
        }

        let total_elapsed = start.elapsed().as_secs_f64() * 1e6;
        metrics.total_us = total_elapsed;
        metrics.budget_exceeded = self.wcet_budget_us > 0.0 && total_elapsed > self.wcet_budget_us;
        self.last_metrics = metrics;

        self.context.advance();

        let result: Vec<AudioBuf> = node_outputs.into_iter().map(|o| o.unwrap_or_else(|| {
            AudioBuf::new(self.context.buffer_size, self.context.channels, self.context.sample_rate)
        })).collect();

        Ok(result)
    }

    /// Get the output of a specific node after `process()` has run.
    /// (Convenience: just use the return value of `process()` directly.)
    pub fn context(&self) -> &ExecutionContext {
        &self.context
    }

    pub fn last_metrics(&self) -> &PerformanceMetrics {
        &self.last_metrics
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Reset all node processors.
    pub fn reset(&mut self) {
        for node in &mut self.nodes {
            node.processor.reset();
        }
        self.context.elapsed_samples = 0;
        self.context.elapsed_time = 0.0;
        self.param_manager.reset_all();
    }

    pub fn param_manager(&self) -> &ParameterManager {
        &self.param_manager
    }

    pub fn param_manager_mut(&mut self) -> &mut ParameterManager {
        &mut self.param_manager
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sr() -> u32 { 44100 }
    fn bs() -> usize { 256 }

    // -- ExecutionContext ------------------------------------------------------

    #[test]
    fn context_advance() {
        let mut ctx = ExecutionContext::new(44100, 256, 2);
        ctx.advance();
        assert_eq!(ctx.elapsed_samples, 256);
        assert!((ctx.elapsed_time - 256.0 / 44100.0).abs() < 1e-9);
    }

    // -- BufferPool -----------------------------------------------------------

    #[test]
    fn pool_acquire_release() {
        let mut pool = BufferPool::new(4, 128, 1, 44100);
        assert_eq!(pool.free_count(), 4);
        let b1 = pool.acquire();
        assert_eq!(pool.active_count(), 1);
        assert_eq!(pool.free_count(), 3);
        pool.release(b1);
        assert_eq!(pool.active_count(), 0);
        assert_eq!(pool.free_count(), 4);
    }

    #[test]
    fn pool_grows_on_demand() {
        let mut pool = BufferPool::new(1, 64, 1, 44100);
        let _b1 = pool.acquire();
        let _b2 = pool.acquire(); // grows beyond initial capacity
        assert_eq!(pool.active_count(), 2);
    }

    #[test]
    fn pool_ensure_capacity() {
        let mut pool = BufferPool::new(0, 64, 1, 44100);
        pool.ensure_capacity(10);
        assert!(pool.free_count() >= 10);
    }

    // -- NodeProcessor implementations ----------------------------------------

    #[test]
    fn sine_node_produces_output() {
        let mut node = SineOscillatorNode::new(440.0, sr() as f64);
        let mut out = AudioBuf::new(bs(), 1, sr());
        let ctx = ExecutionContext::new(sr(), bs(), 1);
        node.process(&[], &mut out, &ctx);
        let energy: f64 = out.data.iter().map(|&s| (s as f64).powi(2)).sum();
        assert!(energy > 0.0);
    }

    #[test]
    fn gain_node_applies_gain() {
        let mut node = GainNode::new(0.5);
        let mut input = AudioBuf::new(bs(), 1, sr());
        for s in input.data.iter_mut() { *s = 1.0; }
        let mut out = AudioBuf::new(bs(), 1, sr());
        let ctx = ExecutionContext::new(sr(), bs(), 1);
        node.process(&[&input], &mut out, &ctx);
        assert!((out.data[0] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn pan_node_stereo() {
        let mut node = PanNode::new(1.0); // hard right
        let mut input = AudioBuf::new(bs(), 1, sr());
        for s in input.data.iter_mut() { *s = 1.0; }
        let mut out = AudioBuf::new(bs(), 2, sr());
        let ctx = ExecutionContext::new(sr(), bs(), 2);
        node.process(&[&input], &mut out, &ctx);
        assert!(out.get(0, 0) < 0.01); // left near silent
        assert!(out.get(0, 1) > 0.9);  // right loud
    }

    #[test]
    fn summing_mixer_sums() {
        let mut node = SummingMixerNode::new(2);
        let a = {
            let mut b = AudioBuf::new(bs(), 1, sr());
            for s in b.data.iter_mut() { *s = 0.3; }
            b
        };
        let b = {
            let mut b = AudioBuf::new(bs(), 1, sr());
            for s in b.data.iter_mut() { *s = 0.7; }
            b
        };
        let mut out = AudioBuf::new(bs(), 1, sr());
        let ctx = ExecutionContext::new(sr(), bs(), 1);
        node.process(&[&a, &b], &mut out, &ctx);
        assert!((out.data[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn noise_generator_node() {
        let mut node = NoiseGeneratorNode::new(NoiseColor::White);
        let mut out = AudioBuf::new(bs(), 1, sr());
        let ctx = ExecutionContext::new(sr(), bs(), 1);
        node.process(&[], &mut out, &ctx);
        let energy: f64 = out.data.iter().map(|&s| (s as f64).powi(2)).sum();
        assert!(energy > 0.0);
    }

    // -- AudioGraphExecutor ---------------------------------------------------

    #[test]
    fn executor_single_node() {
        let mut exec = AudioGraphExecutor::new(sr(), bs(), 1);
        exec.add_node(1, Box::new(SineOscillatorNode::new(440.0, sr() as f64)));
        exec.build_schedule().unwrap();
        let bufs = exec.process().unwrap();
        assert_eq!(bufs.len(), 1);
        let energy: f64 = bufs[0].data.iter().map(|&s| (s as f64).powi(2)).sum();
        assert!(energy > 0.0);
    }

    #[test]
    fn executor_chain() {
        let mut exec = AudioGraphExecutor::new(sr(), bs(), 1);
        let osc = exec.add_node(1, Box::new(SineOscillatorNode::new(440.0, sr() as f64)));
        let gain = exec.add_node(2, Box::new(GainNode::new(0.5)));
        exec.connect(osc, gain);
        exec.build_schedule().unwrap();
        let bufs = exec.process().unwrap();
        // Gain node output should be half amplitude of oscillator
        let osc_energy: f64 = bufs[0].data.iter().map(|&s| (s as f64).powi(2)).sum();
        let gain_energy: f64 = bufs[1].data.iter().map(|&s| (s as f64).powi(2)).sum();
        assert!(gain_energy < osc_energy);
    }

    #[test]
    fn executor_cycle_detection() {
        let mut exec = AudioGraphExecutor::new(sr(), bs(), 1);
        let a = exec.add_node(1, Box::new(GainNode::new(1.0)));
        let b = exec.add_node(2, Box::new(GainNode::new(1.0)));
        exec.connect(a, b);
        exec.connect(b, a);
        let result = exec.build_schedule();
        assert!(result.is_err());
    }

    #[test]
    fn executor_metrics() {
        let mut exec = AudioGraphExecutor::new(sr(), bs(), 1);
        exec.add_node(1, Box::new(SineOscillatorNode::new(440.0, sr() as f64)));
        exec.set_wcet_budget(10_000.0); // 10 ms
        exec.build_schedule().unwrap();
        exec.process().unwrap();
        assert!(exec.last_metrics().total_us > 0.0);
    }

    #[test]
    fn executor_reset() {
        let mut exec = AudioGraphExecutor::new(sr(), bs(), 1);
        exec.add_node(1, Box::new(SineOscillatorNode::new(440.0, sr() as f64)));
        exec.build_schedule().unwrap();
        exec.process().unwrap();
        assert!(exec.context().elapsed_samples > 0);
        exec.reset();
        assert_eq!(exec.context().elapsed_samples, 0);
    }

    #[test]
    fn executor_parameter_updates() {
        let mut exec = AudioGraphExecutor::new(sr(), bs(), 1);
        exec.add_node(1, Box::new(SineOscillatorNode::new(440.0, sr() as f64)));
        exec.register_parameter(Parameter::new(100, "freq", 440.0, 20.0, 20000.0));
        exec.build_schedule().unwrap();
        exec.queue_parameter_change(100, 880.0);
        exec.process().unwrap();
        let snap = exec.param_manager().snapshot();
        assert!((snap[&100] - 880.0).abs() < 1e-6);
    }
}

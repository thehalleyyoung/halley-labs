//! Streaming pipeline: end-to-end data → mapping → rendering → audio output.
//!
//! Provides a `PipelineStage` trait, concrete stage implementations, a state
//! machine, metrics collection, and a fluent `PipelineBuilder`.

use crate::buffer::RingBuffer;
use crate::data_stream::StreamDataValue;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// PipelineState
// ---------------------------------------------------------------------------

/// Pipeline state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PipelineState {
    Idle,
    Running,
    Paused,
    Error,
}

impl Default for PipelineState {
    fn default() -> Self {
        Self::Idle
    }
}

// ---------------------------------------------------------------------------
// PipelineMetrics
// ---------------------------------------------------------------------------

/// Metrics collected from the pipeline while it runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineMetrics {
    pub total_frames_processed: u64,
    pub latency_seconds: f64,
    pub avg_latency_seconds: f64,
    pub max_latency_seconds: f64,
    pub buffer_fill_ratio: f64,
    pub cpu_load_estimate: f64,
    pub underrun_count: u64,
    pub overrun_count: u64,
}

impl Default for PipelineMetrics {
    fn default() -> Self {
        Self {
            total_frames_processed: 0,
            latency_seconds: 0.0,
            avg_latency_seconds: 0.0,
            max_latency_seconds: 0.0,
            buffer_fill_ratio: 0.0,
            cpu_load_estimate: 0.0,
            underrun_count: 0,
            overrun_count: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// AudioChunk – the unit that flows through pipeline stages
// ---------------------------------------------------------------------------

/// A chunk of audio flowing between pipeline stages.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub samples: Vec<f32>,
    pub channels: usize,
    pub sample_rate: u32,
    pub timestamp_samples: u64,
}

impl AudioChunk {
    pub fn new(channels: usize, frames: usize, sample_rate: u32) -> Self {
        Self {
            samples: vec![0.0; channels * frames],
            channels,
            sample_rate,
            timestamp_samples: 0,
        }
    }

    pub fn frames(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.samples.len() / self.channels
        }
    }

    pub fn duration_seconds(&self) -> f64 {
        self.frames() as f64 / self.sample_rate as f64
    }
}

// ---------------------------------------------------------------------------
// MappedParameters – output of MappingStage
// ---------------------------------------------------------------------------

/// Parameters produced by the mapping stage for the renderer.
#[derive(Debug, Clone)]
pub struct MappedParameters {
    pub frequency_hz: f64,
    pub amplitude: f64,
    pub pan: f64,
    pub timbre_index: usize,
    pub duration_seconds: f64,
    pub data_value: f64,
}

impl Default for MappedParameters {
    fn default() -> Self {
        Self {
            frequency_hz: 440.0,
            amplitude: 0.5,
            pan: 0.0,
            timbre_index: 0,
            duration_seconds: 0.05,
            data_value: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// PipelineStage trait
// ---------------------------------------------------------------------------

/// A processing stage within the streaming pipeline.
pub trait PipelineStage: Send {
    /// Human-readable name of this stage.
    fn name(&self) -> &str;

    /// Process data. Receives data values and produces audio chunks.
    fn process(
        &mut self,
        input_data: &[StreamDataValue],
        output_audio: &mut Vec<AudioChunk>,
    );

    /// Reset internal state.
    fn reset(&mut self);

    /// Estimated latency introduced by this stage in seconds.
    fn latency_seconds(&self) -> f64 {
        0.0
    }
}

// ---------------------------------------------------------------------------
// DataInputStage
// ---------------------------------------------------------------------------

/// Receives streaming data and buffers it for downstream stages.
pub struct DataInputStage {
    name: String,
    buffer: VecDeque<StreamDataValue>,
    capacity: usize,
    total_received: u64,
}

impl DataInputStage {
    pub fn new(name: impl Into<String>, capacity: usize) -> Self {
        Self {
            name: name.into(),
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            total_received: 0,
        }
    }

    /// Push data into this stage from an external source.
    pub fn feed(&mut self, values: &[StreamDataValue]) {
        for v in values {
            if self.buffer.len() >= self.capacity {
                self.buffer.pop_front();
            }
            self.buffer.push_back(v.clone());
            self.total_received += 1;
        }
    }

    pub fn total_received(&self) -> u64 {
        self.total_received
    }

    pub fn available(&self) -> usize {
        self.buffer.len()
    }
}

impl PipelineStage for DataInputStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn process(
        &mut self,
        _input_data: &[StreamDataValue],
        _output_audio: &mut Vec<AudioChunk>,
    ) {
        // DataInputStage is a source — it doesn't consume upstream data.
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.total_received = 0;
    }
}

// ---------------------------------------------------------------------------
// MappingFunction
// ---------------------------------------------------------------------------

/// A configurable mapping from data values to sonification parameters.
#[derive(Debug, Clone)]
pub struct MappingFunction {
    pub data_min: f64,
    pub data_max: f64,
    pub freq_min_hz: f64,
    pub freq_max_hz: f64,
    pub amp_min: f64,
    pub amp_max: f64,
    pub pan_min: f64,
    pub pan_max: f64,
}

impl Default for MappingFunction {
    fn default() -> Self {
        Self {
            data_min: 0.0,
            data_max: 1.0,
            freq_min_hz: 200.0,
            freq_max_hz: 2000.0,
            amp_min: 0.1,
            amp_max: 0.9,
            pan_min: -1.0,
            pan_max: 1.0,
        }
    }
}

impl MappingFunction {
    fn map_range(value: f64, in_min: f64, in_max: f64, out_min: f64, out_max: f64) -> f64 {
        if (in_max - in_min).abs() < 1e-12 {
            return out_min;
        }
        let t = ((value - in_min) / (in_max - in_min)).clamp(0.0, 1.0);
        out_min + t * (out_max - out_min)
    }

    pub fn apply(&self, value: f64) -> MappedParameters {
        MappedParameters {
            frequency_hz: Self::map_range(
                value,
                self.data_min,
                self.data_max,
                self.freq_min_hz,
                self.freq_max_hz,
            ),
            amplitude: Self::map_range(
                value,
                self.data_min,
                self.data_max,
                self.amp_min,
                self.amp_max,
            ),
            pan: Self::map_range(
                value,
                self.data_min,
                self.data_max,
                self.pan_min,
                self.pan_max,
            ),
            timbre_index: 0,
            duration_seconds: 0.05,
            data_value: value,
        }
    }
}

// ---------------------------------------------------------------------------
// MappingStage
// ---------------------------------------------------------------------------

/// Applies a sonification mapping to data values, producing `MappedParameters`.
pub struct MappingStage {
    name: String,
    mapping: MappingFunction,
    params_buffer: Vec<MappedParameters>,
}

impl MappingStage {
    pub fn new(name: impl Into<String>, mapping: MappingFunction) -> Self {
        Self {
            name: name.into(),
            mapping,
            params_buffer: Vec::new(),
        }
    }

    pub fn set_mapping(&mut self, mapping: MappingFunction) {
        self.mapping = mapping;
    }

    pub fn latest_params(&self) -> Option<&MappedParameters> {
        self.params_buffer.last()
    }

    pub fn drain_params(&mut self) -> Vec<MappedParameters> {
        std::mem::take(&mut self.params_buffer)
    }
}

impl PipelineStage for MappingStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn process(
        &mut self,
        input_data: &[StreamDataValue],
        _output_audio: &mut Vec<AudioChunk>,
    ) {
        for v in input_data {
            if let Some(f) = v.as_f64() {
                let params = self.mapping.apply(f);
                self.params_buffer.push(params);
            }
        }
    }

    fn reset(&mut self) {
        self.params_buffer.clear();
    }
}

// ---------------------------------------------------------------------------
// RenderStage
// ---------------------------------------------------------------------------

/// Renders audio chunks from mapped parameters using simple oscillator
/// synthesis.
pub struct RenderStage {
    name: String,
    sample_rate: u32,
    channels: usize,
    frames_per_chunk: usize,
    phase: f64,
}

impl RenderStage {
    pub fn new(
        name: impl Into<String>,
        sample_rate: u32,
        channels: usize,
        frames_per_chunk: usize,
    ) -> Self {
        Self {
            name: name.into(),
            sample_rate,
            channels: channels.max(1),
            frames_per_chunk,
            phase: 0.0,
        }
    }

    /// Render one chunk from the given parameters.
    pub fn render(&mut self, params: &MappedParameters) -> AudioChunk {
        let mut chunk = AudioChunk::new(self.channels, self.frames_per_chunk, self.sample_rate);
        let freq = params.frequency_hz;
        let amp = params.amplitude as f32;
        let phase_inc = freq / self.sample_rate as f64;

        for frame in 0..self.frames_per_chunk {
            let sample = (self.phase * std::f64::consts::TAU).sin() as f32 * amp;
            self.phase += phase_inc;
            if self.phase >= 1.0 {
                self.phase -= 1.0;
            }
            for ch in 0..self.channels {
                // Simple pan: attenuate left/right
                let pan = params.pan as f32;
                let gain = if self.channels == 2 {
                    if ch == 0 {
                        ((1.0 - pan) / 2.0).sqrt()
                    } else {
                        ((1.0 + pan) / 2.0).sqrt()
                    }
                } else {
                    1.0
                };
                chunk.samples[frame * self.channels + ch] = sample * gain;
            }
        }
        chunk
    }
}

impl PipelineStage for RenderStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn process(
        &mut self,
        _input_data: &[StreamDataValue],
        _output_audio: &mut Vec<AudioChunk>,
    ) {
        // RenderStage is driven by MappedParameters, not raw data.
    }

    fn reset(&mut self) {
        self.phase = 0.0;
    }

    fn latency_seconds(&self) -> f64 {
        self.frames_per_chunk as f64 / self.sample_rate as f64
    }
}

// ---------------------------------------------------------------------------
// OutputStage
// ---------------------------------------------------------------------------

/// Collects rendered audio chunks into a final output ring buffer.
pub struct OutputStage {
    name: String,
    chunks: VecDeque<AudioChunk>,
    capacity: usize,
    total_output_frames: u64,
}

impl OutputStage {
    pub fn new(name: impl Into<String>, capacity: usize) -> Self {
        Self {
            name: name.into(),
            chunks: VecDeque::with_capacity(capacity),
            capacity,
            total_output_frames: 0,
        }
    }

    pub fn deliver(&mut self, chunk: AudioChunk) {
        self.total_output_frames += chunk.frames() as u64;
        if self.chunks.len() >= self.capacity {
            self.chunks.pop_front();
        }
        self.chunks.push_back(chunk);
    }

    pub fn take_chunk(&mut self) -> Option<AudioChunk> {
        self.chunks.pop_front()
    }

    pub fn available_chunks(&self) -> usize {
        self.chunks.len()
    }

    pub fn total_output_frames(&self) -> u64 {
        self.total_output_frames
    }
}

impl PipelineStage for OutputStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn process(
        &mut self,
        _input_data: &[StreamDataValue],
        _output_audio: &mut Vec<AudioChunk>,
    ) {
        // OutputStage is a sink.
    }

    fn reset(&mut self) {
        self.chunks.clear();
        self.total_output_frames = 0;
    }
}

// ---------------------------------------------------------------------------
// StreamingPipeline
// ---------------------------------------------------------------------------

/// End-to-end streaming pipeline: data input → mapping → rendering → output.
pub struct StreamingPipeline {
    state: PipelineState,
    input: DataInputStage,
    mapping: MappingStage,
    render: RenderStage,
    output: OutputStage,
    metrics: PipelineMetrics,
    latency_history: VecDeque<f64>,
    max_latency_history: usize,
    process_start: Option<Instant>,
    callback_budget_seconds: f64,
}

impl StreamingPipeline {
    pub fn new(
        input: DataInputStage,
        mapping: MappingStage,
        render: RenderStage,
        output: OutputStage,
    ) -> Self {
        let budget = render.latency_seconds();
        Self {
            state: PipelineState::Idle,
            input,
            mapping,
            render,
            output,
            metrics: PipelineMetrics::default(),
            latency_history: VecDeque::with_capacity(256),
            max_latency_history: 256,
            process_start: None,
            callback_budget_seconds: budget,
        }
    }

    pub fn state(&self) -> PipelineState {
        self.state
    }

    pub fn start(&mut self) {
        if self.state == PipelineState::Idle || self.state == PipelineState::Paused {
            self.state = PipelineState::Running;
        }
    }

    pub fn stop(&mut self) {
        self.state = PipelineState::Idle;
        self.input.reset();
        self.mapping.reset();
        self.render.reset();
        self.output.reset();
        self.metrics = PipelineMetrics::default();
        self.latency_history.clear();
    }

    pub fn pause(&mut self) {
        if self.state == PipelineState::Running {
            self.state = PipelineState::Paused;
        }
    }

    pub fn resume(&mut self) {
        if self.state == PipelineState::Paused {
            self.state = PipelineState::Running;
        }
    }

    /// Feed data values from external source.
    pub fn feed_data(&mut self, values: &[StreamDataValue]) {
        self.input.feed(values);
    }

    /// Run one processing cycle: map buffered data and render audio.
    pub fn process(&mut self) {
        if self.state != PipelineState::Running {
            return;
        }
        let start = Instant::now();
        self.process_start = Some(start);

        // Drain data from input
        let mut data_batch: Vec<StreamDataValue> = Vec::new();
        while let Some(v) = self.input.buffer.pop_front() {
            data_batch.push(v);
        }

        // Map
        let mut audio_out = Vec::new();
        self.mapping.process(&data_batch, &mut audio_out);

        // Render each mapped parameter set
        let params = self.mapping.drain_params();
        for p in &params {
            let chunk = self.render.render(p);
            self.output.deliver(chunk);
            self.metrics.total_frames_processed += self.render.frames_per_chunk as u64;
        }

        // Latency measurement
        let elapsed = start.elapsed().as_secs_f64();
        self.metrics.latency_seconds = elapsed;
        if self.latency_history.len() >= self.max_latency_history {
            self.latency_history.pop_front();
        }
        self.latency_history.push_back(elapsed);

        if !self.latency_history.is_empty() {
            let sum: f64 = self.latency_history.iter().sum();
            self.metrics.avg_latency_seconds = sum / self.latency_history.len() as f64;
            self.metrics.max_latency_seconds = self
                .latency_history
                .iter()
                .cloned()
                .fold(0.0f64, f64::max);
        }

        // Buffer fill
        let cap = self.output.capacity as f64;
        self.metrics.buffer_fill_ratio = if cap > 0.0 {
            self.output.available_chunks() as f64 / cap
        } else {
            0.0
        };

        // CPU load estimate (processing time / callback budget)
        if self.callback_budget_seconds > 0.0 {
            self.metrics.cpu_load_estimate = elapsed / self.callback_budget_seconds;
        }

        // Underrun detection
        if data_batch.is_empty() {
            self.metrics.underrun_count += 1;
        }
    }

    /// Take the next rendered audio chunk from the output.
    pub fn take_output(&mut self) -> Option<AudioChunk> {
        self.output.take_chunk()
    }

    pub fn metrics(&self) -> &PipelineMetrics {
        &self.metrics
    }

    pub fn output_available(&self) -> usize {
        self.output.available_chunks()
    }

    pub fn input_available(&self) -> usize {
        self.input.available()
    }
}

// ---------------------------------------------------------------------------
// PipelineBuilder
// ---------------------------------------------------------------------------

/// Fluent API for constructing a `StreamingPipeline`.
pub struct PipelineBuilder {
    input_capacity: usize,
    output_capacity: usize,
    sample_rate: u32,
    channels: usize,
    frames_per_chunk: usize,
    mapping: MappingFunction,
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineBuilder {
    pub fn new() -> Self {
        Self {
            input_capacity: 1024,
            output_capacity: 64,
            sample_rate: 44100,
            channels: 2,
            frames_per_chunk: 256,
            mapping: MappingFunction::default(),
        }
    }

    pub fn input_capacity(mut self, cap: usize) -> Self {
        self.input_capacity = cap;
        self
    }

    pub fn output_capacity(mut self, cap: usize) -> Self {
        self.output_capacity = cap;
        self
    }

    pub fn sample_rate(mut self, sr: u32) -> Self {
        self.sample_rate = sr;
        self
    }

    pub fn channels(mut self, ch: usize) -> Self {
        self.channels = ch;
        self
    }

    pub fn frames_per_chunk(mut self, f: usize) -> Self {
        self.frames_per_chunk = f;
        self
    }

    pub fn mapping(mut self, m: MappingFunction) -> Self {
        self.mapping = m;
        self
    }

    pub fn data_range(mut self, min: f64, max: f64) -> Self {
        self.mapping.data_min = min;
        self.mapping.data_max = max;
        self
    }

    pub fn frequency_range(mut self, min_hz: f64, max_hz: f64) -> Self {
        self.mapping.freq_min_hz = min_hz;
        self.mapping.freq_max_hz = max_hz;
        self
    }

    pub fn amplitude_range(mut self, min: f64, max: f64) -> Self {
        self.mapping.amp_min = min;
        self.mapping.amp_max = max;
        self
    }

    pub fn build(self) -> StreamingPipeline {
        let input = DataInputStage::new("data-input", self.input_capacity);
        let mapping = MappingStage::new("mapping", self.mapping);
        let render = RenderStage::new(
            "render",
            self.sample_rate,
            self.channels,
            self.frames_per_chunk,
        );
        let output = OutputStage::new("output", self.output_capacity);
        StreamingPipeline::new(input, mapping, render, output)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_builder_defaults() {
        let p = PipelineBuilder::new().build();
        assert_eq!(p.state(), PipelineState::Idle);
    }

    #[test]
    fn pipeline_state_transitions() {
        let mut p = PipelineBuilder::new().build();
        p.start();
        assert_eq!(p.state(), PipelineState::Running);
        p.pause();
        assert_eq!(p.state(), PipelineState::Paused);
        p.resume();
        assert_eq!(p.state(), PipelineState::Running);
        p.stop();
        assert_eq!(p.state(), PipelineState::Idle);
    }

    #[test]
    fn pipeline_feed_and_process() {
        let mut p = PipelineBuilder::new()
            .data_range(0.0, 100.0)
            .frequency_range(200.0, 800.0)
            .build();
        p.start();
        p.feed_data(&[StreamDataValue::Float(50.0)]);
        p.process();
        assert!(p.output_available() > 0);
    }

    #[test]
    fn pipeline_process_idle_noop() {
        let mut p = PipelineBuilder::new().build();
        p.feed_data(&[StreamDataValue::Float(1.0)]);
        p.process();
        assert_eq!(p.output_available(), 0);
    }

    #[test]
    fn pipeline_metrics_update() {
        let mut p = PipelineBuilder::new().build();
        p.start();
        p.feed_data(&[StreamDataValue::Float(0.5)]);
        p.process();
        assert!(p.metrics().total_frames_processed > 0);
    }

    #[test]
    fn pipeline_stop_resets() {
        let mut p = PipelineBuilder::new().build();
        p.start();
        p.feed_data(&[StreamDataValue::Float(0.5)]);
        p.process();
        p.stop();
        assert_eq!(p.metrics().total_frames_processed, 0);
        assert_eq!(p.output_available(), 0);
    }

    #[test]
    fn audio_chunk_basics() {
        let chunk = AudioChunk::new(2, 128, 44100);
        assert_eq!(chunk.frames(), 128);
        assert_eq!(chunk.samples.len(), 256);
    }

    #[test]
    fn mapping_function_map_range() {
        let mf = MappingFunction {
            data_min: 0.0,
            data_max: 100.0,
            freq_min_hz: 200.0,
            freq_max_hz: 2000.0,
            amp_min: 0.0,
            amp_max: 1.0,
            pan_min: -1.0,
            pan_max: 1.0,
        };
        let p = mf.apply(50.0);
        assert!((p.frequency_hz - 1100.0).abs() < 1.0);
        assert!((p.amplitude - 0.5).abs() < 0.01);
    }

    #[test]
    fn render_stage_produces_audio() {
        let mut rs = RenderStage::new("r", 44100, 1, 64);
        let params = MappedParameters::default();
        let chunk = rs.render(&params);
        assert_eq!(chunk.frames(), 64);
        assert!(chunk.samples.iter().any(|&s| s != 0.0));
    }

    #[test]
    fn output_stage_deliver_take() {
        let mut os = OutputStage::new("out", 4);
        let chunk = AudioChunk::new(1, 32, 44100);
        os.deliver(chunk);
        assert_eq!(os.available_chunks(), 1);
        let c = os.take_chunk().unwrap();
        assert_eq!(c.frames(), 32);
    }

    #[test]
    fn data_input_stage_feed() {
        let mut dis = DataInputStage::new("in", 8);
        dis.feed(&[StreamDataValue::Float(1.0), StreamDataValue::Float(2.0)]);
        assert_eq!(dis.available(), 2);
        assert_eq!(dis.total_received(), 2);
    }

    #[test]
    fn pipeline_underrun_detection() {
        let mut p = PipelineBuilder::new().build();
        p.start();
        p.process(); // no data → underrun
        assert!(p.metrics().underrun_count > 0);
    }

    #[test]
    fn pipeline_builder_custom() {
        let p = PipelineBuilder::new()
            .sample_rate(48000)
            .channels(1)
            .frames_per_chunk(512)
            .input_capacity(2048)
            .output_capacity(32)
            .build();
        assert_eq!(p.state(), PipelineState::Idle);
    }

    #[test]
    fn pipeline_multiple_process_cycles() {
        let mut p = PipelineBuilder::new().build();
        p.start();
        for i in 0..10 {
            p.feed_data(&[StreamDataValue::Float(i as f64)]);
            p.process();
        }
        assert!(p.metrics().total_frames_processed > 0);
        assert!(p.metrics().avg_latency_seconds >= 0.0);
    }
}

//! High-level rendering API: offline, real-time, preview, and session
//! management.
//!
//! These renderers orchestrate an [`AudioGraphExecutor`] over the lifetime of a
//! sonification, feeding data, processing buffers, and writing output.

use std::time::Instant;

use crate::executor::{AudioGraphExecutor, ExecutionContext, NodeProcessor, PerformanceMetrics};
use crate::output::{BufferedOutput, NullOutput, OutputFormat, WavWriter, Metering};
use crate::parameter::Parameter;
use crate::{AudioBuf, RendererResult, RendererError};

// ---------------------------------------------------------------------------
// RenderProgress
// ---------------------------------------------------------------------------

/// Progress information emitted by renderers.
#[derive(Debug, Clone, Copy)]
pub struct RenderProgress {
    /// Buffers processed so far.
    pub buffers_done: u64,
    /// Total buffers expected.
    pub buffers_total: u64,
    /// Wall-clock time elapsed in seconds.
    pub elapsed_seconds: f64,
    /// Estimated time remaining in seconds.
    pub eta_seconds: f64,
}

impl RenderProgress {
    pub fn fraction(&self) -> f64 {
        if self.buffers_total == 0 { return 1.0; }
        self.buffers_done as f64 / self.buffers_total as f64
    }
}

// ---------------------------------------------------------------------------
// OfflineRenderer
// ---------------------------------------------------------------------------

/// Renders an entire sonification to a buffer or WAV file, processing all
/// buffers sequentially with no real-time constraint.
pub struct OfflineRenderer {
    pub executor: AudioGraphExecutor,
    pub sample_rate: u32,
    pub buffer_size: usize,
    pub channels: usize,
    /// Duration in seconds to render.
    pub duration: f64,
    /// Optional progress callback.
    progress_callback: Option<Box<dyn FnMut(RenderProgress)>>,
}

impl OfflineRenderer {
    pub fn new(sample_rate: u32, buffer_size: usize, channels: usize, duration: f64) -> Self {
        Self {
            executor: AudioGraphExecutor::new(sample_rate, buffer_size, channels),
            sample_rate,
            buffer_size,
            channels,
            duration,
            progress_callback: None,
        }
    }

    pub fn set_progress_callback<F: FnMut(RenderProgress) + 'static>(&mut self, cb: F) {
        self.progress_callback = Some(Box::new(cb));
    }

    /// Add a node to the underlying executor. Returns the node index.
    pub fn add_node(&mut self, id: u64, processor: Box<dyn NodeProcessor>) -> usize {
        self.executor.add_node(id, processor)
    }

    pub fn connect(&mut self, from: usize, to: usize) {
        self.executor.connect(from, to);
    }

    pub fn register_parameter(&mut self, param: Parameter) {
        self.executor.register_parameter(param);
    }

    /// Render the entire duration into a [`BufferedOutput`].
    pub fn render_to_buffer(&mut self) -> RendererResult<BufferedOutput> {
        self.executor.build_schedule()?;

        let total_samples = (self.duration * self.sample_rate as f64) as u64;
        let total_buffers = (total_samples as f64 / self.buffer_size as f64).ceil() as u64;
        let mut output = BufferedOutput::new(self.channels, self.sample_rate);
        let start = Instant::now();

        for buf_idx in 0..total_buffers {
            let bufs = self.executor.process()?;
            // Take the last node's output as the final mix
            if let Some(last) = bufs.last() {
                output.write_buffer(last);
            }

            if let Some(ref mut cb) = self.progress_callback {
                let elapsed = start.elapsed().as_secs_f64();
                let rate = if buf_idx > 0 { elapsed / buf_idx as f64 } else { 0.0 };
                cb(RenderProgress {
                    buffers_done: buf_idx + 1,
                    buffers_total: total_buffers,
                    elapsed_seconds: elapsed,
                    eta_seconds: rate * (total_buffers - buf_idx - 1) as f64,
                });
            }
        }

        Ok(output)
    }

    /// Render directly to a WAV file.
    pub fn render_to_wav(&mut self, path: &str, format: OutputFormat) -> RendererResult<()> {
        let buffered = self.render_to_buffer()?;
        let buf = buffered.as_audio_buf();
        crate::output::write_wav_file(path, &buf, format)
    }
}

impl std::fmt::Debug for OfflineRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OfflineRenderer")
            .field("sample_rate", &self.sample_rate)
            .field("buffer_size", &self.buffer_size)
            .field("channels", &self.channels)
            .field("duration", &self.duration)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// RealTimeRenderer
// ---------------------------------------------------------------------------

/// Rendering state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderState {
    Idle,
    Running,
    Underrun,
}

/// Renderer that operates under real-time constraints with double-buffering,
/// underrun detection, and WCET monitoring.
pub struct RealTimeRenderer {
    pub executor: AudioGraphExecutor,
    pub sample_rate: u32,
    pub buffer_size: usize,
    pub channels: usize,
    state: RenderState,
    /// Double buffers (ping-pong).
    front_buffer: AudioBuf,
    back_buffer: AudioBuf,
    /// Underrun count.
    underrun_count: u64,
    /// WCET budget in microseconds.
    wcet_budget_us: f64,
}

impl RealTimeRenderer {
    pub fn new(sample_rate: u32, buffer_size: usize, channels: usize) -> Self {
        Self {
            executor: AudioGraphExecutor::new(sample_rate, buffer_size, channels),
            sample_rate,
            buffer_size,
            channels,
            state: RenderState::Idle,
            front_buffer: AudioBuf::new(buffer_size, channels, sample_rate),
            back_buffer: AudioBuf::new(buffer_size, channels, sample_rate),
            underrun_count: 0,
            wcet_budget_us: 0.0,
        }
    }

    pub fn set_wcet_budget(&mut self, budget_us: f64) {
        self.wcet_budget_us = budget_us;
        self.executor.set_wcet_budget(budget_us);
    }

    /// Start the renderer.
    pub fn start(&mut self) -> RendererResult<()> {
        self.executor.build_schedule()?;
        self.state = RenderState::Running;
        Ok(())
    }

    /// Audio callback: fills `output` with the next buffer.
    /// Returns `true` if an underrun occurred.
    pub fn audio_callback(&mut self, output: &mut AudioBuf) -> bool {
        if self.state != RenderState::Running {
            output.zero();
            return false;
        }

        let start = Instant::now();
        match self.executor.process() {
            Ok(bufs) => {
                if let Some(last) = bufs.last() {
                    output.copy_from_buf(last);
                } else {
                    output.zero();
                }
            }
            Err(_) => {
                output.zero();
                self.underrun_count += 1;
                self.state = RenderState::Underrun;
                return true;
            }
        }

        let elapsed_us = start.elapsed().as_secs_f64() * 1e6;
        let budget = if self.wcet_budget_us > 0.0 {
            self.wcet_budget_us
        } else {
            // Default budget: buffer duration minus 10%
            self.buffer_size as f64 / self.sample_rate as f64 * 1e6 * 0.9
        };

        if elapsed_us > budget {
            self.underrun_count += 1;
            self.state = RenderState::Underrun;
            return true;
        }

        false
    }

    /// Attempt recovery from underrun.
    pub fn recover(&mut self) {
        self.state = RenderState::Running;
    }

    /// Swap front and back buffers.
    pub fn swap_buffers(&mut self) {
        std::mem::swap(&mut self.front_buffer, &mut self.back_buffer);
    }

    /// Get the front (display) buffer.
    pub fn front_buffer(&self) -> &AudioBuf {
        &self.front_buffer
    }

    pub fn state(&self) -> RenderState {
        self.state
    }

    pub fn underrun_count(&self) -> u64 {
        self.underrun_count
    }

    pub fn stop(&mut self) {
        self.state = RenderState::Idle;
    }
}

impl std::fmt::Debug for RealTimeRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RealTimeRenderer")
            .field("sample_rate", &self.sample_rate)
            .field("buffer_size", &self.buffer_size)
            .field("state", &self.state)
            .field("underruns", &self.underrun_count)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// PreviewRenderer
// ---------------------------------------------------------------------------

/// Quick preview renderer with reduced quality for fast iteration.
#[derive(Debug)]
pub struct PreviewRenderer {
    pub executor: AudioGraphExecutor,
    /// Preview sample rate (typically lower than production).
    pub sample_rate: u32,
    pub buffer_size: usize,
    pub channels: usize,
    /// Maximum duration to render in seconds.
    pub max_duration: f64,
}

impl PreviewRenderer {
    /// Create a preview renderer at a reduced sample rate.
    pub fn new(max_duration: f64) -> Self {
        let sample_rate = 22050; // half of CD quality
        let buffer_size = 512;
        let channels = 1; // mono for speed
        Self {
            executor: AudioGraphExecutor::new(sample_rate, buffer_size, channels),
            sample_rate,
            buffer_size,
            channels,
            max_duration,
        }
    }

    /// Create with custom parameters.
    pub fn with_config(sample_rate: u32, buffer_size: usize, channels: usize, max_duration: f64) -> Self {
        Self {
            executor: AudioGraphExecutor::new(sample_rate, buffer_size, channels),
            sample_rate,
            buffer_size,
            channels,
            max_duration,
        }
    }

    /// Render a quick preview into memory.
    pub fn render(&mut self) -> RendererResult<BufferedOutput> {
        self.executor.build_schedule()?;
        let total_samples = (self.max_duration * self.sample_rate as f64) as u64;
        let total_buffers = (total_samples as f64 / self.buffer_size as f64).ceil() as u64;
        let mut output = BufferedOutput::new(self.channels, self.sample_rate);

        for _ in 0..total_buffers {
            let bufs = self.executor.process()?;
            if let Some(last) = bufs.last() {
                output.write_buffer(last);
            }
        }
        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// RenderSession
// ---------------------------------------------------------------------------

/// Session state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    Created,
    Running,
    Paused,
    Stopped,
}

/// Manages a rendering session with start/pause/resume/stop control and
/// metering queries.
pub struct RenderSession {
    pub executor: AudioGraphExecutor,
    pub sample_rate: u32,
    pub buffer_size: usize,
    pub channels: usize,
    state: SessionState,
    metering: Metering,
    buffers_processed: u64,
    accumulated: BufferedOutput,
    start_time: Option<Instant>,
}

impl RenderSession {
    pub fn new(sample_rate: u32, buffer_size: usize, channels: usize) -> Self {
        Self {
            executor: AudioGraphExecutor::new(sample_rate, buffer_size, channels),
            sample_rate,
            buffer_size,
            channels,
            state: SessionState::Created,
            metering: Metering::new(channels, sample_rate as f64),
            buffers_processed: 0,
            accumulated: BufferedOutput::new(channels, sample_rate),
            start_time: None,
        }
    }

    /// Start (or restart) the session.
    pub fn start(&mut self) -> RendererResult<()> {
        self.executor.build_schedule()?;
        self.state = SessionState::Running;
        self.start_time = Some(Instant::now());
        Ok(())
    }

    pub fn pause(&mut self) {
        if self.state == SessionState::Running {
            self.state = SessionState::Paused;
        }
    }

    pub fn resume(&mut self) {
        if self.state == SessionState::Paused {
            self.state = SessionState::Running;
        }
    }

    pub fn stop(&mut self) {
        self.state = SessionState::Stopped;
    }

    pub fn state(&self) -> SessionState {
        self.state
    }

    /// Process one buffer if the session is running. Returns `None` if paused
    /// or stopped.
    pub fn process_buffer(&mut self) -> RendererResult<Option<AudioBuf>> {
        if self.state != SessionState::Running {
            return Ok(None);
        }

        let bufs = self.executor.process()?;
        self.buffers_processed += 1;

        if let Some(last) = bufs.last() {
            self.metering.process(last);
            self.accumulated.write_buffer(last);
            Ok(Some(last.clone()))
        } else {
            Ok(None)
        }
    }

    /// Current playback position in seconds.
    pub fn position_seconds(&self) -> f64 {
        self.buffers_processed as f64 * self.buffer_size as f64 / self.sample_rate as f64
    }

    /// Current playback position in samples.
    pub fn position_samples(&self) -> u64 {
        self.buffers_processed * self.buffer_size as u64
    }

    /// Query the metering state.
    pub fn metering(&self) -> &Metering {
        &self.metering
    }

    /// Get accumulated output.
    pub fn output(&self) -> &BufferedOutput {
        &self.accumulated
    }

    /// Last execution metrics.
    pub fn metrics(&self) -> &PerformanceMetrics {
        self.executor.last_metrics()
    }

    /// Wall-clock time since start.
    pub fn wall_time(&self) -> f64 {
        self.start_time.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0)
    }

    /// Buffers processed.
    pub fn buffers_processed(&self) -> u64 {
        self.buffers_processed
    }

    pub fn reset(&mut self) {
        self.executor.reset();
        self.metering.reset();
        self.accumulated.reset();
        self.buffers_processed = 0;
        self.state = SessionState::Created;
        self.start_time = None;
    }
}

impl std::fmt::Debug for RenderSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenderSession")
            .field("sample_rate", &self.sample_rate)
            .field("state", &self.state)
            .field("position_s", &self.position_seconds())
            .field("buffers", &self.buffers_processed)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::{SineOscillatorNode, GainNode};

    fn sr() -> u32 { 44100 }
    fn bs() -> usize { 256 }

    // -- OfflineRenderer -------------------------------------------------------

    #[test]
    fn offline_render_basic() {
        let mut r = OfflineRenderer::new(sr(), bs(), 1, 0.1);
        r.add_node(1, Box::new(SineOscillatorNode::new(440.0, sr() as f64)));
        r.executor.build_schedule().unwrap();
        let buf = r.render_to_buffer().unwrap();
        assert!(buf.frames() > 0);
    }

    #[test]
    fn offline_render_duration() {
        let dur = 0.5;
        let mut r = OfflineRenderer::new(sr(), bs(), 1, dur);
        r.add_node(1, Box::new(SineOscillatorNode::new(440.0, sr() as f64)));
        let buf = r.render_to_buffer().unwrap();
        let expected = (dur * sr() as f64) as usize;
        // Allow ±buffer_size tolerance
        assert!((buf.frames() as i64 - expected as i64).unsigned_abs() < bs() as u64 * 2);
    }

    #[test]
    fn offline_progress_callback() {
        let mut r = OfflineRenderer::new(sr(), bs(), 1, 0.05);
        r.add_node(1, Box::new(SineOscillatorNode::new(440.0, sr() as f64)));
        let mut last_fraction = 0.0f64;
        r.set_progress_callback(move |p| {
            assert!(p.fraction() >= last_fraction);
            last_fraction = p.fraction();
        });
        r.render_to_buffer().unwrap();
    }

    // -- RealTimeRenderer ------------------------------------------------------

    #[test]
    fn realtime_start_stop() {
        let mut rt = RealTimeRenderer::new(sr(), bs(), 1);
        rt.executor.add_node(1, Box::new(SineOscillatorNode::new(440.0, sr() as f64)));
        rt.start().unwrap();
        assert_eq!(rt.state(), RenderState::Running);
        let mut out = AudioBuf::new(bs(), 1, sr());
        let underrun = rt.audio_callback(&mut out);
        assert!(!underrun);
        rt.stop();
        assert_eq!(rt.state(), RenderState::Idle);
    }

    #[test]
    fn realtime_produces_audio() {
        let mut rt = RealTimeRenderer::new(sr(), bs(), 1);
        rt.executor.add_node(1, Box::new(SineOscillatorNode::new(440.0, sr() as f64)));
        rt.start().unwrap();
        let mut out = AudioBuf::new(bs(), 1, sr());
        rt.audio_callback(&mut out);
        let energy: f64 = out.data.iter().map(|&s| (s as f64).powi(2)).sum();
        assert!(energy > 0.0);
    }

    #[test]
    fn realtime_recover_from_underrun() {
        let mut rt = RealTimeRenderer::new(sr(), bs(), 1);
        rt.executor.add_node(1, Box::new(SineOscillatorNode::new(440.0, sr() as f64)));
        rt.start().unwrap();
        // Simulate underrun by setting an impossible budget
        rt.set_wcet_budget(0.001); // 1 ns – practically impossible
        let mut out = AudioBuf::new(bs(), 1, sr());
        let _underrun = rt.audio_callback(&mut out);
        // Whether or not it underran, recovery should be possible
        rt.recover();
        assert_eq!(rt.state(), RenderState::Running);
    }

    // -- PreviewRenderer -------------------------------------------------------

    #[test]
    fn preview_renders_quickly() {
        let mut pv = PreviewRenderer::new(0.1);
        pv.executor.add_node(1, Box::new(SineOscillatorNode::new(440.0, pv.sample_rate as f64)));
        let buf = pv.render().unwrap();
        assert!(buf.frames() > 0);
    }

    #[test]
    fn preview_reduced_sample_rate() {
        let pv = PreviewRenderer::new(1.0);
        assert_eq!(pv.sample_rate, 22050);
    }

    // -- RenderSession ---------------------------------------------------------

    #[test]
    fn session_lifecycle() {
        let mut sess = RenderSession::new(sr(), bs(), 1);
        assert_eq!(sess.state(), SessionState::Created);
        sess.executor.add_node(1, Box::new(SineOscillatorNode::new(440.0, sr() as f64)));
        sess.start().unwrap();
        assert_eq!(sess.state(), SessionState::Running);
        sess.pause();
        assert_eq!(sess.state(), SessionState::Paused);
        sess.resume();
        assert_eq!(sess.state(), SessionState::Running);
        sess.stop();
        assert_eq!(sess.state(), SessionState::Stopped);
    }

    #[test]
    fn session_processes_buffers() {
        let mut sess = RenderSession::new(sr(), bs(), 1);
        sess.executor.add_node(1, Box::new(SineOscillatorNode::new(440.0, sr() as f64)));
        sess.start().unwrap();
        let buf = sess.process_buffer().unwrap();
        assert!(buf.is_some());
        assert_eq!(sess.buffers_processed(), 1);
    }

    #[test]
    fn session_paused_returns_none() {
        let mut sess = RenderSession::new(sr(), bs(), 1);
        sess.executor.add_node(1, Box::new(SineOscillatorNode::new(440.0, sr() as f64)));
        sess.start().unwrap();
        sess.pause();
        let buf = sess.process_buffer().unwrap();
        assert!(buf.is_none());
    }

    #[test]
    fn session_position() {
        let mut sess = RenderSession::new(sr(), bs(), 1);
        sess.executor.add_node(1, Box::new(SineOscillatorNode::new(440.0, sr() as f64)));
        sess.start().unwrap();
        sess.process_buffer().unwrap();
        sess.process_buffer().unwrap();
        let expected = 2.0 * bs() as f64 / sr() as f64;
        assert!((sess.position_seconds() - expected).abs() < 1e-6);
    }

    #[test]
    fn session_metering() {
        let mut sess = RenderSession::new(sr(), bs(), 1);
        sess.executor.add_node(1, Box::new(SineOscillatorNode::new(440.0, sr() as f64)));
        sess.start().unwrap();
        sess.process_buffer().unwrap();
        let peaks = sess.metering().peak_levels();
        assert!(peaks[0] > 0.0);
    }

    #[test]
    fn session_reset() {
        let mut sess = RenderSession::new(sr(), bs(), 1);
        sess.executor.add_node(1, Box::new(SineOscillatorNode::new(440.0, sr() as f64)));
        sess.start().unwrap();
        sess.process_buffer().unwrap();
        sess.reset();
        assert_eq!(sess.state(), SessionState::Created);
        assert_eq!(sess.buffers_processed(), 0);
    }

    #[test]
    fn render_progress_fraction() {
        let p = RenderProgress {
            buffers_done: 50,
            buffers_total: 100,
            elapsed_seconds: 1.0,
            eta_seconds: 1.0,
        };
        assert!((p.fraction() - 0.5).abs() < 1e-12);
    }
}

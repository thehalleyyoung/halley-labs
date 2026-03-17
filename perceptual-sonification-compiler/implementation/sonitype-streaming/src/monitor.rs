//! Streaming monitoring: buffer health, latency statistics, data throughput,
//! audio quality metrics, health checks, and performance logging.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant, SystemTime};

// ---------------------------------------------------------------------------
// StreamMonitor
// ---------------------------------------------------------------------------

/// Tracks buffer underruns/overruns, latency statistics, data throughput,
/// and audio quality metrics (clipping, silence detection).
#[derive(Debug, Clone)]
pub struct StreamMonitor {
    // Buffer health
    underrun_count: u64,
    overrun_count: u64,

    // Latency
    latency_history: VecDeque<f64>,
    max_latency_history: usize,
    min_latency: f64,
    max_latency: f64,
    latency_sum: f64,
    latency_count: u64,

    // Throughput
    data_samples_received: u64,
    audio_samples_output: u64,
    throughput_window_start: Instant,
    throughput_window_data: u64,
    throughput_window_audio: u64,

    // Audio quality
    clip_count: u64,
    silence_frames: u64,
    total_frames: u64,
    silence_threshold: f32,
    clip_threshold: f32,
}

impl StreamMonitor {
    pub fn new() -> Self {
        Self {
            underrun_count: 0,
            overrun_count: 0,
            latency_history: VecDeque::with_capacity(1024),
            max_latency_history: 1024,
            min_latency: f64::INFINITY,
            max_latency: 0.0,
            latency_sum: 0.0,
            latency_count: 0,
            data_samples_received: 0,
            audio_samples_output: 0,
            throughput_window_start: Instant::now(),
            throughput_window_data: 0,
            throughput_window_audio: 0,
            clip_count: 0,
            silence_frames: 0,
            total_frames: 0,
            silence_threshold: 1e-6,
            clip_threshold: 0.999,
        }
    }

    // -- Buffer health ------------------------------------------------------

    pub fn record_underrun(&mut self) {
        self.underrun_count += 1;
    }

    pub fn record_overrun(&mut self) {
        self.overrun_count += 1;
    }

    pub fn underrun_count(&self) -> u64 {
        self.underrun_count
    }

    pub fn overrun_count(&self) -> u64 {
        self.overrun_count
    }

    // -- Latency ------------------------------------------------------------

    pub fn record_latency(&mut self, seconds: f64) {
        if self.latency_history.len() >= self.max_latency_history {
            self.latency_history.pop_front();
        }
        self.latency_history.push_back(seconds);
        if seconds < self.min_latency {
            self.min_latency = seconds;
        }
        if seconds > self.max_latency {
            self.max_latency = seconds;
        }
        self.latency_sum += seconds;
        self.latency_count += 1;
    }

    pub fn min_latency(&self) -> f64 {
        if self.latency_count == 0 {
            0.0
        } else {
            self.min_latency
        }
    }

    pub fn max_latency(&self) -> f64 {
        self.max_latency
    }

    pub fn average_latency(&self) -> f64 {
        if self.latency_count == 0 {
            0.0
        } else {
            self.latency_sum / self.latency_count as f64
        }
    }

    /// Latency jitter: standard deviation over recent history.
    pub fn latency_jitter(&self) -> f64 {
        if self.latency_history.len() < 2 {
            return 0.0;
        }
        let mean = self.latency_history.iter().sum::<f64>()
            / self.latency_history.len() as f64;
        let var = self
            .latency_history
            .iter()
            .map(|v| (v - mean) * (v - mean))
            .sum::<f64>()
            / self.latency_history.len() as f64;
        var.sqrt()
    }

    // -- Throughput ---------------------------------------------------------

    pub fn record_data_samples(&mut self, count: u64) {
        self.data_samples_received += count;
        self.throughput_window_data += count;
    }

    pub fn record_audio_samples(&mut self, count: u64) {
        self.audio_samples_output += count;
        self.throughput_window_audio += count;
    }

    /// Data throughput over the current window in samples/sec.
    pub fn data_throughput(&self) -> f64 {
        let elapsed = self.throughput_window_start.elapsed().as_secs_f64();
        if elapsed < 1e-9 {
            0.0
        } else {
            self.throughput_window_data as f64 / elapsed
        }
    }

    /// Audio output throughput over the current window in samples/sec.
    pub fn audio_throughput(&self) -> f64 {
        let elapsed = self.throughput_window_start.elapsed().as_secs_f64();
        if elapsed < 1e-9 {
            0.0
        } else {
            self.throughput_window_audio as f64 / elapsed
        }
    }

    /// Reset the throughput measurement window.
    pub fn reset_throughput_window(&mut self) {
        self.throughput_window_start = Instant::now();
        self.throughput_window_data = 0;
        self.throughput_window_audio = 0;
    }

    pub fn total_data_samples(&self) -> u64 {
        self.data_samples_received
    }

    pub fn total_audio_samples(&self) -> u64 {
        self.audio_samples_output
    }

    // -- Audio quality ------------------------------------------------------

    /// Analyse an audio buffer for clipping and silence.
    pub fn analyse_audio(&mut self, samples: &[f32]) {
        self.total_frames += samples.len() as u64;
        let mut is_silent = true;
        for &s in samples {
            if s.abs() > self.clip_threshold {
                self.clip_count += 1;
            }
            if s.abs() > self.silence_threshold {
                is_silent = false;
            }
        }
        if is_silent && !samples.is_empty() {
            self.silence_frames += samples.len() as u64;
        }
    }

    pub fn clip_count(&self) -> u64 {
        self.clip_count
    }

    pub fn silence_ratio(&self) -> f64 {
        if self.total_frames == 0 {
            0.0
        } else {
            self.silence_frames as f64 / self.total_frames as f64
        }
    }

    pub fn set_silence_threshold(&mut self, t: f32) {
        self.silence_threshold = t.abs();
    }

    pub fn set_clip_threshold(&mut self, t: f32) {
        self.clip_threshold = t.abs();
    }

    // -- Reset --------------------------------------------------------------

    pub fn reset(&mut self) {
        self.underrun_count = 0;
        self.overrun_count = 0;
        self.latency_history.clear();
        self.min_latency = f64::INFINITY;
        self.max_latency = 0.0;
        self.latency_sum = 0.0;
        self.latency_count = 0;
        self.data_samples_received = 0;
        self.audio_samples_output = 0;
        self.throughput_window_start = Instant::now();
        self.throughput_window_data = 0;
        self.throughput_window_audio = 0;
        self.clip_count = 0;
        self.silence_frames = 0;
        self.total_frames = 0;
    }
}

impl Default for StreamMonitor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// HealthStatus
// ---------------------------------------------------------------------------

/// Overall health status of the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Critical,
}

// ---------------------------------------------------------------------------
// StageHealth
// ---------------------------------------------------------------------------

/// Health snapshot for a single pipeline stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageHealth {
    pub name: String,
    pub status: HealthStatus,
    pub latency_seconds: f64,
    pub message: String,
}

// ---------------------------------------------------------------------------
// HealthCheck
// ---------------------------------------------------------------------------

/// Periodic health check of the pipeline and per-stage monitoring.
#[derive(Debug, Clone)]
pub struct HealthCheck {
    stages: Vec<StageHealth>,
    overall: HealthStatus,
    latency_warning_threshold: f64,
    latency_critical_threshold: f64,
    underrun_warning_threshold: u64,
    underrun_critical_threshold: u64,
    alerts: Vec<String>,
}

impl HealthCheck {
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            overall: HealthStatus::Healthy,
            latency_warning_threshold: 0.010,
            latency_critical_threshold: 0.050,
            underrun_warning_threshold: 5,
            underrun_critical_threshold: 50,
            alerts: Vec::new(),
        }
    }

    pub fn set_latency_thresholds(&mut self, warning: f64, critical: f64) {
        self.latency_warning_threshold = warning;
        self.latency_critical_threshold = critical;
    }

    pub fn set_underrun_thresholds(&mut self, warning: u64, critical: u64) {
        self.underrun_warning_threshold = warning;
        self.underrun_critical_threshold = critical;
    }

    /// Evaluate the health of the pipeline given a monitor snapshot.
    pub fn evaluate(&mut self, monitor: &StreamMonitor) {
        self.alerts.clear();
        self.stages.clear();
        let mut worst = HealthStatus::Healthy;

        // Latency check
        let avg_lat = monitor.average_latency();
        let lat_status = if avg_lat > self.latency_critical_threshold {
            self.alerts.push(format!(
                "CRITICAL: average latency {:.3}s exceeds threshold",
                avg_lat
            ));
            HealthStatus::Critical
        } else if avg_lat > self.latency_warning_threshold {
            self.alerts.push(format!(
                "WARNING: average latency {:.3}s elevated",
                avg_lat
            ));
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };
        self.stages.push(StageHealth {
            name: "latency".into(),
            status: lat_status,
            latency_seconds: avg_lat,
            message: format!("avg={:.4}s jitter={:.4}s", avg_lat, monitor.latency_jitter()),
        });
        worst = worst_status(worst, lat_status);

        // Underrun check
        let ur = monitor.underrun_count();
        let ur_status = if ur > self.underrun_critical_threshold {
            self.alerts
                .push(format!("CRITICAL: {} buffer underruns", ur));
            HealthStatus::Critical
        } else if ur > self.underrun_warning_threshold {
            self.alerts
                .push(format!("WARNING: {} buffer underruns", ur));
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };
        self.stages.push(StageHealth {
            name: "buffer".into(),
            status: ur_status,
            latency_seconds: 0.0,
            message: format!("underruns={} overruns={}", ur, monitor.overrun_count()),
        });
        worst = worst_status(worst, ur_status);

        // Audio quality check
        let clip = monitor.clip_count();
        let aq_status = if clip > 100 {
            self.alerts
                .push(format!("WARNING: {} clipped samples", clip));
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };
        self.stages.push(StageHealth {
            name: "audio-quality".into(),
            status: aq_status,
            latency_seconds: 0.0,
            message: format!(
                "clips={} silence={:.1}%",
                clip,
                monitor.silence_ratio() * 100.0
            ),
        });
        worst = worst_status(worst, aq_status);

        self.overall = worst;
    }

    pub fn overall_status(&self) -> HealthStatus {
        self.overall
    }

    pub fn stages(&self) -> &[StageHealth] {
        &self.stages
    }

    pub fn alerts(&self) -> &[String] {
        &self.alerts
    }

    pub fn is_healthy(&self) -> bool {
        self.overall == HealthStatus::Healthy
    }
}

impl Default for HealthCheck {
    fn default() -> Self {
        Self::new()
    }
}

fn worst_status(a: HealthStatus, b: HealthStatus) -> HealthStatus {
    match (a, b) {
        (HealthStatus::Critical, _) | (_, HealthStatus::Critical) => HealthStatus::Critical,
        (HealthStatus::Degraded, _) | (_, HealthStatus::Degraded) => HealthStatus::Degraded,
        _ => HealthStatus::Healthy,
    }
}

// ---------------------------------------------------------------------------
// PerformanceSnapshot
// ---------------------------------------------------------------------------

/// A timestamped snapshot of performance metrics for logging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    pub timestamp: String,
    pub health: HealthStatus,
    pub avg_latency: f64,
    pub max_latency: f64,
    pub latency_jitter: f64,
    pub underruns: u64,
    pub overruns: u64,
    pub clip_count: u64,
    pub silence_ratio: f64,
    pub data_throughput: f64,
    pub audio_throughput: f64,
}

// ---------------------------------------------------------------------------
// PerformanceLog
// ---------------------------------------------------------------------------

/// Logs performance events and periodic snapshots, with JSON export.
#[derive(Debug, Clone)]
pub struct PerformanceLog {
    snapshots: Vec<PerformanceSnapshot>,
    events: Vec<PerformanceEvent>,
    max_snapshots: usize,
    max_events: usize,
}

/// A logged performance event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceEvent {
    pub timestamp: String,
    pub severity: EventSeverity,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventSeverity {
    Info,
    Warning,
    Error,
}

impl PerformanceLog {
    pub fn new(max_snapshots: usize, max_events: usize) -> Self {
        Self {
            snapshots: Vec::new(),
            events: Vec::new(),
            max_snapshots,
            max_events,
        }
    }

    /// Record a periodic snapshot from the monitor and health check.
    pub fn record_snapshot(&mut self, monitor: &StreamMonitor, health: &HealthCheck) {
        if self.snapshots.len() >= self.max_snapshots {
            self.snapshots.remove(0);
        }
        let timestamp = format!("{:?}", SystemTime::now());
        self.snapshots.push(PerformanceSnapshot {
            timestamp,
            health: health.overall_status(),
            avg_latency: monitor.average_latency(),
            max_latency: monitor.max_latency(),
            latency_jitter: monitor.latency_jitter(),
            underruns: monitor.underrun_count(),
            overruns: monitor.overrun_count(),
            clip_count: monitor.clip_count(),
            silence_ratio: monitor.silence_ratio(),
            data_throughput: monitor.data_throughput(),
            audio_throughput: monitor.audio_throughput(),
        });
    }

    /// Log an event.
    pub fn log_event(&mut self, severity: EventSeverity, message: impl Into<String>) {
        if self.events.len() >= self.max_events {
            self.events.remove(0);
        }
        self.events.push(PerformanceEvent {
            timestamp: format!("{:?}", SystemTime::now()),
            severity,
            message: message.into(),
        });
    }

    pub fn snapshots(&self) -> &[PerformanceSnapshot] {
        &self.snapshots
    }

    pub fn events(&self) -> &[PerformanceEvent] {
        &self.events
    }

    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }

    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Export all snapshots to JSON.
    pub fn export_snapshots_json(&self) -> String {
        serde_json::to_string_pretty(&self.snapshots).unwrap_or_else(|_| "[]".to_string())
    }

    /// Export all events to JSON.
    pub fn export_events_json(&self) -> String {
        serde_json::to_string_pretty(&self.events).unwrap_or_else(|_| "[]".to_string())
    }

    pub fn clear(&mut self) {
        self.snapshots.clear();
        self.events.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monitor_underrun_overrun() {
        let mut m = StreamMonitor::new();
        m.record_underrun();
        m.record_underrun();
        m.record_overrun();
        assert_eq!(m.underrun_count(), 2);
        assert_eq!(m.overrun_count(), 1);
    }

    #[test]
    fn monitor_latency_stats() {
        let mut m = StreamMonitor::new();
        m.record_latency(0.005);
        m.record_latency(0.010);
        m.record_latency(0.003);
        assert!((m.average_latency() - 0.006).abs() < 0.001);
        assert!((m.min_latency() - 0.003).abs() < 0.001);
        assert!((m.max_latency() - 0.010).abs() < 0.001);
    }

    #[test]
    fn monitor_latency_jitter() {
        let mut m = StreamMonitor::new();
        m.record_latency(0.005);
        m.record_latency(0.015);
        assert!(m.latency_jitter() > 0.0);
    }

    #[test]
    fn monitor_throughput() {
        let mut m = StreamMonitor::new();
        m.record_data_samples(1000);
        m.record_audio_samples(44100);
        assert!(m.data_throughput() > 0.0);
        assert_eq!(m.total_data_samples(), 1000);
    }

    #[test]
    fn monitor_audio_quality_clipping() {
        let mut m = StreamMonitor::new();
        m.analyse_audio(&[0.5, 1.0, -1.0, 0.3]);
        assert!(m.clip_count() >= 2);
    }

    #[test]
    fn monitor_audio_quality_silence() {
        let mut m = StreamMonitor::new();
        m.analyse_audio(&[0.0, 0.0, 0.0]);
        assert!((m.silence_ratio() - 1.0).abs() < 0.01);
    }

    #[test]
    fn monitor_reset() {
        let mut m = StreamMonitor::new();
        m.record_underrun();
        m.record_latency(0.01);
        m.reset();
        assert_eq!(m.underrun_count(), 0);
        assert_eq!(m.average_latency(), 0.0);
    }

    #[test]
    fn health_check_healthy() {
        let mut hc = HealthCheck::new();
        let m = StreamMonitor::new();
        hc.evaluate(&m);
        assert!(hc.is_healthy());
        assert!(hc.alerts().is_empty());
    }

    #[test]
    fn health_check_degraded_latency() {
        let mut hc = HealthCheck::new();
        hc.set_latency_thresholds(0.005, 0.050);
        let mut m = StreamMonitor::new();
        m.record_latency(0.020);
        hc.evaluate(&m);
        assert_eq!(hc.overall_status(), HealthStatus::Degraded);
        assert!(!hc.alerts().is_empty());
    }

    #[test]
    fn health_check_critical_underruns() {
        let mut hc = HealthCheck::new();
        hc.set_underrun_thresholds(5, 10);
        let mut m = StreamMonitor::new();
        for _ in 0..20 {
            m.record_underrun();
        }
        hc.evaluate(&m);
        assert_eq!(hc.overall_status(), HealthStatus::Critical);
    }

    #[test]
    fn health_check_stages() {
        let mut hc = HealthCheck::new();
        let m = StreamMonitor::new();
        hc.evaluate(&m);
        assert!(hc.stages().len() >= 3);
    }

    #[test]
    fn perf_log_snapshot() {
        let mut log = PerformanceLog::new(100, 100);
        let m = StreamMonitor::new();
        let mut hc = HealthCheck::new();
        hc.evaluate(&m);
        log.record_snapshot(&m, &hc);
        assert_eq!(log.snapshot_count(), 1);
    }

    #[test]
    fn perf_log_event() {
        let mut log = PerformanceLog::new(100, 100);
        log.log_event(EventSeverity::Warning, "test warning");
        assert_eq!(log.event_count(), 1);
        assert_eq!(log.events()[0].severity, EventSeverity::Warning);
    }

    #[test]
    fn perf_log_export_json() {
        let mut log = PerformanceLog::new(100, 100);
        let m = StreamMonitor::new();
        let mut hc = HealthCheck::new();
        hc.evaluate(&m);
        log.record_snapshot(&m, &hc);
        let json = log.export_snapshots_json();
        assert!(json.contains("avg_latency"));
    }

    #[test]
    fn perf_log_clear() {
        let mut log = PerformanceLog::new(100, 100);
        log.log_event(EventSeverity::Info, "x");
        log.clear();
        assert_eq!(log.event_count(), 0);
    }

    #[test]
    fn perf_log_max_capacity() {
        let mut log = PerformanceLog::new(2, 2);
        log.log_event(EventSeverity::Info, "a");
        log.log_event(EventSeverity::Info, "b");
        log.log_event(EventSeverity::Info, "c");
        assert_eq!(log.event_count(), 2);
    }
}

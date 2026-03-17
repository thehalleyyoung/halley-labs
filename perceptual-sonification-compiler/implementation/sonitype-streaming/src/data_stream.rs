//! Streaming data input: push/pull data sources, multiplexing, and rate
//! adaptation for bridging arbitrary data rates to audio sample rates.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};

// ---------------------------------------------------------------------------
// DataValue (lightweight local enum to avoid coupling to core's variant)
// ---------------------------------------------------------------------------

/// A single data value in a stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StreamDataValue {
    Float(f64),
    Int(i64),
    Bool(bool),
    Text(String),
    Array(Vec<StreamDataValue>),
    Null,
}

impl StreamDataValue {
    /// Attempt to interpret the value as f64.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            Self::Int(v) => Some(*v as f64),
            Self::Bool(v) => Some(if *v { 1.0 } else { 0.0 }),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// WindowedStatistics
// ---------------------------------------------------------------------------

/// Running windowed statistics over the last N numeric values.
#[derive(Debug, Clone)]
pub struct WindowedStatistics {
    window: VecDeque<f64>,
    capacity: usize,
    sum: f64,
    sum_sq: f64,
    min: f64,
    max: f64,
}

impl WindowedStatistics {
    pub fn new(window_size: usize) -> Self {
        let cap = window_size.max(1);
        Self {
            window: VecDeque::with_capacity(cap),
            capacity: cap,
            sum: 0.0,
            sum_sq: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    pub fn push(&mut self, value: f64) {
        if self.window.len() == self.capacity {
            if let Some(old) = self.window.pop_front() {
                self.sum -= old;
                self.sum_sq -= old * old;
            }
        }
        self.window.push_back(value);
        self.sum += value;
        self.sum_sq += value * value;
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
    }

    pub fn count(&self) -> usize {
        self.window.len()
    }

    pub fn mean(&self) -> f64 {
        if self.window.is_empty() {
            0.0
        } else {
            self.sum / self.window.len() as f64
        }
    }

    pub fn variance(&self) -> f64 {
        let n = self.window.len() as f64;
        if n < 2.0 {
            return 0.0;
        }
        let mean = self.sum / n;
        (self.sum_sq / n) - (mean * mean)
    }

    pub fn stddev(&self) -> f64 {
        self.variance().max(0.0).sqrt()
    }

    pub fn min(&self) -> f64 {
        if self.window.is_empty() {
            0.0
        } else {
            *self.window.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
        }
    }

    pub fn max(&self) -> f64 {
        if self.window.is_empty() {
            0.0
        } else {
            *self.window.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
        }
    }

    pub fn reset(&mut self) {
        self.window.clear();
        self.sum = 0.0;
        self.sum_sq = 0.0;
        self.min = f64::INFINITY;
        self.max = f64::NEG_INFINITY;
    }
}

// ---------------------------------------------------------------------------
// ChangeDetector
// ---------------------------------------------------------------------------

/// Detects significant value changes above a configurable threshold.
#[derive(Debug, Clone)]
pub struct ChangeDetector {
    threshold: f64,
    last_value: Option<f64>,
    change_count: u64,
}

impl ChangeDetector {
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold: threshold.abs(),
            last_value: None,
            change_count: 0,
        }
    }

    /// Feed a new value. Returns `true` if the change from the previous value
    /// exceeds the threshold.
    pub fn detect(&mut self, value: f64) -> bool {
        let changed = match self.last_value {
            Some(prev) => (value - prev).abs() > self.threshold,
            None => true,
        };
        if changed {
            self.last_value = Some(value);
            self.change_count += 1;
        }
        changed
    }

    pub fn change_count(&self) -> u64 {
        self.change_count
    }

    pub fn last_value(&self) -> Option<f64> {
        self.last_value
    }

    pub fn reset(&mut self) {
        self.last_value = None;
        self.change_count = 0;
    }
}

// ---------------------------------------------------------------------------
// DataStream
// ---------------------------------------------------------------------------

/// A continuous data stream with push and pull interfaces, windowed statistics,
/// and change detection.
#[derive(Debug, Clone)]
pub struct DataStream {
    name: String,
    buffer: VecDeque<StreamDataValue>,
    buffer_capacity: usize,
    stats: WindowedStatistics,
    change_detector: ChangeDetector,
    total_pushed: u64,
    total_pulled: u64,
}

impl DataStream {
    pub fn new(name: impl Into<String>, buffer_capacity: usize, change_threshold: f64) -> Self {
        let cap = buffer_capacity.max(1);
        Self {
            name: name.into(),
            buffer: VecDeque::with_capacity(cap),
            buffer_capacity: cap,
            stats: WindowedStatistics::new(256),
            change_detector: ChangeDetector::new(change_threshold),
            total_pushed: 0,
            total_pulled: 0,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Push a value into the stream (producer side).
    pub fn push(&mut self, value: StreamDataValue) {
        if let Some(v) = value.as_f64() {
            self.stats.push(v);
            self.change_detector.detect(v);
        }
        if self.buffer.len() >= self.buffer_capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(value);
        self.total_pushed += 1;
    }

    /// Poll the next value (consumer side).
    pub fn poll(&mut self) -> Option<StreamDataValue> {
        let v = self.buffer.pop_front();
        if v.is_some() {
            self.total_pulled += 1;
        }
        v
    }

    /// Peek at the next value without consuming.
    pub fn peek(&self) -> Option<&StreamDataValue> {
        self.buffer.front()
    }

    pub fn available(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn statistics(&self) -> &WindowedStatistics {
        &self.stats
    }

    pub fn change_detector(&self) -> &ChangeDetector {
        &self.change_detector
    }

    pub fn total_pushed(&self) -> u64 {
        self.total_pushed
    }

    pub fn total_pulled(&self) -> u64 {
        self.total_pulled
    }

    /// Whether the last pushed value was a significant change.
    pub fn last_was_change(&self) -> bool {
        self.change_detector.change_count() > 0
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

// ---------------------------------------------------------------------------
// DataStreamMultiplexer
// ---------------------------------------------------------------------------

/// Multiplexes several `DataStream`s, providing synchronisation and time
/// alignment.
#[derive(Debug, Clone)]
pub struct DataStreamMultiplexer {
    streams: BTreeMap<String, DataStream>,
    sync_timestamp: f64,
    max_skew_seconds: f64,
}

/// A single synchronised snapshot from all streams.
#[derive(Debug, Clone)]
pub struct MultiplexedSample {
    pub timestamp: f64,
    pub values: BTreeMap<String, StreamDataValue>,
}

impl DataStreamMultiplexer {
    pub fn new(max_skew_seconds: f64) -> Self {
        Self {
            streams: BTreeMap::new(),
            sync_timestamp: 0.0,
            max_skew_seconds: max_skew_seconds.max(0.0),
        }
    }

    pub fn add_stream(&mut self, stream: DataStream) {
        self.streams.insert(stream.name().to_owned(), stream);
    }

    pub fn remove_stream(&mut self, name: &str) -> Option<DataStream> {
        self.streams.remove(name)
    }

    pub fn stream_names(&self) -> Vec<&str> {
        self.streams.keys().map(|s| s.as_str()).collect()
    }

    pub fn stream(&self, name: &str) -> Option<&DataStream> {
        self.streams.get(name)
    }

    pub fn stream_mut(&mut self, name: &str) -> Option<&mut DataStream> {
        self.streams.get_mut(name)
    }

    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }

    /// Push a value into a named stream.
    pub fn push(&mut self, stream_name: &str, value: StreamDataValue) {
        if let Some(s) = self.streams.get_mut(stream_name) {
            s.push(value);
        }
    }

    /// Poll a synchronised snapshot. Each stream contributes its next value
    /// (if available); missing values are filled with `Null`.
    pub fn poll_synchronized(&mut self, timestamp: f64) -> MultiplexedSample {
        self.sync_timestamp = timestamp;
        let mut values = BTreeMap::new();
        for (name, stream) in &mut self.streams {
            let v = stream.poll().unwrap_or(StreamDataValue::Null);
            values.insert(name.clone(), v);
        }
        MultiplexedSample {
            timestamp,
            values,
        }
    }

    /// Check whether all streams have at least one value ready.
    pub fn all_ready(&self) -> bool {
        self.streams.values().all(|s| !s.is_empty())
    }

    /// Return streams that are empty (late/missing data).
    pub fn missing_streams(&self) -> Vec<&str> {
        self.streams
            .iter()
            .filter(|(_, s)| s.is_empty())
            .map(|(n, _)| n.as_str())
            .collect()
    }

    pub fn sync_timestamp(&self) -> f64 {
        self.sync_timestamp
    }

    pub fn max_skew(&self) -> f64 {
        self.max_skew_seconds
    }
}

// ---------------------------------------------------------------------------
// InterpolationMethod
// ---------------------------------------------------------------------------

/// Interpolation method for rate adaptation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterpolationMethod {
    /// Nearest-neighbor (zero-order hold).
    NearestNeighbor,
    /// Linear interpolation between adjacent samples.
    Linear,
    /// Cubic Hermite interpolation.
    CubicHermite,
}

// ---------------------------------------------------------------------------
// DataRateAdapter
// ---------------------------------------------------------------------------

/// Adapts between mismatched data rates and audio sample rates through
/// interpolation (up-sampling) or decimation (down-sampling).
#[derive(Debug, Clone)]
pub struct DataRateAdapter {
    source_rate_hz: f64,
    target_rate_hz: f64,
    method: InterpolationMethod,
    history: VecDeque<f64>,
    phase: f64,
    output_buffer: VecDeque<f64>,
}

impl DataRateAdapter {
    pub fn new(source_rate_hz: f64, target_rate_hz: f64, method: InterpolationMethod) -> Self {
        Self {
            source_rate_hz: source_rate_hz.max(0.001),
            target_rate_hz: target_rate_hz.max(0.001),
            method,
            history: VecDeque::with_capacity(4),
            phase: 0.0,
            output_buffer: VecDeque::new(),
        }
    }

    pub fn source_rate(&self) -> f64 {
        self.source_rate_hz
    }

    pub fn target_rate(&self) -> f64 {
        self.target_rate_hz
    }

    pub fn ratio(&self) -> f64 {
        self.target_rate_hz / self.source_rate_hz
    }

    pub fn set_source_rate(&mut self, rate: f64) {
        self.source_rate_hz = rate.max(0.001);
    }

    pub fn set_target_rate(&mut self, rate: f64) {
        self.target_rate_hz = rate.max(0.001);
    }

    pub fn set_method(&mut self, method: InterpolationMethod) {
        self.method = method;
    }

    /// Feed one source sample. Internally produces zero or more output
    /// samples depending on the rate ratio.
    pub fn push(&mut self, value: f64) {
        self.history.push_back(value);
        if self.history.len() > 4 {
            self.history.pop_front();
        }

        let ratio = self.ratio();
        if ratio >= 1.0 {
            // Upsample: produce `ratio` output samples per input
            let count = ratio as usize;
            for i in 0..count {
                let frac = i as f64 / count as f64;
                let sample = self.interpolate(frac);
                self.output_buffer.push_back(sample);
            }
        } else {
            // Downsample: accumulate phase, emit when phase wraps
            self.phase += ratio;
            if self.phase >= 1.0 {
                self.phase -= 1.0;
                let sample = self.interpolate(0.0);
                self.output_buffer.push_back(sample);
            }
        }
    }

    /// Read the next output sample, if available.
    pub fn pop(&mut self) -> Option<f64> {
        self.output_buffer.pop_front()
    }

    /// Number of output samples ready.
    pub fn available(&self) -> usize {
        self.output_buffer.len()
    }

    /// Drain all available output samples.
    pub fn drain(&mut self) -> Vec<f64> {
        self.output_buffer.drain(..).collect()
    }

    fn interpolate(&self, frac: f64) -> f64 {
        match self.method {
            InterpolationMethod::NearestNeighbor => self.interp_nearest(),
            InterpolationMethod::Linear => self.interp_linear(frac),
            InterpolationMethod::CubicHermite => self.interp_cubic(frac),
        }
    }

    fn interp_nearest(&self) -> f64 {
        self.history.back().copied().unwrap_or(0.0)
    }

    fn interp_linear(&self, frac: f64) -> f64 {
        let n = self.history.len();
        if n < 2 {
            return self.history.back().copied().unwrap_or(0.0);
        }
        let a = self.history[n - 2];
        let b = self.history[n - 1];
        a + frac * (b - a)
    }

    fn interp_cubic(&self, t: f64) -> f64 {
        let n = self.history.len();
        if n < 4 {
            return self.interp_linear(t);
        }
        let p0 = self.history[n - 4];
        let p1 = self.history[n - 3];
        let p2 = self.history[n - 2];
        let p3 = self.history[n - 1];
        let t2 = t * t;
        let t3 = t2 * t;
        let a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
        let b = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
        let c = -0.5 * p0 + 0.5 * p2;
        let d = p1;
        a * t3 + b * t2 + c * t + d
    }

    pub fn reset(&mut self) {
        self.history.clear();
        self.output_buffer.clear();
        self.phase = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windowed_stats_mean_variance() {
        let mut ws = WindowedStatistics::new(4);
        ws.push(1.0);
        ws.push(2.0);
        ws.push(3.0);
        ws.push(4.0);
        assert!((ws.mean() - 2.5).abs() < 1e-9);
        assert!(ws.variance() > 0.0);
    }

    #[test]
    fn windowed_stats_window_rolls() {
        let mut ws = WindowedStatistics::new(3);
        ws.push(10.0);
        ws.push(20.0);
        ws.push(30.0);
        ws.push(40.0); // 10.0 should be evicted
        assert!((ws.mean() - 30.0).abs() < 1e-9);
    }

    #[test]
    fn windowed_stats_min_max() {
        let mut ws = WindowedStatistics::new(10);
        ws.push(5.0);
        ws.push(1.0);
        ws.push(9.0);
        assert!((ws.min() - 1.0).abs() < 1e-9);
        assert!((ws.max() - 9.0).abs() < 1e-9);
    }

    #[test]
    fn change_detector_threshold() {
        let mut cd = ChangeDetector::new(5.0);
        assert!(cd.detect(10.0)); // first value is always a change
        assert!(!cd.detect(12.0)); // only +2, below threshold
        assert!(cd.detect(20.0)); // +8, above threshold
        assert_eq!(cd.change_count(), 2);
    }

    #[test]
    fn data_stream_push_poll() {
        let mut ds = DataStream::new("test", 10, 1.0);
        ds.push(StreamDataValue::Float(1.0));
        ds.push(StreamDataValue::Float(2.0));
        assert_eq!(ds.available(), 2);
        assert_eq!(ds.poll(), Some(StreamDataValue::Float(1.0)));
        assert_eq!(ds.poll(), Some(StreamDataValue::Float(2.0)));
        assert!(ds.poll().is_none());
    }

    #[test]
    fn data_stream_overflow() {
        let mut ds = DataStream::new("test", 2, 0.0);
        ds.push(StreamDataValue::Int(1));
        ds.push(StreamDataValue::Int(2));
        ds.push(StreamDataValue::Int(3)); // evicts 1
        assert_eq!(ds.available(), 2);
        assert_eq!(ds.poll(), Some(StreamDataValue::Int(2)));
    }

    #[test]
    fn data_stream_statistics() {
        let mut ds = DataStream::new("s", 100, 0.0);
        for i in 0..10 {
            ds.push(StreamDataValue::Float(i as f64));
        }
        assert!((ds.statistics().mean() - 4.5).abs() < 1e-9);
    }

    #[test]
    fn multiplexer_sync_poll() {
        let mut mux = DataStreamMultiplexer::new(0.1);
        mux.add_stream(DataStream::new("a", 10, 0.0));
        mux.add_stream(DataStream::new("b", 10, 0.0));
        mux.push("a", StreamDataValue::Float(1.0));
        mux.push("b", StreamDataValue::Float(2.0));
        let snap = mux.poll_synchronized(0.0);
        assert_eq!(snap.values.len(), 2);
        assert_eq!(snap.values["a"], StreamDataValue::Float(1.0));
    }

    #[test]
    fn multiplexer_missing_streams() {
        let mut mux = DataStreamMultiplexer::new(0.1);
        mux.add_stream(DataStream::new("a", 10, 0.0));
        mux.add_stream(DataStream::new("b", 10, 0.0));
        mux.push("a", StreamDataValue::Float(1.0));
        let missing = mux.missing_streams();
        assert_eq!(missing, vec!["b"]);
    }

    #[test]
    fn rate_adapter_upsample() {
        let mut ra = DataRateAdapter::new(10.0, 40.0, InterpolationMethod::NearestNeighbor);
        ra.push(1.0);
        assert!(ra.available() >= 4);
    }

    #[test]
    fn rate_adapter_downsample() {
        let mut ra = DataRateAdapter::new(100.0, 10.0, InterpolationMethod::Linear);
        for i in 0..20 {
            ra.push(i as f64);
        }
        let output = ra.drain();
        assert!(output.len() < 20);
        assert!(!output.is_empty());
    }

    #[test]
    fn rate_adapter_linear_interp() {
        let mut ra = DataRateAdapter::new(1.0, 4.0, InterpolationMethod::Linear);
        ra.push(0.0);
        ra.push(10.0);
        let samples = ra.drain();
        assert!(samples.len() >= 4);
    }

    #[test]
    fn data_stream_peek() {
        let mut ds = DataStream::new("p", 10, 0.0);
        ds.push(StreamDataValue::Bool(true));
        assert_eq!(ds.peek(), Some(&StreamDataValue::Bool(true)));
        assert_eq!(ds.available(), 1);
    }

    #[test]
    fn rate_adapter_reset() {
        let mut ra = DataRateAdapter::new(10.0, 40.0, InterpolationMethod::NearestNeighbor);
        ra.push(5.0);
        ra.reset();
        assert_eq!(ra.available(), 0);
    }
}

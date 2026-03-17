//! Real-time scheduling utilities: priority management, callback timing
//! analysis, jitter measurement, CPU-load monitoring, and load balancing.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// CallbackTimer
// ---------------------------------------------------------------------------

/// Measures the duration of audio callbacks, tracking min/max/average and
/// detecting deadline misses.
#[derive(Debug, Clone)]
pub struct CallbackTimer {
    budget: Duration,
    history: VecDeque<Duration>,
    max_history: usize,
    min_duration: Duration,
    max_duration: Duration,
    total: Duration,
    count: u64,
    misses: u64,
    current_start: Option<Instant>,
}

impl CallbackTimer {
    /// Create a new timer with the given callback budget (deadline).
    pub fn new(budget: Duration, max_history: usize) -> Self {
        Self {
            budget,
            history: VecDeque::with_capacity(max_history),
            max_history,
            min_duration: Duration::MAX,
            max_duration: Duration::ZERO,
            total: Duration::ZERO,
            count: 0,
            misses: 0,
            current_start: None,
        }
    }

    pub fn budget(&self) -> Duration {
        self.budget
    }

    /// Mark the start of a callback.
    pub fn begin(&mut self) {
        self.current_start = Some(Instant::now());
    }

    /// Mark the end of a callback. Returns the elapsed duration and whether
    /// it was a deadline miss.
    pub fn end(&mut self) -> (Duration, bool) {
        let elapsed = match self.current_start.take() {
            Some(start) => start.elapsed(),
            None => Duration::ZERO,
        };
        self.record(elapsed);
        let miss = elapsed > self.budget;
        if miss {
            self.misses += 1;
        }
        (elapsed, miss)
    }

    /// Record a measured duration (for external timing).
    pub fn record(&mut self, d: Duration) {
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(d);
        if d < self.min_duration {
            self.min_duration = d;
        }
        if d > self.max_duration {
            self.max_duration = d;
        }
        self.total += d;
        self.count += 1;
    }

    pub fn min_duration(&self) -> Duration {
        if self.count == 0 {
            Duration::ZERO
        } else {
            self.min_duration
        }
    }

    pub fn max_duration(&self) -> Duration {
        self.max_duration
    }

    pub fn average_duration(&self) -> Duration {
        if self.count == 0 {
            return Duration::ZERO;
        }
        self.total / self.count as u32
    }

    pub fn deadline_misses(&self) -> u64 {
        self.misses
    }

    pub fn total_callbacks(&self) -> u64 {
        self.count
    }

    pub fn miss_rate(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.misses as f64 / self.count as f64
        }
    }

    /// Jitter: standard deviation of recent callback durations.
    pub fn jitter(&self) -> Duration {
        if self.history.len() < 2 {
            return Duration::ZERO;
        }
        let mean = self
            .history
            .iter()
            .map(|d| d.as_secs_f64())
            .sum::<f64>()
            / self.history.len() as f64;
        let var = self
            .history
            .iter()
            .map(|d| {
                let diff = d.as_secs_f64() - mean;
                diff * diff
            })
            .sum::<f64>()
            / self.history.len() as f64;
        Duration::from_secs_f64(var.sqrt())
    }

    pub fn reset(&mut self) {
        self.history.clear();
        self.min_duration = Duration::MAX;
        self.max_duration = Duration::ZERO;
        self.total = Duration::ZERO;
        self.count = 0;
        self.misses = 0;
        self.current_start = None;
    }
}

// ---------------------------------------------------------------------------
// CpuLoadEstimator
// ---------------------------------------------------------------------------

/// Estimates CPU load as the ratio of processing time to real time budget.
#[derive(Debug, Clone)]
struct CpuLoadEstimator {
    budget: Duration,
    load_history: VecDeque<f64>,
    max_history: usize,
}

impl CpuLoadEstimator {
    fn new(budget: Duration, max_history: usize) -> Self {
        Self {
            budget,
            load_history: VecDeque::with_capacity(max_history),
            max_history,
        }
    }

    fn record(&mut self, processing_time: Duration) {
        let load = processing_time.as_secs_f64() / self.budget.as_secs_f64();
        if self.load_history.len() >= self.max_history {
            self.load_history.pop_front();
        }
        self.load_history.push_back(load);
    }

    fn current_load(&self) -> f64 {
        self.load_history.back().copied().unwrap_or(0.0)
    }

    fn average_load(&self) -> f64 {
        if self.load_history.is_empty() {
            return 0.0;
        }
        self.load_history.iter().sum::<f64>() / self.load_history.len() as f64
    }

    fn peak_load(&self) -> f64 {
        self.load_history
            .iter()
            .cloned()
            .fold(0.0f64, f64::max)
    }
}

// ---------------------------------------------------------------------------
// RealtimeScheduler
// ---------------------------------------------------------------------------

/// Priority management and timing analysis for the audio processing thread.
pub struct RealtimeScheduler {
    callback_timer: CallbackTimer,
    cpu_estimator: CpuLoadEstimator,
    priority: SchedulerPriority,
    is_audio_thread: bool,
}

/// Priority level hint for the real-time thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchedulerPriority {
    Normal,
    High,
    Realtime,
}

impl RealtimeScheduler {
    pub fn new(callback_budget: Duration, priority: SchedulerPriority) -> Self {
        Self {
            callback_timer: CallbackTimer::new(callback_budget, 512),
            cpu_estimator: CpuLoadEstimator::new(callback_budget, 512),
            priority,
            is_audio_thread: false,
        }
    }

    pub fn priority(&self) -> SchedulerPriority {
        self.priority
    }

    pub fn set_priority(&mut self, p: SchedulerPriority) {
        self.priority = p;
    }

    /// Mark the beginning of an audio callback.
    pub fn begin_callback(&mut self) {
        self.callback_timer.begin();
    }

    /// Mark the end of an audio callback. Returns `true` if the deadline was
    /// missed.
    pub fn end_callback(&mut self) -> bool {
        let (elapsed, miss) = self.callback_timer.end();
        self.cpu_estimator.record(elapsed);
        miss
    }

    pub fn callback_timer(&self) -> &CallbackTimer {
        &self.callback_timer
    }

    pub fn current_cpu_load(&self) -> f64 {
        self.cpu_estimator.current_load()
    }

    pub fn average_cpu_load(&self) -> f64 {
        self.cpu_estimator.average_load()
    }

    pub fn peak_cpu_load(&self) -> f64 {
        self.cpu_estimator.peak_load()
    }

    pub fn jitter(&self) -> Duration {
        self.callback_timer.jitter()
    }

    pub fn reset(&mut self) {
        self.callback_timer.reset();
    }

    /// Report a summary of the current scheduling health.
    pub fn report(&self) -> SchedulerReport {
        SchedulerReport {
            priority: self.priority,
            total_callbacks: self.callback_timer.total_callbacks(),
            deadline_misses: self.callback_timer.deadline_misses(),
            miss_rate: self.callback_timer.miss_rate(),
            avg_duration: self.callback_timer.average_duration(),
            max_duration: self.callback_timer.max_duration(),
            jitter: self.callback_timer.jitter(),
            current_cpu_load: self.cpu_estimator.current_load(),
            average_cpu_load: self.cpu_estimator.average_load(),
            peak_cpu_load: self.cpu_estimator.peak_load(),
        }
    }
}

/// Summary report from the scheduler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerReport {
    pub priority: SchedulerPriority,
    pub total_callbacks: u64,
    pub deadline_misses: u64,
    pub miss_rate: f64,
    #[serde(with = "duration_serde")]
    pub avg_duration: Duration,
    #[serde(with = "duration_serde")]
    pub max_duration: Duration,
    #[serde(with = "duration_serde")]
    pub jitter: Duration,
    pub current_cpu_load: f64,
    pub average_cpu_load: f64,
    pub peak_cpu_load: f64,
}

mod duration_serde {
    use serde::{self, Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(d: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_f64(d.as_secs_f64())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = f64::deserialize(deserializer)?;
        Ok(Duration::from_secs_f64(secs.max(0.0)))
    }
}

// ---------------------------------------------------------------------------
// LoadBalancer
// ---------------------------------------------------------------------------

/// Distributes processing across logical processing slots (streams) and
/// adjusts quality when CPU load is high.
pub struct LoadBalancer {
    slots: Vec<ProcessingSlot>,
    quality_level: QualityLevel,
    high_load_threshold: f64,
    critical_load_threshold: f64,
}

/// Priority and load info for a single processing slot.
#[derive(Debug, Clone)]
pub struct ProcessingSlot {
    pub name: String,
    pub priority: u8,
    pub enabled: bool,
    pub last_processing_time: Duration,
}

/// Quality levels for graceful degradation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityLevel {
    Full,
    Reduced,
    Minimal,
}

impl LoadBalancer {
    pub fn new(high_load_threshold: f64, critical_load_threshold: f64) -> Self {
        Self {
            slots: Vec::new(),
            quality_level: QualityLevel::Full,
            high_load_threshold: high_load_threshold.clamp(0.0, 1.0),
            critical_load_threshold: critical_load_threshold.clamp(0.0, 1.0),
        }
    }

    pub fn add_slot(&mut self, name: impl Into<String>, priority: u8) {
        self.slots.push(ProcessingSlot {
            name: name.into(),
            priority,
            enabled: true,
            last_processing_time: Duration::ZERO,
        });
        self.slots.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    pub fn enabled_slots(&self) -> Vec<&ProcessingSlot> {
        self.slots.iter().filter(|s| s.enabled).collect()
    }

    pub fn quality_level(&self) -> QualityLevel {
        self.quality_level
    }

    /// Update quality level based on current CPU load.
    pub fn update_load(&mut self, cpu_load: f64) {
        if cpu_load >= self.critical_load_threshold {
            self.quality_level = QualityLevel::Minimal;
            // Disable lowest-priority slots
            let len = self.slots.len();
            for (i, slot) in self.slots.iter_mut().enumerate() {
                slot.enabled = i < len / 2 + 1;
            }
        } else if cpu_load >= self.high_load_threshold {
            self.quality_level = QualityLevel::Reduced;
            // Keep most slots enabled, disable bottom 25%
            let len = self.slots.len();
            let keep = (len as f64 * 0.75).ceil() as usize;
            for (i, slot) in self.slots.iter_mut().enumerate() {
                slot.enabled = i < keep;
            }
        } else {
            self.quality_level = QualityLevel::Full;
            for slot in &mut self.slots {
                slot.enabled = true;
            }
        }
    }

    /// Record processing time for a slot.
    pub fn record_slot_time(&mut self, name: &str, time: Duration) {
        if let Some(slot) = self.slots.iter_mut().find(|s| s.name == name) {
            slot.last_processing_time = time;
        }
    }

    /// Get the ordered list of slot names that should be processed this cycle.
    pub fn processing_order(&self) -> Vec<&str> {
        self.slots
            .iter()
            .filter(|s| s.enabled)
            .map(|s| s.name.as_str())
            .collect()
    }

    pub fn reset(&mut self) {
        self.quality_level = QualityLevel::Full;
        for slot in &mut self.slots {
            slot.enabled = true;
            slot.last_processing_time = Duration::ZERO;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn callback_timer_record() {
        let mut ct = CallbackTimer::new(Duration::from_millis(10), 100);
        ct.record(Duration::from_millis(5));
        ct.record(Duration::from_millis(8));
        assert_eq!(ct.total_callbacks(), 2);
        assert!(ct.average_duration() > Duration::ZERO);
    }

    #[test]
    fn callback_timer_deadline_miss() {
        let mut ct = CallbackTimer::new(Duration::from_millis(1), 100);
        ct.record(Duration::from_millis(2));
        ct.begin();
        thread::sleep(Duration::from_millis(5));
        let (_, miss) = ct.end();
        assert!(miss);
        assert!(ct.deadline_misses() >= 1);
    }

    #[test]
    fn callback_timer_jitter() {
        let mut ct = CallbackTimer::new(Duration::from_millis(10), 100);
        ct.record(Duration::from_millis(3));
        ct.record(Duration::from_millis(7));
        ct.record(Duration::from_millis(5));
        assert!(ct.jitter() > Duration::ZERO);
    }

    #[test]
    fn callback_timer_reset() {
        let mut ct = CallbackTimer::new(Duration::from_millis(10), 100);
        ct.record(Duration::from_millis(5));
        ct.reset();
        assert_eq!(ct.total_callbacks(), 0);
    }

    #[test]
    fn callback_timer_miss_rate() {
        let mut ct = CallbackTimer::new(Duration::from_millis(5), 100);
        ct.record(Duration::from_millis(3));
        ct.record(Duration::from_millis(3));
        assert!((ct.miss_rate() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn scheduler_begin_end() {
        let mut sched =
            RealtimeScheduler::new(Duration::from_millis(10), SchedulerPriority::High);
        sched.begin_callback();
        thread::sleep(Duration::from_millis(1));
        let miss = sched.end_callback();
        assert!(!miss);
        assert!(sched.current_cpu_load() > 0.0);
    }

    #[test]
    fn scheduler_report() {
        let mut sched =
            RealtimeScheduler::new(Duration::from_millis(10), SchedulerPriority::Realtime);
        sched.begin_callback();
        let _ = sched.end_callback();
        let report = sched.report();
        assert_eq!(report.total_callbacks, 1);
        assert_eq!(report.priority, SchedulerPriority::Realtime);
    }

    #[test]
    fn load_balancer_full_quality() {
        let mut lb = LoadBalancer::new(0.7, 0.9);
        lb.add_slot("a", 10);
        lb.add_slot("b", 5);
        lb.update_load(0.3);
        assert_eq!(lb.quality_level(), QualityLevel::Full);
        assert_eq!(lb.enabled_slots().len(), 2);
    }

    #[test]
    fn load_balancer_reduced() {
        let mut lb = LoadBalancer::new(0.7, 0.9);
        lb.add_slot("a", 10);
        lb.add_slot("b", 5);
        lb.add_slot("c", 1);
        lb.add_slot("d", 0);
        lb.update_load(0.8);
        assert_eq!(lb.quality_level(), QualityLevel::Reduced);
        assert!(lb.enabled_slots().len() < 4);
    }

    #[test]
    fn load_balancer_minimal() {
        let mut lb = LoadBalancer::new(0.7, 0.9);
        lb.add_slot("a", 10);
        lb.add_slot("b", 5);
        lb.add_slot("c", 1);
        lb.add_slot("d", 0);
        lb.update_load(0.95);
        assert_eq!(lb.quality_level(), QualityLevel::Minimal);
    }

    #[test]
    fn load_balancer_processing_order() {
        let mut lb = LoadBalancer::new(0.7, 0.9);
        lb.add_slot("lo", 1);
        lb.add_slot("hi", 10);
        let order = lb.processing_order();
        assert_eq!(order[0], "hi");
    }

    #[test]
    fn load_balancer_record_time() {
        let mut lb = LoadBalancer::new(0.7, 0.9);
        lb.add_slot("s", 5);
        lb.record_slot_time("s", Duration::from_millis(3));
        assert_eq!(
            lb.slots[0].last_processing_time,
            Duration::from_millis(3)
        );
    }

    #[test]
    fn load_balancer_reset() {
        let mut lb = LoadBalancer::new(0.7, 0.9);
        lb.add_slot("a", 10);
        lb.update_load(0.95);
        lb.reset();
        assert_eq!(lb.quality_level(), QualityLevel::Full);
    }
}

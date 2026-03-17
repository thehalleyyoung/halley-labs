//! # sim-monitor
//!
//! Real-time conservation law monitoring for running simulations.
//!
//! This crate provides [`Monitor`] — a configurable runtime observer that
//! evaluates conservation laws after every *N* time-steps and emits
//! [`MonitorEvent`] diagnostics when invariant drift exceeds a threshold.
//!
//! ## Quick start
//!
//! ```rust,ignore
//! use sim_monitor::{Monitor, MonitorConfig};
//!
//! let config = MonitorConfig::default();
//! let mut monitor = Monitor::new(config);
//! // feed simulation states …
//! monitor.observe(&state);
//! for event in monitor.drain_events() {
//!     eprintln!("{event:?}");
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Configuration for the runtime monitor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorConfig {
    /// How many time-steps between conservation checks.
    pub check_interval: usize,
    /// Relative tolerance for declaring a violation.
    pub relative_tolerance: f64,
    /// Absolute tolerance for declaring a violation.
    pub absolute_tolerance: f64,
    /// Maximum number of events retained in the ring buffer.
    pub max_events: usize,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            check_interval: 1,
            relative_tolerance: 1e-6,
            absolute_tolerance: 1e-12,
            max_events: 1024,
        }
    }
}

/// A conservation-law monitoring event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorEvent {
    /// Simulation time at which the event was raised.
    pub time: f64,
    /// Step number.
    pub step: usize,
    /// Name of the conservation law.
    pub law_name: String,
    /// Severity level.
    pub severity: Severity,
    /// Measured value of the conserved quantity.
    pub measured: f64,
    /// Expected (initial) value.
    pub expected: f64,
    /// Relative deviation.
    pub relative_error: f64,
}

/// Severity levels for monitor events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    /// Informational — drift is within tolerance but trending.
    Info,
    /// Warning — drift has exceeded the relative tolerance.
    Warning,
    /// Error — drift has exceeded the absolute tolerance.
    Error,
}

/// The runtime conservation monitor.
///
/// Feed simulation snapshots via [`Monitor::observe_values`] and retrieve
/// accumulated diagnostics with [`Monitor::drain_events`].
#[derive(Debug)]
pub struct Monitor {
    config: MonitorConfig,
    step: usize,
    initial_values: Option<Vec<(String, f64)>>,
    events: VecDeque<MonitorEvent>,
}

impl Monitor {
    /// Create a new monitor with the given configuration.
    pub fn new(config: MonitorConfig) -> Self {
        Self {
            config,
            step: 0,
            initial_values: None,
            events: VecDeque::new(),
        }
    }

    /// Observe a set of named conserved-quantity values at the current step.
    ///
    /// On the first call the values are recorded as the reference baseline.
    /// Subsequent calls compare against the baseline and emit events when
    /// tolerances are exceeded.
    pub fn observe_values(&mut self, time: f64, values: &[(String, f64)]) {
        self.step += 1;

        if self.initial_values.is_none() {
            self.initial_values = Some(values.to_vec());
            return;
        }

        if self.step % self.config.check_interval != 0 {
            return;
        }

        let initial = self.initial_values.as_ref().expect("initial values set above");

        for (init, current) in initial.iter().zip(values.iter()) {
            let (ref name, expected) = *init;
            let (_, measured) = *current;

            let abs_err = (measured - expected).abs();
            let rel_err = if expected.abs() > f64::EPSILON {
                abs_err / expected.abs()
            } else {
                abs_err
            };

            let severity = if abs_err > self.config.absolute_tolerance {
                Severity::Error
            } else if rel_err > self.config.relative_tolerance {
                Severity::Warning
            } else {
                continue; // within tolerance
            };

            let event = MonitorEvent {
                time,
                step: self.step,
                law_name: name.clone(),
                severity,
                measured,
                expected,
                relative_error: rel_err,
            };

            if self.events.len() >= self.config.max_events {
                self.events.pop_front();
            }
            self.events.push_back(event);
        }
    }

    /// Drain all accumulated events from the monitor.
    pub fn drain_events(&mut self) -> Vec<MonitorEvent> {
        self.events.drain(..).collect()
    }

    /// Return the number of steps observed so far.
    pub fn step_count(&self) -> usize {
        self.step
    }

    /// Return the number of pending events.
    pub fn pending_event_count(&self) -> usize {
        self.events.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitor_no_events_within_tolerance() {
        let mut m = Monitor::new(MonitorConfig::default());
        let vals = vec![("energy".to_string(), 100.0)];
        m.observe_values(0.0, &vals);
        m.observe_values(0.1, &vals);
        assert!(m.drain_events().is_empty());
    }

    #[test]
    fn test_monitor_detects_violation() {
        let mut m = Monitor::new(MonitorConfig {
            relative_tolerance: 1e-3,
            absolute_tolerance: 1.0,
            ..Default::default()
        });
        m.observe_values(0.0, &[("energy".to_string(), 100.0)]);
        m.observe_values(0.1, &[("energy".to_string(), 100.5)]);
        let events = m.drain_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].severity, Severity::Warning);
    }
}

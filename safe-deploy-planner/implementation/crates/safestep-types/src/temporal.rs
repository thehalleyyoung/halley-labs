// Temporal types for the SafeStep deployment planner.

use std::fmt;
use std::ops::{Add, Sub};

use chrono::Datelike;
use serde::{Deserialize, Serialize};

use crate::error::{Result, SafeStepError};

// ─── Timestamp ──────────────────────────────────────────────────────────

/// A point in time represented as milliseconds since the Unix epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Timestamp(pub i64);

impl Timestamp {
    /// Create a timestamp from epoch milliseconds.
    pub fn from_epoch_millis(millis: i64) -> Self {
        Self(millis)
    }

    /// Return the current time.
    pub fn now() -> Self {
        Self(chrono::Utc::now().timestamp_millis())
    }

    /// Return epoch milliseconds.
    pub fn as_epoch_millis(&self) -> i64 {
        self.0
    }

    /// Return epoch seconds (truncated).
    pub fn as_epoch_secs(&self) -> i64 {
        self.0 / 1000
    }

    /// Format as RFC 3339 string.
    pub fn to_rfc3339(&self) -> String {
        let dt = chrono::DateTime::from_timestamp(self.0 / 1000, ((self.0 % 1000) * 1_000_000) as u32);
        match dt {
            Some(d) => d.to_rfc3339(),
            None => format!("epoch({})", self.0),
        }
    }

    /// Parse from RFC 3339 string.
    pub fn from_rfc3339(s: &str) -> Result<Self> {
        let dt = chrono::DateTime::parse_from_rfc3339(s)
            .map_err(|e| SafeStepError::config(format!("invalid timestamp: {}", e)))?;
        Ok(Self(dt.timestamp_millis()))
    }

    /// Return the duration since another timestamp.
    pub fn duration_since(&self, earlier: Timestamp) -> Duration {
        let diff = self.0.saturating_sub(earlier.0);
        Duration::from_millis(diff.max(0) as u64)
    }

    /// Add a duration to this timestamp.
    pub fn add_duration(&self, d: Duration) -> Self {
        Self(self.0.saturating_add(d.millis as i64))
    }

    /// Subtract a duration from this timestamp.
    pub fn sub_duration(&self, d: Duration) -> Self {
        Self(self.0.saturating_sub(d.millis as i64))
    }

    /// Zero timestamp (epoch).
    pub fn epoch() -> Self {
        Self(0)
    }

    /// Check if this timestamp is the epoch.
    pub fn is_epoch(&self) -> bool {
        self.0 == 0
    }

    /// Min of two timestamps.
    pub fn min(self, other: Self) -> Self {
        if self.0 <= other.0 { self } else { other }
    }

    /// Max of two timestamps.
    pub fn max(self, other: Self) -> Self {
        if self.0 >= other.0 { self } else { other }
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_rfc3339())
    }
}

impl Add<Duration> for Timestamp {
    type Output = Timestamp;
    fn add(self, rhs: Duration) -> Self::Output {
        self.add_duration(rhs)
    }
}

impl Sub<Duration> for Timestamp {
    type Output = Timestamp;
    fn sub(self, rhs: Duration) -> Self::Output {
        self.sub_duration(rhs)
    }
}

impl Sub<Timestamp> for Timestamp {
    type Output = Duration;
    fn sub(self, rhs: Timestamp) -> Self::Output {
        self.duration_since(rhs)
    }
}

impl Default for Timestamp {
    fn default() -> Self {
        Self::epoch()
    }
}

// ─── Duration ───────────────────────────────────────────────────────────

/// A span of time in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Duration {
    pub millis: u64,
}

impl Duration {
    /// Zero duration.
    pub fn zero() -> Self {
        Self { millis: 0 }
    }

    /// Create from milliseconds.
    pub fn from_millis(millis: u64) -> Self {
        Self { millis }
    }

    /// Create from seconds.
    pub fn from_secs(secs: u64) -> Self {
        Self { millis: secs * 1000 }
    }

    /// Create from minutes.
    pub fn from_mins(mins: u64) -> Self {
        Self { millis: mins * 60_000 }
    }

    /// Create from hours.
    pub fn from_hours(hours: u64) -> Self {
        Self { millis: hours * 3_600_000 }
    }

    /// Return as milliseconds.
    pub fn as_millis(&self) -> u64 {
        self.millis
    }

    /// Return as seconds (truncated).
    pub fn as_secs(&self) -> u64 {
        self.millis / 1000
    }

    /// Return as fractional seconds.
    pub fn as_secs_f64(&self) -> f64 {
        self.millis as f64 / 1000.0
    }

    /// Return as minutes (truncated).
    pub fn as_mins(&self) -> u64 {
        self.millis / 60_000
    }

    /// Check if zero.
    pub fn is_zero(&self) -> bool {
        self.millis == 0
    }

    /// Saturating addition.
    pub fn saturating_add(&self, other: Duration) -> Duration {
        Duration { millis: self.millis.saturating_add(other.millis) }
    }

    /// Saturating subtraction.
    pub fn saturating_sub(&self, other: Duration) -> Duration {
        Duration { millis: self.millis.saturating_sub(other.millis) }
    }

    /// Convert to std::time::Duration.
    pub fn to_std(&self) -> std::time::Duration {
        std::time::Duration::from_millis(self.millis)
    }

    /// Convert from std::time::Duration.
    pub fn from_std(d: std::time::Duration) -> Self {
        Self { millis: d.as_millis() as u64 }
    }

    /// Min of two durations.
    pub fn min(self, other: Self) -> Self {
        if self.millis <= other.millis { self } else { other }
    }

    /// Max of two durations.
    pub fn max(self, other: Self) -> Self {
        if self.millis >= other.millis { self } else { other }
    }

    /// Multiply by a scalar.
    pub fn mul(self, factor: u64) -> Self {
        Self { millis: self.millis.saturating_mul(factor) }
    }
}

impl Add for Duration {
    type Output = Duration;
    fn add(self, rhs: Duration) -> Self::Output {
        self.saturating_add(rhs)
    }
}

impl Sub for Duration {
    type Output = Duration;
    fn sub(self, rhs: Duration) -> Self::Output {
        self.saturating_sub(rhs)
    }
}

impl fmt::Display for Duration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let secs = self.millis / 1000;
        let ms = self.millis % 1000;
        if secs >= 3600 {
            write!(f, "{}h{}m{}s", secs / 3600, (secs % 3600) / 60, secs % 60)
        } else if secs >= 60 {
            write!(f, "{}m{}s", secs / 60, secs % 60)
        } else if ms > 0 {
            write!(f, "{}.{:03}s", secs, ms)
        } else {
            write!(f, "{}s", secs)
        }
    }
}

impl Default for Duration {
    fn default() -> Self {
        Self::zero()
    }
}

// ─── TimeWindow ─────────────────────────────────────────────────────────

/// A time window defined by a start and end timestamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TimeWindow {
    pub start: Timestamp,
    pub end: Timestamp,
}

impl TimeWindow {
    /// Create a new time window.
    pub fn new(start: Timestamp, end: Timestamp) -> Result<Self> {
        if end.0 < start.0 {
            return Err(SafeStepError::config(
                "time window end must be >= start",
            ));
        }
        Ok(Self { start, end })
    }

    /// Unchecked constructor for known-valid windows.
    pub fn new_unchecked(start: Timestamp, end: Timestamp) -> Self {
        Self { start, end }
    }

    /// Check if a timestamp falls within this window (inclusive).
    pub fn contains(&self, ts: Timestamp) -> bool {
        ts.0 >= self.start.0 && ts.0 <= self.end.0
    }

    /// Check if two windows overlap.
    pub fn overlaps(&self, other: &TimeWindow) -> bool {
        self.start.0 <= other.end.0 && other.start.0 <= self.end.0
    }

    /// Return the duration of this window.
    pub fn duration(&self) -> Duration {
        self.end.duration_since(self.start)
    }

    /// Merge two overlapping or adjacent windows into one, or return None if disjoint.
    pub fn merge(&self, other: &TimeWindow) -> Option<TimeWindow> {
        if self.overlaps(other) || self.end.0 + 1 == other.start.0 || other.end.0 + 1 == self.start.0 {
            Some(TimeWindow {
                start: self.start.min(other.start),
                end: self.end.max(other.end),
            })
        } else {
            None
        }
    }

    /// Compute the intersection of two windows, if they overlap.
    pub fn intersection(&self, other: &TimeWindow) -> Option<TimeWindow> {
        if !self.overlaps(other) {
            return None;
        }
        Some(TimeWindow {
            start: self.start.max(other.start),
            end: self.end.min(other.end),
        })
    }

    /// Check if this window fully contains another.
    pub fn fully_contains(&self, other: &TimeWindow) -> bool {
        self.start.0 <= other.start.0 && self.end.0 >= other.end.0
    }

    /// Shift the window by a duration.
    pub fn shift(&self, d: Duration) -> Self {
        Self {
            start: self.start + d,
            end: self.end + d,
        }
    }

    /// Check if the window is empty (zero duration).
    pub fn is_empty(&self) -> bool {
        self.start.0 == self.end.0
    }

    /// Midpoint of the window.
    pub fn midpoint(&self) -> Timestamp {
        Timestamp((self.start.0 + self.end.0) / 2)
    }
}

impl fmt::Display for TimeWindow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{} .. {}]", self.start, self.end)
    }
}

// ─── Schedule ───────────────────────────────────────────────────────────

/// A schedule is a sequence of non-overlapping time windows during which
/// deployment actions are permitted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Schedule {
    windows: Vec<TimeWindow>,
}

impl Schedule {
    /// Create an empty schedule.
    pub fn new() -> Self {
        Self { windows: Vec::new() }
    }

    /// Create a schedule from a list of windows. Sorts and merges overlapping windows.
    pub fn from_windows(mut windows: Vec<TimeWindow>) -> Self {
        windows.sort_by_key(|w| w.start.0);
        let merged = Self::merge_sorted(windows);
        Self { windows: merged }
    }

    /// Add a window to the schedule.
    pub fn add_window(&mut self, w: TimeWindow) {
        self.windows.push(w);
        self.windows.sort_by_key(|w| w.start.0);
        self.windows = Self::merge_sorted(self.windows.clone());
    }

    fn merge_sorted(windows: Vec<TimeWindow>) -> Vec<TimeWindow> {
        let mut merged: Vec<TimeWindow> = Vec::new();
        for w in windows {
            if let Some(last) = merged.last_mut() {
                if let Some(m) = last.merge(&w) {
                    *last = m;
                    continue;
                }
            }
            merged.push(w);
        }
        merged
    }

    /// Check if a timestamp falls within any scheduled window.
    pub fn is_allowed(&self, ts: Timestamp) -> bool {
        self.windows.iter().any(|w| w.contains(ts))
    }

    /// Return the next window that starts at or after the given timestamp.
    pub fn next_window(&self, after: Timestamp) -> Option<&TimeWindow> {
        // If currently inside a window, return that window
        for w in &self.windows {
            if w.contains(after) {
                return Some(w);
            }
            if w.start.0 > after.0 {
                return Some(w);
            }
        }
        None
    }

    /// Return the time until the next allowed window.
    pub fn time_until_allowed(&self, from: Timestamp) -> Option<Duration> {
        if self.is_allowed(from) {
            return Some(Duration::zero());
        }
        self.next_window(from).map(|w| w.start.duration_since(from))
    }

    /// Total time covered by all windows.
    pub fn total_duration(&self) -> Duration {
        self.windows.iter().fold(Duration::zero(), |acc, w| acc + w.duration())
    }

    /// Number of windows.
    pub fn window_count(&self) -> usize {
        self.windows.len()
    }

    /// Check if the schedule is empty.
    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }

    /// Iterate over windows.
    pub fn windows(&self) -> &[TimeWindow] {
        &self.windows
    }

    /// Return windows that overlap a given window.
    pub fn overlapping_windows(&self, window: &TimeWindow) -> Vec<&TimeWindow> {
        self.windows.iter().filter(|w| w.overlaps(window)).collect()
    }

    /// Check if the schedule covers an entire time window.
    pub fn covers(&self, window: &TimeWindow) -> bool {
        let mut covered_until = window.start.0;
        for w in &self.windows {
            if w.start.0 > covered_until {
                return false;
            }
            if w.end.0 >= window.end.0 {
                return true;
            }
            covered_until = covered_until.max(w.end.0);
        }
        covered_until >= window.end.0
    }
}

impl Default for Schedule {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Schedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Schedule({} windows, total {})", self.windows.len(), self.total_duration())
    }
}

// ─── Deadline ───────────────────────────────────────────────────────────

/// A deadline for completing a deployment or phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Deadline {
    pub timestamp: Timestamp,
    pub is_hard: bool,
}

impl Deadline {
    /// Create a hard deadline.
    pub fn hard(ts: Timestamp) -> Self {
        Self { timestamp: ts, is_hard: true }
    }

    /// Create a soft deadline.
    pub fn soft(ts: Timestamp) -> Self {
        Self { timestamp: ts, is_hard: false }
    }

    /// Check if the deadline has expired.
    pub fn is_expired(&self, now: Timestamp) -> bool {
        now.0 >= self.timestamp.0
    }

    /// Return the remaining time until the deadline.
    pub fn remaining(&self, now: Timestamp) -> Duration {
        if now.0 >= self.timestamp.0 {
            Duration::zero()
        } else {
            self.timestamp.duration_since(now)
        }
    }

    /// Check if there's enough time remaining for a given duration.
    pub fn has_time_for(&self, now: Timestamp, needed: Duration) -> bool {
        self.remaining(now).millis >= needed.millis
    }

    /// Create a deadline from a duration from now.
    pub fn from_now(d: Duration) -> Self {
        Self {
            timestamp: Timestamp::now() + d,
            is_hard: true,
        }
    }

    /// Create a deadline from a duration after a given start.
    pub fn after(start: Timestamp, d: Duration) -> Self {
        Self {
            timestamp: start + d,
            is_hard: true,
        }
    }
}

impl fmt::Display for Deadline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = if self.is_hard { "hard" } else { "soft" };
        write!(f, "{} deadline at {}", kind, self.timestamp)
    }
}

// ─── Phase ──────────────────────────────────────────────────────────────

/// Deployment phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Phase {
    Preparation,
    Execution,
    Verification,
    Rollback,
    Complete,
}

impl Phase {
    /// All phases in order.
    pub fn all() -> &'static [Phase] {
        &[Phase::Preparation, Phase::Execution, Phase::Verification, Phase::Rollback, Phase::Complete]
    }

    /// Typical order index.
    pub fn order_index(&self) -> usize {
        match self {
            Phase::Preparation => 0,
            Phase::Execution => 1,
            Phase::Verification => 2,
            Phase::Rollback => 3,
            Phase::Complete => 4,
        }
    }

    /// Check if this is a terminal phase.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Phase::Complete | Phase::Rollback)
    }
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Phase::Preparation => write!(f, "Preparation"),
            Phase::Execution => write!(f, "Execution"),
            Phase::Verification => write!(f, "Verification"),
            Phase::Rollback => write!(f, "Rollback"),
            Phase::Complete => write!(f, "Complete"),
        }
    }
}

// ─── PhaseDuration ──────────────────────────────────────────────────────

/// Duration information for a deployment phase.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhaseDuration {
    pub phase: Phase,
    pub planned: Duration,
    pub actual: Option<Duration>,
}

impl PhaseDuration {
    /// Create a new phase duration with planned time.
    pub fn new(phase: Phase, planned: Duration) -> Self {
        Self { phase, planned, actual: None }
    }

    /// Set the actual duration.
    pub fn with_actual(mut self, actual: Duration) -> Self {
        self.actual = Some(actual);
        self
    }

    /// Return the variance between planned and actual (positive = over, negative = under).
    pub fn variance(&self) -> Option<i64> {
        self.actual.map(|a| a.millis as i64 - self.planned.millis as i64)
    }

    /// Return the ratio of actual / planned duration.
    pub fn ratio(&self) -> Option<f64> {
        self.actual.map(|a| {
            if self.planned.millis == 0 {
                if a.millis == 0 { 1.0 } else { f64::INFINITY }
            } else {
                a.millis as f64 / self.planned.millis as f64
            }
        })
    }

    /// Check if the actual duration exceeded the planned duration.
    pub fn is_overrun(&self) -> bool {
        self.variance().map_or(false, |v| v > 0)
    }
}

impl fmt::Display for PhaseDuration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.actual {
            Some(actual) => write!(f, "{}: planned={}, actual={}", self.phase, self.planned, actual),
            None => write!(f, "{}: planned={}", self.phase, self.planned),
        }
    }
}

// ─── DeploymentTiming ───────────────────────────────────────────────────

/// Complete timing information for a deployment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeploymentTiming {
    pub planned_start: Timestamp,
    pub planned_end: Timestamp,
    pub actual_start: Option<Timestamp>,
    pub actual_end: Option<Timestamp>,
    pub phase_durations: Vec<PhaseDuration>,
}

impl DeploymentTiming {
    /// Create new deployment timing with planned times.
    pub fn new(planned_start: Timestamp, planned_end: Timestamp) -> Self {
        Self {
            planned_start,
            planned_end,
            actual_start: None,
            actual_end: None,
            phase_durations: Vec::new(),
        }
    }

    /// Create from planned start and total duration.
    pub fn from_duration(start: Timestamp, duration: Duration) -> Self {
        Self::new(start, start + duration)
    }

    /// Set the actual start time.
    pub fn with_actual_start(mut self, ts: Timestamp) -> Self {
        self.actual_start = Some(ts);
        self
    }

    /// Set the actual end time.
    pub fn with_actual_end(mut self, ts: Timestamp) -> Self {
        self.actual_end = Some(ts);
        self
    }

    /// Add a phase duration entry.
    pub fn add_phase(&mut self, pd: PhaseDuration) {
        self.phase_durations.push(pd);
    }

    /// Planned total duration.
    pub fn planned_duration(&self) -> Duration {
        self.planned_end.duration_since(self.planned_start)
    }

    /// Actual total duration, if started and ended.
    pub fn actual_duration(&self) -> Option<Duration> {
        match (self.actual_start, self.actual_end) {
            (Some(s), Some(e)) => Some(e.duration_since(s)),
            _ => None,
        }
    }

    /// Is the deployment currently in progress?
    pub fn is_in_progress(&self) -> bool {
        self.actual_start.is_some() && self.actual_end.is_none()
    }

    /// Is the deployment complete?
    pub fn is_complete(&self) -> bool {
        self.actual_start.is_some() && self.actual_end.is_some()
    }

    /// Has the deployment started?
    pub fn has_started(&self) -> bool {
        self.actual_start.is_some()
    }

    /// Total overrun compared to planned duration.
    pub fn overrun(&self) -> Option<i64> {
        self.actual_duration().map(|a| {
            a.millis as i64 - self.planned_duration().millis as i64
        })
    }

    /// Get the phase duration for a specific phase.
    pub fn phase_duration(&self, phase: Phase) -> Option<&PhaseDuration> {
        self.phase_durations.iter().find(|pd| pd.phase == phase)
    }

    /// Sum of all planned phase durations.
    pub fn total_planned_phase_time(&self) -> Duration {
        self.phase_durations
            .iter()
            .fold(Duration::zero(), |acc, pd| acc + pd.planned)
    }

    /// Sum of all actual phase durations.
    pub fn total_actual_phase_time(&self) -> Option<Duration> {
        let mut total = Duration::zero();
        for pd in &self.phase_durations {
            match pd.actual {
                Some(a) => total = total + a,
                None => return None,
            }
        }
        Some(total)
    }

    /// Phases that have overrun their planned duration.
    pub fn overrun_phases(&self) -> Vec<&PhaseDuration> {
        self.phase_durations.iter().filter(|pd| pd.is_overrun()).collect()
    }

    /// Convert to a time window.
    pub fn as_planned_window(&self) -> TimeWindow {
        TimeWindow::new_unchecked(self.planned_start, self.planned_end)
    }

    /// Convert actual timing to a time window, if complete.
    pub fn as_actual_window(&self) -> Option<TimeWindow> {
        match (self.actual_start, self.actual_end) {
            (Some(s), Some(e)) => Some(TimeWindow::new_unchecked(s, e)),
            _ => None,
        }
    }
}

impl fmt::Display for DeploymentTiming {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DeploymentTiming(planned: {} -> {}", self.planned_start, self.planned_end)?;
        if let Some(s) = self.actual_start {
            write!(f, ", started: {}", s)?;
        }
        if let Some(e) = self.actual_end {
            write!(f, ", ended: {}", e)?;
        }
        write!(f, ", {} phases)", self.phase_durations.len())
    }
}

impl Default for DeploymentTiming {
    fn default() -> Self {
        Self::new(Timestamp::epoch(), Timestamp::epoch())
    }
}

// ─── Recurring schedule builder ─────────────────────────────────────────

/// Day of week for scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DayOfWeek {
    Monday,
    Tuesday,
    Wednesday,
    Thursday,
    Friday,
    Saturday,
    Sunday,
}

impl DayOfWeek {
    /// Convert from chrono weekday.
    pub fn from_chrono(wd: chrono::Weekday) -> Self {
        match wd {
            chrono::Weekday::Mon => DayOfWeek::Monday,
            chrono::Weekday::Tue => DayOfWeek::Tuesday,
            chrono::Weekday::Wed => DayOfWeek::Wednesday,
            chrono::Weekday::Thu => DayOfWeek::Thursday,
            chrono::Weekday::Fri => DayOfWeek::Friday,
            chrono::Weekday::Sat => DayOfWeek::Saturday,
            chrono::Weekday::Sun => DayOfWeek::Sunday,
        }
    }

    /// Convert to chrono weekday.
    pub fn to_chrono(&self) -> chrono::Weekday {
        match self {
            DayOfWeek::Monday => chrono::Weekday::Mon,
            DayOfWeek::Tuesday => chrono::Weekday::Tue,
            DayOfWeek::Wednesday => chrono::Weekday::Wed,
            DayOfWeek::Thursday => chrono::Weekday::Thu,
            DayOfWeek::Friday => chrono::Weekday::Fri,
            DayOfWeek::Saturday => chrono::Weekday::Sat,
            DayOfWeek::Sunday => chrono::Weekday::Sun,
        }
    }

    /// Is this a weekday (Mon-Fri)?
    pub fn is_weekday(&self) -> bool {
        !matches!(self, DayOfWeek::Saturday | DayOfWeek::Sunday)
    }
}

impl fmt::Display for DayOfWeek {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DayOfWeek::Monday => write!(f, "Monday"),
            DayOfWeek::Tuesday => write!(f, "Tuesday"),
            DayOfWeek::Wednesday => write!(f, "Wednesday"),
            DayOfWeek::Thursday => write!(f, "Thursday"),
            DayOfWeek::Friday => write!(f, "Friday"),
            DayOfWeek::Saturday => write!(f, "Saturday"),
            DayOfWeek::Sunday => write!(f, "Sunday"),
        }
    }
}

// ─── MaintenanceWindow ──────────────────────────────────────────────────

/// Defines a recurring maintenance window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaintenanceWindow {
    pub allowed_days: Vec<DayOfWeek>,
    pub start_hour: u8,
    pub start_minute: u8,
    pub duration: Duration,
    pub timezone_offset_hours: i8,
}

impl MaintenanceWindow {
    /// Create a new maintenance window.
    pub fn new(
        allowed_days: Vec<DayOfWeek>,
        start_hour: u8,
        start_minute: u8,
        duration: Duration,
    ) -> Result<Self> {
        if start_hour >= 24 {
            return Err(SafeStepError::config("start_hour must be < 24"));
        }
        if start_minute >= 60 {
            return Err(SafeStepError::config("start_minute must be < 60"));
        }
        Ok(Self {
            allowed_days,
            start_hour,
            start_minute,
            duration,
            timezone_offset_hours: 0,
        })
    }

    /// Set timezone offset.
    pub fn with_timezone_offset(mut self, hours: i8) -> Self {
        self.timezone_offset_hours = hours;
        self
    }

    /// Check if a timestamp falls within this maintenance window.
    pub fn is_active(&self, ts: Timestamp) -> bool {
        let offset_ms = self.timezone_offset_hours as i64 * 3_600_000;
        let adjusted = ts.0 + offset_ms;
        let dt = chrono::DateTime::from_timestamp(adjusted / 1000, ((adjusted % 1000).max(0) * 1_000_000) as u32);
        let dt = match dt {
            Some(d) => d,
            None => return false,
        };
        let wd = DayOfWeek::from_chrono(dt.weekday());
        if !self.allowed_days.contains(&wd) {
            return false;
        }
        let hour = dt.format("%H").to_string().parse::<u8>().unwrap_or(0);
        let minute = dt.format("%M").to_string().parse::<u8>().unwrap_or(0);
        let time_in_mins = hour as u64 * 60 + minute as u64;
        let start_mins = self.start_hour as u64 * 60 + self.start_minute as u64;
        let end_mins = start_mins + self.duration.as_mins();
        time_in_mins >= start_mins && time_in_mins < end_mins
    }
}

impl fmt::Display for MaintenanceWindow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let days: Vec<String> = self.allowed_days.iter().map(|d| format!("{}", d)).collect();
        write!(
            f,
            "MaintenanceWindow([{}] {:02}:{:02} for {})",
            days.join(", "),
            self.start_hour,
            self.start_minute,
            self.duration
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestamp_basic() {
        let ts = Timestamp::from_epoch_millis(1000);
        assert_eq!(ts.as_epoch_millis(), 1000);
        assert_eq!(ts.as_epoch_secs(), 1);
    }

    #[test]
    fn test_timestamp_epoch() {
        let ts = Timestamp::epoch();
        assert!(ts.is_epoch());
        assert_eq!(ts.as_epoch_millis(), 0);
    }

    #[test]
    fn test_timestamp_arithmetic() {
        let ts = Timestamp::from_epoch_millis(5000);
        let d = Duration::from_secs(2);
        let ts2 = ts + d;
        assert_eq!(ts2.as_epoch_millis(), 7000);
        let ts3 = ts2 - d;
        assert_eq!(ts3.as_epoch_millis(), 5000);
    }

    #[test]
    fn test_timestamp_sub_timestamps() {
        let a = Timestamp::from_epoch_millis(10000);
        let b = Timestamp::from_epoch_millis(3000);
        let d = a - b;
        assert_eq!(d.as_millis(), 7000);
    }

    #[test]
    fn test_timestamp_min_max() {
        let a = Timestamp::from_epoch_millis(100);
        let b = Timestamp::from_epoch_millis(200);
        assert_eq!(a.min(b), a);
        assert_eq!(a.max(b), b);
    }

    #[test]
    fn test_timestamp_rfc3339_roundtrip() {
        let ts = Timestamp::from_epoch_millis(1_700_000_000_000);
        let s = ts.to_rfc3339();
        let ts2 = Timestamp::from_rfc3339(&s).unwrap();
        // Allow for sub-second truncation
        assert!((ts.0 - ts2.0).abs() < 1000);
    }

    #[test]
    fn test_duration_basic() {
        let d = Duration::from_secs(90);
        assert_eq!(d.as_secs(), 90);
        assert_eq!(d.as_millis(), 90_000);
        assert_eq!(d.as_mins(), 1);
    }

    #[test]
    fn test_duration_zero() {
        let d = Duration::zero();
        assert!(d.is_zero());
        assert_eq!(d.as_millis(), 0);
    }

    #[test]
    fn test_duration_add_sub() {
        let a = Duration::from_secs(10);
        let b = Duration::from_secs(3);
        assert_eq!((a + b).as_secs(), 13);
        assert_eq!((a - b).as_secs(), 7);
    }

    #[test]
    fn test_duration_saturating_sub() {
        let a = Duration::from_secs(1);
        let b = Duration::from_secs(5);
        let diff = a - b;
        assert!(diff.is_zero());
    }

    #[test]
    fn test_duration_display() {
        assert_eq!(format!("{}", Duration::from_secs(5)), "5s");
        assert_eq!(format!("{}", Duration::from_secs(90)), "1m30s");
        assert_eq!(format!("{}", Duration::from_secs(3661)), "1h1m1s");
        assert_eq!(format!("{}", Duration::from_millis(1500)), "1.500s");
    }

    #[test]
    fn test_duration_from_hours() {
        let d = Duration::from_hours(2);
        assert_eq!(d.as_secs(), 7200);
        assert_eq!(d.as_mins(), 120);
    }

    #[test]
    fn test_duration_mul() {
        let d = Duration::from_secs(10);
        assert_eq!(d.mul(3).as_secs(), 30);
    }

    #[test]
    fn test_duration_to_std_roundtrip() {
        let d = Duration::from_millis(12345);
        let std = d.to_std();
        let d2 = Duration::from_std(std);
        assert_eq!(d, d2);
    }

    #[test]
    fn test_time_window_basic() {
        let w = TimeWindow::new(
            Timestamp::from_epoch_millis(1000),
            Timestamp::from_epoch_millis(5000),
        ).unwrap();
        assert!(w.contains(Timestamp::from_epoch_millis(3000)));
        assert!(!w.contains(Timestamp::from_epoch_millis(6000)));
        assert_eq!(w.duration().as_millis(), 4000);
    }

    #[test]
    fn test_time_window_invalid() {
        let result = TimeWindow::new(
            Timestamp::from_epoch_millis(5000),
            Timestamp::from_epoch_millis(1000),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_time_window_overlap() {
        let a = TimeWindow::new_unchecked(
            Timestamp::from_epoch_millis(1000),
            Timestamp::from_epoch_millis(5000),
        );
        let b = TimeWindow::new_unchecked(
            Timestamp::from_epoch_millis(3000),
            Timestamp::from_epoch_millis(8000),
        );
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn test_time_window_no_overlap() {
        let a = TimeWindow::new_unchecked(
            Timestamp::from_epoch_millis(1000),
            Timestamp::from_epoch_millis(2000),
        );
        let b = TimeWindow::new_unchecked(
            Timestamp::from_epoch_millis(3000),
            Timestamp::from_epoch_millis(4000),
        );
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_time_window_merge() {
        let a = TimeWindow::new_unchecked(
            Timestamp::from_epoch_millis(1000),
            Timestamp::from_epoch_millis(5000),
        );
        let b = TimeWindow::new_unchecked(
            Timestamp::from_epoch_millis(4000),
            Timestamp::from_epoch_millis(8000),
        );
        let m = a.merge(&b).unwrap();
        assert_eq!(m.start.0, 1000);
        assert_eq!(m.end.0, 8000);
    }

    #[test]
    fn test_time_window_merge_disjoint() {
        let a = TimeWindow::new_unchecked(
            Timestamp::from_epoch_millis(1000),
            Timestamp::from_epoch_millis(2000),
        );
        let b = TimeWindow::new_unchecked(
            Timestamp::from_epoch_millis(5000),
            Timestamp::from_epoch_millis(6000),
        );
        assert!(a.merge(&b).is_none());
    }

    #[test]
    fn test_time_window_intersection() {
        let a = TimeWindow::new_unchecked(
            Timestamp::from_epoch_millis(1000),
            Timestamp::from_epoch_millis(5000),
        );
        let b = TimeWindow::new_unchecked(
            Timestamp::from_epoch_millis(3000),
            Timestamp::from_epoch_millis(8000),
        );
        let inter = a.intersection(&b).unwrap();
        assert_eq!(inter.start.0, 3000);
        assert_eq!(inter.end.0, 5000);
    }

    #[test]
    fn test_time_window_fully_contains() {
        let outer = TimeWindow::new_unchecked(
            Timestamp::from_epoch_millis(1000),
            Timestamp::from_epoch_millis(10000),
        );
        let inner = TimeWindow::new_unchecked(
            Timestamp::from_epoch_millis(3000),
            Timestamp::from_epoch_millis(7000),
        );
        assert!(outer.fully_contains(&inner));
        assert!(!inner.fully_contains(&outer));
    }

    #[test]
    fn test_time_window_shift() {
        let w = TimeWindow::new_unchecked(
            Timestamp::from_epoch_millis(1000),
            Timestamp::from_epoch_millis(5000),
        );
        let shifted = w.shift(Duration::from_secs(10));
        assert_eq!(shifted.start.0, 11000);
        assert_eq!(shifted.end.0, 15000);
    }

    #[test]
    fn test_schedule_basic() {
        let mut sched = Schedule::new();
        assert!(sched.is_empty());
        sched.add_window(TimeWindow::new_unchecked(
            Timestamp::from_epoch_millis(1000),
            Timestamp::from_epoch_millis(5000),
        ));
        assert!(!sched.is_empty());
        assert_eq!(sched.window_count(), 1);
        assert!(sched.is_allowed(Timestamp::from_epoch_millis(3000)));
        assert!(!sched.is_allowed(Timestamp::from_epoch_millis(6000)));
    }

    #[test]
    fn test_schedule_merge_overlapping() {
        let sched = Schedule::from_windows(vec![
            TimeWindow::new_unchecked(Timestamp::from_epoch_millis(1000), Timestamp::from_epoch_millis(5000)),
            TimeWindow::new_unchecked(Timestamp::from_epoch_millis(4000), Timestamp::from_epoch_millis(8000)),
        ]);
        assert_eq!(sched.window_count(), 1);
        assert_eq!(sched.total_duration().as_millis(), 7000);
    }

    #[test]
    fn test_schedule_next_window() {
        let sched = Schedule::from_windows(vec![
            TimeWindow::new_unchecked(Timestamp::from_epoch_millis(1000), Timestamp::from_epoch_millis(2000)),
            TimeWindow::new_unchecked(Timestamp::from_epoch_millis(5000), Timestamp::from_epoch_millis(6000)),
        ]);
        let next = sched.next_window(Timestamp::from_epoch_millis(3000)).unwrap();
        assert_eq!(next.start.0, 5000);
    }

    #[test]
    fn test_schedule_time_until_allowed() {
        let sched = Schedule::from_windows(vec![
            TimeWindow::new_unchecked(Timestamp::from_epoch_millis(5000), Timestamp::from_epoch_millis(10000)),
        ]);
        let wait = sched.time_until_allowed(Timestamp::from_epoch_millis(2000)).unwrap();
        assert_eq!(wait.as_millis(), 3000);
        let wait2 = sched.time_until_allowed(Timestamp::from_epoch_millis(7000)).unwrap();
        assert!(wait2.is_zero());
    }

    #[test]
    fn test_schedule_covers() {
        let sched = Schedule::from_windows(vec![
            TimeWindow::new_unchecked(Timestamp::from_epoch_millis(1000), Timestamp::from_epoch_millis(5000)),
            TimeWindow::new_unchecked(Timestamp::from_epoch_millis(5000), Timestamp::from_epoch_millis(10000)),
        ]);
        let window = TimeWindow::new_unchecked(Timestamp::from_epoch_millis(2000), Timestamp::from_epoch_millis(8000));
        assert!(sched.covers(&window));
    }

    #[test]
    fn test_deadline_basic() {
        let dl = Deadline::hard(Timestamp::from_epoch_millis(10000));
        assert!(dl.is_hard);
        assert!(!dl.is_expired(Timestamp::from_epoch_millis(5000)));
        assert!(dl.is_expired(Timestamp::from_epoch_millis(10000)));
        assert_eq!(dl.remaining(Timestamp::from_epoch_millis(5000)).as_millis(), 5000);
    }

    #[test]
    fn test_deadline_soft() {
        let dl = Deadline::soft(Timestamp::from_epoch_millis(5000));
        assert!(!dl.is_hard);
    }

    #[test]
    fn test_deadline_has_time_for() {
        let dl = Deadline::hard(Timestamp::from_epoch_millis(10000));
        assert!(dl.has_time_for(Timestamp::from_epoch_millis(5000), Duration::from_secs(4)));
        assert!(!dl.has_time_for(Timestamp::from_epoch_millis(5000), Duration::from_secs(6)));
    }

    #[test]
    fn test_deadline_remaining_expired() {
        let dl = Deadline::hard(Timestamp::from_epoch_millis(5000));
        assert!(dl.remaining(Timestamp::from_epoch_millis(10000)).is_zero());
    }

    #[test]
    fn test_phase_all() {
        let phases = Phase::all();
        assert_eq!(phases.len(), 5);
        assert_eq!(phases[0], Phase::Preparation);
    }

    #[test]
    fn test_phase_terminal() {
        assert!(Phase::Complete.is_terminal());
        assert!(Phase::Rollback.is_terminal());
        assert!(!Phase::Execution.is_terminal());
    }

    #[test]
    fn test_phase_display() {
        assert_eq!(format!("{}", Phase::Execution), "Execution");
    }

    #[test]
    fn test_phase_duration_basic() {
        let pd = PhaseDuration::new(Phase::Execution, Duration::from_secs(60));
        assert_eq!(pd.variance(), None);
        assert!(!pd.is_overrun());
    }

    #[test]
    fn test_phase_duration_with_actual() {
        let pd = PhaseDuration::new(Phase::Execution, Duration::from_secs(60))
            .with_actual(Duration::from_secs(90));
        assert_eq!(pd.variance(), Some(30_000));
        assert!(pd.is_overrun());
        assert!((pd.ratio().unwrap() - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_phase_duration_underrun() {
        let pd = PhaseDuration::new(Phase::Verification, Duration::from_secs(60))
            .with_actual(Duration::from_secs(30));
        assert!(!pd.is_overrun());
        assert!((pd.ratio().unwrap() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_deployment_timing_basic() {
        let t = DeploymentTiming::new(
            Timestamp::from_epoch_millis(1000),
            Timestamp::from_epoch_millis(5000),
        );
        assert_eq!(t.planned_duration().as_millis(), 4000);
        assert!(!t.has_started());
        assert!(!t.is_in_progress());
        assert!(!t.is_complete());
    }

    #[test]
    fn test_deployment_timing_in_progress() {
        let t = DeploymentTiming::new(
            Timestamp::from_epoch_millis(1000),
            Timestamp::from_epoch_millis(5000),
        ).with_actual_start(Timestamp::from_epoch_millis(1100));
        assert!(t.has_started());
        assert!(t.is_in_progress());
        assert!(!t.is_complete());
    }

    #[test]
    fn test_deployment_timing_complete() {
        let t = DeploymentTiming::new(
            Timestamp::from_epoch_millis(1000),
            Timestamp::from_epoch_millis(5000),
        )
        .with_actual_start(Timestamp::from_epoch_millis(1100))
        .with_actual_end(Timestamp::from_epoch_millis(4500));
        assert!(t.is_complete());
        let actual = t.actual_duration().unwrap();
        assert_eq!(actual.as_millis(), 3400);
    }

    #[test]
    fn test_deployment_timing_overrun() {
        let t = DeploymentTiming::new(
            Timestamp::from_epoch_millis(0),
            Timestamp::from_epoch_millis(4000),
        )
        .with_actual_start(Timestamp::from_epoch_millis(0))
        .with_actual_end(Timestamp::from_epoch_millis(6000));
        assert_eq!(t.overrun(), Some(2000));
    }

    #[test]
    fn test_deployment_timing_phases() {
        let mut t = DeploymentTiming::default();
        t.add_phase(PhaseDuration::new(Phase::Preparation, Duration::from_secs(10))
            .with_actual(Duration::from_secs(12)));
        t.add_phase(PhaseDuration::new(Phase::Execution, Duration::from_secs(30))
            .with_actual(Duration::from_secs(25)));
        assert_eq!(t.total_planned_phase_time().as_secs(), 40);
        assert_eq!(t.total_actual_phase_time().unwrap().as_secs(), 37);
        assert_eq!(t.overrun_phases().len(), 1);
    }

    #[test]
    fn test_deployment_timing_as_windows() {
        let t = DeploymentTiming::new(
            Timestamp::from_epoch_millis(1000),
            Timestamp::from_epoch_millis(5000),
        )
        .with_actual_start(Timestamp::from_epoch_millis(1000))
        .with_actual_end(Timestamp::from_epoch_millis(4500));
        let pw = t.as_planned_window();
        assert_eq!(pw.duration().as_millis(), 4000);
        let aw = t.as_actual_window().unwrap();
        assert_eq!(aw.duration().as_millis(), 3500);
    }

    #[test]
    fn test_day_of_week() {
        assert!(DayOfWeek::Monday.is_weekday());
        assert!(!DayOfWeek::Saturday.is_weekday());
        assert_eq!(format!("{}", DayOfWeek::Friday), "Friday");
    }

    #[test]
    fn test_day_of_week_chrono_roundtrip() {
        for day in &[DayOfWeek::Monday, DayOfWeek::Tuesday, DayOfWeek::Wednesday,
                     DayOfWeek::Thursday, DayOfWeek::Friday, DayOfWeek::Saturday, DayOfWeek::Sunday] {
            let chrono_day = day.to_chrono();
            let back = DayOfWeek::from_chrono(chrono_day);
            assert_eq!(*day, back);
        }
    }

    #[test]
    fn test_maintenance_window_creation() {
        let mw = MaintenanceWindow::new(
            vec![DayOfWeek::Monday, DayOfWeek::Wednesday],
            22, 0,
            Duration::from_hours(4),
        ).unwrap();
        assert_eq!(mw.allowed_days.len(), 2);
        assert_eq!(mw.start_hour, 22);
    }

    #[test]
    fn test_maintenance_window_invalid_hour() {
        let result = MaintenanceWindow::new(
            vec![DayOfWeek::Monday],
            25, 0,
            Duration::from_hours(1),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_maintenance_window_invalid_minute() {
        let result = MaintenanceWindow::new(
            vec![DayOfWeek::Monday],
            10, 60,
            Duration::from_hours(1),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_maintenance_window_display() {
        let mw = MaintenanceWindow::new(
            vec![DayOfWeek::Monday],
            22, 30,
            Duration::from_hours(2),
        ).unwrap();
        let s = format!("{}", mw);
        assert!(s.contains("Monday"));
        assert!(s.contains("22:30"));
    }

    #[test]
    fn test_serde_timestamp() {
        let ts = Timestamp::from_epoch_millis(1234567890);
        let json = serde_json::to_string(&ts).unwrap();
        let ts2: Timestamp = serde_json::from_str(&json).unwrap();
        assert_eq!(ts, ts2);
    }

    #[test]
    fn test_serde_duration() {
        let d = Duration::from_secs(42);
        let json = serde_json::to_string(&d).unwrap();
        let d2: Duration = serde_json::from_str(&json).unwrap();
        assert_eq!(d, d2);
    }

    #[test]
    fn test_serde_time_window() {
        let w = TimeWindow::new_unchecked(
            Timestamp::from_epoch_millis(1000),
            Timestamp::from_epoch_millis(5000),
        );
        let json = serde_json::to_string(&w).unwrap();
        let w2: TimeWindow = serde_json::from_str(&json).unwrap();
        assert_eq!(w, w2);
    }

    #[test]
    fn test_serde_schedule() {
        let sched = Schedule::from_windows(vec![
            TimeWindow::new_unchecked(Timestamp::from_epoch_millis(1000), Timestamp::from_epoch_millis(2000)),
        ]);
        let json = serde_json::to_string(&sched).unwrap();
        let sched2: Schedule = serde_json::from_str(&json).unwrap();
        assert_eq!(sched, sched2);
    }

    #[test]
    fn test_serde_deployment_timing() {
        let t = DeploymentTiming::new(
            Timestamp::from_epoch_millis(1000),
            Timestamp::from_epoch_millis(5000),
        );
        let json = serde_json::to_string(&t).unwrap();
        let t2: DeploymentTiming = serde_json::from_str(&json).unwrap();
        assert_eq!(t, t2);
    }

    #[test]
    fn test_serde_deadline() {
        let dl = Deadline::hard(Timestamp::from_epoch_millis(10000));
        let json = serde_json::to_string(&dl).unwrap();
        let dl2: Deadline = serde_json::from_str(&json).unwrap();
        assert_eq!(dl, dl2);
    }

    #[test]
    fn test_timestamp_display() {
        let ts = Timestamp::from_epoch_millis(0);
        let s = format!("{}", ts);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_deployment_timing_from_duration() {
        let t = DeploymentTiming::from_duration(
            Timestamp::from_epoch_millis(1000),
            Duration::from_secs(60),
        );
        assert_eq!(t.planned_end.as_epoch_millis(), 61000);
    }

    #[test]
    fn test_schedule_overlapping_windows() {
        let sched = Schedule::from_windows(vec![
            TimeWindow::new_unchecked(Timestamp::from_epoch_millis(1000), Timestamp::from_epoch_millis(5000)),
            TimeWindow::new_unchecked(Timestamp::from_epoch_millis(10000), Timestamp::from_epoch_millis(15000)),
        ]);
        let query = TimeWindow::new_unchecked(Timestamp::from_epoch_millis(4000), Timestamp::from_epoch_millis(11000));
        let overlapping = sched.overlapping_windows(&query);
        assert_eq!(overlapping.len(), 2);
    }

    #[test]
    fn test_time_window_midpoint() {
        let w = TimeWindow::new_unchecked(
            Timestamp::from_epoch_millis(1000),
            Timestamp::from_epoch_millis(5000),
        );
        assert_eq!(w.midpoint().as_epoch_millis(), 3000);
    }

    #[test]
    fn test_duration_as_secs_f64() {
        let d = Duration::from_millis(1500);
        assert!((d.as_secs_f64() - 1.5).abs() < 0.001);
    }
}

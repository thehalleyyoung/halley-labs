//! Progress tracking for long-running planner operations.
//!
//! Provides [`ProgressTracker`] for managing multi-phase operations,
//! [`ProgressBar`] for rendering visual progress indicators, and
//! [`ProgressCallback`] / [`ConsoleProgress`] for observing progress events.

use std::cell::RefCell;
use std::fmt;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// PhaseStatus
// ---------------------------------------------------------------------------

/// Status of a single phase within a tracked operation.
#[derive(Debug, Clone)]
pub enum PhaseStatus {
    Pending,
    InProgress,
    Complete,
    Failed(String),
}

impl PhaseStatus {
    /// Returns `true` when the phase has completed successfully.
    pub fn is_complete(&self) -> bool {
        matches!(self, PhaseStatus::Complete)
    }

    /// Returns `true` when the phase has failed.
    pub fn is_failed(&self) -> bool {
        matches!(self, PhaseStatus::Failed(_))
    }
}

impl fmt::Display for PhaseStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PhaseStatus::Pending => write!(f, "Pending"),
            PhaseStatus::InProgress => write!(f, "InProgress"),
            PhaseStatus::Complete => write!(f, "Complete"),
            PhaseStatus::Failed(reason) => write!(f, "Failed({})", reason),
        }
    }
}

impl PartialEq for PhaseStatus {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (PhaseStatus::Pending, PhaseStatus::Pending) => true,
            (PhaseStatus::InProgress, PhaseStatus::InProgress) => true,
            (PhaseStatus::Complete, PhaseStatus::Complete) => true,
            (PhaseStatus::Failed(a), PhaseStatus::Failed(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for PhaseStatus {}

// ---------------------------------------------------------------------------
// PhaseInfo
// ---------------------------------------------------------------------------

/// Information about a single phase of a tracked operation.
#[derive(Debug, Clone)]
pub struct PhaseInfo {
    /// Human-readable name of this phase.
    pub name: String,
    /// When the phase started executing.
    pub started_at: Option<Instant>,
    /// When the phase finished executing.
    pub completed_at: Option<Instant>,
    /// Fractional progress within this phase (0.0 – 1.0).
    pub progress: f64,
    /// Current status of the phase.
    pub status: PhaseStatus,
}

impl PhaseInfo {
    /// Create a new pending phase with the given name.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            started_at: None,
            completed_at: None,
            progress: 0.0,
            status: PhaseStatus::Pending,
        }
    }

    /// Wall-clock duration of this phase.
    ///
    /// * If the phase has both `started_at` and `completed_at`, returns the
    ///   difference.
    /// * If the phase has `started_at` but is still running, returns the time
    ///   elapsed since it started.
    /// * Otherwise returns `None`.
    pub fn duration(&self) -> Option<Duration> {
        match (self.started_at, self.completed_at) {
            (Some(start), Some(end)) => Some(end.duration_since(start)),
            (Some(start), None) => Some(start.elapsed()),
            _ => None,
        }
    }

    /// Duration of this phase in milliseconds, or `0` if not yet started.
    pub fn duration_ms(&self) -> u64 {
        self.duration().map_or(0, |d| d.as_millis() as u64)
    }
}

// ---------------------------------------------------------------------------
// ProgressTracker
// ---------------------------------------------------------------------------

/// Tracks progress of a multi-phase operation.
///
/// Phases are added sequentially via [`start_phase`](Self::start_phase).
/// Each phase can report fractional progress and be completed or failed.
#[derive(Debug)]
pub struct ProgressTracker {
    /// Name of the currently active phase (if any).
    pub current_phase: Option<String>,
    /// All phases that have been registered.
    pub phases: Vec<PhaseInfo>,
    /// When this tracker was created.
    pub start_time: Instant,
}

impl ProgressTracker {
    /// Create a new tracker. The clock starts immediately.
    pub fn new() -> Self {
        Self {
            current_phase: None,
            phases: Vec::new(),
            start_time: Instant::now(),
        }
    }

    /// Begin a new phase. If a previous phase was still in progress it is
    /// automatically completed first.
    pub fn start_phase(&mut self, name: &str) {
        // Auto-complete any in-progress phase.
        if self.current_phase.is_some() {
            self.complete_phase();
        }

        let mut info = PhaseInfo::new(name);
        info.status = PhaseStatus::InProgress;
        info.started_at = Some(Instant::now());
        self.phases.push(info);
        self.current_phase = Some(name.to_string());
    }

    /// Update progress of the current phase.
    ///
    /// `current` and `total` represent items processed out of a known total.
    /// If `total` is zero the progress is set to `0.0` to avoid division by
    /// zero.
    pub fn update_progress(&mut self, current: usize, total: usize) {
        if let Some(ref phase_name) = self.current_phase {
            if let Some(info) = self.phases.iter_mut().rev().find(|p| p.name == *phase_name) {
                info.progress = if total == 0 {
                    0.0
                } else {
                    (current as f64 / total as f64).clamp(0.0, 1.0)
                };
            }
        }
    }

    /// Mark the current phase as successfully completed.
    pub fn complete_phase(&mut self) {
        if let Some(ref phase_name) = self.current_phase.take() {
            if let Some(info) = self.phases.iter_mut().rev().find(|p| p.name == *phase_name) {
                info.status = PhaseStatus::Complete;
                info.progress = 1.0;
                info.completed_at = Some(Instant::now());
            }
        }
    }

    /// Mark the current phase as failed with the given `reason`.
    pub fn fail_phase(&mut self, reason: &str) {
        if let Some(ref phase_name) = self.current_phase.take() {
            if let Some(info) = self.phases.iter_mut().rev().find(|p| p.name == *phase_name) {
                info.status = PhaseStatus::Failed(reason.to_string());
                info.completed_at = Some(Instant::now());
            }
        }
    }

    /// Total wall-clock time since the tracker was created.
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Estimate the remaining time based on the average duration of completed
    /// phases and the number of incomplete phases.
    ///
    /// Returns `None` if no phases have been completed yet (there is nothing
    /// to extrapolate from).
    pub fn estimated_remaining(&self) -> Option<Duration> {
        let completed: Vec<&PhaseInfo> = self
            .phases
            .iter()
            .filter(|p| p.status.is_complete())
            .collect();

        if completed.is_empty() {
            return None;
        }

        let total_completed_ms: u64 = completed.iter().map(|p| p.duration_ms()).sum();
        let avg_ms = total_completed_ms as f64 / completed.len() as f64;

        // Count phases that are not yet complete (pending + in-progress).
        let remaining_count = self
            .phases
            .iter()
            .filter(|p| !p.status.is_complete() && !p.status.is_failed())
            .count();

        // For the currently in-progress phase, subtract elapsed time.
        let current_remaining_ms = self
            .current_phase
            .as_ref()
            .and_then(|name| self.phases.iter().rev().find(|p| p.name == *name))
            .and_then(|p| p.duration())
            .map(|d| {
                let elapsed = d.as_millis() as f64;
                (avg_ms - elapsed).max(0.0)
            })
            .unwrap_or(0.0);

        // Remaining full phases (minus the current one which is partially done).
        let full_remaining = if remaining_count > 0 {
            remaining_count.saturating_sub(1)
        } else {
            0
        };

        let est_ms = (full_remaining as f64 * avg_ms) + current_remaining_ms;
        Some(Duration::from_millis(est_ms as u64))
    }

    /// Overall progress across all phases as a value in `0.0 ..= 1.0`.
    ///
    /// Each phase contributes equally. A completed phase counts as `1.0`, a
    /// phase still in progress counts as its fractional `progress`, and a
    /// pending phase counts as `0.0`.
    pub fn overall_progress(&self) -> f64 {
        if self.phases.is_empty() {
            return 0.0;
        }

        let sum: f64 = self
            .phases
            .iter()
            .map(|p| match p.status {
                PhaseStatus::Complete => 1.0,
                PhaseStatus::InProgress => p.progress,
                PhaseStatus::Failed(_) => p.progress,
                PhaseStatus::Pending => 0.0,
            })
            .sum();

        (sum / self.phases.len() as f64).clamp(0.0, 1.0)
    }

    /// Total number of registered phases.
    pub fn phase_count(&self) -> usize {
        self.phases.len()
    }

    /// Number of phases that have completed successfully.
    pub fn completed_count(&self) -> usize {
        self.phases.iter().filter(|p| p.status.is_complete()).count()
    }

    /// Name of the currently active phase, if any.
    pub fn current_phase_name(&self) -> Option<&str> {
        self.current_phase.as_deref()
    }
}

impl Default for ProgressTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ProgressBar
// ---------------------------------------------------------------------------

/// Static helpers for rendering text-based progress bars.
pub struct ProgressBar;

impl ProgressBar {
    /// Render a progress bar of the given `width` (in characters, excluding
    /// brackets and percentage label).
    ///
    /// ```text
    /// [████████░░░░░░░░] 50%
    /// ```
    pub fn render(progress: f64, width: usize) -> String {
        let clamped = progress.clamp(0.0, 1.0);
        let filled = (clamped * width as f64).round() as usize;
        let empty = width.saturating_sub(filled);

        let bar_filled: String = "█".repeat(filled);
        let bar_empty: String = "░".repeat(empty);
        let pct = (clamped * 100.0).round() as u32;

        format!("[{}{}] {}%", bar_filled, bar_empty, pct)
    }

    /// Render a progress bar with an appended label.
    ///
    /// ```text
    /// [████░░░░] 50% Downloading
    /// ```
    pub fn render_with_label(progress: f64, width: usize, label: &str) -> String {
        let bar = Self::render(progress, width);
        format!("{} {}", bar, label)
    }

    /// Return a braille-based spinner character for the given frame index.
    ///
    /// The sequence cycles through: ⣾ ⣽ ⣻ ⢿ ⡿ ⣟ ⣯ ⣷
    pub fn spinner(frame: usize) -> char {
        const FRAMES: [char; 8] = ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷'];
        FRAMES[frame % FRAMES.len()]
    }
}

// ---------------------------------------------------------------------------
// ProgressCallback trait
// ---------------------------------------------------------------------------

/// Observer interface for progress events.
///
/// Implement this trait to react to phase transitions and progress updates
/// from a [`ProgressTracker`].
pub trait ProgressCallback {
    /// Called when a new phase begins.
    fn on_phase_start(&self, phase: &str);

    /// Called when the current phase reports progress.
    fn on_progress(&self, current: usize, total: usize);

    /// Called when a phase completes successfully.
    fn on_phase_complete(&self, phase: &str, duration_ms: u64);

    /// Called for arbitrary informational messages.
    fn on_message(&self, msg: &str);
}

// ---------------------------------------------------------------------------
// ConsoleProgress
// ---------------------------------------------------------------------------

/// A [`ProgressCallback`] implementation that records formatted messages
/// internally.
///
/// Uses interior mutability (`RefCell`) so that callback methods can take
/// `&self` while still appending to the message log.
#[derive(Debug)]
pub struct ConsoleProgress {
    messages: RefCell<Vec<String>>,
}

impl ConsoleProgress {
    /// Create a new, empty console progress recorder.
    pub fn new() -> Self {
        Self {
            messages: RefCell::new(Vec::new()),
        }
    }

    /// Access the recorded messages for inspection / testing.
    pub fn messages(&self) -> Vec<String> {
        self.messages.borrow().clone()
    }

    /// Number of recorded messages.
    pub fn message_count(&self) -> usize {
        self.messages.borrow().len()
    }

    /// Clear all recorded messages.
    pub fn clear(&self) {
        self.messages.borrow_mut().clear();
    }

    fn push(&self, msg: String) {
        self.messages.borrow_mut().push(msg);
    }
}

impl Default for ConsoleProgress {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgressCallback for ConsoleProgress {
    fn on_phase_start(&self, phase: &str) {
        self.push(format!("▶ Starting phase: {}", phase));
    }

    fn on_progress(&self, current: usize, total: usize) {
        let progress = if total == 0 {
            0.0
        } else {
            current as f64 / total as f64
        };
        let bar = ProgressBar::render(progress, 20);
        self.push(bar);
    }

    fn on_phase_complete(&self, phase: &str, duration_ms: u64) {
        self.push(format!("✓ Phase {} complete ({}ms)", phase, duration_ms));
    }

    fn on_message(&self, msg: &str) {
        self.push(msg.to_string());
    }
}

// ---------------------------------------------------------------------------
// NullProgress
// ---------------------------------------------------------------------------

/// A no-op [`ProgressCallback`] implementation that discards all events.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullProgress;

impl NullProgress {
    pub fn new() -> Self {
        Self
    }
}

impl ProgressCallback for NullProgress {
    fn on_phase_start(&self, _phase: &str) {}
    fn on_progress(&self, _current: usize, _total: usize) {}
    fn on_phase_complete(&self, _phase: &str, _duration_ms: u64) {}
    fn on_message(&self, _msg: &str) {}
}

// ---------------------------------------------------------------------------
// Tracked operation helper
// ---------------------------------------------------------------------------

/// Run a closure while driving a [`ProgressTracker`] and emitting events to a
/// [`ProgressCallback`].
///
/// This is a convenience wrapper that starts a phase, invokes the callback,
/// and completes (or fails) the phase based on the closure's `Result`.
pub fn run_tracked_phase<F, T, E>(
    tracker: &mut ProgressTracker,
    callback: &dyn ProgressCallback,
    phase_name: &str,
    f: F,
) -> Result<T, E>
where
    F: FnOnce() -> Result<T, E>,
    E: fmt::Display,
{
    tracker.start_phase(phase_name);
    callback.on_phase_start(phase_name);

    match f() {
        Ok(value) => {
            let dur_ms = tracker
                .phases
                .last()
                .map(|p| p.duration_ms())
                .unwrap_or(0);
            tracker.complete_phase();
            callback.on_phase_complete(phase_name, dur_ms);
            Ok(value)
        }
        Err(e) => {
            let reason = e.to_string();
            tracker.fail_phase(&reason);
            Err(e)
        }
    }
}

/// Format a [`Duration`] as a human-friendly string.
///
/// Examples: `"0ms"`, `"142ms"`, `"3.2s"`, `"1m 5s"`, `"1h 2m"`.
pub fn format_duration(d: Duration) -> String {
    let total_ms = d.as_millis();
    if total_ms < 1000 {
        return format!("{}ms", total_ms);
    }

    let total_secs = d.as_secs();
    if total_secs < 60 {
        let frac = d.as_secs_f64();
        return format!("{:.1}s", frac);
    }

    let minutes = total_secs / 60;
    let secs = total_secs % 60;
    if minutes < 60 {
        return format!("{}m {}s", minutes, secs);
    }

    let hours = minutes / 60;
    let mins = minutes % 60;
    format!("{}h {}m", hours, mins)
}

/// Compute a simple ETA string from a progress fraction and elapsed duration.
///
/// Returns `None` if `progress` is `<= 0.0` (cannot estimate).
pub fn eta_string(progress: f64, elapsed: Duration) -> Option<String> {
    if progress <= 0.0 || progress > 1.0 {
        return None;
    }
    let elapsed_ms = elapsed.as_millis() as f64;
    let total_est_ms = elapsed_ms / progress;
    let remaining_ms = (total_est_ms - elapsed_ms).max(0.0);
    Some(format_duration(Duration::from_millis(remaining_ms as u64)))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    // -- PhaseStatus tests --------------------------------------------------

    #[test]
    fn phase_status_is_complete() {
        assert!(PhaseStatus::Complete.is_complete());
        assert!(!PhaseStatus::Pending.is_complete());
        assert!(!PhaseStatus::InProgress.is_complete());
        assert!(!PhaseStatus::Failed("err".into()).is_complete());
    }

    #[test]
    fn phase_status_is_failed() {
        assert!(PhaseStatus::Failed("boom".into()).is_failed());
        assert!(!PhaseStatus::Pending.is_failed());
        assert!(!PhaseStatus::InProgress.is_failed());
        assert!(!PhaseStatus::Complete.is_failed());
    }

    #[test]
    fn phase_status_display() {
        assert_eq!(PhaseStatus::Pending.to_string(), "Pending");
        assert_eq!(PhaseStatus::InProgress.to_string(), "InProgress");
        assert_eq!(PhaseStatus::Complete.to_string(), "Complete");
        assert_eq!(
            PhaseStatus::Failed("oops".into()).to_string(),
            "Failed(oops)"
        );
    }

    #[test]
    fn phase_status_equality() {
        assert_eq!(PhaseStatus::Pending, PhaseStatus::Pending);
        assert_eq!(
            PhaseStatus::Failed("x".into()),
            PhaseStatus::Failed("x".into())
        );
        assert_ne!(
            PhaseStatus::Failed("x".into()),
            PhaseStatus::Failed("y".into())
        );
        assert_ne!(PhaseStatus::Pending, PhaseStatus::Complete);
    }

    // -- PhaseInfo tests ----------------------------------------------------

    #[test]
    fn phase_info_new_defaults() {
        let info = PhaseInfo::new("test-phase");
        assert_eq!(info.name, "test-phase");
        assert!(info.started_at.is_none());
        assert!(info.completed_at.is_none());
        assert_eq!(info.progress, 0.0);
        assert_eq!(info.status, PhaseStatus::Pending);
    }

    #[test]
    fn phase_info_duration_not_started() {
        let info = PhaseInfo::new("no-start");
        assert!(info.duration().is_none());
        assert_eq!(info.duration_ms(), 0);
    }

    #[test]
    fn phase_info_duration_in_progress() {
        let mut info = PhaseInfo::new("running");
        info.started_at = Some(Instant::now());
        info.status = PhaseStatus::InProgress;
        thread::sleep(Duration::from_millis(5));
        let d = info.duration().expect("should have duration");
        assert!(d.as_millis() >= 1, "elapsed should be >= 1ms");
    }

    #[test]
    fn phase_info_duration_completed() {
        let mut info = PhaseInfo::new("done");
        let start = Instant::now();
        info.started_at = Some(start);
        thread::sleep(Duration::from_millis(5));
        info.completed_at = Some(Instant::now());
        info.status = PhaseStatus::Complete;
        let d = info.duration().expect("should have duration");
        assert!(d.as_millis() >= 1);
    }

    // -- ProgressTracker tests ----------------------------------------------

    #[test]
    fn tracker_new_defaults() {
        let t = ProgressTracker::new();
        assert!(t.current_phase.is_none());
        assert!(t.phases.is_empty());
        assert_eq!(t.phase_count(), 0);
        assert_eq!(t.completed_count(), 0);
        assert_eq!(t.overall_progress(), 0.0);
    }

    #[test]
    fn tracker_start_and_complete_phase() {
        let mut t = ProgressTracker::new();
        t.start_phase("alpha");
        assert_eq!(t.current_phase_name(), Some("alpha"));
        assert_eq!(t.phase_count(), 1);
        assert_eq!(t.completed_count(), 0);

        thread::sleep(Duration::from_millis(2));
        t.complete_phase();
        assert!(t.current_phase_name().is_none());
        assert_eq!(t.completed_count(), 1);
        assert!(t.phases[0].status.is_complete());
        assert!(t.phases[0].completed_at.is_some());
    }

    #[test]
    fn tracker_fail_phase() {
        let mut t = ProgressTracker::new();
        t.start_phase("will-fail");
        t.fail_phase("something broke");
        assert!(t.current_phase_name().is_none());
        assert_eq!(
            t.phases[0].status,
            PhaseStatus::Failed("something broke".into())
        );
        assert!(t.phases[0].completed_at.is_some());
    }

    #[test]
    fn tracker_update_progress() {
        let mut t = ProgressTracker::new();
        t.start_phase("loading");
        t.update_progress(25, 100);
        assert!((t.phases[0].progress - 0.25).abs() < f64::EPSILON);

        t.update_progress(100, 100);
        assert!((t.phases[0].progress - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn tracker_update_progress_zero_total() {
        let mut t = ProgressTracker::new();
        t.start_phase("empty");
        t.update_progress(5, 0);
        assert_eq!(t.phases[0].progress, 0.0);
    }

    #[test]
    fn tracker_update_progress_no_current_phase() {
        let mut t = ProgressTracker::new();
        // Should not panic when there is no active phase.
        t.update_progress(1, 10);
    }

    #[test]
    fn tracker_auto_complete_on_start() {
        let mut t = ProgressTracker::new();
        t.start_phase("first");
        thread::sleep(Duration::from_millis(1));
        t.start_phase("second");

        assert_eq!(t.phase_count(), 2);
        assert!(t.phases[0].status.is_complete());
        assert_eq!(t.current_phase_name(), Some("second"));
    }

    #[test]
    fn tracker_overall_progress_no_phases() {
        let t = ProgressTracker::new();
        assert_eq!(t.overall_progress(), 0.0);
    }

    #[test]
    fn tracker_overall_progress_mixed() {
        let mut t = ProgressTracker::new();
        t.start_phase("a");
        t.complete_phase(); // 1.0

        t.start_phase("b");
        t.update_progress(50, 100); // 0.5

        // Two phases: (1.0 + 0.5) / 2 = 0.75
        let p = t.overall_progress();
        assert!((p - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn tracker_overall_progress_all_complete() {
        let mut t = ProgressTracker::new();
        t.start_phase("x");
        t.complete_phase();
        t.start_phase("y");
        t.complete_phase();
        assert!((t.overall_progress() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn tracker_elapsed_is_positive() {
        let t = ProgressTracker::new();
        thread::sleep(Duration::from_millis(2));
        assert!(t.elapsed().as_millis() >= 1);
    }

    #[test]
    fn tracker_estimated_remaining_none_without_completed() {
        let mut t = ProgressTracker::new();
        t.start_phase("first");
        assert!(t.estimated_remaining().is_none());
    }

    #[test]
    fn tracker_estimated_remaining_with_completed() {
        let mut t = ProgressTracker::new();
        t.start_phase("phase-1");
        thread::sleep(Duration::from_millis(10));
        t.complete_phase();

        t.start_phase("phase-2");
        // One completed, one in progress → estimated remaining should be Some.
        let est = t.estimated_remaining();
        assert!(est.is_some());
    }

    #[test]
    fn tracker_complete_without_starting() {
        let mut t = ProgressTracker::new();
        // complete_phase with no current phase should be a no-op.
        t.complete_phase();
        assert_eq!(t.phase_count(), 0);
    }

    #[test]
    fn tracker_fail_without_starting() {
        let mut t = ProgressTracker::new();
        t.fail_phase("no phase");
        assert_eq!(t.phase_count(), 0);
    }

    #[test]
    fn tracker_default_trait() {
        let t = ProgressTracker::default();
        assert!(t.phases.is_empty());
    }

    // -- ProgressBar tests --------------------------------------------------

    #[test]
    fn progress_bar_zero() {
        let bar = ProgressBar::render(0.0, 10);
        assert_eq!(bar, "[░░░░░░░░░░] 0%");
    }

    #[test]
    fn progress_bar_full() {
        let bar = ProgressBar::render(1.0, 10);
        assert_eq!(bar, "[██████████] 100%");
    }

    #[test]
    fn progress_bar_half() {
        let bar = ProgressBar::render(0.5, 10);
        assert_eq!(bar, "[█████░░░░░] 50%");
    }

    #[test]
    fn progress_bar_quarter() {
        let bar = ProgressBar::render(0.25, 8);
        assert_eq!(bar, "[██░░░░░░] 25%");
    }

    #[test]
    fn progress_bar_clamps_over_one() {
        let bar = ProgressBar::render(1.5, 10);
        assert_eq!(bar, "[██████████] 100%");
    }

    #[test]
    fn progress_bar_clamps_negative() {
        let bar = ProgressBar::render(-0.5, 10);
        assert_eq!(bar, "[░░░░░░░░░░] 0%");
    }

    #[test]
    fn progress_bar_width_zero() {
        let bar = ProgressBar::render(0.5, 0);
        assert_eq!(bar, "[] 50%");
    }

    #[test]
    fn progress_bar_with_label() {
        let bar = ProgressBar::render_with_label(0.5, 10, "Downloading");
        assert_eq!(bar, "[█████░░░░░] 50% Downloading");
    }

    #[test]
    fn progress_bar_with_empty_label() {
        let bar = ProgressBar::render_with_label(0.75, 4, "");
        assert_eq!(bar, "[███░] 75% ");
    }

    // -- Spinner tests ------------------------------------------------------

    #[test]
    fn spinner_cycles() {
        let expected = ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷'];
        for (i, &ch) in expected.iter().enumerate() {
            assert_eq!(ProgressBar::spinner(i), ch);
        }
        // Verify wrap-around.
        assert_eq!(ProgressBar::spinner(8), '⣾');
        assert_eq!(ProgressBar::spinner(9), '⣽');
        assert_eq!(ProgressBar::spinner(16), '⣾');
    }

    // -- ConsoleProgress tests ----------------------------------------------

    #[test]
    fn console_progress_phase_start() {
        let cp = ConsoleProgress::new();
        cp.on_phase_start("build");
        let msgs = cp.messages();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0], "▶ Starting phase: build");
    }

    #[test]
    fn console_progress_on_progress() {
        let cp = ConsoleProgress::new();
        cp.on_progress(50, 100);
        let msgs = cp.messages();
        assert_eq!(msgs.len(), 1);
        assert!(msgs[0].contains("50%"));
    }

    #[test]
    fn console_progress_phase_complete() {
        let cp = ConsoleProgress::new();
        cp.on_phase_complete("deploy", 1234);
        let msgs = cp.messages();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0], "✓ Phase deploy complete (1234ms)");
    }

    #[test]
    fn console_progress_on_message() {
        let cp = ConsoleProgress::new();
        cp.on_message("hello world");
        let msgs = cp.messages();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0], "hello world");
    }

    #[test]
    fn console_progress_message_count_and_clear() {
        let cp = ConsoleProgress::new();
        cp.on_message("one");
        cp.on_message("two");
        assert_eq!(cp.message_count(), 2);
        cp.clear();
        assert_eq!(cp.message_count(), 0);
    }

    #[test]
    fn console_progress_on_progress_zero_total() {
        let cp = ConsoleProgress::new();
        cp.on_progress(5, 0);
        let msgs = cp.messages();
        assert!(msgs[0].contains("0%"));
    }

    // -- NullProgress tests -------------------------------------------------

    #[test]
    fn null_progress_is_noop() {
        let np = NullProgress::new();
        np.on_phase_start("x");
        np.on_progress(1, 2);
        np.on_phase_complete("x", 100);
        np.on_message("hi");
        // No panic, no state — just verify it compiles and runs.
    }

    // -- run_tracked_phase tests --------------------------------------------

    #[test]
    fn run_tracked_phase_ok() {
        let mut tracker = ProgressTracker::new();
        let cb = ConsoleProgress::new();
        let result: Result<i32, String> =
            run_tracked_phase(&mut tracker, &cb, "compute", || Ok(42));
        assert_eq!(result.unwrap(), 42);
        assert_eq!(tracker.completed_count(), 1);
        let msgs = cb.messages();
        assert!(msgs.iter().any(|m| m.contains("Starting phase: compute")));
        assert!(msgs.iter().any(|m| m.contains("✓ Phase compute complete")));
    }

    #[test]
    fn run_tracked_phase_err() {
        let mut tracker = ProgressTracker::new();
        let cb = ConsoleProgress::new();
        let result: Result<i32, String> =
            run_tracked_phase(&mut tracker, &cb, "fail-op", || Err("broken".to_string()));
        assert!(result.is_err());
        assert!(tracker.phases[0].status.is_failed());
    }

    // -- format_duration tests ----------------------------------------------

    #[test]
    fn format_duration_milliseconds() {
        assert_eq!(format_duration(Duration::from_millis(0)), "0ms");
        assert_eq!(format_duration(Duration::from_millis(42)), "42ms");
        assert_eq!(format_duration(Duration::from_millis(999)), "999ms");
    }

    #[test]
    fn format_duration_seconds() {
        assert_eq!(format_duration(Duration::from_millis(1500)), "1.5s");
        assert_eq!(format_duration(Duration::from_secs(30)), "30.0s");
    }

    #[test]
    fn format_duration_minutes() {
        assert_eq!(format_duration(Duration::from_secs(65)), "1m 5s");
        assert_eq!(format_duration(Duration::from_secs(600)), "10m 0s");
    }

    #[test]
    fn format_duration_hours() {
        assert_eq!(format_duration(Duration::from_secs(3720)), "1h 2m");
    }

    // -- eta_string tests ---------------------------------------------------

    #[test]
    fn eta_string_none_at_zero() {
        assert!(eta_string(0.0, Duration::from_secs(10)).is_none());
    }

    #[test]
    fn eta_string_none_at_negative() {
        assert!(eta_string(-0.1, Duration::from_secs(10)).is_none());
    }

    #[test]
    fn eta_string_some_at_half() {
        // 50% done after 10s → ~10s remaining
        let result = eta_string(0.5, Duration::from_secs(10));
        assert!(result.is_some());
        let s = result.unwrap();
        assert!(s.contains("10"), "expected ~10s, got: {}", s);
    }

    #[test]
    fn eta_string_small_remaining() {
        // 99% done after 99ms → ~1ms remaining
        let result = eta_string(0.99, Duration::from_millis(99));
        assert!(result.is_some());
    }
}

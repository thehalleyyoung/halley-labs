//! Compilation and rendering progress reporting.
//!
//! Provides structured progress updates for long-running operations such as
//! full compilation pipelines and audio rendering passes.

use std::fmt;
use std::time::{Duration, Instant};

// ── Compilation Phases ──────────────────────────────────────────────────────

/// Enumeration of every discrete phase in the compilation pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompilationPhase {
    Parsing,
    SemanticAnalysis,
    TypeChecking,
    Optimization,
    IrGeneration,
    Passes,
    Codegen,
    WcetVerification,
}

impl CompilationPhase {
    /// All phases in pipeline order.
    pub const ALL: &'static [CompilationPhase] = &[
        CompilationPhase::Parsing,
        CompilationPhase::SemanticAnalysis,
        CompilationPhase::TypeChecking,
        CompilationPhase::Optimization,
        CompilationPhase::IrGeneration,
        CompilationPhase::Passes,
        CompilationPhase::Codegen,
        CompilationPhase::WcetVerification,
    ];

    /// Zero-based index used for ordering / progress fraction.
    pub fn index(self) -> usize {
        match self {
            Self::Parsing => 0,
            Self::SemanticAnalysis => 1,
            Self::TypeChecking => 2,
            Self::Optimization => 3,
            Self::IrGeneration => 4,
            Self::Passes => 5,
            Self::Codegen => 6,
            Self::WcetVerification => 7,
        }
    }

    /// Total number of phases.
    pub fn total() -> usize {
        Self::ALL.len()
    }
}

impl fmt::Display for CompilationPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parsing => write!(f, "Parsing"),
            Self::SemanticAnalysis => write!(f, "Semantic Analysis"),
            Self::TypeChecking => write!(f, "Type Checking"),
            Self::Optimization => write!(f, "Optimization"),
            Self::IrGeneration => write!(f, "IR Generation"),
            Self::Passes => write!(f, "Optimization Passes"),
            Self::Codegen => write!(f, "Code Generation"),
            Self::WcetVerification => write!(f, "WCET Verification"),
        }
    }
}

// ── Phase Timer ─────────────────────────────────────────────────────────────

/// Records wall-clock duration for a single compilation phase.
#[derive(Debug, Clone)]
pub struct PhaseTimerEntry {
    pub phase: CompilationPhase,
    pub duration: Duration,
}

/// Accumulates timing data for an entire compilation run.
#[derive(Debug, Clone, Default)]
pub struct PhaseTimer {
    entries: Vec<PhaseTimerEntry>,
    current_start: Option<(CompilationPhase, Instant)>,
}

impl PhaseTimer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Begin timing a phase. Panics (debug) if another phase is already open.
    pub fn start(&mut self, phase: CompilationPhase) {
        debug_assert!(
            self.current_start.is_none(),
            "start() called while phase {:?} is still running",
            self.current_start.as_ref().map(|s| s.0)
        );
        self.current_start = Some((phase, Instant::now()));
    }

    /// Finish the current phase and record its duration.
    pub fn finish(&mut self) -> Option<PhaseTimerEntry> {
        let (phase, start) = self.current_start.take()?;
        let entry = PhaseTimerEntry {
            phase,
            duration: start.elapsed(),
        };
        self.entries.push(entry.clone());
        Some(entry)
    }

    /// Return all recorded entries.
    pub fn entries(&self) -> &[PhaseTimerEntry] {
        &self.entries
    }

    /// Total compilation wall-clock time.
    pub fn total_duration(&self) -> Duration {
        self.entries.iter().map(|e| e.duration).sum()
    }

    /// Format a human-readable timing report.
    pub fn report(&self) -> String {
        let total = self.total_duration();
        let mut out = String::from("Phase Timing Report\n");
        out.push_str(&"─".repeat(50));
        out.push('\n');
        for entry in &self.entries {
            let pct = if total.as_nanos() > 0 {
                (entry.duration.as_nanos() as f64 / total.as_nanos() as f64) * 100.0
            } else {
                0.0
            };
            out.push_str(&format!(
                "  {:<25} {:>8.2?}  ({:>5.1}%)\n",
                entry.phase.to_string(),
                entry.duration,
                pct
            ));
        }
        out.push_str(&"─".repeat(50));
        out.push('\n');
        out.push_str(&format!("  {:<25} {:>8.2?}\n", "Total", total));
        out
    }
}

// ── Render Progress ─────────────────────────────────────────────────────────

/// Tracks progress during audio rendering.
#[derive(Debug, Clone)]
pub struct RenderProgress {
    pub total_samples: u64,
    pub rendered_samples: u64,
    pub start_time: Instant,
    pub sample_rate: u32,
}

impl RenderProgress {
    pub fn new(total_samples: u64, sample_rate: u32) -> Self {
        Self {
            total_samples,
            rendered_samples: 0,
            start_time: Instant::now(),
            sample_rate,
        }
    }

    /// Record that `n` more samples have been rendered.
    pub fn advance(&mut self, n: u64) {
        self.rendered_samples = (self.rendered_samples + n).min(self.total_samples);
    }

    /// Fraction complete in [0.0, 1.0].
    pub fn fraction(&self) -> f64 {
        if self.total_samples == 0 {
            return 1.0;
        }
        self.rendered_samples as f64 / self.total_samples as f64
    }

    /// Percentage complete.
    pub fn percentage(&self) -> f64 {
        self.fraction() * 100.0
    }

    /// Elapsed wall-clock time since rendering started.
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Estimated time remaining based on current throughput.
    pub fn eta(&self) -> Option<Duration> {
        if self.rendered_samples == 0 {
            return None;
        }
        let elapsed = self.elapsed().as_secs_f64();
        let rate = self.rendered_samples as f64 / elapsed;
        let remaining = (self.total_samples - self.rendered_samples) as f64;
        Some(Duration::from_secs_f64(remaining / rate))
    }

    /// Audio duration rendered so far.
    pub fn audio_duration_rendered(&self) -> Duration {
        if self.sample_rate == 0 {
            return Duration::ZERO;
        }
        Duration::from_secs_f64(self.rendered_samples as f64 / self.sample_rate as f64)
    }

    /// Total audio duration.
    pub fn total_audio_duration(&self) -> Duration {
        if self.sample_rate == 0 {
            return Duration::ZERO;
        }
        Duration::from_secs_f64(self.total_samples as f64 / self.sample_rate as f64)
    }

    /// Render a simple ASCII progress bar.
    pub fn bar(&self, width: usize) -> String {
        let filled = (self.fraction() * width as f64).round() as usize;
        let empty = width.saturating_sub(filled);
        let eta_str = self
            .eta()
            .map(|d| format!("ETA {:.1}s", d.as_secs_f64()))
            .unwrap_or_else(|| "ETA --".into());
        format!(
            "[{}{}] {:>5.1}%  {}",
            "█".repeat(filled),
            "░".repeat(empty),
            self.percentage(),
            eta_str,
        )
    }

    /// Whether rendering is complete.
    pub fn is_done(&self) -> bool {
        self.rendered_samples >= self.total_samples
    }
}

// ── Progress Reporter ───────────────────────────────────────────────────────

/// Verbosity-aware reporter that wraps phase timing and render progress.
#[derive(Debug)]
pub struct ProgressReporter {
    pub verbose: bool,
    pub quiet: bool,
    pub phase_timer: PhaseTimer,
}

impl ProgressReporter {
    pub fn new(verbose: bool, quiet: bool) -> Self {
        Self {
            verbose,
            quiet,
            phase_timer: PhaseTimer::new(),
        }
    }

    /// Called when a compilation phase begins.
    pub fn phase_start(&mut self, phase: CompilationPhase) {
        self.phase_timer.start(phase);
        if self.verbose && !self.quiet {
            eprintln!("  ▶ {}...", phase);
        }
    }

    /// Called when the current compilation phase completes.
    pub fn phase_end(&mut self) {
        if let Some(entry) = self.phase_timer.finish() {
            if self.verbose && !self.quiet {
                eprintln!("  ✓ {} ({:.2?})", entry.phase, entry.duration);
            }
        }
    }

    /// Print a single render-progress update line (overwrites previous).
    pub fn render_tick(&self, progress: &RenderProgress) {
        if self.quiet {
            return;
        }
        eprint!("\r  {}", progress.bar(30));
    }

    /// Print the final render-progress line.
    pub fn render_done(&self, progress: &RenderProgress) {
        if self.quiet {
            return;
        }
        eprintln!("\r  {}", progress.bar(30));
    }

    /// Print the full phase timing summary.
    pub fn print_summary(&self) {
        if self.quiet {
            return;
        }
        eprintln!("\n{}", self.phase_timer.report());
    }

    /// Return a serialisable summary of phase timings.
    pub fn timing_summary(&self) -> Vec<(String, f64)> {
        self.phase_timer
            .entries()
            .iter()
            .map(|e| (e.phase.to_string(), e.duration.as_secs_f64()))
            .collect()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_display() {
        assert_eq!(CompilationPhase::Parsing.to_string(), "Parsing");
        assert_eq!(
            CompilationPhase::WcetVerification.to_string(),
            "WCET Verification"
        );
    }

    #[test]
    fn phase_index_ordering() {
        for (i, phase) in CompilationPhase::ALL.iter().enumerate() {
            assert_eq!(phase.index(), i);
        }
    }

    #[test]
    fn phase_total() {
        assert_eq!(CompilationPhase::total(), 8);
    }

    #[test]
    fn phase_timer_record_and_report() {
        let mut timer = PhaseTimer::new();
        timer.start(CompilationPhase::Parsing);
        std::thread::sleep(Duration::from_millis(5));
        let entry = timer.finish().unwrap();
        assert_eq!(entry.phase, CompilationPhase::Parsing);
        assert!(entry.duration >= Duration::from_millis(1));
        assert_eq!(timer.entries().len(), 1);
    }

    #[test]
    fn phase_timer_total_duration() {
        let mut timer = PhaseTimer::new();
        timer.start(CompilationPhase::Parsing);
        std::thread::sleep(Duration::from_millis(5));
        timer.finish();
        timer.start(CompilationPhase::TypeChecking);
        std::thread::sleep(Duration::from_millis(5));
        timer.finish();
        assert!(timer.total_duration() >= Duration::from_millis(8));
    }

    #[test]
    fn phase_timer_report_format() {
        let mut timer = PhaseTimer::new();
        timer.start(CompilationPhase::Parsing);
        timer.finish();
        let report = timer.report();
        assert!(report.contains("Parsing"));
        assert!(report.contains("Total"));
    }

    #[test]
    fn render_progress_fraction() {
        let mut rp = RenderProgress::new(1000, 44100);
        assert_eq!(rp.fraction(), 0.0);
        rp.advance(500);
        assert!((rp.fraction() - 0.5).abs() < f64::EPSILON);
        rp.advance(500);
        assert!((rp.fraction() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn render_progress_bar() {
        let mut rp = RenderProgress::new(100, 44100);
        rp.advance(50);
        let bar = rp.bar(20);
        assert!(bar.contains("50.0%"));
    }

    #[test]
    fn render_progress_clamps() {
        let mut rp = RenderProgress::new(100, 44100);
        rp.advance(200);
        assert_eq!(rp.rendered_samples, 100);
        assert!(rp.is_done());
    }

    #[test]
    fn render_progress_audio_duration() {
        let rp = RenderProgress::new(44100, 44100);
        let dur = rp.total_audio_duration();
        assert!((dur.as_secs_f64() - 1.0).abs() < 0.01);
    }

    #[test]
    fn render_progress_eta_none_at_start() {
        let rp = RenderProgress::new(44100, 44100);
        assert!(rp.eta().is_none());
    }

    #[test]
    fn progress_reporter_timing_summary() {
        let mut reporter = ProgressReporter::new(false, true);
        reporter.phase_start(CompilationPhase::Parsing);
        reporter.phase_end();
        let summary = reporter.timing_summary();
        assert_eq!(summary.len(), 1);
        assert_eq!(summary[0].0, "Parsing");
    }

    #[test]
    fn render_progress_zero_total() {
        let rp = RenderProgress::new(0, 44100);
        assert_eq!(rp.fraction(), 1.0);
        assert!(rp.is_done());
    }

    #[test]
    fn render_progress_zero_sample_rate() {
        let rp = RenderProgress::new(100, 0);
        assert_eq!(rp.total_audio_duration(), Duration::ZERO);
    }
}

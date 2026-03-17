//! Output formatting and reporting for the CollusionProof CLI.
//!
//! Provides multiple output backends (JSON, text, table) for pipeline results,
//! certificates, evaluation summaries, and scenario listings.

use std::fmt;
use std::io::Write;

use serde::{Deserialize, Serialize};
use shared_types::{
    ConfidenceInterval, EvidenceBundle, EvidenceStrength, HypothesisTestResult,
    OracleAccessLevel, PriceTrajectory,
};

use crate::logging::VerbosityLevel;

// ── ResultFormatter trait ───────────────────────────────────────────────────

/// Trait for formatting pipeline results into displayable strings.
pub trait ResultFormatter {
    /// Format a detection result.
    fn format_detection(&self, result: &DetectionResultView) -> String;
    /// Format a certificate summary.
    fn format_certificate(&self, cert: &CertificateView) -> String;
    /// Format evaluation results.
    fn format_evaluation(&self, results: &EvaluationResultsView) -> String;
    /// Format a scenario table.
    fn format_scenarios(&self, scenarios: &[ScenarioView]) -> String;
    /// Format a verification result.
    fn format_verification(&self, result: &VerificationView) -> String;
}

// ── View types (decoupled from crate internals) ─────────────────────────────

/// Displayable view of a detection result.
#[derive(Debug, Clone, Serialize)]
pub struct DetectionResultView {
    pub scenario_id: String,
    pub classification: String,
    pub confidence: f64,
    pub collusion_premium: f64,
    pub evidence_strength: String,
    pub oracle_level: String,
    pub num_tests: usize,
    pub significant_tests: usize,
    pub layer_summaries: Vec<LayerSummaryView>,
}

/// Summary of a single detection layer.
#[derive(Debug, Clone, Serialize)]
pub struct LayerSummaryView {
    pub layer: String,
    pub reject_null: bool,
    pub p_value: Option<f64>,
    pub test_count: usize,
}

/// Displayable view of a certificate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateView {
    pub id: String,
    pub scenario_id: String,
    pub classification: String,
    pub confidence: f64,
    pub collusion_premium: f64,
    pub timestamp: String,
    pub hash: String,
    pub num_evidence: usize,
    pub num_tests: usize,
}

/// Displayable view of evaluation results.
#[derive(Debug, Clone, Serialize)]
pub struct EvaluationResultsView {
    pub mode: String,
    pub total_scenarios: usize,
    pub total_runs: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub type1_error: f64,
    pub power: f64,
    pub per_class: Vec<ClassMetricsView>,
    pub per_scenario: Vec<ScenarioResultView>,
    pub baseline_comparisons: Vec<BaselineComparisonView>,
    pub duration_secs: f64,
}

/// Per-class metrics view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassMetricsView {
    pub class: String,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub support: usize,
}

/// Per-scenario result view.
#[derive(Debug, Clone, Serialize)]
pub struct ScenarioResultView {
    pub scenario_id: String,
    pub ground_truth: String,
    pub predicted: String,
    pub correct: bool,
    pub confidence: f64,
    pub collusion_premium: f64,
}

/// Baseline comparison view.
#[derive(Debug, Clone, Serialize)]
pub struct BaselineComparisonView {
    pub name: String,
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

/// Scenario listing view.
#[derive(Debug, Clone, Serialize)]
pub struct ScenarioView {
    pub id: String,
    pub name: String,
    pub description: String,
    pub market_type: String,
    pub num_players: usize,
    pub algorithm: String,
    pub ground_truth: String,
    pub difficulty: String,
    pub num_rounds: usize,
}

/// Verification result view.
#[derive(Debug, Clone, Serialize)]
pub struct VerificationView {
    pub valid: bool,
    pub hash_valid: bool,
    pub tests_consistent: bool,
    pub evidence_sufficient: bool,
    pub issues: Vec<String>,
}

// ── JSON formatter ──────────────────────────────────────────────────────────

/// Outputs results as pretty-printed JSON.
pub struct JsonFormatter;

impl ResultFormatter for JsonFormatter {
    fn format_detection(&self, result: &DetectionResultView) -> String {
        serde_json::to_string_pretty(result).unwrap_or_else(|e| format!("JSON error: {}", e))
    }

    fn format_certificate(&self, cert: &CertificateView) -> String {
        serde_json::to_string_pretty(cert).unwrap_or_else(|e| format!("JSON error: {}", e))
    }

    fn format_evaluation(&self, results: &EvaluationResultsView) -> String {
        serde_json::to_string_pretty(results).unwrap_or_else(|e| format!("JSON error: {}", e))
    }

    fn format_scenarios(&self, scenarios: &[ScenarioView]) -> String {
        serde_json::to_string_pretty(scenarios).unwrap_or_else(|e| format!("JSON error: {}", e))
    }

    fn format_verification(&self, result: &VerificationView) -> String {
        serde_json::to_string_pretty(result).unwrap_or_else(|e| format!("JSON error: {}", e))
    }
}

// ── Text formatter ──────────────────────────────────────────────────────────

/// Human-readable text output.
pub struct TextFormatter {
    pub use_color: bool,
}

impl TextFormatter {
    pub fn new(use_color: bool) -> Self {
        Self { use_color }
    }

    fn color(&self, code: &str, text: &str) -> String {
        if self.use_color {
            format!("{}{}\x1b[0m", code, text)
        } else {
            text.to_string()
        }
    }

    fn green(&self, text: &str) -> String {
        self.color("\x1b[32m", text)
    }

    fn red(&self, text: &str) -> String {
        self.color("\x1b[31m", text)
    }

    fn yellow(&self, text: &str) -> String {
        self.color("\x1b[33m", text)
    }

    fn bold(&self, text: &str) -> String {
        self.color("\x1b[1m", text)
    }

    fn dim(&self, text: &str) -> String {
        self.color("\x1b[2m", text)
    }

    fn classify_color(&self, classification: &str) -> String {
        match classification {
            "Collusive" => self.red(classification),
            "Competitive" => self.green(classification),
            _ => self.yellow(classification),
        }
    }
}

impl ResultFormatter for TextFormatter {
    fn format_detection(&self, result: &DetectionResultView) -> String {
        let mut s = String::new();
        s.push_str(&self.bold("═══ Detection Result ═══\n"));
        s.push_str(&format!("  Scenario:          {}\n", result.scenario_id));
        s.push_str(&format!(
            "  Classification:    {}\n",
            self.classify_color(&result.classification)
        ));
        s.push_str(&format!("  Confidence:        {:.1}%\n", result.confidence * 100.0));
        s.push_str(&format!("  Collusion Premium: {:.4}\n", result.collusion_premium));
        s.push_str(&format!("  Evidence Strength: {}\n", result.evidence_strength));
        s.push_str(&format!("  Oracle Level:      {}\n", result.oracle_level));
        s.push_str(&format!(
            "  Tests:             {}/{} significant\n",
            result.significant_tests, result.num_tests
        ));
        s.push('\n');

        for layer in &result.layer_summaries {
            let status = if layer.reject_null {
                self.red("REJECT")
            } else {
                self.green("ACCEPT")
            };
            let pval = layer
                .p_value
                .map(|p| format!("{:.4}", p))
                .unwrap_or_else(|| "N/A".into());
            s.push_str(&format!(
                "  {} — {} (p={}, {} tests)\n",
                layer.layer, status, pval, layer.test_count
            ));
        }
        s
    }

    fn format_certificate(&self, cert: &CertificateView) -> String {
        let mut s = String::new();
        s.push_str(&self.bold("═══ Certificate Summary ═══\n"));
        s.push_str(&format!("  ID:                {}\n", cert.id));
        s.push_str(&format!("  Scenario:          {}\n", cert.scenario_id));
        s.push_str(&format!(
            "  Classification:    {}\n",
            self.classify_color(&cert.classification)
        ));
        s.push_str(&format!("  Confidence:        {:.1}%\n", cert.confidence * 100.0));
        s.push_str(&format!("  Collusion Premium: {:.4}\n", cert.collusion_premium));
        s.push_str(&format!("  Timestamp:         {}\n", cert.timestamp));
        s.push_str(&format!("  Hash:              {}\n", &cert.hash[..16.min(cert.hash.len())]));
        s.push_str(&format!("  Evidence Items:    {}\n", cert.num_evidence));
        s.push_str(&format!("  Statistical Tests: {}\n", cert.num_tests));
        s
    }

    fn format_evaluation(&self, results: &EvaluationResultsView) -> String {
        let mut s = String::new();
        s.push_str(&self.bold("═══ Evaluation Results ═══\n"));
        s.push_str(&format!("  Mode:       {}\n", results.mode));
        s.push_str(&format!("  Scenarios:  {}\n", results.total_scenarios));
        s.push_str(&format!("  Total Runs: {}\n", results.total_runs));
        s.push_str(&format!("  Duration:   {:.1}s\n\n", results.duration_secs));

        s.push_str(&self.bold("  Aggregate Metrics:\n"));
        s.push_str(&format!("    Precision:    {:.3}\n", results.precision));
        s.push_str(&format!("    Recall:       {:.3}\n", results.recall));
        s.push_str(&format!("    F1 Score:     {:.3}\n", results.f1_score));
        s.push_str(&format!("    Type I Error: {:.3}\n", results.type1_error));
        s.push_str(&format!("    Power:        {:.3}\n\n", results.power));

        s.push_str(&self.bold("  Per-Class Metrics:\n"));
        s.push_str(&format!(
            "    {:15} {:>9} {:>9} {:>9} {:>7}\n",
            "Class", "Precision", "Recall", "F1", "Support"
        ));
        s.push_str(&format!("    {}\n", "-".repeat(55)));
        for c in &results.per_class {
            s.push_str(&format!(
                "    {:15} {:>9.3} {:>9.3} {:>9.3} {:>7}\n",
                c.class, c.precision, c.recall, c.f1, c.support
            ));
        }
        s.push('\n');

        if !results.per_scenario.is_empty() {
            s.push_str(&self.bold("  Per-Scenario Results:\n"));
            s.push_str(&format!(
                "    {:30} {:>12} {:>12} {:>7} {:>8}\n",
                "Scenario", "Truth", "Predicted", "Correct", "Conf"
            ));
            s.push_str(&format!("    {}\n", "-".repeat(75)));
            for r in &results.per_scenario {
                let correct = if r.correct {
                    self.green("YES")
                } else {
                    self.red("NO")
                };
                s.push_str(&format!(
                    "    {:30} {:>12} {:>12} {:>7} {:>7.1}%\n",
                    r.scenario_id,
                    r.ground_truth,
                    r.predicted,
                    correct,
                    r.confidence * 100.0
                ));
            }
            s.push('\n');
        }

        if !results.baseline_comparisons.is_empty() {
            s.push_str(&self.bold("  Baseline Comparisons:\n"));
            s.push_str(&format!(
                "    {:30} {:>9} {:>9} {:>9} {:>9}\n",
                "Baseline", "Accuracy", "Precision", "Recall", "F1"
            ));
            s.push_str(&format!("    {}\n", "-".repeat(70)));
            for b in &results.baseline_comparisons {
                s.push_str(&format!(
                    "    {:30} {:>9.3} {:>9.3} {:>9.3} {:>9.3}\n",
                    b.name, b.accuracy, b.precision, b.recall, b.f1
                ));
            }
        }
        s
    }

    fn format_scenarios(&self, scenarios: &[ScenarioView]) -> String {
        let mut s = String::new();
        s.push_str(&self.bold("═══ Available Scenarios ═══\n\n"));
        s.push_str(&format!(
            "  {:30} {:>10} {:>8} {:>12} {:>12} {:>8}\n",
            "ID", "Market", "Players", "Algorithm", "Truth", "Diff."
        ));
        s.push_str(&format!("  {}\n", "-".repeat(85)));
        for sc in scenarios {
            let truth_colored = self.classify_color(&sc.ground_truth);
            s.push_str(&format!(
                "  {:30} {:>10} {:>8} {:>12} {:>12} {:>8}\n",
                sc.id, sc.market_type, sc.num_players, sc.algorithm, truth_colored, sc.difficulty
            ));
        }
        s.push_str(&format!("\n  Total: {} scenarios\n", scenarios.len()));
        s
    }

    fn format_verification(&self, result: &VerificationView) -> String {
        let mut s = String::new();
        s.push_str(&self.bold("═══ Verification Result ═══\n"));
        let status = if result.valid {
            self.green("VALID")
        } else {
            self.red("INVALID")
        };
        s.push_str(&format!("  Status:              {}\n", status));
        s.push_str(&format!("  Hash Valid:          {}\n", fmt_bool(result.hash_valid)));
        s.push_str(&format!(
            "  Tests Consistent:    {}\n",
            fmt_bool(result.tests_consistent)
        ));
        s.push_str(&format!(
            "  Evidence Sufficient: {}\n",
            fmt_bool(result.evidence_sufficient)
        ));
        if !result.issues.is_empty() {
            s.push_str("\n  Issues:\n");
            for issue in &result.issues {
                s.push_str(&format!("    - {}\n", self.red(issue)));
            }
        }
        s
    }
}

fn fmt_bool(b: bool) -> &'static str {
    if b { "yes" } else { "NO" }
}

// ── Table formatter ─────────────────────────────────────────────────────────

/// Tabular output for terminal with aligned columns.
pub struct TableFormatter {
    pub use_color: bool,
}

impl TableFormatter {
    pub fn new(use_color: bool) -> Self {
        Self { use_color }
    }
}

impl ResultFormatter for TableFormatter {
    fn format_detection(&self, result: &DetectionResultView) -> String {
        let text = TextFormatter::new(self.use_color);
        text.format_detection(result)
    }

    fn format_certificate(&self, cert: &CertificateView) -> String {
        let text = TextFormatter::new(self.use_color);
        text.format_certificate(cert)
    }

    fn format_evaluation(&self, results: &EvaluationResultsView) -> String {
        let text = TextFormatter::new(self.use_color);
        text.format_evaluation(results)
    }

    fn format_scenarios(&self, scenarios: &[ScenarioView]) -> String {
        let text = TextFormatter::new(self.use_color);
        text.format_scenarios(scenarios)
    }

    fn format_verification(&self, result: &VerificationView) -> String {
        let text = TextFormatter::new(self.use_color);
        text.format_verification(result)
    }
}

// ── Formatter factory ───────────────────────────────────────────────────────

/// Create a formatter based on the requested output format.
pub fn create_formatter(
    format: &crate::commands::OutputFormat,
    use_color: bool,
) -> Box<dyn ResultFormatter> {
    match format {
        crate::commands::OutputFormat::Json => Box::new(JsonFormatter),
        crate::commands::OutputFormat::Text => Box::new(TextFormatter::new(use_color)),
        crate::commands::OutputFormat::Table => Box::new(TableFormatter::new(use_color)),
    }
}

// ── Progress bar ────────────────────────────────────────────────────────────

/// Terminal progress bar for long-running operations.
pub struct ProgressBar {
    total: usize,
    current: usize,
    width: usize,
    label: String,
    use_color: bool,
}

impl ProgressBar {
    pub fn new(total: usize, label: impl Into<String>) -> Self {
        Self {
            total,
            current: 0,
            width: 40,
            label: label.into(),
            use_color: true,
        }
    }

    pub fn with_color(mut self, use_color: bool) -> Self {
        self.use_color = use_color;
        self
    }

    pub fn with_width(mut self, width: usize) -> Self {
        self.width = width;
        self
    }

    /// Advance by one step.
    pub fn tick(&mut self) {
        self.current = (self.current + 1).min(self.total);
        self.render();
    }

    /// Set to a specific position.
    pub fn set(&mut self, position: usize) {
        self.current = position.min(self.total);
        self.render();
    }

    /// Mark as complete.
    pub fn finish(&mut self) {
        self.current = self.total;
        self.render();
        eprint!("\n");
    }

    /// Finish with a message.
    pub fn finish_with_message(&mut self, msg: &str) {
        self.current = self.total;
        eprint!("\r{}: {} {}\n", self.label, self.bar_string(), msg);
    }

    fn render(&self) {
        let pct = if self.total > 0 {
            self.current as f64 / self.total as f64
        } else {
            0.0
        };
        eprint!(
            "\r{}: {} {}/{}  ({:.0}%)",
            self.label,
            self.bar_string(),
            self.current,
            self.total,
            pct * 100.0
        );
    }

    fn bar_string(&self) -> String {
        let pct = if self.total > 0 {
            self.current as f64 / self.total as f64
        } else {
            0.0
        };
        let filled = (pct * self.width as f64) as usize;
        let empty = self.width.saturating_sub(filled);
        let bar = format!("[{}{}]", "█".repeat(filled), "░".repeat(empty));
        if self.use_color {
            if pct >= 1.0 {
                format!("\x1b[32m{}\x1b[0m", bar)
            } else {
                format!("\x1b[33m{}\x1b[0m", bar)
            }
        } else {
            bar
        }
    }

    pub fn progress_fraction(&self) -> f64 {
        if self.total > 0 {
            self.current as f64 / self.total as f64
        } else {
            0.0
        }
    }

    pub fn is_complete(&self) -> bool {
        self.current >= self.total
    }
}

// ── Color output helper ─────────────────────────────────────────────────────

/// Helper for colored terminal output.
pub struct ColorOutput {
    enabled: bool,
}

impl ColorOutput {
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    pub fn detect() -> Self {
        let enabled = std::env::var("NO_COLOR").is_err() && atty_stdout();
        Self { enabled }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn green(&self, text: &str) -> String {
        if self.enabled {
            format!("\x1b[32m{}\x1b[0m", text)
        } else {
            text.to_string()
        }
    }

    pub fn red(&self, text: &str) -> String {
        if self.enabled {
            format!("\x1b[31m{}\x1b[0m", text)
        } else {
            text.to_string()
        }
    }

    pub fn yellow(&self, text: &str) -> String {
        if self.enabled {
            format!("\x1b[33m{}\x1b[0m", text)
        } else {
            text.to_string()
        }
    }

    pub fn bold(&self, text: &str) -> String {
        if self.enabled {
            format!("\x1b[1m{}\x1b[0m", text)
        } else {
            text.to_string()
        }
    }

    pub fn dim(&self, text: &str) -> String {
        if self.enabled {
            format!("\x1b[2m{}\x1b[0m", text)
        } else {
            text.to_string()
        }
    }

    pub fn cyan(&self, text: &str) -> String {
        if self.enabled {
            format!("\x1b[36m{}\x1b[0m", text)
        } else {
            text.to_string()
        }
    }
}

/// Best-effort check if stdout is a TTY.
fn atty_stdout() -> bool {
    // Simplified check - actual implementation would use libc::isatty
    std::env::var("TERM").is_ok()
}

// ── Convenience formatting functions ────────────────────────────────────────

/// Format a trajectory summary for display.
pub fn format_trajectory_summary(traj: &PriceTrajectory) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "Trajectory: {} rounds, {} players\n",
        traj.len(),
        traj.num_players
    ));
    for p in 0..traj.num_players {
        let prices: Vec<f64> = traj.prices_for_player(shared_types::PlayerId::new(p))
            .iter().map(|pr| pr.0).collect();
        let mean = prices.iter().sum::<f64>() / prices.len().max(1) as f64;
        let min = prices.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        s.push_str(&format!(
            "  Player {}: mean={:.3}, min={:.3}, max={:.3}\n",
            p, mean, min, max
        ));
    }
    s
}

/// Format an evidence bundle summary.
pub fn format_evidence_summary(bundle: &EvidenceBundle) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "Evidence Bundle: {} items, overall strength: {:?}\n",
        bundle.items.len(),
        bundle.overall_strength
    ));
    for item in &bundle.items {
        s.push_str(&format!("  [{:?}] {}\n", item.strength(), item.description()));
    }
    s
}

/// Format a duration in human-readable form.
pub fn format_duration(secs: f64) -> String {
    if secs < 1.0 {
        format!("{:.0}ms", secs * 1000.0)
    } else if secs < 60.0 {
        format!("{:.1}s", secs)
    } else if secs < 3600.0 {
        format!("{:.0}m {:.0}s", secs / 60.0, secs % 60.0)
    } else {
        format!("{:.0}h {:.0}m", secs / 3600.0, (secs % 3600.0) / 60.0)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_detection() -> DetectionResultView {
        DetectionResultView {
            scenario_id: "bertrand_qlearning_2p".into(),
            classification: "Collusive".into(),
            confidence: 0.95,
            collusion_premium: 0.82,
            evidence_strength: "Strong".into(),
            oracle_level: "Layer0".into(),
            num_tests: 5,
            significant_tests: 3,
            layer_summaries: vec![LayerSummaryView {
                layer: "Layer0".into(),
                reject_null: true,
                p_value: Some(0.003),
                test_count: 5,
            }],
        }
    }

    fn sample_certificate() -> CertificateView {
        CertificateView {
            id: "cert-123".into(),
            scenario_id: "test".into(),
            classification: "Competitive".into(),
            confidence: 0.85,
            collusion_premium: 0.1,
            timestamp: "2024-01-01T00:00:00Z".into(),
            hash: "abcdef1234567890".into(),
            num_evidence: 3,
            num_tests: 5,
        }
    }

    fn sample_evaluation() -> EvaluationResultsView {
        EvaluationResultsView {
            mode: "smoke".into(),
            total_scenarios: 5,
            total_runs: 10,
            precision: 0.9,
            recall: 0.85,
            f1_score: 0.87,
            type1_error: 0.05,
            power: 0.8,
            per_class: vec![ClassMetricsView {
                class: "Collusive".into(),
                precision: 0.9,
                recall: 0.85,
                f1: 0.87,
                support: 5,
            }],
            per_scenario: vec![ScenarioResultView {
                scenario_id: "test".into(),
                ground_truth: "Collusive".into(),
                predicted: "Collusive".into(),
                correct: true,
                confidence: 0.95,
                collusion_premium: 0.8,
            }],
            baseline_comparisons: vec![BaselineComparisonView {
                name: "PriceCorrelation".into(),
                accuracy: 0.7,
                precision: 0.65,
                recall: 0.6,
                f1: 0.62,
            }],
            duration_secs: 12.5,
        }
    }

    #[test]
    fn test_json_formatter_detection() {
        let fmt = JsonFormatter;
        let result = sample_detection();
        let json = fmt.format_detection(&result);
        assert!(json.contains("bertrand_qlearning_2p"));
        assert!(json.contains("Collusive"));
    }

    #[test]
    fn test_json_formatter_certificate() {
        let fmt = JsonFormatter;
        let cert = sample_certificate();
        let json = fmt.format_certificate(&cert);
        assert!(json.contains("cert-123"));
    }

    #[test]
    fn test_json_formatter_evaluation() {
        let fmt = JsonFormatter;
        let results = sample_evaluation();
        let json = fmt.format_evaluation(&results);
        assert!(json.contains("smoke"));
        assert!(json.contains("precision"));
    }

    #[test]
    fn test_text_formatter_detection() {
        let fmt = TextFormatter::new(false);
        let result = sample_detection();
        let text = fmt.format_detection(&result);
        assert!(text.contains("Collusive"));
        assert!(text.contains("95.0%"));
        assert!(text.contains("3/5"));
    }

    #[test]
    fn test_text_formatter_certificate() {
        let fmt = TextFormatter::new(false);
        let cert = sample_certificate();
        let text = fmt.format_certificate(&cert);
        assert!(text.contains("Competitive"));
        assert!(text.contains("cert-123"));
    }

    #[test]
    fn test_text_formatter_evaluation() {
        let fmt = TextFormatter::new(false);
        let results = sample_evaluation();
        let text = fmt.format_evaluation(&results);
        assert!(text.contains("Precision"));
        assert!(text.contains("Recall"));
        assert!(text.contains("PriceCorrelation"));
    }

    #[test]
    fn test_text_formatter_scenarios() {
        let fmt = TextFormatter::new(false);
        let scenarios = vec![ScenarioView {
            id: "test_scenario".into(),
            name: "Test".into(),
            description: "A test scenario".into(),
            market_type: "Bertrand".into(),
            num_players: 2,
            algorithm: "Q-Learning".into(),
            ground_truth: "Collusive".into(),
            difficulty: "Easy".into(),
            num_rounds: 1000,
        }];
        let text = fmt.format_scenarios(&scenarios);
        assert!(text.contains("test_scenario"));
        assert!(text.contains("Total: 1 scenarios"));
    }

    #[test]
    fn test_text_formatter_verification_valid() {
        let fmt = TextFormatter::new(false);
        let result = VerificationView {
            valid: true,
            hash_valid: true,
            tests_consistent: true,
            evidence_sufficient: true,
            issues: vec![],
        };
        let text = fmt.format_verification(&result);
        assert!(text.contains("VALID"));
    }

    #[test]
    fn test_text_formatter_verification_invalid() {
        let fmt = TextFormatter::new(false);
        let result = VerificationView {
            valid: false,
            hash_valid: false,
            tests_consistent: true,
            evidence_sufficient: true,
            issues: vec!["Hash mismatch".into()],
        };
        let text = fmt.format_verification(&result);
        assert!(text.contains("INVALID"));
        assert!(text.contains("Hash mismatch"));
    }

    #[test]
    fn test_progress_bar() {
        let mut pb = ProgressBar::new(10, "Test").with_color(false);
        assert!(!pb.is_complete());
        assert_eq!(pb.progress_fraction(), 0.0);
        pb.tick();
        assert!((pb.progress_fraction() - 0.1).abs() < 0.01);
        pb.set(10);
        assert!(pb.is_complete());
    }

    #[test]
    fn test_color_output() {
        let co = ColorOutput::new(true);
        assert!(co.green("test").contains("\x1b[32m"));
        assert!(co.red("test").contains("\x1b[31m"));

        let co_off = ColorOutput::new(false);
        assert_eq!(co_off.green("test"), "test");
        assert_eq!(co_off.red("test"), "test");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(0.5), "500ms");
        assert_eq!(format_duration(5.0), "5.0s");
        assert_eq!(format_duration(90.0), "2m 30s");
        assert_eq!(format_duration(3700.0), "1h 2m");
    }

    #[test]
    fn test_format_trajectory_summary() {
        let outcomes: Vec<shared_types::MarketOutcome> = (0..10)
            .map(|r| shared_types::MarketOutcome::new(
                shared_types::RoundNumber::new(r),
                vec![
                    shared_types::PlayerAction::new(shared_types::PlayerId::new(0), shared_types::Price::new(3.0)),
                    shared_types::PlayerAction::new(shared_types::PlayerId::new(1), shared_types::Price::new(4.0)),
                ],
                vec![shared_types::Price::new(3.0), shared_types::Price::new(4.0)],
                vec![shared_types::Quantity::new(1.0), shared_types::Quantity::new(1.0)],
                vec![shared_types::Profit::new(2.0), shared_types::Profit::new(3.0)],
            ))
            .collect();
        let traj = PriceTrajectory::new(
            outcomes,
            shared_types::MarketType::Bertrand,
            2,
            shared_types::AlgorithmType::QLearning,
            42,
        );
        let summary = format_trajectory_summary(&traj);
        assert!(summary.contains("10 rounds"));
        assert!(summary.contains("2 players"));
        assert!(summary.contains("Player 0"));
    }

    #[test]
    fn test_table_formatter_delegates() {
        let fmt = TableFormatter::new(false);
        let result = sample_detection();
        let text = fmt.format_detection(&result);
        assert!(text.contains("Collusive"));
    }

    #[test]
    fn test_json_scenarios() {
        let fmt = JsonFormatter;
        let scenarios = vec![ScenarioView {
            id: "s1".into(),
            name: "Scenario 1".into(),
            description: "desc".into(),
            market_type: "Bertrand".into(),
            num_players: 2,
            algorithm: "QLearning".into(),
            ground_truth: "Collusive".into(),
            difficulty: "Easy".into(),
            num_rounds: 1000,
        }];
        let json = fmt.format_scenarios(&scenarios);
        assert!(json.contains("s1"));
    }
}

//! Output formatting for terminal display.

use std::collections::HashMap;

// ── Verbosity ───────────────────────────────────────────────────────────────

/// Verbosity level for output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerbosityLevel {
    Quiet,
    Normal,
    Verbose,
    Debug,
}

// ── Color scheme ────────────────────────────────────────────────────────────

/// ANSI color codes for different severity levels.
pub struct ColorScheme {
    pub critical: &'static str,
    pub error: &'static str,
    pub warning: &'static str,
    pub info: &'static str,
    pub success: &'static str,
    pub reset: &'static str,
    pub bold: &'static str,
    pub dim: &'static str,
}

impl ColorScheme {
    pub fn enabled() -> Self {
        Self {
            critical: "\x1b[1;31m",
            error: "\x1b[31m",
            warning: "\x1b[33m",
            info: "\x1b[36m",
            success: "\x1b[32m",
            reset: "\x1b[0m",
            bold: "\x1b[1m",
            dim: "\x1b[2m",
        }
    }

    pub fn disabled() -> Self {
        Self {
            critical: "",
            error: "",
            warning: "",
            info: "",
            success: "",
            reset: "",
            bold: "",
            dim: "",
        }
    }
}

// ── Output formatter ────────────────────────────────────────────────────────

/// Formats output for the terminal.
pub struct OutputFormatter {
    pub verbosity: VerbosityLevel,
    pub color: ColorScheme,
}

impl OutputFormatter {
    pub fn new(verbosity: VerbosityLevel) -> Self {
        Self {
            verbosity,
            color: ColorScheme::enabled(),
        }
    }

    pub fn without_color(verbosity: VerbosityLevel) -> Self {
        Self {
            verbosity,
            color: ColorScheme::disabled(),
        }
    }

    pub fn print_header(&self, title: &str) {
        if self.verbosity == VerbosityLevel::Quiet {
            return;
        }
        let bar = "─".repeat(title.len() + 4);
        println!(
            "{}┌{}┐\n│ {} │\n└{}┘{}",
            self.color.bold, bar, title, bar, self.color.reset
        );
    }

    pub fn print_info(&self, msg: &str) {
        if self.verbosity == VerbosityLevel::Quiet {
            return;
        }
        println!("{}ℹ{} {msg}", self.color.info, self.color.reset);
    }

    pub fn print_success(&self, msg: &str) {
        if self.verbosity == VerbosityLevel::Quiet {
            return;
        }
        println!("{}✓{} {msg}", self.color.success, self.color.reset);
    }

    pub fn print_warning(&self, msg: &str) {
        println!(
            "{}⚠{} {msg}",
            self.color.warning, self.color.reset
        );
    }

    pub fn print_error(&self, msg: &str) {
        eprintln!(
            "{}✗{} {msg}",
            self.color.error, self.color.reset
        );
    }

    pub fn print_critical(&self, msg: &str) {
        eprintln!(
            "{}✗✗{} {msg}",
            self.color.critical, self.color.reset
        );
    }

    pub fn print_verbose(&self, msg: &str) {
        if matches!(self.verbosity, VerbosityLevel::Verbose | VerbosityLevel::Debug) {
            println!("{}  {msg}{}", self.color.dim, self.color.reset);
        }
    }

    pub fn print_debug(&self, msg: &str) {
        if self.verbosity == VerbosityLevel::Debug {
            println!("{}[DBG]{} {msg}", self.color.dim, self.color.reset);
        }
    }
}

// ── Suspiciousness table ────────────────────────────────────────────────────

/// Format a suspiciousness ranking table.
pub fn format_suspiciousness_table(
    rankings: &[(String, f64, usize)],
    color: bool,
) -> String {
    let cs = if color {
        ColorScheme::enabled()
    } else {
        ColorScheme::disabled()
    };

    let mut out = String::new();
    let header = format!(
        "{}{:<4} {:<25} {:<12} {:<8}{}",
        cs.bold, "Rank", "Stage", "Score", "Status", cs.reset
    );
    out.push_str(&header);
    out.push('\n');
    out.push_str(&"─".repeat(52));
    out.push('\n');

    for (name, score, rank) in rankings {
        let status_color = if *score > 0.8 {
            cs.critical
        } else if *score > 0.5 {
            cs.error
        } else if *score > 0.2 {
            cs.warning
        } else {
            cs.info
        };
        let status = if *score > 0.8 {
            "CRIT"
        } else if *score > 0.5 {
            "HIGH"
        } else if *score > 0.2 {
            "MED"
        } else {
            "LOW"
        };
        out.push_str(&format!(
            "{:<4} {:<25} {:<12.4} {}{:<8}{}\n",
            rank, name, score, status_color, status, cs.reset
        ));
    }
    out
}

/// Format causal decomposition results.
pub fn format_causal_results(
    decompositions: &[(String, f64, f64, f64, String)],
    color: bool,
) -> String {
    let cs = if color {
        ColorScheme::enabled()
    } else {
        ColorScheme::disabled()
    };

    let mut out = String::new();
    out.push_str(&format!(
        "{}{:<20} {:<10} {:<10} {:<10} {:<15}{}",
        cs.bold, "Stage", "DCE", "IE", "Total", "Fault Type", cs.reset
    ));
    out.push('\n');
    out.push_str(&"─".repeat(68));
    out.push('\n');

    for (stage, dce, ie, total, ft) in decompositions {
        let ft_color = match ft.as_str() {
            "Introduction" => cs.critical,
            "Amplification" => cs.error,
            "Masking" => cs.warning,
            _ => cs.info,
        };
        out.push_str(&format!(
            "{:<20} {:<10.4} {:<10.4} {:<10.4} {}{:<15}{}\n",
            stage, dce, ie, total, ft_color, ft, cs.reset
        ));
    }
    out
}

/// Format a counterexample for display.
pub fn format_counterexample(
    original: &str,
    shrunk: &str,
    transformation: &str,
    violated_mr: &str,
    color: bool,
) -> String {
    let cs = if color {
        ColorScheme::enabled()
    } else {
        ColorScheme::disabled()
    };
    format!(
        "{}Counterexample:{}\n  Transform: {}\n  Violated:  {}\n  Original:  \"{}\"\n  {}Shrunk:    \"{}\"{}\n",
        cs.bold,
        cs.reset,
        transformation,
        violated_mr,
        original,
        cs.error,
        shrunk,
        cs.reset,
    )
}

/// Format a calibration report summary.
pub fn format_calibration_report(
    baselines: &HashMap<String, (f64, f64, f64)>,
    quality: f64,
    color: bool,
) -> String {
    let cs = if color {
        ColorScheme::enabled()
    } else {
        ColorScheme::disabled()
    };

    let mut out = format!(
        "{}Calibration Report{} (quality: {:.2})\n\n",
        cs.bold, cs.reset, quality
    );
    out.push_str(&format!(
        "{:<20} {:<10} {:<10} {:<10}\n",
        "Stage", "Mean", "Std Dev", "Threshold"
    ));
    out.push_str(&"─".repeat(52));
    out.push('\n');

    for (name, (mean, std_dev, threshold)) in baselines {
        out.push_str(&format!(
            "{:<20} {:<10.4} {:<10.4} {:<10.4}\n",
            name, mean, std_dev, threshold
        ));
    }
    out
}

/// Format a compact atlas summary.
pub fn format_atlas_summary(
    stages: &[(String, f64, String)],
    color: bool,
) -> String {
    let cs = if color {
        ColorScheme::enabled()
    } else {
        ColorScheme::disabled()
    };

    let mut out = format!("{}Atlas Summary{}\n\n", cs.bold, cs.reset);
    for (name, bfi, interp) in stages {
        let bfi_color = if *bfi > 1.2 {
            cs.error
        } else if *bfi < 0.8 {
            cs.success
        } else {
            cs.info
        };
        out.push_str(&format!(
            "  {:<20} BFI={}{:.3}{} ({})\n",
            name, bfi_color, bfi, cs.reset, interp
        ));
    }
    out
}

// ── Table formatter ─────────────────────────────────────────────────────────

/// Formats data into aligned columns.
pub struct TableFormatter {
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
    pub border: bool,
}

impl TableFormatter {
    pub fn new(headers: Vec<String>) -> Self {
        Self {
            headers,
            rows: Vec::new(),
            border: true,
        }
    }

    pub fn add_row(&mut self, row: Vec<String>) {
        self.rows.push(row);
    }

    pub fn render(&self) -> String {
        let n_cols = self.headers.len();
        let mut widths = vec![0usize; n_cols];
        for (i, h) in self.headers.iter().enumerate() {
            widths[i] = widths[i].max(h.len());
        }
        for row in &self.rows {
            for (i, cell) in row.iter().enumerate().take(n_cols) {
                widths[i] = widths[i].max(cell.len());
            }
        }

        let mut out = String::new();

        // Header
        let header_line: Vec<String> = self
            .headers
            .iter()
            .enumerate()
            .map(|(i, h)| format!("{:<width$}", h, width = widths[i]))
            .collect();
        out.push_str(&header_line.join(" | "));
        out.push('\n');

        if self.border {
            let sep: Vec<String> = widths.iter().map(|&w| "─".repeat(w)).collect();
            out.push_str(&sep.join("─┼─"));
            out.push('\n');
        }

        // Rows
        for row in &self.rows {
            let cells: Vec<String> = row
                .iter()
                .enumerate()
                .take(n_cols)
                .map(|(i, c)| format!("{:<width$}", c, width = widths[i]))
                .collect();
            out.push_str(&cells.join(" | "));
            out.push('\n');
        }

        out
    }
}

// ── Progress bar ────────────────────────────────────────────────────────────

/// Simple terminal progress bar.
pub struct ProgressBar {
    pub total: usize,
    pub current: usize,
    pub width: usize,
    pub label: String,
}

impl ProgressBar {
    pub fn new(total: usize, label: impl Into<String>) -> Self {
        Self {
            total,
            current: 0,
            width: 40,
            label: label.into(),
        }
    }

    pub fn set(&mut self, current: usize) {
        self.current = current.min(self.total);
    }

    pub fn increment(&mut self) {
        self.current = (self.current + 1).min(self.total);
    }

    pub fn render(&self) -> String {
        let frac = if self.total > 0 {
            self.current as f64 / self.total as f64
        } else {
            0.0
        };
        let filled = (frac * self.width as f64) as usize;
        let empty = self.width - filled;
        let pct = (frac * 100.0) as u32;
        format!(
            "{} [{}{}] {:>3}% ({}/{})",
            self.label,
            "█".repeat(filled),
            "░".repeat(empty),
            pct,
            self.current,
            self.total
        )
    }

    pub fn display(&self) {
        print!("\r{}", self.render());
    }

    pub fn finish(&self) {
        println!("\r{}", self.render());
    }
}

// ── Helper formatters ───────────────────────────────────────────────────────

/// Format a duration in human-readable form.
pub fn format_duration(millis: u64) -> String {
    if millis < 1000 {
        format!("{millis}ms")
    } else if millis < 60_000 {
        format!("{:.1}s", millis as f64 / 1000.0)
    } else {
        let mins = millis / 60_000;
        let secs = (millis % 60_000) / 1000;
        format!("{mins}m{secs}s")
    }
}

/// Format a percentage.
pub fn format_percentage(value: f64) -> String {
    format!("{:.1}%", value * 100.0)
}

/// Format a float with appropriate precision.
pub fn format_float(value: f64, precision: usize) -> String {
    format!("{:.prec$}", value, prec = precision)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(500), "500ms");
        assert_eq!(format_duration(1500), "1.5s");
        assert_eq!(format_duration(90_000), "1m30s");
    }

    #[test]
    fn test_format_percentage() {
        assert_eq!(format_percentage(0.5), "50.0%");
        assert_eq!(format_percentage(1.0), "100.0%");
        assert_eq!(format_percentage(0.0), "0.0%");
    }

    #[test]
    fn test_format_float() {
        assert_eq!(format_float(3.14159, 2), "3.14");
        assert_eq!(format_float(0.1, 4), "0.1000");
    }

    #[test]
    fn test_table_formatter() {
        let mut table = TableFormatter::new(vec!["Name".into(), "Score".into()]);
        table.add_row(vec!["parser".into(), "0.85".into()]);
        table.add_row(vec!["tokenizer".into(), "0.30".into()]);
        let rendered = table.render();
        assert!(rendered.contains("Name"));
        assert!(rendered.contains("parser"));
        assert!(rendered.contains("tokenizer"));
        assert!(rendered.contains("─"));
    }

    #[test]
    fn test_progress_bar() {
        let mut bar = ProgressBar::new(100, "Loading");
        bar.set(50);
        let rendered = bar.render();
        assert!(rendered.contains("Loading"));
        assert!(rendered.contains("50%"));
        assert!(rendered.contains("50/100"));
    }

    #[test]
    fn test_progress_bar_increment() {
        let mut bar = ProgressBar::new(10, "Test");
        for _ in 0..10 {
            bar.increment();
        }
        assert_eq!(bar.current, 10);
        let rendered = bar.render();
        assert!(rendered.contains("100%"));
    }

    #[test]
    fn test_suspiciousness_table() {
        let rankings = vec![
            ("parser".to_string(), 0.85, 1),
            ("tokenizer".to_string(), 0.30, 2),
            ("ner".to_string(), 0.10, 3),
        ];
        let table = format_suspiciousness_table(&rankings, false);
        assert!(table.contains("parser"));
        assert!(table.contains("CRIT"));
        assert!(table.contains("MED"));
    }

    #[test]
    fn test_causal_results() {
        let decomps = vec![
            ("tok".to_string(), 0.1, 0.0, 0.1, "Benign".to_string()),
            ("parser".to_string(), 0.5, 0.1, 0.6, "Introduction".to_string()),
        ];
        let table = format_causal_results(&decomps, false);
        assert!(table.contains("tok"));
        assert!(table.contains("Introduction"));
    }

    #[test]
    fn test_counterexample_format() {
        let output = format_counterexample(
            "The cat was chased by the dog.",
            "Cat chased.",
            "passive",
            "SemanticEquivalence",
            false,
        );
        assert!(output.contains("Counterexample"));
        assert!(output.contains("passive"));
        assert!(output.contains("Cat chased."));
    }

    #[test]
    fn test_calibration_report() {
        let mut baselines = HashMap::new();
        baselines.insert("tok".to_string(), (0.1, 0.05, 0.2));
        baselines.insert("parser".to_string(), (0.3, 0.1, 0.5));
        let report = format_calibration_report(&baselines, 0.92, false);
        assert!(report.contains("Calibration Report"));
        assert!(report.contains("0.92"));
    }

    #[test]
    fn test_atlas_summary() {
        let stages = vec![
            ("tok".to_string(), 0.5, "Absorbing".to_string()),
            ("parser".to_string(), 2.5, "Amplifying".to_string()),
        ];
        let summary = format_atlas_summary(&stages, false);
        assert!(summary.contains("Atlas Summary"));
        assert!(summary.contains("tok"));
        assert!(summary.contains("Amplifying"));
    }

    #[test]
    fn test_output_formatter_quiet() {
        let fmt = OutputFormatter::new(VerbosityLevel::Quiet);
        // Quiet mode should not panic when printing
        fmt.print_info("should not print");
        fmt.print_success("should not print");
    }

    #[test]
    fn test_color_scheme_disabled() {
        let cs = ColorScheme::disabled();
        assert!(cs.critical.is_empty());
        assert!(cs.reset.is_empty());
    }
}

//! Output formatting for the BiCut CLI.
//!
//! Provides JSON output mode, human-readable output, table formatting,
//! text-based progress bars, solution display, statistics summaries,
//! and colour support detection.

use bicut_types::{
    BilevelProblem, ConstraintSense, LpProblem, LpSolution, LpStatus, ValidInequality,
};
use serde::Serialize;
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::io::{self, Write};

// ── Colour support ─────────────────────────────────────────────────

/// Detect whether the current terminal supports ANSI colour.
pub fn supports_color() -> bool {
    if std::env::var("NO_COLOR").is_ok() {
        return false;
    }
    if let Ok(term) = std::env::var("TERM") {
        if term == "dumb" {
            return false;
        }
    }
    // On Unix, check if stdout is a TTY via isatty-style heuristic.
    #[cfg(unix)]
    {
        unsafe { libc_isatty(1) }
    }
    #[cfg(not(unix))]
    {
        false
    }
}

#[cfg(unix)]
unsafe fn libc_isatty(fd: i32) -> bool {
    extern "C" {
        fn isatty(fd: i32) -> i32;
    }
    unsafe { isatty(fd) != 0 }
}

/// ANSI styling helpers.
pub struct Style {
    pub enabled: bool,
}

impl Style {
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    pub fn bold(&self, s: &str) -> String {
        if self.enabled {
            format!("\x1b[1m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }

    pub fn green(&self, s: &str) -> String {
        if self.enabled {
            format!("\x1b[32m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }

    pub fn red(&self, s: &str) -> String {
        if self.enabled {
            format!("\x1b[31m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }

    pub fn yellow(&self, s: &str) -> String {
        if self.enabled {
            format!("\x1b[33m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }

    pub fn cyan(&self, s: &str) -> String {
        if self.enabled {
            format!("\x1b[36m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }

    pub fn dim(&self, s: &str) -> String {
        if self.enabled {
            format!("\x1b[2m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }
}

// ── Table formatting ───────────────────────────────────────────────

/// A simple column-aligned text table.
pub struct Table {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    alignments: Vec<Alignment>,
}

/// Column alignment.
#[derive(Clone, Copy)]
pub enum Alignment {
    Left,
    Right,
    Center,
}

impl Table {
    pub fn new(headers: Vec<String>) -> Self {
        let n = headers.len();
        Self {
            headers,
            rows: Vec::new(),
            alignments: vec![Alignment::Left; n],
        }
    }

    pub fn set_alignment(&mut self, col: usize, align: Alignment) {
        if col < self.alignments.len() {
            self.alignments[col] = align;
        }
    }

    pub fn add_row(&mut self, row: Vec<String>) {
        // Pad or truncate to match header count.
        let mut r = row;
        r.resize(self.headers.len(), String::new());
        self.rows.push(r);
    }

    /// Compute the width of each column.
    fn column_widths(&self) -> Vec<usize> {
        let mut widths: Vec<usize> = self.headers.iter().map(|h| h.len()).collect();
        for row in &self.rows {
            for (i, cell) in row.iter().enumerate() {
                if i < widths.len() {
                    widths[i] = widths[i].max(cell.len());
                }
            }
        }
        widths
    }

    /// Render the table into a string.
    pub fn render(&self) -> String {
        let widths = self.column_widths();
        let mut buf = String::new();

        // Header row.
        self.render_row(&self.headers, &widths, &mut buf);

        // Separator.
        let sep: Vec<String> = widths.iter().map(|&w| "-".repeat(w)).collect();
        self.render_row(&sep, &widths, &mut buf);

        // Data rows.
        for row in &self.rows {
            self.render_row(row, &widths, &mut buf);
        }
        buf
    }

    fn render_row(&self, cells: &[String], widths: &[usize], buf: &mut String) {
        let parts: Vec<String> = cells
            .iter()
            .enumerate()
            .map(|(i, cell)| {
                let w = widths.get(i).copied().unwrap_or(cell.len());
                let align = self.alignments.get(i).copied().unwrap_or(Alignment::Left);
                align_cell(cell, w, align)
            })
            .collect();
        let _ = writeln!(buf, "  {}", parts.join("  "));
    }
}

fn align_cell(s: &str, width: usize, align: Alignment) -> String {
    let w = width.max(s.len());
    match align {
        Alignment::Left => format!("{:<w$}", s),
        Alignment::Right => format!("{:>w$}", s),
        Alignment::Center => {
            let pad = w.saturating_sub(s.len());
            let left = pad / 2;
            let right = pad - left;
            format!("{}{}{}", " ".repeat(left), s, " ".repeat(right))
        }
    }
}

// ── Progress bar ───────────────────────────────────────────────────

/// A simple text-based progress bar that writes to stderr.
pub struct ProgressBar {
    total: usize,
    current: usize,
    width: usize,
    label: String,
    finished: bool,
}

impl ProgressBar {
    pub fn new(total: usize, label: &str) -> Self {
        Self {
            total: total.max(1),
            current: 0,
            width: 40,
            label: label.to_string(),
            finished: false,
        }
    }

    pub fn with_width(mut self, w: usize) -> Self {
        self.width = w.max(10);
        self
    }

    /// Increment by one and redraw.
    pub fn inc(&mut self) {
        self.set(self.current + 1);
    }

    /// Set the current position and redraw.
    pub fn set(&mut self, pos: usize) {
        self.current = pos.min(self.total);
        self.draw();
    }

    /// Mark the progress bar as finished.
    pub fn finish(&mut self) {
        self.current = self.total;
        self.finished = true;
        self.draw();
        eprintln!();
    }

    /// Finish and clear the line.
    pub fn finish_and_clear(&mut self) {
        self.finished = true;
        eprint!("\r{}\r", " ".repeat(self.width + self.label.len() + 20));
    }

    fn draw(&self) {
        let frac = self.current as f64 / self.total as f64;
        let filled = (frac * self.width as f64).round() as usize;
        let empty = self.width.saturating_sub(filled);
        let pct = (frac * 100.0).min(100.0);
        let bar = format!(
            "\r{}: [{}{}] {:>5.1}% ({}/{})",
            self.label,
            "#".repeat(filled),
            ".".repeat(empty),
            pct,
            self.current,
            self.total
        );
        let _ = eprint!("{bar}");
        let _ = io::stderr().flush();
    }

    /// Build a snapshot of the current state (for testing).
    pub fn snapshot(&self) -> ProgressSnapshot {
        ProgressSnapshot {
            current: self.current,
            total: self.total,
            fraction: self.current as f64 / self.total as f64,
            finished: self.finished,
        }
    }
}

/// Immutable snapshot of progress bar state.
#[derive(Debug, Clone)]
pub struct ProgressSnapshot {
    pub current: usize,
    pub total: usize,
    pub fraction: f64,
    pub finished: bool,
}

// ── Solution display ───────────────────────────────────────────────

/// Formatted solution report.
#[derive(Debug, Clone, Serialize)]
pub struct SolutionReport {
    pub status: String,
    pub objective: f64,
    pub upper_vars: Vec<(String, f64)>,
    pub lower_vars: Vec<(String, f64)>,
    pub iterations: u64,
    pub cuts_generated: usize,
    pub wall_time_secs: f64,
}

impl SolutionReport {
    pub fn from_lp_solution(
        sol: &LpSolution,
        num_upper: usize,
        cuts: usize,
        wall_secs: f64,
    ) -> Self {
        let upper_vars: Vec<(String, f64)> = sol
            .primal
            .iter()
            .take(num_upper)
            .enumerate()
            .map(|(i, &v)| (format!("x_{i}"), v))
            .collect();
        let lower_vars: Vec<(String, f64)> = sol
            .primal
            .iter()
            .skip(num_upper)
            .enumerate()
            .map(|(i, &v)| (format!("y_{i}"), v))
            .collect();
        SolutionReport {
            status: sol.status.to_string(),
            objective: sol.objective,
            upper_vars,
            lower_vars,
            iterations: sol.iterations,
            cuts_generated: cuts,
            wall_time_secs: wall_secs,
        }
    }

    pub fn render_human(&self, style: &Style, precision: usize) -> String {
        let mut buf = String::new();
        let _ = writeln!(buf, "{}", style.bold("═══ Solution Report ═══"));
        let _ = writeln!(buf);

        let status_colored = match self.status.as_str() {
            "Optimal" => style.green(&self.status),
            "Infeasible" | "Unbounded" => style.red(&self.status),
            _ => style.yellow(&self.status),
        };
        let _ = writeln!(buf, "  Status     : {status_colored}");
        let _ = writeln!(
            buf,
            "  Objective  : {:.prec$}",
            self.objective,
            prec = precision
        );
        let _ = writeln!(buf, "  Iterations : {}", self.iterations);
        let _ = writeln!(buf, "  Cuts       : {}", self.cuts_generated);
        let _ = writeln!(buf, "  Wall time  : {:.3}s", self.wall_time_secs);
        let _ = writeln!(buf);

        if !self.upper_vars.is_empty() {
            let _ = writeln!(buf, "  {}", style.cyan("Upper-level variables:"));
            for (name, val) in &self.upper_vars {
                let _ = writeln!(buf, "    {name:>8} = {val:.prec$}", prec = precision);
            }
        }
        if !self.lower_vars.is_empty() {
            let _ = writeln!(buf, "  {}", style.cyan("Lower-level variables:"));
            for (name, val) in &self.lower_vars {
                let _ = writeln!(buf, "    {name:>8} = {val:.prec$}", prec = precision);
            }
        }
        buf
    }

    pub fn render_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    pub fn render_compact(&self, precision: usize) -> String {
        format!(
            "{} obj={:.prec$} iter={} cuts={} time={:.3}s",
            self.status,
            self.objective,
            self.iterations,
            self.cuts_generated,
            self.wall_time_secs,
            prec = precision,
        )
    }
}

// ── Statistics summary ─────────────────────────────────────────────

/// Aggregate statistics for a set of benchmark runs.
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkStats {
    pub num_instances: usize,
    pub num_solved: usize,
    pub num_infeasible: usize,
    pub num_timeout: usize,
    pub total_time_secs: f64,
    pub mean_time_secs: f64,
    pub median_time_secs: f64,
    pub min_time_secs: f64,
    pub max_time_secs: f64,
    pub mean_iterations: f64,
    pub total_cuts: usize,
}

impl BenchmarkStats {
    /// Compute statistics from a list of (time, iterations, cuts, status) tuples.
    pub fn compute(results: &[(f64, u64, usize, String)]) -> Self {
        let n = results.len();
        if n == 0 {
            return Self {
                num_instances: 0,
                num_solved: 0,
                num_infeasible: 0,
                num_timeout: 0,
                total_time_secs: 0.0,
                mean_time_secs: 0.0,
                median_time_secs: 0.0,
                min_time_secs: 0.0,
                max_time_secs: 0.0,
                mean_iterations: 0.0,
                total_cuts: 0,
            };
        }

        let mut times: Vec<f64> = results.iter().map(|r| r.0).collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let total_time: f64 = times.iter().sum();
        let total_iter: u64 = results.iter().map(|r| r.1).sum();
        let total_cuts: usize = results.iter().map(|r| r.2).sum();
        let num_solved = results.iter().filter(|r| r.3 == "Optimal").count();
        let num_infeasible = results.iter().filter(|r| r.3 == "Infeasible").count();
        let num_timeout = results.iter().filter(|r| r.3 == "IterationLimit").count();

        let median = if n % 2 == 1 {
            times[n / 2]
        } else {
            (times[n / 2 - 1] + times[n / 2]) / 2.0
        };

        Self {
            num_instances: n,
            num_solved,
            num_infeasible,
            num_timeout,
            total_time_secs: total_time,
            mean_time_secs: total_time / n as f64,
            median_time_secs: median,
            min_time_secs: times[0],
            max_time_secs: times[n - 1],
            mean_iterations: total_iter as f64 / n as f64,
            total_cuts,
        }
    }

    pub fn render_human(&self, style: &Style) -> String {
        let mut buf = String::new();
        let _ = writeln!(buf, "{}", style.bold("═══ Benchmark Summary ═══"));
        let _ = writeln!(buf);

        let mut tbl = Table::new(vec!["Metric".to_string(), "Value".to_string()]);
        tbl.set_alignment(1, Alignment::Right);
        tbl.add_row(vec![
            "Instances".to_string(),
            self.num_instances.to_string(),
        ]);
        tbl.add_row(vec!["Solved".to_string(), self.num_solved.to_string()]);
        tbl.add_row(vec![
            "Infeasible".to_string(),
            self.num_infeasible.to_string(),
        ]);
        tbl.add_row(vec!["Timeout".to_string(), self.num_timeout.to_string()]);
        tbl.add_row(vec![
            "Total time".to_string(),
            format!("{:.3}s", self.total_time_secs),
        ]);
        tbl.add_row(vec![
            "Mean time".to_string(),
            format!("{:.3}s", self.mean_time_secs),
        ]);
        tbl.add_row(vec![
            "Median time".to_string(),
            format!("{:.3}s", self.median_time_secs),
        ]);
        tbl.add_row(vec![
            "Min time".to_string(),
            format!("{:.3}s", self.min_time_secs),
        ]);
        tbl.add_row(vec![
            "Max time".to_string(),
            format!("{:.3}s", self.max_time_secs),
        ]);
        tbl.add_row(vec![
            "Mean iters".to_string(),
            format!("{:.1}", self.mean_iterations),
        ]);
        tbl.add_row(vec!["Total cuts".to_string(), self.total_cuts.to_string()]);

        buf.push_str(&tbl.render());
        buf
    }

    pub fn render_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }
}

// ── Problem display ────────────────────────────────────────────────

/// Format a bilevel problem summary for display.
pub fn format_problem_summary(prob: &BilevelProblem, style: &Style) -> String {
    let mut buf = String::new();
    let _ = writeln!(buf, "{}", style.bold("═══ Bilevel Problem Summary ═══"));
    let _ = writeln!(buf);
    let _ = writeln!(buf, "  Upper-level variables  : {}", prob.num_upper_vars);
    let _ = writeln!(buf, "  Lower-level variables  : {}", prob.num_lower_vars);
    let _ = writeln!(
        buf,
        "  Lower-level constraints: {}",
        prob.num_lower_constraints
    );
    let _ = writeln!(
        buf,
        "  Upper-level constraints: {}",
        prob.num_upper_constraints
    );
    let _ = writeln!(
        buf,
        "  Lower-level matrix nnz : {}",
        prob.lower_a.entries.len()
    );
    let _ = writeln!(
        buf,
        "  Linking matrix nnz     : {}",
        prob.lower_linking_b.entries.len()
    );

    let total_vars = prob.num_upper_vars + prob.num_lower_vars;
    let total_constraints = prob.num_lower_constraints + prob.num_upper_constraints;
    let _ = writeln!(buf);
    let _ = writeln!(buf, "  Total variables        : {}", total_vars);
    let _ = writeln!(buf, "  Total constraints      : {}", total_constraints);

    // Density of the lower-level matrix.
    let capacity = prob.num_lower_constraints * prob.num_lower_vars;
    let density = if capacity > 0 {
        prob.lower_a.entries.len() as f64 / capacity as f64 * 100.0
    } else {
        0.0
    };
    let _ = writeln!(buf, "  Lower A density        : {density:.1}%");
    buf
}

/// Format an LP problem for display.
pub fn format_lp_summary(lp: &LpProblem, name: &str, style: &Style) -> String {
    let mut buf = String::new();
    let _ = writeln!(buf, "{}", style.bold(&format!("═══ LP: {name} ═══")));
    let _ = writeln!(buf, "  Direction  : {:?}", lp.direction);
    let _ = writeln!(buf, "  Variables  : {}", lp.num_vars);
    let _ = writeln!(buf, "  Constraints: {}", lp.num_constraints);
    let _ = writeln!(buf, "  Matrix nnz : {}", lp.a_matrix.entries.len());

    let num_eq = lp
        .senses
        .iter()
        .filter(|&&s| s == ConstraintSense::Eq)
        .count();
    let num_le = lp
        .senses
        .iter()
        .filter(|&&s| s == ConstraintSense::Le)
        .count();
    let num_ge = lp
        .senses
        .iter()
        .filter(|&&s| s == ConstraintSense::Ge)
        .count();
    let _ = writeln!(buf, "  Sense dist : ≤:{num_le}  =:{num_eq}  ≥:{num_ge}");
    buf
}

/// Format a list of valid inequalities.
pub fn format_cuts_summary(cuts: &[ValidInequality], style: &Style) -> String {
    let mut buf = String::new();
    let _ = writeln!(
        buf,
        "{} ({} cuts)",
        style.bold("═══ Generated Cuts ═══"),
        cuts.len()
    );
    for (i, cut) in cuts.iter().enumerate().take(20) {
        let alpha_nnz = cut.alpha.iter().filter(|&&v| v.abs() > 1e-10).count();
        let beta_nnz = cut.beta.iter().filter(|&&v| v.abs() > 1e-10).count();
        let _ = writeln!(
            buf,
            "  cut {i:>3}: α_nnz={alpha_nnz}, β_nnz={beta_nnz}, γ={:.6}",
            cut.gamma
        );
    }
    if cuts.len() > 20 {
        let _ = writeln!(buf, "  ... and {} more", cuts.len() - 20);
    }
    buf
}

// ── Analysis report ────────────────────────────────────────────────

/// Structural analysis report for a bilevel problem.
#[derive(Debug, Clone, Serialize)]
pub struct AnalysisReport {
    pub num_upper_vars: usize,
    pub num_lower_vars: usize,
    pub num_lower_constraints: usize,
    pub num_upper_constraints: usize,
    pub lower_matrix_density: f64,
    pub linking_matrix_density: f64,
    pub lower_obj_sparsity: f64,
    pub upper_obj_sparsity: f64,
    pub has_upper_constraints: bool,
    pub is_pessimistic: bool,
    pub estimated_kkt_vars: usize,
    pub estimated_kkt_constraints: usize,
    pub structural_properties: HashMap<String, String>,
}

impl AnalysisReport {
    pub fn from_problem(prob: &BilevelProblem) -> Self {
        let lower_capacity = (prob.num_lower_constraints * prob.num_lower_vars).max(1);
        let lower_density = prob.lower_a.entries.len() as f64 / lower_capacity as f64;
        let linking_capacity = (prob.num_lower_constraints * prob.num_upper_vars).max(1);
        let linking_density = prob.lower_linking_b.entries.len() as f64 / linking_capacity as f64;
        let lower_obj_nnz = prob
            .lower_obj_c
            .iter()
            .filter(|&&v| v.abs() > 1e-10)
            .count();
        let lower_obj_sparsity = 1.0 - lower_obj_nnz as f64 / prob.num_lower_vars.max(1) as f64;
        let upper_obj_nnz = prob
            .upper_obj_c_x
            .iter()
            .chain(prob.upper_obj_c_y.iter())
            .filter(|&&v| v.abs() > 1e-10)
            .count();
        let total_upper_vars = prob.num_upper_vars + prob.num_lower_vars;
        let upper_obj_sparsity = 1.0 - upper_obj_nnz as f64 / total_upper_vars.max(1) as f64;

        // KKT reformulation size estimate.
        let kkt_vars = prob.num_upper_vars
            + prob.num_lower_vars
            + prob.num_lower_constraints  // dual multipliers
            + prob.num_lower_constraints; // complementarity binary vars
        let kkt_constraints = prob.num_upper_constraints
            + prob.num_lower_constraints   // primal feasibility
            + prob.num_lower_vars          // stationarity
            + prob.num_lower_constraints   // complementarity
            + prob.num_lower_constraints; // big-M linearisation

        let mut props = HashMap::new();
        if prob.num_upper_constraints == 0 {
            props.insert(
                "structure".to_string(),
                "simple (no upper constraints)".to_string(),
            );
        } else {
            props.insert("structure".to_string(), "general".to_string());
        }
        if linking_density < 0.1 {
            props.insert("linking".to_string(), "sparse".to_string());
        } else {
            props.insert("linking".to_string(), "dense".to_string());
        }

        Self {
            num_upper_vars: prob.num_upper_vars,
            num_lower_vars: prob.num_lower_vars,
            num_lower_constraints: prob.num_lower_constraints,
            num_upper_constraints: prob.num_upper_constraints,
            lower_matrix_density: lower_density,
            linking_matrix_density: linking_density,
            lower_obj_sparsity,
            upper_obj_sparsity,
            has_upper_constraints: prob.num_upper_constraints > 0,
            is_pessimistic: false,
            estimated_kkt_vars: kkt_vars,
            estimated_kkt_constraints: kkt_constraints,
            structural_properties: props,
        }
    }

    pub fn render_human(&self, style: &Style) -> String {
        let mut buf = String::new();
        let _ = writeln!(buf, "{}", style.bold("═══ Structural Analysis ═══"));
        let _ = writeln!(buf);

        let mut tbl = Table::new(vec!["Property".to_string(), "Value".to_string()]);
        tbl.set_alignment(1, Alignment::Right);
        tbl.add_row(vec!["Upper vars".into(), self.num_upper_vars.to_string()]);
        tbl.add_row(vec!["Lower vars".into(), self.num_lower_vars.to_string()]);
        tbl.add_row(vec![
            "Lower constraints".into(),
            self.num_lower_constraints.to_string(),
        ]);
        tbl.add_row(vec![
            "Upper constraints".into(),
            self.num_upper_constraints.to_string(),
        ]);
        tbl.add_row(vec![
            "Lower A density".into(),
            format!("{:.2}%", self.lower_matrix_density * 100.0),
        ]);
        tbl.add_row(vec![
            "Linking density".into(),
            format!("{:.2}%", self.linking_matrix_density * 100.0),
        ]);
        tbl.add_row(vec![
            "Est. KKT vars".into(),
            self.estimated_kkt_vars.to_string(),
        ]);
        tbl.add_row(vec![
            "Est. KKT constraints".into(),
            self.estimated_kkt_constraints.to_string(),
        ]);

        buf.push_str(&tbl.render());
        let _ = writeln!(buf);

        for (k, v) in &self.structural_properties {
            let _ = writeln!(buf, "  {}: {}", style.cyan(k), v);
        }
        buf
    }

    pub fn render_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }
}

// ── Certificate display ────────────────────────────────────────────

/// Display a verification result.
#[derive(Debug, Clone, Serialize)]
pub struct VerificationReport {
    pub verified: bool,
    pub objective: f64,
    pub primal_feasible: bool,
    pub dual_feasible: bool,
    pub complementarity_satisfied: bool,
    pub lower_level_optimal: bool,
    pub violation: f64,
    pub checks: Vec<(String, bool, String)>,
}

impl VerificationReport {
    pub fn render_human(&self, style: &Style) -> String {
        let mut buf = String::new();
        let verdict = if self.verified {
            style.green("VERIFIED ✓")
        } else {
            style.red("FAILED ✗")
        };
        let _ = writeln!(buf, "{} {verdict}", style.bold("═══ Verification ═══"));
        let _ = writeln!(buf, "  Objective : {:.8}", self.objective);
        let _ = writeln!(buf, "  Violation : {:.2e}", self.violation);
        let _ = writeln!(buf);

        for (name, passed, detail) in &self.checks {
            let mark = if *passed {
                style.green("✓")
            } else {
                style.red("✗")
            };
            let _ = writeln!(buf, "  {mark} {name}: {detail}");
        }
        buf
    }

    pub fn render_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }
}

// ── Compilation report ─────────────────────────────────────────────

/// Report for a compilation step.
#[derive(Debug, Clone, Serialize)]
pub struct CompilationReport {
    pub reformulation: String,
    pub original_upper_vars: usize,
    pub original_lower_vars: usize,
    pub original_lower_constraints: usize,
    pub milp_vars: usize,
    pub milp_constraints: usize,
    pub milp_binary_vars: usize,
    pub compilation_time_secs: f64,
}

impl CompilationReport {
    pub fn render_human(&self, style: &Style) -> String {
        let mut buf = String::new();
        let _ = writeln!(buf, "{}", style.bold("═══ Compilation Report ═══"));
        let _ = writeln!(buf, "  Reformulation     : {}", self.reformulation);
        let _ = writeln!(
            buf,
            "  Original upper    : {} vars",
            self.original_upper_vars
        );
        let _ = writeln!(
            buf,
            "  Original lower    : {} vars, {} constraints",
            self.original_lower_vars, self.original_lower_constraints
        );
        let _ = writeln!(
            buf,
            "  MILP vars         : {} ({} binary)",
            self.milp_vars, self.milp_binary_vars
        );
        let _ = writeln!(buf, "  MILP constraints  : {}", self.milp_constraints);
        let _ = writeln!(
            buf,
            "  Compilation time  : {:.3}s",
            self.compilation_time_secs
        );
        buf
    }

    pub fn render_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use bicut_types::SparseMatrix;

    fn make_style() -> Style {
        Style::new(false)
    }

    #[test]
    fn test_table_render() {
        let mut tbl = Table::new(vec!["Name".into(), "Value".into()]);
        tbl.add_row(vec!["alpha".into(), "1.0".into()]);
        tbl.add_row(vec!["beta".into(), "2.0".into()]);
        let rendered = tbl.render();
        assert!(rendered.contains("Name"));
        assert!(rendered.contains("alpha"));
        assert!(rendered.contains("2.0"));
    }

    #[test]
    fn test_table_alignment() {
        let mut tbl = Table::new(vec!["X".into(), "Y".into()]);
        tbl.set_alignment(0, Alignment::Right);
        tbl.set_alignment(1, Alignment::Center);
        tbl.add_row(vec!["hi".into(), "there".into()]);
        let rendered = tbl.render();
        assert!(rendered.contains("hi"));
    }

    #[test]
    fn test_progress_bar_snapshot() {
        let mut pb = ProgressBar::new(10, "test");
        pb.set(5);
        let snap = pb.snapshot();
        assert_eq!(snap.current, 5);
        assert_eq!(snap.total, 10);
        assert!((snap.fraction - 0.5).abs() < 1e-10);
        assert!(!snap.finished);
    }

    #[test]
    fn test_progress_bar_finish() {
        let mut pb = ProgressBar::new(10, "test");
        pb.finish();
        let snap = pb.snapshot();
        assert!(snap.finished);
        assert_eq!(snap.current, 10);
    }

    #[test]
    fn test_solution_report_human() {
        let sol = LpSolution {
            status: LpStatus::Optimal,
            objective: 42.5,
            primal: vec![1.0, 2.0, 3.0],
            dual: vec![0.5],
            basis: vec![],
            iterations: 10,
        };
        let report = SolutionReport::from_lp_solution(&sol, 1, 5, 0.123);
        let rendered = report.render_human(&make_style(), 4);
        assert!(rendered.contains("Optimal"));
        assert!(rendered.contains("42.5"));
        assert!(rendered.contains("x_0"));
        assert!(rendered.contains("y_0"));
    }

    #[test]
    fn test_solution_report_json() {
        let sol = LpSolution {
            status: LpStatus::Optimal,
            objective: 10.0,
            primal: vec![1.0],
            dual: vec![],
            basis: vec![],
            iterations: 3,
        };
        let report = SolutionReport::from_lp_solution(&sol, 1, 0, 0.01);
        let json = report.render_json();
        assert!(json.contains("\"status\""));
        assert!(json.contains("10.0"));
    }

    #[test]
    fn test_benchmark_stats_compute() {
        let results = vec![
            (1.0, 100, 5, "Optimal".to_string()),
            (2.0, 200, 10, "Optimal".to_string()),
            (3.0, 50, 3, "Infeasible".to_string()),
        ];
        let stats = BenchmarkStats::compute(&results);
        assert_eq!(stats.num_instances, 3);
        assert_eq!(stats.num_solved, 2);
        assert_eq!(stats.num_infeasible, 1);
        assert!((stats.mean_time_secs - 2.0).abs() < 1e-10);
        assert!((stats.median_time_secs - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_benchmark_stats_empty() {
        let stats = BenchmarkStats::compute(&[]);
        assert_eq!(stats.num_instances, 0);
        assert_eq!(stats.mean_time_secs, 0.0);
    }

    #[test]
    fn test_analysis_report_from_problem() {
        let prob = BilevelProblem {
            upper_obj_c_x: vec![1.0, 0.0],
            upper_obj_c_y: vec![0.0, 1.0],
            lower_obj_c: vec![1.0, 1.0],
            lower_a: SparseMatrix::new(2, 2),
            lower_b: vec![1.0, 1.0],
            lower_linking_b: SparseMatrix::new(2, 2),
            upper_constraints_a: SparseMatrix::new(0, 4),
            upper_constraints_b: vec![],
            num_upper_vars: 2,
            num_lower_vars: 2,
            num_lower_constraints: 2,
            num_upper_constraints: 0,
        };
        let report = AnalysisReport::from_problem(&prob);
        assert_eq!(report.num_upper_vars, 2);
        assert!(!report.has_upper_constraints);
        assert!(report.estimated_kkt_vars > 0);
    }

    #[test]
    fn test_style_disabled() {
        let style = Style::new(false);
        assert_eq!(style.bold("hello"), "hello");
        assert_eq!(style.green("ok"), "ok");
        assert_eq!(style.red("err"), "err");
    }

    #[test]
    fn test_style_enabled() {
        let style = Style::new(true);
        let bold = style.bold("x");
        assert!(bold.contains("\x1b[1m"));
        assert!(bold.contains("x"));
    }

    #[test]
    fn test_verification_report_render() {
        let report = VerificationReport {
            verified: true,
            objective: 42.0,
            primal_feasible: true,
            dual_feasible: true,
            complementarity_satisfied: true,
            lower_level_optimal: true,
            violation: 1e-12,
            checks: vec![
                ("Primal feasibility".into(), true, "ok".into()),
                ("Dual feasibility".into(), true, "ok".into()),
            ],
        };
        let rendered = report.render_human(&make_style());
        assert!(rendered.contains("VERIFIED"));
        assert!(rendered.contains("42.0"));
    }
}

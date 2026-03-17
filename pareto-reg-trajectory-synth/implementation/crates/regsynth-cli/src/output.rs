use crate::OutputFormat;
use anyhow::Result;
use serde::Serialize;
use std::io::Write;
use std::path::PathBuf;

/// Handles formatting and writing output in text, JSON, or CSV formats.
pub struct OutputFormatter {
    pub format: OutputFormat,
    pub output_path: Option<PathBuf>,
}

impl OutputFormatter {
    pub fn new(format: OutputFormat, output_path: Option<PathBuf>) -> Self {
        Self { format, output_path }
    }

    /// Write any serializable value using the configured format.
    pub fn write_value<T: Serialize + ?Sized>(&self, value: &T) -> Result<()> {
        let text = match self.format {
            OutputFormat::Json => serde_json::to_string_pretty(value)?,
            OutputFormat::Csv | OutputFormat::Text => {
                serde_json::to_string_pretty(value)?
            }
        };
        self.write_raw(&text)
    }

    /// Write raw text output to the configured destination.
    pub fn write_raw(&self, text: &str) -> Result<()> {
        if let Some(ref path) = self.output_path {
            std::fs::write(path, text)?;
            log::info!("Output written to {}", path.display());
        } else {
            print!("{}", text);
        }
        Ok(())
    }

    /// Write a line to stderr (for progress/status messages).
    pub fn status(&self, msg: &str) {
        eprintln!("{}", msg);
    }

    /// Format and write a results summary table.
    pub fn write_table(&self, headers: &[&str], rows: &[Vec<String>]) -> Result<()> {
        match self.format {
            OutputFormat::Text => {
                let text = format_text_table(headers, rows, 80);
                self.write_raw(&text)
            }
            OutputFormat::Json => {
                let json_rows: Vec<serde_json::Value> = rows
                    .iter()
                    .map(|row| {
                        let mut map = serde_json::Map::new();
                        for (i, header) in headers.iter().enumerate() {
                            map.insert(
                                header.to_string(),
                                serde_json::Value::String(
                                    row.get(i).cloned().unwrap_or_default(),
                                ),
                            );
                        }
                        serde_json::Value::Object(map)
                    })
                    .collect();
                let text = serde_json::to_string_pretty(&json_rows)?;
                self.write_raw(&text)
            }
            OutputFormat::Csv => {
                let mut out = String::new();
                out.push_str(&headers.join(","));
                out.push('\n');
                for row in rows {
                    let escaped: Vec<String> = row
                        .iter()
                        .map(|cell| {
                            if cell.contains(',') || cell.contains('"') || cell.contains('\n') {
                                format!("\"{}\"", cell.replace('"', "\"\""))
                            } else {
                                cell.clone()
                            }
                        })
                        .collect();
                    out.push_str(&escaped.join(","));
                    out.push('\n');
                }
                self.write_raw(&out)
            }
        }
    }

    /// Write solver result summary.
    pub fn write_solver_result(
        &self,
        status: &str,
        objective: Option<f64>,
        satisfied: usize,
        waived: usize,
        conflicts: &[String],
    ) -> Result<()> {
        match self.format {
            OutputFormat::Text => {
                let mut out = String::new();
                out.push_str(&format!("\n{}\n", section_header("Solver Result")));
                out.push_str(&format!("  Status:      {}\n", colorize_status(status)));
                if let Some(obj) = objective {
                    out.push_str(&format!("  Objective:   {:.6}\n", obj));
                }
                out.push_str(&format!("  Satisfied:   {}\n", satisfied));
                out.push_str(&format!("  Waived:      {}\n", waived));
                if !conflicts.is_empty() {
                    out.push_str(&format!("  Conflicts:   {}\n", conflicts.len()));
                    for c in conflicts {
                        out.push_str(&format!("    - {}\n", c));
                    }
                }
                self.write_raw(&out)
            }
            _ => {
                let result = serde_json::json!({
                    "status": status,
                    "objective": objective,
                    "satisfied_count": satisfied,
                    "waived_count": waived,
                    "conflicts": conflicts,
                });
                self.write_value(&result)
            }
        }
    }

    /// Write Pareto frontier summary.
    pub fn write_pareto_frontier(
        &self,
        strategies: &[ParetoDisplayEntry],
        objective_names: &[String],
    ) -> Result<()> {
        match self.format {
            OutputFormat::Text => {
                let mut out = String::new();
                out.push_str(&format!("\n{}\n", section_header("Pareto Frontier")));
                out.push_str(&format!(
                    "  {} non-dominated strategies over {} objectives\n\n",
                    strategies.len(),
                    objective_names.len()
                ));
                let mut headers: Vec<&str> = vec!["#", "Name"];
                let obj_name_refs: Vec<&str> = objective_names.iter().map(|s| s.as_str()).collect();
                headers.extend_from_slice(&obj_name_refs);
                headers.push("Score");

                let rows: Vec<Vec<String>> = strategies
                    .iter()
                    .enumerate()
                    .map(|(i, s)| {
                        let mut row = vec![format!("{}", i), s.name.clone()];
                        for v in &s.objectives {
                            row.push(format!("{:.4}", v));
                        }
                        row.push(format!("{:.4}", s.compliance_score));
                        row
                    })
                    .collect();

                out.push_str(&format_text_table(&headers, &rows, 100));
                self.write_raw(&out)
            }
            _ => self.write_value(strategies),
        }
    }

    /// Write roadmap summary.
    pub fn write_roadmap(
        &self,
        phases: &[PhaseDisplayEntry],
        total_days: u32,
        total_cost: f64,
    ) -> Result<()> {
        match self.format {
            OutputFormat::Text => {
                let mut out = String::new();
                out.push_str(&format!("\n{}\n", section_header("Remediation Roadmap")));
                out.push_str(&format!(
                    "  Total duration: {} days | Estimated cost: ${:.2}\n\n",
                    total_days, total_cost
                ));
                for (i, phase) in phases.iter().enumerate() {
                    out.push_str(&format!(
                        "  Phase {} — {} ({} tasks, {} days)\n",
                        i + 1,
                        phase.name,
                        phase.task_count,
                        phase.duration_days
                    ));
                    for task in &phase.tasks {
                        let status_icon = match task.status.as_str() {
                            "Completed" => "✓",
                            "InProgress" => "►",
                            "Blocked" => "✗",
                            _ => "○",
                        };
                        out.push_str(&format!(
                            "    {} {} ({:.0}d, ${:.0})\n",
                            status_icon, task.name, task.effort_days, task.cost
                        ));
                    }
                    out.push('\n');
                }
                self.write_raw(&out)
            }
            _ => self.write_value(phases),
        }
    }

    /// Write a certificate summary.
    pub fn write_certificate_summary(
        &self,
        kind: &str,
        subject: &str,
        hash: &str,
        proof_steps: usize,
        valid: bool,
    ) -> Result<()> {
        match self.format {
            OutputFormat::Text => {
                let mut out = String::new();
                out.push_str(&format!("\n{}\n", section_header("Certificate")));
                out.push_str(&format!("  Type:        {}\n", kind));
                out.push_str(&format!("  Subject:     {}\n", subject));
                out.push_str(&format!("  Hash:        {}…\n", &hash[..hash.len().min(16)]));
                out.push_str(&format!("  Proof steps: {}\n", proof_steps));
                out.push_str(&format!(
                    "  Valid:       {}\n",
                    if valid { "✓ YES" } else { "✗ NO" }
                ));
                self.write_raw(&out)
            }
            _ => {
                let result = serde_json::json!({
                    "kind": kind,
                    "subject": subject,
                    "content_hash": hash,
                    "proof_steps": proof_steps,
                    "valid": valid,
                });
                self.write_value(&result)
            }
        }
    }
}

/// Data struct for Pareto display entries.
#[derive(Debug, Clone, Serialize)]
pub struct ParetoDisplayEntry {
    pub name: String,
    pub objectives: Vec<f64>,
    pub compliance_score: f64,
    pub risk_score: f64,
}

/// Data struct for phase display entries.
#[derive(Debug, Clone, Serialize)]
pub struct PhaseDisplayEntry {
    pub name: String,
    pub task_count: usize,
    pub duration_days: i64,
    pub tasks: Vec<TaskDisplayEntry>,
}

/// Data struct for task display entries.
#[derive(Debug, Clone, Serialize)]
pub struct TaskDisplayEntry {
    pub name: String,
    pub status: String,
    pub effort_days: f64,
    pub cost: f64,
}

/// Text-based progress bar.
pub struct ProgressBar {
    total: usize,
    current: usize,
    width: usize,
    label: String,
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

    pub fn advance(&mut self, n: usize) {
        self.current = (self.current + n).min(self.total);
        self.render();
    }

    pub fn finish(&mut self) {
        self.current = self.total;
        self.render();
        eprintln!();
    }

    fn render(&self) {
        let fraction = if self.total > 0 {
            self.current as f64 / self.total as f64
        } else {
            1.0
        };
        let filled = (fraction * self.width as f64) as usize;
        let empty = self.width - filled;
        eprint!(
            "\r  {} [{}{}] {}/{}",
            self.label,
            "█".repeat(filled),
            "░".repeat(empty),
            self.current,
            self.total,
        );
        let _ = std::io::stderr().flush();
    }
}

fn format_text_table(headers: &[&str], rows: &[Vec<String>], max_width: usize) -> String {
    let num_cols = headers.len();
    let mut col_widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();

    for row in rows {
        for (i, cell) in row.iter().enumerate() {
            if i < num_cols {
                col_widths[i] = col_widths[i].max(cell.len());
            }
        }
    }

    // Clamp total width
    let total: usize = col_widths.iter().sum::<usize>() + (num_cols - 1) * 3;
    if total > max_width && num_cols > 0 {
        let excess = total - max_width;
        let max_col = col_widths
            .iter()
            .enumerate()
            .max_by_key(|(_, w)| *w)
            .map(|(i, _)| i)
            .unwrap_or(0);
        col_widths[max_col] = col_widths[max_col].saturating_sub(excess);
    }

    let mut out = String::new();

    // Header row
    let header_line: Vec<String> = headers
        .iter()
        .enumerate()
        .map(|(i, h)| format!("{:width$}", h, width = col_widths.get(i).copied().unwrap_or(10)))
        .collect();
    out.push_str(&format!("  {}\n", header_line.join(" │ ")));

    // Separator
    let sep_parts: Vec<String> = col_widths.iter().map(|w| "─".repeat(*w)).collect();
    out.push_str(&format!("  {}\n", sep_parts.join("─┼─")));

    // Data rows
    for row in rows {
        let cells: Vec<String> = (0..num_cols)
            .map(|i| {
                let cell = row.get(i).map(|s| s.as_str()).unwrap_or("");
                let w = col_widths.get(i).copied().unwrap_or(10);
                if cell.len() > w {
                    format!("{}…", &cell[..w.saturating_sub(1)])
                } else {
                    format!("{:width$}", cell, width = w)
                }
            })
            .collect();
        out.push_str(&format!("  {}\n", cells.join(" │ ")));
    }

    out
}

fn section_header(title: &str) -> String {
    let line = "═".repeat(title.len() + 4);
    format!("╔{}╗\n║  {}  ║\n╚{}╝", line, title, line)
}

fn colorize_status(status: &str) -> String {
    match status {
        "FEASIBLE" | "SAT" => format!("\x1b[32m{}\x1b[0m", status), // green
        "INFEASIBLE" | "UNSAT" => format!("\x1b[31m{}\x1b[0m", status), // red
        "TIMEOUT" => format!("\x1b[33m{}\x1b[0m", status),          // yellow
        _ => status.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_text_table() {
        let headers = vec!["Name", "Value", "Status"];
        let rows = vec![
            vec!["alpha".into(), "42".into(), "OK".into()],
            vec!["beta".into(), "99".into(), "FAIL".into()],
        ];
        let result = format_text_table(&headers, &rows, 80);
        assert!(result.contains("Name"));
        assert!(result.contains("alpha"));
        assert!(result.contains("beta"));
        assert!(result.contains("│"));
    }

    #[test]
    fn test_section_header() {
        let h = section_header("Test");
        assert!(h.contains("Test"));
        assert!(h.contains("╔"));
        assert!(h.contains("╚"));
    }

    #[test]
    fn test_colorize_status() {
        assert!(colorize_status("FEASIBLE").contains("32m"));
        assert!(colorize_status("INFEASIBLE").contains("31m"));
        assert!(colorize_status("TIMEOUT").contains("33m"));
        assert_eq!(colorize_status("OTHER"), "OTHER");
    }

    #[test]
    fn test_progress_bar() {
        let mut pb = ProgressBar::new(10, "Test");
        pb.advance(5);
        assert_eq!(pb.current, 5);
        pb.advance(10);
        assert_eq!(pb.current, 10);
    }

    #[test]
    fn test_csv_escaping() {
        let formatter = OutputFormatter::new(OutputFormat::Csv, None);
        let headers = vec!["a", "b"];
        let rows = vec![vec!["hello, world".into(), "plain".into()]];
        // Just ensure it doesn't panic
        let _ = formatter.write_table(&headers, &rows);
    }
}

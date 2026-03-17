use anyhow::{Context, Result};
use std::path::PathBuf;

use crate::config::AppConfig;
use crate::output::{OutputFormatter, ParetoDisplayEntry, PhaseDisplayEntry, TaskDisplayEntry};
use crate::pipeline::{PipelineRunner, PipelineStage};
use regsynth_solver::ComplianceResult;

/// Run the full analysis pipeline: parse → typecheck → encode → solve → pareto → plan → certify.
pub fn run(
    config: &AppConfig,
    formatter: &OutputFormatter,
    files: &[PathBuf],
    no_certify: bool,
    max_iterations: usize,
    epsilon: f64,
) -> Result<()> {
    formatter.status("╔════════════════════════════════════════╗");
    formatter.status("║   RegSynth Full Analysis Pipeline      ║");
    formatter.status("╚════════════════════════════════════════╝");
    formatter.status("");

    // Validate all input files exist
    for file in files {
        if !file.exists() {
            anyhow::bail!("Input file not found: {}", file.display());
        }
    }

    formatter.status(&format!(
        "  Input files:     {}",
        files.iter().map(|f| f.display().to_string()).collect::<Vec<_>>().join(", ")
    ));
    formatter.status(&format!("  Solver:          {}", config.solver.backend));
    formatter.status(&format!("  Timeout:         {}s", config.solver.timeout_seconds));
    formatter.status(&format!("  Pareto epsilon:  {}", epsilon));
    formatter.status(&format!("  Max iterations:  {}", max_iterations));
    formatter.status(&format!("  Certify:         {}", if no_certify { "no" } else { "yes" }));
    formatter.status("");

    let mut runner = PipelineRunner::new(config);
    let stats = runner.run_full(files, no_certify, max_iterations, epsilon)?;

    // Display results
    formatter.status("");
    formatter.status("═══ Pipeline Results ═══");
    formatter.status("");

    // Stage timing table
    let headers = vec!["Stage", "Duration", "Status"];
    let rows: Vec<Vec<String>> = runner
        .stage_results()
        .iter()
        .map(|sr| {
            vec![
                sr.stage.label().to_string(),
                format!("{}ms", sr.duration_ms),
                if sr.success { "✓".into() } else { "✗".into() },
            ]
        })
        .collect();
    formatter.write_table(&headers, &rows)?;

    // Summary statistics
    formatter.status("");
    formatter.status(&format!("  Obligations:   {}", stats.obligations_count));
    formatter.status(&format!("  Constraints:   {}", stats.constraints_count));
    formatter.status(&format!("  Frontier size: {}", stats.frontier_size));
    formatter.status(&format!("  Total time:    {}ms", stats.total_ms));

    // Display Pareto frontier if available
    if let Some(ref frontier) = runner.artifacts().pareto_frontier {
        let objective_names: Vec<String> = vec![
            "cost".into(), "time".into(), "risk".into(), "complexity".into(),
        ];
        let entries: Vec<ParetoDisplayEntry> = frontier
            .entries()
            .iter()
            .map(|entry| ParetoDisplayEntry {
                name: entry.point.name.clone(),
                objectives: entry.cost.values.clone(),
                compliance_score: entry.point.compliance_score,
                risk_score: entry.point.risk_score,
            })
            .collect();
        formatter.write_pareto_frontier(&entries, &objective_names)?;
    }

    // Display solver result
    if let Some(ref sr) = runner.artifacts().solver_result {
        match sr {
            ComplianceResult::Feasible(sol) => {
                formatter.write_solver_result(
                    "FEASIBLE",
                    Some(sol.objective_value),
                    sol.satisfied_obligations.len(),
                    sol.waived_obligations.len(),
                    &[],
                )?;
            }
            ComplianceResult::Infeasible(core) => {
                formatter.write_solver_result(
                    "INFEASIBLE",
                    None,
                    0,
                    0,
                    &[core.explanation.clone()],
                )?;
            }
            ComplianceResult::Timeout => {
                formatter.write_solver_result("TIMEOUT", None, 0, 0, &[])?;
            }
            ComplianceResult::Unknown => {
                formatter.write_solver_result("UNKNOWN", None, 0, 0, &[])?;
            }
        }
    }

    // Write full results as JSON if output file specified
    let full_results = serde_json::json!({
        "pipeline_stats": stats,
        "stage_results": runner.stage_results(),
        "artifacts": {
            "obligations": runner.artifacts().obligations,
            "solver_result": runner.artifacts().solver_result,
            "pareto_frontier": runner.artifacts().pareto_frontier,
            "diagnostics": runner.artifacts().diagnostics,
        }
    });
    formatter.write_value(&full_results)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn test_config() -> AppConfig {
        AppConfig::default()
    }

    fn test_formatter() -> OutputFormatter {
        OutputFormatter::new(crate::OutputFormat::Json, None)
    }

    #[test]
    fn test_missing_file() {
        let cfg = test_config();
        let fmt = test_formatter();
        let result = run(&cfg, &fmt, &[PathBuf::from("nonexistent.dsl")], false, 10, 0.01);
        assert!(result.is_err());
    }
}

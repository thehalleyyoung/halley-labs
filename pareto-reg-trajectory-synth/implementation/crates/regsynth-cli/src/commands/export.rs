use anyhow::{Context, Result};
use std::path::PathBuf;

use crate::config::AppConfig;
use crate::output::OutputFormatter;
use crate::OutputFormat;

/// Run the export command: read analysis results and export in the specified format.
pub fn run(
    _config: &AppConfig,
    formatter: &OutputFormatter,
    input: &PathBuf,
    detailed: bool,
) -> Result<()> {
    formatter.status("Exporting results...");

    let content = std::fs::read_to_string(input)
        .with_context(|| format!("Failed to read {}", input.display()))?;

    let data: serde_json::Value = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse JSON from {}", input.display()))?;

    match &formatter.format {
        OutputFormat::Json => export_json(formatter, &data, detailed),
        OutputFormat::Csv => export_csv(formatter, &data, detailed),
        OutputFormat::Text => export_text(formatter, &data, detailed),
    }
}

/// Export as formatted JSON.
fn export_json(formatter: &OutputFormatter, data: &serde_json::Value, detailed: bool) -> Result<()> {
    let output = if detailed {
        data.clone()
    } else {
        summarize_json(data)
    };
    formatter.write_value(&output)
}

/// Export as CSV.
fn export_csv(formatter: &OutputFormatter, data: &serde_json::Value, detailed: bool) -> Result<()> {
    let mut csv_output = String::new();

    // Extract arrays from the data for CSV export
    if let Some(obligations) = data.get("obligations").or_else(|| {
        data.get("artifacts").and_then(|a| a.get("obligations"))
    }) {
        if let Some(arr) = obligations.as_array() {
            if !arr.is_empty() {
                // Header
                let headers: Vec<String> = if let Some(obj) = arr[0].as_object() {
                    obj.keys().cloned().collect()
                } else {
                    vec!["value".into()]
                };
                csv_output.push_str(&headers.join(","));
                csv_output.push('\n');

                // Rows
                for item in arr {
                    if let Some(obj) = item.as_object() {
                        let row: Vec<String> = headers
                            .iter()
                            .map(|h| {
                                let val = obj.get(h).unwrap_or(&serde_json::Value::Null);
                                csv_escape(val)
                            })
                            .collect();
                        csv_output.push_str(&row.join(","));
                    } else {
                        csv_output.push_str(&csv_escape(item));
                    }
                    csv_output.push('\n');
                }
            }
        }
    }

    // Export frontier data if available
    if let Some(frontier) = data.get("frontier").or_else(|| {
        data.get("artifacts").and_then(|a| a.get("pareto_frontier"))
    }) {
        if let Some(strategies) = frontier.get("strategies").and_then(|s| s.as_array()) {
            if !strategies.is_empty() {
                csv_output.push_str("\n# Pareto Frontier\n");
                csv_output.push_str("name,compliance_score,risk_score,total_cost\n");
                for s in strategies {
                    let name = s.get("name").and_then(|v| v.as_str()).unwrap_or("?");
                    let compliance = s.get("compliance_score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let risk = s.get("risk_score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let cost = s
                        .get("total_cost")
                        .and_then(|v| v.get("amount"))
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    csv_output.push_str(&format!("{},{:.4},{:.4},{:.2}\n", name, compliance, risk, cost));
                }
            }
        }
    }

    // Export stage timings if available and detailed
    if detailed {
        if let Some(stats) = data.get("pipeline_stats") {
            if let Some(timings) = stats.get("stage_timings").and_then(|t| t.as_array()) {
                csv_output.push_str("\n# Stage Timings\n");
                csv_output.push_str("stage,duration_ms\n");
                for timing in timings {
                    if let Some(arr) = timing.as_array() {
                        let stage = arr.first().and_then(|v| v.as_str()).unwrap_or("?");
                        let ms = arr.get(1).and_then(|v| v.as_u64()).unwrap_or(0);
                        csv_output.push_str(&format!("{},{}\n", stage, ms));
                    }
                }
            }
        }
    }

    if csv_output.is_empty() {
        csv_output = "# No tabular data found in input\n".into();
    }

    formatter.write_raw(&csv_output)
}

/// Export as human-readable text.
fn export_text(formatter: &OutputFormatter, data: &serde_json::Value, detailed: bool) -> Result<()> {
    let mut out = String::new();

    out.push_str("═══ RegSynth Analysis Results ═══\n\n");

    // Pipeline stats
    if let Some(stats) = data.get("pipeline_stats") {
        out.push_str("Pipeline Statistics:\n");
        if let Some(total) = stats.get("total_ms").and_then(|v| v.as_u64()) {
            out.push_str(&format!("  Total time:      {}ms\n", total));
        }
        if let Some(obls) = stats.get("obligations_count").and_then(|v| v.as_u64()) {
            out.push_str(&format!("  Obligations:     {}\n", obls));
        }
        if let Some(cons) = stats.get("constraints_count").and_then(|v| v.as_u64()) {
            out.push_str(&format!("  Constraints:     {}\n", cons));
        }
        if let Some(front) = stats.get("frontier_size").and_then(|v| v.as_u64()) {
            out.push_str(&format!("  Frontier size:   {}\n", front));
        }
        out.push('\n');
    }

    // Solver result
    if let Some(sr) = data.get("result").or_else(|| {
        data.get("artifacts").and_then(|a| a.get("solver_result"))
    }) {
        out.push_str("Solver Result:\n");
        if sr.get("Feasible").is_some() {
            let sol = &sr["Feasible"];
            out.push_str("  Status:    FEASIBLE\n");
            if let Some(obj) = sol.get("objective_value").and_then(|v| v.as_f64()) {
                out.push_str(&format!("  Objective: {:.6}\n", obj));
            }
            if let Some(sat) = sol.get("satisfied_obligations").and_then(|v| v.as_array()) {
                out.push_str(&format!("  Satisfied: {}\n", sat.len()));
            }
        } else if sr.get("Infeasible").is_some() {
            out.push_str("  Status: INFEASIBLE\n");
            if let Some(expl) = sr["Infeasible"].get("explanation").and_then(|v| v.as_str()) {
                out.push_str(&format!("  Explanation: {}\n", expl));
            }
        } else if sr.get("Timeout").is_some() {
            out.push_str("  Status: TIMEOUT\n");
        } else {
            out.push_str("  Status: UNKNOWN\n");
        }
        out.push('\n');
    }

    // Obligations summary
    if let Some(obligations) = data.get("obligations").or_else(|| {
        data.get("artifacts").and_then(|a| a.get("obligations"))
    }) {
        if let Some(arr) = obligations.as_array() {
            out.push_str(&format!("Obligations ({}):\n", arr.len()));
            let limit = if detailed { arr.len() } else { 10 };
            for (i, obl) in arr.iter().take(limit).enumerate() {
                let id = obl.get("id").and_then(|v| v.as_str()).unwrap_or("?");
                let kind = obl.get("kind").and_then(|v| v.as_str()).unwrap_or("?");
                let jurisdiction = obl.get("jurisdiction").and_then(|v| v.as_str()).unwrap_or("?");
                out.push_str(&format!(
                    "  {:>3}. [{}] {} ({})\n",
                    i + 1,
                    kind,
                    id,
                    jurisdiction
                ));
            }
            if arr.len() > limit {
                out.push_str(&format!("  ... and {} more\n", arr.len() - limit));
            }
            out.push('\n');
        }
    }

    formatter.write_raw(&out)
}

/// Create a summary-only version of the JSON data.
fn summarize_json(data: &serde_json::Value) -> serde_json::Value {
    let mut summary = serde_json::Map::new();

    if let Some(stats) = data.get("pipeline_stats") {
        summary.insert("pipeline_stats".into(), stats.clone());
    }

    if let Some(sr) = data.get("result").or_else(|| {
        data.get("artifacts").and_then(|a| a.get("solver_result"))
    }) {
        let status = if sr.get("Feasible").is_some() {
            "FEASIBLE"
        } else if sr.get("Infeasible").is_some() {
            "INFEASIBLE"
        } else if sr.get("Timeout").is_some() {
            "TIMEOUT"
        } else {
            "UNKNOWN"
        };
        summary.insert("solver_status".into(), serde_json::Value::String(status.into()));
    }

    if let Some(obligations) = data.get("obligations").or_else(|| {
        data.get("artifacts").and_then(|a| a.get("obligations"))
    }) {
        if let Some(arr) = obligations.as_array() {
            summary.insert("obligations_count".into(), serde_json::Value::Number(arr.len().into()));
        }
    }

    if let Some(frontier) = data.get("frontier").or_else(|| {
        data.get("artifacts").and_then(|a| a.get("pareto_frontier"))
    }) {
        if let Some(strategies) = frontier.get("strategies").and_then(|s| s.as_array()) {
            summary.insert("frontier_size".into(), serde_json::Value::Number(strategies.len().into()));
        }
    }

    serde_json::Value::Object(summary)
}

fn csv_escape(value: &serde_json::Value) -> String {
    let s = match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Null => String::new(),
        other => other.to_string(),
    };
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csv_escape_plain() {
        assert_eq!(csv_escape(&serde_json::json!("hello")), "hello");
        assert_eq!(csv_escape(&serde_json::json!(42)), "42");
        assert_eq!(csv_escape(&serde_json::json!(true)), "true");
    }

    #[test]
    fn test_csv_escape_special() {
        assert_eq!(csv_escape(&serde_json::json!("a,b")), "\"a,b\"");
        assert_eq!(csv_escape(&serde_json::json!("a\"b")), "\"a\"\"b\"");
    }

    #[test]
    fn test_summarize_json() {
        let data = serde_json::json!({
            "pipeline_stats": {"total_ms": 100},
            "obligations": [{"id": "a"}, {"id": "b"}],
        });
        let summary = summarize_json(&data);
        assert_eq!(summary.get("obligations_count").unwrap(), 2);
    }

    #[test]
    fn test_summarize_empty() {
        let data = serde_json::json!({});
        let summary = summarize_json(&data);
        assert!(summary.is_object());
    }
}

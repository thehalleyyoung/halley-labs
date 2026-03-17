use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::PathBuf;

use regsynth_dsl::source_map::SourceMap;
use regsynth_temporal::Obligation;

use crate::config::AppConfig;
use crate::output::OutputFormatter;
use crate::pipeline;

/// Diagnostic from type checking.
#[derive(Debug, Clone, serde::Serialize)]
struct CheckDiagnostic {
    level: String,
    file: String,
    line: Option<usize>,
    message: String,
    suggestion: Option<String>,
}

/// Run the DSL type-checking command.
pub fn run(
    config: &AppConfig,
    formatter: &OutputFormatter,
    files: &[PathBuf],
    include_warnings: bool,
) -> Result<()> {
    formatter.status("Checking DSL files...");

    let mut all_obligations: Vec<Obligation> = Vec::new();
    let mut diagnostics: Vec<CheckDiagnostic> = Vec::new();
    let mut source_map = SourceMap::new();
    let mut error_count = 0usize;
    let mut warning_count = 0usize;

    for file in files {
        if !file.exists() {
            diagnostics.push(CheckDiagnostic {
                level: "error".into(),
                file: file.display().to_string(),
                line: None,
                message: format!("File not found: {}", file.display()),
                suggestion: Some("Check the file path and try again".into()),
            });
            error_count += 1;
            continue;
        }

        let source = std::fs::read_to_string(file)
            .with_context(|| format!("Failed to read {}", file.display()))?;

        let file_name = file.display().to_string();
        source_map.add_file(&file_name, &source);

        match pipeline::parse_dsl_source(&source, file) {
            Ok(obligations) => {
                // Run type-check validations on parsed obligations
                let file_diags = type_check_obligations(&obligations, &file_name);
                for d in &file_diags {
                    match d.level.as_str() {
                        "error" => error_count += 1,
                        "warning" => warning_count += 1,
                        _ => {}
                    }
                }
                diagnostics.extend(file_diags);
                all_obligations.extend(obligations);
            }
            Err(e) => {
                diagnostics.push(CheckDiagnostic {
                    level: "error".into(),
                    file: file_name,
                    line: None,
                    message: format!("Parse error: {:#}", e),
                    suggestion: None,
                });
                error_count += 1;
            }
        }
    }

    // Cross-file checks
    let cross_diags = cross_file_check(&all_obligations);
    for d in &cross_diags {
        match d.level.as_str() {
            "error" => error_count += 1,
            "warning" => warning_count += 1,
            _ => {}
        }
    }
    diagnostics.extend(cross_diags);

    // Filter warnings if not requested
    let display_diags: Vec<&CheckDiagnostic> = if include_warnings {
        diagnostics.iter().collect()
    } else {
        diagnostics.iter().filter(|d| d.level == "error").collect()
    };

    // Output results
    if display_diags.is_empty() && error_count == 0 {
        formatter.status(&format!(
            "✓ All {} obligations in {} files pass type checking",
            all_obligations.len(),
            files.len()
        ));
    } else {
        for diag in &display_diags {
            let prefix = match diag.level.as_str() {
                "error" => "ERROR",
                "warning" => "WARN ",
                _ => "INFO ",
            };
            let loc = diag
                .line
                .map(|l| format!("{}:{}", diag.file, l))
                .unwrap_or_else(|| diag.file.clone());
            formatter.status(&format!("[{}] {}: {}", prefix, loc, diag.message));
            if let Some(ref suggestion) = diag.suggestion {
                formatter.status(&format!("        suggestion: {}", suggestion));
            }
        }
    }

    formatter.status("");
    formatter.status(&format!(
        "Check complete: {} obligations, {} errors, {} warnings",
        all_obligations.len(),
        error_count,
        warning_count
    ));

    let result = serde_json::json!({
        "obligations_count": all_obligations.len(),
        "errors": error_count,
        "warnings": warning_count,
        "diagnostics": diagnostics,
        "pass": error_count == 0,
    });
    formatter.write_value(&result)?;

    if error_count > 0 {
        anyhow::bail!("Type checking failed with {} errors", error_count);
    }

    Ok(())
}

/// Type-check obligations within a single file.
fn type_check_obligations(obligations: &[Obligation], file: &str) -> Vec<CheckDiagnostic> {
    let mut diags = Vec::new();

    for obl in obligations {
        // Check for empty jurisdiction
        if obl.jurisdiction.0.is_empty() {
            diags.push(CheckDiagnostic {
                level: "error".into(),
                file: file.into(),
                line: None,
                message: format!("Obligation '{}': jurisdiction must not be empty", obl.id),
                suggestion: Some("Add a jurisdiction like EU, US, or UK".into()),
            });
        }

        // Check for empty description
        if obl.description.trim().is_empty() {
            diags.push(CheckDiagnostic {
                level: "warning".into(),
                file: file.into(),
                line: None,
                message: format!("Obligation '{}': description is empty", obl.id),
                suggestion: Some("Add a human-readable description".into()),
            });
        }

        // Warn on high-risk obligations without grade
        if obl.risk_level >= Some(regsynth_types::RiskLevel::High)
            && obl.grade >= regsynth_types::FormalizabilityGrade::F4
        {
            diags.push(CheckDiagnostic {
                level: "warning".into(),
                file: file.into(),
                line: None,
                message: format!(
                    "Obligation '{}': high-risk ({:?}) with low formalizability ({})",
                    obl.id, obl.risk_level, obl.grade
                ),
                suggestion: Some("Consider breaking this into more formalizable sub-obligations".into()),
            });
        }

        // Check that obligation ID is valid (alphanumeric + hyphens)
        if !obl.id.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_') {
            diags.push(CheckDiagnostic {
                level: "error".into(),
                file: file.into(),
                line: None,
                message: format!("Obligation '{}': ID contains invalid characters", obl.id),
                suggestion: Some("Use only alphanumeric characters, hyphens, and underscores".into()),
            });
        }
    }

    diags
}

/// Cross-file consistency checks.
fn cross_file_check(obligations: &[Obligation]) -> Vec<CheckDiagnostic> {
    let mut diags = Vec::new();
    let mut id_counts: HashMap<&str, usize> = HashMap::new();

    for obl in obligations {
        *id_counts.entry(&obl.id).or_insert(0) += 1;
    }

    for (id, count) in &id_counts {
        if *count > 1 {
            diags.push(CheckDiagnostic {
                level: "error".into(),
                file: "(cross-file)".into(),
                line: None,
                message: format!(
                    "Duplicate obligation ID '{}' appears {} times across files",
                    id, count
                ),
                suggestion: Some("Ensure each obligation has a unique ID".into()),
            });
        }
    }

    // Check for jurisdiction conflicts in same-named obligations
    let mut jurisdiction_map: HashMap<&str, Vec<&str>> = HashMap::new();
    for obl in obligations {
        jurisdiction_map
            .entry(&obl.id)
            .or_default()
            .push(&obl.jurisdiction.0);
    }

    for (id, jurisdictions) in &jurisdiction_map {
        let unique: std::collections::HashSet<&&str> = jurisdictions.iter().collect();
        if unique.len() > 1 {
            diags.push(CheckDiagnostic {
                level: "warning".into(),
                file: "(cross-file)".into(),
                line: None,
                message: format!(
                    "Obligation '{}' has conflicting jurisdictions: {:?}",
                    id,
                    jurisdictions
                ),
                suggestion: Some("Use distinct IDs for obligations in different jurisdictions".into()),
            });
        }
    }

    diags
}

#[cfg(test)]
mod tests {
    use super::*;
    use regsynth_types::*;

    #[test]
    fn test_type_check_empty_jurisdiction() {
        let obl = regsynth_temporal::Obligation::new(
            "test",
            ObligationKind::Obligation,
            Jurisdiction::new(""),
            "desc",
        );
        let diags = type_check_obligations(&[obl], "test.dsl");
        assert!(diags.iter().any(|d| d.level == "error" && d.message.contains("jurisdiction")));
    }

    #[test]
    fn test_cross_file_duplicate() {
        let obl1 = regsynth_temporal::Obligation::new("dup", ObligationKind::Obligation, Jurisdiction::new("EU"), "first");
        let obl2 = regsynth_temporal::Obligation::new("dup", ObligationKind::Obligation, Jurisdiction::new("EU"), "second");
        let diags = cross_file_check(&[obl1, obl2]);
        assert!(diags.iter().any(|d| d.message.contains("Duplicate")));
    }
}

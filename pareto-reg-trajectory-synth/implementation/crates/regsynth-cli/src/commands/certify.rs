use anyhow::{Context, Result};
use std::path::PathBuf;

use regsynth_certificate::Certificate;
use regsynth_pareto::ComplianceStrategy;
use regsynth_pareto::ParetoFrontier;
use regsynth_solver::ComplianceResult;

use crate::config::AppConfig;
use crate::output::OutputFormatter;
use crate::CertificateType;

/// Input format for certify command.
#[derive(Debug, serde::Deserialize)]
struct AnalysisInput {
    #[serde(default)]
    solver_result: Option<ComplianceResult>,
    #[serde(default)]
    pareto_frontier: Option<ParetoFrontier<ComplianceStrategy>>,
}

/// Run the certificate generation command.
pub fn run(
    _config: &AppConfig,
    formatter: &OutputFormatter,
    input: &PathBuf,
    cert_type: &CertificateType,
    subject: &str,
) -> Result<()> {
    formatter.status("Generating certificate...");

    let content = std::fs::read_to_string(input)
        .with_context(|| format!("Failed to read {}", input.display()))?;

    let analysis: AnalysisInput = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse analysis results from {}", input.display()))?;

    let cert = match cert_type {
        CertificateType::Compliance => generate_compliance_cert(&analysis, subject)?,
        CertificateType::Infeasibility => generate_infeasibility_cert(&analysis, subject)?,
        CertificateType::Pareto => generate_pareto_cert(&analysis, subject)?,
    };

    let kind_str = &cert.certificate_type;

    formatter.write_certificate_summary(
        kind_str,
        subject,
        &cert.fingerprint.hex_digest,
        0,
        cert.verify_integrity(),
    )?;

    // Write certificate to output
    let output = serde_json::json!({
        "certificate": cert,
        "verification": {
            "integrity_valid": cert.verify_integrity(),
        }
    });
    formatter.write_value(&output)?;

    formatter.status(&format!(
        "\n✓ Certificate generated: {}",
        kind_str,
    ));

    Ok(())
}

fn generate_compliance_cert(analysis: &AnalysisInput, subject: &str) -> Result<Certificate> {
    let (satisfied_count, objective) = match &analysis.solver_result {
        Some(ComplianceResult::Feasible(sol)) => {
            (sol.satisfied_obligations.len(), sol.objective_value)
        }
        _ => anyhow::bail!("Cannot generate compliance certificate: no feasible solution found"),
    };

    let payload = serde_json::json!({
        "type": "compliance",
        "subject": subject,
        "satisfied_obligations": satisfied_count,
        "objective_value": objective,
    });

    Certificate::wrap("Compliance", &payload, Vec::new())
        .map_err(|e| anyhow::anyhow!("Certificate generation failed: {}", e))
}

fn generate_infeasibility_cert(analysis: &AnalysisInput, subject: &str) -> Result<Certificate> {
    match &analysis.solver_result {
        Some(ComplianceResult::Infeasible(core)) => {
            let payload = serde_json::json!({
                "type": "infeasibility",
                "subject": subject,
                "conflict_core_size": core.size(),
                "explanation": core.explanation,
                "conflict_type": format!("{:?}", core.conflict_type),
            });

            Certificate::wrap("Infeasibility", &payload, Vec::new())
                .map_err(|e| anyhow::anyhow!("Certificate generation failed: {}", e))
        }
        _ => anyhow::bail!("Cannot generate infeasibility certificate: no conflict core found"),
    }
}

fn generate_pareto_cert(analysis: &AnalysisInput, subject: &str) -> Result<Certificate> {
    let frontier = analysis
        .pareto_frontier
        .as_ref()
        .context("Cannot generate Pareto certificate: no frontier found")?;

    let payload = serde_json::json!({
        "type": "pareto_optimality",
        "subject": subject,
        "frontier_size": frontier.size(),
        "dimension": frontier.dimension(),
    });

    Certificate::wrap("ParetoOptimality", &payload, Vec::new())
        .map_err(|e| anyhow::anyhow!("Certificate generation failed: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use regsynth_pareto::ParetoFrontier;
    use regsynth_solver::*;
    use regsynth_types::Id;

    #[test]
    fn test_generate_compliance_cert() {
        let analysis = AnalysisInput {
            solver_result: Some(ComplianceResult::Feasible(Solution {
                objective_value: 42.0,
                variable_assignments: vec![],
                satisfied_obligations: vec![Id::new(), Id::new()],
                waived_obligations: vec![],
            })),
            pareto_frontier: None,
        };
        let cert = generate_compliance_cert(&analysis, "test").unwrap();
        assert_eq!(cert.certificate_type, "Compliance");
        assert!(cert.verify_integrity());
    }

    #[test]
    fn test_generate_infeasibility_cert() {
        let analysis = AnalysisInput {
            solver_result: Some(ComplianceResult::Infeasible(ConflictCore::new(
                vec![Id::new()],
                "conflicting",
                ConflictType::LogicalContradiction,
            ))),
            pareto_frontier: None,
        };
        let cert = generate_infeasibility_cert(&analysis, "test").unwrap();
        assert_eq!(cert.certificate_type, "Infeasibility");
        assert!(cert.verify_integrity());
    }

    #[test]
    fn test_generate_pareto_cert() {
        use regsynth_pareto::CostVector;
        let mut frontier: ParetoFrontier<ComplianceStrategy> = ParetoFrontier::new(2);
        let s = ComplianceStrategy::new("s1", vec![]);
        frontier.add_point(s, CostVector::new(vec![1.0, 2.0]));
        let analysis = AnalysisInput {
            solver_result: None,
            pareto_frontier: Some(frontier),
        };
        let cert = generate_pareto_cert(&analysis, "test").unwrap();
        assert_eq!(cert.certificate_type, "ParetoOptimality");
    }

    #[test]
    fn test_compliance_cert_no_solution() {
        let analysis = AnalysisInput {
            solver_result: Some(ComplianceResult::Timeout),
            pareto_frontier: None,
        };
        assert!(generate_compliance_cert(&analysis, "test").is_err());
    }
}

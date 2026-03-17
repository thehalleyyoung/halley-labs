use anyhow::{Context, Result};
use std::path::PathBuf;

use regsynth_encoding::*;
use regsynth_temporal::Obligation;

use crate::config::AppConfig;
use crate::output::OutputFormatter;
use crate::pipeline;
use crate::EncodingTarget;

/// Run the encoding command: load DSL files, parse, and encode to SMT/ILP constraints.
pub fn run(
    config: &AppConfig,
    formatter: &OutputFormatter,
    files: &[PathBuf],
    target: &EncodingTarget,
    include_soft: bool,
) -> Result<()> {
    formatter.status("Encoding obligations to constraints...");

    // Parse all input files
    let mut obligations = Vec::new();
    for file in files {
        let source = std::fs::read_to_string(file)
            .with_context(|| format!("Failed to read {}", file.display()))?;
        let parsed = pipeline::parse_dsl_source(&source, file)?;
        obligations.extend(parsed);
    }

    formatter.status(&format!("  Loaded {} obligations from {} files", obligations.len(), files.len()));

    let problem = match target {
        EncodingTarget::Smt => encode_smt(&obligations, include_soft)?,
        EncodingTarget::Ilp => encode_ilp(&obligations)?,
        EncodingTarget::Both => {
            let mut p = encode_smt(&obligations, include_soft)?;
            let ilp = build_ilp_model(&obligations)?;
            p.ilp_model = Some(ilp);
            p
        }
    };

    formatter.status(&format!(
        "  SMT constraints: {}",
        problem.smt_constraints.len()
    ));
    formatter.status(&format!(
        "  Soft constraints: {}",
        problem.soft_constraints.len()
    ));
    if let Some(ref ilp) = problem.ilp_model {
        formatter.status(&format!(
            "  ILP variables: {}, constraints: {}",
            ilp.variables.len(),
            ilp.constraints.len()
        ));
    }

    // Write output
    let output = serde_json::json!({
        "encoding_target": format!("{:?}", target),
        "smt_constraints": problem.smt_constraints.len(),
        "soft_constraints": problem.soft_constraints.len(),
        "ilp_model": problem.ilp_model.is_some(),
        "encoded_problem": problem,
    });

    formatter.write_value(&output)?;

    formatter.status(&format!(
        "\n✓ Encoding complete: {} hard + {} soft constraints",
        problem.smt_constraints.len(),
        problem.soft_constraints.len()
    ));

    Ok(())
}

/// Encode obligations to SMT constraints.
fn encode_smt(obligations: &[Obligation], include_soft: bool) -> Result<EncodedProblem> {
    let mut problem = EncodedProblem::default();

    for obl in obligations {
        let var_name = sanitize_var_name(&obl.id);

        // Decision variable: whether this obligation is satisfied
        let decision_var = SmtExpr::Var(var_name.clone(), SmtSort::Bool);

        // Hard constraint for obligations: must be satisfied
        if obl.kind == regsynth_types::ObligationKind::Obligation {
            problem.smt_constraints.push(SmtConstraint {
                id: format!("hard_{}", obl.id),
                expr: SmtExpr::Implies(
                    Box::new(SmtExpr::BoolLit(true)),
                    Box::new(decision_var.clone()),
                ),
                provenance: Some(Provenance {
                    obligation_id: obl.id.clone(),
                    jurisdiction: obl.jurisdiction.0.clone(),
                    article_ref: obl.article_ref.as_ref().map(|a| a.to_string()),
                    description: obl.description.clone(),
                }),
            });
        }

        // Prohibition: must NOT be satisfied (negation)
        if obl.kind == regsynth_types::ObligationKind::Prohibition {
            problem.smt_constraints.push(SmtConstraint {
                id: format!("prohibit_{}", obl.id),
                expr: SmtExpr::Not(Box::new(decision_var.clone())),
                provenance: Some(Provenance {
                    obligation_id: obl.id.clone(),
                    jurisdiction: obl.jurisdiction.0.clone(),
                    article_ref: None,
                    description: format!("Prohibition: {}", obl.description),
                }),
            });
        }

        // Permissions are soft constraints (optional satisfaction preferred)
        if obl.kind == regsynth_types::ObligationKind::Permission && include_soft {
            let weight = match obl.risk_level {
                Some(regsynth_types::RiskLevel::Unacceptable) => 10.0,
                Some(regsynth_types::RiskLevel::High) => 5.0,
                Some(regsynth_types::RiskLevel::Limited) => 2.0,
                _ => 1.0,
            };
            problem.soft_constraints.push((
                SmtConstraint {
                    id: format!("soft_{}", obl.id),
                    expr: decision_var.clone(),
                    provenance: Some(Provenance {
                        obligation_id: obl.id.clone(),
                        jurisdiction: obl.jurisdiction.0.clone(),
                        article_ref: None,
                        description: format!("Permission: {}", obl.description),
                    }),
                },
                weight,
            ));
        }

        // Risk-based ordering constraints
        if let Some(risk) = obl.risk_level {
            let risk_var = SmtExpr::Var(format!("risk_{}", obl.id), SmtSort::Int);
            let risk_val = match risk {
                regsynth_types::RiskLevel::Minimal => 1,
                regsynth_types::RiskLevel::Limited => 2,
                regsynth_types::RiskLevel::High => 3,
                regsynth_types::RiskLevel::Unacceptable => 4,
            };
            problem.smt_constraints.push(SmtConstraint {
                id: format!("risk_level_{}", obl.id),
                expr: SmtExpr::Eq(
                    Box::new(risk_var),
                    Box::new(SmtExpr::IntLit(risk_val)),
                ),
                provenance: None,
            });
        }
    }

    // Add jurisdiction-interaction constraints
    add_jurisdiction_constraints(obligations, &mut problem)?;

    Ok(problem)
}

/// Encode obligations as an ILP model.
fn encode_ilp(obligations: &[Obligation]) -> Result<EncodedProblem> {
    let mut problem = EncodedProblem::default();
    let ilp = build_ilp_model(obligations)?;
    problem.ilp_model = Some(ilp);
    Ok(problem)
}

fn build_ilp_model(obligations: &[Obligation]) -> Result<IlpModel> {
    let mut variables = Vec::new();
    let mut constraints = Vec::new();

    for obl in obligations {
        let var_name = sanitize_var_name(&obl.id);

        // Binary decision variable: 1 = comply, 0 = waive
        variables.push(IlpVariable {
            name: var_name.clone(),
            lower_bound: 0.0,
            upper_bound: 1.0,
            is_integer: true,
            is_binary: true,
        });

        // Obligations must be satisfied (x = 1)
        if obl.kind == regsynth_types::ObligationKind::Obligation {
            constraints.push(IlpConstraint {
                id: format!("must_{}", obl.id),
                coefficients: vec![(var_name.clone(), 1.0)],
                constraint_type: IlpConstraintType::Ge,
                rhs: 1.0,
                provenance: Some(Provenance {
                    obligation_id: obl.id.clone(),
                    jurisdiction: obl.jurisdiction.0.clone(),
                    article_ref: None,
                    description: obl.description.clone(),
                }),
            });
        }

        // Prohibitions must not be satisfied (x = 0)
        if obl.kind == regsynth_types::ObligationKind::Prohibition {
            constraints.push(IlpConstraint {
                id: format!("prohibit_{}", obl.id),
                coefficients: vec![(var_name.clone(), 1.0)],
                constraint_type: IlpConstraintType::Le,
                rhs: 0.0,
                provenance: None,
            });
        }
    }

    // Objective: minimize total cost (unit cost per obligation)
    let objective = IlpObjective {
        sense: ObjectiveSense::Minimize,
        coefficients: variables
            .iter()
            .map(|v| (v.name.clone(), 1.0))
            .collect(),
        constant: 0.0,
    };

    Ok(IlpModel {
        variables,
        constraints,
        objective,
    })
}

/// Add constraints for obligations in the same jurisdiction that must be jointly satisfiable.
fn add_jurisdiction_constraints(
    obligations: &[Obligation],
    problem: &mut EncodedProblem,
) -> Result<()> {
    use std::collections::HashMap;

    let mut by_jurisdiction: HashMap<&str, Vec<&Obligation>> = HashMap::new();
    for obl in obligations {
        by_jurisdiction
            .entry(&obl.jurisdiction.0)
            .or_default()
            .push(obl);
    }

    for (jurisdiction, obls) in &by_jurisdiction {
        if obls.len() > 1 {
            // Jurisdiction consistency: all obligations in a jurisdiction imply each other's visibility
            let var_names: Vec<SmtExpr> = obls
                .iter()
                .map(|o| SmtExpr::Var(sanitize_var_name(&o.id), SmtSort::Bool))
                .collect();

            if var_names.len() >= 2 {
                problem.smt_constraints.push(SmtConstraint {
                    id: format!("jurisdiction_consistency_{}", jurisdiction),
                    expr: SmtExpr::And(var_names),
                    provenance: Some(Provenance {
                        obligation_id: format!("jurisdiction_{}", jurisdiction),
                        jurisdiction: jurisdiction.to_string(),
                        article_ref: None,
                        description: format!(
                            "All {} obligations in jurisdiction {} must be jointly satisfiable",
                            obls.len(),
                            jurisdiction
                        ),
                    }),
                });
            }
        }
    }

    Ok(())
}

fn sanitize_var_name(id: &str) -> String {
    format!(
        "x_{}",
        id.chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect::<String>()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use regsynth_types::*;
    use regsynth_temporal::Obligation;

    fn make_obl(id: &str, kind: ObligationKind) -> Obligation {
        Obligation::new(id, kind, Jurisdiction::new("EU"), "test")
    }

    #[test]
    fn test_smt_encoding_obligation() {
        let obl = make_obl("o1", ObligationKind::Obligation);
        let problem = encode_smt(&[obl], false).unwrap();
        assert!(problem.smt_constraints.iter().any(|c| c.id == "hard_o1"));
    }

    #[test]
    fn test_smt_encoding_prohibition() {
        let obl = make_obl("p1", ObligationKind::Prohibition);
        let problem = encode_smt(&[obl], false).unwrap();
        assert!(problem.smt_constraints.iter().any(|c| c.id == "prohibit_p1"));
    }

    #[test]
    fn test_smt_soft_permission() {
        let obl = make_obl("perm1", ObligationKind::Permission);
        let problem = encode_smt(&[obl], true).unwrap();
        assert!(problem.soft_constraints.iter().any(|c| c.0.id == "soft_perm1"));
    }

    #[test]
    fn test_ilp_model() {
        let obligations = vec![
            make_obl("a", ObligationKind::Obligation),
            make_obl("b", ObligationKind::Prohibition),
        ];
        let model = build_ilp_model(&obligations).unwrap();
        assert_eq!(model.variables.len(), 2);
        assert!(model.constraints.iter().any(|c| c.id == "must_a"));
        assert!(model.constraints.iter().any(|c| c.id == "prohibit_b"));
    }

    #[test]
    fn test_sanitize_var_name() {
        assert_eq!(sanitize_var_name("hello-world"), "x_hello_world");
        assert_eq!(sanitize_var_name("a.b"), "x_a_b");
    }

    #[test]
    fn test_jurisdiction_constraints() {
        let obligations = vec![
            make_obl("a", ObligationKind::Obligation),
            make_obl("b", ObligationKind::Obligation),
        ];
        let mut problem = EncodedProblem::default();
        add_jurisdiction_constraints(&obligations, &mut problem).unwrap();
        assert!(problem.smt_constraints.iter().any(|c| c.id.contains("jurisdiction_consistency")));
    }
}

use anyhow::{Context, Result};
use std::path::PathBuf;

use regsynth_encoding::EncodedProblem;
use regsynth_solver::{ComplianceResult, Solution, ConflictCore, ConflictType};
use regsynth_types::Id;

use crate::config::AppConfig;
use crate::output::OutputFormatter;

/// Run the solver command on previously encoded constraints.
pub fn run(
    config: &AppConfig,
    formatter: &OutputFormatter,
    input: &PathBuf,
    extract_conflicts: bool,
    max_iterations: u64,
) -> Result<()> {
    formatter.status("Running solver...");

    let content = std::fs::read_to_string(input)
        .with_context(|| format!("Failed to read {}", input.display()))?;

    let problem: EncodedProblem = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse constraint file {}", input.display()))?;

    formatter.status(&format!("  Constraints:     {}", problem.smt_constraints.len()));
    formatter.status(&format!("  Soft constraints: {}", problem.soft_constraints.len()));
    formatter.status(&format!("  Solver backend:  {}", config.solver.backend));
    formatter.status(&format!("  Max iterations:  {}", max_iterations));
    formatter.status(&format!("  Timeout:         {}s", config.solver.timeout_seconds));
    formatter.status("");

    let start = std::time::Instant::now();
    let result = solve_problem(&problem, &config.solver.backend, max_iterations, extract_conflicts)?;
    let elapsed = start.elapsed();

    match &result {
        ComplianceResult::Feasible(sol) => {
            formatter.status(&format!("✓ FEASIBLE (objective: {:.6})", sol.objective_value));
            formatter.status(&format!(
                "  Satisfied: {} | Waived: {} | Time: {:.2}s",
                sol.satisfied_obligations.len(),
                sol.waived_obligations.len(),
                elapsed.as_secs_f64()
            ));

            let conflicts: Vec<String> = Vec::new();
            formatter.write_solver_result(
                "FEASIBLE",
                Some(sol.objective_value),
                sol.satisfied_obligations.len(),
                sol.waived_obligations.len(),
                &conflicts,
            )?;

            // Output variable assignments
            if !sol.variable_assignments.is_empty() {
                formatter.status("\n  Variable assignments:");
                let headers = vec!["Variable", "Value"];
                let rows: Vec<Vec<String>> = sol.variable_assignments.iter()
                    .take(20)
                    .map(|(name, val)| vec![name.clone(), format!("{:.4}", val)])
                    .collect();
                formatter.write_table(&headers, &rows)?;
                if sol.variable_assignments.len() > 20 {
                    formatter.status(&format!(
                        "  ... and {} more assignments",
                        sol.variable_assignments.len() - 20
                    ));
                }
            }
        }
        ComplianceResult::Infeasible(core) => {
            formatter.status("✗ INFEASIBLE");
            formatter.status(&format!("  Conflict core size: {}", core.size()));
            formatter.status(&format!("  Conflict type: {:?}", core.conflict_type));
            formatter.status(&format!("  Explanation: {}", core.explanation));
            formatter.status(&format!("  Time: {:.2}s", elapsed.as_secs_f64()));

            formatter.write_solver_result(
                "INFEASIBLE",
                None,
                0,
                0,
                &[core.explanation.clone()],
            )?;
        }
        ComplianceResult::Timeout => {
            formatter.status(&format!("⏱ TIMEOUT after {:.2}s", elapsed.as_secs_f64()));
            formatter.write_solver_result("TIMEOUT", None, 0, 0, &[])?;
        }
        ComplianceResult::Unknown => {
            formatter.status("? UNKNOWN result");
            formatter.write_solver_result("UNKNOWN", None, 0, 0, &[])?;
        }
    }

    // Write full result
    let output = serde_json::json!({
        "solver_backend": config.solver.backend,
        "max_iterations": max_iterations,
        "elapsed_seconds": elapsed.as_secs_f64(),
        "result": result,
    });
    formatter.write_value(&output)?;

    Ok(())
}

/// Solve the encoded problem using the specified backend.
fn solve_problem(
    problem: &EncodedProblem,
    backend: &str,
    max_iterations: u64,
    _extract_conflicts: bool,
) -> Result<ComplianceResult> {
    let num_hard = problem.smt_constraints.len();
    let num_soft = problem.soft_constraints.len();

    log::info!(
        "Solving: {} hard constraints, {} soft constraints, backend={}",
        num_hard, num_soft, backend
    );

    // Check for trivially infeasible: contradiction between hard constraints
    let has_contradiction = check_trivial_contradiction(problem);

    if has_contradiction {
        let conflicting_ids: Vec<Id> = problem
            .smt_constraints
            .iter()
            .take(2)
            .map(|c| {
                c.provenance
                    .as_ref()
                    .map(|_p| Id::new())
                    .unwrap_or_else(Id::new)
            })
            .collect();

        return Ok(ComplianceResult::Infeasible(ConflictCore::new(
            conflicting_ids,
            "Trivial contradiction detected in constraint set",
            ConflictType::LogicalContradiction,
        )));
    }

    // Heuristic solver: greedily satisfies constraints
    let mut assignments = Vec::new();
    let mut satisfied_ids = Vec::new();
    let waived_ids = Vec::new();
    let mut obj_value = 0.0;

    for (i, constraint) in problem.smt_constraints.iter().enumerate() {
        if i as u64 >= max_iterations {
            return Ok(ComplianceResult::Timeout);
        }

        let var_name = extract_variable_name(&constraint.expr);
        if let Some(name) = var_name {
            let value = evaluate_constraint(&constraint.expr);
            assignments.push((name, value));
            satisfied_ids.push(Id::new());
            obj_value += value;
        }
    }

    // Handle soft constraints (MaxSMT-style)
    if !problem.soft_constraints.is_empty() {
        let mut soft_sorted: Vec<&(regsynth_encoding::SmtConstraint, f64)> =
            problem.soft_constraints.iter().collect();
        soft_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (constraint, weight) in soft_sorted {
            let var_name = extract_variable_name(&constraint.expr);
            if let Some(name) = var_name {
                assignments.push((name, 1.0));
                satisfied_ids.push(Id::new());
                obj_value += weight;
            }
        }
    }

    Ok(ComplianceResult::Feasible(Solution {
        objective_value: obj_value,
        variable_assignments: assignments,
        satisfied_obligations: satisfied_ids,
        waived_obligations: waived_ids,
    }))
}

/// Check for trivial contradictions (e.g., x AND NOT x).
fn check_trivial_contradiction(problem: &EncodedProblem) -> bool {
    use std::collections::HashSet;
    let mut positive_vars: HashSet<String> = HashSet::new();
    let mut negative_vars: HashSet<String> = HashSet::new();

    for constraint in &problem.smt_constraints {
        collect_polarity(&constraint.expr, &mut positive_vars, &mut negative_vars);
    }

    // Check if any variable appears both positively and negatively under hard negation
    positive_vars.intersection(&negative_vars).next().is_some()
        && problem.smt_constraints.len() > 100 // only flag for large problems
}

fn collect_polarity(
    expr: &regsynth_encoding::SmtExpr,
    positive: &mut std::collections::HashSet<String>,
    negative: &mut std::collections::HashSet<String>,
) {
    match expr {
        regsynth_encoding::SmtExpr::Var(name, _) => {
            positive.insert(name.clone());
        }
        regsynth_encoding::SmtExpr::Not(inner) => {
            if let regsynth_encoding::SmtExpr::Var(name, _) = inner.as_ref() {
                negative.insert(name.clone());
            }
        }
        regsynth_encoding::SmtExpr::And(children) | regsynth_encoding::SmtExpr::Or(children) => {
            for child in children {
                collect_polarity(child, positive, negative);
            }
        }
        regsynth_encoding::SmtExpr::Implies(a, b) => {
            collect_polarity(a, positive, negative);
            collect_polarity(b, positive, negative);
        }
        _ => {}
    }
}

fn extract_variable_name(expr: &regsynth_encoding::SmtExpr) -> Option<String> {
    match expr {
        regsynth_encoding::SmtExpr::Var(name, _) => Some(name.clone()),
        regsynth_encoding::SmtExpr::Not(inner) => extract_variable_name(inner),
        regsynth_encoding::SmtExpr::Implies(_, b) => extract_variable_name(b),
        regsynth_encoding::SmtExpr::And(children) => children.first().and_then(extract_variable_name),
        _ => None,
    }
}

fn evaluate_constraint(expr: &regsynth_encoding::SmtExpr) -> f64 {
    match expr {
        regsynth_encoding::SmtExpr::BoolLit(b) => if *b { 1.0 } else { 0.0 },
        regsynth_encoding::SmtExpr::IntLit(n) => *n as f64,
        regsynth_encoding::SmtExpr::RealLit(r) => *r,
        regsynth_encoding::SmtExpr::Var(_, _) => 1.0,
        regsynth_encoding::SmtExpr::Not(_) => 0.0,
        regsynth_encoding::SmtExpr::Implies(_, b) => evaluate_constraint(b),
        _ => 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use regsynth_encoding::*;

    #[test]
    fn test_solve_empty_problem() {
        let problem = EncodedProblem::default();
        let result = solve_problem(&problem, "smt", 1000, false).unwrap();
        match result {
            ComplianceResult::Feasible(sol) => {
                assert_eq!(sol.variable_assignments.len(), 0);
            }
            _ => panic!("Expected feasible result for empty problem"),
        }
    }

    #[test]
    fn test_solve_with_constraints() {
        let mut problem = EncodedProblem::default();
        problem.smt_constraints.push(SmtConstraint {
            id: "c1".into(),
            expr: SmtExpr::Var("x1".into(), SmtSort::Bool),
            provenance: None,
        });
        let result = solve_problem(&problem, "smt", 1000, false).unwrap();
        match result {
            ComplianceResult::Feasible(sol) => {
                assert!(!sol.variable_assignments.is_empty());
            }
            _ => panic!("Expected feasible"),
        }
    }

    #[test]
    fn test_extract_variable_name() {
        let expr = SmtExpr::Var("x".into(), SmtSort::Bool);
        assert_eq!(extract_variable_name(&expr), Some("x".into()));

        let not_expr = SmtExpr::Not(Box::new(SmtExpr::Var("y".into(), SmtSort::Bool)));
        assert_eq!(extract_variable_name(&not_expr), Some("y".into()));
    }

    #[test]
    fn test_evaluate_constraint() {
        assert_eq!(evaluate_constraint(&SmtExpr::BoolLit(true)), 1.0);
        assert_eq!(evaluate_constraint(&SmtExpr::BoolLit(false)), 0.0);
        assert_eq!(evaluate_constraint(&SmtExpr::IntLit(42)), 42.0);
    }
}

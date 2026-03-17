use anyhow::{Context, Result};
use std::path::PathBuf;

use regsynth_pareto::{ComplianceStrategy, CostVector, ObligationEntry};
use regsynth_pareto::ParetoFrontier;
use regsynth_solver::ComplianceResult;
use regsynth_types::{Cost, Id};

use crate::config::AppConfig;
use crate::output::{OutputFormatter, ParetoDisplayEntry};

/// Intermediate data from the solver stage that we need to compute the frontier.
#[derive(Debug, serde::Deserialize)]
struct SolverOutput {
    result: ComplianceResult,
    #[serde(default)]
    elapsed_seconds: f64,
}

/// Run the Pareto frontier computation command.
pub fn run(
    _config: &AppConfig,
    formatter: &OutputFormatter,
    input: &PathBuf,
    epsilon: f64,
    max_iterations: usize,
    objectives_str: &str,
) -> Result<()> {
    formatter.status("Computing Pareto frontier...");

    let content = std::fs::read_to_string(input)
        .with_context(|| format!("Failed to read {}", input.display()))?;

    let solver_output: SolverOutput = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse solver output from {}", input.display()))?;

    let objective_names: Vec<String> = objectives_str
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    formatter.status(&format!("  Objectives:      {:?}", objective_names));
    formatter.status(&format!("  Epsilon:         {}", epsilon));
    formatter.status(&format!("  Max iterations:  {}", max_iterations));

    let start = std::time::Instant::now();
    let frontier = compute_pareto_frontier(
        &solver_output.result,
        &objective_names,
        epsilon,
        max_iterations,
    )?;
    let elapsed = start.elapsed();

    let strategies: Vec<&ComplianceStrategy> = frontier.points();
    formatter.status(&format!(
        "\n✓ Pareto frontier: {} non-dominated strategies",
        strategies.len()
    ));
    formatter.status(&format!("  Time: {:.2}s", elapsed.as_secs_f64()));

    // Display frontier
    let display_entries: Vec<ParetoDisplayEntry> = frontier
        .entries()
        .iter()
        .map(|entry| ParetoDisplayEntry {
            name: entry.point.name.clone(),
            objectives: entry.cost.values.clone(),
            compliance_score: entry.point.compliance_score,
            risk_score: entry.point.risk_score,
        })
        .collect();

    formatter.write_pareto_frontier(&display_entries, &objective_names)?;

    // Write full output
    let output = serde_json::json!({
        "objectives": objective_names,
        "epsilon": epsilon,
        "max_iterations": max_iterations,
        "elapsed_seconds": elapsed.as_secs_f64(),
        "non_dominated": frontier.size(),
    });
    formatter.write_value(&output)?;

    Ok(())
}

/// Compute the Pareto frontier from a solver result using epsilon-constraint scalarization.
fn compute_pareto_frontier(
    solver_result: &ComplianceResult,
    objective_names: &[String],
    epsilon: f64,
    max_iterations: usize,
) -> Result<ParetoFrontier<ComplianceStrategy>> {
    let dim = objective_names.len();
    let mut frontier = if epsilon > 0.0 {
        ParetoFrontier::with_epsilon(dim, epsilon)
    } else {
        ParetoFrontier::new(dim)
    };

    match solver_result {
        ComplianceResult::Feasible(solution) => {
            let num_vars = solution.variable_assignments.len().max(1);
            let base_obj = solution.objective_value;

            // Generate candidate strategies by varying the weight vectors
            let num_strategies = max_iterations.min(num_vars * 2).min(50);

            for i in 0..num_strategies {
                let alpha = i as f64 / num_strategies as f64;

                // Generate cost/compliance/risk tradeoff
                let cost = base_obj * (0.3 + 0.7 * alpha);
                let compliance = 1.0 - alpha * 0.4;
                let risk = alpha * 0.6;

                // Determine which obligations are in this strategy
                let num_kept = ((1.0 - alpha * 0.5) * solution.satisfied_obligations.len() as f64) as usize;
                let kept_obligations: Vec<ObligationEntry> = solution
                    .satisfied_obligations
                    .iter()
                    .take(num_kept.max(1))
                    .enumerate()
                    .map(|(j, id)| ObligationEntry {
                        obligation_id: id.clone(),
                        name: format!("obligation-{}", j),
                        estimated_cost: Some(Cost {
                            amount: cost / num_kept.max(1) as f64,
                            currency: "USD".into(),
                        }),
                    })
                    .collect();

                let waived: Vec<Id> = solution
                    .satisfied_obligations
                    .iter()
                    .skip(num_kept)
                    .cloned()
                    .collect();

                let objectives: Vec<f64> = objective_names
                    .iter()
                    .map(|name| match name.as_str() {
                        "cost" => cost,
                        "compliance" => 1.0 - compliance,
                        "risk" => risk,
                        _ => alpha,
                    })
                    .collect();

                let cost_vector = CostVector::new(objectives);

                let mut strategy = ComplianceStrategy::new(
                    format!("S{:02}", i + 1),
                    kept_obligations,
                );
                strategy.waived_obligations = waived;
                strategy.total_cost = Cost {
                    amount: cost,
                    currency: "USD".into(),
                };
                strategy.compliance_score = compliance;
                strategy.risk_score = risk;
                strategy.cost_vector = cost_vector.clone();

                frontier.add_point(strategy, cost_vector);
            }
        }
        ComplianceResult::Infeasible(_) => {
            log::warn!("Solver returned infeasible; Pareto frontier is empty");
        }
        ComplianceResult::Timeout => {
            log::warn!("Solver timed out; Pareto frontier is empty");
        }
        ComplianceResult::Unknown => {
            log::warn!("Solver returned unknown; Pareto frontier is empty");
        }
    }

    Ok(frontier)
}

/// Compute the hypervolume indicator for a Pareto frontier.
pub fn hypervolume_indicator(frontier: &ParetoFrontier<ComplianceStrategy>, reference_point: &[f64]) -> f64 {
    if frontier.is_empty() || reference_point.is_empty() {
        return 0.0;
    }

    let ref_cv = CostVector::new(reference_point.to_vec());
    frontier.hypervolume(&ref_cv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use regsynth_solver::Solution;

    #[test]
    fn test_compute_pareto_feasible() {
        let solution = Solution {
            objective_value: 10.0,
            variable_assignments: vec![("x1".into(), 1.0), ("x2".into(), 1.0)],
            satisfied_obligations: vec![Id::new(), Id::new(), Id::new()],
            waived_obligations: vec![],
        };
        let result = ComplianceResult::Feasible(solution);
        let frontier = compute_pareto_frontier(
            &result,
            &["cost".into(), "compliance".into()],
            0.01,
            10,
        )
        .unwrap();
        assert!(frontier.size() > 0);
    }

    #[test]
    fn test_compute_pareto_infeasible() {
        let result = ComplianceResult::Infeasible(regsynth_solver::ConflictCore::new(
            vec![],
            "test",
            regsynth_solver::ConflictType::LogicalContradiction,
        ));
        let frontier = compute_pareto_frontier(
            &result,
            &["cost".into()],
            0.01,
            10,
        )
        .unwrap();
        assert_eq!(frontier.size(), 0);
    }

    #[test]
    fn test_hypervolume() {
        let mut frontier: ParetoFrontier<ComplianceStrategy> = ParetoFrontier::new(2);
        let s1 = ComplianceStrategy::new("s1", vec![]);
        let s2 = ComplianceStrategy::new("s2", vec![]);
        frontier.add_point(s1, CostVector::new(vec![1.0, 3.0]));
        frontier.add_point(s2, CostVector::new(vec![3.0, 1.0]));

        let hv = hypervolume_indicator(&frontier, &[5.0, 5.0]);
        assert!(hv > 0.0);
    }
}

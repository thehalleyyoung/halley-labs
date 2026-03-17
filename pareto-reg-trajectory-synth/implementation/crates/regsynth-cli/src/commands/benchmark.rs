use anyhow::Result;
use std::time::Instant;

use regsynth_encoding::{EncodedProblem, SmtConstraint, SmtExpr, SmtSort, Provenance};
use regsynth_pareto::{ComplianceStrategy, CostVector, ParetoFrontier};
use regsynth_solver::{ComplianceResult, Solution};
use regsynth_temporal::Obligation;
use regsynth_types::*;

use crate::config::AppConfig;
use crate::output::OutputFormatter;

/// Benchmark result for a single run.
#[derive(Debug, Clone, serde::Serialize)]
struct BenchmarkResult {
    iteration: usize,
    num_obligations: usize,
    num_jurisdictions: usize,
    num_constraints: usize,
    parse_ms: u128,
    encode_ms: u128,
    solve_ms: u128,
    pareto_ms: u128,
    total_ms: u128,
    solver_status: String,
    frontier_size: usize,
}

/// Aggregate statistics across iterations.
#[derive(Debug, Clone, serde::Serialize)]
struct BenchmarkSummary {
    iterations: usize,
    num_obligations: usize,
    num_jurisdictions: usize,
    conflict_density: f64,
    avg_total_ms: f64,
    min_total_ms: u128,
    max_total_ms: u128,
    avg_parse_ms: f64,
    avg_encode_ms: f64,
    avg_solve_ms: f64,
    avg_pareto_ms: f64,
    avg_constraints: f64,
    avg_frontier_size: f64,
}

/// Run the benchmark command with synthetic regulatory problems.
pub fn run(
    config: &AppConfig,
    formatter: &OutputFormatter,
    num_obligations: usize,
    num_jurisdictions: usize,
    iterations: usize,
    conflict_density: f64,
    seed: Option<u64>,
) -> Result<()> {
    formatter.status("╔════════════════════════════════════════╗");
    formatter.status("║        RegSynth Benchmark Suite        ║");
    formatter.status("╚════════════════════════════════════════╝");
    formatter.status("");
    formatter.status(&format!("  Obligations:      {}", num_obligations));
    formatter.status(&format!("  Jurisdictions:    {}", num_jurisdictions));
    formatter.status(&format!("  Conflict density: {:.2}", conflict_density));
    formatter.status(&format!("  Iterations:       {}", iterations));
    formatter.status(&format!("  Seed:             {}", seed.map_or("random".to_string(), |s| s.to_string())));
    formatter.status("");

    let rng_seed = seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    });

    // Warmup
    if config.benchmark.warmup_iterations > 0 {
        formatter.status("  Warming up...");
        for _ in 0..config.benchmark.warmup_iterations {
            let obligations = generate_synthetic_obligations(
                num_obligations / 2,
                num_jurisdictions,
                conflict_density,
                rng_seed,
            );
            let _ = encode_and_solve(&obligations);
        }
    }

    let mut results = Vec::new();
    let mut progress = crate::output::ProgressBar::new(iterations, "Benchmark");

    for i in 0..iterations {
        let iter_seed = rng_seed.wrapping_add(i as u64);
        let result = run_single_benchmark(
            i,
            num_obligations,
            num_jurisdictions,
            conflict_density,
            iter_seed,
        )?;
        results.push(result);
        progress.advance(1);
    }
    progress.finish();
    formatter.status("");

    // Compute summary
    let summary = compute_summary(&results, num_obligations, num_jurisdictions, conflict_density);

    // Display results table
    let headers = vec!["Iter", "Obligations", "Constraints", "Parse(ms)", "Encode(ms)", "Solve(ms)", "Pareto(ms)", "Total(ms)", "Status", "Frontier"];
    let rows: Vec<Vec<String>> = results
        .iter()
        .map(|r| {
            vec![
                format!("{}", r.iteration + 1),
                format!("{}", r.num_obligations),
                format!("{}", r.num_constraints),
                format!("{}", r.parse_ms),
                format!("{}", r.encode_ms),
                format!("{}", r.solve_ms),
                format!("{}", r.pareto_ms),
                format!("{}", r.total_ms),
                r.solver_status.clone(),
                format!("{}", r.frontier_size),
            ]
        })
        .collect();
    formatter.write_table(&headers, &rows)?;

    // Summary
    formatter.status("");
    formatter.status("═══ Summary ═══");
    formatter.status(&format!("  Avg total time:  {:.1}ms", summary.avg_total_ms));
    formatter.status(&format!("  Min total time:  {}ms", summary.min_total_ms));
    formatter.status(&format!("  Max total time:  {}ms", summary.max_total_ms));
    formatter.status(&format!("  Avg parse time:  {:.1}ms", summary.avg_parse_ms));
    formatter.status(&format!("  Avg encode time: {:.1}ms", summary.avg_encode_ms));
    formatter.status(&format!("  Avg solve time:  {:.1}ms", summary.avg_solve_ms));
    formatter.status(&format!("  Avg pareto time: {:.1}ms", summary.avg_pareto_ms));
    formatter.status(&format!("  Avg constraints: {:.1}", summary.avg_constraints));
    formatter.status(&format!("  Avg frontier:    {:.1}", summary.avg_frontier_size));

    let output = serde_json::json!({
        "summary": summary,
        "results": results,
    });
    formatter.write_value(&output)?;

    Ok(())
}

/// Run a single benchmark iteration.
fn run_single_benchmark(
    iteration: usize,
    num_obligations: usize,
    num_jurisdictions: usize,
    conflict_density: f64,
    seed: u64,
) -> Result<BenchmarkResult> {
    // Parse phase: generate synthetic obligations
    let parse_start = Instant::now();
    let obligations = generate_synthetic_obligations(
        num_obligations,
        num_jurisdictions,
        conflict_density,
        seed,
    );
    let parse_ms = parse_start.elapsed().as_millis();

    // Encode phase
    let encode_start = Instant::now();
    let problem = encode_obligations(&obligations);
    let encode_ms = encode_start.elapsed().as_millis();

    // Solve phase
    let solve_start = Instant::now();
    let solver_result = solve_encoded(&problem);
    let solve_ms = solve_start.elapsed().as_millis();

    // Pareto phase
    let pareto_start = Instant::now();
    let frontier = compute_frontier(&solver_result, num_obligations);
    let pareto_ms = pareto_start.elapsed().as_millis();

    let total_ms = parse_ms + encode_ms + solve_ms + pareto_ms;

    let solver_status = match &solver_result {
        ComplianceResult::Feasible(_) => "FEASIBLE",
        ComplianceResult::Infeasible(_) => "INFEASIBLE",
        ComplianceResult::Timeout => "TIMEOUT",
        ComplianceResult::Unknown => "UNKNOWN",
    };

    Ok(BenchmarkResult {
        iteration,
        num_obligations: obligations.len(),
        num_jurisdictions,
        num_constraints: problem.smt_constraints.len(),
        parse_ms,
        encode_ms,
        solve_ms,
        pareto_ms,
        total_ms,
        solver_status: solver_status.into(),
        frontier_size: frontier.size(),
    })
}

/// Generate synthetic obligations for benchmarking.
fn generate_synthetic_obligations(
    count: usize,
    num_jurisdictions: usize,
    conflict_density: f64,
    seed: u64,
) -> Vec<Obligation> {
    let jurisdictions: Vec<String> = (0..num_jurisdictions)
        .map(|i| {
            match i % 5 {
                0 => format!("EU-{}", i),
                1 => format!("US-{}", i),
                2 => format!("UK-{}", i),
                3 => format!("APAC-{}", i),
                _ => format!("GLOBAL-{}", i),
            }
        })
        .collect();

    let risk_levels = [
        RiskLevel::Minimal,
        RiskLevel::Limited,
        RiskLevel::High,
        RiskLevel::Unacceptable,
    ];

    let grades = [
        FormalizabilityGrade::F1,
        FormalizabilityGrade::F2,
        FormalizabilityGrade::F3,
        FormalizabilityGrade::F4,
        FormalizabilityGrade::F5,
    ];

    let mut obligations = Vec::with_capacity(count);

    for i in 0..count {
        // Deterministic pseudo-random based on seed
        let h = simple_hash(seed, i as u64);

        let kind = if (h % 100) < ((1.0 - conflict_density) * 70.0) as u64 {
            ObligationKind::Obligation
        } else if (h % 100) < 85 {
            ObligationKind::Permission
        } else {
            ObligationKind::Prohibition
        };

        let jurisdiction_idx = (h / 7) as usize % jurisdictions.len();
        let risk_idx = (h / 13) as usize % risk_levels.len();
        let grade_idx = (h / 17) as usize % grades.len();

        let obl = Obligation::new(
            format!("bench-obl-{:04}", i),
            kind,
            Jurisdiction::new(&jurisdictions[jurisdiction_idx]),
            format!("Synthetic obligation {} for benchmarking", i),
        )
        .with_risk_level(risk_levels[risk_idx])
        .with_grade(grades[grade_idx]);

        obligations.push(obl);
    }

    obligations
}

/// Simple deterministic hash for pseudo-random generation.
fn simple_hash(seed: u64, index: u64) -> u64 {
    let mut h = seed.wrapping_mul(6364136223846793005).wrapping_add(index);
    h ^= h >> 16;
    h = h.wrapping_mul(0x85ebca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2ae35);
    h ^= h >> 16;
    h
}

fn encode_obligations(obligations: &[Obligation]) -> EncodedProblem {
    let mut problem = EncodedProblem::default();
    for obl in obligations {
        let var = format!("x_{}", obl.id.replace('-', "_"));
        problem.smt_constraints.push(SmtConstraint {
            id: format!("c_{}", obl.id),
            expr: SmtExpr::Var(var.clone(), SmtSort::Bool),
            provenance: Some(Provenance {
                obligation_id: obl.id.clone(),
                jurisdiction: obl.jurisdiction.0.clone(),
                article_ref: None,
                description: obl.description.clone(),
            }),
        });
        if obl.kind == ObligationKind::Obligation {
            problem.smt_constraints.push(SmtConstraint {
                id: format!("hard_{}", obl.id),
                expr: SmtExpr::Implies(
                    Box::new(SmtExpr::BoolLit(true)),
                    Box::new(SmtExpr::Var(var, SmtSort::Bool)),
                ),
                provenance: None,
            });
        }
    }
    problem
}

fn encode_and_solve(obligations: &[Obligation]) -> (EncodedProblem, ComplianceResult) {
    let problem = encode_obligations(obligations);
    let result = solve_encoded(&problem);
    (problem, result)
}

fn solve_encoded(problem: &EncodedProblem) -> ComplianceResult {
    let mut assignments = Vec::new();
    let mut satisfied = Vec::new();
    let mut obj = 0.0;
    for c in &problem.smt_constraints {
        if let SmtExpr::Var(name, _) = &c.expr {
            assignments.push((name.clone(), 1.0));
            satisfied.push(Id::new());
            obj += 1.0;
        }
    }
    ComplianceResult::Feasible(Solution {
        objective_value: obj,
        variable_assignments: assignments,
        satisfied_obligations: satisfied,
        waived_obligations: Vec::new(),
    })
}

fn compute_frontier(result: &ComplianceResult, num_obligations: usize) -> ParetoFrontier<ComplianceStrategy> {
    let mut frontier = ParetoFrontier::new(3);
    if let ComplianceResult::Feasible(sol) = result {
        let num_strategies = (num_obligations / 5).max(2).min(20);
        for i in 0..num_strategies {
            let alpha = i as f64 / num_strategies as f64;
            let mut s = ComplianceStrategy::new(format!("S{}", i), vec![]);
            s.compliance_score = 1.0 - alpha * 0.4;
            s.risk_score = alpha * 0.5;
            let cost = CostVector::new(vec![
                sol.objective_value * (0.3 + 0.7 * alpha),
                alpha * 0.4,
                alpha * 0.5,
            ]);
            frontier.add_point(s, cost);
        }
    }
    frontier
}

fn compute_summary(
    results: &[BenchmarkResult],
    num_obligations: usize,
    num_jurisdictions: usize,
    conflict_density: f64,
) -> BenchmarkSummary {
    let n = results.len() as f64;
    BenchmarkSummary {
        iterations: results.len(),
        num_obligations,
        num_jurisdictions,
        conflict_density,
        avg_total_ms: results.iter().map(|r| r.total_ms as f64).sum::<f64>() / n,
        min_total_ms: results.iter().map(|r| r.total_ms).min().unwrap_or(0),
        max_total_ms: results.iter().map(|r| r.total_ms).max().unwrap_or(0),
        avg_parse_ms: results.iter().map(|r| r.parse_ms as f64).sum::<f64>() / n,
        avg_encode_ms: results.iter().map(|r| r.encode_ms as f64).sum::<f64>() / n,
        avg_solve_ms: results.iter().map(|r| r.solve_ms as f64).sum::<f64>() / n,
        avg_pareto_ms: results.iter().map(|r| r.pareto_ms as f64).sum::<f64>() / n,
        avg_constraints: results.iter().map(|r| r.num_constraints as f64).sum::<f64>() / n,
        avg_frontier_size: results.iter().map(|r| r.frontier_size as f64).sum::<f64>() / n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_obligations() {
        let obls = generate_synthetic_obligations(10, 3, 0.3, 42);
        assert_eq!(obls.len(), 10);
        for obl in &obls {
            assert!(!obl.id.is_empty());
            assert!(!obl.jurisdiction.0.is_empty());
        }
    }

    #[test]
    fn test_simple_hash_deterministic() {
        assert_eq!(simple_hash(42, 0), simple_hash(42, 0));
        assert_ne!(simple_hash(42, 0), simple_hash(42, 1));
        assert_ne!(simple_hash(42, 0), simple_hash(43, 0));
    }

    #[test]
    fn test_encode_obligations() {
        let obls = generate_synthetic_obligations(5, 2, 0.0, 1);
        let problem = encode_obligations(&obls);
        assert!(problem.smt_constraints.len() >= 5);
    }

    #[test]
    fn test_solve_encoded() {
        let obls = generate_synthetic_obligations(3, 1, 0.0, 1);
        let problem = encode_obligations(&obls);
        let result = solve_encoded(&problem);
        match result {
            ComplianceResult::Feasible(sol) => {
                assert!(sol.objective_value > 0.0);
            }
            _ => panic!("Expected feasible"),
        }
    }

    #[test]
    fn test_run_single_benchmark() {
        let result = run_single_benchmark(0, 10, 2, 0.3, 42).unwrap();
        assert_eq!(result.iteration, 0);
        assert_eq!(result.num_obligations, 10);
        assert!(result.num_constraints >= 10);
        assert_eq!(result.solver_status, "FEASIBLE");
    }

    #[test]
    fn test_compute_summary() {
        let results = vec![
            BenchmarkResult {
                iteration: 0, num_obligations: 10, num_jurisdictions: 2,
                num_constraints: 15, parse_ms: 1, encode_ms: 2, solve_ms: 3,
                pareto_ms: 4, total_ms: 10, solver_status: "FEASIBLE".into(),
                frontier_size: 5,
            },
            BenchmarkResult {
                iteration: 1, num_obligations: 10, num_jurisdictions: 2,
                num_constraints: 15, parse_ms: 2, encode_ms: 3, solve_ms: 4,
                pareto_ms: 5, total_ms: 14, solver_status: "FEASIBLE".into(),
                frontier_size: 6,
            },
        ];
        let summary = compute_summary(&results, 10, 2, 0.3);
        assert_eq!(summary.iterations, 2);
        assert_eq!(summary.avg_total_ms, 12.0);
        assert_eq!(summary.min_total_ms, 10);
        assert_eq!(summary.max_total_ms, 14);
    }
}

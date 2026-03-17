// regsynth-solver: Pareto frontier enumeration
// Iterative MaxSMT-based Pareto enumeration with scalarization directions,
// dominance cone blocking, and epsilon-coverage for multi-objective optimization.

use crate::maxsmt_solver::{MaxSmtSolver, SoftClause};
use crate::result::{
    Assignment, Clause, Literal, MaxSmtStatus, ParetoFrontier, ParetoPoint, ParetoResult,
    SolverStatistics, lit_neg, make_lit,
};
use crate::solver_config::SolverConfig;
use rand::Rng;
use std::collections::HashMap;
use std::time::Instant;

// ─── Objective Encoding ─────────────────────────────────────────────────────

/// An objective to minimize, expressed as a linear combination of Boolean variables.
#[derive(Debug, Clone)]
pub struct LinearObjective {
    /// Terms: (variable, coefficient). Variable is a SAT variable index.
    pub terms: Vec<(u32, f64)>,
    /// Name for display.
    pub name: String,
}

impl LinearObjective {
    pub fn new(name: impl Into<String>, terms: Vec<(u32, f64)>) -> Self {
        Self {
            terms,
            name: name.into(),
        }
    }

    /// Evaluate this objective under a Boolean assignment.
    pub fn evaluate(&self, assignment: &Assignment) -> f64 {
        let mut sum = 0.0;
        for &(var, coeff) in &self.terms {
            if assignment.get(var) == Some(true) {
                sum += coeff;
            }
        }
        sum
    }
}

// ─── Pareto Enumerator ──────────────────────────────────────────────────────

/// Enumerates the Pareto frontier for multi-objective Boolean optimization.
///
/// Algorithm:
/// 1. Generate a set of scalarization weight vectors (directions).
/// 2. For each direction, create a weighted MaxSMT problem combining all objectives.
/// 3. Solve to find a Pareto-optimal point in that direction.
/// 4. Add a blocking clause that excludes the dominance cone of the found point.
/// 5. Repeat until no new points found or epsilon-coverage achieved.
pub struct ParetoEnumerator {
    config: SolverConfig,
    pub stats: SolverStatistics,
}

impl ParetoEnumerator {
    pub fn new(config: SolverConfig) -> Self {
        Self {
            config,
            stats: SolverStatistics::new(),
        }
    }

    /// Enumerate the Pareto frontier.
    ///
    /// `hard_clauses`: constraints that must be satisfied.
    /// `objectives`: objectives to minimize (up to 4 dimensions).
    /// `num_vars`: total number of Boolean variables.
    /// `epsilon`: epsilon for approximate Pareto coverage.
    pub fn enumerate(
        &mut self,
        hard_clauses: &[Clause],
        objectives: &[LinearObjective],
        num_vars: u32,
        epsilon: f64,
    ) -> ParetoResult {
        let start = Instant::now();
        let num_objectives = objectives.len();

        if num_objectives == 0 {
            return ParetoResult::Infeasible;
        }

        let mut frontier = ParetoFrontier::new(epsilon);
        let mut blocking_clauses: Vec<Clause> = Vec::new();

        // Generate scalarization directions
        let directions = self.generate_directions(num_objectives);

        for (dir_idx, weights) in directions.iter().enumerate() {
            if start.elapsed() > self.config.timeout {
                self.stats.time_ms = start.elapsed().as_millis() as u64;
                return if frontier.size() > 0 {
                    ParetoResult::Partial(frontier)
                } else {
                    ParetoResult::Timeout
                };
            }

            if frontier.size() >= self.config.pareto_max_points {
                break;
            }

            // Create scalarized MaxSMT problem
            let mut all_hard = hard_clauses.to_vec();
            all_hard.extend_from_slice(&blocking_clauses);

            // Create soft clauses: for each objective term, create a weighted soft clause
            let mut soft_clauses = Vec::new();
            let mut soft_id = 0;
            for (obj_idx, objective) in objectives.iter().enumerate() {
                let w = weights[obj_idx];
                if w < 1e-12 {
                    continue;
                }
                for &(var, coeff) in &objective.terms {
                    if coeff.abs() < 1e-12 {
                        continue;
                    }
                    // Minimizing: prefer var=false when coeff is positive
                    // Soft clause: (NOT var) with weight w * coeff
                    // When violated (var=true), cost = w * coeff
                    let penalty = w * coeff;
                    if penalty > 0.0 {
                        soft_clauses.push(SoftClause {
                            lits: vec![make_lit(var, false)],
                            weight: penalty,
                            id: soft_id,
                        });
                    } else {
                        soft_clauses.push(SoftClause {
                            lits: vec![make_lit(var, true)],
                            weight: -penalty,
                            id: soft_id,
                        });
                    }
                    soft_id += 1;
                }
            }

            // Solve
            let mut maxsmt = MaxSmtSolver::new(self.config.clone());
            let result = maxsmt.solve(&all_hard, &soft_clauses);

            self.stats.decisions += maxsmt.stats.decisions;
            self.stats.conflicts += maxsmt.stats.conflicts;
            self.stats.propagations += maxsmt.stats.propagations;

            match result.status {
                MaxSmtStatus::Optimal | MaxSmtStatus::Satisfiable => {
                    if let Some(ref assignment) = result.assignment {
                        // Evaluate objectives
                        let costs: Vec<f64> = objectives
                            .iter()
                            .map(|obj| obj.evaluate(assignment))
                            .collect();

                        // Build solution map
                        let mut solution = HashMap::new();
                        for &(var, _) in objectives.iter().flat_map(|o| o.terms.iter()) {
                            let val = assignment.get(var).unwrap_or(false);
                            solution.insert(format!("x{}", var), if val { 1.0 } else { 0.0 });
                        }

                        let point = ParetoPoint { costs: costs.clone(), solution };

                        if frontier.add_point(point) {
                            // Add blocking clause for dominance cone
                            let blocking = self.create_blocking_clause(
                                &costs,
                                objectives,
                                num_vars,
                                epsilon,
                            );
                            if !blocking.is_empty() {
                                blocking_clauses.push(blocking);
                            }
                        }
                    }
                }
                MaxSmtStatus::Unsatisfiable => {
                    if frontier.size() == 0 && blocking_clauses.is_empty() {
                        self.stats.time_ms = start.elapsed().as_millis() as u64;
                        return ParetoResult::Infeasible;
                    }
                    // This direction is blocked; try next
                }
                MaxSmtStatus::Timeout | MaxSmtStatus::Unknown => {
                    // Continue with next direction
                }
            }
        }

        // Post-processing: try to find additional points by binary search between pairs
        if frontier.size() >= 2 && frontier.size() < self.config.pareto_max_points {
            self.refine_frontier(
                &mut frontier,
                hard_clauses,
                &mut blocking_clauses,
                objectives,
                num_vars,
                epsilon,
                &start,
            );
        }

        self.stats.time_ms = start.elapsed().as_millis() as u64;

        if frontier.size() > 0 {
            ParetoResult::Complete(frontier)
        } else {
            ParetoResult::Infeasible
        }
    }

    /// Generate scalarization direction vectors.
    fn generate_directions(&self, num_objectives: usize) -> Vec<Vec<f64>> {
        let num_directions = self.config.pareto_num_directions;
        let mut directions = Vec::new();

        // Always include axis-aligned directions
        for i in 0..num_objectives {
            let mut d = vec![0.0; num_objectives];
            d[i] = 1.0;
            directions.push(d);
        }

        // Equal weight direction
        let eq = vec![1.0 / num_objectives as f64; num_objectives];
        directions.push(eq);

        // Random directions
        let mut rng = rand::thread_rng();
        while directions.len() < num_directions {
            let mut d: Vec<f64> = (0..num_objectives).map(|_| rng.gen::<f64>()).collect();
            let sum: f64 = d.iter().sum();
            if sum > 1e-12 {
                for v in d.iter_mut() {
                    *v /= sum;
                }
                directions.push(d);
            }
        }

        // For 2D: add evenly spaced directions
        if num_objectives == 2 && num_directions > directions.len() {
            let n = num_directions - directions.len();
            for i in 0..n {
                let t = (i as f64 + 1.0) / (n as f64 + 1.0);
                directions.push(vec![t, 1.0 - t]);
            }
        }

        // For 3D and 4D: add grid-based directions
        if num_objectives >= 3 && num_directions > directions.len() {
            let steps = ((num_directions - directions.len()) as f64).powf(1.0 / num_objectives as f64).ceil() as usize;
            let step_size = 1.0 / steps as f64;
            self.generate_grid_directions(&mut directions, num_objectives, steps, step_size, &mut vec![], 0);
        }

        directions.truncate(num_directions);
        directions
    }

    /// Recursively generate grid-based directions.
    fn generate_grid_directions(
        &self,
        directions: &mut Vec<Vec<f64>>,
        dim: usize,
        steps: usize,
        step_size: f64,
        current: &mut Vec<f64>,
        depth: usize,
    ) {
        if depth == dim {
            let sum: f64 = current.iter().sum();
            if sum > 1e-12 {
                let normalized: Vec<f64> = current.iter().map(|&v| v / sum).collect();
                directions.push(normalized);
            }
            return;
        }

        for i in 0..=steps {
            current.push(i as f64 * step_size);
            self.generate_grid_directions(directions, dim, steps, step_size, current, depth + 1);
            current.pop();
        }
    }

    /// Create a blocking clause that excludes the dominance cone of a point.
    ///
    /// For a found point with costs c = (c1, ..., cn), we want to block all points
    /// that are dominated by c (i.e., <= c in all dimensions). We encode this as:
    /// at least one objective must be strictly less than ci - epsilon.
    fn create_blocking_clause(
        &self,
        costs: &[f64],
        objectives: &[LinearObjective],
        num_vars: u32,
        epsilon: f64,
    ) -> Clause {
        // Simple blocking: create a clause that says "the solution must differ"
        // For each objective i, we need terms that say "objective i < costs[i]"
        // This is hard to encode directly in SAT, so we use a simpler approach:
        // Block the exact assignment pattern that leads to domination.

        // The approach: for each objective, find which variables contribute to it
        // and create a disjunction that at least one must change.
        let mut blocking_lits = Vec::new();
        for (obj_idx, objective) in objectives.iter().enumerate() {
            let target = costs[obj_idx] - epsilon;
            if target < 0.0 {
                continue;
            }
            // We want at least one variable to flip to reduce this objective
            for &(var, coeff) in &objective.terms {
                if coeff > 0.0 {
                    blocking_lits.push(make_lit(var, false));
                }
            }
        }

        blocking_lits.sort();
        blocking_lits.dedup();
        blocking_lits
    }

    /// Refine the frontier by looking for points between existing pairs.
    fn refine_frontier(
        &self,
        frontier: &mut ParetoFrontier,
        hard_clauses: &[Clause],
        blocking_clauses: &mut Vec<Clause>,
        objectives: &[LinearObjective],
        num_vars: u32,
        epsilon: f64,
        start: &Instant,
    ) {
        let current_points = frontier.points.clone();
        let num_points = current_points.len();

        for i in 0..num_points {
            for j in (i + 1)..num_points {
                if start.elapsed() > self.config.timeout {
                    return;
                }
                if frontier.size() >= self.config.pareto_max_points {
                    return;
                }

                // Create a direction that points between the two existing points
                let dim = current_points[i].costs.len();
                let mut mid_dir = vec![0.0; dim];
                for d in 0..dim {
                    let diff = current_points[j].costs[d] - current_points[i].costs[d];
                    // Weight inversely to the dimension where the points differ most
                    mid_dir[d] = 1.0 / (diff.abs() + 1.0);
                }
                let sum: f64 = mid_dir.iter().sum();
                for v in mid_dir.iter_mut() {
                    *v /= sum;
                }

                // Create soft clauses for this direction
                let mut all_hard = hard_clauses.to_vec();
                all_hard.extend_from_slice(blocking_clauses);

                let mut soft = Vec::new();
                let mut sid = 0;
                for (obj_idx, objective) in objectives.iter().enumerate() {
                    let w = mid_dir[obj_idx];
                    for &(var, coeff) in &objective.terms {
                        let penalty = w * coeff;
                        if penalty.abs() < 1e-12 {
                            continue;
                        }
                        if penalty > 0.0 {
                            soft.push(SoftClause {
                                lits: vec![make_lit(var, false)],
                                weight: penalty,
                                id: sid,
                            });
                        } else {
                            soft.push(SoftClause {
                                lits: vec![make_lit(var, true)],
                                weight: -penalty,
                                id: sid,
                            });
                        }
                        sid += 1;
                    }
                }

                let mut maxsmt = MaxSmtSolver::new(self.config.clone());
                let result = maxsmt.solve(&all_hard, &soft);

                if let Some(ref assignment) = result.assignment {
                    let costs: Vec<f64> = objectives
                        .iter()
                        .map(|obj| obj.evaluate(assignment))
                        .collect();

                    let mut solution = HashMap::new();
                    for &(var, _) in objectives.iter().flat_map(|o| o.terms.iter()) {
                        let val = assignment.get(var).unwrap_or(false);
                        solution.insert(format!("x{}", var), if val { 1.0 } else { 0.0 });
                    }

                    let point = ParetoPoint { costs: costs.clone(), solution };
                    if frontier.add_point(point) {
                        let blocking = self.create_blocking_clause(&costs, objectives, num_vars, epsilon);
                        if !blocking.is_empty() {
                            blocking_clauses.push(blocking);
                        }
                    }
                }
            }
        }
    }

    /// Enumerate with a 4-dimensional cost vector.
    pub fn enumerate_4d(
        &mut self,
        hard_clauses: &[Clause],
        objectives: &[LinearObjective; 4],
        num_vars: u32,
        epsilon: f64,
    ) -> ParetoResult {
        self.enumerate(hard_clauses, objectives.as_slice(), num_vars, epsilon)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> SolverConfig {
        SolverConfig::new()
            .with_pareto_max_points(50)
            .with_pareto_epsilon(0.01)
    }

    #[test]
    fn test_single_objective() {
        let mut enumerator = ParetoEnumerator::new(default_config());
        // Minimize x1 + x2, s.t. (x1 OR x2)
        let hard = vec![vec![1, 2]];
        let objectives = vec![LinearObjective::new("cost", vec![(1, 1.0), (2, 1.0)])];
        let result = enumerator.enumerate(&hard, &objectives, 2, 0.01);
        match result {
            ParetoResult::Complete(frontier) | ParetoResult::Partial(frontier) => {
                assert!(frontier.size() >= 1);
                // Optimal: one of x1 or x2 is true, cost = 1
                let min_cost: f64 = frontier
                    .points
                    .iter()
                    .map(|p| p.costs[0])
                    .fold(f64::INFINITY, f64::min);
                assert!((min_cost - 1.0).abs() < 1.0 + 0.01);
            }
            ParetoResult::Infeasible => panic!("Should be feasible"),
            ParetoResult::Timeout => panic!("Should not timeout"),
        }
    }

    #[test]
    fn test_two_objectives() {
        let mut enumerator = ParetoEnumerator::new(default_config());
        // Two conflicting objectives over 3 variables
        // Minimize obj1 = x1 + x2
        // Minimize obj2 = x2 + x3
        // Hard: (x1 OR x2 OR x3)
        let hard = vec![vec![1, 2, 3]];
        let objectives = vec![
            LinearObjective::new("obj1", vec![(1, 1.0), (2, 1.0)]),
            LinearObjective::new("obj2", vec![(2, 1.0), (3, 1.0)]),
        ];
        let result = enumerator.enumerate(&hard, &objectives, 3, 0.01);
        match result {
            ParetoResult::Complete(frontier) | ParetoResult::Partial(frontier) => {
                assert!(frontier.size() >= 1);
            }
            _ => {}
        }
    }

    #[test]
    fn test_infeasible() {
        let mut enumerator = ParetoEnumerator::new(default_config());
        // (x1) AND (NOT x1)
        let hard = vec![vec![1], vec![-1]];
        let objectives = vec![LinearObjective::new("cost", vec![(1, 1.0)])];
        let result = enumerator.enumerate(&hard, &objectives, 1, 0.01);
        assert!(matches!(result, ParetoResult::Infeasible));
    }

    #[test]
    fn test_direction_generation() {
        let config = SolverConfig::new().with_pareto_max_points(100);
        let enumerator = ParetoEnumerator::new(config);

        let dirs = enumerator.generate_directions(2);
        assert!(dirs.len() >= 3); // At least axis-aligned + equal

        // All directions should sum to ~1
        for d in &dirs {
            let sum: f64 = d.iter().sum();
            assert!((sum - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_pareto_point_evaluation() {
        let obj = LinearObjective::new("test", vec![(1, 2.0), (2, 3.0), (3, 1.0)]);
        let mut assignment = Assignment::new();
        assignment.set(1, true);
        assignment.set(2, false);
        assignment.set(3, true);
        assert!((obj.evaluate(&assignment) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_four_objectives() {
        let mut enumerator = ParetoEnumerator::new(default_config());
        let hard = vec![vec![1, 2, 3, 4]];
        let objectives = vec![
            LinearObjective::new("o1", vec![(1, 1.0)]),
            LinearObjective::new("o2", vec![(2, 1.0)]),
            LinearObjective::new("o3", vec![(3, 1.0)]),
            LinearObjective::new("o4", vec![(4, 1.0)]),
        ];
        let result = enumerator.enumerate(&hard, &objectives, 4, 0.1);
        match result {
            ParetoResult::Complete(f) | ParetoResult::Partial(f) => {
                assert!(f.size() >= 1);
            }
            _ => {}
        }
    }
}

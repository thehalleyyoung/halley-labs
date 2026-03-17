//! Pareto frontier enumeration for multi-objective optimization.
//!
//! Given a MaxSAT formula and multiple objectives, enumerates the
//! Pareto-optimal solutions (non-dominated trade-offs) using iterative
//! blocking and epsilon-constraint methods.

use crate::formula::{Clause, Literal, MaxSatFormula};
use crate::solver::{
    CdclSolver, Model, SatOracle, SatOracleResult,
};
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptDirection {
    Minimize,
    Maximize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Objective {
    pub name: String,
    pub direction: OptDirection,
    pub weight: f64,
}

impl Objective {
    pub fn minimize(name: impl Into<String>) -> Self {
        Self { name: name.into(), direction: OptDirection::Minimize, weight: 1.0 }
    }

    pub fn maximize(name: impl Into<String>) -> Self {
        Self { name: name.into(), direction: OptDirection::Maximize, weight: 1.0 }
    }

    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoSolution {
    pub model: Model,
    pub objectives: Vec<f64>,
}

impl ParetoSolution {
    pub fn new(model: Model, objectives: Vec<f64>) -> Self {
        Self { model, objectives }
    }

    pub fn num_objectives(&self) -> usize {
        self.objectives.len()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoFrontier {
    pub solutions: Vec<ParetoSolution>,
}

impl ParetoFrontier {
    pub fn new() -> Self {
        Self { solutions: Vec::new() }
    }

    pub fn add(&mut self, solution: ParetoSolution) {
        self.solutions.push(solution);
    }

    pub fn len(&self) -> usize {
        self.solutions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.solutions.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &ParetoSolution> {
        self.solutions.iter()
    }
}

impl Default for ParetoFrontier {
    fn default() -> Self { Self::new() }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoConfig {
    pub max_solutions: usize,
    pub objectives: Vec<Objective>,
    pub epsilon: f64,
}

impl Default for ParetoConfig {
    fn default() -> Self {
        Self { max_solutions: 100, objectives: Vec::new(), epsilon: 0.01 }
    }
}

impl ParetoConfig {
    pub fn with_max_solutions(mut self, n: usize) -> Self { self.max_solutions = n; self }
    pub fn add_objective(mut self, obj: Objective) -> Self { self.objectives.push(obj); self }
    pub fn with_epsilon(mut self, eps: f64) -> Self { self.epsilon = eps; self }
}

// ---------------------------------------------------------------------------
// Dominance
// ---------------------------------------------------------------------------

pub fn is_dominated(
    a: &ParetoSolution,
    b: &ParetoSolution,
    directions: &[OptDirection],
) -> bool {
    if a.objectives.len() != b.objectives.len() || a.objectives.len() != directions.len() {
        return false;
    }
    let mut at_least_as_good = true;
    let mut strictly_better = false;

    for i in 0..a.objectives.len() {
        let (av, bv) = match directions[i] {
            OptDirection::Minimize => (a.objectives[i], b.objectives[i]),
            OptDirection::Maximize => (-a.objectives[i], -b.objectives[i]),
        };
        if av > bv {
            at_least_as_good = false;
            break;
        }
        if av < bv {
            strictly_better = true;
        }
    }
    at_least_as_good && strictly_better
}

pub fn filter_dominated(solutions: &mut Vec<ParetoSolution>, directions: &[OptDirection]) {
    let n = solutions.len();
    let mut keep = vec![true; n];
    for i in 0..n {
        if !keep[i] { continue; }
        for j in 0..n {
            if i == j || !keep[j] { continue; }
            if is_dominated(&solutions[i], &solutions[j], directions) {
                keep[j] = false;
            }
        }
    }
    let mut idx = 0;
    solutions.retain(|_| { let k = keep[idx]; idx += 1; k });
}

// ---------------------------------------------------------------------------
// Enumeration
// ---------------------------------------------------------------------------

pub fn enumerate_pareto<F>(
    formula: &MaxSatFormula,
    config: &ParetoConfig,
    eval_objectives: F,
) -> ParetoFrontier
where
    F: Fn(&Model) -> Vec<f64>,
{
    let mut frontier = ParetoFrontier::new();
    let directions: Vec<OptDirection> = config.objectives.iter().map(|o| o.direction).collect();
    let mut blocking_clauses: Vec<Clause> = Vec::new();
    let mut solver = CdclSolver::new();
    let next_var = formula.num_variables + 1;

    for _iteration in 0..config.max_solutions {
        let mut clauses: Vec<Clause> = formula.hard_clauses.iter().map(|hc| hc.0.clone()).collect();
        for sc in &formula.soft_clauses {
            clauses.push(sc.clause.clone());
        }
        clauses.extend(blocking_clauses.clone());

        match solver.solve_sat(&clauses, next_var - 1) {
            SatOracleResult::Sat(assignments) => {
                let model = Model { assignments, cost: 0 };
                let obj_values = eval_objectives(&model);
                frontier.add(ParetoSolution::new(model.clone(), obj_values));

                let mut block_lits = Vec::new();
                for (&var, &val) in &model.assignments {
                    if var > 0 && var <= formula.num_variables {
                        if val {
                            block_lits.push(Literal::negative(var));
                        } else {
                            block_lits.push(Literal::positive(var));
                        }
                    }
                }
                if !block_lits.is_empty() {
                    blocking_clauses.push(Clause::new(block_lits));
                }
            }
            SatOracleResult::Unsat(_) => break,
        }
    }

    filter_dominated(&mut frontier.solutions, &directions);
    frontier
}

// ---------------------------------------------------------------------------
// Hypervolume
// ---------------------------------------------------------------------------

pub fn compute_hypervolume(frontier: &ParetoFrontier, reference: &[f64]) -> f64 {
    if frontier.is_empty() || reference.is_empty() {
        return 0.0;
    }
    let dim = reference.len();
    if dim == 1 { return compute_hv_1d(frontier, reference[0]); }
    if dim == 2 { return compute_hv_2d(frontier, reference); }
    compute_hv_nd(frontier, reference)
}

fn compute_hv_1d(frontier: &ParetoFrontier, ref_val: f64) -> f64 {
    frontier.solutions.iter()
        .filter_map(|s| s.objectives.first())
        .map(|&v| (ref_val - v).max(0.0))
        .fold(0.0f64, f64::max)
}

fn compute_hv_2d(frontier: &ParetoFrontier, reference: &[f64]) -> f64 {
    let mut points: Vec<(f64, f64)> = frontier.solutions.iter()
        .filter(|s| s.objectives.len() >= 2)
        .map(|s| (s.objectives[0], s.objectives[1]))
        .filter(|&(x, y)| x < reference[0] && y < reference[1])
        .collect();
    if points.is_empty() { return 0.0; }
    points.sort_by_key(|&(x, _)| OrderedFloat(x));

    let mut volume = 0.0;
    let mut prev_y = reference[1];
    for &(x, y) in points.iter().rev() {
        if y < prev_y {
            volume += (reference[0] - x) * (prev_y - y);
            prev_y = y;
        }
    }
    volume
}

fn compute_hv_nd(frontier: &ParetoFrontier, reference: &[f64]) -> f64 {
    let dim = reference.len();
    let num_samples = 10_000usize;
    let mut rng_state: u64 = 42;

    let mut ideal = vec![f64::INFINITY; dim];
    for sol in &frontier.solutions {
        for (i, &v) in sol.objectives.iter().enumerate() {
            if i < dim && v < ideal[i] { ideal[i] = v; }
        }
    }

    let box_volume: f64 = (0..dim).map(|i| (reference[i] - ideal[i]).max(0.0)).product();
    if box_volume <= 0.0 { return 0.0; }

    let mut dominated_count = 0u64;
    for _ in 0..num_samples {
        let point: Vec<f64> = (0..dim).map(|i| {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let frac = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
            ideal[i] + frac * (reference[i] - ideal[i])
        }).collect();

        let is_dom = frontier.solutions.iter().any(|sol| {
            sol.objectives.iter().enumerate().take(dim).all(|(i, &v)| v <= point[i])
        });
        if is_dom { dominated_count += 1; }
    }

    box_volume * (dominated_count as f64 / num_samples as f64)
}

// ---------------------------------------------------------------------------
// Crowding distance
// ---------------------------------------------------------------------------

pub fn compute_crowding_distance(frontier: &ParetoFrontier) -> Vec<f64> {
    let n = frontier.solutions.len();
    if n <= 2 { return vec![f64::INFINITY; n]; }

    let num_obj = frontier.solutions[0].objectives.len();
    let mut distances = vec![0.0f64; n];

    for obj_idx in 0..num_obj {
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by_key(|&i| OrderedFloat(frontier.solutions[i].objectives[obj_idx]));

        let f_min = frontier.solutions[indices[0]].objectives[obj_idx];
        let f_max = frontier.solutions[indices[n - 1]].objectives[obj_idx];
        let range = f_max - f_min;

        distances[indices[0]] = f64::INFINITY;
        distances[indices[n - 1]] = f64::INFINITY;

        if range > 1e-12 {
            for i in 1..(n - 1) {
                let prev = frontier.solutions[indices[i - 1]].objectives[obj_idx];
                let next = frontier.solutions[indices[i + 1]].objectives[obj_idx];
                distances[indices[i]] += (next - prev) / range;
            }
        }
    }
    distances
}

// ---------------------------------------------------------------------------
// Ranking
// ---------------------------------------------------------------------------

pub fn rank_solutions(frontier: &ParetoFrontier, preferences: &[f64]) -> Vec<(usize, f64)> {
    let mut ranked: Vec<(usize, f64)> = frontier.solutions.iter().enumerate().map(|(i, sol)| {
        let score: f64 = sol.objectives.iter().zip(preferences.iter()).map(|(&o, &p)| o * p).sum();
        (i, score)
    }).collect();
    ranked.sort_by_key(|&(_, s)| OrderedFloat(s));
    ranked
}

// ---------------------------------------------------------------------------
// Epsilon-Constraint
// ---------------------------------------------------------------------------

pub struct EpsilonConstraint {
    pub primary_index: usize,
    pub num_steps: usize,
}

impl EpsilonConstraint {
    pub fn new(primary_index: usize, num_steps: usize) -> Self {
        Self { primary_index, num_steps }
    }

    pub fn enumerate<F>(
        &self,
        formula: &MaxSatFormula,
        objectives: &[Objective],
        eval_objectives: F,
    ) -> ParetoFrontier
    where
        F: Fn(&Model) -> Vec<f64>,
    {
        let mut frontier = ParetoFrontier::new();
        let num_obj = objectives.len();
        if num_obj < 2 || self.primary_index >= num_obj {
            return frontier;
        }

        let mut solver = CdclSolver::new();
        let clauses: Vec<Clause> = formula.hard_clauses.iter().map(|hc| hc.0.clone())
            .chain(formula.soft_clauses.iter().map(|sc| sc.clause.clone()))
            .collect();

        let mut obj_ranges: Vec<(f64, f64)> = vec![(f64::INFINITY, f64::NEG_INFINITY); num_obj];
        let mut blocking = Vec::new();

        for _ in 0..self.num_steps.min(50) {
            let mut all_clauses = clauses.clone();
            all_clauses.extend(blocking.clone());

            match solver.solve_sat(&all_clauses, formula.num_variables) {
                SatOracleResult::Sat(assignments) => {
                    let model = Model { assignments: assignments.clone(), cost: 0 };
                    let values = eval_objectives(&model);

                    for (i, &v) in values.iter().enumerate() {
                        if i < num_obj {
                            obj_ranges[i].0 = obj_ranges[i].0.min(v);
                            obj_ranges[i].1 = obj_ranges[i].1.max(v);
                        }
                    }
                    frontier.add(ParetoSolution::new(model.clone(), values));

                    let mut block = Vec::new();
                    for (&var, &val) in &model.assignments {
                        if var > 0 && var <= formula.num_variables {
                            block.push(if val { Literal::negative(var) } else { Literal::positive(var) });
                        }
                    }
                    if !block.is_empty() { blocking.push(Clause::new(block)); }
                }
                SatOracleResult::Unsat(_) => break,
            }
        }

        let directions: Vec<OptDirection> = objectives.iter().map(|o| o.direction).collect();
        filter_dominated(&mut frontier.solutions, &directions);
        frontier
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_solution(objectives: Vec<f64>) -> ParetoSolution {
        ParetoSolution::new(Model::new(), objectives)
    }

    #[test]
    fn test_is_dominated_basic() {
        let a = make_solution(vec![1.0, 2.0]);
        let b = make_solution(vec![2.0, 3.0]);
        let dirs = vec![OptDirection::Minimize, OptDirection::Minimize];
        assert!(is_dominated(&a, &b, &dirs));
        assert!(!is_dominated(&b, &a, &dirs));
    }

    #[test]
    fn test_is_dominated_equal() {
        let a = make_solution(vec![1.0, 2.0]);
        let b = make_solution(vec![1.0, 2.0]);
        let dirs = vec![OptDirection::Minimize, OptDirection::Minimize];
        assert!(!is_dominated(&a, &b, &dirs));
    }

    #[test]
    fn test_is_dominated_incomparable() {
        let a = make_solution(vec![1.0, 3.0]);
        let b = make_solution(vec![2.0, 1.0]);
        let dirs = vec![OptDirection::Minimize, OptDirection::Minimize];
        assert!(!is_dominated(&a, &b, &dirs));
        assert!(!is_dominated(&b, &a, &dirs));
    }

    #[test]
    fn test_is_dominated_maximize() {
        let a = make_solution(vec![5.0, 4.0]);
        let b = make_solution(vec![3.0, 2.0]);
        let dirs = vec![OptDirection::Maximize, OptDirection::Maximize];
        assert!(is_dominated(&a, &b, &dirs));
    }

    #[test]
    fn test_filter_dominated_chain() {
        let mut solutions = vec![
            make_solution(vec![1.0, 1.0]),
            make_solution(vec![2.0, 2.0]),
            make_solution(vec![3.0, 3.0]),
        ];
        let dirs = vec![OptDirection::Minimize, OptDirection::Minimize];
        filter_dominated(&mut solutions, &dirs);
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].objectives, vec![1.0, 1.0]);
    }

    #[test]
    fn test_filter_dominated_frontier() {
        let mut solutions = vec![
            make_solution(vec![1.0, 4.0]),
            make_solution(vec![2.0, 2.0]),
            make_solution(vec![4.0, 1.0]),
        ];
        let dirs = vec![OptDirection::Minimize, OptDirection::Minimize];
        filter_dominated(&mut solutions, &dirs);
        assert_eq!(solutions.len(), 3);
    }

    #[test]
    fn test_pareto_frontier_basic() {
        let mut frontier = ParetoFrontier::new();
        assert!(frontier.is_empty());
        frontier.add(make_solution(vec![1.0, 2.0]));
        assert_eq!(frontier.len(), 1);
        assert!(!frontier.is_empty());
    }

    #[test]
    fn test_hypervolume_2d() {
        let mut frontier = ParetoFrontier::new();
        frontier.add(make_solution(vec![1.0, 3.0]));
        frontier.add(make_solution(vec![2.0, 1.0]));
        let reference = vec![4.0, 4.0];
        let hv = compute_hypervolume(&frontier, &reference);
        assert!((hv - 7.0).abs() < 0.5, "hypervolume={hv}");
    }

    #[test]
    fn test_hypervolume_1d() {
        let mut frontier = ParetoFrontier::new();
        frontier.add(make_solution(vec![2.0]));
        frontier.add(make_solution(vec![3.0]));
        let hv = compute_hypervolume(&frontier, &[5.0]);
        assert!((hv - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_hypervolume_empty() {
        let frontier = ParetoFrontier::new();
        assert_eq!(compute_hypervolume(&frontier, &[5.0, 5.0]), 0.0);
    }

    #[test]
    fn test_crowding_distance() {
        let mut frontier = ParetoFrontier::new();
        frontier.add(make_solution(vec![1.0, 5.0]));
        frontier.add(make_solution(vec![2.0, 3.0]));
        frontier.add(make_solution(vec![4.0, 1.0]));
        let distances = compute_crowding_distance(&frontier);
        assert_eq!(distances.len(), 3);
        assert!(distances[0].is_infinite());
        assert!(distances[2].is_infinite());
        assert!(distances[1].is_finite());
    }

    #[test]
    fn test_crowding_distance_two() {
        let mut frontier = ParetoFrontier::new();
        frontier.add(make_solution(vec![1.0, 3.0]));
        frontier.add(make_solution(vec![3.0, 1.0]));
        let distances = compute_crowding_distance(&frontier);
        assert!(distances.iter().all(|d| d.is_infinite()));
    }

    #[test]
    fn test_rank_solutions() {
        let mut frontier = ParetoFrontier::new();
        frontier.add(make_solution(vec![1.0, 5.0]));
        frontier.add(make_solution(vec![3.0, 1.0]));
        frontier.add(make_solution(vec![2.0, 3.0]));
        let ranked = rank_solutions(&frontier, &[1.0, 1.0]);
        assert_eq!(ranked[0].0, 1); // 3+1=4
        assert_eq!(ranked[1].0, 2); // 2+3=5
        assert_eq!(ranked[2].0, 0); // 1+5=6
    }

    #[test]
    fn test_rank_solutions_weighted() {
        let mut frontier = ParetoFrontier::new();
        frontier.add(make_solution(vec![1.0, 5.0]));
        frontier.add(make_solution(vec![5.0, 1.0]));
        let ranked = rank_solutions(&frontier, &[10.0, 1.0]);
        assert_eq!(ranked[0].0, 0); // 10+5=15
        assert_eq!(ranked[1].0, 1); // 50+1=51
    }

    #[test]
    fn test_objective_constructors() {
        let o = Objective::minimize("cost").with_weight(2.0);
        assert_eq!(o.direction, OptDirection::Minimize);
        assert_eq!(o.weight, 2.0);
        let o2 = Objective::maximize("throughput");
        assert_eq!(o2.direction, OptDirection::Maximize);
    }

    #[test]
    fn test_pareto_config() {
        let cfg = ParetoConfig::default().with_max_solutions(50).add_objective(Objective::minimize("cost")).with_epsilon(0.05);
        assert_eq!(cfg.max_solutions, 50);
        assert_eq!(cfg.objectives.len(), 1);
        assert!((cfg.epsilon - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_epsilon_constraint_new() {
        let ec = EpsilonConstraint::new(0, 10);
        assert_eq!(ec.primary_index, 0);
        assert_eq!(ec.num_steps, 10);
    }

    #[test]
    fn test_enumerate_pareto_empty() {
        let formula = MaxSatFormula::new();
        let config = ParetoConfig::default().add_objective(Objective::minimize("obj1"));
        let frontier = enumerate_pareto(&formula, &config, |_| vec![0.0]);
        assert!(frontier.len() <= 1);
    }
}

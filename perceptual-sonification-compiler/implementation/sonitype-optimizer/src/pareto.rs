//! Multi-objective optimization: Pareto front computation, NSGA-II style
//! non-dominated sorting, crowding distance, and archive maintenance.

use std::collections::HashMap;
use std::fmt;

use crate::{
    MappingConfig, OptimizerError, OptimizerResult, OptimizationSolution,
    StreamId, StreamMapping,
};
use crate::config::OptimizerConfig;
use crate::constraints::ConstraintSet;
use crate::objective::ObjectiveFn;

// ─────────────────────────────────────────────────────────────────────────────
// ObjectiveVector
// ─────────────────────────────────────────────────────────────────────────────

/// A vector of objective values for a candidate solution.
#[derive(Debug, Clone)]
pub struct ObjectiveVector {
    pub values: Vec<f64>,
    pub names: Vec<String>,
}

impl ObjectiveVector {
    pub fn new(names: Vec<String>, values: Vec<f64>) -> Self {
        ObjectiveVector { values, names }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Check if self dominates other (all objectives at least as good,
    /// at least one strictly better). Assumes maximization.
    pub fn dominates(&self, other: &ObjectiveVector) -> bool {
        if self.values.len() != other.values.len() {
            return false;
        }
        let mut at_least_one_better = false;
        for (a, b) in self.values.iter().zip(other.values.iter()) {
            if a < b {
                return false;
            }
            if a > b {
                at_least_one_better = true;
            }
        }
        at_least_one_better
    }

    /// Euclidean distance to another objective vector.
    pub fn distance(&self, other: &ObjectiveVector) -> f64 {
        self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    }

    /// Scalarize with weights.
    pub fn weighted_sum(&self, weights: &[f64]) -> f64 {
        self.values
            .iter()
            .zip(weights.iter())
            .map(|(v, w)| v * w)
            .sum()
    }

    /// Get value by name.
    pub fn get(&self, name: &str) -> Option<f64> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|i| self.values[i])
    }
}

impl fmt::Display for ObjectiveVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self
            .names
            .iter()
            .zip(self.values.iter())
            .map(|(n, v)| format!("{}={:.4}", n, v))
            .collect();
        write!(f, "({})", parts.join(", "))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ParetoSolution
// ─────────────────────────────────────────────────────────────────────────────

/// A solution with its objective vector.
#[derive(Debug, Clone)]
pub struct ParetoSolution {
    pub config: MappingConfig,
    pub objectives: ObjectiveVector,
    pub rank: usize,
    pub crowding_distance: f64,
}

impl ParetoSolution {
    pub fn new(config: MappingConfig, objectives: ObjectiveVector) -> Self {
        ParetoSolution {
            config,
            objectives,
            rank: 0,
            crowding_distance: 0.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ParetoFront
// ─────────────────────────────────────────────────────────────────────────────

/// Set of non-dominated solutions forming the Pareto front.
#[derive(Debug, Clone)]
pub struct ParetoFront {
    pub solutions: Vec<ParetoSolution>,
    pub max_size: usize,
}

impl ParetoFront {
    pub fn new(max_size: usize) -> Self {
        ParetoFront {
            solutions: Vec::new(),
            max_size,
        }
    }

    /// Add a solution; remove any solutions it dominates.
    /// Returns true if the solution was added.
    pub fn add(&mut self, solution: ParetoSolution) -> bool {
        // Check if dominated by any existing solution
        for existing in &self.solutions {
            if existing.objectives.dominates(&solution.objectives) {
                return false;
            }
        }

        // Remove solutions dominated by the new one
        self.solutions
            .retain(|s| !solution.objectives.dominates(&s.objectives));

        // Add the new solution
        self.solutions.push(solution);

        // If over capacity, remove the solution with smallest crowding distance
        if self.solutions.len() > self.max_size {
            self.compute_crowding_distances();
            let min_idx = self
                .solutions
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    a.1.crowding_distance
                        .partial_cmp(&b.1.crowding_distance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i);
            if let Some(idx) = min_idx {
                self.solutions.remove(idx);
            }
        }

        true
    }

    /// Check if a solution is dominated by any solution in the front.
    pub fn is_dominated(&self, objectives: &ObjectiveVector) -> bool {
        self.solutions
            .iter()
            .any(|s| s.objectives.dominates(objectives))
    }

    /// Number of solutions on the front.
    pub fn len(&self) -> usize {
        self.solutions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.solutions.is_empty()
    }

    /// Compute the hypervolume indicator relative to a reference point.
    pub fn hypervolume(&self, reference: &[f64]) -> f64 {
        if self.solutions.is_empty() || reference.is_empty() {
            return 0.0;
        }

        let dim = reference.len();
        if dim == 1 {
            return self.hypervolume_1d(reference);
        }
        if dim == 2 {
            return self.hypervolume_2d(reference);
        }

        // For higher dimensions, use inclusion-exclusion approximation
        self.hypervolume_approximate(reference)
    }

    fn hypervolume_1d(&self, reference: &[f64]) -> f64 {
        self.solutions
            .iter()
            .map(|s| (s.objectives.values[0] - reference[0]).max(0.0))
            .fold(0.0_f64, f64::max)
    }

    fn hypervolume_2d(&self, reference: &[f64]) -> f64 {
        let mut points: Vec<(f64, f64)> = self
            .solutions
            .iter()
            .map(|s| (s.objectives.values[0], s.objectives.values[1]))
            .filter(|(x, y)| *x > reference[0] && *y > reference[1])
            .collect();

        // Sort by first objective descending
        points.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut hv = 0.0;
        let mut prev_y = reference[1];

        for (x, y) in &points {
            if *y > prev_y {
                hv += (*x - reference[0]) * (*y - prev_y);
                prev_y = *y;
            }
        }

        hv
    }

    fn hypervolume_approximate(&self, reference: &[f64]) -> f64 {
        // Monte Carlo approximation for high dimensions
        let n_samples = 10000;
        let dim = reference.len();

        // Find bounds
        let mut upper = vec![f64::NEG_INFINITY; dim];
        for sol in &self.solutions {
            for (i, &v) in sol.objectives.values.iter().enumerate() {
                if i < dim && v > upper[i] {
                    upper[i] = v;
                }
            }
        }

        let total_volume: f64 = (0..dim)
            .map(|i| (upper[i] - reference[i]).max(0.0))
            .product();

        if total_volume <= 0.0 {
            return 0.0;
        }

        // Simple deterministic sampling (grid-like)
        let mut dominated_count = 0;
        let steps = (n_samples as f64).powf(1.0 / dim as f64).ceil() as usize;

        let mut point = vec![0.0; dim];
        let mut idx = 0;
        let total_points = steps.pow(dim as u32);

        for sample_idx in 0..total_points.min(n_samples) {
            let mut tmp = sample_idx;
            for d in 0..dim {
                let step_val = tmp % steps;
                tmp /= steps;
                point[d] = reference[d]
                    + (step_val as f64 + 0.5) / steps as f64 * (upper[d] - reference[d]);
            }

            let sample_obj = ObjectiveVector::new(vec![], point.clone());
            if self
                .solutions
                .iter()
                .any(|s| s.objectives.dominates(&sample_obj))
            {
                dominated_count += 1;
            }
        }

        let fraction = dominated_count as f64 / total_points.min(n_samples) as f64;
        fraction * total_volume
    }

    /// Compute crowding distances for all solutions in the front.
    pub fn compute_crowding_distances(&mut self) {
        let n = self.solutions.len();
        if n <= 2 {
            for sol in &mut self.solutions {
                sol.crowding_distance = f64::INFINITY;
            }
            return;
        }

        // Reset distances
        for sol in &mut self.solutions {
            sol.crowding_distance = 0.0;
        }

        let num_objectives = self.solutions[0].objectives.len();

        for obj_idx in 0..num_objectives {
            // Sort by this objective
            let mut indices: Vec<usize> = (0..n).collect();
            indices.sort_by(|&a, &b| {
                self.solutions[a].objectives.values[obj_idx]
                    .partial_cmp(&self.solutions[b].objectives.values[obj_idx])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Boundary points get infinite distance
            let first = indices[0];
            let last = indices[n - 1];
            self.solutions[first].crowding_distance = f64::INFINITY;
            self.solutions[last].crowding_distance = f64::INFINITY;

            let range = self.solutions[last].objectives.values[obj_idx]
                - self.solutions[first].objectives.values[obj_idx];

            if range < 1e-12 {
                continue;
            }

            for i in 1..(n - 1) {
                let prev = indices[i - 1];
                let next = indices[i + 1];
                let idx = indices[i];
                let dist = (self.solutions[next].objectives.values[obj_idx]
                    - self.solutions[prev].objectives.values[obj_idx])
                    / range;
                self.solutions[idx].crowding_distance += dist;
            }
        }
    }

    /// Select solution closest to given preference weights.
    pub fn select_by_weights(&self, weights: &[f64]) -> Option<&ParetoSolution> {
        self.solutions
            .iter()
            .max_by(|a, b| {
                let wa = a.objectives.weighted_sum(weights);
                let wb = b.objectives.weighted_sum(weights);
                wa.partial_cmp(&wb).unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WeightedSum scalarizer
// ─────────────────────────────────────────────────────────────────────────────

/// Scalarize multiple objectives via weighted sum.
#[derive(Debug, Clone)]
pub struct WeightedSum {
    pub weights: Vec<f64>,
    pub names: Vec<String>,
}

impl WeightedSum {
    pub fn new(names: Vec<String>, weights: Vec<f64>) -> Self {
        WeightedSum { weights, names }
    }

    pub fn uniform(names: Vec<String>) -> Self {
        let n = names.len();
        let w = 1.0 / n as f64;
        WeightedSum {
            weights: vec![w; n],
            names,
        }
    }

    pub fn scalarize(&self, objectives: &ObjectiveVector) -> f64 {
        objectives.weighted_sum(&self.weights)
    }

    pub fn normalize(&mut self) {
        let total: f64 = self.weights.iter().sum();
        if total > 0.0 {
            for w in &mut self.weights {
                *w /= total;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Non-dominated sorting (NSGA-II)
// ─────────────────────────────────────────────────────────────────────────────

/// Perform NSGA-II style non-dominated sorting on a population.
/// Returns a vector of fronts, where each front is a vector of indices.
pub fn non_dominated_sort(solutions: &[ParetoSolution]) -> Vec<Vec<usize>> {
    let n = solutions.len();
    let mut domination_count = vec![0usize; n];
    let mut dominated_by: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut fronts: Vec<Vec<usize>> = Vec::new();

    // Compute domination relationships
    for i in 0..n {
        for j in (i + 1)..n {
            if solutions[i].objectives.dominates(&solutions[j].objectives) {
                dominated_by[i].push(j);
                domination_count[j] += 1;
            } else if solutions[j].objectives.dominates(&solutions[i].objectives) {
                dominated_by[j].push(i);
                domination_count[i] += 1;
            }
        }
    }

    // First front: solutions not dominated by any
    let mut front: Vec<usize> = (0..n).filter(|&i| domination_count[i] == 0).collect();

    while !front.is_empty() {
        fronts.push(front.clone());
        let mut next_front = Vec::new();

        for &i in &front {
            for &j in &dominated_by[i] {
                domination_count[j] -= 1;
                if domination_count[j] == 0 {
                    next_front.push(j);
                }
            }
        }

        front = next_front;
    }

    fronts
}

// ─────────────────────────────────────────────────────────────────────────────
// ParetoOptimizer
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-objective optimizer finding the Pareto front.
pub struct ParetoOptimizer {
    pub config: OptimizerConfig,
    pub objectives: Vec<Box<dyn ObjectiveFn>>,
}

impl ParetoOptimizer {
    pub fn new(config: OptimizerConfig) -> Self {
        ParetoOptimizer {
            config,
            objectives: Vec::new(),
        }
    }

    pub fn add_objective(&mut self, objective: Box<dyn ObjectiveFn>) {
        self.objectives.push(objective);
    }

    /// Evaluate all objectives for a configuration.
    pub fn evaluate(&self, config: &MappingConfig) -> OptimizerResult<ObjectiveVector> {
        let mut values = Vec::new();
        let mut names = Vec::new();

        for obj in &self.objectives {
            let val = obj.evaluate(config)?;
            values.push(val);
            names.push(obj.name().to_string());
        }

        Ok(ObjectiveVector::new(names, values))
    }

    /// Run multi-objective optimization via random sampling + Pareto filtering.
    pub fn optimize(
        &self,
        initial_configs: Vec<MappingConfig>,
        constraints: &ConstraintSet,
    ) -> OptimizerResult<ParetoFront> {
        let archive_size = self.config.multi_objective.archive_size;
        let mut front = ParetoFront::new(archive_size);

        for config in &initial_configs {
            let report = constraints.check_all(config);
            if report.all_satisfied {
                let objectives = self.evaluate(config)?;
                let solution = ParetoSolution::new(config.clone(), objectives);
                front.add(solution);
            }
        }

        // Generate additional candidates by perturbing existing solutions
        let num_generations = self.config.multi_objective.num_generations;
        let pop_size = self.config.multi_objective.population_size;

        for gen in 0..num_generations {
            if front.is_empty() {
                break;
            }

            let base_idx = gen % front.solutions.len();
            let base_config = front.solutions[base_idx].config.clone();

            // Create perturbations
            for p in 0..pop_size.min(10) {
                let mut perturbed = base_config.clone();
                // Perturb global params
                for (_, val) in perturbed.global_params.iter_mut() {
                    let perturbation = (gen as f64 * 0.01 + p as f64 * 0.1).sin() * 0.1;
                    *val *= 1.0 + perturbation;
                }

                // Perturb stream params
                let stream_ids: Vec<StreamId> = perturbed.stream_params.keys().cloned().collect();
                for sid in &stream_ids {
                    if let Some(mapping) = perturbed.stream_params.get_mut(sid) {
                        let freq_perturb =
                            (gen as f64 * 0.037 + p as f64 * 0.23).sin() * 50.0;
                        mapping.frequency_hz = (mapping.frequency_hz + freq_perturb).max(20.0);
                        let amp_perturb =
                            (gen as f64 * 0.071 + p as f64 * 0.13).cos() * 3.0;
                        mapping.amplitude_db += amp_perturb;
                    }
                }

                let report = constraints.check_all(&perturbed);
                if report.all_satisfied {
                    if let Ok(objectives) = self.evaluate(&perturbed) {
                        front.add(ParetoSolution::new(perturbed, objectives));
                    }
                }
            }
        }

        front.compute_crowding_distances();
        Ok(front)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn obj_vec(vals: Vec<f64>) -> ObjectiveVector {
        let names: Vec<String> = (0..vals.len()).map(|i| format!("obj_{}", i)).collect();
        ObjectiveVector::new(names, vals)
    }

    #[test]
    fn test_dominance_clear() {
        let a = obj_vec(vec![3.0, 4.0]);
        let b = obj_vec(vec![1.0, 2.0]);
        assert!(a.dominates(&b));
        assert!(!b.dominates(&a));
    }

    #[test]
    fn test_dominance_incomparable() {
        let a = obj_vec(vec![3.0, 1.0]);
        let b = obj_vec(vec![1.0, 3.0]);
        assert!(!a.dominates(&b));
        assert!(!b.dominates(&a));
    }

    #[test]
    fn test_dominance_equal() {
        let a = obj_vec(vec![2.0, 2.0]);
        let b = obj_vec(vec![2.0, 2.0]);
        assert!(!a.dominates(&b));
    }

    #[test]
    fn test_pareto_front_add() {
        let mut front = ParetoFront::new(100);
        let sol1 = ParetoSolution::new(MappingConfig::new(), obj_vec(vec![3.0, 1.0]));
        let sol2 = ParetoSolution::new(MappingConfig::new(), obj_vec(vec![1.0, 3.0]));
        let sol3 = ParetoSolution::new(MappingConfig::new(), obj_vec(vec![2.0, 2.0]));

        assert!(front.add(sol1));
        assert!(front.add(sol2));
        assert!(front.add(sol3));
        assert_eq!(front.len(), 3); // All non-dominated
    }

    #[test]
    fn test_pareto_front_removes_dominated() {
        let mut front = ParetoFront::new(100);
        let sol1 = ParetoSolution::new(MappingConfig::new(), obj_vec(vec![1.0, 1.0]));
        let sol2 = ParetoSolution::new(MappingConfig::new(), obj_vec(vec![3.0, 3.0]));

        front.add(sol1);
        assert_eq!(front.len(), 1);
        front.add(sol2);
        assert_eq!(front.len(), 1); // sol1 was dominated
    }

    #[test]
    fn test_pareto_front_rejects_dominated() {
        let mut front = ParetoFront::new(100);
        let sol1 = ParetoSolution::new(MappingConfig::new(), obj_vec(vec![3.0, 3.0]));
        let sol2 = ParetoSolution::new(MappingConfig::new(), obj_vec(vec![1.0, 1.0]));

        front.add(sol1);
        assert!(!front.add(sol2)); // Dominated by sol1
        assert_eq!(front.len(), 1);
    }

    #[test]
    fn test_hypervolume_2d() {
        let mut front = ParetoFront::new(100);
        front.add(ParetoSolution::new(MappingConfig::new(), obj_vec(vec![3.0, 1.0])));
        front.add(ParetoSolution::new(MappingConfig::new(), obj_vec(vec![1.0, 3.0])));

        let hv = front.hypervolume(&[0.0, 0.0]);
        assert!(hv > 0.0, "Hypervolume should be positive, got {}", hv);
    }

    #[test]
    fn test_crowding_distance() {
        let mut front = ParetoFront::new(100);
        front.add(ParetoSolution::new(MappingConfig::new(), obj_vec(vec![1.0, 5.0])));
        front.add(ParetoSolution::new(MappingConfig::new(), obj_vec(vec![3.0, 3.0])));
        front.add(ParetoSolution::new(MappingConfig::new(), obj_vec(vec![5.0, 1.0])));

        front.compute_crowding_distances();
        // Boundary solutions should have infinite distance
        let distances: Vec<f64> = front.solutions.iter().map(|s| s.crowding_distance).collect();
        assert!(distances.iter().any(|d| d.is_infinite()));
    }

    #[test]
    fn test_weighted_sum() {
        let ws = WeightedSum::new(
            vec!["a".into(), "b".into()],
            vec![0.7, 0.3],
        );
        let obj = obj_vec(vec![10.0, 20.0]);
        let scalar = ws.scalarize(&obj);
        assert!((scalar - 13.0).abs() < 0.01);
    }

    #[test]
    fn test_weighted_sum_normalize() {
        let mut ws = WeightedSum::new(
            vec!["a".into(), "b".into()],
            vec![2.0, 3.0],
        );
        ws.normalize();
        let total: f64 = ws.weights.iter().sum();
        assert!((total - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_non_dominated_sort() {
        let solutions = vec![
            ParetoSolution::new(MappingConfig::new(), obj_vec(vec![5.0, 5.0])),
            ParetoSolution::new(MappingConfig::new(), obj_vec(vec![3.0, 3.0])),
            ParetoSolution::new(MappingConfig::new(), obj_vec(vec![1.0, 1.0])),
            ParetoSolution::new(MappingConfig::new(), obj_vec(vec![4.0, 2.0])),
        ];

        let fronts = non_dominated_sort(&solutions);
        assert!(!fronts.is_empty());
        // First front should contain solution 0 (5,5) as it dominates all others
        assert!(fronts[0].contains(&0));
    }

    #[test]
    fn test_select_by_weights() {
        let mut front = ParetoFront::new(100);
        front.add(ParetoSolution::new(MappingConfig::new(), obj_vec(vec![5.0, 1.0])));
        front.add(ParetoSolution::new(MappingConfig::new(), obj_vec(vec![1.0, 5.0])));

        // With higher weight on first objective, should prefer first solution
        let selected = front.select_by_weights(&[0.9, 0.1]).unwrap();
        assert!(selected.objectives.values[0] > selected.objectives.values[1]);
    }

    #[test]
    fn test_objective_vector_display() {
        let obj = ObjectiveVector::new(
            vec!["mi".into(), "lat".into()],
            vec![3.14, 0.95],
        );
        let s = format!("{}", obj);
        assert!(s.contains("mi="));
        assert!(s.contains("lat="));
    }

    #[test]
    fn test_objective_vector_get() {
        let obj = ObjectiveVector::new(
            vec!["alpha".into(), "beta".into()],
            vec![1.0, 2.0],
        );
        assert_eq!(obj.get("alpha"), Some(1.0));
        assert_eq!(obj.get("beta"), Some(2.0));
        assert_eq!(obj.get("gamma"), None);
    }

    #[test]
    fn test_pareto_front_capacity() {
        let mut front = ParetoFront::new(3);
        for i in 0..10 {
            let sol = ParetoSolution::new(
                MappingConfig::new(),
                obj_vec(vec![i as f64, (10 - i) as f64]),
            );
            front.add(sol);
        }
        assert!(front.len() <= 3);
    }
}

//! Multi-objective scalarization methods.
//!
//! Each scalarizer converts a multi-objective cost vector into a single
//! scalar value so that standard single-objective solvers can be used.

use crate::CostVector;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Common output type
// ---------------------------------------------------------------------------

/// A scalarized objective: a single numeric value plus metadata describing
/// how it was derived.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarizedObjective {
    pub value: f64,
    pub method: String,
    pub weights: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Weighted Sum
// ---------------------------------------------------------------------------

/// Weighted-sum scalarization: f(x) = Σ wᵢ · cᵢ.
///
/// Only finds supported (convex hull) Pareto-optimal points.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightedSumScalarizer {
    pub weights: Vec<f64>,
}

impl WeightedSumScalarizer {
    pub fn new(weights: Vec<f64>) -> Self {
        Self { weights }
    }

    /// Create with uniform weights (1/d each).
    pub fn uniform(dim: usize) -> Self {
        let w = 1.0 / dim as f64;
        Self {
            weights: vec![w; dim],
        }
    }

    pub fn scalarize(&self, cost: &CostVector) -> ScalarizedObjective {
        let value = cost.weighted_sum(&self.weights);
        ScalarizedObjective {
            value,
            method: "weighted_sum".into(),
            weights: self.weights.clone(),
        }
    }

    /// Generate a set of weight vectors that uniformly sample the weight
    /// simplex with the given number of divisions per axis.
    pub fn simplex_weights(dim: usize, divisions: usize) -> Vec<Vec<f64>> {
        let mut result = Vec::new();
        let mut current = vec![0usize; dim];
        generate_simplex_weights(dim, divisions, 0, divisions, &mut current, &mut result);
        result
    }
}

fn generate_simplex_weights(
    dim: usize,
    divisions: usize,
    depth: usize,
    remaining: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Vec<f64>>,
) {
    if depth == dim - 1 {
        current[depth] = remaining;
        let weights: Vec<f64> = current.iter().map(|&v| v as f64 / divisions as f64).collect();
        result.push(weights);
        return;
    }
    for v in 0..=remaining {
        current[depth] = v;
        generate_simplex_weights(dim, divisions, depth + 1, remaining - v, current, result);
    }
}

// ---------------------------------------------------------------------------
// Epsilon-Constraint
// ---------------------------------------------------------------------------

/// ε-constraint scalarization: minimise one objective subject to upper
/// bounds on all others.
///
/// min c_{primary} s.t. c_i ≤ ε_i for all i ≠ primary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpsilonConstraintScalarizer {
    pub primary: usize,
    pub bounds: Vec<(usize, f64)>,
}

impl EpsilonConstraintScalarizer {
    /// `primary`: index of the objective to minimise.
    /// `bounds`: (index, upper_bound) for every other objective.
    pub fn new(primary: usize, bounds: Vec<(usize, f64)>) -> Self {
        Self { primary, bounds }
    }

    /// Check whether a cost vector satisfies the epsilon constraints.
    pub fn is_feasible(&self, cost: &CostVector) -> bool {
        self.bounds
            .iter()
            .all(|&(idx, bound)| cost.values[idx] <= bound + f64::EPSILON)
    }

    /// Scalarize: returns the primary objective value if feasible,
    /// `f64::INFINITY` otherwise (acts as a penalty).
    pub fn scalarize(&self, cost: &CostVector) -> ScalarizedObjective {
        let value = if self.is_feasible(cost) {
            cost.values[self.primary]
        } else {
            // Penalty proportional to constraint violation
            let violation: f64 = self
                .bounds
                .iter()
                .map(|&(idx, bound)| (cost.values[idx] - bound).max(0.0))
                .sum();
            cost.values[self.primary] + 1e6 * violation
        };
        let mut weights = vec![0.0; cost.dim()];
        weights[self.primary] = 1.0;
        ScalarizedObjective {
            value,
            method: "epsilon_constraint".into(),
            weights,
        }
    }

    /// Generate a grid of ε-constraint problems scanning the non-primary
    /// objectives between their ideal and nadir values.
    pub fn grid(
        primary: usize,
        dim: usize,
        ideal: &CostVector,
        nadir: &CostVector,
        divisions: usize,
    ) -> Vec<EpsilonConstraintScalarizer> {
        let other_indices: Vec<usize> = (0..dim).filter(|&i| i != primary).collect();
        let ranges: Vec<(usize, f64, f64)> = other_indices
            .iter()
            .map(|&i| (i, ideal.values[i], nadir.values[i]))
            .collect();

        let mut bounds_sets: Vec<Vec<(usize, f64)>> = vec![Vec::new()];
        for &(idx, lo, hi) in &ranges {
            let mut new_sets = Vec::new();
            for existing in &bounds_sets {
                for step in 0..=divisions {
                    let frac = step as f64 / divisions as f64;
                    let bound = lo + frac * (hi - lo);
                    let mut new = existing.clone();
                    new.push((idx, bound));
                    new_sets.push(new);
                }
            }
            bounds_sets = new_sets;
        }

        bounds_sets
            .into_iter()
            .map(|bounds| EpsilonConstraintScalarizer::new(primary, bounds))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Chebyshev (Tchebycheff)
// ---------------------------------------------------------------------------

/// Chebyshev scalarization: minimise max_i { wᵢ · |cᵢ − z*_i| }.
///
/// Finds all Pareto-optimal points (including non-supported / non-convex)
/// given an ideal point z*.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChebyshevScalarizer {
    pub weights: Vec<f64>,
    pub ideal: CostVector,
    /// Small augmentation term coefficient to guarantee strict Pareto optimality.
    pub rho: f64,
}

impl ChebyshevScalarizer {
    pub fn new(weights: Vec<f64>, ideal: CostVector) -> Self {
        Self {
            weights,
            ideal,
            rho: 1e-4,
        }
    }

    pub fn with_rho(mut self, rho: f64) -> Self {
        self.rho = rho;
        self
    }

    /// Augmented Chebyshev scalarization:
    ///   max_i { wᵢ · (cᵢ − z*_i) }  +  ρ · Σ wᵢ · (cᵢ − z*_i)
    pub fn scalarize(&self, cost: &CostVector) -> ScalarizedObjective {
        let d = cost.dim();
        let mut max_term = f64::NEG_INFINITY;
        let mut sum_term = 0.0;
        for i in 0..d {
            let diff = cost.values[i] - self.ideal.values[i];
            let weighted = self.weights[i] * diff;
            max_term = max_term.max(weighted);
            sum_term += weighted;
        }
        let value = max_term + self.rho * sum_term;
        ScalarizedObjective {
            value,
            method: "chebyshev".into(),
            weights: self.weights.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Normal-Boundary Intersection (NBI) weight generation
// ---------------------------------------------------------------------------

/// Generate weight vectors using the Normal-Boundary Intersection approach.
///
/// Returns a set of uniformly distributed weight vectors on the simplex
/// that define search directions orthogonal to the convex hull of
/// individual minima (CHIM). This is used with Pascoletti-Serafini
/// or augmented methods.
pub fn nbi_reference_directions(dim: usize, divisions: usize) -> Vec<CostVector> {
    let raw = WeightedSumScalarizer::simplex_weights(dim, divisions);
    raw.into_iter().map(|w| CostVector::new(w)).collect()
}

/// Adaptive weight generation: given an existing frontier, generate new
/// weights targeting sparsely-covered regions.
///
/// Works by identifying the pair of adjacent frontier points (sorted by
/// first objective) with the largest gap and producing a midpoint weight.
pub fn adaptive_weights(
    frontier_costs: &[CostVector],
    dim: usize,
    num_new: usize,
) -> Vec<Vec<f64>> {
    if frontier_costs.is_empty() || dim == 0 {
        return WeightedSumScalarizer::simplex_weights(dim, num_new.max(2));
    }

    // Compute nearest-neighbour distances for each point
    let mut gaps: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..frontier_costs.len() {
        for j in (i + 1)..frontier_costs.len() {
            let dist = frontier_costs[i].euclidean_distance(&frontier_costs[j]);
            gaps.push((i, j, dist));
        }
    }
    gaps.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    let mut new_weights = Vec::new();
    for k in 0..num_new.min(gaps.len()) {
        let (i, j, _) = gaps[k];
        // Midpoint in objective space, normalized to weight simplex
        let mid: Vec<f64> = frontier_costs[i]
            .values
            .iter()
            .zip(frontier_costs[j].values.iter())
            .map(|(a, b)| (a + b) / 2.0)
            .collect();
        let sum: f64 = mid.iter().sum();
        if sum > f64::EPSILON {
            new_weights.push(mid.iter().map(|v| v / sum).collect());
        } else {
            new_weights.push(vec![1.0 / dim as f64; dim]);
        }
    }

    // Pad with simplex weights if needed
    while new_weights.len() < num_new {
        new_weights.push(vec![1.0 / dim as f64; dim]);
    }

    new_weights
}

// ---------------------------------------------------------------------------
// Pascoletti-Serafini
// ---------------------------------------------------------------------------

/// Pascoletti-Serafini scalarization:
///   min t  s.t.  r + t·d ≥ f(x)  (component-wise)
///
/// Where `r` is a reference point and `d` is a direction vector.
/// This can find any Pareto-optimal point regardless of convexity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PascolettiSerafiniScalarizer {
    pub reference: CostVector,
    pub direction: CostVector,
}

impl PascolettiSerafiniScalarizer {
    pub fn new(reference: CostVector, direction: CostVector) -> Self {
        Self {
            reference,
            direction,
        }
    }

    /// Compute the minimum t such that reference + t * direction ≥ cost.
    /// i.e. t ≥ (cost_i − reference_i) / direction_i for each i where direction_i > 0.
    pub fn scalarize(&self, cost: &CostVector) -> ScalarizedObjective {
        let d = cost.dim();
        let mut t = f64::NEG_INFINITY;
        for i in 0..d {
            let di = self.direction.values[i];
            if di.abs() > f64::EPSILON {
                let required = (cost.values[i] - self.reference.values[i]) / di;
                t = t.max(required);
            } else if cost.values[i] > self.reference.values[i] + f64::EPSILON {
                // Direction component is zero but cost exceeds reference → infeasible
                t = f64::INFINITY;
                break;
            }
        }
        ScalarizedObjective {
            value: t,
            method: "pascoletti_serafini".into(),
            weights: self.direction.values.clone(),
        }
    }

    /// Generate a set of PS scalarizers covering the objective space by
    /// using NBI reference directions from an ideal point.
    pub fn from_nbi(
        ideal: &CostVector,
        nadir: &CostVector,
        divisions: usize,
    ) -> Vec<PascolettiSerafiniScalarizer> {
        let dim = ideal.dim();
        let directions = nbi_reference_directions(dim, divisions);
        directions
            .into_iter()
            .map(|dir| {
                // Reference is the ideal point; direction points from ideal toward nadir
                let scaled_dir = CostVector::new(
                    dir.values
                        .iter()
                        .zip(nadir.values.iter().zip(ideal.values.iter()))
                        .map(|(w, (n, id))| w * (n - id).max(f64::EPSILON))
                        .collect(),
                );
                PascolettiSerafiniScalarizer::new(ideal.clone(), scaled_dir)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cv(vals: &[f64]) -> CostVector {
        CostVector::new(vals.to_vec())
    }

    #[test]
    fn test_weighted_sum_basic() {
        let ws = WeightedSumScalarizer::new(vec![0.5, 0.5]);
        let result = ws.scalarize(&cv(&[2.0, 6.0]));
        assert!((result.value - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_sum_uniform() {
        let ws = WeightedSumScalarizer::uniform(3);
        let result = ws.scalarize(&cv(&[3.0, 3.0, 3.0]));
        assert!((result.value - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_simplex_weights_2d() {
        let weights = WeightedSumScalarizer::simplex_weights(2, 4);
        // Should produce (0,1), (0.25,0.75), (0.5,0.5), (0.75,0.25), (1,0)
        assert_eq!(weights.len(), 5);
        for w in &weights {
            let sum: f64 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_simplex_weights_3d() {
        let weights = WeightedSumScalarizer::simplex_weights(3, 3);
        // C(3+3-1, 3-1) = C(5,2) = 10
        assert_eq!(weights.len(), 10);
        for w in &weights {
            let sum: f64 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_epsilon_constraint_feasible() {
        let ec = EpsilonConstraintScalarizer::new(0, vec![(1, 5.0)]);
        let result = ec.scalarize(&cv(&[2.0, 4.0]));
        assert!((result.value - 2.0).abs() < 1e-10);
        assert!(ec.is_feasible(&cv(&[2.0, 4.0])));
    }

    #[test]
    fn test_epsilon_constraint_infeasible() {
        let ec = EpsilonConstraintScalarizer::new(0, vec![(1, 3.0)]);
        assert!(!ec.is_feasible(&cv(&[2.0, 4.0])));
        let result = ec.scalarize(&cv(&[2.0, 4.0]));
        assert!(result.value > 2.0); // penalized
    }

    #[test]
    fn test_epsilon_constraint_grid() {
        let ideal = cv(&[0.0, 0.0, 0.0]);
        let nadir = cv(&[10.0, 10.0, 10.0]);
        let grid = EpsilonConstraintScalarizer::grid(0, 3, &ideal, &nadir, 3);
        // 2 non-primary objectives, 4 steps each → 4 * 4 = 16
        assert_eq!(grid.len(), 16);
    }

    #[test]
    fn test_chebyshev_basic() {
        let cs = ChebyshevScalarizer::new(vec![1.0, 1.0], cv(&[0.0, 0.0])).with_rho(0.0);
        let result = cs.scalarize(&cv(&[3.0, 5.0]));
        assert!((result.value - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_chebyshev_weighted() {
        let cs = ChebyshevScalarizer::new(vec![2.0, 1.0], cv(&[1.0, 1.0])).with_rho(0.0);
        let result = cs.scalarize(&cv(&[3.0, 5.0]));
        // max(2*(3-1), 1*(5-1)) = max(4, 4) = 4
        assert!((result.value - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_chebyshev_augmented() {
        let cs = ChebyshevScalarizer::new(vec![1.0, 1.0], cv(&[0.0, 0.0])).with_rho(0.01);
        let r1 = cs.scalarize(&cv(&[3.0, 5.0]));
        let r2 = cs.scalarize(&cv(&[4.0, 5.0]));
        // Both have max term = 5, but augmentation breaks tie
        assert!(r1.value < r2.value);
    }

    #[test]
    fn test_nbi_directions() {
        let dirs = nbi_reference_directions(3, 2);
        // C(2+3-1, 3-1) = C(4,2) = 6
        assert_eq!(dirs.len(), 6);
    }

    #[test]
    fn test_pascoletti_serafini_basic() {
        let ps = PascolettiSerafiniScalarizer::new(cv(&[0.0, 0.0]), cv(&[1.0, 1.0]));
        let result = ps.scalarize(&cv(&[3.0, 5.0]));
        // t >= max(3/1, 5/1) = 5
        assert!((result.value - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_pascoletti_serafini_anisotropic() {
        let ps = PascolettiSerafiniScalarizer::new(cv(&[1.0, 1.0]), cv(&[2.0, 1.0]));
        let result = ps.scalarize(&cv(&[5.0, 4.0]));
        // t >= max((5-1)/2, (4-1)/1) = max(2, 3) = 3
        assert!((result.value - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_pascoletti_serafini_from_nbi() {
        let ideal = cv(&[0.0, 0.0]);
        let nadir = cv(&[10.0, 10.0]);
        let scalers = PascolettiSerafiniScalarizer::from_nbi(&ideal, &nadir, 4);
        assert_eq!(scalers.len(), 5);
    }

    #[test]
    fn test_adaptive_weights_gap_targeting() {
        let costs = vec![cv(&[0.0, 1.0]), cv(&[1.0, 0.0])];
        let weights = adaptive_weights(&costs, 2, 1);
        assert_eq!(weights.len(), 1);
        let sum: f64 = weights[0].iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}

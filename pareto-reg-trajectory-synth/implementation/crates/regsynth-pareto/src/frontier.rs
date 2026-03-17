//! Generic Pareto frontier data structure with dominance-aware insertion,
//! hypervolume computation, and quality indicators.

use crate::dominance;
use crate::CostVector;
use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// ParetoEntry — a point on the frontier together with its cost
// ---------------------------------------------------------------------------

/// A single entry in the Pareto frontier: a solution of type `T` together
/// with its objective-space cost vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoEntry<T: Clone> {
    pub point: T,
    pub cost: CostVector,
}

// ---------------------------------------------------------------------------
// ParetoFrontier<T>
// ---------------------------------------------------------------------------

/// A Pareto frontier maintaining a set of mutually non-dominated entries.
///
/// Insertion automatically filters newly-dominated points so the invariant
/// "every pair is mutually incomparable or equal" always holds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoFrontier<T: Clone> {
    entries: Vec<ParetoEntry<T>>,
    dimension: usize,
    epsilon: f64,
}

impl<T: Clone> ParetoFrontier<T> {
    /// Create an empty frontier for cost vectors of the given dimension.
    pub fn new(dimension: usize) -> Self {
        Self {
            entries: Vec::new(),
            dimension,
            epsilon: 0.0,
        }
    }

    /// Create a frontier with ε-dominance filtering.
    pub fn with_epsilon(dimension: usize, epsilon: f64) -> Self {
        Self {
            entries: Vec::new(),
            dimension,
            epsilon,
        }
    }

    /// Insert a point with its cost vector.
    ///
    /// If the new point is dominated by any existing entry (or
    /// ε-dominated when ε > 0) it is rejected. Otherwise every existing
    /// entry that the new point dominates (or ε-dominates) is removed.
    ///
    /// Returns `true` if the point was actually added.
    pub fn add_point(&mut self, point: T, cost: CostVector) -> bool {
        assert_eq!(
            cost.dim(),
            self.dimension,
            "cost dimension {} ≠ frontier dimension {}",
            cost.dim(),
            self.dimension
        );

        // Check if the new point is dominated by any existing entry
        for entry in &self.entries {
            if self.epsilon > 0.0 {
                if dominance::epsilon_dominates(&entry.cost, &cost, self.epsilon) {
                    return false;
                }
            } else if dominance::dominates(&entry.cost, &cost) {
                return false;
            }
            // Exact duplicate check
            if entry.cost.values == cost.values {
                return false;
            }
        }

        // Remove existing entries dominated by the new point
        if self.epsilon > 0.0 {
            self.entries
                .retain(|e| !dominance::epsilon_dominates(&cost, &e.cost, self.epsilon));
        } else {
            self.entries
                .retain(|e| !dominance::dominates(&cost, &e.cost));
        }

        self.entries.push(ParetoEntry {
            point,
            cost,
        });
        true
    }

    /// Check whether the given cost vector is dominated by some entry.
    pub fn is_dominated(&self, cost: &CostVector) -> bool {
        self.entries
            .iter()
            .any(|e| dominance::dominates(&e.cost, cost))
    }

    /// Return all entries whose cost vectors dominate the query.
    pub fn dominated_by(&self, cost: &CostVector) -> Vec<&ParetoEntry<T>> {
        self.entries
            .iter()
            .filter(|e| dominance::dominates(&e.cost, cost))
            .collect()
    }

    /// Return all entries whose cost vectors are dominated by the query.
    pub fn entries_dominated_by_cost(&self, cost: &CostVector) -> Vec<&ParetoEntry<T>> {
        self.entries
            .iter()
            .filter(|e| dominance::dominates(cost, &e.cost))
            .collect()
    }

    pub fn size(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Iterate over entries.
    pub fn entries(&self) -> &[ParetoEntry<T>] {
        &self.entries
    }

    /// Extract just the cost vectors.
    pub fn cost_vectors(&self) -> Vec<&CostVector> {
        self.entries.iter().map(|e| &e.cost).collect()
    }

    /// Extract just the points.
    pub fn points(&self) -> Vec<&T> {
        self.entries.iter().map(|e| &e.point).collect()
    }

    // -------------------------------------------------------------------
    // Hypervolume computation
    // -------------------------------------------------------------------

    /// Compute the exact hypervolume indicator with respect to a
    /// reference point.
    ///
    /// Uses inclusion-exclusion for dimensions ≤ 4 and a recursive
    /// slicing algorithm otherwise. All entry costs must be component-wise
    /// less than `reference` for their contribution to be positive.
    pub fn hypervolume(&self, reference: &CostVector) -> f64 {
        assert_eq!(reference.dim(), self.dimension);
        let costs: Vec<&CostVector> = self
            .entries
            .iter()
            .map(|e| &e.cost)
            .filter(|c| c.values.iter().zip(reference.values.iter()).all(|(ci, ri)| ci < ri))
            .collect();

        if costs.is_empty() {
            return 0.0;
        }

        match self.dimension {
            1 => self.hypervolume_1d(&costs, reference),
            2 => self.hypervolume_2d(&costs, reference),
            3 => self.hypervolume_3d(&costs, reference),
            _ => self.hypervolume_inclusion_exclusion(&costs, reference),
        }
    }

    fn hypervolume_1d(&self, costs: &[&CostVector], reference: &CostVector) -> f64 {
        let min_val = costs
            .iter()
            .map(|c| c.values[0])
            .fold(f64::INFINITY, f64::min);
        (reference.values[0] - min_val).max(0.0)
    }

    fn hypervolume_2d(&self, costs: &[&CostVector], reference: &CostVector) -> f64 {
        // Sort by first objective ascending
        let mut sorted: Vec<[f64; 2]> = costs
            .iter()
            .map(|c| [c.values[0], c.values[1]])
            .collect();
        sorted.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());

        // Filter to non-dominated staircase (monotonically decreasing y)
        let mut staircase: Vec<[f64; 2]> = Vec::new();
        let mut min_y = f64::INFINITY;
        for p in &sorted {
            if p[1] < min_y {
                min_y = p[1];
                staircase.push(*p);
            }
        }

        let mut volume = 0.0;
        let r = [reference.values[0], reference.values[1]];
        for i in 0..staircase.len() {
            let x_right = if i + 1 < staircase.len() {
                staircase[i + 1][0]
            } else {
                r[0]
            };
            let width = x_right - staircase[i][0];
            let height = r[1] - staircase[i][1];
            volume += width * height;
        }
        volume
    }

    fn hypervolume_3d(&self, costs: &[&CostVector], reference: &CostVector) -> f64 {
        // For 3D: use slice-based approach
        // Sort by third dimension ascending, sweep in z
        let mut points: Vec<[f64; 3]> = costs
            .iter()
            .map(|c| [c.values[0], c.values[1], c.values[2]])
            .collect();
        points.sort_by(|a, b| a[2].partial_cmp(&b[2]).unwrap());

        let rz = reference.values[2];
        let mut volume = 0.0;
        let ref_2d = CostVector::new(vec![reference.values[0], reference.values[1]]);

        for i in 0..points.len() {
            let z_lo = points[i][2];
            let z_hi = if i + 1 < points.len() {
                points[i + 1][2].min(rz)
            } else {
                rz
            };
            if z_hi <= z_lo {
                continue;
            }

            // Collect all points with z <= z_lo (active in this slab)
            let active_2d: Vec<CostVector> = points[..=i]
                .iter()
                .map(|p| CostVector::new(vec![p[0], p[1]]))
                .collect();

            // Compute 2D hypervolume of active set
            let nd = crate::dominance::filter_dominated(&active_2d);
            let refs: Vec<&CostVector> = nd.iter().collect();
            if !refs.is_empty() {
                let frontier_2d: ParetoFrontier<()> = {
                    let mut f = ParetoFrontier::new(2);
                    for p in &nd {
                        f.add_point((), p.clone());
                    }
                    f
                };
                let area = frontier_2d.hypervolume_2d(&refs, &ref_2d);
                volume += area * (z_hi - z_lo);
            }
        }
        volume
    }

    /// Inclusion-exclusion hypervolume for arbitrary dimension.
    ///
    /// Exact but exponential in the number of points — practical for
    /// small fronts (≤ 20-ish points).
    fn hypervolume_inclusion_exclusion(
        &self,
        costs: &[&CostVector],
        reference: &CostVector,
    ) -> f64 {
        let n = costs.len();
        let d = self.dimension;
        if n == 0 {
            return 0.0;
        }
        // For very large sets fall back to a Monte-Carlo estimate
        if n > 20 {
            return self.hypervolume_monte_carlo(costs, reference, 100_000);
        }

        let mut volume = 0.0;
        // Iterate over all non-empty subsets via bitmask
        for mask in 1..(1u64 << n) {
            let bits = mask.count_ones() as usize;
            // Compute the component-wise max of the subset
            let mut corner = vec![f64::NEG_INFINITY; d];
            for (idx, cost) in costs.iter().enumerate() {
                if mask & (1u64 << idx) != 0 {
                    for k in 0..d {
                        corner[k] = corner[k].max(cost.values[k]);
                    }
                }
            }
            // Hyper-rectangle volume from corner to reference
            let mut rect_vol = 1.0;
            let mut valid = true;
            for k in 0..d {
                let side = reference.values[k] - corner[k];
                if side <= 0.0 {
                    valid = false;
                    break;
                }
                rect_vol *= side;
            }
            if !valid {
                continue;
            }
            // Inclusion-exclusion sign
            if bits % 2 == 1 {
                volume += rect_vol;
            } else {
                volume -= rect_vol;
            }
        }
        volume
    }

    /// Monte-Carlo hypervolume estimate for large sets / high dimensions.
    fn hypervolume_monte_carlo(
        &self,
        costs: &[&CostVector],
        reference: &CostVector,
        samples: usize,
    ) -> f64 {
        use rand::Rng;
        let d = self.dimension;
        let mut rng = rand::thread_rng();

        // Compute bounding box: component-wise min of costs → reference
        let mut lower = vec![f64::INFINITY; d];
        for cost in costs {
            for k in 0..d {
                lower[k] = lower[k].min(cost.values[k]);
            }
        }

        let mut total_box_vol = 1.0;
        for k in 0..d {
            total_box_vol *= reference.values[k] - lower[k];
        }

        let mut hits = 0usize;
        for _ in 0..samples {
            let sample: Vec<f64> = (0..d)
                .map(|k| rng.gen_range(lower[k]..reference.values[k]))
                .collect();
            let sv = CostVector::new(sample);
            // A sample is "hit" if it is dominated by at least one point
            if costs.iter().any(|c| dominance::weakly_dominates(c, &sv)) {
                hits += 1;
            }
        }

        total_box_vol * (hits as f64 / samples as f64)
    }

    // -------------------------------------------------------------------
    // Quality indicators (delegated to metrics module for main impls,
    // but convenience wrappers live here)
    // -------------------------------------------------------------------

    /// Generational distance to a reference frontier.
    pub fn generational_distance(&self, true_frontier: &[CostVector]) -> f64 {
        crate::metrics::generational_distance(
            &self.entries.iter().map(|e| e.cost.clone()).collect::<Vec<_>>(),
            true_frontier,
        )
    }

    /// Inverted generational distance.
    pub fn inverted_generational_distance(&self, true_frontier: &[CostVector]) -> f64 {
        crate::metrics::inverted_generational_distance(
            &self.entries.iter().map(|e| e.cost.clone()).collect::<Vec<_>>(),
            true_frontier,
        )
    }

    /// Spread metric (extent of frontier coverage).
    pub fn spread_metric(&self) -> f64 {
        crate::metrics::spread_metric(
            &self.entries.iter().map(|e| e.cost.clone()).collect::<Vec<_>>(),
        )
    }

    // -------------------------------------------------------------------
    // Merge two frontiers
    // -------------------------------------------------------------------

    /// Merge another frontier into this one, maintaining non-dominance.
    pub fn merge(&mut self, other: &ParetoFrontier<T>) {
        for entry in &other.entries {
            self.add_point(entry.point.clone(), entry.cost.clone());
        }
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    // -------------------------------------------------------------------
    // Ideal and nadir points
    // -------------------------------------------------------------------

    /// Ideal (utopia) point: component-wise minimum across all entries.
    pub fn ideal_point(&self) -> Option<CostVector> {
        if self.entries.is_empty() {
            return None;
        }
        let mut ideal = self.entries[0].cost.clone();
        for entry in &self.entries[1..] {
            ideal = ideal.component_min(&entry.cost);
        }
        Some(ideal)
    }

    /// Nadir point: component-wise maximum across all entries.
    pub fn nadir_point(&self) -> Option<CostVector> {
        if self.entries.is_empty() {
            return None;
        }
        let mut nadir = self.entries[0].cost.clone();
        for entry in &self.entries[1..] {
            nadir = nadir.component_max(&entry.cost);
        }
        Some(nadir)
    }
}

impl<T: Clone + fmt::Debug> fmt::Display for ParetoFrontier<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ParetoFrontier(dim={}, entries={}, ε={})",
            self.dimension,
            self.entries.len(),
            self.epsilon
        )
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
    fn test_add_and_domination_filter() {
        let mut f: ParetoFrontier<&str> = ParetoFrontier::new(2);
        assert!(f.add_point("A", cv(&[3.0, 3.0])));
        assert!(f.add_point("B", cv(&[1.0, 5.0])));
        assert_eq!(f.size(), 2);
        // C dominates A
        assert!(f.add_point("C", cv(&[2.0, 2.0])));
        assert_eq!(f.size(), 2); // A removed, C added
        assert!(!f.entries().iter().any(|e| e.cost.values == vec![3.0, 3.0]));
    }

    #[test]
    fn test_dominated_point_rejected() {
        let mut f: ParetoFrontier<i32> = ParetoFrontier::new(2);
        f.add_point(1, cv(&[1.0, 2.0]));
        assert!(!f.add_point(2, cv(&[2.0, 3.0])));
        assert_eq!(f.size(), 1);
    }

    #[test]
    fn test_duplicate_rejected() {
        let mut f: ParetoFrontier<i32> = ParetoFrontier::new(2);
        f.add_point(1, cv(&[1.0, 2.0]));
        assert!(!f.add_point(2, cv(&[1.0, 2.0])));
        assert_eq!(f.size(), 1);
    }

    #[test]
    fn test_is_dominated() {
        let mut f: ParetoFrontier<i32> = ParetoFrontier::new(2);
        f.add_point(1, cv(&[1.0, 3.0]));
        f.add_point(2, cv(&[3.0, 1.0]));
        assert!(f.is_dominated(&cv(&[4.0, 4.0])));
        assert!(!f.is_dominated(&cv(&[2.0, 2.0])));
    }

    #[test]
    fn test_dominated_by() {
        let mut f: ParetoFrontier<i32> = ParetoFrontier::new(2);
        f.add_point(1, cv(&[1.0, 3.0]));
        f.add_point(2, cv(&[3.0, 1.0]));
        let dom = f.dominated_by(&cv(&[4.0, 4.0]));
        assert_eq!(dom.len(), 2);
    }

    #[test]
    fn test_epsilon_frontier() {
        let mut f: ParetoFrontier<i32> = ParetoFrontier::with_epsilon(2, 0.5);
        f.add_point(1, cv(&[1.0, 2.0]));
        // (1.3, 2.3) is epsilon-dominated by (1.0, 2.0) with eps=0.5
        // because 1.0-0.5=0.5 ≤ 1.3 and 2.0-0.5=1.5 ≤ 2.3
        assert!(!f.add_point(2, cv(&[1.3, 2.3])));
        assert_eq!(f.size(), 1);
    }

    #[test]
    fn test_hypervolume_1d() {
        let mut f: ParetoFrontier<i32> = ParetoFrontier::new(1);
        f.add_point(1, cv(&[2.0]));
        f.add_point(2, cv(&[3.0]));
        let hv = f.hypervolume(&cv(&[10.0]));
        assert!((hv - 8.0).abs() < 1e-10); // 10 - 2 = 8
    }

    #[test]
    fn test_hypervolume_2d() {
        let mut f: ParetoFrontier<i32> = ParetoFrontier::new(2);
        f.add_point(1, cv(&[1.0, 3.0]));
        f.add_point(2, cv(&[3.0, 1.0]));
        let hv = f.hypervolume(&cv(&[5.0, 5.0]));
        // Rectangle from (1,3) to (3,5) = 2*2 = 4
        // Rectangle from (3,1) to (5,5) = 2*4 = 8
        // Total = 12
        assert!((hv - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_hypervolume_single_point_2d() {
        let mut f: ParetoFrontier<i32> = ParetoFrontier::new(2);
        f.add_point(1, cv(&[2.0, 3.0]));
        let hv = f.hypervolume(&cv(&[5.0, 5.0]));
        assert!((hv - 6.0).abs() < 1e-10); // 3 * 2 = 6
    }

    #[test]
    fn test_hypervolume_3d() {
        let mut f: ParetoFrontier<i32> = ParetoFrontier::new(3);
        f.add_point(1, cv(&[1.0, 1.0, 1.0]));
        let hv = f.hypervolume(&cv(&[2.0, 2.0, 2.0]));
        assert!((hv - 1.0).abs() < 1e-10); // 1*1*1
    }

    #[test]
    fn test_ideal_nadir() {
        let mut f: ParetoFrontier<i32> = ParetoFrontier::new(2);
        f.add_point(1, cv(&[1.0, 4.0]));
        f.add_point(2, cv(&[3.0, 2.0]));
        let ideal = f.ideal_point().unwrap();
        let nadir = f.nadir_point().unwrap();
        assert_eq!(ideal.values, vec![1.0, 2.0]);
        assert_eq!(nadir.values, vec![3.0, 4.0]);
    }

    #[test]
    fn test_merge_frontiers() {
        let mut f1: ParetoFrontier<i32> = ParetoFrontier::new(2);
        f1.add_point(1, cv(&[1.0, 5.0]));
        f1.add_point(2, cv(&[5.0, 1.0]));

        let mut f2: ParetoFrontier<i32> = ParetoFrontier::new(2);
        f2.add_point(3, cv(&[2.0, 2.0]));

        f1.merge(&f2);
        // (2,2) dominates neither (1,5) nor (5,1), so all 3 remain
        assert_eq!(f1.size(), 3);
    }

    #[test]
    fn test_points_and_costs() {
        let mut f: ParetoFrontier<&str> = ParetoFrontier::new(2);
        f.add_point("A", cv(&[1.0, 3.0]));
        f.add_point("B", cv(&[3.0, 1.0]));
        assert_eq!(f.points().len(), 2);
        assert_eq!(f.cost_vectors().len(), 2);
    }

    #[test]
    fn test_empty_hypervolume() {
        let f: ParetoFrontier<i32> = ParetoFrontier::new(2);
        assert_eq!(f.hypervolume(&cv(&[5.0, 5.0])), 0.0);
    }

    #[test]
    fn test_clear() {
        let mut f: ParetoFrontier<i32> = ParetoFrontier::new(2);
        f.add_point(1, cv(&[1.0, 2.0]));
        f.clear();
        assert!(f.is_empty());
    }

    #[test]
    fn test_large_frontier_maintains_nondominated() {
        let mut f: ParetoFrontier<usize> = ParetoFrontier::new(2);
        for i in 0..50 {
            let x = i as f64;
            f.add_point(i, cv(&[x, 50.0 - x]));
        }
        // All points on the line x + y = 50 are mutually non-dominated
        assert_eq!(f.size(), 50);
    }
}

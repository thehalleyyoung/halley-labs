//! Dominance checking operations for multi-objective optimization.
//!
//! Provides dominance comparisons on [`CostVector`]s, epsilon-dominance,
//! non-dominated filtering, and NSGA-II–style fast non-dominated sorting.

use crate::CostVector;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// ParetoOrdering
// ---------------------------------------------------------------------------

/// Outcome of comparing two cost vectors in Pareto sense.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ParetoOrdering {
    /// `a` strictly dominates `b`: a[i] ≤ b[i] ∀i, a[j] < b[j] ∃j.
    Dominates,
    /// `b` strictly dominates `a`.
    Dominated,
    /// Neither dominates the other and they are not equal.
    Incomparable,
    /// Component-wise equal.
    Equal,
}

// ---------------------------------------------------------------------------
// Core dominance predicates
// ---------------------------------------------------------------------------

/// Returns `true` iff `a` Pareto-dominates `b` (all objectives minimised).
///
/// `a` dominates `b` ⟺ ∀i: a\[i\] ≤ b\[i\] ∧ ∃j: a\[j\] < b\[j\].
pub fn dominates(a: &CostVector, b: &CostVector) -> bool {
    assert_eq!(a.dim(), b.dim(), "dimension mismatch in dominates()");
    let mut strictly_better = false;
    for (ai, bi) in a.values.iter().zip(b.values.iter()) {
        if ai > bi {
            return false;
        }
        if ai < bi {
            strictly_better = true;
        }
    }
    strictly_better
}

/// Returns `true` iff `a` weakly dominates `b`: ∀i: a\[i\] ≤ b\[i\].
pub fn weakly_dominates(a: &CostVector, b: &CostVector) -> bool {
    assert_eq!(a.dim(), b.dim());
    a.values.iter().zip(b.values.iter()).all(|(ai, bi)| ai <= bi)
}

/// Epsilon-dominance: `a` ε-dominates `b` iff ∀i: a\[i\] − ε ≤ b\[i\]
/// **and** ∃j: a\[j\] − ε < b\[j\].
///
/// Equivalently a shifted copy of `a` dominates `b`.
pub fn epsilon_dominates(a: &CostVector, b: &CostVector, epsilon: f64) -> bool {
    assert_eq!(a.dim(), b.dim());
    let mut strictly_better = false;
    for (ai, bi) in a.values.iter().zip(b.values.iter()) {
        if ai - epsilon > *bi {
            return false;
        }
        if ai - epsilon < *bi {
            strictly_better = true;
        }
    }
    strictly_better
}

/// Full Pareto comparison yielding a [`ParetoOrdering`].
pub fn pareto_compare(a: &CostVector, b: &CostVector) -> ParetoOrdering {
    assert_eq!(a.dim(), b.dim());
    let mut a_better = false;
    let mut b_better = false;
    for (ai, bi) in a.values.iter().zip(b.values.iter()) {
        if ai < bi {
            a_better = true;
        } else if ai > bi {
            b_better = true;
        }
        if a_better && b_better {
            return ParetoOrdering::Incomparable;
        }
    }
    match (a_better, b_better) {
        (true, false) => ParetoOrdering::Dominates,
        (false, true) => ParetoOrdering::Dominated,
        (false, false) => ParetoOrdering::Equal,
        _ => ParetoOrdering::Incomparable, // unreachable due to early exit
    }
}

// ---------------------------------------------------------------------------
// Filtering helpers
// ---------------------------------------------------------------------------

/// Returns the non-dominated subset of `points`.
///
/// A point survives iff no other point in the set dominates it.
/// Runs in O(n² · d) time.
pub fn filter_dominated(points: &[CostVector]) -> Vec<CostVector> {
    let n = points.len();
    let mut dominated = vec![false; n];
    for i in 0..n {
        if dominated[i] {
            continue;
        }
        for j in (i + 1)..n {
            if dominated[j] {
                continue;
            }
            match pareto_compare(&points[i], &points[j]) {
                ParetoOrdering::Dominates => dominated[j] = true,
                ParetoOrdering::Dominated => {
                    dominated[i] = true;
                    break;
                }
                _ => {}
            }
        }
    }
    points
        .iter()
        .enumerate()
        .filter(|(i, _)| !dominated[*i])
        .map(|(_, p)| p.clone())
        .collect()
}

/// Returns the non-dominated subset together with original indices.
pub fn filter_dominated_indexed(points: &[CostVector]) -> Vec<(usize, CostVector)> {
    let n = points.len();
    let mut dominated = vec![false; n];
    for i in 0..n {
        if dominated[i] {
            continue;
        }
        for j in (i + 1)..n {
            if dominated[j] {
                continue;
            }
            match pareto_compare(&points[i], &points[j]) {
                ParetoOrdering::Dominates => dominated[j] = true,
                ParetoOrdering::Dominated => {
                    dominated[i] = true;
                    break;
                }
                _ => {}
            }
        }
    }
    points
        .iter()
        .enumerate()
        .filter(|(i, _)| !dominated[*i])
        .map(|(i, p)| (i, p.clone()))
        .collect()
}

// ---------------------------------------------------------------------------
// Dominance cone
// ---------------------------------------------------------------------------

/// Compute the dominance cone of `point`: the half-space of all points
/// dominated by `point`.  Represented by its vertex (the point itself)
/// and the direction vectors (positive coordinate axes for minimisation).
///
/// Returns `(vertex, axis_directions)`.
pub fn dominance_cone(point: &CostVector) -> (CostVector, Vec<CostVector>) {
    let d = point.dim();
    let mut axes = Vec::with_capacity(d);
    for i in 0..d {
        let mut dir = vec![0.0; d];
        dir[i] = 1.0; // positive axis → worse cost
        axes.push(CostVector::new(dir));
    }
    (point.clone(), axes)
}

/// Tests whether `query` lies inside the dominance cone of `vertex`,
/// i.e. `vertex` dominates `query` (or weakly dominates).
pub fn in_dominance_cone(vertex: &CostVector, query: &CostVector) -> bool {
    weakly_dominates(vertex, query)
}

// ---------------------------------------------------------------------------
// Fast non-dominated sorting (NSGA-II)
// ---------------------------------------------------------------------------

/// NSGA-II fast non-dominated sort.
///
/// Partitions `points` into successive Pareto fronts.
/// Returns a `Vec` of fronts; each front is a `Vec` of indices into the
/// original slice.
///
/// Front 0 = non-dominated set, front 1 = non-dominated after removing
/// front 0, etc.
pub fn fast_non_dominated_sort(points: &[CostVector]) -> Vec<Vec<usize>> {
    let n = points.len();
    if n == 0 {
        return Vec::new();
    }

    // S[i] = set of solutions dominated by i
    let mut dominated_by: Vec<Vec<usize>> = vec![Vec::new(); n];
    // n[i] = number of solutions dominating i
    let mut domination_count: Vec<usize> = vec![0; n];

    for i in 0..n {
        for j in (i + 1)..n {
            match pareto_compare(&points[i], &points[j]) {
                ParetoOrdering::Dominates => {
                    dominated_by[i].push(j);
                    domination_count[j] += 1;
                }
                ParetoOrdering::Dominated => {
                    dominated_by[j].push(i);
                    domination_count[i] += 1;
                }
                _ => {}
            }
        }
    }

    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut current_front: Vec<usize> = (0..n)
        .filter(|&i| domination_count[i] == 0)
        .collect();

    while !current_front.is_empty() {
        let mut next_front = Vec::new();
        for &i in &current_front {
            for &j in &dominated_by[i] {
                domination_count[j] -= 1;
                if domination_count[j] == 0 {
                    next_front.push(j);
                }
            }
        }
        fronts.push(current_front);
        current_front = next_front;
    }

    fronts
}

/// Crowding distance assignment for a single front (NSGA-II).
///
/// Returns a vector of crowding distances, one per element in `front_indices`.
/// Points on the boundary of each objective receive `f64::INFINITY`.
pub fn crowding_distance(points: &[CostVector], front_indices: &[usize]) -> Vec<f64> {
    let n = front_indices.len();
    if n <= 2 {
        return vec![f64::INFINITY; n];
    }

    let d = points[front_indices[0]].dim();
    let mut distances = vec![0.0_f64; n];

    for obj in 0..d {
        // Sort front members by this objective
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            let va = points[front_indices[a]].values[obj];
            let vb = points[front_indices[b]].values[obj];
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let f_min = points[front_indices[order[0]]].values[obj];
        let f_max = points[front_indices[order[n - 1]]].values[obj];
        let range = f_max - f_min;

        distances[order[0]] = f64::INFINITY;
        distances[order[n - 1]] = f64::INFINITY;

        if range > f64::EPSILON {
            for k in 1..(n - 1) {
                let prev_val = points[front_indices[order[k - 1]]].values[obj];
                let next_val = points[front_indices[order[k + 1]]].values[obj];
                distances[order[k]] += (next_val - prev_val) / range;
            }
        }
    }

    distances
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
    fn test_dominates_basic() {
        let a = cv(&[1.0, 2.0]);
        let b = cv(&[2.0, 3.0]);
        assert!(dominates(&a, &b));
        assert!(!dominates(&b, &a));
    }

    #[test]
    fn test_dominates_equal() {
        let a = cv(&[1.0, 2.0]);
        assert!(!dominates(&a, &a));
    }

    #[test]
    fn test_dominates_incomparable() {
        let a = cv(&[1.0, 3.0]);
        let b = cv(&[2.0, 2.0]);
        assert!(!dominates(&a, &b));
        assert!(!dominates(&b, &a));
    }

    #[test]
    fn test_epsilon_dominates() {
        let a = cv(&[1.0, 2.0]);
        let b = cv(&[1.5, 2.5]);
        // With eps=0, epsilon_dominates ≡ dominates → true (a dom b)
        assert!(epsilon_dominates(&a, &b, 0.0));
        assert!(dominates(&a, &b));
        // With epsilon = 1.0, a_shifted = [0.0, 1.0] which dominates [1.5, 2.5]
        assert!(epsilon_dominates(&a, &b, 1.0));
        // Near-equal points: c does NOT epsilon-dominate d with small eps
        let c = cv(&[1.0, 2.0]);
        let d = cv(&[1.0, 2.0]);
        assert!(!epsilon_dominates(&c, &d, 0.0)); // equal → not strictly
    }

    #[test]
    fn test_pareto_compare_all_orderings() {
        let a = cv(&[1.0, 2.0]);
        let b = cv(&[2.0, 3.0]);
        let c = cv(&[1.0, 3.0]);
        let d = cv(&[1.0, 2.0]);
        assert_eq!(pareto_compare(&a, &b), ParetoOrdering::Dominates);
        assert_eq!(pareto_compare(&b, &a), ParetoOrdering::Dominated);
        assert_eq!(pareto_compare(&a, &c), ParetoOrdering::Dominates);
        assert_eq!(pareto_compare(&a, &d), ParetoOrdering::Equal);
        // incomparable
        let e = cv(&[1.0, 4.0]);
        let f = cv(&[3.0, 1.0]);
        assert_eq!(pareto_compare(&e, &f), ParetoOrdering::Incomparable);
    }

    #[test]
    fn test_filter_dominated() {
        let points = vec![
            cv(&[1.0, 5.0]),
            cv(&[2.0, 3.0]),
            cv(&[3.0, 2.0]),
            cv(&[4.0, 4.0]), // dominated by (2,3) or (3,2)
            cv(&[1.5, 4.5]), // dominated by (1,5)? No, 1.5>1 but 4.5<5 → incomparable
        ];
        let nd = filter_dominated(&points);
        // Non-dominated: (1,5), (2,3), (3,2), (1.5,4.5)
        // (4,4) dominated by (2,3): 2<=4, 3<=4 with strict? 2<4 yes → dominated
        assert_eq!(nd.len(), 4);
        assert!(!nd.iter().any(|p| p.values == vec![4.0, 4.0]));
    }

    #[test]
    fn test_fast_non_dominated_sort() {
        let points = vec![
            cv(&[1.0, 4.0]), // front 0
            cv(&[2.0, 2.0]), // front 0
            cv(&[4.0, 1.0]), // front 0
            cv(&[3.0, 3.0]), // front 1 (dominated by (2,2))
            cv(&[5.0, 5.0]), // front 2
        ];
        let fronts = fast_non_dominated_sort(&points);
        assert!(fronts.len() >= 2);
        assert_eq!(fronts[0].len(), 3); // first front has 3 points
        assert!(fronts[0].contains(&0));
        assert!(fronts[0].contains(&1));
        assert!(fronts[0].contains(&2));
        assert!(fronts[1].contains(&3));
    }

    #[test]
    fn test_fast_non_dominated_sort_empty() {
        let fronts = fast_non_dominated_sort(&[]);
        assert!(fronts.is_empty());
    }

    #[test]
    fn test_dominance_cone() {
        let p = cv(&[2.0, 3.0, 1.0]);
        let (vertex, axes) = dominance_cone(&p);
        assert_eq!(vertex.values, p.values);
        assert_eq!(axes.len(), 3);
        assert_eq!(axes[0].values, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_in_dominance_cone() {
        let vertex = cv(&[1.0, 2.0]);
        assert!(in_dominance_cone(&vertex, &cv(&[2.0, 3.0])));
        assert!(in_dominance_cone(&vertex, &cv(&[1.0, 2.0])));
        assert!(!in_dominance_cone(&vertex, &cv(&[0.5, 3.0])));
    }

    #[test]
    fn test_crowding_distance_boundary() {
        let points = vec![
            cv(&[1.0, 4.0]),
            cv(&[2.0, 2.0]),
            cv(&[4.0, 1.0]),
        ];
        let cd = crowding_distance(&points, &[0, 1, 2]);
        assert!(cd[0].is_infinite());
        assert!(cd[2].is_infinite());
        assert!(cd[1].is_finite());
    }

    #[test]
    fn test_weakly_dominates() {
        let a = cv(&[1.0, 2.0]);
        let b = cv(&[1.0, 2.0]);
        assert!(weakly_dominates(&a, &b));
        assert!(!dominates(&a, &b));
    }
}

//! Pareto quality metrics: hypervolume indicator, generational distance,
//! inverted generational distance, spread, spacing, and epsilon-indicator.

use crate::CostVector;

// ---------------------------------------------------------------------------
// Generational Distance (GD)
// ---------------------------------------------------------------------------

/// Generational distance: average minimum distance from each point in the
/// computed set to the nearest point in the true Pareto front.
///
/// GD = (1/|C|) * √(Σ dᵢ²)  where dᵢ = min over true front of Euclidean dist.
pub fn generational_distance(computed: &[CostVector], true_front: &[CostVector]) -> f64 {
    if computed.is_empty() || true_front.is_empty() {
        return f64::INFINITY;
    }
    let sum_sq: f64 = computed
        .iter()
        .map(|c| {
            true_front
                .iter()
                .map(|t| c.euclidean_distance(t))
                .fold(f64::INFINITY, f64::min)
                .powi(2)
        })
        .sum();
    (sum_sq / computed.len() as f64).sqrt()
}

// ---------------------------------------------------------------------------
// Inverted Generational Distance (IGD)
// ---------------------------------------------------------------------------

/// Inverted generational distance: average minimum distance from each point
/// in the true front to the nearest point in the computed set.
///
/// IGD simultaneously measures convergence and diversity.
pub fn inverted_generational_distance(computed: &[CostVector], true_front: &[CostVector]) -> f64 {
    if computed.is_empty() || true_front.is_empty() {
        return f64::INFINITY;
    }
    let sum_sq: f64 = true_front
        .iter()
        .map(|t| {
            computed
                .iter()
                .map(|c| t.euclidean_distance(c))
                .fold(f64::INFINITY, f64::min)
                .powi(2)
        })
        .sum();
    (sum_sq / true_front.len() as f64).sqrt()
}

// ---------------------------------------------------------------------------
// Spread Metric (Δ)
// ---------------------------------------------------------------------------

/// Spread metric measuring the extent of the frontier.
///
/// Computes the standard deviation of consecutive distances between
/// sorted frontier points normalized by the mean distance.
/// Smaller spread → more uniform distribution.
pub fn spread_metric(points: &[CostVector]) -> f64 {
    if points.len() < 2 {
        return 0.0;
    }

    // Compute pairwise nearest-neighbour distances
    let mut nn_distances: Vec<f64> = Vec::with_capacity(points.len());
    for (i, p) in points.iter().enumerate() {
        let min_dist = points
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, q)| p.euclidean_distance(q))
            .fold(f64::INFINITY, f64::min);
        nn_distances.push(min_dist);
    }

    let mean = nn_distances.iter().sum::<f64>() / nn_distances.len() as f64;
    if mean < f64::EPSILON {
        return 0.0;
    }

    let variance = nn_distances
        .iter()
        .map(|d| (d - mean).powi(2))
        .sum::<f64>()
        / nn_distances.len() as f64;

    variance.sqrt() / mean
}

// ---------------------------------------------------------------------------
// Spacing Metric
// ---------------------------------------------------------------------------

/// Spacing metric: standard deviation of nearest-neighbour distances.
///
/// A value of 0 means perfectly uniform spacing.
pub fn spacing_metric(points: &[CostVector]) -> f64 {
    if points.len() < 2 {
        return 0.0;
    }

    let mut nn_distances: Vec<f64> = Vec::with_capacity(points.len());
    for (i, p) in points.iter().enumerate() {
        let min_dist = points
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, q)| {
                // Spacing traditionally uses L1 (Manhattan) distance
                p.values
                    .iter()
                    .zip(q.values.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f64>()
            })
            .fold(f64::INFINITY, f64::min);
        nn_distances.push(min_dist);
    }

    let mean = nn_distances.iter().sum::<f64>() / nn_distances.len() as f64;
    let variance = nn_distances
        .iter()
        .map(|d| (d - mean).powi(2))
        .sum::<f64>()
        / nn_distances.len() as f64;

    variance.sqrt()
}

// ---------------------------------------------------------------------------
// Epsilon Indicator (Iε+)
// ---------------------------------------------------------------------------

/// Additive epsilon indicator: the minimum ε such that every point in
/// `frontier_b` is ε-dominated by some point in `frontier_a`.
///
/// I_ε+(A, B) = max over b∈B [ min over a∈A [ max over k (a_k − b_k) ] ]
///
/// If A completely dominates B, the result is ≤ 0.
pub fn epsilon_indicator(frontier_a: &[CostVector], frontier_b: &[CostVector]) -> f64 {
    if frontier_a.is_empty() || frontier_b.is_empty() {
        return f64::INFINITY;
    }
    frontier_b
        .iter()
        .map(|b| {
            frontier_a
                .iter()
                .map(|a| {
                    a.values
                        .iter()
                        .zip(b.values.iter())
                        .map(|(ai, bi)| ai - bi)
                        .fold(f64::NEG_INFINITY, f64::max)
                })
                .fold(f64::INFINITY, f64::min)
        })
        .fold(f64::NEG_INFINITY, f64::max)
}

// ---------------------------------------------------------------------------
// Hypervolume indicator (standalone, for use without ParetoFrontier)
// ---------------------------------------------------------------------------

/// Standalone hypervolume indicator for a set of cost vectors.
///
/// Uses inclusion-exclusion for sets up to 20 points, Monte-Carlo otherwise.
pub fn hypervolume_indicator(points: &[CostVector], reference: &CostVector) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    let d = reference.dim();

    // Filter to points strictly less than reference in all dimensions
    let valid: Vec<&CostVector> = points
        .iter()
        .filter(|p| p.values.iter().zip(reference.values.iter()).all(|(pi, ri)| pi < ri))
        .collect();

    if valid.is_empty() {
        return 0.0;
    }

    if valid.len() <= 20 {
        // Inclusion-exclusion
        let mut volume = 0.0;
        for mask in 1..(1u64 << valid.len()) {
            let bits = mask.count_ones() as usize;
            let mut corner = vec![f64::NEG_INFINITY; d];
            for (idx, cost) in valid.iter().enumerate() {
                if mask & (1u64 << idx) != 0 {
                    for k in 0..d {
                        corner[k] = corner[k].max(cost.values[k]);
                    }
                }
            }
            let mut rect_vol = 1.0;
            let mut ok = true;
            for k in 0..d {
                let side = reference.values[k] - corner[k];
                if side <= 0.0 {
                    ok = false;
                    break;
                }
                rect_vol *= side;
            }
            if !ok {
                continue;
            }
            if bits % 2 == 1 {
                volume += rect_vol;
            } else {
                volume -= rect_vol;
            }
        }
        volume
    } else {
        // Monte-Carlo
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let samples = 200_000;

        let mut lower = vec![f64::INFINITY; d];
        for cost in &valid {
            for k in 0..d {
                lower[k] = lower[k].min(cost.values[k]);
            }
        }

        let mut total_vol = 1.0;
        for k in 0..d {
            total_vol *= reference.values[k] - lower[k];
        }

        let mut hits = 0usize;
        for _ in 0..samples {
            let sample: Vec<f64> = (0..d)
                .map(|k| rng.gen_range(lower[k]..reference.values[k]))
                .collect();
            let sv = CostVector::new(sample);
            if valid
                .iter()
                .any(|c| crate::dominance::weakly_dominates(c, &sv))
            {
                hits += 1;
            }
        }

        total_vol * (hits as f64 / samples as f64)
    }
}

// ---------------------------------------------------------------------------
// ParetoMetrics — convenience struct bundling all indicators
// ---------------------------------------------------------------------------

/// Aggregated quality metrics for a Pareto frontier.
#[derive(Debug, Clone)]
pub struct ParetoMetrics {
    pub hypervolume: f64,
    pub generational_distance: f64,
    pub inverted_generational_distance: f64,
    pub spread: f64,
    pub spacing: f64,
    pub epsilon_indicator: f64,
    pub num_points: usize,
}

impl ParetoMetrics {
    /// Compute all metrics for `computed` against `true_front` using
    /// the given `reference` point for hypervolume.
    pub fn compute(
        computed: &[CostVector],
        true_front: &[CostVector],
        reference: &CostVector,
    ) -> Self {
        Self {
            hypervolume: hypervolume_indicator(computed, reference),
            generational_distance: generational_distance(computed, true_front),
            inverted_generational_distance: inverted_generational_distance(computed, true_front),
            spread: spread_metric(computed),
            spacing: spacing_metric(computed),
            epsilon_indicator: epsilon_indicator(computed, true_front),
            num_points: computed.len(),
        }
    }

    /// Compute metrics without a true reference front (only self-metrics).
    pub fn compute_self(computed: &[CostVector], reference: &CostVector) -> Self {
        Self {
            hypervolume: hypervolume_indicator(computed, reference),
            generational_distance: 0.0,
            inverted_generational_distance: 0.0,
            spread: spread_metric(computed),
            spacing: spacing_metric(computed),
            epsilon_indicator: 0.0,
            num_points: computed.len(),
        }
    }
}

impl std::fmt::Display for ParetoMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ParetoMetrics {{ HV={:.4}, GD={:.4}, IGD={:.4}, Spread={:.4}, Spacing={:.4}, ε={:.4}, n={} }}",
            self.hypervolume,
            self.generational_distance,
            self.inverted_generational_distance,
            self.spread,
            self.spacing,
            self.epsilon_indicator,
            self.num_points
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
    fn test_generational_distance_perfect() {
        let front = vec![cv(&[1.0, 3.0]), cv(&[2.0, 2.0]), cv(&[3.0, 1.0])];
        let gd = generational_distance(&front, &front);
        assert!(gd < 1e-10);
    }

    #[test]
    fn test_generational_distance_offset() {
        let computed = vec![cv(&[2.0, 2.0])];
        let true_front = vec![cv(&[1.0, 1.0])];
        let gd = generational_distance(&computed, &true_front);
        let expected = (2.0_f64).sqrt(); // sqrt((1)^2 + (1)^2)
        assert!((gd - expected).abs() < 1e-10);
    }

    #[test]
    fn test_igd_perfect() {
        let front = vec![cv(&[1.0, 3.0]), cv(&[2.0, 2.0]), cv(&[3.0, 1.0])];
        let igd = inverted_generational_distance(&front, &front);
        assert!(igd < 1e-10);
    }

    #[test]
    fn test_spread_uniform() {
        // Uniformly spaced points should have low spread
        let points: Vec<CostVector> = (0..10)
            .map(|i| cv(&[i as f64, 10.0 - i as f64]))
            .collect();
        let s = spread_metric(&points);
        assert!(s < 0.1); // Nearly zero variation
    }

    #[test]
    fn test_spacing_uniform() {
        let points: Vec<CostVector> = (0..5)
            .map(|i| cv(&[i as f64 * 2.0, 10.0 - i as f64 * 2.0]))
            .collect();
        let s = spacing_metric(&points);
        assert!(s < 1e-10); // All nn-distances are equal
    }

    #[test]
    fn test_epsilon_indicator_dominating() {
        let a = vec![cv(&[1.0, 1.0])];
        let b = vec![cv(&[2.0, 2.0])];
        let eps = epsilon_indicator(&a, &b);
        // max_k(a_k - b_k) = max(1-2, 1-2) = -1
        assert!((eps - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_epsilon_indicator_same() {
        let a = vec![cv(&[1.0, 2.0])];
        let b = vec![cv(&[1.0, 2.0])];
        let eps = epsilon_indicator(&a, &b);
        assert!(eps.abs() < 1e-10);
    }

    #[test]
    fn test_epsilon_indicator_incomparable() {
        let a = vec![cv(&[1.0, 5.0])];
        let b = vec![cv(&[3.0, 2.0])];
        // min over a { max_k(a_k - b_k) } = max(1-3, 5-2) = max(-2, 3) = 3
        let eps = epsilon_indicator(&a, &b);
        assert!((eps - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_hypervolume_indicator_2d() {
        let points = vec![cv(&[1.0, 3.0]), cv(&[3.0, 1.0])];
        let ref_pt = cv(&[5.0, 5.0]);
        let hv = hypervolume_indicator(&points, &ref_pt);
        assert!((hv - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_hypervolume_indicator_empty() {
        let hv = hypervolume_indicator(&[], &cv(&[5.0, 5.0]));
        assert_eq!(hv, 0.0);
    }

    #[test]
    fn test_pareto_metrics_struct() {
        let computed = vec![cv(&[1.0, 3.0]), cv(&[2.0, 2.0]), cv(&[3.0, 1.0])];
        let true_front = computed.clone();
        let reference = cv(&[5.0, 5.0]);
        let m = ParetoMetrics::compute(&computed, &true_front, &reference);
        assert!(m.generational_distance < 1e-10);
        assert!(m.inverted_generational_distance < 1e-10);
        assert!(m.hypervolume > 0.0);
        assert_eq!(m.num_points, 3);
    }

    #[test]
    fn test_spread_single_point() {
        assert_eq!(spread_metric(&[cv(&[1.0, 2.0])]), 0.0);
    }

    #[test]
    fn test_spacing_two_points() {
        let s = spacing_metric(&[cv(&[1.0, 3.0]), cv(&[3.0, 1.0])]);
        assert!(s < 1e-10); // Only two points, both nn-distance equal
    }

    #[test]
    fn test_hypervolume_4d() {
        let points = vec![cv(&[1.0, 1.0, 1.0, 1.0])];
        let reference = cv(&[2.0, 2.0, 2.0, 2.0]);
        let hv = hypervolume_indicator(&points, &reference);
        assert!((hv - 1.0).abs() < 1e-10); // 1^4
    }
}

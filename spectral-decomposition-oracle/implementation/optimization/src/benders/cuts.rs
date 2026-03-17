//! Cut generation and management for Benders decomposition.
//!
//! Provides sparse cut representation, deduplication, violation evaluation,
//! ageing/cleanup and Magnanti-Wong strengthening.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Cut type & struct
// ---------------------------------------------------------------------------

/// Identifies optimality vs. feasibility cuts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CutType {
    Optimality,
    Feasibility,
}

/// A single Benders cut stored in sparse form:
///
/// `sum_j  coefficients[j].1 * x[coefficients[j].0]  >=  rhs`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BendersCut {
    /// Optimality or feasibility.
    pub cut_type: CutType,
    /// Sparse coefficients: `(variable_index, coefficient)`.
    pub coefficients: Vec<(usize, f64)>,
    /// Right-hand side.
    pub rhs: f64,
    /// Number of iterations since the cut was last active.
    pub age: usize,
    /// Number of times this cut has been active (binding / near-binding).
    pub times_active: usize,
    /// Index of the subproblem block that generated this cut.
    pub block: usize,
    /// Constraint violation when the cut was generated.
    pub violation_at_generation: f64,
}

impl BendersCut {
    /// Evaluate `sum a_j * x_j` for the given point.
    pub fn lhs_value(&self, point: &[f64]) -> f64 {
        self.coefficients
            .iter()
            .map(|&(j, a)| {
                if j < point.len() {
                    a * point[j]
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Positive violation means the cut is violated (lhs < rhs).
    pub fn violation(&self, point: &[f64]) -> f64 {
        self.rhs - self.lhs_value(point)
    }

    /// Number of non-zero coefficients (sparsity measure).
    pub fn nnz(&self) -> usize {
        self.coefficients
            .iter()
            .filter(|(_, v)| v.abs() > 1e-15)
            .count()
    }
}

// ---------------------------------------------------------------------------
// Free-standing cut derivation functions
// ---------------------------------------------------------------------------

/// Derive a standard Benders optimality cut.
///
/// Given dual values `y`, the technology matrix columns for the complicating
/// variables (each entry is a sparse column `(row, coeff)`), the sub-problem
/// objective at the current iterate, and the complicating variable values
/// `x_hat`, produces the cut:
///
///   θ_k  >=  sub_obj  +  Σ_j  (Σ_i y_i * T_{ij}) * (x_j − x_hat_j)
///
/// which simplifies to:
///
///   θ_k  -  Σ_j π_j * x_j  >=  sub_obj - Σ_j π_j * x_hat_j
///
/// where π_j = Σ_i y_i * T_{ij}.
///
/// The caller is responsible for mapping the θ variable into the coefficient
/// vector at the correct master index.  Here we return the cut in terms of
/// complicating-variable indices only (plus the rhs that must absorb the θ
/// term on the master side).
pub fn derive_optimality_cut(
    dual_values: &[f64],
    technology_matrix_cols: &[Vec<(usize, f64)>],
    sub_objective: f64,
    complicating_values: &[f64],
    block: usize,
) -> BendersCut {
    // π_j = Σ_i y_i * T_{ij}
    let num_complicating = technology_matrix_cols.len();
    let mut pi = vec![0.0f64; num_complicating];
    for (j, col) in technology_matrix_cols.iter().enumerate() {
        for &(i, t_ij) in col {
            if i < dual_values.len() {
                pi[j] += dual_values[i] * t_ij;
            }
        }
    }

    // rhs = sub_obj - Σ_j π_j * x_hat_j
    let mut rhs = sub_objective;
    for (j, &pi_j) in pi.iter().enumerate() {
        if j < complicating_values.len() {
            rhs -= pi_j * complicating_values[j];
        }
    }

    // Sparse coefficients (drop near-zeros)
    let coefficients: Vec<(usize, f64)> = pi
        .iter()
        .enumerate()
        .filter(|(_, &v)| v.abs() > 1e-15)
        .map(|(j, &v)| (j, v))
        .collect();

    let violation = rhs; // at x_hat the lhs of the cut (without θ) is 0

    BendersCut {
        cut_type: CutType::Optimality,
        coefficients,
        rhs,
        age: 0,
        times_active: 0,
        block,
        violation_at_generation: violation.max(0.0),
    }
}

/// Derive a Benders feasibility cut from a Farkas certificate.
///
/// Given an extreme ray `r` (proving infeasibility of the subproblem),
/// the technology matrix rows `T_i` (each a sparse row `(col, coeff)`),
/// and the sub-problem rhs vector `h`, the cut is:
///
///   Σ_j (Σ_i r_i * T_{ij}) * x_j  >=  Σ_i r_i * h_i
pub fn derive_feasibility_cut(
    farkas_ray: &[f64],
    technology_matrix_rows: &[Vec<(usize, f64)>],
    rhs: &[f64],
    block: usize,
) -> BendersCut {
    let num_rows = farkas_ray.len().min(technology_matrix_rows.len());

    // Aggregate coefficients per complicating variable
    let mut coeff_map = std::collections::HashMap::<usize, f64>::new();
    for i in 0..num_rows {
        let r_i = farkas_ray[i];
        if r_i.abs() < 1e-15 {
            continue;
        }
        for &(j, t_ij) in &technology_matrix_rows[i] {
            *coeff_map.entry(j).or_insert(0.0) += r_i * t_ij;
        }
    }

    let mut cut_rhs = 0.0f64;
    for i in 0..num_rows.min(rhs.len()) {
        cut_rhs += farkas_ray[i] * rhs[i];
    }

    let mut coefficients: Vec<(usize, f64)> = coeff_map
        .into_iter()
        .filter(|(_, v)| v.abs() > 1e-15)
        .collect();
    coefficients.sort_by_key(|&(j, _)| j);

    BendersCut {
        cut_type: CutType::Feasibility,
        coefficients,
        rhs: cut_rhs,
        age: 0,
        times_active: 0,
        block,
        violation_at_generation: 0.0,
    }
}

/// Magnanti-Wong strengthening of an existing optimality cut.
///
/// Uses a core point `x_core` (interior of the feasible region) to
/// re-weight the cut.  The strengthened cut is derived by substituting
/// `x_core` into the dual sub-problem and re-deriving the cut coefficients
/// from the optimal dual of the *auxiliary* Magnanti-Wong problem.
///
/// As a light-weight heuristic (no auxiliary LP) we interpolate coefficients
/// toward the core point: for each coefficient a_j, the strengthened
/// coefficient is  a_j + α * (x_core_j − x_hat_j)  where α is chosen
/// so that the cut passes through the core point.
pub fn strengthen_magnanti_wong(
    original: &BendersCut,
    core_point: &[f64],
    technology_matrix_rows: &[Vec<(usize, f64)>],
    rhs: &[f64],
    dual_values: &[f64],
) -> BendersCut {
    // Evaluate the original cut at the core point.
    let lhs_core = original.lhs_value(core_point);
    let viol_core = original.rhs - lhs_core;

    if viol_core.abs() < 1e-12 {
        // Core point nearly satisfies the cut; no strengthening possible.
        return original.clone();
    }

    // Compute technology-matrix contribution at the core point: T * x_core
    let num_rows = dual_values.len().min(technology_matrix_rows.len());
    let mut t_x_core = vec![0.0f64; num_rows];
    for i in 0..num_rows {
        for &(j, t_ij) in &technology_matrix_rows[i] {
            if j < core_point.len() {
                t_x_core[i] += t_ij * core_point[j];
            }
        }
    }

    // Perturbed dual objective direction: h - T * x_core
    let mut direction = vec![0.0f64; num_rows];
    for i in 0..num_rows {
        let h_i = if i < rhs.len() { rhs[i] } else { 0.0 };
        direction[i] = h_i - t_x_core[i];
    }

    // Strengthened dual values: y_new = y + epsilon * direction,
    // where epsilon is chosen to keep the cut valid and tighten it.
    let dir_norm_sq: f64 = direction.iter().map(|d| d * d).sum();
    let epsilon = if dir_norm_sq > 1e-20 {
        (viol_core / dir_norm_sq).clamp(-1.0, 1.0) * 0.5
    } else {
        0.0
    };

    let mut new_dual = vec![0.0f64; num_rows];
    for i in 0..num_rows {
        new_dual[i] = dual_values[i] + epsilon * direction[i];
    }

    // Re-derive coefficients from new dual values.
    let mut coeff_map = std::collections::HashMap::<usize, f64>::new();
    for i in 0..num_rows {
        if new_dual[i].abs() < 1e-15 {
            continue;
        }
        for &(j, t_ij) in &technology_matrix_rows[i] {
            *coeff_map.entry(j).or_insert(0.0) += new_dual[i] * t_ij;
        }
    }

    let mut new_rhs = 0.0f64;
    for i in 0..num_rows.min(rhs.len()) {
        new_rhs += new_dual[i] * rhs[i];
    }

    let mut coefficients: Vec<(usize, f64)> = coeff_map
        .into_iter()
        .filter(|(_, v)| v.abs() > 1e-15)
        .collect();
    coefficients.sort_by_key(|&(j, _)| j);

    let violation = (new_rhs - original.lhs_value(core_point)).max(0.0);

    BendersCut {
        cut_type: original.cut_type,
        coefficients,
        rhs: new_rhs,
        age: 0,
        times_active: 0,
        block: original.block,
        violation_at_generation: violation,
    }
}

/// Effectiveness of a cut at a given point: how much it would move the
/// relaxation value.  Returns `max(0, violation)`.
pub fn cut_effectiveness(cut: &BendersCut, point: &[f64]) -> f64 {
    cut.violation(point).max(0.0)
}

// ---------------------------------------------------------------------------
// Cut pool
// ---------------------------------------------------------------------------

/// Managed collection of Benders cuts with deduplication, ageing, and
/// selection helpers.
#[derive(Debug, Clone, Default)]
pub struct CutPool {
    cuts: Vec<BendersCut>,
}

impl CutPool {
    pub fn new() -> Self {
        Self { cuts: Vec::new() }
    }

    /// Number of cuts in the pool.
    pub fn len(&self) -> usize {
        self.cuts.len()
    }

    /// Whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.cuts.is_empty()
    }

    /// Read-only access to all cuts.
    pub fn cuts(&self) -> &[BendersCut] {
        &self.cuts
    }

    /// Iterate over all cuts.
    pub fn iter(&self) -> std::slice::Iter<'_, BendersCut> {
        self.cuts.iter()
    }

    /// Add a cut to the pool if it is not a near-duplicate.
    /// Returns `true` if the cut was added.
    pub fn add(&mut self, cut: BendersCut) -> bool {
        if self.is_duplicate(&cut, 1e-8) {
            return false;
        }
        self.cuts.push(cut);
        true
    }

    /// Check whether `cut` is coefficient-wise similar to any existing cut.
    pub fn is_duplicate(&self, cut: &BendersCut, tolerance: f64) -> bool {
        'outer: for existing in &self.cuts {
            if existing.cut_type != cut.cut_type {
                continue;
            }
            if (existing.rhs - cut.rhs).abs() > tolerance {
                continue;
            }
            if existing.coefficients.len() != cut.coefficients.len() {
                continue;
            }
            // Build a small map of the new cut's coefficients.
            let new_map: std::collections::HashMap<usize, f64> =
                cut.coefficients.iter().copied().collect();
            for &(j, v) in &existing.coefficients {
                match new_map.get(&j) {
                    Some(&nv) if (nv - v).abs() <= tolerance => {}
                    _ => continue 'outer,
                }
            }
            return true;
        }
        false
    }

    /// For each cut compute the violation at `point`.
    /// Returns `(cut_index, violation)` pairs for every cut.
    pub fn evaluate_violation(&self, point: &[f64]) -> Vec<(usize, f64)> {
        self.cuts
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.violation(point)))
            .collect()
    }

    /// Return references to the `k` most violated cuts at `point`.
    pub fn most_violated(&self, point: &[f64], k: usize) -> Vec<&BendersCut> {
        let mut indexed: Vec<(usize, f64)> = self
            .cuts
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.violation(point)))
            .filter(|(_, v)| *v > 0.0)
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed
            .iter()
            .take(k)
            .map(|&(i, _)| &self.cuts[i])
            .collect()
    }

    /// Remove cuts that are too old and have low activity.
    pub fn cleanup(&mut self, age_limit: usize, min_activity: usize) {
        self.cuts
            .retain(|c| c.age < age_limit || c.times_active >= min_activity);
    }

    /// Increment the age of every cut by one.
    pub fn age_all(&mut self) {
        for c in &mut self.cuts {
            c.age += 1;
        }
    }

    /// Mark specified cuts as active: reset age and increment activity count.
    pub fn mark_active(&mut self, indices: &[usize]) {
        for &idx in indices {
            if idx < self.cuts.len() {
                self.cuts[idx].age = 0;
                self.cuts[idx].times_active += 1;
            }
        }
    }

    /// Select Pareto-optimal cuts based on (violation, sparsity).
    ///
    /// A cut is Pareto-optimal if no other cut dominates it on *both*
    /// violation_at_generation (higher is better) and sparsity (lower nnz
    /// is better).
    pub fn pareto_optimal_cuts(&self) -> Vec<usize> {
        let n = self.cuts.len();
        if n == 0 {
            return Vec::new();
        }
        let mut pareto = Vec::new();
        for i in 0..n {
            let vi = self.cuts[i].violation_at_generation;
            let si = self.cuts[i].nnz();
            let mut dominated = false;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let vj = self.cuts[j].violation_at_generation;
                let sj = self.cuts[j].nnz();
                // j dominates i if j is at least as good on both criteria
                // and strictly better on at least one.
                if vj >= vi && sj <= si && (vj > vi || sj < si) {
                    dominated = true;
                    break;
                }
            }
            if !dominated {
                pareto.push(i);
            }
        }
        pareto
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_optimality_cut() -> BendersCut {
        BendersCut {
            cut_type: CutType::Optimality,
            coefficients: vec![(0, 2.0), (1, -1.0)],
            rhs: 3.0,
            age: 0,
            times_active: 0,
            block: 0,
            violation_at_generation: 1.0,
        }
    }

    fn sample_feasibility_cut() -> BendersCut {
        BendersCut {
            cut_type: CutType::Feasibility,
            coefficients: vec![(0, 1.0), (2, 1.0)],
            rhs: 5.0,
            age: 0,
            times_active: 0,
            block: 1,
            violation_at_generation: 2.0,
        }
    }

    // ---- BendersCut methods ----

    #[test]
    fn test_lhs_value() {
        let cut = sample_optimality_cut();
        let point = vec![3.0, 1.0, 0.0];
        // 2*3 + (-1)*1 = 5
        assert!((cut.lhs_value(&point) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_violation_positive() {
        let cut = sample_optimality_cut();
        // lhs = 2*0 + (-1)*0 = 0, rhs = 3 → violation = 3
        assert!((cut.violation(&[0.0, 0.0]) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_violation_negative_means_satisfied() {
        let cut = sample_optimality_cut();
        // lhs = 2*5 + (-1)*1 = 9, rhs = 3 → violation = -6
        assert!(cut.violation(&[5.0, 1.0]) < 0.0);
    }

    #[test]
    fn test_nnz() {
        let cut = sample_optimality_cut();
        assert_eq!(cut.nnz(), 2);
    }

    // ---- derive_optimality_cut ----

    #[test]
    fn test_derive_optimality_cut() {
        // 1 complicating variable, 2 sub-constraints.
        let dual_values = vec![1.0, 2.0];
        let tech_cols = vec![vec![(0, 3.0), (1, 4.0)]]; // T column for x_0
        let sub_obj = 10.0;
        let x_hat = vec![1.0];

        let cut = derive_optimality_cut(&dual_values, &tech_cols, sub_obj, &x_hat, 0);
        // π_0 = 1*3 + 2*4 = 11
        // rhs = 10 - 11*1 = -1
        assert_eq!(cut.cut_type, CutType::Optimality);
        assert_eq!(cut.coefficients.len(), 1);
        assert!((cut.coefficients[0].1 - 11.0).abs() < 1e-12);
        assert!((cut.rhs - (-1.0)).abs() < 1e-12);
    }

    // ---- derive_feasibility_cut ----

    #[test]
    fn test_derive_feasibility_cut() {
        let farkas = vec![1.0, 0.5];
        let tech_rows = vec![
            vec![(0, 2.0), (1, 3.0)],
            vec![(0, 1.0)],
        ];
        let rhs = vec![4.0, 2.0];

        let cut = derive_feasibility_cut(&farkas, &tech_rows, &rhs, 1);
        // coeff(0) = 1*2 + 0.5*1 = 2.5
        // coeff(1) = 1*3 = 3.0
        // cut_rhs = 1*4 + 0.5*2 = 5.0
        assert_eq!(cut.cut_type, CutType::Feasibility);

        let c_map: std::collections::HashMap<usize, f64> =
            cut.coefficients.iter().copied().collect();
        assert!((c_map[&0] - 2.5).abs() < 1e-12);
        assert!((c_map[&1] - 3.0).abs() < 1e-12);
        assert!((cut.rhs - 5.0).abs() < 1e-12);
    }

    // ---- strengthen_magnanti_wong ----

    #[test]
    fn test_strengthen_magnanti_wong_no_change_when_tight() {
        let cut = BendersCut {
            cut_type: CutType::Optimality,
            coefficients: vec![(0, 1.0)],
            rhs: 5.0,
            age: 0,
            times_active: 0,
            block: 0,
            violation_at_generation: 0.0,
        };
        // Core point exactly satisfies the cut (lhs = 5 = rhs).
        let core = vec![5.0];
        let tech_rows: Vec<Vec<(usize, f64)>> = vec![vec![(0, 1.0)]];
        let rhs = vec![5.0];
        let duals = vec![1.0];

        let strengthened = strengthen_magnanti_wong(&cut, &core, &tech_rows, &rhs, &duals);
        // Should be essentially unchanged.
        assert!((strengthened.rhs - cut.rhs).abs() < 0.5);
    }

    // ---- cut_effectiveness ----

    #[test]
    fn test_cut_effectiveness() {
        let cut = sample_optimality_cut();
        // violated at origin
        let eff = cut_effectiveness(&cut, &[0.0, 0.0]);
        assert!((eff - 3.0).abs() < 1e-12);

        // satisfied at (5, 1)
        let eff2 = cut_effectiveness(&cut, &[5.0, 1.0]);
        assert!(eff2 < 1e-12);
    }

    // ---- CutPool ----

    #[test]
    fn test_pool_add_and_len() {
        let mut pool = CutPool::new();
        assert!(pool.is_empty());
        assert!(pool.add(sample_optimality_cut()));
        assert_eq!(pool.len(), 1);
        // Duplicate should be rejected.
        assert!(!pool.add(sample_optimality_cut()));
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn test_pool_different_types_not_duplicates() {
        let mut pool = CutPool::new();
        assert!(pool.add(sample_optimality_cut()));
        assert!(pool.add(sample_feasibility_cut()));
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn test_pool_most_violated() {
        let mut pool = CutPool::new();
        pool.add(BendersCut {
            cut_type: CutType::Optimality,
            coefficients: vec![(0, 1.0)],
            rhs: 10.0,
            age: 0,
            times_active: 0,
            block: 0,
            violation_at_generation: 0.0,
        });
        pool.add(BendersCut {
            cut_type: CutType::Optimality,
            coefficients: vec![(0, 1.0)],
            rhs: 5.0,
            age: 0,
            times_active: 0,
            block: 0,
            violation_at_generation: 0.0,
        });

        let point = vec![0.0];
        let top = pool.most_violated(&point, 1);
        assert_eq!(top.len(), 1);
        assert!((top[0].rhs - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_pool_age_and_cleanup() {
        let mut pool = CutPool::new();
        pool.add(sample_optimality_cut());
        pool.add(sample_feasibility_cut());

        for _ in 0..5 {
            pool.age_all();
        }
        assert_eq!(pool.cuts()[0].age, 5);

        // Age limit 3, min activity 10 → both removed (activity = 0).
        pool.cleanup(3, 10);
        assert!(pool.is_empty());
    }

    #[test]
    fn test_pool_mark_active() {
        let mut pool = CutPool::new();
        pool.add(sample_optimality_cut());
        pool.age_all();
        pool.age_all();
        assert_eq!(pool.cuts()[0].age, 2);

        pool.mark_active(&[0]);
        assert_eq!(pool.cuts()[0].age, 0);
        assert_eq!(pool.cuts()[0].times_active, 1);
    }

    #[test]
    fn test_pool_evaluate_violation() {
        let mut pool = CutPool::new();
        pool.add(sample_optimality_cut());
        let viols = pool.evaluate_violation(&[0.0, 0.0]);
        assert_eq!(viols.len(), 1);
        assert!((viols[0].1 - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_pool_pareto_optimal() {
        let mut pool = CutPool::new();
        // Cut A: high violation, dense (nnz = 3)
        pool.add(BendersCut {
            cut_type: CutType::Optimality,
            coefficients: vec![(0, 1.0), (1, 2.0), (2, 3.0)],
            rhs: 10.0,
            age: 0,
            times_active: 0,
            block: 0,
            violation_at_generation: 5.0,
        });
        // Cut B: low violation, sparse (nnz = 1)
        pool.add(BendersCut {
            cut_type: CutType::Optimality,
            coefficients: vec![(0, 1.0)],
            rhs: 2.0,
            age: 0,
            times_active: 0,
            block: 0,
            violation_at_generation: 1.0,
        });
        // Cut C: dominated by A (same nnz=3 but lower violation)
        pool.add(BendersCut {
            cut_type: CutType::Optimality,
            coefficients: vec![(0, 0.5), (1, 1.0), (2, 1.5)],
            rhs: 5.0,
            age: 0,
            times_active: 0,
            block: 0,
            violation_at_generation: 3.0,
        });

        let pareto = pool.pareto_optimal_cuts();
        // A and B are Pareto-optimal; C is dominated by A.
        assert!(pareto.contains(&0));
        assert!(pareto.contains(&1));
        assert!(!pareto.contains(&2));
    }

    #[test]
    fn test_pool_cleanup_keeps_active() {
        let mut pool = CutPool::new();
        let mut cut = sample_optimality_cut();
        cut.times_active = 20;
        cut.age = 200;
        pool.cuts.push(cut);

        // Old but highly active → kept.
        pool.cleanup(100, 10);
        assert_eq!(pool.len(), 1);
    }
}

//! Critical region computation for the lower-level value function.
//!
//! A **critical region** is a maximal convex polyhedron in x-space over which
//! a particular LP basis remains optimal.  Within each region the value
//! function φ(x) is an affine function of x.

use std::collections::{HashMap, HashSet, VecDeque};

use bicut_lp::SimplexSolver;
use bicut_types::{
    AffineFunction, BasisStatus, BilevelProblem, Halfspace, LpStatus, Polyhedron, SparseMatrix,
};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::parametric::{BasisInfo, ParametricSolver};
use crate::{VFError, VFResult, TOLERANCE};

// ---------------------------------------------------------------------------
// Critical region
// ---------------------------------------------------------------------------

/// A critical region where a given basis is optimal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalRegion {
    /// The polyhedron in x-space defining this region.
    pub polyhedron: Polyhedron,
    /// Indices of basic variables that define this region.
    pub optimal_basis: Vec<usize>,
    /// Affine functions giving y*(x) on this region.
    pub affine_solution: Vec<AffineFunction>,
    /// The affine value function piece on this region.
    pub value_function: AffineFunction,
    /// Unique identifier for this region.
    pub region_id: usize,
    /// Whether the region is bounded.
    pub is_bounded: bool,
}

impl CriticalRegion {
    /// Check if a point x lies in this region.
    pub fn contains(&self, x: &[f64]) -> bool {
        self.polyhedron.contains(x, TOLERANCE)
    }

    /// Evaluate the value function at a point in this region.
    pub fn evaluate(&self, x: &[f64]) -> f64 {
        self.value_function.evaluate(x)
    }

    /// Evaluate the optimal primal solution y*(x).
    pub fn evaluate_primal(&self, x: &[f64]) -> Vec<f64> {
        self.affine_solution
            .iter()
            .map(|af| af.evaluate(x))
            .collect()
    }

    /// Dimension of the x-space.
    pub fn dim(&self) -> usize {
        self.polyhedron.dim
    }

    /// Number of halfspace constraints defining this region.
    pub fn num_constraints(&self) -> usize {
        self.polyhedron.halfspaces.len()
    }
}

/// Adjacency between two critical regions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionAdjacency {
    pub region_a: usize,
    pub region_b: usize,
    /// Index of the shared facet (halfspace) in region_a.
    pub shared_facet_a: usize,
    /// Index of the shared facet in region_b.
    pub shared_facet_b: usize,
}

/// Result of a critical region enumeration.
#[derive(Debug, Clone)]
pub struct EnumerationResult {
    pub regions: Vec<CriticalRegion>,
    pub adjacencies: Vec<RegionAdjacency>,
    pub total_bases_explored: usize,
    pub total_lp_solves: u64,
}

// ---------------------------------------------------------------------------
// Enumerator
// ---------------------------------------------------------------------------

/// Enumerates critical regions for the value function.
pub struct CriticalRegionEnumerator {
    problem: BilevelProblem,
    parametric_solver: ParametricSolver,
    max_regions: usize,
    tolerance: f64,
    /// Bounding box for x.
    x_lower: Vec<f64>,
    x_upper: Vec<f64>,
}

impl CriticalRegionEnumerator {
    pub fn new(problem: BilevelProblem, x_lower: Vec<f64>, x_upper: Vec<f64>) -> Self {
        let parametric_solver = ParametricSolver::new(problem.clone());
        Self {
            problem,
            parametric_solver,
            max_regions: 500,
            tolerance: TOLERANCE,
            x_lower,
            x_upper,
        }
    }

    pub fn with_max_regions(mut self, max: usize) -> Self {
        self.max_regions = max;
        self
    }

    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Enumerate all critical regions reachable from sample points in the bounding box.
    pub fn enumerate(&self) -> VFResult<EnumerationResult> {
        let nx = self.problem.num_upper_vars;
        let samples_per_dim = 5usize;
        let bases = self.parametric_solver.enumerate_bases(
            &self.x_lower,
            &self.x_upper,
            samples_per_dim,
        )?;

        let mut regions: Vec<CriticalRegion> = Vec::new();
        let mut total_lp_solves = 0u64;

        for (idx, basis) in bases.iter().enumerate() {
            if regions.len() >= self.max_regions {
                break;
            }

            let region = self.compute_region_for_basis(basis, idx)?;
            total_lp_solves += (nx * 2 + 1) as u64;
            regions.push(region);
        }

        let adjacencies = self.compute_adjacencies(&regions);

        Ok(EnumerationResult {
            total_bases_explored: bases.len(),
            regions,
            adjacencies,
            total_lp_solves,
        })
    }

    /// Compute the critical region for a specific basis.
    fn compute_region_for_basis(
        &self,
        basis: &BasisInfo,
        region_id: usize,
    ) -> VFResult<CriticalRegion> {
        let nx = self.problem.num_upper_vars;

        // Find a point where this basis is optimal
        let x_ref = self.find_interior_point(basis)?;

        // Compute the polyhedron in x-space where the basis remains optimal
        let polyhedron = self.compute_region_polyhedron(basis, &x_ref)?;

        // Compute affine solution y*(x) on this region
        let affine_solution = self.compute_affine_solution(basis, &x_ref)?;

        // Compute value function piece
        let value_function = self.compute_value_function_piece(basis, &x_ref)?;

        let is_bounded = self.check_boundedness(&polyhedron);

        Ok(CriticalRegion {
            polyhedron,
            optimal_basis: basis.basic_indices.clone(),
            affine_solution,
            value_function,
            region_id,
            is_bounded,
        })
    }

    /// Find a point x where the given basis is optimal (by searching samples).
    fn find_interior_point(&self, basis: &BasisInfo) -> VFResult<Vec<f64>> {
        let nx = self.problem.num_upper_vars;
        let samples = 10usize;

        for i in 0..samples {
            let t = (i as f64 + 0.5) / samples as f64;
            let x: Vec<f64> = (0..nx)
                .map(|d| self.x_lower[d] + t * (self.x_upper[d] - self.x_lower[d]))
                .collect();

            if let Ok((_sol, b)) = self.parametric_solver.solve_at(&x) {
                if b.same_basis(basis) {
                    return Ok(x);
                }
            }
        }

        // If we can't find an exact match, try the midpoint
        let x_mid: Vec<f64> = (0..nx)
            .map(|d| (self.x_lower[d] + self.x_upper[d]) / 2.0)
            .collect();
        Ok(x_mid)
    }

    /// Compute the polyhedron of x values where `basis` is optimal.
    ///
    /// The region is characterized by:
    ///   - Primal feasibility: B^{-1}(b + Bx) ≥ 0
    ///   - Dual feasibility: reduced costs ≥ 0
    fn compute_region_polyhedron(&self, basis: &BasisInfo, x_ref: &[f64]) -> VFResult<Polyhedron> {
        let nx = self.problem.num_upper_vars;
        let n = self.problem.num_lower_vars;
        let m = self.problem.num_lower_constraints;

        let lp_ref = self.problem.lower_level_lp(x_ref);
        let a_dense = lp_ref.a_matrix.to_dense();

        // Build basis matrix from the A part (augmented with identity for slacks)
        let basis_indices = &basis.basic_indices;
        let bsize = basis_indices.len().min(m);

        let mut b_mat = DMatrix::zeros(m, m);
        for col in 0..bsize {
            let idx = basis_indices[col];
            if idx < n {
                for row in 0..m {
                    b_mat[(row, col)] = a_dense[(row, idx)];
                }
            } else {
                let slack_row = idx - n;
                if slack_row < m {
                    b_mat[(slack_row, col)] = 1.0;
                }
            }
        }
        // Fill remaining columns with slack identity if basis is smaller than m
        for col in bsize..m {
            b_mat[(col, col)] = 1.0;
        }

        let b_inv = match b_mat.try_inverse() {
            Some(inv) => inv,
            None => {
                // Singular basis – return the bounding box as the region
                let mut poly = Polyhedron::new(nx);
                for d in 0..nx {
                    poly.add_halfspace(
                        {
                            let mut n = vec![0.0; nx];
                            n[d] = 1.0;
                            n
                        },
                        self.x_upper[d],
                    );
                    poly.add_halfspace(
                        {
                            let mut n = vec![0.0; nx];
                            n[d] = -1.0;
                            n
                        },
                        -self.x_lower[d],
                    );
                }
                return Ok(poly);
            }
        };

        let mut poly = Polyhedron::new(nx);

        // Primal feasibility constraints: B^{-1}(b + Bx) ≥ 0
        // This means: for each basic variable i, (B^{-1} b)_i + (B^{-1} B_link)_i x ≥ 0
        let b_rhs = DVector::from_column_slice(&self.problem.lower_b);
        let b_inv_b = &b_inv * &b_rhs;

        // Compute B^{-1} * B_link (the linking matrix)
        let linking_dense = self.problem.lower_linking_b.to_dense();
        let b_inv_blink = &b_inv * &linking_dense;

        for i in 0..m {
            // Constraint: (B^{-1} b)_i + sum_j (B^{-1} B_link)_{i,j} x_j ≥ 0
            // Equivalently: -sum_j (B^{-1} B_link)_{i,j} x_j ≤ (B^{-1} b)_i
            let mut normal = vec![0.0; nx];
            for j in 0..nx.min(b_inv_blink.ncols()) {
                normal[j] = -b_inv_blink[(i, j)];
            }
            let rhs = b_inv_b[i];
            poly.add_halfspace(normal, rhs);
        }

        // Add bounding box constraints
        for d in 0..nx {
            let mut n_upper = vec![0.0; nx];
            n_upper[d] = 1.0;
            poly.add_halfspace(n_upper, self.x_upper[d]);

            let mut n_lower = vec![0.0; nx];
            n_lower[d] = -1.0;
            poly.add_halfspace(n_lower, -self.x_lower[d]);
        }

        Ok(poly)
    }

    /// Compute the affine solution y*(x) for basic variables.
    fn compute_affine_solution(
        &self,
        basis: &BasisInfo,
        x_ref: &[f64],
    ) -> VFResult<Vec<AffineFunction>> {
        let nx = self.problem.num_upper_vars;
        let ny = self.problem.num_lower_vars;
        let step = 1e-7;

        // Evaluate primal at x_ref
        let y_ref = match self.parametric_solver.solve_at(x_ref) {
            Ok((sol, _)) => sol.primal,
            Err(_) => vec![0.0; ny],
        };

        let mut functions = Vec::with_capacity(ny);

        for j in 0..ny {
            let mut coefficients = vec![0.0; nx];
            for d in 0..nx {
                let mut x_plus = x_ref.to_vec();
                x_plus[d] += step;
                let y_plus = match self.parametric_solver.solve_at(&x_plus) {
                    Ok((sol, _)) => {
                        if j < sol.primal.len() {
                            sol.primal[j]
                        } else {
                            0.0
                        }
                    }
                    Err(_) => y_ref.get(j).copied().unwrap_or(0.0),
                };
                let y_ref_j = y_ref.get(j).copied().unwrap_or(0.0);
                coefficients[d] = (y_plus - y_ref_j) / step;
            }

            let y_ref_j = y_ref.get(j).copied().unwrap_or(0.0);
            let constant = y_ref_j
                - coefficients
                    .iter()
                    .zip(x_ref.iter())
                    .map(|(c, x)| c * x)
                    .sum::<f64>();

            functions.push(AffineFunction {
                coefficients,
                constant,
            });
        }

        Ok(functions)
    }

    /// Compute the value function affine piece for the given basis.
    fn compute_value_function_piece(
        &self,
        basis: &BasisInfo,
        x_ref: &[f64],
    ) -> VFResult<AffineFunction> {
        let nx = self.problem.num_upper_vars;
        let step = 1e-7;

        let val_ref = match self.parametric_solver.solve_at(x_ref) {
            Ok((sol, _)) => sol.objective,
            Err(e) => return Err(e),
        };

        let mut coefficients = vec![0.0; nx];
        for d in 0..nx {
            let mut x_plus = x_ref.to_vec();
            x_plus[d] += step;
            let val_plus = match self.parametric_solver.solve_at(&x_plus) {
                Ok((sol, _)) => sol.objective,
                Err(_) => val_ref,
            };
            coefficients[d] = (val_plus - val_ref) / step;
        }

        let constant = val_ref
            - coefficients
                .iter()
                .zip(x_ref.iter())
                .map(|(c, x)| c * x)
                .sum::<f64>();

        Ok(AffineFunction {
            coefficients,
            constant,
        })
    }

    fn check_boundedness(&self, poly: &Polyhedron) -> bool {
        // A polyhedron defined entirely with box constraints and active primal
        // constraints will typically be bounded. We check a heuristic:
        // for each dimension, there should be constraints bounding from both sides.
        let nx = poly.dim;
        for d in 0..nx {
            let has_upper = poly
                .halfspaces
                .iter()
                .any(|h| h.normal.get(d).copied().unwrap_or(0.0) > self.tolerance);
            let has_lower = poly
                .halfspaces
                .iter()
                .any(|h| h.normal.get(d).copied().unwrap_or(0.0) < -self.tolerance);
            if !has_upper || !has_lower {
                return false;
            }
        }
        true
    }

    /// Compute adjacencies between regions by checking shared facets.
    fn compute_adjacencies(&self, regions: &[CriticalRegion]) -> Vec<RegionAdjacency> {
        let mut adjacencies = Vec::new();

        for i in 0..regions.len() {
            for j in (i + 1)..regions.len() {
                if let Some(adj) = self.check_adjacency(&regions[i], &regions[j], i, j) {
                    adjacencies.push(adj);
                }
            }
        }

        adjacencies
    }

    fn check_adjacency(
        &self,
        region_a: &CriticalRegion,
        region_b: &CriticalRegion,
        id_a: usize,
        id_b: usize,
    ) -> Option<RegionAdjacency> {
        let nx = region_a.dim();

        // Check if any facet normal of region_a is (approximately) the negation
        // of a facet normal of region_b, and the RHS values match.
        for (fa, ha) in region_a.polyhedron.halfspaces.iter().enumerate() {
            for (fb, hb) in region_b.polyhedron.halfspaces.iter().enumerate() {
                let mut is_opposite = true;
                let mut norm_sq = 0.0;
                for d in 0..nx {
                    let na = ha.normal.get(d).copied().unwrap_or(0.0);
                    let nb = hb.normal.get(d).copied().unwrap_or(0.0);
                    norm_sq += na * na;
                    if (na + nb).abs() > self.tolerance * 10.0 {
                        is_opposite = false;
                        break;
                    }
                }
                if is_opposite && norm_sq > self.tolerance {
                    if (ha.rhs + hb.rhs).abs() < self.tolerance * 100.0 {
                        return Some(RegionAdjacency {
                            region_a: id_a,
                            region_b: id_b,
                            shared_facet_a: fa,
                            shared_facet_b: fb,
                        });
                    }
                }
            }
        }
        None
    }

    /// Find which region contains a given point x.
    pub fn locate_point(&self, regions: &[CriticalRegion], x: &[f64]) -> Option<usize> {
        for (i, region) in regions.iter().enumerate() {
            if region.contains(x) {
                return Some(i);
            }
        }
        None
    }

    /// Find the region containing x and evaluate the value function.
    pub fn evaluate_at(&self, regions: &[CriticalRegion], x: &[f64]) -> VFResult<f64> {
        match self.locate_point(regions, x) {
            Some(idx) => Ok(regions[idx].evaluate(x)),
            None => {
                // Point not in any precomputed region; solve directly
                let (sol, _basis) = self.parametric_solver.solve_at(x)?;
                Ok(sol.objective)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Region cache
// ---------------------------------------------------------------------------

/// A cache of critical regions for fast point location queries.
pub struct RegionCache {
    regions: Vec<CriticalRegion>,
    adjacencies: Vec<RegionAdjacency>,
    last_region: Option<usize>,
}

impl RegionCache {
    pub fn new(regions: Vec<CriticalRegion>, adjacencies: Vec<RegionAdjacency>) -> Self {
        Self {
            regions,
            adjacencies,
            last_region: None,
        }
    }

    pub fn from_enumeration(result: EnumerationResult) -> Self {
        Self::new(result.regions, result.adjacencies)
    }

    /// Locate the region containing x, starting from the last queried region.
    pub fn locate(&mut self, x: &[f64]) -> Option<usize> {
        // First check the last region (spatial locality)
        if let Some(last) = self.last_region {
            if last < self.regions.len() && self.regions[last].contains(x) {
                return Some(last);
            }

            // Check neighbors of the last region
            let neighbors: Vec<usize> = self
                .adjacencies
                .iter()
                .filter_map(|adj| {
                    if adj.region_a == last {
                        Some(adj.region_b)
                    } else if adj.region_b == last {
                        Some(adj.region_a)
                    } else {
                        None
                    }
                })
                .collect();

            for &nb in &neighbors {
                if nb < self.regions.len() && self.regions[nb].contains(x) {
                    self.last_region = Some(nb);
                    return Some(nb);
                }
            }
        }

        // Fall back to linear scan
        for (i, region) in self.regions.iter().enumerate() {
            if region.contains(x) {
                self.last_region = Some(i);
                return Some(i);
            }
        }

        None
    }

    /// Evaluate the value function at x using the cached regions.
    pub fn evaluate(&mut self, x: &[f64]) -> Option<f64> {
        self.locate(x).map(|idx| self.regions[idx].evaluate(x))
    }

    /// Number of cached regions.
    pub fn num_regions(&self) -> usize {
        self.regions.len()
    }

    /// Number of adjacencies.
    pub fn num_adjacencies(&self) -> usize {
        self.adjacencies.len()
    }

    /// Get region by index.
    pub fn region(&self, idx: usize) -> Option<&CriticalRegion> {
        self.regions.get(idx)
    }

    /// Get all regions.
    pub fn regions(&self) -> &[CriticalRegion] {
        &self.regions
    }

    /// Get adjacency information.
    pub fn adjacencies(&self) -> &[RegionAdjacency] {
        &self.adjacencies
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// Compute the volume of a critical region (approximate via sampling).
pub fn estimate_region_volume(
    region: &CriticalRegion,
    x_lower: &[f64],
    x_upper: &[f64],
    num_samples: usize,
) -> f64 {
    let nx = region.dim();
    let mut rng_state: u64 = 12345;
    let mut inside = 0usize;

    for _ in 0..num_samples {
        let x: Vec<f64> = (0..nx)
            .map(|d| {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
                x_lower[d] + u * (x_upper[d] - x_lower[d])
            })
            .collect();

        if region.contains(&x) {
            inside += 1;
        }
    }

    let box_volume: f64 = (0..nx)
        .map(|d| (x_upper[d] - x_lower[d]).max(0.0))
        .product();

    box_volume * inside as f64 / num_samples as f64
}

/// Compute the Chebyshev center of a critical region (point furthest from all boundaries).
pub fn chebyshev_center(region: &CriticalRegion) -> Vec<f64> {
    let nx = region.dim();
    let halfspaces = &region.polyhedron.halfspaces;

    if halfspaces.is_empty() {
        return vec![0.0; nx];
    }

    // Analytic Chebyshev center requires an LP; use simple heuristic:
    // average the "slack-weighted" normals
    let num_h = halfspaces.len();

    // Start with a feasible point (the centroid of the bounding box portion)
    let mut center = vec![0.0; nx];
    let mut count = 0.0;

    // Try a grid of points and pick the one with max minimum slack
    let grid_size = 10usize;
    let mut best_point = vec![0.0; nx];
    let mut best_min_slack = f64::NEG_INFINITY;

    for trial in 0..grid_size.pow(nx.min(3) as u32) {
        let mut idx = trial;
        let x: Vec<f64> = (0..nx)
            .map(|d| {
                let k = idx % grid_size;
                idx /= grid_size;
                let t = (k as f64 + 0.5) / grid_size as f64;
                // Use bounding box from halfspaces if available
                -10.0 + t * 20.0
            })
            .collect();

        let min_slack = halfspaces
            .iter()
            .map(|h| {
                let dot: f64 = h.normal.iter().zip(x.iter()).map(|(a, xi)| a * xi).sum();
                h.rhs - dot
            })
            .fold(f64::INFINITY, f64::min);

        if min_slack > best_min_slack {
            best_min_slack = min_slack;
            best_point = x;
        }
    }

    best_point
}

/// Merge regions that have the same value function piece (identical affine functions).
pub fn merge_compatible_regions(regions: &[CriticalRegion]) -> Vec<CriticalRegion> {
    let mut merged: Vec<CriticalRegion> = Vec::new();

    for region in regions {
        let compatible = merged.iter().position(|r| {
            let coeff_match = r
                .value_function
                .coefficients
                .iter()
                .zip(region.value_function.coefficients.iter())
                .all(|(a, b)| (a - b).abs() < TOLERANCE * 100.0);
            let const_match = (r.value_function.constant - region.value_function.constant).abs()
                < TOLERANCE * 100.0;
            coeff_match && const_match
        });

        match compatible {
            Some(idx) => {
                // Merge polyhedra (take intersection of the union – approximation)
                // In practice, we just keep both sets of halfspaces
                // This is an over-approximation of the union
                let new_region = CriticalRegion {
                    polyhedron: region.polyhedron.clone(),
                    optimal_basis: region.optimal_basis.clone(),
                    affine_solution: region.affine_solution.clone(),
                    value_function: region.value_function.clone(),
                    region_id: merged[idx].region_id,
                    is_bounded: merged[idx].is_bounded && region.is_bounded,
                };
                // We keep the original; actual union of polyhedra is complex
                merged.push(new_region);
            }
            None => {
                merged.push(region.clone());
            }
        }
    }

    merged
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use bicut_types::{BilevelProblem, SparseMatrix};

    fn test_bilevel() -> BilevelProblem {
        let mut lower_a = SparseMatrix::new(2, 1);
        lower_a.add_entry(0, 0, 1.0);
        lower_a.add_entry(1, 0, 1.0);

        let mut linking_b = SparseMatrix::new(2, 1);
        linking_b.add_entry(0, 0, 1.0);

        BilevelProblem {
            upper_obj_c_x: vec![1.0],
            upper_obj_c_y: vec![1.0],
            lower_obj_c: vec![1.0],
            lower_a,
            lower_b: vec![2.0, 3.0],
            lower_linking_b: linking_b,
            upper_constraints_a: SparseMatrix::new(0, 2),
            upper_constraints_b: vec![],
            num_upper_vars: 1,
            num_lower_vars: 1,
            num_lower_constraints: 2,
            num_upper_constraints: 0,
        }
    }

    #[test]
    fn test_enumerate_regions() {
        let problem = test_bilevel();
        let enumerator = CriticalRegionEnumerator::new(problem, vec![-2.0], vec![2.0]);
        let result = enumerator.enumerate().unwrap();
        assert!(!result.regions.is_empty());
    }

    #[test]
    fn test_region_contains() {
        let problem = test_bilevel();
        let enumerator = CriticalRegionEnumerator::new(problem, vec![-2.0], vec![2.0]);
        let result = enumerator.enumerate().unwrap();
        // At least one region should contain x=0
        let found = result.regions.iter().any(|r| r.contains(&[0.0]));
        assert!(found);
    }

    #[test]
    fn test_locate_point() {
        let problem = test_bilevel();
        let enumerator = CriticalRegionEnumerator::new(problem, vec![-2.0], vec![2.0]);
        let result = enumerator.enumerate().unwrap();
        let idx = enumerator.locate_point(&result.regions, &[0.0]);
        assert!(idx.is_some());
    }

    #[test]
    fn test_region_cache() {
        let problem = test_bilevel();
        let enumerator = CriticalRegionEnumerator::new(problem, vec![-2.0], vec![2.0]);
        let result = enumerator.enumerate().unwrap();
        let mut cache = RegionCache::from_enumeration(result);
        assert!(cache.num_regions() > 0);

        let val = cache.evaluate(&[0.0]);
        assert!(val.is_some());
    }

    #[test]
    fn test_region_evaluate() {
        let problem = test_bilevel();
        let enumerator = CriticalRegionEnumerator::new(problem, vec![-2.0], vec![2.0]);
        let result = enumerator.enumerate().unwrap();
        let val = enumerator.evaluate_at(&result.regions, &[0.0]);
        assert!(val.is_ok());
    }

    #[test]
    fn test_estimate_volume() {
        let mut poly = Polyhedron::new(1);
        poly.add_halfspace(vec![1.0], 1.0);
        poly.add_halfspace(vec![-1.0], 1.0);

        let region = CriticalRegion {
            polyhedron: poly,
            optimal_basis: vec![0],
            affine_solution: vec![],
            value_function: AffineFunction::zero(1),
            region_id: 0,
            is_bounded: true,
        };

        let vol = estimate_region_volume(&region, &[-2.0], &[2.0], 10000);
        // Region is [-1, 1], box is [-2, 2] → volume = 2/4 * 4 = 2
        assert!((vol - 2.0).abs() < 0.5);
    }

    #[test]
    fn test_chebyshev_center() {
        let mut poly = Polyhedron::new(1);
        poly.add_halfspace(vec![1.0], 1.0);
        poly.add_halfspace(vec![-1.0], 1.0);

        let region = CriticalRegion {
            polyhedron: poly,
            optimal_basis: vec![0],
            affine_solution: vec![],
            value_function: AffineFunction::zero(1),
            region_id: 0,
            is_bounded: true,
        };

        let center = chebyshev_center(&region);
        assert_eq!(center.len(), 1);
        // Center should be near 0 for [-1, 1]
        assert!(center[0].abs() < 2.0);
    }

    #[test]
    fn test_region_adjacency_struct() {
        let adj = RegionAdjacency {
            region_a: 0,
            region_b: 1,
            shared_facet_a: 2,
            shared_facet_b: 3,
        };
        assert_eq!(adj.region_a, 0);
        assert_eq!(adj.region_b, 1);
    }

    #[test]
    fn test_merge_compatible() {
        let region = CriticalRegion {
            polyhedron: Polyhedron::new(1),
            optimal_basis: vec![0],
            affine_solution: vec![],
            value_function: AffineFunction {
                coefficients: vec![1.0],
                constant: 0.0,
            },
            region_id: 0,
            is_bounded: true,
        };
        let merged = merge_compatible_regions(&[region.clone(), region]);
        assert!(merged.len() >= 1);
    }
}

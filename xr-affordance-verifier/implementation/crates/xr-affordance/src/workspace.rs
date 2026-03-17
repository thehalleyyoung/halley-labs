//! Workspace analysis: volume, cross-sections, dexterous workspace.
//!
//! Provides tools for analyzing the kinematic workspace of a parameterized arm:
//! - Monte Carlo volume estimation with spatial hashing
//! - 2D cross-section slicing (horizontal and sagittal)
//! - Dexterous workspace identification (high-manipulability regions)
//! - Workspace comparison across body parameterizations

use xr_types::{
    BodyParameters, KinematicChain, BoundingBox, VerifierResult,
    kinematic::ArmSide,
    geometry::point_distance,
};
use crate::forward_kinematics::ForwardKinematicsSolver;
use crate::reach_envelope::{ReachEnvelope, ReachEnvelopeConfig, VoxelGrid};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for workspace analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceConfig {
    /// Number of Monte Carlo samples for volume estimation.
    pub monte_carlo_samples: usize,
    /// Voxel size for spatial hashing (meters).
    pub voxel_size: f64,
    /// Resolution for 2D cross-section grids.
    pub slice_resolution: usize,
    /// Manipulability threshold for dexterous workspace (0..1 relative to max).
    pub dexterous_threshold: f64,
    /// Number of height slices for volume-distribution analysis.
    pub distribution_slices: usize,
}

impl Default for WorkspaceConfig {
    fn default() -> Self {
        Self {
            monte_carlo_samples: 20_000,
            voxel_size: 0.02,
            slice_resolution: 64,
            dexterous_threshold: 0.15,
            distribution_slices: 20,
        }
    }
}

// ---------------------------------------------------------------------------
// WorkspaceSlice — 2D cross-section
// ---------------------------------------------------------------------------

/// A 2D cross-section of the workspace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceSlice {
    /// Origin of the slice grid (min-x, min-y in the slice's local 2D frame).
    pub origin: [f64; 2],
    /// Cell size of the slice grid.
    pub cell_size: f64,
    /// Grid dimensions (cols, rows).
    pub dims: [usize; 2],
    /// Boolean occupancy grid (row-major).
    pub grid: Vec<bool>,
    /// Height / offset at which this slice was taken.
    pub slice_offset: f64,
    /// Axis perpendicular to the slice ("Y" for horizontal, "X" for sagittal).
    pub normal_axis: String,
}

impl WorkspaceSlice {
    /// Check whether a 2D point (in the slice's local frame) is reachable.
    pub fn contains_point(&self, p: [f64; 2]) -> bool {
        let col = ((p[0] - self.origin[0]) / self.cell_size).floor() as isize;
        let row = ((p[1] - self.origin[1]) / self.cell_size).floor() as isize;
        if col < 0 || row < 0 || col as usize >= self.dims[0] || row as usize >= self.dims[1] {
            return false;
        }
        self.grid[row as usize * self.dims[0] + col as usize]
    }

    /// Compute the approximate area of occupied cells.
    pub fn compute_area(&self) -> f64 {
        let occupied = self.grid.iter().filter(|&&v| v).count();
        occupied as f64 * self.cell_size * self.cell_size
    }

    /// Approximate the convex area using a simple gift-wrapping estimation.
    /// We compute the convex hull perimeter estimate and use the shoelace
    /// formula on the boundary points.
    pub fn convex_area_approx(&self) -> f64 {
        let mut pts: Vec<[f64; 2]> = Vec::new();
        for row in 0..self.dims[1] {
            for col in 0..self.dims[0] {
                if self.grid[row * self.dims[0] + col] {
                    pts.push([
                        self.origin[0] + (col as f64 + 0.5) * self.cell_size,
                        self.origin[1] + (row as f64 + 0.5) * self.cell_size,
                    ]);
                }
            }
        }
        if pts.len() < 3 {
            return self.compute_area();
        }
        let hull = convex_hull_2d(&pts);
        shoelace_area(&hull)
    }
}

// ---------------------------------------------------------------------------
// WorkspaceProperties
// ---------------------------------------------------------------------------

/// Aggregate properties of a workspace analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceProperties {
    /// Total reachable volume (m³).
    pub volume: f64,
    /// Dexterous (high-manipulability) volume (m³).
    pub dexterous_volume: f64,
    /// Maximum reach distance from shoulder.
    pub max_reach: f64,
    /// Minimum reach distance from shoulder.
    pub min_reach: f64,
    /// Centroid of the reachable workspace (world space).
    pub centroid: [f64; 3],
    /// Asymmetry index (0 = symmetric, 1 = fully asymmetric).
    pub asymmetry: f64,
    /// Compactness = volume / bounding-sphere volume.
    pub compactness: f64,
    /// Mean manipulability over sampled configurations.
    pub mean_manipulability: f64,
    /// Reach profile: max-reach at each sampled height.
    pub reach_profile: Vec<(f64, f64)>,
}

// ---------------------------------------------------------------------------
// WorkspaceComparison
// ---------------------------------------------------------------------------

/// Result of comparing workspaces for two different body parameterizations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceComparison {
    /// Volume of body A's workspace.
    pub volume_a: f64,
    /// Volume of body B's workspace.
    pub volume_b: f64,
    /// Ratio volume_b / volume_a.
    pub volume_ratio: f64,
    /// Max-reach difference (B − A).
    pub max_reach_diff: f64,
    /// Centroid displacement between A and B.
    pub centroid_displacement: f64,
    /// Fraction of A's workspace also reachable by B (Jaccard-like).
    pub overlap_fraction: f64,
}

// ---------------------------------------------------------------------------
// WorkspaceAnalyzer
// ---------------------------------------------------------------------------

/// Analyzes the dexterous workspace of a kinematic chain.
pub struct WorkspaceAnalyzer {
    fk: ForwardKinematicsSolver,
    config: WorkspaceConfig,
    envelope_config: ReachEnvelopeConfig,
}

impl WorkspaceAnalyzer {
    pub fn new() -> Self {
        Self {
            fk: ForwardKinematicsSolver::new(),
            config: WorkspaceConfig::default(),
            envelope_config: ReachEnvelopeConfig::default(),
        }
    }

    pub fn with_config(mut self, config: ReachEnvelopeConfig) -> Self {
        self.envelope_config = config;
        self
    }

    pub fn with_workspace_config(mut self, config: WorkspaceConfig) -> Self {
        self.config = config;
        self
    }

    // -- Envelope delegation --

    /// Compute full reach envelope.
    pub fn compute_envelope(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
    ) -> VerifierResult<ReachEnvelope> {
        ReachEnvelope::compute(chain, params, &self.envelope_config)
    }

    // -- Core analysis --

    /// Full workspace analysis returning aggregate properties.
    pub fn analyze(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
    ) -> VerifierResult<WorkspaceProperties> {
        let base = chain.base_position(params);
        let max_r = ForwardKinematicsSolver::max_reach(chain, params);
        let n = self.config.monte_carlo_samples;
        let voxel_size = self.config.voxel_size;

        // Bounding box for spatial hash
        let bb = BoundingBox::new(
            [base[0] - max_r - 0.05, base[1] - max_r - 0.05, base[2] - max_r - 0.05],
            [base[0] + max_r + 0.05, base[1] + max_r + 0.05, base[2] + max_r + 0.05],
        );
        let mut grid = VoxelGrid::from_bounds(&bb, voxel_size);

        let mut rng = rand::thread_rng();
        let mut sum_pos = [0.0f64; 3];
        let mut sum_manip = 0.0f64;
        let mut max_manip = 0.0f64;
        let mut max_dist = 0.0f64;
        let mut min_dist = f64::INFINITY;
        let mut count = 0usize;

        // Collect manipulability per voxel for dexterous analysis
        let mut manip_map: HashMap<usize, f64> = HashMap::new();

        for _ in 0..n {
            let angles = chain.random_config(params, &mut rng);
            if let Ok(pos) = self.fk.solve_position(chain, params, &angles) {
                if !pos[0].is_finite() || !pos[1].is_finite() || !pos[2].is_finite() {
                    continue;
                }
                grid.mark_point(&pos);
                let dist = point_distance(&pos, &base);
                max_dist = max_dist.max(dist);
                if dist > 0.001 {
                    min_dist = min_dist.min(dist);
                }
                sum_pos[0] += pos[0];
                sum_pos[1] += pos[1];
                sum_pos[2] += pos[2];

                let m = self.fk.manipulability(chain, params, &angles).unwrap_or(0.0);
                sum_manip += m;
                max_manip = max_manip.max(m);
                if let Some(idx) = grid.point_to_index(&pos) {
                    let fi = idx[0] * grid.dims[1] * grid.dims[2]
                        + idx[1] * grid.dims[2]
                        + idx[2];
                    let e = manip_map.entry(fi).or_insert(0.0);
                    *e = e.max(m);
                }
                count += 1;
            }
        }

        if min_dist == f64::INFINITY {
            min_dist = 0.0;
        }
        let volume = grid.occupied_volume();
        let centroid = if count > 0 {
            [
                sum_pos[0] / count as f64,
                sum_pos[1] / count as f64,
                sum_pos[2] / count as f64,
            ]
        } else {
            base
        };
        let mean_manipulability = if count > 0 { sum_manip / count as f64 } else { 0.0 };

        // Dexterous volume: voxels whose max manipulability exceeds threshold
        let dex_thresh = self.config.dexterous_threshold * max_manip;
        let dexterous_count = manip_map.values().filter(|&&m| m >= dex_thresh).count();
        let dexterous_volume = dexterous_count as f64 * voxel_size.powi(3);

        // Compactness
        let bounding_sphere_vol = (4.0 / 3.0) * std::f64::consts::PI * max_dist.powi(3);
        let compactness = if bounding_sphere_vol > 1e-12 {
            volume / bounding_sphere_vol
        } else {
            0.0
        };

        // Asymmetry: compare +x vs −x reach from centroid
        let (mut lcount, mut rcount) = (0usize, 0usize);
        for ix in 0..grid.dims[0] {
            for iy in 0..grid.dims[1] {
                for iz in 0..grid.dims[2] {
                    let fi = ix * grid.dims[1] * grid.dims[2] + iy * grid.dims[2] + iz;
                    if grid.occupied[fi] {
                        let pt = grid.index_to_point([ix, iy, iz]);
                        if pt[0] < centroid[0] {
                            lcount += 1;
                        } else {
                            rcount += 1;
                        }
                    }
                }
            }
        }
        let total_occ = (lcount + rcount).max(1);
        let asymmetry = (lcount as f64 - rcount as f64).abs() / total_occ as f64;

        // Reach profile
        let reach_profile = self.reach_profile(chain, params, &grid, &base, max_r);

        Ok(WorkspaceProperties {
            volume,
            dexterous_volume,
            max_reach: max_dist,
            min_reach: min_dist,
            centroid,
            asymmetry,
            compactness,
            mean_manipulability,
            reach_profile,
        })
    }

    // -- Slicing --

    /// Take a horizontal (constant-Y) slice of the workspace.
    pub fn horizontal_slice(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        height: f64,
    ) -> WorkspaceSlice {
        self.take_slice(chain, params, 1, height)
    }

    /// Take a sagittal (constant-X) slice of the workspace.
    pub fn sagittal_slice(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        lateral_offset: f64,
    ) -> WorkspaceSlice {
        self.take_slice(chain, params, 0, lateral_offset)
    }

    /// Generic slicing: fix `axis` at `offset`, sample workspace in the
    /// remaining two axes via Monte Carlo.
    fn take_slice(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        fixed_axis: usize, // 0=X, 1=Y, 2=Z
        offset: f64,
    ) -> WorkspaceSlice {
        let base = chain.base_position(params);
        let max_r = ForwardKinematicsSolver::max_reach(chain, params);
        let res = self.config.slice_resolution;
        let cell_size = 2.0 * (max_r + 0.05) / res as f64;

        // The two free axes
        let (ax_a, ax_b) = match fixed_axis {
            0 => (1usize, 2usize),
            1 => (0, 2),
            _ => (0, 1),
        };
        let origin_a = base[ax_a] - max_r - 0.05;
        let origin_b = base[ax_b] - max_r - 0.05;
        let mut grid_cells = vec![false; res * res];
        let tolerance = cell_size; // slice thickness

        let mut rng = rand::thread_rng();
        let n = self.config.monte_carlo_samples;
        for _ in 0..n {
            let angles = chain.random_config(params, &mut rng);
            if let Ok(pos) = self.fk.solve_position(chain, params, &angles) {
                if !pos[0].is_finite() {
                    continue;
                }
                if (pos[fixed_axis] - offset).abs() > tolerance {
                    continue;
                }
                let ca = ((pos[ax_a] - origin_a) / cell_size).floor() as isize;
                let cb = ((pos[ax_b] - origin_b) / cell_size).floor() as isize;
                if ca >= 0 && cb >= 0 && (ca as usize) < res && (cb as usize) < res {
                    grid_cells[cb as usize * res + ca as usize] = true;
                }
            }
        }

        let normal_axis = match fixed_axis {
            0 => "X".to_string(),
            1 => "Y".to_string(),
            _ => "Z".to_string(),
        };
        WorkspaceSlice {
            origin: [origin_a, origin_b],
            cell_size,
            dims: [res, res],
            grid: grid_cells,
            slice_offset: offset,
            normal_axis,
        }
    }

    // -- Directional reach --

    /// Maximum reach distance from the shoulder in a given unit direction.
    pub fn max_reach_in_direction(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        direction: [f64; 3],
    ) -> f64 {
        let base = chain.base_position(params);
        let len = (direction[0] * direction[0]
            + direction[1] * direction[1]
            + direction[2] * direction[2])
            .sqrt();
        if len < 1e-12 {
            return 0.0;
        }
        let dir = [direction[0] / len, direction[1] / len, direction[2] / len];

        let mut best = 0.0f64;
        let mut rng = rand::thread_rng();
        let n = self.config.monte_carlo_samples / 4;
        for _ in 0..n {
            let angles = chain.random_config(params, &mut rng);
            if let Ok(pos) = self.fk.solve_position(chain, params, &angles) {
                let d = [pos[0] - base[0], pos[1] - base[1], pos[2] - base[2]];
                let proj = d[0] * dir[0] + d[1] * dir[1] + d[2] * dir[2];
                // Check that the point is roughly aligned with the direction
                let dist = point_distance(&pos, &base);
                if dist > 1e-6 {
                    let cos_angle = proj / dist;
                    if cos_angle > 0.9 {
                        best = best.max(proj);
                    }
                }
            }
        }
        best
    }

    // -- Comparison --

    /// Compare workspace properties between two body parameterizations.
    pub fn compare(
        &self,
        chain: &KinematicChain,
        params_a: &BodyParameters,
        params_b: &BodyParameters,
    ) -> VerifierResult<WorkspaceComparison> {
        let props_a = self.analyze(chain, params_a)?;
        let props_b = self.analyze(chain, params_b)?;

        let volume_ratio = if props_a.volume > 1e-12 {
            props_b.volume / props_a.volume
        } else {
            0.0
        };
        let centroid_displacement = point_distance(&props_a.centroid, &props_b.centroid);

        // Estimate overlap by checking random points of A against B's reach
        let max_r_b = ForwardKinematicsSolver::max_reach(chain, params_b);
        let base_b = chain.base_position(params_b);
        let mut rng = rand::thread_rng();
        let mut in_both = 0usize;
        let n = (self.config.monte_carlo_samples / 4).max(2000);
        let mut total_a = 0usize;
        for _ in 0..n {
            let angles = chain.random_config(params_a, &mut rng);
            if let Ok(pos) = self.fk.solve_position(chain, params_a, &angles) {
                if !pos[0].is_finite() {
                    continue;
                }
                total_a += 1;
                let dist_b = point_distance(&pos, &base_b);
                if dist_b <= max_r_b {
                    in_both += 1;
                }
            }
        }
        let overlap_fraction = if total_a > 0 {
            in_both as f64 / total_a as f64
        } else {
            0.0
        };

        Ok(WorkspaceComparison {
            volume_a: props_a.volume,
            volume_b: props_b.volume,
            volume_ratio,
            max_reach_diff: props_b.max_reach - props_a.max_reach,
            centroid_displacement,
            overlap_fraction,
        })
    }

    // -- Volume distribution --

    /// Compute how workspace volume is distributed across heights.
    pub fn volume_distribution(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
    ) -> Vec<(f64, f64)> {
        let base = chain.base_position(params);
        let max_r = ForwardKinematicsSolver::max_reach(chain, params);
        let n_slices = self.config.distribution_slices;
        let y_lo = base[1] - max_r;
        let y_hi = base[1] + max_r;
        let step = (y_hi - y_lo) / n_slices as f64;

        let mut bins = vec![0usize; n_slices];
        let mut rng = rand::thread_rng();
        let n = self.config.monte_carlo_samples;
        for _ in 0..n {
            let angles = chain.random_config(params, &mut rng);
            if let Ok(pos) = self.fk.solve_position(chain, params, &angles) {
                if !pos[1].is_finite() {
                    continue;
                }
                let idx = ((pos[1] - y_lo) / step).floor() as isize;
                if idx >= 0 && (idx as usize) < n_slices {
                    bins[idx as usize] += 1;
                }
            }
        }

        let total: usize = bins.iter().sum();
        let frac = if total > 0 { 1.0 / total as f64 } else { 0.0 };
        (0..n_slices)
            .map(|i| {
                let height = y_lo + (i as f64 + 0.5) * step;
                (height, bins[i] as f64 * frac)
            })
            .collect()
    }

    /// Find the optimal working height (height with maximum workspace area).
    pub fn optimal_working_height(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
    ) -> f64 {
        let dist = self.volume_distribution(chain, params);
        dist.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(h, _)| *h)
            .unwrap_or(chain.base_position(params)[1])
    }

    /// Estimate the volume of the dexterous workspace directly.
    pub fn dexterous_workspace(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
    ) -> VerifierResult<f64> {
        let props = self.analyze(chain, params)?;
        Ok(props.dexterous_volume)
    }

    /// Compute workspace volume estimate via Monte Carlo with spatial hashing.
    pub fn estimate_volume(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
    ) -> VerifierResult<f64> {
        let props = self.analyze(chain, params)?;
        Ok(props.volume)
    }

    /// Compute maximum reach at a given height.
    pub fn max_reach_at_height(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        height: f64,
    ) -> f64 {
        let shoulder = chain.base_position(params);
        let dy = height - shoulder[1];
        let max_r = ForwardKinematicsSolver::max_reach(chain, params);
        if dy.abs() > max_r {
            0.0
        } else {
            (max_r * max_r - dy * dy).sqrt()
        }
    }

    /// Check if a specific point is in the workspace (spherical approximation).
    pub fn is_in_workspace(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        point: [f64; 3],
    ) -> bool {
        let shoulder = chain.base_position(params);
        let dist = point_distance(&point, &shoulder);
        let max_reach = ForwardKinematicsSolver::max_reach(chain, params);
        dist <= max_reach
    }

    /// Compute the cross-section boundary at a given height (circle approx).
    pub fn cross_section_at_height(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        height: f64,
        n_points: usize,
    ) -> Vec<[f64; 2]> {
        let r = self.max_reach_at_height(chain, params, height);
        if r < 1e-6 {
            return Vec::new();
        }
        (0..n_points)
            .map(|i| {
                let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n_points as f64);
                [r * angle.cos(), r * angle.sin()]
            })
            .collect()
    }

    // -- internal helpers --

    fn reach_profile(
        &self,
        chain: &KinematicChain,
        _params: &BodyParameters,
        grid: &VoxelGrid,
        base: &[f64; 3],
        max_r: f64,
    ) -> Vec<(f64, f64)> {
        let n_slices = self.config.distribution_slices;
        let y_lo = base[1] - max_r;
        let y_hi = base[1] + max_r;
        let step = (y_hi - y_lo) / n_slices as f64;

        (0..n_slices)
            .map(|i| {
                let height = y_lo + (i as f64 + 0.5) * step;
                let mut max_horiz = 0.0f64;
                // Scan occupied voxels in this height band
                for iy in 0..grid.dims[1] {
                    let vy = grid.origin[1] + (iy as f64 + 0.5) * grid.voxel_size;
                    if (vy - height).abs() > step * 0.5 {
                        continue;
                    }
                    for ix in 0..grid.dims[0] {
                        for iz in 0..grid.dims[2] {
                            let fi = ix * grid.dims[1] * grid.dims[2]
                                + iy * grid.dims[2]
                                + iz;
                            if grid.occupied[fi] {
                                let pt = grid.index_to_point([ix, iy, iz]);
                                let dx = pt[0] - base[0];
                                let dz = pt[2] - base[2];
                                let horiz = (dx * dx + dz * dz).sqrt();
                                max_horiz = max_horiz.max(horiz);
                            }
                        }
                    }
                }
                (height, max_horiz)
            })
            .collect()
    }
}

impl Default for WorkspaceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// 2D convex hull utilities
// ---------------------------------------------------------------------------

fn convex_hull_2d(points: &[[f64; 2]]) -> Vec<[f64; 2]> {
    if points.len() < 3 {
        return points.to_vec();
    }
    let mut pts = points.to_vec();
    pts.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap().then(a[1].partial_cmp(&b[1]).unwrap()));
    let mut hull: Vec<[f64; 2]> = Vec::new();

    // Lower hull
    for &p in &pts {
        while hull.len() >= 2 && cross_2d(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    // Upper hull
    let lower_len = hull.len() + 1;
    for &p in pts.iter().rev() {
        while hull.len() >= lower_len
            && cross_2d(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0
        {
            hull.pop();
        }
        hull.push(p);
    }
    hull.pop(); // remove duplicate last point
    hull
}

fn cross_2d(o: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

fn shoelace_area(hull: &[[f64; 2]]) -> f64 {
    let n = hull.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += hull[i][0] * hull[j][1];
        area -= hull[j][0] * hull[i][1];
    }
    area.abs() * 0.5
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use xr_types::kinematic::ArmSide;

    fn test_chain_and_params() -> (KinematicChain, BodyParameters) {
        (
            KinematicChain::default_arm(ArmSide::Right),
            BodyParameters::average_male(),
        )
    }

    #[test]
    fn test_analyzer_creation() {
        let _a = WorkspaceAnalyzer::new();
    }

    #[test]
    fn test_analyze_produces_positive_volume() {
        let (chain, params) = test_chain_and_params();
        let analyzer = WorkspaceAnalyzer::new().with_workspace_config(WorkspaceConfig {
            monte_carlo_samples: 5_000,
            voxel_size: 0.04,
            ..Default::default()
        });
        let props = analyzer.analyze(&chain, &params).unwrap();
        assert!(props.volume > 0.0, "Volume should be positive");
        assert!(props.max_reach > props.min_reach);
        assert!(props.compactness > 0.0 && props.compactness <= 1.0);
    }

    #[test]
    fn test_horizontal_slice_area() {
        let (chain, params) = test_chain_and_params();
        let analyzer = WorkspaceAnalyzer::new().with_workspace_config(WorkspaceConfig {
            monte_carlo_samples: 5_000,
            slice_resolution: 32,
            ..Default::default()
        });
        let shoulder = chain.base_position(&params);
        let slice = analyzer.horizontal_slice(&chain, &params, shoulder[1]);
        let area = slice.compute_area();
        assert!(area > 0.0, "Slice at shoulder height should have area");
    }

    #[test]
    fn test_max_reach_in_direction() {
        let (chain, params) = test_chain_and_params();
        let analyzer = WorkspaceAnalyzer::new().with_workspace_config(WorkspaceConfig {
            monte_carlo_samples: 3_000,
            ..Default::default()
        });
        let reach_fwd = analyzer.max_reach_in_direction(&chain, &params, [0.0, 0.0, -1.0]);
        assert!(reach_fwd > 0.0, "Should have forward reach");
    }

    #[test]
    fn test_compare_small_vs_large() {
        let chain = KinematicChain::default_arm(ArmSide::Right);
        let small = BodyParameters::small_female();
        let large = BodyParameters::large_male();
        let analyzer = WorkspaceAnalyzer::new().with_workspace_config(WorkspaceConfig {
            monte_carlo_samples: 3_000,
            voxel_size: 0.05,
            ..Default::default()
        });
        let cmp = analyzer.compare(&chain, &small, &large).unwrap();
        assert!(cmp.volume_b >= cmp.volume_a, "Large body should have >= volume");
        assert!(cmp.max_reach_diff >= 0.0);
    }

    #[test]
    fn test_volume_distribution() {
        let (chain, params) = test_chain_and_params();
        let analyzer = WorkspaceAnalyzer::new().with_workspace_config(WorkspaceConfig {
            monte_carlo_samples: 3_000,
            distribution_slices: 10,
            ..Default::default()
        });
        let dist = analyzer.volume_distribution(&chain, &params);
        assert_eq!(dist.len(), 10);
        let total_frac: f64 = dist.iter().map(|(_, f)| f).sum();
        assert!((total_frac - 1.0).abs() < 0.05, "Fractions should sum to ~1");
    }

    #[test]
    fn test_optimal_working_height() {
        let (chain, params) = test_chain_and_params();
        let analyzer = WorkspaceAnalyzer::new().with_workspace_config(WorkspaceConfig {
            monte_carlo_samples: 3_000,
            distribution_slices: 10,
            ..Default::default()
        });
        let h = analyzer.optimal_working_height(&chain, &params);
        let shoulder = chain.base_position(&params);
        let max_r = ForwardKinematicsSolver::max_reach(&chain, &params);
        assert!(h >= shoulder[1] - max_r && h <= shoulder[1] + max_r);
    }

    #[test]
    fn test_convex_hull_2d() {
        let pts = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];
        let hull = convex_hull_2d(&pts);
        assert!(hull.len() >= 4);
        let area = shoelace_area(&hull);
        assert!((area - 1.0).abs() < 0.01, "Unit square hull area ≈ 1.0");
    }

    #[test]
    fn test_workspace_slice_contains() {
        let (chain, params) = test_chain_and_params();
        let analyzer = WorkspaceAnalyzer::new().with_workspace_config(WorkspaceConfig {
            monte_carlo_samples: 5_000,
            slice_resolution: 32,
            ..Default::default()
        });
        let shoulder = chain.base_position(&params);
        let slice = analyzer.horizontal_slice(&chain, &params, shoulder[1]);
        // The shoulder itself should be somewhere inside
        assert!(
            slice.compute_area() > 0.0,
            "Slice should have some occupied cells"
        );
    }

    #[test]
    fn test_is_in_workspace() {
        let (chain, params) = test_chain_and_params();
        let analyzer = WorkspaceAnalyzer::new();
        let shoulder = chain.base_position(&params);
        assert!(analyzer.is_in_workspace(&chain, &params, shoulder));
        assert!(!analyzer.is_in_workspace(&chain, &params, [100.0, 100.0, 100.0]));
    }
}

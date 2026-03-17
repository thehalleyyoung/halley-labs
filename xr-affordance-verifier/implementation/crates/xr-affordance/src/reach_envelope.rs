//! Reach envelope computation: W(θ) = {FK(θ,q) | q ∈ J(θ)}.
//!
//! Computes the reachable workspace as a voxel grid or point cloud
//! using Monte Carlo estimation and systematic grid sampling.

use serde::{Deserialize, Serialize};
use xr_types::kinematic::{KinematicChain, BodyParameters};
use xr_types::geometry::{BoundingBox, Sphere, point_distance};
use xr_types::error::{VerifierError, VerifierResult};

use crate::forward_kinematics::ForwardKinematicsSolver;

/// Configuration for reach envelope computation.
#[derive(Debug, Clone)]
pub struct ReachEnvelopeConfig {
    /// Voxel grid resolution (size of each voxel in meters).
    pub voxel_size: f64,
    /// Number of Monte Carlo samples for estimation.
    pub monte_carlo_samples: usize,
    /// Grid sampling resolution per joint (for systematic sampling).
    pub grid_samples_per_joint: usize,
    /// Whether to use parallel computation.
    pub use_parallel: bool,
    /// Expansion margin for the bounding box.
    pub margin: f64,
}

impl Default for ReachEnvelopeConfig {
    fn default() -> Self {
        Self {
            voxel_size: 0.02,
            monte_carlo_samples: 50_000,
            grid_samples_per_joint: 8,
            use_parallel: true,
            margin: 0.05,
        }
    }
}

/// A voxel grid representing the reachable workspace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoxelGrid {
    /// Origin of the grid (min corner).
    pub origin: [f64; 3],
    /// Size of each voxel.
    pub voxel_size: f64,
    /// Number of voxels in each dimension.
    pub dims: [usize; 3],
    /// Occupancy grid (flattened 3D array, true = reachable).
    pub occupied: Vec<bool>,
    /// Hit count per voxel (for density estimation).
    pub hit_count: Vec<u32>,
}

impl VoxelGrid {
    /// Create a new empty voxel grid.
    pub fn new(origin: [f64; 3], voxel_size: f64, dims: [usize; 3]) -> Self {
        let total = dims[0] * dims[1] * dims[2];
        Self {
            origin,
            voxel_size,
            dims,
            occupied: vec![false; total],
            hit_count: vec![0; total],
        }
    }

    /// Create a voxel grid from a bounding box.
    pub fn from_bounds(bounds: &BoundingBox, voxel_size: f64) -> Self {
        let extents = bounds.extents();
        let dims = [
            ((extents[0] / voxel_size).ceil() as usize).max(1),
            ((extents[1] / voxel_size).ceil() as usize).max(1),
            ((extents[2] / voxel_size).ceil() as usize).max(1),
        ];
        Self::new(bounds.min, voxel_size, dims)
    }

    /// Convert a world-space point to grid indices.
    pub fn point_to_index(&self, point: &[f64; 3]) -> Option<[usize; 3]> {
        let ix = ((point[0] - self.origin[0]) / self.voxel_size).floor() as isize;
        let iy = ((point[1] - self.origin[1]) / self.voxel_size).floor() as isize;
        let iz = ((point[2] - self.origin[2]) / self.voxel_size).floor() as isize;
        if ix >= 0
            && iy >= 0
            && iz >= 0
            && (ix as usize) < self.dims[0]
            && (iy as usize) < self.dims[1]
            && (iz as usize) < self.dims[2]
        {
            Some([ix as usize, iy as usize, iz as usize])
        } else {
            None
        }
    }

    /// Convert grid indices to a flat array index.
    fn flat_index(&self, idx: [usize; 3]) -> usize {
        idx[0] * self.dims[1] * self.dims[2] + idx[1] * self.dims[2] + idx[2]
    }

    /// Convert grid indices to world-space center point.
    pub fn index_to_point(&self, idx: [usize; 3]) -> [f64; 3] {
        [
            self.origin[0] + (idx[0] as f64 + 0.5) * self.voxel_size,
            self.origin[1] + (idx[1] as f64 + 0.5) * self.voxel_size,
            self.origin[2] + (idx[2] as f64 + 0.5) * self.voxel_size,
        ]
    }

    /// Mark a point as occupied.
    pub fn mark_point(&mut self, point: &[f64; 3]) -> bool {
        if let Some(idx) = self.point_to_index(point) {
            let fi = self.flat_index(idx);
            self.occupied[fi] = true;
            self.hit_count[fi] = self.hit_count[fi].saturating_add(1);
            true
        } else {
            false
        }
    }

    /// Check if a point's voxel is occupied.
    pub fn is_occupied(&self, point: &[f64; 3]) -> bool {
        if let Some(idx) = self.point_to_index(point) {
            self.occupied[self.flat_index(idx)]
        } else {
            false
        }
    }

    /// Get the hit count for a point's voxel.
    pub fn get_hit_count(&self, point: &[f64; 3]) -> u32 {
        if let Some(idx) = self.point_to_index(point) {
            self.hit_count[self.flat_index(idx)]
        } else {
            0
        }
    }

    /// Count occupied voxels.
    pub fn occupied_count(&self) -> usize {
        self.occupied.iter().filter(|&&b| b).count()
    }

    /// Total number of voxels.
    pub fn total_voxels(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    /// Compute the volume of the occupied region.
    pub fn occupied_volume(&self) -> f64 {
        let voxel_vol = self.voxel_size.powi(3);
        self.occupied_count() as f64 * voxel_vol
    }

    /// Compute the fill ratio.
    pub fn fill_ratio(&self) -> f64 {
        let total = self.total_voxels();
        if total == 0 {
            return 0.0;
        }
        self.occupied_count() as f64 / total as f64
    }

    /// Get all occupied voxel centers.
    pub fn occupied_points(&self) -> Vec<[f64; 3]> {
        let mut points = Vec::new();
        for ix in 0..self.dims[0] {
            for iy in 0..self.dims[1] {
                for iz in 0..self.dims[2] {
                    let fi = self.flat_index([ix, iy, iz]);
                    if self.occupied[fi] {
                        points.push(self.index_to_point([ix, iy, iz]));
                    }
                }
            }
        }
        points
    }

    /// Compute the signed distance from a point to the nearest occupied voxel.
    /// Positive = outside, negative = inside.
    pub fn signed_distance(&self, point: &[f64; 3]) -> f64 {
        let is_inside = self.is_occupied(point);
        let mut min_dist = f64::INFINITY;

        // Search in a neighborhood for the closest boundary
        let search_radius = (10.0 * self.voxel_size) as usize + 1;
        let center_idx = self.point_to_index(point);

        if let Some(ci) = center_idx {
            for dx in 0..=search_radius * 2 {
                let ix = ci[0] as isize + dx as isize - search_radius as isize;
                if ix < 0 || ix as usize >= self.dims[0] {
                    continue;
                }
                for dy in 0..=search_radius * 2 {
                    let iy = ci[1] as isize + dy as isize - search_radius as isize;
                    if iy < 0 || iy as usize >= self.dims[1] {
                        continue;
                    }
                    for dz in 0..=search_radius * 2 {
                        let iz = ci[2] as isize + dz as isize - search_radius as isize;
                        if iz < 0 || iz as usize >= self.dims[2] {
                            continue;
                        }
                        let idx = [ix as usize, iy as usize, iz as usize];
                        let fi = self.flat_index(idx);
                        let voxel_occupied = self.occupied[fi];

                        // We want boundary voxels (occupied next to unoccupied or vice versa)
                        if voxel_occupied != is_inside {
                            let voxel_center = self.index_to_point(idx);
                            let dist = point_distance(point, &voxel_center);
                            min_dist = min_dist.min(dist);
                        }
                    }
                }
            }
        } else {
            // Point is outside the grid entirely
            let occupied_pts = self.occupied_points();
            for p in &occupied_pts {
                let dist = point_distance(point, p);
                min_dist = min_dist.min(dist);
            }
        }

        if min_dist == f64::INFINITY {
            if is_inside { -self.voxel_size } else { self.voxel_size * 10.0 }
        } else if is_inside {
            -min_dist
        } else {
            min_dist
        }
    }

    /// Dilate the occupied region by one voxel (morphological dilation).
    pub fn dilate(&mut self) {
        let old_occupied = self.occupied.clone();
        let offsets: Vec<[isize; 3]> = vec![
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1],
        ];

        for ix in 0..self.dims[0] {
            for iy in 0..self.dims[1] {
                for iz in 0..self.dims[2] {
                    let fi = self.flat_index([ix, iy, iz]);
                    if old_occupied[fi] {
                        continue;
                    }
                    for off in &offsets {
                        let nx = ix as isize + off[0];
                        let ny = iy as isize + off[1];
                        let nz = iz as isize + off[2];
                        if nx >= 0
                            && ny >= 0
                            && nz >= 0
                            && (nx as usize) < self.dims[0]
                            && (ny as usize) < self.dims[1]
                            && (nz as usize) < self.dims[2]
                        {
                            let nfi = self.flat_index([nx as usize, ny as usize, nz as usize]);
                            if old_occupied[nfi] {
                                self.occupied[fi] = true;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Erode the occupied region by one voxel (morphological erosion).
    pub fn erode(&mut self) {
        let old_occupied = self.occupied.clone();
        let offsets: Vec<[isize; 3]> = vec![
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1],
        ];

        for ix in 0..self.dims[0] {
            for iy in 0..self.dims[1] {
                for iz in 0..self.dims[2] {
                    let fi = self.flat_index([ix, iy, iz]);
                    if !old_occupied[fi] {
                        continue;
                    }
                    for off in &offsets {
                        let nx = ix as isize + off[0];
                        let ny = iy as isize + off[1];
                        let nz = iz as isize + off[2];
                        if nx < 0
                            || ny < 0
                            || nz < 0
                            || (nx as usize) >= self.dims[0]
                            || (ny as usize) >= self.dims[1]
                            || (nz as usize) >= self.dims[2]
                        {
                            self.occupied[fi] = false;
                            break;
                        }
                        let nfi = self.flat_index([nx as usize, ny as usize, nz as usize]);
                        if !old_occupied[nfi] {
                            self.occupied[fi] = false;
                            break;
                        }
                    }
                }
            }
        }
    }
}

/// Reach envelope representing the reachable workspace for a given body parameterization.
#[derive(Debug, Clone)]
pub struct ReachEnvelope {
    /// The voxel grid representation.
    pub grid: VoxelGrid,
    /// Point cloud of sampled reachable positions.
    pub sample_points: Vec<[f64; 3]>,
    /// Bounding box of the reachable region.
    pub bounds: BoundingBox,
    /// Base position (shoulder).
    pub base_position: [f64; 3],
    /// Body parameters used to compute this envelope.
    pub body_params: BodyParameters,
    /// Number of samples used.
    pub num_samples: usize,
    /// Maximum reach distance observed.
    pub max_reach_distance: f64,
    /// Minimum reach distance observed (closest reachable point to base).
    pub min_reach_distance: f64,
}

impl ReachEnvelope {
    /// Compute the reach envelope using Monte Carlo sampling.
    pub fn compute(
        chain: &KinematicChain,
        params: &BodyParameters,
        config: &ReachEnvelopeConfig,
    ) -> VerifierResult<Self> {
        let fk = ForwardKinematicsSolver::new();
        let base_pos = chain.base_position(params);
        let max_reach = ForwardKinematicsSolver::max_reach(chain, params);

        // Estimate bounding box
        let margin = config.margin;
        let bounds = BoundingBox::new(
            [
                base_pos[0] - max_reach - margin,
                base_pos[1] - max_reach - margin,
                base_pos[2] - max_reach - margin,
            ],
            [
                base_pos[0] + max_reach + margin,
                base_pos[1] + max_reach + margin,
                base_pos[2] + max_reach + margin,
            ],
        );

        let mut grid = VoxelGrid::from_bounds(&bounds, config.voxel_size);
        let mut sample_points = Vec::with_capacity(config.monte_carlo_samples);
        let mut max_dist = 0.0f64;
        let mut min_dist = f64::INFINITY;
        let mut rng = rand::thread_rng();

        // Monte Carlo sampling over joint space
        for _ in 0..config.monte_carlo_samples {
            let angles = chain.random_config(params, &mut rng);
            if let Ok(pos) = fk.solve_position(chain, params, &angles) {
                if pos[0].is_finite() && pos[1].is_finite() && pos[2].is_finite() {
                    grid.mark_point(&pos);
                    sample_points.push(pos);

                    let dist = point_distance(&pos, &base_pos);
                    max_dist = max_dist.max(dist);
                    if dist > 0.001 {
                        min_dist = min_dist.min(dist);
                    }
                }
            }
        }

        if min_dist == f64::INFINITY {
            min_dist = 0.0;
        }

        // Tighten bounding box to actual samples
        let actual_bounds = if let Some(bb) = BoundingBox::from_points(&sample_points) {
            bb.expand(margin)
        } else {
            bounds
        };

        Ok(Self {
            grid,
            sample_points,
            bounds: actual_bounds,
            base_position: base_pos,
            body_params: *params,
            num_samples: config.monte_carlo_samples,
            max_reach_distance: max_dist,
            min_reach_distance: min_dist,
        })
    }

    /// Compute with systematic grid sampling over joint space.
    pub fn compute_grid(
        chain: &KinematicChain,
        params: &BodyParameters,
        config: &ReachEnvelopeConfig,
    ) -> VerifierResult<Self> {
        let fk = ForwardKinematicsSolver::new();
        let base_pos = chain.base_position(params);
        let max_reach = ForwardKinematicsSolver::max_reach(chain, params);

        let margin = config.margin;
        let bounds = BoundingBox::new(
            [
                base_pos[0] - max_reach - margin,
                base_pos[1] - max_reach - margin,
                base_pos[2] - max_reach - margin,
            ],
            [
                base_pos[0] + max_reach + margin,
                base_pos[1] + max_reach + margin,
                base_pos[2] + max_reach + margin,
            ],
        );

        let mut grid = VoxelGrid::from_bounds(&bounds, config.voxel_size);
        let mut sample_points = Vec::new();
        let mut max_dist = 0.0f64;
        let mut min_dist = f64::INFINITY;

        let n_joints = chain.joints.len();
        let samples_per_joint = config.grid_samples_per_joint;

        // Build the grid of joint angles
        let joint_samples: Vec<Vec<f64>> = chain
            .joints
            .iter()
            .map(|joint| {
                let limits = joint.effective_limits(params);
                (0..samples_per_joint)
                    .map(|k| {
                        let t = k as f64 / (samples_per_joint as f64 - 1.0).max(1.0);
                        limits.from_normalized(t)
                    })
                    .collect()
            })
            .collect();

        // Iterate over all combinations (limited by total to avoid exponential blowup)
        let total_combos: usize = joint_samples.iter().map(|s| s.len()).product();
        let max_combos = 500_000;
        let step = if total_combos > max_combos {
            total_combos / max_combos
        } else {
            1
        };

        let mut combo_idx = 0usize;
        let mut num_evaluated = 0usize;
        let mut angles = vec![0.0; n_joints];

        Self::enumerate_grid_recursive(
            &joint_samples,
            &mut angles,
            0,
            &mut combo_idx,
            step,
            &fk,
            chain,
            params,
            &base_pos,
            &mut grid,
            &mut sample_points,
            &mut max_dist,
            &mut min_dist,
            &mut num_evaluated,
            max_combos,
        );

        if min_dist == f64::INFINITY {
            min_dist = 0.0;
        }

        let actual_bounds = if let Some(bb) = BoundingBox::from_points(&sample_points) {
            bb.expand(margin)
        } else {
            bounds
        };

        Ok(Self {
            grid,
            sample_points,
            bounds: actual_bounds,
            base_position: base_pos,
            body_params: *params,
            num_samples: num_evaluated,
            max_reach_distance: max_dist,
            min_reach_distance: min_dist,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn enumerate_grid_recursive(
        joint_samples: &[Vec<f64>],
        angles: &mut Vec<f64>,
        depth: usize,
        combo_idx: &mut usize,
        step: usize,
        fk: &ForwardKinematicsSolver,
        chain: &KinematicChain,
        params: &BodyParameters,
        base_pos: &[f64; 3],
        grid: &mut VoxelGrid,
        sample_points: &mut Vec<[f64; 3]>,
        max_dist: &mut f64,
        min_dist: &mut f64,
        num_evaluated: &mut usize,
        max_combos: usize,
    ) {
        if *num_evaluated >= max_combos {
            return;
        }
        if depth == joint_samples.len() {
            if *combo_idx % step == 0 {
                if let Ok(pos) = fk.solve_position(chain, params, angles) {
                    if pos[0].is_finite() && pos[1].is_finite() && pos[2].is_finite() {
                        grid.mark_point(&pos);
                        sample_points.push(pos);
                        let dist = point_distance(&pos, base_pos);
                        *max_dist = max_dist.max(dist);
                        if dist > 0.001 {
                            *min_dist = min_dist.min(dist);
                        }
                        *num_evaluated += 1;
                    }
                }
            }
            *combo_idx += 1;
            return;
        }

        for &angle in &joint_samples[depth] {
            if *num_evaluated >= max_combos {
                return;
            }
            angles[depth] = angle;
            Self::enumerate_grid_recursive(
                joint_samples, angles, depth + 1, combo_idx, step,
                fk, chain, params, base_pos, grid, sample_points,
                max_dist, min_dist, num_evaluated, max_combos,
            );
        }
    }

    /// Check if a point is within the reach envelope.
    pub fn contains_point(&self, point: &[f64; 3]) -> bool {
        self.grid.is_occupied(point)
    }

    /// Compute the volume of the reach envelope.
    pub fn volume(&self) -> f64 {
        self.grid.occupied_volume()
    }

    /// Compute a bounding sphere for the reach envelope.
    pub fn bounding_sphere(&self) -> Sphere {
        Sphere::new(self.base_position, self.max_reach_distance)
    }

    /// Compute the signed distance from a point to the envelope boundary.
    pub fn signed_distance(&self, point: &[f64; 3]) -> f64 {
        self.grid.signed_distance(point)
    }

    /// Get the density at a point (normalized hit count).
    pub fn density_at(&self, point: &[f64; 3]) -> f64 {
        let hits = self.grid.get_hit_count(point) as f64;
        if self.num_samples == 0 {
            return 0.0;
        }
        hits / self.num_samples as f64
    }

    /// Compute the reach envelope intersection with a bounding box.
    pub fn intersect_with_box(&self, bbox: &BoundingBox) -> Vec<[f64; 3]> {
        self.sample_points
            .iter()
            .filter(|p| bbox.contains_point(p))
            .copied()
            .collect()
    }

    /// Compute the reach fraction (what fraction of the bounding sphere is reachable).
    pub fn reach_fraction(&self) -> f64 {
        let sphere_vol = self.bounding_sphere().volume();
        if sphere_vol < 1e-12 {
            return 0.0;
        }
        self.volume() / sphere_vol
    }

    /// Get a 2D slice at a given height (Y coordinate).
    pub fn slice_at_height(&self, height: f64, tolerance: f64) -> Vec<[f64; 2]> {
        self.sample_points
            .iter()
            .filter(|p| (p[1] - height).abs() <= tolerance)
            .map(|p| [p[0], p[2]])
            .collect()
    }

    /// Get a 2D slice at a given depth (Z coordinate).
    pub fn slice_at_depth(&self, depth: f64, tolerance: f64) -> Vec<[f64; 2]> {
        self.sample_points
            .iter()
            .filter(|p| (p[2] - depth).abs() <= tolerance)
            .map(|p| [p[0], p[1]])
            .collect()
    }

    /// Compute the centroid of the reachable region.
    pub fn centroid(&self) -> [f64; 3] {
        if self.sample_points.is_empty() {
            return self.base_position;
        }
        let n = self.sample_points.len() as f64;
        let sum = self.sample_points.iter().fold([0.0; 3], |acc, p| {
            [acc[0] + p[0], acc[1] + p[1], acc[2] + p[2]]
        });
        [sum[0] / n, sum[1] / n, sum[2] / n]
    }

    /// Compute the reach distance in a specific direction from the base.
    pub fn reach_in_direction(&self, direction: &[f64; 3]) -> f64 {
        let dir_len = (direction[0] * direction[0]
            + direction[1] * direction[1]
            + direction[2] * direction[2])
            .sqrt();
        if dir_len < 1e-12 {
            return 0.0;
        }
        let norm_dir = [
            direction[0] / dir_len,
            direction[1] / dir_len,
            direction[2] / dir_len,
        ];

        let mut max_proj = 0.0f64;
        for p in &self.sample_points {
            let dx = p[0] - self.base_position[0];
            let dy = p[1] - self.base_position[1];
            let dz = p[2] - self.base_position[2];
            let proj = dx * norm_dir[0] + dy * norm_dir[1] + dz * norm_dir[2];
            max_proj = max_proj.max(proj);
        }
        max_proj
    }

    /// Compute the intersection of two reach envelopes.
    /// Returns the set of points that are reachable by both.
    pub fn intersection(&self, other: &ReachEnvelope) -> Vec<[f64; 3]> {
        self.sample_points
            .iter()
            .filter(|p| other.contains_point(p))
            .copied()
            .collect()
    }

    /// Compute the union volume estimate of two reach envelopes.
    pub fn union_volume(&self, other: &ReachEnvelope) -> f64 {
        let v1 = self.volume();
        let v2 = other.volume();
        let intersection_count = self
            .sample_points
            .iter()
            .filter(|p| other.contains_point(p))
            .count();
        let intersection_vol = if !self.sample_points.is_empty() {
            v1 * (intersection_count as f64 / self.sample_points.len() as f64)
        } else {
            0.0
        };
        v1 + v2 - intersection_vol
    }

    /// Apply morphological closing (dilate then erode) to fill small gaps.
    pub fn close_gaps(&mut self) {
        self.grid.dilate();
        self.grid.erode();
    }

    /// Compute reach statistics per direction (azimuth/elevation histogram).
    pub fn directional_reach_histogram(
        &self,
        azimuth_bins: usize,
        elevation_bins: usize,
    ) -> Vec<Vec<f64>> {
        let mut histogram = vec![vec![0.0f64; elevation_bins]; azimuth_bins];

        for p in &self.sample_points {
            let dx = p[0] - self.base_position[0];
            let dy = p[1] - self.base_position[1];
            let dz = p[2] - self.base_position[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            if dist < 1e-6 {
                continue;
            }

            let azimuth = dz.atan2(dx) + std::f64::consts::PI; // [0, 2π]
            let elevation = (dy / dist).asin() + std::f64::consts::FRAC_PI_2; // [0, π]

            let az_idx = ((azimuth / (2.0 * std::f64::consts::PI)) * azimuth_bins as f64)
                .floor() as usize;
            let el_idx = ((elevation / std::f64::consts::PI) * elevation_bins as f64)
                .floor() as usize;

            let az_idx = az_idx.min(azimuth_bins - 1);
            let el_idx = el_idx.min(elevation_bins - 1);

            if dist > histogram[az_idx][el_idx] {
                histogram[az_idx][el_idx] = dist;
            }
        }

        histogram
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xr_types::kinematic::{ArmSide, BodyParameters, KinematicChain};

    fn test_chain() -> KinematicChain {
        KinematicChain::default_arm(ArmSide::Right)
    }

    fn test_params() -> BodyParameters {
        BodyParameters::average_male()
    }

    fn small_config() -> ReachEnvelopeConfig {
        ReachEnvelopeConfig {
            voxel_size: 0.05,
            monte_carlo_samples: 5000,
            grid_samples_per_joint: 4,
            use_parallel: false,
            margin: 0.05,
        }
    }

    #[test]
    fn test_voxel_grid_creation() {
        let grid = VoxelGrid::new([0.0, 0.0, 0.0], 0.1, [10, 10, 10]);
        assert_eq!(grid.total_voxels(), 1000);
        assert_eq!(grid.occupied_count(), 0);
    }

    #[test]
    fn test_voxel_grid_mark_and_query() {
        let mut grid = VoxelGrid::new([0.0, 0.0, 0.0], 0.1, [10, 10, 10]);
        assert!(!grid.is_occupied(&[0.05, 0.05, 0.05]));
        grid.mark_point(&[0.05, 0.05, 0.05]);
        assert!(grid.is_occupied(&[0.05, 0.05, 0.05]));
        assert_eq!(grid.occupied_count(), 1);
    }

    #[test]
    fn test_voxel_grid_out_of_bounds() {
        let grid = VoxelGrid::new([0.0, 0.0, 0.0], 0.1, [10, 10, 10]);
        assert!(!grid.is_occupied(&[-1.0, -1.0, -1.0]));
        assert!(!grid.is_occupied(&[2.0, 2.0, 2.0]));
    }

    #[test]
    fn test_voxel_grid_volume() {
        let mut grid = VoxelGrid::new([0.0, 0.0, 0.0], 0.1, [10, 10, 10]);
        grid.mark_point(&[0.05, 0.05, 0.05]);
        grid.mark_point(&[0.15, 0.15, 0.15]);
        let vol = grid.occupied_volume();
        assert!((vol - 2.0 * 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_voxel_grid_dilate() {
        let mut grid = VoxelGrid::new([0.0, 0.0, 0.0], 0.1, [10, 10, 10]);
        grid.mark_point(&[0.55, 0.55, 0.55]);
        let before = grid.occupied_count();
        grid.dilate();
        assert!(grid.occupied_count() > before);
    }

    #[test]
    fn test_voxel_grid_erode() {
        let mut grid = VoxelGrid::new([0.0, 0.0, 0.0], 0.1, [5, 5, 5]);
        // Fill a 3x3x3 block
        for x in 1..4 {
            for y in 1..4 {
                for z in 1..4 {
                    let point = [x as f64 * 0.1 + 0.05, y as f64 * 0.1 + 0.05, z as f64 * 0.1 + 0.05];
                    grid.mark_point(&point);
                }
            }
        }
        let before = grid.occupied_count();
        grid.erode();
        assert!(grid.occupied_count() <= before);
    }

    #[test]
    fn test_reach_envelope_compute() {
        let chain = test_chain();
        let params = test_params();
        let config = small_config();

        let envelope = ReachEnvelope::compute(&chain, &params, &config).unwrap();
        assert!(!envelope.sample_points.is_empty());
        assert!(envelope.volume() > 0.0);
        assert!(envelope.max_reach_distance > 0.0);
    }

    #[test]
    fn test_reach_envelope_contains_point() {
        let chain = test_chain();
        let params = test_params();
        let config = small_config();

        let envelope = ReachEnvelope::compute(&chain, &params, &config).unwrap();
        // At least some sample points should be "contained"
        let contained = envelope
            .sample_points
            .iter()
            .filter(|p| envelope.contains_point(p))
            .count();
        assert!(contained > 0);
    }

    #[test]
    fn test_reach_envelope_bounding_sphere() {
        let chain = test_chain();
        let params = test_params();
        let config = small_config();

        let envelope = ReachEnvelope::compute(&chain, &params, &config).unwrap();
        let sphere = envelope.bounding_sphere();
        assert!(sphere.radius > 0.0);
        assert!(sphere.radius <= params.total_reach() + 0.5);
    }

    #[test]
    fn test_reach_envelope_centroid() {
        let chain = test_chain();
        let params = test_params();
        let config = small_config();

        let envelope = ReachEnvelope::compute(&chain, &params, &config).unwrap();
        let centroid = envelope.centroid();
        assert!(centroid[0].is_finite());
        assert!(centroid[1].is_finite());
        assert!(centroid[2].is_finite());
    }

    #[test]
    fn test_reach_envelope_directional_reach() {
        let chain = test_chain();
        let params = test_params();
        let config = small_config();

        let envelope = ReachEnvelope::compute(&chain, &params, &config).unwrap();
        let forward_reach = envelope.reach_in_direction(&[0.0, 0.0, 1.0]);
        assert!(forward_reach >= 0.0);
    }

    #[test]
    fn test_reach_envelope_slice() {
        let chain = test_chain();
        let params = test_params();
        let config = small_config();

        let envelope = ReachEnvelope::compute(&chain, &params, &config).unwrap();
        let base = chain.base_position(&params);
        let slice = envelope.slice_at_height(base[1], 0.1);
        assert!(!slice.is_empty());
    }

    #[test]
    fn test_reach_envelope_grid_compute() {
        let chain = test_chain();
        let params = test_params();
        let config = ReachEnvelopeConfig {
            voxel_size: 0.05,
            monte_carlo_samples: 1000,
            grid_samples_per_joint: 3,
            use_parallel: false,
            margin: 0.05,
        };

        let envelope = ReachEnvelope::compute_grid(&chain, &params, &config).unwrap();
        assert!(!envelope.sample_points.is_empty());
        assert!(envelope.volume() > 0.0);
    }

    #[test]
    fn test_reach_fraction() {
        let chain = test_chain();
        let params = test_params();
        let config = small_config();

        let envelope = ReachEnvelope::compute(&chain, &params, &config).unwrap();
        let fraction = envelope.reach_fraction();
        assert!(fraction > 0.0);
        assert!(fraction <= 1.0);
    }

    #[test]
    fn test_directional_histogram() {
        let chain = test_chain();
        let params = test_params();
        let config = small_config();

        let envelope = ReachEnvelope::compute(&chain, &params, &config).unwrap();
        let hist = envelope.directional_reach_histogram(8, 4);
        assert_eq!(hist.len(), 8);
        assert_eq!(hist[0].len(), 4);
    }

    #[test]
    fn test_close_gaps() {
        let chain = test_chain();
        let params = test_params();
        let config = small_config();

        let mut envelope = ReachEnvelope::compute(&chain, &params, &config).unwrap();
        let vol_before = envelope.volume();
        envelope.close_gaps();
        let vol_after = envelope.volume();
        // Volume should not decrease after closing
        assert!(vol_after >= vol_before * 0.99);
    }

    #[test]
    fn test_envelope_config_default() {
        let config = ReachEnvelopeConfig::default();
        assert!(config.voxel_size > 0.0);
        assert!(config.monte_carlo_samples > 0);
    }

    #[test]
    fn test_voxel_grid_from_bounds() {
        let bounds = BoundingBox::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        let grid = VoxelGrid::from_bounds(&bounds, 0.1);
        assert!(grid.dims[0] >= 20);
        assert!(grid.dims[1] >= 20);
        assert!(grid.dims[2] >= 20);
    }

    #[test]
    fn test_voxel_grid_occupied_points() {
        let mut grid = VoxelGrid::new([0.0, 0.0, 0.0], 0.1, [5, 5, 5]);
        grid.mark_point(&[0.05, 0.05, 0.05]);
        grid.mark_point(&[0.15, 0.15, 0.15]);
        let points = grid.occupied_points();
        assert_eq!(points.len(), 2);
    }

    #[test]
    fn test_voxel_grid_fill_ratio() {
        let mut grid = VoxelGrid::new([0.0, 0.0, 0.0], 0.1, [10, 10, 10]);
        assert_eq!(grid.fill_ratio(), 0.0);
        grid.mark_point(&[0.05, 0.05, 0.05]);
        assert!(grid.fill_ratio() > 0.0);
        assert!(grid.fill_ratio() < 1.0);
    }
}

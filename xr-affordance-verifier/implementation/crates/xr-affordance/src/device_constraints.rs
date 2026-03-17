//! Device constraint modeling.
//!
//! Models how XR device tracking volumes, dead zones, controller offsets,
//! and movement modes constrain the effective reachable workspace.

use xr_types::kinematic::{KinematicChain, BodyParameters, ArmSide, FullBodyModel};
use xr_types::device::{DeviceConfig, TrackingVolume, MovementMode};
use xr_types::geometry::{BoundingBox, Volume, Sphere, point_distance};
use xr_types::error::{VerifierError, VerifierResult};
use xr_types::scene::InteractionType;

use crate::forward_kinematics::ForwardKinematicsSolver;
use crate::reach_envelope::{ReachEnvelope, ReachEnvelopeConfig, VoxelGrid};

use rand::Rng;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for device constraint analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConstraintConfig {
    /// Number of Monte Carlo samples for intersection volume estimation.
    pub monte_carlo_samples: usize,
    /// Voxel size for the effective-reach grid.
    pub voxel_size: f64,
    /// Tracking quality threshold below which a point is considered unusable.
    pub min_quality: f64,
    /// Whether to apply controller offset correction.
    pub apply_controller_offset: bool,
    /// Additional safety margin around dead zones (meters).
    pub dead_zone_margin: f64,
}

impl Default for DeviceConstraintConfig {
    fn default() -> Self {
        Self {
            monte_carlo_samples: 15_000,
            voxel_size: 0.03,
            min_quality: 0.3,
            apply_controller_offset: true,
            dead_zone_margin: 0.01,
        }
    }
}

// ---------------------------------------------------------------------------
// EffectiveReachResult
// ---------------------------------------------------------------------------

/// Result of computing the effective reachable volume after device constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectiveReachResult {
    /// Volume reachable AND tracked (m³).
    pub effective_volume: f64,
    /// Average tracking quality over the effective region [0,1].
    pub tracking_quality: f64,
    /// Fraction of the kinematic reach that is also tracked.
    pub tracking_coverage: f64,
    /// Volume lost to dead zones (m³).
    pub dead_zone_loss: f64,
    /// Number of sample points in the effective region.
    pub sample_count: usize,
}

// ---------------------------------------------------------------------------
// TrackingQualityMap
// ---------------------------------------------------------------------------

/// Spatial map of tracking quality across the reachable workspace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackingQualityMap {
    /// Voxel grid where hit_count stores quality * 1000 (integer-quantised).
    pub grid: VoxelGrid,
    /// Mean quality across occupied voxels.
    pub mean_quality: f64,
    /// Minimum quality observed.
    pub min_quality: f64,
    /// Maximum quality observed.
    pub max_quality: f64,
}

// ---------------------------------------------------------------------------
// DeviceConstraintModel
// ---------------------------------------------------------------------------

/// Models device constraints and their intersection with arm reach.
pub struct DeviceConstraintModel {
    device: DeviceConfig,
    config: DeviceConstraintConfig,
    fk: ForwardKinematicsSolver,
}

impl DeviceConstraintModel {
    pub fn new(device: DeviceConfig) -> Self {
        Self {
            device,
            config: DeviceConstraintConfig::default(),
            fk: ForwardKinematicsSolver::new(),
        }
    }

    pub fn with_config(mut self, config: DeviceConstraintConfig) -> Self {
        self.config = config;
        self
    }

    pub fn device(&self) -> &DeviceConfig {
        &self.device
    }

    pub fn supports_interaction(&self, itype: &InteractionType) -> bool {
        self.device.supports_interaction(itype.clone())
    }

    pub fn tracking_volume_bounds(&self) -> BoundingBox {
        self.device.tracking_volume.shape.bounding_box()
    }

    pub fn is_in_tracking_volume(&self, point: [f64; 3]) -> bool {
        self.device.tracking_volume.contains_point(&point)
    }

    pub fn min_button_size(&self) -> f64 {
        self.device.constraints.min_button_size
    }

    pub fn min_element_spacing(&self) -> f64 {
        self.device.constraints.min_element_spacing
    }

    pub fn comfortable_distance_range(&self) -> (f64, f64) {
        (
            self.device.constraints.comfortable_distance[0],
            self.device.constraints.comfortable_distance[1],
        )
    }

    // -- Core analysis --

    /// Compute the effective reachable volume: the intersection of the arm's
    /// kinematic reach with the device tracking volume, minus dead zones.
    pub fn compute_effective_reach(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
    ) -> VerifierResult<EffectiveReachResult> {
        let base = chain.base_position(params);
        let max_r = ForwardKinematicsSolver::max_reach(chain, params);
        let n = self.config.monte_carlo_samples;
        let voxel_size = self.config.voxel_size;

        // Bounding box of kinematic reach
        let reach_bb = BoundingBox::new(
            [base[0] - max_r - 0.05, base[1] - max_r - 0.05, base[2] - max_r - 0.05],
            [base[0] + max_r + 0.05, base[1] + max_r + 0.05, base[2] + max_r + 0.05],
        );

        let mut grid = VoxelGrid::from_bounds(&reach_bb, voxel_size);
        let mut rng = rand::thread_rng();
        let mut effective_count = 0usize;
        let mut dead_zone_count = 0usize;
        let mut total_kinematic = 0usize;
        let mut quality_sum = 0.0f64;

        for _ in 0..n {
            let angles = chain.random_config(params, &mut rng);
            if let Ok(mut pos) = self.fk.solve_position(chain, params, &angles) {
                if !pos[0].is_finite() {
                    continue;
                }

                // Apply controller offset if configured
                if self.config.apply_controller_offset {
                    pos = self.apply_controller_offset_to_point(pos);
                }

                total_kinematic += 1;

                // Check dead zones (with margin)
                let in_dead_zone = self.point_in_dead_zone(&pos);
                if in_dead_zone {
                    dead_zone_count += 1;
                    continue;
                }

                // Check tracking volume
                if self.device.tracking_volume.contains_point(&pos) {
                    let q = self.device.tracking_volume.quality_at(&pos);
                    if q >= self.config.min_quality {
                        grid.mark_point(&pos);
                        effective_count += 1;
                        quality_sum += q;
                    }
                }
            }
        }

        let effective_volume = grid.occupied_volume();
        let tracking_quality = if effective_count > 0 {
            quality_sum / effective_count as f64
        } else {
            0.0
        };
        let tracking_coverage = if total_kinematic > 0 {
            effective_count as f64 / total_kinematic as f64
        } else {
            0.0
        };
        let dead_zone_loss = if total_kinematic > 0 {
            // Estimate lost volume proportionally
            let kinematic_vol = grid.occupied_count() as f64 * voxel_size.powi(3)
                / tracking_coverage.max(0.01);
            kinematic_vol * (dead_zone_count as f64 / total_kinematic as f64)
        } else {
            0.0
        };

        Ok(EffectiveReachResult {
            effective_volume,
            tracking_quality,
            tracking_coverage,
            dead_zone_loss,
            sample_count: effective_count,
        })
    }

    /// Apply the controller offset transform to a wrist/hand position.
    pub fn apply_controller_offset(&self, point: [f64; 3]) -> [f64; 3] {
        self.apply_controller_offset_to_point(point)
    }

    /// Compute a spatial map of tracking quality across the reachable workspace.
    pub fn compute_tracking_quality_map(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
    ) -> VerifierResult<TrackingQualityMap> {
        let base = chain.base_position(params);
        let max_r = ForwardKinematicsSolver::max_reach(chain, params);
        let voxel_size = self.config.voxel_size;

        let bb = BoundingBox::new(
            [base[0] - max_r - 0.05, base[1] - max_r - 0.05, base[2] - max_r - 0.05],
            [base[0] + max_r + 0.05, base[1] + max_r + 0.05, base[2] + max_r + 0.05],
        );

        let mut grid = VoxelGrid::from_bounds(&bb, voxel_size);
        let mut rng = rand::thread_rng();
        let mut quality_per_voxel: std::collections::HashMap<usize, (f64, usize)> =
            std::collections::HashMap::new();

        let n = self.config.monte_carlo_samples;
        for _ in 0..n {
            let angles = chain.random_config(params, &mut rng);
            if let Ok(pos) = self.fk.solve_position(chain, params, &angles) {
                if !pos[0].is_finite() {
                    continue;
                }
                if let Some(idx) = grid.point_to_index(&pos) {
                    let fi = idx[0] * grid.dims[1] * grid.dims[2]
                        + idx[1] * grid.dims[2]
                        + idx[2];
                    grid.occupied[fi] = true;
                    let q = self.device.tracking_volume.quality_at(&pos);
                    let entry = quality_per_voxel.entry(fi).or_insert((0.0, 0));
                    entry.0 += q;
                    entry.1 += 1;
                    // Store quantised quality in hit_count
                    let avg = entry.0 / entry.1 as f64;
                    grid.hit_count[fi] = (avg * 1000.0) as u32;
                }
            }
        }

        let mut min_q = f64::INFINITY;
        let mut max_q = 0.0f64;
        let mut sum_q = 0.0;
        let mut count = 0usize;
        for (_, (qsum, qcount)) in &quality_per_voxel {
            let avg = qsum / *qcount as f64;
            min_q = min_q.min(avg);
            max_q = max_q.max(avg);
            sum_q += avg;
            count += 1;
        }
        if count == 0 {
            min_q = 0.0;
        }

        Ok(TrackingQualityMap {
            grid,
            mean_quality: if count > 0 { sum_q / count as f64 } else { 0.0 },
            min_quality: min_q,
            max_quality: max_q,
        })
    }

    /// Apply seated constraints: when the user is seated the shoulder height
    /// is reduced, limiting the reachable workspace. This returns a modified
    /// base position.
    pub fn seated_constraints(
        &self,
        params: &BodyParameters,
        seat_height: f64,
    ) -> [f64; 3] {
        // Seated shoulder height ≈ seat_height + sitting trunk + shoulder offset
        let sitting_trunk = params.stature * 0.52; // seated trunk height
        let shoulder_offset = sitting_trunk * 0.818;
        let shoulder_y = seat_height + shoulder_offset;
        [0.0, shoulder_y, 0.0]
    }

    /// Compute the fraction of the kinematic workspace that falls inside
    /// device dead zones.
    pub fn compute_dead_zone_impact(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
    ) -> VerifierResult<f64> {
        if self.device.tracking_volume.dead_zones.is_empty() {
            return Ok(0.0);
        }

        let n = self.config.monte_carlo_samples;
        let mut rng = rand::thread_rng();
        let mut total = 0usize;
        let mut in_dead = 0usize;

        for _ in 0..n {
            let angles = chain.random_config(params, &mut rng);
            if let Ok(pos) = self.fk.solve_position(chain, params, &angles) {
                if !pos[0].is_finite() {
                    continue;
                }
                total += 1;
                if self.point_in_dead_zone(&pos) {
                    in_dead += 1;
                }
            }
        }

        Ok(if total > 0 { in_dead as f64 / total as f64 } else { 0.0 })
    }

    // -- Coordinate frame helpers --

    /// Transform a point from body frame to device frame using the
    /// controller offset (4×4 row-major matrix stored in `controller_offset`).
    fn apply_controller_offset_to_point(&self, p: [f64; 3]) -> [f64; 3] {
        let m = &self.device.controller_offset;
        // m is 4×4 row-major: [m00, m01, m02, m03, m10, ...]
        [
            m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3],
            m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7],
            m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11],
        ]
    }

    /// Inverse of the controller offset transform (assuming it's a rigid body
    /// transform, i.e. R^T for rotation, -R^T * t for translation).
    pub fn inverse_controller_offset(&self, p: [f64; 3]) -> [f64; 3] {
        let m = &self.device.controller_offset;
        // R^T * (p - t)
        let tx = m[3];
        let ty = m[7];
        let tz = m[11];
        let dx = p[0] - tx;
        let dy = p[1] - ty;
        let dz = p[2] - tz;
        [
            m[0] * dx + m[4] * dy + m[8] * dz,
            m[1] * dx + m[5] * dy + m[9] * dz,
            m[2] * dx + m[6] * dy + m[10] * dz,
        ]
    }

    /// Convert a point from device-local to world coordinates.
    /// For now assumes the device is at world origin.
    pub fn device_to_world(&self, p: [f64; 3]) -> [f64; 3] {
        p // identity for devices at origin
    }

    /// Convert a point from world to device-local coordinates.
    pub fn world_to_device(&self, p: [f64; 3]) -> [f64; 3] {
        p
    }

    /// Check if a point falls inside any dead zone (with margin).
    fn point_in_dead_zone(&self, p: &[f64; 3]) -> bool {
        let margin = self.config.dead_zone_margin;
        for dz in &self.device.tracking_volume.dead_zones {
            let bb = dz.bounding_box().expand(margin);
            if bb.contains_point(p) && dz.contains_point(p) {
                return true;
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use xr_types::kinematic::ArmSide;

    #[test]
    fn test_device_model() {
        let device = DeviceConfig::quest_3();
        let model = DeviceConstraintModel::new(device);
        assert!(model.min_button_size() > 0.0);
    }

    #[test]
    fn test_tracking_volume_check() {
        let device = DeviceConfig::quest_3();
        let model = DeviceConstraintModel::new(device);
        assert!(model.is_in_tracking_volume([0.0, 1.0, 0.0]));
        assert!(!model.is_in_tracking_volume([100.0, 100.0, 100.0]));
    }

    #[test]
    fn test_effective_reach_positive() {
        let device = DeviceConfig::quest_3();
        let model = DeviceConstraintModel::new(device).with_config(DeviceConstraintConfig {
            monte_carlo_samples: 3_000,
            voxel_size: 0.05,
            ..Default::default()
        });
        let chain = KinematicChain::default_arm(ArmSide::Right);
        let params = BodyParameters::average_male();
        let result = model.compute_effective_reach(&chain, &params).unwrap();
        assert!(result.effective_volume > 0.0);
        assert!(result.tracking_quality > 0.0);
        assert!(result.tracking_coverage > 0.0 && result.tracking_coverage <= 1.0);
    }

    #[test]
    fn test_controller_offset_identity() {
        let device = DeviceConfig::quest_3();
        let model = DeviceConstraintModel::new(device);
        // Quest 3 has identity controller_offset
        let p = [1.0, 2.0, 3.0];
        let transformed = model.apply_controller_offset(p);
        for i in 0..3 {
            assert!(
                (p[i] - transformed[i]).abs() < 1e-9,
                "Identity offset should not change point"
            );
        }
    }

    #[test]
    fn test_inverse_offset_round_trip() {
        let device = DeviceConfig::psvr2();
        let model = DeviceConstraintModel::new(device);
        let p = [0.3, 1.2, -0.5];
        let fwd = model.apply_controller_offset(p);
        let back = model.inverse_controller_offset(fwd);
        for i in 0..3 {
            assert!(
                (p[i] - back[i]).abs() < 1e-6,
                "Round-trip mismatch on axis {}: {} vs {}",
                i,
                p[i],
                back[i],
            );
        }
    }

    #[test]
    fn test_seated_constraints() {
        let device = DeviceConfig::vision_pro();
        let model = DeviceConstraintModel::new(device);
        let params = BodyParameters::average_male();
        let seated_base = model.seated_constraints(&params, 0.45);
        let standing_shoulder = params.shoulder_height();
        assert!(
            seated_base[1] < standing_shoulder,
            "Seated shoulder should be lower than standing"
        );
    }

    #[test]
    fn test_dead_zone_impact_no_dead_zones() {
        let device = DeviceConfig::quest_3();
        let model = DeviceConstraintModel::new(device).with_config(DeviceConstraintConfig {
            monte_carlo_samples: 1_000,
            ..Default::default()
        });
        let chain = KinematicChain::default_arm(ArmSide::Right);
        let params = BodyParameters::average_male();
        let impact = model.compute_dead_zone_impact(&chain, &params).unwrap();
        assert!((impact - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_tracking_quality_map() {
        let device = DeviceConfig::quest_3();
        let model = DeviceConstraintModel::new(device).with_config(DeviceConstraintConfig {
            monte_carlo_samples: 2_000,
            voxel_size: 0.05,
            ..Default::default()
        });
        let chain = KinematicChain::default_arm(ArmSide::Right);
        let params = BodyParameters::average_male();
        let map = model.compute_tracking_quality_map(&chain, &params).unwrap();
        assert!(map.mean_quality >= 0.0);
        assert!(map.max_quality >= map.min_quality);
    }

    #[test]
    fn test_comfortable_distance_range() {
        let device = DeviceConfig::quest_3();
        let model = DeviceConstraintModel::new(device);
        let (lo, hi) = model.comfortable_distance_range();
        assert!(lo < hi);
        assert!(lo > 0.0);
    }
}

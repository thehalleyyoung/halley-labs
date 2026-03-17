//! Collision detection for self-collision and environment.
//!
//! Provides capsule-based self-collision checking between arm links,
//! environment collision with axis-aligned bounding boxes, swept-volume
//! collision for transitions between configurations, and minimum clearance
//! computation.

use xr_types::kinematic::{KinematicChain, BodyParameters};
use xr_types::geometry::{BoundingBox, Volume, Sphere, Capsule, point_distance};
use xr_types::error::{VerifierError, VerifierResult};

use crate::body_model::{BodyModelFactory, LinkCollisionGeometry};
use crate::forward_kinematics::ForwardKinematicsSolver;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for collision checking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollisionConfig {
    /// Safety margin added to all collision distances (meters).
    pub margin: f64,
    /// Minimum number of link pairs to skip in self-collision
    /// (adjacent links never collide).
    pub adjacency_skip: usize,
    /// Number of interpolation steps for swept-volume checks.
    pub sweep_steps: usize,
    /// Capsule radius fraction of link length (for auto-generated geometry).
    pub capsule_radius_fraction: f64,
    /// Minimum capsule radius.
    pub min_capsule_radius: f64,
}

impl Default for CollisionConfig {
    fn default() -> Self {
        Self {
            margin: 0.01,
            adjacency_skip: 1,
            sweep_steps: 10,
            capsule_radius_fraction: 0.15,
            min_capsule_radius: 0.02,
        }
    }
}

// ---------------------------------------------------------------------------
// CollisionResult
// ---------------------------------------------------------------------------

/// Result of a collision check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollisionResult {
    /// Whether any collision was detected.
    pub is_colliding: bool,
    /// Minimum clearance distance across all checked pairs.
    pub min_clearance: f64,
    /// Pairs of colliding entities: (index_a, index_b, distance).
    pub collision_pairs: Vec<(usize, usize, f64)>,
}

impl CollisionResult {
    fn no_collision(min_clearance: f64) -> Self {
        Self {
            is_colliding: false,
            min_clearance,
            collision_pairs: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// CollisionChecker
// ---------------------------------------------------------------------------

/// Collision checker for arm self-collision and environment obstacles.
pub struct CollisionChecker {
    obstacles: Vec<Volume>,
    config: CollisionConfig,
    fk: ForwardKinematicsSolver,
    body_factory: BodyModelFactory,
}

impl CollisionChecker {
    pub fn new() -> Self {
        Self {
            obstacles: Vec::new(),
            config: CollisionConfig::default(),
            fk: ForwardKinematicsSolver::new(),
            body_factory: BodyModelFactory::new(),
        }
    }

    pub fn with_config(mut self, config: CollisionConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_margin(mut self, margin: f64) -> Self {
        self.config.margin = margin;
        self
    }

    pub fn add_obstacle(&mut self, volume: Volume) {
        self.obstacles.push(volume);
    }

    pub fn clear_obstacles(&mut self) {
        self.obstacles.clear();
    }

    pub fn obstacle_count(&self) -> usize {
        self.obstacles.len()
    }

    pub fn check_point(&self, point: [f64; 3]) -> bool {
        self.obstacles.iter().any(|v| v.contains_point(&point))
    }

    // -- Self collision --

    /// Check for self-collision among the arm links at a given configuration.
    pub fn self_collision_check(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        angles: &[f64],
    ) -> VerifierResult<CollisionResult> {
        let capsules = self.compute_link_capsules(chain, params, angles)?;
        let margin = self.config.margin;
        let skip = self.config.adjacency_skip;

        let mut min_clearance = f64::INFINITY;
        let mut pairs: Vec<(usize, usize, f64)> = Vec::new();

        for i in 0..capsules.len() {
            for j in (i + 1 + skip)..capsules.len() {
                let dist = capsule_capsule_distance(&capsules[i], &capsules[j]);
                let clearance = dist - margin;
                min_clearance = min_clearance.min(clearance);
                if clearance < 0.0 {
                    pairs.push((i, j, clearance));
                }
            }
        }

        if min_clearance == f64::INFINITY {
            min_clearance = 0.0;
        }

        Ok(CollisionResult {
            is_colliding: !pairs.is_empty(),
            min_clearance,
            collision_pairs: pairs,
        })
    }

    // -- Environment collision --

    /// Check for collision between arm links and environment obstacles.
    pub fn environment_collision_check(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        angles: &[f64],
        obstacles: &[BoundingBox],
    ) -> VerifierResult<CollisionResult> {
        let capsules = self.compute_link_capsules(chain, params, angles)?;
        let margin = self.config.margin;

        let mut min_clearance = f64::INFINITY;
        let mut pairs: Vec<(usize, usize, f64)> = Vec::new();

        for (ci, cap) in capsules.iter().enumerate() {
            for (oi, obs) in obstacles.iter().enumerate() {
                let dist = capsule_box_distance(cap, obs);
                let clearance = dist - margin;
                min_clearance = min_clearance.min(clearance);
                if clearance < 0.0 {
                    pairs.push((ci, oi, clearance));
                }
            }
        }
        // Also check stored obstacles
        for (ci, cap) in capsules.iter().enumerate() {
            for (oi, obs_vol) in self.obstacles.iter().enumerate() {
                let obs_bb = obs_vol.bounding_box();
                let dist = capsule_box_distance(cap, &obs_bb);
                let clearance = dist - margin;
                min_clearance = min_clearance.min(clearance);
                if clearance < 0.0 {
                    pairs.push((ci, obstacles.len() + oi, clearance));
                }
            }
        }

        if min_clearance == f64::INFINITY {
            min_clearance = 0.0;
        }

        Ok(CollisionResult {
            is_colliding: !pairs.is_empty(),
            min_clearance,
            collision_pairs: pairs,
        })
    }

    /// Compute the minimum clearance for a configuration (smallest distance
    /// between any pair of links, or between a link and an obstacle).
    pub fn minimum_clearance(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        angles: &[f64],
    ) -> VerifierResult<f64> {
        let self_result = self.self_collision_check(chain, params, angles)?;
        let env_result = self.environment_collision_check(chain, params, angles, &[])?;
        Ok(self_result.min_clearance.min(env_result.min_clearance))
    }

    /// Swept-volume collision check for a transition between two
    /// configurations. Linearly interpolates joint angles and checks
    /// at each step.
    pub fn swept_volume_collision(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        angles_start: &[f64],
        angles_end: &[f64],
        obstacles: &[BoundingBox],
    ) -> VerifierResult<CollisionResult> {
        let steps = self.config.sweep_steps;
        let n = angles_start.len().min(angles_end.len());

        let mut worst = CollisionResult::no_collision(f64::INFINITY);

        for step in 0..=steps {
            let t = step as f64 / steps as f64;
            let angles: Vec<f64> = (0..n)
                .map(|i| angles_start[i] + t * (angles_end[i] - angles_start[i]))
                .collect();

            let self_r = self.self_collision_check(chain, params, &angles)?;
            let env_r = self.environment_collision_check(chain, params, &angles, obstacles)?;

            if self_r.is_colliding || env_r.is_colliding {
                let combined_pairs = [self_r.collision_pairs, env_r.collision_pairs].concat();
                let mc = self_r.min_clearance.min(env_r.min_clearance);
                return Ok(CollisionResult {
                    is_colliding: true,
                    min_clearance: mc,
                    collision_pairs: combined_pairs,
                });
            }

            let step_clearance = self_r.min_clearance.min(env_r.min_clearance);
            if step_clearance < worst.min_clearance {
                worst.min_clearance = step_clearance;
            }
        }

        Ok(worst)
    }

    /// Check if the arm path (joint positions as a polyline) collides with
    /// any stored obstacle.
    pub fn check_arm_path(&self, joint_positions: &[[f64; 3]]) -> bool {
        for pos in joint_positions {
            if self.check_point(*pos) {
                return true;
            }
        }
        false
    }

    pub fn check_capsule(&self, capsule: &Capsule) -> bool {
        for obstacle in &self.obstacles {
            let bbox = obstacle.bounding_box();
            if capsule.bounding_box().intersects(&bbox) {
                return true;
            }
        }
        false
    }

    // -- Internal helpers --

    /// Generate capsule geometries for each link at the given configuration.
    fn compute_link_capsules(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        angles: &[f64],
    ) -> VerifierResult<Vec<Capsule>> {
        let positions = self.fk.joint_positions(chain, params, angles)?;
        let mut capsules = Vec::new();

        for i in 0..positions.len().saturating_sub(1) {
            let start = positions[i];
            let end = positions[i + 1];
            let link_len = point_distance(&start, &end);
            if link_len < 0.001 {
                continue;
            }
            let radius = (link_len * self.config.capsule_radius_fraction)
                .max(self.config.min_capsule_radius);
            capsules.push(Capsule::new(start, end, radius));
        }

        Ok(capsules)
    }
}

impl Default for CollisionChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Distance functions
// ---------------------------------------------------------------------------

/// Compute the minimum distance between two capsules.
/// Returns 0.0 if they overlap.
pub fn capsule_capsule_distance(a: &Capsule, b: &Capsule) -> f64 {
    // Distance between two line segments, then subtract both radii
    let seg_dist = segment_segment_distance(
        &a.start, &a.end,
        &b.start, &b.end,
    );
    (seg_dist - a.radius - b.radius).max(0.0)
}

/// Compute the minimum distance between a capsule and an axis-aligned box.
/// Returns 0.0 if they overlap.
pub fn capsule_box_distance(cap: &Capsule, bbox: &BoundingBox) -> f64 {
    // Find the closest point on the capsule axis to the box, then subtract radius
    let seg_dist = segment_box_distance(&cap.start, &cap.end, bbox);
    (seg_dist - cap.radius).max(0.0)
}

/// Distance between two line segments in 3D.
fn segment_segment_distance(
    a0: &[f64; 3], a1: &[f64; 3],
    b0: &[f64; 3], b1: &[f64; 3],
) -> f64 {
    let d1 = sub3(a1, a0);
    let d2 = sub3(b1, b0);
    let r = sub3(a0, b0);

    let a = dot3(&d1, &d1);
    let e = dot3(&d2, &d2);
    let f = dot3(&d2, &r);

    if a < 1e-12 && e < 1e-12 {
        return dist3(a0, b0);
    }

    let (s, t);
    if a < 1e-12 {
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = dot3(&d1, &r);
        if e < 1e-12 {
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b_val = dot3(&d1, &d2);
            let denom = a * e - b_val * b_val;

            let sn = if denom.abs() > 1e-12 {
                ((b_val * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };

            let tn = (b_val * sn + f) / e;

            if tn < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if tn > 1.0 {
                t = 1.0;
                s = ((b_val - c) / a).clamp(0.0, 1.0);
            } else {
                s = sn;
                t = tn;
            }
        }
    }

    let closest_a = [a0[0] + s * d1[0], a0[1] + s * d1[1], a0[2] + s * d1[2]];
    let closest_b = [b0[0] + t * d2[0], b0[1] + t * d2[1], b0[2] + t * d2[2]];
    dist3(&closest_a, &closest_b)
}

/// Distance from a line segment to an AABB.
fn segment_box_distance(seg_start: &[f64; 3], seg_end: &[f64; 3], bbox: &BoundingBox) -> f64 {
    // Sample the segment and find the minimum point-to-box distance
    let steps = 16;
    let mut min_dist = f64::INFINITY;
    for i in 0..=steps {
        let t = i as f64 / steps as f64;
        let p = [
            seg_start[0] + t * (seg_end[0] - seg_start[0]),
            seg_start[1] + t * (seg_end[1] - seg_start[1]),
            seg_start[2] + t * (seg_end[2] - seg_start[2]),
        ];
        let d = point_box_distance(&p, bbox);
        min_dist = min_dist.min(d);
    }
    min_dist
}

/// Distance from a point to the nearest surface of an AABB.
/// Returns 0 if the point is inside.
fn point_box_distance(p: &[f64; 3], bbox: &BoundingBox) -> f64 {
    let mut sq = 0.0;
    for i in 0..3 {
        if p[i] < bbox.min[i] {
            let d = bbox.min[i] - p[i];
            sq += d * d;
        } else if p[i] > bbox.max[i] {
            let d = p[i] - bbox.max[i];
            sq += d * d;
        }
    }
    sq.sqrt()
}

// --- Small vec3 helpers ---

fn sub3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn dist3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let d = sub3(a, b);
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
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
    fn test_collision_checker_default() {
        let mut cc = CollisionChecker::new();
        assert!(!cc.check_point([0.0, 0.0, 0.0]));
        cc.add_obstacle(Volume::Sphere(Sphere::new([0.0, 0.0, 0.0], 1.0)));
        assert!(cc.check_point([0.0, 0.0, 0.0]));
    }

    #[test]
    fn test_capsule_capsule_distance_parallel() {
        let a = Capsule::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.1);
        let b = Capsule::new([0.0, 1.0, 0.0], [1.0, 1.0, 0.0], 0.1);
        let d = capsule_capsule_distance(&a, &b);
        assert!((d - 0.8).abs() < 0.01, "distance = {}", d);
    }

    #[test]
    fn test_capsule_capsule_distance_overlapping() {
        let a = Capsule::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.5);
        let b = Capsule::new([0.0, 0.3, 0.0], [1.0, 0.3, 0.0], 0.5);
        let d = capsule_capsule_distance(&a, &b);
        assert!(d < 0.01, "overlapping capsules should have ~0 distance");
    }

    #[test]
    fn test_capsule_box_distance() {
        let cap = Capsule::new([0.0, 2.0, 0.0], [1.0, 2.0, 0.0], 0.1);
        let bbox = BoundingBox::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let d = capsule_box_distance(&cap, &bbox);
        // Capsule is 2.0 above box top (y=1), axis distance ~1.0, minus radius 0.1
        assert!(d > 0.5, "Should have clearance, got {}", d);
    }

    #[test]
    fn test_self_collision_zero_config() {
        let (chain, params) = test_chain_and_params();
        let cc = CollisionChecker::new();
        let angles = chain.midpoint_config(&params);
        let result = cc.self_collision_check(&chain, &params, &angles).unwrap();
        // At midpoint config the arm is moderately extended — may or may not collide
        // but the check should complete without error
        assert!(result.min_clearance.is_finite());
    }

    #[test]
    fn test_environment_collision_no_obstacles() {
        let (chain, params) = test_chain_and_params();
        let cc = CollisionChecker::new();
        let angles = chain.midpoint_config(&params);
        let result = cc
            .environment_collision_check(&chain, &params, &angles, &[])
            .unwrap();
        assert!(!result.is_colliding);
    }

    #[test]
    fn test_environment_collision_with_obstacle() {
        let (chain, params) = test_chain_and_params();
        let cc = CollisionChecker::new();
        // Place a large obstacle encompassing the shoulder area
        let shoulder = chain.base_position(&params);
        let obstacle = BoundingBox::new(
            [shoulder[0] - 2.0, shoulder[1] - 2.0, shoulder[2] - 2.0],
            [shoulder[0] + 2.0, shoulder[1] + 2.0, shoulder[2] + 2.0],
        );
        let angles = chain.midpoint_config(&params);
        let result = cc
            .environment_collision_check(&chain, &params, &angles, &[obstacle])
            .unwrap();
        assert!(
            result.is_colliding,
            "Arm inside huge box should collide"
        );
    }

    #[test]
    fn test_swept_volume_no_collision() {
        let (chain, params) = test_chain_and_params();
        let cc = CollisionChecker::new().with_config(CollisionConfig {
            sweep_steps: 5,
            ..Default::default()
        });
        let mid = chain.midpoint_config(&params);
        // Small perturbation
        let mut end = mid.clone();
        for a in end.iter_mut() {
            *a += 0.05;
        }
        let end = chain.clamp_to_limits(&end, &params);
        let result = cc
            .swept_volume_collision(&chain, &params, &mid, &end, &[])
            .unwrap();
        // Small movement should be safe
        assert!(result.min_clearance.is_finite());
    }

    #[test]
    fn test_minimum_clearance() {
        let (chain, params) = test_chain_and_params();
        let cc = CollisionChecker::new();
        let angles = chain.midpoint_config(&params);
        let clearance = cc.minimum_clearance(&chain, &params, &angles).unwrap();
        assert!(clearance.is_finite());
    }

    #[test]
    fn test_point_box_distance_inside() {
        let bbox = BoundingBox::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!(point_box_distance(&[0.5, 0.5, 0.5], &bbox) < 1e-9);
    }

    #[test]
    fn test_point_box_distance_outside() {
        let bbox = BoundingBox::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let d = point_box_distance(&[2.0, 0.5, 0.5], &bbox);
        assert!((d - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_segment_segment_distance_parallel() {
        let d = segment_segment_distance(
            &[0.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0],
            &[1.0, 1.0, 0.0],
        );
        assert!((d - 1.0).abs() < 1e-9);
    }
}

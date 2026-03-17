//! Body model management: creating kinematic chains from BodyParameters.
//!
//! Implements 7-DOF arm model (3-DOF shoulder + 1-DOF elbow + 3-DOF wrist)
//! with parameter-dependent link lengths and joint limits.

use xr_types::kinematic::{
    KinematicChain, Joint, JointType, JointLimits, BodyParameters, ArmSide, FullBodyModel,
};
use xr_types::geometry::{Capsule, BoundingBox, Sphere, point_distance};
use xr_types::error::{VerifierError, VerifierResult};
use xr_types::deg_to_rad;

/// Configuration for body model creation.
#[derive(Debug, Clone)]
pub struct BodyModelConfig {
    /// Collision capsule radius as fraction of link length.
    pub capsule_radius_fraction: f64,
    /// Minimum collision capsule radius in meters.
    pub min_capsule_radius: f64,
    /// Self-collision safety margin in meters.
    pub self_collision_margin: f64,
    /// Comfort zone shoulder elevation limit (radians).
    pub comfort_elevation_limit: f64,
    /// Comfort zone forward reach fraction.
    pub comfort_reach_fraction: f64,
}

impl Default for BodyModelConfig {
    fn default() -> Self {
        Self {
            capsule_radius_fraction: 0.15,
            min_capsule_radius: 0.02,
            self_collision_margin: 0.01,
            comfort_elevation_limit: deg_to_rad(90.0),
            comfort_reach_fraction: 0.7,
        }
    }
}

/// Collision geometry for a body link.
#[derive(Debug, Clone)]
pub struct LinkCollisionGeometry {
    /// Link name.
    pub name: String,
    /// Capsule representing the link.
    pub capsule: Capsule,
    /// Parent joint index.
    pub parent_joint: usize,
    /// Child joint index.
    pub child_joint: usize,
}

/// Comfort zone definition for a body parameterization.
#[derive(Debug, Clone)]
pub struct ComfortZone {
    /// Center of the comfort zone in world space.
    pub center: [f64; 3],
    /// Maximum comfortable reach distance.
    pub max_reach: f64,
    /// Minimum comfortable reach distance (too close is uncomfortable).
    pub min_reach: f64,
    /// Upper height limit of comfort zone.
    pub height_max: f64,
    /// Lower height limit of comfort zone.
    pub height_min: f64,
    /// Forward reach limit.
    pub forward_max: f64,
    /// Lateral reach limit.
    pub lateral_max: f64,
}

impl ComfortZone {
    /// Check if a point is within the comfort zone.
    pub fn contains_point(&self, point: &[f64; 3]) -> bool {
        let dist = point_distance(point, &self.center);
        if dist < self.min_reach || dist > self.max_reach {
            return false;
        }
        if point[1] < self.height_min || point[1] > self.height_max {
            return false;
        }
        let forward = point[2] - self.center[2];
        if forward.abs() > self.forward_max {
            return false;
        }
        let lateral = point[0] - self.center[0];
        if lateral.abs() > self.lateral_max {
            return false;
        }
        true
    }

    /// Compute a comfort score (0 to 1) for a point, 1 being most comfortable.
    pub fn comfort_score(&self, point: &[f64; 3]) -> f64 {
        if !self.contains_point(point) {
            return 0.0;
        }
        let dist = point_distance(point, &self.center);
        let optimal_dist = (self.min_reach + self.max_reach) * 0.5;
        let dist_score = 1.0
            - ((dist - optimal_dist) / (self.max_reach - self.min_reach))
                .abs()
                .min(1.0);

        let optimal_height = (self.height_min + self.height_max) * 0.5;
        let height_range = self.height_max - self.height_min;
        let height_score = if height_range > 1e-6 {
            1.0 - ((point[1] - optimal_height) / (height_range * 0.5))
                .abs()
                .min(1.0)
        } else {
            1.0
        };

        dist_score * height_score
    }
}

/// Factory for creating body models from body parameters.
#[derive(Debug, Clone)]
pub struct BodyModelFactory {
    config: BodyModelConfig,
}

impl BodyModelFactory {
    /// Create a new factory with default configuration.
    pub fn new() -> Self {
        Self {
            config: BodyModelConfig::default(),
        }
    }

    /// Create a new factory with custom configuration.
    pub fn with_config(config: BodyModelConfig) -> Self {
        Self { config }
    }

    /// Create a 7-DOF arm kinematic chain from body parameters.
    pub fn create_arm(&self, params: &BodyParameters, side: ArmSide) -> KinematicChain {
        let sign = match side {
            ArmSide::Left => -1.0,
            ArmSide::Right => 1.0,
        };

        let joints = vec![
            // Shoulder flexion/extension (X axis)
            {
                let mut j = Joint::revolute("shoulder_flexion", JointType::RevoluteX, 0.0);
                j.limits = JointLimits::new(deg_to_rad(-60.0), deg_to_rad(180.0));
                j.link_mass = 2.0;
                j.parameter_dependent = true;
                j.min_limit_coefficients = [0.0, -0.05, 0.0, 0.0, 0.0];
                j.max_limit_coefficients = [0.0, 0.02, 0.0, 0.0, 0.0];
                j
            },
            // Shoulder abduction/adduction (Z axis)
            {
                let mut j = Joint::revolute("shoulder_abduction", JointType::RevoluteZ, 0.0);
                j.limits = JointLimits::new(deg_to_rad(-10.0), deg_to_rad(180.0));
                j.link_mass = 0.5;
                j
            },
            // Shoulder internal/external rotation (Y axis)
            {
                let mut j = Joint::revolute("shoulder_rotation", JointType::RevoluteY, 0.0);
                j.limits = JointLimits::new(deg_to_rad(-90.0), deg_to_rad(90.0));
                j.link_length = params.arm_length;
                j.parameter_dependent = true;
                j.length_coefficients = [0.0, 0.8, 0.0, 0.0, 0.0];
                j.link_mass = 2.5;
                j
            },
            // Elbow flexion/extension (X axis)
            {
                let mut j = Joint::revolute("elbow_flexion", JointType::RevoluteX, 0.0);
                j.limits = JointLimits::new(0.0, deg_to_rad(145.0));
                j.link_length = params.forearm_length;
                j.parameter_dependent = true;
                j.length_coefficients = [0.0, 0.0, 0.0, 0.8, 0.0];
                j.link_mass = 1.5;
                j
            },
            // Wrist pronation/supination (Y axis)
            {
                let mut j = Joint::revolute("wrist_pronation", JointType::RevoluteY, 0.0);
                j.limits = JointLimits::new(deg_to_rad(-80.0), deg_to_rad(80.0));
                j.link_mass = 0.3;
                j
            },
            // Wrist flexion/extension (X axis)
            {
                let mut j = Joint::revolute("wrist_flexion", JointType::RevoluteX, 0.0);
                j.limits = JointLimits::new(deg_to_rad(-70.0), deg_to_rad(70.0));
                j.link_mass = 0.2;
                j
            },
            // Wrist radial/ulnar deviation (Z axis)
            {
                let mut j = Joint::revolute("wrist_deviation", JointType::RevoluteZ, 0.0);
                j.limits = JointLimits::new(deg_to_rad(-20.0), deg_to_rad(30.0));
                j.link_length = params.hand_length;
                j.parameter_dependent = true;
                j.length_coefficients = [0.0, 0.0, 0.0, 0.0, 0.8];
                j.link_mass = 0.4;
                j
            },
        ];

        let shoulder_offset_x = sign * params.shoulder_breadth * 0.5;
        let shoulder_height = params.shoulder_height();

        KinematicChain {
            name: format!("{:?} Arm (parameterized)", side),
            joints,
            base_transform: [
                1.0, 0.0, 0.0, shoulder_offset_x,
                0.0, 1.0, 0.0, shoulder_height,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
            side,
            base_position_coefficients: [
                [0.0, 0.0, sign * 0.5, 0.0, 0.0],
                [0.818, 0.0, 0.0, 0.0, 0.0],
                [0.0; 5],
            ],
        }
    }

    /// Create both arms.
    pub fn create_both_arms(&self, params: &BodyParameters) -> (KinematicChain, KinematicChain) {
        (
            self.create_arm(params, ArmSide::Left),
            self.create_arm(params, ArmSide::Right),
        )
    }

    /// Create a full body model (both arms) from body parameters.
    pub fn create_full_body(
        &self,
        params: &BodyParameters,
        seated: bool,
        seat_height: f64,
    ) -> FullBodyModel {
        if seated {
            FullBodyModel::seated(*params, seat_height)
        } else {
            FullBodyModel::standing(*params)
        }
    }

    /// Create collision geometry for an arm chain.
    pub fn create_collision_geometry(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
    ) -> Vec<LinkCollisionGeometry> {
        let fk = crate::forward_kinematics::ForwardKinematicsSolver::new();
        let n = chain.joints.len();
        let zero_angles = vec![0.0; chain.total_dof()];
        let positions = fk
            .joint_positions(chain, params, &zero_angles)
            .unwrap_or_default();

        let mut geometries = Vec::new();

        for i in 0..n {
            if i + 1 >= positions.len() {
                break;
            }
            let start = positions[i];
            let end = positions[i + 1];
            let link_len = point_distance(&start, &end);
            if link_len < 0.001 {
                continue;
            }

            let radius = (link_len * self.config.capsule_radius_fraction)
                .max(self.config.min_capsule_radius);

            geometries.push(LinkCollisionGeometry {
                name: chain.joints[i].name.clone(),
                capsule: Capsule::new(start, end, radius),
                parent_joint: i,
                child_joint: i + 1,
            });
        }

        geometries
    }

    /// Compute the comfort zone for a body parameterization.
    pub fn compute_comfort_zone(
        &self,
        params: &BodyParameters,
        side: ArmSide,
    ) -> ComfortZone {
        let chain = KinematicChain::default_arm(side);
        let base_pos = chain.base_position(params);
        let max_reach = params.total_reach();
        let comfortable_reach = max_reach * self.config.comfort_reach_fraction;

        let shoulder_height = params.shoulder_height();
        let elbow_height = params.elbow_height();

        ComfortZone {
            center: base_pos,
            max_reach: comfortable_reach,
            min_reach: params.forearm_length * 0.3,
            height_max: shoulder_height + params.arm_length * 0.3,
            height_min: elbow_height - params.forearm_length * 0.5,
            forward_max: comfortable_reach * 0.9,
            lateral_max: comfortable_reach * 0.7,
        }
    }

    /// Validate body parameters for model creation.
    pub fn validate_params(&self, params: &BodyParameters) -> VerifierResult<()> {
        params
            .validate()
            .map_err(|e| VerifierError::KinematicModel(e))
    }

    /// Create a simplified 3-DOF arm for fast analysis.
    pub fn create_simplified_arm(
        &self,
        params: &BodyParameters,
        side: ArmSide,
    ) -> KinematicChain {
        let sign = match side {
            ArmSide::Left => -1.0,
            ArmSide::Right => 1.0,
        };
        let joints = vec![
            Joint::revolute("shoulder", JointType::RevoluteX, params.arm_length)
                .with_limits(JointLimits::new(deg_to_rad(-60.0), deg_to_rad(180.0))),
            Joint::revolute("elbow", JointType::RevoluteX, params.forearm_length)
                .with_limits(JointLimits::new(0.0, deg_to_rad(145.0))),
            Joint::revolute("wrist", JointType::RevoluteX, params.hand_length)
                .with_limits(JointLimits::new(deg_to_rad(-70.0), deg_to_rad(70.0))),
        ];
        let sh = params.shoulder_height();
        KinematicChain {
            name: format!("{:?} Arm (simplified)", side),
            joints,
            base_transform: [
                1.0, 0.0, 0.0, sign * params.shoulder_breadth * 0.5,
                0.0, 1.0, 0.0, sh,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
            side,
            base_position_coefficients: [[0.0; 5]; 3],
        }
    }

    /// Compute link masses based on body parameters.
    pub fn compute_link_masses(params: &BodyParameters) -> Vec<f64> {
        let total_mass = params.stature * 45.0;
        vec![
            total_mass * 0.028, // upper arm
            total_mass * 0.028, // upper arm continued
            total_mass * 0.028, // upper arm to elbow
            total_mass * 0.016, // forearm
            total_mass * 0.006, // hand/wrist
            total_mass * 0.006, // hand/wrist
            total_mass * 0.006, // hand
        ]
    }

    /// Generate body parameters at a specific percentile.
    pub fn params_at_percentile(percentile: f64) -> BodyParameters {
        let small = BodyParameters::small_female();
        let large = BodyParameters::large_male();
        let t = percentile.clamp(0.0, 1.0);
        small.lerp(&large, t)
    }

    /// Generate a range of body parameters for population coverage.
    pub fn generate_param_range(num_steps: usize) -> Vec<BodyParameters> {
        (0..num_steps)
            .map(|i| {
                let t = i as f64 / (num_steps as f64 - 1.0).max(1.0);
                Self::params_at_percentile(t)
            })
            .collect()
    }

    /// Get the set of representative body parameter samples for population coverage.
    pub fn population_samples(&self, n: usize) -> Vec<BodyParameters> {
        Self::generate_param_range(n)
    }

    /// Compute the torso bounding box for self-collision checks.
    pub fn torso_bounds(params: &BodyParameters) -> BoundingBox {
        let half_shoulder = params.shoulder_breadth * 0.5;
        let hip_height = params.stature * 0.52;
        let shoulder_height = params.shoulder_height();
        let depth = 0.15;
        BoundingBox::new(
            [-half_shoulder, hip_height, -depth],
            [half_shoulder, shoulder_height, depth],
        )
    }

    /// Compute the head bounding sphere for collision avoidance.
    pub fn head_sphere(params: &BodyParameters) -> Sphere {
        let head_radius = 0.10;
        let head_center_height = params.stature - head_radius;
        Sphere::new([0.0, head_center_height, 0.0], head_radius)
    }

    /// Compute shoulder positions for both arms.
    pub fn shoulder_positions(
        &self,
        params: &BodyParameters,
    ) -> ([f64; 3], [f64; 3]) {
        let left = KinematicChain::default_arm(ArmSide::Left);
        let right = KinematicChain::default_arm(ArmSide::Right);
        (left.base_position(params), right.base_position(params))
    }
}

impl Default for BodyModelFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xr_types::kinematic::{ArmSide, BodyParameters};

    #[test]
    fn test_create_arm() {
        let factory = BodyModelFactory::new();
        let params = BodyParameters::average_male();
        let chain = factory.create_arm(&params, ArmSide::Right);
        assert_eq!(chain.total_dof(), 7);
        assert_eq!(chain.joints.len(), 7);
    }

    #[test]
    fn test_create_left_arm() {
        let factory = BodyModelFactory::new();
        let params = BodyParameters::average_male();
        let chain = factory.create_arm(&params, ArmSide::Left);
        let base = chain.base_position(&params);
        assert!(base[0] < 0.0);
    }

    #[test]
    fn test_create_right_arm() {
        let factory = BodyModelFactory::new();
        let params = BodyParameters::average_male();
        let chain = factory.create_arm(&params, ArmSide::Right);
        let base = chain.base_position(&params);
        assert!(base[0] > 0.0);
    }

    #[test]
    fn test_create_full_body_standing() {
        let factory = BodyModelFactory::new();
        let params = BodyParameters::average_male();
        let model = factory.create_full_body(&params, false, 0.0);
        assert!(!model.seated);
    }

    #[test]
    fn test_create_full_body_seated() {
        let factory = BodyModelFactory::new();
        let params = BodyParameters::average_male();
        let model = factory.create_full_body(&params, true, 0.45);
        assert!(model.seated);
    }

    #[test]
    fn test_collision_geometry() {
        let factory = BodyModelFactory::new();
        let params = BodyParameters::average_male();
        let chain = factory.create_arm(&params, ArmSide::Right);
        let geom = factory.create_collision_geometry(&chain, &params);
        assert!(!geom.is_empty());
        for g in &geom {
            assert!(g.capsule.radius > 0.0);
        }
    }

    #[test]
    fn test_comfort_zone() {
        let factory = BodyModelFactory::new();
        let params = BodyParameters::average_male();
        let zone = factory.compute_comfort_zone(&params, ArmSide::Right);
        assert!(zone.max_reach > zone.min_reach);
        assert!(zone.height_max > zone.height_min);
    }

    #[test]
    fn test_comfort_zone_score() {
        let factory = BodyModelFactory::new();
        let params = BodyParameters::average_male();
        let zone = factory.compute_comfort_zone(&params, ArmSide::Right);
        let near_center = [
            zone.center[0],
            (zone.height_min + zone.height_max) * 0.5,
            zone.center[2] + zone.forward_max * 0.5,
        ];
        let score = zone.comfort_score(&near_center);
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_simplified_arm() {
        let factory = BodyModelFactory::new();
        let params = BodyParameters::average_male();
        let chain = factory.create_simplified_arm(&params, ArmSide::Right);
        assert_eq!(chain.total_dof(), 3);
    }

    #[test]
    fn test_params_at_percentile() {
        let p50 = BodyModelFactory::params_at_percentile(0.5);
        let small = BodyParameters::small_female();
        let large = BodyParameters::large_male();
        assert!(p50.stature > small.stature);
        assert!(p50.stature < large.stature);
    }

    #[test]
    fn test_generate_param_range() {
        let range = BodyModelFactory::generate_param_range(5);
        assert_eq!(range.len(), 5);
        assert!(range[0].stature < range[4].stature);
    }

    #[test]
    fn test_link_masses() {
        let params = BodyParameters::average_male();
        let masses = BodyModelFactory::compute_link_masses(&params);
        assert_eq!(masses.len(), 7);
        for m in &masses {
            assert!(*m > 0.0);
        }
    }

    #[test]
    fn test_torso_bounds() {
        let params = BodyParameters::average_male();
        let bounds = BodyModelFactory::torso_bounds(&params);
        assert!(bounds.volume() > 0.0);
    }

    #[test]
    fn test_head_sphere() {
        let params = BodyParameters::average_male();
        let head = BodyModelFactory::head_sphere(&params);
        assert!(head.radius > 0.0);
        assert!(head.center[1] > params.stature * 0.9);
    }

    #[test]
    fn test_validate_params() {
        let factory = BodyModelFactory::new();
        let params = BodyParameters::average_male();
        assert!(factory.validate_params(&params).is_ok());
    }

    #[test]
    fn test_different_body_sizes() {
        let factory = BodyModelFactory::new();
        let small = BodyParameters::small_female();
        let large = BodyParameters::large_male();
        let chain_s = factory.create_arm(&small, ArmSide::Right);
        let chain_l = factory.create_arm(&large, ArmSide::Right);
        assert!(chain_l.total_link_length(&large) > chain_s.total_link_length(&small));
    }

    #[test]
    fn test_both_arms() {
        let factory = BodyModelFactory::new();
        let params = BodyParameters::average_male();
        let (left, right) = factory.create_both_arms(&params);
        assert_eq!(left.side, ArmSide::Left);
        assert_eq!(right.side, ArmSide::Right);
    }

    #[test]
    fn test_population_samples() {
        let factory = BodyModelFactory::new();
        let samples = factory.population_samples(5);
        assert_eq!(samples.len(), 5);
    }

    #[test]
    fn test_shoulder_positions() {
        let factory = BodyModelFactory::new();
        let params = BodyParameters::average_male();
        let (left, right) = factory.shoulder_positions(&params);
        assert!(left[0] < right[0]);
    }
}

//! Kinematic body model types (Definition D2).
//!
//! Parameterized kinematic chain B = (Θ, K, J, FK) for modeling
//! human arm reachability in XR.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Body parameters from ANSUR-II anthropometric database.
/// θ ∈ Θ ⊂ R^5: stature, arm_length, shoulder_breadth, forearm_length, hand_length.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BodyParameters {
    /// Total body height in meters.
    pub stature: f64,
    /// Upper arm length in meters (shoulder to elbow).
    pub arm_length: f64,
    /// Shoulder breadth (bi-deltoid) in meters.
    pub shoulder_breadth: f64,
    /// Forearm length in meters (elbow to wrist).
    pub forearm_length: f64,
    /// Hand length in meters (wrist to fingertip).
    pub hand_length: f64,
}

impl BodyParameters {
    /// Create body parameters from raw values.
    pub fn new(stature: f64, arm_length: f64, shoulder_breadth: f64, forearm_length: f64, hand_length: f64) -> Self {
        Self {
            stature,
            arm_length,
            shoulder_breadth,
            forearm_length,
            hand_length,
        }
    }

    /// Create a 50th-percentile male body model.
    pub fn average_male() -> Self {
        Self {
            stature: 1.756,
            arm_length: 0.366,
            shoulder_breadth: 0.481,
            forearm_length: 0.268,
            hand_length: 0.193,
        }
    }

    /// Create a 50th-percentile female body model.
    pub fn average_female() -> Self {
        Self {
            stature: 1.626,
            arm_length: 0.330,
            shoulder_breadth: 0.420,
            forearm_length: 0.240,
            hand_length: 0.178,
        }
    }

    /// Create a 5th-percentile female body model (small).
    pub fn small_female() -> Self {
        Self {
            stature: 1.511,
            arm_length: 0.299,
            shoulder_breadth: 0.375,
            forearm_length: 0.218,
            hand_length: 0.163,
        }
    }

    /// Create a 95th-percentile male body model (large).
    pub fn large_male() -> Self {
        Self {
            stature: 1.883,
            arm_length: 0.401,
            shoulder_breadth: 0.528,
            forearm_length: 0.298,
            hand_length: 0.213,
        }
    }

    /// Get total arm reach (arm + forearm + hand).
    pub fn total_reach(&self) -> f64 {
        self.arm_length + self.forearm_length + self.hand_length
    }

    /// Get shoulder height (approximate).
    pub fn shoulder_height(&self) -> f64 {
        self.stature * 0.818
    }

    /// Get elbow height when arm is at side.
    pub fn elbow_height(&self) -> f64 {
        self.shoulder_height() - self.arm_length
    }

    /// Get the parameter vector as an array.
    pub fn to_array(&self) -> [f64; 5] {
        [
            self.stature,
            self.arm_length,
            self.shoulder_breadth,
            self.forearm_length,
            self.hand_length,
        ]
    }

    /// Create from array.
    pub fn from_array(arr: &[f64; 5]) -> Self {
        Self {
            stature: arr[0],
            arm_length: arr[1],
            shoulder_breadth: arr[2],
            forearm_length: arr[3],
            hand_length: arr[4],
        }
    }

    /// Linear interpolation between two body parameter sets.
    pub fn lerp(&self, other: &BodyParameters, t: f64) -> BodyParameters {
        BodyParameters {
            stature: crate::lerp(self.stature, other.stature, t),
            arm_length: crate::lerp(self.arm_length, other.arm_length, t),
            shoulder_breadth: crate::lerp(self.shoulder_breadth, other.shoulder_breadth, t),
            forearm_length: crate::lerp(self.forearm_length, other.forearm_length, t),
            hand_length: crate::lerp(self.hand_length, other.hand_length, t),
        }
    }

    /// Compute the normalized parameter vector (each dimension in [0, 1]).
    pub fn normalized(&self, min: &BodyParameters, max: &BodyParameters) -> [f64; 5] {
        let a = self.to_array();
        let lo = min.to_array();
        let hi = max.to_array();
        let mut result = [0.0; 5];
        for i in 0..5 {
            let range = hi[i] - lo[i];
            result[i] = if range > 1e-12 {
                (a[i] - lo[i]) / range
            } else {
                0.5
            };
        }
        result
    }

    /// Validate that parameters are within reasonable ranges.
    pub fn validate(&self) -> Result<(), String> {
        if self.stature < 1.0 || self.stature > 2.5 {
            return Err(format!("Stature {} out of range [1.0, 2.5]", self.stature));
        }
        if self.arm_length < 0.15 || self.arm_length > 0.60 {
            return Err(format!("Arm length {} out of range [0.15, 0.60]", self.arm_length));
        }
        if self.shoulder_breadth < 0.25 || self.shoulder_breadth > 0.70 {
            return Err(format!("Shoulder breadth {} out of range [0.25, 0.70]", self.shoulder_breadth));
        }
        if self.forearm_length < 0.12 || self.forearm_length > 0.45 {
            return Err(format!("Forearm length {} out of range [0.12, 0.45]", self.forearm_length));
        }
        if self.hand_length < 0.10 || self.hand_length > 0.30 {
            return Err(format!("Hand length {} out of range [0.10, 0.30]", self.hand_length));
        }
        Ok(())
    }
}

impl Default for BodyParameters {
    fn default() -> Self {
        Self::average_male()
    }
}

/// Joint type in the kinematic chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum JointType {
    /// Rotation about X axis.
    RevoluteX,
    /// Rotation about Y axis.
    RevoluteY,
    /// Rotation about Z axis.
    RevoluteZ,
    /// Ball joint (3 DOF) - represented as 3 revolute joints.
    Ball,
    /// Fixed joint (0 DOF).
    Fixed,
}

impl JointType {
    /// Number of degrees of freedom.
    pub fn dof(&self) -> usize {
        match self {
            JointType::RevoluteX | JointType::RevoluteY | JointType::RevoluteZ => 1,
            JointType::Ball => 3,
            JointType::Fixed => 0,
        }
    }

    /// Get the rotation axis for revolute joints.
    pub fn axis(&self) -> [f64; 3] {
        match self {
            JointType::RevoluteX => [1.0, 0.0, 0.0],
            JointType::RevoluteY => [0.0, 1.0, 0.0],
            JointType::RevoluteZ => [0.0, 0.0, 1.0],
            JointType::Ball | JointType::Fixed => [0.0, 0.0, 0.0],
        }
    }
}

/// Joint limits for a single joint.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct JointLimits {
    /// Minimum angle in radians.
    pub min: f64,
    /// Maximum angle in radians.
    pub max: f64,
    /// Maximum angular velocity in rad/s.
    pub max_velocity: f64,
    /// Maximum angular acceleration in rad/s^2.
    pub max_acceleration: f64,
}

impl JointLimits {
    /// Create new joint limits.
    pub fn new(min: f64, max: f64) -> Self {
        Self {
            min,
            max,
            max_velocity: 3.0,
            max_acceleration: 10.0,
        }
    }

    /// Create symmetric limits.
    pub fn symmetric(half_range: f64) -> Self {
        Self::new(-half_range, half_range)
    }

    /// Get the range of the joint.
    pub fn range(&self) -> f64 {
        self.max - self.min
    }

    /// Get the midpoint of the range.
    pub fn midpoint(&self) -> f64 {
        (self.min + self.max) * 0.5
    }

    /// Check if an angle is within limits.
    pub fn contains(&self, angle: f64) -> bool {
        angle >= self.min && angle <= self.max
    }

    /// Clamp an angle to the limits.
    pub fn clamp(&self, angle: f64) -> f64 {
        crate::clamp(angle, self.min, self.max)
    }

    /// Compute the normalized position (0 = min, 1 = max).
    pub fn normalized(&self, angle: f64) -> f64 {
        let range = self.range();
        if range < 1e-12 {
            0.5
        } else {
            (angle - self.min) / range
        }
    }

    /// Convert from normalized to angle.
    pub fn from_normalized(&self, t: f64) -> f64 {
        self.min + t * self.range()
    }

    /// Linear interpolation of limits between two body parameterizations.
    pub fn lerp(&self, other: &JointLimits, t: f64) -> JointLimits {
        JointLimits {
            min: crate::lerp(self.min, other.min, t),
            max: crate::lerp(self.max, other.max, t),
            max_velocity: crate::lerp(self.max_velocity, other.max_velocity, t),
            max_acceleration: crate::lerp(self.max_acceleration, other.max_acceleration, t),
        }
    }
}

impl Default for JointLimits {
    fn default() -> Self {
        Self::symmetric(std::f64::consts::PI)
    }
}

/// A single joint in the kinematic chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Joint {
    /// Joint name.
    pub name: String,
    /// Joint type.
    pub joint_type: JointType,
    /// Joint limits.
    pub limits: JointLimits,
    /// Link length to next joint (in meters).
    pub link_length: f64,
    /// Static transform from parent joint frame.
    pub static_transform: [f64; 16],
    /// Link mass in kg (for dynamics).
    pub link_mass: f64,
    /// Whether the joint limits vary with body parameters.
    pub parameter_dependent: bool,
    /// Coefficients for body-parameter-dependent link length:
    /// link_length(θ) = base_length + Σ coeffs[i] * θ[i]
    pub length_coefficients: [f64; 5],
    /// Coefficients for body-parameter-dependent min limit.
    pub min_limit_coefficients: [f64; 5],
    /// Coefficients for body-parameter-dependent max limit.
    pub max_limit_coefficients: [f64; 5],
}

impl Joint {
    /// Create a new revolute joint.
    pub fn revolute(name: impl Into<String>, joint_type: JointType, link_length: f64) -> Self {
        Self {
            name: name.into(),
            joint_type,
            limits: JointLimits::default(),
            link_length,
            static_transform: [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
            link_mass: 1.0,
            parameter_dependent: false,
            length_coefficients: [0.0; 5],
            min_limit_coefficients: [0.0; 5],
            max_limit_coefficients: [0.0; 5],
        }
    }

    /// Set the joint limits.
    pub fn with_limits(mut self, limits: JointLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Compute the link length for given body parameters.
    pub fn effective_link_length(&self, params: &BodyParameters) -> f64 {
        if !self.parameter_dependent {
            return self.link_length;
        }
        let p = params.to_array();
        let mut length = self.link_length;
        for i in 0..5 {
            length += self.length_coefficients[i] * p[i];
        }
        length.max(0.01)
    }

    /// Compute the effective joint limits for given body parameters.
    pub fn effective_limits(&self, params: &BodyParameters) -> JointLimits {
        if !self.parameter_dependent {
            return self.limits;
        }
        let p = params.to_array();
        let mut min = self.limits.min;
        let mut max = self.limits.max;
        for i in 0..5 {
            min += self.min_limit_coefficients[i] * p[i];
            max += self.max_limit_coefficients[i] * p[i];
        }
        JointLimits {
            min,
            max,
            max_velocity: self.limits.max_velocity,
            max_acceleration: self.limits.max_acceleration,
        }
    }
}

/// A kinematic chain representing a human arm (7 DOF).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KinematicChain {
    /// Chain name.
    pub name: String,
    /// Joints in the chain (ordered from base to tip).
    pub joints: Vec<Joint>,
    /// Base transform (shoulder position relative to torso).
    pub base_transform: [f64; 16],
    /// Whether this is the left or right arm.
    pub side: ArmSide,
    /// Coefficients for body-parameter-dependent base position.
    pub base_position_coefficients: [[f64; 5]; 3],
}

impl KinematicChain {
    /// Create a default 7-DOF arm chain.
    pub fn default_arm(side: ArmSide) -> Self {
        let sign = match side {
            ArmSide::Left => -1.0,
            ArmSide::Right => 1.0,
        };

        let joints = vec![
            Joint::revolute("shoulder_flexion", JointType::RevoluteX, 0.0)
                .with_limits(JointLimits::new(-crate::deg_to_rad(60.0), crate::deg_to_rad(180.0))),
            Joint::revolute("shoulder_abduction", JointType::RevoluteZ, 0.0)
                .with_limits(JointLimits::new(-crate::deg_to_rad(10.0), crate::deg_to_rad(180.0))),
            Joint::revolute("shoulder_rotation", JointType::RevoluteY, 0.366)
                .with_limits(JointLimits::new(-crate::deg_to_rad(90.0), crate::deg_to_rad(90.0))),
            Joint::revolute("elbow_flexion", JointType::RevoluteX, 0.268)
                .with_limits(JointLimits::new(0.0, crate::deg_to_rad(145.0))),
            Joint::revolute("wrist_pronation", JointType::RevoluteY, 0.0)
                .with_limits(JointLimits::new(-crate::deg_to_rad(80.0), crate::deg_to_rad(80.0))),
            Joint::revolute("wrist_flexion", JointType::RevoluteX, 0.0)
                .with_limits(JointLimits::new(-crate::deg_to_rad(70.0), crate::deg_to_rad(70.0))),
            Joint::revolute("wrist_deviation", JointType::RevoluteZ, 0.193)
                .with_limits(JointLimits::new(-crate::deg_to_rad(20.0), crate::deg_to_rad(30.0))),
        ];

        Self {
            name: format!("{:?} Arm", side),
            joints,
            base_transform: [
                1.0, 0.0, 0.0, sign * 0.24,
                0.0, 1.0, 0.0, 1.435,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
            side,
            base_position_coefficients: [
                [0.0, 0.0, sign * 0.5, 0.0, 0.0],  // x depends on shoulder_breadth
                [0.818, 0.0, 0.0, 0.0, 0.0],        // y depends on stature
                [0.0; 5],                             // z
            ],
        }
    }

    /// Total degrees of freedom.
    pub fn total_dof(&self) -> usize {
        self.joints.iter().map(|j| j.joint_type.dof()).sum()
    }

    /// Total link length sum.
    pub fn total_link_length(&self, params: &BodyParameters) -> f64 {
        self.joints
            .iter()
            .map(|j| j.effective_link_length(params))
            .sum()
    }

    /// Get the base position for given body parameters.
    pub fn base_position(&self, params: &BodyParameters) -> [f64; 3] {
        let p = params.to_array();
        let mut pos = [
            self.base_transform[3],
            self.base_transform[7],
            self.base_transform[11],
        ];
        for axis in 0..3 {
            for i in 0..5 {
                pos[axis] += self.base_position_coefficients[axis][i] * p[i];
            }
        }
        pos
    }

    /// Check if a joint angle configuration is within limits.
    pub fn is_within_limits(&self, angles: &[f64], params: &BodyParameters) -> bool {
        for (i, joint) in self.joints.iter().enumerate() {
            if i >= angles.len() {
                break;
            }
            let limits = joint.effective_limits(params);
            if !limits.contains(angles[i]) {
                return false;
            }
        }
        true
    }

    /// Clamp joint angles to limits.
    pub fn clamp_to_limits(&self, angles: &[f64], params: &BodyParameters) -> Vec<f64> {
        angles
            .iter()
            .enumerate()
            .map(|(i, &a)| {
                if i < self.joints.len() {
                    self.joints[i].effective_limits(params).clamp(a)
                } else {
                    a
                }
            })
            .collect()
    }

    /// Get the midpoint configuration.
    pub fn midpoint_config(&self, params: &BodyParameters) -> Vec<f64> {
        self.joints
            .iter()
            .map(|j| j.effective_limits(params).midpoint())
            .collect()
    }

    /// Random configuration within limits.
    pub fn random_config(&self, params: &BodyParameters, rng: &mut impl rand::Rng) -> Vec<f64> {
        use rand::Rng;
        self.joints
            .iter()
            .map(|j| {
                let limits = j.effective_limits(params);
                rng.gen_range(limits.min..=limits.max)
            })
            .collect()
    }
}

/// Which arm side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArmSide {
    Left,
    Right,
}

/// A full body model with both arms and torso parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullBodyModel {
    /// Body parameters.
    pub params: BodyParameters,
    /// Left arm kinematic chain.
    pub left_arm: KinematicChain,
    /// Right arm kinematic chain.
    pub right_arm: KinematicChain,
    /// Torso position (hip center in world space).
    pub torso_position: [f64; 3],
    /// Torso orientation as quaternion.
    pub torso_orientation: [f64; 4],
    /// Whether the user is seated.
    pub seated: bool,
    /// Seat height (if seated) in meters.
    pub seat_height: f64,
    /// Eye height (computed from stature and posture).
    pub eye_height: f64,
}

impl FullBodyModel {
    /// Create a standing body model.
    pub fn standing(params: BodyParameters) -> Self {
        let eye_height = params.stature * 0.936;
        Self {
            left_arm: KinematicChain::default_arm(ArmSide::Left),
            right_arm: KinematicChain::default_arm(ArmSide::Right),
            torso_position: [0.0, 0.0, 0.0],
            torso_orientation: [1.0, 0.0, 0.0, 0.0],
            seated: false,
            seat_height: 0.0,
            eye_height,
            params,
        }
    }

    /// Create a seated body model.
    pub fn seated(params: BodyParameters, seat_height: f64) -> Self {
        let sitting_height = params.stature * 0.52;
        let eye_height = seat_height + sitting_height * 0.936;
        Self {
            left_arm: KinematicChain::default_arm(ArmSide::Left),
            right_arm: KinematicChain::default_arm(ArmSide::Right),
            torso_position: [0.0, seat_height, 0.0],
            torso_orientation: [1.0, 0.0, 0.0, 0.0],
            seated: true,
            seat_height,
            eye_height,
            params,
        }
    }

    /// Get the arm for a given side.
    pub fn arm(&self, side: ArmSide) -> &KinematicChain {
        match side {
            ArmSide::Left => &self.left_arm,
            ArmSide::Right => &self.right_arm,
        }
    }

    /// Get the shoulder position for a given side.
    pub fn shoulder_position(&self, side: ArmSide) -> [f64; 3] {
        let arm = self.arm(side);
        let base = arm.base_position(&self.params);
        [
            base[0] + self.torso_position[0],
            base[1] + self.torso_position[1],
            base[2] + self.torso_position[2],
        ]
    }
}

/// Body parameter ranges for population coverage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodyParameterRange {
    /// Minimum values per parameter.
    pub min: BodyParameters,
    /// Maximum values per parameter.
    pub max: BodyParameters,
    /// Mean values per parameter.
    pub mean: BodyParameters,
    /// Standard deviation per parameter.
    pub std_dev: [f64; 5],
    /// Correlation matrix (5x5, row-major).
    pub correlation: [f64; 25],
    /// Population percentile range (e.g., 5th to 95th).
    pub percentile_low: f64,
    pub percentile_high: f64,
}

impl Default for BodyParameterRange {
    fn default() -> Self {
        Self {
            min: BodyParameters::small_female(),
            max: BodyParameters::large_male(),
            mean: BodyParameters::average_male(),
            std_dev: [0.074, 0.021, 0.026, 0.017, 0.011],
            correlation: {
                let mut c = [0.0; 25];
                for i in 0..5 { c[i * 5 + i] = 1.0; }
                c[0 * 5 + 1] = 0.85; c[1 * 5 + 0] = 0.85;  // stature-arm
                c[0 * 5 + 2] = 0.60; c[2 * 5 + 0] = 0.60;  // stature-shoulder
                c[0 * 5 + 3] = 0.82; c[3 * 5 + 0] = 0.82;  // stature-forearm
                c[0 * 5 + 4] = 0.72; c[4 * 5 + 0] = 0.72;  // stature-hand
                c[1 * 5 + 3] = 0.78; c[3 * 5 + 1] = 0.78;  // arm-forearm
                c[1 * 5 + 4] = 0.65; c[4 * 5 + 1] = 0.65;  // arm-hand
                c[2 * 5 + 3] = 0.55; c[3 * 5 + 2] = 0.55;  // shoulder-forearm
                c
            },
            percentile_low: 0.05,
            percentile_high: 0.95,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_body_params_creation() {
        let params = BodyParameters::average_male();
        assert!((params.stature - 1.756).abs() < 1e-6);
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_body_params_reach() {
        let male = BodyParameters::average_male();
        let female = BodyParameters::average_female();
        assert!(male.total_reach() > female.total_reach());
    }

    #[test]
    fn test_kinematic_chain() {
        let chain = KinematicChain::default_arm(ArmSide::Right);
        assert_eq!(chain.total_dof(), 7);
        let params = BodyParameters::average_male();
        assert!(chain.total_link_length(&params) > 0.5);
    }

    #[test]
    fn test_joint_limits() {
        let limits = JointLimits::new(-1.0, 1.0);
        assert!(limits.contains(0.0));
        assert!(!limits.contains(1.5));
        assert_eq!(limits.clamp(1.5), 1.0);
        assert!((limits.midpoint() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_body_params_lerp() {
        let a = BodyParameters::small_female();
        let b = BodyParameters::large_male();
        let mid = a.lerp(&b, 0.5);
        assert!((mid.stature - (a.stature + b.stature) / 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_full_body_model() {
        let params = BodyParameters::average_male();
        let model = FullBodyModel::standing(params);
        assert!(!model.seated);
        assert!(model.eye_height > 1.5);

        let seated = FullBodyModel::seated(params, 0.45);
        assert!(seated.seated);
    }

    #[test]
    fn test_body_params_validation() {
        let mut params = BodyParameters::average_male();
        assert!(params.validate().is_ok());
        params.stature = 3.0;
        assert!(params.validate().is_err());
    }
}

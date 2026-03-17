//! Body pose estimation, smoothing, and spatial queries (reach, pointing, etc.).

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Math helpers (local)
// ---------------------------------------------------------------------------

fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn sub(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn mag(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn normalize(v: &[f64; 3]) -> [f64; 3] {
    let len = mag(v);
    if len < 1e-12 {
        return [0.0, 0.0, 0.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

fn lerp3(a: &[f64; 3], b: &[f64; 3], t: f64) -> [f64; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

fn lerp4(a: &[f64; 4], b: &[f64; 4], t: f64) -> [f64; 4] {
    let mut out = [0.0; 4];
    for i in 0..4 {
        out[i] = a[i] + (b[i] - a[i]) * t;
    }
    // Renormalise.
    let len = (out[0] * out[0] + out[1] * out[1] + out[2] * out[2] + out[3] * out[3]).sqrt();
    if len > 1e-12 {
        for v in &mut out {
            *v /= len;
        }
    }
    out
}

/// Forward direction from a quaternion `[x, y, z, w]` (rotate `[0, 0, -1]`).
fn quat_forward(q: &[f64; 4]) -> [f64; 3] {
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    // Rotate [0, 0, -1] by quaternion.
    let fx = -2.0 * (x * z + w * y);
    let fy = -2.0 * (y * z - w * x);
    let fz = -(1.0 - 2.0 * (x * x + y * y));
    normalize(&[fx, fy, fz])
}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Transform of one hand (position, rotation, confidence).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandTransform {
    pub position: [f64; 3],
    /// Quaternion `[x, y, z, w]`.
    pub rotation: [f64; 4],
    pub confidence: f64,
}

impl HandTransform {
    pub fn new(position: [f64; 3], rotation: [f64; 4], confidence: f64) -> Self {
        Self {
            position,
            rotation,
            confidence,
        }
    }

    /// Identity rotation at position `[0,0,0]`.
    pub fn identity() -> Self {
        Self {
            position: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            confidence: 0.0,
        }
    }

    /// Forward direction derived from the hand rotation.
    pub fn forward(&self) -> [f64; 3] {
        quat_forward(&self.rotation)
    }
}

/// Full body pose for a single frame (head + both hands).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodyPose {
    pub head_position: [f64; 3],
    /// Quaternion `[x, y, z, w]`.
    pub head_rotation: [f64; 4],
    pub left_hand: HandTransform,
    pub right_hand: HandTransform,
}

impl BodyPose {
    /// A default T-pose at the origin.
    pub fn default_pose() -> Self {
        Self {
            head_position: [0.0, 1.7, 0.0],
            head_rotation: [0.0, 0.0, 0.0, 1.0],
            left_hand: HandTransform::new([-0.4, 1.0, -0.3], [0.0, 0.0, 0.0, 1.0], 1.0),
            right_hand: HandTransform::new([0.4, 1.0, -0.3], [0.0, 0.0, 0.0, 1.0], 1.0),
        }
    }
}

/// Timestamped body pose with confidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoseEstimate {
    pub body_pose: BodyPose,
    pub confidence: f64,
    pub timestamp: f64,
}

// ---------------------------------------------------------------------------
// Pose queries
// ---------------------------------------------------------------------------

/// Maximum distance either hand is from the head.
pub fn compute_reach_extent(pose: &BodyPose) -> f64 {
    let dl = dist(&pose.head_position, &pose.left_hand.position);
    let dr = dist(&pose.head_position, &pose.right_hand.position);
    dl.max(dr)
}

/// Arm extension factor for the specified side.
///
/// Returns a value in `[0, 1]` where `0` means the hand is at the head
/// position and `1` means the hand is at `max_arm_length` distance.
/// The default `max_arm_length` is 0.75 m.
pub fn compute_arm_extension(pose: &BodyPose, side: &str) -> f64 {
    const MAX_ARM_LENGTH: f64 = 0.75;
    let hand = match side {
        "left" => &pose.left_hand,
        "right" => &pose.right_hand,
        _ => &pose.right_hand,
    };
    let d = dist(&pose.head_position, &hand.position);
    (d / MAX_ARM_LENGTH).clamp(0.0, 1.0)
}

/// Returns `true` if either hand's forward direction points at `target`
/// within `tolerance` radians.
pub fn is_pointing_at(pose: &BodyPose, target: &[f64; 3], tolerance: f64) -> bool {
    for hand in [&pose.left_hand, &pose.right_hand] {
        if hand.confidence < 0.1 {
            continue;
        }
        let to_target = normalize(&sub(target, &hand.position));
        let fwd = hand.forward();
        let cos_angle = dot(&to_target, &fwd);
        let angle = cos_angle.clamp(-1.0, 1.0).acos();
        if angle <= tolerance {
            return true;
        }
    }
    false
}

/// Forward direction derived from the head rotation quaternion.
pub fn compute_head_direction(pose: &BodyPose) -> [f64; 3] {
    quat_forward(&pose.head_rotation)
}

/// Linearly interpolate between two body poses. `t` in `[0, 1]`.
pub fn interpolate_poses(a: &BodyPose, b: &BodyPose, t: f64) -> BodyPose {
    let t = t.clamp(0.0, 1.0);
    BodyPose {
        head_position: lerp3(&a.head_position, &b.head_position, t),
        head_rotation: lerp4(&a.head_rotation, &b.head_rotation, t),
        left_hand: HandTransform {
            position: lerp3(&a.left_hand.position, &b.left_hand.position, t),
            rotation: lerp4(&a.left_hand.rotation, &b.left_hand.rotation, t),
            confidence: a.left_hand.confidence + (b.left_hand.confidence - a.left_hand.confidence) * t,
        },
        right_hand: HandTransform {
            position: lerp3(&a.right_hand.position, &b.right_hand.position, t),
            rotation: lerp4(&a.right_hand.rotation, &b.right_hand.rotation, t),
            confidence: a.right_hand.confidence
                + (b.right_hand.confidence - a.right_hand.confidence) * t,
        },
    }
}

// ---------------------------------------------------------------------------
// PoseSmoother
// ---------------------------------------------------------------------------

/// Applies exponential smoothing over a rolling window of pose estimates.
#[derive(Debug, Clone)]
pub struct PoseSmoother {
    history: VecDeque<PoseEstimate>,
    max_history: usize,
    alpha: f64,
    current: Option<SmoothedState>,
}

#[derive(Debug, Clone)]
struct SmoothedState {
    head_position: [f64; 3],
    head_rotation: [f64; 4],
    left_position: [f64; 3],
    left_rotation: [f64; 4],
    right_position: [f64; 3],
    right_rotation: [f64; 4],
}

impl PoseSmoother {
    /// Create a new smoother.
    ///
    /// * `alpha` — blending factor in `(0, 1]`; lower = more smoothing.
    /// * `max_history` — how many estimates to keep for diagnostics.
    pub fn new(alpha: f64, max_history: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(max_history),
            max_history: max_history.max(1),
            alpha: alpha.clamp(0.01, 1.0),
            current: None,
        }
    }

    /// Feed a new estimate and return the smoothed result.
    pub fn smooth(&mut self, estimate: &PoseEstimate) -> PoseEstimate {
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(estimate.clone());

        let bp = &estimate.body_pose;
        let state = match &self.current {
            None => SmoothedState {
                head_position: bp.head_position,
                head_rotation: bp.head_rotation,
                left_position: bp.left_hand.position,
                left_rotation: bp.left_hand.rotation,
                right_position: bp.right_hand.position,
                right_rotation: bp.right_hand.rotation,
            },
            Some(prev) => SmoothedState {
                head_position: lerp3(&prev.head_position, &bp.head_position, self.alpha),
                head_rotation: lerp4(&prev.head_rotation, &bp.head_rotation, self.alpha),
                left_position: lerp3(&prev.left_position, &bp.left_hand.position, self.alpha),
                left_rotation: lerp4(&prev.left_rotation, &bp.left_hand.rotation, self.alpha),
                right_position: lerp3(&prev.right_position, &bp.right_hand.position, self.alpha),
                right_rotation: lerp4(&prev.right_rotation, &bp.right_hand.rotation, self.alpha),
            },
        };

        self.current = Some(state.clone());

        PoseEstimate {
            body_pose: BodyPose {
                head_position: state.head_position,
                head_rotation: state.head_rotation,
                left_hand: HandTransform {
                    position: state.left_position,
                    rotation: state.left_rotation,
                    confidence: bp.left_hand.confidence,
                },
                right_hand: HandTransform {
                    position: state.right_position,
                    rotation: state.right_rotation,
                    confidence: bp.right_hand.confidence,
                },
            },
            confidence: estimate.confidence,
            timestamp: estimate.timestamp,
        }
    }

    /// How many estimates are stored.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Reset smoother state.
    pub fn reset(&mut self) {
        self.history.clear();
        self.current = None;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_estimate(t: f64) -> PoseEstimate {
        PoseEstimate {
            body_pose: BodyPose::default_pose(),
            confidence: 1.0,
            timestamp: t,
        }
    }

    // -- HandTransform --

    #[test]
    fn test_hand_transform_identity_forward() {
        let h = HandTransform::identity();
        let fwd = h.forward();
        // Identity quaternion → forward is [0,0,-1].
        assert!((fwd[2] - (-1.0)).abs() < 1e-9);
    }

    // -- BodyPose --

    #[test]
    fn test_default_pose_head_height() {
        let pose = BodyPose::default_pose();
        assert!((pose.head_position[1] - 1.7).abs() < 1e-9);
    }

    // -- Reach / extension --

    #[test]
    fn test_reach_extent() {
        let pose = BodyPose::default_pose();
        let reach = compute_reach_extent(&pose);
        assert!(reach > 0.0);
    }

    #[test]
    fn test_arm_extension_at_head() {
        let mut pose = BodyPose::default_pose();
        pose.right_hand.position = pose.head_position;
        let ext = compute_arm_extension(&pose, "right");
        assert!(ext < 0.01, "Hand at head should be ~0 extension");
    }

    #[test]
    fn test_arm_extension_fully_extended() {
        let mut pose = BodyPose::default_pose();
        pose.right_hand.position = [
            pose.head_position[0],
            pose.head_position[1],
            pose.head_position[2] - 0.75,
        ];
        let ext = compute_arm_extension(&pose, "right");
        assert!((ext - 1.0).abs() < 0.01, "Should be fully extended");
    }

    // -- Pointing --

    #[test]
    fn test_is_pointing_at_target_in_front() {
        let pose = BodyPose::default_pose();
        // Target straight in front of the right hand.
        let target = [
            pose.right_hand.position[0],
            pose.right_hand.position[1],
            pose.right_hand.position[2] - 2.0,
        ];
        assert!(is_pointing_at(&pose, &target, 0.5));
    }

    #[test]
    fn test_is_not_pointing_behind() {
        let pose = BodyPose::default_pose();
        let target = [0.0, 1.0, 5.0]; // behind
        assert!(!is_pointing_at(&pose, &target, 0.3));
    }

    // -- Head direction --

    #[test]
    fn test_head_direction_identity() {
        let pose = BodyPose::default_pose();
        let dir = compute_head_direction(&pose);
        assert!((dir[2] - (-1.0)).abs() < 1e-9);
    }

    // -- Interpolation --

    #[test]
    fn test_interpolate_endpoints() {
        let a = BodyPose::default_pose();
        let mut b = BodyPose::default_pose();
        b.head_position = [0.0, 2.0, 0.0];
        let at0 = interpolate_poses(&a, &b, 0.0);
        let at1 = interpolate_poses(&a, &b, 1.0);
        assert!((at0.head_position[1] - 1.7).abs() < 1e-9);
        assert!((at1.head_position[1] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_interpolate_midpoint() {
        let a = BodyPose::default_pose();
        let mut b = BodyPose::default_pose();
        b.head_position = [0.0, 2.7, 0.0];
        let mid = interpolate_poses(&a, &b, 0.5);
        let expected = (1.7 + 2.7) / 2.0;
        assert!((mid.head_position[1] - expected).abs() < 1e-9);
    }

    // -- PoseSmoother --

    #[test]
    fn test_smoother_first_sample_passes_through() {
        let mut smoother = PoseSmoother::new(0.5, 10);
        let est = default_estimate(0.0);
        let smoothed = smoother.smooth(&est);
        assert!((smoothed.body_pose.head_position[1] - 1.7).abs() < 1e-9);
    }

    #[test]
    fn test_smoother_converges() {
        let mut smoother = PoseSmoother::new(0.5, 20);
        let mut est = default_estimate(0.0);
        // First feed the default, then shift head.
        smoother.smooth(&est);
        est.body_pose.head_position = [0.0, 2.0, 0.0];
        let mut last = smoother.smooth(&est);
        for i in 2..50 {
            est.timestamp = i as f64;
            last = smoother.smooth(&est);
        }
        assert!(
            (last.body_pose.head_position[1] - 2.0).abs() < 0.01,
            "Should converge to 2.0"
        );
    }

    #[test]
    fn test_smoother_history_limit() {
        let mut smoother = PoseSmoother::new(0.5, 5);
        for i in 0..10 {
            smoother.smooth(&default_estimate(i as f64));
        }
        assert_eq!(smoother.history_len(), 5);
    }

    #[test]
    fn test_smoother_reset() {
        let mut smoother = PoseSmoother::new(0.5, 10);
        smoother.smooth(&default_estimate(0.0));
        smoother.reset();
        assert_eq!(smoother.history_len(), 0);
        assert!(smoother.current.is_none());
    }

    // -- quat_forward --

    #[test]
    fn test_quat_forward_identity() {
        let fwd = quat_forward(&[0.0, 0.0, 0.0, 1.0]);
        assert!((fwd[2] - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn test_quat_forward_180_yaw() {
        // 180° around Y: q = (0, sin(90°), 0, cos(90°)) = (0, 1, 0, 0)
        let fwd = quat_forward(&[0.0, 1.0, 0.0, 0.0]);
        // Should point [0, 0, 1] (backward).
        assert!((fwd[2] - 1.0).abs() < 1e-9);
    }
}

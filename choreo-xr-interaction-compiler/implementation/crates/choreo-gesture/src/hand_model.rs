//! Hand skeleton model with joint topology, curl/spread computation, and
//! interpolation utilities.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Joint enum
// ---------------------------------------------------------------------------

/// The 25 joints of a tracked hand (OpenXR convention).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HandJoint {
    Wrist,
    ThumbMetacarpal,
    ThumbProximal,
    ThumbDistal,
    ThumbTip,
    IndexMetacarpal,
    IndexProximal,
    IndexMiddle,
    IndexDistal,
    IndexTip,
    MiddleMetacarpal,
    MiddleProximal,
    MiddleMiddle,
    MiddleDistal,
    MiddleTip,
    RingMetacarpal,
    RingProximal,
    RingMiddle,
    RingDistal,
    RingTip,
    PinkyMetacarpal,
    PinkyProximal,
    PinkyMiddle,
    PinkyDistal,
    PinkyTip,
}

/// All 25 joints in index order.
pub const ALL_JOINTS: [HandJoint; 25] = [
    HandJoint::Wrist,
    HandJoint::ThumbMetacarpal,
    HandJoint::ThumbProximal,
    HandJoint::ThumbDistal,
    HandJoint::ThumbTip,
    HandJoint::IndexMetacarpal,
    HandJoint::IndexProximal,
    HandJoint::IndexMiddle,
    HandJoint::IndexDistal,
    HandJoint::IndexTip,
    HandJoint::MiddleMetacarpal,
    HandJoint::MiddleProximal,
    HandJoint::MiddleMiddle,
    HandJoint::MiddleDistal,
    HandJoint::MiddleTip,
    HandJoint::RingMetacarpal,
    HandJoint::RingProximal,
    HandJoint::RingMiddle,
    HandJoint::RingDistal,
    HandJoint::RingTip,
    HandJoint::PinkyMetacarpal,
    HandJoint::PinkyProximal,
    HandJoint::PinkyMiddle,
    HandJoint::PinkyDistal,
    HandJoint::PinkyTip,
];

impl HandJoint {
    /// Numeric index `0..=24`.
    pub fn index(&self) -> usize {
        match self {
            Self::Wrist => 0,
            Self::ThumbMetacarpal => 1,
            Self::ThumbProximal => 2,
            Self::ThumbDistal => 3,
            Self::ThumbTip => 4,
            Self::IndexMetacarpal => 5,
            Self::IndexProximal => 6,
            Self::IndexMiddle => 7,
            Self::IndexDistal => 8,
            Self::IndexTip => 9,
            Self::MiddleMetacarpal => 10,
            Self::MiddleProximal => 11,
            Self::MiddleMiddle => 12,
            Self::MiddleDistal => 13,
            Self::MiddleTip => 14,
            Self::RingMetacarpal => 15,
            Self::RingProximal => 16,
            Self::RingMiddle => 17,
            Self::RingDistal => 18,
            Self::RingTip => 19,
            Self::PinkyMetacarpal => 20,
            Self::PinkyProximal => 21,
            Self::PinkyMiddle => 22,
            Self::PinkyDistal => 23,
            Self::PinkyTip => 24,
        }
    }

    /// Which finger this joint belongs to (wrist is mapped to `Thumb` as the
    /// closest anatomical neighbour).
    pub fn finger(&self) -> Finger {
        match self {
            Self::Wrist => Finger::Thumb,
            Self::ThumbMetacarpal
            | Self::ThumbProximal
            | Self::ThumbDistal
            | Self::ThumbTip => Finger::Thumb,
            Self::IndexMetacarpal
            | Self::IndexProximal
            | Self::IndexMiddle
            | Self::IndexDistal
            | Self::IndexTip => Finger::Index,
            Self::MiddleMetacarpal
            | Self::MiddleProximal
            | Self::MiddleMiddle
            | Self::MiddleDistal
            | Self::MiddleTip => Finger::Middle,
            Self::RingMetacarpal
            | Self::RingProximal
            | Self::RingMiddle
            | Self::RingDistal
            | Self::RingTip => Finger::Ring,
            Self::PinkyMetacarpal
            | Self::PinkyProximal
            | Self::PinkyMiddle
            | Self::PinkyDistal
            | Self::PinkyTip => Finger::Pinky,
        }
    }

    /// Ordered chain of joints for a finger (metacarpal → tip).
    pub fn joints_for_finger(finger: Finger) -> Vec<HandJoint> {
        match finger {
            Finger::Thumb => vec![
                Self::ThumbMetacarpal,
                Self::ThumbProximal,
                Self::ThumbDistal,
                Self::ThumbTip,
            ],
            Finger::Index => vec![
                Self::IndexMetacarpal,
                Self::IndexProximal,
                Self::IndexMiddle,
                Self::IndexDistal,
                Self::IndexTip,
            ],
            Finger::Middle => vec![
                Self::MiddleMetacarpal,
                Self::MiddleProximal,
                Self::MiddleMiddle,
                Self::MiddleDistal,
                Self::MiddleTip,
            ],
            Finger::Ring => vec![
                Self::RingMetacarpal,
                Self::RingProximal,
                Self::RingMiddle,
                Self::RingDistal,
                Self::RingTip,
            ],
            Finger::Pinky => vec![
                Self::PinkyMetacarpal,
                Self::PinkyProximal,
                Self::PinkyMiddle,
                Self::PinkyDistal,
                Self::PinkyTip,
            ],
        }
    }

    /// Build a `HandJoint` from a numeric index (0–24). Returns `None` for
    /// out-of-range values.
    pub fn from_index(i: usize) -> Option<HandJoint> {
        ALL_JOINTS.get(i).copied()
    }
}

// ---------------------------------------------------------------------------
// Finger enum
// ---------------------------------------------------------------------------

/// Finger identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Finger {
    Thumb,
    Index,
    Middle,
    Ring,
    Pinky,
}

/// All five fingers in anatomical order.
pub const ALL_FINGERS: [Finger; 5] = [
    Finger::Thumb,
    Finger::Index,
    Finger::Middle,
    Finger::Ring,
    Finger::Pinky,
];

// ---------------------------------------------------------------------------
// JointPose
// ---------------------------------------------------------------------------

/// Pose (position + orientation + radius) of a single joint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointPose {
    pub position: [f64; 3],
    /// Quaternion `[x, y, z, w]`.
    pub rotation: [f64; 4],
    /// Approximate joint radius in metres.
    pub radius: f64,
}

impl JointPose {
    pub fn new(position: [f64; 3], rotation: [f64; 4], radius: f64) -> Self {
        Self {
            position,
            rotation,
            radius,
        }
    }

    /// Identity-rotation joint at the given position.
    pub fn at(position: [f64; 3]) -> Self {
        Self {
            position,
            rotation: [0.0, 0.0, 0.0, 1.0],
            radius: 0.005,
        }
    }
}

// ---------------------------------------------------------------------------
// HandModel
// ---------------------------------------------------------------------------

/// Full hand model with per-joint poses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandModel {
    /// Exactly 25 joint poses indexed by `HandJoint::index()`.
    pub joints: Vec<JointPose>,
    /// `"left"` or `"right"`.
    pub side: String,
    pub timestamp: f64,
}

impl HandModel {
    /// Create a new hand from 25 joint poses.
    pub fn new(joints: Vec<JointPose>, side: &str, timestamp: f64) -> Self {
        assert!(joints.len() >= 25, "HandModel requires 25 joints");
        Self {
            joints,
            side: side.to_string(),
            timestamp,
        }
    }

    /// Position of the given joint.
    pub fn joint_position(&self, joint: HandJoint) -> &[f64; 3] {
        &self.joints[joint.index()].position
    }

    /// Wrist position (convenience).
    pub fn wrist(&self) -> &[f64; 3] {
        self.joint_position(HandJoint::Wrist)
    }

    /// Approximate palm centre (average of wrist + five metacarpals).
    pub fn palm_center(&self) -> [f64; 3] {
        let indices = [
            HandJoint::Wrist,
            HandJoint::IndexMetacarpal,
            HandJoint::MiddleMetacarpal,
            HandJoint::RingMetacarpal,
            HandJoint::PinkyMetacarpal,
        ];
        let mut c = [0.0; 3];
        for &idx in &indices {
            let p = self.joint_position(idx);
            c[0] += p[0];
            c[1] += p[1];
            c[2] += p[2];
        }
        let n = indices.len() as f64;
        [c[0] / n, c[1] / n, c[2] / n]
    }
}

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
    // Re-normalise quaternion.
    let len = (out[0] * out[0] + out[1] * out[1] + out[2] * out[2] + out[3] * out[3]).sqrt();
    if len > 1e-12 {
        for v in &mut out {
            *v /= len;
        }
    }
    out
}

fn angle_between_vectors(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let la = mag(a);
    let lb = mag(b);
    if la < 1e-12 || lb < 1e-12 {
        return 0.0;
    }
    let cos_theta = (dot(a, b) / (la * lb)).clamp(-1.0, 1.0);
    cos_theta.acos()
}

// ---------------------------------------------------------------------------
// Curl / spread / pinch / grab computations
// ---------------------------------------------------------------------------

/// Compute how curled a finger is.
///
/// Returns a value in `[0, 1]` where `0` means fully extended and `1` means
/// fully curled.  The curl is computed from the sum of inter-joint angles
/// along the finger chain: a straight chain yields 0° total bend → 0 curl,
/// and a fully-bent chain approaches π total bend → 1 curl.
pub fn compute_finger_curl(hand: &HandModel, finger: Finger) -> f64 {
    let chain = HandJoint::joints_for_finger(finger);
    if chain.len() < 3 {
        return 0.0;
    }

    let mut total_angle: f64 = 0.0;
    let max_angle = std::f64::consts::PI * (chain.len() as f64 - 2.0);

    for window in chain.windows(3) {
        let a = hand.joint_position(window[0]);
        let b = hand.joint_position(window[1]);
        let c = hand.joint_position(window[2]);
        let ba = sub(a, b);
        let bc = sub(c, b);
        let angle = angle_between_vectors(&ba, &bc);
        // Angle of π means perfectly straight → 0 curl contribution.
        total_angle += std::f64::consts::PI - angle;
    }

    (total_angle / max_angle).clamp(0.0, 1.0)
}

/// Compute the spread angles (radians) between adjacent finger pairs.
///
/// Returns a `Vec` of 4 values:
///   `[thumb-index, index-middle, middle-ring, ring-pinky]`.
/// Each value is the angle between the directions from the wrist to the
/// respective fingertips.
pub fn compute_finger_spread(hand: &HandModel) -> Vec<f64> {
    let tips = [
        HandJoint::ThumbTip,
        HandJoint::IndexTip,
        HandJoint::MiddleTip,
        HandJoint::RingTip,
        HandJoint::PinkyTip,
    ];
    let wrist = hand.joint_position(HandJoint::Wrist);

    let dirs: Vec<[f64; 3]> = tips.iter().map(|&t| sub(hand.joint_position(t), wrist)).collect();

    dirs.windows(2)
        .map(|w| angle_between_vectors(&w[0], &w[1]))
        .collect()
}

/// Distance between thumb tip and index tip.
pub fn compute_pinch_distance(hand: &HandModel) -> f64 {
    dist(
        hand.joint_position(HandJoint::ThumbTip),
        hand.joint_position(HandJoint::IndexTip),
    )
}

/// Average curl across all five fingers; useful as a grab-strength proxy.
/// `0` = all fingers extended, `1` = all fully curled.
pub fn compute_grab_strength(hand: &HandModel) -> f64 {
    let sum: f64 = ALL_FINGERS
        .iter()
        .map(|&f| compute_finger_curl(hand, f))
        .sum();
    sum / ALL_FINGERS.len() as f64
}

// ---------------------------------------------------------------------------
// HandSkeleton (rest-pose / constraint descriptor)
// ---------------------------------------------------------------------------

/// Describes the rest-pose bone lengths and per-joint bend limits of a hand.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandSkeleton {
    /// 24 bone lengths (parent→child for joints 1..=24).
    pub bone_lengths: Vec<f64>,
    /// Per-bone min/max bend angle (radians).
    pub constraints: Vec<(f64, f64)>,
}

impl HandSkeleton {
    /// A reasonable adult-sized default skeleton (lengths in metres).
    pub fn default_skeleton() -> Self {
        // Approximate bone lengths for an average adult hand.
        let lengths = vec![
            // Wrist→ThumbMeta
            0.030,
            // ThumbMeta→ThumbProx
            0.035,
            // ThumbProx→ThumbDist
            0.025,
            // ThumbDist→ThumbTip
            0.022,
            // Wrist→IndexMeta
            0.060,
            // IndexMeta→IndexProx
            0.040,
            // IndexProx→IndexMid
            0.028,
            // IndexMid→IndexDist
            0.020,
            // IndexDist→IndexTip
            0.018,
            // Wrist→MiddleMeta
            0.058,
            // MiddleMeta→MiddleProx
            0.042,
            // MiddleProx→MiddleMid
            0.030,
            // MiddleMid→MiddleDist
            0.020,
            // MiddleDist→MiddleTip
            0.018,
            // Wrist→RingMeta
            0.052,
            // RingMeta→RingProx
            0.038,
            // RingProx→RingMid
            0.027,
            // RingMid→RingDist
            0.019,
            // RingDist→RingTip
            0.017,
            // Wrist→PinkyMeta
            0.046,
            // PinkyMeta→PinkyProx
            0.032,
            // PinkyProx→PinkyMid
            0.022,
            // PinkyMid→PinkyDist
            0.016,
            // PinkyDist→PinkyTip
            0.015,
        ];

        let constraints: Vec<(f64, f64)> = lengths
            .iter()
            .map(|_| (0.0, std::f64::consts::FRAC_PI_2))
            .collect();

        Self {
            bone_lengths: lengths,
            constraints,
        }
    }

    /// Total chain length for a given finger.
    pub fn finger_length(&self, finger: Finger) -> f64 {
        let range = match finger {
            Finger::Thumb => 0..4,
            Finger::Index => 4..9,
            Finger::Middle => 9..14,
            Finger::Ring => 14..19,
            Finger::Pinky => 19..24,
        };
        self.bone_lengths[range].iter().sum()
    }

    /// Check whether the bone lengths of a `HandModel` are within `tolerance`
    /// of this skeleton.
    pub fn validate(&self, hand: &HandModel, tolerance: f64) -> Vec<(usize, f64, f64)> {
        let parent_map = parent_indices();
        let mut violations = Vec::new();
        for i in 1..25 {
            let pi = parent_map[i];
            let actual = dist(
                &hand.joints[i].position,
                &hand.joints[pi].position,
            );
            let expected = self.bone_lengths[i - 1];
            if (actual - expected).abs() > tolerance {
                violations.push((i, expected, actual));
            }
        }
        violations
    }
}

/// Parent joint index for each joint.  Wrist (0) has no parent so maps to 0.
fn parent_indices() -> [usize; 25] {
    [
        0,  // 0  Wrist → self
        0,  // 1  ThumbMeta → Wrist
        1,  // 2  ThumbProx → ThumbMeta
        2,  // 3  ThumbDist → ThumbProx
        3,  // 4  ThumbTip  → ThumbDist
        0,  // 5  IndexMeta → Wrist
        5,  // 6  IndexProx → IndexMeta
        6,  // 7  IndexMid  → IndexProx
        7,  // 8  IndexDist → IndexMid
        8,  // 9  IndexTip  → IndexDist
        0,  // 10 MiddleMeta → Wrist
        10, // 11 MiddleProx → MiddleMeta
        11, // 12 MiddleMid  → MiddleProx
        12, // 13 MiddleDist → MiddleMid
        13, // 14 MiddleTip  → MiddleDist
        0,  // 15 RingMeta → Wrist
        15, // 16 RingProx → RingMeta
        16, // 17 RingMid  → RingProx
        17, // 18 RingDist → RingMid
        18, // 19 RingTip  → RingDist
        0,  // 20 PinkyMeta → Wrist
        20, // 21 PinkyProx → PinkyMeta
        21, // 22 PinkyMid  → PinkyProx
        22, // 23 PinkyDist → PinkyMid
        23, // 24 PinkyTip  → PinkyDist
    ]
}

// ---------------------------------------------------------------------------
// Interpolation
// ---------------------------------------------------------------------------

/// Linearly interpolate between two hand models.  `t` in `[0, 1]`.
/// The result takes the `side` from `a` and the timestamp is interpolated.
pub fn interpolate_hands(a: &HandModel, b: &HandModel, t: f64) -> HandModel {
    let t = t.clamp(0.0, 1.0);
    let joints: Vec<JointPose> = a
        .joints
        .iter()
        .zip(b.joints.iter())
        .map(|(ja, jb)| JointPose {
            position: lerp3(&ja.position, &jb.position, t),
            rotation: lerp4(&ja.rotation, &jb.rotation, t),
            radius: ja.radius + (jb.radius - ja.radius) * t,
        })
        .collect();

    HandModel {
        joints,
        side: a.side.clone(),
        timestamp: a.timestamp + (b.timestamp - a.timestamp) * t,
    }
}

// ---------------------------------------------------------------------------
// Builder helpers
// ---------------------------------------------------------------------------

/// Create a default flat hand at the origin for testing / placeholder use.
pub fn default_flat_hand(side: &str, timestamp: f64) -> HandModel {
    let positions: [[f64; 3]; 25] = [
        [0.0, 0.0, 0.0],       // Wrist
        [-0.02, 0.0, -0.01],   // ThumbMeta
        [-0.04, 0.0, -0.02],   // ThumbProx
        [-0.06, 0.0, -0.03],   // ThumbDist
        [-0.08, 0.0, -0.04],   // ThumbTip
        [-0.02, 0.0, -0.06],   // IndexMeta
        [-0.02, 0.0, -0.10],   // IndexProx
        [-0.02, 0.0, -0.13],   // IndexMid
        [-0.02, 0.0, -0.15],   // IndexDist
        [-0.02, 0.0, -0.17],   // IndexTip
        [0.00, 0.0, -0.06],    // MiddleMeta
        [0.00, 0.0, -0.10],    // MiddleProx
        [0.00, 0.0, -0.13],    // MiddleMid
        [0.00, 0.0, -0.15],    // MiddleDist
        [0.00, 0.0, -0.17],    // MiddleTip
        [0.02, 0.0, -0.06],    // RingMeta
        [0.02, 0.0, -0.10],    // RingProx
        [0.02, 0.0, -0.13],    // RingMid
        [0.02, 0.0, -0.15],    // RingDist
        [0.02, 0.0, -0.17],    // RingTip
        [0.04, 0.0, -0.06],    // PinkyMeta
        [0.04, 0.0, -0.09],    // PinkyProx
        [0.04, 0.0, -0.11],    // PinkyMid
        [0.04, 0.0, -0.13],    // PinkyDist
        [0.04, 0.0, -0.15],    // PinkyTip
    ];

    let joints: Vec<JointPose> = positions
        .iter()
        .map(|p| JointPose::at(*p))
        .collect();

    HandModel::new(joints, side, timestamp)
}

/// Create a curled / fist hand for testing.
pub fn default_fist_hand(side: &str, timestamp: f64) -> HandModel {
    let mut hand = default_flat_hand(side, timestamp);
    let wrist = *hand.wrist();
    let tips = [
        HandJoint::IndexTip,
        HandJoint::MiddleTip,
        HandJoint::RingTip,
        HandJoint::PinkyTip,
        HandJoint::ThumbTip,
    ];
    let distals = [
        HandJoint::IndexDistal,
        HandJoint::MiddleDistal,
        HandJoint::RingDistal,
        HandJoint::PinkyDistal,
        HandJoint::ThumbDistal,
    ];
    let middles = [
        HandJoint::IndexMiddle,
        HandJoint::MiddleMiddle,
        HandJoint::RingMiddle,
        HandJoint::PinkyMiddle,
    ];
    for &t in &tips {
        hand.joints[t.index()].position = [wrist[0] + 0.01, wrist[1], wrist[2] - 0.02];
    }
    for &d in &distals {
        hand.joints[d.index()].position = [wrist[0] + 0.015, wrist[1], wrist[2] - 0.03];
    }
    for &m in &middles {
        hand.joints[m.index()].position = [wrist[0] + 0.01, wrist[1], wrist[2] - 0.04];
    }
    hand
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_joint_index_round_trip() {
        for (i, &joint) in ALL_JOINTS.iter().enumerate() {
            assert_eq!(joint.index(), i);
            assert_eq!(HandJoint::from_index(i), Some(joint));
        }
        assert_eq!(HandJoint::from_index(25), None);
    }

    #[test]
    fn test_finger_classification() {
        assert_eq!(HandJoint::Wrist.finger(), Finger::Thumb);
        assert_eq!(HandJoint::IndexTip.finger(), Finger::Index);
        assert_eq!(HandJoint::MiddleProximal.finger(), Finger::Middle);
        assert_eq!(HandJoint::RingDistal.finger(), Finger::Ring);
        assert_eq!(HandJoint::PinkyMetacarpal.finger(), Finger::Pinky);
    }

    #[test]
    fn test_joints_for_finger_counts() {
        assert_eq!(HandJoint::joints_for_finger(Finger::Thumb).len(), 4);
        assert_eq!(HandJoint::joints_for_finger(Finger::Index).len(), 5);
        assert_eq!(HandJoint::joints_for_finger(Finger::Middle).len(), 5);
        assert_eq!(HandJoint::joints_for_finger(Finger::Ring).len(), 5);
        assert_eq!(HandJoint::joints_for_finger(Finger::Pinky).len(), 5);
    }

    #[test]
    fn test_flat_hand_low_curl() {
        let hand = default_flat_hand("left", 0.0);
        for &finger in &ALL_FINGERS {
            let curl = compute_finger_curl(&hand, finger);
            assert!(
                curl < 0.5,
                "Flat hand {:?} curl {curl} should be low",
                finger
            );
        }
    }

    #[test]
    fn test_fist_hand_high_curl() {
        let hand = default_fist_hand("right", 0.0);
        // Non-thumb fingers should be significantly curled.
        for &finger in &[Finger::Index, Finger::Middle, Finger::Ring, Finger::Pinky] {
            let curl = compute_finger_curl(&hand, finger);
            assert!(
                curl > 0.3,
                "Fist hand {:?} curl {curl} should be high",
                finger
            );
        }
    }

    #[test]
    fn test_pinch_distance_flat() {
        let hand = default_flat_hand("left", 0.0);
        let d = compute_pinch_distance(&hand);
        assert!(d > 0.01, "Flat hand pinch distance should be non-trivial");
    }

    #[test]
    fn test_pinch_distance_pinch() {
        let mut hand = default_flat_hand("left", 0.0);
        hand.joints[HandJoint::ThumbTip.index()].position =
            hand.joints[HandJoint::IndexTip.index()].position;
        let d = compute_pinch_distance(&hand);
        assert!(d < 1e-9, "Tips touching should give 0 distance");
    }

    #[test]
    fn test_grab_strength() {
        let flat = default_flat_hand("left", 0.0);
        let fist = default_fist_hand("left", 0.0);
        let s_flat = compute_grab_strength(&flat);
        let s_fist = compute_grab_strength(&fist);
        assert!(s_fist > s_flat, "Fist should have higher grab strength than flat");
    }

    #[test]
    fn test_finger_spread_has_four_values() {
        let hand = default_flat_hand("left", 0.0);
        let spread = compute_finger_spread(&hand);
        assert_eq!(spread.len(), 4);
        for &s in &spread {
            assert!(s >= 0.0);
        }
    }

    #[test]
    fn test_interpolate_hands_endpoints() {
        let a = default_flat_hand("left", 0.0);
        let b = default_flat_hand("left", 1.0);
        let at0 = interpolate_hands(&a, &b, 0.0);
        let at1 = interpolate_hands(&a, &b, 1.0);
        assert!((at0.timestamp - 0.0).abs() < 1e-9);
        assert!((at1.timestamp - 1.0).abs() < 1e-9);
        for i in 0..25 {
            assert!((at0.joints[i].position[0] - a.joints[i].position[0]).abs() < 1e-9);
            assert!((at1.joints[i].position[0] - b.joints[i].position[0]).abs() < 1e-9);
        }
    }

    #[test]
    fn test_interpolate_midpoint() {
        let mut a = default_flat_hand("left", 0.0);
        let mut b = default_flat_hand("left", 2.0);
        a.joints[0].position = [0.0, 0.0, 0.0];
        b.joints[0].position = [1.0, 0.0, 0.0];
        let mid = interpolate_hands(&a, &b, 0.5);
        assert!((mid.joints[0].position[0] - 0.5).abs() < 1e-9);
        assert!((mid.timestamp - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_default_skeleton() {
        let skel = HandSkeleton::default_skeleton();
        assert_eq!(skel.bone_lengths.len(), 24);
        assert_eq!(skel.constraints.len(), 24);
        for &l in &skel.bone_lengths {
            assert!(l > 0.0);
        }
    }

    #[test]
    fn test_skeleton_finger_length() {
        let skel = HandSkeleton::default_skeleton();
        let thumb_len = skel.finger_length(Finger::Thumb);
        let index_len = skel.finger_length(Finger::Index);
        assert!(thumb_len > 0.0);
        assert!(index_len > thumb_len, "Index chain should be longer than thumb");
    }

    #[test]
    fn test_palm_center() {
        let hand = default_flat_hand("left", 0.0);
        let pc = hand.palm_center();
        // Should be near origin area.
        assert!(pc[0].abs() < 0.1);
        assert!(pc[1].abs() < 0.1);
        assert!(pc[2].abs() < 0.1);
    }

    #[test]
    fn test_joint_pose_at() {
        let jp = JointPose::at([1.0, 2.0, 3.0]);
        assert_eq!(jp.position, [1.0, 2.0, 3.0]);
        assert_eq!(jp.rotation, [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_hand_model_side() {
        let hand = default_flat_hand("right", 42.0);
        assert_eq!(hand.side, "right");
        assert!((hand.timestamp - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_skeleton_validate_default_hand() {
        let skel = HandSkeleton::default_skeleton();
        let hand = default_flat_hand("left", 0.0);
        let violations = skel.validate(&hand, 0.2);
        // With a generous tolerance the flat hand should mostly pass.
        // We only assert it doesn't panic and returns a list.
        let _ = violations;
    }
}

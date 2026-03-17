//! WebXR Device API types for XR affordance verification.
//!
//! Models the WebXR Device API concepts relevant to accessibility verification
//! without depending on an external WebXR crate. These types bridge browser-side
//! WebXR sessions to the internal device configuration and verification pipeline.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::device::{
    ControllerType, DeviceConfig, DeviceConstraints, DeviceType, MovementMode, TrackingVolume,
};
use crate::geometry::BoundingBox;
use crate::scene::InteractionType;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// WebXR session mode (mirrors `XRSessionMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WebXrSessionMode {
    Inline,
    ImmersiveVr,
    ImmersiveAr,
}

impl WebXrSessionMode {
    /// Whether this mode provides an immersive experience.
    pub fn is_immersive(&self) -> bool {
        matches!(self, Self::ImmersiveVr | Self::ImmersiveAr)
    }
}

/// WebXR reference space type (mirrors `XRReferenceSpaceType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WebXrReferenceSpaceType {
    Viewer,
    Local,
    LocalFloor,
    BoundedFloor,
    Unbounded,
}

/// WebXR input-source handedness (mirrors `XRHandedness`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WebXrHandedness {
    #[serde(rename = "none")]
    None,
    #[serde(rename = "left")]
    Left,
    #[serde(rename = "right")]
    Right,
}

/// WebXR target-ray mode (mirrors `XRTargetRayMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WebXrTargetRayMode {
    Gaze,
    TrackedPointer,
    Screen,
    TransientPointer,
}

/// WebXR session visibility state (mirrors `XRVisibilityState`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WebXrVisibilityState {
    Visible,
    VisibleBlurred,
    Hidden,
}

/// WebXR eye identifier for stereo views.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WebXrEye {
    None,
    Left,
    Right,
}

/// WebXR hand joint (mirrors `XRHandJoint`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WebXrHandJoint {
    Wrist,
    ThumbMetacarpal,
    ThumbPhalanxProximal,
    ThumbPhalanxDistal,
    ThumbTip,
    IndexFingerMetacarpal,
    IndexFingerPhalanxProximal,
    IndexFingerPhalanxIntermediate,
    IndexFingerPhalanxDistal,
    IndexFingerTip,
    MiddleFingerMetacarpal,
    MiddleFingerPhalanxProximal,
    MiddleFingerPhalanxIntermediate,
    MiddleFingerPhalanxDistal,
    MiddleFingerTip,
    RingFingerMetacarpal,
    RingFingerPhalanxProximal,
    RingFingerPhalanxIntermediate,
    RingFingerPhalanxDistal,
    RingFingerTip,
    PinkyFingerMetacarpal,
    PinkyFingerPhalanxProximal,
    PinkyFingerPhalanxIntermediate,
    PinkyFingerPhalanxDistal,
    PinkyFingerTip,
}

/// WebXR plane orientation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum WebXrPlaneOrientation {
    Horizontal,
    Vertical,
}

// ---------------------------------------------------------------------------
// Core structs
// ---------------------------------------------------------------------------

/// A rigid transform (position + orientation), mirrors `XRRigidTransform`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebXrRigidTransform {
    /// Position as [x, y, z].
    pub position: [f64; 3],
    /// Orientation as unit quaternion [x, y, z, w].
    pub orientation: [f64; 4],
    /// The 4×4 column-major transformation matrix.
    pub matrix: [f64; 16],
}

impl WebXrRigidTransform {
    /// Identity transform.
    pub fn identity() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
            matrix: [
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                1.0,
            ],
        }
    }

    /// Create a transform from a position vector only (identity rotation).
    pub fn from_position(pos: [f64; 3]) -> Self {
        let mut t = Self::identity();
        t.position = pos;
        t.matrix[12] = pos[0];
        t.matrix[13] = pos[1];
        t.matrix[14] = pos[2];
        t
    }

    /// Inverse of this transform.
    pub fn inverse(&self) -> Self {
        let [qx, qy, qz, qw] = self.orientation;
        let inv_orient = [-qx, -qy, -qz, qw];

        // Rotate the negated position by the conjugate quaternion.
        let px = -self.position[0];
        let py = -self.position[1];
        let pz = -self.position[2];
        let inv_pos = quat_rotate(inv_orient, [px, py, pz]);

        let mut result = Self {
            position: inv_pos,
            orientation: inv_orient,
            matrix: [0.0; 16],
        };
        result.recompute_matrix();
        result
    }

    /// Recompute the 4×4 matrix from position and orientation.
    fn recompute_matrix(&mut self) {
        let [qx, qy, qz, qw] = self.orientation;
        let xx = qx * qx;
        let yy = qy * qy;
        let zz = qz * qz;
        let xy = qx * qy;
        let xz = qx * qz;
        let yz = qy * qz;
        let wx = qw * qx;
        let wy = qw * qy;
        let wz = qw * qz;

        self.matrix = [
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy + wz),
            2.0 * (xz - wy),
            0.0,
            2.0 * (xy - wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz + wx),
            0.0,
            2.0 * (xz + wy),
            2.0 * (yz - wx),
            1.0 - 2.0 * (xx + yy),
            0.0,
            self.position[0],
            self.position[1],
            self.position[2],
            1.0,
        ];
    }
}

/// Rotate a vector by a unit quaternion [x, y, z, w].
fn quat_rotate(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let [qx, qy, qz, qw] = q;
    let ux = qy * v[2] - qz * v[1];
    let uy = qz * v[0] - qx * v[2];
    let uz = qx * v[1] - qy * v[0];
    [
        v[0] + 2.0 * (qw * ux + qy * uz - qz * uy),
        v[1] + 2.0 * (qw * uy + qz * ux - qx * uz),
        v[2] + 2.0 * (qw * uz + qx * uy - qy * ux),
    ]
}

/// A pose in space, mirrors `XRPose`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebXrPose {
    pub transform: WebXrRigidTransform,
    pub emulated_position: bool,
    /// Linear velocity in m/s, if available.
    pub linear_velocity: Option<[f64; 3]>,
    /// Angular velocity in rad/s, if available.
    pub angular_velocity: Option<[f64; 3]>,
}

/// A viewport describing the render target sub-rectangle.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebXrViewport {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// A single view within a frame (one per eye in stereo).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebXrView {
    pub eye: WebXrEye,
    /// 4×4 column-major projection matrix.
    pub projection_matrix: [f64; 16],
    pub transform: WebXrRigidTransform,
    pub recommended_viewport: WebXrViewport,
}

/// Render state for the session.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebXrRenderState {
    pub depth_near: f64,
    pub depth_far: f64,
    pub inline_vertical_field_of_view: Option<f64>,
}

impl Default for WebXrRenderState {
    fn default() -> Self {
        Self {
            depth_near: 0.1,
            depth_far: 1000.0,
            inline_vertical_field_of_view: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Spaces
// ---------------------------------------------------------------------------

/// A coordinate space, mirrors `XRSpace`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebXrSpace {
    pub id: Uuid,
    pub space_type: WebXrReferenceSpaceType,
    /// Optional bounds geometry as a polygon (x, z pairs on the floor plane).
    pub bounds_geometry: Option<Vec<[f64; 2]>>,
}

/// A joint space on a tracked hand.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebXrJointSpace {
    pub joint: WebXrHandJoint,
    pub radius: f64,
    pub pose: Option<WebXrPose>,
}

// ---------------------------------------------------------------------------
// Input sources
// ---------------------------------------------------------------------------

/// A WebXR input source, mirrors `XRInputSource`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebXrInputSource {
    pub id: Uuid,
    pub handedness: WebXrHandedness,
    pub target_ray_mode: WebXrTargetRayMode,
    /// Active profiles for this input source (most-to-least specific).
    pub profiles: Vec<String>,
    pub target_ray_space: WebXrSpace,
    pub grip_space: Option<WebXrSpace>,
    pub hand: Option<WebXrHand>,
}

impl WebXrInputSource {
    /// Whether this source represents a tracked hand (no controller).
    pub fn is_hand(&self) -> bool {
        self.hand.is_some()
    }

    /// Whether this source provides grip pose information.
    pub fn has_grip(&self) -> bool {
        self.grip_space.is_some()
    }
}

/// Ordered collection of active input sources, mirrors `XRInputSourceArray`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebXrInputSourceArray {
    pub sources: Vec<WebXrInputSource>,
}

impl WebXrInputSourceArray {
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.sources.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    /// Get all sources matching a handedness.
    pub fn by_handedness(&self, handedness: WebXrHandedness) -> Vec<&WebXrInputSource> {
        self.sources
            .iter()
            .filter(|s| s.handedness == handedness)
            .collect()
    }

    /// Get all tracked-pointer sources.
    pub fn tracked_pointers(&self) -> Vec<&WebXrInputSource> {
        self.sources
            .iter()
            .filter(|s| s.target_ray_mode == WebXrTargetRayMode::TrackedPointer)
            .collect()
    }

    /// Get all hand sources.
    pub fn hands(&self) -> Vec<&WebXrInputSource> {
        self.sources.iter().filter(|s| s.is_hand()).collect()
    }
}

impl Default for WebXrInputSourceArray {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Hand tracking
// ---------------------------------------------------------------------------

/// A tracked hand, mirrors `XRHand`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebXrHand {
    pub joints: Vec<WebXrJointSpace>,
}

impl WebXrHand {
    /// Number of joints in a fully-tracked hand.
    pub const JOINT_COUNT: usize = 25;

    /// Look up a specific joint by its identifier.
    pub fn joint(&self, joint: WebXrHandJoint) -> Option<&WebXrJointSpace> {
        self.joints.iter().find(|j| j.joint == joint)
    }

    /// Whether every joint has a valid pose.
    pub fn is_fully_tracked(&self) -> bool {
        self.joints.len() == Self::JOINT_COUNT && self.joints.iter().all(|j| j.pose.is_some())
    }

    /// Average joint radius (useful for collision sizing).
    pub fn average_joint_radius(&self) -> f64 {
        if self.joints.is_empty() {
            return 0.0;
        }
        self.joints.iter().map(|j| j.radius).sum::<f64>() / self.joints.len() as f64
    }
}

// ---------------------------------------------------------------------------
// Hit-test, anchors, and planes
// ---------------------------------------------------------------------------

/// Hit-test source descriptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebXrHitTestSource {
    pub id: Uuid,
    pub space: WebXrSpace,
    /// Entity types to test against: `"point"`, `"plane"`, `"mesh"`.
    pub entity_types: Vec<String>,
    pub offset_ray: Option<WebXrRigidTransform>,
}

/// A single hit-test result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebXrHitTestResult {
    pub pose: WebXrPose,
    /// If the hit was on a plane, contains the plane id.
    pub plane_id: Option<Uuid>,
}

/// A world-anchored pose, mirrors `XRAnchor`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebXrAnchor {
    pub id: Uuid,
    pub anchor_space: WebXrSpace,
    pub created_at: f64,
}

/// A detected real-world plane, mirrors `XRPlane`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebXrPlane {
    pub id: Uuid,
    pub orientation: WebXrPlaneOrientation,
    pub plane_space: WebXrSpace,
    /// Polygon vertices [x, z] on the plane's local coordinate system.
    pub polygon: Vec<[f64; 2]>,
    pub last_changed_time: f64,
}

// ---------------------------------------------------------------------------
// Feature descriptors
// ---------------------------------------------------------------------------

/// A feature that can be requested when creating a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebXrFeatureDescriptor {
    /// Feature name as specified in the WebXR spec (e.g. `"local-floor"`).
    pub name: String,
    /// Whether this feature is required or optional.
    pub required: bool,
}

impl WebXrFeatureDescriptor {
    pub fn required(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            required: true,
        }
    }

    pub fn optional(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            required: false,
        }
    }
}

/// Well-known WebXR feature names.
pub mod features {
    pub const LOCAL: &str = "local";
    pub const LOCAL_FLOOR: &str = "local-floor";
    pub const BOUNDED_FLOOR: &str = "bounded-floor";
    pub const UNBOUNDED: &str = "unbounded";
    pub const HAND_TRACKING: &str = "hand-tracking";
    pub const LAYERS: &str = "layers";
    pub const ANCHORS: &str = "anchors";
    pub const PLANE_DETECTION: &str = "plane-detection";
    pub const HIT_TEST: &str = "hit-test";
    pub const DEPTH_SENSING: &str = "depth-sensing";
    pub const LIGHT_ESTIMATION: &str = "light-estimation";
    pub const DOM_OVERLAY: &str = "dom-overlay";
}

// ---------------------------------------------------------------------------
// Frame
// ---------------------------------------------------------------------------

/// A single animation frame, mirrors `XRFrame`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebXrFrame {
    pub session_id: Uuid,
    pub timestamp: f64,
    pub viewer_pose: Option<WebXrPose>,
    pub views: Vec<WebXrView>,
    pub input_sources: WebXrInputSourceArray,
    pub detected_planes: Vec<WebXrPlane>,
    pub hit_test_results: Vec<WebXrHitTestResult>,
    pub anchors: Vec<WebXrAnchor>,
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

/// A WebXR session, mirrors `XRSession`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebXrSession {
    pub id: Uuid,
    pub mode: WebXrSessionMode,
    pub visibility_state: WebXrVisibilityState,
    pub render_state: WebXrRenderState,
    pub reference_space_type: WebXrReferenceSpaceType,
    pub input_sources: WebXrInputSourceArray,
    pub enabled_features: Vec<String>,
    pub supported_frame_rates: Vec<f64>,
    pub frame_rate: Option<f64>,
}

impl WebXrSession {
    /// Whether the session has the hand-tracking feature enabled.
    pub fn has_hand_tracking(&self) -> bool {
        self.enabled_features
            .iter()
            .any(|f| f == features::HAND_TRACKING)
    }

    /// Whether the session has plane detection enabled.
    pub fn has_plane_detection(&self) -> bool {
        self.enabled_features
            .iter()
            .any(|f| f == features::PLANE_DETECTION)
    }

    /// Whether the session has hit-testing enabled.
    pub fn has_hit_test(&self) -> bool {
        self.enabled_features
            .iter()
            .any(|f| f == features::HIT_TEST)
    }

    /// Check whether a feature is enabled.
    pub fn is_feature_enabled(&self, name: &str) -> bool {
        self.enabled_features.iter().any(|f| f == name)
    }

    /// Current frame rate, falling back to the first supported rate or 60 Hz.
    pub fn effective_frame_rate(&self) -> f64 {
        self.frame_rate
            .or_else(|| self.supported_frame_rates.first().copied())
            .unwrap_or(60.0)
    }
}

// ---------------------------------------------------------------------------
// Mapping helpers
// ---------------------------------------------------------------------------

/// Map a `WebXrReferenceSpaceType` to a `MovementMode`.
pub fn movement_mode_from_reference_space(space: WebXrReferenceSpaceType) -> MovementMode {
    match space {
        WebXrReferenceSpaceType::Viewer | WebXrReferenceSpaceType::Local => MovementMode::Seated,
        WebXrReferenceSpaceType::LocalFloor => MovementMode::Standing,
        WebXrReferenceSpaceType::BoundedFloor | WebXrReferenceSpaceType::Unbounded => {
            MovementMode::RoomScale
        }
    }
}

/// Map a `WebXrSessionMode` to a `DeviceType`.
pub fn device_type_from_session_mode(mode: WebXrSessionMode) -> DeviceType {
    match mode {
        WebXrSessionMode::Inline => DeviceType::PhoneAR,
        WebXrSessionMode::ImmersiveVr => DeviceType::Standalone,
        WebXrSessionMode::ImmersiveAr => DeviceType::ARGlasses,
    }
}

/// Infer a `TrackingVolume` from the reference-space type.
fn tracking_volume_from_reference_space(space: WebXrReferenceSpaceType) -> TrackingVolume {
    match space {
        WebXrReferenceSpaceType::Viewer => TrackingVolume::new_room(0.5, 0.5, 2.0),
        WebXrReferenceSpaceType::Local => TrackingVolume::new_room(1.0, 1.0, 2.0),
        WebXrReferenceSpaceType::LocalFloor => TrackingVolume::new_room(1.5, 1.5, 2.5),
        WebXrReferenceSpaceType::BoundedFloor => TrackingVolume::new_room(3.0, 3.0, 2.5),
        WebXrReferenceSpaceType::Unbounded => TrackingVolume::new_room(10.0, 10.0, 3.0),
    }
}

/// Derive supported `InteractionType`s from WebXR input sources and session
/// features.
fn interactions_from_session(session: &WebXrSession) -> Vec<InteractionType> {
    let mut interactions = Vec::new();

    let has_pointer = session
        .input_sources
        .sources
        .iter()
        .any(|s| s.target_ray_mode == WebXrTargetRayMode::TrackedPointer);
    let has_gaze = session
        .input_sources
        .sources
        .iter()
        .any(|s| s.target_ray_mode == WebXrTargetRayMode::Gaze);
    let has_screen = session
        .input_sources
        .sources
        .iter()
        .any(|s| s.target_ray_mode == WebXrTargetRayMode::Screen);
    let has_hand = session.has_hand_tracking()
        || session.input_sources.sources.iter().any(|s| s.is_hand());

    // Pointer-based controllers provide click, grab, drag, hover.
    if has_pointer {
        interactions.push(InteractionType::Click);
        interactions.push(InteractionType::Grab);
        interactions.push(InteractionType::Drag);
        interactions.push(InteractionType::Hover);
        interactions.push(InteractionType::Slider);
        interactions.push(InteractionType::Dial);
        interactions.push(InteractionType::Toggle);
    }

    // Gaze adds gaze interaction.
    if has_gaze {
        interactions.push(InteractionType::Gaze);
    }

    // Screen touch adds click/drag/proximity.
    if has_screen {
        if !interactions.contains(&InteractionType::Click) {
            interactions.push(InteractionType::Click);
        }
        if !interactions.contains(&InteractionType::Drag) {
            interactions.push(InteractionType::Drag);
        }
        interactions.push(InteractionType::Proximity);
    }

    // Hand tracking enables gesture and two-handed interactions.
    if has_hand {
        interactions.push(InteractionType::Gesture);
        if !interactions.contains(&InteractionType::Grab) {
            interactions.push(InteractionType::Grab);
        }
        if !interactions.contains(&InteractionType::Proximity) {
            interactions.push(InteractionType::Proximity);
        }

        // Two hands present → two-handed interactions.
        let hand_count = session.input_sources.hands().len();
        if hand_count >= 2 {
            interactions.push(InteractionType::TwoHanded);
        }
    }

    interactions
}

/// Infer a `ControllerType` from WebXR input-source profile strings.
fn controller_type_from_profiles(profiles: &[String]) -> Option<ControllerType> {
    for profile in profiles {
        let p = profile.to_lowercase();
        if p.contains("meta-quest") && p.contains("touch-plus") {
            return Some(ControllerType::Quest3Controller);
        }
        if p.contains("oculus-touch") || p.contains("meta-quest-touch") {
            return Some(ControllerType::QuestTouch);
        }
        if p.contains("sony-dualsense") || p.contains("playstation") {
            return Some(ControllerType::PSVR2Sense);
        }
        if p.contains("valve-index") || p.contains("knuckles") {
            return Some(ControllerType::ValveIndex);
        }
        if p.contains("htc-vive-cosmos") {
            return Some(ControllerType::ViveCosmos);
        }
        if p.contains("microsoft-mixed-reality") || p.contains("windows-mr") {
            return Some(ControllerType::WindowsMR);
        }
        if p.contains("generic-trigger-squeeze-touchpad")
            || p.contains("generic-trigger-squeeze-thumbstick")
        {
            return Some(ControllerType::GenericGamepad);
        }
    }
    None
}

/// Derive field-of-view estimates from session mode and render state.
fn fov_from_session(session: &WebXrSession) -> [f64; 2] {
    match session.mode {
        WebXrSessionMode::Inline => {
            let vfov = session
                .render_state
                .inline_vertical_field_of_view
                .unwrap_or(std::f64::consts::FRAC_PI_2);
            let vfov_deg = vfov.to_degrees();
            [vfov_deg * 1.33, vfov_deg]
        }
        WebXrSessionMode::ImmersiveVr => [100.0, 90.0],
        WebXrSessionMode::ImmersiveAr => [60.0, 50.0],
    }
}

/// Build a `DeviceConfig` from a `WebXrSession`.
///
/// This maps WebXR session properties—mode, reference space, input sources,
/// and enabled features—onto the internal device configuration used by the
/// verification pipeline.
pub fn device_config_from_webxr(session: &WebXrSession) -> DeviceConfig {
    let device_type = device_type_from_session_mode(session.mode);
    let movement_mode = movement_mode_from_reference_space(session.reference_space_type);
    let tracking_volume = tracking_volume_from_reference_space(session.reference_space_type);
    let supported_interactions = interactions_from_session(session);
    let hand_tracking = session.has_hand_tracking();
    let refresh_rate = session.effective_frame_rate();
    let fov = fov_from_session(session);

    // Determine controller type from the first tracked-pointer input source.
    let controller_type = session
        .input_sources
        .tracked_pointers()
        .first()
        .and_then(|src| controller_type_from_profiles(&src.profiles));

    // Eye tracking is inferred from gaze-based input sources.
    let eye_tracking = session
        .input_sources
        .sources
        .iter()
        .any(|s| s.target_ray_mode == WebXrTargetRayMode::Gaze);

    DeviceConfig {
        id: Uuid::new_v4(),
        name: format!("WebXR:{:?}", session.mode),
        device_type,
        tracking_volume,
        supported_interactions,
        movement_mode,
        controller_offset: [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
        tracking_precision: 0.002,
        refresh_rate,
        field_of_view: fov,
        hand_tracking,
        eye_tracking,
        controller_type,
        constraints: DeviceConstraints::default(),
    }
}

// ---------------------------------------------------------------------------
// Verification target mapping
// ---------------------------------------------------------------------------

/// A verification target derived from a WebXR input source.
///
/// The affordance verifier uses these to determine which interaction modalities
/// a scene element must be accessible through.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebXrVerificationTarget {
    /// The originating input source id.
    pub source_id: Uuid,
    pub handedness: WebXrHandedness,
    pub target_ray_mode: WebXrTargetRayMode,
    /// Interaction types this source can perform.
    pub supported_interactions: Vec<InteractionType>,
    /// Whether precise hand-joint positions are available.
    pub has_articulated_hand: bool,
    /// Approximate reach bounding box in the reference space.
    pub reach_bounds: BoundingBox,
}

/// Map all input sources in a session to verification targets.
pub fn verification_targets_from_session(session: &WebXrSession) -> Vec<WebXrVerificationTarget> {
    session
        .input_sources
        .sources
        .iter()
        .map(|src| {
            let mut interactions = Vec::new();
            match src.target_ray_mode {
                WebXrTargetRayMode::TrackedPointer => {
                    interactions.extend_from_slice(&[
                        InteractionType::Click,
                        InteractionType::Grab,
                        InteractionType::Drag,
                        InteractionType::Hover,
                        InteractionType::Slider,
                        InteractionType::Dial,
                        InteractionType::Toggle,
                    ]);
                }
                WebXrTargetRayMode::Gaze => {
                    interactions.push(InteractionType::Gaze);
                    interactions.push(InteractionType::Click);
                }
                WebXrTargetRayMode::Screen => {
                    interactions.push(InteractionType::Click);
                    interactions.push(InteractionType::Drag);
                }
                WebXrTargetRayMode::TransientPointer => {
                    interactions.push(InteractionType::Click);
                    interactions.push(InteractionType::Hover);
                }
            }
            if src.is_hand() {
                interactions.push(InteractionType::Gesture);
                if !interactions.contains(&InteractionType::Grab) {
                    interactions.push(InteractionType::Grab);
                }
                interactions.push(InteractionType::Proximity);
            }

            let reach_half = if src.is_hand() { 0.8 } else { 1.0 };
            let reach_bounds = BoundingBox::new(
                [-reach_half, -0.3, -reach_half],
                [reach_half, 2.2, reach_half],
            );

            WebXrVerificationTarget {
                source_id: src.id,
                handedness: src.handedness,
                target_ray_mode: src.target_ray_mode,
                supported_interactions: interactions,
                has_articulated_hand: src
                    .hand
                    .as_ref()
                    .map_or(false, |h| h.is_fully_tracked()),
                reach_bounds,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_session(mode: WebXrSessionMode) -> WebXrSession {
        WebXrSession {
            id: Uuid::new_v4(),
            mode,
            visibility_state: WebXrVisibilityState::Visible,
            render_state: WebXrRenderState::default(),
            reference_space_type: WebXrReferenceSpaceType::LocalFloor,
            input_sources: WebXrInputSourceArray::new(),
            enabled_features: vec![
                features::LOCAL_FLOOR.to_string(),
                features::HAND_TRACKING.to_string(),
            ],
            supported_frame_rates: vec![72.0, 90.0, 120.0],
            frame_rate: Some(90.0),
        }
    }

    fn make_pointer_source(handedness: WebXrHandedness) -> WebXrInputSource {
        WebXrInputSource {
            id: Uuid::new_v4(),
            handedness,
            target_ray_mode: WebXrTargetRayMode::TrackedPointer,
            profiles: vec!["meta-quest-touch-plus".to_string()],
            target_ray_space: WebXrSpace {
                id: Uuid::new_v4(),
                space_type: WebXrReferenceSpaceType::Viewer,
                bounds_geometry: None,
            },
            grip_space: Some(WebXrSpace {
                id: Uuid::new_v4(),
                space_type: WebXrReferenceSpaceType::Viewer,
                bounds_geometry: None,
            }),
            hand: None,
        }
    }

    // -- enum tests --

    #[test]
    fn test_session_mode_immersive() {
        assert!(!WebXrSessionMode::Inline.is_immersive());
        assert!(WebXrSessionMode::ImmersiveVr.is_immersive());
        assert!(WebXrSessionMode::ImmersiveAr.is_immersive());
    }

    #[test]
    fn test_handedness_serde_round_trip() {
        let json = serde_json::to_string(&WebXrHandedness::Left).unwrap();
        assert_eq!(json, "\"left\"");
        let h: WebXrHandedness = serde_json::from_str(&json).unwrap();
        assert_eq!(h, WebXrHandedness::Left);
    }

    #[test]
    fn test_visibility_state_serde() {
        let json = serde_json::to_string(&WebXrVisibilityState::VisibleBlurred).unwrap();
        assert_eq!(json, "\"visible-blurred\"");
    }

    // -- rigid transform tests --

    #[test]
    fn test_identity_transform() {
        let t = WebXrRigidTransform::identity();
        assert_eq!(t.position, [0.0, 0.0, 0.0]);
        assert_eq!(t.orientation[3], 1.0); // w
        assert!((t.matrix[0] - 1.0).abs() < 1e-12);
        assert!((t.matrix[5] - 1.0).abs() < 1e-12);
        assert!((t.matrix[10] - 1.0).abs() < 1e-12);
        assert!((t.matrix[15] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_from_position() {
        let t = WebXrRigidTransform::from_position([1.0, 2.0, 3.0]);
        assert_eq!(t.position, [1.0, 2.0, 3.0]);
        assert!((t.matrix[12] - 1.0).abs() < 1e-12);
        assert!((t.matrix[13] - 2.0).abs() < 1e-12);
        assert!((t.matrix[14] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_inverse_identity() {
        let t = WebXrRigidTransform::identity();
        let inv = t.inverse();
        assert!((inv.position[0]).abs() < 1e-12);
        assert!((inv.position[1]).abs() < 1e-12);
        assert!((inv.position[2]).abs() < 1e-12);
    }

    #[test]
    fn test_inverse_translation() {
        let t = WebXrRigidTransform::from_position([5.0, 0.0, 0.0]);
        let inv = t.inverse();
        assert!((inv.position[0] + 5.0).abs() < 1e-12);
    }

    // -- session tests --

    #[test]
    fn test_session_feature_queries() {
        let session = make_session(WebXrSessionMode::ImmersiveVr);
        assert!(session.has_hand_tracking());
        assert!(!session.has_plane_detection());
        assert!(!session.has_hit_test());
        assert!(session.is_feature_enabled(features::LOCAL_FLOOR));
    }

    #[test]
    fn test_effective_frame_rate() {
        let mut session = make_session(WebXrSessionMode::ImmersiveVr);
        assert!((session.effective_frame_rate() - 90.0).abs() < 1e-12);

        session.frame_rate = None;
        assert!((session.effective_frame_rate() - 72.0).abs() < 1e-12);

        session.supported_frame_rates.clear();
        assert!((session.effective_frame_rate() - 60.0).abs() < 1e-12);
    }

    // -- input source array tests --

    #[test]
    fn test_input_source_array() {
        let mut arr = WebXrInputSourceArray::new();
        assert!(arr.is_empty());

        arr.sources
            .push(make_pointer_source(WebXrHandedness::Left));
        arr.sources
            .push(make_pointer_source(WebXrHandedness::Right));

        assert_eq!(arr.len(), 2);
        assert_eq!(arr.by_handedness(WebXrHandedness::Left).len(), 1);
        assert_eq!(arr.tracked_pointers().len(), 2);
        assert!(arr.hands().is_empty());
    }

    // -- mapping tests --

    #[test]
    fn test_movement_mode_mapping() {
        assert_eq!(
            movement_mode_from_reference_space(WebXrReferenceSpaceType::Viewer),
            MovementMode::Seated
        );
        assert_eq!(
            movement_mode_from_reference_space(WebXrReferenceSpaceType::LocalFloor),
            MovementMode::Standing
        );
        assert_eq!(
            movement_mode_from_reference_space(WebXrReferenceSpaceType::BoundedFloor),
            MovementMode::RoomScale
        );
    }

    #[test]
    fn test_device_type_mapping() {
        assert_eq!(
            device_type_from_session_mode(WebXrSessionMode::Inline),
            DeviceType::PhoneAR
        );
        assert_eq!(
            device_type_from_session_mode(WebXrSessionMode::ImmersiveVr),
            DeviceType::Standalone
        );
        assert_eq!(
            device_type_from_session_mode(WebXrSessionMode::ImmersiveAr),
            DeviceType::ARGlasses
        );
    }

    #[test]
    fn test_controller_type_detection() {
        let profiles = vec!["meta-quest-touch-plus".to_string()];
        assert_eq!(
            controller_type_from_profiles(&profiles),
            Some(ControllerType::Quest3Controller)
        );

        let profiles = vec!["valve-index-controller".to_string()];
        assert_eq!(
            controller_type_from_profiles(&profiles),
            Some(ControllerType::ValveIndex)
        );

        let profiles: Vec<String> = vec![];
        assert_eq!(controller_type_from_profiles(&profiles), None);
    }

    #[test]
    fn test_device_config_from_immersive_vr() {
        let mut session = make_session(WebXrSessionMode::ImmersiveVr);
        session
            .input_sources
            .sources
            .push(make_pointer_source(WebXrHandedness::Left));
        session
            .input_sources
            .sources
            .push(make_pointer_source(WebXrHandedness::Right));

        let config = device_config_from_webxr(&session);
        assert_eq!(config.device_type, DeviceType::Standalone);
        assert_eq!(config.movement_mode, MovementMode::Standing);
        assert!(config.hand_tracking);
        assert!((config.refresh_rate - 90.0).abs() < 1e-12);
        assert!(config
            .supported_interactions
            .contains(&InteractionType::Click));
        assert!(config
            .supported_interactions
            .contains(&InteractionType::Gesture));
        assert_eq!(
            config.controller_type,
            Some(ControllerType::Quest3Controller)
        );
    }

    #[test]
    fn test_device_config_from_inline() {
        let session = make_session(WebXrSessionMode::Inline);
        let config = device_config_from_webxr(&session);
        assert_eq!(config.device_type, DeviceType::PhoneAR);
    }

    // -- verification target tests --

    #[test]
    fn test_verification_targets() {
        let mut session = make_session(WebXrSessionMode::ImmersiveVr);
        session
            .input_sources
            .sources
            .push(make_pointer_source(WebXrHandedness::Left));
        session
            .input_sources
            .sources
            .push(make_pointer_source(WebXrHandedness::Right));

        let targets = verification_targets_from_session(&session);
        assert_eq!(targets.len(), 2);
        assert_eq!(targets[0].handedness, WebXrHandedness::Left);
        assert!(targets[0]
            .supported_interactions
            .contains(&InteractionType::Click));
        assert!(!targets[0].has_articulated_hand);
    }

    #[test]
    fn test_verification_target_gaze_source() {
        let mut session = make_session(WebXrSessionMode::ImmersiveVr);
        session.input_sources.sources.push(WebXrInputSource {
            id: Uuid::new_v4(),
            handedness: WebXrHandedness::None,
            target_ray_mode: WebXrTargetRayMode::Gaze,
            profiles: vec![],
            target_ray_space: WebXrSpace {
                id: Uuid::new_v4(),
                space_type: WebXrReferenceSpaceType::Viewer,
                bounds_geometry: None,
            },
            grip_space: None,
            hand: None,
        });

        let targets = verification_targets_from_session(&session);
        assert_eq!(targets.len(), 1);
        assert!(targets[0]
            .supported_interactions
            .contains(&InteractionType::Gaze));
    }

    // -- hand tracking tests --

    #[test]
    fn test_hand_joint_lookup() {
        let hand = WebXrHand {
            joints: vec![
                WebXrJointSpace {
                    joint: WebXrHandJoint::Wrist,
                    radius: 0.02,
                    pose: Some(WebXrPose {
                        transform: WebXrRigidTransform::identity(),
                        emulated_position: false,
                        linear_velocity: None,
                        angular_velocity: None,
                    }),
                },
                WebXrJointSpace {
                    joint: WebXrHandJoint::IndexFingerTip,
                    radius: 0.008,
                    pose: Some(WebXrPose {
                        transform: WebXrRigidTransform::from_position([0.1, 0.0, 0.0]),
                        emulated_position: false,
                        linear_velocity: None,
                        angular_velocity: None,
                    }),
                },
            ],
        };

        assert!(hand.joint(WebXrHandJoint::Wrist).is_some());
        assert!(hand.joint(WebXrHandJoint::IndexFingerTip).is_some());
        assert!(hand.joint(WebXrHandJoint::ThumbTip).is_none());
        assert!(!hand.is_fully_tracked());
        assert!(hand.average_joint_radius() > 0.0);
    }

    // -- feature descriptor tests --

    #[test]
    fn test_feature_descriptors() {
        let req = WebXrFeatureDescriptor::required(features::LOCAL_FLOOR);
        assert!(req.required);
        assert_eq!(req.name, "local-floor");

        let opt = WebXrFeatureDescriptor::optional(features::HAND_TRACKING);
        assert!(!opt.required);
    }

    // -- render state tests --

    #[test]
    fn test_render_state_default() {
        let rs = WebXrRenderState::default();
        assert!((rs.depth_near - 0.1).abs() < 1e-12);
        assert!((rs.depth_far - 1000.0).abs() < 1e-12);
        assert!(rs.inline_vertical_field_of_view.is_none());
    }

    // -- serde round-trip --

    #[test]
    fn test_session_serde_round_trip() {
        let session = make_session(WebXrSessionMode::ImmersiveVr);
        let json = serde_json::to_string(&session).unwrap();
        let restored: WebXrSession = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.id, session.id);
        assert_eq!(restored.mode, session.mode);
        assert!(restored.has_hand_tracking());
    }
}

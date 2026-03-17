//! OpenXR standard types for XR affordance verification.
//!
//! Models the OpenXR API concepts relevant to accessibility verification
//! without depending on any external OpenXR crate. Provides interaction
//! profile definitions and mapping to internal [`DeviceConfig`] types.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::device::*;
use crate::geometry::*;
use crate::scene::InteractionType;

// ---------------------------------------------------------------------------
// Core handle / instance types
// ---------------------------------------------------------------------------

/// Represents an OpenXR runtime instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenXrInstance {
    pub id: Uuid,
    pub runtime_name: String,
    pub runtime_version: String,
    pub api_version: OpenXrVersion,
    pub enabled_extensions: Vec<OpenXrExtension>,
    pub system_properties: Option<OpenXrSystemProperties>,
}

/// Semantic version triplet used throughout OpenXR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenXrVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl OpenXrVersion {
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self { major, minor, patch }
    }

    /// OpenXR 1.0.0 – baseline spec.
    pub fn v1_0() -> Self {
        Self::new(1, 0, 0)
    }

    /// OpenXR 1.1.0 – latest ratified revision.
    pub fn v1_1() -> Self {
        Self::new(1, 1, 0)
    }
}

impl std::fmt::Display for OpenXrVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Properties reported by `xrGetSystemProperties`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenXrSystemProperties {
    pub system_id: u64,
    pub system_name: String,
    pub vendor_id: u32,
    pub max_swapchain_width: u32,
    pub max_swapchain_height: u32,
    pub max_layer_count: u32,
    pub orientation_tracking: bool,
    pub position_tracking: bool,
    pub hand_tracking_supported: bool,
    pub eye_tracking_supported: bool,
}

impl Default for OpenXrSystemProperties {
    fn default() -> Self {
        Self {
            system_id: 0,
            system_name: String::new(),
            vendor_id: 0,
            max_swapchain_width: 4096,
            max_swapchain_height: 4096,
            max_layer_count: 16,
            orientation_tracking: true,
            position_tracking: true,
            hand_tracking_supported: false,
            eye_tracking_supported: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

/// An active OpenXR session.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenXrSession {
    pub id: Uuid,
    pub instance_id: Uuid,
    pub state: OpenXrSessionState,
    pub reference_space: OpenXrReferenceSpaceType,
    pub view_configuration: OpenXrViewConfigurationType,
    pub active_interaction_profile: Option<String>,
    pub blend_mode: OpenXrBlendMode,
}

/// Session lifecycle states (XrSessionState).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OpenXrSessionState {
    Unknown,
    Idle,
    Ready,
    Synchronized,
    Visible,
    Focused,
    Stopping,
    LossPending,
    Exiting,
}

impl OpenXrSessionState {
    /// Returns `true` when the session can accept input.
    pub fn accepts_input(&self) -> bool {
        matches!(self, Self::Focused)
    }

    /// Returns `true` when the session should render frames.
    pub fn should_render(&self) -> bool {
        matches!(self, Self::Visible | Self::Focused)
    }
}

/// Environment blend modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OpenXrBlendMode {
    Opaque,
    Additive,
    AlphaBlend,
}

// ---------------------------------------------------------------------------
// Spaces
// ---------------------------------------------------------------------------

/// An OpenXR spatial anchor / reference space.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenXrSpace {
    pub id: Uuid,
    pub space_type: OpenXrReferenceSpaceType,
    /// Pose offset from the base space origin (4×4 column-major).
    pub pose_offset: [f64; 16],
    pub bounds: Option<BoundingBox>,
}

/// Well-known reference space types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OpenXrReferenceSpaceType {
    /// Head-locked, resets each frame.
    View,
    /// Gravity-aligned origin at initial device position.
    Local,
    /// Floor-level origin at the center of the play area.
    Stage,
    /// Unbounded world-anchored coordinate system (AR).
    Unbounded,
}

impl OpenXrReferenceSpaceType {
    pub fn supports_bounds(&self) -> bool {
        matches!(self, Self::Stage)
    }
}

// ---------------------------------------------------------------------------
// View configuration
// ---------------------------------------------------------------------------

/// View configuration types (stereo, mono, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OpenXrViewConfigurationType {
    Mono,
    Stereo,
    /// Quad views (e.g. foveated rendering with inner/outer pairs).
    StereoWithFoveatedInset,
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

/// A named action within an action set.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenXrAction {
    pub name: String,
    pub localized_name: String,
    pub action_type: OpenXrActionType,
    pub subaction_paths: Vec<OpenXrPath>,
}

/// Logical grouping of related actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenXrActionSet {
    pub name: String,
    pub localized_name: String,
    pub priority: u32,
    pub actions: Vec<OpenXrAction>,
}

/// OpenXR action types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OpenXrActionType {
    BooleanInput,
    FloatInput,
    Vector2fInput,
    PoseInput,
    VibrationOutput,
}

impl OpenXrActionType {
    /// Whether this type represents an input (vs. output).
    pub fn is_input(&self) -> bool {
        !matches!(self, Self::VibrationOutput)
    }
}

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

/// A typed OpenXR path string, e.g. `/user/hand/left/input/trigger/value`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OpenXrPath(pub String);

impl OpenXrPath {
    pub fn new(path: impl Into<String>) -> Self {
        Self(path.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    // Convenience constructors ------------------------------------------------

    pub fn left_hand() -> Self {
        Self::new("/user/hand/left")
    }

    pub fn right_hand() -> Self {
        Self::new("/user/hand/right")
    }

    pub fn head() -> Self {
        Self::new("/user/head")
    }

    pub fn gamepad() -> Self {
        Self::new("/user/gamepad")
    }

    pub fn left_trigger_value() -> Self {
        Self::new("/user/hand/left/input/trigger/value")
    }

    pub fn right_trigger_value() -> Self {
        Self::new("/user/hand/right/input/trigger/value")
    }

    pub fn left_squeeze_value() -> Self {
        Self::new("/user/hand/left/input/squeeze/value")
    }

    pub fn right_squeeze_value() -> Self {
        Self::new("/user/hand/right/input/squeeze/value")
    }

    pub fn left_thumbstick() -> Self {
        Self::new("/user/hand/left/input/thumbstick")
    }

    pub fn right_thumbstick() -> Self {
        Self::new("/user/hand/right/input/thumbstick")
    }

    pub fn left_grip_pose() -> Self {
        Self::new("/user/hand/left/input/grip/pose")
    }

    pub fn right_grip_pose() -> Self {
        Self::new("/user/hand/right/input/grip/pose")
    }

    pub fn left_aim_pose() -> Self {
        Self::new("/user/hand/left/input/aim/pose")
    }

    pub fn right_aim_pose() -> Self {
        Self::new("/user/hand/right/input/aim/pose")
    }

    pub fn left_haptic() -> Self {
        Self::new("/user/hand/left/output/haptic")
    }

    pub fn right_haptic() -> Self {
        Self::new("/user/hand/right/output/haptic")
    }
}

impl std::fmt::Display for OpenXrPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ---------------------------------------------------------------------------
// Swapchain & composition layers
// ---------------------------------------------------------------------------

/// Rendering swapchain handle.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenXrSwapchain {
    pub id: Uuid,
    pub width: u32,
    pub height: u32,
    pub sample_count: u32,
    pub format: String,
    pub usage_flags: Vec<String>,
}

/// A single composition layer submitted to the runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenXrCompositionLayer {
    pub layer_type: OpenXrLayerType,
    pub space_id: Uuid,
    pub swapchain_id: Uuid,
    pub visibility: OpenXrEyeVisibility,
}

/// Supported composition layer types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OpenXrLayerType {
    Projection,
    Quad,
    Cylinder,
    Cube,
    Equirect,
    Passthrough,
}

/// Per-eye visibility flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OpenXrEyeVisibility {
    Both,
    Left,
    Right,
}

// ---------------------------------------------------------------------------
// Extensions
// ---------------------------------------------------------------------------

/// A runtime extension.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenXrExtension {
    pub name: OpenXrExtensionName,
    pub version: u32,
}

/// Well-known extension identifiers relevant to accessibility verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OpenXrExtensionName {
    /// Hand tracking input.
    #[serde(rename = "XR_EXT_hand_tracking")]
    ExtHandTracking,
    /// Full-body tracking (Meta).
    #[serde(rename = "XR_FB_body_tracking")]
    FbBodyTracking,
    /// Eye gaze interaction.
    #[serde(rename = "XR_EXT_eye_gaze_interaction")]
    ExtEyeGazeInteraction,
    /// Hand interaction poses (Microsoft).
    #[serde(rename = "XR_MSFT_hand_interaction")]
    MsftHandInteraction,
    /// Vive Cosmos controller profile.
    #[serde(rename = "XR_HTC_vive_cosmos_controller_interaction")]
    HtcViveCosmosControllerInteraction,
    /// Passthrough layer (Meta).
    #[serde(rename = "XR_FB_passthrough")]
    FbPassthrough,
    /// Spatial anchors (Microsoft).
    #[serde(rename = "XR_MSFT_spatial_anchor")]
    MsftSpatialAnchor,
    /// Scene understanding (Meta).
    #[serde(rename = "XR_FB_scene")]
    FbScene,
    /// Controller interaction (Meta Touch Pro).
    #[serde(rename = "XR_FB_touch_controller_pro")]
    FbTouchControllerPro,
    /// HP mixed reality controller.
    #[serde(rename = "XR_EXT_hp_mixed_reality_controller")]
    ExtHpMixedRealityController,
    /// Samsung Odyssey controller.
    #[serde(rename = "XR_EXT_samsung_odyssey_controller")]
    ExtSamsungOdysseyController,
}

impl OpenXrExtensionName {
    pub fn spec_name(&self) -> &'static str {
        match self {
            Self::ExtHandTracking => "XR_EXT_hand_tracking",
            Self::FbBodyTracking => "XR_FB_body_tracking",
            Self::ExtEyeGazeInteraction => "XR_EXT_eye_gaze_interaction",
            Self::MsftHandInteraction => "XR_MSFT_hand_interaction",
            Self::HtcViveCosmosControllerInteraction => {
                "XR_HTC_vive_cosmos_controller_interaction"
            }
            Self::FbPassthrough => "XR_FB_passthrough",
            Self::MsftSpatialAnchor => "XR_MSFT_spatial_anchor",
            Self::FbScene => "XR_FB_scene",
            Self::FbTouchControllerPro => "XR_FB_touch_controller_pro",
            Self::ExtHpMixedRealityController => "XR_EXT_hp_mixed_reality_controller",
            Self::ExtSamsungOdysseyController => "XR_EXT_samsung_odyssey_controller",
        }
    }
}

// ---------------------------------------------------------------------------
// Interaction profiles & bindings
// ---------------------------------------------------------------------------

/// An OpenXR interaction profile that maps physical device inputs to actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenXrInteractionProfile {
    /// Canonical profile path, e.g. `/interaction_profiles/oculus/touch_controller`.
    pub profile_path: String,
    /// Human-readable name for display.
    pub display_name: String,
    /// Vendor or standards body that defined this profile.
    pub vendor: String,
    /// Action bindings associated with this profile.
    pub bindings: Vec<OpenXrActionBinding>,
    /// Extensions required to use this profile.
    pub required_extensions: Vec<OpenXrExtensionName>,
    /// Top-level user paths this profile uses.
    pub user_paths: Vec<OpenXrPath>,
}

/// A single binding that maps an action to an input/output path on a device.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenXrActionBinding {
    pub action_name: String,
    pub binding_path: OpenXrPath,
    pub action_type: OpenXrActionType,
}

impl OpenXrActionBinding {
    pub fn new(
        action_name: impl Into<String>,
        binding_path: OpenXrPath,
        action_type: OpenXrActionType,
    ) -> Self {
        Self {
            action_name: action_name.into(),
            binding_path,
            action_type,
        }
    }
}

// ---------------------------------------------------------------------------
// Well-known interaction profile constructors
// ---------------------------------------------------------------------------

/// KHR Simple Controller – baseline two-button profile.
pub fn simple_controller_profile() -> OpenXrInteractionProfile {
    OpenXrInteractionProfile {
        profile_path: "/interaction_profiles/khr/simple_controller".to_string(),
        display_name: "Khronos Simple Controller".to_string(),
        vendor: "Khronos".to_string(),
        bindings: vec![
            OpenXrActionBinding::new("select", OpenXrPath::new("/user/hand/left/input/select/click"), OpenXrActionType::BooleanInput),
            OpenXrActionBinding::new("select", OpenXrPath::new("/user/hand/right/input/select/click"), OpenXrActionType::BooleanInput),
            OpenXrActionBinding::new("menu", OpenXrPath::new("/user/hand/left/input/menu/click"), OpenXrActionType::BooleanInput),
            OpenXrActionBinding::new("menu", OpenXrPath::new("/user/hand/right/input/menu/click"), OpenXrActionType::BooleanInput),
            OpenXrActionBinding::new("grip_pose", OpenXrPath::left_grip_pose(), OpenXrActionType::PoseInput),
            OpenXrActionBinding::new("grip_pose", OpenXrPath::right_grip_pose(), OpenXrActionType::PoseInput),
            OpenXrActionBinding::new("aim_pose", OpenXrPath::left_aim_pose(), OpenXrActionType::PoseInput),
            OpenXrActionBinding::new("aim_pose", OpenXrPath::right_aim_pose(), OpenXrActionType::PoseInput),
            OpenXrActionBinding::new("haptic", OpenXrPath::left_haptic(), OpenXrActionType::VibrationOutput),
            OpenXrActionBinding::new("haptic", OpenXrPath::right_haptic(), OpenXrActionType::VibrationOutput),
        ],
        required_extensions: vec![],
        user_paths: vec![OpenXrPath::left_hand(), OpenXrPath::right_hand()],
    }
}

/// Meta Quest Touch Controller.
pub fn oculus_touch_profile() -> OpenXrInteractionProfile {
    let common_bindings = vec![
        // Triggers
        OpenXrActionBinding::new("trigger", OpenXrPath::left_trigger_value(), OpenXrActionType::FloatInput),
        OpenXrActionBinding::new("trigger", OpenXrPath::right_trigger_value(), OpenXrActionType::FloatInput),
        // Squeeze / grip
        OpenXrActionBinding::new("squeeze", OpenXrPath::left_squeeze_value(), OpenXrActionType::FloatInput),
        OpenXrActionBinding::new("squeeze", OpenXrPath::right_squeeze_value(), OpenXrActionType::FloatInput),
        // Thumbsticks
        OpenXrActionBinding::new("thumbstick", OpenXrPath::left_thumbstick(), OpenXrActionType::Vector2fInput),
        OpenXrActionBinding::new("thumbstick", OpenXrPath::right_thumbstick(), OpenXrActionType::Vector2fInput),
        // Buttons
        OpenXrActionBinding::new("x_button", OpenXrPath::new("/user/hand/left/input/x/click"), OpenXrActionType::BooleanInput),
        OpenXrActionBinding::new("y_button", OpenXrPath::new("/user/hand/left/input/y/click"), OpenXrActionType::BooleanInput),
        OpenXrActionBinding::new("a_button", OpenXrPath::new("/user/hand/right/input/a/click"), OpenXrActionType::BooleanInput),
        OpenXrActionBinding::new("b_button", OpenXrPath::new("/user/hand/right/input/b/click"), OpenXrActionType::BooleanInput),
        // Poses
        OpenXrActionBinding::new("grip_pose", OpenXrPath::left_grip_pose(), OpenXrActionType::PoseInput),
        OpenXrActionBinding::new("grip_pose", OpenXrPath::right_grip_pose(), OpenXrActionType::PoseInput),
        OpenXrActionBinding::new("aim_pose", OpenXrPath::left_aim_pose(), OpenXrActionType::PoseInput),
        OpenXrActionBinding::new("aim_pose", OpenXrPath::right_aim_pose(), OpenXrActionType::PoseInput),
        // Haptics
        OpenXrActionBinding::new("haptic", OpenXrPath::left_haptic(), OpenXrActionType::VibrationOutput),
        OpenXrActionBinding::new("haptic", OpenXrPath::right_haptic(), OpenXrActionType::VibrationOutput),
    ];

    OpenXrInteractionProfile {
        profile_path: "/interaction_profiles/oculus/touch_controller".to_string(),
        display_name: "Meta Quest Touch Controller".to_string(),
        vendor: "Meta".to_string(),
        bindings: common_bindings,
        required_extensions: vec![],
        user_paths: vec![OpenXrPath::left_hand(), OpenXrPath::right_hand()],
    }
}

/// Meta Quest Touch Pro Controller.
pub fn oculus_touch_pro_profile() -> OpenXrInteractionProfile {
    let mut profile = oculus_touch_profile();
    profile.profile_path =
        "/interaction_profiles/oculus/touch_controller_pro".to_string();
    profile.display_name = "Meta Quest Touch Pro Controller".to_string();
    profile.required_extensions = vec![OpenXrExtensionName::FbTouchControllerPro];
    // Touch Pro adds thumb-rest force and stylus-tip pose
    profile.bindings.push(OpenXrActionBinding::new(
        "thumbrest_force",
        OpenXrPath::new("/user/hand/left/input/thumbrest/force"),
        OpenXrActionType::FloatInput,
    ));
    profile.bindings.push(OpenXrActionBinding::new(
        "thumbrest_force",
        OpenXrPath::new("/user/hand/right/input/thumbrest/force"),
        OpenXrActionType::FloatInput,
    ));
    profile
}

/// Valve Index Controller (Knuckles).
pub fn valve_index_profile() -> OpenXrInteractionProfile {
    OpenXrInteractionProfile {
        profile_path: "/interaction_profiles/valve/index_controller".to_string(),
        display_name: "Valve Index Controller".to_string(),
        vendor: "Valve".to_string(),
        bindings: vec![
            OpenXrActionBinding::new("trigger", OpenXrPath::left_trigger_value(), OpenXrActionType::FloatInput),
            OpenXrActionBinding::new("trigger", OpenXrPath::right_trigger_value(), OpenXrActionType::FloatInput),
            OpenXrActionBinding::new("squeeze", OpenXrPath::new("/user/hand/left/input/squeeze/force"), OpenXrActionType::FloatInput),
            OpenXrActionBinding::new("squeeze", OpenXrPath::new("/user/hand/right/input/squeeze/force"), OpenXrActionType::FloatInput),
            OpenXrActionBinding::new("thumbstick", OpenXrPath::left_thumbstick(), OpenXrActionType::Vector2fInput),
            OpenXrActionBinding::new("thumbstick", OpenXrPath::right_thumbstick(), OpenXrActionType::Vector2fInput),
            OpenXrActionBinding::new("trackpad", OpenXrPath::new("/user/hand/left/input/trackpad"), OpenXrActionType::Vector2fInput),
            OpenXrActionBinding::new("trackpad", OpenXrPath::new("/user/hand/right/input/trackpad"), OpenXrActionType::Vector2fInput),
            OpenXrActionBinding::new("a_button", OpenXrPath::new("/user/hand/left/input/a/click"), OpenXrActionType::BooleanInput),
            OpenXrActionBinding::new("b_button", OpenXrPath::new("/user/hand/left/input/b/click"), OpenXrActionType::BooleanInput),
            OpenXrActionBinding::new("a_button", OpenXrPath::new("/user/hand/right/input/a/click"), OpenXrActionType::BooleanInput),
            OpenXrActionBinding::new("b_button", OpenXrPath::new("/user/hand/right/input/b/click"), OpenXrActionType::BooleanInput),
            OpenXrActionBinding::new("grip_pose", OpenXrPath::left_grip_pose(), OpenXrActionType::PoseInput),
            OpenXrActionBinding::new("grip_pose", OpenXrPath::right_grip_pose(), OpenXrActionType::PoseInput),
            OpenXrActionBinding::new("aim_pose", OpenXrPath::left_aim_pose(), OpenXrActionType::PoseInput),
            OpenXrActionBinding::new("aim_pose", OpenXrPath::right_aim_pose(), OpenXrActionType::PoseInput),
            OpenXrActionBinding::new("haptic", OpenXrPath::left_haptic(), OpenXrActionType::VibrationOutput),
            OpenXrActionBinding::new("haptic", OpenXrPath::right_haptic(), OpenXrActionType::VibrationOutput),
        ],
        required_extensions: vec![],
        user_paths: vec![OpenXrPath::left_hand(), OpenXrPath::right_hand()],
    }
}

/// HTC Vive Cosmos Controller.
pub fn htc_vive_cosmos_profile() -> OpenXrInteractionProfile {
    OpenXrInteractionProfile {
        profile_path: "/interaction_profiles/htc/vive_cosmos_controller".to_string(),
        display_name: "HTC Vive Cosmos Controller".to_string(),
        vendor: "HTC".to_string(),
        bindings: vec![
            OpenXrActionBinding::new("trigger", OpenXrPath::left_trigger_value(), OpenXrActionType::FloatInput),
            OpenXrActionBinding::new("trigger", OpenXrPath::right_trigger_value(), OpenXrActionType::FloatInput),
            OpenXrActionBinding::new("squeeze", OpenXrPath::new("/user/hand/left/input/squeeze/click"), OpenXrActionType::BooleanInput),
            OpenXrActionBinding::new("squeeze", OpenXrPath::new("/user/hand/right/input/squeeze/click"), OpenXrActionType::BooleanInput),
            OpenXrActionBinding::new("thumbstick", OpenXrPath::left_thumbstick(), OpenXrActionType::Vector2fInput),
            OpenXrActionBinding::new("thumbstick", OpenXrPath::right_thumbstick(), OpenXrActionType::Vector2fInput),
            OpenXrActionBinding::new("x_button", OpenXrPath::new("/user/hand/left/input/x/click"), OpenXrActionType::BooleanInput),
            OpenXrActionBinding::new("y_button", OpenXrPath::new("/user/hand/left/input/y/click"), OpenXrActionType::BooleanInput),
            OpenXrActionBinding::new("a_button", OpenXrPath::new("/user/hand/right/input/a/click"), OpenXrActionType::BooleanInput),
            OpenXrActionBinding::new("b_button", OpenXrPath::new("/user/hand/right/input/b/click"), OpenXrActionType::BooleanInput),
            OpenXrActionBinding::new("grip_pose", OpenXrPath::left_grip_pose(), OpenXrActionType::PoseInput),
            OpenXrActionBinding::new("grip_pose", OpenXrPath::right_grip_pose(), OpenXrActionType::PoseInput),
            OpenXrActionBinding::new("aim_pose", OpenXrPath::left_aim_pose(), OpenXrActionType::PoseInput),
            OpenXrActionBinding::new("aim_pose", OpenXrPath::right_aim_pose(), OpenXrActionType::PoseInput),
            OpenXrActionBinding::new("haptic", OpenXrPath::left_haptic(), OpenXrActionType::VibrationOutput),
            OpenXrActionBinding::new("haptic", OpenXrPath::right_haptic(), OpenXrActionType::VibrationOutput),
        ],
        required_extensions: vec![
            OpenXrExtensionName::HtcViveCosmosControllerInteraction,
        ],
        user_paths: vec![OpenXrPath::left_hand(), OpenXrPath::right_hand()],
    }
}

/// Microsoft Windows Mixed Reality Controller.
pub fn microsoft_mixed_reality_profile() -> OpenXrInteractionProfile {
    OpenXrInteractionProfile {
        profile_path:
            "/interaction_profiles/microsoft/motion_controller".to_string(),
        display_name: "Windows Mixed Reality Controller".to_string(),
        vendor: "Microsoft".to_string(),
        bindings: vec![
            OpenXrActionBinding::new("trigger", OpenXrPath::left_trigger_value(), OpenXrActionType::FloatInput),
            OpenXrActionBinding::new("trigger", OpenXrPath::right_trigger_value(), OpenXrActionType::FloatInput),
            OpenXrActionBinding::new("squeeze", OpenXrPath::new("/user/hand/left/input/squeeze/click"), OpenXrActionType::BooleanInput),
            OpenXrActionBinding::new("squeeze", OpenXrPath::new("/user/hand/right/input/squeeze/click"), OpenXrActionType::BooleanInput),
            OpenXrActionBinding::new("thumbstick", OpenXrPath::left_thumbstick(), OpenXrActionType::Vector2fInput),
            OpenXrActionBinding::new("thumbstick", OpenXrPath::right_thumbstick(), OpenXrActionType::Vector2fInput),
            OpenXrActionBinding::new("trackpad", OpenXrPath::new("/user/hand/left/input/trackpad"), OpenXrActionType::Vector2fInput),
            OpenXrActionBinding::new("trackpad", OpenXrPath::new("/user/hand/right/input/trackpad"), OpenXrActionType::Vector2fInput),
            OpenXrActionBinding::new("grip_pose", OpenXrPath::left_grip_pose(), OpenXrActionType::PoseInput),
            OpenXrActionBinding::new("grip_pose", OpenXrPath::right_grip_pose(), OpenXrActionType::PoseInput),
            OpenXrActionBinding::new("aim_pose", OpenXrPath::left_aim_pose(), OpenXrActionType::PoseInput),
            OpenXrActionBinding::new("aim_pose", OpenXrPath::right_aim_pose(), OpenXrActionType::PoseInput),
            OpenXrActionBinding::new("haptic", OpenXrPath::left_haptic(), OpenXrActionType::VibrationOutput),
            OpenXrActionBinding::new("haptic", OpenXrPath::right_haptic(), OpenXrActionType::VibrationOutput),
        ],
        required_extensions: vec![],
        user_paths: vec![OpenXrPath::left_hand(), OpenXrPath::right_hand()],
    }
}

/// Microsoft hand interaction profile (controller-free).
pub fn microsoft_hand_interaction_profile() -> OpenXrInteractionProfile {
    OpenXrInteractionProfile {
        profile_path:
            "/interaction_profiles/microsoft/hand_interaction".to_string(),
        display_name: "Microsoft Hand Interaction".to_string(),
        vendor: "Microsoft".to_string(),
        bindings: vec![
            OpenXrActionBinding::new("pinch", OpenXrPath::new("/user/hand/left/input/pinch_ext/value"), OpenXrActionType::FloatInput),
            OpenXrActionBinding::new("pinch", OpenXrPath::new("/user/hand/right/input/pinch_ext/value"), OpenXrActionType::FloatInput),
            OpenXrActionBinding::new("grasp", OpenXrPath::new("/user/hand/left/input/grasp_ext/value"), OpenXrActionType::FloatInput),
            OpenXrActionBinding::new("grasp", OpenXrPath::new("/user/hand/right/input/grasp_ext/value"), OpenXrActionType::FloatInput),
            OpenXrActionBinding::new("grip_pose", OpenXrPath::left_grip_pose(), OpenXrActionType::PoseInput),
            OpenXrActionBinding::new("grip_pose", OpenXrPath::right_grip_pose(), OpenXrActionType::PoseInput),
            OpenXrActionBinding::new("aim_pose", OpenXrPath::left_aim_pose(), OpenXrActionType::PoseInput),
            OpenXrActionBinding::new("aim_pose", OpenXrPath::right_aim_pose(), OpenXrActionType::PoseInput),
        ],
        required_extensions: vec![OpenXrExtensionName::MsftHandInteraction],
        user_paths: vec![OpenXrPath::left_hand(), OpenXrPath::right_hand()],
    }
}

// ---------------------------------------------------------------------------
// Profile → DeviceConfig mapping
// ---------------------------------------------------------------------------

/// Identity 4×4 matrix (column-major flat).
const IDENTITY_4X4: [f64; 16] = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
];

/// Derive interaction types that a profile's bindings can support.
fn interactions_from_bindings(bindings: &[OpenXrActionBinding]) -> Vec<InteractionType> {
    let mut interactions = vec![InteractionType::Click];

    let has = |name: &str| bindings.iter().any(|b| b.action_name == name);
    let has_type = |t: OpenXrActionType| bindings.iter().any(|b| b.action_type == t);

    if has("squeeze") || has("grasp") {
        interactions.push(InteractionType::Grab);
    }
    if has_type(OpenXrActionType::PoseInput) {
        interactions.push(InteractionType::Drag);
    }
    if has_type(OpenXrActionType::FloatInput) {
        interactions.push(InteractionType::Slider);
    }
    if has("thumbstick") || has("trackpad") {
        interactions.push(InteractionType::Dial);
    }
    interactions.push(InteractionType::Hover);
    interactions
}

/// Infer a [`ControllerType`] from the interaction profile path.
fn controller_type_for_profile(path: &str) -> Option<ControllerType> {
    if path.contains("oculus/touch_controller_pro") {
        Some(ControllerType::Quest3Controller)
    } else if path.contains("oculus/touch_controller") {
        Some(ControllerType::QuestTouch)
    } else if path.contains("valve/index") {
        Some(ControllerType::ValveIndex)
    } else if path.contains("htc/vive_cosmos") {
        Some(ControllerType::ViveCosmos)
    } else if path.contains("microsoft/motion") {
        Some(ControllerType::WindowsMR)
    } else if path.contains("khr/simple_controller") {
        Some(ControllerType::GenericGamepad)
    } else {
        None
    }
}

/// Infer a [`DeviceType`] from the interaction profile path.
fn device_type_for_profile(path: &str) -> DeviceType {
    if path.contains("oculus") {
        DeviceType::Standalone
    } else if path.contains("microsoft/hand_interaction") {
        DeviceType::Standalone
    } else {
        DeviceType::Tethered
    }
}

/// Infer a [`MovementMode`] from the interaction profile.
fn movement_mode_for_profile(path: &str) -> MovementMode {
    if path.contains("microsoft/hand_interaction") {
        MovementMode::Standing
    } else if path.contains("valve") || path.contains("htc") {
        MovementMode::RoomScale
    } else {
        MovementMode::RoomScale
    }
}

/// Map an [`OpenXrInteractionProfile`] to a [`DeviceConfig`].
///
/// This is the primary bridge between the OpenXR model and the verifier's
/// internal device representation. Values that cannot be inferred from the
/// profile alone (e.g. exact FoV, refresh rate) use sensible defaults.
pub fn device_config_from_openxr_profile(profile: &OpenXrInteractionProfile) -> DeviceConfig {
    let path = profile.profile_path.as_str();

    let hand_tracking = profile.required_extensions.iter().any(|e| {
        matches!(
            e,
            OpenXrExtensionName::ExtHandTracking | OpenXrExtensionName::MsftHandInteraction
        )
    });

    let eye_tracking = profile
        .required_extensions
        .iter()
        .any(|e| matches!(e, OpenXrExtensionName::ExtEyeGazeInteraction));

    DeviceConfig {
        id: Uuid::new_v4(),
        name: profile.display_name.clone(),
        device_type: device_type_for_profile(path),
        tracking_volume: TrackingVolume::new_room(3.0, 3.0, 2.5),
        supported_interactions: interactions_from_bindings(&profile.bindings),
        movement_mode: movement_mode_for_profile(path),
        controller_offset: IDENTITY_4X4,
        tracking_precision: 0.002,
        refresh_rate: 90.0,
        field_of_view: [100.0, 90.0],
        hand_tracking,
        eye_tracking,
        controller_type: controller_type_for_profile(path),
        constraints: DeviceConstraints::default(),
    }
}

// ---------------------------------------------------------------------------
// Convenience: list all built-in profiles
// ---------------------------------------------------------------------------

/// Return every well-known interaction profile shipped with the verifier.
pub fn all_builtin_profiles() -> Vec<OpenXrInteractionProfile> {
    vec![
        simple_controller_profile(),
        oculus_touch_profile(),
        oculus_touch_pro_profile(),
        valve_index_profile(),
        htc_vive_cosmos_profile(),
        microsoft_mixed_reality_profile(),
        microsoft_hand_interaction_profile(),
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_display() {
        assert_eq!(OpenXrVersion::v1_0().to_string(), "1.0.0");
        assert_eq!(OpenXrVersion::v1_1().to_string(), "1.1.0");
    }

    #[test]
    fn test_session_state_transitions() {
        assert!(!OpenXrSessionState::Idle.accepts_input());
        assert!(!OpenXrSessionState::Visible.accepts_input());
        assert!(OpenXrSessionState::Focused.accepts_input());
        assert!(OpenXrSessionState::Visible.should_render());
        assert!(OpenXrSessionState::Focused.should_render());
        assert!(!OpenXrSessionState::Ready.should_render());
    }

    #[test]
    fn test_reference_space_bounds() {
        assert!(OpenXrReferenceSpaceType::Stage.supports_bounds());
        assert!(!OpenXrReferenceSpaceType::Local.supports_bounds());
        assert!(!OpenXrReferenceSpaceType::View.supports_bounds());
        assert!(!OpenXrReferenceSpaceType::Unbounded.supports_bounds());
    }

    #[test]
    fn test_action_type_is_input() {
        assert!(OpenXrActionType::BooleanInput.is_input());
        assert!(OpenXrActionType::FloatInput.is_input());
        assert!(OpenXrActionType::Vector2fInput.is_input());
        assert!(OpenXrActionType::PoseInput.is_input());
        assert!(!OpenXrActionType::VibrationOutput.is_input());
    }

    #[test]
    fn test_path_display() {
        let p = OpenXrPath::left_trigger_value();
        assert_eq!(p.to_string(), "/user/hand/left/input/trigger/value");
    }

    #[test]
    fn test_simple_controller_profile() {
        let profile = simple_controller_profile();
        assert_eq!(
            profile.profile_path,
            "/interaction_profiles/khr/simple_controller"
        );
        assert!(profile.required_extensions.is_empty());
        assert_eq!(profile.user_paths.len(), 2);
        // Must have select, menu, poses, and haptics
        assert!(profile.bindings.len() >= 8);
    }

    #[test]
    fn test_oculus_touch_profile() {
        let profile = oculus_touch_profile();
        assert_eq!(profile.vendor, "Meta");
        let trigger_bindings: Vec<_> = profile
            .bindings
            .iter()
            .filter(|b| b.action_name == "trigger")
            .collect();
        assert_eq!(trigger_bindings.len(), 2); // left + right
    }

    #[test]
    fn test_oculus_touch_pro_extends_base() {
        let base = oculus_touch_profile();
        let pro = oculus_touch_pro_profile();
        assert!(pro.bindings.len() > base.bindings.len());
        assert!(pro
            .required_extensions
            .contains(&OpenXrExtensionName::FbTouchControllerPro));
    }

    #[test]
    fn test_valve_index_profile() {
        let profile = valve_index_profile();
        assert_eq!(profile.vendor, "Valve");
        let trackpad: Vec<_> = profile
            .bindings
            .iter()
            .filter(|b| b.action_name == "trackpad")
            .collect();
        assert_eq!(trackpad.len(), 2);
    }

    #[test]
    fn test_htc_cosmos_requires_extension() {
        let profile = htc_vive_cosmos_profile();
        assert!(profile
            .required_extensions
            .contains(&OpenXrExtensionName::HtcViveCosmosControllerInteraction));
    }

    #[test]
    fn test_hand_interaction_profile() {
        let profile = microsoft_hand_interaction_profile();
        assert!(profile
            .required_extensions
            .contains(&OpenXrExtensionName::MsftHandInteraction));
        // Hand profile has no haptic outputs
        let haptics: Vec<_> = profile
            .bindings
            .iter()
            .filter(|b| b.action_type == OpenXrActionType::VibrationOutput)
            .collect();
        assert!(haptics.is_empty());
    }

    #[test]
    fn test_device_config_from_oculus_profile() {
        let profile = oculus_touch_profile();
        let config = device_config_from_openxr_profile(&profile);
        assert_eq!(config.name, "Meta Quest Touch Controller");
        assert_eq!(config.device_type, DeviceType::Standalone);
        assert_eq!(config.controller_type, Some(ControllerType::QuestTouch));
        assert!(!config.hand_tracking);
        assert!(!config.eye_tracking);
        assert!(config.supported_interactions.contains(&InteractionType::Click));
        assert!(config.supported_interactions.contains(&InteractionType::Grab));
    }

    #[test]
    fn test_device_config_from_valve_index() {
        let profile = valve_index_profile();
        let config = device_config_from_openxr_profile(&profile);
        assert_eq!(config.device_type, DeviceType::Tethered);
        assert_eq!(config.controller_type, Some(ControllerType::ValveIndex));
        assert_eq!(config.movement_mode, MovementMode::RoomScale);
    }

    #[test]
    fn test_device_config_from_hand_interaction() {
        let profile = microsoft_hand_interaction_profile();
        let config = device_config_from_openxr_profile(&profile);
        assert!(config.hand_tracking);
        assert!(config.controller_type.is_none());
    }

    #[test]
    fn test_all_builtin_profiles() {
        let profiles = all_builtin_profiles();
        assert_eq!(profiles.len(), 7);
        // Every profile must have at least one binding
        for p in &profiles {
            assert!(!p.bindings.is_empty(), "profile {} has no bindings", p.display_name);
        }
    }

    #[test]
    fn test_all_profiles_produce_valid_device_configs() {
        for profile in all_builtin_profiles() {
            let config = device_config_from_openxr_profile(&profile);
            assert!(!config.name.is_empty());
            assert!(!config.supported_interactions.is_empty());
            assert!(config.refresh_rate > 0.0);
            assert!(config.tracking_precision > 0.0);
        }
    }

    #[test]
    fn test_extension_name_spec_strings() {
        assert_eq!(
            OpenXrExtensionName::ExtHandTracking.spec_name(),
            "XR_EXT_hand_tracking"
        );
        assert_eq!(
            OpenXrExtensionName::FbBodyTracking.spec_name(),
            "XR_FB_body_tracking"
        );
        assert_eq!(
            OpenXrExtensionName::ExtEyeGazeInteraction.spec_name(),
            "XR_EXT_eye_gaze_interaction"
        );
    }

    #[test]
    fn test_system_properties_default() {
        let props = OpenXrSystemProperties::default();
        assert!(props.orientation_tracking);
        assert!(props.position_tracking);
        assert_eq!(props.max_swapchain_width, 4096);
    }

    #[test]
    fn test_interactions_from_bindings_minimal() {
        // A profile with only boolean inputs should still yield Click
        let bindings = vec![OpenXrActionBinding::new(
            "select",
            OpenXrPath::new("/user/hand/left/input/select/click"),
            OpenXrActionType::BooleanInput,
        )];
        let interactions = interactions_from_bindings(&bindings);
        assert!(interactions.contains(&InteractionType::Click));
        assert!(interactions.contains(&InteractionType::Hover));
    }
}

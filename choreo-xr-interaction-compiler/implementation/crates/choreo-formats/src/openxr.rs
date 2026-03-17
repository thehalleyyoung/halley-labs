//! OpenXR action manifest generator for Choreo.
//!
//! Converts Choreo interaction declarations into OpenXR-compatible action
//! manifest JSON, including action sets, actions, and per-controller-profile
//! binding suggestions.

use std::collections::HashMap;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum OpenXrError {
    #[error("serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("unknown interaction pattern: {0}")]
    UnknownPattern(String),

    #[error("unsupported controller profile: {0}")]
    UnsupportedProfile(String),

    #[error("no actions defined")]
    NoActions,
}

pub type OpenXrResult<T> = Result<T, OpenXrError>;

// ---------------------------------------------------------------------------
// OpenXR action manifest schema
// ---------------------------------------------------------------------------

/// Top-level OpenXR action manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionManifest {
    pub action_sets: Vec<ActionSet>,
    pub interaction_profiles: Vec<InteractionProfile>,
}

/// A named set of actions (e.g. "choreo_interactions").
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionSet {
    pub name: String,
    pub localized_name: String,
    pub priority: u32,
    pub actions: Vec<Action>,
}

/// A single OpenXR action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    pub name: String,
    pub localized_name: String,
    pub action_type: ActionType,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub subaction_paths: Vec<String>,
}

/// OpenXR action types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionType {
    BooleanInput,
    FloatInput,
    Vector2fInput,
    PoseInput,
    VibrationOutput,
}

/// A controller interaction profile with bindings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionProfile {
    pub profile_path: String,
    pub bindings: Vec<Binding>,
}

/// A binding of an action to a controller component path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Binding {
    pub action: String,
    pub path: String,
}

// ---------------------------------------------------------------------------
// Choreo interaction patterns
// ---------------------------------------------------------------------------

/// Supported Choreo interaction patterns that can map to OpenXR actions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ChoreoInteractionPattern {
    Gaze,
    Grab,
    Touch,
    Proximity,
    Point,
    Custom(String),
}

impl ChoreoInteractionPattern {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "gaze" => Self::Gaze,
            "grab" => Self::Grab,
            "touch" => Self::Touch,
            "proximity" => Self::Proximity,
            "point" => Self::Point,
            other => Self::Custom(other.to_string()),
        }
    }
}

/// A declared interaction to be compiled into OpenXR actions.
#[derive(Debug, Clone)]
pub struct InteractionDeclaration {
    pub name: String,
    pub pattern: ChoreoInteractionPattern,
    pub hands: HandSelection,
}

/// Which hand(s) an interaction applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandSelection {
    Left,
    Right,
    Both,
}

impl HandSelection {
    fn subaction_paths(&self) -> Vec<String> {
        match self {
            HandSelection::Left => vec!["/user/hand/left".to_string()],
            HandSelection::Right => vec!["/user/hand/right".to_string()],
            HandSelection::Both => vec![
                "/user/hand/left".to_string(),
                "/user/hand/right".to_string(),
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Controller profile definitions
// ---------------------------------------------------------------------------

/// Well-known OpenXR interaction profile paths.
pub const PROFILE_KHR_SIMPLE: &str = "/interaction_profiles/khr/simple_controller";
pub const PROFILE_OCULUS_TOUCH: &str = "/interaction_profiles/oculus/touch_controller";
pub const PROFILE_MICROSOFT_MOTION: &str = "/interaction_profiles/microsoft/motion_controller";
pub const PROFILE_HTC_VIVE: &str = "/interaction_profiles/htc/vive_controller";

/// All supported profiles.
pub const SUPPORTED_PROFILES: &[&str] = &[
    PROFILE_KHR_SIMPLE,
    PROFILE_OCULUS_TOUCH,
    PROFILE_MICROSOFT_MOTION,
    PROFILE_HTC_VIVE,
];

/// Per-profile component paths for each interaction pattern.
fn profile_bindings() -> HashMap<&'static str, IndexMap<&'static str, Vec<(&'static str, &'static str)>>>
{
    let mut profiles: HashMap<&str, IndexMap<&str, Vec<(&str, &str)>>> = HashMap::new();

    // --- KHR Simple Controller ---
    let mut khr = IndexMap::new();
    khr.insert(
        "gaze",
        vec![("gaze_tracking", "/user/head/input/pose")],
    );
    khr.insert(
        "grab",
        vec![
            ("grip_pose", "{hand}/input/grip/pose"),
            ("select_click", "{hand}/input/select/click"),
        ],
    );
    khr.insert(
        "touch",
        vec![("select_click", "{hand}/input/select/click")],
    );
    khr.insert(
        "point",
        vec![("aim_pose", "{hand}/input/aim/pose")],
    );
    profiles.insert(PROFILE_KHR_SIMPLE, khr);

    // --- Oculus Touch Controller ---
    let mut oculus = IndexMap::new();
    oculus.insert(
        "gaze",
        vec![("gaze_tracking", "/user/head/input/pose")],
    );
    oculus.insert(
        "grab",
        vec![
            ("grip_pose", "{hand}/input/grip/pose"),
            ("squeeze_value", "{hand}/input/squeeze/value"),
        ],
    );
    oculus.insert(
        "touch",
        vec![
            ("trigger_touch", "{hand}/input/trigger/touch"),
            ("trigger_value", "{hand}/input/trigger/value"),
        ],
    );
    oculus.insert(
        "point",
        vec![("aim_pose", "{hand}/input/aim/pose")],
    );
    profiles.insert(PROFILE_OCULUS_TOUCH, oculus);

    // --- Microsoft Motion Controller ---
    let mut ms = IndexMap::new();
    ms.insert(
        "gaze",
        vec![("gaze_tracking", "/user/head/input/pose")],
    );
    ms.insert(
        "grab",
        vec![
            ("grip_pose", "{hand}/input/grip/pose"),
            ("squeeze_click", "{hand}/input/squeeze/click"),
        ],
    );
    ms.insert(
        "touch",
        vec![("trigger_value", "{hand}/input/trigger/value")],
    );
    ms.insert(
        "point",
        vec![("aim_pose", "{hand}/input/aim/pose")],
    );
    profiles.insert(PROFILE_MICROSOFT_MOTION, ms);

    // --- HTC Vive Controller ---
    let mut vive = IndexMap::new();
    vive.insert(
        "gaze",
        vec![("gaze_tracking", "/user/head/input/pose")],
    );
    vive.insert(
        "grab",
        vec![
            ("grip_pose", "{hand}/input/grip/pose"),
            ("squeeze_click", "{hand}/input/squeeze/click"),
        ],
    );
    vive.insert(
        "touch",
        vec![("trigger_value", "{hand}/input/trigger/value")],
    );
    vive.insert(
        "point",
        vec![("aim_pose", "{hand}/input/aim/pose")],
    );
    profiles.insert(PROFILE_HTC_VIVE, vive);

    profiles
}

fn action_type_for(action_name: &str) -> ActionType {
    if action_name.ends_with("_pose") || action_name == "gaze_tracking" {
        ActionType::PoseInput
    } else if action_name.ends_with("_value") {
        ActionType::FloatInput
    } else if action_name.ends_with("_click") || action_name.ends_with("_touch") {
        ActionType::BooleanInput
    } else {
        ActionType::BooleanInput
    }
}

fn localize_name(name: &str) -> String {
    name.replace('_', " ")
        .split_whitespace()
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                None => String::new(),
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

// ---------------------------------------------------------------------------
// Generator
// ---------------------------------------------------------------------------

/// Generates OpenXR action manifests from Choreo interaction declarations.
pub struct OpenXrManifestGenerator {
    /// Name for the generated action set.
    pub action_set_name: String,
    /// Interaction declarations to map.
    pub interactions: Vec<InteractionDeclaration>,
    /// Which profiles to generate bindings for (default: all supported).
    pub profiles: Vec<String>,
}

impl OpenXrManifestGenerator {
    pub fn new() -> Self {
        Self {
            action_set_name: "choreo_interactions".to_string(),
            interactions: Vec::new(),
            profiles: SUPPORTED_PROFILES.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Add an interaction declaration.
    pub fn add_interaction(&mut self, decl: InteractionDeclaration) {
        self.interactions.push(decl);
    }

    /// Generate the full action manifest.
    pub fn generate_manifest(&self) -> OpenXrResult<ActionManifest> {
        if self.interactions.is_empty() {
            return Err(OpenXrError::NoActions);
        }

        let (actions, all_bindings) = self.build_actions_and_bindings()?;

        let action_set = ActionSet {
            name: self.action_set_name.clone(),
            localized_name: localize_name(&self.action_set_name),
            priority: 0,
            actions,
        };

        let interaction_profiles = all_bindings
            .into_iter()
            .map(|(profile, bindings)| InteractionProfile {
                profile_path: profile,
                bindings,
            })
            .collect();

        Ok(ActionManifest {
            action_sets: vec![action_set],
            interaction_profiles,
        })
    }

    /// Generate the manifest and serialize to a pretty-printed JSON string.
    pub fn generate_manifest_json(&self) -> OpenXrResult<String> {
        let manifest = self.generate_manifest()?;
        serde_json::to_string_pretty(&manifest).map_err(OpenXrError::SerializationError)
    }

    /// Generate per-profile binding suggestions.
    pub fn generate_binding_suggestions(
        &self,
    ) -> OpenXrResult<Vec<InteractionProfile>> {
        let manifest = self.generate_manifest()?;
        Ok(manifest.interaction_profiles)
    }

    // -- internals --

    fn build_actions_and_bindings(
        &self,
    ) -> OpenXrResult<(Vec<Action>, Vec<(String, Vec<Binding>)>)> {
        let binding_db = profile_bindings();

        // Collect all unique actions across all interactions.
        let mut action_map: IndexMap<String, Action> = IndexMap::new();
        // Per-profile bindings.
        let mut profile_bindings_out: IndexMap<String, Vec<Binding>> = IndexMap::new();

        for profile_path in &self.profiles {
            profile_bindings_out
                .entry(profile_path.clone())
                .or_default();
        }

        for decl in &self.interactions {
            let pattern_key = match &decl.pattern {
                ChoreoInteractionPattern::Gaze => "gaze",
                ChoreoInteractionPattern::Grab => "grab",
                ChoreoInteractionPattern::Touch => "touch",
                ChoreoInteractionPattern::Point => "point",
                ChoreoInteractionPattern::Proximity => "gaze", // proximity derived from pose tracking
                ChoreoInteractionPattern::Custom(s) => {
                    return Err(OpenXrError::UnknownPattern(s.clone()));
                }
            };

            let subaction_paths = if pattern_key == "gaze" {
                vec![] // gaze/head tracking has no subaction paths
            } else {
                decl.hands.subaction_paths()
            };

            for profile_path in &self.profiles {
                let profile_db = binding_db.get(profile_path.as_str());
                let entries = profile_db
                    .and_then(|db| db.get(pattern_key))
                    .cloned()
                    .unwrap_or_default();

                for (action_name, component_path) in &entries {
                    let qualified_name = format!("{}_{}", decl.name, action_name);

                    action_map
                        .entry(qualified_name.clone())
                        .or_insert_with(|| Action {
                            name: qualified_name.clone(),
                            localized_name: localize_name(&qualified_name),
                            action_type: action_type_for(action_name),
                            subaction_paths: subaction_paths.clone(),
                        });

                    let concrete_paths = expand_hand_paths(component_path, &decl.hands);
                    let bindings = profile_bindings_out
                        .entry(profile_path.clone())
                        .or_default();
                    for cp in concrete_paths {
                        bindings.push(Binding {
                            action: qualified_name.clone(),
                            path: cp,
                        });
                    }
                }
            }
        }

        let actions: Vec<Action> = action_map.into_values().collect();
        let all_bindings: Vec<(String, Vec<Binding>)> = profile_bindings_out.into_iter().collect();

        Ok((actions, all_bindings))
    }
}

impl Default for OpenXrManifestGenerator {
    fn default() -> Self {
        Self::new()
    }
}

fn expand_hand_paths(template: &str, hands: &HandSelection) -> Vec<String> {
    if !template.contains("{hand}") {
        return vec![template.to_string()];
    }
    match hands {
        HandSelection::Left => {
            vec![template.replace("{hand}", "/user/hand/left")]
        }
        HandSelection::Right => {
            vec![template.replace("{hand}", "/user/hand/right")]
        }
        HandSelection::Both => {
            vec![
                template.replace("{hand}", "/user/hand/left"),
                template.replace("{hand}", "/user/hand/right"),
            ]
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn basic_generator() -> OpenXrManifestGenerator {
        let mut gen = OpenXrManifestGenerator::new();
        gen.add_interaction(InteractionDeclaration {
            name: "object_grab".to_string(),
            pattern: ChoreoInteractionPattern::Grab,
            hands: HandSelection::Both,
        });
        gen
    }

    #[test]
    fn test_generate_manifest_basic() {
        let gen = basic_generator();
        let manifest = gen.generate_manifest().unwrap();
        assert_eq!(manifest.action_sets.len(), 1);
        assert!(!manifest.action_sets[0].actions.is_empty());
        assert!(!manifest.interaction_profiles.is_empty());
    }

    #[test]
    fn test_manifest_json_is_valid() {
        let gen = basic_generator();
        let json = gen.generate_manifest_json().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.get("action_sets").is_some());
        assert!(parsed.get("interaction_profiles").is_some());
    }

    #[test]
    fn test_grab_produces_grip_and_squeeze() {
        let gen = basic_generator();
        let manifest = gen.generate_manifest().unwrap();
        let action_names: Vec<&str> = manifest.action_sets[0]
            .actions
            .iter()
            .map(|a| a.name.as_str())
            .collect();
        assert!(action_names.contains(&"object_grab_grip_pose"));
        // Squeeze name varies by profile, check at least one squeeze action.
        assert!(action_names.iter().any(|n| n.contains("squeeze")));
    }

    #[test]
    fn test_gaze_action() {
        let mut gen = OpenXrManifestGenerator::new();
        gen.add_interaction(InteractionDeclaration {
            name: "look".to_string(),
            pattern: ChoreoInteractionPattern::Gaze,
            hands: HandSelection::Both,
        });
        let manifest = gen.generate_manifest().unwrap();
        let action_names: Vec<&str> = manifest.action_sets[0]
            .actions
            .iter()
            .map(|a| a.name.as_str())
            .collect();
        assert!(action_names.contains(&"look_gaze_tracking"));
        // Gaze should be a PoseInput.
        let gaze_action = manifest.action_sets[0]
            .actions
            .iter()
            .find(|a| a.name == "look_gaze_tracking")
            .unwrap();
        assert_eq!(gaze_action.action_type, ActionType::PoseInput);
    }

    #[test]
    fn test_touch_action() {
        let mut gen = OpenXrManifestGenerator::new();
        gen.add_interaction(InteractionDeclaration {
            name: "button".to_string(),
            pattern: ChoreoInteractionPattern::Touch,
            hands: HandSelection::Right,
        });
        let manifest = gen.generate_manifest().unwrap();
        let actions = &manifest.action_sets[0].actions;
        assert!(!actions.is_empty());
        // Touch on Oculus produces trigger_touch (boolean).
        assert!(actions.iter().any(|a| a.name.contains("trigger")));
    }

    #[test]
    fn test_proximity_maps_to_gaze_pose() {
        let mut gen = OpenXrManifestGenerator::new();
        gen.add_interaction(InteractionDeclaration {
            name: "near".to_string(),
            pattern: ChoreoInteractionPattern::Proximity,
            hands: HandSelection::Both,
        });
        let manifest = gen.generate_manifest().unwrap();
        let action_names: Vec<&str> = manifest.action_sets[0]
            .actions
            .iter()
            .map(|a| a.name.as_str())
            .collect();
        assert!(action_names.contains(&"near_gaze_tracking"));
    }

    #[test]
    fn test_multiple_profiles_generated() {
        let gen = basic_generator();
        let manifest = gen.generate_manifest().unwrap();
        let profile_paths: Vec<&str> = manifest
            .interaction_profiles
            .iter()
            .map(|p| p.profile_path.as_str())
            .collect();
        assert!(profile_paths.contains(&PROFILE_KHR_SIMPLE));
        assert!(profile_paths.contains(&PROFILE_OCULUS_TOUCH));
        assert!(profile_paths.contains(&PROFILE_MICROSOFT_MOTION));
        assert!(profile_paths.contains(&PROFILE_HTC_VIVE));
    }

    #[test]
    fn test_binding_suggestions() {
        let gen = basic_generator();
        let suggestions = gen.generate_binding_suggestions().unwrap();
        assert_eq!(suggestions.len(), 4);
        for profile in &suggestions {
            assert!(!profile.bindings.is_empty());
        }
    }

    #[test]
    fn test_hand_expansion_both() {
        let paths = expand_hand_paths("{hand}/input/grip/pose", &HandSelection::Both);
        assert_eq!(paths.len(), 2);
        assert!(paths.contains(&"/user/hand/left/input/grip/pose".to_string()));
        assert!(paths.contains(&"/user/hand/right/input/grip/pose".to_string()));
    }

    #[test]
    fn test_hand_expansion_single() {
        let paths = expand_hand_paths("{hand}/input/grip/pose", &HandSelection::Left);
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], "/user/hand/left/input/grip/pose");
    }

    #[test]
    fn test_hand_expansion_no_placeholder() {
        let paths = expand_hand_paths("/user/head/input/pose", &HandSelection::Both);
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], "/user/head/input/pose");
    }

    #[test]
    fn test_empty_interactions_error() {
        let gen = OpenXrManifestGenerator::new();
        let result = gen.generate_manifest();
        assert!(result.is_err());
    }

    #[test]
    fn test_custom_pattern_error() {
        let mut gen = OpenXrManifestGenerator::new();
        gen.add_interaction(InteractionDeclaration {
            name: "unknown".to_string(),
            pattern: ChoreoInteractionPattern::Custom("teleport".to_string()),
            hands: HandSelection::Both,
        });
        let result = gen.generate_manifest();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("teleport"));
    }

    #[test]
    fn test_action_type_classification() {
        assert_eq!(action_type_for("grip_pose"), ActionType::PoseInput);
        assert_eq!(action_type_for("gaze_tracking"), ActionType::PoseInput);
        assert_eq!(action_type_for("squeeze_value"), ActionType::FloatInput);
        assert_eq!(action_type_for("trigger_touch"), ActionType::BooleanInput);
        assert_eq!(action_type_for("select_click"), ActionType::BooleanInput);
    }

    #[test]
    fn test_localize_name() {
        assert_eq!(localize_name("grip_pose"), "Grip Pose");
        assert_eq!(
            localize_name("choreo_interactions"),
            "Choreo Interactions"
        );
    }

    #[test]
    fn test_subaction_paths_on_grab() {
        let gen = basic_generator();
        let manifest = gen.generate_manifest().unwrap();
        let grip = manifest.action_sets[0]
            .actions
            .iter()
            .find(|a| a.name == "object_grab_grip_pose")
            .unwrap();
        assert_eq!(grip.subaction_paths.len(), 2);
        assert!(grip
            .subaction_paths
            .contains(&"/user/hand/left".to_string()));
        assert!(grip
            .subaction_paths
            .contains(&"/user/hand/right".to_string()));
    }

    #[test]
    fn test_gaze_no_subaction_paths() {
        let mut gen = OpenXrManifestGenerator::new();
        gen.add_interaction(InteractionDeclaration {
            name: "look".to_string(),
            pattern: ChoreoInteractionPattern::Gaze,
            hands: HandSelection::Both,
        });
        let manifest = gen.generate_manifest().unwrap();
        let gaze = manifest.action_sets[0]
            .actions
            .iter()
            .find(|a| a.name == "look_gaze_tracking")
            .unwrap();
        assert!(gaze.subaction_paths.is_empty());
    }

    #[test]
    fn test_custom_action_set_name() {
        let mut gen = basic_generator();
        gen.action_set_name = "my_app".to_string();
        let manifest = gen.generate_manifest().unwrap();
        assert_eq!(manifest.action_sets[0].name, "my_app");
        assert_eq!(manifest.action_sets[0].localized_name, "My App");
    }

    #[test]
    fn test_selective_profiles() {
        let mut gen = basic_generator();
        gen.profiles = vec![PROFILE_OCULUS_TOUCH.to_string()];
        let manifest = gen.generate_manifest().unwrap();
        assert_eq!(manifest.interaction_profiles.len(), 1);
        assert_eq!(
            manifest.interaction_profiles[0].profile_path,
            PROFILE_OCULUS_TOUCH
        );
    }

    #[test]
    fn test_multiple_interactions() {
        let mut gen = OpenXrManifestGenerator::new();
        gen.add_interaction(InteractionDeclaration {
            name: "grab_obj".to_string(),
            pattern: ChoreoInteractionPattern::Grab,
            hands: HandSelection::Both,
        });
        gen.add_interaction(InteractionDeclaration {
            name: "look_at".to_string(),
            pattern: ChoreoInteractionPattern::Gaze,
            hands: HandSelection::Both,
        });
        gen.add_interaction(InteractionDeclaration {
            name: "tap".to_string(),
            pattern: ChoreoInteractionPattern::Touch,
            hands: HandSelection::Right,
        });
        let manifest = gen.generate_manifest().unwrap();
        let action_names: Vec<&str> = manifest.action_sets[0]
            .actions
            .iter()
            .map(|a| a.name.as_str())
            .collect();
        assert!(action_names.iter().any(|n| n.starts_with("grab_obj_")));
        assert!(action_names.iter().any(|n| n.starts_with("look_at_")));
        assert!(action_names.iter().any(|n| n.starts_with("tap_")));
    }

    #[test]
    fn test_manifest_serde_roundtrip() {
        let gen = basic_generator();
        let json = gen.generate_manifest_json().unwrap();
        let deserialized: ActionManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.action_sets.len(), 1);
        assert!(!deserialized.interaction_profiles.is_empty());
    }

    #[test]
    fn test_point_action() {
        let mut gen = OpenXrManifestGenerator::new();
        gen.add_interaction(InteractionDeclaration {
            name: "laser".to_string(),
            pattern: ChoreoInteractionPattern::Point,
            hands: HandSelection::Right,
        });
        let manifest = gen.generate_manifest().unwrap();
        let action_names: Vec<&str> = manifest.action_sets[0]
            .actions
            .iter()
            .map(|a| a.name.as_str())
            .collect();
        assert!(action_names.contains(&"laser_aim_pose"));
    }

    #[test]
    fn test_pattern_from_str() {
        assert_eq!(
            ChoreoInteractionPattern::from_str("gaze"),
            ChoreoInteractionPattern::Gaze
        );
        assert_eq!(
            ChoreoInteractionPattern::from_str("GRAB"),
            ChoreoInteractionPattern::Grab
        );
        assert_eq!(
            ChoreoInteractionPattern::from_str("Touch"),
            ChoreoInteractionPattern::Touch
        );
        assert_eq!(
            ChoreoInteractionPattern::from_str("something"),
            ChoreoInteractionPattern::Custom("something".to_string())
        );
    }
}

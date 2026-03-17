//! Demo scene generation for testing and showcasing the verifier.
//!
//! Generates realistic XR scene files that exercise various verification
//! features and lint rules.

use uuid::Uuid;

use xr_types::device::DeviceConfig;
use xr_types::geometry::{BoundingBox, Sphere, Capsule, Volume};
use xr_types::scene::{
    ActuatorType, DependencyEdge, DependencyType, FeedbackType, InteractableElement,
    InteractionType, PoseConstraint, SceneModel, VisualProperties,
};

/// Generates demo scenes for testing and demonstration.
pub struct DemoSceneGenerator;

impl DemoSceneGenerator {
    pub fn new() -> Self {
        Self
    }

    /// Generate a simple button panel with 5 buttons at various heights.
    ///
    /// Tests height-related lint rules and basic reachability.
    pub fn simple_button_panel(&self) -> SceneModel {
        let mut scene = SceneModel::new("button_panel");
        scene.description = "Simple button panel with buttons at different heights".into();
        scene.bounds = BoundingBox::from_center_extents([0.0, 1.0, -0.5], [1.0, 2.0, 1.0]);
        scene.devices.push(DeviceConfig::quest_3());

        let heights = [0.8, 1.0, 1.2, 1.4, 1.6];
        let names = ["power", "mode", "confirm", "settings", "emergency"];
        let colors: [[f32; 4]; 5] = [
            [0.8, 0.2, 0.2, 1.0],
            [0.2, 0.6, 0.8, 1.0],
            [0.2, 0.8, 0.2, 1.0],
            [0.6, 0.6, 0.6, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ];

        for i in 0..5 {
            let x_offset = (i as f64 - 2.0) * 0.15;
            let mut elem = InteractableElement::new(
                format!("btn_{}", names[i]),
                [x_offset, heights[i], -0.45],
                InteractionType::Click,
            );
            elem.interaction_type = InteractionType::Click;
            elem.actuator = ActuatorType::Hand;
            elem.activation_volume = Volume::Sphere(Sphere::new(
                [x_offset, heights[i], -0.45],
                0.03,
            ));
            elem.visual = VisualProperties {
                color: colors[i],
                visible: true,
                opacity: 1.0,
                hover_highlight: true,
                label: Some(format!("{} Button", capitalize(names[i]))),
            };
            elem.feedback_type = FeedbackType::VisualHaptic;
            elem.min_duration = 0.0;
            elem.priority = if names[i] == "emergency" { 10 } else { 1 };
            scene.add_element(elem);
        }

        // Sequential dependency: power -> mode -> confirm
        scene.add_dependency(0, 1, DependencyType::Sequential);
        scene.add_dependency(1, 2, DependencyType::Sequential);

        scene.recompute_bounds();
        scene
    }

    /// Generate a VR control room with 20+ interactable elements.
    ///
    /// Tests variety of interaction types, device constraints, and
    /// spatial distribution.
    pub fn vr_control_room(&self) -> SceneModel {
        let mut scene = SceneModel::new("vr_control_room");
        scene.description =
            "VR control room with diverse interactable elements".into();
        scene.bounds = BoundingBox::from_center_extents([0.0, 1.5, 0.0], [3.0, 3.0, 3.0]);
        scene.devices.push(DeviceConfig::quest_3());

        // Main console buttons (row of 6)
        for i in 0..6 {
            let x = (i as f64 - 2.5) * 0.12;
            let mut elem = InteractableElement::new(
                format!("console_btn_{}", i),
                [x, 0.95, -0.5],
                InteractionType::Click,
            );
            elem.interaction_type = InteractionType::Click;
            elem.actuator = ActuatorType::Hand;
            elem.activation_volume = Volume::Sphere(Sphere::new([x, 0.95, -0.5], 0.025));
            elem.visual = VisualProperties {
                color: [0.3, 0.5, 0.9, 1.0],
                visible: true,
                opacity: 1.0,
                hover_highlight: true,
                label: Some(format!("Console Button {}", i + 1)),
            };
            elem.feedback_type = FeedbackType::VisualHaptic;
            scene.add_element(elem);
        }

        // Sliders (3 vertical)
        for i in 0..3 {
            let x = -0.6 + (i as f64) * 0.25;
            let mut elem = InteractableElement::new(
                format!("slider_{}", i),
                [x, 1.15, -0.48],
                InteractionType::Slider,
            );
            elem.interaction_type = InteractionType::Slider;
            elem.actuator = ActuatorType::Hand;
            elem.activation_volume = Volume::Capsule(Capsule::new(
                [x, 1.0, -0.48],
                [x, 1.3, -0.48],
                0.02,
            ));
            elem.visual = VisualProperties {
                color: [0.9, 0.7, 0.2, 1.0],
                visible: true,
                opacity: 1.0,
                hover_highlight: true,
                label: Some(format!("Slider {}", i + 1)),
            };
            elem.feedback_type = FeedbackType::VisualHaptic;
            elem.sustained_contact = true;
            scene.add_element(elem);
        }

        // Dials (2)
        for i in 0..2 {
            let x = 0.4 + (i as f64) * 0.2;
            let mut elem = InteractableElement::new(
                format!("dial_{}", i),
                [x, 1.1, -0.46],
                InteractionType::Dial,
            );
            elem.interaction_type = InteractionType::Dial;
            elem.actuator = ActuatorType::Hand;
            elem.activation_volume = Volume::Sphere(Sphere::new(
                [x, 1.1, -0.46],
                0.04,
            ));
            elem.visual = VisualProperties {
                color: [0.7, 0.3, 0.7, 1.0],
                visible: true,
                opacity: 1.0,
                hover_highlight: true,
                label: Some(format!("Dial {}", i + 1)),
            };
            elem.feedback_type = FeedbackType::All;
            elem.sustained_contact = true;
            scene.add_element(elem);
        }

        // Grab handles (2)
        for i in 0..2 {
            let x = if i == 0 { -0.8 } else { 0.8 };
            let mut elem = InteractableElement::new(
                format!("handle_{}", i),
                [x, 1.3, -0.3],
                InteractionType::Grab,
            );
            elem.interaction_type = InteractionType::Grab;
            elem.actuator = ActuatorType::Hand;
            elem.activation_volume = Volume::Capsule(Capsule::new(
                [x, 1.2, -0.3],
                [x, 1.4, -0.3],
                0.03,
            ));
            elem.visual = VisualProperties {
                color: [0.5, 0.5, 0.5, 1.0],
                visible: true,
                opacity: 1.0,
                hover_highlight: true,
                label: Some(format!("Handle {}", i + 1)),
            };
            elem.feedback_type = FeedbackType::Haptic;
            scene.add_element(elem);
        }

        // Overhead display (hover interaction)
        let mut overhead = InteractableElement::new("overhead_display", [0.0, 1.8, -0.4], InteractionType::Hover);
        overhead.interaction_type = InteractionType::Hover;
        overhead.actuator = ActuatorType::Hand;
        overhead.activation_volume = Volume::Box(BoundingBox::from_center_extents(
            [0.0, 1.8, -0.4],
            [0.3, 0.1, 0.1],
        ));
        overhead.visual = VisualProperties {
            color: [0.1, 0.1, 0.3, 0.8],
            visible: true,
            opacity: 0.8,
            hover_highlight: true,
            label: Some("Overhead Display".into()),
        };
        overhead.feedback_type = FeedbackType::Visual;
        scene.add_element(overhead);

        // Gaze target
        let mut gaze = InteractableElement::new("gaze_target", [0.0, 1.6, -1.5], InteractionType::Gaze);
        gaze.interaction_type = InteractionType::Gaze;
        gaze.actuator = ActuatorType::Eye;
        gaze.activation_volume = Volume::Sphere(Sphere::new([0.0, 1.6, -1.5], 0.15));
        gaze.visual = VisualProperties {
            color: [0.2, 0.8, 0.6, 1.0],
            visible: true,
            opacity: 1.0,
            hover_highlight: true,
            label: Some("Gaze Target".into()),
        };
        gaze.feedback_type = FeedbackType::Visual;
        scene.add_element(gaze);

        // Proximity sensor
        let mut prox = InteractableElement::new("proximity_sensor", [0.0, 1.0, -0.2], InteractionType::Proximity);
        prox.interaction_type = InteractionType::Proximity;
        prox.actuator = ActuatorType::Body;
        prox.activation_volume = Volume::Sphere(Sphere::new([0.0, 1.0, -0.2], 0.3));
        prox.visual = VisualProperties {
            color: [0.0, 0.0, 0.0, 0.0],
            visible: false,
            opacity: 0.0,
            hover_highlight: false,
            label: Some("Proximity Sensor".into()),
        };
        prox.feedback_type = FeedbackType::Audio;
        scene.add_element(prox);

        // Two-handed lever
        let mut lever = InteractableElement::new("main_lever", [0.0, 0.85, -0.35], InteractionType::TwoHanded);
        lever.interaction_type = InteractionType::TwoHanded;
        lever.actuator = ActuatorType::BothHands;
        lever.activation_volume = Volume::Capsule(Capsule::new(
            [-0.15, 0.85, -0.35],
            [0.15, 0.85, -0.35],
            0.04,
        ));
        lever.visual = VisualProperties {
            color: [0.8, 0.2, 0.1, 1.0],
            visible: true,
            opacity: 1.0,
            hover_highlight: true,
            label: Some("Main Lever".into()),
        };
        lever.feedback_type = FeedbackType::HapticAudio;
        lever.sustained_contact = true;
        scene.add_element(lever);

        // Toggle switches (3)
        for i in 0..3 {
            let x = -0.3 + (i as f64) * 0.15;
            let mut elem = InteractableElement::new(
                format!("toggle_{}", i),
                [x, 1.05, -0.49],
                InteractionType::Toggle,
            );
            elem.interaction_type = InteractionType::Toggle;
            elem.actuator = ActuatorType::Hand;
            elem.activation_volume = Volume::Sphere(Sphere::new(
                [x, 1.05, -0.49],
                0.015,
            ));
            elem.visual = VisualProperties {
                color: [0.9, 0.9, 0.9, 1.0],
                visible: true,
                opacity: 1.0,
                hover_highlight: true,
                label: Some(format!("Toggle {}", i + 1)),
            };
            elem.feedback_type = FeedbackType::VisualHaptic;
            scene.add_element(elem);
        }

        // Dependencies: some console buttons enable sliders
        scene.add_dependency(0, 6, DependencyType::Enable); // console_btn_0 enables slider_0
        scene.add_dependency(1, 7, DependencyType::Enable); // console_btn_1 enables slider_1

        scene.recompute_bounds();
        scene
    }

    /// Generate a manufacturing training scenario with multi-step interactions.
    ///
    /// Tests dependency chains, sequential workflows, and complex
    /// interaction patterns.
    pub fn manufacturing_training(&self) -> SceneModel {
        let mut scene = SceneModel::new("manufacturing_training");
        scene.description =
            "Manufacturing training with multi-step assembly workflow".into();
        scene.bounds = BoundingBox::from_center_extents([0.0, 1.0, 0.0], [2.0, 2.5, 2.0]);
        scene.devices.push(DeviceConfig::quest_3());

        // Station 1: Tool selection (grab from rack)
        let tool_names = ["wrench", "screwdriver", "pliers"];
        for (i, name) in tool_names.iter().enumerate() {
            let x = -0.5 + (i as f64) * 0.25;
            let mut elem = InteractableElement::new(
                format!("tool_{}", name),
                [x, 1.3, -0.3],
                InteractionType::Grab,
            );
            elem.interaction_type = InteractionType::Grab;
            elem.actuator = ActuatorType::Hand;
            elem.activation_volume = Volume::Capsule(Capsule::new(
                [x, 1.2, -0.3],
                [x, 1.4, -0.3],
                0.025,
            ));
            elem.visual = VisualProperties {
                color: [0.6, 0.6, 0.6, 1.0],
                visible: true,
                opacity: 1.0,
                hover_highlight: true,
                label: Some(capitalize(name)),
            };
            elem.feedback_type = FeedbackType::VisualHaptic;
            scene.add_element(elem);
        }

        // Station 2: Component placement (drag to target)
        let components = ["bracket", "plate", "gasket"];
        for (i, name) in components.iter().enumerate() {
            let z = 0.2 + (i as f64) * 0.2;
            let mut elem = InteractableElement::new(
                format!("place_{}", name),
                [0.3, 0.9, z],
                InteractionType::Drag,
            );
            elem.interaction_type = InteractionType::Drag;
            elem.actuator = ActuatorType::Hand;
            elem.activation_volume = Volume::Box(BoundingBox::from_center_extents(
                [0.3, 0.9, z],
                [0.08, 0.04, 0.08],
            ));
            elem.visual = VisualProperties {
                color: [0.3, 0.7, 0.3, 1.0],
                visible: true,
                opacity: 1.0,
                hover_highlight: true,
                label: Some(format!("Place {}", capitalize(name))),
            };
            elem.feedback_type = FeedbackType::All;
            elem.sustained_contact = true;
            scene.add_element(elem);
        }

        // Station 3: Fastening (click targets)
        for i in 0..4 {
            let angle = (i as f64) * std::f64::consts::FRAC_PI_2;
            let x = 0.1 * angle.cos();
            let z = 0.1 * angle.sin();
            let mut elem = InteractableElement::new(
                format!("fasten_{}", i),
                [x + 0.3, 0.9, z + 0.8],
                InteractionType::Click,
            );
            elem.interaction_type = InteractionType::Click;
            elem.actuator = ActuatorType::Hand;
            elem.activation_volume = Volume::Sphere(Sphere::new(
                [x + 0.3, 0.9, z + 0.8],
                0.015,
            ));
            elem.visual = VisualProperties {
                color: [0.9, 0.5, 0.1, 1.0],
                visible: true,
                opacity: 1.0,
                hover_highlight: true,
                label: Some(format!("Fasten Point {}", i + 1)),
            };
            elem.feedback_type = FeedbackType::VisualHaptic;
            scene.add_element(elem);
        }

        // Station 4: Quality check (two-handed inspection)
        let mut inspect = InteractableElement::new("quality_inspect", [0.3, 1.1, 1.1], InteractionType::TwoHanded);
        inspect.interaction_type = InteractionType::TwoHanded;
        inspect.actuator = ActuatorType::BothHands;
        inspect.activation_volume = Volume::Box(BoundingBox::from_center_extents(
            [0.3, 1.1, 1.1],
            [0.15, 0.1, 0.15],
        ));
        inspect.visual = VisualProperties {
            color: [0.2, 0.2, 0.8, 1.0],
            visible: true,
            opacity: 1.0,
            hover_highlight: true,
            label: Some("Quality Inspection".into()),
        };
        inspect.feedback_type = FeedbackType::All;
        inspect.min_duration = 2.0;
        scene.add_element(inspect);

        // Confirmation button
        let mut confirm = InteractableElement::new("confirm_assembly", [0.0, 1.0, 1.3], InteractionType::Click);
        confirm.interaction_type = InteractionType::Click;
        confirm.actuator = ActuatorType::Hand;
        confirm.activation_volume = Volume::Sphere(Sphere::new([0.0, 1.0, 1.3], 0.04));
        confirm.visual = VisualProperties {
            color: [0.1, 0.9, 0.1, 1.0],
            visible: true,
            opacity: 1.0,
            hover_highlight: true,
            label: Some("Confirm Assembly".into()),
        };
        confirm.feedback_type = FeedbackType::All;
        scene.add_element(confirm);

        // Dependencies: tool_0 → place_0 → fasten_0..3 → inspect → confirm
        // Tool selection enables placement
        for i in 0..3 {
            scene.add_dependency(i, 3 + i, DependencyType::Sequential);
        }

        // Placement enables fastening
        for i in 0..3 {
            for j in 0..4 {
                scene.add_dependency(3 + i, 6 + j, DependencyType::Enable);
            }
        }

        // All fastening points must be done before inspection
        for j in 0..4 {
            scene.add_dependency(6 + j, 10, DependencyType::Sequential);
        }

        // Inspection enables confirmation
        scene.add_dependency(10, 11, DependencyType::Sequential);

        scene.recompute_bounds();
        scene
    }

    /// Generate an accessibility showcase that triggers various lint rules.
    ///
    /// Contains elements specifically designed to trigger each lint rule,
    /// including too-low, too-high, too-close, missing labels, etc.
    pub fn accessibility_showcase(&self) -> SceneModel {
        let mut scene = SceneModel::new("accessibility_showcase");
        scene.description =
            "Showcase scene designed to trigger all lint rules for testing".into();
        scene.bounds = BoundingBox::from_center_extents([0.0, 1.0, 0.0], [3.0, 3.0, 3.0]);
        scene.devices.push(DeviceConfig::quest_3());

        // Element too low (triggers HeightTooLow)
        let mut too_low = InteractableElement::new("too_low_btn", [0.0, 0.1, -0.5], InteractionType::Click);
        too_low.interaction_type = InteractionType::Click;
        too_low.actuator = ActuatorType::Hand;
        too_low.activation_volume = Volume::Sphere(Sphere::new([0.0, 0.1, -0.5], 0.03));
        too_low.visual.label = Some("Floor Button".into());
        too_low.feedback_type = FeedbackType::Visual;
        scene.add_element(too_low);

        // Element too high (triggers HeightTooHigh)
        let mut too_high = InteractableElement::new("too_high_btn", [0.0, 2.8, -0.5], InteractionType::Click);
        too_high.interaction_type = InteractionType::Click;
        too_high.actuator = ActuatorType::Hand;
        too_high.activation_volume = Volume::Sphere(Sphere::new([0.0, 2.8, -0.5], 0.03));
        too_high.visual.label = Some("Ceiling Button".into());
        too_high.feedback_type = FeedbackType::Visual;
        scene.add_element(too_high);

        // Two elements too close (triggers ElementSpacingTooSmall)
        let mut close_a = InteractableElement::new("close_a", [0.5, 1.2, -0.5], InteractionType::Click);
        close_a.interaction_type = InteractionType::Click;
        close_a.actuator = ActuatorType::Hand;
        close_a.activation_volume = Volume::Sphere(Sphere::new([0.5, 1.2, -0.5], 0.02));
        close_a.visual.label = Some("Close A".into());
        close_a.feedback_type = FeedbackType::Visual;
        scene.add_element(close_a);

        let mut close_b = InteractableElement::new("close_b", [0.51, 1.2, -0.5], InteractionType::Click);
        close_b.interaction_type = InteractionType::Click;
        close_b.actuator = ActuatorType::Hand;
        close_b.activation_volume = Volume::Sphere(Sphere::new([0.51, 1.2, -0.5], 0.02));
        close_b.visual.label = Some("Close B".into());
        close_b.feedback_type = FeedbackType::Visual;
        scene.add_element(close_b);

        // Element with tiny activation volume (triggers VolumeTooSmall)
        let mut tiny = InteractableElement::new("tiny_btn", [-0.5, 1.2, -0.5], InteractionType::Click);
        tiny.interaction_type = InteractionType::Click;
        tiny.actuator = ActuatorType::Hand;
        tiny.activation_volume = Volume::Sphere(Sphere::new([-0.5, 1.2, -0.5], 0.0001));
        tiny.visual.label = Some("Tiny Button".into());
        tiny.feedback_type = FeedbackType::Visual;
        scene.add_element(tiny);

        // Element outside bounds (triggers OutOfBounds)
        let mut out_of_bounds = InteractableElement::new("outside_btn", [10.0, 1.2, 10.0], InteractionType::Click);
        out_of_bounds.interaction_type = InteractionType::Click;
        out_of_bounds.actuator = ActuatorType::Hand;
        out_of_bounds.activation_volume =
            Volume::Sphere(Sphere::new([10.0, 1.2, 10.0], 0.03));
        out_of_bounds.visual.label = Some("Outside Button".into());
        out_of_bounds.feedback_type = FeedbackType::Visual;
        scene.add_element(out_of_bounds);

        // Element missing label (triggers MissingLabel)
        let mut no_label = InteractableElement::new("no_label_btn", [-0.3, 1.2, -0.5], InteractionType::Click);
        no_label.interaction_type = InteractionType::Click;
        no_label.actuator = ActuatorType::Hand;
        no_label.activation_volume = Volume::Sphere(Sphere::new([-0.3, 1.2, -0.5], 0.03));
        no_label.visual.label = None; // Missing!
        no_label.feedback_type = FeedbackType::Visual;
        scene.add_element(no_label);

        // Element missing feedback (triggers MissingFeedback)
        let mut no_feedback = InteractableElement::new("no_feedback_btn", [0.3, 1.2, -0.3], InteractionType::Click);
        no_feedback.interaction_type = InteractionType::Click;
        no_feedback.actuator = ActuatorType::Hand;
        no_feedback.activation_volume =
            Volume::Sphere(Sphere::new([0.3, 1.2, -0.3], 0.03));
        no_feedback.visual.label = Some("No Feedback".into());
        no_feedback.feedback_type = FeedbackType::None; // Missing!
        scene.add_element(no_feedback);

        // TwoHanded with wrong actuator (triggers TwoHandedIncomplete)
        let mut bad_twohanded = InteractableElement::new("bad_twohanded", [0.0, 1.0, -0.4], InteractionType::TwoHanded);
        bad_twohanded.interaction_type = InteractionType::TwoHanded;
        bad_twohanded.actuator = ActuatorType::Hand; // Should be BothHands!
        bad_twohanded.activation_volume =
            Volume::Sphere(Sphere::new([0.0, 1.0, -0.4], 0.05));
        bad_twohanded.visual.label = Some("Bad TwoHanded".into());
        bad_twohanded.feedback_type = FeedbackType::Visual;
        scene.add_element(bad_twohanded);

        // Good element for comparison
        let mut good = InteractableElement::new("good_btn", [0.0, 1.2, -0.5], InteractionType::Click);
        good.interaction_type = InteractionType::Click;
        good.actuator = ActuatorType::Hand;
        good.activation_volume = Volume::Sphere(Sphere::new([0.0, 1.2, -0.5], 0.03));
        good.visual = VisualProperties {
            color: [0.2, 0.8, 0.2, 1.0],
            visible: true,
            opacity: 1.0,
            hover_highlight: true,
            label: Some("Good Button".into()),
        };
        good.feedback_type = FeedbackType::All;
        scene.add_element(good);

        // Gaze element (requires eye tracking device support)
        let mut gaze = InteractableElement::new("gaze_only", [0.0, 1.5, -1.0], InteractionType::Gaze);
        gaze.interaction_type = InteractionType::Gaze;
        gaze.actuator = ActuatorType::Eye;
        gaze.activation_volume = Volume::Sphere(Sphere::new([0.0, 1.5, -1.0], 0.1));
        gaze.visual.label = Some("Gaze Only".into());
        gaze.feedback_type = FeedbackType::Visual;
        scene.add_element(gaze);

        // Voice command element
        let mut voice = InteractableElement::new("voice_cmd", [0.0, 1.5, -0.8], InteractionType::Voice);
        voice.interaction_type = InteractionType::Voice;
        voice.actuator = ActuatorType::Head;
        voice.activation_volume = Volume::Sphere(Sphere::new([0.0, 1.5, -0.8], 0.2));
        voice.visual.label = Some("Voice Command".into());
        voice.feedback_type = FeedbackType::VisualAudio;
        scene.add_element(voice);

        // Gesture element
        let mut gesture = InteractableElement::new("gesture_wave", [-0.5, 1.3, -0.6], InteractionType::Gesture);
        gesture.interaction_type = InteractionType::Gesture;
        gesture.actuator = ActuatorType::Hand;
        gesture.activation_volume =
            Volume::Sphere(Sphere::new([-0.5, 1.3, -0.6], 0.15));
        gesture.visual.label = Some("Wave Gesture".into());
        gesture.feedback_type = FeedbackType::Visual;
        scene.add_element(gesture);

        scene.recompute_bounds();
        scene
    }

    /// Generate a repaired version of the accessibility showcase.
    ///
    /// This scene preserves element identities from the original broken
    /// showcase while applying practical layout and affordance fixes so that
    /// a demo can show a before/after remediation flow.
    pub fn accessibility_showcase_remediated(&self) -> SceneModel {
        let mut scene = self.accessibility_showcase();
        self.apply_accessibility_showcase_remediation(&mut scene);
        scene
    }

    /// Return the broken/remediated pair used by the showcase bundle.
    pub fn accessibility_showcase_pair(&self) -> (SceneModel, SceneModel) {
        let before = self.accessibility_showcase();
        let mut after = before.clone();
        self.apply_accessibility_showcase_remediation(&mut after);
        (before, after)
    }

    /// Human-readable repair notes for the accessibility remediation story.
    pub fn accessibility_showcase_repair_notes(&self) -> Vec<String> {
        vec![
            "Raised the floor-level and ceiling-level controls into an ergonomic reach envelope.".into(),
            "Separated clustered controls and enlarged the tiny activation target for comfortable selection.".into(),
            "Moved the out-of-bounds affordance back into the main workspace volume.".into(),
            "Restored missing labels and feedback channels so the UI is understandable and responsive.".into(),
            "Fixed the two-handed affordance to require `BothHands` and richer feedback, matching the intended interaction semantics.".into(),
        ]
    }

    fn find_element_mut<'a>(
        &self,
        scene: &'a mut SceneModel,
        name: &str,
    ) -> Option<&'a mut InteractableElement> {
        scene.elements.iter_mut().find(|elem| elem.name == name)
    }

    fn update_sphere_element(
        &self,
        scene: &mut SceneModel,
        name: &str,
        position: [f64; 3],
        radius: Option<f64>,
    ) {
        if let Some(elem) = self.find_element_mut(scene, name) {
            elem.position = position;
            if let Volume::Sphere(sphere) = &mut elem.activation_volume {
                sphere.center = position;
                if let Some(radius) = radius {
                    sphere.radius = radius;
                }
            }
        }
    }

    fn apply_accessibility_showcase_remediation(&self, scene: &mut SceneModel) {
        scene.name = "accessibility_showcase_remediated".into();
        scene.description =
            "Accessibility showcase after applying remediation fixes for live demo comparison".into();

        // Raise the floor-level button into a comfortable seated/standing band.
        self.update_sphere_element(scene, "too_low_btn", [0.0, 0.85, -0.5], Some(0.04));

        // Lower the ceiling button to an ergonomic reachable zone.
        self.update_sphere_element(scene, "too_high_btn", [0.0, 1.55, -0.5], Some(0.04));

        // Add spacing between clustered controls.
        self.update_sphere_element(scene, "close_a", [0.46, 1.2, -0.5], Some(0.025));
        self.update_sphere_element(scene, "close_b", [0.60, 1.2, -0.5], Some(0.025));

        // Make the tiny target large enough for comfortable activation.
        self.update_sphere_element(scene, "tiny_btn", [-0.5, 1.2, -0.5], Some(0.028));

        // Move the out-of-bounds element back into the usable workspace.
        self.update_sphere_element(scene, "outside_btn", [0.78, 1.28, -0.44], Some(0.035));

        // Restore missing label and feedback.
        if let Some(elem) = self.find_element_mut(scene, "no_label_btn") {
            elem.visual.label = Some("Primary Action".into());
            elem.visual.hover_highlight = true;
        }
        if let Some(elem) = self.find_element_mut(scene, "no_feedback_btn") {
            elem.feedback_type = FeedbackType::VisualHaptic;
        }

        // Fix the two-handed affordance semantics.
        if let Some(elem) = self.find_element_mut(scene, "bad_twohanded") {
            elem.actuator = ActuatorType::BothHands;
            elem.feedback_type = FeedbackType::All;
            elem.sustained_contact = true;
        }

        // Make the scene feel intentionally polished.
        if let Some(elem) = self.find_element_mut(scene, "good_btn") {
            elem.priority = 10;
            elem.visual.label = Some("Launch Sequence".into());
        }
        if let Some(elem) = self.find_element_mut(scene, "gaze_only") {
            elem.priority = 8;
            elem.visual.label = Some("Status Beacon".into());
        }

        scene.recompute_bounds();
    }
}

impl Default for DemoSceneGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Capitalize the first letter of a string.
fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().to_string() + chars.as_str(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xr_lint::SceneLinter;

    #[test]
    fn test_simple_button_panel() {
        let gen = DemoSceneGenerator::new();
        let scene = gen.simple_button_panel();

        assert_eq!(scene.name, "button_panel");
        assert_eq!(scene.elements.len(), 5);
        assert_eq!(scene.dependencies.len(), 2);
        assert!(!scene.devices.is_empty());

        // All buttons should have labels
        for elem in &scene.elements {
            assert!(
                elem.visual.label.is_some(),
                "Element '{}' missing label",
                elem.name
            );
        }

        // Heights should be in expected range
        let heights: Vec<f64> = scene.elements.iter().map(|e| e.position[1]).collect();
        assert!(heights.iter().all(|&h| h >= 0.8 && h <= 1.6));
    }

    #[test]
    fn test_vr_control_room() {
        let gen = DemoSceneGenerator::new();
        let scene = gen.vr_control_room();

        assert_eq!(scene.name, "vr_control_room");
        assert!(
            scene.elements.len() >= 20,
            "Expected >= 20 elements, got {}",
            scene.elements.len()
        );

        // Check interaction type diversity
        let types: std::collections::HashSet<_> = scene
            .elements
            .iter()
            .map(|e| std::mem::discriminant(&e.interaction_type))
            .collect();
        assert!(
            types.len() >= 6,
            "Expected >= 6 interaction types, got {}",
            types.len()
        );
    }

    #[test]
    fn test_manufacturing_training() {
        let gen = DemoSceneGenerator::new();
        let scene = gen.manufacturing_training();

        assert_eq!(scene.name, "manufacturing_training");
        assert!(scene.elements.len() >= 10);
        assert!(!scene.dependencies.is_empty());

        // Should be a DAG
        assert!(scene.is_dag(), "Manufacturing scene should be a DAG");

        // Should have multi-step depth
        let depth = scene.max_interaction_depth();
        assert!(
            depth >= 3,
            "Expected depth >= 3, got {}",
            depth
        );
    }

    #[test]
    fn test_accessibility_showcase() {
        let gen = DemoSceneGenerator::new();
        let scene = gen.accessibility_showcase();

        assert_eq!(scene.name, "accessibility_showcase");
        assert!(scene.elements.len() >= 10);

        // Run lint — should find multiple issues
        let linter = SceneLinter::new();
        let report = linter.lint(&scene);

        assert!(
            report.has_errors(),
            "Accessibility showcase should trigger lint errors"
        );
        assert!(
            report.findings.len() >= 5,
            "Expected >= 5 findings, got {}",
            report.findings.len()
        );

        // Check specific rules triggered
        let rules: std::collections::HashSet<_> =
            report.findings.iter().map(|f| &f.rule).collect();

        assert!(
            rules.contains(&xr_lint::LintRuleId::HeightTooLow),
            "Should trigger HeightTooLow"
        );
        assert!(
            rules.contains(&xr_lint::LintRuleId::HeightTooHigh),
            "Should trigger HeightTooHigh"
        );
        assert!(
            rules.contains(&xr_lint::LintRuleId::MissingLabel),
            "Should trigger MissingLabel"
        );
        assert!(
            rules.contains(&xr_lint::LintRuleId::MissingFeedback),
            "Should trigger MissingFeedback"
        );
    }

    #[test]
    fn test_accessibility_showcase_remediation_improves_lint_results() {
        let gen = DemoSceneGenerator::new();
        let broken = gen.accessibility_showcase();
        let fixed = gen.accessibility_showcase_remediated();

        assert_eq!(broken.elements.len(), fixed.elements.len());

        let linter = SceneLinter::new();
        let broken_report = linter.lint(&broken);
        let fixed_report = linter.lint(&fixed);

        assert!(
            fixed_report.errors().len() < broken_report.errors().len(),
            "Remediated scene should have fewer lint errors"
        );
        assert!(
            fixed_report.findings.len() < broken_report.findings.len(),
            "Remediated scene should have fewer findings"
        );
    }

    #[test]
    fn test_accessibility_showcase_pair_preserves_element_ids() {
        let gen = DemoSceneGenerator::new();
        let (broken, fixed) = gen.accessibility_showcase_pair();

        for (before, after) in broken.elements.iter().zip(fixed.elements.iter()) {
            assert_eq!(before.id, after.id, "Element ids should remain stable across remediation");
        }
    }

    #[test]
    fn test_capitalize() {
        assert_eq!(capitalize("hello"), "Hello");
        assert_eq!(capitalize(""), "");
        assert_eq!(capitalize("a"), "A");
    }

    #[test]
    fn test_all_scenes_serialize() {
        let gen = DemoSceneGenerator::new();
        let scenes = [
            gen.simple_button_panel(),
            gen.vr_control_room(),
            gen.manufacturing_training(),
            gen.accessibility_showcase(),
        ];

        for scene in &scenes {
            let json = xr_scene::parser::scene_to_json(scene);
            assert!(
                json.is_ok(),
                "Failed to serialize scene '{}': {:?}",
                scene.name,
                json.err()
            );
            let json_str = json.unwrap();
            assert!(json_str.len() > 100);
        }
    }
}

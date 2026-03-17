//! glTF-style scene import demonstration.
//!
//! Constructs a scene graph programmatically (simulating what a real glTF
//! loader would produce), converts it to a `SceneModel`, and runs
//! accessibility verification on the imported scene.

use xr_types::device::DeviceConfig;
use xr_types::geometry::{BoundingBox, Sphere, Volume};
use xr_types::kinematic::BodyParameterRange;
use xr_types::scene::{FeedbackType, InteractableElement, InteractionType, SceneModel};

use xr_lint::tier1_engine::Tier1Engine;
use xr_lint::SceneLinter;

/// Simulate a glTF scene by building it from a hierarchy of nodes.
///
/// A real implementation would parse `.gltf` / `.glb` files; here we
/// construct the equivalent programmatically.
fn import_gltf_scene() -> SceneModel {
    let mut scene = SceneModel::new("imported_gltf_cockpit");
    scene.description = "Simulated glTF import of a VR cockpit scene".into();
    scene.bounds = BoundingBox::from_center_extents([0.0, 1.0, -1.0], [3.0, 2.0, 3.0]);

    // Root node children — panels and controls
    let nodes: Vec<(&str, [f64; 3], InteractionType, f64)> = vec![
        // Main instrument panel
        ("altimeter_dial", [-0.3, 1.1, -0.8], InteractionType::Dial, 0.04),
        ("heading_knob", [0.0, 1.1, -0.8], InteractionType::Dial, 0.03),
        ("throttle_slider", [0.3, 0.9, -0.6], InteractionType::Slider, 0.05),
        // Overhead panel
        ("master_switch", [0.0, 1.8, -0.7], InteractionType::Toggle, 0.03),
        ("light_dimmer", [-0.15, 1.75, -0.7], InteractionType::Slider, 0.03),
        // Side console
        ("radio_button_1", [-0.5, 1.0, -0.5], InteractionType::Click, 0.03),
        ("radio_button_2", [-0.5, 1.05, -0.5], InteractionType::Click, 0.03),
        ("fuel_valve", [0.5, 0.85, -0.5], InteractionType::Grab, 0.04),
        // Yoke / grab handles
        ("yoke_left", [-0.2, 0.9, -0.4], InteractionType::Grab, 0.06),
        ("yoke_right", [0.2, 0.9, -0.4], InteractionType::Grab, 0.06),
    ];

    for (name, pos, itype, radius) in &nodes {
        let mut elem = InteractableElement::new(*name, *pos, *itype);
        elem.activation_volume = Volume::Sphere(Sphere::new(*pos, *radius));
        elem.visual.label = Some(name.replace('_', " "));
        elem.feedback_type = FeedbackType::VisualHaptic;
        scene.add_element(elem);
    }

    // Add a target device
    scene.devices.push(DeviceConfig::quest_3());

    scene
}

fn main() {
    println!("═══════════════════════════════════════════════════════");
    println!("  XR Affordance Verifier – glTF Scene Import");
    println!("═══════════════════════════════════════════════════════\n");

    // --- Import phase --------------------------------------------------------
    println!("── Importing scene ────────────────────────────────────");
    let scene = import_gltf_scene();

    let validation_errors = scene.validate();
    if validation_errors.is_empty() {
        println!("  ✅ Scene validation passed");
    } else {
        println!("  ⚠️  Validation issues:");
        for e in &validation_errors {
            println!("    - {e}");
        }
    }

    println!(
        "  Name:      {}",
        scene.name
    );
    println!(
        "  Elements:  {}",
        scene.elements.len()
    );
    println!(
        "  Devices:   {}",
        scene.devices.len()
    );
    println!(
        "  Bounds:    min={:?}  max={:?}",
        scene.bounds.min, scene.bounds.max
    );

    let type_counts = scene.count_by_type();
    println!("  Interaction types:");
    for (itype, count) in &type_counts {
        println!("    {:?}: {count}", itype);
    }
    println!();

    // --- Lint imported scene --------------------------------------------------
    println!("── Lint Pass ──────────────────────────────────────────");
    let linter = SceneLinter::new();
    let report = linter.lint(&scene);

    for f in &report.findings {
        let sev = match f.severity {
            xr_types::error::Severity::Error | xr_types::error::Severity::Critical => "ERR ",
            xr_types::error::Severity::Warning => "WARN",
            xr_types::error::Severity::Info => "INFO",
        };
        println!(
            "  [{sev}] {} | {}",
            f.element_name.as_deref().unwrap_or("scene"),
            f.message,
        );
    }
    println!(
        "  {} findings in {:.1}ms\n",
        report.findings.len(),
        report.elapsed_ms
    );

    // --- Tier 1 reachability -------------------------------------------------
    println!("── Tier 1 Reachability (full population) ──────────────");
    let range = BodyParameterRange::default();
    let engine = Tier1Engine::new(&range);
    let results = engine.full_scene_check(&scene);

    for r in &results {
        let icon = match r.classification {
            xr_lint::tier1_engine::Classification::Green => "🟢",
            xr_lint::tier1_engine::Classification::Yellow => "🟡",
            xr_lint::tier1_engine::Classification::Red => "🔴",
        };
        println!(
            "  {icon} {:<20} boundary={:+.4}  confidence={:.2}",
            r.element_name, r.distance_to_boundary, r.confidence,
        );
    }

    let green = results
        .iter()
        .filter(|r| matches!(r.classification, xr_lint::tier1_engine::Classification::Green))
        .count();
    let yellow = results
        .iter()
        .filter(|r| r.needs_tier2())
        .count();
    let red = results.len() - green - yellow;

    println!(
        "\n  Summary: {green} green, {yellow} yellow, {red} red (of {})\n",
        results.len()
    );

    println!("Done.");
}

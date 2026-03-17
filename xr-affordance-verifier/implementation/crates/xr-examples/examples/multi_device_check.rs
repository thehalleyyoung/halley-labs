//! Multi-device accessibility check.
//!
//! Builds an XR scene with various interaction types and tests it across
//! multiple headsets (Quest 3, Vision Pro, PSVR2, and a custom Pico 4
//! config) to show which interactions are available on which device and
//! how device capabilities intersect with body reach.

use xr_types::device::DeviceConfig;
use xr_types::geometry::{BoundingBox, Sphere, Volume};
use xr_types::kinematic::BodyParameterRange;
use xr_types::scene::{FeedbackType, InteractableElement, InteractionType, SceneModel};

use xr_lint::tier1_engine::Tier1Engine;

/// Create a Pico 4 configuration (not built-in, so we construct it manually).
fn pico_4_config() -> DeviceConfig {
    let mut cfg = DeviceConfig::quest_3();
    cfg.name = "Pico 4".to_string();
    cfg.supported_interactions = vec![
        InteractionType::Click,
        InteractionType::Grab,
        InteractionType::Drag,
        InteractionType::Hover,
        InteractionType::Gesture,
    ];
    cfg.field_of_view = [105.0, 105.0];
    cfg.hand_tracking = true;
    cfg.eye_tracking = false;
    cfg
}

fn build_scene() -> SceneModel {
    let mut scene = SceneModel::new("Multi-Device Test Scene");
    scene.bounds = BoundingBox::from_center_extents([0.0, 1.2, -0.5], [4.0, 3.0, 4.0]);

    let elements = [
        ("Click Button", [0.0, 1.2, -0.5], InteractionType::Click),
        ("Grab Handle", [0.2, 1.0, -0.4], InteractionType::Grab),
        ("Drag Object", [-0.2, 1.1, -0.45], InteractionType::Drag),
        ("Gaze Target", [0.0, 1.4, -1.0], InteractionType::Gaze),
        ("Voice Command", [0.0, 1.3, -0.5], InteractionType::Voice),
        ("Gesture Bloom", [0.1, 1.2, -0.3], InteractionType::Gesture),
        ("Hover Panel", [-0.1, 1.3, -0.5], InteractionType::Hover),
        ("Two-Hand Resize", [0.0, 1.1, -0.5], InteractionType::TwoHanded),
        ("Proximity Sensor", [0.3, 0.9, -0.3], InteractionType::Proximity),
        ("Slider Control", [-0.3, 1.2, -0.5], InteractionType::Slider),
    ];

    for (name, pos, itype) in &elements {
        let mut elem = InteractableElement::new(*name, *pos, *itype);
        elem.activation_volume = Volume::Sphere(Sphere::new(*pos, 0.04));
        elem.visual.label = Some(name.to_string());
        elem.feedback_type = FeedbackType::Visual;
        scene.add_element(elem);
    }

    scene
}

fn main() {
    println!("═══════════════════════════════════════════════════════");
    println!("  XR Affordance Verifier – Multi-Device Check");
    println!("═══════════════════════════════════════════════════════\n");

    let scene = build_scene();
    let devices = [
        DeviceConfig::quest_3(),
        DeviceConfig::vision_pro(),
        DeviceConfig::psvr2(),
        pico_4_config(),
    ];

    println!(
        "Scene: \"{}\" ({} elements, {} interaction types)\n",
        scene.name,
        scene.elements.len(),
        scene.count_by_type().len(),
    );

    // --- Interaction support matrix ------------------------------------------
    println!("── Interaction Support Matrix ──────────────────────────");
    print!("  {:<20}", "Interaction");
    for d in &devices {
        print!(" {:>12}", d.name);
    }
    println!();
    print!("  {:<20}", "────────────────────");
    for _ in &devices {
        print!(" {:>12}", "────────────");
    }
    println!();

    for elem in &scene.elements {
        print!("  {:<20}", elem.name);
        for d in &devices {
            let supported = d.supports_interaction(elem.interaction_type);
            let icon = if supported { "  ✅" } else { "  ❌" };
            print!(" {:>12}", icon);
        }
        println!();
    }
    println!();

    // --- Per-device summary --------------------------------------------------
    println!("── Per-Device Summary ──────────────────────────────────");
    for d in &devices {
        let supported: Vec<_> = scene
            .elements
            .iter()
            .filter(|e| d.supports_interaction(e.interaction_type))
            .collect();
        let unsupported: Vec<_> = scene
            .elements
            .iter()
            .filter(|e| !d.supports_interaction(e.interaction_type))
            .collect();

        println!("  {} ({:?})", d.name, d.device_type);
        println!(
            "    hand_tracking={} eye_tracking={} fov={:.0}°×{:.0}°",
            d.hand_tracking, d.eye_tracking, d.field_of_view[0], d.field_of_view[1],
        );
        println!(
            "    supported: {}/{}  unsupported: {}",
            supported.len(),
            scene.elements.len(),
            unsupported.len(),
        );
        if !unsupported.is_empty() {
            for u in &unsupported {
                println!("      ❌ {} ({:?})", u.name, u.interaction_type);
            }
        }
        println!();
    }

    // --- Reach × device capability intersection ------------------------------
    println!("── Reach + Device Capability Intersection ─────────────");
    let range = BodyParameterRange::default();
    let engine = Tier1Engine::new(&range);
    let tier1_results = engine.full_scene_check(&scene);

    for d in &devices {
        println!("  {} ──", d.name);
        for (i, r) in tier1_results.iter().enumerate() {
            let elem = &scene.elements[i];
            let device_ok = d.supports_interaction(elem.interaction_type);
            let reach_ok = !matches!(
                r.classification,
                xr_lint::tier1_engine::Classification::Red
            );

            let status = match (device_ok, reach_ok) {
                (true, true) => "✅ accessible",
                (true, false) => "⚠️  out of reach",
                (false, true) => "🚫 unsupported interaction",
                (false, false) => "❌ unreachable & unsupported",
            };
            println!("    {:<20} {status}", elem.name);
        }
        println!();
    }

    println!("Done.");
}

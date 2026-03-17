//! Wheelchair accessibility verification.
//!
//! Builds an XR workspace, configures body parameters for seated (wheelchair)
//! users with reduced reach, verifies element reachability, and generates
//! fix suggestions for inaccessible elements.

use xr_types::geometry::{BoundingBox, Sphere, Volume};
use xr_types::kinematic::BodyParameters;
use xr_types::scene::{FeedbackType, InteractableElement, InteractionType, SceneModel};

use xr_lint::tier1_engine::{Classification, Tier1Engine};
use xr_lint::{LintConfig, SceneLinter};

/// Wheelchair-seated body parameters: shoulder height is lower (seated),
/// and effective reach is reduced because the user cannot lean forward freely.
fn wheelchair_body() -> BodyParameters {
    BodyParameters {
        // Seated shoulder height is roughly 0.58 × stature (vs 0.818 standing).
        // We encode this by scaling stature so that stature × 0.818 ≈ seated shoulder height.
        // For a 1.70 m person seated shoulder height ≈ 1.0 m → effective stature ≈ 1.22 m.
        stature: 1.22,
        arm_length: 0.33,
        shoulder_breadth: 0.44,
        forearm_length: 0.24,
        hand_length: 0.18,
    }
}

fn build_workspace() -> SceneModel {
    let mut scene = SceneModel::new("VR Workspace – Mixed Heights");
    scene.bounds = BoundingBox::from_center_extents([0.0, 1.0, -0.5], [3.0, 2.5, 3.0]);

    let elements = [
        ("Desk Button", [0.0, 0.75, -0.4], InteractionType::Click, 0.04),
        ("Monitor Toggle", [0.2, 1.1, -0.5], InteractionType::Toggle, 0.03),
        ("Filing Cabinet", [-0.3, 0.5, -0.3], InteractionType::Grab, 0.06),
        ("Overhead Light Switch", [0.0, 1.9, -0.5], InteractionType::Click, 0.04),
        ("High Shelf Object", [0.15, 2.0, -0.6], InteractionType::Grab, 0.05),
        ("Wall Panel (mid)", [-0.2, 1.3, -0.55], InteractionType::Slider, 0.05),
        ("Floor Pedal", [0.0, 0.1, -0.3], InteractionType::Click, 0.05),
        ("Desk Dial", [0.3, 0.8, -0.4], InteractionType::Dial, 0.04),
        ("Ceiling Vent Control", [0.0, 2.3, -0.5], InteractionType::Toggle, 0.04),
        ("Side Drawer", [-0.5, 0.6, -0.2], InteractionType::Grab, 0.06),
    ];

    for (name, pos, itype, radius) in &elements {
        let mut elem = InteractableElement::new(*name, *pos, *itype);
        elem.activation_volume = Volume::Sphere(Sphere::new(*pos, *radius));
        elem.visual.label = Some(name.to_string());
        elem.feedback_type = FeedbackType::Visual;
        scene.add_element(elem);
    }

    scene
}

fn suggest_fix(elem: &InteractableElement, seated_shoulder_h: f64, reach: f64) -> String {
    let max_reachable_height = seated_shoulder_h + reach;
    let min_comfortable_height = seated_shoulder_h - reach * 0.7;

    if elem.position[1] > max_reachable_height {
        format!(
            "Lower element from {:.2}m to at most {:.2}m (seated max reach)",
            elem.position[1], max_reachable_height
        )
    } else if elem.position[1] < min_comfortable_height {
        format!(
            "Raise element from {:.2}m to at least {:.2}m (seated min comfort)",
            elem.position[1], min_comfortable_height
        )
    } else {
        let dist = ((elem.position[0].powi(2) + elem.position[2].powi(2)) as f64).sqrt();
        if dist > reach {
            format!(
                "Move element closer: horizontal distance {:.2}m exceeds reach {:.2}m",
                dist, reach
            )
        } else {
            "Element appears reachable; consider increasing activation volume.".into()
        }
    }
}

fn main() {
    println!("═══════════════════════════════════════════════════════");
    println!("  XR Affordance Verifier – Wheelchair Accessibility");
    println!("═══════════════════════════════════════════════════════\n");

    let scene = build_workspace();
    let wheelchair = wheelchair_body();
    let standing = BodyParameters::average_male();

    println!(
        "Scene: \"{}\" ({} elements)\n",
        scene.name,
        scene.elements.len()
    );

    // --- Lint pass -----------------------------------------------------------
    println!("── Lint Pass ──────────────────────────────────────────");
    let linter = SceneLinter::with_config(LintConfig {
        min_element_height: 0.0,
        ..LintConfig::default()
    });
    let lint_report = linter.lint(&scene);
    println!(
        "  {} findings ({} errors, {} warnings)\n",
        lint_report.findings.len(),
        lint_report.errors().len(),
        lint_report.warnings().len(),
    );

    // --- Standing user -------------------------------------------------------
    println!("── Standing user (50th percentile) ────────────────────");
    let standing_engine = Tier1Engine::from_ranges([
        (standing.stature - 0.01, standing.stature + 0.01),
        (standing.arm_length - 0.005, standing.arm_length + 0.005),
        (standing.shoulder_breadth - 0.005, standing.shoulder_breadth + 0.005),
        (standing.forearm_length - 0.005, standing.forearm_length + 0.005),
        (standing.hand_length - 0.005, standing.hand_length + 0.005),
    ]);
    let standing_results = standing_engine.full_scene_check(&scene);

    for r in &standing_results {
        let icon = match r.classification {
            Classification::Green => "🟢",
            Classification::Yellow => "🟡",
            Classification::Red => "🔴",
        };
        println!("  {icon} {:<25} {}", r.element_name, r.classification);
    }
    println!();

    // --- Wheelchair user -----------------------------------------------------
    println!("── Wheelchair user (seated) ────────────────────────────");
    println!(
        "  effective stature={:.2}m  reach={:.3}m  shoulder_h≈{:.2}m\n",
        wheelchair.stature,
        wheelchair.total_reach(),
        wheelchair.shoulder_height(),
    );

    let wc_engine = Tier1Engine::from_ranges([
        (wheelchair.stature - 0.01, wheelchair.stature + 0.01),
        (wheelchair.arm_length - 0.005, wheelchair.arm_length + 0.005),
        (wheelchair.shoulder_breadth - 0.005, wheelchair.shoulder_breadth + 0.005),
        (wheelchair.forearm_length - 0.005, wheelchair.forearm_length + 0.005),
        (wheelchair.hand_length - 0.005, wheelchair.hand_length + 0.005),
    ]);
    let wc_results = wc_engine.full_scene_check(&scene);

    let mut inaccessible = Vec::new();

    for (i, r) in wc_results.iter().enumerate() {
        let icon = match r.classification {
            Classification::Green => "🟢",
            Classification::Yellow => "🟡",
            Classification::Red => "🔴",
        };
        println!("  {icon} {:<25} {}", r.element_name, r.classification);
        if r.classification == Classification::Red {
            inaccessible.push(i);
        }
    }

    // --- Fix suggestions -----------------------------------------------------
    if !inaccessible.is_empty() {
        println!("\n── Fix Suggestions ────────────────────────────────────");
        for &idx in &inaccessible {
            let elem = &scene.elements[idx];
            let fix = suggest_fix(elem, wheelchair.shoulder_height(), wheelchair.total_reach());
            println!("  🔧 {:<25} {}", elem.name, fix);
        }
    }

    println!(
        "\nAccessibility: {}/{} elements reachable from wheelchair\n",
        scene.elements.len() - inaccessible.len(),
        scene.elements.len(),
    );

    println!("Done.");
}

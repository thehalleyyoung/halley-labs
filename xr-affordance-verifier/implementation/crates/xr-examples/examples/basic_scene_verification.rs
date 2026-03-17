//! Basic XR scene accessibility verification.
//!
//! Builds a simple scene with interactable elements, sets up body parameters
//! for different user percentiles, runs Tier 1 linting, and prints results.

use xr_types::geometry::{BoundingBox, Sphere, Volume};
use xr_types::kinematic::{BodyParameterRange, BodyParameters};
use xr_types::scene::{FeedbackType, InteractableElement, InteractionType, SceneModel};

use xr_lint::tier1_engine::Tier1Engine;
use xr_lint::{LintConfig, SceneLinter};

fn build_scene() -> SceneModel {
    let mut scene = SceneModel::new("VR Control Panel");
    scene.bounds = BoundingBox::from_center_extents([0.0, 1.2, -0.5], [3.0, 2.0, 3.0]);

    let elements = [
        ("OK Button", [0.0, 1.2, -0.5], InteractionType::Click, 0.04),
        ("Cancel Button", [0.15, 1.2, -0.5], InteractionType::Click, 0.04),
        ("Volume Slider", [-0.2, 1.3, -0.45], InteractionType::Slider, 0.05),
        ("Grab Handle", [0.0, 0.8, -0.3], InteractionType::Grab, 0.06),
        ("Mode Toggle", [0.1, 1.5, -0.5], InteractionType::Toggle, 0.03),
        ("Settings Dial", [-0.15, 1.6, -0.5], InteractionType::Dial, 0.05),
        ("High Shelf Item", [0.0, 2.1, -0.6], InteractionType::Grab, 0.05),
        ("Floor Button", [0.0, 0.15, -0.5], InteractionType::Click, 0.04),
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

fn main() {
    println!("═══════════════════════════════════════════════════════");
    println!("  XR Affordance Verifier – Basic Scene Verification");
    println!("═══════════════════════════════════════════════════════\n");

    let scene = build_scene();
    println!(
        "Scene: \"{}\" ({} elements)\n",
        scene.name,
        scene.elements.len()
    );

    // --- Lint pass -----------------------------------------------------------
    println!("── Lint Pass ──────────────────────────────────────────");
    let linter = SceneLinter::with_config(LintConfig {
        require_labels: true,
        require_feedback: true,
        ..LintConfig::default()
    });
    let lint_report = linter.lint(&scene);

    if lint_report.has_errors() {
        for f in lint_report.errors() {
            println!(
                "  [ERR]  {} | {} | {}",
                f.rule,
                f.element_name.as_deref().unwrap_or("—"),
                f.message,
            );
            if let Some(ref s) = f.suggestion {
                println!("         ↳ suggestion: {s}");
            }
        }
    }
    for f in lint_report.warnings() {
        println!(
            "  [WARN] {} | {} | {}",
            f.rule,
            f.element_name.as_deref().unwrap_or("—"),
            f.message,
        );
    }
    println!(
        "\n  {} elements checked, {} findings ({:.1} ms)\n",
        lint_report.elements_checked,
        lint_report.findings.len(),
        lint_report.elapsed_ms,
    );

    // --- Tier 1 interval-arithmetic check per body type ----------------------
    let body_types: [(&str, BodyParameters); 3] = [
        ("5th-percentile female", BodyParameters::small_female()),
        ("50th-percentile average", BodyParameters::average_male()),
        ("95th-percentile male", BodyParameters::large_male()),
    ];

    for (label, body) in &body_types {
        println!("── Tier 1 Check: {label} ─────────────────────────");
        println!(
            "   stature={:.3}m  reach={:.3}m  shoulder_h={:.3}m",
            body.stature,
            body.total_reach(),
            body.shoulder_height(),
        );

        let engine = Tier1Engine::from_ranges([
            (body.stature - 0.01, body.stature + 0.01),
            (body.arm_length - 0.005, body.arm_length + 0.005),
            (body.shoulder_breadth - 0.005, body.shoulder_breadth + 0.005),
            (body.forearm_length - 0.005, body.forearm_length + 0.005),
            (body.hand_length - 0.005, body.hand_length + 0.005),
        ]);

        let results = engine.full_scene_check(&scene);

        let mut green = 0usize;
        let mut yellow = 0usize;
        let mut red = 0usize;

        for r in &results {
            let tag = match r.classification {
                xr_lint::tier1_engine::Classification::Green => {
                    green += 1;
                    "🟢"
                }
                xr_lint::tier1_engine::Classification::Yellow => {
                    yellow += 1;
                    "🟡"
                }
                xr_lint::tier1_engine::Classification::Red => {
                    red += 1;
                    "🔴"
                }
            };
            println!(
                "   {tag} {:<20} dist_to_boundary={:+.4}  {}",
                r.element_name, r.distance_to_boundary, r.details,
            );
        }

        println!("   Summary: {green} green, {yellow} yellow, {red} red\n");
    }

    // --- Full population range -----------------------------------------------
    println!("── Tier 1 Check: Full Population (5th–95th) ──────────");
    let range = BodyParameterRange::default();
    let engine = Tier1Engine::new(&range);
    let results = engine.full_scene_check(&scene);

    for r in &results {
        println!(
            "   {:>6} {:<20} boundary={:+.4}  confidence={:.2}",
            r.classification, r.element_name, r.distance_to_boundary, r.confidence,
        );
    }

    let needs_tier2: Vec<_> = results.iter().filter(|r| r.needs_tier2()).collect();
    println!(
        "\n   {} elements need Tier 2 analysis\n",
        needs_tier2.len()
    );

    println!("Done.");
}

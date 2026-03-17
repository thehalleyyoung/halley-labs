//! Coverage certificate generation.
//!
//! Builds a scene, runs Tier 2 certificate generation with stratified
//! sampling, computes ε/δ bounds, and exports the certificate as JSON.

use xr_types::certificate::{
    CoverageCertificate, SampleVerdict, VerifiedRegion, ViolationSeverity,
    ViolationSurface,
};
use xr_types::geometry::{BoundingBox, Sphere, Volume};
use xr_types::kinematic::BodyParameterRange;
use xr_types::scene::{FeedbackType, InteractableElement, InteractionType, SceneModel};

use xr_certificate::{CertificateConfig, CertificateGenerator};

fn build_scene() -> SceneModel {
    let mut scene = SceneModel::new("Certificate Demo Scene");
    scene.bounds = BoundingBox::from_center_extents([0.0, 1.2, -0.5], [3.0, 2.0, 3.0]);

    let elements = [
        ("Main Button", [0.0, 1.2, -0.5], InteractionType::Click),
        ("Slider", [-0.2, 1.3, -0.45], InteractionType::Slider),
        ("Grab Handle", [0.1, 1.0, -0.4], InteractionType::Grab),
        ("Toggle", [0.0, 1.5, -0.5], InteractionType::Toggle),
        ("Dial", [-0.15, 1.1, -0.5], InteractionType::Dial),
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

/// Generate stratified sample verdicts across the body-parameter space.
fn stratified_sample(scene: &SceneModel, num_strata: usize, samples_per_stratum: usize) -> Vec<SampleVerdict> {
    let range = BodyParameterRange::default();
    let min = range.min;
    let max = range.max;
    let mut verdicts = Vec::new();

    for stratum in 0..num_strata {
        let t = stratum as f64 / num_strata as f64;
        let t_next = (stratum + 1) as f64 / num_strata as f64;

        for s in 0..samples_per_stratum {
            let frac = t + (t_next - t) * (s as f64 + 0.5) / samples_per_stratum as f64;
            let body = min.lerp(&max, frac);

            for elem in &scene.elements {
                let shoulder_h = body.shoulder_height();
                let reach = body.total_reach();
                let dx = elem.position[0];
                let dy = elem.position[1] - shoulder_h;
                let dz = elem.position[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                let verdict = if dist <= reach {
                    SampleVerdict::pass(body.to_array().to_vec(), elem.id)
                } else {
                    SampleVerdict::fail(
                        body.to_array().to_vec(),
                        elem.id,
                        format!("distance {dist:.3} > reach {reach:.3}"),
                    )
                };

                verdicts.push(verdict.with_stratum(stratum));
            }
        }
    }

    verdicts
}

fn main() {
    println!("═══════════════════════════════════════════════════════");
    println!("  XR Affordance Verifier – Coverage Certificate");
    println!("═══════════════════════════════════════════════════════\n");

    let scene = build_scene();
    let start = std::time::Instant::now();

    // --- Stratified sampling -------------------------------------------------
    let num_strata = 10;
    let samples_per_stratum = 5;
    println!(
        "Stratified sampling: {num_strata} strata × {samples_per_stratum} samples/stratum × {} elements",
        scene.elements.len(),
    );

    let samples = stratified_sample(&scene, num_strata, samples_per_stratum);
    let n_pass = samples.iter().filter(|s| s.is_pass()).count();
    let n_fail = samples.len() - n_pass;
    println!(
        "  Total samples: {} ({n_pass} pass, {n_fail} fail)\n",
        samples.len(),
    );

    // --- Build some verified regions (simulating SMT results) ----------------
    let verified_regions: Vec<VerifiedRegion> = scene
        .elements
        .iter()
        .map(|elem| {
            VerifiedRegion::new(
                format!("core_region_{}", elem.name),
                vec![1.60, 0.31, 0.39, 0.23, 0.17],
                vec![1.80, 0.37, 0.48, 0.27, 0.20],
                elem.id,
            )
        })
        .collect();

    // --- Flag one element as having a violation surface -----------------------
    let mut violation = ViolationSurface::new(
        "Small-female reach boundary for Dial",
        scene.elements[4].id,
        ViolationSeverity::Low,
    );
    violation.add_sample(vec![1.51, 0.30, 0.38, 0.22, 0.16]);
    violation.estimated_measure = 0.002;

    let violations = vec![violation];

    let elapsed = start.elapsed().as_secs_f64();

    // --- Generate certificate ------------------------------------------------
    println!("── Certificate Generation ─────────────────────────────");
    let generator = CertificateGenerator::with_config(CertificateConfig {
        min_kappa: 0.90,
        target_confidence: 0.95,
        include_samples: true,
        max_linearization_error: 0.01,
    });

    let cert = generator
        .generate(&scene, samples, verified_regions, violations, 0.003, elapsed)
        .expect("certificate generation failed");

    println!("{cert}");

    // --- Validate ------------------------------------------------------------
    let issues = generator.validate_certificate(&cert);
    if issues.is_empty() {
        println!("  ✅ Certificate passes all validation checks.");
    } else {
        println!("  ⚠️  Validation issues:");
        for issue in &issues {
            println!("    - {issue}");
        }
    }
    println!();

    // --- Summary -------------------------------------------------------------
    let summary = cert.summary();
    println!("── Summary ────────────────────────────────────────────");
    println!("  {summary}");
    println!();

    // --- Per-element coverage ------------------------------------------------
    println!("── Per-Element Coverage ────────────────────────────────");
    for elem in &scene.elements {
        let cov = cert.element_coverage.get(&elem.id).copied().unwrap_or(0.0);
        let bar_len = (cov * 30.0) as usize;
        let bar: String = "█".repeat(bar_len) + &"░".repeat(30 - bar_len);
        println!("  {:<20} [{bar}] {:.1}%", elem.name, cov * 100.0);
    }
    println!();

    // --- Export as JSON ------------------------------------------------------
    println!("── JSON Export ────────────────────────────────────────");
    let json = cert.to_json().expect("JSON serialization failed");
    let preview: String = json.chars().take(300).collect();
    println!("  {preview}…");
    println!("  ({} bytes total)\n", json.len());

    // --- Hoeffding bound demonstration ---------------------------------------
    println!("── Hoeffding Bound ε(n, δ) ─────────────────────────────");
    for &n in &[50, 100, 250, 500, 1000, 5000] {
        let eps = CoverageCertificate::compute_epsilon_estimated(n, 0.05);
        println!("  n={n:>5}  δ=0.05  →  ε={eps:.6}");
    }
    println!();

    println!("Done.");
}

//! Shared helpers for XR Affordance Verifier examples and benchmarks.

use xr_types::geometry::{BoundingBox, Sphere, Volume};
use xr_types::scene::{FeedbackType, InteractableElement, InteractionType, SceneModel};

/// Build a scene with `n` interactable elements spread in a grid pattern.
pub fn generate_scene(n: usize) -> SceneModel {
    let mut scene = SceneModel::new(format!("bench_scene_{n}"));
    scene.bounds = BoundingBox::from_center_extents([0.0, 1.2, 0.0], [10.0, 3.0, 10.0]);

    let cols = (n as f64).sqrt().ceil() as usize;
    let spacing = 0.15;

    for i in 0..n {
        let row = i / cols;
        let col = i % cols;
        let x = (col as f64 - cols as f64 / 2.0) * spacing;
        let y = 0.8 + (i % 5) as f64 * 0.25;
        let z = -0.4 - row as f64 * spacing;

        let itype = match i % 5 {
            0 => InteractionType::Click,
            1 => InteractionType::Grab,
            2 => InteractionType::Slider,
            3 => InteractionType::Toggle,
            _ => InteractionType::Hover,
        };

        let mut elem = InteractableElement::new(format!("element_{i}"), [x, y, z], itype);
        elem.activation_volume =
            Volume::Sphere(Sphere::new([x, y, z], 0.03 + (i % 3) as f64 * 0.01));
        elem.visual.label = Some(format!("Element {i}"));
        elem.feedback_type = FeedbackType::Visual;
        scene.add_element(elem);
    }

    scene
}

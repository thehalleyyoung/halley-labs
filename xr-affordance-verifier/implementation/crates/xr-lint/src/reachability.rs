//! Reachability analysis using a sphere-based reach model.
//!
//! For each element in a scene, [`ReachabilityAnalyzer`] determines whether it
//! can be reached by the target population.  The model is:
//!
//!   reach_radius(θ) = arm_length(θ) + forearm_length(θ) + hand_length(θ)
//!
//! centred at the shoulder (position derived from stature and shoulder breadth).
//! An element is reachable if its activation volume intersects the reach sphere.

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use xr_types::geometry::{Sphere, Volume};
use xr_types::kinematic::{BodyParameterRange, BodyParameters};
use xr_types::scene::{InteractableElement, SceneModel};

use crate::linter::{dist3, dist3_sq};

// ── Core types ──────────────────────────────────────────────────────────────

/// Result of a reachability analysis for a single element.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReachabilityResult {
    /// Element id.
    pub element_id: Uuid,
    /// Element name.
    pub element_name: String,
    /// True if reachable by the *entire* target population.
    pub universally_reachable: bool,
    /// Estimated fraction of the population that can reach the element (0.0–1.0).
    pub coverage_fraction: f64,
    /// The smallest body that can still reach the element, if any.
    pub min_reachable_body: Option<BodyParameters>,
    /// The largest body that can reach the element.
    pub max_reachable_body: Option<BodyParameters>,
    /// Distance from the nearest shoulder centre to the element centre.
    pub shoulder_distance: f64,
    /// Required reach radius to touch the activation volume boundary.
    pub required_reach: f64,
    /// Population min and max reach.
    pub reach_range: (f64, f64),
}

/// Summary across all elements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReachabilityReport {
    /// Per-element results.
    pub results: Vec<ReachabilityResult>,
    /// Number of elements universally reachable.
    pub universal_count: usize,
    /// Number of partially reachable elements.
    pub partial_count: usize,
    /// Number of completely unreachable elements.
    pub unreachable_count: usize,
    /// Average coverage fraction across all elements.
    pub mean_coverage: f64,
}

// ── Analyzer ────────────────────────────────────────────────────────────────

/// Reachability analyzer using a simple sphere model.
pub struct ReachabilityAnalyzer {
    body_range: BodyParameterRange,
    /// Number of stratified samples for coverage estimation.
    num_samples: usize,
}

impl ReachabilityAnalyzer {
    /// Create with the default population range.
    pub fn new() -> Self {
        Self {
            body_range: BodyParameterRange::default(),
            num_samples: 32,
        }
    }

    /// Create with a custom population range.
    pub fn with_body_range(range: BodyParameterRange) -> Self {
        Self {
            body_range: range,
            num_samples: 32,
        }
    }

    /// Override the number of samples for coverage estimation.
    pub fn with_samples(mut self, n: usize) -> Self {
        self.num_samples = n.max(4);
        self
    }

    /// Analyse a single element.
    pub fn analyze_element(&self, elem: &InteractableElement) -> ReachabilityResult {
        let min_body = &self.body_range.min;
        let max_body = &self.body_range.max;

        let min_reach = min_body.total_reach();
        let max_reach = max_body.total_reach();

        // Shoulder positions for the *smallest* body (worst case)
        let shoulder_dist = Self::min_shoulder_distance(elem, min_body, max_body);

        // Account for activation volume radius
        let vol_radius = approx_vol_radius(&elem.activation_volume);
        let required = (shoulder_dist - vol_radius).max(0.0);

        let universally_reachable = required <= min_reach;
        let totally_unreachable = required > max_reach;

        let coverage = if universally_reachable {
            1.0
        } else if totally_unreachable {
            0.0
        } else {
            self.estimate_coverage(elem, required)
        };

        let min_reachable_body = if totally_unreachable {
            None
        } else {
            Some(self.find_boundary_body(required, true))
        };

        let max_reachable_body = if totally_unreachable {
            None
        } else {
            Some(*max_body)
        };

        ReachabilityResult {
            element_id: elem.id,
            element_name: elem.name.clone(),
            universally_reachable,
            coverage_fraction: coverage,
            min_reachable_body,
            max_reachable_body,
            shoulder_distance: shoulder_dist,
            required_reach: required,
            reach_range: (min_reach, max_reach),
        }
    }

    /// Analyse all elements in a scene.
    pub fn analyze_scene(&self, scene: &SceneModel) -> ReachabilityReport {
        let results: Vec<_> = scene
            .elements
            .iter()
            .map(|e| self.analyze_element(e))
            .collect();

        let universal_count = results.iter().filter(|r| r.universally_reachable).count();
        let unreachable_count = results.iter().filter(|r| r.coverage_fraction == 0.0).count();
        let partial_count = results.len() - universal_count - unreachable_count;
        let mean_coverage = if results.is_empty() {
            1.0
        } else {
            results.iter().map(|r| r.coverage_fraction).sum::<f64>() / results.len() as f64
        };

        ReachabilityReport {
            results,
            universal_count,
            partial_count,
            unreachable_count,
            mean_coverage,
        }
    }

    /// Compute the minimum distance from any shoulder position in the
    /// population to the element centre.  We consider both left and right
    /// shoulders across the body range.
    fn min_shoulder_distance(
        elem: &InteractableElement,
        min_body: &BodyParameters,
        max_body: &BodyParameters,
    ) -> f64 {
        let mut best = f64::MAX;
        // Sample a few body sizes to find the closest shoulder
        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let body = min_body.lerp(max_body, t);
            let sh = body.shoulder_height();
            let half_sb = body.shoulder_breadth / 2.0;

            let left = [half_sb, sh, 0.0];
            let right = [-half_sb, sh, 0.0];

            let dl = dist3(&elem.position, &left);
            let dr = dist3(&elem.position, &right);
            best = best.min(dl).min(dr);
        }
        best
    }

    /// Estimate coverage fraction by sampling across the body range.
    fn estimate_coverage(&self, elem: &InteractableElement, required: f64) -> f64 {
        let min_body = &self.body_range.min;
        let max_body = &self.body_range.max;
        let n = self.num_samples;
        let mut reachable = 0usize;

        for i in 0..n {
            let t = (i as f64 + 0.5) / n as f64;
            let body = min_body.lerp(max_body, t);
            let reach = body.total_reach();
            if reach >= required {
                reachable += 1;
            }
        }
        reachable as f64 / n as f64
    }

    /// Find the boundary body (smallest that can reach `required`).
    fn find_boundary_body(&self, required: f64, _smallest: bool) -> BodyParameters {
        let min_body = &self.body_range.min;
        let max_body = &self.body_range.max;

        // Binary search for the t where lerp(min, max, t).total_reach() == required
        let mut lo = 0.0_f64;
        let mut hi = 1.0_f64;
        for _ in 0..50 {
            let mid = (lo + hi) / 2.0;
            let body = min_body.lerp(max_body, mid);
            if body.total_reach() < required {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        min_body.lerp(max_body, hi)
    }
}

impl Default for ReachabilityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ── Reach sphere helpers ────────────────────────────────────────────────────

/// Compute the reach sphere for a given body and arm side.
pub fn reach_sphere(body: &BodyParameters, left: bool) -> Sphere {
    let sh = body.shoulder_height();
    let half_sb = body.shoulder_breadth / 2.0;
    let x = if left { half_sb } else { -half_sb };
    Sphere::new([x, sh, 0.0], body.total_reach())
}

/// Check if an activation volume intersects a reach sphere.
pub fn volume_intersects_sphere(vol: &Volume, sphere: &Sphere) -> bool {
    let bb = vol.bounding_box();
    sphere.intersects_box(&bb)
}

/// Compute the minimum distance from any reach sphere (left or right arm,
/// across population) to an element's activation volume boundary.
pub fn min_reach_gap(
    elem: &InteractableElement,
    body_range: &BodyParameterRange,
) -> f64 {
    let mut best_gap = f64::MAX;
    for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let body = body_range.min.lerp(&body_range.max, t);
        for left in [true, false] {
            let s = reach_sphere(&body, left);
            let d = dist3(&elem.position, &s.center);
            let gap = d - s.radius - approx_vol_radius(&elem.activation_volume);
            best_gap = best_gap.min(gap);
        }
    }
    best_gap
}

/// Check if a position is inside a reach sphere for any arm.
pub fn is_point_reachable(pos: &[f64; 3], body: &BodyParameters) -> bool {
    let reach = body.total_reach();
    let sh = body.shoulder_height();
    let half_sb = body.shoulder_breadth / 2.0;

    let left = [half_sb, sh, 0.0];
    let right = [-half_sb, sh, 0.0];

    dist3_sq(pos, &left) <= reach * reach || dist3_sq(pos, &right) <= reach * reach
}

/// Estimate population coverage for a single point.
pub fn point_coverage(
    pos: &[f64; 3],
    body_range: &BodyParameterRange,
    samples: usize,
) -> f64 {
    let n = samples.max(4);
    let mut reachable = 0usize;
    for i in 0..n {
        let t = (i as f64 + 0.5) / n as f64;
        let body = body_range.min.lerp(&body_range.max, t);
        if is_point_reachable(pos, &body) {
            reachable += 1;
        }
    }
    reachable as f64 / n as f64
}

// ── Internal helpers ────────────────────────────────────────────────────────

fn approx_vol_radius(vol: &Volume) -> f64 {
    let bb = vol.bounding_box();
    let he = bb.half_extents();
    (he[0] * he[0] + he[1] * he[1] + he[2] * he[2]).sqrt()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use xr_types::geometry::BoundingBox;
    use xr_types::scene::*;

    fn make_element(name: &str, pos: [f64; 3]) -> InteractableElement {
        let mut e = InteractableElement::new(name, pos);
        e.feedback_type = FeedbackType::VisualHaptic;
        e
    }

    fn make_scene(elements: Vec<InteractableElement>) -> SceneModel {
        let mut scene = SceneModel::new("reach_test");
        scene.bounds = BoundingBox::from_center_extents([0.0, 1.0, 0.0], [5.0, 3.0, 5.0]);
        for e in elements {
            scene.add_element(e);
        }
        scene
    }

    #[test]
    fn test_close_element_universally_reachable() {
        let elem = make_element("close", [0.0, 1.2, -0.3]);
        let analyzer = ReachabilityAnalyzer::new();
        let result = analyzer.analyze_element(&elem);
        assert!(
            result.universally_reachable,
            "Element at 0.3m should be universally reachable"
        );
        assert!((result.coverage_fraction - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_far_element_unreachable() {
        let elem = make_element("far", [0.0, 1.2, -5.0]);
        let analyzer = ReachabilityAnalyzer::new();
        let result = analyzer.analyze_element(&elem);
        assert!(!result.universally_reachable);
        assert!(
            result.coverage_fraction < 0.01,
            "Element at 5m should be unreachable: coverage={}",
            result.coverage_fraction
        );
    }

    #[test]
    fn test_partial_reachability() {
        // Place at a distance between min and max reach
        let range = BodyParameterRange::default();
        let mid_reach = (range.min.total_reach() + range.max.total_reach()) / 2.0;
        let sh = range.min.shoulder_height();
        let elem = make_element("mid", [0.0, sh, -mid_reach]);

        let analyzer = ReachabilityAnalyzer::new();
        let result = analyzer.analyze_element(&elem);
        assert!(
            result.coverage_fraction > 0.1 && result.coverage_fraction < 0.95,
            "Expected partial coverage, got {}",
            result.coverage_fraction
        );
    }

    #[test]
    fn test_scene_analysis() {
        let scene = make_scene(vec![
            make_element("close", [0.0, 1.2, -0.3]),
            make_element("far", [0.0, 1.2, -5.0]),
        ]);
        let analyzer = ReachabilityAnalyzer::new();
        let report = analyzer.analyze_scene(&scene);
        assert_eq!(report.results.len(), 2);
        assert!(report.universal_count >= 1);
        assert!(report.unreachable_count >= 1);
    }

    #[test]
    fn test_reach_sphere_properties() {
        let body = BodyParameters::average_male();
        let left = reach_sphere(&body, true);
        let right = reach_sphere(&body, false);
        assert!(left.center[0] > 0.0, "Left shoulder should be at +x");
        assert!(right.center[0] < 0.0, "Right shoulder should be at -x");
        assert!((left.radius - body.total_reach()).abs() < 1e-12);
    }

    #[test]
    fn test_volume_intersection() {
        let body = BodyParameters::average_male();
        let sphere = reach_sphere(&body, true);
        let near = Volume::Box(BoundingBox::from_center_extents(
            [0.2, 1.2, -0.3],
            [0.05, 0.05, 0.05],
        ));
        assert!(volume_intersects_sphere(&near, &sphere));

        let far = Volume::Box(BoundingBox::from_center_extents(
            [0.0, 1.2, -5.0],
            [0.05, 0.05, 0.05],
        ));
        assert!(!volume_intersects_sphere(&far, &sphere));
    }

    #[test]
    fn test_is_point_reachable() {
        let body = BodyParameters::average_male();
        assert!(is_point_reachable(&[0.2, 1.3, -0.3], &body));
        assert!(!is_point_reachable(&[0.0, 1.3, -5.0], &body));
    }

    #[test]
    fn test_point_coverage() {
        let range = BodyParameterRange::default();
        let close_cov = point_coverage(&[0.0, 1.2, -0.2], &range, 32);
        assert!(close_cov > 0.9);
        let far_cov = point_coverage(&[0.0, 1.2, -5.0], &range, 32);
        assert!(far_cov < 0.05);
    }

    #[test]
    fn test_min_reach_gap_near() {
        let range = BodyParameterRange::default();
        let elem = make_element("near", [0.0, 1.2, -0.3]);
        let gap = min_reach_gap(&elem, &range);
        assert!(gap < 0.0, "Near element should have negative gap (inside)");
    }

    #[test]
    fn test_min_reach_gap_far() {
        let range = BodyParameterRange::default();
        let elem = make_element("far", [0.0, 1.2, -5.0]);
        let gap = min_reach_gap(&elem, &range);
        assert!(gap > 0.0, "Far element should have positive gap (outside)");
    }

    #[test]
    fn test_analyzer_custom_samples() {
        let analyzer = ReachabilityAnalyzer::new().with_samples(64);
        let elem = make_element("btn", [0.0, 1.2, -0.3]);
        let result = analyzer.analyze_element(&elem);
        assert!(result.universally_reachable);
    }

    #[test]
    fn test_boundary_body_is_valid() {
        let analyzer = ReachabilityAnalyzer::new();
        let range = BodyParameterRange::default();
        let mid_reach = (range.min.total_reach() + range.max.total_reach()) / 2.0;
        let boundary = analyzer.find_boundary_body(mid_reach, true);
        assert!(
            (boundary.total_reach() - mid_reach).abs() < 0.005,
            "Boundary body reach {:.4} should be close to required {:.4}",
            boundary.total_reach(),
            mid_reach
        );
    }
}

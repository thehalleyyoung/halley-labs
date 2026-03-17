//! Tier 1 fast verification using affine arithmetic.

use crate::affine::AffineForm;
use crate::interval::Interval;
use crate::bounds::BoundComputer;
use crate::region::RegionClassification;
use crate::interval::IntervalVector;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use uuid::Uuid;
use xr_types::{
    BoundingBox, BodyParameters, InteractableElement, SceneModel, Volume,
    VerifierResult,
};

/// Alias for backward compatibility.
pub type ElementClassification = RegionClassification;

/// Result of Tier 1 verification for a single element.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tier1Result {
    pub element_id: Uuid,
    pub element_name: String,
    pub classification: RegionClassification,
    pub reach_interval: Option<(f64, f64)>,
    pub distance_to_boundary: f64,
    pub elapsed: Duration,
    pub details: String,
}

impl Tier1Result {
    pub fn needs_tier2(&self) -> bool {
        self.classification == RegionClassification::Yellow
    }
}

/// Tier 1 fast verifier using affine arithmetic.
pub struct Tier1Verifier {
    /// Body parameter ranges (lo, hi) for each of the 5 parameters.
    pub param_ranges: [(f64, f64); 5],
    /// Maximum time budget per element.
    pub time_budget: Duration,
}

impl Tier1Verifier {
    pub fn new(param_ranges: [(f64, f64); 5]) -> Self {
        Self {
            param_ranges,
            time_budget: Duration::from_millis(100),
        }
    }

    pub fn with_time_budget(mut self, budget: Duration) -> Self {
        self.time_budget = budget;
        self
    }

    /// Create affine forms for each body parameter.
    fn body_param_affine_forms(&self) -> [AffineForm; 5] {
        let mut forms = Vec::with_capacity(5);
        for &(lo, hi) in &self.param_ranges {
            forms.push(AffineForm::from_interval(&Interval::new(lo, hi)));
        }
        [
            forms[0].clone(),
            forms[1].clone(),
            forms[2].clone(),
            forms[3].clone(),
            forms[4].clone(),
        ]
    }

    /// Compute conservative reach radius using affine arithmetic.
    /// Uses the kinematic model: reach ≈ arm_length + forearm_length + hand_length
    fn compute_reach_envelope_interval(&self) -> Interval {
        let params = self.body_param_affine_forms();
        let arm = &params[1];
        let forearm = &params[3];
        let hand = &params[4];
        let total_reach_af = arm.add(forearm).add(hand);
        total_reach_af.to_interval()
    }

    /// Compute conservative shoulder height interval.
    fn compute_shoulder_height_interval(&self) -> Interval {
        let params = self.body_param_affine_forms();
        let stature = &params[0];
        let shoulder_frac = AffineForm::constant(0.818);
        let shoulder_height = stature.mul(&shoulder_frac);
        shoulder_height.to_interval()
    }

    /// Check a single element using affine arithmetic.
    pub fn check_element(
        &self,
        element: &InteractableElement,
        scene_bounds: &BoundingBox,
    ) -> Tier1Result {
        let start = Instant::now();
        let reach_iv = self.compute_reach_envelope_interval();
        let shoulder_height_iv = self.compute_shoulder_height_interval();

        let elem_pos = element.position;
        let elem_bbox = element.activation_volume.bounding_box();
        let elem_center = elem_bbox.center();

        let shoulder_pos_x = Interval::point(0.0);
        let shoulder_pos_y = shoulder_height_iv;
        let shoulder_pos_z = Interval::point(0.0);

        let dx = Interval::point(elem_center[0]) - shoulder_pos_x;
        let dy = Interval::point(elem_center[1]) - shoulder_pos_y;
        let dz = Interval::point(elem_center[2]) - shoulder_pos_z;

        let dist_sq = dx.sqr() + dy.sqr() + dz.sqr();
        let dist = dist_sq.sqrt().unwrap_or(Interval::point(f64::INFINITY));

        let bbox_radius = {
            let e = elem_bbox.half_extents();
            (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt()
        };
        let effective_dist = Interval::new(
            (dist.lo - bbox_radius).max(0.0),
            dist.hi + bbox_radius,
        );

        let classification = if effective_dist.hi <= reach_iv.lo {
            RegionClassification::Green
        } else if effective_dist.lo > reach_iv.hi {
            RegionClassification::Red
        } else {
            RegionClassification::Yellow
        };

        let distance_to_boundary = if classification == RegionClassification::Green {
            reach_iv.lo - effective_dist.hi
        } else if classification == RegionClassification::Red {
            effective_dist.lo - reach_iv.hi
        } else {
            0.0
        };

        Tier1Result {
            element_id: element.id,
            element_name: element.name.clone(),
            classification,
            reach_interval: Some((reach_iv.lo, reach_iv.hi)),
            distance_to_boundary,
            elapsed: start.elapsed(),
            details: format!(
                "Reach: {}, Dist: {}, Effective: {}",
                reach_iv, dist, effective_dist
            ),
        }
    }

    /// Run Tier 1 verification on all elements in a scene.
    pub fn check_scene(&self, scene: &SceneModel) -> Vec<Tier1Result> {
        scene
            .elements
            .iter()
            .map(|e| self.check_element(e, &scene.bounds))
            .collect()
    }

    /// Summary statistics for a scene check.
    pub fn scene_summary(results: &[Tier1Result]) -> (usize, usize, usize) {
        let green = results
            .iter()
            .filter(|r| r.classification == RegionClassification::Green)
            .count();
        let yellow = results
            .iter()
            .filter(|r| r.classification == RegionClassification::Yellow)
            .count();
        let red = results
            .iter()
            .filter(|r| r.classification == RegionClassification::Red)
            .count();
        (green, yellow, red)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_region_classification() {
        assert!(RegionClassification::Green.is_definite());
        assert!(RegionClassification::Yellow.is_uncertain());
        assert!(RegionClassification::Red.is_definite());
    }

    #[test]
    fn test_verifier_creation() {
        let ranges = [
            (1.5, 1.9),
            (0.25, 0.40),
            (0.35, 0.50),
            (0.22, 0.33),
            (0.16, 0.22),
        ];
        let v = Tier1Verifier::new(ranges);
        let reach = v.compute_reach_envelope_interval();
        assert!(reach.lo > 0.5);
        assert!(reach.hi < 1.2);
    }
}

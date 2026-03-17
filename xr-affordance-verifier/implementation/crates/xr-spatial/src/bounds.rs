//! Bound computation using affine arithmetic and wrapping-factor analysis.
//!
//! Implements conservative reach-envelope bounding per Theorem B1:
//!   w ≤ ∏(1 + cᵢ · Δᵢ²)
//! and Tier 1 classification logic driven by these bounds.

use crate::affine::AffineForm;
use crate::interval::{Interval, IntervalVector};
use crate::region::{ParameterRegion, RegionClassification};
use xr_types::{BoundingBox, KinematicChain};

// ── BoundComputer ───────────────────────────────────────────────────────

/// Conservative bound computation engine.
///
/// Uses affine arithmetic to compute reach-envelope bounds that track
/// correlations between body parameters, yielding tighter enclosures
/// than plain interval arithmetic.
pub struct BoundComputer {
    /// Safety margin (meters) added to every reach bound.
    pub margin: f64,
    /// Sensitivity coefficients for wrapping-factor analysis.
    pub sensitivity_coefficients: [f64; 5],
}

impl BoundComputer {
    pub fn new() -> Self {
        Self {
            margin: 0.01,
            sensitivity_coefficients: [1.0; 5],
        }
    }

    pub fn with_margin(mut self, m: f64) -> Self {
        self.margin = m;
        self
    }

    pub fn with_sensitivity(mut self, coeffs: [f64; 5]) -> Self {
        self.sensitivity_coefficients = coeffs;
        self
    }

    // ── Core reach bound (interval) ─────────────────────────────

    /// Compute conservative reach interval for a parameter region.
    pub fn reach_bound(&self, region: &ParameterRegion) -> Interval {
        let b = &region.bounds;
        if b.dim() < 5 {
            return Interval::new(0.0, 2.0);
        }
        let total = b.components[1] + b.components[3] + b.components[4];
        total.expand(self.margin)
    }

    /// Compute conservative shoulder-height interval.
    pub fn shoulder_height_bound(&self, region: &ParameterRegion) -> Interval {
        let b = &region.bounds;
        if b.dim() < 1 {
            return Interval::new(1.0, 1.8);
        }
        b.components[0] * Interval::new(0.80, 0.84)
    }

    // ── Affine-arithmetic reach bound ───────────────────────────

    /// Compute reach bound using affine arithmetic (tighter than intervals).
    pub fn affine_reach_bound(&self, region: &ParameterRegion) -> AffineForm {
        let b = &region.bounds;
        if b.dim() < 5 {
            return AffineForm::constant(0.7);
        }
        let arm = AffineForm::from_interval(&b.components[1]);
        let forearm = AffineForm::from_interval(&b.components[3]);
        let hand = AffineForm::from_interval(&b.components[4]);
        arm.add(&forearm).add(&hand)
    }

    /// Compute affine shoulder-height bound.
    pub fn affine_shoulder_height_bound(&self, region: &ParameterRegion) -> AffineForm {
        let b = &region.bounds;
        if b.dim() < 1 {
            return AffineForm::constant(1.4);
        }
        AffineForm::from_interval(&b.components[0]).scale(0.818)
    }

    // ── Chain-aware reach bounds ────────────────────────────────

    /// Compute conservative reach-envelope bounding box for a kinematic
    /// chain over a parameter region.
    pub fn compute_reach_bounds(
        &self,
        chain: &KinematicChain,
        param_region: &ParameterRegion,
    ) -> BoundingBox {
        let b = &param_region.bounds;
        if b.dim() < 5 {
            return self.workspace_bbox(param_region);
        }

        let mut total_reach_af = AffineForm::constant(0.0);
        for joint in &chain.joints {
            if joint.parameter_dependent {
                let mut link_af = AffineForm::constant(joint.link_length);
                for i in 0..5.min(b.dim()) {
                    let coeff = joint.length_coefficients[i];
                    if coeff.abs() > 1e-15 {
                        let param_af = AffineForm::from_interval(&b.components[i]);
                        link_af = link_af.add(&param_af.scale(coeff));
                    }
                }
                total_reach_af = total_reach_af.add(&link_af);
            } else if joint.link_length > 1e-10 {
                total_reach_af = total_reach_af.add(&AffineForm::constant(joint.link_length));
            }
        }

        let reach_iv = total_reach_af.to_interval().expand(self.margin);

        let mut shoulder_y_af = AffineForm::constant(chain.base_transform[7]);
        for i in 0..5.min(b.dim()) {
            let coeff = chain.base_position_coefficients[1][i];
            if coeff.abs() > 1e-15 {
                let param_af = AffineForm::from_interval(&b.components[i]);
                shoulder_y_af = shoulder_y_af.add(&param_af.scale(coeff));
            }
        }
        let shoulder_y_iv = shoulder_y_af.to_interval();

        let mut shoulder_x_af = AffineForm::constant(chain.base_transform[3]);
        for i in 0..5.min(b.dim()) {
            let coeff = chain.base_position_coefficients[0][i];
            if coeff.abs() > 1e-15 {
                let param_af = AffineForm::from_interval(&b.components[i]);
                shoulder_x_af = shoulder_x_af.add(&param_af.scale(coeff));
            }
        }
        let shoulder_x_iv = shoulder_x_af.to_interval();

        BoundingBox::new(
            [
                shoulder_x_iv.lo - reach_iv.hi,
                shoulder_y_iv.lo - reach_iv.hi,
                -reach_iv.hi,
            ],
            [
                shoulder_x_iv.hi + reach_iv.hi,
                shoulder_y_iv.hi + reach_iv.hi,
                reach_iv.hi,
            ],
        )
    }

    // ── Wrapping factor (Theorem B1) ────────────────────────────

    /// Compute the wrapping factor for a kinematic chain over a joint range.
    ///
    /// Per Theorem B1:  w ≤ ∏ᵢ (1 + cᵢ · Δᵢ²)
    pub fn compute_wrapping_factor(
        &self,
        chain: &KinematicChain,
        joint_range: &IntervalVector,
    ) -> f64 {
        let n = chain.joints.len().min(joint_range.dim());
        let mut w = 1.0_f64;

        for i in 0..n {
            let delta = joint_range.components[i].width();
            let link_len = chain.joints[i].link_length.max(0.01);
            let c_i = link_len
                * self
                    .sensitivity_coefficients
                    .get(i.min(4))
                    .copied()
                    .unwrap_or(1.0);
            w *= 1.0 + c_i * delta * delta;
        }
        w
    }

    /// Wrapping factor for a parameter region (simplified interval model).
    pub fn wrapping_factor(&self, region: &ParameterRegion) -> f64 {
        let iv_w = self.reach_bound(region).width();
        let af_w = self.affine_reach_bound(region).to_interval().width();
        if af_w < 1e-15 {
            1.0
        } else {
            iv_w / af_w
        }
    }

    // ── Completeness gap ────────────────────────────────────────

    /// Estimate the completeness gap: gap ≈ (w − 1) × diameter.
    pub fn estimate_completeness_gap(&self, wrapping_factor: f64, diameter: f64) -> f64 {
        (wrapping_factor - 1.0).max(0.0) * diameter
    }

    // ── Workspace bounding box ──────────────────────────────────

    /// Simple workspace bounding box from the parameter region.
    pub fn workspace_bbox(&self, region: &ParameterRegion) -> BoundingBox {
        let reach = self.reach_bound(region);
        let sh = self.shoulder_height_bound(region);
        BoundingBox::new(
            [-reach.hi, sh.lo - reach.hi, -reach.hi],
            [reach.hi, sh.hi + reach.hi, reach.hi],
        )
    }

    // ── Tier 1 classification ───────────────────────────────────

    /// Classify a target bounding box against the reach envelope.
    pub fn classify_target(
        &self,
        chain: &KinematicChain,
        param_region: &ParameterRegion,
        target_bbox: &BoundingBox,
    ) -> RegionClassification {
        let reach_bbox = self.compute_reach_bounds(chain, param_region);

        if reach_bbox.contains_box(target_bbox) {
            let reach_iv = self.reach_bound(param_region);
            let sh = self.shoulder_height_bound(param_region);

            let tc = target_bbox.center();
            let sh_mid = sh.midpoint();
            let dx = tc[0];
            let dy = tc[1] - sh_mid;
            let dz = tc[2];
            let target_dist = (dx * dx + dy * dy + dz * dz).sqrt();
            let target_radius = target_bbox.diagonal() * 0.5;

            if target_dist + target_radius <= reach_iv.lo {
                return RegionClassification::Green;
            }
            RegionClassification::Yellow
        } else if !reach_bbox.intersects(target_bbox) {
            RegionClassification::Red
        } else {
            RegionClassification::Yellow
        }
    }

    /// Quick classification using only the spherical reach model.
    pub fn classify_simple(
        &self,
        param_region: &ParameterRegion,
        target_pos: [f64; 3],
        activation_radius: f64,
    ) -> RegionClassification {
        let reach_iv = self.reach_bound(param_region);
        let sh = self.shoulder_height_bound(param_region);

        let dx = target_pos[0];
        let dy_iv = Interval::point(target_pos[1]) - sh;
        let dz = target_pos[2];

        let dist_sq = Interval::point(dx * dx + dz * dz) + dy_iv.sqr();
        let dist = dist_sq.sqrt().unwrap_or(Interval::entire());

        let effective = Interval::new(
            (dist.lo - activation_radius).max(0.0),
            dist.hi + activation_radius,
        );

        if effective.hi <= reach_iv.lo {
            RegionClassification::Green
        } else if effective.lo > reach_iv.hi {
            RegionClassification::Red
        } else {
            RegionClassification::Yellow
        }
    }

    /// Maximum useful subdivision depth before completeness gap is small enough.
    pub fn max_useful_depth(
        &self,
        chain: &KinematicChain,
        root_joint_range: &IntervalVector,
        gap_threshold: f64,
    ) -> usize {
        let mut joint_range = root_joint_range.clone();
        let root_diameter = joint_range.max_width();
        let mut depth = 0;

        loop {
            let w = self.compute_wrapping_factor(chain, &joint_range);
            let diameter = root_diameter / (2.0_f64.powi(depth as i32));
            let gap = self.estimate_completeness_gap(w, diameter);
            if gap < gap_threshold || depth >= 20 {
                break;
            }
            let (left, _) = joint_range.bisect_widest();
            joint_range = left;
            depth += 1;
        }
        depth
    }
}

impl Default for BoundComputer {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use xr_types::ArmSide;

    fn sample_region() -> ParameterRegion {
        ParameterRegion::new(IntervalVector::from_ranges(&[
            (1.5, 1.9),
            (0.25, 0.40),
            (0.35, 0.50),
            (0.22, 0.33),
            (0.16, 0.22),
        ]))
    }

    fn sample_chain() -> KinematicChain {
        KinematicChain::default_arm(ArmSide::Right)
    }

    #[test]
    fn test_reach_bound() {
        let r = sample_region();
        let reach = BoundComputer::new().reach_bound(&r);
        assert!(reach.lo > 0.5);
        assert!(reach.hi < 1.1);
    }

    #[test]
    fn test_shoulder_height_bound() {
        let r = sample_region();
        let sh = BoundComputer::new().shoulder_height_bound(&r);
        assert!(sh.lo >= 1.19);
        assert!(sh.hi <= 1.60);
    }

    #[test]
    fn test_affine_reach_tighter_than_interval() {
        let r = sample_region();
        let bc = BoundComputer::new();
        let iv_width = bc.reach_bound(&r).width();
        let af_width = bc.affine_reach_bound(&r).to_interval().width();
        assert!(af_width <= iv_width + 1e-10);
    }

    #[test]
    fn test_workspace_bbox() {
        let r = sample_region();
        let bbox = BoundComputer::new().workspace_bbox(&r);
        assert!(bbox.volume() > 0.0);
        assert!(bbox.min[1] < 0.5);
        assert!(bbox.max[1] > 1.5);
    }

    #[test]
    fn test_wrapping_factor_identity_for_point() {
        let r = ParameterRegion::new(IntervalVector::from_ranges(&[
            (1.7, 1.7),
            (0.30, 0.30),
            (0.40, 0.40),
            (0.25, 0.25),
            (0.19, 0.19),
        ]));
        let w = BoundComputer::new().wrapping_factor(&r);
        assert!((w - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_wrapping_factor_with_chain() {
        let chain = sample_chain();
        let jrange = IntervalVector::from_ranges(&[
            (-1.0, 3.14),
            (-0.17, 3.14),
            (-1.57, 1.57),
            (0.0, 2.53),
            (-1.40, 1.40),
            (-1.22, 1.22),
            (-0.35, 0.52),
        ]);
        let w = BoundComputer::new().compute_wrapping_factor(&chain, &jrange);
        assert!(w >= 1.0, "wrapping factor must be >= 1, got {w}");
    }

    #[test]
    fn test_compute_reach_bounds_with_chain() {
        let chain = sample_chain();
        let region = sample_region();
        let bbox = BoundComputer::new().compute_reach_bounds(&chain, &region);
        assert!(bbox.volume() > 0.0);
    }

    #[test]
    fn test_completeness_gap() {
        let bc = BoundComputer::new();
        let gap = bc.estimate_completeness_gap(1.5, 0.1);
        assert!((gap - 0.05).abs() < 1e-10);
        let gap_tight = bc.estimate_completeness_gap(1.0, 0.1);
        assert!((gap_tight - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_classify_simple_green() {
        let region = sample_region();
        let bc = BoundComputer::new();
        let class = bc.classify_simple(&region, [0.0, 1.4, -0.1], 0.1);
        assert_eq!(class, RegionClassification::Green);
    }

    #[test]
    fn test_classify_simple_red() {
        let region = sample_region();
        let bc = BoundComputer::new();
        let class = bc.classify_simple(&region, [0.0, 1.4, -10.0], 0.05);
        assert_eq!(class, RegionClassification::Red);
    }

    #[test]
    fn test_classify_simple_yellow() {
        let region = sample_region();
        let bc = BoundComputer::new();
        let reach_hi = bc.reach_bound(&region).hi;
        let class = bc.classify_simple(&region, [0.0, 1.4, -reach_hi], 0.05);
        assert_eq!(class, RegionClassification::Yellow);
    }

    #[test]
    fn test_classify_target_with_chain() {
        let chain = sample_chain();
        let region = sample_region();
        let bc = BoundComputer::new();
        let far = BoundingBox::from_center_extents([50.0, 50.0, 50.0], [0.1, 0.1, 0.1]);
        let class = bc.classify_target(&chain, &region, &far);
        assert_eq!(class, RegionClassification::Red);
    }

    #[test]
    fn test_max_useful_depth() {
        let chain = sample_chain();
        let jrange = IntervalVector::from_ranges(&[
            (-1.0, 3.14),
            (-0.17, 3.14),
            (-1.57, 1.57),
            (0.0, 2.53),
            (-1.40, 1.40),
            (-1.22, 1.22),
            (-0.35, 0.52),
        ]);
        let depth = BoundComputer::new().max_useful_depth(&chain, &jrange, 0.01);
        assert!(depth > 0);
        assert!(depth <= 20);
    }

    #[test]
    fn test_default_bound_computer() {
        let bc = BoundComputer::default();
        assert!((bc.margin - 0.01).abs() < 1e-10);
    }
}

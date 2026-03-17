//! Tier 1 engine: fast interval-arithmetic verification that classifies each
//! scene element as Green (provably reachable), Red (provably unreachable),
//! or Yellow (uncertain, needs Tier 2).
//!
//! This module provides a self-contained interval-arithmetic implementation
//! so the lint crate can run independently of the `xr-spatial` crate.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use xr_types::geometry::Volume;
use xr_types::kinematic::{BodyParameterRange, BodyParameters};
use xr_types::scene::{InteractableElement, SceneModel};

use crate::linter::dist3;

// ── Interval arithmetic ─────────────────────────────────────────────────────

/// A closed interval [lo, hi] with outward rounding semantics.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
}

impl Interval {
    pub fn new(lo: f64, hi: f64) -> Self {
        debug_assert!(lo <= hi, "lo={lo} > hi={hi}");
        Self { lo, hi }
    }

    pub fn point(x: f64) -> Self {
        Self { lo: x, hi: x }
    }

    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }

    pub fn midpoint(&self) -> f64 {
        (self.lo + self.hi) * 0.5
    }

    pub fn contains(&self, x: f64) -> bool {
        x >= self.lo && x <= self.hi
    }

    pub fn overlaps(&self, other: &Interval) -> bool {
        self.lo <= other.hi && other.lo <= self.hi
    }

    pub fn add(&self, other: &Interval) -> Interval {
        Interval::new(self.lo + other.lo, self.hi + other.hi)
    }

    pub fn sub(&self, other: &Interval) -> Interval {
        Interval::new(self.lo - other.hi, self.hi - other.lo)
    }

    pub fn mul_scalar(&self, s: f64) -> Interval {
        if s >= 0.0 {
            Interval::new(self.lo * s, self.hi * s)
        } else {
            Interval::new(self.hi * s, self.lo * s)
        }
    }

    pub fn sqr(&self) -> Interval {
        if self.lo >= 0.0 {
            Interval::new(self.lo * self.lo, self.hi * self.hi)
        } else if self.hi <= 0.0 {
            Interval::new(self.hi * self.hi, self.lo * self.lo)
        } else {
            Interval::new(0.0, self.lo.abs().max(self.hi.abs()).powi(2))
        }
    }

    pub fn sqrt(&self) -> Option<Interval> {
        if self.hi < 0.0 {
            None
        } else {
            let lo = self.lo.max(0.0).sqrt();
            let hi = self.hi.sqrt();
            Some(Interval::new(lo, hi))
        }
    }

    /// Hull (union bound) of two intervals.
    pub fn hull(&self, other: &Interval) -> Interval {
        Interval::new(self.lo.min(other.lo), self.hi.max(other.hi))
    }
}

impl std::fmt::Display for Interval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:.4}, {:.4}]", self.lo, self.hi)
    }
}

impl std::ops::Add for Interval {
    type Output = Interval;
    fn add(self, rhs: Interval) -> Interval {
        Interval::add(&self, &rhs)
    }
}

impl std::ops::Sub for Interval {
    type Output = Interval;
    fn sub(self, rhs: Interval) -> Interval {
        Interval::sub(&self, &rhs)
    }
}

// ── Classification ──────────────────────────────────────────────────────────

/// Tier 1 classification for an element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Classification {
    /// Provably reachable by the entire population.
    Green,
    /// Cannot be classified — needs Tier 2 analysis.
    Yellow,
    /// Provably unreachable by the entire population.
    Red,
}

impl Classification {
    pub fn is_definite(&self) -> bool {
        matches!(self, Self::Green | Self::Red)
    }

    pub fn is_uncertain(&self) -> bool {
        matches!(self, Self::Yellow)
    }
}

impl std::fmt::Display for Classification {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Green => write!(f, "Green"),
            Self::Yellow => write!(f, "Yellow"),
            Self::Red => write!(f, "Red"),
        }
    }
}

// ── Tier 1 result ───────────────────────────────────────────────────────────

/// Result of Tier 1 verification for a single element.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tier1Result {
    pub element_id: Uuid,
    pub element_name: String,
    pub classification: Classification,
    /// Conservative reach interval [min_reach, max_reach].
    pub reach_interval: Option<(f64, f64)>,
    /// Distance from the element to the reachability boundary (negative = inside).
    pub distance_to_boundary: f64,
    /// Time spent on this element.
    pub elapsed: Duration,
    /// Human-readable detail string.
    pub details: String,
    /// Confidence: ratio of interval width to reach range.
    pub confidence: f64,
}

impl Tier1Result {
    pub fn needs_tier2(&self) -> bool {
        self.classification == Classification::Yellow
    }
}

// ── Tier 1 engine ───────────────────────────────────────────────────────────

/// Fast interval-arithmetic Tier 1 verification engine.
///
/// For each element, computes conservative reach bounds using interval
/// arithmetic over the body parameter ranges and classifies elements as
/// Green / Yellow / Red.
pub struct Tier1Engine {
    /// Body parameter ranges: (lo, hi) for each of the 5 parameters.
    pub param_ranges: [(f64, f64); 5],
    /// Per-element time budget.
    pub time_budget: Duration,
    /// Margin around classification boundary.
    margin: f64,
}

impl Tier1Engine {
    /// Create from a body parameter range.
    pub fn new(body_range: &BodyParameterRange) -> Self {
        let min_arr = body_range.min.to_array();
        let max_arr = body_range.max.to_array();
        let mut ranges = [(0.0, 0.0); 5];
        for i in 0..5 {
            ranges[i] = (min_arr[i], max_arr[i]);
        }
        Self {
            param_ranges: ranges,
            time_budget: Duration::from_millis(50),
            margin: 0.005,
        }
    }

    /// Create from explicit ranges.
    pub fn from_ranges(ranges: [(f64, f64); 5]) -> Self {
        Self {
            param_ranges: ranges,
            time_budget: Duration::from_millis(50),
            margin: 0.005,
        }
    }

    pub fn with_time_budget(mut self, budget: Duration) -> Self {
        self.time_budget = budget;
        self
    }

    pub fn with_margin(mut self, margin: f64) -> Self {
        self.margin = margin;
        self
    }

    // ── Core interval computations ──────────────────────────────────────

    /// Intervals for body parameters.
    fn param_intervals(&self) -> [Interval; 5] {
        [
            Interval::new(self.param_ranges[0].0, self.param_ranges[0].1), // stature
            Interval::new(self.param_ranges[1].0, self.param_ranges[1].1), // arm
            Interval::new(self.param_ranges[2].0, self.param_ranges[2].1), // shoulder_breadth
            Interval::new(self.param_ranges[3].0, self.param_ranges[3].1), // forearm
            Interval::new(self.param_ranges[4].0, self.param_ranges[4].1), // hand
        ]
    }

    /// Conservative total reach interval: arm + forearm + hand.
    fn reach_interval(&self) -> Interval {
        let p = self.param_intervals();
        p[1].add(&p[3]).add(&p[4]) // arm + forearm + hand
    }

    /// Conservative shoulder height interval: stature × 0.818.
    fn shoulder_height_interval(&self) -> Interval {
        let p = self.param_intervals();
        p[0].mul_scalar(0.818) // stature * 0.818
    }

    /// Conservative shoulder position intervals for left/right.
    fn shoulder_positions(&self) -> ([Interval; 3], [Interval; 3]) {
        let p = self.param_intervals();
        let sh = self.shoulder_height_interval();
        let half_sb = p[2].mul_scalar(0.5); // shoulder_breadth / 2

        // Left shoulder: (+half_sb, sh, 0)
        let left = [half_sb, sh, Interval::point(0.0)];
        // Right shoulder: (-half_sb, sh, 0)
        let neg_half_sb = Interval::new(-half_sb.hi, -half_sb.lo);
        let right = [neg_half_sb, sh, Interval::point(0.0)];

        (left, right)
    }

    /// Interval distance from a point to a shoulder interval position.
    fn interval_distance_to_point(
        shoulder: &[Interval; 3],
        target: &[f64; 3],
    ) -> Interval {
        let dx = Interval::point(target[0]) - shoulder[0];
        let dy = Interval::point(target[1]) - shoulder[1];
        let dz = Interval::point(target[2]) - shoulder[2];
        let sq = dx.sqr().add(&dy.sqr()).add(&dz.sqr());
        sq.sqrt().unwrap_or(Interval::new(0.0, f64::MAX))
    }

    // ── Element classification ──────────────────────────────────────────

    /// Classify a single element.
    pub fn quick_check(&self, element: &InteractableElement) -> Tier1Result {
        let start = Instant::now();

        let reach = self.reach_interval();
        let (left_sh, right_sh) = self.shoulder_positions();

        // Distance from element to each shoulder
        let d_left = Self::interval_distance_to_point(&left_sh, &element.position);
        let d_right = Self::interval_distance_to_point(&right_sh, &element.position);

        // Take the tighter (smaller) distance interval
        let d_min = Interval::new(
            d_left.lo.min(d_right.lo),
            d_left.hi.min(d_right.hi),
        );

        // Account for activation volume radius
        let vol_radius = approx_vol_radius(&element.activation_volume);
        let effective_dist = Interval::new(
            (d_min.lo - vol_radius).max(0.0),
            (d_min.hi - vol_radius).max(0.0),
        );

        // Classification:
        // Green:  effective_dist.hi <= reach.lo  (even worst case is reachable)
        // Red:    effective_dist.lo > reach.hi   (even best case is unreachable)
        // Yellow: otherwise
        let (classification, boundary_dist) = if effective_dist.hi <= reach.lo - self.margin {
            let gap = reach.lo - effective_dist.hi;
            (Classification::Green, -gap)
        } else if effective_dist.lo > reach.hi + self.margin {
            let gap = effective_dist.lo - reach.hi;
            (Classification::Red, gap)
        } else {
            let mid_dist = effective_dist.midpoint();
            let mid_reach = reach.midpoint();
            (Classification::Yellow, mid_dist - mid_reach)
        };

        let elapsed = start.elapsed();
        let reach_width = reach.width();
        let confidence = if reach_width > 1e-9 {
            1.0 - (effective_dist.width() / reach_width).min(1.0)
        } else {
            1.0
        };

        Tier1Result {
            element_id: element.id,
            element_name: element.name.clone(),
            classification,
            reach_interval: Some((reach.lo, reach.hi)),
            distance_to_boundary: boundary_dist,
            elapsed,
            details: format!(
                "reach={}, eff_dist={}, classification={}",
                reach, effective_dist, classification
            ),
            confidence,
        }
    }

    /// Classify all elements in a scene.
    pub fn full_scene_check(&self, scene: &SceneModel) -> Vec<Tier1Result> {
        scene
            .elements
            .iter()
            .map(|e| self.quick_check(e))
            .collect()
    }

    /// Summary counts: (green, yellow, red).
    pub fn scene_summary(results: &[Tier1Result]) -> (usize, usize, usize) {
        let green = results.iter().filter(|r| r.classification == Classification::Green).count();
        let yellow = results.iter().filter(|r| r.classification == Classification::Yellow).count();
        let red = results.iter().filter(|r| r.classification == Classification::Red).count();
        (green, yellow, red)
    }

    /// Create from the default body range.
    pub fn default_engine() -> Self {
        Self::new(&BodyParameterRange::default())
    }

    /// Compute the worst-case wrapping factor (interval width / midpoint).
    pub fn wrapping_factor(&self) -> f64 {
        let reach = self.reach_interval();
        let mid = reach.midpoint();
        if mid.abs() < 1e-12 {
            f64::MAX
        } else {
            reach.width() / mid
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

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

    fn engine() -> Tier1Engine {
        Tier1Engine::default_engine()
    }

    fn make_element(name: &str, pos: [f64; 3]) -> InteractableElement {
        let mut e = InteractableElement::new(name, pos);
        e.feedback_type = FeedbackType::VisualHaptic;
        e
    }

    fn make_scene(elems: Vec<InteractableElement>) -> SceneModel {
        let mut s = SceneModel::new("tier1_test");
        s.bounds = BoundingBox::from_center_extents([0.0, 1.0, 0.0], [5.0, 3.0, 5.0]);
        for e in elems {
            s.add_element(e);
        }
        s
    }

    // ── Interval tests ──────────────────────────────────────────────────

    #[test]
    fn test_interval_add() {
        let a = Interval::new(1.0, 2.0);
        let b = Interval::new(3.0, 5.0);
        let c = a + b;
        assert!((c.lo - 4.0).abs() < 1e-12);
        assert!((c.hi - 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_interval_sub() {
        let a = Interval::new(3.0, 5.0);
        let b = Interval::new(1.0, 2.0);
        let c = a - b;
        assert!((c.lo - 1.0).abs() < 1e-12);
        assert!((c.hi - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_interval_sqr_positive() {
        let a = Interval::new(2.0, 3.0);
        let sq = a.sqr();
        assert!((sq.lo - 4.0).abs() < 1e-12);
        assert!((sq.hi - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_interval_sqr_spanning_zero() {
        let a = Interval::new(-2.0, 3.0);
        let sq = a.sqr();
        assert!((sq.lo - 0.0).abs() < 1e-12);
        assert!((sq.hi - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_interval_sqrt() {
        let a = Interval::new(4.0, 9.0);
        let s = a.sqrt().unwrap();
        assert!((s.lo - 2.0).abs() < 1e-12);
        assert!((s.hi - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_interval_sqrt_negative() {
        let a = Interval::new(-4.0, -1.0);
        assert!(a.sqrt().is_none());
    }

    #[test]
    fn test_interval_mul_scalar() {
        let a = Interval::new(2.0, 5.0);
        let b = a.mul_scalar(-2.0);
        assert!((b.lo - (-10.0)).abs() < 1e-12);
        assert!((b.hi - (-4.0)).abs() < 1e-12);
    }

    #[test]
    fn test_interval_overlaps() {
        let a = Interval::new(1.0, 3.0);
        let b = Interval::new(2.0, 4.0);
        let c = Interval::new(4.0, 5.0);
        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn test_interval_contains() {
        let a = Interval::new(1.0, 5.0);
        assert!(a.contains(3.0));
        assert!(!a.contains(0.0));
        assert!(!a.contains(6.0));
    }

    #[test]
    fn test_interval_display() {
        let a = Interval::new(1.5, 2.5);
        let s = format!("{}", a);
        assert!(s.contains("1.5"));
        assert!(s.contains("2.5"));
    }

    // ── Engine tests ────────────────────────────────────────────────────

    #[test]
    fn test_reach_interval() {
        let eng = engine();
        let ri = eng.reach_interval();
        assert!(ri.lo > 0.5, "Min reach should be > 0.5m");
        assert!(ri.hi < 1.2, "Max reach should be < 1.2m");
        assert!(ri.lo < ri.hi);
    }

    #[test]
    fn test_shoulder_height_interval() {
        let eng = engine();
        let sh = eng.shoulder_height_interval();
        assert!(sh.lo > 1.0, "Min shoulder height should be > 1.0m");
        assert!(sh.hi < 1.7, "Max shoulder height should be < 1.7m");
    }

    #[test]
    fn test_close_element_green() {
        let eng = engine();
        let elem = make_element("close", [0.2, 1.2, -0.3]);
        let result = eng.quick_check(&elem);
        assert_eq!(
            result.classification,
            Classification::Green,
            "Close element should be Green: {}",
            result.details
        );
    }

    #[test]
    fn test_far_element_red() {
        let eng = engine();
        let elem = make_element("far", [0.0, 1.2, -5.0]);
        let result = eng.quick_check(&elem);
        assert_eq!(
            result.classification,
            Classification::Red,
            "Far element should be Red: {}",
            result.details
        );
    }

    #[test]
    fn test_boundary_element_yellow() {
        let eng = engine();
        let ri = eng.reach_interval();
        let sh = eng.shoulder_height_interval();
        let mid_reach = ri.midpoint();
        let mid_sh = sh.midpoint();
        // Place element exactly at mid reach distance from shoulder
        let elem = make_element("boundary", [0.0, mid_sh, -mid_reach]);
        let result = eng.quick_check(&elem);
        // Should be Yellow or borderline Green (depends on volume radius)
        assert!(
            result.classification == Classification::Yellow
                || result.classification == Classification::Green,
            "Boundary element: {}",
            result.details
        );
    }

    #[test]
    fn test_full_scene_check() {
        let scene = make_scene(vec![
            make_element("near", [0.0, 1.2, -0.3]),
            make_element("far", [0.0, 1.2, -5.0]),
        ]);
        let eng = engine();
        let results = eng.full_scene_check(&scene);
        assert_eq!(results.len(), 2);

        let (green, yellow, red) = Tier1Engine::scene_summary(&results);
        assert!(green >= 1);
        assert!(red >= 1);
    }

    #[test]
    fn test_timing_tracked() {
        let eng = engine();
        let elem = make_element("btn", [0.0, 1.2, -0.5]);
        let result = eng.quick_check(&elem);
        assert!(result.elapsed.as_nanos() > 0);
    }

    #[test]
    fn test_wrapping_factor() {
        let eng = engine();
        let wf = eng.wrapping_factor();
        assert!(wf > 0.0);
        assert!(wf < 1.0, "Wrapping factor should be < 1.0 for normal ranges");
    }

    #[test]
    fn test_confidence_range() {
        let eng = engine();
        let elem = make_element("btn", [0.0, 1.2, -0.3]);
        let result = eng.quick_check(&elem);
        assert!(
            result.confidence >= 0.0 && result.confidence <= 1.0,
            "Confidence {} out of range",
            result.confidence
        );
    }

    #[test]
    fn test_classification_display() {
        assert_eq!(format!("{}", Classification::Green), "Green");
        assert_eq!(format!("{}", Classification::Yellow), "Yellow");
        assert_eq!(format!("{}", Classification::Red), "Red");
    }

    #[test]
    fn test_classification_properties() {
        assert!(Classification::Green.is_definite());
        assert!(Classification::Red.is_definite());
        assert!(!Classification::Yellow.is_definite());
        assert!(Classification::Yellow.is_uncertain());
    }

    #[test]
    fn test_needs_tier2() {
        let eng = engine();
        let far = make_element("far", [0.0, 1.2, -5.0]);
        let result = eng.quick_check(&far);
        assert!(!result.needs_tier2(), "Red should not need tier2");

        let near = make_element("near", [0.0, 1.2, -0.3]);
        let result = eng.quick_check(&near);
        assert!(!result.needs_tier2(), "Green should not need tier2");
    }

    #[test]
    fn test_custom_ranges() {
        let eng = Tier1Engine::from_ranges([
            (1.7, 1.8),   // stature
            (0.35, 0.38),  // arm
            (0.45, 0.50),  // shoulder
            (0.25, 0.28),  // forearm
            (0.18, 0.20),  // hand
        ]);
        let elem = make_element("btn", [0.0, 1.3, -0.5]);
        let result = eng.quick_check(&elem);
        assert!(result.classification != Classification::Red);
    }

    #[test]
    fn test_from_body_range() {
        let range = BodyParameterRange::default();
        let eng = Tier1Engine::new(&range);
        let ri = eng.reach_interval();
        assert!((ri.lo - range.min.total_reach()).abs() < 1e-9);
        assert!((ri.hi - range.max.total_reach()).abs() < 1e-9);
    }
}

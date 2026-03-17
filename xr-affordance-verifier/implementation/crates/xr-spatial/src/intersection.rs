//! Geometric intersection tests for conservative spatial verification.
//!
//! Provides conservative (over-approximate) intersection primitives
//! used by the Tier 1 verifier to determine overlap between reach
//! envelopes and activation volumes.

use crate::affine::{AffineForm, AffineVector3};
use crate::interval::{Interval, IntervalVector};
use xr_types::{BoundingBox, Capsule, Sphere, Volume};

// ── IntersectionResult ──────────────────────────────────────────────────

/// Result of a geometric intersection test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IntersectionVerdict {
    /// The two volumes are provably disjoint.
    Disjoint,
    /// The volumes overlap (or cannot be proven disjoint).
    Overlapping,
    /// One volume is fully contained within the other.
    Contained,
}

/// Detailed intersection result with quantitative information.
#[derive(Debug, Clone, Copy)]
pub struct IntersectionResult {
    /// Verdict: disjoint, overlapping, or contained.
    pub verdict: IntersectionVerdict,
    /// Conservative lower bound on the signed separation distance.
    /// Negative means overlap by at least this magnitude.
    pub separation_distance: f64,
    /// Conservative estimate of the overlap volume (0.0 if disjoint).
    pub overlap_volume_estimate: f64,
    /// Approximate contact normal direction (unit vector, zero if disjoint).
    pub contact_normal: [f64; 3],
}

impl IntersectionResult {
    pub fn disjoint(distance: f64) -> Self {
        Self {
            verdict: IntersectionVerdict::Disjoint,
            separation_distance: distance,
            overlap_volume_estimate: 0.0,
            contact_normal: [0.0; 3],
        }
    }

    pub fn overlapping(penetration: f64, overlap_vol: f64, normal: [f64; 3]) -> Self {
        Self {
            verdict: IntersectionVerdict::Overlapping,
            separation_distance: -penetration,
            overlap_volume_estimate: overlap_vol,
            contact_normal: normal,
        }
    }

    pub fn contained(overlap_vol: f64) -> Self {
        Self {
            verdict: IntersectionVerdict::Contained,
            separation_distance: f64::NEG_INFINITY,
            overlap_volume_estimate: overlap_vol,
            contact_normal: [0.0; 3],
        }
    }

    pub fn is_disjoint(&self) -> bool {
        self.verdict == IntersectionVerdict::Disjoint
    }

    pub fn is_overlapping(&self) -> bool {
        matches!(
            self.verdict,
            IntersectionVerdict::Overlapping | IntersectionVerdict::Contained
        )
    }
}

// ── IntersectionTest (main API) ─────────────────────────────────────────

/// Conservative geometric intersection tester.
///
/// All tests are designed to be *sound*: they may report overlap when
/// none exists (conservative), but never miss a true overlap.
pub struct IntersectionTest;

impl IntersectionTest {
    // ── Box-Box ─────────────────────────────────────────────────

    /// Axis-aligned bounding box vs bounding box intersection.
    pub fn box_box(a: &BoundingBox, b: &BoundingBox) -> IntersectionResult {
        let mut max_sep = f64::NEG_INFINITY;
        let mut sep_axis = 0usize;

        for i in 0..3 {
            let sep = (b.min[i] - a.max[i]).max(a.min[i] - b.max[i]);
            if sep > max_sep {
                max_sep = sep;
                sep_axis = i;
            }
        }

        if max_sep > 0.0 {
            let mut normal = [0.0; 3];
            normal[sep_axis] = if a.center()[sep_axis] < b.center()[sep_axis] {
                -1.0
            } else {
                1.0
            };
            return IntersectionResult::disjoint(max_sep);
        }

        // Overlap: compute overlap volume
        if let Some(isect) = a.intersection(b) {
            let vol = isect.volume();
            if a.contains_box(b) || b.contains_box(a) {
                return IntersectionResult::contained(vol);
            }
            let mut normal = [0.0; 3];
            let ac = a.center();
            let bc = b.center();
            let dx = bc[0] - ac[0];
            let dy = bc[1] - ac[1];
            let dz = bc[2] - ac[2];
            let len = (dx * dx + dy * dy + dz * dz).sqrt();
            if len > 1e-12 {
                normal = [dx / len, dy / len, dz / len];
            }
            IntersectionResult::overlapping(-max_sep, vol, normal)
        } else {
            IntersectionResult::disjoint(0.0)
        }
    }

    // ── Box-Sphere ──────────────────────────────────────────────

    /// Bounding box vs sphere intersection.
    pub fn box_sphere(bbox: &BoundingBox, sphere: &Sphere) -> IntersectionResult {
        let mut dist_sq = 0.0f64;
        let mut closest = sphere.center;

        for i in 0..3 {
            if sphere.center[i] < bbox.min[i] {
                let d = bbox.min[i] - sphere.center[i];
                dist_sq += d * d;
                closest[i] = bbox.min[i];
            } else if sphere.center[i] > bbox.max[i] {
                let d = sphere.center[i] - bbox.max[i];
                dist_sq += d * d;
                closest[i] = bbox.max[i];
            }
        }

        let dist = dist_sq.sqrt();
        let sep = dist - sphere.radius;

        if sep > 0.0 {
            return IntersectionResult::disjoint(sep);
        }

        // Compute contact normal from sphere center to closest box point
        let dx = closest[0] - sphere.center[0];
        let dy = closest[1] - sphere.center[1];
        let dz = closest[2] - sphere.center[2];
        let nlen = (dx * dx + dy * dy + dz * dz).sqrt();
        let normal = if nlen > 1e-12 {
            [dx / nlen, dy / nlen, dz / nlen]
        } else {
            [0.0, 1.0, 0.0]
        };

        // Estimate overlap volume conservatively using sphere cap
        let penetration = -sep;
        let overlap_vol = estimate_sphere_cap_volume(sphere.radius, penetration);

        if bbox.contains_point(&sphere.center)
            && sphere.radius
                <= bbox
                    .half_extents()
                    .iter()
                    .cloned()
                    .fold(f64::INFINITY, f64::min)
        {
            IntersectionResult::contained(sphere.volume())
        } else {
            IntersectionResult::overlapping(penetration, overlap_vol, normal)
        }
    }

    // ── Sphere-Sphere ───────────────────────────────────────────

    /// Sphere vs sphere intersection.
    pub fn sphere_sphere(a: &Sphere, b: &Sphere) -> IntersectionResult {
        let dx = b.center[0] - a.center[0];
        let dy = b.center[1] - a.center[1];
        let dz = b.center[2] - a.center[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        let sep = dist - (a.radius + b.radius);

        if sep > 0.0 {
            return IntersectionResult::disjoint(sep);
        }

        let nlen = dist;
        let normal = if nlen > 1e-12 {
            [dx / nlen, dy / nlen, dz / nlen]
        } else {
            [0.0, 1.0, 0.0]
        };

        // Check containment: one sphere fully inside the other
        if dist + a.radius.min(b.radius) <= a.radius.max(b.radius) {
            let contained_vol = a.volume().min(b.volume());
            return IntersectionResult::contained(contained_vol);
        }

        let penetration = -sep;
        let overlap_vol = estimate_sphere_sphere_overlap(a.radius, b.radius, dist);
        IntersectionResult::overlapping(penetration, overlap_vol, normal)
    }

    // ── Capsule-Box ─────────────────────────────────────────────

    /// Capsule vs bounding box intersection.
    pub fn capsule_box(capsule: &Capsule, bbox: &BoundingBox) -> IntersectionResult {
        // Conservative: expand the box by capsule radius, then test
        // the capsule axis segment against the expanded box.
        let expanded = bbox.expand(capsule.radius);
        let axis_bbox = capsule.bounding_box();

        if !expanded.intersects(&axis_bbox) {
            // Compute separation distance
            let seg_closest = closest_point_on_segment_to_box(
                &capsule.start,
                &capsule.end,
                bbox,
            );
            let box_dist = point_to_box_distance(&seg_closest, bbox);
            let sep = box_dist - capsule.radius;
            return IntersectionResult::disjoint(sep.max(0.0));
        }

        // Check if the axis segment intersects the expanded box more precisely
        let seg_closest =
            closest_point_on_segment_to_box(&capsule.start, &capsule.end, bbox);
        let box_dist = point_to_box_distance(&seg_closest, bbox);
        let sep = box_dist - capsule.radius;

        if sep > 0.0 {
            return IntersectionResult::disjoint(sep);
        }

        let penetration = -sep;
        // Rough overlap volume estimate
        let overlap_vol = if let Some(isect) = capsule.bounding_box().intersection(bbox) {
            isect.volume()
        } else {
            0.0
        };

        // Normal from closest point on segment toward box center
        let bc = bbox.center();
        let dx = bc[0] - seg_closest[0];
        let dy = bc[1] - seg_closest[1];
        let dz = bc[2] - seg_closest[2];
        let nlen = (dx * dx + dy * dy + dz * dz).sqrt();
        let normal = if nlen > 1e-12 {
            [dx / nlen, dy / nlen, dz / nlen]
        } else {
            [0.0, 1.0, 0.0]
        };

        IntersectionResult::overlapping(penetration, overlap_vol, normal)
    }

    // ── Interval-based tests ────────────────────────────────────

    /// Conservative reachability check using interval arithmetic.
    /// Tests whether a target point (with activation radius) is reachable
    /// from a shoulder position interval with a given reach interval.
    pub fn reachability_check(
        shoulder: &[Interval; 3],
        target: [f64; 3],
        reach: &Interval,
        activation_radius: f64,
    ) -> IntersectionVerdict {
        let dx = Interval::point(target[0]) - shoulder[0];
        let dy = Interval::point(target[1]) - shoulder[1];
        let dz = Interval::point(target[2]) - shoulder[2];

        let dist_sq = dx.sqr() + dy.sqr() + dz.sqr();
        let dist = dist_sq.sqrt().unwrap_or(Interval::entire());

        let effective = Interval::new(
            (dist.lo - activation_radius).max(0.0),
            dist.hi + activation_radius,
        );

        if effective.hi <= reach.lo {
            IntersectionVerdict::Contained
        } else if effective.lo > reach.hi {
            IntersectionVerdict::Disjoint
        } else {
            IntersectionVerdict::Overlapping
        }
    }

    /// Affine-arithmetic sphere vs bounding box test.
    pub fn affine_sphere_vs_bbox(
        center: &AffineVector3,
        radius: &AffineForm,
        bbox: &BoundingBox,
    ) -> IntersectionVerdict {
        let [cx, cy, cz] = center.to_intervals();
        let r_iv = radius.to_interval();

        let expanded = BoundingBox::new(
            [
                bbox.min[0] - r_iv.hi,
                bbox.min[1] - r_iv.hi,
                bbox.min[2] - r_iv.hi,
            ],
            [
                bbox.max[0] + r_iv.hi,
                bbox.max[1] + r_iv.hi,
                bbox.max[2] + r_iv.hi,
            ],
        );
        let center_box = BoundingBox::new([cx.lo, cy.lo, cz.lo], [cx.hi, cy.hi, cz.hi]);

        if !expanded.intersects(&center_box) {
            return IntersectionVerdict::Disjoint;
        }

        // Check containment: box fully inside the sphere
        let shrunk = BoundingBox::new(
            [
                bbox.min[0] + r_iv.lo,
                bbox.min[1] + r_iv.lo,
                bbox.min[2] + r_iv.lo,
            ],
            [
                bbox.max[0] - r_iv.lo,
                bbox.max[1] - r_iv.lo,
                bbox.max[2] - r_iv.lo,
            ],
        );
        if shrunk.min[0] <= shrunk.max[0]
            && shrunk.min[1] <= shrunk.max[1]
            && shrunk.min[2] <= shrunk.max[2]
            && shrunk.contains_box(&center_box)
        {
            return IntersectionVerdict::Contained;
        }

        IntersectionVerdict::Overlapping
    }

    /// Interval box (first 3 components) vs bounding box.
    pub fn interval_box_vs_bbox(iv_box: &IntervalVector, bbox: &BoundingBox) -> bool {
        if iv_box.dim() < 3 {
            return false;
        }
        let iv_bbox = BoundingBox::new(
            [
                iv_box.components[0].lo,
                iv_box.components[1].lo,
                iv_box.components[2].lo,
            ],
            [
                iv_box.components[0].hi,
                iv_box.components[1].hi,
                iv_box.components[2].hi,
            ],
        );
        iv_bbox.intersects(bbox)
    }

    // ── Volume-generic tests ────────────────────────────────────

    /// Test intersection of a Volume against a BoundingBox.
    pub fn volume_vs_bbox(vol: &Volume, bbox: &BoundingBox) -> IntersectionResult {
        match vol {
            Volume::Box(b) => Self::box_box(b, bbox),
            Volume::Sphere(s) => Self::box_sphere(bbox, s),
            Volume::Capsule(c) => Self::capsule_box(c, bbox),
            _ => {
                // Fall back to bounding-box test (conservative).
                Self::box_box(&vol.bounding_box(), bbox)
            }
        }
    }

    /// Test intersection between two Volumes.
    pub fn volume_vs_volume(a: &Volume, b: &Volume) -> IntersectionResult {
        match (a, b) {
            (Volume::Box(ba), Volume::Box(bb)) => Self::box_box(ba, bb),
            (Volume::Sphere(sa), Volume::Sphere(sb)) => Self::sphere_sphere(sa, sb),
            (Volume::Box(ba), Volume::Sphere(sb)) | (Volume::Sphere(sb), Volume::Box(ba)) => {
                Self::box_sphere(ba, sb)
            }
            (Volume::Capsule(ca), Volume::Box(bb)) | (Volume::Box(bb), Volume::Capsule(ca)) => {
                Self::capsule_box(ca, bb)
            }
            _ => {
                // Conservative fallback: bounding box of each.
                Self::box_box(&a.bounding_box(), &b.bounding_box())
            }
        }
    }

    // ── GJK-inspired separation distance ────────────────────────

    /// Compute conservative separation distance between two bounding boxes.
    /// Uses a GJK-inspired approach on the Minkowski difference support.
    pub fn gjk_separation_boxes(a: &BoundingBox, b: &BoundingBox) -> f64 {
        // The Minkowski difference of two AABBs is itself an AABB.
        // Separation distance = distance from origin to that AABB.
        let mink_min = [
            a.min[0] - b.max[0],
            a.min[1] - b.max[1],
            a.min[2] - b.max[2],
        ];
        let mink_max = [
            a.max[0] - b.min[0],
            a.max[1] - b.min[1],
            a.max[2] - b.min[2],
        ];

        // Signed distance from origin to the Minkowski difference AABB
        let origin = [0.0, 0.0, 0.0];
        let mink = BoundingBox::new(mink_min, mink_max);
        let sd = mink.signed_distance(&origin);
        sd
    }

    /// Compute conservative separation distance between a sphere and a box.
    pub fn separation_sphere_box(sphere: &Sphere, bbox: &BoundingBox) -> f64 {
        point_to_box_distance(&sphere.center, bbox) - sphere.radius
    }

    /// Compute conservative separation distance between two spheres.
    pub fn separation_sphere_sphere(a: &Sphere, b: &Sphere) -> f64 {
        let dx = b.center[0] - a.center[0];
        let dy = b.center[1] - a.center[1];
        let dz = b.center[2] - a.center[2];
        (dx * dx + dy * dy + dz * dz).sqrt() - a.radius - b.radius
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Euclidean distance from a point to the nearest point on a bounding box.
fn point_to_box_distance(p: &[f64; 3], bbox: &BoundingBox) -> f64 {
    let mut dist_sq = 0.0f64;
    for i in 0..3 {
        if p[i] < bbox.min[i] {
            let d = bbox.min[i] - p[i];
            dist_sq += d * d;
        } else if p[i] > bbox.max[i] {
            let d = p[i] - bbox.max[i];
            dist_sq += d * d;
        }
    }
    dist_sq.sqrt()
}

/// Closest point on a line segment [a, b] to the center of a bounding box.
fn closest_point_on_segment_to_box(a: &[f64; 3], b: &[f64; 3], bbox: &BoundingBox) -> [f64; 3] {
    let c = bbox.center();
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let ab_sq = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    if ab_sq < 1e-24 {
        return *a;
    }
    let t = (ac[0] * ab[0] + ac[1] * ab[1] + ac[2] * ab[2]) / ab_sq;
    let t = t.clamp(0.0, 1.0);
    [a[0] + t * ab[0], a[1] + t * ab[1], a[2] + t * ab[2]]
}

/// Estimate volume of a sphere cap with given penetration depth.
fn estimate_sphere_cap_volume(radius: f64, depth: f64) -> f64 {
    let h = depth.min(2.0 * radius).max(0.0);
    std::f64::consts::PI / 3.0 * h * h * (3.0 * radius - h)
}

/// Estimate volume of intersection of two spheres.
fn estimate_sphere_sphere_overlap(r1: f64, r2: f64, dist: f64) -> f64 {
    if dist >= r1 + r2 {
        return 0.0;
    }
    if dist + r1.min(r2) <= r1.max(r2) {
        // One contained in the other
        let small_r = r1.min(r2);
        return (4.0 / 3.0) * std::f64::consts::PI * small_r.powi(3);
    }
    // Lens formula for sphere-sphere intersection
    let d = dist;
    let part1 = (r1 + r2 - d).powi(2);
    let part2 = d * d + 2.0 * d * (r1 + r2) - 3.0 * (r1 - r2).powi(2);
    let vol = std::f64::consts::PI / (12.0 * d) * part1 * part2;
    vol.max(0.0)
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_box_disjoint() {
        let a = BoundingBox::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b = BoundingBox::new([2.0, 0.0, 0.0], [3.0, 1.0, 1.0]);
        let r = IntersectionTest::box_box(&a, &b);
        assert!(r.is_disjoint());
        assert!(r.separation_distance > 0.9);
    }

    #[test]
    fn test_box_box_overlap() {
        let a = BoundingBox::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        let b = BoundingBox::new([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]);
        let r = IntersectionTest::box_box(&a, &b);
        assert!(r.is_overlapping());
        assert!(r.overlap_volume_estimate > 0.0);
    }

    #[test]
    fn test_box_box_contained() {
        let a = BoundingBox::new([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
        let b = BoundingBox::new([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]);
        let r = IntersectionTest::box_box(&a, &b);
        assert_eq!(r.verdict, IntersectionVerdict::Contained);
    }

    #[test]
    fn test_box_sphere_disjoint() {
        let bbox = BoundingBox::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let sphere = Sphere::new([5.0, 5.0, 5.0], 0.5);
        let r = IntersectionTest::box_sphere(&bbox, &sphere);
        assert!(r.is_disjoint());
    }

    #[test]
    fn test_box_sphere_overlap() {
        let bbox = BoundingBox::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let sphere = Sphere::new([0.5, 0.5, 0.5], 0.3);
        let r = IntersectionTest::box_sphere(&bbox, &sphere);
        assert!(r.is_overlapping());
    }

    #[test]
    fn test_sphere_sphere_disjoint() {
        let a = Sphere::new([0.0, 0.0, 0.0], 1.0);
        let b = Sphere::new([5.0, 0.0, 0.0], 1.0);
        let r = IntersectionTest::sphere_sphere(&a, &b);
        assert!(r.is_disjoint());
        assert!((r.separation_distance - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_sphere_sphere_overlap() {
        let a = Sphere::new([0.0, 0.0, 0.0], 1.0);
        let b = Sphere::new([1.0, 0.0, 0.0], 1.0);
        let r = IntersectionTest::sphere_sphere(&a, &b);
        assert!(r.is_overlapping());
        assert!(r.overlap_volume_estimate > 0.0);
    }

    #[test]
    fn test_sphere_sphere_contained() {
        let a = Sphere::new([0.0, 0.0, 0.0], 5.0);
        let b = Sphere::new([0.0, 0.0, 0.0], 1.0);
        let r = IntersectionTest::sphere_sphere(&a, &b);
        assert_eq!(r.verdict, IntersectionVerdict::Contained);
    }

    #[test]
    fn test_capsule_box_disjoint() {
        let capsule = Capsule::new([10.0, 10.0, 10.0], [11.0, 10.0, 10.0], 0.1);
        let bbox = BoundingBox::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let r = IntersectionTest::capsule_box(&capsule, &bbox);
        assert!(r.is_disjoint());
    }

    #[test]
    fn test_capsule_box_overlap() {
        let capsule = Capsule::new([0.0, 0.5, 0.5], [2.0, 0.5, 0.5], 0.3);
        let bbox = BoundingBox::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let r = IntersectionTest::capsule_box(&capsule, &bbox);
        assert!(r.is_overlapping());
    }

    #[test]
    fn test_reachability_contained() {
        let shoulder = [
            Interval::point(0.0),
            Interval::new(1.3, 1.5),
            Interval::point(0.0),
        ];
        let r = IntersectionTest::reachability_check(
            &shoulder,
            [0.0, 1.4, -0.3],
            &Interval::new(0.6, 0.8),
            0.05,
        );
        assert_eq!(r, IntersectionVerdict::Contained);
    }

    #[test]
    fn test_reachability_disjoint() {
        let shoulder = [
            Interval::point(0.0),
            Interval::new(1.3, 1.5),
            Interval::point(0.0),
        ];
        let r = IntersectionTest::reachability_check(
            &shoulder,
            [0.0, 1.4, -5.0],
            &Interval::new(0.6, 0.8),
            0.05,
        );
        assert_eq!(r, IntersectionVerdict::Disjoint);
    }

    #[test]
    fn test_affine_sphere_vs_bbox_disjoint() {
        let center = AffineVector3::constant(10.0, 10.0, 10.0);
        let radius = AffineForm::constant(0.5);
        let bbox = BoundingBox::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let r = IntersectionTest::affine_sphere_vs_bbox(&center, &radius, &bbox);
        assert_eq!(r, IntersectionVerdict::Disjoint);
    }

    #[test]
    fn test_affine_sphere_vs_bbox_overlap() {
        let center = AffineVector3::constant(0.5, 0.5, 0.5);
        let radius = AffineForm::constant(0.3);
        let bbox = BoundingBox::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let r = IntersectionTest::affine_sphere_vs_bbox(&center, &radius, &bbox);
        assert!(r != IntersectionVerdict::Disjoint);
    }

    #[test]
    fn test_interval_box_vs_bbox() {
        let iv = IntervalVector::from_ranges(&[(0.0, 2.0), (0.0, 2.0), (0.0, 2.0)]);
        let bbox = BoundingBox::new([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]);
        assert!(IntersectionTest::interval_box_vs_bbox(&iv, &bbox));

        let iv2 = IntervalVector::from_ranges(&[(10.0, 20.0), (10.0, 20.0), (10.0, 20.0)]);
        assert!(!IntersectionTest::interval_box_vs_bbox(&iv2, &bbox));
    }

    #[test]
    fn test_gjk_separation_disjoint() {
        let a = BoundingBox::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b = BoundingBox::new([3.0, 0.0, 0.0], [4.0, 1.0, 1.0]);
        let sep = IntersectionTest::gjk_separation_boxes(&a, &b);
        assert!(sep > 1.9, "separation = {sep}");
    }

    #[test]
    fn test_gjk_separation_overlapping() {
        let a = BoundingBox::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        let b = BoundingBox::new([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]);
        let sep = IntersectionTest::gjk_separation_boxes(&a, &b);
        assert!(sep < 0.0, "separation = {sep}");
    }

    #[test]
    fn test_volume_vs_volume() {
        let a = Volume::Sphere(Sphere::new([0.0, 0.0, 0.0], 1.0));
        let b = Volume::Box(BoundingBox::new([0.5, 0.5, 0.5], [2.0, 2.0, 2.0]));
        let r = IntersectionTest::volume_vs_volume(&a, &b);
        assert!(r.is_overlapping());
    }

    #[test]
    fn test_separation_sphere_box() {
        let s = Sphere::new([5.0, 0.0, 0.0], 1.0);
        let b = BoundingBox::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let sep = IntersectionTest::separation_sphere_box(&s, &b);
        assert!((sep - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_separation_sphere_sphere() {
        let a = Sphere::new([0.0, 0.0, 0.0], 1.0);
        let b = Sphere::new([5.0, 0.0, 0.0], 1.0);
        let sep = IntersectionTest::separation_sphere_sphere(&a, &b);
        assert!((sep - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_sphere_cap_volume() {
        // Full sphere: cap with h = 2r
        let v = estimate_sphere_cap_volume(1.0, 2.0);
        let expected = (4.0 / 3.0) * std::f64::consts::PI;
        assert!((v - expected).abs() < 1e-10);
    }

    #[test]
    fn test_sphere_sphere_overlap_volume() {
        // Coincident spheres: overlap = full sphere volume
        let v = estimate_sphere_sphere_overlap(1.0, 1.0, 0.0);
        let sphere_vol = (4.0 / 3.0) * std::f64::consts::PI;
        assert!((v - sphere_vol).abs() < 1e-6);
    }
}

//! Accessibility frontier analysis: detecting the boundary between
//! accessible and inaccessible regions in the body-parameter space.
//!
//! The frontier is the hypersurface separating body parameters that can
//! reach an interactable element from those that cannot. This module
//! identifies frontier segments, computes their measure, and implements
//! neighborhood exclusion for robust coverage analysis.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use xr_types::certificate::SampleVerdict;
use xr_types::{ElementId, NUM_BODY_PARAMS};

// ──────────────────── Frontier Segment ─────────────────────────────────────

/// A segment of the accessibility frontier in parameter space.
///
/// Represents a local portion of the boundary between passing and failing
/// regions, characterized by sample points on either side.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrontierSegment {
    /// Unique identifier for this segment.
    pub id: usize,
    /// Element this frontier pertains to.
    pub element_id: ElementId,
    /// Center point of the frontier segment (midpoint between pass/fail).
    pub center: [f64; NUM_BODY_PARAMS],
    /// Estimated normal direction to the frontier (from pass to fail).
    pub normal: [f64; NUM_BODY_PARAMS],
    /// Estimated local curvature of the frontier.
    pub curvature: f64,
    /// Passing sample points near this segment.
    pub pass_points: Vec<[f64; NUM_BODY_PARAMS]>,
    /// Failing sample points near this segment.
    pub fail_points: Vec<[f64; NUM_BODY_PARAMS]>,
    /// Confidence in the frontier location (based on sample density).
    pub confidence: f64,
    /// Estimated local Lipschitz constant of the frontier.
    pub local_lipschitz: f64,
}

impl FrontierSegment {
    /// Create a new frontier segment from a pass-fail pair.
    pub fn from_pair(
        id: usize,
        element_id: ElementId,
        pass_point: [f64; NUM_BODY_PARAMS],
        fail_point: [f64; NUM_BODY_PARAMS],
    ) -> Self {
        let mut center = [0.0; NUM_BODY_PARAMS];
        let mut normal = [0.0; NUM_BODY_PARAMS];
        let mut dist_sq = 0.0;

        for i in 0..NUM_BODY_PARAMS {
            center[i] = (pass_point[i] + fail_point[i]) * 0.5;
            normal[i] = fail_point[i] - pass_point[i];
            dist_sq += normal[i] * normal[i];
        }

        // Normalize
        let dist = dist_sq.sqrt();
        if dist > 1e-15 {
            for n in &mut normal {
                *n /= dist;
            }
        }

        Self {
            id,
            element_id,
            center,
            normal,
            curvature: 0.0,
            pass_points: vec![pass_point],
            fail_points: vec![fail_point],
            confidence: 1.0,
            local_lipschitz: 0.0,
        }
    }

    /// Distance between the pass and fail sides (frontier thickness).
    pub fn thickness(&self) -> f64 {
        if self.pass_points.is_empty() || self.fail_points.is_empty() {
            return f64::INFINITY;
        }
        let p = &self.pass_points[0];
        let f = &self.fail_points[0];
        euclidean_distance(p, f)
    }

    /// Check if a point is within `epsilon` of this frontier segment.
    pub fn is_near(&self, point: &[f64; NUM_BODY_PARAMS], epsilon: f64) -> bool {
        euclidean_distance(&self.center, point) < epsilon
    }

    /// Signed distance of a point from the frontier (positive = pass side).
    pub fn signed_distance(&self, point: &[f64; NUM_BODY_PARAMS]) -> f64 {
        let mut dot = 0.0;
        for i in 0..NUM_BODY_PARAMS {
            dot += (point[i] - self.center[i]) * self.normal[i];
        }
        -dot // negative because normal points from pass to fail
    }

    /// Merge another frontier segment into this one.
    pub fn merge(&mut self, other: &FrontierSegment) {
        self.pass_points.extend_from_slice(&other.pass_points);
        self.fail_points.extend_from_slice(&other.fail_points);

        // Recompute center as average of all pass/fail midpoints
        let n = self.pass_points.len().min(self.fail_points.len());
        if n > 0 {
            let mut new_center = [0.0; NUM_BODY_PARAMS];
            for i in 0..n {
                for d in 0..NUM_BODY_PARAMS {
                    new_center[d] +=
                        (self.pass_points[i][d] + self.fail_points[i][d]) * 0.5;
                }
            }
            for c in &mut new_center {
                *c /= n as f64;
            }
            self.center = new_center;
        }

        self.confidence = (self.confidence + other.confidence) / 2.0;
    }
}

// ──────────────────── Excluded Region ──────────────────────────────────────

/// A region excluded from coverage analysis due to frontier proximity.
///
/// Points within `epsilon` of a frontier segment are excluded because
/// their accessibility status is inherently uncertain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExcludedRegion {
    /// The frontier segment causing the exclusion.
    pub frontier_id: usize,
    /// Center of the excluded region (same as frontier center).
    pub center: [f64; NUM_BODY_PARAMS],
    /// Exclusion radius.
    pub epsilon: f64,
    /// Estimated volume of the excluded region.
    pub estimated_volume: f64,
    /// Element this exclusion pertains to.
    pub element_id: ElementId,
}

impl ExcludedRegion {
    /// Create a new excluded region around a frontier segment.
    pub fn from_frontier(segment: &FrontierSegment, epsilon: f64) -> Self {
        // Volume of a 5-D ball: V = π^(5/2) / Γ(7/2) · r^5
        // Γ(7/2) = 15π^(1/2)/8
        // V = 8π²/15 · r^5
        let volume = 8.0 * std::f64::consts::PI.powi(2) / 15.0 * epsilon.powi(5);

        Self {
            frontier_id: segment.id,
            center: segment.center,
            epsilon,
            estimated_volume: volume,
            element_id: segment.element_id,
        }
    }

    /// Check if a point is within this excluded region.
    pub fn contains(&self, point: &[f64; NUM_BODY_PARAMS]) -> bool {
        euclidean_distance(&self.center, point) < self.epsilon
    }
}

// ──────────────────── Transition Surface ──────────────────────────────────

/// A joint-limit transition surface in the parameter space.
///
/// These surfaces occur where a joint limit becomes active or inactive
/// as body parameters change, causing discontinuities in reachability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionSurface {
    /// The body-parameter dimension along which the transition occurs.
    pub parameter_dim: usize,
    /// The parameter value at which the transition happens.
    pub transition_value: f64,
    /// Width of the transition zone.
    pub transition_width: f64,
    /// Element affected by this transition.
    pub element_id: ElementId,
    /// Estimated Lipschitz constant across this transition.
    pub lipschitz_estimate: f64,
}

impl TransitionSurface {
    /// Check if a point is near this transition surface.
    pub fn is_near(&self, point: &[f64; NUM_BODY_PARAMS], tolerance: f64) -> bool {
        if self.parameter_dim >= NUM_BODY_PARAMS {
            return false;
        }
        (point[self.parameter_dim] - self.transition_value).abs() < tolerance
    }
}

// ──────────────────── Frontier Detector ───────────────────────────────────

/// Detects the accessibility frontier from sample verdicts.
///
/// The frontier detection algorithm:
/// 1. Group samples by element and classify as pass/fail
/// 2. Find nearest pass-fail pairs (approximate frontier points)
/// 3. Cluster frontier points into segments
/// 4. Estimate normal direction and curvature
/// 5. Detect piecewise-Lipschitz transitions (joint-limit surfaces)
pub struct FrontierDetector {
    /// Maximum distance for two points to be considered neighbors.
    pub neighbor_radius: f64,
    /// Minimum number of pass-fail pairs to form a segment.
    pub min_segment_size: usize,
    /// Clustering merge distance for frontier segments.
    pub merge_distance: f64,
}

impl FrontierDetector {
    /// Create a new frontier detector with default settings.
    pub fn new() -> Self {
        Self {
            neighbor_radius: 0.1,
            min_segment_size: 2,
            merge_distance: 0.05,
        }
    }

    /// Create with custom parameters.
    pub fn with_params(
        neighbor_radius: f64,
        min_segment_size: usize,
        merge_distance: f64,
    ) -> Self {
        Self {
            neighbor_radius,
            min_segment_size,
            merge_distance,
        }
    }

    /// Detect frontier segments from sample verdicts.
    pub fn detect_frontier(
        &self,
        samples: &[SampleVerdict],
    ) -> Vec<FrontierSegment> {
        // Group by element
        let mut by_element: HashMap<ElementId, (Vec<usize>, Vec<usize>)> = HashMap::new();
        for (i, s) in samples.iter().enumerate() {
            let entry = by_element
                .entry(s.element_id)
                .or_insert_with(|| (Vec::new(), Vec::new()));
            if s.is_pass() {
                entry.0.push(i);
            } else {
                entry.1.push(i);
            }
        }

        let mut all_segments = Vec::new();
        let mut segment_id = 0;

        for (element_id, (pass_indices, fail_indices)) in &by_element {
            if pass_indices.is_empty() || fail_indices.is_empty() {
                continue;
            }

            // Find nearest pass-fail pairs
            let pairs = self.find_nearest_pairs(
                samples,
                pass_indices,
                fail_indices,
            );

            // Create initial segments from pairs
            let mut segments: Vec<FrontierSegment> = pairs
                .into_iter()
                .map(|(pi, fi)| {
                    let pass_point = to_array(&samples[pi].body_params);
                    let fail_point = to_array(&samples[fi].body_params);
                    let seg = FrontierSegment::from_pair(
                        segment_id,
                        *element_id,
                        pass_point,
                        fail_point,
                    );
                    segment_id += 1;
                    seg
                })
                .collect();

            // Merge nearby segments
            segments = self.merge_segments(segments);

            // Estimate curvature and Lipschitz for each segment
            for seg in &mut segments {
                seg.curvature = self.estimate_curvature(seg);
                seg.local_lipschitz = self.estimate_local_lipschitz(seg);
                seg.confidence = self.compute_confidence(seg, samples);
            }

            all_segments.extend(segments);
        }

        all_segments
    }

    /// Find nearest pass-fail pairs using brute-force search.
    fn find_nearest_pairs(
        &self,
        samples: &[SampleVerdict],
        pass_indices: &[usize],
        fail_indices: &[usize],
    ) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();

        for &pi in pass_indices {
            let pass_point = &samples[pi].body_params;
            let mut best_dist = f64::INFINITY;
            let mut best_fi = fail_indices[0];

            for &fi in fail_indices {
                let d = vec_distance(pass_point, &samples[fi].body_params);
                if d < best_dist {
                    best_dist = d;
                    best_fi = fi;
                }
            }

            if best_dist < self.neighbor_radius {
                pairs.push((pi, best_fi));
            }
        }

        // Deduplicate: keep the closest pair for each fail point
        let mut best_for_fail: HashMap<usize, (usize, f64)> = HashMap::new();
        for &(pi, fi) in &pairs {
            let d = vec_distance(&samples[pi].body_params, &samples[fi].body_params);
            let entry = best_for_fail.entry(fi).or_insert((pi, f64::INFINITY));
            if d < entry.1 {
                *entry = (pi, d);
            }
        }

        best_for_fail
            .into_iter()
            .map(|(fi, (pi, _))| (pi, fi))
            .collect()
    }

    /// Merge frontier segments that are close together.
    fn merge_segments(
        &self,
        mut segments: Vec<FrontierSegment>,
    ) -> Vec<FrontierSegment> {
        if segments.len() <= 1 {
            return segments;
        }

        let mut merged = true;
        while merged {
            merged = false;
            let n = segments.len();
            let mut to_merge: Option<(usize, usize)> = None;

            'outer: for i in 0..n {
                for j in (i + 1)..n {
                    let d = euclidean_distance(&segments[i].center, &segments[j].center);
                    if d < self.merge_distance {
                        to_merge = Some((i, j));
                        break 'outer;
                    }
                }
            }

            if let Some((i, j)) = to_merge {
                let other = segments.remove(j);
                segments[i].merge(&other);
                merged = true;
            }
        }

        segments
    }

    /// Estimate local curvature of a frontier segment.
    fn estimate_curvature(&self, segment: &FrontierSegment) -> f64 {
        if segment.pass_points.len() < 3 || segment.fail_points.len() < 3 {
            return 0.0;
        }

        // Estimate curvature from variation in frontier normals
        let mut normal_variance = 0.0;
        let n = segment.pass_points.len().min(segment.fail_points.len());

        for i in 1..n {
            let mut local_normal = [0.0; NUM_BODY_PARAMS];
            for d in 0..NUM_BODY_PARAMS {
                local_normal[d] =
                    segment.fail_points[i][d] - segment.pass_points[i][d];
            }
            let len = norm(&local_normal);
            if len > 1e-15 {
                for n_val in &mut local_normal {
                    *n_val /= len;
                }
            }

            let mut dot = 0.0;
            for d in 0..NUM_BODY_PARAMS {
                dot += local_normal[d] * segment.normal[d];
            }
            normal_variance += (1.0 - dot.abs()).powi(2);
        }

        if n > 1 {
            (normal_variance / (n - 1) as f64).sqrt()
        } else {
            0.0
        }
    }

    /// Estimate the local Lipschitz constant at a frontier segment.
    fn estimate_local_lipschitz(&self, segment: &FrontierSegment) -> f64 {
        if segment.pass_points.len() < 2 {
            return 0.0;
        }

        // Lipschitz = max |f(x) - f(y)| / |x - y| along the frontier
        // Here f is the accessibility indicator, so |f(x)-f(y)| = 1 for
        // pass-fail transitions. Use the inverse of the minimum
        // pass-fail distance as a proxy.
        let mut min_dist = f64::INFINITY;
        for p in &segment.pass_points {
            for f in &segment.fail_points {
                let d = euclidean_distance(p, f);
                if d > 1e-15 && d < min_dist {
                    min_dist = d;
                }
            }
        }

        if min_dist < f64::INFINITY {
            1.0 / min_dist
        } else {
            0.0
        }
    }

    /// Compute confidence in a frontier segment based on sample density.
    fn compute_confidence(
        &self,
        segment: &FrontierSegment,
        _samples: &[SampleVerdict],
    ) -> f64 {
        let n_points = segment.pass_points.len() + segment.fail_points.len();
        // Confidence increases with more supporting points, saturating at ~10
        1.0 - (-0.3 * n_points as f64).exp()
    }

    /// Compute the total measure (hyper-area) of the frontier.
    ///
    /// Approximates the frontier measure as the sum of local measures
    /// of each segment, estimated from the cross-sectional area.
    pub fn compute_frontier_measure(
        &self,
        segments: &[FrontierSegment],
    ) -> f64 {
        segments
            .iter()
            .map(|s| {
                let thickness = s.thickness();
                let n_points = (s.pass_points.len() + s.fail_points.len()) as f64;
                // Local measure ≈ n_points × thickness^(d-1) for d-dimensional space
                // This is a rough approximation
                let local_area = n_points * thickness.powi(NUM_BODY_PARAMS as i32 - 1);
                local_area * s.confidence
            })
            .sum()
    }

    /// Create excluded regions around each frontier segment.
    pub fn neighborhood_exclusion(
        &self,
        segments: &[FrontierSegment],
        epsilon: f64,
    ) -> Vec<ExcludedRegion> {
        segments
            .iter()
            .map(|s| ExcludedRegion::from_frontier(s, epsilon))
            .collect()
    }

    /// Detect piecewise-Lipschitz transition surfaces.
    ///
    /// These are joint-limit transitions where the accessibility function
    /// has a discontinuity or steep gradient due to a joint reaching its
    /// mechanical limit.
    pub fn detect_transitions(
        &self,
        samples: &[SampleVerdict],
    ) -> Vec<TransitionSurface> {
        let mut transitions = Vec::new();

        // Group by element
        let mut by_element: HashMap<ElementId, Vec<&SampleVerdict>> = HashMap::new();
        for s in samples {
            by_element.entry(s.element_id).or_default().push(s);
        }

        for (element_id, elem_samples) in &by_element {
            // For each dimension, check if there's a sharp transition
            for dim in 0..NUM_BODY_PARAMS {
                if let Some(ts) =
                    self.detect_dim_transition(dim, elem_samples, *element_id)
                {
                    transitions.push(ts);
                }
            }
        }

        transitions
    }

    /// Detect a transition surface along a single parameter dimension.
    fn detect_dim_transition(
        &self,
        dim: usize,
        samples: &[&SampleVerdict],
        element_id: ElementId,
    ) -> Option<TransitionSurface> {
        if samples.len() < 4 {
            return None;
        }

        // Sort samples by this dimension's value
        let mut sorted: Vec<(f64, bool)> = samples
            .iter()
            .filter_map(|s| {
                if s.body_params.len() > dim {
                    Some((s.body_params[dim], s.is_pass()))
                } else {
                    None
                }
            })
            .collect();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Find the sharpest transition (maximum change in pass rate)
        let window = (sorted.len() / 4).max(2);
        let mut best_score = 0.0;
        let mut best_idx = 0;

        for i in window..sorted.len().saturating_sub(window) {
            let left_pass: f64 = sorted[i.saturating_sub(window)..i]
                .iter()
                .map(|&(_, p)| if p { 1.0 } else { 0.0 })
                .sum::<f64>()
                / window as f64;
            let right_pass: f64 = sorted[i..i.saturating_add(window).min(sorted.len())]
                .iter()
                .map(|&(_, p)| if p { 1.0 } else { 0.0 })
                .sum::<f64>()
                / window as f64;

            let score = (left_pass - right_pass).abs();
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }

        // Require a meaningful transition
        if best_score < 0.3 {
            return None;
        }

        let transition_value = sorted[best_idx].0;
        let transition_width = if best_idx > 0 && best_idx < sorted.len() - 1 {
            (sorted[best_idx + 1].0 - sorted[best_idx - 1].0).abs()
        } else {
            0.01
        };

        Some(TransitionSurface {
            parameter_dim: dim,
            transition_value,
            transition_width,
            element_id,
            lipschitz_estimate: best_score / transition_width.max(1e-10),
        })
    }

    /// Compute the maximum Lipschitz constant across all frontier segments.
    pub fn max_lipschitz(&self, segments: &[FrontierSegment]) -> f64 {
        segments
            .iter()
            .map(|s| s.local_lipschitz)
            .fold(0.0_f64, f64::max)
    }

    /// Get frontier segments sorted by confidence (descending).
    pub fn segments_by_confidence(
        &self,
        segments: &mut [FrontierSegment],
    ) {
        segments.sort_by(|a, b| {
            b.confidence.partial_cmp(&a.confidence).unwrap()
        });
    }
}

impl Default for FrontierDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────── Utility Functions ───────────────────────────────────

/// Euclidean distance between two points.
fn euclidean_distance(a: &[f64; NUM_BODY_PARAMS], b: &[f64; NUM_BODY_PARAMS]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// Euclidean distance between two Vecs.
fn vec_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// Euclidean norm of a vector.
fn norm(v: &[f64; NUM_BODY_PARAMS]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Convert a Vec<f64> to a fixed-size array, padding with zeros.
fn to_array(v: &[f64]) -> [f64; NUM_BODY_PARAMS] {
    let mut arr = [0.0; NUM_BODY_PARAMS];
    for (i, &val) in v.iter().take(NUM_BODY_PARAMS).enumerate() {
        arr[i] = val;
    }
    arr
}

// ────────────────────────────── Tests ──────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn test_element() -> ElementId {
        Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap()
    }

    fn make_pass(params: [f64; 5]) -> SampleVerdict {
        SampleVerdict::pass(params.to_vec(), test_element())
    }

    fn make_fail(params: [f64; 5]) -> SampleVerdict {
        SampleVerdict::fail(params.to_vec(), test_element(), "unreachable".into())
    }

    #[test]
    fn test_frontier_segment_from_pair() {
        let pass = [0.5; 5];
        let fail = [0.6; 5];
        let seg = FrontierSegment::from_pair(0, test_element(), pass, fail);

        assert_eq!(seg.pass_points.len(), 1);
        assert_eq!(seg.fail_points.len(), 1);
        assert!(seg.thickness() > 0.0);
        for i in 0..5 {
            assert!((seg.center[i] - 0.55).abs() < 1e-10);
        }
    }

    #[test]
    fn test_frontier_signed_distance() {
        let pass = [0.0; 5];
        let fail = [1.0; 5];
        let seg = FrontierSegment::from_pair(0, test_element(), pass, fail);

        // A point on the pass side should have positive signed distance
        let pass_point = [0.1; 5];
        let fail_point = [0.9; 5];
        let sd_pass = seg.signed_distance(&pass_point);
        let sd_fail = seg.signed_distance(&fail_point);
        assert!(sd_pass > sd_fail);
    }

    #[test]
    fn test_excluded_region() {
        let seg = FrontierSegment::from_pair(0, test_element(), [0.5; 5], [0.6; 5]);
        let excl = ExcludedRegion::from_frontier(&seg, 0.1);
        assert!(excl.estimated_volume > 0.0);
        assert!(excl.contains(&[0.55; 5]));
        assert!(!excl.contains(&[1.0; 5]));
    }

    #[test]
    fn test_detect_frontier_basic() {
        let detector = FrontierDetector::new();
        let samples: Vec<SampleVerdict> = vec![
            make_pass([0.1, 0.1, 0.1, 0.1, 0.1]),
            make_pass([0.2, 0.1, 0.1, 0.1, 0.1]),
            make_fail([0.15, 0.1, 0.1, 0.1, 0.1]),
            make_fail([0.18, 0.1, 0.1, 0.1, 0.1]),
        ];

        let segments = detector.detect_frontier(&samples);
        // Should find at least one frontier segment
        assert!(!segments.is_empty());
    }

    #[test]
    fn test_detect_frontier_no_crossing() {
        let detector = FrontierDetector::new();
        let samples: Vec<SampleVerdict> = (0..10)
            .map(|i| make_pass([i as f64 * 0.1; 5]))
            .collect();

        let segments = detector.detect_frontier(&samples);
        assert!(segments.is_empty()); // no fail points means no frontier
    }

    #[test]
    fn test_frontier_measure() {
        let detector = FrontierDetector::new();
        let seg = FrontierSegment::from_pair(0, test_element(), [0.5; 5], [0.51; 5]);
        let measure = detector.compute_frontier_measure(&[seg]);
        assert!(measure > 0.0);
    }

    #[test]
    fn test_neighborhood_exclusion() {
        let detector = FrontierDetector::new();
        let seg = FrontierSegment::from_pair(0, test_element(), [0.5; 5], [0.51; 5]);
        let excluded = detector.neighborhood_exclusion(&[seg], 0.05);
        assert_eq!(excluded.len(), 1);
        assert!(excluded[0].contains(&[0.505; 5]));
    }

    #[test]
    fn test_transition_detection() {
        let detector = FrontierDetector::new();

        // Create samples with a clear transition at dim 0 = 0.5
        let mut samples = Vec::new();
        for i in 0..20 {
            let v = i as f64 * 0.05;
            if v < 0.5 {
                samples.push(make_pass([v, 0.5, 0.5, 0.5, 0.5]));
            } else {
                samples.push(make_fail([v, 0.5, 0.5, 0.5, 0.5]));
            }
        }

        let transitions = detector.detect_transitions(&samples);
        // Should detect a transition along dim 0
        let dim0_transitions: Vec<_> = transitions
            .iter()
            .filter(|t| t.parameter_dim == 0)
            .collect();
        assert!(!dim0_transitions.is_empty());
    }

    #[test]
    fn test_segment_merge() {
        let detector = FrontierDetector::with_params(0.1, 2, 0.05);

        let seg1 = FrontierSegment::from_pair(0, test_element(), [0.50; 5], [0.51; 5]);
        let seg2 = FrontierSegment::from_pair(1, test_element(), [0.52; 5], [0.53; 5]);

        let segments = vec![seg1, seg2];
        let merged = detector.merge_segments(segments);
        assert_eq!(merged.len(), 1); // should merge (centers are close)
    }

    #[test]
    fn test_max_lipschitz() {
        let detector = FrontierDetector::new();
        let mut seg1 = FrontierSegment::from_pair(0, test_element(), [0.5; 5], [0.51; 5]);
        seg1.local_lipschitz = 10.0;
        let mut seg2 = FrontierSegment::from_pair(1, test_element(), [0.7; 5], [0.71; 5]);
        seg2.local_lipschitz = 20.0;

        let max_l = detector.max_lipschitz(&[seg1, seg2]);
        assert!((max_l - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_to_array() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // extra element
        let arr = to_array(&v);
        assert_eq!(arr, [1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_to_array_short() {
        let v = vec![1.0, 2.0];
        let arr = to_array(&v);
        assert_eq!(arr, [1.0, 2.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = [0.0; 5];
        let b = [1.0; 5];
        let d = euclidean_distance(&a, &b);
        assert!((d - 5.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_transition_surface_near() {
        let ts = TransitionSurface {
            parameter_dim: 0,
            transition_value: 0.5,
            transition_width: 0.01,
            element_id: test_element(),
            lipschitz_estimate: 100.0,
        };
        assert!(ts.is_near(&[0.501, 0.0, 0.0, 0.0, 0.0], 0.01));
        assert!(!ts.is_near(&[0.6, 0.0, 0.0, 0.0, 0.0], 0.01));
    }

    #[test]
    fn test_confidence_increases_with_points() {
        let detector = FrontierDetector::new();
        let samples: Vec<SampleVerdict> = Vec::new();

        let seg_small = FrontierSegment {
            id: 0,
            element_id: test_element(),
            center: [0.5; 5],
            normal: [1.0, 0.0, 0.0, 0.0, 0.0],
            curvature: 0.0,
            pass_points: vec![[0.49; 5]],
            fail_points: vec![[0.51; 5]],
            confidence: 0.0,
            local_lipschitz: 0.0,
        };

        let seg_large = FrontierSegment {
            id: 0,
            element_id: test_element(),
            center: [0.5; 5],
            normal: [1.0, 0.0, 0.0, 0.0, 0.0],
            curvature: 0.0,
            pass_points: vec![[0.49; 5]; 10],
            fail_points: vec![[0.51; 5]; 10],
            confidence: 0.0,
            local_lipschitz: 0.0,
        };

        let conf_small = detector.compute_confidence(&seg_small, &samples);
        let conf_large = detector.compute_confidence(&seg_large, &samples);
        assert!(conf_large > conf_small);
    }
}

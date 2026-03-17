//! Spatial regions for multi-resolution accessibility analysis.
//!
//! Defines hyper-rectangle regions in parameter, joint, and pose spaces
//! with containment, intersection, subdivision, and classification.

use crate::interval::{Interval, IntervalVector};
use serde::{Deserialize, Serialize};
use xr_types::BoundingBox;

// ── Classification ──────────────────────────────────────────────────────

/// Classification of a spatial region relative to the reach workspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegionClassification {
    /// Provably accessible by every body-parameter sample.
    Green,
    /// Cannot be classified – needs deeper analysis.
    Yellow,
    /// Provably inaccessible by every body-parameter sample.
    Red,
}

impl RegionClassification {
    pub fn is_definite(&self) -> bool {
        matches!(self, Self::Green | Self::Red)
    }

    pub fn is_uncertain(&self) -> bool {
        matches!(self, Self::Yellow)
    }

    /// Merge two classifications conservatively: definite wins only when
    /// both agree; otherwise the result is uncertain.
    pub fn merge(self, other: Self) -> Self {
        match (self, other) {
            (Self::Green, Self::Green) => Self::Green,
            (Self::Red, Self::Red) => Self::Red,
            _ => Self::Yellow,
        }
    }

    /// Merge a collection of classifications. Empty → Green (vacuously true).
    pub fn merge_all(iter: impl IntoIterator<Item = Self>) -> Self {
        let mut result: Option<Self> = None;
        for c in iter {
            result = Some(match result {
                None => c,
                Some(prev) => prev.merge(c),
            });
        }
        result.unwrap_or(Self::Green)
    }
}

impl std::fmt::Display for RegionClassification {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Green => write!(f, "Green (accessible)"),
            Self::Yellow => write!(f, "Yellow (uncertain)"),
            Self::Red => write!(f, "Red (inaccessible)"),
        }
    }
}

// ── ParameterRegion (5-D body-parameter hyperrectangle) ─────────────────

/// Hyperrectangle in the 5-D body-parameter space
/// (stature, arm_length, shoulder_breadth, forearm_length, hand_length).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterRegion {
    pub bounds: IntervalVector,
    pub classification: Option<RegionClassification>,
}

impl ParameterRegion {
    /// Create an unclassified region from an interval vector.
    pub fn new(bounds: IntervalVector) -> Self {
        Self {
            bounds,
            classification: None,
        }
    }

    /// Create a region from explicit ranges for the 5 body parameters.
    pub fn from_ranges(ranges: &[(f64, f64); 5]) -> Self {
        Self::new(IntervalVector::from_ranges(ranges))
    }

    /// Create a region from BodyParameters min/max bounds.
    pub fn from_body_params(lo: &xr_types::BodyParameters, hi: &xr_types::BodyParameters) -> Self {
        let lo_arr = lo.to_array();
        let hi_arr = hi.to_array();
        let ranges: Vec<(f64, f64)> = lo_arr
            .iter()
            .zip(hi_arr.iter())
            .map(|(&a, &b)| (a.min(b), a.max(b)))
            .collect();
        Self::new(IntervalVector::from_ranges(&ranges))
    }

    /// Set the classification and return self.
    pub fn with_classification(mut self, c: RegionClassification) -> Self {
        self.classification = Some(c);
        self
    }

    /// Set the classification in place.
    pub fn set_classification(&mut self, c: RegionClassification) {
        self.classification = Some(c);
    }

    /// Hyper-volume of the region (product of axis widths).
    pub fn volume(&self) -> f64 {
        self.bounds.volume()
    }

    /// Check if the region has zero volume in any dimension.
    pub fn is_empty(&self) -> bool {
        self.bounds
            .components
            .iter()
            .any(|c| c.width() < f64::EPSILON)
    }

    /// Midpoint of the region in each dimension.
    pub fn midpoint(&self) -> Vec<f64> {
        self.bounds.midpoint()
    }

    /// Maximum width across all dimensions.
    pub fn diameter(&self) -> f64 {
        self.bounds.max_width()
    }

    /// Euclidean diameter (diagonal of the hyperrectangle).
    pub fn euclidean_diameter(&self) -> f64 {
        self.bounds
            .components
            .iter()
            .map(|c| c.width() * c.width())
            .sum::<f64>()
            .sqrt()
    }

    /// Check if a point (given as a slice) is inside the region.
    pub fn contains(&self, point: &[f64]) -> bool {
        self.bounds.contains_point(point)
    }

    /// Check if a BodyParameters sample is inside the region.
    pub fn contains_params(&self, params: &xr_types::BodyParameters) -> bool {
        self.bounds.contains_point(&params.to_array())
    }

    /// Test whether this region overlaps another.
    pub fn overlaps(&self, other: &ParameterRegion) -> bool {
        self.bounds.overlaps(&other.bounds)
    }

    /// Compute the intersection of two parameter regions, or None if disjoint.
    pub fn intersection(&self, other: &ParameterRegion) -> Option<ParameterRegion> {
        if self.bounds.dim() != other.bounds.dim() {
            return None;
        }
        let mut components = Vec::with_capacity(self.bounds.dim());
        for (a, b) in self
            .bounds
            .components
            .iter()
            .zip(other.bounds.components.iter())
        {
            match a.intersection(b) {
                Some(iv) => components.push(iv),
                None => return None,
            }
        }
        Some(ParameterRegion::new(IntervalVector::new(components)))
    }

    /// Check if this region is a subset of another.
    pub fn is_subset_of(&self, other: &ParameterRegion) -> bool {
        self.bounds.is_subset_of(&other.bounds)
    }

    /// Hull (smallest enclosing region) of two regions.
    pub fn hull(&self, other: &ParameterRegion) -> ParameterRegion {
        ParameterRegion::new(self.bounds.hull(&other.bounds))
    }

    /// Bisect along the widest dimension, returning two child regions.
    pub fn bisect(&self) -> (ParameterRegion, ParameterRegion) {
        let (l, r) = self.bounds.bisect_widest();
        (ParameterRegion::new(l), ParameterRegion::new(r))
    }

    /// Bisect along a specific dimension.
    pub fn bisect_dim(&self, dim: usize) -> (ParameterRegion, ParameterRegion) {
        assert!(dim < self.bounds.dim(), "dimension out of range");
        let (lo, hi) = self.bounds.components[dim].bisect();
        let mut left = self.bounds.components.clone();
        let mut right = self.bounds.components.clone();
        left[dim] = lo;
        right[dim] = hi;
        (
            ParameterRegion::new(IntervalVector::new(left)),
            ParameterRegion::new(IntervalVector::new(right)),
        )
    }

    /// Uniformly subdivide into 2^n sub-regions (one bisect per dimension).
    pub fn subdivision(&self) -> Vec<ParameterRegion> {
        let mut regions = vec![self.clone()];
        for dim in 0..self.bounds.dim() {
            let mut next = Vec::with_capacity(regions.len() * 2);
            for r in &regions {
                let (l, h) = r.bisect_dim(dim);
                next.push(l);
                next.push(h);
            }
            regions = next;
        }
        regions
    }

    /// Expand the region uniformly by a margin in every dimension.
    pub fn expand(&self, margin: f64) -> ParameterRegion {
        let components = self
            .bounds
            .components
            .iter()
            .map(|c| c.expand(margin))
            .collect();
        ParameterRegion::new(IntervalVector::new(components))
    }

    /// Get the interval for a specific parameter dimension.
    pub fn param_interval(&self, dim: usize) -> Interval {
        self.bounds.components[dim]
    }
}

// ── PoseRegion (SE(3) region) ────────────────────────────────────────────

/// Region in SE(3) represented as position bounds (BoundingBox) plus
/// orientation bounds (interval ranges on roll, pitch, yaw Euler angles).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoseRegion {
    /// Position bounds (3-D axis-aligned box).
    pub bounds: BoundingBox,
    /// Orientation bounds as Euler-angle intervals [roll, pitch, yaw] in radians.
    pub orientation_bounds: [Interval; 3],
}

impl PoseRegion {
    /// Create a PoseRegion with full orientation freedom.
    pub fn new(bounds: BoundingBox) -> Self {
        let full = Interval::new(-std::f64::consts::PI, std::f64::consts::PI);
        Self {
            bounds,
            orientation_bounds: [full, full, full],
        }
    }

    /// Create a pose region centered at a position with a given radius.
    pub fn from_center_radius(center: [f64; 3], radius: f64) -> Self {
        Self::new(BoundingBox::from_center_extents(
            center,
            [radius, radius, radius],
        ))
    }

    /// Create with explicit orientation bounds.
    pub fn with_orientation_bounds(
        bounds: BoundingBox,
        roll: Interval,
        pitch: Interval,
        yaw: Interval,
    ) -> Self {
        Self {
            bounds,
            orientation_bounds: [roll, pitch, yaw],
        }
    }

    /// Volume of the position component.
    pub fn position_volume(&self) -> f64 {
        self.bounds.volume()
    }

    /// Volume of the orientation component (product of Euler-angle ranges).
    pub fn orientation_volume(&self) -> f64 {
        self.orientation_bounds
            .iter()
            .map(|iv| iv.width())
            .product()
    }

    /// Combined (position × orientation) volume measure.
    pub fn volume(&self) -> f64 {
        self.position_volume() * self.orientation_volume()
    }

    /// Check if a position is inside the position bounds.
    pub fn contains_point(&self, p: &[f64; 3]) -> bool {
        self.bounds.contains_point(p)
    }

    /// Check if a position + Euler orientation is inside the region.
    pub fn contains_pose(&self, pos: &[f64; 3], euler: &[f64; 3]) -> bool {
        self.bounds.contains_point(pos)
            && self.orientation_bounds[0].contains(euler[0])
            && self.orientation_bounds[1].contains(euler[1])
            && self.orientation_bounds[2].contains(euler[2])
    }

    /// Check if two pose regions overlap (position and all orientation axes).
    pub fn intersects(&self, other: &PoseRegion) -> bool {
        self.bounds.intersects(&other.bounds)
            && self.orientation_bounds[0].overlaps(&other.orientation_bounds[0])
            && self.orientation_bounds[1].overlaps(&other.orientation_bounds[1])
            && self.orientation_bounds[2].overlaps(&other.orientation_bounds[2])
    }

    /// Compute the intersection of two pose regions.
    pub fn intersection(&self, other: &PoseRegion) -> Option<PoseRegion> {
        let pos_box = self.bounds.intersection(&other.bounds)?;
        let mut orient = [Interval::point(0.0); 3];
        for i in 0..3 {
            match self.orientation_bounds[i].intersection(&other.orientation_bounds[i]) {
                Some(iv) => orient[i] = iv,
                None => return None,
            }
        }
        Some(PoseRegion {
            bounds: pos_box,
            orientation_bounds: orient,
        })
    }

    /// True if the position box has zero volume.
    pub fn is_empty(&self) -> bool {
        let ext = self.bounds.extents();
        ext[0] < f64::EPSILON || ext[1] < f64::EPSILON || ext[2] < f64::EPSILON
    }

    /// Subdivide the position box along its longest axis.
    pub fn subdivision(&self) -> (PoseRegion, PoseRegion) {
        let (a, b) = self.bounds.subdivide_longest();
        (
            PoseRegion {
                bounds: a,
                orientation_bounds: self.orientation_bounds,
            },
            PoseRegion {
                bounds: b,
                orientation_bounds: self.orientation_bounds,
            },
        )
    }

    /// Expand the position bounds by a uniform margin.
    pub fn expand(&self, margin: f64) -> PoseRegion {
        PoseRegion {
            bounds: self.bounds.expand(margin),
            orientation_bounds: self.orientation_bounds,
        }
    }
}

// ── JointRegion (7-D joint-space hyperrectangle) ────────────────────────

/// Hyperrectangle in 7-D joint space (one interval per DOF).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointRegion {
    pub bounds: IntervalVector,
}

impl JointRegion {
    /// Create from an IntervalVector (typically 7-D).
    pub fn new(bounds: IntervalVector) -> Self {
        Self { bounds }
    }

    /// Create from JointLimits of a KinematicChain for given body parameters.
    pub fn from_chain(chain: &xr_types::KinematicChain, params: &xr_types::BodyParameters) -> Self {
        let intervals: Vec<Interval> = chain
            .joints
            .iter()
            .map(|j| {
                let lim = j.effective_limits(params);
                Interval::new(lim.min, lim.max)
            })
            .collect();
        Self::new(IntervalVector::new(intervals))
    }

    /// Number of degrees of freedom.
    pub fn dof(&self) -> usize {
        self.bounds.dim()
    }

    /// Hyper-volume of the joint region (product of angle ranges).
    pub fn volume(&self) -> f64 {
        self.bounds.volume()
    }

    /// Check if a joint configuration is within the region.
    pub fn contains_config(&self, config: &[f64]) -> bool {
        self.bounds.contains_point(config)
    }

    /// Check if two joint regions overlap.
    pub fn overlaps(&self, other: &JointRegion) -> bool {
        self.bounds.overlaps(&other.bounds)
    }

    /// Compute the intersection of two joint regions.
    pub fn intersection(&self, other: &JointRegion) -> Option<JointRegion> {
        if self.bounds.dim() != other.bounds.dim() {
            return None;
        }
        let mut components = Vec::with_capacity(self.bounds.dim());
        for (a, b) in self
            .bounds
            .components
            .iter()
            .zip(other.bounds.components.iter())
        {
            match a.intersection(b) {
                Some(iv) => components.push(iv),
                None => return None,
            }
        }
        Some(JointRegion::new(IntervalVector::new(components)))
    }

    /// Check if the region is empty (any dimension has zero width).
    pub fn is_empty(&self) -> bool {
        self.bounds
            .components
            .iter()
            .any(|c| c.width() < f64::EPSILON)
    }

    /// Bisect along the widest joint dimension.
    pub fn bisect(&self) -> (JointRegion, JointRegion) {
        let (l, r) = self.bounds.bisect_widest();
        (JointRegion::new(l), JointRegion::new(r))
    }

    /// Full subdivision (one bisect per dimension → 2^dof sub-regions).
    pub fn subdivision(&self) -> Vec<JointRegion> {
        let mut regions = vec![self.clone()];
        for dim in 0..self.bounds.dim() {
            let mut next = Vec::with_capacity(regions.len() * 2);
            for r in &regions {
                let (lo, hi) = r.bounds.components[dim].bisect();
                let mut left = r.bounds.components.clone();
                let mut right = r.bounds.components.clone();
                left[dim] = lo;
                right[dim] = hi;
                next.push(JointRegion::new(IntervalVector::new(left)));
                next.push(JointRegion::new(IntervalVector::new(right)));
            }
            regions = next;
        }
        regions
    }

    /// Midpoint configuration.
    pub fn midpoint_config(&self) -> Vec<f64> {
        self.bounds.midpoint()
    }

    /// Euclidean diameter of the joint region.
    pub fn diameter(&self) -> f64 {
        self.bounds
            .components
            .iter()
            .map(|c| c.width() * c.width())
            .sum::<f64>()
            .sqrt()
    }

    /// Expand every joint range by a symmetric margin.
    pub fn expand(&self, margin: f64) -> JointRegion {
        let components = self
            .bounds
            .components
            .iter()
            .map(|c| c.expand(margin))
            .collect();
        JointRegion::new(IntervalVector::new(components))
    }
}

// ── RegionNode (hierarchical multi-resolution tree) ─────────────────────

/// A node in a region hierarchy for multi-resolution spatial analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionNode {
    /// The parameter region at this node.
    pub region: ParameterRegion,
    /// Classification assigned to this node (None if not yet evaluated).
    pub classification: Option<RegionClassification>,
    /// Depth in the hierarchy (0 = root).
    pub depth: usize,
    /// Children produced by subdivision.
    pub children: Vec<RegionNode>,
}

impl RegionNode {
    /// Create a root node from a parameter region.
    pub fn root(region: ParameterRegion) -> Self {
        Self {
            region,
            classification: None,
            depth: 0,
            children: Vec::new(),
        }
    }

    /// Create a child node at depth + 1.
    pub fn child(region: ParameterRegion, depth: usize) -> Self {
        Self {
            region,
            classification: None,
            depth,
            children: Vec::new(),
        }
    }

    /// Whether this node is a leaf (has no children).
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Recursively count all leaves under this node.
    pub fn leaf_count(&self) -> usize {
        if self.is_leaf() {
            1
        } else {
            self.children.iter().map(|c| c.leaf_count()).sum()
        }
    }

    /// Total number of nodes (this node + all descendants).
    pub fn node_count(&self) -> usize {
        1 + self.children.iter().map(|c| c.node_count()).sum::<usize>()
    }

    /// Maximum depth of any leaf.
    pub fn max_depth(&self) -> usize {
        if self.is_leaf() {
            self.depth
        } else {
            self.children
                .iter()
                .map(|c| c.max_depth())
                .max()
                .unwrap_or(self.depth)
        }
    }

    /// Subdivide this leaf node, turning it into an interior node.
    /// Returns the number of children created.
    pub fn subdivide(&mut self) -> usize {
        if !self.is_leaf() {
            return 0;
        }
        let (left, right) = self.region.bisect();
        self.children = vec![
            RegionNode::child(left, self.depth + 1),
            RegionNode::child(right, self.depth + 1),
        ];
        2
    }

    /// Subdivide into 2^dim sub-regions (one bisect per dimension).
    pub fn subdivide_full(&mut self) -> usize {
        if !self.is_leaf() {
            return 0;
        }
        let subs = self.region.subdivision();
        let n = subs.len();
        self.children = subs
            .into_iter()
            .map(|r| RegionNode::child(r, self.depth + 1))
            .collect();
        n
    }

    /// Collect all leaf nodes into a flat vector.
    pub fn collect_leaves(&self) -> Vec<&RegionNode> {
        if self.is_leaf() {
            vec![self]
        } else {
            self.children
                .iter()
                .flat_map(|c| c.collect_leaves())
                .collect()
        }
    }

    /// Collect all leaf nodes classified as Yellow (uncertain).
    pub fn frontier_leaves(&self) -> Vec<&RegionNode> {
        self.collect_leaves()
            .into_iter()
            .filter(|n| n.classification == Some(RegionClassification::Yellow))
            .collect()
    }

    /// Volume fraction classified as Green.
    pub fn green_volume_fraction(&self) -> f64 {
        let leaves = self.collect_leaves();
        let total: f64 = leaves.iter().map(|n| n.region.volume()).sum();
        if total < 1e-30 {
            return 0.0;
        }
        let green: f64 = leaves
            .iter()
            .filter(|n| n.classification == Some(RegionClassification::Green))
            .map(|n| n.region.volume())
            .sum();
        green / total
    }

    /// Volume fraction classified as Red.
    pub fn red_volume_fraction(&self) -> f64 {
        let leaves = self.collect_leaves();
        let total: f64 = leaves.iter().map(|n| n.region.volume()).sum();
        if total < 1e-30 {
            return 0.0;
        }
        let red: f64 = leaves
            .iter()
            .filter(|n| n.classification == Some(RegionClassification::Red))
            .map(|n| n.region.volume())
            .sum();
        red / total
    }

    /// Volume fraction still uncertain (Yellow or unclassified).
    pub fn uncertain_volume_fraction(&self) -> f64 {
        1.0 - self.green_volume_fraction() - self.red_volume_fraction()
    }

    /// Merge children if they all have the same definite classification.
    /// Returns true if a merge occurred.
    pub fn try_merge(&mut self) -> bool {
        if self.is_leaf() {
            return false;
        }
        // Recursively merge children first.
        for child in &mut self.children {
            child.try_merge();
        }
        // All children must be leaves with the same definite classification.
        if !self.children.iter().all(|c| c.is_leaf()) {
            return false;
        }
        let first = match self.children[0].classification {
            Some(c) if c.is_definite() => c,
            _ => return false,
        };
        if self.children.iter().all(|c| c.classification == Some(first)) {
            self.classification = Some(first);
            self.children.clear();
            true
        } else {
            false
        }
    }

    /// Classify this leaf node.
    pub fn set_classification(&mut self, c: RegionClassification) {
        self.classification = Some(c);
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_5d() -> IntervalVector {
        IntervalVector::from_ranges(&[
            (1.5, 1.9),
            (0.25, 0.40),
            (0.35, 0.50),
            (0.22, 0.33),
            (0.16, 0.22),
        ])
    }

    #[test]
    fn test_classification_merge() {
        assert_eq!(
            RegionClassification::Green.merge(RegionClassification::Green),
            RegionClassification::Green
        );
        assert_eq!(
            RegionClassification::Red.merge(RegionClassification::Red),
            RegionClassification::Red
        );
        assert_eq!(
            RegionClassification::Green.merge(RegionClassification::Red),
            RegionClassification::Yellow
        );
    }

    #[test]
    fn test_classification_merge_all() {
        let all_green = vec![
            RegionClassification::Green,
            RegionClassification::Green,
        ];
        assert_eq!(RegionClassification::merge_all(all_green), RegionClassification::Green);

        let mixed = vec![
            RegionClassification::Green,
            RegionClassification::Red,
        ];
        assert_eq!(RegionClassification::merge_all(mixed), RegionClassification::Yellow);

        assert_eq!(
            RegionClassification::merge_all(std::iter::empty()),
            RegionClassification::Green
        );
    }

    #[test]
    fn test_parameter_region_volume() {
        let r = ParameterRegion::new(sample_5d());
        let vol = r.volume();
        // (0.4)*(0.15)*(0.15)*(0.11)*(0.06) = 5.94e-6
        assert!((vol - 5.94e-6).abs() < 1e-8);
    }

    #[test]
    fn test_parameter_region_contains() {
        let r = ParameterRegion::new(sample_5d());
        assert!(r.contains(&[1.7, 0.30, 0.40, 0.28, 0.19]));
        assert!(!r.contains(&[1.0, 0.30, 0.40, 0.28, 0.19]));
    }

    #[test]
    fn test_parameter_region_intersection() {
        let a = ParameterRegion::new(IntervalVector::from_ranges(&[
            (0.0, 2.0),
            (0.0, 2.0),
        ]));
        let b = ParameterRegion::new(IntervalVector::from_ranges(&[
            (1.0, 3.0),
            (1.0, 3.0),
        ]));
        let c = a.intersection(&b).unwrap();
        assert!((c.bounds.components[0].lo - 1.0).abs() < 1e-10);
        assert!((c.bounds.components[0].hi - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_parameter_region_disjoint() {
        let a = ParameterRegion::new(IntervalVector::from_ranges(&[(0.0, 1.0)]));
        let b = ParameterRegion::new(IntervalVector::from_ranges(&[(2.0, 3.0)]));
        assert!(a.intersection(&b).is_none());
    }

    #[test]
    fn test_parameter_region_bisect() {
        let r = ParameterRegion::new(IntervalVector::from_ranges(&[(0.0, 4.0), (0.0, 1.0)]));
        let (l, h) = r.bisect();
        assert!((l.volume() - r.volume() / 2.0).abs() < 1e-10);
        assert!((h.volume() - r.volume() / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_parameter_region_subdivision() {
        let r = ParameterRegion::new(IntervalVector::from_ranges(&[
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
        ]));
        let subs = r.subdivision();
        assert_eq!(subs.len(), 8); // 2^3
        let total_vol: f64 = subs.iter().map(|s| s.volume()).sum();
        assert!((total_vol - r.volume()).abs() < 1e-10);
    }

    #[test]
    fn test_parameter_region_is_empty() {
        let r = ParameterRegion::new(IntervalVector::from_ranges(&[(1.0, 1.0), (0.0, 1.0)]));
        assert!(r.is_empty());
        let r2 = ParameterRegion::new(IntervalVector::from_ranges(&[(0.0, 1.0), (0.0, 1.0)]));
        assert!(!r2.is_empty());
    }

    #[test]
    fn test_parameter_region_diameter() {
        let r = ParameterRegion::new(IntervalVector::from_ranges(&[(0.0, 3.0), (0.0, 4.0)]));
        assert!((r.euclidean_diameter() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_pose_region_basic() {
        let pr = PoseRegion::from_center_radius([0.0, 1.0, 0.0], 0.5);
        assert!(pr.contains_point(&[0.1, 1.2, -0.3]));
        assert!(!pr.contains_point(&[5.0, 5.0, 5.0]));
        assert!(pr.position_volume() > 0.0);
    }

    #[test]
    fn test_pose_region_orientation() {
        let pr = PoseRegion::with_orientation_bounds(
            BoundingBox::from_center_extents([0.0; 3], [1.0; 3]),
            Interval::new(-0.5, 0.5),
            Interval::new(-0.5, 0.5),
            Interval::new(-0.5, 0.5),
        );
        assert!(pr.contains_pose(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]));
        assert!(!pr.contains_pose(&[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0]));
    }

    #[test]
    fn test_pose_region_intersection() {
        let a = PoseRegion::from_center_radius([0.0, 0.0, 0.0], 1.0);
        let b = PoseRegion::from_center_radius([0.5, 0.5, 0.5], 1.0);
        let c = a.intersection(&b);
        assert!(c.is_some());
        let c = c.unwrap();
        assert!(c.position_volume() > 0.0);
    }

    #[test]
    fn test_pose_region_disjoint() {
        let a = PoseRegion::from_center_radius([0.0, 0.0, 0.0], 0.1);
        let b = PoseRegion::from_center_radius([10.0, 10.0, 10.0], 0.1);
        assert!(!a.intersects(&b));
        assert!(a.intersection(&b).is_none());
    }

    #[test]
    fn test_joint_region() {
        let jr = JointRegion::new(IntervalVector::from_ranges(&[
            (-1.0, 1.0),
            (-2.0, 2.0),
            (-0.5, 0.5),
            (0.0, 2.5),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-0.3, 0.5),
        ]));
        assert_eq!(jr.dof(), 7);
        assert!(jr.volume() > 0.0);
        assert!(jr.contains_config(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.1]));
        assert!(!jr.contains_config(&[5.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.1]));
    }

    #[test]
    fn test_joint_region_intersection() {
        let a = JointRegion::new(IntervalVector::from_ranges(&[(0.0, 2.0), (0.0, 2.0)]));
        let b = JointRegion::new(IntervalVector::from_ranges(&[(1.0, 3.0), (1.0, 3.0)]));
        let c = a.intersection(&b).unwrap();
        assert!((c.bounds.components[0].lo - 1.0).abs() < 1e-10);
        assert!((c.bounds.components[0].hi - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_joint_region_subdivision() {
        let jr = JointRegion::new(IntervalVector::from_ranges(&[(-1.0, 1.0), (-1.0, 1.0)]));
        let subs = jr.subdivision();
        assert_eq!(subs.len(), 4); // 2^2
        let total_vol: f64 = subs.iter().map(|s| s.volume()).sum();
        assert!((total_vol - jr.volume()).abs() < 1e-10);
    }

    #[test]
    fn test_region_node_basic() {
        let r = ParameterRegion::new(IntervalVector::from_ranges(&[(0.0, 1.0), (0.0, 1.0)]));
        let mut root = RegionNode::root(r);
        assert!(root.is_leaf());
        assert_eq!(root.leaf_count(), 1);
        assert_eq!(root.node_count(), 1);

        root.subdivide();
        assert!(!root.is_leaf());
        assert_eq!(root.leaf_count(), 2);
        assert_eq!(root.node_count(), 3);
    }

    #[test]
    fn test_region_node_merge() {
        let r = ParameterRegion::new(IntervalVector::from_ranges(&[(0.0, 1.0), (0.0, 1.0)]));
        let mut root = RegionNode::root(r);
        root.subdivide();
        root.children[0].set_classification(RegionClassification::Green);
        root.children[1].set_classification(RegionClassification::Green);

        assert!(root.try_merge());
        assert!(root.is_leaf());
        assert_eq!(root.classification, Some(RegionClassification::Green));
    }

    #[test]
    fn test_region_node_no_merge_mixed() {
        let r = ParameterRegion::new(IntervalVector::from_ranges(&[(0.0, 1.0), (0.0, 1.0)]));
        let mut root = RegionNode::root(r);
        root.subdivide();
        root.children[0].set_classification(RegionClassification::Green);
        root.children[1].set_classification(RegionClassification::Red);

        assert!(!root.try_merge());
        assert!(!root.is_leaf());
    }

    #[test]
    fn test_region_node_volume_fractions() {
        let r = ParameterRegion::new(IntervalVector::from_ranges(&[(0.0, 1.0), (0.0, 1.0)]));
        let mut root = RegionNode::root(r);
        root.subdivide();
        root.children[0].set_classification(RegionClassification::Green);
        root.children[1].set_classification(RegionClassification::Red);

        assert!((root.green_volume_fraction() - 0.5).abs() < 1e-10);
        assert!((root.red_volume_fraction() - 0.5).abs() < 1e-10);
        assert!(root.uncertain_volume_fraction().abs() < 1e-10);
    }

    #[test]
    fn test_region_node_subdivide_full() {
        let r = ParameterRegion::new(IntervalVector::from_ranges(&[
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
        ]));
        let mut root = RegionNode::root(r);
        let n = root.subdivide_full();
        assert_eq!(n, 8);
        assert_eq!(root.leaf_count(), 8);
    }

    #[test]
    fn test_region_node_frontier_leaves() {
        let r = ParameterRegion::new(IntervalVector::from_ranges(&[(0.0, 1.0)]));
        let mut root = RegionNode::root(r);
        root.subdivide();
        root.children[0].set_classification(RegionClassification::Green);
        root.children[1].set_classification(RegionClassification::Yellow);

        let frontier = root.frontier_leaves();
        assert_eq!(frontier.len(), 1);
        assert_eq!(frontier[0].classification, Some(RegionClassification::Yellow));
    }

    #[test]
    fn test_from_body_params() {
        let lo = xr_types::BodyParameters::small_female();
        let hi = xr_types::BodyParameters::large_male();
        let r = ParameterRegion::from_body_params(&lo, &hi);
        assert_eq!(r.bounds.dim(), 5);
        assert!(r.contains_params(&xr_types::BodyParameters::average_male()));
    }
}

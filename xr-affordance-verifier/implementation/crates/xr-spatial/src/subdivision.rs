//! Subdivision strategies for multi-resolution parameter-space refinement.
//!
//! Provides adaptive, uniform, and octree-based refinement of the
//! 5-D body-parameter space, driving the Tier 1 verification loop.

use crate::interval::{Interval, IntervalVector};
use crate::region::{ParameterRegion, RegionClassification, RegionNode};
use serde::{Deserialize, Serialize};

// ── SubdivisionStrategy trait ───────────────────────────────────────────

/// Strategy for subdividing a parameter region into finer sub-regions.
pub trait SubdivisionStrategy: std::fmt::Debug {
    /// Produce sub-regions from a parent region.
    fn subdivide(&self, region: &ParameterRegion) -> Vec<ParameterRegion>;

    /// Human-readable name for the strategy.
    fn name(&self) -> &str;
}

// ── UniformGrid ─────────────────────────────────────────────────────────

/// Uniform grid subdivision: bisect every dimension `depth` times,
/// yielding 2^(dim × depth) cells of equal size.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct UniformGrid {
    /// Number of bisections per dimension.
    pub depth: usize,
}

impl UniformGrid {
    pub fn new(depth: usize) -> Self {
        Self { depth }
    }
}

impl SubdivisionStrategy for UniformGrid {
    fn subdivide(&self, region: &ParameterRegion) -> Vec<ParameterRegion> {
        let dim = region.bounds.dim();
        let mut regions = vec![region.clone()];

        for _ in 0..self.depth {
            let mut next = Vec::with_capacity(regions.len() * 2 * dim);
            for r in &regions {
                for d in 0..dim {
                    let (lo, hi) = r.bounds.components[d].bisect();
                    let mut left = r.bounds.components.clone();
                    let mut right = r.bounds.components.clone();
                    left[d] = lo;
                    right[d] = hi;
                    // Only push if this is the last dimension or first pass
                    if d == dim - 1 {
                        // Do nothing here; we accumulate below
                    }
                }
            }
            // Actually, build the grid correctly: bisect all dims per cell
            next.clear();
            for r in &regions {
                let subs = subdivide_all_dims_once(r);
                next.extend(subs);
            }
            regions = next;
        }
        regions
    }

    fn name(&self) -> &str {
        "UniformGrid"
    }
}

/// Bisect every dimension once → 2^dim sub-regions.
fn subdivide_all_dims_once(region: &ParameterRegion) -> Vec<ParameterRegion> {
    let dim = region.bounds.dim();
    let mut regions = vec![region.clone()];
    for d in 0..dim {
        let mut next = Vec::with_capacity(regions.len() * 2);
        for r in &regions {
            let (lo, hi) = r.bounds.components[d].bisect();
            let mut left = r.bounds.components.clone();
            let mut right = r.bounds.components.clone();
            left[d] = lo;
            right[d] = hi;
            next.push(ParameterRegion::new(IntervalVector::new(left)));
            next.push(ParameterRegion::new(IntervalVector::new(right)));
        }
        regions = next;
    }
    regions
}

// ── AdaptiveOctree ──────────────────────────────────────────────────────

/// Adaptive octree-style subdivision: bisect only the widest dimension,
/// giving two children per step. Mimics octree behavior in high dimensions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AdaptiveOctree {
    /// Minimum cell diameter below which subdivision stops.
    pub min_diameter: f64,
    /// Maximum depth of the tree.
    pub max_depth: usize,
}

impl AdaptiveOctree {
    pub fn new(min_diameter: f64, max_depth: usize) -> Self {
        Self {
            min_diameter,
            max_depth,
        }
    }
}

impl SubdivisionStrategy for AdaptiveOctree {
    fn subdivide(&self, region: &ParameterRegion) -> Vec<ParameterRegion> {
        if region.diameter() < self.min_diameter {
            return vec![region.clone()];
        }
        let (left, right) = region.bisect();
        vec![left, right]
    }

    fn name(&self) -> &str {
        "AdaptiveOctree"
    }
}

// ── FrontierAdaptive ────────────────────────────────────────────────────

/// Frontier-adaptive subdivision: refine only Yellow (uncertain) regions,
/// using the wrapping factor to decide the splitting dimension.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FrontierAdaptive {
    /// Minimum cell diameter.
    pub min_diameter: f64,
    /// Maximum total cells before stopping.
    pub max_cells: usize,
    /// Wrapping-factor threshold below which we stop refining.
    pub wrapping_threshold: f64,
}

impl FrontierAdaptive {
    pub fn new(min_diameter: f64, max_cells: usize) -> Self {
        Self {
            min_diameter,
            max_cells,
            wrapping_threshold: 2.0,
        }
    }

    pub fn with_wrapping_threshold(mut self, t: f64) -> Self {
        self.wrapping_threshold = t;
        self
    }

    /// Choose the best dimension to split based on the widest interval
    /// weighted by the ratio of interval-to-affine width (a proxy for
    /// wrapping contribution). Falls back to widest dimension.
    fn choose_split_dim(&self, region: &ParameterRegion) -> usize {
        let dim = region.bounds.dim();
        let mut best_dim = 0;
        let mut best_score = f64::NEG_INFINITY;

        for d in 0..dim {
            let w = region.bounds.components[d].width();
            // Weight dimensions that are wider more heavily. For a more
            // sophisticated approach the caller could supply per-dimension
            // wrapping sensitivities; here we use w^2 as a heuristic
            // (variance contribution scales quadratically).
            let score = w * w;
            if score > best_score {
                best_score = score;
                best_dim = d;
            }
        }
        best_dim
    }
}

impl SubdivisionStrategy for FrontierAdaptive {
    fn subdivide(&self, region: &ParameterRegion) -> Vec<ParameterRegion> {
        if region.diameter() < self.min_diameter {
            return vec![region.clone()];
        }
        let dim = self.choose_split_dim(region);
        let (l, r) = region.bisect_dim(dim);
        vec![l, r]
    }

    fn name(&self) -> &str {
        "FrontierAdaptive"
    }
}

// ── SimpleBisect (for backward compat) ──────────────────────────────────

/// Simple bisection along the widest dimension.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Bisect;

impl SubdivisionStrategy for Bisect {
    fn subdivide(&self, region: &ParameterRegion) -> Vec<ParameterRegion> {
        let (l, r) = region.bisect();
        vec![l, r]
    }

    fn name(&self) -> &str {
        "Bisect"
    }
}

// ── SubdivisionTree ─────────────────────────────────────────────────────

/// Tree that manages a hierarchy of subdivided parameter regions.
///
/// Maintains a flat list of leaf regions together with an optional
/// hierarchical `RegionNode` tree for deeper structural queries.
#[derive(Debug, Clone)]
pub struct SubdivisionTree {
    /// Flat list of current leaf regions.
    pub regions: Vec<ParameterRegion>,
    /// Maximum depth before subdivision halts.
    pub max_depth: usize,
    /// Current depth counter.
    current_depth: usize,
}

impl SubdivisionTree {
    /// Create a new tree with a single root region.
    pub fn new(root: ParameterRegion) -> Self {
        Self {
            regions: vec![root],
            max_depth: 20,
            current_depth: 0,
        }
    }

    pub fn with_max_depth(mut self, d: usize) -> Self {
        self.max_depth = d;
        self
    }

    /// Initial uniform subdivision using a strategy.
    pub fn subdivide(&mut self, strategy: &dyn SubdivisionStrategy) {
        if self.current_depth >= self.max_depth {
            return;
        }
        let mut next = Vec::new();
        for r in &self.regions {
            next.extend(strategy.subdivide(r));
        }
        self.regions = next;
        self.current_depth += 1;
    }

    /// Refine only the uncertain (Yellow) regions using a strategy.
    /// Returns the number of regions that were refined.
    pub fn refine_uncertain(&mut self, strategy: &dyn SubdivisionStrategy) -> usize {
        if self.current_depth >= self.max_depth {
            return 0;
        }
        let (uncertain, definite): (Vec<_>, Vec<_>) = self
            .regions
            .drain(..)
            .partition(|r| r.classification == Some(RegionClassification::Yellow));

        let count = uncertain.len();
        let mut next = definite;
        for r in &uncertain {
            next.extend(strategy.subdivide(r));
        }
        self.regions = next;
        self.current_depth += 1;
        count
    }

    /// Merge adjacent regions that share the same definite classification.
    /// Returns the number of merges performed (rough count).
    pub fn merge(&mut self) -> usize {
        // Group regions by classification, merge only adjacent greens/reds.
        // Simple approach: try to merge consecutive same-class regions.
        let mut merged = 0usize;
        if self.regions.len() < 2 {
            return 0;
        }

        let mut new_regions: Vec<ParameterRegion> = Vec::new();
        let mut i = 0;
        while i < self.regions.len() {
            let class = self.regions[i].classification;
            // Try to merge with next region if same definite classification
            if i + 1 < self.regions.len()
                && class.map(|c| c.is_definite()).unwrap_or(false)
                && self.regions[i + 1].classification == class
            {
                let hull = self.regions[i].hull(&self.regions[i + 1]);
                let mut merged_region = hull;
                merged_region.classification = class;
                new_regions.push(merged_region);
                merged += 1;
                i += 2;
            } else {
                new_regions.push(self.regions[i].clone());
                i += 1;
            }
        }
        self.regions = new_regions;
        merged
    }

    /// Total number of cells (leaf regions).
    pub fn count_cells(&self) -> usize {
        self.regions.len()
    }

    /// Find frontier cells (classified Yellow / uncertain).
    pub fn find_frontier_cells(&self) -> Vec<&ParameterRegion> {
        self.regions
            .iter()
            .filter(|r| {
                r.classification == Some(RegionClassification::Yellow)
                    || r.classification.is_none()
            })
            .collect()
    }

    /// Find cells classified Green.
    pub fn find_green_cells(&self) -> Vec<&ParameterRegion> {
        self.regions
            .iter()
            .filter(|r| r.classification == Some(RegionClassification::Green))
            .collect()
    }

    /// Find cells classified Red.
    pub fn find_red_cells(&self) -> Vec<&ParameterRegion> {
        self.regions
            .iter()
            .filter(|r| r.classification == Some(RegionClassification::Red))
            .collect()
    }

    /// Volume fraction of Green-classified cells.
    pub fn green_volume_fraction(&self) -> f64 {
        let total: f64 = self.regions.iter().map(|r| r.volume()).sum();
        if total < 1e-30 {
            return 0.0;
        }
        let green: f64 = self
            .regions
            .iter()
            .filter(|r| r.classification == Some(RegionClassification::Green))
            .map(|r| r.volume())
            .sum();
        green / total
    }

    /// Volume fraction of Red-classified cells.
    pub fn red_volume_fraction(&self) -> f64 {
        let total: f64 = self.regions.iter().map(|r| r.volume()).sum();
        if total < 1e-30 {
            return 0.0;
        }
        let red: f64 = self
            .regions
            .iter()
            .filter(|r| r.classification == Some(RegionClassification::Red))
            .map(|r| r.volume())
            .sum();
        red / total
    }

    /// Volume fraction still uncertain.
    pub fn uncertain_volume_fraction(&self) -> f64 {
        1.0 - self.green_volume_fraction() - self.red_volume_fraction()
    }

    /// Current depth.
    pub fn depth(&self) -> usize {
        self.current_depth
    }

    /// Classify a region by index.
    pub fn classify(&mut self, index: usize, class: RegionClassification) {
        if index < self.regions.len() {
            self.regions[index].set_classification(class);
        }
    }

    /// Convert flat region list to a `RegionNode` tree. The tree has a
    /// synthetic root whose volume is the hull of all regions.
    pub fn to_region_tree(&self) -> RegionNode {
        if self.regions.is_empty() {
            return RegionNode::root(ParameterRegion::new(IntervalVector::new(vec![])));
        }
        let mut hull = self.regions[0].clone();
        for r in &self.regions[1..] {
            hull = hull.hull(r);
        }
        let mut root = RegionNode::root(hull);
        for r in &self.regions {
            let mut child = RegionNode::child(r.clone(), 1);
            child.classification = r.classification;
            root.children.push(child);
        }
        root
    }

    /// Summary string.
    pub fn summary(&self) -> String {
        let total = self.count_cells();
        let green = self.find_green_cells().len();
        let red = self.find_red_cells().len();
        let yellow = self.find_frontier_cells().len();
        format!(
            "SubdivisionTree: {total} cells (G={green}, R={red}, Y/unclassified={yellow}), depth={}",
            self.current_depth
        )
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn region_2d() -> ParameterRegion {
        ParameterRegion::new(IntervalVector::from_ranges(&[(0.0, 1.0), (0.0, 1.0)]))
    }

    fn region_5d() -> ParameterRegion {
        ParameterRegion::new(IntervalVector::from_ranges(&[
            (1.5, 1.9),
            (0.25, 0.40),
            (0.35, 0.50),
            (0.22, 0.33),
            (0.16, 0.22),
        ]))
    }

    #[test]
    fn test_bisect_strategy() {
        let r = region_2d();
        let subs = Bisect.subdivide(&r);
        assert_eq!(subs.len(), 2);
        let total_vol: f64 = subs.iter().map(|s| s.volume()).sum();
        assert!((total_vol - r.volume()).abs() < 1e-10);
    }

    #[test]
    fn test_uniform_grid_depth_1() {
        let r = region_2d();
        let subs = UniformGrid::new(1).subdivide(&r);
        // depth=1 → bisect all dims once → 2^2 = 4 cells
        assert_eq!(subs.len(), 4);
        let total_vol: f64 = subs.iter().map(|s| s.volume()).sum();
        assert!((total_vol - r.volume()).abs() < 1e-10);
    }

    #[test]
    fn test_uniform_grid_depth_2() {
        let r = region_2d();
        let subs = UniformGrid::new(2).subdivide(&r);
        // depth=2 → 2^(2*2) = 16 cells
        assert_eq!(subs.len(), 16);
    }

    #[test]
    fn test_adaptive_octree() {
        let r = region_5d();
        let strategy = AdaptiveOctree::new(0.001, 20);
        let subs = strategy.subdivide(&r);
        assert_eq!(subs.len(), 2);
    }

    #[test]
    fn test_adaptive_octree_min_diameter() {
        let tiny = ParameterRegion::new(IntervalVector::from_ranges(&[(0.0, 0.0001)]));
        let strategy = AdaptiveOctree::new(0.001, 20);
        let subs = strategy.subdivide(&tiny);
        assert_eq!(subs.len(), 1); // too small to subdivide
    }

    #[test]
    fn test_frontier_adaptive() {
        let r = region_5d();
        let strategy = FrontierAdaptive::new(0.001, 1000);
        let subs = strategy.subdivide(&r);
        assert_eq!(subs.len(), 2);
    }

    #[test]
    fn test_frontier_adaptive_splits_widest_weighted() {
        let r = ParameterRegion::new(IntervalVector::from_ranges(&[
            (0.0, 10.0),
            (0.0, 0.1),
        ]));
        let strategy = FrontierAdaptive::new(0.001, 1000);
        let subs = strategy.subdivide(&r);
        assert_eq!(subs.len(), 2);
        // Should split along dim 0 (much wider)
        assert!((subs[0].bounds.components[0].width() - 5.0).abs() < 1e-10);
        assert!((subs[0].bounds.components[1].width() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_subdivision_tree_basic() {
        let r = region_2d();
        let mut tree = SubdivisionTree::new(r);
        assert_eq!(tree.count_cells(), 1);

        tree.subdivide(&Bisect);
        assert_eq!(tree.count_cells(), 2);
        assert_eq!(tree.depth(), 1);
    }

    #[test]
    fn test_subdivision_tree_refine_uncertain() {
        let r = region_2d();
        let mut tree = SubdivisionTree::new(r);
        tree.subdivide(&Bisect);

        tree.classify(0, RegionClassification::Green);
        tree.classify(1, RegionClassification::Yellow);

        let refined = tree.refine_uncertain(&Bisect);
        assert_eq!(refined, 1);
        assert_eq!(tree.count_cells(), 3); // 1 green + 2 new from yellow
    }

    #[test]
    fn test_subdivision_tree_volume_fractions() {
        let r = region_2d();
        let mut tree = SubdivisionTree::new(r);
        tree.subdivide(&Bisect);

        tree.classify(0, RegionClassification::Green);
        tree.classify(1, RegionClassification::Red);

        assert!((tree.green_volume_fraction() - 0.5).abs() < 1e-10);
        assert!((tree.red_volume_fraction() - 0.5).abs() < 1e-10);
        assert!(tree.uncertain_volume_fraction().abs() < 1e-10);
    }

    #[test]
    fn test_subdivision_tree_find_frontier() {
        let r = region_2d();
        let mut tree = SubdivisionTree::new(r);
        tree.subdivide(&Bisect);

        tree.classify(0, RegionClassification::Green);
        tree.classify(1, RegionClassification::Yellow);

        let frontier = tree.find_frontier_cells();
        assert_eq!(frontier.len(), 1);
    }

    #[test]
    fn test_subdivision_tree_merge() {
        let r = region_2d();
        let mut tree = SubdivisionTree::new(r.clone());
        tree.subdivide(&Bisect);

        tree.classify(0, RegionClassification::Green);
        tree.classify(1, RegionClassification::Green);

        let merged = tree.merge();
        assert!(merged >= 1);
        assert!(tree.count_cells() < 2);
    }

    #[test]
    fn test_subdivision_tree_to_region_tree() {
        let r = region_2d();
        let mut tree = SubdivisionTree::new(r);
        tree.subdivide(&Bisect);
        tree.classify(0, RegionClassification::Green);
        tree.classify(1, RegionClassification::Red);

        let node = tree.to_region_tree();
        assert_eq!(node.children.len(), 2);
        assert_eq!(
            node.children[0].classification,
            Some(RegionClassification::Green)
        );
        assert_eq!(
            node.children[1].classification,
            Some(RegionClassification::Red)
        );
    }

    #[test]
    fn test_subdivision_tree_max_depth() {
        let r = region_2d();
        let mut tree = SubdivisionTree::new(r).with_max_depth(2);
        tree.subdivide(&Bisect);
        tree.subdivide(&Bisect);
        tree.subdivide(&Bisect); // should be no-op
        assert_eq!(tree.depth(), 2);
    }

    #[test]
    fn test_subdivision_tree_summary() {
        let r = region_2d();
        let tree = SubdivisionTree::new(r);
        let s = tree.summary();
        assert!(s.contains("1 cells"));
    }

    #[test]
    fn test_5d_uniform_grid() {
        let r = region_5d();
        let subs = UniformGrid::new(1).subdivide(&r);
        // 2^5 = 32 cells
        assert_eq!(subs.len(), 32);
        let total_vol: f64 = subs.iter().map(|s| s.volume()).sum();
        assert!((total_vol - r.volume()).abs() / r.volume() < 1e-10);
    }
}

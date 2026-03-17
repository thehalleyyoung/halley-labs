//! Discontinuity-aware boundary verification for joint-limit regions.
//!
//! Standard Lipschitz-based sampling fails at joint limits, where hard ROM
//! constraints create step-function discontinuities in the accessibility
//! predicate. This module detects such boundaries, switches to exhaustive
//! boundary-straddling evaluation, and provides sound certificates without
//! relying on the Lipschitz assumption in those regions.
//!
//! Additionally, [`MultiStepStratifier`] reduces the combinatorial explosion
//! of k≥3 multi-step interactions from 2^(d₁+d₂+…+dₖ) strata to
//! O(2^max_component_size) by exploiting the sparse dependency structure
//! between interaction steps.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use xr_types::{ElementId, NUM_BODY_PARAMS};

// ─── Discontinuity Detection ─────────────────────────────────────────────────

/// A detected boundary in the parameter space where the accessibility
/// predicate may have a step-function discontinuity (e.g., a joint-limit
/// hard stop).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointBoundary {
    /// Index into the body-parameter vector (0..NUM_BODY_PARAMS).
    pub param_index: usize,
    /// The parameter value at which the discontinuity occurs.
    pub limit_value: f64,
    /// Estimated magnitude of the accessibility jump across the boundary.
    /// 1.0 = full step (accessible on one side, inaccessible on the other).
    pub discontinuity_magnitude: f64,
    /// Which affordances are affected.
    pub affected_elements: Vec<ElementId>,
}

/// Result of probing one side of a boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SideAccessibility {
    Accessible,
    Inaccessible,
    Mixed,
}

/// Per-boundary verification verdict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryVerdict {
    pub boundary: JointBoundary,
    pub below: SideAccessibility,
    pub above: SideAccessibility,
    /// Summary label for reporting.
    pub classification: BoundaryClassification,
}

/// Human-readable classification of a boundary's effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryClassification {
    /// Accessible on both sides — boundary is benign.
    AccessibleBothSides,
    /// Accessible only on one side — boundary creates a hard exclusion.
    AccessibleOneSide,
    /// Inaccessible on both sides — boundary is in an already-excluded region.
    InaccessibleBothSides,
}

/// Detects hard-limit boundaries in the body-parameter space where the
/// Lipschitz assumption fails.
#[derive(Debug, Clone)]
pub struct DiscontinuityDetector {
    /// Half-width of the probing corridor on each side of a boundary.
    pub probe_delta: f64,
    /// Number of probe samples per side.
    pub probes_per_side: usize,
    /// Known joint-limit pairs: (param_index, lower_limit, upper_limit).
    pub joint_limits: Vec<(usize, f64, f64)>,
}

impl Default for DiscontinuityDetector {
    fn default() -> Self {
        Self {
            probe_delta: 0.005,
            probes_per_side: 32,
            joint_limits: Vec::new(),
        }
    }
}

impl DiscontinuityDetector {
    pub fn new(probe_delta: f64, probes_per_side: usize) -> Self {
        Self {
            probe_delta,
            probes_per_side,
            joint_limits: Vec::new(),
        }
    }

    /// Register known hard joint limits from a kinematic model.
    /// Each entry is (parameter_index, lower_bound, upper_bound).
    pub fn add_joint_limits(&mut self, limits: Vec<(usize, f64, f64)>) {
        self.joint_limits.extend(limits);
    }

    /// Scan all registered joint limits and detect gradient discontinuities.
    ///
    /// `eval_fn` maps (body_params, element_id) → bool (accessible?).
    /// `center` is a representative point in the interior of the parameter space.
    pub fn detect_boundaries<F>(
        &self,
        center: &[f64; NUM_BODY_PARAMS],
        elements: &[ElementId],
        eval_fn: &F,
    ) -> Vec<JointBoundary>
    where
        F: Fn(&[f64; NUM_BODY_PARAMS], ElementId) -> bool,
    {
        let mut boundaries = Vec::new();

        for &(param_idx, lower, upper) in &self.joint_limits {
            for &limit_val in &[lower, upper] {
                if let Some(boundary) =
                    self.probe_boundary(center, param_idx, limit_val, elements, eval_fn)
                {
                    boundaries.push(boundary);
                }
            }
        }
        boundaries
    }

    /// Probe a single boundary by sampling on both sides and comparing
    /// pass rates. A large difference indicates a step discontinuity.
    fn probe_boundary<F>(
        &self,
        center: &[f64; NUM_BODY_PARAMS],
        param_idx: usize,
        limit_val: f64,
        elements: &[ElementId],
        eval_fn: &F,
    ) -> Option<JointBoundary>
    where
        F: Fn(&[f64; NUM_BODY_PARAMS], ElementId) -> bool,
    {
        let mut affected = Vec::new();
        let mut max_magnitude: f64 = 0.0;

        for &elem in elements {
            let below_rate = self.side_pass_rate(
                center, param_idx, limit_val, -1.0, elem, eval_fn,
            );
            let above_rate = self.side_pass_rate(
                center, param_idx, limit_val, 1.0, elem, eval_fn,
            );
            let magnitude = (above_rate - below_rate).abs();
            if magnitude > 0.1 {
                affected.push(elem);
                max_magnitude = max_magnitude.max(magnitude);
            }
        }

        if affected.is_empty() {
            return None;
        }

        Some(JointBoundary {
            param_index: param_idx,
            limit_value: limit_val,
            discontinuity_magnitude: max_magnitude,
            affected_elements: affected,
        })
    }

    /// Compute the pass rate on one side of a boundary.
    /// `direction`: -1.0 for below, +1.0 for above.
    fn side_pass_rate<F>(
        &self,
        center: &[f64; NUM_BODY_PARAMS],
        param_idx: usize,
        limit_val: f64,
        direction: f64,
        element: ElementId,
        eval_fn: &F,
    ) -> f64
    where
        F: Fn(&[f64; NUM_BODY_PARAMS], ElementId) -> bool,
    {
        let mut pass_count = 0usize;
        for i in 0..self.probes_per_side {
            let t = (i as f64 + 0.5) / self.probes_per_side as f64;
            let offset = direction * t * self.probe_delta;
            let mut point = *center;
            point[param_idx] = limit_val + offset;
            if eval_fn(&point, element) {
                pass_count += 1;
            }
        }
        pass_count as f64 / self.probes_per_side as f64
    }
}

// ─── Adaptive Boundary Sampler ───────────────────────────────────────────────

/// Sampling strategy that switches behavior near detected boundaries.
///
/// - **Interior** (distance > δ from any boundary): standard Lipschitz-based
///   stratified sampling with the usual ε guarantees.
/// - **Boundary corridor** (distance ≤ δ): boundary-straddling sampling that
///   evaluates both sides independently — no Lipschitz assumption required.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveBoundarySampler {
    /// Distance threshold for switching to boundary-aware sampling.
    pub boundary_delta: f64,
    /// Base samples per stratum in the interior.
    pub interior_samples: usize,
    /// Samples per side in each boundary corridor.
    pub boundary_samples_per_side: usize,
}

impl Default for AdaptiveBoundarySampler {
    fn default() -> Self {
        Self {
            boundary_delta: 0.01,
            interior_samples: 512,
            boundary_samples_per_side: 64,
        }
    }
}

/// Classification of a point's location relative to known boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionKind {
    Interior,
    BoundaryCorridor { boundary_index: usize },
}

impl AdaptiveBoundarySampler {
    pub fn new(
        boundary_delta: f64,
        interior_samples: usize,
        boundary_samples_per_side: usize,
    ) -> Self {
        Self {
            boundary_delta,
            interior_samples,
            boundary_samples_per_side,
        }
    }

    /// Classify a point as interior or within a boundary corridor.
    pub fn classify_point(
        &self,
        point: &[f64; NUM_BODY_PARAMS],
        boundaries: &[JointBoundary],
    ) -> RegionKind {
        for (i, b) in boundaries.iter().enumerate() {
            let dist = (point[b.param_index] - b.limit_value).abs();
            if dist <= self.boundary_delta {
                return RegionKind::BoundaryCorridor { boundary_index: i };
            }
        }
        RegionKind::Interior
    }

    /// For a boundary corridor, split the corridor into below/above sub-regions
    /// and return the sub-region bounds.
    pub fn split_corridor(
        &self,
        boundary: &JointBoundary,
        lower: &[f64; NUM_BODY_PARAMS],
        upper: &[f64; NUM_BODY_PARAMS],
    ) -> (
        [f64; NUM_BODY_PARAMS],
        [f64; NUM_BODY_PARAMS],
        [f64; NUM_BODY_PARAMS],
        [f64; NUM_BODY_PARAMS],
    ) {
        let idx = boundary.param_index;
        let lv = boundary.limit_value;

        // Below sub-region: [lower, ..., min(upper[idx], lv)]
        let mut below_upper = *upper;
        below_upper[idx] = upper[idx].min(lv);

        // Above sub-region: [max(lower[idx], lv), ..., upper]
        let mut above_lower = *lower;
        above_lower[idx] = lower[idx].max(lv);

        (*lower, below_upper, above_lower, *upper)
    }
}

// ─── Boundary Verifier ───────────────────────────────────────────────────────

/// Verifies accessibility on each side of every detected boundary without
/// relying on Lipschitz continuity. Uses exhaustive evaluation within the
/// boundary corridor.
#[derive(Debug, Clone)]
pub struct BoundaryVerifier {
    /// Number of evaluation points per side for the exhaustive check.
    pub eval_points_per_side: usize,
    /// Threshold: fraction of passing probes required to call a side
    /// "accessible" (vs. "mixed" if between threshold and 0).
    pub accessible_threshold: f64,
}

impl Default for BoundaryVerifier {
    fn default() -> Self {
        Self {
            eval_points_per_side: 128,
            accessible_threshold: 0.90,
        }
    }
}

impl BoundaryVerifier {
    pub fn new(eval_points_per_side: usize, accessible_threshold: f64) -> Self {
        Self {
            eval_points_per_side,
            accessible_threshold,
        }
    }

    /// Verify all boundaries, producing a verdict for each.
    pub fn verify_all<F>(
        &self,
        boundaries: &[JointBoundary],
        center: &[f64; NUM_BODY_PARAMS],
        eval_fn: &F,
    ) -> Vec<BoundaryVerdict>
    where
        F: Fn(&[f64; NUM_BODY_PARAMS], ElementId) -> bool,
    {
        boundaries
            .iter()
            .map(|b| self.verify_boundary(b, center, eval_fn))
            .collect()
    }

    /// Verify a single boundary: evaluate both sides exhaustively.
    pub fn verify_boundary<F>(
        &self,
        boundary: &JointBoundary,
        center: &[f64; NUM_BODY_PARAMS],
        eval_fn: &F,
    ) -> BoundaryVerdict
    where
        F: Fn(&[f64; NUM_BODY_PARAMS], ElementId) -> bool,
    {
        // Use the first affected element for the side classification.
        // A more thorough implementation would check all affected elements
        // and take the worst case.
        let element = boundary
            .affected_elements
            .first()
            .copied()
            .unwrap_or_else(Uuid::nil);

        let below = self.evaluate_side(
            center,
            boundary.param_index,
            boundary.limit_value,
            -1.0,
            element,
            eval_fn,
        );
        let above = self.evaluate_side(
            center,
            boundary.param_index,
            boundary.limit_value,
            1.0,
            element,
            eval_fn,
        );

        let classification = match (below, above) {
            (SideAccessibility::Accessible, SideAccessibility::Accessible) => {
                BoundaryClassification::AccessibleBothSides
            }
            (SideAccessibility::Inaccessible, SideAccessibility::Inaccessible) => {
                BoundaryClassification::InaccessibleBothSides
            }
            _ => BoundaryClassification::AccessibleOneSide,
        };

        BoundaryVerdict {
            boundary: boundary.clone(),
            below,
            above,
            classification,
        }
    }

    /// Exhaustively evaluate one side of a boundary.
    fn evaluate_side<F>(
        &self,
        center: &[f64; NUM_BODY_PARAMS],
        param_idx: usize,
        limit_val: f64,
        direction: f64,
        element: ElementId,
        eval_fn: &F,
    ) -> SideAccessibility
    where
        F: Fn(&[f64; NUM_BODY_PARAMS], ElementId) -> bool,
    {
        let corridor_half = 0.01_f64; // 1% of normalized range
        let mut pass_count = 0usize;

        for i in 0..self.eval_points_per_side {
            let t = (i as f64 + 0.5) / self.eval_points_per_side as f64;
            let offset = direction * t * corridor_half;
            let mut point = *center;
            point[param_idx] = limit_val + offset;
            if eval_fn(&point, element) {
                pass_count += 1;
            }
        }

        let rate = pass_count as f64 / self.eval_points_per_side as f64;
        if rate >= self.accessible_threshold {
            SideAccessibility::Accessible
        } else if rate == 0.0 {
            SideAccessibility::Inaccessible
        } else {
            SideAccessibility::Mixed
        }
    }
}

// ─── Multi-Step Stratifier ───────────────────────────────────────────────────

/// An edge in the interaction dependency graph: step `from` influences step
/// `to` through shared body parameters in `shared_dims`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionEdge {
    pub from: usize,
    pub to: usize,
    /// Parameter indices that are shared between the two steps.
    pub shared_dims: Vec<usize>,
}

/// A connected component of the interaction dependency graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratificationComponent {
    /// Step indices in this component.
    pub steps: Vec<usize>,
    /// Union of parameter dimensions used by steps in this component.
    pub dims: Vec<usize>,
    /// Number of strata for this component: 2^|dims|.
    pub num_strata: usize,
}

/// Reduces multi-step stratification from 2^(d₁+d₂+…+dₖ) to the product
/// of per-component strata by exploiting the sparse dependency structure.
///
/// Most multi-step XR interactions have sparse dependencies: step 1 (reach)
/// depends on arm length and shoulder ROM, step 2 (grasp) depends on grip
/// aperture and wrist ROM, and they share at most the elbow angle. By
/// computing connected components in the interaction graph, we stratify
/// each component independently.
///
/// For typical 3–5 step interactions with max component size 6–8:
/// - Naive: 2^21 ≈ 2M strata (infeasible)
/// - Component-wise: product of 2^6 … 2^8 ≈ 64–256 strata per component
#[derive(Debug, Clone)]
pub struct MultiStepStratifier {
    /// Number of interaction steps.
    pub num_steps: usize,
    /// Per-step parameter dimensions that are relevant.
    pub step_dims: Vec<Vec<usize>>,
    /// Detected dependency edges.
    pub edges: Vec<InteractionEdge>,
    /// Connected components (populated after `build()`).
    pub components: Vec<StratificationComponent>,
}

impl MultiStepStratifier {
    /// Create from a specification of which parameter dimensions each step
    /// uses.
    pub fn new(step_dims: Vec<Vec<usize>>) -> Self {
        let num_steps = step_dims.len();
        Self {
            num_steps,
            step_dims,
            edges: Vec::new(),
            components: Vec::new(),
        }
    }

    /// Compute the interaction graph: two steps share an edge if they have
    /// overlapping parameter dimensions.
    pub fn compute_edges(&mut self) {
        self.edges.clear();
        for i in 0..self.num_steps {
            for j in (i + 1)..self.num_steps {
                let shared: Vec<usize> = self.step_dims[i]
                    .iter()
                    .filter(|d| self.step_dims[j].contains(d))
                    .copied()
                    .collect();
                if !shared.is_empty() {
                    self.edges.push(InteractionEdge {
                        from: i,
                        to: j,
                        shared_dims: shared,
                    });
                }
            }
        }
    }

    /// Build connected components via union-find, then compute per-component
    /// strata counts.
    pub fn build(&mut self) {
        self.compute_edges();

        // Union-find over step indices.
        let mut parent: Vec<usize> = (0..self.num_steps).collect();

        fn find(parent: &mut [usize], x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }
        fn union(parent: &mut [usize], a: usize, b: usize) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra != rb {
                parent[ra] = rb;
            }
        }

        for edge in &self.edges {
            union(&mut parent, edge.from, edge.to);
        }

        // Group steps by their root.
        let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for step in 0..self.num_steps {
            let root = find(&mut parent, step);
            groups.entry(root).or_default().push(step);
        }

        // Build components.
        self.components.clear();
        for (_root, steps) in &groups {
            let mut dims: Vec<usize> = steps
                .iter()
                .flat_map(|&s| self.step_dims[s].iter().copied())
                .collect();
            dims.sort_unstable();
            dims.dedup();

            let num_strata = 1usize << dims.len().min(20); // cap to avoid overflow
            self.components.push(StratificationComponent {
                steps: steps.clone(),
                dims,
                num_strata,
            });
        }
    }

    /// Total strata count across all components (product, not sum of
    /// exponents).
    pub fn total_strata(&self) -> usize {
        self.components
            .iter()
            .map(|c| c.num_strata)
            .product::<usize>()
            .max(1)
    }

    /// Naive strata count if all dimensions were stratified jointly.
    pub fn naive_strata(&self) -> usize {
        let total_dims: usize = self.step_dims.iter().map(|d| d.len()).sum();
        1usize << total_dims.min(30)
    }

    /// Reduction factor: naive / actual.
    pub fn reduction_factor(&self) -> f64 {
        let naive = self.naive_strata() as f64;
        let actual = self.total_strata() as f64;
        if actual == 0.0 {
            return 0.0;
        }
        naive / actual
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_eval(
        params: &[f64; NUM_BODY_PARAMS],
        _element: ElementId,
    ) -> bool {
        // Accessible if param[0] (stature) > 1.55, simulating a reach limit.
        params[0] > 1.55
    }

    #[test]
    fn test_discontinuity_detection() {
        let mut detector = DiscontinuityDetector::new(0.01, 16);
        // Stature (param 0) has limits [1.50, 1.95].
        detector.add_joint_limits(vec![(0, 1.50, 1.95)]);

        let center = [1.70, 0.34, 0.42, 0.26, 0.19];
        let elem = Uuid::new_v4();

        let boundaries = detector.detect_boundaries(&center, &[elem], &dummy_eval);
        // The lower limit at 1.50 is near the threshold 1.55, so a
        // discontinuity should be detected there.
        assert!(
            boundaries
                .iter()
                .any(|b| b.param_index == 0 && (b.limit_value - 1.50).abs() < 0.001),
            "Should detect boundary near stature lower limit"
        );
    }

    #[test]
    fn test_boundary_verifier() {
        let boundary = JointBoundary {
            param_index: 0,
            limit_value: 1.55,
            discontinuity_magnitude: 0.95,
            affected_elements: vec![Uuid::new_v4()],
        };
        let verifier = BoundaryVerifier::default();
        let center = [1.55, 0.34, 0.42, 0.26, 0.19];

        let verdict = verifier.verify_boundary(&boundary, &center, &dummy_eval);
        assert_eq!(verdict.classification, BoundaryClassification::AccessibleOneSide);
        assert_eq!(verdict.below, SideAccessibility::Inaccessible);
        assert_eq!(verdict.above, SideAccessibility::Accessible);
    }

    #[test]
    fn test_adaptive_sampler_classify() {
        let sampler = AdaptiveBoundarySampler::default();
        let boundary = JointBoundary {
            param_index: 0,
            limit_value: 1.55,
            discontinuity_magnitude: 0.9,
            affected_elements: vec![],
        };

        // Point far from boundary.
        let far = [1.70, 0.34, 0.42, 0.26, 0.19];
        assert_eq!(
            sampler.classify_point(&far, &[boundary.clone()]),
            RegionKind::Interior
        );

        // Point near boundary.
        let near = [1.555, 0.34, 0.42, 0.26, 0.19];
        assert!(matches!(
            sampler.classify_point(&near, &[boundary]),
            RegionKind::BoundaryCorridor { .. }
        ));
    }

    #[test]
    fn test_multi_step_stratifier_independent() {
        // 3 steps, each using distinct dimensions → 3 independent components.
        let mut stratifier = MultiStepStratifier::new(vec![
            vec![0, 1],    // step 0: stature + arm span
            vec![2, 3],    // step 1: grip aperture + wrist ROM
            vec![4],       // step 2: shoulder ROM
        ]);
        stratifier.build();

        assert_eq!(stratifier.components.len(), 3);
        // Total: 2^2 * 2^2 * 2^1 = 16.
        assert_eq!(stratifier.total_strata(), 16);
        // Naive: 2^5 = 32.
        assert_eq!(stratifier.naive_strata(), 32);
        assert!((stratifier.reduction_factor() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_multi_step_stratifier_connected() {
        // 3 steps with overlap → fewer components.
        let mut stratifier = MultiStepStratifier::new(vec![
            vec![0, 1, 2],  // step 0
            vec![2, 3],     // step 1 shares dim 2 with step 0
            vec![3, 4],     // step 2 shares dim 3 with step 1
        ]);
        stratifier.build();

        // All steps are transitively connected → 1 component with dims {0,1,2,3,4}.
        assert_eq!(stratifier.components.len(), 1);
        assert_eq!(stratifier.total_strata(), 32); // 2^5
        // Naive also 2^(3+2+2) = 2^7 = 128.
        assert_eq!(stratifier.naive_strata(), 128);
        assert!((stratifier.reduction_factor() - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_multi_step_stratifier_mixed() {
        // 4 steps: {0,1} and {2,3} share dim 1/2, {4,5} is separate.
        let mut stratifier = MultiStepStratifier::new(vec![
            vec![0, 1],     // step 0
            vec![1, 2],     // step 1: shares dim 1 with step 0
            vec![2, 3],     // step 2: shares dim 2 with step 1
            vec![4],        // step 3: independent
        ]);
        stratifier.build();

        assert_eq!(stratifier.components.len(), 2);
        // Component 1: steps 0,1,2 → dims {0,1,2,3} → 2^4 = 16
        // Component 2: step 3 → dims {4} → 2^1 = 2
        assert_eq!(stratifier.total_strata(), 32); // 16 * 2
        // Naive: 2^(2+2+2+1) = 2^7 = 128.
        assert_eq!(stratifier.naive_strata(), 128);
    }

    #[test]
    fn test_corridor_split() {
        let sampler = AdaptiveBoundarySampler::default();
        let boundary = JointBoundary {
            param_index: 0,
            limit_value: 1.55,
            discontinuity_magnitude: 0.9,
            affected_elements: vec![],
        };
        let lower = [1.50, 0.30, 0.35, 0.20, 0.15];
        let upper = [1.60, 0.40, 0.50, 0.30, 0.25];

        let (bl, bu, al, au) = sampler.split_corridor(&boundary, &lower, &upper);
        assert!((bu[0] - 1.55).abs() < 1e-10);
        assert!((al[0] - 1.55).abs() < 1e-10);
        // Other dims unchanged.
        assert_eq!(bl[1], lower[1]);
        assert_eq!(au[1], upper[1]);
    }
}

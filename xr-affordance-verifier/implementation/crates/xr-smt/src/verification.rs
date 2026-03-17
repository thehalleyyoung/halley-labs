//! Verification orchestration for region-based accessibility proofs.
//!
//! This module provides [`SmtVerifier`], the top-level driver that subdivides
//! the body-parameter space into axis-aligned hyper-rectangular regions,
//! linearizes forward kinematics at each region midpoint, encodes the
//! reachability predicate as QF_LRA assertions, and solves to produce a
//! [`RegionVerdict`] per region.  A [`VerificationReport`] aggregates
//! per-region results and volume accounting.
//!
//! The recursive subdivision strategy ([`SmtVerifier::verify_recursive`])
//! splits inconclusive regions along their widest dimension until the solver
//! returns a definitive answer or the depth / volume budget is exhausted.

use serde::{Deserialize, Serialize};
use std::time::Instant;
use uuid::Uuid;

use xr_types::{BodyParameters, KinematicChain, VerifierError};

use crate::constraints::{BoundedVariable, ConstraintSet};
use crate::expr::{SmtDecl, SmtExpr, SmtSort};
use crate::linearization::{LinearizationEngine, LinearizedModel};
use crate::proof::SmtProof;
use crate::solver::{InternalSolver, SmtSolver, SolverResult};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of body-parameter dimensions (stature, arm_length,
/// shoulder_breadth, forearm_length, hand_length).
const BODY_PARAM_DIM: usize = 5;

/// Default endpoint-to-target tolerance in metres.
const DEFAULT_TOLERANCE: f64 = 0.01;

/// Default per-query solver timeout in seconds.
const DEFAULT_TIMEOUT_SECS: f64 = 10.0;

/// Default maximum recursive subdivision depth.
const DEFAULT_MAX_DEPTH: usize = 8;

/// Default minimum region volume below which subdivision stops.
const DEFAULT_MIN_REGION_VOL: f64 = 1e-12;

/// Backwards-compatible alias: other modules (e.g. `optimization`) reference
/// this name when working with body-parameter regions.
pub type ParameterRegion = VerificationRegion;

// ---------------------------------------------------------------------------
// RegionVerdict
// ---------------------------------------------------------------------------

/// Outcome of verifying a single body-parameter region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegionVerdict {
    /// The linearized reachability query is satisfiable for the region —
    /// there exists a joint-angle configuration reaching the target for
    /// body parameters within the region bounds.
    Verified {
        /// Optional formal proof certificate.
        proof: Option<Box<SmtProof>>,
    },
    /// The query is unsatisfiable — a body-parameter vector was found
    /// (or inferred) that cannot reach the target.
    Refuted {
        /// Body-parameter values constituting the counterexample.
        counterexample: Vec<f64>,
    },
    /// The solver could not determine accessibility.
    Unknown {
        /// Human-readable explanation.
        reason: String,
    },
    /// The solver exceeded its time budget.
    Timeout,
}

impl RegionVerdict {
    /// Returns `true` when the region was verified accessible.
    pub fn is_verified(&self) -> bool {
        matches!(self, RegionVerdict::Verified { .. })
    }

    /// Returns `true` when a counterexample was found.
    pub fn is_refuted(&self) -> bool {
        matches!(self, RegionVerdict::Refuted { .. })
    }

    /// Returns the counterexample body-parameter slice, if any.
    pub fn counterexample(&self) -> Option<&[f64]> {
        match self {
            RegionVerdict::Refuted { counterexample } => Some(counterexample),
            _ => None,
        }
    }

    /// Returns `true` when the verdict is `Unknown`.
    pub fn is_unknown(&self) -> bool {
        matches!(self, RegionVerdict::Unknown { .. })
    }

    /// Returns `true` when the solver timed out.
    pub fn is_timeout(&self) -> bool {
        matches!(self, RegionVerdict::Timeout)
    }
}

// ---------------------------------------------------------------------------
// VerificationRegion
// ---------------------------------------------------------------------------

/// Axis-aligned hyper-rectangular region in body-parameter space.
///
/// Each dimension `i` has inclusive bounds `[lower[i], upper[i]]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationRegion {
    /// Lower bounds per dimension.
    pub lower: Vec<f64>,
    /// Upper bounds per dimension.
    pub upper: Vec<f64>,
    /// Unique identifier for tracing and logging.
    pub id: Uuid,
}

impl VerificationRegion {
    /// Create a new region.  `lower` and `upper` must have equal length.
    pub fn new(lower: Vec<f64>, upper: Vec<f64>) -> Self {
        debug_assert_eq!(lower.len(), upper.len(), "lower/upper dimension mismatch");
        Self {
            lower,
            upper,
            id: Uuid::new_v4(),
        }
    }

    /// Component-wise midpoint of the region.
    pub fn midpoint(&self) -> Vec<f64> {
        self.lower
            .iter()
            .zip(&self.upper)
            .map(|(lo, hi)| (lo + hi) * 0.5)
            .collect()
    }

    /// Hyper-volume (product of side lengths).  Returns 0 for degenerate regions.
    pub fn volume(&self) -> f64 {
        self.lower
            .iter()
            .zip(&self.upper)
            .map(|(lo, hi)| (hi - lo).max(0.0))
            .product()
    }

    /// Test whether `point` lies inside the region (inclusive on all faces).
    pub fn contains(&self, point: &[f64]) -> bool {
        if point.len() != self.lower.len() {
            return false;
        }
        point
            .iter()
            .enumerate()
            .all(|(i, &v)| v >= self.lower[i] && v <= self.upper[i])
    }

    /// Split the region into two halves along dimension `dim`.
    ///
    /// The first child covers `[lower, mid]` and the second `[mid, upper]`
    /// where `mid` is the midpoint along `dim`.
    pub fn split(&self, dim: usize) -> (Self, Self) {
        let mid = (self.lower[dim] + self.upper[dim]) * 0.5;

        let mut lo_upper = self.upper.clone();
        lo_upper[dim] = mid;

        let mut hi_lower = self.lower.clone();
        hi_lower[dim] = mid;

        (
            VerificationRegion::new(self.lower.clone(), lo_upper),
            VerificationRegion::new(hi_lower, self.upper.clone()),
        )
    }

    /// Maximum side length across all dimensions (L∞ diameter).
    pub fn diameter(&self) -> f64 {
        self.lower
            .iter()
            .zip(&self.upper)
            .map(|(lo, hi)| (hi - lo).abs())
            .fold(0.0_f64, f64::max)
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.lower.len()
    }

    /// Index of the widest dimension (used to pick the split axis).
    fn widest_dim(&self) -> usize {
        self.lower
            .iter()
            .zip(&self.upper)
            .enumerate()
            .max_by(|(_, (al, au)), (_, (bl, bu))| {
                let wa = (*au - *al).abs();
                let wb = (*bu - *bl).abs();
                wa.partial_cmp(&wb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Build a [`ConstraintSet`] encoding the region bounds on body-parameter
    /// variables named `p_0` … `p_{n-1}`.
    fn to_constraint_set(&self) -> ConstraintSet {
        let mut cs = ConstraintSet::new();
        for i in 0..self.lower.len() {
            cs.add_variable(BoundedVariable::new(format!("p_{}", i), SmtSort::Real)
                .with_bounds(self.lower[i], self.upper[i]));
        }
        cs
    }
}

// ---------------------------------------------------------------------------
// SmtVerifier
// ---------------------------------------------------------------------------

/// Orchestrates region-based SMT verification of kinematic reachability.
///
/// Construct with [`SmtVerifier::new`] and customise via the builder methods
/// [`with_timeout`](SmtVerifier::with_timeout),
/// [`with_max_depth`](SmtVerifier::with_max_depth), and
/// [`with_min_region_volume`](SmtVerifier::with_min_region_volume).
pub struct SmtVerifier {
    /// Maximum solver time per query (seconds).
    pub timeout_secs: f64,
    /// Maximum recursive subdivision depth.
    pub max_depth: usize,
    /// Minimum region volume below which subdivision stops.
    pub min_region_volume: f64,
    /// Engine used to linearize forward kinematics.
    linearization_engine: LinearizationEngine,
    /// Endpoint-to-target tolerance (metres).
    tolerance: f64,
}

impl SmtVerifier {
    /// Create a verifier with sensible defaults.
    pub fn new() -> Self {
        Self {
            timeout_secs: DEFAULT_TIMEOUT_SECS,
            max_depth: DEFAULT_MAX_DEPTH,
            min_region_volume: DEFAULT_MIN_REGION_VOL,
            linearization_engine: LinearizationEngine::new(),
            tolerance: DEFAULT_TOLERANCE,
        }
    }

    /// Set the per-query solver timeout.
    pub fn with_timeout(mut self, secs: f64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Set the maximum recursion depth for subdivision.
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Set the minimum region volume threshold.
    pub fn with_min_region_volume(mut self, vol: f64) -> Self {
        self.min_region_volume = vol;
        self
    }

    // ------------------------------------------------------------------
    // Core verification
    // ------------------------------------------------------------------

    /// Verify a single region by linearizing at its midpoint.
    ///
    /// 1. Convert the midpoint to [`BodyParameters`].
    /// 2. Linearize FK at those parameters with the supplied reference
    ///    joint-angle configuration.
    /// 3. Encode the reachability predicate as QF_LRA assertions.
    /// 4. Solve.  `Sat` → [`RegionVerdict::Verified`],
    ///    `Unsat` → [`RegionVerdict::Refuted`] (midpoint used as the
    ///    counterexample), otherwise `Unknown` / `Timeout`.
    pub fn verify_region(
        &self,
        chain: &KinematicChain,
        target: &[f64; 3],
        region: &VerificationRegion,
        ref_config: &[f64],
    ) -> Result<RegionVerdict, VerifierError> {
        let midpoint = region.midpoint();
        if midpoint.len() < BODY_PARAM_DIM {
            return Err(VerifierError::DimensionMismatch {
                expected: BODY_PARAM_DIM,
                got: midpoint.len(),
            });
        }

        let params = BodyParameters::from_array(&[
            midpoint[0],
            midpoint[1],
            midpoint[2],
            midpoint[3],
            midpoint[4],
        ]);

        // Linearize FK at the region midpoint.
        let model = self
            .linearization_engine
            .linearize(chain, &params, ref_config)?;

        // Build the SMT query (declarations + assertions).
        let (decls, assertions) = self.build_reachability_query(&model, target, region)?;

        // Create and configure the solver.
        let mut solver = InternalSolver::new();
        for decl in &decls {
            solver.declare(decl.clone());
        }

        let combined = if assertions.len() == 1 {
            vec![assertions.into_iter().next().unwrap()]
        } else {
            vec![SmtExpr::And(assertions)]
        };

        // Solve and map the result to a verdict.
        match solver.check_sat(&combined)? {
            SolverResult::Sat(_) => Ok(RegionVerdict::Verified { proof: None }),
            SolverResult::Unsat => Ok(RegionVerdict::Refuted {
                counterexample: midpoint,
            }),
            SolverResult::Unknown(_) => Ok(RegionVerdict::Unknown {
                reason: "solver returned unknown".into(),
            }),
            SolverResult::Timeout => Ok(RegionVerdict::Timeout),
        }
    }

    /// Recursively subdivide and verify a region.
    ///
    /// If the initial check yields `Unknown` or `Timeout` and the depth and
    /// volume budgets have not been exhausted, the region is split along its
    /// widest dimension and both halves are verified independently.
    ///
    /// * A `Refuted` child propagates immediately (early exit).
    /// * Two `Verified` children produce a `Verified` parent.
    /// * Mixed or inconclusive results produce `Unknown`.
    pub fn verify_recursive(
        &self,
        chain: &KinematicChain,
        target: &[f64; 3],
        region: &VerificationRegion,
        ref_config: &[f64],
        depth: usize,
    ) -> Result<RegionVerdict, VerifierError> {
        let verdict = self.verify_region(chain, target, region, ref_config)?;

        match &verdict {
            RegionVerdict::Verified { .. } | RegionVerdict::Refuted { .. } => {
                return Ok(verdict);
            }
            RegionVerdict::Unknown { .. } | RegionVerdict::Timeout => {
                if depth >= self.max_depth || region.volume() < self.min_region_volume {
                    return Ok(verdict);
                }
            }
        }

        // Subdivide along the widest dimension.
        let dim = region.widest_dim();
        let (left, right) = region.split(dim);

        let left_v = self.verify_recursive(chain, target, &left, ref_config, depth + 1)?;
        if left_v.is_refuted() {
            return Ok(left_v);
        }

        let right_v = self.verify_recursive(chain, target, &right, ref_config, depth + 1)?;
        if right_v.is_refuted() {
            return Ok(right_v);
        }

        // Both halves verified → parent verified.
        if left_v.is_verified() && right_v.is_verified() {
            return Ok(RegionVerdict::Verified { proof: None });
        }

        // At least one side was inconclusive.
        Ok(RegionVerdict::Unknown {
            reason: format!(
                "subdivision at depth {} inconclusive (left={}, right={})",
                depth,
                verdict_label(&left_v),
                verdict_label(&right_v),
            ),
        })
    }

    /// Build SMT declarations and assertions for a reachability query.
    ///
    /// Three groups of constraints are emitted:
    ///
    /// 1. **Body-parameter bounds** — one real-sorted variable `p_i` per
    ///    dimension, bounded by the region.
    /// 2. **Linearized model** — declarations and assertions produced by
    ///    [`LinearizedModel::to_smt_expressions`], which encode joint-angle
    ///    variables, their kinematic limits, and the linearized FK equations
    ///    defining `endpoint_x`, `endpoint_y`, `endpoint_z`.
    /// 3. **Endpoint tolerance** — the linearized endpoint must lie within
    ///    `self.tolerance` of `target` in each Cartesian axis.
    pub fn build_reachability_query(
        &self,
        model: &LinearizedModel,
        target: &[f64; 3],
        region: &VerificationRegion,
    ) -> Result<(Vec<SmtDecl>, Vec<SmtExpr>), VerifierError> {
        let mut decls: Vec<SmtDecl> = Vec::new();
        let mut assertions: Vec<SmtExpr> = Vec::new();

        // ---- 1. Body-parameter bounds ----
        let n_params = region.lower.len();
        for i in 0..n_params {
            let name = format!("p_{}", i);
            decls.push(SmtDecl::new(&name, SmtSort::Real));

            // p_i >= lower[i]
            assertions.push(SmtExpr::Ge(
                Box::new(SmtExpr::Var(name.clone())),
                Box::new(SmtExpr::Const(region.lower[i])),
            ));
            // p_i <= upper[i]
            assertions.push(SmtExpr::Le(
                Box::new(SmtExpr::Var(name)),
                Box::new(SmtExpr::Const(region.upper[i])),
            ));
        }

        // ---- 2. Linearized model (joint vars, limits, FK equations) ----
        let model_assertions = model.to_smt_expressions("p", "q");
        assertions.extend(model_assertions);

        // ---- 3. Endpoint tolerance ----
        let axis_names = ["x", "y", "z"];
        for (axis_idx, axis_name) in axis_names.iter().enumerate() {
            let ep_var = format!("endpoint_{}", axis_name);

            // endpoint >= target - tolerance
            assertions.push(SmtExpr::Ge(
                Box::new(SmtExpr::Var(ep_var.clone())),
                Box::new(SmtExpr::Const(target[axis_idx] - self.tolerance)),
            ));
            // endpoint <= target + tolerance
            assertions.push(SmtExpr::Le(
                Box::new(SmtExpr::Var(ep_var)),
                Box::new(SmtExpr::Const(target[axis_idx] + self.tolerance)),
            ));
        }

        Ok((decls, assertions))
    }

    /// Verify a batch of regions and collect results into a
    /// [`VerificationReport`].
    ///
    /// Each region is verified via [`verify_recursive`](Self::verify_recursive)
    /// starting at depth 0.  Wall-clock time is recorded.
    pub fn run(
        &self,
        chain: &KinematicChain,
        target: &[f64; 3],
        regions: &[VerificationRegion],
        ref_config: &[f64],
    ) -> Result<VerificationReport, VerifierError> {
        let start = Instant::now();
        let mut report = VerificationReport::new();
        for region in regions {
            let verdict = self.verify_recursive(chain, target, region, ref_config, 0)?;
            report.add_result(region.clone(), verdict);
        }
        report.elapsed_secs = start.elapsed().as_secs_f64();
        Ok(report)
    }
}

// ---------------------------------------------------------------------------
// VerificationReport
// ---------------------------------------------------------------------------

/// Accumulated results of verifying a collection of body-parameter regions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationReport {
    /// Per-region verdicts.
    pub regions: Vec<(VerificationRegion, RegionVerdict)>,
    /// Sum of volumes of all submitted regions.
    pub total_volume: f64,
    /// Sum of volumes of verified regions.
    pub verified_volume: f64,
    /// Sum of volumes of refuted regions.
    pub refuted_volume: f64,
    /// Sum of volumes of unknown / timed-out regions.
    pub unknown_volume: f64,
    /// Wall-clock time for the entire verification run (seconds).
    pub elapsed_secs: f64,
}

impl VerificationReport {
    /// Create an empty report.
    pub fn new() -> Self {
        Self {
            regions: Vec::new(),
            total_volume: 0.0,
            verified_volume: 0.0,
            refuted_volume: 0.0,
            unknown_volume: 0.0,
            elapsed_secs: 0.0,
        }
    }

    /// Record a region and its verdict, updating volume accumulators.
    pub fn add_result(&mut self, region: VerificationRegion, verdict: RegionVerdict) {
        let vol = region.volume();
        self.total_volume += vol;
        match &verdict {
            RegionVerdict::Verified { .. } => self.verified_volume += vol,
            RegionVerdict::Refuted { .. } => self.refuted_volume += vol,
            RegionVerdict::Unknown { .. } | RegionVerdict::Timeout => {
                self.unknown_volume += vol;
            }
        }
        self.regions.push((region, verdict));
    }

    /// Fraction of total volume that is verified (`verified / total`).
    ///
    /// Returns `0.0` when the total volume is zero or negative.
    pub fn coverage(&self) -> f64 {
        if self.total_volume <= 0.0 {
            return 0.0;
        }
        self.verified_volume / self.total_volume
    }

    /// One-line human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "VerificationReport: {} regions, coverage {:.2}% \
             (verified {:.6}, refuted {:.6}, unknown {:.6}), elapsed {:.3}s",
            self.regions.len(),
            self.coverage() * 100.0,
            self.verified_volume,
            self.refuted_volume,
            self.unknown_volume,
            self.elapsed_secs,
        )
    }

    /// Number of distinct regions recorded.
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Number of verified regions.
    pub fn verified_count(&self) -> usize {
        self.regions
            .iter()
            .filter(|(_, v)| v.is_verified())
            .count()
    }

    /// Number of refuted regions.
    pub fn refuted_count(&self) -> usize {
        self.regions
            .iter()
            .filter(|(_, v)| v.is_refuted())
            .count()
    }
}

impl Default for VerificationReport {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Short label for log / debug messages.
fn verdict_label(v: &RegionVerdict) -> &'static str {
    match v {
        RegionVerdict::Verified { .. } => "verified",
        RegionVerdict::Refuted { .. } => "refuted",
        RegionVerdict::Unknown { .. } => "unknown",
        RegionVerdict::Timeout => "timeout",
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- RegionVerdict ----------------------------------------------------

    #[test]
    fn test_region_verdict_methods() {
        let verified = RegionVerdict::Verified { proof: None };
        assert!(verified.is_verified());
        assert!(!verified.is_refuted());
        assert!(!verified.is_unknown());
        assert!(!verified.is_timeout());
        assert!(verified.counterexample().is_none());

        let refuted = RegionVerdict::Refuted {
            counterexample: vec![1.7, 0.32, 0.44, 0.26, 0.19],
        };
        assert!(!refuted.is_verified());
        assert!(refuted.is_refuted());
        assert_eq!(
            refuted.counterexample(),
            Some(&[1.7, 0.32, 0.44, 0.26, 0.19][..]),
        );

        let unknown = RegionVerdict::Unknown {
            reason: "solver gave up".into(),
        };
        assert!(!unknown.is_verified());
        assert!(!unknown.is_refuted());
        assert!(unknown.is_unknown());
        assert!(unknown.counterexample().is_none());

        let timeout = RegionVerdict::Timeout;
        assert!(!timeout.is_verified());
        assert!(timeout.is_timeout());
        assert!(timeout.counterexample().is_none());
    }

    // ---- VerificationRegion -----------------------------------------------

    #[test]
    fn test_verification_region_split() {
        let region = VerificationRegion::new(vec![0.0, 0.0], vec![4.0, 6.0]);

        // Split along dimension 0.
        let (left, right) = region.split(0);
        assert_eq!(left.lower, vec![0.0, 0.0]);
        assert_eq!(left.upper, vec![2.0, 6.0]);
        assert_eq!(right.lower, vec![2.0, 0.0]);
        assert_eq!(right.upper, vec![4.0, 6.0]);

        // Volumes sum to the original.
        let orig_vol = region.volume();
        let sum_vol = left.volume() + right.volume();
        assert!((orig_vol - sum_vol).abs() < 1e-12);

        // Split along dimension 1.
        let (lo, hi) = region.split(1);
        assert!((lo.upper[1] - 3.0).abs() < 1e-12);
        assert!((hi.lower[1] - 3.0).abs() < 1e-12);
        assert!((lo.volume() + hi.volume() - orig_vol).abs() < 1e-12);

        // Children receive distinct UUIDs.
        assert_ne!(left.id, right.id);
    }

    #[test]
    fn test_verification_region_volume() {
        // 1-D
        let r1 = VerificationRegion::new(vec![0.0], vec![5.0]);
        assert!((r1.volume() - 5.0).abs() < 1e-12);

        // 3-D: (4−1)×(6−2)×(5−3) = 3×4×2 = 24
        let r2 = VerificationRegion::new(vec![1.0, 2.0, 3.0], vec![4.0, 6.0, 5.0]);
        assert!((r2.volume() - 24.0).abs() < 1e-12);

        // Degenerate (zero-thickness in one dimension).
        let r3 = VerificationRegion::new(vec![1.0, 1.0], vec![1.0, 2.0]);
        assert!(r3.volume().abs() < 1e-12);

        // Midpoint.
        let mid = r2.midpoint();
        assert!((mid[0] - 2.5).abs() < 1e-12);
        assert!((mid[1] - 4.0).abs() < 1e-12);
        assert!((mid[2] - 4.0).abs() < 1e-12);

        // Contains.
        assert!(r2.contains(&[2.5, 4.0, 4.0]));
        assert!(!r2.contains(&[0.0, 4.0, 4.0]));
        assert!(!r2.contains(&[2.5])); // wrong dimension

        // Diameter (max side = 6 − 2 = 4).
        assert!((r2.diameter() - 4.0).abs() < 1e-12);

        // ndim.
        assert_eq!(r2.ndim(), 3);
    }

    // ---- VerificationReport -----------------------------------------------

    #[test]
    fn test_verification_report() {
        let mut report = VerificationReport::new();
        assert_eq!(report.coverage(), 0.0);
        assert_eq!(report.region_count(), 0);

        // Verified region (volume = 1).
        let r1 = VerificationRegion::new(vec![0.0, 0.0], vec![1.0, 1.0]);
        report.add_result(r1, RegionVerdict::Verified { proof: None });

        // Refuted region (volume = 1).
        let r2 = VerificationRegion::new(vec![1.0, 0.0], vec![2.0, 1.0]);
        report.add_result(
            r2,
            RegionVerdict::Refuted {
                counterexample: vec![1.5, 0.5],
            },
        );

        // Unknown region (volume = 1).
        let r3 = VerificationRegion::new(vec![2.0, 0.0], vec![3.0, 1.0]);
        report.add_result(
            r3,
            RegionVerdict::Unknown {
                reason: "inconclusive".into(),
            },
        );

        // Timeout region (volume = 2).
        let r4 = VerificationRegion::new(vec![3.0, 0.0], vec![5.0, 1.0]);
        report.add_result(r4, RegionVerdict::Timeout);

        assert_eq!(report.region_count(), 4);
        assert_eq!(report.verified_count(), 1);
        assert_eq!(report.refuted_count(), 1);
        assert!((report.total_volume - 5.0).abs() < 1e-12);
        assert!((report.verified_volume - 1.0).abs() < 1e-12);
        assert!((report.refuted_volume - 1.0).abs() < 1e-12);
        assert!((report.unknown_volume - 3.0).abs() < 1e-12);
        assert!((report.coverage() - 0.2).abs() < 1e-12);

        let summary = report.summary();
        assert!(summary.contains("4 regions"));
        assert!(summary.contains("20.00%"));
    }

    // ---- build_reachability_query (structural) ----------------------------

    #[test]
    fn test_build_reachability_query() {
        let verifier = SmtVerifier::new();

        let chain = KinematicChain::default_arm(xr_types::ArmSide::Right);
        let params = BodyParameters::average_male();
        let ref_config = chain.midpoint_config(&params);

        let model = verifier
            .linearization_engine
            .linearize(&chain, &params, &ref_config);

        let region = VerificationRegion::new(
            vec![1.50, 0.28, 0.35, 0.22, 0.15],
            vec![1.90, 0.38, 0.48, 0.30, 0.22],
        );
        let target = [0.5, 0.3, 1.2];

        let (decls, assertions) = verifier
            .build_reachability_query(&model, &target, &region)
            .expect("query construction should succeed");

        // At least 5 body-param declarations.
        let body_decl_count = decls.iter().filter(|d| d.name.starts_with("p_")).count();
        assert!(body_decl_count >= BODY_PARAM_DIM);

        // Body-param bound assertions: 2 per dim = 10
        // Endpoint tolerance: 2 per axis = 6
        // Plus model assertions (≥ 0).
        assert!(assertions.len() >= 10 + 6);
    }

    // ---- SmtVerifier builder ----------------------------------------------

    #[test]
    fn test_smtverifier_construction() {
        let v = SmtVerifier::new();
        assert!((v.timeout_secs - DEFAULT_TIMEOUT_SECS).abs() < 1e-12);
        assert_eq!(v.max_depth, DEFAULT_MAX_DEPTH);
        assert!((v.min_region_volume - DEFAULT_MIN_REGION_VOL).abs() < 1e-20);
        assert!((v.tolerance - DEFAULT_TOLERANCE).abs() < 1e-12);

        let v2 = SmtVerifier::new()
            .with_timeout(30.0)
            .with_max_depth(12)
            .with_min_region_volume(1e-15);

        assert!((v2.timeout_secs - 30.0).abs() < 1e-12);
        assert_eq!(v2.max_depth, 12);
        assert!((v2.min_region_volume - 1e-15).abs() < 1e-20);
    }
}

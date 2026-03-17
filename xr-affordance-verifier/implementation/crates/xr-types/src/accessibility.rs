//! Accessibility predicate types per Definition D4.
//!
//! Provides types for per-element accessibility verdicts, population
//! accessibility tracking, and accessibility frontiers in the
//! body-parameter space.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::kinematic::BodyParameters;
use crate::scene::InteractionType;
use crate::traits::Verdict;
use crate::{ElementId, NUM_BODY_PARAMS};

// ---------------------------------------------------------------------------
// AccessibilityVerdict
// ---------------------------------------------------------------------------

/// Accessibility verdict for a single (element, body-params, device) triple.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AccessibilityVerdict {
    /// The element is accessible: a witness joint configuration exists.
    Accessible {
        /// Joint-angle witness demonstrating reachability.
        witness_joints: Vec<f64>,
        /// End-effector position achieved.
        end_effector_position: [f64; 3],
        /// Residual distance to the element interaction point (m).
        residual_distance: f64,
    },
    /// The element is inaccessible for these body parameters.
    Inaccessible {
        /// Human-readable reason.
        reason: String,
        /// The closest approach distance (m).
        closest_distance: f64,
        /// The closest end-effector position achieved.
        closest_position: Option<[f64; 3]>,
    },
    /// The check could not determine the result within resource bounds.
    Unknown {
        /// Explanation.
        reason: String,
    },
}

impl AccessibilityVerdict {
    /// Whether the element is accessible.
    pub fn is_accessible(&self) -> bool {
        matches!(self, AccessibilityVerdict::Accessible { .. })
    }

    /// Whether the element is inaccessible.
    pub fn is_inaccessible(&self) -> bool {
        matches!(self, AccessibilityVerdict::Inaccessible { .. })
    }

    /// Whether the check was inconclusive.
    pub fn is_unknown(&self) -> bool {
        matches!(self, AccessibilityVerdict::Unknown { .. })
    }

    /// Convert to a generic [`Verdict`].
    pub fn to_verdict(&self) -> Verdict {
        match self {
            AccessibilityVerdict::Accessible { .. } => Verdict::Pass,
            AccessibilityVerdict::Inaccessible { reason, .. } => Verdict::Fail {
                reason: reason.clone(),
                witness: None,
            },
            AccessibilityVerdict::Unknown { reason } => Verdict::Unknown {
                reason: reason.clone(),
            },
        }
    }

    /// Extract the residual distance (0 for unknown).
    pub fn distance(&self) -> f64 {
        match self {
            AccessibilityVerdict::Accessible {
                residual_distance, ..
            } => *residual_distance,
            AccessibilityVerdict::Inaccessible {
                closest_distance, ..
            } => *closest_distance,
            AccessibilityVerdict::Unknown { .. } => f64::NAN,
        }
    }
}

impl std::fmt::Display for AccessibilityVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AccessibilityVerdict::Accessible {
                residual_distance, ..
            } => write!(f, "ACCESSIBLE (residual {residual_distance:.4}m)"),
            AccessibilityVerdict::Inaccessible {
                reason,
                closest_distance,
                ..
            } => write!(f, "INACCESSIBLE: {reason} (closest {closest_distance:.4}m)"),
            AccessibilityVerdict::Unknown { reason } => write!(f, "UNKNOWN: {reason}"),
        }
    }
}

// ---------------------------------------------------------------------------
// AccessibilityResult
// ---------------------------------------------------------------------------

/// Aggregated accessibility result for one element across many samples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessibilityResult {
    /// Element being checked.
    pub element_id: ElementId,
    /// Element name.
    pub element_name: String,
    /// Interaction type for this element.
    pub interaction_type: InteractionType,
    /// Device name used.
    pub device_name: String,
    /// Per-sample verdicts (body params → verdict).
    pub verdicts: Vec<SampleAccessibility>,
    /// Summary statistics.
    pub stats: AccessibilityStats,
}

/// A single sample's accessibility check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleAccessibility {
    /// Body parameters used.
    pub body_params: BodyParameters,
    /// Percentiles for these body params.
    pub percentiles: [f64; NUM_BODY_PARAMS],
    /// The verdict.
    pub verdict: AccessibilityVerdict,
    /// Computation time (seconds).
    pub time_s: f64,
}

/// Summary statistics for a set of accessibility checks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessibilityStats {
    /// Total number of checks.
    pub total: usize,
    /// Number accessible.
    pub accessible: usize,
    /// Number inaccessible.
    pub inaccessible: usize,
    /// Number unknown.
    pub unknown: usize,
    /// Coverage ratio (accessible / total).
    pub coverage: f64,
    /// Mean distance for accessible samples.
    pub mean_accessible_distance: f64,
    /// Mean distance for inaccessible samples.
    pub mean_inaccessible_distance: f64,
    /// Total computation time (seconds).
    pub total_time_s: f64,
}

impl AccessibilityStats {
    /// Compute stats from a slice of sample verdicts.
    pub fn from_samples(samples: &[SampleAccessibility]) -> Self {
        let total = samples.len();
        let accessible = samples
            .iter()
            .filter(|s| s.verdict.is_accessible())
            .count();
        let inaccessible = samples
            .iter()
            .filter(|s| s.verdict.is_inaccessible())
            .count();
        let unknown = samples.iter().filter(|s| s.verdict.is_unknown()).count();

        let accessible_dists: Vec<f64> = samples
            .iter()
            .filter_map(|s| {
                if let AccessibilityVerdict::Accessible {
                    residual_distance, ..
                } = &s.verdict
                {
                    Some(*residual_distance)
                } else {
                    None
                }
            })
            .collect();

        let inaccessible_dists: Vec<f64> = samples
            .iter()
            .filter_map(|s| {
                if let AccessibilityVerdict::Inaccessible {
                    closest_distance, ..
                } = &s.verdict
                {
                    Some(*closest_distance)
                } else {
                    None
                }
            })
            .collect();

        let mean_acc = if accessible_dists.is_empty() {
            0.0
        } else {
            accessible_dists.iter().sum::<f64>() / accessible_dists.len() as f64
        };

        let mean_inacc = if inaccessible_dists.is_empty() {
            0.0
        } else {
            inaccessible_dists.iter().sum::<f64>() / inaccessible_dists.len() as f64
        };

        let total_time = samples.iter().map(|s| s.time_s).sum();

        Self {
            total,
            accessible,
            inaccessible,
            unknown,
            coverage: if total > 0 {
                accessible as f64 / total as f64
            } else {
                0.0
            },
            mean_accessible_distance: mean_acc,
            mean_inaccessible_distance: mean_inacc,
            total_time_s: total_time,
        }
    }
}

impl std::fmt::Display for AccessibilityStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}/{} accessible ({:.1}%), {} inaccessible, {} unknown, {:.2}s",
            self.accessible,
            self.total,
            self.coverage * 100.0,
            self.inaccessible,
            self.unknown,
            self.total_time_s,
        )
    }
}

impl AccessibilityResult {
    /// Create a new result, computing statistics from verdicts.
    pub fn new(
        element_id: ElementId,
        element_name: impl Into<String>,
        interaction_type: InteractionType,
        device_name: impl Into<String>,
        verdicts: Vec<SampleAccessibility>,
    ) -> Self {
        let stats = AccessibilityStats::from_samples(&verdicts);
        Self {
            element_id,
            element_name: element_name.into(),
            interaction_type,
            device_name: device_name.into(),
            verdicts,
            stats,
        }
    }

    /// Whether the element meets the coverage target.
    pub fn meets_target(&self, target_coverage: f64) -> bool {
        self.stats.coverage >= target_coverage
    }

    /// Get the body parameters that failed.
    pub fn failure_params(&self) -> Vec<&BodyParameters> {
        self.verdicts
            .iter()
            .filter(|s| s.verdict.is_inaccessible())
            .map(|s| &s.body_params)
            .collect()
    }

    /// Get the body parameters that succeeded.
    pub fn success_params(&self) -> Vec<&BodyParameters> {
        self.verdicts
            .iter()
            .filter(|s| s.verdict.is_accessible())
            .map(|s| &s.body_params)
            .collect()
    }
}

impl std::fmt::Display for AccessibilityResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} ({:?}): {}",
            self.element_name, self.interaction_type, self.stats
        )
    }
}

// ---------------------------------------------------------------------------
// PopulationAccessibility
// ---------------------------------------------------------------------------

/// Population-level accessibility tracking across elements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationAccessibility {
    /// Per-element results.
    pub element_results: Vec<AccessibilityResult>,
    /// Overall coverage (min across elements).
    pub overall_coverage: f64,
    /// Elements that fail the target.
    pub failing_elements: Vec<ElementId>,
    /// Percentile range used.
    pub percentile_range: (f64, f64),
    /// Device name.
    pub device_name: String,
}

impl PopulationAccessibility {
    /// Build population accessibility from element results.
    pub fn from_results(
        results: Vec<AccessibilityResult>,
        target_coverage: f64,
        percentile_range: (f64, f64),
        device_name: impl Into<String>,
    ) -> Self {
        let overall = results
            .iter()
            .map(|r| r.stats.coverage)
            .fold(f64::INFINITY, f64::min)
            .min(1.0);

        let failing: Vec<ElementId> = results
            .iter()
            .filter(|r| !r.meets_target(target_coverage))
            .map(|r| r.element_id)
            .collect();

        Self {
            element_results: results,
            overall_coverage: if overall == f64::INFINITY {
                0.0
            } else {
                overall
            },
            failing_elements: failing,
            percentile_range,
            device_name: device_name.into(),
        }
    }

    /// Whether all elements meet the target coverage.
    pub fn all_pass(&self) -> bool {
        self.failing_elements.is_empty()
    }

    /// Number of elements checked.
    pub fn num_elements(&self) -> usize {
        self.element_results.len()
    }

    /// Total number of sample checks performed.
    pub fn total_checks(&self) -> usize {
        self.element_results.iter().map(|r| r.stats.total).sum()
    }

    /// Get result for a specific element.
    pub fn result_for(&self, element_id: ElementId) -> Option<&AccessibilityResult> {
        self.element_results
            .iter()
            .find(|r| r.element_id == element_id)
    }
}

impl std::fmt::Display for PopulationAccessibility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Population accessibility ({}, p={:.0}%-{:.0}%):",
            self.device_name,
            self.percentile_range.0 * 100.0,
            self.percentile_range.1 * 100.0,
        )?;
        for r in &self.element_results {
            writeln!(f, "  {r}")?;
        }
        writeln!(f, "  Overall coverage: {:.1}%", self.overall_coverage * 100.0)?;
        if !self.failing_elements.is_empty() {
            writeln!(f, "  Failing elements: {}", self.failing_elements.len())?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// AccessibilityFrontier
// ---------------------------------------------------------------------------

/// The boundary in parameter space between accessible and inaccessible regions.
///
/// Represented as a set of boundary sample points and their local normals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessibilityFrontier {
    /// Element this frontier pertains to.
    pub element_id: ElementId,
    /// Device name.
    pub device_name: String,
    /// Boundary points in parameter space (each is `NUM_BODY_PARAMS`-dimensional).
    pub boundary_points: Vec<FrontierPoint>,
    /// Bounding box of the frontier in parameter space.
    pub parameter_bounds: Option<(Vec<f64>, Vec<f64>)>,
    /// Estimated measure of the inaccessible region.
    pub inaccessible_volume: f64,
    /// Estimated measure of the accessible region.
    pub accessible_volume: f64,
}

/// A single point on the accessibility frontier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrontierPoint {
    /// Parameter-space coordinates.
    pub params: Vec<f64>,
    /// Approximate outward normal (pointing towards the inaccessible side).
    pub normal: Vec<f64>,
    /// The verdict at this point (should be borderline accessible).
    pub verdict: AccessibilityVerdict,
    /// Gradient of the distance function at this point.
    pub distance_gradient: Option<Vec<f64>>,
}

impl AccessibilityFrontier {
    /// Create a new empty frontier.
    pub fn new(element_id: ElementId, device_name: impl Into<String>) -> Self {
        Self {
            element_id,
            device_name: device_name.into(),
            boundary_points: Vec::new(),
            parameter_bounds: None,
            inaccessible_volume: 0.0,
            accessible_volume: 0.0,
        }
    }

    /// Add a boundary point.
    pub fn add_point(&mut self, point: FrontierPoint) {
        self.boundary_points.push(point);
    }

    /// Number of boundary points.
    pub fn num_points(&self) -> usize {
        self.boundary_points.len()
    }

    /// Compute the frontier from a set of accessible and inaccessible samples.
    ///
    /// Identifies pairs of samples that straddle the boundary (one accessible,
    /// one inaccessible) and returns approximate boundary points.
    pub fn from_samples(
        element_id: ElementId,
        device_name: impl Into<String>,
        accessible: &[(Vec<f64>, AccessibilityVerdict)],
        inaccessible: &[(Vec<f64>, AccessibilityVerdict)],
    ) -> Self {
        let device_name = device_name.into();
        let mut frontier = Self::new(element_id, &device_name);

        // For each inaccessible sample, find the closest accessible sample.
        for (inacc_params, inacc_verdict) in inaccessible {
            let mut best_dist = f64::INFINITY;
            let mut best_acc: Option<&Vec<f64>> = None;

            for (acc_params, _) in accessible {
                let dist: f64 = inacc_params
                    .iter()
                    .zip(acc_params.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if dist < best_dist {
                    best_dist = dist;
                    best_acc = Some(acc_params);
                }
            }

            if let Some(acc) = best_acc {
                // Midpoint between closest accessible and this inaccessible sample.
                let midpoint: Vec<f64> = inacc_params
                    .iter()
                    .zip(acc.iter())
                    .map(|(a, b)| (a + b) * 0.5)
                    .collect();

                // Approximate normal: direction from accessible to inaccessible.
                let normal: Vec<f64> = inacc_params
                    .iter()
                    .zip(acc.iter())
                    .map(|(i, a)| i - a)
                    .collect();
                let norm_len: f64 = normal.iter().map(|x| x * x).sum::<f64>().sqrt();
                let normal = if norm_len > 1e-12 {
                    normal.iter().map(|x| x / norm_len).collect()
                } else {
                    vec![0.0; inacc_params.len()]
                };

                frontier.add_point(FrontierPoint {
                    params: midpoint,
                    normal,
                    verdict: inacc_verdict.clone(),
                    distance_gradient: None,
                });
            }
        }

        frontier
    }

    /// Estimate the fraction of parameter volume that is accessible.
    pub fn accessible_fraction(&self) -> f64 {
        let total = self.accessible_volume + self.inaccessible_volume;
        if total <= 0.0 {
            0.0
        } else {
            self.accessible_volume / total
        }
    }
}

impl std::fmt::Display for AccessibilityFrontier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Frontier for {} ({}): {} boundary points, accessible fraction: {:.2}%",
            self.element_id,
            self.device_name,
            self.boundary_points.len(),
            self.accessible_fraction() * 100.0,
        )
    }
}

// ---------------------------------------------------------------------------
// Multi-step accessibility
// ---------------------------------------------------------------------------

/// Result of a multi-step accessibility check through an interaction FSM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiStepAccessibilityResult {
    /// Element being checked.
    pub element_id: ElementId,
    /// FSM used.
    pub fsm_id: Uuid,
    /// Number of steps in the required interaction.
    pub num_steps: usize,
    /// Per-step verdicts.
    pub step_verdicts: Vec<StepVerdict>,
    /// Overall verdict.
    pub overall: AccessibilityVerdict,
    /// Body parameters used.
    pub body_params: BodyParameters,
}

/// Verdict for a single step in a multi-step interaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepVerdict {
    /// Step index (0-based).
    pub step: usize,
    /// State ID at this step.
    pub state_id: String,
    /// Transition taken.
    pub transition_id: Option<String>,
    /// Accessibility verdict for this step.
    pub verdict: AccessibilityVerdict,
    /// Joint configuration at this step.
    pub joint_config: Vec<f64>,
}

impl MultiStepAccessibilityResult {
    /// Create a result for a successful multi-step check.
    pub fn accessible(
        element_id: ElementId,
        fsm_id: Uuid,
        body_params: BodyParameters,
        steps: Vec<StepVerdict>,
    ) -> Self {
        let last = steps.last().and_then(|s| {
            if s.verdict.is_accessible() {
                if let AccessibilityVerdict::Accessible {
                    witness_joints,
                    end_effector_position,
                    residual_distance,
                } = &s.verdict
                {
                    Some(AccessibilityVerdict::Accessible {
                        witness_joints: witness_joints.clone(),
                        end_effector_position: *end_effector_position,
                        residual_distance: *residual_distance,
                    })
                } else {
                    None
                }
            } else {
                None
            }
        });

        let overall = last.unwrap_or_else(|| AccessibilityVerdict::Inaccessible {
            reason: "not all steps succeeded".into(),
            closest_distance: f64::INFINITY,
            closest_position: None,
        });

        Self {
            element_id,
            fsm_id,
            num_steps: steps.len(),
            step_verdicts: steps,
            overall,
            body_params,
        }
    }

    /// Create a result for a failed multi-step check.
    pub fn inaccessible(
        element_id: ElementId,
        fsm_id: Uuid,
        body_params: BodyParameters,
        failing_step: usize,
        reason: String,
        closest_distance: f64,
    ) -> Self {
        Self {
            element_id,
            fsm_id,
            num_steps: failing_step + 1,
            step_verdicts: Vec::new(),
            overall: AccessibilityVerdict::Inaccessible {
                reason,
                closest_distance,
                closest_position: None,
            },
            body_params,
        }
    }

    /// Whether the overall result is accessible.
    pub fn is_accessible(&self) -> bool {
        self.overall.is_accessible()
    }

    /// The step at which the check failed (if any).
    pub fn failing_step(&self) -> Option<usize> {
        self.step_verdicts
            .iter()
            .find(|s| s.verdict.is_inaccessible())
            .map(|s| s.step)
    }
}

impl std::fmt::Display for MultiStepAccessibilityResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Multi-step ({} steps): {}",
            self.num_steps, self.overall
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_eid() -> ElementId {
        Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap()
    }

    #[test]
    fn test_verdict_accessible() {
        let v = AccessibilityVerdict::Accessible {
            witness_joints: vec![0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0],
            end_effector_position: [0.5, 1.0, 0.3],
            residual_distance: 0.002,
        };
        assert!(v.is_accessible());
        assert!(!v.is_inaccessible());
        assert!((v.distance() - 0.002).abs() < 1e-12);
    }

    #[test]
    fn test_verdict_inaccessible() {
        let v = AccessibilityVerdict::Inaccessible {
            reason: "out of reach".into(),
            closest_distance: 0.15,
            closest_position: Some([0.4, 0.9, 0.3]),
        };
        assert!(v.is_inaccessible());
        assert!(!v.is_accessible());
        assert!((v.distance() - 0.15).abs() < 1e-12);
    }

    #[test]
    fn test_verdict_unknown() {
        let v = AccessibilityVerdict::Unknown {
            reason: "timeout".into(),
        };
        assert!(v.is_unknown());
        assert!(v.distance().is_nan());
    }

    #[test]
    fn test_verdict_to_generic() {
        let v = AccessibilityVerdict::Accessible {
            witness_joints: vec![],
            end_effector_position: [0.0; 3],
            residual_distance: 0.0,
        };
        assert!(v.to_verdict().is_pass());

        let v2 = AccessibilityVerdict::Inaccessible {
            reason: "x".into(),
            closest_distance: 1.0,
            closest_position: None,
        };
        assert!(v2.to_verdict().is_fail());
    }

    #[test]
    fn test_verdict_display() {
        let v = AccessibilityVerdict::Accessible {
            witness_joints: vec![],
            end_effector_position: [0.5, 1.0, 0.3],
            residual_distance: 0.001,
        };
        let s = format!("{v}");
        assert!(s.contains("ACCESSIBLE"));
    }

    #[test]
    fn test_stats_from_samples() {
        let samples = vec![
            SampleAccessibility {
                body_params: BodyParameters::average_male(),
                percentiles: [0.5; 5],
                verdict: AccessibilityVerdict::Accessible {
                    witness_joints: vec![],
                    end_effector_position: [0.0; 3],
                    residual_distance: 0.01,
                },
                time_s: 0.1,
            },
            SampleAccessibility {
                body_params: BodyParameters::small_female(),
                percentiles: [0.05; 5],
                verdict: AccessibilityVerdict::Inaccessible {
                    reason: "out of reach".into(),
                    closest_distance: 0.1,
                    closest_position: None,
                },
                time_s: 0.2,
            },
        ];

        let stats = AccessibilityStats::from_samples(&samples);
        assert_eq!(stats.total, 2);
        assert_eq!(stats.accessible, 1);
        assert_eq!(stats.inaccessible, 1);
        assert!((stats.coverage - 0.5).abs() < 1e-12);
        assert!((stats.total_time_s - 0.3).abs() < 1e-12);
    }

    #[test]
    fn test_stats_display() {
        let stats = AccessibilityStats {
            total: 100,
            accessible: 90,
            inaccessible: 8,
            unknown: 2,
            coverage: 0.90,
            mean_accessible_distance: 0.01,
            mean_inaccessible_distance: 0.15,
            total_time_s: 5.0,
        };
        let s = format!("{stats}");
        assert!(s.contains("90/100"));
        assert!(s.contains("90.0%"));
    }

    #[test]
    fn test_accessibility_result() {
        let samples = vec![
            SampleAccessibility {
                body_params: BodyParameters::average_male(),
                percentiles: [0.5; 5],
                verdict: AccessibilityVerdict::Accessible {
                    witness_joints: vec![],
                    end_effector_position: [0.0; 3],
                    residual_distance: 0.01,
                },
                time_s: 0.1,
            },
        ];

        let result = AccessibilityResult::new(
            test_eid(),
            "Button A",
            InteractionType::Click,
            "Quest 3",
            samples,
        );
        assert!(result.meets_target(0.5));
        assert_eq!(result.failure_params().len(), 0);
        assert_eq!(result.success_params().len(), 1);
    }

    #[test]
    fn test_population_accessibility() {
        let r1 = AccessibilityResult::new(
            test_eid(),
            "Button A",
            InteractionType::Click,
            "Quest 3",
            vec![SampleAccessibility {
                body_params: BodyParameters::average_male(),
                percentiles: [0.5; 5],
                verdict: AccessibilityVerdict::Accessible {
                    witness_joints: vec![],
                    end_effector_position: [0.0; 3],
                    residual_distance: 0.01,
                },
                time_s: 0.1,
            }],
        );

        let pop = PopulationAccessibility::from_results(
            vec![r1],
            0.90,
            (0.05, 0.95),
            "Quest 3",
        );
        assert_eq!(pop.num_elements(), 1);
        assert!(pop.all_pass());
        assert!((pop.overall_coverage - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_population_failing_elements() {
        let samples: Vec<SampleAccessibility> = (0..10)
            .map(|i| SampleAccessibility {
                body_params: BodyParameters::average_male(),
                percentiles: [0.5; 5],
                verdict: if i < 5 {
                    AccessibilityVerdict::Accessible {
                        witness_joints: vec![],
                        end_effector_position: [0.0; 3],
                        residual_distance: 0.01,
                    }
                } else {
                    AccessibilityVerdict::Inaccessible {
                        reason: "x".into(),
                        closest_distance: 0.1,
                        closest_position: None,
                    }
                },
                time_s: 0.05,
            })
            .collect();

        let r = AccessibilityResult::new(
            test_eid(),
            "Slider",
            InteractionType::Drag,
            "Quest 3",
            samples,
        );

        let pop = PopulationAccessibility::from_results(
            vec![r],
            0.90,
            (0.05, 0.95),
            "Quest 3",
        );
        assert!(!pop.all_pass());
        assert_eq!(pop.failing_elements.len(), 1);
    }

    #[test]
    fn test_frontier_from_samples() {
        let accessible = vec![
            (vec![1.7, 0.36, 0.48, 0.27, 0.19], AccessibilityVerdict::Accessible {
                witness_joints: vec![],
                end_effector_position: [0.0; 3],
                residual_distance: 0.0,
            }),
        ];
        let inaccessible = vec![
            (vec![1.5, 0.30, 0.38, 0.22, 0.16], AccessibilityVerdict::Inaccessible {
                reason: "x".into(),
                closest_distance: 0.1,
                closest_position: None,
            }),
        ];

        let frontier = AccessibilityFrontier::from_samples(
            test_eid(),
            "Quest 3",
            &accessible,
            &inaccessible,
        );
        assert_eq!(frontier.num_points(), 1);
        // Midpoint should be between the two samples.
        let mid = &frontier.boundary_points[0].params;
        assert!((mid[0] - 1.6).abs() < 0.01);
    }

    #[test]
    fn test_frontier_display() {
        let frontier = AccessibilityFrontier::new(test_eid(), "Quest 3");
        let s = format!("{frontier}");
        assert!(s.contains("Frontier"));
    }

    #[test]
    fn test_frontier_accessible_fraction() {
        let mut frontier = AccessibilityFrontier::new(test_eid(), "Quest 3");
        frontier.accessible_volume = 0.8;
        frontier.inaccessible_volume = 0.2;
        assert!((frontier.accessible_fraction() - 0.8).abs() < 1e-12);
    }

    #[test]
    fn test_multi_step_accessible() {
        let steps = vec![
            StepVerdict {
                step: 0,
                state_id: "idle".into(),
                transition_id: Some("t0".into()),
                verdict: AccessibilityVerdict::Accessible {
                    witness_joints: vec![0.1; 7],
                    end_effector_position: [0.5, 1.0, 0.3],
                    residual_distance: 0.01,
                },
                joint_config: vec![0.1; 7],
            },
            StepVerdict {
                step: 1,
                state_id: "done".into(),
                transition_id: None,
                verdict: AccessibilityVerdict::Accessible {
                    witness_joints: vec![0.2; 7],
                    end_effector_position: [0.5, 1.0, 0.3],
                    residual_distance: 0.002,
                },
                joint_config: vec![0.2; 7],
            },
        ];

        let result = MultiStepAccessibilityResult::accessible(
            test_eid(),
            Uuid::new_v4(),
            BodyParameters::average_male(),
            steps,
        );
        assert!(result.is_accessible());
        assert_eq!(result.num_steps, 2);
    }

    #[test]
    fn test_multi_step_inaccessible() {
        let result = MultiStepAccessibilityResult::inaccessible(
            test_eid(),
            Uuid::new_v4(),
            BodyParameters::small_female(),
            1,
            "arm too short".into(),
            0.15,
        );
        assert!(!result.is_accessible());
        assert_eq!(result.failing_step(), None); // no step_verdicts
    }

    #[test]
    fn test_multi_step_display() {
        let result = MultiStepAccessibilityResult::inaccessible(
            test_eid(),
            Uuid::new_v4(),
            BodyParameters::small_female(),
            1,
            "arm too short".into(),
            0.15,
        );
        let s = format!("{result}");
        assert!(s.contains("Multi-step"));
    }

    #[test]
    fn test_population_display() {
        let pop = PopulationAccessibility::from_results(
            vec![],
            0.90,
            (0.05, 0.95),
            "Quest 3",
        );
        let s = format!("{pop}");
        assert!(s.contains("Population"));
    }

    #[test]
    fn test_serde_roundtrip_verdict() {
        let v = AccessibilityVerdict::Accessible {
            witness_joints: vec![0.1, 0.2],
            end_effector_position: [0.5, 1.0, 0.3],
            residual_distance: 0.01,
        };
        let json = serde_json::to_string(&v).unwrap();
        let back: AccessibilityVerdict = serde_json::from_str(&json).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn test_result_for_element() {
        let r = AccessibilityResult::new(
            test_eid(),
            "Button A",
            InteractionType::Click,
            "Quest 3",
            vec![],
        );
        let pop = PopulationAccessibility::from_results(
            vec![r],
            0.90,
            (0.05, 0.95),
            "Quest 3",
        );
        assert!(pop.result_for(test_eid()).is_some());
        assert!(pop.result_for(Uuid::new_v4()).is_none());
    }
}

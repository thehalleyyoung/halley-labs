//! Comfort and ergonomic scoring for pose assessment.
//!
//! Implements a RULA-inspired (Rapid Upper Limb Assessment) comfort scoring
//! system that evaluates arm configurations based on joint angles, reach
//! fraction, and sustained-hold duration.

use xr_types::kinematic::{KinematicChain, BodyParameters, ArmSide};
use xr_types::geometry::{BoundingBox, point_distance};
use xr_types::error::{VerifierError, VerifierResult};
use xr_types::deg_to_rad;

use crate::forward_kinematics::ForwardKinematicsSolver;
use crate::reach_envelope::VoxelGrid;

use rand::Rng;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the comfort model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComfortConfig {
    /// Weight for shoulder contribution in total score.
    pub shoulder_weight: f64,
    /// Weight for elbow contribution in total score.
    pub elbow_weight: f64,
    /// Weight for wrist contribution in total score.
    pub wrist_weight: f64,
    /// Weight for reach-fraction penalty.
    pub reach_weight: f64,
    /// Number of Monte Carlo samples for comfort map.
    pub map_samples: usize,
    /// Voxel size for comfort map grid.
    pub map_voxel_size: f64,
    /// Fatigue time constant (seconds): time at which fatigue penalty = 0.5.
    pub fatigue_half_life: f64,
}

impl Default for ComfortConfig {
    fn default() -> Self {
        Self {
            shoulder_weight: 0.30,
            elbow_weight: 0.30,
            wrist_weight: 0.20,
            reach_weight: 0.20,
            map_samples: 10_000,
            map_voxel_size: 0.03,
            fatigue_half_life: 30.0,
        }
    }
}

// ---------------------------------------------------------------------------
// ComfortScore
// ---------------------------------------------------------------------------

/// Comfort score for a single arm configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComfortScore {
    /// Total comfort score [0,1], 1 = most comfortable.
    pub total_score: f64,
    /// Shoulder comfort sub-score [0,1].
    pub shoulder_score: f64,
    /// Elbow comfort sub-score [0,1].
    pub elbow_score: f64,
    /// Wrist comfort sub-score [0,1].
    pub wrist_score: f64,
    /// Fatigue factor [0,1], 1 = no fatigue.
    pub fatigue_factor: f64,
}

impl ComfortScore {
    fn combine(shoulder: f64, elbow: f64, wrist: f64, fatigue: f64, cfg: &ComfortConfig) -> Self {
        let raw = cfg.shoulder_weight * shoulder
            + cfg.elbow_weight * elbow
            + cfg.wrist_weight * wrist;
        // Normalise so max possible = 1.0
        let weight_sum = cfg.shoulder_weight + cfg.elbow_weight + cfg.wrist_weight;
        let normalised = if weight_sum > 1e-12 { raw / weight_sum } else { 0.0 };
        let total = (normalised * fatigue).clamp(0.0, 1.0);
        Self {
            total_score: total,
            shoulder_score: shoulder,
            elbow_score: elbow,
            wrist_score: wrist,
            fatigue_factor: fatigue,
        }
    }
}

// ---------------------------------------------------------------------------
// ComfortModel
// ---------------------------------------------------------------------------

/// Comfort model for ergonomic pose assessment.
pub struct ComfortModel {
    config: ComfortConfig,
    fk: ForwardKinematicsSolver,
}

impl ComfortModel {
    pub fn new() -> Self {
        Self {
            config: ComfortConfig::default(),
            fk: ForwardKinematicsSolver::new(),
        }
    }

    pub fn with_config(mut self, config: ComfortConfig) -> Self {
        self.config = config;
        self
    }

    // -- Core evaluation --

    /// Evaluate comfort for a given joint configuration.
    ///
    /// The 7-DOF arm joint order is:
    ///   [0] shoulder flexion   (X)
    ///   [1] shoulder abduction (Z)
    ///   [2] shoulder rotation  (Y)
    ///   [3] elbow flexion      (X)
    ///   [4] wrist pronation    (Y)
    ///   [5] wrist flexion      (X)
    ///   [6] wrist deviation    (Z)
    pub fn evaluate(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        angles: &[f64],
    ) -> VerifierResult<ComfortScore> {
        if angles.len() < 7 {
            return Err(VerifierError::Configuration(
                "Need at least 7 joint angles".to_string(),
            ));
        }

        let shoulder = self.evaluate_shoulder(angles[0], angles[1], angles[2]);
        let elbow = self.evaluate_elbow(angles[3]);
        let wrist = self.evaluate_wrist(angles[4], angles[5], angles[6]);

        // Reach-fraction penalty
        if let Ok(pos) = self.fk.solve_position(chain, params, angles) {
            let base = chain.base_position(params);
            let dist = point_distance(&pos, &base);
            let max_reach = ForwardKinematicsSolver::max_reach(chain, params);
            let frac = dist / max_reach.max(1e-6);
            let reach_penalty = reach_comfort(frac);

            // Blend in the reach penalty
            let blended_shoulder = shoulder * (1.0 - self.config.reach_weight)
                + reach_penalty * self.config.reach_weight;
            return Ok(ComfortScore::combine(
                blended_shoulder,
                elbow,
                wrist,
                1.0,
                &self.config,
            ));
        }

        Ok(ComfortScore::combine(shoulder, elbow, wrist, 1.0, &self.config))
    }

    // -- Per-joint evaluation (RULA-inspired) --

    /// Shoulder comfort: penalise high elevation, abduction, and extreme rotation.
    pub fn evaluate_shoulder(
        &self,
        flexion: f64,   // joint 0
        abduction: f64,  // joint 1
        rotation: f64,   // joint 2
    ) -> f64 {
        // RULA shoulder scoring (simplified):
        //   0-20° flexion → score 1  (best)
        //  20-45° → 2
        //  45-90° → 3
        //  >90°   → 4 (worst)
        // We invert to [0,1] comfort.
        let flex_deg = flexion.to_degrees().abs();
        let flex_score = if flex_deg <= 20.0 {
            1.0
        } else if flex_deg <= 45.0 {
            1.0 - (flex_deg - 20.0) / 25.0 * 0.25
        } else if flex_deg <= 90.0 {
            0.75 - (flex_deg - 45.0) / 45.0 * 0.35
        } else {
            (0.4 - (flex_deg - 90.0) / 90.0 * 0.3).max(0.1)
        };

        // Abduction penalty: any abduction above ~20° is penalised
        let abd_deg = abduction.to_degrees().abs();
        let abd_penalty = if abd_deg <= 20.0 {
            1.0
        } else if abd_deg <= 60.0 {
            1.0 - (abd_deg - 20.0) / 40.0 * 0.4
        } else {
            (0.6 - (abd_deg - 60.0) / 120.0 * 0.4).max(0.2)
        };

        // Rotation penalty: extreme rotation is uncomfortable
        let rot_deg = rotation.to_degrees().abs();
        let rot_penalty = if rot_deg <= 30.0 {
            1.0
        } else if rot_deg <= 60.0 {
            1.0 - (rot_deg - 30.0) / 30.0 * 0.3
        } else {
            (0.7 - (rot_deg - 60.0) / 30.0 * 0.3).max(0.3)
        };

        (flex_score * abd_penalty * rot_penalty).clamp(0.0, 1.0)
    }

    /// Elbow comfort: penalise extreme flexion and full extension.
    pub fn evaluate_elbow(&self, flexion: f64) -> f64 {
        let deg = flexion.to_degrees();
        // Optimal: 80-100°. Full extension (0°) and extreme flexion (>130°) are bad.
        if deg < 0.0 {
            0.3 // hyperextension
        } else if deg <= 20.0 {
            0.5 + deg / 20.0 * 0.2
        } else if deg <= 60.0 {
            0.7 + (deg - 20.0) / 40.0 * 0.2
        } else if deg <= 100.0 {
            0.9 + (deg - 60.0) / 40.0 * 0.1
        } else if deg <= 130.0 {
            1.0 - (deg - 100.0) / 30.0 * 0.2
        } else {
            (0.8 - (deg - 130.0) / 15.0 * 0.3).max(0.3)
        }
    }

    /// Wrist comfort: penalise pronation, flexion, and ulnar/radial deviation.
    pub fn evaluate_wrist(
        &self,
        pronation: f64,  // joint 4
        flexion: f64,     // joint 5
        deviation: f64,   // joint 6
    ) -> f64 {
        // Wrist neutral = most comfortable
        let pron_deg = pronation.to_degrees().abs();
        let pron_score = if pron_deg <= 15.0 {
            1.0
        } else if pron_deg <= 45.0 {
            1.0 - (pron_deg - 15.0) / 30.0 * 0.3
        } else {
            (0.7 - (pron_deg - 45.0) / 35.0 * 0.3).max(0.3)
        };

        let flex_deg = flexion.to_degrees().abs();
        let flex_score = if flex_deg <= 15.0 {
            1.0
        } else if flex_deg <= 45.0 {
            1.0 - (flex_deg - 15.0) / 30.0 * 0.3
        } else {
            (0.7 - (flex_deg - 45.0) / 25.0 * 0.3).max(0.3)
        };

        // Ulnar/radial deviation — strong penalty
        let dev_deg = deviation.to_degrees().abs();
        let dev_score = if dev_deg <= 10.0 {
            1.0
        } else if dev_deg <= 20.0 {
            1.0 - (dev_deg - 10.0) / 10.0 * 0.3
        } else {
            (0.7 - (dev_deg - 20.0) / 10.0 * 0.3).max(0.3)
        };

        (pron_score * flex_score * dev_score).clamp(0.0, 1.0)
    }

    // -- Fatigue / sustained hold --

    /// Compute the fatigue penalty for holding a pose for `duration` seconds.
    /// Returns a factor in (0, 1] that multiplies the comfort score.
    pub fn sustained_hold_penalty(&self, duration: f64) -> f64 {
        if duration <= 0.0 {
            return 1.0;
        }
        let half_life = self.config.fatigue_half_life;
        // Exponential decay: factor = 2^(-t / half_life)
        (-(duration / half_life) * std::f64::consts::LN_2).exp()
    }

    // -- Comfort map --

    /// Compute a spatial comfort map by sampling random configurations and
    /// recording the best comfort score observed at each voxel.
    pub fn compute_comfort_map(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
    ) -> VerifierResult<VoxelGrid> {
        let base = chain.base_position(params);
        let max_r = ForwardKinematicsSolver::max_reach(chain, params);
        let vs = self.config.map_voxel_size;

        let bb = BoundingBox::new(
            [base[0] - max_r - 0.05, base[1] - max_r - 0.05, base[2] - max_r - 0.05],
            [base[0] + max_r + 0.05, base[1] + max_r + 0.05, base[2] + max_r + 0.05],
        );
        let mut grid = VoxelGrid::from_bounds(&bb, vs);
        let mut rng = rand::thread_rng();
        let n = self.config.map_samples;

        for _ in 0..n {
            let angles = chain.random_config(params, &mut rng);
            let score = match self.evaluate(chain, params, &angles) {
                Ok(s) => s.total_score,
                Err(_) => continue,
            };
            if let Ok(pos) = self.fk.solve_position(chain, params, &angles) {
                if !pos[0].is_finite() {
                    continue;
                }
                if let Some(idx) = grid.point_to_index(&pos) {
                    let fi = idx[0] * grid.dims[1] * grid.dims[2]
                        + idx[1] * grid.dims[2]
                        + idx[2];
                    grid.occupied[fi] = true;
                    // Store quantised score (0–1000)
                    let q = (score * 1000.0) as u32;
                    grid.hit_count[fi] = grid.hit_count[fi].max(q);
                }
            }
        }

        Ok(grid)
    }

    /// Identify the subset of the workspace that meets a comfort threshold.
    /// Returns the fraction of the reachable workspace that is "comfortable".
    pub fn identify_comfortable_subset(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        threshold: f64,
    ) -> VerifierResult<f64> {
        let grid = self.compute_comfort_map(chain, params)?;
        let occupied = grid.occupied.iter().filter(|&&b| b).count();
        if occupied == 0 {
            return Ok(0.0);
        }
        let comfortable = grid
            .occupied
            .iter()
            .zip(grid.hit_count.iter())
            .filter(|(&occ, &score)| occ && (score as f64 / 1000.0) >= threshold)
            .count();
        Ok(comfortable as f64 / occupied as f64)
    }

    // -- Legacy compatibility --

    /// Score comfort of reaching a point (position-only, no joint angles).
    pub fn score_reach(
        &self,
        params: &BodyParameters,
        shoulder_pos: [f64; 3],
        target: [f64; 3],
    ) -> f64 {
        let dx = target[0] - shoulder_pos[0];
        let dy = target[1] - shoulder_pos[1];
        let dz = target[2] - shoulder_pos[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        let max_reach = params.total_reach();
        let reach_frac = dist / max_reach;

        let comfort_reach = reach_comfort(reach_frac);
        let height_penalty = if dy > 0.3 {
            (1.0 - (dy - 0.3).min(0.5) / 0.5) * 0.8
        } else if dy < -0.5 {
            (1.0 - (-0.5 - dy).min(0.5) / 0.5) * 0.7
        } else {
            1.0
        };
        (comfort_reach * self.config.reach_weight
            + height_penalty * (1.0 - self.config.reach_weight))
            .clamp(0.0, 1.0)
    }

    /// Score comfort of a gaze angle (degrees from forward).
    pub fn score_gaze_angle(&self, angle_deg: f64) -> f64 {
        if angle_deg <= 15.0 {
            1.0
        } else if angle_deg <= 30.0 {
            1.0 - (angle_deg - 15.0) / 15.0 * 0.3
        } else if angle_deg <= 55.0 {
            0.7 - (angle_deg - 30.0) / 25.0 * 0.5
        } else {
            0.2_f64.max(1.0 - angle_deg / 90.0)
        }
    }

    /// Evaluate if a position is in the comfort zone.
    pub fn is_comfortable(
        &self,
        params: &BodyParameters,
        shoulder_pos: [f64; 3],
        target: [f64; 3],
    ) -> bool {
        self.score_reach(params, shoulder_pos, target) >= 0.6
    }
}

impl Default for ComfortModel {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Comfort as a function of reach fraction [0,1].
fn reach_comfort(frac: f64) -> f64 {
    if frac <= 0.6 {
        1.0
    } else if frac <= 0.85 {
        1.0 - (frac - 0.6) / 0.25 * 0.5
    } else if frac <= 1.0 {
        0.5 - (frac - 0.85) / 0.15 * 0.4
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use xr_types::kinematic::ArmSide;

    fn test_chain_and_params() -> (KinematicChain, BodyParameters) {
        (
            KinematicChain::default_arm(ArmSide::Right),
            BodyParameters::average_male(),
        )
    }

    #[test]
    fn test_comfort_scoring() {
        let model = ComfortModel::new();
        let params = BodyParameters::average_male();
        let shoulder = [0.0, 1.4, 0.0];
        let near = [0.0, 1.2, -0.3];
        let far = [0.0, 1.4, -0.8];
        assert!(
            model.score_reach(&params, shoulder, near)
                > model.score_reach(&params, shoulder, far)
        );
    }

    #[test]
    fn test_gaze_scoring() {
        let model = ComfortModel::new();
        assert!(model.score_gaze_angle(10.0) > model.score_gaze_angle(45.0));
    }

    #[test]
    fn test_evaluate_neutral_pose() {
        let (chain, params) = test_chain_and_params();
        let model = ComfortModel::new();
        // Roughly neutral: slight flexion, moderate elbow bend
        let angles = [
            deg_to_rad(10.0),  // shoulder flex
            deg_to_rad(5.0),   // shoulder abduction
            deg_to_rad(0.0),   // shoulder rotation
            deg_to_rad(90.0),  // elbow flex
            deg_to_rad(0.0),   // wrist pronation
            deg_to_rad(0.0),   // wrist flexion
            deg_to_rad(0.0),   // wrist deviation
        ];
        let score = model.evaluate(&chain, &params, &angles).unwrap();
        assert!(
            score.total_score > 0.5,
            "Neutral pose should be comfortable, got {}",
            score.total_score
        );
        assert!(score.shoulder_score > 0.5);
        assert!(score.elbow_score > 0.8);
        assert!(score.wrist_score > 0.8);
    }

    #[test]
    fn test_evaluate_extreme_pose() {
        let (chain, params) = test_chain_and_params();
        let model = ComfortModel::new();
        // Overhead reach with extreme wrist deviation
        let angles = [
            deg_to_rad(170.0), // full shoulder flex
            deg_to_rad(170.0), // extreme abduction
            deg_to_rad(80.0),  // extreme rotation
            deg_to_rad(5.0),   // near-extended elbow
            deg_to_rad(70.0),  // extreme pronation
            deg_to_rad(60.0),  // extreme wrist flex
            deg_to_rad(25.0),  // ulnar deviation
        ];
        let score = model.evaluate(&chain, &params, &angles).unwrap();
        assert!(
            score.total_score < 0.5,
            "Extreme pose should be uncomfortable, got {}",
            score.total_score,
        );
    }

    #[test]
    fn test_shoulder_scoring_monotonic() {
        let model = ComfortModel::new();
        let s0 = model.evaluate_shoulder(deg_to_rad(10.0), 0.0, 0.0);
        let s1 = model.evaluate_shoulder(deg_to_rad(45.0), 0.0, 0.0);
        let s2 = model.evaluate_shoulder(deg_to_rad(100.0), 0.0, 0.0);
        assert!(s0 > s1, "More flexion → lower comfort");
        assert!(s1 > s2, "Even more flexion → even lower comfort");
    }

    #[test]
    fn test_elbow_optimal_range() {
        let model = ComfortModel::new();
        let low = model.evaluate_elbow(deg_to_rad(10.0));
        let optimal = model.evaluate_elbow(deg_to_rad(90.0));
        let high = model.evaluate_elbow(deg_to_rad(140.0));
        assert!(optimal > low);
        assert!(optimal > high);
    }

    #[test]
    fn test_wrist_neutral_best() {
        let model = ComfortModel::new();
        let neutral = model.evaluate_wrist(0.0, 0.0, 0.0);
        let deviated = model.evaluate_wrist(deg_to_rad(60.0), deg_to_rad(50.0), deg_to_rad(25.0));
        assert!(neutral > deviated);
        assert!((neutral - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_sustained_hold_penalty() {
        let model = ComfortModel::new();
        let f0 = model.sustained_hold_penalty(0.0);
        let f10 = model.sustained_hold_penalty(10.0);
        let f30 = model.sustained_hold_penalty(30.0);
        let f60 = model.sustained_hold_penalty(60.0);
        assert!((f0 - 1.0).abs() < 1e-9);
        assert!(f10 < 1.0 && f10 > 0.5);
        assert!((f30 - 0.5).abs() < 0.01, "At half-life, factor ≈ 0.5, got {}", f30);
        assert!(f60 < f30);
    }

    #[test]
    fn test_comfort_map_produces_grid() {
        let (chain, params) = test_chain_and_params();
        let model = ComfortModel::new().with_config(ComfortConfig {
            map_samples: 2_000,
            map_voxel_size: 0.05,
            ..Default::default()
        });
        let grid = model.compute_comfort_map(&chain, &params).unwrap();
        assert!(grid.occupied_count() > 0);
    }

    #[test]
    fn test_identify_comfortable_subset() {
        let (chain, params) = test_chain_and_params();
        let model = ComfortModel::new().with_config(ComfortConfig {
            map_samples: 3_000,
            map_voxel_size: 0.05,
            ..Default::default()
        });
        let frac_low = model
            .identify_comfortable_subset(&chain, &params, 0.1)
            .unwrap();
        let frac_high = model
            .identify_comfortable_subset(&chain, &params, 0.9)
            .unwrap();
        assert!(
            frac_low >= frac_high,
            "Lower threshold should include more workspace: {} vs {}",
            frac_low,
            frac_high,
        );
    }

    #[test]
    fn test_reach_comfort_function() {
        assert!((reach_comfort(0.0) - 1.0).abs() < 1e-9);
        assert!((reach_comfort(0.5) - 1.0).abs() < 1e-9);
        assert!(reach_comfort(0.7) < 1.0);
        assert!(reach_comfort(0.9) < 0.5);
        assert!((reach_comfort(1.1) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_is_comfortable() {
        let model = ComfortModel::new();
        let params = BodyParameters::average_male();
        let shoulder = [0.0, 1.4, 0.0];
        assert!(model.is_comfortable(&params, shoulder, [0.0, 1.3, -0.2]));
    }
}

//! Inverse kinematics solver using damped least squares (Levenberg-Marquardt).
//!
//! Solves the IK problem: given a target position p_target, find joint angles q
//! such that FK(θ, q) ≈ p_target, subject to joint limits.

use nalgebra::{DMatrix, DVector};
use xr_types::kinematic::{KinematicChain, BodyParameters, JointType};
use xr_types::geometry::transform_position;
use xr_types::error::{VerifierError, VerifierResult};

use crate::forward_kinematics::ForwardKinematicsSolver;

/// Configuration for the IK solver.
#[derive(Debug, Clone)]
pub struct IKConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Position convergence tolerance (meters).
    pub position_tolerance: f64,
    /// Orientation convergence tolerance (radians).
    pub orientation_tolerance: f64,
    /// Initial damping factor for Levenberg-Marquardt.
    pub initial_damping: f64,
    /// Minimum damping factor.
    pub min_damping: f64,
    /// Maximum damping factor.
    pub max_damping: f64,
    /// Damping increase factor when step fails.
    pub damping_increase: f64,
    /// Damping decrease factor when step succeeds.
    pub damping_decrease: f64,
    /// Step size limit to prevent large jumps.
    pub max_step_size: f64,
    /// Number of random restarts.
    pub num_restarts: usize,
    /// Whether to use null-space optimization for redundant chains.
    pub use_null_space: bool,
    /// Weight for null-space rest-pose optimization.
    pub null_space_weight: f64,
    /// Joint limit margin (shrink limits by this amount in radians).
    pub joint_limit_margin: f64,
}

impl Default for IKConfig {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            position_tolerance: 1e-4,
            orientation_tolerance: 1e-3,
            initial_damping: 0.01,
            min_damping: 1e-6,
            max_damping: 100.0,
            damping_increase: 10.0,
            damping_decrease: 0.1,
            max_step_size: 0.5,
            num_restarts: 5,
            use_null_space: true,
            null_space_weight: 0.1,
            joint_limit_margin: 0.01,
        }
    }
}

/// Result of an IK solve attempt.
#[derive(Debug, Clone)]
pub struct IKSolution {
    /// Solved joint angles.
    pub angles: Vec<f64>,
    /// Final position error (distance to target).
    pub position_error: f64,
    /// Number of iterations used.
    pub iterations: usize,
    /// Whether the solution converged.
    pub converged: bool,
    /// Final end-effector position.
    pub final_position: [f64; 3],
}

/// Inverse kinematics solver using damped least squares.
#[derive(Debug, Clone)]
pub struct InverseKinematicsSolver {
    config: IKConfig,
    fk_solver: ForwardKinematicsSolver,
}

impl InverseKinematicsSolver {
    /// Create a new IK solver with default configuration.
    pub fn new() -> Self {
        Self {
            config: IKConfig::default(),
            fk_solver: ForwardKinematicsSolver::new(),
        }
    }

    /// Create a new IK solver with custom configuration.
    pub fn with_config(config: IKConfig) -> Self {
        Self {
            config,
            fk_solver: ForwardKinematicsSolver::new(),
        }
    }

    /// Solve IK for a target position.
    ///
    /// Uses damped least squares (Levenberg-Marquardt) with joint limit clamping
    /// and optional null-space optimization.
    pub fn solve(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        target_pos: &[f64; 3],
        initial_guess: Option<&[f64]>,
    ) -> VerifierResult<IKSolution> {
        // Check if target is within maximum reach
        let base_pos = chain.base_position(params);
        let max_reach = ForwardKinematicsSolver::max_reach(chain, params);
        let target_dist = xr_types::geometry::point_distance(target_pos, &base_pos);
        if target_dist > max_reach * 1.05 {
            return Err(VerifierError::KinematicModel(format!(
                "Target at distance {:.4} exceeds maximum reach {:.4}",
                target_dist, max_reach
            )));
        }

        let n_dof = chain.total_dof();
        let rest_pose = chain.midpoint_config(params);

        // Try from initial guess first
        let mut best_solution: Option<IKSolution> = None;

        if let Some(guess) = initial_guess {
            if guess.len() == n_dof {
                let sol = self.solve_from_seed(chain, params, target_pos, guess, &rest_pose)?;
                if sol.converged {
                    return Ok(sol);
                }
                best_solution = Some(sol);
            }
        }

        // Try from midpoint configuration
        let sol = self.solve_from_seed(chain, params, target_pos, &rest_pose, &rest_pose)?;
        if sol.converged {
            return Ok(sol);
        }
        best_solution = Self::better_solution(best_solution, sol);

        // Random restarts
        let mut rng = rand::thread_rng();
        for _ in 0..self.config.num_restarts {
            let seed = chain.random_config(params, &mut rng);
            let sol = self.solve_from_seed(chain, params, target_pos, &seed, &rest_pose)?;
            if sol.converged {
                return Ok(sol);
            }
            best_solution = Self::better_solution(best_solution, sol);
        }

        // Try analytical seed if available
        if let Some(analytical_seed) = self.analytical_seed(chain, params, target_pos) {
            let sol = self.solve_from_seed(chain, params, target_pos, &analytical_seed, &rest_pose)?;
            if sol.converged {
                return Ok(sol);
            }
            best_solution = Self::better_solution(best_solution, sol);
        }

        best_solution.ok_or(VerifierError::InverseKinematicsConvergence {
            iterations: self.config.max_iterations,
        })
    }

    /// Solve IK starting from a specific seed configuration.
    fn solve_from_seed(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        target_pos: &[f64; 3],
        seed: &[f64],
        rest_pose: &[f64],
    ) -> VerifierResult<IKSolution> {
        let n = seed.len();
        let mut q = DVector::from_row_slice(seed);
        let target = DVector::from_row_slice(target_pos);
        let q_rest = DVector::from_row_slice(rest_pose);

        let mut damping = self.config.initial_damping;
        let mut prev_error = f64::INFINITY;

        for iter in 0..self.config.max_iterations {
            // Clamp to joint limits
            let q_clamped = self.clamp_to_limits(chain, params, &q);
            q = q_clamped;

            // Compute current position
            let q_slice: Vec<f64> = q.iter().copied().collect();
            let current_pos = self.fk_solver.solve_position(chain, params, &q_slice)?;
            let current = DVector::from_row_slice(&current_pos);

            // Compute error
            let error_vec = &target - &current;
            let error = error_vec.norm();

            // Check convergence
            if error < self.config.position_tolerance {
                return Ok(IKSolution {
                    angles: q.iter().copied().collect(),
                    position_error: error,
                    iterations: iter,
                    converged: true,
                    final_position: current_pos,
                });
            }

            // Adjust damping using Levenberg-Marquardt strategy
            if error < prev_error {
                damping = (damping * self.config.damping_decrease).max(self.config.min_damping);
            } else {
                damping = (damping * self.config.damping_increase).min(self.config.max_damping);
            }
            prev_error = error;

            // Compute positional Jacobian
            let jac = self.fk_solver.jacobian_position(chain, params, &q_slice)?;

            // Damped least squares: Δq = J^T (J J^T + λ²I)^{-1} e
            let jjt = &jac * jac.transpose();
            let damping_matrix = DMatrix::identity(3, 3) * (damping * damping);
            let a = jjt + damping_matrix;

            // Solve the 3x3 system
            let decomp = a.lu();
            let y = decomp.solve(&error_vec);

            let delta_q = match y {
                Some(y_val) => jac.transpose() * y_val,
                None => {
                    // Fallback: gradient descent step
                    jac.transpose() * &error_vec * 0.01
                }
            };

            // Add null-space optimization for redundant chains (n > 3)
            let delta_q_final = if self.config.use_null_space && n > 3 {
                let null_space_term = self.null_space_projection(&jac, &q, &q_rest);
                &delta_q + &null_space_term
            } else {
                delta_q
            };

            // Limit step size
            let step_norm = delta_q_final.norm();
            let scale = if step_norm > self.config.max_step_size {
                self.config.max_step_size / step_norm
            } else {
                1.0
            };

            q += delta_q_final * scale;
        }

        // Return best found (non-converged)
        let q_final: Vec<f64> = q.iter().copied().collect();
        let final_pos = self.fk_solver.solve_position(chain, params, &q_final)?;
        let final_error = xr_types::geometry::point_distance(&final_pos, target_pos);

        Ok(IKSolution {
            angles: q_final,
            position_error: final_error,
            iterations: self.config.max_iterations,
            converged: false,
            final_position: final_pos,
        })
    }

    /// Compute null-space projection for redundancy resolution.
    /// Projects the rest-pose gradient into the Jacobian null space.
    fn null_space_projection(
        &self,
        jac: &DMatrix<f64>,
        q_current: &DVector<f64>,
        q_rest: &DVector<f64>,
    ) -> DVector<f64> {
        let n = q_current.len();

        // Compute pseudo-inverse of J
        let jjt = jac * jac.transpose();
        let decomp = jjt.lu();

        // Null-space projector: N = I - J^+ J
        let identity = DMatrix::identity(n, n);
        let j_pinv = match decomp.solve(&DMatrix::identity(3, 3)) {
            Some(jjt_inv) => jac.transpose() * jjt_inv,
            None => return DVector::zeros(n),
        };
        let j_pinv_j = &j_pinv * jac;
        let null_proj = &identity - &j_pinv_j;

        // Gradient toward rest pose
        let rest_gradient = q_rest - q_current;

        // Project into null space and scale
        null_proj * rest_gradient * self.config.null_space_weight
    }

    /// Clamp joint angles to their limits (with margin).
    fn clamp_to_limits(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        q: &DVector<f64>,
    ) -> DVector<f64> {
        let mut result = q.clone();
        let margin = self.config.joint_limit_margin;
        for (i, joint) in chain.joints.iter().enumerate() {
            if i >= result.len() {
                break;
            }
            let limits = joint.effective_limits(params);
            let min = limits.min + margin;
            let max = limits.max - margin;
            if min < max {
                result[i] = result[i].clamp(min, max);
            } else {
                result[i] = (limits.min + limits.max) / 2.0;
            }
        }
        result
    }

    /// Generate an analytical seed for simple 2-link planar case.
    /// Uses geometric IK for approximate shoulder/elbow angles.
    fn analytical_seed(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        target: &[f64; 3],
    ) -> Option<Vec<f64>> {
        if chain.joints.len() < 4 {
            return None;
        }

        let base_pos = chain.base_position(params);
        let dx = target[0] - base_pos[0];
        let dy = target[1] - base_pos[1];
        let dz = target[2] - base_pos[2];
        let target_dist = (dx * dx + dy * dy + dz * dz).sqrt();

        // Link lengths from body parameters
        let l1 = chain.joints.get(2).map(|j| j.effective_link_length(params)).unwrap_or(0.3);
        let l2 = chain.joints.get(3).map(|j| j.effective_link_length(params)).unwrap_or(0.25);

        if l1 + l2 < 1e-6 {
            return None;
        }

        // Clamp target distance to reachable range
        let d = target_dist.clamp((l1 - l2).abs() + 0.01, l1 + l2 - 0.01);

        // Elbow angle via law of cosines
        let cos_elbow = ((l1 * l1 + l2 * l2 - d * d) / (2.0 * l1 * l2)).clamp(-1.0, 1.0);
        let elbow_angle = std::f64::consts::PI - cos_elbow.acos();

        // Shoulder elevation
        let horizontal_dist = (dx * dx + dz * dz).sqrt();
        let elevation = if horizontal_dist > 1e-6 {
            dy.atan2(horizontal_dist)
        } else {
            if dy >= 0.0 { std::f64::consts::FRAC_PI_2 } else { -std::f64::consts::FRAC_PI_2 }
        };

        // Shoulder rotation (yaw)
        let yaw = dz.atan2(dx);

        // Shoulder pitch adjustment
        let cos_shoulder = ((l1 * l1 + d * d - l2 * l2) / (2.0 * l1 * d)).clamp(-1.0, 1.0);
        let shoulder_offset = cos_shoulder.acos();

        let n_dof = chain.total_dof();
        let mut seed = vec![0.0; n_dof];

        // Map to 7-DOF chain: [shoulder_flex, shoulder_abd, shoulder_rot, elbow, wrist...]
        if n_dof >= 4 {
            seed[0] = elevation + shoulder_offset; // shoulder flexion
            seed[1] = 0.0;                         // shoulder abduction
            seed[2] = yaw;                          // shoulder rotation
            seed[3] = elbow_angle;                  // elbow flexion
        }

        // Clamp to limits
        let clamped = chain.clamp_to_limits(&seed, params);
        Some(clamped)
    }

    /// Solve IK with full pose (position + orientation).
    pub fn solve_pose(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        target_pos: &[f64; 3],
        target_rot: &nalgebra::Matrix3<f64>,
        initial_guess: Option<&[f64]>,
    ) -> VerifierResult<IKSolution> {
        let n_dof = chain.total_dof();
        let rest_pose = chain.midpoint_config(params);
        let seed = initial_guess
            .map(|g| g.to_vec())
            .unwrap_or_else(|| rest_pose.clone());

        let mut q = DVector::from_row_slice(&seed);
        let mut damping = self.config.initial_damping;
        let mut prev_error = f64::INFINITY;

        for iter in 0..self.config.max_iterations {
            q = self.clamp_to_limits(chain, params, &q);
            let q_slice: Vec<f64> = q.iter().copied().collect();

            let current_t = self.fk_solver.solve(chain, params, &q_slice)?;
            let current_pos = transform_position(&current_t);

            // Position error
            let pos_error = [
                target_pos[0] - current_pos[0],
                target_pos[1] - current_pos[1],
                target_pos[2] - current_pos[2],
            ];

            // Orientation error (log map of R_target * R_current^T)
            let current_rot = nalgebra::Matrix3::new(
                current_t[(0, 0)], current_t[(0, 1)], current_t[(0, 2)],
                current_t[(1, 0)], current_t[(1, 1)], current_t[(1, 2)],
                current_t[(2, 0)], current_t[(2, 1)], current_t[(2, 2)],
            );
            let rot_error_mat = target_rot * current_rot.transpose();
            let rot_error = self.log_rotation_3x3(&rot_error_mat);

            // Combined 6D error
            let error_6d = DVector::from_row_slice(&[
                pos_error[0], pos_error[1], pos_error[2],
                rot_error[0], rot_error[1], rot_error[2],
            ]);
            let error = error_6d.norm();

            let pos_err_norm = (pos_error[0] * pos_error[0] + pos_error[1] * pos_error[1] + pos_error[2] * pos_error[2]).sqrt();
            let rot_err_norm = (rot_error[0] * rot_error[0] + rot_error[1] * rot_error[1] + rot_error[2] * rot_error[2]).sqrt();

            if pos_err_norm < self.config.position_tolerance
                && rot_err_norm < self.config.orientation_tolerance
            {
                return Ok(IKSolution {
                    angles: q.iter().copied().collect(),
                    position_error: pos_err_norm,
                    iterations: iter,
                    converged: true,
                    final_position: current_pos,
                });
            }

            // Adaptive damping
            if error < prev_error {
                damping = (damping * self.config.damping_decrease).max(self.config.min_damping);
            } else {
                damping = (damping * self.config.damping_increase).min(self.config.max_damping);
            }
            prev_error = error;

            // Full 6×N Jacobian
            let jac = self.fk_solver.jacobian(chain, params, &q_slice)?;

            let jjt = &jac * jac.transpose();
            let damping_matrix = DMatrix::identity(6, 6) * (damping * damping);
            let a = jjt + damping_matrix;

            let delta_q = match a.lu().solve(&error_6d) {
                Some(y) => jac.transpose() * y,
                None => jac.transpose() * &error_6d * 0.01,
            };

            let step_norm = delta_q.norm();
            let scale = if step_norm > self.config.max_step_size {
                self.config.max_step_size / step_norm
            } else {
                1.0
            };

            q += delta_q * scale;
        }

        let q_final: Vec<f64> = q.iter().copied().collect();
        let final_pos = self.fk_solver.solve_position(chain, params, &q_final)?;
        let final_error = xr_types::geometry::point_distance(&final_pos, target_pos);

        Ok(IKSolution {
            angles: q_final,
            position_error: final_error,
            iterations: self.config.max_iterations,
            converged: false,
            final_position: final_pos,
        })
    }

    /// Log map for 3×3 rotation matrix.
    fn log_rotation_3x3(&self, r: &nalgebra::Matrix3<f64>) -> [f64; 3] {
        let trace = r[(0, 0)] + r[(1, 1)] + r[(2, 2)];
        let cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0);
        let angle = cos_angle.acos();

        if angle.abs() < 1e-10 {
            return [
                (r[(2, 1)] - r[(1, 2)]) / 2.0,
                (r[(0, 2)] - r[(2, 0)]) / 2.0,
                (r[(1, 0)] - r[(0, 1)]) / 2.0,
            ];
        }

        let scale = angle / (2.0 * angle.sin());
        [
            scale * (r[(2, 1)] - r[(1, 2)]),
            scale * (r[(0, 2)] - r[(2, 0)]),
            scale * (r[(1, 0)] - r[(0, 1)]),
        ]
    }

    /// Solve IK for multiple targets and return all solutions.
    pub fn solve_batch(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        targets: &[[f64; 3]],
    ) -> Vec<VerifierResult<IKSolution>> {
        targets
            .iter()
            .map(|target| self.solve(chain, params, target, None))
            .collect()
    }

    /// Check if a target position is reachable (tries to solve IK).
    pub fn is_reachable(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        target_pos: &[f64; 3],
        tolerance: f64,
    ) -> bool {
        match self.solve(chain, params, target_pos, None) {
            Ok(sol) => sol.converged && sol.position_error < tolerance,
            Err(_) => false,
        }
    }

    /// Solve analytical IK for a 2-link planar manipulator.
    /// Returns up to 2 solutions (elbow-up and elbow-down).
    pub fn solve_2link_planar(
        l1: f64,
        l2: f64,
        target_x: f64,
        target_y: f64,
    ) -> Vec<[f64; 2]> {
        let d_sq = target_x * target_x + target_y * target_y;
        let d = d_sq.sqrt();

        if d > l1 + l2 || d < (l1 - l2).abs() {
            return Vec::new();
        }

        let cos_q2 = (d_sq - l1 * l1 - l2 * l2) / (2.0 * l1 * l2);
        let cos_q2 = cos_q2.clamp(-1.0, 1.0);

        let mut solutions = Vec::new();

        // Elbow-down
        let q2 = cos_q2.acos();
        let k1 = l1 + l2 * q2.cos();
        let k2 = l2 * q2.sin();
        let q1 = target_y.atan2(target_x) - k2.atan2(k1);
        solutions.push([q1, q2]);

        // Elbow-up
        let q2_up = -q2;
        let k1_up = l1 + l2 * q2_up.cos();
        let k2_up = l2 * q2_up.sin();
        let q1_up = target_y.atan2(target_x) - k2_up.atan2(k1_up);
        solutions.push([q1_up, q2_up]);

        solutions
    }

    /// Solve analytical IK for a 3-DOF spherical wrist.
    /// Given a desired rotation R, find Euler angles (ZYZ convention).
    pub fn solve_spherical_wrist(target_rot: &nalgebra::Matrix3<f64>) -> Option<[f64; 3]> {
        let r = target_rot;

        // ZYZ Euler angles
        let beta = r[(2, 2)].clamp(-1.0, 1.0).acos();

        if beta.abs() < 1e-10 {
            // Gimbal lock at β ≈ 0: α + γ determines the rotation
            let alpha = r[(1, 0)].atan2(r[(0, 0)]);
            return Some([alpha, 0.0, 0.0]);
        }

        if (beta - std::f64::consts::PI).abs() < 1e-10 {
            let alpha = (-r[(1, 0)]).atan2(-r[(0, 0)]);
            return Some([alpha, std::f64::consts::PI, 0.0]);
        }

        let sin_beta = beta.sin();
        let alpha = (r[(1, 2)] / sin_beta).atan2(r[(0, 2)] / sin_beta);
        let gamma = (r[(2, 1)] / sin_beta).atan2(-r[(2, 0)] / sin_beta);

        Some([alpha, beta, gamma])
    }

    /// Compare two solutions and return the better one (lower error).
    fn better_solution(
        current: Option<IKSolution>,
        new: IKSolution,
    ) -> Option<IKSolution> {
        match current {
            None => Some(new),
            Some(curr) => {
                if new.position_error < curr.position_error {
                    Some(new)
                } else {
                    Some(curr)
                }
            }
        }
    }

    /// Compute the pseudo-inverse of the Jacobian with damping.
    pub fn damped_pseudo_inverse(
        jac: &DMatrix<f64>,
        damping: f64,
    ) -> DMatrix<f64> {
                let jjt = jac.clone() * jac.transpose();
        let m = jjt.nrows();
        let damping_matrix = DMatrix::identity(m, m) * (damping * damping);
        let a = &jjt + damping_matrix;
        match a.lu().solve(&DMatrix::identity(m, m)) {
            Some(a_inv) => jac.transpose() * a_inv,
            None => {
                // Fallback: use transpose with scaling
                let scale = 1.0 / (jjt.diagonal().max() + damping * damping);
                jac.transpose() * scale
            }
        }
    }

    /// Perform gradient descent IK (simpler fallback method).
    pub fn solve_gradient_descent(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        target_pos: &[f64; 3],
        initial_guess: &[f64],
        step_size: f64,
        max_iters: usize,
    ) -> VerifierResult<IKSolution> {
        let mut q = initial_guess.to_vec();

        for iter in 0..max_iters {
            q = chain.clamp_to_limits(&q, params);
            let current_pos = self.fk_solver.solve_position(chain, params, &q)?;
            let error = xr_types::geometry::point_distance(&current_pos, target_pos);

            if error < self.config.position_tolerance {
                return Ok(IKSolution {
                    angles: q,
                    position_error: error,
                    iterations: iter,
                    converged: true,
                    final_position: current_pos,
                });
            }

            let jac = self.fk_solver.jacobian_position(chain, params, &q)?;
            let error_vec = DVector::from_row_slice(&[
                target_pos[0] - current_pos[0],
                target_pos[1] - current_pos[1],
                target_pos[2] - current_pos[2],
            ]);

            // Gradient: J^T * e
            let gradient = jac.transpose() * &error_vec;
            for (i, val) in gradient.iter().enumerate() {
                if i < q.len() {
                    q[i] += step_size * val;
                }
            }
        }

        let final_pos = self.fk_solver.solve_position(chain, params, &q)?;
        let final_error = xr_types::geometry::point_distance(&final_pos, target_pos);

        Ok(IKSolution {
            angles: q,
            position_error: final_error,
            iterations: max_iters,
            converged: false,
            final_position: final_pos,
        })
    }

    /// Solve IK with weighted joints (some joints prefer to stay near rest).
    pub fn solve_weighted(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        target_pos: &[f64; 3],
        joint_weights: &[f64],
        initial_guess: Option<&[f64]>,
    ) -> VerifierResult<IKSolution> {
        let n_dof = chain.total_dof();
        let rest_pose = chain.midpoint_config(params);
        let seed = initial_guess
            .map(|g| g.to_vec())
            .unwrap_or_else(|| rest_pose.clone());

        let mut q = DVector::from_row_slice(&seed);
        let mut damping = self.config.initial_damping;
        let mut prev_error = f64::INFINITY;

        // Build weight matrix
        let w_inv = DMatrix::from_diagonal(&DVector::from_iterator(
            n_dof,
            (0..n_dof).map(|i| {
                let w = if i < joint_weights.len() { joint_weights[i] } else { 1.0 };
                if w > 1e-12 { 1.0 / w } else { 1e12 }
            }),
        ));

        for iter in 0..self.config.max_iterations {
            q = self.clamp_to_limits(chain, params, &q);
            let q_slice: Vec<f64> = q.iter().copied().collect();
            let current_pos = self.fk_solver.solve_position(chain, params, &q_slice)?;

            let error_vec = DVector::from_row_slice(&[
                target_pos[0] - current_pos[0],
                target_pos[1] - current_pos[1],
                target_pos[2] - current_pos[2],
            ]);
            let error = error_vec.norm();

            if error < self.config.position_tolerance {
                return Ok(IKSolution {
                    angles: q.iter().copied().collect(),
                    position_error: error,
                    iterations: iter,
                    converged: true,
                    final_position: current_pos,
                });
            }

            if error < prev_error {
                damping = (damping * self.config.damping_decrease).max(self.config.min_damping);
            } else {
                damping = (damping * self.config.damping_increase).min(self.config.max_damping);
            }
            prev_error = error;

            let jac = self.fk_solver.jacobian_position(chain, params, &q_slice)?;

            // Weighted damped least squares: Δq = W^{-1} J^T (J W^{-1} J^T + λ²I)^{-1} e
            let jw = &jac * &w_inv;
            let jwjt = &jw * jac.transpose();
            let damping_matrix = DMatrix::identity(3, 3) * (damping * damping);
            let a = jwjt + damping_matrix;

            let delta_q = match a.lu().solve(&error_vec) {
                Some(y) => &w_inv * jac.transpose() * y,
                None => &w_inv * jac.transpose() * &error_vec * 0.01,
            };

            let step_norm = delta_q.norm();
            let scale = if step_norm > self.config.max_step_size {
                self.config.max_step_size / step_norm
            } else {
                1.0
            };

            q += delta_q * scale;
        }

        let q_final: Vec<f64> = q.iter().copied().collect();
        let final_pos = self.fk_solver.solve_position(chain, params, &q_final)?;
        let final_error = xr_types::geometry::point_distance(&final_pos, target_pos);

        Ok(IKSolution {
            angles: q_final,
            position_error: final_error,
            iterations: self.config.max_iterations,
            converged: false,
            final_position: final_pos,
        })
    }
}

impl Default for InverseKinematicsSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xr_types::kinematic::{ArmSide, BodyParameters, KinematicChain};

    fn test_chain() -> KinematicChain {
        KinematicChain::default_arm(ArmSide::Right)
    }

    fn test_params() -> BodyParameters {
        BodyParameters::average_male()
    }

    #[test]
    fn test_ik_solve_reachable_target() {
        let ik = InverseKinematicsSolver::new();
        let fk = ForwardKinematicsSolver::new();
        let chain = test_chain();
        let params = test_params();

        // Use a known reachable position (from FK with midpoint config)
        let mid_angles = chain.midpoint_config(&params);
        let target = fk.solve_position(&chain, &params, &mid_angles).unwrap();

        let result = ik.solve(&chain, &params, &target, None);
        assert!(result.is_ok());
        let sol = result.unwrap();
        assert!(sol.position_error < 0.01, "Error: {}", sol.position_error);
    }

    #[test]
    fn test_ik_unreachable_target() {
        let ik = InverseKinematicsSolver::new();
        let chain = test_chain();
        let params = test_params();

        let result = ik.solve(&chain, &params, &[100.0, 100.0, 100.0], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_ik_with_initial_guess() {
        let ik = InverseKinematicsSolver::new();
        let fk = ForwardKinematicsSolver::new();
        let chain = test_chain();
        let params = test_params();

        let mid_angles = chain.midpoint_config(&params);
        let target = fk.solve_position(&chain, &params, &mid_angles).unwrap();

        let result = ik.solve(&chain, &params, &target, Some(&mid_angles));
        assert!(result.is_ok());
        let sol = result.unwrap();
        assert!(sol.converged);
    }

    #[test]
    fn test_2link_planar_ik() {
        let solutions = InverseKinematicsSolver::solve_2link_planar(1.0, 1.0, 1.0, 0.0);
        assert!(!solutions.is_empty());
        // Verify solutions via FK
        for [q1, q2] in &solutions {
            let x = q1.cos() * 1.0 + (q1 + q2).cos() * 1.0;
            let y = q1.sin() * 1.0 + (q1 + q2).sin() * 1.0;
            assert!((x - 1.0).abs() < 1e-6, "x error: {}", (x - 1.0).abs());
            assert!(y.abs() < 1e-6, "y error: {}", y.abs());
        }
    }

    #[test]
    fn test_2link_unreachable() {
        let solutions = InverseKinematicsSolver::solve_2link_planar(1.0, 1.0, 3.0, 0.0);
        assert!(solutions.is_empty());
    }

    #[test]
    fn test_spherical_wrist_ik() {
        let identity = nalgebra::Matrix3::identity();
        let result = InverseKinematicsSolver::solve_spherical_wrist(&identity);
        assert!(result.is_some());
        let angles = result.unwrap();
        // Identity rotation should give near-zero angles
        assert!(angles[1].abs() < 1e-6);
    }

    #[test]
    fn test_is_reachable() {
        let ik = InverseKinematicsSolver::new();
        let fk = ForwardKinematicsSolver::new();
        let chain = test_chain();
        let params = test_params();

        let mid_angles = chain.midpoint_config(&params);
        let target = fk.solve_position(&chain, &params, &mid_angles).unwrap();

        assert!(ik.is_reachable(&chain, &params, &target, 0.01));
        assert!(!ik.is_reachable(&chain, &params, &[100.0, 100.0, 100.0], 0.01));
    }

    #[test]
    fn test_gradient_descent_ik() {
        let ik = InverseKinematicsSolver::new();
        let fk = ForwardKinematicsSolver::new();
        let chain = test_chain();
        let params = test_params();

        let mid_angles = chain.midpoint_config(&params);
        let target = fk.solve_position(&chain, &params, &mid_angles).unwrap();

        let result = ik.solve_gradient_descent(
            &chain, &params, &target, &mid_angles, 0.1, 500,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_damped_pseudo_inverse() {
        let jac = DMatrix::from_row_slice(3, 4, &[
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]);
        let pinv = InverseKinematicsSolver::damped_pseudo_inverse(&jac, 0.01);
        assert_eq!(pinv.nrows(), 4);
        assert_eq!(pinv.ncols(), 3);
    }

    #[test]
    fn test_ik_batch_solve() {
        let ik = InverseKinematicsSolver::new();
        let fk = ForwardKinematicsSolver::new();
        let chain = test_chain();
        let params = test_params();

        let mid_angles = chain.midpoint_config(&params);
        let pos = fk.solve_position(&chain, &params, &mid_angles).unwrap();

        let targets = vec![pos, pos];
        let results = ik.solve_batch(&chain, &params, &targets);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_ik_weighted_solve() {
        let ik = InverseKinematicsSolver::new();
        let fk = ForwardKinematicsSolver::new();
        let chain = test_chain();
        let params = test_params();

        let mid_angles = chain.midpoint_config(&params);
        let target = fk.solve_position(&chain, &params, &mid_angles).unwrap();
        let weights = vec![1.0; chain.total_dof()];

        let result = ik.solve_weighted(&chain, &params, &target, &weights, Some(&mid_angles));
        assert!(result.is_ok());
    }

    #[test]
    fn test_ik_config_defaults() {
        let config = IKConfig::default();
        assert!(config.max_iterations > 0);
        assert!(config.position_tolerance > 0.0);
        assert!(config.num_restarts > 0);
    }

    #[test]
    fn test_ik_solution_struct() {
        let sol = IKSolution {
            angles: vec![0.0; 7],
            position_error: 0.001,
            iterations: 10,
            converged: true,
            final_position: [0.0, 0.0, 0.0],
        };
        assert!(sol.converged);
        assert_eq!(sol.angles.len(), 7);
    }
}

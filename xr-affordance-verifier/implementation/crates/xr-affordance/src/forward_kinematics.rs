//! Forward kinematics solver for parameterized kinematic chains.
//!
//! Implements FK(θ, q) = T_base(θ) · Π R_i(q_i) · T_i(θ)
//! where θ are body parameters and q are joint angles.

use nalgebra::{Matrix4, DMatrix};
use xr_types::kinematic::{KinematicChain, BodyParameters, JointType};
use xr_types::geometry::{
    rotation_x, rotation_y, rotation_z, translation_transform,
    identity_transform, transform_position,
};
use xr_types::error::{VerifierError, VerifierResult};

/// Configuration for the forward kinematics solver.
#[derive(Debug, Clone)]
pub struct FKConfig {
    /// Step size for finite-difference Jacobian computation.
    pub jacobian_step: f64,
    /// Threshold for singularity detection (minimum singular value).
    pub singularity_threshold: f64,
    /// Whether to use DH parameter conversion.
    pub use_dh_parameters: bool,
}

impl Default for FKConfig {
    fn default() -> Self {
        Self {
            jacobian_step: 1e-6,
            singularity_threshold: 1e-4,
            use_dh_parameters: false,
        }
    }
}

/// Denavit-Hartenberg parameters for a single link.
#[derive(Debug, Clone, Copy)]
pub struct DHParameters {
    /// Link length a_i (distance along x_i).
    pub a: f64,
    /// Link twist α_i (angle about x_i).
    pub alpha: f64,
    /// Link offset d_i (distance along z_{i-1}).
    pub d: f64,
    /// Joint angle θ_i (angle about z_{i-1}).
    pub theta: f64,
}

impl DHParameters {
    /// Create new DH parameters.
    pub fn new(a: f64, alpha: f64, d: f64, theta: f64) -> Self {
        Self { a, alpha, d, theta }
    }

    /// Compute the 4x4 homogeneous transformation for these DH parameters.
    pub fn to_transform(&self) -> Matrix4<f64> {
        let ct = self.theta.cos();
        let st = self.theta.sin();
        let ca = self.alpha.cos();
        let sa = self.alpha.sin();
        Matrix4::new(
            ct, -st * ca,  st * sa, self.a * ct,
            st,  ct * ca, -ct * sa, self.a * st,
            0.0,      sa,       ca, self.d,
            0.0,     0.0,      0.0, 1.0,
        )
    }

    /// Compute the transform with an additional joint angle offset.
    pub fn to_transform_with_offset(&self, q: f64) -> Matrix4<f64> {
        let theta_total = self.theta + q;
        let ct = theta_total.cos();
        let st = theta_total.sin();
        let ca = self.alpha.cos();
        let sa = self.alpha.sin();
        Matrix4::new(
            ct, -st * ca,  st * sa, self.a * ct,
            st,  ct * ca, -ct * sa, self.a * st,
            0.0,      sa,       ca, self.d,
            0.0,     0.0,      0.0, 1.0,
        )
    }
}

/// Result of a singularity analysis.
#[derive(Debug, Clone)]
pub struct SingularityInfo {
    /// Minimum singular value of the Jacobian.
    pub min_singular_value: f64,
    /// Whether the configuration is near a singularity.
    pub is_singular: bool,
    /// The manipulability measure (sqrt of det(J * J^T)).
    pub manipulability: f64,
    /// Direction of the lost DOF (if singular).
    pub singular_direction: Option<[f64; 3]>,
}

/// Result of workspace boundary estimation.
#[derive(Debug, Clone)]
pub struct WorkspaceBoundary {
    /// Sampled boundary points.
    pub points: Vec<[f64; 3]>,
    /// Estimated maximum reach distance from base.
    pub max_reach: f64,
    /// Estimated minimum reach distance from base.
    pub min_reach: f64,
    /// Approximate bounding sphere radius.
    pub bounding_radius: f64,
}

/// Forward kinematics solver for parameterized kinematic chains.
#[derive(Debug, Clone)]
pub struct ForwardKinematicsSolver {
    config: FKConfig,
}

impl ForwardKinematicsSolver {
    /// Create a new FK solver with default configuration.
    pub fn new() -> Self {
        Self {
            config: FKConfig::default(),
        }
    }

    /// Create a new FK solver with custom configuration.
    pub fn with_config(config: FKConfig) -> Self {
        Self { config }
    }

    /// Compute the base transformation matrix from body parameters.
    /// T_base(θ) positions the shoulder in world space.
    fn compute_base_transform(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
    ) -> Matrix4<f64> {
        let base_pos = chain.base_position(params);
        let mut base = Matrix4::identity();
        // Copy the rotation part from the chain's base transform (row-major [f64;16])
        base[(0, 0)] = chain.base_transform[0];
        base[(0, 1)] = chain.base_transform[1];
        base[(0, 2)] = chain.base_transform[2];
        base[(1, 0)] = chain.base_transform[4];
        base[(1, 1)] = chain.base_transform[5];
        base[(1, 2)] = chain.base_transform[6];
        base[(2, 0)] = chain.base_transform[8];
        base[(2, 1)] = chain.base_transform[9];
        base[(2, 2)] = chain.base_transform[10];
        // Set translation from body-parameter-dependent position
        base[(0, 3)] = base_pos[0];
        base[(1, 3)] = base_pos[1];
        base[(2, 3)] = base_pos[2];
        base
    }

    /// Compute the rotation matrix for a single joint given its type and angle.
    fn joint_rotation(joint_type: JointType, angle: f64) -> Matrix4<f64> {
        match joint_type {
            JointType::RevoluteX => rotation_x(angle),
            JointType::RevoluteY => rotation_y(angle),
            JointType::RevoluteZ => rotation_z(angle),
            JointType::Ball => {
                // Ball joint uses a single angle parameter; in practice,
                // ball joints are decomposed into 3 revolute joints
                rotation_x(angle)
            }
            JointType::Fixed => identity_transform(),
        }
    }

    /// Compute the link translation for a joint (along Z axis by convention).
    fn link_translation(link_length: f64) -> Matrix4<f64> {
        if link_length.abs() < 1e-12 {
            identity_transform()
        } else {
            translation_transform(0.0, -link_length, 0.0)
        }
    }

    /// Compute the static transform for a joint from its [f64; 16] representation.
    fn static_transform(joint: &xr_types::kinematic::Joint) -> Matrix4<f64> {
        let t = &joint.static_transform;
        Matrix4::new(
            t[0], t[1], t[2], t[3],
            t[4], t[5], t[6], t[7],
            t[8], t[9], t[10], t[11],
            t[12], t[13], t[14], t[15],
        )
    }

    /// Solve forward kinematics: compute the end-effector transform.
    ///
    /// FK(θ, q) = T_base(θ) · Π (S_i · R_i(q_i) · T_i(θ))
    ///
    /// where S_i is the static transform, R_i is the joint rotation,
    /// and T_i is the link translation.
    pub fn solve(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        angles: &[f64],
    ) -> VerifierResult<Matrix4<f64>> {
        self.validate_angles(chain, angles)?;
        let transforms = self.compute_all_transforms(chain, params, angles);
        Ok(transforms.last().cloned().unwrap_or_else(identity_transform))
    }

    /// Solve FK and return only the end-effector position.
    pub fn solve_position(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        angles: &[f64],
    ) -> VerifierResult<[f64; 3]> {
        let t = self.solve(chain, params, angles)?;
        Ok(transform_position(&t))
    }

    /// Compute all intermediate joint transforms (cumulative).
    /// Returns N+1 transforms: T_base, T_base*T_0, T_base*T_0*T_1, ...
    pub fn solve_all_joints(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        angles: &[f64],
    ) -> VerifierResult<Vec<Matrix4<f64>>> {
        self.validate_angles(chain, angles)?;
        Ok(self.compute_all_transforms(chain, params, angles))
    }

    /// Internal method to compute all cumulative transforms.
    fn compute_all_transforms(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        angles: &[f64],
    ) -> Vec<Matrix4<f64>> {
        let base = self.compute_base_transform(chain, params);
        let mut transforms = Vec::with_capacity(chain.joints.len() + 1);
        transforms.push(base);

        let mut current = base;
        let mut angle_idx = 0;

        for joint in &chain.joints {
            // Apply static transform
            let static_t = Self::static_transform(joint);
            if static_t != identity_transform() {
                current = current * static_t;
            }

            // Apply joint rotation
            let dof = joint.joint_type.dof();
            if dof > 0 && angle_idx < angles.len() {
                let rot = Self::joint_rotation(joint.joint_type, angles[angle_idx]);
                current = current * rot;
                angle_idx += 1;
            }

            // Apply link translation (body-parameter-dependent)
            let link_len = joint.effective_link_length(params);
            if link_len.abs() > 1e-12 {
                let link_t = Self::link_translation(link_len);
                current = current * link_t;
            }

            transforms.push(current);
        }

        transforms
    }

    /// Validate that the angles vector has the right dimension.
    fn validate_angles(
        &self,
        chain: &KinematicChain,
        angles: &[f64],
    ) -> VerifierResult<()> {
        let expected_dof = chain.total_dof();
        if angles.len() != expected_dof {
            return Err(VerifierError::DimensionMismatch {
                expected: expected_dof,
                got: angles.len(),
            });
        }
        Ok(())
    }

    /// Compute the 6×N Jacobian matrix via finite differences.
    ///
    /// The Jacobian maps joint velocities to end-effector twist:
    /// [v_x, v_y, v_z, ω_x, ω_y, ω_z]^T = J(q) * q_dot
    ///
    /// Uses central finite differences for numerical stability.
    pub fn jacobian(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        angles: &[f64],
    ) -> VerifierResult<DMatrix<f64>> {
        let n = angles.len();
        let h = self.config.jacobian_step;
        let mut jac = DMatrix::zeros(6, n);

        // Compute the nominal end-effector transform
        let t_nom = self.solve(chain, params, angles)?;
        let p_nom = transform_position(&t_nom);

        for i in 0..n {
            // Forward perturbation
            let mut angles_plus = angles.to_vec();
            angles_plus[i] += h;
            let t_plus = self.solve(chain, params, &angles_plus)?;
            let p_plus = transform_position(&t_plus);

            // Backward perturbation
            let mut angles_minus = angles.to_vec();
            angles_minus[i] -= h;
            let t_minus = self.solve(chain, params, &angles_minus)?;
            let p_minus = transform_position(&t_minus);

            // Linear velocity (central difference)
            let inv_2h = 1.0 / (2.0 * h);
            jac[(0, i)] = (p_plus[0] - p_minus[0]) * inv_2h;
            jac[(1, i)] = (p_plus[1] - p_minus[1]) * inv_2h;
            jac[(2, i)] = (p_plus[2] - p_minus[2]) * inv_2h;

            // Angular velocity (approximate from rotation difference)
            let rot_diff = self.rotation_difference(&t_plus, &t_minus);
            jac[(3, i)] = rot_diff[0] * inv_2h;
            jac[(4, i)] = rot_diff[1] * inv_2h;
            jac[(5, i)] = rot_diff[2] * inv_2h;
        }

        Ok(jac)
    }

    /// Compute the 3×N positional Jacobian (linear velocity only).
    pub fn jacobian_position(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        angles: &[f64],
    ) -> VerifierResult<DMatrix<f64>> {
        let n = angles.len();
        let h = self.config.jacobian_step;
        let mut jac = DMatrix::zeros(3, n);

        for i in 0..n {
            let mut angles_plus = angles.to_vec();
            angles_plus[i] += h;
            let p_plus = self.solve_position(chain, params, &angles_plus)?;

            let mut angles_minus = angles.to_vec();
            angles_minus[i] -= h;
            let p_minus = self.solve_position(chain, params, &angles_minus)?;

            let inv_2h = 1.0 / (2.0 * h);
            jac[(0, i)] = (p_plus[0] - p_minus[0]) * inv_2h;
            jac[(1, i)] = (p_plus[1] - p_minus[1]) * inv_2h;
            jac[(2, i)] = (p_plus[2] - p_minus[2]) * inv_2h;
        }

        Ok(jac)
    }

    /// Extract the angular velocity vector from the difference of two rotation matrices.
    /// Uses the logarithmic map approximation.
    fn rotation_difference(&self, t_a: &Matrix4<f64>, t_b: &Matrix4<f64>) -> [f64; 3] {
        // Extract 3x3 rotation submatrices
        let r_a = nalgebra::Matrix3::new(
            t_a[(0, 0)], t_a[(0, 1)], t_a[(0, 2)],
            t_a[(1, 0)], t_a[(1, 1)], t_a[(1, 2)],
            t_a[(2, 0)], t_a[(2, 1)], t_a[(2, 2)],
        );
        let r_b = nalgebra::Matrix3::new(
            t_b[(0, 0)], t_b[(0, 1)], t_b[(0, 2)],
            t_b[(1, 0)], t_b[(1, 1)], t_b[(1, 2)],
            t_b[(2, 0)], t_b[(2, 1)], t_b[(2, 2)],
        );

        // R_diff = R_a * R_b^T
        let r_diff = r_a * r_b.transpose();

        // Extract rotation vector via logarithmic map
        self.log_rotation(&r_diff)
    }

    /// Compute the logarithmic map of a rotation matrix (rotation vector).
    fn log_rotation(&self, r: &nalgebra::Matrix3<f64>) -> [f64; 3] {
        let trace = r[(0, 0)] + r[(1, 1)] + r[(2, 2)];
        let cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0);
        let angle = cos_angle.acos();

        if angle.abs() < 1e-10 {
            // Near identity: use first-order approximation
            return [
                (r[(2, 1)] - r[(1, 2)]) / 2.0,
                (r[(0, 2)] - r[(2, 0)]) / 2.0,
                (r[(1, 0)] - r[(0, 1)]) / 2.0,
            ];
        }

        if (angle - std::f64::consts::PI).abs() < 1e-10 {
            // Near π: special handling
            let diag = [r[(0, 0)], r[(1, 1)], r[(2, 2)]];
            let max_idx = if diag[0] >= diag[1] && diag[0] >= diag[2] {
                0
            } else if diag[1] >= diag[2] {
                1
            } else {
                2
            };
            let mut axis = [0.0; 3];
            axis[max_idx] = ((diag[max_idx] + 1.0) / 2.0).sqrt();
            let inv = 1.0 / (2.0 * axis[max_idx]);
            for i in 0..3 {
                if i != max_idx {
                    axis[i] = r[(i, max_idx)] * inv;
                }
            }
            return [
                axis[0] * angle,
                axis[1] * angle,
                axis[2] * angle,
            ];
        }

        let scale = angle / (2.0 * angle.sin());
        [
            scale * (r[(2, 1)] - r[(1, 2)]),
            scale * (r[(0, 2)] - r[(2, 0)]),
            scale * (r[(1, 0)] - r[(0, 1)]),
        ]
    }

    /// Detect singularities by analyzing the Jacobian.
    pub fn detect_singularity(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        angles: &[f64],
    ) -> VerifierResult<SingularityInfo> {
        let jac = self.jacobian(chain, params, angles)?;

        // Compute J * J^T
        let jjt = &jac * jac.transpose();

        // Use eigenvalue decomposition to find singular values
        let eig = jjt.symmetric_eigen();
        let eigenvalues = eig.eigenvalues;

        let min_eigenvalue = eigenvalues
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
            .max(0.0);
        let min_sv = min_eigenvalue.sqrt();

        let product: f64 = eigenvalues
            .iter()
            .copied()
            .map(|e| e.max(0.0))
            .product();
        let manipulability = product.sqrt();

        let is_singular = min_sv < self.config.singularity_threshold;

        let singular_direction = if is_singular {
            // Find the eigenvector corresponding to the smallest eigenvalue
            let min_idx = eigenvalues
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            let ev = eig.eigenvectors.column(min_idx);
            if ev.len() >= 3 {
                Some([ev[0], ev[1], ev[2]])
            } else {
                None
            }
        } else {
            None
        };

        Ok(SingularityInfo {
            min_singular_value: min_sv,
            is_singular,
            manipulability,
            singular_direction,
        })
    }

    /// Compute the manipulability measure at a configuration.
    /// w(q) = sqrt(det(J(q) * J(q)^T))
    pub fn manipulability(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        angles: &[f64],
    ) -> VerifierResult<f64> {
        let jac = self.jacobian_position(chain, params, angles)?;
        let jjt = &jac * jac.transpose();
        Ok(jjt.determinant().abs().sqrt())
    }

    /// Estimate workspace boundaries by sampling extremal configurations.
    pub fn estimate_workspace_boundary(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        num_directions: usize,
        samples_per_direction: usize,
    ) -> VerifierResult<WorkspaceBoundary> {
        let base_pos = chain.base_position(params);
        let mut boundary_points = Vec::new();
        let mut max_reach = 0.0f64;
        let mut min_reach = f64::INFINITY;

        // Sample directions uniformly on the sphere using the Fibonacci spiral
        let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let directions: Vec<[f64; 3]> = (0..num_directions)
            .map(|i| {
                let theta = (2.0 * std::f64::consts::PI * i as f64) / golden_ratio;
                let phi = (1.0 - 2.0 * (i as f64 + 0.5) / num_directions as f64).acos();
                [
                    phi.sin() * theta.cos(),
                    phi.sin() * theta.sin(),
                    phi.cos(),
                ]
            })
            .collect();

        let mut rng = rand::thread_rng();

        for dir in &directions {
            let mut best_dist = 0.0f64;
            let mut best_point = base_pos;
            let mut closest_dist = f64::INFINITY;
            let mut closest_point = base_pos;

            for _ in 0..samples_per_direction {
                let angles = chain.random_config(params, &mut rng);
                if let Ok(pos) = self.solve_position(chain, params, &angles) {
                    let dx = pos[0] - base_pos[0];
                    let dy = pos[1] - base_pos[1];
                    let dz = pos[2] - base_pos[2];
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                    // Project onto direction
                    let proj = dx * dir[0] + dy * dir[1] + dz * dir[2];
                    if proj > best_dist {
                        best_dist = proj;
                        best_point = pos;
                    }
                    if dist > 0.01 && dist < closest_dist {
                        closest_dist = dist;
                        closest_point = pos;
                    }

                    max_reach = max_reach.max(dist);
                    if dist > 0.01 {
                        min_reach = min_reach.min(dist);
                    }
                }
            }

            if best_dist > 0.0 {
                boundary_points.push(best_point);
            }
            if closest_dist < f64::INFINITY {
                let _ = closest_point; // Used for min_reach tracking
            }
        }

        if min_reach == f64::INFINITY {
            min_reach = 0.0;
        }

        Ok(WorkspaceBoundary {
            points: boundary_points,
            max_reach,
            min_reach,
            bounding_radius: max_reach,
        })
    }

    /// Convert joint parameters to DH parameters for a 7-DOF arm.
    pub fn to_dh_parameters(
        chain: &KinematicChain,
        params: &BodyParameters,
    ) -> Vec<DHParameters> {
        let mut dh_params = Vec::with_capacity(chain.joints.len());
        for joint in &chain.joints {
            let link_length = joint.effective_link_length(params);
            let (a, alpha, d, theta_offset) = match joint.joint_type {
                JointType::RevoluteX => (0.0, 0.0, link_length, 0.0),
                JointType::RevoluteY => (0.0, -std::f64::consts::FRAC_PI_2, link_length, 0.0),
                JointType::RevoluteZ => (0.0, std::f64::consts::FRAC_PI_2, link_length, 0.0),
                JointType::Ball => (0.0, 0.0, link_length, 0.0),
                JointType::Fixed => (link_length, 0.0, 0.0, 0.0),
            };
            dh_params.push(DHParameters::new(a, alpha, d, theta_offset));
        }
        dh_params
    }

    /// Solve FK using DH parameter convention.
    pub fn solve_dh(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        angles: &[f64],
    ) -> VerifierResult<Matrix4<f64>> {
        self.validate_angles(chain, angles)?;
        let base = self.compute_base_transform(chain, params);
        let dh_params = Self::to_dh_parameters(chain, params);

        let mut current = base;
        let mut angle_idx = 0;

        for (i, dh) in dh_params.iter().enumerate() {
            let dof = chain.joints[i].joint_type.dof();
            if dof > 0 && angle_idx < angles.len() {
                current = current * dh.to_transform_with_offset(angles[angle_idx]);
                angle_idx += 1;
            } else {
                current = current * dh.to_transform();
            }
        }

        Ok(current)
    }

    /// Compute end-effector velocity given joint velocities.
    /// v = J(q) * q_dot
    pub fn end_effector_velocity(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        angles: &[f64],
        joint_velocities: &[f64],
    ) -> VerifierResult<[f64; 6]> {
        let jac = self.jacobian(chain, params, angles)?;
        let q_dot = nalgebra::DVector::from_row_slice(joint_velocities);
        let v = &jac * q_dot;
        Ok([v[0], v[1], v[2], v[3], v[4], v[5]])
    }

    /// Compute the condition number of the Jacobian.
    /// A high condition number indicates proximity to a singularity.
    pub fn condition_number(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        angles: &[f64],
    ) -> VerifierResult<f64> {
        let jac = self.jacobian_position(chain, params, angles)?;
        let jjt = &jac * jac.transpose();
        let eig = jjt.symmetric_eigen();
        let eigenvalues: Vec<f64> = eig.eigenvalues.iter().copied().collect();

        let max_ev = eigenvalues.iter().copied().fold(f64::NEG_INFINITY, f64::max).max(0.0);
        let min_ev = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min).max(0.0);

        if min_ev < 1e-15 {
            Ok(f64::INFINITY)
        } else {
            Ok((max_ev / min_ev).sqrt())
        }
    }

    /// Compute all joint positions in world space.
    pub fn joint_positions(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        angles: &[f64],
    ) -> VerifierResult<Vec<[f64; 3]>> {
        let transforms = self.solve_all_joints(chain, params, angles)?;
        Ok(transforms.iter().map(|t| transform_position(t)).collect())
    }

    /// Compute the distance from the end-effector to the base.
    pub fn end_effector_distance(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        angles: &[f64],
    ) -> VerifierResult<f64> {
        let pos = self.solve_position(chain, params, angles)?;
        let base_pos = chain.base_position(params);
        let dx = pos[0] - base_pos[0];
        let dy = pos[1] - base_pos[1];
        let dz = pos[2] - base_pos[2];
        Ok((dx * dx + dy * dy + dz * dz).sqrt())
    }

    /// Check if a configuration achieves a target position within tolerance.
    pub fn reaches_target(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        angles: &[f64],
        target: &[f64; 3],
        tolerance: f64,
    ) -> VerifierResult<bool> {
        let pos = self.solve_position(chain, params, angles)?;
        let dx = pos[0] - target[0];
        let dy = pos[1] - target[1];
        let dz = pos[2] - target[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        Ok(dist <= tolerance)
    }

    /// Compute the geometric Jacobian analytically using joint axes and positions.
    pub fn geometric_jacobian(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        angles: &[f64],
    ) -> VerifierResult<DMatrix<f64>> {
        let transforms = self.solve_all_joints(chain, params, angles)?;
        let n = chain.joints.len();
        let ee_pos = transform_position(transforms.last().unwrap_or(&identity_transform()));
        let mut jac = DMatrix::zeros(6, n);

        for (i, joint) in chain.joints.iter().enumerate() {
            if joint.joint_type == JointType::Fixed {
                continue;
            }

            let t_i = &transforms[i];
            let p_i = transform_position(t_i);

            // Get the joint axis in world frame
            let local_axis = joint.joint_type.axis();
            let world_axis = [
                t_i[(0, 0)] * local_axis[0] + t_i[(0, 1)] * local_axis[1] + t_i[(0, 2)] * local_axis[2],
                t_i[(1, 0)] * local_axis[0] + t_i[(1, 1)] * local_axis[1] + t_i[(1, 2)] * local_axis[2],
                t_i[(2, 0)] * local_axis[0] + t_i[(2, 1)] * local_axis[1] + t_i[(2, 2)] * local_axis[2],
            ];

            // Vector from joint to end-effector
            let r = [
                ee_pos[0] - p_i[0],
                ee_pos[1] - p_i[1],
                ee_pos[2] - p_i[2],
            ];

            // Linear velocity: z_i × r_i
            let v = xr_types::geometry::cross3(&world_axis, &r);
            jac[(0, i)] = v[0];
            jac[(1, i)] = v[1];
            jac[(2, i)] = v[2];

            // Angular velocity: z_i
            jac[(3, i)] = world_axis[0];
            jac[(4, i)] = world_axis[1];
            jac[(5, i)] = world_axis[2];
        }

        Ok(jac)
    }

    /// Compute the maximum isotropic reach distance.
    pub fn max_reach(chain: &KinematicChain, params: &BodyParameters) -> f64 {
        chain.joints.iter().map(|j| j.effective_link_length(params)).sum()
    }

    /// Evaluate the isotropy index at a configuration.
    /// Returns a value in [0, 1] where 1 is perfectly isotropic.
    pub fn isotropy_index(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        angles: &[f64],
    ) -> VerifierResult<f64> {
        let cond = self.condition_number(chain, params, angles)?;
        if cond.is_infinite() || cond < 1.0 {
            Ok(0.0)
        } else {
            Ok(1.0 / cond)
        }
    }
}

impl Default for ForwardKinematicsSolver {
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
    fn test_fk_zero_angles() {
        let solver = ForwardKinematicsSolver::new();
        let chain = test_chain();
        let params = test_params();
        let angles = vec![0.0; chain.total_dof()];

        let result = solver.solve(&chain, &params, &angles);
        assert!(result.is_ok());
        let t = result.unwrap();
        // End-effector should be at some valid position
        let pos = transform_position(&t);
        assert!(pos[0].is_finite());
        assert!(pos[1].is_finite());
        assert!(pos[2].is_finite());
    }

    #[test]
    fn test_fk_position() {
        let solver = ForwardKinematicsSolver::new();
        let chain = test_chain();
        let params = test_params();
        let angles = vec![0.0; chain.total_dof()];

        let pos = solver.solve_position(&chain, &params, &angles).unwrap();
        // Position should be reachable (within arm length of shoulder)
        let base = chain.base_position(&params);
        let dx = pos[0] - base[0];
        let dy = pos[1] - base[1];
        let dz = pos[2] - base[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        assert!(dist <= params.total_reach() + 0.1);
    }

    #[test]
    fn test_fk_all_joints() {
        let solver = ForwardKinematicsSolver::new();
        let chain = test_chain();
        let params = test_params();
        let angles = vec![0.0; chain.total_dof()];

        let transforms = solver.solve_all_joints(&chain, &params, &angles).unwrap();
        // Should have N+1 transforms (base + one per joint)
        assert_eq!(transforms.len(), chain.joints.len() + 1);
    }

    #[test]
    fn test_fk_dimension_mismatch() {
        let solver = ForwardKinematicsSolver::new();
        let chain = test_chain();
        let params = test_params();
        let angles = vec![0.0; 3]; // Wrong dimension

        let result = solver.solve(&chain, &params, &angles);
        assert!(result.is_err());
    }

    #[test]
    fn test_jacobian_size() {
        let solver = ForwardKinematicsSolver::new();
        let chain = test_chain();
        let params = test_params();
        let angles = vec![0.0; chain.total_dof()];

        let jac = solver.jacobian(&chain, &params, &angles).unwrap();
        assert_eq!(jac.nrows(), 6);
        assert_eq!(jac.ncols(), chain.total_dof());
    }

    #[test]
    fn test_jacobian_position_size() {
        let solver = ForwardKinematicsSolver::new();
        let chain = test_chain();
        let params = test_params();
        let angles = vec![0.0; chain.total_dof()];

        let jac = solver.jacobian_position(&chain, &params, &angles).unwrap();
        assert_eq!(jac.nrows(), 3);
        assert_eq!(jac.ncols(), chain.total_dof());
    }

    #[test]
    fn test_singularity_detection() {
        let solver = ForwardKinematicsSolver::new();
        let chain = test_chain();
        let params = test_params();
        let angles = vec![0.0; chain.total_dof()];

        let info = solver.detect_singularity(&chain, &params, &angles).unwrap();
        assert!(info.min_singular_value >= 0.0);
        assert!(info.manipulability >= 0.0);
    }

    #[test]
    fn test_dh_parameters() {
        let chain = test_chain();
        let params = test_params();
        let dh = ForwardKinematicsSolver::to_dh_parameters(&chain, &params);
        assert_eq!(dh.len(), chain.joints.len());
    }

    #[test]
    fn test_dh_transform() {
        let dh = DHParameters::new(0.0, 0.0, 0.3, 0.0);
        let t = dh.to_transform();
        // Should be a valid transform
        assert!((t[(3, 3)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_joint_positions() {
        let solver = ForwardKinematicsSolver::new();
        let chain = test_chain();
        let params = test_params();
        let angles = vec![0.0; chain.total_dof()];

        let positions = solver.joint_positions(&chain, &params, &angles).unwrap();
        assert_eq!(positions.len(), chain.joints.len() + 1);
        // Each position should be finite
        for pos in &positions {
            assert!(pos[0].is_finite() && pos[1].is_finite() && pos[2].is_finite());
        }
    }

    #[test]
    fn test_end_effector_distance() {
        let solver = ForwardKinematicsSolver::new();
        let chain = test_chain();
        let params = test_params();
        let angles = vec![0.0; chain.total_dof()];

        let dist = solver.end_effector_distance(&chain, &params, &angles).unwrap();
        assert!(dist > 0.0);
        assert!(dist <= params.total_reach() + 0.1);
    }

    #[test]
    fn test_reaches_target() {
        let solver = ForwardKinematicsSolver::new();
        let chain = test_chain();
        let params = test_params();
        let angles = vec![0.0; chain.total_dof()];

        let pos = solver.solve_position(&chain, &params, &angles).unwrap();
        // Should reach itself
        assert!(solver.reaches_target(&chain, &params, &angles, &pos, 0.01).unwrap());
        // Should not reach a far-away point
        assert!(!solver.reaches_target(&chain, &params, &angles, &[100.0, 100.0, 100.0], 0.01).unwrap());
    }

    #[test]
    fn test_manipulability() {
        let solver = ForwardKinematicsSolver::new();
        let chain = test_chain();
        let params = test_params();
        let angles = chain.midpoint_config(&params);

        let m = solver.manipulability(&chain, &params, &angles).unwrap();
        assert!(m >= 0.0);
    }

    #[test]
    fn test_condition_number() {
        let solver = ForwardKinematicsSolver::new();
        let chain = test_chain();
        let params = test_params();
        let angles = chain.midpoint_config(&params);

        let cn = solver.condition_number(&chain, &params, &angles).unwrap();
        assert!(cn >= 1.0 || cn.is_infinite());
    }

    #[test]
    fn test_isotropy_index() {
        let solver = ForwardKinematicsSolver::new();
        let chain = test_chain();
        let params = test_params();
        let angles = chain.midpoint_config(&params);

        let iso = solver.isotropy_index(&chain, &params, &angles).unwrap();
        assert!(iso >= 0.0 && iso <= 1.0);
    }

    #[test]
    fn test_max_reach() {
        let chain = test_chain();
        let params = test_params();
        let max_r = ForwardKinematicsSolver::max_reach(&chain, &params);
        assert!(max_r > 0.5);
        assert!(max_r < 2.0);
    }

    #[test]
    fn test_fk_different_body_params() {
        let solver = ForwardKinematicsSolver::new();
        let chain = test_chain();
        let small = BodyParameters::small_female();
        let large = BodyParameters::large_male();
        let angles = vec![0.0; chain.total_dof()];

        let pos_small = solver.solve_position(&chain, &small, &angles).unwrap();
        let pos_large = solver.solve_position(&chain, &large, &angles).unwrap();

        // Large body should have different reach
        let dist_small = (pos_small[0] * pos_small[0] + pos_small[1] * pos_small[1] + pos_small[2] * pos_small[2]).sqrt();
        let dist_large = (pos_large[0] * pos_large[0] + pos_large[1] * pos_large[1] + pos_large[2] * pos_large[2]).sqrt();
        assert!((dist_small - dist_large).abs() > 0.01);
    }

    #[test]
    fn test_geometric_jacobian() {
        let solver = ForwardKinematicsSolver::new();
        let chain = test_chain();
        let params = test_params();
        let angles = chain.midpoint_config(&params);

        let jac = solver.geometric_jacobian(&chain, &params, &angles).unwrap();
        assert_eq!(jac.nrows(), 6);
        assert_eq!(jac.ncols(), chain.joints.len());
    }

    #[test]
    fn test_ee_velocity() {
        let solver = ForwardKinematicsSolver::new();
        let chain = test_chain();
        let params = test_params();
        let angles = vec![0.0; chain.total_dof()];
        let vel = vec![0.0; chain.total_dof()];

        let ee_vel = solver.end_effector_velocity(&chain, &params, &angles, &vel).unwrap();
        // Zero joint velocities should give zero EE velocity
        for v in &ee_vel {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn test_workspace_boundary_estimation() {
        let solver = ForwardKinematicsSolver::new();
        let chain = test_chain();
        let params = test_params();

        let boundary = solver.estimate_workspace_boundary(&chain, &params, 20, 50).unwrap();
        assert!(!boundary.points.is_empty());
        assert!(boundary.max_reach > 0.0);
        assert!(boundary.max_reach <= params.total_reach() + 0.5);
    }

    #[test]
    fn test_log_rotation_identity() {
        let solver = ForwardKinematicsSolver::new();
        let identity = nalgebra::Matrix3::identity();
        let log = solver.log_rotation(&identity);
        assert!(log[0].abs() < 1e-10);
        assert!(log[1].abs() < 1e-10);
        assert!(log[2].abs() < 1e-10);
    }
}

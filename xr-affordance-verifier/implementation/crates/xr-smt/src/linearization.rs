//! Linearized kinematics for FK Taylor approximation.
//!
//! This module implements the linearization step of the verification pipeline:
//! given a kinematic chain and a reference configuration (body parameters + joint
//! angles), it computes a first-order Taylor expansion of the forward kinematics
//! map. The resulting [`LinearizedModel`] contains Jacobians that let us
//! over-approximate FK as an affine function, which is directly encodable into
//! QF_LRA for SMT solving.
//!
//! # Soundness
//!
//! The error bound follows Theorem C2 (Appendix C):
//!
//! ```text
//! ‖FK(θ, q) − FK_lin(θ, q)‖ ≤ C_FK · (Δq² + Δθ · Δq) · L_sum
//! ```
//!
//! where `C_FK = n / 2` for an `n`-joint chain and `L_sum` is the sum of link
//! lengths.

use nalgebra::{DMatrix, DVector, Matrix4, Vector3, Vector4};
use serde::{Deserialize, Serialize};

use xr_types::{BodyParameters, JointType, KinematicChain, VerifierError};
use xr_types::kinematic::Joint;

use crate::expr::SmtExpr;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of body parameters (stature, arm_length, shoulder_breadth,
/// forearm_length, hand_length).
const NUM_PARAMS: usize = 5;

/// Spatial dimension of FK output.
const SPATIAL_DIM: usize = 3;

/// Default finite-difference step size for numerical Jacobians.
const DEFAULT_EPSILON: f64 = 1e-7;

/// Default maximum acceptable linearization error (meters).
const DEFAULT_MAX_ERROR_BOUND: f64 = 0.05;

/// Acceptable deviation scale for `is_within_error_envelope`.
const ENVELOPE_SCALE: f64 = 0.2;

// ---------------------------------------------------------------------------
// LinearizedModel
// ---------------------------------------------------------------------------

/// A first-order linearized forward-kinematics model.
///
/// Given reference body parameters `θ₀` and joint configuration `q₀`, the
/// linearized FK is:
///
/// ```text
/// FK_lin(θ, q) = p₀ + J_θ (θ − θ₀) + J_q (q − q₀)
/// ```
///
/// where `p₀ = FK(θ₀, q₀)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearizedModel {
    /// Jacobian w.r.t. body parameters (3 × num_params).
    pub jacobian_theta: DMatrix<f64>,
    /// Jacobian w.r.t. joint angles (3 × num_joints).
    pub jacobian_q: DMatrix<f64>,
    /// Reference body parameter values.
    pub reference_params: DVector<f64>,
    /// Reference joint configuration.
    pub reference_config: DVector<f64>,
    /// FK evaluated at the reference point.
    pub reference_position: Vector3<f64>,
    /// Conservative upper bound on the linearization error (meters).
    pub error_bound: f64,
    /// Number of joints in the chain.
    pub num_joints: usize,
    /// Number of body parameters.
    pub num_params: usize,
    /// Link lengths evaluated at the reference body parameters.
    pub link_lengths: Vec<f64>,
}

impl LinearizedModel {
    /// Evaluate the linearized FK at `(params, config)`.
    ///
    /// ```text
    /// p₀ + J_θ (θ − θ₀) + J_q (q − q₀)
    /// ```
    pub fn evaluate_linear(
        &self,
        params: &DVector<f64>,
        config: &DVector<f64>,
    ) -> Result<Vector3<f64>, VerifierError> {
        if params.len() != self.num_params {
            return Err(VerifierError::DimensionMismatch {
                expected: self.num_params,
                got: params.len(),
            });
        }
        if config.len() != self.num_joints {
            return Err(VerifierError::DimensionMismatch {
                expected: self.num_joints,
                got: config.len(),
            });
        }

        let delta_theta = params - &self.reference_params;
        let delta_q = config - &self.reference_config;

        let contrib_theta = &self.jacobian_theta * delta_theta;
        let contrib_q = &self.jacobian_q * delta_q;

        let mut result = self.reference_position;
        for i in 0..SPATIAL_DIM {
            result[i] += contrib_theta[i] + contrib_q[i];
        }
        Ok(result)
    }

    /// Returns `true` when the parameter / configuration deltas are small
    /// enough for the linearization to be trustworthy.
    ///
    /// The heuristic checks that:
    /// - `‖Δθ‖∞ ≤ ENVELOPE_SCALE`
    /// - `‖Δq‖∞ ≤ ENVELOPE_SCALE`
    pub fn is_within_error_envelope(
        &self,
        params: &DVector<f64>,
        config: &DVector<f64>,
    ) -> bool {
        if params.len() != self.num_params || config.len() != self.num_joints {
            return false;
        }

        let delta_theta = params - &self.reference_params;
        let delta_q = config - &self.reference_config;

        let max_theta = delta_theta.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let max_q = delta_q.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);

        max_theta <= ENVELOPE_SCALE && max_q <= ENVELOPE_SCALE
    }

    /// Convert the linearized FK into SMT expressions (one per coordinate).
    ///
    /// Each expression is of the form:
    ///
    /// ```text
    /// ref_pos[i] + Σ_j J_θ[i,j] * (param_j − ref_param_j)
    ///            + Σ_k J_q[i,k] * (config_k − ref_config_k)
    /// ```
    ///
    /// Variables are named `"{param_prefix}_{j}"` and `"{config_prefix}_{k}"`.
    pub fn to_smt_expressions(
        &self,
        param_prefix: &str,
        config_prefix: &str,
    ) -> Vec<SmtExpr> {
        let mut coord_exprs = Vec::with_capacity(SPATIAL_DIM);

        for i in 0..SPATIAL_DIM {
            // Start with the reference position constant.
            let mut terms: Vec<SmtExpr> = vec![SmtExpr::real(self.reference_position[i])];

            // Body-parameter contribution: J_θ[i,j] * (param_j − ref_j)
            for j in 0..self.num_params {
                let coeff = self.jacobian_theta[(i, j)];
                if coeff.abs() < 1e-15 {
                    continue;
                }
                let var = SmtExpr::var(format!("{}_{}", param_prefix, j));
                let ref_val = SmtExpr::real(self.reference_params[j]);
                let delta = SmtExpr::sub(var, ref_val);
                terms.push(SmtExpr::mul(SmtExpr::real(coeff), delta));
            }

            // Joint-angle contribution: J_q[i,k] * (config_k − ref_k)
            for k in 0..self.num_joints {
                let coeff = self.jacobian_q[(i, k)];
                if coeff.abs() < 1e-15 {
                    continue;
                }
                let var = SmtExpr::var(format!("{}_{}", config_prefix, k));
                let ref_val = SmtExpr::real(self.reference_config[k]);
                let delta = SmtExpr::sub(var, ref_val);
                terms.push(SmtExpr::mul(SmtExpr::real(coeff), delta));
            }

            // Fold into a single addition tree.
            let expr = fold_sum(terms);
            coord_exprs.push(expr);
        }

        coord_exprs
    }
}

// ---------------------------------------------------------------------------
// LinearizationEngine
// ---------------------------------------------------------------------------

/// Engine that computes [`LinearizedModel`] instances via finite differences.
#[derive(Debug, Clone)]
pub struct LinearizationEngine {
    /// Step size for numerical differentiation.
    epsilon: f64,
    /// Maximum acceptable linearization error bound.
    max_error_bound: f64,
}

impl Default for LinearizationEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearizationEngine {
    /// Create a new engine with default parameters.
    pub fn new() -> Self {
        Self {
            epsilon: DEFAULT_EPSILON,
            max_error_bound: DEFAULT_MAX_ERROR_BOUND,
        }
    }

    /// Set the finite-difference epsilon.
    pub fn with_epsilon(mut self, eps: f64) -> Self {
        self.epsilon = eps;
        self
    }

    /// Set the maximum acceptable linearization error.
    pub fn with_max_error_bound(mut self, bound: f64) -> Self {
        self.max_error_bound = bound;
        self
    }

    /// Linearize the FK of `chain` around `(ref_params, ref_config)`.
    ///
    /// Returns a [`LinearizedModel`] whose error bound is checked against
    /// `max_error_bound`.
    pub fn linearize(
        &self,
        chain: &KinematicChain,
        ref_params: &BodyParameters,
        ref_config: &[f64],
    ) -> Result<LinearizedModel, VerifierError> {
        let num_joints = chain.joints.len();
        if ref_config.len() != num_joints {
            return Err(VerifierError::DimensionMismatch {
                expected: num_joints,
                got: ref_config.len(),
            });
        }

        let reference_position = self.evaluate_fk(chain, ref_params, ref_config)?;
        let jacobian_theta = self.compute_jacobian_params(chain, ref_params, ref_config)?;
        let jacobian_q = self.compute_jacobian_config(chain, ref_params, ref_config)?;
        let error_bound = self.compute_error_bound(chain, ref_params, num_joints);

        if error_bound > self.max_error_bound {
            return Err(VerifierError::LinearizationError {
                actual: error_bound,
                bound: self.max_error_bound,
            });
        }

        let reference_params_vec = params_to_dvector(ref_params);
        let reference_config_vec = DVector::from_column_slice(ref_config);

        let link_lengths: Vec<f64> = chain
            .joints
            .iter()
            .map(|j| j.effective_link_length(ref_params))
            .collect();

        Ok(LinearizedModel {
            jacobian_theta,
            jacobian_q,
            reference_params: reference_params_vec,
            reference_config: reference_config_vec,
            reference_position,
            error_bound,
            num_joints,
            num_params: NUM_PARAMS,
            link_lengths,
        })
    }

    /// Numerical Jacobian of FK w.r.t. the five body parameters.
    ///
    /// Uses central finite differences:
    /// `∂FK/∂θ_j ≈ (FK(θ₀ + εe_j, q₀) − FK(θ₀ − εe_j, q₀)) / (2ε)`.
    pub fn compute_jacobian_params(
        &self,
        chain: &KinematicChain,
        ref_params: &BodyParameters,
        ref_config: &[f64],
    ) -> Result<DMatrix<f64>, VerifierError> {
        let mut jac = DMatrix::zeros(SPATIAL_DIM, NUM_PARAMS);
        let base = ref_params.to_array();

        for j in 0..NUM_PARAMS {
            let mut plus = base;
            let mut minus = base;
            plus[j] += self.epsilon;
            minus[j] -= self.epsilon;

            let params_plus = BodyParameters::from_array(&plus);
            let params_minus = BodyParameters::from_array(&minus);

            let fk_plus = self.evaluate_fk(chain, &params_plus, ref_config)?;
            let fk_minus = self.evaluate_fk(chain, &params_minus, ref_config)?;

            let deriv = (fk_plus - fk_minus) / (2.0 * self.epsilon);
            for i in 0..SPATIAL_DIM {
                jac[(i, j)] = deriv[i];
            }
        }

        Ok(jac)
    }

    /// Numerical Jacobian of FK w.r.t. the joint angles.
    ///
    /// Uses central finite differences:
    /// `∂FK/∂q_k ≈ (FK(θ₀, q₀ + εe_k) − FK(θ₀, q₀ − εe_k)) / (2ε)`.
    pub fn compute_jacobian_config(
        &self,
        chain: &KinematicChain,
        ref_params: &BodyParameters,
        ref_config: &[f64],
    ) -> Result<DMatrix<f64>, VerifierError> {
        let num_joints = ref_config.len();
        let mut jac = DMatrix::zeros(SPATIAL_DIM, num_joints);

        for k in 0..num_joints {
            let mut q_plus = ref_config.to_vec();
            let mut q_minus = ref_config.to_vec();
            q_plus[k] += self.epsilon;
            q_minus[k] -= self.epsilon;

            let fk_plus = self.evaluate_fk(chain, ref_params, &q_plus)?;
            let fk_minus = self.evaluate_fk(chain, ref_params, &q_minus)?;

            let deriv = (fk_plus - fk_minus) / (2.0 * self.epsilon);
            for i in 0..SPATIAL_DIM {
                jac[(i, k)] = deriv[i];
            }
        }

        Ok(jac)
    }

    /// Compute the soundness envelope from Theorem C2.
    ///
    /// ```text
    /// error ≤ C_FK · (Δq² + Δθ · Δq) · L_sum
    /// ```
    ///
    /// with `C_FK = n / 2`, `Δq = ENVELOPE_SCALE`, `Δθ = ENVELOPE_SCALE`, and
    /// `L_sum = Σ link lengths`.
    pub fn compute_error_bound(
        &self,
        chain: &KinematicChain,
        ref_params: &BodyParameters,
        num_joints: usize,
    ) -> f64 {
        let c_fk = num_joints as f64 / 2.0;
        let l_sum: f64 = chain
            .joints
            .iter()
            .map(|j| j.effective_link_length(ref_params))
            .sum();

        let delta_q = ENVELOPE_SCALE;
        let delta_theta = ENVELOPE_SCALE;

        c_fk * (delta_q * delta_q + delta_theta * delta_q) * l_sum
    }

    /// Evaluate forward kinematics for the chain.
    ///
    /// Computes the end-effector position by multiplying the base transform,
    /// then each joint's static transform and rotation, accumulating link
    /// translations along the local z-axis (DH-like convention).
    pub fn evaluate_fk(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        config: &[f64],
    ) -> Result<Vector3<f64>, VerifierError> {
        if config.len() != chain.joints.len() {
            return Err(VerifierError::DimensionMismatch {
                expected: chain.joints.len(),
                got: config.len(),
            });
        }

        // Start with the body-parameter-dependent base position.
        let base_pos = chain.base_position(params);
        let mut transform = Matrix4::identity();
        transform[(0, 3)] = base_pos[0];
        transform[(1, 3)] = base_pos[1];
        transform[(2, 3)] = base_pos[2];

        // Walk down the chain, accumulating transforms.
        for (idx, joint) in chain.joints.iter().enumerate() {
            // Apply the joint's static (parent-to-joint) transform.
            let static_tf = matrix4_from_array(&joint.static_transform);
            transform *= static_tf;

            // Apply the joint rotation at the given angle.
            let angle = config[idx];
            let rotation = build_joint_transform(joint, angle);
            transform *= rotation;

            // Translate along the local z-axis by the effective link length.
            let link_len = joint.effective_link_length(params);
            if link_len.abs() > 1e-15 {
                let mut link_tf = Matrix4::identity();
                link_tf[(2, 3)] = link_len;
                transform *= link_tf;
            }
        }

        // Extract the end-effector position from the last column.
        let tip = transform * Vector4::new(0.0, 0.0, 0.0, 1.0);
        Ok(Vector3::new(tip[0], tip[1], tip[2]))
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Build a 4×4 homogeneous rotation matrix for a single joint at `angle`.
///
/// The rotation axis is determined by the joint type (`RevoluteX`, `RevoluteY`,
/// `RevoluteZ`). `Ball` and `Fixed` joints produce the identity.
pub fn build_joint_transform(joint: &Joint, angle: f64) -> Matrix4<f64> {
    let (s, c) = angle.sin_cos();
    match joint.joint_type {
        JointType::RevoluteX => Matrix4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0,   c,  -s, 0.0,
            0.0,   s,   c, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ),
        JointType::RevoluteY => Matrix4::new(
              c, 0.0,   s, 0.0,
            0.0, 1.0, 0.0, 0.0,
             -s, 0.0,   c, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ),
        JointType::RevoluteZ => Matrix4::new(
              c,  -s, 0.0, 0.0,
              s,   c, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ),
        // Ball and Fixed joints do not contribute a rotation here.
        // Ball joints are decomposed into three revolute joints upstream.
        JointType::Ball | JointType::Fixed => Matrix4::identity(),
    }
}

/// Convert [`BodyParameters`] to a [`DVector<f64>`] of length 5.
///
/// Order: stature, arm_length, shoulder_breadth, forearm_length, hand_length.
pub fn params_to_dvector(params: &BodyParameters) -> DVector<f64> {
    let arr = params.to_array();
    DVector::from_column_slice(&arr)
}

/// Reconstruct [`BodyParameters`] from a [`DVector<f64>`].
#[allow(dead_code)]
fn dvector_to_params(v: &DVector<f64>) -> BodyParameters {
    BodyParameters::new(v[0], v[1], v[2], v[3], v[4])
}

/// Convert a row-major `[f64; 16]` into a [`Matrix4<f64>`].
///
/// The `xr_types` crate stores 4×4 transforms in row-major order.
fn matrix4_from_array(arr: &[f64; 16]) -> Matrix4<f64> {
    #[rustfmt::skip]
    let m = Matrix4::new(
        arr[0],  arr[1],  arr[2],  arr[3],
        arr[4],  arr[5],  arr[6],  arr[7],
        arr[8],  arr[9],  arr[10], arr[11],
        arr[12], arr[13], arr[14], arr[15],
    );
    m
}

/// Fold a list of [`SmtExpr`] terms into a single sum expression.
///
/// An empty list yields `0.0`; a single element yields itself.
fn fold_sum(terms: Vec<SmtExpr>) -> SmtExpr {
    match terms.len() {
        0 => SmtExpr::real(0.0),
        1 => terms.into_iter().next().unwrap(),
        _ => {
            let mut it = terms.into_iter();
            let first = it.next().unwrap();
            it.fold(first, |acc, t| SmtExpr::add(acc, t))
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;
    use xr_types::{ArmSide, BodyParameters, KinematicChain};

    /// Helper: build a default right-arm chain and average-male params.
    fn default_fixtures() -> (KinematicChain, BodyParameters) {
        (
            KinematicChain::default_arm(ArmSide::Right),
            BodyParameters::average_male(),
        )
    }

    #[test]
    fn test_linearized_model_evaluate() {
        let (chain, params) = default_fixtures();
        let config = chain.midpoint_config(&params);
        let engine = LinearizationEngine::new().with_max_error_bound(1.0);

        let model = engine.linearize(&chain, &params, &config).unwrap();

        // At the reference point the linear approximation must be exact.
        let ref_p = params_to_dvector(&params);
        let ref_q = DVector::from_column_slice(&config);
        let pos = model.evaluate_linear(&ref_p, &ref_q).unwrap();

        let diff = (pos - model.reference_position).norm();
        assert!(
            diff < 1e-12,
            "At reference point, linear eval should match FK exactly; diff = {diff}"
        );
    }

    #[test]
    fn test_linearization_engine_basic() {
        let (chain, params) = default_fixtures();
        let config = chain.midpoint_config(&params);
        let engine = LinearizationEngine::new().with_max_error_bound(1.0);

        let model = engine.linearize(&chain, &params, &config).unwrap();

        assert_eq!(model.num_joints, chain.joints.len());
        assert_eq!(model.num_params, NUM_PARAMS);
        assert_eq!(model.jacobian_theta.nrows(), SPATIAL_DIM);
        assert_eq!(model.jacobian_theta.ncols(), NUM_PARAMS);
        assert_eq!(model.jacobian_q.nrows(), SPATIAL_DIM);
        assert_eq!(model.jacobian_q.ncols(), model.num_joints);
        assert_eq!(model.link_lengths.len(), model.num_joints);
        assert!(model.error_bound >= 0.0);
    }

    #[test]
    fn test_jacobian_symmetry() {
        // The Jacobian J_q should be non-trivial for a reasonable mid-range
        // configuration, and contain no NaN / Inf entries.
        let (chain, params) = default_fixtures();
        let config = chain.midpoint_config(&params);
        let engine = LinearizationEngine::new();

        let jac_q = engine
            .compute_jacobian_config(&chain, &params, &config)
            .unwrap();

        for val in jac_q.iter() {
            assert!(val.is_finite(), "Jacobian contains non-finite value: {val}");
        }

        let mut any_nonzero = false;
        for k in 0..jac_q.ncols() {
            let col_norm: f64 = (0..SPATIAL_DIM)
                .map(|i| jac_q[(i, k)].powi(2))
                .sum::<f64>()
                .sqrt();
            if col_norm > 1e-6 {
                any_nonzero = true;
            }
        }
        assert!(
            any_nonzero,
            "Jacobian J_q is all-zero at midpoint — chain may be degenerate"
        );
    }

    #[test]
    fn test_error_bound_computation() {
        let (chain, params) = default_fixtures();
        let n = chain.joints.len();
        let engine = LinearizationEngine::new();

        let bound = engine.compute_error_bound(&chain, &params, n);

        let c_fk = n as f64 / 2.0;
        let l_sum = chain.total_link_length(&params);
        let expected =
            c_fk * (ENVELOPE_SCALE.powi(2) + ENVELOPE_SCALE * ENVELOPE_SCALE) * l_sum;

        assert!(
            (bound - expected).abs() < 1e-12,
            "Error bound {bound} != expected {expected}"
        );
        assert!(bound > 0.0, "Error bound should be positive");
    }

    #[test]
    fn test_smt_expression_generation() {
        let (chain, params) = default_fixtures();
        let config = chain.midpoint_config(&params);
        let engine = LinearizationEngine::new().with_max_error_bound(1.0);

        let model = engine.linearize(&chain, &params, &config).unwrap();
        let exprs = model.to_smt_expressions("theta", "q");

        assert_eq!(exprs.len(), SPATIAL_DIM);

        for (i, expr) in exprs.iter().enumerate() {
            let fv = expr.free_variables();
            let has_theta = fv.iter().any(|v| v.starts_with("theta_"));
            let has_q = fv.iter().any(|v| v.starts_with("q_"));
            assert!(
                has_theta || has_q,
                "Coordinate {i} expression has no free variables"
            );
        }
    }

    #[test]
    fn test_fk_evaluation() {
        let (chain, params) = default_fixtures();
        let engine = LinearizationEngine::new();

        let zero_config = vec![0.0; chain.joints.len()];
        let pos_zero = engine.evaluate_fk(&chain, &params, &zero_config).unwrap();
        for i in 0..3 {
            assert!(
                pos_zero[i].is_finite(),
                "FK position component {i} is not finite"
            );
        }
        assert!(
            pos_zero.norm() > 0.1,
            "FK at zero config unexpectedly near origin: {pos_zero}"
        );

        let mut perturbed = zero_config.clone();
        perturbed[0] = 0.1;
        let pos_perturbed = engine.evaluate_fk(&chain, &params, &perturbed).unwrap();
        let delta = (pos_perturbed - pos_zero).norm();
        assert!(
            delta > 1e-6,
            "Perturbing joint 0 by 0.1 rad should move the end-effector"
        );
        assert!(
            delta < 2.0,
            "Perturbation moved end-effector unreasonably far: {delta} m"
        );
    }

    #[test]
    fn test_linearization_accuracy() {
        let (chain, params) = default_fixtures();
        let config = chain.midpoint_config(&params);
        let engine = LinearizationEngine::new().with_max_error_bound(1.0);

        let model = engine.linearize(&chain, &params, &config).unwrap();

        let mut q_pert = config.clone();
        q_pert[3] += 0.02;

        let fk_true = engine.evaluate_fk(&chain, &params, &q_pert).unwrap();

        let ref_p = params_to_dvector(&params);
        let q_vec = DVector::from_column_slice(&q_pert);
        let fk_lin = model.evaluate_linear(&ref_p, &q_vec).unwrap();

        let approx_error = (fk_true - fk_lin).norm();
        assert!(
            approx_error < 0.01,
            "Linearization error {approx_error} too large for a 0.02 rad perturbation"
        );
    }

    #[test]
    fn test_dimension_mismatch_errors() {
        let (chain, params) = default_fixtures();
        let engine = LinearizationEngine::new();

        let bad_config = vec![0.0; chain.joints.len() + 1];
        let result = engine.evaluate_fk(&chain, &params, &bad_config);
        assert!(result.is_err());
    }

    #[test]
    fn test_envelope_check() {
        let (chain, params) = default_fixtures();
        let config = chain.midpoint_config(&params);
        let engine = LinearizationEngine::new().with_max_error_bound(1.0);

        let model = engine.linearize(&chain, &params, &config).unwrap();

        let ref_p = params_to_dvector(&params);
        let ref_q = DVector::from_column_slice(&config);

        assert!(model.is_within_error_envelope(&ref_p, &ref_q));

        let far_q = DVector::from_element(model.num_joints, 100.0);
        assert!(!model.is_within_error_envelope(&ref_p, &far_q));
    }
}

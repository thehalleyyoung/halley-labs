//! Accessibility SMT encoder.
//!
//! Converts kinematic accessibility predicates into QF_LRA SMT expressions.
//! The encoder linearizes forward kinematics around a reference configuration
//! and produces constraints over joint-angle and body-parameter perturbation
//! variables.  Non-linear activation volumes (spheres, capsules, cylinders) are
//! conservatively over-approximated by their axis-aligned bounding boxes so
//! that every generated constraint is a linear inequality.

use crate::expr::{SmtDecl, SmtExpr, SmtSort};
use xr_types::{
    BoundingBox, BodyParameters, Capsule, DeviceConfig,
    InteractableElement, KinematicChain, Sphere, Volume,
};
use xr_types::geometry::Cylinder;

// ---------------------------------------------------------------------------
// AccessibilityEncoder
// ---------------------------------------------------------------------------

/// Top-level encoder that translates accessibility predicates into SMT.
#[derive(Debug, Clone)]
pub struct AccessibilityEncoder {
    /// Numerical tolerance for zero-checks.
    epsilon: f64,
    /// Accumulated variable declarations.
    declarations: Vec<SmtDecl>,
    /// Monotone counter for generating unique variable names.
    var_counter: usize,
}

impl Default for AccessibilityEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl AccessibilityEncoder {
    /// Create a new encoder with default epsilon.
    pub fn new() -> Self {
        Self {
            epsilon: 1e-8,
            declarations: Vec::new(),
            var_counter: 0,
        }
    }

    /// Builder: override epsilon.
    pub fn with_epsilon(mut self, eps: f64) -> Self {
        self.epsilon = eps;
        self
    }

    /// Return current epsilon.
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Mint a fresh SMT variable with a unique name derived from `prefix`.
    pub fn fresh_var(&mut self, prefix: &str, sort: SmtSort) -> SmtExpr {
        self.var_counter += 1;
        let name = format!("{}{}", prefix, self.var_counter);
        self.declarations.push(SmtDecl::new(&name, sort.clone()));
        SmtExpr::var(name)
    }

    /// All variable declarations accumulated so far.
    pub fn declarations(&self) -> &[SmtDecl] {
        &self.declarations
    }

    /// Reset declarations and counter.
    pub fn reset(&mut self) {
        self.declarations.clear();
        self.var_counter = 0;
    }

    // -----------------------------------------------------------------------
    // High-level encoding
    // -----------------------------------------------------------------------

    /// Encode a full reachability predicate for `element` with respect to a
    /// kinematic chain, reference body parameters, and reference joint angles.
    ///
    /// Returns a conjunction of:
    /// - linearised FK endpoint ∈ activation volume
    /// - joint-angle perturbation variables within limits
    /// - body-parameter perturbation variables within ±20% of reference
    pub fn encode_reachability(
        &mut self,
        element: &InteractableElement,
        chain: &KinematicChain,
        params: &BodyParameters,
        reference_angles: &[f64],
    ) -> SmtExpr {
        let nj = chain.joints.len().min(reference_angles.len());

        // Fresh joint-angle variables
        let avars: Vec<SmtExpr> = (0..nj)
            .map(|i| self.fresh_var(&format!("q{}", i), SmtSort::Real))
            .collect();

        // Fresh body-parameter perturbation variables
        let dpvars: [SmtExpr; 5] = std::array::from_fn(|i| {
            self.fresh_var(&format!("dp{}", i), SmtSort::Real)
        });

        // Compute joint-angle Jacobian via central differences
        let h = 1e-5;
        let bp = simple_fk(chain, params, reference_angles);
        let mut jq = vec![[0.0_f64; 3]; nj];
        for i in 0..nj {
            let mut ap = reference_angles.to_vec();
            let mut am = reference_angles.to_vec();
            ap[i] += h;
            am[i] -= h;
            let pp = simple_fk(chain, params, &ap);
            let pm = simple_fk(chain, params, &am);
            for a in 0..3 {
                jq[i][a] = (pp[a] - pm[a]) / (2.0 * h);
            }
        }

        // Compute body-parameter Jacobian
        let jp = compute_param_jac(chain, params, reference_angles, h);

        // Build linearised position variables: pv[ax] = bp[ax] + Jq·Δq + Jp·Δp
        let pv: [SmtExpr; 3] = std::array::from_fn(|ax| {
            let mut terms: Vec<SmtExpr> = vec![SmtExpr::real(bp[ax])];
            for j in 0..nj {
                if jq[j][ax].abs() > 1e-12 {
                    terms.push(SmtExpr::mul(
                        SmtExpr::real(jq[j][ax]),
                        SmtExpr::sub(avars[j].clone(), SmtExpr::real(reference_angles[j])),
                    ));
                }
            }
            for p in 0..5 {
                if jp[p][ax].abs() > 1e-12 {
                    terms.push(SmtExpr::mul(
                        SmtExpr::real(jp[p][ax]),
                        dpvars[p].clone(),
                    ));
                }
            }
            SmtExpr::sum(terms)
        });

        // Activation volume containment
        let vc = self.encode_activation_volume(&element.activation_volume, &pv);

        // Joint limit constraints on the angle variables
        let jc = self.encode_joint_limit_vars(chain, params, &avars);

        // Body-parameter perturbation bounds (±20% of reference values)
        let pa = params.to_array();
        let pb: Vec<SmtExpr> = (0..5)
            .flat_map(|i| {
                let b = (pa[i] * 0.2).abs().max(0.01);
                vec![
                    SmtExpr::ge(dpvars[i].clone(), SmtExpr::real(-b)),
                    SmtExpr::le(dpvars[i].clone(), SmtExpr::real(b)),
                ]
            })
            .collect();

        // Conjunction of all sub-constraints
        let mut all = vec![vc, jc];
        all.extend(pb);
        SmtExpr::and(all)
    }

    /// Encode joint limits using automatically-named variables `q0..qN`.
    pub fn encode_joint_limits(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
    ) -> SmtExpr {
        let cs: Vec<SmtExpr> = chain
            .joints
            .iter()
            .enumerate()
            .flat_map(|(i, j)| {
                let lim = j.effective_limits(params);
                let v = SmtExpr::var(format!("q{}", i));
                vec![
                    SmtExpr::ge(v.clone(), SmtExpr::real(lim.min)),
                    SmtExpr::le(v, SmtExpr::real(lim.max)),
                ]
            })
            .collect();
        if cs.is_empty() {
            SmtExpr::BoolConst(true)
        } else {
            SmtExpr::and(cs)
        }
    }

    /// Encode joint limits for explicit angle variables.
    pub fn encode_joint_limit_vars(
        &self,
        chain: &KinematicChain,
        params: &BodyParameters,
        avars: &[SmtExpr],
    ) -> SmtExpr {
        let cs: Vec<SmtExpr> = chain
            .joints
            .iter()
            .enumerate()
            .filter(|(i, _)| *i < avars.len())
            .flat_map(|(i, j)| {
                let lim = j.effective_limits(params);
                vec![
                    SmtExpr::ge(avars[i].clone(), SmtExpr::real(lim.min)),
                    SmtExpr::le(avars[i].clone(), SmtExpr::real(lim.max)),
                ]
            })
            .collect();
        if cs.is_empty() {
            SmtExpr::BoolConst(true)
        } else {
            SmtExpr::and(cs)
        }
    }

    // -----------------------------------------------------------------------
    // Volume encoding
    // -----------------------------------------------------------------------

    /// Encode containment of the position vector `pv` inside `volume`.
    ///
    /// Non-linear volumes are over-approximated by their bounding box.
    pub fn encode_activation_volume(
        &self,
        volume: &Volume,
        pv: &[SmtExpr; 3],
    ) -> SmtExpr {
        match volume {
            Volume::Box(b) => encode_box_containment(b, pv),
            Volume::Sphere(s) => encode_sphere_approx(s, pv),
            Volume::Capsule(c) => encode_capsule_approx(c, pv),
            Volume::Cylinder(c) => encode_cylinder_approx(c, pv),
            Volume::ConvexHull(h) => encode_box_containment(&h.bounding_box(), pv),
            Volume::Composite(vs) => {
                if vs.is_empty() {
                    return SmtExpr::BoolConst(false);
                }
                let sub: Vec<SmtExpr> = vs
                    .iter()
                    .map(|v| self.encode_activation_volume(v, pv))
                    .collect();
                if sub.len() == 1 {
                    sub.into_iter().next().unwrap()
                } else {
                    SmtExpr::or(sub)
                }
            }
        }
    }

    /// Encode that `pv` is inside a device's tracking volume and outside
    /// all dead zones.
    pub fn encode_tracking_volume(
        &self,
        dev: &DeviceConfig,
        pv: &[SmtExpr; 3],
    ) -> SmtExpr {
        let shape = self.encode_activation_volume(&dev.tracking_volume.shape, pv);
        let dead_zone_constraints: Vec<SmtExpr> = dev
            .tracking_volume
            .dead_zones
            .iter()
            .map(|d| SmtExpr::not(self.encode_activation_volume(d, pv)))
            .collect();
        let mut all = vec![shape];
        all.extend(dead_zone_constraints);
        SmtExpr::and(all)
    }

    /// Encode that orientation variables `ov` (roll, pitch, yaw) are within
    /// `tol` radians of the Euler angles derived from `quat`.
    pub fn encode_orientation_constraint(
        &self,
        quat: &[f64; 4],
        tol: f64,
        ov: &[SmtExpr; 3],
    ) -> SmtExpr {
        let (r, p, y) = quat_to_euler(quat);
        let targets = [r, p, y];
        let cs: Vec<SmtExpr> = (0..3)
            .flat_map(|i| {
                vec![
                    SmtExpr::ge(ov[i].clone(), SmtExpr::real(targets[i] - tol)),
                    SmtExpr::le(ov[i].clone(), SmtExpr::real(targets[i] + tol)),
                ]
            })
            .collect();
        SmtExpr::and(cs)
    }

    // -----------------------------------------------------------------------
    // Linearised FK helper
    // -----------------------------------------------------------------------

    /// Encode linearised FK: position[ax] = bp[ax] + Σ_j jac[ax][j] · avars[j].
    pub fn encode_linearized_fk(
        &self,
        jac: &[[f64; 7]; 3],
        bp: &[f64; 3],
        avars: &[SmtExpr],
    ) -> [SmtExpr; 3] {
        std::array::from_fn(|ax| {
            let mut terms: Vec<SmtExpr> = vec![SmtExpr::real(bp[ax])];
            for (j, v) in avars.iter().enumerate() {
                if j < 7 && jac[ax][j].abs() > 1e-12 {
                    terms.push(SmtExpr::mul(SmtExpr::real(jac[ax][j]), v.clone()));
                }
            }
            SmtExpr::sum(terms)
        })
    }

    // -----------------------------------------------------------------------
    // Bounds helpers
    // -----------------------------------------------------------------------

    /// Encode per-component bounds on parameter variables.
    pub fn encode_parameter_bounds(
        &self,
        pvars: &[SmtExpr; 5],
        lo: &[f64; 5],
        hi: &[f64; 5],
    ) -> SmtExpr {
        let cs: Vec<SmtExpr> = (0..5)
            .flat_map(|i| {
                vec![
                    SmtExpr::ge(pvars[i].clone(), SmtExpr::real(lo[i])),
                    SmtExpr::le(pvars[i].clone(), SmtExpr::real(hi[i])),
                ]
            })
            .collect();
        SmtExpr::and(cs)
    }

    /// Encode an L1-style bound: each expression must lie within ±(bound / n).
    pub fn encode_l1_bound(&self, exprs: &[SmtExpr], bound: f64) -> SmtExpr {
        if exprs.is_empty() {
            return SmtExpr::BoolConst(true);
        }
        let p = bound / exprs.len() as f64;
        let cs: Vec<SmtExpr> = exprs
            .iter()
            .flat_map(|e| {
                vec![
                    SmtExpr::ge(e.clone(), SmtExpr::real(-p)),
                    SmtExpr::le(e.clone(), SmtExpr::real(p)),
                ]
            })
            .collect();
        SmtExpr::and(cs)
    }

    // -----------------------------------------------------------------------
    // SMT-LIB2 output
    // -----------------------------------------------------------------------

    /// Emit all declarations as SMT-LIB2 with a QF_LRA logic header.
    pub fn to_smtlib2_declarations(&self) -> String {
        let mut out = String::from("(set-logic QF_LRA)\n");
        for d in &self.declarations {
            out.push_str(&d.to_smtlib2());
            out.push('\n');
        }
        out
    }

    /// Encode a full SMT-LIB2 check-sat script for an element.
    pub fn encode_full_check(
        &mut self,
        elem: &InteractableElement,
        chain: &KinematicChain,
        params: &BodyParameters,
        ref_angles: &[f64],
        dev: Option<&DeviceConfig>,
    ) -> String {
        let reach = self.encode_reachability(elem, chain, params, ref_angles);
        let mut assertions = vec![reach];
        if let Some(d) = dev {
            let pv: [SmtExpr; 3] = std::array::from_fn(|ax| {
                self.fresh_var(&format!("tv{}", ax), SmtSort::Real)
            });
            assertions.push(self.encode_tracking_volume(d, &pv));
        }
        let mut out = self.to_smtlib2_declarations();
        for a in &assertions {
            out.push_str(&format!("(assert {})\n", a.to_smtlib2()));
        }
        out.push_str("(check-sat)\n(get-model)\n");
        out
    }
}

// ---------------------------------------------------------------------------
// Free-standing volume encoders
// ---------------------------------------------------------------------------

/// Encode axis-aligned bounding-box containment as a conjunction of 6 linear
/// inequalities.
pub fn encode_box_containment(b: &BoundingBox, p: &[SmtExpr; 3]) -> SmtExpr {
    let cs: Vec<SmtExpr> = (0..3)
        .flat_map(|i| {
            vec![
                SmtExpr::ge(p[i].clone(), SmtExpr::real(b.min[i])),
                SmtExpr::le(p[i].clone(), SmtExpr::real(b.max[i])),
            ]
        })
        .collect();
    SmtExpr::and(cs)
}

/// Over-approximate sphere containment by its bounding box.
pub fn encode_sphere_approx(s: &Sphere, p: &[SmtExpr; 3]) -> SmtExpr {
    encode_box_containment(&s.bounding_box(), p)
}

/// Over-approximate capsule containment by its bounding box.
pub fn encode_capsule_approx(c: &Capsule, p: &[SmtExpr; 3]) -> SmtExpr {
    encode_box_containment(&c.bounding_box(), p)
}

/// Over-approximate cylinder containment by its bounding box.
pub fn encode_cylinder_approx(c: &Cylinder, p: &[SmtExpr; 3]) -> SmtExpr {
    encode_box_containment(&c.bounding_box(), p)
}

// ---------------------------------------------------------------------------
// Simple FK helpers
// ---------------------------------------------------------------------------

/// Evaluate forward kinematics with Rodrigues rotation (no nalgebra dependency).
fn simple_fk(
    chain: &KinematicChain,
    params: &BodyParameters,
    angles: &[f64],
) -> [f64; 3] {
    let mut pos = chain.base_position(params);
    let mut rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for (i, jt) in chain.joints.iter().enumerate() {
        let a = if i < angles.len() { angles[i] } else { 0.0 };
        let lk = jt.effective_link_length(params);
        let ax = jt.joint_type.axis();
        rot = rotm(&rot, &ax, a);
        if lk.abs() > 1e-12 {
            for x in 0..3 {
                pos[x] += rot[x][0] * lk;
            }
        }
    }
    pos
}

/// Compute body-parameter Jacobian via central differences.
fn compute_param_jac(
    chain: &KinematicChain,
    params: &BodyParameters,
    angles: &[f64],
    h: f64,
) -> [[f64; 3]; 5] {
    let mut jac = [[0.0_f64; 3]; 5];
    let pa = params.to_array();
    for p in 0..5 {
        let mut ap = pa;
        let mut am = pa;
        ap[p] += h;
        am[p] -= h;
        let pp = simple_fk(chain, &array_to_bp(&ap), angles);
        let pm = simple_fk(chain, &array_to_bp(&am), angles);
        for a in 0..3 {
            jac[p][a] = (pp[a] - pm[a]) / (2.0 * h);
        }
    }
    jac
}

/// Reconstruct `BodyParameters` from a 5-element array.
fn array_to_bp(a: &[f64; 5]) -> BodyParameters {
    BodyParameters::new(a[0], a[1], a[2], a[3], a[4])
}

/// Rodrigues rotation: R_new = R_old · Rot(axis, angle).
fn rotm(rot: &[[f64; 3]; 3], axis: &[f64; 3], ang: f64) -> [[f64; 3]; 3] {
    let ls = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
    if ls < 1e-12 || ang.abs() < 1e-15 {
        return *rot;
    }
    let l = ls.sqrt();
    let (ux, uy, uz) = (axis[0] / l, axis[1] / l, axis[2] / l);
    let c = ang.cos();
    let s = ang.sin();
    let t = 1.0 - c;
    let r = [
        [t * ux * ux + c, t * ux * uy - s * uz, t * ux * uz + s * uy],
        [t * uy * ux + s * uz, t * uy * uy + c, t * uy * uz - s * ux],
        [t * uz * ux - s * uy, t * uz * uy + s * ux, t * uz * uz + c],
    ];
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] =
                rot[i][0] * r[0][j] + rot[i][1] * r[1][j] + rot[i][2] * r[2][j];
        }
    }
    out
}

/// Convert quaternion [w,x,y,z] to Euler angles (roll, pitch, yaw).
fn quat_to_euler(q: &[f64; 4]) -> (f64, f64, f64) {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    let roll = (2.0 * (w * x + y * z)).atan2(1.0 - 2.0 * (x * x + y * y));
    let sp = 2.0 * (w * y - z * x);
    let pitch = if sp.abs() >= 1.0 {
        std::f64::consts::FRAC_PI_2.copysign(sp)
    } else {
        sp.asin()
    };
    let yaw = (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z));
    (roll, pitch, yaw)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use xr_types::kinematic::ArmSide;

    fn chain() -> KinematicChain {
        KinematicChain::default_arm(ArmSide::Right)
    }
    fn params() -> BodyParameters {
        BodyParameters::average_male()
    }

    #[test]
    fn test_new() {
        let enc = AccessibilityEncoder::new();
        assert!(enc.declarations().is_empty());
        assert!((enc.epsilon() - 1e-8).abs() < 1e-15);
    }

    #[test]
    fn test_with_epsilon() {
        let enc = AccessibilityEncoder::new().with_epsilon(1e-4);
        assert!((enc.epsilon() - 1e-4).abs() < 1e-15);
    }

    #[test]
    fn test_fresh_var_unique() {
        let mut enc = AccessibilityEncoder::new();
        let a = enc.fresh_var("q", SmtSort::Real);
        let b = enc.fresh_var("q", SmtSort::Real);
        assert_ne!(a, b);
        assert_eq!(enc.declarations().len(), 2);
    }

    #[test]
    fn test_reset() {
        let mut enc = AccessibilityEncoder::new();
        enc.fresh_var("x", SmtSort::Real);
        enc.fresh_var("y", SmtSort::Real);
        assert_eq!(enc.declarations().len(), 2);
        enc.reset();
        assert!(enc.declarations().is_empty());
    }

    #[test]
    fn test_box_containment() {
        let b = BoundingBox::new([0.0; 3], [1.0; 3]);
        let p = [SmtExpr::var("x"), SmtExpr::var("y"), SmtExpr::var("z")];
        let expr = encode_box_containment(&b, &p);
        let s = expr.to_smtlib2();
        assert!(s.contains("and"));
        assert!(s.contains("x"));
        assert!(s.contains("y"));
        assert!(s.contains("z"));
    }

    #[test]
    fn test_sphere_approx() {
        let s = Sphere::new([0.5; 3], 0.5);
        let p = [SmtExpr::var("x"), SmtExpr::var("y"), SmtExpr::var("z")];
        let expr = encode_sphere_approx(&s, &p);
        assert!(expr.to_smtlib2().contains("and"));
    }

    #[test]
    fn test_capsule_approx() {
        let c = Capsule::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.1);
        let p = [SmtExpr::var("x"), SmtExpr::var("y"), SmtExpr::var("z")];
        let expr = encode_capsule_approx(&c, &p);
        assert!(expr.to_smtlib2().contains("and"));
    }

    #[test]
    fn test_cylinder_approx() {
        let c = Cylinder::new([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.5, 1.0);
        let p = [SmtExpr::var("x"), SmtExpr::var("y"), SmtExpr::var("z")];
        let expr = encode_cylinder_approx(&c, &p);
        assert!(expr.to_smtlib2().contains("and"));
    }

    #[test]
    fn test_joint_limits() {
        let c = chain();
        let p = params();
        let s = AccessibilityEncoder::new()
            .encode_joint_limits(&c, &p)
            .to_smtlib2();
        assert!(s.contains("q0"));
        assert!(s.contains(&format!("q{}", c.joints.len() - 1)));
    }

    #[test]
    fn test_joint_limit_vars() {
        let c = chain();
        let p = params();
        let avars: Vec<SmtExpr> = (0..c.joints.len())
            .map(|i| SmtExpr::var(format!("a{}", i)))
            .collect();
        let expr = AccessibilityEncoder::new().encode_joint_limit_vars(&c, &p, &avars);
        let s = expr.to_smtlib2();
        assert!(s.contains("a0"));
    }

    #[test]
    fn test_activation_volume_composite() {
        let v = Volume::Composite(vec![
            Volume::Box(BoundingBox::new([0.0; 3], [1.0; 3])),
            Volume::Box(BoundingBox::new([2.0; 3], [3.0; 3])),
        ]);
        let p = [SmtExpr::var("x"), SmtExpr::var("y"), SmtExpr::var("z")];
        let expr = AccessibilityEncoder::new().encode_activation_volume(&v, &p);
        assert!(expr.to_smtlib2().contains("or"));
    }

    #[test]
    fn test_activation_volume_empty_composite() {
        let v = Volume::Composite(vec![]);
        let p = [SmtExpr::var("x"), SmtExpr::var("y"), SmtExpr::var("z")];
        let expr = AccessibilityEncoder::new().encode_activation_volume(&v, &p);
        assert_eq!(expr, SmtExpr::BoolConst(false));
    }

    #[test]
    fn test_tracking_volume() {
        let d = DeviceConfig::quest_3();
        let p = [SmtExpr::var("x"), SmtExpr::var("y"), SmtExpr::var("z")];
        let expr = AccessibilityEncoder::new().encode_tracking_volume(&d, &p);
        assert!(expr.to_smtlib2().len() > 10);
    }

    #[test]
    fn test_parameter_bounds() {
        let pvars: [SmtExpr; 5] =
            std::array::from_fn(|i| SmtExpr::var(format!("p{}", i)));
        let expr = AccessibilityEncoder::new()
            .encode_parameter_bounds(&pvars, &[0.0; 5], &[2.0; 5]);
        let s = expr.to_smtlib2();
        assert!(s.contains("p0"));
        assert!(s.contains("p4"));
    }

    #[test]
    fn test_orientation_constraint() {
        let ov = [SmtExpr::var("r"), SmtExpr::var("p"), SmtExpr::var("y")];
        let expr = AccessibilityEncoder::new()
            .encode_orientation_constraint(&[1.0, 0.0, 0.0, 0.0], 0.1, &ov);
        let s = expr.to_smtlib2();
        assert!(s.contains("r"));
        assert!(s.contains("p"));
    }

    #[test]
    fn test_linearized_fk() {
        let jac = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let av: Vec<SmtExpr> = (0..7).map(|i| SmtExpr::var(format!("q{}", i))).collect();
        let fk = AccessibilityEncoder::new().encode_linearized_fk(&jac, &[0.5, 1.5, 0.0], &av);
        assert!(fk[0].to_smtlib2().contains("q0"));
        assert!(fk[1].to_smtlib2().contains("q1"));
        assert!(fk[2].to_smtlib2().contains("q2"));
    }

    #[test]
    fn test_l1_bound() {
        let exprs = vec![SmtExpr::var("a"), SmtExpr::var("b")];
        let expr = AccessibilityEncoder::new().encode_l1_bound(&exprs, 1.0);
        let s = expr.to_smtlib2();
        assert!(s.contains("a"));
        assert!(s.contains("b"));
    }

    #[test]
    fn test_l1_bound_empty() {
        let expr = AccessibilityEncoder::new().encode_l1_bound(&[], 1.0);
        assert_eq!(expr, SmtExpr::BoolConst(true));
    }

    #[test]
    fn test_simple_fk_finite() {
        let c = chain();
        let p = params();
        let angles = vec![0.0; c.joints.len()];
        let pos = simple_fk(&c, &p, &angles);
        assert!(pos.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_simple_fk_deterministic() {
        let c = chain();
        let p = params();
        let angles = vec![0.1, -0.2, 0.3, 0.0, 0.15, -0.1, 0.0];
        let p1 = simple_fk(&c, &p, &angles);
        let p2 = simple_fk(&c, &p, &angles);
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_quat_identity() {
        let (r, p, y) = quat_to_euler(&[1.0, 0.0, 0.0, 0.0]);
        assert!(r.abs() < 1e-10);
        assert!(p.abs() < 1e-10);
        assert!(y.abs() < 1e-10);
    }

    #[test]
    fn test_rotm_identity() {
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let result = rotm(&id, &[0.0, 0.0, 1.0], 0.0);
        for i in 0..3 {
            for j in 0..3 {
                assert!((result[i][j] - id[i][j]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_rotm_90_deg_z() {
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let result = rotm(&id, &[0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2);
        assert!((result[0][0]).abs() < 1e-10);
        assert!((result[1][0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_param_jac_finite() {
        let c = chain();
        let p = params();
        let angles = vec![0.0; c.joints.len()];
        let jac = compute_param_jac(&c, &p, &angles, 1e-5);
        for row in &jac {
            for &v in row {
                assert!(v.is_finite(), "Jacobian entry is not finite: {}", v);
            }
        }
    }

    #[test]
    fn test_smtlib2_declarations() {
        let mut enc = AccessibilityEncoder::new();
        enc.fresh_var("x", SmtSort::Real);
        enc.fresh_var("y", SmtSort::Bool);
        let s = enc.to_smtlib2_declarations();
        assert!(s.contains("set-logic QF_LRA"));
        assert!(s.contains("x1"));
        assert!(s.contains("y2"));
    }

    #[test]
    fn test_encode_reachability() {
        let c = chain();
        let p = params();
        let a = c.midpoint_config(&p);
        let elem = InteractableElement {
            id: uuid::Uuid::new_v4(),
            name: "btn".into(),
            position: [0.5, 1.5, 0.3],
            orientation: [1.0, 0.0, 0.0, 0.0],
            scale: [1.0; 3],
            activation_volume: Volume::Box(BoundingBox::new(
                [0.4, 1.4, 0.2],
                [0.6, 1.6, 0.4],
            )),
            interaction_type: xr_types::InteractionType::Click,
            actuator: xr_types::scene::ActuatorType::Hand,
            transform_node: None,
            min_duration: 0.0,
            sustained_contact: false,
            feedback_type: xr_types::scene::FeedbackType::Visual,
            priority: 1,
        };
        let mut enc = AccessibilityEncoder::new();
        let expr = enc.encode_reachability(&elem, &c, &p, &a);
        let s = expr.to_smtlib2();
        assert!(s.contains("and"));
        assert!(enc.declarations().len() >= 5);
    }

    #[test]
    fn test_encode_full_check() {
        let c = chain();
        let p = params();
        let a = c.midpoint_config(&p);
        let elem = InteractableElement {
            id: uuid::Uuid::new_v4(),
            name: "btn".into(),
            position: [0.5, 1.5, 0.3],
            orientation: [1.0, 0.0, 0.0, 0.0],
            scale: [1.0; 3],
            activation_volume: Volume::Box(BoundingBox::new(
                [0.4, 1.4, 0.2],
                [0.6, 1.6, 0.4],
            )),
            interaction_type: xr_types::InteractionType::Click,
            actuator: xr_types::scene::ActuatorType::Hand,
            transform_node: None,
            min_duration: 0.0,
            sustained_contact: false,
            feedback_type: xr_types::scene::FeedbackType::Visual,
            priority: 1,
        };
        let mut enc = AccessibilityEncoder::new();
        let script = enc.encode_full_check(&elem, &c, &p, &a, None);
        assert!(script.contains("set-logic QF_LRA"));
        assert!(script.contains("assert"));
        assert!(script.contains("check-sat"));
    }

    #[test]
    fn test_array_to_bp_roundtrip() {
        let orig = BodyParameters::average_male();
        let arr = orig.to_array();
        let restored = array_to_bp(&arr);
        assert!((restored.stature - orig.stature).abs() < 1e-12);
        assert!((restored.arm_length - orig.arm_length).abs() < 1e-12);
    }
}

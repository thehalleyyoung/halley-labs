//! Core traits for the XR Affordance Verifier pipeline.
//!
//! These traits define the fundamental abstractions used across
//! the verification system: verifiability, spatial queries,
//! transformability, certification, sampling, and SMT encoding.

use crate::error::VerifierResult;
use crate::geometry::BoundingBox;
use crate::{ElementId, ParamVec, Point3, Transform, Vector3};

// ---------------------------------------------------------------------------
// Verdict
// ---------------------------------------------------------------------------

/// Outcome of a single verification check.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Verdict {
    /// The property holds unconditionally.
    Pass,
    /// The property is violated with a witness configuration.
    Fail {
        /// Human-readable reason for the failure.
        reason: String,
        /// Optional witness body-parameter vector.
        witness: Option<Vec<f64>>,
    },
    /// The check could not determine the result within resource bounds.
    Unknown {
        /// Explanation of why the result is indeterminate.
        reason: String,
    },
}

impl Verdict {
    /// Returns `true` when the verdict is [`Verdict::Pass`].
    pub fn is_pass(&self) -> bool {
        matches!(self, Verdict::Pass)
    }

    /// Returns `true` when the verdict is [`Verdict::Fail`].
    pub fn is_fail(&self) -> bool {
        matches!(self, Verdict::Fail { .. })
    }

    /// Returns `true` when the verdict is [`Verdict::Unknown`].
    pub fn is_unknown(&self) -> bool {
        matches!(self, Verdict::Unknown { .. })
    }

    /// Merge two verdicts conservatively: Fail > Unknown > Pass.
    pub fn merge(&self, other: &Verdict) -> Verdict {
        match (self, other) {
            (Verdict::Fail { .. }, _) | (_, Verdict::Fail { .. }) => {
                if let Verdict::Fail { reason, witness } = self {
                    Verdict::Fail {
                        reason: reason.clone(),
                        witness: witness.clone(),
                    }
                } else if let Verdict::Fail { reason, witness } = other {
                    Verdict::Fail {
                        reason: reason.clone(),
                        witness: witness.clone(),
                    }
                } else {
                    unreachable!()
                }
            }
            (Verdict::Unknown { reason }, _) | (_, Verdict::Unknown { reason }) => {
                Verdict::Unknown {
                    reason: reason.clone(),
                }
            }
            _ => Verdict::Pass,
        }
    }
}

impl std::fmt::Display for Verdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Verdict::Pass => write!(f, "PASS"),
            Verdict::Fail { reason, .. } => write!(f, "FAIL: {reason}"),
            Verdict::Unknown { reason } => write!(f, "UNKNOWN: {reason}"),
        }
    }
}

// ---------------------------------------------------------------------------
// SmtExpr – lightweight S-expression tree for SMT-LIB encoding
// ---------------------------------------------------------------------------

/// Minimal S-expression tree for SMT-LIB encoding.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum SmtExpr {
    /// Symbolic variable.
    Symbol(String),
    /// Numeric literal.
    Numeral(f64),
    /// Boolean literal.
    Bool(bool),
    /// Function/operator application.
    App {
        /// Operator or function name.
        op: String,
        /// Argument sub-expressions.
        args: Vec<SmtExpr>,
    },
    /// Quantifier (forall / exists).
    Quantifier {
        /// `"forall"` or `"exists"`.
        kind: String,
        /// Bound variable names.
        vars: Vec<(String, String)>,
        /// Quantified body.
        body: Box<SmtExpr>,
    },
    /// Let-binding.
    Let {
        /// Bindings: name → expression.
        bindings: Vec<(String, SmtExpr)>,
        /// Body where bindings are in scope.
        body: Box<SmtExpr>,
    },
}

impl SmtExpr {
    /// Create a symbol expression.
    pub fn sym(name: impl Into<String>) -> Self {
        SmtExpr::Symbol(name.into())
    }

    /// Create a numeric literal.
    pub fn num(v: f64) -> Self {
        SmtExpr::Numeral(v)
    }

    /// Create a boolean literal.
    pub fn bool_lit(v: bool) -> Self {
        SmtExpr::Bool(v)
    }

    /// Create a function/operator application.
    pub fn app(op: impl Into<String>, args: Vec<SmtExpr>) -> Self {
        SmtExpr::App {
            op: op.into(),
            args,
        }
    }

    /// Convenience: binary addition.
    pub fn add(a: SmtExpr, b: SmtExpr) -> Self {
        Self::app("+", vec![a, b])
    }

    /// Convenience: binary subtraction.
    pub fn sub(a: SmtExpr, b: SmtExpr) -> Self {
        Self::app("-", vec![a, b])
    }

    /// Convenience: binary multiplication.
    pub fn mul(a: SmtExpr, b: SmtExpr) -> Self {
        Self::app("*", vec![a, b])
    }

    /// Convenience: less-than comparison.
    pub fn lt(a: SmtExpr, b: SmtExpr) -> Self {
        Self::app("<", vec![a, b])
    }

    /// Convenience: less-than-or-equal comparison.
    pub fn le(a: SmtExpr, b: SmtExpr) -> Self {
        Self::app("<=", vec![a, b])
    }

    /// Convenience: conjunction.
    pub fn and(args: Vec<SmtExpr>) -> Self {
        Self::app("and", args)
    }

    /// Convenience: disjunction.
    pub fn or(args: Vec<SmtExpr>) -> Self {
        Self::app("or", args)
    }

    /// Convenience: negation.
    pub fn not(a: SmtExpr) -> Self {
        Self::app("not", vec![a])
    }

    /// Render to an SMT-LIB compatible string.
    pub fn to_smtlib(&self) -> String {
        match self {
            SmtExpr::Symbol(s) => s.clone(),
            SmtExpr::Numeral(n) => {
                if *n < 0.0 {
                    format!("(- {})", (-n))
                } else {
                    format!("{n}")
                }
            }
            SmtExpr::Bool(b) => if *b { "true" } else { "false" }.to_string(),
            SmtExpr::App { op, args } => {
                let arg_strs: Vec<String> = args.iter().map(|a| a.to_smtlib()).collect();
                format!("({} {})", op, arg_strs.join(" "))
            }
            SmtExpr::Quantifier { kind, vars, body } => {
                let var_strs: Vec<String> =
                    vars.iter().map(|(n, s)| format!("({n} {s})")).collect();
                format!("({} ({}) {})", kind, var_strs.join(" "), body.to_smtlib())
            }
            SmtExpr::Let { bindings, body } => {
                let bind_strs: Vec<String> = bindings
                    .iter()
                    .map(|(n, e)| format!("({} {})", n, e.to_smtlib()))
                    .collect();
                format!("(let ({}) {})", bind_strs.join(" "), body.to_smtlib())
            }
        }
    }

    /// Count the total number of nodes in this expression tree.
    pub fn node_count(&self) -> usize {
        match self {
            SmtExpr::Symbol(_) | SmtExpr::Numeral(_) | SmtExpr::Bool(_) => 1,
            SmtExpr::App { args, .. } => 1 + args.iter().map(|a| a.node_count()).sum::<usize>(),
            SmtExpr::Quantifier { body, .. } => 1 + body.node_count(),
            SmtExpr::Let { bindings, body } => {
                1 + bindings.iter().map(|(_, e)| e.node_count()).sum::<usize>() + body.node_count()
            }
        }
    }

    /// Collect all free symbols in the expression.
    pub fn free_symbols(&self) -> Vec<String> {
        let mut syms = Vec::new();
        self.collect_symbols(&mut syms, &[]);
        syms.sort();
        syms.dedup();
        syms
    }

    fn collect_symbols(&self, out: &mut Vec<String>, bound: &[String]) {
        match self {
            SmtExpr::Symbol(s) => {
                if !bound.contains(s) {
                    out.push(s.clone());
                }
            }
            SmtExpr::Numeral(_) | SmtExpr::Bool(_) => {}
            SmtExpr::App { args, .. } => {
                for a in args {
                    a.collect_symbols(out, bound);
                }
            }
            SmtExpr::Quantifier { vars, body, .. } => {
                let mut new_bound: Vec<String> = bound.to_vec();
                for (n, _) in vars {
                    new_bound.push(n.clone());
                }
                body.collect_symbols(out, &new_bound);
            }
            SmtExpr::Let { bindings, body } => {
                for (_, e) in bindings {
                    e.collect_symbols(out, bound);
                }
                let mut new_bound: Vec<String> = bound.to_vec();
                for (n, _) in bindings {
                    new_bound.push(n.clone());
                }
                body.collect_symbols(out, &new_bound);
            }
        }
    }
}

impl std::fmt::Display for SmtExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_smtlib())
    }
}

// ---------------------------------------------------------------------------
// Core Traits
// ---------------------------------------------------------------------------

/// A type that can be verified against an accessibility property.
///
/// Implementors produce a [`Verdict`] indicating whether the property
/// holds, fails (with a witness), or cannot be determined.
pub trait Verifiable {
    /// Run verification and produce a verdict.
    fn verify(&self, config: &crate::config::VerifierConfig) -> VerifierResult<Verdict>;

    /// Short human-readable description of what is being verified.
    fn description(&self) -> String;

    /// The element id being verified, if applicable.
    fn element_id(&self) -> Option<ElementId> {
        None
    }
}

/// Spatial containment and distance queries on a geometric object.
pub trait SpatialQuery {
    /// Returns `true` if the point lies inside or on the boundary.
    fn contains(&self, point: &Point3) -> bool;

    /// Returns `true` if this object intersects the given bounding box.
    fn intersects_aabb(&self, bbox: &BoundingBox) -> bool;

    /// Signed distance from the point to the surface (negative = inside).
    fn signed_distance(&self, point: &Point3) -> f64;

    /// Unsigned distance from the point to the nearest surface point.
    fn distance(&self, point: &Point3) -> f64 {
        self.signed_distance(point).abs()
    }

    /// Axis-aligned bounding box of this object.
    fn aabb(&self) -> BoundingBox;
}

/// A geometric object that can be transformed in 3-D space.
///
/// Implementations should return a new transformed copy, leaving
/// the original unchanged.
pub trait Transformable: Sized {
    /// Apply a 4×4 homogeneous transform and return the result.
    fn transform(&self, mat: &Transform) -> Self;

    /// Translate by a vector.
    fn translate(&self, offset: &Vector3) -> Self {
        let mut m = Transform::identity();
        m[(0, 3)] = offset.x;
        m[(1, 3)] = offset.y;
        m[(2, 3)] = offset.z;
        self.transform(&m)
    }

    /// Uniform scale about the origin.
    fn scale_uniform(&self, factor: f64) -> Self {
        let mut m = Transform::identity();
        m[(0, 0)] = factor;
        m[(1, 1)] = factor;
        m[(2, 2)] = factor;
        self.transform(&m)
    }
}

/// A type that can produce a [`CoverageCertificate`] summarising
/// how thoroughly the parameter space has been verified.
pub trait Certifiable {
    /// Build a coverage certificate from the accumulated verification results.
    fn certify(&self) -> VerifierResult<crate::certificate::CoverageCertificate>;
}

/// Sampling interface for exploring the body-parameter space.
pub trait Samplable {
    /// Draw `n` samples uniformly from the parameter space.
    fn sample_uniform(&self, n: usize, seed: u64) -> Vec<ParamVec>;

    /// Draw stratified samples with `strata_per_dim` strata along each axis.
    fn sample_stratified(&self, strata_per_dim: usize, seed: u64) -> Vec<ParamVec>;

    /// Return the axis-aligned bounding box of the parameter domain.
    fn parameter_bounds(&self) -> (ParamVec, ParamVec);
}

/// Encode a geometric / kinematic constraint into an [`SmtExpr`]
/// suitable for an SMT solver.
pub trait SmtEncodable {
    /// Produce the SMT-LIB expression encoding this object.
    fn encode_smt(&self) -> VerifierResult<SmtExpr>;

    /// Declare the variables used by the encoding.
    fn smt_variable_declarations(&self) -> Vec<(String, String)>;
}

/// Forward-kinematics solver for a kinematic chain.
///
/// Given a joint-angle vector and body parameters, compute the
/// end-effector transform.
pub trait ForwardKinematicsSolver {
    /// Solve forward kinematics.
    fn solve_fk(
        &self,
        joint_angles: &[f64],
        body_params: &crate::kinematic::BodyParameters,
    ) -> VerifierResult<Transform>;

    /// Compute the Jacobian at the given configuration.
    fn jacobian(
        &self,
        joint_angles: &[f64],
        body_params: &crate::kinematic::BodyParameters,
    ) -> VerifierResult<Vec<Vector3>>;
}

/// Accessibility checker for a single interactable element.
pub trait AccessibilityChecker {
    /// Check whether the given element is accessible for specific body parameters and device.
    fn check_accessibility(
        &self,
        element: &crate::scene::InteractableElement,
        body_params: &crate::kinematic::BodyParameters,
        device: &crate::device::DeviceConfig,
    ) -> VerifierResult<crate::accessibility::AccessibilityVerdict>;
}

/// Objects that can be validated for internal consistency.
pub trait Validatable {
    /// Return a list of human-readable validation errors (empty = valid).
    fn validate(&self) -> Vec<String>;

    /// Returns `true` if validation passes.
    fn is_valid(&self) -> bool {
        self.validate().is_empty()
    }
}

/// Bounding-volume computation for arbitrary objects.
pub trait Bounded {
    /// Compute the tightest axis-aligned bounding box.
    fn bounding_box(&self) -> BoundingBox;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verdict_pass() {
        let v = Verdict::Pass;
        assert!(v.is_pass());
        assert!(!v.is_fail());
        assert!(!v.is_unknown());
        assert_eq!(format!("{v}"), "PASS");
    }

    #[test]
    fn test_verdict_fail() {
        let v = Verdict::Fail {
            reason: "unreachable".into(),
            witness: Some(vec![1.7, 0.35, 0.48, 0.27, 0.19]),
        };
        assert!(v.is_fail());
        assert!(format!("{v}").contains("unreachable"));
    }

    #[test]
    fn test_verdict_merge() {
        let pass = Verdict::Pass;
        let fail = Verdict::Fail {
            reason: "bad".into(),
            witness: None,
        };
        let unknown = Verdict::Unknown {
            reason: "timeout".into(),
        };

        assert!(pass.merge(&pass).is_pass());
        assert!(pass.merge(&fail).is_fail());
        assert!(pass.merge(&unknown).is_unknown());
        assert!(fail.merge(&unknown).is_fail());
    }

    #[test]
    fn test_smt_expr_symbol() {
        let e = SmtExpr::sym("x");
        assert_eq!(e.to_smtlib(), "x");
        assert_eq!(e.node_count(), 1);
    }

    #[test]
    fn test_smt_expr_num() {
        let e = SmtExpr::num(3.14);
        assert_eq!(e.to_smtlib(), "3.14");
    }

    #[test]
    fn test_smt_expr_app() {
        let e = SmtExpr::add(SmtExpr::sym("x"), SmtExpr::num(1.0));
        assert_eq!(e.to_smtlib(), "(+ x 1)");
        assert_eq!(e.node_count(), 3);
    }

    #[test]
    fn test_smt_expr_nested() {
        let e = SmtExpr::and(vec![
            SmtExpr::le(SmtExpr::sym("x"), SmtExpr::num(10.0)),
            SmtExpr::lt(SmtExpr::num(0.0), SmtExpr::sym("x")),
        ]);
        assert!(e.to_smtlib().contains("and"));
        assert!(e.to_smtlib().contains("<="));
    }

    #[test]
    fn test_smt_expr_quantifier() {
        let body = SmtExpr::lt(SmtExpr::sym("x"), SmtExpr::num(5.0));
        let expr = SmtExpr::Quantifier {
            kind: "forall".into(),
            vars: vec![("x".into(), "Real".into())],
            body: Box::new(body),
        };
        let s = expr.to_smtlib();
        assert!(s.starts_with("(forall"));
        assert!(s.contains("(x Real)"));
    }

    #[test]
    fn test_smt_expr_let() {
        let expr = SmtExpr::Let {
            bindings: vec![("a".into(), SmtExpr::num(2.0))],
            body: Box::new(SmtExpr::add(SmtExpr::sym("a"), SmtExpr::num(3.0))),
        };
        let s = expr.to_smtlib();
        assert!(s.starts_with("(let"));
    }

    #[test]
    fn test_free_symbols() {
        let e = SmtExpr::add(SmtExpr::sym("x"), SmtExpr::sym("y"));
        let syms = e.free_symbols();
        assert_eq!(syms, vec!["x", "y"]);
    }

    #[test]
    fn test_free_symbols_with_binding() {
        let expr = SmtExpr::Quantifier {
            kind: "forall".into(),
            vars: vec![("x".into(), "Real".into())],
            body: Box::new(SmtExpr::add(SmtExpr::sym("x"), SmtExpr::sym("y"))),
        };
        let syms = expr.free_symbols();
        assert_eq!(syms, vec!["y"]);
    }

    #[test]
    fn test_smt_expr_negative_num() {
        let e = SmtExpr::num(-2.5);
        assert_eq!(e.to_smtlib(), "(- 2.5)");
    }

    #[test]
    fn test_verdict_serde_roundtrip() {
        let v = Verdict::Fail {
            reason: "test".into(),
            witness: Some(vec![1.0, 2.0]),
        };
        let json = serde_json::to_string(&v).unwrap();
        let back: Verdict = serde_json::from_str(&json).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn test_smt_bool_lit() {
        assert_eq!(SmtExpr::bool_lit(true).to_smtlib(), "true");
        assert_eq!(SmtExpr::bool_lit(false).to_smtlib(), "false");
    }

    #[test]
    fn test_smt_not() {
        let e = SmtExpr::not(SmtExpr::sym("p"));
        assert_eq!(e.to_smtlib(), "(not p)");
    }

    #[test]
    fn test_smt_or() {
        let e = SmtExpr::or(vec![SmtExpr::sym("a"), SmtExpr::sym("b")]);
        assert_eq!(e.to_smtlib(), "(or a b)");
    }

    #[test]
    fn test_smt_expr_serde_roundtrip() {
        let e = SmtExpr::and(vec![
            SmtExpr::le(SmtExpr::sym("x"), SmtExpr::num(10.0)),
            SmtExpr::lt(SmtExpr::num(0.0), SmtExpr::sym("y")),
        ]);
        let json = serde_json::to_string(&e).unwrap();
        let back: SmtExpr = serde_json::from_str(&json).unwrap();
        assert_eq!(e, back);
    }
}

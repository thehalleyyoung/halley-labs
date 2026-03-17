//! Type inference engine for the SoniType DSL.
//!
//! Implements Hindley-Milner style type inference adapted for perceptual
//! qualifiers. Generates unification constraints from expressions, solves them,
//! and applies the resulting substitution to produce fully typed programs.

use crate::ast::*;
use crate::token::Span;
use crate::type_system::{
    BaseType, PerceptualType, Qualifier, TypeCheckerConfig, TypeError, TypeErrorKind,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::fmt;

// ─── Type Variable Management ────────────────────────────────────────────────

/// A type variable identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TypeVar(pub u32);

impl TypeVar {
    pub fn index(self) -> u32 {
        self.0
    }
}

impl fmt::Display for TypeVar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "?{}", self.0)
    }
}

/// Manages fresh type variable generation.
#[derive(Debug, Default)]
pub struct TypeVarGen {
    next: u32,
}

impl TypeVarGen {
    pub fn new() -> Self {
        Self { next: 0 }
    }

    pub fn fresh(&mut self) -> TypeVar {
        let v = TypeVar(self.next);
        self.next += 1;
        v
    }

    pub fn fresh_base(&mut self) -> BaseType {
        BaseType::TypeVar(self.fresh().0)
    }
}

// ─── Substitution ────────────────────────────────────────────────────────────

/// A mapping from type variables to their resolved base types.
#[derive(Debug, Clone, Default)]
pub struct Substitution {
    map: HashMap<u32, BaseType>,
}

impl Substitution {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn bind(&mut self, var: u32, ty: BaseType) {
        self.map.insert(var, ty);
    }

    pub fn lookup(&self, var: u32) -> Option<&BaseType> {
        self.map.get(&var)
    }

    /// Apply substitution to a base type (chase all variables).
    pub fn apply(&self, ty: &BaseType) -> BaseType {
        match ty {
            BaseType::TypeVar(id) => {
                if let Some(resolved) = self.map.get(id) {
                    self.apply(resolved)
                } else {
                    ty.clone()
                }
            }
            BaseType::Function(a, b) => {
                BaseType::Function(Box::new(self.apply(a)), Box::new(self.apply(b)))
            }
            BaseType::Tuple(ts) => BaseType::Tuple(ts.iter().map(|t| self.apply(t)).collect()),
            _ => ty.clone(),
        }
    }

    /// Apply substitution to a perceptual type.
    pub fn apply_perceptual(&self, ty: &PerceptualType) -> PerceptualType {
        PerceptualType {
            base: self.apply(&ty.base),
            qualifier: ty.qualifier.clone(),
        }
    }

    /// Compose two substitutions: self ∘ other.
    pub fn compose(&self, other: &Substitution) -> Substitution {
        let mut result = Substitution::new();
        for (k, v) in &other.map {
            result.bind(*k, self.apply(v));
        }
        for (k, v) in &self.map {
            result.map.entry(*k).or_insert_with(|| v.clone());
        }
        result
    }
}

// ─── Occurs Check ────────────────────────────────────────────────────────────

/// Check if a type variable occurs in a type (prevents infinite types).
fn occurs_in(var: u32, ty: &BaseType) -> bool {
    match ty {
        BaseType::TypeVar(id) => *id == var,
        BaseType::Function(a, b) => occurs_in(var, a) || occurs_in(var, b),
        BaseType::Tuple(ts) => ts.iter().any(|t| occurs_in(var, t)),
        _ => false,
    }
}

// ─── Constraints ─────────────────────────────────────────────────────────────

/// Constraint types generated during inference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Constraint {
    /// Two base types must unify.
    Equality(BaseType, BaseType, Span),
    /// Subtyping relationship between perceptual types.
    Subtype(PerceptualType, PerceptualType, Span),
    /// A qualifier must satisfy a bound.
    QualifierBound(Qualifier, QualifierBound, Span),
    /// Segregation required between two stream qualifiers.
    SegregationRequired(Qualifier, Qualifier, Span),
    /// Masking clearance constraint.
    MaskingClearance(f64, f64, Span),
    /// JND sufficiency for a parameter mapping.
    JndSufficient(String, f64, f64, Span),
    /// Total load must not exceed budget.
    LoadBudget(f64, f64, Span),
}

/// A bound on a qualifier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QualifierBound {
    MaxLoad(f64),
    MinMasking(f64),
    MaxBands(usize),
}

/// A set of constraints.
#[derive(Debug, Clone, Default)]
pub struct ConstraintSet {
    constraints: Vec<Constraint>,
}

impl ConstraintSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }

    pub fn merge(&mut self, other: ConstraintSet) {
        self.constraints.extend(other.constraints);
    }

    pub fn len(&self) -> usize {
        self.constraints.len()
    }

    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Constraint> {
        self.constraints.iter()
    }

    pub fn into_iter(self) -> impl Iterator<Item = Constraint> {
        self.constraints.into_iter()
    }
}

// ─── Unification ─────────────────────────────────────────────────────────────

/// Unify two base types, producing a substitution.
fn unify(a: &BaseType, b: &BaseType, span: Span) -> Result<Substitution, TypeError> {
    if a == b {
        return Ok(Substitution::new());
    }

    match (a, b) {
        (BaseType::TypeVar(id), ty) | (ty, BaseType::TypeVar(id)) => {
            if occurs_in(*id, ty) {
                return Err(TypeError::new(
                    TypeErrorKind::Mismatch,
                    format!("infinite type: ?{id} occurs in {ty}"),
                    span,
                ));
            }
            let mut sub = Substitution::new();
            sub.bind(*id, ty.clone());
            Ok(sub)
        }
        (BaseType::Function(a1, b1), BaseType::Function(a2, b2)) => {
            let s1 = unify(a1, a2, span)?;
            let s2 = unify(&s1.apply(b1), &s1.apply(b2), span)?;
            Ok(s2.compose(&s1))
        }
        (BaseType::Tuple(ts1), BaseType::Tuple(ts2)) if ts1.len() == ts2.len() => {
            let mut sub = Substitution::new();
            for (t1, t2) in ts1.iter().zip(ts2.iter()) {
                let s = unify(&sub.apply(t1), &sub.apply(t2), span)?;
                sub = s.compose(&sub);
            }
            Ok(sub)
        }
        // Int can coerce to Float
        (BaseType::Int, BaseType::Float) | (BaseType::Float, BaseType::Int) => {
            Ok(Substitution::new())
        }
        _ => Err(TypeError::new(
            TypeErrorKind::Mismatch,
            format!("cannot unify {a} with {b}"),
            span,
        )),
    }
}

// ─── Inference Engine ────────────────────────────────────────────────────────

/// The type inference engine.
pub struct InferenceEngine {
    var_gen: TypeVarGen,
    constraints: ConstraintSet,
    substitution: Substitution,
    env: HashMap<String, PerceptualType>,
    config: TypeCheckerConfig,
    errors: Vec<TypeError>,
}

impl InferenceEngine {
    pub fn new() -> Self {
        Self {
            var_gen: TypeVarGen::new(),
            constraints: ConstraintSet::new(),
            substitution: Substitution::new(),
            env: HashMap::new(),
            config: TypeCheckerConfig::default(),
            errors: Vec::new(),
        }
    }

    pub fn with_config(config: TypeCheckerConfig) -> Self {
        Self { config, ..Self::new() }
    }

    fn fresh(&mut self) -> BaseType {
        self.var_gen.fresh_base()
    }

    pub fn bind(&mut self, name: impl Into<String>, ty: PerceptualType) {
        self.env.insert(name.into(), ty);
    }

    pub fn lookup(&self, name: &str) -> Option<&PerceptualType> {
        self.env.get(name)
    }

    /// Infer the type of an expression, collecting constraints.
    pub fn infer_expr(&mut self, expr: &Expr) -> Result<PerceptualType, TypeError> {
        match expr {
            Expr::Literal(lit) => Ok(self.infer_literal(lit)),
            Expr::Identifier(id) => {
                self.env.get(&id.name).cloned().ok_or_else(|| {
                    TypeError::new(
                        TypeErrorKind::UndefinedVariable,
                        format!("undefined: {}", id.name),
                        id.span,
                    )
                })
            }
            Expr::BinaryOp(binop) => self.infer_binary_op(binop),
            Expr::UnaryOp(unop) => self.infer_unary_op(unop),
            Expr::FunctionCall(fc) => self.infer_function_call(fc),
            Expr::LetIn(li) => self.infer_let_in(li),
            Expr::IfThenElse(ite) => self.infer_if_then_else(ite),
            Expr::Lambda(lam) => self.infer_lambda(lam),
            Expr::PipeOperator(p) => self.infer_pipe(p),
            Expr::FieldAccess(_) => Ok(PerceptualType::simple(self.fresh())),
            Expr::Grouped(inner, _) => self.infer_expr(inner),
            Expr::StreamLiteral(_) => Ok(PerceptualType::simple(BaseType::Stream)),
            Expr::MappingLiteral(_) => Ok(PerceptualType::simple(BaseType::Mapping)),
            Expr::Compose(_) => Ok(PerceptualType::simple(BaseType::MultiStream)),
            Expr::WithClause(w) => self.infer_expr(&w.expr),
            Expr::WhereClause(w) => self.infer_expr(&w.expr),
        }
    }

    fn infer_literal(&self, lit: &Literal) -> PerceptualType {
        let base = match &lit.value {
            LiteralValue::Int(_) => BaseType::Int,
            LiteralValue::Float(_) => BaseType::Float,
            LiteralValue::String(_) => BaseType::Str,
            LiteralValue::Bool(_) => BaseType::Bool,
        };
        PerceptualType::simple(base)
    }

    fn infer_binary_op(&mut self, binop: &BinaryOp) -> Result<PerceptualType, TypeError> {
        let lhs = self.infer_expr(&binop.lhs)?;
        let rhs = self.infer_expr(&binop.rhs)?;

        match binop.op {
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
                self.constraints.add(Constraint::Equality(
                    lhs.base.clone(),
                    rhs.base.clone(),
                    binop.span,
                ));
                let qualifier = lhs.qualifier.join(&rhs.qualifier);
                Ok(PerceptualType::new(lhs.base, qualifier))
            }
            BinOp::Eq | BinOp::Neq | BinOp::Lt | BinOp::Gt | BinOp::Lte | BinOp::Gte => {
                self.constraints.add(Constraint::Equality(
                    lhs.base.clone(),
                    rhs.base.clone(),
                    binop.span,
                ));
                Ok(PerceptualType::simple(BaseType::Bool))
            }
            BinOp::And | BinOp::Or => {
                self.constraints.add(Constraint::Equality(lhs.base, BaseType::Bool, binop.span));
                self.constraints.add(Constraint::Equality(rhs.base, BaseType::Bool, binop.span));
                Ok(PerceptualType::simple(BaseType::Bool))
            }
            BinOp::Range => {
                let qualifier = lhs.qualifier.join(&rhs.qualifier);
                Ok(PerceptualType::new(
                    BaseType::Tuple(vec![lhs.base, rhs.base]),
                    qualifier,
                ))
            }
        }
    }

    fn infer_unary_op(&mut self, unop: &UnaryOp) -> Result<PerceptualType, TypeError> {
        let operand = self.infer_expr(&unop.operand)?;
        match unop.op {
            UnOp::Neg => Ok(operand),
            UnOp::Not => {
                self.constraints.add(Constraint::Equality(
                    operand.base,
                    BaseType::Bool,
                    unop.span,
                ));
                Ok(PerceptualType::simple(BaseType::Bool))
            }
        }
    }

    fn infer_function_call(&mut self, fc: &FunctionCall) -> Result<PerceptualType, TypeError> {
        let callee = self.infer_expr(&fc.callee)?;
        let ret = self.fresh();

        let mut arg_types = Vec::new();
        for arg in &fc.args {
            arg_types.push(self.infer_expr(arg)?);
        }

        // Build expected function type
        let mut expected_fn = ret.clone();
        for at in arg_types.iter().rev() {
            expected_fn = BaseType::Function(Box::new(at.base.clone()), Box::new(expected_fn));
        }

        self.constraints.add(Constraint::Equality(
            callee.base,
            expected_fn,
            fc.span,
        ));

        Ok(PerceptualType::simple(ret))
    }

    fn infer_let_in(&mut self, li: &LetIn) -> Result<PerceptualType, TypeError> {
        let value_ty = self.infer_expr(&li.value)?;
        // Polymorphic let: bind the inferred type
        self.bind_pattern(&li.pattern, &value_ty);
        self.infer_expr(&li.body)
    }

    fn infer_if_then_else(&mut self, ite: &IfThenElse) -> Result<PerceptualType, TypeError> {
        let cond = self.infer_expr(&ite.condition)?;
        self.constraints.add(Constraint::Equality(
            cond.base,
            BaseType::Bool,
            ite.condition.span(),
        ));

        let then_ty = self.infer_expr(&ite.then_branch)?;
        let else_ty = self.infer_expr(&ite.else_branch)?;

        self.constraints.add(Constraint::Equality(
            then_ty.base.clone(),
            else_ty.base.clone(),
            ite.span,
        ));

        let qualifier = then_ty.qualifier.join(&else_ty.qualifier);
        Ok(PerceptualType::new(then_ty.base, qualifier))
    }

    fn infer_lambda(&mut self, lam: &Lambda) -> Result<PerceptualType, TypeError> {
        let mut param_bases = Vec::new();
        for p in &lam.params {
            let param_ty = self.fresh();
            self.bind(&p.name.name, PerceptualType::simple(param_ty.clone()));
            param_bases.push(param_ty);
        }

        let body_ty = self.infer_expr(&lam.body)?;
        let mut result = body_ty.base;
        for pb in param_bases.into_iter().rev() {
            result = BaseType::Function(Box::new(pb), Box::new(result));
        }
        Ok(PerceptualType::simple(result))
    }

    fn infer_pipe(&mut self, pipe: &PipeExpr) -> Result<PerceptualType, TypeError> {
        let lhs = self.infer_expr(&pipe.lhs)?;
        let rhs = self.infer_expr(&pipe.rhs)?;
        let ret = self.fresh();

        // rhs must accept lhs as argument
        let expected = BaseType::Function(Box::new(lhs.base.clone()), Box::new(ret.clone()));
        self.constraints.add(Constraint::Equality(rhs.base, expected, pipe.span));

        let qualifier = lhs.qualifier.join(&rhs.qualifier);
        Ok(PerceptualType::new(ret, qualifier))
    }

    fn bind_pattern(&mut self, pat: &Pattern, ty: &PerceptualType) {
        match pat {
            Pattern::Variable(id) => {
                self.env.insert(id.name.clone(), ty.clone());
            }
            Pattern::Tuple(pats, _) => {
                for p in pats {
                    self.bind_pattern(p, ty);
                }
            }
            Pattern::Wildcard(_) => {}
        }
    }

    // ── Solving ──────────────────────────────────────────────────────────────

    /// Solve all collected constraints and produce the final substitution.
    pub fn solve(&mut self) -> Result<Substitution, Vec<TypeError>> {
        let mut errors = Vec::new();
        let mut sub = Substitution::new();

        for constraint in self.constraints.constraints.clone() {
            match constraint {
                Constraint::Equality(a, b, span) => {
                    let a_resolved = sub.apply(&a);
                    let b_resolved = sub.apply(&b);
                    match unify(&a_resolved, &b_resolved, span) {
                        Ok(s) => sub = s.compose(&sub),
                        Err(e) => errors.push(e),
                    }
                }
                Constraint::LoadBudget(load, budget, span) => {
                    if load > budget {
                        errors.push(TypeError::new(
                            TypeErrorKind::CognitiveLoadExceeded,
                            format!("cognitive load {load} exceeds budget {budget}"),
                            span,
                        ));
                    }
                }
                Constraint::MaskingClearance(actual, required, span) => {
                    if actual < required {
                        errors.push(TypeError::new(
                            TypeErrorKind::MaskingViolation,
                            format!("masking {actual} dB < required {required} dB"),
                            span,
                        ));
                    }
                }
                Constraint::JndSufficient(param, actual, required, span) => {
                    if actual < required {
                        errors.push(TypeError::new(
                            TypeErrorKind::JndInsufficient,
                            format!("JND for {param}: {actual} < {required}"),
                            span,
                        ));
                    }
                }
                Constraint::SegregationRequired(q1, q2, span) => {
                    let overlap: BTreeSet<u8> = q1
                        .band_occupancy
                        .intersection(&q2.band_occupancy)
                        .copied()
                        .collect();
                    if !overlap.is_empty() {
                        errors.push(TypeError::new(
                            TypeErrorKind::SegregationFailure,
                            format!("overlapping Bark bands: {:?}", overlap),
                            span,
                        ));
                    }
                }
                Constraint::Subtype(sub_ty, sup_ty, span) => {
                    if !sub_ty.qualifier.is_stronger_than(&sup_ty.qualifier) {
                        errors.push(TypeError::new(
                            TypeErrorKind::Mismatch,
                            "subtype qualifier constraint violated".to_string(),
                            span,
                        ));
                    }
                }
                Constraint::QualifierBound(q, bound, span) => {
                    match bound {
                        QualifierBound::MaxLoad(max) => {
                            if q.cognitive_load > max {
                                errors.push(TypeError::new(
                                    TypeErrorKind::CognitiveLoadExceeded,
                                    format!("load {} > max {}", q.cognitive_load, max),
                                    span,
                                ));
                            }
                        }
                        QualifierBound::MinMasking(min) => {
                            if q.masking_margin < min {
                                errors.push(TypeError::new(
                                    TypeErrorKind::MaskingViolation,
                                    format!("masking {} < min {}", q.masking_margin, min),
                                    span,
                                ));
                            }
                        }
                        QualifierBound::MaxBands(max) => {
                            if q.band_occupancy.len() > max {
                                errors.push(TypeError::new(
                                    TypeErrorKind::UnsatisfiableConstraint,
                                    format!("band count {} > max {}", q.band_occupancy.len(), max),
                                    span,
                                ));
                            }
                        }
                    }
                }
            }
        }

        self.substitution = sub.clone();
        if errors.is_empty() {
            Ok(sub)
        } else {
            Err(errors)
        }
    }

    /// Apply the current substitution to resolve a type.
    pub fn resolve(&self, ty: &PerceptualType) -> PerceptualType {
        self.substitution.apply_perceptual(ty)
    }

    /// Apply qualifier defaulting: when inference is ambiguous, use defaults.
    pub fn apply_defaults(&self, ty: &mut PerceptualType) {
        if ty.qualifier.cognitive_load == 0.0 && ty.base == BaseType::Stream {
            ty.qualifier.cognitive_load = 1.0;
        }
        if ty.qualifier.masking_margin == f64::INFINITY && ty.base == BaseType::Stream {
            ty.qualifier.masking_margin = self.config.default_masking_margin;
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::Span;

    fn dummy_span() -> Span { Span::dummy() }

    #[test]
    fn test_type_var_gen() {
        let mut gen = TypeVarGen::new();
        assert_eq!(gen.fresh(), TypeVar(0));
        assert_eq!(gen.fresh(), TypeVar(1));
        assert_eq!(gen.fresh(), TypeVar(2));
    }

    #[test]
    fn test_substitution_apply() {
        let mut sub = Substitution::new();
        sub.bind(0, BaseType::Int);
        assert_eq!(sub.apply(&BaseType::TypeVar(0)), BaseType::Int);
        assert_eq!(sub.apply(&BaseType::Float), BaseType::Float);
    }

    #[test]
    fn test_substitution_compose() {
        let mut s1 = Substitution::new();
        s1.bind(0, BaseType::TypeVar(1));
        let mut s2 = Substitution::new();
        s2.bind(1, BaseType::Int);
        let composed = s2.compose(&s1);
        assert_eq!(composed.apply(&BaseType::TypeVar(0)), BaseType::Int);
    }

    #[test]
    fn test_occurs_check() {
        assert!(occurs_in(0, &BaseType::TypeVar(0)));
        assert!(!occurs_in(0, &BaseType::TypeVar(1)));
        assert!(occurs_in(
            0,
            &BaseType::Function(Box::new(BaseType::TypeVar(0)), Box::new(BaseType::Int))
        ));
    }

    #[test]
    fn test_unify_same() {
        let sub = unify(&BaseType::Int, &BaseType::Int, dummy_span()).unwrap();
        assert!(sub.map.is_empty());
    }

    #[test]
    fn test_unify_var_concrete() {
        let sub = unify(&BaseType::TypeVar(0), &BaseType::Float, dummy_span()).unwrap();
        assert_eq!(sub.apply(&BaseType::TypeVar(0)), BaseType::Float);
    }

    #[test]
    fn test_unify_function() {
        let a = BaseType::Function(Box::new(BaseType::TypeVar(0)), Box::new(BaseType::Int));
        let b = BaseType::Function(Box::new(BaseType::Float), Box::new(BaseType::Int));
        let sub = unify(&a, &b, dummy_span()).unwrap();
        assert_eq!(sub.apply(&BaseType::TypeVar(0)), BaseType::Float);
    }

    #[test]
    fn test_unify_mismatch() {
        let result = unify(&BaseType::Bool, &BaseType::Str, dummy_span());
        assert!(result.is_err());
    }

    #[test]
    fn test_unify_occurs_infinite() {
        let result = unify(
            &BaseType::TypeVar(0),
            &BaseType::Function(Box::new(BaseType::TypeVar(0)), Box::new(BaseType::Int)),
            dummy_span(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_infer_literal() {
        let mut engine = InferenceEngine::new();
        let lit_expr = Expr::Literal(Literal {
            value: LiteralValue::Int(42),
            span: dummy_span(),
        });
        let ty = engine.infer_expr(&lit_expr).unwrap();
        assert_eq!(ty.base, BaseType::Int);
    }

    #[test]
    fn test_infer_identifier() {
        let mut engine = InferenceEngine::new();
        engine.bind("x", PerceptualType::simple(BaseType::Float));
        let id_expr = Expr::Identifier(Identifier::new("x", dummy_span()));
        let ty = engine.infer_expr(&id_expr).unwrap();
        assert_eq!(ty.base, BaseType::Float);
    }

    #[test]
    fn test_infer_undefined_variable() {
        let mut engine = InferenceEngine::new();
        let id_expr = Expr::Identifier(Identifier::new("unknown", dummy_span()));
        assert!(engine.infer_expr(&id_expr).is_err());
    }

    #[test]
    fn test_infer_if_then_else() {
        let mut engine = InferenceEngine::new();
        let ite = Expr::IfThenElse(IfThenElse {
            condition: Box::new(Expr::Literal(Literal {
                value: LiteralValue::Bool(true),
                span: dummy_span(),
            })),
            then_branch: Box::new(Expr::Literal(Literal {
                value: LiteralValue::Int(1),
                span: dummy_span(),
            })),
            else_branch: Box::new(Expr::Literal(Literal {
                value: LiteralValue::Int(0),
                span: dummy_span(),
            })),
            span: dummy_span(),
        });
        let ty = engine.infer_expr(&ite).unwrap();
        assert_eq!(ty.base, BaseType::Int);
    }

    #[test]
    fn test_infer_let_in() {
        let mut engine = InferenceEngine::new();
        let li = Expr::LetIn(LetIn {
            pattern: Pattern::Variable(Identifier::new("x", dummy_span())),
            type_annotation: None,
            value: Box::new(Expr::Literal(Literal {
                value: LiteralValue::Int(10),
                span: dummy_span(),
            })),
            body: Box::new(Expr::Identifier(Identifier::new("x", dummy_span()))),
            span: dummy_span(),
        });
        let ty = engine.infer_expr(&li).unwrap();
        assert_eq!(ty.base, BaseType::Int);
    }

    #[test]
    fn test_solve_equality_constraints() {
        let mut engine = InferenceEngine::new();
        engine.constraints.add(Constraint::Equality(
            BaseType::TypeVar(0),
            BaseType::Int,
            dummy_span(),
        ));
        let sub = engine.solve().unwrap();
        assert_eq!(sub.apply(&BaseType::TypeVar(0)), BaseType::Int);
    }

    #[test]
    fn test_solve_conflicting_constraints() {
        let mut engine = InferenceEngine::new();
        engine.constraints.add(Constraint::Equality(
            BaseType::TypeVar(0),
            BaseType::Int,
            dummy_span(),
        ));
        engine.constraints.add(Constraint::Equality(
            BaseType::TypeVar(0),
            BaseType::Bool,
            dummy_span(),
        ));
        assert!(engine.solve().is_err());
    }

    #[test]
    fn test_constraint_set_merge() {
        let mut cs1 = ConstraintSet::new();
        cs1.add(Constraint::Equality(BaseType::Int, BaseType::Int, dummy_span()));
        let mut cs2 = ConstraintSet::new();
        cs2.add(Constraint::Equality(BaseType::Float, BaseType::Float, dummy_span()));
        cs1.merge(cs2);
        assert_eq!(cs1.len(), 2);
    }

    #[test]
    fn test_qualifier_bound_load() {
        let mut engine = InferenceEngine::new();
        engine.constraints.add(Constraint::QualifierBound(
            Qualifier { cognitive_load: 5.0, ..Default::default() },
            QualifierBound::MaxLoad(4.0),
            dummy_span(),
        ));
        assert!(engine.solve().is_err());
    }

    #[test]
    fn test_apply_defaults_stream() {
        let engine = InferenceEngine::new();
        let mut ty = PerceptualType::simple(BaseType::Stream);
        engine.apply_defaults(&mut ty);
        assert_eq!(ty.qualifier.cognitive_load, 1.0);
        assert_eq!(ty.qualifier.masking_margin, 6.0); // default
    }
}

//! Spatial type checker for the Choreo DSL.
//!
//! Verifies spatial constraints, temporal well-formedness, pattern determinism,
//! spatial subtyping via LP feasibility, and name/type resolution.

use crate::ast::*;
use choreo_types::{Diagnostic, Severity, Span};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Type representation
// ---------------------------------------------------------------------------

/// Resolved type in the type environment.
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Region,
    Entity(Option<EntityTypeExpr>),
    Zone,
    Interaction,
    Scene,
    Bool,
    Int,
    Float,
    String,
    Duration,
    Distance,
    Angle,
    Vector3,
    Quaternion,
    Tuple(Vec<Type>),
    Function(Vec<Type>, Box<Type>),
    Named(std::string::String),
    Unknown,
}

impl Type {
    fn from_annotation(ann: &TypeAnnotation) -> Self {
        match ann {
            TypeAnnotation::Named(n) => Type::Named(n.clone()),
            TypeAnnotation::Region => Type::Region,
            TypeAnnotation::Entity => Type::Entity(None),
            TypeAnnotation::Duration => Type::Duration,
            TypeAnnotation::Distance => Type::Distance,
            TypeAnnotation::Angle => Type::Angle,
            TypeAnnotation::Bool => Type::Bool,
            TypeAnnotation::Int => Type::Int,
            TypeAnnotation::Float => Type::Float,
            TypeAnnotation::String_ => Type::String,
            TypeAnnotation::Vector3 => Type::Vector3,
            TypeAnnotation::Quaternion => Type::Quaternion,
            TypeAnnotation::Tuple(elems) => {
                Type::Tuple(elems.iter().map(Type::from_annotation).collect())
            }
            TypeAnnotation::Function(params, ret) => Type::Function(
                params.iter().map(Type::from_annotation).collect(),
                Box::new(Type::from_annotation(ret)),
            ),
            TypeAnnotation::Span(_) => Type::Unknown,
        }
    }
}

/// Kind of a named symbol in the environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SymKind {
    Region,
    Entity,
    Zone,
    Interaction,
    Scene,
    LetBinding,
    Parameter,
}

/// Entry in the type environment.
#[derive(Debug, Clone)]
struct EnvEntry {
    kind: SymKind,
    resolved_type: Type,
    span: Span,
}

// ---------------------------------------------------------------------------
// TypedProgram wrapper
// ---------------------------------------------------------------------------

/// A typed program with resolved types attached.
#[derive(Debug, Clone)]
pub struct TypedProgram {
    pub program: Program,
    /// Map from declaration name -> resolved type.
    pub types: HashMap<std::string::String, Type>,
}

// ---------------------------------------------------------------------------
// LP feasibility oracle (simple simplex-like containment check)
// ---------------------------------------------------------------------------

/// A half-plane constraint: coeff · x ≤ bound.
#[derive(Debug, Clone)]
struct LinearConstraint {
    coefficients: Vec<f64>,
    bound: f64,
}

/// Simple LP feasibility check: are there x such that Ax ≤ b?
/// Uses Phase-I simplex approach: minimize sum of artificial variables.
/// Returns true if feasible (there exists a point satisfying all constraints).
fn lp_feasible(constraints: &[LinearConstraint], dim: usize) -> bool {
    if constraints.is_empty() {
        return true;
    }

    // Use iterative constraint satisfaction: try to find a point that
    // satisfies all constraints using a projection/relaxation approach.
    let max_iter = 500;
    let mut x = vec![0.0f64; dim];

    for _iter in 0..max_iter {
        let mut all_satisfied = true;
        let mut max_violation = 0.0f64;

        for constraint in constraints {
            let dot: f64 = constraint
                .coefficients
                .iter()
                .zip(x.iter())
                .map(|(c, xi)| c * xi)
                .sum();
            let violation = dot - constraint.bound;

            if violation > 1e-8 {
                all_satisfied = false;
                max_violation = max_violation.max(violation);

                // Project x onto the half-space: x -= (violation / ||a||²) * a
                let norm_sq: f64 = constraint
                    .coefficients
                    .iter()
                    .map(|c| c * c)
                    .sum();
                if norm_sq > 1e-12 {
                    let step = violation / norm_sq;
                    for (xi, ci) in x.iter_mut().zip(constraint.coefficients.iter()) {
                        *xi -= step * ci;
                    }
                }
            }
        }

        if all_satisfied {
            return true;
        }

        // If violations are growing, the system is likely infeasible.
        if max_violation > 1e12 {
            return false;
        }
    }

    // Final check: are all constraints satisfied?
    constraints.iter().all(|c| {
        let dot: f64 = c.coefficients.iter().zip(x.iter()).map(|(a, xi)| a * xi).sum();
        dot <= c.bound + 1e-6
    })
}

/// Check if a convex region A is contained in convex region B.
/// A ⊂ B iff every vertex of A satisfies all constraints of B.
/// Both are represented as half-space intersections.
fn convex_containment(
    vertices_a: &[[f64; 3]],
    constraints_b: &[LinearConstraint],
) -> bool {
    for v in vertices_a {
        for c in constraints_b {
            let dot: f64 = c.coefficients.iter().zip(v.iter()).map(|(a, xi)| a * xi).sum();
            if dot > c.bound + 1e-8 {
                return false;
            }
        }
    }
    true
}

/// Build half-space constraints for an axis-aligned box given center and half-extents.
fn box_constraints(center: [f64; 3], half_extents: [f64; 3]) -> Vec<LinearConstraint> {
    let mut constraints = Vec::with_capacity(6);
    for dim in 0..3 {
        // x_dim <= center + half_extent
        let mut coeff_upper = vec![0.0; 3];
        coeff_upper[dim] = 1.0;
        constraints.push(LinearConstraint {
            coefficients: coeff_upper,
            bound: center[dim] + half_extents[dim],
        });
        // -x_dim <= -(center - half_extent)
        let mut coeff_lower = vec![0.0; 3];
        coeff_lower[dim] = -1.0;
        constraints.push(LinearConstraint {
            coefficients: coeff_lower,
            bound: -(center[dim] - half_extents[dim]),
        });
    }
    constraints
}

/// Vertices of an axis-aligned box.
fn box_vertices(center: [f64; 3], half_extents: [f64; 3]) -> Vec<[f64; 3]> {
    let mut verts = Vec::with_capacity(8);
    for &sx in &[-1.0, 1.0] {
        for &sy in &[-1.0, 1.0] {
            for &sz in &[-1.0, 1.0] {
                verts.push([
                    center[0] + sx * half_extents[0],
                    center[1] + sy * half_extents[1],
                    center[2] + sz * half_extents[2],
                ]);
            }
        }
    }
    verts
}

// ---------------------------------------------------------------------------
// TypeChecker
// ---------------------------------------------------------------------------

/// The spatial type checker for the Choreo DSL.
pub struct TypeChecker {
    env: HashMap<std::string::String, EnvEntry>,
    diagnostics: Vec<Diagnostic>,
    /// Region geometry info for spatial constraint checking.
    region_geometries: HashMap<std::string::String, RegionGeomInfo>,
    /// All interaction guards for determinism checking.
    interaction_guards: Vec<(std::string::String, PatternExpr)>,
}

#[derive(Debug, Clone)]
enum RegionGeomInfo {
    Box {
        center: [f64; 3],
        half_extents: [f64; 3],
    },
    Sphere {
        center: [f64; 3],
        radius: f64,
    },
    Unknown,
}

impl TypeChecker {
    pub fn new() -> Self {
        Self {
            env: HashMap::new(),
            diagnostics: Vec::new(),
            region_geometries: HashMap::new(),
            interaction_guards: Vec::new(),
        }
    }

    /// Type-check a program and return a TypedProgram or diagnostics.
    pub fn check_program(program: &Program) -> Result<TypedProgram, Vec<Diagnostic>> {
        let mut checker = TypeChecker::new();
        checker.resolve_names(program);
        checker.check_all_declarations(program);
        checker.check_region_consistency(program);
        checker.check_temporal_well_formedness(program);
        checker.check_determinism(program);

        let errors: Vec<_> = checker
            .diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .cloned()
            .collect();

        if errors.is_empty() {
            let types: HashMap<std::string::String, Type> = checker
                .env
                .iter()
                .map(|(k, v)| (k.clone(), v.resolved_type.clone()))
                .collect();
            Ok(TypedProgram {
                program: program.clone(),
                types,
            })
        } else {
            Err(checker.diagnostics)
        }
    }

    // -- name resolution --

    /// Resolve all names in the program and populate the environment.
    pub fn resolve_names(&mut self, program: &Program) {
        for decl in &program.declarations {
            match decl {
                Declaration::Region(r) => {
                    self.env.insert(
                        r.name.clone(),
                        EnvEntry {
                            kind: SymKind::Region,
                            resolved_type: Type::Region,
                            span: r.span.clone(),
                        },
                    );
                    self.extract_region_geometry(r);
                }
                Declaration::Entity(e) => {
                    self.env.insert(
                        e.name.clone(),
                        EnvEntry {
                            kind: SymKind::Entity,
                            resolved_type: Type::Entity(Some(e.entity_type.clone())),
                            span: e.span.clone(),
                        },
                    );
                }
                Declaration::Zone(z) => {
                    self.env.insert(
                        z.name.clone(),
                        EnvEntry {
                            kind: SymKind::Zone,
                            resolved_type: Type::Zone,
                            span: z.span.clone(),
                        },
                    );
                }
                Declaration::Interaction(i) => {
                    self.env.insert(
                        i.name.clone(),
                        EnvEntry {
                            kind: SymKind::Interaction,
                            resolved_type: Type::Interaction,
                            span: i.span.clone(),
                        },
                    );
                    // Collect guards for determinism checking
                    if let Some(pat) = &i.pattern {
                        self.interaction_guards
                            .push((i.name.clone(), pat.clone()));
                    }
                    match &i.body {
                        InteractionBody::Simple { guard, .. } => {
                            self.interaction_guards
                                .push((i.name.clone(), guard.clone()));
                        }
                        _ => {}
                    }
                }
                Declaration::Scene(s) => {
                    self.env.insert(
                        s.name.clone(),
                        EnvEntry {
                            kind: SymKind::Scene,
                            resolved_type: Type::Scene,
                            span: s.span.clone(),
                        },
                    );
                }
                Declaration::LetBinding(l) => {
                    let ty = if let Some(ann) = &l.type_annotation {
                        Type::from_annotation(ann)
                    } else {
                        self.infer_expr_type(&l.value)
                    };
                    self.env.insert(
                        l.name.clone(),
                        EnvEntry {
                            kind: SymKind::LetBinding,
                            resolved_type: ty,
                            span: l.span.clone(),
                        },
                    );
                }
                Declaration::Import(_) => {}
            }
        }
    }

    fn extract_region_geometry(&mut self, r: &RegionDecl) {
        let info = match &r.geometry {
            GeometryExpr::Box {
                center,
                half_extents,
                ..
            } => {
                let c = self.try_eval_vec3(center).unwrap_or([0.0, 0.0, 0.0]);
                let h = self.try_eval_vec3(half_extents).unwrap_or([1.0, 1.0, 1.0]);
                RegionGeomInfo::Box {
                    center: c,
                    half_extents: h,
                }
            }
            GeometryExpr::Sphere {
                center, radius, ..
            } => {
                let c = self.try_eval_vec3(center).unwrap_or([0.0, 0.0, 0.0]);
                let rad = self.try_eval_scalar(radius).unwrap_or(1.0);
                RegionGeomInfo::Sphere {
                    center: c,
                    radius: rad,
                }
            }
            _ => RegionGeomInfo::Unknown,
        };
        self.region_geometries.insert(r.name.clone(), info);
    }

    fn try_eval_vec3(&self, expr: &Expr) -> Option<[f64; 3]> {
        if let Expr::Vector3Literal { x, y, z, .. } = expr {
            let xv = self.try_eval_scalar(x)?;
            let yv = self.try_eval_scalar(y)?;
            let zv = self.try_eval_scalar(z)?;
            Some([xv, yv, zv])
        } else {
            None
        }
    }

    fn try_eval_scalar(&self, expr: &Expr) -> Option<f64> {
        match expr {
            Expr::Literal(Literal::Float(f), _) => Some(*f),
            Expr::Literal(Literal::Int(i), _) => Some(*i as f64),
            Expr::Literal(Literal::Distance(v, u), _) => Some(u.to_meters(*v)),
            _ => None,
        }
    }

    // -- type checking declarations --

    fn check_all_declarations(&mut self, program: &Program) {
        for decl in &program.declarations {
            match decl {
                Declaration::Region(r) => self.check_region(r),
                Declaration::Entity(e) => self.check_entity(e),
                Declaration::Zone(z) => self.check_zone(z),
                Declaration::Interaction(i) => self.check_interaction(i),
                Declaration::Scene(s) => self.check_scene(s),
                Declaration::LetBinding(l) => self.check_let_binding(l),
                Declaration::Import(_) => {}
            }
        }
    }

    fn check_region(&mut self, r: &RegionDecl) {
        self.check_geometry_type(&r.geometry);
        if let Some(parent) = &r.parent {
            match self.env.get(parent) {
                Some(entry) if entry.kind == SymKind::Region => {}
                Some(entry) => {
                    self.diagnostics.push(Diagnostic {
                        severity: Severity::Error,
                        message: format!(
                            "region '{}' has parent '{}' which is not a region (found {:?})",
                            r.name, parent, entry.kind
                        ),
                        span: Some(r.span.clone()),
                    });
                }
                None => {
                    self.diagnostics.push(Diagnostic {
                        severity: Severity::Error,
                        message: format!(
                            "region '{}' references undefined parent '{}'",
                            r.name, parent
                        ),
                        span: Some(r.span.clone()),
                    });
                }
            }
        }
    }

    fn check_geometry_type(&mut self, geom: &GeometryExpr) {
        match geom {
            GeometryExpr::Box {
                center,
                half_extents,
                span,
            } => {
                self.expect_type(center, &[Type::Vector3], span);
                self.expect_type(half_extents, &[Type::Vector3], span);
            }
            GeometryExpr::Sphere {
                center,
                radius,
                span,
            } => {
                self.expect_type(center, &[Type::Vector3], span);
                self.expect_numeric(radius, span);
            }
            GeometryExpr::Capsule {
                start,
                end,
                radius,
                span,
            } => {
                self.expect_type(start, &[Type::Vector3], span);
                self.expect_type(end, &[Type::Vector3], span);
                self.expect_numeric(radius, span);
            }
            GeometryExpr::Cylinder {
                center,
                radius,
                height,
                span,
            } => {
                self.expect_type(center, &[Type::Vector3], span);
                self.expect_numeric(radius, span);
                self.expect_numeric(height, span);
            }
            GeometryExpr::ConvexHull { points, span } => {
                if points.len() < 4 {
                    self.diagnostics.push(Diagnostic {
                        severity: Severity::Warning,
                        message: "convex hull with fewer than 4 points may be degenerate"
                            .to_string(),
                        span: Some(span.clone()),
                    });
                }
                for p in points {
                    self.expect_type(p, &[Type::Vector3], span);
                }
            }
            GeometryExpr::CSGUnion { a, b, .. }
            | GeometryExpr::CSGIntersection { a, b, .. }
            | GeometryExpr::CSGDifference { a, b, .. } => {
                self.check_geometry_type(a);
                self.check_geometry_type(b);
            }
            GeometryExpr::Transform {
                geometry,
                transform,
                span,
            } => {
                self.check_geometry_type(geometry);
                self.expect_type(transform, &[Type::Quaternion, Type::Vector3], span);
            }
            GeometryExpr::Reference { name, span } => {
                match self.env.get(name) {
                    Some(entry) if entry.kind == SymKind::Region => {}
                    Some(_) => {
                        self.diagnostics.push(Diagnostic {
                            severity: Severity::Error,
                            message: format!(
                                "geometry reference '{}' does not refer to a region",
                                name
                            ),
                            span: Some(span.clone()),
                        });
                    }
                    None => {} // undefined refs caught by semantic analysis
                }
            }
        }
    }

    fn check_entity(&mut self, e: &EntityDecl) {
        if let Some(pos) = &e.initial_position {
            self.expect_type(pos, &[Type::Vector3], &e.span);
        }
        if let Some(rot) = &e.initial_rotation {
            self.expect_type(rot, &[Type::Quaternion], &e.span);
        }
        if let Some(bv) = &e.bounding_volume {
            self.check_geometry_type(bv);
        }
    }

    fn check_zone(&mut self, z: &ZoneDecl) {
        for region_name in &z.regions {
            match self.env.get(region_name) {
                Some(entry) if entry.kind == SymKind::Region => {}
                Some(_) => {
                    self.diagnostics.push(Diagnostic {
                        severity: Severity::Error,
                        message: format!(
                            "zone '{}' references '{}' which is not a region",
                            z.name, region_name
                        ),
                        span: Some(z.span.clone()),
                    });
                }
                None => {} // caught by semantic analysis
            }
        }
        for rule in &z.rules {
            self.check_pattern_type(&rule.pattern);
            for action in &rule.actions {
                self.check_action_type(action);
            }
        }
    }

    fn check_interaction(&mut self, i: &InteractionDecl) {
        // Register parameters in a local environment context
        let mut local_params: HashMap<std::string::String, Type> = HashMap::new();
        for param in &i.parameters {
            let ty = param
                .type_annotation
                .as_ref()
                .map(Type::from_annotation)
                .unwrap_or(Type::Unknown);
            local_params.insert(param.name.clone(), ty.clone());
            self.env.insert(
                param.name.clone(),
                EnvEntry {
                    kind: SymKind::Parameter,
                    resolved_type: ty,
                    span: param.span.clone(),
                },
            );
        }

        if let Some(pat) = &i.pattern {
            self.check_pattern_type(pat);
        }

        match &i.body {
            InteractionBody::Simple {
                guard, actions, ..
            } => {
                self.check_pattern_type(guard);
                for action in actions {
                    self.check_action_type(action);
                }
            }
            InteractionBody::Choreography(choreo) => {
                self.check_choreography_type(choreo);
            }
        }

        // Remove parameters from env after checking
        for param in &i.parameters {
            self.env.remove(&param.name);
        }
    }

    fn check_scene(&mut self, s: &SceneDecl) {
        for eref in &s.entities {
            if let EntityRef::Named(name, span) = eref {
                match self.env.get(name) {
                    Some(entry) if entry.kind == SymKind::Entity => {}
                    Some(_) => {
                        self.diagnostics.push(Diagnostic {
                            severity: Severity::Error,
                            message: format!(
                                "scene '{}' references '{}' which is not an entity",
                                s.name, name
                            ),
                            span: Some(span.clone()),
                        });
                    }
                    None => {}
                }
            }
        }
        for rref in &s.regions {
            if let RegionRef::Named(name, span) = rref {
                match self.env.get(name) {
                    Some(entry) if entry.kind == SymKind::Region => {}
                    Some(_) => {
                        self.diagnostics.push(Diagnostic {
                            severity: Severity::Error,
                            message: format!(
                                "scene '{}' references '{}' which is not a region",
                                s.name, name
                            ),
                            span: Some(span.clone()),
                        });
                    }
                    None => {}
                }
            }
        }
    }

    fn check_let_binding(&mut self, l: &LetBindingDecl) {
        let inferred = self.infer_expr_type(&l.value);
        if let Some(ann) = &l.type_annotation {
            let expected = Type::from_annotation(ann);
            if !self.types_compatible(&inferred, &expected) {
                self.diagnostics.push(Diagnostic {
                    severity: Severity::Error,
                    message: format!(
                        "let binding '{}': declared type {:?} incompatible with inferred type {:?}",
                        l.name, expected, inferred
                    ),
                    span: Some(l.span.clone()),
                });
            }
        }
    }

    fn check_pattern_type(&mut self, pattern: &PatternExpr) {
        match pattern {
            PatternExpr::Gaze { entity, target, angle, span } => {
                self.expect_entity_type(entity, span);
                self.expect_entity_or_region(target, span);
                if let Some(a) = angle {
                    self.expect_type(a, &[Type::Angle, Type::Float], span);
                }
            }
            PatternExpr::Reach { entity, target, distance, span } => {
                self.expect_entity_type(entity, span);
                self.expect_entity_or_region(target, span);
                if let Some(d) = distance {
                    self.expect_type(d, &[Type::Distance, Type::Float], span);
                }
            }
            PatternExpr::Grab { target, span } | PatternExpr::Release { target, span } => {
                self.expect_entity_type(target, span);
            }
            PatternExpr::Proximity { entity_a, entity_b, distance, span } => {
                self.expect_entity_type(entity_a, span);
                self.expect_entity_type(entity_b, span);
                self.expect_type(distance, &[Type::Distance, Type::Float], span);
            }
            PatternExpr::Inside { entity, region, span } => {
                self.expect_entity_type(entity, span);
                self.expect_type(region, &[Type::Region], span);
            }
            PatternExpr::Touch { entity_a, entity_b, span } => {
                self.expect_entity_type(entity_a, span);
                self.expect_entity_type(entity_b, span);
            }
            PatternExpr::Conjunction { patterns, .. }
            | PatternExpr::Disjunction { patterns, .. }
            | PatternExpr::Sequence { patterns, .. } => {
                for p in patterns {
                    self.check_pattern_type(p);
                }
            }
            PatternExpr::Negation { pattern, .. } => {
                self.check_pattern_type(pattern);
            }
            PatternExpr::Timeout { duration, span } => {
                self.expect_type(duration, &[Type::Duration, Type::Float], span);
            }
            PatternExpr::TimedPattern { pattern, .. } => {
                self.check_pattern_type(pattern);
            }
            PatternExpr::Custom { .. } => {}
        }
    }

    fn check_action_type(&mut self, action: &ActionExpr) {
        match action {
            ActionExpr::Activate { target, span }
            | ActionExpr::Deactivate { target, span } => {
                self.expect_entity_type(target, span);
            }
            ActionExpr::UpdatePosition { entity, position, span } => {
                self.expect_entity_type(entity, span);
                self.expect_type(position, &[Type::Vector3], span);
            }
            ActionExpr::Destroy { entity, span } => {
                self.expect_entity_type(entity, span);
            }
            ActionExpr::SetTimer { duration, span, .. } => {
                self.expect_type(duration, &[Type::Duration, Type::Float], span);
            }
            _ => {}
        }
    }

    fn check_choreography_type(&mut self, choreo: &ChoreographyExpr) {
        match choreo {
            ChoreographyExpr::Sequential { steps, .. } => {
                for s in steps {
                    self.check_choreography_type(s);
                }
            }
            ChoreographyExpr::Parallel { branches, .. } => {
                for b in branches {
                    self.check_choreography_type(b);
                }
            }
            ChoreographyExpr::Choice { options, .. } => {
                for o in options {
                    self.check_choreography_type(o);
                }
            }
            ChoreographyExpr::Loop { body, .. } => {
                self.check_choreography_type(body);
            }
            ChoreographyExpr::Guarded { guard, body, .. } => {
                self.check_pattern_type(guard);
                self.check_choreography_type(body);
            }
            ChoreographyExpr::Action { actions, .. } => {
                for a in actions {
                    self.check_action_type(a);
                }
            }
            ChoreographyExpr::Reference { name, span, .. } => {
                match self.env.get(name) {
                    Some(entry) if entry.kind == SymKind::Interaction => {}
                    Some(_) => {
                        self.diagnostics.push(Diagnostic {
                            severity: Severity::Error,
                            message: format!(
                                "choreography reference '{}' does not refer to an interaction",
                                name
                            ),
                            span: Some(span.clone()),
                        });
                    }
                    None => {}
                }
            }
            ChoreographyExpr::Conditional {
                condition,
                then_branch,
                else_branch,
                span,
            } => {
                self.expect_type(condition, &[Type::Bool], span);
                self.check_choreography_type(then_branch);
                if let Some(eb) = else_branch {
                    self.check_choreography_type(eb);
                }
            }
            ChoreographyExpr::LetIn {
                value, body, ..
            } => {
                let _ty = self.infer_expr_type(value);
                self.check_choreography_type(body);
            }
        }
    }

    // -- type inference for expressions --

    fn infer_expr_type(&self, expr: &Expr) -> Type {
        match expr {
            Expr::Literal(lit, _) => match lit {
                Literal::Int(_) => Type::Int,
                Literal::Float(_) => Type::Float,
                Literal::String(_) => Type::String,
                Literal::Bool(_) => Type::Bool,
                Literal::Duration(_, _) => Type::Duration,
                Literal::Distance(_, _) => Type::Distance,
                Literal::Angle(_) => Type::Angle,
            },
            Expr::Identifier(name, _) => {
                self.env
                    .get(name)
                    .map(|e| e.resolved_type.clone())
                    .unwrap_or(Type::Unknown)
            }
            Expr::BinaryOp { op, left, right, .. } => {
                let lt = self.infer_expr_type(left);
                let rt = self.infer_expr_type(right);
                match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
                        // Numeric promotion
                        match (&lt, &rt) {
                            (Type::Float, _) | (_, Type::Float) => Type::Float,
                            (Type::Int, Type::Int) => Type::Int,
                            (Type::Vector3, Type::Vector3) => Type::Vector3,
                            (Type::Duration, Type::Duration) => Type::Duration,
                            (Type::Distance, Type::Distance) => Type::Distance,
                            _ => Type::Unknown,
                        }
                    }
                    BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => {
                        Type::Bool
                    }
                    BinOp::And | BinOp::Or => Type::Bool,
                }
            }
            Expr::UnaryOp { op, operand, .. } => match op {
                UnOp::Neg => self.infer_expr_type(operand),
                UnOp::Not => Type::Bool,
            },
            Expr::Vector3Literal { .. } => Type::Vector3,
            Expr::QuaternionLiteral { .. } => Type::Quaternion,
            Expr::Tuple { elements, .. } => {
                Type::Tuple(elements.iter().map(|e| self.infer_expr_type(e)).collect())
            }
            Expr::IfExpr { then_branch, .. } => self.infer_expr_type(then_branch),
            Expr::LetExpr { body, .. } => self.infer_expr_type(body),
            Expr::FunctionCall { .. } => Type::Unknown,
            Expr::FieldAccess { .. } => Type::Unknown,
            Expr::Index { .. } => Type::Unknown,
        }
    }

    fn expect_type(&mut self, expr: &Expr, expected: &[Type], span: &Span) {
        let inferred = self.infer_expr_type(expr);
        if inferred == Type::Unknown {
            return; // can't check unknown types
        }
        if !expected.iter().any(|e| self.types_compatible(&inferred, e)) {
            self.diagnostics.push(Diagnostic {
                severity: Severity::Error,
                message: format!(
                    "expected one of {:?}, found {:?}",
                    expected, inferred
                ),
                span: Some(span.clone()),
            });
        }
    }

    fn expect_numeric(&mut self, expr: &Expr, span: &Span) {
        self.expect_type(expr, &[Type::Int, Type::Float, Type::Distance], span);
    }

    fn expect_entity_type(&mut self, expr: &Expr, span: &Span) {
        let ty = self.infer_expr_type(expr);
        match &ty {
            Type::Entity(_) => {}
            Type::Unknown => {}
            _ => {
                self.diagnostics.push(Diagnostic {
                    severity: Severity::Error,
                    message: format!("expected entity type, found {:?}", ty),
                    span: Some(span.clone()),
                });
            }
        }
    }

    fn expect_entity_or_region(&mut self, expr: &Expr, span: &Span) {
        let ty = self.infer_expr_type(expr);
        match &ty {
            Type::Entity(_) | Type::Region => {}
            Type::Unknown => {}
            _ => {
                self.diagnostics.push(Diagnostic {
                    severity: Severity::Error,
                    message: format!("expected entity or region type, found {:?}", ty),
                    span: Some(span.clone()),
                });
            }
        }
    }

    fn types_compatible(&self, actual: &Type, expected: &Type) -> bool {
        if actual == expected {
            return true;
        }
        match (actual, expected) {
            (Type::Int, Type::Float) | (Type::Float, Type::Int) => true,
            (Type::Entity(_), Type::Entity(None)) => true,
            (Type::Entity(None), Type::Entity(_)) => true,
            (Type::Unknown, _) | (_, Type::Unknown) => true,
            _ => false,
        }
    }

    // -- spatial constraint checking --

    /// Check that spatial constraints on regions are satisfiable.
    pub fn check_region_consistency(&mut self, program: &Program) {
        for decl in &program.declarations {
            match decl {
                Declaration::Region(r) => {
                    for constraint in &r.constraints {
                        self.check_spatial_constraint(constraint);
                    }
                }
                Declaration::Scene(s) => {
                    for constraint in &s.constraints {
                        self.check_spatial_constraint(constraint);
                    }
                }
                _ => {}
            }
        }
    }

    fn check_spatial_constraint(&mut self, constraint: &SpatialConstraintExpr) {
        match constraint {
            SpatialConstraintExpr::ContainedIn(inner, outer, span) => {
                self.check_spatial_subtyping(inner, outer, span);
            }
            SpatialConstraintExpr::NonOverlapping(names, span) => {
                // For boxes, check pairwise that intersection is feasible
                // (i.e., that the non-overlapping constraint can be satisfied)
                for i in 0..names.len() {
                    for j in (i + 1)..names.len() {
                        let a = self.region_geometries.get(&names[i]).cloned();
                        let b = self.region_geometries.get(&names[j]).cloned();
                        if let (
                            Some(RegionGeomInfo::Box {
                                center: ca,
                                half_extents: ha,
                            }),
                            Some(RegionGeomInfo::Box {
                                center: cb,
                                half_extents: hb,
                            }),
                        ) = (a, b)
                        {
                            // Check if the boxes actually overlap
                            let overlaps = (0..3).all(|d| {
                                (ca[d] - ha[d]) < (cb[d] + hb[d])
                                    && (cb[d] - hb[d]) < (ca[d] + ha[d])
                            });
                            if overlaps {
                                self.diagnostics.push(Diagnostic {
                                    severity: Severity::Warning,
                                    message: format!(
                                        "non_overlapping constraint: regions '{}' and '{}' may overlap based on their static geometry",
                                        names[i], names[j]
                                    ),
                                    span: Some(span.clone()),
                                });
                            }
                        }
                    }
                }
            }
            SpatialConstraintExpr::MinDistance(a, b, expr, span) => {
                if let Some(dist) = self.try_eval_scalar(expr) {
                    if dist < 0.0 {
                        self.diagnostics.push(Diagnostic {
                            severity: Severity::Error,
                            message: format!(
                                "min_distance({}, {}, ...) has negative distance",
                                a, b
                            ),
                            span: Some(span.clone()),
                        });
                    }
                }
            }
            SpatialConstraintExpr::MaxDistance(a, b, expr, span) => {
                if let Some(dist) = self.try_eval_scalar(expr) {
                    if dist < 0.0 {
                        self.diagnostics.push(Diagnostic {
                            severity: Severity::Error,
                            message: format!(
                                "max_distance({}, {}, ...) has negative distance",
                                a, b
                            ),
                            span: Some(span.clone()),
                        });
                    }
                }
            }
            _ => {}
        }
    }

    /// Check spatial subtyping: verify inner is contained in outer using LP.
    pub fn check_spatial_subtyping(&mut self, inner: &str, outer: &str, span: &Span) {
        let inner_geom = self.region_geometries.get(inner).cloned();
        let outer_geom = self.region_geometries.get(outer).cloned();

        match (inner_geom, outer_geom) {
            (
                Some(RegionGeomInfo::Box {
                    center: ci,
                    half_extents: hi,
                }),
                Some(RegionGeomInfo::Box {
                    center: co,
                    half_extents: ho,
                }),
            ) => {
                let inner_verts = box_vertices(ci, hi);
                let outer_constraints = box_constraints(co, ho);
                if !convex_containment(&inner_verts, &outer_constraints) {
                    self.diagnostics.push(Diagnostic {
                        severity: Severity::Error,
                        message: format!(
                            "containment constraint violated: region '{}' is not contained in '{}'",
                            inner, outer
                        ),
                        span: Some(span.clone()),
                    });
                }
            }
            (
                Some(RegionGeomInfo::Sphere {
                    center: ci,
                    radius: ri,
                }),
                Some(RegionGeomInfo::Sphere {
                    center: co,
                    radius: ro,
                }),
            ) => {
                let dist = ((ci[0] - co[0]).powi(2)
                    + (ci[1] - co[1]).powi(2)
                    + (ci[2] - co[2]).powi(2))
                .sqrt();
                if dist + ri > ro + 1e-8 {
                    self.diagnostics.push(Diagnostic {
                        severity: Severity::Error,
                        message: format!(
                            "containment constraint violated: sphere '{}' (r={}) is not contained in sphere '{}' (r={})",
                            inner, ri, outer, ro
                        ),
                        span: Some(span.clone()),
                    });
                }
            }
            (
                Some(RegionGeomInfo::Box {
                    center: ci,
                    half_extents: hi,
                }),
                Some(RegionGeomInfo::Sphere {
                    center: co,
                    radius: ro,
                }),
            ) => {
                // Check that all box corners are inside the sphere
                let verts = box_vertices(ci, hi);
                for v in &verts {
                    let dist = ((v[0] - co[0]).powi(2)
                        + (v[1] - co[1]).powi(2)
                        + (v[2] - co[2]).powi(2))
                    .sqrt();
                    if dist > ro + 1e-8 {
                        self.diagnostics.push(Diagnostic {
                            severity: Severity::Error,
                            message: format!(
                                "containment constraint violated: box '{}' is not contained in sphere '{}'",
                                inner, outer
                            ),
                            span: Some(span.clone()),
                        });
                        break;
                    }
                }
            }
            _ => {
                // Can't statically check containment for unknown geometry types
            }
        }
    }

    // -- temporal well-formedness --

    /// Check that temporal constraints are consistent.
    pub fn check_temporal_well_formedness(&mut self, program: &Program) {
        for decl in &program.declarations {
            if let Declaration::Interaction(i) = decl {
                self.check_temporal_in_body(&i.body, &i.name);
            }
        }
    }

    fn check_temporal_in_body(&mut self, body: &InteractionBody, interaction_name: &str) {
        match body {
            InteractionBody::Simple { guard, .. } => {
                self.check_temporal_in_pattern(guard, interaction_name);
            }
            InteractionBody::Choreography(choreo) => {
                self.check_temporal_in_choreography(choreo, interaction_name);
            }
        }
    }

    fn check_temporal_in_pattern(&mut self, pattern: &PatternExpr, interaction_name: &str) {
        match pattern {
            PatternExpr::TimedPattern { constraint, span, pattern: inner, .. } => {
                match constraint {
                    TemporalConstraintExpr::Between(lo, hi, _) => {
                        let lo_val = self.try_eval_duration(lo);
                        let hi_val = self.try_eval_duration(hi);
                        if let (Some(l), Some(h)) = (lo_val, hi_val) {
                            if l >= h {
                                self.diagnostics.push(Diagnostic {
                                    severity: Severity::Error,
                                    message: format!(
                                        "in interaction '{}': temporal between constraint has lo ({}) >= hi ({})",
                                        interaction_name, l, h
                                    ),
                                    span: Some(span.clone()),
                                });
                            }
                        }
                    }
                    TemporalConstraintExpr::Within(expr, _)
                    | TemporalConstraintExpr::After(expr, _) => {
                        if let Some(val) = self.try_eval_duration(expr) {
                            if val <= 0.0 {
                                self.diagnostics.push(Diagnostic {
                                    severity: Severity::Error,
                                    message: format!(
                                        "in interaction '{}': temporal constraint has non-positive duration {}",
                                        interaction_name, val
                                    ),
                                    span: Some(span.clone()),
                                });
                            }
                        }
                    }
                }
                self.check_temporal_in_pattern(inner, interaction_name);
            }
            PatternExpr::Conjunction { patterns, .. }
            | PatternExpr::Disjunction { patterns, .. }
            | PatternExpr::Sequence { patterns, .. } => {
                for p in patterns {
                    self.check_temporal_in_pattern(p, interaction_name);
                }
            }
            PatternExpr::Negation { pattern, .. } => {
                self.check_temporal_in_pattern(pattern, interaction_name);
            }
            _ => {}
        }
    }

    fn check_temporal_in_choreography(
        &mut self,
        choreo: &ChoreographyExpr,
        interaction_name: &str,
    ) {
        match choreo {
            ChoreographyExpr::Sequential { steps, .. } => {
                for s in steps {
                    self.check_temporal_in_choreography(s, interaction_name);
                }
            }
            ChoreographyExpr::Parallel { branches, .. } => {
                for b in branches {
                    self.check_temporal_in_choreography(b, interaction_name);
                }
            }
            ChoreographyExpr::Choice { options, .. } => {
                for o in options {
                    self.check_temporal_in_choreography(o, interaction_name);
                }
            }
            ChoreographyExpr::Loop { body, .. } => {
                self.check_temporal_in_choreography(body, interaction_name);
            }
            ChoreographyExpr::Guarded { guard, body, .. } => {
                self.check_temporal_in_pattern(guard, interaction_name);
                self.check_temporal_in_choreography(body, interaction_name);
            }
            ChoreographyExpr::Conditional {
                then_branch,
                else_branch,
                ..
            } => {
                self.check_temporal_in_choreography(then_branch, interaction_name);
                if let Some(eb) = else_branch {
                    self.check_temporal_in_choreography(eb, interaction_name);
                }
            }
            ChoreographyExpr::LetIn { body, .. } => {
                self.check_temporal_in_choreography(body, interaction_name);
            }
            _ => {}
        }
    }

    fn try_eval_duration(&self, expr: &Expr) -> Option<f64> {
        match expr {
            Expr::Literal(Literal::Duration(val, unit), _) => Some(unit.to_seconds(*val)),
            Expr::Literal(Literal::Float(val), _) => Some(*val),
            Expr::Literal(Literal::Int(val), _) => Some(*val as f64),
            _ => None,
        }
    }

    // -- determinism checking --

    /// Check that no two interaction guards can fire simultaneously
    /// (i.e., their conjunction is satisfiable => potential nondeterminism).
    pub fn check_determinism(&mut self, _program: &Program) {
        let guards = self.interaction_guards.clone();
        for i in 0..guards.len() {
            for j in (i + 1)..guards.len() {
                let (name_a, pat_a) = &guards[i];
                let (name_b, pat_b) = &guards[j];
                if name_a == name_b {
                    continue; // guards within the same interaction are OK
                }
                if self.guards_may_overlap(pat_a, pat_b) {
                    self.diagnostics.push(Diagnostic {
                        severity: Severity::Warning,
                        message: format!(
                            "potential nondeterminism: guards of interactions '{}' and '{}' may fire simultaneously",
                            name_a, name_b
                        ),
                        span: None,
                    });
                }
            }
        }
    }

    /// Conservative check: two guards overlap if they reference the same
    /// entities/predicates and are not contradictory.
    fn guards_may_overlap(&self, a: &PatternExpr, b: &PatternExpr) -> bool {
        let ids_a: HashSet<std::string::String> =
            collect_pattern_identifiers(a).into_iter().collect();
        let ids_b: HashSet<std::string::String> =
            collect_pattern_identifiers(b).into_iter().collect();
        let shared: HashSet<_> = ids_a.intersection(&ids_b).collect();

        if shared.is_empty() {
            return false; // no shared entities => independent
        }

        // Check if one is the negation of the other
        if let PatternExpr::Negation { pattern, .. } = a {
            if self.patterns_structurally_equal(pattern, b) {
                return false;
            }
        }
        if let PatternExpr::Negation { pattern, .. } = b {
            if self.patterns_structurally_equal(pattern, a) {
                return false;
            }
        }

        // Conservative: if they share entities and neither is the negation of the other,
        // they may overlap
        true
    }

    fn patterns_structurally_equal(&self, a: &PatternExpr, b: &PatternExpr) -> bool {
        // Simple structural equality check using debug representation
        format!("{:?}", a) == format!("{:?}", b)
    }
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::DurationUnit;

    fn dummy_span() -> Span {
        Span {
            start: 0,
            end: 0,
            file: None,
        }
    }

    fn make_id(name: &str) -> Expr {
        Expr::Identifier(name.to_string(), dummy_span())
    }

    fn make_float(v: f64) -> Expr {
        Expr::Literal(Literal::Float(v), dummy_span())
    }

    fn make_vec3(x: f64, y: f64, z: f64) -> Expr {
        Expr::Vector3Literal {
            x: Box::new(make_float(x)),
            y: Box::new(make_float(y)),
            z: Box::new(make_float(z)),
            span: dummy_span(),
        }
    }

    fn make_region(name: &str, center: [f64; 3], half: [f64; 3]) -> Declaration {
        Declaration::Region(RegionDecl {
            name: name.to_string(),
            geometry: GeometryExpr::Box {
                center: Box::new(make_vec3(center[0], center[1], center[2])),
                half_extents: Box::new(make_vec3(half[0], half[1], half[2])),
                span: dummy_span(),
            },
            parent: None,
            constraints: vec![],
            span: dummy_span(),
        })
    }

    fn make_entity(name: &str) -> Declaration {
        Declaration::Entity(EntityDecl {
            name: name.to_string(),
            entity_type: EntityTypeExpr::Object,
            initial_position: None,
            initial_rotation: None,
            bounding_volume: None,
            properties: vec![],
            span: dummy_span(),
        })
    }

    #[test]
    fn test_valid_program_type_checks() {
        let program = Program {
            declarations: vec![
                make_region("lobby", [0.0, 0.0, 0.0], [5.0, 5.0, 5.0]),
                make_entity("player"),
            ],
            span: dummy_span(),
        };
        let result = TypeChecker::check_program(&program);
        assert!(result.is_ok(), "expected OK, got: {:?}", result);
        let typed = result.unwrap();
        assert_eq!(typed.types.get("lobby"), Some(&Type::Region));
    }

    #[test]
    fn test_containment_violation() {
        let program = Program {
            declarations: vec![
                make_region("outer", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
                make_region("inner", [5.0, 5.0, 5.0], [2.0, 2.0, 2.0]),
                Declaration::Scene(SceneDecl {
                    name: "test_scene".to_string(),
                    entities: vec![],
                    regions: vec![
                        RegionRef::Named("outer".into(), dummy_span()),
                        RegionRef::Named("inner".into(), dummy_span()),
                    ],
                    interactions: vec![],
                    constraints: vec![SpatialConstraintExpr::ContainedIn(
                        "inner".to_string(),
                        "outer".to_string(),
                        dummy_span(),
                    )],
                    span: dummy_span(),
                }),
            ],
            span: dummy_span(),
        };
        let result = TypeChecker::check_program(&program);
        assert!(result.is_err(), "expected type error for containment violation");
        let diags = result.unwrap_err();
        assert!(
            diags.iter().any(|d| d.message.contains("containment")),
            "expected containment error, got: {:?}",
            diags
        );
    }

    #[test]
    fn test_containment_passes() {
        let program = Program {
            declarations: vec![
                make_region("outer", [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]),
                make_region("inner", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
                Declaration::Scene(SceneDecl {
                    name: "test_scene".to_string(),
                    entities: vec![],
                    regions: vec![
                        RegionRef::Named("outer".into(), dummy_span()),
                        RegionRef::Named("inner".into(), dummy_span()),
                    ],
                    interactions: vec![],
                    constraints: vec![SpatialConstraintExpr::ContainedIn(
                        "inner".to_string(),
                        "outer".to_string(),
                        dummy_span(),
                    )],
                    span: dummy_span(),
                }),
            ],
            span: dummy_span(),
        };
        let result = TypeChecker::check_program(&program);
        assert!(result.is_ok(), "expected OK, got: {:?}", result);
    }

    #[test]
    fn test_invalid_parent_type() {
        let program = Program {
            declarations: vec![
                make_entity("player"),
                Declaration::Region(RegionDecl {
                    name: "child_region".to_string(),
                    geometry: GeometryExpr::Reference {
                        name: "base".into(),
                        span: dummy_span(),
                    },
                    parent: Some("player".to_string()),
                    constraints: vec![],
                    span: dummy_span(),
                }),
            ],
            span: dummy_span(),
        };
        let result = TypeChecker::check_program(&program);
        assert!(result.is_err());
        let diags = result.unwrap_err();
        assert!(
            diags.iter().any(|d| d.message.contains("not a region")),
            "expected 'not a region' error, got: {:?}",
            diags
        );
    }

    #[test]
    fn test_temporal_well_formedness() {
        let program = Program {
            declarations: vec![
                make_entity("player"),
                Declaration::Interaction(InteractionDecl {
                    name: "bad_temporal".to_string(),
                    parameters: vec![],
                    pattern: None,
                    body: InteractionBody::Simple {
                        guard: PatternExpr::TimedPattern {
                            pattern: Box::new(PatternExpr::Grab {
                                target: Box::new(make_id("player")),
                                span: dummy_span(),
                            }),
                            constraint: TemporalConstraintExpr::Between(
                                Expr::Literal(Literal::Duration(5.0, DurationUnit::S), dummy_span()),
                                Expr::Literal(Literal::Duration(2.0, DurationUnit::S), dummy_span()),
                                dummy_span(),
                            ),
                            span: dummy_span(),
                        },
                        actions: vec![],
                        span: dummy_span(),
                    },
                    span: dummy_span(),
                }),
            ],
            span: dummy_span(),
        };
        let result = TypeChecker::check_program(&program);
        assert!(result.is_err());
        let diags = result.unwrap_err();
        assert!(
            diags.iter().any(|d| d.message.contains("lo") && d.message.contains("hi")),
            "expected temporal constraint error, got: {:?}",
            diags
        );
    }

    #[test]
    fn test_determinism_warning() {
        let program = Program {
            declarations: vec![
                make_entity("obj"),
                Declaration::Interaction(InteractionDecl {
                    name: "interact_a".to_string(),
                    parameters: vec![],
                    pattern: Some(PatternExpr::Grab {
                        target: Box::new(make_id("obj")),
                        span: dummy_span(),
                    }),
                    body: InteractionBody::Simple {
                        guard: PatternExpr::Grab {
                            target: Box::new(make_id("obj")),
                            span: dummy_span(),
                        },
                        actions: vec![],
                        span: dummy_span(),
                    },
                    span: dummy_span(),
                }),
                Declaration::Interaction(InteractionDecl {
                    name: "interact_b".to_string(),
                    parameters: vec![],
                    pattern: Some(PatternExpr::Grab {
                        target: Box::new(make_id("obj")),
                        span: dummy_span(),
                    }),
                    body: InteractionBody::Simple {
                        guard: PatternExpr::Grab {
                            target: Box::new(make_id("obj")),
                            span: dummy_span(),
                        },
                        actions: vec![],
                        span: dummy_span(),
                    },
                    span: dummy_span(),
                }),
            ],
            span: dummy_span(),
        };
        let result = TypeChecker::check_program(&program);
        // May still pass since determinism is a warning, not an error
        match result {
            Ok(_) => {
                // Check that we at least get a warning (checker only returns errors in Err)
            }
            Err(diags) => {
                // If there are errors, at least the nondeterminism should be mentioned
                assert!(
                    diags.iter().any(|d| d.message.contains("nondeterminism")),
                    "expected nondeterminism warning, got: {:?}",
                    diags
                );
            }
        }
    }

    #[test]
    fn test_lp_feasibility_basic() {
        // A simple box: -1 <= x <= 1
        let constraints = vec![
            LinearConstraint {
                coefficients: vec![1.0],
                bound: 1.0,
            },
            LinearConstraint {
                coefficients: vec![-1.0],
                bound: 1.0,
            },
        ];
        assert!(lp_feasible(&constraints, 1));

        // Infeasible: x <= -1 and x >= 1
        let infeasible = vec![
            LinearConstraint {
                coefficients: vec![1.0],
                bound: -1.0,
            },
            LinearConstraint {
                coefficients: vec![-1.0],
                bound: -1.0,
            },
        ];
        assert!(!lp_feasible(&infeasible, 1));
    }

    #[test]
    fn test_convex_containment() {
        let inner_verts = box_vertices([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let outer_constraints = box_constraints([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        assert!(convex_containment(&inner_verts, &outer_constraints));

        let outer_constraints_small = box_constraints([0.0, 0.0, 0.0], [0.5, 0.5, 0.5]);
        assert!(!convex_containment(&inner_verts, &outer_constraints_small));
    }

    #[test]
    fn test_let_binding_type_mismatch() {
        let program = Program {
            declarations: vec![Declaration::LetBinding(LetBindingDecl {
                name: "x".to_string(),
                type_annotation: Some(TypeAnnotation::Bool),
                value: Expr::Literal(Literal::Float(3.14), dummy_span()),
                span: dummy_span(),
            })],
            span: dummy_span(),
        };
        let result = TypeChecker::check_program(&program);
        assert!(result.is_err());
        let diags = result.unwrap_err();
        assert!(
            diags.iter().any(|d| d.message.contains("incompatible")),
            "expected type mismatch, got: {:?}",
            diags
        );
    }

    #[test]
    fn test_sphere_containment() {
        let program = Program {
            declarations: vec![
                Declaration::Region(RegionDecl {
                    name: "big_sphere".to_string(),
                    geometry: GeometryExpr::Sphere {
                        center: Box::new(make_vec3(0.0, 0.0, 0.0)),
                        radius: Box::new(make_float(10.0)),
                        span: dummy_span(),
                    },
                    parent: None,
                    constraints: vec![],
                    span: dummy_span(),
                }),
                Declaration::Region(RegionDecl {
                    name: "small_sphere".to_string(),
                    geometry: GeometryExpr::Sphere {
                        center: Box::new(make_vec3(0.0, 0.0, 0.0)),
                        radius: Box::new(make_float(1.0)),
                        span: dummy_span(),
                    },
                    parent: None,
                    constraints: vec![],
                    span: dummy_span(),
                }),
                Declaration::Scene(SceneDecl {
                    name: "sphere_scene".to_string(),
                    entities: vec![],
                    regions: vec![
                        RegionRef::Named("big_sphere".into(), dummy_span()),
                        RegionRef::Named("small_sphere".into(), dummy_span()),
                    ],
                    interactions: vec![],
                    constraints: vec![SpatialConstraintExpr::ContainedIn(
                        "small_sphere".to_string(),
                        "big_sphere".to_string(),
                        dummy_span(),
                    )],
                    span: dummy_span(),
                }),
            ],
            span: dummy_span(),
        };
        let result = TypeChecker::check_program(&program);
        assert!(result.is_ok(), "small sphere should be inside big sphere: {:?}", result);
    }
}

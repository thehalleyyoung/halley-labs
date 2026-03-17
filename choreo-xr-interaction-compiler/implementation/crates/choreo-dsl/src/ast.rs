//! Abstract Syntax Tree for the Choreo DSL.
//!
//! Represents the full grammar of the Choreo spatial-temporal choreography
//! language, including region declarations, entity declarations, interaction
//! definitions, scene compositions, geometry expressions, pattern expressions,
//! action expressions, choreography combinators, and general expressions.

use choreo_types::Span;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Top-level program
// ---------------------------------------------------------------------------

/// A complete Choreo program consisting of a sequence of declarations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Program {
    pub declarations: Vec<Declaration>,
    pub span: Span,
}

impl Program {
    pub fn new(declarations: Vec<Declaration>, span: Span) -> Self {
        Self { declarations, span }
    }
}

// ---------------------------------------------------------------------------
// Declarations
// ---------------------------------------------------------------------------

/// A top-level declaration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Declaration {
    Region(RegionDecl),
    Entity(EntityDecl),
    Zone(ZoneDecl),
    Interaction(InteractionDecl),
    Scene(SceneDecl),
    LetBinding(LetBindingDecl),
    Import(ImportDecl),
}

impl Declaration {
    pub fn span(&self) -> &Span {
        match self {
            Declaration::Region(d) => &d.span,
            Declaration::Entity(d) => &d.span,
            Declaration::Zone(d) => &d.span,
            Declaration::Interaction(d) => &d.span,
            Declaration::Scene(d) => &d.span,
            Declaration::LetBinding(d) => &d.span,
            Declaration::Import(d) => &d.span,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Declaration::Region(d) => &d.name,
            Declaration::Entity(d) => &d.name,
            Declaration::Zone(d) => &d.name,
            Declaration::Interaction(d) => &d.name,
            Declaration::Scene(d) => &d.name,
            Declaration::LetBinding(d) => &d.name,
            Declaration::Import(d) => &d.path,
        }
    }
}

/// A region declaration defining a named spatial region.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegionDecl {
    pub name: String,
    pub geometry: GeometryExpr,
    pub parent: Option<String>,
    pub constraints: Vec<SpatialConstraintExpr>,
    pub span: Span,
}

/// An entity declaration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityDecl {
    pub name: String,
    pub entity_type: EntityTypeExpr,
    pub initial_position: Option<Expr>,
    pub initial_rotation: Option<Expr>,
    pub bounding_volume: Option<GeometryExpr>,
    pub properties: Vec<(String, Expr)>,
    pub span: Span,
}

/// Entity type annotation in the DSL.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityTypeExpr {
    User,
    Hand(HandSideExpr),
    Head,
    Object,
    Controller(HandSideExpr),
    Anchor,
    Custom(String),
}

/// Hand side expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HandSideExpr {
    Left,
    Right,
}

/// A zone declaration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ZoneDecl {
    pub name: String,
    pub regions: Vec<String>,
    pub rules: Vec<ZoneRule>,
    pub span: Span,
}

/// A rule within a zone.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ZoneRule {
    pub pattern: PatternExpr,
    pub actions: Vec<ActionExpr>,
    pub span: Span,
}

/// An interaction declaration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InteractionDecl {
    pub name: String,
    pub parameters: Vec<Parameter>,
    pub pattern: Option<PatternExpr>,
    pub body: InteractionBody,
    pub span: Span,
}

/// A parameter with optional type annotation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Parameter {
    pub name: String,
    pub type_annotation: Option<TypeAnnotation>,
    pub span: Span,
}

/// A type annotation in the DSL.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypeAnnotation {
    Named(String),
    Region,
    Entity,
    Duration,
    Distance,
    Angle,
    Bool,
    Int,
    Float,
    String_,
    Vector3,
    Quaternion,
    Tuple(Vec<TypeAnnotation>),
    Function(Vec<TypeAnnotation>, Box<TypeAnnotation>),
    Span(Span),
}

/// A scene declaration composing entities, regions, and interactions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneDecl {
    pub name: String,
    pub entities: Vec<EntityRef>,
    pub regions: Vec<RegionRef>,
    pub interactions: Vec<InteractionRef>,
    pub constraints: Vec<SpatialConstraintExpr>,
    pub span: Span,
}

/// A reference to an entity (possibly inline).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EntityRef {
    Named(String, Span),
    Inline(EntityDecl),
}

/// A reference to a region (possibly inline).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RegionRef {
    Named(String, Span),
    Inline(RegionDecl),
}

/// A reference to an interaction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InteractionRef {
    Named(String, Vec<Expr>, Span),
    Inline(InteractionDecl),
}

/// A let binding at the top level.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LetBindingDecl {
    pub name: String,
    pub type_annotation: Option<TypeAnnotation>,
    pub value: Expr,
    pub span: Span,
}

/// An import declaration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportDecl {
    pub path: String,
    pub items: Option<Vec<String>>,
    pub span: Span,
}

// ---------------------------------------------------------------------------
// Spatial constraint expression
// ---------------------------------------------------------------------------

/// Spatial constraints that can appear in region/scene declarations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SpatialConstraintExpr {
    NonOverlapping(Vec<String>, Span),
    ContainedIn(String, String, Span),
    MinDistance(String, String, Expr, Span),
    MaxDistance(String, String, Expr, Span),
    WithinRegion(String, String, Span),
    Custom(String, Vec<Expr>, Span),
}

// ---------------------------------------------------------------------------
// Geometry expressions
// ---------------------------------------------------------------------------

/// Geometry expressions for defining spatial regions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GeometryExpr {
    /// `box(center, half_extents)`
    Box {
        center: Box<Expr>,
        half_extents: Box<Expr>,
        span: Span,
    },
    /// `sphere(center, radius)`
    Sphere {
        center: Box<Expr>,
        radius: Box<Expr>,
        span: Span,
    },
    /// `capsule(start, end, radius)`
    Capsule {
        start: Box<Expr>,
        end: Box<Expr>,
        radius: Box<Expr>,
        span: Span,
    },
    /// `cylinder(center, radius, height)`
    Cylinder {
        center: Box<Expr>,
        radius: Box<Expr>,
        height: Box<Expr>,
        span: Span,
    },
    /// `convex_hull(points...)`
    ConvexHull {
        points: Vec<Expr>,
        span: Span,
    },
    /// `union(a, b)`
    CSGUnion {
        a: Box<GeometryExpr>,
        b: Box<GeometryExpr>,
        span: Span,
    },
    /// `intersection(a, b)`
    CSGIntersection {
        a: Box<GeometryExpr>,
        b: Box<GeometryExpr>,
        span: Span,
    },
    /// `difference(a, b)`
    CSGDifference {
        a: Box<GeometryExpr>,
        b: Box<GeometryExpr>,
        span: Span,
    },
    /// `transform(geometry, transform_expr)`
    Transform {
        geometry: Box<GeometryExpr>,
        transform: Box<Expr>,
        span: Span,
    },
    /// Reference to a named region
    Reference {
        name: String,
        span: Span,
    },
}

impl GeometryExpr {
    pub fn span(&self) -> &Span {
        match self {
            GeometryExpr::Box { span, .. }
            | GeometryExpr::Sphere { span, .. }
            | GeometryExpr::Capsule { span, .. }
            | GeometryExpr::Cylinder { span, .. }
            | GeometryExpr::ConvexHull { span, .. }
            | GeometryExpr::CSGUnion { span, .. }
            | GeometryExpr::CSGIntersection { span, .. }
            | GeometryExpr::CSGDifference { span, .. }
            | GeometryExpr::Transform { span, .. }
            | GeometryExpr::Reference { span, .. } => span,
        }
    }
}

// ---------------------------------------------------------------------------
// Pattern expressions
// ---------------------------------------------------------------------------

/// Pattern expressions for matching spatial-temporal events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PatternExpr {
    /// `gaze(entity, target, angle?)`
    Gaze {
        entity: Box<Expr>,
        target: Box<Expr>,
        angle: Option<Box<Expr>>,
        span: Span,
    },
    /// `reach(entity, target, distance?)`
    Reach {
        entity: Box<Expr>,
        target: Box<Expr>,
        distance: Option<Box<Expr>>,
        span: Span,
    },
    /// `grab(entity)`
    Grab {
        target: Box<Expr>,
        span: Span,
    },
    /// `release(entity)`
    Release {
        target: Box<Expr>,
        span: Span,
    },
    /// `proximity(a, b, distance)`
    Proximity {
        entity_a: Box<Expr>,
        entity_b: Box<Expr>,
        distance: Box<Expr>,
        span: Span,
    },
    /// `inside(entity, region)`
    Inside {
        entity: Box<Expr>,
        region: Box<Expr>,
        span: Span,
    },
    /// `touch(a, b)`
    Touch {
        entity_a: Box<Expr>,
        entity_b: Box<Expr>,
        span: Span,
    },
    /// `p1 and p2 and ...`
    Conjunction {
        patterns: Vec<PatternExpr>,
        span: Span,
    },
    /// `p1 or p2 or ...`
    Disjunction {
        patterns: Vec<PatternExpr>,
        span: Span,
    },
    /// `p1 ; p2 ; ...` (sequential pattern match)
    Sequence {
        patterns: Vec<PatternExpr>,
        span: Span,
    },
    /// `not p`
    Negation {
        pattern: Box<PatternExpr>,
        span: Span,
    },
    /// `timeout(duration)`
    Timeout {
        duration: Box<Expr>,
        span: Span,
    },
    /// `p within duration` or `p after duration`
    TimedPattern {
        pattern: Box<PatternExpr>,
        constraint: TemporalConstraintExpr,
        span: Span,
    },
    /// A custom/named pattern
    Custom {
        name: String,
        args: Vec<Expr>,
        span: Span,
    },
}

impl PatternExpr {
    pub fn span(&self) -> &Span {
        match self {
            PatternExpr::Gaze { span, .. }
            | PatternExpr::Reach { span, .. }
            | PatternExpr::Grab { span, .. }
            | PatternExpr::Release { span, .. }
            | PatternExpr::Proximity { span, .. }
            | PatternExpr::Inside { span, .. }
            | PatternExpr::Touch { span, .. }
            | PatternExpr::Conjunction { span, .. }
            | PatternExpr::Disjunction { span, .. }
            | PatternExpr::Sequence { span, .. }
            | PatternExpr::Negation { span, .. }
            | PatternExpr::Timeout { span, .. }
            | PatternExpr::TimedPattern { span, .. }
            | PatternExpr::Custom { span, .. } => span,
        }
    }
}

/// A temporal constraint within a pattern.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TemporalConstraintExpr {
    Within(Expr, Span),
    After(Expr, Span),
    Between(Expr, Expr, Span),
}

// ---------------------------------------------------------------------------
// Action expressions
// ---------------------------------------------------------------------------

/// Actions that can be triggered in response to patterns.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ActionExpr {
    /// `activate(target)`
    Activate {
        target: Box<Expr>,
        span: Span,
    },
    /// `deactivate(target)`
    Deactivate {
        target: Box<Expr>,
        span: Span,
    },
    /// `emit(event_name, args...)`
    Emit {
        event_name: String,
        args: Vec<Expr>,
        span: Span,
    },
    /// `set_timer(name, duration)`
    SetTimer {
        name: String,
        duration: Box<Expr>,
        span: Span,
    },
    /// `cancel_timer(name)`
    CancelTimer {
        name: String,
        span: Span,
    },
    /// Update entity position
    UpdatePosition {
        entity: Box<Expr>,
        position: Box<Expr>,
        span: Span,
    },
    /// `spawn(entity_decl)`
    Spawn {
        entity: Box<EntityDecl>,
        span: Span,
    },
    /// `destroy(entity)`
    Destroy {
        entity: Box<Expr>,
        span: Span,
    },
    /// Custom named action
    Custom {
        name: String,
        args: Vec<Expr>,
        span: Span,
    },
}

impl ActionExpr {
    pub fn span(&self) -> &Span {
        match self {
            ActionExpr::Activate { span, .. }
            | ActionExpr::Deactivate { span, .. }
            | ActionExpr::Emit { span, .. }
            | ActionExpr::SetTimer { span, .. }
            | ActionExpr::CancelTimer { span, .. }
            | ActionExpr::UpdatePosition { span, .. }
            | ActionExpr::Spawn { span, .. }
            | ActionExpr::Destroy { span, .. }
            | ActionExpr::Custom { span, .. } => span,
        }
    }
}

// ---------------------------------------------------------------------------
// Interaction body
// ---------------------------------------------------------------------------

/// The body of an interaction declaration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InteractionBody {
    /// `when pattern then { actions }`
    Simple {
        guard: PatternExpr,
        actions: Vec<ActionExpr>,
        span: Span,
    },
    /// Choreography expression
    Choreography(ChoreographyExpr),
}

// ---------------------------------------------------------------------------
// Choreography expressions
// ---------------------------------------------------------------------------

/// Choreography combinators.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChoreographyExpr {
    /// `seq { step1; step2; ... }`
    Sequential {
        steps: Vec<ChoreographyExpr>,
        span: Span,
    },
    /// `par { branch1 | branch2 | ... }`
    Parallel {
        branches: Vec<ChoreographyExpr>,
        span: Span,
    },
    /// `choice { option1 | option2 | ... }`
    Choice {
        options: Vec<ChoreographyExpr>,
        span: Span,
    },
    /// `loop(bound?) { body }`
    Loop {
        body: Box<ChoreographyExpr>,
        bound: Option<Box<Expr>>,
        span: Span,
    },
    /// `when pattern => choreography`
    Guarded {
        guard: PatternExpr,
        body: Box<ChoreographyExpr>,
        span: Span,
    },
    /// A list of actions
    Action {
        actions: Vec<ActionExpr>,
        span: Span,
    },
    /// Reference to a named choreography
    Reference {
        name: String,
        args: Vec<Expr>,
        span: Span,
    },
    /// `if expr { ... } else { ... }`
    Conditional {
        condition: Box<Expr>,
        then_branch: Box<ChoreographyExpr>,
        else_branch: Option<Box<ChoreographyExpr>>,
        span: Span,
    },
    /// `let name = expr in choreography`
    LetIn {
        name: String,
        value: Box<Expr>,
        body: Box<ChoreographyExpr>,
        span: Span,
    },
}

impl ChoreographyExpr {
    pub fn span(&self) -> &Span {
        match self {
            ChoreographyExpr::Sequential { span, .. }
            | ChoreographyExpr::Parallel { span, .. }
            | ChoreographyExpr::Choice { span, .. }
            | ChoreographyExpr::Loop { span, .. }
            | ChoreographyExpr::Guarded { span, .. }
            | ChoreographyExpr::Action { span, .. }
            | ChoreographyExpr::Reference { span, .. }
            | ChoreographyExpr::Conditional { span, .. }
            | ChoreographyExpr::LetIn { span, .. } => span,
        }
    }
}

// ---------------------------------------------------------------------------
// General expressions
// ---------------------------------------------------------------------------

/// General expressions used in various positions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    /// A literal value
    Literal(Literal, Span),
    /// An identifier reference
    Identifier(String, Span),
    /// `a op b`
    BinaryOp {
        op: BinOp,
        left: Box<Expr>,
        right: Box<Expr>,
        span: Span,
    },
    /// `op a`
    UnaryOp {
        op: UnOp,
        operand: Box<Expr>,
        span: Span,
    },
    /// `f(args...)`
    FunctionCall {
        function: Box<Expr>,
        args: Vec<Expr>,
        span: Span,
    },
    /// `expr.field`
    FieldAccess {
        object: Box<Expr>,
        field: String,
        span: Span,
    },
    /// `expr[index]`
    Index {
        object: Box<Expr>,
        index: Box<Expr>,
        span: Span,
    },
    /// `(a, b, c)`
    Tuple {
        elements: Vec<Expr>,
        span: Span,
    },
    /// `vec3(x, y, z)`
    Vector3Literal {
        x: Box<Expr>,
        y: Box<Expr>,
        z: Box<Expr>,
        span: Span,
    },
    /// `quat(w, x, y, z)`
    QuaternionLiteral {
        w: Box<Expr>,
        x: Box<Expr>,
        y: Box<Expr>,
        z: Box<Expr>,
        span: Span,
    },
    /// `if cond { then } else { else }`
    IfExpr {
        condition: Box<Expr>,
        then_branch: Box<Expr>,
        else_branch: Option<Box<Expr>>,
        span: Span,
    },
    /// `let name = value in body`
    LetExpr {
        name: String,
        value: Box<Expr>,
        body: Box<Expr>,
        span: Span,
    },
}

impl Expr {
    pub fn span(&self) -> &Span {
        match self {
            Expr::Literal(_, span)
            | Expr::Identifier(_, span)
            | Expr::BinaryOp { span, .. }
            | Expr::UnaryOp { span, .. }
            | Expr::FunctionCall { span, .. }
            | Expr::FieldAccess { span, .. }
            | Expr::Index { span, .. }
            | Expr::Tuple { span, .. }
            | Expr::Vector3Literal { span, .. }
            | Expr::QuaternionLiteral { span, .. }
            | Expr::IfExpr { span, .. }
            | Expr::LetExpr { span, .. } => span,
        }
    }
}

/// Literal values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Literal {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
    Duration(f64, crate::token::DurationUnit),
    Distance(f64, crate::token::DistanceUnit),
    Angle(f64),
}

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
}

impl BinOp {
    /// Binding power (precedence) for Pratt parsing: (left_bp, right_bp).
    pub fn binding_power(self) -> (u8, u8) {
        match self {
            BinOp::Or => (1, 2),
            BinOp::And => (3, 4),
            BinOp::Eq | BinOp::Ne => (5, 6),
            BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => (7, 8),
            BinOp::Add | BinOp::Sub => (9, 10),
            BinOp::Mul | BinOp::Div => (11, 12),
        }
    }
}

impl std::fmt::Display for BinOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinOp::Add => write!(f, "+"),
            BinOp::Sub => write!(f, "-"),
            BinOp::Mul => write!(f, "*"),
            BinOp::Div => write!(f, "/"),
            BinOp::Eq => write!(f, "="),
            BinOp::Ne => write!(f, "!="),
            BinOp::Lt => write!(f, "<"),
            BinOp::Gt => write!(f, ">"),
            BinOp::Le => write!(f, "<="),
            BinOp::Ge => write!(f, ">="),
            BinOp::And => write!(f, "and"),
            BinOp::Or => write!(f, "or"),
        }
    }
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnOp {
    Neg,
    Not,
}

impl std::fmt::Display for UnOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnOp::Neg => write!(f, "-"),
            UnOp::Not => write!(f, "not"),
        }
    }
}

// ---------------------------------------------------------------------------
// Visitor trait
// ---------------------------------------------------------------------------

/// Visitor trait for traversing the AST.
pub trait Visitor {
    type Result;

    fn visit_program(&mut self, program: &Program) -> Self::Result;
    fn visit_declaration(&mut self, decl: &Declaration) -> Self::Result;
    fn visit_region_decl(&mut self, decl: &RegionDecl) -> Self::Result;
    fn visit_entity_decl(&mut self, decl: &EntityDecl) -> Self::Result;
    fn visit_interaction_decl(&mut self, decl: &InteractionDecl) -> Self::Result;
    fn visit_scene_decl(&mut self, decl: &SceneDecl) -> Self::Result;
    fn visit_geometry_expr(&mut self, expr: &GeometryExpr) -> Self::Result;
    fn visit_pattern_expr(&mut self, expr: &PatternExpr) -> Self::Result;
    fn visit_action_expr(&mut self, expr: &ActionExpr) -> Self::Result;
    fn visit_choreography_expr(&mut self, expr: &ChoreographyExpr) -> Self::Result;
    fn visit_expr(&mut self, expr: &Expr) -> Self::Result;
}

/// Mutable visitor trait for transforming the AST.
pub trait MutVisitor {
    fn visit_program(&mut self, program: &mut Program);
    fn visit_declaration(&mut self, decl: &mut Declaration);
    fn visit_expr(&mut self, expr: &mut Expr);
    fn visit_pattern_expr(&mut self, pattern: &mut PatternExpr);
    fn visit_action_expr(&mut self, action: &mut ActionExpr);
    fn visit_choreography_expr(&mut self, choreo: &mut ChoreographyExpr);
    fn visit_geometry_expr(&mut self, geom: &mut GeometryExpr);
}

// ---------------------------------------------------------------------------
// Helper: collect all identifiers referenced in an expression
// ---------------------------------------------------------------------------

/// Collect all free identifiers in an expression.
pub fn collect_identifiers(expr: &Expr) -> Vec<String> {
    let mut result = Vec::new();
    collect_ids_inner(expr, &mut result);
    result
}

fn collect_ids_inner(expr: &Expr, out: &mut Vec<String>) {
    match expr {
        Expr::Identifier(name, _) => out.push(name.clone()),
        Expr::BinaryOp { left, right, .. } => {
            collect_ids_inner(left, out);
            collect_ids_inner(right, out);
        }
        Expr::UnaryOp { operand, .. } => {
            collect_ids_inner(operand, out);
        }
        Expr::FunctionCall { function, args, .. } => {
            collect_ids_inner(function, out);
            for arg in args {
                collect_ids_inner(arg, out);
            }
        }
        Expr::FieldAccess { object, .. } => {
            collect_ids_inner(object, out);
        }
        Expr::Index { object, index, .. } => {
            collect_ids_inner(object, out);
            collect_ids_inner(index, out);
        }
        Expr::Tuple { elements, .. } => {
            for e in elements {
                collect_ids_inner(e, out);
            }
        }
        Expr::Vector3Literal { x, y, z, .. } => {
            collect_ids_inner(x, out);
            collect_ids_inner(y, out);
            collect_ids_inner(z, out);
        }
        Expr::QuaternionLiteral { w, x, y, z, .. } => {
            collect_ids_inner(w, out);
            collect_ids_inner(x, out);
            collect_ids_inner(y, out);
            collect_ids_inner(z, out);
        }
        Expr::IfExpr {
            condition,
            then_branch,
            else_branch,
            ..
        } => {
            collect_ids_inner(condition, out);
            collect_ids_inner(then_branch, out);
            if let Some(eb) = else_branch {
                collect_ids_inner(eb, out);
            }
        }
        Expr::LetExpr { value, body, .. } => {
            collect_ids_inner(value, out);
            collect_ids_inner(body, out);
        }
        Expr::Literal(_, _) => {}
    }
}

/// Collect all identifiers referenced in a pattern expression.
pub fn collect_pattern_identifiers(pattern: &PatternExpr) -> Vec<String> {
    let mut result = Vec::new();
    collect_pattern_ids_inner(pattern, &mut result);
    result
}

fn collect_pattern_ids_inner(pattern: &PatternExpr, out: &mut Vec<String>) {
    match pattern {
        PatternExpr::Gaze {
            entity,
            target,
            angle,
            ..
        } => {
            collect_ids_inner(entity, out);
            collect_ids_inner(target, out);
            if let Some(a) = angle {
                collect_ids_inner(a, out);
            }
        }
        PatternExpr::Reach {
            entity,
            target,
            distance,
            ..
        } => {
            collect_ids_inner(entity, out);
            collect_ids_inner(target, out);
            if let Some(d) = distance {
                collect_ids_inner(d, out);
            }
        }
        PatternExpr::Grab { target, .. } | PatternExpr::Release { target, .. } => {
            collect_ids_inner(target, out);
        }
        PatternExpr::Proximity {
            entity_a,
            entity_b,
            distance,
            ..
        } => {
            collect_ids_inner(entity_a, out);
            collect_ids_inner(entity_b, out);
            collect_ids_inner(distance, out);
        }
        PatternExpr::Inside {
            entity, region, ..
        } => {
            collect_ids_inner(entity, out);
            collect_ids_inner(region, out);
        }
        PatternExpr::Touch {
            entity_a,
            entity_b,
            ..
        } => {
            collect_ids_inner(entity_a, out);
            collect_ids_inner(entity_b, out);
        }
        PatternExpr::Conjunction { patterns, .. }
        | PatternExpr::Disjunction { patterns, .. }
        | PatternExpr::Sequence { patterns, .. } => {
            for p in patterns {
                collect_pattern_ids_inner(p, out);
            }
        }
        PatternExpr::Negation { pattern, .. } => {
            collect_pattern_ids_inner(pattern, out);
        }
        PatternExpr::Timeout { duration, .. } => {
            collect_ids_inner(duration, out);
        }
        PatternExpr::TimedPattern { pattern, .. } => {
            collect_pattern_ids_inner(pattern, out);
        }
        PatternExpr::Custom { args, .. } => {
            for a in args {
                collect_ids_inner(a, out);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_span() -> Span {
        Span {
            start: 0,
            end: 0,
            file: None,
        }
    }

    #[test]
    fn test_program_creation() {
        let program = Program::new(vec![], dummy_span());
        assert!(program.declarations.is_empty());
    }

    #[test]
    fn test_declaration_name() {
        let decl = Declaration::Region(RegionDecl {
            name: "my_region".into(),
            geometry: GeometryExpr::Reference {
                name: "base".into(),
                span: dummy_span(),
            },
            parent: None,
            constraints: vec![],
            span: dummy_span(),
        });
        assert_eq!(decl.name(), "my_region");
    }

    #[test]
    fn test_binop_precedence() {
        assert!(BinOp::Mul.binding_power().0 > BinOp::Add.binding_power().0);
        assert!(BinOp::Add.binding_power().0 > BinOp::Eq.binding_power().0);
        assert!(BinOp::Eq.binding_power().0 > BinOp::And.binding_power().0);
        assert!(BinOp::And.binding_power().0 > BinOp::Or.binding_power().0);
    }

    #[test]
    fn test_binop_display() {
        assert_eq!(format!("{}", BinOp::Add), "+");
        assert_eq!(format!("{}", BinOp::And), "and");
        assert_eq!(format!("{}", BinOp::Le), "<=");
    }

    #[test]
    fn test_unop_display() {
        assert_eq!(format!("{}", UnOp::Neg), "-");
        assert_eq!(format!("{}", UnOp::Not), "not");
    }

    #[test]
    fn test_collect_identifiers() {
        let expr = Expr::BinaryOp {
            op: BinOp::Add,
            left: Box::new(Expr::Identifier("x".into(), dummy_span())),
            right: Box::new(Expr::Identifier("y".into(), dummy_span())),
            span: dummy_span(),
        };
        let ids = collect_identifiers(&expr);
        assert_eq!(ids, vec!["x", "y"]);
    }

    #[test]
    fn test_collect_identifiers_nested() {
        let expr = Expr::FunctionCall {
            function: Box::new(Expr::Identifier("f".into(), dummy_span())),
            args: vec![
                Expr::Identifier("a".into(), dummy_span()),
                Expr::Literal(Literal::Int(1), dummy_span()),
            ],
            span: dummy_span(),
        };
        let ids = collect_identifiers(&expr);
        assert_eq!(ids, vec!["f", "a"]);
    }

    #[test]
    fn test_collect_pattern_identifiers() {
        let pattern = PatternExpr::Proximity {
            entity_a: Box::new(Expr::Identifier("user".into(), dummy_span())),
            entity_b: Box::new(Expr::Identifier("obj".into(), dummy_span())),
            distance: Box::new(Expr::Literal(Literal::Float(1.5), dummy_span())),
            span: dummy_span(),
        };
        let ids = collect_pattern_identifiers(&pattern);
        assert_eq!(ids, vec!["user", "obj"]);
    }

    #[test]
    fn test_geometry_expr_span() {
        let geom = GeometryExpr::Sphere {
            center: Box::new(Expr::Literal(Literal::Int(0), dummy_span())),
            radius: Box::new(Expr::Literal(Literal::Float(1.0), dummy_span())),
            span: Span {
                start: 10,
                end: 20,
                file: None,
            },
        };
        assert_eq!(geom.span().start, 10);
    }

    #[test]
    fn test_pattern_conjunction() {
        let p1 = PatternExpr::Grab {
            target: Box::new(Expr::Identifier("obj".into(), dummy_span())),
            span: dummy_span(),
        };
        let p2 = PatternExpr::Inside {
            entity: Box::new(Expr::Identifier("user".into(), dummy_span())),
            region: Box::new(Expr::Identifier("zone".into(), dummy_span())),
            span: dummy_span(),
        };
        let conj = PatternExpr::Conjunction {
            patterns: vec![p1, p2],
            span: dummy_span(),
        };
        if let PatternExpr::Conjunction { patterns, .. } = &conj {
            assert_eq!(patterns.len(), 2);
        } else {
            panic!("expected conjunction");
        }
    }

    #[test]
    fn test_choreography_sequential() {
        let choreo = ChoreographyExpr::Sequential {
            steps: vec![
                ChoreographyExpr::Action {
                    actions: vec![ActionExpr::Activate {
                        target: Box::new(Expr::Identifier("obj".into(), dummy_span())),
                        span: dummy_span(),
                    }],
                    span: dummy_span(),
                },
                ChoreographyExpr::Action {
                    actions: vec![ActionExpr::Deactivate {
                        target: Box::new(Expr::Identifier("obj".into(), dummy_span())),
                        span: dummy_span(),
                    }],
                    span: dummy_span(),
                },
            ],
            span: dummy_span(),
        };
        if let ChoreographyExpr::Sequential { steps, .. } = &choreo {
            assert_eq!(steps.len(), 2);
        } else {
            panic!("expected sequential");
        }
    }

    #[test]
    fn test_interaction_body_simple() {
        let body = InteractionBody::Simple {
            guard: PatternExpr::Grab {
                target: Box::new(Expr::Identifier("x".into(), dummy_span())),
                span: dummy_span(),
            },
            actions: vec![ActionExpr::Activate {
                target: Box::new(Expr::Identifier("x".into(), dummy_span())),
                span: dummy_span(),
            }],
            span: dummy_span(),
        };
        assert!(matches!(body, InteractionBody::Simple { .. }));
    }

    #[test]
    fn test_entity_type_expr() {
        assert_eq!(EntityTypeExpr::User, EntityTypeExpr::User);
        assert_ne!(EntityTypeExpr::User, EntityTypeExpr::Head);
        assert_eq!(
            EntityTypeExpr::Hand(HandSideExpr::Left),
            EntityTypeExpr::Hand(HandSideExpr::Left)
        );
    }

    #[test]
    fn test_expr_span() {
        let e = Expr::Literal(
            Literal::Int(42),
            Span {
                start: 5,
                end: 7,
                file: None,
            },
        );
        assert_eq!(e.span().start, 5);
        assert_eq!(e.span().end, 7);
    }

    #[test]
    fn test_action_expr_span() {
        let a = ActionExpr::Emit {
            event_name: "click".into(),
            args: vec![],
            span: Span {
                start: 1,
                end: 10,
                file: None,
            },
        };
        assert_eq!(a.span().start, 1);
    }

    #[test]
    fn test_import_decl() {
        let decl = Declaration::Import(ImportDecl {
            path: "std.spatial".into(),
            items: Some(vec!["Box".into(), "Sphere".into()]),
            span: dummy_span(),
        });
        assert_eq!(decl.name(), "std.spatial");
    }

    #[test]
    fn test_let_binding_decl() {
        let decl = Declaration::LetBinding(LetBindingDecl {
            name: "threshold".into(),
            type_annotation: Some(TypeAnnotation::Distance),
            value: Expr::Literal(Literal::Float(1.5), dummy_span()),
            span: dummy_span(),
        });
        assert_eq!(decl.name(), "threshold");
    }

    #[test]
    fn test_spatial_constraint_expr() {
        let c = SpatialConstraintExpr::NonOverlapping(
            vec!["r1".into(), "r2".into()],
            dummy_span(),
        );
        assert!(matches!(c, SpatialConstraintExpr::NonOverlapping(..)));
    }
}

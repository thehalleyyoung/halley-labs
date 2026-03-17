//! Abstract Syntax Tree for the SoniType DSL.
//!
//! Represents the complete syntactic structure of a SoniType program, including
//! stream declarations, data-to-sound mappings, composition expressions,
//! perceptual type annotations, and a full expression language with
//! lambda calculus, let-bindings, and pipe operators.

use crate::token::Span;
use serde::{Deserialize, Serialize};
use std::fmt;

// ─── Program ─────────────────────────────────────────────────────────────────

/// A complete SoniType program.
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

// ─── Declarations ────────────────────────────────────────────────────────────

/// Top-level declarations in a SoniType program.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Declaration {
    StreamDecl(StreamDecl),
    MappingDecl(MappingDecl),
    ComposeDecl(ComposeDecl),
    DataDecl(DataDecl),
    LetBinding(LetBinding),
    SpecDecl(SpecDecl),
    ImportDecl(ImportDecl),
}

impl Declaration {
    pub fn span(&self) -> Span {
        match self {
            Declaration::StreamDecl(d) => d.span,
            Declaration::MappingDecl(d) => d.span,
            Declaration::ComposeDecl(d) => d.span,
            Declaration::DataDecl(d) => d.span,
            Declaration::LetBinding(d) => d.span,
            Declaration::SpecDecl(d) => d.span,
            Declaration::ImportDecl(d) => d.span,
        }
    }
}

/// `stream name = stream { freq: ..., timbre: ..., pan: ..., ... }`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamDecl {
    pub name: Identifier,
    pub expr: StreamExpr,
    pub type_annotation: Option<TypeAnnotation>,
    pub exported: bool,
    pub span: Span,
}

/// `mapping name = data.field -> param(lo..hi)`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MappingDecl {
    pub name: Identifier,
    pub expr: MappingExpr,
    pub type_annotation: Option<TypeAnnotation>,
    pub exported: bool,
    pub span: Span,
}

/// `compose name = compose { s1 || s2 } where { ... } with { ... }`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComposeDecl {
    pub name: Identifier,
    pub expr: ComposeExpr,
    pub where_clause: Option<WhereClause>,
    pub with_clause: Option<WithClause>,
    pub exported: bool,
    pub span: Span,
}

/// `data name = { schema fields }`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DataDecl {
    pub name: Identifier,
    pub fields: Vec<DataField>,
    pub source: Option<String>,
    pub span: Span,
}

/// `let pattern = expr`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LetBinding {
    pub pattern: Pattern,
    pub type_annotation: Option<TypeAnnotation>,
    pub value: Box<Expr>,
    pub exported: bool,
    pub span: Span,
}

/// `spec name = { ... }`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpecDecl {
    pub name: Identifier,
    pub body: Vec<Declaration>,
    pub span: Span,
}

/// `import "path"` or `import name from "path"`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportDecl {
    pub path: String,
    pub names: Vec<Identifier>,
    pub span: Span,
}

// ─── Expressions ─────────────────────────────────────────────────────────────

/// An expression in the SoniType DSL.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    /// Integer, float, string, or boolean literal.
    Literal(Literal),
    /// Variable reference.
    Identifier(Identifier),
    /// Binary operation: `lhs op rhs`.
    BinaryOp(BinaryOp),
    /// Unary operation: `op expr`.
    UnaryOp(UnaryOp),
    /// Function call: `f(args)`.
    FunctionCall(FunctionCall),
    /// `let pattern = value in body`.
    LetIn(LetIn),
    /// `if cond then t else e`.
    IfThenElse(IfThenElse),
    /// `\param -> body` or `\(p1, p2) -> body`.
    Lambda(Lambda),
    /// Stream literal expression.
    StreamLiteral(StreamExpr),
    /// Mapping literal expression.
    MappingLiteral(MappingExpr),
    /// Composition of streams.
    Compose(ComposeExpr),
    /// `expr with { constraints }`.
    WithClause(WithExpr),
    /// `expr where { bindings }`.
    WhereClause(WhereExpr),
    /// `lhs |> rhs` pipe operator.
    PipeOperator(PipeExpr),
    /// Field access: `expr.field`.
    FieldAccess(FieldAccess),
    /// Parenthesized expression.
    Grouped(Box<Expr>, Span),
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::Literal(l) => l.span,
            Expr::Identifier(i) => i.span,
            Expr::BinaryOp(b) => b.span,
            Expr::UnaryOp(u) => u.span,
            Expr::FunctionCall(f) => f.span,
            Expr::LetIn(l) => l.span,
            Expr::IfThenElse(i) => i.span,
            Expr::Lambda(l) => l.span,
            Expr::StreamLiteral(s) => s.span,
            Expr::MappingLiteral(m) => m.span,
            Expr::Compose(c) => c.span,
            Expr::WithClause(w) => w.span,
            Expr::WhereClause(w) => w.span,
            Expr::PipeOperator(p) => p.span,
            Expr::FieldAccess(f) => f.span,
            Expr::Grouped(_, span) => *span,
        }
    }
}

// ── Expression sub-types ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Literal {
    pub value: LiteralValue,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LiteralValue {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Identifier {
    pub name: String,
    pub span: Span,
}

impl Identifier {
    pub fn new(name: impl Into<String>, span: Span) -> Self {
        Self { name: name.into(), span }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinaryOp {
    pub op: BinOp,
    pub lhs: Box<Expr>,
    pub rhs: Box<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    And,
    Or,
    Eq,
    Neq,
    Lt,
    Gt,
    Lte,
    Gte,
    Range, // ..
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinOp::Add => write!(f, "+"),
            BinOp::Sub => write!(f, "-"),
            BinOp::Mul => write!(f, "*"),
            BinOp::Div => write!(f, "/"),
            BinOp::And => write!(f, "&&"),
            BinOp::Or => write!(f, "||"),
            BinOp::Eq => write!(f, "=="),
            BinOp::Neq => write!(f, "!="),
            BinOp::Lt => write!(f, "<"),
            BinOp::Gt => write!(f, ">"),
            BinOp::Lte => write!(f, "<="),
            BinOp::Gte => write!(f, ">="),
            BinOp::Range => write!(f, ".."),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnaryOp {
    pub op: UnOp,
    pub operand: Box<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnOp {
    Neg,
    Not,
}

impl fmt::Display for UnOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnOp::Neg => write!(f, "-"),
            UnOp::Not => write!(f, "!"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionCall {
    pub callee: Box<Expr>,
    pub args: Vec<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LetIn {
    pub pattern: Pattern,
    pub type_annotation: Option<TypeAnnotation>,
    pub value: Box<Expr>,
    pub body: Box<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IfThenElse {
    pub condition: Box<Expr>,
    pub then_branch: Box<Expr>,
    pub else_branch: Box<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Lambda {
    pub params: Vec<LambdaParam>,
    pub body: Box<Expr>,
    pub return_type: Option<TypeAnnotation>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LambdaParam {
    pub name: Identifier,
    pub type_annotation: Option<TypeAnnotation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PipeExpr {
    pub lhs: Box<Expr>,
    pub rhs: Box<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldAccess {
    pub object: Box<Expr>,
    pub field: Identifier,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WithExpr {
    pub expr: Box<Expr>,
    pub constraints: Vec<Constraint>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WhereExpr {
    pub expr: Box<Expr>,
    pub bindings: Vec<WhereBinding>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WhereBinding {
    pub name: Identifier,
    pub value: Expr,
    pub span: Span,
}

// ─── Stream Expression ───────────────────────────────────────────────────────

/// A stream literal: `stream { freq: 440.0, timbre: "sine", ... }`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamExpr {
    pub params: Vec<StreamParam>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamParam {
    pub name: Identifier,
    pub value: Expr,
    pub span: Span,
}

// ─── Mapping Expression ──────────────────────────────────────────────────────

/// A mapping expression: maps a data field to an audio parameter with a range.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MappingExpr {
    pub source: DataRef,
    pub target: MappingTarget,
    pub span: Span,
}

/// Reference to a data field: `data.field_name`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DataRef {
    pub source: Identifier,
    pub field: Identifier,
    pub span: Span,
}

/// Target of a mapping: `param(lo..hi)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MappingTarget {
    pub param: AudioParamKind,
    pub range: Option<(Box<Expr>, Box<Expr>)>,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioParamKind {
    Pitch,
    Timbre,
    Pan,
    Amplitude,
    Duration,
}

impl fmt::Display for AudioParamKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AudioParamKind::Pitch => write!(f, "pitch"),
            AudioParamKind::Timbre => write!(f, "timbre"),
            AudioParamKind::Pan => write!(f, "pan"),
            AudioParamKind::Amplitude => write!(f, "amplitude"),
            AudioParamKind::Duration => write!(f, "duration"),
        }
    }
}

// ─── Compose Expression ──────────────────────────────────────────────────────

/// Parallel composition of streams: `compose { s1 || s2 || s3 }`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComposeExpr {
    pub streams: Vec<Expr>,
    pub span: Span,
}

// ─── Clauses ─────────────────────────────────────────────────────────────────

/// `where { binding: value, ... }` clause.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WhereClause {
    pub bindings: Vec<WhereBinding>,
    pub span: Span,
}

/// `with { constraint: value, ... }` clause.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WithClause {
    pub constraints: Vec<Constraint>,
    pub span: Span,
}

/// A constraint in a `with` clause.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Constraint {
    pub name: Identifier,
    pub value: Expr,
    pub span: Span,
}

// ─── Data ────────────────────────────────────────────────────────────────────

/// A field in a data declaration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DataField {
    pub name: Identifier,
    pub ty: Type,
    pub span: Span,
}

// ─── Patterns ────────────────────────────────────────────────────────────────

/// Pattern for let-bindings and function parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Pattern {
    /// Simple variable binding.
    Variable(Identifier),
    /// Tuple destructuring: `(a, b, c)`.
    Tuple(Vec<Pattern>, Span),
    /// Wildcard: `_`.
    Wildcard(Span),
}

impl Pattern {
    pub fn span(&self) -> Span {
        match self {
            Pattern::Variable(id) => id.span,
            Pattern::Tuple(_, span) => *span,
            Pattern::Wildcard(span) => *span,
        }
    }
}

// ─── Types ───────────────────────────────────────────────────────────────────

/// A type in the SoniType type system.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Type {
    /// Named base type: Stream, Mapping, etc.
    Named(String, Span),
    /// Parameterized type: `Stream<Pitch>`.
    Parameterized(String, Vec<Type>, Span),
    /// Function type: `A -> B`.
    Function(Box<Type>, Box<Type>, Span),
    /// Tuple type: `(A, B)`.
    Tuple(Vec<Type>, Span),
    /// A type with a perceptual annotation: `τ⟨φ⟩`.
    Qualified(Box<Type>, PerceptualAnnotation, Span),
    /// Type variable (for inference): `'a`.
    Variable(String, Span),
}

impl Type {
    pub fn span(&self) -> Span {
        match self {
            Type::Named(_, s)
            | Type::Parameterized(_, _, s)
            | Type::Function(_, _, s)
            | Type::Tuple(_, s)
            | Type::Qualified(_, _, s)
            | Type::Variable(_, s) => *s,
        }
    }

    /// Returns `true` if this is a base type (Stream, Mapping, Float, etc.).
    pub fn is_base(&self) -> bool {
        matches!(self, Type::Named(..))
    }

    /// Returns `true` if this is a function type.
    pub fn is_function(&self) -> bool {
        matches!(self, Type::Function(..))
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Named(name, _) => write!(f, "{name}"),
            Type::Parameterized(name, args, _) => {
                write!(f, "{name}<")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{arg}")?;
                }
                write!(f, ">")
            }
            Type::Function(from, to, _) => write!(f, "{from} -> {to}"),
            Type::Tuple(elems, _) => {
                write!(f, "(")?;
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{e}")?;
                }
                write!(f, ")")
            }
            Type::Qualified(base, ann, _) => write!(f, "{base}⟨{ann}⟩"),
            Type::Variable(name, _) => write!(f, "'{name}"),
        }
    }
}

// ─── Type annotation (syntax-level) ─────────────────────────────────────────

/// Type annotation as written in source: `name : Type⟨qualifier⟩`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypeAnnotation {
    pub ty: Type,
    pub qualifier: Option<PerceptualAnnotation>,
    pub span: Span,
}

/// Perceptual qualifier annotation at the syntax level.
///
/// Represents `⟨band: {1,3,5}, load: 1.0, masking: 6.0⟩` or similar.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerceptualAnnotation {
    pub predicates: Vec<PerceptualPredicate>,
    pub span: Span,
}

impl fmt::Display for PerceptualAnnotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "⟨")?;
        for (i, p) in self.predicates.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{p}")?;
        }
        write!(f, "⟩")
    }
}

/// Individual predicates in a perceptual qualifier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PerceptualPredicate {
    /// `band: {1, 3, 5}` – Bark band occupancy.
    BandOccupancy(Vec<u8>),
    /// `load: 1.0` – cognitive load contribution.
    CognitiveLoad(f64),
    /// `masking: 6.0` – masking margin in dB.
    MaskingMargin(f64),
    /// `segregation: true` – requires stream segregation.
    Segregation(bool),
    /// `jnd(param): value` – JND constraint for a parameter.
    Jnd(AudioParamKind, f64),
}

impl fmt::Display for PerceptualPredicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PerceptualPredicate::BandOccupancy(bands) => {
                write!(f, "band: {{")?;
                for (i, b) in bands.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{b}")?;
                }
                write!(f, "}}")
            }
            PerceptualPredicate::CognitiveLoad(v) => write!(f, "load: {v}"),
            PerceptualPredicate::MaskingMargin(v) => write!(f, "masking: {v}"),
            PerceptualPredicate::Segregation(b) => write!(f, "segregation: {b}"),
            PerceptualPredicate::Jnd(param, v) => write!(f, "jnd({param}): {v}"),
        }
    }
}

// ─── Visitor Trait ───────────────────────────────────────────────────────────

/// Visitor trait for AST traversal.
pub trait Visitor {
    type Result;

    fn visit_program(&mut self, program: &Program) -> Self::Result;
    fn visit_declaration(&mut self, decl: &Declaration) -> Self::Result;
    fn visit_expr(&mut self, expr: &Expr) -> Self::Result;
    fn visit_type(&mut self, ty: &Type) -> Self::Result;
    fn visit_pattern(&mut self, pat: &Pattern) -> Self::Result;
}

/// Mutable visitor trait for AST transformation.
pub trait MutVisitor {
    fn visit_program_mut(&mut self, program: &mut Program);
    fn visit_declaration_mut(&mut self, decl: &mut Declaration);
    fn visit_expr_mut(&mut self, expr: &mut Expr);
    fn visit_type_mut(&mut self, ty: &mut Type);
    fn visit_pattern_mut(&mut self, pat: &mut Pattern);
}

/// Walk all declarations in a program.
pub fn walk_program<V: Visitor>(visitor: &mut V, program: &Program) -> Vec<V::Result> {
    program.declarations.iter().map(|d| visitor.visit_declaration(d)).collect()
}

/// Walk all sub-expressions of an expression.
pub fn walk_expr<V: Visitor>(visitor: &mut V, expr: &Expr) -> Vec<V::Result> {
    match expr {
        Expr::Literal(_) | Expr::Identifier(_) => vec![],
        Expr::BinaryOp(b) => {
            vec![visitor.visit_expr(&b.lhs), visitor.visit_expr(&b.rhs)]
        }
        Expr::UnaryOp(u) => vec![visitor.visit_expr(&u.operand)],
        Expr::FunctionCall(f) => {
            let mut results = vec![visitor.visit_expr(&f.callee)];
            for arg in &f.args {
                results.push(visitor.visit_expr(arg));
            }
            results
        }
        Expr::LetIn(l) => {
            vec![visitor.visit_expr(&l.value), visitor.visit_expr(&l.body)]
        }
        Expr::IfThenElse(i) => {
            vec![
                visitor.visit_expr(&i.condition),
                visitor.visit_expr(&i.then_branch),
                visitor.visit_expr(&i.else_branch),
            ]
        }
        Expr::Lambda(l) => vec![visitor.visit_expr(&l.body)],
        Expr::PipeOperator(p) => {
            vec![visitor.visit_expr(&p.lhs), visitor.visit_expr(&p.rhs)]
        }
        Expr::FieldAccess(f) => vec![visitor.visit_expr(&f.object)],
        Expr::Grouped(inner, _) => vec![visitor.visit_expr(inner)],
        Expr::StreamLiteral(s) => {
            s.params.iter().map(|p| visitor.visit_expr(&p.value)).collect()
        }
        Expr::MappingLiteral(_) => vec![],
        Expr::Compose(c) => {
            c.streams.iter().map(|s| visitor.visit_expr(s)).collect()
        }
        Expr::WithClause(w) => {
            let mut r = vec![visitor.visit_expr(&w.expr)];
            for c in &w.constraints {
                r.push(visitor.visit_expr(&c.value));
            }
            r
        }
        Expr::WhereClause(w) => {
            let mut r = vec![visitor.visit_expr(&w.expr)];
            for b in &w.bindings {
                r.push(visitor.visit_expr(&b.value));
            }
            r
        }
    }
}

// ─── Display implementations ─────────────────────────────────────────────────

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for decl in &self.declarations {
            writeln!(f, "{decl}")?;
        }
        Ok(())
    }
}

impl fmt::Display for Declaration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Declaration::StreamDecl(d) => {
                if d.exported { write!(f, "export ")?; }
                write!(f, "stream {} = {}", d.name.name, d.expr)
            }
            Declaration::MappingDecl(d) => {
                if d.exported { write!(f, "export ")?; }
                write!(f, "mapping {} = {}", d.name.name, d.expr)
            }
            Declaration::ComposeDecl(d) => {
                if d.exported { write!(f, "export ")?; }
                write!(f, "compose {} = {}", d.name.name, d.expr)
            }
            Declaration::DataDecl(d) => {
                write!(f, "data {} = {{ ... }}", d.name.name)
            }
            Declaration::LetBinding(d) => {
                if d.exported { write!(f, "export ")?; }
                write!(f, "let {} = {}", d.pattern, d.value)
            }
            Declaration::SpecDecl(d) => write!(f, "spec {} {{ ... }}", d.name.name),
            Declaration::ImportDecl(d) => write!(f, "import \"{}\"", d.path),
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Literal(l) => write!(f, "{}", l.value),
            Expr::Identifier(i) => write!(f, "{}", i.name),
            Expr::BinaryOp(b) => write!(f, "({} {} {})", b.lhs, b.op, b.rhs),
            Expr::UnaryOp(u) => write!(f, "({}{})", u.op, u.operand),
            Expr::FunctionCall(fc) => {
                write!(f, "{}(", fc.callee)?;
                for (i, a) in fc.args.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{a}")?;
                }
                write!(f, ")")
            }
            Expr::LetIn(l) => write!(f, "let {} = {} in {}", l.pattern, l.value, l.body),
            Expr::IfThenElse(i) => {
                write!(f, "if {} then {} else {}", i.condition, i.then_branch, i.else_branch)
            }
            Expr::Lambda(l) => {
                write!(f, "\\(")?;
                for (i, p) in l.params.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", p.name.name)?;
                }
                write!(f, ") -> {}", l.body)
            }
            Expr::StreamLiteral(s) => write!(f, "{s}"),
            Expr::MappingLiteral(m) => write!(f, "{m}"),
            Expr::Compose(c) => write!(f, "{c}"),
            Expr::WithClause(w) => write!(f, "{} with {{ ... }}", w.expr),
            Expr::WhereClause(w) => write!(f, "{} where {{ ... }}", w.expr),
            Expr::PipeOperator(p) => write!(f, "{} |> {}", p.lhs, p.rhs),
            Expr::FieldAccess(fa) => write!(f, "{}.{}", fa.object, fa.field.name),
            Expr::Grouped(inner, _) => write!(f, "({inner})"),
        }
    }
}

impl fmt::Display for LiteralValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LiteralValue::Int(v) => write!(f, "{v}"),
            LiteralValue::Float(v) => write!(f, "{v}"),
            LiteralValue::String(s) => write!(f, "\"{s}\""),
            LiteralValue::Bool(b) => write!(f, "{b}"),
        }
    }
}

impl fmt::Display for Pattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Pattern::Variable(id) => write!(f, "{}", id.name),
            Pattern::Tuple(pats, _) => {
                write!(f, "(")?;
                for (i, p) in pats.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{p}")?;
                }
                write!(f, ")")
            }
            Pattern::Wildcard(_) => write!(f, "_"),
        }
    }
}

impl fmt::Display for StreamExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "stream {{ ")?;
        for (i, p) in self.params.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{}: {}", p.name.name, p.value)?;
        }
        write!(f, " }}")
    }
}

impl fmt::Display for MappingExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}.{} -> {}",
            self.source.source.name, self.source.field.name, self.target.param
        )?;
        if let Some((lo, hi)) = &self.target.range {
            write!(f, "({lo}..{hi})")?;
        }
        Ok(())
    }
}

impl fmt::Display for ComposeExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "compose {{ ")?;
        for (i, s) in self.streams.iter().enumerate() {
            if i > 0 { write!(f, " || ")?; }
            write!(f, "{s}")?;
        }
        write!(f, " }}")
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_span() -> Span {
        Span::dummy()
    }

    fn ident(name: &str) -> Identifier {
        Identifier::new(name, dummy_span())
    }

    fn int_lit(v: i64) -> Expr {
        Expr::Literal(Literal { value: LiteralValue::Int(v), span: dummy_span() })
    }

    fn float_lit(v: f64) -> Expr {
        Expr::Literal(Literal { value: LiteralValue::Float(v), span: dummy_span() })
    }

    fn ident_expr(name: &str) -> Expr {
        Expr::Identifier(ident(name))
    }

    #[test]
    fn test_program_display() {
        let prog = Program {
            declarations: vec![Declaration::LetBinding(LetBinding {
                pattern: Pattern::Variable(ident("x")),
                type_annotation: None,
                value: Box::new(int_lit(42)),
                exported: false,
                span: dummy_span(),
            })],
            span: dummy_span(),
        };
        let s = format!("{prog}");
        assert!(s.contains("let x = 42"));
    }

    #[test]
    fn test_binary_op_display() {
        let expr = Expr::BinaryOp(BinaryOp {
            op: BinOp::Add,
            lhs: Box::new(int_lit(1)),
            rhs: Box::new(int_lit(2)),
            span: dummy_span(),
        });
        assert_eq!(format!("{expr}"), "(1 + 2)");
    }

    #[test]
    fn test_unary_op_display() {
        let expr = Expr::UnaryOp(UnaryOp {
            op: UnOp::Neg,
            operand: Box::new(int_lit(5)),
            span: dummy_span(),
        });
        assert_eq!(format!("{expr}"), "(-5)");
    }

    #[test]
    fn test_function_call_display() {
        let expr = Expr::FunctionCall(FunctionCall {
            callee: Box::new(ident_expr("foo")),
            args: vec![int_lit(1), int_lit(2)],
            span: dummy_span(),
        });
        assert_eq!(format!("{expr}"), "foo(1, 2)");
    }

    #[test]
    fn test_let_in_display() {
        let expr = Expr::LetIn(LetIn {
            pattern: Pattern::Variable(ident("x")),
            type_annotation: None,
            value: Box::new(int_lit(1)),
            body: Box::new(ident_expr("x")),
            span: dummy_span(),
        });
        assert_eq!(format!("{expr}"), "let x = 1 in x");
    }

    #[test]
    fn test_if_then_else_display() {
        let expr = Expr::IfThenElse(IfThenElse {
            condition: Box::new(Expr::Literal(Literal {
                value: LiteralValue::Bool(true),
                span: dummy_span(),
            })),
            then_branch: Box::new(int_lit(1)),
            else_branch: Box::new(int_lit(0)),
            span: dummy_span(),
        });
        assert_eq!(format!("{expr}"), "if true then 1 else 0");
    }

    #[test]
    fn test_lambda_display() {
        let expr = Expr::Lambda(Lambda {
            params: vec![LambdaParam {
                name: ident("x"),
                type_annotation: None,
            }],
            body: Box::new(ident_expr("x")),
            return_type: None,
            span: dummy_span(),
        });
        assert_eq!(format!("{expr}"), "\\(x) -> x");
    }

    #[test]
    fn test_stream_literal_display() {
        let expr = StreamExpr {
            params: vec![
                StreamParam {
                    name: ident("freq"),
                    value: float_lit(440.0),
                    span: dummy_span(),
                },
                StreamParam {
                    name: ident("pan"),
                    value: float_lit(0.0),
                    span: dummy_span(),
                },
            ],
            span: dummy_span(),
        };
        let s = format!("{expr}");
        assert!(s.contains("freq: 440"));
        assert!(s.contains("pan: 0"));
    }

    #[test]
    fn test_compose_display() {
        let expr = ComposeExpr {
            streams: vec![ident_expr("s1"), ident_expr("s2")],
            span: dummy_span(),
        };
        assert_eq!(format!("{expr}"), "compose { s1 || s2 }");
    }

    #[test]
    fn test_type_display() {
        let ty = Type::Function(
            Box::new(Type::Named("Stream".to_string(), dummy_span())),
            Box::new(Type::Named("Mapping".to_string(), dummy_span())),
            dummy_span(),
        );
        assert_eq!(format!("{ty}"), "Stream -> Mapping");
    }

    #[test]
    fn test_parameterized_type_display() {
        let ty = Type::Parameterized(
            "Stream".to_string(),
            vec![Type::Named("Pitch".to_string(), dummy_span())],
            dummy_span(),
        );
        assert_eq!(format!("{ty}"), "Stream<Pitch>");
    }

    #[test]
    fn test_perceptual_annotation_display() {
        let ann = PerceptualAnnotation {
            predicates: vec![
                PerceptualPredicate::BandOccupancy(vec![1, 3, 5]),
                PerceptualPredicate::CognitiveLoad(1.0),
            ],
            span: dummy_span(),
        };
        assert_eq!(format!("{ann}"), "⟨band: {1, 3, 5}, load: 1⟩");
    }

    #[test]
    fn test_pattern_display() {
        let p = Pattern::Tuple(
            vec![
                Pattern::Variable(ident("a")),
                Pattern::Wildcard(dummy_span()),
            ],
            dummy_span(),
        );
        assert_eq!(format!("{p}"), "(a, _)");
    }

    #[test]
    fn test_pipe_operator_display() {
        let expr = Expr::PipeOperator(PipeExpr {
            lhs: Box::new(ident_expr("data")),
            rhs: Box::new(ident_expr("filter")),
            span: dummy_span(),
        });
        assert_eq!(format!("{expr}"), "data |> filter");
    }

    #[test]
    fn test_field_access_display() {
        let expr = Expr::FieldAccess(FieldAccess {
            object: Box::new(ident_expr("data")),
            field: ident("temperature"),
            span: dummy_span(),
        });
        assert_eq!(format!("{expr}"), "data.temperature");
    }

    #[test]
    fn test_visitor_walk_binary() {
        struct Counter(usize);
        impl Visitor for Counter {
            type Result = ();
            fn visit_program(&mut self, _: &Program) {}
            fn visit_declaration(&mut self, _: &Declaration) {}
            fn visit_expr(&mut self, expr: &Expr) {
                self.0 += 1;
                walk_expr(self, expr);
            }
            fn visit_type(&mut self, _: &Type) {}
            fn visit_pattern(&mut self, _: &Pattern) {}
        }

        let expr = Expr::BinaryOp(BinaryOp {
            op: BinOp::Add,
            lhs: Box::new(int_lit(1)),
            rhs: Box::new(int_lit(2)),
            span: dummy_span(),
        });
        let mut counter = Counter(0);
        counter.visit_expr(&expr);
        assert_eq!(counter.0, 3); // root + lhs + rhs
    }
}

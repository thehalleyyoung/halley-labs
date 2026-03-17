use crate::source_map::Span;
use regsynth_types::{CompositionOp, FormalizabilityGrade, ObligationKind, RiskLevel};
use serde::{Deserialize, Serialize};

// ─── Program ────────────────────────────────────────────────────

/// A complete DSL program.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub declarations: Vec<Declaration>,
    pub span: Span,
}

// ─── Declarations ───────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct Declaration {
    pub kind: DeclarationKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DeclarationKind {
    Jurisdiction(JurisdictionDecl),
    Obligation(ObligationDecl),
    Permission(PermissionDecl),
    Prohibition(ProhibitionDecl),
    Framework(FrameworkDecl),
    Strategy(StrategyDecl),
    Cost(CostDecl),
    Temporal(TemporalDecl),
    Mapping(MappingDecl),
}

#[derive(Debug, Clone, PartialEq)]
pub struct JurisdictionDecl {
    pub name: String,
    pub parent: Option<String>,
    pub body: Vec<Declaration>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObligationDecl {
    pub name: String,
    pub jurisdiction: Option<String>,
    pub body: ObligationBody,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PermissionDecl {
    pub name: String,
    pub jurisdiction: Option<String>,
    pub body: ObligationBody,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProhibitionDecl {
    pub name: String,
    pub jurisdiction: Option<String>,
    pub body: ObligationBody,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FrameworkDecl {
    pub name: String,
    pub jurisdiction: Option<String>,
    pub body: Vec<Declaration>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StrategyDecl {
    pub name: String,
    pub body: Vec<StrategyItem>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StrategyItem {
    pub key: String,
    pub value: Expression,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CostDecl {
    pub name: String,
    pub amount: Expression,
    pub currency: Option<String>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TemporalDecl {
    pub name: String,
    pub start: Option<String>,
    pub end: Option<String>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MappingDecl {
    pub name: String,
    pub entries: Vec<MappingEntry>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MappingEntry {
    pub from: Expression,
    pub to: Expression,
    pub span: Span,
}

// ─── Obligation Body ────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ObligationBody {
    pub conditions: Vec<Expression>,
    pub exemptions: Vec<Expression>,
    pub temporal_start: Option<String>,
    pub temporal_end: Option<String>,
    pub risk_level: Option<RiskLevel>,
    pub domain: Option<String>,
    pub formalizability: Option<FormalizabilityGrade>,
    pub article_refs: Vec<ArticleRefNode>,
    pub compositions: Vec<CompositionNode>,
    pub extra_fields: Vec<FieldNode>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArticleRefNode {
    pub framework: String,
    pub article: String,
    pub paragraph: Option<String>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompositionNode {
    pub op: CompositionOp,
    pub operand: String,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FieldNode {
    pub key: String,
    pub value: Expression,
    pub span: Span,
}

// ─── Expressions ────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct Expression {
    pub kind: ExpressionKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExpressionKind {
    /// Binary operation: left op right
    BinaryOp {
        op: BinOp,
        left: Box<Expression>,
        right: Box<Expression>,
    },
    /// Unary operation: op expr
    UnaryOp {
        op: UnOp,
        operand: Box<Expression>,
    },
    /// Variable reference
    Variable(String),
    /// Literal value
    Literal(Literal),
    /// if cond then a else b
    IfThenElse {
        condition: Box<Expression>,
        then_branch: Box<Expression>,
        else_branch: Box<Expression>,
    },
    /// forall x : T . body  or  exists x : T . body
    Quantifier {
        kind: QuantifierKind,
        variable: String,
        domain: Option<Box<Expression>>,
        body: Box<Expression>,
    },
    /// function_name(args...)
    FunctionCall {
        function: String,
        args: Vec<Expression>,
    },
    /// article "Framework" "Art. N" (optional paragraph)
    ArticleRef {
        framework: String,
        article: String,
        paragraph: Option<String>,
    },
    /// expr.field
    FieldAccess {
        object: Box<Expression>,
        field: String,
    },
    /// Composition: left ⊗ right, left ⊕ right, etc.
    Composition {
        op: CompositionOp,
        left: Box<Expression>,
        right: Box<Expression>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    NotEq,
    Lt,
    Gt,
    LtEq,
    GtEq,
    And,
    Or,
    Implies,
}

impl BinOp {
    /// Precedence for precedence climbing (higher = tighter binding).
    pub fn precedence(&self) -> u8 {
        match self {
            Self::Implies => 1,
            Self::Or => 2,
            Self::And => 3,
            Self::Eq | Self::NotEq => 4,
            Self::Lt | Self::Gt | Self::LtEq | Self::GtEq => 5,
            Self::Add | Self::Sub => 6,
            Self::Mul | Self::Div => 7,
        }
    }

    pub fn is_right_assoc(&self) -> bool {
        matches!(self, Self::Implies)
    }
}

impl std::fmt::Display for BinOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Add => write!(f, "+"),
            Self::Sub => write!(f, "-"),
            Self::Mul => write!(f, "*"),
            Self::Div => write!(f, "/"),
            Self::Eq => write!(f, "=="),
            Self::NotEq => write!(f, "!="),
            Self::Lt => write!(f, "<"),
            Self::Gt => write!(f, ">"),
            Self::LtEq => write!(f, "<="),
            Self::GtEq => write!(f, ">="),
            Self::And => write!(f, "and"),
            Self::Or => write!(f, "or"),
            Self::Implies => write!(f, "implies"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnOp {
    Not,
    Neg,
}

impl std::fmt::Display for UnOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Not => write!(f, "not"),
            Self::Neg => write!(f, "-"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Literal {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Date(String),
}

impl std::fmt::Display for Literal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bool(b) => write!(f, "{}", b),
            Self::Int(n) => write!(f, "{}", n),
            Self::Float(n) => write!(f, "{}", n),
            Self::Str(s) => write!(f, "\"{}\"", s),
            Self::Date(d) => write!(f, "#{}", d),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantifierKind {
    Forall,
    Exists,
}

impl std::fmt::Display for QuantifierKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Forall => write!(f, "forall"),
            Self::Exists => write!(f, "exists"),
        }
    }
}

// ─── Visitor Pattern ────────────────────────────────────────────

/// Visitor trait for traversing the AST.
pub trait AstVisitor {
    type Result;

    fn visit_program(&mut self, program: &Program) -> Self::Result;
    fn visit_declaration(&mut self, decl: &Declaration) -> Self::Result;
    fn visit_expression(&mut self, expr: &Expression) -> Self::Result;
    fn visit_obligation_body(&mut self, body: &ObligationBody) -> Self::Result;
}

/// Default walking visitor that visits all children.
pub trait AstWalker {
    fn walk_program(&mut self, program: &Program) {
        for decl in &program.declarations {
            self.walk_declaration(decl);
        }
    }

    fn walk_declaration(&mut self, decl: &Declaration) {
        match &decl.kind {
            DeclarationKind::Jurisdiction(j) => {
                for child in &j.body {
                    self.walk_declaration(child);
                }
            }
            DeclarationKind::Obligation(o) => self.walk_obligation_body(&o.body),
            DeclarationKind::Permission(p) => self.walk_obligation_body(&p.body),
            DeclarationKind::Prohibition(p) => self.walk_obligation_body(&p.body),
            DeclarationKind::Framework(f) => {
                for child in &f.body {
                    self.walk_declaration(child);
                }
            }
            DeclarationKind::Strategy(s) => {
                for item in &s.body {
                    self.walk_expression(&item.value);
                }
            }
            DeclarationKind::Cost(c) => self.walk_expression(&c.amount),
            DeclarationKind::Temporal(_) => {}
            DeclarationKind::Mapping(m) => {
                for entry in &m.entries {
                    self.walk_expression(&entry.from);
                    self.walk_expression(&entry.to);
                }
            }
        }
    }

    fn walk_obligation_body(&mut self, body: &ObligationBody) {
        for cond in &body.conditions {
            self.walk_expression(cond);
        }
        for exempt in &body.exemptions {
            self.walk_expression(exempt);
        }
        for field in &body.extra_fields {
            self.walk_expression(&field.value);
        }
    }

    fn walk_expression(&mut self, expr: &Expression) {
        match &expr.kind {
            ExpressionKind::BinaryOp { left, right, .. } => {
                self.walk_expression(left);
                self.walk_expression(right);
            }
            ExpressionKind::UnaryOp { operand, .. } => {
                self.walk_expression(operand);
            }
            ExpressionKind::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => {
                self.walk_expression(condition);
                self.walk_expression(then_branch);
                self.walk_expression(else_branch);
            }
            ExpressionKind::Quantifier { domain, body, .. } => {
                if let Some(d) = domain {
                    self.walk_expression(d);
                }
                self.walk_expression(body);
            }
            ExpressionKind::FunctionCall { args, .. } => {
                for arg in args {
                    self.walk_expression(arg);
                }
            }
            ExpressionKind::FieldAccess { object, .. } => {
                self.walk_expression(object);
            }
            ExpressionKind::Composition { left, right, .. } => {
                self.walk_expression(left);
                self.walk_expression(right);
            }
            ExpressionKind::Variable(_)
            | ExpressionKind::Literal(_)
            | ExpressionKind::ArticleRef { .. } => {}
        }
    }
}

/// Collect all variable names referenced in an expression.
pub fn collect_variables(expr: &Expression) -> Vec<String> {
    let mut vars = Vec::new();
    collect_vars_inner(expr, &mut vars);
    vars
}

fn collect_vars_inner(expr: &Expression, vars: &mut Vec<String>) {
    match &expr.kind {
        ExpressionKind::Variable(name) => vars.push(name.clone()),
        ExpressionKind::BinaryOp { left, right, .. } => {
            collect_vars_inner(left, vars);
            collect_vars_inner(right, vars);
        }
        ExpressionKind::UnaryOp { operand, .. } => collect_vars_inner(operand, vars),
        ExpressionKind::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => {
            collect_vars_inner(condition, vars);
            collect_vars_inner(then_branch, vars);
            collect_vars_inner(else_branch, vars);
        }
        ExpressionKind::Quantifier { body, domain, .. } => {
            if let Some(d) = domain {
                collect_vars_inner(d, vars);
            }
            collect_vars_inner(body, vars);
        }
        ExpressionKind::FunctionCall { args, .. } => {
            for arg in args {
                collect_vars_inner(arg, vars);
            }
        }
        ExpressionKind::FieldAccess { object, .. } => collect_vars_inner(object, vars),
        ExpressionKind::Composition { left, right, .. } => {
            collect_vars_inner(left, vars);
            collect_vars_inner(right, vars);
        }
        ExpressionKind::Literal(_) | ExpressionKind::ArticleRef { .. } => {}
    }
}

/// Get the obligation kind for a declaration, if applicable.
pub fn decl_obligation_kind(decl: &Declaration) -> Option<ObligationKind> {
    match &decl.kind {
        DeclarationKind::Obligation(_) => Some(ObligationKind::Obligation),
        DeclarationKind::Permission(_) => Some(ObligationKind::Permission),
        DeclarationKind::Prohibition(_) => Some(ObligationKind::Prohibition),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_var(name: &str) -> Expression {
        Expression {
            kind: ExpressionKind::Variable(name.to_string()),
            span: Span::empty(),
        }
    }

    fn make_lit_int(n: i64) -> Expression {
        Expression {
            kind: ExpressionKind::Literal(Literal::Int(n)),
            span: Span::empty(),
        }
    }

    #[test]
    fn test_binop_precedence() {
        assert!(BinOp::Mul.precedence() > BinOp::Add.precedence());
        assert!(BinOp::And.precedence() > BinOp::Or.precedence());
        assert!(BinOp::Or.precedence() > BinOp::Implies.precedence());
    }

    #[test]
    fn test_collect_variables() {
        let expr = Expression {
            kind: ExpressionKind::BinaryOp {
                op: BinOp::Add,
                left: Box::new(make_var("x")),
                right: Box::new(make_var("y")),
            },
            span: Span::empty(),
        };
        let vars = collect_variables(&expr);
        assert_eq!(vars, vec!["x", "y"]);
    }

    #[test]
    fn test_collect_variables_nested() {
        let expr = Expression {
            kind: ExpressionKind::IfThenElse {
                condition: Box::new(make_var("a")),
                then_branch: Box::new(make_var("b")),
                else_branch: Box::new(make_lit_int(0)),
            },
            span: Span::empty(),
        };
        let vars = collect_variables(&expr);
        assert_eq!(vars, vec!["a", "b"]);
    }

    #[test]
    fn test_literal_display() {
        assert_eq!(Literal::Bool(true).to_string(), "true");
        assert_eq!(Literal::Int(42).to_string(), "42");
        assert_eq!(Literal::Str("hi".into()).to_string(), "\"hi\"");
        assert_eq!(Literal::Date("2024-01-01".into()).to_string(), "#2024-01-01");
    }

    #[test]
    fn test_decl_obligation_kind() {
        let obl = Declaration {
            kind: DeclarationKind::Obligation(ObligationDecl {
                name: "test".into(),
                jurisdiction: None,
                body: ObligationBody::default(),
                span: Span::empty(),
            }),
            span: Span::empty(),
        };
        assert_eq!(decl_obligation_kind(&obl), Some(ObligationKind::Obligation));

        let jur = Declaration {
            kind: DeclarationKind::Jurisdiction(JurisdictionDecl {
                name: "EU".into(),
                parent: None,
                body: vec![],
                span: Span::empty(),
            }),
            span: Span::empty(),
        };
        assert_eq!(decl_obligation_kind(&jur), None);
    }

    #[test]
    fn test_quantifier_display() {
        assert_eq!(QuantifierKind::Forall.to_string(), "forall");
        assert_eq!(QuantifierKind::Exists.to_string(), "exists");
    }

    #[test]
    fn test_walker_visits_all() {
        struct Counter(usize);
        impl AstWalker for Counter {
            fn walk_expression(&mut self, _expr: &Expression) {
                self.0 += 1;
            }
        }
        let body = ObligationBody {
            conditions: vec![make_var("a"), make_var("b")],
            exemptions: vec![make_var("c")],
            ..Default::default()
        };
        let mut counter = Counter(0);
        counter.walk_obligation_body(&body);
        assert_eq!(counter.0, 3);
    }
}

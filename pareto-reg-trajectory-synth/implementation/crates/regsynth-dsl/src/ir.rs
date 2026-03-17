use crate::ast;
use crate::source_map::Span;
use regsynth_types::{CompositionOp, FormalizabilityGrade, ObligationKind, RiskLevel};
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════
// Stage 1 IR: Desugared AST with resolved names
// ═══════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage1Program {
    pub items: Vec<Stage1Item>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage1Item {
    pub kind: Stage1ItemKind,
    pub span: SpanInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanInfo {
    pub start: usize,
    pub end: usize,
}

impl From<Span> for SpanInfo {
    fn from(s: Span) -> Self {
        Self {
            start: s.start,
            end: s.end,
        }
    }
}

impl From<SpanInfo> for Span {
    fn from(s: SpanInfo) -> Self {
        Self {
            start: s.start,
            end: s.end,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Stage1ItemKind {
    Obligation(Stage1Obligation),
    JurisdictionGroup {
        jurisdiction: String,
        items: Vec<Stage1Item>,
    },
    Strategy(Stage1Strategy),
    Mapping(Stage1Mapping),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage1Obligation {
    pub name: String,
    pub kind: ObligationKind,
    pub jurisdiction: Option<String>,
    pub conditions: Vec<Stage1Expr>,
    pub exemptions: Vec<Stage1Expr>,
    pub temporal_start: Option<String>,
    pub temporal_end: Option<String>,
    pub risk_level: Option<RiskLevel>,
    pub domain: Option<String>,
    pub formalizability: Option<FormalizabilityGrade>,
    pub article_refs: Vec<Stage1ArticleRef>,
    pub compositions: Vec<Stage1Composition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage1ArticleRef {
    pub framework: String,
    pub article: String,
    pub paragraph: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage1Composition {
    pub op: CompositionOp,
    pub operand: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Stage1Expr {
    Var(String),
    BoolLit(bool),
    IntLit(i64),
    FloatLit(f64),
    StringLit(String),
    DateLit(String),
    BinOp {
        op: String,
        left: Box<Stage1Expr>,
        right: Box<Stage1Expr>,
    },
    UnaryOp {
        op: String,
        operand: Box<Stage1Expr>,
    },
    IfThenElse {
        cond: Box<Stage1Expr>,
        then_br: Box<Stage1Expr>,
        else_br: Box<Stage1Expr>,
    },
    Quantifier {
        kind: String,
        var: String,
        domain: Option<Box<Stage1Expr>>,
        body: Box<Stage1Expr>,
    },
    FnCall {
        name: String,
        args: Vec<Stage1Expr>,
    },
    FieldAccess {
        object: Box<Stage1Expr>,
        field: String,
    },
    ArticleRef {
        framework: String,
        article: String,
        paragraph: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage1Strategy {
    pub name: String,
    pub fields: Vec<(String, Stage1Expr)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage1Mapping {
    pub name: String,
    pub entries: Vec<(Stage1Expr, Stage1Expr)>,
}

// ═══════════════════════════════════════════════════════════════
// Stage 2 IR: Typed, with obligation compositions expanded
// ═══════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage2Program {
    pub obligations: Vec<Stage2Obligation>,
    pub strategies: Vec<Stage2Strategy>,
    pub mappings: Vec<Stage1Mapping>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage2Obligation {
    pub id: String,
    pub name: String,
    pub kind: ObligationKind,
    pub jurisdiction: Option<String>,
    pub ty: String, // Serialized DslType representation
    pub conditions: Vec<Stage2Expr>,
    pub exemptions: Vec<Stage2Expr>,
    pub temporal_start: Option<String>,
    pub temporal_end: Option<String>,
    pub risk_level: Option<RiskLevel>,
    pub domain: Option<String>,
    pub formalizability: Option<FormalizabilityGrade>,
    pub article_refs: Vec<Stage1ArticleRef>,
    /// Expanded compositions: each is a pair (op, fully resolved obligation id)
    pub composed_with: Vec<Stage2ComposedRef>,
    pub span: SpanInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage2ComposedRef {
    pub op: CompositionOp,
    pub target_id: String,
    pub target_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Stage2Expr {
    Var {
        name: String,
        ty: String,
    },
    BoolLit(bool),
    IntLit(i64),
    FloatLit(f64),
    StringLit(String),
    DateLit(String),
    BinOp {
        op: String,
        left: Box<Stage2Expr>,
        right: Box<Stage2Expr>,
        result_ty: String,
    },
    UnaryOp {
        op: String,
        operand: Box<Stage2Expr>,
        result_ty: String,
    },
    IfThenElse {
        cond: Box<Stage2Expr>,
        then_br: Box<Stage2Expr>,
        else_br: Box<Stage2Expr>,
        result_ty: String,
    },
    Quantifier {
        kind: String,
        var: String,
        domain: Option<Box<Stage2Expr>>,
        body: Box<Stage2Expr>,
    },
    FnCall {
        name: String,
        args: Vec<Stage2Expr>,
        result_ty: String,
    },
    FieldAccess {
        object: Box<Stage2Expr>,
        field: String,
        result_ty: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage2Strategy {
    pub name: String,
    pub fields: Vec<(String, Stage2Expr)>,
}

// ═══════════════════════════════════════════════════════════════
// Stage 3 IR: Constraint-ready, temporal annotations resolved
// ═══════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage3Program {
    pub constraints: Vec<Stage3Constraint>,
    pub variables: Vec<Stage3Variable>,
    pub objectives: Vec<Stage3Objective>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage3Variable {
    pub id: String,
    pub name: String,
    pub var_type: Stage3VarType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Stage3VarType {
    Bool,
    Int,
    Float,
    Obligation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage3Constraint {
    pub id: String,
    pub kind: Stage3ConstraintKind,
    pub source_obligation: Option<String>,
    pub jurisdiction: Option<String>,
    pub temporal_start: Option<String>,
    pub temporal_end: Option<String>,
    pub formalizability: Option<FormalizabilityGrade>,
    pub span: SpanInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Stage3ConstraintKind {
    /// Boolean condition that must hold
    Condition(Stage3BoolExpr),
    /// Exemption: negates a condition
    Exemption(Stage3BoolExpr),
    /// Conjunction of two obligation constraints
    Conjunction {
        left: String,
        right: String,
    },
    /// Disjunction: at least one must hold
    Disjunction {
        left: String,
        right: String,
    },
    /// Override: left overrides right
    Override {
        primary: String,
        secondary: String,
    },
    /// Exception: left minus right
    Exception {
        base: String,
        exception: String,
    },
    /// Risk level bound
    RiskBound {
        level: RiskLevel,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Stage3BoolExpr {
    Var(String),
    Lit(bool),
    And(Box<Stage3BoolExpr>, Box<Stage3BoolExpr>),
    Or(Box<Stage3BoolExpr>, Box<Stage3BoolExpr>),
    Not(Box<Stage3BoolExpr>),
    Implies(Box<Stage3BoolExpr>, Box<Stage3BoolExpr>),
    Comparison {
        op: String,
        left: String,
        right: String,
    },
    Forall {
        var: String,
        body: Box<Stage3BoolExpr>,
    },
    Exists {
        var: String,
        body: Box<Stage3BoolExpr>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage3Objective {
    pub name: String,
    pub kind: Stage3ObjectiveKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Stage3ObjectiveKind {
    MinimizeCost,
    MaximizeCompliance,
    MinimizeRisk,
}

// ═══════════════════════════════════════════════════════════════
// Conversion functions
// ═══════════════════════════════════════════════════════════════

/// Convert an AST expression to a Stage1 expression.
pub fn ast_expr_to_stage1(expr: &ast::Expression) -> Stage1Expr {
    match &expr.kind {
        ast::ExpressionKind::Variable(name) => Stage1Expr::Var(name.clone()),
        ast::ExpressionKind::Literal(lit) => match lit {
            ast::Literal::Bool(b) => Stage1Expr::BoolLit(*b),
            ast::Literal::Int(n) => Stage1Expr::IntLit(*n),
            ast::Literal::Float(f) => Stage1Expr::FloatLit(*f),
            ast::Literal::Str(s) => Stage1Expr::StringLit(s.clone()),
            ast::Literal::Date(d) => Stage1Expr::DateLit(d.clone()),
        },
        ast::ExpressionKind::BinaryOp { op, left, right } => Stage1Expr::BinOp {
            op: op.to_string(),
            left: Box::new(ast_expr_to_stage1(left)),
            right: Box::new(ast_expr_to_stage1(right)),
        },
        ast::ExpressionKind::UnaryOp { op, operand } => Stage1Expr::UnaryOp {
            op: op.to_string(),
            operand: Box::new(ast_expr_to_stage1(operand)),
        },
        ast::ExpressionKind::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => Stage1Expr::IfThenElse {
            cond: Box::new(ast_expr_to_stage1(condition)),
            then_br: Box::new(ast_expr_to_stage1(then_branch)),
            else_br: Box::new(ast_expr_to_stage1(else_branch)),
        },
        ast::ExpressionKind::Quantifier {
            kind,
            variable,
            domain,
            body,
        } => Stage1Expr::Quantifier {
            kind: kind.to_string(),
            var: variable.clone(),
            domain: domain.as_ref().map(|d| Box::new(ast_expr_to_stage1(d))),
            body: Box::new(ast_expr_to_stage1(body)),
        },
        ast::ExpressionKind::FunctionCall { function, args } => Stage1Expr::FnCall {
            name: function.clone(),
            args: args.iter().map(ast_expr_to_stage1).collect(),
        },
        ast::ExpressionKind::FieldAccess { object, field } => Stage1Expr::FieldAccess {
            object: Box::new(ast_expr_to_stage1(object)),
            field: field.clone(),
        },
        ast::ExpressionKind::ArticleRef {
            framework,
            article,
            paragraph,
        } => Stage1Expr::ArticleRef {
            framework: framework.clone(),
            article: article.clone(),
            paragraph: paragraph.clone(),
        },
        // Desugar composition expressions into binary ops
        ast::ExpressionKind::Composition { op, left, right } => Stage1Expr::BinOp {
            op: format!("{}", op),
            left: Box::new(ast_expr_to_stage1(left)),
            right: Box::new(ast_expr_to_stage1(right)),
        },
    }
}

/// Convert an AST program to Stage1 IR.
pub fn ast_to_stage1(program: &ast::Program) -> Stage1Program {
    Stage1Program {
        items: program
            .declarations
            .iter()
            .filter_map(|d| ast_decl_to_stage1(d, &None))
            .collect(),
    }
}

fn ast_decl_to_stage1(
    decl: &ast::Declaration,
    parent_jur: &Option<String>,
) -> Option<Stage1Item> {
    let span = SpanInfo::from(decl.span);
    match &decl.kind {
        ast::DeclarationKind::Jurisdiction(j) => {
            let items = j
                .body
                .iter()
                .filter_map(|d| ast_decl_to_stage1(d, &Some(j.name.clone())))
                .collect();
            Some(Stage1Item {
                kind: Stage1ItemKind::JurisdictionGroup {
                    jurisdiction: j.name.clone(),
                    items,
                },
                span,
            })
        }
        ast::DeclarationKind::Obligation(o) => {
            let jur = o.jurisdiction.clone().or_else(|| parent_jur.clone());
            Some(Stage1Item {
                kind: Stage1ItemKind::Obligation(obligation_body_to_stage1(
                    &o.name,
                    ObligationKind::Obligation,
                    jur,
                    &o.body,
                )),
                span,
            })
        }
        ast::DeclarationKind::Permission(p) => {
            let jur = p.jurisdiction.clone().or_else(|| parent_jur.clone());
            Some(Stage1Item {
                kind: Stage1ItemKind::Obligation(obligation_body_to_stage1(
                    &p.name,
                    ObligationKind::Permission,
                    jur,
                    &p.body,
                )),
                span,
            })
        }
        ast::DeclarationKind::Prohibition(p) => {
            let jur = p.jurisdiction.clone().or_else(|| parent_jur.clone());
            Some(Stage1Item {
                kind: Stage1ItemKind::Obligation(obligation_body_to_stage1(
                    &p.name,
                    ObligationKind::Prohibition,
                    jur,
                    &p.body,
                )),
                span,
            })
        }
        ast::DeclarationKind::Framework(f) => {
            let jur = f.jurisdiction.clone().or_else(|| parent_jur.clone());
            let items = f
                .body
                .iter()
                .filter_map(|d| ast_decl_to_stage1(d, &jur))
                .collect();
            Some(Stage1Item {
                kind: Stage1ItemKind::JurisdictionGroup {
                    jurisdiction: jur.unwrap_or_else(|| f.name.clone()),
                    items,
                },
                span,
            })
        }
        ast::DeclarationKind::Strategy(s) => Some(Stage1Item {
            kind: Stage1ItemKind::Strategy(Stage1Strategy {
                name: s.name.clone(),
                fields: s
                    .body
                    .iter()
                    .map(|item| (item.key.clone(), ast_expr_to_stage1(&item.value)))
                    .collect(),
            }),
            span,
        }),
        ast::DeclarationKind::Mapping(m) => Some(Stage1Item {
            kind: Stage1ItemKind::Mapping(Stage1Mapping {
                name: m.name.clone(),
                entries: m
                    .entries
                    .iter()
                    .map(|e| (ast_expr_to_stage1(&e.from), ast_expr_to_stage1(&e.to)))
                    .collect(),
            }),
            span,
        }),
        ast::DeclarationKind::Cost(_) | ast::DeclarationKind::Temporal(_) => {
            // These are absorbed into obligation bodies
            None
        }
    }
}

fn obligation_body_to_stage1(
    name: &str,
    kind: ObligationKind,
    jurisdiction: Option<String>,
    body: &ast::ObligationBody,
) -> Stage1Obligation {
    Stage1Obligation {
        name: name.to_string(),
        kind,
        jurisdiction,
        conditions: body.conditions.iter().map(ast_expr_to_stage1).collect(),
        exemptions: body.exemptions.iter().map(ast_expr_to_stage1).collect(),
        temporal_start: body.temporal_start.clone(),
        temporal_end: body.temporal_end.clone(),
        risk_level: body.risk_level,
        domain: body.domain.clone(),
        formalizability: body.formalizability,
        article_refs: body
            .article_refs
            .iter()
            .map(|a| Stage1ArticleRef {
                framework: a.framework.clone(),
                article: a.article.clone(),
                paragraph: a.paragraph.clone(),
            })
            .collect(),
        compositions: body
            .compositions
            .iter()
            .map(|c| Stage1Composition {
                op: c.op,
                operand: c.operand.clone(),
            })
            .collect(),
    }
}

/// Convert a Stage1 expression to a Stage2 expression with type annotations.
pub fn stage1_expr_to_stage2(expr: &Stage1Expr, default_ty: &str) -> Stage2Expr {
    match expr {
        Stage1Expr::Var(name) => Stage2Expr::Var {
            name: name.clone(),
            ty: default_ty.to_string(),
        },
        Stage1Expr::BoolLit(b) => Stage2Expr::BoolLit(*b),
        Stage1Expr::IntLit(n) => Stage2Expr::IntLit(*n),
        Stage1Expr::FloatLit(f) => Stage2Expr::FloatLit(*f),
        Stage1Expr::StringLit(s) => Stage2Expr::StringLit(s.clone()),
        Stage1Expr::DateLit(d) => Stage2Expr::DateLit(d.clone()),
        Stage1Expr::BinOp { op, left, right } => Stage2Expr::BinOp {
            op: op.clone(),
            left: Box::new(stage1_expr_to_stage2(left, default_ty)),
            right: Box::new(stage1_expr_to_stage2(right, default_ty)),
            result_ty: default_ty.to_string(),
        },
        Stage1Expr::UnaryOp { op, operand } => Stage2Expr::UnaryOp {
            op: op.clone(),
            operand: Box::new(stage1_expr_to_stage2(operand, default_ty)),
            result_ty: default_ty.to_string(),
        },
        Stage1Expr::IfThenElse {
            cond,
            then_br,
            else_br,
        } => Stage2Expr::IfThenElse {
            cond: Box::new(stage1_expr_to_stage2(cond, "Bool")),
            then_br: Box::new(stage1_expr_to_stage2(then_br, default_ty)),
            else_br: Box::new(stage1_expr_to_stage2(else_br, default_ty)),
            result_ty: default_ty.to_string(),
        },
        Stage1Expr::Quantifier {
            kind,
            var,
            domain,
            body,
        } => Stage2Expr::Quantifier {
            kind: kind.clone(),
            var: var.clone(),
            domain: domain
                .as_ref()
                .map(|d| Box::new(stage1_expr_to_stage2(d, default_ty))),
            body: Box::new(stage1_expr_to_stage2(body, "Bool")),
        },
        Stage1Expr::FnCall { name, args } => Stage2Expr::FnCall {
            name: name.clone(),
            args: args
                .iter()
                .map(|a| stage1_expr_to_stage2(a, default_ty))
                .collect(),
            result_ty: default_ty.to_string(),
        },
        Stage1Expr::FieldAccess { object, field } => Stage2Expr::FieldAccess {
            object: Box::new(stage1_expr_to_stage2(object, default_ty)),
            field: field.clone(),
            result_ty: default_ty.to_string(),
        },
        Stage1Expr::ArticleRef { .. } => {
            Stage2Expr::StringLit("article_ref".to_string())
        }
    }
}

/// Convert Stage1 program to Stage2 program.
pub fn stage1_to_stage2(s1: &Stage1Program) -> Stage2Program {
    let mut obligations = Vec::new();
    let mut strategies = Vec::new();
    let mut mappings = Vec::new();
    let mut counter = 0u64;

    fn collect_obligations(
        items: &[Stage1Item],
        obligations: &mut Vec<Stage2Obligation>,
        strategies: &mut Vec<Stage2Strategy>,
        mappings: &mut Vec<Stage1Mapping>,
        counter: &mut u64,
    ) {
        for item in items {
            match &item.kind {
                Stage1ItemKind::Obligation(obl) => {
                    *counter += 1;
                    let id = format!("obl_{}", counter);
                    obligations.push(Stage2Obligation {
                        id: id.clone(),
                        name: obl.name.clone(),
                        kind: obl.kind,
                        jurisdiction: obl.jurisdiction.clone(),
                        ty: format!("{}[{}]", obl.kind, obl.jurisdiction.as_deref().unwrap_or("?")),
                        conditions: obl
                            .conditions
                            .iter()
                            .map(|e| stage1_expr_to_stage2(e, "Bool"))
                            .collect(),
                        exemptions: obl
                            .exemptions
                            .iter()
                            .map(|e| stage1_expr_to_stage2(e, "Bool"))
                            .collect(),
                        temporal_start: obl.temporal_start.clone(),
                        temporal_end: obl.temporal_end.clone(),
                        risk_level: obl.risk_level,
                        domain: obl.domain.clone(),
                        formalizability: obl.formalizability,
                        article_refs: obl.article_refs.clone(),
                        composed_with: obl
                            .compositions
                            .iter()
                            .map(|c| Stage2ComposedRef {
                                op: c.op,
                                target_id: format!("obl_ref_{}", c.operand),
                                target_name: c.operand.clone(),
                            })
                            .collect(),
                        span: item.span.clone(),
                    });
                }
                Stage1ItemKind::JurisdictionGroup { items: children, .. } => {
                    collect_obligations(children, obligations, strategies, mappings, counter);
                }
                Stage1ItemKind::Strategy(s) => {
                    strategies.push(Stage2Strategy {
                        name: s.name.clone(),
                        fields: s
                            .fields
                            .iter()
                            .map(|(k, v)| (k.clone(), stage1_expr_to_stage2(v, "Any")))
                            .collect(),
                    });
                }
                Stage1ItemKind::Mapping(m) => {
                    mappings.push(m.clone());
                }
            }
        }
    }

    collect_obligations(&s1.items, &mut obligations, &mut strategies, &mut mappings, &mut counter);

    Stage2Program {
        obligations,
        strategies,
        mappings,
    }
}

/// Convert a Stage2 boolean expression to Stage3 boolean expression.
fn stage2_expr_to_stage3_bool(expr: &Stage2Expr) -> Stage3BoolExpr {
    match expr {
        Stage2Expr::Var { name, .. } => Stage3BoolExpr::Var(name.clone()),
        Stage2Expr::BoolLit(b) => Stage3BoolExpr::Lit(*b),
        Stage2Expr::BinOp { op, left, right, .. } => match op.as_str() {
            "and" => Stage3BoolExpr::And(
                Box::new(stage2_expr_to_stage3_bool(left)),
                Box::new(stage2_expr_to_stage3_bool(right)),
            ),
            "or" => Stage3BoolExpr::Or(
                Box::new(stage2_expr_to_stage3_bool(left)),
                Box::new(stage2_expr_to_stage3_bool(right)),
            ),
            "implies" => Stage3BoolExpr::Implies(
                Box::new(stage2_expr_to_stage3_bool(left)),
                Box::new(stage2_expr_to_stage3_bool(right)),
            ),
            _ => Stage3BoolExpr::Comparison {
                op: op.clone(),
                left: format!("{:?}", left),
                right: format!("{:?}", right),
            },
        },
        Stage2Expr::UnaryOp { op, operand, .. } if op == "not" => {
            Stage3BoolExpr::Not(Box::new(stage2_expr_to_stage3_bool(operand)))
        }
        Stage2Expr::Quantifier {
            kind, var, body, ..
        } => {
            if kind == "forall" {
                Stage3BoolExpr::Forall {
                    var: var.clone(),
                    body: Box::new(stage2_expr_to_stage3_bool(body)),
                }
            } else {
                Stage3BoolExpr::Exists {
                    var: var.clone(),
                    body: Box::new(stage2_expr_to_stage3_bool(body)),
                }
            }
        }
        _ => Stage3BoolExpr::Var(format!("{:?}", expr)),
    }
}

/// Convert Stage2 to Stage3 (constraint-ready).
pub fn stage2_to_stage3(s2: &Stage2Program) -> Stage3Program {
    let mut constraints = Vec::new();
    let mut variables = Vec::new();
    let mut constraint_counter = 0u64;

    for obl in &s2.obligations {
        // Generate a variable for this obligation
        variables.push(Stage3Variable {
            id: obl.id.clone(),
            name: obl.name.clone(),
            var_type: Stage3VarType::Obligation,
        });

        // Convert conditions to constraints
        for cond in &obl.conditions {
            constraint_counter += 1;
            constraints.push(Stage3Constraint {
                id: format!("c_{}", constraint_counter),
                kind: Stage3ConstraintKind::Condition(stage2_expr_to_stage3_bool(cond)),
                source_obligation: Some(obl.id.clone()),
                jurisdiction: obl.jurisdiction.clone(),
                temporal_start: obl.temporal_start.clone(),
                temporal_end: obl.temporal_end.clone(),
                formalizability: obl.formalizability,
                span: obl.span.clone(),
            });
        }

        // Convert exemptions
        for exempt in &obl.exemptions {
            constraint_counter += 1;
            constraints.push(Stage3Constraint {
                id: format!("c_{}", constraint_counter),
                kind: Stage3ConstraintKind::Exemption(stage2_expr_to_stage3_bool(exempt)),
                source_obligation: Some(obl.id.clone()),
                jurisdiction: obl.jurisdiction.clone(),
                temporal_start: obl.temporal_start.clone(),
                temporal_end: obl.temporal_end.clone(),
                formalizability: obl.formalizability,
                span: obl.span.clone(),
            });
        }

        // Risk level constraint
        if let Some(risk) = &obl.risk_level {
            constraint_counter += 1;
            constraints.push(Stage3Constraint {
                id: format!("c_{}", constraint_counter),
                kind: Stage3ConstraintKind::RiskBound { level: *risk },
                source_obligation: Some(obl.id.clone()),
                jurisdiction: obl.jurisdiction.clone(),
                temporal_start: obl.temporal_start.clone(),
                temporal_end: obl.temporal_end.clone(),
                formalizability: obl.formalizability,
                span: obl.span.clone(),
            });
        }

        // Composition constraints
        for comp in &obl.composed_with {
            constraint_counter += 1;
            let kind = match comp.op {
                CompositionOp::Conjunction => Stage3ConstraintKind::Conjunction {
                    left: obl.id.clone(),
                    right: comp.target_id.clone(),
                },
                CompositionOp::Disjunction => Stage3ConstraintKind::Disjunction {
                    left: obl.id.clone(),
                    right: comp.target_id.clone(),
                },
                CompositionOp::Override => Stage3ConstraintKind::Override {
                    primary: obl.id.clone(),
                    secondary: comp.target_id.clone(),
                },
                CompositionOp::Exception => Stage3ConstraintKind::Exception {
                    base: obl.id.clone(),
                    exception: comp.target_id.clone(),
                },
            };
            constraints.push(Stage3Constraint {
                id: format!("c_{}", constraint_counter),
                kind,
                source_obligation: Some(obl.id.clone()),
                jurisdiction: obl.jurisdiction.clone(),
                temporal_start: obl.temporal_start.clone(),
                temporal_end: obl.temporal_end.clone(),
                formalizability: obl.formalizability,
                span: obl.span.clone(),
            });
        }
    }

    Stage3Program {
        constraints,
        variables,
        objectives: vec![
            Stage3Objective {
                name: "compliance".to_string(),
                kind: Stage3ObjectiveKind::MaximizeCompliance,
            },
            Stage3Objective {
                name: "cost".to_string(),
                kind: Stage3ObjectiveKind::MinimizeCost,
            },
        ],
    }
}

/// Full pipeline: AST → Stage1 → Stage2 → Stage3
pub fn lower_to_stage3(program: &ast::Program) -> Stage3Program {
    let s1 = ast_to_stage1(program);
    let s2 = stage1_to_stage2(&s1);
    stage2_to_stage3(&s2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;
    use crate::parser::parse;

    fn compile_to_stage3(src: &str) -> Stage3Program {
        let (tokens, _) = lex(src);
        let (program, _) = parse(tokens);
        lower_to_stage3(&program)
    }

    #[test]
    fn test_ast_to_stage1() {
        let (tokens, _) = lex(r#"
            jurisdiction "EU" {
                obligation transparency {
                    requires: true;
                    formalizability: 2;
                }
            }
        "#);
        let (program, _) = parse(tokens);
        let s1 = ast_to_stage1(&program);
        assert_eq!(s1.items.len(), 1);
    }

    #[test]
    fn test_stage1_to_stage2() {
        let (tokens, _) = lex(r#"
            obligation test_obl {
                requires: true;
                risk_level: high;
            }
        "#);
        let (program, _) = parse(tokens);
        let s1 = ast_to_stage1(&program);
        let s2 = stage1_to_stage2(&s1);
        assert_eq!(s2.obligations.len(), 1);
        assert_eq!(s2.obligations[0].name, "test_obl");
        assert_eq!(s2.obligations[0].risk_level, Some(RiskLevel::High));
    }

    #[test]
    fn test_full_pipeline() {
        let s3 = compile_to_stage3(r#"
            obligation obl_a {
                requires: true;
                risk_level: high;
                formalizability: 2;
            }
            obligation obl_b {
                requires: false;
                compose: ⊗ obl_a;
            }
        "#);
        // Should have constraints for conditions, risk bound, and composition
        assert!(!s3.constraints.is_empty());
        assert!(!s3.variables.is_empty());
    }

    #[test]
    fn test_stage3_has_objectives() {
        let s3 = compile_to_stage3("obligation test { requires: true; }");
        assert_eq!(s3.objectives.len(), 2);
    }

    #[test]
    fn test_span_info_conversion() {
        let span = Span::new(10, 20);
        let info: SpanInfo = span.into();
        assert_eq!(info.start, 10);
        assert_eq!(info.end, 20);
        let back: Span = info.into();
        assert_eq!(back, span);
    }
}

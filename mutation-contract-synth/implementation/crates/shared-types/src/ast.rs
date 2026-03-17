//! Complete AST for the MutSpec imperative language.
//!
//! Expressions, statements, functions, and programs are defined here, along
//! with a visitor trait, Display implementations for pretty-printing, and
//! builder utilities.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::errors::SpanInfo;
use crate::types::{QfLiaType, Variable};

// ---------------------------------------------------------------------------
// Operator enums
// ---------------------------------------------------------------------------

/// Arithmetic binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

impl ArithOp {
    pub fn symbol(&self) -> &'static str {
        match self {
            ArithOp::Add => "+",
            ArithOp::Sub => "-",
            ArithOp::Mul => "*",
            ArithOp::Div => "/",
            ArithOp::Mod => "%",
        }
    }

    pub fn all() -> &'static [ArithOp] {
        &[
            ArithOp::Add,
            ArithOp::Sub,
            ArithOp::Mul,
            ArithOp::Div,
            ArithOp::Mod,
        ]
    }

    /// Precedence (higher = tighter binding).
    pub fn precedence(&self) -> u8 {
        match self {
            ArithOp::Add | ArithOp::Sub => 1,
            ArithOp::Mul | ArithOp::Div | ArithOp::Mod => 2,
        }
    }

    pub fn is_commutative(&self) -> bool {
        matches!(self, ArithOp::Add | ArithOp::Mul)
    }

    /// Parse from symbol string.
    pub fn from_symbol(s: &str) -> Option<ArithOp> {
        match s {
            "+" => Some(ArithOp::Add),
            "-" => Some(ArithOp::Sub),
            "*" => Some(ArithOp::Mul),
            "/" => Some(ArithOp::Div),
            "%" => Some(ArithOp::Mod),
            _ => None,
        }
    }
}

impl fmt::Display for ArithOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.symbol())
    }
}

/// Relational operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelOp {
    Lt,
    Le,
    Eq,
    Ne,
    Ge,
    Gt,
}

impl RelOp {
    pub fn symbol(&self) -> &'static str {
        match self {
            RelOp::Lt => "<",
            RelOp::Le => "<=",
            RelOp::Eq => "==",
            RelOp::Ne => "!=",
            RelOp::Ge => ">=",
            RelOp::Gt => ">",
        }
    }

    pub fn all() -> &'static [RelOp] {
        &[
            RelOp::Lt,
            RelOp::Le,
            RelOp::Eq,
            RelOp::Ne,
            RelOp::Ge,
            RelOp::Gt,
        ]
    }

    /// Negate the relation (e.g., < becomes >=).
    pub fn negate(&self) -> RelOp {
        match self {
            RelOp::Lt => RelOp::Ge,
            RelOp::Le => RelOp::Gt,
            RelOp::Eq => RelOp::Ne,
            RelOp::Ne => RelOp::Eq,
            RelOp::Ge => RelOp::Lt,
            RelOp::Gt => RelOp::Le,
        }
    }

    /// Flip operands (e.g., < becomes >).
    pub fn flip(&self) -> RelOp {
        match self {
            RelOp::Lt => RelOp::Gt,
            RelOp::Le => RelOp::Ge,
            RelOp::Eq => RelOp::Eq,
            RelOp::Ne => RelOp::Ne,
            RelOp::Ge => RelOp::Le,
            RelOp::Gt => RelOp::Lt,
        }
    }

    pub fn from_symbol(s: &str) -> Option<RelOp> {
        match s {
            "<" => Some(RelOp::Lt),
            "<=" => Some(RelOp::Le),
            "==" => Some(RelOp::Eq),
            "!=" => Some(RelOp::Ne),
            ">=" => Some(RelOp::Ge),
            ">" => Some(RelOp::Gt),
            _ => None,
        }
    }
}

impl fmt::Display for RelOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.symbol())
    }
}

/// Logical operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LogicOp {
    And,
    Or,
}

impl LogicOp {
    pub fn symbol(&self) -> &'static str {
        match self {
            LogicOp::And => "&&",
            LogicOp::Or => "||",
        }
    }

    pub fn from_symbol(s: &str) -> Option<LogicOp> {
        match s {
            "&&" => Some(LogicOp::And),
            "||" => Some(LogicOp::Or),
            _ => None,
        }
    }
}

impl fmt::Display for LogicOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.symbol())
    }
}

// ---------------------------------------------------------------------------
// Expression
// ---------------------------------------------------------------------------

/// An expression in the MutSpec language.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Expression {
    /// Integer literal.
    IntLiteral(i64),
    /// Boolean literal.
    BoolLiteral(bool),
    /// Variable reference.
    Var(String),
    /// Binary arithmetic: lhs op rhs.
    BinaryArith {
        op: ArithOp,
        lhs: Box<Expression>,
        rhs: Box<Expression>,
    },
    /// Unary arithmetic negation.
    UnaryArith(Box<Expression>),
    /// Relational comparison.
    Relational {
        op: RelOp,
        lhs: Box<Expression>,
        rhs: Box<Expression>,
    },
    /// Logical AND.
    LogicalAnd(Box<Expression>, Box<Expression>),
    /// Logical OR.
    LogicalOr(Box<Expression>, Box<Expression>),
    /// Logical NOT.
    LogicalNot(Box<Expression>),
    /// Array access: array[index].
    ArrayAccess {
        array: Box<Expression>,
        index: Box<Expression>,
    },
    /// Function call.
    FunctionCall { name: String, args: Vec<Expression> },
    /// Ternary conditional: condition ? then_expr : else_expr.
    Conditional {
        condition: Box<Expression>,
        then_expr: Box<Expression>,
        else_expr: Box<Expression>,
    },
}

impl Expression {
    // -- Constructors -------------------------------------------------------

    pub fn int_lit(v: i64) -> Self {
        Expression::IntLiteral(v)
    }

    pub fn bool_lit(v: bool) -> Self {
        Expression::BoolLiteral(v)
    }

    pub fn var(name: impl Into<String>) -> Self {
        Expression::Var(name.into())
    }

    pub fn binary_arith(op: ArithOp, lhs: Expression, rhs: Expression) -> Self {
        Expression::BinaryArith {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    pub fn add(lhs: Expression, rhs: Expression) -> Self {
        Self::binary_arith(ArithOp::Add, lhs, rhs)
    }

    pub fn sub(lhs: Expression, rhs: Expression) -> Self {
        Self::binary_arith(ArithOp::Sub, lhs, rhs)
    }

    pub fn mul(lhs: Expression, rhs: Expression) -> Self {
        Self::binary_arith(ArithOp::Mul, lhs, rhs)
    }

    pub fn div(lhs: Expression, rhs: Expression) -> Self {
        Self::binary_arith(ArithOp::Div, lhs, rhs)
    }

    pub fn modulo(lhs: Expression, rhs: Expression) -> Self {
        Self::binary_arith(ArithOp::Mod, lhs, rhs)
    }

    pub fn negate(expr: Expression) -> Self {
        Expression::UnaryArith(Box::new(expr))
    }

    pub fn relational(op: RelOp, lhs: Expression, rhs: Expression) -> Self {
        Expression::Relational {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    pub fn lt(lhs: Expression, rhs: Expression) -> Self {
        Self::relational(RelOp::Lt, lhs, rhs)
    }

    pub fn le(lhs: Expression, rhs: Expression) -> Self {
        Self::relational(RelOp::Le, lhs, rhs)
    }

    pub fn eq(lhs: Expression, rhs: Expression) -> Self {
        Self::relational(RelOp::Eq, lhs, rhs)
    }

    pub fn ne(lhs: Expression, rhs: Expression) -> Self {
        Self::relational(RelOp::Ne, lhs, rhs)
    }

    pub fn ge(lhs: Expression, rhs: Expression) -> Self {
        Self::relational(RelOp::Ge, lhs, rhs)
    }

    pub fn gt(lhs: Expression, rhs: Expression) -> Self {
        Self::relational(RelOp::Gt, lhs, rhs)
    }

    pub fn and(lhs: Expression, rhs: Expression) -> Self {
        Expression::LogicalAnd(Box::new(lhs), Box::new(rhs))
    }

    pub fn or(lhs: Expression, rhs: Expression) -> Self {
        Expression::LogicalOr(Box::new(lhs), Box::new(rhs))
    }

    pub fn not(expr: Expression) -> Self {
        Expression::LogicalNot(Box::new(expr))
    }

    pub fn array_access(array: Expression, index: Expression) -> Self {
        Expression::ArrayAccess {
            array: Box::new(array),
            index: Box::new(index),
        }
    }

    pub fn call(name: impl Into<String>, args: Vec<Expression>) -> Self {
        Expression::FunctionCall {
            name: name.into(),
            args,
        }
    }

    pub fn conditional(cond: Expression, then_e: Expression, else_e: Expression) -> Self {
        Expression::Conditional {
            condition: Box::new(cond),
            then_expr: Box::new(then_e),
            else_expr: Box::new(else_e),
        }
    }

    // -- Queries ------------------------------------------------------------

    /// Is this a literal constant?
    pub fn is_literal(&self) -> bool {
        matches!(self, Expression::IntLiteral(_) | Expression::BoolLiteral(_))
    }

    /// Is this a simple variable reference?
    pub fn is_var(&self) -> bool {
        matches!(self, Expression::Var(_))
    }

    /// Is this a leaf expression (no sub-expressions)?
    pub fn is_leaf(&self) -> bool {
        matches!(
            self,
            Expression::IntLiteral(_) | Expression::BoolLiteral(_) | Expression::Var(_)
        )
    }

    /// Collect all variable names referenced in this expression.
    pub fn referenced_vars(&self) -> Vec<String> {
        let mut vars = Vec::new();
        self.collect_vars(&mut vars);
        vars.sort();
        vars.dedup();
        vars
    }

    fn collect_vars(&self, vars: &mut Vec<String>) {
        match self {
            Expression::Var(name) => vars.push(name.clone()),
            Expression::IntLiteral(_) | Expression::BoolLiteral(_) => {}
            Expression::BinaryArith { lhs, rhs, .. } => {
                lhs.collect_vars(vars);
                rhs.collect_vars(vars);
            }
            Expression::UnaryArith(e) | Expression::LogicalNot(e) => e.collect_vars(vars),
            Expression::Relational { lhs, rhs, .. } => {
                lhs.collect_vars(vars);
                rhs.collect_vars(vars);
            }
            Expression::LogicalAnd(l, r) | Expression::LogicalOr(l, r) => {
                l.collect_vars(vars);
                r.collect_vars(vars);
            }
            Expression::ArrayAccess { array, index } => {
                array.collect_vars(vars);
                index.collect_vars(vars);
            }
            Expression::FunctionCall { args, .. } => {
                for arg in args {
                    arg.collect_vars(vars);
                }
            }
            Expression::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                condition.collect_vars(vars);
                then_expr.collect_vars(vars);
                else_expr.collect_vars(vars);
            }
        }
    }

    /// Size of the expression tree (number of nodes).
    pub fn size(&self) -> usize {
        match self {
            Expression::IntLiteral(_) | Expression::BoolLiteral(_) | Expression::Var(_) => 1,
            Expression::BinaryArith { lhs, rhs, .. } => 1 + lhs.size() + rhs.size(),
            Expression::UnaryArith(e) | Expression::LogicalNot(e) => 1 + e.size(),
            Expression::Relational { lhs, rhs, .. } => 1 + lhs.size() + rhs.size(),
            Expression::LogicalAnd(l, r) | Expression::LogicalOr(l, r) => 1 + l.size() + r.size(),
            Expression::ArrayAccess { array, index } => 1 + array.size() + index.size(),
            Expression::FunctionCall { args, .. } => {
                1 + args.iter().map(|a| a.size()).sum::<usize>()
            }
            Expression::Conditional {
                condition,
                then_expr,
                else_expr,
            } => 1 + condition.size() + then_expr.size() + else_expr.size(),
        }
    }

    /// Depth of the expression tree.
    pub fn depth(&self) -> usize {
        match self {
            Expression::IntLiteral(_) | Expression::BoolLiteral(_) | Expression::Var(_) => 1,
            Expression::BinaryArith { lhs, rhs, .. } | Expression::Relational { lhs, rhs, .. } => {
                1 + lhs.depth().max(rhs.depth())
            }
            Expression::UnaryArith(e) | Expression::LogicalNot(e) => 1 + e.depth(),
            Expression::LogicalAnd(l, r) | Expression::LogicalOr(l, r) => {
                1 + l.depth().max(r.depth())
            }
            Expression::ArrayAccess { array, index } => 1 + array.depth().max(index.depth()),
            Expression::FunctionCall { args, .. } => {
                1 + args.iter().map(|a| a.depth()).max().unwrap_or(0)
            }
            Expression::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                1 + condition
                    .depth()
                    .max(then_expr.depth())
                    .max(else_expr.depth())
            }
        }
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::IntLiteral(v) => write!(f, "{v}"),
            Expression::BoolLiteral(b) => write!(f, "{b}"),
            Expression::Var(name) => write!(f, "{name}"),
            Expression::BinaryArith { op, lhs, rhs } => write!(f, "({lhs} {op} {rhs})"),
            Expression::UnaryArith(e) => write!(f, "(-{e})"),
            Expression::Relational { op, lhs, rhs } => write!(f, "({lhs} {op} {rhs})"),
            Expression::LogicalAnd(l, r) => write!(f, "({l} && {r})"),
            Expression::LogicalOr(l, r) => write!(f, "({l} || {r})"),
            Expression::LogicalNot(e) => write!(f, "(!{e})"),
            Expression::ArrayAccess { array, index } => write!(f, "{array}[{index}]"),
            Expression::FunctionCall { name, args } => {
                write!(f, "{name}(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                write!(f, ")")
            }
            Expression::Conditional {
                condition,
                then_expr,
                else_expr,
            } => write!(f, "({condition} ? {then_expr} : {else_expr})"),
        }
    }
}

// ---------------------------------------------------------------------------
// Statement
// ---------------------------------------------------------------------------

/// A statement in the MutSpec language.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Statement {
    /// Variable assignment: target = value.
    Assign {
        target: String,
        value: Expression,
        span: Option<SpanInfo>,
    },
    /// If-else branching.
    IfElse {
        condition: Expression,
        then_branch: Box<Statement>,
        else_branch: Option<Box<Statement>>,
        span: Option<SpanInfo>,
    },
    /// Sequence of statements.
    Sequence(Vec<Statement>),
    /// Return statement.
    Return {
        value: Option<Expression>,
        span: Option<SpanInfo>,
    },
    /// Assertion.
    Assert {
        condition: Expression,
        message: Option<String>,
        span: Option<SpanInfo>,
    },
    /// Variable declaration with optional initializer.
    VarDecl {
        var: Variable,
        init: Option<Expression>,
        span: Option<SpanInfo>,
    },
    /// Block scope.
    Block(Vec<Statement>),
}

impl Statement {
    // -- Constructors -------------------------------------------------------

    pub fn assign(target: impl Into<String>, value: Expression) -> Self {
        Statement::Assign {
            target: target.into(),
            value,
            span: None,
        }
    }

    pub fn if_else(
        condition: Expression,
        then_branch: Statement,
        else_branch: Option<Statement>,
    ) -> Self {
        Statement::IfElse {
            condition,
            then_branch: Box::new(then_branch),
            else_branch: else_branch.map(Box::new),
            span: None,
        }
    }

    pub fn sequence(stmts: Vec<Statement>) -> Self {
        if stmts.len() == 1 {
            stmts.into_iter().next().unwrap()
        } else {
            Statement::Sequence(stmts)
        }
    }

    pub fn ret(value: Option<Expression>) -> Self {
        Statement::Return { value, span: None }
    }

    pub fn assert(condition: Expression, message: Option<String>) -> Self {
        Statement::Assert {
            condition,
            message,
            span: None,
        }
    }

    pub fn var_decl(var: Variable, init: Option<Expression>) -> Self {
        Statement::VarDecl {
            var,
            init,
            span: None,
        }
    }

    pub fn block(stmts: Vec<Statement>) -> Self {
        Statement::Block(stmts)
    }

    // -- Span attachment ----------------------------------------------------

    pub fn with_span(mut self, span: SpanInfo) -> Self {
        match &mut self {
            Statement::Assign { span: s, .. }
            | Statement::IfElse { span: s, .. }
            | Statement::Return { span: s, .. }
            | Statement::Assert { span: s, .. }
            | Statement::VarDecl { span: s, .. } => {
                *s = Some(span);
            }
            Statement::Sequence(_) | Statement::Block(_) => {}
        }
        self
    }

    pub fn span(&self) -> Option<&SpanInfo> {
        match self {
            Statement::Assign { span, .. }
            | Statement::IfElse { span, .. }
            | Statement::Return { span, .. }
            | Statement::Assert { span, .. }
            | Statement::VarDecl { span, .. } => span.as_ref(),
            Statement::Sequence(_) | Statement::Block(_) => None,
        }
    }

    // -- Queries ------------------------------------------------------------

    /// Collect all variable names assigned in this statement (and children).
    pub fn assigned_vars(&self) -> Vec<String> {
        let mut vars = Vec::new();
        self.collect_assigned(&mut vars);
        vars.sort();
        vars.dedup();
        vars
    }

    fn collect_assigned(&self, vars: &mut Vec<String>) {
        match self {
            Statement::Assign { target, .. } => vars.push(target.clone()),
            Statement::VarDecl { var, .. } => vars.push(var.name.clone()),
            Statement::IfElse {
                then_branch,
                else_branch,
                ..
            } => {
                then_branch.collect_assigned(vars);
                if let Some(eb) = else_branch {
                    eb.collect_assigned(vars);
                }
            }
            Statement::Sequence(stmts) | Statement::Block(stmts) => {
                for s in stmts {
                    s.collect_assigned(vars);
                }
            }
            Statement::Return { .. } | Statement::Assert { .. } => {}
        }
    }

    /// Number of statements (recursive count).
    pub fn statement_count(&self) -> usize {
        match self {
            Statement::Sequence(stmts) | Statement::Block(stmts) => {
                stmts.iter().map(|s| s.statement_count()).sum()
            }
            Statement::IfElse {
                then_branch,
                else_branch,
                ..
            } => {
                1 + then_branch.statement_count()
                    + else_branch.as_ref().map_or(0, |e| e.statement_count())
            }
            _ => 1,
        }
    }

    /// Returns true if this is a return statement.
    pub fn is_return(&self) -> bool {
        matches!(self, Statement::Return { .. })
    }

    /// Returns true if this is an empty sequence or block.
    pub fn is_empty(&self) -> bool {
        match self {
            Statement::Sequence(stmts) | Statement::Block(stmts) => stmts.is_empty(),
            _ => false,
        }
    }

    /// Flatten nested sequences into a single sequence.
    pub fn flatten(self) -> Statement {
        match self {
            Statement::Sequence(stmts) => {
                let mut flat = Vec::new();
                for s in stmts {
                    match s.flatten() {
                        Statement::Sequence(inner) => flat.extend(inner),
                        other => flat.push(other),
                    }
                }
                Statement::Sequence(flat)
            }
            Statement::Block(stmts) => {
                let mut flat = Vec::new();
                for s in stmts {
                    flat.push(s.flatten());
                }
                Statement::Block(flat)
            }
            Statement::IfElse {
                condition,
                then_branch,
                else_branch,
                span,
            } => Statement::IfElse {
                condition,
                then_branch: Box::new(then_branch.flatten()),
                else_branch: else_branch.map(|e| Box::new(e.flatten())),
                span,
            },
            other => other,
        }
    }
}

impl fmt::Display for Statement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Statement::Assign { target, value, .. } => {
                write!(f, "{target} = {value};")
            }
            Statement::IfElse {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                write!(f, "if ({condition}) {{ {then_branch} }}")?;
                if let Some(eb) = else_branch {
                    write!(f, " else {{ {eb} }}")?;
                }
                Ok(())
            }
            Statement::Sequence(stmts) => {
                for (i, s) in stmts.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{s}")?;
                }
                Ok(())
            }
            Statement::Return { value: Some(v), .. } => write!(f, "return {v};"),
            Statement::Return { value: None, .. } => write!(f, "return;"),
            Statement::Assert {
                condition,
                message: Some(msg),
                ..
            } => write!(f, "assert({condition}, \"{msg}\");"),
            Statement::Assert {
                condition,
                message: None,
                ..
            } => write!(f, "assert({condition});"),
            Statement::VarDecl {
                var,
                init: Some(init),
                ..
            } => write!(f, "{} {} = {init};", var.ty, var.name),
            Statement::VarDecl {
                var, init: None, ..
            } => write!(f, "{} {};", var.ty, var.name),
            Statement::Block(stmts) => {
                write!(f, "{{ ")?;
                for s in stmts {
                    write!(f, "{s} ")?;
                }
                write!(f, "}}")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Function / Program
// ---------------------------------------------------------------------------

/// A function in the MutSpec language.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    pub params: Vec<Variable>,
    pub return_type: QfLiaType,
    pub body: Statement,
    pub span: Option<SpanInfo>,
}

impl Function {
    pub fn new(
        name: impl Into<String>,
        params: Vec<Variable>,
        return_type: QfLiaType,
        body: Statement,
    ) -> Self {
        Self {
            name: name.into(),
            params,
            return_type,
            body,
            span: None,
        }
    }

    pub fn with_span(mut self, span: SpanInfo) -> Self {
        self.span = Some(span);
        self
    }

    pub fn arity(&self) -> usize {
        self.params.len()
    }

    pub fn param_names(&self) -> Vec<&str> {
        self.params.iter().map(|p| p.name.as_str()).collect()
    }

    pub fn param_types(&self) -> Vec<QfLiaType> {
        self.params.iter().map(|p| p.ty).collect()
    }

    /// Build a [`FunctionSignature`] from this function.
    pub fn signature(&self) -> crate::types::FunctionSignature {
        crate::types::FunctionSignature::new(
            self.name.clone(),
            self.params.iter().map(|p| (p.name.clone(), p.ty)).collect(),
            self.return_type,
        )
    }

    /// Statement count.
    pub fn statement_count(&self) -> usize {
        self.body.statement_count()
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}(", self.return_type, self.name)?;
        for (i, p) in self.params.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{} {}", p.ty, p.name)?;
        }
        writeln!(f, ") {{")?;
        writeln!(f, "  {}", self.body)?;
        writeln!(f, "}}")
    }
}

/// A complete program.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Program {
    pub functions: Vec<Function>,
    pub name: Option<String>,
}

impl Program {
    pub fn new(functions: Vec<Function>) -> Self {
        Self {
            functions,
            name: None,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn function(&self, name: &str) -> Option<&Function> {
        self.functions.iter().find(|f| f.name == name)
    }

    pub fn function_mut(&mut self, name: &str) -> Option<&mut Function> {
        self.functions.iter_mut().find(|f| f.name == name)
    }

    pub fn function_names(&self) -> Vec<&str> {
        self.functions.iter().map(|f| f.name.as_str()).collect()
    }

    pub fn is_empty(&self) -> bool {
        self.functions.is_empty()
    }

    pub fn function_count(&self) -> usize {
        self.functions.len()
    }

    /// Total statement count across all functions.
    pub fn total_statements(&self) -> usize {
        self.functions.iter().map(|f| f.statement_count()).sum()
    }
}

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref name) = self.name {
            writeln!(f, "// Program: {name}")?;
        }
        for func in &self.functions {
            writeln!(f)?;
            write!(f, "{func}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// AstVisitor
// ---------------------------------------------------------------------------

/// Visitor trait for walking AST nodes.
pub trait AstVisitor {
    /// Called before visiting an expression. Return false to skip children.
    fn visit_expr(&mut self, _expr: &Expression) -> bool {
        true
    }

    /// Called after visiting an expression and its children.
    fn post_visit_expr(&mut self, _expr: &Expression) {}

    /// Called before visiting a statement. Return false to skip children.
    fn visit_stmt(&mut self, _stmt: &Statement) -> bool {
        true
    }

    /// Called after visiting a statement and its children.
    fn post_visit_stmt(&mut self, _stmt: &Statement) {}

    /// Called for each function.
    fn visit_function(&mut self, _func: &Function) {}

    /// Called for the program.
    fn visit_program(&mut self, _program: &Program) {}
}

/// Walk an expression tree with a visitor.
pub fn walk_expr(visitor: &mut dyn AstVisitor, expr: &Expression) {
    if !visitor.visit_expr(expr) {
        return;
    }
    match expr {
        Expression::IntLiteral(_) | Expression::BoolLiteral(_) | Expression::Var(_) => {}
        Expression::BinaryArith { lhs, rhs, .. } | Expression::Relational { lhs, rhs, .. } => {
            walk_expr(visitor, lhs);
            walk_expr(visitor, rhs);
        }
        Expression::UnaryArith(e) | Expression::LogicalNot(e) => {
            walk_expr(visitor, e);
        }
        Expression::LogicalAnd(l, r) | Expression::LogicalOr(l, r) => {
            walk_expr(visitor, l);
            walk_expr(visitor, r);
        }
        Expression::ArrayAccess { array, index } => {
            walk_expr(visitor, array);
            walk_expr(visitor, index);
        }
        Expression::FunctionCall { args, .. } => {
            for arg in args {
                walk_expr(visitor, arg);
            }
        }
        Expression::Conditional {
            condition,
            then_expr,
            else_expr,
        } => {
            walk_expr(visitor, condition);
            walk_expr(visitor, then_expr);
            walk_expr(visitor, else_expr);
        }
    }
    visitor.post_visit_expr(expr);
}

/// Walk a statement tree with a visitor.
pub fn walk_stmt(visitor: &mut dyn AstVisitor, stmt: &Statement) {
    if !visitor.visit_stmt(stmt) {
        return;
    }
    match stmt {
        Statement::Assign { value, .. } => walk_expr(visitor, value),
        Statement::IfElse {
            condition,
            then_branch,
            else_branch,
            ..
        } => {
            walk_expr(visitor, condition);
            walk_stmt(visitor, then_branch);
            if let Some(eb) = else_branch {
                walk_stmt(visitor, eb);
            }
        }
        Statement::Sequence(stmts) | Statement::Block(stmts) => {
            for s in stmts {
                walk_stmt(visitor, s);
            }
        }
        Statement::Return { value: Some(v), .. } => walk_expr(visitor, v),
        Statement::Return { value: None, .. } => {}
        Statement::Assert { condition, .. } => walk_expr(visitor, condition),
        Statement::VarDecl { init: Some(e), .. } => walk_expr(visitor, e),
        Statement::VarDecl { init: None, .. } => {}
    }
    visitor.post_visit_stmt(stmt);
}

/// Walk an entire program with a visitor.
pub fn walk_program(visitor: &mut dyn AstVisitor, program: &Program) {
    visitor.visit_program(program);
    for func in &program.functions {
        visitor.visit_function(func);
        walk_stmt(visitor, &func.body);
    }
}

// ---------------------------------------------------------------------------
// Counting visitor (example utility)
// ---------------------------------------------------------------------------

/// A visitor that counts expression and statement nodes.
pub struct CountingVisitor {
    pub expr_count: usize,
    pub stmt_count: usize,
}

impl CountingVisitor {
    pub fn new() -> Self {
        Self {
            expr_count: 0,
            stmt_count: 0,
        }
    }
}

impl Default for CountingVisitor {
    fn default() -> Self {
        Self::new()
    }
}

impl AstVisitor for CountingVisitor {
    fn visit_expr(&mut self, _expr: &Expression) -> bool {
        self.expr_count += 1;
        true
    }

    fn visit_stmt(&mut self, _stmt: &Statement) -> bool {
        self.stmt_count += 1;
        true
    }
}

// ---------------------------------------------------------------------------
// Variable-collecting visitor
// ---------------------------------------------------------------------------

/// Collects all variable names referenced in the AST.
pub struct VarCollector {
    pub vars: Vec<String>,
}

impl VarCollector {
    pub fn new() -> Self {
        Self { vars: Vec::new() }
    }

    pub fn unique_vars(&self) -> Vec<String> {
        let mut v = self.vars.clone();
        v.sort();
        v.dedup();
        v
    }
}

impl Default for VarCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl AstVisitor for VarCollector {
    fn visit_expr(&mut self, expr: &Expression) -> bool {
        if let Expression::Var(name) = expr {
            self.vars.push(name.clone());
        }
        true
    }
}

// ---------------------------------------------------------------------------
// ExprBuilder for ergonomic AST construction
// ---------------------------------------------------------------------------

/// Builder for constructing expressions in a readable chain style.
pub struct ExprBuilder;

impl ExprBuilder {
    pub fn int(v: i64) -> Expression {
        Expression::int_lit(v)
    }

    pub fn boolean(v: bool) -> Expression {
        Expression::bool_lit(v)
    }

    pub fn var(name: impl Into<String>) -> Expression {
        Expression::var(name)
    }

    pub fn add(l: Expression, r: Expression) -> Expression {
        Expression::add(l, r)
    }

    pub fn sub(l: Expression, r: Expression) -> Expression {
        Expression::sub(l, r)
    }

    pub fn mul(l: Expression, r: Expression) -> Expression {
        Expression::mul(l, r)
    }

    pub fn lt(l: Expression, r: Expression) -> Expression {
        Expression::lt(l, r)
    }

    pub fn gt(l: Expression, r: Expression) -> Expression {
        Expression::gt(l, r)
    }

    pub fn eq(l: Expression, r: Expression) -> Expression {
        Expression::eq(l, r)
    }

    pub fn and(l: Expression, r: Expression) -> Expression {
        Expression::and(l, r)
    }

    pub fn or(l: Expression, r: Expression) -> Expression {
        Expression::or(l, r)
    }

    pub fn not(e: Expression) -> Expression {
        Expression::not(e)
    }

    pub fn cond(c: Expression, t: Expression, e: Expression) -> Expression {
        Expression::conditional(c, t, e)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::QfLiaType;

    // -- Operator tests --

    #[test]
    fn test_arith_op() {
        assert_eq!(ArithOp::Add.symbol(), "+");
        assert_eq!(ArithOp::all().len(), 5);
        assert!(ArithOp::Add.is_commutative());
        assert!(!ArithOp::Sub.is_commutative());
        assert!(ArithOp::Add.precedence() < ArithOp::Mul.precedence());
    }

    #[test]
    fn test_arith_op_from_symbol() {
        assert_eq!(ArithOp::from_symbol("+"), Some(ArithOp::Add));
        assert_eq!(ArithOp::from_symbol("?"), None);
    }

    #[test]
    fn test_rel_op() {
        assert_eq!(RelOp::Lt.symbol(), "<");
        assert_eq!(RelOp::Lt.negate(), RelOp::Ge);
        assert_eq!(RelOp::Lt.flip(), RelOp::Gt);
        assert_eq!(RelOp::all().len(), 6);
    }

    #[test]
    fn test_rel_op_from_symbol() {
        assert_eq!(RelOp::from_symbol("=="), Some(RelOp::Eq));
        assert_eq!(RelOp::from_symbol("?"), None);
    }

    #[test]
    fn test_logic_op() {
        assert_eq!(LogicOp::And.symbol(), "&&");
        assert_eq!(LogicOp::from_symbol("||"), Some(LogicOp::Or));
        assert_eq!(LogicOp::from_symbol("!"), None);
    }

    // -- Expression tests --

    #[test]
    fn test_expression_int_literal() {
        let e = Expression::int_lit(42);
        assert!(e.is_literal());
        assert!(e.is_leaf());
        assert_eq!(e.size(), 1);
        assert_eq!(e.to_string(), "42");
    }

    #[test]
    fn test_expression_bool_literal() {
        let e = Expression::bool_lit(true);
        assert!(e.is_literal());
        assert_eq!(e.to_string(), "true");
    }

    #[test]
    fn test_expression_var() {
        let e = Expression::var("x");
        assert!(e.is_var());
        assert!(e.is_leaf());
        assert_eq!(e.referenced_vars(), vec!["x"]);
    }

    #[test]
    fn test_expression_binary_arith() {
        let e = Expression::add(Expression::var("x"), Expression::int_lit(1));
        assert_eq!(e.size(), 3);
        assert_eq!(e.depth(), 2);
        let s = e.to_string();
        assert!(s.contains("+"));
    }

    #[test]
    fn test_expression_relational() {
        let e = Expression::lt(Expression::var("x"), Expression::int_lit(10));
        let s = e.to_string();
        assert!(s.contains("<"));
    }

    #[test]
    fn test_expression_logical() {
        let e = Expression::and(Expression::bool_lit(true), Expression::bool_lit(false));
        let s = e.to_string();
        assert!(s.contains("&&"));
    }

    #[test]
    fn test_expression_not() {
        let e = Expression::not(Expression::bool_lit(true));
        let s = e.to_string();
        assert!(s.contains("!"));
    }

    #[test]
    fn test_expression_array_access() {
        let e = Expression::array_access(Expression::var("a"), Expression::int_lit(0));
        let s = e.to_string();
        assert!(s.contains("a[0]"));
    }

    #[test]
    fn test_expression_function_call() {
        let e = Expression::call("max", vec![Expression::var("a"), Expression::var("b")]);
        let s = e.to_string();
        assert!(s.contains("max(a, b)"));
    }

    #[test]
    fn test_expression_conditional() {
        let e = Expression::conditional(
            Expression::bool_lit(true),
            Expression::int_lit(1),
            Expression::int_lit(0),
        );
        let s = e.to_string();
        assert!(s.contains("?"));
        assert!(s.contains(":"));
    }

    #[test]
    fn test_expression_referenced_vars() {
        let e = Expression::add(
            Expression::add(Expression::var("x"), Expression::var("y")),
            Expression::var("x"),
        );
        let vars = e.referenced_vars();
        assert_eq!(vars, vec!["x", "y"]);
    }

    // -- Statement tests --

    #[test]
    fn test_statement_assign() {
        let s = Statement::assign("x", Expression::int_lit(42));
        assert_eq!(s.assigned_vars(), vec!["x"]);
        assert_eq!(s.statement_count(), 1);
        let display = s.to_string();
        assert!(display.contains("x = 42"));
    }

    #[test]
    fn test_statement_if_else() {
        let s = Statement::if_else(
            Expression::bool_lit(true),
            Statement::assign("x", Expression::int_lit(1)),
            Some(Statement::assign("x", Expression::int_lit(0))),
        );
        assert_eq!(s.statement_count(), 3); // if + 2 assigns
        let display = s.to_string();
        assert!(display.contains("if"));
        assert!(display.contains("else"));
    }

    #[test]
    fn test_statement_sequence() {
        let s = Statement::sequence(vec![
            Statement::assign("x", Expression::int_lit(1)),
            Statement::assign("y", Expression::int_lit(2)),
        ]);
        assert_eq!(s.statement_count(), 2);
    }

    #[test]
    fn test_statement_return() {
        let s = Statement::ret(Some(Expression::var("x")));
        assert!(s.is_return());
        let display = s.to_string();
        assert!(display.contains("return x"));
    }

    #[test]
    fn test_statement_assert() {
        let s = Statement::assert(Expression::bool_lit(true), Some("must be true".into()));
        let display = s.to_string();
        assert!(display.contains("assert"));
        assert!(display.contains("must be true"));
    }

    #[test]
    fn test_statement_var_decl() {
        let v = Variable::local("x", QfLiaType::Int);
        let s = Statement::var_decl(v, Some(Expression::int_lit(0)));
        assert_eq!(s.assigned_vars(), vec!["x"]);
        let display = s.to_string();
        assert!(display.contains("int x = 0"));
    }

    #[test]
    fn test_statement_block() {
        let s = Statement::block(vec![Statement::assign("x", Expression::int_lit(1))]);
        let display = s.to_string();
        assert!(display.contains("{"));
        assert!(display.contains("}"));
    }

    #[test]
    fn test_statement_flatten() {
        let inner = Statement::sequence(vec![
            Statement::assign("x", Expression::int_lit(1)),
            Statement::assign("y", Expression::int_lit(2)),
        ]);
        let outer =
            Statement::sequence(vec![inner, Statement::assign("z", Expression::int_lit(3))]);
        let flat = outer.flatten();
        match flat {
            Statement::Sequence(stmts) => assert_eq!(stmts.len(), 3),
            _ => panic!("expected Sequence"),
        }
    }

    #[test]
    fn test_statement_with_span() {
        use crate::errors::SourceLocation;
        let span = SpanInfo::point(SourceLocation::new("a.ms", 1, 1));
        let s = Statement::assign("x", Expression::int_lit(1)).with_span(span.clone());
        assert_eq!(s.span(), Some(&span));
    }

    #[test]
    fn test_statement_is_empty() {
        assert!(Statement::Sequence(vec![]).is_empty());
        assert!(!Statement::assign("x", Expression::int_lit(1)).is_empty());
    }

    // -- Function tests --

    #[test]
    fn test_function_basic() {
        let f = Function::new(
            "max",
            vec![
                Variable::param("a", QfLiaType::Int),
                Variable::param("b", QfLiaType::Int),
            ],
            QfLiaType::Int,
            Statement::ret(Some(Expression::var("a"))),
        );
        assert_eq!(f.arity(), 2);
        assert_eq!(f.param_names(), vec!["a", "b"]);
        let sig = f.signature();
        assert_eq!(sig.arity(), 2);
    }

    #[test]
    fn test_function_display() {
        let f = Function::new(
            "id",
            vec![Variable::param("x", QfLiaType::Int)],
            QfLiaType::Int,
            Statement::ret(Some(Expression::var("x"))),
        );
        let s = f.to_string();
        assert!(s.contains("int id("));
        assert!(s.contains("return x"));
    }

    // -- Program tests --

    #[test]
    fn test_program_basic() {
        let f = Function::new("f", vec![], QfLiaType::Void, Statement::ret(None));
        let p = Program::new(vec![f]).with_name("test");
        assert_eq!(p.function_count(), 1);
        assert!(p.function("f").is_some());
        assert!(p.function("g").is_none());
        assert_eq!(p.function_names(), vec!["f"]);
    }

    #[test]
    fn test_program_display() {
        let f = Function::new("main", vec![], QfLiaType::Void, Statement::ret(None));
        let p = Program::new(vec![f]).with_name("demo");
        let s = p.to_string();
        assert!(s.contains("demo"));
        assert!(s.contains("main"));
    }

    #[test]
    fn test_program_total_statements() {
        let f1 = Function::new(
            "f1",
            vec![],
            QfLiaType::Void,
            Statement::sequence(vec![
                Statement::assign("x", Expression::int_lit(1)),
                Statement::ret(None),
            ]),
        );
        let f2 = Function::new("f2", vec![], QfLiaType::Void, Statement::ret(None));
        let p = Program::new(vec![f1, f2]);
        assert_eq!(p.total_statements(), 3);
    }

    // -- Visitor tests --

    #[test]
    fn test_counting_visitor() {
        let body = Statement::sequence(vec![
            Statement::assign(
                "x",
                Expression::add(Expression::var("a"), Expression::int_lit(1)),
            ),
            Statement::ret(Some(Expression::var("x"))),
        ]);
        let f = Function::new(
            "f",
            vec![Variable::param("a", QfLiaType::Int)],
            QfLiaType::Int,
            body,
        );
        let p = Program::new(vec![f]);
        let mut counter = CountingVisitor::new();
        walk_program(&mut counter, &p);
        assert!(counter.expr_count > 0);
        assert!(counter.stmt_count > 0);
    }

    #[test]
    fn test_var_collector() {
        let body = Statement::assign(
            "r",
            Expression::add(Expression::var("a"), Expression::var("b")),
        );
        let f = Function::new(
            "f",
            vec![
                Variable::param("a", QfLiaType::Int),
                Variable::param("b", QfLiaType::Int),
            ],
            QfLiaType::Int,
            body,
        );
        let p = Program::new(vec![f]);
        let mut collector = VarCollector::new();
        walk_program(&mut collector, &p);
        let vars = collector.unique_vars();
        assert!(vars.contains(&"a".to_string()));
        assert!(vars.contains(&"b".to_string()));
    }

    // -- ExprBuilder tests --

    #[test]
    fn test_expr_builder() {
        let e = ExprBuilder::add(ExprBuilder::var("x"), ExprBuilder::int(1));
        assert_eq!(e.size(), 3);
        let cond = ExprBuilder::cond(
            ExprBuilder::gt(ExprBuilder::var("a"), ExprBuilder::var("b")),
            ExprBuilder::var("a"),
            ExprBuilder::var("b"),
        );
        assert!(cond.size() > 3);
    }

    #[test]
    fn test_expression_serialization() {
        let e = Expression::add(Expression::var("x"), Expression::int_lit(1));
        let json = serde_json::to_string(&e).unwrap();
        let e2: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(e, e2);
    }

    #[test]
    fn test_statement_serialization() {
        let s = Statement::assign("x", Expression::int_lit(42));
        let json = serde_json::to_string(&s).unwrap();
        let s2: Statement = serde_json::from_str(&json).unwrap();
        assert_eq!(s, s2);
    }

    #[test]
    fn test_arith_op_display() {
        assert_eq!(ArithOp::Add.to_string(), "+");
        assert_eq!(ArithOp::Mod.to_string(), "%");
    }

    #[test]
    fn test_rel_op_display() {
        assert_eq!(RelOp::Lt.to_string(), "<");
        assert_eq!(RelOp::Ne.to_string(), "!=");
    }

    #[test]
    fn test_logic_op_display() {
        assert_eq!(LogicOp::And.to_string(), "&&");
        assert_eq!(LogicOp::Or.to_string(), "||");
    }

    #[test]
    fn test_unary_arith_display() {
        let e = Expression::negate(Expression::var("x"));
        assert_eq!(e.to_string(), "(-x)");
    }

    #[test]
    fn test_return_void_display() {
        let s = Statement::ret(None);
        assert_eq!(s.to_string(), "return;");
    }
}

//! Compatibility type definitions for dependency crates.
//!
//! These types mirror the public APIs of `shared-types`, `mutation-core`,
//! `smt-solver`, `contract-synth`, `coverage`, and `program-analysis`.
//! They will be replaced by re-exports once those crates are implemented.

use std::collections::HashMap;
use std::fmt;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ─── shared-types ────────────────────────────────────────────────────────────

/// Quantifier-free linear integer arithmetic type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QfLiaType {
    Int,
    Bool,
    Array(Box<QfLiaType>, Box<QfLiaType>),
}

impl QfLiaType {
    pub fn is_int(&self) -> bool {
        matches!(self, QfLiaType::Int)
    }
    pub fn is_bool(&self) -> bool {
        matches!(self, QfLiaType::Bool)
    }
    pub fn is_array(&self) -> bool {
        matches!(self, QfLiaType::Array(..))
    }
}

impl fmt::Display for QfLiaType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QfLiaType::Int => write!(f, "Int"),
            QfLiaType::Bool => write!(f, "Bool"),
            QfLiaType::Array(idx, elem) => write!(f, "Array<{}, {}>", idx, elem),
        }
    }
}

/// Runtime value.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Value {
    Integer(i64),
    Boolean(bool),
}

impl Value {
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            Value::Integer(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_boolean(&self) -> Option<bool> {
        match self {
            Value::Boolean(v) => Some(*v),
            _ => None,
        }
    }
    pub fn ty(&self) -> QfLiaType {
        match self {
            Value::Integer(_) => QfLiaType::Int,
            Value::Boolean(_) => QfLiaType::Bool,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Integer(v) => write!(f, "{}", v),
            Value::Boolean(v) => write!(f, "{}", v),
        }
    }
}

/// A typed variable binding.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Variable {
    pub name: String,
    pub ty: QfLiaType,
}

impl Variable {
    pub fn new(name: impl Into<String>, ty: QfLiaType) -> Self {
        Self { name: name.into(), ty }
    }
    pub fn int(name: impl Into<String>) -> Self {
        Self::new(name, QfLiaType::Int)
    }
    pub fn bool(name: impl Into<String>) -> Self {
        Self::new(name, QfLiaType::Bool)
    }
}

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.name, self.ty)
    }
}

/// Function signature.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionSignature {
    pub name: String,
    pub params: Vec<Variable>,
    pub return_type: QfLiaType,
}

impl FunctionSignature {
    pub fn new(
        name: impl Into<String>,
        params: Vec<Variable>,
        return_type: QfLiaType,
    ) -> Self {
        Self {
            name: name.into(),
            params,
            return_type,
        }
    }
    pub fn arity(&self) -> usize {
        self.params.len()
    }
}

impl fmt::Display for FunctionSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let params: Vec<String> = self.params.iter().map(|p| p.to_string()).collect();
        write!(f, "{}({}) -> {}", self.name, params.join(", "), self.return_type)
    }
}

/// Source location in the original program.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SourceLocation {
    pub file: String,
    pub line: u32,
    pub column: u32,
    pub end_line: Option<u32>,
    pub end_column: Option<u32>,
}

impl SourceLocation {
    pub fn new(file: impl Into<String>, line: u32, column: u32) -> Self {
        Self {
            file: file.into(),
            line,
            column,
            end_line: None,
            end_column: None,
        }
    }

    pub fn with_end(mut self, end_line: u32, end_column: u32) -> Self {
        self.end_line = Some(end_line);
        self.end_column = Some(end_column);
        self
    }
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.file, self.line, self.column)
    }
}

// ─── Formula / Contract types ────────────────────────────────────────────────

/// Arithmetic operations.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

impl fmt::Display for ArithOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArithOp::Add => write!(f, "+"),
            ArithOp::Sub => write!(f, "-"),
            ArithOp::Mul => write!(f, "*"),
            ArithOp::Div => write!(f, "/"),
            ArithOp::Mod => write!(f, "%"),
        }
    }
}

/// Relational operations.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl fmt::Display for RelOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RelOp::Eq => write!(f, "=="),
            RelOp::Ne => write!(f, "!="),
            RelOp::Lt => write!(f, "<"),
            RelOp::Le => write!(f, "<="),
            RelOp::Gt => write!(f, ">"),
            RelOp::Ge => write!(f, ">="),
        }
    }
}

/// Logical operations.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LogicOp {
    And,
    Or,
    Not,
    Implies,
}

/// A formula in QF_LIA (quantifier-free linear integer arithmetic).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Formula {
    /// Boolean literal.
    BoolLit(bool),
    /// Integer literal.
    IntLit(i64),
    /// Variable reference.
    Var(String),
    /// Arithmetic operation.
    Arith(ArithOp, Box<Formula>, Box<Formula>),
    /// Relational comparison.
    Rel(RelOp, Box<Formula>, Box<Formula>),
    /// Logical AND of sub-formulas.
    And(Vec<Formula>),
    /// Logical OR of sub-formulas.
    Or(Vec<Formula>),
    /// Logical negation.
    Not(Box<Formula>),
    /// Implication: lhs => rhs.
    Implies(Box<Formula>, Box<Formula>),
    /// If-then-else.
    Ite(Box<Formula>, Box<Formula>, Box<Formula>),
    /// Let binding: let name = value in body.
    Let(String, Box<Formula>, Box<Formula>),
    /// Forall quantifier (for completeness, though QF_LIA is quantifier-free
    /// the solver may need it for bounded model checking).
    Forall(Vec<Variable>, Box<Formula>),
    /// Array select: a[i].
    Select(Box<Formula>, Box<Formula>),
    /// Array store: a{i := v}.
    Store(Box<Formula>, Box<Formula>, Box<Formula>),
}

impl Formula {
    pub fn bool_lit(v: bool) -> Self {
        Formula::BoolLit(v)
    }
    pub fn int_lit(v: i64) -> Self {
        Formula::IntLit(v)
    }
    pub fn var(name: impl Into<String>) -> Self {
        Formula::Var(name.into())
    }
    pub fn and(conjuncts: Vec<Formula>) -> Self {
        if conjuncts.len() == 1 {
            conjuncts.into_iter().next().unwrap()
        } else {
            Formula::And(conjuncts)
        }
    }
    pub fn or(disjuncts: Vec<Formula>) -> Self {
        if disjuncts.len() == 1 {
            disjuncts.into_iter().next().unwrap()
        } else {
            Formula::Or(disjuncts)
        }
    }
    pub fn not(inner: Formula) -> Self {
        Formula::Not(Box::new(inner))
    }
    pub fn implies(lhs: Formula, rhs: Formula) -> Self {
        Formula::Implies(Box::new(lhs), Box::new(rhs))
    }
    pub fn eq(lhs: Formula, rhs: Formula) -> Self {
        Formula::Rel(RelOp::Eq, Box::new(lhs), Box::new(rhs))
    }
    pub fn ne(lhs: Formula, rhs: Formula) -> Self {
        Formula::Rel(RelOp::Ne, Box::new(lhs), Box::new(rhs))
    }
    pub fn lt(lhs: Formula, rhs: Formula) -> Self {
        Formula::Rel(RelOp::Lt, Box::new(lhs), Box::new(rhs))
    }
    pub fn le(lhs: Formula, rhs: Formula) -> Self {
        Formula::Rel(RelOp::Le, Box::new(lhs), Box::new(rhs))
    }
    pub fn gt(lhs: Formula, rhs: Formula) -> Self {
        Formula::Rel(RelOp::Gt, Box::new(lhs), Box::new(rhs))
    }
    pub fn ge(lhs: Formula, rhs: Formula) -> Self {
        Formula::Rel(RelOp::Ge, Box::new(lhs), Box::new(rhs))
    }
    pub fn add(lhs: Formula, rhs: Formula) -> Self {
        Formula::Arith(ArithOp::Add, Box::new(lhs), Box::new(rhs))
    }
    pub fn sub(lhs: Formula, rhs: Formula) -> Self {
        Formula::Arith(ArithOp::Sub, Box::new(lhs), Box::new(rhs))
    }
    pub fn mul(lhs: Formula, rhs: Formula) -> Self {
        Formula::Arith(ArithOp::Mul, Box::new(lhs), Box::new(rhs))
    }

    /// Collect free variable names in the formula.
    pub fn free_vars(&self) -> Vec<String> {
        let mut vars = Vec::new();
        self.collect_vars(&mut vars);
        vars.sort();
        vars.dedup();
        vars
    }

    fn collect_vars(&self, acc: &mut Vec<String>) {
        match self {
            Formula::Var(name) => acc.push(name.clone()),
            Formula::Arith(_, l, r) | Formula::Rel(_, l, r) | Formula::Implies(l, r) => {
                l.collect_vars(acc);
                r.collect_vars(acc);
            }
            Formula::And(fs) | Formula::Or(fs) => {
                for f in fs {
                    f.collect_vars(acc);
                }
            }
            Formula::Not(inner) => inner.collect_vars(acc),
            Formula::Ite(c, t, e) | Formula::Store(c, t, e) => {
                c.collect_vars(acc);
                t.collect_vars(acc);
                e.collect_vars(acc);
            }
            Formula::Select(a, i) => {
                a.collect_vars(acc);
                i.collect_vars(acc);
            }
            Formula::Let(_, val, body) => {
                val.collect_vars(acc);
                body.collect_vars(acc);
            }
            Formula::Forall(_, body) => body.collect_vars(acc),
            Formula::BoolLit(_) | Formula::IntLit(_) => {}
        }
    }

    /// Substitute a variable name with another formula.
    pub fn substitute(&self, var: &str, replacement: &Formula) -> Formula {
        match self {
            Formula::Var(name) if name == var => replacement.clone(),
            Formula::Var(_) | Formula::BoolLit(_) | Formula::IntLit(_) => self.clone(),
            Formula::Arith(op, l, r) => Formula::Arith(
                op.clone(),
                Box::new(l.substitute(var, replacement)),
                Box::new(r.substitute(var, replacement)),
            ),
            Formula::Rel(op, l, r) => Formula::Rel(
                op.clone(),
                Box::new(l.substitute(var, replacement)),
                Box::new(r.substitute(var, replacement)),
            ),
            Formula::And(fs) => Formula::And(
                fs.iter().map(|f| f.substitute(var, replacement)).collect(),
            ),
            Formula::Or(fs) => Formula::Or(
                fs.iter().map(|f| f.substitute(var, replacement)).collect(),
            ),
            Formula::Not(inner) => Formula::Not(Box::new(inner.substitute(var, replacement))),
            Formula::Implies(l, r) => Formula::Implies(
                Box::new(l.substitute(var, replacement)),
                Box::new(r.substitute(var, replacement)),
            ),
            Formula::Ite(c, t, e) => Formula::Ite(
                Box::new(c.substitute(var, replacement)),
                Box::new(t.substitute(var, replacement)),
                Box::new(e.substitute(var, replacement)),
            ),
            Formula::Let(name, val, body) => {
                let new_val = Box::new(val.substitute(var, replacement));
                if name == var {
                    Formula::Let(name.clone(), new_val, body.clone())
                } else {
                    Formula::Let(
                        name.clone(),
                        new_val,
                        Box::new(body.substitute(var, replacement)),
                    )
                }
            }
            Formula::Forall(vars, body) => {
                if vars.iter().any(|v| v.name == var) {
                    self.clone()
                } else {
                    Formula::Forall(vars.clone(), Box::new(body.substitute(var, replacement)))
                }
            }
            Formula::Select(a, i) => Formula::Select(
                Box::new(a.substitute(var, replacement)),
                Box::new(i.substitute(var, replacement)),
            ),
            Formula::Store(a, i, v) => Formula::Store(
                Box::new(a.substitute(var, replacement)),
                Box::new(i.substitute(var, replacement)),
                Box::new(v.substitute(var, replacement)),
            ),
        }
    }

    /// Evaluate the formula in a given variable assignment.
    pub fn evaluate(&self, env: &HashMap<String, Value>) -> Option<Value> {
        match self {
            Formula::BoolLit(b) => Some(Value::Boolean(*b)),
            Formula::IntLit(i) => Some(Value::Integer(*i)),
            Formula::Var(name) => env.get(name).cloned(),
            Formula::Arith(op, l, r) => {
                let lv = l.evaluate(env)?.as_integer()?;
                let rv = r.evaluate(env)?.as_integer()?;
                let result = match op {
                    ArithOp::Add => lv.checked_add(rv)?,
                    ArithOp::Sub => lv.checked_sub(rv)?,
                    ArithOp::Mul => lv.checked_mul(rv)?,
                    ArithOp::Div => {
                        if rv == 0 { return None; }
                        lv.checked_div(rv)?
                    }
                    ArithOp::Mod => {
                        if rv == 0 { return None; }
                        lv.checked_rem(rv)?
                    }
                };
                Some(Value::Integer(result))
            }
            Formula::Rel(op, l, r) => {
                let lv = l.evaluate(env)?.as_integer()?;
                let rv = r.evaluate(env)?.as_integer()?;
                let result = match op {
                    RelOp::Eq => lv == rv,
                    RelOp::Ne => lv != rv,
                    RelOp::Lt => lv < rv,
                    RelOp::Le => lv <= rv,
                    RelOp::Gt => lv > rv,
                    RelOp::Ge => lv >= rv,
                };
                Some(Value::Boolean(result))
            }
            Formula::And(fs) => {
                for f in fs {
                    if !f.evaluate(env)?.as_boolean()? {
                        return Some(Value::Boolean(false));
                    }
                }
                Some(Value::Boolean(true))
            }
            Formula::Or(fs) => {
                for f in fs {
                    if f.evaluate(env)?.as_boolean()? {
                        return Some(Value::Boolean(true));
                    }
                }
                Some(Value::Boolean(false))
            }
            Formula::Not(inner) => {
                let v = inner.evaluate(env)?.as_boolean()?;
                Some(Value::Boolean(!v))
            }
            Formula::Implies(l, r) => {
                let lv = l.evaluate(env)?.as_boolean()?;
                let rv = r.evaluate(env)?.as_boolean()?;
                Some(Value::Boolean(!lv || rv))
            }
            Formula::Ite(c, t, e) => {
                if c.evaluate(env)?.as_boolean()? {
                    t.evaluate(env)
                } else {
                    e.evaluate(env)
                }
            }
            Formula::Let(name, val, body) => {
                let v = val.evaluate(env)?;
                let mut new_env = env.clone();
                new_env.insert(name.clone(), v);
                body.evaluate(&new_env)
            }
            _ => None,
        }
    }

    /// Count the number of AST nodes (for complexity estimation).
    pub fn size(&self) -> usize {
        match self {
            Formula::BoolLit(_) | Formula::IntLit(_) | Formula::Var(_) => 1,
            Formula::Arith(_, l, r)
            | Formula::Rel(_, l, r)
            | Formula::Implies(l, r)
            | Formula::Select(l, r) => 1 + l.size() + r.size(),
            Formula::And(fs) | Formula::Or(fs) => 1 + fs.iter().map(|f| f.size()).sum::<usize>(),
            Formula::Not(inner) => 1 + inner.size(),
            Formula::Ite(c, t, e) | Formula::Store(c, t, e) => {
                1 + c.size() + t.size() + e.size()
            }
            Formula::Let(_, v, b) => 1 + v.size() + b.size(),
            Formula::Forall(_, body) => 1 + body.size(),
        }
    }
}

impl fmt::Display for Formula {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Formula::BoolLit(b) => write!(f, "{}", b),
            Formula::IntLit(i) => write!(f, "{}", i),
            Formula::Var(name) => write!(f, "{}", name),
            Formula::Arith(op, l, r) => write!(f, "({} {} {})", l, op, r),
            Formula::Rel(op, l, r) => write!(f, "({} {} {})", l, op, r),
            Formula::And(fs) => {
                let parts: Vec<String> = fs.iter().map(|x| x.to_string()).collect();
                write!(f, "({})", parts.join(" ∧ "))
            }
            Formula::Or(fs) => {
                let parts: Vec<String> = fs.iter().map(|x| x.to_string()).collect();
                write!(f, "({})", parts.join(" ∨ "))
            }
            Formula::Not(inner) => write!(f, "¬{}", inner),
            Formula::Implies(l, r) => write!(f, "({} ⇒ {})", l, r),
            Formula::Ite(c, t, e) => write!(f, "(if {} then {} else {})", c, t, e),
            Formula::Let(name, val, body) => write!(f, "(let {} = {} in {})", name, val, body),
            Formula::Forall(vars, body) => {
                let vs: Vec<String> = vars.iter().map(|v| v.to_string()).collect();
                write!(f, "(∀ {}. {})", vs.join(", "), body)
            }
            Formula::Select(a, i) => write!(f, "{}[{}]", a, i),
            Formula::Store(a, i, v) => write!(f, "{}{{{}:={}}}", a, i, v),
        }
    }
}

/// Strength tier for a synthesized contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ContractStrength {
    Trivial,
    Weak,
    Medium,
    Strong,
    Exact,
}

impl fmt::Display for ContractStrength {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContractStrength::Trivial => write!(f, "trivial"),
            ContractStrength::Weak => write!(f, "weak"),
            ContractStrength::Medium => write!(f, "medium"),
            ContractStrength::Strong => write!(f, "strong"),
            ContractStrength::Exact => write!(f, "exact"),
        }
    }
}

/// Where a contract clause came from.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContractProvenance {
    /// Synthesized by CEGIS from mutation analysis.
    CegisSynthesis,
    /// Inferred from test observations.
    TestObservation,
    /// Provided by the user.
    UserAnnotation,
    /// Derived by static analysis / weakest precondition.
    StaticAnalysis,
    /// Strengthened from an existing clause.
    Strengthening(Box<ContractProvenance>),
}

/// A single clause in a contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContractClause {
    pub id: String,
    pub formula: Formula,
    pub strength: ContractStrength,
    pub provenance: ContractProvenance,
    pub description: Option<String>,
}

impl ContractClause {
    pub fn new(
        formula: Formula,
        strength: ContractStrength,
        provenance: ContractProvenance,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            formula,
            strength,
            provenance,
            description: None,
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }
}

/// A full contract (pre/post-condition pair) for a function.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Contract {
    pub function_name: String,
    pub preconditions: Vec<ContractClause>,
    pub postconditions: Vec<ContractClause>,
    pub invariants: Vec<ContractClause>,
    pub overall_strength: ContractStrength,
}

impl Contract {
    pub fn new(function_name: impl Into<String>) -> Self {
        Self {
            function_name: function_name.into(),
            preconditions: Vec::new(),
            postconditions: Vec::new(),
            invariants: Vec::new(),
            overall_strength: ContractStrength::Trivial,
        }
    }

    pub fn with_precondition(mut self, clause: ContractClause) -> Self {
        self.preconditions.push(clause);
        self.recompute_strength();
        self
    }

    pub fn with_postcondition(mut self, clause: ContractClause) -> Self {
        self.postconditions.push(clause);
        self.recompute_strength();
        self
    }

    pub fn with_invariant(mut self, clause: ContractClause) -> Self {
        self.invariants.push(clause);
        self.recompute_strength();
        self
    }

    pub fn add_precondition(&mut self, clause: ContractClause) {
        self.preconditions.push(clause);
        self.recompute_strength();
    }

    pub fn add_postcondition(&mut self, clause: ContractClause) {
        self.postconditions.push(clause);
        self.recompute_strength();
    }

    pub fn all_clauses(&self) -> Vec<&ContractClause> {
        self.preconditions
            .iter()
            .chain(self.postconditions.iter())
            .chain(self.invariants.iter())
            .collect()
    }

    pub fn clause_count(&self) -> usize {
        self.preconditions.len() + self.postconditions.len() + self.invariants.len()
    }

    pub fn is_empty(&self) -> bool {
        self.clause_count() == 0
    }

    /// Combine the contract into a single formula: pre ⇒ post ∧ inv.
    pub fn as_formula(&self) -> Formula {
        let pre = if self.preconditions.is_empty() {
            Formula::BoolLit(true)
        } else {
            Formula::And(self.preconditions.iter().map(|c| c.formula.clone()).collect())
        };

        let mut post_parts: Vec<Formula> = self
            .postconditions
            .iter()
            .map(|c| c.formula.clone())
            .collect();
        post_parts.extend(self.invariants.iter().map(|c| c.formula.clone()));

        let post = if post_parts.is_empty() {
            Formula::BoolLit(true)
        } else {
            Formula::And(post_parts)
        };

        Formula::Implies(Box::new(pre), Box::new(post))
    }

    fn recompute_strength(&mut self) {
        let all: Vec<ContractStrength> = self
            .all_clauses()
            .into_iter()
            .map(|c| c.strength)
            .collect();
        self.overall_strength = if all.is_empty() {
            ContractStrength::Trivial
        } else {
            *all.iter().min().unwrap_or(&ContractStrength::Trivial)
        };
    }
}

// ─── mutation-core types ─────────────────────────────────────────────────────

/// Unique identifier for a mutant.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MutantId(pub String);

impl MutantId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
    pub fn generate() -> Self {
        Self(Uuid::new_v4().to_string())
    }
}

impl fmt::Display for MutantId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The kind of mutation operator applied.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MutationOperator {
    ArithmeticReplacement { from: ArithOp, to: ArithOp },
    RelationalReplacement { from: RelOp, to: RelOp },
    LogicalReplacement { from: LogicOp, to: LogicOp },
    ConstantReplacement { from: i64, to: i64 },
    VariableReplacement { from: String, to: String },
    StatementDeletion,
    ConditionalNegation,
    ReturnValueMutation,
    BoundaryShift { direction: i64 },
    UnaryInsertion,
    NullCheck,
}

impl fmt::Display for MutationOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MutationOperator::ArithmeticReplacement { from, to } => {
                write!(f, "AOR({} → {})", from, to)
            }
            MutationOperator::RelationalReplacement { from, to } => {
                write!(f, "ROR({} → {})", from, to)
            }
            MutationOperator::LogicalReplacement { .. } => write!(f, "LOR"),
            MutationOperator::ConstantReplacement { from, to } => {
                write!(f, "COR({} → {})", from, to)
            }
            MutationOperator::VariableReplacement { from, to } => {
                write!(f, "VOR({} → {})", from, to)
            }
            MutationOperator::StatementDeletion => write!(f, "SDL"),
            MutationOperator::ConditionalNegation => write!(f, "CND"),
            MutationOperator::ReturnValueMutation => write!(f, "RVM"),
            MutationOperator::BoundaryShift { direction } => {
                write!(f, "BSH({})", direction)
            }
            MutationOperator::UnaryInsertion => write!(f, "UOI"),
            MutationOperator::NullCheck => write!(f, "NCK"),
        }
    }
}

/// Current status of a mutant in the testing pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MutantStatus {
    Alive,
    Killed,
    Equivalent,
    Timeout,
    Error,
    Skipped,
}

impl fmt::Display for MutantStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MutantStatus::Alive => write!(f, "alive"),
            MutantStatus::Killed => write!(f, "killed"),
            MutantStatus::Equivalent => write!(f, "equivalent"),
            MutantStatus::Timeout => write!(f, "timeout"),
            MutantStatus::Error => write!(f, "error"),
            MutantStatus::Skipped => write!(f, "skipped"),
        }
    }
}

/// Descriptor for a single mutant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MutantDescriptor {
    pub id: MutantId,
    pub function_name: String,
    pub operator: MutationOperator,
    pub location: SourceLocation,
    pub original_code: String,
    pub mutated_code: String,
    pub status: MutantStatus,
    /// The semantic delta: original_expr - mutant_expr (as a formula).
    pub semantic_delta: Option<Formula>,
}

impl MutantDescriptor {
    pub fn new(
        function_name: impl Into<String>,
        operator: MutationOperator,
        location: SourceLocation,
        original_code: impl Into<String>,
        mutated_code: impl Into<String>,
    ) -> Self {
        Self {
            id: MutantId::generate(),
            function_name: function_name.into(),
            operator,
            location,
            original_code: original_code.into(),
            mutated_code: mutated_code.into(),
            status: MutantStatus::Alive,
            semantic_delta: None,
        }
    }

    pub fn with_id(mut self, id: MutantId) -> Self {
        self.id = id;
        self
    }

    pub fn with_status(mut self, status: MutantStatus) -> Self {
        self.status = status;
        self
    }

    pub fn with_semantic_delta(mut self, delta: Formula) -> Self {
        self.semantic_delta = Some(delta);
        self
    }

    pub fn is_alive(&self) -> bool {
        self.status == MutantStatus::Alive
    }

    pub fn is_killed(&self) -> bool {
        self.status == MutantStatus::Killed
    }
}

/// Unique identifier for a test case.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TestId(pub String);

impl TestId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl fmt::Display for TestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Kill matrix: records which tests kill which mutants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KillMatrix {
    pub entries: IndexMap<MutantId, Vec<TestId>>,
    pub test_count: usize,
    pub mutant_count: usize,
}

impl KillMatrix {
    pub fn new() -> Self {
        Self {
            entries: IndexMap::new(),
            test_count: 0,
            mutant_count: 0,
        }
    }

    pub fn record_kill(&mut self, mutant: MutantId, test: TestId) {
        self.entries
            .entry(mutant)
            .or_insert_with(Vec::new)
            .push(test);
    }

    pub fn is_killed(&self, mutant: &MutantId) -> bool {
        self.entries
            .get(mutant)
            .map(|tests| !tests.is_empty())
            .unwrap_or(false)
    }

    pub fn killing_tests(&self, mutant: &MutantId) -> &[TestId] {
        self.entries
            .get(mutant)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    pub fn survivors(&self) -> Vec<&MutantId> {
        self.entries
            .keys()
            .filter(|m| !self.is_killed(m))
            .collect()
    }

    pub fn mutation_score(&self) -> f64 {
        if self.mutant_count == 0 {
            return 1.0;
        }
        let killed = self.entries.values().filter(|tests| !tests.is_empty()).count();
        killed as f64 / self.mutant_count as f64
    }
}

impl Default for KillMatrix {
    fn default() -> Self {
        Self::new()
    }
}

// ─── smt-solver types ────────────────────────────────────────────────────────

/// Result of an SMT satisfiability check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SmtResult {
    Sat(SmtModel),
    Unsat,
    Unknown(String),
    Timeout,
    Error(String),
}

impl SmtResult {
    pub fn is_sat(&self) -> bool {
        matches!(self, SmtResult::Sat(_))
    }
    pub fn is_unsat(&self) -> bool {
        matches!(self, SmtResult::Unsat)
    }
    pub fn model(&self) -> Option<&SmtModel> {
        match self {
            SmtResult::Sat(model) => Some(model),
            _ => None,
        }
    }
}

/// A model (satisfying assignment) from the SMT solver.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SmtModel {
    pub assignments: IndexMap<String, Value>,
}

impl SmtModel {
    pub fn new() -> Self {
        Self {
            assignments: IndexMap::new(),
        }
    }

    pub fn with_assignment(mut self, name: impl Into<String>, value: Value) -> Self {
        self.assignments.insert(name.into(), value);
        self
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        self.assignments.get(name)
    }

    pub fn as_hashmap(&self) -> HashMap<String, Value> {
        self.assignments.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &Value)> {
        self.assignments.iter()
    }

    pub fn len(&self) -> usize {
        self.assignments.len()
    }

    pub fn is_empty(&self) -> bool {
        self.assignments.is_empty()
    }
}

impl Default for SmtModel {
    fn default() -> Self {
        Self::new()
    }
}

/// An SMT solver interface.
///
/// In the full system this wraps CVC5; here we provide a mock implementation
/// that can be used for testing.
#[derive(Debug, Clone)]
pub struct SmtSolver {
    pub timeout_ms: u64,
    pub logic: String,
}

impl SmtSolver {
    pub fn new() -> Self {
        Self {
            timeout_ms: 5000,
            logic: "QF_LIA".to_string(),
        }
    }

    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Check satisfiability of a formula.
    ///
    /// In the real implementation this encodes the formula to SMT-LIB2 and
    /// invokes the solver. Here we provide a deterministic mock that handles
    /// common formula patterns used in testing.
    pub fn check_sat(&self, formula: &Formula) -> SmtResult {
        match formula {
            Formula::BoolLit(true) => SmtResult::Sat(SmtModel::new()),
            Formula::BoolLit(false) => SmtResult::Unsat,
            Formula::And(conjuncts) => {
                if conjuncts.iter().any(|f| matches!(f, Formula::BoolLit(false))) {
                    return SmtResult::Unsat;
                }
                let vars = formula.free_vars();
                let mut model = SmtModel::new();
                for v in &vars {
                    model.assignments.insert(v.clone(), Value::Integer(0));
                }
                self.try_eval_model(formula, &mut model);
                SmtResult::Sat(model)
            }
            _ => {
                let vars = formula.free_vars();
                let mut model = SmtModel::new();
                for v in &vars {
                    model.assignments.insert(v.clone(), Value::Integer(0));
                }
                self.try_eval_model(formula, &mut model);
                SmtResult::Sat(model)
            }
        }
    }

    /// Check if two formulas are equivalent (their negated XOR is unsatisfiable).
    pub fn check_equivalence(&self, f1: &Formula, f2: &Formula) -> SmtResult {
        let diff = Formula::Not(Box::new(Formula::Rel(
            RelOp::Eq,
            Box::new(f1.clone()),
            Box::new(f2.clone()),
        )));
        self.check_sat(&diff)
    }

    /// Check if a formula implies another.
    pub fn check_implication(&self, premise: &Formula, conclusion: &Formula) -> SmtResult {
        let negated = Formula::And(vec![
            premise.clone(),
            Formula::Not(Box::new(conclusion.clone())),
        ]);
        self.check_sat(&negated)
    }

    fn try_eval_model(&self, formula: &Formula, model: &mut SmtModel) {
        let env = model.as_hashmap();
        if let Some(Value::Boolean(false)) = formula.evaluate(&env) {
            for (_, val) in model.assignments.iter_mut() {
                if let Value::Integer(ref mut v) = val {
                    *v = 1;
                }
            }
        }
    }
}

impl Default for SmtSolver {
    fn default() -> Self {
        Self::new()
    }
}

// ─── contract-synth types ────────────────────────────────────────────────────

/// Result of contract synthesis for a function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisResult {
    pub function_name: String,
    pub contract: Contract,
    pub killed_mutants: Vec<MutantId>,
    pub surviving_mutants: Vec<MutantId>,
    pub synthesis_time_ms: u64,
    pub iterations: usize,
}

/// Tier-based synthesis configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SynthesisTier {
    /// Tier 1: observation-based (fast, weak).
    Observation,
    /// Tier 2: lattice walk (medium speed, medium strength).
    LatticeWalk,
    /// Tier 3: CEGIS (slow, strong).
    Cegis,
}

// ─── coverage types ──────────────────────────────────────────────────────────

/// Whether a mutant is semantically equivalent to the original.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EquivalenceResult {
    Equivalent,
    NonEquivalent { distinguishing_input: SmtModel },
    Unknown,
}

// ─── program-analysis types ──────────────────────────────────────────────────

/// An error predicate describing a bug pattern.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorPredicate {
    pub description: String,
    pub formula: Formula,
    pub severity: ErrorSeverity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ErrorSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorSeverity::Info => write!(f, "info"),
            ErrorSeverity::Warning => write!(f, "warning"),
            ErrorSeverity::Error => write!(f, "error"),
            ErrorSeverity::Critical => write!(f, "critical"),
        }
    }
}

// ─── Shared configuration ────────────────────────────────────────────────────

/// Global MutSpec configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutSpecConfig {
    pub smt_timeout_ms: u64,
    pub max_witnesses_per_mutant: usize,
    pub parallel_jobs: usize,
    pub output_dir: String,
    pub verbosity: u8,
}

impl Default for MutSpecConfig {
    fn default() -> Self {
        Self {
            smt_timeout_ms: 5000,
            max_witnesses_per_mutant: 3,
            parallel_jobs: 4,
            output_dir: "output".to_string(),
            verbosity: 1,
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_formula_display() {
        let f = Formula::and(vec![
            Formula::gt(Formula::var("x"), Formula::int_lit(0)),
            Formula::le(Formula::var("y"), Formula::int_lit(10)),
        ]);
        let s = f.to_string();
        assert!(s.contains("x"));
        assert!(s.contains("y"));
    }

    #[test]
    fn test_formula_free_vars() {
        let f = Formula::and(vec![
            Formula::gt(Formula::var("x"), Formula::int_lit(0)),
            Formula::le(Formula::var("y"), Formula::int_lit(10)),
        ]);
        let vars = f.free_vars();
        assert_eq!(vars, vec!["x", "y"]);
    }

    #[test]
    fn test_formula_substitute() {
        let f = Formula::add(Formula::var("x"), Formula::int_lit(1));
        let result = f.substitute("x", &Formula::int_lit(5));
        assert_eq!(result, Formula::add(Formula::int_lit(5), Formula::int_lit(1)));
    }

    #[test]
    fn test_formula_evaluate() {
        let f = Formula::add(Formula::var("x"), Formula::var("y"));
        let mut env = HashMap::new();
        env.insert("x".to_string(), Value::Integer(3));
        env.insert("y".to_string(), Value::Integer(4));
        assert_eq!(f.evaluate(&env), Some(Value::Integer(7)));
    }

    #[test]
    fn test_formula_evaluate_relational() {
        let f = Formula::gt(Formula::var("x"), Formula::int_lit(0));
        let mut env = HashMap::new();
        env.insert("x".to_string(), Value::Integer(5));
        assert_eq!(f.evaluate(&env), Some(Value::Boolean(true)));
        env.insert("x".to_string(), Value::Integer(-1));
        assert_eq!(f.evaluate(&env), Some(Value::Boolean(false)));
    }

    #[test]
    fn test_formula_size() {
        let f = Formula::and(vec![
            Formula::gt(Formula::var("x"), Formula::int_lit(0)),
            Formula::le(Formula::var("y"), Formula::int_lit(10)),
        ]);
        assert!(f.size() > 4);
    }

    #[test]
    fn test_contract_builder() {
        let c = Contract::new("foo")
            .with_precondition(ContractClause::new(
                Formula::gt(Formula::var("x"), Formula::int_lit(0)),
                ContractStrength::Medium,
                ContractProvenance::CegisSynthesis,
            ))
            .with_postcondition(ContractClause::new(
                Formula::gt(Formula::var("result"), Formula::int_lit(0)),
                ContractStrength::Strong,
                ContractProvenance::CegisSynthesis,
            ));
        assert_eq!(c.clause_count(), 2);
        assert!(!c.is_empty());
    }

    #[test]
    fn test_contract_as_formula() {
        let c = Contract::new("foo")
            .with_precondition(ContractClause::new(
                Formula::gt(Formula::var("x"), Formula::int_lit(0)),
                ContractStrength::Medium,
                ContractProvenance::CegisSynthesis,
            ))
            .with_postcondition(ContractClause::new(
                Formula::gt(Formula::var("result"), Formula::int_lit(0)),
                ContractStrength::Strong,
                ContractProvenance::CegisSynthesis,
            ));
        let f = c.as_formula();
        assert!(matches!(f, Formula::Implies(..)));
    }

    #[test]
    fn test_mutant_descriptor() {
        let m = MutantDescriptor::new(
            "foo",
            MutationOperator::ArithmeticReplacement {
                from: ArithOp::Add,
                to: ArithOp::Sub,
            },
            SourceLocation::new("test.rs", 10, 5),
            "a + b",
            "a - b",
        );
        assert!(m.is_alive());
        assert!(!m.is_killed());
    }

    #[test]
    fn test_kill_matrix() {
        let mut km = KillMatrix::new();
        km.mutant_count = 3;
        let m1 = MutantId::new("m1");
        let m2 = MutantId::new("m2");
        let m3 = MutantId::new("m3");
        km.entries.insert(m1.clone(), vec![]);
        km.entries.insert(m2.clone(), vec![TestId::new("t1")]);
        km.entries.insert(m3.clone(), vec![]);
        km.record_kill(m1.clone(), TestId::new("t2"));
        assert!(km.is_killed(&m1));
        assert!(km.is_killed(&m2));
        assert!(!km.is_killed(&m3));
    }

    #[test]
    fn test_smt_solver_basic() {
        let solver = SmtSolver::new();
        assert!(solver.check_sat(&Formula::BoolLit(true)).is_sat());
        assert!(solver.check_sat(&Formula::BoolLit(false)).is_unsat());
    }

    #[test]
    fn test_smt_model() {
        let model = SmtModel::new()
            .with_assignment("x", Value::Integer(42))
            .with_assignment("y", Value::Boolean(true));
        assert_eq!(model.get("x"), Some(&Value::Integer(42)));
        assert_eq!(model.get("y"), Some(&Value::Boolean(true)));
        assert_eq!(model.len(), 2);
    }

    #[test]
    fn test_source_location() {
        let loc = SourceLocation::new("main.rs", 10, 5).with_end(10, 15);
        assert_eq!(loc.end_line, Some(10));
        assert_eq!(loc.to_string(), "main.rs:10:5");
    }

    #[test]
    fn test_mutation_operator_display() {
        let op = MutationOperator::ArithmeticReplacement {
            from: ArithOp::Add,
            to: ArithOp::Sub,
        };
        assert_eq!(op.to_string(), "AOR(+ → -)");
    }

    #[test]
    fn test_qf_lia_type() {
        assert!(QfLiaType::Int.is_int());
        assert!(QfLiaType::Bool.is_bool());
        assert!(QfLiaType::Array(Box::new(QfLiaType::Int), Box::new(QfLiaType::Int)).is_array());
    }

    #[test]
    fn test_value_ty() {
        assert_eq!(Value::Integer(0).ty(), QfLiaType::Int);
        assert_eq!(Value::Boolean(true).ty(), QfLiaType::Bool);
    }

    #[test]
    fn test_variable_constructors() {
        let v = Variable::int("x");
        assert_eq!(v.name, "x");
        assert_eq!(v.ty, QfLiaType::Int);
        let b = Variable::bool("flag");
        assert_eq!(b.ty, QfLiaType::Bool);
    }

    #[test]
    fn test_contract_strength_ordering() {
        assert!(ContractStrength::Trivial < ContractStrength::Weak);
        assert!(ContractStrength::Weak < ContractStrength::Medium);
        assert!(ContractStrength::Medium < ContractStrength::Strong);
        assert!(ContractStrength::Strong < ContractStrength::Exact);
    }

    #[test]
    fn test_kill_matrix_mutation_score() {
        let mut km = KillMatrix::new();
        km.mutant_count = 4;
        km.entries.insert(MutantId::new("m1"), vec![TestId::new("t1")]);
        km.entries.insert(MutantId::new("m2"), vec![TestId::new("t1")]);
        km.entries.insert(MutantId::new("m3"), vec![]);
        km.entries.insert(MutantId::new("m4"), vec![TestId::new("t2")]);
        assert!((km.mutation_score() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_formula_evaluate_logic() {
        let f = Formula::and(vec![
            Formula::BoolLit(true),
            Formula::BoolLit(true),
        ]);
        let env = HashMap::new();
        assert_eq!(f.evaluate(&env), Some(Value::Boolean(true)));

        let f2 = Formula::and(vec![
            Formula::BoolLit(true),
            Formula::BoolLit(false),
        ]);
        assert_eq!(f2.evaluate(&env), Some(Value::Boolean(false)));
    }

    #[test]
    fn test_formula_evaluate_implies() {
        let env = HashMap::new();
        let f = Formula::implies(Formula::BoolLit(false), Formula::BoolLit(false));
        assert_eq!(f.evaluate(&env), Some(Value::Boolean(true)));
    }

    #[test]
    fn test_formula_evaluate_ite() {
        let env = HashMap::new();
        let f = Formula::Ite(
            Box::new(Formula::BoolLit(true)),
            Box::new(Formula::IntLit(1)),
            Box::new(Formula::IntLit(2)),
        );
        assert_eq!(f.evaluate(&env), Some(Value::Integer(1)));
    }

    #[test]
    fn test_formula_division_by_zero() {
        let f = Formula::Arith(
            ArithOp::Div,
            Box::new(Formula::IntLit(10)),
            Box::new(Formula::IntLit(0)),
        );
        let env = HashMap::new();
        assert_eq!(f.evaluate(&env), None);
    }

    #[test]
    fn test_contract_empty() {
        let c = Contract::new("empty_fn");
        assert!(c.is_empty());
        assert_eq!(c.overall_strength, ContractStrength::Trivial);
    }

    #[test]
    fn test_smt_result_methods() {
        let sat = SmtResult::Sat(SmtModel::new());
        assert!(sat.is_sat());
        assert!(!sat.is_unsat());
        assert!(sat.model().is_some());

        let unsat = SmtResult::Unsat;
        assert!(!unsat.is_sat());
        assert!(unsat.is_unsat());
        assert!(unsat.model().is_none());
    }
}

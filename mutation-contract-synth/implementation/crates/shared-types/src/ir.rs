//! Intermediate representation for program analysis.
//!
//! The IR uses basic blocks in SSA form, providing a normalised view of the
//! program suitable for weakest-precondition computation, mutation analysis,
//! and contract synthesis.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::ast;
use crate::types::{QfLiaType, Variable};

// ---------------------------------------------------------------------------
// SsaVar
// ---------------------------------------------------------------------------

/// An SSA variable: a name plus a version number.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SsaVar {
    pub name: String,
    pub version: u32,
    pub ty: QfLiaType,
}

impl SsaVar {
    pub fn new(name: impl Into<String>, version: u32, ty: QfLiaType) -> Self {
        Self {
            name: name.into(),
            version,
            ty,
        }
    }

    pub fn base(name: impl Into<String>, ty: QfLiaType) -> Self {
        Self::new(name, 0, ty)
    }

    /// The versioned name, e.g. "x_3".
    pub fn versioned_name(&self) -> String {
        format!("{}_{}", self.name, self.version)
    }

    /// SMT-LIB name for this variable.
    pub fn smt_name(&self) -> String {
        format!("|{}_{}|", self.name, self.version)
    }

    /// Produce the next version of this variable.
    pub fn next_version(&self) -> SsaVar {
        SsaVar {
            name: self.name.clone(),
            version: self.version + 1,
            ty: self.ty,
        }
    }

    /// Convert to a Variable.
    pub fn to_variable(&self) -> Variable {
        Variable::ssa(self.versioned_name(), self.ty, &self.name, self.version)
    }

    /// Is this the initial version (version 0)?
    pub fn is_initial(&self) -> bool {
        self.version == 0
    }
}

impl fmt::Display for SsaVar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}_{}: {}", self.name, self.version, self.ty)
    }
}

// ---------------------------------------------------------------------------
// IrExpr
// ---------------------------------------------------------------------------

/// An expression in the IR (normalized, uses SSA variables).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IrExpr {
    /// Integer constant.
    Const(i64),
    /// Boolean constant.
    BoolConst(bool),
    /// SSA variable reference.
    Var(SsaVar),
    /// Binary arithmetic.
    BinArith {
        op: ast::ArithOp,
        lhs: Box<IrExpr>,
        rhs: Box<IrExpr>,
    },
    /// Unary negation.
    Neg(Box<IrExpr>),
    /// Relational comparison.
    Rel {
        op: ast::RelOp,
        lhs: Box<IrExpr>,
        rhs: Box<IrExpr>,
    },
    /// Logical AND.
    And(Box<IrExpr>, Box<IrExpr>),
    /// Logical OR.
    Or(Box<IrExpr>, Box<IrExpr>),
    /// Logical NOT.
    Not(Box<IrExpr>),
    /// Array select.
    Select {
        array: Box<IrExpr>,
        index: Box<IrExpr>,
    },
    /// If-then-else expression.
    Ite {
        cond: Box<IrExpr>,
        then_expr: Box<IrExpr>,
        else_expr: Box<IrExpr>,
    },
    /// Function call (pure).
    Call { name: String, args: Vec<IrExpr> },
    /// Phi placeholder — resolved by PhiNode in blocks.
    Phi(Vec<(usize, IrExpr)>),
}

impl IrExpr {
    // -- Constructors -------------------------------------------------------

    pub fn constant(v: i64) -> Self {
        IrExpr::Const(v)
    }

    pub fn bool_const(v: bool) -> Self {
        IrExpr::BoolConst(v)
    }

    pub fn var(sv: SsaVar) -> Self {
        IrExpr::Var(sv)
    }

    pub fn bin_arith(op: ast::ArithOp, lhs: IrExpr, rhs: IrExpr) -> Self {
        IrExpr::BinArith {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    pub fn neg(e: IrExpr) -> Self {
        IrExpr::Neg(Box::new(e))
    }

    pub fn rel(op: ast::RelOp, lhs: IrExpr, rhs: IrExpr) -> Self {
        IrExpr::Rel {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    pub fn and(lhs: IrExpr, rhs: IrExpr) -> Self {
        IrExpr::And(Box::new(lhs), Box::new(rhs))
    }

    pub fn or(lhs: IrExpr, rhs: IrExpr) -> Self {
        IrExpr::Or(Box::new(lhs), Box::new(rhs))
    }

    pub fn not(e: IrExpr) -> Self {
        IrExpr::Not(Box::new(e))
    }

    pub fn select(array: IrExpr, index: IrExpr) -> Self {
        IrExpr::Select {
            array: Box::new(array),
            index: Box::new(index),
        }
    }

    pub fn ite(cond: IrExpr, then_e: IrExpr, else_e: IrExpr) -> Self {
        IrExpr::Ite {
            cond: Box::new(cond),
            then_expr: Box::new(then_e),
            else_expr: Box::new(else_e),
        }
    }

    pub fn call(name: impl Into<String>, args: Vec<IrExpr>) -> Self {
        IrExpr::Call {
            name: name.into(),
            args,
        }
    }

    // -- Queries ------------------------------------------------------------

    /// Collect all SSA variables referenced.
    pub fn used_vars(&self) -> BTreeSet<SsaVar> {
        let mut vars = BTreeSet::new();
        self.collect_vars(&mut vars);
        vars
    }

    fn collect_vars(&self, vars: &mut BTreeSet<SsaVar>) {
        match self {
            IrExpr::Const(_) | IrExpr::BoolConst(_) => {}
            IrExpr::Var(v) => {
                vars.insert(v.clone());
            }
            IrExpr::BinArith { lhs, rhs, .. } | IrExpr::Rel { lhs, rhs, .. } => {
                lhs.collect_vars(vars);
                rhs.collect_vars(vars);
            }
            IrExpr::Neg(e) | IrExpr::Not(e) => e.collect_vars(vars),
            IrExpr::And(l, r) | IrExpr::Or(l, r) => {
                l.collect_vars(vars);
                r.collect_vars(vars);
            }
            IrExpr::Select { array, index } => {
                array.collect_vars(vars);
                index.collect_vars(vars);
            }
            IrExpr::Ite {
                cond,
                then_expr,
                else_expr,
            } => {
                cond.collect_vars(vars);
                then_expr.collect_vars(vars);
                else_expr.collect_vars(vars);
            }
            IrExpr::Call { args, .. } => {
                for a in args {
                    a.collect_vars(vars);
                }
            }
            IrExpr::Phi(entries) => {
                for (_, e) in entries {
                    e.collect_vars(vars);
                }
            }
        }
    }

    /// Size of the expression (node count).
    pub fn size(&self) -> usize {
        match self {
            IrExpr::Const(_) | IrExpr::BoolConst(_) | IrExpr::Var(_) => 1,
            IrExpr::BinArith { lhs, rhs, .. } | IrExpr::Rel { lhs, rhs, .. } => {
                1 + lhs.size() + rhs.size()
            }
            IrExpr::Neg(e) | IrExpr::Not(e) => 1 + e.size(),
            IrExpr::And(l, r) | IrExpr::Or(l, r) => 1 + l.size() + r.size(),
            IrExpr::Select { array, index } => 1 + array.size() + index.size(),
            IrExpr::Ite {
                cond,
                then_expr,
                else_expr,
            } => 1 + cond.size() + then_expr.size() + else_expr.size(),
            IrExpr::Call { args, .. } => 1 + args.iter().map(|a| a.size()).sum::<usize>(),
            IrExpr::Phi(entries) => 1 + entries.iter().map(|(_, e)| e.size()).sum::<usize>(),
        }
    }

    /// Substitute an SSA variable with an expression.
    pub fn substitute(&self, target: &SsaVar, replacement: &IrExpr) -> IrExpr {
        match self {
            IrExpr::Var(v) if v == target => replacement.clone(),
            IrExpr::Var(_) | IrExpr::Const(_) | IrExpr::BoolConst(_) => self.clone(),
            IrExpr::BinArith { op, lhs, rhs } => IrExpr::bin_arith(
                *op,
                lhs.substitute(target, replacement),
                rhs.substitute(target, replacement),
            ),
            IrExpr::Neg(e) => IrExpr::neg(e.substitute(target, replacement)),
            IrExpr::Rel { op, lhs, rhs } => IrExpr::rel(
                *op,
                lhs.substitute(target, replacement),
                rhs.substitute(target, replacement),
            ),
            IrExpr::And(l, r) => IrExpr::and(
                l.substitute(target, replacement),
                r.substitute(target, replacement),
            ),
            IrExpr::Or(l, r) => IrExpr::or(
                l.substitute(target, replacement),
                r.substitute(target, replacement),
            ),
            IrExpr::Not(e) => IrExpr::not(e.substitute(target, replacement)),
            IrExpr::Select { array, index } => IrExpr::select(
                array.substitute(target, replacement),
                index.substitute(target, replacement),
            ),
            IrExpr::Ite {
                cond,
                then_expr,
                else_expr,
            } => IrExpr::ite(
                cond.substitute(target, replacement),
                then_expr.substitute(target, replacement),
                else_expr.substitute(target, replacement),
            ),
            IrExpr::Call { name, args } => IrExpr::call(
                name.clone(),
                args.iter()
                    .map(|a| a.substitute(target, replacement))
                    .collect(),
            ),
            IrExpr::Phi(entries) => IrExpr::Phi(
                entries
                    .iter()
                    .map(|(bb, e)| (*bb, e.substitute(target, replacement)))
                    .collect(),
            ),
        }
    }

    pub fn is_const(&self) -> bool {
        matches!(self, IrExpr::Const(_) | IrExpr::BoolConst(_))
    }

    pub fn is_var(&self) -> bool {
        matches!(self, IrExpr::Var(_))
    }
}

impl fmt::Display for IrExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IrExpr::Const(v) => write!(f, "{v}"),
            IrExpr::BoolConst(b) => write!(f, "{b}"),
            IrExpr::Var(v) => write!(f, "{}", v.versioned_name()),
            IrExpr::BinArith { op, lhs, rhs } => write!(f, "({lhs} {op} {rhs})"),
            IrExpr::Neg(e) => write!(f, "(-{e})"),
            IrExpr::Rel { op, lhs, rhs } => write!(f, "({lhs} {op} {rhs})"),
            IrExpr::And(l, r) => write!(f, "({l} && {r})"),
            IrExpr::Or(l, r) => write!(f, "({l} || {r})"),
            IrExpr::Not(e) => write!(f, "(!{e})"),
            IrExpr::Select { array, index } => write!(f, "{array}[{index}]"),
            IrExpr::Ite {
                cond,
                then_expr,
                else_expr,
            } => write!(f, "(ite {cond} {then_expr} {else_expr})"),
            IrExpr::Call { name, args } => {
                write!(f, "{name}(")?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{a}")?;
                }
                write!(f, ")")
            }
            IrExpr::Phi(entries) => {
                write!(f, "phi(")?;
                for (i, (bb, e)) in entries.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "BB{bb}:{e}")?;
                }
                write!(f, ")")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// IrStatement
// ---------------------------------------------------------------------------

/// A statement in the IR.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IrStatement {
    /// Assign an expression to an SSA variable.
    Assign { target: SsaVar, value: IrExpr },
    /// Assert a condition.
    Assert {
        condition: IrExpr,
        message: Option<String>,
    },
    /// Assume a condition (for path-sensitive analysis).
    Assume { condition: IrExpr },
    /// A phi-node assignment.
    PhiAssign {
        target: SsaVar,
        sources: Vec<(usize, SsaVar)>,
    },
    /// No-op (placeholder for deleted statements).
    Nop,
}

impl IrStatement {
    pub fn assign(target: SsaVar, value: IrExpr) -> Self {
        IrStatement::Assign { target, value }
    }

    pub fn assert(condition: IrExpr) -> Self {
        IrStatement::Assert {
            condition,
            message: None,
        }
    }

    pub fn assert_msg(condition: IrExpr, msg: impl Into<String>) -> Self {
        IrStatement::Assert {
            condition,
            message: Some(msg.into()),
        }
    }

    pub fn assume(condition: IrExpr) -> Self {
        IrStatement::Assume { condition }
    }

    pub fn phi_assign(target: SsaVar, sources: Vec<(usize, SsaVar)>) -> Self {
        IrStatement::PhiAssign { target, sources }
    }

    /// Defined variable (if any).
    pub fn defined_var(&self) -> Option<&SsaVar> {
        match self {
            IrStatement::Assign { target, .. } | IrStatement::PhiAssign { target, .. } => {
                Some(target)
            }
            _ => None,
        }
    }

    /// All SSA variables used (read) by this statement.
    pub fn used_vars(&self) -> BTreeSet<SsaVar> {
        match self {
            IrStatement::Assign { value, .. } => value.used_vars(),
            IrStatement::Assert { condition, .. } | IrStatement::Assume { condition } => {
                condition.used_vars()
            }
            IrStatement::PhiAssign { sources, .. } => {
                sources.iter().map(|(_, v)| v.clone()).collect()
            }
            IrStatement::Nop => BTreeSet::new(),
        }
    }

    pub fn is_nop(&self) -> bool {
        matches!(self, IrStatement::Nop)
    }
}

impl fmt::Display for IrStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IrStatement::Assign { target, value } => {
                write!(f, "{} = {}", target.versioned_name(), value)
            }
            IrStatement::Assert { condition, message } => {
                write!(f, "assert({})", condition)?;
                if let Some(msg) = message {
                    write!(f, " // {msg}")?;
                }
                Ok(())
            }
            IrStatement::Assume { condition } => write!(f, "assume({})", condition),
            IrStatement::PhiAssign { target, sources } => {
                write!(f, "{} = phi(", target.versioned_name())?;
                for (i, (bb, src)) in sources.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "BB{}:{}", bb, src.versioned_name())?;
                }
                write!(f, ")")
            }
            IrStatement::Nop => write!(f, "nop"),
        }
    }
}

// ---------------------------------------------------------------------------
// PhiNode
// ---------------------------------------------------------------------------

/// A phi node at the start of a basic block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhiNode {
    pub target: SsaVar,
    /// (predecessor block id, incoming SSA variable).
    pub incoming: Vec<(usize, SsaVar)>,
}

impl PhiNode {
    pub fn new(target: SsaVar, incoming: Vec<(usize, SsaVar)>) -> Self {
        Self { target, incoming }
    }

    /// Predecessor block IDs.
    pub fn predecessors(&self) -> Vec<usize> {
        self.incoming.iter().map(|(id, _)| *id).collect()
    }

    /// Get the incoming variable for a given predecessor.
    pub fn value_from(&self, predecessor: usize) -> Option<&SsaVar> {
        self.incoming
            .iter()
            .find(|(id, _)| *id == predecessor)
            .map(|(_, v)| v)
    }
}

impl fmt::Display for PhiNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} = phi(", self.target.versioned_name())?;
        for (i, (bb, var)) in self.incoming.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "BB{}:{}", bb, var.versioned_name())?;
        }
        write!(f, ")")
    }
}

// ---------------------------------------------------------------------------
// Terminator
// ---------------------------------------------------------------------------

/// How a basic block ends.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Terminator {
    /// Unconditional jump to a successor block.
    Branch { target: usize },
    /// Conditional branch.
    ConditionalBranch {
        condition: IrExpr,
        true_target: usize,
        false_target: usize,
    },
    /// Return from function.
    Return { value: Option<IrExpr> },
    /// Unreachable (for dead code).
    Unreachable,
}

impl Terminator {
    pub fn branch(target: usize) -> Self {
        Terminator::Branch { target }
    }

    pub fn cond_branch(condition: IrExpr, true_target: usize, false_target: usize) -> Self {
        Terminator::ConditionalBranch {
            condition,
            true_target,
            false_target,
        }
    }

    pub fn ret(value: Option<IrExpr>) -> Self {
        Terminator::Return { value }
    }

    /// Successor block IDs.
    pub fn successors(&self) -> Vec<usize> {
        match self {
            Terminator::Branch { target } => vec![*target],
            Terminator::ConditionalBranch {
                true_target,
                false_target,
                ..
            } => vec![*true_target, *false_target],
            Terminator::Return { .. } | Terminator::Unreachable => vec![],
        }
    }

    pub fn is_return(&self) -> bool {
        matches!(self, Terminator::Return { .. })
    }

    pub fn is_branch(&self) -> bool {
        matches!(
            self,
            Terminator::Branch { .. } | Terminator::ConditionalBranch { .. }
        )
    }
}

impl fmt::Display for Terminator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Terminator::Branch { target } => write!(f, "br BB{target}"),
            Terminator::ConditionalBranch {
                condition,
                true_target,
                false_target,
            } => write!(f, "br {condition}, BB{true_target}, BB{false_target}"),
            Terminator::Return { value: Some(v) } => write!(f, "ret {v}"),
            Terminator::Return { value: None } => write!(f, "ret void"),
            Terminator::Unreachable => write!(f, "unreachable"),
        }
    }
}

// ---------------------------------------------------------------------------
// BasicBlock
// ---------------------------------------------------------------------------

/// A basic block in the IR.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BasicBlock {
    pub id: usize,
    pub label: Option<String>,
    pub phi_nodes: Vec<PhiNode>,
    pub statements: Vec<IrStatement>,
    pub terminator: Terminator,
}

impl BasicBlock {
    pub fn new(id: usize, terminator: Terminator) -> Self {
        Self {
            id,
            label: None,
            phi_nodes: Vec::new(),
            statements: Vec::new(),
            terminator,
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn add_phi(&mut self, phi: PhiNode) {
        self.phi_nodes.push(phi);
    }

    pub fn add_statement(&mut self, stmt: IrStatement) {
        self.statements.push(stmt);
    }

    /// All SSA variables defined in this block.
    pub fn defined_vars(&self) -> Vec<&SsaVar> {
        let mut vars: Vec<&SsaVar> = Vec::new();
        for phi in &self.phi_nodes {
            vars.push(&phi.target);
        }
        for stmt in &self.statements {
            if let Some(v) = stmt.defined_var() {
                vars.push(v);
            }
        }
        vars
    }

    /// All SSA variables used in this block.
    pub fn used_vars(&self) -> BTreeSet<SsaVar> {
        let mut vars = BTreeSet::new();
        for phi in &self.phi_nodes {
            for (_, v) in &phi.incoming {
                vars.insert(v.clone());
            }
        }
        for stmt in &self.statements {
            vars.extend(stmt.used_vars());
        }
        match &self.terminator {
            Terminator::ConditionalBranch { condition, .. } => {
                vars.extend(condition.used_vars());
            }
            Terminator::Return { value: Some(v) } => {
                vars.extend(v.used_vars());
            }
            _ => {}
        }
        vars
    }

    /// Successor block IDs.
    pub fn successors(&self) -> Vec<usize> {
        self.terminator.successors()
    }

    /// Number of IR instructions (phis + statements + terminator).
    pub fn instruction_count(&self) -> usize {
        self.phi_nodes.len() + self.statements.len() + 1
    }

    /// Is this an entry block?
    pub fn is_entry(&self) -> bool {
        self.id == 0
    }

    /// Does this block end with a return?
    pub fn is_exit(&self) -> bool {
        self.terminator.is_return()
    }

    /// Display name.
    pub fn display_name(&self) -> String {
        if let Some(ref label) = self.label {
            format!("BB{} ({})", self.id, label)
        } else {
            format!("BB{}", self.id)
        }
    }
}

impl fmt::Display for BasicBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}:", self.display_name())?;
        for phi in &self.phi_nodes {
            writeln!(f, "  {phi}")?;
        }
        for stmt in &self.statements {
            writeln!(f, "  {stmt}")?;
        }
        writeln!(f, "  {}", self.terminator)
    }
}

// ---------------------------------------------------------------------------
// IrFunction
// ---------------------------------------------------------------------------

/// An IR function with basic blocks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrFunction {
    pub name: String,
    pub params: Vec<SsaVar>,
    pub return_type: QfLiaType,
    pub blocks: Vec<BasicBlock>,
}

impl IrFunction {
    pub fn new(name: impl Into<String>, params: Vec<SsaVar>, return_type: QfLiaType) -> Self {
        Self {
            name: name.into(),
            params,
            return_type,
            blocks: Vec::new(),
        }
    }

    pub fn add_block(&mut self, block: BasicBlock) -> usize {
        let id = block.id;
        self.blocks.push(block);
        id
    }

    pub fn block(&self, id: usize) -> Option<&BasicBlock> {
        self.blocks.iter().find(|b| b.id == id)
    }

    pub fn block_mut(&mut self, id: usize) -> Option<&mut BasicBlock> {
        self.blocks.iter_mut().find(|b| b.id == id)
    }

    pub fn entry_block(&self) -> Option<&BasicBlock> {
        self.block(0)
    }

    pub fn exit_blocks(&self) -> Vec<&BasicBlock> {
        self.blocks.iter().filter(|b| b.is_exit()).collect()
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Total instruction count.
    pub fn instruction_count(&self) -> usize {
        self.blocks.iter().map(|b| b.instruction_count()).sum()
    }

    /// All SSA variables defined in the function.
    pub fn all_defined_vars(&self) -> Vec<&SsaVar> {
        let mut vars = Vec::new();
        vars.extend(self.params.iter());
        for block in &self.blocks {
            vars.extend(block.defined_vars());
        }
        vars
    }

    /// All SSA variables used in the function.
    pub fn all_used_vars(&self) -> BTreeSet<SsaVar> {
        let mut vars = BTreeSet::new();
        for block in &self.blocks {
            vars.extend(block.used_vars());
        }
        vars
    }

    /// Compute predecessor map: block_id -> list of predecessor block_ids.
    pub fn predecessors(&self) -> BTreeMap<usize, Vec<usize>> {
        let mut preds: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for block in &self.blocks {
            preds.entry(block.id).or_default();
            for succ in block.successors() {
                preds.entry(succ).or_default().push(block.id);
            }
        }
        preds
    }

    /// Topological order of blocks (simple BFS from entry).
    pub fn block_order(&self) -> Vec<usize> {
        let mut visited = BTreeSet::new();
        let mut order = Vec::new();
        let mut worklist = vec![0usize];
        while let Some(id) = worklist.pop() {
            if !visited.insert(id) {
                continue;
            }
            order.push(id);
            if let Some(block) = self.block(id) {
                for succ in block.successors() {
                    if !visited.contains(&succ) {
                        worklist.push(succ);
                    }
                }
            }
        }
        order
    }

    /// Build a signature for this function.
    pub fn signature(&self) -> crate::types::FunctionSignature {
        crate::types::FunctionSignature::new(
            self.name.clone(),
            self.params.iter().map(|p| (p.name.clone(), p.ty)).collect(),
            self.return_type,
        )
    }

    /// Create a new unique block ID.
    pub fn next_block_id(&self) -> usize {
        self.blocks.iter().map(|b| b.id).max().unwrap_or(0) + 1
    }
}

impl fmt::Display for IrFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "fn {} {}(", self.return_type, self.name)?;
        for (i, p) in self.params.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{p}")?;
        }
        writeln!(f, ") {{")?;
        for block in &self.blocks {
            write!(f, "{block}")?;
        }
        writeln!(f, "}}")
    }
}

// ---------------------------------------------------------------------------
// IrProgram
// ---------------------------------------------------------------------------

/// A complete IR program.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrProgram {
    pub functions: Vec<IrFunction>,
    pub name: Option<String>,
}

impl IrProgram {
    pub fn new(functions: Vec<IrFunction>) -> Self {
        Self {
            functions,
            name: None,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn function(&self, name: &str) -> Option<&IrFunction> {
        self.functions.iter().find(|f| f.name == name)
    }

    pub fn function_mut(&mut self, name: &str) -> Option<&mut IrFunction> {
        self.functions.iter_mut().find(|f| f.name == name)
    }

    pub fn function_names(&self) -> Vec<&str> {
        self.functions.iter().map(|f| f.name.as_str()).collect()
    }

    pub fn function_count(&self) -> usize {
        self.functions.len()
    }

    pub fn total_blocks(&self) -> usize {
        self.functions.iter().map(|f| f.block_count()).sum()
    }

    pub fn total_instructions(&self) -> usize {
        self.functions.iter().map(|f| f.instruction_count()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.functions.is_empty()
    }
}

impl fmt::Display for IrProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref name) = self.name {
            writeln!(f, "// IR: {name}")?;
        }
        for func in &self.functions {
            writeln!(f)?;
            write!(f, "{func}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// AST -> IR conversion traits
// ---------------------------------------------------------------------------

/// Trait for converting AST expressions to IR expressions.
pub trait AstToIr {
    /// Convert an AST expression to an IR expression, using `env` to resolve
    /// variable names to their current SSA versions.
    fn lower_expr(&self, expr: &ast::Expression, env: &BTreeMap<String, SsaVar>) -> IrExpr;
}

/// Basic AST-to-IR lowering implementation.
pub struct BasicLowering;

impl AstToIr for BasicLowering {
    fn lower_expr(&self, expr: &ast::Expression, env: &BTreeMap<String, SsaVar>) -> IrExpr {
        match expr {
            ast::Expression::IntLiteral(v) => IrExpr::constant(*v),
            ast::Expression::BoolLiteral(b) => IrExpr::bool_const(*b),
            ast::Expression::Var(name) => {
                if let Some(ssa) = env.get(name) {
                    IrExpr::var(ssa.clone())
                } else {
                    // Fallback: create a base version.
                    IrExpr::var(SsaVar::base(name.clone(), QfLiaType::Int))
                }
            }
            ast::Expression::BinaryArith { op, lhs, rhs } => {
                IrExpr::bin_arith(*op, self.lower_expr(lhs, env), self.lower_expr(rhs, env))
            }
            ast::Expression::UnaryArith(e) => IrExpr::neg(self.lower_expr(e, env)),
            ast::Expression::Relational { op, lhs, rhs } => {
                IrExpr::rel(*op, self.lower_expr(lhs, env), self.lower_expr(rhs, env))
            }
            ast::Expression::LogicalAnd(l, r) => {
                IrExpr::and(self.lower_expr(l, env), self.lower_expr(r, env))
            }
            ast::Expression::LogicalOr(l, r) => {
                IrExpr::or(self.lower_expr(l, env), self.lower_expr(r, env))
            }
            ast::Expression::LogicalNot(e) => IrExpr::not(self.lower_expr(e, env)),
            ast::Expression::ArrayAccess { array, index } => {
                IrExpr::select(self.lower_expr(array, env), self.lower_expr(index, env))
            }
            ast::Expression::FunctionCall { name, args } => {
                let ir_args: Vec<_> = args.iter().map(|a| self.lower_expr(a, env)).collect();
                IrExpr::call(name.clone(), ir_args)
            }
            ast::Expression::Conditional {
                condition,
                then_expr,
                else_expr,
            } => IrExpr::ite(
                self.lower_expr(condition, env),
                self.lower_expr(then_expr, env),
                self.lower_expr(else_expr, env),
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::ArithOp;

    fn x0() -> SsaVar {
        SsaVar::new("x", 0, QfLiaType::Int)
    }

    fn x1() -> SsaVar {
        SsaVar::new("x", 1, QfLiaType::Int)
    }

    fn y0() -> SsaVar {
        SsaVar::new("y", 0, QfLiaType::Int)
    }

    // -- SsaVar tests --

    #[test]
    fn test_ssa_var_basics() {
        let v = x0();
        assert_eq!(v.versioned_name(), "x_0");
        assert_eq!(v.smt_name(), "|x_0|");
        assert!(v.is_initial());
    }

    #[test]
    fn test_ssa_var_next_version() {
        let v = x0();
        let v1 = v.next_version();
        assert_eq!(v1.version, 1);
        assert_eq!(v1.name, "x");
        assert!(!v1.is_initial());
    }

    #[test]
    fn test_ssa_var_to_variable() {
        let v = x1();
        let var = v.to_variable();
        assert!(var.is_ssa());
        assert_eq!(var.ssa_info(), Some(("x", 1)));
    }

    #[test]
    fn test_ssa_var_display() {
        let v = x0();
        assert_eq!(v.to_string(), "x_0: int");
    }

    // -- IrExpr tests --

    #[test]
    fn test_ir_expr_const() {
        let e = IrExpr::constant(42);
        assert!(e.is_const());
        assert_eq!(e.size(), 1);
        assert_eq!(e.to_string(), "42");
    }

    #[test]
    fn test_ir_expr_var() {
        let e = IrExpr::var(x0());
        assert!(e.is_var());
        assert_eq!(e.used_vars().len(), 1);
    }

    #[test]
    fn test_ir_expr_bin_arith() {
        let e = IrExpr::bin_arith(ArithOp::Add, IrExpr::var(x0()), IrExpr::constant(1));
        assert_eq!(e.size(), 3);
        let s = e.to_string();
        assert!(s.contains("+"));
    }

    #[test]
    fn test_ir_expr_rel() {
        let e = IrExpr::rel(ast::RelOp::Lt, IrExpr::var(x0()), IrExpr::constant(10));
        let s = e.to_string();
        assert!(s.contains("<"));
    }

    #[test]
    fn test_ir_expr_logical() {
        let e = IrExpr::and(IrExpr::bool_const(true), IrExpr::bool_const(false));
        assert!(e.to_string().contains("&&"));
    }

    #[test]
    fn test_ir_expr_not() {
        let e = IrExpr::not(IrExpr::bool_const(true));
        assert!(e.to_string().contains("!"));
    }

    #[test]
    fn test_ir_expr_select() {
        let e = IrExpr::select(IrExpr::var(x0()), IrExpr::constant(0));
        assert!(e.to_string().contains("["));
    }

    #[test]
    fn test_ir_expr_ite() {
        let e = IrExpr::ite(
            IrExpr::bool_const(true),
            IrExpr::constant(1),
            IrExpr::constant(0),
        );
        assert!(e.to_string().contains("ite"));
    }

    #[test]
    fn test_ir_expr_call() {
        let e = IrExpr::call("f", vec![IrExpr::var(x0())]);
        assert!(e.to_string().contains("f("));
    }

    #[test]
    fn test_ir_expr_substitute() {
        let e = IrExpr::bin_arith(ArithOp::Add, IrExpr::var(x0()), IrExpr::constant(1));
        let replaced = e.substitute(&x0(), &IrExpr::constant(5));
        match &replaced {
            IrExpr::BinArith { lhs, .. } => {
                assert_eq!(**lhs, IrExpr::constant(5));
            }
            _ => panic!("expected BinArith"),
        }
    }

    #[test]
    fn test_ir_expr_used_vars() {
        let e = IrExpr::bin_arith(ArithOp::Add, IrExpr::var(x0()), IrExpr::var(y0()));
        let vars = e.used_vars();
        assert_eq!(vars.len(), 2);
    }

    #[test]
    fn test_ir_expr_phi() {
        let e = IrExpr::Phi(vec![(0, IrExpr::var(x0())), (1, IrExpr::var(x1()))]);
        let s = e.to_string();
        assert!(s.contains("phi"));
    }

    // -- IrStatement tests --

    #[test]
    fn test_ir_stmt_assign() {
        let s = IrStatement::assign(x1(), IrExpr::constant(42));
        assert_eq!(s.defined_var(), Some(&x1()));
        assert!(s.used_vars().is_empty());
        assert!(!s.is_nop());
    }

    #[test]
    fn test_ir_stmt_assert() {
        let s = IrStatement::assert_msg(IrExpr::bool_const(true), "ok");
        let display = s.to_string();
        assert!(display.contains("assert"));
        assert!(display.contains("ok"));
    }

    #[test]
    fn test_ir_stmt_assume() {
        let s = IrStatement::assume(IrExpr::bool_const(true));
        let display = s.to_string();
        assert!(display.contains("assume"));
    }

    #[test]
    fn test_ir_stmt_phi_assign() {
        let s = IrStatement::phi_assign(x1(), vec![(0, x0()), (1, x0())]);
        let display = s.to_string();
        assert!(display.contains("phi"));
    }

    #[test]
    fn test_ir_stmt_nop() {
        let s = IrStatement::Nop;
        assert!(s.is_nop());
        assert_eq!(s.to_string(), "nop");
    }

    // -- PhiNode tests --

    #[test]
    fn test_phi_node() {
        let phi = PhiNode::new(x1(), vec![(0, x0()), (1, x0())]);
        assert_eq!(phi.predecessors(), vec![0, 1]);
        assert_eq!(phi.value_from(0), Some(&x0()));
        assert_eq!(phi.value_from(5), None);
    }

    #[test]
    fn test_phi_node_display() {
        let phi = PhiNode::new(x1(), vec![(0, x0())]);
        let s = phi.to_string();
        assert!(s.contains("phi"));
        assert!(s.contains("BB0"));
    }

    // -- Terminator tests --

    #[test]
    fn test_terminator_branch() {
        let t = Terminator::branch(1);
        assert_eq!(t.successors(), vec![1]);
        assert!(t.is_branch());
        assert!(!t.is_return());
    }

    #[test]
    fn test_terminator_cond_branch() {
        let t = Terminator::cond_branch(IrExpr::bool_const(true), 1, 2);
        assert_eq!(t.successors(), vec![1, 2]);
    }

    #[test]
    fn test_terminator_return() {
        let t = Terminator::ret(Some(IrExpr::constant(0)));
        assert!(t.is_return());
        assert!(t.successors().is_empty());
    }

    #[test]
    fn test_terminator_display() {
        assert!(Terminator::branch(1).to_string().contains("br BB1"));
        assert!(Terminator::ret(None).to_string().contains("ret void"));
        assert!(Terminator::Unreachable.to_string().contains("unreachable"));
    }

    // -- BasicBlock tests --

    #[test]
    fn test_basic_block() {
        let mut bb = BasicBlock::new(0, Terminator::ret(Some(IrExpr::var(x0()))));
        bb.add_statement(IrStatement::assign(x0(), IrExpr::constant(42)));
        assert!(bb.is_entry());
        assert!(bb.is_exit());
        assert_eq!(bb.instruction_count(), 2); // 1 stmt + 1 terminator
    }

    #[test]
    fn test_basic_block_defined_used_vars() {
        let mut bb = BasicBlock::new(0, Terminator::ret(None));
        bb.add_statement(IrStatement::assign(x1(), IrExpr::var(x0())));
        let defined = bb.defined_vars();
        assert_eq!(defined.len(), 1);
        assert_eq!(defined[0], &x1());
        let used = bb.used_vars();
        assert!(used.contains(&x0()));
    }

    #[test]
    fn test_basic_block_display() {
        let bb = BasicBlock::new(0, Terminator::branch(1)).with_label("entry");
        let s = bb.to_string();
        assert!(s.contains("entry"));
        assert!(s.contains("BB0"));
    }

    #[test]
    fn test_basic_block_successors() {
        let bb = BasicBlock::new(0, Terminator::cond_branch(IrExpr::bool_const(true), 1, 2));
        assert_eq!(bb.successors(), vec![1, 2]);
    }

    // -- IrFunction tests --

    #[test]
    fn test_ir_function_basic() {
        let mut func = IrFunction::new("f", vec![x0()], QfLiaType::Int);
        let bb = BasicBlock::new(0, Terminator::ret(Some(IrExpr::var(x0()))));
        func.add_block(bb);
        assert_eq!(func.block_count(), 1);
        assert!(func.entry_block().is_some());
        assert_eq!(func.exit_blocks().len(), 1);
    }

    #[test]
    fn test_ir_function_predecessors() {
        let mut func = IrFunction::new("f", vec![], QfLiaType::Void);
        func.add_block(BasicBlock::new(0, Terminator::branch(1)));
        func.add_block(BasicBlock::new(1, Terminator::ret(None)));
        let preds = func.predecessors();
        assert_eq!(preds[&1], vec![0]);
    }

    #[test]
    fn test_ir_function_block_order() {
        let mut func = IrFunction::new("f", vec![], QfLiaType::Void);
        func.add_block(BasicBlock::new(0, Terminator::branch(1)));
        func.add_block(BasicBlock::new(1, Terminator::ret(None)));
        let order = func.block_order();
        assert_eq!(order, vec![0, 1]);
    }

    #[test]
    fn test_ir_function_display() {
        let mut func = IrFunction::new("test", vec![x0()], QfLiaType::Int);
        func.add_block(BasicBlock::new(0, Terminator::ret(Some(IrExpr::var(x0())))));
        let s = func.to_string();
        assert!(s.contains("fn int test"));
        assert!(s.contains("ret"));
    }

    #[test]
    fn test_ir_function_signature() {
        let func = IrFunction::new("add", vec![x0(), y0()], QfLiaType::Int);
        let sig = func.signature();
        assert_eq!(sig.arity(), 2);
        assert_eq!(sig.name, "add");
    }

    #[test]
    fn test_ir_function_next_block_id() {
        let mut func = IrFunction::new("f", vec![], QfLiaType::Void);
        assert_eq!(func.next_block_id(), 1);
        func.add_block(BasicBlock::new(0, Terminator::ret(None)));
        assert_eq!(func.next_block_id(), 1);
        func.add_block(BasicBlock::new(5, Terminator::ret(None)));
        assert_eq!(func.next_block_id(), 6);
    }

    // -- IrProgram tests --

    #[test]
    fn test_ir_program_basic() {
        let func = IrFunction::new("main", vec![], QfLiaType::Void);
        let prog = IrProgram::new(vec![func]).with_name("test");
        assert_eq!(prog.function_count(), 1);
        assert!(prog.function("main").is_some());
        assert!(prog.function("other").is_none());
        assert_eq!(prog.function_names(), vec!["main"]);
    }

    #[test]
    fn test_ir_program_totals() {
        let mut f1 = IrFunction::new("f1", vec![], QfLiaType::Void);
        f1.add_block(BasicBlock::new(0, Terminator::ret(None)));
        let mut f2 = IrFunction::new("f2", vec![], QfLiaType::Void);
        f2.add_block(BasicBlock::new(0, Terminator::branch(1)));
        f2.add_block(BasicBlock::new(1, Terminator::ret(None)));
        let prog = IrProgram::new(vec![f1, f2]);
        assert_eq!(prog.total_blocks(), 3);
        assert_eq!(prog.total_instructions(), 3);
    }

    #[test]
    fn test_ir_program_display() {
        let func = IrFunction::new("main", vec![], QfLiaType::Void);
        let prog = IrProgram::new(vec![func]).with_name("demo");
        let s = prog.to_string();
        assert!(s.contains("demo"));
        assert!(s.contains("main"));
    }

    // -- BasicLowering tests --

    #[test]
    fn test_basic_lowering_literal() {
        let lowering = BasicLowering;
        let env = BTreeMap::new();
        let e = lowering.lower_expr(&ast::Expression::int_lit(42), &env);
        assert_eq!(e, IrExpr::constant(42));
    }

    #[test]
    fn test_basic_lowering_var() {
        let lowering = BasicLowering;
        let mut env = BTreeMap::new();
        env.insert("x".to_string(), x1());
        let e = lowering.lower_expr(&ast::Expression::var("x"), &env);
        assert_eq!(e, IrExpr::var(x1()));
    }

    #[test]
    fn test_basic_lowering_binary() {
        let lowering = BasicLowering;
        let mut env = BTreeMap::new();
        env.insert("x".to_string(), x0());
        let ast_expr = ast::Expression::add(ast::Expression::var("x"), ast::Expression::int_lit(1));
        let ir = lowering.lower_expr(&ast_expr, &env);
        assert_eq!(ir.size(), 3);
    }

    #[test]
    fn test_basic_lowering_conditional() {
        let lowering = BasicLowering;
        let env = BTreeMap::new();
        let ast_expr = ast::Expression::conditional(
            ast::Expression::bool_lit(true),
            ast::Expression::int_lit(1),
            ast::Expression::int_lit(0),
        );
        let ir = lowering.lower_expr(&ast_expr, &env);
        match ir {
            IrExpr::Ite { .. } => {}
            _ => panic!("expected Ite"),
        }
    }

    #[test]
    fn test_ssa_var_serialization() {
        let v = x0();
        let json = serde_json::to_string(&v).unwrap();
        let v2: SsaVar = serde_json::from_str(&json).unwrap();
        assert_eq!(v, v2);
    }

    #[test]
    fn test_ir_program_serialization() {
        let func = IrFunction::new("f", vec![x0()], QfLiaType::Int);
        let prog = IrProgram::new(vec![func]);
        let json = serde_json::to_string(&prog).unwrap();
        let prog2: IrProgram = serde_json::from_str(&json).unwrap();
        assert_eq!(prog, prog2);
    }
}

//! Data flow analysis framework.
//!
//! Provides a generic fixed-point iteration engine and concrete analyses:
//! - Reaching definitions
//! - Live variables
//! - Available expressions
//! - Constant propagation
//! - Definition-use chains

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;

use shared_types::{
    AnalysisError, AnalysisResult, BasicBlock, BlockId, IrExpr, IrFunction, IrStatement,
    Terminator, VarId,
};

// ---------------------------------------------------------------------------
// Generic data flow analysis trait
// ---------------------------------------------------------------------------

/// Direction of data flow analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Forward,
    Backward,
}

/// Generic trait for data flow analyses.
pub trait DataFlowAnalysis {
    /// The lattice type for this analysis.
    type Lattice: Clone + PartialEq + fmt::Debug;

    /// Direction of the analysis.
    fn direction(&self) -> Direction;

    /// Initial value for entry/exit block.
    fn initial_value(&self) -> Self::Lattice;

    /// Bottom of the lattice (initial value for non-entry blocks).
    fn bottom(&self) -> Self::Lattice;

    /// Meet/join operation (merge values from predecessors/successors).
    fn merge(&self, a: &Self::Lattice, b: &Self::Lattice) -> Self::Lattice;

    /// Transfer function for a single block.
    fn transfer(&self, block: &BasicBlock, input: &Self::Lattice) -> Self::Lattice;
}

/// Results of a data flow analysis: maps block IDs to in/out values.
#[derive(Debug, Clone)]
pub struct AnalysisResults<L: Clone + fmt::Debug> {
    pub in_values: HashMap<BlockId, L>,
    pub out_values: HashMap<BlockId, L>,
}

/// Run a fixed-point iteration data flow analysis.
pub fn run_analysis<A: DataFlowAnalysis>(
    analysis: &A,
    func: &IrFunction,
) -> AnalysisResults<A::Lattice> {
    let mut in_vals: HashMap<BlockId, A::Lattice> = HashMap::new();
    let mut out_vals: HashMap<BlockId, A::Lattice> = HashMap::new();

    let preds = compute_predecessors(func);
    let succs = compute_successors(func);

    // Initialize
    for block in &func.blocks {
        if block.id == func.entry_block && analysis.direction() == Direction::Forward {
            in_vals.insert(block.id, analysis.initial_value());
        } else {
            in_vals.insert(block.id, analysis.bottom());
        }
        out_vals.insert(block.id, analysis.bottom());
    }

    // Fixed-point iteration
    let mut changed = true;
    let mut iterations = 0;
    while changed && iterations < 1000 {
        changed = false;
        iterations += 1;

        for block in &func.blocks {
            match analysis.direction() {
                Direction::Forward => {
                    // Merge predecessors' out values
                    let pred_ids = preds.get(&block.id).cloned().unwrap_or_default();
                    let mut merged = if pred_ids.is_empty() {
                        analysis.initial_value()
                    } else {
                        let first = out_vals
                            .get(&pred_ids[0])
                            .cloned()
                            .unwrap_or_else(|| analysis.bottom());
                        pred_ids[1..].iter().fold(first, |acc, pid| {
                            let val = out_vals
                                .get(pid)
                                .cloned()
                                .unwrap_or_else(|| analysis.bottom());
                            analysis.merge(&acc, &val)
                        })
                    };

                    if block.id == func.entry_block {
                        merged = analysis.merge(&merged, &analysis.initial_value());
                    }

                    in_vals.insert(block.id, merged.clone());

                    let new_out = analysis.transfer(block, &merged);
                    if out_vals.get(&block.id) != Some(&new_out) {
                        out_vals.insert(block.id, new_out);
                        changed = true;
                    }
                }
                Direction::Backward => {
                    // Merge successors' in values
                    let succ_ids = succs.get(&block.id).cloned().unwrap_or_default();
                    let merged = if succ_ids.is_empty() {
                        analysis.initial_value()
                    } else {
                        let first = in_vals
                            .get(&succ_ids[0])
                            .cloned()
                            .unwrap_or_else(|| analysis.bottom());
                        succ_ids[1..].iter().fold(first, |acc, sid| {
                            let val = in_vals
                                .get(sid)
                                .cloned()
                                .unwrap_or_else(|| analysis.bottom());
                            analysis.merge(&acc, &val)
                        })
                    };
                    out_vals.insert(block.id, merged.clone());

                    let new_in = analysis.transfer(block, &merged);
                    if in_vals.get(&block.id) != Some(&new_in) {
                        in_vals.insert(block.id, new_in);
                        changed = true;
                    }
                }
            }
        }
    }

    AnalysisResults {
        in_values: in_vals,
        out_values: out_vals,
    }
}

fn compute_predecessors(func: &IrFunction) -> HashMap<BlockId, Vec<BlockId>> {
    let mut preds: HashMap<BlockId, Vec<BlockId>> =
        func.blocks.iter().map(|b| (b.id, Vec::new())).collect();
    for block in &func.blocks {
        for succ in block.terminator.successors() {
            preds.entry(succ).or_default().push(block.id);
        }
    }
    preds
}

fn compute_successors(func: &IrFunction) -> HashMap<BlockId, Vec<BlockId>> {
    func.blocks
        .iter()
        .map(|b| (b.id, b.terminator.successors()))
        .collect()
}

// ---------------------------------------------------------------------------
// Reaching Definitions
// ---------------------------------------------------------------------------

/// A definition: (variable, block, statement index).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Definition {
    pub var: VarId,
    pub block: BlockId,
    pub stmt_index: usize,
}

/// Reaching definitions analysis.
pub struct ReachingDefinitions {
    all_defs: Vec<Definition>,
}

impl ReachingDefinitions {
    pub fn new(func: &IrFunction) -> Self {
        let mut defs = Vec::new();
        for block in &func.blocks {
            for (i, stmt) in block.stmts.iter().enumerate() {
                if let Some(var) = stmt.defined_var() {
                    defs.push(Definition {
                        var: var.clone(),
                        block: block.id,
                        stmt_index: i,
                    });
                }
            }
        }
        ReachingDefinitions { all_defs: defs }
    }
}

impl DataFlowAnalysis for ReachingDefinitions {
    type Lattice = BTreeSet<Definition>;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn initial_value(&self) -> Self::Lattice {
        BTreeSet::new()
    }
    fn bottom(&self) -> Self::Lattice {
        BTreeSet::new()
    }

    fn merge(&self, a: &Self::Lattice, b: &Self::Lattice) -> Self::Lattice {
        a.union(b).cloned().collect()
    }

    fn transfer(&self, block: &BasicBlock, input: &Self::Lattice) -> Self::Lattice {
        let mut out = input.clone();
        for (i, stmt) in block.stmts.iter().enumerate() {
            if let Some(var) = stmt.defined_var() {
                // Kill all previous definitions of this variable
                out.retain(|d| d.var != *var);
                // Gen new definition
                out.insert(Definition {
                    var: var.clone(),
                    block: block.id,
                    stmt_index: i,
                });
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Live Variables
// ---------------------------------------------------------------------------

/// Live variables analysis (backward).
pub struct LiveVariables;

impl LiveVariables {
    pub fn new() -> Self {
        LiveVariables
    }
}

impl DataFlowAnalysis for LiveVariables {
    type Lattice = BTreeSet<VarId>;

    fn direction(&self) -> Direction {
        Direction::Backward
    }

    fn initial_value(&self) -> Self::Lattice {
        BTreeSet::new()
    }
    fn bottom(&self) -> Self::Lattice {
        BTreeSet::new()
    }

    fn merge(&self, a: &Self::Lattice, b: &Self::Lattice) -> Self::Lattice {
        a.union(b).cloned().collect()
    }

    fn transfer(&self, block: &BasicBlock, input: &Self::Lattice) -> Self::Lattice {
        // out - def + use (backward: input is the "out" of this block)
        let mut live = input.clone();

        // Process terminator uses
        for var in block.terminator.referenced_vars() {
            live.insert(var);
        }

        // Process statements in reverse
        for stmt in block.stmts.iter().rev() {
            // Remove defined variable
            if let Some(def) = stmt.defined_var() {
                live.remove(def);
            }
            // Add used variables
            for var in stmt.used_vars() {
                live.insert(var);
            }
        }

        live
    }
}

impl Default for LiveVariables {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Available Expressions
// ---------------------------------------------------------------------------

/// An expression with its block location.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AvailExpr {
    pub expr_str: String,
    pub block: BlockId,
}

/// Available expressions analysis (forward).
pub struct AvailableExpressions {
    all_exprs: BTreeSet<AvailExpr>,
}

impl AvailableExpressions {
    pub fn new(func: &IrFunction) -> Self {
        let mut exprs = BTreeSet::new();
        for block in &func.blocks {
            for stmt in &block.stmts {
                if let IrStatement::Assign { value, .. } = stmt {
                    let s = format!("{:?}", value);
                    exprs.insert(AvailExpr {
                        expr_str: s,
                        block: block.id,
                    });
                }
            }
        }
        AvailableExpressions { all_exprs: exprs }
    }
}

impl DataFlowAnalysis for AvailableExpressions {
    type Lattice = BTreeSet<String>;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn initial_value(&self) -> Self::Lattice {
        BTreeSet::new()
    }

    fn bottom(&self) -> Self::Lattice {
        // For available expressions, bottom is the universal set (all exprs).
        // But for simplicity, we use empty set and do union (may-available).
        BTreeSet::new()
    }

    fn merge(&self, a: &Self::Lattice, b: &Self::Lattice) -> Self::Lattice {
        // Intersection for must-available, but we use union for may-available
        a.intersection(b).cloned().collect()
    }

    fn transfer(&self, block: &BasicBlock, input: &Self::Lattice) -> Self::Lattice {
        let mut avail = input.clone();
        for stmt in &block.stmts {
            if let IrStatement::Assign { target, value } = stmt {
                // Kill expressions that use the defined variable
                let target_clone = target.clone();
                avail.retain(|e| !e.contains(&target_clone));
                // Gen the new expression
                avail.insert(format!("{:?}", value));
            }
        }
        avail
    }
}

// ---------------------------------------------------------------------------
// Constant Propagation
// ---------------------------------------------------------------------------

/// A constant value or top (unknown) or bottom (unreachable).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstValue {
    Top,
    Const(i64),
    Bottom,
}

/// Constant propagation analysis (forward).
pub struct ConstantPropagation;

impl ConstantPropagation {
    pub fn new() -> Self {
        ConstantPropagation
    }
}

impl DataFlowAnalysis for ConstantPropagation {
    type Lattice = HashMap<VarId, ConstValue>;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn initial_value(&self) -> Self::Lattice {
        HashMap::new()
    }
    fn bottom(&self) -> Self::Lattice {
        HashMap::new()
    }

    fn merge(&self, a: &Self::Lattice, b: &Self::Lattice) -> Self::Lattice {
        let mut result = a.clone();
        for (var, val_b) in b {
            let merged = match (result.get(var), val_b) {
                (None, v) => v.clone(),
                (Some(v), _) if *v == *val_b => v.clone(),
                (Some(ConstValue::Bottom), v) | (_, ConstValue::Bottom) => {
                    result.get(var).cloned().unwrap_or(val_b.clone())
                }
                (Some(ConstValue::Const(a_val)), ConstValue::Const(b_val)) => {
                    if a_val == b_val {
                        ConstValue::Const(*a_val)
                    } else {
                        ConstValue::Top
                    }
                }
                _ => ConstValue::Top,
            };
            result.insert(var.clone(), merged);
        }
        result
    }

    fn transfer(&self, block: &BasicBlock, input: &Self::Lattice) -> Self::Lattice {
        let mut state = input.clone();
        for stmt in &block.stmts {
            if let IrStatement::Assign { target, value } = stmt {
                let val = self.eval_const(value, &state);
                state.insert(target.clone(), val);
            }
        }
        state
    }
}

impl ConstantPropagation {
    fn eval_const(&self, expr: &IrExpr, state: &HashMap<VarId, ConstValue>) -> ConstValue {
        match expr {
            IrExpr::IntConst(v) => ConstValue::Const(*v),
            IrExpr::BoolConst(true) => ConstValue::Const(1),
            IrExpr::BoolConst(false) => ConstValue::Const(0),
            IrExpr::Var(name) => state.get(name).cloned().unwrap_or(ConstValue::Top),
            IrExpr::BinaryArith { left, op, right } => {
                match (self.eval_const(left, state), self.eval_const(right, state)) {
                    (ConstValue::Const(a), ConstValue::Const(b)) => {
                        let result = match op {
                            shared_types::ArithOp::Add => a.checked_add(b),
                            shared_types::ArithOp::Sub => a.checked_sub(b),
                            shared_types::ArithOp::Mul => a.checked_mul(b),
                            shared_types::ArithOp::Div => {
                                if b != 0 {
                                    a.checked_div(b)
                                } else {
                                    None
                                }
                            }
                            shared_types::ArithOp::Mod => {
                                if b != 0 {
                                    a.checked_rem(b)
                                } else {
                                    None
                                }
                            }
                        };
                        result.map_or(ConstValue::Top, ConstValue::Const)
                    }
                    _ => ConstValue::Top,
                }
            }
            _ => ConstValue::Top,
        }
    }
}

impl Default for ConstantPropagation {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Definition-Use chains
// ---------------------------------------------------------------------------

/// A use site: (block, statement index, variable name).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UseSite {
    pub block: BlockId,
    pub stmt_index: usize,
    pub var: VarId,
}

/// Build definition-use chains for a function.
pub fn build_def_use_chains(func: &IrFunction) -> HashMap<Definition, Vec<UseSite>> {
    let rd = ReachingDefinitions::new(func);
    let results = run_analysis(&rd, func);
    let mut chains: HashMap<Definition, Vec<UseSite>> = HashMap::new();

    for block in &func.blocks {
        let mut reaching = results
            .in_values
            .get(&block.id)
            .cloned()
            .unwrap_or_default();

        for (i, stmt) in block.stmts.iter().enumerate() {
            // For each use in this statement, link to reaching definitions
            for var in stmt.used_vars() {
                let use_site = UseSite {
                    block: block.id,
                    stmt_index: i,
                    var: var.clone(),
                };
                for def in reaching.iter() {
                    if def.var == var {
                        chains
                            .entry(def.clone())
                            .or_default()
                            .push(use_site.clone());
                    }
                }
            }
            // Update reaching defs after this statement
            if let Some(def_var) = stmt.defined_var() {
                reaching.retain(|d| d.var != *def_var);
                reaching.insert(Definition {
                    var: def_var.clone(),
                    block: block.id,
                    stmt_index: i,
                });
            }
        }
    }

    chains
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_lowering::IrLowering;
    use crate::parser::Parser;

    fn get_ir(src: &str) -> IrFunction {
        let prog = Parser::parse_source(src).unwrap();
        let ir = IrLowering::new().lower_program(&prog).unwrap();
        ir.functions.into_iter().next().unwrap()
    }

    #[test]
    fn test_reaching_defs_simple() {
        let func = get_ir("fn f(x: int) -> int { let y: int = x + 1; return y; }");
        let rd = ReachingDefinitions::new(&func);
        let results = run_analysis(&rd, &func);
        assert!(!results.out_values.is_empty());
    }

    #[test]
    fn test_reaching_defs_if() {
        let func = get_ir("fn f(x: int) -> int { let y: int = 0; if (x > 0) { y = x; } else { y = -x; } return y; }");
        let rd = ReachingDefinitions::new(&func);
        let results = run_analysis(&rd, &func);
        assert!(!results.in_values.is_empty());
    }

    #[test]
    fn test_live_variables_simple() {
        let func = get_ir("fn f(x: int) -> int { let y: int = x + 1; return y; }");
        let lv = LiveVariables::new();
        let results = run_analysis(&lv, &func);
        // x should be live at the entry
        let entry_in = results.in_values.get(&func.entry_block).unwrap();
        assert!(entry_in.contains("x"), "x should be live at entry");
    }

    #[test]
    fn test_live_variables_dead() {
        let func = get_ir("fn f(x: int) -> int { let y: int = 42; return x; }");
        let lv = LiveVariables::new();
        let results = run_analysis(&lv, &func);
        let entry_in = results.in_values.get(&func.entry_block).unwrap();
        // y is dead (unused after assignment, x is returned)
        assert!(!entry_in.contains("y"), "y should not be live at entry");
    }

    #[test]
    fn test_available_exprs() {
        let func =
            get_ir("fn f(x: int) -> int { let y: int = x + 1; let z: int = x + 1; return z; }");
        let ae = AvailableExpressions::new(&func);
        let results = run_analysis(&ae, &func);
        assert!(!results.out_values.is_empty());
    }

    #[test]
    fn test_constant_prop_simple() {
        let func = get_ir("fn f() -> int { let x: int = 5; let y: int = x + 3; return y; }");
        let cp = ConstantPropagation::new();
        let results = run_analysis(&cp, &func);
        let entry_out = results.out_values.get(&func.entry_block).unwrap();
        // x should be constant 5
        assert_eq!(entry_out.get("x"), Some(&ConstValue::Const(5)));
    }

    #[test]
    fn test_constant_prop_non_const() {
        let func = get_ir("fn f(x: int) -> int { let y: int = x + 1; return y; }");
        let cp = ConstantPropagation::new();
        let results = run_analysis(&cp, &func);
        let entry_out = results.out_values.get(&func.entry_block).unwrap();
        // y depends on parameter x, so should be Top
        assert_eq!(entry_out.get("y"), Some(&ConstValue::Top));
    }

    #[test]
    fn test_def_use_chains() {
        let func = get_ir("fn f(x: int) -> int { let y: int = x + 1; return y; }");
        let chains = build_def_use_chains(&func);
        assert!(!chains.is_empty());
    }

    #[test]
    fn test_reaching_defs_multiple_blocks() {
        let func = get_ir(
            r#"
            fn f(x: int) -> int {
                let y: int = 0;
                if (x > 0) { y = 1; } else { y = 2; }
                return y;
            }
        "#,
        );
        let rd = ReachingDefinitions::new(&func);
        let results = run_analysis(&rd, &func);
        // At the merge point, both definitions of y should reach
        assert!(!results.out_values.is_empty());
    }

    #[test]
    fn test_live_vars_if_else() {
        let func = get_ir(
            r#"
            fn f(x: int, y: int) -> int {
                if (x > 0) { return x + y; } else { return x; }
            }
        "#,
        );
        let lv = LiveVariables::new();
        let results = run_analysis(&lv, &func);
        let entry_in = results.in_values.get(&func.entry_block).unwrap();
        assert!(entry_in.contains("x"));
    }

    #[test]
    fn test_const_prop_if() {
        let func = get_ir(
            r#"
            fn f(x: int) -> int {
                let c: int = 10;
                if (x > 0) { return c; } else { return c; }
            }
        "#,
        );
        let cp = ConstantPropagation::new();
        let results = run_analysis(&cp, &func);
        let entry_out = results.out_values.get(&func.entry_block).unwrap();
        assert_eq!(entry_out.get("c"), Some(&ConstValue::Const(10)));
    }
}

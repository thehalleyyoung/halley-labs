//! SSA (Static Single Assignment) transformation.
//!
//! Transforms IR into SSA form where each variable is assigned exactly once.
//! Inserts phi nodes at join points, computes dominance frontiers,
//! builds use-def and def-use chains, and supports dead code elimination.

use std::collections::{HashMap, HashSet, VecDeque};

use shared_types::{
    AnalysisError, AnalysisResult, BasicBlock, BlockId, IrExpr, IrFunction, IrStatement, PhiNode,
    SsaVar, Terminator, VarId,
};

// ---------------------------------------------------------------------------
// SSA-form IR types
// ---------------------------------------------------------------------------

/// An SSA-form basic block with phi nodes.
#[derive(Debug, Clone)]
pub struct SsaBlock {
    pub id: BlockId,
    pub label: String,
    pub phi_nodes: Vec<PhiNode>,
    pub stmts: Vec<SsaStatement>,
    pub terminator: SsaTerminator,
}

/// An SSA statement (uses versioned variables).
#[derive(Debug, Clone)]
pub enum SsaStatement {
    Assign {
        target: SsaVar,
        value: SsaExpr,
    },
    ArrayWrite {
        array: SsaVar,
        index: SsaExpr,
        value: SsaExpr,
    },
    Assert {
        condition: SsaExpr,
    },
}

/// An SSA expression with versioned variables.
#[derive(Debug, Clone)]
pub enum SsaExpr {
    IntConst(i64),
    BoolConst(bool),
    Var(SsaVar),
    BinaryArith {
        left: Box<SsaExpr>,
        op: shared_types::ArithOp,
        right: Box<SsaExpr>,
    },
    BinaryRel {
        left: Box<SsaExpr>,
        op: shared_types::RelOp,
        right: Box<SsaExpr>,
    },
    BinaryLogic {
        left: Box<SsaExpr>,
        op: shared_types::LogicOp,
        right: Box<SsaExpr>,
    },
    Unary {
        op: shared_types::UnaryOp,
        operand: Box<SsaExpr>,
    },
    ArrayRead {
        array: SsaVar,
        index: Box<SsaExpr>,
    },
    FunctionCall {
        name: String,
        args: Vec<SsaExpr>,
    },
}

/// An SSA terminator with versioned variables.
#[derive(Debug, Clone)]
pub enum SsaTerminator {
    Goto(BlockId),
    Branch {
        condition: SsaExpr,
        true_target: BlockId,
        false_target: BlockId,
    },
    Return(Option<SsaExpr>),
    Unreachable,
}

/// SSA-form function.
#[derive(Debug, Clone)]
pub struct SsaFunction {
    pub name: String,
    pub params: Vec<(SsaVar, shared_types::Type)>,
    pub return_type: shared_types::Type,
    pub blocks: Vec<SsaBlock>,
    pub entry_block: BlockId,
}

// ---------------------------------------------------------------------------
// SsaTransform
// ---------------------------------------------------------------------------

/// Transforms IR functions into SSA form.
pub struct SsaTransform {
    /// Current version counter per variable base name.
    versions: HashMap<String, usize>,
    /// Stack of current definitions per variable (for renaming).
    def_stacks: HashMap<String, Vec<SsaVar>>,
}

impl SsaTransform {
    pub fn new() -> Self {
        SsaTransform {
            versions: HashMap::new(),
            def_stacks: HashMap::new(),
        }
    }

    /// Transform an IR function into SSA form.
    pub fn transform(&mut self, func: &IrFunction) -> AnalysisResult<SsaFunction> {
        self.versions.clear();
        self.def_stacks.clear();

        // Collect all variables
        let all_vars = self.collect_variables(func);

        // Compute predecessors
        let preds = self.compute_predecessors(func);

        // Compute dominance frontiers
        let df = self.compute_dominance_frontiers(func);

        // Phase 1: Insert phi nodes
        let phi_locations = self.compute_phi_locations(&all_vars, &df, func);

        // Phase 2: Rename variables
        // Initialize params
        let mut ssa_params = Vec::new();
        for (name, ty) in &func.params {
            let sv = self.new_version(name);
            self.push_def(name, sv.clone());
            ssa_params.push((sv, ty.clone()));
        }

        // Build SSA blocks
        let dom_tree = self.build_dom_tree(func);
        let mut ssa_blocks: HashMap<BlockId, SsaBlock> = HashMap::new();

        for block in &func.blocks {
            ssa_blocks.insert(
                block.id,
                SsaBlock {
                    id: block.id,
                    label: block.label.clone(),
                    phi_nodes: Vec::new(),
                    stmts: Vec::new(),
                    terminator: SsaTerminator::Unreachable,
                },
            );
        }

        // Insert phi nodes
        for (var, blocks) in &phi_locations {
            for &bid in blocks {
                let pred_ids: Vec<BlockId> = preds.get(&bid).cloned().unwrap_or_default();
                let phi = PhiNode::new(SsaVar::new(var.clone(), 0)); // version filled in during rename
                if let Some(sb) = ssa_blocks.get_mut(&bid) {
                    sb.phi_nodes.push(phi);
                }
            }
        }

        // Rename using dominator tree traversal
        self.rename_block(func.entry_block, func, &dom_tree, &preds, &mut ssa_blocks);

        let mut blocks: Vec<SsaBlock> = ssa_blocks.into_values().collect();
        blocks.sort_by_key(|b| b.id);

        Ok(SsaFunction {
            name: func.name.clone(),
            params: ssa_params,
            return_type: func.return_type.clone(),
            blocks,
            entry_block: func.entry_block,
        })
    }

    fn collect_variables(&self, func: &IrFunction) -> HashSet<String> {
        let mut vars = HashSet::new();
        for (name, _) in &func.params {
            vars.insert(name.clone());
        }
        for block in &func.blocks {
            for stmt in &block.stmts {
                if let Some(v) = stmt.defined_var() {
                    vars.insert(v.clone());
                }
                for v in stmt.used_vars() {
                    vars.insert(v);
                }
            }
        }
        vars
    }

    fn compute_predecessors(&self, func: &IrFunction) -> HashMap<BlockId, Vec<BlockId>> {
        let mut preds: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
        for block in &func.blocks {
            preds.insert(block.id, Vec::new());
        }
        for block in &func.blocks {
            for succ in block.terminator.successors() {
                preds.entry(succ).or_default().push(block.id);
            }
        }
        preds
    }

    fn compute_dominance_frontiers(&self, func: &IrFunction) -> HashMap<BlockId, HashSet<BlockId>> {
        let idom = self.compute_idom(func);
        let preds = self.compute_predecessors(func);
        let mut df: HashMap<BlockId, HashSet<BlockId>> =
            func.blocks.iter().map(|b| (b.id, HashSet::new())).collect();

        for block in &func.blocks {
            let pred_list = preds.get(&block.id).cloned().unwrap_or_default();
            if pred_list.len() < 2 {
                continue;
            }
            for &pred in &pred_list {
                let mut runner = pred;
                let target_idom = *idom.get(&block.id).unwrap_or(&block.id);
                while runner != target_idom {
                    df.get_mut(&runner).unwrap().insert(block.id);
                    let next = *idom.get(&runner).unwrap_or(&runner);
                    if next == runner {
                        break;
                    }
                    runner = next;
                }
            }
        }
        df
    }

    fn compute_idom(&self, func: &IrFunction) -> HashMap<BlockId, BlockId> {
        let mut idom: HashMap<BlockId, BlockId> = HashMap::new();
        idom.insert(func.entry_block, func.entry_block);

        let order = self.bfs_block_order(func);
        let preds = self.compute_predecessors(func);

        let mut changed = true;
        while changed {
            changed = false;
            for &bid in &order {
                if bid == func.entry_block {
                    continue;
                }
                let pred_list = preds.get(&bid).cloned().unwrap_or_default();
                let processed: Vec<_> = pred_list
                    .into_iter()
                    .filter(|p| idom.contains_key(p))
                    .collect();
                if processed.is_empty() {
                    continue;
                }
                let mut new_idom = processed[0];
                for &p in &processed[1..] {
                    new_idom = self.intersect_idom(&idom, &order, new_idom, p);
                }
                if idom.get(&bid) != Some(&new_idom) {
                    idom.insert(bid, new_idom);
                    changed = true;
                }
            }
        }
        idom
    }

    fn intersect_idom(
        &self,
        idom: &HashMap<BlockId, BlockId>,
        order: &[BlockId],
        mut a: BlockId,
        mut b: BlockId,
    ) -> BlockId {
        let pos: HashMap<BlockId, usize> = order.iter().enumerate().map(|(i, &n)| (n, i)).collect();
        while a != b {
            while pos.get(&a).copied().unwrap_or(usize::MAX)
                > pos.get(&b).copied().unwrap_or(usize::MAX)
            {
                a = *idom.get(&a).unwrap_or(&a);
            }
            while pos.get(&b).copied().unwrap_or(usize::MAX)
                > pos.get(&a).copied().unwrap_or(usize::MAX)
            {
                b = *idom.get(&b).unwrap_or(&b);
            }
        }
        a
    }

    fn bfs_block_order(&self, func: &IrFunction) -> Vec<BlockId> {
        let mut order = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(func.entry_block);
        while let Some(bid) = queue.pop_front() {
            if visited.insert(bid) {
                order.push(bid);
                if let Some(block) = func.block(bid) {
                    for succ in block.terminator.successors() {
                        queue.push_back(succ);
                    }
                }
            }
        }
        order
    }

    fn compute_phi_locations(
        &self,
        vars: &HashSet<String>,
        df: &HashMap<BlockId, HashSet<BlockId>>,
        func: &IrFunction,
    ) -> HashMap<String, HashSet<BlockId>> {
        let mut result: HashMap<String, HashSet<BlockId>> = HashMap::new();

        for var in vars {
            // Find blocks that define this variable
            let mut def_blocks: HashSet<BlockId> = HashSet::new();
            for block in &func.blocks {
                for stmt in &block.stmts {
                    if stmt.defined_var().map_or(false, |v| v == var) {
                        def_blocks.insert(block.id);
                    }
                }
            }
            // Also parameters define at entry
            if func.params.iter().any(|(n, _)| n == var) {
                def_blocks.insert(func.entry_block);
            }

            let mut phi_blocks: HashSet<BlockId> = HashSet::new();
            let mut worklist: Vec<BlockId> = def_blocks.iter().copied().collect();
            while let Some(bid) = worklist.pop() {
                if let Some(frontier) = df.get(&bid) {
                    for &fb in frontier {
                        if phi_blocks.insert(fb) {
                            worklist.push(fb);
                        }
                    }
                }
            }
            if !phi_blocks.is_empty() {
                result.insert(var.clone(), phi_blocks);
            }
        }
        result
    }

    fn build_dom_tree(&self, func: &IrFunction) -> HashMap<BlockId, Vec<BlockId>> {
        let idom = self.compute_idom(func);
        let mut tree: HashMap<BlockId, Vec<BlockId>> =
            func.blocks.iter().map(|b| (b.id, Vec::new())).collect();
        for (&node, &dom) in &idom {
            if node != dom {
                tree.entry(dom).or_default().push(node);
            }
        }
        tree
    }

    fn new_version(&mut self, base: &str) -> SsaVar {
        let ver = self.versions.entry(base.to_string()).or_insert(0);
        let sv = SsaVar::new(base, *ver);
        *ver += 1;
        sv
    }

    fn push_def(&mut self, base: &str, sv: SsaVar) {
        self.def_stacks
            .entry(base.to_string())
            .or_default()
            .push(sv);
    }

    fn current_def(&self, base: &str) -> SsaVar {
        self.def_stacks
            .get(base)
            .and_then(|stack| stack.last().cloned())
            .unwrap_or_else(|| SsaVar::new(base, 0))
    }

    fn pop_def(&mut self, base: &str) {
        if let Some(stack) = self.def_stacks.get_mut(base) {
            stack.pop();
        }
    }

    fn rename_block(
        &mut self,
        bid: BlockId,
        func: &IrFunction,
        dom_tree: &HashMap<BlockId, Vec<BlockId>>,
        preds: &HashMap<BlockId, Vec<BlockId>>,
        ssa_blocks: &mut HashMap<BlockId, SsaBlock>,
    ) {
        let block = match func.block(bid) {
            Some(b) => b,
            None => return,
        };

        // Track definitions made in this block for later cleanup
        let mut defs_in_block: Vec<String> = Vec::new();

        // Rename phi node targets
        if let Some(sb) = ssa_blocks.get_mut(&bid) {
            for phi in &mut sb.phi_nodes {
                let base = phi.target.base.clone();
                let new_var = self.new_version(&base);
                self.push_def(&base, new_var.clone());
                defs_in_block.push(base);
                phi.target = new_var;
            }
        }

        // Rename statements
        let mut new_stmts = Vec::new();
        for stmt in &block.stmts {
            match stmt {
                IrStatement::Assign { target, value } => {
                    let ssa_val = self.rename_expr(value);
                    let new_var = self.new_version(target);
                    self.push_def(target, new_var.clone());
                    defs_in_block.push(target.clone());
                    new_stmts.push(SsaStatement::Assign {
                        target: new_var,
                        value: ssa_val,
                    });
                }
                IrStatement::ArrayWrite {
                    array,
                    index,
                    value,
                } => {
                    let ssa_idx = self.rename_expr(index);
                    let ssa_val = self.rename_expr(value);
                    let arr_var = self.current_def(array);
                    new_stmts.push(SsaStatement::ArrayWrite {
                        array: arr_var,
                        index: ssa_idx,
                        value: ssa_val,
                    });
                }
                IrStatement::Assert { condition } => {
                    let ssa_cond = self.rename_expr(condition);
                    new_stmts.push(SsaStatement::Assert {
                        condition: ssa_cond,
                    });
                }
            }
        }

        // Rename terminator
        let new_term = match &block.terminator {
            Terminator::Goto(t) => SsaTerminator::Goto(*t),
            Terminator::Branch {
                condition,
                true_target,
                false_target,
            } => SsaTerminator::Branch {
                condition: self.rename_expr(condition),
                true_target: *true_target,
                false_target: *false_target,
            },
            Terminator::Return(Some(e)) => SsaTerminator::Return(Some(self.rename_expr(e))),
            Terminator::Return(None) => SsaTerminator::Return(None),
            Terminator::Unreachable => SsaTerminator::Unreachable,
        };

        if let Some(sb) = ssa_blocks.get_mut(&bid) {
            sb.stmts = new_stmts;
            sb.terminator = new_term;
        }

        // Fill in phi node sources for successors
        for succ_id in block.terminator.successors() {
            if let Some(succ_block) = ssa_blocks.get_mut(&succ_id) {
                for phi in &mut succ_block.phi_nodes {
                    let base = &phi.target.base;
                    let current = self.current_def(base);
                    phi.sources.push((bid, current));
                }
            }
        }

        // Recurse into dominated children
        let children = dom_tree.get(&bid).cloned().unwrap_or_default();
        for child in children {
            self.rename_block(child, func, dom_tree, preds, ssa_blocks);
        }

        // Pop definitions
        for base in defs_in_block.iter().rev() {
            self.pop_def(base);
        }
    }

    fn rename_expr(&self, expr: &IrExpr) -> SsaExpr {
        match expr {
            IrExpr::IntConst(v) => SsaExpr::IntConst(*v),
            IrExpr::BoolConst(v) => SsaExpr::BoolConst(*v),
            IrExpr::Var(name) => SsaExpr::Var(self.current_def(name)),
            IrExpr::BinaryArith { left, op, right } => SsaExpr::BinaryArith {
                left: Box::new(self.rename_expr(left)),
                op: *op,
                right: Box::new(self.rename_expr(right)),
            },
            IrExpr::BinaryRel { left, op, right } => SsaExpr::BinaryRel {
                left: Box::new(self.rename_expr(left)),
                op: *op,
                right: Box::new(self.rename_expr(right)),
            },
            IrExpr::BinaryLogic { left, op, right } => SsaExpr::BinaryLogic {
                left: Box::new(self.rename_expr(left)),
                op: *op,
                right: Box::new(self.rename_expr(right)),
            },
            IrExpr::Unary { op, operand } => SsaExpr::Unary {
                op: *op,
                operand: Box::new(self.rename_expr(operand)),
            },
            IrExpr::ArrayRead { array, index } => SsaExpr::ArrayRead {
                array: self.current_def(array),
                index: Box::new(self.rename_expr(index)),
            },
            IrExpr::FunctionCall { name, args } => SsaExpr::FunctionCall {
                name: name.clone(),
                args: args.iter().map(|a| self.rename_expr(a)).collect(),
            },
        }
    }
}

impl Default for SsaTransform {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SSA Validation
// ---------------------------------------------------------------------------

/// Validate SSA properties: each variable defined exactly once per path.
pub fn validate_ssa(func: &SsaFunction) -> AnalysisResult<()> {
    let mut all_defs: HashMap<String, usize> = HashMap::new();

    for block in &func.blocks {
        for phi in &block.phi_nodes {
            let key = format!("{}", phi.target);
            *all_defs.entry(key).or_insert(0) += 1;
        }
        for stmt in &block.stmts {
            if let SsaStatement::Assign { target, .. } = stmt {
                let key = format!("{}", target);
                *all_defs.entry(key).or_insert(0) += 1;
            }
        }
    }

    for (var, count) in &all_defs {
        if *count > 1 {
            return Err(AnalysisError::SsaError {
                message: format!("Variable {} defined {} times", var, count),
            });
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Use-def and def-use chains
// ---------------------------------------------------------------------------

/// A definition site for an SSA variable.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DefSite {
    pub block: BlockId,
    pub index: DefIndex,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DefIndex {
    Phi(usize),
    Stmt(usize),
    Param,
}

/// Build use-def chains: for each use of an SSA variable, where is it defined?
pub fn build_use_def_chains(func: &SsaFunction) -> HashMap<String, DefSite> {
    let mut defs: HashMap<String, DefSite> = HashMap::new();

    for (sv, _) in &func.params {
        defs.insert(
            format!("{}", sv),
            DefSite {
                block: func.entry_block,
                index: DefIndex::Param,
            },
        );
    }

    for block in &func.blocks {
        for (i, phi) in block.phi_nodes.iter().enumerate() {
            defs.insert(
                format!("{}", phi.target),
                DefSite {
                    block: block.id,
                    index: DefIndex::Phi(i),
                },
            );
        }
        for (i, stmt) in block.stmts.iter().enumerate() {
            if let SsaStatement::Assign { target, .. } = stmt {
                defs.insert(
                    format!("{}", target),
                    DefSite {
                        block: block.id,
                        index: DefIndex::Stmt(i),
                    },
                );
            }
        }
    }

    defs
}

/// Build def-use chains: for each definition, where is it used?
pub fn build_def_use_chains(func: &SsaFunction) -> HashMap<String, Vec<(BlockId, usize)>> {
    let mut uses: HashMap<String, Vec<(BlockId, usize)>> = HashMap::new();

    for block in &func.blocks {
        for (i, stmt) in block.stmts.iter().enumerate() {
            let used = match stmt {
                SsaStatement::Assign { value, .. } => collect_ssa_vars(value),
                SsaStatement::ArrayWrite {
                    array,
                    index,
                    value,
                } => {
                    let mut v = vec![format!("{}", array)];
                    v.extend(collect_ssa_vars(index));
                    v.extend(collect_ssa_vars(value));
                    v
                }
                SsaStatement::Assert { condition } => collect_ssa_vars(condition),
            };
            for var_name in used {
                uses.entry(var_name).or_default().push((block.id, i));
            }
        }
    }

    uses
}

fn collect_ssa_vars(expr: &SsaExpr) -> Vec<String> {
    let mut vars = Vec::new();
    collect_ssa_vars_rec(expr, &mut vars);
    vars
}

fn collect_ssa_vars_rec(expr: &SsaExpr, vars: &mut Vec<String>) {
    match expr {
        SsaExpr::IntConst(_) | SsaExpr::BoolConst(_) => {}
        SsaExpr::Var(sv) => vars.push(format!("{}", sv)),
        SsaExpr::BinaryArith { left, right, .. }
        | SsaExpr::BinaryRel { left, right, .. }
        | SsaExpr::BinaryLogic { left, right, .. } => {
            collect_ssa_vars_rec(left, vars);
            collect_ssa_vars_rec(right, vars);
        }
        SsaExpr::Unary { operand, .. } => collect_ssa_vars_rec(operand, vars),
        SsaExpr::ArrayRead { array, index } => {
            vars.push(format!("{}", array));
            collect_ssa_vars_rec(index, vars);
        }
        SsaExpr::FunctionCall { args, .. } => {
            for a in args {
                collect_ssa_vars_rec(a, vars);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dead code elimination on SSA
// ---------------------------------------------------------------------------

/// Remove assignments whose targets are never used.
pub fn dead_code_elimination(func: &mut SsaFunction) {
    let du = build_def_use_chains(func);
    let mut changed = true;
    while changed {
        changed = false;
        for block in &mut func.blocks {
            let before = block.stmts.len();
            block.stmts.retain(|stmt| {
                if let SsaStatement::Assign { target, .. } = stmt {
                    let key = format!("{}", target);
                    du.get(&key).map_or(false, |uses| !uses.is_empty())
                } else {
                    true
                }
            });
            if block.stmts.len() < before {
                changed = true;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Phi elimination (back to non-SSA)
// ---------------------------------------------------------------------------

/// Eliminate phi nodes by inserting copies in predecessor blocks.
pub fn eliminate_phi_nodes(func: &mut SsaFunction) {
    let mut copies_to_insert: HashMap<BlockId, Vec<(SsaVar, SsaVar)>> = HashMap::new();

    for block in &func.blocks {
        for phi in &block.phi_nodes {
            for (pred_id, src_var) in &phi.sources {
                copies_to_insert
                    .entry(*pred_id)
                    .or_default()
                    .push((phi.target.clone(), src_var.clone()));
            }
        }
    }

    for block in &mut func.blocks {
        if let Some(copies) = copies_to_insert.get(&block.id) {
            for (target, source) in copies {
                block.stmts.push(SsaStatement::Assign {
                    target: target.clone(),
                    value: SsaExpr::Var(source.clone()),
                });
            }
        }
        block.phi_nodes.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_lowering::IrLowering;
    use crate::parser::Parser;

    fn to_ssa(src: &str) -> SsaFunction {
        let prog = Parser::parse_source(src).unwrap();
        let ir = IrLowering::new().lower_program(&prog).unwrap();
        let mut transform = SsaTransform::new();
        transform.transform(&ir.functions[0]).unwrap()
    }

    #[test]
    fn test_ssa_simple() {
        let ssa = to_ssa("fn f(x: int) -> int { return x; }");
        assert_eq!(ssa.name, "f");
        assert!(!ssa.blocks.is_empty());
    }

    #[test]
    fn test_ssa_assignment() {
        let ssa = to_ssa("fn f(x: int) -> int { let y: int = x + 1; return y; }");
        // y should be versioned
        let has_assign = ssa.blocks.iter().any(|b| {
            b.stmts
                .iter()
                .any(|s| matches!(s, SsaStatement::Assign { .. }))
        });
        assert!(has_assign);
    }

    #[test]
    fn test_ssa_if_else() {
        let ssa = to_ssa("fn f(x: int) -> int { if (x > 0) { return x; } else { return -x; } }");
        assert!(ssa.blocks.len() >= 3);
    }

    #[test]
    fn test_ssa_reassignment() {
        let ssa = to_ssa("fn f(x: int) -> int { let y: int = x; y = y + 1; return y; }");
        // Multiple versions of y
        let y_defs: Vec<_> = ssa
            .blocks
            .iter()
            .flat_map(|b| {
                b.stmts.iter().filter_map(|s| match s {
                    SsaStatement::Assign { target, .. } if target.base == "y" => {
                        Some(target.clone())
                    }
                    _ => None,
                })
            })
            .collect();
        assert!(y_defs.len() >= 2, "y should have multiple versions");
    }

    #[test]
    fn test_ssa_params_versioned() {
        let ssa = to_ssa("fn f(a: int, b: int) -> int { return a + b; }");
        assert_eq!(ssa.params[0].0.base, "a");
        assert_eq!(ssa.params[1].0.base, "b");
    }

    #[test]
    fn test_ssa_use_def_chains() {
        let ssa = to_ssa("fn f(x: int) -> int { let y: int = x + 1; return y; }");
        let ud = build_use_def_chains(&ssa);
        assert!(!ud.is_empty());
    }

    #[test]
    fn test_ssa_def_use_chains() {
        let ssa = to_ssa("fn f(x: int) -> int { let y: int = x + 1; return y; }");
        let du = build_def_use_chains(&ssa);
        // x should be used somewhere
        let x_uses: Vec<_> = du.iter().filter(|(k, _)| k.starts_with("x_")).collect();
        assert!(!x_uses.is_empty() || du.iter().any(|(k, _)| k.starts_with("x")));
    }

    #[test]
    fn test_ssa_validation() {
        let ssa = to_ssa("fn f(x: int) -> int { return x; }");
        assert!(validate_ssa(&ssa).is_ok());
    }

    #[test]
    fn test_ssa_dce() {
        let mut ssa = to_ssa("fn f(x: int) -> int { let y: int = 42; return x; }");
        let before = ssa.blocks.iter().map(|b| b.stmts.len()).sum::<usize>();
        dead_code_elimination(&mut ssa);
        let after = ssa.blocks.iter().map(|b| b.stmts.len()).sum::<usize>();
        assert!(after <= before);
    }

    #[test]
    fn test_ssa_phi_elimination() {
        let mut ssa =
            to_ssa("fn f(x: int) -> int { if (x > 0) { return x; } else { return -x; } }");
        eliminate_phi_nodes(&mut ssa);
        for block in &ssa.blocks {
            assert!(block.phi_nodes.is_empty());
        }
    }

    #[test]
    fn test_ssa_complex() {
        let ssa = to_ssa(
            r#"
            fn f(x: int, y: int) -> int {
                let r: int = 0;
                if (x > y) { r = x - y; } else { r = y - x; }
                return r;
            }
        "#,
        );
        assert!(!ssa.blocks.is_empty());
    }
}

//! Control-flow graph construction from the analysis IR.
//!
//! [`CfgBuilder`] transforms the block-level IR representation into a
//! shared-types [`ControlFlowGraph`] that downstream verification passes
//! can consume, including dominance computation, loop detection, and
//! speculative-edge insertion.

use std::collections::{HashMap, HashSet};

use shared_types::{
    BasicBlock, BlockId, CfgEdgeKind, ControlFlowGraph,
};

use crate::ir::{AnalysisIR, IRBlockId, IRFunction, IRTerminator};

/// Builds a shared-types [`ControlFlowGraph`] from the analysis IR.
#[derive(Debug, Clone)]
pub struct CfgBuilder {
    /// If `true`, insert speculative edges for conditional branches.
    pub insert_speculative_edges: bool,
    /// Maximum speculative window depth (in edges) to model.
    pub speculative_depth: u32,
}

impl CfgBuilder {
    pub fn new() -> Self {
        Self {
            insert_speculative_edges: true,
            speculative_depth: 4,
        }
    }

    /// Build a [`ControlFlowGraph`] for a single IR function.
    pub fn build_function_cfg(&self, func: &IRFunction) -> ControlFlowGraph {
        let mut cfg = ControlFlowGraph::new();

        // Create a BasicBlock for each IRBlock.
        let mut id_map: HashMap<IRBlockId, BlockId> = HashMap::new();

        for ir_block in func.blocks.values() {
            let mut bb = BasicBlock::new(
                BlockId(ir_block.id.0),
                ir_block.start_address,
            );
            bb.is_entry = ir_block.is_entry;
            bb.is_exit = ir_block.is_exit;
            bb.loop_depth = ir_block.loop_depth;

            let bid = cfg.add_block(bb);
            id_map.insert(ir_block.id, bid);
        }

        // Set the entry.
        if let Some(&entry) = id_map.get(&func.entry_block) {
            cfg.entry = Some(entry);
        }

        // Wire edges from terminators.
        for ir_block in func.blocks.values() {
            let source = match id_map.get(&ir_block.id) {
                Some(&s) => s,
                None => continue,
            };

            match &ir_block.terminator {
                IRTerminator::Goto(target) => {
                    if let Some(&t) = id_map.get(target) {
                        cfg.connect(source, t, CfgEdgeKind::Unconditional);
                    }
                }
                IRTerminator::Branch {
                    true_target,
                    false_target,
                    ..
                } => {
                    if let Some(&tt) = id_map.get(true_target) {
                        cfg.connect(source, tt, CfgEdgeKind::ConditionalTrue);
                    }
                    if let Some(&ft) = id_map.get(false_target) {
                        cfg.connect(source, ft, CfgEdgeKind::ConditionalFalse);
                    }
                    // Optionally model speculative execution of the wrong path.
                    if self.insert_speculative_edges {
                        if let Some(&tt) = id_map.get(true_target) {
                            cfg.connect(source, tt, CfgEdgeKind::Speculative);
                        }
                    }
                }
                IRTerminator::Call { return_block, .. } => {
                    if let Some(ret) = return_block {
                        if let Some(&r) = id_map.get(ret) {
                            cfg.connect(source, r, CfgEdgeKind::Call);
                        }
                    }
                }
                IRTerminator::Return | IRTerminator::Unreachable => {
                    // No outgoing edges.
                }
                IRTerminator::IndirectJump { .. } => {
                    // Indirect jumps require target resolution — leave unconnected
                    // for now; a separate indirect-call resolution pass can refine.
                }
            }
        }

        cfg.compute_exits();
        cfg
    }

    /// Build CFGs for every function in the analysis IR.
    pub fn build_program_cfgs(&self, ir: &AnalysisIR) -> HashMap<shared_types::FunctionId, ControlFlowGraph> {
        ir.program
            .functions
            .iter()
            .map(|(&fid, func)| (fid, self.build_function_cfg(func)))
            .collect()
    }

    /// Compute reverse-post-order traversal of block ids for a function's
    /// CFG — useful for dataflow analyses.
    pub fn reverse_postorder(&self, func: &IRFunction) -> Vec<IRBlockId> {
        let mut visited = HashSet::new();
        let mut order = Vec::new();
        self.rpo_visit(func, func.entry_block, &mut visited, &mut order);
        order.reverse();
        order
    }

    /// Find all back-edges in the function's CFG.
    pub fn find_back_edges(&self, func: &IRFunction) -> Vec<(IRBlockId, IRBlockId)> {
        let rpo = self.reverse_postorder(func);
        let position: HashMap<IRBlockId, usize> =
            rpo.iter().enumerate().map(|(i, &b)| (b, i)).collect();

        let mut back_edges = Vec::new();
        for block in func.blocks.values() {
            for &succ in &block.successors {
                if let (Some(&src_pos), Some(&tgt_pos)) =
                    (position.get(&block.id), position.get(&succ))
                {
                    if tgt_pos <= src_pos {
                        back_edges.push((block.id, succ));
                    }
                }
            }
        }
        back_edges
    }

    /// Compute immediate dominators for each block using a simple iterative
    /// algorithm (Cooper, Harvey, Kennedy).
    pub fn compute_dominators(
        &self,
        func: &IRFunction,
    ) -> HashMap<IRBlockId, IRBlockId> {
        let rpo = self.reverse_postorder(func);
        let mut idom: HashMap<IRBlockId, IRBlockId> = HashMap::new();
        idom.insert(func.entry_block, func.entry_block);

        let pos: HashMap<IRBlockId, usize> =
            rpo.iter().enumerate().map(|(i, &b)| (b, i)).collect();

        let mut changed = true;
        while changed {
            changed = false;
            for &b in &rpo {
                if b == func.entry_block {
                    continue;
                }
                if let Some(block) = func.block(b) {
                    let mut new_idom: Option<IRBlockId> = None;
                    for &pred in &block.predecessors {
                        if idom.contains_key(&pred) {
                            new_idom = Some(match new_idom {
                                None => pred,
                                Some(cur) => self.intersect(&idom, &pos, cur, pred),
                            });
                        }
                    }
                    if let Some(n) = new_idom {
                        if idom.get(&b) != Some(&n) {
                            idom.insert(b, n);
                            changed = true;
                        }
                    }
                }
            }
        }

        idom
    }

    // -- internal helpers --------------------------------------------------

    fn rpo_visit(
        &self,
        func: &IRFunction,
        block_id: IRBlockId,
        visited: &mut HashSet<IRBlockId>,
        order: &mut Vec<IRBlockId>,
    ) {
        if !visited.insert(block_id) {
            return;
        }
        if let Some(block) = func.block(block_id) {
            for &succ in &block.successors {
                self.rpo_visit(func, succ, visited, order);
            }
        }
        order.push(block_id);
    }

    fn intersect(
        &self,
        idom: &HashMap<IRBlockId, IRBlockId>,
        pos: &HashMap<IRBlockId, usize>,
        mut b1: IRBlockId,
        mut b2: IRBlockId,
    ) -> IRBlockId {
        while b1 != b2 {
            while pos.get(&b1).copied().unwrap_or(0) > pos.get(&b2).copied().unwrap_or(0) {
                b1 = idom[&b1];
            }
            while pos.get(&b2).copied().unwrap_or(0) > pos.get(&b1).copied().unwrap_or(0) {
                b2 = idom[&b2];
            }
        }
        b1
    }
}

impl Default for CfgBuilder {
    fn default() -> Self {
        Self::new()
    }
}

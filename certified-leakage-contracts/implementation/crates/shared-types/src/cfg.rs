//! Control flow graph representation.

use serde::{Deserialize, Serialize};
use crate::address::VirtualAddress;
use crate::instruction::Instruction;
use std::collections::{HashMap, HashSet, BTreeMap};
use std::fmt;

/// Unique identifier for a basic block within a function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BlockId(pub u32);

impl BlockId {
    pub fn new(id: u32) -> Self { Self(id) }
    pub fn as_usize(self) -> usize { self.0 as usize }
    pub fn entry() -> Self { Self(0) }
}

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

/// Kind of control flow edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CfgEdgeKind {
    /// Unconditional fallthrough or jump.
    Unconditional,
    /// True branch of a conditional.
    ConditionalTrue,
    /// False branch of a conditional.
    ConditionalFalse,
    /// Function call edge (to callee entry).
    Call,
    /// Return edge (back to caller).
    Return,
    /// Speculative edge (branch misprediction).
    Speculative,
    /// Back-edge in a loop.
    BackEdge,
    /// Exception / signal edge.
    Exception,
}

impl CfgEdgeKind {
    pub fn is_speculative(self) -> bool { self == Self::Speculative }
    pub fn is_conditional(self) -> bool {
        matches!(self, Self::ConditionalTrue | Self::ConditionalFalse)
    }
}

/// An edge in the control flow graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfgEdge {
    pub source: BlockId,
    pub target: BlockId,
    pub kind: CfgEdgeKind,
    pub is_back_edge: bool,
}

impl CfgEdge {
    pub fn new(source: BlockId, target: BlockId, kind: CfgEdgeKind) -> Self {
        Self { source, target, kind, is_back_edge: false }
    }

    pub fn with_back_edge(mut self) -> Self {
        self.is_back_edge = true;
        self
    }
}

/// A basic block in the control flow graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicBlock {
    pub id: BlockId,
    pub start_address: VirtualAddress,
    pub end_address: VirtualAddress,
    pub instructions: Vec<Instruction>,
    pub successors: Vec<BlockId>,
    pub predecessors: Vec<BlockId>,
    pub is_entry: bool,
    pub is_exit: bool,
    pub is_loop_header: bool,
    pub loop_depth: u32,
}

impl BasicBlock {
    pub fn new(id: BlockId, start: VirtualAddress) -> Self {
        Self {
            id,
            start_address: start,
            end_address: start,
            instructions: Vec::new(),
            successors: Vec::new(),
            predecessors: Vec::new(),
            is_entry: false,
            is_exit: false,
            is_loop_header: false,
            loop_depth: 0,
        }
    }

    pub fn add_instruction(&mut self, instr: Instruction) {
        if self.instructions.is_empty() {
            self.start_address = instr.address;
        }
        self.end_address = VirtualAddress(instr.address.0 + instr.length as u64);
        self.instructions.push(instr);
    }

    pub fn instruction_count(&self) -> usize {
        self.instructions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }

    pub fn last_instruction(&self) -> Option<&Instruction> {
        self.instructions.last()
    }

    pub fn first_instruction(&self) -> Option<&Instruction> {
        self.instructions.first()
    }

    pub fn contains_address(&self, addr: VirtualAddress) -> bool {
        addr.0 >= self.start_address.0 && addr.0 < self.end_address.0
    }

    pub fn has_memory_access(&self) -> bool {
        self.instructions.iter().any(|i| {
            i.operands.iter().any(|op| matches!(op.kind, crate::operand::OperandKind::Memory(_)))
        })
    }

    pub fn memory_access_count(&self) -> usize {
        self.instructions.iter()
            .flat_map(|i| i.operands.iter())
            .filter(|op| matches!(op.kind, crate::operand::OperandKind::Memory(_)))
            .count()
    }
}

impl fmt::Display for BasicBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{} ({} instrs)", self.id, self.start_address, self.instruction_count())
    }
}

/// A control flow graph for a single function.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ControlFlowGraph {
    pub blocks: BTreeMap<BlockId, BasicBlock>,
    pub edges: Vec<CfgEdge>,
    pub entry: Option<BlockId>,
    pub exits: Vec<BlockId>,
    next_block_id: u32,
}

impl ControlFlowGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_block(&mut self, mut block: BasicBlock) -> BlockId {
        let id = BlockId(self.next_block_id);
        self.next_block_id += 1;
        block.id = id;
        if self.blocks.is_empty() {
            block.is_entry = true;
            self.entry = Some(id);
        }
        self.blocks.insert(id, block);
        id
    }

    pub fn add_edge(&mut self, edge: CfgEdge) {
        if let Some(src) = self.blocks.get_mut(&edge.source) {
            if !src.successors.contains(&edge.target) {
                src.successors.push(edge.target);
            }
        }
        if let Some(tgt) = self.blocks.get_mut(&edge.target) {
            if !tgt.predecessors.contains(&edge.source) {
                tgt.predecessors.push(edge.source);
            }
        }
        self.edges.push(edge);
    }

    pub fn connect(&mut self, from: BlockId, to: BlockId, kind: CfgEdgeKind) {
        self.add_edge(CfgEdge::new(from, to, kind));
    }

    pub fn block(&self, id: BlockId) -> Option<&BasicBlock> {
        self.blocks.get(&id)
    }

    pub fn block_mut(&mut self, id: BlockId) -> Option<&mut BasicBlock> {
        self.blocks.get_mut(&id)
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn total_instructions(&self) -> usize {
        self.blocks.values().map(|b| b.instruction_count()).sum()
    }

    pub fn entry_block(&self) -> Option<&BasicBlock> {
        self.entry.and_then(|id| self.blocks.get(&id))
    }

    pub fn successors(&self, id: BlockId) -> Vec<BlockId> {
        self.blocks.get(&id).map_or(Vec::new(), |b| b.successors.clone())
    }

    pub fn predecessors(&self, id: BlockId) -> Vec<BlockId> {
        self.blocks.get(&id).map_or(Vec::new(), |b| b.predecessors.clone())
    }

    /// Returns blocks in reverse postorder (good for forward dataflow).
    pub fn reverse_postorder(&self) -> Vec<BlockId> {
        let entry = match self.entry {
            Some(e) => e,
            None => return Vec::new(),
        };
        let mut visited = HashSet::new();
        let mut order = Vec::new();
        self.rpo_visit(entry, &mut visited, &mut order);
        order.reverse();
        order
    }

    fn rpo_visit(&self, id: BlockId, visited: &mut HashSet<BlockId>, order: &mut Vec<BlockId>) {
        if !visited.insert(id) { return; }
        for &succ in self.blocks.get(&id).map_or(&vec![], |b| &b.successors) {
            self.rpo_visit(succ, visited, order);
        }
        order.push(id);
    }

    /// Returns blocks in postorder (good for backward dataflow).
    pub fn postorder(&self) -> Vec<BlockId> {
        let entry = match self.entry {
            Some(e) => e,
            None => return Vec::new(),
        };
        let mut visited = HashSet::new();
        let mut order = Vec::new();
        self.rpo_visit(entry, &mut visited, &mut order);
        order
    }

    /// Identify loop headers using a simple dominance-based approach.
    pub fn find_loop_headers(&self) -> Vec<BlockId> {
        let rpo = self.reverse_postorder();
        let rpo_index: HashMap<BlockId, usize> = rpo.iter().enumerate().map(|(i, &b)| (b, i)).collect();
        let mut headers = Vec::new();
        for edge in &self.edges {
            if let (Some(&src_idx), Some(&tgt_idx)) = (rpo_index.get(&edge.source), rpo_index.get(&edge.target)) {
                if tgt_idx <= src_idx {
                    headers.push(edge.target);
                }
            }
        }
        headers.sort();
        headers.dedup();
        headers
    }

    /// Mark exit blocks (blocks with no successors or with return edges).
    pub fn compute_exits(&mut self) {
        let exit_ids: Vec<BlockId> = self.blocks.values()
            .filter(|b| b.successors.is_empty())
            .map(|b| b.id)
            .collect();
        self.exits = exit_ids.clone();
        for id in exit_ids {
            if let Some(b) = self.blocks.get_mut(&id) {
                b.is_exit = true;
            }
        }
    }

    /// Compute dominators using the Cooper-Harvey-Kennedy algorithm.
    pub fn compute_dominators(&self) -> HashMap<BlockId, BlockId> {
        let rpo = self.reverse_postorder();
        if rpo.is_empty() { return HashMap::new(); }

        let entry = rpo[0];
        let rpo_num: HashMap<BlockId, usize> = rpo.iter().enumerate().map(|(i, &b)| (b, i)).collect();
        let mut doms: HashMap<BlockId, BlockId> = HashMap::new();
        doms.insert(entry, entry);

        let intersect = |doms: &HashMap<BlockId, BlockId>, rpo_num: &HashMap<BlockId, usize>, mut a: BlockId, mut b: BlockId| -> BlockId {
            while a != b {
                while rpo_num.get(&a).copied().unwrap_or(usize::MAX) > rpo_num.get(&b).copied().unwrap_or(usize::MAX) {
                    a = *doms.get(&a).unwrap_or(&a);
                }
                while rpo_num.get(&b).copied().unwrap_or(usize::MAX) > rpo_num.get(&a).copied().unwrap_or(usize::MAX) {
                    b = *doms.get(&b).unwrap_or(&b);
                }
            }
            a
        };

        let mut changed = true;
        while changed {
            changed = false;
            for &b in rpo.iter().skip(1) {
                let preds = self.predecessors(b);
                let mut new_idom = None;
                for p in &preds {
                    if doms.contains_key(p) {
                        new_idom = Some(match new_idom {
                            None => *p,
                            Some(cur) => intersect(&doms, &rpo_num, cur, *p),
                        });
                    }
                }
                if let Some(idom) = new_idom {
                    if doms.get(&b) != Some(&idom) {
                        doms.insert(b, idom);
                        changed = true;
                    }
                }
            }
        }
        doms
    }

    /// Compute the dominance frontier for each block.
    pub fn dominance_frontier(&self) -> HashMap<BlockId, HashSet<BlockId>> {
        let doms = self.compute_dominators();
        let mut df: HashMap<BlockId, HashSet<BlockId>> = HashMap::new();
        for (&b, _) in &self.blocks {
            df.insert(b, HashSet::new());
        }
        for (&b, block) in &self.blocks {
            if block.predecessors.len() >= 2 {
                for &p in &block.predecessors {
                    let mut runner = p;
                    while runner != *doms.get(&b).unwrap_or(&b) {
                        df.entry(runner).or_default().insert(b);
                        runner = *doms.get(&runner).unwrap_or(&runner);
                        if runner == *doms.get(&runner).unwrap_or(&runner) && runner != *doms.get(&b).unwrap_or(&b) {
                            break;
                        }
                    }
                }
            }
        }
        df
    }

    /// Check if the CFG is reducible (all back edges lead to dominators).
    pub fn is_reducible(&self) -> bool {
        let doms = self.compute_dominators();
        let rpo = self.reverse_postorder();
        let rpo_index: HashMap<BlockId, usize> = rpo.iter().enumerate().map(|(i, &b)| (b, i)).collect();

        for edge in &self.edges {
            if let (Some(&src_idx), Some(&tgt_idx)) = (rpo_index.get(&edge.source), rpo_index.get(&edge.target)) {
                if tgt_idx <= src_idx {
                    // Back edge: target must dominate source
                    let mut runner = edge.source;
                    let mut dominates = false;
                    loop {
                        if runner == edge.target {
                            dominates = true;
                            break;
                        }
                        match doms.get(&runner) {
                            Some(&d) if d != runner => runner = d,
                            _ => break,
                        }
                    }
                    if !dominates { return false; }
                }
            }
        }
        true
    }

    /// Get strongly connected components (natural loops).
    pub fn natural_loops(&self) -> Vec<Vec<BlockId>> {
        let headers = self.find_loop_headers();
        let mut loops = Vec::new();
        for header in headers {
            let mut body = HashSet::new();
            body.insert(header);
            // Find back edges to this header
            let back_sources: Vec<BlockId> = self.edges.iter()
                .filter(|e| e.target == header)
                .filter(|e| {
                    // Check if it's a back edge
                    let rpo = self.reverse_postorder();
                    let rpo_idx: HashMap<BlockId, usize> = rpo.iter().enumerate().map(|(i, &b)| (b, i)).collect();
                    rpo_idx.get(&e.source).copied().unwrap_or(0) >= rpo_idx.get(&e.target).copied().unwrap_or(0)
                })
                .map(|e| e.source)
                .collect();

            let mut worklist: Vec<BlockId> = back_sources;
            while let Some(node) = worklist.pop() {
                if body.insert(node) {
                    for &pred in self.blocks.get(&node).map_or(&vec![], |b| &b.predecessors) {
                        if !body.contains(&pred) {
                            worklist.push(pred);
                        }
                    }
                }
            }
            let mut body_vec: Vec<BlockId> = body.into_iter().collect();
            body_vec.sort();
            loops.push(body_vec);
        }
        loops
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_diamond_cfg() -> ControlFlowGraph {
        let mut cfg = ControlFlowGraph::new();
        let b0 = cfg.add_block(BasicBlock::new(BlockId(0), VirtualAddress(0x1000)));
        let b1 = cfg.add_block(BasicBlock::new(BlockId(1), VirtualAddress(0x1010)));
        let b2 = cfg.add_block(BasicBlock::new(BlockId(2), VirtualAddress(0x1020)));
        let b3 = cfg.add_block(BasicBlock::new(BlockId(3), VirtualAddress(0x1030)));
        cfg.connect(b0, b1, CfgEdgeKind::ConditionalTrue);
        cfg.connect(b0, b2, CfgEdgeKind::ConditionalFalse);
        cfg.connect(b1, b3, CfgEdgeKind::Unconditional);
        cfg.connect(b2, b3, CfgEdgeKind::Unconditional);
        cfg.compute_exits();
        cfg
    }

    fn make_loop_cfg() -> ControlFlowGraph {
        let mut cfg = ControlFlowGraph::new();
        let b0 = cfg.add_block(BasicBlock::new(BlockId(0), VirtualAddress(0x2000)));
        let b1 = cfg.add_block(BasicBlock::new(BlockId(1), VirtualAddress(0x2010)));
        let b2 = cfg.add_block(BasicBlock::new(BlockId(2), VirtualAddress(0x2020)));
        cfg.connect(b0, b1, CfgEdgeKind::Unconditional);
        cfg.connect(b1, b2, CfgEdgeKind::ConditionalTrue);
        cfg.connect(b1, b1, CfgEdgeKind::ConditionalFalse); // self-loop / back-edge
        cfg.compute_exits();
        cfg
    }

    #[test]
    fn test_diamond_rpo() {
        let cfg = make_diamond_cfg();
        let rpo = cfg.reverse_postorder();
        assert_eq!(rpo.len(), 4);
        assert_eq!(rpo[0], BlockId(0));
    }

    #[test]
    fn test_diamond_dominators() {
        let cfg = make_diamond_cfg();
        let doms = cfg.compute_dominators();
        assert_eq!(doms[&BlockId(0)], BlockId(0));
        assert_eq!(doms[&BlockId(1)], BlockId(0));
        assert_eq!(doms[&BlockId(2)], BlockId(0));
        assert_eq!(doms[&BlockId(3)], BlockId(0));
    }

    #[test]
    fn test_loop_headers() {
        let cfg = make_loop_cfg();
        let headers = cfg.find_loop_headers();
        assert!(headers.contains(&BlockId(1)));
    }

    #[test]
    fn test_reducible() {
        let cfg = make_diamond_cfg();
        assert!(cfg.is_reducible());
    }

    #[test]
    fn test_exits() {
        let cfg = make_diamond_cfg();
        assert_eq!(cfg.exits.len(), 1);
        assert_eq!(cfg.exits[0], BlockId(3));
    }

    #[test]
    fn test_postorder() {
        let cfg = make_diamond_cfg();
        let po = cfg.postorder();
        assert_eq!(po.len(), 4);
        assert_eq!(*po.last().unwrap(), BlockId(0));
    }
}

//! Control Flow Graph construction and analysis.
//!
//! Builds a CFG from IR basic blocks using petgraph, computes dominators,
//! enumerates paths, detects loops (validating loop-free property),
//! and supports DOT visualization.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;

use shared_types::{BasicBlock, ErrorContext, IrExpr, IrFunction, MutSpecError, Terminator};

// ---------------------------------------------------------------------------
// CFG node and edge types
// ---------------------------------------------------------------------------

/// A node in the control flow graph.
#[derive(Debug, Clone)]
pub enum CfgNode {
    Entry,
    Exit,
    Block { id: usize, label: String },
}

impl CfgNode {
    pub fn block_id(&self) -> Option<usize> {
        match self {
            CfgNode::Block { id, .. } => Some(*id),
            _ => None,
        }
    }
    pub fn is_entry(&self) -> bool {
        matches!(self, CfgNode::Entry)
    }
    pub fn is_exit(&self) -> bool {
        matches!(self, CfgNode::Exit)
    }
}

impl fmt::Display for CfgNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CfgNode::Entry => write!(f, "ENTRY"),
            CfgNode::Exit => write!(f, "EXIT"),
            CfgNode::Block { id, label } => write!(f, "B{}({})", id, label),
        }
    }
}

/// An edge in the control flow graph.
#[derive(Debug, Clone)]
pub struct CfgEdge {
    pub kind: EdgeKind,
}

#[derive(Debug, Clone)]
pub enum EdgeKind {
    Unconditional,
    TrueBranch(IrExpr),
    FalseBranch(IrExpr),
    EntryEdge,
    ExitEdge,
}

impl fmt::Display for CfgEdge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            EdgeKind::Unconditional => write!(f, ""),
            EdgeKind::TrueBranch(_) => write!(f, "T"),
            EdgeKind::FalseBranch(_) => write!(f, "F"),
            EdgeKind::EntryEdge => write!(f, "entry"),
            EdgeKind::ExitEdge => write!(f, "exit"),
        }
    }
}

// ---------------------------------------------------------------------------
// ControlFlowGraph
// ---------------------------------------------------------------------------

/// Control flow graph built from an IR function.
pub struct ControlFlowGraph {
    pub graph: DiGraph<CfgNode, CfgEdge>,
    pub entry: NodeIndex,
    pub exit: NodeIndex,
    block_to_node: HashMap<usize, NodeIndex>,
    node_to_block: HashMap<NodeIndex, usize>,
}

impl ControlFlowGraph {
    /// Build a CFG from an IR function.
    pub fn build(func: &IrFunction) -> shared_types::Result<Self> {
        let mut graph = DiGraph::new();
        let entry = graph.add_node(CfgNode::Entry);
        let exit = graph.add_node(CfgNode::Exit);
        let mut block_to_node = HashMap::new();
        let mut node_to_block = HashMap::new();

        for block in &func.blocks {
            let node = graph.add_node(CfgNode::Block {
                id: block.id,
                label: block.label.clone().unwrap_or_default(),
            });
            block_to_node.insert(block.id, node);
            node_to_block.insert(node, block.id);
        }

        if let Some(entry_bb) = func.entry_block() {
            if let Some(&entry_node) = block_to_node.get(&entry_bb.id) {
                graph.add_edge(
                    entry,
                    entry_node,
                    CfgEdge {
                        kind: EdgeKind::EntryEdge,
                    },
                );
            }
        }

        for block in &func.blocks {
            let src = block_to_node[&block.id];
            match &block.terminator {
                Terminator::Branch { target } => {
                    if let Some(&dst) = block_to_node.get(target) {
                        graph.add_edge(
                            src,
                            dst,
                            CfgEdge {
                                kind: EdgeKind::Unconditional,
                            },
                        );
                    }
                }
                Terminator::ConditionalBranch {
                    condition,
                    true_target,
                    false_target,
                } => {
                    if let Some(&t) = block_to_node.get(true_target) {
                        graph.add_edge(
                            src,
                            t,
                            CfgEdge {
                                kind: EdgeKind::TrueBranch(condition.clone()),
                            },
                        );
                    }
                    if let Some(&f) = block_to_node.get(false_target) {
                        graph.add_edge(
                            src,
                            f,
                            CfgEdge {
                                kind: EdgeKind::FalseBranch(condition.clone()),
                            },
                        );
                    }
                }
                Terminator::Return { .. } => {
                    graph.add_edge(
                        src,
                        exit,
                        CfgEdge {
                            kind: EdgeKind::ExitEdge,
                        },
                    );
                }
                Terminator::Unreachable => {}
            }
        }

        Ok(ControlFlowGraph {
            graph,
            entry,
            exit,
            block_to_node,
            node_to_block,
        })
    }

    pub fn block_node(&self, id: usize) -> Option<NodeIndex> {
        self.block_to_node.get(&id).copied()
    }
    pub fn node_block(&self, idx: NodeIndex) -> Option<usize> {
        self.node_to_block.get(&idx).copied()
    }

    pub fn block_ids(&self) -> Vec<usize> {
        let mut ids: Vec<_> = self.block_to_node.keys().copied().collect();
        ids.sort();
        ids
    }

    pub fn successors(&self, node: NodeIndex) -> Vec<NodeIndex> {
        self.graph
            .neighbors_directed(node, Direction::Outgoing)
            .collect()
    }

    pub fn predecessors(&self, node: NodeIndex) -> Vec<NodeIndex> {
        self.graph
            .neighbors_directed(node, Direction::Incoming)
            .collect()
    }

    // -- Dominator computation ----------------------------------------------

    pub fn compute_dominators(&self) -> HashMap<NodeIndex, NodeIndex> {
        let mut idom: HashMap<NodeIndex, NodeIndex> = HashMap::new();
        idom.insert(self.entry, self.entry);
        let order = self.bfs_order(self.entry);

        let mut changed = true;
        while changed {
            changed = false;
            for &node in &order {
                if node == self.entry {
                    continue;
                }
                let preds: Vec<_> = self
                    .predecessors(node)
                    .into_iter()
                    .filter(|p| idom.contains_key(p))
                    .collect();
                if preds.is_empty() {
                    continue;
                }
                let mut new_idom = preds[0];
                for &pred in &preds[1..] {
                    new_idom = self.intersect_dom(&idom, &order, new_idom, pred);
                }
                if idom.get(&node) != Some(&new_idom) {
                    idom.insert(node, new_idom);
                    changed = true;
                }
            }
        }
        idom
    }

    fn intersect_dom(
        &self,
        idom: &HashMap<NodeIndex, NodeIndex>,
        order: &[NodeIndex],
        mut a: NodeIndex,
        mut b: NodeIndex,
    ) -> NodeIndex {
        let pos: HashMap<NodeIndex, usize> =
            order.iter().enumerate().map(|(i, &n)| (n, i)).collect();
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

    pub fn compute_post_dominators(&self) -> HashMap<NodeIndex, NodeIndex> {
        let mut ipdom: HashMap<NodeIndex, NodeIndex> = HashMap::new();
        ipdom.insert(self.exit, self.exit);
        let order = self.reverse_bfs_order(self.exit);

        let mut changed = true;
        while changed {
            changed = false;
            for &node in &order {
                if node == self.exit {
                    continue;
                }
                let succs: Vec<_> = self
                    .successors(node)
                    .into_iter()
                    .filter(|s| ipdom.contains_key(s))
                    .collect();
                if succs.is_empty() {
                    continue;
                }
                let mut new_ipdom = succs[0];
                for &succ in &succs[1..] {
                    new_ipdom = self.intersect_pdom(&ipdom, &order, new_ipdom, succ);
                }
                if ipdom.get(&node) != Some(&new_ipdom) {
                    ipdom.insert(node, new_ipdom);
                    changed = true;
                }
            }
        }
        ipdom
    }

    fn intersect_pdom(
        &self,
        ipdom: &HashMap<NodeIndex, NodeIndex>,
        order: &[NodeIndex],
        mut a: NodeIndex,
        mut b: NodeIndex,
    ) -> NodeIndex {
        let pos: HashMap<NodeIndex, usize> =
            order.iter().enumerate().map(|(i, &n)| (n, i)).collect();
        while a != b {
            while pos.get(&a).copied().unwrap_or(usize::MAX)
                > pos.get(&b).copied().unwrap_or(usize::MAX)
            {
                a = *ipdom.get(&a).unwrap_or(&a);
            }
            while pos.get(&b).copied().unwrap_or(usize::MAX)
                > pos.get(&a).copied().unwrap_or(usize::MAX)
            {
                b = *ipdom.get(&b).unwrap_or(&b);
            }
        }
        a
    }

    // -- Loop detection / validation ----------------------------------------

    pub fn detect_loops(&self) -> Vec<(NodeIndex, NodeIndex)> {
        let dom = self.compute_dominators();
        let mut back_edges = Vec::new();
        for edge in self.graph.edge_references() {
            let (src, dst) = (edge.source(), edge.target());
            if self.dominates(&dom, dst, src) && dst != self.entry {
                back_edges.push((src, dst));
            }
        }
        back_edges
    }

    pub fn validate_loop_free(&self) -> shared_types::Result<()> {
        let loops = self.detect_loops();
        if loops.is_empty() {
            Ok(())
        } else {
            Err(MutSpecError::Analysis {
                message: format!("CFG contains {} back edges (loops)", loops.len()),
                context: ErrorContext::new(),
            })
        }
    }

    fn dominates(
        &self,
        dom: &HashMap<NodeIndex, NodeIndex>,
        a: NodeIndex,
        mut b: NodeIndex,
    ) -> bool {
        let mut visited = HashSet::new();
        while b != a {
            if !visited.insert(b) {
                return false;
            }
            match dom.get(&b) {
                Some(&d) => b = d,
                None => return false,
            }
        }
        true
    }

    // -- Path enumeration ---------------------------------------------------

    pub fn enumerate_paths(&self) -> Vec<Vec<NodeIndex>> {
        let mut paths = Vec::new();
        let mut current = vec![self.entry];
        self.dfs_paths(self.entry, &mut current, &mut paths, &mut HashSet::new());
        paths
    }

    fn dfs_paths(
        &self,
        node: NodeIndex,
        current: &mut Vec<NodeIndex>,
        paths: &mut Vec<Vec<NodeIndex>>,
        visited: &mut HashSet<NodeIndex>,
    ) {
        if node == self.exit {
            paths.push(current.clone());
            return;
        }
        if !visited.insert(node) {
            return;
        }
        for succ in self.successors(node) {
            current.push(succ);
            self.dfs_paths(succ, current, paths, visited);
            current.pop();
        }
        visited.remove(&node);
    }

    // -- Reachability -------------------------------------------------------

    pub fn reachable_from_entry(&self) -> HashSet<NodeIndex> {
        self.bfs_set(self.entry, Direction::Outgoing)
    }

    pub fn can_reach_exit(&self) -> HashSet<NodeIndex> {
        self.bfs_set(self.exit, Direction::Incoming)
    }

    pub fn unreachable_nodes(&self) -> Vec<NodeIndex> {
        let r = self.reachable_from_entry();
        self.graph
            .node_indices()
            .filter(|n| !r.contains(n))
            .collect()
    }

    fn bfs_set(&self, start: NodeIndex, dir: Direction) -> HashSet<NodeIndex> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        while let Some(node) = queue.pop_front() {
            if visited.insert(node) {
                for nb in self.graph.neighbors_directed(node, dir) {
                    queue.push_back(nb);
                }
            }
        }
        visited
    }

    // -- Simplification -----------------------------------------------------

    pub fn simplify(&mut self) {
        let mut changed = true;
        while changed {
            changed = false;
            let nodes: Vec<_> = self.graph.node_indices().collect();
            for node in nodes {
                if !self.graph.contains_node(node) {
                    continue;
                }
                let succs: Vec<_> = self.successors(node);
                if succs.len() != 1 {
                    continue;
                }
                let succ = succs[0];
                if succ == self.entry || succ == self.exit {
                    continue;
                }
                let preds: Vec<_> = self.predecessors(succ);
                if preds.len() != 1 || preds[0] != node {
                    continue;
                }

                let succ_edges: Vec<_> = self
                    .graph
                    .edges_directed(succ, Direction::Outgoing)
                    .map(|e| (e.target(), e.weight().clone()))
                    .collect();
                if let Some(e) = self.graph.find_edge(node, succ) {
                    self.graph.remove_edge(e);
                }
                for (t, w) in succ_edges {
                    self.graph.add_edge(node, t, w);
                }
                if let Some(bid) = self.node_to_block.remove(&succ) {
                    self.block_to_node.remove(&bid);
                }
                self.graph.remove_node(succ);
                changed = true;
                break;
            }
        }
    }

    // -- DOT output ---------------------------------------------------------

    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph CFG {\n  rankdir=TB;\n");
        for node in self.graph.node_indices() {
            let (label, shape) = match &self.graph[node] {
                CfgNode::Entry => ("ENTRY".into(), "ellipse"),
                CfgNode::Exit => ("EXIT".into(), "ellipse"),
                CfgNode::Block { id, label } => (format!("B{}\\n{}", id, label), "box"),
            };
            dot.push_str(&format!(
                "  n{} [label=\"{}\", shape={}];\n",
                node.index(),
                label,
                shape
            ));
        }
        for edge in self.graph.edge_references() {
            let label = format!("{}", edge.weight());
            if label.is_empty() {
                dot.push_str(&format!(
                    "  n{} -> n{};\n",
                    edge.source().index(),
                    edge.target().index()
                ));
            } else {
                dot.push_str(&format!(
                    "  n{} -> n{} [label=\"{}\"];\n",
                    edge.source().index(),
                    edge.target().index(),
                    label
                ));
            }
        }
        dot.push_str("}\n");
        dot
    }

    // -- Dominance frontier -------------------------------------------------

    pub fn dominance_frontier(&self) -> HashMap<NodeIndex, HashSet<NodeIndex>> {
        let idom = self.compute_dominators();
        let mut df: HashMap<NodeIndex, HashSet<NodeIndex>> = self
            .graph
            .node_indices()
            .map(|n| (n, HashSet::new()))
            .collect();

        for node in self.graph.node_indices() {
            let preds: Vec<_> = self.predecessors(node);
            if preds.len() < 2 {
                continue;
            }
            for &pred in &preds {
                let mut runner = pred;
                let target_idom = *idom.get(&node).unwrap_or(&node);
                while runner != target_idom {
                    df.get_mut(&runner).unwrap().insert(node);
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

    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    fn bfs_order(&self, start: NodeIndex) -> Vec<NodeIndex> {
        let mut order = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        while let Some(node) = queue.pop_front() {
            if visited.insert(node) {
                order.push(node);
                for succ in self.successors(node) {
                    queue.push_back(succ);
                }
            }
        }
        order
    }

    fn reverse_bfs_order(&self, start: NodeIndex) -> Vec<NodeIndex> {
        let mut order = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        while let Some(node) = queue.pop_front() {
            if visited.insert(node) {
                order.push(node);
                for pred in self.predecessors(node) {
                    queue.push_back(pred);
                }
            }
        }
        order
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

    fn build_cfg(src: &str) -> ControlFlowGraph {
        let prog = Parser::parse_source(src).unwrap();
        let ir = IrLowering::new().lower_program(&prog).unwrap();
        ControlFlowGraph::build(&ir.functions[0]).unwrap()
    }

    #[test]
    fn test_cfg_empty() {
        let cfg = build_cfg("fn f() -> void { }");
        assert!(cfg.node_count() >= 3);
    }

    #[test]
    fn test_cfg_linear() {
        let cfg = build_cfg("fn f() -> int { let x: int = 1; return x; }");
        let r = cfg.reachable_from_entry();
        assert!(r.contains(&cfg.entry) && r.contains(&cfg.exit));
    }

    #[test]
    fn test_cfg_if_else() {
        let cfg = build_cfg("fn f(x: int) -> int { if (x > 0) { return x; } else { return 0; } }");
        assert!(cfg.node_count() >= 5 && cfg.edge_count() >= 4);
    }

    #[test]
    fn test_cfg_loop_free() {
        let cfg = build_cfg("fn f(x: int) -> int { if (x > 0) { return x; } else { return -x; } }");
        assert!(cfg.validate_loop_free().is_ok());
    }

    #[test]
    fn test_cfg_dom() {
        let cfg = build_cfg("fn f(x: int) -> int { if (x > 0) { return x; } else { return 0; } }");
        let dom = cfg.compute_dominators();
        assert_eq!(dom[&cfg.entry], cfg.entry);
    }

    #[test]
    fn test_cfg_pdom() {
        let cfg = build_cfg("fn f(x: int) -> int { if (x > 0) { return x; } else { return 0; } }");
        let pdom = cfg.compute_post_dominators();
        assert_eq!(pdom[&cfg.exit], cfg.exit);
    }

    #[test]
    fn test_cfg_paths() {
        let cfg = build_cfg("fn f(x: int) -> int { if (x > 0) { return x; } else { return 0; } }");
        let paths = cfg.enumerate_paths();
        assert!(paths.len() >= 2);
    }

    #[test]
    fn test_cfg_dot() {
        let cfg = build_cfg("fn f(x: int) -> int { if (x > 0) { return x; } else { return 0; } }");
        let dot = cfg.to_dot();
        assert!(dot.contains("digraph") && dot.contains("ENTRY") && dot.contains("EXIT"));
    }

    #[test]
    fn test_cfg_nested() {
        let cfg = build_cfg("fn f(x: int) -> int { if (x > 0) { return 1; } else if (x < 0) { return -1; } else { return 0; } }");
        assert!(cfg.enumerate_paths().len() >= 3);
    }

    #[test]
    fn test_cfg_simplify() {
        let mut cfg = build_cfg("fn f() -> int { let x: int = 1; return x; }");
        let before = cfg.node_count();
        cfg.simplify();
        assert!(cfg.node_count() <= before);
    }

    #[test]
    fn test_cfg_df() {
        let cfg = build_cfg("fn f(x: int) -> int { if (x > 0) { return x; } else { return 0; } }");
        let df = cfg.dominance_frontier();
        assert!(df.contains_key(&cfg.entry));
    }

    #[test]
    fn test_cfg_succs_preds() {
        let cfg = build_cfg("fn f(x: int) -> int { if (x > 0) { return x; } else { return 0; } }");
        assert!(!cfg.successors(cfg.entry).is_empty());
        assert!(!cfg.predecessors(cfg.exit).is_empty());
    }

    #[test]
    fn test_cfg_block_ids() {
        let cfg = build_cfg("fn f(x: int) -> int { if (x > 0) { return x; } else { return 0; } }");
        assert!(!cfg.block_ids().is_empty());
    }
}

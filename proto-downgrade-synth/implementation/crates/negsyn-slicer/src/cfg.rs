//! Control flow graph operations.
//!
//! Provides CFG construction from IR functions, dominator trees, post-dominator
//! trees, control dependence graphs, loop detection, and path enumeration.

use std::collections::{HashMap, HashSet, VecDeque, BTreeSet};
use std::fmt;
use serde::{Serialize, Deserialize};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::algo;
use indexmap::IndexMap;

use crate::ir::{Function, BasicBlock, Instruction};
use crate::{SlicerError, SlicerResult};

// ---------------------------------------------------------------------------
// CFG
// ---------------------------------------------------------------------------

/// Control flow graph for a single function.
pub struct CFG {
    pub function_name: String,
    pub graph: DiGraph<String, CFGEdgeKind>,
    pub node_map: HashMap<String, NodeIndex>,
    pub entry: Option<NodeIndex>,
    pub exits: Vec<NodeIndex>,
}

/// Kind of CFG edge.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CFGEdgeKind {
    /// Unconditional branch.
    Unconditional,
    /// True branch of a conditional.
    TrueBranch,
    /// False branch of a conditional.
    FalseBranch,
    /// Switch case edge with case value.
    SwitchCase(i64),
    /// Switch default edge.
    SwitchDefault,
    /// Normal return from invoke.
    InvokeNormal,
    /// Unwind from invoke.
    InvokeUnwind,
    /// Back edge (detected during loop analysis).
    BackEdge,
}

impl CFG {
    /// Build a CFG from a function.
    pub fn from_function(func: &Function) -> Self {
        let mut graph = DiGraph::new();
        let mut node_map = HashMap::new();

        // Create nodes.
        for (bname, _block) in &func.blocks {
            let idx = graph.add_node(bname.clone());
            node_map.insert(bname.clone(), idx);
        }

        // Create edges from terminators.
        for (bname, block) in &func.blocks {
            let src = node_map[bname];
            if let Some(term) = block.terminator() {
                match term {
                    Instruction::Br { dest } => {
                        if let Some(&tgt) = node_map.get(dest.as_str()) {
                            graph.add_edge(src, tgt, CFGEdgeKind::Unconditional);
                        }
                    }
                    Instruction::CondBr { true_dest, false_dest, .. } => {
                        if let Some(&tgt) = node_map.get(true_dest.as_str()) {
                            graph.add_edge(src, tgt, CFGEdgeKind::TrueBranch);
                        }
                        if let Some(&tgt) = node_map.get(false_dest.as_str()) {
                            graph.add_edge(src, tgt, CFGEdgeKind::FalseBranch);
                        }
                    }
                    Instruction::Switch { default_dest, cases, .. } => {
                        if let Some(&tgt) = node_map.get(default_dest.as_str()) {
                            graph.add_edge(src, tgt, CFGEdgeKind::SwitchDefault);
                        }
                        for (val, dest) in cases {
                            if let Some(&tgt) = node_map.get(dest.as_str()) {
                                let case_val = match val {
                                    crate::ir::Value::IntConst(v, _) => *v,
                                    _ => 0,
                                };
                                graph.add_edge(src, tgt, CFGEdgeKind::SwitchCase(case_val));
                            }
                        }
                    }
                    Instruction::Invoke { normal_dest, unwind_dest, .. } => {
                        if let Some(&tgt) = node_map.get(normal_dest.as_str()) {
                            graph.add_edge(src, tgt, CFGEdgeKind::InvokeNormal);
                        }
                        if let Some(&tgt) = node_map.get(unwind_dest.as_str()) {
                            graph.add_edge(src, tgt, CFGEdgeKind::InvokeUnwind);
                        }
                    }
                    _ => {} // Ret, Unreachable, Resume — no successor edges
                }
            }
        }

        // Identify entry and exits.
        let entry = func.blocks.keys().next().and_then(|name| node_map.get(name).copied());
        let exits: Vec<NodeIndex> = func.blocks.iter()
            .filter(|(_, block)| {
                block.terminator().map_or(false, |t| {
                    matches!(t, Instruction::Ret { .. } | Instruction::Unreachable | Instruction::Resume { .. })
                })
            })
            .filter_map(|(name, _)| node_map.get(name).copied())
            .collect();

        CFG { function_name: func.name.clone(), graph, node_map, entry, exits }
    }

    /// Get successors of a block.
    pub fn successors(&self, block: &str) -> Vec<&str> {
        self.node_map.get(block).map(|&idx| {
            self.graph.neighbors(idx)
                .map(|n| self.graph[n].as_str())
                .collect()
        }).unwrap_or_default()
    }

    /// Get predecessors of a block.
    pub fn predecessors(&self, block: &str) -> Vec<&str> {
        self.node_map.get(block).map(|&idx| {
            self.graph.neighbors_directed(idx, petgraph::Direction::Incoming)
                .map(|n| self.graph[n].as_str())
                .collect()
        }).unwrap_or_default()
    }

    /// Number of blocks.
    pub fn num_blocks(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of edges.
    pub fn num_edges(&self) -> usize {
        self.graph.edge_count()
    }

    /// Block names in reverse post-order.
    pub fn reverse_post_order(&self) -> Vec<String> {
        let entry = match self.entry {
            Some(e) => e,
            None => return Vec::new(),
        };
        let mut visited = HashSet::new();
        let mut rpo = Vec::new();
        self.dfs_post_order(entry, &mut visited, &mut rpo);
        rpo.reverse();
        rpo
    }

    fn dfs_post_order(&self, node: NodeIndex, visited: &mut HashSet<NodeIndex>, result: &mut Vec<String>) {
        if !visited.insert(node) { return; }
        for succ in self.graph.neighbors(node) {
            self.dfs_post_order(succ, visited, result);
        }
        result.push(self.graph[node].clone());
    }

    /// Detect dead (unreachable) blocks.
    pub fn dead_blocks(&self) -> Vec<String> {
        let entry = match self.entry {
            Some(e) => e,
            None => return self.graph.node_indices().map(|n| self.graph[n].clone()).collect(),
        };
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(entry);
        visited.insert(entry);
        while let Some(node) = queue.pop_front() {
            for succ in self.graph.neighbors(node) {
                if visited.insert(succ) {
                    queue.push_back(succ);
                }
            }
        }
        self.graph.node_indices()
            .filter(|n| !visited.contains(n))
            .map(|n| self.graph[n].clone())
            .collect()
    }

    /// Eliminate dead blocks, returning a new CFG.
    pub fn eliminate_dead_blocks(&self, func: &Function) -> CFG {
        let dead = self.dead_blocks();
        if dead.is_empty() { return CFG::from_function(func); }

        let dead_set: HashSet<&str> = dead.iter().map(|s| s.as_str()).collect();
        let mut new_graph = DiGraph::new();
        let mut new_map = HashMap::new();

        for idx in self.graph.node_indices() {
            let name = &self.graph[idx];
            if !dead_set.contains(name.as_str()) {
                let new_idx = new_graph.add_node(name.clone());
                new_map.insert(name.clone(), new_idx);
            }
        }

        for edge in self.graph.edge_references() {
            let src_name = &self.graph[edge.source()];
            let tgt_name = &self.graph[edge.target()];
            if let (Some(&src), Some(&tgt)) = (new_map.get(src_name), new_map.get(tgt_name)) {
                new_graph.add_edge(src, tgt, edge.weight().clone());
            }
        }

        let entry = self.entry.and_then(|e| {
            let name = &self.graph[e];
            new_map.get(name).copied()
        });
        let exits: Vec<NodeIndex> = self.exits.iter().filter_map(|&e| {
            let name = &self.graph[e];
            new_map.get(name).copied()
        }).collect();

        CFG {
            function_name: self.function_name.clone(),
            graph: new_graph,
            node_map: new_map,
            entry,
            exits,
        }
    }

    /// Fold unconditional branches (simplify chains of single-successor/single-predecessor blocks).
    pub fn fold_branches(&self) -> Vec<(String, String)> {
        let mut folds = Vec::new();
        for idx in self.graph.node_indices() {
            let succs: Vec<NodeIndex> = self.graph.neighbors(idx).collect();
            if succs.len() == 1 {
                let succ = succs[0];
                let preds: Vec<NodeIndex> = self.graph.neighbors_directed(succ, petgraph::Direction::Incoming).collect();
                if preds.len() == 1 {
                    folds.push((self.graph[idx].clone(), self.graph[succ].clone()));
                }
            }
        }
        folds
    }
}

// ---------------------------------------------------------------------------
// Dominator Tree
// ---------------------------------------------------------------------------

/// Dominator tree computed via the iterative algorithm.
pub struct DominatorTree {
    /// Immediate dominator for each block.
    pub idom: HashMap<String, String>,
    /// Dominance frontier for each block.
    pub frontier: HashMap<String, HashSet<String>>,
    pub entry: String,
}

impl DominatorTree {
    /// Compute the dominator tree for a CFG using the iterative algorithm.
    pub fn compute(cfg: &CFG) -> Option<Self> {
        let entry_idx = cfg.entry?;
        let entry_name = cfg.graph[entry_idx].clone();
        let rpo = cfg.reverse_post_order();
        if rpo.is_empty() { return None; }

        let mut idom: HashMap<String, String> = HashMap::new();
        idom.insert(entry_name.clone(), entry_name.clone());

        let rpo_number: HashMap<String, usize> = rpo.iter().enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        let mut changed = true;
        while changed {
            changed = false;
            for block in &rpo {
                if block == &entry_name { continue; }
                let preds = cfg.predecessors(block);
                let mut new_idom: Option<String> = None;

                for pred in &preds {
                    if idom.contains_key(*pred) {
                        new_idom = Some(match new_idom {
                            None => pred.to_string(),
                            Some(current) => {
                                Self::intersect(&current, pred, &idom, &rpo_number)
                            }
                        });
                    }
                }

                if let Some(ref new) = new_idom {
                    if idom.get(block) != Some(new) {
                        idom.insert(block.clone(), new.clone());
                        changed = true;
                    }
                }
            }
        }

        // Compute dominance frontiers.
        let mut frontier: HashMap<String, HashSet<String>> = HashMap::new();
        for block in &rpo {
            frontier.insert(block.clone(), HashSet::new());
        }

        for block in &rpo {
            let preds = cfg.predecessors(block);
            if preds.len() >= 2 {
                for pred in &preds {
                    let mut runner = pred.to_string();
                    while runner != *idom.get(block).unwrap_or(&entry_name) {
                        frontier.entry(runner.clone()).or_default().insert(block.clone());
                        runner = match idom.get(&runner) {
                            Some(r) => r.clone(),
                            None => break,
                        };
                    }
                }
            }
        }

        Some(DominatorTree { idom, frontier, entry: entry_name })
    }

    fn intersect(
        b1: &str, b2: &str,
        idom: &HashMap<String, String>,
        rpo_number: &HashMap<String, usize>,
    ) -> String {
        let mut finger1 = b1.to_string();
        let mut finger2 = b2.to_string();
        while finger1 != finger2 {
            let n1 = rpo_number.get(&finger1).copied().unwrap_or(usize::MAX);
            let n2 = rpo_number.get(&finger2).copied().unwrap_or(usize::MAX);
            if n1 > n2 {
                finger1 = idom.get(&finger1).cloned().unwrap_or_else(|| finger1.clone());
            } else {
                finger2 = idom.get(&finger2).cloned().unwrap_or_else(|| finger2.clone());
            }
            // Safety: if we're stuck, break.
            if finger1 == finger2 { break; }
            if !idom.contains_key(&finger1) || !idom.contains_key(&finger2) { break; }
        }
        finger1
    }

    /// Check if `a` dominates `b`.
    pub fn dominates(&self, a: &str, b: &str) -> bool {
        if a == b { return true; }
        let mut current = b.to_string();
        loop {
            match self.idom.get(&current) {
                Some(dom) => {
                    if dom == a { return true; }
                    if dom == &current { return false; } // reached root
                    current = dom.clone();
                }
                None => return false,
            }
        }
    }

    /// Get the immediate dominator.
    pub fn immediate_dominator(&self, block: &str) -> Option<&str> {
        self.idom.get(block).map(|s| s.as_str())
    }

    /// Get the dominance frontier of a block.
    pub fn dominance_frontier(&self, block: &str) -> HashSet<&str> {
        self.frontier.get(block)
            .map(|s| s.iter().map(|x| x.as_str()).collect())
            .unwrap_or_default()
    }

    /// Get all blocks dominated by a given block.
    pub fn dominated_by(&self, block: &str) -> Vec<String> {
        self.idom.iter()
            .filter(|(b, _)| self.dominates(block, b))
            .map(|(b, _)| b.clone())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Post-dominator tree
// ---------------------------------------------------------------------------

/// Post-dominator tree: computed by reversing the CFG.
pub struct PostDominatorTree {
    pub idom: HashMap<String, String>,
    pub frontier: HashMap<String, HashSet<String>>,
}

impl PostDominatorTree {
    /// Compute the post-dominator tree.
    pub fn compute(cfg: &CFG) -> Option<Self> {
        // Build reversed CFG.
        let mut rev_graph = DiGraph::new();
        let mut rev_map = HashMap::new();

        for idx in cfg.graph.node_indices() {
            let name = &cfg.graph[idx];
            let new_idx = rev_graph.add_node(name.clone());
            rev_map.insert(name.clone(), new_idx);
        }
        for edge in cfg.graph.edge_references() {
            let src = &cfg.graph[edge.source()];
            let tgt = &cfg.graph[edge.target()];
            if let (Some(&s), Some(&t)) = (rev_map.get(tgt), rev_map.get(src)) {
                rev_graph.add_edge(s, t, edge.weight().clone());
            }
        }

        // Add a synthetic exit node that is the entry of the reversed CFG.
        let exit_name = "__postdom_exit__".to_string();
        let exit_idx = rev_graph.add_node(exit_name.clone());
        rev_map.insert(exit_name.clone(), exit_idx);
        for &exit in &cfg.exits {
            let name = &cfg.graph[exit];
            if let Some(&idx) = rev_map.get(name) {
                rev_graph.add_edge(exit_idx, idx, CFGEdgeKind::Unconditional);
            }
        }

        let rev_cfg = CFG {
            function_name: format!("{}_rev", cfg.function_name),
            graph: rev_graph,
            node_map: rev_map,
            entry: Some(exit_idx),
            exits: cfg.entry.into_iter().collect(),
        };

        let dom = DominatorTree::compute(&rev_cfg)?;
        Some(PostDominatorTree {
            idom: dom.idom,
            frontier: dom.frontier,
        })
    }

    /// Check if `a` post-dominates `b`.
    pub fn post_dominates(&self, a: &str, b: &str) -> bool {
        if a == b { return true; }
        let mut current = b.to_string();
        loop {
            match self.idom.get(&current) {
                Some(dom) => {
                    if dom == a { return true; }
                    if dom == &current { return false; }
                    current = dom.clone();
                }
                None => return false,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Control dependence
// ---------------------------------------------------------------------------

/// Control dependence graph.
pub struct ControlDependence {
    /// For each block, the set of blocks it is control-dependent on.
    pub dependences: HashMap<String, HashSet<String>>,
}

impl ControlDependence {
    /// Compute control dependences from CFG and post-dominator tree.
    pub fn compute(cfg: &CFG, pdt: &PostDominatorTree) -> Self {
        let mut deps: HashMap<String, HashSet<String>> = HashMap::new();
        for idx in cfg.graph.node_indices() {
            deps.insert(cfg.graph[idx].clone(), HashSet::new());
        }

        // For each edge (A, B) in the CFG where B is NOT the immediate post-dominator of A:
        // all blocks on the path from B to ipdom(A) in the post-dominator tree (exclusive of ipdom(A))
        // are control-dependent on A.
        for edge in cfg.graph.edge_references() {
            let a_name = cfg.graph[edge.source()].clone();
            let b_name = cfg.graph[edge.target()].clone();

            let ipdom_a = pdt.idom.get(&a_name).cloned().unwrap_or_default();
            if b_name == ipdom_a { continue; }

            let mut runner = b_name.clone();
            let mut visited = HashSet::new();
            while runner != ipdom_a && visited.insert(runner.clone()) {
                deps.entry(runner.clone()).or_default().insert(a_name.clone());
                runner = match pdt.idom.get(&runner) {
                    Some(r) => r.clone(),
                    None => break,
                };
            }
        }

        ControlDependence { dependences: deps }
    }

    /// Get blocks that a given block is control-dependent on.
    pub fn control_dependences(&self, block: &str) -> HashSet<&str> {
        self.dependences.get(block)
            .map(|s| s.iter().map(|x| x.as_str()).collect())
            .unwrap_or_default()
    }

    /// Get all blocks that are control-dependent on a given block.
    pub fn dependent_on(&self, block: &str) -> Vec<String> {
        self.dependences.iter()
            .filter(|(_, deps)| deps.contains(block))
            .map(|(b, _)| b.clone())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Loop detection
// ---------------------------------------------------------------------------

/// A natural loop in the CFG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NaturalLoop {
    pub header: String,
    pub back_edge_source: String,
    pub body: HashSet<String>,
    pub exits: Vec<String>,
    pub nesting_depth: usize,
}

impl NaturalLoop {
    /// Whether a block is in this loop.
    pub fn contains(&self, block: &str) -> bool {
        self.body.contains(block)
    }

    /// Number of blocks.
    pub fn size(&self) -> usize {
        self.body.len()
    }
}

/// Detect all natural loops in a CFG.
pub fn detect_loops(cfg: &CFG) -> Vec<NaturalLoop> {
    let dom_tree = match DominatorTree::compute(cfg) {
        Some(dt) => dt,
        None => return Vec::new(),
    };

    // Find back edges: edges (B → H) where H dominates B.
    let mut back_edges: Vec<(String, String)> = Vec::new();
    for edge in cfg.graph.edge_references() {
        let src = &cfg.graph[edge.source()];
        let tgt = &cfg.graph[edge.target()];
        if dom_tree.dominates(tgt, src) {
            back_edges.push((src.clone(), tgt.clone()));
        }
    }

    // For each back edge, compute the natural loop body.
    let mut loops = Vec::new();
    for (back_src, header) in &back_edges {
        let mut body = HashSet::new();
        body.insert(header.clone());

        // Collect loop body via reverse reachability from back_src to header.
        let mut stack = vec![back_src.clone()];
        while let Some(node) = stack.pop() {
            if body.insert(node.clone()) {
                for pred in cfg.predecessors(&node) {
                    if !body.contains(pred) {
                        stack.push(pred.to_string());
                    }
                }
            }
        }

        // Find exit blocks: blocks in the loop with successors outside.
        let exits: Vec<String> = body.iter()
            .filter(|b| {
                cfg.successors(b).iter().any(|s| !body.contains(*s))
            })
            .cloned()
            .collect();

        loops.push(NaturalLoop {
            header: header.clone(),
            back_edge_source: back_src.clone(),
            body,
            exits,
            nesting_depth: 0,
        });
    }

    // Compute nesting depths.
    let num_loops = loops.len();
    for i in 0..num_loops {
        let mut depth = 0;
        for j in 0..num_loops {
            if i != j && loops[j].body.is_superset(&loops[i].body) {
                depth += 1;
            }
        }
        loops[i].nesting_depth = depth;
    }

    loops
}

// ---------------------------------------------------------------------------
// Path enumeration
// ---------------------------------------------------------------------------

/// Enumerate paths through the CFG up to a maximum count.
pub fn enumerate_paths(cfg: &CFG, max_paths: usize) -> Vec<Vec<String>> {
    let entry = match cfg.entry {
        Some(e) => e,
        None => return Vec::new(),
    };

    let exit_set: HashSet<NodeIndex> = cfg.exits.iter().copied().collect();
    let mut paths = Vec::new();
    let mut stack: Vec<(NodeIndex, Vec<String>, HashSet<NodeIndex>)> = Vec::new();

    let entry_name = cfg.graph[entry].clone();
    let mut initial_visited = HashSet::new();
    initial_visited.insert(entry);
    stack.push((entry, vec![entry_name], initial_visited));

    while let Some((current, path, visited)) = stack.pop() {
        if paths.len() >= max_paths { break; }

        if exit_set.contains(&current) {
            paths.push(path);
            continue;
        }

        let succs: Vec<NodeIndex> = cfg.graph.neighbors(current).collect();
        if succs.is_empty() {
            paths.push(path);
            continue;
        }

        for succ in succs {
            if !visited.contains(&succ) {
                let mut new_path = path.clone();
                new_path.push(cfg.graph[succ].clone());
                let mut new_visited = visited.clone();
                new_visited.insert(succ);
                stack.push((succ, new_path, new_visited));
            }
        }
    }

    paths
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Module, Function, BasicBlock, Instruction, Value, Type};

    fn make_diamond_function() -> Function {
        let mut func = Function::new("diamond", Type::i32());
        func.add_param("x", Type::i32());

        let entry = func.add_block("entry");
        entry.push(Instruction::ICmp {
            dest: "cond".into(),
            pred: crate::ir::IntPredicate::Sgt,
            lhs: Value::reg("x", Type::i32()),
            rhs: Value::int(0, 32),
        });
        entry.push(Instruction::CondBr {
            cond: Value::reg("cond", Type::Int(1)),
            true_dest: "then".into(),
            false_dest: "else_bb".into(),
        });

        let then_bb = func.add_block("then");
        then_bb.push(Instruction::Br { dest: "merge".into() });

        let else_bb = func.add_block("else_bb");
        else_bb.push(Instruction::Br { dest: "merge".into() });

        let merge = func.add_block("merge");
        merge.push(Instruction::Ret { value: Some(Value::int(0, 32)) });

        func.compute_predecessors();
        func
    }

    fn make_loop_function() -> Function {
        let mut func = Function::new("loop_func", Type::Void);

        let entry = func.add_block("entry");
        entry.push(Instruction::Br { dest: "header".into() });

        let header = func.add_block("header");
        header.push(Instruction::CondBr {
            cond: Value::int(1, 1),
            true_dest: "body".into(),
            false_dest: "exit".into(),
        });

        let body = func.add_block("body");
        body.push(Instruction::Br { dest: "header".into() });

        let exit = func.add_block("exit");
        exit.push(Instruction::Ret { value: None });

        func.compute_predecessors();
        func
    }

    #[test]
    fn test_cfg_construction() {
        let func = make_diamond_function();
        let cfg = CFG::from_function(&func);
        assert_eq!(cfg.num_blocks(), 4);
        assert!(cfg.entry.is_some());
        assert_eq!(cfg.exits.len(), 1);
    }

    #[test]
    fn test_cfg_successors() {
        let func = make_diamond_function();
        let cfg = CFG::from_function(&func);
        let succs = cfg.successors("entry");
        assert_eq!(succs.len(), 2);
        assert!(succs.contains(&"then"));
        assert!(succs.contains(&"else_bb"));
    }

    #[test]
    fn test_cfg_predecessors() {
        let func = make_diamond_function();
        let cfg = CFG::from_function(&func);
        let preds = cfg.predecessors("merge");
        assert_eq!(preds.len(), 2);
    }

    #[test]
    fn test_reverse_post_order() {
        let func = make_diamond_function();
        let cfg = CFG::from_function(&func);
        let rpo = cfg.reverse_post_order();
        assert_eq!(rpo.len(), 4);
        assert_eq!(rpo[0], "entry");
    }

    #[test]
    fn test_dead_blocks() {
        let mut func = make_diamond_function();
        // Add a disconnected block.
        let dead = func.add_block("dead_block");
        dead.push(Instruction::Ret { value: None });
        func.compute_predecessors();

        let cfg = CFG::from_function(&func);
        let dead_blocks = cfg.dead_blocks();
        assert!(dead_blocks.contains(&"dead_block".to_string()));
    }

    #[test]
    fn test_dominator_tree() {
        let func = make_diamond_function();
        let cfg = CFG::from_function(&func);
        let dom = DominatorTree::compute(&cfg).unwrap();

        assert!(dom.dominates("entry", "then"));
        assert!(dom.dominates("entry", "else_bb"));
        assert!(dom.dominates("entry", "merge"));
        assert!(!dom.dominates("then", "else_bb"));
    }

    #[test]
    fn test_immediate_dominator() {
        let func = make_diamond_function();
        let cfg = CFG::from_function(&func);
        let dom = DominatorTree::compute(&cfg).unwrap();

        assert_eq!(dom.immediate_dominator("then"), Some("entry"));
        assert_eq!(dom.immediate_dominator("else_bb"), Some("entry"));
    }

    #[test]
    fn test_post_dominator_tree() {
        let func = make_diamond_function();
        let cfg = CFG::from_function(&func);
        let pdt = PostDominatorTree::compute(&cfg).unwrap();

        assert!(pdt.post_dominates("merge", "then"));
        assert!(pdt.post_dominates("merge", "else_bb"));
    }

    #[test]
    fn test_control_dependence() {
        let func = make_diamond_function();
        let cfg = CFG::from_function(&func);
        let pdt = PostDominatorTree::compute(&cfg).unwrap();
        let cd = ControlDependence::compute(&cfg, &pdt);

        // "then" and "else_bb" should be control-dependent on "entry".
        let then_deps = cd.control_dependences("then");
        assert!(then_deps.contains("entry"));
    }

    #[test]
    fn test_loop_detection() {
        let func = make_loop_function();
        let cfg = CFG::from_function(&func);
        let loops = detect_loops(&cfg);

        assert!(!loops.is_empty());
        let lp = &loops[0];
        assert_eq!(lp.header, "header");
        assert!(lp.body.contains("body"));
        assert!(lp.body.contains("header"));
    }

    #[test]
    fn test_loop_nesting() {
        let func = make_loop_function();
        let cfg = CFG::from_function(&func);
        let loops = detect_loops(&cfg);
        // Single loop, depth 0.
        assert!(loops.iter().all(|l| l.nesting_depth == 0));
    }

    #[test]
    fn test_path_enumeration() {
        let func = make_diamond_function();
        let cfg = CFG::from_function(&func);
        let paths = enumerate_paths(&cfg, 10);
        // Should find 2 paths: entry→then→merge, entry→else_bb→merge.
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn test_fold_branches() {
        let mut func = Function::new("chain", Type::Void);
        let a = func.add_block("a");
        a.push(Instruction::Br { dest: "b".into() });
        let b = func.add_block("b");
        b.push(Instruction::Br { dest: "c".into() });
        let c = func.add_block("c");
        c.push(Instruction::Ret { value: None });
        func.compute_predecessors();

        let cfg = CFG::from_function(&func);
        let folds = cfg.fold_branches();
        assert!(!folds.is_empty());
    }

    #[test]
    fn test_cfg_from_test_module() {
        let module = Module::test_module();
        for (_fname, func) in &module.functions {
            let cfg = CFG::from_function(func);
            assert!(cfg.num_blocks() > 0);
        }
    }

    #[test]
    fn test_dominated_by() {
        let func = make_diamond_function();
        let cfg = CFG::from_function(&func);
        let dom = DominatorTree::compute(&cfg).unwrap();

        let dominated = dom.dominated_by("entry");
        assert!(dominated.len() >= 4);
    }

    #[test]
    fn test_natural_loop_contains() {
        let lp = NaturalLoop {
            header: "h".into(),
            back_edge_source: "b".into(),
            body: vec!["h".into(), "b".into(), "c".into()].into_iter().collect(),
            exits: vec!["e".into()],
            nesting_depth: 0,
        };
        assert!(lp.contains("h"));
        assert!(lp.contains("b"));
        assert!(!lp.contains("x"));
        assert_eq!(lp.size(), 3);
    }
}

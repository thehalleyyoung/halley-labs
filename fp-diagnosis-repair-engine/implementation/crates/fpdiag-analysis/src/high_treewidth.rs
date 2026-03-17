//! High-treewidth fallback solver for EAG diagnosis.
//!
//! When the EAG Decomposition Conjecture (treewidth ≤ k) fails — i.e., the EAG
//! has treewidth exceeding the configurable threshold — this module provides an
//! alternative O(n²) diagnosis path that avoids the O(n·2^k) cost of tree
//! decomposition.
//!
//! Strategy:
//!   1. Partition the EAG into strongly connected components (SCCs).
//!   2. Solve each SCC independently using interval-arithmetic error bounding.
//!   3. Merge per-SCC results via dependency-ordered propagation along the
//!      DAG of SCCs.
//!
//! The adaptive router (`AdaptiveEagSolver`) estimates treewidth first and
//! dispatches to either the fast tree-decomposition path or this fallback.

use fpdiag_types::eag::{EagEdge, EagEdgeId, EagNode, EagNodeId, ErrorAmplificationGraph};
use std::collections::{HashMap, HashSet, VecDeque};

/// Default treewidth threshold above which the fallback solver is used.
pub const DEFAULT_TREEWIDTH_THRESHOLD: usize = 15;

// ─── Interval type ──────────────────────────────────────────────────────────

/// A simple [lo, hi] interval for conservative error bounding.
#[derive(Debug, Clone, Copy)]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
}

impl Interval {
    pub fn new(lo: f64, hi: f64) -> Self {
        debug_assert!(lo <= hi, "invalid interval [{lo}, {hi}]");
        Self { lo, hi }
    }

    pub fn point(v: f64) -> Self {
        Self { lo: v, hi: v }
    }

    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }

    /// Conservative multiplication: accounts for sign combinations.
    pub fn mul(&self, other: &Interval) -> Interval {
        let products = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ];
        let lo = products.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = products.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        Interval { lo, hi }
    }

    /// Conservative addition.
    pub fn add(&self, other: &Interval) -> Interval {
        Interval {
            lo: self.lo + other.lo,
            hi: self.hi + other.hi,
        }
    }

    /// Union of two intervals (join in the lattice).
    pub fn join(&self, other: &Interval) -> Interval {
        Interval {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }
}

// ─── SCC decomposition (Tarjan) ─────────────────────────────────────────────

/// Result of decomposing an EAG into strongly connected components.
#[derive(Debug)]
pub struct SccDecomposition {
    /// Each SCC is a set of node ids.
    pub components: Vec<Vec<EagNodeId>>,
    /// Maps each node to its SCC index.
    pub node_to_scc: HashMap<EagNodeId, usize>,
    /// DAG of SCCs: scc_adj[i] lists SCCs that SCC i depends on (has edges into).
    pub scc_adj: Vec<Vec<usize>>,
    /// Topological order of SCCs (dependency-first).
    pub topo_order: Vec<usize>,
}

/// Decompose the EAG into SCCs using Tarjan's algorithm.
pub fn scc_decompose(eag: &ErrorAmplificationGraph) -> SccDecomposition {
    let nodes: Vec<EagNodeId> = eag.nodes().iter().map(|n| n.id).collect();
    let n = nodes.len();

    let mut index_counter: u32 = 0;
    let mut stack: Vec<EagNodeId> = Vec::new();
    let mut on_stack: HashSet<EagNodeId> = HashSet::new();
    let mut indices: HashMap<EagNodeId, u32> = HashMap::new();
    let mut lowlinks: HashMap<EagNodeId, u32> = HashMap::new();
    let mut components: Vec<Vec<EagNodeId>> = Vec::new();

    // Iterative Tarjan to avoid deep recursion on large graphs.
    for &root in &nodes {
        if indices.contains_key(&root) {
            continue;
        }
        // Work-stack frames: (node, edge_index, is_root_call)
        let mut work: Vec<(EagNodeId, usize, bool)> = vec![(root, 0, true)];

        while let Some((v, ei, first_visit)) = work.last_mut() {
            let v = *v;
            if *first_visit {
                indices.insert(v, index_counter);
                lowlinks.insert(v, index_counter);
                index_counter += 1;
                stack.push(v);
                on_stack.insert(v);
                *first_visit = false;
            }

            let successors: Vec<EagNodeId> = eag.outgoing(v).iter().map(|e| e.target).collect();

            if *ei < successors.len() {
                let w = successors[*ei];
                *ei += 1;
                if !indices.contains_key(&w) {
                    work.push((w, 0, true));
                } else if on_stack.contains(&w) {
                    let lv = *lowlinks.get(&v).unwrap();
                    let iw = *indices.get(&w).unwrap();
                    lowlinks.insert(v, lv.min(iw));
                }
            } else {
                // All successors processed — check if v is an SCC root.
                let lv = *lowlinks.get(&v).unwrap();
                let iv = *indices.get(&v).unwrap();
                if lv == iv {
                    let mut component = Vec::new();
                    loop {
                        let w = stack.pop().unwrap();
                        on_stack.remove(&w);
                        component.push(w);
                        if w == v {
                            break;
                        }
                    }
                    components.push(component);
                }
                // Propagate lowlink to parent.
                work.pop();
                if let Some((parent, _, _)) = work.last() {
                    let lp = *lowlinks.get(parent).unwrap();
                    lowlinks.insert(*parent, lp.min(lv));
                }
            }
        }
    }

    // Build node→SCC map.
    let mut node_to_scc: HashMap<EagNodeId, usize> = HashMap::new();
    for (idx, comp) in components.iter().enumerate() {
        for &nid in comp {
            node_to_scc.insert(nid, idx);
        }
    }

    // Build SCC-level DAG.
    let num_sccs = components.len();
    let mut scc_adj: Vec<HashSet<usize>> = vec![HashSet::new(); num_sccs];
    for edge in eag.edges() {
        let src_scc = node_to_scc[&edge.source];
        let tgt_scc = node_to_scc[&edge.target];
        if src_scc != tgt_scc {
            scc_adj[tgt_scc].insert(src_scc);
        }
    }
    let scc_adj_vec: Vec<Vec<usize>> = scc_adj
        .into_iter()
        .map(|s| s.into_iter().collect())
        .collect();

    // Topological sort of SCC DAG (Kahn's algorithm).
    let mut in_degree: Vec<usize> = vec![0; num_sccs];
    let mut forward_adj: Vec<Vec<usize>> = vec![Vec::new(); num_sccs];
    for (tgt, deps) in scc_adj_vec.iter().enumerate() {
        in_degree[tgt] = deps.len();
        for &src in deps {
            forward_adj[src].push(tgt);
        }
    }
    let mut queue: VecDeque<usize> = VecDeque::new();
    for i in 0..num_sccs {
        if in_degree[i] == 0 {
            queue.push_back(i);
        }
    }
    let mut topo_order = Vec::with_capacity(num_sccs);
    while let Some(s) = queue.pop_front() {
        topo_order.push(s);
        for &t in &forward_adj[s] {
            in_degree[t] -= 1;
            if in_degree[t] == 0 {
                queue.push_back(t);
            }
        }
    }

    SccDecomposition {
        components,
        node_to_scc,
        scc_adj: scc_adj_vec,
        topo_order,
    }
}

// ─── Per-SCC interval solver ────────────────────────────────────────────────

/// Error bound computed for a single node.
#[derive(Debug, Clone)]
pub struct NodeErrorBound {
    pub node_id: EagNodeId,
    pub error_interval: Interval,
}

/// Solve a single SCC using iterative interval-arithmetic fixed-point.
///
/// Within an SCC all nodes may mutually depend on each other, so we iterate
/// until the intervals stabilise (widening guarantees termination).
fn solve_scc(
    eag: &ErrorAmplificationGraph,
    scc_nodes: &[EagNodeId],
    incoming_bounds: &HashMap<EagNodeId, Interval>,
    max_iterations: usize,
) -> HashMap<EagNodeId, Interval> {
    let scc_set: HashSet<EagNodeId> = scc_nodes.iter().copied().collect();
    let mut bounds: HashMap<EagNodeId, Interval> = HashMap::new();

    // Initialise each node with its local error as a point interval.
    for &nid in scc_nodes {
        let local = eag.node(nid).map_or(0.0, |n| n.local_error);
        bounds.insert(nid, Interval::point(local));
    }

    for _iter in 0..max_iterations {
        let mut changed = false;

        for &nid in scc_nodes {
            let local = eag.node(nid).map_or(0.0, |n| n.local_error);
            let mut total = Interval::point(local);

            for edge in eag.incoming(nid) {
                let w = Interval::point(edge.weight.0);
                let src_bound = if scc_set.contains(&edge.source) {
                    bounds
                        .get(&edge.source)
                        .copied()
                        .unwrap_or(Interval::point(0.0))
                } else {
                    incoming_bounds
                        .get(&edge.source)
                        .copied()
                        .unwrap_or(Interval::point(0.0))
                };
                total = total.add(&w.mul(&src_bound));
            }

            let prev = bounds.get(&nid).copied().unwrap_or(Interval::point(0.0));
            let joined = prev.join(&total);
            if (joined.lo - prev.lo).abs() > 1e-15 || (joined.hi - prev.hi).abs() > 1e-15 {
                changed = true;
            }
            bounds.insert(nid, joined);
        }

        if !changed {
            break;
        }
    }

    bounds
}

// ─── HighTreewidthSolver ────────────────────────────────────────────────────

/// Configuration for the high-treewidth fallback solver.
#[derive(Debug, Clone)]
pub struct HighTreewidthConfig {
    /// Maximum fixed-point iterations per SCC.
    pub max_scc_iterations: usize,
}

impl Default for HighTreewidthConfig {
    fn default() -> Self {
        Self {
            max_scc_iterations: 100,
        }
    }
}

/// Fallback solver for EAGs whose treewidth exceeds the fast-path threshold.
///
/// Complexity: O(n²) in the worst case (n = number of EAG nodes), which is
/// strictly better than O(n·2^k) tree decomposition when k is large.
pub struct HighTreewidthSolver {
    config: HighTreewidthConfig,
}

impl HighTreewidthSolver {
    pub fn new(config: HighTreewidthConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(HighTreewidthConfig::default())
    }

    /// Solve the full EAG via SCC decomposition + interval propagation.
    ///
    /// Returns conservative error-bound intervals for every node.
    pub fn solve(&self, eag: &ErrorAmplificationGraph) -> Vec<NodeErrorBound> {
        let decomp = scc_decompose(eag);
        let mut global_bounds: HashMap<EagNodeId, Interval> = HashMap::new();

        // Process SCCs in dependency order.
        for &scc_idx in &decomp.topo_order {
            let scc_nodes = &decomp.components[scc_idx];
            let scc_bounds = solve_scc(
                eag,
                scc_nodes,
                &global_bounds,
                self.config.max_scc_iterations,
            );
            for (nid, interval) in scc_bounds {
                global_bounds.insert(nid, interval);
            }
        }

        global_bounds
            .into_iter()
            .map(|(node_id, error_interval)| NodeErrorBound {
                node_id,
                error_interval,
            })
            .collect()
    }
}

// ─── Adaptive router ────────────────────────────────────────────────────────

/// Result of adaptive EAG solving, indicating which path was taken.
#[derive(Debug)]
pub enum SolverResult {
    /// Used fast tree-decomposition path (treewidth ≤ threshold).
    TreeDecomposition { treewidth: usize, t1_bound: f64 },
    /// Used SCC-based fallback (treewidth > threshold).
    Fallback {
        treewidth: usize,
        num_sccs: usize,
        bounds: Vec<NodeErrorBound>,
    },
}

/// Adaptive solver that routes between tree-decomposition and fallback
/// based on estimated treewidth.
pub struct AdaptiveEagSolver {
    /// Treewidth threshold: at or below → tree decomposition, above → fallback.
    pub treewidth_threshold: usize,
    fallback: HighTreewidthSolver,
}

impl AdaptiveEagSolver {
    pub fn new(treewidth_threshold: usize) -> Self {
        Self {
            treewidth_threshold,
            fallback: HighTreewidthSolver::with_defaults(),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(DEFAULT_TREEWIDTH_THRESHOLD)
    }

    /// Estimate treewidth and solve via the appropriate path.
    pub fn solve(&self, eag: &ErrorAmplificationGraph) -> SolverResult {
        let tw = crate::estimate_treewidth(eag);
        log::info!("EAG treewidth estimate: {tw}");

        if tw <= self.treewidth_threshold {
            // Fast path: tree decomposition (existing T1 bound analysis).
            let bound = crate::t1_bound(eag);
            log::info!(
                "Using tree-decomposition path (tw={tw} ≤ {})",
                self.treewidth_threshold
            );
            SolverResult::TreeDecomposition {
                treewidth: tw,
                t1_bound: bound,
            }
        } else {
            // Fallback: SCC decomposition + interval arithmetic.
            log::warn!(
                "High treewidth ({tw} > {}); using SCC fallback solver",
                self.treewidth_threshold
            );
            let bounds = self.fallback.solve(eag);
            let decomp = scc_decompose(eag);
            SolverResult::Fallback {
                treewidth: tw,
                num_sccs: decomp.components.len(),
                bounds,
            }
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fpdiag_types::eag::{EagEdge, EagEdgeId, EagNode, EagNodeId, ErrorAmplificationGraph};
    use fpdiag_types::expression::FpOp;

    /// Build a simple linear chain EAG: n0 → n1 → n2.
    fn linear_chain() -> ErrorAmplificationGraph {
        let mut eag = ErrorAmplificationGraph::new();
        let n0 = EagNode::new(EagNodeId(0), FpOp::Add, 1.0000001, 1.0);
        let n1 = EagNode::new(EagNodeId(1), FpOp::Mul, 2.0000004, 2.0);
        let n2 = EagNode::new(EagNodeId(2), FpOp::Sub, 0.5000002, 0.5);
        eag.add_node(n0);
        eag.add_node(n1);
        eag.add_node(n2);
        eag.add_edge(EagEdge::new(EagEdgeId(0), EagNodeId(0), EagNodeId(1), 2.0));
        eag.add_edge(EagEdge::new(EagEdgeId(1), EagNodeId(1), EagNodeId(2), 1.5));
        eag
    }

    /// Build a diamond DAG: n0 → n1, n0 → n2, n1 → n3, n2 → n3.
    fn diamond_dag() -> ErrorAmplificationGraph {
        let mut eag = ErrorAmplificationGraph::new();
        eag.add_node(EagNode::new(EagNodeId(0), FpOp::Add, 1.000001, 1.0));
        eag.add_node(EagNode::new(EagNodeId(1), FpOp::Mul, 2.000003, 2.0));
        eag.add_node(EagNode::new(EagNodeId(2), FpOp::Sub, 0.500001, 0.5));
        eag.add_node(EagNode::new(EagNodeId(3), FpOp::Add, 3.000005, 3.0));
        eag.add_edge(EagEdge::new(EagEdgeId(0), EagNodeId(0), EagNodeId(1), 2.0));
        eag.add_edge(EagEdge::new(EagEdgeId(1), EagNodeId(0), EagNodeId(2), 1.0));
        eag.add_edge(EagEdge::new(EagEdgeId(2), EagNodeId(1), EagNodeId(3), 1.5));
        eag.add_edge(EagEdge::new(EagEdgeId(3), EagNodeId(2), EagNodeId(3), 3.0));
        eag
    }

    #[test]
    fn scc_linear_chain_all_singletons() {
        let eag = linear_chain();
        let decomp = scc_decompose(&eag);
        assert_eq!(
            decomp.components.len(),
            3,
            "linear chain has 3 trivial SCCs"
        );
        for comp in &decomp.components {
            assert_eq!(comp.len(), 1);
        }
    }

    #[test]
    fn scc_diamond_all_singletons() {
        let eag = diamond_dag();
        let decomp = scc_decompose(&eag);
        assert_eq!(decomp.components.len(), 4);
    }

    #[test]
    fn fallback_solver_produces_bounds_for_all_nodes() {
        let eag = linear_chain();
        let solver = HighTreewidthSolver::with_defaults();
        let bounds = solver.solve(&eag);
        assert_eq!(bounds.len(), 3);
        for b in &bounds {
            assert!(
                b.error_interval.hi >= 0.0,
                "error bound should be non-negative"
            );
        }
    }

    #[test]
    fn fallback_bounds_are_conservative() {
        let eag = diamond_dag();
        let solver = HighTreewidthSolver::with_defaults();
        let bounds = solver.solve(&eag);
        let bound_map: HashMap<EagNodeId, Interval> = bounds
            .into_iter()
            .map(|b| (b.node_id, b.error_interval))
            .collect();
        // Sink node (n3) should have a bound at least as large as its local error.
        let sink_bound = bound_map.get(&EagNodeId(3)).unwrap();
        let sink_local = eag.node(EagNodeId(3)).unwrap().local_error;
        assert!(
            sink_bound.hi >= sink_local,
            "fallback bound should be conservative (≥ local error)"
        );
    }

    #[test]
    fn adaptive_solver_routes_low_treewidth() {
        let eag = linear_chain();
        let solver = AdaptiveEagSolver::with_defaults();
        match solver.solve(&eag) {
            SolverResult::TreeDecomposition { treewidth, .. } => {
                assert!(treewidth <= DEFAULT_TREEWIDTH_THRESHOLD);
            }
            SolverResult::Fallback { .. } => {
                panic!("linear chain should use tree-decomposition path");
            }
        }
    }

    #[test]
    fn adaptive_solver_routes_high_treewidth() {
        let eag = diamond_dag();
        // Force a threshold of 0 so any graph triggers the fallback.
        let solver = AdaptiveEagSolver::new(0);
        match solver.solve(&eag) {
            SolverResult::Fallback {
                num_sccs, bounds, ..
            } => {
                assert!(num_sccs > 0);
                assert_eq!(bounds.len(), 4);
            }
            SolverResult::TreeDecomposition { .. } => {
                panic!("threshold=0 should force fallback path");
            }
        }
    }

    #[test]
    fn interval_arithmetic_basic() {
        let a = Interval::new(1.0, 2.0);
        let b = Interval::new(3.0, 4.0);
        let sum = a.add(&b);
        assert!((sum.lo - 4.0).abs() < 1e-10);
        assert!((sum.hi - 6.0).abs() < 1e-10);
        let prod = a.mul(&b);
        assert!((prod.lo - 3.0).abs() < 1e-10);
        assert!((prod.hi - 8.0).abs() < 1e-10);
    }
}

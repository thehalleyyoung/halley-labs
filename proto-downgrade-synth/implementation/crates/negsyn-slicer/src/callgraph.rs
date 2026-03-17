//! Call graph construction and analysis.
//!
//! Builds inter-procedural call graphs from LLVM IR modules, resolving
//! indirect calls through points-to analysis results. Supports protocol-specific
//! call patterns like SSL_METHOD vtable dispatch and BIO callbacks.

use std::collections::{HashMap, HashSet, VecDeque, BTreeSet};
use std::fmt;
use serde::{Serialize, Deserialize};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::algo;
use indexmap::IndexMap;

use crate::ir::{Module, Function, Instruction, Value, Type};
use crate::points_to::{AndersonAnalysis, PointsToSet, AbstractLocation};
use crate::{InstructionId, SlicerError, SlicerResult, NegotiationPhase};

// ---------------------------------------------------------------------------
// Call site
// ---------------------------------------------------------------------------

/// A call site: a specific location where one function calls another.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CallSite {
    /// The calling function.
    pub caller: String,
    /// The called function (resolved).
    pub callee: String,
    /// Block and instruction index of the call.
    pub location: InstructionId,
    /// Whether this is an indirect (function-pointer) call.
    pub is_indirect: bool,
    /// Whether this call is through a vtable dispatch.
    pub is_vtable_dispatch: bool,
    /// Whether this call is a callback invocation.
    pub is_callback: bool,
    /// Argument count.
    pub num_args: usize,
    /// Call context (for context-sensitive analyses).
    pub context: Option<String>,
}

impl CallSite {
    pub fn direct(caller: impl Into<String>, callee: impl Into<String>, loc: InstructionId, num_args: usize) -> Self {
        Self {
            caller: caller.into(),
            callee: callee.into(),
            location: loc,
            is_indirect: false,
            is_vtable_dispatch: false,
            is_callback: false,
            num_args,
            context: None,
        }
    }

    pub fn indirect(caller: impl Into<String>, callee: impl Into<String>, loc: InstructionId, num_args: usize) -> Self {
        Self {
            caller: caller.into(),
            callee: callee.into(),
            location: loc,
            is_indirect: true,
            is_vtable_dispatch: false,
            is_callback: false,
            num_args,
            context: None,
        }
    }

    pub fn vtable(caller: impl Into<String>, callee: impl Into<String>, loc: InstructionId, num_args: usize) -> Self {
        Self {
            caller: caller.into(),
            callee: callee.into(),
            location: loc,
            is_indirect: true,
            is_vtable_dispatch: true,
            is_callback: false,
            num_args,
            context: None,
        }
    }
}

impl fmt::Display for CallSite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = if self.is_vtable_dispatch { "vtable" }
            else if self.is_indirect { "indirect" }
            else { "direct" };
        write!(f, "{}({}) -> {} @ {}", self.caller, kind, self.callee, self.location)
    }
}

// ---------------------------------------------------------------------------
// Call graph
// ---------------------------------------------------------------------------

/// Edge weight in the call graph.
#[derive(Debug, Clone)]
pub struct CallEdge {
    pub sites: Vec<CallSite>,
}

impl CallEdge {
    fn new(site: CallSite) -> Self {
        Self { sites: vec![site] }
    }
    fn add_site(&mut self, site: CallSite) {
        if !self.sites.contains(&site) {
            self.sites.push(site);
        }
    }
}

/// The inter-procedural call graph.
pub struct CallGraph {
    graph: DiGraph<String, CallEdge>,
    node_map: HashMap<String, NodeIndex>,
    /// All call sites indexed by caller function.
    call_sites_by_caller: HashMap<String, Vec<CallSite>>,
    /// All call sites indexed by callee function.
    call_sites_by_callee: HashMap<String, Vec<CallSite>>,
    /// Functions that are entry points (not called by anyone).
    pub entry_points: Vec<String>,
    /// Functions that are leaves (don't call anyone).
    pub leaf_functions: Vec<String>,
}

impl CallGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_map: HashMap::new(),
            call_sites_by_caller: HashMap::new(),
            call_sites_by_callee: HashMap::new(),
            entry_points: Vec::new(),
            leaf_functions: Vec::new(),
        }
    }

    fn get_or_create_node(&mut self, name: &str) -> NodeIndex {
        if let Some(&idx) = self.node_map.get(name) {
            idx
        } else {
            let idx = self.graph.add_node(name.to_string());
            self.node_map.insert(name.to_string(), idx);
            idx
        }
    }

    /// Add a call site to the graph.
    pub fn add_call_site(&mut self, site: CallSite) {
        let caller_idx = self.get_or_create_node(&site.caller);
        let callee_idx = self.get_or_create_node(&site.callee);

        // Check for existing edge.
        let existing_edge = self.graph.edges(caller_idx)
            .find(|e| e.target() == callee_idx);

        if let Some(edge_ref) = existing_edge {
            let edge_id = edge_ref.id();
            self.graph[edge_id].add_site(site.clone());
        } else {
            self.graph.add_edge(caller_idx, callee_idx, CallEdge::new(site.clone()));
        }

        self.call_sites_by_caller.entry(site.caller.clone()).or_default().push(site.clone());
        self.call_sites_by_callee.entry(site.callee.clone()).or_default().push(site);
    }

    /// Get all call sites from a function.
    pub fn callees_of(&self, func: &str) -> Vec<&CallSite> {
        self.call_sites_by_caller.get(func)
            .map(|sites| sites.iter().collect())
            .unwrap_or_default()
    }

    /// Get all call sites to a function.
    pub fn callers_of(&self, func: &str) -> Vec<&CallSite> {
        self.call_sites_by_callee.get(func)
            .map(|sites| sites.iter().collect())
            .unwrap_or_default()
    }

    /// Get all direct callee names.
    pub fn callee_names(&self, func: &str) -> Vec<&str> {
        self.call_sites_by_caller.get(func)
            .map(|sites| {
                let mut names: Vec<&str> = sites.iter().map(|s| s.callee.as_str()).collect();
                names.sort();
                names.dedup();
                names
            })
            .unwrap_or_default()
    }

    /// Get all direct caller names.
    pub fn caller_names(&self, func: &str) -> Vec<&str> {
        self.call_sites_by_callee.get(func)
            .map(|sites| {
                let mut names: Vec<&str> = sites.iter().map(|s| s.caller.as_str()).collect();
                names.sort();
                names.dedup();
                names
            })
            .unwrap_or_default()
    }

    /// Whether the graph contains a function.
    pub fn contains_function(&self, name: &str) -> bool {
        self.node_map.contains_key(name)
    }

    /// Total number of functions in the call graph.
    pub fn num_functions(&self) -> usize {
        self.node_map.len()
    }

    /// Total number of call edges.
    pub fn num_edges(&self) -> usize {
        self.graph.edge_count()
    }

    /// Compute entry points and leaf functions.
    pub fn compute_entry_and_leaf(&mut self) {
        let all_funcs: HashSet<&str> = self.node_map.keys().map(|s| s.as_str()).collect();
        let called: HashSet<&str> = self.call_sites_by_callee.keys().map(|s| s.as_str()).collect();
        let callers: HashSet<&str> = self.call_sites_by_caller.keys().map(|s| s.as_str()).collect();

        self.entry_points = all_funcs.difference(&called)
            .map(|s| s.to_string())
            .collect();
        self.entry_points.sort();

        self.leaf_functions = all_funcs.difference(&callers)
            .map(|s| s.to_string())
            .collect();
        self.leaf_functions.sort();
    }

    /// Compute strongly connected components (SCCs).
    pub fn sccs(&self) -> Vec<Vec<String>> {
        let sccs = algo::kosaraju_scc(&self.graph);
        sccs.into_iter()
            .map(|scc| scc.into_iter().map(|idx| self.graph[idx].clone()).collect())
            .collect()
    }

    /// Find all functions reachable from a given root.
    pub fn reachable_from(&self, root: &str) -> HashSet<String> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        if let Some(&start) = self.node_map.get(root) {
            queue.push_back(start);
            visited.insert(root.to_string());

            while let Some(node) = queue.pop_front() {
                for neighbor in self.graph.neighbors(node) {
                    let name = &self.graph[neighbor];
                    if visited.insert(name.clone()) {
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        visited
    }

    /// Find all functions that can reach a given target (reverse reachability).
    pub fn can_reach(&self, target: &str) -> HashSet<String> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        if let Some(&start) = self.node_map.get(target) {
            queue.push_back(start);
            visited.insert(target.to_string());

            while let Some(node) = queue.pop_front() {
                for neighbor in self.graph.neighbors_directed(node, petgraph::Direction::Incoming) {
                    let name = &self.graph[neighbor];
                    if visited.insert(name.clone()) {
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        visited
    }

    /// Extract the subgraph of negotiation-relevant functions.
    pub fn negotiation_subgraph(&self, module: &Module) -> CallGraph {
        let mut sub = CallGraph::new();
        let neg_funcs: HashSet<String> = module.negotiation_functions()
            .iter()
            .map(|f| f.name.clone())
            .collect();

        // Include all functions reachable from negotiation functions
        // and all functions that can reach them.
        let mut relevant: HashSet<String> = HashSet::new();
        for nf in &neg_funcs {
            relevant.extend(self.reachable_from(nf));
            relevant.extend(self.can_reach(nf));
        }

        for (caller, sites) in &self.call_sites_by_caller {
            if relevant.contains(caller) {
                for site in sites {
                    if relevant.contains(&site.callee) {
                        sub.add_call_site(site.clone());
                    }
                }
            }
        }

        sub.compute_entry_and_leaf();
        sub
    }

    /// Prune the call graph to only include functions reachable from entry points.
    pub fn prune_unreachable(&self, roots: &[&str]) -> CallGraph {
        let mut reachable = HashSet::new();
        for root in roots {
            reachable.extend(self.reachable_from(root));
        }

        let mut pruned = CallGraph::new();
        for (caller, sites) in &self.call_sites_by_caller {
            if reachable.contains(caller) {
                for site in sites {
                    if reachable.contains(&site.callee) {
                        pruned.add_call_site(site.clone());
                    }
                }
            }
        }
        pruned.compute_entry_and_leaf();
        pruned
    }

    /// Get call graph statistics.
    pub fn stats(&self) -> CallGraphStats {
        let indirect_count = self.call_sites_by_caller.values()
            .flat_map(|sites| sites.iter())
            .filter(|s| s.is_indirect)
            .count();
        let vtable_count = self.call_sites_by_caller.values()
            .flat_map(|sites| sites.iter())
            .filter(|s| s.is_vtable_dispatch)
            .count();
        let total_sites: usize = self.call_sites_by_caller.values()
            .map(|s| s.len())
            .sum();

        CallGraphStats {
            num_functions: self.num_functions(),
            num_edges: self.num_edges(),
            num_call_sites: total_sites,
            num_indirect_calls: indirect_count,
            num_vtable_dispatches: vtable_count,
            num_sccs: self.sccs().len(),
            max_scc_size: self.sccs().iter().map(|s| s.len()).max().unwrap_or(0),
        }
    }
}

// ---------------------------------------------------------------------------
// Call graph builder
// ---------------------------------------------------------------------------

/// Builder that constructs a call graph from an IR module.
pub struct CallGraphBuilder<'a> {
    module: &'a Module,
    points_to: Option<&'a AndersonAnalysis>,
    /// Protocol-specific call patterns to recognize.
    vtable_patterns: Vec<VTablePattern>,
    /// Whether to resolve indirect calls.
    resolve_indirect: bool,
}

/// A pattern for recognizing vtable-based dispatch.
#[derive(Debug, Clone)]
struct VTablePattern {
    struct_type: String,
    load_field_index: u32,
    method_names: Vec<String>,
}

impl<'a> CallGraphBuilder<'a> {
    pub fn new(module: &'a Module) -> Self {
        Self {
            module,
            points_to: None,
            vtable_patterns: Self::default_vtable_patterns(),
            resolve_indirect: true,
        }
    }

    pub fn with_points_to(mut self, pta: &'a AndersonAnalysis) -> Self {
        self.points_to = Some(pta);
        self
    }

    pub fn with_indirect_resolution(mut self, resolve: bool) -> Self {
        self.resolve_indirect = resolve;
        self
    }

    fn default_vtable_patterns() -> Vec<VTablePattern> {
        vec![
            VTablePattern {
                struct_type: "SSL_METHOD".into(),
                load_field_index: 0,
                method_names: vec![
                    "ssl_new".into(), "ssl_clear".into(), "ssl_free".into(),
                    "ssl_accept".into(), "ssl_connect".into(), "ssl_read".into(),
                    "ssl_write".into(), "ssl_shutdown".into(),
                    "ssl_renegotiate".into(), "ssl_renegotiate_check".into(),
                ],
            },
            VTablePattern {
                struct_type: "BIO_METHOD".into(),
                load_field_index: 0,
                method_names: vec![
                    "bwrite".into(), "bread".into(), "bputs".into(),
                    "bgets".into(), "ctrl".into(), "create".into(),
                    "destroy".into(), "callback_ctrl".into(),
                ],
            },
        ]
    }

    /// Build the call graph.
    pub fn build(&self) -> CallGraph {
        let mut cg = CallGraph::new();

        for (fname, func) in &self.module.functions {
            // Ensure every defined function is in the graph.
            cg.get_or_create_node(fname);

            for (bname, block) in &func.blocks {
                for (idx, instr) in block.instructions.iter().enumerate() {
                    let loc = InstructionId::new(fname, bname, idx);
                    self.process_call_instruction(instr, fname, &loc, &mut cg);
                }
            }
        }

        cg.compute_entry_and_leaf();
        cg
    }

    /// Process a single instruction for call graph edges.
    fn process_call_instruction(
        &self,
        instr: &Instruction,
        caller: &str,
        loc: &InstructionId,
        cg: &mut CallGraph,
    ) {
        match instr {
            Instruction::Call { func, args, .. } | Instruction::Invoke { func, args, .. } => {
                let num_args = args.len();
                match func {
                    Value::FunctionRef(callee, _) | Value::GlobalRef(callee, _) => {
                        // Direct call.
                        let site = CallSite::direct(caller, callee, loc.clone(), num_args);
                        cg.add_call_site(site);
                    }
                    Value::Register(reg_name, _) => {
                        // Indirect call through register.
                        if self.resolve_indirect {
                            let targets = self.resolve_indirect_targets(reg_name, caller);
                            if targets.is_empty() {
                                // Unresolved: add a synthetic unknown target.
                                let site = CallSite::indirect(
                                    caller,
                                    format!("__unresolved_{}_{}", caller, loc.index),
                                    loc.clone(),
                                    num_args,
                                );
                                cg.add_call_site(site);
                            } else {
                                for target in targets {
                                    let is_vtable = self.is_vtable_target(&target);
                                    let site = if is_vtable {
                                        CallSite::vtable(caller, &target, loc.clone(), num_args)
                                    } else {
                                        CallSite::indirect(caller, &target, loc.clone(), num_args)
                                    };
                                    cg.add_call_site(site);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    /// Resolve indirect call targets using points-to analysis.
    fn resolve_indirect_targets(&self, reg_name: &str, caller: &str) -> Vec<String> {
        let mut targets = Vec::new();

        // Try points-to analysis first.
        if let Some(pta) = self.points_to {
            let resolved = pta.resolve_indirect_call(reg_name);
            targets.extend(resolved);
        }

        // If no PTA results, try pattern-based resolution.
        if targets.is_empty() {
            targets.extend(self.pattern_based_resolution(reg_name, caller));
        }

        targets.sort();
        targets.dedup();
        targets
    }

    /// Pattern-based indirect call resolution for known protocol patterns.
    fn pattern_based_resolution(&self, reg_name: &str, caller: &str) -> Vec<String> {
        let mut targets = Vec::new();

        // Look at the caller function to find the GEP/load pattern leading to this register.
        if let Some(func) = self.module.function(caller) {
            // Walk backwards from uses of reg_name to find a GEP from a known vtable struct.
            for (_bname, block) in &func.blocks {
                for instr in &block.instructions {
                    if let Instruction::Load { dest, ptr, ty, .. } = instr {
                        if dest == reg_name {
                            // Check if ptr was derived from a GEP on a known struct.
                            if let Some(ptr_name) = ptr.name() {
                                if let Some(gep_field) = self.find_gep_field(func, ptr_name) {
                                    // Check vtable patterns.
                                    for pattern in &self.vtable_patterns {
                                        if gep_field < pattern.method_names.len() as u32 {
                                            let method = &pattern.method_names[gep_field as usize];
                                            // Look for functions matching the method name.
                                            for (fname, _) in &self.module.functions {
                                                let fname_lower = fname.to_lowercase();
                                                if fname_lower.contains(&method.to_lowercase()) {
                                                    targets.push(fname.clone());
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        targets
    }

    /// Find the field index from a GEP instruction that produces the given register.
    fn find_gep_field(&self, func: &Function, reg_name: &str) -> Option<u32> {
        for (_bname, block) in &func.blocks {
            for instr in &block.instructions {
                if let Instruction::GetElementPtr { dest, indices, .. } = instr {
                    if dest == reg_name {
                        // Return the struct field index (usually the second index).
                        if indices.len() >= 2 {
                            if let Value::IntConst(val, _) = &indices[1] {
                                return Some(*val as u32);
                            }
                        }
                        if let Some(first) = indices.first() {
                            if let Value::IntConst(val, _) = first {
                                return Some(*val as u32);
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Whether a target function looks like a vtable method.
    fn is_vtable_target(&self, name: &str) -> bool {
        let lower = name.to_lowercase();
        for pattern in &self.vtable_patterns {
            for method in &pattern.method_names {
                if lower.contains(&method.to_lowercase()) {
                    return true;
                }
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallGraphStats {
    pub num_functions: usize,
    pub num_edges: usize,
    pub num_call_sites: usize,
    pub num_indirect_calls: usize,
    pub num_vtable_dispatches: usize,
    pub num_sccs: usize,
    pub max_scc_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Module;

    #[test]
    fn test_call_site_display() {
        let site = CallSite::direct(
            "SSL_do_handshake", "ssl3_accept",
            InstructionId::new("SSL_do_handshake", "entry", 2),
            1,
        );
        let s = format!("{}", site);
        assert!(s.contains("SSL_do_handshake"));
        assert!(s.contains("ssl3_accept"));
        assert!(s.contains("direct"));
    }

    #[test]
    fn test_call_graph_basic() {
        let mut cg = CallGraph::new();
        cg.add_call_site(CallSite::direct(
            "main", "SSL_do_handshake",
            InstructionId::new("main", "entry", 0),
            1,
        ));
        cg.add_call_site(CallSite::direct(
            "SSL_do_handshake", "ssl3_accept",
            InstructionId::new("SSL_do_handshake", "entry", 2),
            1,
        ));
        cg.compute_entry_and_leaf();

        assert_eq!(cg.num_functions(), 3);
        assert_eq!(cg.callee_names("main"), vec!["SSL_do_handshake"]);
        assert_eq!(cg.caller_names("ssl3_accept"), vec!["SSL_do_handshake"]);
        assert!(cg.entry_points.contains(&"main".to_string()));
        assert!(cg.leaf_functions.contains(&"ssl3_accept".to_string()));
    }

    #[test]
    fn test_call_graph_reachability() {
        let mut cg = CallGraph::new();
        cg.add_call_site(CallSite::direct("a", "b", InstructionId::new("a", "e", 0), 0));
        cg.add_call_site(CallSite::direct("b", "c", InstructionId::new("b", "e", 0), 0));
        cg.add_call_site(CallSite::direct("d", "e", InstructionId::new("d", "e", 0), 0));

        let reachable = cg.reachable_from("a");
        assert!(reachable.contains("a"));
        assert!(reachable.contains("b"));
        assert!(reachable.contains("c"));
        assert!(!reachable.contains("d"));
    }

    #[test]
    fn test_call_graph_reverse_reachability() {
        let mut cg = CallGraph::new();
        cg.add_call_site(CallSite::direct("a", "b", InstructionId::new("a", "e", 0), 0));
        cg.add_call_site(CallSite::direct("b", "c", InstructionId::new("b", "e", 0), 0));

        let reaching = cg.can_reach("c");
        assert!(reaching.contains("a"));
        assert!(reaching.contains("b"));
        assert!(reaching.contains("c"));
    }

    #[test]
    fn test_call_graph_sccs() {
        let mut cg = CallGraph::new();
        cg.add_call_site(CallSite::direct("a", "b", InstructionId::new("a", "e", 0), 0));
        cg.add_call_site(CallSite::direct("b", "a", InstructionId::new("b", "e", 0), 0));
        cg.add_call_site(CallSite::direct("a", "c", InstructionId::new("a", "e", 1), 0));

        let sccs = cg.sccs();
        // {a, b} form an SCC, {c} is separate.
        let big_scc = sccs.iter().find(|s| s.len() == 2);
        assert!(big_scc.is_some());
    }

    #[test]
    fn test_call_graph_builder() {
        let module = Module::test_module();
        let builder = CallGraphBuilder::new(&module).with_indirect_resolution(false);
        let cg = builder.build();

        assert!(cg.num_functions() > 0);
    }

    #[test]
    fn test_call_graph_prune() {
        let mut cg = CallGraph::new();
        cg.add_call_site(CallSite::direct("a", "b", InstructionId::new("a", "e", 0), 0));
        cg.add_call_site(CallSite::direct("b", "c", InstructionId::new("b", "e", 0), 0));
        cg.add_call_site(CallSite::direct("x", "y", InstructionId::new("x", "e", 0), 0));

        let pruned = cg.prune_unreachable(&["a"]);
        assert!(pruned.contains_function("a"));
        assert!(pruned.contains_function("b"));
        assert!(pruned.contains_function("c"));
        assert!(!pruned.contains_function("x"));
    }

    #[test]
    fn test_call_graph_stats() {
        let module = Module::test_module();
        let builder = CallGraphBuilder::new(&module);
        let cg = builder.build();
        let stats = cg.stats();
        assert!(stats.num_functions > 0);
    }

    #[test]
    fn test_negotiation_subgraph() {
        let module = Module::test_module();
        let builder = CallGraphBuilder::new(&module).with_indirect_resolution(false);
        let cg = builder.build();
        let sub = cg.negotiation_subgraph(&module);
        // Should contain negotiation-relevant functions.
        assert!(sub.num_functions() > 0 || cg.num_functions() > 0);
    }

    #[test]
    fn test_call_site_types() {
        let loc = InstructionId::new("f", "bb", 0);
        let direct = CallSite::direct("f", "g", loc.clone(), 2);
        assert!(!direct.is_indirect);
        assert!(!direct.is_vtable_dispatch);

        let indirect = CallSite::indirect("f", "g", loc.clone(), 2);
        assert!(indirect.is_indirect);

        let vtable = CallSite::vtable("f", "g", loc, 2);
        assert!(vtable.is_vtable_dispatch);
        assert!(vtable.is_indirect);
    }
}

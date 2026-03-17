//! Points-to analysis for resolving indirect calls and vtable dispatch.
//!
//! Implements Andersen-style inclusion-based and Steensgaard-style
//! union-find-based pointer analyses, with protocol-specific extensions
//! for SSL_METHOD vtable resolution.

use std::collections::{HashMap, HashSet, VecDeque, BTreeSet};
use std::fmt;
use serde::{Serialize, Deserialize};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use indexmap::IndexMap;

use crate::ir::{Module, Function, BasicBlock, Instruction, Value, Type};
use crate::{InstructionId, SlicerError, SlicerResult};

// ---------------------------------------------------------------------------
// Abstract locations
// ---------------------------------------------------------------------------

/// An abstract memory location in the points-to analysis.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AbstractLocation {
    /// Stack allocation: function + alloca name.
    Stack { function: String, variable: String },
    /// Heap allocation: identified by call-site.
    Heap { alloc_site: InstructionId },
    /// Global variable.
    Global { name: String },
    /// A VTable struct (SSL_METHOD, BIO_METHOD, etc.).
    VTable { type_name: String, instance: String },
    /// A specific slot within a callback/vtable struct.
    CallbackSlot { base: Box<AbstractLocation>, field_index: u32 },
    /// Function pointer.
    FunctionPtr { name: String },
    /// Unknown / external location.
    Unknown { label: String },
    /// Null location.
    Null,
}

impl AbstractLocation {
    pub fn stack(func: impl Into<String>, var: impl Into<String>) -> Self {
        AbstractLocation::Stack { function: func.into(), variable: var.into() }
    }
    pub fn global(name: impl Into<String>) -> Self {
        AbstractLocation::Global { name: name.into() }
    }
    pub fn heap(site: InstructionId) -> Self {
        AbstractLocation::Heap { alloc_site: site }
    }
    pub fn vtable(type_name: impl Into<String>, instance: impl Into<String>) -> Self {
        AbstractLocation::VTable { type_name: type_name.into(), instance: instance.into() }
    }
    pub fn func_ptr(name: impl Into<String>) -> Self {
        AbstractLocation::FunctionPtr { name: name.into() }
    }
    pub fn callback_slot(base: AbstractLocation, field: u32) -> Self {
        AbstractLocation::CallbackSlot { base: Box::new(base), field_index: field }
    }

    /// Whether this is a function pointer or callback slot.
    pub fn is_callable(&self) -> bool {
        matches!(self,
            AbstractLocation::FunctionPtr { .. } | AbstractLocation::CallbackSlot { .. }
        )
    }

    /// Whether this is a vtable.
    pub fn is_vtable(&self) -> bool {
        matches!(self, AbstractLocation::VTable { .. })
    }

    /// Get function name if this is a FunctionPtr.
    pub fn function_name(&self) -> Option<&str> {
        match self {
            AbstractLocation::FunctionPtr { name } => Some(name),
            _ => None,
        }
    }
}

impl fmt::Display for AbstractLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AbstractLocation::Stack { function, variable } =>
                write!(f, "stack({}.{})", function, variable),
            AbstractLocation::Heap { alloc_site } =>
                write!(f, "heap({})", alloc_site),
            AbstractLocation::Global { name } =>
                write!(f, "global({})", name),
            AbstractLocation::VTable { type_name, instance } =>
                write!(f, "vtable({}.{})", type_name, instance),
            AbstractLocation::CallbackSlot { base, field_index } =>
                write!(f, "{}.slot[{}]", base, field_index),
            AbstractLocation::FunctionPtr { name } =>
                write!(f, "fptr({})", name),
            AbstractLocation::Unknown { label } =>
                write!(f, "unknown({})", label),
            AbstractLocation::Null => write!(f, "null"),
        }
    }
}

// ---------------------------------------------------------------------------
// Points-to set
// ---------------------------------------------------------------------------

/// A set of abstract locations that a pointer may point to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PointsToSet {
    locations: BTreeSet<String>,
    location_objects: HashMap<String, AbstractLocation>,
}

impl PointsToSet {
    pub fn empty() -> Self {
        Self { locations: BTreeSet::new(), location_objects: HashMap::new() }
    }

    pub fn singleton(loc: AbstractLocation) -> Self {
        let mut pts = Self::empty();
        pts.insert(loc);
        pts
    }

    pub fn insert(&mut self, loc: AbstractLocation) -> bool {
        let key = format!("{}", loc);
        let new = self.locations.insert(key.clone());
        if new {
            self.location_objects.insert(key, loc);
        }
        new
    }

    pub fn contains(&self, loc: &AbstractLocation) -> bool {
        let key = format!("{}", loc);
        self.locations.contains(&key)
    }

    pub fn union(&self, other: &PointsToSet) -> PointsToSet {
        let mut result = self.clone();
        for (key, loc) in &other.location_objects {
            if result.locations.insert(key.clone()) {
                result.location_objects.insert(key.clone(), loc.clone());
            }
        }
        result
    }

    pub fn union_inplace(&mut self, other: &PointsToSet) -> bool {
        let mut changed = false;
        for (key, loc) in &other.location_objects {
            if self.locations.insert(key.clone()) {
                self.location_objects.insert(key.clone(), loc.clone());
                changed = true;
            }
        }
        changed
    }

    pub fn is_empty(&self) -> bool { self.locations.is_empty() }
    pub fn len(&self) -> usize { self.locations.len() }

    pub fn iter(&self) -> impl Iterator<Item = &AbstractLocation> {
        self.location_objects.values()
    }

    /// Get all function pointer targets.
    pub fn function_targets(&self) -> Vec<&str> {
        self.location_objects.values()
            .filter_map(|loc| loc.function_name())
            .collect()
    }

    /// Get all vtable locations.
    pub fn vtable_locations(&self) -> Vec<&AbstractLocation> {
        self.location_objects.values()
            .filter(|loc| loc.is_vtable())
            .collect()
    }

    /// Whether this set subsumes another (for fixed-point).
    pub fn subsumes(&self, other: &PointsToSet) -> bool {
        other.locations.is_subset(&self.locations)
    }
}

impl fmt::Display for PointsToSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{")?;
        for (i, loc) in self.location_objects.values().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{}", loc)?;
        }
        write!(f, "}}")
    }
}

// ---------------------------------------------------------------------------
// Points-to graph
// ---------------------------------------------------------------------------

/// Edge in the points-to constraint graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PtEdge {
    /// p → q means pts(p) ⊇ pts(q) (copy/assign).
    Copy,
    /// p → q means pts(p) ⊇ {q} (address-of).
    AddressOf,
    /// p → q means for all o ∈ pts(p), pts(o) ⊇ pts(q) (store *p = q).
    Store,
    /// p → q means for all o ∈ pts(q), pts(p) ⊇ pts(o) (load p = *q).
    Load,
    /// Field access at given offset.
    FieldAccess(u32),
}

/// Points-to constraint graph.
pub struct PointsToGraph {
    graph: DiGraph<String, PtEdge>,
    node_map: HashMap<String, NodeIndex>,
    points_to: HashMap<String, PointsToSet>,
}

impl PointsToGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_map: HashMap::new(),
            points_to: HashMap::new(),
        }
    }

    /// Get or create a node for a variable.
    fn get_or_create_node(&mut self, name: &str) -> NodeIndex {
        if let Some(&idx) = self.node_map.get(name) {
            idx
        } else {
            let idx = self.graph.add_node(name.to_string());
            self.node_map.insert(name.to_string(), idx);
            idx
        }
    }

    /// Add a constraint edge.
    pub fn add_constraint(&mut self, from: &str, to: &str, kind: PtEdge) {
        let from_idx = self.get_or_create_node(from);
        let to_idx = self.get_or_create_node(to);
        self.graph.add_edge(from_idx, to_idx, kind);
    }

    /// Add a base address-of constraint: pts(p) ⊇ {loc}.
    pub fn add_addr_of(&mut self, ptr: &str, loc: AbstractLocation) {
        let pts = self.points_to.entry(ptr.to_string()).or_insert_with(PointsToSet::empty);
        pts.insert(loc);
    }

    /// Get the points-to set for a variable.
    pub fn get_points_to(&self, var: &str) -> PointsToSet {
        self.points_to.get(var).cloned().unwrap_or_else(PointsToSet::empty)
    }

    /// Number of variables tracked.
    pub fn num_variables(&self) -> usize {
        self.node_map.len()
    }

    /// Number of constraint edges.
    pub fn num_constraints(&self) -> usize {
        self.graph.edge_count()
    }

    /// All variable names.
    pub fn variables(&self) -> Vec<&str> {
        self.node_map.keys().map(|s| s.as_str()).collect()
    }
}

// ---------------------------------------------------------------------------
// Andersen's Analysis (inclusion-based)
// ---------------------------------------------------------------------------

/// Andersen's inclusion-based points-to analysis.
pub struct AndersonAnalysis {
    graph: PointsToGraph,
    max_iterations: usize,
    /// Whether to use context-sensitivity for SSL_METHOD vtables.
    context_sensitive_vtables: bool,
}

impl AndersonAnalysis {
    pub fn new(max_iterations: usize) -> Self {
        Self {
            graph: PointsToGraph::new(),
            max_iterations,
            context_sensitive_vtables: true,
        }
    }

    /// Build constraints from a module.
    pub fn build_constraints(&mut self, module: &Module) {
        for (fname, func) in &module.functions {
            self.process_function(func);
        }
        // Process global initializers.
        for (gname, global) in &module.globals {
            let loc = AbstractLocation::global(gname.clone());
            self.graph.add_addr_of(gname, loc.clone());

            if let Some(ref init) = global.initializer {
                self.process_global_initializer(gname, init);
            }
        }
    }

    /// Process a single function to extract constraints.
    fn process_function(&mut self, func: &Function) {
        // Parameters get stack locations.
        for (pname, _pty) in &func.params {
            let loc = AbstractLocation::stack(&func.name, pname);
            self.graph.add_addr_of(pname, loc);
        }

        for (bname, block) in &func.blocks {
            for (idx, instr) in block.instructions.iter().enumerate() {
                self.process_instruction(instr, &func.name, bname, idx);
            }
        }
    }

    /// Process a single instruction to extract points-to constraints.
    fn process_instruction(
        &mut self,
        instr: &Instruction,
        func_name: &str,
        block_name: &str,
        idx: usize,
    ) {
        match instr {
            Instruction::Alloca { dest, .. } => {
                let loc = AbstractLocation::stack(func_name, dest);
                self.graph.add_addr_of(dest, loc);
            }
            Instruction::Load { dest, ptr, .. } => {
                // dest = *ptr  →  Load constraint
                if let Some(ptr_name) = ptr.name() {
                    self.graph.add_constraint(dest, ptr_name, PtEdge::Load);
                }
            }
            Instruction::Store { value, ptr, .. } => {
                // *ptr = value  →  Store constraint
                if let (Some(val_name), Some(ptr_name)) = (value.name(), ptr.name()) {
                    self.graph.add_constraint(ptr_name, val_name, PtEdge::Store);
                }
            }
            Instruction::GetElementPtr { dest, ptr, indices, inbounds, .. } => {
                // GEP: dest points to a field of what ptr points to.
                if let Some(ptr_name) = ptr.name() {
                    let field_idx = self.extract_field_index(indices);
                    self.graph.add_constraint(dest, ptr_name, PtEdge::FieldAccess(field_idx));
                    // Also add a copy for conservative correctness.
                    self.graph.add_constraint(dest, ptr_name, PtEdge::Copy);
                }
            }
            Instruction::Cast { dest, value, .. } => {
                // Casts (bitcast, inttoptr, etc.) are copies.
                if let Some(val_name) = value.name() {
                    self.graph.add_constraint(dest, val_name, PtEdge::Copy);
                }
            }
            Instruction::Phi { dest, incoming, .. } => {
                for (val, _bb) in incoming {
                    if let Some(val_name) = val.name() {
                        self.graph.add_constraint(dest, val_name, PtEdge::Copy);
                    }
                }
            }
            Instruction::Select { dest, true_val, false_val, .. } => {
                if let Some(tv_name) = true_val.name() {
                    self.graph.add_constraint(dest, tv_name, PtEdge::Copy);
                }
                if let Some(fv_name) = false_val.name() {
                    self.graph.add_constraint(dest, fv_name, PtEdge::Copy);
                }
            }
            Instruction::Call { dest, func, args, .. } => {
                // If calling a known allocation function, create a heap location.
                if let Some(callee) = func.name() {
                    let is_alloc = callee == "malloc" || callee == "calloc"
                        || callee == "OPENSSL_malloc" || callee == "CRYPTO_malloc"
                        || callee == "OPENSSL_zalloc";
                    if is_alloc {
                        if let Some(d) = dest {
                            let loc = AbstractLocation::heap(
                                InstructionId::new(func_name, block_name, idx)
                            );
                            self.graph.add_addr_of(d, loc);
                        }
                    } else {
                        // For other calls: conservatively, return may alias args.
                        if let Some(d) = dest {
                            for arg in args {
                                if let Some(arg_name) = arg.name() {
                                    if arg.ty().is_pointer() {
                                        self.graph.add_constraint(d, arg_name, PtEdge::Copy);
                                    }
                                }
                            }
                        }
                    }

                    // Handle SSL_METHOD vtable creation.
                    if self.context_sensitive_vtables && self.is_vtable_creator(callee) {
                        if let Some(d) = dest {
                            let vtable_loc = AbstractLocation::vtable(
                                self.vtable_type_for(callee),
                                format!("{}@{}:{}", func_name, block_name, idx),
                            );
                            self.graph.add_addr_of(d, vtable_loc);
                        }
                    }
                }
                // For indirect calls through function pointers.
                if func.name().is_none() {
                    // The function operand is a computed value — we'll resolve later.
                    if let Some(fn_reg) = func.name() {
                        if let Some(d) = dest {
                            self.graph.add_constraint(d, fn_reg, PtEdge::Load);
                        }
                    }
                }
            }
            Instruction::ExtractValue { dest, aggregate, indices, .. } => {
                if let Some(agg_name) = aggregate.name() {
                    let field = indices.first().copied().unwrap_or(0);
                    self.graph.add_constraint(dest, agg_name, PtEdge::FieldAccess(field));
                }
            }
            _ => {}
        }
    }

    fn process_global_initializer(&mut self, gname: &str, init: &Value) {
        match init {
            Value::FunctionRef(fname, _) => {
                let loc = AbstractLocation::func_ptr(fname.clone());
                self.graph.add_addr_of(gname, loc);
            }
            Value::GlobalRef(ref_name, _) => {
                self.graph.add_constraint(gname, ref_name, PtEdge::Copy);
            }
            Value::Aggregate(vals, _) => {
                for (i, val) in vals.iter().enumerate() {
                    let field_name = format!("{}.{}", gname, i);
                    self.process_global_initializer(&field_name, val);
                }
            }
            _ => {}
        }
    }

    /// Run the Andersen's fixed-point solver.
    pub fn solve(&mut self) -> SlicerResult<()> {
        let mut worklist: VecDeque<String> = self.graph.node_map.keys().cloned().collect();
        let mut iterations = 0;

        while let Some(var) = worklist.pop_front() {
            iterations += 1;
            if iterations > self.max_iterations {
                return Err(SlicerError::PointsToTimeout { iterations: self.max_iterations });
            }

            let node_idx = match self.graph.node_map.get(&var) {
                Some(&idx) => idx,
                None => continue,
            };

            // Process all outgoing edges from this node.
            let edges: Vec<(NodeIndex, PtEdge)> = self.graph.graph
                .edges(node_idx)
                .map(|e| (e.target(), e.weight().clone()))
                .collect();

            let mut changed = false;
            for (target_idx, edge_kind) in edges {
                let target_name = self.graph.graph[target_idx].clone();
                match edge_kind {
                    PtEdge::Copy => {
                        let src_pts = self.graph.get_points_to(&target_name);
                        let dst_pts = self.graph.points_to
                            .entry(var.clone())
                            .or_insert_with(PointsToSet::empty);
                        if dst_pts.union_inplace(&src_pts) {
                            changed = true;
                        }
                    }
                    PtEdge::AddressOf => {
                        // Already handled during constraint generation.
                    }
                    PtEdge::Load => {
                        // p = *q: for each o ∈ pts(q), pts(p) ⊇ pts(o)
                        let q_pts = self.graph.get_points_to(&target_name);
                        let mut to_add = PointsToSet::empty();
                        for loc in q_pts.iter() {
                            let loc_key = format!("{}", loc);
                            let o_pts = self.graph.get_points_to(&loc_key);
                            to_add = to_add.union(&o_pts);
                        }
                        let p_pts = self.graph.points_to
                            .entry(var.clone())
                            .or_insert_with(PointsToSet::empty);
                        if p_pts.union_inplace(&to_add) {
                            changed = true;
                        }
                    }
                    PtEdge::Store => {
                        // *p = q: for each o ∈ pts(p), pts(o) ⊇ pts(q)
                        let p_pts = self.graph.get_points_to(&var);
                        let q_pts = self.graph.get_points_to(&target_name);
                        let p_locs: Vec<String> = p_pts.iter().map(|l| format!("{}", l)).collect();
                        for loc_key in p_locs {
                            let o_pts = self.graph.points_to
                                .entry(loc_key.clone())
                                .or_insert_with(PointsToSet::empty);
                            if o_pts.union_inplace(&q_pts) {
                                changed = true;
                                if !worklist.contains(&loc_key) {
                                    worklist.push_back(loc_key);
                                }
                            }
                        }
                    }
                    PtEdge::FieldAccess(field) => {
                        let src_pts = self.graph.get_points_to(&target_name);
                        let mut field_pts = PointsToSet::empty();
                        for loc in src_pts.iter() {
                            let field_loc = AbstractLocation::callback_slot(loc.clone(), field);
                            field_pts.insert(field_loc);
                        }
                        let dst_pts = self.graph.points_to
                            .entry(var.clone())
                            .or_insert_with(PointsToSet::empty);
                        if dst_pts.union_inplace(&field_pts) {
                            changed = true;
                        }
                    }
                }
            }

            if changed {
                // Re-enqueue nodes that depend on this one.
                let incoming: Vec<String> = self.graph.graph
                    .neighbors_directed(node_idx, petgraph::Direction::Incoming)
                    .map(|n| self.graph.graph[n].clone())
                    .collect();
                for n in incoming {
                    if !worklist.contains(&n) {
                        worklist.push_back(n);
                    }
                }
            }
        }

        log::debug!("Andersen's analysis converged in {} iterations", iterations);
        Ok(())
    }

    /// Get the resolved points-to set for a variable.
    pub fn query(&self, var: &str) -> PointsToSet {
        self.graph.get_points_to(var)
    }

    /// Resolve indirect call targets for a function pointer variable.
    pub fn resolve_indirect_call(&self, func_ptr_var: &str) -> Vec<String> {
        let pts = self.graph.get_points_to(func_ptr_var);
        let mut targets = Vec::new();
        for loc in pts.iter() {
            match loc {
                AbstractLocation::FunctionPtr { name } => {
                    targets.push(name.clone());
                }
                AbstractLocation::CallbackSlot { base, field_index } => {
                    // Look up what the base vtable slot points to.
                    let slot_key = format!("{}", loc);
                    let slot_pts = self.graph.get_points_to(&slot_key);
                    for target in slot_pts.iter() {
                        if let Some(fname) = target.function_name() {
                            targets.push(fname.to_string());
                        }
                    }
                }
                _ => {}
            }
        }
        targets.sort();
        targets.dedup();
        targets
    }

    /// Whether a callee name looks like a vtable creator.
    fn is_vtable_creator(&self, callee: &str) -> bool {
        let lower = callee.to_lowercase();
        lower.contains("method") && (lower.contains("tls") || lower.contains("ssl")
            || lower.contains("dtls"))
            || lower == "bio_s_mem" || lower == "bio_s_socket"
    }

    /// Infer vtable type name from creator function.
    fn vtable_type_for(&self, callee: &str) -> String {
        if callee.contains("TLS") || callee.contains("tls") {
            "SSL_METHOD".into()
        } else if callee.contains("DTLS") || callee.contains("dtls") {
            "SSL_METHOD".into()
        } else if callee.contains("BIO") || callee.contains("bio") {
            "BIO_METHOD".into()
        } else {
            "UNKNOWN_VTABLE".into()
        }
    }

    /// Extract field index from GEP indices.
    fn extract_field_index(&self, indices: &[Value]) -> u32 {
        // Typically the second index is the struct field.
        if indices.len() >= 2 {
            if let Value::IntConst(val, _) = &indices[1] {
                return *val as u32;
            }
        }
        0
    }

    /// Statistics about the analysis.
    pub fn stats(&self) -> PointsToStats {
        let total_vars = self.graph.num_variables();
        let total_constraints = self.graph.num_constraints();
        let total_pts_entries: usize = self.graph.points_to.values().map(|p| p.len()).sum();
        let max_pts_size = self.graph.points_to.values().map(|p| p.len()).max().unwrap_or(0);
        PointsToStats {
            total_variables: total_vars,
            total_constraints,
            total_points_to_entries: total_pts_entries,
            max_points_to_size: max_pts_size,
        }
    }
}

// ---------------------------------------------------------------------------
// Steensgaard's Analysis (union-find based)
// ---------------------------------------------------------------------------

/// Union-Find data structure for Steensgaard's analysis.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) -> bool {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry { return false; }
        if self.rank[rx] < self.rank[ry] {
            self.parent[rx] = ry;
        } else if self.rank[rx] > self.rank[ry] {
            self.parent[ry] = rx;
        } else {
            self.parent[ry] = rx;
            self.rank[rx] += 1;
        }
        true
    }

    fn same_set(&mut self, x: usize, y: usize) -> bool {
        self.find(x) == self.find(y)
    }
}

/// Steensgaard's fast unification-based points-to analysis.
/// Less precise than Andersen's but runs in near-linear time.
pub struct SteensgaardAnalysis {
    var_to_id: HashMap<String, usize>,
    id_to_var: Vec<String>,
    uf: UnionFind,
    loc_map: HashMap<usize, PointsToSet>,
}

impl SteensgaardAnalysis {
    pub fn new() -> Self {
        Self {
            var_to_id: HashMap::new(),
            id_to_var: Vec::new(),
            uf: UnionFind::new(0),
            loc_map: HashMap::new(),
        }
    }

    /// Get or allocate an ID for a variable.
    fn get_id(&mut self, var: &str) -> usize {
        if let Some(&id) = self.var_to_id.get(var) {
            id
        } else {
            let id = self.id_to_var.len();
            self.var_to_id.insert(var.to_string(), id);
            self.id_to_var.push(var.to_string());
            // Grow union-find.
            self.uf.parent.push(id);
            self.uf.rank.push(0);
            id
        }
    }

    /// Build and solve from a module.
    pub fn analyze(&mut self, module: &Module) {
        // First pass: collect all variables.
        for (fname, func) in &module.functions {
            for (pname, _) in &func.params {
                self.get_id(pname);
            }
            for (_bname, block) in &func.blocks {
                for instr in &block.instructions {
                    if let Some(dest) = instr.dest() {
                        self.get_id(dest);
                    }
                    for reg in instr.used_registers() {
                        self.get_id(reg);
                    }
                }
            }
        }
        for (gname, _) in &module.globals {
            self.get_id(gname);
        }

        // Second pass: process constraints.
        for (fname, func) in &module.functions {
            for (pname, _) in &func.params {
                let loc = AbstractLocation::stack(&func.name, pname);
                let id = self.get_id(pname);
                self.loc_map.entry(id).or_insert_with(PointsToSet::empty).insert(loc);
            }
            for (bname, block) in &func.blocks {
                for (idx, instr) in block.instructions.iter().enumerate() {
                    self.process_instruction(instr, fname, bname, idx);
                }
            }
        }
    }

    fn process_instruction(
        &mut self,
        instr: &Instruction,
        func_name: &str,
        block_name: &str,
        idx: usize,
    ) {
        match instr {
            Instruction::Alloca { dest, .. } => {
                let id = self.get_id(dest);
                let loc = AbstractLocation::stack(func_name, dest);
                self.loc_map.entry(id).or_insert_with(PointsToSet::empty).insert(loc);
            }
            Instruction::Load { dest, ptr, .. } => {
                // Unify dest with dereferenced ptr contents.
                if let Some(ptr_name) = ptr.name() {
                    let dest_id = self.get_id(dest);
                    let ptr_id = self.get_id(ptr_name);
                    // In Steensgaard's, load is conceptually: pts(dest) = *pts(ptr).
                    // We approximate by merging the location sets.
                    let ptr_repr = self.uf.find(ptr_id);
                    let ptr_locs = self.loc_map.get(&ptr_repr).cloned()
                        .unwrap_or_else(PointsToSet::empty);
                    let dest_repr = self.uf.find(dest_id);
                    self.loc_map.entry(dest_repr)
                        .or_insert_with(PointsToSet::empty)
                        .union_inplace(&ptr_locs);
                }
            }
            Instruction::Store { value, ptr, .. } => {
                if let (Some(val_name), Some(ptr_name)) = (value.name(), ptr.name()) {
                    let val_id = self.get_id(val_name);
                    let ptr_id = self.get_id(ptr_name);
                    self.uf.union(val_id, ptr_id);
                }
            }
            Instruction::Cast { dest, value, .. } | Instruction::Freeze { dest, value, .. } => {
                if let Some(val_name) = value.name() {
                    let dest_id = self.get_id(dest);
                    let val_id = self.get_id(val_name);
                    self.uf.union(dest_id, val_id);
                }
            }
            Instruction::Phi { dest, incoming, .. } => {
                let dest_id = self.get_id(dest);
                for (val, _) in incoming {
                    if let Some(val_name) = val.name() {
                        let val_id = self.get_id(val_name);
                        self.uf.union(dest_id, val_id);
                    }
                }
            }
            Instruction::Select { dest, true_val, false_val, .. } => {
                let dest_id = self.get_id(dest);
                if let Some(name) = true_val.name() {
                    let true_id = self.get_id(name);
                    self.uf.union(dest_id, true_id);
                }
                if let Some(name) = false_val.name() {
                    let false_id = self.get_id(name);
                    self.uf.union(dest_id, false_id);
                }
            }
            Instruction::GetElementPtr { dest, ptr, .. } => {
                if let Some(ptr_name) = ptr.name() {
                    let dest_id = self.get_id(dest);
                    let ptr_id = self.get_id(ptr_name);
                    self.uf.union(dest_id, ptr_id);
                }
            }
            Instruction::Call { dest, func, args, .. } => {
                if let Some(callee) = func.name() {
                    let is_alloc = callee == "malloc" || callee == "calloc"
                        || callee == "OPENSSL_malloc";
                    if is_alloc {
                        if let Some(d) = dest {
                            let id = self.get_id(d);
                            let loc = AbstractLocation::heap(
                                InstructionId::new(func_name, block_name, idx)
                            );
                            self.loc_map.entry(id).or_insert_with(PointsToSet::empty).insert(loc);
                        }
                    }
                }
                // Unify return with pointer args (conservative).
                if let Some(d) = dest {
                    let dest_id = self.get_id(d);
                    for arg in args {
                        if let Some(arg_name) = arg.name() {
                            if arg.ty().is_pointer() {
                                let arg_id = self.get_id(arg_name);
                                self.uf.union(dest_id, arg_id);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Query the points-to set for a variable.
    pub fn query(&mut self, var: &str) -> PointsToSet {
        let id = match self.var_to_id.get(var) {
            Some(&id) => id,
            None => return PointsToSet::empty(),
        };
        let repr = self.uf.find(id);

        // Collect all variables in the same equivalence class.
        let mut result = self.loc_map.get(&repr).cloned().unwrap_or_else(PointsToSet::empty);

        // Also collect from all members of the equivalence class.
        for i in 0..self.id_to_var.len() {
            if self.uf.find(i) == repr {
                if let Some(locs) = self.loc_map.get(&i) {
                    result.union_inplace(locs);
                }
            }
        }
        result
    }

    /// Number of equivalence classes.
    pub fn num_classes(&mut self) -> usize {
        let n = self.id_to_var.len();
        let mut roots = HashSet::new();
        for i in 0..n {
            roots.insert(self.uf.find(i));
        }
        roots.len()
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointsToStats {
    pub total_variables: usize,
    pub total_constraints: usize,
    pub total_points_to_entries: usize,
    pub max_points_to_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Module, Function, BasicBlock, Instruction, Value, Type};

    #[test]
    fn test_abstract_location_display() {
        let loc = AbstractLocation::stack("foo", "x");
        assert_eq!(format!("{}", loc), "stack(foo.x)");
        let glob = AbstractLocation::global("SSL_CTX_data");
        assert_eq!(format!("{}", glob), "global(SSL_CTX_data)");
    }

    #[test]
    fn test_points_to_set_operations() {
        let mut pts = PointsToSet::empty();
        assert!(pts.is_empty());

        pts.insert(AbstractLocation::stack("f", "x"));
        assert_eq!(pts.len(), 1);

        pts.insert(AbstractLocation::stack("f", "y"));
        assert_eq!(pts.len(), 2);

        // Duplicate insert.
        let added = pts.insert(AbstractLocation::stack("f", "x"));
        assert!(!added);
        assert_eq!(pts.len(), 2);
    }

    #[test]
    fn test_points_to_set_union() {
        let mut a = PointsToSet::empty();
        a.insert(AbstractLocation::stack("f", "x"));
        let mut b = PointsToSet::empty();
        b.insert(AbstractLocation::stack("f", "y"));

        let c = a.union(&b);
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn test_points_to_set_subsumes() {
        let mut a = PointsToSet::empty();
        a.insert(AbstractLocation::stack("f", "x"));
        a.insert(AbstractLocation::stack("f", "y"));

        let mut b = PointsToSet::empty();
        b.insert(AbstractLocation::stack("f", "x"));

        assert!(a.subsumes(&b));
        assert!(!b.subsumes(&a));
    }

    #[test]
    fn test_points_to_graph() {
        let mut graph = PointsToGraph::new();
        graph.add_addr_of("p", AbstractLocation::stack("f", "x"));
        graph.add_constraint("q", "p", PtEdge::Copy);

        assert_eq!(graph.num_variables(), 2);
        assert_eq!(graph.num_constraints(), 1);
        assert_eq!(graph.get_points_to("p").len(), 1);
    }

    #[test]
    fn test_anderson_basic() {
        let module = Module::test_module();
        let mut analysis = AndersonAnalysis::new(100);
        analysis.build_constraints(&module);
        analysis.solve().unwrap();

        let stats = analysis.stats();
        assert!(stats.total_variables > 0);
    }

    #[test]
    fn test_anderson_alloca() {
        let mut module = Module::new("test");
        let mut f = Function::new("test_func", Type::Void);
        {
            let bb = f.add_block("entry");
            bb.push(Instruction::Alloca {
                dest: "ptr".into(),
                ty: Type::i32(),
                num_elements: None,
                align: Some(4),
            });
            bb.push(Instruction::Ret { value: None });
        }
        module.add_function(f);

        let mut analysis = AndersonAnalysis::new(100);
        analysis.build_constraints(&module);
        analysis.solve().unwrap();

        let pts = analysis.query("ptr");
        assert!(!pts.is_empty());
    }

    #[test]
    fn test_union_find() {
        let mut uf = UnionFind::new(5);
        assert!(!uf.same_set(0, 1));
        uf.union(0, 1);
        assert!(uf.same_set(0, 1));
        uf.union(2, 3);
        assert!(!uf.same_set(1, 2));
        uf.union(1, 2);
        assert!(uf.same_set(0, 3));
    }

    #[test]
    fn test_steensgaard_basic() {
        let module = Module::test_module();
        let mut analysis = SteensgaardAnalysis::new();
        analysis.analyze(&module);

        let classes = analysis.num_classes();
        assert!(classes > 0);
    }

    #[test]
    fn test_resolve_indirect_call() {
        let mut graph = PointsToGraph::new();
        graph.add_addr_of("fptr", AbstractLocation::func_ptr("ssl3_accept"));
        let analysis = AndersonAnalysis {
            graph,
            max_iterations: 100,
            context_sensitive_vtables: true,
        };
        let targets = analysis.resolve_indirect_call("fptr");
        assert_eq!(targets, vec!["ssl3_accept"]);
    }

    #[test]
    fn test_abstract_location_callable() {
        assert!(AbstractLocation::func_ptr("f").is_callable());
        assert!(AbstractLocation::callback_slot(
            AbstractLocation::vtable("SSL_METHOD", "inst"), 3
        ).is_callable());
        assert!(!AbstractLocation::stack("f", "x").is_callable());
    }

    #[test]
    fn test_function_targets() {
        let mut pts = PointsToSet::empty();
        pts.insert(AbstractLocation::func_ptr("ssl_read"));
        pts.insert(AbstractLocation::func_ptr("ssl_write"));
        pts.insert(AbstractLocation::stack("f", "x"));

        let targets = pts.function_targets();
        assert_eq!(targets.len(), 2);
    }

    #[test]
    fn test_vtable_locations() {
        let mut pts = PointsToSet::empty();
        pts.insert(AbstractLocation::vtable("SSL_METHOD", "tls12"));
        pts.insert(AbstractLocation::stack("f", "x"));

        assert_eq!(pts.vtable_locations().len(), 1);
    }
}

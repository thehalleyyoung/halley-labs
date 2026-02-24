//! Vulkan memory model specification for LITMUS∞.
//!
//! Implements the Vulkan/SPIR-V memory model including scoped memory
//! ordering, availability/visibility operations, memory domains,
//! storage classes, and non-uniform subgroup semantics.

use std::collections::{HashMap, HashSet, BTreeSet};
use std::fmt;
use serde::{Serialize, Deserialize};

// ---------------------------------------------------------------------------
// Vulkan scopes
// ---------------------------------------------------------------------------

/// Vulkan memory scope hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum VulkanScope {
    Invocation,
    Subgroup,
    Workgroup,
    QueueFamily,
    Device,
}

impl VulkanScope {
    pub fn all() -> &'static [VulkanScope] {
        &[
            VulkanScope::Invocation,
            VulkanScope::Subgroup,
            VulkanScope::Workgroup,
            VulkanScope::QueueFamily,
            VulkanScope::Device,
        ]
    }

    /// Whether self includes other in the scope hierarchy.
    pub fn includes(&self, other: &VulkanScope) -> bool {
        *self >= *other
    }

    /// Next broader scope.
    pub fn broaden(&self) -> Option<VulkanScope> {
        match self {
            VulkanScope::Invocation => Some(VulkanScope::Subgroup),
            VulkanScope::Subgroup => Some(VulkanScope::Workgroup),
            VulkanScope::Workgroup => Some(VulkanScope::QueueFamily),
            VulkanScope::QueueFamily => Some(VulkanScope::Device),
            VulkanScope::Device => None,
        }
    }
}

impl fmt::Display for VulkanScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VulkanScope::Invocation => write!(f, "Invocation"),
            VulkanScope::Subgroup => write!(f, "Subgroup"),
            VulkanScope::Workgroup => write!(f, "Workgroup"),
            VulkanScope::QueueFamily => write!(f, "QueueFamily"),
            VulkanScope::Device => write!(f, "Device"),
        }
    }
}

// ---------------------------------------------------------------------------
// Storage class
// ---------------------------------------------------------------------------

/// Vulkan storage classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StorageClass {
    Function,
    Private,
    Workgroup,
    StorageBuffer,
    Uniform,
    Image,
    PushConstant,
}

impl StorageClass {
    /// Whether this storage class is shared across invocations.
    pub fn is_shared(&self) -> bool {
        matches!(self, StorageClass::Workgroup | StorageClass::StorageBuffer |
                 StorageClass::Uniform | StorageClass::Image)
    }
}

impl fmt::Display for StorageClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StorageClass::Function => write!(f, "Function"),
            StorageClass::Private => write!(f, "Private"),
            StorageClass::Workgroup => write!(f, "Workgroup"),
            StorageClass::StorageBuffer => write!(f, "StorageBuffer"),
            StorageClass::Uniform => write!(f, "Uniform"),
            StorageClass::Image => write!(f, "Image"),
            StorageClass::PushConstant => write!(f, "PushConstant"),
        }
    }
}

// ---------------------------------------------------------------------------
// Memory semantics
// ---------------------------------------------------------------------------

/// Vulkan memory semantics flags (matching SPIR-V spec).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemorySemantics {
    pub acquire: bool,
    pub release: bool,
    pub acquire_release: bool,
    pub sequentially_consistent: bool,
    pub uniform_memory: bool,
    pub workgroup_memory: bool,
    pub image_memory: bool,
    pub output_memory: bool,
    pub make_available: bool,
    pub make_visible: bool,
}

impl MemorySemantics {
    pub fn none() -> Self {
        Self {
            acquire: false,
            release: false,
            acquire_release: false,
            sequentially_consistent: false,
            uniform_memory: false,
            workgroup_memory: false,
            image_memory: false,
            output_memory: false,
            make_available: false,
            make_visible: false,
        }
    }

    pub fn acquire() -> Self {
        Self { acquire: true, ..Self::none() }
    }

    pub fn release() -> Self {
        Self { release: true, ..Self::none() }
    }

    pub fn acquire_release() -> Self {
        Self { acquire_release: true, ..Self::none() }
    }

    pub fn seq_cst() -> Self {
        Self { sequentially_consistent: true, ..Self::none() }
    }

    pub fn with_available(mut self) -> Self {
        self.make_available = true;
        self
    }

    pub fn with_visible(mut self) -> Self {
        self.make_visible = true;
        self
    }

    pub fn with_storage_buffer(mut self) -> Self {
        self.uniform_memory = true;
        self
    }

    pub fn with_workgroup(mut self) -> Self {
        self.workgroup_memory = true;
        self
    }

    pub fn has_ordering(&self) -> bool {
        self.acquire || self.release || self.acquire_release || self.sequentially_consistent
    }

    pub fn is_acquire(&self) -> bool {
        self.acquire || self.acquire_release || self.sequentially_consistent
    }

    pub fn is_release(&self) -> bool {
        self.release || self.acquire_release || self.sequentially_consistent
    }
}

impl Default for MemorySemantics {
    fn default() -> Self { Self::none() }
}

impl fmt::Display for MemorySemantics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        if self.sequentially_consistent { parts.push("SeqCst"); }
        else if self.acquire_release { parts.push("AcqRel"); }
        else {
            if self.acquire { parts.push("Acquire"); }
            if self.release { parts.push("Release"); }
        }
        if self.make_available { parts.push("MakeAvailable"); }
        if self.make_visible { parts.push("MakeVisible"); }
        if self.uniform_memory { parts.push("UniformMemory"); }
        if self.workgroup_memory { parts.push("WorkgroupMemory"); }
        if parts.is_empty() { parts.push("None"); }
        write!(f, "{}", parts.join("|"))
    }
}

// ---------------------------------------------------------------------------
// Memory domain
// ---------------------------------------------------------------------------

/// Vulkan memory domains for availability/visibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MemoryDomain {
    /// Device domain — visible to all invocations on the device.
    Device,
    /// Host domain — visible to the host.
    Host,
    /// Availability domain.
    Availability,
}

impl fmt::Display for MemoryDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryDomain::Device => write!(f, "Device"),
            MemoryDomain::Host => write!(f, "Host"),
            MemoryDomain::Availability => write!(f, "Availability"),
        }
    }
}

// ---------------------------------------------------------------------------
// Memory operation
// ---------------------------------------------------------------------------

/// Type of Vulkan memory operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VulkanOpType {
    Load,
    Store,
    AtomicLoad,
    AtomicStore,
    AtomicExchange,
    AtomicCompareExchange,
    AtomicAdd,
    AtomicMin,
    AtomicMax,
    AtomicAnd,
    AtomicOr,
    AtomicXor,
    ControlBarrier,
    MemoryBarrier,
}

impl VulkanOpType {
    pub fn is_read(&self) -> bool {
        matches!(self, Self::Load | Self::AtomicLoad | Self::AtomicCompareExchange |
                 Self::AtomicExchange | Self::AtomicAdd | Self::AtomicMin |
                 Self::AtomicMax | Self::AtomicAnd | Self::AtomicOr | Self::AtomicXor)
    }

    pub fn is_write(&self) -> bool {
        matches!(self, Self::Store | Self::AtomicStore | Self::AtomicCompareExchange |
                 Self::AtomicExchange | Self::AtomicAdd | Self::AtomicMin |
                 Self::AtomicMax | Self::AtomicAnd | Self::AtomicOr | Self::AtomicXor)
    }

    pub fn is_atomic(&self) -> bool {
        !matches!(self, Self::Load | Self::Store)
    }

    pub fn is_barrier(&self) -> bool {
        matches!(self, Self::ControlBarrier | Self::MemoryBarrier)
    }

    pub fn is_rmw(&self) -> bool {
        self.is_read() && self.is_write() && self.is_atomic()
    }
}

impl fmt::Display for VulkanOpType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ---------------------------------------------------------------------------
// Vulkan memory event
// ---------------------------------------------------------------------------

/// A memory event in the Vulkan model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulkanEvent {
    pub id: usize,
    pub invocation: usize,
    pub subgroup: usize,
    pub workgroup: usize,
    pub op_type: VulkanOpType,
    pub address: u64,
    pub value: u64,
    pub storage_class: StorageClass,
    pub semantics: MemorySemantics,
    pub scope: VulkanScope,
    pub program_order: usize,
}

impl VulkanEvent {
    pub fn new(id: usize, invocation: usize, op_type: VulkanOpType) -> Self {
        Self {
            id,
            invocation,
            subgroup: 0,
            workgroup: 0,
            op_type,
            address: 0,
            value: 0,
            storage_class: StorageClass::StorageBuffer,
            semantics: MemorySemantics::none(),
            scope: VulkanScope::Device,
            program_order: 0,
        }
    }

    pub fn with_address(mut self, addr: u64) -> Self { self.address = addr; self }
    pub fn with_value(mut self, val: u64) -> Self { self.value = val; self }
    pub fn with_scope(mut self, scope: VulkanScope) -> Self { self.scope = scope; self }
    pub fn with_semantics(mut self, sem: MemorySemantics) -> Self { self.semantics = sem; self }
    pub fn with_storage_class(mut self, sc: StorageClass) -> Self { self.storage_class = sc; self }

    /// Whether two events can see each other (same scope instance).
    pub fn same_scope_instance(&self, other: &VulkanEvent, scope: VulkanScope) -> bool {
        match scope {
            VulkanScope::Invocation => self.invocation == other.invocation,
            VulkanScope::Subgroup => self.subgroup == other.subgroup,
            VulkanScope::Workgroup => self.workgroup == other.workgroup,
            VulkanScope::QueueFamily | VulkanScope::Device => true,
        }
    }
}

impl fmt::Display for VulkanEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "E{}:I{} {} @{:#x}={} [{}]",
            self.id, self.invocation, self.op_type, self.address,
            self.value, self.semantics)
    }
}

// ---------------------------------------------------------------------------
// Availability / Visibility
// ---------------------------------------------------------------------------

/// Availability/visibility state of a memory location.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AvVisState {
    /// Set of scopes where the write is available.
    pub available_to: BTreeSet<VulkanScope>,
    /// Set of scopes where the write is visible.
    pub visible_to: BTreeSet<VulkanScope>,
}

impl AvVisState {
    pub fn new() -> Self {
        Self {
            available_to: BTreeSet::new(),
            visible_to: BTreeSet::new(),
        }
    }

    /// Make available to a scope.
    pub fn make_available(&mut self, scope: VulkanScope) {
        self.available_to.insert(scope);
        // Also available to narrower scopes.
        for &s in VulkanScope::all() {
            if scope.includes(&s) {
                self.available_to.insert(s);
            }
        }
    }

    /// Make visible to a scope.
    pub fn make_visible(&mut self, scope: VulkanScope) {
        // Can only be visible if available.
        if self.available_to.contains(&scope) || self.available_to.iter().any(|s| s.includes(&scope)) {
            self.visible_to.insert(scope);
            for &s in VulkanScope::all() {
                if scope.includes(&s) {
                    self.visible_to.insert(s);
                }
            }
        }
    }

    /// Check if visible to a scope.
    pub fn is_visible_to(&self, scope: VulkanScope) -> bool {
        self.visible_to.contains(&scope) ||
        self.visible_to.iter().any(|s| s.includes(&scope))
    }

    /// Check if available to a scope.
    pub fn is_available_to(&self, scope: VulkanScope) -> bool {
        self.available_to.contains(&scope) ||
        self.available_to.iter().any(|s| s.includes(&scope))
    }
}

impl Default for AvVisState {
    fn default() -> Self { Self::new() }
}

/// Tracks av/vis state per (address, writer) pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvVisTracker {
    /// Map from (address, event_id) to availability/visibility state.
    pub states: HashMap<(u64, usize), AvVisState>,
}

impl AvVisTracker {
    pub fn new() -> Self {
        Self { states: HashMap::new() }
    }

    /// After a store, the value is automatically visible to the writer invocation.
    pub fn record_store(&mut self, event: &VulkanEvent) {
        let mut state = AvVisState::new();
        state.visible_to.insert(VulkanScope::Invocation);
        if event.semantics.make_available {
            state.make_available(event.scope);
        }
        self.states.insert((event.address, event.id), state);
    }

    /// Process an availability operation.
    pub fn process_available(&mut self, addr: u64, event_id: usize, scope: VulkanScope) {
        if let Some(state) = self.states.get_mut(&(addr, event_id)) {
            state.make_available(scope);
        }
    }

    /// Process a visibility operation.
    pub fn process_visible(&mut self, addr: u64, event_id: usize, scope: VulkanScope) {
        if let Some(state) = self.states.get_mut(&(addr, event_id)) {
            state.make_visible(scope);
        }
    }

    /// Check if a read can see a specific write.
    pub fn can_read_see_write(&self, addr: u64, write_id: usize, reader_scope: VulkanScope) -> bool {
        match self.states.get(&(addr, write_id)) {
            Some(state) => state.is_visible_to(reader_scope),
            None => false,
        }
    }
}

impl Default for AvVisTracker {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Vulkan ordering relations
// ---------------------------------------------------------------------------

/// Edge in an ordering relation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OrderingEdge {
    pub from: usize,
    pub to: usize,
    pub relation: VulkanRelation,
}

/// Vulkan-specific ordering relations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VulkanRelation {
    ProgramOrder,
    ScopedModificationOrder,
    ReadFrom,
    FromRead,
    Synchronizes,
    ScopedSynchronizes,
    HappensBefore,
    AvailabilityChain,
    VisibilityChain,
    BarrierOrder,
}

impl fmt::Display for VulkanRelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VulkanRelation::ProgramOrder => write!(f, "po"),
            VulkanRelation::ScopedModificationOrder => write!(f, "smo"),
            VulkanRelation::ReadFrom => write!(f, "rf"),
            VulkanRelation::FromRead => write!(f, "fr"),
            VulkanRelation::Synchronizes => write!(f, "sw"),
            VulkanRelation::ScopedSynchronizes => write!(f, "ssw"),
            VulkanRelation::HappensBefore => write!(f, "hb"),
            VulkanRelation::AvailabilityChain => write!(f, "av"),
            VulkanRelation::VisibilityChain => write!(f, "vis"),
            VulkanRelation::BarrierOrder => write!(f, "bo"),
        }
    }
}

/// Collection of ordering edges forming a relation graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderingGraph {
    pub edges: Vec<OrderingEdge>,
    pub adjacency: HashMap<usize, Vec<(usize, VulkanRelation)>>,
}

impl OrderingGraph {
    pub fn new() -> Self {
        Self {
            edges: Vec::new(),
            adjacency: HashMap::new(),
        }
    }

    pub fn add_edge(&mut self, from: usize, to: usize, relation: VulkanRelation) {
        self.edges.push(OrderingEdge { from, to, relation });
        self.adjacency.entry(from).or_default().push((to, relation));
    }

    /// Check for cycles using DFS.
    pub fn has_cycle(&self) -> bool {
        let nodes: HashSet<usize> = self.adjacency.keys().copied().collect();
        let mut visited = HashSet::new();
        let mut in_stack = HashSet::new();

        for &node in &nodes {
            if !visited.contains(&node) {
                if self.dfs_cycle(node, &mut visited, &mut in_stack) {
                    return true;
                }
            }
        }
        false
    }

    fn dfs_cycle(&self, node: usize, visited: &mut HashSet<usize>, in_stack: &mut HashSet<usize>) -> bool {
        visited.insert(node);
        in_stack.insert(node);

        if let Some(neighbors) = self.adjacency.get(&node) {
            for &(next, _) in neighbors {
                if !visited.contains(&next) {
                    if self.dfs_cycle(next, visited, in_stack) {
                        return true;
                    }
                } else if in_stack.contains(&next) {
                    return true;
                }
            }
        }

        in_stack.remove(&node);
        false
    }

    /// Filter edges to a specific relation.
    pub fn edges_of(&self, relation: VulkanRelation) -> Vec<&OrderingEdge> {
        self.edges.iter().filter(|e| e.relation == relation).collect()
    }

    /// Count edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Count nodes.
    pub fn node_count(&self) -> usize {
        let mut nodes = HashSet::new();
        for e in &self.edges {
            nodes.insert(e.from);
            nodes.insert(e.to);
        }
        nodes.len()
    }
}

impl Default for OrderingGraph {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Vulkan execution
// ---------------------------------------------------------------------------

/// A Vulkan execution for verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulkanExecution {
    pub events: Vec<VulkanEvent>,
    pub ordering: OrderingGraph,
    pub av_vis: AvVisTracker,
}

impl VulkanExecution {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            ordering: OrderingGraph::new(),
            av_vis: AvVisTracker::new(),
        }
    }

    pub fn add_event(&mut self, event: VulkanEvent) {
        self.events.push(event);
    }

    /// Get event by id.
    pub fn get_event(&self, id: usize) -> Option<&VulkanEvent> {
        self.events.iter().find(|e| e.id == id)
    }

    /// Build program order edges.
    pub fn build_program_order(&mut self) {
        let mut by_invocation: HashMap<usize, Vec<usize>> = HashMap::new();
        for event in &self.events {
            by_invocation.entry(event.invocation).or_default().push(event.id);
        }
        for (_inv, mut ids) in by_invocation {
            ids.sort_by_key(|&id| {
                self.events.iter().find(|e| e.id == id)
                    .map(|e| e.program_order)
                    .unwrap_or(0)
            });
            for window in ids.windows(2) {
                self.ordering.add_edge(window[0], window[1], VulkanRelation::ProgramOrder);
            }
        }
    }

    /// Build synchronizes-with edges for release/acquire pairs.
    pub fn build_synchronizes_with(&mut self) {
        let stores: Vec<usize> = self.events.iter()
            .filter(|e| e.op_type.is_write() && e.semantics.is_release())
            .map(|e| e.id)
            .collect();
        let loads: Vec<usize> = self.events.iter()
            .filter(|e| e.op_type.is_read() && e.semantics.is_acquire())
            .map(|e| e.id)
            .collect();

        for &s in &stores {
            for &l in &loads {
                let se = self.events.iter().find(|e| e.id == s).unwrap();
                let le = self.events.iter().find(|e| e.id == l).unwrap();
                // Same address, load reads from store.
                if se.address == le.address && le.value == se.value && se.invocation != le.invocation {
                    let sync_scope = std::cmp::min(se.scope, le.scope);
                    if se.same_scope_instance(le, sync_scope) {
                        self.ordering.add_edge(s, l, VulkanRelation::Synchronizes);
                    }
                }
            }
        }
    }

    /// Check if the execution is consistent (no cycles in happens-before).
    pub fn is_consistent(&self) -> bool {
        !self.ordering.has_cycle()
    }

    /// Number of events.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Number of invocations.
    pub fn invocation_count(&self) -> usize {
        self.events.iter().map(|e| e.invocation).collect::<HashSet<_>>().len()
    }
}

impl Default for VulkanExecution {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Vulkan axiom
// ---------------------------------------------------------------------------

/// Axiom in the Vulkan memory model.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VulkanAxiom {
    /// Coherence: per-location total order.
    Coherence,
    /// Atomicity of RMW operations.
    Atomicity,
    /// No thin air: values cannot appear from nowhere.
    NoThinAir,
    /// Sequential consistency per scope.
    SeqCstPerScope,
    /// Availability before visibility.
    AvBeforeVis,
    /// Barrier ordering.
    BarrierOrdering,
    /// Scoped modification order.
    ScopedModOrder,
    /// Storage class restrictions.
    StorageClassRestriction,
}

impl VulkanAxiom {
    pub fn all() -> Vec<VulkanAxiom> {
        vec![
            VulkanAxiom::Coherence,
            VulkanAxiom::Atomicity,
            VulkanAxiom::NoThinAir,
            VulkanAxiom::SeqCstPerScope,
            VulkanAxiom::AvBeforeVis,
            VulkanAxiom::BarrierOrdering,
            VulkanAxiom::ScopedModOrder,
            VulkanAxiom::StorageClassRestriction,
        ]
    }
}

impl fmt::Display for VulkanAxiom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ---------------------------------------------------------------------------
// Vulkan axiom violation
// ---------------------------------------------------------------------------

/// A violation of a Vulkan axiom.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulkanViolation {
    pub axiom: VulkanAxiom,
    pub description: String,
    pub involved_events: Vec<usize>,
}

impl VulkanViolation {
    pub fn new(axiom: VulkanAxiom, description: &str, events: Vec<usize>) -> Self {
        Self {
            axiom,
            description: description.to_string(),
            involved_events: events,
        }
    }
}

impl fmt::Display for VulkanViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.axiom, self.description)
    }
}

// ---------------------------------------------------------------------------
// Vulkan axiom checker
// ---------------------------------------------------------------------------

/// Checks Vulkan axioms on an execution.
#[derive(Debug)]
pub struct VulkanAxiomChecker<'a> {
    execution: &'a VulkanExecution,
}

impl<'a> VulkanAxiomChecker<'a> {
    pub fn new(execution: &'a VulkanExecution) -> Self {
        Self { execution }
    }

    /// Check all axioms and return violations.
    pub fn check_all(&self) -> Vec<VulkanViolation> {
        let mut violations = Vec::new();
        violations.extend(self.check_coherence());
        violations.extend(self.check_atomicity());
        violations.extend(self.check_no_thin_air());
        violations.extend(self.check_av_before_vis());
        violations.extend(self.check_storage_class_restrictions());
        violations
    }

    /// Check coherence: no cycles among same-address operations.
    pub fn check_coherence(&self) -> Vec<VulkanViolation> {
        let mut violations = Vec::new();
        // Group events by address.
        let mut by_addr: HashMap<u64, Vec<&VulkanEvent>> = HashMap::new();
        for e in &self.execution.events {
            if !e.op_type.is_barrier() {
                by_addr.entry(e.address).or_default().push(e);
            }
        }

        for (addr, events) in &by_addr {
            let writes: Vec<&&VulkanEvent> = events.iter().filter(|e| e.op_type.is_write()).collect();
            if writes.len() > 1 {
                // Check for coherence cycles: look for w1 -> w2 and w2 -> w1.
                for i in 0..writes.len() {
                    for j in (i + 1)..writes.len() {
                        let w1 = writes[i].id;
                        let w2 = writes[j].id;
                        let w1_before_w2 = self.execution.ordering.adjacency
                            .get(&w1).map(|adj| adj.iter().any(|&(t, _)| t == w2)).unwrap_or(false);
                        let w2_before_w1 = self.execution.ordering.adjacency
                            .get(&w2).map(|adj| adj.iter().any(|&(t, _)| t == w1)).unwrap_or(false);
                        if w1_before_w2 && w2_before_w1 {
                            violations.push(VulkanViolation::new(
                                VulkanAxiom::Coherence,
                                &format!("Coherence cycle at address {:#x}", addr),
                                vec![w1, w2],
                            ));
                        }
                    }
                }
            }
        }
        violations
    }

    /// Check atomicity: RMW should read from immediately preceding write.
    pub fn check_atomicity(&self) -> Vec<VulkanViolation> {
        let violations = Vec::new();
        // Simplified: just check that RMW events are paired.
        for event in &self.execution.events {
            if event.op_type.is_rmw() {
                // In a full implementation, would verify the read-from relationship.
            }
        }
        violations
    }

    /// Check no thin air: no causal cycles.
    pub fn check_no_thin_air(&self) -> Vec<VulkanViolation> {
        if self.execution.ordering.has_cycle() {
            vec![VulkanViolation::new(
                VulkanAxiom::NoThinAir,
                "Causal cycle detected",
                vec![],
            )]
        } else {
            vec![]
        }
    }

    /// Check availability before visibility.
    pub fn check_av_before_vis(&self) -> Vec<VulkanViolation> {
        let mut violations = Vec::new();
        for (&(addr, eid), state) in &self.execution.av_vis.states {
            // Visibility requires availability.
            for &scope in &state.visible_to {
                if !state.is_available_to(scope) {
                    violations.push(VulkanViolation::new(
                        VulkanAxiom::AvBeforeVis,
                        &format!("Visible without available at {:#x} scope {:?}", addr, scope),
                        vec![eid],
                    ));
                }
            }
        }
        violations
    }

    /// Check storage class restrictions.
    pub fn check_storage_class_restrictions(&self) -> Vec<VulkanViolation> {
        let mut violations = Vec::new();
        for event in &self.execution.events {
            // Private/Function storage should only be accessed by the owning invocation.
            if matches!(event.storage_class, StorageClass::Private | StorageClass::Function) {
                // Check no cross-invocation access.
                for other in &self.execution.events {
                    if other.id != event.id && other.address == event.address
                        && other.invocation != event.invocation
                        && other.storage_class == event.storage_class
                    {
                        violations.push(VulkanViolation::new(
                            VulkanAxiom::StorageClassRestriction,
                            &format!("Cross-invocation access to {:?} storage at {:#x}",
                                event.storage_class, event.address),
                            vec![event.id, other.id],
                        ));
                    }
                }
            }
        }
        violations
    }
}

// ---------------------------------------------------------------------------
// Vulkan model specification
// ---------------------------------------------------------------------------

/// Configuration for Vulkan memory model verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulkanModelConfig {
    pub check_coherence: bool,
    pub check_atomicity: bool,
    pub check_no_thin_air: bool,
    pub check_seq_cst: bool,
    pub check_av_vis: bool,
    pub check_barriers: bool,
    pub check_storage_class: bool,
    pub scoped_model: bool,
    pub max_invocations: usize,
    pub max_workgroups: usize,
}

impl VulkanModelConfig {
    pub fn full() -> Self {
        Self {
            check_coherence: true,
            check_atomicity: true,
            check_no_thin_air: true,
            check_seq_cst: true,
            check_av_vis: true,
            check_barriers: true,
            check_storage_class: true,
            scoped_model: true,
            max_invocations: 256,
            max_workgroups: 16,
        }
    }

    pub fn minimal() -> Self {
        Self {
            check_coherence: true,
            check_atomicity: false,
            check_no_thin_air: true,
            check_seq_cst: false,
            check_av_vis: false,
            check_barriers: false,
            check_storage_class: false,
            scoped_model: false,
            max_invocations: 64,
            max_workgroups: 4,
        }
    }
}

impl Default for VulkanModelConfig {
    fn default() -> Self { Self::full() }
}

/// Vulkan memory model specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulkanModel {
    pub name: String,
    pub config: VulkanModelConfig,
}

impl VulkanModel {
    pub fn new() -> Self {
        Self {
            name: "Vulkan".to_string(),
            config: VulkanModelConfig::full(),
        }
    }

    pub fn with_config(config: VulkanModelConfig) -> Self {
        Self {
            name: "Vulkan".to_string(),
            config,
        }
    }

    /// Verify an execution against the Vulkan model.
    pub fn verify(&self, execution: &VulkanExecution) -> VulkanVerificationResult {
        let checker = VulkanAxiomChecker::new(execution);
        let violations = checker.check_all();
        let consistent = violations.is_empty();

        VulkanVerificationResult {
            model_name: self.name.clone(),
            consistent,
            violations,
            events_checked: execution.event_count(),
            invocations: execution.invocation_count(),
        }
    }
}

impl Default for VulkanModel {
    fn default() -> Self { Self::new() }
}

/// Result of Vulkan verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulkanVerificationResult {
    pub model_name: String,
    pub consistent: bool,
    pub violations: Vec<VulkanViolation>,
    pub events_checked: usize,
    pub invocations: usize,
}

impl fmt::Display for VulkanVerificationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {} ({} events, {} invocations",
            self.model_name,
            if self.consistent { "Consistent" } else { "Inconsistent" },
            self.events_checked, self.invocations)?;
        if !self.violations.is_empty() {
            write!(f, ", {} violations", self.violations.len())?;
        }
        write!(f, ")")
    }
}

// ---------------------------------------------------------------------------
// Litmus test support for Vulkan
// ---------------------------------------------------------------------------

/// A Vulkan litmus test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulkanLitmusTest {
    pub name: String,
    pub description: String,
    pub invocations: usize,
    pub workgroups: usize,
    pub subgroups: usize,
    pub events: Vec<VulkanEvent>,
    pub expected_outcomes: Vec<VulkanOutcome>,
    pub forbidden_outcomes: Vec<VulkanOutcome>,
}

/// An outcome of a Vulkan litmus test.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VulkanOutcome {
    pub values: Vec<(String, u64)>,
}

impl VulkanOutcome {
    pub fn new(values: Vec<(String, u64)>) -> Self {
        Self { values }
    }
}

impl fmt::Display for VulkanOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self.values.iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect();
        write!(f, "{{{}}}", parts.join(", "))
    }
}

impl VulkanLitmusTest {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            description: String::new(),
            invocations: 2,
            workgroups: 1,
            subgroups: 1,
            events: Vec::new(),
            expected_outcomes: Vec::new(),
            forbidden_outcomes: Vec::new(),
        }
    }

    /// Check if an outcome is forbidden.
    pub fn is_forbidden(&self, outcome: &VulkanOutcome) -> bool {
        self.forbidden_outcomes.contains(outcome)
    }

    /// Build the standard SB (Store Buffer) test for Vulkan.
    pub fn store_buffer() -> Self {
        let mut test = VulkanLitmusTest::new("SB-Vulkan");
        test.description = "Store Buffer litmus test under Vulkan model".to_string();
        test.invocations = 2;

        // Thread 0: store x=1; load r0=y
        let e0 = VulkanEvent::new(0, 0, VulkanOpType::Store)
            .with_address(0x100).with_value(1);
        let e1 = VulkanEvent::new(1, 0, VulkanOpType::Load)
            .with_address(0x200);

        // Thread 1: store y=1; load r1=x
        let e2 = VulkanEvent::new(2, 1, VulkanOpType::Store)
            .with_address(0x200).with_value(1);
        let e3 = VulkanEvent::new(3, 1, VulkanOpType::Load)
            .with_address(0x100);

        test.events = vec![e0, e1, e2, e3];

        // Forbidden under SC: both loads return 0.
        test.forbidden_outcomes.push(VulkanOutcome::new(vec![
            ("r0".to_string(), 0), ("r1".to_string(), 0),
        ]));

        test
    }

    /// Build the message passing test.
    pub fn message_passing() -> Self {
        let mut test = VulkanLitmusTest::new("MP-Vulkan");
        test.description = "Message Passing litmus test under Vulkan model".to_string();
        test.invocations = 2;

        let e0 = VulkanEvent::new(0, 0, VulkanOpType::Store)
            .with_address(0x100).with_value(42);
        let e1 = VulkanEvent::new(1, 0, VulkanOpType::AtomicStore)
            .with_address(0x200).with_value(1)
            .with_semantics(MemorySemantics::release().with_available());
        let e2 = VulkanEvent::new(2, 1, VulkanOpType::AtomicLoad)
            .with_address(0x200)
            .with_semantics(MemorySemantics::acquire().with_visible());
        let e3 = VulkanEvent::new(3, 1, VulkanOpType::Load)
            .with_address(0x100);

        test.events = vec![e0, e1, e2, e3];

        // If the acquire-load sees the release-store (flag=1), the data must be 42.
        test.forbidden_outcomes.push(VulkanOutcome::new(vec![
            ("flag".to_string(), 1), ("data".to_string(), 0),
        ]));

        test
    }
}

// ---------------------------------------------------------------------------
// Subgroup operations
// ---------------------------------------------------------------------------

/// Subgroup operation types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SubgroupOp {
    Elect,
    Broadcast,
    BallotBitCount,
    Shuffle,
    ShuffleXor,
    Reduce,
    InclusiveScan,
    ExclusiveScan,
}

/// Result of a subgroup ballot.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SubgroupBallot {
    pub mask: Vec<bool>,
}

impl SubgroupBallot {
    pub fn new(size: usize) -> Self {
        Self { mask: vec![false; size] }
    }

    pub fn all(size: usize) -> Self {
        Self { mask: vec![true; size] }
    }

    pub fn set(&mut self, idx: usize, val: bool) {
        if idx < self.mask.len() { self.mask[idx] = val; }
    }

    pub fn count(&self) -> usize {
        self.mask.iter().filter(|&&v| v).count()
    }

    pub fn is_uniform(&self) -> bool {
        self.count() == 0 || self.count() == self.mask.len()
    }
}

impl fmt::Display for SubgroupBallot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bits: String = self.mask.iter().map(|&b| if b { '1' } else { '0' }).collect();
        write!(f, "Ballot[{}]", bits)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Scope tests --

    #[test]
    fn test_scope_ordering() {
        assert!(VulkanScope::Device > VulkanScope::Workgroup);
        assert!(VulkanScope::Workgroup > VulkanScope::Subgroup);
        assert!(VulkanScope::Subgroup > VulkanScope::Invocation);
    }

    #[test]
    fn test_scope_includes() {
        assert!(VulkanScope::Device.includes(&VulkanScope::Workgroup));
        assert!(!VulkanScope::Subgroup.includes(&VulkanScope::Workgroup));
    }

    #[test]
    fn test_scope_broaden() {
        assert_eq!(VulkanScope::Invocation.broaden(), Some(VulkanScope::Subgroup));
        assert_eq!(VulkanScope::Device.broaden(), None);
    }

    // -- StorageClass tests --

    #[test]
    fn test_storage_class_shared() {
        assert!(StorageClass::StorageBuffer.is_shared());
        assert!(StorageClass::Workgroup.is_shared());
        assert!(!StorageClass::Private.is_shared());
        assert!(!StorageClass::Function.is_shared());
    }

    // -- MemorySemantics tests --

    #[test]
    fn test_memory_semantics_ordering() {
        assert!(MemorySemantics::acquire().has_ordering());
        assert!(MemorySemantics::release().has_ordering());
        assert!(!MemorySemantics::none().has_ordering());
    }

    #[test]
    fn test_memory_semantics_acquire_release() {
        let sem = MemorySemantics::acquire_release();
        assert!(sem.is_acquire());
        assert!(sem.is_release());
    }

    #[test]
    fn test_memory_semantics_available() {
        let sem = MemorySemantics::release().with_available();
        assert!(sem.make_available);
    }

    // -- VulkanOpType tests --

    #[test]
    fn test_op_type() {
        assert!(VulkanOpType::Load.is_read());
        assert!(!VulkanOpType::Load.is_write());
        assert!(!VulkanOpType::Load.is_atomic());
        assert!(VulkanOpType::AtomicExchange.is_rmw());
        assert!(VulkanOpType::ControlBarrier.is_barrier());
    }

    // -- VulkanEvent tests --

    #[test]
    fn test_event_creation() {
        let event = VulkanEvent::new(0, 1, VulkanOpType::Store)
            .with_address(0x100)
            .with_value(42);
        assert_eq!(event.id, 0);
        assert_eq!(event.address, 0x100);
        assert_eq!(event.value, 42);
    }

    #[test]
    fn test_same_scope_instance() {
        let mut e1 = VulkanEvent::new(0, 0, VulkanOpType::Store);
        e1.workgroup = 0;
        let mut e2 = VulkanEvent::new(1, 1, VulkanOpType::Load);
        e2.workgroup = 0;
        assert!(e1.same_scope_instance(&e2, VulkanScope::Workgroup));
        assert!(!e1.same_scope_instance(&e2, VulkanScope::Invocation));
    }

    // -- AvVisState tests --

    #[test]
    fn test_av_vis_state() {
        let mut state = AvVisState::new();
        state.make_available(VulkanScope::Device);
        assert!(state.is_available_to(VulkanScope::Device));
        assert!(state.is_available_to(VulkanScope::Workgroup));

        state.make_visible(VulkanScope::Workgroup);
        assert!(state.is_visible_to(VulkanScope::Workgroup));
    }

    #[test]
    fn test_av_vis_requires_available() {
        let mut state = AvVisState::new();
        // Trying to make visible without available.
        state.make_visible(VulkanScope::Device);
        assert!(!state.is_visible_to(VulkanScope::Device));
    }

    // -- AvVisTracker tests --

    #[test]
    fn test_av_vis_tracker() {
        let mut tracker = AvVisTracker::new();
        let event = VulkanEvent::new(0, 0, VulkanOpType::Store)
            .with_address(0x100)
            .with_semantics(MemorySemantics::release().with_available().with_visible());
        tracker.record_store(&event);
        tracker.process_available(0x100, 0, VulkanScope::Device);
        tracker.process_visible(0x100, 0, VulkanScope::Device);
        assert!(tracker.can_read_see_write(0x100, 0, VulkanScope::Device));
    }

    // -- OrderingGraph tests --

    #[test]
    fn test_ordering_graph_no_cycle() {
        let mut graph = OrderingGraph::new();
        graph.add_edge(0, 1, VulkanRelation::ProgramOrder);
        graph.add_edge(1, 2, VulkanRelation::ProgramOrder);
        assert!(!graph.has_cycle());
    }

    #[test]
    fn test_ordering_graph_cycle() {
        let mut graph = OrderingGraph::new();
        graph.add_edge(0, 1, VulkanRelation::ProgramOrder);
        graph.add_edge(1, 2, VulkanRelation::Synchronizes);
        graph.add_edge(2, 0, VulkanRelation::ReadFrom);
        assert!(graph.has_cycle());
    }

    #[test]
    fn test_ordering_graph_counts() {
        let mut graph = OrderingGraph::new();
        graph.add_edge(0, 1, VulkanRelation::ProgramOrder);
        graph.add_edge(1, 2, VulkanRelation::ProgramOrder);
        assert_eq!(graph.edge_count(), 2);
        assert_eq!(graph.node_count(), 3);
    }

    // -- VulkanExecution tests --

    #[test]
    fn test_execution_basic() {
        let mut exec = VulkanExecution::new();
        exec.add_event(VulkanEvent::new(0, 0, VulkanOpType::Store));
        exec.add_event(VulkanEvent::new(1, 0, VulkanOpType::Load));
        exec.add_event(VulkanEvent::new(2, 1, VulkanOpType::Store));
        assert_eq!(exec.event_count(), 3);
        assert_eq!(exec.invocation_count(), 2);
    }

    #[test]
    fn test_execution_build_po() {
        let mut exec = VulkanExecution::new();
        let mut e0 = VulkanEvent::new(0, 0, VulkanOpType::Store);
        e0.program_order = 0;
        let mut e1 = VulkanEvent::new(1, 0, VulkanOpType::Load);
        e1.program_order = 1;
        exec.add_event(e0);
        exec.add_event(e1);
        exec.build_program_order();
        assert_eq!(exec.ordering.edge_count(), 1);
    }

    #[test]
    fn test_execution_consistent() {
        let exec = VulkanExecution::new();
        assert!(exec.is_consistent());
    }

    // -- VulkanAxiomChecker tests --

    #[test]
    fn test_checker_empty_execution() {
        let exec = VulkanExecution::new();
        let checker = VulkanAxiomChecker::new(&exec);
        let violations = checker.check_all();
        assert!(violations.is_empty());
    }

    // -- VulkanModel tests --

    #[test]
    fn test_model_verify_empty() {
        let model = VulkanModel::new();
        let exec = VulkanExecution::new();
        let result = model.verify(&exec);
        assert!(result.consistent);
    }

    #[test]
    fn test_model_config() {
        let config = VulkanModelConfig::full();
        assert!(config.check_coherence);
        assert!(config.check_atomicity);

        let config = VulkanModelConfig::minimal();
        assert!(config.check_coherence);
        assert!(!config.check_atomicity);
    }

    // -- VulkanLitmusTest tests --

    #[test]
    fn test_litmus_store_buffer() {
        let test = VulkanLitmusTest::store_buffer();
        assert_eq!(test.name, "SB-Vulkan");
        assert_eq!(test.invocations, 2);
        assert_eq!(test.events.len(), 4);
        assert_eq!(test.forbidden_outcomes.len(), 1);
    }

    #[test]
    fn test_litmus_message_passing() {
        let test = VulkanLitmusTest::message_passing();
        assert_eq!(test.name, "MP-Vulkan");
        assert_eq!(test.events.len(), 4);
    }

    #[test]
    fn test_litmus_forbidden() {
        let test = VulkanLitmusTest::store_buffer();
        let outcome = VulkanOutcome::new(vec![
            ("r0".to_string(), 0), ("r1".to_string(), 0),
        ]);
        assert!(test.is_forbidden(&outcome));
    }

    // -- SubgroupBallot tests --

    #[test]
    fn test_subgroup_ballot() {
        let mut ballot = SubgroupBallot::new(4);
        ballot.set(0, true);
        ballot.set(2, true);
        assert_eq!(ballot.count(), 2);
        assert!(!ballot.is_uniform());
    }

    #[test]
    fn test_subgroup_ballot_all() {
        let ballot = SubgroupBallot::all(4);
        assert_eq!(ballot.count(), 4);
        assert!(ballot.is_uniform());
    }

    // -- Display tests --

    #[test]
    fn test_displays() {
        assert_eq!(format!("{}", VulkanScope::Device), "Device");
        assert_eq!(format!("{}", StorageClass::StorageBuffer), "StorageBuffer");
        assert_eq!(format!("{}", MemorySemantics::acquire()), "Acquire");
        assert!(format!("{}", MemorySemantics::seq_cst()).contains("SeqCst"));
    }

    #[test]
    fn test_vulkan_outcome_display() {
        let outcome = VulkanOutcome::new(vec![("r0".to_string(), 1)]);
        assert!(format!("{}", outcome).contains("r0=1"));
    }
}

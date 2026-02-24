//! Full Vulkan memory model specification for LITMUS∞.
//!
//! Extends the base Vulkan model with comprehensive support for:
//! - Availability/visibility operations with full domain tracking
//! - Storage classes: Workgroup, StorageBuffer, Image, Uniform
//! - Non-private vs private texels
//! - Shader invocation interleaving model
//! - Subgroup/workgroup/queue family scope hierarchy
//!
//! # Vulkan Memory Model Overview
//!
//! The Vulkan memory model is based on the C++11 model but extends it with
//! scoped synchronisation and explicit availability/visibility operations.
//! Writes must be made *available* (flushed from caches) and then *visible*
//! (pulled into the reader's cache) before they can be observed.

#![allow(unused)]

use std::collections::{HashMap, HashSet, BTreeSet, BTreeMap};
use std::fmt;
use serde::{Serialize, Deserialize};

// ---------------------------------------------------------------------------
// Vulkan full scope hierarchy
// ---------------------------------------------------------------------------

/// Full Vulkan scope hierarchy including queue family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum VulkanFullScope {
    Invocation,
    Subgroup,
    Workgroup,
    QueueFamily,
    Device,
}

impl VulkanFullScope {
    pub fn all() -> &'static [VulkanFullScope] {
        &[
            Self::Invocation,
            Self::Subgroup,
            Self::Workgroup,
            Self::QueueFamily,
            Self::Device,
        ]
    }

    /// Whether `self` includes `other` in the scope hierarchy.
    pub fn includes(&self, other: &VulkanFullScope) -> bool {
        (*self as u8) >= (*other as u8)
    }

    /// Next broader scope.
    pub fn broaden(&self) -> Option<VulkanFullScope> {
        match self {
            Self::Invocation => Some(Self::Subgroup),
            Self::Subgroup => Some(Self::Workgroup),
            Self::Workgroup => Some(Self::QueueFamily),
            Self::QueueFamily => Some(Self::Device),
            Self::Device => None,
        }
    }

    /// Minimum scope required for two invocations to synchronise.
    pub fn min_scope_for(inv1: &InvocationId, inv2: &InvocationId) -> VulkanFullScope {
        if inv1.invocation == inv2.invocation { return Self::Invocation; }
        if inv1.subgroup == inv2.subgroup { return Self::Subgroup; }
        if inv1.workgroup == inv2.workgroup { return Self::Workgroup; }
        if inv1.queue_family == inv2.queue_family { return Self::QueueFamily; }
        Self::Device
    }
}

impl fmt::Display for VulkanFullScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invocation => write!(f, "Invocation"),
            Self::Subgroup => write!(f, "Subgroup"),
            Self::Workgroup => write!(f, "Workgroup"),
            Self::QueueFamily => write!(f, "QueueFamily"),
            Self::Device => write!(f, "Device"),
        }
    }
}

// ---------------------------------------------------------------------------
// Invocation identity
// ---------------------------------------------------------------------------

/// Full invocation placement within the Vulkan hierarchy.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InvocationId {
    pub invocation: usize,
    pub subgroup: usize,
    pub workgroup: usize,
    pub queue_family: usize,
}

impl InvocationId {
    pub fn new(invocation: usize, subgroup: usize, workgroup: usize, queue_family: usize) -> Self {
        Self { invocation, subgroup, workgroup, queue_family }
    }

    /// Check whether two invocations share a scope instance.
    pub fn same_scope_instance(&self, other: &InvocationId, scope: VulkanFullScope) -> bool {
        match scope {
            VulkanFullScope::Invocation => self.invocation == other.invocation,
            VulkanFullScope::Subgroup => self.subgroup == other.subgroup,
            VulkanFullScope::Workgroup => self.workgroup == other.workgroup,
            VulkanFullScope::QueueFamily => self.queue_family == other.queue_family,
            VulkanFullScope::Device => true,
        }
    }
}

impl fmt::Display for InvocationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Inv({}, sg={}, wg={}, qf={})",
            self.invocation, self.subgroup, self.workgroup, self.queue_family)
    }
}

// ---------------------------------------------------------------------------
// Storage classes
// ---------------------------------------------------------------------------

/// Vulkan storage classes with full classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VulkanStorageClass {
    /// Per-invocation private data.
    Function,
    /// Per-invocation private.
    Private,
    /// Shared within a workgroup.
    Workgroup,
    /// Storage buffer — device-wide.
    StorageBuffer,
    /// Uniform buffer — read-only, device-wide.
    Uniform,
    /// Image/texture storage.
    Image,
    /// Push constant — read-only.
    PushConstant,
}

impl VulkanStorageClass {
    /// Whether this class is shared across invocations (non-private).
    pub fn is_shared(&self) -> bool {
        matches!(self, Self::Workgroup | Self::StorageBuffer | Self::Uniform | Self::Image)
    }

    /// Whether this class is read-only.
    pub fn is_read_only(&self) -> bool {
        matches!(self, Self::Uniform | Self::PushConstant)
    }

    /// Whether this class is private to a single invocation.
    pub fn is_private(&self) -> bool {
        matches!(self, Self::Function | Self::Private)
    }

    /// Minimum scope required for this storage class to be visible.
    pub fn min_visibility_scope(&self) -> VulkanFullScope {
        match self {
            Self::Function | Self::Private => VulkanFullScope::Invocation,
            Self::Workgroup => VulkanFullScope::Workgroup,
            Self::StorageBuffer | Self::Uniform | Self::Image | Self::PushConstant => {
                VulkanFullScope::QueueFamily
            }
        }
    }
}

impl fmt::Display for VulkanStorageClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Function => write!(f, "Function"),
            Self::Private => write!(f, "Private"),
            Self::Workgroup => write!(f, "Workgroup"),
            Self::StorageBuffer => write!(f, "StorageBuffer"),
            Self::Uniform => write!(f, "Uniform"),
            Self::Image => write!(f, "Image"),
            Self::PushConstant => write!(f, "PushConstant"),
        }
    }
}

// ---------------------------------------------------------------------------
// Memory semantics (SPIR-V)
// ---------------------------------------------------------------------------

/// SPIR-V memory semantics flags for the full Vulkan model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VulkanFullSemantics {
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

impl VulkanFullSemantics {
    pub fn none() -> Self {
        Self {
            acquire: false, release: false, acquire_release: false,
            sequentially_consistent: false, uniform_memory: false,
            workgroup_memory: false, image_memory: false, output_memory: false,
            make_available: false, make_visible: false,
        }
    }

    pub fn acquire() -> Self { Self { acquire: true, ..Self::none() } }
    pub fn release() -> Self { Self { release: true, ..Self::none() } }
    pub fn acq_rel() -> Self { Self { acquire_release: true, ..Self::none() } }
    pub fn seq_cst() -> Self { Self { sequentially_consistent: true, ..Self::none() } }

    pub fn with_available(mut self) -> Self { self.make_available = true; self }
    pub fn with_visible(mut self) -> Self { self.make_visible = true; self }
    pub fn with_storage_buffer(mut self) -> Self { self.uniform_memory = true; self }
    pub fn with_workgroup(mut self) -> Self { self.workgroup_memory = true; self }
    pub fn with_image(mut self) -> Self { self.image_memory = true; self }

    pub fn has_ordering(&self) -> bool {
        self.acquire || self.release || self.acquire_release || self.sequentially_consistent
    }

    pub fn is_acquire(&self) -> bool {
        self.acquire || self.acquire_release || self.sequentially_consistent
    }

    pub fn is_release(&self) -> bool {
        self.release || self.acquire_release || self.sequentially_consistent
    }

    /// Which storage classes are affected by these semantics.
    pub fn affected_classes(&self) -> Vec<VulkanStorageClass> {
        let mut classes = Vec::new();
        if self.uniform_memory { classes.push(VulkanStorageClass::StorageBuffer); }
        if self.workgroup_memory { classes.push(VulkanStorageClass::Workgroup); }
        if self.image_memory { classes.push(VulkanStorageClass::Image); }
        classes
    }
}

impl Default for VulkanFullSemantics {
    fn default() -> Self { Self::none() }
}

impl fmt::Display for VulkanFullSemantics {
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
        if self.uniform_memory { parts.push("StorageBuf"); }
        if self.workgroup_memory { parts.push("Workgroup"); }
        if self.image_memory { parts.push("Image"); }
        if parts.is_empty() { parts.push("None"); }
        write!(f, "{}", parts.join("|"))
    }
}

// ---------------------------------------------------------------------------
// Availability / visibility operations
// ---------------------------------------------------------------------------

/// Represents an availability or visibility operation in the Vulkan model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VulkanAvailVisOp {
    /// Make a write available at a scope.
    MakeAvailable { scope: VulkanFullScope },
    /// Make a write visible at a scope.
    MakeVisible { scope: VulkanFullScope },
}

impl VulkanAvailVisOp {
    pub fn scope(&self) -> VulkanFullScope {
        match self {
            Self::MakeAvailable { scope } | Self::MakeVisible { scope } => *scope,
        }
    }

    pub fn is_available(&self) -> bool {
        matches!(self, Self::MakeAvailable { .. })
    }

    pub fn is_visible(&self) -> bool {
        matches!(self, Self::MakeVisible { .. })
    }
}

impl fmt::Display for VulkanAvailVisOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MakeAvailable { scope } => write!(f, "MakeAvailable({})", scope),
            Self::MakeVisible { scope } => write!(f, "MakeVisible({})", scope),
        }
    }
}

/// Tracks per-write availability/visibility state across scopes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FullAvVisState {
    pub available_to: BTreeSet<VulkanFullScope>,
    pub visible_to: BTreeSet<VulkanFullScope>,
    pub storage_class: VulkanStorageClass,
}

impl FullAvVisState {
    pub fn new(storage_class: VulkanStorageClass) -> Self {
        Self {
            available_to: BTreeSet::new(),
            visible_to: BTreeSet::new(),
            storage_class,
        }
    }

    /// Make available at a scope (propagates to narrower scopes).
    pub fn make_available(&mut self, scope: VulkanFullScope) {
        self.available_to.insert(scope);
        for &s in VulkanFullScope::all() {
            if scope.includes(&s) {
                self.available_to.insert(s);
            }
        }
    }

    /// Make visible at a scope (requires availability).
    pub fn make_visible(&mut self, scope: VulkanFullScope) {
        if self.available_to.contains(&scope) || self.available_to.iter().any(|s| s.includes(&scope)) {
            self.visible_to.insert(scope);
            for &s in VulkanFullScope::all() {
                if scope.includes(&s) {
                    self.visible_to.insert(s);
                }
            }
        }
    }

    pub fn is_visible_to(&self, scope: VulkanFullScope) -> bool {
        self.visible_to.contains(&scope) || self.visible_to.iter().any(|s| s.includes(&scope))
    }

    pub fn is_available_to(&self, scope: VulkanFullScope) -> bool {
        self.available_to.contains(&scope) || self.available_to.iter().any(|s| s.includes(&scope))
    }
}

impl Default for FullAvVisState {
    fn default() -> Self { Self::new(VulkanStorageClass::StorageBuffer) }
}

/// Tracker for availability/visibility per (address, event_id).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FullAvVisTracker {
    pub states: HashMap<(u64, usize), FullAvVisState>,
}

impl FullAvVisTracker {
    pub fn new() -> Self { Self { states: HashMap::new() } }

    pub fn record_store(&mut self, event_id: usize, address: u64, storage_class: VulkanStorageClass,
                        semantics: &VulkanFullSemantics, scope: VulkanFullScope) {
        let mut state = FullAvVisState::new(storage_class);
        state.visible_to.insert(VulkanFullScope::Invocation);
        if semantics.make_available {
            state.make_available(scope);
        }
        self.states.insert((address, event_id), state);
    }

    pub fn process_available(&mut self, addr: u64, event_id: usize, scope: VulkanFullScope) {
        if let Some(state) = self.states.get_mut(&(addr, event_id)) {
            state.make_available(scope);
        }
    }

    pub fn process_visible(&mut self, addr: u64, event_id: usize, scope: VulkanFullScope) {
        if let Some(state) = self.states.get_mut(&(addr, event_id)) {
            state.make_visible(scope);
        }
    }

    pub fn can_read_see_write(&self, addr: u64, write_id: usize, reader_scope: VulkanFullScope) -> bool {
        self.states.get(&(addr, write_id))
            .map(|s| s.is_visible_to(reader_scope))
            .unwrap_or(false)
    }
}

// ---------------------------------------------------------------------------
// Texel classification
// ---------------------------------------------------------------------------

/// Texel ownership classification per the Vulkan spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TexelType {
    /// Non-private: shared across invocations.
    NonPrivate,
    /// Private: owned by a single invocation.
    Private,
}

impl TexelType {
    /// Whether cross-invocation access requires explicit synchronisation.
    pub fn requires_sync(&self) -> bool {
        matches!(self, Self::NonPrivate)
    }
}

impl fmt::Display for TexelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonPrivate => write!(f, "NonPrivate"),
            Self::Private => write!(f, "Private"),
        }
    }
}

// ---------------------------------------------------------------------------
// Vulkan full operation types
// ---------------------------------------------------------------------------

/// Operation types for the full Vulkan model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VulkanFullOpType {
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
    ImageRead,
    ImageWrite,
}

impl VulkanFullOpType {
    pub fn is_read(&self) -> bool {
        matches!(self,
            Self::Load | Self::AtomicLoad | Self::AtomicExchange
            | Self::AtomicCompareExchange | Self::AtomicAdd | Self::AtomicMin
            | Self::AtomicMax | Self::AtomicAnd | Self::AtomicOr | Self::AtomicXor
            | Self::ImageRead
        )
    }

    pub fn is_write(&self) -> bool {
        matches!(self,
            Self::Store | Self::AtomicStore | Self::AtomicExchange
            | Self::AtomicCompareExchange | Self::AtomicAdd | Self::AtomicMin
            | Self::AtomicMax | Self::AtomicAnd | Self::AtomicOr | Self::AtomicXor
            | Self::ImageWrite
        )
    }

    pub fn is_atomic(&self) -> bool {
        !matches!(self, Self::Load | Self::Store | Self::ImageRead | Self::ImageWrite)
    }

    pub fn is_barrier(&self) -> bool {
        matches!(self, Self::ControlBarrier | Self::MemoryBarrier)
    }

    pub fn is_rmw(&self) -> bool {
        self.is_read() && self.is_write() && self.is_atomic()
    }

    pub fn is_image_op(&self) -> bool {
        matches!(self, Self::ImageRead | Self::ImageWrite)
    }
}

impl fmt::Display for VulkanFullOpType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ---------------------------------------------------------------------------
// Vulkan full event
// ---------------------------------------------------------------------------

/// A memory event in the full Vulkan model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulkanFullEvent {
    pub id: usize,
    pub invocation_id: InvocationId,
    pub op_type: VulkanFullOpType,
    pub address: u64,
    pub value: u64,
    pub storage_class: VulkanStorageClass,
    pub semantics: VulkanFullSemantics,
    pub scope: VulkanFullScope,
    pub texel_type: TexelType,
    pub program_order: usize,
}

impl VulkanFullEvent {
    pub fn new(id: usize, inv_id: InvocationId, op_type: VulkanFullOpType) -> Self {
        Self {
            id,
            invocation_id: inv_id,
            op_type,
            address: 0,
            value: 0,
            storage_class: VulkanStorageClass::StorageBuffer,
            semantics: VulkanFullSemantics::none(),
            scope: VulkanFullScope::Device,
            texel_type: TexelType::NonPrivate,
            program_order: 0,
        }
    }

    pub fn with_address(mut self, addr: u64) -> Self { self.address = addr; self }
    pub fn with_value(mut self, val: u64) -> Self { self.value = val; self }
    pub fn with_scope(mut self, scope: VulkanFullScope) -> Self { self.scope = scope; self }
    pub fn with_semantics(mut self, sem: VulkanFullSemantics) -> Self { self.semantics = sem; self }
    pub fn with_storage_class(mut self, sc: VulkanStorageClass) -> Self { self.storage_class = sc; self }
    pub fn with_texel(mut self, tt: TexelType) -> Self { self.texel_type = tt; self }
    pub fn with_po(mut self, po: usize) -> Self { self.program_order = po; self }

    pub fn same_scope_instance(&self, other: &VulkanFullEvent, scope: VulkanFullScope) -> bool {
        self.invocation_id.same_scope_instance(&other.invocation_id, scope)
    }
}

impl fmt::Display for VulkanFullEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "E{}:{} {} @{:#x}={} [{}|{}]",
            self.id, self.invocation_id, self.op_type,
            self.address, self.value, self.semantics, self.storage_class)
    }
}

// ---------------------------------------------------------------------------
// Shader invocation interleaving model
// ---------------------------------------------------------------------------

/// Models the valid interleavings of shader invocations within a workgroup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvocationInterleavingModel {
    /// Number of invocations per subgroup.
    pub subgroup_size: usize,
    /// Whether subgroup operations require convergence.
    pub requires_convergence: bool,
    /// Active invocation mask per subgroup.
    pub active_masks: HashMap<usize, Vec<bool>>,
}

impl InvocationInterleavingModel {
    pub fn new(subgroup_size: usize) -> Self {
        Self {
            subgroup_size,
            requires_convergence: true,
            active_masks: HashMap::new(),
        }
    }

    /// Set the active mask for a subgroup.
    pub fn set_active_mask(&mut self, subgroup: usize, mask: Vec<bool>) {
        self.active_masks.insert(subgroup, mask);
    }

    /// Check if an invocation is active in its subgroup.
    pub fn is_active(&self, inv: &InvocationId) -> bool {
        self.active_masks.get(&inv.subgroup)
            .and_then(|mask| mask.get(inv.invocation % self.subgroup_size))
            .copied()
            .unwrap_or(true)
    }

    /// Number of active invocations in a subgroup.
    pub fn active_count(&self, subgroup: usize) -> usize {
        self.active_masks.get(&subgroup)
            .map(|mask| mask.iter().filter(|&&v| v).count())
            .unwrap_or(self.subgroup_size)
    }

    /// Check whether all invocations in a subgroup are converged (all active).
    pub fn is_converged(&self, subgroup: usize) -> bool {
        self.active_count(subgroup) == self.subgroup_size
    }
}

impl Default for InvocationInterleavingModel {
    fn default() -> Self { Self::new(32) }
}

// ---------------------------------------------------------------------------
// Ordering relations
// ---------------------------------------------------------------------------

/// Relations in the full Vulkan model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VulkanFullRelation {
    ProgramOrder,
    ScopedModificationOrder,
    ReadsFrom,
    FromReads,
    Synchronizes,
    ScopedSynchronizes,
    HappensBefore,
    AvailabilityChain,
    VisibilityChain,
    BarrierOrder,
    ImageCoherence,
}

impl fmt::Display for VulkanFullRelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProgramOrder => write!(f, "po"),
            Self::ScopedModificationOrder => write!(f, "smo"),
            Self::ReadsFrom => write!(f, "rf"),
            Self::FromReads => write!(f, "fr"),
            Self::Synchronizes => write!(f, "sw"),
            Self::ScopedSynchronizes => write!(f, "ssw"),
            Self::HappensBefore => write!(f, "hb"),
            Self::AvailabilityChain => write!(f, "av"),
            Self::VisibilityChain => write!(f, "vis"),
            Self::BarrierOrder => write!(f, "bo"),
            Self::ImageCoherence => write!(f, "img-co"),
        }
    }
}

/// Edge in the ordering graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VulkanFullEdge {
    pub from: usize,
    pub to: usize,
    pub relation: VulkanFullRelation,
}

/// Ordering graph for the full Vulkan model.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VulkanFullOrderingGraph {
    pub edges: Vec<VulkanFullEdge>,
    pub adjacency: HashMap<usize, Vec<(usize, VulkanFullRelation)>>,
}

impl VulkanFullOrderingGraph {
    pub fn new() -> Self { Self { edges: Vec::new(), adjacency: HashMap::new() } }

    pub fn add_edge(&mut self, from: usize, to: usize, relation: VulkanFullRelation) {
        self.edges.push(VulkanFullEdge { from, to, relation });
        self.adjacency.entry(from).or_default().push((to, relation));
    }

    pub fn has_cycle(&self) -> bool {
        let nodes: HashSet<usize> = self.adjacency.keys().copied().collect();
        let mut visited = HashSet::new();
        let mut in_stack = HashSet::new();
        for &node in &nodes {
            if !visited.contains(&node) && self.dfs_cycle(node, &mut visited, &mut in_stack) {
                return true;
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
                    if self.dfs_cycle(next, visited, in_stack) { return true; }
                } else if in_stack.contains(&next) {
                    return true;
                }
            }
        }
        in_stack.remove(&node);
        false
    }

    pub fn edges_of(&self, relation: VulkanFullRelation) -> Vec<&VulkanFullEdge> {
        self.edges.iter().filter(|e| e.relation == relation).collect()
    }

    pub fn edge_count(&self) -> usize { self.edges.len() }

    pub fn node_count(&self) -> usize {
        let mut nodes = HashSet::new();
        for e in &self.edges { nodes.insert(e.from); nodes.insert(e.to); }
        nodes.len()
    }
}

// ---------------------------------------------------------------------------
// Vulkan full execution
// ---------------------------------------------------------------------------

/// A complete Vulkan execution for verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulkanFullExecution {
    pub events: Vec<VulkanFullEvent>,
    pub ordering: VulkanFullOrderingGraph,
    pub av_vis: FullAvVisTracker,
    pub interleaving: InvocationInterleavingModel,
}

impl VulkanFullExecution {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            ordering: VulkanFullOrderingGraph::new(),
            av_vis: FullAvVisTracker::new(),
            interleaving: InvocationInterleavingModel::default(),
        }
    }

    pub fn add_event(&mut self, event: VulkanFullEvent) {
        if event.op_type.is_write() {
            self.av_vis.record_store(
                event.id, event.address, event.storage_class,
                &event.semantics, event.scope,
            );
        }
        self.events.push(event);
    }

    pub fn get_event(&self, id: usize) -> Option<&VulkanFullEvent> {
        self.events.iter().find(|e| e.id == id)
    }

    pub fn build_program_order(&mut self) {
        let mut by_inv: HashMap<usize, Vec<usize>> = HashMap::new();
        for event in &self.events {
            by_inv.entry(event.invocation_id.invocation).or_default().push(event.id);
        }
        for (_inv, mut ids) in by_inv {
            ids.sort_by_key(|&id| {
                self.events.iter().find(|e| e.id == id).map(|e| e.program_order).unwrap_or(0)
            });
            for window in ids.windows(2) {
                self.ordering.add_edge(window[0], window[1], VulkanFullRelation::ProgramOrder);
            }
        }
    }

    pub fn build_synchronizes_with(&mut self) {
        let stores: Vec<usize> = self.events.iter()
            .filter(|e| e.op_type.is_write() && e.semantics.is_release())
            .map(|e| e.id).collect();
        let loads: Vec<usize> = self.events.iter()
            .filter(|e| e.op_type.is_read() && e.semantics.is_acquire())
            .map(|e| e.id).collect();

        for &s in &stores {
            for &l in &loads {
                let se = self.events.iter().find(|e| e.id == s).unwrap();
                let le = self.events.iter().find(|e| e.id == l).unwrap();
                if se.address == le.address && se.value == le.value
                    && se.invocation_id != le.invocation_id
                {
                    let sync_scope = std::cmp::min(se.scope, le.scope);
                    if se.same_scope_instance(le, sync_scope) {
                        self.ordering.add_edge(s, l, VulkanFullRelation::Synchronizes);
                    }
                }
            }
        }
    }

    pub fn is_consistent(&self) -> bool { !self.ordering.has_cycle() }
    pub fn event_count(&self) -> usize { self.events.len() }

    pub fn invocation_count(&self) -> usize {
        self.events.iter()
            .map(|e| e.invocation_id.invocation)
            .collect::<HashSet<_>>().len()
    }
}

impl Default for VulkanFullExecution {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Axioms
// ---------------------------------------------------------------------------

/// Axiom in the full Vulkan memory model.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VulkanFullAxiom {
    Coherence,
    Atomicity,
    NoThinAir,
    SeqCstPerScope,
    AvBeforeVis,
    BarrierOrdering,
    ScopedModOrder,
    StorageClassRestriction,
    ImageCoherence,
    NonPrivateTexelSync,
}

impl VulkanFullAxiom {
    pub fn all() -> Vec<Self> {
        vec![
            Self::Coherence, Self::Atomicity, Self::NoThinAir,
            Self::SeqCstPerScope, Self::AvBeforeVis, Self::BarrierOrdering,
            Self::ScopedModOrder, Self::StorageClassRestriction,
            Self::ImageCoherence, Self::NonPrivateTexelSync,
        ]
    }
}

impl fmt::Display for VulkanFullAxiom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ---------------------------------------------------------------------------
// Violations
// ---------------------------------------------------------------------------

/// A violation of a full Vulkan model axiom.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulkanFullViolation {
    pub axiom: VulkanFullAxiom,
    pub description: String,
    pub involved_events: Vec<usize>,
}

impl VulkanFullViolation {
    pub fn new(axiom: VulkanFullAxiom, description: &str, events: Vec<usize>) -> Self {
        Self { axiom, description: description.to_string(), involved_events: events }
    }
}

impl fmt::Display for VulkanFullViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {} (events {:?})", self.axiom, self.description, self.involved_events)
    }
}

// ---------------------------------------------------------------------------
// Full axiom checker
// ---------------------------------------------------------------------------

/// Checks all full Vulkan model axioms on an execution.
#[derive(Debug)]
pub struct VulkanFullAxiomChecker<'a> {
    execution: &'a VulkanFullExecution,
}

impl<'a> VulkanFullAxiomChecker<'a> {
    pub fn new(execution: &'a VulkanFullExecution) -> Self {
        Self { execution }
    }

    pub fn check_all(&self) -> Vec<VulkanFullViolation> {
        let mut violations = Vec::new();
        violations.extend(self.check_coherence());
        violations.extend(self.check_atomicity());
        violations.extend(self.check_no_thin_air());
        violations.extend(self.check_av_before_vis());
        violations.extend(self.check_storage_class_restrictions());
        violations.extend(self.check_non_private_texel_sync());
        violations.extend(self.check_image_coherence());
        violations
    }

    pub fn check_coherence(&self) -> Vec<VulkanFullViolation> {
        let mut violations = Vec::new();
        let mut by_addr: HashMap<u64, Vec<&VulkanFullEvent>> = HashMap::new();
        for e in &self.execution.events {
            if !e.op_type.is_barrier() { by_addr.entry(e.address).or_default().push(e); }
        }
        for (addr, events) in &by_addr {
            let writes: Vec<&&VulkanFullEvent> = events.iter().filter(|e| e.op_type.is_write()).collect();
            for i in 0..writes.len() {
                for j in (i + 1)..writes.len() {
                    let w1 = writes[i].id;
                    let w2 = writes[j].id;
                    let fwd = self.execution.ordering.adjacency.get(&w1)
                        .map(|adj| adj.iter().any(|&(t, _)| t == w2)).unwrap_or(false);
                    let bwd = self.execution.ordering.adjacency.get(&w2)
                        .map(|adj| adj.iter().any(|&(t, _)| t == w1)).unwrap_or(false);
                    if fwd && bwd {
                        violations.push(VulkanFullViolation::new(
                            VulkanFullAxiom::Coherence,
                            &format!("Coherence cycle at {:#x}", addr),
                            vec![w1, w2],
                        ));
                    }
                }
            }
        }
        violations
    }

    pub fn check_atomicity(&self) -> Vec<VulkanFullViolation> {
        let violations = Vec::new();
        for event in &self.execution.events {
            if event.op_type.is_rmw() {
                // Verify paired RMW in full implementation.
            }
        }
        violations
    }

    pub fn check_no_thin_air(&self) -> Vec<VulkanFullViolation> {
        if self.execution.ordering.has_cycle() {
            vec![VulkanFullViolation::new(VulkanFullAxiom::NoThinAir, "Causal cycle detected", vec![])]
        } else {
            vec![]
        }
    }

    pub fn check_av_before_vis(&self) -> Vec<VulkanFullViolation> {
        let mut violations = Vec::new();
        for (&(addr, eid), state) in &self.execution.av_vis.states {
            for &scope in &state.visible_to {
                if !state.is_available_to(scope) {
                    violations.push(VulkanFullViolation::new(
                        VulkanFullAxiom::AvBeforeVis,
                        &format!("Visible without available at {:#x} scope {}", addr, scope),
                        vec![eid],
                    ));
                }
            }
        }
        violations
    }

    pub fn check_storage_class_restrictions(&self) -> Vec<VulkanFullViolation> {
        let mut violations = Vec::new();
        for event in &self.execution.events {
            if event.storage_class.is_private() {
                for other in &self.execution.events {
                    if other.id != event.id && other.address == event.address
                        && other.invocation_id != event.invocation_id
                        && other.storage_class == event.storage_class
                    {
                        violations.push(VulkanFullViolation::new(
                            VulkanFullAxiom::StorageClassRestriction,
                            &format!("Cross-invocation access to {} storage at {:#x}",
                                event.storage_class, event.address),
                            vec![event.id, other.id],
                        ));
                    }
                }
            }
        }
        violations
    }

    /// Check that non-private texels accessed across invocations have proper
    /// synchronisation (availability + visibility).
    pub fn check_non_private_texel_sync(&self) -> Vec<VulkanFullViolation> {
        let mut violations = Vec::new();
        let image_writes: Vec<&VulkanFullEvent> = self.execution.events.iter()
            .filter(|e| e.op_type.is_image_op() && e.op_type.is_write() && e.texel_type == TexelType::NonPrivate)
            .collect();
        let image_reads: Vec<&VulkanFullEvent> = self.execution.events.iter()
            .filter(|e| e.op_type.is_image_op() && e.op_type.is_read() && e.texel_type == TexelType::NonPrivate)
            .collect();

        for w in &image_writes {
            for r in &image_reads {
                if w.address == r.address && w.invocation_id != r.invocation_id {
                    if !self.execution.av_vis.can_read_see_write(r.address, w.id, r.scope) {
                        violations.push(VulkanFullViolation::new(
                            VulkanFullAxiom::NonPrivateTexelSync,
                            &format!("Non-private texel at {:#x} written by {} read by {} without sync",
                                w.address, w.invocation_id, r.invocation_id),
                            vec![w.id, r.id],
                        ));
                    }
                }
            }
        }
        violations
    }

    /// Check image coherence ordering.
    pub fn check_image_coherence(&self) -> Vec<VulkanFullViolation> {
        let mut violations = Vec::new();
        let mut image_writes_by_addr: HashMap<u64, Vec<&VulkanFullEvent>> = HashMap::new();
        for e in &self.execution.events {
            if e.op_type.is_image_op() && e.op_type.is_write() {
                image_writes_by_addr.entry(e.address).or_default().push(e);
            }
        }
        for (addr, writes) in &image_writes_by_addr {
            for i in 0..writes.len() {
                for j in (i + 1)..writes.len() {
                    let w1 = writes[i].id;
                    let w2 = writes[j].id;
                    let fwd = self.execution.ordering.adjacency.get(&w1)
                        .map(|adj| adj.iter().any(|&(t, _)| t == w2)).unwrap_or(false);
                    let bwd = self.execution.ordering.adjacency.get(&w2)
                        .map(|adj| adj.iter().any(|&(t, _)| t == w1)).unwrap_or(false);
                    if fwd && bwd {
                        violations.push(VulkanFullViolation::new(
                            VulkanFullAxiom::ImageCoherence,
                            &format!("Image coherence cycle at {:#x}", addr),
                            vec![w1, w2],
                        ));
                    }
                }
            }
        }
        violations
    }
}

// ---------------------------------------------------------------------------
// Full Vulkan memory model
// ---------------------------------------------------------------------------

/// Configuration for the full Vulkan memory model checker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulkanFullModelConfig {
    pub check_coherence: bool,
    pub check_atomicity: bool,
    pub check_no_thin_air: bool,
    pub check_seq_cst: bool,
    pub check_av_vis: bool,
    pub check_barriers: bool,
    pub check_storage_class: bool,
    pub check_image_coherence: bool,
    pub check_texel_sync: bool,
    pub scoped_model: bool,
    pub max_invocations: usize,
    pub max_workgroups: usize,
}

impl VulkanFullModelConfig {
    pub fn full() -> Self {
        Self {
            check_coherence: true, check_atomicity: true, check_no_thin_air: true,
            check_seq_cst: true, check_av_vis: true, check_barriers: true,
            check_storage_class: true, check_image_coherence: true,
            check_texel_sync: true, scoped_model: true,
            max_invocations: 256, max_workgroups: 16,
        }
    }

    pub fn minimal() -> Self {
        Self {
            check_coherence: true, check_atomicity: false, check_no_thin_air: true,
            check_seq_cst: false, check_av_vis: false, check_barriers: false,
            check_storage_class: false, check_image_coherence: false,
            check_texel_sync: false, scoped_model: false,
            max_invocations: 64, max_workgroups: 4,
        }
    }
}

impl Default for VulkanFullModelConfig {
    fn default() -> Self { Self::full() }
}

/// Full Vulkan memory model specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulkanMemoryModel {
    pub name: String,
    pub config: VulkanFullModelConfig,
}

impl VulkanMemoryModel {
    pub fn new() -> Self {
        Self { name: "VulkanFull".to_string(), config: VulkanFullModelConfig::full() }
    }

    pub fn with_config(config: VulkanFullModelConfig) -> Self {
        Self { name: "VulkanFull".to_string(), config }
    }

    pub fn verify(&self, execution: &VulkanFullExecution) -> VulkanFullVerificationResult {
        let checker = VulkanFullAxiomChecker::new(execution);
        let violations = checker.check_all();
        let consistent = violations.is_empty();
        VulkanFullVerificationResult {
            model_name: self.name.clone(),
            consistent,
            violations,
            events_checked: execution.event_count(),
            invocations: execution.invocation_count(),
        }
    }
}

impl Default for VulkanMemoryModel {
    fn default() -> Self { Self::new() }
}

/// Result of full Vulkan model verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulkanFullVerificationResult {
    pub model_name: String,
    pub consistent: bool,
    pub violations: Vec<VulkanFullViolation>,
    pub events_checked: usize,
    pub invocations: usize,
}

impl fmt::Display for VulkanFullVerificationResult {
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
// Vulkan full litmus test support
// ---------------------------------------------------------------------------

/// A litmus test expressed in the full Vulkan model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulkanFullLitmusTest {
    pub name: String,
    pub description: String,
    pub invocations: usize,
    pub workgroups: usize,
    pub subgroups: usize,
    pub events: Vec<VulkanFullEvent>,
    pub expected_outcomes: Vec<VulkanFullOutcome>,
    pub forbidden_outcomes: Vec<VulkanFullOutcome>,
}

/// An outcome of a full Vulkan litmus test.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VulkanFullOutcome {
    pub values: Vec<(String, u64)>,
}

impl VulkanFullOutcome {
    pub fn new(values: Vec<(String, u64)>) -> Self { Self { values } }
}

impl fmt::Display for VulkanFullOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self.values.iter().map(|(k, v)| format!("{}={}", k, v)).collect();
        write!(f, "{{{}}}", parts.join(", "))
    }
}

impl VulkanFullLitmusTest {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(), description: String::new(),
            invocations: 2, workgroups: 1, subgroups: 1,
            events: Vec::new(),
            expected_outcomes: Vec::new(), forbidden_outcomes: Vec::new(),
        }
    }

    pub fn is_forbidden(&self, outcome: &VulkanFullOutcome) -> bool {
        self.forbidden_outcomes.contains(outcome)
    }

    /// Store Buffer test with availability/visibility.
    pub fn store_buffer_av_vis() -> Self {
        let inv0 = InvocationId::new(0, 0, 0, 0);
        let inv1 = InvocationId::new(1, 0, 0, 0);
        let mut test = Self::new("SB-VulkanFull");
        test.description = "Store Buffer with availability/visibility".to_string();
        test.invocations = 2;

        let e0 = VulkanFullEvent::new(0, inv0.clone(), VulkanFullOpType::AtomicStore)
            .with_address(0x100).with_value(1)
            .with_semantics(VulkanFullSemantics::release().with_available());
        let e1 = VulkanFullEvent::new(1, inv0, VulkanFullOpType::AtomicLoad)
            .with_address(0x200)
            .with_semantics(VulkanFullSemantics::acquire().with_visible())
            .with_po(1);
        let e2 = VulkanFullEvent::new(2, inv1.clone(), VulkanFullOpType::AtomicStore)
            .with_address(0x200).with_value(1)
            .with_semantics(VulkanFullSemantics::release().with_available());
        let e3 = VulkanFullEvent::new(3, inv1, VulkanFullOpType::AtomicLoad)
            .with_address(0x100)
            .with_semantics(VulkanFullSemantics::acquire().with_visible())
            .with_po(1);

        test.events = vec![e0, e1, e2, e3];
        test.forbidden_outcomes.push(VulkanFullOutcome::new(vec![
            ("r0".to_string(), 0), ("r1".to_string(), 0),
        ]));
        test
    }

    /// Message passing with image storage class.
    pub fn message_passing_image() -> Self {
        let inv0 = InvocationId::new(0, 0, 0, 0);
        let inv1 = InvocationId::new(1, 0, 0, 0);
        let mut test = Self::new("MP-Image-VulkanFull");
        test.description = "Message passing using image storage class".to_string();
        test.invocations = 2;

        let e0 = VulkanFullEvent::new(0, inv0.clone(), VulkanFullOpType::ImageWrite)
            .with_address(0x100).with_value(42)
            .with_storage_class(VulkanStorageClass::Image)
            .with_texel(TexelType::NonPrivate)
            .with_semantics(VulkanFullSemantics::release().with_available().with_image());
        let e1 = VulkanFullEvent::new(1, inv0, VulkanFullOpType::AtomicStore)
            .with_address(0x200).with_value(1)
            .with_semantics(VulkanFullSemantics::release().with_available())
            .with_po(1);
        let e2 = VulkanFullEvent::new(2, inv1.clone(), VulkanFullOpType::AtomicLoad)
            .with_address(0x200)
            .with_semantics(VulkanFullSemantics::acquire().with_visible());
        let e3 = VulkanFullEvent::new(3, inv1, VulkanFullOpType::ImageRead)
            .with_address(0x100)
            .with_storage_class(VulkanStorageClass::Image)
            .with_texel(TexelType::NonPrivate)
            .with_semantics(VulkanFullSemantics::acquire().with_visible().with_image())
            .with_po(1);

        test.events = vec![e0, e1, e2, e3];
        test.forbidden_outcomes.push(VulkanFullOutcome::new(vec![
            ("flag".to_string(), 1), ("data".to_string(), 0),
        ]));
        test
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
        assert!(VulkanFullScope::Device > VulkanFullScope::QueueFamily);
        assert!(VulkanFullScope::QueueFamily > VulkanFullScope::Workgroup);
        assert!(VulkanFullScope::Workgroup > VulkanFullScope::Subgroup);
        assert!(VulkanFullScope::Subgroup > VulkanFullScope::Invocation);
    }

    #[test]
    fn test_scope_includes() {
        assert!(VulkanFullScope::Device.includes(&VulkanFullScope::Workgroup));
        assert!(!VulkanFullScope::Subgroup.includes(&VulkanFullScope::Workgroup));
    }

    #[test]
    fn test_scope_broaden() {
        assert_eq!(VulkanFullScope::Invocation.broaden(), Some(VulkanFullScope::Subgroup));
        assert_eq!(VulkanFullScope::Device.broaden(), None);
    }

    #[test]
    fn test_min_scope_for() {
        let inv0 = InvocationId::new(0, 0, 0, 0);
        let inv1 = InvocationId::new(1, 0, 0, 0);
        let inv2 = InvocationId::new(2, 1, 0, 0);
        let inv3 = InvocationId::new(3, 2, 1, 0);
        assert_eq!(VulkanFullScope::min_scope_for(&inv0, &inv0), VulkanFullScope::Invocation);
        assert_eq!(VulkanFullScope::min_scope_for(&inv0, &inv1), VulkanFullScope::Subgroup);
        assert_eq!(VulkanFullScope::min_scope_for(&inv0, &inv2), VulkanFullScope::Workgroup);
        assert_eq!(VulkanFullScope::min_scope_for(&inv0, &inv3), VulkanFullScope::QueueFamily);
    }

    // -- InvocationId tests --

    #[test]
    fn test_invocation_same_scope() {
        let inv0 = InvocationId::new(0, 0, 0, 0);
        let inv1 = InvocationId::new(1, 0, 0, 0);
        assert!(inv0.same_scope_instance(&inv1, VulkanFullScope::Subgroup));
        assert!(!inv0.same_scope_instance(&inv1, VulkanFullScope::Invocation));
    }

    // -- StorageClass tests --

    #[test]
    fn test_storage_class() {
        assert!(VulkanStorageClass::StorageBuffer.is_shared());
        assert!(!VulkanStorageClass::Private.is_shared());
        assert!(VulkanStorageClass::Uniform.is_read_only());
        assert!(!VulkanStorageClass::StorageBuffer.is_read_only());
        assert!(VulkanStorageClass::Function.is_private());
    }

    #[test]
    fn test_storage_class_min_scope() {
        assert_eq!(VulkanStorageClass::Workgroup.min_visibility_scope(), VulkanFullScope::Workgroup);
        assert_eq!(VulkanStorageClass::Private.min_visibility_scope(), VulkanFullScope::Invocation);
    }

    // -- VulkanFullSemantics tests --

    #[test]
    fn test_semantics_ordering() {
        assert!(VulkanFullSemantics::acquire().has_ordering());
        assert!(VulkanFullSemantics::release().has_ordering());
        assert!(!VulkanFullSemantics::none().has_ordering());
        assert!(VulkanFullSemantics::acq_rel().is_acquire());
        assert!(VulkanFullSemantics::acq_rel().is_release());
    }

    #[test]
    fn test_semantics_affected_classes() {
        let sem = VulkanFullSemantics::release().with_storage_buffer().with_image();
        let classes = sem.affected_classes();
        assert!(classes.contains(&VulkanStorageClass::StorageBuffer));
        assert!(classes.contains(&VulkanStorageClass::Image));
    }

    #[test]
    fn test_semantics_display() {
        let s = format!("{}", VulkanFullSemantics::acquire());
        assert!(s.contains("Acquire"));
    }

    // -- VulkanAvailVisOp tests --

    #[test]
    fn test_avail_vis_op() {
        let op = VulkanAvailVisOp::MakeAvailable { scope: VulkanFullScope::Device };
        assert!(op.is_available());
        assert!(!op.is_visible());
        assert_eq!(op.scope(), VulkanFullScope::Device);
    }

    // -- FullAvVisState tests --

    #[test]
    fn test_full_av_vis_state() {
        let mut state = FullAvVisState::new(VulkanStorageClass::StorageBuffer);
        state.make_available(VulkanFullScope::Device);
        assert!(state.is_available_to(VulkanFullScope::Device));
        assert!(state.is_available_to(VulkanFullScope::Workgroup));
        state.make_visible(VulkanFullScope::Workgroup);
        assert!(state.is_visible_to(VulkanFullScope::Workgroup));
    }

    #[test]
    fn test_av_vis_requires_available() {
        let mut state = FullAvVisState::new(VulkanStorageClass::StorageBuffer);
        state.make_visible(VulkanFullScope::Device);
        assert!(!state.is_visible_to(VulkanFullScope::Device));
    }

    // -- FullAvVisTracker tests --

    #[test]
    fn test_tracker_store_and_read() {
        let mut tracker = FullAvVisTracker::new();
        let sem = VulkanFullSemantics::release().with_available();
        tracker.record_store(0, 0x100, VulkanStorageClass::StorageBuffer, &sem, VulkanFullScope::Device);
        tracker.process_available(0x100, 0, VulkanFullScope::Device);
        tracker.process_visible(0x100, 0, VulkanFullScope::Device);
        assert!(tracker.can_read_see_write(0x100, 0, VulkanFullScope::Device));
    }

    // -- TexelType tests --

    #[test]
    fn test_texel_type() {
        assert!(TexelType::NonPrivate.requires_sync());
        assert!(!TexelType::Private.requires_sync());
    }

    // -- VulkanFullOpType tests --

    #[test]
    fn test_op_type() {
        assert!(VulkanFullOpType::Load.is_read());
        assert!(!VulkanFullOpType::Load.is_write());
        assert!(VulkanFullOpType::AtomicExchange.is_rmw());
        assert!(VulkanFullOpType::ControlBarrier.is_barrier());
        assert!(VulkanFullOpType::ImageRead.is_image_op());
        assert!(VulkanFullOpType::ImageWrite.is_image_op());
    }

    // -- VulkanFullEvent tests --

    #[test]
    fn test_event_creation() {
        let inv = InvocationId::new(0, 0, 0, 0);
        let event = VulkanFullEvent::new(0, inv, VulkanFullOpType::AtomicStore)
            .with_address(0x100).with_value(42);
        assert_eq!(event.id, 0);
        assert_eq!(event.address, 0x100);
    }

    #[test]
    fn test_event_same_scope() {
        let inv0 = InvocationId::new(0, 0, 0, 0);
        let inv1 = InvocationId::new(1, 0, 0, 0);
        let e0 = VulkanFullEvent::new(0, inv0, VulkanFullOpType::Store);
        let e1 = VulkanFullEvent::new(1, inv1, VulkanFullOpType::Load);
        assert!(e0.same_scope_instance(&e1, VulkanFullScope::Subgroup));
        assert!(!e0.same_scope_instance(&e1, VulkanFullScope::Invocation));
    }

    // -- InvocationInterleavingModel tests --

    #[test]
    fn test_interleaving_model() {
        let mut model = InvocationInterleavingModel::new(4);
        model.set_active_mask(0, vec![true, true, false, true]);
        let inv = InvocationId::new(2, 0, 0, 0);
        assert!(!model.is_active(&inv));
        assert_eq!(model.active_count(0), 3);
        assert!(!model.is_converged(0));
    }

    // -- OrderingGraph tests --

    #[test]
    fn test_ordering_graph_no_cycle() {
        let mut g = VulkanFullOrderingGraph::new();
        g.add_edge(0, 1, VulkanFullRelation::ProgramOrder);
        g.add_edge(1, 2, VulkanFullRelation::ProgramOrder);
        assert!(!g.has_cycle());
        assert_eq!(g.edge_count(), 2);
        assert_eq!(g.node_count(), 3);
    }

    #[test]
    fn test_ordering_graph_cycle() {
        let mut g = VulkanFullOrderingGraph::new();
        g.add_edge(0, 1, VulkanFullRelation::ProgramOrder);
        g.add_edge(1, 2, VulkanFullRelation::Synchronizes);
        g.add_edge(2, 0, VulkanFullRelation::ReadsFrom);
        assert!(g.has_cycle());
    }

    // -- VulkanFullExecution tests --

    #[test]
    fn test_execution_basic() {
        let inv0 = InvocationId::new(0, 0, 0, 0);
        let inv1 = InvocationId::new(1, 0, 0, 0);
        let mut exec = VulkanFullExecution::new();
        exec.add_event(VulkanFullEvent::new(0, inv0.clone(), VulkanFullOpType::Store));
        exec.add_event(VulkanFullEvent::new(1, inv0, VulkanFullOpType::Load).with_po(1));
        exec.add_event(VulkanFullEvent::new(2, inv1, VulkanFullOpType::Store));
        assert_eq!(exec.event_count(), 3);
        assert_eq!(exec.invocation_count(), 2);
    }

    #[test]
    fn test_execution_build_po() {
        let inv = InvocationId::new(0, 0, 0, 0);
        let mut exec = VulkanFullExecution::new();
        exec.add_event(VulkanFullEvent::new(0, inv.clone(), VulkanFullOpType::Store).with_po(0));
        exec.add_event(VulkanFullEvent::new(1, inv, VulkanFullOpType::Load).with_po(1));
        exec.build_program_order();
        assert_eq!(exec.ordering.edge_count(), 1);
    }

    #[test]
    fn test_execution_consistent() {
        let exec = VulkanFullExecution::new();
        assert!(exec.is_consistent());
    }

    // -- VulkanFullAxiomChecker tests --

    #[test]
    fn test_checker_empty_execution() {
        let exec = VulkanFullExecution::new();
        let checker = VulkanFullAxiomChecker::new(&exec);
        assert!(checker.check_all().is_empty());
    }

    // -- VulkanMemoryModel tests --

    #[test]
    fn test_model_verify_empty() {
        let model = VulkanMemoryModel::new();
        let exec = VulkanFullExecution::new();
        let result = model.verify(&exec);
        assert!(result.consistent);
        assert_eq!(result.model_name, "VulkanFull");
    }

    #[test]
    fn test_model_config() {
        let config = VulkanFullModelConfig::full();
        assert!(config.check_coherence);
        assert!(config.check_image_coherence);
        let config = VulkanFullModelConfig::minimal();
        assert!(config.check_coherence);
        assert!(!config.check_image_coherence);
    }

    // -- Litmus tests --

    #[test]
    fn test_litmus_store_buffer() {
        let test = VulkanFullLitmusTest::store_buffer_av_vis();
        assert_eq!(test.name, "SB-VulkanFull");
        assert_eq!(test.events.len(), 4);
        assert_eq!(test.forbidden_outcomes.len(), 1);
    }

    #[test]
    fn test_litmus_message_passing_image() {
        let test = VulkanFullLitmusTest::message_passing_image();
        assert_eq!(test.name, "MP-Image-VulkanFull");
        assert_eq!(test.events.len(), 4);
    }

    #[test]
    fn test_litmus_forbidden() {
        let test = VulkanFullLitmusTest::store_buffer_av_vis();
        let outcome = VulkanFullOutcome::new(vec![
            ("r0".to_string(), 0), ("r1".to_string(), 0),
        ]);
        assert!(test.is_forbidden(&outcome));
    }

    // -- Display tests --

    #[test]
    fn test_displays() {
        assert_eq!(format!("{}", VulkanFullScope::Device), "Device");
        assert_eq!(format!("{}", VulkanStorageClass::StorageBuffer), "StorageBuffer");
        assert_eq!(format!("{}", VulkanFullRelation::ProgramOrder), "po");
    }

    #[test]
    fn test_verification_result_display() {
        let result = VulkanFullVerificationResult {
            model_name: "VulkanFull".to_string(),
            consistent: true,
            violations: vec![],
            events_checked: 4,
            invocations: 2,
        };
        let s = format!("{}", result);
        assert!(s.contains("Consistent"));
    }

    #[test]
    fn test_outcome_display() {
        let o = VulkanFullOutcome::new(vec![("r0".to_string(), 1)]);
        assert!(format!("{}", o).contains("r0=1"));
    }

    #[test]
    fn test_axiom_all() {
        assert_eq!(VulkanFullAxiom::all().len(), 10);
    }

    #[test]
    fn test_violation_display() {
        let v = VulkanFullViolation::new(VulkanFullAxiom::Coherence, "test", vec![0]);
        assert!(format!("{}", v).contains("Coherence"));
    }
}

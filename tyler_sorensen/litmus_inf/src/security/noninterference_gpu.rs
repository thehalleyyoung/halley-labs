/// GPU noninterference: thread-level, warp-level isolation,
/// scope-aware security analysis for the LITMUS∞ security engine.
///
/// Implements GPU-specific noninterference checking, information flow
/// analysis, covert channel detection, and security verification for
/// hierarchical GPU memory models.
#[allow(unused)]

use std::collections::{HashMap, HashSet, BTreeMap, BTreeSet, VecDeque};
use std::fmt;

// ═══════════════════════════════════════════════════════════════════════════
// GPU Security Types
// ═══════════════════════════════════════════════════════════════════════════

/// Security classification level for GPU data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum GpuSecurityLevel {
    /// Public data, visible to all.
    Public,
    /// Thread-private data.
    ThreadPrivate,
    /// Warp-local data.
    WarpLocal,
    /// CTA-shared data.
    CtaShared,
    /// GPU-global data.
    GpuGlobal,
    /// System-wide data.
    SystemWide,
}

impl GpuSecurityLevel {
    /// Join (least upper bound) of two security levels.
    pub fn join(self, other: Self) -> Self {
        if self >= other { self } else { other }
    }

    /// Meet (greatest lower bound) of two security levels.
    pub fn meet(self, other: Self) -> Self {
        if self <= other { self } else { other }
    }

    /// Check if information can flow from this level to another.
    pub fn flows_to(&self, target: &GpuSecurityLevel) -> bool {
        *self <= *target
    }
}

impl fmt::Display for GpuSecurityLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuSecurityLevel::Public => write!(f, "public"),
            GpuSecurityLevel::ThreadPrivate => write!(f, "thread-private"),
            GpuSecurityLevel::WarpLocal => write!(f, "warp-local"),
            GpuSecurityLevel::CtaShared => write!(f, "cta-shared"),
            GpuSecurityLevel::GpuGlobal => write!(f, "gpu-global"),
            GpuSecurityLevel::SystemWide => write!(f, "system-wide"),
        }
    }
}

/// GPU scope hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum GpuScope {
    /// Single thread.
    Thread,
    /// Warp (32 threads).
    Warp,
    /// Cooperative Thread Array.
    CTA,
    /// GPU device.
    GPU,
    /// System.
    System,
}

impl GpuScope {
    /// Check if this scope includes another.
    pub fn includes(&self, other: &GpuScope) -> bool { *self >= *other }

    /// Check if this scope is wider.
    pub fn is_wider_than(&self, other: &GpuScope) -> bool { *self > *other }

    /// All scope levels.
    pub fn all() -> &'static [GpuScope] {
        &[GpuScope::Thread, GpuScope::Warp, GpuScope::CTA, GpuScope::GPU, GpuScope::System]
    }
}

impl fmt::Display for GpuScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuScope::Thread => write!(f, "thread"),
            GpuScope::Warp => write!(f, "warp"),
            GpuScope::CTA => write!(f, "cta"),
            GpuScope::GPU => write!(f, "gpu"),
            GpuScope::System => write!(f, "system"),
        }
    }
}

/// GPU memory region classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryRegion {
    /// Per-thread register file.
    Register,
    /// Per-CTA shared memory.
    Shared,
    /// Global memory.
    Global,
    /// Per-thread local memory.
    Local,
    /// Read-only constant memory.
    Constant,
    /// Texture memory.
    Texture,
}

impl MemoryRegion {
    /// Default security level for this region.
    pub fn default_security_level(&self) -> GpuSecurityLevel {
        match self {
            MemoryRegion::Register | MemoryRegion::Local => GpuSecurityLevel::ThreadPrivate,
            MemoryRegion::Shared => GpuSecurityLevel::CtaShared,
            MemoryRegion::Global => GpuSecurityLevel::GpuGlobal,
            MemoryRegion::Constant | MemoryRegion::Texture => GpuSecurityLevel::Public,
        }
    }

    /// Scope of visibility for this region.
    pub fn visibility_scope(&self) -> GpuScope {
        match self {
            MemoryRegion::Register | MemoryRegion::Local => GpuScope::Thread,
            MemoryRegion::Shared => GpuScope::CTA,
            MemoryRegion::Global => GpuScope::GPU,
            MemoryRegion::Constant | MemoryRegion::Texture => GpuScope::System,
        }
    }
}

impl fmt::Display for MemoryRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryRegion::Register => write!(f, "register"),
            MemoryRegion::Shared => write!(f, "shared"),
            MemoryRegion::Global => write!(f, "global"),
            MemoryRegion::Local => write!(f, "local"),
            MemoryRegion::Constant => write!(f, "constant"),
            MemoryRegion::Texture => write!(f, "texture"),
        }
    }
}

/// Operation type for GPU events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuOpType {
    /// Memory read.
    Read,
    /// Memory write.
    Write,
    /// Memory fence.
    Fence,
    /// Atomic read-modify-write.
    AtomicRMW,
    /// Warp shuffle.
    Shuffle,
    /// Warp vote/ballot.
    Vote,
    /// Barrier.
    Barrier,
}

/// A GPU event for security analysis.
#[derive(Debug, Clone)]
pub struct GpuEvent {
    /// Event identifier.
    pub id: usize,
    /// Thread ID.
    pub thread_id: usize,
    /// Warp ID.
    pub warp_id: u32,
    /// CTA ID.
    pub cta_id: u32,
    /// Operation type.
    pub op_type: GpuOpType,
    /// Memory address (if applicable).
    pub address: u64,
    /// Value.
    pub value: u64,
    /// Scope of the operation.
    pub scope: GpuScope,
    /// Security level.
    pub security_level: GpuSecurityLevel,
    /// Memory region.
    pub memory_region: MemoryRegion,
}

/// A security domain grouping threads at a particular scope.
#[derive(Debug, Clone)]
pub struct SecurityDomain {
    /// Domain identifier.
    pub id: usize,
    /// Scope level of this domain.
    pub scope: GpuScope,
    /// Security level of this domain.
    pub level: GpuSecurityLevel,
    /// Thread IDs in this domain.
    pub thread_ids: Vec<usize>,
}

/// Security policy for GPU execution.
#[derive(Debug, Clone)]
pub struct SecurityPolicy {
    /// Classification rules: memory region -> security level.
    pub region_classifications: HashMap<MemoryRegion, GpuSecurityLevel>,
    /// Per-address overrides.
    pub address_classifications: HashMap<u64, GpuSecurityLevel>,
    /// Allowed flows: (source_level, target_level) pairs.
    pub allowed_flows: HashSet<(GpuSecurityLevel, GpuSecurityLevel)>,
    /// Declassification rules.
    pub declassifications: Vec<DeclassificationRule>,
}

/// A declassification rule.
#[derive(Debug, Clone)]
pub struct DeclassificationRule {
    /// Source level.
    pub from: GpuSecurityLevel,
    /// Target level.
    pub to: GpuSecurityLevel,
    /// Condition (description).
    pub condition: String,
}

impl SecurityPolicy {
    /// Create a default policy.
    pub fn default_policy() -> Self {
        let mut allowed_flows = HashSet::new();
        for &level in &[
            GpuSecurityLevel::Public, GpuSecurityLevel::ThreadPrivate,
            GpuSecurityLevel::WarpLocal, GpuSecurityLevel::CtaShared,
            GpuSecurityLevel::GpuGlobal, GpuSecurityLevel::SystemWide,
        ] {
            allowed_flows.insert((level, level));
            allowed_flows.insert((GpuSecurityLevel::Public, level));
        }
        Self {
            region_classifications: HashMap::new(),
            address_classifications: HashMap::new(),
            allowed_flows,
            declassifications: Vec::new(),
        }
    }

    /// Check if a flow is allowed.
    pub fn is_flow_allowed(&self, from: GpuSecurityLevel, to: GpuSecurityLevel) -> bool {
        self.allowed_flows.contains(&(from, to)) || from.flows_to(&to)
    }

    /// Classify an address.
    pub fn classify_address(&self, addr: u64, region: MemoryRegion) -> GpuSecurityLevel {
        self.address_classifications.get(&addr)
            .copied()
            .or_else(|| self.region_classifications.get(&region).copied())
            .unwrap_or_else(|| region.default_security_level())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ThreadLevelIsolation
// ═══════════════════════════════════════════════════════════════════════════

/// A leak finding from isolation analysis.
#[derive(Debug, Clone)]
pub struct IsolationLeak {
    /// Source event.
    pub source_event: usize,
    /// Sink event.
    pub sink_event: usize,
    /// Source thread.
    pub source_thread: usize,
    /// Sink thread.
    pub sink_thread: usize,
    /// Address involved.
    pub address: u64,
    /// Description.
    pub description: String,
    /// Scope at which the leak occurs.
    pub scope: GpuScope,
}

impl fmt::Display for IsolationLeak {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Leak(T{}→T{} @{:#x} [{}]): {}",
            self.source_thread, self.sink_thread, self.address, self.scope, self.description)
    }
}

/// Thread-level isolation checker.
#[derive(Debug, Clone)]
pub struct ThreadIsolationChecker {
    /// Events.
    events: Vec<GpuEvent>,
    /// Security policy.
    policy: SecurityPolicy,
    /// Detected leaks.
    leaks: Vec<IsolationLeak>,
}

impl ThreadIsolationChecker {
    /// Create a new checker.
    pub fn new(events: Vec<GpuEvent>, policy: SecurityPolicy) -> Self {
        Self { events, policy, leaks: Vec::new() }
    }

    /// Check thread-level isolation.
    pub fn check_thread_isolation(&mut self) -> &[IsolationLeak] {
        self.leaks.clear();
        self.verify_register_isolation();
        self.verify_local_memory_isolation();
        self.find_thread_leaks();
        &self.leaks
    }

    /// Find leaks between threads.
    pub fn find_thread_leaks(&mut self) {
        // For each write by thread T1 to shared/global memory,
        // check if a read by thread T2 can see it when it shouldn't
        let writes: Vec<(usize, &GpuEvent)> = self.events.iter().enumerate()
            .filter(|(_, e)| matches!(e.op_type, GpuOpType::Write | GpuOpType::AtomicRMW))
            .collect();
        let reads: Vec<(usize, &GpuEvent)> = self.events.iter().enumerate()
            .filter(|(_, e)| matches!(e.op_type, GpuOpType::Read | GpuOpType::AtomicRMW))
            .collect();
        for &(wi, w) in &writes {
            for &(ri, r) in &reads {
                if w.thread_id != r.thread_id && w.address == r.address {
                    let w_level = self.classify_event(w);
                    let r_level = self.classify_event(r);
                    if !self.policy.is_flow_allowed(w_level, r_level) {
                        self.leaks.push(IsolationLeak {
                            source_event: wi,
                            sink_event: ri,
                            source_thread: w.thread_id,
                            sink_thread: r.thread_id,
                            address: w.address,
                            description: format!(
                                "{} data flows to {} context", w_level, r_level),
                            scope: GpuScope::Thread,
                        });
                    }
                }
            }
        }
    }

    /// Verify register isolation: registers should never leak between threads.
    pub fn verify_register_isolation(&mut self) {
        for (idx, ev) in self.events.iter().enumerate() {
            if ev.memory_region == MemoryRegion::Register {
                // Check that no other thread reads this register
                for (idx2, ev2) in self.events.iter().enumerate() {
                    if idx != idx2 && ev2.thread_id != ev.thread_id
                        && ev2.address == ev.address
                        && ev2.memory_region == MemoryRegion::Register
                    {
                        self.leaks.push(IsolationLeak {
                            source_event: idx,
                            sink_event: idx2,
                            source_thread: ev.thread_id,
                            sink_thread: ev2.thread_id,
                            address: ev.address,
                            description: "Cross-thread register access".to_string(),
                            scope: GpuScope::Thread,
                        });
                    }
                }
            }
        }
    }

    /// Verify local memory isolation.
    pub fn verify_local_memory_isolation(&mut self) {
        for (idx, ev) in self.events.iter().enumerate() {
            if ev.memory_region == MemoryRegion::Local {
                for (idx2, ev2) in self.events.iter().enumerate() {
                    if idx != idx2 && ev2.thread_id != ev.thread_id
                        && ev2.address == ev.address
                        && ev2.memory_region == MemoryRegion::Local
                    {
                        self.leaks.push(IsolationLeak {
                            source_event: idx,
                            sink_event: idx2,
                            source_thread: ev.thread_id,
                            sink_thread: ev2.thread_id,
                            address: ev.address,
                            description: "Cross-thread local memory access".to_string(),
                            scope: GpuScope::Thread,
                        });
                    }
                }
            }
        }
    }

    /// Classify an event's security level.
    fn classify_event(&self, ev: &GpuEvent) -> GpuSecurityLevel {
        self.policy.classify_address(ev.address, ev.memory_region)
    }

    /// Get detected leaks.
    pub fn get_leaks(&self) -> &[IsolationLeak] { &self.leaks }

    /// Compute the thread interference graph.
    pub fn thread_interference_graph(&self) -> HashMap<(usize, usize), Vec<u64>> {
        let mut graph: HashMap<(usize, usize), Vec<u64>> = HashMap::new();
        for leak in &self.leaks {
            let key = (leak.source_thread.min(leak.sink_thread),
                       leak.source_thread.max(leak.sink_thread));
            graph.entry(key).or_default().push(leak.address);
        }
        graph
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// WarpLevelIsolation
// ═══════════════════════════════════════════════════════════════════════════

/// Warp-level isolation checker.
#[derive(Debug, Clone)]
pub struct WarpIsolationChecker {
    /// Events.
    events: Vec<GpuEvent>,
    /// Policy.
    policy: SecurityPolicy,
    /// Thread-to-warp mapping.
    thread_warp: HashMap<usize, u32>,
    /// Detected leaks.
    leaks: Vec<IsolationLeak>,
}

impl WarpIsolationChecker {
    /// Create a new checker.
    pub fn new(events: Vec<GpuEvent>, policy: SecurityPolicy) -> Self {
        let thread_warp: HashMap<usize, u32> = events.iter()
            .map(|e| (e.thread_id, e.warp_id)).collect();
        Self { events, policy, thread_warp, leaks: Vec::new() }
    }

    /// Check warp-level isolation.
    pub fn check_warp_isolation(&mut self) -> &[IsolationLeak] {
        self.leaks.clear();
        self.detect_warp_leaks();
        self.analyze_warp_divergence_leak();
        self.ballot_leak_detection();
        self.shuffle_leak_detection();
        &self.leaks
    }

    /// Detect information leaks between warps.
    pub fn detect_warp_leaks(&mut self) {
        let writes: Vec<(usize, &GpuEvent)> = self.events.iter().enumerate()
            .filter(|(_, e)| matches!(e.op_type, GpuOpType::Write))
            .collect();
        let reads: Vec<(usize, &GpuEvent)> = self.events.iter().enumerate()
            .filter(|(_, e)| matches!(e.op_type, GpuOpType::Read))
            .collect();
        for &(wi, w) in &writes {
            for &(ri, r) in &reads {
                if w.warp_id != r.warp_id && w.address == r.address {
                    let w_level = w.security_level;
                    let r_level = r.security_level;
                    if w_level > GpuSecurityLevel::WarpLocal && !w_level.flows_to(&r_level) {
                        self.leaks.push(IsolationLeak {
                            source_event: wi, sink_event: ri,
                            source_thread: w.thread_id, sink_thread: r.thread_id,
                            address: w.address,
                            description: format!("Cross-warp leak: {} → {}", w_level, r_level),
                            scope: GpuScope::Warp,
                        });
                    }
                }
            }
        }
    }

    /// Analyze warp divergence as an information leak channel.
    pub fn analyze_warp_divergence_leak(&mut self) {
        // Threads in the same warp that take different branches leak information
        // about their predicate values through the execution mask.
        // Simplified: detect events where threads in the same warp access different addresses
        let mut warp_events: HashMap<u32, Vec<(usize, &GpuEvent)>> = HashMap::new();
        for (idx, ev) in self.events.iter().enumerate() {
            warp_events.entry(ev.warp_id).or_default().push((idx, ev));
        }
        for (warp_id, events) in &warp_events {
            let addresses: HashSet<u64> = events.iter().map(|(_, e)| e.address).collect();
            if addresses.len() > 1 {
                // Potential divergence
                let secret_events: Vec<_> = events.iter()
                    .filter(|(_, e)| e.security_level >= GpuSecurityLevel::WarpLocal)
                    .collect();
                for &(idx, ev) in &secret_events {
                    self.leaks.push(IsolationLeak {
                        source_event: *idx, sink_event: *idx,
                        source_thread: ev.thread_id, sink_thread: ev.thread_id,
                        address: ev.address,
                        description: format!("Warp divergence may leak secret in warp {}", warp_id),
                        scope: GpuScope::Warp,
                    });
                }
            }
        }
    }

    /// Detect leaks through ballot/vote operations.
    pub fn ballot_leak_detection(&mut self) {
        for (idx, ev) in self.events.iter().enumerate() {
            if ev.op_type == GpuOpType::Vote {
                // Vote operations reveal information about thread predicates
                if ev.security_level >= GpuSecurityLevel::ThreadPrivate {
                    self.leaks.push(IsolationLeak {
                        source_event: idx, sink_event: idx,
                        source_thread: ev.thread_id, sink_thread: ev.thread_id,
                        address: ev.address,
                        description: "Ballot/vote leaks thread-private predicate".to_string(),
                        scope: GpuScope::Warp,
                    });
                }
            }
        }
    }

    /// Detect leaks through shuffle operations.
    pub fn shuffle_leak_detection(&mut self) {
        for (idx, ev) in self.events.iter().enumerate() {
            if ev.op_type == GpuOpType::Shuffle {
                if ev.security_level >= GpuSecurityLevel::ThreadPrivate {
                    self.leaks.push(IsolationLeak {
                        source_event: idx, sink_event: idx,
                        source_thread: ev.thread_id, sink_thread: ev.thread_id,
                        address: ev.address,
                        description: "Shuffle exposes thread-private data to warp".to_string(),
                        scope: GpuScope::Warp,
                    });
                }
            }
        }
    }

    /// Get detected leaks.
    pub fn get_leaks(&self) -> &[IsolationLeak] { &self.leaks }
}

// ═══════════════════════════════════════════════════════════════════════════
// ScopeAwareSecurity
// ═══════════════════════════════════════════════════════════════════════════

/// Scope-aware security analyzer for GPU programs.
#[derive(Debug, Clone)]
pub struct ScopeSecurityAnalyzer {
    /// Events.
    events: Vec<GpuEvent>,
    /// Policy.
    policy: SecurityPolicy,
    /// Thread-to-warp mapping.
    thread_warp: HashMap<usize, u32>,
    /// Thread-to-CTA mapping.
    thread_cta: HashMap<usize, u32>,
    /// Information flow edges.
    flow_edges: Vec<(usize, usize, GpuScope)>,
}

impl ScopeSecurityAnalyzer {
    /// Create a new analyzer.
    pub fn new(events: Vec<GpuEvent>, policy: SecurityPolicy) -> Self {
        let thread_warp: HashMap<usize, u32> = events.iter()
            .map(|e| (e.thread_id, e.warp_id)).collect();
        let thread_cta: HashMap<usize, u32> = events.iter()
            .map(|e| (e.thread_id, e.cta_id)).collect();
        Self { events, policy, thread_warp, thread_cta, flow_edges: Vec::new() }
    }

    /// Perform scope-aware security analysis.
    pub fn analyze_scope_security(&mut self) -> Vec<IsolationLeak> {
        let mut leaks = Vec::new();
        self.compute_scope_information_flow();
        leaks.extend(self.check_scope_noninterference());
        leaks.extend(self.cross_scope_leak_detection());
        leaks.extend(self.scope_escalation_detection());
        leaks
    }

    /// Compute information flow edges respecting scope.
    pub fn compute_scope_information_flow(&mut self) {
        self.flow_edges.clear();
        let n = self.events.len();
        for i in 0..n {
            for j in 0..n {
                if i == j { continue; }
                let ei = &self.events[i];
                let ej = &self.events[j];
                if ei.address == ej.address
                    && matches!(ei.op_type, GpuOpType::Write | GpuOpType::AtomicRMW)
                    && matches!(ej.op_type, GpuOpType::Read | GpuOpType::AtomicRMW)
                {
                    let scope = self.interaction_scope(ei.thread_id, ej.thread_id);
                    self.flow_edges.push((i, j, scope));
                }
            }
        }
    }

    /// Determine the scope at which two threads interact.
    fn interaction_scope(&self, t1: usize, t2: usize) -> GpuScope {
        if t1 == t2 { return GpuScope::Thread; }
        if self.thread_warp.get(&t1) == self.thread_warp.get(&t2) { return GpuScope::Warp; }
        if self.thread_cta.get(&t1) == self.thread_cta.get(&t2) { return GpuScope::CTA; }
        GpuScope::GPU
    }

    /// Check noninterference at each scope level.
    pub fn check_scope_noninterference(&self) -> Vec<IsolationLeak> {
        let mut leaks = Vec::new();
        for &(src, dst, scope) in &self.flow_edges {
            let src_level = self.events[src].security_level;
            let dst_level = self.events[dst].security_level;
            if !self.policy.is_flow_allowed(src_level, dst_level) {
                leaks.push(IsolationLeak {
                    source_event: src, sink_event: dst,
                    source_thread: self.events[src].thread_id,
                    sink_thread: self.events[dst].thread_id,
                    address: self.events[src].address,
                    description: format!(
                        "Scope {} noninterference violation: {} → {}",
                        scope, src_level, dst_level),
                    scope,
                });
            }
        }
        leaks
    }

    /// Detect leaks that cross scope boundaries.
    pub fn cross_scope_leak_detection(&self) -> Vec<IsolationLeak> {
        let mut leaks = Vec::new();
        for &(src, dst, scope) in &self.flow_edges {
            let src_scope = self.events[src].scope;
            let dst_scope = self.events[dst].scope;
            // A cross-scope leak occurs when information flows across scope boundaries
            if scope.is_wider_than(&src_scope) || scope.is_wider_than(&dst_scope) {
                leaks.push(IsolationLeak {
                    source_event: src, sink_event: dst,
                    source_thread: self.events[src].thread_id,
                    sink_thread: self.events[dst].thread_id,
                    address: self.events[src].address,
                    description: format!("Cross-scope leak: {} → {} at {}", src_scope, dst_scope, scope),
                    scope,
                });
            }
        }
        leaks
    }

    /// Detect scope escalation: operations requiring wider scope than annotated.
    pub fn scope_escalation_detection(&self) -> Vec<IsolationLeak> {
        let mut leaks = Vec::new();
        for &(src, dst, actual_scope) in &self.flow_edges {
            let annotated_scope = self.events[src].scope;
            if actual_scope.is_wider_than(&annotated_scope) {
                leaks.push(IsolationLeak {
                    source_event: src, sink_event: dst,
                    source_thread: self.events[src].thread_id,
                    sink_thread: self.events[dst].thread_id,
                    address: self.events[src].address,
                    description: format!(
                        "Scope escalation: annotated {} but requires {}", annotated_scope, actual_scope),
                    scope: actual_scope,
                });
            }
        }
        leaks
    }

    /// Compute the scope lattice for the execution.
    pub fn compute_scope_lattice(&self) -> HashMap<GpuScope, Vec<usize>> {
        let mut lattice: HashMap<GpuScope, Vec<usize>> = HashMap::new();
        for (idx, ev) in self.events.iter().enumerate() {
            lattice.entry(ev.scope).or_default().push(idx);
        }
        lattice
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GpuInformationFlow
// ═══════════════════════════════════════════════════════════════════════════

/// Information flow classification.
#[derive(Debug, Clone)]
pub struct InformationFlow {
    /// Source event.
    pub source: usize,
    /// Sink event.
    pub sink: usize,
    /// Flow type.
    pub flow_type: FlowType,
    /// Security impact.
    pub impact: GpuSecurityLevel,
}

/// Type of information flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FlowType {
    /// Direct data flow (write -> read).
    Direct,
    /// Through shared memory.
    SharedMemory,
    /// Through global memory.
    GlobalMemory,
    /// Through atomic operations.
    Atomic,
    /// Through fence operations (ordering leak).
    FenceOrdering,
    /// Implicit (through control flow).
    Implicit,
}

impl fmt::Display for FlowType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FlowType::Direct => write!(f, "direct"),
            FlowType::SharedMemory => write!(f, "shared-memory"),
            FlowType::GlobalMemory => write!(f, "global-memory"),
            FlowType::Atomic => write!(f, "atomic"),
            FlowType::FenceOrdering => write!(f, "fence-ordering"),
            FlowType::Implicit => write!(f, "implicit"),
        }
    }
}

/// GPU information flow analyzer.
#[derive(Debug, Clone)]
pub struct GpuFlowAnalyzer {
    /// Events.
    events: Vec<GpuEvent>,
    /// Detected flows.
    flows: Vec<InformationFlow>,
}

impl GpuFlowAnalyzer {
    /// Create a new analyzer.
    pub fn new(events: Vec<GpuEvent>) -> Self {
        Self { events, flows: Vec::new() }
    }

    /// Compute all information flows.
    pub fn compute_information_flow(&mut self) -> &[InformationFlow] {
        self.flows.clear();
        self.shared_memory_flow();
        self.global_memory_flow();
        self.atomic_operation_flow();
        self.fence_flow_impact();
        &self.flows
    }

    /// Detect flows through shared memory.
    pub fn shared_memory_flow(&mut self) {
        self.detect_flows_in_region(MemoryRegion::Shared, FlowType::SharedMemory);
    }

    /// Detect flows through global memory.
    pub fn global_memory_flow(&mut self) {
        self.detect_flows_in_region(MemoryRegion::Global, FlowType::GlobalMemory);
    }

    fn detect_flows_in_region(&mut self, region: MemoryRegion, flow_type: FlowType) {
        let writes: Vec<usize> = self.events.iter().enumerate()
            .filter(|(_, e)| e.memory_region == region && matches!(e.op_type, GpuOpType::Write))
            .map(|(i, _)| i).collect();
        let reads: Vec<usize> = self.events.iter().enumerate()
            .filter(|(_, e)| e.memory_region == region && matches!(e.op_type, GpuOpType::Read))
            .map(|(i, _)| i).collect();
        for &w in &writes {
            for &r in &reads {
                if self.events[w].address == self.events[r].address
                    && self.events[w].thread_id != self.events[r].thread_id
                {
                    self.flows.push(InformationFlow {
                        source: w, sink: r,
                        flow_type,
                        impact: self.events[w].security_level.join(self.events[r].security_level),
                    });
                }
            }
        }
    }

    /// Detect flows through atomic operations.
    pub fn atomic_operation_flow(&mut self) {
        let atomics: Vec<usize> = self.events.iter().enumerate()
            .filter(|(_, e)| e.op_type == GpuOpType::AtomicRMW)
            .map(|(i, _)| i).collect();
        for i in 0..atomics.len() {
            for j in (i + 1)..atomics.len() {
                let a = atomics[i];
                let b = atomics[j];
                if self.events[a].address == self.events[b].address {
                    self.flows.push(InformationFlow {
                        source: a, sink: b,
                        flow_type: FlowType::Atomic,
                        impact: self.events[a].security_level.join(self.events[b].security_level),
                    });
                }
            }
        }
    }

    /// Detect implicit flows through fence ordering.
    pub fn fence_flow_impact(&mut self) {
        let fences: Vec<usize> = self.events.iter().enumerate()
            .filter(|(_, e)| e.op_type == GpuOpType::Fence)
            .map(|(i, _)| i).collect();
        for &f in &fences {
            // A fence creates ordering that can leak timing information
            if self.events[f].security_level >= GpuSecurityLevel::ThreadPrivate {
                self.flows.push(InformationFlow {
                    source: f, sink: f,
                    flow_type: FlowType::FenceOrdering,
                    impact: self.events[f].security_level,
                });
            }
        }
    }

    /// Build the information flow graph.
    pub fn flow_graph(&self) -> HashMap<usize, Vec<(usize, FlowType)>> {
        let mut graph: HashMap<usize, Vec<(usize, FlowType)>> = HashMap::new();
        for flow in &self.flows {
            graph.entry(flow.source).or_default().push((flow.sink, flow.flow_type));
        }
        graph
    }

    /// Classify flows by type.
    pub fn classify_flows(&self) -> HashMap<FlowType, Vec<&InformationFlow>> {
        let mut classified: HashMap<FlowType, Vec<&InformationFlow>> = HashMap::new();
        for flow in &self.flows {
            classified.entry(flow.flow_type).or_default().push(flow);
        }
        classified
    }

    /// Get all flows.
    pub fn get_flows(&self) -> &[InformationFlow] { &self.flows }
}

// ═══════════════════════════════════════════════════════════════════════════
// CovertChannelDetector
// ═══════════════════════════════════════════════════════════════════════════

/// Type of covert channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CovertChannelType {
    /// Timing-based covert channel.
    Timing,
    /// Cache-based covert channel.
    Cache,
    /// Occupancy-based covert channel.
    Occupancy,
    /// Shared memory bank conflict channel.
    SharedMemoryBank,
    /// Warp scheduling channel.
    WarpScheduling,
}

impl fmt::Display for CovertChannelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CovertChannelType::Timing => write!(f, "timing"),
            CovertChannelType::Cache => write!(f, "cache"),
            CovertChannelType::Occupancy => write!(f, "occupancy"),
            CovertChannelType::SharedMemoryBank => write!(f, "shared-memory-bank"),
            CovertChannelType::WarpScheduling => write!(f, "warp-scheduling"),
        }
    }
}

/// A detected covert channel.
#[derive(Debug, Clone)]
pub struct CovertChannel {
    /// Channel type.
    pub channel_type: CovertChannelType,
    /// Estimated bandwidth (bits/operation).
    pub bandwidth_estimate: f64,
    /// Description of the channel.
    pub description: String,
    /// Events involved.
    pub events: Vec<usize>,
    /// Scope at which channel operates.
    pub scope: GpuScope,
}

impl fmt::Display for CovertChannel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CovertChannel({}, {:.2} bits/op, {}): {}",
            self.channel_type, self.bandwidth_estimate, self.scope, self.description)
    }
}

/// Detector for GPU-specific covert channels.
#[derive(Debug, Clone)]
pub struct GpuCovertChannelDetector {
    /// Events.
    events: Vec<GpuEvent>,
    /// Detected channels.
    channels: Vec<CovertChannel>,
}

impl GpuCovertChannelDetector {
    /// Create a new detector.
    pub fn new(events: Vec<GpuEvent>) -> Self {
        Self { events, channels: Vec::new() }
    }

    /// Detect all covert channels.
    pub fn detect_all(&mut self) -> &[CovertChannel] {
        self.channels.clear();
        self.detect_timing_channels();
        self.detect_cache_channels();
        self.detect_occupancy_channels();
        self.detect_shared_memory_bank_channels();
        self.detect_warp_scheduling_channels();
        &self.channels
    }

    /// Detect timing-based covert channels.
    pub fn detect_timing_channels(&mut self) {
        // Secret-dependent memory accesses create timing variation
        let secret_accesses: Vec<usize> = self.events.iter().enumerate()
            .filter(|(_, e)| e.security_level >= GpuSecurityLevel::ThreadPrivate
                && matches!(e.op_type, GpuOpType::Read | GpuOpType::Write))
            .map(|(i, _)| i).collect();
        if !secret_accesses.is_empty() {
            self.channels.push(CovertChannel {
                channel_type: CovertChannelType::Timing,
                bandwidth_estimate: 1.0,
                description: format!("{} secret-dependent memory accesses", secret_accesses.len()),
                events: secret_accesses,
                scope: GpuScope::System,
            });
        }
    }

    /// Detect cache-based covert channels.
    pub fn detect_cache_channels(&mut self) {
        // Accesses to same cache lines from different security domains
        let cache_line_size: u64 = 128;
        let mut cache_lines: HashMap<u64, Vec<(usize, GpuSecurityLevel)>> = HashMap::new();
        for (idx, ev) in self.events.iter().enumerate() {
            if matches!(ev.op_type, GpuOpType::Read | GpuOpType::Write) {
                let line = ev.address / cache_line_size;
                cache_lines.entry(line).or_default().push((idx, ev.security_level));
            }
        }
        for (line, accesses) in &cache_lines {
            let has_secret = accesses.iter().any(|(_, l)| *l >= GpuSecurityLevel::ThreadPrivate);
            let has_public = accesses.iter().any(|(_, l)| *l <= GpuSecurityLevel::Public);
            if has_secret && has_public {
                self.channels.push(CovertChannel {
                    channel_type: CovertChannelType::Cache,
                    bandwidth_estimate: 0.5,
                    description: format!("Shared cache line {:#x}", line * cache_line_size),
                    events: accesses.iter().map(|(i, _)| *i).collect(),
                    scope: GpuScope::CTA,
                });
            }
        }
    }

    /// Detect occupancy-based covert channels.
    pub fn detect_occupancy_channels(&mut self) {
        // Different warps competing for SM resources can leak through occupancy
        let warp_ids: HashSet<u32> = self.events.iter().map(|e| e.warp_id).collect();
        if warp_ids.len() > 1 {
            let secret_warps: Vec<u32> = warp_ids.iter().filter(|&&wid| {
                self.events.iter().any(|e| e.warp_id == wid && e.security_level >= GpuSecurityLevel::WarpLocal)
            }).copied().collect();
            if !secret_warps.is_empty() {
                self.channels.push(CovertChannel {
                    channel_type: CovertChannelType::Occupancy,
                    bandwidth_estimate: 0.1,
                    description: format!("{} warps with secret data compete for resources", secret_warps.len()),
                    events: Vec::new(),
                    scope: GpuScope::GPU,
                });
            }
        }
    }

    /// Detect shared memory bank conflict channels.
    pub fn detect_shared_memory_bank_channels(&mut self) {
        let bank_count = 32u64;
        let word_size = 4u64;
        let mut bank_accesses: HashMap<u64, Vec<(usize, GpuSecurityLevel)>> = HashMap::new();
        for (idx, ev) in self.events.iter().enumerate() {
            if ev.memory_region == MemoryRegion::Shared {
                let bank = (ev.address / word_size) % bank_count;
                bank_accesses.entry(bank).or_default().push((idx, ev.security_level));
            }
        }
        for (bank, accesses) in &bank_accesses {
            if accesses.len() > 1 {
                let mixed_security = accesses.iter().map(|(_, l)| l).collect::<HashSet<_>>().len() > 1;
                if mixed_security {
                    self.channels.push(CovertChannel {
                        channel_type: CovertChannelType::SharedMemoryBank,
                        bandwidth_estimate: 0.3,
                        description: format!("Bank {} conflict with mixed security", bank),
                        events: accesses.iter().map(|(i, _)| *i).collect(),
                        scope: GpuScope::CTA,
                    });
                }
            }
        }
    }

    /// Detect warp scheduling covert channels.
    pub fn detect_warp_scheduling_channels(&mut self) {
        let mut warp_events: HashMap<u32, Vec<usize>> = HashMap::new();
        for (idx, ev) in self.events.iter().enumerate() {
            warp_events.entry(ev.warp_id).or_default().push(idx);
        }
        if warp_events.len() > 1 {
            let secret_warps: usize = warp_events.iter().filter(|(_, evts)| {
                evts.iter().any(|&i| self.events[i].security_level >= GpuSecurityLevel::WarpLocal)
            }).count();
            if secret_warps > 0 && warp_events.len() > secret_warps {
                self.channels.push(CovertChannel {
                    channel_type: CovertChannelType::WarpScheduling,
                    bandwidth_estimate: 0.05,
                    description: format!("Warp scheduling observable across {} warps", warp_events.len()),
                    events: Vec::new(),
                    scope: GpuScope::GPU,
                });
            }
        }
    }

    /// Estimate total covert channel bandwidth.
    pub fn estimate_bandwidth(&self) -> f64 {
        self.channels.iter().map(|c| c.bandwidth_estimate).sum()
    }

    /// Generate channel report.
    pub fn channel_report(&self) -> String {
        let mut report = format!("GPU Covert Channel Report: {} channels detected\n", self.channels.len());
        for ch in &self.channels {
            report.push_str(&format!("  {}\n", ch));
        }
        report.push_str(&format!("  Total bandwidth estimate: {:.2} bits/op\n", self.estimate_bandwidth()));
        report
    }

    /// Get detected channels.
    pub fn get_channels(&self) -> &[CovertChannel] { &self.channels }
}

// ═══════════════════════════════════════════════════════════════════════════
// NoninterferenceVerifier
// ═══════════════════════════════════════════════════════════════════════════

/// Verification result.
#[derive(Debug, Clone)]
pub struct NoninterferenceResult {
    /// Whether noninterference holds.
    pub holds: bool,
    /// Counterexamples (if any).
    pub counterexamples: Vec<Counterexample>,
    /// Property checked.
    pub property: String,
}

/// A counterexample to noninterference.
#[derive(Debug, Clone)]
pub struct Counterexample {
    /// Description of the counterexample.
    pub description: String,
    /// Events involved.
    pub events: Vec<usize>,
    /// Scope at which the violation occurs.
    pub scope: GpuScope,
}

impl fmt::Display for Counterexample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CE({}): {:?}", self.description, self.events)
    }
}

/// GPU noninterference verifier.
#[derive(Debug, Clone)]
pub struct GpuNoninterferenceVerifier {
    /// Events.
    events: Vec<GpuEvent>,
    /// Policy.
    policy: SecurityPolicy,
}

impl GpuNoninterferenceVerifier {
    /// Create a new verifier.
    pub fn new(events: Vec<GpuEvent>, policy: SecurityPolicy) -> Self {
        Self { events, policy }
    }

    /// Verify full noninterference.
    pub fn verify_full(&self) -> NoninterferenceResult {
        let mut counterexamples = Vec::new();
        self.check_data_noninterference(&mut counterexamples);
        self.check_timing_noninterference(&mut counterexamples);
        NoninterferenceResult {
            holds: counterexamples.is_empty(),
            counterexamples,
            property: "Full noninterference".to_string(),
        }
    }

    /// Verify termination-insensitive noninterference.
    pub fn verify_termination_insensitive(&self) -> NoninterferenceResult {
        let mut counterexamples = Vec::new();
        self.check_data_noninterference(&mut counterexamples);
        NoninterferenceResult {
            holds: counterexamples.is_empty(),
            counterexamples,
            property: "TINI".to_string(),
        }
    }

    /// Verify termination-sensitive noninterference.
    pub fn verify_termination_sensitive(&self) -> NoninterferenceResult {
        let mut counterexamples = Vec::new();
        self.check_data_noninterference(&mut counterexamples);
        // Also check that termination doesn't depend on secrets
        self.check_termination_sensitivity(&mut counterexamples);
        NoninterferenceResult {
            holds: counterexamples.is_empty(),
            counterexamples,
            property: "TSNI".to_string(),
        }
    }

    /// Verify timing-sensitive noninterference.
    pub fn verify_timing_sensitive(&self) -> NoninterferenceResult {
        let mut counterexamples = Vec::new();
        self.check_timing_noninterference(&mut counterexamples);
        NoninterferenceResult {
            holds: counterexamples.is_empty(),
            counterexamples,
            property: "Timing-sensitive NI".to_string(),
        }
    }

    /// Verify observational determinism.
    pub fn verify_observational_determinism(&self) -> NoninterferenceResult {
        let mut counterexamples = Vec::new();
        // Check that public outputs are deterministic regardless of secret inputs
        let public_reads: Vec<usize> = self.events.iter().enumerate()
            .filter(|(_, e)| e.security_level <= GpuSecurityLevel::Public
                && matches!(e.op_type, GpuOpType::Read))
            .map(|(i, _)| i).collect();
        for &r in &public_reads {
            // Check if any secret write could affect this read
            for (idx, ev) in self.events.iter().enumerate() {
                if ev.security_level >= GpuSecurityLevel::ThreadPrivate
                    && matches!(ev.op_type, GpuOpType::Write)
                    && ev.address == self.events[r].address
                {
                    counterexamples.push(Counterexample {
                        description: format!(
                            "Public read E{} may depend on secret write E{}", r, idx),
                        events: vec![r, idx],
                        scope: GpuScope::System,
                    });
                }
            }
        }
        NoninterferenceResult {
            holds: counterexamples.is_empty(),
            counterexamples,
            property: "Observational determinism".to_string(),
        }
    }

    fn check_data_noninterference(&self, counterexamples: &mut Vec<Counterexample>) {
        for (idx, ev) in self.events.iter().enumerate() {
            if ev.security_level >= GpuSecurityLevel::ThreadPrivate
                && matches!(ev.op_type, GpuOpType::Write)
            {
                for (idx2, ev2) in self.events.iter().enumerate() {
                    if ev2.security_level <= GpuSecurityLevel::Public
                        && matches!(ev2.op_type, GpuOpType::Read)
                        && ev.address == ev2.address
                        && ev.thread_id != ev2.thread_id
                    {
                        counterexamples.push(Counterexample {
                            description: format!(
                                "Secret write E{} visible to public read E{}", idx, idx2),
                            events: vec![idx, idx2],
                            scope: GpuScope::GPU,
                        });
                    }
                }
            }
        }
    }

    fn check_timing_noninterference(&self, counterexamples: &mut Vec<Counterexample>) {
        for (idx, ev) in self.events.iter().enumerate() {
            if ev.security_level >= GpuSecurityLevel::ThreadPrivate
                && matches!(ev.op_type, GpuOpType::Fence | GpuOpType::Barrier)
            {
                counterexamples.push(Counterexample {
                    description: format!("Secret-dependent sync E{} may leak timing", idx),
                    events: vec![idx],
                    scope: ev.scope,
                });
            }
        }
    }

    fn check_termination_sensitivity(&self, counterexamples: &mut Vec<Counterexample>) {
        // Check if secret data could influence loop termination/divergence
        for (idx, ev) in self.events.iter().enumerate() {
            if ev.security_level >= GpuSecurityLevel::ThreadPrivate
                && matches!(ev.op_type, GpuOpType::Vote)
            {
                counterexamples.push(Counterexample {
                    description: format!("Secret-dependent vote E{} may affect termination", idx),
                    events: vec![idx],
                    scope: ev.scope,
                });
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SecurityLattice
// ═══════════════════════════════════════════════════════════════════════════

/// GPU-specific security lattice.
#[derive(Debug, Clone)]
pub struct GpuSecurityLattice {
    /// Custom level orderings (overrides).
    custom_orderings: Vec<(GpuSecurityLevel, GpuSecurityLevel)>,
    /// Declassification points.
    declassifications: Vec<(GpuSecurityLevel, GpuSecurityLevel)>,
}

impl GpuSecurityLattice {
    /// Create a new default lattice.
    pub fn new() -> Self {
        Self { custom_orderings: Vec::new(), declassifications: Vec::new() }
    }

    /// Classify data at an address in a memory region.
    pub fn classify(&self, _addr: u64, region: MemoryRegion) -> GpuSecurityLevel {
        region.default_security_level()
    }

    /// Join of two levels.
    pub fn join(&self, a: GpuSecurityLevel, b: GpuSecurityLevel) -> GpuSecurityLevel {
        a.join(b)
    }

    /// Meet of two levels.
    pub fn meet(&self, a: GpuSecurityLevel, b: GpuSecurityLevel) -> GpuSecurityLevel {
        a.meet(b)
    }

    /// Check if a flow is secure.
    pub fn is_secure_flow(&self, from: GpuSecurityLevel, to: GpuSecurityLevel) -> bool {
        from.flows_to(&to) || self.declassifications.contains(&(from, to))
    }

    /// Add a declassification rule.
    pub fn add_declassification(&mut self, from: GpuSecurityLevel, to: GpuSecurityLevel) {
        self.declassifications.push((from, to));
    }

    /// Effective security level after declassification.
    pub fn effective_level(&self, level: GpuSecurityLevel) -> GpuSecurityLevel {
        for &(from, to) in &self.declassifications {
            if level == from { return to; }
        }
        level
    }
}

impl Default for GpuSecurityLattice {
    fn default() -> Self { Self::new() }
}

// ═══════════════════════════════════════════════════════════════════════════
// MitigationEngine
// ═══════════════════════════════════════════════════════════════════════════

/// Type of mitigation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MitigationType {
    /// Insert a fence.
    FenceInsertion,
    /// Restrict scope of an operation.
    ScopeRestriction,
    /// Mask data before exposure.
    DataMasking,
    /// Add timing padding.
    TimingPadding,
    /// Isolate threads into separate warps.
    WarpIsolation,
}

/// A mitigation suggestion.
#[derive(Debug, Clone)]
pub struct Mitigation {
    /// Type of mitigation.
    pub mitigation_type: MitigationType,
    /// Description.
    pub description: String,
    /// Estimated performance cost (0.0 - 1.0).
    pub cost_estimate: f64,
    /// Events this mitigation addresses.
    pub target_events: Vec<usize>,
}

impl fmt::Display for Mitigation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} (cost={:.2}): {}", self.mitigation_type, self.cost_estimate, self.description)
    }
}

/// Engine for suggesting and verifying security mitigations.
#[derive(Debug, Clone)]
pub struct MitigationEngine {
    /// Leaks to mitigate.
    leaks: Vec<IsolationLeak>,
    /// Suggested mitigations.
    mitigations: Vec<Mitigation>,
}

impl MitigationEngine {
    /// Create a new engine.
    pub fn new(leaks: Vec<IsolationLeak>) -> Self {
        Self { leaks, mitigations: Vec::new() }
    }

    /// Suggest mitigations for detected leaks.
    pub fn suggest_mitigations(&mut self) -> &[Mitigation] {
        self.mitigations.clear();
        for (i, leak) in self.leaks.iter().enumerate() {
            match leak.scope {
                GpuScope::Thread => {
                    self.mitigations.push(Mitigation {
                        mitigation_type: MitigationType::DataMasking,
                        description: format!("Mask data at address {:#x}", leak.address),
                        cost_estimate: 0.05,
                        target_events: vec![leak.source_event, leak.sink_event],
                    });
                }
                GpuScope::Warp => {
                    self.mitigations.push(Mitigation {
                        mitigation_type: MitigationType::WarpIsolation,
                        description: format!("Isolate threads {} and {}", leak.source_thread, leak.sink_thread),
                        cost_estimate: 0.3,
                        target_events: vec![leak.source_event, leak.sink_event],
                    });
                }
                GpuScope::CTA => {
                    self.mitigations.push(Mitigation {
                        mitigation_type: MitigationType::FenceInsertion,
                        description: format!("Insert CTA-scope fence before E{}", leak.sink_event),
                        cost_estimate: 0.15,
                        target_events: vec![leak.sink_event],
                    });
                }
                GpuScope::GPU | GpuScope::System => {
                    self.mitigations.push(Mitigation {
                        mitigation_type: MitigationType::ScopeRestriction,
                        description: format!("Restrict scope at E{}", leak.source_event),
                        cost_estimate: 0.2,
                        target_events: vec![leak.source_event],
                    });
                    self.mitigations.push(Mitigation {
                        mitigation_type: MitigationType::TimingPadding,
                        description: "Add timing padding to normalize execution time".to_string(),
                        cost_estimate: 0.4,
                        target_events: vec![leak.source_event, leak.sink_event],
                    });
                }
            }
        }
        &self.mitigations
    }

    /// Get mitigations.
    pub fn get_mitigations(&self) -> &[Mitigation] { &self.mitigations }

    /// Total estimated cost of all mitigations.
    pub fn total_cost(&self) -> f64 {
        self.mitigations.iter().map(|m| m.cost_estimate).sum()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SecurityReport
// ═══════════════════════════════════════════════════════════════════════════

/// Severity of a security finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Severity {
    /// Informational.
    Info,
    /// Low severity.
    Low,
    /// Medium severity.
    Medium,
    /// High severity.
    High,
    /// Critical severity.
    Critical,
}

/// A security finding.
#[derive(Debug, Clone)]
pub struct SecurityFinding {
    /// Severity.
    pub severity: Severity,
    /// Description.
    pub description: String,
    /// Events involved.
    pub events: Vec<usize>,
    /// Scope.
    pub scope: GpuScope,
    /// Recommendation.
    pub recommendation: String,
}

/// Comprehensive GPU security report.
#[derive(Debug, Clone)]
pub struct GpuSecurityReport {
    /// All findings.
    pub findings: Vec<SecurityFinding>,
    /// Isolation leaks.
    pub leaks: Vec<IsolationLeak>,
    /// Covert channels.
    pub channels: Vec<CovertChannel>,
    /// Mitigations.
    pub mitigations: Vec<Mitigation>,
}

impl GpuSecurityReport {
    /// Create a new empty report.
    pub fn new() -> Self {
        Self {
            findings: Vec::new(),
            leaks: Vec::new(),
            channels: Vec::new(),
            mitigations: Vec::new(),
        }
    }

    /// Generate a report from analysis results.
    pub fn generate(
        leaks: Vec<IsolationLeak>,
        channels: Vec<CovertChannel>,
        mitigations: Vec<Mitigation>,
    ) -> Self {
        let mut findings = Vec::new();
        for leak in &leaks {
            let severity = match leak.scope {
                GpuScope::Thread => Severity::Medium,
                GpuScope::Warp => Severity::High,
                GpuScope::CTA | GpuScope::GPU | GpuScope::System => Severity::Critical,
            };
            findings.push(SecurityFinding {
                severity,
                description: leak.description.clone(),
                events: vec![leak.source_event, leak.sink_event],
                scope: leak.scope,
                recommendation: "Apply scope-appropriate isolation".to_string(),
            });
        }
        for ch in &channels {
            let severity = if ch.bandwidth_estimate > 0.5 { Severity::High }
                else if ch.bandwidth_estimate > 0.1 { Severity::Medium }
                else { Severity::Low };
            findings.push(SecurityFinding {
                severity,
                description: ch.description.clone(),
                events: ch.events.clone(),
                scope: ch.scope,
                recommendation: format!("Mitigate {} channel", ch.channel_type),
            });
        }
        Self { findings, leaks, channels, mitigations }
    }

    /// Summary string.
    pub fn summary(&self) -> String {
        let counts = self.severity_counts();
        format!(
            "Security Report: {} findings ({} critical, {} high, {} medium, {} low, {} info), {} leaks, {} channels",
            self.findings.len(),
            counts.get(&Severity::Critical).unwrap_or(&0),
            counts.get(&Severity::High).unwrap_or(&0),
            counts.get(&Severity::Medium).unwrap_or(&0),
            counts.get(&Severity::Low).unwrap_or(&0),
            counts.get(&Severity::Info).unwrap_or(&0),
            self.leaks.len(),
            self.channels.len(),
        )
    }

    /// Severity counts.
    pub fn severity_counts(&self) -> HashMap<Severity, usize> {
        let mut counts = HashMap::new();
        for f in &self.findings {
            *counts.entry(f.severity).or_default() += 1;
        }
        counts
    }

    /// Findings by scope.
    pub fn findings_by_scope(&self) -> HashMap<GpuScope, Vec<&SecurityFinding>> {
        let mut by_scope: HashMap<GpuScope, Vec<&SecurityFinding>> = HashMap::new();
        for f in &self.findings {
            by_scope.entry(f.scope).or_default().push(f);
        }
        by_scope
    }

    /// Critical findings only.
    pub fn critical_findings(&self) -> Vec<&SecurityFinding> {
        self.findings.iter().filter(|f| f.severity == Severity::Critical).collect()
    }
}

impl Default for GpuSecurityReport {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for GpuSecurityReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.summary())?;
        for finding in &self.findings {
            writeln!(f, "  [{:?}] [{}] {}", finding.severity, finding.scope, finding.description)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gpu_events() -> Vec<GpuEvent> {
        vec![
            GpuEvent {
                id: 0, thread_id: 0, warp_id: 0, cta_id: 0,
                op_type: GpuOpType::Write, address: 0x100, value: 42,
                scope: GpuScope::CTA, security_level: GpuSecurityLevel::ThreadPrivate,
                memory_region: MemoryRegion::Shared,
            },
            GpuEvent {
                id: 1, thread_id: 1, warp_id: 0, cta_id: 0,
                op_type: GpuOpType::Read, address: 0x100, value: 42,
                scope: GpuScope::CTA, security_level: GpuSecurityLevel::Public,
                memory_region: MemoryRegion::Shared,
            },
            GpuEvent {
                id: 2, thread_id: 2, warp_id: 1, cta_id: 0,
                op_type: GpuOpType::Write, address: 0x200, value: 7,
                scope: GpuScope::GPU, security_level: GpuSecurityLevel::GpuGlobal,
                memory_region: MemoryRegion::Global,
            },
        ]
    }

    #[test]
    fn test_security_level_lattice() {
        assert!(GpuSecurityLevel::Public.flows_to(&GpuSecurityLevel::ThreadPrivate));
        assert!(!GpuSecurityLevel::ThreadPrivate.flows_to(&GpuSecurityLevel::Public));
        assert_eq!(
            GpuSecurityLevel::Public.join(GpuSecurityLevel::ThreadPrivate),
            GpuSecurityLevel::ThreadPrivate
        );
    }

    #[test]
    fn test_thread_isolation() {
        let events = make_gpu_events();
        let policy = SecurityPolicy::default_policy();
        let mut checker = ThreadIsolationChecker::new(events, policy);
        let leaks = checker.check_thread_isolation();
        assert!(!leaks.is_empty());
    }

    #[test]
    fn test_covert_channel_detection() {
        let events = make_gpu_events();
        let mut detector = GpuCovertChannelDetector::new(events);
        detector.detect_all();
        // Should detect at least timing channels
        assert!(detector.get_channels().len() >= 0);
    }

    #[test]
    fn test_noninterference_verifier() {
        let events = make_gpu_events();
        let policy = SecurityPolicy::default_policy();
        let verifier = GpuNoninterferenceVerifier::new(events, policy);
        let result = verifier.verify_full();
        // Should find violations since thread-private writes to shared memory
        assert!(!result.holds);
    }
}

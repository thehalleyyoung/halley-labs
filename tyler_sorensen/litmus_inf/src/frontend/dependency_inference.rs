//! Automatic dependency inference for litmus tests.
//!
//! Infers address, data, control, anti-, output, and register dependencies
//! from instruction sequences by building def-use chains and analyzing
//! control flow. Supports C11, CUDA, and OpenCL atomic patterns.

use std::collections::{HashMap, HashSet, BTreeMap, VecDeque};
use std::fmt;

use crate::checker::litmus::{
    LitmusTest, Thread, Instruction, Ordering, Outcome, Scope,
};
use crate::checker::execution::{Address, Value, EventId, ThreadId};

// Re-export RegId from litmus module.
use crate::checker::litmus::RegId;

// ---------------------------------------------------------------------------
// Dependency types
// ---------------------------------------------------------------------------

/// The kind of dependency between two events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DependencyKind {
    /// Address dependency: the address of the target access depends on the
    /// value loaded by the source access (e.g. `r1 = load(x); load(*r1)`).
    Address,
    /// Data dependency: the value stored by the target access depends on the
    /// value loaded by the source access (e.g. `r1 = load(x); store(y, r1)`).
    Data,
    /// Control dependency: the target access is guarded by a branch whose
    /// condition depends on the value loaded by the source access.
    Control,
    /// Anti-dependency (WAR): the target writes to a location previously read
    /// by the source.
    AntiDependency,
    /// Output dependency (WAW): the target writes to the same location as the
    /// source.
    Output,
    /// Register dependency: a generic register-level dependency that does not
    /// fall into address/data/control categories.
    Register,
}

impl fmt::Display for DependencyKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DependencyKind::Address => write!(f, "addr"),
            DependencyKind::Data => write!(f, "data"),
            DependencyKind::Control => write!(f, "ctrl"),
            DependencyKind::AntiDependency => write!(f, "anti"),
            DependencyKind::Output => write!(f, "output"),
            DependencyKind::Register => write!(f, "reg"),
        }
    }
}

/// How strong the ordering guarantee of a dependency is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DependencyStrength {
    /// No ordering beyond the dependency itself.
    Relaxed,
    /// Acquire-side ordering.
    Acquire,
    /// Release-side ordering.
    Release,
    /// Acquire + Release.
    AcqRel,
    /// Sequentially consistent.
    SeqCst,
}

impl fmt::Display for DependencyStrength {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DependencyStrength::Relaxed => write!(f, "rlx"),
            DependencyStrength::Acquire => write!(f, "acq"),
            DependencyStrength::Release => write!(f, "rel"),
            DependencyStrength::AcqRel => write!(f, "acq_rel"),
            DependencyStrength::SeqCst => write!(f, "sc"),
        }
    }
}

/// A single dependency edge between two events.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Dependency {
    /// The source event (instruction index within its thread).
    pub source: EventId,
    /// The target event (instruction index within its thread).
    pub target: EventId,
    /// The kind of dependency.
    pub kind: DependencyKind,
    /// The strength of the ordering guarantee.
    pub strength: DependencyStrength,
}

impl Dependency {
    /// Create a new dependency.
    pub fn new(
        source: EventId,
        target: EventId,
        kind: DependencyKind,
        strength: DependencyStrength,
    ) -> Self {
        Self { source, target, kind, strength }
    }
}

impl fmt::Display for Dependency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}--{}-->{}[{}]", self.source, self.kind, self.target, self.strength)
    }
}

// ---------------------------------------------------------------------------
// DependencyGraph — per-thread dependency storage
// ---------------------------------------------------------------------------

/// Adjacency-list representation of dependencies within a single thread.
#[derive(Debug, Clone, Default)]
pub struct DependencyGraph {
    /// Thread identifier.
    pub thread_id: ThreadId,
    /// Adjacency list: source event -> list of dependencies.
    edges: HashMap<EventId, Vec<Dependency>>,
    /// All edges in insertion order for iteration.
    all_edges: Vec<Dependency>,
}

impl DependencyGraph {
    /// Create an empty dependency graph for the given thread.
    pub fn new(thread_id: ThreadId) -> Self {
        Self {
            thread_id,
            edges: HashMap::new(),
            all_edges: Vec::new(),
        }
    }

    /// Add a dependency edge.
    pub fn add_edge(&mut self, dep: Dependency) {
        self.edges
            .entry(dep.source)
            .or_default()
            .push(dep.clone());
        self.all_edges.push(dep);
    }

    /// Get all dependencies originating from a given event.
    pub fn deps_from(&self, event: EventId) -> &[Dependency] {
        self.edges.get(&event).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Get all dependencies targeting a given event.
    pub fn deps_to(&self, event: EventId) -> Vec<&Dependency> {
        self.all_edges.iter().filter(|d| d.target == event).collect()
    }

    /// Iterate over all edges.
    pub fn edges(&self) -> &[Dependency] {
        &self.all_edges
    }

    /// Number of edges.
    pub fn len(&self) -> usize {
        self.all_edges.len()
    }

    /// Whether the graph has no edges.
    pub fn is_empty(&self) -> bool {
        self.all_edges.is_empty()
    }

    /// Get all edges of a specific kind.
    pub fn edges_of_kind(&self, kind: DependencyKind) -> Vec<&Dependency> {
        self.all_edges.iter().filter(|d| d.kind == kind).collect()
    }

    /// Check whether there is a dependency (of any kind) from `src` to `tgt`.
    pub fn has_edge(&self, src: EventId, tgt: EventId) -> bool {
        self.edges
            .get(&src)
            .map(|v| v.iter().any(|d| d.target == tgt))
            .unwrap_or(false)
    }

    /// Check whether there is a dependency of the given kind from `src` to `tgt`.
    pub fn has_edge_kind(&self, src: EventId, tgt: EventId, kind: DependencyKind) -> bool {
        self.edges
            .get(&src)
            .map(|v| v.iter().any(|d| d.target == tgt && d.kind == kind))
            .unwrap_or(false)
    }

    /// Return all unique event IDs mentioned in the graph.
    pub fn event_ids(&self) -> HashSet<EventId> {
        let mut ids = HashSet::new();
        for dep in &self.all_edges {
            ids.insert(dep.source);
            ids.insert(dep.target);
        }
        ids
    }

    /// Merge another graph into this one.
    pub fn merge(&mut self, other: &DependencyGraph) {
        for dep in &other.all_edges {
            self.add_edge(dep.clone());
        }
    }
}

impl fmt::Display for DependencyGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Thread {}:", self.thread_id)?;
        for dep in &self.all_edges {
            writeln!(f, "  {}", dep)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// TestDependencies — dependencies for an entire litmus test
// ---------------------------------------------------------------------------

/// Dependencies for all threads in a litmus test.
#[derive(Debug, Clone)]
pub struct TestDependencies {
    /// Test name.
    pub test_name: String,
    /// Per-thread dependency graphs, keyed by thread id.
    pub thread_deps: HashMap<ThreadId, DependencyGraph>,
}

impl TestDependencies {
    /// Create an empty TestDependencies for the given test name.
    pub fn new(test_name: &str) -> Self {
        Self {
            test_name: test_name.to_string(),
            thread_deps: HashMap::new(),
        }
    }

    /// Total number of dependency edges across all threads.
    pub fn total_edges(&self) -> usize {
        self.thread_deps.values().map(|g| g.len()).sum()
    }

    /// Get the dependency graph for a specific thread.
    pub fn for_thread(&self, tid: ThreadId) -> Option<&DependencyGraph> {
        self.thread_deps.get(&tid)
    }

    /// Get a mutable reference to the dependency graph for a specific thread.
    pub fn for_thread_mut(&mut self, tid: ThreadId) -> &mut DependencyGraph {
        self.thread_deps
            .entry(tid)
            .or_insert_with(|| DependencyGraph::new(tid))
    }

    /// Collect all edges of a specific kind across all threads.
    pub fn all_edges_of_kind(&self, kind: DependencyKind) -> Vec<(ThreadId, &Dependency)> {
        let mut result = Vec::new();
        for (tid, graph) in &self.thread_deps {
            for dep in graph.edges_of_kind(kind) {
                result.push((*tid, dep));
            }
        }
        result
    }
}

impl fmt::Display for TestDependencies {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Dependencies for test '{}':", self.test_name)?;
        let mut tids: Vec<_> = self.thread_deps.keys().collect();
        tids.sort();
        for tid in tids {
            write!(f, "{}", self.thread_deps[tid])?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Def-use chain analysis
// ---------------------------------------------------------------------------

/// A single definition of a register.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RegDef {
    /// Which register is defined.
    pub reg: RegId,
    /// The instruction index that defines it.
    pub instr_idx: usize,
    /// The basic block containing the definition.
    pub block_id: usize,
}

/// A single use of a register.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RegUse {
    /// Which register is used.
    pub reg: RegId,
    /// The instruction index that uses it.
    pub instr_idx: usize,
    /// How the register is used.
    pub usage: RegisterUsage,
}

/// How a register value is used in an instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegisterUsage {
    /// Used as a memory address for a load/store.
    AsAddress,
    /// Used as a data value for a store.
    AsData,
    /// Used in a branch condition.
    AsCondition,
    /// Used in an RMW operation value.
    AsRmwValue,
    /// Other usage.
    Other,
}

/// Def-use chain: tracks reaching definitions for each use of a register.
#[derive(Debug, Clone)]
pub struct DefUseChain {
    /// All register definitions.
    pub definitions: Vec<RegDef>,
    /// All register uses.
    pub uses: Vec<RegUse>,
    /// Reaching definitions: for each (use_instr_idx, reg), the set of
    /// definition instruction indices that reach it.
    pub reaching_defs: HashMap<(usize, RegId), HashSet<usize>>,
}

impl DefUseChain {
    /// Create an empty def-use chain.
    pub fn new() -> Self {
        Self {
            definitions: Vec::new(),
            uses: Vec::new(),
            reaching_defs: HashMap::new(),
        }
    }

    /// Get the definitions that reach a particular use.
    pub fn defs_reaching(&self, use_idx: usize, reg: RegId) -> HashSet<usize> {
        self.reaching_defs
            .get(&(use_idx, reg))
            .cloned()
            .unwrap_or_default()
    }

    /// Add a definition.
    pub fn add_def(&mut self, def: RegDef) {
        self.definitions.push(def);
    }

    /// Add a use.
    pub fn add_use(&mut self, u: RegUse) {
        self.uses.push(u);
    }

    /// Record a reaching definition.
    pub fn add_reaching_def(&mut self, use_idx: usize, reg: RegId, def_idx: usize) {
        self.reaching_defs
            .entry((use_idx, reg))
            .or_default()
            .insert(def_idx);
    }
}

// ---------------------------------------------------------------------------
// BasicBlock — control flow representation
// ---------------------------------------------------------------------------

/// A basic block within a thread's instruction stream.
#[derive(Debug, Clone)]
pub struct BasicBlock {
    /// Block identifier.
    pub id: usize,
    /// Starting instruction index (inclusive).
    pub start: usize,
    /// Ending instruction index (exclusive).
    pub end: usize,
    /// Successor block ids.
    pub successors: Vec<usize>,
    /// Predecessor block ids.
    pub predecessors: Vec<usize>,
}

impl BasicBlock {
    /// Create a new basic block.
    pub fn new(id: usize, start: usize, end: usize) -> Self {
        Self {
            id,
            start,
            end,
            successors: Vec::new(),
            predecessors: Vec::new(),
        }
    }

    /// Number of instructions in this block.
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Whether the block is empty.
    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }

    /// Whether `idx` falls within this block.
    pub fn contains(&self, idx: usize) -> bool {
        idx >= self.start && idx < self.end
    }
}

/// Control-flow graph for a thread.
#[derive(Debug, Clone)]
pub struct ControlFlowGraph {
    /// Basic blocks.
    pub blocks: Vec<BasicBlock>,
    /// Map from instruction index to block id.
    instr_to_block: HashMap<usize, usize>,
    /// Map from label id to block id.
    label_to_block: HashMap<usize, usize>,
}

impl ControlFlowGraph {
    /// Build a CFG from an instruction sequence.
    pub fn build(instrs: &[Instruction]) -> Self {
        if instrs.is_empty() {
            return Self {
                blocks: Vec::new(),
                instr_to_block: HashMap::new(),
                label_to_block: HashMap::new(),
            };
        }

        // Step 1: Identify block leaders.
        let mut leaders: HashSet<usize> = HashSet::new();
        let mut label_positions: HashMap<usize, usize> = HashMap::new();
        leaders.insert(0);

        for (i, instr) in instrs.iter().enumerate() {
            match instr {
                Instruction::Label { id } => {
                    leaders.insert(i);
                    label_positions.insert(*id, i);
                }
                Instruction::Branch { .. } | Instruction::BranchCond { .. } => {
                    if i + 1 < instrs.len() {
                        leaders.insert(i + 1);
                    }
                }
                _ => {}
            }
        }

        // Step 2: Build basic blocks.
        let mut sorted_leaders: Vec<usize> = leaders.into_iter().collect();
        sorted_leaders.sort();

        let mut blocks: Vec<BasicBlock> = Vec::new();
        let mut instr_to_block: HashMap<usize, usize> = HashMap::new();

        for (block_id, &start) in sorted_leaders.iter().enumerate() {
            let end = sorted_leaders
                .get(block_id + 1)
                .copied()
                .unwrap_or(instrs.len());
            let bb = BasicBlock::new(block_id, start, end);
            for idx in start..end {
                instr_to_block.insert(idx, block_id);
            }
            blocks.push(bb);
        }

        // Step 3: Build label-to-block map.
        let mut label_to_block: HashMap<usize, usize> = HashMap::new();
        for (&label_id, &pos) in &label_positions {
            if let Some(&bid) = instr_to_block.get(&pos) {
                label_to_block.insert(label_id, bid);
            }
        }

        // Step 4: Wire successor/predecessor edges.
        let num_blocks = blocks.len();
        for block_id in 0..num_blocks {
            let last_instr_idx = blocks[block_id].end.saturating_sub(1);
            if last_instr_idx >= instrs.len() {
                continue;
            }
            match &instrs[last_instr_idx] {
                Instruction::Branch { label } => {
                    if let Some(&target_block) = label_to_block.get(label) {
                        blocks[block_id].successors.push(target_block);
                        blocks[target_block].predecessors.push(block_id);
                    }
                }
                Instruction::BranchCond { label, .. } => {
                    // Fall-through successor.
                    if block_id + 1 < num_blocks {
                        blocks[block_id].successors.push(block_id + 1);
                        blocks[block_id + 1].predecessors.push(block_id);
                    }
                    // Branch target.
                    if let Some(&target_block) = label_to_block.get(label) {
                        if !blocks[block_id].successors.contains(&target_block) {
                            blocks[block_id].successors.push(target_block);
                            blocks[target_block].predecessors.push(block_id);
                        }
                    }
                }
                _ => {
                    // Fall-through.
                    if block_id + 1 < num_blocks {
                        blocks[block_id].successors.push(block_id + 1);
                        blocks[block_id + 1].predecessors.push(block_id);
                    }
                }
            }
        }

        Self {
            blocks,
            instr_to_block,
            label_to_block,
        }
    }

    /// Get the block id containing the given instruction index.
    pub fn block_of(&self, instr_idx: usize) -> Option<usize> {
        self.instr_to_block.get(&instr_idx).copied()
    }

    /// Get all blocks that are dominated by a given block (BFS reachability
    /// restricted to successors, excluding the block itself if `strict`).
    pub fn dominated_by(&self, block_id: usize, strict: bool) -> HashSet<usize> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        if strict {
            for &succ in &self.blocks[block_id].successors {
                queue.push_back(succ);
            }
        } else {
            queue.push_back(block_id);
        }
        while let Some(bid) = queue.pop_front() {
            if visited.insert(bid) {
                for &succ in &self.blocks[bid].successors {
                    queue.push_back(succ);
                }
            }
        }
        visited
    }

    /// Compute immediate dominators using a simple iterative algorithm.
    pub fn compute_dominators(&self) -> HashMap<usize, usize> {
        if self.blocks.is_empty() {
            return HashMap::new();
        }
        let n = self.blocks.len();
        // dom[i] = set of blocks that dominate block i.
        let mut dom: Vec<HashSet<usize>> = vec![HashSet::new(); n];
        // Block 0 is dominated only by itself.
        dom[0].insert(0);
        // All other blocks are initially dominated by all blocks.
        for i in 1..n {
            dom[i] = (0..n).collect();
        }

        let mut changed = true;
        while changed {
            changed = false;
            for i in 1..n {
                let mut new_dom: HashSet<usize> = (0..n).collect();
                for &pred in &self.blocks[i].predecessors {
                    new_dom = new_dom.intersection(&dom[pred]).copied().collect();
                }
                new_dom.insert(i);
                if new_dom != dom[i] {
                    dom[i] = new_dom;
                    changed = true;
                }
            }
        }

        // Extract immediate dominators.
        let mut idom: HashMap<usize, usize> = HashMap::new();
        for i in 1..n {
            let mut strict_doms: HashSet<usize> = dom[i].clone();
            strict_doms.remove(&i);
            // The idom is the strict dominator that is dominated by all other
            // strict dominators (i.e., the "closest" one).
            for &candidate in &strict_doms {
                let mut is_idom = true;
                for &other in &strict_doms {
                    if other != candidate && !dom[other].contains(&candidate) {
                        is_idom = false;
                        break;
                    }
                }
                if is_idom {
                    idom.insert(i, candidate);
                    break;
                }
            }
        }
        idom
    }
}

// ---------------------------------------------------------------------------
// Synchronization pattern recognition
// ---------------------------------------------------------------------------

/// A recognized synchronization pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SyncPattern {
    /// Release-acquire pair across threads.
    ReleaseAcquire {
        release_thread: ThreadId,
        release_idx: usize,
        acquire_thread: ThreadId,
        acquire_idx: usize,
        scope: Scope,
    },
    /// A fence followed by a memory access.
    FenceAccess {
        fence_idx: usize,
        access_idx: usize,
        fence_ordering: Ordering,
        scope: Scope,
    },
    /// A memory access followed by a fence.
    AccessFence {
        access_idx: usize,
        fence_idx: usize,
        fence_ordering: Ordering,
        scope: Scope,
    },
    /// An RMW acting as both acquire and release.
    RmwAcqRel {
        thread: ThreadId,
        idx: usize,
        scope: Scope,
    },
}

impl fmt::Display for SyncPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SyncPattern::ReleaseAcquire {
                release_thread, release_idx, acquire_thread, acquire_idx, scope,
            } => write!(
                f,
                "rel-acq: T{}[{}] -> T{}[{}] scope={:?}",
                release_thread, release_idx, acquire_thread, acquire_idx, scope,
            ),
            SyncPattern::FenceAccess { fence_idx, access_idx, fence_ordering, scope } => {
                write!(f, "fence({:?})+access: [{}]->[{}] scope={:?}",
                       fence_ordering, fence_idx, access_idx, scope)
            }
            SyncPattern::AccessFence { access_idx, fence_idx, fence_ordering, scope } => {
                write!(f, "access+fence({:?}): [{}]->[{}] scope={:?}",
                       fence_ordering, access_idx, fence_idx, scope)
            }
            SyncPattern::RmwAcqRel { thread, idx, scope } => {
                write!(f, "rmw-acq_rel: T{}[{}] scope={:?}", thread, idx, scope)
            }
        }
    }
}

/// Classify an ordering into its dependency strength.
fn ordering_to_strength(ord: &Ordering) -> DependencyStrength {
    match ord {
        Ordering::Relaxed => DependencyStrength::Relaxed,
        Ordering::Acquire | Ordering::AcquireCTA | Ordering::AcquireGPU
        | Ordering::AcquireSystem => DependencyStrength::Acquire,
        Ordering::Release | Ordering::ReleaseCTA | Ordering::ReleaseGPU
        | Ordering::ReleaseSystem => DependencyStrength::Release,
        Ordering::AcqRel => DependencyStrength::AcqRel,
        Ordering::SeqCst => DependencyStrength::SeqCst,
    }
}

/// Extract the scope from an ordering (for GPU-scoped orderings).
fn ordering_scope(ord: &Ordering) -> Scope {
    match ord {
        Ordering::AcquireCTA | Ordering::ReleaseCTA => Scope::CTA,
        Ordering::AcquireGPU | Ordering::ReleaseGPU => Scope::GPU,
        Ordering::AcquireSystem | Ordering::ReleaseSystem => Scope::System,
        _ => Scope::None,
    }
}

/// Return `true` if the ordering has acquire semantics.
fn is_acquire(ord: &Ordering) -> bool {
    matches!(
        ord,
        Ordering::Acquire | Ordering::AcqRel | Ordering::SeqCst
        | Ordering::AcquireCTA | Ordering::AcquireGPU | Ordering::AcquireSystem
    )
}

/// Return `true` if the ordering has release semantics.
fn is_release(ord: &Ordering) -> bool {
    matches!(
        ord,
        Ordering::Release | Ordering::AcqRel | Ordering::SeqCst
        | Ordering::ReleaseCTA | Ordering::ReleaseGPU | Ordering::ReleaseSystem
    )
}

// ---------------------------------------------------------------------------
// DefUseChainBuilder
// ---------------------------------------------------------------------------

/// Builds def-use chains from an instruction sequence.
struct DefUseChainBuilder<'a> {
    instrs: &'a [Instruction],
    cfg: &'a ControlFlowGraph,
}

impl<'a> DefUseChainBuilder<'a> {
    fn new(instrs: &'a [Instruction], cfg: &'a ControlFlowGraph) -> Self {
        Self { instrs, cfg }
    }

    /// Build the complete def-use chain.
    fn build(&self) -> DefUseChain {
        let mut chain = DefUseChain::new();

        // Pass 1: Collect all definitions and uses.
        for (i, instr) in self.instrs.iter().enumerate() {
            let block_id = self.cfg.block_of(i).unwrap_or(0);
            match instr {
                Instruction::Load { reg, addr: _, ordering: _ } => {
                    chain.add_def(RegDef {
                        reg: *reg,
                        instr_idx: i,
                        block_id,
                    });
                }
                Instruction::Store { addr: _, value: _, ordering: _ } => {
                    // Stores don't define registers but we record register
                    // uses below by scanning all instructions for register
                    // operands.
                }
                Instruction::RMW { reg, addr: _, value: _, ordering: _ } => {
                    chain.add_def(RegDef {
                        reg: *reg,
                        instr_idx: i,
                        block_id,
                    });
                }
                Instruction::BranchCond { reg, .. } => {
                    chain.add_use(RegUse {
                        reg: *reg,
                        instr_idx: i,
                        usage: RegisterUsage::AsCondition,
                    });
                }
                Instruction::Label { .. }
                | Instruction::Branch { .. }
                | Instruction::Fence { .. } => {}
            }
        }

        // Pass 2: Compute reaching definitions using iterative dataflow.
        // gen[i] = definitions generated at instruction i
        // kill[i] = definitions killed at instruction i
        let n = self.instrs.len();
        // reaching[i] = set of (reg, def_instr_idx) reaching the *entry* of instruction i
        let mut reaching: Vec<HashMap<RegId, HashSet<usize>>> = vec![HashMap::new(); n + 1];

        let mut changed = true;
        while changed {
            changed = false;
            for i in 0..n {
                // Merge reaching from predecessors.
                let mut incoming: HashMap<RegId, HashSet<usize>> = HashMap::new();

                if i == 0 {
                    // Entry point: no predecessors — nothing reaches here yet.
                } else {
                    // For straight-line code, the predecessor is i-1.
                    // For branch targets, we also merge from branch sources.
                    let my_block = self.cfg.block_of(i).unwrap_or(0);
                    let prev_block = if i > 0 { self.cfg.block_of(i - 1) } else { None };

                    if prev_block == Some(my_block) {
                        // Same block: predecessor is the previous instruction's
                        // exit state.
                        Self::merge_reaching(&mut incoming, &reaching[i]);
                    }

                    // If this is a block leader, also merge from predecessor
                    // blocks' last instructions.
                    if self.cfg.blocks.get(my_block).map(|b| b.start) == Some(i) {
                        for &pred_block in &self.cfg.blocks[my_block].predecessors {
                            let pred_end = self.cfg.blocks[pred_block].end;
                            if pred_end > 0 {
                                Self::merge_reaching(&mut incoming, &reaching[pred_end]);
                            }
                        }
                    }
                }

                // Apply gen/kill for instruction i.
                let mut exit_state = incoming.clone();

                match &self.instrs[i] {
                    Instruction::Load { reg, .. } | Instruction::RMW { reg, .. } => {
                        // Kill previous definitions of this register.
                        exit_state.insert(*reg, {
                            let mut s = HashSet::new();
                            s.insert(i);
                            s
                        });
                    }
                    _ => {}
                }

                if reaching[i] != incoming {
                    reaching[i] = incoming;
                    changed = true;
                }
                if reaching.get(i + 1) != Some(&exit_state) {
                    if i + 1 <= n {
                        reaching[i + 1] = exit_state;
                    }
                    changed = true;
                }
            }
        }

        // Pass 3: For each use, record the reaching definitions.
        // Collect uses to avoid borrow conflict with chain.
        let uses_snapshot: Vec<(usize, RegId)> = chain
            .uses
            .iter()
            .map(|u| (u.instr_idx, u.reg))
            .collect();
        for (use_idx, reg) in uses_snapshot {
            if let Some(reach) = reaching.get(use_idx) {
                if let Some(defs) = reach.get(&reg) {
                    for &def_idx in defs {
                        chain.add_reaching_def(use_idx, reg, def_idx);
                    }
                }
            }
        }

        chain
    }

    /// Merge a reaching-definitions map into `target`.
    fn merge_reaching(
        target: &mut HashMap<RegId, HashSet<usize>>,
        source: &HashMap<RegId, HashSet<usize>>,
    ) {
        for (reg, defs) in source {
            target.entry(*reg).or_default().extend(defs);
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers: extract register operands from instructions
// ---------------------------------------------------------------------------

/// Return the register defined by an instruction (if any).
fn defined_reg(instr: &Instruction) -> Option<RegId> {
    match instr {
        Instruction::Load { reg, .. } | Instruction::RMW { reg, .. } => Some(*reg),
        _ => None,
    }
}

/// Return registers used by an instruction, together with their usage kind.
fn used_regs(instr: &Instruction) -> Vec<(RegId, RegisterUsage)> {
    match instr {
        Instruction::BranchCond { reg, .. } => {
            vec![(*reg, RegisterUsage::AsCondition)]
        }
        // Note: In the current Instruction encoding, addresses and values are
        // raw u64 constants, not register references. The register-based
        // dependency patterns (addr/data) are detected by matching a load's
        // destination register against a subsequent instruction's address/value
        // field via the def-use chain. We thus do not report uses here for
        // Load/Store address/value fields — those are handled by the
        // DependencyInference pass directly.
        _ => Vec::new(),
    }
}

/// Return the address accessed by a memory instruction (if any).
fn accessed_address(instr: &Instruction) -> Option<Address> {
    match instr {
        Instruction::Load { addr, .. }
        | Instruction::Store { addr, .. }
        | Instruction::RMW { addr, .. } => Some(*addr),
        _ => None,
    }
}

/// Return `true` if the instruction is a memory read.
fn is_read(instr: &Instruction) -> bool {
    matches!(instr, Instruction::Load { .. } | Instruction::RMW { .. })
}

/// Return `true` if the instruction is a memory write.
fn is_write(instr: &Instruction) -> bool {
    matches!(instr, Instruction::Store { .. } | Instruction::RMW { .. })
}

/// Return the memory ordering of an instruction (if any).
fn instr_ordering(instr: &Instruction) -> Option<Ordering> {
    match instr {
        Instruction::Load { ordering, .. }
        | Instruction::Store { ordering, .. }
        | Instruction::RMW { ordering, .. }
        | Instruction::Fence { ordering, .. } => Some(*ordering),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// DependencyInference — main entry point
// ---------------------------------------------------------------------------

/// Main entry point for dependency inference.
pub struct DependencyInference;

impl DependencyInference {
    /// Infer all intra-thread dependencies from a sequence of instructions.
    ///
    /// Returns a `DependencyGraph` with thread_id = 0.
    pub fn infer_from_instructions(instrs: &[Instruction]) -> DependencyGraph {
        let mut graph = DependencyGraph::new(0);
        if instrs.is_empty() {
            return graph;
        }

        let cfg = ControlFlowGraph::build(instrs);
        let du_chain = DefUseChainBuilder::new(instrs, &cfg).build();

        Self::detect_address_deps(instrs, &du_chain, &mut graph);
        Self::detect_data_deps(instrs, &du_chain, &mut graph);
        Self::detect_control_deps(instrs, &cfg, &du_chain, &mut graph);
        Self::detect_anti_deps(instrs, &mut graph);
        Self::detect_output_deps(instrs, &mut graph);
        Self::detect_register_deps(instrs, &du_chain, &mut graph);

        graph
    }

    /// Infer dependencies for every thread of a litmus test.
    pub fn infer_from_litmus_test(test: &LitmusTest) -> TestDependencies {
        let mut test_deps = TestDependencies::new(&test.name);
        for thread in &test.threads {
            let mut graph = Self::infer_from_instructions(&thread.instructions);
            graph.thread_id = thread.id;
            test_deps.thread_deps.insert(thread.id, graph);
        }
        test_deps
    }

    // ----- Address dependency detection ------------------------------------

    /// Detect address dependencies.
    ///
    /// Pattern: `r1 = load(x); r2 = load(r1)` — the address of the second
    /// load depends on the value produced by the first load.
    ///
    /// Because addresses in the `Instruction` encoding are raw `u64` values
    /// (not register names), we approximate address dependencies by checking
    /// whether a load's result register is later used as an address for
    /// another memory operation. We detect this through the def-use chain:
    /// if a load defines register `r` at index `i`, and a subsequent memory
    /// instruction at index `j` accesses address `A` where `A == r`, then
    /// there is an address dependency from `i` to `j`.
    fn detect_address_deps(
        instrs: &[Instruction],
        du_chain: &DefUseChain,
        graph: &mut DependencyGraph,
    ) {
        // Build a map: instruction idx -> defined register.
        let _def_map: HashMap<usize, RegId> = du_chain
            .definitions
            .iter()
            .map(|d| (d.instr_idx, d.reg))
            .collect();

        for (j, instr_j) in instrs.iter().enumerate() {
            let addr_j = match accessed_address(instr_j) {
                Some(a) => a,
                None => continue,
            };

            // Check if any earlier load/rmw defined a register whose numeric
            // id matches the address value. This is the heuristic for detecting
            // indirect address computation (the Instruction representation uses
            // flat u64 for both addresses and register IDs).
            for def in &du_chain.definitions {
                let i = def.instr_idx;
                if i >= j {
                    continue;
                }
                // Heuristic: if the register id matches the address value, the
                // address depends on that load.
                if def.reg as u64 == addr_j {
                    let strength = Self::combined_strength(instrs, i, j);
                    graph.add_edge(Dependency::new(
                        i, j, DependencyKind::Address, strength,
                    ));
                }
            }
        }
    }

    // ----- Data dependency detection ---------------------------------------

    /// Detect data dependencies.
    ///
    /// Pattern: `r1 = load(x); store(y, r1)` — the stored value depends on
    /// the loaded value.
    fn detect_data_deps(
        instrs: &[Instruction],
        du_chain: &DefUseChain,
        graph: &mut DependencyGraph,
    ) {
        for (j, instr_j) in instrs.iter().enumerate() {
            let store_value = match instr_j {
                Instruction::Store { value, .. } => *value,
                Instruction::RMW { value, .. } => *value,
                _ => continue,
            };

            // Check if any earlier load/rmw defined a register whose id
            // matches the stored value.
            for def in &du_chain.definitions {
                let i = def.instr_idx;
                if i >= j {
                    continue;
                }
                if def.reg as u64 == store_value {
                    let strength = Self::combined_strength(instrs, i, j);
                    graph.add_edge(Dependency::new(
                        i, j, DependencyKind::Data, strength,
                    ));
                }
            }
        }
    }

    // ----- Control dependency detection ------------------------------------

    /// Detect control dependencies.
    ///
    /// Pattern: `r1 = load(x); if r1 == 0: store(y, 1)` — the store is
    /// control-dependent on the load.
    fn detect_control_deps(
        instrs: &[Instruction],
        cfg: &ControlFlowGraph,
        du_chain: &DefUseChain,
        graph: &mut DependencyGraph,
    ) {
        // For each BranchCond, find the load that defines its condition
        // register. Then, all memory accesses in the blocks reachable only
        // through the branch are control-dependent on that load.
        for (branch_idx, instr) in instrs.iter().enumerate() {
            let cond_reg = match instr {
                Instruction::BranchCond { reg, .. } => *reg,
                _ => continue,
            };

            // Find the loads that define the condition register.
            let source_defs = du_chain.defs_reaching(branch_idx, cond_reg);
            if source_defs.is_empty() {
                continue;
            }

            // Find blocks reachable from the branch (its successors).
            let branch_block = match cfg.block_of(branch_idx) {
                Some(b) => b,
                None => continue,
            };
            let reachable = cfg.dominated_by(branch_block, true);

            // Every memory access in a reachable block is control-dependent on
            // the defining loads.
            for &block_id in &reachable {
                if block_id >= cfg.blocks.len() {
                    continue;
                }
                let block = &cfg.blocks[block_id];
                for j in block.start..block.end {
                    if j >= instrs.len() {
                        break;
                    }
                    if is_read(&instrs[j]) || is_write(&instrs[j]) {
                        for &def_idx in &source_defs {
                            let strength = Self::combined_strength(instrs, def_idx, j);
                            graph.add_edge(Dependency::new(
                                def_idx,
                                j,
                                DependencyKind::Control,
                                strength,
                            ));
                        }
                    }
                }
            }

            // Also, any memory access that appears *after* the branch in the
            // same block is considered control-dependent (conservatively).
            if let Some(block) = cfg.blocks.get(branch_block) {
                for j in (branch_idx + 1)..block.end {
                    if j >= instrs.len() {
                        break;
                    }
                    if is_read(&instrs[j]) || is_write(&instrs[j]) {
                        for &def_idx in &source_defs {
                            if !graph.has_edge_kind(def_idx, j, DependencyKind::Control) {
                                let strength = Self::combined_strength(instrs, def_idx, j);
                                graph.add_edge(Dependency::new(
                                    def_idx,
                                    j,
                                    DependencyKind::Control,
                                    strength,
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    // ----- Anti-dependency detection (WAR) ---------------------------------

    /// Detect anti-dependencies (WAR): a write that follows a read to the
    /// same address.
    fn detect_anti_deps(instrs: &[Instruction], graph: &mut DependencyGraph) {
        for (i, instr_i) in instrs.iter().enumerate() {
            if !is_read(instr_i) {
                continue;
            }
            let addr_i = match accessed_address(instr_i) {
                Some(a) => a,
                None => continue,
            };
            for (j, instr_j) in instrs.iter().enumerate().skip(i + 1) {
                if !is_write(instr_j) {
                    continue;
                }
                let addr_j = match accessed_address(instr_j) {
                    Some(a) => a,
                    None => continue,
                };
                if addr_i == addr_j {
                    let strength = Self::combined_strength(instrs, i, j);
                    graph.add_edge(Dependency::new(
                        i, j, DependencyKind::AntiDependency, strength,
                    ));
                }
            }
        }
    }

    // ----- Output dependency detection (WAW) -------------------------------

    /// Detect output dependencies (WAW): two writes to the same address.
    fn detect_output_deps(instrs: &[Instruction], graph: &mut DependencyGraph) {
        for (i, instr_i) in instrs.iter().enumerate() {
            if !is_write(instr_i) {
                continue;
            }
            let addr_i = match accessed_address(instr_i) {
                Some(a) => a,
                None => continue,
            };
            for (j, instr_j) in instrs.iter().enumerate().skip(i + 1) {
                if !is_write(instr_j) {
                    continue;
                }
                let addr_j = match accessed_address(instr_j) {
                    Some(a) => a,
                    None => continue,
                };
                if addr_i == addr_j {
                    let strength = Self::combined_strength(instrs, i, j);
                    graph.add_edge(Dependency::new(
                        i, j, DependencyKind::Output, strength,
                    ));
                }
            }
        }
    }

    // ----- Generic register dependency detection ---------------------------

    /// Detect register dependencies that are not already covered by
    /// address/data/control analyses (e.g. a load result used as an RMW value).
    fn detect_register_deps(
        instrs: &[Instruction],
        du_chain: &DefUseChain,
        graph: &mut DependencyGraph,
    ) {
        for u in &du_chain.uses {
            let reaching = du_chain.defs_reaching(u.instr_idx, u.reg);
            for def_idx in reaching {
                // Skip if we already have a more specific dependency.
                if graph.has_edge(def_idx, u.instr_idx) {
                    continue;
                }
                let strength = Self::combined_strength(instrs, def_idx, u.instr_idx);
                graph.add_edge(Dependency::new(
                    def_idx,
                    u.instr_idx,
                    DependencyKind::Register,
                    strength,
                ));
            }
        }
    }

    // ----- Strength helpers ------------------------------------------------

    /// Compute the combined ordering strength of two instructions.
    fn combined_strength(
        instrs: &[Instruction],
        src: usize,
        tgt: usize,
    ) -> DependencyStrength {
        let s1 = instrs
            .get(src)
            .and_then(|i| instr_ordering(i))
            .map(|o| ordering_to_strength(&o))
            .unwrap_or(DependencyStrength::Relaxed);
        let s2 = instrs
            .get(tgt)
            .and_then(|i| instr_ordering(i))
            .map(|o| ordering_to_strength(&o))
            .unwrap_or(DependencyStrength::Relaxed);
        std::cmp::max(s1, s2)
    }

    // ----- Synchronization pattern recognition -----------------------------

    /// Recognize synchronization patterns within a single thread.
    pub fn recognize_sync_patterns(
        instrs: &[Instruction],
        thread_id: ThreadId,
    ) -> Vec<SyncPattern> {
        let mut patterns = Vec::new();

        for (i, instr_i) in instrs.iter().enumerate() {
            match instr_i {
                // Fence followed by memory access.
                Instruction::Fence { ordering, scope } => {
                    for (j, instr_j) in instrs.iter().enumerate().skip(i + 1) {
                        if is_read(instr_j) || is_write(instr_j) {
                            patterns.push(SyncPattern::FenceAccess {
                                fence_idx: i,
                                access_idx: j,
                                fence_ordering: *ordering,
                                scope: *scope,
                            });
                            break; // Only the first access after the fence.
                        }
                    }
                    // Memory access before the fence.
                    for k in (0..i).rev() {
                        if is_read(&instrs[k]) || is_write(&instrs[k]) {
                            patterns.push(SyncPattern::AccessFence {
                                access_idx: k,
                                fence_idx: i,
                                fence_ordering: *ordering,
                                scope: *scope,
                            });
                            break;
                        }
                    }
                }
                // RMW with AcqRel semantics.
                Instruction::RMW { ordering, .. } => {
                    if is_acquire(ordering) && is_release(ordering) {
                        patterns.push(SyncPattern::RmwAcqRel {
                            thread: thread_id,
                            idx: i,
                            scope: ordering_scope(ordering),
                        });
                    }
                }
                _ => {}
            }
        }
        patterns
    }

    /// Recognize release-acquire pairs across threads in a litmus test.
    pub fn recognize_cross_thread_sync(test: &LitmusTest) -> Vec<SyncPattern> {
        let mut patterns = Vec::new();
        let n = test.threads.len();

        for ti in 0..n {
            for (i, instr_i) in test.threads[ti].instructions.iter().enumerate() {
                let (rel_addr, rel_ordering) = match instr_i {
                    Instruction::Store { addr, ordering, .. } if is_release(ordering) => {
                        (*addr, *ordering)
                    }
                    Instruction::RMW { addr, ordering, .. } if is_release(ordering) => {
                        (*addr, *ordering)
                    }
                    _ => continue,
                };
                for tj in 0..n {
                    if ti == tj {
                        continue;
                    }
                    for (j, instr_j) in test.threads[tj].instructions.iter().enumerate() {
                        let (acq_addr, acq_ordering) = match instr_j {
                            Instruction::Load { addr, ordering, .. } if is_acquire(ordering) => {
                                (*addr, *ordering)
                            }
                            Instruction::RMW { addr, ordering, .. } if is_acquire(ordering) => {
                                (*addr, *ordering)
                            }
                            _ => continue,
                        };
                        if rel_addr == acq_addr {
                            let scope = std::cmp::max(
                                ordering_scope(&rel_ordering) as u8,
                                ordering_scope(&acq_ordering) as u8,
                            );
                            let combined_scope = match scope {
                                s if s == Scope::System as u8 => Scope::System,
                                s if s == Scope::GPU as u8 => Scope::GPU,
                                s if s == Scope::CTA as u8 => Scope::CTA,
                                _ => Scope::None,
                            };
                            patterns.push(SyncPattern::ReleaseAcquire {
                                release_thread: test.threads[ti].id,
                                release_idx: i,
                                acquire_thread: test.threads[tj].id,
                                acquire_idx: j,
                                scope: combined_scope,
                            });
                        }
                    }
                }
            }
        }
        patterns
    }
}

// ---------------------------------------------------------------------------
// Output formatting
// ---------------------------------------------------------------------------

/// Annotation for a single dependency in a litmus test.
#[derive(Debug, Clone)]
pub struct DependencyAnnotation {
    /// Thread id.
    pub thread: ThreadId,
    /// Source instruction index.
    pub source: EventId,
    /// Target instruction index.
    pub target: EventId,
    /// Kind of dependency.
    pub kind: DependencyKind,
    /// Strength.
    pub strength: DependencyStrength,
}

impl fmt::Display for DependencyAnnotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "T{}:{}--{}-->{}[{}]",
            self.thread, self.source, self.kind, self.target, self.strength,
        )
    }
}

/// Generate dependency annotations for a litmus test.
pub fn generate_annotations(deps: &TestDependencies) -> Vec<DependencyAnnotation> {
    let mut annotations = Vec::new();
    let mut tids: Vec<_> = deps.thread_deps.keys().collect();
    tids.sort();
    for &tid in &tids {
        let graph = &deps.thread_deps[&tid];
        for dep in graph.edges() {
            annotations.push(DependencyAnnotation {
                thread: *tid,
                source: dep.source,
                target: dep.target,
                kind: dep.kind,
                strength: dep.strength,
            });
        }
    }
    annotations
}

/// Format dependency annotations in herd7-compatible syntax.
///
/// Herd7 uses relation names like `addr`, `data`, `ctrl`, `rf`, `co`, `fr`.
/// We map our dependency kinds to the closest herd7 relation.
pub fn format_herd7(deps: &TestDependencies) -> String {
    let mut lines = Vec::new();
    lines.push(format!("(* Dependencies for {} *)", deps.test_name));

    let mut tids: Vec<_> = deps.thread_deps.keys().collect();
    tids.sort();

    for &tid in &tids {
        let graph = &deps.thread_deps[&tid];
        if graph.is_empty() {
            continue;
        }
        lines.push(format!("(* Thread {} *)", tid));
        for dep in graph.edges() {
            let rel_name = match dep.kind {
                DependencyKind::Address => "addr",
                DependencyKind::Data => "data",
                DependencyKind::Control => "ctrl",
                DependencyKind::AntiDependency => "fr",
                DependencyKind::Output => "co",
                DependencyKind::Register => "iico_data",
            };
            lines.push(format!(
                "{} T{}:{} T{}:{}",
                rel_name, tid, dep.source, tid, dep.target,
            ));
        }
    }
    lines.join("\n")
}

/// Generate a DOT graph visualization of dependencies.
pub fn to_dot(deps: &TestDependencies) -> String {
    let mut lines = Vec::new();
    lines.push("digraph dependencies {".to_string());
    lines.push("  rankdir=TB;".to_string());
    lines.push(format!("  label=\"{}\";", deps.test_name));
    lines.push("  node [shape=box, fontname=\"monospace\"];".to_string());

    let mut tids: Vec<_> = deps.thread_deps.keys().collect();
    tids.sort();

    for &tid in &tids {
        let graph = &deps.thread_deps[&tid];
        lines.push(format!("  subgraph cluster_t{} {{", tid));
        lines.push(format!("    label=\"Thread {}\";", tid));

        let mut event_ids: Vec<_> = graph.event_ids().into_iter().collect();
        event_ids.sort();
        for eid in &event_ids {
            lines.push(format!("    t{}e{} [label=\"T{}[{}]\"];", tid, eid, tid, eid));
        }
        lines.push("  }".to_string());

        for dep in graph.edges() {
            let color = match dep.kind {
                DependencyKind::Address => "blue",
                DependencyKind::Data => "red",
                DependencyKind::Control => "green",
                DependencyKind::AntiDependency => "orange",
                DependencyKind::Output => "purple",
                DependencyKind::Register => "gray",
            };
            let style = match dep.strength {
                DependencyStrength::SeqCst => "bold",
                DependencyStrength::AcqRel => "bold",
                _ => "solid",
            };
            lines.push(format!(
                "  t{}e{} -> t{}e{} [label=\"{}\", color={}, style={}];",
                tid, dep.source, tid, dep.target, dep.kind, color, style,
            ));
        }
    }

    lines.push("}".to_string());
    lines.join("\n")
}

/// Compact textual summary of a `TestDependencies`.
pub fn summary(deps: &TestDependencies) -> String {
    let mut parts = Vec::new();
    parts.push(format!("Test: {}", deps.test_name));

    let kinds = [
        DependencyKind::Address,
        DependencyKind::Data,
        DependencyKind::Control,
        DependencyKind::AntiDependency,
        DependencyKind::Output,
        DependencyKind::Register,
    ];

    for kind in &kinds {
        let count = deps.all_edges_of_kind(*kind).len();
        if count > 0 {
            parts.push(format!("  {}: {}", kind, count));
        }
    }
    parts.push(format!("  total: {}", deps.total_edges()));
    parts.join("\n")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to build a simple instruction list.
    fn make_load(reg: RegId, addr: Address, ordering: Ordering) -> Instruction {
        Instruction::Load { reg, addr, ordering }
    }

    fn make_store(addr: Address, value: Value, ordering: Ordering) -> Instruction {
        Instruction::Store { addr, value, ordering }
    }

    fn make_rmw(reg: RegId, addr: Address, value: Value, ordering: Ordering) -> Instruction {
        Instruction::RMW { reg, addr, value, ordering }
    }

    fn make_fence(ordering: Ordering, scope: Scope) -> Instruction {
        Instruction::Fence { ordering, scope }
    }

    fn make_branch_cond(reg: RegId, expected: Value, label: usize) -> Instruction {
        Instruction::BranchCond { reg, expected, label }
    }

    fn make_label(id: usize) -> Instruction {
        Instruction::Label { id }
    }

    fn make_branch(label: usize) -> Instruction {
        Instruction::Branch { label }
    }

    // -----------------------------------------------------------------------
    // DependencyKind display
    // -----------------------------------------------------------------------

    #[test]
    fn test_dependency_kind_display() {
        assert_eq!(format!("{}", DependencyKind::Address), "addr");
        assert_eq!(format!("{}", DependencyKind::Data), "data");
        assert_eq!(format!("{}", DependencyKind::Control), "ctrl");
        assert_eq!(format!("{}", DependencyKind::AntiDependency), "anti");
        assert_eq!(format!("{}", DependencyKind::Output), "output");
        assert_eq!(format!("{}", DependencyKind::Register), "reg");
    }

    // -----------------------------------------------------------------------
    // DependencyStrength display and ordering
    // -----------------------------------------------------------------------

    #[test]
    fn test_dependency_strength_display() {
        assert_eq!(format!("{}", DependencyStrength::Relaxed), "rlx");
        assert_eq!(format!("{}", DependencyStrength::SeqCst), "sc");
    }

    #[test]
    fn test_strength_ordering() {
        assert!(DependencyStrength::Relaxed < DependencyStrength::Acquire);
        assert!(DependencyStrength::Acquire < DependencyStrength::Release);
        assert!(DependencyStrength::Release < DependencyStrength::AcqRel);
        assert!(DependencyStrength::AcqRel < DependencyStrength::SeqCst);
    }

    // -----------------------------------------------------------------------
    // DependencyGraph
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_graph() {
        let g = DependencyGraph::new(0);
        assert!(g.is_empty());
        assert_eq!(g.len(), 0);
        assert!(g.deps_from(0).is_empty());
        assert!(g.deps_to(0).is_empty());
    }

    #[test]
    fn test_graph_add_edge() {
        let mut g = DependencyGraph::new(0);
        g.add_edge(Dependency::new(0, 1, DependencyKind::Data, DependencyStrength::Relaxed));
        assert_eq!(g.len(), 1);
        assert!(!g.is_empty());
        assert!(g.has_edge(0, 1));
        assert!(!g.has_edge(1, 0));
        assert!(g.has_edge_kind(0, 1, DependencyKind::Data));
        assert!(!g.has_edge_kind(0, 1, DependencyKind::Address));
    }

    #[test]
    fn test_graph_edges_of_kind() {
        let mut g = DependencyGraph::new(0);
        g.add_edge(Dependency::new(0, 1, DependencyKind::Data, DependencyStrength::Relaxed));
        g.add_edge(Dependency::new(1, 2, DependencyKind::Address, DependencyStrength::Relaxed));
        g.add_edge(Dependency::new(0, 2, DependencyKind::Data, DependencyStrength::Relaxed));
        assert_eq!(g.edges_of_kind(DependencyKind::Data).len(), 2);
        assert_eq!(g.edges_of_kind(DependencyKind::Address).len(), 1);
        assert_eq!(g.edges_of_kind(DependencyKind::Control).len(), 0);
    }

    #[test]
    fn test_graph_event_ids() {
        let mut g = DependencyGraph::new(0);
        g.add_edge(Dependency::new(0, 1, DependencyKind::Data, DependencyStrength::Relaxed));
        g.add_edge(Dependency::new(2, 3, DependencyKind::Data, DependencyStrength::Relaxed));
        let ids = g.event_ids();
        assert_eq!(ids.len(), 4);
        assert!(ids.contains(&0));
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert!(ids.contains(&3));
    }

    #[test]
    fn test_graph_merge() {
        let mut g1 = DependencyGraph::new(0);
        g1.add_edge(Dependency::new(0, 1, DependencyKind::Data, DependencyStrength::Relaxed));
        let mut g2 = DependencyGraph::new(0);
        g2.add_edge(Dependency::new(2, 3, DependencyKind::Address, DependencyStrength::Acquire));
        g1.merge(&g2);
        assert_eq!(g1.len(), 2);
    }

    #[test]
    fn test_graph_display() {
        let mut g = DependencyGraph::new(1);
        g.add_edge(Dependency::new(0, 1, DependencyKind::Data, DependencyStrength::Relaxed));
        let s = format!("{}", g);
        assert!(s.contains("Thread 1"));
        assert!(s.contains("data"));
    }

    // -----------------------------------------------------------------------
    // TestDependencies
    // -----------------------------------------------------------------------

    #[test]
    fn test_test_dependencies_empty() {
        let td = TestDependencies::new("MP");
        assert_eq!(td.total_edges(), 0);
        assert!(td.for_thread(0).is_none());
    }

    #[test]
    fn test_test_dependencies_add_thread() {
        let mut td = TestDependencies::new("SB");
        let g = td.for_thread_mut(0);
        g.add_edge(Dependency::new(0, 1, DependencyKind::Data, DependencyStrength::Relaxed));
        assert_eq!(td.total_edges(), 1);
        assert!(td.for_thread(0).is_some());
    }

    // -----------------------------------------------------------------------
    // BasicBlock / CFG
    // -----------------------------------------------------------------------

    #[test]
    fn test_basic_block_properties() {
        let bb = BasicBlock::new(0, 0, 3);
        assert_eq!(bb.len(), 3);
        assert!(!bb.is_empty());
        assert!(bb.contains(0));
        assert!(bb.contains(2));
        assert!(!bb.contains(3));
    }

    #[test]
    fn test_cfg_empty() {
        let cfg = ControlFlowGraph::build(&[]);
        assert!(cfg.blocks.is_empty());
    }

    #[test]
    fn test_cfg_straight_line() {
        let instrs = vec![
            make_load(1, 0x100, Ordering::Relaxed),
            make_store(0x200, 1, Ordering::Relaxed),
        ];
        let cfg = ControlFlowGraph::build(&instrs);
        assert_eq!(cfg.blocks.len(), 1);
        assert_eq!(cfg.blocks[0].start, 0);
        assert_eq!(cfg.blocks[0].end, 2);
    }

    #[test]
    fn test_cfg_with_branch() {
        let instrs = vec![
            make_load(1, 0x100, Ordering::Relaxed),       // 0
            make_branch_cond(1, 0, 10),                    // 1: if r1==0 goto L10
            make_store(0x200, 1, Ordering::Relaxed),       // 2: fall-through
            make_label(10),                                 // 3: L10
            make_store(0x300, 2, Ordering::Relaxed),       // 4
        ];
        let cfg = ControlFlowGraph::build(&instrs);
        // Blocks: [0..2], [2..3], [3..5]
        assert!(cfg.blocks.len() >= 2);
        // Block containing instruction 0 should have a successor leading to
        // the label block.
        let block_0 = cfg.block_of(0).unwrap();
        assert!(!cfg.blocks[block_0].successors.is_empty());
    }

    #[test]
    fn test_cfg_dominators() {
        let instrs = vec![
            make_load(1, 0x100, Ordering::Relaxed),       // 0
            make_branch_cond(1, 0, 10),                    // 1
            make_store(0x200, 1, Ordering::Relaxed),       // 2
            make_label(10),                                 // 3
            make_store(0x300, 2, Ordering::Relaxed),       // 4
        ];
        let cfg = ControlFlowGraph::build(&instrs);
        let idom = cfg.compute_dominators();
        // Block 0 should dominate all other blocks.
        // idom should not be empty (unless there's only one block).
        if cfg.blocks.len() > 1 {
            assert!(!idom.is_empty());
        }
    }

    // -----------------------------------------------------------------------
    // DefUseChain
    // -----------------------------------------------------------------------

    #[test]
    fn test_def_use_chain_simple() {
        let instrs = vec![
            make_load(1, 0x100, Ordering::Relaxed),   // 0: r1 = load(x)
            make_store(0x200, 1, Ordering::Relaxed),   // 1: store(y, r1)
        ];
        let cfg = ControlFlowGraph::build(&instrs);
        let du = DefUseChainBuilder::new(&instrs, &cfg).build();
        assert!(!du.definitions.is_empty());
        assert_eq!(du.definitions[0].reg, 1);
        assert_eq!(du.definitions[0].instr_idx, 0);
    }

    // -----------------------------------------------------------------------
    // Data dependency detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_data_dependency_load_store() {
        // r1 = load(x); store(y, r1)
        let instrs = vec![
            make_load(1, 0x100, Ordering::Relaxed),
            make_store(0x200, 1, Ordering::Relaxed),
        ];
        let graph = DependencyInference::infer_from_instructions(&instrs);
        assert!(graph.has_edge_kind(0, 1, DependencyKind::Data));
    }

    #[test]
    fn test_data_dependency_no_match() {
        // r1 = load(x); store(y, 42) — value 42 != reg 1
        let instrs = vec![
            make_load(1, 0x100, Ordering::Relaxed),
            make_store(0x200, 42, Ordering::Relaxed),
        ];
        let graph = DependencyInference::infer_from_instructions(&instrs);
        assert!(!graph.has_edge_kind(0, 1, DependencyKind::Data));
    }

    #[test]
    fn test_data_dependency_chain() {
        // r1 = load(x); r2 = load(y); store(z, r1)
        let instrs = vec![
            make_load(1, 0x100, Ordering::Relaxed),
            make_load(2, 0x200, Ordering::Relaxed),
            make_store(0x300, 1, Ordering::Relaxed),
        ];
        let graph = DependencyInference::infer_from_instructions(&instrs);
        assert!(graph.has_edge_kind(0, 2, DependencyKind::Data));
        // No data dep from r2=load(y) to store(z, r1) since value is 1 not 2.
        assert!(!graph.has_edge_kind(1, 2, DependencyKind::Data));
    }

    // -----------------------------------------------------------------------
    // Address dependency detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_address_dependency() {
        // r1 = load(x); r2 = load(r1) — addr of second load == reg id 1
        let instrs = vec![
            make_load(1, 0x100, Ordering::Relaxed),
            make_load(2, 1, Ordering::Relaxed), // addr=1, same as reg id
        ];
        let graph = DependencyInference::infer_from_instructions(&instrs);
        assert!(graph.has_edge_kind(0, 1, DependencyKind::Address));
    }

    #[test]
    fn test_address_dependency_no_match() {
        // r1 = load(x); r2 = load(0x300) — no address dep
        let instrs = vec![
            make_load(1, 0x100, Ordering::Relaxed),
            make_load(2, 0x300, Ordering::Relaxed),
        ];
        let graph = DependencyInference::infer_from_instructions(&instrs);
        assert!(graph.edges_of_kind(DependencyKind::Address).is_empty());
    }

    // -----------------------------------------------------------------------
    // Control dependency detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_control_dependency() {
        // r1 = load(x); if r1==0 goto L; store(y,1); L: store(z,2)
        let instrs = vec![
            make_load(1, 0x100, Ordering::Relaxed),       // 0
            make_branch_cond(1, 0, 10),                    // 1
            make_store(0x200, 42, Ordering::Relaxed),      // 2: fall-through
            make_label(10),                                 // 3
            make_store(0x300, 99, Ordering::Relaxed),      // 4
        ];
        let graph = DependencyInference::infer_from_instructions(&instrs);
        // The stores at 2 and 4 should be control-dependent on the load at 0.
        let ctrl_edges = graph.edges_of_kind(DependencyKind::Control);
        assert!(!ctrl_edges.is_empty());
        // At least one control dep from 0 to a store.
        let has_ctrl = ctrl_edges.iter().any(|d| d.source == 0);
        assert!(has_ctrl);
    }

    #[test]
    fn test_no_control_dep_without_branch() {
        let instrs = vec![
            make_load(1, 0x100, Ordering::Relaxed),
            make_store(0x200, 42, Ordering::Relaxed),
        ];
        let graph = DependencyInference::infer_from_instructions(&instrs);
        assert!(graph.edges_of_kind(DependencyKind::Control).is_empty());
    }

    // -----------------------------------------------------------------------
    // Anti-dependency (WAR) detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_anti_dependency() {
        // r1 = load(x); store(x, 42) — WAR on x
        let instrs = vec![
            make_load(1, 0x100, Ordering::Relaxed),
            make_store(0x100, 42, Ordering::Relaxed),
        ];
        let graph = DependencyInference::infer_from_instructions(&instrs);
        assert!(graph.has_edge_kind(0, 1, DependencyKind::AntiDependency));
    }

    #[test]
    fn test_no_anti_dep_different_addr() {
        let instrs = vec![
            make_load(1, 0x100, Ordering::Relaxed),
            make_store(0x200, 42, Ordering::Relaxed),
        ];
        let graph = DependencyInference::infer_from_instructions(&instrs);
        assert!(graph.edges_of_kind(DependencyKind::AntiDependency).is_empty());
    }

    // -----------------------------------------------------------------------
    // Output dependency (WAW) detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_output_dependency() {
        // store(x, 1); store(x, 2) — WAW on x
        let instrs = vec![
            make_store(0x100, 1, Ordering::Relaxed),
            make_store(0x100, 2, Ordering::Relaxed),
        ];
        let graph = DependencyInference::infer_from_instructions(&instrs);
        assert!(graph.has_edge_kind(0, 1, DependencyKind::Output));
    }

    #[test]
    fn test_no_output_dep_different_addr() {
        let instrs = vec![
            make_store(0x100, 1, Ordering::Relaxed),
            make_store(0x200, 2, Ordering::Relaxed),
        ];
        let graph = DependencyInference::infer_from_instructions(&instrs);
        assert!(graph.edges_of_kind(DependencyKind::Output).is_empty());
    }

    // -----------------------------------------------------------------------
    // Ordering strength
    // -----------------------------------------------------------------------

    #[test]
    fn test_strength_from_ordering() {
        assert_eq!(ordering_to_strength(&Ordering::Relaxed), DependencyStrength::Relaxed);
        assert_eq!(ordering_to_strength(&Ordering::Acquire), DependencyStrength::Acquire);
        assert_eq!(ordering_to_strength(&Ordering::Release), DependencyStrength::Release);
        assert_eq!(ordering_to_strength(&Ordering::AcqRel), DependencyStrength::AcqRel);
        assert_eq!(ordering_to_strength(&Ordering::SeqCst), DependencyStrength::SeqCst);
    }

    #[test]
    fn test_combined_strength_uses_max() {
        let instrs = vec![
            make_load(1, 0x100, Ordering::Relaxed),
            make_store(0x100, 1, Ordering::Release),
        ];
        // Combined strength should be Release (max of Relaxed, Release).
        let strength = DependencyInference::combined_strength(&instrs, 0, 1);
        assert_eq!(strength, DependencyStrength::Release);
    }

    // -----------------------------------------------------------------------
    // GPU-scoped ordering helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_ordering_scope() {
        assert_eq!(ordering_scope(&Ordering::AcquireCTA), Scope::CTA);
        assert_eq!(ordering_scope(&Ordering::ReleaseGPU), Scope::GPU);
        assert_eq!(ordering_scope(&Ordering::AcquireSystem), Scope::System);
        assert_eq!(ordering_scope(&Ordering::Relaxed), Scope::None);
    }

    #[test]
    fn test_is_acquire_release() {
        assert!(is_acquire(&Ordering::Acquire));
        assert!(is_acquire(&Ordering::AcqRel));
        assert!(is_acquire(&Ordering::SeqCst));
        assert!(is_acquire(&Ordering::AcquireCTA));
        assert!(!is_acquire(&Ordering::Relaxed));
        assert!(!is_acquire(&Ordering::Release));

        assert!(is_release(&Ordering::Release));
        assert!(is_release(&Ordering::AcqRel));
        assert!(is_release(&Ordering::SeqCst));
        assert!(is_release(&Ordering::ReleaseCTA));
        assert!(!is_release(&Ordering::Relaxed));
        assert!(!is_release(&Ordering::Acquire));
    }

    // -----------------------------------------------------------------------
    // Synchronization pattern recognition
    // -----------------------------------------------------------------------

    #[test]
    fn test_recognize_fence_access_pattern() {
        let instrs = vec![
            make_store(0x100, 1, Ordering::Relaxed),
            make_fence(Ordering::Release, Scope::None),
            make_store(0x200, 1, Ordering::Relaxed),
        ];
        let patterns = DependencyInference::recognize_sync_patterns(&instrs, 0);
        let has_fence_access = patterns.iter().any(|p| matches!(p, SyncPattern::FenceAccess { .. }));
        let has_access_fence = patterns.iter().any(|p| matches!(p, SyncPattern::AccessFence { .. }));
        assert!(has_fence_access);
        assert!(has_access_fence);
    }

    #[test]
    fn test_recognize_rmw_acqrel() {
        let instrs = vec![
            make_rmw(1, 0x100, 1, Ordering::AcqRel),
        ];
        let patterns = DependencyInference::recognize_sync_patterns(&instrs, 0);
        assert_eq!(patterns.len(), 1);
        assert!(matches!(&patterns[0], SyncPattern::RmwAcqRel { thread: 0, idx: 0, .. }));
    }

    #[test]
    fn test_recognize_cross_thread_release_acquire() {
        let test = LitmusTest {
            name: "MP".to_string(),
            threads: vec![
                Thread {
                    id: 0,
                    instructions: vec![
                        make_store(0x100, 1, Ordering::Relaxed),
                        make_store(0x200, 1, Ordering::Release),
                    ],
                },
                Thread {
                    id: 1,
                    instructions: vec![
                        make_load(1, 0x200, Ordering::Acquire),
                        make_load(2, 0x100, Ordering::Relaxed),
                    ],
                },
            ],
            initial_state: HashMap::new(),
            expected_outcomes: Vec::new(),
        };
        let patterns = DependencyInference::recognize_cross_thread_sync(&test);
        assert!(!patterns.is_empty());
        let has_rel_acq = patterns.iter().any(|p| matches!(p, SyncPattern::ReleaseAcquire { .. }));
        assert!(has_rel_acq);
    }

    // -----------------------------------------------------------------------
    // Full litmus test inference
    // -----------------------------------------------------------------------

    #[test]
    fn test_infer_message_passing() {
        let test = LitmusTest {
            name: "MP".to_string(),
            threads: vec![
                Thread {
                    id: 0,
                    instructions: vec![
                        make_store(0x100, 1, Ordering::Relaxed),
                        make_store(0x200, 1, Ordering::Release),
                    ],
                },
                Thread {
                    id: 1,
                    instructions: vec![
                        make_load(1, 0x200, Ordering::Acquire),
                        make_load(2, 0x100, Ordering::Relaxed),
                    ],
                },
            ],
            initial_state: HashMap::new(),
            expected_outcomes: Vec::new(),
        };
        let deps = DependencyInference::infer_from_litmus_test(&test);
        assert_eq!(deps.test_name, "MP");
        assert!(deps.for_thread(0).is_some());
        assert!(deps.for_thread(1).is_some());
    }

    #[test]
    fn test_infer_store_buffering() {
        let test = LitmusTest {
            name: "SB".to_string(),
            threads: vec![
                Thread {
                    id: 0,
                    instructions: vec![
                        make_store(0x100, 1, Ordering::SeqCst),
                        make_load(1, 0x200, Ordering::SeqCst),
                    ],
                },
                Thread {
                    id: 1,
                    instructions: vec![
                        make_store(0x200, 1, Ordering::SeqCst),
                        make_load(2, 0x100, Ordering::SeqCst),
                    ],
                },
            ],
            initial_state: HashMap::new(),
            expected_outcomes: Vec::new(),
        };
        let deps = DependencyInference::infer_from_litmus_test(&test);
        assert_eq!(deps.total_edges(), 0); // No intra-thread deps in SB.
    }

    // -----------------------------------------------------------------------
    // Output formatting
    // -----------------------------------------------------------------------

    #[test]
    fn test_generate_annotations() {
        let mut td = TestDependencies::new("test");
        let g = td.for_thread_mut(0);
        g.add_edge(Dependency::new(0, 1, DependencyKind::Data, DependencyStrength::Relaxed));
        let anns = generate_annotations(&td);
        assert_eq!(anns.len(), 1);
        assert_eq!(anns[0].thread, 0);
        assert_eq!(anns[0].source, 0);
        assert_eq!(anns[0].target, 1);
    }

    #[test]
    fn test_herd7_format() {
        let mut td = TestDependencies::new("MP");
        let g = td.for_thread_mut(0);
        g.add_edge(Dependency::new(0, 1, DependencyKind::Data, DependencyStrength::Relaxed));
        let output = format_herd7(&td);
        assert!(output.contains("data T0:0 T0:1"));
        assert!(output.contains("Dependencies for MP"));
    }

    #[test]
    fn test_dot_output() {
        let mut td = TestDependencies::new("LB");
        let g = td.for_thread_mut(0);
        g.add_edge(Dependency::new(0, 1, DependencyKind::Data, DependencyStrength::Relaxed));
        let dot = to_dot(&td);
        assert!(dot.contains("digraph dependencies"));
        assert!(dot.contains("Thread 0"));
        assert!(dot.contains("data"));
        assert!(dot.contains("red"));
    }

    #[test]
    fn test_summary_output() {
        let mut td = TestDependencies::new("test");
        let g = td.for_thread_mut(0);
        g.add_edge(Dependency::new(0, 1, DependencyKind::Data, DependencyStrength::Relaxed));
        g.add_edge(Dependency::new(1, 2, DependencyKind::Address, DependencyStrength::Relaxed));
        let s = summary(&td);
        assert!(s.contains("data: 1"));
        assert!(s.contains("addr: 1"));
        assert!(s.contains("total: 2"));
    }

    // -----------------------------------------------------------------------
    // Dependency display
    // -----------------------------------------------------------------------

    #[test]
    fn test_dependency_display() {
        let d = Dependency::new(0, 1, DependencyKind::Data, DependencyStrength::Acquire);
        let s = format!("{}", d);
        assert!(s.contains("data"));
        assert!(s.contains("acq"));
    }

    // -----------------------------------------------------------------------
    // Annotation display
    // -----------------------------------------------------------------------

    #[test]
    fn test_annotation_display() {
        let a = DependencyAnnotation {
            thread: 0,
            source: 1,
            target: 2,
            kind: DependencyKind::Control,
            strength: DependencyStrength::SeqCst,
        };
        let s = format!("{}", a);
        assert!(s.contains("T0"));
        assert!(s.contains("ctrl"));
        assert!(s.contains("sc"));
    }

    // -----------------------------------------------------------------------
    // SyncPattern display
    // -----------------------------------------------------------------------

    #[test]
    fn test_sync_pattern_display() {
        let p = SyncPattern::ReleaseAcquire {
            release_thread: 0,
            release_idx: 1,
            acquire_thread: 1,
            acquire_idx: 0,
            scope: Scope::None,
        };
        let s = format!("{}", p);
        assert!(s.contains("rel-acq"));
        assert!(s.contains("T0"));
        assert!(s.contains("T1"));
    }

    // -----------------------------------------------------------------------
    // Helper function tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_defined_reg_load() {
        let instr = make_load(5, 0x100, Ordering::Relaxed);
        assert_eq!(defined_reg(&instr), Some(5));
    }

    #[test]
    fn test_defined_reg_store() {
        let instr = make_store(0x100, 1, Ordering::Relaxed);
        assert_eq!(defined_reg(&instr), None);
    }

    #[test]
    fn test_is_read_write() {
        assert!(is_read(&make_load(1, 0x100, Ordering::Relaxed)));
        assert!(!is_read(&make_store(0x100, 1, Ordering::Relaxed)));
        assert!(is_read(&make_rmw(1, 0x100, 1, Ordering::Relaxed)));

        assert!(!is_write(&make_load(1, 0x100, Ordering::Relaxed)));
        assert!(is_write(&make_store(0x100, 1, Ordering::Relaxed)));
        assert!(is_write(&make_rmw(1, 0x100, 1, Ordering::Relaxed)));
    }

    #[test]
    fn test_accessed_address() {
        assert_eq!(
            accessed_address(&make_load(1, 0x100, Ordering::Relaxed)),
            Some(0x100),
        );
        assert_eq!(
            accessed_address(&make_store(0x200, 1, Ordering::Relaxed)),
            Some(0x200),
        );
        assert_eq!(accessed_address(&make_fence(Ordering::SeqCst, Scope::None)), None);
    }

    #[test]
    fn test_instr_ordering() {
        assert_eq!(
            instr_ordering(&make_load(1, 0x100, Ordering::Acquire)),
            Some(Ordering::Acquire),
        );
        assert_eq!(
            instr_ordering(&make_fence(Ordering::SeqCst, Scope::None)),
            Some(Ordering::SeqCst),
        );
        assert_eq!(instr_ordering(&make_label(0)), None);
    }

    // -----------------------------------------------------------------------
    // RMW dependency detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_rmw_data_dependency() {
        // r1 = load(x); rmw(y, r1) — data dep from load to rmw value
        let instrs = vec![
            make_load(1, 0x100, Ordering::Relaxed),
            make_rmw(2, 0x200, 1, Ordering::Relaxed),
        ];
        let graph = DependencyInference::infer_from_instructions(&instrs);
        assert!(graph.has_edge_kind(0, 1, DependencyKind::Data));
    }

    #[test]
    fn test_rmw_anti_dependency() {
        // rmw(x, 1) followed by store(x, 2) — WAR on x
        let instrs = vec![
            make_rmw(1, 0x100, 1, Ordering::Relaxed),
            make_store(0x100, 2, Ordering::Relaxed),
        ];
        let graph = DependencyInference::infer_from_instructions(&instrs);
        assert!(graph.has_edge_kind(0, 1, DependencyKind::AntiDependency));
    }

    #[test]
    fn test_rmw_output_dependency() {
        // rmw(x, 1) followed by store(x, 2) — WAW on x
        let instrs = vec![
            make_rmw(1, 0x100, 1, Ordering::Relaxed),
            make_store(0x100, 2, Ordering::Relaxed),
        ];
        let graph = DependencyInference::infer_from_instructions(&instrs);
        assert!(graph.has_edge_kind(0, 1, DependencyKind::Output));
    }

    // -----------------------------------------------------------------------
    // Complex patterns
    // -----------------------------------------------------------------------

    #[test]
    fn test_multiple_dependency_kinds_same_pair() {
        // r1 = load(x); store(x, r1) — data + anti dep on same pair
        let instrs = vec![
            make_load(1, 0x100, Ordering::Relaxed),
            make_store(0x100, 1, Ordering::Relaxed),
        ];
        let graph = DependencyInference::infer_from_instructions(&instrs);
        assert!(graph.has_edge_kind(0, 1, DependencyKind::Data));
        assert!(graph.has_edge_kind(0, 1, DependencyKind::AntiDependency));
    }

    #[test]
    fn test_fence_in_sequence() {
        // store(x,1); fence(rel); store(y,1)
        let instrs = vec![
            make_store(0x100, 1, Ordering::Relaxed),
            make_fence(Ordering::Release, Scope::None),
            make_store(0x200, 1, Ordering::Relaxed),
        ];
        let graph = DependencyInference::infer_from_instructions(&instrs);
        // No data/addr/ctrl deps expected; fence doesn't create intra-thread
        // data-flow dependencies.
        assert!(graph.edges_of_kind(DependencyKind::Data).is_empty());
    }

    #[test]
    fn test_gpu_scoped_release_acquire() {
        let test = LitmusTest {
            name: "GPU_MP".to_string(),
            threads: vec![
                Thread {
                    id: 0,
                    instructions: vec![
                        make_store(0x100, 1, Ordering::Relaxed),
                        make_store(0x200, 1, Ordering::ReleaseCTA),
                    ],
                },
                Thread {
                    id: 1,
                    instructions: vec![
                        make_load(1, 0x200, Ordering::AcquireCTA),
                        make_load(2, 0x100, Ordering::Relaxed),
                    ],
                },
            ],
            initial_state: HashMap::new(),
            expected_outcomes: Vec::new(),
        };
        let patterns = DependencyInference::recognize_cross_thread_sync(&test);
        assert!(!patterns.is_empty());
        match &patterns[0] {
            SyncPattern::ReleaseAcquire { scope, .. } => {
                assert_eq!(*scope, Scope::CTA);
            }
            _ => panic!("Expected ReleaseAcquire pattern"),
        }
    }

    // -----------------------------------------------------------------------
    // Empty input edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_instructions() {
        let graph = DependencyInference::infer_from_instructions(&[]);
        assert!(graph.is_empty());
    }

    #[test]
    fn test_single_instruction() {
        let instrs = vec![make_load(1, 0x100, Ordering::Relaxed)];
        let graph = DependencyInference::infer_from_instructions(&instrs);
        assert!(graph.is_empty()); // Single instruction, no deps.
    }

    #[test]
    fn test_empty_litmus_test() {
        let test = LitmusTest {
            name: "empty".to_string(),
            threads: Vec::new(),
            initial_state: HashMap::new(),
            expected_outcomes: Vec::new(),
        };
        let deps = DependencyInference::infer_from_litmus_test(&test);
        assert_eq!(deps.total_edges(), 0);
    }
}

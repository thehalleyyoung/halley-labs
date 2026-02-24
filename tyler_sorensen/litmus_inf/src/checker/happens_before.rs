/// Happens-before relation construction, transitive closure, race detection,
/// and synchronization analysis for LITMUS∞ memory model verification.
///
/// Implements program-order, reads-from, synchronization composition into
/// the happens-before partial order, with multiple transitive closure algorithms,
/// data race detection, vector clock tracking, and scope-aware analysis.
#[allow(unused)]

use std::collections::{HashMap, HashSet, BTreeMap, BTreeSet, VecDeque};
use std::fmt;

use super::execution::{EventId, ThreadId, Address, Value, OpType, Scope, Event, BitMatrix, ExecutionGraph, Relation};

// ═══════════════════════════════════════════════════════════════════════════
// HappensBeforeRelation
// ═══════════════════════════════════════════════════════════════════════════

/// Enumeration of the component relations that compose happens-before.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HbComponent {
    /// Program order within a thread.
    ProgramOrder,
    /// Reads-from: a write is seen by a read.
    ReadsFrom,
    /// Synchronization (barriers, fences, release/acquire).
    Synchronization,
    /// From-read (derived).
    FromRead,
    /// Coherence order.
    CoherenceOrder,
}

impl fmt::Display for HbComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HbComponent::ProgramOrder => write!(f, "po"),
            HbComponent::ReadsFrom => write!(f, "rf"),
            HbComponent::Synchronization => write!(f, "sync"),
            HbComponent::FromRead => write!(f, "fr"),
            HbComponent::CoherenceOrder => write!(f, "co"),
        }
    }
}

/// The happens-before relation for a memory model execution.
///
/// Composed from program order (po), reads-from (rf), synchronization
/// (sync), and optionally from-read (fr) and coherence order (co).
#[derive(Debug, Clone)]
pub struct HappensBeforeRelation {
    /// Number of events.
    n: usize,
    /// Program order relation.
    po: BitMatrix,
    /// Reads-from relation.
    rf: BitMatrix,
    /// Synchronization relation.
    sync_rel: BitMatrix,
    /// From-read relation.
    fr: BitMatrix,
    /// Coherence order.
    co: BitMatrix,
    /// The composed happens-before relation (transitive closure of union).
    hb: BitMatrix,
    /// Whether the HB relation has been computed.
    computed: bool,
    /// Component provenance for each edge.
    provenance: HashMap<(usize, usize), Vec<HbComponent>>,
}

impl HappensBeforeRelation {
    /// Create a new empty happens-before relation for n events.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            po: BitMatrix::new(n),
            rf: BitMatrix::new(n),
            sync_rel: BitMatrix::new(n),
            fr: BitMatrix::new(n),
            co: BitMatrix::new(n),
            hb: BitMatrix::new(n),
            computed: false,
            provenance: HashMap::new(),
        }
    }

    /// Build the happens-before relation from an execution graph.
    pub fn from_execution(exec: &ExecutionGraph) -> Self {
        let n = exec.events.len();
        let mut hb = Self::new(n);
        hb.po = exec.po.clone();
        hb.rf = exec.rf.clone();
        for rel in &exec.extra {
            if rel.name == "co" {
                hb.co = rel.matrix.clone();
            }
        }
        hb.compute_fr();
        hb.compute();
        hb
    }

    /// Add a program-order edge.
    pub fn add_po_edge(&mut self, from: usize, to: usize) {
        self.po.set(from, to, true);
        self.computed = false;
    }

    /// Add a reads-from edge.
    pub fn add_rf_edge(&mut self, write: usize, read: usize) {
        self.rf.set(write, read, true);
        self.computed = false;
    }

    /// Add a synchronization edge.
    pub fn add_sync_edge(&mut self, from: usize, to: usize) {
        self.sync_rel.set(from, to, true);
        self.computed = false;
    }

    /// Add a coherence-order edge.
    pub fn add_co_edge(&mut self, earlier: usize, later: usize) {
        self.co.set(earlier, later, true);
        self.computed = false;
    }

    /// Compute the from-read relation: fr = rf^{-1} ; co.
    fn compute_fr(&mut self) {
        let rf_inv = self.rf.inverse();
        self.fr = rf_inv.compose(&self.co);
    }

    /// Compute the happens-before relation as the transitive closure
    /// of the union of all component relations.
    pub fn compute(&mut self) {
        let mut combined = self.po.union(&self.rf);
        combined = combined.union(&self.sync_rel);
        combined = combined.union(&self.fr);
        combined = combined.union(&self.co);
        self.hb = combined.transitive_closure();
        self.computed = true;
        self.build_provenance();
    }

    /// Build provenance information mapping each HB edge to its source components.
    fn build_provenance(&mut self) {
        self.provenance.clear();
        for i in 0..self.n {
            for j in 0..self.n {
                if self.hb.get(i, j) {
                    let mut components = Vec::new();
                    if self.po.get(i, j) { components.push(HbComponent::ProgramOrder); }
                    if self.rf.get(i, j) { components.push(HbComponent::ReadsFrom); }
                    if self.sync_rel.get(i, j) { components.push(HbComponent::Synchronization); }
                    if self.fr.get(i, j) { components.push(HbComponent::FromRead); }
                    if self.co.get(i, j) { components.push(HbComponent::CoherenceOrder); }
                    if !components.is_empty() {
                        self.provenance.insert((i, j), components);
                    }
                }
            }
        }
    }

    /// Check whether event i happens-before event j.
    pub fn is_related(&self, i: usize, j: usize) -> bool {
        self.hb.get(i, j)
    }

    /// Get the underlying happens-before bit matrix.
    pub fn get_relation_matrix(&self) -> &BitMatrix {
        &self.hb
    }

    /// Get the program order matrix.
    pub fn get_po(&self) -> &BitMatrix { &self.po }

    /// Get the reads-from matrix.
    pub fn get_rf(&self) -> &BitMatrix { &self.rf }

    /// Get the synchronization matrix.
    pub fn get_sync(&self) -> &BitMatrix { &self.sync_rel }

    /// Get the from-read matrix.
    pub fn get_fr(&self) -> &BitMatrix { &self.fr }

    /// Get the coherence-order matrix.
    pub fn get_co(&self) -> &BitMatrix { &self.co }

    /// Number of events.
    pub fn size(&self) -> usize { self.n }

    /// Number of edges in the happens-before relation.
    pub fn edge_count(&self) -> usize { self.hb.count_edges() }

    /// Whether the relation has been computed.
    pub fn is_computed(&self) -> bool { self.computed }

    /// Get provenance of an edge (which components contributed).
    pub fn edge_provenance(&self, i: usize, j: usize) -> Option<&Vec<HbComponent>> {
        self.provenance.get(&(i, j))
    }

    /// Check whether the happens-before relation is acyclic.
    pub fn is_acyclic(&self) -> bool {
        for i in 0..self.n {
            if self.hb.get(i, i) {
                return false;
            }
        }
        true
    }

    /// Find all cycles in the happens-before relation.
    pub fn find_cycles(&self) -> Vec<Vec<usize>> {
        let mut cycles = Vec::new();
        let mut visited = vec![false; self.n];
        let mut on_stack = vec![false; self.n];
        let mut stack = Vec::new();
        for start in 0..self.n {
            if !visited[start] {
                self.dfs_cycles(start, &mut visited, &mut on_stack, &mut stack, &mut cycles);
            }
        }
        cycles
    }

    fn dfs_cycles(
        &self,
        node: usize,
        visited: &mut Vec<bool>,
        on_stack: &mut Vec<bool>,
        stack: &mut Vec<usize>,
        cycles: &mut Vec<Vec<usize>>,
    ) {
        visited[node] = true;
        on_stack[node] = true;
        stack.push(node);
        for succ in self.hb.successors(node) {
            if !visited[succ] {
                self.dfs_cycles(succ, visited, on_stack, stack, cycles);
            } else if on_stack[succ] {
                let pos = stack.iter().position(|&x| x == succ).unwrap_or(0);
                cycles.push(stack[pos..].to_vec());
            }
        }
        stack.pop();
        on_stack[node] = false;
    }

    /// Get all events that happen before a given event.
    pub fn predecessors(&self, event: usize) -> Vec<usize> {
        self.hb.predecessors(event).collect()
    }

    /// Get all events that happen after a given event.
    pub fn successors(&self, event: usize) -> Vec<usize> {
        self.hb.successors(event).collect()
    }

    /// Check whether two events are concurrent (neither ordered).
    pub fn are_concurrent(&self, a: usize, b: usize) -> bool {
        !self.hb.get(a, b) && !self.hb.get(b, a) && a != b
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TransitiveClosure — multiple algorithms
// ═══════════════════════════════════════════════════════════════════════════

/// Algorithm choice for transitive closure computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TcAlgorithm {
    /// Warshall's algorithm — O(n³).
    Warshall,
    /// Incremental — maintain closure under single-edge additions.
    Incremental,
    /// Semi-naive — iterative fixpoint with only-new-edges optimization.
    SemiNaive,
    /// Strassen-style via matrix multiplication.
    MatrixMultiply,
    /// Bit-parallel — uses u64 word-level parallelism.
    BitParallel,
}

impl fmt::Display for TcAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TcAlgorithm::Warshall => write!(f, "Warshall"),
            TcAlgorithm::Incremental => write!(f, "Incremental"),
            TcAlgorithm::SemiNaive => write!(f, "Semi-naive"),
            TcAlgorithm::MatrixMultiply => write!(f, "MatrixMultiply"),
            TcAlgorithm::BitParallel => write!(f, "BitParallel"),
        }
    }
}

/// Engine for computing and maintaining transitive closures.
#[derive(Debug, Clone)]
pub struct TransitiveClosureEngine {
    /// The current relation matrix.
    relation: BitMatrix,
    /// The computed transitive closure.
    closure: BitMatrix,
    /// Whether the closure is up to date.
    up_to_date: bool,
    /// Statistics on operations performed.
    stats: TcStats,
}

/// Statistics for transitive closure operations.
#[derive(Debug, Clone, Default)]
pub struct TcStats {
    /// Number of closure computations.
    pub computations: u64,
    /// Number of incremental updates.
    pub incremental_updates: u64,
    /// Total edges processed.
    pub edges_processed: u64,
    /// Number of iterations in fixpoint algorithms.
    pub iterations: u64,
}

impl TransitiveClosureEngine {
    /// Create a new engine from a relation matrix.
    pub fn new(relation: BitMatrix) -> Self {
        let n = relation.dim();
        Self {
            relation,
            closure: BitMatrix::new(n),
            up_to_date: false,
            stats: TcStats::default(),
        }
    }

    /// Create from a dimension (empty relation).
    pub fn empty(n: usize) -> Self {
        Self::new(BitMatrix::new(n))
    }

    /// Get the current closure. Computes if not up to date using Warshall.
    pub fn get_closure(&mut self) -> &BitMatrix {
        if !self.up_to_date {
            self.compute_warshall();
        }
        &self.closure
    }

    /// Get statistics.
    pub fn stats(&self) -> &TcStats { &self.stats }

    /// Compute transitive closure using Warshall's algorithm.
    pub fn compute_warshall(&mut self) -> &BitMatrix {
        let n = self.relation.dim();
        self.closure = self.relation.clone();
        for k in 0..n {
            for i in 0..n {
                if self.closure.get(i, k) {
                    for j in 0..n {
                        if self.closure.get(k, j) {
                            self.closure.set(i, j, true);
                        }
                    }
                }
            }
        }
        self.up_to_date = true;
        self.stats.computations += 1;
        &self.closure
    }

    /// Add an edge and update the closure incrementally.
    ///
    /// When adding edge (u, v), any pair (x, y) where x ->* u and v ->* y
    /// becomes related in the closure.
    pub fn add_edge_incremental(&mut self, u: usize, v: usize) {
        self.relation.set(u, v, true);
        if self.closure.get(u, v) {
            return;
        }
        let n = self.relation.dim();
        // Collect predecessors of u (including u) and successors of v (including v)
        let mut preds: Vec<usize> = vec![u];
        for i in 0..n {
            if self.closure.get(i, u) {
                preds.push(i);
            }
        }
        let mut succs: Vec<usize> = vec![v];
        for j in 0..n {
            if self.closure.get(v, j) {
                succs.push(j);
            }
        }
        for &x in &preds {
            for &y in &succs {
                self.closure.set(x, y, true);
            }
        }
        self.stats.incremental_updates += 1;
        self.stats.edges_processed += (preds.len() * succs.len()) as u64;
    }

    /// Compute transitive closure using semi-naive iteration.
    ///
    /// At each step, only the newly discovered edges are used to find
    /// further reachability, avoiding redundant work.
    pub fn compute_semi_naive(&mut self) -> &BitMatrix {
        let n = self.relation.dim();
        self.closure = self.relation.clone();
        let mut delta = self.relation.clone();
        let mut iteration = 0u64;
        loop {
            let new_edges = delta.compose(&self.relation);
            let mut next_delta = BitMatrix::new(n);
            let mut changed = false;
            for i in 0..n {
                for j in 0..n {
                    if new_edges.get(i, j) && !self.closure.get(i, j) {
                        self.closure.set(i, j, true);
                        next_delta.set(i, j, true);
                        changed = true;
                    }
                }
            }
            iteration += 1;
            if !changed {
                break;
            }
            delta = next_delta;
        }
        self.up_to_date = true;
        self.stats.computations += 1;
        self.stats.iterations += iteration;
        &self.closure
    }

    /// Compute transitive closure via repeated squaring (matrix multiplication).
    ///
    /// Computes R ∪ R² ∪ R⁴ ∪ ... until no new edges appear.
    pub fn compute_matrix_multiply(&mut self) -> &BitMatrix {
        let n = self.relation.dim();
        self.closure = self.relation.clone();
        let mut power = self.relation.clone();
        let mut iteration = 0u64;
        loop {
            power = power.compose(&power);
            let combined = self.closure.union(&power);
            if combined.count_edges() == self.closure.count_edges() {
                break;
            }
            self.closure = combined;
            iteration += 1;
            if iteration > (n as u64) {
                break;
            }
        }
        self.up_to_date = true;
        self.stats.computations += 1;
        self.stats.iterations += iteration;
        &self.closure
    }

    /// Compute transitive closure using bit-parallel operations.
    ///
    /// Uses the BitMatrix's internal u64 representation for word-level
    /// parallelism in the inner loop of Warshall's algorithm.
    pub fn compute_bit_parallel(&mut self) -> &BitMatrix {
        self.closure = self.relation.transitive_closure();
        self.up_to_date = true;
        self.stats.computations += 1;
        &self.closure
    }

    /// Run all algorithms and compare results (for testing).
    pub fn benchmark_all(&mut self) -> BTreeMap<String, (usize, u64)> {
        let mut results = BTreeMap::new();
        let saved = self.relation.clone();

        self.compute_warshall();
        let warshall_edges = self.closure.count_edges();
        results.insert("Warshall".to_string(), (warshall_edges, self.stats.computations));

        self.relation = saved.clone();
        self.up_to_date = false;
        self.compute_semi_naive();
        let semi_edges = self.closure.count_edges();
        results.insert("SemiNaive".to_string(), (semi_edges, self.stats.computations));

        self.relation = saved.clone();
        self.up_to_date = false;
        self.compute_matrix_multiply();
        let mm_edges = self.closure.count_edges();
        results.insert("MatrixMultiply".to_string(), (mm_edges, self.stats.computations));

        self.relation = saved.clone();
        self.up_to_date = false;
        self.compute_bit_parallel();
        let bp_edges = self.closure.count_edges();
        results.insert("BitParallel".to_string(), (bp_edges, self.stats.computations));

        self.relation = saved;
        results
    }

    /// Check if the current closure is valid (idempotent under composition).
    pub fn validate_closure(&self) -> bool {
        let composed = self.closure.compose(&self.closure);
        let n = self.closure.dim();
        for i in 0..n {
            for j in 0..n {
                if composed.get(i, j) && !self.closure.get(i, j) {
                    return false;
                }
            }
        }
        true
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// RaceDetector
// ═══════════════════════════════════════════════════════════════════════════

/// Classification of a data race.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RaceType {
    /// Read–Write race.
    ReadWrite,
    /// Write–Read race.
    WriteRead,
    /// Write–Write race.
    WriteWrite,
}

impl fmt::Display for RaceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RaceType::ReadWrite => write!(f, "R-W"),
            RaceType::WriteRead => write!(f, "W-R"),
            RaceType::WriteWrite => write!(f, "W-W"),
        }
    }
}

/// A data race between two events.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DataRace {
    /// First event in the race.
    pub event_a: usize,
    /// Second event in the race.
    pub event_b: usize,
    /// The memory address involved.
    pub address: Address,
    /// Classification of the race.
    pub race_type: RaceType,
    /// Thread of event_a.
    pub thread_a: ThreadId,
    /// Thread of event_b.
    pub thread_b: ThreadId,
}

impl fmt::Display for DataRace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Race({} E{}×E{} @{:#x} T{}↔T{})",
            self.race_type, self.event_a, self.event_b,
            self.address, self.thread_a, self.thread_b)
    }
}

/// Detector for data races in execution graphs.
#[derive(Debug, Clone)]
pub struct RaceDetector {
    /// Happens-before relation used for ordering.
    hb: BitMatrix,
    /// Events in the execution.
    events: Vec<Event>,
    /// Events grouped by address.
    events_by_address: HashMap<Address, Vec<usize>>,
    /// Detected races.
    races: Vec<DataRace>,
}

impl RaceDetector {
    /// Create a new race detector.
    pub fn new(events: Vec<Event>, hb: BitMatrix) -> Self {
        let mut events_by_address: HashMap<Address, Vec<usize>> = HashMap::new();
        for (idx, ev) in events.iter().enumerate() {
            events_by_address.entry(ev.address).or_default().push(idx);
        }
        Self {
            hb,
            events,
            events_by_address,
            races: Vec::new(),
        }
    }

    /// Detect all data races in the execution.
    pub fn detect_races(&mut self) -> &[DataRace] {
        self.races.clear();
        let addresses: Vec<Address> = self.events_by_address.keys().copied().collect();
        for addr in addresses {
            self.detect_races_on_address(addr);
        }
        &self.races
    }

    /// Detect races involving a specific address.
    pub fn detect_races_on_address(&mut self, addr: Address) {
        let event_ids = match self.events_by_address.get(&addr) {
            Some(ids) => ids.clone(),
            None => return,
        };
        for i in 0..event_ids.len() {
            for j in (i + 1)..event_ids.len() {
                let a = event_ids[i];
                let b = event_ids[j];
                if self.is_conflicting(a, b) && self.is_concurrent(a, b) {
                    if let Some(race) = self.classify_race(a, b, addr) {
                        self.races.push(race);
                    }
                }
            }
        }
    }

    /// Check whether two events are conflicting (at least one write, same address).
    pub fn is_conflicting(&self, a: usize, b: usize) -> bool {
        let ea = &self.events[a];
        let eb = &self.events[b];
        ea.address == eb.address
            && ea.thread != eb.thread
            && (ea.is_write() || eb.is_write())
    }

    /// Check whether two events are concurrent (not ordered by HB).
    pub fn is_concurrent(&self, a: usize, b: usize) -> bool {
        !self.hb.get(a, b) && !self.hb.get(b, a)
    }

    /// Classify the type of race between two events.
    pub fn classify_race(&self, a: usize, b: usize, addr: Address) -> Option<DataRace> {
        let ea = &self.events[a];
        let eb = &self.events[b];
        let race_type = match (ea.is_write(), eb.is_write()) {
            (true, true) => RaceType::WriteWrite,
            (true, false) => RaceType::WriteRead,
            (false, true) => RaceType::ReadWrite,
            (false, false) => return None,
        };
        Some(DataRace {
            event_a: a,
            event_b: b,
            address: addr,
            race_type,
            thread_a: ea.thread,
            thread_b: eb.thread,
        })
    }

    /// Get all detected races.
    pub fn get_races(&self) -> &[DataRace] { &self.races }

    /// Get races filtered by type.
    pub fn races_by_type(&self, ty: RaceType) -> Vec<&DataRace> {
        self.races.iter().filter(|r| r.race_type == ty).collect()
    }

    /// Get races filtered by address.
    pub fn races_at_address(&self, addr: Address) -> Vec<&DataRace> {
        self.races.iter().filter(|r| r.address == addr).collect()
    }

    /// Get races filtered by thread pair.
    pub fn races_between_threads(&self, t1: ThreadId, t2: ThreadId) -> Vec<&DataRace> {
        self.races.iter().filter(|r| {
            (r.thread_a == t1 && r.thread_b == t2) || (r.thread_a == t2 && r.thread_b == t1)
        }).collect()
    }

    /// Summary of detected races.
    pub fn race_summary(&self) -> RaceSummary {
        let mut by_type: HashMap<RaceType, usize> = HashMap::new();
        let mut by_address: HashMap<Address, usize> = HashMap::new();
        let mut thread_pairs: HashSet<(ThreadId, ThreadId)> = HashSet::new();
        for race in &self.races {
            *by_type.entry(race.race_type).or_default() += 1;
            *by_address.entry(race.address).or_default() += 1;
            let pair = if race.thread_a < race.thread_b {
                (race.thread_a, race.thread_b)
            } else {
                (race.thread_b, race.thread_a)
            };
            thread_pairs.insert(pair);
        }
        RaceSummary {
            total_races: self.races.len(),
            by_type,
            by_address,
            thread_pairs: thread_pairs.into_iter().collect(),
        }
    }
}

/// Summary of race detection results.
#[derive(Debug, Clone)]
pub struct RaceSummary {
    /// Total number of races detected.
    pub total_races: usize,
    /// Races grouped by type.
    pub by_type: HashMap<RaceType, usize>,
    /// Races grouped by address.
    pub by_address: HashMap<Address, usize>,
    /// Thread pairs involved in races.
    pub thread_pairs: Vec<(ThreadId, ThreadId)>,
}

impl fmt::Display for RaceSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Race Summary: {} total races", self.total_races)?;
        for (ty, count) in &self.by_type {
            writeln!(f, "  {}: {}", ty, count)?;
        }
        writeln!(f, "  Addresses: {}", self.by_address.len())?;
        writeln!(f, "  Thread pairs: {}", self.thread_pairs.len())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SynchronizationAnalysis
// ═══════════════════════════════════════════════════════════════════════════

/// Classification of synchronization operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SyncPattern {
    /// A barrier synchronization.
    Barrier,
    /// Lock acquisition.
    Lock,
    /// Lock release.
    Unlock,
    /// Memory fence.
    Fence,
    /// Atomic read-modify-write.
    AtomicRMW,
    /// Release semantics.
    Release,
    /// Acquire semantics.
    Acquire,
    /// Acquire-release semantics.
    AcquireRelease,
    /// Sequentially consistent.
    SeqCst,
}

impl fmt::Display for SyncPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SyncPattern::Barrier => write!(f, "barrier"),
            SyncPattern::Lock => write!(f, "lock"),
            SyncPattern::Unlock => write!(f, "unlock"),
            SyncPattern::Fence => write!(f, "fence"),
            SyncPattern::AtomicRMW => write!(f, "atomic_rmw"),
            SyncPattern::Release => write!(f, "release"),
            SyncPattern::Acquire => write!(f, "acquire"),
            SyncPattern::AcquireRelease => write!(f, "acq_rel"),
            SyncPattern::SeqCst => write!(f, "seq_cst"),
        }
    }
}

/// A synchronization event in an execution.
#[derive(Debug, Clone)]
pub struct SyncEvent {
    /// The event index.
    pub event_id: usize,
    /// The thread that performed the synchronization.
    pub thread: ThreadId,
    /// The synchronization pattern.
    pub pattern: SyncPattern,
    /// The scope of the synchronization.
    pub scope: Scope,
    /// Optional address (for lock/unlock operations).
    pub address: Option<Address>,
}

/// Analyzer for synchronization patterns in executions.
#[derive(Debug, Clone)]
pub struct SyncAnalyzer {
    /// Events in the execution.
    events: Vec<Event>,
    /// Extracted synchronization events.
    sync_events: Vec<SyncEvent>,
    /// Synchronization relation (sync edges).
    sync_relation: BitMatrix,
}

impl SyncAnalyzer {
    /// Create a new synchronization analyzer.
    pub fn new(events: Vec<Event>) -> Self {
        let n = events.len();
        Self {
            events,
            sync_events: Vec::new(),
            sync_relation: BitMatrix::new(n),
        }
    }

    /// Extract synchronization events from the execution.
    pub fn extract_sync_events(&mut self) -> &[SyncEvent] {
        self.sync_events.clear();
        for (idx, ev) in self.events.iter().enumerate() {
            let pattern = match ev.op_type {
                OpType::Fence => Some(SyncPattern::Fence),
                OpType::RMW => Some(SyncPattern::AtomicRMW),
                _ => None,
            };
            if let Some(pat) = pattern {
                self.sync_events.push(SyncEvent {
                    event_id: idx,
                    thread: ev.thread,
                    pattern: pat,
                    scope: ev.scope,
                    address: Some(ev.address),
                });
            }
        }
        &self.sync_events
    }

    /// Compute the synchronization relation from extracted events.
    pub fn compute_sync_relation(&mut self) -> &BitMatrix {
        let n = self.events.len();
        self.sync_relation = BitMatrix::new(n);
        // Release-acquire pairs: a release on thread T1 synchronizes with
        // an acquire on thread T2 if they access the same address.
        let releases: Vec<&SyncEvent> = self.sync_events.iter()
            .filter(|s| matches!(s.pattern, SyncPattern::Release | SyncPattern::AcquireRelease | SyncPattern::SeqCst))
            .collect();
        let acquires: Vec<&SyncEvent> = self.sync_events.iter()
            .filter(|s| matches!(s.pattern, SyncPattern::Acquire | SyncPattern::AcquireRelease | SyncPattern::SeqCst))
            .collect();
        for rel in &releases {
            for acq in &acquires {
                if rel.thread != acq.thread && rel.address == acq.address {
                    self.sync_relation.set(rel.event_id, acq.event_id, true);
                }
            }
        }
        // Barrier pairs: all events before a barrier HB all events after.
        let barriers: Vec<&SyncEvent> = self.sync_events.iter()
            .filter(|s| s.pattern == SyncPattern::Barrier)
            .collect();
        for i in 0..barriers.len() {
            for j in (i + 1)..barriers.len() {
                let b1 = barriers[i];
                let b2 = barriers[j];
                if b1.thread != b2.thread {
                    self.sync_relation.set(b1.event_id, b2.event_id, true);
                    self.sync_relation.set(b2.event_id, b1.event_id, true);
                }
            }
        }
        &self.sync_relation
    }

    /// Find release-acquire synchronization pairs.
    pub fn find_sync_pairs(&self) -> Vec<(usize, usize, SyncPattern)> {
        let mut pairs = Vec::new();
        let n = self.sync_relation.dim();
        for i in 0..n {
            for j in self.sync_relation.successors(i) {
                let pattern = self.sync_events.iter()
                    .find(|s| s.event_id == i)
                    .map(|s| s.pattern)
                    .unwrap_or(SyncPattern::Fence);
                pairs.push((i, j, pattern));
            }
        }
        pairs
    }

    /// Compute synchronization coverage: fraction of cross-thread pairs ordered.
    pub fn sync_coverage(&self, hb: &BitMatrix) -> f64 {
        let n = self.events.len();
        let mut cross_thread_pairs = 0u64;
        let mut ordered_pairs = 0u64;
        for i in 0..n {
            for j in 0..n {
                if self.events[i].thread != self.events[j].thread && i != j {
                    cross_thread_pairs += 1;
                    if hb.get(i, j) || hb.get(j, i) {
                        ordered_pairs += 1;
                    }
                }
            }
        }
        if cross_thread_pairs == 0 { 1.0 } else { ordered_pairs as f64 / cross_thread_pairs as f64 }
    }

    /// Detect potentially redundant synchronization operations.
    pub fn redundant_sync_detection(&self, hb: &BitMatrix) -> Vec<usize> {
        let mut redundant = Vec::new();
        for sync_ev in &self.sync_events {
            let eid = sync_ev.event_id;
            // A sync is redundant if removing its edges doesn't change reachability
            // Simplified: check if all pairs ordered through this sync are also
            // ordered through other paths.
            let mut is_redundant = true;
            for pred in hb.predecessors(eid) {
                for succ in hb.successors(eid) {
                    // Check if pred -> succ is supported by other edges
                    let mut has_alt_path = false;
                    for mid in hb.successors(pred) {
                        if mid != eid && hb.get(mid, succ) {
                            has_alt_path = true;
                            break;
                        }
                    }
                    if !has_alt_path {
                        is_redundant = false;
                        break;
                    }
                }
                if !is_redundant { break; }
            }
            if is_redundant {
                redundant.push(eid);
            }
        }
        redundant
    }

    /// Get sync events.
    pub fn get_sync_events(&self) -> &[SyncEvent] { &self.sync_events }

    /// Count of sync events by pattern.
    pub fn sync_pattern_counts(&self) -> HashMap<SyncPattern, usize> {
        let mut counts = HashMap::new();
        for se in &self.sync_events {
            *counts.entry(se.pattern).or_default() += 1;
        }
        counts
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CausalityChecker
// ═══════════════════════════════════════════════════════════════════════════

/// Result of a causality check.
#[derive(Debug, Clone)]
pub enum CausalityViolation {
    /// A cycle was found in the causal order.
    CausalCycle(Vec<usize>),
    /// A thin-air read was detected (a read whose value is not justified).
    ThinAirRead {
        /// The read event.
        read_event: usize,
        /// The value read.
        value: Value,
    },
    /// An unjustified rf edge.
    UnjustifiedRf {
        /// The write event.
        write_event: usize,
        /// The read event.
        read_event: usize,
    },
}

impl fmt::Display for CausalityViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CausalityViolation::CausalCycle(cycle) => {
                write!(f, "Causal cycle: {:?}", cycle)
            }
            CausalityViolation::ThinAirRead { read_event, value } => {
                write!(f, "Thin-air read: E{} = {}", read_event, value)
            }
            CausalityViolation::UnjustifiedRf { write_event, read_event } => {
                write!(f, "Unjustified rf: E{} -> E{}", write_event, read_event)
            }
        }
    }
}

/// Checker for causality constraints in executions.
#[derive(Debug, Clone)]
pub struct CausalityChecker {
    /// Events.
    events: Vec<Event>,
    /// Reads-from relation.
    rf: BitMatrix,
    /// Program-order relation.
    po: BitMatrix,
    /// Coherence-order relation.
    co: BitMatrix,
    /// Detected violations.
    violations: Vec<CausalityViolation>,
}

impl CausalityChecker {
    /// Create a new causality checker.
    pub fn new(events: Vec<Event>, rf: BitMatrix, po: BitMatrix, co: BitMatrix) -> Self {
        Self {
            events,
            rf,
            po,
            co,
            violations: Vec::new(),
        }
    }

    /// Run all causality checks.
    pub fn check_causality(&mut self) -> &[CausalityViolation] {
        self.violations.clear();
        self.find_causal_cycles();
        self.detect_thin_air();
        self.validate_rf_justification();
        &self.violations
    }

    /// Find cycles in the causal dependency graph.
    pub fn find_causal_cycles(&mut self) {
        // Causal order = po ∪ rf
        let causal = self.po.union(&self.rf);
        let closure = causal.transitive_closure();
        let n = self.events.len();
        for i in 0..n {
            if closure.get(i, i) {
                // Found a cycle — extract it
                let cycle = self.extract_cycle(&causal, i);
                self.violations.push(CausalityViolation::CausalCycle(cycle));
            }
        }
    }

    /// Extract a cycle starting from a given node.
    fn extract_cycle(&self, relation: &BitMatrix, start: usize) -> Vec<usize> {
        let mut cycle = vec![start];
        let mut current = start;
        let mut visited = HashSet::new();
        visited.insert(start);
        loop {
            let mut found_next = false;
            for succ in relation.successors(current) {
                if succ == start && cycle.len() > 1 {
                    return cycle;
                }
                if !visited.contains(&succ) {
                    visited.insert(succ);
                    cycle.push(succ);
                    current = succ;
                    found_next = true;
                    break;
                }
            }
            if !found_next {
                break;
            }
        }
        cycle
    }

    /// Detect thin-air reads: reads whose value has no justifying write.
    pub fn detect_thin_air(&mut self) {
        let n = self.events.len();
        for i in 0..n {
            if self.events[i].is_read() {
                // Check if there is any write that rf-writes to this read
                let has_source = (0..n).any(|w| self.rf.get(w, i));
                if !has_source && self.events[i].value != 0 {
                    self.violations.push(CausalityViolation::ThinAirRead {
                        read_event: i,
                        value: self.events[i].value,
                    });
                }
            }
        }
    }

    /// Validate that each rf edge is justified.
    pub fn validate_rf_justification(&mut self) {
        let n = self.events.len();
        for w in 0..n {
            for r in 0..n {
                if self.rf.get(w, r) {
                    // Write must be to the same address as the read
                    if self.events[w].address != self.events[r].address {
                        self.violations.push(CausalityViolation::UnjustifiedRf {
                            write_event: w,
                            read_event: r,
                        });
                    }
                    // Write value must equal read value
                    if self.events[w].value != self.events[r].value {
                        self.violations.push(CausalityViolation::UnjustifiedRf {
                            write_event: w,
                            read_event: r,
                        });
                    }
                }
            }
        }
    }

    /// Build the causal dependency graph as a BitMatrix.
    pub fn causal_dependency_graph(&self) -> BitMatrix {
        let mut deps = self.po.union(&self.rf);
        // Add data dependencies: read -> subsequent write to same address
        let n = self.events.len();
        for r in 0..n {
            if self.events[r].is_read() {
                for w in 0..n {
                    if self.events[w].is_write()
                        && self.events[w].thread == self.events[r].thread
                        && self.po.get(r, w)
                    {
                        deps.set(r, w, true);
                    }
                }
            }
        }
        deps
    }

    /// Get violations.
    pub fn get_violations(&self) -> &[CausalityViolation] { &self.violations }

    /// Check whether causality holds (no violations).
    pub fn is_causal(&self) -> bool { self.violations.is_empty() }
}

// ═══════════════════════════════════════════════════════════════════════════
// VectorClockTracker
// ═══════════════════════════════════════════════════════════════════════════

/// A vector clock mapping thread IDs to logical timestamps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorClock {
    /// Map from thread ID to timestamp.
    clocks: HashMap<ThreadId, u64>,
}

impl VectorClock {
    /// Create a new empty vector clock.
    pub fn new() -> Self {
        Self { clocks: HashMap::new() }
    }

    /// Create with a specific thread initialized to 0.
    pub fn for_thread(tid: ThreadId) -> Self {
        let mut vc = Self::new();
        vc.clocks.insert(tid, 0);
        vc
    }

    /// Get the clock value for a thread.
    pub fn get(&self, tid: ThreadId) -> u64 {
        self.clocks.get(&tid).copied().unwrap_or(0)
    }

    /// Increment the clock for a thread.
    pub fn increment(&mut self, tid: ThreadId) {
        let entry = self.clocks.entry(tid).or_insert(0);
        *entry += 1;
    }

    /// Join (pointwise maximum) with another vector clock.
    pub fn join(&mut self, other: &VectorClock) {
        for (&tid, &ts) in &other.clocks {
            let entry = self.clocks.entry(tid).or_insert(0);
            *entry = (*entry).max(ts);
        }
    }

    /// Meet (pointwise minimum) with another vector clock.
    pub fn meet(&mut self, other: &VectorClock) {
        let all_threads: HashSet<ThreadId> = self.clocks.keys()
            .chain(other.clocks.keys())
            .copied()
            .collect();
        for tid in all_threads {
            let a = self.get(tid);
            let b = other.get(tid);
            self.clocks.insert(tid, a.min(b));
        }
    }

    /// Check if this clock happens-before another: ∀t. self[t] ≤ other[t].
    pub fn happens_before(&self, other: &VectorClock) -> bool {
        let all_threads: HashSet<ThreadId> = self.clocks.keys()
            .chain(other.clocks.keys())
            .copied()
            .collect();
        for tid in all_threads {
            if self.get(tid) > other.get(tid) {
                return false;
            }
        }
        true
    }

    /// Check if this clock is concurrent with another.
    pub fn concurrent(&self, other: &VectorClock) -> bool {
        !self.happens_before(other) && !other.happens_before(self)
    }

    /// Check if this clock is strictly before another.
    pub fn strictly_before(&self, other: &VectorClock) -> bool {
        self.happens_before(other) && self != other
    }

    /// Merge with another clock (same as join).
    pub fn merge(&mut self, other: &VectorClock) {
        self.join(other);
    }

    /// Number of threads tracked.
    pub fn num_threads(&self) -> usize { self.clocks.len() }

    /// Get all thread IDs.
    pub fn threads(&self) -> Vec<ThreadId> {
        self.clocks.keys().copied().collect()
    }
}

impl Default for VectorClock {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for VectorClock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut entries: Vec<_> = self.clocks.iter().collect();
        entries.sort_by_key(|(&tid, _)| tid);
        write!(f, "VC[")?;
        for (i, (tid, ts)) in entries.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "T{}:{}", tid, ts)?;
        }
        write!(f, "]")
    }
}

/// System of vector clocks for dynamic happens-before tracking.
#[derive(Debug, Clone)]
pub struct VectorClockSystem {
    /// Per-thread vector clocks.
    thread_clocks: HashMap<ThreadId, VectorClock>,
    /// Per-variable (address) last-write clock.
    write_clocks: HashMap<Address, (ThreadId, VectorClock)>,
    /// Per-variable last-read clocks (per-thread).
    read_clocks: HashMap<Address, HashMap<ThreadId, VectorClock>>,
    /// Detected races.
    races: Vec<DataRace>,
    /// Event counter.
    event_counter: u64,
}

impl VectorClockSystem {
    /// Create a new vector clock system.
    pub fn new() -> Self {
        Self {
            thread_clocks: HashMap::new(),
            write_clocks: HashMap::new(),
            read_clocks: HashMap::new(),
            races: Vec::new(),
            event_counter: 0,
        }
    }

    /// Initialize a thread's clock.
    pub fn init_thread(&mut self, tid: ThreadId) {
        self.thread_clocks.insert(tid, VectorClock::for_thread(tid));
    }

    /// Get the clock for a thread.
    pub fn get_clock(&self, tid: ThreadId) -> Option<&VectorClock> {
        self.thread_clocks.get(&tid)
    }

    /// Process a read event.
    pub fn process_read(&mut self, tid: ThreadId, addr: Address) {
        self.event_counter += 1;
        let tc = self.thread_clocks.entry(tid).or_insert_with(|| VectorClock::for_thread(tid));
        tc.increment(tid);

        // Check for W-R race
        if let Some((writer_tid, writer_vc)) = self.write_clocks.get(&addr) {
            if *writer_tid != tid && !writer_vc.happens_before(tc) {
                self.races.push(DataRace {
                    event_a: (self.event_counter - 1) as usize,
                    event_b: self.event_counter as usize,
                    address: addr,
                    race_type: RaceType::WriteRead,
                    thread_a: *writer_tid,
                    thread_b: tid,
                });
            }
        }

        // Record this read
        let tc_clone = tc.clone();
        self.read_clocks.entry(addr).or_default().insert(tid, tc_clone);
    }

    /// Process a write event.
    pub fn process_write(&mut self, tid: ThreadId, addr: Address) {
        self.event_counter += 1;
        let tc = self.thread_clocks.entry(tid).or_insert_with(|| VectorClock::for_thread(tid));
        tc.increment(tid);

        // Check for R-W races
        if let Some(readers) = self.read_clocks.get(&addr) {
            for (&reader_tid, reader_vc) in readers {
                if reader_tid != tid && !reader_vc.happens_before(tc) {
                    self.races.push(DataRace {
                        event_a: (self.event_counter - 1) as usize,
                        event_b: self.event_counter as usize,
                        address: addr,
                        race_type: RaceType::ReadWrite,
                        thread_a: reader_tid,
                        thread_b: tid,
                    });
                }
            }
        }

        // Check for W-W race
        if let Some((writer_tid, writer_vc)) = self.write_clocks.get(&addr) {
            if *writer_tid != tid && !writer_vc.happens_before(tc) {
                self.races.push(DataRace {
                    event_a: (self.event_counter - 1) as usize,
                    event_b: self.event_counter as usize,
                    address: addr,
                    race_type: RaceType::WriteWrite,
                    thread_a: *writer_tid,
                    thread_b: tid,
                });
            }
        }

        // Record this write
        let tc_clone = tc.clone();
        self.write_clocks.insert(addr, (tid, tc_clone));
    }

    /// Process a synchronization (release on tid1, acquire on tid2).
    pub fn process_sync(&mut self, releaser: ThreadId, acquirer: ThreadId) {
        let release_vc = self.thread_clocks.get(&releaser).cloned()
            .unwrap_or_else(VectorClock::new);
        let acq_clock = self.thread_clocks.entry(acquirer)
            .or_insert_with(|| VectorClock::for_thread(acquirer));
        acq_clock.join(&release_vc);
    }

    /// Process a barrier (all participating threads synchronize).
    pub fn process_barrier(&mut self, threads: &[ThreadId]) {
        // Compute the join of all clocks
        let mut joined = VectorClock::new();
        for &tid in threads {
            if let Some(vc) = self.thread_clocks.get(&tid) {
                joined.join(vc);
            }
        }
        // Set all clocks to the joined value
        for &tid in threads {
            let tc = self.thread_clocks.entry(tid)
                .or_insert_with(|| VectorClock::for_thread(tid));
            tc.join(&joined);
        }
    }

    /// Get detected races.
    pub fn get_races(&self) -> &[DataRace] { &self.races }

    /// Reset the system.
    pub fn reset(&mut self) {
        self.thread_clocks.clear();
        self.write_clocks.clear();
        self.read_clocks.clear();
        self.races.clear();
        self.event_counter = 0;
    }
}

impl Default for VectorClockSystem {
    fn default() -> Self { Self::new() }
}

// ═══════════════════════════════════════════════════════════════════════════
// HappensBeforeGraph — graph representation and analysis
// ═══════════════════════════════════════════════════════════════════════════

/// A graph representation of the happens-before relation
/// with analysis and visualization support.
#[derive(Debug, Clone)]
pub struct HappensBeforeGraph {
    /// Number of nodes.
    n: usize,
    /// Adjacency list representation.
    adj: Vec<Vec<usize>>,
    /// Reverse adjacency list.
    rev_adj: Vec<Vec<usize>>,
    /// Node labels.
    labels: Vec<String>,
    /// Edge labels (optional annotation per edge).
    edge_labels: HashMap<(usize, usize), String>,
}

impl HappensBeforeGraph {
    /// Create from a BitMatrix.
    pub fn from_matrix(matrix: &BitMatrix) -> Self {
        let n = matrix.dim();
        let mut adj = vec![Vec::new(); n];
        let mut rev_adj = vec![Vec::new(); n];
        for (i, j) in matrix.edges() {
            adj[i].push(j);
            rev_adj[j].push(i);
        }
        Self {
            n,
            adj,
            rev_adj,
            labels: (0..n).map(|i| format!("E{}", i)).collect(),
            edge_labels: HashMap::new(),
        }
    }

    /// Create from a HappensBeforeRelation.
    pub fn from_hb(hb: &HappensBeforeRelation) -> Self {
        Self::from_matrix(hb.get_relation_matrix())
    }

    /// Set node labels.
    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        self.labels = labels;
        self
    }

    /// Add an edge label.
    pub fn add_edge_label(&mut self, from: usize, to: usize, label: String) {
        self.edge_labels.insert((from, to), label);
    }

    /// Generate DOT format for Graphviz visualization.
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph HappensBefore {
");
        dot.push_str("  rankdir=TB;
");
        dot.push_str("  node [shape=box];
");
        for i in 0..self.n {
            dot.push_str(&format!("  {} [label=\"{}\"];\n", i, self.labels[i]));
        }
        for i in 0..self.n {
            for &j in &self.adj[i] {
                if let Some(label) = self.edge_labels.get(&(i, j)) {
                    dot.push_str(&format!("  {} -> {} [label=\"{}\"];\n", i, j, label));
                } else {
                    dot.push_str(&format!("  {} -> {};
", i, j));
                }
            }
        }
        dot.push_str("}
");
        dot
    }

    /// Topological sort (Kahn's algorithm).
    pub fn topological_sort(&self) -> Option<Vec<usize>> {
        let mut in_degree = vec![0usize; self.n];
        for i in 0..self.n {
            for &j in &self.adj[i] {
                in_degree[j] += 1;
            }
        }
        let mut queue: VecDeque<usize> = VecDeque::new();
        for i in 0..self.n {
            if in_degree[i] == 0 {
                queue.push_back(i);
            }
        }
        let mut order = Vec::new();
        while let Some(node) = queue.pop_front() {
            order.push(node);
            for &succ in &self.adj[node] {
                in_degree[succ] -= 1;
                if in_degree[succ] == 0 {
                    queue.push_back(succ);
                }
            }
        }
        if order.len() == self.n { Some(order) } else { None }
    }

    /// Compute the longest path in the DAG.
    pub fn longest_path(&self) -> (usize, Vec<usize>) {
        let topo = match self.topological_sort() {
            Some(t) => t,
            None => return (0, Vec::new()),
        };
        let mut dist = vec![0usize; self.n];
        let mut pred = vec![usize::MAX; self.n];
        for &u in &topo {
            for &v in &self.adj[u] {
                if dist[u] + 1 > dist[v] {
                    dist[v] = dist[u] + 1;
                    pred[v] = u;
                }
            }
        }
        let (end, &max_dist) = dist.iter().enumerate().max_by_key(|(_, d)| *d).unwrap_or((0, &0));
        let mut path = Vec::new();
        let mut cur = end;
        while cur != usize::MAX {
            path.push(cur);
            cur = pred[cur];
        }
        path.reverse();
        (max_dist, path)
    }

    /// Compute the critical path (longest path).
    pub fn critical_path(&self) -> Vec<usize> {
        self.longest_path().1
    }

    /// Reachability query: can node a reach node b?
    pub fn reachability_query(&self, a: usize, b: usize) -> bool {
        if a == b { return true; }
        let mut visited = vec![false; self.n];
        let mut queue = VecDeque::new();
        queue.push_back(a);
        visited[a] = true;
        while let Some(node) = queue.pop_front() {
            for &succ in &self.adj[node] {
                if succ == b { return true; }
                if !visited[succ] {
                    visited[succ] = true;
                    queue.push_back(succ);
                }
            }
        }
        false
    }

    /// Find connected components (of the undirected version).
    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let mut visited = vec![false; self.n];
        let mut components = Vec::new();
        for i in 0..self.n {
            if !visited[i] {
                let mut component = Vec::new();
                let mut queue = VecDeque::new();
                queue.push_back(i);
                visited[i] = true;
                while let Some(node) = queue.pop_front() {
                    component.push(node);
                    for &succ in &self.adj[node] {
                        if !visited[succ] {
                            visited[succ] = true;
                            queue.push_back(succ);
                        }
                    }
                    for &pred in &self.rev_adj[node] {
                        if !visited[pred] {
                            visited[pred] = true;
                            queue.push_back(pred);
                        }
                    }
                }
                components.push(component);
            }
        }
        components
    }

    /// Compute the condensation (DAG of SCCs).
    pub fn condensation(&self) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let sccs = self.compute_sccs();
        let mut scc_id = vec![0usize; self.n];
        for (idx, scc) in sccs.iter().enumerate() {
            for &node in scc {
                scc_id[node] = idx;
            }
        }
        let mut cond_adj: Vec<HashSet<usize>> = vec![HashSet::new(); sccs.len()];
        for i in 0..self.n {
            for &j in &self.adj[i] {
                if scc_id[i] != scc_id[j] {
                    cond_adj[scc_id[i]].insert(scc_id[j]);
                }
            }
        }
        let cond_edges: Vec<Vec<usize>> = cond_adj.into_iter()
            .map(|s| s.into_iter().collect())
            .collect();
        (sccs, cond_edges)
    }

    /// Compute strongly connected components (Tarjan's algorithm).
    fn compute_sccs(&self) -> Vec<Vec<usize>> {
        let mut index_counter = 0usize;
        let mut stack = Vec::new();
        let mut on_stack = vec![false; self.n];
        let mut index = vec![usize::MAX; self.n];
        let mut lowlink = vec![usize::MAX; self.n];
        let mut sccs = Vec::new();
        for i in 0..self.n {
            if index[i] == usize::MAX {
                self.tarjan_dfs(i, &mut index_counter, &mut stack, &mut on_stack,
                    &mut index, &mut lowlink, &mut sccs);
            }
        }
        sccs
    }

    fn tarjan_dfs(
        &self,
        v: usize,
        counter: &mut usize,
        stack: &mut Vec<usize>,
        on_stack: &mut Vec<bool>,
        index: &mut Vec<usize>,
        lowlink: &mut Vec<usize>,
        sccs: &mut Vec<Vec<usize>>,
    ) {
        index[v] = *counter;
        lowlink[v] = *counter;
        *counter += 1;
        stack.push(v);
        on_stack[v] = true;
        for &w in &self.adj[v] {
            if index[w] == usize::MAX {
                self.tarjan_dfs(w, counter, stack, on_stack, index, lowlink, sccs);
                lowlink[v] = lowlink[v].min(lowlink[w]);
            } else if on_stack[w] {
                lowlink[v] = lowlink[v].min(index[w]);
            }
        }
        if lowlink[v] == index[v] {
            let mut scc = Vec::new();
            while let Some(w) = stack.pop() {
                on_stack[w] = false;
                scc.push(w);
                if w == v { break; }
            }
            sccs.push(scc);
        }
    }

    /// Number of nodes.
    pub fn size(&self) -> usize { self.n }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.adj.iter().map(|a| a.len()).sum()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ScopeAwareHappensBefore
// ═══════════════════════════════════════════════════════════════════════════

/// GPU scope levels for scope-aware happens-before.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum GpuScopeLevel {
    /// Single thread.
    Thread,
    /// Warp (32 threads).
    Warp,
    /// Cooperative Thread Array (threadblock).
    CTA,
    /// Entire GPU device.
    GPU,
    /// System (GPU + CPU).
    System,
}

impl fmt::Display for GpuScopeLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuScopeLevel::Thread => write!(f, "thread"),
            GpuScopeLevel::Warp => write!(f, "warp"),
            GpuScopeLevel::CTA => write!(f, "cta"),
            GpuScopeLevel::GPU => write!(f, "gpu"),
            GpuScopeLevel::System => write!(f, "system"),
        }
    }
}

impl GpuScopeLevel {
    /// Check if this scope includes another.
    pub fn includes(&self, other: &GpuScopeLevel) -> bool {
        *self >= *other
    }

    /// Check if this scope is wider than another.
    pub fn is_wider_than(&self, other: &GpuScopeLevel) -> bool {
        *self > *other
    }

    /// Get all scope levels.
    pub fn all() -> &'static [GpuScopeLevel] {
        &[GpuScopeLevel::Thread, GpuScopeLevel::Warp, GpuScopeLevel::CTA,
          GpuScopeLevel::GPU, GpuScopeLevel::System]
    }

    /// Convert from Scope enum.
    pub fn from_scope(s: Scope) -> Self {
        match s {
            Scope::CTA => GpuScopeLevel::CTA,
            Scope::GPU => GpuScopeLevel::GPU,
            Scope::System => GpuScopeLevel::System,
            Scope::None => GpuScopeLevel::System,
        }
    }
}

/// Scope-aware happens-before for GPU memory models.
#[derive(Debug, Clone)]
pub struct ScopeAwareHappensBefore {
    /// Number of events.
    n: usize,
    /// Per-scope happens-before relations.
    per_scope: HashMap<GpuScopeLevel, BitMatrix>,
    /// Combined (system-wide) happens-before.
    combined: BitMatrix,
    /// Event scopes.
    event_scopes: Vec<GpuScopeLevel>,
    /// Thread-to-warp mapping.
    thread_warp: HashMap<ThreadId, u32>,
    /// Thread-to-CTA mapping.
    thread_cta: HashMap<ThreadId, u32>,
}

impl ScopeAwareHappensBefore {
    /// Create a new scope-aware HB relation.
    pub fn new(n: usize) -> Self {
        let mut per_scope = HashMap::new();
        for &scope in GpuScopeLevel::all() {
            per_scope.insert(scope, BitMatrix::new(n));
        }
        Self {
            n,
            per_scope,
            combined: BitMatrix::new(n),
            event_scopes: vec![GpuScopeLevel::System; n],
            thread_warp: HashMap::new(),
            thread_cta: HashMap::new(),
        }
    }

    /// Set the warp and CTA mappings.
    pub fn set_thread_mapping(&mut self, warp: HashMap<ThreadId, u32>, cta: HashMap<ThreadId, u32>) {
        self.thread_warp = warp;
        self.thread_cta = cta;
    }

    /// Add a scope-qualified edge.
    pub fn add_edge(&mut self, from: usize, to: usize, scope: GpuScopeLevel) {
        if let Some(mat) = self.per_scope.get_mut(&scope) {
            mat.set(from, to, true);
        }
        // Also add to all wider scopes
        for &s in GpuScopeLevel::all() {
            if s.includes(&scope) {
                if let Some(mat) = self.per_scope.get_mut(&s) {
                    mat.set(from, to, true);
                }
            }
        }
    }

    /// Compute all per-scope closures.
    pub fn compute(&mut self) {
        for scope in GpuScopeLevel::all() {
            if let Some(mat) = self.per_scope.get_mut(scope) {
                *mat = mat.transitive_closure();
            }
        }
        if let Some(sys) = self.per_scope.get(&GpuScopeLevel::System) {
            self.combined = sys.clone();
        }
    }

    /// Query happens-before at a specific scope.
    pub fn is_related_at_scope(&self, a: usize, b: usize, scope: GpuScopeLevel) -> bool {
        self.per_scope.get(&scope).map_or(false, |m| m.get(a, b))
    }

    /// Check for scope-aware races.
    pub fn scope_aware_races(
        &self,
        events: &[Event],
        min_scope: GpuScopeLevel,
    ) -> Vec<DataRace> {
        let mut races = Vec::new();
        let hb = self.per_scope.get(&min_scope).unwrap();
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                if events[i].address == events[j].address
                    && events[i].thread != events[j].thread
                    && (events[i].is_write() || events[j].is_write())
                    && !hb.get(i, j) && !hb.get(j, i)
                {
                    let race_type = match (events[i].is_write(), events[j].is_write()) {
                        (true, true) => RaceType::WriteWrite,
                        (true, false) => RaceType::WriteRead,
                        (false, true) => RaceType::ReadWrite,
                        _ => continue,
                    };
                    races.push(DataRace {
                        event_a: i,
                        event_b: j,
                        address: events[i].address,
                        race_type,
                        thread_a: events[i].thread,
                        thread_b: events[j].thread,
                    });
                }
            }
        }
        races
    }

    /// Analyze which scope level is required to order two events.
    pub fn minimum_ordering_scope(&self, a: usize, b: usize) -> Option<GpuScopeLevel> {
        for &scope in GpuScopeLevel::all() {
            if self.is_related_at_scope(a, b, scope) || self.is_related_at_scope(b, a, scope) {
                return Some(scope);
            }
        }
        None
    }

    /// Get the combined (system-wide) happens-before.
    pub fn get_combined(&self) -> &BitMatrix { &self.combined }

    /// Get the per-scope relation for a specific scope.
    pub fn get_scope_relation(&self, scope: GpuScopeLevel) -> Option<&BitMatrix> {
        self.per_scope.get(&scope)
    }

    /// Compute scope escalation: events that require wider scope than annotated.
    pub fn scope_escalation_analysis(&self, events: &[Event]) -> Vec<(usize, GpuScopeLevel, GpuScopeLevel)> {
        let mut escalations = Vec::new();
        for i in 0..self.n {
            let annotated = GpuScopeLevel::from_scope(events[i].scope);
            let required = self.minimum_ordering_scope_for_event(i, events);
            if let Some(req) = required {
                if req.is_wider_than(&annotated) {
                    escalations.push((i, annotated, req));
                }
            }
        }
        escalations
    }

    /// Determine the minimum scope level needed to order all conflicting accesses
    /// involving a given event.
    fn minimum_ordering_scope_for_event(&self, event: usize, events: &[Event]) -> Option<GpuScopeLevel> {
        let mut max_scope = GpuScopeLevel::Thread;
        for j in 0..self.n {
            if j != event
                && events[j].address == events[event].address
                && (events[j].is_write() || events[event].is_write())
            {
                if let Some(scope) = self.minimum_ordering_scope(event, j) {
                    if scope.is_wider_than(&max_scope) {
                        max_scope = scope;
                    }
                }
            }
        }
        Some(max_scope)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// IncrementalHappensBefore
// ═══════════════════════════════════════════════════════════════════════════

/// Maintains a happens-before relation incrementally under edge additions/removals.
#[derive(Debug, Clone)]
pub struct IncrementalHappensBefore {
    /// Current relation.
    relation: BitMatrix,
    /// Current transitive closure.
    closure: BitMatrix,
    /// History of operations for rollback.
    history: Vec<IncrementalOp>,
    /// Checkpoint stack for batch rollback.
    checkpoints: Vec<usize>,
}

/// An incremental operation on the happens-before relation.
#[derive(Debug, Clone)]
pub enum IncrementalOp {
    /// Added an edge.
    AddEdge(usize, usize),
    /// Removed an edge.
    RemoveEdge(usize, usize),
}

impl IncrementalHappensBefore {
    /// Create a new incremental HB relation.
    pub fn new(n: usize) -> Self {
        Self {
            relation: BitMatrix::new(n),
            closure: BitMatrix::new(n),
            history: Vec::new(),
            checkpoints: Vec::new(),
        }
    }

    /// Create from an existing relation.
    pub fn from_relation(relation: BitMatrix) -> Self {
        let closure = relation.transitive_closure();
        Self {
            relation,
            closure,
            history: Vec::new(),
            checkpoints: Vec::new(),
        }
    }

    /// Add an edge incrementally.
    pub fn add_edge(&mut self, from: usize, to: usize) {
        if self.relation.get(from, to) { return; }
        self.relation.set(from, to, true);
        self.history.push(IncrementalOp::AddEdge(from, to));
        // Incrementally update the closure
        let n = self.relation.dim();
        let mut preds = vec![from];
        for i in 0..n {
            if self.closure.get(i, from) { preds.push(i); }
        }
        let mut succs = vec![to];
        for j in 0..n {
            if self.closure.get(to, j) { succs.push(j); }
        }
        for &p in &preds {
            for &s in &succs {
                self.closure.set(p, s, true);
            }
        }
    }

    /// Remove an edge (requires full recomputation of closure).
    pub fn remove_edge(&mut self, from: usize, to: usize) {
        if !self.relation.get(from, to) { return; }
        self.relation.set(from, to, false);
        self.history.push(IncrementalOp::RemoveEdge(from, to));
        self.closure = self.relation.transitive_closure();
    }

    /// Add multiple edges as a batch.
    pub fn add_edges_batch(&mut self, edges: &[(usize, usize)]) {
        for &(from, to) in edges {
            self.relation.set(from, to, true);
            self.history.push(IncrementalOp::AddEdge(from, to));
        }
        self.closure = self.relation.transitive_closure();
    }

    /// Save a checkpoint for potential rollback.
    pub fn checkpoint(&mut self) {
        self.checkpoints.push(self.history.len());
    }

    /// Rollback to the last checkpoint.
    pub fn rollback(&mut self) -> bool {
        if let Some(target) = self.checkpoints.pop() {
            while self.history.len() > target {
                if let Some(op) = self.history.pop() {
                    match op {
                        IncrementalOp::AddEdge(from, to) => {
                            self.relation.set(from, to, false);
                        }
                        IncrementalOp::RemoveEdge(from, to) => {
                            self.relation.set(from, to, true);
                        }
                    }
                }
            }
            self.closure = self.relation.transitive_closure();
            true
        } else {
            false
        }
    }

    /// Get the current relation.
    pub fn get_relation(&self) -> &BitMatrix { &self.relation }

    /// Get the current closure.
    pub fn get_closure(&self) -> &BitMatrix { &self.closure }

    /// Check if a is related to b in the closure.
    pub fn is_related(&self, a: usize, b: usize) -> bool {
        self.closure.get(a, b)
    }

    /// Number of edges in the base relation.
    pub fn base_edge_count(&self) -> usize { self.relation.count_edges() }

    /// Number of edges in the closure.
    pub fn closure_edge_count(&self) -> usize { self.closure.count_edges() }

    /// Number of operations in history.
    pub fn history_len(&self) -> usize { self.history.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_events(n: usize) -> Vec<Event> {
        (0..n).map(|i| Event {
            id: i,
            thread: i % 2,
            op_type: if i % 3 == 0 { OpType::Write } else { OpType::Read },
            address: 0x100,
            value: i as u64,
            scope: Scope::System,
            po_index: i,
        }).collect()
    }

    #[test]
    fn test_hb_basic() {
        let mut hb = HappensBeforeRelation::new(4);
        hb.add_po_edge(0, 1);
        hb.add_po_edge(2, 3);
        hb.add_rf_edge(1, 2);
        hb.compute();
        assert!(hb.is_related(0, 1));
        assert!(hb.is_related(1, 2));
        assert!(hb.is_related(0, 3));
        assert!(hb.is_acyclic());
    }

    #[test]
    fn test_tc_warshall() {
        let mut mat = BitMatrix::new(3);
        mat.set(0, 1, true);
        mat.set(1, 2, true);
        let mut engine = TransitiveClosureEngine::new(mat);
        engine.compute_warshall();
        assert!(engine.closure.get(0, 2));
        assert!(engine.validate_closure());
    }

    #[test]
    fn test_race_detection() {
        let events = make_events(4);
        let hb = BitMatrix::new(4);
        let mut detector = RaceDetector::new(events, hb);
        let races = detector.detect_races();
        assert!(races.len() > 0);
    }

    #[test]
    fn test_vector_clock() {
        let mut vc1 = VectorClock::for_thread(0);
        vc1.increment(0);
        let mut vc2 = VectorClock::for_thread(1);
        vc2.increment(1);
        assert!(vc1.concurrent(&vc2));
        vc2.join(&vc1);
        assert!(vc1.happens_before(&vc2));
    }

    #[test]
    fn test_incremental_hb() {
        let mut inc = IncrementalHappensBefore::new(4);
        inc.add_edge(0, 1);
        inc.add_edge(1, 2);
        assert!(inc.is_related(0, 2));
        inc.checkpoint();
        inc.add_edge(2, 3);
        assert!(inc.is_related(0, 3));
        inc.rollback();
        assert!(!inc.is_related(0, 3));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// HappensBeforeBuilder — fluent construction interface
// ═══════════════════════════════════════════════════════════════════════════

/// Builder for constructing happens-before relations incrementally.
#[derive(Debug, Clone)]
pub struct HappensBeforeBuilder {
    /// Number of events.
    n: usize,
    /// Pending PO edges.
    po_edges: Vec<(usize, usize)>,
    /// Pending RF edges.
    rf_edges: Vec<(usize, usize)>,
    /// Pending sync edges.
    sync_edges: Vec<(usize, usize)>,
    /// Pending CO edges.
    co_edges: Vec<(usize, usize)>,
    /// Thread assignments for events.
    thread_map: HashMap<usize, ThreadId>,
    /// Address assignments for events.
    address_map: HashMap<usize, Address>,
    /// Operation type assignments for events.
    op_map: HashMap<usize, OpType>,
}

impl HappensBeforeBuilder {
    /// Create a new builder.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            po_edges: Vec::new(),
            rf_edges: Vec::new(),
            sync_edges: Vec::new(),
            co_edges: Vec::new(),
            thread_map: HashMap::new(),
            address_map: HashMap::new(),
            op_map: HashMap::new(),
        }
    }

    /// Add a program-order edge.
    pub fn po(mut self, from: usize, to: usize) -> Self {
        self.po_edges.push((from, to));
        self
    }

    /// Add a reads-from edge.
    pub fn rf(mut self, write: usize, read: usize) -> Self {
        self.rf_edges.push((write, read));
        self
    }

    /// Add a synchronization edge.
    pub fn sync(mut self, from: usize, to: usize) -> Self {
        self.sync_edges.push((from, to));
        self
    }

    /// Add a coherence-order edge.
    pub fn co(mut self, earlier: usize, later: usize) -> Self {
        self.co_edges.push((earlier, later));
        self
    }

    /// Assign an event to a thread.
    pub fn thread(mut self, event: usize, tid: ThreadId) -> Self {
        self.thread_map.insert(event, tid);
        self
    }

    /// Assign an address to an event.
    pub fn address(mut self, event: usize, addr: Address) -> Self {
        self.address_map.insert(event, addr);
        self
    }

    /// Assign an operation type to an event.
    pub fn op(mut self, event: usize, op: OpType) -> Self {
        self.op_map.insert(event, op);
        self
    }

    /// Build the happens-before relation.
    pub fn build(self) -> HappensBeforeRelation {
        let mut hb = HappensBeforeRelation::new(self.n);
        for (from, to) in &self.po_edges {
            hb.add_po_edge(*from, *to);
        }
        for (write, read) in &self.rf_edges {
            hb.add_rf_edge(*write, *read);
        }
        for (from, to) in &self.sync_edges {
            hb.add_sync_edge(*from, *to);
        }
        for (earlier, later) in &self.co_edges {
            hb.add_co_edge(*earlier, *later);
        }
        hb.compute();
        hb
    }

    /// Build events from the configured metadata.
    pub fn build_events(&self) -> Vec<Event> {
        (0..self.n).map(|i| Event {
            id: i,
            thread: *self.thread_map.get(&i).unwrap_or(&0),
            op_type: *self.op_map.get(&i).unwrap_or(&OpType::Read),
            address: *self.address_map.get(&i).unwrap_or(&0x100),
            value: i as u64,
            scope: Scope::System,
            po_index: i,
        }).collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// HappensBeforeStatistics
// ═══════════════════════════════════════════════════════════════════════════

/// Statistics about a happens-before relation.
#[derive(Debug, Clone)]
pub struct HappensBeforeStatistics {
    /// Total number of events.
    pub num_events: usize,
    /// Number of PO edges.
    pub po_edges: usize,
    /// Number of RF edges.
    pub rf_edges: usize,
    /// Number of sync edges.
    pub sync_edges: usize,
    /// Number of CO edges.
    pub co_edges: usize,
    /// Number of FR edges.
    pub fr_edges: usize,
    /// Number of HB edges (total after closure).
    pub hb_edges: usize,
    /// Whether the relation is acyclic.
    pub is_acyclic: bool,
    /// Number of concurrent event pairs.
    pub concurrent_pairs: usize,
    /// Longest chain length.
    pub longest_chain: usize,
    /// Number of connected components.
    pub num_components: usize,
    /// Density: fraction of possible edges present.
    pub density: f64,
}

impl HappensBeforeStatistics {
    /// Compute statistics from a HappensBeforeRelation.
    pub fn compute(hb: &HappensBeforeRelation) -> Self {
        let n = hb.size();
        let total_possible = if n > 1 { n * (n - 1) } else { 1 };
        let hb_edges = hb.edge_count();

        let mut concurrent_pairs = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                if hb.are_concurrent(i, j) {
                    concurrent_pairs += 1;
                }
            }
        }

        let graph = HappensBeforeGraph::from_hb(hb);
        let (longest, _) = graph.longest_path();
        let components = graph.connected_components();

        Self {
            num_events: n,
            po_edges: hb.get_po().count_edges(),
            rf_edges: hb.get_rf().count_edges(),
            sync_edges: hb.get_sync().count_edges(),
            co_edges: hb.get_co().count_edges(),
            fr_edges: hb.get_fr().count_edges(),
            hb_edges,
            is_acyclic: hb.is_acyclic(),
            concurrent_pairs,
            longest_chain: longest,
            num_components: components.len(),
            density: hb_edges as f64 / total_possible as f64,
        }
    }
}

impl fmt::Display for HappensBeforeStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "HB Statistics:")?;
        writeln!(f, "  Events: {}", self.num_events)?;
        writeln!(f, "  PO edges: {}", self.po_edges)?;
        writeln!(f, "  RF edges: {}", self.rf_edges)?;
        writeln!(f, "  Sync edges: {}", self.sync_edges)?;
        writeln!(f, "  CO edges: {}", self.co_edges)?;
        writeln!(f, "  FR edges: {}", self.fr_edges)?;
        writeln!(f, "  HB edges: {}", self.hb_edges)?;
        writeln!(f, "  Acyclic: {}", self.is_acyclic)?;
        writeln!(f, "  Concurrent pairs: {}", self.concurrent_pairs)?;
        writeln!(f, "  Longest chain: {}", self.longest_chain)?;
        writeln!(f, "  Components: {}", self.num_components)?;
        writeln!(f, "  Density: {:.4}", self.density)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ThreadClusterAnalysis — identify thread interaction patterns
// ═══════════════════════════════════════════════════════════════════════════

/// Analysis of thread interaction patterns through happens-before.
#[derive(Debug, Clone)]
pub struct ThreadClusterAnalysis {
    /// Number of threads.
    num_threads: usize,
    /// Thread interaction matrix: how many HB edges between thread pairs.
    interaction_matrix: Vec<Vec<usize>>,
    /// Clusters of tightly interacting threads.
    clusters: Vec<Vec<ThreadId>>,
}

impl ThreadClusterAnalysis {
    /// Compute thread clusters from events and HB relation.
    pub fn compute(events: &[Event], hb: &HappensBeforeRelation) -> Self {
        let threads: BTreeSet<ThreadId> = events.iter().map(|e| e.thread).collect();
        let thread_list: Vec<ThreadId> = threads.into_iter().collect();
        let num_threads = thread_list.len();
        let thread_idx: HashMap<ThreadId, usize> = thread_list.iter()
            .enumerate().map(|(i, &t)| (t, i)).collect();

        let mut interaction_matrix = vec![vec![0usize; num_threads]; num_threads];
        for i in 0..events.len() {
            for j in 0..events.len() {
                if hb.is_related(i, j) && events[i].thread != events[j].thread {
                    let ti = thread_idx[&events[i].thread];
                    let tj = thread_idx[&events[j].thread];
                    interaction_matrix[ti][tj] += 1;
                }
            }
        }

        // Simple clustering: group threads with strong interactions
        let mut clusters = Vec::new();
        let mut assigned = vec![false; num_threads];
        for i in 0..num_threads {
            if assigned[i] { continue; }
            let mut cluster = vec![thread_list[i]];
            assigned[i] = true;
            for j in (i + 1)..num_threads {
                if !assigned[j] && (interaction_matrix[i][j] > 0 || interaction_matrix[j][i] > 0) {
                    cluster.push(thread_list[j]);
                    assigned[j] = true;
                }
            }
            clusters.push(cluster);
        }

        Self { num_threads, interaction_matrix, clusters }
    }

    /// Get the interaction count between two threads (by index).
    pub fn interaction_count(&self, t1: usize, t2: usize) -> usize {
        self.interaction_matrix[t1][t2] + self.interaction_matrix[t2][t1]
    }

    /// Get the clusters.
    pub fn get_clusters(&self) -> &[Vec<ThreadId>] { &self.clusters }

    /// Number of clusters.
    pub fn num_clusters(&self) -> usize { self.clusters.len() }

    /// Get the interaction matrix.
    pub fn get_interaction_matrix(&self) -> &Vec<Vec<usize>> { &self.interaction_matrix }
}

// ═══════════════════════════════════════════════════════════════════════════
// PartialOrderExtension — extending HB to total orders
// ═══════════════════════════════════════════════════════════════════════════

/// Extends a partial order (happens-before) to a total order (linearization).
#[derive(Debug, Clone)]
pub struct PartialOrderExtension {
    /// The base partial order.
    hb: BitMatrix,
    /// Number of events.
    n: usize,
}

impl PartialOrderExtension {
    /// Create from a HB relation.
    pub fn new(hb: &HappensBeforeRelation) -> Self {
        Self {
            hb: hb.get_relation_matrix().clone(),
            n: hb.size(),
        }
    }

    /// Create from a BitMatrix.
    pub fn from_matrix(hb: BitMatrix) -> Self {
        let n = hb.dim();
        Self { hb, n }
    }

    /// Compute one valid linearization (topological sort).
    pub fn linearize(&self) -> Option<Vec<usize>> {
        let mut in_degree = vec![0usize; self.n];
        for i in 0..self.n {
            for j in 0..self.n {
                if self.hb.get(i, j) {
                    in_degree[j] += 1;
                }
            }
        }
        let mut queue: VecDeque<usize> = VecDeque::new();
        for i in 0..self.n {
            if in_degree[i] == 0 {
                queue.push_back(i);
            }
        }
        let mut order = Vec::new();
        while let Some(node) = queue.pop_front() {
            order.push(node);
            for j in 0..self.n {
                if self.hb.get(node, j) {
                    in_degree[j] -= 1;
                    if in_degree[j] == 0 {
                        queue.push_back(j);
                    }
                }
            }
        }
        if order.len() == self.n { Some(order) } else { None }
    }

    /// Enumerate all linearizations (valid for small n).
    pub fn enumerate_linearizations(&self) -> Vec<Vec<usize>> {
        let mut results = Vec::new();
        let mut current = Vec::new();
        let mut available: Vec<bool> = vec![true; self.n];
        self.enumerate_recursive(&mut current, &mut available, &mut results);
        results
    }

    fn enumerate_recursive(
        &self,
        current: &mut Vec<usize>,
        available: &mut Vec<bool>,
        results: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == self.n {
            results.push(current.clone());
            return;
        }
        // Find candidates: available events with all predecessors already placed
        for i in 0..self.n {
            if !available[i] { continue; }
            let all_preds_placed = (0..self.n).all(|j| {
                !self.hb.get(j, i) || !available[j]
            });
            if all_preds_placed {
                available[i] = false;
                current.push(i);
                self.enumerate_recursive(current, available, results);
                current.pop();
                available[i] = true;
            }
        }
    }

    /// Count the number of valid linearizations.
    pub fn count_linearizations(&self) -> usize {
        self.enumerate_linearizations().len()
    }

    /// Check if a proposed linearization is valid.
    pub fn is_valid_linearization(&self, order: &[usize]) -> bool {
        if order.len() != self.n { return false; }
        let mut pos = vec![0usize; self.n];
        for (i, &e) in order.iter().enumerate() {
            pos[e] = i;
        }
        for i in 0..self.n {
            for j in 0..self.n {
                if self.hb.get(i, j) && pos[i] >= pos[j] {
                    return false;
                }
            }
        }
        true
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// AcyclicityChecker — efficient acyclicity checking
// ═══════════════════════════════════════════════════════════════════════════

/// Efficient acyclicity checker for happens-before relations.
#[derive(Debug, Clone)]
pub struct AcyclicityChecker {
    /// Number of nodes.
    n: usize,
}

impl AcyclicityChecker {
    /// Create a new checker.
    pub fn new(n: usize) -> Self {
        Self { n }
    }

    /// Check if a BitMatrix is acyclic using DFS.
    pub fn is_acyclic(&self, matrix: &BitMatrix) -> bool {
        let mut color = vec![0u8; self.n]; // 0=white, 1=gray, 2=black
        for i in 0..self.n {
            if color[i] == 0 {
                if self.has_cycle_dfs(matrix, i, &mut color) {
                    return false;
                }
            }
        }
        true
    }

    fn has_cycle_dfs(&self, matrix: &BitMatrix, node: usize, color: &mut Vec<u8>) -> bool {
        color[node] = 1; // gray
        for succ in matrix.successors(node) {
            if color[succ] == 1 {
                return true; // back edge = cycle
            }
            if color[succ] == 0 && self.has_cycle_dfs(matrix, succ, color) {
                return true;
            }
        }
        color[node] = 2; // black
        false
    }

    /// Find all cycles (returns list of cycle descriptions).
    pub fn find_all_cycles(&self, matrix: &BitMatrix) -> Vec<Vec<usize>> {
        let mut cycles = Vec::new();
        let mut visited = vec![false; self.n];
        let mut on_stack = vec![false; self.n];
        let mut stack = Vec::new();
        for i in 0..self.n {
            if !visited[i] {
                self.find_cycles_dfs(matrix, i, &mut visited, &mut on_stack, &mut stack, &mut cycles);
            }
        }
        cycles
    }

    fn find_cycles_dfs(
        &self,
        matrix: &BitMatrix,
        node: usize,
        visited: &mut Vec<bool>,
        on_stack: &mut Vec<bool>,
        stack: &mut Vec<usize>,
        cycles: &mut Vec<Vec<usize>>,
    ) {
        visited[node] = true;
        on_stack[node] = true;
        stack.push(node);
        for succ in matrix.successors(node) {
            if !visited[succ] {
                self.find_cycles_dfs(matrix, succ, visited, on_stack, stack, cycles);
            } else if on_stack[succ] {
                if let Some(pos) = stack.iter().position(|&x| x == succ) {
                    cycles.push(stack[pos..].to_vec());
                }
            }
        }
        stack.pop();
        on_stack[node] = false;
    }

    /// Check if adding an edge would create a cycle.
    pub fn would_create_cycle(&self, matrix: &BitMatrix, from: usize, to: usize) -> bool {
        if from == to { return true; }
        // Check if to ->* from (i.e., from is reachable from to)
        let mut visited = vec![false; self.n];
        let mut queue = VecDeque::new();
        queue.push_back(to);
        visited[to] = true;
        while let Some(node) = queue.pop_front() {
            if node == from { return true; }
            for succ in matrix.successors(node) {
                if !visited[succ] {
                    visited[succ] = true;
                    queue.push_back(succ);
                }
            }
        }
        false
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ReachabilityIndex — efficient reachability queries
// ═══════════════════════════════════════════════════════════════════════════

/// Pre-computed index for efficient reachability queries.
#[derive(Debug, Clone)]
pub struct ReachabilityIndex {
    /// Number of nodes.
    n: usize,
    /// Topological order (if DAG).
    topo_order: Option<Vec<usize>>,
    /// Position in topological order.
    topo_pos: Vec<usize>,
    /// Pre-order DFS numbering.
    pre_order: Vec<usize>,
    /// Post-order DFS numbering.
    post_order: Vec<usize>,
    /// Transitive closure (for small graphs).
    closure: Option<BitMatrix>,
}

impl ReachabilityIndex {
    /// Build an index from a BitMatrix.
    pub fn build(matrix: &BitMatrix) -> Self {
        let n = matrix.dim();
        let mut pre_order = vec![0; n];
        let mut post_order = vec![0; n];
        let mut pre_counter = 0usize;
        let mut post_counter = 0usize;
        let mut visited = vec![false; n];

        for start in 0..n {
            if !visited[start] {
                Self::dfs_numbering(
                    matrix, start, &mut visited,
                    &mut pre_order, &mut post_order,
                    &mut pre_counter, &mut post_counter,
                );
            }
        }

        // Try topological sort
        let graph = HappensBeforeGraph::from_matrix(matrix);
        let topo_order = graph.topological_sort();
        let mut topo_pos = vec![0; n];
        if let Some(ref order) = topo_order {
            for (i, &node) in order.iter().enumerate() {
                topo_pos[node] = i;
            }
        }

        let closure = if n <= 512 {
            Some(matrix.transitive_closure())
        } else {
            None
        };

        Self { n, topo_order, topo_pos, pre_order, post_order, closure }
    }

    fn dfs_numbering(
        matrix: &BitMatrix,
        node: usize,
        visited: &mut Vec<bool>,
        pre_order: &mut Vec<usize>,
        post_order: &mut Vec<usize>,
        pre_counter: &mut usize,
        post_counter: &mut usize,
    ) {
        visited[node] = true;
        pre_order[node] = *pre_counter;
        *pre_counter += 1;
        for succ in matrix.successors(node) {
            if !visited[succ] {
                Self::dfs_numbering(matrix, succ, visited, pre_order, post_order, pre_counter, post_counter);
            }
        }
        post_order[node] = *post_counter;
        *post_counter += 1;
    }

    /// Query whether a can reach b.
    pub fn can_reach(&self, a: usize, b: usize) -> bool {
        if a == b { return true; }
        if let Some(ref closure) = self.closure {
            return closure.get(a, b);
        }
        // Use topological order for filtering
        if self.topo_order.is_some() && self.topo_pos[a] >= self.topo_pos[b] {
            return false;
        }
        // Fallback: DFS check would be needed for larger graphs
        // For now return false as a conservative answer
        false
    }

    /// Get the pre-order numbering.
    pub fn get_pre_order(&self) -> &[usize] { &self.pre_order }

    /// Get the post-order numbering.
    pub fn get_post_order(&self) -> &[usize] { &self.post_order }

    /// Get the topological order (if available).
    pub fn get_topo_order(&self) -> Option<&Vec<usize>> { self.topo_order.as_ref() }
}

// ═══════════════════════════════════════════════════════════════════════════
// HbDiff — comparing happens-before relations
// ═══════════════════════════════════════════════════════════════════════════

/// Difference between two happens-before relations.
#[derive(Debug, Clone)]
pub struct HbDiff {
    /// Edges only in the first relation.
    pub only_in_first: Vec<(usize, usize)>,
    /// Edges only in the second relation.
    pub only_in_second: Vec<(usize, usize)>,
    /// Common edges.
    pub common: Vec<(usize, usize)>,
}

impl HbDiff {
    /// Compute the difference between two HB relations.
    pub fn compute(hb1: &HappensBeforeRelation, hb2: &HappensBeforeRelation) -> Self {
        let n = hb1.size().max(hb2.size());
        let m1 = hb1.get_relation_matrix();
        let m2 = hb2.get_relation_matrix();
        let mut only_in_first = Vec::new();
        let mut only_in_second = Vec::new();
        let mut common = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let in1 = i < m1.dim() && j < m1.dim() && m1.get(i, j);
                let in2 = i < m2.dim() && j < m2.dim() && m2.get(i, j);
                match (in1, in2) {
                    (true, true) => common.push((i, j)),
                    (true, false) => only_in_first.push((i, j)),
                    (false, true) => only_in_second.push((i, j)),
                    (false, false) => {}
                }
            }
        }
        Self { only_in_first, only_in_second, common }
    }

    /// Check if the two relations are identical.
    pub fn are_identical(&self) -> bool {
        self.only_in_first.is_empty() && self.only_in_second.is_empty()
    }

    /// Check if the first is a subset of the second.
    pub fn first_subset_of_second(&self) -> bool {
        self.only_in_first.is_empty()
    }

    /// Jaccard similarity of the two edge sets.
    pub fn jaccard_similarity(&self) -> f64 {
        let union_size = self.only_in_first.len() + self.only_in_second.len() + self.common.len();
        if union_size == 0 { return 1.0; }
        self.common.len() as f64 / union_size as f64
    }
}

impl fmt::Display for HbDiff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "HB Diff:")?;
        writeln!(f, "  Only in first: {} edges", self.only_in_first.len())?;
        writeln!(f, "  Only in second: {} edges", self.only_in_second.len())?;
        writeln!(f, "  Common: {} edges", self.common.len())?;
        writeln!(f, "  Jaccard similarity: {:.4}", self.jaccard_similarity())
    }
}

//! GPU scoped memory model support.
//!
//! Implements scope hierarchies (Thread, Warp, CTA, GPU, System),
//! scope-qualified relations, scope-qualified fences, scope inclusion
//! testing, and multi-scope constraint checking for PTX and CUDA semantics.

use std::collections::HashMap;
use std::fmt;
use serde::{Serialize, Deserialize};

use super::execution::{BitMatrix, ExecutionGraph, Event, EventId, OpType, Scope as ExecScope};
use super::memory_model::{MemoryModel, RelationExpr, PredicateExpr, Constraint};

// ---------------------------------------------------------------------------
// ScopeLevel — hierarchical scope levels
// ---------------------------------------------------------------------------

/// Hierarchical scope levels for GPU memory models.
/// Ordered from narrowest to widest scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ScopeLevel {
    /// Single thread (private).
    Thread = 0,
    /// Warp (32 threads typically, SIMD unit).
    Warp = 1,
    /// Cooperative Thread Array (threadblock).
    CTA = 2,
    /// GPU device scope.
    GPU = 3,
    /// System-wide scope (all devices + host).
    System = 4,
}

impl ScopeLevel {
    /// All scope levels from narrowest to widest.
    pub fn all() -> &'static [ScopeLevel] {
        &[
            ScopeLevel::Thread,
            ScopeLevel::Warp,
            ScopeLevel::CTA,
            ScopeLevel::GPU,
            ScopeLevel::System,
        ]
    }

    /// Whether self includes (is at least as wide as) other.
    pub fn includes(&self, other: &ScopeLevel) -> bool {
        (*self as u8) >= (*other as u8)
    }

    /// Whether self is strictly wider than other.
    pub fn is_wider_than(&self, other: &ScopeLevel) -> bool {
        (*self as u8) > (*other as u8)
    }

    /// Whether self is strictly narrower than other.
    pub fn is_narrower_than(&self, other: &ScopeLevel) -> bool {
        (*self as u8) < (*other as u8)
    }

    /// Convert from ExecScope.
    pub fn from_exec_scope(s: ExecScope) -> Self {
        match s {
            ExecScope::CTA => ScopeLevel::CTA,
            ExecScope::GPU => ScopeLevel::GPU,
            ExecScope::System => ScopeLevel::System,
            ExecScope::None => ScopeLevel::System,
        }
    }

    /// Convert to ExecScope.
    pub fn to_exec_scope(&self) -> ExecScope {
        match self {
            ScopeLevel::Thread | ScopeLevel::Warp => ExecScope::None,
            ScopeLevel::CTA => ExecScope::CTA,
            ScopeLevel::GPU => ExecScope::GPU,
            ScopeLevel::System => ExecScope::System,
        }
    }

    /// Minimum (narrowest) of two scopes.
    pub fn min_scope(a: ScopeLevel, b: ScopeLevel) -> ScopeLevel {
        if (a as u8) <= (b as u8) { a } else { b }
    }

    /// Maximum (widest) of two scopes.
    pub fn max_scope(a: ScopeLevel, b: ScopeLevel) -> ScopeLevel {
        if (a as u8) >= (b as u8) { a } else { b }
    }
}

impl fmt::Display for ScopeLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScopeLevel::Thread => write!(f, "thread"),
            ScopeLevel::Warp   => write!(f, "warp"),
            ScopeLevel::CTA    => write!(f, "cta"),
            ScopeLevel::GPU    => write!(f, "gpu"),
            ScopeLevel::System => write!(f, "sys"),
        }
    }
}

// ---------------------------------------------------------------------------
// ScopeHierarchy — describes the scope structure
// ---------------------------------------------------------------------------

/// Describes the GPU scope hierarchy for a particular execution.
/// Maps each event to its position in the hierarchy (thread, warp, CTA, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopeHierarchy {
    /// Thread ID for each event.
    pub thread_id: Vec<usize>,
    /// Warp ID for each event (threads 0..31 → warp 0, etc.).
    pub warp_id: Vec<usize>,
    /// CTA (threadblock) ID for each event.
    pub cta_id: Vec<usize>,
    /// GPU device ID for each event.
    pub gpu_id: Vec<usize>,
    /// Number of events.
    pub n: usize,
}

impl ScopeHierarchy {
    /// Create a hierarchy from event assignments.
    pub fn new(
        thread_id: Vec<usize>,
        warp_id: Vec<usize>,
        cta_id: Vec<usize>,
        gpu_id: Vec<usize>,
    ) -> Self {
        let n = thread_id.len();
        assert_eq!(warp_id.len(), n);
        assert_eq!(cta_id.len(), n);
        assert_eq!(gpu_id.len(), n);
        Self { thread_id, warp_id, cta_id, gpu_id, n }
    }

    /// Create a simple hierarchy from thread IDs, assigning warps and CTAs automatically.
    /// Assumes warp_size threads per warp, warps_per_cta warps per CTA.
    pub fn from_threads(
        thread_ids: &[usize],
        warp_size: usize,
        warps_per_cta: usize,
    ) -> Self {
        let n = thread_ids.len();
        let mut warp_id = vec![0; n];
        let mut cta_id = vec![0; n];
        let gpu_id = vec![0; n];

        for (i, &tid) in thread_ids.iter().enumerate() {
            warp_id[i] = tid / warp_size;
            cta_id[i] = tid / (warp_size * warps_per_cta);
        }

        Self::new(thread_ids.to_vec(), warp_id, cta_id, gpu_id)
    }

    /// Create from an execution graph, using thread IDs and simple defaults.
    pub fn from_exec(exec: &ExecutionGraph) -> Self {
        let thread_ids: Vec<usize> = exec.events.iter().map(|e| e.thread).collect();
        Self::from_threads(&thread_ids, 32, 1)
    }

    /// Build the "same-scope" relation for a given scope level.
    /// Two events are in the same scope if they share the same scope-level ID.
    pub fn same_scope(&self, level: ScopeLevel) -> BitMatrix {
        let mut m = BitMatrix::new(self.n);
        for i in 0..self.n {
            for j in 0..self.n {
                if self.same_scope_pair(i, j, level) {
                    m.set(i, j, true);
                }
            }
        }
        m
    }

    /// Check if two events are in the same scope at the given level.
    fn same_scope_pair(&self, i: usize, j: usize, level: ScopeLevel) -> bool {
        match level {
            ScopeLevel::Thread => self.thread_id[i] == self.thread_id[j],
            ScopeLevel::Warp => self.warp_id[i] == self.warp_id[j],
            ScopeLevel::CTA => self.cta_id[i] == self.cta_id[j],
            ScopeLevel::GPU => self.gpu_id[i] == self.gpu_id[j],
            ScopeLevel::System => true,
        }
    }

    /// Get the scope ID for an event at a given level.
    pub fn scope_id(&self, event: usize, level: ScopeLevel) -> usize {
        match level {
            ScopeLevel::Thread => self.thread_id[event],
            ScopeLevel::Warp => self.warp_id[event],
            ScopeLevel::CTA => self.cta_id[event],
            ScopeLevel::GPU => self.gpu_id[event],
            ScopeLevel::System => 0,
        }
    }
}

// ---------------------------------------------------------------------------
// ScopeQualifiedRelation — a relation restricted to a scope
// ---------------------------------------------------------------------------

/// A relation qualified by a scope level.
/// E.g., po_cta = po ∩ same-CTA.
#[derive(Debug, Clone)]
pub struct ScopeQualifiedRelation {
    pub base_name: String,
    pub scope: ScopeLevel,
    pub matrix: BitMatrix,
}

impl ScopeQualifiedRelation {
    /// Create a scope-qualified relation by intersecting with same-scope.
    pub fn new(
        name: &str,
        base: &BitMatrix,
        hierarchy: &ScopeHierarchy,
        scope: ScopeLevel,
    ) -> Self {
        let same = hierarchy.same_scope(scope);
        Self {
            base_name: name.to_string(),
            scope,
            matrix: base.intersection(&same),
        }
    }

    /// Name of this qualified relation.
    pub fn qualified_name(&self) -> String {
        format!("{}_{}", self.base_name, self.scope)
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.matrix.count_edges()
    }

    /// Edges.
    pub fn edges(&self) -> Vec<(usize, usize)> {
        self.matrix.edges()
    }
}

impl fmt::Display for ScopeQualifiedRelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {} edges", self.qualified_name(), self.edge_count())
    }
}

// ---------------------------------------------------------------------------
// ScopedFence — fence qualified by scope
// ---------------------------------------------------------------------------

/// A fence event with scope qualification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScopedFence {
    pub event_id: EventId,
    pub scope: ScopeLevel,
}

impl ScopedFence {
    pub fn new(event_id: EventId, scope: ScopeLevel) -> Self {
        Self { event_id, scope }
    }
}

/// Extract scoped fences from an execution graph.
pub fn extract_scoped_fences(exec: &ExecutionGraph) -> Vec<ScopedFence> {
    exec.events.iter()
        .filter(|e| e.is_fence())
        .map(|e| ScopedFence::new(e.id, ScopeLevel::from_exec_scope(e.scope)))
        .collect()
}

/// Build fence-induced ordering for a given scope.
/// For each fence at scope S, if two events a,b are po-before/after the fence
/// and are within the same S scope, then (a,b) is in the ordering.
pub fn fence_ordering_at_scope(
    exec: &ExecutionGraph,
    hierarchy: &ScopeHierarchy,
    fence_scope: ScopeLevel,
) -> BitMatrix {
    let n = exec.len();
    let mut result = BitMatrix::new(n);
    let same = hierarchy.same_scope(fence_scope);

    for fence in extract_scoped_fences(exec) {
        if !fence.scope.includes(&fence_scope) {
            continue;
        }
        let fid = fence.event_id;
        // Events po-before the fence.
        let preds: Vec<usize> = (0..n)
            .filter(|&i| exec.po.get(i, fid) && !exec.events[i].is_fence())
            .collect();
        // Events po-after the fence.
        let succs: Vec<usize> = (0..n)
            .filter(|&j| exec.po.get(fid, j) && !exec.events[j].is_fence())
            .collect();

        for &a in &preds {
            for &b in &succs {
                if same.get(a, b) {
                    result.set(a, b, true);
                }
            }
        }
    }
    result
}

// ---------------------------------------------------------------------------
// ScopeInclusionChecker — check scope compatibility
// ---------------------------------------------------------------------------

/// Check scope inclusion relationships between operations.
pub struct ScopeInclusionChecker<'a> {
    hierarchy: &'a ScopeHierarchy,
}

impl<'a> ScopeInclusionChecker<'a> {
    pub fn new(hierarchy: &'a ScopeHierarchy) -> Self {
        Self { hierarchy }
    }

    /// Check if event `a`'s scope includes event `b`'s scope at the given level.
    pub fn scope_includes(&self, a: usize, b: usize, level: ScopeLevel) -> bool {
        self.hierarchy.same_scope_pair(a, b, level)
    }

    /// Find the narrowest scope that includes both events.
    pub fn narrowest_common_scope(&self, a: usize, b: usize) -> ScopeLevel {
        for level in ScopeLevel::all() {
            if self.hierarchy.same_scope_pair(a, b, *level) {
                return *level;
            }
        }
        ScopeLevel::System
    }

    /// Check if two events can communicate at the given scope.
    /// Communication is possible if both events are within the same scope instance.
    pub fn can_communicate(&self, a: usize, b: usize, scope: ScopeLevel) -> bool {
        self.hierarchy.same_scope_pair(a, b, scope)
    }
}

// ---------------------------------------------------------------------------
// MultiScopeConstraintChecker — check constraints across scopes
// ---------------------------------------------------------------------------

/// Check memory model constraints that are scope-qualified.
pub struct MultiScopeConstraintChecker<'a> {
    exec: &'a ExecutionGraph,
    hierarchy: &'a ScopeHierarchy,
}

impl<'a> MultiScopeConstraintChecker<'a> {
    pub fn new(exec: &'a ExecutionGraph, hierarchy: &'a ScopeHierarchy) -> Self {
        Self { exec, hierarchy }
    }

    /// Check acyclicity of a relation restricted to a given scope.
    pub fn check_acyclic_at_scope(
        &self,
        relation: &BitMatrix,
        scope: ScopeLevel,
    ) -> bool {
        let same = self.hierarchy.same_scope(scope);
        let scoped = relation.intersection(&same);
        scoped.is_acyclic()
    }

    /// Check irreflexivity of a relation restricted to a given scope.
    pub fn check_irreflexive_at_scope(
        &self,
        relation: &BitMatrix,
        scope: ScopeLevel,
    ) -> bool {
        let same = self.hierarchy.same_scope(scope);
        let scoped = relation.intersection(&same);
        scoped.is_irreflexive()
    }

    /// Check a constraint at all scope levels.
    pub fn check_at_all_scopes(
        &self,
        relation: &BitMatrix,
    ) -> HashMap<ScopeLevel, bool> {
        let mut results = HashMap::new();
        for &level in ScopeLevel::all() {
            results.insert(level, self.check_acyclic_at_scope(relation, level));
        }
        results
    }

    /// Find the widest scope at which a relation is acyclic.
    pub fn widest_acyclic_scope(&self, relation: &BitMatrix) -> Option<ScopeLevel> {
        for level in ScopeLevel::all().iter().rev() {
            if self.check_acyclic_at_scope(relation, *level) {
                return Some(*level);
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// PTX scope semantics
// ---------------------------------------------------------------------------

/// PTX memory model scope semantics.
pub struct PtxScopeSemantics;

impl PtxScopeSemantics {
    /// Build the PTX scoped communication relation.
    /// In PTX, rf/co/fr are scope-qualified: they only apply between
    /// events whose scope is sufficient to see each other.
    pub fn scoped_rf(
        exec: &ExecutionGraph,
        hierarchy: &ScopeHierarchy,
    ) -> HashMap<ScopeLevel, BitMatrix> {
        let mut result = HashMap::new();
        for &level in ScopeLevel::all() {
            let same = hierarchy.same_scope(level);
            result.insert(level, exec.rf.intersection(&same));
        }
        result
    }

    /// Build PTX preserved program order (ppo) with scope qualification.
    pub fn scoped_ppo(
        exec: &ExecutionGraph,
        hierarchy: &ScopeHierarchy,
    ) -> HashMap<ScopeLevel, BitMatrix> {
        let mut result = HashMap::new();
        for &level in ScopeLevel::all() {
            let same = hierarchy.same_scope(level);
            let scoped_po = exec.po.intersection(&same);
            result.insert(level, scoped_po);
        }
        result
    }

    /// Build the PTX causality order (similar to happens-before).
    pub fn causality_order(
        exec: &ExecutionGraph,
        hierarchy: &ScopeHierarchy,
        fence_scope: ScopeLevel,
    ) -> BitMatrix {
        let po = &exec.po;
        let rf = &exec.rf;
        let fence_ord = fence_ordering_at_scope(exec, hierarchy, fence_scope);

        // Causality = (po ∪ rf ∪ fence_ordering)+
        let base = po.union(rf).union(&fence_ord);
        base.transitive_closure()
    }

    /// Check PTX coherence at a given scope.
    pub fn check_coherence(
        exec: &ExecutionGraph,
        hierarchy: &ScopeHierarchy,
        scope: ScopeLevel,
    ) -> bool {
        let same = hierarchy.same_scope(scope);
        let scoped_co = exec.co.intersection(&same);
        scoped_co.is_acyclic()
    }
}

// ---------------------------------------------------------------------------
// CUDA scope semantics
// ---------------------------------------------------------------------------

/// CUDA memory model scope semantics (based on PTX but with additional rules).
pub struct CudaScopeSemantics;

impl CudaScopeSemantics {
    /// Build CUDA synchronizes-with relation.
    /// A release-acquire pair synchronizes if the release's scope includes
    /// the acquiring thread's scope.
    pub fn synchronizes_with(
        exec: &ExecutionGraph,
        hierarchy: &ScopeHierarchy,
    ) -> BitMatrix {
        let n = exec.len();
        let mut sw = BitMatrix::new(n);

        for (w, r) in exec.rf.edges() {
            let w_event = &exec.events[w];
            let r_event = &exec.events[r];

            let w_scope = ScopeLevel::from_exec_scope(w_event.scope);
            let r_scope = ScopeLevel::from_exec_scope(r_event.scope);

            // Synchronization requires matching release/acquire scopes
            // at a level that includes both events.
            let min = ScopeLevel::min_scope(w_scope, r_scope);
            if hierarchy.same_scope_pair(w, r, min) {
                sw.set(w, r, true);
            }
        }
        sw
    }

    /// Build CUDA happens-before relation.
    pub fn happens_before(
        exec: &ExecutionGraph,
        hierarchy: &ScopeHierarchy,
    ) -> BitMatrix {
        let sw = Self::synchronizes_with(exec, hierarchy);
        let po = &exec.po;
        // hb = (po ∪ sw)+
        po.union(&sw).transitive_closure()
    }

    /// Check CUDA consistency: hb must be acyclic and coherence must be
    /// consistent with happens-before.
    pub fn check_consistency(
        exec: &ExecutionGraph,
        hierarchy: &ScopeHierarchy,
    ) -> CudaConsistencyResult {
        let hb = Self::happens_before(exec, hierarchy);
        let hb_acyclic = hb.is_acyclic();

        // co must be consistent with hb: no hb-after co-before.
        let co_hb = exec.co.compose(&hb);
        let hb_co = hb.compose(&exec.co);
        let co_consistent = co_hb.intersection(&hb_co.inverse()).is_irreflexive();

        CudaConsistencyResult {
            hb_acyclic,
            co_consistent,
            consistent: hb_acyclic && co_consistent,
        }
    }
}

/// Result of CUDA consistency check.
#[derive(Debug, Clone)]
pub struct CudaConsistencyResult {
    pub hb_acyclic: bool,
    pub co_consistent: bool,
    pub consistent: bool,
}

impl fmt::Display for CudaConsistencyResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CUDA consistency: {} (hb_acyclic={}, co_consistent={})",
            if self.consistent { "✓" } else { "✗" },
            self.hb_acyclic, self.co_consistent)
    }
}

// ---------------------------------------------------------------------------
// Scoped litmus test builder
// ---------------------------------------------------------------------------

/// Helper for building scoped litmus tests.
pub struct ScopedLitmusBuilder {
    /// CTA assignment for each thread.
    pub thread_cta: Vec<usize>,
    /// Warp assignment for each thread.
    pub thread_warp: Vec<usize>,
}

impl ScopedLitmusBuilder {
    /// Create with per-thread CTA assignments.
    pub fn new(thread_cta: Vec<usize>) -> Self {
        let thread_warp = thread_cta.clone(); // Default: warp = CTA for simplicity.
        Self { thread_cta, thread_warp }
    }

    /// Build scope hierarchy for an execution graph.
    pub fn build_hierarchy(&self, exec: &ExecutionGraph) -> ScopeHierarchy {
        let n = exec.len();
        let mut thread_id = vec![0; n];
        let mut warp_id = vec![0; n];
        let mut cta_id = vec![0; n];
        let gpu_id = vec![0; n];

        for (i, e) in exec.events.iter().enumerate() {
            thread_id[i] = e.thread;
            if e.thread < self.thread_warp.len() {
                warp_id[i] = self.thread_warp[e.thread];
            }
            if e.thread < self.thread_cta.len() {
                cta_id[i] = self.thread_cta[e.thread];
            }
        }

        ScopeHierarchy::new(thread_id, warp_id, cta_id, gpu_id)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checker::execution::{Event, OpType, Scope as ExecScope};

    fn make_two_thread_exec() -> ExecutionGraph {
        let events = vec![
            Event::new(0, 0, OpType::Write, 0x100, 1).with_po_index(0),
            Event::new(1, 0, OpType::Read, 0x200, 0).with_po_index(1),
            Event::new(2, 1, OpType::Write, 0x200, 1).with_po_index(0),
            Event::new(3, 1, OpType::Read, 0x100, 0).with_po_index(1),
        ];
        let mut graph = ExecutionGraph::new(events);
        graph.rf.set(2, 1, true); // T1's write to 0x200 read by T0
        graph.rf.set(0, 3, true); // T0's write to 0x100 read by T1
        graph.derive_fr();
        graph
    }

    fn make_scoped_exec() -> (ExecutionGraph, ScopeHierarchy) {
        let events = vec![
            Event::new(0, 0, OpType::Write, 0x100, 1)
                .with_po_index(0).with_scope(ExecScope::CTA),
            Event::new(1, 0, OpType::Fence, 0, 0)
                .with_po_index(1).with_scope(ExecScope::CTA),
            Event::new(2, 0, OpType::Read, 0x200, 0)
                .with_po_index(2).with_scope(ExecScope::CTA),
            Event::new(3, 1, OpType::Write, 0x200, 1)
                .with_po_index(0).with_scope(ExecScope::CTA),
            Event::new(4, 1, OpType::Read, 0x100, 0)
                .with_po_index(1).with_scope(ExecScope::CTA),
        ];
        let graph = ExecutionGraph::new(events);
        // Threads 0,1 in same CTA
        let hierarchy = ScopeHierarchy::new(
            vec![0, 0, 0, 1, 1],
            vec![0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0],
        );
        (graph, hierarchy)
    }

    fn make_multi_cta_exec() -> (ExecutionGraph, ScopeHierarchy) {
        let events = vec![
            Event::new(0, 0, OpType::Write, 0x100, 1)
                .with_po_index(0).with_scope(ExecScope::GPU),
            Event::new(1, 0, OpType::Read, 0x200, 0)
                .with_po_index(1).with_scope(ExecScope::GPU),
            Event::new(2, 1, OpType::Write, 0x200, 1)
                .with_po_index(0).with_scope(ExecScope::GPU),
            Event::new(3, 1, OpType::Read, 0x100, 0)
                .with_po_index(1).with_scope(ExecScope::GPU),
        ];
        let graph = ExecutionGraph::new(events);
        // Threads in different CTAs but same GPU
        let hierarchy = ScopeHierarchy::new(
            vec![0, 0, 1, 1],
            vec![0, 0, 1, 1],
            vec![0, 0, 1, 1], // Different CTAs!
            vec![0, 0, 0, 0],
        );
        (graph, hierarchy)
    }

    #[test]
    fn test_scope_level_ordering() {
        assert!(ScopeLevel::Thread.is_narrower_than(&ScopeLevel::Warp));
        assert!(ScopeLevel::Warp.is_narrower_than(&ScopeLevel::CTA));
        assert!(ScopeLevel::CTA.is_narrower_than(&ScopeLevel::GPU));
        assert!(ScopeLevel::GPU.is_narrower_than(&ScopeLevel::System));
        assert!(ScopeLevel::System.includes(&ScopeLevel::Thread));
    }

    #[test]
    fn test_scope_level_includes() {
        assert!(ScopeLevel::System.includes(&ScopeLevel::CTA));
        assert!(ScopeLevel::GPU.includes(&ScopeLevel::GPU));
        assert!(!ScopeLevel::CTA.includes(&ScopeLevel::GPU));
    }

    #[test]
    fn test_scope_level_min_max() {
        assert_eq!(ScopeLevel::min_scope(ScopeLevel::CTA, ScopeLevel::GPU), ScopeLevel::CTA);
        assert_eq!(ScopeLevel::max_scope(ScopeLevel::CTA, ScopeLevel::GPU), ScopeLevel::GPU);
    }

    #[test]
    fn test_scope_hierarchy_same_scope() {
        let hierarchy = ScopeHierarchy::new(
            vec![0, 0, 1, 1],
            vec![0, 0, 1, 1],
            vec![0, 0, 0, 0],
            vec![0, 0, 0, 0],
        );

        let same_thread = hierarchy.same_scope(ScopeLevel::Thread);
        assert!(same_thread.get(0, 1)); // same thread
        assert!(!same_thread.get(0, 2)); // different threads

        let same_cta = hierarchy.same_scope(ScopeLevel::CTA);
        assert!(same_cta.get(0, 2)); // same CTA
        assert!(same_cta.get(1, 3)); // same CTA

        let same_sys = hierarchy.same_scope(ScopeLevel::System);
        assert!(same_sys.get(0, 3)); // always same system
    }

    #[test]
    fn test_scope_hierarchy_from_threads() {
        let hierarchy = ScopeHierarchy::from_threads(&[0, 1, 2, 3], 2, 2);
        // warp_size=2: threads 0,1 → warp 0; threads 2,3 → warp 1
        assert_eq!(hierarchy.warp_id, vec![0, 0, 1, 1]);
        // warps_per_cta=2: all in CTA 0
        assert_eq!(hierarchy.cta_id, vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_scope_qualified_relation() {
        let (exec, hierarchy) = make_scoped_exec();
        let sq = ScopeQualifiedRelation::new("po", &exec.po, &hierarchy, ScopeLevel::CTA);
        assert_eq!(sq.qualified_name(), "po_cta");
        assert!(sq.edge_count() > 0);
    }

    #[test]
    fn test_scope_qualified_relation_different_cta() {
        let (exec, hierarchy) = make_multi_cta_exec();
        let sq_cta = ScopeQualifiedRelation::new("po", &exec.po, &hierarchy, ScopeLevel::CTA);
        let sq_gpu = ScopeQualifiedRelation::new("po", &exec.po, &hierarchy, ScopeLevel::GPU);
        // po within same CTA should have edges; po across CTAs should not.
        assert!(sq_cta.edge_count() <= sq_gpu.edge_count());
    }

    #[test]
    fn test_extract_scoped_fences() {
        let (exec, _) = make_scoped_exec();
        let fences = extract_scoped_fences(&exec);
        assert_eq!(fences.len(), 1);
        assert_eq!(fences[0].scope, ScopeLevel::CTA);
    }

    #[test]
    fn test_fence_ordering_at_scope() {
        let (exec, hierarchy) = make_scoped_exec();
        let ordering = fence_ordering_at_scope(&exec, &hierarchy, ScopeLevel::CTA);
        // The fence at CTA scope should create ordering between
        // events before and after it on the same thread.
        assert!(ordering.count_edges() > 0);
    }

    #[test]
    fn test_scope_inclusion_checker() {
        let hierarchy = ScopeHierarchy::new(
            vec![0, 0, 1, 1],
            vec![0, 0, 1, 1],
            vec![0, 0, 0, 0],
            vec![0, 0, 0, 0],
        );
        let checker = ScopeInclusionChecker::new(&hierarchy);

        assert!(checker.scope_includes(0, 1, ScopeLevel::Thread));
        assert!(!checker.scope_includes(0, 2, ScopeLevel::Thread));
        assert!(checker.scope_includes(0, 2, ScopeLevel::CTA));
    }

    #[test]
    fn test_narrowest_common_scope() {
        let hierarchy = ScopeHierarchy::new(
            vec![0, 0, 1, 1],
            vec![0, 0, 1, 1],
            vec![0, 0, 0, 0],
            vec![0, 0, 0, 0],
        );
        let checker = ScopeInclusionChecker::new(&hierarchy);

        assert_eq!(checker.narrowest_common_scope(0, 1), ScopeLevel::Thread);
        assert_eq!(checker.narrowest_common_scope(0, 2), ScopeLevel::CTA);
    }

    #[test]
    fn test_multi_scope_constraint_checker() {
        let (exec, hierarchy) = make_multi_cta_exec();
        let checker = MultiScopeConstraintChecker::new(&exec, &hierarchy);
        // PO should be acyclic at all scopes.
        let results = checker.check_at_all_scopes(&exec.po);
        for (_, &acyclic) in &results {
            assert!(acyclic);
        }
    }

    #[test]
    fn test_widest_acyclic_scope() {
        let (exec, hierarchy) = make_scoped_exec();
        let checker = MultiScopeConstraintChecker::new(&exec, &hierarchy);
        let widest = checker.widest_acyclic_scope(&exec.po);
        assert!(widest.is_some());
    }

    #[test]
    fn test_ptx_scoped_rf() {
        let (exec, hierarchy) = make_scoped_exec();
        let scoped_rf = PtxScopeSemantics::scoped_rf(&exec, &hierarchy);
        // All scope levels should have rf entries at CTA level and above.
        assert!(scoped_rf.contains_key(&ScopeLevel::CTA));
    }

    #[test]
    fn test_ptx_causality_order() {
        let (exec, hierarchy) = make_scoped_exec();
        let causality = PtxScopeSemantics::causality_order(&exec, &hierarchy, ScopeLevel::CTA);
        assert!(causality.is_acyclic());
    }

    #[test]
    fn test_ptx_check_coherence() {
        let (exec, hierarchy) = make_scoped_exec();
        assert!(PtxScopeSemantics::check_coherence(&exec, &hierarchy, ScopeLevel::CTA));
    }

    #[test]
    fn test_cuda_synchronizes_with() {
        let (exec, hierarchy) = make_scoped_exec();
        let sw = CudaScopeSemantics::synchronizes_with(&exec, &hierarchy);
        // SW is derived from rf, so it has at most rf edges.
        assert!(sw.count_edges() <= exec.rf.count_edges());
    }

    #[test]
    fn test_cuda_happens_before() {
        let (exec, hierarchy) = make_scoped_exec();
        let hb = CudaScopeSemantics::happens_before(&exec, &hierarchy);
        // HB should include PO.
        for (i, j) in exec.po.edges() {
            assert!(hb.get(i, j), "HB should include PO edge ({}, {})", i, j);
        }
    }

    #[test]
    fn test_cuda_consistency() {
        let (exec, hierarchy) = make_scoped_exec();
        let result = CudaScopeSemantics::check_consistency(&exec, &hierarchy);
        assert!(result.hb_acyclic);
    }

    #[test]
    fn test_scoped_litmus_builder() {
        let builder = ScopedLitmusBuilder::new(vec![0, 0, 1, 1]);
        let exec = make_two_thread_exec();
        let hierarchy = builder.build_hierarchy(&exec);
        assert_eq!(hierarchy.n, exec.len());
    }

    #[test]
    fn test_scope_level_display() {
        assert_eq!(format!("{}", ScopeLevel::Thread), "thread");
        assert_eq!(format!("{}", ScopeLevel::CTA), "cta");
        assert_eq!(format!("{}", ScopeLevel::System), "sys");
    }

    #[test]
    fn test_scope_level_from_exec_scope() {
        assert_eq!(ScopeLevel::from_exec_scope(ExecScope::CTA), ScopeLevel::CTA);
        assert_eq!(ScopeLevel::from_exec_scope(ExecScope::GPU), ScopeLevel::GPU);
        assert_eq!(ScopeLevel::from_exec_scope(ExecScope::System), ScopeLevel::System);
        assert_eq!(ScopeLevel::from_exec_scope(ExecScope::None), ScopeLevel::System);
    }

    #[test]
    fn test_scope_level_to_exec_scope() {
        assert_eq!(ScopeLevel::CTA.to_exec_scope(), ExecScope::CTA);
        assert_eq!(ScopeLevel::GPU.to_exec_scope(), ExecScope::GPU);
        assert_eq!(ScopeLevel::System.to_exec_scope(), ExecScope::System);
    }

    #[test]
    fn test_scope_hierarchy_scope_id() {
        let hierarchy = ScopeHierarchy::new(
            vec![0, 1, 2, 3],
            vec![0, 0, 1, 1],
            vec![0, 0, 0, 0],
            vec![0, 0, 0, 0],
        );
        assert_eq!(hierarchy.scope_id(0, ScopeLevel::Thread), 0);
        assert_eq!(hierarchy.scope_id(2, ScopeLevel::Warp), 1);
        assert_eq!(hierarchy.scope_id(3, ScopeLevel::CTA), 0);
    }

    #[test]
    fn test_can_communicate() {
        let hierarchy = ScopeHierarchy::new(
            vec![0, 0, 1, 1],
            vec![0, 0, 1, 1],
            vec![0, 0, 1, 1],
            vec![0, 0, 0, 0],
        );
        let checker = ScopeInclusionChecker::new(&hierarchy);
        assert!(checker.can_communicate(0, 1, ScopeLevel::Thread));
        assert!(!checker.can_communicate(0, 2, ScopeLevel::CTA));
        assert!(checker.can_communicate(0, 2, ScopeLevel::GPU));
    }

    #[test]
    fn test_cuda_consistency_result_display() {
        let result = CudaConsistencyResult {
            hb_acyclic: true,
            co_consistent: true,
            consistent: true,
        };
        let s = format!("{}", result);
        assert!(s.contains("✓"));
    }
}

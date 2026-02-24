//! POWER/ARM relaxed memory model for the LITMUS∞ checker.
//!
//! Implements the POWER memory model with its barrier taxonomy
//! (sync, lwsync, isync, eieio), preserved program order via
//! dependencies, propagation and observation axioms, and coherence
//! order constraints.

use std::collections::HashMap;
use std::fmt;
use serde::{Serialize, Deserialize};

use crate::checker::execution::{
    ThreadId, OpType,
    ExecutionGraph, BitMatrix,
};
use crate::checker::memory_model::{
    MemoryModel, RelationExpr,
};
use crate::checker::litmus::{LitmusTest, Thread, Ordering};

// ═══════════════════════════════════════════════════════════════════════════
// POWER Barriers
// ═══════════════════════════════════════════════════════════════════════════

/// POWER barrier types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PowerBarrier {
    /// Heavyweight sync (full barrier): orders all operations.
    Sync,
    /// Lightweight sync: orders W→W, R→R, R→W, but NOT W→R.
    Lwsync,
    /// Instruction sync: orders R→R,R→W via control flow.
    Isync,
    /// Enforce in-order execution of I/O.
    Eieio,
}

impl PowerBarrier {
    /// Whether this barrier orders the given pair of operation types.
    pub fn orders(&self, before: OpType, after: OpType) -> bool {
        match self {
            PowerBarrier::Sync => true, // orders everything
            PowerBarrier::Lwsync => {
                // Orders all except W→R
                !(before == OpType::Write && after == OpType::Read)
            }
            PowerBarrier::Isync => {
                // Orders R→R and R→W (via control dependency)
                before == OpType::Read
            }
            PowerBarrier::Eieio => {
                // Orders W→W (for I/O)
                before == OpType::Write && after == OpType::Write
            }
        }
    }

    /// Barrier strength (higher = more orderings).
    pub fn strength(&self) -> u8 {
        match self {
            PowerBarrier::Eieio => 1,
            PowerBarrier::Isync => 2,
            PowerBarrier::Lwsync => 3,
            PowerBarrier::Sync => 4,
        }
    }

    /// Whether self is at least as strong as other.
    pub fn is_at_least(&self, other: &PowerBarrier) -> bool {
        self.strength() >= other.strength()
    }

    /// All barrier types.
    pub fn all() -> &'static [PowerBarrier] {
        &[PowerBarrier::Eieio, PowerBarrier::Isync, PowerBarrier::Lwsync, PowerBarrier::Sync]
    }
}

impl fmt::Display for PowerBarrier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PowerBarrier::Sync => write!(f, "sync"),
            PowerBarrier::Lwsync => write!(f, "lwsync"),
            PowerBarrier::Isync => write!(f, "isync"),
            PowerBarrier::Eieio => write!(f, "eieio"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ARM Barriers
// ═══════════════════════════════════════════════════════════════════════════

/// ARM barrier types (DMB/DSB variants).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ARMBarrier {
    /// DMB SY: full data memory barrier.
    DmbFull,
    /// DMB LD: data memory barrier for loads.
    DmbLd,
    /// DMB ST: data memory barrier for stores.
    DmbSt,
    /// DSB SY: full data synchronization barrier.
    DsbFull,
    /// DSB LD: data sync barrier for loads.
    DsbLd,
    /// DSB ST: data sync barrier for stores.
    DsbSt,
    /// ISB: instruction synchronization barrier.
    Isb,
}

impl ARMBarrier {
    /// Map to equivalent POWER barrier.
    pub fn to_power_barrier(&self) -> PowerBarrier {
        match self {
            ARMBarrier::DmbFull | ARMBarrier::DsbFull => PowerBarrier::Sync,
            ARMBarrier::DmbLd | ARMBarrier::DsbLd => PowerBarrier::Lwsync,
            ARMBarrier::DmbSt | ARMBarrier::DsbSt => PowerBarrier::Eieio,
            ARMBarrier::Isb => PowerBarrier::Isync,
        }
    }

    /// Whether this barrier orders the given pair of operation types.
    pub fn orders(&self, before: OpType, after: OpType) -> bool {
        match self {
            ARMBarrier::DmbFull | ARMBarrier::DsbFull => true,
            ARMBarrier::DmbLd | ARMBarrier::DsbLd => before == OpType::Read,
            ARMBarrier::DmbSt | ARMBarrier::DsbSt => {
                before == OpType::Write && after == OpType::Write
            }
            ARMBarrier::Isb => before == OpType::Read,
        }
    }

    /// Barrier strength.
    pub fn strength(&self) -> u8 {
        match self {
            ARMBarrier::DmbSt | ARMBarrier::DsbSt => 1,
            ARMBarrier::Isb => 2,
            ARMBarrier::DmbLd | ARMBarrier::DsbLd => 3,
            ARMBarrier::DmbFull | ARMBarrier::DsbFull => 4,
        }
    }
}

impl fmt::Display for ARMBarrier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ARMBarrier::DmbFull => write!(f, "DMB SY"),
            ARMBarrier::DmbLd => write!(f, "DMB LD"),
            ARMBarrier::DmbSt => write!(f, "DMB ST"),
            ARMBarrier::DsbFull => write!(f, "DSB SY"),
            ARMBarrier::DsbLd => write!(f, "DSB LD"),
            ARMBarrier::DsbSt => write!(f, "DSB ST"),
            ARMBarrier::Isb => write!(f, "ISB"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Preserved Program Order
// ═══════════════════════════════════════════════════════════════════════════

/// Dependency types in the POWER model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DependencyKind {
    /// Data dependency (register flow).
    Data,
    /// Address dependency (computed address).
    Address,
    /// Control dependency (branch condition).
    Control,
    /// Control+Isync dependency.
    ControlIsync,
}

/// Preserved program order computation for POWER.
#[derive(Debug)]
pub struct PreservedProgramOrder;

impl PreservedProgramOrder {
    /// Compute data dependencies between events.
    /// A data dependency exists when a read feeds a register that is used
    /// by a subsequent write's value.
    pub fn compute_data_deps(exec: &ExecutionGraph) -> BitMatrix {
        let n = exec.events.len();
        let mut deps = BitMatrix::new(n);

        // Group events by thread
        let mut by_thread: HashMap<ThreadId, Vec<usize>> = HashMap::new();
        for (i, ev) in exec.events.iter().enumerate() {
            by_thread.entry(ev.thread).or_default().push(i);
        }

        // For each thread, compute data deps
        for (_, events) in &by_thread {
            for &i in events {
                for &j in events {
                    if i == j { continue; }
                    let ei = &exec.events[i];
                    let ej = &exec.events[j];
                    if ei.po_index >= ej.po_index { continue; }

                    // Data dependency: Read → Write where read provides the value
                    if ei.op_type == OpType::Read && ej.op_type == OpType::Write {
                        // Heuristic: if read's value matches write's value, flag as data dep
                        if ei.value == ej.value && ei.value != 0 {
                            deps.set(i, j, true);
                        }
                    }
                }
            }
        }

        deps
    }

    /// Compute address dependencies between events.
    /// An address dependency exists when a read provides the address
    /// for a subsequent memory operation.
    pub fn compute_addr_deps(exec: &ExecutionGraph) -> BitMatrix {
        let n = exec.events.len();
        let mut deps = BitMatrix::new(n);

        let mut by_thread: HashMap<ThreadId, Vec<usize>> = HashMap::new();
        for (i, ev) in exec.events.iter().enumerate() {
            by_thread.entry(ev.thread).or_default().push(i);
        }

        for (_, events) in &by_thread {
            for &i in events {
                for &j in events {
                    if i == j { continue; }
                    let ei = &exec.events[i];
                    let ej = &exec.events[j];
                    if ei.po_index >= ej.po_index { continue; }

                    // Address dependency: Read → any mem op where read provides the address
                    if ei.op_type == OpType::Read
                        && (ej.op_type == OpType::Read || ej.op_type == OpType::Write)
                    {
                        // Heuristic: if read value equals the subsequent op's address
                        if ei.value == ej.address {
                            deps.set(i, j, true);
                        }
                    }
                }
            }
        }

        deps
    }

    /// Compute control dependencies between events.
    /// A control dependency exists when a read feeds a branch condition
    /// that determines whether a subsequent operation executes.
    pub fn compute_ctrl_deps(exec: &ExecutionGraph) -> BitMatrix {
        let n = exec.events.len();
        let mut deps = BitMatrix::new(n);

        let mut by_thread: HashMap<ThreadId, Vec<usize>> = HashMap::new();
        for (i, ev) in exec.events.iter().enumerate() {
            by_thread.entry(ev.thread).or_default().push(i);
        }

        for (_, events) in &by_thread {
            for &i in events {
                let ei = &exec.events[i];
                if ei.op_type != OpType::Read { continue; }
                // All subsequent events on the same thread have a potential control dep
                for &j in events {
                    let ej = &exec.events[j];
                    if ej.po_index > ei.po_index && ej.op_type == OpType::Write {
                        // Control dependency from read to subsequent write
                        deps.set(i, j, true);
                    }
                }
            }
        }

        deps
    }

    /// Compute the full preserved program order for POWER.
    /// ppo = deps ∪ (fence-induced orderings)
    pub fn compute_ppo(exec: &ExecutionGraph) -> BitMatrix {
        let n = exec.events.len();
        let data = Self::compute_data_deps(exec);
        let addr = Self::compute_addr_deps(exec);
        let ctrl = Self::compute_ctrl_deps(exec);

        let mut ppo = BitMatrix::new(n);

        // PPO includes all dependencies
        for i in 0..n {
            for j in 0..n {
                if data.get(i, j) || addr.get(i, j) || ctrl.get(i, j) {
                    ppo.set(i, j, true);
                }
            }
        }

        // Same-address ordering: R→W and W→W to same address
        for i in 0..n {
            for j in 0..n {
                if i == j { continue; }
                let ei = &exec.events[i];
                let ej = &exec.events[j];
                if ei.thread != ej.thread || ei.po_index >= ej.po_index { continue; }
                if ei.address != ej.address { continue; }

                // W→W to same address is always preserved
                if ei.op_type == OpType::Write && ej.op_type == OpType::Write {
                    ppo.set(i, j, true);
                }
                // R→W to same address via data dependency
                if ei.op_type == OpType::Read && ej.op_type == OpType::Write {
                    ppo.set(i, j, true);
                }
            }
        }

        ppo
    }

    /// Compute fence-induced ordering.
    pub fn compute_fence_ordering(exec: &ExecutionGraph, barrier: PowerBarrier) -> BitMatrix {
        let n = exec.events.len();
        let mut fence_order = BitMatrix::new(n);

        for f in 0..n {
            if exec.events[f].op_type != OpType::Fence { continue; }
            let thread = exec.events[f].thread;

            for i in 0..n {
                let ei = &exec.events[i];
                if ei.thread != thread || ei.po_index >= exec.events[f].po_index { continue; }
                if ei.op_type == OpType::Fence { continue; }

                for j in 0..n {
                    let ej = &exec.events[j];
                    if ej.thread != thread || ej.po_index <= exec.events[f].po_index { continue; }
                    if ej.op_type == OpType::Fence { continue; }

                    if barrier.orders(ei.op_type, ej.op_type) {
                        fence_order.set(i, j, true);
                    }
                }
            }
        }

        fence_order
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Coherence Constraints
// ═══════════════════════════════════════════════════════════════════════════

/// Coherence order constraint computation.
#[derive(Debug)]
pub struct CoherenceConstraints;

impl CoherenceConstraints {
    /// Compute the communication relation: com = rf ∪ co ∪ fr.
    pub fn compute_com(exec: &ExecutionGraph) -> BitMatrix {
        let n = exec.events.len();
        let mut com = BitMatrix::new(n);

        for (w, r) in exec.rf.edges() {
            com.set(w, r, true);
        }
        for (w1, w2) in exec.co.edges() {
            com.set(w1, w2, true);
        }
        // fr = rf⁻¹;co
        for (w_rf, r) in exec.rf.edges() {
            for (w_co_from, w_co_to) in exec.co.edges() {
                if w_rf == w_co_from {
                    com.set(r, w_co_to, true);
                }
            }
        }

        com
    }

    /// Compute per-location coherence: po-loc ∪ com must be acyclic.
    pub fn check_coherence(exec: &ExecutionGraph) -> bool {
        let n = exec.events.len();
        let com = Self::compute_com(exec);
        let mut rel = com;

        // Add po-loc edges
        for i in 0..n {
            for j in 0..n {
                if i == j { continue; }
                let ei = &exec.events[i];
                let ej = &exec.events[j];
                if ei.thread == ej.thread && ei.po_index < ej.po_index && ei.address == ej.address {
                    rel.set(i, j, true);
                }
            }
        }

        rel.is_acyclic()
    }

    /// Compute from-reads relation: fr = rf⁻¹ ; co.
    pub fn compute_fr(exec: &ExecutionGraph) -> BitMatrix {
        let n = exec.events.len();
        let mut fr = BitMatrix::new(n);

        for (w_rf, r) in exec.rf.edges() {
            for (w_co_from, w_co_to) in exec.co.edges() {
                if w_rf == w_co_from {
                    fr.set(r, w_co_to, true);
                }
            }
        }

        fr
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// POWER Axioms
// ═══════════════════════════════════════════════════════════════════════════

/// The POWER axiomatic memory model.
#[derive(Debug)]
pub struct PowerAxioms;

impl PowerAxioms {
    /// Build the full POWER happens-before relation.
    /// hb = ppo ∪ fence ∪ rfe
    pub fn compute_hb(exec: &ExecutionGraph) -> BitMatrix {
        let n = exec.events.len();
        let ppo = PreservedProgramOrder::compute_ppo(exec);
        let fence = PreservedProgramOrder::compute_fence_ordering(exec, PowerBarrier::Sync);
        let mut hb = BitMatrix::new(n);

        for i in 0..n {
            for j in 0..n {
                if ppo.get(i, j) || fence.get(i, j) {
                    hb.set(i, j, true);
                }
            }
        }

        // rfe (external rf)
        for (w, r) in exec.rf.edges() {
            if exec.events[w].thread != exec.events[r].thread {
                hb.set(w, r, true);
            }
        }

        hb.transitive_closure()
    }

    /// Compute the propagation relation.
    /// prop = (fence+ ; hb*) ∩ (same-address writes)
    pub fn compute_prop(exec: &ExecutionGraph) -> BitMatrix {
        let hb = Self::compute_hb(exec);
        let fence = PreservedProgramOrder::compute_fence_ordering(exec, PowerBarrier::Sync);

        // prop = fence+ ; hb*
        let fence_plus = fence.transitive_closure();
        let hb_star = hb.reflexive_transitive_closure();
        fence_plus.compose(&hb_star)
    }

    /// Check no-thin-air axiom: acyclicity of hb.
    pub fn check_no_thin_air(exec: &ExecutionGraph) -> bool {
        let hb = Self::compute_hb(exec);
        for i in 0..exec.events.len() {
            if hb.get(i, i) { return false; }
        }
        true
    }

    /// Check observation axiom: irreflexivity of fre;prop;hb*.
    pub fn check_observation(exec: &ExecutionGraph) -> bool {
        let n = exec.events.len();
        let hb = Self::compute_hb(exec);
        let prop = Self::compute_prop(exec);
        let hb_star = hb.reflexive_transitive_closure();

        // fre = external from-reads
        let fr = CoherenceConstraints::compute_fr(exec);
        let mut fre = BitMatrix::new(n);
        for i in 0..n {
            for j in 0..n {
                if fr.get(i, j) && exec.events[i].thread != exec.events[j].thread {
                    fre.set(i, j, true);
                }
            }
        }

        let composition = fre.compose(&prop).compose(&hb_star);
        for i in 0..n {
            if composition.get(i, i) { return false; }
        }
        true
    }

    /// Check propagation axiom: acyclicity of co ∪ prop.
    pub fn check_propagation(exec: &ExecutionGraph) -> bool {
        let prop = Self::compute_prop(exec);
        let mut rel = prop;

        for (w1, w2) in exec.co.edges() {
            rel.set(w1, w2, true);
        }

        rel.is_acyclic()
    }

    /// Full POWER consistency check.
    pub fn check_consistency(exec: &ExecutionGraph) -> bool {
        CoherenceConstraints::check_coherence(exec)
            && Self::check_no_thin_air(exec)
            && Self::check_observation(exec)
            && Self::check_propagation(exec)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// POWER Model Checker
// ═══════════════════════════════════════════════════════════════════════════

/// POWER model checker result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerCheckResult {
    /// Whether the execution is POWER-consistent.
    pub consistent: bool,
    /// Which axioms were violated.
    pub violated_axioms: Vec<String>,
    /// Statistics.
    pub stats: PowerCheckStats,
}

/// Statistics from POWER model checking.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PowerCheckStats {
    /// Number of executions checked.
    pub executions_checked: usize,
    /// Number of dependency edges found.
    pub dependency_edges: usize,
    /// Number of fence edges.
    pub fence_edges: usize,
}

/// The POWER memory model checker.
#[derive(Debug)]
pub struct PowerModelChecker {
    /// Statistics.
    pub stats: PowerCheckStats,
}

impl PowerModelChecker {
    /// Create a new checker.
    pub fn new() -> Self {
        PowerModelChecker { stats: PowerCheckStats::default() }
    }

    /// Build POWER as a `MemoryModel`.
    pub fn build_memory_model() -> MemoryModel {
        let mut model = MemoryModel::new("POWER");

        model.add_derived(
            "ppo",
            RelationExpr::union_many(vec![
                RelationExpr::base("data"),
                RelationExpr::base("addr"),
                RelationExpr::base("ctrl"),
            ]),
            "Preserved program order",
        );

        model.add_derived(
            "com",
            RelationExpr::union_many(vec![
                RelationExpr::base("rf"),
                RelationExpr::base("co"),
                RelationExpr::base("fr"),
            ]),
            "Communication",
        );

        model.add_derived(
            "hb",
            RelationExpr::plus(RelationExpr::union_many(vec![
                RelationExpr::base("ppo"),
                RelationExpr::base("fence"),
                RelationExpr::base("rfe"),
            ])),
            "Happens-before",
        );

        model.add_acyclic(RelationExpr::base("hb"));
        model.add_acyclic(RelationExpr::union_many(vec![
            RelationExpr::base("po-loc"),
            RelationExpr::base("com"),
        ]));

        model
    }

    /// Check execution against POWER model.
    pub fn check_execution(&mut self, exec: &ExecutionGraph) -> PowerCheckResult {
        self.stats.executions_checked += 1;

        let mut violated = Vec::new();

        if !CoherenceConstraints::check_coherence(exec) {
            violated.push("coherence".to_string());
        }
        if !PowerAxioms::check_no_thin_air(exec) {
            violated.push("no-thin-air".to_string());
        }
        if !PowerAxioms::check_observation(exec) {
            violated.push("observation".to_string());
        }
        if !PowerAxioms::check_propagation(exec) {
            violated.push("propagation".to_string());
        }

        let ppo = PreservedProgramOrder::compute_ppo(exec);
        self.stats.dependency_edges = ppo.count_edges();

        PowerCheckResult {
            consistent: violated.is_empty(),
            violated_axioms: violated,
            stats: self.stats.clone(),
        }
    }
}

impl Default for PowerModelChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ARM Model Checker
// ═══════════════════════════════════════════════════════════════════════════

/// ARM memory model checker (based on POWER with ARM-specific barriers).
#[derive(Debug)]
pub struct ARMModelChecker {
    /// The underlying POWER checker.
    pub power_checker: PowerModelChecker,
}

impl ARMModelChecker {
    /// Create a new ARM checker.
    pub fn new() -> Self {
        ARMModelChecker {
            power_checker: PowerModelChecker::new(),
        }
    }

    /// Build ARM model.
    pub fn build_memory_model() -> MemoryModel {
        let mut model = MemoryModel::new("ARM");

        model.add_derived(
            "ppo",
            RelationExpr::union_many(vec![
                RelationExpr::base("data"),
                RelationExpr::base("addr"),
                RelationExpr::base("ctrl"),
            ]),
            "Preserved program order (ARM)",
        );

        model.add_derived(
            "hb",
            RelationExpr::plus(RelationExpr::union_many(vec![
                RelationExpr::base("ppo"),
                RelationExpr::base("dmb"),
                RelationExpr::base("rfe"),
            ])),
            "Happens-before (ARM)",
        );

        model.add_acyclic(RelationExpr::base("hb"));
        model
    }

    /// Check execution against ARM model.
    pub fn check_execution(&mut self, exec: &ExecutionGraph) -> PowerCheckResult {
        // ARM is very similar to POWER
        self.power_checker.check_execution(exec)
    }
}

impl Default for ARMModelChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// POWER Litmus Builder
// ═══════════════════════════════════════════════════════════════════════════

/// Builder for standard POWER/ARM litmus tests.
pub struct PowerLitmusBuilder;

impl PowerLitmusBuilder {
    /// Message Passing with lwsync+sync.
    pub fn mp_lwsync_sync() -> LitmusTest {
        let mut test = LitmusTest::new("MP+lwsync+sync");
        let mut t0 = Thread::new(0);
        t0.store(0, 1, Ordering::Relaxed); // data = 1
        t0.fence(Ordering::Release, crate::checker::litmus::Scope::None); // lwsync
        t0.store(1, 1, Ordering::Relaxed); // flag = 1
        test.add_thread(t0);

        let mut t1 = Thread::new(1);
        t1.load(0, 1, Ordering::Relaxed); // r0 = flag
        t1.fence(Ordering::SeqCst, crate::checker::litmus::Scope::None); // sync
        t1.load(1, 0, Ordering::Relaxed); // r1 = data
        test.add_thread(t1);

        test
    }

    /// Store Buffering with syncs.
    pub fn sb_syncs() -> LitmusTest {
        let mut test = LitmusTest::new("SB+syncs");
        let mut t0 = Thread::new(0);
        t0.store(0, 1, Ordering::Relaxed);
        t0.fence(Ordering::SeqCst, crate::checker::litmus::Scope::None);
        t0.load(0, 1, Ordering::Relaxed);
        test.add_thread(t0);

        let mut t1 = Thread::new(1);
        t1.store(1, 1, Ordering::Relaxed);
        t1.fence(Ordering::SeqCst, crate::checker::litmus::Scope::None);
        t1.load(1, 0, Ordering::Relaxed);
        test.add_thread(t1);

        test
    }

    /// WRC (Write-Read Causality) variant.
    pub fn wrc() -> LitmusTest {
        let mut test = LitmusTest::new("WRC");
        let mut t0 = Thread::new(0);
        t0.store(0, 1, Ordering::Relaxed); // x = 1
        test.add_thread(t0);

        let mut t1 = Thread::new(1);
        t1.load(0, 0, Ordering::Relaxed); // r0 = x
        t1.store(1, 1, Ordering::Relaxed); // y = 1
        test.add_thread(t1);

        let mut t2 = Thread::new(2);
        t2.load(1, 1, Ordering::Relaxed); // r1 = y
        t2.load(2, 0, Ordering::Relaxed); // r2 = x
        test.add_thread(t2);

        test
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Model Comparison
// ═══════════════════════════════════════════════════════════════════════════

/// Compare outcomes across different memory models.
#[derive(Debug)]
pub struct ModelComparison;

impl ModelComparison {
    /// Check if an execution is allowed under POWER but forbidden under TSO.
    pub fn is_power_only(exec: &ExecutionGraph) -> bool {
        // POWER: use full dependency-based check
        let power_ok = PowerAxioms::check_consistency(exec);
        // TSO: use ghb acyclicity
        let tso_ok = crate::checker::x86_model::TSOAxioms::check_consistency(exec);
        power_ok && !tso_ok
    }

    /// Check if an execution is SC-consistent.
    pub fn is_sc_consistent(exec: &ExecutionGraph) -> bool {
        crate::checker::x86_model::TSOComparison::is_sc_consistent(exec)
    }

    /// Classify an execution by the weakest model that allows it.
    pub fn classify(exec: &ExecutionGraph) -> ModelClass {
        if Self::is_sc_consistent(exec) {
            ModelClass::SC
        } else if crate::checker::x86_model::TSOAxioms::check_consistency(exec) {
            ModelClass::TSO
        } else if PowerAxioms::check_consistency(exec) {
            ModelClass::POWER
        } else {
            ModelClass::Forbidden
        }
    }
}

/// Classification of an execution by memory model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelClass {
    /// Allowed under Sequential Consistency.
    SC,
    /// Allowed under TSO but not SC.
    TSO,
    /// Allowed under POWER but not TSO.
    POWER,
    /// Forbidden under all models.
    Forbidden,
}

impl fmt::Display for ModelClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelClass::SC => write!(f, "SC"),
            ModelClass::TSO => write!(f, "TSO-only"),
            ModelClass::POWER => write!(f, "POWER-only"),
            ModelClass::Forbidden => write!(f, "Forbidden"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_barrier_ordering() {
        assert!(PowerBarrier::Sync.orders(OpType::Write, OpType::Read));
        assert!(PowerBarrier::Sync.orders(OpType::Read, OpType::Write));
        assert!(!PowerBarrier::Lwsync.orders(OpType::Write, OpType::Read));
        assert!(PowerBarrier::Lwsync.orders(OpType::Write, OpType::Write));
        assert!(PowerBarrier::Lwsync.orders(OpType::Read, OpType::Read));
    }

    #[test]
    fn test_power_barrier_strength() {
        assert!(PowerBarrier::Sync.is_at_least(&PowerBarrier::Lwsync));
        assert!(PowerBarrier::Lwsync.is_at_least(&PowerBarrier::Eieio));
        assert!(!PowerBarrier::Eieio.is_at_least(&PowerBarrier::Sync));
    }

    #[test]
    fn test_arm_barrier_mapping() {
        assert_eq!(ARMBarrier::DmbFull.to_power_barrier(), PowerBarrier::Sync);
        assert_eq!(ARMBarrier::DmbSt.to_power_barrier(), PowerBarrier::Eieio);
        assert_eq!(ARMBarrier::Isb.to_power_barrier(), PowerBarrier::Isync);
    }

    #[test]
    fn test_arm_barrier_ordering() {
        assert!(ARMBarrier::DmbFull.orders(OpType::Write, OpType::Read));
        assert!(!ARMBarrier::DmbSt.orders(OpType::Read, OpType::Write));
        assert!(ARMBarrier::DmbSt.orders(OpType::Write, OpType::Write));
    }

    #[test]
    fn test_dependency_computation() {
        let exec = ExecutionGraph::new(vec![]);
        let data = PreservedProgramOrder::compute_data_deps(&exec);
        assert_eq!(data.count_edges(), 0); // empty graph
    }

    #[test]
    fn test_coherence_empty() {
        let exec = ExecutionGraph::new(vec![]);
        assert!(CoherenceConstraints::check_coherence(&exec));
    }

    #[test]
    fn test_com_computation() {
        let exec = ExecutionGraph::new(vec![]);
        let com = CoherenceConstraints::compute_com(&exec);
        assert_eq!(com.count_edges(), 0);
    }

    #[test]
    fn test_power_model_builder() {
        let model = PowerModelChecker::build_memory_model();
        assert_eq!(model.name, "POWER");
        assert!(!model.derived_relations.is_empty());
    }

    #[test]
    fn test_arm_model_builder() {
        let model = ARMModelChecker::build_memory_model();
        assert_eq!(model.name, "ARM");
    }

    #[test]
    fn test_power_checker() {
        let mut checker = PowerModelChecker::new();
        let exec = ExecutionGraph::new(vec![]);
        let result = checker.check_execution(&exec);
        assert!(result.consistent);
    }

    #[test]
    fn test_litmus_mp() {
        let test = PowerLitmusBuilder::mp_lwsync_sync();
        assert_eq!(test.thread_count(), 2);
    }

    #[test]
    fn test_litmus_sb_syncs() {
        let test = PowerLitmusBuilder::sb_syncs();
        assert_eq!(test.thread_count(), 2);
    }

    #[test]
    fn test_litmus_wrc() {
        let test = PowerLitmusBuilder::wrc();
        assert_eq!(test.thread_count(), 3);
    }

    #[test]
    fn test_model_class_display() {
        assert_eq!(format!("{}", ModelClass::SC), "SC");
        assert_eq!(format!("{}", ModelClass::TSO), "TSO-only");
        assert_eq!(format!("{}", ModelClass::POWER), "POWER-only");
    }

    #[test]
    fn test_classify_empty() {
        let exec = ExecutionGraph::new(vec![]);
        let class = ModelComparison::classify(&exec);
        assert_eq!(class, ModelClass::SC);
    }
}

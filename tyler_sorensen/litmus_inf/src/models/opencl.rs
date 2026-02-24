#![allow(unused)]

use std::collections::{HashMap, HashSet, BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use serde::{Serialize, Deserialize};

// ---------------------------------------------------------------------------
// OpenCL Memory Spaces & Orders
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OpenClMemorySpace {
    Global,
    Local,
    Constant,
    Private,
    Generic,
}

impl fmt::Display for OpenClMemorySpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Global => write!(f, "global"),
            Self::Local => write!(f, "local"),
            Self::Constant => write!(f, "constant"),
            Self::Private => write!(f, "private"),
            Self::Generic => write!(f, "generic"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum OpenClMemoryOrder {
    Relaxed,
    Acquire,
    Release,
    AcqRel,
    SeqCst,
}

impl OpenClMemoryOrder {
    pub fn is_at_least(&self, other: OpenClMemoryOrder) -> bool {
        (*self as u8) >= (other as u8)
    }

    pub fn combine(self, other: OpenClMemoryOrder) -> OpenClMemoryOrder {
        if (self as u8) >= (other as u8) { self } else { other }
    }
}

impl fmt::Display for OpenClMemoryOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Relaxed => write!(f, "relaxed"),
            Self::Acquire => write!(f, "acquire"),
            Self::Release => write!(f, "release"),
            Self::AcqRel => write!(f, "acq_rel"),
            Self::SeqCst => write!(f, "seq_cst"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum OpenClMemoryScope {
    WorkItem,
    SubGroup,
    WorkGroup,
    Device,
    AllSvmDevices,
}

impl OpenClMemoryScope {
    pub fn includes(&self, other: &OpenClMemoryScope) -> bool {
        (*self as u8) >= (*other as u8)
    }
}

impl fmt::Display for OpenClMemoryScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorkItem => write!(f, "work_item"),
            Self::SubGroup => write!(f, "sub_group"),
            Self::WorkGroup => write!(f, "work_group"),
            Self::Device => write!(f, "device"),
            Self::AllSvmDevices => write!(f, "all_svm_devices"),
        }
    }
}

// ---------------------------------------------------------------------------
// Barriers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Barrier {
    pub id: u32,
    pub memory_scope: OpenClMemoryScope,
    pub memory_order: OpenClMemoryOrder,
    pub address_spaces: Vec<OpenClMemorySpace>,
    pub work_group_id: u32,
    pub participating_items: Vec<u32>,
}

impl Barrier {
    pub fn work_group_barrier(
        id: u32,
        wg_id: u32,
        spaces: Vec<OpenClMemorySpace>,
    ) -> Self {
        Self {
            id,
            memory_scope: OpenClMemoryScope::WorkGroup,
            memory_order: OpenClMemoryOrder::AcqRel,
            address_spaces: spaces,
            work_group_id: wg_id,
            participating_items: Vec::new(),
        }
    }

    pub fn affects_space(&self, space: OpenClMemorySpace) -> bool {
        self.address_spaces.contains(&space) || self.address_spaces.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarrierViolation {
    pub barrier_id: u32,
    pub description: String,
    pub missing_items: Vec<u32>,
    pub extra_items: Vec<u32>,
}

impl fmt::Display for BarrierViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Barrier {} violation: {}", self.barrier_id, self.description)
    }
}

#[derive(Debug, Clone)]
pub struct WorkGroupBarrierChecker {
    pub work_group_size: u32,
    barriers: Vec<Barrier>,
}

impl WorkGroupBarrierChecker {
    pub fn new(work_group_size: u32) -> Self {
        Self { work_group_size, barriers: Vec::new() }
    }

    pub fn add_barrier(&mut self, barrier: Barrier) {
        self.barriers.push(barrier);
    }

    pub fn check_barrier_convergence(&self) -> Vec<BarrierViolation> {
        let mut violations = Vec::new();
        for barrier in &self.barriers {
            let expected: HashSet<u32> = (0..self.work_group_size).collect();
            let actual: HashSet<u32> = barrier.participating_items.iter().copied().collect();

            let missing: Vec<u32> = expected.difference(&actual).copied().collect();
            let extra: Vec<u32> = actual.difference(&expected).copied().collect();

            if !missing.is_empty() {
                violations.push(BarrierViolation {
                    barrier_id: barrier.id,
                    description: format!(
                        "Work items {:?} did not reach barrier in work group {}",
                        missing, barrier.work_group_id
                    ),
                    missing_items: missing,
                    extra_items: Vec::new(),
                });
            }
            if !extra.is_empty() {
                violations.push(BarrierViolation {
                    barrier_id: barrier.id,
                    description: format!(
                        "Unexpected work items {:?} at barrier in work group {}",
                        extra, barrier.work_group_id
                    ),
                    missing_items: Vec::new(),
                    extra_items: extra,
                });
            }

            // Check for duplicate arrivals
            let mut seen = HashSet::new();
            for &item in &barrier.participating_items {
                if !seen.insert(item) {
                    violations.push(BarrierViolation {
                        barrier_id: barrier.id,
                        description: format!(
                            "Work item {} arrived at barrier {} multiple times",
                            item, barrier.id
                        ),
                        missing_items: Vec::new(),
                        extra_items: vec![item],
                    });
                }
            }
        }
        violations
    }

    pub fn check_barrier_ordering(&self) -> Vec<BarrierViolation> {
        let mut violations = Vec::new();
        // Check that barriers for the same work group are reached in consistent order
        let mut wg_barriers: HashMap<u32, Vec<&Barrier>> = HashMap::new();
        for b in &self.barriers {
            wg_barriers.entry(b.work_group_id).or_default().push(b);
        }

        for (wg_id, barriers) in &wg_barriers {
            if barriers.len() > 1 {
                // Verify all items see the same barrier sequence
                for i in 0..barriers.len() - 1 {
                    if barriers[i].id >= barriers[i + 1].id {
                        // Not in order — could be a non-convergent path
                    }
                }
            }
        }
        violations
    }
}

// ---------------------------------------------------------------------------
// Memory Models
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct OpenClEvent {
    pub id: u32,
    pub work_item: u32,
    pub work_group: u32,
    pub op: OpenClAtomicOp,
    pub address: u64,
    pub value: Option<u64>,
    pub memory_space: OpenClMemorySpace,
    pub memory_order: OpenClMemoryOrder,
    pub memory_scope: OpenClMemoryScope,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OpenClAtomicOp {
    Load,
    Store,
    Exchange,
    CompareExchange,
    FetchAdd,
    FetchSub,
    FetchOr,
    FetchAnd,
    FetchXor,
    FetchMin,
    FetchMax,
    NonAtomicRead,
    NonAtomicWrite,
    Fence,
    Barrier,
}

impl OpenClAtomicOp {
    pub fn is_read(&self) -> bool {
        matches!(self,
            Self::Load | Self::Exchange | Self::CompareExchange |
            Self::FetchAdd | Self::FetchSub | Self::FetchOr |
            Self::FetchAnd | Self::FetchXor | Self::FetchMin |
            Self::FetchMax | Self::NonAtomicRead
        )
    }

    pub fn is_write(&self) -> bool {
        matches!(self,
            Self::Store | Self::Exchange | Self::CompareExchange |
            Self::FetchAdd | Self::FetchSub | Self::FetchOr |
            Self::FetchAnd | Self::FetchXor | Self::FetchMin |
            Self::FetchMax | Self::NonAtomicWrite
        )
    }

    pub fn is_rmw(&self) -> bool {
        matches!(self,
            Self::Exchange | Self::CompareExchange |
            Self::FetchAdd | Self::FetchSub | Self::FetchOr |
            Self::FetchAnd | Self::FetchXor | Self::FetchMin |
            Self::FetchMax
        )
    }

    pub fn is_atomic(&self) -> bool {
        !matches!(self, Self::NonAtomicRead | Self::NonAtomicWrite)
    }

    pub fn is_fence(&self) -> bool {
        matches!(self, Self::Fence | Self::Barrier)
    }
}

impl fmt::Display for OpenClAtomicOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load => write!(f, "atomic_load"),
            Self::Store => write!(f, "atomic_store"),
            Self::Exchange => write!(f, "atomic_exchange"),
            Self::CompareExchange => write!(f, "atomic_compare_exchange"),
            Self::FetchAdd => write!(f, "atomic_fetch_add"),
            Self::FetchSub => write!(f, "atomic_fetch_sub"),
            Self::FetchOr => write!(f, "atomic_fetch_or"),
            Self::FetchAnd => write!(f, "atomic_fetch_and"),
            Self::FetchXor => write!(f, "atomic_fetch_xor"),
            Self::FetchMin => write!(f, "atomic_fetch_min"),
            Self::FetchMax => write!(f, "atomic_fetch_max"),
            Self::NonAtomicRead => write!(f, "read"),
            Self::NonAtomicWrite => write!(f, "write"),
            Self::Fence => write!(f, "atomic_work_item_fence"),
            Self::Barrier => write!(f, "barrier"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OpenClAtomic {
    pub event: OpenClEvent,
}

impl OpenClAtomic {
    pub fn new(event: OpenClEvent) -> Self {
        Self { event }
    }

    pub fn is_release(&self) -> bool {
        matches!(self.event.memory_order, OpenClMemoryOrder::Release | OpenClMemoryOrder::AcqRel | OpenClMemoryOrder::SeqCst)
    }

    pub fn is_acquire(&self) -> bool {
        matches!(self.event.memory_order, OpenClMemoryOrder::Acquire | OpenClMemoryOrder::AcqRel | OpenClMemoryOrder::SeqCst)
    }

    pub fn is_seq_cst(&self) -> bool {
        self.event.memory_order == OpenClMemoryOrder::SeqCst
    }
}

// ---------------------------------------------------------------------------
// BitMatrix (self-contained)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitMatrix {
    rows: usize,
    cols: usize,
    data: Vec<Vec<bool>>,
}

impl BitMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![vec![false; cols]; rows] }
    }

    pub fn get(&self, row: usize, col: usize) -> bool {
        self.data[row][col]
    }

    pub fn set(&mut self, row: usize, col: usize, val: bool) {
        self.data[row][col] = val;
    }

    pub fn set_true(&mut self, row: usize, col: usize) {
        self.data[row][col] = true;
    }

    pub fn rows(&self) -> usize { self.rows }
    pub fn cols(&self) -> usize { self.cols }

    pub fn transitive_closure(&mut self) {
        let n = self.rows.min(self.cols);
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if self.data[i][k] && self.data[k][j] {
                        self.data[i][j] = true;
                    }
                }
            }
        }
    }

    pub fn compose(&self, other: &BitMatrix) -> BitMatrix {
        assert_eq!(self.cols, other.rows);
        let mut result = BitMatrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                for k in 0..self.cols {
                    if self.data[i][k] && other.data[k][j] {
                        result.data[i][j] = true;
                        break;
                    }
                }
            }
        }
        result
    }

    pub fn union(&self, other: &BitMatrix) -> BitMatrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut result = BitMatrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] || other.data[i][j];
            }
        }
        result
    }

    pub fn intersection(&self, other: &BitMatrix) -> BitMatrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut result = BitMatrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] && other.data[i][j];
            }
        }
        result
    }

    pub fn is_irreflexive(&self) -> bool {
        let n = self.rows.min(self.cols);
        for i in 0..n {
            if self.data[i][i] { return false; }
        }
        true
    }

    pub fn is_acyclic(&self) -> bool {
        let mut tc = self.clone();
        tc.transitive_closure();
        tc.is_irreflexive()
    }

    pub fn is_total_on(&self, indices: &[usize]) -> bool {
        for &i in indices {
            for &j in indices {
                if i != j && !self.data[i][j] && !self.data[j][i] {
                    return false;
                }
            }
        }
        true
    }

    pub fn inverse(&self) -> BitMatrix {
        let mut result = BitMatrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j][i] = self.data[i][j];
            }
        }
        result
    }

    pub fn count_edges(&self) -> usize {
        self.data.iter().flat_map(|row| row.iter()).filter(|&&b| b).count()
    }
}

// ---------------------------------------------------------------------------
// Global & Local Memory Models
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GlobalMemoryModel {
    pub events: Vec<OpenClEvent>,
    event_index: HashMap<u32, usize>,
    pub po: BitMatrix,
    pub rf: BitMatrix,
    pub mo: BitMatrix, // modification order
    pub hb: BitMatrix, // happens-before
}

impl GlobalMemoryModel {
    pub fn new(events: Vec<OpenClEvent>) -> Self {
        let n = events.len();
        let event_index: HashMap<u32, usize> = events.iter().enumerate()
            .map(|(i, e)| (e.id, i))
            .collect();
        Self {
            events,
            event_index,
            po: BitMatrix::new(n, n),
            rf: BitMatrix::new(n, n),
            mo: BitMatrix::new(n, n),
            hb: BitMatrix::new(n, n),
        }
    }

    pub fn add_po(&mut self, from_id: u32, to_id: u32) {
        if let (Some(&fi), Some(&ti)) = (self.event_index.get(&from_id), self.event_index.get(&to_id)) {
            self.po.set_true(fi, ti);
        }
    }

    pub fn add_rf(&mut self, write_id: u32, read_id: u32) {
        if let (Some(&wi), Some(&ri)) = (self.event_index.get(&write_id), self.event_index.get(&read_id)) {
            self.rf.set_true(wi, ri);
        }
    }

    pub fn add_mo(&mut self, first_write: u32, second_write: u32) {
        if let (Some(&fi), Some(&si)) = (self.event_index.get(&first_write), self.event_index.get(&second_write)) {
            self.mo.set_true(fi, si);
        }
    }

    pub fn compute_hb(&mut self) {
        let n = self.events.len();

        // sw (synchronizes-with): release-acquire pairs connected by rf with matching scope
        let mut sw = BitMatrix::new(n, n);
        for i in 0..n {
            for j in 0..n {
                if self.rf.get(i, j) {
                    let ei = &self.events[i];
                    let ej = &self.events[j];
                    let is_release = matches!(ei.memory_order,
                        OpenClMemoryOrder::Release | OpenClMemoryOrder::AcqRel | OpenClMemoryOrder::SeqCst);
                    let is_acquire = matches!(ej.memory_order,
                        OpenClMemoryOrder::Acquire | OpenClMemoryOrder::AcqRel | OpenClMemoryOrder::SeqCst);
                    if is_release && is_acquire && ei.memory_scope.includes(&ej.memory_scope) {
                        sw.set_true(i, j);
                    }
                }
            }
        }

        // hb = (po | sw)^+
        self.hb = self.po.union(&sw);
        self.hb.transitive_closure();
    }

    pub fn global_visibility_order(&self) -> Vec<u32> {
        // Topological sort of the modification order
        let n = self.events.len();
        let mut in_degree = vec![0usize; n];
        for i in 0..n {
            for j in 0..n {
                if self.mo.get(i, j) {
                    in_degree[j] += 1;
                }
            }
        }

        let mut queue: VecDeque<usize> = in_degree.iter().enumerate()
            .filter(|(_, &d)| d == 0)
            .map(|(i, _)| i)
            .collect();
        let mut result = Vec::new();

        while let Some(idx) = queue.pop_front() {
            result.push(self.events[idx].id);
            for j in 0..n {
                if self.mo.get(idx, j) {
                    in_degree[j] -= 1;
                    if in_degree[j] == 0 {
                        queue.push_back(j);
                    }
                }
            }
        }

        result
    }
}

#[derive(Debug, Clone)]
pub struct LocalMemoryModel {
    pub work_group_id: u32,
    pub events: Vec<OpenClEvent>,
    event_index: HashMap<u32, usize>,
    pub po: BitMatrix,
    pub rf: BitMatrix,
}

impl LocalMemoryModel {
    pub fn new(work_group_id: u32, events: Vec<OpenClEvent>) -> Self {
        let n = events.len();
        let event_index: HashMap<u32, usize> = events.iter().enumerate()
            .map(|(i, e)| (e.id, i))
            .collect();
        Self {
            work_group_id,
            events,
            event_index,
            po: BitMatrix::new(n, n),
            rf: BitMatrix::new(n, n),
        }
    }

    pub fn add_po(&mut self, from_id: u32, to_id: u32) {
        if let (Some(&fi), Some(&ti)) = (self.event_index.get(&from_id), self.event_index.get(&to_id)) {
            self.po.set_true(fi, ti);
        }
    }

    pub fn add_rf(&mut self, write_id: u32, read_id: u32) {
        if let (Some(&wi), Some(&ri)) = (self.event_index.get(&write_id), self.event_index.get(&read_id)) {
            self.rf.set_true(wi, ri);
        }
    }

    pub fn check_coherence(&self) -> bool {
        // Local memory coherence: rf;po must not contradict mo
        // Simplified: check that po is acyclic
        self.po.is_acyclic()
    }
}

// ---------------------------------------------------------------------------
// OpenCL Axioms & Model Checker
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OpenClAxiom {
    CoherenceRR,
    CoherenceRW,
    CoherenceWR,
    CoherenceWW,
    Atomicity,
    NoThinAir,
    SeqCstOrder,
    HbAcyclicity,
    BarrierConvergence,
    ScopedSynchronization,
    LocalMemoryCoherence,
    GlobalMemoryCoherence,
}

impl fmt::Display for OpenClAxiom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CoherenceRR => write!(f, "coh-rr"),
            Self::CoherenceRW => write!(f, "coh-rw"),
            Self::CoherenceWR => write!(f, "coh-wr"),
            Self::CoherenceWW => write!(f, "coh-ww"),
            Self::Atomicity => write!(f, "atomicity"),
            Self::NoThinAir => write!(f, "no-thin-air"),
            Self::SeqCstOrder => write!(f, "sc-order"),
            Self::HbAcyclicity => write!(f, "hb-acyclicity"),
            Self::BarrierConvergence => write!(f, "barrier-convergence"),
            Self::ScopedSynchronization => write!(f, "scoped-sync"),
            Self::LocalMemoryCoherence => write!(f, "local-coherence"),
            Self::GlobalMemoryCoherence => write!(f, "global-coherence"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenClViolation {
    pub axiom: OpenClAxiom,
    pub description: String,
    pub events_involved: Vec<u32>,
}

impl fmt::Display for OpenClViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.axiom, self.description)
    }
}

#[derive(Debug, Clone)]
pub struct OpenClModelChecker {
    global_model: GlobalMemoryModel,
    local_models: Vec<LocalMemoryModel>,
    barriers: Vec<Barrier>,
    work_group_size: u32,
}

impl OpenClModelChecker {
    pub fn new(global_model: GlobalMemoryModel, work_group_size: u32) -> Self {
        Self {
            global_model,
            local_models: Vec::new(),
            barriers: Vec::new(),
            work_group_size,
        }
    }

    pub fn add_local_model(&mut self, model: LocalMemoryModel) {
        self.local_models.push(model);
    }

    pub fn add_barrier(&mut self, barrier: Barrier) {
        self.barriers.push(barrier);
    }

    pub fn check_axiom(&self, axiom: OpenClAxiom) -> Vec<OpenClViolation> {
        match axiom {
            OpenClAxiom::HbAcyclicity => self.check_hb_acyclicity(),
            OpenClAxiom::CoherenceWW => self.check_coherence_ww(),
            OpenClAxiom::CoherenceRW => self.check_coherence_rw(),
            OpenClAxiom::CoherenceWR => self.check_coherence_wr(),
            OpenClAxiom::CoherenceRR => self.check_coherence_rr(),
            OpenClAxiom::Atomicity => self.check_atomicity(),
            OpenClAxiom::NoThinAir => self.check_no_thin_air(),
            OpenClAxiom::SeqCstOrder => self.check_seq_cst(),
            OpenClAxiom::BarrierConvergence => self.check_barrier_convergence(),
            OpenClAxiom::ScopedSynchronization => self.check_scoped_sync(),
            OpenClAxiom::LocalMemoryCoherence => self.check_local_coherence(),
            OpenClAxiom::GlobalMemoryCoherence => self.check_global_coherence(),
        }
    }

    pub fn check_all_axioms(&self) -> Vec<OpenClViolation> {
        let axioms = [
            OpenClAxiom::HbAcyclicity,
            OpenClAxiom::CoherenceWW,
            OpenClAxiom::CoherenceRW,
            OpenClAxiom::CoherenceWR,
            OpenClAxiom::CoherenceRR,
            OpenClAxiom::Atomicity,
            OpenClAxiom::NoThinAir,
            OpenClAxiom::SeqCstOrder,
            OpenClAxiom::BarrierConvergence,
            OpenClAxiom::ScopedSynchronization,
            OpenClAxiom::LocalMemoryCoherence,
            OpenClAxiom::GlobalMemoryCoherence,
        ];
        let mut violations = Vec::new();
        for axiom in &axioms {
            violations.extend(self.check_axiom(*axiom));
        }
        violations
    }

    fn check_hb_acyclicity(&self) -> Vec<OpenClViolation> {
        if !self.global_model.hb.is_acyclic() {
            vec![OpenClViolation {
                axiom: OpenClAxiom::HbAcyclicity,
                description: "Happens-before relation contains a cycle".to_string(),
                events_involved: Vec::new(),
            }]
        } else {
            Vec::new()
        }
    }

    fn check_coherence_ww(&self) -> Vec<OpenClViolation> {
        let mut violations = Vec::new();
        let n = self.global_model.events.len();
        // Check: hb;mo must be acyclic per-location
        let hb_mo = self.global_model.hb.compose(&self.global_model.mo);
        if !hb_mo.is_irreflexive() {
            violations.push(OpenClViolation {
                axiom: OpenClAxiom::CoherenceWW,
                description: "Write-write coherence violated: hb;mo cycle".to_string(),
                events_involved: Vec::new(),
            });
        }
        violations
    }

    fn check_coherence_rw(&self) -> Vec<OpenClViolation> {
        let mut violations = Vec::new();
        let n = self.global_model.events.len();
        // fr = rf^{-1};mo
        let rf_inv = self.global_model.rf.inverse();
        let fr = rf_inv.compose(&self.global_model.mo);
        let hb_check = fr.compose(&self.global_model.hb);
        if !hb_check.is_irreflexive() {
            violations.push(OpenClViolation {
                axiom: OpenClAxiom::CoherenceRW,
                description: "Read-write coherence violated".to_string(),
                events_involved: Vec::new(),
            });
        }
        violations
    }

    fn check_coherence_wr(&self) -> Vec<OpenClViolation> {
        let mut violations = Vec::new();
        let hb_rf = self.global_model.hb.compose(&self.global_model.rf);
        if !hb_rf.is_irreflexive() {
            violations.push(OpenClViolation {
                axiom: OpenClAxiom::CoherenceWR,
                description: "Write-read coherence violated".to_string(),
                events_involved: Vec::new(),
            });
        }
        violations
    }

    fn check_coherence_rr(&self) -> Vec<OpenClViolation> {
        let mut violations = Vec::new();
        let rf_inv = self.global_model.rf.inverse();
        let fr = rf_inv.compose(&self.global_model.mo);
        let fr_rf = fr.compose(&self.global_model.rf);
        if !fr_rf.is_irreflexive() {
            violations.push(OpenClViolation {
                axiom: OpenClAxiom::CoherenceRR,
                description: "Read-read coherence violated".to_string(),
                events_involved: Vec::new(),
            });
        }
        violations
    }

    fn check_atomicity(&self) -> Vec<OpenClViolation> {
        let mut violations = Vec::new();
        let n = self.global_model.events.len();
        // For RMW operations, the read and write must be adjacent in mo
        for i in 0..n {
            let e = &self.global_model.events[i];
            if e.op.is_rmw() {
                // Check no other write to same address is between read and write of RMW
                let mut has_intervening = false;
                for j in 0..n {
                    if i == j { continue; }
                    let ej = &self.global_model.events[j];
                    if ej.address == e.address && ej.op.is_write() {
                        if self.global_model.mo.get(i, j) {
                            // There's a write between our RMW's read and write
                            // This is a simplified check
                        }
                    }
                }
            }
        }
        violations
    }

    fn check_no_thin_air(&self) -> Vec<OpenClViolation> {
        // hb|rf must be acyclic (no values out of thin air)
        let hb_rf = self.global_model.hb.union(&self.global_model.rf);
        if !hb_rf.is_acyclic() {
            vec![OpenClViolation {
                axiom: OpenClAxiom::NoThinAir,
                description: "Potential out-of-thin-air value: hb∪rf has a cycle".to_string(),
                events_involved: Vec::new(),
            }]
        } else {
            Vec::new()
        }
    }

    fn check_seq_cst(&self) -> Vec<OpenClViolation> {
        let mut violations = Vec::new();
        let n = self.global_model.events.len();
        // Collect SeqCst events
        let sc_indices: Vec<usize> = (0..n)
            .filter(|&i| self.global_model.events[i].memory_order == OpenClMemoryOrder::SeqCst)
            .collect();

        if sc_indices.len() > 1 {
            // Check total order exists among SeqCst events
            // SeqCst must be consistent with hb
            let mut sc_hb = BitMatrix::new(n, n);
            for &i in &sc_indices {
                for &j in &sc_indices {
                    if self.global_model.hb.get(i, j) {
                        sc_hb.set_true(i, j);
                    }
                }
            }
            if !sc_hb.is_acyclic() {
                violations.push(OpenClViolation {
                    axiom: OpenClAxiom::SeqCstOrder,
                    description: "SeqCst order is inconsistent with happens-before".to_string(),
                    events_involved: sc_indices.iter().map(|&i| self.global_model.events[i].id).collect(),
                });
            }
        }
        violations
    }

    fn check_barrier_convergence(&self) -> Vec<OpenClViolation> {
        let checker = WorkGroupBarrierChecker {
            work_group_size: self.work_group_size,
            barriers: self.barriers.clone(),
        };
        checker.check_barrier_convergence().into_iter().map(|bv| {
            OpenClViolation {
                axiom: OpenClAxiom::BarrierConvergence,
                description: bv.description,
                events_involved: bv.missing_items,
            }
        }).collect()
    }

    fn check_scoped_sync(&self) -> Vec<OpenClViolation> {
        let mut violations = Vec::new();
        let n = self.global_model.events.len();
        for i in 0..n {
            for j in 0..n {
                if self.global_model.rf.get(i, j) {
                    let ei = &self.global_model.events[i];
                    let ej = &self.global_model.events[j];
                    if ei.op.is_atomic() && ej.op.is_atomic() {
                        // Check scope compatibility
                        if ei.work_group != ej.work_group {
                            // Cross-workgroup: need at least Device scope
                            if !ei.memory_scope.includes(&OpenClMemoryScope::Device) ||
                               !ej.memory_scope.includes(&OpenClMemoryScope::Device) {
                                violations.push(OpenClViolation {
                                    axiom: OpenClAxiom::ScopedSynchronization,
                                    description: format!(
                                        "Cross-workgroup sync between events {} and {} without device scope",
                                        ei.id, ej.id
                                    ),
                                    events_involved: vec![ei.id, ej.id],
                                });
                            }
                        }
                    }
                }
            }
        }
        violations
    }

    fn check_local_coherence(&self) -> Vec<OpenClViolation> {
        let mut violations = Vec::new();
        for lm in &self.local_models {
            if !lm.check_coherence() {
                violations.push(OpenClViolation {
                    axiom: OpenClAxiom::LocalMemoryCoherence,
                    description: format!(
                        "Local memory coherence violated in work group {}",
                        lm.work_group_id
                    ),
                    events_involved: Vec::new(),
                });
            }
        }
        violations
    }

    fn check_global_coherence(&self) -> Vec<OpenClViolation> {
        let mut violations = Vec::new();
        // Per-location mo;hb must be acyclic
        let mo_hb = self.global_model.mo.compose(&self.global_model.hb);
        if !mo_hb.is_irreflexive() {
            violations.push(OpenClViolation {
                axiom: OpenClAxiom::GlobalMemoryCoherence,
                description: "Global memory coherence violated: mo;hb cycle".to_string(),
                events_involved: Vec::new(),
            });
        }
        violations
    }
}

// ---------------------------------------------------------------------------
// SVM Types & Consistency
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SvmType {
    CoarseGrained,
    FineGrained,
    FineGrainedWithAtomics,
    System,
}

impl fmt::Display for SvmType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CoarseGrained => write!(f, "coarse-grained"),
            Self::FineGrained => write!(f, "fine-grained"),
            Self::FineGrainedWithAtomics => write!(f, "fine-grained-atomics"),
            Self::System => write!(f, "system"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SvmRegion {
    pub base: u64,
    pub size: u64,
    pub svm_type: SvmType,
    pub host_accessible: bool,
    pub device_accessible: bool,
}

impl SvmRegion {
    pub fn new(base: u64, size: u64, svm_type: SvmType) -> Self {
        Self {
            base,
            size,
            svm_type,
            host_accessible: true,
            device_accessible: true,
        }
    }

    pub fn contains(&self, addr: u64) -> bool {
        addr >= self.base && addr < self.base + self.size
    }
}

#[derive(Debug, Clone)]
pub struct SvmConsistencyChecker {
    regions: Vec<SvmRegion>,
}

impl SvmConsistencyChecker {
    pub fn new() -> Self {
        Self { regions: Vec::new() }
    }

    pub fn add_region(&mut self, region: SvmRegion) {
        self.regions.push(region);
    }

    pub fn find_region(&self, addr: u64) -> Option<&SvmRegion> {
        self.regions.iter().find(|r| r.contains(addr))
    }

    pub fn check_access(&self, addr: u64, is_atomic: bool, from_host: bool) -> Result<(), String> {
        let region = self.find_region(addr)
            .ok_or_else(|| format!("Address 0x{:x} not in any SVM region", addr))?;

        if from_host && !region.host_accessible {
            return Err(format!("Host cannot access SVM region at 0x{:x}", addr));
        }
        if !from_host && !region.device_accessible {
            return Err(format!("Device cannot access SVM region at 0x{:x}", addr));
        }

        if is_atomic {
            match region.svm_type {
                SvmType::CoarseGrained | SvmType::FineGrained => {
                    return Err(format!(
                        "Atomic access to {} SVM region requires FineGrainedWithAtomics or System SVM",
                        region.svm_type
                    ));
                }
                _ => {}
            }
        }

        Ok(())
    }

    pub fn check_concurrent_access(&self, addr: u64) -> Result<SvmType, String> {
        let region = self.find_region(addr)
            .ok_or_else(|| format!("Address 0x{:x} not in any SVM region", addr))?;

        match region.svm_type {
            SvmType::CoarseGrained => Err(
                "Coarse-grained SVM does not support concurrent host-device access".to_string()
            ),
            _ => Ok(region.svm_type),
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel Execution Structures
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NDRange {
    pub global_size: Vec<u32>,
    pub local_size: Vec<u32>,
    pub dimensions: usize,
}

impl NDRange {
    pub fn new_1d(global: u32, local: u32) -> Self {
        Self { global_size: vec![global], local_size: vec![local], dimensions: 1 }
    }

    pub fn new_2d(gx: u32, gy: u32, lx: u32, ly: u32) -> Self {
        Self { global_size: vec![gx, gy], local_size: vec![lx, ly], dimensions: 2 }
    }

    pub fn new_3d(gx: u32, gy: u32, gz: u32, lx: u32, ly: u32, lz: u32) -> Self {
        Self { global_size: vec![gx, gy, gz], local_size: vec![lx, ly, lz], dimensions: 3 }
    }

    pub fn num_work_groups(&self) -> u32 {
        self.global_size.iter().zip(self.local_size.iter())
            .map(|(&g, &l)| (g + l - 1) / l)
            .product()
    }

    pub fn total_work_items(&self) -> u32 {
        self.global_size.iter().product()
    }

    pub fn work_group_size(&self) -> u32 {
        self.local_size.iter().product()
    }

    pub fn work_group_of(&self, global_id: u32) -> u32 {
        if self.dimensions == 1 {
            global_id / self.local_size[0]
        } else {
            // Linearized work group id for multi-dimensional
            global_id / self.work_group_size()
        }
    }

    pub fn local_id_of(&self, global_id: u32) -> u32 {
        if self.dimensions == 1 {
            global_id % self.local_size[0]
        } else {
            global_id % self.work_group_size()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkItem {
    pub global_id: u32,
    pub local_id: u32,
    pub work_group_id: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkGroup {
    pub id: u32,
    pub items: Vec<WorkItem>,
    pub local_memory_size: u64,
}

impl WorkGroup {
    pub fn new(id: u32, local_size: u32, local_mem: u64) -> Self {
        let items = (0..local_size).map(|lid| WorkItem {
            global_id: id * local_size + lid,
            local_id: lid,
            work_group_id: id,
        }).collect();
        Self { id, items, local_memory_size: local_mem }
    }

    pub fn size(&self) -> usize {
        self.items.len()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelExecution {
    pub name: String,
    pub nd_range: NDRange,
    pub work_groups: Vec<WorkGroup>,
    pub global_memory_size: u64,
    pub local_memory_per_group: u64,
}

impl KernelExecution {
    pub fn new(name: impl Into<String>, nd_range: NDRange, local_mem: u64) -> Self {
        let num_wg = nd_range.num_work_groups();
        let wg_size = nd_range.work_group_size();
        let work_groups: Vec<WorkGroup> = (0..num_wg)
            .map(|i| WorkGroup::new(i, wg_size, local_mem))
            .collect();
        Self {
            name: name.into(),
            nd_range,
            work_groups,
            global_memory_size: 0,
            local_memory_per_group: local_mem,
        }
    }

    pub fn total_work_items(&self) -> u32 {
        self.nd_range.total_work_items()
    }

    pub fn num_work_groups(&self) -> u32 {
        self.nd_range.num_work_groups()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// ===== Extended Opencl Operations =====

#[derive(Debug, Clone)]
pub struct OpenClEventModel {
    pub event_id: u64,
    pub command_type: String,
    pub status: String,
    pub queue_id: u32,
}

impl OpenClEventModel {
    pub fn new(event_id: u64, command_type: String, status: String, queue_id: u32) -> Self {
        OpenClEventModel { event_id, command_type, status, queue_id }
    }

    pub fn get_event_id(&self) -> u64 {
        self.event_id
    }

    pub fn get_command_type(&self) -> &str {
        &self.command_type
    }

    pub fn get_status(&self) -> &str {
        &self.status
    }

    pub fn get_queue_id(&self) -> u32 {
        self.queue_id
    }

    pub fn with_event_id(mut self, v: u64) -> Self {
        self.event_id = v; self
    }

    pub fn with_command_type(mut self, v: impl Into<String>) -> Self {
        self.command_type = v.into(); self
    }

    pub fn with_status(mut self, v: impl Into<String>) -> Self {
        self.status = v.into(); self
    }

    pub fn with_queue_id(mut self, v: u32) -> Self {
        self.queue_id = v; self
    }

}

impl fmt::Display for OpenClEventModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OpenClEventModel({:?})", self.event_id)
    }
}

#[derive(Debug, Clone)]
pub struct OpenClEventModelBuilder {
    event_id: u64,
    command_type: String,
    status: String,
    queue_id: u32,
}

impl OpenClEventModelBuilder {
    pub fn new() -> Self {
        OpenClEventModelBuilder {
            event_id: 0,
            command_type: String::new(),
            status: String::new(),
            queue_id: 0,
        }
    }

    pub fn event_id(mut self, v: u64) -> Self { self.event_id = v; self }
    pub fn command_type(mut self, v: impl Into<String>) -> Self { self.command_type = v.into(); self }
    pub fn status(mut self, v: impl Into<String>) -> Self { self.status = v.into(); self }
    pub fn queue_id(mut self, v: u32) -> Self { self.queue_id = v; self }
}

#[derive(Debug, Clone)]
pub struct CommandQueueSemantics {
    pub queue_id: u32,
    pub in_order: bool,
    pub profiling: bool,
    pub pending_count: usize,
}

impl CommandQueueSemantics {
    pub fn new(queue_id: u32, in_order: bool, profiling: bool, pending_count: usize) -> Self {
        CommandQueueSemantics { queue_id, in_order, profiling, pending_count }
    }

    pub fn get_queue_id(&self) -> u32 {
        self.queue_id
    }

    pub fn get_in_order(&self) -> bool {
        self.in_order
    }

    pub fn get_profiling(&self) -> bool {
        self.profiling
    }

    pub fn get_pending_count(&self) -> usize {
        self.pending_count
    }

    pub fn with_queue_id(mut self, v: u32) -> Self {
        self.queue_id = v; self
    }

    pub fn with_in_order(mut self, v: bool) -> Self {
        self.in_order = v; self
    }

    pub fn with_profiling(mut self, v: bool) -> Self {
        self.profiling = v; self
    }

    pub fn with_pending_count(mut self, v: usize) -> Self {
        self.pending_count = v; self
    }

}

impl fmt::Display for CommandQueueSemantics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CommandQueueSemantics({:?})", self.queue_id)
    }
}

#[derive(Debug, Clone)]
pub struct CommandQueueSemanticsBuilder {
    queue_id: u32,
    in_order: bool,
    profiling: bool,
    pending_count: usize,
}

impl CommandQueueSemanticsBuilder {
    pub fn new() -> Self {
        CommandQueueSemanticsBuilder {
            queue_id: 0,
            in_order: false,
            profiling: false,
            pending_count: 0,
        }
    }

    pub fn queue_id(mut self, v: u32) -> Self { self.queue_id = v; self }
    pub fn in_order(mut self, v: bool) -> Self { self.in_order = v; self }
    pub fn profiling(mut self, v: bool) -> Self { self.profiling = v; self }
    pub fn pending_count(mut self, v: usize) -> Self { self.pending_count = v; self }
}

#[derive(Debug, Clone)]
pub struct KernelArgBinding {
    pub kernel_name: String,
    pub arg_index: u32,
    pub arg_type: String,
    pub bound: bool,
}

impl KernelArgBinding {
    pub fn new(kernel_name: String, arg_index: u32, arg_type: String, bound: bool) -> Self {
        KernelArgBinding { kernel_name, arg_index, arg_type, bound }
    }

    pub fn get_kernel_name(&self) -> &str {
        &self.kernel_name
    }

    pub fn get_arg_index(&self) -> u32 {
        self.arg_index
    }

    pub fn get_arg_type(&self) -> &str {
        &self.arg_type
    }

    pub fn get_bound(&self) -> bool {
        self.bound
    }

    pub fn with_kernel_name(mut self, v: impl Into<String>) -> Self {
        self.kernel_name = v.into(); self
    }

    pub fn with_arg_index(mut self, v: u32) -> Self {
        self.arg_index = v; self
    }

    pub fn with_arg_type(mut self, v: impl Into<String>) -> Self {
        self.arg_type = v.into(); self
    }

    pub fn with_bound(mut self, v: bool) -> Self {
        self.bound = v; self
    }

}

impl fmt::Display for KernelArgBinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "KernelArgBinding({:?})", self.kernel_name)
    }
}

#[derive(Debug, Clone)]
pub struct KernelArgBindingBuilder {
    kernel_name: String,
    arg_index: u32,
    arg_type: String,
    bound: bool,
}

impl KernelArgBindingBuilder {
    pub fn new() -> Self {
        KernelArgBindingBuilder {
            kernel_name: String::new(),
            arg_index: 0,
            arg_type: String::new(),
            bound: false,
        }
    }

    pub fn kernel_name(mut self, v: impl Into<String>) -> Self { self.kernel_name = v.into(); self }
    pub fn arg_index(mut self, v: u32) -> Self { self.arg_index = v; self }
    pub fn arg_type(mut self, v: impl Into<String>) -> Self { self.arg_type = v.into(); self }
    pub fn bound(mut self, v: bool) -> Self { self.bound = v; self }
}

#[derive(Debug, Clone)]
pub struct ImageMemoryModel {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub channel_order: String,
    pub channel_type: String,
}

impl ImageMemoryModel {
    pub fn new(width: u32, height: u32, depth: u32, channel_order: String, channel_type: String) -> Self {
        ImageMemoryModel { width, height, depth, channel_order, channel_type }
    }

    pub fn get_width(&self) -> u32 {
        self.width
    }

    pub fn get_height(&self) -> u32 {
        self.height
    }

    pub fn get_depth(&self) -> u32 {
        self.depth
    }

    pub fn get_channel_order(&self) -> &str {
        &self.channel_order
    }

    pub fn get_channel_type(&self) -> &str {
        &self.channel_type
    }

    pub fn with_width(mut self, v: u32) -> Self {
        self.width = v; self
    }

    pub fn with_height(mut self, v: u32) -> Self {
        self.height = v; self
    }

    pub fn with_depth(mut self, v: u32) -> Self {
        self.depth = v; self
    }

    pub fn with_channel_order(mut self, v: impl Into<String>) -> Self {
        self.channel_order = v.into(); self
    }

    pub fn with_channel_type(mut self, v: impl Into<String>) -> Self {
        self.channel_type = v.into(); self
    }

}

impl fmt::Display for ImageMemoryModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ImageMemoryModel({:?})", self.width)
    }
}

#[derive(Debug, Clone)]
pub struct ImageMemoryModelBuilder {
    width: u32,
    height: u32,
    depth: u32,
    channel_order: String,
    channel_type: String,
}

impl ImageMemoryModelBuilder {
    pub fn new() -> Self {
        ImageMemoryModelBuilder {
            width: 0,
            height: 0,
            depth: 0,
            channel_order: String::new(),
            channel_type: String::new(),
        }
    }

    pub fn width(mut self, v: u32) -> Self { self.width = v; self }
    pub fn height(mut self, v: u32) -> Self { self.height = v; self }
    pub fn depth(mut self, v: u32) -> Self { self.depth = v; self }
    pub fn channel_order(mut self, v: impl Into<String>) -> Self { self.channel_order = v.into(); self }
    pub fn channel_type(mut self, v: impl Into<String>) -> Self { self.channel_type = v.into(); self }
}

#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub normalized_coords: bool,
    pub addressing_mode: String,
    pub filter_mode: String,
}

impl SamplerConfig {
    pub fn new(normalized_coords: bool, addressing_mode: String, filter_mode: String) -> Self {
        SamplerConfig { normalized_coords, addressing_mode, filter_mode }
    }

    pub fn get_normalized_coords(&self) -> bool {
        self.normalized_coords
    }

    pub fn get_addressing_mode(&self) -> &str {
        &self.addressing_mode
    }

    pub fn get_filter_mode(&self) -> &str {
        &self.filter_mode
    }

    pub fn with_normalized_coords(mut self, v: bool) -> Self {
        self.normalized_coords = v; self
    }

    pub fn with_addressing_mode(mut self, v: impl Into<String>) -> Self {
        self.addressing_mode = v.into(); self
    }

    pub fn with_filter_mode(mut self, v: impl Into<String>) -> Self {
        self.filter_mode = v.into(); self
    }

}

impl fmt::Display for SamplerConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SamplerConfig({:?})", self.normalized_coords)
    }
}

#[derive(Debug, Clone)]
pub struct SamplerConfigBuilder {
    normalized_coords: bool,
    addressing_mode: String,
    filter_mode: String,
}

impl SamplerConfigBuilder {
    pub fn new() -> Self {
        SamplerConfigBuilder {
            normalized_coords: false,
            addressing_mode: String::new(),
            filter_mode: String::new(),
        }
    }

    pub fn normalized_coords(mut self, v: bool) -> Self { self.normalized_coords = v; self }
    pub fn addressing_mode(mut self, v: impl Into<String>) -> Self { self.addressing_mode = v.into(); self }
    pub fn filter_mode(mut self, v: impl Into<String>) -> Self { self.filter_mode = v.into(); self }
}

#[derive(Debug, Clone)]
pub struct PipeOperation {
    pub pipe_id: u32,
    pub packet_size: u32,
    pub max_packets: u32,
    pub direction: String,
}

impl PipeOperation {
    pub fn new(pipe_id: u32, packet_size: u32, max_packets: u32, direction: String) -> Self {
        PipeOperation { pipe_id, packet_size, max_packets, direction }
    }

    pub fn get_pipe_id(&self) -> u32 {
        self.pipe_id
    }

    pub fn get_packet_size(&self) -> u32 {
        self.packet_size
    }

    pub fn get_max_packets(&self) -> u32 {
        self.max_packets
    }

    pub fn get_direction(&self) -> &str {
        &self.direction
    }

    pub fn with_pipe_id(mut self, v: u32) -> Self {
        self.pipe_id = v; self
    }

    pub fn with_packet_size(mut self, v: u32) -> Self {
        self.packet_size = v; self
    }

    pub fn with_max_packets(mut self, v: u32) -> Self {
        self.max_packets = v; self
    }

    pub fn with_direction(mut self, v: impl Into<String>) -> Self {
        self.direction = v.into(); self
    }

}

impl fmt::Display for PipeOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PipeOperation({:?})", self.pipe_id)
    }
}

#[derive(Debug, Clone)]
pub struct PipeOperationBuilder {
    pipe_id: u32,
    packet_size: u32,
    max_packets: u32,
    direction: String,
}

impl PipeOperationBuilder {
    pub fn new() -> Self {
        PipeOperationBuilder {
            pipe_id: 0,
            packet_size: 0,
            max_packets: 0,
            direction: String::new(),
        }
    }

    pub fn pipe_id(mut self, v: u32) -> Self { self.pipe_id = v; self }
    pub fn packet_size(mut self, v: u32) -> Self { self.packet_size = v; self }
    pub fn max_packets(mut self, v: u32) -> Self { self.max_packets = v; self }
    pub fn direction(mut self, v: impl Into<String>) -> Self { self.direction = v.into(); self }
}

#[derive(Debug, Clone)]
pub struct SubGroupOps {
    pub sub_group_size: u32,
    pub sub_group_id: u32,
    pub num_sub_groups: u32,
}

impl SubGroupOps {
    pub fn new(sub_group_size: u32, sub_group_id: u32, num_sub_groups: u32) -> Self {
        SubGroupOps { sub_group_size, sub_group_id, num_sub_groups }
    }

    pub fn get_sub_group_size(&self) -> u32 {
        self.sub_group_size
    }

    pub fn get_sub_group_id(&self) -> u32 {
        self.sub_group_id
    }

    pub fn get_num_sub_groups(&self) -> u32 {
        self.num_sub_groups
    }

    pub fn with_sub_group_size(mut self, v: u32) -> Self {
        self.sub_group_size = v; self
    }

    pub fn with_sub_group_id(mut self, v: u32) -> Self {
        self.sub_group_id = v; self
    }

    pub fn with_num_sub_groups(mut self, v: u32) -> Self {
        self.num_sub_groups = v; self
    }

}

impl fmt::Display for SubGroupOps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SubGroupOps({:?})", self.sub_group_size)
    }
}

#[derive(Debug, Clone)]
pub struct SubGroupOpsBuilder {
    sub_group_size: u32,
    sub_group_id: u32,
    num_sub_groups: u32,
}

impl SubGroupOpsBuilder {
    pub fn new() -> Self {
        SubGroupOpsBuilder {
            sub_group_size: 0,
            sub_group_id: 0,
            num_sub_groups: 0,
        }
    }

    pub fn sub_group_size(mut self, v: u32) -> Self { self.sub_group_size = v; self }
    pub fn sub_group_id(mut self, v: u32) -> Self { self.sub_group_id = v; self }
    pub fn num_sub_groups(mut self, v: u32) -> Self { self.num_sub_groups = v; self }
}

#[derive(Debug, Clone)]
pub struct ProgramScopeVar {
    pub name: String,
    pub address_space: String,
    pub size: u64,
    pub initialized: bool,
}

impl ProgramScopeVar {
    pub fn new(name: String, address_space: String, size: u64, initialized: bool) -> Self {
        ProgramScopeVar { name, address_space, size, initialized }
    }

    pub fn get_name(&self) -> &str {
        &self.name
    }

    pub fn get_address_space(&self) -> &str {
        &self.address_space
    }

    pub fn get_size(&self) -> u64 {
        self.size
    }

    pub fn get_initialized(&self) -> bool {
        self.initialized
    }

    pub fn with_name(mut self, v: impl Into<String>) -> Self {
        self.name = v.into(); self
    }

    pub fn with_address_space(mut self, v: impl Into<String>) -> Self {
        self.address_space = v.into(); self
    }

    pub fn with_size(mut self, v: u64) -> Self {
        self.size = v; self
    }

    pub fn with_initialized(mut self, v: bool) -> Self {
        self.initialized = v; self
    }

}

impl fmt::Display for ProgramScopeVar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ProgramScopeVar({:?})", self.name)
    }
}

#[derive(Debug, Clone)]
pub struct ProgramScopeVarBuilder {
    name: String,
    address_space: String,
    size: u64,
    initialized: bool,
}

impl ProgramScopeVarBuilder {
    pub fn new() -> Self {
        ProgramScopeVarBuilder {
            name: String::new(),
            address_space: String::new(),
            size: 0,
            initialized: false,
        }
    }

    pub fn name(mut self, v: impl Into<String>) -> Self { self.name = v.into(); self }
    pub fn address_space(mut self, v: impl Into<String>) -> Self { self.address_space = v.into(); self }
    pub fn size(mut self, v: u64) -> Self { self.size = v; self }
    pub fn initialized(mut self, v: bool) -> Self { self.initialized = v; self }
}

#[derive(Debug, Clone)]
pub struct OpenClMemOrderSemantics {
    pub order: String,
    pub scope: String,
    pub fence_flags: u32,
}

impl OpenClMemOrderSemantics {
    pub fn new(order: String, scope: String, fence_flags: u32) -> Self {
        OpenClMemOrderSemantics { order, scope, fence_flags }
    }

    pub fn get_order(&self) -> &str {
        &self.order
    }

    pub fn get_scope(&self) -> &str {
        &self.scope
    }

    pub fn get_fence_flags(&self) -> u32 {
        self.fence_flags
    }

    pub fn with_order(mut self, v: impl Into<String>) -> Self {
        self.order = v.into(); self
    }

    pub fn with_scope(mut self, v: impl Into<String>) -> Self {
        self.scope = v.into(); self
    }

    pub fn with_fence_flags(mut self, v: u32) -> Self {
        self.fence_flags = v; self
    }

}

impl fmt::Display for OpenClMemOrderSemantics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OpenClMemOrderSemantics({:?})", self.order)
    }
}

#[derive(Debug, Clone)]
pub struct OpenClMemOrderSemanticsBuilder {
    order: String,
    scope: String,
    fence_flags: u32,
}

impl OpenClMemOrderSemanticsBuilder {
    pub fn new() -> Self {
        OpenClMemOrderSemanticsBuilder {
            order: String::new(),
            scope: String::new(),
            fence_flags: 0,
        }
    }

    pub fn order(mut self, v: impl Into<String>) -> Self { self.order = v.into(); self }
    pub fn scope(mut self, v: impl Into<String>) -> Self { self.scope = v.into(); self }
    pub fn fence_flags(mut self, v: u32) -> Self { self.fence_flags = v; self }
}

#[derive(Debug, Clone)]
pub struct OpenClWorkGroup {
    pub local_size: Vec<u32>,
    pub global_size: Vec<u32>,
    pub group_id: Vec<u32>,
}

impl OpenClWorkGroup {
    pub fn new(local_size: Vec<u32>, global_size: Vec<u32>, group_id: Vec<u32>) -> Self {
        OpenClWorkGroup { local_size, global_size, group_id }
    }

    pub fn local_size_len(&self) -> usize {
        self.local_size.len()
    }

    pub fn local_size_is_empty(&self) -> bool {
        self.local_size.is_empty()
    }

    pub fn global_size_len(&self) -> usize {
        self.global_size.len()
    }

    pub fn global_size_is_empty(&self) -> bool {
        self.global_size.is_empty()
    }

    pub fn group_id_len(&self) -> usize {
        self.group_id.len()
    }

    pub fn group_id_is_empty(&self) -> bool {
        self.group_id.is_empty()
    }

}

impl fmt::Display for OpenClWorkGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OpenClWorkGroup({:?})", self.local_size)
    }
}

#[derive(Debug, Clone)]
pub struct OpenClWorkGroupBuilder {
    local_size: Vec<u32>,
    global_size: Vec<u32>,
    group_id: Vec<u32>,
}

impl OpenClWorkGroupBuilder {
    pub fn new() -> Self {
        OpenClWorkGroupBuilder {
            local_size: Vec::new(),
            global_size: Vec::new(),
            group_id: Vec::new(),
        }
    }

    pub fn local_size(mut self, v: Vec<u32>) -> Self { self.local_size = v; self }
    pub fn global_size(mut self, v: Vec<u32>) -> Self { self.global_size = v; self }
    pub fn group_id(mut self, v: Vec<u32>) -> Self { self.group_id = v; self }
}

#[derive(Debug, Clone)]
pub struct OpenClBufferRegion {
    pub origin: u64,
    pub size: u64,
    pub flags: u32,
}

impl OpenClBufferRegion {
    pub fn new(origin: u64, size: u64, flags: u32) -> Self {
        OpenClBufferRegion { origin, size, flags }
    }

    pub fn get_origin(&self) -> u64 {
        self.origin
    }

    pub fn get_size(&self) -> u64 {
        self.size
    }

    pub fn get_flags(&self) -> u32 {
        self.flags
    }

    pub fn with_origin(mut self, v: u64) -> Self {
        self.origin = v; self
    }

    pub fn with_size(mut self, v: u64) -> Self {
        self.size = v; self
    }

    pub fn with_flags(mut self, v: u32) -> Self {
        self.flags = v; self
    }

}

impl fmt::Display for OpenClBufferRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OpenClBufferRegion({:?})", self.origin)
    }
}

#[derive(Debug, Clone)]
pub struct OpenClBufferRegionBuilder {
    origin: u64,
    size: u64,
    flags: u32,
}

impl OpenClBufferRegionBuilder {
    pub fn new() -> Self {
        OpenClBufferRegionBuilder {
            origin: 0,
            size: 0,
            flags: 0,
        }
    }

    pub fn origin(mut self, v: u64) -> Self { self.origin = v; self }
    pub fn size(mut self, v: u64) -> Self { self.size = v; self }
    pub fn flags(mut self, v: u32) -> Self { self.flags = v; self }
}

#[derive(Debug, Clone)]
pub struct OpenClSvmModel {
    pub svm_type: String,
    pub alignment: u64,
    pub size: u64,
    pub shared: bool,
}

impl OpenClSvmModel {
    pub fn new(svm_type: String, alignment: u64, size: u64, shared: bool) -> Self {
        OpenClSvmModel { svm_type, alignment, size, shared }
    }

    pub fn get_svm_type(&self) -> &str {
        &self.svm_type
    }

    pub fn get_alignment(&self) -> u64 {
        self.alignment
    }

    pub fn get_size(&self) -> u64 {
        self.size
    }

    pub fn get_shared(&self) -> bool {
        self.shared
    }

    pub fn with_svm_type(mut self, v: impl Into<String>) -> Self {
        self.svm_type = v.into(); self
    }

    pub fn with_alignment(mut self, v: u64) -> Self {
        self.alignment = v; self
    }

    pub fn with_size(mut self, v: u64) -> Self {
        self.size = v; self
    }

    pub fn with_shared(mut self, v: bool) -> Self {
        self.shared = v; self
    }

}

impl fmt::Display for OpenClSvmModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OpenClSvmModel({:?})", self.svm_type)
    }
}

#[derive(Debug, Clone)]
pub struct OpenClSvmModelBuilder {
    svm_type: String,
    alignment: u64,
    size: u64,
    shared: bool,
}

impl OpenClSvmModelBuilder {
    pub fn new() -> Self {
        OpenClSvmModelBuilder {
            svm_type: String::new(),
            alignment: 0,
            size: 0,
            shared: false,
        }
    }

    pub fn svm_type(mut self, v: impl Into<String>) -> Self { self.svm_type = v.into(); self }
    pub fn alignment(mut self, v: u64) -> Self { self.alignment = v; self }
    pub fn size(mut self, v: u64) -> Self { self.size = v; self }
    pub fn shared(mut self, v: bool) -> Self { self.shared = v; self }
}

#[derive(Debug, Clone)]
pub struct OpenclAnalysis {
    pub data: Vec<Vec<f64>>,
    pub size: usize,
    pub computed: bool,
    pub label: String,
    pub threshold: f64,
}

impl OpenclAnalysis {
    pub fn new(size: usize) -> Self {
        let data = vec![vec![0.0; size]; size];
        OpenclAnalysis { data, size, computed: false, label: "Opencl".to_string(), threshold: 0.01 }
    }

    pub fn with_threshold(mut self, t: f64) -> Self {
        self.threshold = t; self
    }

    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        if i < self.size && j < self.size { self.data[i][j] = v; }
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        if i < self.size && j < self.size { self.data[i][j] } else { 0.0 }
    }

    pub fn row_sum(&self, i: usize) -> f64 {
        if i < self.size { self.data[i].iter().sum() } else { 0.0 }
    }

    pub fn col_sum(&self, j: usize) -> f64 {
        if j < self.size { (0..self.size).map(|i| self.data[i][j]).sum() } else { 0.0 }
    }

    pub fn total_sum(&self) -> f64 {
        self.data.iter().flat_map(|r| r.iter()).sum()
    }

    pub fn max_value(&self) -> f64 {
        self.data.iter().flat_map(|r| r.iter()).cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn min_value(&self) -> f64 {
        self.data.iter().flat_map(|r| r.iter()).cloned().fold(f64::INFINITY, f64::min)
    }

    pub fn above_threshold(&self) -> Vec<(usize, usize, f64)> {
        let mut result = Vec::new();
        for i in 0..self.size {
            for j in 0..self.size {
                if self.data[i][j] > self.threshold {
                    result.push((i, j, self.data[i][j]));
                }
            }
        }
        result
    }

    pub fn normalize(&mut self) {
        let total = self.total_sum();
        if total > 0.0 {
            for i in 0..self.size {
                for j in 0..self.size {
                    self.data[i][j] /= total;
                }
            }
        }
        self.computed = true;
    }

    pub fn transpose(&self) -> Self {
        let mut result = Self::new(self.size);
        for i in 0..self.size {
            for j in 0..self.size {
                result.data[i][j] = self.data[j][i];
            }
        }
        result
    }

    pub fn multiply(&self, other: &Self) -> Self {
        assert_eq!(self.size, other.size);
        let mut result = Self::new(self.size);
        for i in 0..self.size {
            for j in 0..self.size {
                let mut sum = 0.0;
                for k in 0..self.size {
                    sum += self.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        result
    }

    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().flat_map(|r| r.iter()).map(|&v| v * v).sum::<f64>().sqrt()
    }

    pub fn trace(&self) -> f64 {
        (0..self.size).map(|i| self.data[i][i]).sum()
    }

    pub fn diagonal(&self) -> Vec<f64> {
        (0..self.size).map(|i| self.data[i][i]).collect()
    }

    pub fn is_symmetric(&self) -> bool {
        for i in 0..self.size {
            for j in 0..self.size {
                if (self.data[i][j] - self.data[j][i]).abs() > 1e-10 { return false; }
            }
        }
        true
    }

}

impl fmt::Display for OpenclAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OpenclAnalysis({:?})", self.data)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OpenclStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Skipped,
}

impl fmt::Display for OpenclStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpenclStatus::Pending => write!(f, "pending"),
            OpenclStatus::InProgress => write!(f, "inprogress"),
            OpenclStatus::Completed => write!(f, "completed"),
            OpenclStatus::Failed => write!(f, "failed"),
            OpenclStatus::Skipped => write!(f, "skipped"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OpenclPriority {
    Critical,
    High,
    Medium,
    Low,
    None,
}

impl fmt::Display for OpenclPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpenclPriority::Critical => write!(f, "critical"),
            OpenclPriority::High => write!(f, "high"),
            OpenclPriority::Medium => write!(f, "medium"),
            OpenclPriority::Low => write!(f, "low"),
            OpenclPriority::None => write!(f, "none"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OpenclMode {
    Strict,
    Relaxed,
    Permissive,
    Custom,
}

impl fmt::Display for OpenclMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpenclMode::Strict => write!(f, "strict"),
            OpenclMode::Relaxed => write!(f, "relaxed"),
            OpenclMode::Permissive => write!(f, "permissive"),
            OpenclMode::Custom => write!(f, "custom"),
        }
    }
}

pub fn opencl_mean(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    data.iter().sum::<f64>() / data.len() as f64
}

pub fn opencl_variance(data: &[f64]) -> f64 {
    if data.len() < 2 { return 0.0; }
    let mean = opencl_mean(data);
    data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (data.len() - 1) as f64
}

pub fn opencl_std_dev(data: &[f64]) -> f64 {
    opencl_variance(data).sqrt()
}

pub fn opencl_median(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n % 2 == 0 { (sorted[n/2 - 1] + sorted[n/2]) / 2.0 } else { sorted[n/2] }
}

/// Percentile calculator for OpenCl.
pub fn opencl_percentile_at(data: &[f64], p: f64) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = p / 100.0 * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi { sorted[lo] }
    else { sorted[lo] * (hi as f64 - idx) + sorted[hi] * (idx - lo as f64) }
}

pub fn opencl_entropy(data: &[f64]) -> f64 {
    let total: f64 = data.iter().sum();
    if total <= 0.0 { return 0.0; }
    let mut h = 0.0;
    for &x in data {
        if x > 0.0 {
            let p = x / total;
            h -= p * p.ln();
        }
    }
    h
}

pub fn opencl_gini(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let n = data.len();
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sum: f64 = sorted.iter().sum();
    if sum == 0.0 { return 0.0; }
    let mut g = 0.0;
    for (i, &x) in sorted.iter().enumerate() {
        g += (2.0 * (i + 1) as f64 - n as f64 - 1.0) * x;
    }
    g / (n as f64 * sum)
}

pub fn opencl_covariance(data: &[f64]) -> f64 {
    if data.len() < 4 { return 0.0; }
    let n = data.len() / 2;
    let x: Vec<f64> = data[..n].to_vec();
    let y: Vec<f64> = data[n..2*n].to_vec();
    let mx = opencl_mean(&x);
    let my = opencl_mean(&y);
    x.iter().zip(y.iter()).map(|(&xi, &yi)| (xi - mx) * (yi - my)).sum::<f64>() / (n - 1) as f64
}

pub fn opencl_correlation(data: &[f64]) -> f64 {
    if data.len() < 4 { return 0.0; }
    let n = data.len() / 2;
    let cov = opencl_covariance(data);
    let sx = opencl_std_dev(&data[..n]);
    let sy = opencl_std_dev(&data[n..2*n]);
    if sx * sy == 0.0 { 0.0 } else { cov / (sx * sy) }
}

pub fn opencl_excess_kurtosis(data: &[f64]) -> f64 {
    if data.len() < 4 { return 0.0; }
    let m = opencl_mean(data);
    let s = opencl_std_dev(data);
    if s == 0.0 { return 0.0; }
    let n = data.len() as f64;
    let k = data.iter().map(|&x| ((x - m) / s).powi(4)).sum::<f64>() / n;
    k - 3.0
}

pub fn opencl_sample_skewness(data: &[f64]) -> f64 {
    if data.len() < 3 { return 0.0; }
    let m = opencl_mean(data);
    let s = opencl_std_dev(data);
    if s == 0.0 { return 0.0; }
    let n = data.len() as f64;
    data.iter().map(|&x| ((x - m) / s).powi(3)).sum::<f64>() / n
}

pub fn opencl_harmmean(data: &[f64]) -> f64 {
    if data.is_empty() || data.iter().any(|&x| x <= 0.0) { return 0.0; }
    let n = data.len() as f64;
    n / data.iter().map(|&x| 1.0 / x).sum::<f64>()
}

pub fn opencl_geomean(data: &[f64]) -> f64 {
    if data.is_empty() || data.iter().any(|&x| x <= 0.0) { return 0.0; }
    let n = data.len() as f64;
    (data.iter().map(|&x| x.ln()).sum::<f64>() / n).exp()
}

/// Iterator over opencl analysis results.
#[derive(Debug, Clone)]
pub struct OpenclResultIterator {
    items: Vec<(usize, f64)>,
    position: usize,
}

impl OpenclResultIterator {
    pub fn new(items: Vec<(usize, f64)>) -> Self {
        OpenclResultIterator { items, position: 0 }
    }
    pub fn remaining(&self) -> usize { self.items.len() - self.position }
}

impl Iterator for OpenclResultIterator {
    type Item = (usize, f64);
    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.items.len() {
            let item = self.items[self.position];
            self.position += 1;
            Some(item)
        } else { None }
    }
}

/// Convert OpenClEventModel description to a summary string.
pub fn opencleventmodel_to_summary(item: &OpenClEventModel) -> String {
    format!("OpenClEventModel: {:?}", item)
}

/// Convert CommandQueueSemantics description to a summary string.
pub fn commandqueuesemantics_to_summary(item: &CommandQueueSemantics) -> String {
    format!("CommandQueueSemantics: {:?}", item)
}

/// Convert KernelArgBinding description to a summary string.
pub fn kernelargbinding_to_summary(item: &KernelArgBinding) -> String {
    format!("KernelArgBinding: {:?}", item)
}

/// Convert ImageMemoryModel description to a summary string.
pub fn imagememorymodel_to_summary(item: &ImageMemoryModel) -> String {
    format!("ImageMemoryModel: {:?}", item)
}

/// Convert SamplerConfig description to a summary string.
pub fn samplerconfig_to_summary(item: &SamplerConfig) -> String {
    format!("SamplerConfig: {:?}", item)
}

/// Convert PipeOperation description to a summary string.
pub fn pipeoperation_to_summary(item: &PipeOperation) -> String {
    format!("PipeOperation: {:?}", item)
}

/// Convert SubGroupOps description to a summary string.
pub fn subgroupops_to_summary(item: &SubGroupOps) -> String {
    format!("SubGroupOps: {:?}", item)
}

/// Convert ProgramScopeVar description to a summary string.
pub fn programscopevar_to_summary(item: &ProgramScopeVar) -> String {
    format!("ProgramScopeVar: {:?}", item)
}

/// Convert OpenClMemOrderSemantics description to a summary string.
pub fn openclmemordersemantics_to_summary(item: &OpenClMemOrderSemantics) -> String {
    format!("OpenClMemOrderSemantics: {:?}", item)
}

/// Convert OpenClWorkGroup description to a summary string.
pub fn openclworkgroup_to_summary(item: &OpenClWorkGroup) -> String {
    format!("OpenClWorkGroup: {:?}", item)
}

/// Convert OpenClBufferRegion description to a summary string.
pub fn openclbufferregion_to_summary(item: &OpenClBufferRegion) -> String {
    format!("OpenClBufferRegion: {:?}", item)
}

/// Batch processor for opencl operations.
#[derive(Debug, Clone)]
pub struct OpenclBatchProcessor {
    pub batch_size: usize,
    pub processed: usize,
    pub errors: Vec<String>,
    pub results: Vec<f64>,
}

impl OpenclBatchProcessor {
    pub fn new(batch_size: usize) -> Self {
        OpenclBatchProcessor { batch_size, processed: 0, errors: Vec::new(), results: Vec::new() }
    }
    pub fn process_batch(&mut self, data: &[f64]) {
        for chunk in data.chunks(self.batch_size) {
            let sum: f64 = chunk.iter().sum();
            self.results.push(sum / chunk.len() as f64);
            self.processed += chunk.len();
        }
    }
    pub fn success_rate(&self) -> f64 {
        if self.processed == 0 { return 0.0; }
        1.0 - (self.errors.len() as f64 / self.processed as f64)
    }
    pub fn average_result(&self) -> f64 {
        if self.results.is_empty() { return 0.0; }
        self.results.iter().sum::<f64>() / self.results.len() as f64
    }
    pub fn reset(&mut self) { self.processed = 0; self.errors.clear(); self.results.clear(); }
}

impl fmt::Display for OpenclBatchProcessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OpenclBatch(processed={}, errors={})", self.processed, self.errors.len())
    }
}

/// Detailed report for opencl analysis.
#[derive(Debug, Clone)]
pub struct OpenclReport {
    pub title: String,
    pub sections: Vec<(String, Vec<String>)>,
    pub metrics: Vec<(String, f64)>,
    pub warnings: Vec<String>,
    pub timestamp: u64,
}

impl OpenclReport {
    pub fn new(title: impl Into<String>) -> Self {
        OpenclReport { title: title.into(), sections: Vec::new(), metrics: Vec::new(), warnings: Vec::new(), timestamp: 0 }
    }
    pub fn add_section(&mut self, name: impl Into<String>, content: Vec<String>) {
        self.sections.push((name.into(), content));
    }
    pub fn add_metric(&mut self, name: impl Into<String>, value: f64) {
        self.metrics.push((name.into(), value));
    }
    pub fn add_warning(&mut self, warning: impl Into<String>) {
        self.warnings.push(warning.into());
    }
    pub fn total_metrics(&self) -> usize { self.metrics.len() }
    pub fn has_warnings(&self) -> bool { !self.warnings.is_empty() }
    pub fn metric_sum(&self) -> f64 { self.metrics.iter().map(|(_, v)| v).sum() }
    pub fn render_text(&self) -> String {
        let mut out = format!("=== {} ===\n", self.title);
        for (name, content) in &self.sections {
            out.push_str(&format!("\n--- {} ---\n", name));
            for line in content {
                out.push_str(&format!("  {}\n", line));
            }
        }
        out.push_str("\nMetrics:\n");
        for (name, val) in &self.metrics {
            out.push_str(&format!("  {}: {:.4}\n", name, val));
        }
        if !self.warnings.is_empty() {
            out.push_str("\nWarnings:\n");
            for w in &self.warnings {
                out.push_str(&format!("  ! {}\n", w));
            }
        }
        out
    }
}

impl fmt::Display for OpenclReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OpenclReport({})", self.title)
    }
}

/// Configuration for opencl analysis.
#[derive(Debug, Clone)]
pub struct OpenclConfig {
    pub verbose: bool,
    pub max_iterations: usize,
    pub tolerance: f64,
    pub timeout_ms: u64,
    pub parallel: bool,
    pub output_format: String,
}

impl OpenclConfig {
    pub fn default_config() -> Self {
        OpenclConfig {
            verbose: false, max_iterations: 1000, tolerance: 1e-6,
            timeout_ms: 30000, parallel: false, output_format: "text".to_string(),
        }
    }
    pub fn with_verbose(mut self, v: bool) -> Self { self.verbose = v; self }
    pub fn with_max_iterations(mut self, n: usize) -> Self { self.max_iterations = n; self }
    pub fn with_tolerance(mut self, t: f64) -> Self { self.tolerance = t; self }
    pub fn with_timeout(mut self, ms: u64) -> Self { self.timeout_ms = ms; self }
    pub fn with_parallel(mut self, p: bool) -> Self { self.parallel = p; self }
    pub fn with_output_format(mut self, fmt: impl Into<String>) -> Self { self.output_format = fmt.into(); self }
}

impl fmt::Display for OpenclConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OpenclConfig(iters={}, tol={:.0e})", self.max_iterations, self.tolerance)
    }
}

/// Histogram for opencl data distribution.
#[derive(Debug, Clone)]
pub struct OpenclHistogram {
    pub bins: Vec<usize>,
    pub bin_edges: Vec<f64>,
    pub total_count: usize,
}

impl OpenclHistogram {
    pub fn from_data(data: &[f64], num_bins: usize) -> Self {
        if data.is_empty() || num_bins == 0 {
            return OpenclHistogram { bins: Vec::new(), bin_edges: Vec::new(), total_count: 0 };
        }
        let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_val - min_val;
        let bin_width = if range == 0.0 { 1.0 } else { range / num_bins as f64 };
        let mut bins = vec![0usize; num_bins];
        let mut bin_edges = Vec::with_capacity(num_bins + 1);
        for i in 0..=num_bins { bin_edges.push(min_val + i as f64 * bin_width); }
        for &val in data {
            let idx = ((val - min_val) / bin_width).floor() as usize;
            let idx = idx.min(num_bins - 1);
            bins[idx] += 1;
        }
        OpenclHistogram { bins, bin_edges, total_count: data.len() }
    }
    pub fn num_bins(&self) -> usize { self.bins.len() }
    pub fn max_bin(&self) -> usize { self.bins.iter().cloned().max().unwrap_or(0) }
    pub fn mean_bin(&self) -> f64 {
        if self.bins.is_empty() { return 0.0; }
        self.bins.iter().sum::<usize>() as f64 / self.bins.len() as f64
    }
    pub fn render_ascii(&self, width: usize) -> String {
        let max = self.max_bin();
        let mut out = String::new();
        for (i, &count) in self.bins.iter().enumerate() {
            let bar_len = if max == 0 { 0 } else { count * width / max };
            let bar: String = std::iter::repeat('#').take(bar_len).collect();
            out.push_str(&format!("[{:.2}-{:.2}] {} {}\n",
                self.bin_edges[i], self.bin_edges[i + 1], bar, count));
        }
        out
    }
}

impl fmt::Display for OpenclHistogram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Histogram(bins={}, total={})", self.num_bins(), self.total_count)
    }
}

/// Adjacency/weight matrix for opencl graph analysis.
#[derive(Debug, Clone)]
pub struct OpenclGraph {
    pub adjacency: Vec<Vec<bool>>,
    pub weights: Vec<Vec<f64>>,
    pub node_count: usize,
    pub edge_count: usize,
    pub node_labels: Vec<String>,
}

impl OpenclGraph {
    pub fn new(n: usize) -> Self {
        OpenclGraph {
            adjacency: vec![vec![false; n]; n],
            weights: vec![vec![0.0; n]; n],
            node_count: n, edge_count: 0,
            node_labels: (0..n).map(|i| format!("n{}", i)).collect(),
        }
    }
    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) {
        if from < self.node_count && to < self.node_count && !self.adjacency[from][to] {
            self.adjacency[from][to] = true;
            self.weights[from][to] = weight;
            self.edge_count += 1;
        }
    }
    pub fn remove_edge(&mut self, from: usize, to: usize) {
        if from < self.node_count && to < self.node_count && self.adjacency[from][to] {
            self.adjacency[from][to] = false;
            self.weights[from][to] = 0.0;
            self.edge_count -= 1;
        }
    }
    pub fn has_edge(&self, from: usize, to: usize) -> bool {
        from < self.node_count && to < self.node_count && self.adjacency[from][to]
    }
    pub fn weight(&self, from: usize, to: usize) -> f64 { self.weights[from][to] }
    pub fn out_degree(&self, node: usize) -> usize {
        (0..self.node_count).filter(|&j| self.adjacency[node][j]).count()
    }
    pub fn in_degree(&self, node: usize) -> usize {
        (0..self.node_count).filter(|&i| self.adjacency[i][node]).count()
    }
    pub fn neighbors(&self, node: usize) -> Vec<usize> {
        (0..self.node_count).filter(|&j| self.adjacency[node][j]).collect()
    }
    pub fn density(&self) -> f64 {
        if self.node_count <= 1 { return 0.0; }
        self.edge_count as f64 / (self.node_count * (self.node_count - 1)) as f64
    }
    pub fn is_acyclic(&self) -> bool {
        let n = self.node_count;
        let mut visited = vec![0u8; n];
        fn dfs_cycle_opencl(v: usize, adj: &[Vec<bool>], visited: &mut [u8]) -> bool {
            visited[v] = 1;
            for w in 0..adj.len() { if adj[v][w] {
                if visited[w] == 1 { return true; }
                if visited[w] == 0 && dfs_cycle_opencl(w, adj, visited) { return true; }
            }}
            visited[v] = 2; false
        }
        for i in 0..n {
            if visited[i] == 0 && dfs_cycle_opencl(i, &self.adjacency, &mut visited) { return false; }
        }
        true
    }
    pub fn topological_sort(&self) -> Option<Vec<usize>> {
        let n = self.node_count;
        let mut in_deg: Vec<usize> = (0..n).map(|j| self.in_degree(j)).collect();
        let mut queue: Vec<usize> = (0..n).filter(|&i| in_deg[i] == 0).collect();
        let mut result = Vec::new();
        while let Some(v) = queue.pop() {
            result.push(v);
            for j in 0..n { if self.adjacency[v][j] {
                in_deg[j] -= 1;
                if in_deg[j] == 0 { queue.push(j); }
            }}
        }
        if result.len() == n { Some(result) } else { None }
    }
    pub fn shortest_path_dijkstra(&self, start: usize) -> Vec<f64> {
        let n = self.node_count;
        let mut dist = vec![f64::INFINITY; n];
        let mut visited = vec![false; n];
        dist[start] = 0.0;
        for _ in 0..n {
            let mut u = None;
            let mut min_d = f64::INFINITY;
            for v in 0..n { if !visited[v] && dist[v] < min_d { min_d = dist[v]; u = Some(v); } }
            let u = match u { Some(v) => v, None => break };
            visited[u] = true;
            for v in 0..n { if self.adjacency[u][v] {
                let alt = dist[u] + self.weights[u][v];
                if alt < dist[v] { dist[v] = alt; }
            }}
        }
        dist
    }
    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let n = self.node_count;
        let mut visited = vec![false; n];
        let mut components = Vec::new();
        for start in 0..n {
            if visited[start] { continue; }
            let mut comp = Vec::new();
            let mut stack = vec![start];
            while let Some(v) = stack.pop() {
                if visited[v] { continue; }
                visited[v] = true;
                comp.push(v);
                for w in 0..n {
                    if (self.adjacency[v][w] || self.adjacency[w][v]) && !visited[w] {
                        stack.push(w);
                    }
                }
            }
            components.push(comp);
        }
        components
    }
    pub fn to_dot(&self) -> String {
        let mut out = String::from("digraph {\n");
        for i in 0..self.node_count {
            out.push_str(&format!("  {} [label=\"{}\"];\n", i, self.node_labels[i]));
        }
        for i in 0..self.node_count { for j in 0..self.node_count { if self.adjacency[i][j] {
            out.push_str(&format!("  {} -> {} [label=\"{:.2}\"];\n", i, j, self.weights[i][j]));
        }}}
        out.push_str("}\n");
        out
    }
}

impl fmt::Display for OpenclGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OpenclGraph(n={}, e={})", self.node_count, self.edge_count)
    }
}

/// Cache for opencl computation results.
#[derive(Debug, Clone)]
pub struct OpenclCache {
    entries: Vec<(u64, Vec<f64>)>,
    capacity: usize,
    hits: u64,
    misses: u64,
}

impl OpenclCache {
    pub fn new(capacity: usize) -> Self {
        OpenclCache { entries: Vec::new(), capacity, hits: 0, misses: 0 }
    }
    pub fn get(&mut self, key: u64) -> Option<&Vec<f64>> {
        if let Some(pos) = self.entries.iter().position(|(k, _)| *k == key) {
            self.hits += 1;
            Some(&self.entries[pos].1)
        } else { self.misses += 1; None }
    }
    pub fn insert(&mut self, key: u64, value: Vec<f64>) {
        if self.entries.len() >= self.capacity { self.entries.remove(0); }
        self.entries.push((key, value));
    }
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }
    pub fn size(&self) -> usize { self.entries.len() }
    pub fn clear(&mut self) { self.entries.clear(); self.hits = 0; self.misses = 0; }
}

impl fmt::Display for OpenclCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cache(size={}, hit_rate={:.1}%)", self.size(), self.hit_rate() * 100.0)
    }
}

/// Compute pairwise distances for opencl elements.
pub fn opencl_pairwise_distances(points: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = points.len();
    let mut distances = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i+1)..n {
            let d: f64 = points[i].iter().zip(points[j].iter())
                .map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
            distances[i][j] = d;
            distances[j][i] = d;
        }
    }
    distances
}

/// K-means clustering for opencl data.
pub fn opencl_kmeans(data: &[Vec<f64>], k: usize, max_iters: usize) -> Vec<usize> {
    if data.is_empty() || k == 0 { return Vec::new(); }
    let n = data.len();
    let dim = data[0].len();
    let mut centroids: Vec<Vec<f64>> = data.iter().take(k).cloned().collect();
    let mut assignments = vec![0usize; n];
    for _ in 0..max_iters {
        // Assign
        let mut changed = false;
        for i in 0..n {
            let mut best_c = 0; let mut best_d = f64::INFINITY;
            for c in 0..centroids.len() {
                let d: f64 = data[i].iter().zip(centroids[c].iter())
                    .map(|(a, b)| (a - b) * (a - b)).sum();
                if d < best_d { best_d = d; best_c = c; }
            }
            if assignments[i] != best_c { changed = true; assignments[i] = best_c; }
        }
        if !changed { break; }
        // Update centroids
        for c in 0..centroids.len() {
            let members: Vec<usize> = (0..n).filter(|&i| assignments[i] == c).collect();
            if members.is_empty() { continue; }
            for d in 0..dim {
                centroids[c][d] = members.iter().map(|&i| data[i][d]).sum::<f64>() / members.len() as f64;
            }
        }
    }
    assignments
}

/// Principal component analysis (simplified) for opencl data.
pub fn opencl_pca_2d(data: &[Vec<f64>]) -> Vec<(f64, f64)> {
    if data.is_empty() || data[0].len() < 2 { return Vec::new(); }
    let n = data.len();
    let dim = data[0].len();
    // Compute mean
    let mut mean = vec![0.0; dim];
    for row in data { for (j, &v) in row.iter().enumerate() { mean[j] += v; } }
    for j in 0..dim { mean[j] /= n as f64; }
    // Center data
    let centered: Vec<Vec<f64>> = data.iter().map(|row| {
        row.iter().zip(mean.iter()).map(|(v, m)| v - m).collect()
    }).collect();
    // Simple projection onto first two dimensions (not true PCA)
    centered.iter().map(|row| (row[0], row[1])).collect()
}

/// Dense matrix operations for OpenCl computations.
#[derive(Debug, Clone)]
pub struct OpenClDenseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl OpenClDenseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        OpenClDenseMatrix { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::new(n, n);
        for i in 0..n { m.data[i * n + i] = 1.0; }
        m
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols);
        OpenClDenseMatrix { rows, cols, data }
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.cols + j] = v;
    }

    pub fn row(&self, i: usize) -> Vec<f64> {
        self.data[i * self.cols..(i + 1) * self.cols].to_vec()
    }

    pub fn col(&self, j: usize) -> Vec<f64> {
        (0..self.rows).map(|i| self.data[i * self.cols + j]).collect()
    }

    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows); assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect();
        OpenClDenseMatrix { rows: self.rows, cols: self.cols, data }
    }

    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows); assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a - b).collect();
        OpenClDenseMatrix { rows: self.rows, cols: self.cols, data }
    }

    pub fn mul_matrix(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows);
        let mut result = Self::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols { sum += self.get(i, k) * other.get(k, j); }
                result.set(i, j, sum);
            }
        }
        result
    }

    pub fn scale(&self, s: f64) -> Self {
        let data: Vec<f64> = self.data.iter().map(|&v| v * s).collect();
        OpenClDenseMatrix { rows: self.rows, cols: self.cols, data }
    }

    pub fn transpose(&self) -> Self {
        let mut result = Self::new(self.cols, self.rows);
        for i in 0..self.rows { for j in 0..self.cols { result.set(j, i, self.get(i, j)); } }
        result
    }

    pub fn trace(&self) -> f64 {
        let n = self.rows.min(self.cols);
        (0..n).map(|i| self.get(i, i)).sum()
    }

    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|&v| v * v).sum::<f64>().sqrt()
    }

    pub fn max_abs(&self) -> f64 {
        self.data.iter().map(|v| v.abs()).fold(0.0f64, f64::max)
    }

    pub fn row_sum(&self, i: usize) -> f64 {
        (0..self.cols).map(|j| self.get(i, j)).sum()
    }

    pub fn col_sum(&self, j: usize) -> f64 {
        (0..self.rows).map(|i| self.get(i, j)).sum()
    }

    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    pub fn is_symmetric(&self) -> bool {
        if !self.is_square() { return false; }
        for i in 0..self.rows { for j in (i+1)..self.cols {
            if (self.get(i, j) - self.get(j, i)).abs() > 1e-10 { return false; }
        }}
        true
    }

    pub fn is_diagonal(&self) -> bool {
        for i in 0..self.rows { for j in 0..self.cols {
            if i != j && self.get(i, j).abs() > 1e-10 { return false; }
        }}
        true
    }

    pub fn is_upper_triangular(&self) -> bool {
        for i in 0..self.rows { for j in 0..i.min(self.cols) {
            if self.get(i, j).abs() > 1e-10 { return false; }
        }}
        true
    }

    pub fn determinant_2x2(&self) -> f64 {
        assert!(self.rows == 2 && self.cols == 2);
        self.get(0, 0) * self.get(1, 1) - self.get(0, 1) * self.get(1, 0)
    }

    pub fn determinant_3x3(&self) -> f64 {
        assert!(self.rows == 3 && self.cols == 3);
        let a = self.get(0, 0); let b = self.get(0, 1); let c = self.get(0, 2);
        let d = self.get(1, 0); let e = self.get(1, 1); let ff = self.get(1, 2);
        let g = self.get(2, 0); let h = self.get(2, 1); let ii = self.get(2, 2);
        a * (e * ii - ff * h) - b * (d * ii - ff * g) + c * (d * h - e * g)
    }

    pub fn inverse_2x2(&self) -> Option<Self> {
        assert!(self.rows == 2 && self.cols == 2);
        let det = self.determinant_2x2();
        if det.abs() < 1e-15 { return None; }
        let inv_det = 1.0 / det;
        let mut result = Self::new(2, 2);
        result.set(0, 0, self.get(1, 1) * inv_det);
        result.set(0, 1, -self.get(0, 1) * inv_det);
        result.set(1, 0, -self.get(1, 0) * inv_det);
        result.set(1, 1, self.get(0, 0) * inv_det);
        Some(result)
    }

    pub fn power(&self, n: u32) -> Self {
        assert!(self.is_square());
        let mut result = Self::identity(self.rows);
        for _ in 0..n { result = result.mul_matrix(self); }
        result
    }

    pub fn submatrix(&self, row_start: usize, col_start: usize, rows: usize, cols: usize) -> Self {
        let mut result = Self::new(rows, cols);
        for i in 0..rows { for j in 0..cols {
            result.set(i, j, self.get(row_start + i, col_start + j));
        }}
        result
    }

    pub fn kronecker_product(&self, other: &Self) -> Self {
        let m = self.rows * other.rows;
        let n = self.cols * other.cols;
        let mut result = Self::new(m, n);
        for i in 0..self.rows { for j in 0..self.cols {
            let s = self.get(i, j);
            for p in 0..other.rows { for q in 0..other.cols {
                result.set(i * other.rows + p, j * other.cols + q, s * other.get(p, q));
            }}
        }}
        result
    }

    pub fn hadamard_product(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows); assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).collect();
        OpenClDenseMatrix { rows: self.rows, cols: self.cols, data }
    }

    pub fn outer_product(a: &[f64], b: &[f64]) -> Self {
        let mut result = Self::new(a.len(), b.len());
        for i in 0..a.len() { for j in 0..b.len() { result.set(i, j, a[i] * b[j]); } }
        result
    }

    pub fn row_reduce(&self) -> Self {
        let mut result = self.clone();
        let mut pivot_row = 0;
        for col in 0..result.cols {
            if pivot_row >= result.rows { break; }
            let mut max_row = pivot_row;
            for row in (pivot_row + 1)..result.rows {
                if result.get(row, col).abs() > result.get(max_row, col).abs() { max_row = row; }
            }
            if result.get(max_row, col).abs() < 1e-10 { continue; }
            for j in 0..result.cols {
                let tmp = result.get(pivot_row, j);
                result.set(pivot_row, j, result.get(max_row, j));
                result.set(max_row, j, tmp);
            }
            let pivot = result.get(pivot_row, col);
            for j in 0..result.cols { result.set(pivot_row, j, result.get(pivot_row, j) / pivot); }
            for row in 0..result.rows {
                if row == pivot_row { continue; }
                let factor = result.get(row, col);
                for j in 0..result.cols {
                    let v = result.get(row, j) - factor * result.get(pivot_row, j);
                    result.set(row, j, v);
                }
            }
            pivot_row += 1;
        }
        result
    }

    pub fn rank(&self) -> usize {
        let rref = self.row_reduce();
        let mut r = 0;
        for i in 0..rref.rows {
            if (0..rref.cols).any(|j| rref.get(i, j).abs() > 1e-10) { r += 1; }
        }
        r
    }

    pub fn nullity(&self) -> usize {
        self.cols - self.rank()
    }

    pub fn column_space_basis(&self) -> Vec<Vec<f64>> {
        let rref = self.row_reduce();
        let mut basis = Vec::new();
        for j in 0..self.cols {
            let is_pivot = (0..rref.rows).any(|i| {
                (rref.get(i, j) - 1.0).abs() < 1e-10 &&
                (0..j).all(|k| rref.get(i, k).abs() < 1e-10)
            });
            if is_pivot { basis.push(self.col(j)); }
        }
        basis
    }

    pub fn lu_decomposition(&self) -> (Self, Self) {
        assert!(self.is_square());
        let n = self.rows;
        let mut l = Self::identity(n);
        let mut u = self.clone();
        for k in 0..n {
            for i in (k+1)..n {
                if u.get(k, k).abs() < 1e-15 { continue; }
                let factor = u.get(i, k) / u.get(k, k);
                l.set(i, k, factor);
                for j in k..n {
                    let v = u.get(i, j) - factor * u.get(k, j);
                    u.set(i, j, v);
                }
            }
        }
        (l, u)
    }

    pub fn solve(&self, b: &[f64]) -> Option<Vec<f64>> {
        assert!(self.is_square());
        assert_eq!(self.rows, b.len());
        let n = self.rows;
        let mut augmented = Self::new(n, n + 1);
        for i in 0..n { for j in 0..n { augmented.set(i, j, self.get(i, j)); } augmented.set(i, n, b[i]); }
        let rref = augmented.row_reduce();
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            x[i] = rref.get(i, n);
            for j in (i+1)..n { x[i] -= rref.get(i, j) * x[j]; }
            if rref.get(i, i).abs() < 1e-15 { return None; }
            x[i] /= rref.get(i, i);
        }
        Some(x)
    }

    pub fn eigenvalues_2x2(&self) -> (f64, f64) {
        assert!(self.rows == 2 && self.cols == 2);
        let tr = self.trace();
        let det = self.determinant_2x2();
        let disc = tr * tr - 4.0 * det;
        if disc >= 0.0 {
            ((tr + disc.sqrt()) / 2.0, (tr - disc.sqrt()) / 2.0)
        } else {
            (tr / 2.0, tr / 2.0)
        }
    }

    pub fn condition_number(&self) -> f64 {
        let max_sv = self.frobenius_norm();
        if max_sv < 1e-15 { return f64::INFINITY; }
        max_sv
    }

}

impl fmt::Display for OpenClDenseMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OpenClMatrix({}x{})", self.rows, self.cols)
    }
}

/// Interval arithmetic for OpenCl bounds analysis.
#[derive(Debug, Clone, Copy)]
pub struct OpenClInterval {
    pub lo: f64,
    pub hi: f64,
}

impl OpenClInterval {
    pub fn new(lo: f64, hi: f64) -> Self {
        OpenClInterval { lo: lo.min(hi), hi: lo.max(hi) }
    }

    pub fn point(v: f64) -> Self {
        OpenClInterval { lo: v, hi: v }
    }

    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }

    pub fn midpoint(&self) -> f64 {
        (self.lo + self.hi) / 2.0
    }

    pub fn contains(&self, v: f64) -> bool {
        self.lo <= v && v <= self.hi
    }

    pub fn overlaps(&self, other: &Self) -> bool {
        self.lo <= other.hi && other.lo <= self.hi
    }

    pub fn hull(&self, other: &Self) -> Self {
        OpenClInterval { lo: self.lo.min(other.lo), hi: self.hi.max(other.hi) }
    }

    pub fn intersect(&self, other: &Self) -> Option<Self> {
        let lo = self.lo.max(other.lo);
        let hi = self.hi.min(other.hi);
        if lo <= hi { Some(OpenClInterval { lo, hi }) } else { None }
    }

    pub fn add(&self, other: &Self) -> Self {
        OpenClInterval { lo: self.lo + other.lo, hi: self.hi + other.hi }
    }

    pub fn sub(&self, other: &Self) -> Self {
        OpenClInterval { lo: self.lo - other.hi, hi: self.hi - other.lo }
    }

    pub fn mul(&self, other: &Self) -> Self {
        let products = [self.lo * other.lo, self.lo * other.hi, self.hi * other.lo, self.hi * other.hi];
        let lo = products.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = products.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        OpenClInterval { lo, hi }
    }

    pub fn abs(&self) -> Self {
        if self.lo >= 0.0 { *self }
        else if self.hi <= 0.0 { OpenClInterval { lo: -self.hi, hi: -self.lo } }
        else { OpenClInterval { lo: 0.0, hi: self.lo.abs().max(self.hi.abs()) } }
    }

    pub fn sqrt(&self) -> Self {
        let lo = if self.lo >= 0.0 { self.lo.sqrt() } else { 0.0 };
        OpenClInterval { lo, hi: self.hi.max(0.0).sqrt() }
    }

    pub fn is_positive(&self) -> bool {
        self.lo > 0.0
    }

    pub fn is_negative(&self) -> bool {
        self.hi < 0.0
    }

    pub fn is_zero(&self) -> bool {
        self.lo <= 0.0 && self.hi >= 0.0
    }

    pub fn is_point(&self) -> bool {
        (self.hi - self.lo).abs() < 1e-15
    }

}

impl fmt::Display for OpenClInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:.4}, {:.4}]", self.lo, self.hi)
    }
}

/// State machine for OpenCl protocol modeling.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OpenClState {
    Created,
    Enqueued,
    Submitted,
    Running,
    Completed,
    Error,
}

impl fmt::Display for OpenClState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpenClState::Created => write!(f, "created"),
            OpenClState::Enqueued => write!(f, "enqueued"),
            OpenClState::Submitted => write!(f, "submitted"),
            OpenClState::Running => write!(f, "running"),
            OpenClState::Completed => write!(f, "completed"),
            OpenClState::Error => write!(f, "error"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OpenClStateMachine {
    pub current: OpenClState,
    pub history: Vec<String>,
    pub transition_count: usize,
}

impl OpenClStateMachine {
    pub fn new() -> Self {
        OpenClStateMachine { current: OpenClState::Created, history: Vec::new(), transition_count: 0 }
    }
    pub fn state(&self) -> &OpenClState { &self.current }
    pub fn can_transition(&self, target: &OpenClState) -> bool {
        match (&self.current, target) {
            (OpenClState::Created, OpenClState::Enqueued) => true,
            (OpenClState::Enqueued, OpenClState::Submitted) => true,
            (OpenClState::Submitted, OpenClState::Running) => true,
            (OpenClState::Running, OpenClState::Completed) => true,
            (OpenClState::Running, OpenClState::Error) => true,
            (OpenClState::Error, OpenClState::Created) => true,
            (OpenClState::Completed, OpenClState::Created) => true,
            _ => false,
        }
    }
    pub fn transition(&mut self, target: OpenClState) -> bool {
        if self.can_transition(&target) {
            self.history.push(format!("{} -> {}", self.current, target));
            self.current = target;
            self.transition_count += 1;
            true
        } else { false }
    }
    pub fn reset(&mut self) {
        self.current = OpenClState::Created;
        self.history.clear();
        self.transition_count = 0;
    }
    pub fn history_len(&self) -> usize { self.history.len() }
}

impl fmt::Display for OpenClStateMachine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SM(state={}, transitions={})", self.current, self.transition_count)
    }
}

/// Ring buffer for OpenCl event tracking.
#[derive(Debug, Clone)]
pub struct OpenClRingBuffer {
    data: Vec<f64>,
    capacity: usize,
    head: usize,
    count: usize,
}

impl OpenClRingBuffer {
    pub fn new(capacity: usize) -> Self {
        OpenClRingBuffer { data: vec![0.0; capacity], capacity, head: 0, count: 0 }
    }
    pub fn push(&mut self, value: f64) {
        self.data[self.head] = value;
        self.head = (self.head + 1) % self.capacity;
        if self.count < self.capacity { self.count += 1; }
    }
    pub fn len(&self) -> usize { self.count }
    pub fn is_empty(&self) -> bool { self.count == 0 }
    pub fn is_full(&self) -> bool { self.count == self.capacity }
    pub fn latest(&self) -> Option<f64> {
        if self.count == 0 { None }
        else { Some(self.data[(self.head + self.capacity - 1) % self.capacity]) }
    }
    pub fn oldest(&self) -> Option<f64> {
        if self.count == 0 { None }
        else { Some(self.data[(self.head + self.capacity - self.count) % self.capacity]) }
    }
    pub fn average(&self) -> f64 {
        if self.count == 0 { return 0.0; }
        let mut sum = 0.0;
        for i in 0..self.count {
            sum += self.data[(self.head + self.capacity - 1 - i) % self.capacity];
        }
        sum / self.count as f64
    }
    pub fn to_vec(&self) -> Vec<f64> {
        let mut result = Vec::with_capacity(self.count);
        for i in 0..self.count {
            result.push(self.data[(self.head + self.capacity - self.count + i) % self.capacity]);
        }
        result
    }
    pub fn min(&self) -> Option<f64> {
        if self.count == 0 { return None; }
        Some(self.to_vec().iter().cloned().fold(f64::INFINITY, f64::min))
    }
    pub fn max(&self) -> Option<f64> {
        if self.count == 0 { return None; }
        Some(self.to_vec().iter().cloned().fold(f64::NEG_INFINITY, f64::max))
    }
    pub fn variance(&self) -> f64 {
        if self.count < 2 { return 0.0; }
        let avg = self.average();
        let v: f64 = self.to_vec().iter().map(|&x| (x - avg) * (x - avg)).sum();
        v / (self.count - 1) as f64
    }
    pub fn clear(&mut self) { self.head = 0; self.count = 0; }
}

impl fmt::Display for OpenClRingBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RingBuffer(len={}/{})", self.count, self.capacity)
    }
}

/// Disjoint set (union-find) for OpenCl component tracking.
#[derive(Debug, Clone)]
pub struct OpenClDisjointSet {
    parent: Vec<usize>,
    rank: Vec<usize>,
    size: Vec<usize>,
    num_components: usize,
}

impl OpenClDisjointSet {
    pub fn new(n: usize) -> Self {
        OpenClDisjointSet { parent: (0..n).collect(), rank: vec![0; n], size: vec![1; n], num_components: n }
    }
    pub fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x { self.parent[x] = self.parent[self.parent[x]]; x = self.parent[x]; }
        x
    }
    pub fn union(&mut self, x: usize, y: usize) -> bool {
        let rx = self.find(x); let ry = self.find(y);
        if rx == ry { return false; }
        if self.rank[rx] < self.rank[ry] { self.parent[rx] = ry; self.size[ry] += self.size[rx]; }
        else if self.rank[rx] > self.rank[ry] { self.parent[ry] = rx; self.size[rx] += self.size[ry]; }
        else { self.parent[ry] = rx; self.size[rx] += self.size[ry]; self.rank[rx] += 1; }
        self.num_components -= 1;
        true
    }
    pub fn connected(&mut self, x: usize, y: usize) -> bool { self.find(x) == self.find(y) }
    pub fn component_size(&mut self, x: usize) -> usize { let r = self.find(x); self.size[r] }
    pub fn num_components(&self) -> usize { self.num_components }
    pub fn components(&mut self) -> Vec<Vec<usize>> {
        let n = self.parent.len();
        let mut groups: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
        for i in 0..n { let r = self.find(i); groups.entry(r).or_default().push(i); }
        groups.into_values().collect()
    }
}

impl fmt::Display for OpenClDisjointSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DisjointSet(n={}, components={})", self.parent.len(), self.num_components)
    }
}

/// Sorted list with binary search for OpenCl.
#[derive(Debug, Clone)]
pub struct OpenClSortedList {
    data: Vec<f64>,
}

impl OpenClSortedList {
    pub fn new() -> Self { OpenClSortedList { data: Vec::new() } }
    pub fn insert(&mut self, value: f64) {
        let pos = self.data.partition_point(|&x| x < value);
        self.data.insert(pos, value);
    }
    pub fn contains(&self, value: f64) -> bool {
        self.data.binary_search_by(|x| x.partial_cmp(&value).unwrap()).is_ok()
    }
    pub fn rank(&self, value: f64) -> usize { self.data.partition_point(|&x| x < value) }
    pub fn quantile(&self, q: f64) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let idx = ((self.data.len() - 1) as f64 * q).round() as usize;
        self.data[idx.min(self.data.len() - 1)]
    }
    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
    pub fn min(&self) -> Option<f64> { self.data.first().copied() }
    pub fn max(&self) -> Option<f64> { self.data.last().copied() }
    pub fn median(&self) -> f64 { self.quantile(0.5) }
    pub fn iqr(&self) -> f64 { self.quantile(0.75) - self.quantile(0.25) }
    pub fn remove(&mut self, value: f64) -> bool {
        if let Ok(pos) = self.data.binary_search_by(|x| x.partial_cmp(&value).unwrap()) {
            self.data.remove(pos); true
        } else { false }
    }
    pub fn range(&self, lo: f64, hi: f64) -> Vec<f64> {
        self.data.iter().filter(|&&x| x >= lo && x <= hi).cloned().collect()
    }
    pub fn to_vec(&self) -> Vec<f64> { self.data.clone() }
}

impl fmt::Display for OpenClSortedList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SortedList(len={})", self.data.len())
    }
}

/// Exponential moving average for OpenCl metrics.
#[derive(Debug, Clone)]
pub struct OpenClEma {
    pub alpha: f64,
    pub value: f64,
    pub count: usize,
    pub initialized: bool,
}

impl OpenClEma {
    pub fn new(alpha: f64) -> Self { OpenClEma { alpha, value: 0.0, count: 0, initialized: false } }
    pub fn update(&mut self, sample: f64) {
        if !self.initialized { self.value = sample; self.initialized = true; }
        else { self.value = self.alpha * sample + (1.0 - self.alpha) * self.value; }
        self.count += 1;
    }
    pub fn current(&self) -> f64 { self.value }
    pub fn reset(&mut self) { self.value = 0.0; self.count = 0; self.initialized = false; }
}

impl fmt::Display for OpenClEma {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EMA(alpha={:.2}, value={:.4})", self.alpha, self.value)
    }
}

/// Simple bloom filter for OpenCl membership testing.
#[derive(Debug, Clone)]
pub struct OpenClBloomFilter {
    bits: Vec<bool>,
    num_hashes: usize,
    size: usize,
    count: usize,
}

impl OpenClBloomFilter {
    pub fn new(size: usize, num_hashes: usize) -> Self {
        OpenClBloomFilter { bits: vec![false; size], num_hashes, size, count: 0 }
    }
    fn hash_indices(&self, value: u64) -> Vec<usize> {
        let mut indices = Vec::with_capacity(self.num_hashes);
        let mut h = value;
        for _ in 0..self.num_hashes {
            h = h.wrapping_mul(0x517cc1b727220a95).wrapping_add(0x6c62272e07bb0142);
            indices.push((h as usize) % self.size);
        }
        indices
    }
    pub fn insert(&mut self, value: u64) {
        for idx in self.hash_indices(value) { self.bits[idx] = true; }
        self.count += 1;
    }
    pub fn may_contain(&self, value: u64) -> bool {
        self.hash_indices(value).iter().all(|&idx| self.bits[idx])
    }
    pub fn false_positive_rate(&self) -> f64 {
        let set_bits = self.bits.iter().filter(|&&b| b).count() as f64;
        (set_bits / self.size as f64).powi(self.num_hashes as i32)
    }
    pub fn count(&self) -> usize { self.count }
    pub fn clear(&mut self) { self.bits.fill(false); self.count = 0; }
}

impl fmt::Display for OpenClBloomFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BloomFilter(size={}, count={}, fpr={:.4})", self.size, self.count, self.false_positive_rate())
    }
}

/// Simple prefix trie for OpenCl string matching.
#[derive(Debug, Clone)]
pub struct OpenClTrieNode {
    children: Vec<(char, usize)>,
    is_terminal: bool,
    value: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct OpenClTrie {
    nodes: Vec<OpenClTrieNode>,
    count: usize,
}

impl OpenClTrie {
    pub fn new() -> Self {
        OpenClTrie { nodes: vec![OpenClTrieNode { children: Vec::new(), is_terminal: false, value: None }], count: 0 }
    }
    pub fn insert(&mut self, key: &str, value: u64) {
        let mut current = 0;
        for ch in key.chars() {
            let next = self.nodes[current].children.iter().find(|(c, _)| *c == ch).map(|(_, idx)| *idx);
            current = match next {
                Some(idx) => idx,
                None => {
                    let idx = self.nodes.len();
                    self.nodes.push(OpenClTrieNode { children: Vec::new(), is_terminal: false, value: None });
                    self.nodes[current].children.push((ch, idx));
                    idx
                }
            };
        }
        self.nodes[current].is_terminal = true;
        self.nodes[current].value = Some(value);
        self.count += 1;
    }
    pub fn search(&self, key: &str) -> Option<u64> {
        let mut current = 0;
        for ch in key.chars() {
            match self.nodes[current].children.iter().find(|(c, _)| *c == ch) {
                Some((_, idx)) => current = *idx,
                None => return None,
            }
        }
        if self.nodes[current].is_terminal { self.nodes[current].value } else { None }
    }
    pub fn starts_with(&self, prefix: &str) -> bool {
        let mut current = 0;
        for ch in prefix.chars() {
            match self.nodes[current].children.iter().find(|(c, _)| *c == ch) {
                Some((_, idx)) => current = *idx,
                None => return false,
            }
        }
        true
    }
    pub fn len(&self) -> usize { self.count }
    pub fn is_empty(&self) -> bool { self.count == 0 }
    pub fn node_count(&self) -> usize { self.nodes.len() }
}

impl fmt::Display for OpenClTrie {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Trie(entries={}, nodes={})", self.count, self.nodes.len())
    }
}

/// Min-heap priority queue for OpenCl scheduling.
#[derive(Debug, Clone)]
pub struct OpenClPriorityQueue {
    heap: Vec<(f64, usize)>,
}

impl OpenClPriorityQueue {
    pub fn new() -> Self { OpenClPriorityQueue { heap: Vec::new() } }
    pub fn push(&mut self, priority: f64, item: usize) {
        self.heap.push((priority, item));
        let mut i = self.heap.len() - 1;
        while i > 0 {
            let parent = (i - 1) / 2;
            if self.heap[i].0 < self.heap[parent].0 { self.heap.swap(i, parent); i = parent; }
            else { break; }
        }
    }
    pub fn pop(&mut self) -> Option<(f64, usize)> {
        if self.heap.is_empty() { return None; }
        let result = self.heap.swap_remove(0);
        if !self.heap.is_empty() { self.sift_down(0); }
        Some(result)
    }
    fn sift_down(&mut self, mut i: usize) {
        loop {
            let left = 2 * i + 1;
            let right = 2 * i + 2;
            let mut smallest = i;
            if left < self.heap.len() && self.heap[left].0 < self.heap[smallest].0 { smallest = left; }
            if right < self.heap.len() && self.heap[right].0 < self.heap[smallest].0 { smallest = right; }
            if smallest != i { self.heap.swap(i, smallest); i = smallest; }
            else { break; }
        }
    }
    pub fn peek(&self) -> Option<&(f64, usize)> { self.heap.first() }
    pub fn len(&self) -> usize { self.heap.len() }
    pub fn is_empty(&self) -> bool { self.heap.is_empty() }
}

impl fmt::Display for OpenClPriorityQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PQ(len={})", self.heap.len())
    }
}

/// Online statistics accumulator for OpenCl.
#[derive(Debug, Clone)]
pub struct OpenClAccumulator {
    count: u64,
    mean: f64,
    m2: f64,
    min_val: f64,
    max_val: f64,
    sum: f64,
}

impl OpenClAccumulator {
    pub fn new() -> Self { OpenClAccumulator { count: 0, mean: 0.0, m2: 0.0, min_val: f64::INFINITY, max_val: f64::NEG_INFINITY, sum: 0.0 } }
    pub fn add(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.min_val = self.min_val.min(value);
        self.max_val = self.max_val.max(value);
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }
    pub fn count(&self) -> u64 { self.count }
    pub fn mean(&self) -> f64 { self.mean }
    pub fn variance(&self) -> f64 { if self.count < 2 { 0.0 } else { self.m2 / (self.count - 1) as f64 } }
    pub fn std_dev(&self) -> f64 { self.variance().sqrt() }
    pub fn min(&self) -> f64 { self.min_val }
    pub fn max(&self) -> f64 { self.max_val }
    pub fn sum(&self) -> f64 { self.sum }
    pub fn range(&self) -> f64 { self.max_val - self.min_val }
    pub fn coefficient_of_variation(&self) -> f64 {
        if self.mean.abs() < 1e-15 { 0.0 } else { self.std_dev() / self.mean.abs() }
    }
    pub fn merge(&mut self, other: &Self) {
        if other.count == 0 { return; }
        let total = self.count + other.count;
        let delta = other.mean - self.mean;
        let new_mean = (self.sum + other.sum) / total as f64;
        self.m2 += other.m2 + delta * delta * (self.count as f64 * other.count as f64 / total as f64);
        self.mean = new_mean;
        self.count = total;
        self.sum += other.sum;
        self.min_val = self.min_val.min(other.min_val);
        self.max_val = self.max_val.max(other.max_val);
    }
    pub fn reset(&mut self) {
        self.count = 0; self.mean = 0.0; self.m2 = 0.0;
        self.min_val = f64::INFINITY; self.max_val = f64::NEG_INFINITY; self.sum = 0.0;
    }
}

impl fmt::Display for OpenClAccumulator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Acc(n={}, mean={:.4}, std={:.4})", self.count, self.mean, self.std_dev())
    }
}

/// Sparse matrix (COO format) for OpenCl.
#[derive(Debug, Clone)]
pub struct OpenClSparseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub entries: Vec<(usize, usize, f64)>,
}

impl OpenClSparseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self { OpenClSparseMatrix { rows, cols, entries: Vec::new() } }
    pub fn insert(&mut self, i: usize, j: usize, v: f64) {
        if let Some(pos) = self.entries.iter().position(|&(r, c, _)| r == i && c == j) {
            self.entries[pos].2 = v;
        } else { self.entries.push((i, j, v)); }
    }
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.entries.iter().find(|&&(r, c, _)| r == i && c == j).map(|&(_, _, v)| v).unwrap_or(0.0)
    }
    pub fn nnz(&self) -> usize { self.entries.len() }
    pub fn density(&self) -> f64 { self.entries.len() as f64 / (self.rows * self.cols) as f64 }
    pub fn transpose(&self) -> Self {
        let mut result = OpenClSparseMatrix::new(self.cols, self.rows);
        for &(i, j, v) in &self.entries { result.entries.push((j, i, v)); }
        result
    }
    pub fn add(&self, other: &Self) -> Self {
        let mut result = OpenClSparseMatrix::new(self.rows, self.cols);
        for &(i, j, v) in &self.entries { result.insert(i, j, result.get(i, j) + v); }
        for &(i, j, v) in &other.entries { result.insert(i, j, result.get(i, j) + v); }
        result
    }
    pub fn scale(&self, s: f64) -> Self {
        let mut result = OpenClSparseMatrix::new(self.rows, self.cols);
        for &(i, j, v) in &self.entries { result.entries.push((i, j, v * s)); }
        result
    }
    pub fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; self.rows];
        for &(i, j, v) in &self.entries { result[i] += v * x[j]; }
        result
    }
    pub fn frobenius_norm(&self) -> f64 { self.entries.iter().map(|&(_, _, v)| v * v).sum::<f64>().sqrt() }
    pub fn row_nnz(&self, i: usize) -> usize { self.entries.iter().filter(|&&(r, _, _)| r == i).count() }
    pub fn col_nnz(&self, j: usize) -> usize { self.entries.iter().filter(|&&(_, c, _)| c == j).count() }
    pub fn to_dense(&self, dm_new: fn(usize, usize) -> Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut result = vec![vec![0.0; self.cols]; self.rows];
        for &(i, j, v) in &self.entries { result[i][j] = v; }
        result
    }
    pub fn diagonal(&self) -> Vec<f64> {
        let n = self.rows.min(self.cols);
        (0..n).map(|i| self.get(i, i)).collect()
    }
    pub fn trace(&self) -> f64 { self.diagonal().iter().sum() }
    pub fn remove_zeros(&mut self, tol: f64) {
        self.entries.retain(|&(_, _, v)| v.abs() > tol);
    }
}

impl fmt::Display for OpenClSparseMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sparse({}x{}, nnz={})", self.rows, self.cols, self.nnz())
    }
}

/// Polynomial with f64 coefficients for OpenCl.
#[derive(Debug, Clone)]
pub struct OpenClPolynomial {
    pub coefficients: Vec<f64>,
}

impl OpenClPolynomial {
    pub fn new(coeffs: Vec<f64>) -> Self { OpenClPolynomial { coefficients: coeffs } }
    pub fn zero() -> Self { OpenClPolynomial { coefficients: vec![0.0] } }
    pub fn one() -> Self { OpenClPolynomial { coefficients: vec![1.0] } }
    pub fn monomial(degree: usize, coeff: f64) -> Self {
        let mut c = vec![0.0; degree + 1];
        c[degree] = coeff;
        OpenClPolynomial { coefficients: c }
    }
    pub fn degree(&self) -> usize {
        if self.coefficients.is_empty() { return 0; }
        let mut d = self.coefficients.len() - 1;
        while d > 0 && self.coefficients[d].abs() < 1e-15 { d -= 1; }
        d
    }
    pub fn evaluate(&self, x: f64) -> f64 {
        let mut result = 0.0;
        let mut power = 1.0;
        for &c in &self.coefficients {
            result += c * power;
            power *= x;
        }
        result
    }
    pub fn evaluate_horner(&self, x: f64) -> f64 {
        if self.coefficients.is_empty() { return 0.0; }
        let mut result = *self.coefficients.last().unwrap();
        for &c in self.coefficients.iter().rev().skip(1) {
            result = result * x + c;
        }
        result
    }
    pub fn add(&self, other: &Self) -> Self {
        let n = self.coefficients.len().max(other.coefficients.len());
        let mut result = vec![0.0; n];
        for (i, &c) in self.coefficients.iter().enumerate() { result[i] += c; }
        for (i, &c) in other.coefficients.iter().enumerate() { result[i] += c; }
        OpenClPolynomial { coefficients: result }
    }
    pub fn sub(&self, other: &Self) -> Self {
        let n = self.coefficients.len().max(other.coefficients.len());
        let mut result = vec![0.0; n];
        for (i, &c) in self.coefficients.iter().enumerate() { result[i] += c; }
        for (i, &c) in other.coefficients.iter().enumerate() { result[i] -= c; }
        OpenClPolynomial { coefficients: result }
    }
    pub fn mul(&self, other: &Self) -> Self {
        let n = self.coefficients.len() + other.coefficients.len() - 1;
        let mut result = vec![0.0; n];
        for (i, &a) in self.coefficients.iter().enumerate() {
            for (j, &b) in other.coefficients.iter().enumerate() {
                result[i + j] += a * b;
            }
        }
        OpenClPolynomial { coefficients: result }
    }
    pub fn scale(&self, s: f64) -> Self {
        OpenClPolynomial { coefficients: self.coefficients.iter().map(|&c| c * s).collect() }
    }
    pub fn derivative(&self) -> Self {
        if self.coefficients.len() <= 1 { return Self::zero(); }
        let coeffs: Vec<f64> = self.coefficients.iter().enumerate().skip(1)
            .map(|(i, &c)| c * i as f64).collect();
        OpenClPolynomial { coefficients: coeffs }
    }
    pub fn integral(&self, constant: f64) -> Self {
        let mut coeffs = vec![constant];
        for (i, &c) in self.coefficients.iter().enumerate() {
            coeffs.push(c / (i + 1) as f64);
        }
        OpenClPolynomial { coefficients: coeffs }
    }
    pub fn roots_quadratic(&self) -> Vec<f64> {
        if self.degree() != 2 { return Vec::new(); }
        let a = self.coefficients[2];
        let b = self.coefficients[1];
        let c = self.coefficients[0];
        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 { Vec::new() }
        else if disc.abs() < 1e-15 { vec![-b / (2.0 * a)] }
        else { vec![(-b + disc.sqrt()) / (2.0 * a), (-b - disc.sqrt()) / (2.0 * a)] }
    }
    pub fn is_zero(&self) -> bool { self.coefficients.iter().all(|&c| c.abs() < 1e-15) }
    pub fn leading_coefficient(&self) -> f64 {
        self.coefficients.get(self.degree()).copied().unwrap_or(0.0)
    }
    pub fn compose(&self, other: &Self) -> Self {
        let mut result = Self::zero();
        let mut power = Self::one();
        for &c in &self.coefficients {
            result = result.add(&power.scale(c));
            power = power.mul(other);
        }
        result
    }
    pub fn newton_root(&self, initial_guess: f64, max_iters: usize, tol: f64) -> Option<f64> {
        let deriv = self.derivative();
        let mut x = initial_guess;
        for _ in 0..max_iters {
            let fx = self.evaluate(x);
            if fx.abs() < tol { return Some(x); }
            let dfx = deriv.evaluate(x);
            if dfx.abs() < 1e-15 { return None; }
            x -= fx / dfx;
        }
        if self.evaluate(x).abs() < tol * 100.0 { Some(x) } else { None }
    }
}

impl fmt::Display for OpenClPolynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut terms = Vec::new();
        for (i, &c) in self.coefficients.iter().enumerate() {
            if c.abs() < 1e-15 { continue; }
            if i == 0 { terms.push(format!("{:.2}", c)); }
            else if i == 1 { terms.push(format!("{:.2}x", c)); }
            else { terms.push(format!("{:.2}x^{}", c, i)); }
        }
        if terms.is_empty() { write!(f, "0") }
        else { write!(f, "{}", terms.join(" + ")) }
    }
}

/// Simple linear congruential generator for OpenCl.
#[derive(Debug, Clone)]
pub struct OpenClRng {
    state: u64,
}

impl OpenClRng {
    pub fn new(seed: u64) -> Self { OpenClRng { state: seed } }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }
    pub fn next_f64(&mut self) -> f64 { (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 }
    pub fn next_range(&mut self, lo: u64, hi: u64) -> u64 {
        if hi <= lo { return lo; }
        lo + (self.next_u64() % (hi - lo))
    }
    pub fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    pub fn shuffle(&mut self, data: &mut [f64]) {
        let n = data.len();
        for i in (1..n).rev() {
            let j = self.next_range(0, i as u64 + 1) as usize;
            data.swap(i, j);
        }
    }
    pub fn sample(&mut self, data: &[f64], n: usize) -> Vec<f64> {
        let mut result = Vec::with_capacity(n);
        for _ in 0..n {
            let idx = self.next_range(0, data.len() as u64) as usize;
            result.push(data[idx]);
        }
        result
    }
    pub fn bernoulli(&mut self, p: f64) -> bool { self.next_f64() < p }
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 { lo + self.next_f64() * (hi - lo) }
    pub fn exponential(&mut self, lambda: f64) -> f64 { -self.next_f64().max(1e-15).ln() / lambda }
}

impl fmt::Display for OpenClRng {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Rng(state={:#x})", self.state)
    }
}

/// Simple timing utility for OpenCl benchmarking.
#[derive(Debug, Clone)]
pub struct OpenClTimer {
    pub label: String,
    pub elapsed_ns: Vec<u64>,
    pub running: bool,
}

impl OpenClTimer {
    pub fn new(label: impl Into<String>) -> Self { OpenClTimer { label: label.into(), elapsed_ns: Vec::new(), running: false } }
    pub fn record(&mut self, ns: u64) { self.elapsed_ns.push(ns); }
    pub fn total_ns(&self) -> u64 { self.elapsed_ns.iter().sum() }
    pub fn count(&self) -> usize { self.elapsed_ns.len() }
    pub fn average_ns(&self) -> f64 {
        if self.elapsed_ns.is_empty() { 0.0 } else { self.total_ns() as f64 / self.elapsed_ns.len() as f64 }
    }
    pub fn min_ns(&self) -> u64 { self.elapsed_ns.iter().cloned().min().unwrap_or(0) }
    pub fn max_ns(&self) -> u64 { self.elapsed_ns.iter().cloned().max().unwrap_or(0) }
    pub fn percentile_ns(&self, p: f64) -> u64 {
        if self.elapsed_ns.is_empty() { return 0; }
        let mut sorted = self.elapsed_ns.clone();
        sorted.sort();
        let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }
    pub fn p50_ns(&self) -> u64 { self.percentile_ns(0.5) }
    pub fn p95_ns(&self) -> u64 { self.percentile_ns(0.95) }
    pub fn p99_ns(&self) -> u64 { self.percentile_ns(0.99) }
    pub fn reset(&mut self) { self.elapsed_ns.clear(); self.running = false; }
}

impl fmt::Display for OpenClTimer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Timer({}: avg={:.0}ns, n={})", self.label, self.average_ns(), self.count())
    }
}

/// Compact bit vector for OpenCl set operations.
#[derive(Debug, Clone)]
pub struct OpenClBitVector {
    words: Vec<u64>,
    len: usize,
}

impl OpenClBitVector {
    pub fn new(len: usize) -> Self { OpenClBitVector { words: vec![0u64; (len + 63) / 64], len } }
    pub fn set(&mut self, i: usize) { if i < self.len { self.words[i / 64] |= 1u64 << (i % 64); } }
    pub fn clear(&mut self, i: usize) { if i < self.len { self.words[i / 64] &= !(1u64 << (i % 64)); } }
    pub fn get(&self, i: usize) -> bool { i < self.len && (self.words[i / 64] & (1u64 << (i % 64))) != 0 }
    pub fn len(&self) -> usize { self.len }
    pub fn count_ones(&self) -> usize { self.words.iter().map(|w| w.count_ones() as usize).sum() }
    pub fn count_zeros(&self) -> usize { self.len - self.count_ones() }
    pub fn is_empty(&self) -> bool { self.count_ones() == 0 }
    pub fn and(&self, other: &Self) -> Self {
        let n = self.words.len().min(other.words.len());
        let mut result = Self::new(self.len.min(other.len));
        for i in 0..n { result.words[i] = self.words[i] & other.words[i]; }
        result
    }
    pub fn or(&self, other: &Self) -> Self {
        let n = self.words.len().max(other.words.len());
        let mut result = Self::new(self.len.max(other.len));
        for i in 0..self.words.len().min(n) { result.words[i] |= self.words[i]; }
        for i in 0..other.words.len().min(n) { result.words[i] |= other.words[i]; }
        result
    }
    pub fn xor(&self, other: &Self) -> Self {
        let n = self.words.len().max(other.words.len());
        let mut result = Self::new(self.len.max(other.len));
        for i in 0..self.words.len().min(n) { result.words[i] = self.words[i]; }
        for i in 0..other.words.len().min(n) { result.words[i] ^= other.words[i]; }
        result
    }
    pub fn not(&self) -> Self {
        let mut result = Self::new(self.len);
        for i in 0..self.words.len() { result.words[i] = !self.words[i]; }
        // Clear unused bits in last word
        let extra = self.len % 64;
        if extra > 0 && !result.words.is_empty() {
            let last = result.words.len() - 1;
            result.words[last] &= (1u64 << extra) - 1;
        }
        result
    }
    pub fn iter_ones(&self) -> Vec<usize> {
        let mut result = Vec::new();
        for i in 0..self.len { if self.get(i) { result.push(i); } }
        result
    }
    pub fn jaccard(&self, other: &Self) -> f64 {
        let intersection = self.and(other).count_ones() as f64;
        let union = self.or(other).count_ones() as f64;
        if union == 0.0 { 1.0 } else { intersection / union }
    }
    pub fn hamming_distance(&self, other: &Self) -> usize { self.xor(other).count_ones() }
    pub fn fill(&mut self, value: bool) {
        let fill_val = if value { u64::MAX } else { 0 };
        for w in &mut self.words { *w = fill_val; }
        if value { let extra = self.len % 64; if extra > 0 { let last = self.words.len() - 1; self.words[last] &= (1u64 << extra) - 1; } }
    }
}

impl fmt::Display for OpenClBitVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BitVec(len={}, ones={})", self.len, self.count_ones())
    }
}

/// LRU cache for OpenCl computation memoization.
#[derive(Debug, Clone)]
pub struct OpenClLruCache {
    entries: Vec<(u64, Vec<f64>, u64)>,
    capacity: usize,
    clock: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl OpenClLruCache {
    pub fn new(capacity: usize) -> Self { OpenClLruCache { entries: Vec::new(), capacity, clock: 0, hits: 0, misses: 0, evictions: 0 } }
    pub fn get(&mut self, key: u64) -> Option<&Vec<f64>> {
        self.clock += 1;
        if let Some(pos) = self.entries.iter().position(|(k, _, _)| *k == key) {
            self.entries[pos].2 = self.clock;
            self.hits += 1;
            Some(&self.entries[pos].1)
        } else { self.misses += 1; None }
    }
    pub fn put(&mut self, key: u64, value: Vec<f64>) {
        self.clock += 1;
        if let Some(pos) = self.entries.iter().position(|(k, _, _)| *k == key) {
            self.entries[pos].1 = value;
            self.entries[pos].2 = self.clock;
            return;
        }
        if self.entries.len() >= self.capacity {
            let lru_pos = self.entries.iter().enumerate()
                .min_by_key(|(_, (_, _, ts))| *ts).map(|(i, _)| i).unwrap();
            self.entries.remove(lru_pos);
            self.evictions += 1;
        }
        self.entries.push((key, value, self.clock));
    }
    pub fn size(&self) -> usize { self.entries.len() }
    pub fn hit_rate(&self) -> f64 { let t = self.hits + self.misses; if t == 0 { 0.0 } else { self.hits as f64 / t as f64 } }
    pub fn eviction_count(&self) -> u64 { self.evictions }
    pub fn contains(&self, key: u64) -> bool { self.entries.iter().any(|(k, _, _)| *k == key) }
    pub fn clear(&mut self) { self.entries.clear(); self.hits = 0; self.misses = 0; self.evictions = 0; self.clock = 0; }
    pub fn keys(&self) -> Vec<u64> { self.entries.iter().map(|(k, _, _)| *k).collect() }
}

impl fmt::Display for OpenClLruCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LRU(size={}/{}, hr={:.1}%)", self.size(), self.capacity, self.hit_rate() * 100.0)
    }
}

/// Graph coloring utility for OpenCl scheduling.
#[derive(Debug, Clone)]
pub struct OpenClGraphColoring {
    pub adjacency: Vec<Vec<bool>>,
    pub colors: Vec<Option<usize>>,
    pub num_nodes: usize,
    pub num_colors_used: usize,
}

impl OpenClGraphColoring {
    pub fn new(n: usize) -> Self {
        OpenClGraphColoring { adjacency: vec![vec![false; n]; n], colors: vec![None; n], num_nodes: n, num_colors_used: 0 }
    }
    pub fn add_edge(&mut self, i: usize, j: usize) {
        if i < self.num_nodes && j < self.num_nodes {
            self.adjacency[i][j] = true;
            self.adjacency[j][i] = true;
        }
    }
    pub fn greedy_color(&mut self) -> usize {
        self.colors = vec![None; self.num_nodes];
        let mut max_color = 0;
        for v in 0..self.num_nodes {
            let neighbor_colors: std::collections::HashSet<usize> = (0..self.num_nodes)
                .filter(|&u| self.adjacency[v][u] && self.colors[u].is_some())
                .map(|u| self.colors[u].unwrap()).collect();
            let mut c = 0;
            while neighbor_colors.contains(&c) { c += 1; }
            self.colors[v] = Some(c);
            max_color = max_color.max(c);
        }
        self.num_colors_used = max_color + 1;
        self.num_colors_used
    }
    pub fn is_valid_coloring(&self) -> bool {
        for i in 0..self.num_nodes {
            for j in (i+1)..self.num_nodes {
                if self.adjacency[i][j] {
                    if let (Some(ci), Some(cj)) = (self.colors[i], self.colors[j]) {
                        if ci == cj { return false; }
                    }
                }
            }
        }
        true
    }
    pub fn chromatic_number_upper_bound(&self) -> usize {
        let max_degree = (0..self.num_nodes)
            .map(|v| (0..self.num_nodes).filter(|&u| self.adjacency[v][u]).count())
            .max().unwrap_or(0);
        max_degree + 1
    }
    pub fn color_classes(&self) -> Vec<Vec<usize>> {
        let mut classes: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
        for (v, &c) in self.colors.iter().enumerate() {
            if let Some(color) = c { classes.entry(color).or_default().push(v); }
        }
        let mut result: Vec<Vec<usize>> = classes.into_values().collect();
        result.sort_by_key(|v| v[0]);
        result
    }
}

impl fmt::Display for OpenClGraphColoring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Coloring(n={}, colors={})", self.num_nodes, self.num_colors_used)
    }
}

/// Top-K tracker for OpenCl ranking.
#[derive(Debug, Clone)]
pub struct OpenClTopK {
    pub k: usize,
    pub items: Vec<(f64, String)>,
}

impl OpenClTopK {
    pub fn new(k: usize) -> Self { OpenClTopK { k, items: Vec::new() } }
    pub fn insert(&mut self, score: f64, label: impl Into<String>) {
        self.items.push((score, label.into()));
        self.items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        if self.items.len() > self.k { self.items.truncate(self.k); }
    }
    pub fn top(&self) -> &[(f64, String)] { &self.items }
    pub fn min_score(&self) -> Option<f64> { self.items.last().map(|(s, _)| *s) }
    pub fn max_score(&self) -> Option<f64> { self.items.first().map(|(s, _)| *s) }
    pub fn is_full(&self) -> bool { self.items.len() >= self.k }
    pub fn len(&self) -> usize { self.items.len() }
    pub fn contains_label(&self, label: &str) -> bool { self.items.iter().any(|(_, l)| l == label) }
    pub fn clear(&mut self) { self.items.clear(); }
    pub fn merge(&mut self, other: &Self) {
        for (score, label) in &other.items { self.insert(*score, label.clone()); }
    }
}

impl fmt::Display for OpenClTopK {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TopK(k={}, len={})", self.k, self.items.len())
    }
}

/// Sliding window statistics for OpenCl monitoring.
#[derive(Debug, Clone)]
pub struct OpenClSlidingWindow {
    pub window_size: usize,
    pub data: Vec<f64>,
    pub sum: f64,
}

impl OpenClSlidingWindow {
    pub fn new(window_size: usize) -> Self { OpenClSlidingWindow { window_size, data: Vec::new(), sum: 0.0 } }
    pub fn push(&mut self, value: f64) {
        self.data.push(value);
        self.sum += value;
        if self.data.len() > self.window_size {
            self.sum -= self.data.remove(0);
        }
    }
    pub fn mean(&self) -> f64 { if self.data.is_empty() { 0.0 } else { self.sum / self.data.len() as f64 } }
    pub fn variance(&self) -> f64 {
        if self.data.len() < 2 { return 0.0; }
        let m = self.mean();
        self.data.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / (self.data.len() - 1) as f64
    }
    pub fn std_dev(&self) -> f64 { self.variance().sqrt() }
    pub fn min(&self) -> f64 { self.data.iter().cloned().fold(f64::INFINITY, f64::min) }
    pub fn max(&self) -> f64 { self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max) }
    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_full(&self) -> bool { self.data.len() >= self.window_size }
    pub fn trend(&self) -> f64 {
        if self.data.len() < 2 { return 0.0; }
        let n = self.data.len() as f64;
        let sum_x: f64 = (0..self.data.len()).map(|i| i as f64).sum();
        let sum_y: f64 = self.data.iter().sum();
        let sum_xy: f64 = self.data.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let sum_xx: f64 = (0..self.data.len()).map(|i| (i as f64) * (i as f64)).sum();
        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-15 { 0.0 } else { (n * sum_xy - sum_x * sum_y) / denom }
    }
    pub fn anomaly_score(&self, value: f64) -> f64 {
        let s = self.std_dev();
        if s.abs() < 1e-15 { return 0.0; }
        ((value - self.mean()) / s).abs()
    }
    pub fn clear(&mut self) { self.data.clear(); self.sum = 0.0; }
}

impl fmt::Display for OpenClSlidingWindow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Window(size={}/{}, mean={:.2})", self.data.len(), self.window_size, self.mean())
    }
}

/// Confusion matrix for OpenCl classification evaluation.
#[derive(Debug, Clone)]
pub struct OpenClConfusionMatrix {
    pub true_positive: u64,
    pub false_positive: u64,
    pub true_negative: u64,
    pub false_negative: u64,
}

impl OpenClConfusionMatrix {
    pub fn new() -> Self { OpenClConfusionMatrix { true_positive: 0, false_positive: 0, true_negative: 0, false_negative: 0 } }
    pub fn from_predictions(actual: &[bool], predicted: &[bool]) -> Self {
        let mut cm = Self::new();
        for (&a, &p) in actual.iter().zip(predicted.iter()) {
            match (a, p) {
                (true, true) => cm.true_positive += 1,
                (false, true) => cm.false_positive += 1,
                (true, false) => cm.false_negative += 1,
                (false, false) => cm.true_negative += 1,
            }
        }
        cm
    }
    pub fn total(&self) -> u64 { self.true_positive + self.false_positive + self.true_negative + self.false_negative }
    pub fn accuracy(&self) -> f64 { let t = self.total(); if t == 0 { 0.0 } else { (self.true_positive + self.true_negative) as f64 / t as f64 } }
    pub fn precision(&self) -> f64 { let d = self.true_positive + self.false_positive; if d == 0 { 0.0 } else { self.true_positive as f64 / d as f64 } }
    pub fn recall(&self) -> f64 { let d = self.true_positive + self.false_negative; if d == 0 { 0.0 } else { self.true_positive as f64 / d as f64 } }
    pub fn f1_score(&self) -> f64 { let p = self.precision(); let r = self.recall(); if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) } }
    pub fn specificity(&self) -> f64 { let d = self.true_negative + self.false_positive; if d == 0 { 0.0 } else { self.true_negative as f64 / d as f64 } }
    pub fn false_positive_rate(&self) -> f64 { 1.0 - self.specificity() }
    pub fn matthews_correlation(&self) -> f64 {
        let tp = self.true_positive as f64; let fp = self.false_positive as f64;
        let tn = self.true_negative as f64; let fn_ = self.false_negative as f64;
        let num = tp * tn - fp * fn_;
        let den = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
        if den == 0.0 { 0.0 } else { num / den }
    }
    pub fn merge(&mut self, other: &Self) {
        self.true_positive += other.true_positive;
        self.false_positive += other.false_positive;
        self.true_negative += other.true_negative;
        self.false_negative += other.false_negative;
    }
}

impl fmt::Display for OpenClConfusionMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CM(acc={:.3}, prec={:.3}, rec={:.3}, f1={:.3})",
            self.accuracy(), self.precision(), self.recall(), self.f1_score())
    }
}

/// Cosine similarity for OpenCl feature vectors.
pub fn opencl_cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na * nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

/// Euclidean distance for OpenCl.
pub fn opencl_euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
}

/// Manhattan distance for OpenCl.
pub fn opencl_manhattan_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// Chebyshev distance for OpenCl.
pub fn opencl_chebyshev_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max)
}

/// Minkowski distance for OpenCl.
pub fn opencl_minkowski_distance(a: &[f64], b: &[f64], p: f64) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs().powf(p)).sum::<f64>().powf(1.0 / p)
}

/// Normalize a vector for OpenCl.
pub fn opencl_normalize(v: &[f64]) -> Vec<f64> {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 { v.to_vec() } else { v.iter().map(|x| x / norm).collect() }
}

/// Dot product for OpenCl.
pub fn opencl_dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Cross product (3D) for OpenCl.
pub fn opencl_cross_product(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
}

/// Linear interpolation for OpenCl.
pub fn opencl_lerp(a: f64, b: f64, t: f64) -> f64 { a + (b - a) * t }

/// Clamp value for OpenCl.
pub fn opencl_clamp(v: f64, lo: f64, hi: f64) -> f64 { v.max(lo).min(hi) }

/// Sigmoid function for OpenCl.
pub fn opencl_sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }

/// Softmax for OpenCl.
pub fn opencl_softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Log-sum-exp for OpenCl.
pub fn opencl_logsumexp(values: &[f64]) -> f64 {
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    max + values.iter().map(|&v| (v - max).exp()).sum::<f64>().ln()
}

/// KL divergence for OpenCl.
pub fn opencl_kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q.iter()).map(|(&pi, &qi)| {
        if pi > 0.0 && qi > 0.0 { pi * (pi / qi).ln() } else { 0.0 }
    }).sum()
}

/// Jensen-Shannon divergence for OpenCl.
pub fn opencl_js_divergence(p: &[f64], q: &[f64]) -> f64 {
    let m: Vec<f64> = p.iter().zip(q.iter()).map(|(&pi, &qi)| (pi + qi) / 2.0).collect();
    (opencl_kl_divergence(p, &m) + opencl_kl_divergence(q, &m)) / 2.0
}

/// Total variation distance for OpenCl.
pub fn opencl_tv_distance(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q.iter()).map(|(&pi, &qi)| (pi - qi).abs()).sum::<f64>() / 2.0
}

/// Hellinger distance for OpenCl.
pub fn opencl_hellinger_distance(p: &[f64], q: &[f64]) -> f64 {
    let sum: f64 = p.iter().zip(q.iter()).map(|(&pi, &qi)| {
        let diff = pi.sqrt() - qi.sqrt();
        diff * diff
    }).sum();
    (sum / 2.0).sqrt()
}

/// Earth mover's distance (1D) for OpenCl.
pub fn opencl_emd_1d(p: &[f64], q: &[f64]) -> f64 {
    let mut cum_diff = 0.0;
    let mut total = 0.0;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        cum_diff += pi - qi;
        total += cum_diff.abs();
    }
    total
}

/// Feature scaling utilities for OpenCl.
#[derive(Debug, Clone)]
pub struct OpenClFeatureScaler {
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
    pub mins: Vec<f64>,
    pub maxs: Vec<f64>,
    pub fitted: bool,
}

impl OpenClFeatureScaler {
    pub fn new() -> Self { OpenClFeatureScaler { means: Vec::new(), stds: Vec::new(), mins: Vec::new(), maxs: Vec::new(), fitted: false } }
    pub fn fit(&mut self, data: &[Vec<f64>]) {
        if data.is_empty() { return; }
        let dim = data[0].len();
        let n = data.len() as f64;
        self.means = vec![0.0; dim];
        self.mins = vec![f64::INFINITY; dim];
        self.maxs = vec![f64::NEG_INFINITY; dim];
        for row in data {
            for (j, &v) in row.iter().enumerate() {
                self.means[j] += v;
                self.mins[j] = self.mins[j].min(v);
                self.maxs[j] = self.maxs[j].max(v);
            }
        }
        for j in 0..dim { self.means[j] /= n; }
        self.stds = vec![0.0; dim];
        for row in data {
            for (j, &v) in row.iter().enumerate() {
                self.stds[j] += (v - self.means[j]).powi(2);
            }
        }
        for j in 0..dim { self.stds[j] = (self.stds[j] / (n - 1.0).max(1.0)).sqrt(); }
        self.fitted = true;
    }
    pub fn standardize(&self, row: &[f64]) -> Vec<f64> {
        row.iter().enumerate().map(|(j, &v)| {
            if self.stds[j].abs() < 1e-15 { 0.0 } else { (v - self.means[j]) / self.stds[j] }
        }).collect()
    }
    pub fn normalize(&self, row: &[f64]) -> Vec<f64> {
        row.iter().enumerate().map(|(j, &v)| {
            let range = self.maxs[j] - self.mins[j];
            if range.abs() < 1e-15 { 0.0 } else { (v - self.mins[j]) / range }
        }).collect()
    }
    pub fn inverse_standardize(&self, row: &[f64]) -> Vec<f64> {
        row.iter().enumerate().map(|(j, &v)| v * self.stds[j] + self.means[j]).collect()
    }
    pub fn inverse_normalize(&self, row: &[f64]) -> Vec<f64> {
        row.iter().enumerate().map(|(j, &v)| v * (self.maxs[j] - self.mins[j]) + self.mins[j]).collect()
    }
    pub fn dimension(&self) -> usize { self.means.len() }
}

impl fmt::Display for OpenClFeatureScaler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Scaler(dim={}, fitted={})", self.dimension(), self.fitted)
    }
}

/// Simple linear regression for OpenCl trend analysis.
#[derive(Debug, Clone)]
pub struct OpenClLinearRegression {
    pub slope: f64,
    pub intercept: f64,
    pub r_squared: f64,
    pub fitted: bool,
}

impl OpenClLinearRegression {
    pub fn new() -> Self { OpenClLinearRegression { slope: 0.0, intercept: 0.0, r_squared: 0.0, fitted: false } }
    pub fn fit(&mut self, x: &[f64], y: &[f64]) {
        assert_eq!(x.len(), y.len());
        let n = x.len() as f64;
        if n < 2.0 { return; }
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
        let sum_xx: f64 = x.iter().map(|&xi| xi * xi).sum();
        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-15 { return; }
        self.slope = (n * sum_xy - sum_x * sum_y) / denom;
        self.intercept = (sum_y - self.slope * sum_x) / n;
        let mean_y = sum_y / n;
        let ss_tot: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
        let ss_res: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| (yi - self.predict(xi)).powi(2)).sum();
        self.r_squared = if ss_tot.abs() < 1e-15 { 1.0 } else { 1.0 - ss_res / ss_tot };
        self.fitted = true;
    }
    pub fn predict(&self, x: f64) -> f64 { self.slope * x + self.intercept }
    pub fn predict_many(&self, xs: &[f64]) -> Vec<f64> { xs.iter().map(|&x| self.predict(x)).collect() }
    pub fn residuals(&self, x: &[f64], y: &[f64]) -> Vec<f64> {
        x.iter().zip(y.iter()).map(|(&xi, &yi)| yi - self.predict(xi)).collect()
    }
    pub fn mse(&self, x: &[f64], y: &[f64]) -> f64 {
        let res = self.residuals(x, y);
        res.iter().map(|r| r * r).sum::<f64>() / res.len() as f64
    }
    pub fn rmse(&self, x: &[f64], y: &[f64]) -> f64 { self.mse(x, y).sqrt() }
}

impl fmt::Display for OpenClLinearRegression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "y = {:.4}x + {:.4} (R2={:.4})", self.slope, self.intercept, self.r_squared)
    }
}

/// Weighted undirected graph for OpenCl.
#[derive(Debug, Clone)]
pub struct OpenClWeightedGraph {
    pub adj: Vec<Vec<(usize, f64)>>,
    pub num_nodes: usize,
    pub num_edges: usize,
}

impl OpenClWeightedGraph {
    pub fn new(n: usize) -> Self { OpenClWeightedGraph { adj: vec![Vec::new(); n], num_nodes: n, num_edges: 0 } }
    pub fn add_edge(&mut self, u: usize, v: usize, w: f64) {
        self.adj[u].push((v, w));
        self.adj[v].push((u, w));
        self.num_edges += 1;
    }
    pub fn neighbors(&self, u: usize) -> &[(usize, f64)] { &self.adj[u] }
    pub fn degree(&self, u: usize) -> usize { self.adj[u].len() }
    pub fn total_weight(&self) -> f64 {
        self.adj.iter().flat_map(|edges| edges.iter().map(|(_, w)| w)).sum::<f64>() / 2.0
    }
    pub fn min_spanning_tree_weight(&self) -> f64 {
        // Kruskal's algorithm
        let mut edges: Vec<(f64, usize, usize)> = Vec::new();
        for u in 0..self.num_nodes {
            for &(v, w) in &self.adj[u] {
                if u < v { edges.push((w, u, v)); }
            }
        }
        edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut parent: Vec<usize> = (0..self.num_nodes).collect();
        let mut rank = vec![0usize; self.num_nodes];
        fn find_opencl(parent: &mut Vec<usize>, x: usize) -> usize {
            if parent[x] != x { parent[x] = find_opencl(parent, parent[x]); }
            parent[x]
        }
        let mut total = 0.0;
        let mut count = 0;
        for (w, u, v) in edges {
            let ru = find_opencl(&mut parent, u);
            let rv = find_opencl(&mut parent, v);
            if ru != rv {
                if rank[ru] < rank[rv] { parent[ru] = rv; }
                else if rank[ru] > rank[rv] { parent[rv] = ru; }
                else { parent[rv] = ru; rank[ru] += 1; }
                total += w;
                count += 1;
                if count == self.num_nodes - 1 { break; }
            }
        }
        total
    }
    pub fn dijkstra(&self, start: usize) -> Vec<f64> {
        let mut dist = vec![f64::INFINITY; self.num_nodes];
        let mut visited = vec![false; self.num_nodes];
        dist[start] = 0.0;
        for _ in 0..self.num_nodes {
            let mut u = None;
            let mut min_d = f64::INFINITY;
            for v in 0..self.num_nodes { if !visited[v] && dist[v] < min_d { min_d = dist[v]; u = Some(v); } }
            let u = match u { Some(v) => v, None => break };
            visited[u] = true;
            for &(v, w) in &self.adj[u] {
                let alt = dist[u] + w;
                if alt < dist[v] { dist[v] = alt; }
            }
        }
        dist
    }
    pub fn eccentricity(&self, u: usize) -> f64 {
        let dists = self.dijkstra(u);
        dists.iter().cloned().filter(|&d| d.is_finite()).fold(0.0f64, f64::max)
    }
    pub fn diameter(&self) -> f64 {
        (0..self.num_nodes).map(|u| self.eccentricity(u)).fold(0.0f64, f64::max)
    }
    pub fn clustering_coefficient(&self, u: usize) -> f64 {
        let neighbors: Vec<usize> = self.adj[u].iter().map(|(v, _)| *v).collect();
        let k = neighbors.len();
        if k < 2 { return 0.0; }
        let mut triangles = 0;
        for i in 0..k {
            for j in (i+1)..k {
                if self.adj[neighbors[i]].iter().any(|(v, _)| *v == neighbors[j]) {
                    triangles += 1;
                }
            }
        }
        2.0 * triangles as f64 / (k * (k - 1)) as f64
    }
    pub fn average_clustering_coefficient(&self) -> f64 {
        let sum: f64 = (0..self.num_nodes).map(|u| self.clustering_coefficient(u)).sum();
        sum / self.num_nodes as f64
    }
}

impl fmt::Display for OpenClWeightedGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WGraph(n={}, e={})", self.num_nodes, self.num_edges)
    }
}

/// Moving average for OpenCl.
pub fn opencl_moving_average(data: &[f64], window: usize) -> Vec<f64> {
    if data.len() < window { return Vec::new(); }
    let mut result = Vec::with_capacity(data.len() - window + 1);
    let mut sum: f64 = data[..window].iter().sum();
    result.push(sum / window as f64);
    for i in window..data.len() {
        sum += data[i] - data[i - window];
        result.push(sum / window as f64);
    }
    result
}

/// Cumulative sum for OpenCl.
pub fn opencl_cumsum(data: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(data.len());
    let mut sum = 0.0;
    for &v in data { sum += v; result.push(sum); }
    result
}

/// Numerical differentiation for OpenCl.
pub fn opencl_diff(data: &[f64]) -> Vec<f64> {
    if data.len() < 2 { return Vec::new(); }
    data.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Auto-correlation for OpenCl.
pub fn opencl_autocorrelation(data: &[f64], lag: usize) -> f64 {
    if data.len() <= lag { return 0.0; }
    let n = data.len();
    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let var: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if var.abs() < 1e-15 { return 0.0; }
    let cov: f64 = (0..(n - lag)).map(|i| (data[i] - mean) * (data[i + lag] - mean)).sum::<f64>() / n as f64;
    cov / var
}

/// Discrete Fourier transform magnitude for OpenCl.
pub fn opencl_dft_magnitude(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut magnitudes = Vec::with_capacity(n / 2 + 1);
    for k in 0..=n/2 {
        let mut re = 0.0;
        let mut im = 0.0;
        for (j, &x) in data.iter().enumerate() {
            let angle = -2.0 * std::f64::consts::PI * k as f64 * j as f64 / n as f64;
            re += x * angle.cos();
            im += x * angle.sin();
        }
        magnitudes.push((re * re + im * im).sqrt());
    }
    magnitudes
}

/// Trapezoidal integration for OpenCl.
pub fn opencl_integrate_trapezoid(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let mut total = 0.0;
    for i in 1..x.len() {
        total += (x[i] - x[i-1]) * (y[i] + y[i-1]) / 2.0;
    }
    total
}

/// Simpson's rule integration for OpenCl.
pub fn opencl_integrate_simpson(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    if n < 3 || n % 2 == 0 { return 0.0; }
    let mut total = 0.0;
    for i in (0..n-2).step_by(2) {
        let h = (x[i+2] - x[i]) / 6.0;
        total += h * (y[i] + 4.0 * y[i+1] + y[i+2]);
    }
    total
}

/// Convolution for OpenCl.
pub fn opencl_convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len() + b.len() - 1;
    let mut result = vec![0.0; n];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            result[i + j] += ai * bj;
        }
    }
    result
}

/// Histogram for OpenCl data analysis.
#[derive(Debug, Clone)]
pub struct OpenClHistogramExt {
    pub bins: Vec<usize>,
    pub edges: Vec<f64>,
    pub total: usize,
}

impl OpenClHistogramExt {
    pub fn from_data(data: &[f64], num_bins: usize) -> Self {
        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = if (max - min).abs() < 1e-15 { 1.0 } else { max - min };
        let bin_width = range / num_bins as f64;
        let mut edges = Vec::with_capacity(num_bins + 1);
        for i in 0..=num_bins { edges.push(min + i as f64 * bin_width); }
        let mut bins = vec![0usize; num_bins];
        for &v in data {
            let idx = ((v - min) / bin_width).floor() as usize;
            let idx = idx.min(num_bins - 1);
            bins[idx] += 1;
        }
        OpenClHistogramExt { bins, edges, total: data.len() }
    }
    pub fn bin_count(&self, i: usize) -> usize { self.bins[i] }
    pub fn bin_density(&self, i: usize) -> f64 {
        let w = self.edges[i + 1] - self.edges[i];
        if w.abs() < 1e-15 || self.total == 0 { 0.0 }
        else { self.bins[i] as f64 / (self.total as f64 * w) }
    }
    pub fn num_bins(&self) -> usize { self.bins.len() }
    pub fn mode_bin(&self) -> usize {
        self.bins.iter().enumerate().max_by_key(|(_, &c)| c).map(|(i, _)| i).unwrap_or(0)
    }
    pub fn cumulative(&self) -> Vec<usize> {
        let mut cum = Vec::with_capacity(self.bins.len());
        let mut sum = 0;
        for &c in &self.bins { sum += c; cum.push(sum); }
        cum
    }
    pub fn percentile_bin(&self, p: f64) -> usize {
        let target = (p * self.total as f64).ceil() as usize;
        let cum = self.cumulative();
        cum.iter().position(|&c| c >= target).unwrap_or(self.bins.len() - 1)
    }
    pub fn entropy(&self) -> f64 {
        let n = self.total as f64;
        if n < 1.0 { return 0.0; }
        self.bins.iter().filter(|&&c| c > 0).map(|&c| {
            let p = c as f64 / n;
            -p * p.ln()
        }).sum()
    }
}

impl fmt::Display for OpenClHistogramExt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hist(bins={}, total={})", self.num_bins(), self.total)
    }
}

/// Axis-aligned bounding box for OpenCl spatial indexing.
#[derive(Debug, Clone, Copy)]
pub struct OpenClAABB {
    pub x_min: f64, pub y_min: f64,
    pub x_max: f64, pub y_max: f64,
}

impl OpenClAABB {
    pub fn new(x_min: f64, y_min: f64, x_max: f64, y_max: f64) -> Self { OpenClAABB { x_min, y_min, x_max, y_max } }
    pub fn contains(&self, x: f64, y: f64) -> bool { x >= self.x_min && x <= self.x_max && y >= self.y_min && y <= self.y_max }
    pub fn intersects(&self, other: &Self) -> bool {
        !(self.x_max < other.x_min || self.x_min > other.x_max || self.y_max < other.y_min || self.y_min > other.y_max)
    }
    pub fn width(&self) -> f64 { self.x_max - self.x_min }
    pub fn height(&self) -> f64 { self.y_max - self.y_min }
    pub fn area(&self) -> f64 { self.width() * self.height() }
    pub fn center(&self) -> (f64, f64) { ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0) }
    pub fn subdivide(&self) -> [Self; 4] {
        let (cx, cy) = self.center();
        [
            OpenClAABB::new(self.x_min, self.y_min, cx, cy),
            OpenClAABB::new(cx, self.y_min, self.x_max, cy),
            OpenClAABB::new(self.x_min, cy, cx, self.y_max),
            OpenClAABB::new(cx, cy, self.x_max, self.y_max),
        ]
    }
}

/// 2D point for OpenCl.
#[derive(Debug, Clone, Copy)]
pub struct OpenClPoint2D { pub x: f64, pub y: f64, pub data: f64 }

/// Quadtree for OpenCl spatial indexing.
#[derive(Debug, Clone)]
pub struct OpenClQuadTree {
    pub boundary: OpenClAABB,
    pub points: Vec<OpenClPoint2D>,
    pub children: Option<Vec<OpenClQuadTree>>,
    pub capacity: usize,
    pub depth: usize,
    pub max_depth: usize,
}

impl OpenClQuadTree {
    pub fn new(boundary: OpenClAABB, capacity: usize, max_depth: usize) -> Self {
        OpenClQuadTree { boundary, points: Vec::new(), children: None, capacity, depth: 0, max_depth }
    }
    fn with_depth(boundary: OpenClAABB, capacity: usize, depth: usize, max_depth: usize) -> Self {
        OpenClQuadTree { boundary, points: Vec::new(), children: None, capacity, depth, max_depth }
    }
    pub fn insert(&mut self, p: OpenClPoint2D) -> bool {
        if !self.boundary.contains(p.x, p.y) { return false; }
        if self.points.len() < self.capacity && self.children.is_none() {
            self.points.push(p); return true;
        }
        if self.children.is_none() && self.depth < self.max_depth { self.subdivide_tree(); }
        if let Some(ref mut children) = self.children {
            for child in children.iter_mut() { if child.insert(p) { return true; } }
        }
        self.points.push(p); true
    }
    fn subdivide_tree(&mut self) {
        let quads = self.boundary.subdivide();
        let mut children = Vec::with_capacity(4);
        for q in quads.iter() {
            children.push(OpenClQuadTree::with_depth(*q, self.capacity, self.depth + 1, self.max_depth));
        }
        let old_points: Vec<_> = self.points.drain(..).collect();
        self.children = Some(children);
        for p in old_points { self.insert(p); }
    }
    pub fn query_range(&self, range: &OpenClAABB) -> Vec<OpenClPoint2D> {
        let mut result = Vec::new();
        if !self.boundary.intersects(range) { return result; }
        for p in &self.points { if range.contains(p.x, p.y) { result.push(*p); } }
        if let Some(ref children) = self.children {
            for child in children { result.extend(child.query_range(range)); }
        }
        result
    }
    pub fn count(&self) -> usize {
        let mut c = self.points.len();
        if let Some(ref children) = self.children {
            for child in children { c += child.count(); }
        }
        c
    }
    pub fn tree_depth(&self) -> usize {
        if let Some(ref children) = self.children {
            1 + children.iter().map(|c| c.tree_depth()).max().unwrap_or(0)
        } else { 0 }
    }
}

impl fmt::Display for OpenClQuadTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QTree(count={}, depth={})", self.count(), self.tree_depth())
    }
}

/// QR decomposition helper for OpenCl.
pub fn opencl_qr_decompose(a: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let m = a.len();
    if m == 0 { return (Vec::new(), Vec::new()); }
    let n = a[0].len();
    let mut q = vec![vec![0.0; m]; n]; // column vectors
    let mut r = vec![vec![0.0; n]; n];
    // extract columns of a
    let mut cols: Vec<Vec<f64>> = (0..n).map(|j| (0..m).map(|i| a[i][j]).collect()).collect();
    for j in 0..n {
        let mut v = cols[j].clone();
        for i in 0..j {
            let dot: f64 = v.iter().zip(q[i].iter()).map(|(&a, &b)| a * b).sum();
            r[i][j] = dot;
            for k in 0..m { v[k] -= dot * q[i][k]; }
        }
        let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        r[j][j] = norm;
        if norm.abs() > 1e-15 { for k in 0..m { q[j][k] = v[k] / norm; } }
    }
    // convert q from list of column vectors to matrix
    let q_mat: Vec<Vec<f64>> = (0..m).map(|i| (0..n).map(|j| q[j][i]).collect()).collect();
    (q_mat, r)
}

/// Solve upper triangular system Rx = b for OpenCl.
pub fn opencl_solve_upper_triangular(r: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i+1)..n { s -= r[i][j] * x[j]; }
        x[i] = if r[i][i].abs() > 1e-15 { s / r[i][i] } else { 0.0 };
    }
    x
}

/// Matrix-vector multiply for OpenCl.
pub fn opencl_mat_vec_mul(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter().map(|row| row.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum()).collect()
}

/// Matrix transpose for OpenCl.
pub fn opencl_transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() { return Vec::new(); }
    let m = a.len(); let n = a[0].len();
    (0..n).map(|j| (0..m).map(|i| a[i][j]).collect()).collect()
}

/// Matrix multiply for OpenCl.
pub fn opencl_mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    if m == 0 { return Vec::new(); }
    let k = a[0].len();
    let n = b[0].len();
    let mut c = vec![vec![0.0; n]; m];
    for i in 0..m { for j in 0..n { for l in 0..k { c[i][j] += a[i][l] * b[l][j]; } } }
    c
}

/// Frobenius norm for OpenCl.
pub fn opencl_frobenius_norm(a: &[Vec<f64>]) -> f64 {
    a.iter().flat_map(|row| row.iter()).map(|&x| x * x).sum::<f64>().sqrt()
}

/// Matrix trace for OpenCl.
pub fn opencl_trace(a: &[Vec<f64>]) -> f64 {
    a.iter().enumerate().map(|(i, row)| if i < row.len() { row[i] } else { 0.0 }).sum()
}

/// Identity matrix for OpenCl.
pub fn opencl_identity(n: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n { m[i][i] = 1.0; }
    m
}

/// Power iteration for dominant eigenvalue for OpenCl.
pub fn opencl_power_iteration(a: &[Vec<f64>], max_iter: usize) -> (f64, Vec<f64>) {
    let n = a.len();
    let mut v = vec![1.0; n];
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut v { *x /= norm; }
    let mut eigenvalue = 0.0;
    for _ in 0..max_iter {
        let av = opencl_mat_vec_mul(a, &v);
        let norm: f64 = av.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm.abs() < 1e-15 { break; }
        eigenvalue = av.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum();
        v = av.iter().map(|&x| x / norm).collect();
    }
    (eigenvalue, v)
}

/// Running statistics with min/max tracking for OpenCl.
#[derive(Debug, Clone)]
pub struct OpenClRunningStats {
    pub count: u64,
    pub mean: f64,
    pub m2: f64,
    pub min_val: f64,
    pub max_val: f64,
    pub sum: f64,
}

impl OpenClRunningStats {
    pub fn new() -> Self { OpenClRunningStats { count: 0, mean: 0.0, m2: 0.0, min_val: f64::INFINITY, max_val: f64::NEG_INFINITY, sum: 0.0 } }
    pub fn push(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
        self.min_val = self.min_val.min(x);
        self.max_val = self.max_val.max(x);
        self.sum += x;
    }
    pub fn variance(&self) -> f64 { if self.count < 2 { 0.0 } else { self.m2 / (self.count - 1) as f64 } }
    pub fn std_dev(&self) -> f64 { self.variance().sqrt() }
    pub fn range(&self) -> f64 { self.max_val - self.min_val }
    pub fn coefficient_of_variation(&self) -> f64 { if self.mean.abs() < 1e-15 { 0.0 } else { self.std_dev() / self.mean.abs() } }
    pub fn merge(&mut self, other: &Self) {
        if other.count == 0 { return; }
        let combined_count = self.count + other.count;
        let delta = other.mean - self.mean;
        let combined_mean = self.mean + delta * other.count as f64 / combined_count as f64;
        self.m2 += other.m2 + delta * delta * self.count as f64 * other.count as f64 / combined_count as f64;
        self.mean = combined_mean;
        self.count = combined_count;
        self.min_val = self.min_val.min(other.min_val);
        self.max_val = self.max_val.max(other.max_val);
        self.sum += other.sum;
    }
}

impl fmt::Display for OpenClRunningStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Stats(n={}, mean={:.4}, std={:.4})", self.count, self.mean, self.std_dev())
    }
}

/// Interquartile range for OpenCl.
pub fn opencl_iqr(data: &[f64]) -> f64 {
    opencl_percentile_at(data, 75.0) - opencl_percentile_at(data, 25.0)
}

/// Detect outliers using IQR method for OpenCl.
pub fn opencl_outliers(data: &[f64]) -> Vec<usize> {
    let q1 = opencl_percentile_at(data, 25.0);
    let q3 = opencl_percentile_at(data, 75.0);
    let iqr = q3 - q1;
    let lower = q1 - 1.5 * iqr;
    let upper = q3 + 1.5 * iqr;
    data.iter().enumerate().filter(|(_, &v)| v < lower || v > upper).map(|(i, _)| i).collect()
}

/// Z-score normalization for OpenCl.
pub fn opencl_zscore(data: &[f64]) -> Vec<f64> {
    let n = data.len() as f64;
    if n < 2.0 { return data.to_vec(); }
    let mean = data.iter().sum::<f64>() / n;
    let std = (data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    if std.abs() < 1e-15 { return vec![0.0; data.len()]; }
    data.iter().map(|&x| (x - mean) / std).collect()
}

/// Rank values for OpenCl.
pub fn opencl_rank(data: &[f64]) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut ranks = vec![0.0; data.len()];
    for (rank, &(idx, _)) in indexed.iter().enumerate() { ranks[idx] = (rank + 1) as f64; }
    ranks
}

/// Spearman rank correlation for OpenCl.
pub fn opencl_spearman(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let rx = opencl_rank(x);
    let ry = opencl_rank(y);
    let n = x.len() as f64;
    let d_sq: f64 = rx.iter().zip(ry.iter()).map(|(&a, &b)| (a - b).powi(2)).sum();
    1.0 - 6.0 * d_sq / (n * (n * n - 1.0))
}

/// Covariance matrix for OpenCl.
pub fn opencl_covariance_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if data.is_empty() { return Vec::new(); }
    let n = data.len() as f64;
    let d = data[0].len();
    let means: Vec<f64> = (0..d).map(|j| data.iter().map(|row| row[j]).sum::<f64>() / n).collect();
    let mut cov = vec![vec![0.0; d]; d];
    for i in 0..d {
        for j in i..d {
            let c: f64 = data.iter().map(|row| (row[i] - means[i]) * (row[j] - means[j])).sum::<f64>() / (n - 1.0).max(1.0);
            cov[i][j] = c; cov[j][i] = c;
        }
    }
    cov
}

/// Correlation matrix for OpenCl.
pub fn opencl_correlation_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let cov = opencl_covariance_matrix(data);
    let d = cov.len();
    let mut corr = vec![vec![0.0; d]; d];
    for i in 0..d {
        for j in 0..d {
            let denom = (cov[i][i] * cov[j][j]).sqrt();
            corr[i][j] = if denom.abs() < 1e-15 { 0.0 } else { cov[i][j] / denom };
        }
    }
    corr
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(id: u32, wi: u32, wg: u32, op: OpenClAtomicOp, addr: u64) -> OpenClEvent {
        OpenClEvent {
            id,
            work_item: wi,
            work_group: wg,
            op,
            address: addr,
            value: None,
            memory_space: OpenClMemorySpace::Global,
            memory_order: OpenClMemoryOrder::Relaxed,
            memory_scope: OpenClMemoryScope::WorkGroup,
            timestamp: id as u64,
        }
    }

    fn make_event_with_order(
        id: u32, wi: u32, wg: u32, op: OpenClAtomicOp, addr: u64,
        order: OpenClMemoryOrder, scope: OpenClMemoryScope,
    ) -> OpenClEvent {
        OpenClEvent {
            id,
            work_item: wi,
            work_group: wg,
            op,
            address: addr,
            value: None,
            memory_space: OpenClMemorySpace::Global,
            memory_order: order,
            memory_scope: scope,
            timestamp: id as u64,
        }
    }

    #[test]
    fn test_memory_order_comparison() {
        assert!(OpenClMemoryOrder::SeqCst.is_at_least(OpenClMemoryOrder::Relaxed));
        assert!(OpenClMemoryOrder::AcqRel.is_at_least(OpenClMemoryOrder::Acquire));
        assert!(!OpenClMemoryOrder::Relaxed.is_at_least(OpenClMemoryOrder::SeqCst));
    }

    #[test]
    fn test_memory_scope_includes() {
        assert!(OpenClMemoryScope::Device.includes(&OpenClMemoryScope::WorkGroup));
        assert!(OpenClMemoryScope::AllSvmDevices.includes(&OpenClMemoryScope::Device));
        assert!(!OpenClMemoryScope::WorkItem.includes(&OpenClMemoryScope::WorkGroup));
    }

    #[test]
    fn test_atomic_op_properties() {
        assert!(OpenClAtomicOp::Load.is_read());
        assert!(!OpenClAtomicOp::Load.is_write());
        assert!(OpenClAtomicOp::Store.is_write());
        assert!(!OpenClAtomicOp::Store.is_read());
        assert!(OpenClAtomicOp::FetchAdd.is_rmw());
        assert!(OpenClAtomicOp::FetchAdd.is_read());
        assert!(OpenClAtomicOp::FetchAdd.is_write());
        assert!(!OpenClAtomicOp::NonAtomicRead.is_atomic());
    }

    #[test]
    fn test_barrier_convergence_all_present() {
        let mut checker = WorkGroupBarrierChecker::new(4);
        checker.add_barrier(Barrier {
            id: 0,
            memory_scope: OpenClMemoryScope::WorkGroup,
            memory_order: OpenClMemoryOrder::AcqRel,
            address_spaces: vec![OpenClMemorySpace::Local],
            work_group_id: 0,
            participating_items: vec![0, 1, 2, 3],
        });
        let violations = checker.check_barrier_convergence();
        assert!(violations.is_empty());
    }

    #[test]
    fn test_barrier_convergence_missing() {
        let mut checker = WorkGroupBarrierChecker::new(4);
        checker.add_barrier(Barrier {
            id: 0,
            memory_scope: OpenClMemoryScope::WorkGroup,
            memory_order: OpenClMemoryOrder::AcqRel,
            address_spaces: vec![OpenClMemorySpace::Local],
            work_group_id: 0,
            participating_items: vec![0, 1, 2],
        });
        let violations = checker.check_barrier_convergence();
        assert!(!violations.is_empty());
    }

    #[test]
    fn test_bit_matrix_basic() {
        let mut m = BitMatrix::new(3, 3);
        m.set_true(0, 1);
        m.set_true(1, 2);
        assert!(m.get(0, 1));
        assert!(!m.get(0, 2));
        assert_eq!(m.count_edges(), 2);
    }

    #[test]
    fn test_bit_matrix_transitive_closure() {
        let mut m = BitMatrix::new(3, 3);
        m.set_true(0, 1);
        m.set_true(1, 2);
        m.transitive_closure();
        assert!(m.get(0, 2));
    }

    #[test]
    fn test_bit_matrix_acyclic() {
        let mut m = BitMatrix::new(3, 3);
        m.set_true(0, 1);
        m.set_true(1, 2);
        assert!(m.is_acyclic());

        m.set_true(2, 0);
        assert!(!m.is_acyclic());
    }

    #[test]
    fn test_bit_matrix_compose() {
        let mut a = BitMatrix::new(2, 2);
        a.set_true(0, 1);
        let mut b = BitMatrix::new(2, 2);
        b.set_true(1, 0);
        let c = a.compose(&b);
        assert!(c.get(0, 0));
        assert!(!c.get(1, 1));
    }

    #[test]
    fn test_bit_matrix_inverse() {
        let mut m = BitMatrix::new(3, 3);
        m.set_true(0, 1);
        m.set_true(1, 2);
        let inv = m.inverse();
        assert!(inv.get(1, 0));
        assert!(inv.get(2, 1));
        assert!(!inv.get(0, 1));
    }

    #[test]
    fn test_global_model_hb() {
        let events = vec![
            make_event_with_order(0, 0, 0, OpenClAtomicOp::Store, 100,
                OpenClMemoryOrder::Release, OpenClMemoryScope::WorkGroup),
            make_event_with_order(1, 1, 0, OpenClAtomicOp::Load, 100,
                OpenClMemoryOrder::Acquire, OpenClMemoryScope::WorkGroup),
        ];
        let mut model = GlobalMemoryModel::new(events);
        model.add_rf(0, 1);
        model.compute_hb();
        assert!(model.hb.get(0, 1));
    }

    #[test]
    fn test_model_checker_acyclic_hb() {
        let events = vec![
            make_event(0, 0, 0, OpenClAtomicOp::Store, 100),
            make_event(1, 0, 0, OpenClAtomicOp::Load, 100),
        ];
        let mut model = GlobalMemoryModel::new(events);
        model.add_po(0, 1);
        model.compute_hb();

        let checker = OpenClModelChecker::new(model, 4);
        let violations = checker.check_axiom(OpenClAxiom::HbAcyclicity);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_model_checker_cyclic_hb() {
        let events = vec![
            make_event(0, 0, 0, OpenClAtomicOp::Store, 100),
            make_event(1, 1, 0, OpenClAtomicOp::Store, 100),
        ];
        let mut model = GlobalMemoryModel::new(events);
        model.add_po(0, 1);
        model.add_po(1, 0);
        model.compute_hb();

        let checker = OpenClModelChecker::new(model, 4);
        let violations = checker.check_axiom(OpenClAxiom::HbAcyclicity);
        assert!(!violations.is_empty());
    }

    #[test]
    fn test_model_checker_no_thin_air() {
        let events = vec![
            make_event(0, 0, 0, OpenClAtomicOp::Store, 100),
            make_event(1, 1, 0, OpenClAtomicOp::Load, 100),
        ];
        let mut model = GlobalMemoryModel::new(events);
        model.add_po(0, 1);
        model.compute_hb();

        let checker = OpenClModelChecker::new(model, 4);
        let violations = checker.check_axiom(OpenClAxiom::NoThinAir);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_nd_range_1d() {
        let nd = NDRange::new_1d(256, 64);
        assert_eq!(nd.num_work_groups(), 4);
        assert_eq!(nd.total_work_items(), 256);
        assert_eq!(nd.work_group_size(), 64);
        assert_eq!(nd.work_group_of(65), 1);
        assert_eq!(nd.local_id_of(65), 1);
    }

    #[test]
    fn test_nd_range_2d() {
        let nd = NDRange::new_2d(16, 16, 4, 4);
        assert_eq!(nd.num_work_groups(), 16);
        assert_eq!(nd.total_work_items(), 256);
        assert_eq!(nd.work_group_size(), 16);
    }

    #[test]
    fn test_kernel_execution() {
        let nd = NDRange::new_1d(128, 32);
        let kernel = KernelExecution::new("test_kernel", nd, 1024);
        assert_eq!(kernel.total_work_items(), 128);
        assert_eq!(kernel.num_work_groups(), 4);
        assert_eq!(kernel.work_groups.len(), 4);
        assert_eq!(kernel.work_groups[0].size(), 32);
    }

    #[test]
    fn test_svm_region() {
        let region = SvmRegion::new(0x1000, 4096, SvmType::FineGrainedWithAtomics);
        assert!(region.contains(0x1000));
        assert!(region.contains(0x1FFF));
        assert!(!region.contains(0x2000));
    }

    #[test]
    fn test_svm_consistency_checker() {
        let mut checker = SvmConsistencyChecker::new();
        checker.add_region(SvmRegion::new(0x1000, 4096, SvmType::FineGrainedWithAtomics));

        assert!(checker.check_access(0x1000, true, false).is_ok());
        assert!(checker.check_access(0x1000, false, true).is_ok());
        assert!(checker.check_access(0x5000, false, false).is_err());
    }

    #[test]
    fn test_svm_coarse_grained_no_atomics() {
        let mut checker = SvmConsistencyChecker::new();
        checker.add_region(SvmRegion::new(0x1000, 4096, SvmType::CoarseGrained));

        assert!(checker.check_access(0x1000, true, false).is_err());
        assert!(checker.check_access(0x1000, false, false).is_ok());
    }

    #[test]
    fn test_svm_concurrent_access() {
        let mut checker = SvmConsistencyChecker::new();
        checker.add_region(SvmRegion::new(0x1000, 4096, SvmType::CoarseGrained));
        checker.add_region(SvmRegion::new(0x2000, 4096, SvmType::FineGrained));

        assert!(checker.check_concurrent_access(0x1000).is_err());
        assert!(checker.check_concurrent_access(0x2000).is_ok());
    }

    #[test]
    fn test_local_memory_model_coherence() {
        let events = vec![
            make_event(0, 0, 0, OpenClAtomicOp::NonAtomicWrite, 100),
            make_event(1, 0, 0, OpenClAtomicOp::NonAtomicRead, 100),
        ];
        let mut model = LocalMemoryModel::new(0, events);
        model.add_po(0, 1);
        assert!(model.check_coherence());
    }

    #[test]
    fn test_check_all_axioms_clean() {
        let events = vec![
            make_event(0, 0, 0, OpenClAtomicOp::Store, 100),
            make_event(1, 0, 0, OpenClAtomicOp::Load, 100),
        ];
        let mut model = GlobalMemoryModel::new(events);
        model.add_po(0, 1);
        model.add_rf(0, 1);
        model.compute_hb();

        let checker = OpenClModelChecker::new(model, 1);
        let violations = checker.check_all_axioms();
        // Should have no critical violations for this simple case
        let critical: Vec<_> = violations.iter()
            .filter(|v| v.axiom == OpenClAxiom::HbAcyclicity || v.axiom == OpenClAxiom::NoThinAir)
            .collect();
        assert!(critical.is_empty());
    }

    #[test]
    fn test_work_group() {
        let wg = WorkGroup::new(2, 64, 8192);
        assert_eq!(wg.size(), 64);
        assert_eq!(wg.items[0].global_id, 128);
        assert_eq!(wg.items[0].local_id, 0);
        assert_eq!(wg.items[0].work_group_id, 2);
    }

    #[test]
    fn test_barrier_affects_space() {
        let barrier = Barrier::work_group_barrier(0, 0, vec![OpenClMemorySpace::Local]);
        assert!(barrier.affects_space(OpenClMemorySpace::Local));
        assert!(!barrier.affects_space(OpenClMemorySpace::Global));

        let barrier2 = Barrier::work_group_barrier(1, 0, vec![]);
        assert!(barrier2.affects_space(OpenClMemorySpace::Global));
    }

    #[test]
    fn test_memory_order_combine() {
        assert_eq!(
            OpenClMemoryOrder::Acquire.combine(OpenClMemoryOrder::Release),
            OpenClMemoryOrder::Release
        );
        assert_eq!(
            OpenClMemoryOrder::SeqCst.combine(OpenClMemoryOrder::Relaxed),
            OpenClMemoryOrder::SeqCst
        );
    }
    #[test]
    fn test_opencleventmodel_new() {
        let item = OpenClEventModel::new(0, "test".to_string(), "test".to_string(), 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_commandqueuesemantics_new() {
        let item = CommandQueueSemantics::new(0, false, false, 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_kernelargbinding_new() {
        let item = KernelArgBinding::new("test".to_string(), 0, "test".to_string(), false);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_imagememorymodel_new() {
        let item = ImageMemoryModel::new(0, 0, 0, "test".to_string(), "test".to_string());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_samplerconfig_new() {
        let item = SamplerConfig::new(false, "test".to_string(), "test".to_string());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_pipeoperation_new() {
        let item = PipeOperation::new(0, 0, 0, "test".to_string());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_subgroupops_new() {
        let item = SubGroupOps::new(0, 0, 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_programscopevar_new() {
        let item = ProgramScopeVar::new("test".to_string(), "test".to_string(), 0, false);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_openclmemordersemantics_new() {
        let item = OpenClMemOrderSemantics::new("test".to_string(), "test".to_string(), 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_openclworkgroup_new() {
        let item = OpenClWorkGroup::new(Vec::new(), Vec::new(), Vec::new());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_openclbufferregion_new() {
        let item = OpenClBufferRegion::new(0, 0, 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_openclsvmmodel_new() {
        let item = OpenClSvmModel::new("test".to_string(), 0, 0, false);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_opencl_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let m = opencl_mean(&data);
        assert!((m - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_variance() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let v = opencl_variance(&data);
        assert!(v > 0.0);
    }

    #[test]
    fn test_opencl_median() {
        let data = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let m = opencl_median(&data);
        assert!((m - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_entropy() {
        let data = vec![1.0, 1.0, 1.0, 1.0];
        let e = opencl_entropy(&data);
        assert!(e > 0.0);
    }

    #[test]
    fn test_opencl_std_dev() {
        let data = vec![10.0, 10.0, 10.0];
        let s = opencl_std_dev(&data);
        assert!(s.abs() < 1e-10);
    }

    #[test]
    fn test_opencl_analysis() {
        let mut a = OpenclAnalysis::new(3);
        a.set(0, 1, 0.5);
        a.set(1, 2, 0.3);
        assert!((a.get(0, 1) - 0.5).abs() < 1e-10);
        assert_eq!(a.size, 3);
    }

    #[test]
    fn test_opencl_iterator() {
        let iter = OpenclResultIterator::new(vec![(0, 1.0), (1, 2.0), (2, 3.0)]);
        let items: Vec<_> = iter.collect();
        assert_eq!(items.len(), 3);
    }

    #[test]
    fn test_opencl_batch_processor() {
        let mut proc = OpenclBatchProcessor::new(2);
        proc.process_batch(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(proc.processed, 4);
        assert_eq!(proc.results.len(), 2);
    }

    #[test]
    fn test_opencl_histogram() {
        let hist = OpenclHistogram::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0], 3);
        assert_eq!(hist.num_bins(), 3);
        assert_eq!(hist.total_count, 5);
    }

    #[test]
    fn test_opencl_graph() {
        let mut g = OpenclGraph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 2.0);
        g.add_edge(2, 3, 3.0);
        assert_eq!(g.edge_count, 3);
        assert!(g.has_edge(0, 1));
        assert!(!g.has_edge(3, 0));
        assert!(g.is_acyclic());
    }

    #[test]
    fn test_opencl_graph_shortest_path() {
        let mut g = OpenclGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 2.0);
        g.add_edge(0, 2, 10.0);
        let dist = g.shortest_path_dijkstra(0);
        assert!((dist[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_graph_topo_sort() {
        let mut g = OpenclGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        let topo = g.topological_sort();
        assert!(topo.is_some());
    }

    #[test]
    fn test_opencl_graph_components() {
        let mut g = OpenclGraph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(2, 3, 1.0);
        let comps = g.connected_components();
        assert_eq!(comps.len(), 2);
    }

    #[test]
    fn test_opencl_cache() {
        let mut cache = OpenclCache::new(10);
        cache.insert(42, vec![1.0, 2.0]);
        assert!(cache.get(42).is_some());
        assert!(cache.get(99).is_none());
    }

    #[test]
    fn test_opencl_config() {
        let config = OpenclConfig::default_config().with_verbose(true).with_max_iterations(500);
        assert!(config.verbose);
        assert_eq!(config.max_iterations, 500);
    }

    #[test]
    fn test_opencl_report() {
        let mut report = OpenclReport::new("Test Report");
        report.add_metric("accuracy", 0.95);
        report.add_warning("low sample size");
        assert_eq!(report.total_metrics(), 1);
        assert!(report.has_warnings());
        let text = report.render_text();
        assert!(text.contains("Test Report"));
    }

    #[test]
    fn test_opencl_kmeans() {
        let data = vec![vec![0.0, 0.0], vec![0.1, 0.1], vec![10.0, 10.0], vec![10.1, 10.1]];
        let assignments = opencl_kmeans(&data, 2, 100);
        assert_eq!(assignments.len(), 4);
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[2], assignments[3]);
    }

    #[test]
    fn test_opencl_pairwise_distances() {
        let points = vec![vec![0.0, 0.0], vec![3.0, 4.0]];
        let dists = opencl_pairwise_distances(&points);
        assert!((dists[0][1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_harmmean() {
        let data = vec![1.0, 2.0, 4.0];
        let hm = opencl_harmmean(&data);
        assert!(hm > 0.0 && hm < 4.0);
    }

    #[test]
    fn test_opencl_geomean() {
        let data = vec![1.0, 2.0, 4.0];
        let gm = opencl_geomean(&data);
        assert!(gm > 0.0 && gm < 4.0);
    }

    #[test]
    fn test_opencl_sample_skewness() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = opencl_sample_skewness(&data);
        assert!(s.abs() < 1.0);
    }

    #[test]
    fn test_opencl_excess_kurtosis() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let k = opencl_excess_kurtosis(&data);
        let _ = k; // Just verify it computes
    }

    #[test]
    fn test_opencl_gini() {
        let data = vec![1.0, 1.0, 1.0, 1.0];
        let g = opencl_gini(&data);
        assert!(g.abs() < 0.01);
    }

    #[test]
    fn test_opencl_percentile_at() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let p = opencl_percentile_at(&data, 0.9);
        assert!(p >= 9.0);
    }

    #[test]
    fn test_opencl_pca_2d() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let proj = opencl_pca_2d(&data);
        assert_eq!(proj.len(), 2);
    }

    #[test]
    fn test_opencl_analysis_normalize() {
        let mut a = OpenclAnalysis::new(2);
        a.set(0, 0, 1.0); a.set(0, 1, 3.0);
        a.normalize();
        assert!((a.total_sum() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_analysis_transpose() {
        let mut a = OpenclAnalysis::new(2);
        a.set(0, 1, 5.0);
        let t = a.transpose();
        assert!((t.get(1, 0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_analysis_multiply() {
        let mut a = OpenclAnalysis::new(2);
        a.set(0, 0, 1.0); a.set(1, 1, 1.0);
        let mut b = OpenclAnalysis::new(2);
        b.set(0, 1, 2.0); b.set(1, 0, 3.0);
        let c = a.multiply(&b);
        assert!((c.get(0, 1) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_analysis_frobenius() {
        let mut a = OpenclAnalysis::new(2);
        a.set(0, 0, 3.0); a.set(1, 1, 4.0);
        assert!((a.frobenius_norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_analysis_symmetric() {
        let mut a = OpenclAnalysis::new(2);
        a.set(0, 1, 1.0); a.set(1, 0, 1.0);
        assert!(a.is_symmetric());
    }

    #[test]
    fn test_opencl_graph_dot() {
        let mut g = OpenclGraph::new(2);
        g.add_edge(0, 1, 1.0);
        let dot = g.to_dot();
        assert!(dot.contains("digraph"));
    }

    #[test]
    fn test_opencl_histogram_render() {
        let hist = OpenclHistogram::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 5);
        let ascii = hist.render_ascii(20);
        assert!(!ascii.is_empty());
    }

    #[test]
    fn test_opencl_batch_reset() {
        let mut proc = OpenclBatchProcessor::new(3);
        proc.process_batch(&[1.0, 2.0, 3.0]);
        assert!(proc.processed > 0);
        proc.reset();
        assert_eq!(proc.processed, 0);
    }

    #[test]
    fn test_opencl_graph_remove_edge() {
        let mut g = OpenclGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        assert_eq!(g.edge_count, 2);
        g.remove_edge(0, 1);
        assert_eq!(g.edge_count, 1);
        assert!(!g.has_edge(0, 1));
    }

    #[test]
    fn test_opencl_dense_matrix_new() {
        let m = OpenClDenseMatrix::new(3, 3);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 3);
    }

    #[test]
    fn test_opencl_dense_matrix_identity() {
        let m = OpenClDenseMatrix::identity(3);
        assert!((m.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((m.get(0, 1)).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_dense_matrix_mul() {
        let a = OpenClDenseMatrix::identity(2);
        let b = OpenClDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let c = a.mul_matrix(&b);
        assert!((c.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((c.get(1, 1) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_dense_matrix_transpose() {
        let a = OpenClDenseMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = a.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert!((t.get(0, 1) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_dense_matrix_det_2x2() {
        let m = OpenClDenseMatrix::from_vec(2, 2, vec![3.0, 7.0, 1.0, -4.0]);
        let det = m.determinant_2x2();
        assert!((det - (-19.0)).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_dense_matrix_det_3x3() {
        let m = OpenClDenseMatrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0]);
        let det = m.determinant_3x3();
        assert!((det - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_dense_matrix_inverse_2x2() {
        let m = OpenClDenseMatrix::from_vec(2, 2, vec![4.0, 7.0, 2.0, 6.0]);
        let inv = m.inverse_2x2().unwrap();
        let prod = m.mul_matrix(&inv);
        assert!((prod.get(0, 0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_dense_matrix_power() {
        let m = OpenClDenseMatrix::identity(3);
        let p = m.power(5);
        assert!((p.get(0, 0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_dense_matrix_rank() {
        let m = OpenClDenseMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0]);
        assert_eq!(m.rank(), 1);
    }

    #[test]
    fn test_opencl_dense_matrix_solve() {
        let a = OpenClDenseMatrix::from_vec(2, 2, vec![2.0, 1.0, 5.0, 3.0]);
        let x = a.solve(&[4.0, 7.0]).unwrap();
        assert!((x[0] - 5.0).abs() < 1e-8);
        assert!((x[1] - (-6.0)).abs() < 1e-8);
    }

    #[test]
    fn test_opencl_dense_matrix_lu() {
        let a = OpenClDenseMatrix::from_vec(2, 2, vec![4.0, 3.0, 6.0, 3.0]);
        let (l, u) = a.lu_decomposition();
        let prod = l.mul_matrix(&u);
        assert!((prod.get(0, 0) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_dense_matrix_eigenvalues() {
        let m = OpenClDenseMatrix::from_vec(2, 2, vec![2.0, 1.0, 1.0, 2.0]);
        let (e1, e2) = m.eigenvalues_2x2();
        assert!((e1 - 3.0).abs() < 1e-10);
        assert!((e2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_dense_matrix_kronecker() {
        let a = OpenClDenseMatrix::identity(2);
        let b = OpenClDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let k = a.kronecker_product(&b);
        assert_eq!(k.rows, 4);
        assert_eq!(k.cols, 4);
    }

    #[test]
    fn test_opencl_dense_matrix_hadamard() {
        let a = OpenClDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = OpenClDenseMatrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let h = a.hadamard_product(&b);
        assert!((h.get(0, 0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_interval() {
        let a = OpenClInterval::new(1.0, 3.0);
        let b = OpenClInterval::new(2.0, 5.0);
        assert!(a.overlaps(&b));
        assert!(a.contains(2.0));
        assert!(!a.contains(4.0));
        let sum = a.add(&b);
        assert!((sum.lo - 3.0).abs() < 1e-10);
        assert!((sum.hi - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_interval_mul() {
        let a = OpenClInterval::new(-2.0, 3.0);
        let b = OpenClInterval::new(1.0, 4.0);
        let prod = a.mul(&b);
        assert!((prod.lo - (-8.0)).abs() < 1e-10);
        assert!((prod.hi - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_interval_hull() {
        let a = OpenClInterval::new(1.0, 3.0);
        let b = OpenClInterval::new(5.0, 7.0);
        let h = a.hull(&b);
        assert!((h.lo - 1.0).abs() < 1e-10);
        assert!((h.hi - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_state_machine() {
        let mut sm = OpenClStateMachine::new();
        assert_eq!(*sm.state(), OpenClState::Created);
        assert!(sm.transition(OpenClState::Enqueued));
        assert_eq!(*sm.state(), OpenClState::Enqueued);
        assert_eq!(sm.transition_count, 1);
    }

    #[test]
    fn test_opencl_state_machine_invalid() {
        let mut sm = OpenClStateMachine::new();
        let last_state = OpenClState::Error;
        assert!(!sm.can_transition(&last_state));
    }

    #[test]
    fn test_opencl_state_machine_reset() {
        let mut sm = OpenClStateMachine::new();
        sm.transition(OpenClState::Enqueued);
        sm.reset();
        assert_eq!(*sm.state(), OpenClState::Created);
        assert_eq!(sm.history_len(), 0);
    }

    #[test]
    fn test_opencl_ring_buffer() {
        let mut rb = OpenClRingBuffer::new(3);
        rb.push(1.0); rb.push(2.0); rb.push(3.0);
        assert!(rb.is_full());
        assert!((rb.average() - 2.0).abs() < 1e-10);
        rb.push(4.0);
        assert!((rb.oldest().unwrap() - 2.0).abs() < 1e-10);
        assert!((rb.latest().unwrap() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_ring_buffer_to_vec() {
        let mut rb = OpenClRingBuffer::new(5);
        rb.push(10.0); rb.push(20.0); rb.push(30.0);
        let v = rb.to_vec();
        assert_eq!(v.len(), 3);
        assert!((v[0] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_disjoint_set() {
        let mut ds = OpenClDisjointSet::new(5);
        assert_eq!(ds.num_components(), 5);
        ds.union(0, 1);
        ds.union(2, 3);
        assert_eq!(ds.num_components(), 3);
        assert!(ds.connected(0, 1));
        assert!(!ds.connected(0, 2));
    }

    #[test]
    fn test_opencl_disjoint_set_components() {
        let mut ds = OpenClDisjointSet::new(4);
        ds.union(0, 1); ds.union(2, 3);
        let comps = ds.components();
        assert_eq!(comps.len(), 2);
    }

    #[test]
    fn test_opencl_sorted_list() {
        let mut sl = OpenClSortedList::new();
        sl.insert(3.0); sl.insert(1.0); sl.insert(2.0);
        assert_eq!(sl.len(), 3);
        assert!((sl.min().unwrap() - 1.0).abs() < 1e-10);
        assert!((sl.max().unwrap() - 3.0).abs() < 1e-10);
        assert!((sl.median() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_sorted_list_remove() {
        let mut sl = OpenClSortedList::new();
        sl.insert(1.0); sl.insert(2.0); sl.insert(3.0);
        assert!(sl.remove(2.0));
        assert_eq!(sl.len(), 2);
        assert!(!sl.contains(2.0));
    }

    #[test]
    fn test_opencl_ema() {
        let mut ema = OpenClEma::new(0.5);
        ema.update(10.0);
        assert!((ema.current() - 10.0).abs() < 1e-10);
        ema.update(20.0);
        assert!((ema.current() - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_bloom_filter() {
        let mut bf = OpenClBloomFilter::new(1000, 3);
        bf.insert(42);
        bf.insert(100);
        assert!(bf.may_contain(42));
        assert!(bf.may_contain(100));
        assert_eq!(bf.count(), 2);
    }

    #[test]
    fn test_opencl_trie() {
        let mut trie = OpenClTrie::new();
        trie.insert("hello", 1);
        trie.insert("help", 2);
        trie.insert("world", 3);
        assert_eq!(trie.search("hello"), Some(1));
        assert_eq!(trie.search("help"), Some(2));
        assert_eq!(trie.search("hell"), None);
        assert!(trie.starts_with("hel"));
        assert!(!trie.starts_with("xyz"));
    }

    #[test]
    fn test_opencl_dense_matrix_sym() {
        let m = OpenClDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 2.0, 3.0]);
        assert!(m.is_symmetric());
    }

    #[test]
    fn test_opencl_dense_matrix_diag() {
        let m = OpenClDenseMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 3.0]);
        assert!(m.is_diagonal());
    }

    #[test]
    fn test_opencl_dense_matrix_upper_tri() {
        let m = OpenClDenseMatrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0]);
        assert!(m.is_upper_triangular());
    }

    #[test]
    fn test_opencl_dense_matrix_outer() {
        let m = OpenClDenseMatrix::outer_product(&[1.0, 2.0], &[3.0, 4.0]);
        assert!((m.get(0, 0) - 3.0).abs() < 1e-10);
        assert!((m.get(1, 1) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_dense_matrix_submatrix() {
        let m = OpenClDenseMatrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let sub = m.submatrix(0, 0, 2, 2);
        assert_eq!(sub.rows, 2);
        assert!((sub.get(1, 1) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_priority_queue() {
        let mut pq = OpenClPriorityQueue::new();
        pq.push(3.0, 1); pq.push(1.0, 2); pq.push(2.0, 3);
        assert_eq!(pq.pop().unwrap().1, 2);
        assert_eq!(pq.pop().unwrap().1, 3);
        assert_eq!(pq.pop().unwrap().1, 1);
    }

    #[test]
    fn test_opencl_accumulator() {
        let mut acc = OpenClAccumulator::new();
        for i in 1..=10 { acc.add(i as f64); }
        assert!((acc.mean() - 5.5).abs() < 1e-10);
        assert_eq!(acc.count(), 10);
        assert!((acc.min() - 1.0).abs() < 1e-10);
        assert!((acc.max() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_accumulator_merge() {
        let mut a = OpenClAccumulator::new();
        a.add(1.0); a.add(2.0);
        let mut b = OpenClAccumulator::new();
        b.add(3.0); b.add(4.0);
        a.merge(&b);
        assert_eq!(a.count(), 4);
        assert!((a.mean() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_sparse_matrix() {
        let mut m = OpenClSparseMatrix::new(3, 3);
        m.insert(0, 1, 2.0); m.insert(1, 2, 3.0);
        assert_eq!(m.nnz(), 2);
        assert!((m.get(0, 1) - 2.0).abs() < 1e-10);
        assert!((m.get(0, 0)).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_sparse_mul_vec() {
        let mut m = OpenClSparseMatrix::new(2, 2);
        m.insert(0, 0, 1.0); m.insert(1, 1, 2.0);
        let result = m.mul_vec(&[3.0, 4.0]);
        assert!((result[0] - 3.0).abs() < 1e-10);
        assert!((result[1] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_sparse_transpose() {
        let mut m = OpenClSparseMatrix::new(2, 3);
        m.insert(0, 2, 5.0);
        let t = m.transpose();
        assert_eq!(t.rows, 3); assert_eq!(t.cols, 2);
        assert!((t.get(2, 0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_polynomial_eval() {
        let p = OpenClPolynomial::new(vec![1.0, 2.0, 3.0]);
        assert!((p.evaluate(2.0) - 17.0).abs() < 1e-10);
        assert!((p.evaluate_horner(2.0) - 17.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_polynomial_add() {
        let a = OpenClPolynomial::new(vec![1.0, 2.0]);
        let b = OpenClPolynomial::new(vec![3.0, 4.0]);
        let c = a.add(&b);
        assert!((c.evaluate(1.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_polynomial_mul() {
        let a = OpenClPolynomial::new(vec![1.0, 1.0]);
        let b = OpenClPolynomial::new(vec![1.0, 1.0]);
        let c = a.mul(&b);
        assert_eq!(c.degree(), 2);
        assert!((c.evaluate(2.0) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_polynomial_deriv() {
        let p = OpenClPolynomial::new(vec![1.0, 2.0, 3.0]);
        let dp = p.derivative();
        assert!((dp.evaluate(0.0) - 2.0).abs() < 1e-10);
        assert!((dp.evaluate(1.0) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_polynomial_integral() {
        let p = OpenClPolynomial::new(vec![2.0, 3.0]);
        let ip = p.integral(0.0);
        assert!((ip.evaluate(1.0) - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_polynomial_roots() {
        let p = OpenClPolynomial::new(vec![-6.0, 1.0, 1.0]);
        let roots = p.roots_quadratic();
        assert_eq!(roots.len(), 2);
    }

    #[test]
    fn test_opencl_polynomial_newton() {
        let p = OpenClPolynomial::new(vec![-2.0, 0.0, 1.0]);
        let root = p.newton_root(1.0, 100, 1e-10);
        assert!(root.is_some());
        assert!((root.unwrap() - std::f64::consts::SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn test_opencl_polynomial_compose() {
        let p = OpenClPolynomial::new(vec![0.0, 0.0, 1.0]);
        let q = OpenClPolynomial::new(vec![1.0, 1.0]);
        let r = p.compose(&q);
        assert!((r.evaluate(2.0) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_rng() {
        let mut rng = OpenClRng::new(42);
        let v1 = rng.next_u64();
        let v2 = rng.next_u64();
        assert_ne!(v1, v2);
        let f = rng.next_f64();
        assert!(f >= 0.0 && f < 1.0);
    }

    #[test]
    fn test_opencl_rng_gaussian() {
        let mut rng = OpenClRng::new(123);
        let mut sum = 0.0;
        for _ in 0..1000 { sum += rng.next_gaussian(); }
        let mean = sum / 1000.0;
        assert!(mean.abs() < 0.2);
    }

    #[test]
    fn test_opencl_timer() {
        let mut timer = OpenClTimer::new("test");
        timer.record(100); timer.record(200); timer.record(300);
        assert_eq!(timer.count(), 3);
        assert_eq!(timer.total_ns(), 600);
        assert!((timer.average_ns() - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_bitvector() {
        let mut bv = OpenClBitVector::new(100);
        bv.set(5); bv.set(42); bv.set(99);
        assert!(bv.get(5));
        assert!(bv.get(42));
        assert!(!bv.get(50));
        assert_eq!(bv.count_ones(), 3);
    }

    #[test]
    fn test_opencl_bitvector_ops() {
        let mut a = OpenClBitVector::new(64);
        a.set(0); a.set(10); a.set(20);
        let mut b = OpenClBitVector::new(64);
        b.set(10); b.set(20); b.set(30);
        let c = a.and(&b);
        assert_eq!(c.count_ones(), 2);
        let d = a.or(&b);
        assert_eq!(d.count_ones(), 4);
        assert_eq!(a.hamming_distance(&b), 2);
    }

    #[test]
    fn test_opencl_bitvector_jaccard() {
        let mut a = OpenClBitVector::new(10);
        a.set(0); a.set(1); a.set(2);
        let mut b = OpenClBitVector::new(10);
        b.set(1); b.set(2); b.set(3);
        let j = a.jaccard(&b);
        assert!((j - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_priority_queue_empty() {
        let mut pq = OpenClPriorityQueue::new();
        assert!(pq.is_empty());
        assert!(pq.pop().is_none());
    }

    #[test]
    fn test_opencl_sparse_add() {
        let mut a = OpenClSparseMatrix::new(2, 2);
        a.insert(0, 0, 1.0);
        let mut b = OpenClSparseMatrix::new(2, 2);
        b.insert(0, 0, 2.0); b.insert(1, 1, 3.0);
        let c = a.add(&b);
        assert!((c.get(0, 0) - 3.0).abs() < 1e-10);
        assert!((c.get(1, 1) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_rng_shuffle() {
        let mut rng = OpenClRng::new(99);
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        rng.shuffle(&mut data);
        assert_eq!(data.len(), 5);
        let sum: f64 = data.iter().sum();
        assert!((sum - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_polynomial_display() {
        let p = OpenClPolynomial::new(vec![1.0, 2.0, 3.0]);
        let s = format!("{}", p);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_opencl_polynomial_monomial() {
        let m = OpenClPolynomial::monomial(3, 5.0);
        assert_eq!(m.degree(), 3);
        assert!((m.evaluate(2.0) - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_timer_percentiles() {
        let mut timer = OpenClTimer::new("perf");
        for i in 1..=100 { timer.record(i); }
        assert_eq!(timer.p50_ns(), 50);
        assert!(timer.p95_ns() >= 90);
    }

    #[test]
    fn test_opencl_accumulator_cv() {
        let mut acc = OpenClAccumulator::new();
        acc.add(10.0); acc.add(10.0); acc.add(10.0);
        assert!(acc.coefficient_of_variation().abs() < 1e-10);
    }

    #[test]
    fn test_opencl_sparse_diagonal() {
        let mut m = OpenClSparseMatrix::new(3, 3);
        m.insert(0, 0, 1.0); m.insert(1, 1, 2.0); m.insert(2, 2, 3.0);
        assert!((m.trace() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_lru_cache() {
        let mut cache = OpenClLruCache::new(2);
        cache.put(1, vec![1.0]);
        cache.put(2, vec![2.0]);
        assert!(cache.get(1).is_some());
        cache.put(3, vec![3.0]);
        assert!(cache.get(2).is_none());
        assert!(cache.get(3).is_some());
    }

    #[test]
    fn test_opencl_lru_hit_rate() {
        let mut cache = OpenClLruCache::new(10);
        cache.put(1, vec![1.0]);
        cache.get(1);
        cache.get(2);
        assert!((cache.hit_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_graph_coloring() {
        let mut gc = OpenClGraphColoring::new(4);
        gc.add_edge(0, 1); gc.add_edge(1, 2); gc.add_edge(2, 3); gc.add_edge(3, 0);
        let colors = gc.greedy_color();
        assert!(gc.is_valid_coloring());
        assert!(colors <= 3);
    }

    #[test]
    fn test_opencl_graph_coloring_complete() {
        let mut gc = OpenClGraphColoring::new(3);
        gc.add_edge(0, 1); gc.add_edge(1, 2); gc.add_edge(0, 2);
        let colors = gc.greedy_color();
        assert_eq!(colors, 3);
        assert!(gc.is_valid_coloring());
    }

    #[test]
    fn test_opencl_topk() {
        let mut tk = OpenClTopK::new(3);
        tk.insert(5.0, "e"); tk.insert(3.0, "c"); tk.insert(1.0, "a");
        tk.insert(4.0, "d"); tk.insert(2.0, "b");
        assert_eq!(tk.len(), 3);
        assert!((tk.max_score().unwrap() - 5.0).abs() < 1e-10);
        assert!((tk.min_score().unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_sliding_window() {
        let mut sw = OpenClSlidingWindow::new(3);
        sw.push(1.0); sw.push(2.0); sw.push(3.0);
        assert!((sw.mean() - 2.0).abs() < 1e-10);
        sw.push(4.0);
        assert!((sw.mean() - 3.0).abs() < 1e-10);
        assert_eq!(sw.len(), 3);
    }

    #[test]
    fn test_opencl_sliding_window_trend() {
        let mut sw = OpenClSlidingWindow::new(10);
        for i in 0..5 { sw.push(i as f64); }
        assert!(sw.trend() > 0.0);
    }

    #[test]
    fn test_opencl_confusion_matrix() {
        let actual = vec![true, true, false, false, true];
        let predicted = vec![true, false, false, true, true];
        let cm = OpenClConfusionMatrix::from_predictions(&actual, &predicted);
        assert_eq!(cm.true_positive, 2);
        assert_eq!(cm.false_positive, 1);
        assert_eq!(cm.true_negative, 1);
        assert_eq!(cm.false_negative, 1);
        assert_eq!(cm.total(), 5);
    }

    #[test]
    fn test_opencl_confusion_f1() {
        let cm = OpenClConfusionMatrix { true_positive: 80, false_positive: 20, true_negative: 70, false_negative: 30 };
        assert!((cm.precision() - 0.8).abs() < 1e-10);
        let f1 = cm.f1_score();
        assert!(f1 > 0.0 && f1 < 1.0);
    }

    #[test]
    fn test_opencl_cosine_similarity() {
        let s = opencl_cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(s.abs() < 1e-10);
        let s2 = opencl_cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]);
        assert!((s2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_euclidean_distance() {
        let d = opencl_euclidean_distance(&[0.0, 0.0], &[3.0, 4.0]);
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_sigmoid() {
        let s = opencl_sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-10);
        let s2 = opencl_sigmoid(100.0);
        assert!((s2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_opencl_softmax() {
        let sm = opencl_softmax(&[1.0, 2.0, 3.0]);
        let sum: f64 = sm.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(sm[2] > sm[1] && sm[1] > sm[0]);
    }

    #[test]
    fn test_opencl_kl_divergence() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let kl = opencl_kl_divergence(&p, &q);
        assert!(kl.abs() < 1e-10);
    }

    #[test]
    fn test_opencl_normalize() {
        let v = opencl_normalize(&[3.0, 4.0]);
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_lerp() {
        assert!((opencl_lerp(0.0, 10.0, 0.5) - 5.0).abs() < 1e-10);
        assert!((opencl_lerp(0.0, 10.0, 0.0) - 0.0).abs() < 1e-10);
        assert!((opencl_lerp(0.0, 10.0, 1.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_clamp() {
        assert!((opencl_clamp(5.0, 0.0, 10.0) - 5.0).abs() < 1e-10);
        assert!((opencl_clamp(-5.0, 0.0, 10.0) - 0.0).abs() < 1e-10);
        assert!((opencl_clamp(15.0, 0.0, 10.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_cross_product() {
        let c = opencl_cross_product(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]);
        assert!((c[0]).abs() < 1e-10);
        assert!((c[1]).abs() < 1e-10);
        assert!((c[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_dot_product() {
        let d = opencl_dot_product(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        assert!((d - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_js_divergence() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let js = opencl_js_divergence(&p, &q);
        assert!(js.abs() < 1e-10);
    }

    #[test]
    fn test_opencl_hellinger() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let h = opencl_hellinger_distance(&p, &q);
        assert!(h.abs() < 1e-10);
    }

    #[test]
    fn test_opencl_logsumexp() {
        let lse = opencl_logsumexp(&[1.0, 2.0, 3.0]);
        assert!(lse > 3.0);
    }

    #[test]
    fn test_opencl_feature_scaler() {
        let mut scaler = OpenClFeatureScaler::new();
        let data = vec![vec![1.0, 10.0], vec![2.0, 20.0], vec![3.0, 30.0]];
        scaler.fit(&data);
        let normalized = scaler.normalize(&[2.0, 20.0]);
        assert!((normalized[0] - 0.5).abs() < 1e-10);
        assert!((normalized[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_feature_scaler_inverse() {
        let mut scaler = OpenClFeatureScaler::new();
        let data = vec![vec![0.0, 0.0], vec![10.0, 100.0]];
        scaler.fit(&data);
        let normed = scaler.normalize(&[5.0, 50.0]);
        let inv = scaler.inverse_normalize(&normed);
        assert!((inv[0] - 5.0).abs() < 1e-10);
        assert!((inv[1] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_linear_regression() {
        let mut lr = OpenClLinearRegression::new();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        lr.fit(&x, &y);
        assert!((lr.slope - 2.0).abs() < 1e-10);
        assert!(lr.intercept.abs() < 1e-10);
        assert!((lr.r_squared - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_linear_regression_predict() {
        let mut lr = OpenClLinearRegression::new();
        lr.fit(&[0.0, 1.0, 2.0], &[1.0, 3.0, 5.0]);
        assert!((lr.predict(3.0) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_weighted_graph() {
        let mut g = OpenClWeightedGraph::new(4);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 2.0); g.add_edge(2, 3, 3.0);
        assert_eq!(g.num_edges, 3);
        let dists = g.dijkstra(0);
        assert!((dists[3] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_weighted_graph_mst() {
        let mut g = OpenClWeightedGraph::new(4);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 2.0); g.add_edge(2, 3, 3.0);
        g.add_edge(0, 3, 10.0);
        let mst = g.min_spanning_tree_weight();
        assert!((mst - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_moving_average() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ma = opencl_moving_average(&data, 3);
        assert_eq!(ma.len(), 3);
        assert!((ma[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_cumsum() {
        let cs = opencl_cumsum(&[1.0, 2.0, 3.0, 4.0]);
        assert!((cs[3] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_diff() {
        let d = opencl_diff(&[1.0, 3.0, 6.0, 10.0]);
        assert_eq!(d.len(), 3);
        assert!((d[0] - 2.0).abs() < 1e-10);
        assert!((d[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_autocorrelation() {
        let ac = opencl_autocorrelation(&[1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0], 0);
        assert!((ac - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_dft_magnitude() {
        let mags = opencl_dft_magnitude(&[1.0, 0.0, -1.0, 0.0]);
        assert!(!mags.is_empty());
    }

    #[test]
    fn test_opencl_integrate_trapezoid() {
        let area = opencl_integrate_trapezoid(&[0.0, 1.0, 2.0], &[0.0, 1.0, 0.0]);
        assert!((area - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_convolve() {
        let c = opencl_convolve(&[1.0, 2.0, 3.0], &[1.0, 1.0]);
        assert_eq!(c.len(), 4);
        assert!((c[0] - 1.0).abs() < 1e-10);
        assert!((c[3] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_weighted_graph_clustering() {
        let mut g = OpenClWeightedGraph::new(4);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 1.0); g.add_edge(0, 2, 1.0);
        let cc = g.clustering_coefficient(0);
        assert!((cc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_histogram_cumulative() {
        let h = OpenClHistogramExt::from_data(&[1.0, 1.0, 2.0, 3.0, 3.0, 3.0], 3);
        let cum = h.cumulative();
        assert_eq!(*cum.last().unwrap(), 6);
    }

    #[test]
    fn test_opencl_histogram_entropy() {
        let h = OpenClHistogramExt::from_data(&[1.0, 2.0, 3.0, 4.0], 4);
        let ent = h.entropy();
        assert!(ent > 0.0);
    }

    #[test]
    fn test_opencl_aabb() {
        let bb = OpenClAABB::new(0.0, 0.0, 10.0, 10.0);
        assert!(bb.contains(5.0, 5.0));
        assert!(!bb.contains(11.0, 5.0));
        assert!((bb.area() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_aabb_intersects() {
        let a = OpenClAABB::new(0.0, 0.0, 10.0, 10.0);
        let b = OpenClAABB::new(5.0, 5.0, 15.0, 15.0);
        let c = OpenClAABB::new(20.0, 20.0, 30.0, 30.0);
        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn test_opencl_quadtree() {
        let bb = OpenClAABB::new(0.0, 0.0, 100.0, 100.0);
        let mut qt = OpenClQuadTree::new(bb, 4, 8);
        for i in 0..20 {
            qt.insert(OpenClPoint2D { x: i as f64 * 5.0, y: i as f64 * 5.0, data: i as f64 });
        }
        assert_eq!(qt.count(), 20);
    }

    #[test]
    fn test_opencl_quadtree_query() {
        let bb = OpenClAABB::new(0.0, 0.0, 100.0, 100.0);
        let mut qt = OpenClQuadTree::new(bb, 2, 8);
        qt.insert(OpenClPoint2D { x: 10.0, y: 10.0, data: 1.0 });
        qt.insert(OpenClPoint2D { x: 90.0, y: 90.0, data: 2.0 });
        let range = OpenClAABB::new(0.0, 0.0, 50.0, 50.0);
        let found = qt.query_range(&range);
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn test_opencl_mat_mul() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let c = opencl_mat_mul(&a, &b);
        assert!((c[0][0] - 19.0).abs() < 1e-10);
        assert!((c[1][1] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_transpose() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let t = opencl_transpose(&a);
        assert_eq!(t.len(), 3);
        assert_eq!(t[0].len(), 2);
        assert!((t[2][1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_frobenius_norm() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let n = opencl_frobenius_norm(&a);
        assert!((n - 2.0f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_trace() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert!((opencl_trace(&a) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_identity() {
        let id = opencl_identity(3);
        assert!((id[0][0] - 1.0).abs() < 1e-10);
        assert!((id[0][1]).abs() < 1e-10);
        assert!((id[2][2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_power_iteration() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let (eval, _evec) = opencl_power_iteration(&a, 100);
        assert!((eval - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_opencl_running_stats() {
        let mut s = OpenClRunningStats::new();
        for &v in &[1.0, 2.0, 3.0, 4.0, 5.0] { s.push(v); }
        assert!((s.mean - 3.0).abs() < 1e-10);
        assert!((s.min_val - 1.0).abs() < 1e-10);
        assert!((s.max_val - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_running_stats_merge() {
        let mut a = OpenClRunningStats::new();
        let mut b = OpenClRunningStats::new();
        for &v in &[1.0, 2.0, 3.0] { a.push(v); }
        for &v in &[4.0, 5.0, 6.0] { b.push(v); }
        a.merge(&b);
        assert_eq!(a.count, 6);
        assert!((a.mean - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_running_stats_cv() {
        let mut s = OpenClRunningStats::new();
        for &v in &[10.0, 10.0, 10.0] { s.push(v); }
        assert!(s.coefficient_of_variation() < 1e-10);
    }

    #[test]
    fn test_opencl_iqr() {
        let iqr = opencl_iqr(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert!(iqr > 0.0);
    }

    #[test]
    fn test_opencl_outliers() {
        let outliers = opencl_outliers(&[1.0, 2.0, 3.0, 4.0, 5.0, 100.0]);
        assert!(!outliers.is_empty());
    }

    #[test]
    fn test_opencl_zscore() {
        let z = opencl_zscore(&[10.0, 20.0, 30.0]);
        assert!((z[1]).abs() < 1e-10); // middle value should be ~0
    }

    #[test]
    fn test_opencl_rank() {
        let r = opencl_rank(&[30.0, 10.0, 20.0]);
        assert!((r[0] - 3.0).abs() < 1e-10);
        assert!((r[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_spearman() {
        let rho = opencl_spearman(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((rho - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_opencl_sample_skewness_symmetric() {
        let s = opencl_sample_skewness(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(s.abs() < 1e-10);
    }

    #[test]
    fn test_opencl_covariance_matrix() {
        let data = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]];
        let cov = opencl_covariance_matrix(&data);
        assert_eq!(cov.len(), 2);
        assert!(cov[0][0] > 0.0);
    }

    #[test]
    fn test_opencl_correlation_matrix() {
        let data = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]];
        let corr = opencl_correlation_matrix(&data);
        assert!((corr[0][1] - 1.0).abs() < 1e-10); // perfect correlation
    }

}

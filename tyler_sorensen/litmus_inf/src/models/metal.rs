#![allow(unused)]

use std::collections::{HashMap, HashSet, BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use serde::{Serialize, Deserialize};

// ---------------------------------------------------------------------------
// Metal Address Spaces
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetalAddressSpace {
    Device,
    Constant,
    Threadgroup,
    Thread,
}

impl fmt::Display for MetalAddressSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Device => write!(f, "device"),
            Self::Constant => write!(f, "constant"),
            Self::Threadgroup => write!(f, "threadgroup"),
            Self::Thread => write!(f, "thread"),
        }
    }
}

impl MetalAddressSpace {
    pub fn is_shared(&self) -> bool {
        matches!(self, Self::Device | Self::Threadgroup)
    }

    pub fn is_private(&self) -> bool {
        matches!(self, Self::Thread)
    }

    pub fn is_read_only(&self) -> bool {
        matches!(self, Self::Constant)
    }
}

// ---------------------------------------------------------------------------
// Metal Memory Orders
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum MetalMemoryOrder {
    Relaxed,
    Acquire,
    Release,
    AcqRel,
    SeqCst,
}

impl MetalMemoryOrder {
    pub fn is_at_least(&self, other: MetalMemoryOrder) -> bool {
        (*self as u8) >= (other as u8)
    }

    pub fn combine(self, other: MetalMemoryOrder) -> MetalMemoryOrder {
        if (self as u8) >= (other as u8) { self } else { other }
    }
}

impl fmt::Display for MetalMemoryOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Relaxed => write!(f, "memory_order_relaxed"),
            Self::Acquire => write!(f, "memory_order_acquire"),
            Self::Release => write!(f, "memory_order_release"),
            Self::AcqRel => write!(f, "memory_order_acq_rel"),
            Self::SeqCst => write!(f, "memory_order_seq_cst"),
        }
    }
}

// ---------------------------------------------------------------------------
// Threadgroup Barriers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MetalBarrierFlags {
    pub mem_none: bool,
    pub mem_device: bool,
    pub mem_threadgroup: bool,
    pub mem_texture: bool,
}

impl MetalBarrierFlags {
    pub fn none() -> Self {
        Self { mem_none: true, mem_device: false, mem_threadgroup: false, mem_texture: false }
    }

    pub fn device() -> Self {
        Self { mem_none: false, mem_device: true, mem_threadgroup: false, mem_texture: false }
    }

    pub fn threadgroup() -> Self {
        Self { mem_none: false, mem_device: false, mem_threadgroup: true, mem_texture: false }
    }

    pub fn texture() -> Self {
        Self { mem_none: false, mem_device: false, mem_threadgroup: false, mem_texture: true }
    }

    pub fn all() -> Self {
        Self { mem_none: false, mem_device: true, mem_threadgroup: true, mem_texture: true }
    }

    pub fn affects_device(&self) -> bool { self.mem_device }
    pub fn affects_threadgroup(&self) -> bool { self.mem_threadgroup }
    pub fn affects_texture(&self) -> bool { self.mem_texture }

    pub fn combine(self, other: MetalBarrierFlags) -> MetalBarrierFlags {
        MetalBarrierFlags {
            mem_none: self.mem_none && other.mem_none,
            mem_device: self.mem_device || other.mem_device,
            mem_threadgroup: self.mem_threadgroup || other.mem_threadgroup,
            mem_texture: self.mem_texture || other.mem_texture,
        }
    }
}

impl fmt::Display for MetalBarrierFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        if self.mem_none { parts.push("none"); }
        if self.mem_device { parts.push("device"); }
        if self.mem_threadgroup { parts.push("threadgroup"); }
        if self.mem_texture { parts.push("texture"); }
        write!(f, "{}", parts.join("|"))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadgroupBarrier {
    pub id: u32,
    pub flags: MetalBarrierFlags,
    pub threadgroup_id: u32,
    pub participating_threads: Vec<u32>,
    pub timestamp: u64,
}

impl ThreadgroupBarrier {
    pub fn new(id: u32, flags: MetalBarrierFlags, tg_id: u32) -> Self {
        Self {
            id,
            flags,
            threadgroup_id: tg_id,
            participating_threads: Vec::new(),
            timestamp: 0,
        }
    }

    pub fn with_threads(mut self, threads: Vec<u32>) -> Self {
        self.participating_threads = threads;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadgroupBarrierViolation {
    pub barrier_id: u32,
    pub description: String,
    pub missing_threads: Vec<u32>,
}

impl fmt::Display for ThreadgroupBarrierViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Threadgroup barrier {} violation: {}", self.barrier_id, self.description)
    }
}

#[derive(Debug, Clone)]
pub struct ThreadgroupBarrierChecker {
    pub threadgroup_size: u32,
    barriers: Vec<ThreadgroupBarrier>,
}

impl ThreadgroupBarrierChecker {
    pub fn new(threadgroup_size: u32) -> Self {
        Self { threadgroup_size, barriers: Vec::new() }
    }

    pub fn add_barrier(&mut self, barrier: ThreadgroupBarrier) {
        self.barriers.push(barrier);
    }

    pub fn check_convergence(&self) -> Vec<ThreadgroupBarrierViolation> {
        let mut violations = Vec::new();
        for barrier in &self.barriers {
            let expected: HashSet<u32> = (0..self.threadgroup_size).collect();
            let actual: HashSet<u32> = barrier.participating_threads.iter().copied().collect();

            let mut missing: Vec<u32> = expected.difference(&actual).copied().collect();
            missing.sort_unstable();
            if !missing.is_empty() {
                violations.push(ThreadgroupBarrierViolation {
                    barrier_id: barrier.id,
                    description: format!(
                        "Threads {:?} did not reach barrier in threadgroup {}",
                        missing, barrier.threadgroup_id
                    ),
                    missing_threads: missing,
                });
            }

            // Duplicate check
            let mut seen = HashSet::new();
            for &t in &barrier.participating_threads {
                if !seen.insert(t) {
                    violations.push(ThreadgroupBarrierViolation {
                        barrier_id: barrier.id,
                        description: format!(
                            "Thread {} arrived at barrier {} multiple times",
                            t, barrier.id
                        ),
                        missing_threads: Vec::new(),
                    });
                }
            }
        }
        violations
    }
}

// Helper trait since we reference it inline
trait SortedUnstable: Iterator {
    fn sorted_unstable(self) -> Vec<Self::Item>
    where
        Self: Sized,
        Self::Item: Ord,
    {
        let mut v: Vec<_> = self.collect();
        v.sort_unstable();
        v
    }
}
impl<I: Iterator> SortedUnstable for I {}

// ---------------------------------------------------------------------------
// SIMD Group Barriers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SimdgroupBarrierChecker {
    pub simd_size: u32,
    barriers: Vec<(u32, MetalBarrierFlags, Vec<u32>)>, // (id, flags, threads)
}

impl SimdgroupBarrierChecker {
    pub fn new(simd_size: u32) -> Self {
        Self { simd_size, barriers: Vec::new() }
    }

    pub fn add_barrier(&mut self, id: u32, flags: MetalBarrierFlags, threads: Vec<u32>) {
        self.barriers.push((id, flags, threads));
    }

    pub fn check(&self) -> Vec<String> {
        let mut issues = Vec::new();
        for (id, flags, threads) in &self.barriers {
            let expected: HashSet<u32> = (0..self.simd_size).collect();
            let actual: HashSet<u32> = threads.iter().copied().collect();
            let missing: Vec<u32> = expected.difference(&actual).copied().collect();
            if !missing.is_empty() {
                issues.push(format!(
                    "Simdgroup barrier {} missing threads: {:?}",
                    id, missing
                ));
            }
        }
        issues
    }
}

// ---------------------------------------------------------------------------
// Metal Fences
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetalFenceScope {
    ThreadgroupMemory,
    DeviceMemory,
    TextureMemory,
}

impl fmt::Display for MetalFenceScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ThreadgroupMemory => write!(f, "threadgroup_memory"),
            Self::DeviceMemory => write!(f, "device_memory"),
            Self::TextureMemory => write!(f, "texture_memory"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalFence {
    pub id: u32,
    pub scope: MetalFenceScope,
    pub thread_id: u32,
    pub threadgroup_id: u32,
    pub timestamp: u64,
    pub before_events: Vec<u32>,
    pub after_events: Vec<u32>,
}

impl MetalFence {
    pub fn new(id: u32, scope: MetalFenceScope, thread_id: u32, tg_id: u32) -> Self {
        Self {
            id,
            scope,
            thread_id,
            threadgroup_id: tg_id,
            timestamp: 0,
            before_events: Vec::new(),
            after_events: Vec::new(),
        }
    }

    pub fn with_timestamp(mut self, ts: u64) -> Self {
        self.timestamp = ts;
        self
    }
}

#[derive(Debug, Clone)]
pub struct FenceChecker {
    fences: Vec<MetalFence>,
}

impl FenceChecker {
    pub fn new() -> Self {
        Self { fences: Vec::new() }
    }

    pub fn add_fence(&mut self, fence: MetalFence) {
        self.fences.push(fence);
    }

    pub fn check_ordering(&self) -> Vec<String> {
        let mut issues = Vec::new();
        // Check that fences actually order operations across the intended scope
        for fence in &self.fences {
            if fence.before_events.is_empty() && fence.after_events.is_empty() {
                issues.push(format!(
                    "Fence {} has no ordered events - may be redundant",
                    fence.id
                ));
            }
        }
        issues
    }

    pub fn build_ordering(&self) -> Vec<(u32, u32)> {
        let mut edges = Vec::new();
        for fence in &self.fences {
            for &before in &fence.before_events {
                for &after in &fence.after_events {
                    edges.push((before, after));
                }
            }
        }
        edges
    }

    pub fn fences_for_thread(&self, thread_id: u32) -> Vec<&MetalFence> {
        self.fences.iter().filter(|f| f.thread_id == thread_id).collect()
    }

    pub fn fences_for_scope(&self, scope: MetalFenceScope) -> Vec<&MetalFence> {
        self.fences.iter().filter(|f| f.scope == scope).collect()
    }
}

// ---------------------------------------------------------------------------
// Texture Access & Raster Order Groups
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextureAccess {
    pub id: u32,
    pub texture_id: u32,
    pub coordinate: (u32, u32),
    pub mip_level: u32,
    pub is_write: bool,
    pub is_sample: bool,
    pub thread_id: u32,
    pub timestamp: u64,
    pub raster_order_group: Option<u32>,
}

impl TextureAccess {
    pub fn read(id: u32, texture_id: u32, coord: (u32, u32), thread_id: u32) -> Self {
        Self {
            id,
            texture_id,
            coordinate: coord,
            mip_level: 0,
            is_write: false,
            is_sample: false,
            thread_id,
            timestamp: 0,
            raster_order_group: None,
        }
    }

    pub fn write(id: u32, texture_id: u32, coord: (u32, u32), thread_id: u32) -> Self {
        Self {
            id,
            texture_id,
            coordinate: coord,
            mip_level: 0,
            is_write: true,
            is_sample: false,
            thread_id,
            timestamp: 0,
            raster_order_group: None,
        }
    }

    pub fn with_rog(mut self, group: u32) -> Self {
        self.raster_order_group = Some(group);
        self
    }

    pub fn with_timestamp(mut self, ts: u64) -> Self {
        self.timestamp = ts;
        self
    }
}

#[derive(Debug, Clone)]
pub struct TextureMemoryModel {
    accesses: Vec<TextureAccess>,
    rog_ordering: HashMap<u32, Vec<u32>>, // ROG -> ordered access IDs
}

impl TextureMemoryModel {
    pub fn new() -> Self {
        Self {
            accesses: Vec::new(),
            rog_ordering: HashMap::new(),
        }
    }

    pub fn from_accesses(accesses: Vec<TextureAccess>) -> Self {
        let mut model = Self { accesses, rog_ordering: HashMap::new() };
        model.compute_rog_ordering();
        model
    }

    fn compute_rog_ordering(&mut self) {
        let mut by_rog: HashMap<u32, Vec<&TextureAccess>> = HashMap::new();
        for access in &self.accesses {
            if let Some(rog) = access.raster_order_group {
                by_rog.entry(rog).or_default().push(access);
            }
        }

        for (rog, mut accesses) in by_rog {
            accesses.sort_by_key(|a| a.timestamp);
            let ordered_ids: Vec<u32> = accesses.iter().map(|a| a.id).collect();
            self.rog_ordering.insert(rog, ordered_ids);
        }
    }

    pub fn check_races(&self) -> Vec<TextureRace> {
        let mut races = Vec::new();
        // Accesses to the same texture+coordinate without ROG ordering
        let mut by_location: HashMap<(u32, (u32, u32)), Vec<&TextureAccess>> = HashMap::new();
        for access in &self.accesses {
            by_location
                .entry((access.texture_id, access.coordinate))
                .or_default()
                .push(access);
        }

        for (loc, accesses) in &by_location {
            for i in 0..accesses.len() {
                for j in (i + 1)..accesses.len() {
                    let a1 = accesses[i];
                    let a2 = accesses[j];
                    if a1.thread_id == a2.thread_id { continue; }
                    if !a1.is_write && !a2.is_write { continue; }

                    // Check if ROG provides ordering
                    let ordered = self.rog_ordered(a1.id, a2.id);
                    if !ordered {
                        races.push(TextureRace {
                            access1_id: a1.id,
                            access2_id: a2.id,
                            texture_id: loc.0,
                            coordinate: loc.1,
                            description: format!(
                                "Texture race at texture {} coord ({},{}) between threads {} and {}",
                                loc.0, loc.1.0, loc.1.1, a1.thread_id, a2.thread_id
                            ),
                        });
                    }
                }
            }
        }
        races
    }

    fn rog_ordered(&self, id1: u32, id2: u32) -> bool {
        for (_, order) in &self.rog_ordering {
            let pos1 = order.iter().position(|&id| id == id1);
            let pos2 = order.iter().position(|&id| id == id2);
            if let (Some(p1), Some(p2)) = (pos1, pos2) {
                return true;
            }
        }
        false
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextureRace {
    pub access1_id: u32,
    pub access2_id: u32,
    pub texture_id: u32,
    pub coordinate: (u32, u32),
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RasterOrderGroup {
    pub id: u32,
    pub name: String,
    pub access_ids: Vec<u32>,
}

impl RasterOrderGroup {
    pub fn new(id: u32, name: impl Into<String>) -> Self {
        Self { id, name: name.into(), access_ids: Vec::new() }
    }

    pub fn add_access(&mut self, access_id: u32) {
        self.access_ids.push(access_id);
    }
}

// ---------------------------------------------------------------------------
// Metal Atomics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetalAtomicOp {
    Load,
    Store,
    Exchange,
    CompareExchangeWeak,
    FetchAdd,
    FetchSub,
    FetchAnd,
    FetchOr,
    FetchXor,
    FetchMin,
    FetchMax,
}

impl MetalAtomicOp {
    pub fn is_read(&self) -> bool {
        !matches!(self, Self::Store)
    }

    pub fn is_write(&self) -> bool {
        !matches!(self, Self::Load)
    }

    pub fn is_rmw(&self) -> bool {
        matches!(self,
            Self::Exchange | Self::CompareExchangeWeak |
            Self::FetchAdd | Self::FetchSub |
            Self::FetchAnd | Self::FetchOr | Self::FetchXor |
            Self::FetchMin | Self::FetchMax
        )
    }
}

impl fmt::Display for MetalAtomicOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load => write!(f, "atomic_load"),
            Self::Store => write!(f, "atomic_store"),
            Self::Exchange => write!(f, "atomic_exchange"),
            Self::CompareExchangeWeak => write!(f, "atomic_compare_exchange_weak"),
            Self::FetchAdd => write!(f, "atomic_fetch_add"),
            Self::FetchSub => write!(f, "atomic_fetch_sub"),
            Self::FetchAnd => write!(f, "atomic_fetch_and"),
            Self::FetchOr => write!(f, "atomic_fetch_or"),
            Self::FetchXor => write!(f, "atomic_fetch_xor"),
            Self::FetchMin => write!(f, "atomic_fetch_min"),
            Self::FetchMax => write!(f, "atomic_fetch_max"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MetalAtomic {
    pub id: u32,
    pub op: MetalAtomicOp,
    pub order: MetalMemoryOrder,
    pub address_space: MetalAddressSpace,
    pub address: u64,
    pub thread_id: u32,
    pub threadgroup_id: u32,
    pub simdgroup_id: u32,
    pub value: Option<u64>,
    pub timestamp: u64,
}

impl MetalAtomic {
    pub fn new(id: u32, op: MetalAtomicOp, order: MetalMemoryOrder, addr: u64) -> Self {
        Self {
            id,
            op,
            order,
            address_space: MetalAddressSpace::Device,
            address: addr,
            thread_id: 0,
            threadgroup_id: 0,
            simdgroup_id: 0,
            value: None,
            timestamp: 0,
        }
    }

    pub fn with_thread(mut self, tid: u32, tgid: u32, sgid: u32) -> Self {
        self.thread_id = tid;
        self.threadgroup_id = tgid;
        self.simdgroup_id = sgid;
        self
    }

    pub fn with_address_space(mut self, space: MetalAddressSpace) -> Self {
        self.address_space = space;
        self
    }

    pub fn with_value(mut self, v: u64) -> Self {
        self.value = Some(v);
        self
    }

    pub fn with_timestamp(mut self, ts: u64) -> Self {
        self.timestamp = ts;
        self
    }

    pub fn is_release(&self) -> bool {
        matches!(self.order, MetalMemoryOrder::Release | MetalMemoryOrder::AcqRel | MetalMemoryOrder::SeqCst)
    }

    pub fn is_acquire(&self) -> bool {
        matches!(self.order, MetalMemoryOrder::Acquire | MetalMemoryOrder::AcqRel | MetalMemoryOrder::SeqCst)
    }
}

#[derive(Debug, Clone)]
pub struct AtomicChecker {
    atomics: Vec<MetalAtomic>,
}

impl AtomicChecker {
    pub fn new() -> Self {
        Self { atomics: Vec::new() }
    }

    pub fn add(&mut self, atomic: MetalAtomic) {
        self.atomics.push(atomic);
    }

    pub fn check_ordering_consistency(&self) -> Vec<String> {
        let mut issues = Vec::new();
        // Check that release stores are paired with acquire loads
        let releases: Vec<&MetalAtomic> = self.atomics.iter()
            .filter(|a| a.is_release() && a.op.is_write())
            .collect();
        let acquires: Vec<&MetalAtomic> = self.atomics.iter()
            .filter(|a| a.is_acquire() && a.op.is_read())
            .collect();

        for rel in &releases {
            let matching_acquire = acquires.iter().any(|acq| {
                acq.address == rel.address && acq.timestamp > rel.timestamp
            });
            if !matching_acquire {
                issues.push(format!(
                    "Release store {} at addr 0x{:x} has no matching acquire load",
                    rel.id, rel.address
                ));
            }
        }
        issues
    }

    pub fn check_address_space_validity(&self) -> Vec<String> {
        let mut issues = Vec::new();
        for a in &self.atomics {
            match a.address_space {
                MetalAddressSpace::Constant => {
                    if a.op.is_write() {
                        issues.push(format!(
                            "Atomic {} writes to constant address space (invalid)",
                            a.id
                        ));
                    }
                }
                MetalAddressSpace::Thread => {
                    issues.push(format!(
                        "Atomic {} on thread-local memory is unnecessary",
                        a.id
                    ));
                }
                _ => {}
            }
        }
        issues
    }

    pub fn find_races(&self) -> Vec<(u32, u32)> {
        let mut races = Vec::new();
        for i in 0..self.atomics.len() {
            for j in (i + 1)..self.atomics.len() {
                let a1 = &self.atomics[i];
                let a2 = &self.atomics[j];
                if a1.thread_id == a2.thread_id { continue; }
                if a1.address != a2.address { continue; }
                if !a1.op.is_write() && !a2.op.is_write() { continue; }
                // Both are atomic, so this is not a race in the traditional sense,
                // but report if orders are both relaxed
                if a1.order == MetalMemoryOrder::Relaxed && a2.order == MetalMemoryOrder::Relaxed {
                    if a1.op.is_write() && a2.op.is_read() || a1.op.is_read() && a2.op.is_write() {
                        races.push((a1.id, a2.id));
                    }
                }
            }
        }
        races
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

    pub fn inverse(&self) -> BitMatrix {
        let mut result = BitMatrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j][i] = self.data[i][j];
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

    pub fn count_edges(&self) -> usize {
        self.data.iter().flat_map(|row| row.iter()).filter(|&&b| b).count()
    }
}

// ---------------------------------------------------------------------------
// Metal Axioms & Model Checker
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetalAxiom {
    HbAcyclicity,
    CoherenceWW,
    CoherenceWR,
    CoherenceRW,
    CoherenceRR,
    Atomicity,
    NoThinAir,
    SeqCstConsistency,
    ThreadgroupCoherence,
    DeviceCoherence,
    FenceOrdering,
    RasterOrderConsistency,
}

impl fmt::Display for MetalAxiom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HbAcyclicity => write!(f, "hb-acyclicity"),
            Self::CoherenceWW => write!(f, "coh-ww"),
            Self::CoherenceWR => write!(f, "coh-wr"),
            Self::CoherenceRW => write!(f, "coh-rw"),
            Self::CoherenceRR => write!(f, "coh-rr"),
            Self::Atomicity => write!(f, "atomicity"),
            Self::NoThinAir => write!(f, "no-thin-air"),
            Self::SeqCstConsistency => write!(f, "sc-consistency"),
            Self::ThreadgroupCoherence => write!(f, "tg-coherence"),
            Self::DeviceCoherence => write!(f, "device-coherence"),
            Self::FenceOrdering => write!(f, "fence-ordering"),
            Self::RasterOrderConsistency => write!(f, "rog-consistency"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalViolation {
    pub axiom: MetalAxiom,
    pub description: String,
    pub events_involved: Vec<u32>,
}

impl fmt::Display for MetalViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.axiom, self.description)
    }
}

#[derive(Debug, Clone)]
pub struct MetalEvent {
    pub id: u32,
    pub thread_id: u32,
    pub threadgroup_id: u32,
    pub simdgroup_id: u32,
    pub op: MetalAtomicOp,
    pub address: u64,
    pub address_space: MetalAddressSpace,
    pub memory_order: MetalMemoryOrder,
    pub value: Option<u64>,
    pub timestamp: u64,
    pub is_non_atomic: bool,
}

#[derive(Debug, Clone)]
pub struct MetalModelChecker {
    events: Vec<MetalEvent>,
    event_index: HashMap<u32, usize>,
    po: BitMatrix,
    rf: BitMatrix,
    mo: BitMatrix,
    hb: BitMatrix,
    fences: Vec<MetalFence>,
    barriers: Vec<ThreadgroupBarrier>,
    threadgroup_size: u32,
}

impl MetalModelChecker {
    pub fn new(events: Vec<MetalEvent>, threadgroup_size: u32) -> Self {
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
            fences: Vec::new(),
            barriers: Vec::new(),
            threadgroup_size,
        }
    }

    pub fn add_po(&mut self, from: u32, to: u32) {
        if let (Some(&fi), Some(&ti)) = (self.event_index.get(&from), self.event_index.get(&to)) {
            self.po.set_true(fi, ti);
        }
    }

    pub fn add_rf(&mut self, write_id: u32, read_id: u32) {
        if let (Some(&wi), Some(&ri)) = (self.event_index.get(&write_id), self.event_index.get(&read_id)) {
            self.rf.set_true(wi, ri);
        }
    }

    pub fn add_mo(&mut self, w1: u32, w2: u32) {
        if let (Some(&fi), Some(&si)) = (self.event_index.get(&w1), self.event_index.get(&w2)) {
            self.mo.set_true(fi, si);
        }
    }

    pub fn add_fence(&mut self, fence: MetalFence) {
        // Add fence-induced ordering to po
        for &before in &fence.before_events {
            for &after in &fence.after_events {
                if let (Some(&bi), Some(&ai)) = (self.event_index.get(&before), self.event_index.get(&after)) {
                    self.po.set_true(bi, ai);
                }
            }
        }
        self.fences.push(fence);
    }

    pub fn add_barrier(&mut self, barrier: ThreadgroupBarrier) {
        self.barriers.push(barrier);
    }

    pub fn compute_hb(&mut self) {
        let n = self.events.len();
        // sw: release-acquire pairs linked by rf
        let mut sw = BitMatrix::new(n, n);
        for i in 0..n {
            for j in 0..n {
                if self.rf.get(i, j) {
                    let ei = &self.events[i];
                    let ej = &self.events[j];
                    let is_release = matches!(ei.memory_order,
                        MetalMemoryOrder::Release | MetalMemoryOrder::AcqRel | MetalMemoryOrder::SeqCst);
                    let is_acquire = matches!(ej.memory_order,
                        MetalMemoryOrder::Acquire | MetalMemoryOrder::AcqRel | MetalMemoryOrder::SeqCst);
                    if is_release && is_acquire {
                        sw.set_true(i, j);
                    }
                }
            }
        }

        // Add barrier-induced synchronization
        for barrier in &self.barriers {
            // All events before the barrier hb all events after in the same threadgroup
            // Simplified: use timestamp-based ordering
        }

        self.hb = self.po.union(&sw);
        self.hb.transitive_closure();
    }

    pub fn check_axiom(&self, axiom: MetalAxiom) -> Vec<MetalViolation> {
        match axiom {
            MetalAxiom::HbAcyclicity => self.check_hb_acyclicity(),
            MetalAxiom::CoherenceWW => self.check_coherence_ww(),
            MetalAxiom::CoherenceWR => self.check_coherence_wr(),
            MetalAxiom::CoherenceRW => self.check_coherence_rw(),
            MetalAxiom::CoherenceRR => self.check_coherence_rr(),
            MetalAxiom::Atomicity => self.check_atomicity(),
            MetalAxiom::NoThinAir => self.check_no_thin_air(),
            MetalAxiom::SeqCstConsistency => self.check_seq_cst(),
            MetalAxiom::ThreadgroupCoherence => self.check_threadgroup_coherence(),
            MetalAxiom::DeviceCoherence => self.check_device_coherence(),
            MetalAxiom::FenceOrdering => self.check_fence_ordering(),
            MetalAxiom::RasterOrderConsistency => Vec::new(), // checked separately via TextureMemoryModel
        }
    }

    pub fn check_all_axioms(&self) -> Vec<MetalViolation> {
        let axioms = [
            MetalAxiom::HbAcyclicity,
            MetalAxiom::CoherenceWW,
            MetalAxiom::CoherenceWR,
            MetalAxiom::CoherenceRW,
            MetalAxiom::CoherenceRR,
            MetalAxiom::Atomicity,
            MetalAxiom::NoThinAir,
            MetalAxiom::SeqCstConsistency,
            MetalAxiom::ThreadgroupCoherence,
            MetalAxiom::DeviceCoherence,
            MetalAxiom::FenceOrdering,
        ];
        let mut violations = Vec::new();
        for axiom in &axioms {
            violations.extend(self.check_axiom(*axiom));
        }
        violations
    }

    fn check_hb_acyclicity(&self) -> Vec<MetalViolation> {
        if !self.hb.is_acyclic() {
            vec![MetalViolation {
                axiom: MetalAxiom::HbAcyclicity,
                description: "Happens-before relation contains a cycle".to_string(),
                events_involved: Vec::new(),
            }]
        } else {
            Vec::new()
        }
    }

    fn check_coherence_ww(&self) -> Vec<MetalViolation> {
        let hb_mo = self.hb.compose(&self.mo);
        if !hb_mo.is_irreflexive() {
            vec![MetalViolation {
                axiom: MetalAxiom::CoherenceWW,
                description: "Write-write coherence violated: hb;mo cycle".to_string(),
                events_involved: Vec::new(),
            }]
        } else {
            Vec::new()
        }
    }

    fn check_coherence_wr(&self) -> Vec<MetalViolation> {
        let hb_rf = self.hb.compose(&self.rf);
        if !hb_rf.is_irreflexive() {
            vec![MetalViolation {
                axiom: MetalAxiom::CoherenceWR,
                description: "Write-read coherence violated".to_string(),
                events_involved: Vec::new(),
            }]
        } else {
            Vec::new()
        }
    }

    fn check_coherence_rw(&self) -> Vec<MetalViolation> {
        let rf_inv = self.rf.inverse();
        let fr = rf_inv.compose(&self.mo);
        let hb_check = fr.compose(&self.hb);
        if !hb_check.is_irreflexive() {
            vec![MetalViolation {
                axiom: MetalAxiom::CoherenceRW,
                description: "Read-write coherence violated".to_string(),
                events_involved: Vec::new(),
            }]
        } else {
            Vec::new()
        }
    }

    fn check_coherence_rr(&self) -> Vec<MetalViolation> {
        let rf_inv = self.rf.inverse();
        let fr = rf_inv.compose(&self.mo);
        let fr_rf = fr.compose(&self.rf);
        if !fr_rf.is_irreflexive() {
            vec![MetalViolation {
                axiom: MetalAxiom::CoherenceRR,
                description: "Read-read coherence violated".to_string(),
                events_involved: Vec::new(),
            }]
        } else {
            Vec::new()
        }
    }

    fn check_atomicity(&self) -> Vec<MetalViolation> {
        // RMW atomicity: no write can intervene in mo between the read and write of an RMW
        Vec::new()
    }

    fn check_no_thin_air(&self) -> Vec<MetalViolation> {
        let hb_rf = self.hb.union(&self.rf);
        if !hb_rf.is_acyclic() {
            vec![MetalViolation {
                axiom: MetalAxiom::NoThinAir,
                description: "Potential out-of-thin-air value: hb∪rf has a cycle".to_string(),
                events_involved: Vec::new(),
            }]
        } else {
            Vec::new()
        }
    }

    fn check_seq_cst(&self) -> Vec<MetalViolation> {
        let n = self.events.len();
        let sc_indices: Vec<usize> = (0..n)
            .filter(|&i| self.events[i].memory_order == MetalMemoryOrder::SeqCst)
            .collect();

        if sc_indices.len() > 1 {
            let mut sc_hb = BitMatrix::new(n, n);
            for &i in &sc_indices {
                for &j in &sc_indices {
                    if self.hb.get(i, j) {
                        sc_hb.set_true(i, j);
                    }
                }
            }
            if !sc_hb.is_acyclic() {
                return vec![MetalViolation {
                    axiom: MetalAxiom::SeqCstConsistency,
                    description: "SeqCst ordering inconsistent with happens-before".to_string(),
                    events_involved: sc_indices.iter().map(|&i| self.events[i].id).collect(),
                }];
            }
        }
        Vec::new()
    }

    fn check_threadgroup_coherence(&self) -> Vec<MetalViolation> {
        let mut violations = Vec::new();
        // Within a threadgroup, threadgroup memory must be coherent after barriers
        let tg_events: HashMap<u32, Vec<usize>> = {
            let mut map: HashMap<u32, Vec<usize>> = HashMap::new();
            for (i, e) in self.events.iter().enumerate() {
                if e.address_space == MetalAddressSpace::Threadgroup {
                    map.entry(e.threadgroup_id).or_default().push(i);
                }
            }
            map
        };

        for (tg_id, indices) in &tg_events {
            // Check that po restricted to threadgroup memory is respected
            let mut tg_po = BitMatrix::new(indices.len(), indices.len());
            for (li, &gi) in indices.iter().enumerate() {
                for (lj, &gj) in indices.iter().enumerate() {
                    if self.po.get(gi, gj) {
                        tg_po.set_true(li, lj);
                    }
                }
            }
            if !tg_po.is_acyclic() {
                violations.push(MetalViolation {
                    axiom: MetalAxiom::ThreadgroupCoherence,
                    description: format!("Threadgroup {} memory coherence violated", tg_id),
                    events_involved: indices.iter().map(|&i| self.events[i].id).collect(),
                });
            }
        }
        violations
    }

    fn check_device_coherence(&self) -> Vec<MetalViolation> {
        // Device memory coherence: mo;hb must be acyclic
        let mo_hb = self.mo.compose(&self.hb);
        if !mo_hb.is_irreflexive() {
            vec![MetalViolation {
                axiom: MetalAxiom::DeviceCoherence,
                description: "Device memory coherence violated: mo;hb cycle".to_string(),
                events_involved: Vec::new(),
            }]
        } else {
            Vec::new()
        }
    }

    fn check_fence_ordering(&self) -> Vec<MetalViolation> {
        // Fences should not introduce cycles
        // Already incorporated into po, so check po acyclicity
        if !self.po.is_acyclic() {
            vec![MetalViolation {
                axiom: MetalAxiom::FenceOrdering,
                description: "Fence ordering introduces program order cycle".to_string(),
                events_involved: Vec::new(),
            }]
        } else {
            Vec::new()
        }
    }
}

// ---------------------------------------------------------------------------
// GPU Execution Structures
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalGrid {
    pub size: Vec<u32>,
    pub dimensions: usize,
}

impl MetalGrid {
    pub fn new_1d(size: u32) -> Self {
        Self { size: vec![size], dimensions: 1 }
    }

    pub fn new_2d(x: u32, y: u32) -> Self {
        Self { size: vec![x, y], dimensions: 2 }
    }

    pub fn new_3d(x: u32, y: u32, z: u32) -> Self {
        Self { size: vec![x, y, z], dimensions: 3 }
    }

    pub fn total_threads(&self) -> u32 {
        self.size.iter().product()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Threadgroup {
    pub id: u32,
    pub size: u32,
    pub threads: Vec<MetalThread>,
    pub simd_groups: Vec<SimdGroup>,
    pub threadgroup_memory_size: u64,
}

impl Threadgroup {
    pub fn new(id: u32, size: u32, simd_width: u32, tg_mem: u64) -> Self {
        let threads: Vec<MetalThread> = (0..size).map(|i| MetalThread {
            id: id * size + i,
            index_in_threadgroup: i,
            threadgroup_id: id,
            simdgroup_index: i / simd_width,
            index_in_simdgroup: i % simd_width,
        }).collect();

        let num_simd = (size + simd_width - 1) / simd_width;
        let simd_groups: Vec<SimdGroup> = (0..num_simd).map(|sg| {
            let start = sg * simd_width;
            let end = (start + simd_width).min(size);
            let thread_ids: Vec<u32> = (start..end).map(|i| id * size + i).collect();
            SimdGroup {
                id: sg,
                threadgroup_id: id,
                thread_ids,
                width: simd_width,
            }
        }).collect();

        Self {
            id,
            size,
            threads,
            simd_groups,
            threadgroup_memory_size: tg_mem,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalThread {
    pub id: u32,
    pub index_in_threadgroup: u32,
    pub threadgroup_id: u32,
    pub simdgroup_index: u32,
    pub index_in_simdgroup: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimdGroup {
    pub id: u32,
    pub threadgroup_id: u32,
    pub thread_ids: Vec<u32>,
    pub width: u32,
}

impl SimdGroup {
    pub fn size(&self) -> usize {
        self.thread_ids.len()
    }

    pub fn contains_thread(&self, tid: u32) -> bool {
        self.thread_ids.contains(&tid)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalKernel {
    pub name: String,
    pub grid_size: MetalGrid,
    pub threadgroup_size: u32,
    pub simd_width: u32,
    pub threadgroups: Vec<Threadgroup>,
    pub threadgroup_memory_length: u64,
}

impl MetalKernel {
    pub fn new(
        name: impl Into<String>,
        grid_size: MetalGrid,
        tg_size: u32,
        simd_width: u32,
        tg_mem: u64,
    ) -> Self {
        let total = grid_size.total_threads();
        let num_tg = (total + tg_size - 1) / tg_size;
        let threadgroups: Vec<Threadgroup> = (0..num_tg)
            .map(|i| Threadgroup::new(i, tg_size, simd_width, tg_mem))
            .collect();
        Self {
            name: name.into(),
            grid_size,
            threadgroup_size: tg_size,
            simd_width,
            threadgroups,
            threadgroup_memory_length: tg_mem,
        }
    }

    pub fn total_threads(&self) -> u32 {
        self.grid_size.total_threads()
    }

    pub fn num_threadgroups(&self) -> u32 {
        self.threadgroups.len() as u32
    }

    pub fn num_simdgroups_per_threadgroup(&self) -> u32 {
        (self.threadgroup_size + self.simd_width - 1) / self.simd_width
    }
}

// ---------------------------------------------------------------------------
// Indirect Command Buffers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ICBCommandKind {
    Draw,
    DrawIndexed,
    Dispatch,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ICBCommand {
    pub id: u32,
    pub kind: ICBCommandKind,
    pub index: u32,
    pub resources: Vec<u64>,
    pub threadgroup_count: Option<u32>,
}

impl ICBCommand {
    pub fn dispatch(id: u32, index: u32, tg_count: u32) -> Self {
        Self {
            id,
            kind: ICBCommandKind::Dispatch,
            index,
            resources: Vec::new(),
            threadgroup_count: Some(tg_count),
        }
    }

    pub fn draw(id: u32, index: u32) -> Self {
        Self {
            id,
            kind: ICBCommandKind::Draw,
            index,
            resources: Vec::new(),
            threadgroup_count: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndirectCommandBuffer {
    pub id: u32,
    pub commands: Vec<ICBCommand>,
    pub max_commands: u32,
    pub inherit_pipeline_state: bool,
}

impl IndirectCommandBuffer {
    pub fn new(id: u32, max_commands: u32) -> Self {
        Self {
            id,
            commands: Vec::new(),
            max_commands,
            inherit_pipeline_state: false,
        }
    }

    pub fn add_command(&mut self, cmd: ICBCommand) -> Result<(), String> {
        if self.commands.len() >= self.max_commands as usize {
            return Err(format!("ICB {} is full (max {} commands)", self.id, self.max_commands));
        }
        self.commands.push(cmd);
        Ok(())
    }

    pub fn reset(&mut self) {
        self.commands.clear();
    }
}

#[derive(Debug, Clone)]
pub struct ICBOrdering {
    command_order: Vec<u32>,
    dependencies: HashMap<u32, Vec<u32>>,
}

impl ICBOrdering {
    pub fn new() -> Self {
        Self { command_order: Vec::new(), dependencies: HashMap::new() }
    }

    pub fn add_command(&mut self, cmd_id: u32) {
        self.command_order.push(cmd_id);
    }

    pub fn add_dependency(&mut self, cmd_id: u32, depends_on: u32) {
        self.dependencies.entry(cmd_id).or_default().push(depends_on);
    }

    pub fn topological_order(&self) -> Option<Vec<u32>> {
        let mut in_degree: HashMap<u32, usize> = HashMap::new();
        for &cmd in &self.command_order {
            in_degree.entry(cmd).or_insert(0);
        }
        for (cmd, deps) in &self.dependencies {
            for &dep in deps {
                *in_degree.entry(*cmd).or_insert(0) += 1;
                in_degree.entry(dep).or_insert(0);
            }
        }

        let mut queue: VecDeque<u32> = in_degree.iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&id, _)| id)
            .collect();
        let mut result = Vec::new();

        // Build reverse dep map
        let mut rdeps: HashMap<u32, Vec<u32>> = HashMap::new();
        for (cmd, deps) in &self.dependencies {
            for &dep in deps {
                rdeps.entry(dep).or_default().push(*cmd);
            }
        }

        while let Some(cmd) = queue.pop_front() {
            result.push(cmd);
            if let Some(dependents) = rdeps.get(&cmd) {
                for &dependent in dependents {
                    if let Some(deg) = in_degree.get_mut(&dependent) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(dependent);
                        }
                    }
                }
            }
        }

        if result.len() == in_degree.len() {
            Some(result)
        } else {
            None // cycle
        }
    }

    pub fn is_valid_order(&self) -> bool {
        self.topological_order().is_some()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// ===== Extended Metal Operations =====

#[derive(Debug, Clone)]
pub struct MetalResourceBinding {
    pub index: u32,
    pub resource_type: String,
    pub access: String,
    pub bound: bool,
}

impl MetalResourceBinding {
    pub fn new(index: u32, resource_type: String, access: String, bound: bool) -> Self {
        MetalResourceBinding { index, resource_type, access, bound }
    }

    pub fn get_index(&self) -> u32 {
        self.index
    }

    pub fn get_resource_type(&self) -> &str {
        &self.resource_type
    }

    pub fn get_access(&self) -> &str {
        &self.access
    }

    pub fn get_bound(&self) -> bool {
        self.bound
    }

    pub fn with_index(mut self, v: u32) -> Self {
        self.index = v; self
    }

    pub fn with_resource_type(mut self, v: impl Into<String>) -> Self {
        self.resource_type = v.into(); self
    }

    pub fn with_access(mut self, v: impl Into<String>) -> Self {
        self.access = v.into(); self
    }

    pub fn with_bound(mut self, v: bool) -> Self {
        self.bound = v; self
    }

}

impl fmt::Display for MetalResourceBinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MetalResourceBinding({:?})", self.index)
    }
}

#[derive(Debug, Clone)]
pub struct MetalResourceBindingBuilder {
    index: u32,
    resource_type: String,
    access: String,
    bound: bool,
}

impl MetalResourceBindingBuilder {
    pub fn new() -> Self {
        MetalResourceBindingBuilder {
            index: 0,
            resource_type: String::new(),
            access: String::new(),
            bound: false,
        }
    }

    pub fn index(mut self, v: u32) -> Self { self.index = v; self }
    pub fn resource_type(mut self, v: impl Into<String>) -> Self { self.resource_type = v.into(); self }
    pub fn access(mut self, v: impl Into<String>) -> Self { self.access = v.into(); self }
    pub fn bound(mut self, v: bool) -> Self { self.bound = v; self }
}

#[derive(Debug, Clone)]
pub struct ArgumentBufferSemantics {
    pub tier: u32,
    pub max_entries: u32,
    pub encoded_length: u64,
}

impl ArgumentBufferSemantics {
    pub fn new(tier: u32, max_entries: u32, encoded_length: u64) -> Self {
        ArgumentBufferSemantics { tier, max_entries, encoded_length }
    }

    pub fn get_tier(&self) -> u32 {
        self.tier
    }

    pub fn get_max_entries(&self) -> u32 {
        self.max_entries
    }

    pub fn get_encoded_length(&self) -> u64 {
        self.encoded_length
    }

    pub fn with_tier(mut self, v: u32) -> Self {
        self.tier = v; self
    }

    pub fn with_max_entries(mut self, v: u32) -> Self {
        self.max_entries = v; self
    }

    pub fn with_encoded_length(mut self, v: u64) -> Self {
        self.encoded_length = v; self
    }

}

impl fmt::Display for ArgumentBufferSemantics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ArgumentBufferSemantics({:?})", self.tier)
    }
}

#[derive(Debug, Clone)]
pub struct ArgumentBufferSemanticsBuilder {
    tier: u32,
    max_entries: u32,
    encoded_length: u64,
}

impl ArgumentBufferSemanticsBuilder {
    pub fn new() -> Self {
        ArgumentBufferSemanticsBuilder {
            tier: 0,
            max_entries: 0,
            encoded_length: 0,
        }
    }

    pub fn tier(mut self, v: u32) -> Self { self.tier = v; self }
    pub fn max_entries(mut self, v: u32) -> Self { self.max_entries = v; self }
    pub fn encoded_length(mut self, v: u64) -> Self { self.encoded_length = v; self }
}

#[derive(Debug, Clone)]
pub struct HeapAllocationModel {
    pub heap_size: u64,
    pub used_size: u64,
    pub fragmentation: f64,
    pub resource_count: u32,
}

impl HeapAllocationModel {
    pub fn new(heap_size: u64, used_size: u64, fragmentation: f64, resource_count: u32) -> Self {
        HeapAllocationModel { heap_size, used_size, fragmentation, resource_count }
    }

    pub fn get_heap_size(&self) -> u64 {
        self.heap_size
    }

    pub fn get_used_size(&self) -> u64 {
        self.used_size
    }

    pub fn get_fragmentation(&self) -> f64 {
        self.fragmentation
    }

    pub fn get_resource_count(&self) -> u32 {
        self.resource_count
    }

    pub fn with_heap_size(mut self, v: u64) -> Self {
        self.heap_size = v; self
    }

    pub fn with_used_size(mut self, v: u64) -> Self {
        self.used_size = v; self
    }

    pub fn with_fragmentation(mut self, v: f64) -> Self {
        self.fragmentation = v; self
    }

    pub fn with_resource_count(mut self, v: u32) -> Self {
        self.resource_count = v; self
    }

}

impl fmt::Display for HeapAllocationModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HeapAllocationModel({:?})", self.heap_size)
    }
}

#[derive(Debug, Clone)]
pub struct HeapAllocationModelBuilder {
    heap_size: u64,
    used_size: u64,
    fragmentation: f64,
    resource_count: u32,
}

impl HeapAllocationModelBuilder {
    pub fn new() -> Self {
        HeapAllocationModelBuilder {
            heap_size: 0,
            used_size: 0,
            fragmentation: 0.0,
            resource_count: 0,
        }
    }

    pub fn heap_size(mut self, v: u64) -> Self { self.heap_size = v; self }
    pub fn used_size(mut self, v: u64) -> Self { self.used_size = v; self }
    pub fn fragmentation(mut self, v: f64) -> Self { self.fragmentation = v; self }
    pub fn resource_count(mut self, v: u32) -> Self { self.resource_count = v; self }
}

#[derive(Debug, Clone)]
pub struct ResidencyTracker {
    pub resident_resources: Vec<u64>,
    pub total_size: u64,
    pub eviction_count: u32,
}

impl ResidencyTracker {
    pub fn new(resident_resources: Vec<u64>, total_size: u64, eviction_count: u32) -> Self {
        ResidencyTracker { resident_resources, total_size, eviction_count }
    }

    pub fn resident_resources_len(&self) -> usize {
        self.resident_resources.len()
    }

    pub fn resident_resources_is_empty(&self) -> bool {
        self.resident_resources.is_empty()
    }

    pub fn get_total_size(&self) -> u64 {
        self.total_size
    }

    pub fn get_eviction_count(&self) -> u32 {
        self.eviction_count
    }

    pub fn with_total_size(mut self, v: u64) -> Self {
        self.total_size = v; self
    }

    pub fn with_eviction_count(mut self, v: u32) -> Self {
        self.eviction_count = v; self
    }

}

impl fmt::Display for ResidencyTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ResidencyTracker({:?})", self.resident_resources)
    }
}

#[derive(Debug, Clone)]
pub struct ResidencyTrackerBuilder {
    resident_resources: Vec<u64>,
    total_size: u64,
    eviction_count: u32,
}

impl ResidencyTrackerBuilder {
    pub fn new() -> Self {
        ResidencyTrackerBuilder {
            resident_resources: Vec::new(),
            total_size: 0,
            eviction_count: 0,
        }
    }

    pub fn resident_resources(mut self, v: Vec<u64>) -> Self { self.resident_resources = v; self }
    pub fn total_size(mut self, v: u64) -> Self { self.total_size = v; self }
    pub fn eviction_count(mut self, v: u32) -> Self { self.eviction_count = v; self }
}

#[derive(Debug, Clone)]
pub struct GpuFamilyCapabilities {
    pub family: String,
    pub max_threads_per_group: u32,
    pub max_buffer_length: u64,
    pub supports_ray_tracing: bool,
}

impl GpuFamilyCapabilities {
    pub fn new(family: String, max_threads_per_group: u32, max_buffer_length: u64, supports_ray_tracing: bool) -> Self {
        GpuFamilyCapabilities { family, max_threads_per_group, max_buffer_length, supports_ray_tracing }
    }

    pub fn get_family(&self) -> &str {
        &self.family
    }

    pub fn get_max_threads_per_group(&self) -> u32 {
        self.max_threads_per_group
    }

    pub fn get_max_buffer_length(&self) -> u64 {
        self.max_buffer_length
    }

    pub fn get_supports_ray_tracing(&self) -> bool {
        self.supports_ray_tracing
    }

    pub fn with_family(mut self, v: impl Into<String>) -> Self {
        self.family = v.into(); self
    }

    pub fn with_max_threads_per_group(mut self, v: u32) -> Self {
        self.max_threads_per_group = v; self
    }

    pub fn with_max_buffer_length(mut self, v: u64) -> Self {
        self.max_buffer_length = v; self
    }

    pub fn with_supports_ray_tracing(mut self, v: bool) -> Self {
        self.supports_ray_tracing = v; self
    }

}

impl fmt::Display for GpuFamilyCapabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GpuFamilyCapabilities({:?})", self.family)
    }
}

#[derive(Debug, Clone)]
pub struct GpuFamilyCapabilitiesBuilder {
    family: String,
    max_threads_per_group: u32,
    max_buffer_length: u64,
    supports_ray_tracing: bool,
}

impl GpuFamilyCapabilitiesBuilder {
    pub fn new() -> Self {
        GpuFamilyCapabilitiesBuilder {
            family: String::new(),
            max_threads_per_group: 0,
            max_buffer_length: 0,
            supports_ray_tracing: false,
        }
    }

    pub fn family(mut self, v: impl Into<String>) -> Self { self.family = v.into(); self }
    pub fn max_threads_per_group(mut self, v: u32) -> Self { self.max_threads_per_group = v; self }
    pub fn max_buffer_length(mut self, v: u64) -> Self { self.max_buffer_length = v; self }
    pub fn supports_ray_tracing(mut self, v: bool) -> Self { self.supports_ray_tracing = v; self }
}

#[derive(Debug, Clone)]
pub struct TileShadingModel {
    pub tile_width: u32,
    pub tile_height: u32,
    pub threadgroup_memory: u32,
    pub imageblock_size: u32,
}

impl TileShadingModel {
    pub fn new(tile_width: u32, tile_height: u32, threadgroup_memory: u32, imageblock_size: u32) -> Self {
        TileShadingModel { tile_width, tile_height, threadgroup_memory, imageblock_size }
    }

    pub fn get_tile_width(&self) -> u32 {
        self.tile_width
    }

    pub fn get_tile_height(&self) -> u32 {
        self.tile_height
    }

    pub fn get_threadgroup_memory(&self) -> u32 {
        self.threadgroup_memory
    }

    pub fn get_imageblock_size(&self) -> u32 {
        self.imageblock_size
    }

    pub fn with_tile_width(mut self, v: u32) -> Self {
        self.tile_width = v; self
    }

    pub fn with_tile_height(mut self, v: u32) -> Self {
        self.tile_height = v; self
    }

    pub fn with_threadgroup_memory(mut self, v: u32) -> Self {
        self.threadgroup_memory = v; self
    }

    pub fn with_imageblock_size(mut self, v: u32) -> Self {
        self.imageblock_size = v; self
    }

}

impl fmt::Display for TileShadingModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TileShadingModel({:?})", self.tile_width)
    }
}

#[derive(Debug, Clone)]
pub struct TileShadingModelBuilder {
    tile_width: u32,
    tile_height: u32,
    threadgroup_memory: u32,
    imageblock_size: u32,
}

impl TileShadingModelBuilder {
    pub fn new() -> Self {
        TileShadingModelBuilder {
            tile_width: 0,
            tile_height: 0,
            threadgroup_memory: 0,
            imageblock_size: 0,
        }
    }

    pub fn tile_width(mut self, v: u32) -> Self { self.tile_width = v; self }
    pub fn tile_height(mut self, v: u32) -> Self { self.tile_height = v; self }
    pub fn threadgroup_memory(mut self, v: u32) -> Self { self.threadgroup_memory = v; self }
    pub fn imageblock_size(mut self, v: u32) -> Self { self.imageblock_size = v; self }
}

#[derive(Debug, Clone)]
pub struct MeshShaderMemoryModel {
    pub object_threads: u32,
    pub mesh_threads: u32,
    pub payload_size: u32,
    pub max_vertices: u32,
}

impl MeshShaderMemoryModel {
    pub fn new(object_threads: u32, mesh_threads: u32, payload_size: u32, max_vertices: u32) -> Self {
        MeshShaderMemoryModel { object_threads, mesh_threads, payload_size, max_vertices }
    }

    pub fn get_object_threads(&self) -> u32 {
        self.object_threads
    }

    pub fn get_mesh_threads(&self) -> u32 {
        self.mesh_threads
    }

    pub fn get_payload_size(&self) -> u32 {
        self.payload_size
    }

    pub fn get_max_vertices(&self) -> u32 {
        self.max_vertices
    }

    pub fn with_object_threads(mut self, v: u32) -> Self {
        self.object_threads = v; self
    }

    pub fn with_mesh_threads(mut self, v: u32) -> Self {
        self.mesh_threads = v; self
    }

    pub fn with_payload_size(mut self, v: u32) -> Self {
        self.payload_size = v; self
    }

    pub fn with_max_vertices(mut self, v: u32) -> Self {
        self.max_vertices = v; self
    }

}

impl fmt::Display for MeshShaderMemoryModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MeshShaderMemoryModel({:?})", self.object_threads)
    }
}

#[derive(Debug, Clone)]
pub struct MeshShaderMemoryModelBuilder {
    object_threads: u32,
    mesh_threads: u32,
    payload_size: u32,
    max_vertices: u32,
}

impl MeshShaderMemoryModelBuilder {
    pub fn new() -> Self {
        MeshShaderMemoryModelBuilder {
            object_threads: 0,
            mesh_threads: 0,
            payload_size: 0,
            max_vertices: 0,
        }
    }

    pub fn object_threads(mut self, v: u32) -> Self { self.object_threads = v; self }
    pub fn mesh_threads(mut self, v: u32) -> Self { self.mesh_threads = v; self }
    pub fn payload_size(mut self, v: u32) -> Self { self.payload_size = v; self }
    pub fn max_vertices(mut self, v: u32) -> Self { self.max_vertices = v; self }
}

#[derive(Debug, Clone)]
pub struct RayTracingIntersection {
    pub primitive_id: u32,
    pub instance_id: u32,
    pub distance: f64,
    pub barycentrics: Vec<f64>,
}

impl RayTracingIntersection {
    pub fn new(primitive_id: u32, instance_id: u32, distance: f64, barycentrics: Vec<f64>) -> Self {
        RayTracingIntersection { primitive_id, instance_id, distance, barycentrics }
    }

    pub fn get_primitive_id(&self) -> u32 {
        self.primitive_id
    }

    pub fn get_instance_id(&self) -> u32 {
        self.instance_id
    }

    pub fn get_distance(&self) -> f64 {
        self.distance
    }

    pub fn barycentrics_len(&self) -> usize {
        self.barycentrics.len()
    }

    pub fn barycentrics_is_empty(&self) -> bool {
        self.barycentrics.is_empty()
    }

    pub fn with_primitive_id(mut self, v: u32) -> Self {
        self.primitive_id = v; self
    }

    pub fn with_instance_id(mut self, v: u32) -> Self {
        self.instance_id = v; self
    }

    pub fn with_distance(mut self, v: f64) -> Self {
        self.distance = v; self
    }

}

impl fmt::Display for RayTracingIntersection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RayTracingIntersection({:?})", self.primitive_id)
    }
}

#[derive(Debug, Clone)]
pub struct RayTracingIntersectionBuilder {
    primitive_id: u32,
    instance_id: u32,
    distance: f64,
    barycentrics: Vec<f64>,
}

impl RayTracingIntersectionBuilder {
    pub fn new() -> Self {
        RayTracingIntersectionBuilder {
            primitive_id: 0,
            instance_id: 0,
            distance: 0.0,
            barycentrics: Vec::new(),
        }
    }

    pub fn primitive_id(mut self, v: u32) -> Self { self.primitive_id = v; self }
    pub fn instance_id(mut self, v: u32) -> Self { self.instance_id = v; self }
    pub fn distance(mut self, v: f64) -> Self { self.distance = v; self }
    pub fn barycentrics(mut self, v: Vec<f64>) -> Self { self.barycentrics = v; self }
}

#[derive(Debug, Clone)]
pub struct MetalFenceModel {
    pub fence_id: u32,
    pub scope: String,
    pub signaled: bool,
}

impl MetalFenceModel {
    pub fn new(fence_id: u32, scope: String, signaled: bool) -> Self {
        MetalFenceModel { fence_id, scope, signaled }
    }

    pub fn get_fence_id(&self) -> u32 {
        self.fence_id
    }

    pub fn get_scope(&self) -> &str {
        &self.scope
    }

    pub fn get_signaled(&self) -> bool {
        self.signaled
    }

    pub fn with_fence_id(mut self, v: u32) -> Self {
        self.fence_id = v; self
    }

    pub fn with_scope(mut self, v: impl Into<String>) -> Self {
        self.scope = v.into(); self
    }

    pub fn with_signaled(mut self, v: bool) -> Self {
        self.signaled = v; self
    }

}

impl fmt::Display for MetalFenceModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MetalFenceModel({:?})", self.fence_id)
    }
}

#[derive(Debug, Clone)]
pub struct MetalFenceModelBuilder {
    fence_id: u32,
    scope: String,
    signaled: bool,
}

impl MetalFenceModelBuilder {
    pub fn new() -> Self {
        MetalFenceModelBuilder {
            fence_id: 0,
            scope: String::new(),
            signaled: false,
        }
    }

    pub fn fence_id(mut self, v: u32) -> Self { self.fence_id = v; self }
    pub fn scope(mut self, v: impl Into<String>) -> Self { self.scope = v.into(); self }
    pub fn signaled(mut self, v: bool) -> Self { self.signaled = v; self }
}

#[derive(Debug, Clone)]
pub struct MetalEventModel {
    pub event_id: u64,
    pub value: u64,
    pub signaled: bool,
}

impl MetalEventModel {
    pub fn new(event_id: u64, value: u64, signaled: bool) -> Self {
        MetalEventModel { event_id, value, signaled }
    }

    pub fn get_event_id(&self) -> u64 {
        self.event_id
    }

    pub fn get_value(&self) -> u64 {
        self.value
    }

    pub fn get_signaled(&self) -> bool {
        self.signaled
    }

    pub fn with_event_id(mut self, v: u64) -> Self {
        self.event_id = v; self
    }

    pub fn with_value(mut self, v: u64) -> Self {
        self.value = v; self
    }

    pub fn with_signaled(mut self, v: bool) -> Self {
        self.signaled = v; self
    }

}

impl fmt::Display for MetalEventModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MetalEventModel({:?})", self.event_id)
    }
}

#[derive(Debug, Clone)]
pub struct MetalEventModelBuilder {
    event_id: u64,
    value: u64,
    signaled: bool,
}

impl MetalEventModelBuilder {
    pub fn new() -> Self {
        MetalEventModelBuilder {
            event_id: 0,
            value: 0,
            signaled: false,
        }
    }

    pub fn event_id(mut self, v: u64) -> Self { self.event_id = v; self }
    pub fn value(mut self, v: u64) -> Self { self.value = v; self }
    pub fn signaled(mut self, v: bool) -> Self { self.signaled = v; self }
}

#[derive(Debug, Clone)]
pub struct IndirectCommandBufferExt {
    pub max_commands: u32,
    pub command_count: u32,
    pub inherited_buffers: Vec<u32>,
    pub commands: Vec<ICBCommand>,
}

impl IndirectCommandBufferExt {
    pub fn new(max_commands: u32, command_count: u32, inherited_buffers: Vec<u32>) -> Self {
        IndirectCommandBufferExt { max_commands, command_count, inherited_buffers, commands: Vec::new() }
    }

    pub fn from_capacity(max_commands: u32, capacity: u32) -> Self {
        IndirectCommandBufferExt {
            max_commands,
            command_count: capacity,
            inherited_buffers: Vec::new(),
            commands: Vec::new(),
        }
    }

    pub fn add_command(&mut self, cmd: ICBCommand) -> Result<(), String> {
        if self.commands.len() >= self.command_count as usize {
            return Err(format!("ICB is full (max {} commands)", self.command_count));
        }
        self.commands.push(cmd);
        Ok(())
    }

    pub fn get_max_commands(&self) -> u32 {
        self.max_commands
    }

    pub fn get_command_count(&self) -> u32 {
        self.command_count
    }

    pub fn inherited_buffers_len(&self) -> usize {
        self.inherited_buffers.len()
    }

    pub fn inherited_buffers_is_empty(&self) -> bool {
        self.inherited_buffers.is_empty()
    }

    pub fn with_max_commands(mut self, v: u32) -> Self {
        self.max_commands = v; self
    }

    pub fn with_command_count(mut self, v: u32) -> Self {
        self.command_count = v; self
    }

}

impl fmt::Display for IndirectCommandBufferExt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IndirectCommandBufferExt({:?})", self.max_commands)
    }
}

#[derive(Debug, Clone)]
pub struct IndirectCommandBufferExtBuilder {
    max_commands: u32,
    command_count: u32,
    inherited_buffers: Vec<u32>,
}

impl IndirectCommandBufferExtBuilder {
    pub fn new() -> Self {
        IndirectCommandBufferExtBuilder {
            max_commands: 0,
            command_count: 0,
            inherited_buffers: Vec::new(),
        }
    }

    pub fn max_commands(mut self, v: u32) -> Self { self.max_commands = v; self }
    pub fn command_count(mut self, v: u32) -> Self { self.command_count = v; self }
    pub fn inherited_buffers(mut self, v: Vec<u32>) -> Self { self.inherited_buffers = v; self }
}

#[derive(Debug, Clone)]
pub struct MetalTextureDescriptor {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub pixel_format: String,
    pub usage: String,
}

impl MetalTextureDescriptor {
    pub fn new(width: u32, height: u32, depth: u32, pixel_format: String, usage: String) -> Self {
        MetalTextureDescriptor { width, height, depth, pixel_format, usage }
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

    pub fn get_pixel_format(&self) -> &str {
        &self.pixel_format
    }

    pub fn get_usage(&self) -> &str {
        &self.usage
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

    pub fn with_pixel_format(mut self, v: impl Into<String>) -> Self {
        self.pixel_format = v.into(); self
    }

    pub fn with_usage(mut self, v: impl Into<String>) -> Self {
        self.usage = v.into(); self
    }

}

impl fmt::Display for MetalTextureDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MetalTextureDescriptor({:?})", self.width)
    }
}

#[derive(Debug, Clone)]
pub struct MetalTextureDescriptorBuilder {
    width: u32,
    height: u32,
    depth: u32,
    pixel_format: String,
    usage: String,
}

impl MetalTextureDescriptorBuilder {
    pub fn new() -> Self {
        MetalTextureDescriptorBuilder {
            width: 0,
            height: 0,
            depth: 0,
            pixel_format: String::new(),
            usage: String::new(),
        }
    }

    pub fn width(mut self, v: u32) -> Self { self.width = v; self }
    pub fn height(mut self, v: u32) -> Self { self.height = v; self }
    pub fn depth(mut self, v: u32) -> Self { self.depth = v; self }
    pub fn pixel_format(mut self, v: impl Into<String>) -> Self { self.pixel_format = v.into(); self }
    pub fn usage(mut self, v: impl Into<String>) -> Self { self.usage = v.into(); self }
}

#[derive(Debug, Clone)]
pub struct MetalAnalysis {
    pub data: Vec<Vec<f64>>,
    pub size: usize,
    pub computed: bool,
    pub label: String,
    pub threshold: f64,
}

impl MetalAnalysis {
    pub fn new(size: usize) -> Self {
        let data = vec![vec![0.0; size]; size];
        MetalAnalysis { data, size, computed: false, label: "Metal".to_string(), threshold: 0.01 }
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

impl fmt::Display for MetalAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MetalAnalysis({:?})", self.data)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MetalStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Skipped,
}

impl fmt::Display for MetalStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetalStatus::Pending => write!(f, "pending"),
            MetalStatus::InProgress => write!(f, "inprogress"),
            MetalStatus::Completed => write!(f, "completed"),
            MetalStatus::Failed => write!(f, "failed"),
            MetalStatus::Skipped => write!(f, "skipped"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MetalPriority {
    Critical,
    High,
    Medium,
    Low,
    None,
}

impl fmt::Display for MetalPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetalPriority::Critical => write!(f, "critical"),
            MetalPriority::High => write!(f, "high"),
            MetalPriority::Medium => write!(f, "medium"),
            MetalPriority::Low => write!(f, "low"),
            MetalPriority::None => write!(f, "none"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MetalMode {
    Strict,
    Relaxed,
    Permissive,
    Custom,
}

impl fmt::Display for MetalMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetalMode::Strict => write!(f, "strict"),
            MetalMode::Relaxed => write!(f, "relaxed"),
            MetalMode::Permissive => write!(f, "permissive"),
            MetalMode::Custom => write!(f, "custom"),
        }
    }
}

pub fn metal_mean(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    data.iter().sum::<f64>() / data.len() as f64
}

pub fn metal_variance(data: &[f64]) -> f64 {
    if data.len() < 2 { return 0.0; }
    let mean = metal_mean(data);
    data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (data.len() - 1) as f64
}

pub fn metal_std_dev(data: &[f64]) -> f64 {
    metal_variance(data).sqrt()
}

pub fn metal_median(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n % 2 == 0 { (sorted[n/2 - 1] + sorted[n/2]) / 2.0 } else { sorted[n/2] }
}

/// Percentile calculator for Metal.
pub fn metal_percentile_at(data: &[f64], p: f64) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = p / 100.0 * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi { sorted[lo] }
    else { sorted[lo] * (hi as f64 - idx) + sorted[hi] * (idx - lo as f64) }
}

pub fn metal_entropy(data: &[f64]) -> f64 {
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

pub fn metal_gini(data: &[f64]) -> f64 {
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

pub fn metal_covariance(data: &[f64]) -> f64 {
    if data.len() < 4 { return 0.0; }
    let n = data.len() / 2;
    let x: Vec<f64> = data[..n].to_vec();
    let y: Vec<f64> = data[n..2*n].to_vec();
    let mx = metal_mean(&x);
    let my = metal_mean(&y);
    x.iter().zip(y.iter()).map(|(&xi, &yi)| (xi - mx) * (yi - my)).sum::<f64>() / (n - 1) as f64
}

pub fn metal_correlation(data: &[f64]) -> f64 {
    if data.len() < 4 { return 0.0; }
    let n = data.len() / 2;
    let cov = metal_covariance(data);
    let sx = metal_std_dev(&data[..n]);
    let sy = metal_std_dev(&data[n..2*n]);
    if sx * sy == 0.0 { 0.0 } else { cov / (sx * sy) }
}

pub fn metal_excess_kurtosis(data: &[f64]) -> f64 {
    if data.len() < 4 { return 0.0; }
    let m = metal_mean(data);
    let s = metal_std_dev(data);
    if s == 0.0 { return 0.0; }
    let n = data.len() as f64;
    let k = data.iter().map(|&x| ((x - m) / s).powi(4)).sum::<f64>() / n;
    k - 3.0
}

pub fn metal_sample_skewness(data: &[f64]) -> f64 {
    if data.len() < 3 { return 0.0; }
    let m = metal_mean(data);
    let s = metal_std_dev(data);
    if s == 0.0 { return 0.0; }
    let n = data.len() as f64;
    data.iter().map(|&x| ((x - m) / s).powi(3)).sum::<f64>() / n
}

pub fn metal_harmmean(data: &[f64]) -> f64 {
    if data.is_empty() || data.iter().any(|&x| x <= 0.0) { return 0.0; }
    let n = data.len() as f64;
    n / data.iter().map(|&x| 1.0 / x).sum::<f64>()
}

pub fn metal_geomean(data: &[f64]) -> f64 {
    if data.is_empty() || data.iter().any(|&x| x <= 0.0) { return 0.0; }
    let n = data.len() as f64;
    (data.iter().map(|&x| x.ln()).sum::<f64>() / n).exp()
}

/// Iterator over metal analysis results.
#[derive(Debug, Clone)]
pub struct MetalResultIterator {
    items: Vec<(usize, f64)>,
    position: usize,
}

impl MetalResultIterator {
    pub fn new(items: Vec<(usize, f64)>) -> Self {
        MetalResultIterator { items, position: 0 }
    }
    pub fn remaining(&self) -> usize { self.items.len() - self.position }
}

impl Iterator for MetalResultIterator {
    type Item = (usize, f64);
    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.items.len() {
            let item = self.items[self.position];
            self.position += 1;
            Some(item)
        } else { None }
    }
}

/// Convert MetalResourceBinding description to a summary string.
pub fn metalresourcebinding_to_summary(item: &MetalResourceBinding) -> String {
    format!("MetalResourceBinding: {:?}", item)
}

/// Convert ArgumentBufferSemantics description to a summary string.
pub fn argumentbuffersemantics_to_summary(item: &ArgumentBufferSemantics) -> String {
    format!("ArgumentBufferSemantics: {:?}", item)
}

/// Convert HeapAllocationModel description to a summary string.
pub fn heapallocationmodel_to_summary(item: &HeapAllocationModel) -> String {
    format!("HeapAllocationModel: {:?}", item)
}

/// Convert ResidencyTracker description to a summary string.
pub fn residencytracker_to_summary(item: &ResidencyTracker) -> String {
    format!("ResidencyTracker: {:?}", item)
}

/// Convert GpuFamilyCapabilities description to a summary string.
pub fn gpufamilycapabilities_to_summary(item: &GpuFamilyCapabilities) -> String {
    format!("GpuFamilyCapabilities: {:?}", item)
}

/// Convert TileShadingModel description to a summary string.
pub fn tileshadingmodel_to_summary(item: &TileShadingModel) -> String {
    format!("TileShadingModel: {:?}", item)
}

/// Convert MeshShaderMemoryModel description to a summary string.
pub fn meshshadermemorymodel_to_summary(item: &MeshShaderMemoryModel) -> String {
    format!("MeshShaderMemoryModel: {:?}", item)
}

/// Convert RayTracingIntersection description to a summary string.
pub fn raytracingintersection_to_summary(item: &RayTracingIntersection) -> String {
    format!("RayTracingIntersection: {:?}", item)
}

/// Convert MetalFenceModel description to a summary string.
pub fn metalfencemodel_to_summary(item: &MetalFenceModel) -> String {
    format!("MetalFenceModel: {:?}", item)
}

/// Convert MetalEventModel description to a summary string.
pub fn metaleventmodel_to_summary(item: &MetalEventModel) -> String {
    format!("MetalEventModel: {:?}", item)
}

/// Convert IndirectCommandBufferExt description to a summary string.
pub fn indirectcommandbuffer_to_summary(item: &IndirectCommandBufferExt) -> String {
    format!("IndirectCommandBufferExt: {:?}", item)
}

/// Batch processor for metal operations.
#[derive(Debug, Clone)]
pub struct MetalBatchProcessor {
    pub batch_size: usize,
    pub processed: usize,
    pub errors: Vec<String>,
    pub results: Vec<f64>,
}

impl MetalBatchProcessor {
    pub fn new(batch_size: usize) -> Self {
        MetalBatchProcessor { batch_size, processed: 0, errors: Vec::new(), results: Vec::new() }
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

impl fmt::Display for MetalBatchProcessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MetalBatch(processed={}, errors={})", self.processed, self.errors.len())
    }
}

/// Detailed report for metal analysis.
#[derive(Debug, Clone)]
pub struct MetalReport {
    pub title: String,
    pub sections: Vec<(String, Vec<String>)>,
    pub metrics: Vec<(String, f64)>,
    pub warnings: Vec<String>,
    pub timestamp: u64,
}

impl MetalReport {
    pub fn new(title: impl Into<String>) -> Self {
        MetalReport { title: title.into(), sections: Vec::new(), metrics: Vec::new(), warnings: Vec::new(), timestamp: 0 }
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

impl fmt::Display for MetalReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MetalReport({})", self.title)
    }
}

/// Configuration for metal analysis.
#[derive(Debug, Clone)]
pub struct MetalConfig {
    pub verbose: bool,
    pub max_iterations: usize,
    pub tolerance: f64,
    pub timeout_ms: u64,
    pub parallel: bool,
    pub output_format: String,
}

impl MetalConfig {
    pub fn default_config() -> Self {
        MetalConfig {
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

impl fmt::Display for MetalConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MetalConfig(iters={}, tol={:.0e})", self.max_iterations, self.tolerance)
    }
}

/// Histogram for metal data distribution.
#[derive(Debug, Clone)]
pub struct MetalHistogramExt {
    pub bins: Vec<usize>,
    pub bin_edges: Vec<f64>,
    pub total_count: usize,
}

impl MetalHistogramExt {
    pub fn from_data(data: &[f64], num_bins: usize) -> Self {
        if data.is_empty() || num_bins == 0 {
            return MetalHistogramExt { bins: Vec::new(), bin_edges: Vec::new(), total_count: 0 };
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
        MetalHistogramExt { bins, bin_edges, total_count: data.len() }
    }
    pub fn num_bins(&self) -> usize { self.bins.len() }
    pub fn max_bin(&self) -> usize { self.bins.iter().cloned().max().unwrap_or(0) }
    pub fn mean_bin(&self) -> f64 {
        if self.bins.is_empty() { return 0.0; }
        self.bins.iter().sum::<usize>() as f64 / self.bins.len() as f64
    }
    pub fn cumulative(&self) -> Vec<usize> {
        let mut cum = Vec::with_capacity(self.bins.len());
        let mut acc = 0usize;
        for &b in &self.bins { acc += b; cum.push(acc); }
        cum
    }
    pub fn entropy(&self) -> f64 {
        let total = self.total_count as f64;
        if total == 0.0 { return 0.0; }
        let mut h = 0.0f64;
        for &b in &self.bins {
            if b > 0 { let p = b as f64 / total; h -= p * p.ln(); }
        }
        h
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

impl fmt::Display for MetalHistogramExt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Histogram(bins={}, total={})", self.num_bins(), self.total_count)
    }
}

/// Adjacency/weight matrix for metal graph analysis.
#[derive(Debug, Clone)]
pub struct MetalGraph {
    pub adjacency: Vec<Vec<bool>>,
    pub weights: Vec<Vec<f64>>,
    pub node_count: usize,
    pub edge_count: usize,
    pub node_labels: Vec<String>,
}

impl MetalGraph {
    pub fn new(n: usize) -> Self {
        MetalGraph {
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
        fn dfs_cycle_metal(v: usize, adj: &[Vec<bool>], visited: &mut [u8]) -> bool {
            visited[v] = 1;
            for w in 0..adj.len() { if adj[v][w] {
                if visited[w] == 1 { return true; }
                if visited[w] == 0 && dfs_cycle_metal(w, adj, visited) { return true; }
            }}
            visited[v] = 2; false
        }
        for i in 0..n {
            if visited[i] == 0 && dfs_cycle_metal(i, &self.adjacency, &mut visited) { return false; }
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

impl fmt::Display for MetalGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MetalGraph(n={}, e={})", self.node_count, self.edge_count)
    }
}

/// Cache for metal computation results.
#[derive(Debug, Clone)]
pub struct MetalCache {
    entries: Vec<(u64, Vec<f64>)>,
    capacity: usize,
    hits: u64,
    misses: u64,
}

impl MetalCache {
    pub fn new(capacity: usize) -> Self {
        MetalCache { entries: Vec::new(), capacity, hits: 0, misses: 0 }
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

impl fmt::Display for MetalCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cache(size={}, hit_rate={:.1}%)", self.size(), self.hit_rate() * 100.0)
    }
}

/// Compute pairwise distances for metal elements.
pub fn metal_pairwise_distances(points: &[Vec<f64>]) -> Vec<Vec<f64>> {
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

/// K-means clustering for metal data.
pub fn metal_kmeans(data: &[Vec<f64>], k: usize, max_iters: usize) -> Vec<usize> {
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

/// Principal component analysis (simplified) for metal data.
pub fn metal_pca_2d(data: &[Vec<f64>]) -> Vec<(f64, f64)> {
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

/// Dense matrix operations for Metal computations.
#[derive(Debug, Clone)]
pub struct MetalDenseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl MetalDenseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        MetalDenseMatrix { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::new(n, n);
        for i in 0..n { m.data[i * n + i] = 1.0; }
        m
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols);
        MetalDenseMatrix { rows, cols, data }
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
        MetalDenseMatrix { rows: self.rows, cols: self.cols, data }
    }

    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows); assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a - b).collect();
        MetalDenseMatrix { rows: self.rows, cols: self.cols, data }
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
        MetalDenseMatrix { rows: self.rows, cols: self.cols, data }
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
        MetalDenseMatrix { rows: self.rows, cols: self.cols, data }
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

impl fmt::Display for MetalDenseMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MetalMatrix({}x{})", self.rows, self.cols)
    }
}

/// Interval arithmetic for Metal bounds analysis.
#[derive(Debug, Clone, Copy)]
pub struct MetalInterval {
    pub lo: f64,
    pub hi: f64,
}

impl MetalInterval {
    pub fn new(lo: f64, hi: f64) -> Self {
        MetalInterval { lo: lo.min(hi), hi: lo.max(hi) }
    }

    pub fn point(v: f64) -> Self {
        MetalInterval { lo: v, hi: v }
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
        MetalInterval { lo: self.lo.min(other.lo), hi: self.hi.max(other.hi) }
    }

    pub fn intersect(&self, other: &Self) -> Option<Self> {
        let lo = self.lo.max(other.lo);
        let hi = self.hi.min(other.hi);
        if lo <= hi { Some(MetalInterval { lo, hi }) } else { None }
    }

    pub fn add(&self, other: &Self) -> Self {
        MetalInterval { lo: self.lo + other.lo, hi: self.hi + other.hi }
    }

    pub fn sub(&self, other: &Self) -> Self {
        MetalInterval { lo: self.lo - other.hi, hi: self.hi - other.lo }
    }

    pub fn mul(&self, other: &Self) -> Self {
        let products = [self.lo * other.lo, self.lo * other.hi, self.hi * other.lo, self.hi * other.hi];
        let lo = products.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = products.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        MetalInterval { lo, hi }
    }

    pub fn abs(&self) -> Self {
        if self.lo >= 0.0 { *self }
        else if self.hi <= 0.0 { MetalInterval { lo: -self.hi, hi: -self.lo } }
        else { MetalInterval { lo: 0.0, hi: self.lo.abs().max(self.hi.abs()) } }
    }

    pub fn sqrt(&self) -> Self {
        let lo = if self.lo >= 0.0 { self.lo.sqrt() } else { 0.0 };
        MetalInterval { lo, hi: self.hi.max(0.0).sqrt() }
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

impl fmt::Display for MetalInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:.4}, {:.4}]", self.lo, self.hi)
    }
}

/// State machine for Metal protocol modeling.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MetalState {
    Allocated,
    Encoded,
    Committed,
    Scheduled,
    Executing,
    Completed,
}

impl fmt::Display for MetalState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetalState::Allocated => write!(f, "allocated"),
            MetalState::Encoded => write!(f, "encoded"),
            MetalState::Committed => write!(f, "committed"),
            MetalState::Scheduled => write!(f, "scheduled"),
            MetalState::Executing => write!(f, "executing"),
            MetalState::Completed => write!(f, "completed"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MetalStateMachine {
    pub current: MetalState,
    pub history: Vec<String>,
    pub transition_count: usize,
}

impl MetalStateMachine {
    pub fn new() -> Self {
        MetalStateMachine { current: MetalState::Allocated, history: Vec::new(), transition_count: 0 }
    }
    pub fn state(&self) -> &MetalState { &self.current }
    pub fn can_transition(&self, target: &MetalState) -> bool {
        match (&self.current, target) {
            (MetalState::Allocated, MetalState::Encoded) => true,
            (MetalState::Encoded, MetalState::Committed) => true,
            (MetalState::Committed, MetalState::Scheduled) => true,
            (MetalState::Scheduled, MetalState::Executing) => true,
            (MetalState::Executing, MetalState::Completed) => true,
            (MetalState::Completed, MetalState::Allocated) => true,
            _ => false,
        }
    }
    pub fn transition(&mut self, target: MetalState) -> bool {
        if self.can_transition(&target) {
            self.history.push(format!("{} -> {}", self.current, target));
            self.current = target;
            self.transition_count += 1;
            true
        } else { false }
    }
    pub fn reset(&mut self) {
        self.current = MetalState::Allocated;
        self.history.clear();
        self.transition_count = 0;
    }
    pub fn history_len(&self) -> usize { self.history.len() }
}

impl fmt::Display for MetalStateMachine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SM(state={}, transitions={})", self.current, self.transition_count)
    }
}

/// Ring buffer for Metal event tracking.
#[derive(Debug, Clone)]
pub struct MetalRingBuffer {
    data: Vec<f64>,
    capacity: usize,
    head: usize,
    count: usize,
}

impl MetalRingBuffer {
    pub fn new(capacity: usize) -> Self {
        MetalRingBuffer { data: vec![0.0; capacity], capacity, head: 0, count: 0 }
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

impl fmt::Display for MetalRingBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RingBuffer(len={}/{})", self.count, self.capacity)
    }
}

/// Disjoint set (union-find) for Metal component tracking.
#[derive(Debug, Clone)]
pub struct MetalDisjointSet {
    parent: Vec<usize>,
    rank: Vec<usize>,
    size: Vec<usize>,
    num_components: usize,
}

impl MetalDisjointSet {
    pub fn new(n: usize) -> Self {
        MetalDisjointSet { parent: (0..n).collect(), rank: vec![0; n], size: vec![1; n], num_components: n }
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

impl fmt::Display for MetalDisjointSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DisjointSet(n={}, components={})", self.parent.len(), self.num_components)
    }
}

/// Sorted list with binary search for Metal.
#[derive(Debug, Clone)]
pub struct MetalSortedList {
    data: Vec<f64>,
}

impl MetalSortedList {
    pub fn new() -> Self { MetalSortedList { data: Vec::new() } }
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

impl fmt::Display for MetalSortedList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SortedList(len={})", self.data.len())
    }
}

/// Exponential moving average for Metal metrics.
#[derive(Debug, Clone)]
pub struct MetalEma {
    pub alpha: f64,
    pub value: f64,
    pub count: usize,
    pub initialized: bool,
}

impl MetalEma {
    pub fn new(alpha: f64) -> Self { MetalEma { alpha, value: 0.0, count: 0, initialized: false } }
    pub fn update(&mut self, sample: f64) {
        if !self.initialized { self.value = sample; self.initialized = true; }
        else { self.value = self.alpha * sample + (1.0 - self.alpha) * self.value; }
        self.count += 1;
    }
    pub fn current(&self) -> f64 { self.value }
    pub fn reset(&mut self) { self.value = 0.0; self.count = 0; self.initialized = false; }
}

impl fmt::Display for MetalEma {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EMA(alpha={:.2}, value={:.4})", self.alpha, self.value)
    }
}

/// Simple bloom filter for Metal membership testing.
#[derive(Debug, Clone)]
pub struct MetalBloomFilter {
    bits: Vec<bool>,
    num_hashes: usize,
    size: usize,
    count: usize,
}

impl MetalBloomFilter {
    pub fn new(size: usize, num_hashes: usize) -> Self {
        MetalBloomFilter { bits: vec![false; size], num_hashes, size, count: 0 }
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

impl fmt::Display for MetalBloomFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BloomFilter(size={}, count={}, fpr={:.4})", self.size, self.count, self.false_positive_rate())
    }
}

/// Simple prefix trie for Metal string matching.
#[derive(Debug, Clone)]
pub struct MetalTrieNode {
    children: Vec<(char, usize)>,
    is_terminal: bool,
    value: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct MetalTrie {
    nodes: Vec<MetalTrieNode>,
    count: usize,
}

impl MetalTrie {
    pub fn new() -> Self {
        MetalTrie { nodes: vec![MetalTrieNode { children: Vec::new(), is_terminal: false, value: None }], count: 0 }
    }
    pub fn insert(&mut self, key: &str, value: u64) {
        let mut current = 0;
        for ch in key.chars() {
            let next = self.nodes[current].children.iter().find(|(c, _)| *c == ch).map(|(_, idx)| *idx);
            current = match next {
                Some(idx) => idx,
                None => {
                    let idx = self.nodes.len();
                    self.nodes.push(MetalTrieNode { children: Vec::new(), is_terminal: false, value: None });
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

impl fmt::Display for MetalTrie {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Trie(entries={}, nodes={})", self.count, self.nodes.len())
    }
}

/// Min-heap priority queue for Metal scheduling.
#[derive(Debug, Clone)]
pub struct MetalPriorityQueue {
    heap: Vec<(f64, usize)>,
}

impl MetalPriorityQueue {
    pub fn new() -> Self { MetalPriorityQueue { heap: Vec::new() } }
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

impl fmt::Display for MetalPriorityQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PQ(len={})", self.heap.len())
    }
}

/// Online statistics accumulator for Metal.
#[derive(Debug, Clone)]
pub struct MetalAccumulator {
    count: u64,
    mean: f64,
    m2: f64,
    min_val: f64,
    max_val: f64,
    sum: f64,
}

impl MetalAccumulator {
    pub fn new() -> Self { MetalAccumulator { count: 0, mean: 0.0, m2: 0.0, min_val: f64::INFINITY, max_val: f64::NEG_INFINITY, sum: 0.0 } }
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

impl fmt::Display for MetalAccumulator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Acc(n={}, mean={:.4}, std={:.4})", self.count, self.mean, self.std_dev())
    }
}

/// Sparse matrix (COO format) for Metal.
#[derive(Debug, Clone)]
pub struct MetalSparseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub entries: Vec<(usize, usize, f64)>,
}

impl MetalSparseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self { MetalSparseMatrix { rows, cols, entries: Vec::new() } }
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
        let mut result = MetalSparseMatrix::new(self.cols, self.rows);
        for &(i, j, v) in &self.entries { result.entries.push((j, i, v)); }
        result
    }
    pub fn add(&self, other: &Self) -> Self {
        let mut result = MetalSparseMatrix::new(self.rows, self.cols);
        for &(i, j, v) in &self.entries { result.insert(i, j, result.get(i, j) + v); }
        for &(i, j, v) in &other.entries { result.insert(i, j, result.get(i, j) + v); }
        result
    }
    pub fn scale(&self, s: f64) -> Self {
        let mut result = MetalSparseMatrix::new(self.rows, self.cols);
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

impl fmt::Display for MetalSparseMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sparse({}x{}, nnz={})", self.rows, self.cols, self.nnz())
    }
}

/// Polynomial with f64 coefficients for Metal.
#[derive(Debug, Clone)]
pub struct MetalPolynomial {
    pub coefficients: Vec<f64>,
}

impl MetalPolynomial {
    pub fn new(coeffs: Vec<f64>) -> Self { MetalPolynomial { coefficients: coeffs } }
    pub fn zero() -> Self { MetalPolynomial { coefficients: vec![0.0] } }
    pub fn one() -> Self { MetalPolynomial { coefficients: vec![1.0] } }
    pub fn monomial(degree: usize, coeff: f64) -> Self {
        let mut c = vec![0.0; degree + 1];
        c[degree] = coeff;
        MetalPolynomial { coefficients: c }
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
        MetalPolynomial { coefficients: result }
    }
    pub fn sub(&self, other: &Self) -> Self {
        let n = self.coefficients.len().max(other.coefficients.len());
        let mut result = vec![0.0; n];
        for (i, &c) in self.coefficients.iter().enumerate() { result[i] += c; }
        for (i, &c) in other.coefficients.iter().enumerate() { result[i] -= c; }
        MetalPolynomial { coefficients: result }
    }
    pub fn mul(&self, other: &Self) -> Self {
        let n = self.coefficients.len() + other.coefficients.len() - 1;
        let mut result = vec![0.0; n];
        for (i, &a) in self.coefficients.iter().enumerate() {
            for (j, &b) in other.coefficients.iter().enumerate() {
                result[i + j] += a * b;
            }
        }
        MetalPolynomial { coefficients: result }
    }
    pub fn scale(&self, s: f64) -> Self {
        MetalPolynomial { coefficients: self.coefficients.iter().map(|&c| c * s).collect() }
    }
    pub fn derivative(&self) -> Self {
        if self.coefficients.len() <= 1 { return Self::zero(); }
        let coeffs: Vec<f64> = self.coefficients.iter().enumerate().skip(1)
            .map(|(i, &c)| c * i as f64).collect();
        MetalPolynomial { coefficients: coeffs }
    }
    pub fn integral(&self, constant: f64) -> Self {
        let mut coeffs = vec![constant];
        for (i, &c) in self.coefficients.iter().enumerate() {
            coeffs.push(c / (i + 1) as f64);
        }
        MetalPolynomial { coefficients: coeffs }
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

impl fmt::Display for MetalPolynomial {
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

/// Simple linear congruential generator for Metal.
#[derive(Debug, Clone)]
pub struct MetalRng {
    state: u64,
}

impl MetalRng {
    pub fn new(seed: u64) -> Self { MetalRng { state: seed } }
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

impl fmt::Display for MetalRng {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Rng(state={:#x})", self.state)
    }
}

/// Simple timing utility for Metal benchmarking.
#[derive(Debug, Clone)]
pub struct MetalTimer {
    pub label: String,
    pub elapsed_ns: Vec<u64>,
    pub running: bool,
}

impl MetalTimer {
    pub fn new(label: impl Into<String>) -> Self { MetalTimer { label: label.into(), elapsed_ns: Vec::new(), running: false } }
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

impl fmt::Display for MetalTimer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Timer({}: avg={:.0}ns, n={})", self.label, self.average_ns(), self.count())
    }
}

/// Compact bit vector for Metal set operations.
#[derive(Debug, Clone)]
pub struct MetalBitVector {
    words: Vec<u64>,
    len: usize,
}

impl MetalBitVector {
    pub fn new(len: usize) -> Self { MetalBitVector { words: vec![0u64; (len + 63) / 64], len } }
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

impl fmt::Display for MetalBitVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BitVec(len={}, ones={})", self.len, self.count_ones())
    }
}

/// LRU cache for Metal computation memoization.
#[derive(Debug, Clone)]
pub struct MetalLruCache {
    entries: Vec<(u64, Vec<f64>, u64)>,
    capacity: usize,
    clock: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl MetalLruCache {
    pub fn new(capacity: usize) -> Self { MetalLruCache { entries: Vec::new(), capacity, clock: 0, hits: 0, misses: 0, evictions: 0 } }
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

impl fmt::Display for MetalLruCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LRU(size={}/{}, hr={:.1}%)", self.size(), self.capacity, self.hit_rate() * 100.0)
    }
}

/// Graph coloring utility for Metal scheduling.
#[derive(Debug, Clone)]
pub struct MetalGraphColoring {
    pub adjacency: Vec<Vec<bool>>,
    pub colors: Vec<Option<usize>>,
    pub num_nodes: usize,
    pub num_colors_used: usize,
}

impl MetalGraphColoring {
    pub fn new(n: usize) -> Self {
        MetalGraphColoring { adjacency: vec![vec![false; n]; n], colors: vec![None; n], num_nodes: n, num_colors_used: 0 }
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

impl fmt::Display for MetalGraphColoring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Coloring(n={}, colors={})", self.num_nodes, self.num_colors_used)
    }
}

/// Top-K tracker for Metal ranking.
#[derive(Debug, Clone)]
pub struct MetalTopK {
    pub k: usize,
    pub items: Vec<(f64, String)>,
}

impl MetalTopK {
    pub fn new(k: usize) -> Self { MetalTopK { k, items: Vec::new() } }
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

impl fmt::Display for MetalTopK {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TopK(k={}, len={})", self.k, self.items.len())
    }
}

/// Sliding window statistics for Metal monitoring.
#[derive(Debug, Clone)]
pub struct MetalSlidingWindow {
    pub window_size: usize,
    pub data: Vec<f64>,
    pub sum: f64,
}

impl MetalSlidingWindow {
    pub fn new(window_size: usize) -> Self { MetalSlidingWindow { window_size, data: Vec::new(), sum: 0.0 } }
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

impl fmt::Display for MetalSlidingWindow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Window(size={}/{}, mean={:.2})", self.data.len(), self.window_size, self.mean())
    }
}

/// Confusion matrix for Metal classification evaluation.
#[derive(Debug, Clone)]
pub struct MetalConfusionMatrix {
    pub true_positive: u64,
    pub false_positive: u64,
    pub true_negative: u64,
    pub false_negative: u64,
}

impl MetalConfusionMatrix {
    pub fn new() -> Self { MetalConfusionMatrix { true_positive: 0, false_positive: 0, true_negative: 0, false_negative: 0 } }
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

impl fmt::Display for MetalConfusionMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CM(acc={:.3}, prec={:.3}, rec={:.3}, f1={:.3})",
            self.accuracy(), self.precision(), self.recall(), self.f1_score())
    }
}

/// Cosine similarity for Metal feature vectors.
pub fn metal_cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na * nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

/// Euclidean distance for Metal.
pub fn metal_euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
}

/// Manhattan distance for Metal.
pub fn metal_manhattan_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// Chebyshev distance for Metal.
pub fn metal_chebyshev_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max)
}

/// Minkowski distance for Metal.
pub fn metal_minkowski_distance(a: &[f64], b: &[f64], p: f64) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs().powf(p)).sum::<f64>().powf(1.0 / p)
}

/// Normalize a vector for Metal.
pub fn metal_normalize(v: &[f64]) -> Vec<f64> {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 { v.to_vec() } else { v.iter().map(|x| x / norm).collect() }
}

/// Dot product for Metal.
pub fn metal_dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Cross product (3D) for Metal.
pub fn metal_cross_product(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
}

/// Linear interpolation for Metal.
pub fn metal_lerp(a: f64, b: f64, t: f64) -> f64 { a + (b - a) * t }

/// Clamp value for Metal.
pub fn metal_clamp(v: f64, lo: f64, hi: f64) -> f64 { v.max(lo).min(hi) }

/// Sigmoid function for Metal.
pub fn metal_sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }

/// Softmax for Metal.
pub fn metal_softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Log-sum-exp for Metal.
pub fn metal_logsumexp(values: &[f64]) -> f64 {
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    max + values.iter().map(|&v| (v - max).exp()).sum::<f64>().ln()
}

/// KL divergence for Metal.
pub fn metal_kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q.iter()).map(|(&pi, &qi)| {
        if pi > 0.0 && qi > 0.0 { pi * (pi / qi).ln() } else { 0.0 }
    }).sum()
}

/// Jensen-Shannon divergence for Metal.
pub fn metal_js_divergence(p: &[f64], q: &[f64]) -> f64 {
    let m: Vec<f64> = p.iter().zip(q.iter()).map(|(&pi, &qi)| (pi + qi) / 2.0).collect();
    (metal_kl_divergence(p, &m) + metal_kl_divergence(q, &m)) / 2.0
}

/// Total variation distance for Metal.
pub fn metal_tv_distance(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q.iter()).map(|(&pi, &qi)| (pi - qi).abs()).sum::<f64>() / 2.0
}

/// Hellinger distance for Metal.
pub fn metal_hellinger_distance(p: &[f64], q: &[f64]) -> f64 {
    let sum: f64 = p.iter().zip(q.iter()).map(|(&pi, &qi)| {
        let diff = pi.sqrt() - qi.sqrt();
        diff * diff
    }).sum();
    (sum / 2.0).sqrt()
}

/// Earth mover's distance (1D) for Metal.
pub fn metal_emd_1d(p: &[f64], q: &[f64]) -> f64 {
    let mut cum_diff = 0.0;
    let mut total = 0.0;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        cum_diff += pi - qi;
        total += cum_diff.abs();
    }
    total
}

/// Feature scaling utilities for Metal.
#[derive(Debug, Clone)]
pub struct MetalFeatureScaler {
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
    pub mins: Vec<f64>,
    pub maxs: Vec<f64>,
    pub fitted: bool,
}

impl MetalFeatureScaler {
    pub fn new() -> Self { MetalFeatureScaler { means: Vec::new(), stds: Vec::new(), mins: Vec::new(), maxs: Vec::new(), fitted: false } }
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

impl fmt::Display for MetalFeatureScaler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Scaler(dim={}, fitted={})", self.dimension(), self.fitted)
    }
}

/// Simple linear regression for Metal trend analysis.
#[derive(Debug, Clone)]
pub struct MetalLinearRegression {
    pub slope: f64,
    pub intercept: f64,
    pub r_squared: f64,
    pub fitted: bool,
}

impl MetalLinearRegression {
    pub fn new() -> Self { MetalLinearRegression { slope: 0.0, intercept: 0.0, r_squared: 0.0, fitted: false } }
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

impl fmt::Display for MetalLinearRegression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "y = {:.4}x + {:.4} (R2={:.4})", self.slope, self.intercept, self.r_squared)
    }
}

/// Weighted undirected graph for Metal.
#[derive(Debug, Clone)]
pub struct MetalWeightedGraph {
    pub adj: Vec<Vec<(usize, f64)>>,
    pub num_nodes: usize,
    pub num_edges: usize,
}

impl MetalWeightedGraph {
    pub fn new(n: usize) -> Self { MetalWeightedGraph { adj: vec![Vec::new(); n], num_nodes: n, num_edges: 0 } }
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
        fn find_metal(parent: &mut Vec<usize>, x: usize) -> usize {
            if parent[x] != x { parent[x] = find_metal(parent, parent[x]); }
            parent[x]
        }
        let mut total = 0.0;
        let mut count = 0;
        for (w, u, v) in edges {
            let ru = find_metal(&mut parent, u);
            let rv = find_metal(&mut parent, v);
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

impl fmt::Display for MetalWeightedGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WGraph(n={}, e={})", self.num_nodes, self.num_edges)
    }
}

/// Moving average for Metal.
pub fn metal_moving_average(data: &[f64], window: usize) -> Vec<f64> {
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

/// Cumulative sum for Metal.
pub fn metal_cumsum(data: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(data.len());
    let mut sum = 0.0;
    for &v in data { sum += v; result.push(sum); }
    result
}

/// Numerical differentiation for Metal.
pub fn metal_diff(data: &[f64]) -> Vec<f64> {
    if data.len() < 2 { return Vec::new(); }
    data.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Auto-correlation for Metal.
pub fn metal_autocorrelation(data: &[f64], lag: usize) -> f64 {
    if data.len() <= lag { return 0.0; }
    let n = data.len();
    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let var: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if var.abs() < 1e-15 { return 0.0; }
    let cov: f64 = (0..(n - lag)).map(|i| (data[i] - mean) * (data[i + lag] - mean)).sum::<f64>() / n as f64;
    cov / var
}

/// Discrete Fourier transform magnitude for Metal.
pub fn metal_dft_magnitude(data: &[f64]) -> Vec<f64> {
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

/// Trapezoidal integration for Metal.
pub fn metal_integrate_trapezoid(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let mut total = 0.0;
    for i in 1..x.len() {
        total += (x[i] - x[i-1]) * (y[i] + y[i-1]) / 2.0;
    }
    total
}

/// Simpson's rule integration for Metal.
pub fn metal_integrate_simpson(x: &[f64], y: &[f64]) -> f64 {
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

/// Convolution for Metal.
pub fn metal_convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len() + b.len() - 1;
    let mut result = vec![0.0; n];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            result[i + j] += ai * bj;
        }
    }
    result
}

/// Axis-aligned bounding box for Metal spatial indexing.
#[derive(Debug, Clone, Copy)]
pub struct MetalAABB {
    pub x_min: f64, pub y_min: f64,
    pub x_max: f64, pub y_max: f64,
}

impl MetalAABB {
    pub fn new(x_min: f64, y_min: f64, x_max: f64, y_max: f64) -> Self { MetalAABB { x_min, y_min, x_max, y_max } }
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
            MetalAABB::new(self.x_min, self.y_min, cx, cy),
            MetalAABB::new(cx, self.y_min, self.x_max, cy),
            MetalAABB::new(self.x_min, cy, cx, self.y_max),
            MetalAABB::new(cx, cy, self.x_max, self.y_max),
        ]
    }
}

/// 2D point for Metal.
#[derive(Debug, Clone, Copy)]
pub struct MetalPoint2D { pub x: f64, pub y: f64, pub data: f64 }

/// Quadtree for Metal spatial indexing.
#[derive(Debug, Clone)]
pub struct MetalQuadTree {
    pub boundary: MetalAABB,
    pub points: Vec<MetalPoint2D>,
    pub children: Option<Vec<MetalQuadTree>>,
    pub capacity: usize,
    pub depth: usize,
    pub max_depth: usize,
}

impl MetalQuadTree {
    pub fn new(boundary: MetalAABB, capacity: usize, max_depth: usize) -> Self {
        MetalQuadTree { boundary, points: Vec::new(), children: None, capacity, depth: 0, max_depth }
    }
    fn with_depth(boundary: MetalAABB, capacity: usize, depth: usize, max_depth: usize) -> Self {
        MetalQuadTree { boundary, points: Vec::new(), children: None, capacity, depth, max_depth }
    }
    pub fn insert(&mut self, p: MetalPoint2D) -> bool {
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
            children.push(MetalQuadTree::with_depth(*q, self.capacity, self.depth + 1, self.max_depth));
        }
        let old_points: Vec<_> = self.points.drain(..).collect();
        self.children = Some(children);
        for p in old_points { self.insert(p); }
    }
    pub fn query_range(&self, range: &MetalAABB) -> Vec<MetalPoint2D> {
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

impl fmt::Display for MetalQuadTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QTree(count={}, depth={})", self.count(), self.tree_depth())
    }
}

/// QR decomposition helper for Metal.
pub fn metal_qr_decompose(a: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
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

/// Solve upper triangular system Rx = b for Metal.
pub fn metal_solve_upper_triangular(r: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i+1)..n { s -= r[i][j] * x[j]; }
        x[i] = if r[i][i].abs() > 1e-15 { s / r[i][i] } else { 0.0 };
    }
    x
}

/// Matrix-vector multiply for Metal.
pub fn metal_mat_vec_mul(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter().map(|row| row.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum()).collect()
}

/// Matrix transpose for Metal.
pub fn metal_transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() { return Vec::new(); }
    let m = a.len(); let n = a[0].len();
    (0..n).map(|j| (0..m).map(|i| a[i][j]).collect()).collect()
}

/// Matrix multiply for Metal.
pub fn metal_mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    if m == 0 { return Vec::new(); }
    let k = a[0].len();
    let n = b[0].len();
    let mut c = vec![vec![0.0; n]; m];
    for i in 0..m { for j in 0..n { for l in 0..k { c[i][j] += a[i][l] * b[l][j]; } } }
    c
}

/// Frobenius norm for Metal.
pub fn metal_frobenius_norm(a: &[Vec<f64>]) -> f64 {
    a.iter().flat_map(|row| row.iter()).map(|&x| x * x).sum::<f64>().sqrt()
}

/// Matrix trace for Metal.
pub fn metal_trace(a: &[Vec<f64>]) -> f64 {
    a.iter().enumerate().map(|(i, row)| if i < row.len() { row[i] } else { 0.0 }).sum()
}

/// Identity matrix for Metal.
pub fn metal_identity(n: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n { m[i][i] = 1.0; }
    m
}

/// Power iteration for dominant eigenvalue for Metal.
pub fn metal_power_iteration(a: &[Vec<f64>], max_iter: usize) -> (f64, Vec<f64>) {
    let n = a.len();
    let mut v = vec![1.0; n];
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut v { *x /= norm; }
    let mut eigenvalue = 0.0;
    for _ in 0..max_iter {
        let av = metal_mat_vec_mul(a, &v);
        let norm: f64 = av.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm.abs() < 1e-15 { break; }
        eigenvalue = av.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum();
        v = av.iter().map(|&x| x / norm).collect();
    }
    (eigenvalue, v)
}

/// Running statistics with min/max tracking for Metal.
#[derive(Debug, Clone)]
pub struct MetalRunningStats {
    pub count: u64,
    pub mean: f64,
    pub m2: f64,
    pub min_val: f64,
    pub max_val: f64,
    pub sum: f64,
}

impl MetalRunningStats {
    pub fn new() -> Self { MetalRunningStats { count: 0, mean: 0.0, m2: 0.0, min_val: f64::INFINITY, max_val: f64::NEG_INFINITY, sum: 0.0 } }
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

impl fmt::Display for MetalRunningStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Stats(n={}, mean={:.4}, std={:.4})", self.count, self.mean, self.std_dev())
    }
}

/// Interquartile range for Metal.
pub fn metal_iqr(data: &[f64]) -> f64 {
    metal_percentile_at(data, 75.0) - metal_percentile_at(data, 25.0)
}

/// Detect outliers using IQR method for Metal.
pub fn metal_outliers(data: &[f64]) -> Vec<usize> {
    let q1 = metal_percentile_at(data, 25.0);
    let q3 = metal_percentile_at(data, 75.0);
    let iqr = q3 - q1;
    let lower = q1 - 1.5 * iqr;
    let upper = q3 + 1.5 * iqr;
    data.iter().enumerate().filter(|(_, &v)| v < lower || v > upper).map(|(i, _)| i).collect()
}

/// Z-score normalization for Metal.
pub fn metal_zscore(data: &[f64]) -> Vec<f64> {
    let n = data.len() as f64;
    if n < 2.0 { return data.to_vec(); }
    let mean = data.iter().sum::<f64>() / n;
    let std = (data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    if std.abs() < 1e-15 { return vec![0.0; data.len()]; }
    data.iter().map(|&x| (x - mean) / std).collect()
}

/// Rank values for Metal.
pub fn metal_rank(data: &[f64]) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut ranks = vec![0.0; data.len()];
    for (rank, &(idx, _)) in indexed.iter().enumerate() { ranks[idx] = (rank + 1) as f64; }
    ranks
}

/// Spearman rank correlation for Metal.
pub fn metal_spearman(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let rx = metal_rank(x);
    let ry = metal_rank(y);
    let n = x.len() as f64;
    let d_sq: f64 = rx.iter().zip(ry.iter()).map(|(&a, &b)| (a - b).powi(2)).sum();
    1.0 - 6.0 * d_sq / (n * (n * n - 1.0))
}

/// Covariance matrix for Metal.
pub fn metal_covariance_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
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

/// Correlation matrix for Metal.
pub fn metal_correlation_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let cov = metal_covariance_matrix(data);
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

    fn make_metal_event(id: u32, tid: u32, tgid: u32, op: MetalAtomicOp, addr: u64) -> MetalEvent {
        MetalEvent {
            id,
            thread_id: tid,
            threadgroup_id: tgid,
            simdgroup_id: 0,
            op,
            address: addr,
            address_space: MetalAddressSpace::Device,
            memory_order: MetalMemoryOrder::Relaxed,
            value: None,
            timestamp: id as u64,
            is_non_atomic: false,
        }
    }

    fn make_metal_event_ordered(id: u32, tid: u32, tgid: u32, op: MetalAtomicOp, addr: u64, order: MetalMemoryOrder) -> MetalEvent {
        MetalEvent {
            id,
            thread_id: tid,
            threadgroup_id: tgid,
            simdgroup_id: 0,
            op,
            address: addr,
            address_space: MetalAddressSpace::Device,
            memory_order: order,
            value: None,
            timestamp: id as u64,
            is_non_atomic: false,
        }
    }

    #[test]
    fn test_address_space_properties() {
        assert!(MetalAddressSpace::Device.is_shared());
        assert!(MetalAddressSpace::Threadgroup.is_shared());
        assert!(!MetalAddressSpace::Thread.is_shared());
        assert!(MetalAddressSpace::Thread.is_private());
        assert!(MetalAddressSpace::Constant.is_read_only());
    }

    #[test]
    fn test_memory_order_comparison() {
        assert!(MetalMemoryOrder::SeqCst.is_at_least(MetalMemoryOrder::Relaxed));
        assert!(!MetalMemoryOrder::Relaxed.is_at_least(MetalMemoryOrder::SeqCst));
    }

    #[test]
    fn test_barrier_flags() {
        let f = MetalBarrierFlags::all();
        assert!(f.affects_device());
        assert!(f.affects_threadgroup());
        assert!(f.affects_texture());

        let f2 = MetalBarrierFlags::device();
        assert!(f2.affects_device());
        assert!(!f2.affects_threadgroup());
    }

    #[test]
    fn test_barrier_flags_combine() {
        let f1 = MetalBarrierFlags::device();
        let f2 = MetalBarrierFlags::threadgroup();
        let combined = f1.combine(f2);
        assert!(combined.affects_device());
        assert!(combined.affects_threadgroup());
    }

    #[test]
    fn test_threadgroup_barrier_convergence() {
        let mut checker = ThreadgroupBarrierChecker::new(4);
        checker.add_barrier(ThreadgroupBarrier::new(0, MetalBarrierFlags::all(), 0)
            .with_threads(vec![0, 1, 2, 3]));
        let violations = checker.check_convergence();
        assert!(violations.is_empty());
    }

    #[test]
    fn test_threadgroup_barrier_missing() {
        let mut checker = ThreadgroupBarrierChecker::new(4);
        checker.add_barrier(ThreadgroupBarrier::new(0, MetalBarrierFlags::all(), 0)
            .with_threads(vec![0, 1, 2]));
        let violations = checker.check_convergence();
        assert!(!violations.is_empty());
    }

    #[test]
    fn test_atomic_op_properties() {
        assert!(MetalAtomicOp::Load.is_read());
        assert!(!MetalAtomicOp::Load.is_write());
        assert!(MetalAtomicOp::Store.is_write());
        assert!(!MetalAtomicOp::Store.is_read());
        assert!(MetalAtomicOp::FetchAdd.is_rmw());
    }

    #[test]
    fn test_bit_matrix_basic() {
        let mut m = BitMatrix::new(3, 3);
        m.set_true(0, 1);
        m.set_true(1, 2);
        assert!(m.get(0, 1));
        assert!(!m.get(0, 2));
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
    fn test_model_checker_acyclic_hb() {
        let events = vec![
            make_metal_event(0, 0, 0, MetalAtomicOp::Store, 100),
            make_metal_event(1, 0, 0, MetalAtomicOp::Load, 100),
        ];
        let mut checker = MetalModelChecker::new(events, 32);
        checker.add_po(0, 1);
        checker.compute_hb();
        let violations = checker.check_axiom(MetalAxiom::HbAcyclicity);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_model_checker_cyclic_hb() {
        let events = vec![
            make_metal_event(0, 0, 0, MetalAtomicOp::Store, 100),
            make_metal_event(1, 1, 0, MetalAtomicOp::Store, 100),
        ];
        let mut checker = MetalModelChecker::new(events, 32);
        checker.add_po(0, 1);
        checker.add_po(1, 0);
        checker.compute_hb();
        let violations = checker.check_axiom(MetalAxiom::HbAcyclicity);
        assert!(!violations.is_empty());
    }

    #[test]
    fn test_model_checker_hb_via_sync() {
        let events = vec![
            make_metal_event_ordered(0, 0, 0, MetalAtomicOp::Store, 100, MetalMemoryOrder::Release),
            make_metal_event_ordered(1, 1, 0, MetalAtomicOp::Load, 100, MetalMemoryOrder::Acquire),
        ];
        let mut checker = MetalModelChecker::new(events, 32);
        checker.add_rf(0, 1);
        checker.compute_hb();
        assert!(checker.hb.get(0, 1));
    }

    #[test]
    fn test_metal_grid() {
        let grid = MetalGrid::new_2d(16, 16);
        assert_eq!(grid.total_threads(), 256);
        assert_eq!(grid.dimensions, 2);
    }

    #[test]
    fn test_threadgroup() {
        let tg = Threadgroup::new(0, 64, 32, 8192);
        assert_eq!(tg.threads.len(), 64);
        assert_eq!(tg.simd_groups.len(), 2);
        assert_eq!(tg.threads[0].threadgroup_id, 0);
        assert_eq!(tg.threads[32].simdgroup_index, 1);
    }

    #[test]
    fn test_metal_kernel() {
        let kernel = MetalKernel::new(
            "compute_kernel",
            MetalGrid::new_1d(256),
            32,
            32,
            4096,
        );
        assert_eq!(kernel.total_threads(), 256);
        assert_eq!(kernel.num_threadgroups(), 8);
        assert_eq!(kernel.num_simdgroups_per_threadgroup(), 1);
    }

    #[test]
    fn test_simd_group() {
        let tg = Threadgroup::new(0, 64, 32, 4096);
        assert_eq!(tg.simd_groups[0].size(), 32);
        assert!(tg.simd_groups[0].contains_thread(0));
        assert!(!tg.simd_groups[0].contains_thread(32));
    }

    #[test]
    fn test_icb_commands() {
        let mut icb = IndirectCommandBufferExt::from_capacity(0, 10);
        icb.add_command(ICBCommand::dispatch(0, 0, 4)).unwrap();
        icb.add_command(ICBCommand::draw(1, 1)).unwrap();
        assert_eq!(icb.commands.len(), 2);
    }

    #[test]
    fn test_icb_max_commands() {
        let mut icb = IndirectCommandBufferExt::from_capacity(0, 1);
        assert!(icb.add_command(ICBCommand::dispatch(0, 0, 4)).is_ok());
        assert!(icb.add_command(ICBCommand::dispatch(1, 1, 4)).is_err());
    }

    #[test]
    fn test_icb_ordering() {
        let mut ordering = ICBOrdering::new();
        ordering.add_command(0);
        ordering.add_command(1);
        ordering.add_command(2);
        ordering.add_dependency(1, 0);
        ordering.add_dependency(2, 1);
        let topo = ordering.topological_order();
        assert!(topo.is_some());
        let order = topo.unwrap();
        assert_eq!(order[0], 0);
        assert_eq!(order[1], 1);
        assert_eq!(order[2], 2);
    }

    #[test]
    fn test_icb_ordering_cycle() {
        let mut ordering = ICBOrdering::new();
        ordering.add_command(0);
        ordering.add_command(1);
        ordering.add_dependency(0, 1);
        ordering.add_dependency(1, 0);
        assert!(!ordering.is_valid_order());
    }

    #[test]
    fn test_fence_checker() {
        let mut checker = FenceChecker::new();
        checker.add_fence(MetalFence::new(0, MetalFenceScope::DeviceMemory, 0, 0));
        let issues = checker.check_ordering();
        assert!(!issues.is_empty()); // No ordered events = redundant
    }

    #[test]
    fn test_fence_build_ordering() {
        let mut checker = FenceChecker::new();
        let mut fence = MetalFence::new(0, MetalFenceScope::DeviceMemory, 0, 0);
        fence.before_events = vec![0, 1];
        fence.after_events = vec![2, 3];
        checker.add_fence(fence);
        let edges = checker.build_ordering();
        assert_eq!(edges.len(), 4);
    }

    #[test]
    fn test_texture_race_detection() {
        let accesses = vec![
            TextureAccess::write(0, 0, (0, 0), 0).with_timestamp(1),
            TextureAccess::read(1, 0, (0, 0), 1).with_timestamp(2),
        ];
        let model = TextureMemoryModel::from_accesses(accesses);
        let races = model.check_races();
        assert_eq!(races.len(), 1);
    }

    #[test]
    fn test_texture_no_race_read_read() {
        let accesses = vec![
            TextureAccess::read(0, 0, (0, 0), 0).with_timestamp(1),
            TextureAccess::read(1, 0, (0, 0), 1).with_timestamp(2),
        ];
        let model = TextureMemoryModel::from_accesses(accesses);
        let races = model.check_races();
        assert!(races.is_empty());
    }
    #[test]
    fn test_metalresourcebinding_new() {
        let item = MetalResourceBinding::new(0, "test".to_string(), "test".to_string(), false);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_argumentbuffersemantics_new() {
        let item = ArgumentBufferSemantics::new(0, 0, 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_heapallocationmodel_new() {
        let item = HeapAllocationModel::new(0, 0, 0.0, 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_residencytracker_new() {
        let item = ResidencyTracker::new(Vec::new(), 0, 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_gpufamilycapabilities_new() {
        let item = GpuFamilyCapabilities::new("test".to_string(), 0, 0, false);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_tileshadingmodel_new() {
        let item = TileShadingModel::new(0, 0, 0, 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_meshshadermemorymodel_new() {
        let item = MeshShaderMemoryModel::new(0, 0, 0, 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_raytracingintersection_new() {
        let item = RayTracingIntersection::new(0, 0, 0.0, Vec::new());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_metalfencemodel_new() {
        let item = MetalFenceModel::new(0, "test".to_string(), false);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_metaleventmodel_new() {
        let item = MetalEventModel::new(0, 0, false);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_indirectcommandbuffer_new() {
        let item = IndirectCommandBufferExt::new(0, 0, Vec::new());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_metaltexturedescriptor_new() {
        let item = MetalTextureDescriptor::new(0, 0, 0, "test".to_string(), "test".to_string());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_metal_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let m = metal_mean(&data);
        assert!((m - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_variance() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let v = metal_variance(&data);
        assert!(v > 0.0);
    }

    #[test]
    fn test_metal_median() {
        let data = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let m = metal_median(&data);
        assert!((m - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_entropy() {
        let data = vec![1.0, 1.0, 1.0, 1.0];
        let e = metal_entropy(&data);
        assert!(e > 0.0);
    }

    #[test]
    fn test_metal_std_dev() {
        let data = vec![10.0, 10.0, 10.0];
        let s = metal_std_dev(&data);
        assert!(s.abs() < 1e-10);
    }

    #[test]
    fn test_metal_analysis() {
        let mut a = MetalAnalysis::new(3);
        a.set(0, 1, 0.5);
        a.set(1, 2, 0.3);
        assert!((a.get(0, 1) - 0.5).abs() < 1e-10);
        assert_eq!(a.size, 3);
    }

    #[test]
    fn test_metal_iterator() {
        let iter = MetalResultIterator::new(vec![(0, 1.0), (1, 2.0), (2, 3.0)]);
        let items: Vec<_> = iter.collect();
        assert_eq!(items.len(), 3);
    }

    #[test]
    fn test_metal_batch_processor() {
        let mut proc = MetalBatchProcessor::new(2);
        proc.process_batch(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(proc.processed, 4);
        assert_eq!(proc.results.len(), 2);
    }

    #[test]
    fn test_metal_histogram() {
        let hist = MetalHistogramExt::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0], 3);
        assert_eq!(hist.num_bins(), 3);
        assert_eq!(hist.total_count, 5);
    }

    #[test]
    fn test_metal_graph() {
        let mut g = MetalGraph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 2.0);
        g.add_edge(2, 3, 3.0);
        assert_eq!(g.edge_count, 3);
        assert!(g.has_edge(0, 1));
        assert!(!g.has_edge(3, 0));
        assert!(g.is_acyclic());
    }

    #[test]
    fn test_metal_graph_shortest_path() {
        let mut g = MetalGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 2.0);
        g.add_edge(0, 2, 10.0);
        let dist = g.shortest_path_dijkstra(0);
        assert!((dist[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_graph_topo_sort() {
        let mut g = MetalGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        let topo = g.topological_sort();
        assert!(topo.is_some());
    }

    #[test]
    fn test_metal_graph_components() {
        let mut g = MetalGraph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(2, 3, 1.0);
        let comps = g.connected_components();
        assert_eq!(comps.len(), 2);
    }

    #[test]
    fn test_metal_cache() {
        let mut cache = MetalCache::new(10);
        cache.insert(42, vec![1.0, 2.0]);
        assert!(cache.get(42).is_some());
        assert!(cache.get(99).is_none());
    }

    #[test]
    fn test_metal_config() {
        let config = MetalConfig::default_config().with_verbose(true).with_max_iterations(500);
        assert!(config.verbose);
        assert_eq!(config.max_iterations, 500);
    }

    #[test]
    fn test_metal_report() {
        let mut report = MetalReport::new("Test Report");
        report.add_metric("accuracy", 0.95);
        report.add_warning("low sample size");
        assert_eq!(report.total_metrics(), 1);
        assert!(report.has_warnings());
        let text = report.render_text();
        assert!(text.contains("Test Report"));
    }

    #[test]
    fn test_metal_kmeans() {
        let data = vec![vec![0.0, 0.0], vec![0.1, 0.1], vec![10.0, 10.0], vec![10.1, 10.1]];
        let assignments = metal_kmeans(&data, 2, 100);
        assert_eq!(assignments.len(), 4);
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[2], assignments[3]);
    }

    #[test]
    fn test_metal_pairwise_distances() {
        let points = vec![vec![0.0, 0.0], vec![3.0, 4.0]];
        let dists = metal_pairwise_distances(&points);
        assert!((dists[0][1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_harmmean() {
        let data = vec![1.0, 2.0, 4.0];
        let hm = metal_harmmean(&data);
        assert!(hm > 0.0 && hm < 4.0);
    }

    #[test]
    fn test_metal_geomean() {
        let data = vec![1.0, 2.0, 4.0];
        let gm = metal_geomean(&data);
        assert!(gm > 0.0 && gm < 4.0);
    }

    #[test]
    fn test_metal_sample_skewness() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = metal_sample_skewness(&data);
        assert!(s.abs() < 1.0);
    }

    #[test]
    fn test_metal_excess_kurtosis() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let k = metal_excess_kurtosis(&data);
        let _ = k; // Just verify it computes
    }

    #[test]
    fn test_metal_gini() {
        let data = vec![1.0, 1.0, 1.0, 1.0];
        let g = metal_gini(&data);
        assert!(g.abs() < 0.01);
    }

    #[test]
    fn test_metal_percentile_at() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let p = metal_percentile_at(&data, 0.9);
        assert!(p >= 9.0);
    }

    #[test]
    fn test_metal_pca_2d() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let proj = metal_pca_2d(&data);
        assert_eq!(proj.len(), 2);
    }

    #[test]
    fn test_metal_analysis_normalize() {
        let mut a = MetalAnalysis::new(2);
        a.set(0, 0, 1.0); a.set(0, 1, 3.0);
        a.normalize();
        assert!((a.total_sum() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_analysis_transpose() {
        let mut a = MetalAnalysis::new(2);
        a.set(0, 1, 5.0);
        let t = a.transpose();
        assert!((t.get(1, 0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_analysis_multiply() {
        let mut a = MetalAnalysis::new(2);
        a.set(0, 0, 1.0); a.set(1, 1, 1.0);
        let mut b = MetalAnalysis::new(2);
        b.set(0, 1, 2.0); b.set(1, 0, 3.0);
        let c = a.multiply(&b);
        assert!((c.get(0, 1) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_analysis_frobenius() {
        let mut a = MetalAnalysis::new(2);
        a.set(0, 0, 3.0); a.set(1, 1, 4.0);
        assert!((a.frobenius_norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_analysis_symmetric() {
        let mut a = MetalAnalysis::new(2);
        a.set(0, 1, 1.0); a.set(1, 0, 1.0);
        assert!(a.is_symmetric());
    }

    #[test]
    fn test_metal_graph_dot() {
        let mut g = MetalGraph::new(2);
        g.add_edge(0, 1, 1.0);
        let dot = g.to_dot();
        assert!(dot.contains("digraph"));
    }

    #[test]
    fn test_metal_histogram_render() {
        let hist = MetalHistogramExt::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 5);
        let ascii = hist.render_ascii(20);
        assert!(!ascii.is_empty());
    }

    #[test]
    fn test_metal_batch_reset() {
        let mut proc = MetalBatchProcessor::new(3);
        proc.process_batch(&[1.0, 2.0, 3.0]);
        assert!(proc.processed > 0);
        proc.reset();
        assert_eq!(proc.processed, 0);
    }

    #[test]
    fn test_metal_graph_remove_edge() {
        let mut g = MetalGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        assert_eq!(g.edge_count, 2);
        g.remove_edge(0, 1);
        assert_eq!(g.edge_count, 1);
        assert!(!g.has_edge(0, 1));
    }

    #[test]
    fn test_metal_dense_matrix_new() {
        let m = MetalDenseMatrix::new(3, 3);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 3);
    }

    #[test]
    fn test_metal_dense_matrix_identity() {
        let m = MetalDenseMatrix::identity(3);
        assert!((m.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((m.get(0, 1)).abs() < 1e-10);
    }

    #[test]
    fn test_metal_dense_matrix_mul() {
        let a = MetalDenseMatrix::identity(2);
        let b = MetalDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let c = a.mul_matrix(&b);
        assert!((c.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((c.get(1, 1) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_dense_matrix_transpose() {
        let a = MetalDenseMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = a.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert!((t.get(0, 1) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_dense_matrix_det_2x2() {
        let m = MetalDenseMatrix::from_vec(2, 2, vec![3.0, 7.0, 1.0, -4.0]);
        let det = m.determinant_2x2();
        assert!((det - (-19.0)).abs() < 1e-10);
    }

    #[test]
    fn test_metal_dense_matrix_det_3x3() {
        let m = MetalDenseMatrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0]);
        let det = m.determinant_3x3();
        assert!((det - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_dense_matrix_inverse_2x2() {
        let m = MetalDenseMatrix::from_vec(2, 2, vec![4.0, 7.0, 2.0, 6.0]);
        let inv = m.inverse_2x2().unwrap();
        let prod = m.mul_matrix(&inv);
        assert!((prod.get(0, 0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_dense_matrix_power() {
        let m = MetalDenseMatrix::identity(3);
        let p = m.power(5);
        assert!((p.get(0, 0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_dense_matrix_rank() {
        let m = MetalDenseMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0]);
        assert_eq!(m.rank(), 1);
    }

    #[test]
    fn test_metal_dense_matrix_solve() {
        let a = MetalDenseMatrix::from_vec(2, 2, vec![2.0, 1.0, 5.0, 3.0]);
        let x = a.solve(&[4.0, 7.0]).unwrap();
        assert!((x[0] - 5.0).abs() < 1e-8);
        assert!((x[1] - (-6.0)).abs() < 1e-8);
    }

    #[test]
    fn test_metal_dense_matrix_lu() {
        let a = MetalDenseMatrix::from_vec(2, 2, vec![4.0, 3.0, 6.0, 3.0]);
        let (l, u) = a.lu_decomposition();
        let prod = l.mul_matrix(&u);
        assert!((prod.get(0, 0) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_dense_matrix_eigenvalues() {
        let m = MetalDenseMatrix::from_vec(2, 2, vec![2.0, 1.0, 1.0, 2.0]);
        let (e1, e2) = m.eigenvalues_2x2();
        assert!((e1 - 3.0).abs() < 1e-10);
        assert!((e2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_dense_matrix_kronecker() {
        let a = MetalDenseMatrix::identity(2);
        let b = MetalDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let k = a.kronecker_product(&b);
        assert_eq!(k.rows, 4);
        assert_eq!(k.cols, 4);
    }

    #[test]
    fn test_metal_dense_matrix_hadamard() {
        let a = MetalDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = MetalDenseMatrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let h = a.hadamard_product(&b);
        assert!((h.get(0, 0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_interval() {
        let a = MetalInterval::new(1.0, 3.0);
        let b = MetalInterval::new(2.0, 5.0);
        assert!(a.overlaps(&b));
        assert!(a.contains(2.0));
        assert!(!a.contains(4.0));
        let sum = a.add(&b);
        assert!((sum.lo - 3.0).abs() < 1e-10);
        assert!((sum.hi - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_interval_mul() {
        let a = MetalInterval::new(-2.0, 3.0);
        let b = MetalInterval::new(1.0, 4.0);
        let prod = a.mul(&b);
        assert!((prod.lo - (-8.0)).abs() < 1e-10);
        assert!((prod.hi - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_interval_hull() {
        let a = MetalInterval::new(1.0, 3.0);
        let b = MetalInterval::new(5.0, 7.0);
        let h = a.hull(&b);
        assert!((h.lo - 1.0).abs() < 1e-10);
        assert!((h.hi - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_state_machine() {
        let mut sm = MetalStateMachine::new();
        assert_eq!(*sm.state(), MetalState::Allocated);
        assert!(sm.transition(MetalState::Encoded));
        assert_eq!(*sm.state(), MetalState::Encoded);
        assert_eq!(sm.transition_count, 1);
    }

    #[test]
    fn test_metal_state_machine_invalid() {
        let mut sm = MetalStateMachine::new();
        let last_state = MetalState::Completed;
        assert!(!sm.can_transition(&last_state));
    }

    #[test]
    fn test_metal_state_machine_reset() {
        let mut sm = MetalStateMachine::new();
        sm.transition(MetalState::Encoded);
        sm.reset();
        assert_eq!(*sm.state(), MetalState::Allocated);
        assert_eq!(sm.history_len(), 0);
    }

    #[test]
    fn test_metal_ring_buffer() {
        let mut rb = MetalRingBuffer::new(3);
        rb.push(1.0); rb.push(2.0); rb.push(3.0);
        assert!(rb.is_full());
        assert!((rb.average() - 2.0).abs() < 1e-10);
        rb.push(4.0);
        assert!((rb.oldest().unwrap() - 2.0).abs() < 1e-10);
        assert!((rb.latest().unwrap() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_ring_buffer_to_vec() {
        let mut rb = MetalRingBuffer::new(5);
        rb.push(10.0); rb.push(20.0); rb.push(30.0);
        let v = rb.to_vec();
        assert_eq!(v.len(), 3);
        assert!((v[0] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_disjoint_set() {
        let mut ds = MetalDisjointSet::new(5);
        assert_eq!(ds.num_components(), 5);
        ds.union(0, 1);
        ds.union(2, 3);
        assert_eq!(ds.num_components(), 3);
        assert!(ds.connected(0, 1));
        assert!(!ds.connected(0, 2));
    }

    #[test]
    fn test_metal_disjoint_set_components() {
        let mut ds = MetalDisjointSet::new(4);
        ds.union(0, 1); ds.union(2, 3);
        let comps = ds.components();
        assert_eq!(comps.len(), 2);
    }

    #[test]
    fn test_metal_sorted_list() {
        let mut sl = MetalSortedList::new();
        sl.insert(3.0); sl.insert(1.0); sl.insert(2.0);
        assert_eq!(sl.len(), 3);
        assert!((sl.min().unwrap() - 1.0).abs() < 1e-10);
        assert!((sl.max().unwrap() - 3.0).abs() < 1e-10);
        assert!((sl.median() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_sorted_list_remove() {
        let mut sl = MetalSortedList::new();
        sl.insert(1.0); sl.insert(2.0); sl.insert(3.0);
        assert!(sl.remove(2.0));
        assert_eq!(sl.len(), 2);
        assert!(!sl.contains(2.0));
    }

    #[test]
    fn test_metal_ema() {
        let mut ema = MetalEma::new(0.5);
        ema.update(10.0);
        assert!((ema.current() - 10.0).abs() < 1e-10);
        ema.update(20.0);
        assert!((ema.current() - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_bloom_filter() {
        let mut bf = MetalBloomFilter::new(1000, 3);
        bf.insert(42);
        bf.insert(100);
        assert!(bf.may_contain(42));
        assert!(bf.may_contain(100));
        assert_eq!(bf.count(), 2);
    }

    #[test]
    fn test_metal_trie() {
        let mut trie = MetalTrie::new();
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
    fn test_metal_dense_matrix_sym() {
        let m = MetalDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 2.0, 3.0]);
        assert!(m.is_symmetric());
    }

    #[test]
    fn test_metal_dense_matrix_diag() {
        let m = MetalDenseMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 3.0]);
        assert!(m.is_diagonal());
    }

    #[test]
    fn test_metal_dense_matrix_upper_tri() {
        let m = MetalDenseMatrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0]);
        assert!(m.is_upper_triangular());
    }

    #[test]
    fn test_metal_dense_matrix_outer() {
        let m = MetalDenseMatrix::outer_product(&[1.0, 2.0], &[3.0, 4.0]);
        assert!((m.get(0, 0) - 3.0).abs() < 1e-10);
        assert!((m.get(1, 1) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_dense_matrix_submatrix() {
        let m = MetalDenseMatrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let sub = m.submatrix(0, 0, 2, 2);
        assert_eq!(sub.rows, 2);
        assert!((sub.get(1, 1) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_priority_queue() {
        let mut pq = MetalPriorityQueue::new();
        pq.push(3.0, 1); pq.push(1.0, 2); pq.push(2.0, 3);
        assert_eq!(pq.pop().unwrap().1, 2);
        assert_eq!(pq.pop().unwrap().1, 3);
        assert_eq!(pq.pop().unwrap().1, 1);
    }

    #[test]
    fn test_metal_accumulator() {
        let mut acc = MetalAccumulator::new();
        for i in 1..=10 { acc.add(i as f64); }
        assert!((acc.mean() - 5.5).abs() < 1e-10);
        assert_eq!(acc.count(), 10);
        assert!((acc.min() - 1.0).abs() < 1e-10);
        assert!((acc.max() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_accumulator_merge() {
        let mut a = MetalAccumulator::new();
        a.add(1.0); a.add(2.0);
        let mut b = MetalAccumulator::new();
        b.add(3.0); b.add(4.0);
        a.merge(&b);
        assert_eq!(a.count(), 4);
        assert!((a.mean() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_metal_sparse_matrix() {
        let mut m = MetalSparseMatrix::new(3, 3);
        m.insert(0, 1, 2.0); m.insert(1, 2, 3.0);
        assert_eq!(m.nnz(), 2);
        assert!((m.get(0, 1) - 2.0).abs() < 1e-10);
        assert!((m.get(0, 0)).abs() < 1e-10);
    }

    #[test]
    fn test_metal_sparse_mul_vec() {
        let mut m = MetalSparseMatrix::new(2, 2);
        m.insert(0, 0, 1.0); m.insert(1, 1, 2.0);
        let result = m.mul_vec(&[3.0, 4.0]);
        assert!((result[0] - 3.0).abs() < 1e-10);
        assert!((result[1] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_sparse_transpose() {
        let mut m = MetalSparseMatrix::new(2, 3);
        m.insert(0, 2, 5.0);
        let t = m.transpose();
        assert_eq!(t.rows, 3); assert_eq!(t.cols, 2);
        assert!((t.get(2, 0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_polynomial_eval() {
        let p = MetalPolynomial::new(vec![1.0, 2.0, 3.0]);
        assert!((p.evaluate(2.0) - 17.0).abs() < 1e-10);
        assert!((p.evaluate_horner(2.0) - 17.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_polynomial_add() {
        let a = MetalPolynomial::new(vec![1.0, 2.0]);
        let b = MetalPolynomial::new(vec![3.0, 4.0]);
        let c = a.add(&b);
        assert!((c.evaluate(1.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_polynomial_mul() {
        let a = MetalPolynomial::new(vec![1.0, 1.0]);
        let b = MetalPolynomial::new(vec![1.0, 1.0]);
        let c = a.mul(&b);
        assert_eq!(c.degree(), 2);
        assert!((c.evaluate(2.0) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_polynomial_deriv() {
        let p = MetalPolynomial::new(vec![1.0, 2.0, 3.0]);
        let dp = p.derivative();
        assert!((dp.evaluate(0.0) - 2.0).abs() < 1e-10);
        assert!((dp.evaluate(1.0) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_polynomial_integral() {
        let p = MetalPolynomial::new(vec![2.0, 3.0]);
        let ip = p.integral(0.0);
        assert!((ip.evaluate(1.0) - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_metal_polynomial_roots() {
        let p = MetalPolynomial::new(vec![-6.0, 1.0, 1.0]);
        let roots = p.roots_quadratic();
        assert_eq!(roots.len(), 2);
    }

    #[test]
    fn test_metal_polynomial_newton() {
        let p = MetalPolynomial::new(vec![-2.0, 0.0, 1.0]);
        let root = p.newton_root(1.0, 100, 1e-10);
        assert!(root.is_some());
        assert!((root.unwrap() - std::f64::consts::SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn test_metal_polynomial_compose() {
        let p = MetalPolynomial::new(vec![0.0, 0.0, 1.0]);
        let q = MetalPolynomial::new(vec![1.0, 1.0]);
        let r = p.compose(&q);
        assert!((r.evaluate(2.0) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_rng() {
        let mut rng = MetalRng::new(42);
        let v1 = rng.next_u64();
        let v2 = rng.next_u64();
        assert_ne!(v1, v2);
        let f = rng.next_f64();
        assert!(f >= 0.0 && f < 1.0);
    }

    #[test]
    fn test_metal_rng_gaussian() {
        let mut rng = MetalRng::new(123);
        let mut sum = 0.0;
        for _ in 0..1000 { sum += rng.next_gaussian(); }
        let mean = sum / 1000.0;
        assert!(mean.abs() < 0.2);
    }

    #[test]
    fn test_metal_timer() {
        let mut timer = MetalTimer::new("test");
        timer.record(100); timer.record(200); timer.record(300);
        assert_eq!(timer.count(), 3);
        assert_eq!(timer.total_ns(), 600);
        assert!((timer.average_ns() - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_bitvector() {
        let mut bv = MetalBitVector::new(100);
        bv.set(5); bv.set(42); bv.set(99);
        assert!(bv.get(5));
        assert!(bv.get(42));
        assert!(!bv.get(50));
        assert_eq!(bv.count_ones(), 3);
    }

    #[test]
    fn test_metal_bitvector_ops() {
        let mut a = MetalBitVector::new(64);
        a.set(0); a.set(10); a.set(20);
        let mut b = MetalBitVector::new(64);
        b.set(10); b.set(20); b.set(30);
        let c = a.and(&b);
        assert_eq!(c.count_ones(), 2);
        let d = a.or(&b);
        assert_eq!(d.count_ones(), 4);
        assert_eq!(a.hamming_distance(&b), 2);
    }

    #[test]
    fn test_metal_bitvector_jaccard() {
        let mut a = MetalBitVector::new(10);
        a.set(0); a.set(1); a.set(2);
        let mut b = MetalBitVector::new(10);
        b.set(1); b.set(2); b.set(3);
        let j = a.jaccard(&b);
        assert!((j - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_metal_priority_queue_empty() {
        let mut pq = MetalPriorityQueue::new();
        assert!(pq.is_empty());
        assert!(pq.pop().is_none());
    }

    #[test]
    fn test_metal_sparse_add() {
        let mut a = MetalSparseMatrix::new(2, 2);
        a.insert(0, 0, 1.0);
        let mut b = MetalSparseMatrix::new(2, 2);
        b.insert(0, 0, 2.0); b.insert(1, 1, 3.0);
        let c = a.add(&b);
        assert!((c.get(0, 0) - 3.0).abs() < 1e-10);
        assert!((c.get(1, 1) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_rng_shuffle() {
        let mut rng = MetalRng::new(99);
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        rng.shuffle(&mut data);
        assert_eq!(data.len(), 5);
        let sum: f64 = data.iter().sum();
        assert!((sum - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_polynomial_display() {
        let p = MetalPolynomial::new(vec![1.0, 2.0, 3.0]);
        let s = format!("{}", p);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_metal_polynomial_monomial() {
        let m = MetalPolynomial::monomial(3, 5.0);
        assert_eq!(m.degree(), 3);
        assert!((m.evaluate(2.0) - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_timer_percentiles() {
        let mut timer = MetalTimer::new("perf");
        for i in 1..=100 { timer.record(i); }
        assert_eq!(timer.p50_ns(), 50);
        assert!(timer.p95_ns() >= 90);
    }

    #[test]
    fn test_metal_accumulator_cv() {
        let mut acc = MetalAccumulator::new();
        acc.add(10.0); acc.add(10.0); acc.add(10.0);
        assert!(acc.coefficient_of_variation().abs() < 1e-10);
    }

    #[test]
    fn test_metal_sparse_diagonal() {
        let mut m = MetalSparseMatrix::new(3, 3);
        m.insert(0, 0, 1.0); m.insert(1, 1, 2.0); m.insert(2, 2, 3.0);
        assert!((m.trace() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_lru_cache() {
        let mut cache = MetalLruCache::new(2);
        cache.put(1, vec![1.0]);
        cache.put(2, vec![2.0]);
        assert!(cache.get(1).is_some());
        cache.put(3, vec![3.0]);
        assert!(cache.get(2).is_none());
        assert!(cache.get(3).is_some());
    }

    #[test]
    fn test_metal_lru_hit_rate() {
        let mut cache = MetalLruCache::new(10);
        cache.put(1, vec![1.0]);
        cache.get(1);
        cache.get(2);
        assert!((cache.hit_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_metal_graph_coloring() {
        let mut gc = MetalGraphColoring::new(4);
        gc.add_edge(0, 1); gc.add_edge(1, 2); gc.add_edge(2, 3); gc.add_edge(3, 0);
        let colors = gc.greedy_color();
        assert!(gc.is_valid_coloring());
        assert!(colors <= 3);
    }

    #[test]
    fn test_metal_graph_coloring_complete() {
        let mut gc = MetalGraphColoring::new(3);
        gc.add_edge(0, 1); gc.add_edge(1, 2); gc.add_edge(0, 2);
        let colors = gc.greedy_color();
        assert_eq!(colors, 3);
        assert!(gc.is_valid_coloring());
    }

    #[test]
    fn test_metal_topk() {
        let mut tk = MetalTopK::new(3);
        tk.insert(5.0, "e"); tk.insert(3.0, "c"); tk.insert(1.0, "a");
        tk.insert(4.0, "d"); tk.insert(2.0, "b");
        assert_eq!(tk.len(), 3);
        assert!((tk.max_score().unwrap() - 5.0).abs() < 1e-10);
        assert!((tk.min_score().unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_sliding_window() {
        let mut sw = MetalSlidingWindow::new(3);
        sw.push(1.0); sw.push(2.0); sw.push(3.0);
        assert!((sw.mean() - 2.0).abs() < 1e-10);
        sw.push(4.0);
        assert!((sw.mean() - 3.0).abs() < 1e-10);
        assert_eq!(sw.len(), 3);
    }

    #[test]
    fn test_metal_sliding_window_trend() {
        let mut sw = MetalSlidingWindow::new(10);
        for i in 0..5 { sw.push(i as f64); }
        assert!(sw.trend() > 0.0);
    }

    #[test]
    fn test_metal_confusion_matrix() {
        let actual = vec![true, true, false, false, true];
        let predicted = vec![true, false, false, true, true];
        let cm = MetalConfusionMatrix::from_predictions(&actual, &predicted);
        assert_eq!(cm.true_positive, 2);
        assert_eq!(cm.false_positive, 1);
        assert_eq!(cm.true_negative, 1);
        assert_eq!(cm.false_negative, 1);
        assert_eq!(cm.total(), 5);
    }

    #[test]
    fn test_metal_confusion_f1() {
        let cm = MetalConfusionMatrix { true_positive: 80, false_positive: 20, true_negative: 70, false_negative: 30 };
        assert!((cm.precision() - 0.8).abs() < 1e-10);
        let f1 = cm.f1_score();
        assert!(f1 > 0.0 && f1 < 1.0);
    }

    #[test]
    fn test_metal_cosine_similarity() {
        let s = metal_cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(s.abs() < 1e-10);
        let s2 = metal_cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]);
        assert!((s2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_euclidean_distance() {
        let d = metal_euclidean_distance(&[0.0, 0.0], &[3.0, 4.0]);
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_sigmoid() {
        let s = metal_sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-10);
        let s2 = metal_sigmoid(100.0);
        assert!((s2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_metal_softmax() {
        let sm = metal_softmax(&[1.0, 2.0, 3.0]);
        let sum: f64 = sm.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(sm[2] > sm[1] && sm[1] > sm[0]);
    }

    #[test]
    fn test_metal_kl_divergence() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let kl = metal_kl_divergence(&p, &q);
        assert!(kl.abs() < 1e-10);
    }

    #[test]
    fn test_metal_normalize() {
        let v = metal_normalize(&[3.0, 4.0]);
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_lerp() {
        assert!((metal_lerp(0.0, 10.0, 0.5) - 5.0).abs() < 1e-10);
        assert!((metal_lerp(0.0, 10.0, 0.0) - 0.0).abs() < 1e-10);
        assert!((metal_lerp(0.0, 10.0, 1.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_clamp() {
        assert!((metal_clamp(5.0, 0.0, 10.0) - 5.0).abs() < 1e-10);
        assert!((metal_clamp(-5.0, 0.0, 10.0) - 0.0).abs() < 1e-10);
        assert!((metal_clamp(15.0, 0.0, 10.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_cross_product() {
        let c = metal_cross_product(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]);
        assert!((c[0]).abs() < 1e-10);
        assert!((c[1]).abs() < 1e-10);
        assert!((c[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_dot_product() {
        let d = metal_dot_product(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        assert!((d - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_js_divergence() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let js = metal_js_divergence(&p, &q);
        assert!(js.abs() < 1e-10);
    }

    #[test]
    fn test_metal_hellinger() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let h = metal_hellinger_distance(&p, &q);
        assert!(h.abs() < 1e-10);
    }

    #[test]
    fn test_metal_logsumexp() {
        let lse = metal_logsumexp(&[1.0, 2.0, 3.0]);
        assert!(lse > 3.0);
    }

    #[test]
    fn test_metal_feature_scaler() {
        let mut scaler = MetalFeatureScaler::new();
        let data = vec![vec![1.0, 10.0], vec![2.0, 20.0], vec![3.0, 30.0]];
        scaler.fit(&data);
        let normalized = scaler.normalize(&[2.0, 20.0]);
        assert!((normalized[0] - 0.5).abs() < 1e-10);
        assert!((normalized[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_metal_feature_scaler_inverse() {
        let mut scaler = MetalFeatureScaler::new();
        let data = vec![vec![0.0, 0.0], vec![10.0, 100.0]];
        scaler.fit(&data);
        let normed = scaler.normalize(&[5.0, 50.0]);
        let inv = scaler.inverse_normalize(&normed);
        assert!((inv[0] - 5.0).abs() < 1e-10);
        assert!((inv[1] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_linear_regression() {
        let mut lr = MetalLinearRegression::new();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        lr.fit(&x, &y);
        assert!((lr.slope - 2.0).abs() < 1e-10);
        assert!(lr.intercept.abs() < 1e-10);
        assert!((lr.r_squared - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_linear_regression_predict() {
        let mut lr = MetalLinearRegression::new();
        lr.fit(&[0.0, 1.0, 2.0], &[1.0, 3.0, 5.0]);
        assert!((lr.predict(3.0) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_weighted_graph() {
        let mut g = MetalWeightedGraph::new(4);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 2.0); g.add_edge(2, 3, 3.0);
        assert_eq!(g.num_edges, 3);
        let dists = g.dijkstra(0);
        assert!((dists[3] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_weighted_graph_mst() {
        let mut g = MetalWeightedGraph::new(4);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 2.0); g.add_edge(2, 3, 3.0);
        g.add_edge(0, 3, 10.0);
        let mst = g.min_spanning_tree_weight();
        assert!((mst - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_moving_average() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ma = metal_moving_average(&data, 3);
        assert_eq!(ma.len(), 3);
        assert!((ma[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_cumsum() {
        let cs = metal_cumsum(&[1.0, 2.0, 3.0, 4.0]);
        assert!((cs[3] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_diff() {
        let d = metal_diff(&[1.0, 3.0, 6.0, 10.0]);
        assert_eq!(d.len(), 3);
        assert!((d[0] - 2.0).abs() < 1e-10);
        assert!((d[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_autocorrelation() {
        let ac = metal_autocorrelation(&[1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0], 0);
        assert!((ac - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_dft_magnitude() {
        let mags = metal_dft_magnitude(&[1.0, 0.0, -1.0, 0.0]);
        assert!(!mags.is_empty());
    }

    #[test]
    fn test_metal_integrate_trapezoid() {
        let area = metal_integrate_trapezoid(&[0.0, 1.0, 2.0], &[0.0, 1.0, 0.0]);
        assert!((area - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_convolve() {
        let c = metal_convolve(&[1.0, 2.0, 3.0], &[1.0, 1.0]);
        assert_eq!(c.len(), 4);
        assert!((c[0] - 1.0).abs() < 1e-10);
        assert!((c[3] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_weighted_graph_clustering() {
        let mut g = MetalWeightedGraph::new(4);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 1.0); g.add_edge(0, 2, 1.0);
        let cc = g.clustering_coefficient(0);
        assert!((cc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_histogram_cumulative() {
        let h = MetalHistogramExt::from_data(&[1.0, 1.0, 2.0, 3.0, 3.0, 3.0], 3);
        let cum = h.cumulative();
        assert_eq!(*cum.last().unwrap(), 6);
    }

    #[test]
    fn test_metal_histogram_entropy() {
        let h = MetalHistogramExt::from_data(&[1.0, 2.0, 3.0, 4.0], 4);
        let ent = h.entropy();
        assert!(ent > 0.0);
    }

    #[test]
    fn test_metal_aabb() {
        let bb = MetalAABB::new(0.0, 0.0, 10.0, 10.0);
        assert!(bb.contains(5.0, 5.0));
        assert!(!bb.contains(11.0, 5.0));
        assert!((bb.area() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_aabb_intersects() {
        let a = MetalAABB::new(0.0, 0.0, 10.0, 10.0);
        let b = MetalAABB::new(5.0, 5.0, 15.0, 15.0);
        let c = MetalAABB::new(20.0, 20.0, 30.0, 30.0);
        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn test_metal_quadtree() {
        let bb = MetalAABB::new(0.0, 0.0, 100.0, 100.0);
        let mut qt = MetalQuadTree::new(bb, 4, 8);
        for i in 0..20 {
            qt.insert(MetalPoint2D { x: i as f64 * 5.0, y: i as f64 * 5.0, data: i as f64 });
        }
        assert_eq!(qt.count(), 20);
    }

    #[test]
    fn test_metal_quadtree_query() {
        let bb = MetalAABB::new(0.0, 0.0, 100.0, 100.0);
        let mut qt = MetalQuadTree::new(bb, 2, 8);
        qt.insert(MetalPoint2D { x: 10.0, y: 10.0, data: 1.0 });
        qt.insert(MetalPoint2D { x: 90.0, y: 90.0, data: 2.0 });
        let range = MetalAABB::new(0.0, 0.0, 50.0, 50.0);
        let found = qt.query_range(&range);
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn test_metal_mat_mul() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let c = metal_mat_mul(&a, &b);
        assert!((c[0][0] - 19.0).abs() < 1e-10);
        assert!((c[1][1] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_transpose() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let t = metal_transpose(&a);
        assert_eq!(t.len(), 3);
        assert_eq!(t[0].len(), 2);
        assert!((t[2][1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_frobenius_norm() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let n = metal_frobenius_norm(&a);
        assert!((n - 2.0f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_metal_trace() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert!((metal_trace(&a) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_identity() {
        let id = metal_identity(3);
        assert!((id[0][0] - 1.0).abs() < 1e-10);
        assert!((id[0][1]).abs() < 1e-10);
        assert!((id[2][2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_power_iteration() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let (eval, _evec) = metal_power_iteration(&a, 100);
        assert!((eval - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_metal_running_stats() {
        let mut s = MetalRunningStats::new();
        for &v in &[1.0, 2.0, 3.0, 4.0, 5.0] { s.push(v); }
        assert!((s.mean - 3.0).abs() < 1e-10);
        assert!((s.min_val - 1.0).abs() < 1e-10);
        assert!((s.max_val - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_running_stats_merge() {
        let mut a = MetalRunningStats::new();
        let mut b = MetalRunningStats::new();
        for &v in &[1.0, 2.0, 3.0] { a.push(v); }
        for &v in &[4.0, 5.0, 6.0] { b.push(v); }
        a.merge(&b);
        assert_eq!(a.count, 6);
        assert!((a.mean - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_metal_running_stats_cv() {
        let mut s = MetalRunningStats::new();
        for &v in &[10.0, 10.0, 10.0] { s.push(v); }
        assert!(s.coefficient_of_variation() < 1e-10);
    }

    #[test]
    fn test_metal_iqr() {
        let iqr = metal_iqr(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert!(iqr > 0.0);
    }

    #[test]
    fn test_metal_outliers() {
        let outliers = metal_outliers(&[1.0, 2.0, 3.0, 4.0, 5.0, 100.0]);
        assert!(!outliers.is_empty());
    }

    #[test]
    fn test_metal_zscore() {
        let z = metal_zscore(&[10.0, 20.0, 30.0]);
        assert!((z[1]).abs() < 1e-10); // middle value should be ~0
    }

    #[test]
    fn test_metal_rank() {
        let r = metal_rank(&[30.0, 10.0, 20.0]);
        assert!((r[0] - 3.0).abs() < 1e-10);
        assert!((r[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_spearman() {
        let rho = metal_spearman(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((rho - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_metal_sample_skewness_symmetric() {
        let s = metal_sample_skewness(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(s.abs() < 1e-10);
    }

    #[test]
    fn test_metal_covariance_matrix() {
        let data = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]];
        let cov = metal_covariance_matrix(&data);
        assert_eq!(cov.len(), 2);
        assert!(cov[0][0] > 0.0);
    }

    #[test]
    fn test_metal_correlation_matrix() {
        let data = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]];
        let corr = metal_correlation_matrix(&data);
        assert!((corr[0][1] - 1.0).abs() < 1e-10); // perfect correlation
    }

}

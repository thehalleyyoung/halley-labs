#![allow(unused)]

use std::collections::{HashMap, HashSet, BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use serde::{Serialize, Deserialize};

// ---------------------------------------------------------------------------
// Memory Region Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryRegionKind {
    Global,
    Shared,
    Local,
    Constant,
    Texture,
    Private,
}

impl fmt::Display for MemoryRegionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Global => write!(f, "global"),
            Self::Shared => write!(f, "shared"),
            Self::Local => write!(f, "local"),
            Self::Constant => write!(f, "constant"),
            Self::Texture => write!(f, "texture"),
            Self::Private => write!(f, "private"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemoryRegionFlags {
    pub readable: bool,
    pub writable: bool,
    pub executable: bool,
    pub cacheable: bool,
    pub coherent: bool,
}

impl MemoryRegionFlags {
    pub fn read_write() -> Self {
        Self { readable: true, writable: true, executable: false, cacheable: true, coherent: false }
    }
    pub fn read_only() -> Self {
        Self { readable: true, writable: false, executable: false, cacheable: true, coherent: false }
    }
    pub fn all() -> Self {
        Self { readable: true, writable: true, executable: true, cacheable: true, coherent: true }
    }
}

impl Default for MemoryRegionFlags {
    fn default() -> Self { Self::read_write() }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemoryRegion {
    pub id: String,
    pub base: u64,
    pub size: u64,
    pub flags: MemoryRegionFlags,
    pub kind: MemoryRegionKind,
}

impl MemoryRegion {
    pub fn new(id: impl Into<String>, base: u64, size: u64, kind: MemoryRegionKind) -> Self {
        Self { id: id.into(), base, size, flags: MemoryRegionFlags::default(), kind }
    }

    pub fn with_flags(mut self, flags: MemoryRegionFlags) -> Self {
        self.flags = flags;
        self
    }

    pub fn end(&self) -> u64 {
        self.base.saturating_add(self.size)
    }

    pub fn contains(&self, addr: u64) -> bool {
        addr >= self.base && addr < self.end()
    }

    pub fn contains_range(&self, addr: u64, len: u64) -> bool {
        addr >= self.base && addr.saturating_add(len) <= self.end()
    }

    pub fn overlaps(&self, other: &MemoryRegion) -> bool {
        self.base < other.end() && other.base < self.end()
    }

    pub fn overlap_size(&self, other: &MemoryRegion) -> u64 {
        if !self.overlaps(other) {
            return 0;
        }
        let start = self.base.max(other.base);
        let end = self.end().min(other.end());
        end - start
    }
}

impl fmt::Display for MemoryRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[0x{:x}..0x{:x}]({})", self.id, self.base, self.end(), self.kind)
    }
}

// ---------------------------------------------------------------------------
// Memory Layout
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLayout {
    pub regions: Vec<MemoryRegion>,
    index: BTreeMap<u64, usize>,
}

impl MemoryLayout {
    pub fn new() -> Self {
        Self { regions: Vec::new(), index: BTreeMap::new() }
    }

    pub fn add_region(&mut self, region: MemoryRegion) -> Result<(), String> {
        for existing in &self.regions {
            if existing.overlaps(&region) {
                return Err(format!("Region {} overlaps with {}", region, existing));
            }
        }
        let idx = self.regions.len();
        self.index.insert(region.base, idx);
        self.regions.push(region);
        Ok(())
    }

    pub fn find_region(&self, addr: u64) -> Option<&MemoryRegion> {
        // Binary search via BTreeMap: find the region whose base <= addr
        if let Some((&base, &idx)) = self.index.range(..=addr).next_back() {
            let region = &self.regions[idx];
            if region.contains(addr) {
                return Some(region);
            }
        }
        None
    }

    pub fn find_region_for_range(&self, addr: u64, size: u64) -> Option<&MemoryRegion> {
        if let Some(region) = self.find_region(addr) {
            if region.contains_range(addr, size) {
                return Some(region);
            }
        }
        None
    }

    pub fn check_overlap(&self) -> Vec<(usize, usize)> {
        let mut overlaps = Vec::new();
        for i in 0..self.regions.len() {
            for j in (i + 1)..self.regions.len() {
                if self.regions[i].overlaps(&self.regions[j]) {
                    overlaps.push((i, j));
                }
            }
        }
        overlaps
    }

    pub fn total_size(&self) -> u64 {
        self.regions.iter().map(|r| r.size).sum()
    }

    pub fn regions_of_kind(&self, kind: MemoryRegionKind) -> Vec<&MemoryRegion> {
        self.regions.iter().filter(|r| r.kind == kind).collect()
    }

    pub fn merge_adjacent(&mut self) {
        if self.regions.len() < 2 {
            return;
        }
        self.regions.sort_by_key(|r| r.base);
        let mut merged = Vec::new();
        let mut current = self.regions[0].clone();
        for region in self.regions.iter().skip(1) {
            if current.end() == region.base && current.kind == region.kind && current.flags == region.flags {
                current.size += region.size;
            } else {
                merged.push(current);
                current = region.clone();
            }
        }
        merged.push(current);
        self.regions = merged;
        self.rebuild_index();
    }

    fn rebuild_index(&mut self) {
        self.index.clear();
        for (i, region) in self.regions.iter().enumerate() {
            self.index.insert(region.base, i);
        }
    }
}

// ---------------------------------------------------------------------------
// Buffer Access & Overflow Detection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferAccess {
    pub buffer_id: String,
    pub offset: u64,
    pub size: u64,
    pub is_write: bool,
    pub thread_id: u32,
    pub timestamp: u64,
    pub source_location: Option<String>,
}

impl BufferAccess {
    pub fn read(buffer_id: impl Into<String>, offset: u64, size: u64) -> Self {
        Self {
            buffer_id: buffer_id.into(),
            offset,
            size,
            is_write: false,
            thread_id: 0,
            timestamp: 0,
            source_location: None,
        }
    }

    pub fn write(buffer_id: impl Into<String>, offset: u64, size: u64) -> Self {
        Self {
            buffer_id: buffer_id.into(),
            offset,
            size,
            is_write: true,
            thread_id: 0,
            timestamp: 0,
            source_location: None,
        }
    }

    pub fn with_thread(mut self, tid: u32) -> Self {
        self.thread_id = tid;
        self
    }

    pub fn with_timestamp(mut self, ts: u64) -> Self {
        self.timestamp = ts;
        self
    }

    pub fn with_source(mut self, loc: impl Into<String>) -> Self {
        self.source_location = Some(loc.into());
        self
    }

    pub fn end_offset(&self) -> u64 {
        self.offset.saturating_add(self.size)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverflowViolation {
    pub access: BufferAccess,
    pub buffer_size: u64,
    pub overflow_bytes: u64,
    pub description: String,
}

impl fmt::Display for OverflowViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Buffer overflow: access [{},{}] exceeds buffer size {} by {} bytes on buffer '{}'",
            self.access.offset,
            self.access.end_offset(),
            self.buffer_size,
            self.overflow_bytes,
            self.access.buffer_id,
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverflowReport {
    pub violations: Vec<OverflowViolation>,
    pub total_accesses: usize,
    pub safe_accesses: usize,
}

impl OverflowReport {
    pub fn is_safe(&self) -> bool {
        self.violations.is_empty()
    }

    pub fn violation_rate(&self) -> f64 {
        if self.total_accesses == 0 {
            return 0.0;
        }
        self.violations.len() as f64 / self.total_accesses as f64
    }
}

#[derive(Debug, Clone)]
pub struct OverflowDetector {
    buffers: HashMap<String, u64>,
}

impl OverflowDetector {
    pub fn new() -> Self {
        Self { buffers: HashMap::new() }
    }

    pub fn register_buffer(&mut self, id: impl Into<String>, size: u64) {
        self.buffers.insert(id.into(), size);
    }

    pub fn check_access(&self, access: &BufferAccess) -> Option<OverflowViolation> {
        if let Some(&buf_size) = self.buffers.get(&access.buffer_id) {
            let end = access.offset.saturating_add(access.size);
            if end > buf_size {
                let overflow = end - buf_size;
                return Some(OverflowViolation {
                    access: access.clone(),
                    buffer_size: buf_size,
                    overflow_bytes: overflow,
                    description: format!(
                        "Access at offset {} with size {} overflows buffer '{}' (size {})",
                        access.offset, access.size, access.buffer_id, buf_size
                    ),
                });
            }
            if access.offset >= buf_size {
                return Some(OverflowViolation {
                    access: access.clone(),
                    buffer_size: buf_size,
                    overflow_bytes: access.size,
                    description: format!(
                        "Access at offset {} is completely out of bounds for buffer '{}' (size {})",
                        access.offset, access.buffer_id, buf_size
                    ),
                });
            }
            None
        } else {
            Some(OverflowViolation {
                access: access.clone(),
                buffer_size: 0,
                overflow_bytes: access.size,
                description: format!("Unknown buffer '{}'", access.buffer_id),
            })
        }
    }

    pub fn batch_check(&self, accesses: &[BufferAccess]) -> OverflowReport {
        let mut violations = Vec::new();
        let mut safe = 0usize;
        for access in accesses {
            if let Some(v) = self.check_access(access) {
                violations.push(v);
            } else {
                safe += 1;
            }
        }
        OverflowReport {
            total_accesses: accesses.len(),
            safe_accesses: safe,
            violations,
        }
    }
}

// ---------------------------------------------------------------------------
// Use-After-Free Detection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AllocationState {
    Allocated,
    Freed,
    Reallocated,
}

impl fmt::Display for AllocationState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Allocated => write!(f, "allocated"),
            Self::Freed => write!(f, "freed"),
            Self::Reallocated => write!(f, "reallocated"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationRecord {
    pub id: String,
    pub base: u64,
    pub size: u64,
    pub state: AllocationState,
    pub alloc_timestamp: u64,
    pub free_timestamp: Option<u64>,
    pub realloc_timestamps: Vec<u64>,
    pub thread_id: u32,
}

#[derive(Debug, Clone)]
pub struct LifetimeTracker {
    allocations: HashMap<String, AllocationRecord>,
    freed_addresses: HashMap<u64, Vec<String>>,
    current_time: u64,
}

impl LifetimeTracker {
    pub fn new() -> Self {
        Self {
            allocations: HashMap::new(),
            freed_addresses: HashMap::new(),
            current_time: 0,
        }
    }

    pub fn allocate(&mut self, id: impl Into<String>, base: u64, size: u64, thread_id: u32) {
        let id = id.into();
        self.current_time += 1;
        let record = AllocationRecord {
            id: id.clone(),
            base,
            size,
            state: AllocationState::Allocated,
            alloc_timestamp: self.current_time,
            free_timestamp: None,
            realloc_timestamps: Vec::new(),
            thread_id,
        };
        self.allocations.insert(id, record);
    }

    pub fn free(&mut self, id: &str) -> Result<(), String> {
        self.current_time += 1;
        if let Some(record) = self.allocations.get_mut(id) {
            if record.state == AllocationState::Freed {
                return Err(format!("Double free of allocation '{}'", id));
            }
            record.state = AllocationState::Freed;
            record.free_timestamp = Some(self.current_time);
            self.freed_addresses
                .entry(record.base)
                .or_insert_with(Vec::new)
                .push(id.to_string());
            Ok(())
        } else {
            Err(format!("Unknown allocation '{}'", id))
        }
    }

    pub fn reallocate(&mut self, id: &str, new_base: u64, new_size: u64) -> Result<(), String> {
        self.current_time += 1;
        if let Some(record) = self.allocations.get_mut(id) {
            if record.state == AllocationState::Freed {
                return Err(format!("Realloc of freed allocation '{}'", id));
            }
            record.state = AllocationState::Reallocated;
            record.base = new_base;
            record.size = new_size;
            record.realloc_timestamps.push(self.current_time);
            Ok(())
        } else {
            Err(format!("Unknown allocation '{}'", id))
        }
    }

    pub fn get_state(&self, id: &str) -> Option<AllocationState> {
        self.allocations.get(id).map(|r| r.state)
    }

    pub fn is_allocated(&self, id: &str) -> bool {
        matches!(self.get_state(id), Some(AllocationState::Allocated) | Some(AllocationState::Reallocated))
    }

    pub fn all_allocated(&self) -> Vec<&AllocationRecord> {
        self.allocations.values().filter(|r| r.state != AllocationState::Freed).collect()
    }

    pub fn all_freed(&self) -> Vec<&AllocationRecord> {
        self.allocations.values().filter(|r| r.state == AllocationState::Freed).collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UseAfterFreeViolation {
    pub allocation_id: String,
    pub access_offset: u64,
    pub access_size: u64,
    pub free_timestamp: u64,
    pub access_timestamp: u64,
    pub thread_id: u32,
    pub description: String,
}

impl fmt::Display for UseAfterFreeViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Use-after-free: allocation '{}' freed at t={}, accessed at t={} by thread {}",
            self.allocation_id, self.free_timestamp, self.access_timestamp, self.thread_id
        )
    }
}

#[derive(Debug, Clone)]
pub struct UseAfterFreeDetector {
    tracker: LifetimeTracker,
    violations: Vec<UseAfterFreeViolation>,
}

impl UseAfterFreeDetector {
    pub fn new(tracker: LifetimeTracker) -> Self {
        Self { tracker, violations: Vec::new() }
    }

    pub fn check_use(&mut self, alloc_id: &str, offset: u64, size: u64, thread_id: u32, timestamp: u64) -> Option<UseAfterFreeViolation> {
        if let Some(record) = self.tracker.allocations.get(alloc_id) {
            if record.state == AllocationState::Freed {
                let violation = UseAfterFreeViolation {
                    allocation_id: alloc_id.to_string(),
                    access_offset: offset,
                    access_size: size,
                    free_timestamp: record.free_timestamp.unwrap_or(0),
                    access_timestamp: timestamp,
                    thread_id,
                    description: format!(
                        "Access to freed allocation '{}' at offset {} (size {})",
                        alloc_id, offset, size
                    ),
                };
                self.violations.push(violation.clone());
                return Some(violation);
            }
        }
        None
    }

    pub fn violations(&self) -> &[UseAfterFreeViolation] {
        &self.violations
    }
}

pub fn dangling_pointer_check(tracker: &LifetimeTracker, pointers: &[(String, u64)]) -> Vec<String> {
    let mut dangling = Vec::new();
    for (alloc_id, _addr) in pointers {
        if let Some(record) = tracker.allocations.get(alloc_id) {
            if record.state == AllocationState::Freed {
                dangling.push(format!("Dangling pointer to freed allocation '{}'", alloc_id));
            }
        } else {
            dangling.push(format!("Dangling pointer to unknown allocation '{}'", alloc_id));
        }
    }
    dangling
}

pub fn double_free_check(tracker: &LifetimeTracker) -> Vec<String> {
    let mut issues = Vec::new();
    for (addr, ids) in &tracker.freed_addresses {
        if ids.len() > 1 {
            issues.push(format!(
                "Address 0x{:x} freed multiple times via allocations: {:?}",
                addr, ids
            ));
        }
    }
    // Also check for same-id freed twice via realloc timestamps
    for record in tracker.allocations.values() {
        if record.state == AllocationState::Freed && record.realloc_timestamps.len() > 0 {
            if let Some(free_ts) = record.free_timestamp {
                for &realloc_ts in &record.realloc_timestamps {
                    if realloc_ts > free_ts {
                        issues.push(format!(
                            "Allocation '{}' reallocated at t={} after being freed at t={}",
                            record.id, realloc_ts, free_ts
                        ));
                    }
                }
            }
        }
    }
    issues
}

// ---------------------------------------------------------------------------
// Data Race Detection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAccess {
    pub address: u64,
    pub size: u64,
    pub thread_id: u32,
    pub is_write: bool,
    pub timestamp: u64,
    pub source_location: Option<String>,
}

impl MemoryAccess {
    pub fn new(address: u64, size: u64, thread_id: u32, is_write: bool, timestamp: u64) -> Self {
        Self { address, size, thread_id, is_write, timestamp, source_location: None }
    }

    pub fn end_address(&self) -> u64 {
        self.address.saturating_add(self.size)
    }

    pub fn overlaps_with(&self, other: &MemoryAccess) -> bool {
        self.address < other.end_address() && other.address < self.end_address()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RacePair {
    pub access1: MemoryAccess,
    pub access2: MemoryAccess,
    pub shared_range_start: u64,
    pub shared_range_end: u64,
    pub description: String,
}

impl RacePair {
    pub fn new(a1: MemoryAccess, a2: MemoryAccess) -> Self {
        let start = a1.address.max(a2.address);
        let end = a1.end_address().min(a2.end_address());
        let desc = format!(
            "Race between thread {} ({}) and thread {} ({}) on [0x{:x}..0x{:x}]",
            a1.thread_id,
            if a1.is_write { "W" } else { "R" },
            a2.thread_id,
            if a2.is_write { "W" } else { "R" },
            start,
            end,
        );
        Self {
            access1: a1,
            access2: a2,
            shared_range_start: start,
            shared_range_end: end,
            description: desc,
        }
    }
}

impl fmt::Display for RacePair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description)
    }
}

#[derive(Debug, Clone)]
pub struct DataRaceDetector {
    accesses: Vec<MemoryAccess>,
    hb_graph: HappensBeforeGraph,
}

impl DataRaceDetector {
    pub fn new() -> Self {
        Self { accesses: Vec::new(), hb_graph: HappensBeforeGraph::new() }
    }

    pub fn add_access(&mut self, access: MemoryAccess) {
        self.accesses.push(access);
    }

    pub fn add_happens_before(&mut self, before: u64, after: u64) {
        self.hb_graph.add_edge(before, after);
    }

    pub fn detect_races(&self) -> Vec<RacePair> {
        let mut races = Vec::new();
        let closure = self.hb_graph.transitive_closure();

        for i in 0..self.accesses.len() {
            for j in (i + 1)..self.accesses.len() {
                let a1 = &self.accesses[i];
                let a2 = &self.accesses[j];

                // Must be different threads
                if a1.thread_id == a2.thread_id {
                    continue;
                }

                // At least one must be a write
                if !a1.is_write && !a2.is_write {
                    continue;
                }

                // Must access overlapping memory
                if !a1.overlaps_with(a2) {
                    continue;
                }

                // Must not be ordered by happens-before
                if closure.is_ordered(a1.timestamp, a2.timestamp) {
                    continue;
                }

                races.push(RacePair::new(a1.clone(), a2.clone()));
            }
        }
        races
    }

    pub fn detect_races_on_address(&self, target_addr: u64) -> Vec<RacePair> {
        let relevant: Vec<&MemoryAccess> = self.accesses.iter()
            .filter(|a| a.address <= target_addr && a.end_address() > target_addr)
            .collect();
        let closure = self.hb_graph.transitive_closure();
        let mut races = Vec::new();

        for i in 0..relevant.len() {
            for j in (i + 1)..relevant.len() {
                let a1 = relevant[i];
                let a2 = relevant[j];
                if a1.thread_id == a2.thread_id { continue; }
                if !a1.is_write && !a2.is_write { continue; }
                if !closure.is_ordered(a1.timestamp, a2.timestamp) {
                    races.push(RacePair::new(a1.clone(), a2.clone()));
                }
            }
        }
        races
    }
}

// ---------------------------------------------------------------------------
// Happens-Before Graph
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct HappensBeforeGraph {
    edges: HashMap<u64, HashSet<u64>>,
    nodes: HashSet<u64>,
}

impl HappensBeforeGraph {
    pub fn new() -> Self {
        Self { edges: HashMap::new(), nodes: HashSet::new() }
    }

    pub fn add_edge(&mut self, before: u64, after: u64) {
        self.nodes.insert(before);
        self.nodes.insert(after);
        self.edges.entry(before).or_insert_with(HashSet::new).insert(after);
    }

    pub fn is_ordered(&self, a: u64, b: u64) -> bool {
        // BFS from a to b
        if a == b { return true; }
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(a);
        visited.insert(a);
        while let Some(node) = queue.pop_front() {
            if node == b { return true; }
            if let Some(successors) = self.edges.get(&node) {
                for &succ in successors {
                    if !visited.contains(&succ) {
                        visited.insert(succ);
                        queue.push_back(succ);
                    }
                }
            }
        }
        // Also check b -> a
        let mut visited2 = HashSet::new();
        let mut queue2 = VecDeque::new();
        queue2.push_back(b);
        visited2.insert(b);
        while let Some(node) = queue2.pop_front() {
            if node == a { return true; }
            if let Some(successors) = self.edges.get(&node) {
                for &succ in successors {
                    if !visited2.contains(&succ) {
                        visited2.insert(succ);
                        queue2.push_back(succ);
                    }
                }
            }
        }
        false
    }

    pub fn transitive_closure(&self) -> TransitiveClosure {
        let node_list: Vec<u64> = self.nodes.iter().copied().collect();
        let node_index: HashMap<u64, usize> = node_list.iter().enumerate().map(|(i, &n)| (n, i)).collect();
        let n = node_list.len();
        let mut matrix = vec![vec![false; n]; n];

        // Initialize direct edges
        for (&from, tos) in &self.edges {
            if let Some(&fi) = node_index.get(&from) {
                for &to in tos {
                    if let Some(&ti) = node_index.get(&to) {
                        matrix[fi][ti] = true;
                    }
                }
            }
        }

        // Self-loops
        for i in 0..n {
            matrix[i][i] = true;
        }

        // Floyd-Warshall
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if matrix[i][k] && matrix[k][j] {
                        matrix[i][j] = true;
                    }
                }
            }
        }

        TransitiveClosure { node_index, matrix }
    }

    pub fn topological_sort(&self) -> Option<Vec<u64>> {
        let mut in_degree: HashMap<u64, usize> = HashMap::new();
        for &node in &self.nodes {
            in_degree.entry(node).or_insert(0);
        }
        for (_, successors) in &self.edges {
            for &succ in successors {
                *in_degree.entry(succ).or_insert(0) += 1;
            }
        }

        let mut queue: VecDeque<u64> = in_degree.iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&n, _)| n)
            .collect();
        let mut result = Vec::new();

        while let Some(node) = queue.pop_front() {
            result.push(node);
            if let Some(succs) = self.edges.get(&node) {
                for &succ in succs {
                    if let Some(deg) = in_degree.get_mut(&succ) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(succ);
                        }
                    }
                }
            }
        }

        if result.len() == self.nodes.len() {
            Some(result)
        } else {
            None // cycle detected
        }
    }
}

#[derive(Debug, Clone)]
pub struct TransitiveClosure {
    node_index: HashMap<u64, usize>,
    matrix: Vec<Vec<bool>>,
}

impl TransitiveClosure {
    pub fn is_ordered(&self, a: u64, b: u64) -> bool {
        if let (Some(&ai), Some(&bi)) = (self.node_index.get(&a), self.node_index.get(&b)) {
            self.matrix[ai][bi] || self.matrix[bi][ai]
        } else {
            false
        }
    }

    pub fn is_before(&self, a: u64, b: u64) -> bool {
        if let (Some(&ai), Some(&bi)) = (self.node_index.get(&a), self.node_index.get(&b)) {
            self.matrix[ai][bi]
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Barrier Analysis
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarrierInfo {
    pub id: u32,
    pub timestamp: u64,
    pub thread_ids: Vec<u32>,
    pub memory_scope: String,
}

#[derive(Debug, Clone)]
pub struct BarrierAnalysis {
    barriers: Vec<BarrierInfo>,
    thread_count: u32,
}

impl BarrierAnalysis {
    pub fn new(thread_count: u32) -> Self {
        Self { barriers: Vec::new(), thread_count }
    }

    pub fn add_barrier(&mut self, barrier: BarrierInfo) {
        self.barriers.push(barrier);
    }

    pub fn check_convergence(&self) -> Vec<String> {
        let mut issues = Vec::new();
        for barrier in &self.barriers {
            if barrier.thread_ids.len() < self.thread_count as usize {
                let missing: Vec<u32> = (0..self.thread_count)
                    .filter(|t| !barrier.thread_ids.contains(t))
                    .collect();
                issues.push(format!(
                    "Barrier {} not reached by threads {:?}",
                    barrier.id, missing
                ));
            }
            let unique: HashSet<u32> = barrier.thread_ids.iter().copied().collect();
            if unique.len() != barrier.thread_ids.len() {
                issues.push(format!("Barrier {} reached multiple times by same thread", barrier.id));
            }
        }
        issues
    }

    pub fn build_hb_from_barriers(&self) -> HappensBeforeGraph {
        let mut graph = HappensBeforeGraph::new();
        for barrier in &self.barriers {
            // All accesses before barrier happen-before all accesses after barrier
            // We use barrier timestamp as the ordering point
            let ts = barrier.timestamp;
            for &tid in &barrier.thread_ids {
                // Use a simple encoding: thread_id * 1_000_000 + local_timestamp
                let before_event = (tid as u64) * 1_000_000 + ts - 1;
                let after_event = (tid as u64) * 1_000_000 + ts + 1;
                graph.add_edge(before_event, after_event);
            }
            // Cross-thread ordering at the barrier
            let tids: Vec<u32> = barrier.thread_ids.clone();
            for i in 0..tids.len() {
                for j in (i + 1)..tids.len() {
                    let e1 = (tids[i] as u64) * 1_000_000 + ts;
                    let e2 = (tids[j] as u64) * 1_000_000 + ts;
                    graph.add_edge(e1, e2);
                    graph.add_edge(e2, e1);
                }
            }
        }
        graph
    }
}

// ---------------------------------------------------------------------------
// Shared Memory Race Detector (higher-level)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SharedMemoryRaceDetector {
    detector: DataRaceDetector,
    barrier_analysis: BarrierAnalysis,
    lock_sets: HashMap<u32, HashSet<String>>, // thread -> held locks
    lock_history: HashMap<String, Vec<(u32, u64)>>, // lock -> [(thread, timestamp)]
}

impl SharedMemoryRaceDetector {
    pub fn new(thread_count: u32) -> Self {
        Self {
            detector: DataRaceDetector::new(),
            barrier_analysis: BarrierAnalysis::new(thread_count),
            lock_sets: HashMap::new(),
            lock_history: HashMap::new(),
        }
    }

    pub fn add_access(&mut self, access: MemoryAccess) {
        self.detector.add_access(access);
    }

    pub fn acquire_lock(&mut self, thread_id: u32, lock_id: impl Into<String>, timestamp: u64) {
        let lock_id = lock_id.into();
        self.lock_sets.entry(thread_id).or_insert_with(HashSet::new).insert(lock_id.clone());
        self.lock_history.entry(lock_id).or_insert_with(Vec::new).push((thread_id, timestamp));
    }

    pub fn release_lock(&mut self, thread_id: u32, lock_id: &str, timestamp: u64) {
        if let Some(locks) = self.lock_sets.get_mut(&thread_id) {
            locks.remove(lock_id);
        }
        // Add happens-before from release to next acquire of same lock
        if let Some(history) = self.lock_history.get(lock_id) {
            if let Some(&(_, last_ts)) = history.last() {
                self.detector.add_happens_before(last_ts, timestamp);
            }
        }
    }

    pub fn add_barrier(&mut self, barrier: BarrierInfo) {
        let ts = barrier.timestamp;
        let thread_ids = barrier.thread_ids.clone();
        self.barrier_analysis.add_barrier(barrier);
        // Add HB edges for the barrier
        for i in 0..thread_ids.len() {
            for j in (i + 1)..thread_ids.len() {
                self.detector.add_happens_before(
                    (thread_ids[i] as u64) * 1_000_000 + ts,
                    (thread_ids[j] as u64) * 1_000_000 + ts,
                );
            }
        }
    }

    pub fn detect_races(&self) -> Vec<RacePair> {
        self.detector.detect_races()
    }
}

// ---------------------------------------------------------------------------
// Index Bounds Checking
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrayDescriptor {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub element_size: u64,
}

impl ArrayDescriptor {
    pub fn new(name: impl Into<String>, dimensions: Vec<u64>, element_size: u64) -> Self {
        Self { name: name.into(), dimensions, element_size }
    }

    pub fn total_elements(&self) -> u64 {
        self.dimensions.iter().product()
    }

    pub fn total_size(&self) -> u64 {
        self.total_elements() * self.element_size
    }

    pub fn is_in_bounds(&self, indices: &[u64]) -> bool {
        if indices.len() != self.dimensions.len() {
            return false;
        }
        indices.iter().zip(self.dimensions.iter()).all(|(&idx, &dim)| idx < dim)
    }

    pub fn linear_index(&self, indices: &[u64]) -> Option<u64> {
        if !self.is_in_bounds(indices) {
            return None;
        }
        let mut linear = 0u64;
        let mut stride = 1u64;
        for i in (0..indices.len()).rev() {
            linear += indices[i] * stride;
            stride *= self.dimensions[i];
        }
        Some(linear)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutOfBoundsViolation {
    pub array_name: String,
    pub indices: Vec<u64>,
    pub dimensions: Vec<u64>,
    pub violating_dimension: usize,
    pub description: String,
}

impl fmt::Display for OutOfBoundsViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description)
    }
}

#[derive(Debug, Clone)]
pub struct IndexBoundsChecker {
    arrays: HashMap<String, ArrayDescriptor>,
}

impl IndexBoundsChecker {
    pub fn new() -> Self {
        Self { arrays: HashMap::new() }
    }

    pub fn register_array(&mut self, desc: ArrayDescriptor) {
        self.arrays.insert(desc.name.clone(), desc);
    }

    pub fn check_index_bounds(&self, array_name: &str, indices: &[u64]) -> Option<OutOfBoundsViolation> {
        let desc = self.arrays.get(array_name)?;
        if indices.len() != desc.dimensions.len() {
            return Some(OutOfBoundsViolation {
                array_name: array_name.to_string(),
                indices: indices.to_vec(),
                dimensions: desc.dimensions.clone(),
                violating_dimension: 0,
                description: format!(
                    "Dimension mismatch: array '{}' has {} dimensions but {} indices provided",
                    array_name, desc.dimensions.len(), indices.len()
                ),
            });
        }
        for (dim, (&idx, &bound)) in indices.iter().zip(desc.dimensions.iter()).enumerate() {
            if idx >= bound {
                return Some(OutOfBoundsViolation {
                    array_name: array_name.to_string(),
                    indices: indices.to_vec(),
                    dimensions: desc.dimensions.clone(),
                    violating_dimension: dim,
                    description: format!(
                        "Index {} out of bounds in dimension {} of array '{}' (bound={})",
                        idx, dim, array_name, bound
                    ),
                });
            }
        }
        None
    }

    pub fn batch_check(&self, checks: &[(String, Vec<u64>)]) -> Vec<OutOfBoundsViolation> {
        checks.iter()
            .filter_map(|(name, indices)| self.check_index_bounds(name, indices))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Leak Detection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakInfo {
    pub allocation_id: String,
    pub base: u64,
    pub size: u64,
    pub kind: MemoryRegionKind,
    pub alloc_timestamp: u64,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakReport {
    pub leaks: Vec<LeakInfo>,
    pub total_leaked_bytes: u64,
    pub total_allocations: usize,
    pub leaked_allocations: usize,
}

impl LeakReport {
    pub fn is_clean(&self) -> bool {
        self.leaks.is_empty()
    }

    pub fn leak_rate(&self) -> f64 {
        if self.total_allocations == 0 {
            return 0.0;
        }
        self.leaked_allocations as f64 / self.total_allocations as f64
    }
}

#[derive(Debug, Clone)]
pub struct LeakDetector {
    allocations: HashMap<String, (u64, u64, MemoryRegionKind, u64)>, // id -> (base, size, kind, timestamp)
    freed: HashSet<String>,
    reachable: HashSet<String>,
}

impl LeakDetector {
    pub fn new() -> Self {
        Self {
            allocations: HashMap::new(),
            freed: HashSet::new(),
            reachable: HashSet::new(),
        }
    }

    pub fn track_allocation(&mut self, id: impl Into<String>, base: u64, size: u64, kind: MemoryRegionKind, timestamp: u64) {
        self.allocations.insert(id.into(), (base, size, kind, timestamp));
    }

    pub fn track_free(&mut self, id: &str) {
        self.freed.insert(id.to_string());
    }

    pub fn mark_reachable(&mut self, id: &str) {
        self.reachable.insert(id.to_string());
    }

    pub fn detect_leaks(&self) -> LeakReport {
        let mut leaks = Vec::new();
        let mut total_leaked = 0u64;
        for (id, &(base, size, kind, ts)) in &self.allocations {
            if !self.freed.contains(id) && !self.reachable.contains(id) {
                total_leaked += size;
                leaks.push(LeakInfo {
                    allocation_id: id.clone(),
                    base,
                    size,
                    kind,
                    alloc_timestamp: ts,
                    description: format!(
                        "Allocation '{}' at 0x{:x} (size={}) never freed and not reachable",
                        id, base, size
                    ),
                });
            }
        }
        LeakReport {
            total_allocations: self.allocations.len(),
            leaked_allocations: leaks.len(),
            total_leaked_bytes: total_leaked,
            leaks,
        }
    }

    pub fn detect_leaks_by_kind(&self) -> HashMap<MemoryRegionKind, Vec<LeakInfo>> {
        let report = self.detect_leaks();
        let mut by_kind: HashMap<MemoryRegionKind, Vec<LeakInfo>> = HashMap::new();
        for leak in report.leaks {
            by_kind.entry(leak.kind).or_insert_with(Vec::new).push(leak);
        }
        by_kind
    }
}

// ---------------------------------------------------------------------------
// Combined Memory Safety Analyzer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SafetySeverity {
    Info,
    Warning,
    Error,
    Critical,
}

impl fmt::Display for SafetySeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARNING"),
            Self::Error => write!(f, "ERROR"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyFinding {
    pub category: String,
    pub severity: SafetySeverity,
    pub description: String,
    pub location: Option<String>,
}

impl fmt::Display for SafetyFinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}: {}", self.severity, self.category, self.description)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyReport {
    pub findings: Vec<SafetyFinding>,
    pub overflow_report: Option<OverflowReport>,
    pub leak_report: Option<LeakReport>,
    pub race_count: usize,
    pub uaf_count: usize,
    pub oob_count: usize,
    pub total_issues: usize,
    pub max_severity: SafetySeverity,
}

impl SafetyReport {
    pub fn is_safe(&self) -> bool {
        self.total_issues == 0
    }

    pub fn by_severity(&self, severity: SafetySeverity) -> Vec<&SafetyFinding> {
        self.findings.iter().filter(|f| f.severity == severity).collect()
    }

    pub fn by_category(&self, category: &str) -> Vec<&SafetyFinding> {
        self.findings.iter().filter(|f| f.category == category).collect()
    }
}

#[derive(Debug, Clone)]
pub struct MemorySafetyAnalyzer {
    overflow_detector: OverflowDetector,
    bounds_checker: IndexBoundsChecker,
    leak_detector: LeakDetector,
    race_detector: DataRaceDetector,
    lifetime_tracker: LifetimeTracker,
    accesses: Vec<BufferAccess>,
    index_checks: Vec<(String, Vec<u64>)>,
}

impl MemorySafetyAnalyzer {
    pub fn new() -> Self {
        Self {
            overflow_detector: OverflowDetector::new(),
            bounds_checker: IndexBoundsChecker::new(),
            leak_detector: LeakDetector::new(),
            race_detector: DataRaceDetector::new(),
            lifetime_tracker: LifetimeTracker::new(),
            accesses: Vec::new(),
            index_checks: Vec::new(),
        }
    }

    pub fn register_buffer(&mut self, id: impl Into<String>, size: u64) {
        let id_str: String = id.into();
        self.overflow_detector.register_buffer(id_str.clone(), size);
        self.leak_detector.track_allocation(id_str, 0, size, MemoryRegionKind::Global, 0);
    }

    pub fn register_array(&mut self, desc: ArrayDescriptor) {
        self.bounds_checker.register_array(desc);
    }

    pub fn add_access(&mut self, access: BufferAccess) {
        self.race_detector.add_access(MemoryAccess::new(
            access.offset, access.size, access.thread_id, access.is_write, access.timestamp,
        ));
        self.accesses.push(access);
    }

    pub fn add_index_check(&mut self, array_name: String, indices: Vec<u64>) {
        self.index_checks.push((array_name, indices));
    }

    pub fn allocate(&mut self, id: impl Into<String>, base: u64, size: u64, thread_id: u32) {
        self.lifetime_tracker.allocate(id, base, size, thread_id);
    }

    pub fn free(&mut self, id: &str) -> Result<(), String> {
        self.lifetime_tracker.free(id)?;
        self.leak_detector.track_free(id);
        Ok(())
    }

    pub fn add_happens_before(&mut self, before: u64, after: u64) {
        self.race_detector.add_happens_before(before, after);
    }

    pub fn analyze(&self) -> SafetyReport {
        let mut findings = Vec::new();
        let mut max_severity = SafetySeverity::Info;

        // Overflow checks
        let overflow_report = self.overflow_detector.batch_check(&self.accesses);
        for v in &overflow_report.violations {
            let severity = SafetySeverity::Critical;
            if (severity as u8) > (max_severity as u8) {
                max_severity = severity;
            }
            findings.push(SafetyFinding {
                category: "buffer_overflow".to_string(),
                severity,
                description: v.to_string(),
                location: v.access.source_location.clone(),
            });
        }

        // OOB checks
        let oob_violations = self.bounds_checker.batch_check(&self.index_checks);
        for v in &oob_violations {
            let severity = SafetySeverity::Error;
            if (severity as u8) > (max_severity as u8) {
                max_severity = severity;
            }
            findings.push(SafetyFinding {
                category: "out_of_bounds".to_string(),
                severity,
                description: v.to_string(),
                location: None,
            });
        }

        // Race detection
        let races = self.race_detector.detect_races();
        for r in &races {
            let severity = SafetySeverity::Error;
            if (severity as u8) > (max_severity as u8) {
                max_severity = severity;
            }
            findings.push(SafetyFinding {
                category: "data_race".to_string(),
                severity,
                description: r.to_string(),
                location: None,
            });
        }

        // UAF checks
        let mut uaf_detector = UseAfterFreeDetector::new(self.lifetime_tracker.clone());
        for access in &self.accesses {
            uaf_detector.check_use(
                &access.buffer_id,
                access.offset,
                access.size,
                access.thread_id,
                access.timestamp,
            );
        }
        let uaf_violations = uaf_detector.violations();
        for v in uaf_violations {
            let severity = SafetySeverity::Critical;
            if (severity as u8) > (max_severity as u8) {
                max_severity = severity;
            }
            findings.push(SafetyFinding {
                category: "use_after_free".to_string(),
                severity,
                description: v.to_string(),
                location: None,
            });
        }

        // Leak detection
        let leak_report = self.leak_detector.detect_leaks();
        for l in &leak_report.leaks {
            let severity = SafetySeverity::Warning;
            if (severity as u8) > (max_severity as u8) {
                max_severity = severity;
            }
            findings.push(SafetyFinding {
                category: "memory_leak".to_string(),
                severity,
                description: l.description.clone(),
                location: None,
            });
        }

        let total = findings.len();
        SafetyReport {
            race_count: races.len(),
            uaf_count: uaf_violations.len(),
            oob_count: oob_violations.len(),
            overflow_report: Some(overflow_report),
            leak_report: Some(leak_report),
            total_issues: total,
            max_severity,
            findings,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// ===== Extended Memory Safety Operations =====

#[derive(Debug, Clone)]
pub struct MemorySanitizer {
    pub regions: Vec<(u64, u64)>,
    pub tainted_addresses: Vec<u64>,
    pub violation_count: usize,
}

impl MemorySanitizer {
    pub fn new(regions: Vec<(u64, u64)>, tainted_addresses: Vec<u64>, violation_count: usize) -> Self {
        MemorySanitizer { regions, tainted_addresses, violation_count }
    }

    pub fn regions_len(&self) -> usize {
        self.regions.len()
    }

    pub fn regions_is_empty(&self) -> bool {
        self.regions.is_empty()
    }

    pub fn tainted_addresses_len(&self) -> usize {
        self.tainted_addresses.len()
    }

    pub fn tainted_addresses_is_empty(&self) -> bool {
        self.tainted_addresses.is_empty()
    }

    pub fn get_violation_count(&self) -> usize {
        self.violation_count
    }

    pub fn with_violation_count(mut self, v: usize) -> Self {
        self.violation_count = v; self
    }

}

impl fmt::Display for MemorySanitizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MemorySanitizer({:?})", self.regions)
    }
}

#[derive(Debug, Clone)]
pub struct MemorySanitizerBuilder {
    regions: Vec<(u64, u64)>,
    tainted_addresses: Vec<u64>,
    violation_count: usize,
}

impl MemorySanitizerBuilder {
    pub fn new() -> Self {
        MemorySanitizerBuilder {
            regions: Vec::new(),
            tainted_addresses: Vec::new(),
            violation_count: 0,
        }
    }

    pub fn regions(mut self, v: Vec<(u64, u64)>) -> Self { self.regions = v; self }
    pub fn tainted_addresses(mut self, v: Vec<u64>) -> Self { self.tainted_addresses = v; self }
    pub fn violation_count(mut self, v: usize) -> Self { self.violation_count = v; self }
}

#[derive(Debug, Clone)]
pub struct TaintTracker {
    pub taint_map: Vec<(u64, u32)>,
    pub propagation_count: usize,
    pub sources: Vec<String>,
}

impl TaintTracker {
    pub fn new(taint_map: Vec<(u64, u32)>, propagation_count: usize, sources: Vec<String>) -> Self {
        TaintTracker { taint_map, propagation_count, sources }
    }

    pub fn taint_map_len(&self) -> usize {
        self.taint_map.len()
    }

    pub fn taint_map_is_empty(&self) -> bool {
        self.taint_map.is_empty()
    }

    pub fn get_propagation_count(&self) -> usize {
        self.propagation_count
    }

    pub fn sources_len(&self) -> usize {
        self.sources.len()
    }

    pub fn sources_is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    pub fn with_propagation_count(mut self, v: usize) -> Self {
        self.propagation_count = v; self
    }

}

impl fmt::Display for TaintTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TaintTracker({:?})", self.taint_map)
    }
}

#[derive(Debug, Clone)]
pub struct TaintTrackerBuilder {
    taint_map: Vec<(u64, u32)>,
    propagation_count: usize,
    sources: Vec<String>,
}

impl TaintTrackerBuilder {
    pub fn new() -> Self {
        TaintTrackerBuilder {
            taint_map: Vec::new(),
            propagation_count: 0,
            sources: Vec::new(),
        }
    }

    pub fn taint_map(mut self, v: Vec<(u64, u32)>) -> Self { self.taint_map = v; self }
    pub fn propagation_count(mut self, v: usize) -> Self { self.propagation_count = v; self }
    pub fn sources(mut self, v: Vec<String>) -> Self { self.sources = v; self }
}

#[derive(Debug, Clone)]
pub struct PointerProvenance {
    pub pointer_id: u64,
    pub origin: String,
    pub valid: bool,
    pub generation: u32,
}

impl PointerProvenance {
    pub fn new(pointer_id: u64, origin: String, valid: bool, generation: u32) -> Self {
        PointerProvenance { pointer_id, origin, valid, generation }
    }

    pub fn get_pointer_id(&self) -> u64 {
        self.pointer_id
    }

    pub fn get_origin(&self) -> &str {
        &self.origin
    }

    pub fn get_valid(&self) -> bool {
        self.valid
    }

    pub fn get_generation(&self) -> u32 {
        self.generation
    }

    pub fn with_pointer_id(mut self, v: u64) -> Self {
        self.pointer_id = v; self
    }

    pub fn with_origin(mut self, v: impl Into<String>) -> Self {
        self.origin = v.into(); self
    }

    pub fn with_valid(mut self, v: bool) -> Self {
        self.valid = v; self
    }

    pub fn with_generation(mut self, v: u32) -> Self {
        self.generation = v; self
    }

}

impl fmt::Display for PointerProvenance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PointerProvenance({:?})", self.pointer_id)
    }
}

#[derive(Debug, Clone)]
pub struct PointerProvenanceBuilder {
    pointer_id: u64,
    origin: String,
    valid: bool,
    generation: u32,
}

impl PointerProvenanceBuilder {
    pub fn new() -> Self {
        PointerProvenanceBuilder {
            pointer_id: 0,
            origin: String::new(),
            valid: false,
            generation: 0,
        }
    }

    pub fn pointer_id(mut self, v: u64) -> Self { self.pointer_id = v; self }
    pub fn origin(mut self, v: impl Into<String>) -> Self { self.origin = v.into(); self }
    pub fn valid(mut self, v: bool) -> Self { self.valid = v; self }
    pub fn generation(mut self, v: u32) -> Self { self.generation = v; self }
}

#[derive(Debug, Clone)]
pub struct MemoryAccessPattern {
    pub accesses: Vec<u64>,
    pub stride: i64,
    pub is_sequential: bool,
    pub locality_score: f64,
}

impl MemoryAccessPattern {
    pub fn new(accesses: Vec<u64>, stride: i64, is_sequential: bool, locality_score: f64) -> Self {
        MemoryAccessPattern { accesses, stride, is_sequential, locality_score }
    }

    pub fn accesses_len(&self) -> usize {
        self.accesses.len()
    }

    pub fn accesses_is_empty(&self) -> bool {
        self.accesses.is_empty()
    }

    pub fn get_stride(&self) -> i64 {
        self.stride
    }

    pub fn get_is_sequential(&self) -> bool {
        self.is_sequential
    }

    pub fn get_locality_score(&self) -> f64 {
        self.locality_score
    }

    pub fn with_stride(mut self, v: i64) -> Self {
        self.stride = v; self
    }

    pub fn with_is_sequential(mut self, v: bool) -> Self {
        self.is_sequential = v; self
    }

    pub fn with_locality_score(mut self, v: f64) -> Self {
        self.locality_score = v; self
    }

}

impl fmt::Display for MemoryAccessPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MemoryAccessPattern({:?})", self.accesses)
    }
}

#[derive(Debug, Clone)]
pub struct MemoryAccessPatternBuilder {
    accesses: Vec<u64>,
    stride: i64,
    is_sequential: bool,
    locality_score: f64,
}

impl MemoryAccessPatternBuilder {
    pub fn new() -> Self {
        MemoryAccessPatternBuilder {
            accesses: Vec::new(),
            stride: 0,
            is_sequential: false,
            locality_score: 0.0,
        }
    }

    pub fn accesses(mut self, v: Vec<u64>) -> Self { self.accesses = v; self }
    pub fn stride(mut self, v: i64) -> Self { self.stride = v; self }
    pub fn is_sequential(mut self, v: bool) -> Self { self.is_sequential = v; self }
    pub fn locality_score(mut self, v: f64) -> Self { self.locality_score = v; self }
}

#[derive(Debug, Clone)]
pub struct GpuOccupancyAnalysis {
    pub active_warps: u32,
    pub max_warps: u32,
    pub occupancy_pct: f64,
    pub limiting_factor: String,
}

impl GpuOccupancyAnalysis {
    pub fn new(active_warps: u32, max_warps: u32, occupancy_pct: f64, limiting_factor: String) -> Self {
        GpuOccupancyAnalysis { active_warps, max_warps, occupancy_pct, limiting_factor }
    }

    pub fn get_active_warps(&self) -> u32 {
        self.active_warps
    }

    pub fn get_max_warps(&self) -> u32 {
        self.max_warps
    }

    pub fn get_occupancy_pct(&self) -> f64 {
        self.occupancy_pct
    }

    pub fn get_limiting_factor(&self) -> &str {
        &self.limiting_factor
    }

    pub fn with_active_warps(mut self, v: u32) -> Self {
        self.active_warps = v; self
    }

    pub fn with_max_warps(mut self, v: u32) -> Self {
        self.max_warps = v; self
    }

    pub fn with_occupancy_pct(mut self, v: f64) -> Self {
        self.occupancy_pct = v; self
    }

    pub fn with_limiting_factor(mut self, v: impl Into<String>) -> Self {
        self.limiting_factor = v.into(); self
    }

}

impl fmt::Display for GpuOccupancyAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GpuOccupancyAnalysis({:?})", self.active_warps)
    }
}

#[derive(Debug, Clone)]
pub struct GpuOccupancyAnalysisBuilder {
    active_warps: u32,
    max_warps: u32,
    occupancy_pct: f64,
    limiting_factor: String,
}

impl GpuOccupancyAnalysisBuilder {
    pub fn new() -> Self {
        GpuOccupancyAnalysisBuilder {
            active_warps: 0,
            max_warps: 0,
            occupancy_pct: 0.0,
            limiting_factor: String::new(),
        }
    }

    pub fn active_warps(mut self, v: u32) -> Self { self.active_warps = v; self }
    pub fn max_warps(mut self, v: u32) -> Self { self.max_warps = v; self }
    pub fn occupancy_pct(mut self, v: f64) -> Self { self.occupancy_pct = v; self }
    pub fn limiting_factor(mut self, v: impl Into<String>) -> Self { self.limiting_factor = v.into(); self }
}

#[derive(Debug, Clone)]
pub struct BufferZoneVerifier {
    pub zone_size: u64,
    pub verified_count: usize,
    pub violations: Vec<String>,
}

impl BufferZoneVerifier {
    pub fn new(zone_size: u64, verified_count: usize, violations: Vec<String>) -> Self {
        BufferZoneVerifier { zone_size, verified_count, violations }
    }

    pub fn get_zone_size(&self) -> u64 {
        self.zone_size
    }

    pub fn get_verified_count(&self) -> usize {
        self.verified_count
    }

    pub fn violations_len(&self) -> usize {
        self.violations.len()
    }

    pub fn violations_is_empty(&self) -> bool {
        self.violations.is_empty()
    }

    pub fn with_zone_size(mut self, v: u64) -> Self {
        self.zone_size = v; self
    }

    pub fn with_verified_count(mut self, v: usize) -> Self {
        self.verified_count = v; self
    }

}

impl fmt::Display for BufferZoneVerifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BufferZoneVerifier({:?})", self.zone_size)
    }
}

#[derive(Debug, Clone)]
pub struct BufferZoneVerifierBuilder {
    zone_size: u64,
    verified_count: usize,
    violations: Vec<String>,
}

impl BufferZoneVerifierBuilder {
    pub fn new() -> Self {
        BufferZoneVerifierBuilder {
            zone_size: 0,
            verified_count: 0,
            violations: Vec::new(),
        }
    }

    pub fn zone_size(mut self, v: u64) -> Self { self.zone_size = v; self }
    pub fn verified_count(mut self, v: usize) -> Self { self.verified_count = v; self }
    pub fn violations(mut self, v: Vec<String>) -> Self { self.violations = v; self }
}

#[derive(Debug, Clone)]
pub struct MemoryAccessLog {
    pub entries: Vec<(u64, u64, bool)>,
    pub total_reads: u64,
    pub total_writes: u64,
}

impl MemoryAccessLog {
    pub fn new(entries: Vec<(u64, u64, bool)>, total_reads: u64, total_writes: u64) -> Self {
        MemoryAccessLog { entries, total_reads, total_writes }
    }

    pub fn entries_len(&self) -> usize {
        self.entries.len()
    }

    pub fn entries_is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get_total_reads(&self) -> u64 {
        self.total_reads
    }

    pub fn get_total_writes(&self) -> u64 {
        self.total_writes
    }

    pub fn with_total_reads(mut self, v: u64) -> Self {
        self.total_reads = v; self
    }

    pub fn with_total_writes(mut self, v: u64) -> Self {
        self.total_writes = v; self
    }

}

impl fmt::Display for MemoryAccessLog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MemoryAccessLog({:?})", self.entries)
    }
}

#[derive(Debug, Clone)]
pub struct MemoryAccessLogBuilder {
    entries: Vec<(u64, u64, bool)>,
    total_reads: u64,
    total_writes: u64,
}

impl MemoryAccessLogBuilder {
    pub fn new() -> Self {
        MemoryAccessLogBuilder {
            entries: Vec::new(),
            total_reads: 0,
            total_writes: 0,
        }
    }

    pub fn entries(mut self, v: Vec<(u64, u64, bool)>) -> Self { self.entries = v; self }
    pub fn total_reads(mut self, v: u64) -> Self { self.total_reads = v; self }
    pub fn total_writes(mut self, v: u64) -> Self { self.total_writes = v; self }
}

#[derive(Debug, Clone)]
pub struct CoalescingAnalysis {
    pub warp_accesses: Vec<Vec<u64>>,
    pub coalesced_ratio: f64,
    pub transaction_count: u32,
}

impl CoalescingAnalysis {
    pub fn new(warp_accesses: Vec<Vec<u64>>, coalesced_ratio: f64, transaction_count: u32) -> Self {
        CoalescingAnalysis { warp_accesses, coalesced_ratio, transaction_count }
    }

    pub fn warp_accesses_len(&self) -> usize {
        self.warp_accesses.len()
    }

    pub fn warp_accesses_is_empty(&self) -> bool {
        self.warp_accesses.is_empty()
    }

    pub fn get_coalesced_ratio(&self) -> f64 {
        self.coalesced_ratio
    }

    pub fn get_transaction_count(&self) -> u32 {
        self.transaction_count
    }

    pub fn with_coalesced_ratio(mut self, v: f64) -> Self {
        self.coalesced_ratio = v; self
    }

    pub fn with_transaction_count(mut self, v: u32) -> Self {
        self.transaction_count = v; self
    }

}

impl fmt::Display for CoalescingAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CoalescingAnalysis({:?})", self.warp_accesses)
    }
}

#[derive(Debug, Clone)]
pub struct CoalescingAnalysisBuilder {
    warp_accesses: Vec<Vec<u64>>,
    coalesced_ratio: f64,
    transaction_count: u32,
}

impl CoalescingAnalysisBuilder {
    pub fn new() -> Self {
        CoalescingAnalysisBuilder {
            warp_accesses: Vec::new(),
            coalesced_ratio: 0.0,
            transaction_count: 0,
        }
    }

    pub fn warp_accesses(mut self, v: Vec<Vec<u64>>) -> Self { self.warp_accesses = v; self }
    pub fn coalesced_ratio(mut self, v: f64) -> Self { self.coalesced_ratio = v; self }
    pub fn transaction_count(mut self, v: u32) -> Self { self.transaction_count = v; self }
}

#[derive(Debug, Clone)]
pub struct BankConflictDetector {
    pub bank_count: u32,
    pub conflict_count: u32,
    pub total_accesses: u32,
    pub conflict_rate: f64,
}

impl BankConflictDetector {
    pub fn new(bank_count: u32, conflict_count: u32, total_accesses: u32, conflict_rate: f64) -> Self {
        BankConflictDetector { bank_count, conflict_count, total_accesses, conflict_rate }
    }

    pub fn get_bank_count(&self) -> u32 {
        self.bank_count
    }

    pub fn get_conflict_count(&self) -> u32 {
        self.conflict_count
    }

    pub fn get_total_accesses(&self) -> u32 {
        self.total_accesses
    }

    pub fn get_conflict_rate(&self) -> f64 {
        self.conflict_rate
    }

    pub fn with_bank_count(mut self, v: u32) -> Self {
        self.bank_count = v; self
    }

    pub fn with_conflict_count(mut self, v: u32) -> Self {
        self.conflict_count = v; self
    }

    pub fn with_total_accesses(mut self, v: u32) -> Self {
        self.total_accesses = v; self
    }

    pub fn with_conflict_rate(mut self, v: f64) -> Self {
        self.conflict_rate = v; self
    }

}

impl fmt::Display for BankConflictDetector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BankConflictDetector({:?})", self.bank_count)
    }
}

#[derive(Debug, Clone)]
pub struct BankConflictDetectorBuilder {
    bank_count: u32,
    conflict_count: u32,
    total_accesses: u32,
    conflict_rate: f64,
}

impl BankConflictDetectorBuilder {
    pub fn new() -> Self {
        BankConflictDetectorBuilder {
            bank_count: 0,
            conflict_count: 0,
            total_accesses: 0,
            conflict_rate: 0.0,
        }
    }

    pub fn bank_count(mut self, v: u32) -> Self { self.bank_count = v; self }
    pub fn conflict_count(mut self, v: u32) -> Self { self.conflict_count = v; self }
    pub fn total_accesses(mut self, v: u32) -> Self { self.total_accesses = v; self }
    pub fn conflict_rate(mut self, v: f64) -> Self { self.conflict_rate = v; self }
}

#[derive(Debug, Clone)]
pub struct SharedMemoryPadding {
    pub original_size: u64,
    pub padded_size: u64,
    pub padding_locations: Vec<u64>,
    pub bank_width: u32,
}

impl SharedMemoryPadding {
    pub fn new(original_size: u64, padded_size: u64, padding_locations: Vec<u64>, bank_width: u32) -> Self {
        SharedMemoryPadding { original_size, padded_size, padding_locations, bank_width }
    }

    pub fn get_original_size(&self) -> u64 {
        self.original_size
    }

    pub fn get_padded_size(&self) -> u64 {
        self.padded_size
    }

    pub fn padding_locations_len(&self) -> usize {
        self.padding_locations.len()
    }

    pub fn padding_locations_is_empty(&self) -> bool {
        self.padding_locations.is_empty()
    }

    pub fn get_bank_width(&self) -> u32 {
        self.bank_width
    }

    pub fn with_original_size(mut self, v: u64) -> Self {
        self.original_size = v; self
    }

    pub fn with_padded_size(mut self, v: u64) -> Self {
        self.padded_size = v; self
    }

    pub fn with_bank_width(mut self, v: u32) -> Self {
        self.bank_width = v; self
    }

}

impl fmt::Display for SharedMemoryPadding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SharedMemoryPadding({:?})", self.original_size)
    }
}

#[derive(Debug, Clone)]
pub struct SharedMemoryPaddingBuilder {
    original_size: u64,
    padded_size: u64,
    padding_locations: Vec<u64>,
    bank_width: u32,
}

impl SharedMemoryPaddingBuilder {
    pub fn new() -> Self {
        SharedMemoryPaddingBuilder {
            original_size: 0,
            padded_size: 0,
            padding_locations: Vec::new(),
            bank_width: 0,
        }
    }

    pub fn original_size(mut self, v: u64) -> Self { self.original_size = v; self }
    pub fn padded_size(mut self, v: u64) -> Self { self.padded_size = v; self }
    pub fn padding_locations(mut self, v: Vec<u64>) -> Self { self.padding_locations = v; self }
    pub fn bank_width(mut self, v: u32) -> Self { self.bank_width = v; self }
}

#[derive(Debug, Clone)]
pub struct MemoryFenceAnalysis {
    pub fence_count: usize,
    pub fence_locations: Vec<usize>,
    pub redundant_count: usize,
}

impl MemoryFenceAnalysis {
    pub fn new(fence_count: usize, fence_locations: Vec<usize>, redundant_count: usize) -> Self {
        MemoryFenceAnalysis { fence_count, fence_locations, redundant_count }
    }

    pub fn get_fence_count(&self) -> usize {
        self.fence_count
    }

    pub fn fence_locations_len(&self) -> usize {
        self.fence_locations.len()
    }

    pub fn fence_locations_is_empty(&self) -> bool {
        self.fence_locations.is_empty()
    }

    pub fn get_redundant_count(&self) -> usize {
        self.redundant_count
    }

    pub fn with_fence_count(mut self, v: usize) -> Self {
        self.fence_count = v; self
    }

    pub fn with_redundant_count(mut self, v: usize) -> Self {
        self.redundant_count = v; self
    }

}

impl fmt::Display for MemoryFenceAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MemoryFenceAnalysis({:?})", self.fence_count)
    }
}

#[derive(Debug, Clone)]
pub struct MemoryFenceAnalysisBuilder {
    fence_count: usize,
    fence_locations: Vec<usize>,
    redundant_count: usize,
}

impl MemoryFenceAnalysisBuilder {
    pub fn new() -> Self {
        MemoryFenceAnalysisBuilder {
            fence_count: 0,
            fence_locations: Vec::new(),
            redundant_count: 0,
        }
    }

    pub fn fence_count(mut self, v: usize) -> Self { self.fence_count = v; self }
    pub fn fence_locations(mut self, v: Vec<usize>) -> Self { self.fence_locations = v; self }
    pub fn redundant_count(mut self, v: usize) -> Self { self.redundant_count = v; self }
}

#[derive(Debug, Clone)]
pub struct AddressSanitizer {
    pub shadow_memory_base: u64,
    pub quarantine_size: u64,
    pub detections: Vec<String>,
}

impl AddressSanitizer {
    pub fn new(shadow_memory_base: u64, quarantine_size: u64, detections: Vec<String>) -> Self {
        AddressSanitizer { shadow_memory_base, quarantine_size, detections }
    }

    pub fn get_shadow_memory_base(&self) -> u64 {
        self.shadow_memory_base
    }

    pub fn get_quarantine_size(&self) -> u64 {
        self.quarantine_size
    }

    pub fn detections_len(&self) -> usize {
        self.detections.len()
    }

    pub fn detections_is_empty(&self) -> bool {
        self.detections.is_empty()
    }

    pub fn with_shadow_memory_base(mut self, v: u64) -> Self {
        self.shadow_memory_base = v; self
    }

    pub fn with_quarantine_size(mut self, v: u64) -> Self {
        self.quarantine_size = v; self
    }

}

impl fmt::Display for AddressSanitizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AddressSanitizer({:?})", self.shadow_memory_base)
    }
}

#[derive(Debug, Clone)]
pub struct AddressSanitizerBuilder {
    shadow_memory_base: u64,
    quarantine_size: u64,
    detections: Vec<String>,
}

impl AddressSanitizerBuilder {
    pub fn new() -> Self {
        AddressSanitizerBuilder {
            shadow_memory_base: 0,
            quarantine_size: 0,
            detections: Vec::new(),
        }
    }

    pub fn shadow_memory_base(mut self, v: u64) -> Self { self.shadow_memory_base = v; self }
    pub fn quarantine_size(mut self, v: u64) -> Self { self.quarantine_size = v; self }
    pub fn detections(mut self, v: Vec<String>) -> Self { self.detections = v; self }
}

#[derive(Debug, Clone)]
pub struct MemsafeAnalysis {
    pub data: Vec<Vec<f64>>,
    pub size: usize,
    pub computed: bool,
    pub label: String,
    pub threshold: f64,
}

impl MemsafeAnalysis {
    pub fn new(size: usize) -> Self {
        let data = vec![vec![0.0; size]; size];
        MemsafeAnalysis { data, size, computed: false, label: "Memsafe".to_string(), threshold: 0.01 }
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

impl fmt::Display for MemsafeAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MemsafeAnalysis({:?})", self.data)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MemsafeStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Skipped,
}

impl fmt::Display for MemsafeStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemsafeStatus::Pending => write!(f, "pending"),
            MemsafeStatus::InProgress => write!(f, "inprogress"),
            MemsafeStatus::Completed => write!(f, "completed"),
            MemsafeStatus::Failed => write!(f, "failed"),
            MemsafeStatus::Skipped => write!(f, "skipped"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MemsafePriority {
    Critical,
    High,
    Medium,
    Low,
    None,
}

impl fmt::Display for MemsafePriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemsafePriority::Critical => write!(f, "critical"),
            MemsafePriority::High => write!(f, "high"),
            MemsafePriority::Medium => write!(f, "medium"),
            MemsafePriority::Low => write!(f, "low"),
            MemsafePriority::None => write!(f, "none"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MemsafeMode {
    Strict,
    Relaxed,
    Permissive,
    Custom,
}

impl fmt::Display for MemsafeMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemsafeMode::Strict => write!(f, "strict"),
            MemsafeMode::Relaxed => write!(f, "relaxed"),
            MemsafeMode::Permissive => write!(f, "permissive"),
            MemsafeMode::Custom => write!(f, "custom"),
        }
    }
}

pub fn memsafe_mean(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    data.iter().sum::<f64>() / data.len() as f64
}

pub fn memsafe_variance(data: &[f64]) -> f64 {
    if data.len() < 2 { return 0.0; }
    let mean = memsafe_mean(data);
    data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (data.len() - 1) as f64
}

pub fn memsafe_std_dev(data: &[f64]) -> f64 {
    memsafe_variance(data).sqrt()
}

pub fn memsafe_median(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n % 2 == 0 { (sorted[n/2 - 1] + sorted[n/2]) / 2.0 } else { sorted[n/2] }
}

/// Percentile calculator for MemSafe.
pub fn memsafe_percentile_at(data: &[f64], p: f64) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = p / 100.0 * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi { sorted[lo] }
    else { sorted[lo] * (hi as f64 - idx) + sorted[hi] * (idx - lo as f64) }
}

pub fn memsafe_entropy(data: &[f64]) -> f64 {
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

pub fn memsafe_gini(data: &[f64]) -> f64 {
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

pub fn memsafe_covariance(data: &[f64]) -> f64 {
    if data.len() < 4 { return 0.0; }
    let n = data.len() / 2;
    let x: Vec<f64> = data[..n].to_vec();
    let y: Vec<f64> = data[n..2*n].to_vec();
    let mx = memsafe_mean(&x);
    let my = memsafe_mean(&y);
    x.iter().zip(y.iter()).map(|(&xi, &yi)| (xi - mx) * (yi - my)).sum::<f64>() / (n - 1) as f64
}

pub fn memsafe_correlation(data: &[f64]) -> f64 {
    if data.len() < 4 { return 0.0; }
    let n = data.len() / 2;
    let cov = memsafe_covariance(data);
    let sx = memsafe_std_dev(&data[..n]);
    let sy = memsafe_std_dev(&data[n..2*n]);
    if sx * sy == 0.0 { 0.0 } else { cov / (sx * sy) }
}

pub fn memsafe_excess_kurtosis(data: &[f64]) -> f64 {
    if data.len() < 4 { return 0.0; }
    let m = memsafe_mean(data);
    let s = memsafe_std_dev(data);
    if s == 0.0 { return 0.0; }
    let n = data.len() as f64;
    let k = data.iter().map(|&x| ((x - m) / s).powi(4)).sum::<f64>() / n;
    k - 3.0
}

pub fn memsafe_sample_skewness(data: &[f64]) -> f64 {
    if data.len() < 3 { return 0.0; }
    let m = memsafe_mean(data);
    let s = memsafe_std_dev(data);
    if s == 0.0 { return 0.0; }
    let n = data.len() as f64;
    data.iter().map(|&x| ((x - m) / s).powi(3)).sum::<f64>() / n
}

pub fn memsafe_harmmean(data: &[f64]) -> f64 {
    if data.is_empty() || data.iter().any(|&x| x <= 0.0) { return 0.0; }
    let n = data.len() as f64;
    n / data.iter().map(|&x| 1.0 / x).sum::<f64>()
}

pub fn memsafe_geomean(data: &[f64]) -> f64 {
    if data.is_empty() || data.iter().any(|&x| x <= 0.0) { return 0.0; }
    let n = data.len() as f64;
    (data.iter().map(|&x| x.ln()).sum::<f64>() / n).exp()
}

/// Iterator over memsafe analysis results.
#[derive(Debug, Clone)]
pub struct MemsafeResultIterator {
    items: Vec<(usize, f64)>,
    position: usize,
}

impl MemsafeResultIterator {
    pub fn new(items: Vec<(usize, f64)>) -> Self {
        MemsafeResultIterator { items, position: 0 }
    }
    pub fn remaining(&self) -> usize { self.items.len() - self.position }
}

impl Iterator for MemsafeResultIterator {
    type Item = (usize, f64);
    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.items.len() {
            let item = self.items[self.position];
            self.position += 1;
            Some(item)
        } else { None }
    }
}

/// Convert MemorySanitizer description to a summary string.
pub fn memorysanitizer_to_summary(item: &MemorySanitizer) -> String {
    format!("MemorySanitizer: {:?}", item)
}

/// Convert TaintTracker description to a summary string.
pub fn tainttracker_to_summary(item: &TaintTracker) -> String {
    format!("TaintTracker: {:?}", item)
}

/// Convert PointerProvenance description to a summary string.
pub fn pointerprovenance_to_summary(item: &PointerProvenance) -> String {
    format!("PointerProvenance: {:?}", item)
}

/// Convert MemoryAccessPattern description to a summary string.
pub fn memoryaccesspattern_to_summary(item: &MemoryAccessPattern) -> String {
    format!("MemoryAccessPattern: {:?}", item)
}

/// Convert GpuOccupancyAnalysis description to a summary string.
pub fn gpuoccupancyanalysis_to_summary(item: &GpuOccupancyAnalysis) -> String {
    format!("GpuOccupancyAnalysis: {:?}", item)
}

/// Convert BufferZoneVerifier description to a summary string.
pub fn bufferzoneverifier_to_summary(item: &BufferZoneVerifier) -> String {
    format!("BufferZoneVerifier: {:?}", item)
}

/// Convert MemoryAccessLog description to a summary string.
pub fn memoryaccesslog_to_summary(item: &MemoryAccessLog) -> String {
    format!("MemoryAccessLog: {:?}", item)
}

/// Convert CoalescingAnalysis description to a summary string.
pub fn coalescinganalysis_to_summary(item: &CoalescingAnalysis) -> String {
    format!("CoalescingAnalysis: {:?}", item)
}

/// Convert BankConflictDetector description to a summary string.
pub fn bankconflictdetector_to_summary(item: &BankConflictDetector) -> String {
    format!("BankConflictDetector: {:?}", item)
}

/// Convert SharedMemoryPadding description to a summary string.
pub fn sharedmemorypadding_to_summary(item: &SharedMemoryPadding) -> String {
    format!("SharedMemoryPadding: {:?}", item)
}

/// Convert MemoryFenceAnalysis description to a summary string.
pub fn memoryfenceanalysis_to_summary(item: &MemoryFenceAnalysis) -> String {
    format!("MemoryFenceAnalysis: {:?}", item)
}

/// Batch processor for memsafe operations.
#[derive(Debug, Clone)]
pub struct MemsafeBatchProcessor {
    pub batch_size: usize,
    pub processed: usize,
    pub errors: Vec<String>,
    pub results: Vec<f64>,
}

impl MemsafeBatchProcessor {
    pub fn new(batch_size: usize) -> Self {
        MemsafeBatchProcessor { batch_size, processed: 0, errors: Vec::new(), results: Vec::new() }
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

impl fmt::Display for MemsafeBatchProcessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MemsafeBatch(processed={}, errors={})", self.processed, self.errors.len())
    }
}

/// Detailed report for memsafe analysis.
#[derive(Debug, Clone)]
pub struct MemsafeReport {
    pub title: String,
    pub sections: Vec<(String, Vec<String>)>,
    pub metrics: Vec<(String, f64)>,
    pub warnings: Vec<String>,
    pub timestamp: u64,
}

impl MemsafeReport {
    pub fn new(title: impl Into<String>) -> Self {
        MemsafeReport { title: title.into(), sections: Vec::new(), metrics: Vec::new(), warnings: Vec::new(), timestamp: 0 }
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

impl fmt::Display for MemsafeReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MemsafeReport({})", self.title)
    }
}

/// Configuration for memsafe analysis.
#[derive(Debug, Clone)]
pub struct MemsafeConfig {
    pub verbose: bool,
    pub max_iterations: usize,
    pub tolerance: f64,
    pub timeout_ms: u64,
    pub parallel: bool,
    pub output_format: String,
}

impl MemsafeConfig {
    pub fn default_config() -> Self {
        MemsafeConfig {
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

impl fmt::Display for MemsafeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MemsafeConfig(iters={}, tol={:.0e})", self.max_iterations, self.tolerance)
    }
}

/// Histogram for memsafe data distribution.
#[derive(Debug, Clone)]
pub struct MemsafeHistogram {
    pub bins: Vec<usize>,
    pub bin_edges: Vec<f64>,
    pub total_count: usize,
}

impl MemsafeHistogram {
    pub fn from_data(data: &[f64], num_bins: usize) -> Self {
        if data.is_empty() || num_bins == 0 {
            return MemsafeHistogram { bins: Vec::new(), bin_edges: Vec::new(), total_count: 0 };
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
        MemsafeHistogram { bins, bin_edges, total_count: data.len() }
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

impl fmt::Display for MemsafeHistogram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Histogram(bins={}, total={})", self.num_bins(), self.total_count)
    }
}

/// Adjacency/weight matrix for memsafe graph analysis.
#[derive(Debug, Clone)]
pub struct MemsafeGraph {
    pub adjacency: Vec<Vec<bool>>,
    pub weights: Vec<Vec<f64>>,
    pub node_count: usize,
    pub edge_count: usize,
    pub node_labels: Vec<String>,
}

impl MemsafeGraph {
    pub fn new(n: usize) -> Self {
        MemsafeGraph {
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
        fn dfs_cycle_memsafe(v: usize, adj: &[Vec<bool>], visited: &mut [u8]) -> bool {
            visited[v] = 1;
            for w in 0..adj.len() { if adj[v][w] {
                if visited[w] == 1 { return true; }
                if visited[w] == 0 && dfs_cycle_memsafe(w, adj, visited) { return true; }
            }}
            visited[v] = 2; false
        }
        for i in 0..n {
            if visited[i] == 0 && dfs_cycle_memsafe(i, &self.adjacency, &mut visited) { return false; }
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

impl fmt::Display for MemsafeGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MemsafeGraph(n={}, e={})", self.node_count, self.edge_count)
    }
}

/// Cache for memsafe computation results.
#[derive(Debug, Clone)]
pub struct MemsafeCache {
    entries: Vec<(u64, Vec<f64>)>,
    capacity: usize,
    hits: u64,
    misses: u64,
}

impl MemsafeCache {
    pub fn new(capacity: usize) -> Self {
        MemsafeCache { entries: Vec::new(), capacity, hits: 0, misses: 0 }
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

impl fmt::Display for MemsafeCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cache(size={}, hit_rate={:.1}%)", self.size(), self.hit_rate() * 100.0)
    }
}

/// Compute pairwise distances for memsafe elements.
pub fn memsafe_pairwise_distances(points: &[Vec<f64>]) -> Vec<Vec<f64>> {
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

/// K-means clustering for memsafe data.
pub fn memsafe_kmeans(data: &[Vec<f64>], k: usize, max_iters: usize) -> Vec<usize> {
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

/// Principal component analysis (simplified) for memsafe data.
pub fn memsafe_pca_2d(data: &[Vec<f64>]) -> Vec<(f64, f64)> {
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

/// Dense matrix operations for MemSafe computations.
#[derive(Debug, Clone)]
pub struct MemSafeDenseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl MemSafeDenseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        MemSafeDenseMatrix { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::new(n, n);
        for i in 0..n { m.data[i * n + i] = 1.0; }
        m
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols);
        MemSafeDenseMatrix { rows, cols, data }
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
        MemSafeDenseMatrix { rows: self.rows, cols: self.cols, data }
    }

    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows); assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a - b).collect();
        MemSafeDenseMatrix { rows: self.rows, cols: self.cols, data }
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
        MemSafeDenseMatrix { rows: self.rows, cols: self.cols, data }
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
        MemSafeDenseMatrix { rows: self.rows, cols: self.cols, data }
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

impl fmt::Display for MemSafeDenseMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MemSafeMatrix({}x{})", self.rows, self.cols)
    }
}

/// Interval arithmetic for MemSafe bounds analysis.
#[derive(Debug, Clone, Copy)]
pub struct MemSafeInterval {
    pub lo: f64,
    pub hi: f64,
}

impl MemSafeInterval {
    pub fn new(lo: f64, hi: f64) -> Self {
        MemSafeInterval { lo: lo.min(hi), hi: lo.max(hi) }
    }

    pub fn point(v: f64) -> Self {
        MemSafeInterval { lo: v, hi: v }
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
        MemSafeInterval { lo: self.lo.min(other.lo), hi: self.hi.max(other.hi) }
    }

    pub fn intersect(&self, other: &Self) -> Option<Self> {
        let lo = self.lo.max(other.lo);
        let hi = self.hi.min(other.hi);
        if lo <= hi { Some(MemSafeInterval { lo, hi }) } else { None }
    }

    pub fn add(&self, other: &Self) -> Self {
        MemSafeInterval { lo: self.lo + other.lo, hi: self.hi + other.hi }
    }

    pub fn sub(&self, other: &Self) -> Self {
        MemSafeInterval { lo: self.lo - other.hi, hi: self.hi - other.lo }
    }

    pub fn mul(&self, other: &Self) -> Self {
        let products = [self.lo * other.lo, self.lo * other.hi, self.hi * other.lo, self.hi * other.hi];
        let lo = products.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = products.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        MemSafeInterval { lo, hi }
    }

    pub fn abs(&self) -> Self {
        if self.lo >= 0.0 { *self }
        else if self.hi <= 0.0 { MemSafeInterval { lo: -self.hi, hi: -self.lo } }
        else { MemSafeInterval { lo: 0.0, hi: self.lo.abs().max(self.hi.abs()) } }
    }

    pub fn sqrt(&self) -> Self {
        let lo = if self.lo >= 0.0 { self.lo.sqrt() } else { 0.0 };
        MemSafeInterval { lo, hi: self.hi.max(0.0).sqrt() }
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

impl fmt::Display for MemSafeInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:.4}, {:.4}]", self.lo, self.hi)
    }
}

/// State machine for MemSafe protocol modeling.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MemSafeState {
    Clean,
    Scanning,
    Tainted,
    Sanitizing,
    Verified,
    Compromised,
}

impl fmt::Display for MemSafeState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemSafeState::Clean => write!(f, "clean"),
            MemSafeState::Scanning => write!(f, "scanning"),
            MemSafeState::Tainted => write!(f, "tainted"),
            MemSafeState::Sanitizing => write!(f, "sanitizing"),
            MemSafeState::Verified => write!(f, "verified"),
            MemSafeState::Compromised => write!(f, "compromised"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemSafeStateMachine {
    pub current: MemSafeState,
    pub history: Vec<String>,
    pub transition_count: usize,
}

impl MemSafeStateMachine {
    pub fn new() -> Self {
        MemSafeStateMachine { current: MemSafeState::Clean, history: Vec::new(), transition_count: 0 }
    }
    pub fn state(&self) -> &MemSafeState { &self.current }
    pub fn can_transition(&self, target: &MemSafeState) -> bool {
        match (&self.current, target) {
            (MemSafeState::Clean, MemSafeState::Scanning) => true,
            (MemSafeState::Scanning, MemSafeState::Tainted) => true,
            (MemSafeState::Scanning, MemSafeState::Verified) => true,
            (MemSafeState::Tainted, MemSafeState::Sanitizing) => true,
            (MemSafeState::Sanitizing, MemSafeState::Verified) => true,
            (MemSafeState::Sanitizing, MemSafeState::Compromised) => true,
            (MemSafeState::Compromised, MemSafeState::Clean) => true,
            (MemSafeState::Verified, MemSafeState::Clean) => true,
            _ => false,
        }
    }
    pub fn transition(&mut self, target: MemSafeState) -> bool {
        if self.can_transition(&target) {
            self.history.push(format!("{} -> {}", self.current, target));
            self.current = target;
            self.transition_count += 1;
            true
        } else { false }
    }
    pub fn reset(&mut self) {
        self.current = MemSafeState::Clean;
        self.history.clear();
        self.transition_count = 0;
    }
    pub fn history_len(&self) -> usize { self.history.len() }
}

impl fmt::Display for MemSafeStateMachine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SM(state={}, transitions={})", self.current, self.transition_count)
    }
}

/// Ring buffer for MemSafe event tracking.
#[derive(Debug, Clone)]
pub struct MemSafeRingBuffer {
    data: Vec<f64>,
    capacity: usize,
    head: usize,
    count: usize,
}

impl MemSafeRingBuffer {
    pub fn new(capacity: usize) -> Self {
        MemSafeRingBuffer { data: vec![0.0; capacity], capacity, head: 0, count: 0 }
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

impl fmt::Display for MemSafeRingBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RingBuffer(len={}/{})", self.count, self.capacity)
    }
}

/// Disjoint set (union-find) for MemSafe component tracking.
#[derive(Debug, Clone)]
pub struct MemSafeDisjointSet {
    parent: Vec<usize>,
    rank: Vec<usize>,
    size: Vec<usize>,
    num_components: usize,
}

impl MemSafeDisjointSet {
    pub fn new(n: usize) -> Self {
        MemSafeDisjointSet { parent: (0..n).collect(), rank: vec![0; n], size: vec![1; n], num_components: n }
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

impl fmt::Display for MemSafeDisjointSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DisjointSet(n={}, components={})", self.parent.len(), self.num_components)
    }
}

/// Sorted list with binary search for MemSafe.
#[derive(Debug, Clone)]
pub struct MemSafeSortedList {
    data: Vec<f64>,
}

impl MemSafeSortedList {
    pub fn new() -> Self { MemSafeSortedList { data: Vec::new() } }
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

impl fmt::Display for MemSafeSortedList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SortedList(len={})", self.data.len())
    }
}

/// Exponential moving average for MemSafe metrics.
#[derive(Debug, Clone)]
pub struct MemSafeEma {
    pub alpha: f64,
    pub value: f64,
    pub count: usize,
    pub initialized: bool,
}

impl MemSafeEma {
    pub fn new(alpha: f64) -> Self { MemSafeEma { alpha, value: 0.0, count: 0, initialized: false } }
    pub fn update(&mut self, sample: f64) {
        if !self.initialized { self.value = sample; self.initialized = true; }
        else { self.value = self.alpha * sample + (1.0 - self.alpha) * self.value; }
        self.count += 1;
    }
    pub fn current(&self) -> f64 { self.value }
    pub fn reset(&mut self) { self.value = 0.0; self.count = 0; self.initialized = false; }
}

impl fmt::Display for MemSafeEma {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EMA(alpha={:.2}, value={:.4})", self.alpha, self.value)
    }
}

/// Simple bloom filter for MemSafe membership testing.
#[derive(Debug, Clone)]
pub struct MemSafeBloomFilter {
    bits: Vec<bool>,
    num_hashes: usize,
    size: usize,
    count: usize,
}

impl MemSafeBloomFilter {
    pub fn new(size: usize, num_hashes: usize) -> Self {
        MemSafeBloomFilter { bits: vec![false; size], num_hashes, size, count: 0 }
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

impl fmt::Display for MemSafeBloomFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BloomFilter(size={}, count={}, fpr={:.4})", self.size, self.count, self.false_positive_rate())
    }
}

/// Simple prefix trie for MemSafe string matching.
#[derive(Debug, Clone)]
pub struct MemSafeTrieNode {
    children: Vec<(char, usize)>,
    is_terminal: bool,
    value: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct MemSafeTrie {
    nodes: Vec<MemSafeTrieNode>,
    count: usize,
}

impl MemSafeTrie {
    pub fn new() -> Self {
        MemSafeTrie { nodes: vec![MemSafeTrieNode { children: Vec::new(), is_terminal: false, value: None }], count: 0 }
    }
    pub fn insert(&mut self, key: &str, value: u64) {
        let mut current = 0;
        for ch in key.chars() {
            let next = self.nodes[current].children.iter().find(|(c, _)| *c == ch).map(|(_, idx)| *idx);
            current = match next {
                Some(idx) => idx,
                None => {
                    let idx = self.nodes.len();
                    self.nodes.push(MemSafeTrieNode { children: Vec::new(), is_terminal: false, value: None });
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

impl fmt::Display for MemSafeTrie {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Trie(entries={}, nodes={})", self.count, self.nodes.len())
    }
}

/// Min-heap priority queue for MemSafe scheduling.
#[derive(Debug, Clone)]
pub struct MemSafePriorityQueue {
    heap: Vec<(f64, usize)>,
}

impl MemSafePriorityQueue {
    pub fn new() -> Self { MemSafePriorityQueue { heap: Vec::new() } }
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

impl fmt::Display for MemSafePriorityQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PQ(len={})", self.heap.len())
    }
}

/// Online statistics accumulator for MemSafe.
#[derive(Debug, Clone)]
pub struct MemSafeAccumulator {
    count: u64,
    mean: f64,
    m2: f64,
    min_val: f64,
    max_val: f64,
    sum: f64,
}

impl MemSafeAccumulator {
    pub fn new() -> Self { MemSafeAccumulator { count: 0, mean: 0.0, m2: 0.0, min_val: f64::INFINITY, max_val: f64::NEG_INFINITY, sum: 0.0 } }
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

impl fmt::Display for MemSafeAccumulator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Acc(n={}, mean={:.4}, std={:.4})", self.count, self.mean, self.std_dev())
    }
}

/// Sparse matrix (COO format) for MemSafe.
#[derive(Debug, Clone)]
pub struct MemSafeSparseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub entries: Vec<(usize, usize, f64)>,
}

impl MemSafeSparseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self { MemSafeSparseMatrix { rows, cols, entries: Vec::new() } }
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
        let mut result = MemSafeSparseMatrix::new(self.cols, self.rows);
        for &(i, j, v) in &self.entries { result.entries.push((j, i, v)); }
        result
    }
    pub fn add(&self, other: &Self) -> Self {
        let mut result = MemSafeSparseMatrix::new(self.rows, self.cols);
        for &(i, j, v) in &self.entries { result.insert(i, j, result.get(i, j) + v); }
        for &(i, j, v) in &other.entries { result.insert(i, j, result.get(i, j) + v); }
        result
    }
    pub fn scale(&self, s: f64) -> Self {
        let mut result = MemSafeSparseMatrix::new(self.rows, self.cols);
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

impl fmt::Display for MemSafeSparseMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sparse({}x{}, nnz={})", self.rows, self.cols, self.nnz())
    }
}

/// Polynomial with f64 coefficients for MemSafe.
#[derive(Debug, Clone)]
pub struct MemSafePolynomial {
    pub coefficients: Vec<f64>,
}

impl MemSafePolynomial {
    pub fn new(coeffs: Vec<f64>) -> Self { MemSafePolynomial { coefficients: coeffs } }
    pub fn zero() -> Self { MemSafePolynomial { coefficients: vec![0.0] } }
    pub fn one() -> Self { MemSafePolynomial { coefficients: vec![1.0] } }
    pub fn monomial(degree: usize, coeff: f64) -> Self {
        let mut c = vec![0.0; degree + 1];
        c[degree] = coeff;
        MemSafePolynomial { coefficients: c }
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
        MemSafePolynomial { coefficients: result }
    }
    pub fn sub(&self, other: &Self) -> Self {
        let n = self.coefficients.len().max(other.coefficients.len());
        let mut result = vec![0.0; n];
        for (i, &c) in self.coefficients.iter().enumerate() { result[i] += c; }
        for (i, &c) in other.coefficients.iter().enumerate() { result[i] -= c; }
        MemSafePolynomial { coefficients: result }
    }
    pub fn mul(&self, other: &Self) -> Self {
        let n = self.coefficients.len() + other.coefficients.len() - 1;
        let mut result = vec![0.0; n];
        for (i, &a) in self.coefficients.iter().enumerate() {
            for (j, &b) in other.coefficients.iter().enumerate() {
                result[i + j] += a * b;
            }
        }
        MemSafePolynomial { coefficients: result }
    }
    pub fn scale(&self, s: f64) -> Self {
        MemSafePolynomial { coefficients: self.coefficients.iter().map(|&c| c * s).collect() }
    }
    pub fn derivative(&self) -> Self {
        if self.coefficients.len() <= 1 { return Self::zero(); }
        let coeffs: Vec<f64> = self.coefficients.iter().enumerate().skip(1)
            .map(|(i, &c)| c * i as f64).collect();
        MemSafePolynomial { coefficients: coeffs }
    }
    pub fn integral(&self, constant: f64) -> Self {
        let mut coeffs = vec![constant];
        for (i, &c) in self.coefficients.iter().enumerate() {
            coeffs.push(c / (i + 1) as f64);
        }
        MemSafePolynomial { coefficients: coeffs }
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

impl fmt::Display for MemSafePolynomial {
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

/// Simple linear congruential generator for MemSafe.
#[derive(Debug, Clone)]
pub struct MemSafeRng {
    state: u64,
}

impl MemSafeRng {
    pub fn new(seed: u64) -> Self { MemSafeRng { state: seed } }
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

impl fmt::Display for MemSafeRng {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Rng(state={:#x})", self.state)
    }
}

/// Simple timing utility for MemSafe benchmarking.
#[derive(Debug, Clone)]
pub struct MemSafeTimer {
    pub label: String,
    pub elapsed_ns: Vec<u64>,
    pub running: bool,
}

impl MemSafeTimer {
    pub fn new(label: impl Into<String>) -> Self { MemSafeTimer { label: label.into(), elapsed_ns: Vec::new(), running: false } }
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

impl fmt::Display for MemSafeTimer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Timer({}: avg={:.0}ns, n={})", self.label, self.average_ns(), self.count())
    }
}

/// Compact bit vector for MemSafe set operations.
#[derive(Debug, Clone)]
pub struct MemSafeBitVector {
    words: Vec<u64>,
    len: usize,
}

impl MemSafeBitVector {
    pub fn new(len: usize) -> Self { MemSafeBitVector { words: vec![0u64; (len + 63) / 64], len } }
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

impl fmt::Display for MemSafeBitVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BitVec(len={}, ones={})", self.len, self.count_ones())
    }
}

/// LRU cache for MemSafe computation memoization.
#[derive(Debug, Clone)]
pub struct MemSafeLruCache {
    entries: Vec<(u64, Vec<f64>, u64)>,
    capacity: usize,
    clock: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl MemSafeLruCache {
    pub fn new(capacity: usize) -> Self { MemSafeLruCache { entries: Vec::new(), capacity, clock: 0, hits: 0, misses: 0, evictions: 0 } }
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

impl fmt::Display for MemSafeLruCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LRU(size={}/{}, hr={:.1}%)", self.size(), self.capacity, self.hit_rate() * 100.0)
    }
}

/// Graph coloring utility for MemSafe scheduling.
#[derive(Debug, Clone)]
pub struct MemSafeGraphColoring {
    pub adjacency: Vec<Vec<bool>>,
    pub colors: Vec<Option<usize>>,
    pub num_nodes: usize,
    pub num_colors_used: usize,
}

impl MemSafeGraphColoring {
    pub fn new(n: usize) -> Self {
        MemSafeGraphColoring { adjacency: vec![vec![false; n]; n], colors: vec![None; n], num_nodes: n, num_colors_used: 0 }
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

impl fmt::Display for MemSafeGraphColoring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Coloring(n={}, colors={})", self.num_nodes, self.num_colors_used)
    }
}

/// Top-K tracker for MemSafe ranking.
#[derive(Debug, Clone)]
pub struct MemSafeTopK {
    pub k: usize,
    pub items: Vec<(f64, String)>,
}

impl MemSafeTopK {
    pub fn new(k: usize) -> Self { MemSafeTopK { k, items: Vec::new() } }
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

impl fmt::Display for MemSafeTopK {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TopK(k={}, len={})", self.k, self.items.len())
    }
}

/// Sliding window statistics for MemSafe monitoring.
#[derive(Debug, Clone)]
pub struct MemSafeSlidingWindow {
    pub window_size: usize,
    pub data: Vec<f64>,
    pub sum: f64,
}

impl MemSafeSlidingWindow {
    pub fn new(window_size: usize) -> Self { MemSafeSlidingWindow { window_size, data: Vec::new(), sum: 0.0 } }
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

impl fmt::Display for MemSafeSlidingWindow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Window(size={}/{}, mean={:.2})", self.data.len(), self.window_size, self.mean())
    }
}

/// Confusion matrix for MemSafe classification evaluation.
#[derive(Debug, Clone)]
pub struct MemSafeConfusionMatrix {
    pub true_positive: u64,
    pub false_positive: u64,
    pub true_negative: u64,
    pub false_negative: u64,
}

impl MemSafeConfusionMatrix {
    pub fn new() -> Self { MemSafeConfusionMatrix { true_positive: 0, false_positive: 0, true_negative: 0, false_negative: 0 } }
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

impl fmt::Display for MemSafeConfusionMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CM(acc={:.3}, prec={:.3}, rec={:.3}, f1={:.3})",
            self.accuracy(), self.precision(), self.recall(), self.f1_score())
    }
}

/// Cosine similarity for MemSafe feature vectors.
pub fn memsafe_cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na * nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

/// Euclidean distance for MemSafe.
pub fn memsafe_euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
}

/// Manhattan distance for MemSafe.
pub fn memsafe_manhattan_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// Chebyshev distance for MemSafe.
pub fn memsafe_chebyshev_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max)
}

/// Minkowski distance for MemSafe.
pub fn memsafe_minkowski_distance(a: &[f64], b: &[f64], p: f64) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs().powf(p)).sum::<f64>().powf(1.0 / p)
}

/// Normalize a vector for MemSafe.
pub fn memsafe_normalize(v: &[f64]) -> Vec<f64> {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 { v.to_vec() } else { v.iter().map(|x| x / norm).collect() }
}

/// Dot product for MemSafe.
pub fn memsafe_dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Cross product (3D) for MemSafe.
pub fn memsafe_cross_product(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
}

/// Linear interpolation for MemSafe.
pub fn memsafe_lerp(a: f64, b: f64, t: f64) -> f64 { a + (b - a) * t }

/// Clamp value for MemSafe.
pub fn memsafe_clamp(v: f64, lo: f64, hi: f64) -> f64 { v.max(lo).min(hi) }

/// Sigmoid function for MemSafe.
pub fn memsafe_sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }

/// Softmax for MemSafe.
pub fn memsafe_softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Log-sum-exp for MemSafe.
pub fn memsafe_logsumexp(values: &[f64]) -> f64 {
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    max + values.iter().map(|&v| (v - max).exp()).sum::<f64>().ln()
}

/// KL divergence for MemSafe.
pub fn memsafe_kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q.iter()).map(|(&pi, &qi)| {
        if pi > 0.0 && qi > 0.0 { pi * (pi / qi).ln() } else { 0.0 }
    }).sum()
}

/// Jensen-Shannon divergence for MemSafe.
pub fn memsafe_js_divergence(p: &[f64], q: &[f64]) -> f64 {
    let m: Vec<f64> = p.iter().zip(q.iter()).map(|(&pi, &qi)| (pi + qi) / 2.0).collect();
    (memsafe_kl_divergence(p, &m) + memsafe_kl_divergence(q, &m)) / 2.0
}

/// Total variation distance for MemSafe.
pub fn memsafe_tv_distance(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q.iter()).map(|(&pi, &qi)| (pi - qi).abs()).sum::<f64>() / 2.0
}

/// Hellinger distance for MemSafe.
pub fn memsafe_hellinger_distance(p: &[f64], q: &[f64]) -> f64 {
    let sum: f64 = p.iter().zip(q.iter()).map(|(&pi, &qi)| {
        let diff = pi.sqrt() - qi.sqrt();
        diff * diff
    }).sum();
    (sum / 2.0).sqrt()
}

/// Earth mover's distance (1D) for MemSafe.
pub fn memsafe_emd_1d(p: &[f64], q: &[f64]) -> f64 {
    let mut cum_diff = 0.0;
    let mut total = 0.0;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        cum_diff += pi - qi;
        total += cum_diff.abs();
    }
    total
}

/// Feature scaling utilities for MemSafe.
#[derive(Debug, Clone)]
pub struct MemSafeFeatureScaler {
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
    pub mins: Vec<f64>,
    pub maxs: Vec<f64>,
    pub fitted: bool,
}

impl MemSafeFeatureScaler {
    pub fn new() -> Self { MemSafeFeatureScaler { means: Vec::new(), stds: Vec::new(), mins: Vec::new(), maxs: Vec::new(), fitted: false } }
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

impl fmt::Display for MemSafeFeatureScaler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Scaler(dim={}, fitted={})", self.dimension(), self.fitted)
    }
}

/// Simple linear regression for MemSafe trend analysis.
#[derive(Debug, Clone)]
pub struct MemSafeLinearRegression {
    pub slope: f64,
    pub intercept: f64,
    pub r_squared: f64,
    pub fitted: bool,
}

impl MemSafeLinearRegression {
    pub fn new() -> Self { MemSafeLinearRegression { slope: 0.0, intercept: 0.0, r_squared: 0.0, fitted: false } }
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

impl fmt::Display for MemSafeLinearRegression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "y = {:.4}x + {:.4} (R2={:.4})", self.slope, self.intercept, self.r_squared)
    }
}

/// Weighted undirected graph for MemSafe.
#[derive(Debug, Clone)]
pub struct MemSafeWeightedGraph {
    pub adj: Vec<Vec<(usize, f64)>>,
    pub num_nodes: usize,
    pub num_edges: usize,
}

impl MemSafeWeightedGraph {
    pub fn new(n: usize) -> Self { MemSafeWeightedGraph { adj: vec![Vec::new(); n], num_nodes: n, num_edges: 0 } }
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
        fn find_memsafe(parent: &mut Vec<usize>, x: usize) -> usize {
            if parent[x] != x { parent[x] = find_memsafe(parent, parent[x]); }
            parent[x]
        }
        let mut total = 0.0;
        let mut count = 0;
        for (w, u, v) in edges {
            let ru = find_memsafe(&mut parent, u);
            let rv = find_memsafe(&mut parent, v);
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

impl fmt::Display for MemSafeWeightedGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WGraph(n={}, e={})", self.num_nodes, self.num_edges)
    }
}

/// Moving average for MemSafe.
pub fn memsafe_moving_average(data: &[f64], window: usize) -> Vec<f64> {
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

/// Cumulative sum for MemSafe.
pub fn memsafe_cumsum(data: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(data.len());
    let mut sum = 0.0;
    for &v in data { sum += v; result.push(sum); }
    result
}

/// Numerical differentiation for MemSafe.
pub fn memsafe_diff(data: &[f64]) -> Vec<f64> {
    if data.len() < 2 { return Vec::new(); }
    data.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Auto-correlation for MemSafe.
pub fn memsafe_autocorrelation(data: &[f64], lag: usize) -> f64 {
    if data.len() <= lag { return 0.0; }
    let n = data.len();
    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let var: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if var.abs() < 1e-15 { return 0.0; }
    let cov: f64 = (0..(n - lag)).map(|i| (data[i] - mean) * (data[i + lag] - mean)).sum::<f64>() / n as f64;
    cov / var
}

/// Discrete Fourier transform magnitude for MemSafe.
pub fn memsafe_dft_magnitude(data: &[f64]) -> Vec<f64> {
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

/// Trapezoidal integration for MemSafe.
pub fn memsafe_integrate_trapezoid(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let mut total = 0.0;
    for i in 1..x.len() {
        total += (x[i] - x[i-1]) * (y[i] + y[i-1]) / 2.0;
    }
    total
}

/// Simpson's rule integration for MemSafe.
pub fn memsafe_integrate_simpson(x: &[f64], y: &[f64]) -> f64 {
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

/// Convolution for MemSafe.
pub fn memsafe_convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len() + b.len() - 1;
    let mut result = vec![0.0; n];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            result[i + j] += ai * bj;
        }
    }
    result
}

/// Histogram for MemSafe data analysis.
#[derive(Debug, Clone)]
pub struct MemSafeHistogramExt {
    pub bins: Vec<usize>,
    pub edges: Vec<f64>,
    pub total: usize,
}

impl MemSafeHistogramExt {
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
        MemSafeHistogramExt { bins, edges, total: data.len() }
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

impl fmt::Display for MemSafeHistogramExt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hist(bins={}, total={})", self.num_bins(), self.total)
    }
}

/// Axis-aligned bounding box for MemSafe spatial indexing.
#[derive(Debug, Clone, Copy)]
pub struct MemSafeAABB {
    pub x_min: f64, pub y_min: f64,
    pub x_max: f64, pub y_max: f64,
}

impl MemSafeAABB {
    pub fn new(x_min: f64, y_min: f64, x_max: f64, y_max: f64) -> Self { MemSafeAABB { x_min, y_min, x_max, y_max } }
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
            MemSafeAABB::new(self.x_min, self.y_min, cx, cy),
            MemSafeAABB::new(cx, self.y_min, self.x_max, cy),
            MemSafeAABB::new(self.x_min, cy, cx, self.y_max),
            MemSafeAABB::new(cx, cy, self.x_max, self.y_max),
        ]
    }
}

/// 2D point for MemSafe.
#[derive(Debug, Clone, Copy)]
pub struct MemSafePoint2D { pub x: f64, pub y: f64, pub data: f64 }

/// Quadtree for MemSafe spatial indexing.
#[derive(Debug, Clone)]
pub struct MemSafeQuadTree {
    pub boundary: MemSafeAABB,
    pub points: Vec<MemSafePoint2D>,
    pub children: Option<Vec<MemSafeQuadTree>>,
    pub capacity: usize,
    pub depth: usize,
    pub max_depth: usize,
}

impl MemSafeQuadTree {
    pub fn new(boundary: MemSafeAABB, capacity: usize, max_depth: usize) -> Self {
        MemSafeQuadTree { boundary, points: Vec::new(), children: None, capacity, depth: 0, max_depth }
    }
    fn with_depth(boundary: MemSafeAABB, capacity: usize, depth: usize, max_depth: usize) -> Self {
        MemSafeQuadTree { boundary, points: Vec::new(), children: None, capacity, depth, max_depth }
    }
    pub fn insert(&mut self, p: MemSafePoint2D) -> bool {
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
            children.push(MemSafeQuadTree::with_depth(*q, self.capacity, self.depth + 1, self.max_depth));
        }
        let old_points: Vec<_> = self.points.drain(..).collect();
        self.children = Some(children);
        for p in old_points { self.insert(p); }
    }
    pub fn query_range(&self, range: &MemSafeAABB) -> Vec<MemSafePoint2D> {
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

impl fmt::Display for MemSafeQuadTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QTree(count={}, depth={})", self.count(), self.tree_depth())
    }
}

/// QR decomposition helper for MemSafe.
pub fn memsafe_qr_decompose(a: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
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

/// Solve upper triangular system Rx = b for MemSafe.
pub fn memsafe_solve_upper_triangular(r: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i+1)..n { s -= r[i][j] * x[j]; }
        x[i] = if r[i][i].abs() > 1e-15 { s / r[i][i] } else { 0.0 };
    }
    x
}

/// Matrix-vector multiply for MemSafe.
pub fn memsafe_mat_vec_mul(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter().map(|row| row.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum()).collect()
}

/// Matrix transpose for MemSafe.
pub fn memsafe_transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() { return Vec::new(); }
    let m = a.len(); let n = a[0].len();
    (0..n).map(|j| (0..m).map(|i| a[i][j]).collect()).collect()
}

/// Matrix multiply for MemSafe.
pub fn memsafe_mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    if m == 0 { return Vec::new(); }
    let k = a[0].len();
    let n = b[0].len();
    let mut c = vec![vec![0.0; n]; m];
    for i in 0..m { for j in 0..n { for l in 0..k { c[i][j] += a[i][l] * b[l][j]; } } }
    c
}

/// Frobenius norm for MemSafe.
pub fn memsafe_frobenius_norm(a: &[Vec<f64>]) -> f64 {
    a.iter().flat_map(|row| row.iter()).map(|&x| x * x).sum::<f64>().sqrt()
}

/// Matrix trace for MemSafe.
pub fn memsafe_trace(a: &[Vec<f64>]) -> f64 {
    a.iter().enumerate().map(|(i, row)| if i < row.len() { row[i] } else { 0.0 }).sum()
}

/// Identity matrix for MemSafe.
pub fn memsafe_identity(n: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n { m[i][i] = 1.0; }
    m
}

/// Power iteration for dominant eigenvalue for MemSafe.
pub fn memsafe_power_iteration(a: &[Vec<f64>], max_iter: usize) -> (f64, Vec<f64>) {
    let n = a.len();
    let mut v = vec![1.0; n];
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut v { *x /= norm; }
    let mut eigenvalue = 0.0;
    for _ in 0..max_iter {
        let av = memsafe_mat_vec_mul(a, &v);
        let norm: f64 = av.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm.abs() < 1e-15 { break; }
        eigenvalue = av.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum();
        v = av.iter().map(|&x| x / norm).collect();
    }
    (eigenvalue, v)
}

/// Running statistics with min/max tracking for MemSafe.
#[derive(Debug, Clone)]
pub struct MemSafeRunningStats {
    pub count: u64,
    pub mean: f64,
    pub m2: f64,
    pub min_val: f64,
    pub max_val: f64,
    pub sum: f64,
}

impl MemSafeRunningStats {
    pub fn new() -> Self { MemSafeRunningStats { count: 0, mean: 0.0, m2: 0.0, min_val: f64::INFINITY, max_val: f64::NEG_INFINITY, sum: 0.0 } }
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

impl fmt::Display for MemSafeRunningStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Stats(n={}, mean={:.4}, std={:.4})", self.count, self.mean, self.std_dev())
    }
}

/// Interquartile range for MemSafe.
pub fn memsafe_iqr(data: &[f64]) -> f64 {
    memsafe_percentile_at(data, 75.0) - memsafe_percentile_at(data, 25.0)
}

/// Detect outliers using IQR method for MemSafe.
pub fn memsafe_outliers(data: &[f64]) -> Vec<usize> {
    let q1 = memsafe_percentile_at(data, 25.0);
    let q3 = memsafe_percentile_at(data, 75.0);
    let iqr = q3 - q1;
    let lower = q1 - 1.5 * iqr;
    let upper = q3 + 1.5 * iqr;
    data.iter().enumerate().filter(|(_, &v)| v < lower || v > upper).map(|(i, _)| i).collect()
}

/// Z-score normalization for MemSafe.
pub fn memsafe_zscore(data: &[f64]) -> Vec<f64> {
    let n = data.len() as f64;
    if n < 2.0 { return data.to_vec(); }
    let mean = data.iter().sum::<f64>() / n;
    let std = (data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    if std.abs() < 1e-15 { return vec![0.0; data.len()]; }
    data.iter().map(|&x| (x - mean) / std).collect()
}

/// Rank values for MemSafe.
pub fn memsafe_rank(data: &[f64]) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut ranks = vec![0.0; data.len()];
    for (rank, &(idx, _)) in indexed.iter().enumerate() { ranks[idx] = (rank + 1) as f64; }
    ranks
}

/// Spearman rank correlation for MemSafe.
pub fn memsafe_spearman(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let rx = memsafe_rank(x);
    let ry = memsafe_rank(y);
    let n = x.len() as f64;
    let d_sq: f64 = rx.iter().zip(ry.iter()).map(|(&a, &b)| (a - b).powi(2)).sum();
    1.0 - 6.0 * d_sq / (n * (n * n - 1.0))
}

/// Covariance matrix for MemSafe.
pub fn memsafe_covariance_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
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

/// Correlation matrix for MemSafe.
pub fn memsafe_correlation_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let cov = memsafe_covariance_matrix(data);
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

    #[test]
    fn test_memory_region_contains() {
        let region = MemoryRegion::new("r1", 100, 50, MemoryRegionKind::Global);
        assert!(region.contains(100));
        assert!(region.contains(149));
        assert!(!region.contains(150));
        assert!(!region.contains(99));
    }

    #[test]
    fn test_memory_region_overlaps() {
        let r1 = MemoryRegion::new("r1", 100, 50, MemoryRegionKind::Global);
        let r2 = MemoryRegion::new("r2", 120, 50, MemoryRegionKind::Shared);
        let r3 = MemoryRegion::new("r3", 200, 50, MemoryRegionKind::Local);
        assert!(r1.overlaps(&r2));
        assert!(!r1.overlaps(&r3));
        assert_eq!(r1.overlap_size(&r2), 30);
        assert_eq!(r1.overlap_size(&r3), 0);
    }

    #[test]
    fn test_memory_layout_add_and_find() {
        let mut layout = MemoryLayout::new();
        layout.add_region(MemoryRegion::new("r1", 0, 100, MemoryRegionKind::Global)).unwrap();
        layout.add_region(MemoryRegion::new("r2", 200, 100, MemoryRegionKind::Shared)).unwrap();

        assert!(layout.find_region(50).is_some());
        assert_eq!(layout.find_region(50).unwrap().id, "r1");
        assert!(layout.find_region(250).is_some());
        assert_eq!(layout.find_region(250).unwrap().id, "r2");
        assert!(layout.find_region(150).is_none());
    }

    #[test]
    fn test_memory_layout_overlap_detection() {
        let mut layout = MemoryLayout::new();
        layout.add_region(MemoryRegion::new("r1", 0, 100, MemoryRegionKind::Global)).unwrap();
        let result = layout.add_region(MemoryRegion::new("r2", 50, 100, MemoryRegionKind::Shared));
        assert!(result.is_err());
    }

    #[test]
    fn test_overflow_detector_safe_access() {
        let mut det = OverflowDetector::new();
        det.register_buffer("buf1", 1024);
        let access = BufferAccess::read("buf1", 0, 512);
        assert!(det.check_access(&access).is_none());
    }

    #[test]
    fn test_overflow_detector_overflow() {
        let mut det = OverflowDetector::new();
        det.register_buffer("buf1", 1024);
        let access = BufferAccess::write("buf1", 900, 200);
        let violation = det.check_access(&access);
        assert!(violation.is_some());
        let v = violation.unwrap();
        assert_eq!(v.overflow_bytes, 76);
    }

    #[test]
    fn test_overflow_detector_unknown_buffer() {
        let det = OverflowDetector::new();
        let access = BufferAccess::read("unknown", 0, 4);
        assert!(det.check_access(&access).is_some());
    }

    #[test]
    fn test_batch_overflow_check() {
        let mut det = OverflowDetector::new();
        det.register_buffer("buf1", 100);
        let accesses = vec![
            BufferAccess::read("buf1", 0, 50),
            BufferAccess::read("buf1", 50, 50),
            BufferAccess::write("buf1", 80, 30), // overflow
        ];
        let report = det.batch_check(&accesses);
        assert_eq!(report.total_accesses, 3);
        assert_eq!(report.safe_accesses, 2);
        assert_eq!(report.violations.len(), 1);
        assert!(!report.is_safe());
    }

    #[test]
    fn test_lifetime_tracker_basic() {
        let mut tracker = LifetimeTracker::new();
        tracker.allocate("a1", 0x1000, 256, 0);
        assert!(tracker.is_allocated("a1"));
        assert_eq!(tracker.get_state("a1"), Some(AllocationState::Allocated));

        tracker.free("a1").unwrap();
        assert!(!tracker.is_allocated("a1"));
        assert_eq!(tracker.get_state("a1"), Some(AllocationState::Freed));
    }

    #[test]
    fn test_lifetime_tracker_double_free() {
        let mut tracker = LifetimeTracker::new();
        tracker.allocate("a1", 0x1000, 256, 0);
        tracker.free("a1").unwrap();
        let result = tracker.free("a1");
        assert!(result.is_err());
    }

    #[test]
    fn test_use_after_free_detection() {
        let mut tracker = LifetimeTracker::new();
        tracker.allocate("a1", 0x1000, 256, 0);
        tracker.free("a1").unwrap();

        let mut detector = UseAfterFreeDetector::new(tracker);
        let v = detector.check_use("a1", 0, 4, 0, 10);
        assert!(v.is_some());
    }

    #[test]
    fn test_use_after_free_safe() {
        let mut tracker = LifetimeTracker::new();
        tracker.allocate("a1", 0x1000, 256, 0);

        let mut detector = UseAfterFreeDetector::new(tracker);
        let v = detector.check_use("a1", 0, 4, 0, 10);
        assert!(v.is_none());
    }

    #[test]
    fn test_dangling_pointer_check() {
        let mut tracker = LifetimeTracker::new();
        tracker.allocate("a1", 0x1000, 256, 0);
        tracker.free("a1").unwrap();

        let pointers = vec![("a1".to_string(), 0x1000u64)];
        let dangling = dangling_pointer_check(&tracker, &pointers);
        assert_eq!(dangling.len(), 1);
    }

    #[test]
    fn test_data_race_detection_basic() {
        let mut detector = DataRaceDetector::new();
        detector.add_access(MemoryAccess::new(0x100, 4, 0, true, 1));  // write
        detector.add_access(MemoryAccess::new(0x100, 4, 1, true, 2));  // write, different thread
        let races = detector.detect_races();
        assert_eq!(races.len(), 1);
    }

    #[test]
    fn test_data_race_read_read_no_race() {
        let mut detector = DataRaceDetector::new();
        detector.add_access(MemoryAccess::new(0x100, 4, 0, false, 1)); // read
        detector.add_access(MemoryAccess::new(0x100, 4, 1, false, 2)); // read
        let races = detector.detect_races();
        assert_eq!(races.len(), 0);
    }

    #[test]
    fn test_data_race_same_thread_no_race() {
        let mut detector = DataRaceDetector::new();
        detector.add_access(MemoryAccess::new(0x100, 4, 0, true, 1));
        detector.add_access(MemoryAccess::new(0x100, 4, 0, true, 2));
        let races = detector.detect_races();
        assert_eq!(races.len(), 0);
    }

    #[test]
    fn test_data_race_with_hb() {
        let mut detector = DataRaceDetector::new();
        detector.add_access(MemoryAccess::new(0x100, 4, 0, true, 1));
        detector.add_access(MemoryAccess::new(0x100, 4, 1, true, 2));
        detector.add_happens_before(1, 2);
        let races = detector.detect_races();
        assert_eq!(races.len(), 0);
    }

    #[test]
    fn test_data_race_non_overlapping_no_race() {
        let mut detector = DataRaceDetector::new();
        detector.add_access(MemoryAccess::new(0x100, 4, 0, true, 1));
        detector.add_access(MemoryAccess::new(0x200, 4, 1, true, 2));
        let races = detector.detect_races();
        assert_eq!(races.len(), 0);
    }

    #[test]
    fn test_happens_before_transitivity() {
        let mut graph = HappensBeforeGraph::new();
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        let closure = graph.transitive_closure();
        assert!(closure.is_before(1, 3));
        assert!(!closure.is_before(3, 1));
    }

    #[test]
    fn test_happens_before_topological_sort() {
        let mut graph = HappensBeforeGraph::new();
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(1, 3);
        let sorted = graph.topological_sort();
        assert!(sorted.is_some());
        let s = sorted.unwrap();
        assert_eq!(s[0], 1);
        assert_eq!(s[s.len() - 1], 3);
    }

    #[test]
    fn test_index_bounds_checker() {
        let mut checker = IndexBoundsChecker::new();
        checker.register_array(ArrayDescriptor::new("arr", vec![10, 20], 4));
        assert!(checker.check_index_bounds("arr", &[5, 15]).is_none());
        assert!(checker.check_index_bounds("arr", &[10, 15]).is_some());
        assert!(checker.check_index_bounds("arr", &[5, 20]).is_some());
    }

    #[test]
    fn test_index_bounds_dimension_mismatch() {
        let mut checker = IndexBoundsChecker::new();
        checker.register_array(ArrayDescriptor::new("arr", vec![10, 20], 4));
        let v = checker.check_index_bounds("arr", &[5]);
        assert!(v.is_some());
    }

    #[test]
    fn test_array_descriptor_linear_index() {
        let desc = ArrayDescriptor::new("arr", vec![3, 4], 4);
        assert_eq!(desc.linear_index(&[0, 0]), Some(0));
        assert_eq!(desc.linear_index(&[1, 0]), Some(4));
        assert_eq!(desc.linear_index(&[2, 3]), Some(11));
        assert_eq!(desc.linear_index(&[3, 0]), None);
    }

    #[test]
    fn test_leak_detector() {
        let mut detector = LeakDetector::new();
        detector.track_allocation("a1", 0x1000, 256, MemoryRegionKind::Global, 1);
        detector.track_allocation("a2", 0x2000, 512, MemoryRegionKind::Global, 2);
        detector.track_free("a1");

        let report = detector.detect_leaks();
        assert_eq!(report.leaked_allocations, 1);
        assert_eq!(report.total_leaked_bytes, 512);
        assert!(!report.is_clean());
    }

    #[test]
    fn test_leak_detector_no_leaks() {
        let mut detector = LeakDetector::new();
        detector.track_allocation("a1", 0x1000, 256, MemoryRegionKind::Global, 1);
        detector.track_free("a1");

        let report = detector.detect_leaks();
        assert!(report.is_clean());
    }

    #[test]
    fn test_leak_detector_reachable_not_leaked() {
        let mut detector = LeakDetector::new();
        detector.track_allocation("a1", 0x1000, 256, MemoryRegionKind::Global, 1);
        detector.mark_reachable("a1");

        let report = detector.detect_leaks();
        assert!(report.is_clean());
    }

    #[test]
    fn test_memory_safety_analyzer_combined() {
        let mut analyzer = MemorySafetyAnalyzer::new();
        analyzer.register_buffer("buf1", 100);
        analyzer.add_access(BufferAccess::write("buf1", 90, 20).with_thread(0).with_timestamp(1));
        let report = analyzer.analyze();
        assert!(!report.is_safe());
        assert!(report.overflow_report.as_ref().unwrap().violations.len() > 0);
    }

    #[test]
    fn test_memory_safety_analyzer_safe() {
        let mut analyzer = MemorySafetyAnalyzer::new();
        analyzer.register_buffer("buf1", 100);
        analyzer.add_access(BufferAccess::read("buf1", 0, 50).with_thread(0).with_timestamp(1));
        analyzer.free("buf1").ok();
        let report = analyzer.analyze();
        // Should have no overflow
        assert!(report.overflow_report.as_ref().unwrap().violations.is_empty());
    }

    #[test]
    fn test_barrier_convergence_check() {
        let ba = BarrierAnalysis::new(4);
        let barrier = BarrierInfo {
            id: 0,
            timestamp: 10,
            thread_ids: vec![0, 1, 2],
            memory_scope: "workgroup".to_string(),
        };
        let mut ba = ba;
        ba.add_barrier(barrier);
        let issues = ba.check_convergence();
        assert_eq!(issues.len(), 1);
    }

    #[test]
    fn test_barrier_convergence_all_threads() {
        let mut ba = BarrierAnalysis::new(3);
        ba.add_barrier(BarrierInfo {
            id: 0,
            timestamp: 10,
            thread_ids: vec![0, 1, 2],
            memory_scope: "workgroup".to_string(),
        });
        let issues = ba.check_convergence();
        assert!(issues.is_empty());
    }

    #[test]
    fn test_memory_layout_merge_adjacent() {
        let mut layout = MemoryLayout::new();
        layout.add_region(MemoryRegion::new("r1", 0, 100, MemoryRegionKind::Global)).unwrap();
        layout.add_region(MemoryRegion::new("r2", 100, 100, MemoryRegionKind::Global)).unwrap();
        layout.merge_adjacent();
        assert_eq!(layout.regions.len(), 1);
        assert_eq!(layout.regions[0].size, 200);
    }

    #[test]
    fn test_shared_memory_race_detector() {
        let mut det = SharedMemoryRaceDetector::new(2);
        det.add_access(MemoryAccess::new(0x100, 4, 0, true, 1));
        det.add_access(MemoryAccess::new(0x100, 4, 1, true, 2));
        let races = det.detect_races();
        assert_eq!(races.len(), 1);
    }

    #[test]
    fn test_overflow_report_rate() {
        let report = OverflowReport {
            violations: vec![],
            total_accesses: 100,
            safe_accesses: 100,
        };
        assert_eq!(report.violation_rate(), 0.0);
        assert!(report.is_safe());
    }

    #[test]
    fn test_memory_region_kind_display() {
        assert_eq!(format!("{}", MemoryRegionKind::Global), "global");
        assert_eq!(format!("{}", MemoryRegionKind::Shared), "shared");
        assert_eq!(format!("{}", MemoryRegionKind::Texture), "texture");
    }
    #[test]
    fn test_memorysanitizer_new() {
        let item = MemorySanitizer::new(Vec::new(), Vec::new(), 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_tainttracker_new() {
        let item = TaintTracker::new(Vec::new(), 0, Vec::new());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_pointerprovenance_new() {
        let item = PointerProvenance::new(0, "test".to_string(), false, 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_memoryaccesspattern_new() {
        let item = MemoryAccessPattern::new(Vec::new(), 0, false, 0.0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_gpuoccupancyanalysis_new() {
        let item = GpuOccupancyAnalysis::new(0, 0, 0.0, "test".to_string());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_bufferzoneverifier_new() {
        let item = BufferZoneVerifier::new(0, 0, Vec::new());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_memoryaccesslog_new() {
        let item = MemoryAccessLog::new(Vec::new(), 0, 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_coalescinganalysis_new() {
        let item = CoalescingAnalysis::new(Vec::new(), 0.0, 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_bankconflictdetector_new() {
        let item = BankConflictDetector::new(0, 0, 0, 0.0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_sharedmemorypadding_new() {
        let item = SharedMemoryPadding::new(0, 0, Vec::new(), 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_memoryfenceanalysis_new() {
        let item = MemoryFenceAnalysis::new(0, Vec::new(), 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_addresssanitizer_new() {
        let item = AddressSanitizer::new(0, 0, Vec::new());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_memsafe_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let m = memsafe_mean(&data);
        assert!((m - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_variance() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let v = memsafe_variance(&data);
        assert!(v > 0.0);
    }

    #[test]
    fn test_memsafe_median() {
        let data = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let m = memsafe_median(&data);
        assert!((m - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_entropy() {
        let data = vec![1.0, 1.0, 1.0, 1.0];
        let e = memsafe_entropy(&data);
        assert!(e > 0.0);
    }

    #[test]
    fn test_memsafe_std_dev() {
        let data = vec![10.0, 10.0, 10.0];
        let s = memsafe_std_dev(&data);
        assert!(s.abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_analysis() {
        let mut a = MemsafeAnalysis::new(3);
        a.set(0, 1, 0.5);
        a.set(1, 2, 0.3);
        assert!((a.get(0, 1) - 0.5).abs() < 1e-10);
        assert_eq!(a.size, 3);
    }

    #[test]
    fn test_memsafe_iterator() {
        let iter = MemsafeResultIterator::new(vec![(0, 1.0), (1, 2.0), (2, 3.0)]);
        let items: Vec<_> = iter.collect();
        assert_eq!(items.len(), 3);
    }

    #[test]
    fn test_memsafe_batch_processor() {
        let mut proc = MemsafeBatchProcessor::new(2);
        proc.process_batch(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(proc.processed, 4);
        assert_eq!(proc.results.len(), 2);
    }

    #[test]
    fn test_memsafe_histogram() {
        let hist = MemsafeHistogram::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0], 3);
        assert_eq!(hist.num_bins(), 3);
        assert_eq!(hist.total_count, 5);
    }

    #[test]
    fn test_memsafe_graph() {
        let mut g = MemsafeGraph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 2.0);
        g.add_edge(2, 3, 3.0);
        assert_eq!(g.edge_count, 3);
        assert!(g.has_edge(0, 1));
        assert!(!g.has_edge(3, 0));
        assert!(g.is_acyclic());
    }

    #[test]
    fn test_memsafe_graph_shortest_path() {
        let mut g = MemsafeGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 2.0);
        g.add_edge(0, 2, 10.0);
        let dist = g.shortest_path_dijkstra(0);
        assert!((dist[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_graph_topo_sort() {
        let mut g = MemsafeGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        let topo = g.topological_sort();
        assert!(topo.is_some());
    }

    #[test]
    fn test_memsafe_graph_components() {
        let mut g = MemsafeGraph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(2, 3, 1.0);
        let comps = g.connected_components();
        assert_eq!(comps.len(), 2);
    }

    #[test]
    fn test_memsafe_cache() {
        let mut cache = MemsafeCache::new(10);
        cache.insert(42, vec![1.0, 2.0]);
        assert!(cache.get(42).is_some());
        assert!(cache.get(99).is_none());
    }

    #[test]
    fn test_memsafe_config() {
        let config = MemsafeConfig::default_config().with_verbose(true).with_max_iterations(500);
        assert!(config.verbose);
        assert_eq!(config.max_iterations, 500);
    }

    #[test]
    fn test_memsafe_report() {
        let mut report = MemsafeReport::new("Test Report");
        report.add_metric("accuracy", 0.95);
        report.add_warning("low sample size");
        assert_eq!(report.total_metrics(), 1);
        assert!(report.has_warnings());
        let text = report.render_text();
        assert!(text.contains("Test Report"));
    }

    #[test]
    fn test_memsafe_kmeans() {
        let data = vec![vec![0.0, 0.0], vec![0.1, 0.1], vec![10.0, 10.0], vec![10.1, 10.1]];
        let assignments = memsafe_kmeans(&data, 2, 100);
        assert_eq!(assignments.len(), 4);
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[2], assignments[3]);
    }

    #[test]
    fn test_memsafe_pairwise_distances() {
        let points = vec![vec![0.0, 0.0], vec![3.0, 4.0]];
        let dists = memsafe_pairwise_distances(&points);
        assert!((dists[0][1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_harmmean() {
        let data = vec![1.0, 2.0, 4.0];
        let hm = memsafe_harmmean(&data);
        assert!(hm > 0.0 && hm < 4.0);
    }

    #[test]
    fn test_memsafe_geomean() {
        let data = vec![1.0, 2.0, 4.0];
        let gm = memsafe_geomean(&data);
        assert!(gm > 0.0 && gm < 4.0);
    }

    #[test]
    fn test_memsafe_sample_skewness() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = memsafe_sample_skewness(&data);
        assert!(s.abs() < 1.0);
    }

    #[test]
    fn test_memsafe_excess_kurtosis() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let k = memsafe_excess_kurtosis(&data);
        let _ = k; // Just verify it computes
    }

    #[test]
    fn test_memsafe_gini() {
        let data = vec![1.0, 1.0, 1.0, 1.0];
        let g = memsafe_gini(&data);
        assert!(g.abs() < 0.01);
    }

    #[test]
    fn test_memsafe_percentile_at() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let p = memsafe_percentile_at(&data, 0.9);
        assert!(p >= 9.0);
    }

    #[test]
    fn test_memsafe_pca_2d() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let proj = memsafe_pca_2d(&data);
        assert_eq!(proj.len(), 2);
    }

    #[test]
    fn test_memsafe_analysis_normalize() {
        let mut a = MemsafeAnalysis::new(2);
        a.set(0, 0, 1.0); a.set(0, 1, 3.0);
        a.normalize();
        assert!((a.total_sum() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_analysis_transpose() {
        let mut a = MemsafeAnalysis::new(2);
        a.set(0, 1, 5.0);
        let t = a.transpose();
        assert!((t.get(1, 0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_analysis_multiply() {
        let mut a = MemsafeAnalysis::new(2);
        a.set(0, 0, 1.0); a.set(1, 1, 1.0);
        let mut b = MemsafeAnalysis::new(2);
        b.set(0, 1, 2.0); b.set(1, 0, 3.0);
        let c = a.multiply(&b);
        assert!((c.get(0, 1) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_analysis_frobenius() {
        let mut a = MemsafeAnalysis::new(2);
        a.set(0, 0, 3.0); a.set(1, 1, 4.0);
        assert!((a.frobenius_norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_analysis_symmetric() {
        let mut a = MemsafeAnalysis::new(2);
        a.set(0, 1, 1.0); a.set(1, 0, 1.0);
        assert!(a.is_symmetric());
    }

    #[test]
    fn test_memsafe_graph_dot() {
        let mut g = MemsafeGraph::new(2);
        g.add_edge(0, 1, 1.0);
        let dot = g.to_dot();
        assert!(dot.contains("digraph"));
    }

    #[test]
    fn test_memsafe_histogram_render() {
        let hist = MemsafeHistogram::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 5);
        let ascii = hist.render_ascii(20);
        assert!(!ascii.is_empty());
    }

    #[test]
    fn test_memsafe_batch_reset() {
        let mut proc = MemsafeBatchProcessor::new(3);
        proc.process_batch(&[1.0, 2.0, 3.0]);
        assert!(proc.processed > 0);
        proc.reset();
        assert_eq!(proc.processed, 0);
    }

    #[test]
    fn test_memsafe_graph_remove_edge() {
        let mut g = MemsafeGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        assert_eq!(g.edge_count, 2);
        g.remove_edge(0, 1);
        assert_eq!(g.edge_count, 1);
        assert!(!g.has_edge(0, 1));
    }

    #[test]
    fn test_memsafe_dense_matrix_new() {
        let m = MemSafeDenseMatrix::new(3, 3);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 3);
    }

    #[test]
    fn test_memsafe_dense_matrix_identity() {
        let m = MemSafeDenseMatrix::identity(3);
        assert!((m.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((m.get(0, 1)).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_dense_matrix_mul() {
        let a = MemSafeDenseMatrix::identity(2);
        let b = MemSafeDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let c = a.mul_matrix(&b);
        assert!((c.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((c.get(1, 1) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_dense_matrix_transpose() {
        let a = MemSafeDenseMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = a.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert!((t.get(0, 1) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_dense_matrix_det_2x2() {
        let m = MemSafeDenseMatrix::from_vec(2, 2, vec![3.0, 7.0, 1.0, -4.0]);
        let det = m.determinant_2x2();
        assert!((det - (-19.0)).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_dense_matrix_det_3x3() {
        let m = MemSafeDenseMatrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0]);
        let det = m.determinant_3x3();
        assert!((det - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_dense_matrix_inverse_2x2() {
        let m = MemSafeDenseMatrix::from_vec(2, 2, vec![4.0, 7.0, 2.0, 6.0]);
        let inv = m.inverse_2x2().unwrap();
        let prod = m.mul_matrix(&inv);
        assert!((prod.get(0, 0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_dense_matrix_power() {
        let m = MemSafeDenseMatrix::identity(3);
        let p = m.power(5);
        assert!((p.get(0, 0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_dense_matrix_rank() {
        let m = MemSafeDenseMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0]);
        assert_eq!(m.rank(), 1);
    }

    #[test]
    fn test_memsafe_dense_matrix_solve() {
        let a = MemSafeDenseMatrix::from_vec(2, 2, vec![2.0, 1.0, 5.0, 3.0]);
        let x = a.solve(&[4.0, 7.0]).unwrap();
        assert!((x[0] - 5.0).abs() < 1e-8);
        assert!((x[1] - (-6.0)).abs() < 1e-8);
    }

    #[test]
    fn test_memsafe_dense_matrix_lu() {
        let a = MemSafeDenseMatrix::from_vec(2, 2, vec![4.0, 3.0, 6.0, 3.0]);
        let (l, u) = a.lu_decomposition();
        let prod = l.mul_matrix(&u);
        assert!((prod.get(0, 0) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_dense_matrix_eigenvalues() {
        let m = MemSafeDenseMatrix::from_vec(2, 2, vec![2.0, 1.0, 1.0, 2.0]);
        let (e1, e2) = m.eigenvalues_2x2();
        assert!((e1 - 3.0).abs() < 1e-10);
        assert!((e2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_dense_matrix_kronecker() {
        let a = MemSafeDenseMatrix::identity(2);
        let b = MemSafeDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let k = a.kronecker_product(&b);
        assert_eq!(k.rows, 4);
        assert_eq!(k.cols, 4);
    }

    #[test]
    fn test_memsafe_dense_matrix_hadamard() {
        let a = MemSafeDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = MemSafeDenseMatrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let h = a.hadamard_product(&b);
        assert!((h.get(0, 0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_interval() {
        let a = MemSafeInterval::new(1.0, 3.0);
        let b = MemSafeInterval::new(2.0, 5.0);
        assert!(a.overlaps(&b));
        assert!(a.contains(2.0));
        assert!(!a.contains(4.0));
        let sum = a.add(&b);
        assert!((sum.lo - 3.0).abs() < 1e-10);
        assert!((sum.hi - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_interval_mul() {
        let a = MemSafeInterval::new(-2.0, 3.0);
        let b = MemSafeInterval::new(1.0, 4.0);
        let prod = a.mul(&b);
        assert!((prod.lo - (-8.0)).abs() < 1e-10);
        assert!((prod.hi - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_interval_hull() {
        let a = MemSafeInterval::new(1.0, 3.0);
        let b = MemSafeInterval::new(5.0, 7.0);
        let h = a.hull(&b);
        assert!((h.lo - 1.0).abs() < 1e-10);
        assert!((h.hi - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_state_machine() {
        let mut sm = MemSafeStateMachine::new();
        assert_eq!(*sm.state(), MemSafeState::Clean);
        assert!(sm.transition(MemSafeState::Scanning));
        assert_eq!(*sm.state(), MemSafeState::Scanning);
        assert_eq!(sm.transition_count, 1);
    }

    #[test]
    fn test_memsafe_state_machine_invalid() {
        let mut sm = MemSafeStateMachine::new();
        let last_state = MemSafeState::Compromised;
        assert!(!sm.can_transition(&last_state));
    }

    #[test]
    fn test_memsafe_state_machine_reset() {
        let mut sm = MemSafeStateMachine::new();
        sm.transition(MemSafeState::Scanning);
        sm.reset();
        assert_eq!(*sm.state(), MemSafeState::Clean);
        assert_eq!(sm.history_len(), 0);
    }

    #[test]
    fn test_memsafe_ring_buffer() {
        let mut rb = MemSafeRingBuffer::new(3);
        rb.push(1.0); rb.push(2.0); rb.push(3.0);
        assert!(rb.is_full());
        assert!((rb.average() - 2.0).abs() < 1e-10);
        rb.push(4.0);
        assert!((rb.oldest().unwrap() - 2.0).abs() < 1e-10);
        assert!((rb.latest().unwrap() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_ring_buffer_to_vec() {
        let mut rb = MemSafeRingBuffer::new(5);
        rb.push(10.0); rb.push(20.0); rb.push(30.0);
        let v = rb.to_vec();
        assert_eq!(v.len(), 3);
        assert!((v[0] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_disjoint_set() {
        let mut ds = MemSafeDisjointSet::new(5);
        assert_eq!(ds.num_components(), 5);
        ds.union(0, 1);
        ds.union(2, 3);
        assert_eq!(ds.num_components(), 3);
        assert!(ds.connected(0, 1));
        assert!(!ds.connected(0, 2));
    }

    #[test]
    fn test_memsafe_disjoint_set_components() {
        let mut ds = MemSafeDisjointSet::new(4);
        ds.union(0, 1); ds.union(2, 3);
        let comps = ds.components();
        assert_eq!(comps.len(), 2);
    }

    #[test]
    fn test_memsafe_sorted_list() {
        let mut sl = MemSafeSortedList::new();
        sl.insert(3.0); sl.insert(1.0); sl.insert(2.0);
        assert_eq!(sl.len(), 3);
        assert!((sl.min().unwrap() - 1.0).abs() < 1e-10);
        assert!((sl.max().unwrap() - 3.0).abs() < 1e-10);
        assert!((sl.median() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_sorted_list_remove() {
        let mut sl = MemSafeSortedList::new();
        sl.insert(1.0); sl.insert(2.0); sl.insert(3.0);
        assert!(sl.remove(2.0));
        assert_eq!(sl.len(), 2);
        assert!(!sl.contains(2.0));
    }

    #[test]
    fn test_memsafe_ema() {
        let mut ema = MemSafeEma::new(0.5);
        ema.update(10.0);
        assert!((ema.current() - 10.0).abs() < 1e-10);
        ema.update(20.0);
        assert!((ema.current() - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_bloom_filter() {
        let mut bf = MemSafeBloomFilter::new(1000, 3);
        bf.insert(42);
        bf.insert(100);
        assert!(bf.may_contain(42));
        assert!(bf.may_contain(100));
        assert_eq!(bf.count(), 2);
    }

    #[test]
    fn test_memsafe_trie() {
        let mut trie = MemSafeTrie::new();
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
    fn test_memsafe_dense_matrix_sym() {
        let m = MemSafeDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 2.0, 3.0]);
        assert!(m.is_symmetric());
    }

    #[test]
    fn test_memsafe_dense_matrix_diag() {
        let m = MemSafeDenseMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 3.0]);
        assert!(m.is_diagonal());
    }

    #[test]
    fn test_memsafe_dense_matrix_upper_tri() {
        let m = MemSafeDenseMatrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0]);
        assert!(m.is_upper_triangular());
    }

    #[test]
    fn test_memsafe_dense_matrix_outer() {
        let m = MemSafeDenseMatrix::outer_product(&[1.0, 2.0], &[3.0, 4.0]);
        assert!((m.get(0, 0) - 3.0).abs() < 1e-10);
        assert!((m.get(1, 1) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_dense_matrix_submatrix() {
        let m = MemSafeDenseMatrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let sub = m.submatrix(0, 0, 2, 2);
        assert_eq!(sub.rows, 2);
        assert!((sub.get(1, 1) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_priority_queue() {
        let mut pq = MemSafePriorityQueue::new();
        pq.push(3.0, 1); pq.push(1.0, 2); pq.push(2.0, 3);
        assert_eq!(pq.pop().unwrap().1, 2);
        assert_eq!(pq.pop().unwrap().1, 3);
        assert_eq!(pq.pop().unwrap().1, 1);
    }

    #[test]
    fn test_memsafe_accumulator() {
        let mut acc = MemSafeAccumulator::new();
        for i in 1..=10 { acc.add(i as f64); }
        assert!((acc.mean() - 5.5).abs() < 1e-10);
        assert_eq!(acc.count(), 10);
        assert!((acc.min() - 1.0).abs() < 1e-10);
        assert!((acc.max() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_accumulator_merge() {
        let mut a = MemSafeAccumulator::new();
        a.add(1.0); a.add(2.0);
        let mut b = MemSafeAccumulator::new();
        b.add(3.0); b.add(4.0);
        a.merge(&b);
        assert_eq!(a.count(), 4);
        assert!((a.mean() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_sparse_matrix() {
        let mut m = MemSafeSparseMatrix::new(3, 3);
        m.insert(0, 1, 2.0); m.insert(1, 2, 3.0);
        assert_eq!(m.nnz(), 2);
        assert!((m.get(0, 1) - 2.0).abs() < 1e-10);
        assert!((m.get(0, 0)).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_sparse_mul_vec() {
        let mut m = MemSafeSparseMatrix::new(2, 2);
        m.insert(0, 0, 1.0); m.insert(1, 1, 2.0);
        let result = m.mul_vec(&[3.0, 4.0]);
        assert!((result[0] - 3.0).abs() < 1e-10);
        assert!((result[1] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_sparse_transpose() {
        let mut m = MemSafeSparseMatrix::new(2, 3);
        m.insert(0, 2, 5.0);
        let t = m.transpose();
        assert_eq!(t.rows, 3); assert_eq!(t.cols, 2);
        assert!((t.get(2, 0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_polynomial_eval() {
        let p = MemSafePolynomial::new(vec![1.0, 2.0, 3.0]);
        assert!((p.evaluate(2.0) - 17.0).abs() < 1e-10);
        assert!((p.evaluate_horner(2.0) - 17.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_polynomial_add() {
        let a = MemSafePolynomial::new(vec![1.0, 2.0]);
        let b = MemSafePolynomial::new(vec![3.0, 4.0]);
        let c = a.add(&b);
        assert!((c.evaluate(1.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_polynomial_mul() {
        let a = MemSafePolynomial::new(vec![1.0, 1.0]);
        let b = MemSafePolynomial::new(vec![1.0, 1.0]);
        let c = a.mul(&b);
        assert_eq!(c.degree(), 2);
        assert!((c.evaluate(2.0) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_polynomial_deriv() {
        let p = MemSafePolynomial::new(vec![1.0, 2.0, 3.0]);
        let dp = p.derivative();
        assert!((dp.evaluate(0.0) - 2.0).abs() < 1e-10);
        assert!((dp.evaluate(1.0) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_polynomial_integral() {
        let p = MemSafePolynomial::new(vec![2.0, 3.0]);
        let ip = p.integral(0.0);
        assert!((ip.evaluate(1.0) - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_polynomial_roots() {
        let p = MemSafePolynomial::new(vec![-6.0, 1.0, 1.0]);
        let roots = p.roots_quadratic();
        assert_eq!(roots.len(), 2);
    }

    #[test]
    fn test_memsafe_polynomial_newton() {
        let p = MemSafePolynomial::new(vec![-2.0, 0.0, 1.0]);
        let root = p.newton_root(1.0, 100, 1e-10);
        assert!(root.is_some());
        assert!((root.unwrap() - std::f64::consts::SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn test_memsafe_polynomial_compose() {
        let p = MemSafePolynomial::new(vec![0.0, 0.0, 1.0]);
        let q = MemSafePolynomial::new(vec![1.0, 1.0]);
        let r = p.compose(&q);
        assert!((r.evaluate(2.0) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_rng() {
        let mut rng = MemSafeRng::new(42);
        let v1 = rng.next_u64();
        let v2 = rng.next_u64();
        assert_ne!(v1, v2);
        let f = rng.next_f64();
        assert!(f >= 0.0 && f < 1.0);
    }

    #[test]
    fn test_memsafe_rng_gaussian() {
        let mut rng = MemSafeRng::new(123);
        let mut sum = 0.0;
        for _ in 0..1000 { sum += rng.next_gaussian(); }
        let mean = sum / 1000.0;
        assert!(mean.abs() < 0.2);
    }

    #[test]
    fn test_memsafe_timer() {
        let mut timer = MemSafeTimer::new("test");
        timer.record(100); timer.record(200); timer.record(300);
        assert_eq!(timer.count(), 3);
        assert_eq!(timer.total_ns(), 600);
        assert!((timer.average_ns() - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_bitvector() {
        let mut bv = MemSafeBitVector::new(100);
        bv.set(5); bv.set(42); bv.set(99);
        assert!(bv.get(5));
        assert!(bv.get(42));
        assert!(!bv.get(50));
        assert_eq!(bv.count_ones(), 3);
    }

    #[test]
    fn test_memsafe_bitvector_ops() {
        let mut a = MemSafeBitVector::new(64);
        a.set(0); a.set(10); a.set(20);
        let mut b = MemSafeBitVector::new(64);
        b.set(10); b.set(20); b.set(30);
        let c = a.and(&b);
        assert_eq!(c.count_ones(), 2);
        let d = a.or(&b);
        assert_eq!(d.count_ones(), 4);
        assert_eq!(a.hamming_distance(&b), 2);
    }

    #[test]
    fn test_memsafe_bitvector_jaccard() {
        let mut a = MemSafeBitVector::new(10);
        a.set(0); a.set(1); a.set(2);
        let mut b = MemSafeBitVector::new(10);
        b.set(1); b.set(2); b.set(3);
        let j = a.jaccard(&b);
        assert!((j - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_priority_queue_empty() {
        let mut pq = MemSafePriorityQueue::new();
        assert!(pq.is_empty());
        assert!(pq.pop().is_none());
    }

    #[test]
    fn test_memsafe_sparse_add() {
        let mut a = MemSafeSparseMatrix::new(2, 2);
        a.insert(0, 0, 1.0);
        let mut b = MemSafeSparseMatrix::new(2, 2);
        b.insert(0, 0, 2.0); b.insert(1, 1, 3.0);
        let c = a.add(&b);
        assert!((c.get(0, 0) - 3.0).abs() < 1e-10);
        assert!((c.get(1, 1) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_rng_shuffle() {
        let mut rng = MemSafeRng::new(99);
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        rng.shuffle(&mut data);
        assert_eq!(data.len(), 5);
        let sum: f64 = data.iter().sum();
        assert!((sum - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_polynomial_display() {
        let p = MemSafePolynomial::new(vec![1.0, 2.0, 3.0]);
        let s = format!("{}", p);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_memsafe_polynomial_monomial() {
        let m = MemSafePolynomial::monomial(3, 5.0);
        assert_eq!(m.degree(), 3);
        assert!((m.evaluate(2.0) - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_timer_percentiles() {
        let mut timer = MemSafeTimer::new("perf");
        for i in 1..=100 { timer.record(i); }
        assert_eq!(timer.p50_ns(), 50);
        assert!(timer.p95_ns() >= 90);
    }

    #[test]
    fn test_memsafe_accumulator_cv() {
        let mut acc = MemSafeAccumulator::new();
        acc.add(10.0); acc.add(10.0); acc.add(10.0);
        assert!(acc.coefficient_of_variation().abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_sparse_diagonal() {
        let mut m = MemSafeSparseMatrix::new(3, 3);
        m.insert(0, 0, 1.0); m.insert(1, 1, 2.0); m.insert(2, 2, 3.0);
        assert!((m.trace() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_lru_cache() {
        let mut cache = MemSafeLruCache::new(2);
        cache.put(1, vec![1.0]);
        cache.put(2, vec![2.0]);
        assert!(cache.get(1).is_some());
        cache.put(3, vec![3.0]);
        assert!(cache.get(2).is_none());
        assert!(cache.get(3).is_some());
    }

    #[test]
    fn test_memsafe_lru_hit_rate() {
        let mut cache = MemSafeLruCache::new(10);
        cache.put(1, vec![1.0]);
        cache.get(1);
        cache.get(2);
        assert!((cache.hit_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_graph_coloring() {
        let mut gc = MemSafeGraphColoring::new(4);
        gc.add_edge(0, 1); gc.add_edge(1, 2); gc.add_edge(2, 3); gc.add_edge(3, 0);
        let colors = gc.greedy_color();
        assert!(gc.is_valid_coloring());
        assert!(colors <= 3);
    }

    #[test]
    fn test_memsafe_graph_coloring_complete() {
        let mut gc = MemSafeGraphColoring::new(3);
        gc.add_edge(0, 1); gc.add_edge(1, 2); gc.add_edge(0, 2);
        let colors = gc.greedy_color();
        assert_eq!(colors, 3);
        assert!(gc.is_valid_coloring());
    }

    #[test]
    fn test_memsafe_topk() {
        let mut tk = MemSafeTopK::new(3);
        tk.insert(5.0, "e"); tk.insert(3.0, "c"); tk.insert(1.0, "a");
        tk.insert(4.0, "d"); tk.insert(2.0, "b");
        assert_eq!(tk.len(), 3);
        assert!((tk.max_score().unwrap() - 5.0).abs() < 1e-10);
        assert!((tk.min_score().unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_sliding_window() {
        let mut sw = MemSafeSlidingWindow::new(3);
        sw.push(1.0); sw.push(2.0); sw.push(3.0);
        assert!((sw.mean() - 2.0).abs() < 1e-10);
        sw.push(4.0);
        assert!((sw.mean() - 3.0).abs() < 1e-10);
        assert_eq!(sw.len(), 3);
    }

    #[test]
    fn test_memsafe_sliding_window_trend() {
        let mut sw = MemSafeSlidingWindow::new(10);
        for i in 0..5 { sw.push(i as f64); }
        assert!(sw.trend() > 0.0);
    }

    #[test]
    fn test_memsafe_confusion_matrix() {
        let actual = vec![true, true, false, false, true];
        let predicted = vec![true, false, false, true, true];
        let cm = MemSafeConfusionMatrix::from_predictions(&actual, &predicted);
        assert_eq!(cm.true_positive, 2);
        assert_eq!(cm.false_positive, 1);
        assert_eq!(cm.true_negative, 1);
        assert_eq!(cm.false_negative, 1);
        assert_eq!(cm.total(), 5);
    }

    #[test]
    fn test_memsafe_confusion_f1() {
        let cm = MemSafeConfusionMatrix { true_positive: 80, false_positive: 20, true_negative: 70, false_negative: 30 };
        assert!((cm.precision() - 0.8).abs() < 1e-10);
        let f1 = cm.f1_score();
        assert!(f1 > 0.0 && f1 < 1.0);
    }

    #[test]
    fn test_memsafe_cosine_similarity() {
        let s = memsafe_cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(s.abs() < 1e-10);
        let s2 = memsafe_cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]);
        assert!((s2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_euclidean_distance() {
        let d = memsafe_euclidean_distance(&[0.0, 0.0], &[3.0, 4.0]);
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_sigmoid() {
        let s = memsafe_sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-10);
        let s2 = memsafe_sigmoid(100.0);
        assert!((s2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_memsafe_softmax() {
        let sm = memsafe_softmax(&[1.0, 2.0, 3.0]);
        let sum: f64 = sm.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(sm[2] > sm[1] && sm[1] > sm[0]);
    }

    #[test]
    fn test_memsafe_kl_divergence() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let kl = memsafe_kl_divergence(&p, &q);
        assert!(kl.abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_normalize() {
        let v = memsafe_normalize(&[3.0, 4.0]);
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_lerp() {
        assert!((memsafe_lerp(0.0, 10.0, 0.5) - 5.0).abs() < 1e-10);
        assert!((memsafe_lerp(0.0, 10.0, 0.0) - 0.0).abs() < 1e-10);
        assert!((memsafe_lerp(0.0, 10.0, 1.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_clamp() {
        assert!((memsafe_clamp(5.0, 0.0, 10.0) - 5.0).abs() < 1e-10);
        assert!((memsafe_clamp(-5.0, 0.0, 10.0) - 0.0).abs() < 1e-10);
        assert!((memsafe_clamp(15.0, 0.0, 10.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_cross_product() {
        let c = memsafe_cross_product(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]);
        assert!((c[0]).abs() < 1e-10);
        assert!((c[1]).abs() < 1e-10);
        assert!((c[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_dot_product() {
        let d = memsafe_dot_product(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        assert!((d - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_js_divergence() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let js = memsafe_js_divergence(&p, &q);
        assert!(js.abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_hellinger() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let h = memsafe_hellinger_distance(&p, &q);
        assert!(h.abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_logsumexp() {
        let lse = memsafe_logsumexp(&[1.0, 2.0, 3.0]);
        assert!(lse > 3.0);
    }

    #[test]
    fn test_memsafe_feature_scaler() {
        let mut scaler = MemSafeFeatureScaler::new();
        let data = vec![vec![1.0, 10.0], vec![2.0, 20.0], vec![3.0, 30.0]];
        scaler.fit(&data);
        let normalized = scaler.normalize(&[2.0, 20.0]);
        assert!((normalized[0] - 0.5).abs() < 1e-10);
        assert!((normalized[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_feature_scaler_inverse() {
        let mut scaler = MemSafeFeatureScaler::new();
        let data = vec![vec![0.0, 0.0], vec![10.0, 100.0]];
        scaler.fit(&data);
        let normed = scaler.normalize(&[5.0, 50.0]);
        let inv = scaler.inverse_normalize(&normed);
        assert!((inv[0] - 5.0).abs() < 1e-10);
        assert!((inv[1] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_linear_regression() {
        let mut lr = MemSafeLinearRegression::new();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        lr.fit(&x, &y);
        assert!((lr.slope - 2.0).abs() < 1e-10);
        assert!(lr.intercept.abs() < 1e-10);
        assert!((lr.r_squared - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_linear_regression_predict() {
        let mut lr = MemSafeLinearRegression::new();
        lr.fit(&[0.0, 1.0, 2.0], &[1.0, 3.0, 5.0]);
        assert!((lr.predict(3.0) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_weighted_graph() {
        let mut g = MemSafeWeightedGraph::new(4);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 2.0); g.add_edge(2, 3, 3.0);
        assert_eq!(g.num_edges, 3);
        let dists = g.dijkstra(0);
        assert!((dists[3] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_weighted_graph_mst() {
        let mut g = MemSafeWeightedGraph::new(4);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 2.0); g.add_edge(2, 3, 3.0);
        g.add_edge(0, 3, 10.0);
        let mst = g.min_spanning_tree_weight();
        assert!((mst - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_moving_average() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ma = memsafe_moving_average(&data, 3);
        assert_eq!(ma.len(), 3);
        assert!((ma[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_cumsum() {
        let cs = memsafe_cumsum(&[1.0, 2.0, 3.0, 4.0]);
        assert!((cs[3] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_diff() {
        let d = memsafe_diff(&[1.0, 3.0, 6.0, 10.0]);
        assert_eq!(d.len(), 3);
        assert!((d[0] - 2.0).abs() < 1e-10);
        assert!((d[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_autocorrelation() {
        let ac = memsafe_autocorrelation(&[1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0], 0);
        assert!((ac - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_dft_magnitude() {
        let mags = memsafe_dft_magnitude(&[1.0, 0.0, -1.0, 0.0]);
        assert!(!mags.is_empty());
    }

    #[test]
    fn test_memsafe_integrate_trapezoid() {
        let area = memsafe_integrate_trapezoid(&[0.0, 1.0, 2.0], &[0.0, 1.0, 0.0]);
        assert!((area - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_convolve() {
        let c = memsafe_convolve(&[1.0, 2.0, 3.0], &[1.0, 1.0]);
        assert_eq!(c.len(), 4);
        assert!((c[0] - 1.0).abs() < 1e-10);
        assert!((c[3] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_weighted_graph_clustering() {
        let mut g = MemSafeWeightedGraph::new(4);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 1.0); g.add_edge(0, 2, 1.0);
        let cc = g.clustering_coefficient(0);
        assert!((cc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_histogram_cumulative() {
        let h = MemSafeHistogramExt::from_data(&[1.0, 1.0, 2.0, 3.0, 3.0, 3.0], 3);
        let cum = h.cumulative();
        assert_eq!(*cum.last().unwrap(), 6);
    }

    #[test]
    fn test_memsafe_histogram_entropy() {
        let h = MemSafeHistogramExt::from_data(&[1.0, 2.0, 3.0, 4.0], 4);
        let ent = h.entropy();
        assert!(ent > 0.0);
    }

    #[test]
    fn test_memsafe_aabb() {
        let bb = MemSafeAABB::new(0.0, 0.0, 10.0, 10.0);
        assert!(bb.contains(5.0, 5.0));
        assert!(!bb.contains(11.0, 5.0));
        assert!((bb.area() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_aabb_intersects() {
        let a = MemSafeAABB::new(0.0, 0.0, 10.0, 10.0);
        let b = MemSafeAABB::new(5.0, 5.0, 15.0, 15.0);
        let c = MemSafeAABB::new(20.0, 20.0, 30.0, 30.0);
        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn test_memsafe_quadtree() {
        let bb = MemSafeAABB::new(0.0, 0.0, 100.0, 100.0);
        let mut qt = MemSafeQuadTree::new(bb, 4, 8);
        for i in 0..20 {
            qt.insert(MemSafePoint2D { x: i as f64 * 5.0, y: i as f64 * 5.0, data: i as f64 });
        }
        assert_eq!(qt.count(), 20);
    }

    #[test]
    fn test_memsafe_quadtree_query() {
        let bb = MemSafeAABB::new(0.0, 0.0, 100.0, 100.0);
        let mut qt = MemSafeQuadTree::new(bb, 2, 8);
        qt.insert(MemSafePoint2D { x: 10.0, y: 10.0, data: 1.0 });
        qt.insert(MemSafePoint2D { x: 90.0, y: 90.0, data: 2.0 });
        let range = MemSafeAABB::new(0.0, 0.0, 50.0, 50.0);
        let found = qt.query_range(&range);
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn test_memsafe_mat_mul() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let c = memsafe_mat_mul(&a, &b);
        assert!((c[0][0] - 19.0).abs() < 1e-10);
        assert!((c[1][1] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_transpose() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let t = memsafe_transpose(&a);
        assert_eq!(t.len(), 3);
        assert_eq!(t[0].len(), 2);
        assert!((t[2][1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_frobenius_norm() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let n = memsafe_frobenius_norm(&a);
        assert!((n - 2.0f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_trace() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert!((memsafe_trace(&a) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_identity() {
        let id = memsafe_identity(3);
        assert!((id[0][0] - 1.0).abs() < 1e-10);
        assert!((id[0][1]).abs() < 1e-10);
        assert!((id[2][2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_power_iteration() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let (eval, _evec) = memsafe_power_iteration(&a, 100);
        assert!((eval - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_memsafe_running_stats() {
        let mut s = MemSafeRunningStats::new();
        for &v in &[1.0, 2.0, 3.0, 4.0, 5.0] { s.push(v); }
        assert!((s.mean - 3.0).abs() < 1e-10);
        assert!((s.min_val - 1.0).abs() < 1e-10);
        assert!((s.max_val - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_running_stats_merge() {
        let mut a = MemSafeRunningStats::new();
        let mut b = MemSafeRunningStats::new();
        for &v in &[1.0, 2.0, 3.0] { a.push(v); }
        for &v in &[4.0, 5.0, 6.0] { b.push(v); }
        a.merge(&b);
        assert_eq!(a.count, 6);
        assert!((a.mean - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_running_stats_cv() {
        let mut s = MemSafeRunningStats::new();
        for &v in &[10.0, 10.0, 10.0] { s.push(v); }
        assert!(s.coefficient_of_variation() < 1e-10);
    }

    #[test]
    fn test_memsafe_iqr() {
        let iqr = memsafe_iqr(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert!(iqr > 0.0);
    }

    #[test]
    fn test_memsafe_outliers() {
        let outliers = memsafe_outliers(&[1.0, 2.0, 3.0, 4.0, 5.0, 100.0]);
        assert!(!outliers.is_empty());
    }

    #[test]
    fn test_memsafe_zscore() {
        let z = memsafe_zscore(&[10.0, 20.0, 30.0]);
        assert!((z[1]).abs() < 1e-10); // middle value should be ~0
    }

    #[test]
    fn test_memsafe_rank() {
        let r = memsafe_rank(&[30.0, 10.0, 20.0]);
        assert!((r[0] - 3.0).abs() < 1e-10);
        assert!((r[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_spearman() {
        let rho = memsafe_spearman(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((rho - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_sample_skewness_symmetric() {
        let s = memsafe_sample_skewness(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(s.abs() < 1e-10);
    }

    #[test]
    fn test_memsafe_covariance_matrix() {
        let data = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]];
        let cov = memsafe_covariance_matrix(&data);
        assert_eq!(cov.len(), 2);
        assert!(cov[0][0] > 0.0);
    }

    #[test]
    fn test_memsafe_correlation_matrix() {
        let data = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]];
        let corr = memsafe_correlation_matrix(&data);
        assert!((corr[0][1] - 1.0).abs() < 1e-10); // perfect correlation
    }

}

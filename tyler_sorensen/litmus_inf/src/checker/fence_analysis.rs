//! Fence placement analysis for LITMUS∞.
//!
//! Implements fence type classification, optimal fence insertion,
//! fence strength comparison, fence elimination, cost-optimal fencing,
//! and architecture-specific fence mapping.

use std::collections::{HashMap, HashSet, BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::cmp::Ordering as CmpOrdering;
use serde::{Serialize, Deserialize};

use super::execution::{EventId, ThreadId, Address, Value, OpType, Scope, ExecutionGraph, BitMatrix};
use super::memory_model::{MemoryModel, RelationExpr, Constraint};
use super::litmus::{Instruction, Ordering as MemOrdering, Scope as FenceScope};

// ---------------------------------------------------------------------------
// Fence Types
// ---------------------------------------------------------------------------

/// Types of memory fences.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FenceType {
    /// Load-Load fence (prevents load reordering).
    LoadLoad,
    /// Load-Store fence (prevents load-before-store reordering).
    LoadStore,
    /// Store-Store fence (prevents store reordering).
    StoreStore,
    /// Store-Load fence (prevents store-before-load reordering, most expensive).
    StoreLoad,
    /// Full fence (prevents all reordering).
    Full,
    /// Acquire fence (prevents reads/writes from moving before it).
    Acquire,
    /// Release fence (prevents reads/writes from moving after it).
    Release,
    /// Sequential consistency fence.
    SeqCst,
    /// GPU CTA-scoped fence.
    GpuCta,
    /// GPU device-scoped fence.
    GpuDevice,
    /// GPU system-scoped fence.
    GpuSystem,
}

impl FenceType {
    /// All fence types.
    pub fn all() -> Vec<Self> {
        vec![
            Self::LoadLoad, Self::LoadStore, Self::StoreStore, Self::StoreLoad,
            Self::Full, Self::Acquire, Self::Release, Self::SeqCst,
            Self::GpuCta, Self::GpuDevice, Self::GpuSystem,
        ]
    }

    /// CPU fence types only.
    pub fn cpu_fences() -> Vec<Self> {
        vec![
            Self::LoadLoad, Self::LoadStore, Self::StoreStore, Self::StoreLoad,
            Self::Full, Self::Acquire, Self::Release, Self::SeqCst,
        ]
    }

    /// GPU fence types only.
    pub fn gpu_fences() -> Vec<Self> {
        vec![Self::GpuCta, Self::GpuDevice, Self::GpuSystem]
    }

    /// Whether this fence prevents the given reordering.
    pub fn prevents(&self, from: OpType, to: OpType) -> bool {
        match self {
            Self::LoadLoad => from == OpType::Read && to == OpType::Read,
            Self::LoadStore => from == OpType::Read && to == OpType::Write,
            Self::StoreStore => from == OpType::Write && to == OpType::Write,
            Self::StoreLoad => from == OpType::Write && to == OpType::Read,
            Self::Full | Self::SeqCst => true,
            Self::Acquire => to == OpType::Read || to == OpType::Write,
            Self::Release => from == OpType::Read || from == OpType::Write,
            Self::GpuCta | Self::GpuDevice | Self::GpuSystem => true,
        }
    }

    /// Whether this is a GPU fence.
    pub fn is_gpu(&self) -> bool {
        matches!(self, Self::GpuCta | Self::GpuDevice | Self::GpuSystem)
    }

    /// Get the scope for GPU fences.
    pub fn gpu_scope(&self) -> Option<Scope> {
        match self {
            Self::GpuCta => Some(Scope::CTA),
            Self::GpuDevice => Some(Scope::GPU),
            Self::GpuSystem => Some(Scope::System),
            _ => None,
        }
    }
}

impl fmt::Display for FenceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LoadLoad => write!(f, "LL"),
            Self::LoadStore => write!(f, "LS"),
            Self::StoreStore => write!(f, "SS"),
            Self::StoreLoad => write!(f, "SL"),
            Self::Full => write!(f, "FULL"),
            Self::Acquire => write!(f, "ACQ"),
            Self::Release => write!(f, "REL"),
            Self::SeqCst => write!(f, "SC"),
            Self::GpuCta => write!(f, "GPU.CTA"),
            Self::GpuDevice => write!(f, "GPU.DEV"),
            Self::GpuSystem => write!(f, "GPU.SYS"),
        }
    }
}

// ---------------------------------------------------------------------------
// Fence Strength
// ---------------------------------------------------------------------------

/// Fence strength level for comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FenceStrength(u32);

impl FenceStrength {
    /// Strength value (higher = stronger).
    pub fn value(&self) -> u32 {
        self.0
    }

    /// Get the strength of a fence type.
    pub fn of(fence: FenceType) -> Self {
        match fence {
            FenceType::LoadLoad => Self(1),
            FenceType::StoreStore => Self(1),
            FenceType::LoadStore => Self(2),
            FenceType::StoreLoad => Self(3),
            FenceType::Acquire => Self(4),
            FenceType::Release => Self(4),
            FenceType::Full => Self(5),
            FenceType::SeqCst => Self(6),
            FenceType::GpuCta => Self(3),
            FenceType::GpuDevice => Self(5),
            FenceType::GpuSystem => Self(6),
        }
    }
}

impl PartialOrd for FenceStrength {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.0.cmp(&other.0))
    }
}

impl Ord for FenceStrength {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        self.0.cmp(&other.0)
    }
}

// ---------------------------------------------------------------------------
// Fence Strength Lattice
// ---------------------------------------------------------------------------

/// Lattice of fence strengths with join and meet operations.
#[derive(Debug, Clone)]
pub struct FenceStrengthLattice {
    /// Implication map: fence A implies fence B.
    implies: HashMap<FenceType, HashSet<FenceType>>,
}

impl FenceStrengthLattice {
    /// Create the standard fence strength lattice.
    pub fn new() -> Self {
        let mut implies: HashMap<FenceType, HashSet<FenceType>> = HashMap::new();

        // Full implies all directional fences.
        let full_implies: HashSet<FenceType> = [
            FenceType::LoadLoad, FenceType::LoadStore,
            FenceType::StoreStore, FenceType::StoreLoad,
            FenceType::Acquire, FenceType::Release,
        ].into_iter().collect();
        implies.insert(FenceType::Full, full_implies);

        // SeqCst implies Full and everything Full implies.
        let mut sc_implies = implies.get(&FenceType::Full).cloned().unwrap_or_default();
        sc_implies.insert(FenceType::Full);
        implies.insert(FenceType::SeqCst, sc_implies);

        // Acquire implies LoadLoad and LoadStore.
        implies.insert(FenceType::Acquire,
            [FenceType::LoadLoad, FenceType::LoadStore].into_iter().collect());

        // Release implies StoreStore and LoadStore.
        implies.insert(FenceType::Release,
            [FenceType::StoreStore, FenceType::LoadStore].into_iter().collect());

        // StoreLoad implies nothing weaker on its own.
        implies.insert(FenceType::StoreLoad, HashSet::new());

        // GPU hierarchy.
        implies.insert(FenceType::GpuSystem,
            [FenceType::GpuDevice, FenceType::GpuCta].into_iter().collect());
        implies.insert(FenceType::GpuDevice,
            [FenceType::GpuCta].into_iter().collect());

        Self { implies }
    }

    /// Check if fence A implies fence B (A is at least as strong as B).
    pub fn implies(&self, a: FenceType, b: FenceType) -> bool {
        if a == b { return true; }
        self.implies.get(&a).map_or(false, |set| set.contains(&b))
    }

    /// Find the weakest fence that implies both A and B (join/least upper bound).
    pub fn join(&self, a: FenceType, b: FenceType) -> FenceType {
        if self.implies(a, b) { return a; }
        if self.implies(b, a) { return b; }

        // Find the weakest fence that implies both.
        let all_fences = FenceType::all();
        let mut candidates: Vec<_> = all_fences.iter()
            .filter(|&&f| self.implies(f, a) && self.implies(f, b))
            .copied()
            .collect();

        candidates.sort_by_key(|f| FenceStrength::of(*f));
        candidates.first().copied().unwrap_or(FenceType::Full)
    }

    /// Find the strongest fence implied by both A and B (meet/greatest lower bound).
    pub fn meet(&self, a: FenceType, b: FenceType) -> Option<FenceType> {
        if self.implies(a, b) { return Some(b); }
        if self.implies(b, a) { return Some(a); }

        // Find the strongest fence implied by both.
        let all_fences = FenceType::all();
        let mut candidates: Vec<_> = all_fences.iter()
            .filter(|&&f| self.implies(a, f) && self.implies(b, f))
            .copied()
            .collect();

        candidates.sort_by_key(|f| std::cmp::Reverse(FenceStrength::of(*f)));
        candidates.first().copied()
    }

    /// Get all fences implied by the given fence.
    pub fn implied_by(&self, fence: FenceType) -> Vec<FenceType> {
        let mut result = vec![fence];
        if let Some(implied) = self.implies.get(&fence) {
            result.extend(implied.iter().copied());
        }
        result
    }
}

impl Default for FenceStrengthLattice {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Ordering Requirement
// ---------------------------------------------------------------------------

/// An ordering requirement between two events that needs enforcement.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OrderingRequirement {
    /// Source event.
    pub source: EventId,
    /// Target event.
    pub target: EventId,
    /// Source operation type.
    pub source_op: OpType,
    /// Target operation type.
    pub target_op: OpType,
    /// Thread of source.
    pub source_thread: ThreadId,
    /// Thread of target.
    pub target_thread: ThreadId,
    /// Whether the ordering is already guaranteed by the memory model.
    pub guaranteed: bool,
    /// Minimum fence strength needed.
    pub min_fence: Option<FenceType>,
}

impl OrderingRequirement {
    pub fn new(
        source: EventId, target: EventId,
        source_op: OpType, target_op: OpType,
        source_thread: ThreadId, target_thread: ThreadId,
    ) -> Self {
        Self {
            source, target, source_op, target_op,
            source_thread, target_thread,
            guaranteed: false, min_fence: None,
        }
    }

    /// Check if this is a same-thread ordering.
    pub fn is_intra_thread(&self) -> bool {
        self.source_thread == self.target_thread
    }

    /// Determine the minimum fence type needed for this requirement.
    pub fn required_fence_type(&self) -> FenceType {
        match (self.source_op, self.target_op) {
            (OpType::Read, OpType::Read) => FenceType::LoadLoad,
            (OpType::Read, OpType::Write) => FenceType::LoadStore,
            (OpType::Write, OpType::Write) => FenceType::StoreStore,
            (OpType::Write, OpType::Read) => FenceType::StoreLoad,
            _ => FenceType::Full,
        }
    }
}

impl fmt::Display for OrderingRequirement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({}) -> {}({})",
            self.source, self.source_op, self.target, self.target_op)?;
        if self.guaranteed {
            write!(f, " [guaranteed]")?;
        }
        if let Some(fence) = self.min_fence {
            write!(f, " [needs {}]", fence)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Fence Placement
// ---------------------------------------------------------------------------

/// A proposed fence placement.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FencePlacement {
    /// Thread where the fence is placed.
    pub thread: ThreadId,
    /// Position in the thread's instruction sequence (between position-1 and position).
    pub position: usize,
    /// Type of fence.
    pub fence_type: FenceType,
    /// Ordering requirements satisfied by this fence.
    pub satisfies: Vec<(EventId, EventId)>,
}

impl FencePlacement {
    pub fn new(thread: ThreadId, position: usize, fence_type: FenceType) -> Self {
        Self {
            thread, position, fence_type,
            satisfies: Vec::new(),
        }
    }

    /// Add a satisfied ordering requirement.
    pub fn add_satisfied(&mut self, source: EventId, target: EventId) {
        self.satisfies.push((source, target));
    }
}

impl fmt::Display for FencePlacement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "T{}@{}: {} (satisfies {} requirements)",
            self.thread, self.position, self.fence_type, self.satisfies.len())
    }
}

// ---------------------------------------------------------------------------
// Fence Cost Model
// ---------------------------------------------------------------------------

/// Cost model for different fence types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FenceCostModel {
    costs: HashMap<FenceType, f64>,
}

impl FenceCostModel {
    /// Create a default cost model based on typical hardware costs.
    pub fn default_cpu() -> Self {
        let mut costs = HashMap::new();
        costs.insert(FenceType::LoadLoad, 1.0);
        costs.insert(FenceType::StoreStore, 1.0);
        costs.insert(FenceType::LoadStore, 2.0);
        costs.insert(FenceType::StoreLoad, 10.0);
        costs.insert(FenceType::Acquire, 3.0);
        costs.insert(FenceType::Release, 3.0);
        costs.insert(FenceType::Full, 15.0);
        costs.insert(FenceType::SeqCst, 20.0);
        Self { costs }
    }

    /// Create a cost model for GPU fences.
    pub fn default_gpu() -> Self {
        let mut costs = HashMap::new();
        costs.insert(FenceType::GpuCta, 5.0);
        costs.insert(FenceType::GpuDevice, 50.0);
        costs.insert(FenceType::GpuSystem, 200.0);
        costs.insert(FenceType::Full, 100.0);
        Self { costs }
    }

    /// Create a custom cost model.
    pub fn custom(costs: HashMap<FenceType, f64>) -> Self {
        Self { costs }
    }

    /// Get the cost of a fence type.
    pub fn cost(&self, fence: FenceType) -> f64 {
        self.costs.get(&fence).copied().unwrap_or(100.0)
    }

    /// Get the total cost of a set of fence placements.
    pub fn total_cost(&self, placements: &[FencePlacement]) -> f64 {
        placements.iter().map(|p| self.cost(p.fence_type)).sum()
    }

    /// Set the cost for a fence type.
    pub fn set_cost(&mut self, fence: FenceType, cost: f64) {
        self.costs.insert(fence, cost);
    }
}

impl Default for FenceCostModel {
    fn default() -> Self {
        Self::default_cpu()
    }
}

// ---------------------------------------------------------------------------
// Fence Placement Analyzer
// ---------------------------------------------------------------------------

/// Analyzes where fences are needed to enforce a memory model's constraints.
#[derive(Debug, Clone)]
pub struct FencePlacementAnalyzer {
    /// The memory model being targeted.
    model_name: String,
    /// Ordering requirements extracted from the model.
    requirements: Vec<OrderingRequirement>,
    /// Fence strength lattice.
    lattice: FenceStrengthLattice,
}

impl FencePlacementAnalyzer {
    /// Create a new analyzer for the given model.
    pub fn new(model_name: &str) -> Self {
        Self {
            model_name: model_name.to_string(),
            requirements: Vec::new(),
            lattice: FenceStrengthLattice::new(),
        }
    }

    /// Add an ordering requirement.
    pub fn add_requirement(&mut self, req: OrderingRequirement) {
        self.requirements.push(req);
    }

    /// Analyze an execution graph to find critical edges that need fencing.
    pub fn analyze_graph(&mut self, graph: &ExecutionGraph) -> Vec<OrderingRequirement> {
        let n = graph.events.len();
        let events = &graph.events;
        let mut requirements = Vec::new();

        // Check all pairs of events in the same thread.
        for i in 0..n {
            for j in (i+1)..n {
                let ei = &events[i];
                let ej = &events[j];

                if ei.thread != ej.thread {
                    continue;
                }

                // Check if this pair needs ordering beyond program order.
                let needs_fence = self.check_pair_needs_fence(ei.op_type, ej.op_type);
                if needs_fence {
                    let mut req = OrderingRequirement::new(
                        ei.id, ej.id, ei.op_type, ej.op_type,
                        ei.thread, ej.thread,
                    );
                    req.min_fence = Some(req.required_fence_type());
                    requirements.push(req);
                }
            }
        }

        self.requirements = requirements.clone();
        requirements
    }

    /// Check if a pair of operations needs a fence (model-dependent).
    fn check_pair_needs_fence(&self, from: OpType, to: OpType) -> bool {
        match self.model_name.as_str() {
            "SC" => false, // SC preserves all orderings.
            "TSO" | "x86-TSO" => {
                // TSO only allows store-load reordering.
                from == OpType::Write && to == OpType::Read
            }
            "PSO" => {
                // PSO allows store-load and store-store reordering.
                matches!((from, to),
                    (OpType::Write, OpType::Read) | (OpType::Write, OpType::Write)
                )
            }
            "ARM" | "ARMv8" | "RISC-V" => {
                // ARM/RISC-V allow all reorderings except data-dependent.
                true
            }
            _ => true,
        }
    }

    /// Get unguaranteed requirements (those needing fences).
    pub fn unguaranteed_requirements(&self) -> Vec<&OrderingRequirement> {
        self.requirements.iter().filter(|r| !r.guaranteed).collect()
    }

    /// Group requirements by thread.
    pub fn requirements_by_thread(&self) -> HashMap<ThreadId, Vec<&OrderingRequirement>> {
        let mut by_thread: HashMap<ThreadId, Vec<&OrderingRequirement>> = HashMap::new();
        for req in &self.requirements {
            if req.is_intra_thread() {
                by_thread.entry(req.source_thread).or_default().push(req);
            }
        }
        by_thread
    }

    /// Count total requirements.
    pub fn requirement_count(&self) -> usize {
        self.requirements.len()
    }
}

// ---------------------------------------------------------------------------
// Optimal Fence Inserter
// ---------------------------------------------------------------------------

/// Finds minimum-cost fence placement satisfying all ordering constraints.
#[derive(Debug, Clone)]
pub struct OptimalFenceInserter {
    cost_model: FenceCostModel,
    lattice: FenceStrengthLattice,
}

impl OptimalFenceInserter {
    /// Create a new inserter with the given cost model.
    pub fn new(cost_model: FenceCostModel) -> Self {
        Self {
            cost_model,
            lattice: FenceStrengthLattice::new(),
        }
    }

    /// Find optimal fence placements using a greedy approach.
    pub fn find_optimal(&self, requirements: &[OrderingRequirement]) -> Vec<FencePlacement> {
        if requirements.is_empty() {
            return Vec::new();
        }

        let mut placements = Vec::new();
        let mut satisfied: HashSet<usize> = HashSet::new();

        // Group requirements by thread and position.
        let mut by_position: BTreeMap<(ThreadId, usize), Vec<(usize, &OrderingRequirement)>> = BTreeMap::new();
        for (idx, req) in requirements.iter().enumerate() {
            if req.guaranteed || !req.is_intra_thread() {
                satisfied.insert(idx);
                continue;
            }
            // Place fence between source and target.
            let pos = req.source + 1; // After the source event.
            by_position.entry((req.source_thread, pos)).or_default().push((idx, req));
        }

        // For each position, determine the minimum fence that satisfies all requirements.
        for ((thread, position), reqs) in &by_position {
            let unsatisfied: Vec<_> = reqs.iter()
                .filter(|(idx, _)| !satisfied.contains(idx))
                .collect();

            if unsatisfied.is_empty() {
                continue;
            }

            // Find the minimum fence type that covers all requirements at this position.
            let needed_fence = self.find_minimum_fence(&unsatisfied);

            let mut placement = FencePlacement::new(*thread, *position, needed_fence);
            for (idx, req) in &unsatisfied {
                placement.add_satisfied(req.source, req.target);
                satisfied.insert(*idx);
            }
            placements.push(placement);
        }

        placements
    }

    /// Find the minimum fence type that satisfies all given requirements.
    fn find_minimum_fence(&self, requirements: &[&(usize, &OrderingRequirement)]) -> FenceType {
        let mut needed = HashSet::new();
        for (_, req) in requirements {
            needed.insert(req.required_fence_type());
        }

        // Find the cheapest fence that implies all needed fences.
        let all_fences = FenceType::all();
        let mut candidates: Vec<_> = all_fences.iter()
            .filter(|&&f| needed.iter().all(|&n| self.lattice.implies(f, n) || f == n))
            .map(|&f| (f, self.cost_model.cost(f)))
            .collect();

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(CmpOrdering::Equal));
        candidates.first().map(|&(f, _)| f).unwrap_or(FenceType::Full)
    }

    /// Optimize an existing set of placements by merging compatible fences.
    pub fn optimize_placements(&self, placements: &[FencePlacement]) -> Vec<FencePlacement> {
        if placements.len() <= 1 {
            return placements.to_vec();
        }

        let mut optimized = Vec::new();
        let mut by_thread: HashMap<ThreadId, Vec<&FencePlacement>> = HashMap::new();

        for p in placements {
            by_thread.entry(p.thread).or_default().push(p);
        }

        for (thread, mut thread_placements) in by_thread {
            thread_placements.sort_by_key(|p| p.position);

            let mut i = 0;
            while i < thread_placements.len() {
                let mut merged = thread_placements[i].clone();
                let mut j = i + 1;

                // Merge adjacent fences of the same or compatible type.
                while j < thread_placements.len() {
                    let next = thread_placements[j];
                    if next.position == merged.position || next.position == merged.position + 1 {
                        let combined = self.lattice.join(merged.fence_type, next.fence_type);
                        let combined_cost = self.cost_model.cost(combined);
                        let separate_cost = self.cost_model.cost(merged.fence_type)
                            + self.cost_model.cost(next.fence_type);

                        if combined_cost <= separate_cost {
                            merged.fence_type = combined;
                            merged.satisfies.extend_from_slice(&next.satisfies);
                            j += 1;
                            continue;
                        }
                    }
                    break;
                }

                optimized.push(merged);
                i = j;
            }
        }

        optimized
    }

    /// Calculate the total cost of a set of placements.
    pub fn total_cost(&self, placements: &[FencePlacement]) -> f64 {
        self.cost_model.total_cost(placements)
    }
}

// ---------------------------------------------------------------------------
// Fence Eliminator
// ---------------------------------------------------------------------------

/// Identifies and eliminates redundant fences under a given memory model.
#[derive(Debug, Clone)]
pub struct FenceEliminator {
    model_name: String,
    lattice: FenceStrengthLattice,
    /// Fence types that are guaranteed by the model (never needed).
    guaranteed_orderings: HashSet<(OpType, OpType)>,
}

impl FenceEliminator {
    /// Create an eliminator for the given memory model.
    pub fn new(model_name: &str) -> Self {
        let mut guaranteed = HashSet::new();

        match model_name {
            "SC" => {
                // SC guarantees all orderings.
                for &from in &[OpType::Read, OpType::Write] {
                    for &to in &[OpType::Read, OpType::Write] {
                        guaranteed.insert((from, to));
                    }
                }
            }
            "TSO" | "x86-TSO" => {
                // TSO guarantees everything except store-load.
                guaranteed.insert((OpType::Read, OpType::Read));
                guaranteed.insert((OpType::Read, OpType::Write));
                guaranteed.insert((OpType::Write, OpType::Write));
            }
            "PSO" => {
                // PSO guarantees load-load and load-store.
                guaranteed.insert((OpType::Read, OpType::Read));
                guaranteed.insert((OpType::Read, OpType::Write));
            }
            _ => {
                // Weak models: nothing guaranteed by default.
            }
        }

        Self {
            model_name: model_name.to_string(),
            lattice: FenceStrengthLattice::new(),
            guaranteed_orderings: guaranteed,
        }
    }

    /// Check if a fence is redundant under the model.
    pub fn is_redundant(&self, fence: FenceType) -> bool {
        match fence {
            FenceType::LoadLoad => self.guaranteed_orderings.contains(&(OpType::Read, OpType::Read)),
            FenceType::LoadStore => self.guaranteed_orderings.contains(&(OpType::Read, OpType::Write)),
            FenceType::StoreStore => self.guaranteed_orderings.contains(&(OpType::Write, OpType::Write)),
            FenceType::StoreLoad => self.guaranteed_orderings.contains(&(OpType::Write, OpType::Read)),
            FenceType::Full | FenceType::SeqCst => {
                // Full/SC fences are redundant only under SC.
                self.model_name == "SC"
            }
            FenceType::Acquire => {
                self.guaranteed_orderings.contains(&(OpType::Read, OpType::Read))
                    && self.guaranteed_orderings.contains(&(OpType::Read, OpType::Write))
            }
            FenceType::Release => {
                self.guaranteed_orderings.contains(&(OpType::Write, OpType::Write))
                    && self.guaranteed_orderings.contains(&(OpType::Read, OpType::Write))
            }
            _ => false,
        }
    }

    /// Eliminate redundant fences from a set of placements.
    pub fn eliminate(&self, placements: &[FencePlacement]) -> Vec<FencePlacement> {
        placements.iter()
            .filter(|p| !self.is_redundant(p.fence_type))
            .cloned()
            .collect()
    }

    /// Try to weaken fences to cheaper alternatives.
    pub fn weaken(&self, placements: &[FencePlacement]) -> Vec<FencePlacement> {
        placements.iter()
            .map(|p| {
                let weakened = self.find_weakest_sufficient(p.fence_type);
                FencePlacement {
                    fence_type: weakened,
                    ..p.clone()
                }
            })
            .filter(|p| !self.is_redundant(p.fence_type))
            .collect()
    }

    /// Find the weakest fence that is still sufficient (not redundant).
    fn find_weakest_sufficient(&self, fence: FenceType) -> FenceType {
        let implied = self.lattice.implied_by(fence);
        let mut weakest = fence;
        let mut weakest_strength = FenceStrength::of(fence);

        for &f in &implied {
            if !self.is_redundant(f) {
                let strength = FenceStrength::of(f);
                if strength < weakest_strength {
                    weakest = f;
                    weakest_strength = strength;
                }
            }
        }
        weakest
    }

    /// Report which fences are redundant and why.
    pub fn redundancy_report(&self, placements: &[FencePlacement]) -> Vec<RedundancyReport> {
        placements.iter()
            .map(|p| RedundancyReport {
                placement: p.clone(),
                redundant: self.is_redundant(p.fence_type),
                reason: if self.is_redundant(p.fence_type) {
                    format!("{} model guarantees this ordering", self.model_name)
                } else {
                    "Required".to_string()
                },
            })
            .collect()
    }
}

/// Report on fence redundancy.
#[derive(Debug, Clone)]
pub struct RedundancyReport {
    pub placement: FencePlacement,
    pub redundant: bool,
    pub reason: String,
}

impl fmt::Display for RedundancyReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if self.redundant { "REDUNDANT" } else { "REQUIRED" };
        write!(f, "[{}] {} - {}", status, self.placement, self.reason)
    }
}

// ---------------------------------------------------------------------------
// Cost-Optimal Fencer
// ---------------------------------------------------------------------------

/// Finds the cost-optimal fence placement using weighted graph analysis.
#[derive(Debug, Clone)]
pub struct CostOptimalFencer {
    cost_model: FenceCostModel,
    lattice: FenceStrengthLattice,
    inserter: OptimalFenceInserter,
    eliminator: Option<FenceEliminator>,
}

impl CostOptimalFencer {
    /// Create a new cost-optimal fencer.
    pub fn new(cost_model: FenceCostModel) -> Self {
        let inserter = OptimalFenceInserter::new(cost_model.clone());
        Self {
            cost_model,
            lattice: FenceStrengthLattice::new(),
            inserter,
            eliminator: None,
        }
    }

    /// Set the memory model for fence elimination.
    pub fn with_model(mut self, model_name: &str) -> Self {
        self.eliminator = Some(FenceEliminator::new(model_name));
        self
    }

    /// Find cost-optimal fence placement.
    pub fn optimize(&self, requirements: &[OrderingRequirement]) -> FencingResult {
        // Step 1: Find initial placement.
        let initial = self.inserter.find_optimal(requirements);

        // Step 2: Eliminate redundant fences.
        let after_elimination = if let Some(ref elim) = self.eliminator {
            elim.eliminate(&initial)
        } else {
            initial.clone()
        };

        // Step 3: Optimize by merging compatible fences.
        let optimized = self.inserter.optimize_placements(&after_elimination);

        // Step 4: Try weakening fences.
        let weakened = if let Some(ref elim) = self.eliminator {
            elim.weaken(&optimized)
        } else {
            optimized.clone()
        };

        let initial_cost = self.cost_model.total_cost(&initial);
        let final_cost = self.cost_model.total_cost(&weakened);

        FencingResult {
            placements: weakened,
            total_cost: final_cost,
            initial_cost,
            fences_eliminated: initial.len() - after_elimination.len(),
            cost_savings: initial_cost - final_cost,
        }
    }
}

/// Result of cost-optimal fence placement.
#[derive(Debug, Clone)]
pub struct FencingResult {
    pub placements: Vec<FencePlacement>,
    pub total_cost: f64,
    pub initial_cost: f64,
    pub fences_eliminated: usize,
    pub cost_savings: f64,
}

impl fmt::Display for FencingResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Fencing Result:")?;
        writeln!(f, "  Placements: {}", self.placements.len())?;
        writeln!(f, "  Total cost: {:.1}", self.total_cost)?;
        writeln!(f, "  Initial cost: {:.1}", self.initial_cost)?;
        writeln!(f, "  Fences eliminated: {}", self.fences_eliminated)?;
        writeln!(f, "  Cost savings: {:.1}", self.cost_savings)?;
        for p in &self.placements {
            writeln!(f, "    {}", p)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Fence Placement Verifier
// ---------------------------------------------------------------------------

/// Verifies whether a proposed fence placement is sufficient.
#[derive(Debug, Clone)]
pub struct FencePlacementVerifier {
    lattice: FenceStrengthLattice,
}

impl FencePlacementVerifier {
    pub fn new() -> Self {
        Self {
            lattice: FenceStrengthLattice::new(),
        }
    }

    /// Verify that the placements satisfy all requirements.
    pub fn verify(
        &self,
        placements: &[FencePlacement],
        requirements: &[OrderingRequirement],
    ) -> VerificationOutcome {
        let mut unsatisfied = Vec::new();
        let mut satisfied_count = 0;

        for req in requirements {
            if req.guaranteed {
                satisfied_count += 1;
                continue;
            }

            let is_satisfied = placements.iter().any(|p| {
                p.thread == req.source_thread
                    && p.position > req.source
                    && p.position <= req.target
                    && p.fence_type.prevents(req.source_op, req.target_op)
            });

            if is_satisfied {
                satisfied_count += 1;
            } else {
                unsatisfied.push(req.clone());
            }
        }

        VerificationOutcome {
            sufficient: unsatisfied.is_empty(),
            satisfied: satisfied_count,
            total: requirements.len(),
            unsatisfied_requirements: unsatisfied,
        }
    }
}

impl Default for FencePlacementVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Outcome of fence placement verification.
#[derive(Debug, Clone)]
pub struct VerificationOutcome {
    pub sufficient: bool,
    pub satisfied: usize,
    pub total: usize,
    pub unsatisfied_requirements: Vec<OrderingRequirement>,
}

impl fmt::Display for VerificationOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Verification: {}/{} satisfied ({})",
            self.satisfied, self.total,
            if self.sufficient { "SUFFICIENT" } else { "INSUFFICIENT" })?;
        if !self.unsatisfied_requirements.is_empty() {
            write!(f, "\n  Unsatisfied:")?;
            for req in &self.unsatisfied_requirements {
                write!(f, "\n    {}", req)?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Architecture-Specific Fence Mapping
// ---------------------------------------------------------------------------

/// Target architecture for fence mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TargetArch {
    X86,
    ARM,
    AArch64,
    RISCV,
    PTX,
    SPIRV,
}

impl fmt::Display for TargetArch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::X86 => write!(f, "x86"),
            Self::ARM => write!(f, "ARM"),
            Self::AArch64 => write!(f, "AArch64"),
            Self::RISCV => write!(f, "RISC-V"),
            Self::PTX => write!(f, "PTX"),
            Self::SPIRV => write!(f, "SPIR-V"),
        }
    }
}

/// A concrete architecture-specific fence instruction.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ArchFenceInstruction {
    pub mnemonic: String,
    pub arch: TargetArch,
    pub description: String,
    pub latency_cycles: u32,
}

impl fmt::Display for ArchFenceInstruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.mnemonic, self.arch)
    }
}

/// Maps abstract fence types to architecture-specific instructions.
#[derive(Debug, Clone)]
pub struct ArchFenceMapper {
    mappings: HashMap<(TargetArch, FenceType), ArchFenceInstruction>,
}

impl ArchFenceMapper {
    /// Create a mapper with standard mappings.
    pub fn new() -> Self {
        let mut mappings = HashMap::new();

        // x86 mappings.
        mappings.insert((TargetArch::X86, FenceType::Full), ArchFenceInstruction {
            mnemonic: "MFENCE".to_string(), arch: TargetArch::X86,
            description: "Full memory fence".to_string(), latency_cycles: 33,
        });
        mappings.insert((TargetArch::X86, FenceType::StoreLoad), ArchFenceInstruction {
            mnemonic: "MFENCE".to_string(), arch: TargetArch::X86,
            description: "Store-load fence (requires MFENCE on x86)".to_string(), latency_cycles: 33,
        });
        mappings.insert((TargetArch::X86, FenceType::LoadLoad), ArchFenceInstruction {
            mnemonic: "LFENCE".to_string(), arch: TargetArch::X86,
            description: "Load fence".to_string(), latency_cycles: 4,
        });
        mappings.insert((TargetArch::X86, FenceType::StoreStore), ArchFenceInstruction {
            mnemonic: "SFENCE".to_string(), arch: TargetArch::X86,
            description: "Store fence".to_string(), latency_cycles: 4,
        });
        mappings.insert((TargetArch::X86, FenceType::SeqCst), ArchFenceInstruction {
            mnemonic: "MFENCE".to_string(), arch: TargetArch::X86,
            description: "Sequential consistency (MFENCE)".to_string(), latency_cycles: 33,
        });

        // AArch64 mappings.
        mappings.insert((TargetArch::AArch64, FenceType::Full), ArchFenceInstruction {
            mnemonic: "DMB ISH".to_string(), arch: TargetArch::AArch64,
            description: "Data memory barrier, inner shareable".to_string(), latency_cycles: 20,
        });
        mappings.insert((TargetArch::AArch64, FenceType::LoadLoad), ArchFenceInstruction {
            mnemonic: "DMB ISHLD".to_string(), arch: TargetArch::AArch64,
            description: "Load-load barrier".to_string(), latency_cycles: 10,
        });
        mappings.insert((TargetArch::AArch64, FenceType::StoreStore), ArchFenceInstruction {
            mnemonic: "DMB ISHST".to_string(), arch: TargetArch::AArch64,
            description: "Store-store barrier".to_string(), latency_cycles: 10,
        });
        mappings.insert((TargetArch::AArch64, FenceType::Acquire), ArchFenceInstruction {
            mnemonic: "DMB ISHLD".to_string(), arch: TargetArch::AArch64,
            description: "Acquire (load barrier)".to_string(), latency_cycles: 10,
        });
        mappings.insert((TargetArch::AArch64, FenceType::Release), ArchFenceInstruction {
            mnemonic: "DMB ISH".to_string(), arch: TargetArch::AArch64,
            description: "Release (full barrier)".to_string(), latency_cycles: 20,
        });
        mappings.insert((TargetArch::AArch64, FenceType::SeqCst), ArchFenceInstruction {
            mnemonic: "DSB ISH".to_string(), arch: TargetArch::AArch64,
            description: "Data synchronization barrier".to_string(), latency_cycles: 40,
        });

        // RISC-V mappings.
        mappings.insert((TargetArch::RISCV, FenceType::Full), ArchFenceInstruction {
            mnemonic: "fence iorw, iorw".to_string(), arch: TargetArch::RISCV,
            description: "Full fence".to_string(), latency_cycles: 15,
        });
        mappings.insert((TargetArch::RISCV, FenceType::LoadLoad), ArchFenceInstruction {
            mnemonic: "fence ir, ir".to_string(), arch: TargetArch::RISCV,
            description: "Load-load fence".to_string(), latency_cycles: 5,
        });
        mappings.insert((TargetArch::RISCV, FenceType::StoreStore), ArchFenceInstruction {
            mnemonic: "fence ow, ow".to_string(), arch: TargetArch::RISCV,
            description: "Store-store fence".to_string(), latency_cycles: 5,
        });
        mappings.insert((TargetArch::RISCV, FenceType::Acquire), ArchFenceInstruction {
            mnemonic: "fence ir, iorw".to_string(), arch: TargetArch::RISCV,
            description: "Acquire fence".to_string(), latency_cycles: 8,
        });
        mappings.insert((TargetArch::RISCV, FenceType::Release), ArchFenceInstruction {
            mnemonic: "fence iorw, ow".to_string(), arch: TargetArch::RISCV,
            description: "Release fence".to_string(), latency_cycles: 8,
        });
        mappings.insert((TargetArch::RISCV, FenceType::SeqCst), ArchFenceInstruction {
            mnemonic: "fence iorw, iorw".to_string(), arch: TargetArch::RISCV,
            description: "SeqCst fence".to_string(), latency_cycles: 15,
        });

        // PTX mappings.
        mappings.insert((TargetArch::PTX, FenceType::GpuCta), ArchFenceInstruction {
            mnemonic: "membar.cta".to_string(), arch: TargetArch::PTX,
            description: "CTA-scoped memory barrier".to_string(), latency_cycles: 10,
        });
        mappings.insert((TargetArch::PTX, FenceType::GpuDevice), ArchFenceInstruction {
            mnemonic: "membar.gl".to_string(), arch: TargetArch::PTX,
            description: "Device-scoped memory barrier".to_string(), latency_cycles: 100,
        });
        mappings.insert((TargetArch::PTX, FenceType::GpuSystem), ArchFenceInstruction {
            mnemonic: "membar.sys".to_string(), arch: TargetArch::PTX,
            description: "System-scoped memory barrier".to_string(), latency_cycles: 500,
        });

        Self { mappings }
    }

    /// Map a fence type to an architecture-specific instruction.
    pub fn map(&self, arch: TargetArch, fence: FenceType) -> Option<&ArchFenceInstruction> {
        self.mappings.get(&(arch, fence))
    }

    /// Map all placements to architecture-specific instructions.
    pub fn map_placements(
        &self, arch: TargetArch, placements: &[FencePlacement],
    ) -> Vec<(FencePlacement, Option<ArchFenceInstruction>)> {
        placements.iter()
            .map(|p| (p.clone(), self.map(arch, p.fence_type).cloned()))
            .collect()
    }

    /// Get all available instructions for an architecture.
    pub fn available_instructions(&self, arch: TargetArch) -> Vec<&ArchFenceInstruction> {
        self.mappings.iter()
            .filter(|((a, _), _)| *a == arch)
            .map(|(_, instr)| instr)
            .collect()
    }

    /// Add a custom mapping.
    pub fn add_mapping(&mut self, arch: TargetArch, fence: FenceType, instr: ArchFenceInstruction) {
        self.mappings.insert((arch, fence), instr);
    }
}

impl Default for ArchFenceMapper {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Fence Analysis Pipeline
// ---------------------------------------------------------------------------

/// Complete fence analysis pipeline combining all components.
#[derive(Debug, Clone)]
pub struct FenceAnalysisPipeline {
    analyzer: FencePlacementAnalyzer,
    fencer: CostOptimalFencer,
    verifier: FencePlacementVerifier,
    mapper: ArchFenceMapper,
    target_arch: TargetArch,
}

impl FenceAnalysisPipeline {
    /// Create a new pipeline for the given model and architecture.
    pub fn new(model_name: &str, arch: TargetArch) -> Self {
        let cost_model = match arch {
            TargetArch::PTX | TargetArch::SPIRV => FenceCostModel::default_gpu(),
            _ => FenceCostModel::default_cpu(),
        };

        Self {
            analyzer: FencePlacementAnalyzer::new(model_name),
            fencer: CostOptimalFencer::new(cost_model).with_model(model_name),
            verifier: FencePlacementVerifier::new(),
            mapper: ArchFenceMapper::new(),
            target_arch: arch,
        }
    }

    /// Run the complete analysis on an execution graph.
    pub fn analyze(&mut self, graph: &ExecutionGraph) -> PipelineResult {
        // Step 1: Analyze requirements.
        let requirements = self.analyzer.analyze_graph(graph);

        // Step 2: Find optimal fences.
        let fencing = self.fencer.optimize(&requirements);

        // Step 3: Verify sufficiency.
        let verification = self.verifier.verify(&fencing.placements, &requirements);

        // Step 4: Map to architecture.
        let arch_mapped = self.mapper.map_placements(self.target_arch, &fencing.placements);

        PipelineResult {
            requirements_count: requirements.len(),
            fencing_result: fencing,
            verification: verification,
            arch_instructions: arch_mapped,
            target_arch: self.target_arch,
        }
    }

    /// Run analysis on explicit requirements.
    pub fn analyze_requirements(&self, requirements: &[OrderingRequirement]) -> PipelineResult {
        let fencing = self.fencer.optimize(requirements);
        let verification = self.verifier.verify(&fencing.placements, requirements);
        let arch_mapped = self.mapper.map_placements(self.target_arch, &fencing.placements);

        PipelineResult {
            requirements_count: requirements.len(),
            fencing_result: fencing,
            verification,
            arch_instructions: arch_mapped,
            target_arch: self.target_arch,
        }
    }
}

/// Result of the complete fence analysis pipeline.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    pub requirements_count: usize,
    pub fencing_result: FencingResult,
    pub verification: VerificationOutcome,
    pub arch_instructions: Vec<(FencePlacement, Option<ArchFenceInstruction>)>,
    pub target_arch: TargetArch,
}

impl fmt::Display for PipelineResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Fence Analysis Pipeline ===")?;
        writeln!(f, "Target: {}", self.target_arch)?;
        writeln!(f, "Requirements: {}", self.requirements_count)?;
        writeln!(f, "{}", self.fencing_result)?;
        writeln!(f, "{}", self.verification)?;
        writeln!(f, "Architecture instructions:")?;
        for (placement, instr) in &self.arch_instructions {
            if let Some(instr) = instr {
                writeln!(f, "  {} -> {}", placement, instr)?;
            } else {
                writeln!(f, "  {} -> (no mapping)", placement)?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Fence Statistics
// ---------------------------------------------------------------------------

/// Statistics about fence analysis results.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FenceStatistics {
    pub total_requirements: usize,
    pub guaranteed_requirements: usize,
    pub fences_needed: usize,
    pub fences_eliminated: usize,
    pub total_cost: f64,
    pub optimized_cost: f64,
    pub fence_type_counts: HashMap<String, usize>,
}

impl FenceStatistics {
    /// Compute statistics from analysis results.
    pub fn from_results(
        requirements: &[OrderingRequirement],
        placements: &[FencePlacement],
        cost_model: &FenceCostModel,
    ) -> Self {
        let mut fence_type_counts = HashMap::new();
        for p in placements {
            *fence_type_counts.entry(p.fence_type.to_string()).or_insert(0) += 1;
        }

        Self {
            total_requirements: requirements.len(),
            guaranteed_requirements: requirements.iter().filter(|r| r.guaranteed).count(),
            fences_needed: placements.len(),
            fences_eliminated: 0,
            total_cost: cost_model.total_cost(placements),
            optimized_cost: cost_model.total_cost(placements),
            fence_type_counts,
        }
    }
}

impl fmt::Display for FenceStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Fence Statistics:")?;
        writeln!(f, "  Requirements: {} ({} guaranteed)",
            self.total_requirements, self.guaranteed_requirements)?;
        writeln!(f, "  Fences needed: {}", self.fences_needed)?;
        writeln!(f, "  Fences eliminated: {}", self.fences_eliminated)?;
        writeln!(f, "  Total cost: {:.1}", self.total_cost)?;
        writeln!(f, "  Optimized cost: {:.1}", self.optimized_cost)?;
        writeln!(f, "  Types:")?;
        for (ft, count) in &self.fence_type_counts {
            writeln!(f, "    {}: {}", ft, count)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- FenceType tests --

    #[test]
    fn test_fence_type_prevents() {
        assert!(FenceType::LoadLoad.prevents(OpType::Read, OpType::Read));
        assert!(!FenceType::LoadLoad.prevents(OpType::Write, OpType::Read));
        assert!(FenceType::StoreLoad.prevents(OpType::Write, OpType::Read));
        assert!(FenceType::Full.prevents(OpType::Read, OpType::Write));
        assert!(FenceType::Full.prevents(OpType::Write, OpType::Write));
    }

    #[test]
    fn test_fence_type_gpu() {
        assert!(FenceType::GpuCta.is_gpu());
        assert!(FenceType::GpuDevice.is_gpu());
        assert!(!FenceType::Full.is_gpu());
        assert_eq!(FenceType::GpuCta.gpu_scope(), Some(Scope::CTA));
    }

    // -- FenceStrength tests --

    #[test]
    fn test_fence_strength_ordering() {
        let ll = FenceStrength::of(FenceType::LoadLoad);
        let sl = FenceStrength::of(FenceType::StoreLoad);
        let full = FenceStrength::of(FenceType::Full);
        let sc = FenceStrength::of(FenceType::SeqCst);
        assert!(ll < sl);
        assert!(sl < full);
        assert!(full < sc);
    }

    // -- FenceStrengthLattice tests --

    #[test]
    fn test_lattice_implies() {
        let lattice = FenceStrengthLattice::new();
        assert!(lattice.implies(FenceType::Full, FenceType::LoadLoad));
        assert!(lattice.implies(FenceType::Full, FenceType::StoreStore));
        assert!(lattice.implies(FenceType::SeqCst, FenceType::Full));
        assert!(!lattice.implies(FenceType::LoadLoad, FenceType::Full));
    }

    #[test]
    fn test_lattice_implies_self() {
        let lattice = FenceStrengthLattice::new();
        for ft in FenceType::all() {
            assert!(lattice.implies(ft, ft));
        }
    }

    #[test]
    fn test_lattice_acquire_release() {
        let lattice = FenceStrengthLattice::new();
        assert!(lattice.implies(FenceType::Acquire, FenceType::LoadLoad));
        assert!(lattice.implies(FenceType::Release, FenceType::StoreStore));
    }

    #[test]
    fn test_lattice_gpu_hierarchy() {
        let lattice = FenceStrengthLattice::new();
        assert!(lattice.implies(FenceType::GpuSystem, FenceType::GpuDevice));
        assert!(lattice.implies(FenceType::GpuSystem, FenceType::GpuCta));
        assert!(lattice.implies(FenceType::GpuDevice, FenceType::GpuCta));
    }

    #[test]
    fn test_lattice_join() {
        let lattice = FenceStrengthLattice::new();
        let join = lattice.join(FenceType::LoadLoad, FenceType::StoreStore);
        assert!(lattice.implies(join, FenceType::LoadLoad));
        assert!(lattice.implies(join, FenceType::StoreStore));
    }

    // -- OrderingRequirement tests --

    #[test]
    fn test_ordering_requirement() {
        let req = OrderingRequirement::new(0, 1, OpType::Write, OpType::Read, 0, 0);
        assert!(req.is_intra_thread());
        assert_eq!(req.required_fence_type(), FenceType::StoreLoad);
    }

    #[test]
    fn test_ordering_requirement_inter_thread() {
        let req = OrderingRequirement::new(0, 1, OpType::Read, OpType::Write, 0, 1);
        assert!(!req.is_intra_thread());
    }

    // -- FenceCostModel tests --

    #[test]
    fn test_cost_model_cpu() {
        let model = FenceCostModel::default_cpu();
        assert!(model.cost(FenceType::StoreLoad) > model.cost(FenceType::LoadLoad));
        assert!(model.cost(FenceType::Full) > model.cost(FenceType::StoreLoad));
    }

    #[test]
    fn test_cost_model_gpu() {
        let model = FenceCostModel::default_gpu();
        assert!(model.cost(FenceType::GpuSystem) > model.cost(FenceType::GpuDevice));
        assert!(model.cost(FenceType::GpuDevice) > model.cost(FenceType::GpuCta));
    }

    #[test]
    fn test_cost_model_total() {
        let model = FenceCostModel::default_cpu();
        let placements = vec![
            FencePlacement::new(0, 1, FenceType::LoadLoad),
            FencePlacement::new(0, 2, FenceType::StoreLoad),
        ];
        let total = model.total_cost(&placements);
        assert_eq!(total, model.cost(FenceType::LoadLoad) + model.cost(FenceType::StoreLoad));
    }

    // -- FencePlacementAnalyzer tests --

    #[test]
    fn test_analyzer_sc() {
        let analyzer = FencePlacementAnalyzer::new("SC");
        assert!(!analyzer.check_pair_needs_fence(OpType::Write, OpType::Read));
        assert!(!analyzer.check_pair_needs_fence(OpType::Read, OpType::Write));
    }

    #[test]
    fn test_analyzer_tso() {
        let analyzer = FencePlacementAnalyzer::new("TSO");
        assert!(analyzer.check_pair_needs_fence(OpType::Write, OpType::Read));
        assert!(!analyzer.check_pair_needs_fence(OpType::Read, OpType::Read));
    }

    #[test]
    fn test_analyzer_arm() {
        let analyzer = FencePlacementAnalyzer::new("ARM");
        assert!(analyzer.check_pair_needs_fence(OpType::Write, OpType::Read));
        assert!(analyzer.check_pair_needs_fence(OpType::Read, OpType::Read));
        assert!(analyzer.check_pair_needs_fence(OpType::Write, OpType::Write));
    }

    // -- OptimalFenceInserter tests --

    #[test]
    fn test_optimal_inserter_empty() {
        let inserter = OptimalFenceInserter::new(FenceCostModel::default_cpu());
        let result = inserter.find_optimal(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_optimal_inserter_single() {
        let inserter = OptimalFenceInserter::new(FenceCostModel::default_cpu());
        let reqs = vec![
            OrderingRequirement::new(0, 1, OpType::Write, OpType::Read, 0, 0),
        ];
        let result = inserter.find_optimal(&reqs);
        assert!(!result.is_empty());
        assert_eq!(result[0].thread, 0);
    }

    #[test]
    fn test_optimal_inserter_multiple() {
        let inserter = OptimalFenceInserter::new(FenceCostModel::default_cpu());
        let reqs = vec![
            OrderingRequirement::new(0, 1, OpType::Write, OpType::Read, 0, 0),
            OrderingRequirement::new(0, 1, OpType::Write, OpType::Write, 0, 0),
        ];
        let result = inserter.find_optimal(&reqs);
        // Both should be satisfied by the same or merged fence.
        assert!(!result.is_empty());
    }

    // -- FenceEliminator tests --

    #[test]
    fn test_eliminator_sc() {
        let elim = FenceEliminator::new("SC");
        assert!(elim.is_redundant(FenceType::LoadLoad));
        assert!(elim.is_redundant(FenceType::StoreLoad));
        assert!(elim.is_redundant(FenceType::Full));
    }

    #[test]
    fn test_eliminator_tso() {
        let elim = FenceEliminator::new("TSO");
        assert!(elim.is_redundant(FenceType::LoadLoad));
        assert!(elim.is_redundant(FenceType::StoreStore));
        assert!(!elim.is_redundant(FenceType::StoreLoad));
    }

    #[test]
    fn test_eliminator_arm() {
        let elim = FenceEliminator::new("ARM");
        assert!(!elim.is_redundant(FenceType::LoadLoad));
        assert!(!elim.is_redundant(FenceType::StoreStore));
        assert!(!elim.is_redundant(FenceType::Full));
    }

    #[test]
    fn test_eliminator_eliminate() {
        let elim = FenceEliminator::new("TSO");
        let placements = vec![
            FencePlacement::new(0, 1, FenceType::LoadLoad),
            FencePlacement::new(0, 2, FenceType::StoreLoad),
        ];
        let result = elim.eliminate(&placements);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].fence_type, FenceType::StoreLoad);
    }

    // -- CostOptimalFencer tests --

    #[test]
    fn test_cost_optimal_empty() {
        let fencer = CostOptimalFencer::new(FenceCostModel::default_cpu());
        let result = fencer.optimize(&[]);
        assert!(result.placements.is_empty());
        assert_eq!(result.total_cost, 0.0);
    }

    #[test]
    fn test_cost_optimal_with_model() {
        let fencer = CostOptimalFencer::new(FenceCostModel::default_cpu())
            .with_model("TSO");
        let reqs = vec![
            OrderingRequirement::new(0, 1, OpType::Write, OpType::Read, 0, 0),
        ];
        let result = fencer.optimize(&reqs);
        assert!(!result.placements.is_empty());
    }

    // -- FencePlacementVerifier tests --

    #[test]
    fn test_verifier_sufficient() {
        let verifier = FencePlacementVerifier::new();
        let placements = vec![
            FencePlacement::new(0, 1, FenceType::Full),
        ];
        let reqs = vec![
            OrderingRequirement::new(0, 1, OpType::Write, OpType::Read, 0, 0),
        ];
        let result = verifier.verify(&placements, &reqs);
        assert!(result.sufficient);
    }

    #[test]
    fn test_verifier_insufficient() {
        let verifier = FencePlacementVerifier::new();
        let placements = vec![
            FencePlacement::new(0, 1, FenceType::LoadLoad),
        ];
        let reqs = vec![
            OrderingRequirement::new(0, 1, OpType::Write, OpType::Read, 0, 0),
        ];
        let result = verifier.verify(&placements, &reqs);
        // LoadLoad doesn't prevent store-load reordering.
        assert!(!result.sufficient);
    }

    #[test]
    fn test_verifier_guaranteed() {
        let verifier = FencePlacementVerifier::new();
        let mut req = OrderingRequirement::new(0, 1, OpType::Read, OpType::Read, 0, 0);
        req.guaranteed = true;
        let result = verifier.verify(&[], &[req]);
        assert!(result.sufficient);
    }

    // -- ArchFenceMapper tests --

    #[test]
    fn test_arch_mapper_x86() {
        let mapper = ArchFenceMapper::new();
        let instr = mapper.map(TargetArch::X86, FenceType::Full).unwrap();
        assert_eq!(instr.mnemonic, "MFENCE");
    }

    #[test]
    fn test_arch_mapper_aarch64() {
        let mapper = ArchFenceMapper::new();
        let instr = mapper.map(TargetArch::AArch64, FenceType::Full).unwrap();
        assert_eq!(instr.mnemonic, "DMB ISH");
    }

    #[test]
    fn test_arch_mapper_riscv() {
        let mapper = ArchFenceMapper::new();
        let instr = mapper.map(TargetArch::RISCV, FenceType::Full).unwrap();
        assert!(instr.mnemonic.contains("fence"));
    }

    #[test]
    fn test_arch_mapper_ptx() {
        let mapper = ArchFenceMapper::new();
        let instr = mapper.map(TargetArch::PTX, FenceType::GpuCta).unwrap();
        assert_eq!(instr.mnemonic, "membar.cta");
    }

    #[test]
    fn test_arch_mapper_available() {
        let mapper = ArchFenceMapper::new();
        let x86_instrs = mapper.available_instructions(TargetArch::X86);
        assert!(!x86_instrs.is_empty());
    }

    // -- FenceStatistics tests --

    #[test]
    fn test_fence_statistics() {
        let reqs = vec![
            OrderingRequirement::new(0, 1, OpType::Write, OpType::Read, 0, 0),
        ];
        let placements = vec![
            FencePlacement::new(0, 1, FenceType::StoreLoad),
        ];
        let stats = FenceStatistics::from_results(&reqs, &placements, &FenceCostModel::default_cpu());
        assert_eq!(stats.total_requirements, 1);
        assert_eq!(stats.fences_needed, 1);
    }

    // -- Redundancy report test --

    #[test]
    fn test_redundancy_report() {
        let elim = FenceEliminator::new("SC");
        let placements = vec![
            FencePlacement::new(0, 1, FenceType::Full),
        ];
        let reports = elim.redundancy_report(&placements);
        assert_eq!(reports.len(), 1);
        assert!(reports[0].redundant);
    }
}

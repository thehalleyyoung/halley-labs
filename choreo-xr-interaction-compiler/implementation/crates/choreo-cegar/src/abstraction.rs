//! Geometric abstraction for the spatial CEGAR engine.
//!
//! Partitions the continuous spatial domain into discrete abstract blocks,
//! enabling finite-state model checking of infinite-state spatial systems.
//! Supports iterative refinement guided by counterexample analysis.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt;

use indexmap::IndexMap;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::{
    AutomatonDef, CegarError, Guard, Point3, Plane, PredicateValuation, SceneConfiguration,
    SpatialConstraint, SpatialPredicate, SpatialPredicateId, StateId, TransitionId, Vector3, AABB,
};

// ---------------------------------------------------------------------------
// Abstract block identifier
// ---------------------------------------------------------------------------

/// Unique identifier for an abstract block within a partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AbstractBlockId(pub u64);

impl fmt::Display for AbstractBlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ab{}", self.0)
    }
}

/// Unique identifier for an abstract state (block + automaton state pair).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AbstractStateId(pub u64);

impl fmt::Display for AbstractStateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "as{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// AbstractBlock
// ---------------------------------------------------------------------------

/// An abstract block representing a region of the spatial domain.
///
/// Each block has a bounding AABB, a set of predicates that are definitely true
/// or definitely false within the block, and a representative sample point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstractBlock {
    pub id: AbstractBlockId,
    pub bounding_region: AABB,
    pub definite_true: BTreeSet<SpatialPredicateId>,
    pub definite_false: BTreeSet<SpatialPredicateId>,
    pub unknown_predicates: BTreeSet<SpatialPredicateId>,
    pub representative: Point3,
    pub depth: u32,
    pub parent: Option<AbstractBlockId>,
    pub volume: f64,
}

impl AbstractBlock {
    pub fn new(id: AbstractBlockId, region: AABB) -> Self {
        let representative = region.center();
        let volume = region.volume();
        Self {
            id,
            bounding_region: region,
            definite_true: BTreeSet::new(),
            definite_false: BTreeSet::new(),
            unknown_predicates: BTreeSet::new(),
            representative,
            depth: 0,
            parent: None,
            volume,
        }
    }

    /// Check whether a concrete point falls inside this block's region.
    pub fn contains_point(&self, p: &Point3) -> bool {
        self.bounding_region.contains_point(p)
    }

    /// Compute the predicate valuation that is definitely known for this block.
    pub fn known_valuation(&self) -> PredicateValuation {
        let mut val = PredicateValuation::new();
        for &pid in &self.definite_true {
            val.set(pid, true);
        }
        for &pid in &self.definite_false {
            val.set(pid, false);
        }
        val
    }

    /// Classify a predicate for this block by sampling corners and center.
    pub fn classify_predicate(
        &mut self,
        pred_id: SpatialPredicateId,
        pred: &SpatialPredicate,
        scene: &SceneConfiguration,
    ) {
        let sample_points = self.corner_points();
        let mut saw_true = false;
        let mut saw_false = false;

        for pt in &sample_points {
            let val = evaluate_predicate_at_point(pred, pt, scene);
            if val {
                saw_true = true;
            } else {
                saw_false = true;
            }
            if saw_true && saw_false {
                break;
            }
        }

        if saw_true && !saw_false {
            self.definite_true.insert(pred_id);
        } else if saw_false && !saw_true {
            self.definite_false.insert(pred_id);
        } else {
            self.unknown_predicates.insert(pred_id);
        }
    }

    /// Get the 8 corners of the bounding AABB.
    pub fn corner_points(&self) -> SmallVec<[Point3; 9]> {
        let b = &self.bounding_region;
        let mut pts = SmallVec::new();
        for &x in &[b.min.x, b.max.x] {
            for &y in &[b.min.y, b.max.y] {
                for &z in &[b.min.z, b.max.z] {
                    pts.push(Point3::new(x, y, z));
                }
            }
        }
        pts.push(b.center());
        pts
    }

    /// Check whether this block is fully determined (no unknown predicates).
    pub fn is_fully_determined(&self) -> bool {
        self.unknown_predicates.is_empty()
    }
}

/// Evaluate a predicate at a specific point by creating a temporary scene.
fn evaluate_predicate_at_point(
    pred: &SpatialPredicate,
    point: &Point3,
    scene: &SceneConfiguration,
) -> bool {
    match pred {
        SpatialPredicate::Proximity {
            entity_a,
            entity_b,
            threshold,
        } => {
            let pos_a = scene
                .entities
                .iter()
                .find(|e| e.id == *entity_a)
                .map(|e| e.position)
                .unwrap_or(*point);
            let pos_b = scene
                .entities
                .iter()
                .find(|e| e.id == *entity_b)
                .map(|e| e.position)
                .unwrap_or(*point);
            pos_a.distance_to(&pos_b) <= *threshold
        }
        SpatialPredicate::Inside { entity: _, region } => {
            if let Some(reg) = scene.regions.get(region) {
                reg.contains_point(point)
            } else {
                false
            }
        }
        SpatialPredicate::Intersection {
            region_a,
            region_b,
        } => {
            let a = scene.regions.get(region_a);
            let b = scene.regions.get(region_b);
            match (a, b) {
                (Some(ra), Some(rb)) => ra.intersects(rb),
                _ => false,
            }
        }
        SpatialPredicate::Containment { inner, outer } => {
            let i = scene.regions.get(inner);
            let o = scene.regions.get(outer);
            match (i, o) {
                (Some(ri), Some(ro)) => ro.contains_aabb(ri),
                _ => false,
            }
        }
        SpatialPredicate::Alignment {
            entity_a: _,
            entity_b: _,
            tolerance,
        } => {
            // Use point as representative
            point.y.abs() <= *tolerance && point.z.abs() <= *tolerance
        }
        SpatialPredicate::GazeAt { .. } => true,
        SpatialPredicate::Separation {
            entity_a,
            entity_b,
            min_distance,
        } => {
            let pos_a = scene
                .entities
                .iter()
                .find(|e| e.id == *entity_a)
                .map(|e| e.position)
                .unwrap_or(*point);
            let pos_b = scene
                .entities
                .iter()
                .find(|e| e.id == *entity_b)
                .map(|e| e.position)
                .unwrap_or(*point);
            pos_a.distance_to(&pos_b) >= *min_distance
        }
    }
}

// ---------------------------------------------------------------------------
// Block splitting
// ---------------------------------------------------------------------------

/// Split a block along the longest axis at the midpoint.
pub fn split_longest_axis(block: &AbstractBlock, next_id: &mut u64) -> (AbstractBlock, AbstractBlock) {
    let axis = block.bounding_region.longest_axis();
    let mid = block.bounding_region.axis_center(axis);
    let (left_aabb, right_aabb) = block.bounding_region.split_at_axis(axis, mid);

    let id_a = AbstractBlockId(*next_id);
    *next_id += 1;
    let id_b = AbstractBlockId(*next_id);
    *next_id += 1;

    let mut left = AbstractBlock::new(id_a, left_aabb);
    left.depth = block.depth + 1;
    left.parent = Some(block.id);

    let mut right = AbstractBlock::new(id_b, right_aabb);
    right.depth = block.depth + 1;
    right.parent = Some(block.id);

    (left, right)
}

/// Split a block at a specific plane (infeasibility-guided).
pub fn split_at_infeasibility(
    block: &AbstractBlock,
    plane: &Plane,
    next_id: &mut u64,
) -> (AbstractBlock, AbstractBlock) {
    let center = block.bounding_region.center();
    let dist = plane.signed_distance(&center);

    // Project the plane onto the best axis
    let nx = plane.normal.x.abs();
    let ny = plane.normal.y.abs();
    let nz = plane.normal.z.abs();
    let axis = if nx >= ny && nx >= nz {
        0
    } else if ny >= nz {
        1
    } else {
        2
    };

    let split_value = block.bounding_region.axis_center(axis)
        + dist * 0.5 / plane.normal.length().max(1e-12);

    // Clamp to block bounds
    let lo = match axis {
        0 => block.bounding_region.min.x,
        1 => block.bounding_region.min.y,
        _ => block.bounding_region.min.z,
    };
    let hi = match axis {
        0 => block.bounding_region.max.x,
        1 => block.bounding_region.max.y,
        _ => block.bounding_region.max.z,
    };
    let clamped = split_value.max(lo + 1e-6).min(hi - 1e-6);

    let (left_aabb, right_aabb) = block.bounding_region.split_at_axis(axis, clamped);

    let id_a = AbstractBlockId(*next_id);
    *next_id += 1;
    let id_b = AbstractBlockId(*next_id);
    *next_id += 1;

    let mut left = AbstractBlock::new(id_a, left_aabb);
    left.depth = block.depth + 1;
    left.parent = Some(block.id);

    let mut right = AbstractBlock::new(id_b, right_aabb);
    right.depth = block.depth + 1;
    right.parent = Some(block.id);

    (left, right)
}

/// Split a block along a specific axis at a specific value.
pub fn split_block(
    block: &AbstractBlock,
    axis: usize,
    value: f64,
    next_id: &mut u64,
) -> (AbstractBlock, AbstractBlock) {
    let (left_aabb, right_aabb) = block.bounding_region.split_at_axis(axis, value);

    let id_a = AbstractBlockId(*next_id);
    *next_id += 1;
    let id_b = AbstractBlockId(*next_id);
    *next_id += 1;

    let mut left = AbstractBlock::new(id_a, left_aabb);
    left.depth = block.depth + 1;
    left.parent = Some(block.id);

    let mut right = AbstractBlock::new(id_b, right_aabb);
    right.depth = block.depth + 1;
    right.parent = Some(block.id);

    (left, right)
}

/// Merge two adjacent blocks into one.
pub fn merge_blocks(a: &AbstractBlock, b: &AbstractBlock, id: AbstractBlockId) -> AbstractBlock {
    let merged_aabb = a.bounding_region.union(&b.bounding_region);
    let mut merged = AbstractBlock::new(id, merged_aabb);
    merged.depth = a.depth.min(b.depth);

    // Only predicates that are definite in both blocks remain definite
    merged.definite_true = a.definite_true.intersection(&b.definite_true).copied().collect();
    merged.definite_false = a.definite_false.intersection(&b.definite_false).copied().collect();

    // Everything else becomes unknown
    let all_preds: BTreeSet<_> = a
        .definite_true
        .iter()
        .chain(a.definite_false.iter())
        .chain(a.unknown_predicates.iter())
        .chain(b.definite_true.iter())
        .chain(b.definite_false.iter())
        .chain(b.unknown_predicates.iter())
        .copied()
        .collect();
    for pid in &all_preds {
        if !merged.definite_true.contains(pid) && !merged.definite_false.contains(pid) {
            merged.unknown_predicates.insert(*pid);
        }
    }

    merged
}

// ---------------------------------------------------------------------------
// SpatialPartition
// ---------------------------------------------------------------------------

/// A partition of the spatial domain into abstract blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialPartition {
    pub blocks: IndexMap<AbstractBlockId, AbstractBlock>,
    pub domain: AABB,
    next_block_id: u64,
    pub refinement_count: u32,
}

impl SpatialPartition {
    /// Create a new partition with a single block covering the entire domain.
    pub fn new(domain: AABB) -> Self {
        let mut blocks = IndexMap::new();
        let initial = AbstractBlock::new(AbstractBlockId(0), domain);
        blocks.insert(AbstractBlockId(0), initial);
        Self {
            blocks,
            domain,
            next_block_id: 1,
            refinement_count: 0,
        }
    }

    /// Number of blocks in the partition.
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Find which block contains a given point.
    pub fn find_block(&self, point: &Point3) -> Option<AbstractBlockId> {
        for (id, block) in &self.blocks {
            if block.contains_point(point) {
                return Some(*id);
            }
        }
        None
    }

    /// Find all blocks that overlap with a given AABB.
    pub fn find_overlapping_blocks(&self, region: &AABB) -> Vec<AbstractBlockId> {
        self.blocks
            .iter()
            .filter(|(_, b)| b.bounding_region.intersects(region))
            .map(|(id, _)| *id)
            .collect()
    }

    /// Refine the partition by splitting a specific block along its longest axis.
    pub fn refine_block(&mut self, block_id: AbstractBlockId) -> (AbstractBlockId, AbstractBlockId) {
        let block = self.blocks.get(&block_id).expect("Block not found").clone();
        let (left, right) = split_longest_axis(&block, &mut self.next_block_id);
        let left_id = left.id;
        let right_id = right.id;
        self.blocks.swap_remove(&block_id);
        self.blocks.insert(left_id, left);
        self.blocks.insert(right_id, right);
        self.refinement_count += 1;
        (left_id, right_id)
    }

    /// Refine a block with a specific splitting plane.
    pub fn refine_block_at_plane(
        &mut self,
        block_id: AbstractBlockId,
        plane: &Plane,
    ) -> (AbstractBlockId, AbstractBlockId) {
        let block = self.blocks.get(&block_id).expect("Block not found").clone();
        let (left, right) = split_at_infeasibility(&block, plane, &mut self.next_block_id);
        let left_id = left.id;
        let right_id = right.id;
        self.blocks.swap_remove(&block_id);
        self.blocks.insert(left_id, left);
        self.blocks.insert(right_id, right);
        self.refinement_count += 1;
        (left_id, right_id)
    }

    /// Classify predicates for all blocks.
    pub fn classify_all_predicates(
        &mut self,
        predicates: &IndexMap<SpatialPredicateId, SpatialPredicate>,
        scene: &SceneConfiguration,
    ) {
        let pred_list: Vec<_> = predicates.iter().map(|(k, v)| (*k, v.clone())).collect();
        for block in self.blocks.values_mut() {
            block.definite_true.clear();
            block.definite_false.clear();
            block.unknown_predicates.clear();
            for (pid, pred) in &pred_list {
                block.classify_predicate(*pid, pred, scene);
            }
        }
    }

    /// Get all block IDs.
    pub fn block_ids(&self) -> Vec<AbstractBlockId> {
        self.blocks.keys().copied().collect()
    }

    /// Allocate a new block ID.
    pub fn next_id(&mut self) -> AbstractBlockId {
        let id = AbstractBlockId(self.next_block_id);
        self.next_block_id += 1;
        id
    }

    /// Merge two blocks.
    pub fn merge_blocks_in_partition(
        &mut self,
        a_id: AbstractBlockId,
        b_id: AbstractBlockId,
    ) -> AbstractBlockId {
        let a = self.blocks.get(&a_id).expect("Block A not found").clone();
        let b = self.blocks.get(&b_id).expect("Block B not found").clone();
        let new_id = self.next_id();
        let merged = merge_blocks(&a, &b, new_id);
        self.blocks.swap_remove(&a_id);
        self.blocks.swap_remove(&b_id);
        self.blocks.insert(new_id, merged);
        new_id
    }
}

// ---------------------------------------------------------------------------
// PartitionRefinement
// ---------------------------------------------------------------------------

/// Tracks the refinement history of a spatial partition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionRefinement {
    pub history: Vec<RefinementStep>,
    pub total_splits: u64,
    pub total_merges: u64,
    pub max_depth_reached: u32,
}

impl PartitionRefinement {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            total_splits: 0,
            total_merges: 0,
            max_depth_reached: 0,
        }
    }

    pub fn record_split(
        &mut self,
        parent: AbstractBlockId,
        children: (AbstractBlockId, AbstractBlockId),
        reason: RefinementReason,
        depth: u32,
    ) {
        self.history.push(RefinementStep {
            kind: RefinementKind::Split {
                parent,
                left: children.0,
                right: children.1,
            },
            reason,
            iteration: self.total_splits as u32,
        });
        self.total_splits += 1;
        self.max_depth_reached = self.max_depth_reached.max(depth);
    }

    pub fn record_merge(
        &mut self,
        sources: (AbstractBlockId, AbstractBlockId),
        result: AbstractBlockId,
    ) {
        self.history.push(RefinementStep {
            kind: RefinementKind::Merge {
                source_a: sources.0,
                source_b: sources.1,
                result,
            },
            reason: RefinementReason::Coarsening,
            iteration: (self.total_splits + self.total_merges) as u32,
        });
        self.total_merges += 1;
    }

    /// Compute the well-founded ordering measure: total number of blocks.
    pub fn complexity_measure(&self) -> u64 {
        self.total_splits - self.total_merges
    }
}

impl Default for PartitionRefinement {
    fn default() -> Self {
        Self::new()
    }
}

/// A single refinement step in the history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinementStep {
    pub kind: RefinementKind,
    pub reason: RefinementReason,
    pub iteration: u32,
}

/// Kind of refinement step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RefinementKind {
    Split {
        parent: AbstractBlockId,
        left: AbstractBlockId,
        right: AbstractBlockId,
    },
    Merge {
        source_a: AbstractBlockId,
        source_b: AbstractBlockId,
        result: AbstractBlockId,
    },
}

/// Reason for a refinement operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RefinementReason {
    SpuriousCounterexample,
    InfeasibilityGuided,
    UniformRefinement,
    Coarsening,
    InterpolantGuided,
}

// ---------------------------------------------------------------------------
// AbstractDomain trait
// ---------------------------------------------------------------------------

/// Trait for abstract domains supporting the CEGAR refinement cycle.
pub trait AbstractDomain: Clone {
    /// The concrete state type.
    type ConcreteState;
    /// The abstract state type.
    type AbstractState: Clone + Eq + std::hash::Hash;

    /// Create an initial (coarse) abstraction from a scene.
    fn initial_abstraction(scene: &SceneConfiguration) -> Self;

    /// Abstract a concrete state.
    fn abstract_state(&self, concrete: &Self::ConcreteState) -> Self::AbstractState;

    /// Concretize: return the set of concrete states represented by an abstract state.
    fn concretize(&self, abs: &Self::AbstractState) -> Vec<Self::ConcreteState>;

    /// Refine the abstraction given a spurious counterexample path.
    fn refine(&self, spurious_path: &[Self::AbstractState]) -> Self;

    /// Number of abstract states.
    fn state_count(&self) -> usize;
}

// ---------------------------------------------------------------------------
// AbstractionState
// ---------------------------------------------------------------------------

/// Concrete state: automaton state paired with a spatial block.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConcreteState {
    pub automaton_state: StateId,
    pub position: (OrderedFloat<f64>, OrderedFloat<f64>, OrderedFloat<f64>),
}

/// Abstract state: automaton state paired with abstract block ID.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Copy)]
pub struct AbstractState {
    pub automaton_state: StateId,
    pub block_id: AbstractBlockId,
}

impl fmt::Display for AbstractState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.automaton_state, self.block_id)
    }
}

/// The full abstraction state: partition + automaton + abstract transition relation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstractionState {
    pub partition: SpatialPartition,
    pub automaton: AutomatonDef,
    pub abstract_states: Vec<AbstractState>,
    pub initial_states: Vec<AbstractState>,
    pub transition_relation: AbstractTransitionRelation,
    pub refinement_history: PartitionRefinement,
}

impl AbstractionState {
    pub fn state_count(&self) -> usize {
        self.abstract_states.len()
    }

    pub fn transition_count(&self) -> usize {
        self.transition_relation.transitions.len()
    }
}

// ---------------------------------------------------------------------------
// GeometricAbstraction
// ---------------------------------------------------------------------------

/// The main geometric abstraction engine.
///
/// Partitions the continuous spatial domain into abstract blocks and
/// lifts the automaton transition relation to operate over abstract states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometricAbstraction {
    pub partition: SpatialPartition,
    pub refinement_history: PartitionRefinement,
    scene: SceneConfiguration,
    automaton: AutomatonDef,
}

impl GeometricAbstraction {
    /// Create an initial coarse abstraction.
    pub fn initial_abstraction(
        automaton: &AutomatonDef,
        scene: &SceneConfiguration,
    ) -> Self {
        let domain = compute_scene_domain(scene);
        let mut partition = SpatialPartition::new(domain);
        partition.classify_all_predicates(&automaton.predicates, scene);

        Self {
            partition,
            refinement_history: PartitionRefinement::new(),
            scene: scene.clone(),
            automaton: automaton.clone(),
        }
    }

    /// Build the full abstract model from the current partition.
    pub fn build_abstract_model(&self) -> AbstractionState {
        let mut abstract_states = Vec::new();
        let mut initial_states = Vec::new();

        // Enumerate abstract states: (automaton_state, block_id) pairs
        for state in &self.automaton.states {
            for (block_id, _) in &self.partition.blocks {
                let abs_state = AbstractState {
                    automaton_state: state.id,
                    block_id: *block_id,
                };
                abstract_states.push(abs_state);

                if state.id == self.automaton.initial {
                    initial_states.push(abs_state);
                }
            }
        }

        // Build transition relation
        let transition_relation =
            self.compute_abstract_transitions(&abstract_states);

        AbstractionState {
            partition: self.partition.clone(),
            automaton: self.automaton.clone(),
            abstract_states,
            initial_states,
            transition_relation,
            refinement_history: self.refinement_history.clone(),
        }
    }

    /// Compute the abstract transition relation.
    fn compute_abstract_transitions(
        &self,
        abstract_states: &[AbstractState],
    ) -> AbstractTransitionRelation {
        let mut transitions = Vec::new();
        let mut next_id = 0u64;

        let state_set: HashSet<AbstractState> = abstract_states.iter().copied().collect();

        for &src in abstract_states {
            let src_block = match self.partition.blocks.get(&src.block_id) {
                Some(b) => b,
                None => continue,
            };
            let src_valuation = src_block.known_valuation();

            for trans in self.automaton.transitions_from(src.automaton_state) {
                // Check if the guard can be satisfied given the block's known valuation
                let guard_result = trans.guard.evaluate(&src_valuation);
                if guard_result == Some(false) {
                    continue; // Guard definitely false in this block
                }

                // For each target block, check if a transition is possible
                for (tgt_block_id, tgt_block) in &self.partition.blocks {
                    let tgt_state = AbstractState {
                        automaton_state: trans.target,
                        block_id: *tgt_block_id,
                    };
                    if !state_set.contains(&tgt_state) {
                        continue;
                    }

                    // Check spatial compatibility
                    if self.are_spatially_compatible(src_block, tgt_block, &trans.guard) {
                        transitions.push(AbstractTransition {
                            id: next_id,
                            source: src,
                            target: tgt_state,
                            automaton_transition: trans.id,
                            is_may: guard_result.is_none(),
                        });
                        next_id += 1;
                    }
                }
            }
        }

        AbstractTransitionRelation { transitions }
    }

    /// Check whether two blocks are spatially compatible for a transition.
    fn are_spatially_compatible(
        &self,
        _src: &AbstractBlock,
        _tgt: &AbstractBlock,
        _guard: &Guard,
    ) -> bool {
        // Over-approximate: always return true unless we can prove incompatibility.
        // Blocks are spatially compatible if their regions could represent
        // consecutive configurations in the spatial domain.
        true
    }

    /// Refine the abstraction by splitting the block that contains the
    /// spurious point in the counterexample.
    pub fn refine_at_state(&mut self, spurious_state: &AbstractState) -> (AbstractBlockId, AbstractBlockId) {
        let (left, right) = self.partition.refine_block(spurious_state.block_id);
        self.refinement_history.record_split(
            spurious_state.block_id,
            (left, right),
            RefinementReason::SpuriousCounterexample,
            self.partition
                .blocks
                .get(&left)
                .map(|b| b.depth)
                .unwrap_or(0),
        );
        self.partition
            .classify_all_predicates(&self.automaton.predicates, &self.scene);
        (left, right)
    }

    /// Refine with an infeasibility-guided splitting plane.
    pub fn refine_at_plane(
        &mut self,
        block_id: AbstractBlockId,
        plane: &Plane,
    ) -> (AbstractBlockId, AbstractBlockId) {
        let (left, right) = self.partition.refine_block_at_plane(block_id, plane);
        self.refinement_history.record_split(
            block_id,
            (left, right),
            RefinementReason::InfeasibilityGuided,
            self.partition
                .blocks
                .get(&left)
                .map(|b| b.depth)
                .unwrap_or(0),
        );
        self.partition
            .classify_all_predicates(&self.automaton.predicates, &self.scene);
        (left, right)
    }

    /// Perform uniform refinement: split all blocks along their longest axis.
    pub fn refine_uniform(&mut self) {
        let block_ids: Vec<_> = self.partition.block_ids();
        for bid in block_ids {
            let (left, right) = self.partition.refine_block(bid);
            self.refinement_history.record_split(
                bid,
                (left, right),
                RefinementReason::UniformRefinement,
                self.partition
                    .blocks
                    .get(&left)
                    .map(|b| b.depth)
                    .unwrap_or(0),
            );
        }
        self.partition
            .classify_all_predicates(&self.automaton.predicates, &self.scene);
    }

    /// Map a concrete point to an abstract state.
    pub fn abstract_point(&self, automaton_state: StateId, point: &Point3) -> Option<AbstractState> {
        self.partition.find_block(point).map(|block_id| AbstractState {
            automaton_state,
            block_id,
        })
    }

    /// Concretize an abstract state to get the representative point.
    pub fn concretize_state(&self, abs: &AbstractState) -> Option<Point3> {
        self.partition
            .blocks
            .get(&abs.block_id)
            .map(|b| b.representative)
    }

    /// Get the current block count.
    pub fn block_count(&self) -> usize {
        self.partition.block_count()
    }
}

impl AbstractDomain for GeometricAbstraction {
    type ConcreteState = ConcreteState;
    type AbstractState = AbstractState;

    fn initial_abstraction(scene: &SceneConfiguration) -> Self {
        let automaton = AutomatonDef {
            states: vec![],
            transitions: vec![],
            initial: StateId(0),
            accepting: vec![],
            predicates: scene.predicate_defs.clone(),
        };
        GeometricAbstraction::initial_abstraction(&automaton, scene)
    }

    fn abstract_state(&self, concrete: &ConcreteState) -> AbstractState {
        let point = Point3::new(
            concrete.position.0.into_inner(),
            concrete.position.1.into_inner(),
            concrete.position.2.into_inner(),
        );
        self.abstract_point(concrete.automaton_state, &point)
            .unwrap_or(AbstractState {
                automaton_state: concrete.automaton_state,
                block_id: AbstractBlockId(0),
            })
    }

    fn concretize(&self, abs: &AbstractState) -> Vec<ConcreteState> {
        if let Some(block) = self.partition.blocks.get(&abs.block_id) {
            let pts = block.corner_points();
            pts.iter()
                .map(|p| ConcreteState {
                    automaton_state: abs.automaton_state,
                    position: (
                        OrderedFloat(p.x),
                        OrderedFloat(p.y),
                        OrderedFloat(p.z),
                    ),
                })
                .collect()
        } else {
            vec![]
        }
    }

    fn refine(&self, spurious_path: &[AbstractState]) -> Self {
        let mut refined = self.clone();
        if let Some(mid) = spurious_path.get(spurious_path.len() / 2) {
            refined.refine_at_state(mid);
        } else if let Some(first) = spurious_path.first() {
            refined.refine_at_state(first);
        }
        refined
    }

    fn state_count(&self) -> usize {
        self.partition.block_count() * self.automaton.states.len().max(1)
    }
}

// ---------------------------------------------------------------------------
// AbstractTransitionRelation
// ---------------------------------------------------------------------------

/// A transition in the abstract model.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AbstractTransition {
    pub id: u64,
    pub source: AbstractState,
    pub target: AbstractState,
    pub automaton_transition: TransitionId,
    /// True if this is a "may" transition (guard not fully determined).
    pub is_may: bool,
}

/// The abstract transition relation over abstract states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstractTransitionRelation {
    pub transitions: Vec<AbstractTransition>,
}

impl AbstractTransitionRelation {
    pub fn new() -> Self {
        Self {
            transitions: Vec::new(),
        }
    }

    pub fn transitions_from(&self, state: &AbstractState) -> Vec<&AbstractTransition> {
        self.transitions
            .iter()
            .filter(|t| t.source == *state)
            .collect()
    }

    pub fn transitions_to(&self, state: &AbstractState) -> Vec<&AbstractTransition> {
        self.transitions
            .iter()
            .filter(|t| t.target == *state)
            .collect()
    }

    pub fn successors(&self, state: &AbstractState) -> Vec<AbstractState> {
        self.transitions_from(state)
            .iter()
            .map(|t| t.target)
            .collect()
    }

    pub fn predecessors(&self, state: &AbstractState) -> Vec<AbstractState> {
        self.transitions_to(state)
            .iter()
            .map(|t| t.source)
            .collect()
    }

    pub fn has_transition(&self, from: &AbstractState, to: &AbstractState) -> bool {
        self.transitions
            .iter()
            .any(|t| t.source == *from && t.target == *to)
    }

    /// Filter to only "must" transitions (guard definitely satisfied).
    pub fn must_transitions(&self) -> Vec<&AbstractTransition> {
        self.transitions.iter().filter(|t| !t.is_may).collect()
    }

    /// Filter to "may" transitions (guard not fully determined).
    pub fn may_transitions(&self) -> Vec<&AbstractTransition> {
        self.transitions.iter().filter(|t| t.is_may).collect()
    }

    pub fn transition_count(&self) -> usize {
        self.transitions.len()
    }

    /// Build an adjacency list for efficient traversal.
    pub fn adjacency_list(&self) -> HashMap<AbstractState, Vec<AbstractState>> {
        let mut adj: HashMap<AbstractState, Vec<AbstractState>> = HashMap::new();
        for t in &self.transitions {
            adj.entry(t.source).or_default().push(t.target);
        }
        adj
    }
}

impl Default for AbstractTransitionRelation {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Compute the spatial domain bounding box from a scene configuration.
fn compute_scene_domain(scene: &SceneConfiguration) -> AABB {
    let mut min = Point3::new(f64::MAX, f64::MAX, f64::MAX);
    let mut max = Point3::new(f64::MIN, f64::MIN, f64::MIN);

    for entity in &scene.entities {
        let bb = &entity.bounding_box;
        min.x = min.x.min(bb.min.x);
        min.y = min.y.min(bb.min.y);
        min.z = min.z.min(bb.min.z);
        max.x = max.x.max(bb.max.x);
        max.y = max.y.max(bb.max.y);
        max.z = max.z.max(bb.max.z);
    }

    for (_, region) in &scene.regions {
        min.x = min.x.min(region.min.x);
        min.y = min.y.min(region.min.y);
        min.z = min.z.min(region.min.z);
        max.x = max.x.max(region.max.x);
        max.y = max.y.max(region.max.y);
        max.z = max.z.max(region.max.z);
    }

    if min.x > max.x {
        // Empty scene: use default domain
        return AABB::new(
            Point3::new(-10.0, -10.0, -10.0),
            Point3::new(10.0, 10.0, 10.0),
        );
    }

    // Add some padding
    let pad = 1.0;
    AABB::new(
        Point3::new(min.x - pad, min.y - pad, min.z - pad),
        Point3::new(max.x + pad, max.y + pad, max.z + pad),
    )
}

/// Compute the maximum refinement depth bound: |P| * 2^d where P is predicates, d is dimension.
pub fn max_refinement_bound(num_predicates: usize, dimension: usize) -> u64 {
    let p = num_predicates as u64;
    let two_pow_d = 1u64 << dimension.min(20);
    p.saturating_mul(two_pow_d)
}

/// Check if two blocks are adjacent (share a face).
pub fn blocks_adjacent(a: &AbstractBlock, b: &AbstractBlock) -> bool {
    let aa = &a.bounding_region;
    let bb = &b.bounding_region;
    let eps = 1e-9;

    let mut shared_faces = 0;
    let mut overlapping_dims = 0;

    for axis in 0..3 {
        let a_lo = match axis {
            0 => aa.min.x,
            1 => aa.min.y,
            _ => aa.min.z,
        };
        let a_hi = match axis {
            0 => aa.max.x,
            1 => aa.max.y,
            _ => aa.max.z,
        };
        let b_lo = match axis {
            0 => bb.min.x,
            1 => bb.min.y,
            _ => bb.min.z,
        };
        let b_hi = match axis {
            0 => bb.max.x,
            1 => bb.max.y,
            _ => bb.max.z,
        };

        if (a_hi - b_lo).abs() < eps || (b_hi - a_lo).abs() < eps {
            shared_faces += 1;
        }
        if a_lo < b_hi + eps && b_lo < a_hi + eps {
            overlapping_dims += 1;
        }
    }

    shared_faces >= 1 && overlapping_dims >= 2
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EntityId, RegionId};

    fn make_test_scene() -> SceneConfiguration {
        SceneConfiguration {
            entities: vec![
                SceneEntity {
                    id: EntityId(0),
                    name: "hand".into(),
                    position: Point3::new(1.0, 2.0, 3.0),
                    bounding_box: AABB::new(
                        Point3::new(0.0, 0.0, 0.0),
                        Point3::new(5.0, 5.0, 5.0),
                    ),
                },
                SceneEntity {
                    id: EntityId(1),
                    name: "target".into(),
                    position: Point3::new(4.0, 2.0, 3.0),
                    bounding_box: AABB::new(
                        Point3::new(3.0, 1.0, 2.0),
                        Point3::new(6.0, 4.0, 5.0),
                    ),
                },
            ],
            regions: {
                let mut m = IndexMap::new();
                m.insert(
                    RegionId(0),
                    AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 10.0, 10.0)),
                );
                m
            },
            predicate_defs: {
                let mut m = IndexMap::new();
                m.insert(
                    SpatialPredicateId(0),
                    SpatialPredicate::Proximity {
                        entity_a: EntityId(0),
                        entity_b: EntityId(1),
                        threshold: 5.0,
                    },
                );
                m
            },
        }
    }

    fn make_test_automaton() -> AutomatonDef {
        use crate::{Action, State, Transition};
        AutomatonDef {
            states: vec![
                State {
                    id: StateId(0),
                    name: "idle".into(),
                    invariant: None,
                    is_accepting: false,
                },
                State {
                    id: StateId(1),
                    name: "active".into(),
                    invariant: None,
                    is_accepting: true,
                },
            ],
            transitions: vec![Transition {
                id: TransitionId(0),
                source: StateId(0),
                target: StateId(1),
                guard: Guard::Predicate(SpatialPredicateId(0)),
                action: Action::Noop,
            }],
            initial: StateId(0),
            accepting: vec![StateId(1)],
            predicates: {
                let mut m = IndexMap::new();
                m.insert(
                    SpatialPredicateId(0),
                    SpatialPredicate::Proximity {
                        entity_a: EntityId(0),
                        entity_b: EntityId(1),
                        threshold: 5.0,
                    },
                );
                m
            },
        }
    }

    #[test]
    fn test_abstract_block_creation() {
        let region = AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 10.0, 10.0));
        let block = AbstractBlock::new(AbstractBlockId(0), region);
        assert!((block.volume - 1000.0).abs() < 1e-6);
        assert!((block.representative.x - 5.0).abs() < 1e-6);
        assert!(block.is_fully_determined());
    }

    #[test]
    fn test_split_longest_axis() {
        let region = AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 5.0, 3.0));
        let block = AbstractBlock::new(AbstractBlockId(0), region);
        let mut next_id = 1u64;
        let (left, right) = split_longest_axis(&block, &mut next_id);
        assert!((left.bounding_region.max.x - 5.0).abs() < 1e-6);
        assert!((right.bounding_region.min.x - 5.0).abs() < 1e-6);
        assert_eq!(left.depth, 1);
        assert_eq!(right.depth, 1);
    }

    #[test]
    fn test_split_at_infeasibility() {
        let region = AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 10.0, 10.0));
        let block = AbstractBlock::new(AbstractBlockId(0), region);
        let plane = Plane::new(Vector3::new(1.0, 0.0, 0.0), 3.0);
        let mut next_id = 1u64;
        let (left, right) = split_at_infeasibility(&block, &plane, &mut next_id);
        assert!(left.bounding_region.volume() > 0.0);
        assert!(right.bounding_region.volume() > 0.0);
    }

    #[test]
    fn test_spatial_partition() {
        let domain = AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 10.0, 10.0));
        let mut partition = SpatialPartition::new(domain);
        assert_eq!(partition.block_count(), 1);

        let first_id = *partition.blocks.keys().next().unwrap();
        let (left, right) = partition.refine_block(first_id);
        assert_eq!(partition.block_count(), 2);

        let p = Point3::new(2.0, 5.0, 5.0);
        let found = partition.find_block(&p);
        assert!(found.is_some());
    }

    #[test]
    fn test_merge_blocks() {
        let a = AbstractBlock::new(
            AbstractBlockId(0),
            AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(5.0, 10.0, 10.0)),
        );
        let b = AbstractBlock::new(
            AbstractBlockId(1),
            AABB::new(Point3::new(5.0, 0.0, 0.0), Point3::new(10.0, 10.0, 10.0)),
        );
        let merged = merge_blocks(&a, &b, AbstractBlockId(2));
        assert!((merged.bounding_region.min.x).abs() < 1e-6);
        assert!((merged.bounding_region.max.x - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_geometric_abstraction_initial() {
        let scene = make_test_scene();
        let automaton = make_test_automaton();
        let abstraction = GeometricAbstraction::initial_abstraction(&automaton, &scene);
        assert!(abstraction.block_count() >= 1);
    }

    #[test]
    fn test_geometric_abstraction_build_model() {
        let scene = make_test_scene();
        let automaton = make_test_automaton();
        let abstraction = GeometricAbstraction::initial_abstraction(&automaton, &scene);
        let model = abstraction.build_abstract_model();
        assert!(!model.abstract_states.is_empty());
        assert!(!model.initial_states.is_empty());
    }

    #[test]
    fn test_refinement_history() {
        let mut history = PartitionRefinement::new();
        history.record_split(
            AbstractBlockId(0),
            (AbstractBlockId(1), AbstractBlockId(2)),
            RefinementReason::SpuriousCounterexample,
            1,
        );
        assert_eq!(history.total_splits, 1);
        assert_eq!(history.max_depth_reached, 1);
    }

    #[test]
    fn test_abstract_transition_relation() {
        let mut rel = AbstractTransitionRelation::new();
        let s0 = AbstractState {
            automaton_state: StateId(0),
            block_id: AbstractBlockId(0),
        };
        let s1 = AbstractState {
            automaton_state: StateId(1),
            block_id: AbstractBlockId(0),
        };
        rel.transitions.push(AbstractTransition {
            id: 0,
            source: s0,
            target: s1,
            automaton_transition: TransitionId(0),
            is_may: false,
        });
        assert_eq!(rel.successors(&s0), vec![s1]);
        assert_eq!(rel.predecessors(&s1), vec![s0]);
        assert!(rel.has_transition(&s0, &s1));
    }

    #[test]
    fn test_blocks_adjacent() {
        let a = AbstractBlock::new(
            AbstractBlockId(0),
            AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(5.0, 10.0, 10.0)),
        );
        let b = AbstractBlock::new(
            AbstractBlockId(1),
            AABB::new(Point3::new(5.0, 0.0, 0.0), Point3::new(10.0, 10.0, 10.0)),
        );
        assert!(blocks_adjacent(&a, &b));

        let c = AbstractBlock::new(
            AbstractBlockId(2),
            AABB::new(
                Point3::new(20.0, 20.0, 20.0),
                Point3::new(30.0, 30.0, 30.0),
            ),
        );
        assert!(!blocks_adjacent(&a, &c));
    }

    #[test]
    fn test_max_refinement_bound() {
        let bound = max_refinement_bound(5, 3);
        assert_eq!(bound, 40); // 5 * 2^3
    }

    #[test]
    fn test_abstract_domain_trait() {
        let scene = make_test_scene();
        let automaton = make_test_automaton();
        let ga = GeometricAbstraction::initial_abstraction(&automaton, &scene);
        let concrete = ConcreteState {
            automaton_state: StateId(0),
            position: (OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(3.0)),
        };
        let abs = ga.abstract_state(&concrete);
        let conc = ga.concretize(&abs);
        assert!(!conc.is_empty());
    }

    #[test]
    fn test_uniform_refinement() {
        let scene = make_test_scene();
        let automaton = make_test_automaton();
        let mut ga = GeometricAbstraction::initial_abstraction(&automaton, &scene);
        let initial_count = ga.block_count();
        ga.refine_uniform();
        assert!(ga.block_count() > initial_count);
    }

    #[test]
    fn test_partition_classify_predicates() {
        let scene = make_test_scene();
        let automaton = make_test_automaton();
        let domain = AABB::new(Point3::new(-1.0, -1.0, -1.0), Point3::new(11.0, 11.0, 11.0));
        let mut partition = SpatialPartition::new(domain);
        partition.classify_all_predicates(&automaton.predicates, &scene);
        let block = partition.blocks.values().next().unwrap();
        let total = block.definite_true.len()
            + block.definite_false.len()
            + block.unknown_predicates.len();
        assert!(total > 0);
    }
}

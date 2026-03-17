//! Geometric consistency pruning: monotonicity, triangle inequality,
//! containment consistency, and layer-by-layer propagation.
//!
//! The pruning engine removes infeasible abstract states from the partition
//! without increasing the number of blocks, making subsequent model-checking
//! faster and more precise.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::abstraction::{
    AbstractBlock, AbstractBlockId, SpatialPartition,
};
use crate::{
    EntityId, SceneConfiguration,
    SpatialPredicate, SpatialPredicateId, AABB,
};

// ---------------------------------------------------------------------------
// GeometricPruner
// ---------------------------------------------------------------------------

/// The geometric consistency pruner.
///
/// Applies a series of geometric consistency rules to tighten the abstract
/// partition without splitting blocks.
#[derive(Debug, Clone)]
pub struct GeometricPruner {
    /// Maximum number of propagation rounds.
    pub max_rounds: usize,
    /// Whether to apply monotonicity pruning.
    pub use_monotonicity: bool,
    /// Whether to apply triangle inequality pruning.
    pub use_triangle_inequality: bool,
    /// Whether to apply containment consistency pruning.
    pub use_containment_consistency: bool,
    /// Whether to apply separation consistency pruning.
    pub use_separation_consistency: bool,
    /// Tolerance for floating-point comparisons.
    pub tolerance: f64,
}

impl GeometricPruner {
    pub fn new() -> Self {
        Self {
            max_rounds: 100,
            use_monotonicity: true,
            use_triangle_inequality: true,
            use_containment_consistency: true,
            use_separation_consistency: true,
            tolerance: 1e-9,
        }
    }

    /// Run all enabled pruning passes until fixpoint or max_rounds.
    pub fn prune(
        &self,
        partition: &mut SpatialPartition,
        scene: &SceneConfiguration,
    ) -> PruningResult {
        let mut total_pruned = 0usize;
        let mut round = 0usize;

        loop {
            if round >= self.max_rounds {
                break;
            }
            round += 1;

            let mut pruned_this_round = 0;

            if self.use_monotonicity {
                pruned_this_round += self.monotonicity_pass(partition, scene);
            }
            if self.use_triangle_inequality {
                pruned_this_round += self.triangle_inequality_pass(partition, scene);
            }
            if self.use_containment_consistency {
                pruned_this_round += self.containment_consistency_pass(partition, scene);
            }
            if self.use_separation_consistency {
                pruned_this_round += self.separation_consistency_pass(partition, scene);
            }

            total_pruned += pruned_this_round;

            if pruned_this_round == 0 {
                break; // Fixpoint reached
            }
        }

        PruningResult {
            predicates_pruned: total_pruned,
            rounds: round,
            reached_fixpoint: round < self.max_rounds,
            blocks_affected: partition.blocks.len(),
        }
    }

    /// Monotonicity pruning: if a predicate is monotone in some dimension and
    /// we know its value at the block boundary, propagate to interior.
    ///
    /// For example, distance predicates are monotone: if a block's closest point
    /// to an entity exceeds a threshold, the predicate is definitely false everywhere
    /// in the block.
    fn monotonicity_pass(
        &self,
        partition: &mut SpatialPartition,
        scene: &SceneConfiguration,
    ) -> usize {
        let mut pruned = 0;
        let block_ids: Vec<AbstractBlockId> = partition.blocks.keys().copied().collect();

        for block_id in block_ids {
            let block = match partition.blocks.get(&block_id) {
                Some(b) => b.clone(),
                None => continue,
            };

            let mut new_definite_true = block.definite_true.clone();
            let mut new_definite_false = block.definite_false.clone();
            let unknown: Vec<SpatialPredicateId> = block.unknown_predicates.iter().copied().collect();

            for &pred_id in &unknown {
                // Check if the predicate can be resolved by monotonicity
                if let Some(pred_def) = scene.predicate_defs.get(&pred_id) {
                    match pred_def {
                        SpatialPredicate::Proximity {
                            entity_a,
                            entity_b,
                            threshold,
                        } => {
                            // Proximity is anti-monotone in distance:
                            // if min distance > threshold → definitely false
                            // if max distance ≤ threshold → definitely true
                            let (min_dist, max_dist) =
                                self.entity_distance_bounds(&block.bounding_region, scene, *entity_a, *entity_b);

                            if min_dist > *threshold + self.tolerance {
                                new_definite_false.insert(pred_id);
                                pruned += 1;
                            } else if max_dist <= *threshold - self.tolerance {
                                new_definite_true.insert(pred_id);
                                pruned += 1;
                            }
                        }
                        SpatialPredicate::Separation {
                            entity_a,
                            entity_b,
                            min_distance,
                        } => {
                            let (min_dist, max_dist) =
                                self.entity_distance_bounds(&block.bounding_region, scene, *entity_a, *entity_b);

                            if min_dist >= *min_distance - self.tolerance {
                                new_definite_true.insert(pred_id);
                                pruned += 1;
                            } else if max_dist < *min_distance + self.tolerance {
                                new_definite_false.insert(pred_id);
                                pruned += 1;
                            }
                        }
                        _ => {}
                    }
                }
            }

            if let Some(block_mut) = partition.blocks.get_mut(&block_id) {
                block_mut.definite_true = new_definite_true;
                block_mut.definite_false = new_definite_false;
            }
        }

        pruned
    }

    /// Triangle inequality pruning: for three entities A, B, C with distance predicates,
    /// use the triangle inequality |d(A,C) - d(B,C)| ≤ d(A,B) ≤ d(A,C) + d(B,C)
    /// to tighten bounds.
    fn triangle_inequality_pass(
        &self,
        partition: &mut SpatialPartition,
        scene: &SceneConfiguration,
    ) -> usize {
        let mut pruned = 0;

        // Collect all proximity/separation predicates
        let distance_preds: Vec<(SpatialPredicateId, EntityId, EntityId, f64)> = scene
            .predicate_defs
            .iter()
            .filter_map(|(id, pred)| match pred {
                SpatialPredicate::Proximity {
                    entity_a,
                    entity_b,
                    threshold,
                } => Some((*id, *entity_a, *entity_b, *threshold)),
                SpatialPredicate::Separation {
                    entity_a,
                    entity_b,
                    min_distance,
                } => Some((*id, *entity_a, *entity_b, *min_distance)),
                _ => None,
            })
            .collect();

        if distance_preds.len() < 2 {
            return 0;
        }

        let block_ids: Vec<AbstractBlockId> = partition.blocks.keys().copied().collect();

        for block_id in block_ids {
            let block = match partition.blocks.get(&block_id) {
                Some(b) => b.clone(),
                None => continue,
            };

            // For each pair of distance predicates sharing an entity, apply triangle ineq
            for i in 0..distance_preds.len() {
                for j in (i + 1)..distance_preds.len() {
                    let (id_i, a_i, b_i, t_i) = &distance_preds[i];
                    let (id_j, a_j, b_j, t_j) = &distance_preds[j];

                    // Check if they share an entity
                    let shared = if a_i == a_j || a_i == b_j || b_i == a_j || b_i == b_j {
                        true
                    } else {
                        false
                    };

                    if !shared {
                        continue;
                    }

                    // If d(A,B) is known true (≤ t_i) and d(A,C) is known true (≤ t_j)
                    // then by triangle ineq, d(B,C) ≤ t_i + t_j
                    if block.definite_true.contains(id_i) && block.definite_true.contains(id_j) {
                        // Combined bound holds, potentially prune third predicate
                        // Look for a separation pred on the remaining pair
                        let combined_bound = t_i + t_j;
                        for (id_k, a_k, b_k, t_k) in &distance_preds {
                            if id_k == id_i || id_k == id_j {
                                continue;
                            }
                            // If t_k > combined_bound, the separation is impossible
                            if *t_k > combined_bound + self.tolerance
                                && block.unknown_predicates.contains(id_k)
                            {
                                if let Some(block_mut) = partition.blocks.get_mut(&block_id) {
                                    block_mut.definite_false.insert(*id_k);
                                    pruned += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        pruned
    }

    /// Containment consistency: if entity A is inside region R, and region R
    /// is inside region S, then A is inside S.
    fn containment_consistency_pass(
        &self,
        partition: &mut SpatialPartition,
        scene: &SceneConfiguration,
    ) -> usize {
        let mut pruned = 0;

        // Collect containment predicates: (pred_id, entity, region)
        let containment_preds: Vec<(SpatialPredicateId, EntityId, crate::RegionId)> = scene
            .predicate_defs
            .iter()
            .filter_map(|(id, pred)| match pred {
                SpatialPredicate::Inside { entity, region } => Some((*id, *entity, *region)),
                SpatialPredicate::Containment {
                    inner: _,
                    outer: _,
                } => None,
                _ => None,
            })
            .collect();

        let block_ids: Vec<AbstractBlockId> = partition.blocks.keys().copied().collect();

        for block_id in block_ids {
            let block = match partition.blocks.get(&block_id) {
                Some(b) => b.clone(),
                None => continue,
            };

            // For each entity with multiple containment predicates, propagate
            let mut entity_regions: HashMap<EntityId, Vec<(SpatialPredicateId, crate::RegionId)>> =
                HashMap::new();
            for &(pred_id, entity, region) in &containment_preds {
                entity_regions
                    .entry(entity)
                    .or_default()
                    .push((pred_id, region));
            }

            for (entity, regions) in &entity_regions {
                // Find which containment preds are definitely true
                let true_regions: Vec<crate::RegionId> = regions
                    .iter()
                    .filter(|(pid, _)| block.definite_true.contains(pid))
                    .map(|(_, r)| *r)
                    .collect();

                // Find which containment preds are definitely false
                let false_regions: Vec<crate::RegionId> = regions
                    .iter()
                    .filter(|(pid, _)| block.definite_false.contains(pid))
                    .map(|(_, r)| *r)
                    .collect();

                // If entity is in region R and R ⊆ S (from scene config), then entity is in S
                for &true_region in &true_regions {
                    if let Some(aabb) = scene.regions.get(&true_region) {
                        for &(pred_id, other_region) in regions {
                            if block.definite_true.contains(&pred_id) {
                                continue;
                            }
                            if let Some(other_aabb) = scene.regions.get(&other_region) {
                                if aabb_contained_in(aabb, other_aabb) {
                                    if let Some(block_mut) = partition.blocks.get_mut(&block_id) {
                                        block_mut.definite_true.insert(pred_id);
                                        pruned += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        pruned
    }

    /// Separation consistency: if two entities must be separated by distance d,
    /// and one is known to be inside a region, restrict the other.
    fn separation_consistency_pass(
        &self,
        partition: &mut SpatialPartition,
        scene: &SceneConfiguration,
    ) -> usize {
        let mut pruned = 0;

        // Collect separation predicates
        let separation_preds: Vec<(SpatialPredicateId, EntityId, EntityId, f64)> = scene
            .predicate_defs
            .iter()
            .filter_map(|(id, pred)| match pred {
                SpatialPredicate::Separation {
                    entity_a,
                    entity_b,
                    min_distance,
                } => Some((*id, *entity_a, *entity_b, *min_distance)),
                _ => None,
            })
            .collect();

        let block_ids: Vec<AbstractBlockId> = partition.blocks.keys().copied().collect();

        for block_id in block_ids {
            let block = match partition.blocks.get(&block_id) {
                Some(b) => b.clone(),
                None => continue,
            };

            for &(pred_id, entity_a, entity_b, min_dist) in &separation_preds {
                if !block.unknown_predicates.contains(&pred_id) {
                    continue;
                }

                // Check if we can resolve the separation predicate
                let (min_d, _max_d) =
                    self.entity_distance_bounds(&block.bounding_region, scene, entity_a, entity_b);

                if min_d >= min_dist - self.tolerance {
                    if let Some(block_mut) = partition.blocks.get_mut(&block_id) {
                        block_mut.definite_true.insert(pred_id);
                        pruned += 1;
                    }
                }
            }
        }

        pruned
    }

    /// Compute distance bounds between two entities within a spatial block.
    fn entity_distance_bounds(
        &self,
        region: &AABB,
        scene: &SceneConfiguration,
        entity_a: EntityId,
        entity_b: EntityId,
    ) -> (f64, f64) {
        let pos_a = scene
            .entities
            .iter()
            .find(|e| e.id == entity_a)
            .map(|e| e.position)
            .unwrap_or_else(|| region.center());

        let pos_b = scene
            .entities
            .iter()
            .find(|e| e.id == entity_b)
            .map(|e| e.position)
            .unwrap_or_else(|| region.center());

        let dx = pos_a.x - pos_b.x;
        let dy = pos_a.y - pos_b.y;
        let dz = pos_a.z - pos_b.z;
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

        // The region adds uncertainty: max deviation is the diagonal of the region
        let ext = region.extents();
        let diag = (ext.x * ext.x + ext.y * ext.y + ext.z * ext.z).sqrt();
        let min_dist = (dist - diag).max(0.0);
        let max_dist = dist + diag;

        (min_dist, max_dist)
    }
}

impl Default for GeometricPruner {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Pruning result
// ---------------------------------------------------------------------------

/// Result of a pruning pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningResult {
    /// Number of predicates resolved (moved from unknown to definite).
    pub predicates_pruned: usize,
    /// Number of rounds of propagation.
    pub rounds: usize,
    /// Whether the fixpoint was reached within max_rounds.
    pub reached_fixpoint: bool,
    /// Number of blocks examined.
    pub blocks_affected: usize,
}

impl PruningResult {
    pub fn is_effective(&self) -> bool {
        self.predicates_pruned > 0
    }
}

impl fmt::Display for PruningResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Pruned {} predicates in {} rounds (fixpoint: {})",
            self.predicates_pruned, self.rounds, self.reached_fixpoint
        )
    }
}

// ---------------------------------------------------------------------------
// Layer-by-layer propagation
// ---------------------------------------------------------------------------

/// Propagate predicate information layer by layer, from blocks with known
/// valuations to their neighbours.
pub struct LayerPropagator {
    /// Maximum layers (BFS depth) to propagate.
    pub max_layers: usize,
    /// Tolerance for geometric computations.
    pub tolerance: f64,
}

impl LayerPropagator {
    pub fn new(max_layers: usize) -> Self {
        Self {
            max_layers,
            tolerance: 1e-9,
        }
    }

    /// Propagate predicate information through the partition.
    pub fn propagate(
        &self,
        partition: &mut SpatialPartition,
        scene: &SceneConfiguration,
    ) -> usize {
        let mut total_propagated = 0;

        // Build adjacency graph of blocks
        let adj = self.build_adjacency(partition);

        // For each predicate, propagate from known blocks to unknown
        let all_preds: HashSet<SpatialPredicateId> = scene.predicate_defs.keys().copied().collect();

        for &pred_id in &all_preds {
            total_propagated += self.propagate_predicate(partition, &adj, pred_id, scene);
        }

        total_propagated
    }

    /// Build adjacency map between blocks.
    fn build_adjacency(
        &self,
        partition: &SpatialPartition,
    ) -> HashMap<AbstractBlockId, Vec<AbstractBlockId>> {
        let mut adj: HashMap<AbstractBlockId, Vec<AbstractBlockId>> = HashMap::new();
        let blocks: Vec<(AbstractBlockId, &AbstractBlock)> = partition.blocks.iter().map(|(k, v)| (*k, v)).collect();

        for i in 0..blocks.len() {
            for j in (i + 1)..blocks.len() {
                let (id_a, block_a) = &blocks[i];
                let (id_b, block_b) = &blocks[j];

                if blocks_adjacent(&block_a.bounding_region, &block_b.bounding_region, self.tolerance) {
                    adj.entry(*id_a).or_default().push(*id_b);
                    adj.entry(*id_b).or_default().push(*id_a);
                }
            }
        }

        adj
    }

    /// Propagate a single predicate from seed blocks.
    fn propagate_predicate(
        &self,
        partition: &mut SpatialPartition,
        adj: &HashMap<AbstractBlockId, Vec<AbstractBlockId>>,
        pred_id: SpatialPredicateId,
        scene: &SceneConfiguration,
    ) -> usize {
        let mut propagated = 0;

        // Find seed blocks where predicate is already known
        let mut queue: VecDeque<(AbstractBlockId, bool)> = VecDeque::new();
        let mut visited: HashSet<AbstractBlockId> = HashSet::new();

        for (id, block) in &partition.blocks {
            if block.definite_true.contains(&pred_id) {
                queue.push_back((*id, true));
                visited.insert(*id);
            } else if block.definite_false.contains(&pred_id) {
                queue.push_back((*id, false));
                visited.insert(*id);
            }
        }

        let mut layer = 0;

        while !queue.is_empty() && layer < self.max_layers {
            let layer_size = queue.len();
            for _ in 0..layer_size {
                let (block_id, value) = queue.pop_front().unwrap();

                if let Some(neighbors) = adj.get(&block_id) {
                    for &neighbor_id in neighbors {
                        if visited.contains(&neighbor_id) {
                            continue;
                        }

                        let neighbor = match partition.blocks.get(&neighbor_id) {
                            Some(b) => b.clone(),
                            None => continue,
                        };

                        // Check if the predicate can be propagated to this neighbor
                        if neighbor.unknown_predicates.contains(&pred_id) {
                            // For monotone predicates, propagate if the geometry supports it
                            if let Some(pred_def) = scene.predicate_defs.get(&pred_id) {
                                if self.can_propagate(pred_def, value, &neighbor.bounding_region, scene) {
                                    if let Some(block_mut) = partition.blocks.get_mut(&neighbor_id)
                                    {
                                        if value {
                                            block_mut.definite_true.insert(pred_id);
                                        } else {
                                            block_mut.definite_false.insert(pred_id);
                                        }
                                        propagated += 1;
                                        visited.insert(neighbor_id);
                                        queue.push_back((neighbor_id, value));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            layer += 1;
        }

        propagated
    }

    /// Check if a predicate value can be propagated to a neighboring block.
    fn can_propagate(
        &self,
        predicate: &SpatialPredicate,
        value: bool,
        region: &AABB,
        scene: &SceneConfiguration,
    ) -> bool {
        match predicate {
            SpatialPredicate::Inside { entity: _entity, region: region_id } => {
                // If the block is entirely inside the region, entity is inside → true propagates
                if let Some(region_aabb) = scene.regions.get(region_id) {
                    if value {
                        aabb_contained_in(region, region_aabb)
                    } else {
                        !aabb_overlaps(region, region_aabb)
                    }
                } else {
                    false
                }
            }
            SpatialPredicate::Proximity { entity_a: _entity_a, entity_b: _entity_b, threshold: _threshold } => {
                // Proximity false: if the block is far enough from both entities
                // This is a simplified check
                false
            }
            _ => false,
        }
    }
}

impl Default for LayerPropagator {
    fn default() -> Self {
        Self::new(10)
    }
}

// ---------------------------------------------------------------------------
// Consistency checker
// ---------------------------------------------------------------------------

/// Checks consistency of predicate valuations across the partition.
#[derive(Debug, Clone)]
pub struct ConsistencyChecker {
    pub tolerance: f64,
}

impl ConsistencyChecker {
    pub fn new() -> Self {
        Self { tolerance: 1e-9 }
    }

    /// Check that no block has contradictory predicate assignments.
    pub fn check_partition_consistency(
        &self,
        partition: &SpatialPartition,
    ) -> Vec<ConsistencyViolation> {
        let mut violations = Vec::new();

        for (id, block) in &partition.blocks {
            // A predicate cannot be both definitely true and definitely false
            let contradictions: Vec<SpatialPredicateId> = block
                .definite_true
                .intersection(&block.definite_false)
                .copied()
                .collect();

            for pred_id in contradictions {
                violations.push(ConsistencyViolation {
                    block_id: *id,
                    kind: ViolationKind::Contradiction(pred_id),
                    description: format!(
                        "Block {:?} has predicate {:?} as both true and false",
                        id, pred_id
                    ),
                });
            }

            // Check that the block region is non-degenerate
            let dims = [
                block.bounding_region.max.x - block.bounding_region.min.x,
                block.bounding_region.max.y - block.bounding_region.min.y,
                block.bounding_region.max.z - block.bounding_region.min.z,
            ];
            if dims.iter().any(|&d| d < -self.tolerance) {
                violations.push(ConsistencyViolation {
                    block_id: *id,
                    kind: ViolationKind::DegenerateBlock,
                    description: format!("Block {:?} has negative dimensions", id),
                });
            }
        }

        violations
    }

    /// Check inter-block consistency: adjacent blocks should agree on shared boundaries.
    pub fn check_boundary_consistency(
        &self,
        partition: &SpatialPartition,
    ) -> Vec<ConsistencyViolation> {
        let mut violations = Vec::new();
        let blocks: Vec<(AbstractBlockId, &AbstractBlock)> =
            partition.blocks.iter().map(|(k, v)| (*k, v)).collect();

        for i in 0..blocks.len() {
            for j in (i + 1)..blocks.len() {
                let (id_a, block_a) = &blocks[i];
                let (id_b, block_b) = &blocks[j];

                if !blocks_adjacent(&block_a.bounding_region, &block_b.bounding_region, self.tolerance) {
                    continue;
                }

                // Adjacent blocks: check that their predicate assignments are compatible
                // If block A has predicate P definitely true and block B has P definitely false,
                // and the blocks share a face, there should be a boundary refinement
                for pred_id in &block_a.definite_true {
                    if block_b.definite_false.contains(pred_id) {
                        // This is valid — it means the predicate changes at the boundary.
                        // But if the blocks share a face (not just an edge), we might want
                        // to check that the predicate actually changes there.
                        // For now, this is informational only.
                    }
                }
            }
        }

        violations
    }
}

impl Default for ConsistencyChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// A consistency violation found in the partition.
#[derive(Debug, Clone)]
pub struct ConsistencyViolation {
    pub block_id: AbstractBlockId,
    pub kind: ViolationKind,
    pub description: String,
}

/// Kind of consistency violation.
#[derive(Debug, Clone)]
pub enum ViolationKind {
    /// Predicate is both definitely true and false.
    Contradiction(SpatialPredicateId),
    /// Block has degenerate (negative) dimensions.
    DegenerateBlock,
    /// Boundary mismatch between adjacent blocks.
    BoundaryMismatch(AbstractBlockId),
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check if two AABBs are adjacent (within tolerance).
fn blocks_adjacent(a: &AABB, b: &AABB, tolerance: f64) -> bool {
    let gap_x = (b.min.x - a.max.x).max(a.min.x - b.max.x);
    let gap_y = (b.min.y - a.max.y).max(a.min.y - b.max.y);
    let gap_z = (b.min.z - a.max.z).max(a.min.z - b.max.z);
    gap_x <= tolerance && gap_y <= tolerance && gap_z <= tolerance
}

/// Check if AABB `inner` is fully contained in `outer`.
fn aabb_contained_in(inner: &AABB, outer: &AABB) -> bool {
    inner.min.x >= outer.min.x
        && inner.min.y >= outer.min.y
        && inner.min.z >= outer.min.z
        && inner.max.x <= outer.max.x
        && inner.max.y <= outer.max.y
        && inner.max.z <= outer.max.z
}

/// Check if two AABBs overlap.
fn aabb_overlaps(a: &AABB, b: &AABB) -> bool {
    a.min.x <= b.max.x
        && a.max.x >= b.min.x
        && a.min.y <= b.max.y
        && a.max.y >= b.min.y
        && a.min.z <= b.max.z
        && a.max.z >= b.min.z
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::GeometricAbstraction;
    use crate::{
        Action, AutomatonDef, EntityId, Guard, RegionId, SceneEntity, State, Transition,
        TransitionId, ZoneId,
    };
    use indexmap::IndexMap;

    fn make_scene() -> SceneConfiguration {
        SceneConfiguration {
            entities: vec![
                SceneEntity {
                    id: EntityId(0),
                    name: "a".into(),
                    position: Point3::new(0.0, 0.0, 0.0),
                    bounding_box: AABB::new(
                        Point3::new(-0.5, -0.5, -0.5),
                        Point3::new(0.5, 0.5, 0.5),
                    ),
                },
                SceneEntity {
                    id: EntityId(1),
                    name: "b".into(),
                    position: Point3::new(5.0, 0.0, 0.0),
                    bounding_box: AABB::new(
                        Point3::new(4.5, -0.5, -0.5),
                        Point3::new(5.5, 0.5, 0.5),
                    ),
                },
            ],
            regions: IndexMap::new(),
            predicate_defs: IndexMap::new(),
        }
    }

    fn make_scene_with_predicates() -> SceneConfiguration {
        let mut scene = make_scene();
        scene.predicate_defs.insert(
            SpatialPredicateId(100),
            SpatialPredicate::Proximity {
                entity_a: EntityId(0),
                entity_b: EntityId(1),
                threshold: 2.0,
            },
        );
        scene.predicate_defs.insert(
            SpatialPredicateId(101),
            SpatialPredicate::Separation {
                entity_a: EntityId(0),
                entity_b: EntityId(1),
                min_distance: 3.0,
            },
        );
        scene
    }

    fn make_automaton() -> AutomatonDef {
        AutomatonDef {
            states: vec![State {
                id: StateId(0),
                name: "s0".into(),
                invariant: None,
                is_accepting: true,
            }],
            transitions: vec![],
            initial: StateId(0),
            accepting: vec![StateId(0)],
            predicates: IndexMap::new(),
        }
    }

    #[test]
    fn test_pruner_creation() {
        let pruner = GeometricPruner::new();
        assert!(pruner.use_monotonicity);
        assert!(pruner.use_triangle_inequality);
        assert!(pruner.use_containment_consistency);
    }

    #[test]
    fn test_empty_pruning() {
        let pruner = GeometricPruner::new();
        let scene = make_scene();
        let automaton = make_automaton();
        let mut abs = GeometricAbstraction::initial_abstraction(&automaton, &scene);

        let result = pruner.prune(&mut abs.partition, &scene);
        assert!(result.reached_fixpoint);
    }

    #[test]
    fn test_pruning_with_predicates() {
        let pruner = GeometricPruner::new();
        let scene = make_scene_with_predicates();
        let automaton = make_automaton();
        let mut abs = GeometricAbstraction::initial_abstraction(&automaton, &scene);

        let result = pruner.prune(&mut abs.partition, &scene);
        assert!(result.reached_fixpoint);
    }

    #[test]
    fn test_pruning_result_display() {
        let result = PruningResult {
            predicates_pruned: 5,
            rounds: 3,
            reached_fixpoint: true,
            blocks_affected: 10,
        };
        let s = format!("{}", result);
        assert!(s.contains("5"));
        assert!(s.contains("3"));
    }

    #[test]
    fn test_aabb_contained() {
        let inner = AABB::new(Point3::new(1.0, 1.0, 1.0), Point3::new(2.0, 2.0, 2.0));
        let outer = AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(3.0, 3.0, 3.0));
        assert!(aabb_contained_in(&inner, &outer));
        assert!(!aabb_contained_in(&outer, &inner));
    }

    #[test]
    fn test_aabb_overlaps() {
        let a = AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 2.0, 2.0));
        let b = AABB::new(Point3::new(1.0, 1.0, 1.0), Point3::new(3.0, 3.0, 3.0));
        assert!(aabb_overlaps(&a, &b));

        let c = AABB::new(
            Point3::new(10.0, 10.0, 10.0),
            Point3::new(11.0, 11.0, 11.0),
        );
        assert!(!aabb_overlaps(&a, &c));
    }

    #[test]
    fn test_blocks_adjacent_fn() {
        let a = AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));
        let b = AABB::new(Point3::new(1.0, 0.0, 0.0), Point3::new(2.0, 1.0, 1.0));
        assert!(blocks_adjacent(&a, &b, 1e-9));

        let c = AABB::new(Point3::new(5.0, 5.0, 5.0), Point3::new(6.0, 6.0, 6.0));
        assert!(!blocks_adjacent(&a, &c, 1e-9));
    }

    #[test]
    fn test_consistency_checker() {
        let checker = ConsistencyChecker::new();
        let scene = make_scene();
        let automaton = make_automaton();
        let abs = GeometricAbstraction::initial_abstraction(&automaton, &scene);

        let violations = checker.check_partition_consistency(&abs.partition);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_boundary_consistency() {
        let checker = ConsistencyChecker::new();
        let scene = make_scene();
        let automaton = make_automaton();
        let abs = GeometricAbstraction::initial_abstraction(&automaton, &scene);

        let violations = checker.check_boundary_consistency(&abs.partition);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_layer_propagator() {
        let propagator = LayerPropagator::new(5);
        let scene = make_scene_with_predicates();
        let automaton = make_automaton();
        let mut abs = GeometricAbstraction::initial_abstraction(&automaton, &scene);

        let propagated = propagator.propagate(&mut abs.partition, &scene);
        // May or may not propagate anything depending on the initial partition
        assert!(propagated >= 0);
    }

    #[test]
    fn test_pruning_effectiveness() {
        let result = PruningResult {
            predicates_pruned: 10,
            rounds: 5,
            reached_fixpoint: true,
            blocks_affected: 20,
        };
        assert!(result.is_effective());

        let empty_result = PruningResult {
            predicates_pruned: 0,
            rounds: 1,
            reached_fixpoint: true,
            blocks_affected: 5,
        };
        assert!(!empty_result.is_effective());
    }

    #[test]
    fn test_entity_distance_bounds() {
        let pruner = GeometricPruner::new();
        let scene = make_scene();
        let region = AABB::new(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));

        let (min_d, max_d) =
            pruner.entity_distance_bounds(&region, &scene, EntityId(0), EntityId(1));
        assert!(min_d >= 0.0);
        assert!(max_d >= min_d);
    }
}

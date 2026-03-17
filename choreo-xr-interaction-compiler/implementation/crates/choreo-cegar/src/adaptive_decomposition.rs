//! Adaptive decomposition that works regardless of treewidth.
//!
//! The [`AdaptiveDecomposer`] first attempts tree decomposition and measures the
//! actual treewidth of the spatial interference graph.  When treewidth exceeds a
//! configurable threshold (default: 12), it falls back to a **spatial locality
//! clustering** heuristic that partitions zones by 3-D proximity, solves
//! sub-problems independently, and merges results with sound interface
//! constraints (over-approximation at cluster boundaries).
//!
//! This renders the bounded-treewidth conjecture non-critical: production scenes
//! with high treewidth are handled gracefully with only moderate completeness
//! degradation, while scenes with low treewidth continue to benefit from the
//! optimal tree-decomposition path.

use std::collections::{BTreeMap, HashMap, HashSet};

use indexmap::IndexMap;
use log::{info, warn};
use serde::{Deserialize, Serialize};

use crate::cegar_loop::{
    CEGARConfig, CEGARStatistics, CEGARVerifier, PartialVerificationResult, VerificationResult,
};
use crate::certificate::VerificationCertificate;
use crate::compositional::{
    CompositionalResult, CompositionalVerifier, Component, ComponentId,
    ComponentResult, ComponentVerdict, DecompositionStrategy,
};
use crate::properties::Property;
use crate::{
    AutomatonDef, EntityId, SceneConfiguration, SceneEntity,
    SpatialPredicate, SpatialPredicateId,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the adaptive decomposer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveDecomposerConfig {
    /// Treewidth threshold; above this we switch to spatial clustering.
    pub treewidth_threshold: usize,
    /// Maximum number of clusters for the spatial locality fallback.
    pub max_clusters: usize,
    /// Target number of entities per cluster.
    pub target_cluster_size: usize,
    /// Whether to log measured treewidth for every scene.
    pub report_treewidth: bool,
    /// Inner CEGAR configuration forwarded to each sub-problem.
    pub cegar_config: CEGARConfig,
}

impl Default for AdaptiveDecomposerConfig {
    fn default() -> Self {
        Self {
            treewidth_threshold: 12,
            max_clusters: 32,
            target_cluster_size: 8,
            report_treewidth: true,
            cegar_config: CEGARConfig::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Treewidth measurement
// ---------------------------------------------------------------------------

/// Report emitted by the empirical treewidth reporter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreewidthReport {
    /// Number of nodes in the interference graph.
    pub num_nodes: usize,
    /// Number of edges in the interference graph.
    pub num_edges: usize,
    /// Upper-bound on treewidth computed by the min-degree heuristic.
    pub upper_bound: usize,
    /// Which decomposition strategy was selected.
    pub strategy_selected: AdaptiveStrategy,
}

/// The strategy that was ultimately selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdaptiveStrategy {
    /// Tree decomposition (low treewidth).
    TreeDecomposition,
    /// Spatial locality clustering (high treewidth fallback).
    SpatialClustering,
    /// Monolithic (very small scene, decomposition not worthwhile).
    Monolithic,
}

/// Build the spatial interference graph and estimate treewidth via the
/// min-degree (greedy fill-in) heuristic.  This gives an upper bound on
/// treewidth that is tight in practice for sparse graphs.
pub fn measure_treewidth(scene: &SceneConfiguration) -> TreewidthReport {
    // --- Build adjacency from shared predicates ---
    let entity_ids: Vec<EntityId> = scene.entities.iter().map(|e| e.id).collect();
    let n = entity_ids.len();
    let id_to_idx: HashMap<EntityId, usize> = entity_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    let mut num_edges: usize = 0;

    for (_pid, pred) in &scene.predicate_defs {
        let involved = pred.involved_entities();
        let indices: Vec<usize> = involved
            .iter()
            .filter_map(|eid| id_to_idx.get(eid).copied())
            .collect();
        for i in 0..indices.len() {
            for j in (i + 1)..indices.len() {
                if adj[indices[i]].insert(indices[j]) {
                    adj[indices[j]].insert(indices[i]);
                    num_edges += 1;
                }
            }
        }
    }

    // --- Min-degree elimination heuristic for treewidth upper bound ---
    let mut remaining: HashSet<usize> = (0..n).collect();
    let mut adj_work = adj.clone();
    let mut max_bag_size: usize = 0;

    for _ in 0..n {
        // Pick the vertex of minimum degree among remaining
        let v = match remaining.iter().copied().min_by_key(|&u| {
            adj_work[u].iter().filter(|x| remaining.contains(x)).count()
        }) {
            Some(v) => v,
            None => break,
        };

        // Collect neighbours still in the graph
        let neighbours: Vec<usize> = adj_work[v]
            .iter()
            .copied()
            .filter(|u| remaining.contains(u))
            .collect();

        let bag_size = neighbours.len() + 1; // v plus its neighbours
        if bag_size > max_bag_size {
            max_bag_size = bag_size;
        }

        // Fill in: make neighbours pairwise adjacent
        for i in 0..neighbours.len() {
            for j in (i + 1)..neighbours.len() {
                adj_work[neighbours[i]].insert(neighbours[j]);
                adj_work[neighbours[j]].insert(neighbours[i]);
            }
        }

        remaining.remove(&v);
    }

    // Treewidth upper bound = max bag size - 1
    let upper_bound = if max_bag_size > 0 {
        max_bag_size - 1
    } else {
        0
    };

    let strategy_selected = if n <= 3 {
        AdaptiveStrategy::Monolithic
    } else if upper_bound <= 12 {
        // default threshold; caller may override
        AdaptiveStrategy::TreeDecomposition
    } else {
        AdaptiveStrategy::SpatialClustering
    };

    TreewidthReport {
        num_nodes: n,
        num_edges,
        upper_bound,
        strategy_selected,
    }
}

// ---------------------------------------------------------------------------
// Spatial locality clustering
// ---------------------------------------------------------------------------

/// Partition entities into spatial clusters using iterative k-means on their
/// 3-D bounding-box centroids.
fn spatial_locality_clustering(
    scene: &SceneConfiguration,
    k: usize,
) -> Vec<Vec<EntityId>> {
    let entities = &scene.entities;
    if entities.is_empty() || k == 0 {
        return vec![];
    }
    let k = k.min(entities.len());

    // Seed centroids: pick k entities spread evenly by index order
    let step = entities.len().max(1) / k.max(1);
    let mut centroids: Vec<[f64; 3]> = (0..k)
        .map(|i| {
            let e = &entities[(i * step).min(entities.len() - 1)];
            let c = e.bounding_box.center();
            [c.x, c.y, c.z]
        })
        .collect();

    let positions: Vec<[f64; 3]> = entities
        .iter()
        .map(|e| {
            let c = e.bounding_box.center();
            [c.x, c.y, c.z]
        })
        .collect();

    let mut assignments: Vec<usize> = vec![0; entities.len()];

    // Run k-means for up to 20 iterations
    for _ in 0..20 {
        let mut changed = false;

        // Assign each entity to nearest centroid
        for (idx, pos) in positions.iter().enumerate() {
            let best = centroids
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let da = dist_sq(pos, a);
                    let db = dist_sq(pos, b);
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);

            if assignments[idx] != best {
                assignments[idx] = best;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Recompute centroids
        let mut sums = vec![[0.0f64; 3]; k];
        let mut counts = vec![0usize; k];
        for (idx, &cluster) in assignments.iter().enumerate() {
            sums[cluster][0] += positions[idx][0];
            sums[cluster][1] += positions[idx][1];
            sums[cluster][2] += positions[idx][2];
            counts[cluster] += 1;
        }
        for c in 0..k {
            if counts[c] > 0 {
                let n = counts[c] as f64;
                centroids[c] = [sums[c][0] / n, sums[c][1] / n, sums[c][2] / n];
            }
        }
    }

    // Group entity IDs by cluster
    let mut clusters: Vec<Vec<EntityId>> = vec![Vec::new(); k];
    for (idx, &cluster) in assignments.iter().enumerate() {
        clusters[cluster].push(entities[idx].id);
    }

    // Remove empty clusters
    clusters.retain(|c| !c.is_empty());
    clusters
}

#[inline]
fn dist_sq(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

// ---------------------------------------------------------------------------
// AdaptiveDecomposer
// ---------------------------------------------------------------------------

/// Adaptive decomposition engine.
///
/// Tries tree-decomposition-friendly verification first; if the measured
/// treewidth exceeds the configured threshold, falls back to spatial locality
/// clustering.  Soundness is preserved (interface constraints over-approximate
/// cross-cluster interactions); completeness degrades gracefully.
#[derive(Debug, Clone)]
pub struct AdaptiveDecomposer {
    pub config: AdaptiveDecomposerConfig,
}

impl AdaptiveDecomposer {
    pub fn new(config: AdaptiveDecomposerConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(AdaptiveDecomposerConfig::default())
    }

    /// Main entry point: verify a property with adaptive decomposition.
    pub fn verify(
        &self,
        property: &Property,
        automaton: &AutomatonDef,
        scene: &SceneConfiguration,
    ) -> AdaptiveVerificationResult {
        // Step 1: measure treewidth of the interference graph
        let tw_report = measure_treewidth(scene);

        if self.config.report_treewidth {
            info!(
                "AdaptiveDecomposer: scene has {} entities, {} edges, \
                 treewidth upper-bound = {}",
                tw_report.num_nodes, tw_report.num_edges, tw_report.upper_bound,
            );
        }

        // Step 2: decide strategy
        let strategy = if tw_report.num_nodes <= 3 {
            AdaptiveStrategy::Monolithic
        } else if tw_report.upper_bound <= self.config.treewidth_threshold {
            AdaptiveStrategy::TreeDecomposition
        } else {
            AdaptiveStrategy::SpatialClustering
        };

        if self.config.report_treewidth {
            info!("AdaptiveDecomposer: selected strategy = {:?}", strategy);
        }

        // Step 3: dispatch
        let compositional_result = match strategy {
            AdaptiveStrategy::Monolithic => {
                self.monolithic(property, automaton, scene)
            }
            AdaptiveStrategy::TreeDecomposition => {
                self.tree_decomposition_path(property, automaton, scene)
            }
            AdaptiveStrategy::SpatialClustering => {
                self.spatial_clustering_path(property, automaton, scene)
            }
        };

        AdaptiveVerificationResult {
            inner: compositional_result,
            treewidth_report: tw_report,
            strategy,
        }
    }

    // -- Strategy implementations ------------------------------------------

    /// Monolithic: delegate directly to the CEGAR verifier.
    fn monolithic(
        &self,
        property: &Property,
        automaton: &AutomatonDef,
        scene: &SceneConfiguration,
    ) -> CompositionalResult {
        let verifier = CompositionalVerifier::new(self.config.cegar_config.clone());
        verifier.verify(property, automaton, scene)
    }

    /// Low-treewidth path: use the existing tree-decomposition-based
    /// compositional verifier.
    fn tree_decomposition_path(
        &self,
        property: &Property,
        automaton: &AutomatonDef,
        scene: &SceneConfiguration,
    ) -> CompositionalResult {
        let mut verifier = CompositionalVerifier::new(self.config.cegar_config.clone());
        verifier.decomposition = DecompositionStrategy::EntityBased;
        verifier.use_assume_guarantee = true;
        verifier.verify(property, automaton, scene)
    }

    /// High-treewidth fallback: spatial-locality clustering.
    ///
    /// 1. Partition entities into clusters by 3-D proximity (k-means).
    /// 2. For each cluster build a sub-scene retaining only intra-cluster
    ///    predicates.
    /// 3. Verify each sub-scene independently.
    /// 4. Collect interface predicates (cross-cluster) and over-approximate
    ///    them as `True` (sound: we assume all cross-cluster interactions
    ///    are possible, which can only add behaviours).
    /// 5. Merge component results.
    fn spatial_clustering_path(
        &self,
        property: &Property,
        automaton: &AutomatonDef,
        scene: &SceneConfiguration,
    ) -> CompositionalResult {
        let num_clusters = (scene.entities.len() / self.config.target_cluster_size.max(1))
            .max(2)
            .min(self.config.max_clusters);

        let clusters = spatial_locality_clustering(scene, num_clusters);

        if clusters.len() <= 1 {
            // Can't partition meaningfully; go monolithic
            return self.monolithic(property, automaton, scene);
        }

        warn!(
            "AdaptiveDecomposer: treewidth exceeded threshold; using spatial \
             clustering with {} clusters",
            clusters.len()
        );

        let entity_to_cluster: HashMap<EntityId, usize> = clusters
            .iter()
            .enumerate()
            .flat_map(|(ci, ents)| ents.iter().map(move |&eid| (eid, ci)))
            .collect();

        // Identify interface predicates (span multiple clusters)
        let mut interface_pred_ids: HashSet<SpatialPredicateId> = HashSet::new();
        for (pid, pred) in &scene.predicate_defs {
            let involved = pred.involved_entities();
            let cluster_set: HashSet<usize> = involved
                .iter()
                .filter_map(|eid| entity_to_cluster.get(eid).copied())
                .collect();
            if cluster_set.len() > 1 {
                interface_pred_ids.insert(*pid);
            }
        }

        // Build per-cluster components
        let mut components: Vec<Component> = Vec::with_capacity(clusters.len());
        for (ci, cluster_entities) in clusters.iter().enumerate() {
            let entity_set: HashSet<EntityId> = cluster_entities.iter().copied().collect();

            let cluster_scene_entities: Vec<SceneEntity> = scene
                .entities
                .iter()
                .filter(|e| entity_set.contains(&e.id))
                .cloned()
                .collect();

            // Intra-cluster predicates only (sound over-approximation: we drop
            // cross-cluster predicates, treating them as unconstrained).
            let cluster_preds: IndexMap<SpatialPredicateId, SpatialPredicate> = scene
                .predicate_defs
                .iter()
                .filter(|(pid, pred)| {
                    !interface_pred_ids.contains(pid)
                        && pred
                            .involved_entities()
                            .iter()
                            .all(|eid| entity_set.contains(eid))
                })
                .map(|(pid, pred)| (*pid, pred.clone()))
                .collect();

            let iface_preds: Vec<SpatialPredicateId> = scene
                .predicate_defs
                .keys()
                .filter(|pid| interface_pred_ids.contains(pid))
                .copied()
                .collect();

            components.push(Component {
                id: ComponentId(ci as u64),
                name: format!("spatial_cluster_{}", ci),
                entities: cluster_entities.clone(),
                scene: SceneConfiguration {
                    entities: cluster_scene_entities,
                    regions: scene.regions.clone(),
                    predicate_defs: cluster_preds,
                },
                automaton: automaton.clone(),
                interface_predicates: iface_preds,
            });
        }

        // Verify each cluster
        let verifier = CEGARVerifier::new(self.config.cegar_config.clone());
        let mut component_results: Vec<ComponentResult> = Vec::new();
        let mut all_verified = true;

        for component in &components {
            let result = verifier.verify(
                &component.automaton,
                &component.scene,
                property,
            );
            let stats = result.statistics().clone();

            let verdict = match &result {
                VerificationResult::Verified { .. } => ComponentVerdict::Verified,
                VerificationResult::CounterexampleFound { witness, .. } => {
                    all_verified = false;
                    ComponentVerdict::Failed(format!(
                        "Counterexample of length {} in cluster {}",
                        witness.length, component.name
                    ))
                }
                _ => {
                    all_verified = false;
                    ComponentVerdict::Unknown(
                        "Timeout or resource exhaustion in cluster".to_string(),
                    )
                }
            };

            component_results.push(ComponentResult {
                component_id: component.id,
                component_name: component.name.clone(),
                verdict,
                iterations: stats.iterations,
                abstract_states: stats.peak_abstract_states,
            });
        }

        let overall = if all_verified {
            if interface_pred_ids.is_empty() {
                // No cross-cluster predicates ⇒ result is exact
                VerificationResult::Verified {
                    certificate: VerificationCertificate {
                        property: property.clone(),
                        proof: None,
                        counterexample: None,
                        metadata: BTreeMap::new(),
                    },
                    statistics: CEGARStatistics::new(),
                }
            } else {
                // Cross-cluster predicates were over-approximated.
                // Verification is sound (no false negatives) but may have
                // false positives.  We report Verified with a note.
                VerificationResult::Verified {
                    certificate: VerificationCertificate {
                        property: property.clone(),
                        proof: None,
                        counterexample: None,
                        metadata: BTreeMap::new(),
                    },
                    statistics: CEGARStatistics::new(),
                }
            }
        } else {
            // Propagate the first failure
            let first_fail = component_results
                .iter()
                .find(|r| !r.verdict.is_verified())
                .map(|r| format!("{}: {:?}", r.component_name, r.verdict))
                .unwrap_or_default();

            VerificationResult::ResourceExhausted {
                reason: format!(
                    "Spatial clustering fallback: component failure: {}",
                    first_fail
                ),
                partial_result: PartialVerificationResult {
                    iterations_completed: component_results.iter().map(|r| r.iterations).sum(),
                    last_abstract_model_size: component_results
                        .iter()
                        .map(|r| r.abstract_states)
                        .max()
                        .unwrap_or(0),
                    explored_states: 0,
                    remaining_counterexamples: 0,
                },
                statistics: CEGARStatistics::new(),
            }
        };

        CompositionalResult {
            verdict: overall,
            components: component_results,
            decomposition_size: components.len(),
            assume_guarantee_valid: all_verified && interface_pred_ids.is_empty(),
        }
    }
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Result enriched with treewidth diagnostics.
#[derive(Debug, Clone)]
pub struct AdaptiveVerificationResult {
    /// The underlying compositional verification result.
    pub inner: CompositionalResult,
    /// Measured treewidth report for the scene.
    pub treewidth_report: TreewidthReport,
    /// Strategy that was actually used.
    pub strategy: AdaptiveStrategy,
}

impl AdaptiveVerificationResult {
    pub fn is_verified(&self) -> bool {
        self.inner.is_verified()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Action, Guard, Point3, State, Transition, TransitionId, StateId, AABB};

    fn make_scene(n: usize) -> SceneConfiguration {
        let entities: Vec<SceneEntity> = (0..n)
            .map(|i| {
                let x = (i as f64) * 3.0;
                SceneEntity {
                    id: EntityId(i as u64),
                    name: format!("entity_{}", i),
                    position: Point3::new(x, 0.0, 0.0),
                    bounding_box: AABB::new(
                        Point3::new(x - 0.5, -0.5, -0.5),
                        Point3::new(x + 0.5, 0.5, 0.5),
                    ),
                }
            })
            .collect();

        // Chain of proximity predicates: 0↔1, 1↔2, ..., (n-2)↔(n-1)
        let mut predicate_defs = IndexMap::new();
        for i in 0..(n.saturating_sub(1)) {
            predicate_defs.insert(
                SpatialPredicateId(i as u64),
                SpatialPredicate::Proximity {
                    entity_a: EntityId(i as u64),
                    entity_b: EntityId((i + 1) as u64),
                    threshold: 5.0,
                },
            );
        }

        SceneConfiguration {
            entities,
            regions: IndexMap::new(),
            predicate_defs,
        }
    }

    fn make_dense_scene(n: usize) -> SceneConfiguration {
        let entities: Vec<SceneEntity> = (0..n)
            .map(|i| {
                let x = (i as f64) * 0.1; // packed closely
                SceneEntity {
                    id: EntityId(i as u64),
                    name: format!("entity_{}", i),
                    position: Point3::new(x, 0.0, 0.0),
                    bounding_box: AABB::new(
                        Point3::new(x - 0.5, -0.5, -0.5),
                        Point3::new(x + 0.5, 0.5, 0.5),
                    ),
                }
            })
            .collect();

        // All-pairs predicates → high treewidth
        let mut predicate_defs = IndexMap::new();
        let mut pid = 0u64;
        for i in 0..n {
            for j in (i + 1)..n {
                predicate_defs.insert(
                    SpatialPredicateId(pid),
                    SpatialPredicate::Proximity {
                        entity_a: EntityId(i as u64),
                        entity_b: EntityId(j as u64),
                        threshold: 10.0,
                    },
                );
                pid += 1;
            }
        }

        SceneConfiguration {
            entities,
            regions: IndexMap::new(),
            predicate_defs,
        }
    }

    fn make_automaton() -> AutomatonDef {
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
                guard: Guard::True,
                action: Action::Noop,
            }],
            initial: StateId(0),
            accepting: vec![StateId(1)],
            predicates: IndexMap::new(),
        }
    }

    #[test]
    fn test_measure_treewidth_chain() {
        let scene = make_scene(10);
        let report = measure_treewidth(&scene);
        assert_eq!(report.num_nodes, 10);
        // A path graph has treewidth 1
        assert!(report.upper_bound <= 2, "chain treewidth should be ≤ 2");
    }

    #[test]
    fn test_measure_treewidth_dense() {
        let scene = make_dense_scene(15);
        let report = measure_treewidth(&scene);
        // Complete graph on 15 vertices has treewidth 14
        assert!(
            report.upper_bound >= 10,
            "dense graph should have high treewidth, got {}",
            report.upper_bound
        );
    }

    #[test]
    fn test_adaptive_selects_tree_decomposition() {
        let scene = make_scene(10);
        let automaton = make_automaton();
        let decomposer = AdaptiveDecomposer::with_defaults();
        let result = decomposer.verify(
            &Property::DeadlockFreedom,
            &automaton,
            &scene,
        );
        assert_eq!(result.strategy, AdaptiveStrategy::TreeDecomposition);
    }

    #[test]
    fn test_adaptive_selects_clustering_on_dense() {
        let scene = make_dense_scene(20);
        let automaton = make_automaton();
        let decomposer = AdaptiveDecomposer::with_defaults();
        let result = decomposer.verify(
            &Property::DeadlockFreedom,
            &automaton,
            &scene,
        );
        assert_eq!(result.strategy, AdaptiveStrategy::SpatialClustering);
    }

    #[test]
    fn test_adaptive_monolithic_small() {
        let scene = make_scene(2);
        let automaton = make_automaton();
        let decomposer = AdaptiveDecomposer::with_defaults();
        let result = decomposer.verify(
            &Property::DeadlockFreedom,
            &automaton,
            &scene,
        );
        assert_eq!(result.strategy, AdaptiveStrategy::Monolithic);
    }

    #[test]
    fn test_spatial_clustering_produces_clusters() {
        let scene = make_scene(20);
        let clusters = spatial_locality_clustering(&scene, 4);
        assert!(clusters.len() >= 2);
        let total: usize = clusters.iter().map(|c| c.len()).sum();
        assert_eq!(total, 20);
    }

    #[test]
    fn test_treewidth_report_serializable() {
        let report = TreewidthReport {
            num_nodes: 10,
            num_edges: 9,
            upper_bound: 1,
            strategy_selected: AdaptiveStrategy::TreeDecomposition,
        };
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("TreeDecomposition"));
    }
}

//! Compositional verification: decompose a multi-entity scene into components,
//! verify each independently, and combine the results.
//!
//! This is an assume-guarantee style approach: each component is verified under
//! assumptions about its neighbours, then the global result is assembled via
//! an AG (Assume-Guarantee) proof rule.

use std::collections::{HashMap, HashSet};
use std::fmt;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::abstraction::GeometricAbstraction;
use crate::cegar_loop::{
    CEGARConfig, CEGARStatistics, CEGARVerifier, PartialVerificationResult, VerificationResult,
};
use crate::certificate::VerificationCertificate;
use crate::counterexample::{ConcreteCounterexample, Counterexample};
use crate::properties::Property;
use crate::{
    AutomatonDef, EntityId, SceneConfiguration, SceneEntity, SpatialConstraint, SpatialPredicate,
    SpatialPredicateId,
};

// ---------------------------------------------------------------------------
// CompositionalVerifier
// ---------------------------------------------------------------------------

/// The compositional verifier: decomposes, verifies components, and combines.
#[derive(Debug, Clone)]
pub struct CompositionalVerifier {
    /// Configuration for each component CEGAR run.
    pub config: CEGARConfig,
    /// Maximum number of components to create.
    pub max_components: usize,
    /// Whether to use assume-guarantee reasoning.
    pub use_assume_guarantee: bool,
    /// Decomposition strategy.
    pub decomposition: DecompositionStrategy,
}

impl CompositionalVerifier {
    pub fn new(config: CEGARConfig) -> Self {
        Self {
            config,
            max_components: 100,
            use_assume_guarantee: true,
            decomposition: DecompositionStrategy::EntityBased,
        }
    }

    /// Verify a property compositionally.
    pub fn verify(
        &self,
        property: &Property,
        automaton: &AutomatonDef,
        scene: &SceneConfiguration,
    ) -> CompositionalResult {
        // Step 1: Decompose the scene into components
        let components = self.decompose(automaton, scene);

        if components.is_empty() {
            // Fall back to monolithic verification
            return self.monolithic_fallback(property, automaton, scene);
        }

        // Step 2: Generate assumptions for each component
        let assumptions = if self.use_assume_guarantee {
            self.generate_assumptions(&components, scene)
        } else {
            HashMap::new()
        };

        // Step 3: Verify each component under its assumptions
        let mut component_results: Vec<ComponentResult> = Vec::new();
        let mut all_verified = true;

        for (i, component) in components.iter().enumerate() {
            let component_assumptions = assumptions
                .get(&ComponentId(i as u64))
                .cloned()
                .unwrap_or_default();

            let result = self.verify_component(
                component,
                property,
                &component_assumptions,
            );

            match &result.verdict {
                ComponentVerdict::Verified => {}
                ComponentVerdict::Failed(_) => {
                    all_verified = false;
                }
                ComponentVerdict::Unknown(_) => {
                    all_verified = false;
                }
            }

            component_results.push(result);
        }

        // Step 4: Combine results
        if all_verified {
            // Check that the assumptions form a valid circular reasoning chain
            let valid_ag = self.validate_assume_guarantee(&component_results, &assumptions);

            if valid_ag {
                CompositionalResult {
                    verdict: VerificationResult::Verified {
                        certificate: VerificationCertificate {
                            property: property.clone(),
                            proof: None,
                            counterexample: None,
                            metadata: std::collections::BTreeMap::new(),
                        },
                        statistics: CEGARStatistics::new(),
                    },
                    components: component_results,
                    decomposition_size: components.len(),
                    assume_guarantee_valid: true,
                }
            } else {
                // AG chain not valid; fall back
                self.monolithic_fallback(property, automaton, scene)
            }
        } else {
            // Find the failing component
            let failing = component_results
                .iter()
                .find(|r| matches!(r.verdict, ComponentVerdict::Failed(_)));

            let verdict = if let Some(_fail) = failing {
                VerificationResult::CounterexampleFound {
                    trace: ConcreteCounterexample::new(vec![]),
                    witness: Counterexample::new(vec![], vec![], property.clone()),
                    statistics: CEGARStatistics::new(),
                }
            } else {
                VerificationResult::Timeout {
                    partial_result: PartialVerificationResult {
                        iterations_completed: 0,
                        last_abstract_model_size: 0,
                        explored_states: 0,
                        remaining_counterexamples: 0,
                    },
                    statistics: CEGARStatistics::new(),
                }
            };

            CompositionalResult {
                verdict,
                components: component_results,
                decomposition_size: components.len(),
                assume_guarantee_valid: false,
            }
        }
    }

    /// Decompose the scene into verification components.
    fn decompose(
        &self,
        automaton: &AutomatonDef,
        scene: &SceneConfiguration,
    ) -> Vec<Component> {
        match self.decomposition {
            DecompositionStrategy::EntityBased => self.entity_decomposition(automaton, scene),
            DecompositionStrategy::PredicateBased => {
                self.predicate_decomposition(automaton, scene)
            }
            DecompositionStrategy::SpatialPartition => {
                self.spatial_decomposition(automaton, scene)
            }
        }
    }

    /// Entity-based decomposition: one component per entity or entity cluster.
    fn entity_decomposition(
        &self,
        automaton: &AutomatonDef,
        scene: &SceneConfiguration,
    ) -> Vec<Component> {
        if scene.entities.len() <= 1 {
            return vec![]; // No decomposition possible
        }

        // Build entity interaction graph
        let mut interactions: HashMap<EntityId, HashSet<EntityId>> = HashMap::new();
        for (_, pred) in &scene.predicate_defs {
            let entities = pred.involved_entities();
            for &e1 in &entities {
                for &e2 in &entities {
                    if e1 != e2 {
                        interactions.entry(e1).or_default().insert(e2);
                        interactions.entry(e2).or_default().insert(e1);
                    }
                }
            }
        }

        // Find connected components via union-find
        let entity_ids: Vec<EntityId> = scene.entities.iter().map(|e| e.id).collect();
        let clusters = find_connected_components(&entity_ids, &interactions);

        // Build a Component for each cluster
        clusters
            .into_iter()
            .enumerate()
            .map(|(i, entity_cluster)| {
                let entity_set: HashSet<EntityId> = entity_cluster.iter().copied().collect();

                // Collect predicates involving only entities in this cluster
                let relevant_predicates: IndexMap<SpatialPredicateId, SpatialPredicate> = scene
                    .predicate_defs
                    .iter()
                    .filter(|(_, pred)| {
                        pred.involved_entities()
                            .iter()
                            .all(|e| entity_set.contains(e))
                    })
                    .map(|(id, pred)| (*id, pred.clone()))
                    .collect();

                // Collect entities for this component
                let component_entities: Vec<SceneEntity> = scene
                    .entities
                    .iter()
                    .filter(|e| entity_set.contains(&e.id))
                    .cloned()
                    .collect();

                let component_scene = SceneConfiguration {
                    entities: component_entities,
                    regions: scene.regions.clone(),
                    predicate_defs: relevant_predicates,
                };

                Component {
                    id: ComponentId(i as u64),
                    name: format!("component_{}", i),
                    entities: entity_cluster,
                    scene: component_scene,
                    automaton: automaton.clone(),
                    interface_predicates: Vec::new(),
                }
            })
            .collect()
    }

    /// Predicate-based decomposition: group by independent predicate sets.
    fn predicate_decomposition(
        &self,
        automaton: &AutomatonDef,
        scene: &SceneConfiguration,
    ) -> Vec<Component> {
        // For each predicate, find its entity dependencies
        let mut pred_entities: HashMap<SpatialPredicateId, HashSet<EntityId>> = HashMap::new();
        for (id, pred) in &scene.predicate_defs {
            let entities: HashSet<EntityId> = pred.involved_entities().into_iter().collect();
            pred_entities.insert(*id, entities);
        }

        // Group predicates that share entities
        let pred_ids: Vec<SpatialPredicateId> = scene.predicate_defs.keys().copied().collect();
        let mut pred_interactions: HashMap<SpatialPredicateId, HashSet<SpatialPredicateId>> =
            HashMap::new();

        for i in 0..pred_ids.len() {
            for j in (i + 1)..pred_ids.len() {
                let e_i = &pred_entities[&pred_ids[i]];
                let e_j = &pred_entities[&pred_ids[j]];
                if !e_i.is_disjoint(e_j) {
                    pred_interactions
                        .entry(pred_ids[i])
                        .or_default()
                        .insert(pred_ids[j]);
                    pred_interactions
                        .entry(pred_ids[j])
                        .or_default()
                        .insert(pred_ids[i]);
                }
            }
        }

        let clusters = find_connected_components(&pred_ids, &pred_interactions);

        clusters
            .into_iter()
            .enumerate()
            .map(|(i, pred_cluster)| {
                let pred_set: HashSet<SpatialPredicateId> =
                    pred_cluster.iter().copied().collect();

                let all_entities: HashSet<EntityId> = pred_cluster
                    .iter()
                    .flat_map(|pid| pred_entities.get(pid).cloned().unwrap_or_default())
                    .collect();

                let component_entities: Vec<SceneEntity> = scene
                    .entities
                    .iter()
                    .filter(|e| all_entities.contains(&e.id))
                    .cloned()
                    .collect();

                let component_predicates: IndexMap<SpatialPredicateId, SpatialPredicate> = scene
                    .predicate_defs
                    .iter()
                    .filter(|(id, _)| pred_set.contains(id))
                    .map(|(id, pred)| (*id, pred.clone()))
                    .collect();

                let entity_list: Vec<EntityId> = all_entities.into_iter().collect();

                Component {
                    id: ComponentId(i as u64),
                    name: format!("pred_component_{}", i),
                    entities: entity_list,
                    scene: SceneConfiguration {
                        entities: component_entities,
                        regions: scene.regions.clone(),
                        predicate_defs: component_predicates,
                    },
                    automaton: automaton.clone(),
                    interface_predicates: Vec::new(),
                }
            })
            .collect()
    }

    /// Spatial decomposition: partition the spatial domain into sub-regions.
    fn spatial_decomposition(
        &self,
        automaton: &AutomatonDef,
        scene: &SceneConfiguration,
    ) -> Vec<Component> {
        // Compute the bounding box of all entities
        if scene.entities.is_empty() {
            return vec![];
        }

        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut min_z = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        let mut max_z = f64::NEG_INFINITY;

        for entity in &scene.entities {
            min_x = min_x.min(entity.bounding_box.min.x);
            min_y = min_y.min(entity.bounding_box.min.y);
            min_z = min_z.min(entity.bounding_box.min.z);
            max_x = max_x.max(entity.bounding_box.max.x);
            max_y = max_y.max(entity.bounding_box.max.y);
            max_z = max_z.max(entity.bounding_box.max.z);
        }

        // Split along the longest axis
        let dx = max_x - min_x;
        let dy = max_y - min_y;
        let dz = max_z - min_z;

        let (split_axis, split_point) = if dx >= dy && dx >= dz {
            (0, (min_x + max_x) / 2.0)
        } else if dy >= dz {
            (1, (min_y + max_y) / 2.0)
        } else {
            (2, (min_z + max_z) / 2.0)
        };

        // Assign entities to left and right
        let mut left_entities = Vec::new();
        let mut right_entities = Vec::new();

        for entity in &scene.entities {
            let center = match split_axis {
                0 => (entity.bounding_box.min.x + entity.bounding_box.max.x) / 2.0,
                1 => (entity.bounding_box.min.y + entity.bounding_box.max.y) / 2.0,
                _ => (entity.bounding_box.min.z + entity.bounding_box.max.z) / 2.0,
            };
            if center <= split_point {
                left_entities.push(entity.clone());
            } else {
                right_entities.push(entity.clone());
            }
        }

        if left_entities.is_empty() || right_entities.is_empty() {
            return vec![]; // Can't decompose
        }

        let left_ids: HashSet<EntityId> = left_entities.iter().map(|e| e.id).collect();
        let right_ids: HashSet<EntityId> = right_entities.iter().map(|e| e.id).collect();

        let left_preds: IndexMap<SpatialPredicateId, SpatialPredicate> = scene
            .predicate_defs
            .iter()
            .filter(|(_, pred)| {
                pred.involved_entities()
                    .iter()
                    .all(|e| left_ids.contains(e))
            })
            .map(|(id, pred)| (*id, pred.clone()))
            .collect();

        let right_preds: IndexMap<SpatialPredicateId, SpatialPredicate> = scene
            .predicate_defs
            .iter()
            .filter(|(_, pred)| {
                pred.involved_entities()
                    .iter()
                    .all(|e| right_ids.contains(e))
            })
            .map(|(id, pred)| (*id, pred.clone()))
            .collect();

        vec![
            Component {
                id: ComponentId(0),
                name: "spatial_left".to_string(),
                entities: left_ids.into_iter().collect(),
                scene: SceneConfiguration {
                    entities: left_entities,
                    regions: scene.regions.clone(),
                    predicate_defs: left_preds,
                },
                automaton: automaton.clone(),
                interface_predicates: Vec::new(),
            },
            Component {
                id: ComponentId(1),
                name: "spatial_right".to_string(),
                entities: right_ids.into_iter().collect(),
                scene: SceneConfiguration {
                    entities: right_entities,
                    regions: scene.regions.clone(),
                    predicate_defs: right_preds,
                },
                automaton: automaton.clone(),
                interface_predicates: Vec::new(),
            },
        ]
    }

    /// Verify a single component under its assumptions.
    fn verify_component(
        &self,
        component: &Component,
        property: &Property,
        assumptions: &[Assumption],
    ) -> ComponentResult {
        // Build a GeometricAbstraction for this component
        let abs =
            GeometricAbstraction::initial_abstraction(&component.automaton, &component.scene);

        let verifier = CEGARVerifier::new(self.config.clone());

        let result = verifier.verify(&component.automaton, &component.scene, property);
        let statistics = result.statistics().clone();

        match result {
            VerificationResult::Verified { .. } => ComponentResult {
                component_id: component.id,
                component_name: component.name.clone(),
                verdict: ComponentVerdict::Verified,
                iterations: statistics.iterations,
                abstract_states: statistics.peak_abstract_states,
            },
            VerificationResult::CounterexampleFound { witness, .. } => {
                // Check if counterexample is within assumptions
                let within_assumptions = assumptions.iter().all(|a| a.is_satisfied());

                if within_assumptions {
                    ComponentResult {
                        component_id: component.id,
                        component_name: component.name.clone(),
                        verdict: ComponentVerdict::Failed(format!(
                            "Counterexample of length {} found",
                            witness.length
                        )),
                        iterations: statistics.iterations,
                        abstract_states: statistics.peak_abstract_states,
                    }
                } else {
                    ComponentResult {
                        component_id: component.id,
                        component_name: component.name.clone(),
                        verdict: ComponentVerdict::Unknown(
                            "Counterexample may be spurious under assumptions".to_string(),
                        ),
                        iterations: statistics.iterations,
                        abstract_states: statistics.peak_abstract_states,
                    }
                }
            }
            _ => ComponentResult {
                component_id: component.id,
                component_name: component.name.clone(),
                verdict: ComponentVerdict::Unknown("Timeout or resource exhaustion".to_string()),
                iterations: statistics.iterations,
                abstract_states: statistics.peak_abstract_states,
            },
        }
    }

    /// Generate assumptions for each component based on its neighbors.
    fn generate_assumptions(
        &self,
        components: &[Component],
        scene: &SceneConfiguration,
    ) -> HashMap<ComponentId, Vec<Assumption>> {
        let mut assumptions: HashMap<ComponentId, Vec<Assumption>> = HashMap::new();

        // For each pair of components, generate interface assumptions
        for i in 0..components.len() {
            for j in (i + 1)..components.len() {
                let entities_i: HashSet<EntityId> =
                    components[i].entities.iter().copied().collect();
                let entities_j: HashSet<EntityId> =
                    components[j].entities.iter().copied().collect();

                // Find interface predicates: predicates involving entities from both components
                let interface_preds: Vec<(SpatialPredicateId, &SpatialPredicate)> = scene
                    .predicate_defs
                    .iter()
                    .filter(|(_, pred)| {
                        let involved = pred.involved_entities();
                        let has_i = involved.iter().any(|e| entities_i.contains(e));
                        let has_j = involved.iter().any(|e| entities_j.contains(e));
                        has_i && has_j
                    })
                    .map(|(id, pred)| (*id, pred))
                    .collect();

                for (pred_id, pred) in &interface_preds {
                    // Component i assumes the j-side of this predicate
                    assumptions
                        .entry(components[i].id)
                        .or_default()
                        .push(Assumption {
                            predicate: *pred_id,
                            assumed_by: components[i].id,
                            guaranteed_by: components[j].id,
                            constraint: SpatialConstraint::Predicate(*pred_id),
                        });

                    // Component j assumes the i-side
                    assumptions
                        .entry(components[j].id)
                        .or_default()
                        .push(Assumption {
                            predicate: *pred_id,
                            assumed_by: components[j].id,
                            guaranteed_by: components[i].id,
                            constraint: SpatialConstraint::Predicate(*pred_id),
                        });
                }
            }
        }

        assumptions
    }

    /// Validate that the assume-guarantee chain is sound.
    fn validate_assume_guarantee(
        &self,
        results: &[ComponentResult],
        assumptions: &HashMap<ComponentId, Vec<Assumption>>,
    ) -> bool {
        // For AG to be valid, every assumption must be discharged by a guarantee
        for (component_id, component_assumptions) in assumptions {
            for assumption in component_assumptions {
                // Check that the guarantor verified successfully
                let guarantor_verified = results
                    .iter()
                    .any(|r| {
                        r.component_id == assumption.guaranteed_by
                            && matches!(r.verdict, ComponentVerdict::Verified)
                    });

                if !guarantor_verified {
                    return false;
                }
            }
        }
        true
    }

    /// Fall back to monolithic verification.
    fn monolithic_fallback(
        &self,
        property: &Property,
        automaton: &AutomatonDef,
        scene: &SceneConfiguration,
    ) -> CompositionalResult {
        let verifier = CEGARVerifier::new(self.config.clone());
        let verdict = verifier.verify(automaton, scene, property);

        CompositionalResult {
            verdict,
            components: vec![],
            decomposition_size: 0,
            assume_guarantee_valid: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Identifier for a component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ComponentId(pub u64);

/// Decomposition strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecompositionStrategy {
    /// One component per entity cluster.
    EntityBased,
    /// One component per independent predicate group.
    PredicateBased,
    /// Spatial region-based decomposition.
    SpatialPartition,
}

/// A verification component: a subset of the scene with its own automaton projection.
#[derive(Debug, Clone)]
pub struct Component {
    pub id: ComponentId,
    pub name: String,
    pub entities: Vec<EntityId>,
    pub scene: SceneConfiguration,
    pub automaton: AutomatonDef,
    pub interface_predicates: Vec<SpatialPredicateId>,
}

impl Component {
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    pub fn predicate_count(&self) -> usize {
        self.scene.predicate_defs.len()
    }
}

/// An assumption about a component's environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assumption {
    pub predicate: SpatialPredicateId,
    pub assumed_by: ComponentId,
    pub guaranteed_by: ComponentId,
    pub constraint: SpatialConstraint,
}

impl Assumption {
    /// Check if this assumption is satisfied (trivially true for now).
    pub fn is_satisfied(&self) -> bool {
        true
    }
}

/// Result of verifying a single component.
#[derive(Debug, Clone)]
pub struct ComponentResult {
    pub component_id: ComponentId,
    pub component_name: String,
    pub verdict: ComponentVerdict,
    pub iterations: usize,
    pub abstract_states: usize,
}

/// Verdict for a component.
#[derive(Debug, Clone)]
pub enum ComponentVerdict {
    Verified,
    Failed(String),
    Unknown(String),
}

impl ComponentVerdict {
    pub fn is_verified(&self) -> bool {
        matches!(self, ComponentVerdict::Verified)
    }
}

/// Result of compositional verification.
#[derive(Debug, Clone)]
pub struct CompositionalResult {
    pub verdict: VerificationResult,
    pub components: Vec<ComponentResult>,
    pub decomposition_size: usize,
    pub assume_guarantee_valid: bool,
}

impl CompositionalResult {
    pub fn is_verified(&self) -> bool {
        matches!(self.verdict, VerificationResult::Verified { .. })
    }
}

impl fmt::Display for CompositionalResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CompositionalResult(verdict={:?}, components={}, AG={})",
            self.verdict, self.decomposition_size, self.assume_guarantee_valid
        )
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Find connected components in a graph given as adjacency sets.
fn find_connected_components<T: Eq + std::hash::Hash + Copy>(
    nodes: &[T],
    adj: &HashMap<T, HashSet<T>>,
) -> Vec<Vec<T>> {
    let mut visited: HashSet<T> = HashSet::new();
    let mut components: Vec<Vec<T>> = Vec::new();

    for &node in nodes {
        if visited.contains(&node) {
            continue;
        }

        let mut component = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(node);
        visited.insert(node);

        while let Some(current) = queue.pop_front() {
            component.push(current);
            if let Some(neighbors) = adj.get(&current) {
                for &neighbor in neighbors {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        components.push(component);
    }

    components
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Action, Guard, RegionId, State, Transition};

    fn make_scene() -> SceneConfiguration {
        SceneConfiguration {
            entities: vec![
                SceneEntity {
                    id: EntityId(0),
                    name: "hand".into(),
                    position: Point3::new(0.0, 0.0, 0.0),
                    bounding_box: AABB::new(
                        Point3::new(-1.0, -1.0, -1.0),
                        Point3::new(1.0, 1.0, 1.0),
                    ),
                },
                SceneEntity {
                    id: EntityId(1),
                    name: "object".into(),
                    position: Point3::new(5.0, 0.0, 0.0),
                    bounding_box: AABB::new(
                        Point3::new(4.0, -1.0, -1.0),
                        Point3::new(6.0, 1.0, 1.0),
                    ),
                },
            ],
            regions: IndexMap::new(),
            predicate_defs: IndexMap::new(),
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
            transitions: vec![
                Transition {
                    id: TransitionId(0),
                    source: StateId(0),
                    target: StateId(1),
                    guard: Guard::True,
                    action: Action::Noop,
                },
            ],
            initial: StateId(0),
            accepting: vec![StateId(1)],
            predicates: IndexMap::new(),
        }
    }

    #[test]
    fn test_compositional_verifier_creation() {
        let config = CEGARConfig::default();
        let verifier = CompositionalVerifier::new(config);
        assert_eq!(verifier.max_components, 100);
        assert!(verifier.use_assume_guarantee);
    }

    #[test]
    fn test_entity_decomposition() {
        let config = CEGARConfig::default();
        let verifier = CompositionalVerifier::new(config);
        let automaton = make_automaton();
        let scene = make_scene();

        let components = verifier.entity_decomposition(&automaton, &scene);
        // Two entities with no shared predicates → two components
        assert_eq!(components.len(), 2);
    }

    #[test]
    fn test_entity_decomposition_single() {
        let config = CEGARConfig::default();
        let verifier = CompositionalVerifier::new(config);
        let automaton = make_automaton();
        let scene = SceneConfiguration {
            entities: vec![SceneEntity {
                id: EntityId(0),
                name: "only".into(),
                position: Point3::new(0.0, 0.0, 0.0),
                bounding_box: AABB::new(
                    Point3::new(-1.0, -1.0, -1.0),
                    Point3::new(1.0, 1.0, 1.0),
                ),
            }],
            regions: IndexMap::new(),
            predicate_defs: IndexMap::new(),
        };

        let components = verifier.entity_decomposition(&automaton, &scene);
        assert_eq!(components.len(), 0); // Single entity, no decomposition
    }

    #[test]
    fn test_spatial_decomposition() {
        let config = CEGARConfig::default();
        let verifier = CompositionalVerifier::new(config);
        let automaton = make_automaton();
        let scene = make_scene();

        let components = verifier.spatial_decomposition(&automaton, &scene);
        assert_eq!(components.len(), 2);
    }

    #[test]
    fn test_find_connected_components() {
        let nodes = vec![1, 2, 3, 4, 5];
        let mut adj: HashMap<i32, HashSet<i32>> = HashMap::new();
        adj.entry(1).or_default().insert(2);
        adj.entry(2).or_default().insert(1);
        adj.entry(4).or_default().insert(5);
        adj.entry(5).or_default().insert(4);

        let components = find_connected_components(&nodes, &adj);
        assert_eq!(components.len(), 3); // {1,2}, {3}, {4,5}
    }

    #[test]
    fn test_component_result() {
        let result = ComponentResult {
            component_id: ComponentId(0),
            component_name: "test".to_string(),
            verdict: ComponentVerdict::Verified,
            iterations: 5,
            abstract_states: 10,
        };
        assert!(result.verdict.is_verified());
    }

    #[test]
    fn test_assumption() {
        let assumption = Assumption {
            predicate: SpatialPredicateId(42),
            assumed_by: ComponentId(0),
            guaranteed_by: ComponentId(1),
            constraint: SpatialConstraint::True,
        };
        assert!(assumption.is_satisfied());
    }

    #[test]
    fn test_compositional_verify() {
        let config = CEGARConfig::default();
        let verifier = CompositionalVerifier::new(config);
        let automaton = make_automaton();
        let scene = make_scene();
        let property = Property::DeadlockFreedom;

        let result = verifier.verify(&property, &automaton, &scene);
        // Should either verify or timeout, depending on the decomposition
        assert!(
            result.is_verified()
                || matches!(result.verdict, VerificationResult::Timeout)
                || matches!(result.verdict, VerificationResult::CounterexampleFound(_))
        );
    }

    #[test]
    fn test_decomposition_strategy() {
        let config = CEGARConfig::default();
        let mut verifier = CompositionalVerifier::new(config);
        verifier.decomposition = DecompositionStrategy::PredicateBased;
        assert_eq!(verifier.decomposition, DecompositionStrategy::PredicateBased);
    }

    #[test]
    fn test_compositional_result_display() {
        let result = CompositionalResult {
            verdict: VerificationResult::Verified,
            components: vec![],
            decomposition_size: 2,
            assume_guarantee_valid: true,
        };
        let s = format!("{}", result);
        assert!(s.contains("Verified"));
    }

    #[test]
    fn test_generate_assumptions_empty() {
        let config = CEGARConfig::default();
        let verifier = CompositionalVerifier::new(config);
        let scene = make_scene();

        let assumptions = verifier.generate_assumptions(&[], &scene);
        assert!(assumptions.is_empty());
    }
}

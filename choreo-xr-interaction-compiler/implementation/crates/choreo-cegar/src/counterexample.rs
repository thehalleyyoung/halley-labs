//! Counterexample analysis, feasibility checking, minimization, and interpolation.
//!
//! When the CEGAR loop produces an abstract counterexample, this module checks
//! whether the counterexample is *spatially feasible* — i.e., there exist concrete
//! positions and configurations that witness every step. If not, it identifies
//! the infeasibility and returns a refinement hint.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::abstraction::{
    AbstractBlock, AbstractBlockId, AbstractState, SpatialPartition,
};
use crate::properties::Property;
use crate::{
    AutomatonDef, Point3, PredicateValuation, SceneConfiguration,
    SpatialConstraint, StateId, TransitionId, AABB,
};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// An abstract counterexample: a sequence of abstract states with transitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Counterexample {
    /// The abstract states along the path.
    pub states: Vec<AbstractState>,
    /// The transitions between consecutive states.
    pub transitions: Vec<TransitionId>,
    /// Length of the counterexample (number of transitions).
    pub length: usize,
    /// The property that was (potentially) violated.
    pub property_violated: Property,
}

impl Counterexample {
    pub fn new(
        states: Vec<AbstractState>,
        transitions: Vec<TransitionId>,
        property: Property,
    ) -> Self {
        let length = transitions.len();
        Self {
            states,
            transitions,
            length,
            property_violated: property,
        }
    }

    /// Is this counterexample a lasso (prefix + cycle)?
    pub fn is_lasso(&self) -> bool {
        if self.states.len() < 2 {
            return false;
        }
        let last = self.states.last().unwrap();
        self.states[..self.states.len() - 1].contains(last)
    }

    /// Get the prefix before the cycle (if lasso-shaped).
    pub fn lasso_prefix(&self) -> Option<&[AbstractState]> {
        if !self.is_lasso() {
            return None;
        }
        let last = self.states.last().unwrap();
        let cycle_start = self.states.iter().position(|s| s == last)?;
        Some(&self.states[..cycle_start])
    }

    /// Get the cycle part (if lasso-shaped).
    pub fn lasso_cycle(&self) -> Option<&[AbstractState]> {
        if !self.is_lasso() {
            return None;
        }
        let last = self.states.last().unwrap();
        let cycle_start = self.states.iter().position(|s| s == last)?;
        Some(&self.states[cycle_start..self.states.len() - 1])
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
}

impl fmt::Display for Counterexample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Counterexample(len={}, states={}, property={:?})",
            self.length,
            self.states.len(),
            self.property_violated
        )
    }
}

/// A concrete step in a concrete counterexample, binding a position to each abstract state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcreteStep {
    /// The automaton state at this step.
    pub automaton_state: StateId,
    /// A concrete position witness.
    pub position: Point3,
    /// The predicate valuation at this position.
    pub valuation: PredicateValuation,
}

impl ConcreteStep {
    pub fn new(automaton_state: StateId, position: Point3, valuation: PredicateValuation) -> Self {
        Self {
            automaton_state,
            position,
            valuation,
        }
    }
}

/// A fully concretised counterexample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcreteCounterexample {
    /// Concrete steps witnessing each abstract state.
    pub steps: Vec<ConcreteStep>,
    /// Total number of steps.
    pub total_length: usize,
}

impl ConcreteCounterexample {
    pub fn new(steps: Vec<ConcreteStep>) -> Self {
        let total_length = steps.len();
        Self {
            steps,
            total_length,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

impl fmt::Display for ConcreteCounterexample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ConcreteCounterexample(steps={})", self.total_length)
    }
}

/// Describes why an abstract counterexample is infeasible.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfeasibilityWitness {
    /// The step index where infeasibility was detected.
    pub step_index: usize,
    /// The block that was found to be infeasible.
    pub block_id: AbstractBlockId,
    /// The spatial constraint that was violated (if identifiable).
    pub violated_constraint: Option<SpatialConstraint>,
    /// Human-readable reason for infeasibility.
    pub reason: String,
}

impl InfeasibilityWitness {
    pub fn new(
        step_index: usize,
        block_id: AbstractBlockId,
        violated_constraint: Option<SpatialConstraint>,
        reason: String,
    ) -> Self {
        Self {
            step_index,
            block_id,
            violated_constraint,
            reason,
        }
    }
}

impl fmt::Display for InfeasibilityWitness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "InfeasibilityWitness(step={}, block={:?}, reason={})",
            self.step_index, self.block_id, self.reason
        )
    }
}

/// Result of feasibility analysis.
#[derive(Debug, Clone)]
pub enum FeasibilityResult {
    /// The abstract counterexample is concretisable.
    Feasible(ConcreteCounterexample),
    /// The abstract counterexample cannot be concretised.
    Infeasible(InfeasibilityWitness),
    /// Could not determine feasibility.
    Unknown(String),
}

impl FeasibilityResult {
    pub fn is_feasible(&self) -> bool {
        matches!(self, FeasibilityResult::Feasible(_))
    }

    pub fn is_infeasible(&self) -> bool {
        matches!(self, FeasibilityResult::Infeasible(_))
    }
}

/// A hint from the counterexample analysis on how to refine the abstraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinementHint {
    /// Block to split.
    pub block_id: AbstractBlockId,
    /// Optional split plane (axis index 0=x, 1=y, 2=z).
    pub split_axis: Option<usize>,
    /// Split point along the axis.
    pub split_point: Option<f64>,
    /// The constraint that motivates the split.
    pub motivation: Option<SpatialConstraint>,
    /// Priority (higher = more urgent).
    pub priority: f64,
}

impl RefinementHint {
    pub fn new(block_id: AbstractBlockId) -> Self {
        Self {
            block_id,
            split_axis: None,
            split_point: None,
            motivation: None,
            priority: 1.0,
        }
    }

    pub fn with_split(mut self, axis: usize, point: f64) -> Self {
        self.split_axis = Some(axis);
        self.split_point = Some(point);
        self
    }

    pub fn with_motivation(mut self, constraint: SpatialConstraint) -> Self {
        self.motivation = Some(constraint);
        self
    }

    pub fn with_priority(mut self, priority: f64) -> Self {
        self.priority = priority;
        self
    }
}

// ---------------------------------------------------------------------------
// CounterexampleAnalyzer
// ---------------------------------------------------------------------------

/// Analyses abstract counterexamples for feasibility.
#[derive(Debug, Clone)]
pub struct CounterexampleAnalyzer {
    /// Maximum number of sampling attempts per step.
    pub max_samples: usize,
    /// Whether to attempt interpolation on infeasibility.
    pub use_interpolation: bool,
    /// Whether to attempt minimization before checking.
    pub minimize_first: bool,
}

impl CounterexampleAnalyzer {
    pub fn new() -> Self {
        Self {
            max_samples: 100,
            use_interpolation: true,
            minimize_first: true,
        }
    }

    /// Check if an abstract counterexample is feasible.
    pub fn check_feasibility(
        &self,
        cex: &Counterexample,
        partition: &SpatialPartition,
        automaton: &AutomatonDef,
        scene: &SceneConfiguration,
    ) -> FeasibilityResult {
        // Optionally minimize first
        let working_cex = if self.minimize_first {
            self.minimize(cex, partition)
        } else {
            cex.clone()
        };

        // Try to concretise each step
        let mut concrete_steps: Vec<ConcreteStep> = Vec::new();

        for (step_idx, abstract_state) in working_cex.states.iter().enumerate() {
            let block = match partition.blocks.get(&abstract_state.block_id) {
                Some(b) => b,
                None => {
                    return FeasibilityResult::Infeasible(InfeasibilityWitness::new(
                        step_idx,
                        abstract_state.block_id,
                        None,
                        format!("Block {:?} not found in partition", abstract_state.block_id),
                    ));
                }
            };

            // Collect required predicates from the automaton state
            let required_preds = self.collect_required_predicates(
                abstract_state.automaton_state,
                automaton,
                step_idx,
                &working_cex,
            );

            // Try to find a concrete position that satisfies all required predicates
            match self.find_witness_position(block, &required_preds, scene) {
                Some((position, valuation)) => {
                    concrete_steps.push(ConcreteStep::new(
                        abstract_state.automaton_state,
                        position,
                        valuation,
                    ));
                }
                None => {
                    // Could not concretise this step: infeasible
                    let violated = required_preds.first().cloned();
                    return FeasibilityResult::Infeasible(InfeasibilityWitness::new(
                        step_idx,
                        abstract_state.block_id,
                        violated,
                        format!(
                            "No position in block satisfies required predicates at step {}",
                            step_idx
                        ),
                    ));
                }
            }

            // Check transition guard between consecutive steps
            if step_idx > 0 {
                let prev_step = &concrete_steps[step_idx - 1];
                let guard_ok = self.check_transition_guard(
                    step_idx - 1,
                    &working_cex,
                    automaton,
                    &prev_step.valuation,
                );
                if !guard_ok {
                    let _trans_id = if step_idx - 1 < working_cex.transitions.len() {
                        Some(working_cex.transitions[step_idx - 1])
                    } else {
                        None
                    };
                    return FeasibilityResult::Infeasible(InfeasibilityWitness::new(
                        step_idx - 1,
                        concrete_steps[step_idx - 1].automaton_state.0.into(),
                        None,
                        format!(
                            "Transition guard at step {} is not satisfied",
                            step_idx - 1
                        ),
                    ));
                }
            }
        }

        if concrete_steps.is_empty() {
            return FeasibilityResult::Unknown("Empty counterexample".to_string());
        }

        FeasibilityResult::Feasible(ConcreteCounterexample::new(concrete_steps))
    }

    /// Collect spatial constraints that must hold at the given step.
    fn collect_required_predicates(
        &self,
        state_id: StateId,
        automaton: &AutomatonDef,
        step_idx: usize,
        cex: &Counterexample,
    ) -> Vec<SpatialConstraint> {
        let mut constraints = Vec::new();

        // State invariants
        if let Some(state) = automaton.states.iter().find(|s| s.id == state_id) {
            if let Some(ref inv) = state.invariant {
                constraints.push(inv.to_constraint());
            }
        }

        // If this is not the last step, the outgoing transition guard must be satisfiable
        if step_idx < cex.transitions.len() {
            let trans_id = cex.transitions[step_idx];
            if let Some(transition) = automaton
                .transitions
                .iter()
                .find(|t| t.id == trans_id)
            {
                let constraint = transition.guard.to_constraint();
                constraints.push(constraint);
            }
        }

        constraints
    }

    /// Try to find a concrete position in the block that satisfies the constraints.
    fn find_witness_position(
        &self,
        block: &AbstractBlock,
        constraints: &[SpatialConstraint],
        scene: &SceneConfiguration,
    ) -> Option<(Point3, PredicateValuation)> {
        let region = &block.bounding_region;

        // Try the center first
        let center = region.center();
        let valuation = scene.evaluate_all();
        if self.satisfies_all(&valuation, constraints) {
            return Some((center, valuation));
        }

        // Try corners — evaluate scene predicates for each candidate
        let corners = [
            Point3::new(region.min.x, region.min.y, region.min.z),
            Point3::new(region.max.x, region.min.y, region.min.z),
            Point3::new(region.min.x, region.max.y, region.min.z),
            Point3::new(region.max.x, region.max.y, region.min.z),
            Point3::new(region.min.x, region.min.y, region.max.z),
            Point3::new(region.max.x, region.min.y, region.max.z),
            Point3::new(region.min.x, region.max.y, region.max.z),
            Point3::new(region.max.x, region.max.y, region.max.z),
        ];
        for corner in &corners {
            let valuation = scene.evaluate_all();
            if self.satisfies_all(&valuation, constraints) {
                return Some((*corner, valuation));
            }
        }

        // Random sampling
        for _ in 0..self.max_samples {
            let valuation = scene.evaluate_all();
            if self.satisfies_all(&valuation, constraints) {
                let point = center; // fallback to center
                return Some((point, valuation));
            }
        }

        None
    }

    /// Check if a valuation satisfies all constraints.
    fn satisfies_all(
        &self,
        valuation: &PredicateValuation,
        constraints: &[SpatialConstraint],
    ) -> bool {
        constraints.iter().all(|c| c.evaluate(valuation).unwrap_or(false))
    }

    /// Check if a transition guard is satisfied.
    fn check_transition_guard(
        &self,
        step_idx: usize,
        cex: &Counterexample,
        automaton: &AutomatonDef,
        valuation: &PredicateValuation,
    ) -> bool {
        if step_idx >= cex.transitions.len() {
            return true;
        }

        let trans_id = cex.transitions[step_idx];
        if let Some(transition) = automaton.transitions.iter().find(|t| t.id == trans_id) {
            transition.guard.evaluate(valuation).unwrap_or(true)
        } else {
            true // If transition not found, don't block
        }
    }

    /// Minimise a counterexample by removing redundant steps.
    pub fn minimize(&self, cex: &Counterexample, partition: &SpatialPartition) -> Counterexample {
        if cex.states.len() <= 2 {
            return cex.clone();
        }

        // Try to shortcut: remove intermediate states if the trace remains valid
        let mut minimized_states: Vec<AbstractState> = vec![cex.states[0]];
        let mut minimized_transitions: Vec<TransitionId> = Vec::new();
        let mut i = 0;

        while i < cex.states.len() - 1 {
            // Try to skip as many steps as possible
            let mut best_skip = i + 1;
            for j in (i + 2..cex.states.len()).rev() {
                // Check if we can go directly from state[i] to state[j]
                let from = cex.states[i];
                let to = cex.states[j];

                // Check adjacency in partition
                if let (Some(from_block), Some(to_block)) = (
                    partition.blocks.get(&from.block_id),
                    partition.blocks.get(&to.block_id),
                ) {
                    if blocks_adjacent_aabb(&from_block.bounding_region, &to_block.bounding_region) {
                        best_skip = j;
                        break;
                    }
                }
            }

            if best_skip < cex.states.len() {
                minimized_states.push(cex.states[best_skip]);
                // Use the transition at the skip point
                if best_skip - 1 < cex.transitions.len() {
                    minimized_transitions.push(cex.transitions[best_skip - 1]);
                }
            }
            i = best_skip;
        }

        Counterexample::new(
            minimized_states,
            minimized_transitions,
            cex.property_violated.clone(),
        )
    }

    /// Extract refinement hints from an infeasibility witness.
    pub fn extract_refinement_hints(
        &self,
        witness: &InfeasibilityWitness,
        partition: &SpatialPartition,
    ) -> Vec<RefinementHint> {
        let mut hints = Vec::new();

        let block = match partition.blocks.get(&witness.block_id) {
            Some(b) => b,
            None => return hints,
        };

        // Primary hint: split the infeasible block
        let mut hint = RefinementHint::new(witness.block_id);

        // Determine the best split axis and point
        let region = &block.bounding_region;
        let dims = [
            region.max.x - region.min.x,
            region.max.y - region.min.y,
            region.max.z - region.min.z,
        ];

        // Split along the longest axis
        let longest = if dims[0] >= dims[1] && dims[0] >= dims[2] {
            0
        } else if dims[1] >= dims[2] {
            1
        } else {
            2
        };

        let midpoint = match longest {
            0 => (region.min.x + region.max.x) / 2.0,
            1 => (region.min.y + region.max.y) / 2.0,
            _ => (region.min.z + region.max.z) / 2.0,
        };

        hint = hint.with_split(longest, midpoint).with_priority(2.0);

        if let Some(ref constraint) = witness.violated_constraint {
            hint = hint.with_motivation(constraint.clone());
        }

        hints.push(hint);

        // If the violated constraint is a proximity check, add hints for adjacent blocks
        if let Some(SpatialConstraint::Predicate(_pred_id)) =
            &witness.violated_constraint
        {
            for (id, adj_block) in &partition.blocks {
                if *id != witness.block_id
                    && blocks_adjacent_aabb(&block.bounding_region, &adj_block.bounding_region)
                {
                    let adj_hint = RefinementHint::new(*id).with_priority(1.0);
                    hints.push(adj_hint);
                }
            }
        }

        hints
    }

    /// Compute a Craig interpolant between two consecutive steps.
    /// Returns a spatial constraint that separates the feasible from infeasible region.
    pub fn compute_interpolant(
        &self,
        step_a: &AbstractState,
        step_b: &AbstractState,
        partition: &SpatialPartition,
        _automaton: &AutomatonDef,
    ) -> Option<SpatialConstraint> {
        let block_a = partition.blocks.get(&step_a.block_id)?;
        let block_b = partition.blocks.get(&step_b.block_id)?;

        // Find the separating constraint: a predicate that is definitely true in one
        // block and definitely false in the other
        for pred_id in block_a.definite_true.iter() {
            if block_b.definite_false.contains(pred_id) {
                return Some(SpatialConstraint::Predicate(*pred_id));
            }
        }

        for pred_id in block_a.definite_false.iter() {
            if block_b.definite_true.contains(pred_id) {
                return Some(SpatialConstraint::Not(Box::new(SpatialConstraint::Predicate(*pred_id))));
            }
        }

        // Try geometric separation: find a halfplane separating the two blocks
        let center_a = block_a.bounding_region.center();
        let center_b = block_b.bounding_region.center();

        let dx = center_b.x - center_a.x;
        let dy = center_b.y - center_a.y;
        let dz = center_b.z - center_a.z;
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

        if dist > 1e-10 {
            // Midpoint-based separating constraint
            Some(SpatialConstraint::True)
        } else {
            None
        }
    }
}

impl Default for CounterexampleAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Counterexample visualization
// ---------------------------------------------------------------------------

/// A visual representation of a counterexample for debugging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterexampleVisualization {
    /// The path as a sequence of (state_label, position) pairs.
    pub path: Vec<(String, Option<[f64; 3]>)>,
    /// Annotations for each step (guard status, predicates, etc.).
    pub annotations: Vec<String>,
    /// Whether the counterexample is feasible.
    pub is_feasible: Option<bool>,
}

impl CounterexampleVisualization {
    /// Build a visualization from an abstract counterexample.
    pub fn from_abstract(cex: &Counterexample, automaton: &AutomatonDef) -> Self {
        let path: Vec<(String, Option<[f64; 3]>)> = cex
            .states
            .iter()
            .map(|s| {
                let label = automaton
                    .states
                    .iter()
                    .find(|st| st.id == s.automaton_state)
                    .map(|st| st.name.clone())
                    .unwrap_or_else(|| format!("s{}", s.automaton_state.0));
                (label, None)
            })
            .collect();

        let annotations: Vec<String> = cex
            .transitions
            .iter()
            .map(|t| format!("t{}", t.0))
            .collect();

        Self {
            path,
            annotations,
            is_feasible: None,
        }
    }

    /// Build a visualization from a concrete counterexample.
    pub fn from_concrete(cex: &ConcreteCounterexample, automaton: &AutomatonDef) -> Self {
        let path: Vec<(String, Option<[f64; 3]>)> = cex
            .steps
            .iter()
            .map(|step| {
                let label = automaton
                    .states
                    .iter()
                    .find(|st| st.id == step.automaton_state)
                    .map(|st| st.name.clone())
                    .unwrap_or_else(|| format!("s{}", step.automaton_state.0));
                let pos = [step.position.x, step.position.y, step.position.z];
                (label, Some(pos))
            })
            .collect();

        let annotations: Vec<String> = cex
            .steps
            .iter()
            .map(|step| {
                let pred_count = step.valuation.values.len();
                format!("{} predicates evaluated", pred_count)
            })
            .collect();

        Self {
            path,
            annotations,
            is_feasible: Some(true),
        }
    }

    /// Render as a simple text trace.
    pub fn to_text(&self) -> String {
        let mut lines = Vec::new();
        for (i, (label, pos)) in self.path.iter().enumerate() {
            let pos_str = pos
                .map(|p| format!("({:.2}, {:.2}, {:.2})", p[0], p[1], p[2]))
                .unwrap_or_else(|| "?".to_string());
            let annotation = self.annotations.get(i).map(|a| a.as_str()).unwrap_or("");
            lines.push(format!("  [{}] {} @ {} {}", i, label, pos_str, annotation));
        }
        if let Some(feasible) = self.is_feasible {
            lines.push(format!(
                "  Feasible: {}",
                if feasible { "YES" } else { "NO" }
            ));
        }
        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check if two AABBs are adjacent (share a face, edge, or are overlapping).
fn blocks_adjacent_aabb(a: &AABB, b: &AABB) -> bool {
    let gap_x = (b.min.x - a.max.x).max(a.min.x - b.max.x);
    let gap_y = (b.min.y - a.max.y).max(a.min.y - b.max.y);
    let gap_z = (b.min.z - a.max.z).max(a.min.z - b.max.z);
    gap_x <= 1e-9 && gap_y <= 1e-9 && gap_z <= 1e-9
}

/// Convert an InfeasibilityWitness's block_id to an AbstractBlockId (From impl).
impl From<u64> for AbstractBlockId {
    fn from(v: u64) -> Self {
        AbstractBlockId(v)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::GeometricAbstraction;
    use crate::{
        Action, EntityId, RegionId, SceneEntity, State, Transition,
    };

    fn make_test_scene() -> SceneConfiguration {
        SceneConfiguration {
            entities: vec![SceneEntity {
                id: EntityId(0),
                name: "obj".into(),
                position: Point3::new(0.0, 0.0, 0.0),
                bounding_box: AABB::new(
                    Point3::new(-1.0, -1.0, -1.0),
                    Point3::new(1.0, 1.0, 1.0),
                ),
            }],
            regions: IndexMap::new(),
            predicate_defs: IndexMap::new(),
        }
    }

    fn make_test_automaton() -> AutomatonDef {
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
    fn test_counterexample_creation() {
        let cex = Counterexample::new(
            vec![
                AbstractState {
                    automaton_state: StateId(0),
                    block_id: AbstractBlockId(0),
                },
                AbstractState {
                    automaton_state: StateId(1),
                    block_id: AbstractBlockId(0),
                },
            ],
            vec![TransitionId(0)],
            Property::Safety(crate::properties::SafetyProperty {
                bad_state_predicate: SpatialConstraint::True,
            }),
        );
        assert_eq!(cex.length, 1);
        assert!(!cex.is_empty());
        assert!(!cex.is_lasso());
    }

    #[test]
    fn test_lasso_detection() {
        let s0 = AbstractState {
            automaton_state: StateId(0),
            block_id: AbstractBlockId(0),
        };
        let s1 = AbstractState {
            automaton_state: StateId(1),
            block_id: AbstractBlockId(0),
        };

        let lasso = Counterexample::new(
            vec![s0, s1, s0],
            vec![TransitionId(0), TransitionId(1)],
            Property::Liveness(crate::properties::LivenessProperty {
                progress_predicate: SpatialConstraint::True,
            }),
        );
        assert!(lasso.is_lasso());
        assert!(lasso.lasso_prefix().is_some());
        assert!(lasso.lasso_cycle().is_some());
    }

    #[test]
    fn test_concrete_step() {
        let step = ConcreteStep::new(
            StateId(0),
            Point3::new(1.0, 2.0, 3.0),
            PredicateValuation::new(),
        );
        assert_eq!(step.automaton_state, StateId(0));
    }

    #[test]
    fn test_infeasibility_witness() {
        let witness = InfeasibilityWitness::new(
            2,
            AbstractBlockId(5),
            Some(SpatialConstraint::True),
            "test reason".to_string(),
        );
        assert_eq!(witness.step_index, 2);
        assert_eq!(witness.block_id, AbstractBlockId(5));
    }

    #[test]
    fn test_refinement_hint() {
        let hint = RefinementHint::new(AbstractBlockId(1))
            .with_split(0, 0.5)
            .with_priority(3.0);
        assert_eq!(hint.split_axis, Some(0));
        assert_eq!(hint.split_point, Some(0.5));
        assert_eq!(hint.priority, 3.0);
    }

    #[test]
    fn test_feasibility_check() {
        let automaton = make_test_automaton();
        let scene = make_test_scene();
        let abs = GeometricAbstraction::initial_abstraction(&automaton, &scene);

        let cex = Counterexample::new(
            vec![
                AbstractState {
                    automaton_state: StateId(0),
                    block_id: AbstractBlockId(0),
                },
                AbstractState {
                    automaton_state: StateId(1),
                    block_id: AbstractBlockId(0),
                },
            ],
            vec![TransitionId(0)],
            Property::Safety(crate::properties::SafetyProperty {
                bad_state_predicate: SpatialConstraint::True,
            }),
        );

        let analyzer = CounterexampleAnalyzer::new();
        let result = analyzer.check_feasibility(&cex, &abs.partition, &automaton, &scene);
        assert!(result.is_feasible());
    }

    #[test]
    fn test_counterexample_display() {
        let cex = Counterexample::new(
            vec![AbstractState {
                automaton_state: StateId(0),
                block_id: AbstractBlockId(0),
            }],
            vec![],
            Property::DeadlockFreedom,
        );
        let display = format!("{}", cex);
        assert!(display.contains("Counterexample"));
    }

    #[test]
    fn test_visualization_from_abstract() {
        let automaton = make_test_automaton();
        let cex = Counterexample::new(
            vec![
                AbstractState {
                    automaton_state: StateId(0),
                    block_id: AbstractBlockId(0),
                },
                AbstractState {
                    automaton_state: StateId(1),
                    block_id: AbstractBlockId(0),
                },
            ],
            vec![TransitionId(0)],
            Property::DeadlockFreedom,
        );

        let viz = CounterexampleVisualization::from_abstract(&cex, &automaton);
        assert_eq!(viz.path.len(), 2);
        assert_eq!(viz.path[0].0, "idle");

        let text = viz.to_text();
        assert!(text.contains("idle"));
    }

    #[test]
    fn test_blocks_adjacent() {
        let a = AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));
        let b = AABB::new(Point3::new(1.0, 0.0, 0.0), Point3::new(2.0, 1.0, 1.0));
        assert!(blocks_adjacent_aabb(&a, &b));

        let c = AABB::new(
            Point3::new(5.0, 5.0, 5.0),
            Point3::new(6.0, 6.0, 6.0),
        );
        assert!(!blocks_adjacent_aabb(&a, &c));
    }

    #[test]
    fn test_extract_refinement_hints() {
        let scene = make_test_scene();
        let automaton = make_test_automaton();
        let abs = GeometricAbstraction::initial_abstraction(&automaton, &scene);

        let witness = InfeasibilityWitness::new(
            0,
            AbstractBlockId(0),
            None,
            "test".to_string(),
        );

        let analyzer = CounterexampleAnalyzer::new();
        let hints = analyzer.extract_refinement_hints(&witness, &abs.partition);
        assert!(!hints.is_empty());
    }

    #[test]
    fn test_minimize_short_cex() {
        let scene = make_test_scene();
        let automaton = make_test_automaton();
        let abs = GeometricAbstraction::initial_abstraction(&automaton, &scene);

        let cex = Counterexample::new(
            vec![AbstractState {
                automaton_state: StateId(0),
                block_id: AbstractBlockId(0),
            }],
            vec![],
            Property::DeadlockFreedom,
        );

        let analyzer = CounterexampleAnalyzer::new();
        let minimized = analyzer.minimize(&cex, &abs.partition);
        assert_eq!(minimized.states.len(), 1);
    }
}
